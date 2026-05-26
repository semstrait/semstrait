//! Ad-hoc multi-entity join synthesis (Phase 4).
//!
//! When a FROM-less query spans multiple entities, this module builds a
//! cross-entity join plan. Each entity is planned independently via its
//! kind planner (Dataset, Grainset, Unionset, Joinset), then the resulting
//! PlanFragments are joined using relationship definitions.
//!
//! Plan shape:
//! ```text
//! Project(user's requested semantic names)
//!   └── Join(condition, type)
//!         ├── left: [Entity A PlanFragment] or nested Join
//!         └── right: [Entity B PlanFragment]
//! ```

use std::collections::HashSet;

use semstrait_ir::{Expr, Field, LogicalPlan, PlanBuilder, PlanNode, Schema};
use semstrait_manifest::{CompiledManifest, CompiledRelationship};

use crate::additivity::AdditivityResolver;
use crate::entity_resolver::{MatchResult, MatchedEntity};
use crate::error::PlannerError;
use crate::data_kind::PlannerContext;
use crate::data_kind::joinset::map_join_type;
use crate::inline_filter::{lower_inline_filter, InlineFilterError};
use crate::planner::{
    SemanticPlanner, apply_limit, apply_order_by, inject_user_filters,
};
use crate::request::ResolvedQueryRequest;
use crate::validator::ConstraintValidator;

/// A step in the cross-entity join traversal.
struct JoinStep {
    #[allow(dead_code)] // Used in error messages and debugging
    entity_name: String,
    relationship_idx: usize,
    /// Whether this step joins via the reverse direction (to → from).
    reversed: bool,
}

/// Augmented request for a single entity, with join key columns added.
struct EntityRequest {
    #[allow(dead_code)]
    entity_name: String,
    request: ResolvedQueryRequest,
    /// Join key columns not in the user's original select (to be dropped in final projection).
    #[allow(dead_code)]
    extra_join_keys: Vec<String>,
}

/// Build a complete LogicalPlan for a multi-entity ad-hoc query.
///
/// Each entity is planned independently via `resolve_entity()` — the kind planner
/// handles internal complexity (Grainset UNION+re-agg, Joinset internal joins, etc.).
/// Then PlanFragments are joined using relationship definitions.
pub fn build_ad_hoc_join_plan(
    planner: &SemanticPlanner,
    match_result: &MatchResult,
    original_request: &ResolvedQueryRequest,
    manifest: &CompiledManifest,
) -> Result<LogicalPlan, PlannerError> {
    let anchor = &match_result.entities[0];
    let pb = planner.plan_builder();

    // 1. Compute join order from anchor through relationships.
    let join_order = compute_join_order(
        &anchor.entity_name,
        &match_result.entities[1..],
        &match_result.join_path,
        manifest,
    )?;

    // 2. Build augmented per-entity requests with join key columns.
    let entity_requests = build_entity_requests(
        match_result,
        &join_order,
        manifest,
        original_request,
    )?;

    // 3. Plan anchor entity.
    let anchor_req = &entity_requests[0];
    let ctx = PlannerContext {
        manifest,
        catalog: None,
        session: &original_request.session_variables,
        plan_builder: pb,
    };

    // Validate constraints per entity.
    ConstraintValidator::check(&anchor_req.request, manifest)?;

    let (mut anchor_fragment, anchor_measures) =
        planner.resolve_entity(&anchor_req.request, manifest, &ctx)?;

    // Per-entity additivity resolution.
    for measure_name in &anchor_req.request.measures {
        if let Some(measure) = anchor_measures.get(measure_name) {
            anchor_fragment =
                AdditivityResolver::resolve(anchor_fragment, measure, &anchor_req.request)?;
        }
    }
    let mut current_root = anchor_fragment.root;

    // 4. Join each subsequent entity.
    for (step_idx, step) in join_order.iter().enumerate() {
        // entity_requests[0] is anchor, steps are 1-indexed.
        let step_req = &entity_requests[step_idx + 1];

        ConstraintValidator::check(&step_req.request, manifest)?;

        let (mut step_fragment, step_measures) =
            planner.resolve_entity(&step_req.request, manifest, &ctx)?;

        for measure_name in &step_req.request.measures {
            if let Some(measure) = step_measures.get(measure_name) {
                step_fragment =
                    AdditivityResolver::resolve(step_fragment, measure, &step_req.request)?;
            }
        }
        let step_root = step_fragment.root;

        // Build join condition from the relationship.
        let rel = &manifest.relationships[step.relationship_idx];
        let condition = build_semantic_join_condition(rel, step.reversed);
        let join_type = map_join_type(&rel.join_type);

        // Build join output schema (left fields + right fields).
        let mut join_fields: Vec<Field> = current_root.meta().output_schema.fields.clone();
        join_fields.extend(step_root.meta().output_schema.fields.iter().cloned());
        let join_schema = Schema::new(join_fields);

        current_root = pb.build_join(join_schema, current_root, step_root, join_type, condition);
    }

    // 5. Final projection: only user's originally requested fields.
    current_root = build_final_projection(current_root, original_request, pb);

    // 6. Post-processing: user filters, ORDER BY, LIMIT.
    current_root = inject_user_filters(current_root, original_request, pb)?;
    current_root = apply_order_by(current_root, original_request, pb)?;
    current_root = apply_limit(current_root, original_request, pb)?;

    // Build LogicalPlan.
    let output_names: Vec<String> = original_request
        .dimensions
        .iter()
        .chain(original_request.measures.iter())
        .cloned()
        .collect();

    let plan = LogicalPlan::new(current_root, output_names);

    // Optimizer pass.
    planner.optimize(plan)
}

/// Determine join order: BFS-like traversal from anchor through relationships.
///
/// For each non-anchor entity in `other_entities`, finds a connecting relationship
/// from `join_path_indices` that links it to the anchor or an already-visited entity.
fn compute_join_order(
    anchor_name: &str,
    other_entities: &[MatchedEntity],
    join_path_indices: &[usize],
    manifest: &CompiledManifest,
) -> Result<Vec<JoinStep>, PlannerError> {
    let mut visited: HashSet<&str> = HashSet::new();
    visited.insert(anchor_name);

    let mut steps: Vec<JoinStep> = Vec::new();
    let mut remaining: Vec<&MatchedEntity> = other_entities.iter().collect();

    // Iteratively find the next entity that connects to an already-visited entity.
    while !remaining.is_empty() {
        let mut found_idx = None;

        for (i, entity) in remaining.iter().enumerate() {
            // Check each relationship in the join path for a connection.
            for &rel_idx in join_path_indices {
                let rel = &manifest.relationships[rel_idx];

                // Forward: visited entity → this entity.
                if visited.contains(rel.from.as_str()) && rel.to == entity.entity_name {
                    found_idx = Some((i, rel_idx, false));
                    break;
                }
                // Reverse: this entity → visited entity.
                if visited.contains(rel.to.as_str()) && rel.from == entity.entity_name {
                    found_idx = Some((i, rel_idx, true));
                    break;
                }
            }
            if found_idx.is_some() {
                break;
            }
        }

        match found_idx {
            Some((idx, rel_idx, reversed)) => {
                let entity = remaining.remove(idx);
                visited.insert(&entity.entity_name);
                steps.push(JoinStep {
                    entity_name: entity.entity_name.clone(),
                    relationship_idx: rel_idx,
                    reversed,
                });
            }
            None => {
                let unreachable: Vec<&str> = remaining
                    .iter()
                    .map(|e| e.entity_name.as_str())
                    .collect();
                return Err(PlannerError::Internal(format!(
                    "ad-hoc join: entities [{}] unreachable from anchor '{}'",
                    unreachable.join(", "),
                    anchor_name,
                )));
            }
        }
    }

    Ok(steps)
}

/// Build augmented per-entity requests with join key columns added.
///
/// For each entity, starts with the fields assigned by entity_resolver,
/// then adds join key columns from relationships that reference this entity.
/// Tracks extra join keys for removal in final projection.
///
/// Inline raw filters carried as `original_request.pending_inline_filters`
/// are attributed per-field to their owning `MatchedEntity` via anchor-first
/// tiebreak (the leftmost entity in `match_result.entities` whose covered
/// fields include the filter's `field`). Each attributed filter is lowered
/// against that entity's `CompiledInterface` and placed on that entity's
/// `request.inline_filters`, riding the same scan-layer engine as named
/// DataKind filters per `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
fn build_entity_requests(
    match_result: &MatchResult,
    join_order: &[JoinStep],
    manifest: &CompiledManifest,
    original_request: &ResolvedQueryRequest,
) -> Result<Vec<EntityRequest>, PlannerError> {
    // Collect all relationship indices used in the join.
    let rel_indices: HashSet<usize> = join_order
        .iter()
        .map(|s| s.relationship_idx)
        .collect();

    // Original user-requested fields for tracking extras.
    let original_fields: HashSet<&str> = original_request
        .dimensions
        .iter()
        .chain(original_request.measures.iter())
        .map(|s| s.as_str())
        .collect();

    // Attribute each pending inline filter to its owning MatchedEntity.
    // Anchor-first tiebreak: pick the leftmost entity in
    // `match_result.entities` (sorted by coverage best-first) whose
    // `CompiledInterface` lists the field as a Dimension / Key / Measure /
    // Metric. We consult the interface (not `MatchedEntity.covered_*`)
    // because covered_* only mirrors the user's select coverage, while
    // raw_filter fields are independent of the select. Abort with
    // FieldOnNoEntity if no covering entity exists.
    let pending_assignments: Vec<usize> = original_request
        .pending_inline_filters
        .iter()
        .map(|pf| {
            match_result
                .entities
                .iter()
                .position(|me| {
                    manifest
                        .resolve(&me.entity_name)
                        .map(|dk| interface_has_field(dk.interface(), &pf.field))
                        .unwrap_or(false)
                })
                .ok_or_else(|| {
                    PlannerError::from(InlineFilterError::FieldOnNoEntity {
                        field: pf.field.clone(),
                        candidates: match_result
                            .entities
                            .iter()
                            .map(|e| e.entity_name.clone())
                            .collect(),
                    })
                })
        })
        .collect::<Result<Vec<_>, PlannerError>>()?;

    let mut requests = Vec::new();

    for (entity_idx, matched) in match_result.entities.iter().enumerate() {
        let entity_name = &matched.entity_name;

        // Start with this entity's covered fields — reclassify into dims and measures.
        let mut dims: Vec<String> = matched
            .group_by_fields()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let measures: Vec<String> = matched
            .aggregate_fields()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let mut extra_join_keys: Vec<String> = Vec::new();
        let mut dim_set: HashSet<String> = dims.iter().cloned().collect();
        let measure_set: HashSet<String> = measures.iter().cloned().collect();

        // Find relationships that reference this entity and add join key columns.
        for &rel_idx in &rel_indices {
            let rel = &manifest.relationships[rel_idx];

            if rel.from == *entity_name {
                // This entity is the "from" side — need col_pair.from columns.
                for col_pair in &rel.columns {
                    if !dim_set.contains(&col_pair.from)
                        && !measure_set.contains(&col_pair.from)
                    {
                        dims.push(col_pair.from.clone());
                        dim_set.insert(col_pair.from.clone());
                        if !original_fields.contains(col_pair.from.as_str()) {
                            extra_join_keys.push(col_pair.from.clone());
                        }
                    }
                }
            }
            if rel.to == *entity_name {
                // This entity is the "to" side — need col_pair.to columns.
                for col_pair in &rel.columns {
                    if !dim_set.contains(&col_pair.to)
                        && !measure_set.contains(&col_pair.to)
                    {
                        dims.push(col_pair.to.clone());
                        dim_set.insert(col_pair.to.clone());
                        if !original_fields.contains(col_pair.to.as_str()) {
                            extra_join_keys.push(col_pair.to.clone());
                        }
                    }
                }
            }
        }

        // Lower the pending inline filters assigned to this entity against
        // the entity's interface. Synthetic `__inline_filter_<i>` names use
        // the original per-request index so they remain globally unique
        // across the per-entity scans. We resolve the interface lazily —
        // skipping the lookup entirely when no filter lands on this entity
        // keeps the path test-friendly for manifests built without entities.
        let mut lowered_inline_filters = Vec::new();
        let has_pending_for_entity = pending_assignments
            .iter()
            .any(|&assigned| assigned == entity_idx);
        if has_pending_for_entity {
            let iface = manifest
                .resolve(entity_name)
                .ok_or_else(|| PlannerError::KindNotFound(entity_name.clone()))?
                .interface();
            for (i, pending) in original_request.pending_inline_filters.iter().enumerate() {
                if pending_assignments[i] == entity_idx {
                    let cf = lower_inline_filter(pending, iface, entity_name, i)?;
                    lowered_inline_filters.push(cf);
                }
            }
        }

        let mut request = original_request.clone();
        request.entity_name = entity_name.clone();
        request.dimensions = dims;
        request.measures = measures;
        // Clear filters — user filters applied after join in post-processing.
        request.filters = Vec::new();
        request.order_by = Vec::new();
        request.limit = None;
        // Replace any pre-existing inline_filters with this entity's lowered
        // subset. Multi-entity ad-hoc has no source of pre-lowered
        // inline_filters (the parser always emits empty when `from` is
        // None), but we overwrite defensively to guarantee per-entity
        // attribution holds.
        request.inline_filters = lowered_inline_filters;
        request.pending_inline_filters = Vec::new();

        requests.push(EntityRequest {
            entity_name: entity_name.clone(),
            request,
            extra_join_keys,
        });
    }

    Ok(requests)
}

/// Check whether `iface` lists `field` as a Dimension / Key / Measure /
/// Metric. Used by pending-inline-filter attribution in multi-entity
/// ad-hoc mode. Matches `inline_filter::lookup_field_type` shape but
/// returns a bool — we only need existence, not the type.
fn interface_has_field(iface: &semstrait_manifest::CompiledInterface, field: &str) -> bool {
    if iface.dimensions.contains_key(field)
        || iface.measures.contains_key(field)
        || iface.metrics.contains_key(field)
    {
        return true;
    }
    if let Some(keys) = &iface.keys {
        if keys.all_column_names().iter().any(|s| s == field) {
            return true;
        }
    }
    false
}

/// Build a join condition using semantic column names from a relationship.
///
/// PlanFragments output semantic names after projection, so join conditions
/// use semantic names directly from JoinColumnPair.
fn build_semantic_join_condition(rel: &CompiledRelationship, reversed: bool) -> Expr {
    let conditions: Vec<Expr> = rel
        .columns
        .iter()
        .map(|col_pair| {
            if reversed {
                // Reversed: the "to" entity is on the left (already joined), "from" on right.
                Expr::eq(
                    Expr::column(col_pair.to.clone()),
                    Expr::column(col_pair.from.clone()),
                )
            } else {
                Expr::eq(
                    Expr::column(col_pair.from.clone()),
                    Expr::column(col_pair.to.clone()),
                )
            }
        })
        .collect();

    conditions
        .into_iter()
        .reduce(Expr::and)
        .unwrap_or_else(|| Expr::boolean(true))
}

/// Project to only the user's originally requested fields.
///
/// Drops join key columns that were augmented for join conditions
/// but weren't in the user's original select.
fn build_final_projection(
    root: PlanNode,
    original_request: &ResolvedQueryRequest,
    pb: &dyn PlanBuilder,
) -> PlanNode {
    let input_schema = &root.meta().output_schema;

    let mut project_exprs: Vec<Expr> = Vec::new();
    let mut project_fields: Vec<Field> = Vec::new();

    for name in original_request
        .dimensions
        .iter()
        .chain(original_request.measures.iter())
    {
        project_exprs.push(Expr::column(name.clone()));

        // Resolve the field's data type from the input schema.
        let data_type = input_schema
            .ordinal(name)
            .and_then(|i| input_schema.field(i))
            .map(|f| f.data_type.clone())
            .unwrap_or(semstrait_core::DataType::String);

        project_fields.push(Field::new(name.clone(), data_type));
    }

    let project_schema = Schema::new(project_fields);
    pb.build_project(project_schema, root, project_exprs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_resolver::MatchedEntity;
    use indexmap::IndexMap;
    use semstrait_manifest::{
        Cardinality, CompiledRelationship, JoinColumnPair, JoinType as ModelJoinType,
    };

    fn matched(name: &str, dims: &[&str], measures: &[&str]) -> MatchedEntity {
        MatchedEntity {
            entity_name: name.to_string(),
            covered_dimensions: dims.iter().map(|s| s.to_string()).collect(),
            covered_keys: vec![],
            covered_measures: measures.iter().map(|s| s.to_string()).collect(),
            covered_metrics: vec![],
        }
    }

    fn rel(name: &str, from: &str, to: &str, from_col: &str, to_col: &str) -> CompiledRelationship {
        CompiledRelationship {
            name: name.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            join_type: ModelJoinType::Left,
            columns: vec![JoinColumnPair {
                from: from_col.to_string(),
                to: to_col.to_string(),
            }],
            cardinality: Cardinality::ManyToOne,
        }
    }

    fn empty_manifest_with_rels(rels: Vec<CompiledRelationship>) -> CompiledManifest {
        CompiledManifest {
            version: 3,
            compiled_at: chrono::Utc::now(),
            source_hash: "test".to_string(),
            entities: IndexMap::new(),
            relationships: rels,
            relationship_graph: semstrait_manifest::RelationshipGraph::default(),
            field_index: semstrait_manifest::FieldIndex::default(),
            semantic_graph: semstrait_manifest::SemanticGraph::default(),
            diagnostics: semstrait_manifest::CompileDiagnostics::default(),
            model_name: "test".to_string(),
            model_description: None,
            catalog_snapshot: None,
        }
    }

    // ── compute_join_order tests ────────────────────────────────────────

    #[test]
    fn test_compute_join_order_two_entities() {
        let r = rel("r0", "orders", "customers", "customer_id", "id");
        let manifest = empty_manifest_with_rels(vec![r]);
        let other = vec![matched("customers", &["customer_name"], &[])];

        let steps = compute_join_order("orders", &other, &[0], &manifest).unwrap();

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].entity_name, "customers");
        assert_eq!(steps[0].relationship_idx, 0);
        assert!(!steps[0].reversed);
    }

    #[test]
    fn test_compute_join_order_three_entities_chain() {
        // orders -> customers -> regions
        let r0 = rel("r0", "orders", "customers", "customer_id", "id");
        let r1 = rel("r1", "customers", "regions", "region_id", "id");
        let manifest = empty_manifest_with_rels(vec![r0, r1]);
        let other = vec![
            matched("customers", &[], &[]),
            matched("regions", &["region_name"], &[]),
        ];

        let steps = compute_join_order("orders", &other, &[0, 1], &manifest).unwrap();

        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].entity_name, "customers");
        assert_eq!(steps[0].relationship_idx, 0);
        assert!(!steps[0].reversed);
        assert_eq!(steps[1].entity_name, "regions");
        assert_eq!(steps[1].relationship_idx, 1);
        assert!(!steps[1].reversed);
    }

    #[test]
    fn test_compute_join_order_reversed() {
        // Relationship from → to is customers → orders, but anchor is "orders".
        // So joining "customers" requires using the reverse direction.
        let r = rel("r0", "customers", "orders", "id", "customer_id");
        let manifest = empty_manifest_with_rels(vec![r]);
        let other = vec![matched("customers", &["customer_name"], &[])];

        let steps = compute_join_order("orders", &other, &[0], &manifest).unwrap();

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].entity_name, "customers");
        assert!(steps[0].reversed, "should be reversed since rel.to = orders (anchor)");
    }

    // ── build_entity_requests tests ────────────────────────────────────

    #[test]
    fn test_build_entity_requests_augments_join_keys() {
        let r = rel("r0", "orders", "customers", "customer_id", "id");
        let manifest = empty_manifest_with_rels(vec![r]);

        let match_result = MatchResult {
            entities: vec![
                matched("orders", &["date"], &["revenue"]),
                matched("customers", &["customer_name"], &[]),
            ],
            join_path: vec![0],
        };
        let join_order = vec![JoinStep {
            entity_name: "customers".to_string(),
            relationship_idx: 0,
            reversed: false,
        }];
        let original = crate::request::ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: std::collections::HashMap::new(),
        };

        let reqs = build_entity_requests(&match_result, &join_order, &manifest, &original).unwrap();

        // Orders should have "customer_id" augmented (it's the from-side join key).
        let orders_req = &reqs[0];
        assert!(
            orders_req.request.dimensions.contains(&"customer_id".to_string()),
            "orders should have customer_id augmented, got: {:?}",
            orders_req.request.dimensions
        );
        assert!(
            orders_req.extra_join_keys.contains(&"customer_id".to_string()),
            "customer_id should be tracked as extra join key"
        );

        // Customers should have "id" augmented (it's the to-side join key).
        let customers_req = &reqs[1];
        assert!(
            customers_req.request.dimensions.contains(&"id".to_string()),
            "customers should have id augmented, got: {:?}",
            customers_req.request.dimensions
        );
        assert!(
            customers_req.extra_join_keys.contains(&"id".to_string()),
            "id should be tracked as extra join key"
        );
    }

    #[test]
    fn test_build_entity_requests_no_extra_when_already_selected() {
        // If the join key is already in the user's original select, it shouldn't be
        // tracked as an extra (it should remain in the final projection).
        let r = rel("r0", "orders", "customers", "date", "date");
        let manifest = empty_manifest_with_rels(vec![r]);

        let match_result = MatchResult {
            entities: vec![
                matched("orders", &["date"], &["revenue"]),
                matched("customers", &["date", "customer_name"], &[]),
            ],
            join_path: vec![0],
        };
        let join_order = vec![JoinStep {
            entity_name: "customers".to_string(),
            relationship_idx: 0,
            reversed: false,
        }];
        let original = crate::request::ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: std::collections::HashMap::new(),
        };

        let reqs = build_entity_requests(&match_result, &join_order, &manifest, &original).unwrap();

        // "date" is already in the original select, so it should NOT be an extra.
        assert!(
            reqs[0].extra_join_keys.is_empty(),
            "orders should have no extra join keys: {:?}",
            reqs[0].extra_join_keys
        );
        assert!(
            reqs[1].extra_join_keys.is_empty(),
            "customers should have no extra join keys: {:?}",
            reqs[1].extra_join_keys
        );
    }

    // ── build_semantic_join_condition tests ─────────────────────────────

    #[test]
    fn test_build_semantic_join_condition_forward() {
        let r = rel("r0", "orders", "customers", "customer_id", "id");
        let condition = build_semantic_join_condition(&r, false);

        // Should be: customer_id = id
        match &condition {
            Expr::BinaryOp(bin) => {
                assert!(matches!(bin.left.as_ref(), Expr::Column(c) if c.name == "customer_id"));
                assert!(matches!(bin.right.as_ref(), Expr::Column(c) if c.name == "id"));
            }
            _ => panic!("expected BinaryOp, got {:?}", condition),
        }
    }

    #[test]
    fn test_build_semantic_join_condition_reversed() {
        let r = rel("r0", "orders", "customers", "customer_id", "id");
        let condition = build_semantic_join_condition(&r, true);

        // Reversed: the "to" column is on the left (already joined).
        // Should be: id = customer_id
        match &condition {
            Expr::BinaryOp(bin) => {
                assert!(matches!(bin.left.as_ref(), Expr::Column(c) if c.name == "id"));
                assert!(matches!(bin.right.as_ref(), Expr::Column(c) if c.name == "customer_id"));
            }
            _ => panic!("expected BinaryOp, got {:?}", condition),
        }
    }

    #[test]
    fn test_build_semantic_join_condition_multi_column() {
        let r = CompiledRelationship {
            name: "r0".to_string(),
            from: "orders".to_string(),
            to: "customers".to_string(),
            join_type: ModelJoinType::Left,
            columns: vec![
                JoinColumnPair { from: "customer_id".to_string(), to: "id".to_string() },
                JoinColumnPair { from: "region".to_string(), to: "region".to_string() },
            ],
            cardinality: Cardinality::ManyToOne,
        };
        let condition = build_semantic_join_condition(&r, false);

        // Should be: (customer_id = id) AND (region = region)
        match &condition {
            Expr::BinaryOp(bin) => {
                assert_eq!(bin.op, semstrait_ir::BinaryOp::And);
            }
            _ => panic!("expected AND of two conditions, got {:?}", condition),
        }
    }
}
