//! SemanticPlanner — the main entry point for query planning.
//!
//! Orchestrates the full planning pipeline:
//! 1. Constraint evaluation
//! 2. Kind dispatch
//! 3. DataKindPlanner resolution
//! 4. Additivity resolution
//! 5. Filter injection
//! 6. Optimizer application

use std::sync::Arc;

use semstrait_catalog::CatalogProvider;
use semstrait_ir::{
    BinaryOp, DefaultPlanBuilder, Expr, LogicalPlan, PlanBuilder,
    PlanNode, SortKey,
};
use semstrait_manifest::CompiledManifest;

use crate::additivity::AdditivityResolver;
use crate::validator::ConstraintValidator;
use crate::error::PlannerError;
use crate::data_kind::{
    DataKindPlannerRegistry, PlanFragment, PlannerContext, PrunedView,
};
use crate::optimizer::{Optimizer, OptimizerPass};
use crate::request::{FilterOperator, FilterValue, ResolvedQueryRequest, SortDirection};

/// The semantic query planner.
///
/// Builds a `LogicalPlan` from a `ResolvedQueryRequest` + `CompiledManifest`.
/// Constructed via `SemanticPlannerBuilder`.
pub struct SemanticPlanner {
    catalog: Option<Arc<dyn CatalogProvider>>,
    optimizer: Optimizer,
    planners: DataKindPlannerRegistry,
    plan_builder: Box<dyn PlanBuilder>,
}

impl SemanticPlanner {
    /// Create a builder for configuring and constructing a SemanticPlanner.
    pub fn builder() -> SemanticPlannerBuilder {
        SemanticPlannerBuilder::new()
    }

    /// Plan a query request against the compiled manifest.
    ///
    /// If `entity_name` is None, dispatches to ad-hoc resolution which infers
    /// the target entity from the requested field names.
    pub fn plan(
        &self,
        request: &ResolvedQueryRequest,
        manifest: &CompiledManifest,
    ) -> Result<LogicalPlan, PlannerError> {
        // If no entity specified, resolve from fields.
        if request.entity_name.is_empty() {
            return self.plan_ad_hoc(request, manifest);
        }

        // Step 1: Constraint evaluation (pre-resolution validity gate).
        ConstraintValidator::check(request, manifest)?;

        // Step 2: Resolve entity via CompiledDataKind hierarchy.
        let ctx = PlannerContext {
            manifest,
            catalog: self.catalog.as_deref(),
            session: &request.session_variables,
            plan_builder: self.plan_builder.as_ref(),
        };

        let (fragment, entity_measures) =
            self.resolve_entity(request, manifest, &ctx)?;

        // Step 7: Additivity resolution for each measure.
        let mut fragment = fragment;
        for measure_name in &request.measures {
            if let Some(measure) = entity_measures.get(measure_name) {
                fragment =
                    AdditivityResolver::resolve(fragment, measure, request)?;
            }
        }

        // Step 8: Inject filters.
        // Note: entity-level filters are now injected at scan level inside
        // build_layered_plan() (between scan and rename, using physical names).
        let mut root = fragment.root;

        let pb = self.plan_builder.as_ref();

        // 8e: Inject user filters from the request.
        root = inject_user_filters(root, request, pb)?;

        // Step 9: Apply ORDER BY.
        root = apply_order_by(root, request, pb)?;

        // Step 10: Apply LIMIT.
        root = apply_limit(root, request, pb)?;

        // Step 11: Build LogicalPlan.
        let output_names: Vec<String> = request
            .dimensions
            .iter()
            .chain(request.measures.iter())
            .cloned()
            .collect();

        let plan = LogicalPlan::new(root, output_names);

        // Step 12: Optimizer pass.
        self.optimizer.apply(plan)
    }

    /// Resolve the entity and build the plan fragment.
    ///
    /// Returns (fragment, measures_map) where measures_map is borrowed from the
    /// entity for post-resolution processing (additivity).
    /// Used by both `plan()` and `ad_hoc_join` for per-entity planning.
    ///
    /// Entity-level filters are injected at scan level inside `build_layered_plan()`
    /// (between scan and rename, using physical column names) — not here.
    ///
    /// Unified dispatch through CompiledDataKind. Dataset variants use the fast path;
    /// complex kinds (grainset/unionset/joinset) delegate to DataKindPlanner registry.
    pub(crate) fn resolve_entity<'a>(
        &self,
        request: &ResolvedQueryRequest,
        manifest: &'a CompiledManifest,
        ctx: &PlannerContext<'_>,
    ) -> Result<
        (
            PlanFragment,
            &'a indexmap::IndexMap<String, semstrait_manifest::CompiledMeasure>,
        ),
        PlannerError,
    > {
        // Resolve via CompiledDataKind (primary path).
        let entity_name = request.entity_name.as_str();
        let data_kind = manifest
            .resolve(entity_name)
            .ok_or_else(|| PlannerError::KindNotFound(entity_name.to_string()))?;
        let iface = data_kind.interface();

        // Prune bindings by metadata and literal filters (borrow-only, no clone).
        let mut pruned = PrunedView::all(data_kind);
        pruned.prune_by_metadata(request)?;
        pruned.prune_by_literals(request)?;

        // Dispatch through CompiledDataKind.
        let fragment =
            crate::data_kind::dispatch_data_kind(&pruned, request, ctx, &self.planners)?;

        Ok((fragment, &iface.measures))
    }

    /// Access the plan builder (for ad_hoc_join module).
    pub(crate) fn plan_builder(&self) -> &dyn PlanBuilder {
        self.plan_builder.as_ref()
    }

    /// Run the optimizer on a plan (for ad_hoc_join module).
    pub(crate) fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan, PlannerError> {
        self.optimizer.apply(plan)
    }

    /// Plan an ad-hoc query where `FROM` is omitted.
    ///
    /// Uses `entity_resolver::find_covering_entities()` to score all entities and find
    /// the best covering set. For single-entity resolution, reclassifies the requested
    ///
    /// Multi-entity join synthesis returns an error until Phase 4.
    pub fn plan_ad_hoc(
        &self,
        request: &ResolvedQueryRequest,
        manifest: &CompiledManifest,
    ) -> Result<LogicalPlan, PlannerError> {
        use crate::entity_resolver;

        // parse.rs puts ALL select names in request.dimensions for ad-hoc.
        let all_fields: Vec<String> = request
            .dimensions
            .iter()
            .chain(request.measures.iter())
            .cloned()
            .collect();

        // Find covering entity set via unified resolution API.
        let match_result = entity_resolver::find_covering_entities(&all_fields, manifest)?;

        if match_result.is_single() {
            // Single-entity fast path — reclassify fields and delegate to plan().
            let matched = &match_result.entities[0];
            let entity = manifest
                .resolve(&matched.entity_name)
                .ok_or_else(|| PlannerError::KindNotFound(matched.entity_name.clone()))?;
            let reclassified = entity_resolver::reclassify_fields(&all_fields, entity.interface())?;

            let mut targeted = request.clone();
            targeted.entity_name = matched.entity_name.clone();
            // Keys are classified as dimensions for GROUP BY.
            targeted.dimensions = reclassified
                .dimensions
                .into_iter()
                .chain(reclassified.keys)
                .collect();
            targeted.measures = reclassified
                .measures
                .into_iter()
                .chain(reclassified.metrics)
                .collect();

            self.plan(&targeted, manifest)
        } else {
            // Multi-entity join synthesis.
            crate::ad_hoc_join::build_ad_hoc_join_plan(
                self, &match_result, request, manifest,
            )
        }
    }
}

/// Builder for SemanticPlanner.
pub struct SemanticPlannerBuilder {
    catalog: Option<Arc<dyn CatalogProvider>>,
    passes: Vec<Box<dyn OptimizerPass>>,
    plan_builder: Box<dyn PlanBuilder>,
}

impl SemanticPlannerBuilder {
    pub fn new() -> Self {
        Self {
            catalog: None,
            passes: Vec::new(),
            plan_builder: Box::new(DefaultPlanBuilder),
        }
    }

    /// Set the catalog provider.
    pub fn with_catalog(mut self, catalog: Arc<dyn CatalogProvider>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// Add an optimizer pass.
    pub fn with_optimizer_pass(mut self, pass: impl OptimizerPass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Set the engine-specific plan builder.
    pub fn with_plan_builder(mut self, builder: Box<dyn PlanBuilder>) -> Self {
        self.plan_builder = builder;
        self
    }

    /// Build the SemanticPlanner.
    pub fn build(self) -> SemanticPlanner {
        SemanticPlanner {
            catalog: self.catalog,
            optimizer: Optimizer::new(self.passes),
            planners: DataKindPlannerRegistry::new(),
            plan_builder: self.plan_builder,
        }
    }
}

impl Default for SemanticPlannerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Filter injection helpers
// ============================================================================

/// Convert user QueryFilters into FilterNodes wrapping the plan root.
pub(crate) fn inject_user_filters(
    mut root: PlanNode,
    request: &ResolvedQueryRequest,
    plan_builder: &dyn PlanBuilder,
) -> Result<PlanNode, PlannerError> {
    for filter in &request.filters {
        let predicate = query_filter_to_expr(filter)?;
        let schema = (*root.meta().output_schema).clone();
        root = plan_builder.build_filter(schema, root, predicate);
    }
    Ok(root)
}

/// Convert a QueryFilter into an Expr predicate.
fn query_filter_to_expr(
    filter: &crate::request::QueryFilter,
) -> Result<Expr, PlannerError> {
    let column = Expr::column(filter.field.clone());

    match &filter.operator {
        FilterOperator::Eq
        | FilterOperator::NotEq
        | FilterOperator::Lt
        | FilterOperator::LtEq
        | FilterOperator::Gt
        | FilterOperator::GtEq => {
            let first = filter.values.first().ok_or_else(|| {
                PlannerError::Internal(format!(
                    "{:?} filter requires at least 1 value",
                    filter.operator
                ))
            })?;
            let value = filter_value_to_expr(first)?;
            let op = match filter.operator {
                FilterOperator::Eq => BinaryOp::Eq,
                FilterOperator::NotEq => BinaryOp::NotEq,
                FilterOperator::Lt => BinaryOp::Lt,
                FilterOperator::LtEq => BinaryOp::LtEq,
                FilterOperator::Gt => BinaryOp::Gt,
                FilterOperator::GtEq => BinaryOp::GtEq,
                _ => unreachable!(),
            };
            Ok(Expr::binary(column, op, value))
        }
        FilterOperator::In => {
            // IN is translated as OR chain: col = v1 OR col = v2 OR ...
            let mut expr: Option<Expr> = None;
            for val in &filter.values {
                let eq = Expr::eq(column.clone(), filter_value_to_expr(val)?);
                expr = Some(match expr {
                    None => eq,
                    Some(prev) => Expr::or(prev, eq),
                });
            }
            expr.ok_or_else(|| PlannerError::Internal("IN filter with no values".to_string()))
        }
        FilterOperator::NotIn => {
            // NOT IN is translated as AND chain: col != v1 AND col != v2 AND ...
            let mut expr: Option<Expr> = None;
            for val in &filter.values {
                let neq = Expr::ne(column.clone(), filter_value_to_expr(val)?);
                expr = Some(match expr {
                    None => neq,
                    Some(prev) => Expr::and(prev, neq),
                });
            }
            expr.ok_or_else(|| {
                PlannerError::Internal("NOT IN filter with no values".to_string())
            })
        }
        FilterOperator::Between => {
            // BETWEEN is: col >= low AND col <= high
            if filter.values.len() != 2 {
                return Err(PlannerError::Internal(
                    "BETWEEN filter requires exactly 2 values".to_string(),
                ));
            }
            let low = filter_value_to_expr(&filter.values[0])?;
            let high = filter_value_to_expr(&filter.values[1])?;
            Ok(Expr::and(
                Expr::gte(column.clone(), low),
                Expr::lte(column, high),
            ))
        }
        FilterOperator::IsNull => Ok(Expr::is_null(column)),
        FilterOperator::IsNotNull => Ok(Expr::is_not_null(column)),
    }
}

/// Convert a FilterValue to an Expr.
fn filter_value_to_expr(value: &FilterValue) -> Result<Expr, PlannerError> {
    match value {
        FilterValue::String(s) => Ok(Expr::string(s)),
        FilterValue::Number(n) => Ok(Expr::float(*n)),
        FilterValue::Bool(b) => Ok(Expr::boolean(*b)),
        FilterValue::Null => Ok(Expr::null()),
    }
}

// ============================================================================
// ORDER BY and LIMIT helpers
// ============================================================================

/// Apply ORDER BY clauses from the request.
pub(crate) fn apply_order_by(
    root: PlanNode,
    request: &ResolvedQueryRequest,
    plan_builder: &dyn PlanBuilder,
) -> Result<PlanNode, PlannerError> {
    if request.order_by.is_empty() {
        return Ok(root);
    }

    let sort_keys: Vec<SortKey> = request
        .order_by
        .iter()
        .map(|ob| SortKey {
            expr: Expr::column(ob.field.clone()),
            direction: match ob.direction {
                SortDirection::Ascending => semstrait_ir::SortDirection::Ascending,
                SortDirection::Descending => semstrait_ir::SortDirection::Descending,
            },
        })
        .collect();

    let schema = (*root.meta().output_schema).clone();
    Ok(plan_builder.build_sort(schema, root, sort_keys))
}

/// Apply LIMIT from the request.
pub(crate) fn apply_limit(
    root: PlanNode,
    request: &ResolvedQueryRequest,
    plan_builder: &dyn PlanBuilder,
) -> Result<PlanNode, PlannerError> {
    match request.limit {
        None => Ok(root),
        Some(limit) => {
            let schema = (*root.meta().output_schema).clone();
            Ok(plan_builder.build_fetch(schema, root, Some(limit as i64), 0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::helpers::*;
    use crate::request::{FilterOperator, FilterValue, OrderByClause, QueryFilter, SortDirection};

    #[test]
    fn test_plan_basic_grainset() {
        let manifest = make_test_manifest();
        let request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "basic grainset planning should succeed");
        let plan = result.unwrap();
        assert_eq!(plan.output_names.len(), 3); // date, region, revenue
        assert_eq!(plan.output_names, vec!["date", "region", "revenue"]);
    }

    #[test]
    fn test_plan_with_filters() {
        let manifest = make_test_manifest();
        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request.filters = vec![QueryFilter {
            field: "region".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("US".to_string())],
        }];

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "planning with filters should succeed");
        let plan = result.unwrap();

        // Verify FilterNode exists in the plan
        let has_filter = contains_filter_node(&plan.root);
        assert!(has_filter, "plan should contain a FilterNode");
    }

    #[test]
    fn test_plan_with_order_by() {
        let manifest = make_test_manifest();
        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request.order_by = vec![OrderByClause {
            field: "revenue".to_string(),
            direction: SortDirection::Descending,
        }];

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "planning with order_by should succeed");
        let plan = result.unwrap();

        // Verify SortNode exists in the plan
        let has_sort = contains_sort_node(&plan.root);
        assert!(has_sort, "plan should contain a SortNode");
    }

    #[test]
    fn test_plan_with_limit() {
        let manifest = make_test_manifest();
        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request.limit = Some(100);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "planning with limit should succeed");
        let plan = result.unwrap();

        // Verify FetchNode exists in the plan
        let has_fetch = contains_fetch_node(&plan.root);
        assert!(has_fetch, "plan should contain a FetchNode");
    }

    #[test]
    fn test_plan_kind_not_found() {
        let manifest = make_test_manifest();
        let request = make_test_request("nonexistent_kind", vec!["date"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_err(), "should error when kind doesn't exist");
        assert!(matches!(result.unwrap_err(), PlannerError::KindNotFound(_)));
    }

    // Helper functions to check for node types in the plan tree
    fn contains_filter_node(node: &PlanNode) -> bool {
        match node {
            PlanNode::Filter(_) => true,
            PlanNode::Project(n) => contains_filter_node(&n.input),
            PlanNode::Aggregate(n) => contains_filter_node(&n.input),
            PlanNode::Sort(n) => contains_filter_node(&n.input),
            PlanNode::Fetch(n) => contains_filter_node(&n.input),
            PlanNode::Join(n) => contains_filter_node(&n.left) || contains_filter_node(&n.right),
            PlanNode::Union(n) => n.inputs.iter().any(contains_filter_node),
            PlanNode::Scan(_) => false,
        }
    }

    fn contains_sort_node(node: &PlanNode) -> bool {
        match node {
            PlanNode::Sort(_) => true,
            PlanNode::Project(n) => contains_sort_node(&n.input),
            PlanNode::Aggregate(n) => contains_sort_node(&n.input),
            PlanNode::Filter(n) => contains_sort_node(&n.input),
            PlanNode::Fetch(n) => contains_sort_node(&n.input),
            PlanNode::Join(n) => contains_sort_node(&n.left) || contains_sort_node(&n.right),
            PlanNode::Union(n) => n.inputs.iter().any(contains_sort_node),
            PlanNode::Scan(_) => false,
        }
    }

    fn contains_fetch_node(node: &PlanNode) -> bool {
        match node {
            PlanNode::Fetch(_) => true,
            PlanNode::Project(n) => contains_fetch_node(&n.input),
            PlanNode::Aggregate(n) => contains_fetch_node(&n.input),
            PlanNode::Filter(n) => contains_fetch_node(&n.input),
            PlanNode::Sort(n) => contains_fetch_node(&n.input),
            PlanNode::Join(n) => contains_fetch_node(&n.left) || contains_fetch_node(&n.right),
            PlanNode::Union(n) => n.inputs.iter().any(contains_fetch_node),
            PlanNode::Scan(_) => false,
        }
    }

    #[test]
    fn test_plan_with_kind_filter() {
        let mut manifest = make_test_manifest();
        // Add a kind-level filter: region = 'US'.
        if let Some(dk) = manifest.entities.get_mut("orders") {
            dk.interface_mut().filters.push(semstrait_manifest::CompiledFilter {
                name: "us_only".to_string(),
                expr: semstrait_core::Expr::eq(
                    semstrait_core::Expr::column("region"),
                    semstrait_core::Expr::string("US"),
                ),
                expr_source: "region = 'US'".to_string(),
            });
        }

        let request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "plan with kind filter should succeed: {:?}", result.err());

        let plan = result.unwrap();
        // Should have a FilterNode from the kind-level filter.
        assert!(contains_filter_node(&plan.root), "plan should contain kind-level filter");
    }

    #[test]
    fn test_plan_kind_filter_combined_with_user_filter() {
        let mut manifest = make_test_manifest();
        // Add a kind-level filter.
        if let Some(dk) = manifest.entities.get_mut("orders") {
            dk.interface_mut().filters.push(semstrait_manifest::CompiledFilter {
                name: "active_only".to_string(),
                expr: semstrait_core::Expr::eq(
                    semstrait_core::Expr::column("region"),
                    semstrait_core::Expr::string("US"),
                ),
                expr_source: "region = 'US'".to_string(),
            });
        }

        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        // Also add a user filter.
        request.filters = vec![QueryFilter {
            field: "date".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("2024-01-01".to_string())],
        }];

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "plan with both filters should succeed");

        let plan = result.unwrap();
        // Count filter nodes — should be at least 2 (kind + user).
        let filter_count = count_filter_nodes(&plan.root);
        assert!(filter_count >= 2, "should have at least 2 filter nodes (kind + user), got {}", filter_count);
    }

    fn count_filter_nodes(node: &PlanNode) -> usize {
        match node {
            PlanNode::Filter(n) => 1 + count_filter_nodes(&n.input),
            PlanNode::Project(n) => count_filter_nodes(&n.input),
            PlanNode::Aggregate(n) => count_filter_nodes(&n.input),
            PlanNode::Sort(n) => count_filter_nodes(&n.input),
            PlanNode::Fetch(n) => count_filter_nodes(&n.input),
            PlanNode::Join(n) => count_filter_nodes(&n.left) + count_filter_nodes(&n.right),
            PlanNode::Union(n) => n.inputs.iter().map(count_filter_nodes).sum(),
            PlanNode::Scan(_) => 0,
        }
    }

    #[test]
    fn test_plan_with_inline_filter_produces_filter_node() {
        // Request-scope inline filter (via request.inline_filters) should
        // produce a FilterNode at the scan layer — identical engine to
        // named DataKind filters. See `11 §6.4.2` and `19 §7.1`.
        let manifest = make_test_manifest();

        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request.inline_filters = vec![semstrait_manifest::CompiledFilter {
            name: "__inline_filter_0".to_string(),
            expr: semstrait_core::Expr::eq(
                semstrait_core::Expr::entity_ref("region"),
                semstrait_core::Expr::string("US"),
            ),
            expr_source: "region = 'US'".to_string(),
        }];

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "plan with inline filter should succeed: {:?}", result.err());

        let plan = result.unwrap();
        assert!(
            contains_filter_node(&plan.root),
            "plan should contain a FilterNode from the inline filter"
        );
    }

    #[test]
    fn test_plan_inline_filter_equivalent_to_named_filter() {
        // Inline filters share the named-filter scan-layer engine: the resulting
        // plan tree should be structurally identical whether the predicate is
        // carried as an iface-level CompiledFilter or as a request-scope
        // inline filter.

        // Variant A: predicate as a named DataKind filter on the interface.
        let mut manifest_named = make_test_manifest();
        if let Some(dk) = manifest_named.entities.get_mut("orders") {
            dk.interface_mut().filters.push(semstrait_manifest::CompiledFilter {
                name: "us_only".to_string(),
                expr: semstrait_core::Expr::eq(
                    semstrait_core::Expr::entity_ref("region"),
                    semstrait_core::Expr::string("US"),
                ),
                expr_source: "region = 'US'".to_string(),
            });
        }
        let request_named = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        let plan_named = SemanticPlanner::builder()
            .build()
            .plan(&request_named, &manifest_named)
            .expect("named-filter plan should succeed");

        // Variant B: same predicate as a request-scope inline filter.
        let manifest_inline = make_test_manifest();
        let mut request_inline =
            make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request_inline.inline_filters = vec![semstrait_manifest::CompiledFilter {
            name: "__inline_filter_0".to_string(),
            expr: semstrait_core::Expr::eq(
                semstrait_core::Expr::entity_ref("region"),
                semstrait_core::Expr::string("US"),
            ),
            expr_source: "region = 'US'".to_string(),
        }];
        let plan_inline = SemanticPlanner::builder()
            .build()
            .plan(&request_inline, &manifest_inline)
            .expect("inline-filter plan should succeed");

        // Both plans must contain exactly the same number of FilterNodes.
        assert_eq!(
            count_filter_nodes(&plan_named.root),
            count_filter_nodes(&plan_inline.root),
            "inline and named filters should produce the same number of FilterNodes"
        );

        // Output names match.
        assert_eq!(plan_named.output_names, plan_inline.output_names);

        // The extracted Filter predicates must be byte-identical — the
        // CompiledFilter.name doesn't reach the plan tree, only the Expr
        // does, so named vs inline carrier choice is structurally invisible
        // past scan-layer injection.
        let preds_named = collect_filter_predicates(&plan_named.root);
        let preds_inline = collect_filter_predicates(&plan_inline.root);
        assert_eq!(
            preds_named, preds_inline,
            "inline filter should produce the same FilterNode predicates as a named filter"
        );
    }

    fn collect_filter_predicates(node: &PlanNode) -> Vec<semstrait_ir::Expr> {
        let mut out = Vec::new();
        walk_filter_predicates(node, &mut out);
        out
    }

    fn walk_filter_predicates(node: &PlanNode, out: &mut Vec<semstrait_ir::Expr>) {
        match node {
            PlanNode::Filter(n) => {
                out.push(n.predicate.clone());
                walk_filter_predicates(&n.input, out);
            }
            PlanNode::Project(n) => walk_filter_predicates(&n.input, out),
            PlanNode::Aggregate(n) => walk_filter_predicates(&n.input, out),
            PlanNode::Sort(n) => walk_filter_predicates(&n.input, out),
            PlanNode::Fetch(n) => walk_filter_predicates(&n.input, out),
            PlanNode::Join(n) => {
                walk_filter_predicates(&n.left, out);
                walk_filter_predicates(&n.right, out);
            }
            PlanNode::Union(n) => {
                for input in &n.inputs {
                    walk_filter_predicates(input, out);
                }
            }
            PlanNode::Scan(_) => {}
        }
    }

    #[test]
    fn test_plan_inline_filter_combined_with_named_filter() {
        // Inline filters compose with named DataKind filters — both should
        // emit a FilterNode in the same scan-layer pass.
        let mut manifest = make_test_manifest();
        if let Some(dk) = manifest.entities.get_mut("orders") {
            dk.interface_mut().filters.push(semstrait_manifest::CompiledFilter {
                name: "active_only".to_string(),
                expr: semstrait_core::Expr::eq(
                    semstrait_core::Expr::entity_ref("region"),
                    semstrait_core::Expr::string("US"),
                ),
                expr_source: "region = 'US'".to_string(),
            });
        }

        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request.inline_filters = vec![semstrait_manifest::CompiledFilter {
            name: "__inline_filter_0".to_string(),
            expr: semstrait_core::Expr::eq(
                semstrait_core::Expr::entity_ref("date"),
                semstrait_core::Expr::string("2024-01-01"),
            ),
            expr_source: "date = '2024-01-01'".to_string(),
        }];

        let planner = SemanticPlanner::builder().build();
        let plan = planner.plan(&request, &manifest).expect("combined plan should succeed");

        let n_filters = count_filter_nodes(&plan.root);
        assert!(
            n_filters >= 2,
            "expected at least 2 FilterNodes (named + inline), got {}",
            n_filters
        );
    }

    #[test]
    fn test_plan_no_kind_filters() {
        // Verify baseline: no kind filters means no extra filter nodes
        // (unless user adds one).
        let manifest = make_test_manifest();
        let request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let plan = planner.plan(&request, &manifest).expect("should succeed");
        assert!(!contains_filter_node(&plan.root), "no filter should be present without user/kind filters");
    }

    #[test]
    fn test_metadata_dimension_filter_prunes_datasets() {
        use indexmap::IndexMap;
        use semstrait_manifest::{
            CompiledDimension, CompiledMeasure, DimensionType,
            MetadataDimension, PathExtraction,
        };
        use semstrait_manifest::acceleration::{
            CoverageIndex, CompiledDataKind, DatasetBinding, DimensionIndex, CompiledGrainsetKind,
            CompiledInterface, ResolvedColumnMapping,
        };

        // Create a kind with 2 datasets, each with a different source path.
        let mut dimensions = IndexMap::new();
        dimensions.insert(
            "date".to_string(),
            CompiledDimension {
                name: "date".to_string(),
                description: None,
                data_type: semstrait_core::DataType::String,
                dim_type: DimensionType::Categorical(
                    semstrait_manifest::CategoricalDimension { enum_values: None },
                ),
                expr: None,
                expr_source: None,
            },
        );
        dimensions.insert(
            "source".to_string(),
            CompiledDimension {
                name: "source".to_string(),
                description: None,
                data_type: semstrait_core::DataType::String,
                dim_type: DimensionType::Metadata(MetadataDimension {
                    path: Some(PathExtraction { token: 1 }),
                    partition: None,
                }),
                expr: None,
                expr_source: None,
            },
        );

        let mut measures = IndexMap::new();
        measures.insert(
            "revenue".to_string(),
            CompiledMeasure {
                name: "revenue".to_string(),
                description: None,
                data_type: semstrait_core::DataType::Number,
                agg: semstrait_core::Aggregation::Sum,
                expr: semstrait_core::Expr::entity_ref("amount"),
                expr_source: "amount".to_string(),
                additivity: None,
                constraints: None,
                filters: vec![],
            },
        );

        let make_binding = |name: &str, source: &str| -> DatasetBinding {
            let mut physical = IndexMap::new();
            physical.insert("date".to_string(), "order_date".to_string());
            physical.insert("revenue".to_string(), "amount".to_string());
            DatasetBinding {
                dataset_name: name.to_string(),
                column_mapping: ResolvedColumnMapping {
                    physical,
                    literals: std::collections::HashMap::new(),
                    temporal: std::collections::HashMap::new(),
                    anchored: std::collections::HashMap::new(),
                },
                resolved_sources: vec![semstrait_manifest::ResolvedSource::path(source)],
            }
        };

        let bindings = vec![
            make_binding("shopify_data", "bucket/shopify/data.parquet"),
            make_binding("ga4_data", "bucket/ga4/data.parquet"),
        ];

        let interface = CompiledInterface {
            name: "multi_source".to_string(),
            description: None,
            dimensions: dimensions.clone(),
            measures: measures.clone(),
            metrics: IndexMap::new(),
            keys: None,
            filters: vec![],
            temporal_dim: None,
        };

        let coverage_index = CoverageIndex::build(&dimensions, &measures, &bindings);
        let dimension_index = DimensionIndex::build(&dimensions, &bindings);

        let data_kind = CompiledDataKind::Grainset(Box::new(CompiledGrainsetKind {
            interface,
            bindings,
            coverage_index,
            dimension_index,
            metric_order: None,
            grain_map: None,
        }));

        let mut entities = IndexMap::new();
        entities.insert("multi_source".to_string(), data_kind);

        let manifest = semstrait_manifest::CompiledManifest {
            version: 3,
            compiled_at: chrono::Utc::now(),
            source_hash: "test_meta_filter".to_string(),
            relationships: vec![],
            model_name: "test_meta_filter".to_string(),
            model_description: None,
            entities,
            relationship_graph: semstrait_manifest::RelationshipGraph::default(),
            field_index: semstrait_manifest::FieldIndex::default(),
            diagnostics: semstrait_manifest::CompileDiagnostics::default(),
            semantic_graph: semstrait_manifest::SemanticGraph::default(),
            catalog_snapshot: None,
        };

        // Query with a metadata filter: source = 'shopify'
        let mut request = make_test_request(
            "multi_source",
            vec!["date", "source"],
            vec!["revenue"],
        );
        request.filters = vec![QueryFilter {
            field: "source".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("shopify".to_string())],
        }];

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "metadata filter should succeed: {:?}", result.err());

        // The plan should only scan from the shopify dataset.
        let plan = result.unwrap();
        let scan_tables = collect_scan_tables(&plan.root);
        assert_eq!(scan_tables.len(), 1);
        assert!(
            scan_tables[0].contains("shopify"),
            "should scan shopify dataset, got: {:?}",
            scan_tables
        );
    }

    fn collect_scan_tables(node: &PlanNode) -> Vec<String> {
        match node {
            PlanNode::Scan(s) => vec![s.table_name.clone()],
            PlanNode::Project(n) => collect_scan_tables(&n.input),
            PlanNode::Aggregate(n) => collect_scan_tables(&n.input),
            PlanNode::Filter(n) => collect_scan_tables(&n.input),
            PlanNode::Sort(n) => collect_scan_tables(&n.input),
            PlanNode::Fetch(n) => collect_scan_tables(&n.input),
            PlanNode::Join(n) => {
                let mut v = collect_scan_tables(&n.left);
                v.extend(collect_scan_tables(&n.right));
                v
            }
            PlanNode::Union(n) => n.inputs.iter().flat_map(collect_scan_tables).collect(),
        }
    }

    // ================================================================
    // Dataset planning tests
    // ================================================================

    #[test]
    fn test_plan_dataset_basic() {
        let manifest = make_test_manifest();
        let request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "Dataset planning should succeed: {:?}", result.err());
        let plan = result.unwrap();
        assert_eq!(plan.output_names.len(), 3);
        assert_eq!(plan.output_names, vec!["date", "region", "revenue"]);
    }

    #[test]
    fn test_plan_dataset_with_filters() {
        let manifest = make_test_manifest();
        let mut request = make_test_request("orders", vec!["date", "region"], vec!["revenue"]);
        request.filters = vec![QueryFilter {
            field: "region".to_string(),
            operator: FilterOperator::Eq,
            values: vec![FilterValue::String("US".to_string())],
        }];

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "planning with filters should succeed");
        let plan = result.unwrap();
        assert!(contains_filter_node(&plan.root), "plan should contain a FilterNode");
    }

    #[test]
    fn test_plan_dataset_with_order_by_and_limit() {
        let manifest = make_test_manifest();
        let mut request = make_test_request("orders", vec!["date"], vec!["revenue"]);
        request.order_by = vec![OrderByClause {
            field: "revenue".to_string(),
            direction: SortDirection::Descending,
        }];
        request.limit = Some(100);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "planning with order_by + limit should succeed");
        let plan = result.unwrap();
        assert!(contains_sort_node(&plan.root));
        assert!(contains_fetch_node(&plan.root));
    }

    // ================================================================
    // Ad-hoc resolution tests
    // ================================================================

    #[test]
    fn test_ad_hoc_single_dataset_resolution() {
        use semstrait_manifest::{FieldIndex, CompiledDimension, CompiledMeasure};
        use semstrait_manifest::acceleration::{CompiledDataKind, CompiledSimpleKind, DatasetBinding, ResolvedColumnMapping};
        use std::collections::HashSet;

        let mut manifest = make_test_manifest();

        let mut dims = indexmap::IndexMap::new();
        dims.insert("date".to_string(), CompiledDimension {
            name: "date".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: semstrait_manifest::DimensionType::Categorical(
                semstrait_manifest::CategoricalDimension { enum_values: None },
            ),
            expr: None,
            expr_source: None,
        });
        dims.insert("region".to_string(), CompiledDimension {
            name: "region".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: semstrait_manifest::DimensionType::Categorical(
                semstrait_manifest::CategoricalDimension { enum_values: None },
            ),
            expr: None,
            expr_source: None,
        });

        let mut measures = indexmap::IndexMap::new();
        measures.insert("revenue".to_string(), CompiledMeasure {
            name: "revenue".to_string(),
            description: None,
            data_type: semstrait_core::DataType::Number,
            agg: semstrait_core::Aggregation::Sum,
            expr: semstrait_core::Expr::entity_ref("amount"),
            expr_source: "amount".to_string(),
            additivity: None,
            constraints: None,
            filters: vec![],
        });

        // Build CompiledDataKind for ad-hoc resolution.
        let mut physical = indexmap::IndexMap::new();
        physical.insert("date".to_string(), "order_date".to_string());
        physical.insert("region".to_string(), "region_name".to_string());
        physical.insert("revenue".to_string(), "amount".to_string());

        let binding = DatasetBinding {
            dataset_name: "orders_ds".to_string(),
            column_mapping: ResolvedColumnMapping {
                physical,
                literals: std::collections::HashMap::new(),
                temporal: std::collections::HashMap::new(),
                anchored: std::collections::HashMap::new(),
            },
            resolved_sources: vec![],
        };

        let iface = semstrait_manifest::CompiledInterface {
            name: "orders_ds".to_string(),
            description: None,
            dimensions: dims,
            measures,
            metrics: indexmap::IndexMap::new(),
            keys: None,
            filters: vec![],
            temporal_dim: None,
        };

        manifest.entities.insert(
            "orders_ds".to_string(),
            CompiledDataKind::Simple(Box::new(CompiledSimpleKind { interface: iface, binding })),
        );

        // Build a FieldIndex pointing to orders_ds.
        let mut providers = std::collections::HashMap::new();
        providers.insert("date".to_string(), vec!["orders_ds".to_string()]);
        providers.insert("region".to_string(), vec!["orders_ds".to_string()]);
        providers.insert("revenue".to_string(), vec!["orders_ds".to_string()]);

        manifest.field_index = FieldIndex {
            providers,
            all_dimensions: ["date", "region"].iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
            all_measures: ["revenue"].iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
            all_metrics: HashSet::new(),
            all_keys: HashSet::new(),
        };

        // Request with empty entity_name — ad-hoc resolution should find orders_ds.
        let request = make_test_request("", vec!["date", "region"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan_ad_hoc(&request, &manifest);

        assert!(result.is_ok(), "ad-hoc single dataset should succeed: {:?}", result.err());
        let plan = result.unwrap();
        assert_eq!(plan.output_names, vec!["date", "region", "revenue"]);
    }

    #[test]
    fn test_ad_hoc_unknown_field_error() {
        let manifest = make_test_manifest();
        let request = make_test_request("", vec!["nonexistent_field"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan_ad_hoc(&request, &manifest);

        assert!(result.is_err(), "unknown field should fail");
    }

    #[test]
    fn test_v2_plan_kind_not_found() {
        let manifest = make_test_manifest();
        let request = make_test_request("nonexistent", vec!["date"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PlannerError::KindNotFound(_)));
    }

    #[test]
    fn test_plan_computed_dimension() {
        let manifest = make_computed_dim_manifest();
        let request = make_test_request("orders", vec!["date", "market"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);

        assert!(result.is_ok(), "computed dimension planning should succeed: {:?}", result.err());
        let plan = result.unwrap();
        assert_eq!(plan.output_names, vec!["date", "market", "revenue"]);

        // In the layered architecture, computed dims are projected in the Expression layer
        // (before aggregation). The Expression project contains the FunctionCall; the Final
        // Projection just references Column("market"). Find it by traversing through Aggregate.
        let agg = find_agg_node(&plan.root)
            .expect("plan should contain an AggNode");
        // Expression project is the aggregate's input (or Rename if no computed dims).
        let expr_project = match agg.input.as_ref() {
            PlanNode::Project(p) => p,
            other => panic!("expected Expression Project under Aggregate, got {:?}", std::mem::discriminant(other)),
        };
        // Find the computed dim expression — it's appended after passthrough columns.
        let market_expr = expr_project.expressions.last()
            .expect("Expression project should have expressions");
        assert!(
            matches!(market_expr, semstrait_ir::Expr::FunctionCall(_)),
            "computed dim 'market' should be a FunctionCall in Expression project, got: {:?}",
            market_expr
        );
    }

    #[test]
    fn test_plan_computed_dim_not_in_group_by() {
        let manifest = make_computed_dim_manifest();
        // Request only the computed dim + measure (no physical dims except date).
        let request = make_test_request("orders", vec!["date", "market"], vec!["revenue"]);

        let planner = SemanticPlanner::builder().build();
        let plan = planner.plan(&request, &manifest).unwrap();

        // In the layered architecture, computed dims ARE in GROUP BY (they're projected
        // in the Expression layer before aggregation). GROUP BY references semantic Column("market").
        let agg = find_agg_node(&plan.root)
            .expect("plan should contain an AggNode");
        assert_eq!(agg.group_by.len(), 2, "both date and market should be in group_by");
        let names: Vec<&str> = agg.group_by.iter().filter_map(|e| match e {
            semstrait_ir::Expr::Column(c) => Some(c.name.as_str()),
            _ => None,
        }).collect();
        assert!(names.contains(&"date"), "group_by should contain 'date'");
        assert!(names.contains(&"market"), "group_by should contain 'market'");
    }

    fn find_agg_node(node: &PlanNode) -> Option<&semstrait_ir::AggNode> {
        match node {
            PlanNode::Aggregate(a) => Some(a),
            PlanNode::Project(n) => find_agg_node(&n.input),
            PlanNode::Sort(n) => find_agg_node(&n.input),
            PlanNode::Fetch(n) => find_agg_node(&n.input),
            PlanNode::Filter(n) => find_agg_node(&n.input),
            _ => None,
        }
    }
}
