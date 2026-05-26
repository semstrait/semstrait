//! Shared plan-building utilities used by grainset, joinset, and unionset planners.

use crate::error::PlannerError;
use crate::decomposer::{self, DecomposedMeasure};
use crate::resolver::{ExprResolver as _, PhysicalResolver};
use super::{extract_metadata_value_binding, extract_metadata_value_source, partition_dimensions_iface, resolve_guards, PlanFragment, PlannerContext};
use crate::request::ResolvedQueryRequest;
use semstrait_core::DataType;
use semstrait_ir::{
    Aggregation, AggregateMeasure, Expr, Field, PlanBuilder, PlanNode,
    Schema,
};
use indexmap::IndexMap;
use semstrait_manifest::{DatasetBinding, CompiledSimpleKind, CompiledInterface, MetadataDimension, ResolvedSource, TemporalGrain};
use std::collections::{HashMap, HashSet};

use super::collect_column_refs;

/// Resolve scan column type from a DatasetBinding's catalog schema.
///
/// Priority: catalog schema (physical truth) → semantic type fallback → DataType::String.
/// The `semantic_types` map provides physical_col → DataType from the kind interface,
/// used when catalog schema is unavailable (e.g., local files without a catalog provider).
pub(crate) fn resolve_scan_type_binding(
    physical_col: &str,
    binding: &DatasetBinding,
    semantic_types: &HashMap<String, DataType>,
) -> DataType {
    // 1. Catalog schema takes priority (physical truth from the source).
    if let Some(schema) = binding
        .resolved_sources
        .first()
        .and_then(|s| s.schema.as_ref())
    {
        if let Some(col) = schema.iter().find(|c| c.name == physical_col) {
            return col.data_type.clone();
        }
    }
    // 2. Semantic type from kind interface (declared in model YAML).
    if let Some(dt) = semantic_types.get(physical_col) {
        return dt.clone();
    }
    // 3. Last resort fallback.
    DataType::String
}

/// Build a map from physical column name → semantic DataType.
///
/// Inverts the column_mapping (semantic → physical) and resolves types
/// from the CompiledInterface. Used as fallback when catalog schema is unavailable.
pub(crate) fn build_semantic_type_map(
    iface: &CompiledInterface,
    physical_mapping: &IndexMap<String, String>,
) -> HashMap<String, DataType> {
    let mut map = HashMap::new();
    for (semantic_name, physical_name) in physical_mapping {
        if let Some(d) = iface.dimensions.get(semantic_name) {
            map.insert(physical_name.clone(), d.data_type.clone());
        } else if let Some(m) = iface.measures.get(semantic_name) {
            map.insert(physical_name.clone(), m.data_type.clone());
        } else if let Some(m) = iface.metrics.get(semantic_name) {
            map.insert(physical_name.clone(), m.data_type.clone());
        }
        // Join keys and other non-semantic columns are skipped —
        // their types are resolved from the join condition or scan fallback.
    }
    map
}

/// Build a scan node for a DatasetBinding (multi-source aware).
pub(crate) fn build_scan_node_binding(
    binding: &DatasetBinding,
    scan_columns: &[String],
    semantic_types: &HashMap<String, DataType>,
    pb: &dyn PlanBuilder,
) -> PlanNode {
    let scan_schema = Schema::new(
        scan_columns
            .iter()
            .map(|c| Field::new(c.clone(), resolve_scan_type_binding(c, binding, semantic_types)))
            .collect(),
    );

    if binding.resolved_sources.len() <= 1 {
        let first_source = binding.resolved_sources.first();
        let table_name = first_source
            .and_then(|s| s.table_fqn.as_deref())
            .or_else(|| first_source.map(|s| s.reference.as_str()))
            .unwrap_or(&binding.dataset_name);
        pb.build_scan(
            scan_schema,
            table_name.to_string(),
            first_source.and_then(|s| s.location.clone()),
            first_source.and_then(|s| s.format),
            scan_columns.to_vec(),
        )
    } else {
        let inputs: Vec<PlanNode> = binding.resolved_sources
            .iter()
            .map(|source| {
                let table_name = source.table_fqn.as_deref()
                    .unwrap_or(&source.reference);
                pb.build_scan(
                    scan_schema.clone(),
                    table_name.to_string(),
                    source.location.clone(),
                    source.format,
                    scan_columns.to_vec(),
                )
            })
            .collect();
        pb.build_union(scan_schema, inputs, false)
    }
}

/// Infer re-aggregation function from a CompiledInterface measure.
///
/// Re-aggregation preserves MIN/MAX (idempotent); everything else re-aggregates as SUM
/// (partial sums, partial counts). This is correct for fully-additive measures.
///
/// For non-additive measures (AVG, COUNT_DISTINCT), re-aggregation is lossy.
/// A warning is emitted; future versions will error or restructure the plan.
pub(crate) fn infer_aggregation_iface(iface: &CompiledInterface, measure_name: &str) -> Aggregation {
    if let Some(measure) = iface.measures.get(measure_name) {
        // Check additivity — warn on non-additive re-aggregation.
        if let Some(ref additivity) = measure.additivity {
            use semstrait_manifest::AdditivityType;
            match additivity {
                AdditivityType::Non => {
                    tracing::warn!(
                        "re-aggregating non-additive measure '{}' (agg: {:?}) — result may be lossy",
                        measure_name, measure.agg,
                    );
                }
                AdditivityType::Semi(_) => {
                    tracing::warn!(
                        "re-aggregating semi-additive measure '{}' — resolution strategy not yet applied",
                        measure_name,
                    );
                }
                AdditivityType::Full => {}
            }
        }
        return match measure.agg {
            Aggregation::Min => Aggregation::Min,
            Aggregation::Max => Aggregation::Max,
            _ => Aggregation::Sum,
        };
    }
    Aggregation::Sum
}

/// Coverage mode for the unified layered plan builder.
///
/// Controls the 10 delta points between Full (single-dataset) and Partial
/// (UNION branch with null-fill) code paths.
pub(crate) enum CoverageMode {
    /// All dims/measures must be available. Errors on missing.
    Full {
        handle_metrics: bool,
    },
    /// Partial coverage with null-fill for unmapped columns.
    /// Used by unionset/grainset UNION branches.
    Partial {
        covered_measures: Vec<String>,
        unified_schema: Schema,
    },
}

/// Build layered plan for a CompiledSimpleKind (single-dataset fast path).
///
/// Layered architecture: Scan → Rename → Expression → Aggregate → Project.
/// Uses `CompiledInterface` for type resolution and `DatasetBinding` for column mapping.
pub(crate) fn build_dataset_kind_plan(
    dk: &CompiledSimpleKind,
    request: &ResolvedQueryRequest,
    ctx: &PlannerContext<'_>,
) -> Result<PlanFragment, PlannerError> {
    let iface = &dk.interface;
    let binding = &dk.binding;
    let mode = CoverageMode::Full { handle_metrics: true };
    build_layered_plan(iface, binding, request, ctx, &mode, None)
}

/// Build layered plan for a single DatasetBinding with full coverage.
///
/// This is the per-binding plan builder used by grainset/joinset single-dataset paths.
/// Layered architecture: Scan → Rename → Expression → Aggregate → Project.
///
/// `temporal_rollup`: if `Some((dim_name, grain))`, applies DATE_TRUNC to the
/// temporal dimension in the GROUP BY for grain rollup.
pub(crate) fn build_binding_plan(
    iface: &CompiledInterface,
    binding: &DatasetBinding,
    request: &ResolvedQueryRequest,
    ctx: &PlannerContext<'_>,
    handle_metrics: bool,
    temporal_rollup: Option<(&str, TemporalGrain)>,
) -> Result<PlanFragment, PlannerError> {
    let mode = CoverageMode::Full { handle_metrics };
    build_layered_plan(iface, binding, request, ctx, &mode, temporal_rollup)
}

/// Build a single UNION branch for one dataset binding with partial coverage.
///
/// Layered architecture: Scan → Rename → Expression → Aggregate → Project.
/// The final projection outputs the unified schema, using NULL for unmapped
/// dimensions and uncovered measures. Entity-level filters from `iface.filters`
/// are injected into each scan node.
pub(crate) fn build_union_branch(
    iface: &CompiledInterface,
    request: &ResolvedQueryRequest,
    binding: &DatasetBinding,
    params: &UnionBranchParams<'_>,
    unified_schema: &Schema,
    ctx: &PlannerContext<'_>,
) -> Result<PlanNode, PlannerError> {
    let mode = CoverageMode::Partial {
        covered_measures: params.covered_measures.clone(),
        unified_schema: unified_schema.clone(),
    };
    let fragment = build_layered_plan(iface, binding, request, ctx, &mode, params.temporal_rollup)?;
    Ok(fragment.root)
}

/// Core layered plan builder: Scan → Rename → Expression → Aggregate → Project.
///
/// Unified function that handles both Full coverage (single-dataset, errors on
/// missing columns) and Partial coverage (UNION branch, null-fill for unmapped
/// columns). The `CoverageMode` enum controls behavior at 10 delta points.
///
/// After rename, ALL names are semantic. Aggregate GROUP BY and measure expressions
/// reference semantic column names from the rename project output.
///
/// For multi-source bindings (multiple `ResolvedSource`s), builds per-source
/// Scan→Aggregate plans with correct per-source metadata values, UNION ALLs the
/// results, re-aggregates, then applies the final projection.
///
/// Entity-level filters (`iface.filters`) are ALWAYS injected right after scan,
/// regardless of coverage mode.
///
/// `temporal_rollup`: optional `(dim_name, grain)` for DATE_TRUNC in GROUP BY.
fn build_layered_plan(
    iface: &CompiledInterface,
    binding: &DatasetBinding,
    request: &ResolvedQueryRequest,
    ctx: &PlannerContext<'_>,
    mode: &CoverageMode,
    temporal_rollup: Option<(&str, TemporalGrain)>,
) -> Result<PlanFragment, PlannerError> {
    let mapping = &binding.column_mapping;
    let pb = ctx.plan_builder;

    // ── Partition dimensions ────────────────────────────────────────
    let (metadata_dims, regular_dims) = partition_dimensions_iface(&request.dimensions, iface);
    let (physical_dims, computed_dims) = super::split_computed_dims(&regular_dims, iface);

    // All metadata dims from the interface — for SR-10 known_values in expression
    // simplification. This is broader than `metadata_dims` (user-requested only),
    // because computed dims may reference metadata dims not in the user's SELECT.
    let all_metadata_dims = super::collect_all_metadata_dims(iface);

    // ── D1: Computed dimension resolution ──────────────────────────
    // Full: all computed dims must be resolvable (error handled downstream).
    // Partial: track unresolvable computed dims in null_computed for null-fill.
    let known_values_binding = collect_known_values(binding, &all_metadata_dims);
    let mut resolvable_computed: Vec<(String, semstrait_core::Expr)> = Vec::new();
    let mut null_computed: HashSet<String> = HashSet::new();

    for (name, expr) in &computed_dims {
        match mode {
            CoverageMode::Full { .. } => {
                // Full mode: all computed dims are included (errors caught later if deps missing).
                resolvable_computed.push((name.clone(), expr.clone()));
            }
            CoverageMode::Partial { .. } => {
                // Partial mode: check if all dependencies are available in this binding.
                // Simplify first: substitute known_values + simplify prunes dead CASE
                // branches, eliminating column refs that belong to other bindings.
                // E.g., bing's market CASE drops the facebook-specific `country` ref.
                let guard_resolved = resolve_guards(expr);
                let substituted = crate::simplify::substitute(&guard_resolved, &known_values_binding);
                let simplified = crate::simplify::simplify(&substituted);

                let mut refs = Vec::new();
                let mut seen = HashSet::new();
                collect_column_refs(&simplified, &mut refs, &mut seen);
                let all_available = refs.iter().all(|r| {
                    mapping.physical.contains_key(r)
                        || mapping.literals.contains_key(r)
                        || known_values_binding.contains_key(r)
                });
                if all_available {
                    resolvable_computed.push((name.clone(), expr.clone()));
                } else {
                    null_computed.insert(name.clone());
                }
            }
        }
    }

    // ── D2: Physical dimension resolution ──────────────────────────
    // Full: error on unmapped. Partial: track in null_physical.
    let mut null_physical: HashSet<String> = HashSet::new();
    let mut scan_columns: Vec<String> = Vec::new();
    let mut scan_seen: HashSet<String> = HashSet::new();
    let mut dim_physical: Vec<(String, String)> = Vec::new();

    for dim_name in &physical_dims {
        if mapping.literals.contains_key(dim_name) {
            continue;
        } else if let Some(phys) = mapping.physical.get(dim_name) {
            dim_physical.push((dim_name.clone(), phys.clone()));
            if scan_seen.insert(phys.clone()) {
                scan_columns.push(phys.clone());
            }
        } else {
            match mode {
                CoverageMode::Full { .. } => {
                    return Err(PlannerError::DimensionNotFound {
                        kind: iface.name.clone(),
                        dimension: dim_name.clone(),
                    });
                }
                CoverageMode::Partial { .. } => {
                    null_physical.insert(dim_name.clone());
                }
            }
        }
    }

    // Computed dimension dependency columns (physical columns referenced by computed expressions).
    // Filter out phantom refs: PhysicalResolver passes through unmapped names (metadata/literal dims)
    // which are not physical columns and must not appear in scan_columns.
    {
        let physical_values: HashSet<&str> =
            mapping.physical.values().map(|v| v.as_str()).collect();
        for (_, expr) in &resolvable_computed {
            let lowered = PhysicalResolver::new(&mapping.physical).resolve_expr(expr)?;
            let mut temp = Vec::new();
            let mut temp_seen = HashSet::new();
            collect_column_refs(&lowered, &mut temp, &mut temp_seen);
            for col in temp {
                if physical_values.contains(col.as_str()) && scan_seen.insert(col.clone()) {
                    scan_columns.push(col);
                }
            }
        }
    }

    // ── D4/D9: Measure scan columns + decomposition ────────────────
    // Full: all request.measures, with handle_metrics flag.
    // Partial: only covered_measures.
    let phys_resolver = PhysicalResolver::new(&mapping.physical);
    let (scan_measure_names, handle_metrics) = match mode {
        CoverageMode::Full { handle_metrics } => (request.measures.clone(), *handle_metrics),
        CoverageMode::Partial { covered_measures, .. } => (covered_measures.clone(), true),
    };

    for measure_name in &scan_measure_names {
        // Skip literal-mapped measures: they don't reference physical columns.
        if mapping.literals.contains_key(measure_name) {
            continue;
        }
        if let Some(measure) = iface.measures.get(measure_name) {
            let lowered = phys_resolver.resolve_expr(&measure.expr)?;
            collect_column_refs(&lowered, &mut scan_columns, &mut scan_seen);
            for filter in &measure.filters {
                let lowered_filter = phys_resolver.resolve_expr(&filter.expr)?;
                collect_column_refs(&lowered_filter, &mut scan_columns, &mut scan_seen);
            }
        } else if handle_metrics {
            if let Some(metric) = iface.metrics.get(measure_name) {
                let constituents = extract_metric_constituents(metric, iface);
                for cm_name in &constituents {
                    if mapping.literals.contains_key(cm_name) {
                        continue;
                    }
                    if let Some(cm) = iface.measures.get(cm_name) {
                        let lowered = phys_resolver.resolve_expr(&cm.expr)?;
                        collect_column_refs(&lowered, &mut scan_columns, &mut scan_seen);
                        for filter in &cm.filters {
                            let lowered_filter = phys_resolver.resolve_expr(&filter.expr)?;
                            collect_column_refs(&lowered_filter, &mut scan_columns, &mut scan_seen);
                        }
                    }
                }
            } else if matches!(mode, CoverageMode::Full { .. }) {
                return Err(PlannerError::MeasureNotFound {
                    kind: iface.name.clone(),
                    measure: measure_name.clone(),
                });
            }
        } else {
            return Err(PlannerError::MeasureNotFound {
                kind: iface.name.clone(),
                measure: measure_name.clone(),
            });
        }
    }

    // ── D8: Entity-level filter columns — ALWAYS collected ─────────
    // Request inline filters (`11 §6.4.2`) ride the same scan-layer engine as
    // `iface.filters`; chain them so they contribute to scan column collection.
    for filter in iface.filters.iter().chain(request.inline_filters.iter()) {
        let lowered_filter = phys_resolver.resolve_expr(&filter.expr)?;
        collect_column_refs(&lowered_filter, &mut scan_columns, &mut scan_seen);
    }

    // ── D3: GROUP BY construction ──────────────────────────────────
    // Full: include all dims. Partial: skip null dims.
    let mut group_by: Vec<Expr> = Vec::new();
    for dim_name in &request.dimensions {
        if null_physical.contains(dim_name) || null_computed.contains(dim_name) {
            continue; // Partial mode: skip null-filled dims.
        }
        if let Some((td_name, grain)) = temporal_rollup {
            if dim_name == td_name {
                group_by.push(Expr::date_trunc(grain.into(), Expr::column(dim_name.clone())));
                continue;
            }
        }
        group_by.push(Expr::column(dim_name.clone()));
    }

    // ── D4: Measure decomposition ──────────────────────────────────
    // Full: all measures mandatory (Vec<(String, DecomposedMeasure)>).
    // Partial: only covered measures get decomposition (Vec<(String, Option<DecomposedMeasure>)>).
    let identity_physical: IndexMap<String, String> = IndexMap::new();
    let identity_resolver = PhysicalResolver::new(&identity_physical);

    // We use Option<DecomposedMeasure> uniformly — in Full mode all are Some.
    let mut lowered_measures: Vec<(String, Option<DecomposedMeasure>)> = Vec::new();

    match mode {
        CoverageMode::Full { handle_metrics } => {
            for measure_name in &request.measures {
                if let Some(measure) = iface.measures.get(measure_name) {
                    let lowered = decomposer::decompose_measure(
                        &identity_resolver,
                        measure_name,
                        measure.agg,
                        &measure.expr,
                        &measure.filters,
                        &measure.data_type,
                    )?;
                    lowered_measures.push((measure_name.clone(), Some(lowered)));
                } else if *handle_metrics {
                    if let Some(metric) = iface.metrics.get(measure_name) {
                        let lowered = decomposer::decompose_metric(
                            measure_name,
                            metric,
                            iface,
                            &identity_resolver,
                            5,
                        )?;
                        lowered_measures.push((measure_name.clone(), Some(lowered)));
                    } else {
                        return Err(PlannerError::MeasureNotFound {
                            kind: iface.name.clone(),
                            measure: measure_name.clone(),
                        });
                    }
                } else {
                    return Err(PlannerError::MeasureNotFound {
                        kind: iface.name.clone(),
                        measure: measure_name.clone(),
                    });
                }
            }
        }
        CoverageMode::Partial { covered_measures, .. } => {
            let covered_set: HashSet<&str> = covered_measures.iter().map(|s| s.as_str()).collect();
            for measure_name in &request.measures {
                if covered_set.contains(measure_name.as_str()) {
                    if let Some(measure) = iface.measures.get(measure_name) {
                        let lowered = decomposer::decompose_measure(
                            &identity_resolver,
                            measure_name,
                            measure.agg,
                            &measure.expr,
                            &measure.filters,
                            &measure.data_type,
                        )?;
                        lowered_measures.push((measure_name.clone(), Some(lowered)));
                    } else if let Some(metric) = iface.metrics.get(measure_name) {
                        let lowered = decomposer::decompose_metric(
                            measure_name, metric, iface, &identity_resolver, 4,
                        )?;
                        lowered_measures.push((measure_name.clone(), Some(lowered)));
                    } else {
                        lowered_measures.push((measure_name.clone(), None));
                    }
                } else {
                    lowered_measures.push((measure_name.clone(), None));
                }
            }
        }
    }

    // ── D4b: Collect aggregates with deduplication ──────────────────
    // Metrics and direct measures may share constituents (e.g., `clicks`
    // appears both as a direct measure and inside `ctr = clicks / impressions`).
    // After rename, both produce identical aggregates in the semantic domain.
    // Deduplicate by (function, expr_debug, distinct) to avoid phantom columns.
    let mut aggregates: Vec<AggregateMeasure> = Vec::new();
    let mut agg_seen: HashSet<String> = HashSet::new();

    for (_, lowered) in &lowered_measures {
        if let Some(l) = lowered {
            for agg_m in &l.aggregates {
                let dedup_key = format!("{:?}|{:?}|{}", agg_m.function, agg_m.expr, agg_m.distinct);
                if agg_seen.insert(dedup_key) {
                    aggregates.push(agg_m.clone());
                }
            }
        }
    }

    // ── D5: Aggregate schema ───────────────────────────────────────
    // Full: all dim fields + all measure fields.
    // Partial: non-null dim fields (from group_by) + covered measure fields.
    let mut agg_fields: Vec<Field> = match mode {
        CoverageMode::Full { .. } => {
            request
                .dimensions
                .iter()
                .map(|name| Field::new(name.clone(), iface.resolve_dim_type(name)))
                .collect()
        }
        CoverageMode::Partial { .. } => {
            group_by
                .iter()
                .filter_map(|e| match e {
                    Expr::Column(c) => Some(Field::new(c.name.clone(), iface.resolve_dim_type(&c.name))),
                    Expr::DateTrunc(dt) => {
                        if let Expr::Column(c) = dt.expr.as_ref() {
                            Some(Field::new(c.name.clone(), iface.resolve_dim_type(&c.name)))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect()
        }
    };
    // Build aggregate fields from the deduplicated vector.
    // The first aggregate for each measure uses the semantic name;
    // subsequent aggregates (from multi-constituent metrics) use __agg_N.
    let mut named_aggs: HashSet<String> = HashSet::new();
    for agg_m in &aggregates {
        // Derive the field name from the aggregate expression.
        // Simple column refs use the column name; complex exprs get __agg_N.
        let field_name = match &agg_m.expr {
            Expr::Column(col) if !named_aggs.contains(&col.name) => {
                named_aggs.insert(col.name.clone());
                col.name.clone()
            }
            _ => {
                let name = format!("__agg_{}", agg_fields.len());
                name
            }
        };
        let data_type = if let Some(m) = iface.measures.get(&field_name) {
            m.data_type.clone()
        } else {
            agg_m.data_type.clone()
        };
        agg_fields.push(Field::new(field_name, data_type));
    }
    let agg_schema = Schema::new(agg_fields);

    // ── Build plan: single-source vs multi-source ─────────────────
    let sem_types = build_semantic_type_map(iface, &mapping.physical);
    let catalog_types = build_catalog_type_map(binding);

    // ── D8: Pre-resolve entity filters — ALWAYS injected ──────────
    // `iface.filters` are compile-time DataKind filters (`18 §7.1`);
    // `request.inline_filters` are request-scope anonymous filters
    // (`11 §6.4.2`). Both share this scan-layer injection engine per
    // `19 §7.1` — they become indistinguishable downstream.
    let physical_entity_filters: Vec<Expr> = iface
        .filters
        .iter()
        .chain(request.inline_filters.iter())
        .map(|f| phys_resolver.resolve_expr(&f.expr))
        .collect::<Result<Vec<_>, _>>()?;

    let agg_output = if binding.resolved_sources.len() <= 1 {
        // ── Single-source path ────────────────────────────────────
        let known_values = collect_known_values(binding, &all_metadata_dims);
        let mut scan = build_scan_node_binding(binding, &scan_columns, &sem_types, pb);

        // D8: Inject entity-level filters right after scan (physical names).
        for predicate in &physical_entity_filters {
            let schema = (*scan.meta().output_schema).clone();
            scan = pb.build_filter(schema, scan, predicate.clone());
        }

        let rename = build_rename_project(
            scan, &request.dimensions, &dim_physical, &metadata_dims, &known_values,
            &resolvable_computed, &scan_measure_names, iface, mapping, &catalog_types, handle_metrics, pb,
        )?;
        let expr_proj = build_expression_project(&resolvable_computed, &known_values, rename, iface, pb);
        pb.build_aggregate(agg_schema, expr_proj, group_by, aggregates)
    } else {
        // ── Multi-source: per-source plans, UNION ALL, re-aggregate ──
        let mut source_plans: Vec<PlanNode> = Vec::new();
        let scan_schema = Schema::new(
            scan_columns
                .iter()
                .map(|c| Field::new(c.clone(), resolve_scan_type_binding(c, binding, &sem_types)))
                .collect(),
        );

        for source in &binding.resolved_sources {
            let known_values = collect_known_values_for_source(
                source, &mapping.literals, &all_metadata_dims,
            );

            // Scan for this specific source.
            let table_name = source.table_fqn.as_deref().unwrap_or(&source.reference);
            let mut scan = pb.build_scan(
                scan_schema.clone(),
                table_name.to_string(),
                source.location.clone(),
                source.format,
                scan_columns.to_vec(),
            );

            // D8: Inject entity-level filters right after scan (physical names).
            for predicate in &physical_entity_filters {
                let schema = (*scan.meta().output_schema).clone();
                scan = pb.build_filter(schema, scan, predicate.clone());
            }

            // Rename with per-source metadata values.
            let rename = build_rename_project(
                scan, &request.dimensions, &dim_physical, &metadata_dims, &known_values,
                &resolvable_computed, &scan_measure_names, iface, mapping, &catalog_types, handle_metrics, pb,
            )?;

            // Expression (computed dims with per-source known_values).
            let expr_proj = build_expression_project(&resolvable_computed, &known_values, rename, iface, pb);

            // Aggregate (pre-aggregate per source).
            let agg = pb.build_aggregate(
                agg_schema.clone(), expr_proj, group_by.clone(), aggregates.clone(),
            );

            source_plans.push(agg);
        }

        // UNION ALL of pre-aggregated per-source plans.
        let union = pb.build_union(agg_schema.clone(), source_plans, false);

        // ── D6: Re-aggregation GROUP BY ────────────────────────────
        // Full: all dims. Partial: only active (non-null) dims.
        let reagg_dims: Vec<String> = match mode {
            CoverageMode::Full { .. } => request.dimensions.clone(),
            CoverageMode::Partial { .. } => request
                .dimensions
                .iter()
                .filter(|d| !null_physical.contains(*d) && !null_computed.contains(*d))
                .cloned()
                .collect(),
        };

        // Skip re-aggregation when a known-value dimension in the GROUP BY
        // has distinct values per source — no rows from different sources can merge.
        let per_source_known: Vec<HashMap<String, String>> = binding
            .resolved_sources
            .iter()
            .map(|s| collect_known_values_for_source(s, &mapping.literals, &all_metadata_dims))
            .collect();
        if has_distinguishing_known_values(&per_source_known, &reagg_dims) {
            union
        } else {
            // Re-aggregate the union (merge partial aggregates).
            let reagg_group_by: Vec<Expr> = reagg_dims
                .iter()
                .map(|name| Expr::column(name.clone()))
                .collect();

            let num_dims = reagg_group_by.len();
            let reagg_aggregates: Vec<AggregateMeasure> = lowered_measures
                .iter()
                .filter_map(|(_, lowered)| lowered.as_ref())
                .flat_map(|l| &l.aggregates)
                .zip(agg_schema.fields[num_dims..].iter())
                .map(|(orig_agg, field)| {
                    let reagg_fn = match orig_agg.function {
                        Aggregation::Min => Aggregation::Min,
                        Aggregation::Max => Aggregation::Max,
                        _ => Aggregation::Sum,
                    };
                    AggregateMeasure {
                        function: reagg_fn,
                        expr: Expr::column(field.name.clone()),
                        distinct: false,
                        data_type: field.data_type.clone(),
                    }
                })
                .collect();

            let reagg_schema = agg_schema.clone();
            pb.build_aggregate(reagg_schema, union, reagg_group_by, reagg_aggregates)
        }
    };

    // ── D7/D10: Final projection ────────────────────────────────────
    // Full: Expr::column() for all, skip if identity.
    // Partial: Expr::null() for missing dims/measures, project to unified schema.
    let root = match mode {
        CoverageMode::Full { .. } => {
            let mut project_exprs: Vec<Expr> = Vec::new();
            let mut project_fields: Vec<Field> = Vec::new();

            for dim_name in &request.dimensions {
                project_exprs.push(Expr::column(dim_name.clone()));
                project_fields.push(Field::new(dim_name.clone(), iface.resolve_dim_type(dim_name)));
            }
            for (_, lowered) in &lowered_measures {
                project_exprs.push(
                    lowered.as_ref().map_or(Expr::null(), |l| l.post_agg_expr.clone()),
                );
            }
            project_fields.extend(
                lowered_measures
                    .iter()
                    .map(|(name, _)| Field::new(name.clone(), iface.resolve_measure_type(name))),
            );
            let project_schema = Schema::new(project_fields);

            let agg_schema_ref = &agg_output.meta().output_schema;
            let is_identity = is_identity_projection(&project_exprs, agg_schema_ref);
            if is_identity {
                agg_output
            } else {
                pb.build_project(project_schema, agg_output, project_exprs)
            }
        }
        CoverageMode::Partial { unified_schema, .. } => {
            let mut project_exprs: Vec<Expr> = Vec::new();

            for dim_name in &request.dimensions {
                if null_physical.contains(dim_name) || null_computed.contains(dim_name) {
                    project_exprs.push(Expr::null());
                } else {
                    project_exprs.push(Expr::column(dim_name.clone()));
                }
            }
            for (_, lowered) in &lowered_measures {
                project_exprs.push(
                    lowered.as_ref().map_or(Expr::null(), |l| l.post_agg_expr.clone()),
                );
            }

            pb.build_project(unified_schema.clone(), agg_output, project_exprs)
        }
    };

    Ok(PlanFragment { root })
}

/// Build rename project: maps physical → semantic column names.
///
/// Shared by both single-source and multi-source paths. The `known_values`
/// parameter provides per-source metadata dimension values.
///
/// Dimensions are emitted in `request_dims` order to preserve the user's
/// `--select` ordering. Internal dependency columns (computed dim deps,
/// measure source refs) follow after.
///
/// `catalog_types`: physical column name → catalog-reported DataType.
/// When a physical column's catalog type differs from the semantic type,
/// a CAST is emitted to ensure type safety.
#[allow(clippy::too_many_arguments)]
fn build_rename_project(
    scan: PlanNode,
    request_dims: &[String],
    dim_physical: &[(String, String)],
    metadata_dims: &[(String, MetadataDimension)],
    known_values: &HashMap<String, String>,
    computed_dims: &[(String, semstrait_core::Expr)],
    measure_names: &[String],
    iface: &CompiledInterface,
    mapping: &semstrait_manifest::ResolvedColumnMapping,
    catalog_types: &HashMap<String, DataType>,
    handle_metrics: bool,
    pb: &dyn PlanBuilder,
) -> Result<PlanNode, PlannerError> {
    let mut rename_exprs: Vec<Expr> = Vec::new();
    let mut rename_fields: Vec<Field> = Vec::new();

    // Helper: Column(physical) with CAST when catalog type differs from semantic type.
    let maybe_cast = |physical: &str, semantic_type: &DataType| -> Expr {
        if let Some(catalog_type) = catalog_types.get(physical) {
            if catalog_type != semantic_type {
                return Expr::cast(Expr::column(physical.to_string()), semantic_type.clone());
            }
        }
        Expr::column(physical.to_string())
    };

    // Build lookup maps from categorized data for O(1) access during request-order iteration.
    let phys_map: HashMap<&str, &str> = dim_physical.iter().map(|(s, p)| (s.as_str(), p.as_str())).collect();
    let meta_set: HashSet<&str> = metadata_dims.iter().map(|(n, _)| n.as_str()).collect();
    let computed_set: HashSet<&str> = computed_dims.iter().map(|(n, _)| n.as_str()).collect();

    // Emit dimensions in request order (preserves user's --select ordering).
    for dim_name in request_dims {
        if computed_set.contains(dim_name.as_str()) {
            // Computed dims are emitted as post-agg Expression ProjectNode,
            // not in the rename project. Skip here but their deps are added below.
            continue;
        }
        let semantic_type = iface.resolve_dim_type(dim_name);
        if let Some(&physical) = phys_map.get(dim_name.as_str()) {
            // Physical dimension: semantic := Column(physical), with optional CAST.
            rename_exprs.push(maybe_cast(physical, &semantic_type));
            rename_fields.push(Field::new(dim_name.clone(), semantic_type));
        } else if let Some(lit_val) = mapping.literals.get(dim_name) {
            // Literal dimension: semantic := typed literal.
            rename_exprs.push(typed_literal(lit_val, &semantic_type));
            rename_fields.push(Field::new(dim_name.clone(), semantic_type));
        } else if meta_set.contains(dim_name.as_str()) {
            // Metadata dimension: semantic := Literal(extracted_value).
            let value = known_values.get(dim_name).cloned().unwrap_or_default();
            rename_exprs.push(Expr::string(value));
            rename_fields.push(Field::new(dim_name.clone(), semantic_type));
        }
        // Null-physical dims (Partial mode) are not emitted here — handled by final projection.
    }

    // Computed dim dependencies: include physical columns that computed expressions reference.
    for (_, expr) in computed_dims {
        let mut sem_refs: Vec<String> = Vec::new();
        let mut sem_refs_seen: HashSet<String> = HashSet::new();
        collect_column_refs(expr, &mut sem_refs, &mut sem_refs_seen);
        for sem_ref in &sem_refs {
            if !rename_fields.iter().any(|f| f.name == *sem_ref) {
                if let Some(phys) = mapping.physical.get(sem_ref) {
                    let sem_type = resolve_semantic_type(sem_ref, iface);
                    rename_exprs.push(maybe_cast(phys, &sem_type));
                    rename_fields.push(Field::new(sem_ref.clone(), sem_type));
                }
            }
        }
    }

    // Literal measure injection: emit typed constant values before physical ref
    // collection, so the closure doesn't need to handle them.
    for measure_name in measure_names {
        if let Some(lit_val) = mapping.literals.get(measure_name) {
            if !rename_fields.iter().any(|f| f.name == *measure_name) {
                let measure_type = iface.resolve_measure_type(measure_name);
                rename_exprs.push(typed_literal(lit_val, &measure_type));
                rename_fields.push(Field::new(measure_name.clone(), measure_type));
            }
        }
    }

    // Measure source columns: map entity refs to their physical columns, with optional CAST.
    let mut add_physical_ref = |sem_ref: &str| {
        if !rename_fields.iter().any(|f| f.name == sem_ref) {
            if let Some(phys) = mapping.physical.get(sem_ref) {
                let sem_type = resolve_semantic_type(sem_ref, iface);
                rename_exprs.push(maybe_cast(phys, &sem_type));
                rename_fields.push(Field::new(sem_ref.to_string(), sem_type));
            }
        }
    };

    for measure_name in measure_names {
        // Skip literal measures — already injected above.
        if mapping.literals.contains_key(measure_name) {
            continue;
        }
        let expr = iface.measures.get(measure_name).map(|m| &m.expr);

        if let Some(expr) = expr {
            let mut sem_refs: Vec<String> = Vec::new();
            let mut sem_refs_seen: HashSet<String> = HashSet::new();
            collect_semantic_refs(expr, &mut sem_refs, &mut sem_refs_seen);
            for sem_ref in &sem_refs {
                add_physical_ref(sem_ref);
            }
            if let Some(m) = iface.measures.get(measure_name) {
                for filter in &m.filters {
                    let mut filter_refs: Vec<String> = Vec::new();
                    let mut filter_seen: HashSet<String> = HashSet::new();
                    collect_semantic_refs(&filter.expr, &mut filter_refs, &mut filter_seen);
                    for sem_ref in &filter_refs {
                        add_physical_ref(sem_ref);
                    }
                }
            }
        } else if handle_metrics {
            if let Some(metric) = iface.metrics.get(measure_name) {
                let constituents = extract_metric_constituents(metric, iface);
                for cm_name in &constituents {
                    if let Some(cm) = iface.measures.get(cm_name) {
                        let mut sem_refs: Vec<String> = Vec::new();
                        let mut sem_refs_seen: HashSet<String> = HashSet::new();
                        collect_semantic_refs(&cm.expr, &mut sem_refs, &mut sem_refs_seen);
                        for sem_ref in &sem_refs {
                            add_physical_ref(sem_ref);
                        }
                    }
                }
            }
        }
    }

    let rename_schema = Schema::new(rename_fields);
    let scan_schema = scan.meta().output_schema.as_ref();
    if is_rename_identity(&rename_exprs, &rename_schema.fields, scan_schema) {
        Ok(scan)
    } else {
        Ok(pb.build_project(rename_schema, scan, rename_exprs))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Helpers: layered plan construction (rename, expression, known values)
// ═══════════════════════════════════════════════════════════════════

/// Pre-compute compile-time-known dimension values for a binding.
///
/// Collects values from:
/// - Metadata dimensions (extracted from source paths/partitions)
/// - Literal dimensions (from column_mapping.literals)
///
/// These are the "known_values" used for SR-10 static pushdown in
/// computed expression simplification.
pub(crate) fn collect_known_values(
    binding: &DatasetBinding,
    metadata_dims: &[(String, MetadataDimension)],
) -> HashMap<String, String> {
    let mut known = HashMap::new();

    // Metadata dimensions.
    for (name, meta) in metadata_dims {
        let value = extract_metadata_value_binding(meta, binding).unwrap_or_default();
        known.insert(name.clone(), value);
    }

    // Literal dimensions.
    for (name, value) in &binding.column_mapping.literals {
        known.insert(name.clone(), value.clone());
    }

    known
}

/// Pre-compute compile-time-known dimension values for a single resolved source.
///
/// Per-source variant of `collect_known_values` — extracts metadata from the
/// specific source rather than defaulting to `binding.resolved_sources.first()`.
pub(crate) fn collect_known_values_for_source(
    source: &ResolvedSource,
    literals: &HashMap<String, String>,
    metadata_dims: &[(String, MetadataDimension)],
) -> HashMap<String, String> {
    let mut known = HashMap::new();

    for (name, meta) in metadata_dims {
        let value = extract_metadata_value_source(meta, source).unwrap_or_default();
        known.insert(name.clone(), value);
    }

    for (name, value) in literals {
        known.insert(name.clone(), value.clone());
    }

    known
}

/// Check whether any GROUP BY dimension has distinct known values across all
/// items, making re-aggregation a no-op.
///
/// Operates on pre-computed known-value maps — agnostic to value origin
/// (literals, metadata path extraction, catalog properties, etc.).
/// When a dimension like `funnel_account_id` or `dataset_name` produces
/// unique values per item, no two items can produce rows that share the
/// same GROUP BY key, so re-aggregation merges nothing and can be skipped.
///
/// Used by:
/// - Multi-source path (within a binding): per-source known values
/// - Unionset path (across bindings): per-binding known values
pub(crate) fn has_distinguishing_known_values(
    known_values_per_item: &[HashMap<String, String>],
    group_by_dims: &[String],
) -> bool {
    if known_values_per_item.len() <= 1 {
        return false;
    }
    let n = known_values_per_item.len();
    for dim_name in group_by_dims {
        let values: Vec<Option<&str>> = known_values_per_item
            .iter()
            .map(|kv| kv.get(dim_name).map(|s| s.as_str()))
            .collect();
        // All items must have a known value for this dim, and all values must be distinct.
        if values.iter().all(|v| v.is_some()) {
            let unique: HashSet<&str> = values.iter().filter_map(|v| *v).collect();
            if unique.len() == n {
                return true;
            }
        }
    }
    false
}

/// Build a map from physical column name → catalog-reported DataType.
///
/// Uses the first resolved source's schema (all sources in the same binding
/// share the same physical schema). Returns an empty map when no catalog
/// schema is available.
fn build_catalog_type_map(binding: &DatasetBinding) -> HashMap<String, DataType> {
    let mut map = HashMap::new();
    if let Some(schema) = binding
        .resolved_sources
        .first()
        .and_then(|s| s.schema.as_ref())
    {
        for col in schema {
            map.insert(col.name.clone(), col.data_type.clone());
        }
    }
    map
}

/// Check if a projection is an identity (every expr is Column(name) matching the input schema fields).
///
/// Returns `true` when the Final Projection can be skipped (identical to Aggregate output).
fn is_identity_projection(exprs: &[Expr], input_schema: &Schema) -> bool {
    if exprs.len() != input_schema.fields.len() {
        return false;
    }
    exprs
        .iter()
        .zip(input_schema.fields.iter())
        .all(|(expr, field)| matches!(expr, Expr::Column(col) if col.name == field.name))
}

/// Check if the Rename projection is an identity transformation.
///
/// Stronger than `is_identity_projection`: also verifies that output field names
/// match input field names (no physical→semantic renaming). Returns `true` when
/// every expression is `Column(col)` where `col.name` equals both the scan field
/// name and the rename output field name at that position.
fn is_rename_identity(
    rename_exprs: &[Expr],
    rename_fields: &[Field],
    scan_schema: &Schema,
) -> bool {
    if rename_exprs.len() != scan_schema.fields.len()
        || rename_fields.len() != scan_schema.fields.len()
    {
        return false;
    }
    rename_exprs
        .iter()
        .zip(rename_fields.iter())
        .zip(scan_schema.fields.iter())
        .all(|((expr, out_field), in_field)| {
            matches!(expr, Expr::Column(col) if col.name == in_field.name)
                && out_field.name == in_field.name
        })
}

/// Resolve DataType for a semantic name from CompiledInterface.
///
/// Checks dimensions, then measures, then metrics. Falls back to String.
pub(crate) fn resolve_semantic_type(name: &str, iface: &CompiledInterface) -> DataType {
    if let Some(d) = iface.dimensions.get(name) {
        return d.data_type.clone();
    }
    if let Some(m) = iface.measures.get(name) {
        return m.data_type.clone();
    }
    if let Some(m) = iface.metrics.get(name) {
        return m.data_type.clone();
    }
    DataType::String
}

/// Build a typed literal expression from a string value and target DataType.
///
/// Tries to parse the string into the appropriate native Expr constructor
/// (int, float, boolean) to avoid unnecessary `CAST('...' AS type)` in SQL.
/// Falls back to `Expr::cast(Expr::string(value), target_type)` when parsing
/// fails or the type has no natural literal form (Date, Timestamp, Binary).
///
/// For numeric types, uses `Expr::cast(Expr::int/float(...), target_type)` to
/// preserve the literal's numeric nature while ensuring exact target precision.
fn typed_literal(value: &str, target_type: &DataType) -> Expr {
    match target_type {
        DataType::Integer => {
            if let Ok(i) = value.parse::<i64>() {
                return Expr::int(i);
            }
        }
        DataType::Number => {
            if let Ok(f) = value.parse::<f64>() {
                return Expr::float(f);
            }
        }
        DataType::Decimal { .. } => {
            // Emit CAST(numeric_literal AS decimal(p,s)) — preserves numeric nature
            // while ensuring exact precision. Prefer int when possible.
            if let Ok(i) = value.parse::<i64>() {
                return Expr::cast(Expr::int(i), target_type.clone());
            }
            if let Ok(f) = value.parse::<f64>() {
                return Expr::cast(Expr::float(f), target_type.clone());
            }
        }
        DataType::Boolean => {
            if let Ok(b) = value.parse::<bool>() {
                return Expr::boolean(b);
            }
        }
        DataType::String => {
            return Expr::string(value);
        }
        _ => {}
    }
    // Fallback: string cast for types without natural literal form.
    Expr::cast(Expr::string(value), target_type.clone())
}

/// Collect semantic entity/column references from an expression tree.
///
/// Collects both `Column(name)` and `EntityRef(name)` — used to determine
/// which semantic names a measure expression depends on (for rename project).
pub(crate) fn collect_semantic_refs(
    expr: &Expr,
    refs: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    expr.walk(&mut |node| {
        let name = match node {
            Expr::Column(col) => Some(&col.name),
            Expr::EntityRef(er) => Some(&er.name),
            _ => None,
        };
        if let Some(n) = name {
            if seen.insert(n.clone()) {
                refs.push(n.clone());
            }
        }
    });
}

/// Build expression ProjectNode for computed dimensions with SR-10 simplification.
///
/// For each computed dim: `resolve_guards → substitute(known_values) → simplify`.
/// Passes through all existing columns from input.
/// Returns the input unchanged if no computed dims (skip this layer).
pub(crate) fn build_expression_project(
    computed_dims: &[(String, semstrait_core::Expr)],
    known_values: &HashMap<String, String>,
    input: PlanNode,
    iface: &CompiledInterface,
    pb: &dyn PlanBuilder,
) -> PlanNode {
    if computed_dims.is_empty() {
        return input;
    }

    let input_schema = input.meta().output_schema.clone();

    // Passthrough all existing columns.
    let mut project_exprs: Vec<Expr> = input_schema
        .fields
        .iter()
        .map(|f| Expr::column(f.name.clone()))
        .collect();
    let mut project_fields: Vec<Field> = input_schema.fields.clone();

    // Add computed dimension expressions with SR-10 simplification.
    for (dim_name, expr) in computed_dims {
        let guard_resolved = resolve_guards(expr);
        let substituted = crate::simplify::substitute(&guard_resolved, known_values);
        let simplified = crate::simplify::simplify(&substituted);

        project_exprs.push(simplified);
        project_fields.push(Field::new(
            dim_name.clone(),
            iface.resolve_dim_type(dim_name),
        ));
    }

    let schema = Schema::new(project_fields);
    pb.build_project(schema, input, project_exprs)
}

/// Validate that all UNION branches produce the same types.
///
/// Errors on type mismatch rather than falling back to a default type.
/// This ensures type consistency is enforced at plan time.
pub(crate) fn validate_union_types(branches: &[PlanNode]) -> Result<(), PlannerError> {
    if branches.len() <= 1 {
        return Ok(());
    }
    let expected = &branches[0].meta().output_schema.fields;
    for (i, branch) in branches[1..].iter().enumerate() {
        let actual = &branch.meta().output_schema.fields;
        if actual.len() != expected.len() {
            return Err(PlannerError::Internal(format!(
                "UNION branch {}: field count mismatch ({} vs {})",
                i + 1,
                actual.len(),
                expected.len()
            )));
        }
        for (exp, act) in expected.iter().zip(actual.iter()) {
            if exp.data_type != act.data_type {
                return Err(PlannerError::Internal(format!(
                    "UNION branch {}, column '{}': type mismatch ({:?} vs {:?})",
                    i + 1,
                    exp.name,
                    exp.data_type,
                    act.data_type
                )));
            }
        }
    }
    Ok(())
}

/// Build the unified output schema for a UNION plan.
///
/// Produces dimension fields + measure fields from the request and kind interface,
/// used as the target schema for UNION branches and re-aggregation.
pub(crate) fn build_unified_schema(request: &ResolvedQueryRequest, iface: &CompiledInterface) -> Schema {
    let fields: Vec<Field> = request
        .dimensions
        .iter()
        .map(|name| Field::new(name.clone(), iface.resolve_dim_type(name)))
        .chain(
            request
                .measures
                .iter()
                .map(|name| Field::new(name.clone(), iface.resolve_measure_type(name))),
        )
        .collect();
    Schema::new(fields)
}

/// Recursively expands nested metric references to their underlying measures.
/// For example, `roi = profit / cost` where `profit = revenue - cost` returns
/// `["revenue", "cost"]` — the actual physical measures, not intermediate metrics.
pub(crate) fn extract_metric_constituents(
    metric: &semstrait_manifest::CompiledMetric,
    iface: &CompiledInterface,
) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = HashSet::new();
    collect_leaf_measures(&metric.expr, iface, &mut names, &mut seen);
    names
}

/// Collect transitive leaf measure names from an expression tree.
fn collect_leaf_measures(
    expr: &Expr,
    iface: &CompiledInterface,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    match expr {
        Expr::Column(col) => {
            resolve_leaf_or_expand(&col.name, iface, out, seen);
        }
        Expr::EntityRef(er) => {
            resolve_leaf_or_expand(&er.name, iface, out, seen);
        }
        Expr::BinaryOp(bin) => {
            collect_leaf_measures(&bin.left, iface, out, seen);
            collect_leaf_measures(&bin.right, iface, out, seen);
        }
        Expr::Case(case) => {
            for wc in &case.when_then {
                collect_leaf_measures(&wc.condition, iface, out, seen);
                collect_leaf_measures(&wc.result, iface, out, seen);
            }
            if let Some(e) = &case.else_expr {
                collect_leaf_measures(e, iface, out, seen);
            }
        }
        _ => {}
    }
}

/// If `name` is a nested metric, recursively expand; otherwise keep as leaf measure.
fn resolve_leaf_or_expand(
    name: &str,
    iface: &CompiledInterface,
    out: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    if let Some(sub_metric) = iface.metrics.get(name) {
        if seen.insert(format!("__metric__{}", name)) {
            collect_leaf_measures(&sub_metric.expr, iface, out, seen);
        }
    } else if seen.insert(name.to_string()) {
        out.push(name.to_string());
    }
}

/// Parameters for building a single UNION branch.
///
/// Callers determine which measures are covered; any measure not in
/// `covered_measures` is null-filled in the final projection.
pub(crate) struct UnionBranchParams<'a> {
    /// Measures/metrics covered by this binding (aggregated normally).
    pub covered_measures: Vec<String>,
    /// Optional temporal rollup: (dim_name, grain) for DATE_TRUNC in GROUP BY.
    pub temporal_rollup: Option<(&'a str, TemporalGrain)>,
}
