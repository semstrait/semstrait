# semstrait-planner

Semantic query planner with kind-specific planning strategies.

Builds a `LogicalPlan` from a `ResolvedQueryRequest` + `CompiledManifest` by dispatching to the appropriate kind planner, resolving additivity, injecting filters, and applying the optimizer.

---

## Planning Pipeline

The planner follows a 12-step pipeline (synchronous, not async):

```
ResolvedQueryRequest + CompiledManifest
       |
  1. ConstraintValidator::check()     pre-resolution validity gate
  2. Entity resolution                 manifest.entities[name] -> &CompiledDataKind
  3. Binding pruning                   metadata + literal filter pruning
  4. DataKind dispatch                 route to planner by variant
  5. PlannerContext                    manifest + plan_builder + catalog + session
  6. KindPlanner::resolve()           build PlanFragment
  7. AdditivityResolver               semi/non-additive measure handling
  8. Filter injection                 kind-level -> user filters
  9. ORDER BY                         SortNode from request.order_by
 10. LIMIT                            FetchNode from request.limit
 11. Build LogicalPlan                root + output_names
 12. Optimizer::apply()               identity by default (zero passes registered)
       |
       v
  LogicalPlan
```

---

## DataKind Dispatch

The planner resolves entities via `manifest.entities[name]`, which returns a `CompiledDataKind`. Dispatch is variant-based:

```
CompiledDataKind::Simple    -->  simple kind plan (single-dataset fast path)
CompiledDataKind::Grainset  -->  GrainsetPlanner::resolve()
CompiledDataKind::Unionset  -->  UnionsetPlanner::resolve()
CompiledDataKind::Joinset   -->  JoinsetPlanner::resolve()
```

All kind planners extract the variant-specific struct (`CompiledGrainsetKind`, etc.) which embeds:
- **`CompiledInterface`** -- shared semantic fields (dimensions, measures, metrics, filters, keys, domain)
- **`DatasetBinding`** -- per-dataset physical mapping (`ResolvedColumnMapping`, `resolved_sources`)
- **Acceleration indices** -- `CoverageIndex`, `DimensionIndex`, `GrainMap`, etc.

### Binding Pruning

Before dispatch, the planner narrows bindings via two pruning passes:

1. **Metadata pruning** -- if a user filter matches a metadata dimension with `Eq`, bindings whose extracted metadata value doesn't match are excluded
2. **Literal pruning** -- if a user filter matches a field with a literal column mapping value, bindings whose literal doesn't match are excluded

---

## Kind Planners

Each `CompiledDataKind` variant dispatches to a dedicated planner that builds the initial `PlanFragment`:

| Variant | Strategy | Module |
|---------|----------|--------|
| `Simple` | Single-dataset fast path (Scan -> Agg -> Project) | `data_kind/simple.rs` |
| `Grainset` | Route to cheapest covering dataset by grain | `data_kind/grainset.rs` |
| `Unionset` | UNION ALL with NULL-fill for missing columns | `data_kind/unionset.rs` |
| `Joinset` | BFS join chain from anchor dataset | `data_kind/joinset.rs` |

### Computed Dimension Handling

All kind planners partition requested dimensions into three tiers:

1. **Metadata** — `DimensionType::Metadata` — extracted from source paths/partitions (not scanned)
2. **Computed** — `dim.expr.is_some()` — derived from expressions over other columns (post-aggregation)
3. **Physical** — regular columns scanned from datasets and used in GROUP BY

Functions in `expr/mod.rs`:
- `partition_dimensions_iface()` — separates metadata from regular dims
- `split_computed_dims()` — separates computed from physical dims
- `collect_column_refs()` — extracts column references from expression trees
- `extract_metadata_value_binding()` — resolves metadata dimension values from bindings
- `resolve_native_grain_binding()` — finds best-match grain binding for temporal dims
- `grain_to_temporal()` — converts grain enum to temporal truncation expression

Expression resolution in `resolver.rs`:
- `ExprResolver` trait with `PhysicalResolver` and `MappingResolver` implementations
- Resolves semantic column names to physical, expands Guard → Case

Measure decomposition in `decomposer.rs`:
- `decompose_measure()` — declarative path (agg tag + horizontal expr)
- `decompose_metric()` — recursive metric decomposition via CompiledInterface

Computed dimension flow:
1. Expression resolved via `PhysicalResolver::new().resolve_expr()` (semantic → physical column names)
2. Physical columns referenced by the expression are collected for ScanNode
3. Computed dim NOT added to GROUP BY (AggNode groups only physical dims)
4. Computed dim emitted as ProjectNode expression (post-aggregation, alongside measure/metric aliases)

Shows the three layers of a Kind: the **interface** (`CompiledInterface` -- dimensions, measures, metrics, constraints) that users query; the **strategy** (enum variant) that determines plan structure; and the **binding** (`DatasetBinding` -- column mappings, resolved sources) that connects to physical data.

---

## Module Structure

```
src/
  lib.rs                      re-exports, public API
  planner.rs                  SemanticPlanner orchestrator
  request.rs                  ResolvedQueryRequest, QueryFilter, OrderByClause
  error.rs                    PlannerError enum

  resolver.rs                 ExprResolver trait + PhysicalResolver + MappingResolver
  decomposer.rs               DecomposedMeasure, decompose_measure, decompose_metric
  validator.rs                ConstraintValidator (pre-resolution validity gate)
  optimizer.rs                OptimizerPass trait + Optimizer
  additivity.rs               AdditivityResolver
  entity_resolver.rs          entity resolution from field names (ad-hoc queries)
  ad_hoc_join.rs              ad-hoc join resolution (FROM-less queries)
  simplify.rs                 plan simplification passes

  expr/
    mod.rs                    dimension partitioning, column ref collection, grain utils

  data_kind/
    mod.rs                    kind dispatch, PlanFragment, PlannerContext
    plan_layers.rs            shared plan-building utilities (Scan, Rename, Agg, Project)
    simple.rs                 simple kind plan (single-dataset fast path)
    grainset.rs               GrainsetPlanner (grain-aware routing + UNION ALL)
    unionset.rs               UnionsetPlanner (UNION ALL with NULL-fill)
    joinset.rs                JoinsetPlanner (BFS join chain + field resolution)

  tests/
    mod.rs
    helpers.rs                shared test fixtures (manifests, requests, builders)
    integration.rs            end-to-end planning pipeline tests
```

---

## Key Types

```rust
// The main entry point.
pub struct SemanticPlanner { .. }

impl SemanticPlanner {
    pub fn builder() -> SemanticPlannerBuilder;
    pub fn plan(&self, request: &ResolvedQueryRequest, manifest: &CompiledManifest)
        -> Result<LogicalPlan, PlannerError>;
}

// Resolved query request (produced by RequestParser).
pub struct ResolvedQueryRequest {
    pub entity_name: String,
    pub dimensions: Vec<String>,
    pub measures: Vec<String>,
    pub filters: Vec<QueryFilter>,           // user `QueryFilter` predicates (outermost FilterNodes)
    pub inline_filters: Vec<CompiledFilter>, // anonymous request-scope filters — share the kind-filter scan-layer engine
    pub order_by: Vec<OrderByClause>,
    pub limit: Option<u64>,
    pub grain: Option<String>,
    pub domain_hint: Option<String>,
    pub session_variables: SessionVariables,
}
```

---

## Filter Injection Order

Filters are layered in a specific order (inner to outer):

1. **Measure filters** -- conditional aggregation (`CASE WHEN filter THEN expr ELSE NULL END`), applied inside KindPlanner
2. **Metric filters** -- same conditional aggregation pattern, applied during expression lowering
3. **Kind-level filters** -- injected at the scan layer (before rename), apply to all queries against the kind. Drawn from `CompiledInterface.filters`.
4. **Inline request filters** -- request-scope, anonymous `{field, operator, value}` predicates carried on `ResolvedQueryRequest.inline_filters`. Injected at the **same scan-layer pass** as kind-level filters — both are `CompiledFilter`-shaped values and become indistinguishable downstream. Translated by the API layer (`semstrait-api::RequestParser`) from `RawFilter`s into canonical boolean `Expr`s. See [`docs/design/foundations/11_names_and_scopes.md §6.4.2`](../../docs/design/foundations/11_names_and_scopes.md).
5. **User filters** -- legacy `QueryFilter`s from the request (`request.filters: Vec<QueryFilter>`), outermost `FilterNode`s

---

## Dependencies

- `semstrait-core` -- `Expr`, `DataType`, `Grain`
- `semstrait-ir` -- `PlanNode`, `LogicalPlan`, `NodeMeta`
- `semstrait-manifest` -- `CompiledManifest`, `CompiledDataKind`, `CompiledInterface`, `DatasetBinding`
- `semstrait-catalog` -- `CatalogProvider` (optional, for schema checks)
