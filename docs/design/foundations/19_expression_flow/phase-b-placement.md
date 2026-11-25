---
id: 19-phase-b-placement
status: ratified
version: 2.1
impl: implemented
---

# Phase B Placement

This leaf defines Phase B plan-time placement rules — filter placement (WHERE / HAVING split), group_by handoff, computed Dimension placement, Metric decomposition, the function-tag Additivity axis, and the advisory channel. Placement is keyed on ExprLayer and consumes the PhysicalExpr values produced by Phase A. See [two-phase-pipeline.md](two-phase-pipeline.md) for the phase boundary contract and [phase-a-resolution.md](phase-a-resolution.md) for compile-time resolution.

## 6. Phase B — Placement

Phase B is the contract between Phase A's output (per-`Binding` `PhysicalExpr` persisted in the `SemanticManifest`) and the planner's `Strategy` (placement of `PhysicalExpr` into the `PlanNode` tree). By the time Phase B begins, the typed semantic leaves of `[14 §3.5](../14_expressions.md)` have been eliminated — typed leaves carrying `accessor: Some(_)` have lowered to `Window`-rooted subtrees per `[14 §4.2](../14_expressions.md)`, and `Field` / `Dimension` / `Measure` / `Metric` / `Key` leaves have substituted to their resolved expressions per §3. Only `PhysicalLeaf`-shaped operands remain (modulo compile-emitted `Parameter` placeholders bound during placement). The placement rules in this section specify where each authored expression *lands* in the `PlanNode` tree — filter scope, `GROUP BY` axis, aggregate slot, output projection — together with the two-source `Additivity` axis that drives lossy-reaggregation advisories.

Phase B does two things Phase A does not:

- **`Aggregate` lift.** `Aggregate` nodes embedded in `PhysicalExpr` are extracted into `PlanNode::Aggregate` slots; the residual `PhysicalExpr` substitutes `Column` refs to the lifted slots (§7).
- **`Parameter` binding.** Compile-emitted `Parameter` leaves are substituted with concrete values from the `Request` per `[14 §5.3](../14_expressions.md)`. A `Parameter` reaching the adapter is a hard error owned by the planner.

#### 6.0 Placement is keyed on `ExprLayer`

Each persisted expression carries its applicability layer (`ManifestExpression.layer`, `33 §7.2`; `ExprLayer` at `[35 §5.5](../../apis/35_semstrait_ir.md)`), assigned at compile (`§3.2.6`). Strategy reads it directly — it does **not** re-derive placement by re-walking the tree or the semantic registry:

| `ExprLayer` | Plan-tree placement |
|---|---|
| `Scalar` | pre-aggregation `Project` (and pushdown-eligible toward `Scan`); grouping-key Dimensions/Keys materialise here and feed `GROUP BY` (§6.2). |
| `Aggregate` | the expression's `Aggregate` node is lifted into `PlanNode::Aggregate` (§7); pre-/re-aggregation across complex DataKinds is governed by its `Additivity` (§6.5). |
| `PostAggregate` | the post-aggregate residual lands in a `Project` **above** the final `Aggregate` (and above any union/join re-aggregation). Ratio Metrics and Measure-referencing Dimensions land here. |

This is the single piece of metadata that lets the planner allocate expressions optimally across the plan tree and across complex-DataKind composition without reconstructing the semantic layering that lowering would otherwise erase.

### 6.1 Filter placement

Placement is determined by **where the filter is authored**, then by **what the predicate references** for filters that admit mixed-scope predicates. References inside the authored `UnresolvedSemanticExpr` are characterised by their resolved kind — a typed `Dimension` / `Key` leaf (or a `Field` leaf whose name resolves to a Dimension / Key in the registry) is a *grouping-key reference*; a typed `Measure` / `Metric` leaf (or a `Field` leaf resolving to a Measure / Metric) is an *aggregated reference*. Phase A resolution preserves this distinction in the lowered `PhysicalExpr` shape — grouping-key references lower to `Column` leaves over the grouping axis; aggregated references lower to subtrees containing `Aggregate` nodes.

| Authoring site                       | Placement contract |
|--------------------------------------|--------------------|
| `measures.<m>.filters[].expr`        | Inlined into the aggregate via the canonical `Aggregate { filter: Some(p), ... }` form; emits `agg(expr) FILTER (WHERE p)` on engines with native `FILTER`; adapter rewrites to `agg(CASE WHEN p THEN expr END)` on engines without native `FILTER`. |
| `metrics.<m>.filters[].expr`         | Compile-split by reference kind. Dimension / Key references → pushed into every constituent's `Aggregate.filter` (per the measure rule above). Constituent Measure / Metric references → HAVING-like `Filter` node above the Metric's output `Project`. Non-constituent Measure / Metric references → compile error `EXPR_E_xxxx MetricFilterReferencesNonConstituent`. |
| `filters.<f>.expr` (DataKind-level)  | Compile-split by reference kind. Dimension / Key references → WHERE-scope (`Filter` below `Agg`). Measure / Metric references → HAVING-scope (`Filter` above `Agg`). AND-decomposable mixed-scope: split transparently. Non-AND-decomposable mixed-scope (`OR` / `NOT` across scopes): compile error `COMP_E_xxxx MixedScopeFilterUndecomposable`. |
| `keys` members                       | **Structurally rejected.** No grouping-member filter slot. |
| `dimensions.<d>.filter`              | **Structurally rejected.** Use the `filters:` block. |

**Canonical form.** `Aggregate.filter` is the canonical `PhysicalExpr` representation; adapters compensate for engines lacking native `FILTER (WHERE)` via `CASE WHEN` rewrite (`[36 §<rewrite>](../../apis/36_semstrait_adapter.md)`). The canonical IR carries the predicate in its original Boolean shape; engine compensation never leaks back into the manifest.

**Reference-kind classification.** The split between WHERE-scope and HAVING-scope is decided at compile time by inspecting the resolved kinds of the leaves reachable from the predicate. A `UnresolvedSemanticLeaf::Field` whose name is registered as a Dimension classifies the same as a `UnresolvedSemanticLeaf::Dimension` leaf at the same site — kind, not authoring vocabulary, is what placement consults.

### 6.2 `group_by` Handoff

`GROUP BY` is built from `Request` context — no `group_by:` author surface:

| Request element | Plan position |
|---|---|
| Dimensions      | `GROUP BY` key (with optional per-Dimension variation) |
| Keys            | `GROUP BY` key |
| Measures        | aggregate (no grouping) |
| Metrics         | aggregate composition (no grouping) |

**Rollup-aware Dimension carrier.** When a Dimension in the Request needs a temporal-rollup wrap (`DATE_TRUNC(grain, axis)` at the `GROUP BY` axis), the planner consumes a structured `RequestDimensionRef { name, variation }` shape per `[34 §<Request>](../../apis/34_semstrait_planner.md)`. `DimensionVariation::Temporal { grain }` on a temporal Dimension wraps with `DATE_TRUNC`; `None` is the default identity projection. Type mismatch (non-temporal Dimension with `Temporal` variation) surfaces as a `PlanErrorKind` at the planner boundary. Embedder-level surface (e.g. CLI `name.grain` tokens) is not ratified here.

Computed Dimensions whose `expr` is scalar participate in `GROUP BY` as their materialised column; variation does not apply to computed Dimensions in v1.

### 6.3 Computed Dimension placement

Inline pre-aggregation at the source projection layer. The computed Dimension's `expr` resolves at Phase A to a scalar `PhysicalExpr` over `Column` leaves; Phase B materialises it as a projected column participating in `GROUP BY` like any native-binding Dimension. Aggregate-shaped `expr` is rejected at compile per §5 (per-site shape gates). Any placement question that depends on aggregation behaviour delegates to §6.5 (function-tag axis).

### 6.4 Metric semantics

Metric `expr` references other Semantics via typed leaves (`measure(name)`, `metric(name)`, `dim(name)`, or untyped `field(name)` kind-resolved per §3.3.2.c). Metric elements do **not** bind via `semantic_mapping`; a `UnresolvedSemanticLeaf::Column` reached from a Metric `expr` is `CompileError::MetricExprBindsRawColumn`. Metric is "sugar over Measures" — every reference must traverse another declared Semantic.

Measure / Metric `(agg:, expr:)` shape is owned by `[18 §5.2](../18_entities.md)`; per-site shape gates by §5. A Measure with `agg: sum, expr: amount` resolves at Phase A to `Aggregate { op: Sum, args: [<resolved amount>], distinct: false, filter: None }`. A Metric without `agg:` remains a scalar formula over already-aggregated constituent outputs.

**Lowering.** Phase B's Metric decomposer walks the Metric `expr`, lifts every constituent Measure / Metric reference into the plan's `Aggregate` layer (dedup by canonical Semantic name; one aggregate slot per name), and emits the residual scalar `post_agg_expr` into a final `Project`. Because `expr:` syntax never contains inline aggregates (per §5's shape gates), every constituent is a named Semantic — there are no anonymous slots to name.

**`dim` / `field`-resolving-to-Dimension in Metric `expr`.** Evaluates as the **per-group value** (post-aggregate context). Compile emits an advisory listing required Dimensions; plan-time rejects requests omitting them — `PlanErrorKind::MetricRequiresDimensionInRequest { metric, missing_dimension }`.

**Metric → Metric chains.** Unbounded depth, DAG semantics. Compile DFS detects cycles → `CompileError::MetricCycle { path }`. Cycle check runs after kind resolution so `metric("a")` and `field("a")`-resolving-to-Metric-`a` are detected uniformly.

**`agg:` over Dimension- / Key-typed `expr:`.** Admitted aggregations: `min` / `max` / `count` / `count_distinct` / `first` / `last`. Authoring `sum` / `avg` / `median` / `percentile` over a Dimension / Key is rejected by `[14a §3](../14a_function_catalog.md)`'s signature lookup. Windowed access uses typed `Dimension` / `Key` leaves with `accessor: Some(…)` per `[14 §4.1](../14_expressions.md)` — symmetric with the Measure / Metric accessor surfaces.

### 6.5 Function-tag axis — `Additivity`

Placement decisions that depend on aggregation behaviour are driven by `Additivity` with a **two-source separation of concerns** — the two sources are distinct inputs to Strategy and never conflated:

- **Function-level** — `FunctionSpec.additivity` carried on the canonical function per `[14a §3.6](../14a_function_catalog.md)`. Mathematical additivity of the aggregate itself (`SUM` is `Additive`, `AVG` is `NonAdditive`). Not author-declarable in v1.
- **Model-level** — `Measure.additivity` / `Metric.additivity` per `[18 §5.2](../18_entities.md)`. Author's semantic declaration that *this specific* Measure / Metric is narrower than its underlying function (e.g. `sum` over snapshot data, semantically non-additive across snapshot points). Author-declarable in YAML.

Both enums share the unified shape `Additive | SemiAdditive { axes } | NonAdditive` ratified in `[14a §3.6](../14a_function_catalog.md)`. Phase B Strategy reads both sources, then composes the **effective additivity** by restriction:

| Function-level     | Model-level         | Effective                    |
|--------------------|---------------------|------------------------------|
| `Additive`         | unset               | `Additive`                   |
| `Additive`         | `Semi { axes }`     | `Semi { axes }`              |
| `Additive`         | `Non`               | `Non`                        |
| `Semi { axes_fn }` | `Semi { axes_md }`  | `Semi { axes_fn ∩ axes_md }` |
| `Non`              | (any)               | `Non`                        |

Rule: function-level `Non` is dominant; model-level may narrow `Additive`; `Semi`-with-`Semi` intersects the axis sets. Model cannot relax math the function disallows.

**v1 scope — function-only.** In v1 the only source is function-level additivity, accessed through the `AdditivitySource` abstraction (`[14a §3.6.2](../14a_function_catalog.md)`) with the aggregate op as the source; model-level `Measure.additivity` / `Metric.additivity` is **deferred** (the `Semi`/`Non` model rows above are the reserved extension point). Effective additivity therefore equals the function-level value, and `SemiAdditive { axes }` is not produced in v1 (it requires model-level axes). Strategy is written generically over `A: AdditivitySource`, so promoting model-level/composite additivity later is additive with no call-site change. The branching itself (`pre_aggregatable` / `reaggregation`) lives once on `Additivity` (`14a §3.6.1`).

**Strategy behaviour per effective additivity:**

| Effective              | Multi-source branch | Cross-grain JOIN / rollup | Model-level consequence (per `17` / `18`) |
|------------------------|---|---|---|
| `Additive`             | pre-aggregate per branch + re-aggregate final | safe | none |
| `SemiAdditive { axes }`| pre-aggregate per branch when `axes` preserved | advisory if `axes` crossed | may require auto-`FIRST` / `LAST` projection if request rolls up across `axes` |
| `NonAdditive`          | no pre-aggregate; single full-input aggregate | engine-defined; advisory emitted | may require temporal-shape filter to disambiguate snapshot timing |

Auto-projection and snapshot-timing-filter mechanics live in `17` / `18` / `34`; this chapter declares the composition contract and consumes the result.

### 6.6 Advisory channel

Semantic advisories use typed `Diagnostics<PlanErrorKind>` per `[30 §6](../../apis/30_api_contracts.md)`; the `tracing` channel is reserved for system-level observability and is orthogonal (configured independently by embedders).

**Emission rule.** When Strategy detects a known-lossy combination from the **effective** additivity (§6.5), emit a `PLAN_W_*` advisory and proceed (warn, not refuse). The unified cross-DataKind variant is `PlanErrorKind::LossyReaggregation { data_kind, … }` (`PLAN_W_2101`); examples include `NonAdditive Avg` under cross-grain JOIN and `SemiAdditive` axis crossed by request rollup. DataKind-specific advisories remain only when the root cause is structurally distinct and owned by that chapter (e.g. Unionset's `MissingMetadataDisjointnessProof` per `[23](../../data-kinds/23_unionset.md)`).

**Rust encoding convention.** `PLAN_W_*` / `COMP_E_*` / `EXPR_E_*` numeric codes are **spec cross-reference indices** for grep-ability; the runtime API surfaces typed enum variants only. The numeric code appears as a comment adjacent to the variant declaration, never as a runtime field. Project-wide convention codified in `[30 §6](../../apis/30_api_contracts.md)`.

---

## 7. Aggregation handling

`Expr::Aggregate { op, args, distinct, filter }` is a structural variant shared by the authoring `Expr<UnresolvedSemanticLeaf>` and the resolved `PhysicalExpr` per `[14 §3.3](../14_expressions.md)`. Phase A's substitution preserves the variant tag, translating operands structurally (§3.3.3). Phase B's Strategy lifts `Aggregate` nodes into `PlanNode::Aggregate` slots and substitutes column refs to the lifted slots in the residual `PhysicalExpr` — lift mechanics in `[34](../../apis/34_semstrait_planner.md)`, not in `resolve`.

`Aggregate` is admitted only at aggregate-admitting sites (Measure / Metric `expr:`, `filters.<f>.expr`). Outside those sites: `ValidateError::AggregateInScalarContext`.

**`Avg` posture.** Canonical `AggregationOp`, not sugar — no internal `Sum`/`Count` rewrite. Lossy combinations under cross-grain surface as `LossyReaggregation` advisories per §6.6, never refusals.

---

## Testing Strategy

- ExprLayer-keyed placement correctness: verify Scalar-layer expressions are placed in WHERE; verify Aggregate-layer expressions are lifted to `PlanNode::Aggregate`; verify PostAggregate-layer expressions are placed in HAVING / outer SELECT.
- WHERE / HAVING split: verify user filters, computed Dimension `expr:`, and Measure.filters are correctly partitioned between pre-aggregation (WHERE) and post-aggregation (HAVING) based on ExprLayer.
- Aggregate lift: verify `Aggregate` nodes are extracted from PhysicalExpr and placed in `PlanNode::Aggregate` slots; verify residual expressions reference lifted aggregates via column refs.
- Parameter binding: verify `Parameter` leaves are substituted with concrete values from the `Request`; verify unbound `Parameter` reaching the adapter produces a hard error.
- PLAN_W_2101 LossyReaggregation advisory emission: verify the advisory is emitted when a Metric is re-aggregated without explicit Additive function tag; verify it is suppressed when explicit Additive tag is present.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
