---
id: 19-naming-and-scope
status: ratified
version: 2.1
impl: implemented
---

# Naming and Scope

This navigational leaf documents naming conventions and v1 scope boundaries for the expression pipeline.

## 9. Naming Discipline

- New enum coinages avoid the `Kind` suffix when a lighter spelling works (e.g. `Accessor`, not `AccessorKind`; `Parameter`, not `ParameterKey`).
- Established `*Kind` names (`DataKind`, `CompositionKind`, `*ErrorKind`) stay — `*ErrorKind` follows the `std::io::ErrorKind` convention.
- `FunctionCategory` (existing in `14a §3.2`) is the canonical category-axis name; reuse, do not parallel-coin.

---

## 10. Out of Scope for v1

- Runtime `evaluate(&RecordBatch) -> ColumnarValue` expression machinery (DataFusion-style) — `semstrait` is compile-time, not runtime.
- Subquery / Lambda / MaskExpression expression forms — correlated needs ride per-kind typed semantic leaves + `Relationship` (`16 §2`).
- Stringly-typed parameter IDs (`"$1"` style) — superseded by typed `ParameterKey` (`14 §5.2`).
- Substrait (or other canonical-consumer) wire-portable plan emission — architecturally reserved as a capability-driven adapter path owned by `30` / `36` (`[TD-30-ADAPTER-CAPABILITY]`); no concrete canonical-consumer adapter ships in v1.
- UDF surface for author-declarable function-level `Additivity` — function-level `Additivity` is hardcoded in `14a §3.6` per built-in aggregate in v1.
- Per-DataKind advisory specialisation beyond `PLAN_W_2101 LossyReaggregation` — see `[TD-19-ADVISORY-SPECIALISATION]`.

---

## 11. Cross-References

Upstream:

- `[00_overview.md](../../00_overview.md)` — canonical pipeline (§5), design invariants I1–I12 (§9).
- `[10_resolution_pipeline.md](../10_resolution_pipeline.md)` — stage contracts, `Diagnostic` stream.
- `[11_names_and_scopes.md](../11_names_and_scopes.md)` — name resolution algorithm, Tier-1 / Tier-2 occurrence merge, shape inference.
- `[13_types_and_grain.md](../13_types_and_grain.md)` — canonical `DataType` set, widening / narrowing lattice.
- `[14_expressions.md](../14_expressions.md)` — `Expr<L>` AST, leaf sets, per-kind typed `UnresolvedSemanticLeaf`, sugar accessors, `Parameter`, the authoring YAML surface (`14 §6`), per-site shape gates, crate placement.
- `[14a_function_catalog.md](../14a_function_catalog.md)` — `CanonicalFn`, `FunctionRegistry`, `FunctionSpec.additivity`, signature polymorphism, return-type rules, function-resolution error variants.
- `[15_mapping_and_binding.md](../15_mapping_and_binding.md)` — `SemanticMapping` and the `Binding` process consumed at compile.
- `[16_composition.md](../16_composition.md)` — `Relationship` graph; plan-time join-subgraph materialization consuming `PathSignature`.
- `[17_temporal_shape.md](../17_temporal_shape.md)` — temporal-axis semantics consumed by `Parameter::RequestTemporalAxis` and `DimensionAxis::Temporal`.
- `[18_entities.md](../18_entities.md)` — canonical entity types, `Measure` / `Metric` `(agg:, expr:)` pairing, `SemanticMapping` value shape.

Downstream:

- `[../data-kinds/21_dataset.md](../../data-kinds/21_dataset.md)`, `[../data-kinds/22_grainset.md](../../data-kinds/22_grainset.md)`, `[../data-kinds/23_unionset.md](../../data-kinds/23_unionset.md)`, `[../data-kinds/24_joinset.md](../../data-kinds/24_joinset.md)` — consume per-site `expr:` shape rules and advisory cross-refs from §6.6.
- `[../apis/30_api_contracts.md](../../apis/30_api_contracts.md)` — codifies the Rust encoding convention for numeric diagnostic codes (§6.6); owns the `Adapter` capability surface that drives canonical-plan-to-canonical (Path A) vs canonical-plan-to-engine (Path B) dispatch (`[TD-30-ADAPTER-CAPABILITY]`).
- `[../apis/33_semstrait_manifest.md](../../apis/33_semstrait_manifest.md)` — on-disk serialization shape of `ResolvedExprTable` (§3.2.4).
- `[../apis/34_semstrait_planner.md](../../apis/34_semstrait_planner.md)` — `Strategy` consumes `PhysicalExpr`; performs `Aggregate` lift, `Parameter` binding, and `PLAN_W_*` emission per §2.1 and §6.6.
- `[../apis/36_semstrait_adapter.md](../../apis/36_semstrait_adapter.md)` — engine-specific `FILTER (WHERE)` ↔ `CASE WHEN` rewrites, function mapping, and adapter mechanics for both dispatch paths (`[TD-30-ADAPTER-CAPABILITY]`).
- `[../apis/38_semstrait_api.md](../../apis/38_semstrait_api.md)` — orchestration/API layer context; CLI syntax remains embedder-owned (§6.2).

## Testing Strategy

- Navigational leaf — covered by doc-space-check structural gates.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
