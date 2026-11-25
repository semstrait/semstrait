---
id: 14-parameter
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: semstrait-ir owns Parameter + ParameterKey; compile-emitted only; plan-time binding per 34
---

# Parameter Placeholder

This leaf defines the `Parameter` compile-emitted placeholder, the closed `ParameterKey` set, and the non-coercion posture at the canonical layer. `Parameter` is never authored — it is compile-emitted only (primarily for sugar-accessor lowering) and plan-bound. The non-coercion posture keeps semstrait engine-neutral: no implicit type promotion or cross-operand compatibility checks. See [physical-leaves.md](physical-leaves.md) for leaf-set placement and [crate-placement.md](crate-placement.md) for ownership.

<!-- cross-cut: none -->

## 5. Parameter — Compile-Emitted Placeholder

### 5.1 Shape

```rust
pub struct Parameter {
    pub key: ParameterKey,
    pub data_type: DataType,
}
```

`Parameter` carries a typed key (not a stringly identifier) and a mandatory `data_type` at compile-emit time. The `data_type` lets downstream stages reason about the placeholder's eventual concrete shape without re-deriving it.

### 5.2 `ParameterKey` — closed set

```rust
#[non_exhaustive]
pub enum ParameterKey {
    RequestDimensionsMinusTemporal,
    RequestTemporalAxis,
}
```

The closed parameter set is **internal** — adding members is additive per I10 and is not author-extensible. v1 carries exactly the two keys needed by Family-B-sugar elimination (§4.2); future keys land via `#[non_exhaustive]` additions.

### 5.3 Plan-time binding postcondition

Per the canonical pipeline (`[00 §5](../../00_overview.md)`), the planner substitutes `Parameter` leaves against the `Request` during plan construction. The postcondition is that **no `Parameter` survives into adapt-time**: a `Parameter` reaching an adapter is a hard error owned by the planner (`PlanErrorKind`), not the adapter.

The planner-level binding mechanics and the exact `Request` shape that supplies the substitution values live in `[19](../19_expression_flow.md)` and `[34 / planner contract](../../apis/34_semstrait_planner.md)`. This chapter ratifies the placeholder shape and the postcondition only.

### 5.4 Non-coercion posture (pass-through)

semstrait performs **no implicit type coercion or promotion** at the canonical layer:

- **Function calls** — signature matching requires exact `DataType` equality per-argument per `[14a §3.3](../14a_function_catalog.md)`. No `Integer → Long` widening, no `String → Date` parsing. On mismatch: `ir::CompileError::NoMatchingSignature` (`[14a §8](../14a_function_catalog.md)`), raised downstream in the manifest compile pipeline. Authors insert explicit `Cast` when types differ.
- **Binary operators** — arithmetic return type is `SameAs(left_operand)`. Cross-operand type compatibility (`Integer < String`, `Double + Date`) is **not validated at the semstrait layer** — the engine raises its own diagnostics at execution time. This keeps semstrait's compile deterministic and engine-neutral.
- **Comparison operators** — produce `Boolean` regardless of operand types. No canonical comparison-compatibility matrix in v1.
- **NULL handling** — NULL-in / NULL-out behaviour is engine-delegated. `FunctionSpec` carries no `null_handling` field; the canonical IR expresses structural null-test shapes (`IsNull`, `Coalesce`, `NullIf`) but does not model per-function null-propagation semantics.
- **Join-key compatibility** — deferred to the engine per `[13](../13_types_and_grain.md)`.

This posture keeps the canonical layer thin and engine-neutral: semstrait compiles structure, not execution semantics. Engine-specific type compatibility (DataFusion's arrow promotion, DuckDB's implicit casts, Spark's ANSI-strict mode) is the adapter's concern.

## Testing Strategy

- **Compile emission:** `Parameter` is never authored (parse rejects `Parameter` leaf construction). Compile emits `Parameter` only for sugar-accessor lowering (`partition_by` / `order_by` per §4.2).
- **Planner binding:** Every `Parameter` leaf in the compile output is successfully substituted with a concrete value by the planner. `ParameterKey → value` binding per `[34](../../apis/34_semstrait_planner.md)`.
- **Adapt-time postcondition:** No `Parameter` survives into adapt-time. Adapters error if they encounter a `Parameter` leaf.
- **Non-coercion enforcement:** Function-call signature matching enforces exact `DataType` equality per-argument (no implicit widening). Binary operators do not validate cross-operand compatibility; engine-side diagnostics surface at execution.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
