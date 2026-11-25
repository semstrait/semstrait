---
id: 36-rewrite-pipelines
status: ratified
version: 2.1
impl: in-progress
---

# Function and Type Rewrite Pipelines

This leaf owns the function-rewrite pipeline (inputs, tiers, flow, aggregate handling, registry-extension interplay) and the type-rewrite pipeline (inputs, spelling, cast semantics, type-rewrite exceptions).

## 7. Function Rewrite

### 7.1 Inputs

The function-rewrite pipeline consumes two sources of truth:

1. **`FunctionRegistry`** (`35 §8` / `14a §2`) — the canonical catalog. Every `FunctionCall.name` reaching the adapter has already been resolved against this registry at compile time per `19 §3.3`; the adapter consults the registry for return-type / signature metadata only, never to decide whether the call is valid.
2. **`registry/functions_mapping.md`** — the per-engine mapping, authoritative for rewrite-tier classification. Each `(canonical_name, adapter_id)` pair maps to exactly one tier: Name-only / Name-remap / Structural / Unsupported.

### 7.2 Rewrite tiers

Per `registry/functions_mapping/intro-taxonomy.md §1`:

| Tier | Adapter action | `DialectEmit::rewrite_function` return |
|---|---|---|
| **Name-only** | Emit canonical name verbatim | `Ok(RewrittenCall::NameOnly)` |
| **Name-remap** | Emit `new_name(args…)` with unchanged arg list | `Ok(RewrittenCall::Remapped { name })` |
| **Structural** | Emit a reshaped `PhysicalExpr` in place of the `FunctionCall` | `Ok(RewrittenCall::Structural(Box<PhysicalExpr>))` |
| **Unsupported** | Hard fail per `14a §6.3` | `Err(AdaptError::UnsupportedFeature { UnsupportedFeatureKind::Function, name, adapter, location })` with stable code `ADAPT_E_0302` (§10.1) |

### 7.3 Pipeline

At every `FunctionCall` encounter inside `emit-orchestrators.md §9`'s walk, the generic emitter calls `dialect.rewrite_function(call)` and dispatches on the returned variant. `Structural` rewrites produce a `PhysicalExpr` that re-enters the emission walk recursively — the re-entry point is `ProjectNode.projections[*].1` or `FilterNode.predicate` or wherever the original `FunctionCall` sat; the generic emitter substitutes the rewritten expression and continues. Recursion depth is bounded by the `Expr` tree depth of the original plan; structural rewrites MUST NOT introduce unbounded expansion (the registry policy prohibits rewrites that duplicate an arg more than O(1) times; a rewrite that would blow up is forbidden at registration).

### 7.4 Aggregates

The closed five (`Sum` / `Avg` / `Count` / `Min` / `Max`) are carried on `AggregateExpr` (`35 §6.7`), not on `FunctionCall`. They bypass `rewrite_function` — the generic emitter renders them directly through a dialect-specific aggregate-emission path on `DialectEmit::emit_aggregate` (signature detailed in §9.4). Non-closed aggregates (`stddev` / `variance` / `median` / `percentile_cont` / …) ARE `FunctionCall`s with `FunctionCategory::Aggregate` and flow through `rewrite_function` like scalars — the aggregate-vs-scalar distinction is at the tier table, not at the pipeline level.

### 7.5 Registry extension interplay

Adapter-extended functions (`14a §7`) live in each adapter's `RegistryExtension::FUNCTIONS` slice. At `adapt` time, if `FunctionCall.name` resolves in the registry but the target adapter's rewrite table returns no entry for it, the fallback is `Err(AdaptError::UnsupportedFeature { feature: UnsupportedFeatureKind::Function, .. })` — an adapter-extended function registered by DataFusion but called from a Spark adapter is unsupported, exactly matching `14a §6.3`.

## 8. Type Rewrite

### 8.1 Inputs

Two sources of truth, paralleling §7:

1. **Canonical `DataType`** (`13 §2` / `31 §4.1`) — the typed logical variants.
2. **`registry/types_mapping.md`** — per-engine native type spelling, per-engine cast semantics, per-engine gaps.

### 8.2 Spelling (`DialectEmit::type_name`)

Called at every `Expr::Cast { expr, target }` rendering and at projection-alias type annotations (when the adapter emits explicit column-type annotations — rare; only relevant for Spark's `DataFrameWriter`-targeted rendering, not v1 SQL).

The method per §4.4:

```rust
fn type_name(&self, dt: &DataType) -> Result<String, AdaptErrorKind>;
```

Returns:

- `Ok(String)` — the engine-native type name per `registry/types_mapping.md §1`.
- `Err(AdaptErrorKind::UnsupportedType { ty, adapter, context })` — the variant has no engine-native form (e.g. a future `DataType::Json` on ANSI adapter). Matches `14a §6.3`'s hard-error policy.

### 8.3 Cast semantics

Per `registry/types_mapping.md §2`:

- **Widening** (physical narrower → declared wider): silent cast. Emitted as `CAST(expr AS declared_type)` at the Binding-reconciliation Cast site (`14 §6.4` rule 2). The Cast originates at compile; the adapter only *renders* it.
- **Narrowing** (physical wider → declared narrower): `CompileError::PhysicalTypeNarrower` at compile time. Never reaches `36`. If a narrowing Cast somehow reaches `adapt` (bug), the adapter emits it verbatim and trusts the engine's runtime behavior; no `36`-layer safety net.
- **Precision-clamping** (timestamp precision; decimal width): per-engine policies documented in `registry/types_mapping.md §2.5` / `§3.3`. Adapters emit the engine's native spelling and emit a non-fatal `Diagnostic<AdaptErrorKind>` with kind `PrecisionClamped` (`Severity::Warning`; §10.3) if configured.

### 8.4 Type-rewrite exceptions

Per `registry/types_mapping.md §3`:

- Spark-adapter `Time(p)` → String emulation per `types_mapping.md §3.2`'s `TD-ADAPTER-SPARK-TIME`. A `DataType::Time` in a `Cast` target on Spark triggers the emulation wrapping.
- DuckDB-adapter `Timestamp(p)` precision clamp per `types_mapping.md §3.3`'s `TD-ADAPTER-DUCKDB-TIMESTAMP-NS`. `p > 6` emits a `Diagnostic<AdaptErrorKind>` with kind `PrecisionClamped` (`Severity::Warning`) and clamps.

These exception paths live in per-adapter crates; `36` ratifies only the shape (Result + `UnsupportedType` code) and the policy (hard error on genuine gap, warn + coerce on bounded truncation).

## Testing Strategy

- Function-rewrite tier dispatch (Name-only/Name-remap/Structural/Unsupported per §7.2)
- Structural recursion bounded by Expr depth (per §7.3)
- Aggregates (closed five bypass rewrite_function, non-closed flow through FunctionCall per §7.4)
- Registry-extension cross-engine unsupported (adapter X function called from adapter Y fails per §7.5)
- Type-rewrite spelling per-engine (ANSI/DataFusion/DuckDB/Spark per §8.2)
- Cast semantics (widening silent, narrowing compile-time error, precision-clamping warns per §8.3)
- Type-rewrite exceptions (Spark Time→String, DuckDB Timestamp precision clamp per §8.4)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
