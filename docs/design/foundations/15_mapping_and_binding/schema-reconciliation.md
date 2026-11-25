---
id: 15-schema-reconciliation
status: ratified
version: 2.1
impl: implemented
---

# Schema Reconciliation

Schema reconciliation at compile time wraps implicit casts when physical column types differ from declared semantic types. This document defines widening/narrowing/incompatible cast rules (per 14 §6.4), Computed-entry type inference reconciliation, cross-source type agreement (strict), and nullability reconciliation (warning). See [binding-process](binding-process.md) for the compile-time resolution flow. <!-- cross-cut: none -->

## 9. Schema Reconciliation at Compile Time

### 9.1 Widening / narrowing / incompatible casts

When a Semantics declares `data_type: Integer` in its `SemanticInterface` and the physical `PhysicalSource.schema.columns[_].data_type` is `Long`, the compile stage reconciles the two per `14 §6.4`'s cast policy:


| Declared × Physical (per `14 §6.4` subset relevant here)             | Action                                                                                                                                                                                                                        | Diagnostic                                                                                                                                                                                 |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Same logical type                                                    | Pass-through.                                                                                                                                                                                                                 | —                                                                                                                                                                                          |
| Widening numeric (e.g. declared `Long`, physical `Integer`)          | Emit `PhysicalExpr(Cast(Column, declared_type))` on the fly at the Semantics boundary; the `SemanticMappingValue::Column` entry is rewritten to `SemanticMappingValue::Expr { expr: Cast(Column, declared) }` during compile. | `COMP_I_0301 ImplicitWideningCast` (info-level).                                                                                                                                           |
| Narrowing numeric (e.g. declared `Integer`, physical `Long`)         | Emit the same `Cast` wrapping as above.                                                                                                                                                                                       | `COMP_W_0302 ImplicitNarrowingCast` (warning-level, advises the author to double-check — narrowing a real `i64` that overflows `i32` is an engine-level runtime error, not a compile one). |
| Precision widening (Decimal → wider Decimal)                         | Emit `Cast`.                                                                                                                                                                                                                  | `COMP_I_0301`.                                                                                                                                                                             |
| Precision narrowing (Decimal → narrower Decimal)                     | Emit `Cast`.                                                                                                                                                                                                                  | `COMP_W_0302`.                                                                                                                                                                             |
| Float / Decimal cross-cast                                           | Emit `Cast`.                                                                                                                                                                                                                  | `COMP_W_0303 FloatDecimalCrossCast` — these casts can lose precision; always warn.                                                                                                         |
| `String` ↔ non-`String` (e.g. declared `Integer`, physical `String`) | No cast emitted; `CompileError::IncompatiblePhysicalType { semantics, declared, physical }` (COMP_E_0315).                                                                                                                    | Fail-fast.                                                                                                                                                                                 |
| `Date` / `Timestamp` ↔ non-temporal                                  | `CompileError::IncompatiblePhysicalType` (COMP_E_0315).                                                                                                                                                                       | Fail-fast.                                                                                                                                                                                 |
| `Binary` ↔ any non-`Binary`                                          | `CompileError::IncompatiblePhysicalType`.                                                                                                                                                                                     | Fail-fast.                                                                                                                                                                                 |


The full cast matrix lives in `14 §6.4`; §9.1 is the physical-to-semantic reconciliation slice of that matrix. `15` does not re-ratify the matrix; it sites the hook where the matrix is consulted.

**Where `Cast` lives.** After reconciliation, a `SemanticMappingValue::Column { name: "amount" }` over a physical `Long` with a declared Semantics `Integer` is rewritten to `SemanticMappingValue::Expr { expr: Cast(Column("amount"), DataType::Integer) }`. The SemanticManifest-layer `ResolvedColumnMapping.computed` stores the `Cast`-wrapped `PhysicalExpr`; no separate "cast-needed" flag exists. This keeps the planner's code path uniform (every non-literal/non-metadata Semantics is either a direct Column read or an expression evaluation).

### 9.2 `Computed`-entry type inference reconciliation

For `SemanticMappingValue::Expr { expr }`, the `expr.inferred_type()` (from `14 §5`) is compared to the declared Semantics `DataType`:


| Inferred × Declared | Action                                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Same                | Pass-through.                                                                                            |
| Widening            | Wrap: `expr = Cast(expr, declared)`. Info-level diagnostic `COMP_I_0304 ImplicitWideningCastOnComputed`. |
| Narrowing           | Wrap: `expr = Cast(expr, declared)`. Warning-level `COMP_W_0305 ImplicitNarrowingCastOnComputed`.        |
| Incompatible        | `CompileError::ComputedTypeMismatch { semantics, inferred, declared }` (COMP_E_0316).                    |


The wrapping lives on the SemanticManifest-layer `PhysicalExpr` in `ResolvedColumnMapping.computed`.

### 9.3 Cross-source type agreement

A Binding's `sources` may comprise multiple `PhysicalSource`s, each with its own `schema`. A Semantics referencing a physical column `c` through `SemanticMappingValue::Column { name: c }` requires:

- **Every source where Coverage is `Native`** must have `c` in its schema.
- The `DataType` of `c` MUST be identical across all such sources.

If a source has `c` but with a different `DataType` than another source's `c`, the Binding is rejected:

- `CompileError::CrossSourceTypeDisagreement { binding_id, column, types }` (COMP_E_0317) — `types: Vec<(usize, DataType)>` enumerates the divergent cases per-source.

This is strict intentionally: a Binding that accepts `c: Integer` in source 0 and `c: Long` in source 1 would force every plan-time scan to decide which cast to apply per source, pushing the reconciliation into the hot path. Compile-time rejection keeps the post-SemanticManifest contract "one Semantics, one type per Binding."

For `Computed` entries with a cross-source `referenced_columns` set: the same strict rule applies — every column referenced by the expression must have the same logical `DataType` across every `Derived`-covered source.

### 9.4 Nullability reconciliation

The Semantics-level nullability (from `11 §6` / `14 §5.2`) is compared against the physical source's per-column `nullable: bool`:

- Declared Non-nullable + any source reporting nullable → `COMP_W_0306 NullableSourceForNonNullableSemantics` (warning, not error — the source may still contain no nulls in practice; the runtime engine will enforce). Advises the author to either relax the declared type or add a `filter: NOT NULL` in a wrapping Measure/Dimension.
- Declared Nullable + physical non-nullable → silent; this is always safe.

**Proposed (Round 1):** the nullability mismatch is a warning, not an error. Upgrading to an error is a v2 conversation; some authors have legitimate workflows where the source-reported nullability is conservative (Parquet marking `optional` for a column that is in practice always populated). See `questions/open/15_questions.md` Q-MAP-010.

### 9.5 Reconciliation site in `compile`

The reconciliation happens in §10 step 4 (SemanticMapping completeness + reconciliation), after the sources and their schemas are in place but before the Coverage pass (§10 step 5). Steps 2–3 build the cross-source reconciled schema view; step 4 consults it per-Semantics.

## Testing Strategy

- Widening/narrowing cast: §9.1 COMP_I_0301 (widening silent), COMP_W_0302/0303 (narrowing warns)
- Computed type inference: §9.2 COMP_E_0316 (ComputedTypeMismatch) when inferred type disagrees with declared
- Cross-source type agreement: §9.3 COMP_E_0317 (CrossSourceTypeDisagreement) when physical types differ across sources
- Nullability mismatch: §9.4 COMP_W_0306 when nullable physical column binds to non-nullable semantic
- IncompatiblePhysicalType: COMP_E_0315 when cast is not in 14 §6.4 matrix
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
