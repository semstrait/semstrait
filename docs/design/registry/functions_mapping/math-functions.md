---
id: registry-functions-mapping-math
status: ratified
version: 1.0
impl: drift
---

# Math Function Mappings

This leaf specifies math function per-engine mappings from the 11 ratified canonical entries (abs/round/ceil/floor/sqrt/power/exp/ln/log/log10/sign), including structural rewrites (Spark ceil/floor Float cast wrap, DF sign→signum+cast).

## 8. Math Functions (`14a §4.3`)

*Ratified Round-2 (2026-05-21). Canonical FunctionSpecs at `14a §4.3`.*

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `abs` | `(Numeric) -> same` | `abs` | Name-only | `abs` | Name-only | `abs` | Name-only | Type-preserving across all engines. Signed-integer min overflow is engine-visible. |
| `round` | `(Float\|Decimal, [Integer]) -> same` | `round` | Name-only | `round` | Name-only | `round` | Name-only | Half-away-from-zero across all engines. Integer args rejected at compile (no native integer overload on DF). |
| `ceil` | `(Float) -> Float`, `(Decimal(p,s)) -> Decimal(p,0)` | `ceil` | Name-only | `ceil` | Name-only | `cast(ceil(x) as Double)` for Float; `ceil` for Decimal | Structural (Float) | Spark's `ceil(Float) -> BIGINT` requires cast wrap to keep canonical Float-in/Float-out shape. `ceiling` alias accepted by all engine parsers. |
| `floor` | `(Float) -> Float`, `(Decimal(p,s)) -> Decimal(p,0)` | `floor` | Name-only | `floor` | Name-only | `cast(floor(x) as Double)` for Float; `floor` for Decimal | Structural (Float) | Same shape as `ceil` mirrored. |
| `sqrt` | `(Float) -> Float` | `sqrt` | Name-only | `sqrt` | Name-only | `sqrt` | Name-only | Negative input is engine-visible (DuckDB errors, DF/Spark NaN). |
| `power` | `(Float, Float) -> Float` | `power` | Name-only | `power` | Name-only | `power` | Name-only | `pow` is adapter-side alias accepted by all parsers. |
| `exp` | `(Float) -> Float` | `exp` | Name-only | `exp` | Name-only | `exp` | Name-only | Universal. |
| `ln` | `(Float) -> Float` | `ln` | Name-only | `ln` | Name-only | `ln` | Name-only | Natural logarithm. Non-positive input engine-visible. |
| `log` | `(Float, Float) -> Float` (2-arg only) | `log` | Name-only | `log` | Name-only | `log` | Name-only | 2-arg form `log(base, value)` agreed across all three engines. **1-arg `log(x)` is NOT canonical** — DF/DuckDB = log10, Spark = ln; semantic divergence; authors use `ln(x)` or `log10(x)` explicitly. |
| `log10` | `(Float) -> Float` | `log10` | Name-only | `log10` | Name-only | `log10` | Name-only | Base-10 logarithm. |
| `sign` | `(Numeric) -> Integer` | `cast(signum(x) as Integer)` | Structural | `sign` (returns TINYINT, widens to Integer) | Name-only | `sign` | Name-only | DF only exposes `signum` (returns Float), so wrap with cast to canonical Integer. |

Portability summary: 11 ratified canonical entries. Three structural rewrites (DF `sign`→`signum`+cast; Spark `ceil(Float)`/`floor(Float)` cast wrap). Domain-error behavior on `sqrt`/`ln`/`log` for out-of-range inputs is engine-visible quirk, not gated.

### 8.1 Math functions explicitly excluded from canonical

| Name | Reason |
|---|---|
| `mod(a, b)` | Canonical form is `BinaryOpKind::Mod` (`%`) per `14 §3.3` / `14a §4.3`. The function-call form is an adapter-extended convenience on engines that prefer it (DuckDB, Spark). Legacy entry R11 superseded. |
| `trunc` / `truncate` | Legacy §12 notes universal name-only. Not in `14a §4.3` candidate list. **Canonical promotion candidate** — `TD-FUNCS-MAPPING-TRUNC-PROMOTE`. |

---

## Testing Strategy

Math function behavioral gates:

- **Math functions round-trip**: All 11 ratified canonical entries (abs/round/ceil/floor/sqrt/power/exp/ln/log/log10/sign) emit correctly per engine.
- **Spark ceil/floor Float cast wrap**: Spark `ceil(Float)` → `cast(ceil(x) as Double)` to preserve Float→Float canonical shape (Spark native `ceil(Float)`→BIGINT).
- **DF sign structural**: `sign` → `cast(signum(x) as Integer)` (DF `signum` returns Float).
- **log 2-arg form**: 2-arg `log(base, value)` universal (1-arg NOT canonical — DF/DuckDB=log10 vs Spark=ln divergence documented).
- **Negative-input behavior**: `sqrt`/`ln`/`log` negative-input behavior is engine-visible (DuckDB errors, DF/Spark NaN) — not gated.
- **abs signed-integer-min overflow**: Engine-visible behavior (no compile-time error) — not gated.
- **Excluded functions**: `mod` (BinaryOpKind, not FunctionCall) and `trunc` (canonical promotion candidate) documented.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
