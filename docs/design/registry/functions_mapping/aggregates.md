---
id: registry-functions-mapping-aggregates
status: ratified
version: 1.0
impl: drift
---

# Aggregate Function Mappings

This leaf specifies aggregate function per-engine mappings, covering the closed five (Sum/Avg/Count/Min/Max via `AggregationOp` enum) and non-closed canonical aggregates (stddev/variance/median/string_agg/percentile_cont/approx_count_distinct) with their rewrite tiers across DataFusion, DuckDB, and Spark.

## 3. Aggregates

### 3.1 The closed five (`AggregationOp` enum from `14 §3.3`)

`Sum` / `Avg` / `Count` / `Min` / `Max` are expressed in the AST via `Expr::Aggregate { op: AggregateKind, args: Vec<Self>, distinct: bool, filter: Option<Box<Self>> }`, where the closed five are carried as `AggregateKind::Builtin(AggregationOp)` (closed `AggregationOp` enum) — NOT via `FunctionCall`. Per `14a §4.7`, they are not registry entries; adapters render them via the Dialect layer's aggregate rendering path. Return types follow the SQL:2016 promotion rules per `14a §4.7` (`14 §5.4`'s pass-through posture delegates coercion to the engine).

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `sum` | `sum(expr)` | `sum(expr)` | Name-only | `sum(expr)` | Name-only | `sum(expr)` | Name-only | Universal. Closed variant. Return type per `14a §4.7`. |
| `avg` | `avg(expr)` | `avg(expr)` | Name-only | `avg(expr)` | Name-only | `avg(expr)` | Name-only | Universal. Closed variant. Return always `Double` for integer / `Double` input; widened `Decimal` for `Decimal` input. |
| `count` | `count(expr)` / `count(*)` | `count(expr)` / `count(*)` | Name-only | `count(expr)` / `count(*)` | Name-only | `count(expr)` / `count(*)` | Name-only | Universal. Returns `Long`. `count(*)` is represented in the AST as `Aggregate { op: AggregateKind::Builtin(AggregationOp::Count), args: vec![Literal(Integer(1))], distinct: false, filter: None }` per `14 §3.3` notes, rendered as `count(*)` by the adapter's aggregate-render path. |
| `count_distinct` | `count(DISTINCT expr)` | `count(DISTINCT expr)` | Name-only | `count(DISTINCT expr)` | Name-only | `count(DISTINCT expr)` | Name-only | Expressed as `Aggregate { op: AggregateKind::Builtin(AggregationOp::Count), distinct: true }` per `14 §3.3`. Universal. |
| `min` | `min(expr)` | `min(expr)` | Name-only | `min(expr)` | Name-only | `min(expr)` | Name-only | Universal. Return type = operand type. |
| `max` | `max(expr)` | `max(expr)` | Name-only | `max(expr)` | Name-only | `max(expr)` | Name-only | Universal. Return type = operand type. |

Portability summary: **Universal** across all five aggregates.

`SUM DISTINCT` / `AVG DISTINCT` render per `14 §5.4` — `Aggregate { op: AggregateKind::Builtin(AggregationOp::Sum), distinct: true }` renders as `sum(DISTINCT expr)`. All three engines support these forms natively; portability **Universal**.

### 3.2 Non-closed aggregates (`14a §4.6`)

*Ratified Round-2 (2026-05-21). Canonical FunctionSpecs at `14a §4.6`. All entries `FunctionCategory::Aggregate` + `Additivity::NonAdditive`.*

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `stddev` | `(Numeric) -> Double` | `stddev` (alias `stddev_samp`) | Name-only | `stddev` (alias `stddev_samp`) | Name-only | `stddev` (aliases `stddev_samp`, `std`) | Name-only | Bare name = sample across all three (since Spark 1.6). |
| `stddev_pop` | `(Numeric) -> Double` | `stddev_pop` | Name-only | `stddev_pop` | Name-only | `stddev_pop` | Name-only | Population variant. Universal. |
| `variance` | `(Numeric) -> Double` | `var_samp` (PlanBuilder name-remap; DF bare name is `var`) | Name-remap | `variance` (alias `var_samp`) | Name-only | `variance` (alias `var_samp`; since 1.6) | Name-only | Bare name = sample. DF historically uses `var` / `var_samp`; PlanBuilder rewrites to `var_samp` for clarity. |
| `var_pop` | `(Numeric) -> Double` | `var_pop` | Name-only | `var_pop` | Name-only | `var_pop` | Name-only | Population variant. Universal. |
| `median` | `(Numeric) -> Double`, `(Decimal(p,s)) -> Decimal(p,s)` | `median` (exact) | Name-only | `median` | Name-only | `median` (Spark 3.4+) | Name-only | Spark floor at 3.4+ — consistent with `types_mapping.md §3.2`'s 3.4 floor for `TimestampNTZType`. |
| `string_agg` | `(String, String) -> String` | `string_agg(expr, sep)` | Name-only | `string_agg(expr, sep)` (aliases `group_concat`, `listagg`) | Name-only | `string_agg(expr, sep)` (Spark 3.3+) | Name-only | Spark floor at 3.3+. ORDER BY / DISTINCT clause modifiers are adapter-extended. |
| `percentile_cont` | `(Float, Numeric) -> Double` | `percentile_cont(p) WITHIN GROUP (ORDER BY col)` | Name-only (dialect-layer) | `percentile_cont(p) WITHIN GROUP (ORDER BY col)` | Name-only (dialect-layer) | `percentile_cont(p) WITHIN GROUP (ORDER BY col)` (Spark 3.1+) | Name-only (dialect-layer) | Author-facing surface is FunctionCall `percentile_cont(fraction, col)`. Dialect layer renders SQL-standard `WITHIN GROUP (ORDER BY col)` form (same shape as `count(DISTINCT)`'s DISTINCT rendering — pure SQL syntax, not a structural Expr rewrite). |
| `approx_count_distinct` | `(Any) -> Long` | `approx_distinct` (PlanBuilder name-remap) | Name-remap | `approx_count_distinct` | Name-only | `approx_count_distinct` (since 1.6) | Name-only | DF emits `approx_distinct`. Implementation backends differ (HyperLogLog vs HyperLogLog++) — engine-delegated per `14a §6`. |

Portability summary: 8 ratified canonical entries. **Universal** (Name-only) on `stddev`, `stddev_pop`, `var_pop`, `median`, `string_agg`, `percentile_cont`, and `approx_count_distinct`-on-DuckDB/Spark. Two name-remaps on DataFusion (`variance` → `var_samp`; `approx_count_distinct` → `approx_distinct`). Spark version floors: `median` 3.4+, `string_agg` 3.3+, `percentile_cont` 3.1+ — within the existing 3.4+ floor mandated by `types_mapping.md §3.2`.

`percentile_disc` is NOT canonical — DataFusion lacks it entirely. Adapter-extended on DuckDB + Spark only (§12.4).

---

## Testing Strategy

Aggregate behavioral gates:

- **Closed-five round-trip**: Sum/Avg/Count/Min/Max emit via the Dialect aggregate path (not FunctionRegistry) across all engines. `count(*)` renders correctly (not `count(1)` in SQL).
- **DISTINCT aggregates**: `count(DISTINCT expr)` and `sum(DISTINCT expr)` render correctly on all engines (universal).
- **Non-closed aggregate per-engine tier table**: stddev/stddev_pop/variance/var_pop/median/string_agg/percentile_cont/approx_count_distinct map to correct spelling per engine.
- **DataFusion variance name-remap**: `variance` rewrites to `var_samp` (PlanBuilder layer).
- **DataFusion approx_count_distinct name-remap**: `approx_count_distinct` rewrites to `approx_distinct` (PlanBuilder layer).
- **percentile_cont WITHIN GROUP rendering**: Dialect layer emits SQL-standard `WITHIN GROUP (ORDER BY col)` clause (universal, same tier as `count(DISTINCT)` DISTINCT clause).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
