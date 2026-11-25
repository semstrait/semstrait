---
id: registry-functions-mapping-temporal
status: ratified
version: 1.0
impl: drift
---

# Temporal Function Mappings

This leaf specifies temporal function per-engine mappings from the 14 ratified canonical entries (date_part/year/month/day/hour/minute/second/current_date/current_timestamp/date_add/date_sub/date_diff/to_date/to_timestamp), including DF structural rewrites (year→date_part, date_add→BinaryOp) and per-engine divergence on date_diff/to_date/to_timestamp.

## 9. Temporal Functions (`14a §4.4`)

*Ratified Round-2 (2026-05-21). Canonical FunctionSpecs at `14a §4.4`. `DateTrunc` is a dedicated `Expr` variant (not a registry entry) — documented in §11. `EXTRACT(part FROM x)` is parser sugar that lowers to `date_part('part', x)` — `extract` is NOT a registry entry.*

Author-facing surface for every row below is `FunctionCall` (`14a §4.4`). Per-engine emission MAY rewrite to `BinaryOp` infix at the PlanBuilder layer (e.g. DF / Spark `date_add(d, i)` → `d + i`); that is a structural-rewrite-without-demotion, matching the `regexp_extract` precedent.

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `date_part` | `(String, Date\|Timestamp) -> Long` | `date_part(part, expr)` | Name-only | `date_part(part, expr)` | Name-only | `date_part(part, expr)` | Name-only | First arg is a part literal (`'year'`, `'month'`, `'day'`, `'hour'`, `'minute'`, `'second'`, `'millisecond'`, …). All three engines accept. SQL sugar `EXTRACT(part FROM x)` is parsed to canonical `date_part('part', x)`. |
| `year` | `(Date\|Timestamp) -> Long` | `date_part('year', expr)` (PlanBuilder rewrite) | Structural | `year(expr)` | Name-only | `year(expr)` | Name-only | DF lacks the 1-arg `year(x)` function; PlanBuilder rewrites to `date_part('year', x)`. |
| `month` | `(Date\|Timestamp) -> Long` | `date_part('month', expr)` (PlanBuilder rewrite) | Structural | `month(expr)` | Name-only | `month(expr)` | Name-only | Same DF rewrite pattern as `year`. |
| `day` | `(Date\|Timestamp) -> Long` | `date_part('day', expr)` (PlanBuilder rewrite) | Structural | `day(expr)` | Name-only | `day(expr)` / `dayofmonth(expr)` | Name-only | Same DF rewrite pattern. |
| `hour` | `(Timestamp) -> Long` | `date_part('hour', expr)` (PlanBuilder rewrite) | Structural | `hour(expr)` | Name-only | `hour(expr)` | Name-only | Same DF rewrite pattern. |
| `minute` | `(Timestamp) -> Long` | `date_part('minute', expr)` (PlanBuilder rewrite) | Structural | `minute(expr)` | Name-only | `minute(expr)` | Name-only | Same DF rewrite pattern. |
| `second` | `(Timestamp) -> Long` | `date_part('second', expr)` (PlanBuilder rewrite) | Structural | `second(expr)` | Name-only | `second(expr)` | Name-only | Same DF rewrite pattern. Integer seconds; sub-second extraction via `date_part('millisecond', …)`. |
| `current_date` | `() -> Date` | `current_date()` | Name-only | `current_date()` | Name-only | `current_date()` | Name-only | Canonical emission always uses paren form. Per-query determinism (sourced from session). |
| `current_timestamp` | `() -> Timestamp` | `current_timestamp()` | Name-only | `current_timestamp()` | Name-only | `current_timestamp()` | Name-only | Same paren-form rule. |
| `date_add` | `(Date\|Timestamp, Interval) -> same` | `BinaryOp(Add, d, i)` (PlanBuilder rewrite) | Structural | `date_add(d, i)` | Name-only | `BinaryOp(Add, d, i)` (PlanBuilder rewrite) | Structural | DF + Spark have no Interval-arg `date_add` function — `date_add(d, n)` on Spark is integer-days only. Both PlanBuilders rewrite the canonical FunctionCall to `d + i`. Spark's integer-days form is adapter-extended (`Spark.date_add(d: Date, n: Integer)`). |
| `date_sub` | `(Date\|Timestamp, Interval) -> same` | `BinaryOp(Subtract, d, i)` (PlanBuilder rewrite) | Structural | `date_sub(d, i)` | Name-only | `BinaryOp(Subtract, d, i)` (PlanBuilder rewrite) | Structural | Mirrors `date_add`. |
| `date_diff` | `(String, Date\|Timestamp, Date\|Timestamp) -> Long` | `(end - start) extracted via date_part(part, …)` (PlanBuilder structural) | Structural | `date_diff(part, start, end)` | Name-only | `'day'` part → `datediff(end, start)`; other parts → `date_part(part, end) - date_part(part, start)` (PlanBuilder structural) | Structural | Canonical = 3-arg `(part, start, end)`. Returns signed difference in `part` units (positive when `end > start`). DuckDB native; DF + Spark structural. The 2-arg integer-days form (Spark-style `datediff(end, start)`) is adapter-extended. |
| `to_date` | `(String) -> Date` | `to_date(str)` | Name-only | `Cast(str, Date)` (PlanBuilder rewrite) | Structural | `to_date(str)` | Name-only | ISO-8601 input only. DuckDB has no native `to_date()`; PlanBuilder casts string to Date. Format-string overload `(String, String)` is adapter-extended. |
| `to_timestamp` | `(String) -> Timestamp` | `to_timestamp(str)` | Name-only | `Cast(str, Timestamp)` (PlanBuilder rewrite) | Structural | `to_timestamp(str)` | Name-only | Same pattern as `to_date`. Format-string overload adapter-extended. |

Portability summary: 14 ratified canonical entries. Five DF structural rewrites for unary component shortcuts (`year`/`month`/`day`/`hour`/`minute`/`second` → `date_part`). Two structural rewrites for interval arithmetic on DF + Spark (`date_add`/`date_sub` → BinaryOp). Three-engine structural divergence on `date_diff` (DuckDB native; DF + Spark structural). DuckDB structural for `to_date`/`to_timestamp` (Cast).

### 9.1 Temporal forms explicitly excluded from canonical

| Name | Reason |
|---|---|
| `extract(part FROM x)` | Parser sugar — lowers to `date_part('part', x)` at parse time. Not a separate registry entry. |
| 2-arg `to_date(str, fmt)` / `to_timestamp(str, fmt)` | Format-string dialects diverge across engines (DF Chrono strftime, Spark Java DateTimeFormatter, DuckDB strptime syntax) and are mutually incompatible. Adapter-extended on each engine with engine-native format syntax. `TD-FUNCS-MAPPING-TO-DATE-FORMAT` / `TD-FUNCS-MAPPING-DATETIME-FORMATS`. |
| 2-arg `date_diff(start, end)` (integer-days) | Convenience form; canonical 3-arg `date_diff('day', start, end)` covers it. Adapter-extended on engines that prefer the shorter signature (Spark `datediff`). `TD-FUNCS-MAPPING-DATE-DIFF-2ARG`. |
| `date_add(d, n)` (Spark integer-days form) | Spark-specific integer-days variant — adapter-extended only. Canonical uses Interval second arg. |

---

## Testing Strategy

Temporal function behavioral gates:

- **Temporal functions round-trip**: All 14 ratified canonical entries (date_part/year/month/day/hour/minute/second/current_date/current_timestamp/date_add/date_sub/date_diff/to_date/to_timestamp) emit correctly per engine.
- **DF year/month/day/hour/minute/second structural rewrite**: DF → `date_part('year', x)` etc. (PlanBuilder layer).
- **date_add/date_sub DF/Spark BinaryOp rewrite**: `date_add(d, i)` → `d + i` (DF/Spark PlanBuilder layer); DuckDB native.
- **date_diff 3-arg structural**: DuckDB native `date_diff(part, start, end)`; DF/Spark structural (part-wise extraction via date_part).
- **to_date/to_timestamp DuckDB Cast rewrite**: DuckDB → `Cast(str, Date/Timestamp)` (PlanBuilder layer); DF/Spark native.
- **current_date/current_timestamp paren-form**: Canonical emission always uses `current_date()` (paren form, universal).
- **Excluded forms documented**: `extract` (parser sugar), 2-arg `to_date`/`to_timestamp` with format string (adapter-extended), 2-arg `date_diff` (adapter-extended), Spark integer-days `date_add` (adapter-extended).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
