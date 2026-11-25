---
id: registry-functions-mapping-summary
status: ratified
version: 1.0
impl: drift
---

# Per-Adapter Rewrite Summary, Coverage Gaps, and Metadata

This leaf consolidates per-adapter rewrite checklists (DataFusion/DuckDB/Spark/ANSI layering digest), per-engine coverage-gap catalog (TD entries for failed Q10 intersection: initcap, percentile_disc, 2-arg to_date/to_timestamp, if/ifnull/nvl), adapter-extended function inventory seed lists, cast-semantics edge cases (Integer÷Integer, Decimal÷Decimal, Date-Date result-type divergence, count/sum return width), versioning pins, TECH_DEBT index, and interaction cross-references.

## 12. Adapter-Extended Functions (`14a §7`)

Engine-specific functions NOT in the canonical catalog, registered via `RegistryExtension` impls per `14a §7.1`. Listed here for cross-reference; the authoritative inventory lives in each adapter crate. Every entry 🟡 pending `TD-FUNCS-MAPPING-ADAPTER-INVENTORY` (Q-FUNCS-MAP-018).

### 12.1 DataFusion-extended

| Name | Signature | Purpose | Source |
|---|---|---|---|
| `array_element` | `array_element(array, idx) -> element_type` | 1-indexed array-element access. Used internally by the `RegexpExtract` structural rewrite (§11). | DataFusion built-in. |
| `regexp_match` | `regexp_match(str, pattern) -> List<Utf8>` | Returns array of all capture groups. Not canonical because the `List` return type is engine-specific; canonical `regexp_match` returns `Boolean`. | DataFusion built-in. |
| `array_to_string` | `array_to_string(array, sep) -> String` | Array join. | DataFusion built-in. |
| `sha2` / `sha224` / `sha384` / `sha512` | `sha_(str) -> String` | Cryptographic hashes. Not canonical v1. | DataFusion built-in. |
| `nanvl` | `nanvl(x, y) -> same` 🟡 | NaN-safe `ifnull` for floats. | DataFusion built-in. |

### 12.2 DuckDB-extended

| Name | Signature | Purpose | Source |
|---|---|---|---|
| `list_extract` | `list_extract(list, idx) -> element_type` | 1-indexed list access. DuckDB idiom for array element lookup. | DuckDB built-in. |
| `regexp_extract_all` | `regexp_extract_all(str, pattern) -> List<String>` | All matches, not just first. | DuckDB built-in. |
| `list_append` | `list_append(list, elem) -> list` | Immutable list append. | DuckDB built-in. |
| `strptime` | `strptime(str, fmt) -> Timestamp` | Custom-format timestamp parse. See Q-FUNCS-MAP-011. | DuckDB built-in. |
| `epoch_ms` | `epoch_ms(ts) -> Long` 🟡 | Timestamp → epoch milliseconds. | DuckDB built-in. |
| `percentile_disc` | `percentile_disc(p) WITHIN GROUP (ORDER BY col) -> same` | Discrete percentile (returns actual data point). DataFusion lacks an equivalent — adapter-extended only. | DuckDB built-in. |

### 12.3 Spark-extended

| Name | Signature | Purpose | Source |
|---|---|---|---|
| `collect_set` | `collect_set(expr) -> Array<T>` | Aggregate to distinct array. `FunctionCategory::Aggregate`. | Spark built-in. |
| `collect_list` | `collect_list(expr) -> Array<T>` | Aggregate to array (preserving duplicates). `FunctionCategory::Aggregate`. | Spark built-in. |
| `array_join` | `array_join(array, sep, [null_replacement]) -> String` | Array join. | Spark built-in. |
| `percentile_approx` | `percentile_approx(expr, frac) -> same` | Approximate percentile. | Spark built-in. |
| `percentile_disc` | `percentile_disc(p) WITHIN GROUP (ORDER BY col) -> same` (Spark 3.1+) | Discrete percentile. DataFusion lacks an equivalent — adapter-extended only. | Spark built-in. |
| `try_divide` | `try_divide(a, b) -> Double` (Spark 3.3+) 🟡 | NULL-on-zero-divisor safe division. See Q-FUNCS-MAP-017. | Spark built-in. |
| `pmod` | `pmod(a, b) -> same` | Positive modulo. | Spark built-in. |
| `startswith` / `endswith` | Spark native names for `starts_with` / `ends_with`. | — | Spark built-in; adapter name-remaps to canonical if promoted (§7.1). |

### 12.4 Cross-engine adapter-extended (legacy demotions)

These were canonical in legacy `FUNCTION_CATALOG.md` but failed `14a` Q10 intersection; they remain usable under per-adapter registration but are NOT canonical.

| Name | Rationale for demotion | Per-adapter disposition | TD |
|---|---|---|---|
| `initcap` | No DuckDB native. | DataFusion + Spark adapter-extended. Author using DuckDB must emulate or omit. | `TD-FUNCS-MAPPING-INITCAP` |
| `percentile_disc` | DataFusion has no equivalent. | DuckDB + Spark adapter-extended (Spark 3.1+). DF authors use `percentile_cont` or `approx_percentile_cont`. | `TD-FUNCS-MAPPING-PERCENTILE` |
| 2-arg `to_date` / `to_timestamp` | DuckDB's `strptime` reverses arg order. | DataFusion + Spark adapter-extended as 2-arg; DuckDB adapter-extended as `strptime` (separate canonical name). | `TD-FUNCS-MAPPING-TO-DATE-FORMAT` |
| `if(cond, then, else)` | Not universal (Spark-only native). | Spark adapter-extended; canonical authors use `Expr::Case`. | `TD-FUNCS-MAPPING-IF-IFNULL-NVL` |
| `ifnull` / `nvl` | Overlap with `coalesce`; DF lacks `nvl`. | Per-adapter aliases; canonical authors use `Expr::Coalesce`. | Same TD. |

---

<!-- cross-cut: none -->
<!-- waiver rationale: §13 per-adapter Expr-variant→SQL digest is self-owned registry mapping content, not a foundations/14 roster constraint -->

## 13. Per-Adapter Rewrite Summary

Cross-references every rewrite asserted in §§3–11 into a per-adapter layering digest. Useful as a checklist during adapter implementation and as a lookup-by-engine when investigating divergent emission. `PB` = PlanBuilder layer (`EngineAdapter::rewrite_expr`); `DL` = Dialect layer (`SqlDialect` trait method).

### 13.1 DataFusion

**PlanBuilder-layer rewrites** (canonical `Expr` → engine-oriented `Expr`):

| Source | Target | Tier | §ref |
|---|---|---|---|
| `FunctionCall("trim", [s, ...])` | `FunctionCall("btrim", [s, ...])` | Name-remap | §7 |
| `FunctionCall("length", [arr])` (Array arg) | `FunctionCall("array_length", [arr])` | Name-remap | §7 |
| `FunctionCall("reverse", [arr])` (Array arg) | `FunctionCall("array_reverse", [arr])` | Name-remap | §7 |
| `FunctionCall("concat", [a, b, c, ...])` | `BinaryOp(Concat, ... a || b || c ...)` | Structural | §7 |
| `FunctionCall("position", [s, t])` | `FunctionCall("strpos", [t, s])` | Name-remap + arg-reorder | §7 |
| `FunctionCall("sign", [x])` | `Cast(FunctionCall("signum", [x]), Integer)` | Structural (name-remap + cast to canonical Integer) | §8 |
| `FunctionCall("variance", [x])` | `FunctionCall("var_samp", [x])` | Name-remap | §3.2 |
| `FunctionCall("approx_count_distinct", [x])` | `FunctionCall("approx_distinct", [x])` | Name-remap | §3.2 |
| `FunctionCall("year", [x])` | `FunctionCall("date_part", [Literal("year"), x])` | Structural | §9 |
| `FunctionCall("month", [x])` | `FunctionCall("date_part", [Literal("month"), x])` | Structural | §9 |
| `FunctionCall("day", [x])` | `FunctionCall("date_part", [Literal("day"), x])` | Structural | §9 |
| `FunctionCall("hour", [x])` | `FunctionCall("date_part", [Literal("hour"), x])` | Structural | §9 |
| `FunctionCall("minute", [x])` | `FunctionCall("date_part", [Literal("minute"), x])` | Structural | §9 |
| `FunctionCall("second", [x])` | `FunctionCall("date_part", [Literal("second"), x])` | Structural | §9 |
| `FunctionCall("date_add", [d, i])` | `BinaryOp(Add, d, i)` | Structural | §9 |
| `FunctionCall("date_sub", [d, i])` | `BinaryOp(Subtract, d, i)` | Structural | §9 |
| `FunctionCall("date_diff", [Literal(part), start, end])` | `Cast(FunctionCall("date_part", [Literal(part), BinaryOp(Subtract, end, start)]), Long)` (`'day'` part) / part-wise extraction for non-day parts | Structural | §9 |
| `Expr::RegexpExtract { expr, pattern, group }` | `FunctionCall("array_element", [FunctionCall("regexp_match", [expr, pattern]), group + 1])` | Structural | §11 |

**Dialect-layer rendering** (dedicated variants → SQL):

| Variant | SQL | Method |
|---|---|---|
| `Expr::Like` | `expr LIKE pattern` | default `binary_op` path |
| `Expr::ILike` | `expr ILIKE pattern` | `ilike` |
| `Expr::RegexpMatch` | `regexp_like(expr, pattern)` (partial) / `regexp_like(expr, CONCAT('^', pattern, '$'))` (full) | `regexp_match` |
| `Expr::RegexpExtract` | *(rewritten to FunctionCall at PlanBuilder; rendered via generic call path)* | — |
| `Expr::DateTrunc` | `date_trunc('grain', expr)` | `date_trunc` |
| `Expr::Cast` | `CAST(expr AS <type>)` | `type_name` (per `types_mapping §1`) |
| `Aggregate { distinct: true }` | `fn(DISTINCT expr)` | aggregate-render path |
| `FunctionCall("percentile_cont", [p, col])` | `percentile_cont(p) WITHIN GROUP (ORDER BY col)` | aggregate-render path (`WITHIN GROUP` form) |

### 13.2 DuckDB

**PlanBuilder-layer rewrites:**

| Source | Target | Tier | §ref |
|---|---|---|---|
| `FunctionCall("concat", [a, b, c, ...])` | `BinaryOp(Concat, ... a || b || c ...)` | Structural | §7 |
| `FunctionCall("lpad", [s, n])` (2-arg form) | `FunctionCall("lpad", [s, n, Literal(" ")])` | Structural | §7 |
| `FunctionCall("rpad", [s, n])` (2-arg form) | `FunctionCall("rpad", [s, n, Literal(" ")])` | Structural | §7 |
| `FunctionCall("position", [s, t])` | `FunctionCall("strpos", [t, s])` | Name-remap + arg-reorder | §7 |
| `FunctionCall("to_date", [s])` | `Cast(s, Date)` | Structural | §9 |
| `FunctionCall("to_timestamp", [s])` | `Cast(s, Timestamp)` | Structural | §9 |
| `FunctionCall("log", [base, x])` | *(left as-is if DuckDB supports 2-arg form; else arity mismatch)* 🟡 | — | §8 |

**Dialect-layer rendering:**

| Variant | SQL | Method |
|---|---|---|
| `Expr::Like` | `expr LIKE pattern` | default |
| `Expr::ILike` | `expr ILIKE pattern` | `ilike` |
| `Expr::RegexpMatch` | `regexp_matches(expr, pattern)` (partial) / `regexp_matches(expr, CONCAT('^', pattern, '$'))` (full) | `regexp_match` |
| `Expr::RegexpExtract` | `regexp_extract(expr, pattern, group)` | `regexp_extract` |
| `Expr::DateTrunc` | `date_trunc('grain', expr)` | `date_trunc` |
| `Expr::Cast` | `CAST(expr AS <type>)` | `type_name` |
| `FunctionCall("percentile_cont", [p, col])` | `percentile_cont(p) WITHIN GROUP (ORDER BY col)` | aggregate-render path |

### 13.3 Spark

**PlanBuilder-layer rewrites:**

| Source | Target | Tier | §ref |
|---|---|---|---|
| `FunctionCall("length", [arr])` (Array arg) | `FunctionCall("size", [arr])` | Name-remap | §7 |
| `FunctionCall("lpad", [s, n])` (2-arg form) | `FunctionCall("lpad", [s, n, Literal(" ")])` | Structural | §7 |
| `FunctionCall("rpad", [s, n])` (2-arg form) | `FunctionCall("rpad", [s, n, Literal(" ")])` | Structural | §7 |
| `FunctionCall("position", [s, t])` | `FunctionCall("locate", [s, t])` 🟡 | Name-remap (arg order matches canonical) | §7 |
| `FunctionCall("starts_with", [s, p])` | `FunctionCall("startswith", [s, p])` 🟡 | Name-remap (underscore strip) | §7.1 |
| `FunctionCall("ends_with", [s, p])` | `FunctionCall("endswith", [s, p])` 🟡 | Name-remap | §7.1 |
| `FunctionCall("ceil", [x])` (Float arg) | `Cast(FunctionCall("ceil", [x]), Double)` | Structural (cast wrap — Spark's `ceil(Float)` returns BIGINT) | §8 |
| `FunctionCall("floor", [x])` (Float arg) | `Cast(FunctionCall("floor", [x]), Double)` | Structural (cast wrap) | §8 |
| `FunctionCall("date_add", [d, i])` | `BinaryOp(Add, d, i)` | Structural | §9 |
| `FunctionCall("date_sub", [d, i])` | `BinaryOp(Subtract, d, i)` | Structural | §9 |
| `FunctionCall("date_diff", [Literal("day"), start, end])` | `FunctionCall("datediff", [end, start])` | Structural (name-remap + arg-reorder, day part only) | §9 |
| `FunctionCall("date_diff", [Literal(part), start, end])` (non-day part) | `BinaryOp(Subtract, FunctionCall("date_part", [Literal(part), end]), FunctionCall("date_part", [Literal(part), start]))` | Structural (per-part extraction) | §9 |
| `BinaryOp(SafeDivide, a, b)` (Spark 3.3+) | `FunctionCall("try_divide", [a, b])` 🟡 | Structural (optional optimization) | §5.1 |

**Dialect-layer rendering:**

| Variant | SQL | Method |
|---|---|---|
| `Expr::Like` | `expr LIKE pattern` | default |
| `Expr::ILike` | `LOWER(expr) LIKE LOWER(pattern)` | `ilike` (no native ILIKE) |
| `Expr::RegexpMatch` | `expr RLIKE pattern` (full-match) / `expr RLIKE CONCAT('.*', pattern, '.*')` (partial) | `regexp_match` |
| `Expr::RegexpExtract` | `regexp_extract(expr, pattern, group)` | `regexp_extract` |
| `Expr::DateTrunc` | `date_trunc('grain', expr)` | `date_trunc` |
| `Expr::Cast` | `CAST(expr AS <type>)` | `type_name` |
| `FunctionCall("percentile_cont", [p, col])` | `percentile_cont(p) WITHIN GROUP (ORDER BY col)` | aggregate-render path (Spark 3.1+) |

### 13.4 ANSI (fallback dialect)

For adapters that emit ANSI SQL without a specific engine target (used by the catalog-inspection-only paths or as a baseline-comparison form). Every rewrite is Name-only at the Dialect layer:

| Variant | SQL |
|---|---|
| `Expr::ILike` | `LOWER(expr) LIKE LOWER(pattern)` |
| `Expr::RegexpMatch` | `REGEXP_LIKE(expr, pattern)` |
| `Expr::RegexpExtract` | `REGEXP_EXTRACT(expr, pattern, group)` |
| `Expr::DateTrunc` | `DATE_TRUNC('grain', expr)` |
| `Expr::Cast` | `CAST(expr AS <type>)` |

ANSI does not perform PlanBuilder-layer rewrites — it emits the canonical `Expr` tree verbatim where SQL standard supports it.

---

## 14. Per-Engine Coverage Gaps

Canonical-catalog entries from `14a §4` that FAILED the Q10 intersection test and were demoted to adapter-extended. Each gap has a `TD-FUNCS-MAPPING-*` tag tracking the demotion and any potential future promotion if the engine landscape shifts. Cross-references §13's per-adapter rewrite summaries.

| TD ID | Canonical candidate | Failing engine(s) | Demotion reason | Resolution path |
|---|---|---|---|---|
| `TD-FUNCS-MAPPING-INITCAP` | `initcap` | DuckDB | No native equivalent. | DuckDB adds native `initcap`, or adapter accepts structural emulation (e.g. `regexp_replace` pattern). |
| `TD-FUNCS-MAPPING-PERCENTILE` | `percentile_cont` / `percentile_disc` | — / DataFusion (`percentile_disc` absent) | Resolved 2026-05-21. | **CLOSED for `percentile_cont`** — Spark 3.1+ has native exact form; all three engines converge on `WITHIN GROUP` SQL-standard syntax (dialect-layer rendering, not a structural rewrite). `percentile_disc` demoted to adapter-extended (DuckDB + Spark only) per §12.4. |
| `TD-FUNCS-MAPPING-TO-DATE-FORMAT` | 2-arg `to_date(str, fmt)` / `to_timestamp(str, fmt)` | All three (mutually incompatible format-string dialects) | Format-string syntax divergence (DF Chrono strftime / Spark Java DateTimeFormatter / DuckDB strptime). | Adapter-extended on each engine with engine-native format syntax. Canonical 1-arg ISO-only form is ratified. |
| `TD-FUNCS-MAPPING-IF-IFNULL-NVL` | `if` / `ifnull` / `nvl` | — | Overlap with `Case` / `Coalesce` dedicated variants. | **CLOSED 2026-05-21.** Not registered; authors write `Expr::Case` / `Expr::Coalesce` directly. Dedicated variants render natively on all three engines. |
| `TD-FUNCS-MAPPING-DATE-ADD-SPARK` | `date_add(date, interval)` | Spark (only integer-days form native) | Arg-type divergence. | Already structurally rewritten via `date + interval` at Spark adapter's PlanBuilder. Non-blocking. Spark integer-days form `date_add(d: Date, n: Integer)` is adapter-extended. |
| `TD-FUNCS-MAPPING-DATE-DIFF-2ARG` | 2-arg `date_diff(start, end)` integer-days form | — | Convenience overload; canonical 3-arg `date_diff('day', start, end)` covers the use-case. | Adapter-extended on engines that prefer the shorter signature (Spark `datediff`, DuckDB legacy form). |
| `TD-FUNCS-MAPPING-LOG-ARITY` | `log(base, x)` | DuckDB (1-arg base-10 only) | Arity / semantic divergence. | DuckDB adds 2-arg form, or canonical splits into `log` (1-arg base-10 only) + `logb` (2-arg). |
| `TD-FUNCS-MAPPING-SAFEDIVIDE-SPARK` | `SafeDivide` Spark rendering | — | Optimization opportunity, not a gap. | Spark 3.3+ adapter may emit `try_divide`. |
| `TD-FUNCS-MAPPING-BINOP-EMPIRICAL` | BinaryOp promotion tables §5.2–§5.3 | All three | Rows drafted from docs, not empirically verified. | Test harness against live adapter instances. |
| `TD-FUNCS-MAPPING-DECIMAL-DIV` | `Decimal / Decimal` result type | DuckDB (divergent from DF / Spark) | Result-type divergence. | Reconciliation Cast at Semantics boundary. |
| `TD-FUNCS-MAPPING-DATE-SUB-DATE` | `Date - Date` result type | DataFusion (returns `Interval`) vs DuckDB / Spark (return `Integer` days) | Result-type divergence. | Document as expected; author declares `data_type:` appropriately. |
| `TD-FUNCS-MAPPING-AGG-INTERSECTION` | Non-closed aggregates §3.2 | — | Resolved 2026-05-21. | **CLOSED** — Round-2 intersection ratified 8 canonical entries (`stddev`, `stddev_pop`, `variance`, `var_pop`, `median`, `string_agg`, `percentile_cont`, `approx_count_distinct`); `percentile_disc` adapter-extended only. |
| `TD-FUNCS-MAPPING-ADAPTER-INVENTORY` | §12 seed lists | All three | Inventories not yet authoritative. | Per-adapter crate README ratifies full list. |
| `TD-FUNCS-MAPPING-DATETIME-FORMATS` | `to_date` / `to_timestamp` format strings | All three differ (strftime / Java SimpleDateFormat) | Format-string dialect divergence. | Either canonical format-string grammar OR engine-delegated per-call. |
| `TD-FUNCS-MAPPING-CONCAT-WS-PROMOTE` | `concat_ws` | — | Not in `14a §4.2`; universal per legacy. | Add to `14a §4.2` in Round-2. |
| `TD-FUNCS-MAPPING-STARTS-WITH-PROMOTE` | `starts_with` / `ends_with` | Spark (name-remap, no underscore) | Not in `14a §4.2`; minor name-remap only. | Add to `14a §4.2` in Round-2. |
| `TD-FUNCS-MAPPING-REPEAT` | `repeat` | — | Not in `14a §4.2`; universal per legacy. | Add to `14a §4.2` in Round-2. |
| `TD-FUNCS-MAPPING-CONTAINS` | `contains` | — | Not in `14a §4.2`; universal per legacy §12. | Add to `14a §4.2` in Round-2. |
| `TD-FUNCS-MAPPING-TRANSLATE` | `translate` | — | Not in `14a §4.2`; universal per legacy §12. | Add to `14a §4.2` in Round-2. |
| `TD-FUNCS-MAPPING-TRUNC-PROMOTE` | `trunc` / `truncate` | — | Not in `14a §4.3`; universal per legacy §12. | Add to `14a §4.3` in Round-2. |
| `TD-FUNCS-MAPPING-LEFT-RIGHT` | `left` / `right` | Spark (requires ≥ 3.2) | Version-floor constraint. | Confirm Spark floor; add to `14a §4.2`. |
| `TD-FUNCS-MAPPING-REGEXP-REPLACE-4ARG` | 4-arg `regexp_replace` | All three (DF / DuckDB = flags; Spark = position) | Semantic divergence. | Keep canonical as 3-arg; 4-arg adapter-extended with distinct per-engine specs. |

---

## 15. Versioning

Following `registry/types_mapping.md §4` and `README.md` policy, each mapping row SHOULD cite the adapter crate + verified engine version. Current default pins (tentative; may shift per Q-FUNCS-MAP-020):

| Engine | Target version(s) | Notes |
|---|---|---|
| DataFusion | 40.x+ 🟡 | Major releases every ~6-8 weeks. `contains` landed in 40.x; pre-40 adapter floors rejected. |
| DuckDB | 1.1.x 🟡 | Matches `types_mapping.md §1` pin. |
| Spark | 3.5.x (SQL dialect) 🟡 | Floor version for features like native `median`, `split_part`, `try_divide` — see per-row `(Spark 3.Z+)` notes. Spark 3.4 is the minimum for `TimestampNTZType` (per `types_mapping.md §3.2`), which transitively floors the adapter at 3.4. |

Rows documenting features behind a specific engine version carry `(Engine X.Y+)` inline. Unverified rows are marked 🟡; as adapter implementation lands and verifies each row against a live engine, the 🟡 marker is removed and the exact verified version replaces the range.

Breaking changes in an engine (rename, semantic change between major versions) are documented as additional rows or dated annotations — not destructive edits — per `README.md §versioning-and-churn`.

---

## 16. Cast Semantics of Function Results

Edge cases where the per-engine function-result type diverges despite identical canonical signatures. These flow into the Semantics-boundary reconciliation Cast rule (`19 §3.7`) and compose with `registry/types_mapping.md §2`'s canonical cast matrix.

### 16.1 Integer-division result

Per §5.2: canonical `a / b` where `a, b: Integer` produces:

| Engine | Result type | Reconciliation to declared `Integer` | Reconciliation to declared `Double` |
|---|---|---|---|
| DataFusion | `Integer` (truncating) 🟡 | None (identity). | Widening `CAST(... AS DOUBLE)` per `types_mapping §2.3`. |
| DuckDB | `Double` (promoting) 🟡 | Narrowing `CAST(... AS INTEGER)` — emits `EXPR_W_CAST_NARROW`; value-lossy. | None (identity). |
| Spark | `Double` (promoting) 🟡 | Narrowing `CAST(... AS BIGINT)` — emits `EXPR_W_CAST_NARROW`. | None. |

⚠️ Authors targeting multiple engines with a declared integer quotient should write an explicit cast in `expr:` rather than relying on boundary reconciliation — the cross-engine behavior difference is surprising. `TD-FUNCS-MAPPING-INT-DIV-RESULT`.

### 16.2 Decimal-division result

Per §5.3: DuckDB's decimal-division result-type formula (`p1 + s2` / `s1`) is narrower than DataFusion / Spark (Spark-style `p1 - s1 + s2 + max(6, s1 + p2 + 1)` / `max(6, s1 + p2 + 1)`). A Semantics declaring `data_type: Decimal(18, 4)` over a decimal division may need different reconciliation Casts per adapter. `TD-FUNCS-MAPPING-DECIMAL-DIV`.

### 16.3 `Date - Date` difference

Per §5.4: returns `Interval` on DataFusion, `Integer` (days) on DuckDB and Spark. A Semantics declaring `data_type: Integer` over `d2 - d1` reconciles differently:

| Engine | Natural result | Reconciliation to `Integer` days |
|---|---|---|
| DataFusion | `Interval` 🟡 | Structural — extract days via `date_part('day', ...)` or similar. Cannot be a simple `Cast`. |
| DuckDB | `Integer` 🟡 | None (identity). |
| Spark | `Integer` 🟡 | None (identity). |

DataFusion's divergence is best handled by canonicalizing `date_diff(d1, d2)` (§9) rather than writing `d2 - d1` directly. `TD-FUNCS-MAPPING-DATE-SUB-DATE`.

### 16.4 `count(*)` / `count(expr)` return width

| Engine | Return type |
|---|---|
| DataFusion | `Int64` / `Long` |
| DuckDB | `BIGINT` / `Long` |
| Spark | `LongType` / `Long` |

Convergent. Canonical `14a §4.7` specifies `Long`. No reconciliation needed.

### 16.5 `sum(Integer)` return width

| Engine | Return type |
|---|---|
| DataFusion | `Int64` / `Long` |
| DuckDB | `HUGEINT` (128-bit) 🟡 — divergent at extreme values | |
| Spark | `LongType` / `Long` |

⚠️ DuckDB promotes `sum` over 32-bit integers to 128-bit to protect against overflow; DF / Spark promote to 64-bit (SQL:2016 canonical, per `14a §4.7`). Large aggregations over DuckDB MAY produce values outside `Long` range — the adapter emits a warning-level Diagnostic (`EXPR_W_SUM_OVERFLOW_RISK` 🟡) when the physical source's row count × max-absolute-value estimate exceeds `i64::MAX`. `TD-FUNCS-MAPPING-SUM-HUGEINT`.

---

## 17. Interaction with Other Documents

- **`foundations/14_expressions.md`** — this registry is the downstream consumer of `14 §3.3`'s `Expr` variant catalog. Every dedicated variant and every `FunctionCall` name is mapped here.
- **`foundations/14a_function_catalog.md`** — canonical upstream. `14a §4` defines which functions are canonical; this doc maps them. `14a §5` defers per-engine BinaryOp reality here (fulfilled in §5).
- **`registry/README.md`** — shared policy (engine coverage, versioning, Living status).
- **`registry/types_mapping.md`** — canonical type mappings consumed by §11 (`Cast` variant target rendering) and §16 (reconciliation Casts).
- **`apis/36_semstrait_adapter.md`** — the `EngineAdapter` trait and its `Dialect` / `PlanBuilder` layering. This doc specifies WHAT each adapter rewrites; `36` specifies HOW the layering composes.
- **Adapter crates** (future `semstrait-adapter-datafusion`, `semstrait-adapter-duckdb`, `semstrait-adapter-spark`) — own the authoritative `FunctionRewriter` table and `RegistryExtension` implementations. §12 mirrors each adapter's contribution; the adapter's own README is authoritative.
- **`questions/open/functions_mapping_questions.md`** — parked unresolved questions surfaced by Round-2 drafting.

---

## 18. TECH_DEBT Index

Consolidated list of all `TD-FUNCS-MAPPING-*` entries emitted by this doc. Each maps back to its originating §.

| TD ID | § | Current posture |
|---|---|---|
| `TD-FUNCS-MAPPING-INITCAP` | 14 | Demoted to adapter-extended (DF + Spark only). |
| `TD-FUNCS-MAPPING-PERCENTILE` | 14 | `percentile_cont` ratified canonical (dialect-layer `WITHIN GROUP`); `percentile_disc` adapter-extended. |
| `TD-FUNCS-MAPPING-TO-DATE-FORMAT` | 14 | 2-arg forms demoted. |
| `TD-FUNCS-MAPPING-IF-IFNULL-NVL` | 14 | **CLOSED 2026-05-21** — `if` / `ifnull` / `nvl` not registered; authors use `Case` / `Coalesce`. |
| `TD-FUNCS-MAPPING-DATE-ADD-SPARK` | 14 | Structural rewrite in place; integer-days form adapter-extended. |
| `TD-FUNCS-MAPPING-DATE-DIFF-2ARG` | 14 | Canonical = 3-arg; 2-arg form adapter-extended. |
| `TD-FUNCS-MAPPING-LOG-ARITY` | 14 | Open — requires `14a` Round-2 clarification. |
| `TD-FUNCS-MAPPING-SAFEDIVIDE-SPARK` | 14 | Optimization; non-blocking. |
| `TD-FUNCS-MAPPING-BINOP-EMPIRICAL` | 14 | Blocked on test harness. |
| `TD-FUNCS-MAPPING-DECIMAL-DIV` | 14 | Reconciliation Cast at Semantics boundary. |
| `TD-FUNCS-MAPPING-DATE-SUB-DATE` | 14 | Documented as expected divergence. |
| `TD-FUNCS-MAPPING-AGG-INTERSECTION` | 14 | **CLOSED 2026-05-21** — 8 canonical entries ratified at `14a §4.6`. |
| `TD-FUNCS-MAPPING-ADAPTER-INVENTORY` | 14 | Blocked on per-adapter crate readmes. |
| `TD-FUNCS-MAPPING-DATETIME-FORMATS` | 14 | Open design item. |
| `TD-FUNCS-MAPPING-CONCAT-WS-PROMOTE` | 14 | Proposed `14a` Round-2 addition. |
| `TD-FUNCS-MAPPING-STARTS-WITH-PROMOTE` | 14 | Proposed `14a` Round-2 addition. |
| `TD-FUNCS-MAPPING-REPEAT` | 14 | Proposed `14a` Round-2 addition. |
| `TD-FUNCS-MAPPING-CONTAINS` | 14 | Proposed `14a` Round-2 addition. |
| `TD-FUNCS-MAPPING-TRANSLATE` | 14 | Proposed `14a` Round-2 addition. |
| `TD-FUNCS-MAPPING-TRUNC-PROMOTE` | 14 | Proposed `14a` Round-2 addition. |
| `TD-FUNCS-MAPPING-LEFT-RIGHT` | 14 | Proposed `14a` Round-2 addition, version-floor-constrained. |
| `TD-FUNCS-MAPPING-REGEXP-REPLACE-4ARG` | 14 | 4-arg form adapter-extended per engine. |
| `TD-FUNCS-MAPPING-INT-DIV-RESULT` | 16.1 | Divergent per engine; author-side cast recommended. |
| `TD-FUNCS-MAPPING-SUM-HUGEINT` | 16.5 | Warning Diagnostic on DuckDB when range risk exceeds `i64::MAX`. |

## Testing Strategy

Summary and gap behavioral gates:

- **Per-adapter rewrite table completeness**: Every canonical function + dedicated variant is mapped exactly once per engine in §13 (PlanBuilder vs Dialect per function). Cross-reference with §§3–11 sibling leaves.
- **Coverage-gap TD entries ratified**: Each §14 row (initcap/percentile_disc/2-arg to_date/to_timestamp/if/ifnull/nvl) maps to a TD; resolution path documented (adapter-extended vs canonical-promotion-candidate).
- **Adapter-extended inventory seed lists present**: §12.1/§12.2/§12.3/§12.4 seed lists cross-reference per-adapter crate READMEs (authoritative).
- **Cast-semantics edge cases documented**: §16 Integer÷Integer result-type divergence (DF=Integer, DuckDB/Spark=Double), Decimal÷Decimal formula divergence (DuckDB vs DF/Spark), Date-Date result-type divergence (DF=Interval, DuckDB/Spark=Integer), count/sum return width (convergent count=Long; divergent sum DuckDB=HUGEINT).
- **Versioning pins cited**: §15 DataFusion 40.x+ / DuckDB 1.1.x / Spark 3.5.x (floor 3.4 per types_mapping.md).
- **TD index complete**: §18 consolidates all `TD-FUNCS-MAPPING-*` entries with originating § and current posture.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
