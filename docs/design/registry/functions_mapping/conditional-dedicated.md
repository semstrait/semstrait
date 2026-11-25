---
id: registry-functions-mapping-conditional
status: ratified
version: 1.0
impl: drift
---

# Conditional Helpers & Dedicated-Variant Functions

This leaf specifies conditional helpers (greatest/least) and dedicated-variant functions that have an `Expr` variant rather than flowing through `FunctionCall` (case/coalesce/nullif/is_null/is_not_null/in_list/between/like/ilike/regexp_match/regexp_extract/date_trunc/cast/negate/not), including per-engine divergence (ilike Spark structural, regexp_match three-form divergence, regexp_extract DF PlanBuilder structural).

## 10. Logical / Conditional Helpers (`14a §4.5`)

*Ratified Round-2 (2026-05-21). Canonical FunctionSpecs at `14a §4.5`.*

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `greatest` | variadic `(T, T...) -> T` | `greatest(x, ...)` | Name-only | `greatest(x, ...)` | Name-only | `greatest(x, ...)` | Name-only | NULL-skip across all three: returns greatest non-NULL value; NULL only when every arg is NULL. Earlier `TD-FUNCS-MAPPING-GREATEST-LEAST-NULL` (DF/Spark "propagate" vs DuckDB "ignore") was a Round-1 misclassification — closed. |
| `least` | variadic `(T, T...) -> T` | `least(x, ...)` | Name-only | `least(x, ...)` | Name-only | `least(x, ...)` | Name-only | Mirror of `greatest`. Same NULL-skip semantics across all three. |

Portability summary: **Universal** Name-only across all three engines.

### 10.1 Forms explicitly excluded from canonical

| Form | Reason |
|---|---|
| `if(cond, then, else)` | `Expr::Case` (dedicated variant per `14 §3.3`) covers the use-case directly. Authors write `Case { when: [{cond, then}], else_expr: Some(else) }`. Spark / DuckDB native `if(...)` and DF's missing-`if` are all served by `Case` rendering. Not registered to keep the surface minimal. |
| `ifnull(a, b)` / `nvl(a, b)` | `Expr::Coalesce` (dedicated variant) covers both. Authors write `Coalesce([a, b])`. All three adapters render `Coalesce` natively (universal); the DuckDB-lacks-`nvl` and DF-`nvl-as-alias-of-ifnull` quirks disappear at the Coalesce layer. |

---

## 11. Dedicated-Variant Functions (`14 §3.3`)

Functions that have a dedicated `Expr` variant rather than flowing through `FunctionCall` — these are NOT in the `FunctionRegistry` (per `14 §3.3` notes, `14a §4.2–§4.5` exclusions). Adapters render them via the Dialect layer's dedicated methods (`SqlDialect::like`, `ilike`, `regexp_match`, `regexp_extract`, `date_trunc`, `type_name`, etc.) with one exception (DataFusion `RegexpExtract`) that is reshaped to a `FunctionCall` at the PlanBuilder layer.

| Canonical | IR variant | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `case` | `Expr::Case { when, else_expr }` | `CASE WHEN cond THEN val [ELSE val] END` | Name-only | `CASE WHEN cond THEN val [ELSE val] END` | Name-only | `CASE WHEN cond THEN val [ELSE val] END` | Name-only | Universal SQL. Dialect-layer rendering. |
| `coalesce` | `Expr::Coalesce(args)` | `coalesce(a, b, ...)` | Name-only | `coalesce(a, b, ...)` | Name-only | `coalesce(a, b, ...)` | Name-only | Universal. |
| `nullif` | `Expr::NullIf { left, right }` | `nullif(a, b)` | Name-only | `nullif(a, b)` | Name-only | `nullif(a, b)` | Name-only | Universal. |
| `is_null` | `Expr::IsNull(expr)` | `expr IS NULL` | Name-only | `expr IS NULL` | Name-only | `expr IS NULL` | Name-only | Universal. (Also documented in §6 logical.) |
| `is_not_null` | `Expr::IsNotNull(expr)` | `expr IS NOT NULL` | Name-only | `expr IS NOT NULL` | Name-only | `expr IS NOT NULL` | Name-only | Universal. |
| `in_list` | `Expr::InList { expr, list, negated }` | `expr IN (a, b, ...)` / `expr NOT IN (...)` | Name-only | `expr IN (...)` / `expr NOT IN (...)` | Name-only | `expr IN (...)` / `expr NOT IN (...)` | Name-only | Universal. `negated` flag flips `IN` ↔ `NOT IN`. |
| `between` | `Expr::Between { expr, low, high, negated }` | `expr BETWEEN low AND high` / `expr NOT BETWEEN ...` | Name-only | `expr BETWEEN ... AND ...` | Name-only | `expr BETWEEN ... AND ...` | Name-only | Universal. |
| `like` | `Expr::Like { expr, pattern, negated }` | `expr LIKE pattern` / `expr NOT LIKE pattern` | Name-only | `expr LIKE pattern` | Name-only | `expr LIKE pattern` | Name-only | Universal. |
| `ilike` | `Expr::ILike { expr, pattern, negated }` | `expr ILIKE pattern` | Name-only | `expr ILIKE pattern` | Name-only | `LOWER(expr) LIKE LOWER(pattern)` | Structural | ⚠️ Spark has no native ILIKE; Dialect-layer emits the `LOWER(...)` form. ANSI fallback uses the same form. |
| `regexp_match` | `Expr::RegexpMatch { expr, pattern, negated }` → Boolean | `regexp_like(expr, pattern)` | Name-remap | `regexp_matches(expr, pattern)` | Name-remap | `expr RLIKE pattern` (full-match) / `expr RLIKE CONCAT('.*', pattern, '.*')` (partial) | Structural | ⚠️ Three different spellings. Regex dialect is canonical (RE2-compatible subset per `14 §3.3`) — per-engine regex-flavor variance is out of scope. |
| `regexp_extract` | `Expr::RegexpExtract { expr, pattern, group }` → String | `array_element(regexp_match(expr, pattern), group + 1)` 🟡 | Structural (PlanBuilder) | `regexp_extract(expr, pattern, group)` | Name-only | `regexp_extract(expr, pattern, group)` | Name-only | ⚠️ DataFusion has no scalar `regexp_extract`; `regexp_match` returns `List<Utf8>`. PlanBuilder-layer rewrites to `array_element(regexp_match(...), group + 1)` (1-indexed). Canonical `group` convention: `0` = entire match, `1+` = capture groups (matches DuckDB / Spark). |
| `date_trunc` | `Expr::DateTrunc { expr, grain }` → same | `date_trunc('grain', expr)` | Name-only | `date_trunc('grain', expr)` | Name-only | `date_trunc('grain', expr)` | Name-only | Universal. `grain` is a `Grain` enum from `13` — the adapter lowercases the variant's name for `'grain'`. |
| `cast` | `Expr::Cast { expr, target }` | `CAST(expr AS <type>)` | Name-only | `CAST(expr AS <type>)` / `expr::<type>` | Name-only | `CAST(expr AS <type>)` | Name-only | Universal. Target-type spelling per `registry/types_mapping.md §1`. |
| `negate` | `Expr::Negate(expr)` | `-expr` | Name-only | `-expr` | Name-only | `-expr` | Name-only | Universal. Operand type classes (numeric / interval) engine-validated. |
| `not` | `Expr::Not(expr)` | `NOT expr` | Name-only | `NOT expr` | Name-only | `NOT expr` | Name-only | Universal. (Also in §6.) |

Portability summary: **Universal** for most. `ilike` is **Partial** (Spark structural). `regexp_match` is **Partial** (all three differ). `regexp_extract` is **Partial** (DataFusion structural via PlanBuilder).

### 11.1 `regexp_replace` — `FunctionCall`, not a dedicated variant

`regexp_replace` does NOT have a dedicated `Expr` variant (per `14 §3.3`); it is authored as a `FunctionCall` and lives in the canonical catalog (legacy §8 P5).

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `regexp_replace` | `regexp_replace(str, pattern, replacement) -> String` | `regexp_replace(str, regexp, replacement [, flags])` | Name-only | `regexp_replace(str, pattern, replacement [, options])` | Name-only | `regexp_replace(str, regexp, rep)` | Name-only | Canonical = 3-arg form. 4-arg form differs semantically (DF/DuckDB = flags; Spark = start position) — 4-arg is demoted to adapter-extended only. Q-FUNCS-MAP-006. |

---

## Testing Strategy

Conditional and dedicated-variant behavioral gates:

- **greatest/least universal**: Both emit correctly on all engines (universal Name-only); NULL-skip semantics (returns greatest/least non-NULL value).
- **Dedicated variants round-trip**: case/coalesce/nullif/like/ilike/regexp_match/regexp_extract/date_trunc/cast emit correctly per engine.
- **ilike Spark structural**: Spark → `LOWER(expr) LIKE LOWER(pattern)` (Dialect-layer LOWER wrap).
- **regexp_match three-form divergence**: DF `regexp_like`, DuckDB `regexp_matches`, Spark `RLIKE` (all emit correctly per engine).
- **regexp_extract DF PlanBuilder rewrite**: DF → `array_element(regexp_match(...), group + 1)` (PlanBuilder layer, 1-indexed).
- **if/ifnull/nvl NOT registered**: Authors use `Expr::Case` and `Expr::Coalesce` (dedicated variants cover the use-cases).
- **regexp_replace FunctionCall**: 3-arg canonical form universal (not a dedicated variant).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
