---
id: registry-functions-mapping-string
status: ratified
version: 1.0
impl: drift
---

# String Function Mappings

This leaf specifies string function per-engine mappings from the 12 ratified canonical entries (upper/lower/length/substring/trim/ltrim/rtrim/concat/replace/lpad/rpad/reverse/split_part/position), including structural rewrites (concat → || chain on DF/DuckDB, lpad/rpad 2-arg → 3-arg on DuckDB/Spark) and name-remaps (position argument-order divergence).

## 7. String Functions (`14a §4.2`)

*Ratified Round-2 (2026-05-21). Canonical FunctionSpecs at `14a §4.2`.*

| Canonical | Signature | DataFusion | Tier | DuckDB | Tier | Spark | Tier | Notes |
|---|---|---|---|---|---|---|---|---|
| `upper` | `(String) -> String` | `upper` | Name-only | `upper` | Name-only | `upper` | Name-only | Universal. Unicode-aware. |
| `lower` | `(String) -> String` | `lower` | Name-only | `lower` | Name-only | `lower` | Name-only | Universal. Unicode-aware. |
| `length` | `(String) -> Integer`, `(Array<T>) -> Integer` | String → `length` (or `character_length`); Array → `array_length` | Name-remap (Array) | `length` (polymorphic over String + LIST) | Name-only | String → `length`; Array → `size` | Name-remap (Array) | Character count for String; element count for Array. DF returns `Int32` for `Utf8`/`Utf8View`, `Int64` for `LargeUtf8` — adapter normalizes. |
| `substring` | `(String, Integer) -> String`, `(String, Integer, Integer) -> String` | `substr` / `substring` | Name-only | `substring` / `substr` | Name-only | `substring` / `substr` | Name-only | 1-indexed. Positive `pos` only — negative `pos` is non-portable (each engine differs) and rejected at compile. |
| `trim` | `(String) -> String`, `(String, String) -> String` | `btrim` (alias `trim`) | Name-remap | `trim` | Name-only | `trim` | Name-only | Default = ASCII space `0x20`. 2-arg form: strips any character in the second arg from both ends. |
| `ltrim` | `(String) -> String`, `(String, String) -> String` | `ltrim` | Name-only | `ltrim` | Name-only | `ltrim` (Spark 3.x: `(str, set)` order) | Name-only | Default = ASCII space. Set semantics in 2-arg form. |
| `rtrim` | `(String) -> String`, `(String, String) -> String` | `rtrim` | Name-only | `rtrim` | Name-only | `rtrim` (Spark 3.x: `(str, set)` order) | Name-only | Default = ASCII space. Set semantics in 2-arg form. |
| `concat` | variadic `(String, String...) -> String` | `a || b || c || ...` (PlanBuilder rewrite) | Structural | `a || b || c || ...` (PlanBuilder rewrite) | Structural | `concat(...)` | Name-only | NULL-propagating canonical. DF / DuckDB native `concat` is NULL-skip — rewritten to `||`-chain so semantics match Spark's NULL-propagating `concat`. |
| `replace` | `(String, String, String) -> String` | `replace` | Name-only | `replace` | Name-only | `replace` | Name-only | Replaces all occurrences. Literal substring (not regex). |
| `lpad` | `(String, Integer) -> String`, `(String, Integer, String) -> String` | `lpad` (both arities) | Name-only | 2-arg → `lpad(s, n, ' ')` rewrite; 3-arg native | Structural (2-arg) / Name-only (3-arg) | 2-arg → `lpad(s, n, ' ')` rewrite; 3-arg native | Structural (2-arg) / Name-only (3-arg) | DF native for both arities; Spark / DuckDB require 3-arg natively, so 2-arg form is rewritten by injecting `' '` at the PlanBuilder layer. Truncates if input already exceeds target length. |
| `rpad` | `(String, Integer) -> String`, `(String, Integer, String) -> String` | `rpad` (both arities) | Name-only | 2-arg → `rpad(s, n, ' ')` rewrite; 3-arg native | Structural (2-arg) / Name-only (3-arg) | 2-arg → `rpad(s, n, ' ')` rewrite; 3-arg native | Structural (2-arg) / Name-only (3-arg) | Mirror of `lpad`. |
| `reverse` | `(String) -> String`, `(Array<T>) -> Array<T>` | String → `reverse`; Array → `array_reverse` | Name-remap (Array) | `reverse` (polymorphic over String + LIST) | Name-only | `reverse` (polymorphic over String + ARRAY) | Name-only | Reverses by code points (String) / element order (Array). |
| `split_part` | `split_part(str, delim, part_num) -> String` 🟡 | `split_part` | Name-only | `split_part` | Name-only | `split_part` (Spark 3.4+) 🟡 | Name-only | 1-indexed; returns empty string on out-of-range. Spark-version floor constraint per Q-FUNCS-MAP-020. Pre-Round-2 row; not in `14a §4.2`. |
| `position` | `position(substr, str) -> Long` 🟡 | `strpos(str, substr)` | Name-remap | `strpos(str, substr)` / `position(substr IN str)` | Name-remap | `locate(substr, str, [pos])` | Structural | ⚠️ Argument order and function name both diverge on Spark. Pre-Round-2 row; not in `14a §4.2`. See Q-FUNCS-MAP-001. |

Portability summary: 12 ratified canonical entries across the three engines. Two structural rewrites (DF / DuckDB `concat` → `||`-chain; Spark / DuckDB `lpad`/`rpad` 2-arg → 3-arg). All others Name-only or single Name-remap.

### 7.1 String functions demoted from legacy catalog

| Legacy ID | Name | Rationale | Disposition |
|---|---|---|---|
| S4 | `concat_ws` | Legacy canonical, universal name-only. Not in `14a §4.2` candidate list. | **Canonical promotion candidate** pending Round-2 `14a` update. 🟡 as adapter-extended in §8 until `14a` updates. `TD-FUNCS-MAPPING-CONCAT-WS-PROMOTE`. |
| S11 | `left` | Universal name-only on DF / DuckDB; Spark ≥ 3.2. Not in `14a §4.2`. | Adapter-extended (§8) pending `14a` update. Q-FUNCS-MAP-004. |
| S12 | `right` | Same story as `left`. | Adapter-extended (§8). Q-FUNCS-MAP-004. |
| S16 | `starts_with` | DF / DuckDB name-only; Spark name-remap `startswith` (no underscore). | **Canonical promotion candidate**; currently adapter-extended until `14a` updates. `TD-FUNCS-MAPPING-STARTS-WITH-PROMOTE`. |
| S17 | `ends_with` | Same as `starts_with`. | Same disposition. |
| S18 | `initcap` | DuckDB has no equivalent (legacy: "N/A"). Fails Q10 intersection. | **Demoted to adapter-extended** (DataFusion + Spark only). `TD-FUNCS-MAPPING-INITCAP`. Q-FUNCS-MAP-002. |
| S20 | `repeat` | Universal name-only across all three per legacy. Not in `14a §4.2`. | **Canonical promotion candidate**. `TD-FUNCS-MAPPING-REPEAT`. Q-FUNCS-MAP-005. |
| — | `contains` | Universal name-only across all three (legacy §12). Not in `14a §4.2`. | **Canonical promotion candidate**. `TD-FUNCS-MAPPING-CONTAINS`. |
| — | `translate` | Universal name-only (legacy §12). Not in `14a §4.2`. | **Canonical promotion candidate**. `TD-FUNCS-MAPPING-TRANSLATE`. |
| — | `md5` / `sha256` | Universal name-only (legacy §12). Cryptographic — semstrait does not canonicalize these v1. | Adapter-extended only. |

---

## Testing Strategy

String function behavioral gates:

- **String functions round-trip**: All 12 ratified canonical entries (upper/lower/length/substring/trim/ltrim/rtrim/concat/replace/lpad/rpad/reverse/split_part/position) emit correctly per engine.
- **concat structural**: DF/DuckDB emit `a || b || c` chain (PlanBuilder rewrite ensures NULL-propagating semantics matching Spark's `concat`).
- **lpad/rpad 2-arg → 3-arg**: DuckDB/Spark inject space for 2-arg form; DF native for both arities.
- **position arg-order**: Canonical `position(substr, str)` → DF/DuckDB `strpos(str, substr)` (arg-swap), Spark `locate(substr, str)` (no swap but different name).
- **trim → btrim on DF**: DataFusion name-remap.
- **length array-arg name-remap**: DF `length` (String) vs `array_length` (Array); Spark `length` (String) vs `size` (Array).
- **reverse array-arg name-remap**: DF `reverse` (String) vs `array_reverse` (Array).
- **Demoted legacy functions documented**: initcap/left/right/starts_with/ends_with/repeat/concat_ws/contains/translate dispositions (adapter-extended or canonical-promotion-candidate) ratified.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
