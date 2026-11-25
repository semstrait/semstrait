---
prereqs: [00]
status: superseded
---

# functions_mapping (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`functions_mapping/`](functions_mapping/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [registry-functions-mapping-intro](functions_mapping/intro-taxonomy.md) — Function-mapping intro, rewrite-tier taxonomy (Name-only/Name-remap/Structural/Unsupported), two-layer rewrite (PlanBuilder vs Dialect), and scope overview
- [registry-functions-mapping-aggregates](functions_mapping/aggregates.md) — Aggregate function mappings — closed five (Sum/Avg/Count/Min/Max) + non-closed aggregates (stddev/variance/median/string_agg/percentile_cont/approx_count_distinct)
- [registry-functions-mapping-binop](functions_mapping/binop-operators.md) — BinaryOp operators — comparison (Eq/NotEq/Lt/LtEq/Gt/GtEq), arithmetic (Add/Subtract/Multiply/Divide/SafeDivide/Mod), logical (And/Or), per-engine result-type reality (fills 14a §5 gap)
- [registry-functions-mapping-string](functions_mapping/string-functions.md) — String function mappings (upper/lower/length/substring/trim/ltrim/rtrim/concat/replace/lpad/rpad/reverse/split_part/position) — canonical entries from 14a §4.2
- [registry-functions-mapping-math](functions_mapping/math-functions.md) — Math function mappings (abs/round/ceil/floor/sqrt/power/exp/ln/log/log10/sign) — canonical entries from 14a §4.3
- [registry-functions-mapping-temporal](functions_mapping/temporal-functions.md) — Temporal function mappings (date_part/year/month/day/hour/minute/second/current_date/current_timestamp/date_add/date_sub/date_diff/to_date/to_timestamp) — canonical entries from 14a §4.4
- [registry-functions-mapping-conditional](functions_mapping/conditional-dedicated.md) — Conditional helpers (greatest/least) + dedicated-variant functions (case/coalesce/nullif/is_null/is_not_null/in_list/between/like/ilike/regexp_match/regexp_extract/date_trunc/cast/negate/not) from 14 §3.3
- [registry-functions-mapping-summary](functions_mapping/summary-gaps.md) — Per-adapter rewrite summary (§13 DF/DuckDB/Spark/ANSI layering digest), per-engine coverage gaps (§14 TD entries), adapter-extended inventory (§12 seed lists), cast-semantics edge cases (§16), TD index (§18), versioning (§15), interaction (§17)
