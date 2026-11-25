---
doc: design/questions/deferred/functions_mapping_questions
status: Deferred (post-v1 empirical mapping depth)
purpose: Deferred questions originally authored in `open/functions_mapping_questions.md`
depends-on:
  - registry/functions_mapping.md
  - foundations/14a_function_catalog.md
---

# Deferred Questions — `registry/functions_mapping.md`

These questions are mapping-depth and empirical-verification items, not v1 architecture blockers.

---

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-FUNCS-MAP-001 | canonical `position` naming | keep `position` + Spark structural rewrite |
| Q-FUNCS-MAP-002 | `initcap` on DuckDB | adapter-extended only |
| Q-FUNCS-MAP-003 | `concat_ws` null parity verification | empirical check pending |
| Q-FUNCS-MAP-004 | `left` / `right` promotion | keep adapter-extended |
| Q-FUNCS-MAP-005 | `repeat` promotion | candidate for canonical promotion |
| Q-FUNCS-MAP-006 | `regexp_replace` 4-arg divergence | keep 3-arg canonical |
| Q-FUNCS-MAP-007 | `date_part` vs `extract` canonical naming | keep `date_part` |
| Q-FUNCS-MAP-008 | `date_add` signature divergence | keep canonical + structural rewrites |
| Q-FUNCS-MAP-009 | `date_diff` arity/unit reconciliation | keep 2-arg canonical abstraction |
| Q-FUNCS-MAP-010 | parenless current-date forms | always emit paren form |
| Q-FUNCS-MAP-011 | `to_date` formatted variants | 1-arg canonical, 2-arg adapter-extended |
| Q-FUNCS-MAP-012 | BinaryOp promotion empirical tables | verification deferred |
| Q-FUNCS-MAP-013 | non-closed aggregate intersection | verification deferred |
| Q-FUNCS-MAP-014 | `greatest`/`least` null behavior | SQL-standard canonical, DuckDB rewrite candidate |
| Q-FUNCS-MAP-015 | `if`/`ifnull`/`nvl` canonical posture | keep adapter-extended aliases |
| Q-FUNCS-MAP-016 | integer-division result-type divergence | verification deferred |
| Q-FUNCS-MAP-017 | `SafeDivide` emission policy | portable `NULLIF` default |
| Q-FUNCS-MAP-018 | adapter-extended inventory ownership | adapter README authoritative |
| Q-FUNCS-MAP-019 | adapter identity/version citation policy | engine-version-first convention |
| Q-FUNCS-MAP-020 | DataFusion version floor | pending adapter ratification |

---

## Re-open trigger

Re-open when adapter implementation rounds require any of these decisions for shipping behavior in `36`-series work.

