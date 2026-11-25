---
doc: design/questions/deferred/temporal_shape_mapping_questions
status: Deferred (post-v1 engine and adapter mapping depth)
purpose: Deferred questions originally authored in `open/temporal_shape_mapping_questions.md`
depends-on:
  - registry/temporal_shape_mapping.md
  - foundations/17_temporal_shape.md
---

# Deferred Questions — `registry/temporal_shape_mapping.md`

These are engine-specific and coordination-depth items deferred from active v1 backlog.

---

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-TEMPORAL-MAP-001 | DataFusion native `ASOF JOIN` timeline | keep structural tier until native support lands |
| Q-TEMPORAL-MAP-002 | DuckDB Iceberg/Delta extension gating | adapter capability gating |
| Q-TEMPORAL-MAP-003 | Spark `QUALIFY` version floor | Spark 3.5.x floor |
| Q-TEMPORAL-MAP-004 | `ASOF` `valid_to` closure strategy | anchor-family-conditional default |
| Q-TEMPORAL-MAP-005 | cadence-rollup reducer catalog depth | last-in-period only in current mapping |
| Q-TEMPORAL-MAP-006 | cross-catalog engine pin coordination | per-catalog pins currently allowed |
| Q-TEMPORAL-MAP-007 | sentinel-aware `valid_to` semantics | NULL-aware default until `17` extension lands |
| Q-TEMPORAL-MAP-008 | adapter-extended temporal idiom inventory | owned by adapter READMEs |

---

## Re-open trigger

Re-open when adapter implementation requires one of these choices to finalize emitted behavior or supported-engine policy.

