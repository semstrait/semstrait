---
id: registry-join-types-mapping-asof
status: ratified
version: 1.0
impl: planned
---

# AsOf Join & Reserved Variants

This leaf specifies AsOf join variant emission (ScdWindow / SnapshotLatestAtOrBefore anchors), the rewrite-tier table cross-referencing temporal_shape_mapping §4.2 (First-class/Structural/Unsupported per anchor×engine), reserved future variants (Semi/Anti/Lateral DEFERRED per 16 §4.3), and Round-1 alternatives (Semi/Anti via EXISTS/NOT EXISTS filter, Lateral via computed expr).

## 4. `AsOf` Rewrite-Tier Summary Table

Per `§3.5`'s authority boundary, this section is a **cross-referenced summary** of `temporal_shape_mapping.md §4.2`. The full per-(anchor, engine) table with notes lives there; this view is indexed by `JoinType::AsOf` for readers starting from the `JoinType` end.

### 4.1 Tier taxonomy (identical to `temporal_shape_mapping.md §4.1`)

| Tier | Description | Adapter action |
|---|---|---|
| **First-class** | Engine exposes native `ASOF JOIN` (or equivalent) expressing the canonical semantics with zero rewrite (or a single trailing filter for `ScdWindow` per `temporal_shape_mapping.md §3.5.1`). | Emit native syntax; optionally append trailing filter. |
| **Structural** | Engine requires window-function + predicate rewrite. Correctness preserved; performance may differ. | PlanBuilder-layer rewrite per `temporal_shape_mapping.md §6.3` / `§6.4`. |
| **Unsupported** | No correct emission path. Hard error: `ADAPT_E_0302 UnsupportedFeature { canonical: "JoinType::AsOf(...)", engine }`. | Fail at `adapt` time. |

Per `16 §4.3` + `17 §5.2`'s legality posture, canonical v1 guarantees **`Unsupported` does not arise** for `ScdWindow` / `SnapshotLatestAtOrBefore` on any first-class engine — every engine admits either First-class or Structural emission.

### 4.2 Per-(anchor, engine) summary

| Anchor | DataFusion | DuckDB | Spark 3.5+ | Snowflake | BigQuery | Adapter responsibility |
|---|---|---|---|---|---|---|
| `AsOfAnchor::ScdWindow` | Structural | First-class (half-open preferred) 🟡 | Structural (`QUALIFY` + `ROW_NUMBER`) 🟡 | First-class (half-open preferred) 🟡 | Structural (`QUALIFY` + `ROW_NUMBER`) 🟡 | See `temporal_shape_mapping.md §6.3` for half-open predicate; `§3.5.1` for trailing-closure rationale. |
| `AsOfAnchor::SnapshotLatestAtOrBefore` | Structural | First-class (`ASOF JOIN ... ON probe >= anchor`) 🟡 | Structural (`QUALIFY` + `ROW_NUMBER`) 🟡 | First-class (`ASOF JOIN ... MATCH_CONDITION`) 🟡 | Structural (`QUALIFY` + `ROW_NUMBER`) 🟡 | See `temporal_shape_mapping.md §6.4` for `QUALIFY` vs subquery-wrap; `§6.2` for `ROW_NUMBER` variant. |

**One-line per engine (cross-ref to `temporal_shape_mapping.md §4.2`):**

- **DataFusion** — Structural for both anchors; no native `ASOF`. `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE`.
- **DuckDB** — First-class for both. `ScdWindow` uses half-open predicate in practice per `temporal_shape_mapping.md` Q-TEMPORAL-MAP-004 Option C.
- **Spark 3.5+** — Structural for both via `QUALIFY ROW_NUMBER() = 1`. Pre-3.4 requires subquery wrap. `TD-TEMPORAL-ASOF-SPARK-NATIVE`.
- **Snowflake** — First-class for both via `ASOF JOIN ... MATCH_CONDITION`.
- **BigQuery** — Structural for both; `QUALIFY` supported natively.

### 4.3 Cross-catalog synchronization

The `§4.2` table duplicates rows present in `temporal_shape_mapping.md §4.2`. Per Q-JOIN-MAP-004 Option A, divergence between the two is a **documentation bug** — `temporal_shape_mapping.md §4.2` is authoritative. A future refactor MAY eliminate the duplication by factoring the tier table into a shared fragment; Round-1 posture tolerates duplication as a navigation aid (readers arriving from either `JoinType` or `TemporalShape` find the tier in their entry-point doc).

---


## Testing Strategy

AsOf and reserved-variant behavioral gates:

- **AsOf rewrite-tier table (§4.2 rows) matches temporal_shape_mapping §4.2**: Cross-catalog synchronization (authoritative per Q-JOIN-MAP-004 Option A; this doc's §4.2 is a companion summary).
- **ScdWindow half-open predicate preferred**: DuckDB/Snowflake First-class with half-open fallback; DF/Spark/BigQuery Structural. Per Q-TEMPORAL-MAP-004 Option C, half-open predicate is the canonical emission form.
- **SnapshotLatestAtOrBefore native ASOF**: DuckDB/Snowflake First-class (`ASOF JOIN ... ON probe >= anchor`); others Structural via QUALIFY/subquery ROW_NUMBER.
- **Reserved variants NOT canonical**: Semi/Anti/Lateral DEFERRED from v1 (16 §4.3); alternatives documented (Semi/Anti via EXISTS/NOT EXISTS filter, Lateral via computed Dimension/Measure expr in model).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
