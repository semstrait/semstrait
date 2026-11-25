---
id: registry-temporal-shape-mapping-asof
status: ratified
version: 1.0
impl: planned
---

# AsOf Anchor Emission, Idioms, and Gaps

This leaf specifies AsOf anchor emission (ScdWindow / SnapshotLatestAtOrBefore), the rewrite-tier table (AUTHORITATIVE per Q-JOIN-MAP-004 Option A; join_types_mapping §4.2 mirrors this), cross-engine idioms (QUALIFY vs subquery-wrap, WHERE-max, windowed ROW_NUMBER, half-open predicate, default-current), per-engine gap catalog, TD index, versioning, and interaction.

## 4. `AsOf` Rewrite-Tier Table

**The most important table in this doc.** For every (anchor family, engine) pair, classifies the emission into one of three tiers; drives the adapter's PlanBuilder-layer dispatch.

### 4.1 Tier taxonomy

| Tier | Description | Adapter action |
|---|---|---|
| **First-class** | Engine exposes a syntactic construct that expresses the canonical semantics with zero rewrite (or a single trailing filter for `ScdWindow` per §3.5.1). | Emit native syntax; optionally append trailing filter. |
| **Structural** | Engine requires an expression-tree + predicate rewrite to preserve canonical semantics. Correctness is preserved; performance may differ from native. | PlanBuilder-layer rewrite to window-function + predicate form (§6.3 / §6.4). |
| **Unsupported** | No correct emission path exists. Hard error: `ADAPT_E_0302 UnsupportedFeature { canonical: "JoinType::AsOf(...)", engine }`. | Fail at `adapt`-time per `17 §9.3` / adapter contract. |

Per `14a §6.3` / `17`-equivalent posture, canonical v1 guarantees `Unsupported` does not arise for `ScdWindow` / `SnapshotLatestAtOrBefore` on any first-class engine — every engine admits either First-class or Structural emission. `Unsupported` is reserved for future anchor families (e.g. hypothetical `BiTemporalWindow`) on engines that cannot express them.

### 4.2 Per-(anchor, engine) classification

🟡 pending `TD-TEMPORAL-ASOF-EMPIRICAL` — empirical verification against live adapter test harness.

| Anchor | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|
| `AsOfAnchor::ScdWindow` | **Structural** (§6.3) | **First-class** with half-open fallback (§3.5.1, Q-TEMPORAL-MAP-004 Option C) 🟡 | **Structural** (`QUALIFY`-based, §6.4; Spark 3.4+) 🟡 | **First-class** with half-open fallback 🟡 | **Structural** (`QUALIFY`-based) 🟡 |
| `AsOfAnchor::SnapshotLatestAtOrBefore` | **Structural** (§6.4) | **First-class** (`ASOF JOIN ... ON probe >= anchor`) 🟡 | **Structural** (`QUALIFY`-based, Spark 3.4+) 🟡 | **First-class** (`ASOF JOIN ... MATCH_CONDITION (probe >= anchor)`) 🟡 | **Structural** (`QUALIFY`-based) 🟡 |

**One-line per engine:**

- **DataFusion**: Structural for both anchors (no native `ASOF`); emission via `Inner`/`Left` + half-open filter (`ScdWindow`) or subquery-wrapped `ROW_NUMBER` (`SnapshotLatestAtOrBefore`). Tracked as `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE` per Q-TEMPORAL-MAP-001 (Option A: remain Structural indefinitely).
- **DuckDB**: First-class for both; `ScdWindow` picks half-open predicate in practice per Q-TEMPORAL-MAP-004 Option C to enforce upper bound.
- **Spark**: Structural for both; emission via `QUALIFY ROW_NUMBER() = 1` on Spark 3.4+; subquery wrap pre-3.4. Spark 3.5.x floor per Q-TEMPORAL-MAP-003 Option A.
- **Snowflake**: First-class for both (native `ASOF JOIN ... MATCH_CONDITION`).
- **BigQuery**: Structural for both; emission via `QUALIFY`-based window rewrite (BigQuery supports `QUALIFY` natively).

---

## 5. Gap Catalog

Per-engine inventory of canonical `17` semantics the engine does NOT natively support, and the workaround strategy the adapter uses (or a TODO for future implementation).

### 5.1 DataFusion gap inventory

| Canonical feature | DataFusion status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(ScdWindow)` | No native `ASOF JOIN` | Structural rewrite to `Inner`/`Left` + half-open `FilterNode` (§6.3). | `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE` |
| `JoinType::AsOf(SnapshotLatestAtOrBefore)` | No native | Subquery-wrapped `ROW_NUMBER()` latest-at-or-before (§6.4). | same |
| Snapshot time-travel at source level | No vanilla `VERSION AS OF` syntax | Delegate to catalog / Iceberg binding (if present); else WHERE-max (§6.1). | `TD-TEMPORAL-SNAPSHOT-DATAFUSION-NATIVE` |
| `QUALIFY` clause | Not supported 🟡 | Subquery wrap (§6.4). | — |
| `FOR SYSTEM_TIME AS OF` | Not supported | WHERE-max fallback per snapshot source. | same |

**Round-1 posture**: DataFusion is Structural-tier for every temporal shape. `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE` records the door-open option if DataFusion lands native `AsOf` (Q-TEMPORAL-MAP-001 Option A: no migration plan assumed).

### 5.2 DuckDB gap inventory

| Canonical feature | DuckDB status | Adapter workaround | TD |
|---|---|---|---|
| Iceberg `VERSION AS OF` | Via `iceberg` extension 🟡 | Feature-gated adapter capability `DuckDbAdapterConfig::iceberg_extension` per Q-TEMPORAL-MAP-002 Option A; WHERE-max fallback when off. | `TD-TEMPORAL-SNAPSHOT-DUCKDB-ICEBERG` |
| Delta `VERSION AS OF` | Via `delta` extension 🟡 | Same posture, `delta_extension` flag. | same |
| `SCD Type2 valid_to` sentinel | Convention-dependent (sentinel vs NULL) | Round-1 `NULL`-aware emission only per Q-TEMPORAL-MAP-007 Option A. | `TD-TEMPORAL-SENTINEL-RATIFICATION` |

**Round-1 posture**: DuckDB is First-class for `ASOF JOIN` (both anchor families); the gaps are at the **source-format** layer (Iceberg / Delta extensions), not the SQL layer.

### 5.3 Spark gap inventory

| Canonical feature | Spark status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(*)` | No native `ASOF JOIN` | `QUALIFY ROW_NUMBER() = 1` rewrite (Spark 3.4+; §6.4). Pre-3.4 uses subquery wrap. | `TD-TEMPORAL-ASOF-SPARK-NATIVE` |
| `QUALIFY` clause | Spark 3.4+ 🟡 | Subquery-wrap fallback for Spark < 3.4. Round-1 floor is 3.5.x per Q-TEMPORAL-MAP-003. | `TD-TEMPORAL-SPARK-QUALIFY-VERSION` |
| Iceberg `VERSION AS OF` | Iceberg connector; Spark SQL syntax native | Emit directly when source is Iceberg. | — |
| Delta `VERSION AS OF` / `TIMESTAMP AS OF` | Delta connector; Spark SQL syntax native (3.3+) 🟡 | Emit directly when source is Delta. | — |
| `Timestamp` tz-naive emission | `TimestampNTZType` requires Spark 3.4+ | Per `types_mapping.md §3.2`, Spark 3.4 is the floor. This registry transitively requires the same. | `TD-ADAPTER-SPARK-TIME` (shared with `types_mapping`) |

### 5.4 Snowflake gap inventory (planned adapter)

| Canonical feature | Snowflake status | Adapter workaround | TD |
|---|---|---|---|
| `AsOfAnchor::ScdWindow` | Native `ASOF JOIN` 🟡 | Half-open predicate per Q-TEMPORAL-MAP-004 Option C. | `TD-TEMPORAL-ASOF-SCD-WINDOW-CLOSURE` |
| Table-level time-travel | Native `AT`/`BEFORE` 🟡 | Emit directly; falls back to WHERE-max for non-time-travel tables. | — |
| `MATCH_RECOGNIZE` | Native | Adapter-extended per §3.2.1. | `TD-TEMPORAL-MATCH-RECOGNIZE` |

### 5.5 BigQuery gap inventory (planned adapter)

| Canonical feature | BigQuery status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(*)` | No native | `QUALIFY`-based rewrite (BigQuery supports `QUALIFY`). | — |
| Table-level time-travel | `FOR SYSTEM_TIME AS OF` (7-day default retention) | Emit directly; honor retention via catalog-layer validation. | `TD-TEMPORAL-BIGQUERY-RETENTION` 🟡 |
| `MATCH_RECOGNIZE` | Native | Adapter-extended per §3.2.1. | `TD-TEMPORAL-MATCH-RECOGNIZE` |

---

## 6. Cross-Engine Idioms

Patterns shared across multiple adapters. Written once here; §3 / §4 / §5 rows reference these sections rather than inline the emission.

### 6.1 WHERE-max snapshot selection

The canonical fallback for "latest snapshot at or before `:as_of`" on engines without native time-travel. Works universally (DataFusion / DuckDB / Spark / Snowflake / BigQuery; any engine with correlated subqueries).

```sql
SELECT ...
FROM snapshots s
WHERE s.snapshotted_at = (
  SELECT MAX(s2.snapshotted_at)
  FROM snapshots s2
  WHERE s2.snapshotted_at <= :as_of
)
```

**Correctness.** Returns the full snapshot at the maximum `snapshotted_at` that is `<=` the probe. If no snapshot exists `<= :as_of`, returns zero rows — which matches `PLAN_E_1730 SnapshotAsOfNoCoveringSnapshot` behavior at the plan layer (`17 §9.3`).

**Performance.** Self-join-like subquery; engines typically hoist `MAX(snapshotted_at)` to a single scan. §6.2's windowed variant may outperform on wide tables.

### 6.2 Windowed latest-at-or-before (ROW_NUMBER variant)

```sql
SELECT *
FROM (
  SELECT *,
    ROW_NUMBER() OVER (ORDER BY snapshotted_at DESC) AS rn
  FROM snapshots
  WHERE snapshotted_at <= :as_of
) t
WHERE rn = 1
```

**Portability.** Universal across any engine with window functions. DuckDB / Spark 3.4+ / Snowflake / BigQuery replace the outer `WHERE rn = 1` with `QUALIFY ROW_NUMBER() OVER (...) = 1` (see §6.4).

**Per-entity version.** For SCD-style as-of joins, add `PARTITION BY entity_id`:

```sql
SELECT *
FROM (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY valid_from DESC) AS rn
  FROM scd_table
  WHERE valid_from <= :probe
) t
WHERE rn = 1
  AND (valid_to > :probe OR valid_to IS NULL)
```

### 6.3 Half-open valid-window predicate (the canonical `ScdWindow` form)

The canonical emission for `AsOfAnchor::ScdWindow`. Per Q-TEMPORAL-MAP-004 Option C, preferred on every engine — including native-`ASOF` engines — because it enforces the upper bound in a single join clause:

```sql
SELECT e.*, c.*
FROM events e
LEFT JOIN customers_scd c
  ON e.customer_id = c.customer_id
 AND c.valid_from <= e.occurred_at
 AND (c.valid_to > e.occurred_at OR c.valid_to IS NULL)
```

**Cardinality guarantee.** In a well-formed Type 2 SCD (no overlapping windows per entity), this join produces ≤ 1 match per `(e.customer_id, e.occurred_at)` pair — matching `17 §5.3`'s `ManyToOne` cardinality.

**Alternative**: native `ASOF JOIN` (DuckDB / Snowflake). See §3.5.1 for why the half-open form is preferred.

### 6.4 `QUALIFY` vs subquery-wrap

`QUALIFY` is the post-window-function filter clause analogous to `HAVING` for `GROUP BY`. Supported on DuckDB, Spark 3.4+, Snowflake, BigQuery, Trino; NOT supported on DataFusion or vanilla ANSI SQL. The adapter picks per engine:

| Engine | Form |
|---|---|
| DuckDB / Spark 3.4+ / Snowflake / BigQuery | `SELECT ... FROM t QUALIFY ROW_NUMBER() OVER (...) = 1` |
| DataFusion / Spark < 3.4 / ANSI | Subquery wrap: `SELECT ... FROM (SELECT ..., ROW_NUMBER() OVER (...) AS rn FROM t) WHERE rn = 1` |

Semantically identical; `QUALIFY` is shorter and avoids the explicit subquery.

### 6.5 Default-current emission

For a history-preserving SCD kind (`Type2` / `Type5` / `Type6`) queried without `Request.temporal.as_of`, the default-current lookup per `17 §6.3`:

1. If `current_flag_dim` is declared: `WHERE is_current = TRUE`.
2. Else if the source uses `NULL`-tailed open-ended rows: `WHERE valid_to IS NULL`.
3. Else fall back to the `PLAN_W_1731 ScdCurrentRowHeuristic` path — `QUALIFY ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY valid_from DESC) = 1` (or subquery-wrap).

Per Q-TEMPORAL-MAP-007 Option A, Round-1 emission supports only cases 1 and 2; case 3 falls through to the heuristic advisory. Author-declared sentinel support (case 2 with `valid_to = '9999-12-31'` instead of `NULL`) is DEFERRED per `17 §10 D13`.

---

## 7. TECH_DEBT Index

Consolidated list of all `TD-TEMPORAL-*` entries surfaced in this catalog. Each maps back to the originating §; entries retire as adapter implementation, `17` ratification, or engine-landscape shift resolves them.

| TD ID | § | Canonical concern | Engine(s) | Current posture |
|---|---|---|---|---|
| `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE` | §5.1 | Native `ASOF JOIN` support | DataFusion | Open — remain Structural indefinitely (Q-TEMPORAL-MAP-001 Option A). Revisit if DataFusion lands native `AsOf`. |
| `TD-TEMPORAL-ASOF-SPARK-NATIVE` | §5.3 | Native `ASOF JOIN` support | Spark | Open — `QUALIFY`-based rewrite is the long-term plan. |
| `TD-TEMPORAL-ASOF-SCD-WINDOW-CLOSURE` | §3.5.1 | Trailing-filter vs half-open predicate for `ScdWindow` on native-`ASOF` engines | DuckDB, Snowflake | Resolved per Q-TEMPORAL-MAP-004 Option C: half-open predicate preferred. |
| `TD-TEMPORAL-ASOF-EMPIRICAL` | §4.2 | Rewrite-tier rows drafted from docs, not empirically verified | All | Blocked on test harness (parallel to `TD-FUNCS-MAPPING-BINOP-EMPIRICAL`). |
| `TD-TEMPORAL-SNAPSHOT-DATAFUSION-NATIVE` | §5.1 | Source-level `VERSION AS OF` syntax | DataFusion | Open — delegate to catalog / Iceberg bindings. |
| `TD-TEMPORAL-SNAPSHOT-DUCKDB-ICEBERG` | §5.2 | Iceberg / Delta extension gating | DuckDB | Open — feature-gated per Q-TEMPORAL-MAP-002 Option A. |
| `TD-TEMPORAL-SNAPSHOT-CADENCE-ROLLUP` | §3.3.3 | Cadence-rollup reducer catalog | All | Open — last-in-period Round-1; full set waits for `25 §…` per `17 §10 D8`. |
| `TD-TEMPORAL-SPARK-QUALIFY-VERSION` | §5.3 | Spark version-floor pin for `QUALIFY` | Spark | Resolved per Q-TEMPORAL-MAP-003 Option A: Spark 3.5.x floor. |
| `TD-TEMPORAL-SENTINEL-RATIFICATION` | §3.4.3 | Author-declared `valid_to` sentinel support | All | Open — `NULL`-aware only Round 1; gated on `17 §10 D13`. |
| `TD-TEMPORAL-MATCH-RECOGNIZE` | §3.2.1 | Sequential-pattern event detection | Snowflake, BigQuery, Trino | Open — adapter-extended per Q-TEMPORAL-MAP-008 Option B. |
| `TD-TEMPORAL-BIGQUERY-RETENTION` | §5.5 | BigQuery 7-day default time-travel retention | BigQuery | Open — catalog-layer validation. |

---

## 8. Versioning

Following `registry/types_mapping.md §4` and `registry/functions_mapping.md §15` precedent, each mapping row SHOULD cite the engine version it was verified against. Round-1 tentative pins per Q-TEMPORAL-MAP-006 Option A (per-catalog drift):

| Engine | Target version(s) | Notes |
|---|---|---|
| DataFusion | 40.x+ 🟡 | Matches `functions_mapping.md §15`. No native `ASOF`; Structural-tier indefinitely. |
| DuckDB | 1.1.x 🟡 | Matches `functions_mapping.md §15` / `types_mapping.md §1`. `ASOF JOIN` since 0.9 (well below floor). Iceberg / Delta extensions Q-TEMPORAL-MAP-002. |
| Spark | 3.5.x 🟡 | Matches `functions_mapping.md §15`. `QUALIFY` required (Spark 3.4+); `TimestampNTZType` required (3.4+); Round-1 pins at 3.5.x. `TD-TEMPORAL-SPARK-QUALIFY-VERSION`. |
| Snowflake | 2024.x 🟡 | `ASOF JOIN` GA since 2023; `MATCH_CONDITION` syntax stable. Pin firms up at adapter ratification. |
| BigQuery | 2024.x 🟡 | `FOR SYSTEM_TIME AS OF` long-available; 7-day default retention (`TD-TEMPORAL-BIGQUERY-RETENTION`). |
| Substrait | 0.48+ 🟡 | Plan-IR target; `AsOf` emitted via extension URI. |

Rows citing features behind a specific engine version carry `(Engine X.Y+)` inline. Unverified rows are marked 🟡; as adapter implementation lands and verifies each row against a live engine, the 🟡 marker is removed and the exact verified version replaces the range. Breaking changes between engine major versions are documented as additional rows or dated annotations — not destructive edits — per `registry/README.md §versioning-and-churn`.

---

## 9. Round-1 Open Items

Unresolved questions parked in [`questions/open/temporal_shape_mapping_questions.md`](../../questions/open/temporal_shape_mapping_questions.md). Summary:

| Q | Title | Round-1 position | Blocking? |
|---|---|---|---|
| Q-TEMPORAL-MAP-001 | DataFusion native `ASOF JOIN` adoption | Option A — Structural indefinitely | No |
| Q-TEMPORAL-MAP-002 | DuckDB Iceberg / Delta extension gating | Option A — feature-gated | No |
| Q-TEMPORAL-MAP-003 | Spark `QUALIFY` version-floor pin | Option A — Spark 3.5.x floor | No |
| Q-TEMPORAL-MAP-004 | `ASOF JOIN` `valid_to` closure strategy | Option C — anchor-family-conditional | No |
| Q-TEMPORAL-MAP-005 | Snapshot cadence-rollup reducer policies | Option A — last-in-period only | No |
| Q-TEMPORAL-MAP-006 | Engine version pins | Option A — per-catalog drift | No |
| Q-TEMPORAL-MAP-007 | Sentinel-aware `valid_to` emission | Option A — `NULL`-aware only | No |
| Q-TEMPORAL-MAP-008 | Adapter-extended temporal idioms inventory | Option B — defer to adapter READMEs | No |

None blocks ratification of this registry; all are coordination items with adapter implementation (`34` / `36`), `17 §10` deferrals, or `25 §…` ratifications.

---

## 10. Interaction with Other Documents

- **`foundations/17_temporal_shape.md`** — canonical upstream. `17 §2` defines the shape taxonomy this registry maps; `17 §5.1` defines the `AsOfAnchor` enum `§3.5` / `§4` emit. `17` never depends on this registry's specifics (`registry/README.md §versioning-and-churn`).
- **`foundations/16_composition.md`** — canonical `JoinType` enum extended by `17 §5.1` with `AsOf(anchor)`; this registry maps the extension.
- **`registry/README.md`** — shared policy (engine coverage, versioning, Living status).
- **`registry/types_mapping.md`** — sibling catalog consumed by temporal predicates (`Timestamp` / `Date` / `Interval` emission for `valid_from` / `valid_to` / `snapshotted_at`). Spark's `TimestampNTZType` floor (3.4+) transitively constrains this catalog.
- **`registry/functions_mapping.md`** — sibling catalog consumed by `date_trunc` / `date_part` / `row_number` / comparison operators for temporal emission. Spark 3.5.x floor aligns with this catalog per Q-TEMPORAL-MAP-003.
- **`apis/34_semstrait_planner.md`** — planner's shape-aware strategy dispatch; `Request.temporal` consumption; `AsOf` emission. DEFERRED per `17 §10 D1, D4`. This registry describes the emission target.
- **`apis/35_semstrait_ir.md`** — `PlanNode::Join` carriage of `JoinType::AsOf(anchor)`. DEFERRED per `17 §10 D1`.
- **`apis/36_semstrait_adapter.md`** — the `EngineAdapter` trait + PlanBuilder / Dialect layering that consumes this registry. Per-adapter crates ratify their own `FunctionRewriter` and emission paths; §3 / §4 / §5 rows seed each adapter's implementation.
- **Adapter crates** (future `semstrait-adapter-datafusion`, `-duckdb`, `-spark`, `-snowflake`, `-bigquery`, `-substrait`) — own authoritative per-engine emission tables, feature flags (`DuckDbAdapterConfig::iceberg_extension`, etc.), and adapter-extended temporal idioms (§3.2.1, §5.4, §5.5 rows).
- **`questions/open/temporal_shape_mapping_questions.md`** — parked unresolved questions surfaced by Round-1 drafting.

## Testing Strategy

AsOf anchor and gap behavioral gates:

- **AsOf rewrite-tier table (§4.2) authoritative**: Per Q-JOIN-MAP-004 Option A, this doc's §4.2 is the authoritative per-(anchor, engine) rewrite-tier classification; join_types_mapping §4.2 is a companion summary.
- **ScdWindow half-open predicate preferred**: valid_from <= probe AND (valid_to > probe OR valid_to IS NULL) — canonical emission (§6.3); preferred on every engine per Q-TEMPORAL-MAP-004 Option C.
- **SnapshotLatestAtOrBefore native ASOF**: DuckDB/Snowflake First-class (ASOF JOIN ... ON probe >= anchor); others Structural via QUALIFY/subquery ROW_NUMBER.
- **Cross-engine idioms**: Windowed-ASOF emulation (§6.1 QUALIFY vs §6.4 subquery-wrap), WHERE-max (§6.1), windowed ROW_NUMBER (§6.2), default-current (§6.5).
- **Per-engine gaps documented**: §5 DF/Spark AsOf Structural, DuckDB/Snowflake ScdWindow trailing closure (TD entries ratified).
- **TD index complete**: §7 consolidates all TD-TEMPORAL-SHAPE-* entries.
- **Versioning pins cited**: §8 DF 40.x+ / DuckDB 1.1.x / Spark 3.5.x / Snowflake 2024.x / BigQuery 2024.x / Substrait 0.48+.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
