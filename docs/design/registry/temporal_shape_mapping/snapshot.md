---
id: registry-temporal-shape-mapping-snapshot
status: ratified
version: 1.0
impl: planned
---

# Snapshot Shape

This leaf specifies Snapshot shape emission — WHERE-max canonical fallback (universal), native table-level time-travel per engine (Spark Delta/Iceberg VERSION AS OF, Snowflake AT/BEFORE, BigQuery FOR SYSTEM_TIME AS OF), QUALIFY-based latest-at-or-before (vs subquery-wrap ROW_NUMBER), and cadence-rollup reducer policies (last-in-period Round-1 default).

### 3.3 `Snapshot`

**Canonical summary** (`17 §2.1, §4.1`): periodic full-state capture at `snapshotted_at_dim`. Optional declared `cadence: Grain`. As-of semantics: "latest snapshot at or before `Request.temporal.as_of`" (`17 §6.2`). Sub-cadence requests are `PLAN_E_1711` (`17 §9.3`).

#### 3.3.1 Per-engine mapping

| # | Row | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|---|
| 1 | Vanilla SQL latest-at-or-before | WHERE-max pattern (§6.1) | WHERE-max pattern | WHERE-max pattern | WHERE-max pattern | WHERE-max pattern |
| 2 | Native table-level time-travel | *(none)* | *(none in vanilla)* | `VERSION AS OF` / `TIMESTAMP AS OF` (Delta; Spark 3.3+) 🟡 | `AT (TIMESTAMP => ...)` / `BEFORE (STATEMENT => ...)` 🟡 | `FOR SYSTEM_TIME AS OF :ts` |
| 3 | Iceberg `VERSION AS OF` (extension) 🟡 | via Iceberg DataFusion binding 🟡 | `iceberg_scan('path', version => N)` (iceberg extension, Q-TEMPORAL-MAP-002) 🟡 | `VERSION AS OF N` (Iceberg connector) | `AT (BRANCH => 'main', VERSION => N)` 🟡 | — |
| 4 | Iceberg `TIMESTAMP AS OF` 🟡 | — 🟡 | `iceberg_scan('path', timestamp => :ts)` (iceberg extension) 🟡 | `TIMESTAMP AS OF :ts` | `AT (TIMESTAMP => :ts)` | — |
| 5 | Delta `VERSION AS OF` 🟡 | — 🟡 | `delta_scan('path', version => N)` (delta extension) 🟡 | `VERSION AS OF N` | — | — |
| 6 | Row-level `snapshotted_at` filter | `WHERE snapshotted_at = (SELECT MAX(snapshotted_at) FROM ... WHERE snapshotted_at <= :as_of)` | same | same | same | same |
| 7 | `QUALIFY`-based latest | — | `QUALIFY ROW_NUMBER() OVER (PARTITION BY entity ORDER BY snapshotted_at DESC) = 1 WHERE snapshotted_at <= :as_of` 🟡 | same (Spark 3.4+) 🟡 | same | same 🟡 |
| 8 | Subquery-wrapped ROW_NUMBER | `SELECT ... FROM (SELECT *, ROW_NUMBER() OVER (...) AS rn FROM ... WHERE snapshotted_at <= :as_of) WHERE rn = 1` | same | same (pre-3.4) | same | same |
| 9 | Snowflake transactional time-travel | — | — | — | `BEFORE (STATEMENT => '<query_id>')` 🟡 | — |
| 10 | Snowflake clone-at-time | — | — | — | `AT (OFFSET => -60*60)` 🟡 | — |

**Portability summary: Partial.** Rows 1 and 6 are universal (WHERE-max fallback). Rows 2–5 depend on source-format integration (Iceberg / Delta / native). Rows 7–8 are window-function rewrites universal on any engine with `ROW_NUMBER`. Rows 9–10 are Snowflake-specific time-travel idioms outside canonical `17`; documented for reference, adapter-extended.

#### 3.3.2 WHERE-max pattern — the canonical fallback

When no native time-travel is available (DataFusion, vanilla DuckDB / Spark / BigQuery against non-time-travel tables), the adapter emits:

```sql
SELECT ...
FROM snapshots s
WHERE s.snapshotted_at = (
  SELECT MAX(s2.snapshotted_at)
  FROM snapshots s2
  WHERE s2.snapshotted_at <= :as_of
)
```

This form is canonical-equivalent for all snapshot sources. The alternative `QUALIFY ROW_NUMBER() = 1` form (row 7) is preferred on engines that support `QUALIFY` because it avoids the self-join, but both are correct. See §6.1 for the idiom pattern and §6.4 for the QUALIFY rewrite.

#### 3.3.3 Cadence-rollup reducer policies

`17 §4.1` specifies snapshot cadence-rollup requires a reducer (first / last / average / configurable), with last-in-period as the Round-1 default. `17 §10 D8` defers the full reducer-policy catalog to `25 §…`. Per Q-TEMPORAL-MAP-005 Option A, this registry carries only last-in-period emission in Round 1; other reducers wait for `25 §…`.

**Last-in-period** (the Round-1 default). Select, per period, the snapshot with the maximum `snapshotted_at` within that period:

| Engine | Emission |
|---|---|
| DataFusion | `QUALIFY`-absent: subquery-wrapped `ROW_NUMBER() OVER (PARTITION BY date_trunc('week', snapshotted_at) ORDER BY snapshotted_at DESC) = 1` 🟡 |
| DuckDB | `QUALIFY ROW_NUMBER() OVER (PARTITION BY date_trunc('week', snapshotted_at) ORDER BY snapshotted_at DESC) = 1` 🟡 |
| Spark (3.4+) | `QUALIFY ROW_NUMBER() OVER (...)` 🟡 |
| Snowflake | `QUALIFY ROW_NUMBER() OVER (...)` |
| BigQuery | same via subquery wrap (no `QUALIFY`) 🟡 |

Tracked as `TD-TEMPORAL-SNAPSHOT-CADENCE-ROLLUP` in §7.


## Testing Strategy

Snapshot shape behavioral gates:

- **WHERE-max universal**: SELECT ... WHERE snapshotted_at = (SELECT MAX(...) WHERE snapshotted_at <= :as_of) — canonical fallback (§6.1, universal).
- **Native time-travel per engine**: Spark Delta/Iceberg VERSION AS OF / TIMESTAMP AS OF, Snowflake AT/BEFORE, BigQuery FOR SYSTEM_TIME AS OF (table-level time-travel per engine).
- **QUALIFY vs subquery-wrap ROW_NUMBER**: DuckDB/Spark 3.4+/Snowflake/BigQuery QUALIFY; DF/Spark<3.4 subquery (§6.2, §6.4).
- **Cadence-rollup last-in-period**: ROW_NUMBER PARTITION BY date_trunc(cadence) ORDER BY snapshotted_at DESC (Round-1 default reducer per 17 §10 D8; full catalog DEFERRED to 25).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
