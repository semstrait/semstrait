---
id: registry-temporal-shape-mapping-scd
status: ratified
version: 1.0
impl: planned
---

# Scd (Slowly-Changing Dimension) Type0–Type6

This leaf specifies Scd Type0–Type6 emission — valid-window half-open predicate (Type2/5/6 canonical form: valid_from <= probe AND (valid_to > probe OR valid_to IS NULL)), per-subtype mapping (Type0 append-only, Type1 MERGE, Type2 valid_from/valid_to, Type3 prior-value column, Type4/5/6 DEFERRED per 17 §10), sentinel-vs-NULL conventions (Round-1 NULL-aware only), and default-current lookup (current_flag / valid_to IS NULL / ROW_NUMBER heuristic).

### 3.4 `Scd` (Slowly-Changing Dimension) Type0–Type6

**Canonical summary** (`17 §2.2`): seven Kimball subtypes ratified at the vocabulary level. Round-1 planner targets Type1 / Type2; Type0 / Type3 recognized for advisory purposes; Type4 / Type5 / Type6 vocabulary-ratified, planner work DEFERRED (`17 §10 D7, D9, D10`).

#### 3.4.1 Per-subtype emission

| SCD Subtype | Canonical semantics (`17 §2.2`) | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|---|
| `Type0` (retain original) | No update after insert. Append-only. | Regular table; ingest-layer enforces immutability (Q-TEMPORAL-006 Option A: out of scope for semstrait). | same | same | same (or use `TRANSIENT` / write-once policies) 🟡 | same |
| `Type1` (overwrite) | One row per entity; updates overwrite. No history. | `MERGE ... WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT` (DF DML 🟡) | `INSERT OR REPLACE` / `MERGE INTO` | `MERGE INTO` (Delta 🟡) / `UPSERT` | `MERGE INTO` | `MERGE INTO` |
| `Type2` (full history) | `valid_from` / `valid_to` per row; `current_flag` optional. | `Inner`/`Left` + half-open predicate (§6.3) | native `ASOF JOIN` (§3.5) + trailing closure (§3.5.1), or half-open predicate | `QUALIFY ROW_NUMBER()` rewrite, or half-open predicate | native `ASOF JOIN` + trailing closure, or half-open predicate | structural rewrite (no native `ASOF`), or half-open predicate |
| `Type3` (prior-value column) | Current + one prior value. No versioning. | Regular column read; no special emission. | same | same | same | same |
| `Type4` (history table) | Current table + separate history table (per `history_data_kind_ref`). | Plan-level join to history-kind's resolved binding (DEFERRED; `17 §10 D7`) | same | same | same | same |
| `Type5` (Type 4 + mini-dim) | Type 4 + outrigger mini-dim (`mini_dim_ref`). | DEFERRED (`17 §10 D9`) | same | same | same | same |
| `Type6` (hybrid) | Type 1 + Type 2 (+ optional Type 3). Carries both current and historical columns on each row. | half-open predicate for as-of; direct column read for current (`current_value_dim`). Detection of current-vs-history disagreement DEFERRED (`17 §10 D10`). | same | same | same | same |

**Portability summary per subtype:**

- **Type0**: Universal. No query-emission differences; semstrait does not enforce append-only at query-plan time (Q-TEMPORAL-006).
- **Type1**: Universal at query time. Ingest / MERGE forms differ but are outside semstrait's query-emission scope.
- **Type2**: **Partial**. Native `ASOF JOIN` available on DuckDB / Snowflake; Structural rewrite on DataFusion / Spark / BigQuery. See §4 rewrite-tier table.
- **Type3**: Universal. The prior-value column is read as a regular column.
- **Type4 / Type5 / Type6**: DEFERRED per `17 §10 D7, D9, D10`. Structural shape ratified; emission draft in this registry.

#### 3.4.2 Valid-window JOIN pattern (Type 2 / 5 / 6)

The canonical emission for `Events ↔ Scd::Type2` (and its sibling subtypes) with `AsOfAnchor::ScdWindow` is:

```sql
SELECT e.*, c.*
FROM events e
LEFT JOIN customers_scd c
  ON e.customer_id = c.customer_id
 AND c.valid_from <= e.occurred_at
 AND (c.valid_to > e.occurred_at OR c.valid_to IS NULL)
```

The `IS NULL` disjunct handles open-ended current rows (the canonical open-ended convention per `17 §2.2`). For sentinel-bearing sources (e.g. `valid_to = '9999-12-31'` instead of `NULL`), §3.4.3 documents the observed conventions; Round-1 emission uses the `NULL`-aware form only, per Q-TEMPORAL-MAP-007 Option A.

This form is the **universal canonical** for `ScdWindow` anchors: it works on every engine, and per Q-TEMPORAL-MAP-004 Option C it is preferred even on engines with native `ASOF JOIN` (since `ASOF` alone does not enforce the upper bound). See §6.3.

#### 3.4.3 Sentinel-vs-NULL conventions

Observed conventions for `valid_to` on open-ended rows in the wild; documentary only in Round 1:

| Convention | Adopted by | Round-1 emission posture |
|---|---|---|
| `valid_to IS NULL` | canonical open-ended convention per `17 §2.2` | Emitted by adapters; `valid_to > probe OR valid_to IS NULL` predicate. |
| `valid_to = '9999-12-31'` | Kimball classical; many Oracle / SQL Server shops | Not emitted; author may file an adapter-side `SentinelConfig` once `17 §10 D13` lands. Until then, `PLAN_W_1731 ScdCurrentRowHeuristic` fires on sentinel-bearing sources. |
| `valid_to = '2999-12-31'` / `'3000-01-01'` / `'9999-12-31 23:59:59'` | shop-specific | Same posture. |
| `valid_to > '2099-01-01'` as a heuristic "sentinel-ish" range | rare; not recommended | Unsupported; `PLAN_W_1731` heuristic path. |

Tracked as `TD-TEMPORAL-SENTINEL-RATIFICATION` in §7; resolution gated on `17 §10 D13` ratifying author-declared sentinels.

### 3.5 `JoinType::AsOf` variant emission

**Canonical summary** (`17 §5.1`): temporal-proximity join matching the most recent `to`-side row whose anchor satisfies the anchor condition relative to the `from`-side probe timestamp. Two anchor families in canonical v1: `AsOfAnchor::ScdWindow { probe_dim, to_valid_from_dim, to_valid_to_dim }` and `AsOfAnchor::SnapshotLatestAtOrBefore { probe_dim, to_snapshotted_at_dim }`.

**Implementation-DEFERRED status** (`17 §10 D1, D2`). The planner does not yet emit `JoinType::AsOf` in Round 1. This registry describes the **emission target** adapters will converge on once planner support lands.

| Engine | Native `ASOF JOIN`? | `ScdWindow` emission | `SnapshotLatestAtOrBefore` emission |
|---|---|---|---|
| DataFusion | No 🟡 | Structural: `Inner`/`Left` + `FilterNode(valid_from <= probe AND (valid_to > probe OR valid_to IS NULL))` (§6.3) | Structural: subquery-wrapped `ROW_NUMBER()` latest-at-or-before pattern (§6.4) |
| DuckDB | Yes — `ASOF JOIN` since 0.9 | Half-open predicate (§6.3) preferred per Q-TEMPORAL-MAP-004 Option C; `ASOF JOIN ... ON probe >= valid_from` + trailing `WHERE (valid_to > probe OR valid_to IS NULL)` (§3.5.1) as alternative 🟡 | Native `ASOF JOIN ... ON probe >= snapshotted_at` 🟡 |
| Spark | No | Structural: `QUALIFY ROW_NUMBER() OVER (PARTITION BY entity ORDER BY anchor DESC) = 1 WHERE anchor <= probe` (§6.4; Spark 3.4+). Pre-3.4 uses subquery wrap. 🟡 | Same window-function rewrite. |
| Snowflake | Yes — `ASOF JOIN` since 2023 🟡 | Same posture as DuckDB (Q-TEMPORAL-MAP-004 Option C). 🟡 | Native `ASOF JOIN ... MATCH_CONDITION (probe >= snapshotted_at)` 🟡 |
| BigQuery | No | Structural `QUALIFY`-based rewrite (BigQuery supports `QUALIFY`) 🟡 | Same. |

#### 3.5.1 Trailing-closure rationale for `ScdWindow` on native-`ASOF` engines

Native `ASOF JOIN` in DuckDB and Snowflake accepts a **single inequality match condition** (e.g. `probe >= valid_from`), producing the most recent matching row. It does NOT enforce the `valid_to > probe` upper bound required by canonical `AsOfAnchor::ScdWindow` (`17 §5.1`). Two emission strategies:

1. **Native `ASOF JOIN` + trailing filter**: `ASOF JOIN ... ON probe >= valid_from` + `WHERE matched_row.valid_to > probe OR matched_row.valid_to IS NULL`. Captures engine-optimizer `ASOF` fast paths, but adds a filter.
2. **Half-open predicate on `Inner`/`Left` join**: `ON c.valid_from <= e.probe AND (c.valid_to > e.probe OR c.valid_to IS NULL)`. Single emission form across every engine.

Per Q-TEMPORAL-MAP-004 Option C, the adapter selects per anchor family: `ScdWindow` prefers the half-open predicate (§6.3); `SnapshotLatestAtOrBefore` prefers native `ASOF` where available (since the anchor has no upper bound to enforce). Tracked as `TD-TEMPORAL-ASOF-SCD-WINDOW-CLOSURE`.

---


## Testing Strategy

Scd shape behavioral gates:

- **Type0 append-only**: No special emission (universal).
- **Type1 MERGE**: DF/DuckDB/Spark/Snowflake/BigQuery support MERGE (universal overwrite form).
- **Type2 half-open predicate**: valid_from <= probe AND (valid_to > probe OR valid_to IS NULL) — canonical form (§3.4.2, universal).
- **Type3 prior-value column read**: Direct SELECT (no special join rewrite, universal).
- **Type4/5/6 DEFERRED**: Type4 (history table, 17 §10 D7), Type5 (Type4 + mini-dim, 17 §10 D9), Type6 (hybrid Type1+2+3, 17 §10 D10) — not yet canonical.
- **Sentinel-vs-NULL**: Round-1 NULL-aware only (per Q-TEMPORAL-MAP-007); '9999-12-31' sentinel DEFERRED (17 §10 D13).
- **Default-current lookup**: current_flag / valid_to IS NULL / ROW_NUMBER heuristic (§6.5).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
