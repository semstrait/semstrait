---
id: registry-join-types-mapping-cardinality
status: ratified
version: 1.0
impl: implemented
---

# Cardinality-Informed Emission, Idioms, and Gaps

This leaf specifies cardinality-informed emission (DISTINCT/array-aggregation hints for ManyToMany, opt-in only per Q-JOIN-MAP-006), cross-engine idioms (windowed-ASOF emulation via QUALIFY vs subquery-wrap, lateral-join emulation via correlated subquery, FULL OUTER emulation for historical dialects, DISTINCT vs GROUP BY), per-engine gap catalog, TD index, versioning, and interaction.

## 5. Cardinality-informed Emission

`Relationship.cardinality` (ratified in `16 §3`) carries analytic intent about the multiplicity of the join. When `cardinality ∈ {OneToMany, ManyToMany}`, the join may produce row-duplication that the author did not intend; `16 §3.3.2` / `16 §3.3.4` ratify the **planner-level** fanout-safe rewrite (pre-join aggregation). This section documents **adapter-level** emission hints that compose with the planner's rewrite — specifically, idioms for `DISTINCT` wrapping, array aggregation, and deduplication pins.

### 5.1 When the adapter MAY emit `DISTINCT`

The adapter MAY wrap the join output in a `SELECT DISTINCT ...` when:

1. `Relationship.cardinality == ManyToMany` AND
2. The planner's fanout-safe rewrite did NOT fire (the underlying `Relationship.cardinality == ManyToMany` is the structural trigger; the prior `PLAN_W_0502 ManyToManyFanoutAdvisory` flag was retired in `16 §14.4` (2026-04-29) per Q-COMP-005's intent-advisory deferral) AND
3. The composed Request projects only fields that are functionally dependent on the `Relationship.from` primary key.

**Per-engine support.** All five SQL engines natively support `SELECT DISTINCT`; Substrait emits via `AggregateRel` with empty measure list. This is Universal at Round-1 pins.

**Round-1 posture.** Per Q-JOIN-MAP-006 Option A, the adapter does **NOT** automatically emit `DISTINCT` — it is authoring-surface-affecting (changes row count) and risks masking genuine fanout bugs. Authors who want deduplication declare a `Joinset` with explicit deduplication semantics, or submit a Request whose projections trigger `SELECT DISTINCT` through explicit aggregation. `TD-JOIN-CARDINALITY-DISTINCT-AUTO`.

### 5.2 When the adapter MAY emit `collect_list` / array aggregation

For `Cardinality::OneToMany` where the Request wants per-`from`-row aggregation of the `to`-side, some engines admit a more compact `array_agg` / `collect_list` form than the planner-level fanout-safe rewrite:

| Engine | Array-aggregation function | Round-1 adapter posture |
|---|---|---|
| DataFusion | `array_agg(col)` 🟡 | Not emitted automatically. |
| DuckDB | `list(col)` / `array_agg(col)` | Not emitted automatically. |
| Spark | `collect_list(col)` / `collect_set(col)` | Not emitted automatically. |
| Snowflake | `array_agg(col)` / `array_agg(DISTINCT col)` | Not emitted automatically. |
| BigQuery | `ARRAY_AGG(col)` | Not emitted automatically. |

**Round-1 posture.** Same as `§5.1` — these are Semantic-changing projections (produce array-valued output). Only emitted when the author's `Measure` / `Metric` explicitly requests array-valued aggregation via a canonical function in `14a` (Round-2 candidate: `TD-FUNCS-MAPPING-ARRAY-AGG` — parallels to the array-valued-function landscape). The `JoinType` mapping does not drive this emission; it is documented here for cross-reference only.

### 5.3 Cardinality and `JoinType` composition table

Per `16 §12.4`'s soft-check list, certain `(Cardinality, JoinType)` pairs emit advisories. The adapter's emission is unchanged by the advisory — the `JoinType` SQL keyword remains identical regardless of `Cardinality` — but the adapter MAY surface per-engine optimizer hints when available:

| `Cardinality` × `JoinType` | Structural condition | Adapter-level emission hint |
|---|---|---|
| `OneToOne` + any | None (clean per `16 §3.3.1`) | — |
| `OneToMany` + `Inner` / `Left` | Fanout introduced; the prior `PLAN_W_0501 FanoutAdvisory` was retired in `16 §14.4` (2026-04-29). Authors are responsible for additivity choice. | Snowflake `/*+ BROADCAST(to_side) */`-style hint 🟡 when `to` side is known-small dimension; else none. |
| `ManyToOne` + `Inner` / `Left` | Symmetric fanout case; advisory retired alongside `PLAN_W_0501`. | Same broadcast hint opportunity for `from` side when dimension-like. |
| `ManyToMany` + any | Junction-table modeling is preferred (`16 §3.3.4`); the prior `PLAN_W_0502 ManyToManyFanoutAdvisory` was retired in `16 §14.4`. | Adapter MAY append `/*+ HINT */` per Q-JOIN-MAP-006; Round-1 posture: no automatic hint emission. |

**Engine-specific hint syntax** — reference only, NOT emitted in Round 1:

- **Spark**: `/*+ BROADCAST(t) */`, `/*+ SHUFFLE_HASH(t) */`, `/*+ MERGE(t) */`.
- **Snowflake**: query hints via session parameters; no inline hint syntax.
- **BigQuery**: no inline optimizer hints (optimizer is autonomous).
- **DuckDB**: no inline optimizer hints (optimizer is autonomous).
- **DataFusion**: no inline optimizer hints at the SQL layer (analyzer-pass machinery exists at the `LogicalPlan` layer but is not SQL-surface-exposed) 🟡.

Tracked as `TD-JOIN-CARDINALITY-HINTS`.

---

## 6. Cross-Engine Idioms

Patterns shared across multiple adapters. Written once here; `§3` / `§4` / `§7` rows reference these sections rather than inline the emission.

### 6.1 Windowed-`ASOF` emulation

The canonical Structural-tier emission for `AsOf` on engines without native `ASOF JOIN`. See `temporal_shape_mapping.md §6.2` / `§6.4` for the authoritative forms; summarized here for cross-reference.

**`QUALIFY`-variant** (DuckDB / Spark 3.4+ / Snowflake / BigQuery; preferred when available):

```sql
SELECT e.*, c.*
FROM events e LEFT JOIN customers_scd c
  ON e.customer_id = c.customer_id
 AND c.valid_from <= e.occurred_at
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY e.event_id ORDER BY c.valid_from DESC
) = 1
```

**Subquery-wrap variant** (DataFusion / Spark < 3.4 / ANSI fallback):

```sql
SELECT *
FROM (
  SELECT e.*, c.*,
    ROW_NUMBER() OVER (
      PARTITION BY e.event_id ORDER BY c.valid_from DESC
    ) AS rn
  FROM events e LEFT JOIN customers_scd c
    ON e.customer_id = c.customer_id
   AND c.valid_from <= e.occurred_at
) t
WHERE rn = 1
```

Both forms produce identical result sets; `QUALIFY` is shorter and avoids the explicit subquery scope. Adapter selection per `temporal_shape_mapping.md §6.4`.

### 6.2 Lateral-join emulation via correlated subquery

When lateral semantics are required on an engine with no native `LATERAL` (or when the adapter prefers not to emit `LATERAL` per `§3.6.3`'s reserved-variant posture), the universal emulation is a correlated subquery:

```sql
SELECT e.*, (
  SELECT c.segment
  FROM customers c
  WHERE c.id = e.customer_id
  ORDER BY c.valid_from DESC
  LIMIT 1
) AS segment
FROM events e
```

**Per-engine notes.**

- DataFusion / DuckDB / Spark / Snowflake / BigQuery all support correlated scalar subqueries with `LIMIT 1`.
- The correlation variable (`e.customer_id` inside the subquery) ties the subquery to each outer row — semantically equivalent to a `LATERAL` top-1 per group.
- Round-1 posture: the adapter does NOT emit this form automatically — authors declaring a `Relationship` with an `AsOfAnchor::SnapshotLatestAtOrBefore` get the Structural rewrite path of `§6.1` instead, which is uniform across engines.

### 6.3 `FULL OUTER` emulation via `LEFT UNION RIGHT`

For dialects without native `FULL OUTER JOIN` (historical SQLite, MySQL, older DuckDB) — reference only; not emitted at Round-1 pins:

```sql
SELECT l.*, r.* FROM t1 l LEFT JOIN t2 r ON l.k = r.k
UNION ALL
SELECT l.*, r.* FROM t1 l RIGHT JOIN t2 r ON l.k = r.k
WHERE l.k IS NULL
```

Correctness relies on: the `LEFT` branch contributing all `t1` rows with matching-or-NULL `t2` fields; the `RIGHT` branch contributing the `t2`-only rows with NULL `t1` fields. The `WHERE l.k IS NULL` predicate eliminates duplicates (rows that matched on both sides are captured by the `LEFT` branch). `UNION ALL` (not `UNION`) is correct here because the two branches are guaranteed disjoint.

### 6.4 `DISTINCT` vs `GROUP BY` post-join

When adapters need deduplication after a `ManyToMany` join (per `§5.1` opt-in), two emission forms are available:

```sql
SELECT DISTINCT <cols> FROM <join> WHERE ...
```

vs.

```sql
SELECT <cols> FROM <join> WHERE ... GROUP BY <cols>
```

Semantically identical when `<cols>` have no NULL values on any grouping column (since `GROUP BY` and `DISTINCT` have different NULL semantics on certain engines 🟡). All first-class engines accept both; Round-1 posture: `DISTINCT` form for readability.

---

## 7. Gap Catalog

Per-engine inventory of canonical `16` / `17` semantics the engine does NOT natively support for the `JoinType` axis, plus adapter workaround strategy. Parallels `temporal_shape_mapping.md §5`.

### 7.1 DataFusion gap inventory

| Canonical feature | DataFusion status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(*)` | No native `ASOF JOIN` | Structural rewrite per `temporal_shape_mapping.md §6.4` | `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE` (shared) |
| `JoinType::Semi` / `JoinType::Anti` | Natively supported at `LogicalPlan::Join` (`LeftSemi`, `LeftAnti`) but not canonical in v1 | Author uses `EXISTS` / `NOT EXISTS` filter; canonical v2 may promote. | `TD-COMPOSITION-SEMI-ANTI` (shared) |
| `LATERAL` subquery | Supported at `LogicalPlan::SubqueryAlias` 🟡; no SQL-surface keyword | N/A in Round 1; author emits computed Dimension. | `TD-JOIN-LATERAL-CANONICAL` |

**Round-1 posture**: DataFusion is First-class for `Inner / Left / Right / Full`; Structural for both `AsOf` anchor families.

### 7.2 DuckDB gap inventory

| Canonical feature | DuckDB status | Adapter workaround | TD |
|---|---|---|---|
| Historical `FULL OUTER JOIN` gap | Resolved in 0.3.x (well below 1.1.x floor) | — | — |
| `JoinType::Semi` / `JoinType::Anti` | Native `SEMI JOIN` / `ANTI JOIN` keywords 🟡 (DuckDB ≥ 0.7) | Not canonical in v1; if promoted, DuckDB row becomes First-class. | `TD-COMPOSITION-SEMI-ANTI` (shared) |
| `AsOfAnchor::ScdWindow` trailing closure | Native `ASOF JOIN` lacks upper-bound enforcement | Half-open predicate form per `temporal_shape_mapping.md §3.5.1` / Q-TEMPORAL-MAP-004 Option C | `TD-TEMPORAL-ASOF-SCD-WINDOW-CLOSURE` (shared) |

**Round-1 posture**: DuckDB is First-class for all four non-temporal variants AND for both `AsOf` anchor families.

### 7.3 Spark gap inventory

| Canonical feature | Spark status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(*)` | No native | `QUALIFY ROW_NUMBER() = 1` rewrite (3.4+) 🟡; subquery wrap pre-3.4 | `TD-TEMPORAL-ASOF-SPARK-NATIVE` (shared) |
| `RIGHT JOIN` dialect preference | Native but Catalyst normalizes to `LEFT` with operand swap | Emit `RIGHT JOIN` verbatim per `§3.3.1`; optimizer normalizes internally | `TD-JOIN-RIGHT-REWRITE` |
| `JoinType::Semi` / `JoinType::Anti` | Native `LEFT SEMI JOIN` / `LEFT ANTI JOIN` | Not canonical in v1 | `TD-COMPOSITION-SEMI-ANTI` (shared) |
| `LATERAL` (subquery form) | Spark 3.4+ 🟡 | Per-version adapter selection once canonical lateral lands | `TD-JOIN-LATERAL-CANONICAL` |

### 7.4 Snowflake gap inventory (planned adapter)

| Canonical feature | Snowflake status | Adapter workaround | TD |
|---|---|---|---|
| `AsOfAnchor::ScdWindow` | Native `ASOF JOIN ... MATCH_CONDITION` lacks upper bound | Half-open predicate per Q-TEMPORAL-MAP-004 Option C | `TD-TEMPORAL-ASOF-SCD-WINDOW-CLOSURE` (shared) |
| `RIGHT JOIN` | Native; no documented normalization preference | Emit verbatim | — |
| `SEMI` / `ANTI` keywords | Native | Not canonical in v1 | `TD-COMPOSITION-SEMI-ANTI` (shared) |

### 7.5 BigQuery gap inventory (planned adapter)

| Canonical feature | BigQuery status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(*)` | No native | `QUALIFY ROW_NUMBER() = 1` rewrite (BigQuery supports `QUALIFY` natively) | `TD-TEMPORAL-ASOF-SPARK-NATIVE` (structurally shared) |
| `SEMI JOIN` keyword | Absent (use `EXISTS` subquery) | Not canonical in v1; if promoted, Structural via `EXISTS` | `TD-COMPOSITION-SEMI-ANTI` (shared) |
| `ANTI JOIN` keyword | Absent (use `NOT EXISTS`) | Same posture | Same |

### 7.6 Substrait gap inventory (planned adapter)

| Canonical feature | Substrait status | Adapter workaround | TD |
|---|---|---|---|
| `JoinType::AsOf(anchor)` | No native `JoinRel.JoinType` variant | Emit via `extension_uri` with semstrait-specific URI; downstream consumers must understand the extension. | `TD-JOIN-SUBSTRAIT-ASOF-EXT` |
| Reserved variants (`SINGLE`, `MARK`) | Native `JoinRel.JoinType` variants | Not canonical; adapter MAY emit as rewrite of `Inner` + dedup / mark patterns in future. | `TD-JOIN-SUBSTRAIT-INTERNAL` |

---

## 8. Round-1 Open Items

Unresolved questions parked in [`questions/open/join_types_mapping_questions.md`](../../questions/open/join_types_mapping_questions.md). Summary:

| Q | Title | Round-1 position | Blocking? |
|---|---|---|---|
| Q-JOIN-MAP-001 | Explicit `INNER` keyword vs bare `JOIN` | Option A — explicit `INNER JOIN` | No |
| Q-JOIN-MAP-002 | `ON` vs `USING` emission form | Option A — always `ON` | No |
| Q-JOIN-MAP-003 | `RIGHT JOIN` auto-rewrite to `LEFT` with swapped operands | Option A — preserve author orientation, emit `RIGHT` verbatim | No |
| Q-JOIN-MAP-004 | `AsOf` tier-table authority (this doc vs `temporal_shape_mapping.md`) | Option A — `temporal_shape_mapping.md §4.2` authoritative; this doc mirrors | No |
| Q-JOIN-MAP-005 | `LATERAL` reserved-variant disposition | Option A — adapter-extended per engine; no canonical promotion | No |
| Q-JOIN-MAP-006 | Cardinality-informed `DISTINCT` / hint auto-emission | Option A — no automatic emission; opt-in via Request shape | No |
| Q-JOIN-MAP-007 | `FULL OUTER` emulation for historical dialects (SQLite, MySQL) | Deferred — not in Round-1 adapter roster | No |

None blocks ratification of this registry; all are coordination items with adapter implementation (`35` / `36`), `16 §4.3` deferrals, or `17 §10` deferrals shared with `temporal_shape_mapping.md`.

---

## 9. Versioning

Following `registry/README.md §versioning-and-churn` and sibling-catalog precedent (`functions_mapping.md §15`, `types_mapping.md §4`, `temporal_shape_mapping.md §8`), each mapping row SHOULD cite the engine version it was verified against. Round-1 tentative pins align with `temporal_shape_mapping.md §8`:

| Engine | Target version(s) | Notes |
|---|---|---|
| DataFusion | 40.x+ 🟡 | Matches sibling catalogs. All four non-temporal variants stable since early DataFusion releases. |
| DuckDB | 1.1.x 🟡 | `FULL OUTER JOIN` since 0.3.x (well below floor); `ASOF JOIN` since 0.9. |
| Spark | 3.5.x 🟡 | `QUALIFY` requires 3.4+; `LATERAL` subquery (if ever canonicalized) requires 3.4+. |
| Snowflake | 2024.x 🟡 | `ASOF JOIN` GA since 2023; `MATCH_CONDITION` syntax stable. |
| BigQuery | 2024.x 🟡 | `QUALIFY` long-available; no version-floor concerns for the four non-temporal variants. |
| Substrait | 0.48+ 🟡 | `JoinRel.JoinType` stable; `extension_uri` mechanism stable. |

Rows citing features behind a specific engine version carry `(Engine X.Y+)` inline. Unverified rows are marked 🟡; as adapter implementation lands and verifies, the 🟡 marker is removed and the exact verified version replaces the range. Breaking changes between engine majors are documented as additional rows or dated annotations — not destructive edits — per `registry/README.md §versioning-and-churn`.

---

## 10. TECH_DEBT Index

Consolidated list of all `TD-JOIN-*` entries surfaced in this catalog plus shared `TD-COMPOSITION-*` / `TD-TEMPORAL-*` entries that this catalog references. Entries retire as adapter implementation, `16` / `17` ratification, or engine-landscape shift resolves them.

| TD ID | § | Canonical concern | Engine(s) | Current posture |
|---|---|---|---|---|
| `TD-JOIN-RIGHT-REWRITE` | `§3.3.1`, `§7.3` | Automatic `RIGHT` → `LEFT`-with-swap rewrite | Spark (Catalyst preference) | Open — Round-1 posture: preserve orientation. Revisit if empirical perf divergence surfaces. |
| `TD-JOIN-FULL-HISTORICAL-DIALECTS` | `§3.4.1`, `§6.3` | `FULL OUTER` emulation scaffolding | SQLite pre-3.39, MySQL, others | Deferred — not in Round-1 roster. |
| `TD-JOIN-LATERAL-CANONICAL` | `§3.6.3`, `§7.1`, `§7.3` | Canonical `Lateral` join variant | All | Open — adapter-extended candidate per Q-JOIN-MAP-005. |
| `TD-JOIN-SUBSTRAIT-INTERNAL` | `§3.6.4`, `§7.6` | Substrait-only `SINGLE` / `MARK` join variants | Substrait | Open — potential adapter-side rewrite target for `Inner` + dedup. |
| `TD-JOIN-SUBSTRAIT-ASOF-EXT` | `§7.6` | Substrait `AsOf` emission via `extension_uri` | Substrait | Open — blocked on planner `AsOf` support per `17 §10`. |
| `TD-JOIN-ASOF-EMPIRICAL` | `§3.5.1`, `§4.2` | Rewrite-tier rows drafted from docs, not empirically verified | All | Shared with `TD-TEMPORAL-ASOF-EMPIRICAL` — blocked on test harness. |
| `TD-JOIN-CARDINALITY-DISTINCT-AUTO` | `§5.1` | Auto-emit `SELECT DISTINCT` on `ManyToMany` | All | Open — Round-1: no auto-emission; author-driven only. |
| `TD-JOIN-CARDINALITY-HINTS` | `§5.3` | Engine-specific optimizer hints (Spark `BROADCAST`, etc.) | Spark (primary); others lack inline syntax | Open — Round-1 posture: no hint emission. |
| `TD-COMPOSITION-SEMI-ANTI` | `§3.6.1`, `§3.6.2`, `§7.*` | `JoinType::Semi` / `JoinType::Anti` canonical promotion | All | Shared with `16 §4.3`. Deferred to v2. |
| `TD-COMPOSITION-ASOF` | `§3.5` | `JoinType::AsOf` canonical variant | All | Shared with `16 §4.3`. Vocabulary-ratified in `17 §5`; planner implementation DEFERRED per `17 §10`. |
| `TD-TEMPORAL-ASOF-DATAFUSION-NATIVE` | `§4.2`, `§7.1` | Native `ASOF JOIN` on DataFusion | DataFusion | Shared with `temporal_shape_mapping.md §7`. |
| `TD-TEMPORAL-ASOF-SPARK-NATIVE` | `§4.2`, `§7.3` | Native `ASOF JOIN` on Spark | Spark | Same. |
| `TD-TEMPORAL-ASOF-SCD-WINDOW-CLOSURE` | `§3.5`, `§7.2`, `§7.4` | Half-open predicate preferred on First-class-tier engines | DuckDB, Snowflake | Same. |

---

## 11. Interaction with Other Documents

- **`foundations/16_composition.md`** — canonical upstream. `16 §4` defines the `JoinType` enum (`Inner`, `Left`, `Right`, `Full`); `16 §4.3` defers `Semi` / `Anti` / `AsOf`; `16 §4.4` ratifies `PlanNode::Join` carriage. `16` never depends on this registry's specifics (`registry/README.md §versioning-and-churn`).
- **`foundations/17_temporal_shape.md`** — authoritative for the `JoinType::AsOf(AsOfAnchor)` extension: vocabulary (`17 §5.1`), per-shape-pair legality (`17 §5.2`), cardinality implications (`17 §5.3`), reverse-direction forbiddance (`17 §5.3`), and implementation-DEFERRED status (`17 §10`).
- **`registry/README.md`** — shared policy (engine coverage, versioning, Living status).
- **`registry/types_mapping.md`** — sibling. `KeyPair` type-agreement (per `16 §12.2`) consumes this catalog's type mappings at `compile`; this doc assumes `types_mapping.md §2`'s widening rules as background.
- **`registry/functions_mapping.md`** — sibling. `ROW_NUMBER()`, comparison operators, and `NULLIF`-style constructs consumed by `AsOf` structural rewrites (`§3.5`, `§6.1`). `QUALIFY` is an SQL-clause-level construct (not a function); its per-engine version-floor story is owned by this doc's `§2` and by `temporal_shape_mapping.md §5.3`.
- **`registry/temporal_shape_mapping.md`** — sibling and authoritative co-owner of the `AsOf` rewrite-tier table (`§4.2` there; `§4.2` here is a summary). When the two views diverge, `temporal_shape_mapping.md` wins per Q-JOIN-MAP-004 Option A.
- **`apis/35_semstrait_ir.md`** — `PlanNode::Join.join_type: JoinType` carriage per `16 §4.4`; `AsOf(AsOfAnchor)` payload round-tripping per `17 §5.4` (DEFERRED).
- **`apis/36_semstrait_adapter.md`** — the `EngineAdapter` trait + PlanBuilder / Dialect layering that consumes this registry. Per-adapter crates ratify their own emission paths; `§3` / `§4` / `§7` rows seed each adapter's implementation.
- **Adapter crates** (future `semstrait-adapter-datafusion`, `-duckdb`, `-spark`, `-snowflake`, `-bigquery`, `-substrait`) — own authoritative per-engine emission tables and adapter-extended joins (reserved-variant registrations, optimizer hints).
- **`questions/open/join_types_mapping_questions.md`** — parked unresolved questions surfaced during Round-1 drafting.

## Testing Strategy

Cardinality and gap behavioral gates:

- **Cardinality-informed DISTINCT NOT auto-emitted**: Opt-in only (ManyToMany per Q-JOIN-MAP-006 Option A: adapter MAY emit DISTINCT when author opts in, never automatically).
- **Cross-engine idioms**: Windowed-ASOF emulation (QUALIFY vs subquery-wrap §6.1), lateral-join emulation (correlated subquery §6.2), FULL OUTER emulation for historical dialects (LEFT+RIGHT UNION §6.3), DISTINCT vs GROUP BY (§6.4).
- **Per-engine gaps documented**: §7 DF AsOf Structural, DuckDB ScdWindow trailing closure, Spark AsOf Structural QUALIFY, Spark RIGHT-rewrite posture (TD entries ratified).
- **Versioning pins cited**: §9 DF 40.x+ / DuckDB 1.1.x / Spark 3.5.x / Snowflake 2024.x / BigQuery 2024.x / Substrait 0.48+.
- **TD index complete**: §10 consolidates all TD-JOIN-* entries with originating § and current posture.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
