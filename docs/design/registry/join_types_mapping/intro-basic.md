---
id: registry-join-types-mapping-intro-basic
status: ratified
version: 1.0
impl: implemented
---

# Join-Types Mapping — Intro & Basic Variants

This leaf introduces the join-types mapping catalog scope and engine roster, plus the four non-temporal canonical JoinType variants (Inner/Left/Right/Full) with universal per-engine SQL emission (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN), ON-vs-USING preference (always ON), RIGHT-rewrite posture (preserve orientation), and NATURAL JOIN avoidance.

## 1. Purpose and Scope

### 1.1 What this catalog ratifies

Per `16 §1.2`'s forward-refs to per-engine emission and `registry/README.md §versioning-and-churn`'s Living-catalog policy, this registry is the authoritative home for:

- **Per-engine `JoinType` emission.** How each target engine spells each canonical variant — the SQL keyword (`INNER JOIN`, `LEFT OUTER JOIN`, etc.), the ellipsis rules (`OUTER` optional vs required), and the emit-side preferences (`USING` vs `ON`, `NATURAL JOIN` avoidance).
- **`AsOf` rewrite-tier index, keyed by `JoinType`.** A summary cross-referencing `temporal_shape_mapping.md §4.2`'s authoritative per-(anchor, engine) classification. Useful when a reader's entry-point is `JoinType::AsOf` rather than a specific `TemporalShape`.
- **Reserved-variant roster.** Semi / Anti / Lateral / cross / natural — what semstrait does NOT canonicalize in v1, why, and how an author expresses the intent via ratified alternatives.
- **Cardinality-informed adapter hints.** When `Relationship.cardinality` declares `OneToMany` or `ManyToMany`, which engines admit `DISTINCT`-wrap / `collect_list` / array-aggregation hints the adapter MAY emit (outside of the fanout-safe rewrite owned by `16 §3.3.2`).
- **Gap catalog.** Per engine, the canonical `16` / `17` semantics the engine does NOT natively support, plus the workaround strategy.
- **Cross-engine idioms.** Patterns shared across adapters (windowed-ASOF emulation; lateral-join emulation via correlated subqueries) — written once, referenced per row.

### 1.2 What this catalog does NOT ratify

- **Canonical `JoinType` semantics.** `16 §4.2` is authoritative; this registry documents translation, not redefinition. A conflict between engine-native behavior and `16 §4.2` resolves in favor of `16` — the adapter emits the form that preserves canonical semantics.
- **`AsOf` per-shape legality.** `17 §5.2`'s shape-pair matrix is authoritative; this doc references it and documents emission for the legal pairs.
- **`AsOfAnchor` per-engine emission detail.** That lives in `temporal_shape_mapping.md §3.5` / `§4.2` / `§6.3`. `§3.5` of this doc carries a JoinType-first summary and cross-references the sibling catalog for the structural rewrites.
- **Per-adapter `RegistryExtension` of non-canonical joins.** Adapter crates may register engine-specific join variants (e.g. Spark's `LEFT ANTI`, Snowflake's `MINUS` as an anti-join form) as adapter-extended; inventory lives in each adapter's README per the `TD-JOIN-ADAPTER-INVENTORY` precedent (parallel to `functions_mapping.md §12`).
- **Planner choice between `Inner`, `Left`, `Right`, `Full`** for a given `Relationship` — `16 §4.2` plus author declaration decide; this registry renders whatever the planner hands the adapter.
- **Fanout-safe rewrite algorithm.** Owned by `16 §3.3.2` / `20 §…`; this registry documents the adapter-side hinting (DISTINCT etc.) that composes with the planner's rewrite, not the rewrite itself.

### 1.3 Engine-shortfall posture

Same three-tier response as `temporal_shape_mapping.md §1.3` (in decreasing order of preference):

1. **Structural rewrite** into engine-supported constructs preserving canonical semantics. Example: pre-1.1 DuckDB `FULL OUTER JOIN` emulated via `LEFT JOIN ... UNION ALL ... RIGHT JOIN ... WHERE left IS NULL`.
2. **Warning-plus-degradation** when canonical semantics are preserved but performance characteristics differ. Example: `AsOf` rewrite via `ROW_NUMBER()` on engines without native `ASOF JOIN`.
3. **`ADAPT_E_0302 UnsupportedFeature`** when no correct workaround exists. Reserved for future anchor families on engines that cannot express them; does not arise for `Inner` / `Left` / `Right` / `Full` on any first-class engine.

---

## 2. Engine Roster

Engine coverage mirrors `temporal_shape_mapping.md §2` for cross-catalog consistency. The three first-class compute targets ship in v1; additional engines land as columns / rows when their adapters ratify. Apache Calcite is listed as a **reference** for completeness of the join taxonomy — semstrait does not target Calcite as an execution backend, but Calcite's enumeration of `JoinRelType` variants is the widest industry superset and drives reserved-variant triage in `§3.6`.

| Engine | Version target | Adapter crate | Status | Notes |
|---|---|---|---|---|
| **DataFusion** | 40.x+ 🟡 | `semstrait-adapter-datafusion` (planned) | Round-1 target | Primary reference. All of `Inner / Left / Right / Full` native via `LogicalPlan::Join`; `AsOf` Structural-tier (see `temporal_shape_mapping.md §4.2`). |
| **DuckDB** | 1.1.x 🟡 | `semstrait-adapter-duckdb` (planned) | Round-1 target | All four non-temporal variants native. `FULL OUTER JOIN` landed in 0.3.x (well below floor). Native `ASOF JOIN` since 0.9. |
| **Spark** | 3.5.x 🟡 | `semstrait-adapter-spark` (planned) | Round-1 target | All four non-temporal native. `AsOf` Structural via `QUALIFY` + `ROW_NUMBER` (Spark 3.4+). Dialect preference: `LEFT` over `RIGHT` — see `§3.3.2`. |
| **Snowflake** | 2024.x 🟡 | `semstrait-adapter-snowflake` (planned) | Planned | All four native. Native `ASOF JOIN` since 2023 with `MATCH_CONDITION` syntax. |
| **BigQuery** | 2024.x 🟡 | `semstrait-adapter-bigquery` (planned) | Planned | All four native. No native `ASOF`; `AsOf` Structural via `QUALIFY`. |
| **Substrait** | 0.48+ 🟡 | `semstrait-adapter-substrait` (planned) | Planned | Plan-IR emission. `JoinRel.JoinType` enum covers `UNSPECIFIED / INNER / OUTER / LEFT / RIGHT / SEMI / ANTI / SINGLE / MARK`. `AsOf` emitted via `extension_uri`. |
| **Apache Calcite** | — | — | Reference only | `JoinRelType::INNER | LEFT | RIGHT | FULL | SEMI | ANTI` — cited as the widest mainstream taxonomy; used here to triage reserved-variant decisions in `§3.6`. |

Additional engines (ClickHouse, Trino, Iceberg-SQL-REST, Postgres as a reference dialect) add as columns to `§3` / `§4` / `§7` tables when their adapters land. Adding a column MUST NOT change canonical variants — if an engine lacks native support, the catalog documents the adapter's emulation strategy and files a `TD-JOIN-*` entry.

---

## 3. Per-`JoinType` Mapping

One subsection per canonical variant. Each subsection carries: (a) the canonical definition summary from `16 §4.2`, (b) a per-engine SQL-form table, (c) emission notes covering `USING` vs `ON`, `OUTER` keyword optionality, and any per-engine idioms.

### 3.1 `Inner`

**Canonical summary** (`16 §4.2`): produces rows with a match on **both** sides of the `KeyPair`. Non-matching rows on either side are dropped. Combined with `Cardinality::OneToOne` and non-NULL join columns, the canonical "joined table" semantics.

| Canonical | DataFusion | DuckDB | Spark | Snowflake | BigQuery | Substrait | Tier |
|---|---|---|---|---|---|---|---|
| `Inner` | `INNER JOIN` / `JOIN` | `INNER JOIN` / `JOIN` | `INNER JOIN` / `JOIN` | `INNER JOIN` / `JOIN` | `INNER JOIN` / `JOIN` | `JoinRel { join_type: INNER }` | **First-class** |

**Portability summary: Universal.** Every engine accepts the bare `JOIN` keyword as an alias for `INNER JOIN`.

**Emission notes.**

- **Canonical adapter emission uses `INNER JOIN`** (explicit keyword) for readability even though every first-class engine accepts the bare `JOIN` form. See Q-JOIN-MAP-001 for the explicit-vs-implicit debate; Round-1 posture is explicit.
- **`ON` vs `USING`.** All four SQL engines accept both. When every `KeyPair` in the `Relationship` has `left == right` (same `SemanticsName` on both sides), a `USING` emission is syntactically available and more compact. Round-1 posture: **always emit `ON` form** — the `USING` form masks key-column duplicates in the output schema (Postgres / Spark drop one copy; DuckDB / BigQuery collapse the pair into one column), which complicates downstream projection mapping. See Q-JOIN-MAP-002.
- **`NATURAL JOIN` is never emitted.** Implicit-column matching by name is hostile to semstrait's explicit-key-pairing model (`16 §2.3`'s `KeyPair` requires explicit `SemanticsName` pairing). `NATURAL JOIN` is not in this catalog at all.

### 3.2 `Left`

**Canonical summary** (`16 §4.2`): preserves all rows from `Relationship.from`; NULL-pads the `to` side for unmatched `from` rows. Combined with `ManyToOne`, the canonical "enrich facts with dim, keep unmatched facts" pattern.

| Canonical | DataFusion | DuckDB | Spark | Snowflake | BigQuery | Substrait | Tier |
|---|---|---|---|---|---|---|---|
| `Left` | `LEFT OUTER JOIN` / `LEFT JOIN` | `LEFT OUTER JOIN` / `LEFT JOIN` | `LEFT OUTER JOIN` / `LEFT JOIN` | `LEFT OUTER JOIN` / `LEFT JOIN` | `LEFT OUTER JOIN` / `LEFT JOIN` | `JoinRel { join_type: LEFT }` | **First-class** |

**Portability summary: Universal.** The `OUTER` keyword is optional on every SQL engine; the bare `LEFT JOIN` is universally accepted.

**Emission notes.**

- **Canonical adapter emission uses `LEFT JOIN`** (bare form). All five SQL engines treat this identically to `LEFT OUTER JOIN`. Explicit `OUTER` is legal but verbose; Round-1 preference is bare.
- **Reverse-walk emission** (per `16 §2.4.1`). When the planner walks a Relationship with derived `JoinType::Left` in reverse, the effective emission becomes `Right` (see `§3.3`). The adapter flips the keyword at emission time; the canonical-layer `Relationship.optional` value is unchanged. Traversal is always bidirectional in v1 (`16 §2.4`, post-2026-05-12).

### 3.3 `Right`

**Canonical summary** (`16 §4.2`): preserves all rows from `Relationship.to`; NULL-pads the `from` side for unmatched `to` rows. Less common in analytic workloads but symmetric with `Left`.

| Canonical | DataFusion | DuckDB | Spark | Snowflake | BigQuery | Substrait | Tier |
|---|---|---|---|---|---|---|---|
| `Right` | `RIGHT OUTER JOIN` / `RIGHT JOIN` | `RIGHT OUTER JOIN` / `RIGHT JOIN` | `RIGHT OUTER JOIN` / `RIGHT JOIN` | `RIGHT OUTER JOIN` / `RIGHT JOIN` | `RIGHT OUTER JOIN` / `RIGHT JOIN` | `JoinRel { join_type: RIGHT }` | **First-class** |

**Portability summary: Universal.** Every SQL engine supports `RIGHT [OUTER] JOIN` natively.

#### 3.3.1 Dialect-level preference for `LEFT` over `RIGHT`

Several engines recommend rewriting `RIGHT` to `LEFT` at the SQL layer for optimizer-plan consistency:

- **Spark SQL style guide** prefers `LEFT` over `RIGHT` on the grounds that `LEFT JOIN` reads left-to-right aligned with the FROM clause narrative. Spark's Catalyst optimizer normalizes `RIGHT` to `LEFT` internally (by swapping operands) — emitting `RIGHT` directly is semantically identical but inconsistent with the canonical rendering.
- **BigQuery** documentation notes `RIGHT JOIN` is "supported for ANSI compatibility" with a mild nudge toward `LEFT`.
- **DataFusion** accepts `RIGHT` natively; no documented preference.
- **DuckDB** accepts both; no documented preference.
- **Snowflake** accepts both; no documented preference.

**Round-1 posture**: semstrait emits `Right` verbatim as `RIGHT JOIN`. The adapter does NOT automatically rewrite to `LEFT` with swapped operands because:

1. The `Relationship.from` / `.to` orientation is canonical (`16 §2.2`); preserving it in emission preserves the author's intent across debugging / plan-introspection tooling.
2. Engine optimizers normalize internally; the emission form does not change execution cost on any first-class engine 🟡.
3. Authors who want `LEFT`-style narrative swap `from` and `to` at the `Relationship` level (and use `optional: left` to derive `JoinType::Left`).

Tracked as `TD-JOIN-RIGHT-REWRITE` (see Q-JOIN-MAP-003 for the opposite stance — automatic rewrite).

#### 3.3.2 Reverse-walk keyword flip

Per `16 §2.4.1`, a Relationship with derived `JoinType::Left` walked in reverse emits as `RIGHT JOIN` (the `to` side becomes the "left" in the FROM clause; the original `from` side becomes the "right" and is preserved). Symmetrically, derived `Right` walked in reverse emits as `LEFT JOIN`. `Inner` and `Full` are inversion-symmetric and require no keyword flip. The adapter's per-direction emission logic handles this flip; the canonical `Relationship.optional` value is unchanged. (Traversal is always bidirectional in v1 per `16 §2.4`; the authoring-layer `Directionality` enum was retired 2026-05-12.)

### 3.4 `Full`

**Canonical summary** (`16 §4.2`): preserves rows from both sides; NULL-pads whichever side lacks a match. Useful for union-like semantics that the shape of a `Unionset` would not express cleanly.

| Canonical | DataFusion | DuckDB | Spark | Snowflake | BigQuery | Substrait | Tier |
|---|---|---|---|---|---|---|---|
| `Full` | `FULL OUTER JOIN` / `FULL JOIN` | `FULL OUTER JOIN` / `FULL JOIN` (DuckDB ≥ 0.3.x) | `FULL OUTER JOIN` / `FULL JOIN` | `FULL OUTER JOIN` / `FULL JOIN` | `FULL OUTER JOIN` / `FULL JOIN` | `JoinRel { join_type: OUTER }` | **First-class** |

**Portability summary: Universal at current engine floors.** All five SQL engines at Round-1 pinned versions support `FULL [OUTER] JOIN` natively.

**Emission notes.**

- **Canonical adapter emission uses `FULL OUTER JOIN`** (explicit `OUTER`) because the bare `FULL JOIN` form is less recognizable in dialects that rarely see the full-outer pattern. The `OUTER` keyword is a readability aid; both forms are accepted.
- **Substrait.** Note that Substrait's `JoinRel.JoinType::OUTER` variant is the full-outer form (not a generic "outer" umbrella). `LEFT` and `RIGHT` are distinct variants; `OUTER` specifically means full-outer.

#### 3.4.1 Historical gaps — dialect-reference only

Engines outside the first-class Round-1 roster historically lacked `FULL OUTER JOIN`. Documented here for dialect-compatibility reference (not a Round-1 concern; semstrait does not emit to these engines in v1):

- **SQLite** — added `FULL OUTER JOIN` in 3.39 (2022-06-25). Pre-3.39 requires the `LEFT ... UNION ALL ... WHERE IS NULL` emulation.
- **MySQL** — no native `FULL OUTER JOIN` (through MySQL 8.x). The `LEFT ... UNION ... RIGHT ... WHERE IS NULL` emulation is the canonical workaround.
- **DuckDB pre-0.3.x** — missing; the adapter floor `1.1.x` is well above this.

When the adapter roster expands to include SQLite / MySQL / similar gap-bearing engines, the per-engine column gains an "emulation" row and `TD-JOIN-FULL-HISTORICAL-DIALECTS` tracks the required adapter scaffolding. The emulation pattern:

```sql
SELECT ... FROM left_table l LEFT JOIN right_table r ON <keys>
UNION ALL
SELECT ... FROM left_table l RIGHT JOIN right_table r ON <keys>
WHERE l.<any_key> IS NULL
```

This is a documented cross-engine idiom; see `§6.3`.

### 3.5 `AsOf` — companion summary

**Canonical summary** (`17 §5.1`): temporal-proximity join matching the most recent `to`-side row whose anchor satisfies the anchor condition relative to the `from`-side probe timestamp. Two anchor families in canonical v1: `AsOfAnchor::ScdWindow` (half-open `[valid_from, valid_to)` match) and `AsOfAnchor::SnapshotLatestAtOrBefore` (latest-`<=`-probe match). Vocabulary-ratified in `17 §5`, implementation-**DEFERRED** per `17 §10`.

**Authority boundary.** `temporal_shape_mapping.md §3.5` and `§4.2` are the **authoritative** per-(anchor, engine) emission tables. This section is a `JoinType`-indexed companion summary for readers who approach the question as "how does each engine render `JoinType::AsOf`" rather than "how does each engine render `TemporalShape × AsOfAnchor`". The two views of the same matrix must stay in sync; when they diverge, `temporal_shape_mapping` wins (per Q-JOIN-MAP-004 Option A).

#### 3.5.1 Rewrite-tier quick reference

🟡 pending `TD-JOIN-ASOF-EMPIRICAL` — parallel to `temporal_shape_mapping.md §4.2`'s `TD-TEMPORAL-ASOF-EMPIRICAL`.

| Anchor | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|
| `AsOfAnchor::ScdWindow` | Structural | First-class (half-open fallback preferred) 🟡 | Structural (`QUALIFY`-based, Spark 3.4+) 🟡 | First-class (half-open fallback preferred) 🟡 | Structural (`QUALIFY`-based) 🟡 |
| `AsOfAnchor::SnapshotLatestAtOrBefore` | Structural | First-class (`ASOF JOIN`) 🟡 | Structural (`QUALIFY`-based, Spark 3.4+) 🟡 | First-class (`ASOF JOIN ... MATCH_CONDITION`) 🟡 | Structural (`QUALIFY`-based) 🟡 |

See `temporal_shape_mapping.md §4.2` for the authoritative tier table with full notes and per-engine rationale, `§3.5.1` for the trailing-closure rationale on native-`ASOF` engines, `§6.3` for the canonical half-open-predicate emission form, and `§6.4` for the `QUALIFY`-vs-subquery-wrap structural-rewrite pattern.

#### 3.5.2 Round-1 emission status

Per `17 §5.1` and `17 §10`, the planner does not emit `JoinType::AsOf(anchor)` in Round 1; `JoinType` is derived at compile from `Relationship.optional` per `18 §2.9` with the v1 derivation roster staying `Inner | Left | Right | Full`. The `32` YAML surface carries `optional:`, not `join_type:` (post-2026-05-12 rebase). This registry captures the **emission target** adapters will converge on once planner-side `AsOf` activation lands. The cross-referenced `temporal_shape_mapping.md §3.5` / `§4.2` / `§6` rows are equally Round-1 draft; they carry the 🟡 marker pending adapter test-harness verification.

### 3.6 Reserved future variants — DEFERRED

Per `16 §4.3`, three variant families are **deferred from v1**. They appear in Apache Calcite's `JoinRelType` (the widest taxonomy baseline) and in various adapter layers; semstrait keeps them out of canonical until ratified demand surfaces. This subsection documents the reserved roster and the Round-1 alternative authors should use.

#### 3.6.1 `Semi` — DEFERRED

| Calcite | Substrait | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|---|
| `SEMI` | `SEMI` | `LeftSemi`, `RightSemi` (via `LogicalPlan::Join`) | `SEMI JOIN` (DuckDB ≥ 0.7) 🟡 | `LEFT SEMI JOIN` | `SEMI JOIN` / `EXISTS` subquery | `EXISTS` subquery only (no native `SEMI JOIN` keyword) |

**Canonical semantics.** Returns rows from the `from` side where **at least one** matching row exists on the `to` side, **without** duplicating `from` rows for multi-match `to` rows (unlike `Inner` with a fanout). Correctness-preserving optimization of `Inner` + `DISTINCT` on the `from`-side columns.

**Round-1 alternative** (per `16 §4.3`). Authors expressing semi-join intent write a `Filter` constraint referencing an `EXISTS`-style semantics, typically via a computed `Dimension` with a boolean expression. Equivalent result shape; different authoring surface.

**Disposition.** Tracked as `TD-COMPOSITION-SEMI-ANTI` in `16 §4.3`. A future v2 may introduce `JoinType::Semi` as a canonical variant with cross-engine structural rewrite (`EXISTS` subquery on engines lacking the keyword).

#### 3.6.2 `Anti` — DEFERRED

| Calcite | Substrait | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|---|
| `ANTI` | `ANTI` | `LeftAnti`, `RightAnti` | `ANTI JOIN` (DuckDB ≥ 0.7) 🟡 | `LEFT ANTI JOIN` | `ANTI JOIN` / `NOT EXISTS` | `NOT EXISTS` subquery only |

**Canonical semantics.** Returns rows from the `from` side where **no** matching row exists on the `to` side. Correctness-preserving optimization of `Left` + `WHERE right IS NULL`.

**Round-1 alternative.** `Filter` constraint with a `NOT EXISTS`-style expression, or an author-emitted `Left` join plus a filter on `to`-side NULL.

**Disposition.** Same `TD-COMPOSITION-SEMI-ANTI`.

#### 3.6.3 `Lateral` — DEFERRED (adapter-extended candidate)

| Calcite | Substrait | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|---|
| *(not in `JoinRelType`; expressed via `Correlate`)* | *(not a `JoinRel` variant; expressed via subquery correlation)* | *(via `LogicalPlan::SubqueryAlias` + correlation)* 🟡 | `LATERAL` subquery | `LATERAL VIEW` (array / map unnesting) / `LATERAL` subquery (Spark 3.4+) 🟡 | `LATERAL` subquery | `LATERAL` subquery (aka `CROSS JOIN UNNEST`) |

**Canonical semantics.** Row-wise correlated evaluation of a `to`-side subquery, producing zero-or-more rows per `from` row. Primary use-cases: array / struct unnesting, per-row derived subqueries, top-N-per-group pushdown.

**Round-1 stance.** Not a canonical `JoinType` variant; row-wise correlation does not fit `16 §4.2`'s pure-`JoinType` model (which assumes symmetric static operands). Authors needing lateral semantics in v1 either:

1. Emit unnesting explicitly through a `FunctionCall` with array / struct return type (per `14a §7` adapter-extended function registration), or
2. Fold the correlated computation into a `Dimension` / `Measure` `expr:` per `14` / `14a`.

**Disposition.** Tracked as `TD-JOIN-LATERAL-CANONICAL`. Potential adapter-extended registration per engine (Q-JOIN-MAP-005 Option A); no canonical promotion path in v1.

#### 3.6.4 Other variants in Calcite / Substrait — reference only

Calcite's `JoinRelType` is exhausted by `{INNER, LEFT, RIGHT, FULL, SEMI, ANTI}`. Substrait adds `SINGLE` (exactly-one-match) and `MARK` (boolean-indicator join for `EXISTS` rewriting). Both are planner-internal optimizations in their host systems — not author-facing join variants. semstrait does not canonicalize them; a future adapter MAY emit them as rewrites of `Inner` / `Left` + dedup / mark patterns. Tracked as `TD-JOIN-SUBSTRAIT-INTERNAL`.

---


## Testing Strategy

Basic join-types behavioral gates:

- **Inner/Left/Right/Full universal**: All four non-temporal variants emit correct SQL keyword on all engines (INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL OUTER JOIN universal at Round-1 pins).
- **ON vs USING**: Always ON form (USING never emitted per Q-JOIN-MAP-002 Option A).
- **RIGHT JOIN emitted verbatim**: Preserve author orientation; no automatic LEFT-with-swap rewrite (per Q-JOIN-MAP-003 Option A).
- **FULL OUTER JOIN universal at Round-1 pins**: All five SQL engines (DF/DuckDB/Spark/Snowflake/BigQuery) native; pre-0.3.x DuckDB historical emulation documented but not tested (floor is 1.1.x).
- **Reverse-walk keyword flip**: Left→Right, Right→Left when walked in reverse (per 16 §2.4.1); Inner/Full inversion-symmetric.
- **NATURAL JOIN never emitted**: Not in catalog.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
