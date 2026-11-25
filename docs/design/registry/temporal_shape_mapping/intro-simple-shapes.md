---
id: registry-temporal-shape-mapping-intro-simple
status: ratified
version: 1.0
impl: implemented
---

# Temporal-Shape Mapping — Intro & Simple Shapes

This leaf introduces the temporal-shape mapping catalog scope and engine roster, plus Timeseries and Events (simple shapes) with universal per-engine emission via date_trunc + GROUP BY, gap-filling strategies (generate_series/range/sequence per engine), and MATCH_RECOGNIZE explicitly out-of-scope.

## 1. Purpose and Scope

### 1.1 What this catalog ratifies

Per `17 §1.2`'s forward-refs and `registry/README.md §versioning-and-churn`'s Living-catalog policy, this registry is the authoritative home for:

- **Per-engine temporal-shape emission.** How each target engine naturally expresses each canonical `TemporalShape` variant — function calls, syntactic constructs, or workaround predicates when the engine lacks a native idiom.
- **`AsOf` rewrite-tier classification.** Per-(anchor, engine) pair: does the engine admit native `ASOF JOIN` (First-class tier), require structural rewrite to window-functions + predicates (Structural tier), or lack any emission path (Unsupported tier)?
- **Gap catalog.** Per engine, the canonical `17`-ratified semantics the engine does NOT natively support, plus the workaround strategy the adapter uses.
- **Cross-engine idioms.** Patterns shared across multiple adapters (WHERE-max snapshot selection; windowed ROW_NUMBER as-of; half-open valid-window predicate) — written once, referenced from each adapter's row.

### 1.2 What this catalog does NOT ratify

- **Canonical semantics.** Every conflict between an engine-native form and the canonical `17` specification resolves in favor of the canonical form; this registry documents the translation, not a redefinition. Per `registry/README.md §versioning-and-churn`: "canonical specs under `foundations/` may reference the registry but never depend on its specific contents."
- **Planner algorithm for `Request.temporal`.** The multi-shape resolution algorithm (`17 §6.5` DEFERRED; `17 §10 D4, D5`) lands in `34 §…`. This registry documents the per-engine emission the planner will produce; it does not describe how the planner chooses between shapes.
- **Adapter-extended temporal idioms.** Engines expose temporal features outside the canonical `17` roster — Snowflake's `BEFORE (STATEMENT => ...)`, BigQuery's `MATCH_RECOGNIZE`, Iceberg branch-aware reads, DuckDB's `time_bucket()`. These are adapter-extended per Q-TEMPORAL-MAP-008 (Option B adopted: defer to adapter-crate READMEs). `§3.2.1` calls out the `MATCH_RECOGNIZE` exclusion explicitly; the rest live in each adapter's own documentation.
- **Sentinel-value conventions for `valid_to`.** `17 §10 D13` reserves future work to ratify author-declared sentinels (`'9999-12-31'`, etc.); until then this catalog emits `NULL`-aware predicates and documents the sentinel forms adapters encounter (§3.4.3).

### 1.3 Engine-shortfall posture

When an engine lacks native support for a canonical variant, this catalog documents three response strategies in decreasing order of preference:

1. **Structural rewrite** into a combination of engine-supported constructs that preserves canonical semantics. Example: DataFusion `AsOf(ScdWindow)` → `Inner` join + `FilterNode(valid_from <= probe AND (valid_to > probe OR valid_to IS NULL))` (§6.3).
2. **Warning-plus-degradation** when canonical semantics cannot be fully preserved but a reasonable approximation exists. Example: Spark < 3.5 `ASOF` rewrite via subquery-wrapped `ROW_NUMBER()` with a warning about performance (§6.4). Degradation must preserve correctness; only performance characteristics differ.
3. **`ADAPT_E_0302 UnsupportedFeature`** when the gap is fundamental and no correct workaround exists. Documented per engine in §5 with the canonical variant that triggers it.

---

## 2. Engine Roster

Engine coverage follows `registry/README.md §engine-coverage-policy`. The three first-class compute targets ship in v1; additional engines land as columns / rows when their adapters ratify.

| Engine | Version target | Adapter crate | Implementation status | Notes |
|---|---|---|---|---|
| **DataFusion** | 40.x+ 🟡 | `semstrait-adapter-datafusion` (planned) | Round-1 target | Primary reference. No native `ASOF JOIN`; structural rewrites only. No Iceberg / Delta time-travel in vanilla DF. |
| **DuckDB** | 1.1.x 🟡 | `semstrait-adapter-duckdb` (planned) | Round-1 target | Native `ASOF JOIN` since 0.9; Iceberg / Delta extensions opt-in (Q-TEMPORAL-MAP-002). SQL-idiom reference. |
| **Spark** | 3.5.x 🟡 | `semstrait-adapter-spark` (planned) | Round-1 target | No native `ASOF JOIN`; structural rewrites via `QUALIFY` + `ROW_NUMBER` (Spark 3.4+). Delta time-travel via `VERSION AS OF` / `TIMESTAMP AS OF`. |
| **Snowflake** | 2024.x 🟡 | `semstrait-adapter-snowflake` (planned) | Planned | Native `ASOF JOIN` since 2023. Time-travel via `BEFORE (STATEMENT => ...)` / `AT (TIMESTAMP => ...)`. |
| **BigQuery** | 2024.x 🟡 | `semstrait-adapter-bigquery` (planned) | Planned | No native `ASOF JOIN`; structural rewrites. `FOR SYSTEM_TIME AS OF` for snapshot / time-travel. `MATCH_RECOGNIZE` (adapter-extended, §3.2.1). |
| **Substrait** | 0.48+ 🟡 | `semstrait-adapter-substrait` (planned) | Planned | Plan-IR emission. `JoinRel` carries canonical `JoinType`; `AsOf` emitted via Substrait extensions (`extension_uri`). |

Additional engines (ClickHouse, Trino, Iceberg-SQL-REST) add as columns to §3 / §4 / §5 tables when their adapters land. Adding a column MUST NOT change canonical variants — if an engine lacks native support, the catalog documents the adapter's emulation strategy and files a `TD-TEMPORAL-*` entry.

---

## 3. Shape-by-Shape Mapping

One subsection per canonical `TemporalShape` variant. Each subsection carries: (a) the canonical definition summary from `17`, (b) a per-engine mapping table, (c) emission notes for workaround patterns.

### 3.1 `Timeseries`

**Canonical summary** (`17 §2.1`): dense regularly-spaced observations at a known cadence `grain`. Rolls up to any `G' >= grain` via `DateTrunc` bucketing; sub-grain requests are `PLAN_E_1710` (`17 §9.3`).

| Canonical emission | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|
| Rollup bucket | `date_trunc('grain', occurred_at)` | `date_trunc('grain', occurred_at)` / `time_bucket(INTERVAL 'N unit', occurred_at)` 🟡 | `date_trunc('grain', occurred_at)` / `window(occurred_at, 'N unit')` 🟡 | `date_trunc('grain', occurred_at)` | `timestamp_trunc(occurred_at, GRAIN)` |
| Gap-filling (dense → dense) | `generate_series(...)` + `LEFT JOIN` 🟡 | `range(...)` / `generate_series(...)` + `LEFT JOIN` | `sequence(start, end, interval)` + `explode` + `LEFT JOIN` 🟡 | `generator(TIMESTAMPDIFF(...))` 🟡 | `GENERATE_TIMESTAMP_ARRAY(...)` + `UNNEST` 🟡 |
| Per-grain aggregation | `GROUP BY date_trunc(...)` | `GROUP BY date_trunc(...)` / `GROUP BY time_bucket(...)` | `GROUP BY date_trunc(...)` / `window()` | `GROUP BY date_trunc(...)` | `GROUP BY timestamp_trunc(...)` |

**Portability summary: Universal** — every engine emits via `date_trunc` + `GROUP BY`. The registry's `date_trunc` entry in `functions_mapping.md §11` covers the per-engine function-mapping detail; this row just notes that `Timeseries` shape itself adds nothing exotic. DuckDB's `time_bucket` and Spark's `window()` are adapter-extended alternatives preferred for irregular or offset buckets (e.g. "weeks starting Sunday"); they live outside canonical `17` per `17 §1.2`.

**Gap-filling as adapter concern.** `17 §2.1` characterizes `Timeseries` as "dense"; adapters reading a sparse source that declares `Timeseries` shape MAY emit gap-filling scaffold (per engine row) to satisfy the density invariant. `PLAN_W_17xx ShapeAwareRequestOnUnclassifiedDataKind` does not apply — the source is classified; the planner's gap-fill pass is DEFERRED per `17 §10`.

### 3.2 `Events`

**Canonical summary** (`17 §2.1`): sparse discrete occurrences at `occurred_at_dim`. No density assumption; any `grain ∈ occurred_at_dim.grains` is a legal bucket.

| Canonical emission | DataFusion | DuckDB | Spark | Snowflake | BigQuery |
|---|---|---|---|---|---|
| Bucket-then-aggregate | `GROUP BY date_trunc('grain', occurred_at)` | `GROUP BY date_trunc('grain', occurred_at)` | `GROUP BY date_trunc('grain', occurred_at)` | `GROUP BY date_trunc('grain', occurred_at)` | `GROUP BY timestamp_trunc(occurred_at, GRAIN)` |
| Sub-day grain (`hour`, `minute`) | `date_trunc('hour', occurred_at)` | same | same | same | `timestamp_trunc(occurred_at, HOUR)` |
| Filtering by window | `occurred_at >= :start AND occurred_at < :end` | same | same | same | same |

**Portability summary: Universal.** `Events` is the simplest shape — every engine supports `GROUP BY` on a bucket expression. No rewrite-tier concerns; no TD entries originate from this row.

#### 3.2.1 `MATCH_RECOGNIZE` and pattern-matching idioms — explicitly out of scope

Several engines (BigQuery, Snowflake, Trino) expose `MATCH_RECOGNIZE` for sequential-pattern event detection. This is **out of canonical scope** per `17 §1.2`'s forward-ref stance — canonical `Events` expresses "rows exist where something happened" at the aggregation level, not row-sequence pattern matching. `MATCH_RECOGNIZE` is adapter-extended on engines that support it and lives in each adapter crate's README, per Q-TEMPORAL-MAP-008 Option B. `TD-TEMPORAL-MATCH-RECOGNIZE` tracks a possible future canonical `EventPattern` variant if author demand materializes.


## Testing Strategy

Simple temporal-shapes behavioral gates:

- **Timeseries universal**: date_trunc + GROUP BY on all engines (universal).
- **Gap-filling strategies per engine**: generate_series (DF/DuckDB), sequence+explode (Spark), GENERATE_TIMESTAMP_ARRAY+UNNEST (BigQuery). Each engine's native idiom tested.
- **Events universal**: date_trunc bucket-then-aggregate (no density assumption, universal).
- **MATCH_RECOGNIZE explicitly out-of-scope**: Adapter-extended per Q-TEMPORAL-MAP-008; semstrait does NOT canonicalize pattern-matching temporal idioms (Snowflake/BigQuery MATCH_RECOGNIZE, Spark 3.4+ MATCH_RECOGNIZE, DF/DuckDB absence).

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
