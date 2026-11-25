---
id: registry-readme
status: ratified
version: 1.0
impl: none
---

# Registry

The `docs/design/registry/` folder holds **catalog documents**: authoritative per-engine mappings of canonical semstrait primitives (data types, functions, temporal shapes, join types, ...) to their engine-native counterparts in DataFusion, Spark, DuckDB, and future targets.

## Why a separate folder

Design documents under `foundations/`, `data-kinds/`, and `apis/` define **canonical specifications** — what `DataType`, `CanonicalFn`, `Grain`, etc. *are* and how they compose. Those documents stabilize once ratified; they change rarely.

Engine mappings are **living catalogs**. They grow every time we add an engine, extend a function, discover an engine-specific edge case, or resolve a cast-semantics detail. Keeping them in the same documents as the canonical specs would force unrelated churn on foundational docs every time an adapter is extended.

The registry solves this with a clean separation:

- **`foundations/13_types_and_grain.md`** — ratifies the canonical `DataType` set, YAML grammar, and shape-unification rules. Stable.
- **`registry/types_mapping.md`** — documents how each canonical `DataType` maps to each target engine. Extensible; updated whenever an engine is added or mapping detail is refined.

## Current contents

| Document | Scope | Status |
|---|---|---|
| [`types_mapping.md`](types_mapping.md) | Canonical `DataType` ↔ DataFusion / Spark / DuckDB native types, cast semantics, per-engine gaps | Stub pending ratification of adapter surface (`34` / `36` / `37`) |
| [`functions_mapping.md`](functions_mapping.md) | Canonical function catalog ↔ engine function names, rewrite tiers, arity/signature differences, aliases, rewrite rules | Draft — tracks `foundations/14a_function_catalog.md` |
| [`temporal_shape_mapping.md`](temporal_shape_mapping.md) | Per-engine mapping of `TemporalShape` variants (Timeseries, Events, Snapshot, SCD). v1 `ScdType` roster is `{Type1, Type2}` per `foundations/18_entities.md §3.3`; SCD `Type0`/`Type3`/`Type4`/`Type5`/`Type6` are post-v1 deferred. The `AsOf` rewrite-tier matrix is documented as forward-reference (v1 `JoinType` roster excludes `AsOf`). | Draft — tracks `foundations/17_temporal_shape.md` + `foundations/18_entities.md §3` |
| [`join_types_mapping.md`](join_types_mapping.md) | Canonical `JoinType` ↔ engine-native join syntax. v1 roster: `{Inner, Left, Right, Full}` per `foundations/18_entities.md §2`. `AsOf` support matrix is forward-reference / post-v1 deferred. Cardinality-informed emission rules unchanged. | Draft — tracks `foundations/16_composition.md` + `foundations/18_entities.md §2` + `apis/35_semstrait_ir.md` |

## Engine coverage policy

Every registry catalog must cover the three **first-class compute targets**:

- **DataFusion** — primary implementation target; first adapter to ratify.
- **Spark** — SQL dialect and typed DataFrame target; drives lowest-common-denominator decisions (e.g. no native `Time`).
- **DuckDB** — reference warehouse-SQL dialect; drives SQL-idiom decisions.

Additional engines (future: Substrait-native, ClickHouse, BigQuery, Snowflake, Trino) are added as columns to the existing tables when their adapters land. Adding an engine column MUST NOT change the canonical set — if an engine lacks native support for a canonical variant, the catalog documents the adapter's emulation strategy and files a TECH_DEBT entry (see `TD-ADAPTER-SPARK-TIME` in `foundations/13` as a worked example).

## Versioning and churn

Registry documents are **Living**: they track adapter implementation reality. Each entry SHOULD cite the adapter crate and version where it was verified (e.g. "DuckDB 1.1.x"). Breaking changes in an engine (a rename, a semantics change between engine major versions) are documented as additional rows or dated annotations, not destructive edits.

Canonical specs under `foundations/` may reference the registry but never depend on its specific contents. A canonical variant is defined by the semantic it carries, not by any single engine's mapping.

## Open questions

Open items surfaced against any registry catalog live in the corresponding `questions/open/<catalog>_questions.md` sidecar. For a one-file view of every registry open question across all catalogs, see [`../questions/open/registry_questions.md`](../questions/open/registry_questions.md) — a navigation-only aggregate index. The per-sidecar files remain authoritative for the detailed bodies.

## Testing Strategy

- Run `doc-space-check` for the structural checks (the `semstrait-doc`
  MAINTAINER procedure that used to own them was removed when this tree was
  frozen — see `AGENTS.md §2`; the script is now a drift guard).
- Review every mapping row for a cited engine version.
- Review that catalog changes map existing canonical variants and link their
  foundation owners; they must not define new canonical semantics.

See [../testing/50_behavioral_contract_gates.md](../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
