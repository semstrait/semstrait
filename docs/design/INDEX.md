# semstrait Design — Index

> Scan-optimized, topic-first index into the `docs/design/` tree. Read `00_overview.md` first if you're new; come here when you already know what concept you're hunting for. Every link below lands on the **single source of truth** for that concept — never a duplicate.

**Precedence rule — the one thing to know.** If two docs appear to disagree on the shape, roster, or spelling of a concept, the doc whose YAML front matter lists that concept under `authoritative-for:` wins. `00_overview.md §4.4` states the full precedence rule; `§8` states the directionality rule that prevents conflicts from arising in the first place.

---

## How to use this index

- **You know the concept name** — jump to the alphabetical table below, follow the link to the single canonical home, then read the surrounding `§` for full context.
- **You know the pipeline stage** — skim `00_overview.md §5` (pipeline diagram) and the per-stage contract in `foundations/10_resolution_pipeline.md`.
- **You know the YAML surface** — start at `apis/32_semstrait_model.md` (root) / `apis/32b_catalogs_yaml.md` (catalogs) and follow the per-variant pointers into `data-kinds/2x`.
- **You need to see what's open** — start at the [open-questions](#open-questions) section; every sidecar is listed with its scope.
- **You're auditing for duplication** — the [duplication-guard table](#duplication-guard) lists every concept the 2026-04-17 consolidation pass ratified as owned by exactly one doc, with a pointer to that doc.

## Folder map (compact)

```
docs/design/
├── 00_overview.md           ← master doc; read first; vocabulary, invariants, map
├── STATUS.md                ← session-handoff + phase map + decision snapshots
├── INDEX.md                 ← this file (scan-optimized topic index)
├── foundations/             ← 10 … 19 : cross-cutting rules, entity shapes, category axis
├── data-kinds/              ← 20 … 26 : per-variant specs (Dataset, Grainset, Unionset, Joinset) + nesting matrix
├── apis/                    ← 30 … 39 : per-crate API contracts
├── implementation/          ← 40 … 42 : refactor plan, deprecations, migration notes
├── registry/                ← living catalogs: per-engine mappings
└── questions/               ← parked / deferred items (one sidecar per parent doc + aggregates)
    ├── open/                ←   files with at least one unresolved Q-ID
    └── closed/              ←   files whose Q-IDs are all resolved (historical record)
```

See `00_overview.md §6` for the full numbering convention, sibling-suffix rule (`14a`, `14b`), and the per-folder responsibility tables.

---

## Topic → canonical home (alphabetical)

Every row points to **one** canonical doc. Cross-references in the right column flag other docs that specialize or extend the concept for a narrower scope, never redefine it.

| Topic | Canonical home | Refined / specialized in |
|---|---|---|
| `AdditivityType` (`Additivity`) | `foundations/18_entities.md §5.2` (enum roster) + `foundations/11_names_and_scopes.md §8` (planner contract) | `17 §7` (advisory warnings when inconsistent with `TemporalShape`) |
| `AiContext` | `foundations/18_entities.md §8` | — |
| `AsOf` joins (post-v1) | `foundations/17_temporal_shape.md §5` (design forward-reference) | `registry/join_types_mapping.md §4.2`, `registry/temporal_shape_mapping.md §4.2` (engine mapping) |
| `Binding` (compile-time process) | `foundations/15_mapping_and_binding.md` | `apis/33_semstrait_manifest.md §5.3` (Manifest-layer `ResolvedColumnMapping` surface) |
| `Cardinality` | `foundations/18_entities.md §2.4` (struct) + `foundations/16_composition.md` (planner semantics) | — |
| `CatalogEntry` / `CatalogAuthMethod` / `CatalogRef` | `apis/32b_catalogs_yaml.md` | — |
| `CatalogProvider` | `apis/37_semstrait_catalog.md` | — |
| `CanonicalFn` + `FunctionRegistry` | `foundations/14a_function_catalog.md` | `registry/functions_mapping.md` (per-engine rewrite) |
| `ComposedSemanticInterface` | `foundations/16_composition.md` | `data-kinds/2x` (per-variant specialization) |
| `Constraint` (planner-time) | `foundations/11_names_and_scopes.md §8` (post-rewrite — Measure / Metric carriers; explicit refinement on top of category-implicit rules) | `foundations/19_categories.md §5` (implicit-vs-explicit contract); `questions/closed/11_constraints_deferred.md` (closed Q-R4.3a–d) |
| `DataKind` taxonomy + sealed trait hierarchy | `data-kinds/20_taxonomy.md` + `apis/32_semstrait_model.md §3` | `data-kinds/21`–`24` (per-variant) |
| `DataKindBase` + variant `*Body` structs | `apis/32_semstrait_model.md §3` | — |
| `DataKindFilter` / `AggregationFilter` | `foundations/18_entities.md §7` | — |
| `DataType` (logical type set) | `foundations/13_types_and_grain.md` | `registry/types_mapping.md` (per-engine mapping) |
| `Dialect` / `DialectId` | `apis/36_semstrait_adapter.md` | — |
| `Diagnostic` + error codes (`*_E_*`) | `apis/30_api_contracts.md §6` | Per-doc: every `3x` doc owns its own diagnostic-code range |
| `Dimension` (struct shape + body roster) | `foundations/18_entities.md §4` | `foundations/11_names_and_scopes.md §4` (planner role) |
| `DimensionType` (enum roster) | `foundations/18_entities.md §4.1` | `foundations/13_types_and_grain.md` (authoring-level DimensionType-to-DataType mapping); `foundations/19_categories.md §2` (category-axis contract — implicit-constraint table per variant) |
| `MeasureCategory` (enum + body structs; implicit-constraint contract) | `foundations/19_categories.md §3` | `foundations/18_entities.md §5` (consumer — `Measure.category:` field) |
| `MetricCategory` (enum + body structs; implicit-constraint contract) | `foundations/19_categories.md §4` | `foundations/18_entities.md §6` (consumer — `Metric.category:` field) |
| Categories — growth recipe + expandability invariants (`SR-CAT-FWD`, `SR-CAT-CLOSED`, `SR-E-19`) | `foundations/19_categories.md §1` (invariants) + `§8` (growth recipe) | — |
| `EngineAdapter` / `EngineArtifact` / `EnginePlan` | `apis/36_semstrait_adapter.md` | — |
| `Expr` / `SemanticExpr` / `PhysicalExpr` | `foundations/14_expressions.md` | `foundations/14b_expression_resolution.md` (compile-time resolution) |
| `ExprSource` (YAML → `Expr` parse) | `foundations/14_expressions.md §2` | — |
| `FileSystem` | `apis/37_semstrait_catalog.md` | — |
| Function catalog (canonical inventory) | `foundations/14a_function_catalog.md` | `registry/functions_mapping.md` |
| `Grain` | `foundations/13_types_and_grain.md` | `foundations/17_temporal_shape.md` (shape × grain matrix) |
| `IoError` / `Source` / `Sink` / `Location` | `apis/31b_semstrait_core_io.md` | — |
| `JoinKeyExprPair` | `foundations/18_entities.md §2.6` | — |
| `JoinType` | `foundations/18_entities.md §2` | `registry/join_types_mapping.md` (per-engine syntax) |
| `Keys` (`{primary, unique, foreign}`) | `foundations/18_entities.md §9` | — |
| Manifest layer types (`Resolved*` prefix) | `apis/33_semstrait_manifest.md` | — |
| `Manifest` + `Repository` | `apis/33_semstrait_manifest.md` | — |
| `Measure` (struct shape) | `foundations/18_entities.md §5` | `foundations/11_names_and_scopes.md §5` (planner role) |
| `Metric` (struct shape) | `foundations/18_entities.md §6` | `foundations/11_names_and_scopes.md §6` (planner role) |
| Names / namespaces / scopes | `foundations/11_names_and_scopes.md` | `foundations/12_nesting_policy.md` (scope-chain-through-nesting rules) |
| Nesting policy / matrix | `data-kinds/26_nesting_matrix.md` (matrix + structural rules R1–R3) + `foundations/12_nesting_policy.md` (per-variant policy) | `data-kinds/2x` (per-variant applicability) |
| `Precondition` (compile-time) | `foundations/10_resolution_pipeline.md` | — |
| `Relationship` + `RelationshipId` | `foundations/18_entities.md §2` (struct) + `foundations/16_composition.md` (planner semantics) + `foundations/14b §4.2` (graph) | — |
| `RelationshipPath` (newtype over `Vec<RelationshipId>`) | `foundations/14b_expression_resolution.md §4.5` | `foundations/16_composition.md §5.2` (consumer, pointer-only) |
| `Request` / `SessionContext` | `apis/34_semstrait_planner.md` | — |
| Resolution pipeline (end-to-end stages) | `foundations/10_resolution_pipeline.md` | — |
| `ResolvedColumnMapping` (Manifest-layer) | `apis/33_semstrait_manifest.md §5.3` | — |
| `ResolvedExprTable` | `foundations/14b_expression_resolution.md` | — |
| SR-E-* entity-level invariants | `foundations/18_entities.md §11` (catalog) | `foundations/19_categories.md §6` (canonical home of SR-E-13 … SR-E-19); `apis/30_api_contracts.md §6.2` (error-code allocation) |
| SR-* (YAML-structural rules) | `apis/32_semstrait_model.md §3.2–§3.5` | — |
| `ScdType` (v1 roster `{Type1, Type2}`) | `foundations/18_entities.md §3.1` | `foundations/17_temporal_shape.md §6` (full Kimball `Type0`–`Type6` as forward-reference) |
| `SemanticInterface` | `foundations/11_names_and_scopes.md` | `foundations/16_composition.md` (composed form) |
| `SemanticMapping` (container struct, compile semantics) | `foundations/15_mapping_and_binding.md` | — |
| `SemanticMappingValue` (enum: `Column`/`Literal`/`Expr`) | `foundations/18_entities.md §10` | `foundations/15_mapping_and_binding.md §5` (compile semantics) |
| `SemanticModel` root type + YAML grammar | `apis/32_semstrait_model.md` | — |
| `SemanticPlan` + `PlanNode` | `apis/35_semstrait_ir.md` | — |
| `semstrait-api` public surface | `apis/38_semstrait_api.md` | — |
| `semstrait` facade public surface | `apis/39_semstrait_facade.md` | — |
| `semstrait-core` public surface | `apis/31_semstrait_core.md` (+ `31b` for I/O) | — |
| `TemporalShape` struct + `TemporalShapeKind` enum | `foundations/18_entities.md §3` (struct) | `foundations/17_temporal_shape.md` (planner-level semantics) |
| `UnionMode` variant roster | `data-kinds/23_unionset.md §4.1` (variant-local) + `foundations/16_composition.md §5` (composition semantics) | — |

---

## Crate-surface mirroring convention

The `3x` api docs (`31`, `31b`, `32`, `33`, …) document what each `semstrait-*` crate publicly exposes. When a crate re-exports a type that is canonically ratified in a `1x` foundations doc, the api-layer doc MAY show the Rust block in-context (for reader convenience) **only** when:

1. A comment pointer cites the canonical home (e.g. `/// See 13 §3.1.`). No alternative ratification is implied.
2. The shape is byte-identical to the canonical home. Any divergence (extra variant, missing field, reordered variants) is a design bug; fix by amending the canonical home, not by diverging the mirror.
3. The mirror appears only once per crate, at the crate's `§N Type-level surface` section — never as a re-ratification in a body clause.

`pub use ::foundations::...` in real code is equivalent to the mirror; the design doc merely shows the shape. Current mirrors (as of 2026-04-17):

| Canonical home | Crate-surface mirror | Re-ratifies? |
|---|---|---|
| `13 §2.1` `DataType` | `31 §4.1` | No — shape identical; comment points to `13 §2.1`. |
| `13 §3.1` `Grain` | `31 §4.2` | No — shape identical; comment points to `13 §3.1` / `§3.2`. |
| `14 §3.2` `Expr` | `31 §2.1` | No — `31` re-shows because `Expr` is the single most-referenced type in the workspace; comment points to `14 §3.2`. |
| `14 §3.4` `SemanticExpr` / `PhysicalExpr` | `31 §2.3` | No — shape identical. |
| `14a §3` `FunctionSpec` / `FnSignature` / etc. | `31 §5` | No — shape identical. |
| `30 §6` `Diagnostic` / `Severity` / `Location` | `31 §6` | No — 30 is the canonical ratification; 31 re-exports. |

Mirrors are a **reader convenience**, not a competing ratification. When `1x` changes shape, `3x` mirrors update in the same commit. The precedence rule in `00 §4.4` resolves any latent drift in favor of the canonical home.

## Duplication guard

Concepts the 2026-04-17 consolidation pass ratified as owned by **exactly one** doc. When adding new prose, check this table before defining a struct / enum inline.

| Concept | Owner | Notes on where consumers may restate *parts* of this concept |
|---|---|---|
| `Relationship` struct | `foundations/18_entities.md §2` | Consumers (`14b`, `16`, `24`, `26`, `33`) reference-only. Manifest layer's `ResolvedRelationship` is a DIFFERENT type, named and defined in `33 §5.2`. |
| `RelationshipId` newtype | `foundations/18_entities.md §2.1` | Uses only. Never redefine. |
| `JoinType` enum | `foundations/18_entities.md §2` | `registry/join_types_mapping.md` MAY list engine-mapped equivalents in table rows but never restates the Rust enum definition. |
| `Cardinality` / `Directionality` | `foundations/18_entities.md §2.4` / `§2.5` | — |
| `JoinKeyExprPair` | `foundations/18_entities.md §2.6` | Retired name: `KeyPair`. |
| `TemporalShape` struct | `foundations/18_entities.md §3` | `17` discusses semantics only, no struct fields. |
| `TemporalShapeKind` enum | `foundations/18_entities.md §3` | Same. |
| `ScdType` v1 roster | `foundations/18_entities.md §3.1` | `17 §6` discusses the wider Kimball Type0–Type6 taxonomy as forward-reference. |
| `DimensionType` enum | `foundations/18_entities.md §4.1` | `13` discusses authoring-level and planner-level DimensionType semantics; body struct shapes (`TemporalDimensionBody`, `BucketedDimensionBody`, `MetadataDimensionBody`) live in `18 §4.1`; `19 §2` owns the category-axis contract (implicit-constraint table). |
| `Dimension` / `Measure` / `Metric` structs | `foundations/18_entities.md §4` / `§5` / `§6` | `11 §4` / `§5` / `§6` describe roles and field layouts for the planner. `Measure.category` / `Metric.category` field semantics live in `19 §3` / `§4` (struct field on the canonical home in `18`, behavior in `19`). |
| `MeasureCategory` enum + body structs | `foundations/19_categories.md §3` | `18 §5` consumes via `Measure.category:`; per-variant planner / adapter contracts cross-cut into `25 §2.11` (variant applicability) and `34` (planner routing). |
| `MetricCategory` enum + body structs | `foundations/19_categories.md §4` | `18 §6` consumes via `Metric.category:`; expr-shape locks consumed by `14b` resolution. |
| `AdditivityType` | `foundations/18_entities.md §5.2` | `11 §8` owns the planner contract; `19 §3.3` ratifies that `MeasureCategory::Snapshot` synthesizes `AdditivityType::Semi`, subsuming explicit `additivity: semi` authoring. |
| `AiContext` | `foundations/18_entities.md §8` | — |
| `Keys` | `foundations/18_entities.md §9` | — |
| `DataKindFilter` / `AggregationFilter` | `foundations/18_entities.md §7` | `21` / `22` / `23` / `24` per-variant filter authoring rules. |
| `SemanticMappingValue` | `foundations/18_entities.md §10` | `15 §5` owns compile-time semantics; `33 §5.3` owns the Manifest-layer `ResolvedColumnMapping` (flattened form). |
| `SR-E-*` codes | `foundations/18_entities.md §11` | `30 §6.2` integrates into the cross-subsystem allocation. |
| `Grain` enum | `foundations/13_types_and_grain.md §3.1` | `18 §3` consumes via `TemporalShape.grain`; never redefined. |
| `RelationshipPath` | `foundations/14b_expression_resolution.md §4.5` | `16 §5.2` consumes via `ComposedSemanticInterface.traversed_paths`; pointer-only. |
| `Diagnostic` error-code ranges | `apis/30_api_contracts.md §6` | Per-doc diagnostic rosters reserve sub-ranges; no two docs claim the same range. |

---

## Open questions

Unresolved / parked items live in `docs/design/questions/`. Status is encoded in the parent directory:

- **`questions/open/`** — files with at least one unresolved Q-ID. The active backlog.
- **`questions/closed/`** — files whose Q-IDs are all resolved (historical record). When every Q-ID in an `open/` file closes, the whole file moves to `closed/` in the next consolidation pass.

Each numbered doc has one sidecar (`<n>_questions.md`); registry catalogs have per-catalog sidecars plus one aggregate index.

### `questions/open/` — active backlog

| Sidecar | Parent doc | Scope |
|---|---|---|
| `questions/open/14b_questions.md` | `foundations/14b` | Round-1 deferrals for expression resolution |
| `questions/open/15_questions.md` | `foundations/15` | Mapping / binding deferrals |
| `questions/open/16_questions.md` | `foundations/16` | Composition / `Relationship` deferrals |
| `questions/open/17_questions.md` | `foundations/17` | Temporal-shape open items (partial closure after 18 consolidation — see top-of-file status summary) |
| `questions/open/19_questions.md` | `foundations/19` | Category-axis post-v1 items (Identifier dimension category Q-CAT-001; lenient YAML downgrade `[TD-CAT-LENIENT]`; author-extensible registry `[TD-CAT-REGISTRY]`) |
| `questions/open/20_questions.md` | `data-kinds/20` | Taxonomy deferrals |
| `questions/open/21_questions.md` | `data-kinds/21` | Dataset deferrals |
| `questions/open/22_questions.md` | `data-kinds/22` | Grainset deferrals (Q-GRN-004 / -006 CLOSED by `26` — see top-of-file status summary) |
| `questions/open/23_questions.md` | `data-kinds/23` | Unionset deferrals (Q-UNI-002 / -009 CLOSED — see top-of-file status summary) |
| `questions/open/24_questions.md` | `data-kinds/24` | Joinset deferrals + (new post-v1 clusters Q-24-09 / Q-24-10 folded from the retired `joinset_shape_semantics.md` sidecar) |
| `questions/open/25_questions.md` | `data-kinds/25` | Applicability-matrix deferrals |
| `questions/open/30_questions.md` | `apis/30` | API-contract / error-code deferrals |
| `questions/open/31_questions.md` | `apis/31` | Core public-surface deferrals |
| `questions/open/32_questions.md` | `apis/32` | `SemanticModel` deferrals |
| `questions/open/33_questions.md` | `apis/33` | Manifest deferrals |
| `questions/open/34_questions.md` | `apis/34` | Planner deferrals |
| `questions/open/35_questions.md` | `apis/35` | IR deferrals |
| `questions/open/36_questions.md` | `apis/36` | Adapter deferrals |
| `questions/open/37_questions.md` | `apis/37` | Catalog deferrals |
| `questions/open/38_questions.md` | `apis/38` | API deferrals |
| `questions/open/39_questions.md` | `apis/39` | Facade deferrals |
| `questions/open/40_questions.md` | `implementation/40` | Refactor-plan deferrals |
| `questions/open/41_questions.md` | `implementation/41` | Deprecations deferrals |
| `questions/open/42_questions.md` | `implementation/42` | Migration deferrals |
| `questions/open/registry_questions.md` | **aggregate index** over the three registry sidecars (pure navigation) | — |
| `questions/open/functions_mapping_questions.md` | `registry/functions_mapping` | Per-engine function-mapping deferrals |
| `questions/open/join_types_mapping_questions.md` | `registry/join_types_mapping` | Per-engine join-type-mapping deferrals |
| `questions/open/temporal_shape_mapping_questions.md` | `registry/temporal_shape_mapping` | Per-engine temporal-shape-mapping deferrals |

### `questions/closed/` — historical record

| Sidecar | Parent doc | Scope |
|---|---|---|
| `questions/closed/11_constraints_deferred.md` | `foundations/11` | Constraints DSL — Q-R4.3a … Q-R4.3d resolved as part of the 2026-04-27 categories+constraints expansion (rewrite of `11 §8`). |
| `questions/closed/31b_io_questions.md` | `apis/31b` | I/O layer — all items CLOSED; retained as a historical record. |

**Retired standalone sidecars (folded in 2026-04-17):**

- `open_questions/joinset_shape_semantics.md` → folded into `questions/open/24_questions.md §Post-v1 shape-hint clusters` as Q-24-09 + Q-24-10.

---

## Registry

Living catalogs of per-engine mappings for canonical primitives. Canonical specs in `1x` are authoritative; registry mappings describe how each engine expresses those canonicals.

| Catalog | Canonical source it mirrors | Scope |
|---|---|---|
| `registry/types_mapping.md` | `foundations/13_types_and_grain.md` | `DataType` ↔ DataFusion / Spark / DuckDB native types, cast semantics, per-engine gaps |
| `registry/functions_mapping.md` | `foundations/14a_function_catalog.md` | Canonical function ↔ engine function names, rewrite tiers, arity differences |
| `registry/temporal_shape_mapping.md` | `foundations/17_temporal_shape.md` + `foundations/18_entities.md §3` | Per-engine `TemporalShape` expression (SCD / Events / Snapshot), `AsOf` rewrite tiers |
| `registry/join_types_mapping.md` | `foundations/16_composition.md` + `foundations/18_entities.md §2` | Canonical `JoinType` ↔ engine join variants |

See `registry/README.md` for the engine-coverage policy and versioning rules.

---

## Cross-references & discovery shortcuts

- **Diagnostic-code lookup.** Every error / warning code (`PARSE_E_*`, `COMP_E_*`, `PLAN_E_*`, `VALID_E_*`, `ADAPT_E_*`, `IO_E_*`, `CAT_E_*`, …) is allocated in `apis/30_api_contracts.md §6.2`. Per-doc rosters append to that allocation.
- **YAML `kind:` spelling.** There is no `kind:` discriminator. Top-level uses plural tags `datasets:` / `grainsets:` / `unionsets:` / `joinsets:` per `apis/32 §1–§3`.
- **Renames landed in 2026-04-17 consolidation.**
  - `apis/32c_entities.md` → `foundations/18_entities.md` (promoted to foundations; same content).
  - `ColumnMapping` (model-layer) → `SemanticMapping` (model-layer).
  - `ColumnMappingValue` (model-layer) → `SemanticMappingValue` (model-layer).
  - `ResolvedColumnMapping` (Manifest-layer) **unchanged** per `33 §5.3`.
  - `KeyPair` → `JoinKeyExprPair` (per `18 §2.6`).
- **Renames landed in 2026-04-27 questions restructure (ninth pass).**
  - Directory `open_questions/` → `questions/` with two subfolders: `questions/open/` (28 active backlog files) and `questions/closed/` (1 archived file: `31b_io_questions.md`).
  - Filename suffix `<doc>_open_questions.md` → `<doc>_questions.md` across all 29 sidecars (status now comes from the parent directory). Files without that suffix (`11_constraints_deferred.md`) keep their names.
  - Mechanical reference sweep updated every `open_questions/...` path-prefixed link, every bare `<doc>_open_questions.md` reference, and every `doc:` front-matter field across `docs/design/` + `CLAUDE.md`.
  - No content change inside any sidecar; the rename is structural only. Q-IDs, bodies, and per-file CLOSED status summaries are unchanged.
- **"I know what `*Body` I need."** See the body-struct inventory in `apis/32 §3.1` (top-level wrappers) and the per-variant struct layouts in `data-kinds/21`–`24`. The common fields are on `DataKindBase` (`32 §3.1`); ratified entity-struct shapes live in `18 §§4–10`.

---

## Glossary pointer

The canonical glossary lives in `00_overview.md §4` (Core Nouns + Core Verbs + Banned Terms). This index does not duplicate it — it points to it. If a term is missing from that glossary AND from the alphabetical table above, the concept is either unratified or needs a home; flag it in `STATUS.md`.

---

## Maintenance

- When a new doc lands in any folder, add a row to the folder's table in `00_overview.md §6`, add its sidecar (if any) to the [open-questions](#open-questions) section here, and (if the new doc ratifies a new named concept) add a row to the [alphabetical topic](#topic--canonical-home-alphabetical) table.
- When a concept moves between docs (like the 2026-04-17 `32c` → `18` promotion), update **both** `00_overview.md §4` and this index's tables in the same commit. `STATUS.md` records the rationale; these two files are the navigation surfaces.
- This file is expected to stay under ~300 lines. When a table exceeds that, split it off into a dedicated reference doc rather than growing INDEX.md indefinitely.
