---
id: 15-binding-process
status: ratified
version: 2.1
impl: implemented
---

# Binding Process

> **Struct ownership (2026-04-27 consolidation).** The v1 authoring-layer `SemanticMappingValue` enum roster — **4 variants** `{Column(String), Literal(LiteralValue), Expr(PhysicalExpr), Metadata(MetadataDimensionRecipe)}` — the `LiteralValue` enum, the `MetadataDimensionRecipe` / `MetadataExtraction` shapes, and the `semantic_mapping:` YAML block shape are ratified in `[18_entities.md §10](../18_entities.md#10-semanticmapping-value-shape)`. This doc owns the *Binding process* on top — compile-time resolution, per-source `Coverage`, `PhysicalSource` / `Schema` / `FileFormat` taxonomy, glob expansion, schema reconciliation, the SemanticManifest-layer `ResolvedColumnMapping` materialization, and the layer-3 compile-time mechanics (`path_token`, §8) that produce the per-source resolved metadata literals.
>
> **Rename cascade (body).** `ColumnMapping` → `SemanticMapping` (container), `ColumnMappingValue` → `SemanticMappingValue` (enum), `Computed` → `Expr` (variant). The v1 roster at the authoring surface is **4-variant**: `Metadata` is restored as a distinct variant alongside `Column` / `Literal` / `Expr` (reversing the earlier `[TD-MAP-METADATA-FOLD]` decision — see the resolution note below). The `ResolvedColumnMapping` SemanticManifest-layer name is retained per `33 §5.3` — a future symmetry rename to `ResolvedSemanticMapping` is tracked but is not part of this consolidation.
>
> `**auto` default.** An absent `semantic_mapping:` block on a `Dataset`'s `extras` is equivalent to `semantic_mapping: auto` per `18 §10.3` — every Semantic 1:1 to a physical column of the same name; explicit entries narrow the default.

> `**[TD-MAP-METADATA-FOLD]` — RESOLVED (2026-04-27).** The fold of `Metadata` into `Expr` is reversed; `Metadata` is restored as a distinct 4th variant. The author surface is the Dimension's `type: { metadata: { path: { token: N } } }` block (per `13 §4.7` / `18 §4`), and the binding-resolution pass synthesizes `SemanticMappingValue::Metadata(MetadataDimensionRecipe)` before the completeness check. Per-source resolved `LiteralValue`s land on `ResolvedPhysicalSource.metadata_values` (§7.6), not in the recipe. v1 scope is **path-token extraction only**; partition-level extraction (`partition.level: N`) is deferred to v2 — the Dimension-side body grammar in `13 §4.7` retains the partition shape for forward-compatibility, but compile-time recipe synthesis and runtime extraction are wired only for the path arm.

This document defines the compile-time Binding process: EntityId derivation, source resolution (glob/catalog), schema reconciliation, SemanticMapping completeness, Coverage derivation, and ResolvedBinding materialization. See [catalog-absent-fallback](catalog-absent-fallback.md) for model-as-truth override behavior when no catalog is present.

## 1. Purpose and Scope

### 1.1 What `15` ratifies

`15` is the foundations document that closes the loop between the **Semantics-facing** surface of a `SimpleDataKind` (named and shaped by `11`, typed by `13`, expressed by `14` / `14a`, resolved by `19`) and the **physical-facing** surface underneath it: the files, tables, or snapshots that actually hold the data, the schemas those targets expose, and the recipe (direct column / literal / computed / metadata-extraction) that produces each Semantics value from whatever rows the physical target returns.

Everything that sits **below the `SemanticInterface` boundary for a single `SimpleDataKind`** is authoritative here:

- The `Binding` — the single-instance join between Semantics and Physical, owned by exactly one `SimpleDataKind`.
- The `PhysicalSource` roster — the 1-or-more resolved targets the Binding points at (a file path, a catalog table, an Iceberg snapshot).
- The `SemanticMapping` — the Semantics-name-keyed recipe table.
- The per-source `Coverage` — which Semantics each source actually provides.
- The SemanticManifest-layer counterpart `ResolvedColumnMapping` — the flattened, pre-indexed form the planner consumes at query time.
- The compile-time resolution flow that moves a Model-level binding declaration into a `ResolvedBinding` living on a `ResolvedDataKind`.
- The `MetadataDimension` extraction mechanics introduced structurally in `13 §4.7`. v1 scope: **path-token extraction only** — `15 §8.1` pins down the layer-3 `path_token` compile mechanic, the `Cast`-to-declared-`data_type` policy, and the error conditions. Partition-level extraction (`partition.level: N`) is deferred to v2 (§8.0 v1-scope banner); `15` retains the original section structure as a forward-compatibility hint, with the v2 surface clearly marked.
- The **layer-3 compile-time-mechanic stratum** for non-expression compile work — the third stratum of the layered expression model (banner above). `path_token` is the v1 inhabitant; future inhabitants land here as the recipe surface grows.
- The `SemanticMapping` entries for **implicit `Field` semantics synthesised from computed relationship join keys**. A computed `Relationship` join-key side (`18 §2.8`) decomposes at compile into a synthesised implicit `Field` whose lowered `PhysicalExpr` is bound through the ordinary binding mapping (`SemanticMappingValue::Expr`). The decomposition trigger and the manifest `ResolvedJoinKey` form are owned by `[19 §3.12](../19_expression_flow.md)` and `[33 §8A.2](../../apis/33_semstrait_manifest.md)`; `15` ratifies only that such synthesised semantics bind through the same mapping path as any author-declared semantic.

### 1.2 What `15` does NOT ratify

- `**ComplexDataKind` composition.** `Unionset`, `Grainset`, and `Joinset` do not carry their own `Binding`s; they aggregate the `Binding`s of their constituent `SimpleDataKind`s. The composition mechanics, `ComposedSemanticInterface` shape, and per-composition `Coverage` (which constituent provides each field on the unified surface) all live in `foundations/16_composition.md`. `15` is explicit about the boundary in §6.4.
- **DataKind lifecycle.** How `ResolvedBinding`s attach to `ResolvedDataKind`s during `compile`, and the post-compile guarantees about their ordering inside the `SemanticManifest`, are ratified in `foundations/20_taxonomy.md` and the crate contract in `apis/33_semstrait_manifest.md`. `15 §10` enumerates the steps inside `compile` that produce a `ResolvedBinding`; it does not ratify their position in the `compile` driver.
- **Expression resolution.** `SemanticMappingValue::Expr(PhysicalExpr)` stores a compiled `PhysicalExpr`. The substitution algorithm that produces that `PhysicalExpr` from a `model::UnresolvedSemanticExpr` (via the `FunctionRegistry` in `14a` and the cross-DataKind walk) is owned by `[19 §3](../19_expression_flow.md)`. `15 §5.3` describes only the **storage site** and the **wrapper-invariant contract** the stored `PhysicalExpr` must satisfy.
- **Catalog-provider shape.** The `CatalogProvider` trait surface, the `FileSystem` trait surface, their async posture, and their error enums are ratified in `apis/37_semstrait_catalog.md`. `15 §3.2` uses `CatalogRef` as an **opaque handle** into that surface; consumers of `15` never reach into the catalog crate directly.
- **Per-engine dialect specifics.** Nothing in `15` branches on engine identity (I3). A `PhysicalSource` carries a logical `DataType`-bearing `Schema`; conversion to an engine-specific type is adapter territory (`36`, I2).

### 1.3 Design posture

`15`'s posture is **compile-once, store-flat, plan-fast**:

- **Compile-once.** Every glob is expanded, every catalog table is fetched, every physical schema is reconciled against every declared `DataType`, every `model::UnresolvedSemanticExpr` on the semantic side is compiled into a `PhysicalExpr` over columns/literals, every per-source coverage bit is decided — all before the `SemanticManifest` is sealed. I5 demands it, I8 requires it to make plan-time O(1), and I4 demands it reproducibly.
- **Store-flat.** The Model-layer `SemanticMapping` is an enum-valued map keyed by `SemanticsName`. The SemanticManifest-layer `ResolvedColumnMapping` splits it into four parallel flat maps — one per variant — so the planner's per-Semantics lookup is a single HashMap probe with no enum match in the hot path (I6).
- **Plan-fast.** The planner never re-resolves anything in `15`'s scope. It reads `ResolvedBinding`, picks sources based on Coverage (Unionset §6.1; Grainset `17`), and emits `PlanNode`s. No catalog call, no filesystem call, no expression compilation.

### 1.4 Reference implementations — where `15` sits in the peer-group landscape

The brief is: pick a name for every concept we can't avoid having, and resist importing vocabulary that would re-open ratified decisions. Peers:

- **dbt metricflow.** `data_source.sql_table` / `data_source.sql_query` + `identifiers` + `measures` + `dimensions` — a single model-side block that couples a physical table to a set of Semantics. `15`'s `Binding` is the direct analog. metricflow has no analog for `MetadataDimension` (it does not pattern-match on S3 paths); it relies on the upstream warehouse to have the data already in shape. `15` keeps `MetadataDimension` because semstrait compiles over lake-native paths with partition-encoded metadata (`year=.../month=...`).
- **Cube.js.** `cube.sql_table` / `cube.sql` + `dimensions` + `measures` + `segments` + `pre_aggregations`. The `cube.sql` escape hatch is raw SQL; I1 forbids that here, so `SemanticMappingValue::Expr` carries a typed `PhysicalExpr` instead. Cube's `partition_granularity` is a roll-up concept ratified here by `Grain` + per-source `Coverage`, not by a Binding-level knob.
- **LookML.** `view.sql_table_name` + `view.derived_table` + per-dimension `sql:`. LookML's `${TABLE}.column` and `${other_view.field}` patterns are what `UnresolvedSemanticExpr` bare identifiers replace (resolved per `[14 §6.5](../14_expressions.md)` + `[19 §3](../19_expression_flow.md)`). LookML's `sql_trigger` / PDT machinery is outside `15`'s scope entirely.
- **Iceberg catalog.** The `Snapshot` variant of `PhysicalSource` is directly inspired by Iceberg's snapshot model — `metadata.current-snapshot-id` pins reproducibility (I4) against a moving warehouse state. Iceberg's partition-transform vocabulary (`year`, `month`, `day`, `hour`, `bucket[N]`, `truncate[N]`) informs `PartitionColumn` but is **not** replicated on it — partition-transform awareness is a planner concern (pruning), not a Binding concern.

The peers supply structural precedent and error-case nudges; nothing in the peer set overrides the semstrait vocabulary ratified in `00 §4`.

### 1.5 Guardrails — how `15` upholds `00 §9` invariants


| Invariant                                       | Where `15` keeps it                                                                                                                                                                                                                                                    |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I1** — no raw SQL in canonical layer          | `SemanticMappingValue::Expr` carries `PhysicalExpr` (an `Expr` tree), never a string. The YAML surface for computed entries parses through `14 §6` into `model::UnresolvedSemanticExpr`, then compiles to `PhysicalExpr` per `19 §3`.                                     |
| **I2** — physical types via adapters only       | `PhysicalSource.schema.columns[_].data_type` is logical `DataType` (per `13 §2`). The `14a` registry's promotion lattice is what decides widen/narrow; no Arrow/Spark/DuckDB type leaks into the SemanticManifest.                                                     |
| **I3** — no engine branching in canonical layer | `PhysicalSource` is engine-agnostic. No variant, no field, no error code in `15` names an engine. Adapters read `PhysicalSource` at `adapt` time and decide how to register it with their engine (`36`).                                                               |
| **I4** — SemanticManifest is deterministic      | Glob expansion sorts by lexical order of resolved absolute identifier (§3.5). Catalog-fetch results are sorted by fully-qualified name before being folded into the SemanticManifest. Ties are impossible by construction.                                             |
| **I5** — resolution is compile-time             | All Binding work (glob expansion, catalog fetch, schema resolution, SemanticMapping well-formedness, PhysicalExpr compilation, Coverage derivation) happens in `compile`. Plan-time reads only `ResolvedColumnMapping` and a pre-sorted `ResolvedPhysicalSource` list. |
| **I8** — SemanticManifest is planner-complete   | The SemanticManifest stores `ResolvedBinding`, `ResolvedColumnMapping`, and the per-source `ResolvedPhysicalSource` list. No planner step re-fetches a schema, re-expands a glob, or re-compiles a Computed expression.                                                |
| **I10** — non-exhaustive public sum types       | `PhysicalSource`, `FileFormat`, `SemanticMappingValue`, `CoverageVariant`, `CompileError` (all `15`-owned variants) carry `#[non_exhaustive]`. Adding a new file format, a new coverage variant, a new binding-error kind is MINOR per `30 §2`.                        |


I6 / I11 apply transitively — `15` describes the compile-time surface, which is the only place async / I/O is allowed per I11; the SemanticManifest forms that `15` produces is then consumed synchronously by the planner.

## 2. The `Binding`

### 2.1 Structure

A `Binding` is the single join point between a `SimpleDataKind`'s `SemanticInterface` and its physical backing. Every `SimpleDataKind` owns **exactly one** `Binding`; this is an invariant at both the Model layer (the YAML parser rejects multiple binding blocks on one kind per `11 §5.3`) and the SemanticManifest layer (the compile-time materialization produces a single `ResolvedBinding` per `ResolvedDataKind::Simple`). `ComplexDataKind`s carry **no** `Binding` of their own — they aggregate the `Binding`s of their constituent Simple children, through the `ComposedSemanticInterface` machinery in `16`.

Model-layer shape:

```rust
#[non_exhaustive]
pub struct Binding {
    pub binding_id: EntityId,
    pub sources: Vec<PhysicalSource>,
    pub semantic_mapping: SemanticMapping,
    pub coverage: Option<Coverage>,
}
```

Every field is populated by `compile`; the Model-layer YAML surface in `semstrait-model` uses a parallel `BindingSpec` type whose fields are unresolved (glob patterns, declarative `SourceRef`s, and bare-`Expr<L>` authoring expression values inside `SemanticMappingSpec` — the `ExprSource` wrapper is removed per doc 45; the friendly projection is field-level). The resolution flow in §10 consumes a `BindingSpec` and produces a `Binding` (and, by the time it lands in the SemanticManifest, a `ResolvedBinding` — §7).

The `#[non_exhaustive]` tag is present to allow a future `post_binding_hook: Option<PhysicalExpr>` or similar Semantics-adjacent extension to be added as a MINOR per `30 §4`.

### 2.2 Binding identity (`EntityId`)

A Binding has no model-authored identity, so it is identified by an `EntityId` generated deterministically at compile (UUIDv5 over `(data_kind id, source id, mapping)`, `33 §9.1`). There is no `BindingId` newtype anywhere — the compile working set, the manifest `bindings` collection (`33 §7.1`), and the runtime all use this `EntityId`.

- **Derivation.** The binding `EntityId` is derived deterministically from the binding's content — `(data_kind_id, source_id, mapping)` — so identical `(model, catalog)` inputs produce identical binding ids (I4).
- **Cross-run stability — improved.** Because the id is content-derived rather than counter-allocated, it is stable across runs *and* across model edits that don't change the binding (the old counter-allocated handle shifted whenever an earlier `SimpleDataKind` was inserted). Cross-manifest comparison by binding `EntityId` is meaningful when the binding content is unchanged.
- **Reference shape.** A `SimpleDataKind` owns exactly one Binding; `Dataset.bindings` carries the binding's `EntityId` (`33 §6.4`). The reverse (`source_id -> [binding EntityId]`) is derived at load (`33 §7.1`).
- **Runtime hot-path handles.** Any compact integer index the planner runtime wants for adjacency is a daggy graph handle built at graph build (`34`), addressing the binding by its `EntityId`; no separate integer binding id is persisted or carried.
- **Serialization.** The binding `EntityId` surfaces directly in the persisted manifest form (`33`) and round-trips through `Repository` unchanged.

The earlier Round-1 `u32`-counter proposal (per-compile counter, no cross-manifest namespace) is **superseded**: binding identity is a deterministic content-derived `EntityId` (`33 §9.1`), which is both globally unique and cross-run/cross-edit stable without a counter. `questions/open/15_questions.md` Q-MAP-001 is mooted accordingly.

### 2.3 Cross-reference: `19 §3.2`'s `ResolvedExprKey`

`19 §3.2`'s `ResolvedExprTable` is keyed on `(SemanticsName, EntityId)` precisely because `15` ratifies `EntityId` as the per-Binding identity. `15 §7.2` describes the SemanticManifest's storage split: the `ResolvedExprTable` stores the physical expression bodies; the `ResolvedColumnMapping.computed: HashMap<SemanticsName, PhysicalExpr>` is a **per-Binding denormalization** that copies the `PhysicalExpr` into the Binding's own hashmap for O(1) access without going through the global table. Whether the SemanticManifest stores the `PhysicalExpr` once (in the table) with the `ResolvedColumnMapping.computed` value being a pointer/index, or twice (once in each structure), is an implementation choice ratified in `33`. From the contract surface of `15`, both structures are populated and both are plan-readable.

### 2.4 Cross-reference: Complex composition

Per `00 §4.1` / `11 §5` / `16 §2`, a `ComplexDataKind` does not own a `Binding`. The composition rule is:

- **Unionset** — the union composition carries *N* `Binding`s, one per Simple (or nested) branch. The per-branch `Coverage` is what tells the planner "branch *b* has `NullFill` for Semantics *s*" so it can emit `SELECT NULL AS s` in the branch's `Project`. `15` ratifies the per-Binding `Coverage` shape (§6); `16` ratifies how a Unionset reads it at composition time.
- **Grainset** — each level resolves to a child DataKind with its own Binding set. Grain-selection is a planner decision that reads per-child Binding `Coverage` (specifically, which Semantics resolve to `Native` at which grain). `15` does not ratify the grain axis itself (that is `13 §5` + `17`); it only ratifies that Coverage captures the per-source bit the planner reads.
- **Joinset** — each join member is a Simple (or nested Complex) kind with its own Binding(s). The join-path construction over declared `Relationship`s is owned by `16`. `15`'s `Binding` has no Joinset-specific field.

## 10. Compile-Time Resolution Flow

The binding-resolution flow lives inside the `compile` stage, between `10 §3.3`'s stage-level sub-steps. Per `10`'s breakdown, `compile` orchestrates catalog metadata fetch, source resolution, authoring-expression compilation (`Expr<L>` → `ir::PhysicalExpr`), and SemanticManifest-index construction; `15`'s flow is the sub-sequence specific to each Binding.

For each `SimpleDataKind` in the validated `SemanticModel`:

### 10.1 Step 1 — binding `EntityId` derivation

The `compile` driver derives the Binding's `EntityId` deterministically (UUIDv5 over `(data_kind id, source id, mapping)` per `33 §9.1`). This is a single hash; it carries negligible cost and is stable across recompiles for unchanged binding content.

### 10.2 Step 2 — Source resolution (glob expansion + catalog fetch)

Expand the Model's source spec per §3.5. Outcomes:

- Produce a `Vec<PhysicalSource>` with `sources.len() >= 1` (else `COMP_E_0301 NoSourcesMatched`).
- Every source has `schema` and `partitions` populated.
- Every catalog call is made (fail-fast on `CatalogUnavailable` / `SchemaFetchFailed`).
- `Snapshot`-variant sources pin their `snapshot_id` from the catalog's current-snapshot response.

This is the **I/O-heavy step**: the filesystem is hit per `FileSystem::expand_glob` call, the catalog per `CatalogProvider::load_table_metadata` call. Per I11, these are the only `await` points in `compile`; their async shape is ratified in `37`.

### 10.3 Step 3 — Schema resolution per source

Per source, resolve the schema through §4's per-format strategy:

- Parquet / ORC / Avro: read from file footer / container.
- CSV / JSON: use declared schema if present; else infer from header/sample; emit `COMP_W_0301 SchemaInferenceUsed` for inferred cases.
- Table / Snapshot: read from catalog metadata.

Populate `Schema.columns` (logical `DataType`s, nullability).

### 10.4 Step 4 — `SemanticMapping` well-formedness and reconciliation

1. **Metadata recipe synthesis (§5.5)** — runs FIRST, before the completeness check. For each Dimension whose `type:` is `Metadata(MetadataDimensionBody)` (per `13 §4.7` / `18 §4`):
  - **v2-deferred guard.** If `body.source.partition: Some(_)` → emit `COMP_E_0322 MetadataPartitionDeferredV2`. Path-only is in scope for v1.
  - Construct `SemanticMappingValue::Metadata(MetadataDimensionRecipe { extraction: Path { token }, data_type })` from the Dimension body. `data_type` comes from the Dimension's declared `data_type:` field.
  - Insert into `SemanticMapping.entries[name]`. If the author already authored an entry under that name (a misuse) → `COMP_E_0309 SpuriousBindingEntry` (the metadata recipe wins; the explicit author entry is rejected).
  - The synthesized entries are now part of the SemanticMapping seen by the completeness check below.
2. **Completeness check (§5.6):**
  - Missing Semantics → `COMP_E_0308 MissingBindingEntry`.
  - Spurious entry → `COMP_E_0309 SpuriousBindingEntry`.
3. **Per-entry variant dispatch:**
  - `Column`: validate the column exists on at least one source; if on NO source, `COMP_E_0310`-lite (actually `ColumnMissingOnAllSources { semantics, column }`, COMP_E_0318). Reconcile declared vs physical `DataType` per §9.1; wrap in `Cast` if widening/narrowing. Record cross-source type agreement per §9.3.
  - `Literal`: validate representability (§5.3 error list).
  - `Computed`: invoke `model::UnresolvedSemanticExpr::resolve` per `[19 §3.1](../19_expression_flow.md)`. Use the cross-source reconciled schema as the column-lookup context. Check `expr.referenced_columns` across sources per §9.3. Reconcile `expr.inferred_type` vs declared Semantics `DataType` per §9.2.
  - `Metadata`: per-source applicability check only — actual value resolution happens in step 5 (Coverage derivation). For `Path { token }` extraction: every source in the Binding must be `PhysicalSource::File` (else `COMP_E_0312 MetadataTokenOnNonFileSource`); the Binding must have ≥1 source. The deeper checks (out-of-range token, empty segment, cast failure) run during the per-source pass in step 5.
4. **Synthesize compile-derived entries** for Constraints that require a `SemanticMappingValue::Expr` (§5.6 edge case — e.g. `Measure(Count, DerivesFrom(Key))`). These entries are added to the Model-layer `SemanticMapping` struct in-place during the reconciliation pass, then flow through step 4.3's Computed branch.

### 10.5 Step 5 — `Coverage` derivation

Per §6.2. For each `(source_index, semantics)` pair, compute the `CoverageVariant` and store in `Coverage.entries` iff not `Native`.

For metadata-bound Semantics (§5.5), this pass also performs the **per-source value resolution** that feeds `ResolvedPhysicalSource.metadata_values`:

1. Run the layer-3 mechanic (§8.1) — `path_token(source.path, recipe.extraction.token)`.
2. Cast the resulting `String` to `recipe.data_type` (§8.1.2).
3. Insert the resulting `LiteralValue` into the source's `metadata_values: HashMap<SemanticsName, LiteralValue>` map.
4. Emit `Coverage::Metadata` for the `(source_index, semantics)` pair.

Any failure (`COMP_E_0311`, `COMP_E_0312`, `COMP_E_0320`, `COMP_E_0321`) halts the Binding's resolution per §10.8 fail-fast posture. There is no per-source soft-fallback for metadata extraction in v1 (§8.4).

If any entry is `NullFill` and the Binding's owning `DataKind` is not a Simple constituent of a Unionset (or other NullFill-tolerant consumer), emit `COMP_E_0310 UnusableNullFillInNonUnionContext`. The consumer-tolerance check requires traversing up the Model's parent-DataKind reference; this traversal is the `compile` driver's job, not `15`'s — `15`'s Coverage-derivation pass just computes the per-source variant and flags NullFill entries to the driver.

### 10.6 Step 6 — `ResolvedBinding` / `ResolvedColumnMapping` materialization

Per §7. Build the flat HashMaps:

- `columns: HashMap<SemanticsName, ColumnName>` — populated from surviving `Column` entries.
- `literals: HashMap<SemanticsName, ResolvedLiteral>` — from `Literal`.
- `computed: HashMap<SemanticsName, PhysicalExpr>` — from `Expr` (native) + synthesized Cast-wrapped `Column` entries (§9.1) + synthesized derived-Measure entries (§5.6).
- `metadata: HashMap<SemanticsName, MetadataDimensionRecipe>` — from `Metadata` (the per-Binding recipe; per-source values are on each `ResolvedPhysicalSource.metadata_values`, populated in step 5).
- `source_coverage: HashMap<CoverageKey, CoverageVariant>` — from the Coverage-derivation pass.

Wrap each `PhysicalSource` into a `ResolvedPhysicalSource { source, metadata_values }` (§7.6) using the `metadata_values` maps populated in step 5. Attach to a `ResolvedBinding { binding_id, sources: Vec<ResolvedPhysicalSource>, column_mapping: ResolvedColumnMapping }`.

### 10.7 Step 7 — SemanticManifest-index contribution

The Binding's `ResolvedBinding` is handed off to the SemanticManifest-index-construction stage. The `ResolvedExprTable` (per `19 §3.4`) absorbs every Computed entry as `(SemanticsName, EntityId) → PhysicalExpr`. The per-DataKind Binding index is populated (data-kind `EntityId → Vec<` binding `EntityId>` — vector length 1 for `SimpleDataKind`). These are `33`-owned SemanticManifest structures; `15`'s flow just feeds them.

### 10.8 Error-handling posture

The entire §10 flow is **fail-fast** (per `30 §7` / `10 §3.3`). A compile error in step 2 halts the whole Binding's resolution; subsequent steps are not attempted. Warnings accumulate throughout and are attached to the eventual `CompileError` or to the success `SemanticManifest`.

Some steps CAN collect multiple errors before returning (step 4's completeness check reports every missing / spurious entry in one pass before failing). Others are strictly fail-fast (step 2's glob expansion returns the first failure). The convention: structural well-formedness checks accumulate; I/O-bound and dependency-chain checks fail-fast.

### 10.9 Placement within `10 §3.3`

`10 §3.3` enumerates `compile`'s sub-steps as: **catalog fetch → glob expand → schema resolve → name resolve → authoring-`Expr<L>` compile → index build**. `15 §10`'s per-Binding flow is a refinement: it enumerates the sub-structure of the "authoring-`Expr<L>` compile → index build" cluster specifically for Binding work. The catalog-fetch and glob-expand sub-steps are shared across Bindings (a single catalog snapshot is taken for the whole compile invocation); `15 §10 step 2` calls into those shared resources per-Binding.

## 11. Error Model

### 11.1 Binding-owned `CompileError` variants

All compile-time error variants introduced or re-surfaced by `15`, with proposed stable codes per `30 §6.2`. The `COMP_E_0200-0299` sub-range (catalog/source resolution) hosts source-level errors; the `COMP_E_0300-0399` sub-range (schema/binding) hosts SemanticMapping / reconciliation errors.


| Code          | Variant                                                                                  | Sub-range | Trigger                                                                                                |
| ------------- | ---------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| `COMP_E_0203` | `CatalogUnavailable { catalog_id, cause }`                                               | 0200–0299 | catalog whose `CatalogId` was named on the binding is unreachable / not registered                     |
| `COMP_E_0301` | `NoSourcesMatched { binding_id, pattern }`                                               | 0300–0399 | glob / table-glob produced zero matches                                                                |
| `COMP_E_0302` | `GlobExpansionFailed { binding_id, pattern, cause }`                                     | 0300–0399 | filesystem / catalog raised during glob enumeration                                                    |
| `COMP_E_0303` | `UnrecognizedFileFormat { path }`                                                        | 0300–0399 | file extension does not map to a known `FileFormat`                                                    |
| `COMP_E_0304` | `MixedFormatsInGlob { pattern }`                                                         | 0300–0399 | glob resolved to files with different inferred formats                                                 |
| `COMP_E_0305` | `LiteralOverflow { name, value, data_type }`                                             | 0300–0399 | `SemanticMappingValue::Literal` value does not fit the declared type                                   |
| `COMP_E_0306` | `NullLiteralForNonNullableSemantics { name }`                                            | 0300–0399 | `LiteralValue::Null` declared on a non-nullable Semantics                                              |
| `COMP_E_0307` | `LiteralParseFailed { name, value, target_type }`                                        | 0300–0399 | string literal failed to parse into target temporal type                                               |
| `COMP_E_0308` | `MissingBindingEntry { semantics, binding_id }`                                          | 0300–0399 | Semantics in interface, absent from `SemanticMapping`                                                  |
| `COMP_E_0309` | `SpuriousBindingEntry { name, binding_id }`                                              | 0300–0399 | `SemanticMapping` key not in Semantics                                                                 |
| `COMP_E_0310` | `UnusableNullFillInNonUnionContext { binding_id, source_index, semantics }`              | 0300–0399 | NullFill derived for a Binding whose owning DataKind is not tolerance-consumed                         |
| `COMP_E_0311` | `MetadataTokenOutOfRange { binding_id, source_index, token_index, path }`                | 0300–0399 | v1 path-token: token index ≥ segment count after scheme-strip (§8.1.3)                                 |
| `COMP_E_0312` | `MetadataTokenOnNonFileSource { binding_id, source_index, semantics }`                   | 0300–0399 | v1 path-token: metadata-bound Semantic in a Binding whose source is `Table` or `Snapshot` (§8.1.3)     |
| `COMP_E_0313` | `MetadataPartitionUnavailable { binding_id, source_index, semantics, level }`            | 0300–0399 | **DEFERRED to v2** (partition extraction non-goal in v1; §8.0 / §8.2.2). Reserved code; no v1 emitter. |
| `COMP_E_0314` | `InconsistentPartitioning { binding_id, sources, semantics }`                            | 0300–0399 | **DEFERRED to v2** (§8.0 / §8.2.2). Reserved code; no v1 emitter.                                      |
| `COMP_E_0315` | `IncompatiblePhysicalType { semantics, declared, physical }`                             | 0300–0399 | declared and physical `DataType` not cast-compatible per `14 §6.4`                                     |
| `COMP_E_0316` | `ComputedTypeMismatch { semantics, inferred, declared }`                                 | 0300–0399 | Computed expression's `inferred_type` not cast-compatible with declared Semantics type                 |
| `COMP_E_0317` | `CrossSourceTypeDisagreement { binding_id, column, types }`                              | 0300–0399 | a column has different logical types in different sources of one Binding                               |
| `COMP_E_0318` | `ColumnMissingOnAllSources { binding_id, semantics, column }`                            | 0300–0399 | `Column`-valued Semantics's physical column is absent from every source                                |
| `COMP_E_0319` | `SchemaFetchFailed { source, cause }`                                                    | 0300–0399 | format-specific schema-resolution step failed (Parquet footer unreadable, JSON sample I/O error, etc.) |
| `COMP_E_0320` | `MetadataTokenEmpty { binding_id, source_index, token_index, path }`                     | 0300–0399 | v1 path-token: extracted segment is empty (defensive against degenerate path components; §8.1.3)       |
| `COMP_E_0321` | `MetadataCastFailed { binding_id, source_index, semantics, raw_segment, declared_type }` | 0300–0399 | v1 path-token: raw segment cannot be cast to the recipe's declared `data_type` (§8.1.2)                |
| `COMP_E_0322` | `MetadataPartitionDeferredV2 { binding_id, semantics }`                                  | 0300–0399 | Dimension body declares `partition: Some(_)`; partition extraction is deferred to v2 per §8.0          |


### 11.2 Binding-adjacent warnings


| Code          | Variant                                                             | Trigger                                                       |
| ------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- |
| `COMP_I_0301` | `ImplicitWideningCast { semantics, from, to }`                      | §9.1 widening-cast wrapping                                   |
| `COMP_W_0301` | `SchemaInferenceUsed { source }`                                    | CSV-without-declared / JSON-without-declared schema inference |
| `COMP_W_0302` | `ImplicitNarrowingCast { semantics, from, to }`                     | §9.1 narrowing-cast wrapping                                  |
| `COMP_W_0303` | `FloatDecimalCrossCast { semantics, from, to }`                     | §9.1 float/decimal cross cast                                 |
| `COMP_I_0304` | `ImplicitWideningCastOnComputed { semantics, from, to }`            | §9.2 widening on Computed                                     |
| `COMP_W_0305` | `ImplicitNarrowingCastOnComputed { semantics, from, to }`           | §9.2 narrowing on Computed                                    |
| `COMP_W_0306` | `NullableSourceForNonNullableSemantics { semantics, source_index }` | §9.4 nullability mismatch                                     |


### 11.3 Re-surfaced errors from `14` / `14a` / `19`

Errors raised by `14` / `14a` / `19` during Computed-entry compilation pass through `15`'s resolution flow unmodified; `15` does not re-codify them. Examples:

- `EXPR_E_0201 UnknownReference` (from `19 §3.3` substitution)
- `EXPR_E_0206 ComputedCycle` (from `19 §3.5`'s SCC pass)
- `EXPR_E_0401 TypeInferenceFailed` (from `14 §7`)
- `EXPR_E_0301 FunctionArityMismatch` (from `14a §8`)

These are reported by the owning doc's code ranges; `15` ensures its error-reporting context includes the `binding_id` where relevant (adapters of the `Diagnostic` location field fill this in).

### 11.4 Error location discipline

Every `15`-owned `CompileError` SHOULD carry a `Diagnostic.location: Option<Location>` pointing into the Model YAML source:

- Binding-shaped errors (`NoSourcesMatched`, `GlobExpansionFailed`, `CatalogUnavailable`) → point at the YAML `binding:` block or its `sources:` sub-key.
- SemanticMapping-shaped errors (`MissingBindingEntry`, `SpuriousBindingEntry`, `IncompatiblePhysicalType`) → point at the specific `semantic_mapping[<name>]:` entry.
- Reconciliation-shaped errors (`CrossSourceTypeDisagreement`, `CrossSourceTypeDisagreement`) → point at the Binding; the `types: Vec<(usize, DataType)>` field enumerates per-source divergence.

The precise `Location` / `ByteSpan` shape is ratified in `30 §5` and `32 §?` (SourceId variant). `15` stipulates only the semantic target.

### 11.5 Code-range governance

`15`'s allocation adds twenty-one `COMP_E_`* codes (`COMP_E_0203, 0301-0322`) and seven advisory codes (three `COMP_I_*`, four `COMP_W_*`). All are within the `COMP_E_0300-0399` / `COMP_W_0300-0399` reservation for "schema / binding (per `15`)" in `30 §6.2`. Three of the codes (`COMP_E_0313`, `COMP_E_0314`, `COMP_E_0322`) are partition-extraction-related — `0322` is emitted in v1 as the v2-deferral guard for `partition: Some(_)`; `0313` and `0314` are reserved (no v1 emitter) and will activate when the partition arm lands. Adding further `15`-owned codes is MINOR per `30 §6.3`; the sub-range has remaining space for roughly 47 additional codes before coming close to the `COMP_E_0400-0499` neighbor (relationships / index build).

The `CatalogUnavailable (COMP_E_0203)` code sits in the `COMP_E_0200-0299` catalog/source-resolution range per `30 §6.2`; `15` owns the schema/binding range but is a **consumer** of catalog-resolution errors, so it re-surfaces them at their owning range rather than re-numbering.

## 12. Interaction with Other Documents

### 12.1 `19 §3.2` — `ResolvedExprTable` keying

`19 §3.2` keys its global `ResolvedExprTable` on `(SemanticsName, EntityId)` (the binding's `EntityId`). `15 §2.2` ratifies binding identity (deterministic UUIDv5, `33 §9.1`); the two docs are tightly coupled and should be read together. `19`'s table keys are stable across recompiles for unchanged binding content, exactly as `15`'s identity derivation guarantees.

### 12.2 `16` — Coverage at the `ComposedSemanticInterface` level

`16` extends `Coverage` from the Binding-level X-axis (per-source) to the composition-level axis (per-constituent-DataKind). Concretely:

- `15`'s `Coverage` answers "for Binding *B* with sources `[s0, s1, s2]`, which Semantics does `s1` cover natively vs NullFill?"
- `16`'s `Coverage` answers "for Unionset *U* with constituents `[D0, D1, D2]`, which fields of the composed interface does `D1` cover natively vs NullFill?"

`16` consumes `15`'s Coverage as input: for a composition-level "D1 covers field *f*" decision, `16` looks up the constituent's Binding's coverage of the underlying Semantics. The decision logic is `16`'s; the input data is `15`'s.

### 12.3 `20` — DataKind lifecycle integration

`20` ratifies how a `ResolvedDataKind` is constructed and where its `ResolvedBinding` attaches. `15 §10`'s flow produces a `ResolvedBinding`; `20`'s `compile` driver splices it into the `ResolvedDataKind::Simple` variant's `bindings: Vec<ResolvedBinding>` field (which, per `15 §2.1`, has length exactly 1).

### 12.4 `33` — SemanticManifest persistence of `ResolvedColumnMapping`

`33`'s crate contract ratifies the SemanticManifest struct shape, including the persisted form of `ResolvedBinding` / `ResolvedColumnMapping`. Serialization format (whether `serde_json`, a custom binary, or both) is `33`'s choice; `15` only stipulates that the structural shape in §7.2 is preserved round-trip (per I4).

The v1 design pencils in `serde` derivations on all `15`-ratified types (with the `serde` feature in `semstrait-common` per `31 §10`). The `PhysicalExpr` inside `ResolvedColumnMapping.computed` serializes through `14`'s `Expr` serialization (`31` / `14 §9`).

### 12.5 `37` — `CatalogProvider` integration

`15 §3`'s `CatalogRef` is an opaque handle into `37`'s `CatalogProvider` registry. `15 §3.5.6`'s step 2 makes the provider calls; the async posture and error-enum shape are ratified in `37`. `15`'s `CompileError::CatalogUnavailable` wraps `37`'s `CatalogError` via `IntoDiagnostic` (per `30 §5`).

### 12.6 `21`–`25` — Per-DataKind strategies consume Bindings

- `21 Dataset` — a Dataset (Simple) has exactly one Binding; its strategy is "scan the Binding's sources, read the `ResolvedColumnMapping`, emit a `Scan → Project` sub-plan."
- `22 Grainset` — levels resolve to child DataKinds; each level reads its child's Bindings. Grain-selection reads per-child Binding Coverage.
- `23 Unionset` — branches have Bindings; the per-branch NULL-fill is emitted based on Binding Coverage.
- `24 Joinset` — members have Bindings; the join path is composed via `16`'s Relationships; per-member scan plans read from member Bindings.
- `25 Applicability matrix` — the per-variant table explicitly notes which Binding properties are consumed by which strategy.

### 12.7 `10 §3.3` — Placement in `compile`

`15 §10` is the per-Binding sub-sequence of `10 §3.3`'s "source resolution → schema resolution → name resolution → expression compile → index build" pipeline. `15 §10.9` locates the sub-sequence precisely.

## 13. Ratified Decisions Index

A Q-numbered roll-up of every choice `15` ratifies in Round 1. Each entry cross-references the owning section; the `status` column marks whether the decision is fully ratified (`✓`) or has a parked companion question (`?` → see [`questions/open/15_questions.md`](../../questions/open/15_questions.md)).

| #   | Decision | Ratified in | Status |
| --- | --- | --- | --- |
| R1  | `Binding` is a struct with `binding_id`, `sources`, `semantic_mapping`, `coverage` fields; `#[non_exhaustive]`. | §2.1 | ✓ |
| R2  | Binding identity is a deterministic content-derived `EntityId` (UUIDv5 over `(data_kind_id, source_id, mapping)`, `33 §9.1`); there is no `BindingId` newtype. Content-derivation makes it cross-run / cross-edit stable; the earlier per-compile `u32`-counter proposal is superseded. | §2.2 | ✓ (Q-MAP-001 mooted 2026-05-29) |
| R3  | `PhysicalSource` has three variants: `File`, `Table`, `Snapshot`. Enum is `#[non_exhaustive]`. | §3.1 | ✓ |
| R4  | Every `PhysicalSource` variant carries `schema: Schema` and `partitions: Vec<PartitionColumn>`. | §3.1 | ✓ |
| R5  | `Schema.columns` is an ordered `Vec`; order is source-native. | §3.2 | ✓ |
| R6  | `CatalogRef` is opaque; `catalog_id` routes to a provider, `fqn` names the table. | §3.3 | ✓ |
| R7  | `PartitionColumn.position` is 1-indexed. | §3.4 | ✓ |
| R8  | Partition-transform record lives in `37`'s catalog response, not on `PartitionColumn`. Partition info never reaches `35 ScanNode`; manifest-side carriage only. | §3.4 / §3.5.4 / `35 §4.2.1` | ✓ (Q-MAP-002 closed 2026-04-28) |
| R9  | Source resolution is deterministic (lexical sort after provider enumeration). | §3.5.6 | ✓ |
| R10 | Zero matches for a glob entry is a compile error (`COMP_E_0301`). | §3.5.7 | ✓ |
| R11 | Cross-source schema hard-agreement is a compile error (`COMP_E_0317`); soft-agreement is `Coverage::NullFill`. | §3.5.8 / §6.2 | ✓ |
| R12 | `FileFormat` v1 set: `Parquet`, `Csv(CsvOptions)`, `Json(JsonOptions)`, `Orc`, `Avro`. | §4.1 | ✓ |
| R13 | CSV schema resolution: declared-first, then header-derived; columns are `String` unless declared. | §4.4 | ✓ |
| R14 | JSON schema resolution: declared-first, then sample-inference (scalar-only). | §4.4 | ✓ (Q-MAP-004 closed 2026-04-28) |
| R15 | Format is inferred from file extension when glob spec has no explicit format. | §4.5 | ✓ |
| R16 | Mixed formats in one glob are a compile error (`COMP_E_0304`). | §4.5 | ✓ |
| R17 | `SemanticMapping.entries` is a `BTreeMap<SemanticsName, SemanticMappingValue>`. | §5.1 | ✓ |
| R18 | `SemanticMappingValue` has four variants: `Column`, `Literal`, `Expr`, `Metadata`. Enum is `#[non_exhaustive]`. The `Metadata` variant is compile-synthesized only (§5.5); it has no author-facing YAML under `semantic_mapping:`. | §5.1 / `18 §10` | ✓ |
| R19 | Every Semantics in `SemanticInterface` appears exactly once in `SemanticMapping`. | §5.6 | ✓ |
| R20 | Derived Measures synthesized from `Constraint::DerivesFrom(Key)` are filled in at compile time. | §5.6 | ? Q-MAP-003 |
| R21 | `Coverage` keyed on `(source_index, semantics)`. Default `Native` is not stored. | §6.1 | ✓ |
| R22 | `CoverageVariant`: `Native`, `NullFill`, `Derived`, `Metadata`; enum `#[non_exhaustive]`. | §6.1 | ✓ |
| R23 | `Derived` is a distinct variant from `Native`. | §6.3 | ✓ (Q-MAP-005 closed 2026-04-28) |
| R24 | `NullFill` on a non-Unionset-tolerant consumer is a compile error (`COMP_E_0310`). | §6.1 | ✓ |
| R25 | Binding-level Coverage is `15`'s; composition-level Coverage is `16`'s. | §6.4 | ✓ |
| R26 | `ResolvedColumnMapping` splits variants into four flat HashMaps. | §7.2 | ✓ |
| R27 | Per-Binding `computed` HashMap duplicates `ResolvedExprTable` entries (storage choice). | §7.5 | ✓ (Q-MAP-006 closed 2026-05-28) |
| R28 | `path.token` is 0-indexed post-scheme-strip; segments are slash-delimited; leading-/-collapsed. | §8.1 | ✓ (Q-MAP-007 closed) |
| R29 | `path_token` returns the whole raw segment, not the `=`-suffix value. Authors compose value-after-`=` via a separate Computed Dimension calling `substring_after`. | §8.1.1 | ✓ (Q-MAP-008 closed) |
| R30 | `partition.level: N` is 1-indexed. | §8.2 | DEFERRED v2 (§8.0; COMP_E_0322 v1 guard) |
| R31 | Hive-style partition extraction result-type contract. | §8.2.1 | DEFERRED v2 (Q-MAP-009 deferred) |
| R32 | Partitioning agreement is strict across Binding sources for referenced levels. | §8.2.2 | DEFERRED v2 (§8.0) |
| R33 | Widening casts emit `COMP_I_0301`; narrowing emit `COMP_W_0302`. | §9.1 | ✓ |
| R34 | `Cast` is wrapped into `SemanticMappingValue::Expr`; no separate cast-needed flag. | §9.1 | ✓ |
| R35 | Computed-entry type inference reconciles via `Cast` wrap or `COMP_E_0316`. | §9.2 | ✓ |
| R36 | Cross-source type agreement is strict (`COMP_E_0317`). | §9.3 | ✓ |
| R37 | Nullability mismatch is a warning (`COMP_W_0306`), not an error. | §9.4 | ? Q-MAP-010 |
| R38 | `compile` flow for Bindings is a 7-step sequence (§10.1–§10.7). | §10 | ✓ |
| R39 | Structural checks accumulate; I/O and dependency checks fail-fast. | §10.8 | ✓ |
| R40 | `15`'s code range is `COMP_E_0300-0399` (schema/binding); catalog-availability errors re-surface in `COMP_E_0200-0299`. | §11 | ✓ |
| R41 | Every `15`-owned `CompileError` carries a `Diagnostic.location` pointing into the Model YAML. | §11.4 | ✓ |
| R42 | Expression / function / resolution errors from `14` / `14a` / `19` pass through without re-codification. | §11.3 | ✓ |
| R43 | The expression layer model has three strata: `model::UnresolvedSemanticExpr` (logical, layer 1, concept-owned by `14`, homed in `semstrait-model`), `ir::PhysicalExpr` (lowered SQL-equivalent, layer 2, concept-owned by `14`, homed in `semstrait-ir`), compile-time mechanics (layer 3, `15`-owned). `path_token` is the v1 inhabitant of layer 3; it is not a `PhysicalExpr` variant and not a `14a` registry function. | §1 banner / §8 | ✓ |
| R44 | `SemanticMappingValue::Metadata` is **compile-synthesized only** from a Dimension's `type: { metadata: ... }` block (per `13 §4.7` / `18 §4`). No author-facing `semantic_mapping:` YAML; no peer YAML for the variant. The synthesis runs as compile step 4.0 (§10.4), before the §5.6 completeness check. | §5.5 / §10.4 / `18 §10.4` | ✓ |
| R45 | `MetadataDimensionRecipe` carries `extraction: MetadataExtraction` + `data_type: DataType`. Per-source resolved `LiteralValue`s live on `ResolvedPhysicalSource.metadata_values: HashMap<SemanticsName, LiteralValue>`, not on the recipe (the recipe is global to the Binding; values vary per source). | §5.5 / §7.6 / `18 §10.4` | ✓ |
| R46 | The compile stage casts the layer-3 `String` extraction result to the recipe's declared `data_type:`. Cast failure is fail-fast at compile (`COMP_E_0321 MetadataCastFailed`); there is no plan-time or run-time deferral. | §8.1.2 / §11.1 | ✓ |
| R47 | v1 metadata extraction scope is **path-only**. Partition extraction (`13 §4.7` `partition: PartitionExtraction { level: usize }`) is deferred to v2 — the compile-pass guard `COMP_E_0322 MetadataPartitionDeferredV2` rejects `partition: Some(_)` until v2 ratifies the partition arm. The §8.2 / §8.3 / §8.4 sections are retained as v2 design parking. | §8.0 / §11.1 | ✓ |
| R48 | `[TD-MAP-METADATA-FOLD]` is **resolved (2026-04-27)**: the fold of `Metadata` into `Expr` is reversed; `Metadata` is restored as a distinct 4th variant of `SemanticMappingValue`. The author surface is the Dimension type's `metadata:` block, not a `semantic_mapping:` YAML sugar. The compile flow eagerly resolves per-source literals at compile (§10.5). | banner / §1 | ✓ |

## 14. Non-Goals

Explicit non-goals of `15`. Authors searching for these topics should look elsewhere:

- **Query-time execution** — engines execute; semstrait emits; `15` compiles the input to `adapt`, nothing more.
- **Schema drift against the SemanticManifest** — drift detection is a **query-time** concern, covered by `CatalogProvider::check_schema_drift` per I11 and ratified in `37` / `38`. `15` freezes a schema at `compile` time; any post-compile drift is detected by a separate entry point.
- **Partition pruning** — the `Coverage` captured in `15` records per-source applicability; partition-predicate pushdown is a planner optimization covered in `34 §5`.
- **Per-engine register hooks** — `DataFusionConnector::register_manifest_sources` and peers are adapter-layer concerns (`36`). `15` produces a `ResolvedBinding`; the adapter reads it and decides how to register it.
- **Catalog-provider-specific snapshot semantics** — Iceberg REST snapshot pinning is described in `37`; `15` consumes the resulting `SnapshotId` as an opaque value.
- **Statistics-driven optimization** — `15` never surfaces row counts, histograms, or cardinality estimates. These are future planner concerns.
- **Write paths** — every `PhysicalSource` variant is read-oriented. Write-side semantics (materialized-view refresh, pre-aggregation persistence) are outside semstrait's mandate per `00 §10`.
- **Multi-format heterogeneous sources in one Binding** — a Binding's glob resolves to one format; mixed Parquet + CSV in one glob is rejected by `COMP_E_0304`. A Unionset of two Bindings, each with its own format, is the supported pattern.
- **Column renaming across sources** — a `SemanticMappingValue::Column { name }` resolves to a single physical name across every source. Per-source name mapping is a `Coverage` / NullFill question; semstrait's v1 answer is "rename upstream, in your ingestion job."

## 15. Summary of Vocabulary Anchors

For quick lookup when other docs reference `15`:


| Term                      | Shape (§ref)                                                                                    |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| `Binding`                 | struct §2.1 (identified by `id: EntityId`)                                                       |
| binding `EntityId`        | deterministic UUIDv5 §2.2 (`33 §9.1`)                                                            |
| `PhysicalSource`          | `enum` File/Table/Snapshot §3.1                                                                 |
| `Schema`                  | struct with `Vec<SchemaColumn>` §3.2                                                            |
| `SchemaColumn`            | struct `{ name, data_type, nullable }` §3.2                                                     |
| `CatalogRef`              | struct `{ catalog_id, fqn }` §3.3                                                               |
| `PartitionColumn`         | struct `{ name, position (1-indexed), data_type, nullable }` §3.4                               |
| `FileFormat`              | `enum` Parquet/Csv/Json/Orc/Avro §4.1                                                           |
| `CsvOptions`              | struct §4.2                                                                                     |
| `JsonOptions`             | struct §4.3                                                                                     |
| `SemanticMapping`         | struct holding `BTreeMap<SemanticsName, SemanticMappingValue>` §5.1                             |
| `SemanticMappingValue`    | `enum` Column/Literal/Expr/Metadata §5.1 (struct-owned by `18 §10`)                             |
| `MetadataDimensionRecipe` | struct `{ extraction, data_type }` (`18 §10.4`)                                                 |
| `MetadataExtraction`      | `enum` Path { token: u32 } (v1; v2 adds Partition) (`18 §10.4`)                                 |
| `Coverage`                | struct holding `HashMap<CoverageKey, CoverageVariant>` §6.1                                     |
| `CoverageVariant`         | `enum` Native/NullFill/Derived/Metadata §6.1                                                    |
| `ResolvedBinding`         | struct §7.6                                                                                     |
| `ResolvedPhysicalSource`  | struct `{ source: PhysicalSource, metadata_values: HashMap<SemanticsName, LiteralValue> }` §7.6 |
| `ResolvedColumnMapping`   | struct with four flat HashMaps + coverage §7.2                                                  |
| `ResolvedLiteral`         | struct `{ value, data_type }` §7.2                                                              |
| `path_token`              | layer-3 compile-time mechanic (free fn): `(path: &str, token: u32) -> Result<String, _>` §8.1   |


Everything in this table is `pub`-visible in `semstrait-manifest` (post-resolve) or `semstrait-model` (pre-resolve), per `30 §4` / `33`'s final roster.

---

**End of document.** Open reconciliation items live in `docs/design/questions/open/15_questions.md`; ratified items in `docs/design/questions/closed/15_questions.md`; post-v1 items in `docs/design/questions/deferred/15_questions.md`.

## Testing Strategy

- Error-code table completeness: all 22 codes in §11.1 fire in test corpus
- Warnings: all 7 diagnostics in §11.2 emit correctly
- Glob determinism: I4 byte-identical SemanticManifest under source reordering
- Completeness check: §5.6 MissingBindingEntry/SpuriousBindingEntry
- Cross-source type agreement: §9.3 COMP_E_0317
- Binding-struct catalog-absent fallback: §2.5 override matrix (C14.4/C14.5), COMP_W_CATALOG_OVERRIDE, required-field-absent detection
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
