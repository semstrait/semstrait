---
number: 46
title: Physical sources are flat, specialization sits in the catalog crate, and schema provenance is explicit
status: accepted
date: 2026-08-14
source:
- CONTEXT.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
links:
- target: 12
  kind: amends
- target: 41
  kind: amends
- target: 61
  kind: amendedby
- target: 70
  kind: amendedby
- target: 85
  kind: amendedby
- target: 90
  kind: amendedby
- target: 94
  kind: amendedby
---

# Physical sources are flat, specialization sits in the catalog crate, and schema provenance is explicit

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0061](0061-a-source-schema-may-be-partial.md): a persisted column's canonical type and nullability are optional. In its model-fed sections, two sentences change, and one rejected alternative with them. "The derivation is: the `ColumnRef` leaves of every expression whose `physical_sources` includes this source, **typed from the referencing semantic definition's mandatory `data_type`**." The leaf set survives verbatim as the derivation of the column *set*. The typing clause is struck: a referencing definition's declared type is a claim about the *definition's result*, not about any leaf beneath it. "**`nullable` is `true` on the model-fed arm, meaning 'unknown, treated as nullable'.**" That reading is struck. It was forced by `ir::SchemaColumn.nullable` being a mandatory `bool`. With the manifest carrying its own column type, unknown is spelled `None` and no longer has to be smuggled into a value a consumer will read as an observation. 0046's rejected alternative "**`nullable: false` on the model-fed arm** — claims non-nullability on no evidence" is preserved and generalised: `Some(true)` on that arm claims *nullability* on no evidence, in the same class of fault, and is now equally refused. The carrier is the manifest's own `SourceSchema`, not `ir::Schema`, and it is not a breach of the no-vocabulary rule; the retired native-spelling `SourceColumn { name, source_type: String, nullable }` is not what returns (0061's *The manifest gets its own carrier, and `semstrait-ir` does not move*). `CompileError::UnmappableNativeType` is not deleted but demoted to a Warning-severity diagnostic, emitted only on the catalog-observed arm (0061's *An unmappable native spelling stops the compile no longer*).
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): for path sources, "one `PhysicalSource` per expanded path, not per glob root" is replaced by one source per base path and format. A folder-position wildcard still multiplies sources. Expansion stays at compile, and the narrowing of compile determinism stands. Compile folds every semantic source over one physical source into its listing and, for a model-fed source, its column set, where it kept the first write.
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): in its *Decision Outcome*, "Every type it persists is owned by `semstrait-ir`, `semstrait-model`, or `semstrait-catalog`" also names `semstrait-sql`, which owns the tree, `Literal`, `NamedReference` and `DataType`; in *The schema is canonical; the native spelling is catalog-owned provenance*, the canonical `DataType` is `semstrait-sql`'s. The rest of 0046 stands.
> - [ADR 0090](0090-semstrait-ir-keeps-the-plan-and-its-parameter-and-the-manifest-owns-the-expression-id-and-the-expression-layer.md): In its *Decision Outcome*, "Every type it persists is owned by `semstrait-ir`, `semstrait-model`, or `semstrait-catalog`" also names `semstrait-manifest`, for its own storage types: the expression id and the expression layer (ADR 0090, rules 4 and 5), and a stored expression and its link (ADR 0088, rule 4). In *The schema is canonical; the native spelling is catalog-owned provenance*, the sentence on the "`pub use semstrait_model::{DataType, Grain}` re-import" of `ir::types` becomes: `semstrait-ir` re-exports neither type; `DataType` is `semstrait-sql`'s and `Grain` is `semstrait-model`'s, and a crate that uses either imports it from there (ADR 0090, rule 6). The rest of 0046 stands.
> - [ADR 0094](0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md): *Decision Outcome*: "the manifest is a storage standard, not a vocabulary owner. Every type it persists is owned by `semstrait-ir`, `semstrait-model`, or `semstrait-catalog`", as ADR 0090 last amended it ("also names `semstrait-manifest`, for its own storage types"); and "`Grain` is `semstrait-model`'s, and a crate that uses either imports it from there" (ADR 0090's words, near 0046:37): the names and `Granularity` are `semstrait-common`'s (rules 2 and 3); `semstrait-ir` owns `Integrity`, `CrossFilter`, `AdditivityType` (with `SemiAdditivity`), `SemiAdditivityStrategy`, `ScdType`, `TemporalShape`, `TemporalType`, `JoinsetJoinType` and `SemanticRole`, which the manifest imports; the manifest owns its own `DimensionType` and `AiContext` (rule 4); and stores the aggregation as `semstrait-sql`'s function and a `distinct` flag (rule 5).

## Context and Problem Statement

A proposed design was a variant hierarchy under `PhysicalSource`: `IcebergPhysicalSource`, `ParquetPhysicalSource` and `DeltaPhysicalSource`. The ruling must also say whose types the persisted schema uses, and how a consumer tells a schema *observed* from a catalog from one *assumed* from the model.

## Considered Options

- A flat `PhysicalSource` with typed `StorageProperties` and a two-arm schema-provenance discriminant (chosen).
- **`PhysicalSource → IcebergPhysicalSource / ParquetPhysicalSource / DeltaPhysicalSource`**, the proposal as first posed. Refused because the specialization already exists and is already normalized in `semstrait-catalog`; the three "concrete forms" are values of one `format` field, since `FileFormat` already spells `Parquet | Orc | Avro | Csv | Json | Delta | Iceberg`. It would also place specialization *beside* the type, against the glossary's own rule.
- **A trait object for the concrete form.** Unserializable without a bespoke codec, against the discipline of opaque-per-DTO storage, and it would put dynamic dispatch in a persisted artifact.
- **A native-spelling-only schema, narrowing the Canonical-layer clause to admit it.** This session's first recommendation, withdrawn on retrieving the clause both types cite: an earlier clause ratifies canonical `DataType` on this exact carrier, so it would have entrenched the divergence `semstrait_catalog::SchemaColumn` already represents.
- **A manifest-minted `native_types` sidecar** keyed by column name, for the native type spelling. This session's second recommendation, refused on the no-vocabulary principle rather than on taste: it repeats `SourceColumn`'s error — the manifest inventing a type — under a new name.
- **Extending `ir::SchemaColumn` with the native spelling.** Breaks its declared field-stability and puts a native type inside the canonical-vocabulary crate, which is the one thing the Canonical-layer entry most directly forbids.
- **Dropping the native spelling entirely.** Cheapest, and loses the only answer to "what did the provider actually say" when a canonicalisation looks wrong.
- **Inferring the fork from the absence of a catalog descriptor.** The `JoinKeySpec::Cross`-as-empty-vector pattern; it makes the artifact's most consequential distinction a silent one.
- **Collapsing schema provenance onto the earlier `{File, Table, Snapshot}` axis.** A Table source may be either fork, so target kind does not determine provenance.
- **Keeping `schema_fingerprint` beside `schema_id`.** Two fields for one fact, and the fingerprint's content-hash rule survives anyway as the minting rule for providers that supply no id.
- **`nullable: false` on the model-fed arm.** Claims non-nullability on no evidence.
- **Keeping `locator: String` and continuing to parse it at each use.** The status quo, and already a fiction: `table_ref_from_locator` splits the string on `.` to recover catalog, namespace and table name inside a function whose own doc cites the rule forbidding exactly that.
- **A `provider_metadata` bag for the catalog fields.** Already present, with zero writers and zero readers and `{}` in the committed snapshot fixture; ADR 0043 refused the passthrough class outright.
- **Reserving a statistics carrier**, as ADR 0041's "optionally statistics" invited. `ResolvedTable` has no statistics field and no provider supplies any, so the carrier could not be populated — headroom with no authoring path.
- **Translating every catalog column to canonical, not only referenced ones.** Makes an unreferenced exotic type fail an unrelated compile.

## Decision Outcome

`PhysicalSource` stays a flat struct. It grows a typed `StorageProperties` and an explicit two-arm discriminant for **where its schema came from**, and it grows no variant hierarchy: the `PhysicalSource → IcebergPhysicalSource / ParquetPhysicalSource / DeltaPhysicalSource` split that was proposed is refused, because the specialization it asks for already exists one crate down and is already normalized.

The ruling rests on a principle stated by the human during the grilling and adopted here as the general rule: **the manifest is a storage standard, not a vocabulary owner.** Every type it persists is owned by `semstrait-ir`, `semstrait-model`, or `semstrait-catalog`. That single rule decides most of what follows, and it is what makes the flat shape sufficient.

### It is catalog specialization, not engine specialization

**The word was wrong and the glossary was wrong with it.** `CONTEXT.md`'s **Physical source** entry ends "Engine-specific specialization sits beneath it, not beside it", but `CONTEXT.md` reserves *engine* for the compute target: the **Engine artifact** entry calls emission "the first place engine-specific representation is legal", and the **Canonical layer** entry forbids "SQL, native type, or engine identity" in the middle of the system where the manifest lives. So the glossary asserted, of the manifest, exactly what it forbids two entries up. The subject is a `CatalogProvider` / catalog spec, and the entry is corrected to say so.

**"Beneath, not beside" is satisfied by the crate stack and violated by a variant enum.** `semstrait-catalog` holds `IcebergRestCatalogProvider` and `UnityCatalogProvider` behind one `CatalogProvider` trait, and every provider normalizes to a provider-neutral `ResolvedTable { table, location, format, schema, partitions, snapshot }`. `Partition` already carries `PartitionTransform` — `Identity | Year | Month | Day | Hour | Bucket(u32) | Truncate(u32) | Void` — which *is* the Iceberg partition spec, canonicalised. Specialization therefore already sits beneath `PhysicalSource` in the dependency stack; a sibling-variant enum on the manifest type would put it beside.

**Polaris is not a second concrete form, so the earlier "Polaris and Iceberg are priority" does not imply two carriers.** `iceberg.rs` states the provider is "Compatible with Polaris (Snowflake), Gravitino, Tabular, and other Iceberg REST catalog servers", and the model's `CatalogEntry` carries a `realm` field documented "Provider-specific realm (Polaris uses this; most others ignore)". Two named priorities are one wire protocol plus a configuration field. The proposal's own "Iceberg is the only one on the radar" was closer to the truth than the correction it received.

### The manifest owns no vocabulary, so `SourceColumn` retires

**`ir::Schema` is not merely a permissible reuse — it is the ratified carrier, and the manifest's own type is the divergence.** `ir::Schema`'s doc reads "Per-`PlanNode` output-shape carrier; **also attached to `PhysicalSource`**", and the earlier carrier clause ratifies:

```rust
pub struct SchemaColumn { pub name: ColumnName, pub data_type: DataType, pub nullable: bool }
```

Canonical `DataType`. `ir::SchemaColumn` matches that shape; `semstrait_manifest::SourceColumn { name, source_type: String, nullable }` does not, and neither does `semstrait_catalog::SchemaColumn { name, data_type: String, nullable }` — which cites that clause in its own doc while contradicting it. So persisting canonical types **restores a ratified clause** rather than choosing a new one. Reusing `ir::Schema` also removes a conversion at the seam that matters: `PlanNodeMeta.output_schema` is `Arc<Schema>` of the same type, so a persisted source schema and a plan node's output shape stop being two types describing one thing.

### Schema provenance is an explicit two-arm discriminant

**The fork is legible in the sealed artifact, and it is not inferred from a field's absence.** A consumer must be able to tell whether a source's schema was *observed* from a catalog or *assumed* from the model. Two arms, closed by construction: a catalog was supplied or it was not.

**Two earlier precedents pull opposite ways, and the tie-break is whether a consumer branches on it.** ADR 0045 deleted `DataKindOrigin` because implicitness "is not a planner fact"; the delta pass on the carrier inventory flagged encoding a state as absence — `JoinKeySpec::Cross` as an empty vector — as how a manifest yields a wrong query instead of a rejected compile. Provenance-of-schema differs from implicitness in having consumers that branch: **drift re-validation is meaningless on a model-fed source**, since ADR 0041 records the orchestrator walking persisted sources calling `CatalogProvider::check_schema_drift` per source, and a schema derived from demand cannot meaningfully be checked against a catalog it never read; and **`NullFill` detectability**, the question the coverage-mask decision is blocked on, is precisely "was this column observed or assumed".

**The discriminant carries each arm's payload, which is what retires the always-`Some` / always-`None` field set.** A flat struct with optional catalog descriptors cannot express "this fork has these fields"; a payload-carrying enum makes "model-fed source holding a catalog descriptor" unrepresentable rather than merely wrong. The cost is a public `#[non_exhaustive]` enum consumers match on, so a third fork would be a breaking match — acceptable because the fork set is closed by construction.

**This axis is not the earlier target-kind axis, and the two must not be collapsed.** The earlier design ratified `PhysicalSource` as a sum type `{File, Table, Snapshot}` — evidence that an enum *mechanism* has precedent here, on the axis of **where the data is**. This ADR's enum discriminates **where the schema came from**. The axes correlate without determining one another: a File source is always model-fed (verified — `build_source` gives file sources no catalog round-trip, because the trait resolves tables), while a Table source is catalog-derived only if a catalog was supplied. Target kind therefore cannot stand in for provenance, which is the second reason the discriminant is explicit rather than derived. The target-kind axis is superseded on carrier grounds: the target identity becomes typed fields on `StorageProperties`, and `PhysicalSourceType { Table, File }` — itself the flattened remnant of those three variants — retires with the `locator` it was minimised alongside.

### The schema is canonical; the native spelling is catalog-owned provenance

**The canonical column set is `ir::Schema`. The native spelling is preserved for debugging, stored as the catalog's own type on the catalog arm.** Both halves follow the no-vocabulary rule: canonical vocabulary is `semstrait-ir`'s, native provider vocabulary is `semstrait-catalog`'s, and the manifest mints neither. `semstrait_catalog::Schema` already derives `Serialize`, and `semstrait-manifest` already depends on that crate, so no new edge is created. `ir::types` also gets a more precise spelling of its `pub use semstrait_model::{DataType, Grain}` re-import, documented so that its status as the canonical *resolved* vocabulary is legible at the type rather than only in prose.

**The native spelling is non-load-bearing provenance and must stay out of `schema_id`.** This is ADR 0041's `created_at` ruling applied to a second field: a provider that changes its type *rendering* — `int4` to `integer` — with no canonical change would otherwise flip `schema_id` and signal drift that did not happen. Nothing reads the native spelling to make a decision; it exists to answer "what did the provider actually say".

**That gives the Canonical-layer clause the narrowing it already needed independently.** The clause becomes: the canonical layer never *reads* a native type or a provider identity, though it may *record* either as non-load-bearing provenance. ADR 0041 had already routed "base catalog configuration metadata for identification and traceability" into the manifest, which is a provider identity the unnarrowed clause forbids — so the narrowing settles two instances at once instead of leaving each ruling to violate it quietly.

**`nullable` is `true` on the model-fed arm, meaning "unknown, treated as nullable".** `ir::SchemaColumn.nullable` is a mandatory `bool` and that fork observes nothing — its own doc says nullability "is sourced from upstream metadata when available; it is not inferred from data". `true` is the only direction that cannot mislead: a consumer never wrongly believes a column is non-null. `false` would claim non-nullability on no evidence, which is the dangerous-default failure. Detectability of assumed-versus-observed rides the discriminant, not the column, so `ir::Schema` stays intact at no cost to the coverage-mask decision.

### Inbound native→canonical translation is the catalog stage's, over referenced columns only

**Requiring canonical types creates an obligation with no current owner.** `semstrait-catalog` explicitly refuses it — "Column / partition types are captured as the native Iceberg type spelling… No canonical-type translation happens here" — and the catalog crate defers canonical mapping "to planner / engine registry". Only the outbound direction has ever been placed: `CONTEXT.md`'s **Canonical data type** entry has adapters mapping canonical to native at emission. Nothing translates inbound.

**Compile's catalog stage owns it, and raises one `CompileError` variant on an unmappable native type.** Under ADR 0044 a validity definition is constitutive of the stage that can produce the fault, and the catalog stage is where native metadata enters the system: metadata leaves that stage canonical or the compile fails there. The deferral to the planner is void under this ruling, because the persisted artifact must already be canonical and the planner can no longer be the first canonicaliser. This variant is also the successor ADR 0044 ordered: it deleted `CompileError::UnresolvedColumn` and specified that its replacement, "minted when the source-fork decision lands", carries `(column, expr_id, physical_source_id)`.

**The obligation is scoped to referenced columns.** The field is named `projected_schema` but `project_resolved_table` copies **every** column the catalog returns, so the name has never been accurate. Scoping to referenced columns follows the referenced-columns-only posture, makes the name honest, and stops an unreferenced exotic column — an Iceberg `map`, `struct`, `list`, `fixed[16]` — failing a compile that never reads it. The accepted cost is that drift on unreferenced columns becomes invisible, which is correct: a column never read cannot break a query.

**Two consequences for the persisted column list follow.** The earlier shape contract fixes column order as the source's native order and notes that the planner does not depend on it but **adapters do** (Substrait `RelRoot.names` ordinal alignment, SQL emit order); since the persisted set is now a projection, it preserves *relative* native order, which that contract already admits as "planner-derived projection order". And **column-name uniqueness becomes the manifest's check**: `ir::Schema` deliberately admits duplicates, its own test recording that "uniqueness is a higher-layer invariant (planner/manifest), not a `Schema` constructor concern". Under name-based column resolution a duplicate makes resolution ambiguous, so this is stage-reachable and currently ungated.

### The model-fed fork derives its columns from the expression pool

**The ratified wording has no subject and is restated in the live mechanism.** Both the earlier text and `CONTEXT.md` say columns are "assumed from the nearest reachable semantic interface intersected with the semantic mapping", but ADR 0043 retired `SemanticInterface` as an entity, so there is no interface to intersect.

**The derivation is: the `ColumnRef` leaves of every expression whose `physical_sources` includes this source, typed from the referencing semantic definition's mandatory `data_type`.** That is ADR 0043's own edge set — the semantic→physical edge is `override.expr_id`, the expression→source edge is `expression.physical_sources` — so no new edge and no interface entity is required. The resolved reference is a plain by-name reference to a schema element, `Column("name")`, which is what keeps definition and use simple.

**This makes ADR 0044's vacuity claim exact rather than asserted.** 0044 ruled G4's clause 2 — every `ColumnRef` resolves in the schema of every source the expression lists — "vacuous by construction, not deferred" on the model-fed fork, and left "the obligation to say which fork carries it" to this decision. It is exact because the column set *is* the referenced set: a dangling `ColumnRef` is unrepresentable on this fork. So the rule, and the successor diagnostic, are the **catalog arm's only**.

**Ordering consequence: this fork's schema is a function of the expression pool, so it is derived after expressions are interned** — not during the catalog step, which runs first and is the only I/O step.

### `schema_id` replaces `schema_fingerprint`

**One field, not two.** `schema_id` is derived from the metadata source where one can supply it — Iceberg's own schema id — and minted deterministically otherwise. It **replaces** `schema_fingerprint: Option<[u8; 32]>` rather than joining it; two fields addressing one fact is the defect this map keeps finding.

**The objection that a provider id is not comparable across sources does not apply, because the comparison is never cross-source.** The persisted schema exists as the object of *runtime synchronization* for its own source: the question is always "this source, now versus at compile", for which within-table uniqueness suffices and Iceberg's schema id is precisely the intended signal. `schema_fingerprint`'s content-hash semantics survive as the minting rule for the providers that supply no id, where the earlier design's determinism obligation binds.

### Glob expansion happens at compile, and compile determinism narrows again

**One `PhysicalSource` per expanded path, not per glob root.** This settles a question the earlier design explicitly parked: it gave one source per "table | unique file path / **glob root**", and the question of compile-time versus runtime expansion was listed as deferred. Expansion is at compile, one source per expanded path at the wildcard position, with no deeper recursion into the tree. The plumbing already exists and is deliberately dead: `compile()` takes `fs: &dyn FileSystem` and does `let _ = &fs;`, marked as reserved for deferred glob expansion.

**Compile determinism narrows, and the consequence is named rather than inherited.** Expanding at compile makes the source roster a function of filesystem *contents*, which the invariant that identical `(model, catalog snapshot)` inputs give the same artifact does not cover. ADR 0041 gated recompile on `model_hash` alone, so **a file appearing between two compiles changes the correct artifact while triggering no recompile, and the stale artifact keeps the same content-addressed key.** File sources thereby land in exactly the position catalog drift occupies — detected out of band by a caller-scheduled check — except that no file-drift verb exists. That gap is recorded, not closed here: like catalog drift, it sits with the caller, and this map ends at the artifact's definition.

### Snapshot tracking is deferred, with its reason

**`PhysicalSourceVersionRef` and the snapshot are deferred rather than retired, because an Iceberg snapshot does not change a schema.** Iceberg tracks schema evolution separately from snapshots, so a snapshot pin is not the signal the persisted schema needs; the schema id is. This is why `project_resolved_table` discarding the fetched `SnapshotMetadata` is *correct behaviour at this stage* rather than the wiring gap it first appears to be. It returns if and when time-travel or pinned reads are authored.

### Amends

- **0012:** `schema_id` replaces `schema_fingerprint`.
- **0041:** `PhysicalSource` reserves no statistics carrier, refusing the "optionally statistics" 0041 invited; glob expansion happens at compile, and compile determinism narrows again: a file appearing between two compiles changes the correct artifact while triggering no recompile, and the stale artifact keeps the same content-addressed key. File sources thereby land in exactly the position catalog drift occupies, detected out of band by a caller-scheduled check, except that no file-drift verb exists.

### Consequences

- **What retires in `crates/` — recorded for the implementation effort, not done here.** `SourceColumn`; `PhysicalSourceType { Table, File }`; `PhysicalSource.locator`; `provider_metadata`; `schema_fingerprint` and its `schema_fingerprint()` helper in `compile/catalog.rs`. `PhysicalSourceVersionRef` becomes dormant-by-decision rather than deleted. `semstrait_catalog::SchemaColumn.data_type: String` remains as the native-provenance type and is *not* converted — its contradiction of that clause is resolved by it no longer claiming to be the source's canonical schema.
- **What the catalog stage gains.** Native→canonical translation over referenced columns, one `CompileError` variant for an unmappable native type carrying `(column, expr_id, physical_source_id)`, and population of the four fields `project_resolved_table` currently discards — `location`, `format`, `partitions`, and the typed table identity — into `StorageProperties`.
- **Three checks are named for the roster of checks**, each with its owning stage: column-name uniqueness within a persisted source schema; every referenced `ColumnRef` resolving in the schema, on the catalog arm only; and relative-native-order preservation in the projection.
- **One divergence is left standing as a handoff note.** The earlier clause spells the column name as a `ColumnName(pub String)` newtype; `ir::SchemaColumn.name` is a bare `String`. Nothing in this ruling depends on the difference, and `ir::Schema` is field-stable, so it is recorded rather than reconciled.

## More Information

### Open questions

- **The type mapping table is not ruled here.** Which native Iceberg and Polaris spellings map to which canonical `DataType`, and which have no counterpart, is an external fact to be read from those specifications rather than guessed. It is a research question.
