# semstrait-manifest

The **semantic manifest** — in `CONTEXT.md`'s words, "the sealed, engine-agnostic
artifact in which every name, source, and expression has been resolved and
denormalized per semantic source, so planning reads it by id and needs no
other lookup or I/O" — plus the
model→manifest `compile()` pass that produces it and the artifact's own validator.

**Identity on disk, structure in memory.** The manifest persists identity
primitives in deterministic `EntityId`-keyed `BTreeMap` registries. The **semantic
graph** a consumer builds at setup rebuilds the composed surfaces and join
skeletons this artifact deliberately does not hold, and none of that runtime state
is ever serialized.

Planning is `semstrait-planner`'s; nothing here plans, and no
SQL appears anywhere in this crate.

This file is **current-state reference only**. Decision
authority is `docs/adr/`; terminology authority is `CONTEXT.md`.

---

## Module map

```
src/
  lib.rs                  the crate root's exports of its own items, and the crate's own
                          contract statement

  compile/                the model→manifest lowering engine — one module per stage
    mod.rs                  the stage roster, and what each stage answers
    driver.rs               `compile()`: the sequence, and nothing else
    sugar.rs                a one-child composition replaced by its child, under the
                            composition's name, unless its parent cannot hold the
                            child's data kind
    catalog.rs              the sole I/O step — what does the world say these locators hold?
    catalog/partitions.rs   the partitions that a `Prefix` discovers, and the model's
                            `partition_def` entries checked against them
    sources.rs              the physical-source roster: which relations, which canonical
                            schema, which schema provenance
    definitions/mod.rs      the semantics registry merge over the model's root pools
    definitions/graph.rs    the reference graph over one source's exposed names, and acyclicity
    expressions/mod.rs      realisation: emitting each covered semantics' tree into the
                            expression store, folded through semstrait-sql
    expressions/lower.rs    the body walk: leaf substitution; typing through semstrait-sql;
                            the cast of a tree to its semantics' declared type
    expressions/columns.rs  the type of each column the trees over a physical source read:
                            observed, else its read type, else `Unknown`
    expressions/bodies.rs   authored bodies lowered per source, leaves first
    expressions/projection.rs
                            the composed projection: one signature per semantics a
                            composed reading surface exposes, projecting the operand
                            column that serves it
    assemble.rs             which units exist, what each is called, what each can serve
    relationships.rs        an authored relationship becomes the pooled join condition of
                            the unit it caused
    keysets.rs              the keyset roster, and the per-appearance attribution
    temporal.rs             a composed temporal shape, derived bottom-up over the operands
    semi_additive.rs        a semi-additive Measure's `by`, written or defaulted, and its
                            axes, against the Datasets its source reads
    qualifiers.rs           every stored column names the source alias its place names
    seal.rs                 refuse an entry that no root names
    types.rs                the transient lowering vocabulary the stages share — not a
                            stage, and nothing in it is persisted

  manifest/               the persisted vocabulary — one module per persisted collection
    mod.rs                  the collection roster
    top.rs                  `SemanticManifest`, `ManifestProvenance`
    bitmap.rs               `SemanticBitmap` (the frozen registry), `SemanticBitmask`
    semantic_definition.rs  `SemanticDefinition`, `SemanticRoleDefinition`, `DimensionType`,
                            `AiContext`
    semantic_signature.rs   `SemanticSignature` — one source's implementation of a semantics
    coverage.rs             `Coverage` — the axis, computed and never carried
    semantic_source.rs      `SemanticSource`, `RoleDefinition`, `Selector`
    source_alias.rs         `SourceAlias` — the handle a semantic source and a physical
                            source are known by, and the rule that derives each
    physical_source.rs      `PhysicalSource`, `StorageTarget`, `StorageProperties`, `Uri`,
                            `HivePartitions`, `SourceSchema`, `SourceColumn`,
                            `SchemaProvenance`, `SchemaId`, `PhysicalSourceVersionRef`
    expression.rs           `ManifestExpressions`, `ManifestExpression`, `StoredExpr`, `Link`,
                            `PhysicalExprId`
    expression_layer.rs     `ExprLayer` — the expression layer each entry holds
    read_type.rs            `read_type`, `MappedSemantics` — the type an unobserved column is
                            read as, written into a model-fed schema
    keyset.rs               `Keyset`, `KeysetName`
    metadata.rs             `SemanticManifestMetadata`

  identity.rs             the id-*minting* side: the `mint_*` family, `mint_expr_id`
  error.rs                `CompileError` and its reference-site vocabulary
  validate.rs             `SemanticManifest::validate()`, `FaultReporting`
  repository.rs           `SemanticManifest`'s `Persistable` impl over the generic
                          persistence seam
```

---

## The persisted artifact

```rust
pub struct SemanticManifest {
    // ── identity block (ADR 0041, ADR 0055, ADR 0088) ──
    pub manifest_version: u64,
    pub created_at: SystemTime,          // caller-injected; never read to decide
    pub updated_at: SystemTime,          // caller-injected; never read to decide
    pub model_hash: [u8; 32],            // the model fingerprint, computed by the model
    pub model_id: EntityId,              // the compilation identity, minted from it
    pub dialect: DocumentDialect,        // names each stored call's function: `canonical`

    // ── entity collections ──
    pub semantic_definitions: SemanticBitmap,                  // frozen registry
    pub semantic_signatures: BTreeMap<EntityId, SemanticSignature>,
    pub semantic_sources: BTreeMap<EntityId, SemanticSource>,
    pub sources: BTreeMap<EntityId, PhysicalSource>,
    pub expressions: ManifestExpressions,                      // the expression store
    pub keysets: BTreeMap<EntityId, Keyset>,

    pub metadata: SemanticManifestMetadata,
}
```

Persisted types are `#[non_exhaustive]` and reached through a constructor
(ADR 0007), so a struct literal is unavailable across the crate boundary and
adding a field is not a breaking change for a caller. `SourceSchema` and
`SourceColumn` are the deliberate exceptions: both are plain structs with public
fields, so a producer builds one by literal.

The identity block is exactly those six fields, and **no reverse index is
persisted** (ADR 0055) — in particular there is no name-to-bit-position map,
because `semantic_definitions` is the one authority for a position and a second
one would have to be held in agreement with it forever. `model_hash` is the model
fingerprint; `CONTEXT.md` **Model fingerprint** forbids "checksum" as a spelling
for it.

`manifest_version` is `4`, moved when the persisted *structure* changes, never by
content and never by the clock; it moved to `4` with ADR 0070's shape, and every
later shape change of the planner effort rides that one move (ADR 0040, *The
epoch*; ADR 0043). `dialect` is `semstrait-sql`'s `DocumentDialect`, and each stored call writes
its function's name in it (ADR 0088, rule 8). Compile writes `canonical`, the one
dialect for now, and each stored expression is written in `canonical` whatever the
field holds, so reading and `validate()` refuse any other value.

### `SemanticDefinition` — what a semantics **means**

One entry per semantics name, keyed by an id minted per role from the bare name
(ADR 0065).

```rust
pub struct SemanticDefinition {
    pub id: EntityId,
    pub name: SemanticsName,
    pub data_type: Option<DataType>,
    pub role_definition: SemanticRoleDefinition,  // Dimension | Measure | Metric | Filter
    pub description: Option<String>,
    pub ai_context: Option<AiContext>,
    pub bit_position: u32,                        // assigned by the freeze (ADR 0036)
}
```

A definition owns no expression and no keyset: per-source expression and
per-site `is_key:` attribution ride `SemanticSignature` instead (ADR 0039).

`bit_position` is the dense zero-based index the semantics occupies in this
manifest's bitmap, assigned in canonical `EntityId` order inside
`SemanticBitmap::freeze` and meaningful only inside the manifest that declares
it. It is not an identifier: `EntityId` is the durable handle, and a mask may not
be carried across compiles. `guard_registry_size` bounds the registry at
`n <= u32::MAX` — an `n`-entry registry occupies `0..n-1`, so the largest position
is `n - 1` — and refuses anything past it with
`CompileError::BitmaskValueOverflow` rather than clamping.

### `SemanticSignature` — how a semantics is **obtained**, per source

Exactly one signature per `(source, semantics)` pair — ADR 0043's
disjoint-and-covering invariant as ADR 0049 inverted it and ADR 0065 made total:
exactly one per set bit of the owning source's mask, and none for an unset bit.

```rust
pub struct SemanticSignature {
    pub id: EntityId,                            // minted from `(source, definition)`
    pub definition_id: EntityId,
    pub expression_id: PhysicalExprId,           // the pool handle, and nothing else
    pub filter_expr_id: Option<PhysicalExprId>,  // a Measure/Metric `filter:`, a disjoint root
    pub keysets: Vec<EntityId>,                  // empty = no declaration
}
```

**No source id.** Which source owns a signature is stated by that source's
`semantic_signature_ids` and by nothing on the signature, so there is one authority
rather than two to hold in agreement; `validate()` states the rule from the list's
side — exactly one entry lists any signature.

Every covered semantics emits, so a signature carries a handle rather than a
realisation kind (ADR 0037), and the handle is mandatory. Every reading source owns
signatures — Dataset, Joinset (implicit units and cross-grain units included) and
Unionset all list one per set bit of their `semantic_bitmask`, **for every role**: a
composed Metric is its site's body lowered on the composition, linking the
composition's own signatures (ADR 0074, *A Metric*; ADR 0088, rules 5 and 14), and a
named filter is its body lowered on the composition (ADR 0077), but on an Implicit
Unionset, which reads each reduction leaf's column for both as for every other name
(ADR 0049). A name both operands of a Joinset map reads `Coalesce` in the order of ADR
0074's table. A Joinset lowers on itself a Measure that no operand has a signature for,
whose body holds the Joinset's own tree for each name it reads and no link (ADR 0077, *A
semantics with an `expr:` body*; ADR 0088, rule 5). A composed **Measure** or
**Metric** additionally carries a `filter:` root: the composition's own filter, which on
a Joinset holds ADR 0077's copy for each operand. Each operand's own filter stays on
that operand's signature (ADR 0071, as it amends ADR 0065 D2). A reading source's mask
is derived from those signatures and from nothing else — a set bit and a signature are
one fact, so no operand's bits are folded into a composer that reads. A Grainset owns a
signature only for a Metric that no child maps whole and that its children can compute
together (ADR 0084, rule 10): it routes every other name rather than reading it, so
`semantic_signature_ids` lists only those Metrics, and its mask stays the folded union of
its grains for routing, with the bit of each Metric it computes. Composite keysets (ADR
0048) ride the composite's own signatures, so a Joinset's or Unionset's keyness has a
carrier; a Grainset owns no signature for its keys, so its author's keyset declaration
rides `RoleDefinition::Grainset::keysets` instead — a bare `Vec<EntityId>` of keyset ids, the
same carrier shape as `SemanticSignature::keysets`, populated after ADR 0048's boundary
check and checked by `validate()` for roster resolution. Membership is not recoverable
from this carrier, by ruling (ADR 0065 D5).

`SemanticManifest::coverage_of(semantic_id, data_kind_id)` answers `Coverage` —
following the named source's own signature for that semantics into the store —
`Native`, `Derived` or `NullFill` — as a total function over the artifact. Per
ADR 0059 and `CONTEXT.md` **Coverage**, the axis is read *against a
composition*: a bare `NullFill` is not a claim that the value is null-filled, so
name the composition as well as the contributor.

### `SemanticSource` — the manifest's carrier for one data kind

One flat, name-keyed, acyclic collection in which **every** unit is an ordinary
entry (ADR 0045): an authored top-level data kind, an authored nested child, a
reduction leaf, a synthesised Unionset and every enumerated implicit Joinset
alike. Nesting stops being a type — composition is an `EntityId` into this same
map, carried on a polymorphic `RoleDefinition`.

```rust
pub struct SemanticSource {
    pub id: EntityId,                    // authored id, else UUIDv5 over the scope-qualified name;
                                         // a reduction leaf's is `mint(Dataset id, physical-source id)`
    pub name: DataKindName,              // bare at the top level, `<parent>.<name>` when nested;
                                         // a reduction leaf's is the label `<Dataset name>_<physical-source name>`
    pub alias: SourceAlias,              // `ss`, the source role's letter and the `EntityId`-order index (ADR 0078)
    pub public: bool,                    // may a request name this in its `from`?
    pub semantic_bitmask: SemanticBitmask,
    pub temporal: Option<TemporalShape>,
    pub role_definition: RoleDefinition,
    pub originators: Vec<EntityId>,      // the nearest authored ancestor; empty = authored top-level
    pub semantic_signature_ids: Vec<EntityId>,
}

pub enum RoleDefinition {
    Dataset  { source_id: EntityId, selector: Selector },  // resolves in `sources`, not in this collection;
                                         // Selector::{Prefix, Pattern(Uri), Uris(Vec<Uri>), Table} (ADR 0070)
    Joinset  {
        from: EntityId, to: EntityId,
        on_expr_id: Option<PhysicalExprId>,  // None iff join_type is Cross (ADR 0066)
        filter_expr_id: Option<PhysicalExprId>,
        join_type: JoinsetJoinType,          // Join(semstrait_ir::JoinType) | Cross
        cardinality: Cardinality,
        integrity: Integrity, cross_filter: CrossFilter,
        description: Option<String>, ai_context: Option<AiContext>,
    },
    Unionset { branches: Vec<EntityId>, mode: MultisetOp, null_fill: bool },  // mode: "all" | "unique"
    Grainset { grains: Vec<EntityId>, keysets: Vec<EntityId> },
}
```

**There is no `origin` tag and no recursive variant to descend.** Implicitness is
not a planner fact — nothing in the resolution path branches on whether a human
wrote an entry — and a consumer reaches every unit through this one map rather
than by walking into a parent.

`semantic_bitmask` is **the one bit lane in the artifact** (ADR 0047). A set bit
asserts *serveability* — this unit, exactly as it stands, can serve that
semantics with no further composition — and neither that real data exists for it
nor that it becomes serveable after composing with something else. A set of
semantic sources, keysets or physical sources is an `EntityId` list and never a
mask.

A relationship has no carrier of its own (ADR 0050): the unit it caused is what
persists, carrying the pooled join condition its relationships produced with
every column leaf `<constituent alias>.<semantics name>` (ADR 0057, as ADR 0075
amends it).

### `PhysicalSource` — one resolved relation

```rust
pub struct PhysicalSource {
    pub id: EntityId,                         // a path's `mint(base_path, format)`, a table's `mint("table", locator)`
    pub name: String,                         // a label: a base path's last segment, or a table's name (ADR 0070)
    pub alias: SourceAlias,                   // `ps` and the `EntityId`-order index (ADR 0085, rule 11)
    pub storage: StorageProperties,           // StorageTarget::{Path, Table} + format + partitions
    pub schema_provenance: SchemaProvenance,  // CatalogDerived { schema, native } | ModelFed { schema }
    pub schema_id: SchemaId,                  // SHA-256 over the canonical schema encoding
}
```

Flat, with **explicit** schema provenance (ADR 0046): which of the two routes
reached a source is recorded on the artifact rather than inferred, so do not infer
the arm by counting `None`s — ask the arm. Per `CONTEXT.md` **Physical source**,
the column *set* is complete and authoritative on both routes, while a column's
canonical `data_type` and `nullable` are optional, and absent means "this compile
observed neither" — never "untyped", and never a licence to default (ADR 0061).
The provider's native spelling is kept beside the canonical list as provenance,
never read to decide anything — the only answer to "what did the provider
actually say" when a canonicalisation looks wrong. A referenced column whose
native spelling maps to no canonical type fails the compile outright rather than
persisting absent (ADR 0086, rule 7), so a catalog-derived column always holds a
type: `validate()` refuses one that does not (`UntypedCatalogColumn`).
Where compile observed no type for a column, `read_type` gives the type that a
physical read reads it as, from the semantics mapped directly to it on every
semantic source over the physical source (ADR 0082). Compile writes it into a
model-fed column's `data_type`, and it enters the source's `SchemaId`; a column
that no semantics maps directly keeps `data_type: None`.

Compile types each column that a Dataset's tree reads by its observed type, where
the catalog gave one (ADR 0089, rule 15). A column that compile did not observe, and
that a semantics maps directly, is cast to that semantics' declared type in its tree,
except under a `count` or an `approx_count_distinct`; every other read of it takes
its read type, and a column that no semantics maps is `Unknown` (rule 16). A tree
whose type differs from its semantics' declared type is cast to it, where
`casts_to()` allows, with no warning (rules 17 to 19). A composition types the
column it reads by what its operand gives (ADR 0085, rule 13): a Measure's by the type
of the operand's aggregate argument, and every other semantics' by its declared type.
So a composed Measure's cast is above its final aggregate, as a Dataset's is.

`SourceSchema` / `SourceColumn` are manifest-local for exactly that reason: a
source schema is a claim about the world and may be partial, where IR's `Schema`
is the total column list a plan node emits against. A caller imports IR's `Schema`
and `semstrait_catalog::Schema` from the crate that owns each.

### `ManifestExpressions` — the physical-expression store

```rust
pub struct ManifestExpressions { pub physical: BTreeMap<PhysicalExprId, ManifestExpression> }

pub struct ManifestExpression {
    pub id: PhysicalExprId,                    // a v5 UUID of (owner, root marker)
    pub expr: StoredExpr,                      // the whole tree, with a Link where it references another entry
    pub layer: ExprLayer,                      // the tree's layer, each link counted as its entry's
    pub physical_sources: BTreeSet<EntityId>,  // the sources that consume it
}

pub struct StoredExpr(Expr<Link>);             // semstrait-sql's Expr over the link leaf
pub struct Link(pub PhysicalExprId);           // a reference to another entry
```

One entry per expression of the semantic model (ADR 0088, rules 4 and 5): a
signature's expression or filter, or a Joinset source's condition or relationship
filter. A `StoredExpr` is the whole tree, `semstrait-sql`'s tree over the link leaf, a
bare link included: compile writes a link where a tree references another signature on
its semantic source, and inlines every other part (rule 5), but a Joinset's condition,
relationship filter and filter, and a composition's Dimension and Measure, hold none.
`ManifestExpressions::reassemble(root, signatures, fields)` reads an entry through its
links and gives the plan's `PhysicalExpr`, with no `Parameter` in it (rule 15): each
link's place takes the tree of the entry that it links, read the same way, but for each
leaf that the reader keeps in `fields`, which holds `KeptField`s, a `KeptLeaf` found by
the link or by the column leaf of a column under `auto`, never by equal trees, which
reads as the `Expr::Column` of the output field of the node below that the reader names
for it, outside each aggregate call (rule 4): a link to a Metric whose `agg:` aggregates
a formula, always, and, above an aggregate step, a link to a Dimension or to a row-level
Metric with no `agg:`, or a column under `auto`, that the step groups by (ADR 0074,
*A Metric*).
`signatures` are the ones that the root's semantic source lists, and a link's signature
is the one among them whose `expression_id` is the link, found by id alone, because the
manifest stores no reverse index (ADR 0055). Where that signature has a filter, a
Measure's or a Metric's, its tree is joined with `And`, on the right, onto each
aggregate call of the linked tree, after any filter already there, so a Metric `net =
revenue - refunds` that stores `Link(revenue) - Link(refunds)` reads as `sum(…) FILTER
(WHERE status = 'paid') - sum(…)`. That join is the public
`join_filter_onto_aggregates(tree, filter)`, which the planner also uses for a
signature's own filter on its own tree (ADR 0071, *Aggregation-level conditions*).
`ManifestExpressions::linked(root, signatures)` gives the signature of each entry that
the walk reaches through the links outside each aggregate call, the root's left out
(ADR 0083, rules 3 and 4). `ManifestExpressions::aggregate_signatures(root, signatures,
kept)` gives, in pre-order, the signature whose entry holds each aggregate call of the
tree that `reassemble` reads, found by the call's place through those same links, never
by equal trees, so an entry linked twice gives its signatures twice (ADR 0074; ADR 0083,
rule 3). A link to an entry of `kept`, which the reader keeps as a field (`KeptLeaf`),
holds no call and is not followed (ADR 0088, rule 15).
For both, a dangling link, a link to
no listed signature's expression, and a cycle read as `None`; the walk over the links
does not recurse, and the tree it returns is as deep as its links. On a Grainset, rule 15's reading of a
linked Metric's filter on the leaves below the link is not built. Compile builds a
Grainset signature only for a Metric that the Grainset computes (ADR 0084, rule 8), and
`validate()` refuses any other, one for a row-level Metric included. An entry's `layer` is
`ManifestExpressions::layer_of` its tree, in which a link counts as the layer of the
entry that it links (rule 6); emission derives it and `validate()` checks it. The
manifest declares no variant of the tree, and a stored expression holds no `Parameter`
(rule 4). Its JSON is `semstrait-sql`'s form of the tree (ADR 0088, rules 1 to 3): each
node but a link is `{"type": <tag>, "content": <body>}`, a call's tag is its function
type and its body writes no field at its default, and a link is its entry's id. Each id,
as a key, an entry's `id`, a link or a signature's `expression_id`, is written as UUID
text, the lower-case form with hyphens of an `EntityId`, and reading takes no other form
(ADR 0088, rule 13): `"253641c7-819a-5b60-a3dc-c276bfceb36d"`. Ids are minted from the
owner and which of its trees the entry holds — the `ExprRoot` marker,
`SignatureExpression` | `SignatureFilter` | `JoinsetOn` | `JoinsetFilter`, which is
what lets one owner hold several entries — with no path; nothing is content-addressed
and nothing is shared, and a repeated id is a producer defect the store refuses.
Physical-only, and the
artifact's single exception to the `EntityId` lane. **Every** realisation emits — a
bare column and a broadcast literal no less than an aggregate (ADR 0037) — and
semantic expressions do not survive compile. Every column is a name path (ADR 0085,
rule 9), `semstrait-sql`'s `NamedReference`, and nothing is cut at a dot (rule 12):
every column holds two parts, `[<source alias>, <name>]`, and compile refuses one
that would hold an empty part, which names nothing, such as a semantic mapping's
column `""` in a model built in code, as `CompileError::EmptyColumnPart`. The alias
names the physical source for a Dataset's tree (`["ps0", "amount"]`), the
constituent whose semantics a join-condition leaf names (ADR 0075), and the
immediate operand for a composed projection's signature — ADR 0065 scopes the last
two.

### `Keyset`

```rust
pub struct Keyset { pub id: EntityId, pub name: KeysetName }
```

`{id, name}` and nothing else. A keyset is a named group of a data kind's
dimensions that **advises** how its rows are identified (`CONTEXT.md`
**Keyset**) — uniqueness over the group is assumed, never enforced or verified
against data. The group itself lives on the semantics side, attributed per
source (ADR 0048): the per-source attribution rides
`SemanticSignature::keysets` and nothing pools it on the definition. So *"the members of keyset K"* is
**derived** — the signatures whose attribution names `K` — and the entity
carries no edge in either direction. `Keyset::members` and `Keyset::data_kinds`
do not exist.

Identity is `(semantic_source_id, keyset name)`, with the semantic-source id in
the seed only, so one authored name is one entity **per** semantic source.
ADR 0040's `members`, its `data_kinds` and its `(name, sorted members)` dedup are
all superseded on carrier grounds; everything else in ADR 0040 stands — keyness
advises, `KeysetName` is its own newtype, and there is no data-kind bit lane.

---

## `compile()`

```rust
pub async fn compile(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
    provenance: ManifestProvenance,
) -> Result<
    (SemanticManifest, Diagnostics<CompileError>),
    (Diagnostic<CompileError>, Diagnostics<CompileError>),
>;
```

`Request`-free and deterministic: identical inputs yield equal load-bearing
state, and equal manifests serialize to identical bytes. `Ok` pairs the artifact
with the warnings accumulated across every stage; `Err` pairs the **first** hard
error with the warnings accumulated before it (fail-fast).

`catalog: None` means model-as-source-of-truth. `fs` is **live**: every path source
is listed against it, and a folder wildcard gives one physical source for each
folder it matches (ADR 0070), so the number of entries a model compiles to is a
function of filesystem contents and not of the document alone.

`provenance` is the only thing about any artifact's history `compile()` is told.
There is no prior-manifest argument, no `compile_if_changed` sibling and no
comparison against anything persisted: whether to recompile is the caller's
decision (ADR 0041), taken outside this function.

### The stages, in the order the driver runs them

Data flow forces the order at every step; the driver enforces **no rule of its
own** (ADR 0044) — every rule belongs to the stage that raises it.

| # | Step | Owner |
|---|---|---|
| 0 | Replace a one-child composition by its child, under the composition's name, id and interface (its Dimensions, Measures, Metrics, named filters and keys), unless its parent cannot hold the child's data kind; the child keeps its own temporal shape where it has one, and the composition's fills a child that has none | `sugar::desugar` |
| 1 | Plan the semantic sources over every cross-variant nesting (ADR 0026): which units exist, what each is called, which are public, which originator produced each — and enumerate every legal sub-composition of an authored Joinset (ADR 0054, ceiling `MAX_IMPLICIT_ENUMERATION_COUNT`) | `assemble::plan_semantic_sources` |
| 2 | Resolve every storage entry into the physical sources it reads, each built once from every semantic source over it, and the selector each reads through; discover the partitions under a `Prefix` and refuse any other layout (ADR 0070) — **the sole I/O step** | `catalog::fetch_physical_sources`, `catalog::partitions::discovered` |
| 3 | The Dataset reduction: a leaf declaring several physical sources becomes one leaf per source under a synthesised Unionset carrying the authored name (ADR 0049) | `assemble::reduce_multi_source_leaves` |
| 4 | Build the semantics registry from the model's root pools, then refuse two definition names equal without regard to case (`DuplicateDefinitionName`) and a definition name that starts with a prefix that `Name::RESERVED_PREFIXES` reserves (`ReservedDefinitionName`) (ADR 0075, *Names are unique, and the planner's own names are reserved*), then a semi-additive Measure's axis naming no semantics or a non-Dimension role, or a `by` naming no semantics (`SemiAdditiveByUnknown`) (ADR 0083's callout on ADR 0076's axis checks) | `definitions::build_semantic_definitions`, `definitions::check_no_case_insensitive_duplicate`, `definitions::check_no_reserved_prefix`, `definitions::resolve_semi_additive_axes` |
| 5 | Read the pool halves of every source's body and filter views; the reference graph and its acyclicity pass are **per source** and run inside the seal | `definitions::{pool_bodies, pool_filters}`, `definitions::build_reference_graph`, `definitions::graph::topological_order` |
| 6 | Derive every unit's alias from its source role and its index (ADR 0078), and every physical source's from its index (ADR 0085, rule 11) — before anything is sealed, because a Dataset's tree names its physical source's and a join condition's column leaf its constituent's | `source_alias::SourceAliases` |
| 7 | Build the keysets stage over the plan; each unit's keysets are recorded in step 9, once it is sealed | `keysets::Stage::new` |
| 8 | Compute one read type for each physical-source column, from every semantics mapped directly to it on every leaf over the source — before the seal, because every read of the column and the model-fed schema take that one type (ADR 0082; ADR 0089, rule 16) | `assemble::column_types` |
| 9 | Seal each unit, **operands before composers** (`assemble::seal_order`, not plan order): check a leaf's `partition_def` entries against its partitions (ADR 0081), realise over the mapped leaf (explicit `semantic_mapping` entries lowered), lower each authored body per source, type each tree with `semstrait-sql`'s typing in the canonical dialect, each column by its observed type, else its read type (ADR 0089, rules 11 to 16), cast each tree to its semantics' declared type where the two differ (rules 17 to 19), write a link where a tree references another signature on its source, but in a Joinset's trees (ADR 0088, rule 5), fold each tree with `semstrait-sql`'s fold, a link to a literal entry read as that literal, but a tree that reads an aggregate as written and that the fold would leave reading none, which is stored as written, and emit it into the expression store as one entry, each expression entry of a source in the order in which it lowered, before any filter entry, the one point where compile folds (ADR 0089, rule 23), and emit one `SemanticSignature` per exposed semantics — a Dataset over its own realisations, a Joinset or Unionset over an operand's projected column, or a Joinset's `Coalesce` of both where both serve the name (ADR 0074), and a Metric's and a named filter's body lowered on the composition, as is a Joinset Measure's that no operand serves, a Metric's linking the composition's own signatures (a composed Measure's or Metric's `filter:` is lowered here too, on a Joinset once for each operand, and a row-level Metric with no `agg:` holds it in its own `CASE` — ADR 0065; ADR 0074, *A Metric*; ADR 0077); an authored Joinset's relationship becomes its pooled join condition here, and a unit whose every relationship has a cross basis a `Cross` join with none (ADR 0066); once every sealed unit is stored, every column of their trees is two parts under the alias its place names: a Dataset's tree its physical source's, a join condition a constituent's (a Dataset, a Unionset, a Grainset, or a Joinset that is a Grainset's child), a Joinset's projection an operand's, a Unionset's tree its own, a Grainset's own expression and filter roots its own (ADR 0085, rule 12); once a unit is sealed, it guarantees each root keyset whose every member it serves, checked against its boundary, and `on: auto` reads its operands' (ADR 0048; ADR 0077, *The keysets of a nested composition*) | `assemble::{seal_order, seal_unit, seal_leaf}`, `keysets::Stage::record`, `catalog::partitions::check_claims`, `expressions::projection::seal_composed_projection`, `expressions::{realise, emit_source_attached}`, `expressions::bodies`, `expressions::grainset_metrics::seal_grainset`, `expressions::lower::reconcile_boundary`, `relationships::resolve_joinset`, `qualifiers::misplaced_columns` |
| 10 | Derive every Complex data kind's temporal shape, bottom-up over the operand edges, and persist it flat in the slot a leaf uses (ADR 0056); a composition that is a constituent carries its declared or derived granularity (ADR 0074) | `temporal::derive_composed_shapes` |
| 11 | Default a semi-additive Measure's `by` from the Datasets its source reads and raise its five refusals, and `SemiAdditiveAxisNotServed` for an axis a signing source does not also serve (ADR 0076; ADR 0083's callout: Measure-only) | `semi_additive::check_semi_additive_sources` |
| 12 | Materialise the keyset attribution onto `assembly.signatures`, and each Grainset's onto its own `keysets` field | `keysets::{attribute_signatures, attribute_grainsets}` |
| 13 | Finish the source roster: a source's canonical column list is scoped to the columns something actually reads, and a model-fed column holds its read type from step 8 | `sources::complete_sources` |
| 14 | Every condition leaf names a semantics its constituent serves (ADR 0057, as ADR 0075 amends it) | `relationships::unserved_condition_leaves` |
| 15 | Refuse an entry that no root names (`OrphanExpression`), in one pass over the roots — each signature's `expression_id` and `filter_expr_id` plus every Joinset unit's condition and relationship filter handle | `seal::sweep_orphan_expressions` |
| 16 | Freeze the bitmap, then complete `semantic_sources` — in that order and never the reverse, because a served mask addresses those positions | `SemanticBitmap::freeze`, `assemble::complete_semantic_sources` |

Two steps sit where a reader may not expect them, and each is forced by its
inputs: enumeration runs inside step 1 (`Relationship::integrity` is not an
`Option`, so the default is already applied when the plan reads it), and keyset
resolution runs before relationship resolution (`on: auto` derives its basis from
*guaranteed* keysets, so they must exist before a condition can be built). A third
is the seal loop's own order: it walks `assemble::seal_order`, which puts operands
before composers, because a composed unit's projection asks which operand already
serves a name and the enumeration may list a cumulative unit before
its own sub-composition. `compile/driver.rs` states all three at length, and none
is a ruling: ADR 0044 puts every rule in the stage that states it, so moving a step
here moves no rule.

### Transient lowering vocabulary

`compile::types` holds the working set the stages share, re-exported at the crate
root because a caller reading a diagnostic may need to name it: `Provenance`,
`RecursionState`, `ResolvedExprKey`, `ResolvedExprEntry`, `ResolvedExprTable`,
`SchemaIndex`, `SemanticsIndex`, `ScopeChain`, `ScopeFrame`, `ScopeKind`.
**None of it is persisted.** `LoweringCtx` is deliberately not re-exported — it
is `pub(crate)`, because the only engine that consumes it lives here.

### Scope today (v1)

The driver wires leaf `Dataset`s — any number, top-level or nested, over any
number of physical sources each — under `semantic_mapping: auto` or an explicit
mapping block, plus the `Unionset` that
stacks them (authored or synthesised by the reduction), plus an authored
`Joinset` of any width, plus an authored `Grainset` whose
grains become entries of their own and which emits one implicit cross-grain
Joinset per ordered grain pair, **finer to coarser** — so three grains yield three
units rather than six (ADR 0054). Any composition may be a constituent of another
of a different variant, at every depth, each nested one a private entry with its own
alias (ADR 0026; ADR 0078).

**A covered Dimension, Measure or Metric may carry an authored `expr:` body**,
lowered once per semantic source into one entry (`compile/expressions/bodies.rs`): a
body's leaves resolve first, in the reference graph's order, with each semantic-leaf
reference a link to *that source's* entry for the referenced signature (ADR 0088, rule
5), typed by that entry, and a reference site's own deferred `expr:` replaces the pooled
body for that source. A Measure's or Metric's `filter:` lowers the same way into a
**second** root the signature carries as `filter_expr_id` (`ExprRoot::SignatureFilter`),
and a site's `filter:` replaces the pooled one for that source. Compile stores each
expression entry of a source in the order in which it lowered it, before any filter
entry, so each entry follows each entry it links. **A Metric that references a filtered
Measure reads with that Measure's own predicate on the FILTER of the linked aggregate
call**: reading joins it where it follows the link (rule 15), never a predicate on the
referring Metric's own signature and never a change to the Measure's own. The rule is
**transitive through a Metric-referencing-Metric chain**: a linked Metric's own
predicate is joined onto *every* aggregate call its tree reaches, `And`-ed onto a FILTER
a Measure's predicate already filled, one link down (ADR 0065 §3). On a **composed**
source the same rule runs over the composition's own projected columns, and the composed
Measure's or Metric's signature carries that predicate alone (ADR 0071; ADR 0074, *A
Metric*): on a Unionset it links the Unionset's own signatures, and on a Joinset it holds
no link. A composed Metric's body links the composition's own Measures, so it reads
their composed filters through those links, transitive case included. The predicate is
row-level whatever role owns it, so an authored aggregate and a reference to a
Measure or a Metric are both refused, by role rather than by the substituted
tree's layer — a *scalar*-bodied Metric reference is refused exactly like an
aggregating one. An explicit `semantic_mapping:` block **is** lowered: an entry's value becomes
the leaf tree of the realisation it maps. A window call authored in a body or in
a relationship filter is refused by rule: its partition and order are
request-time facts the manifest doesn't hold (ADR 0091, rule 1; ADR 0066, ruling
7). Three shapes are refused as unruled and left open rather than given a
decided answer: an accessor-bearing leaf, a mapping entry that names a semantics
which itself authors a body, and a `SemanticMappingValue::Expr` under a mapping
block.

---

## Identity

One lane, plus one exception for the entries of the expression store.

Every entity carries an `id: EntityId` equal to its map key, and every
cross-reference is an `EntityId` (ADR 0010). The exception is the expression store,
keyed by `PhysicalExprId` — an entry is one of its owner's trees, not a named entity.
Its id is a v5 UUID too, written as UUID text, as an `EntityId` is (ADR 0088, rule 13).

`EntityId` values are canonical UUID text, and **the lane holds UUIDv5
throughout**: ADR 0051 retired ADR 0010's tolerance of a time-based v7 in the
authoring lane, so an entity that omits `id:` is minted
`UUIDv5(namespace_for_its_class, scope_qualified_name)` rather than
`Uuid::now_v7()`. The **name is the natural key** — what a definition states and
what every reference resolves — and the id is the technical key layered on top.
The version nibble is not validated beyond canonical-UUID-text shape, so a v7 id
carried in from an older document is still accepted; it is a legacy shape, not a
second lane.

Both lanes are derived deterministically, and only their *generation* lives here:
an `EntityId` is its UUID text, and a `PhysicalExprId` holds its `Uuid` and is written
as that text.

| Mint | Seed |
|---|---|
| `mint_model_id` | the model fingerprint |
| `mint_semantic_id` | role + bare name |
| `mint_semantic_source_id` | the scope-qualified name (ADR 0051) |
| `mint_table_source_id` | `("table", locator)` (ADR 0070) |
| `mint_path_source_id` | `(base_path, format)` (ADR 0070) |
| `mint_reduction_leaf_id` | `(Dataset id, physical-source id)` (ADR 0070) |
| `mint_signature_id` | the `(semantic_source, definition)` pair, in that order |
| `mint_keyset_id` | owning semantic source + keyset name |
| `mint_expr_id` | one entry's `(owner, root marker)` (`EXPR_NAMESPACE`) |

The minting primitive and its namespace roster (`mint_entity_id`,
`mint_named_entity_id`, `entity_namespace`, `EntityNamespace`,
`ENTITY_ROOT_NAMESPACE`) are `semstrait-common`'s (ADR 0051), and a consumer of the
artifact imports them from there (ADR 0090, rule 6).

---

## Validity

**Validity is a compile-stage property** (ADR 0042). A manifest in an `Ok` arm is
valid by construction: each stage refuses what it cannot produce validly, so
there is no end-of-compile re-walk, no typestate and no `ValidatedManifest`
wrapper. There is likewise no gate roster and no numbered gate identity
(ADR 0044) — the enumeration of what compile enforces is the `CompileError`
variant set plus the stage that raises each variant, and what a stage
guarantees is stated by its `CompileError` variants and its tests.

`SemanticManifest::validate()` is that **same conjunction** exposed for the one
subject no stage can reach: an artifact decoded from bytes, or one whose `pub`
fields were mutated after compile. It therefore raises `CompileError` and nothing
else. ADR 0042 states the consequence: the validator's check set *is* the
conjunction of every stage's validity definition, so it "cannot be legitimately
stricter than the producer" — where it holds a rule a stage could have asserted
and did not, that is a defect in the stage rather than an accepted asymmetry.

```rust
impl SemanticManifest {
    pub fn validate(&self) -> Result<Diagnostics<CompileError>, Diagnostics<CompileError>>;
    pub fn validate_with(&self, reporting: FaultReporting) -> Result<…, …>;
}
```

ADR 0042 fixes the reporting contract as *accumulate by default, fail fast by
flag, suppress nothing*: these faults are corruption of a compiled artifact rather
than author error, so there is no edit-and-retry loop for cascade suppression to
protect, and the traversal is order-**independent** — the fixed order survives only
so a report is stably ordered. `FaultReporting::FirstFault` is a
**short-circuit** over the same traversal, not a second return class — it returns
the same pair carrying one element, so no caller's `match` changes shape.

Most checks are a *call* to the producing stage's own total function rather than a
second implementation. Nine rules have their only home here, because no producer
can reach their subject:

- global id uniqueness across every collection plus `model_id`;
- `bit_position` density — the occupied set must **equal** `0..n`, not merely
  match `n` (ADR 0036);
- the expression store's entry-derived facts, `id == key` and the `layer` of the
  entry's tree, each link counted as the layer of the entry it links (ADR 0088, rule
  6);
- **one root per entry**: an entry that two roots name is refused (ADR 0088, rule
  14). Ids minted from the owner and marker and an `insert` that refuses a repeated
  one construct it at emission. The other clause, **no orphan**, has a producer that
  can break it and is therefore a call to the sealing stage's own sweep;
- the **options** of each call of a stored tree: an option name that its declaration
  does not have, or a value that the option does not take, is `UndeclaredOption`,
  from `semstrait-sql`'s `Call::check_options` (ADR 0088, rules 3 and 9). Compile
  writes no option, and reading checks none;
- the **links** of each entry (ADR 0088, rule 14), with each entry's owner found from
  the signatures and the Joinset sources that name it: a link that names no entry is
  `DanglingReference`, and each other fault is a `StructuralViolation` with its own
  rule — a link in a Joinset source's condition or relationship filter, in the
  expression entry of a composition's Dimension or Measure signature, in the filter
  entry of a Joinset's or a Grainset's signature, or in the expression entry of a
  Joinset's named Filter signature, which hold none (rule 5); a link in a signature's
  entry that names an entry other than the expression entry of a signature that the
  same semantic source lists, so a Joinset Metric keeps its links to the Joinset's own
  signatures; and a cycle, an entry that its own links reach again, found in one walk
  over the store that keeps a stack and does not recurse, and reported once for each
  link that closes a cycle in that walk. Compile builds none of these;
- a catalog-derived column always holds a type: one with no `data_type` is
  `UntypedCatalogColumn` (ADR 0042; ADR 0086, rule 7);
- signature **ownership**: exactly one `semantic_sources` entry lists any given
  signature. Zero owners is an orphan and two-or-more is a signature two units both
  claim to implement, and neither is reachable from `compile()`, which mints a
  signature id from the source that seals it and lists it there. The mask *agreement*
  beside it is a restatement rather than a second rule: `manifest::serving::Serving::served`
  derives a reading source's mask from its own signatures, so the equality holds by
  construction and `validate()` re-asserts it over an artifact it did not produce;
- a **Dataset signature tree naming its owner's relation**: the expression root, and
  the filter root where there is one, must carry the Dataset arm's `source_id` among
  their consuming `physical_sources`. Lowering a leaf's realisation against the one
  source the leaf reads is what emits that association, so only a decoded or
  field-mutated artifact can attach one Dataset's valid tree to another's signature
  with ownership, definition and mask left intact. Dataset owners only: a composed
  projection and a Joinset condition introduce no column and carry the empty set
  (ADR 0057);

plus ADR 0058's reduction carry-over invariant — a reduction's Unionset carries
every keyset name its leaves carry — which is compile's own fault to break and so
would name a defect no author can repair.

Every alias is **derived again**, once, by compile's own `SourceAliases`, and three
restatements read it (ADR 0078; ADR 0085, rules 11 and 12): a semantic source's alias
that its source role and index do not give is `SourceAliasMismatch`, a physical
source's that its index does not give is `PhysicalSourceAliasMismatch`, and a stored
column that is not two parts under the alias its place names is
`ColumnQualifierMismatch`, from the column qualifier rule's own `misplaced_columns`.

A Joinset's condition or relationship-filter leaf that names no semantics its
constituent serves is `UnservedConditionLeaf`, from compile's own
`unserved_condition_leaves` (ADR 0057, as ADR 0075 amends it).

A Joinset's `on_expr_id` is absent exactly when its `join_type` is `Cross` (ADR 0066).
`relationships::resolve_joinset` derives the join type from the condition, so the
agreement holds by construction; `validate()` re-asserts it as a `StructuralViolation`.

---

## Persistence

`SemanticManifest` implements `semstrait_common::io::Persistable`: its own
identity (`model_id`), extension (`json`) and codec. `serde_json` over the
all-`BTreeMap` artifact is deterministic, so identical manifests serialize to
identical bytes.

`Repository`, `FileSystemRepository` and `RepositoryErrorKind` are the **generic**
seam in `semstrait_common::io`, and a caller imports them from there (ADR 0090,
rule 6); this crate defines no copies of its own. `save` is a map put — last writer wins — and
the seam's own integrity check is shallow id-vs-key agreement.

**Reading a manifest** is its `Deserialize`, which `from_bytes` calls, and it
refuses what it needs to build each node (ADR 0088, rule 9): a `dialect` other than
`canonical` (`CompileError::ManifestDialect`, which names the value as the document
writes it, and which `validate()` gives for a field-mutated one); a call whose name
the registry does not find in that dialect, or whose tag is not its function's type
(`semstrait-sql`'s `JsonFormError`); a call whose arguments do not fit its
signature (their number, a literal at an enum or a constant parameter, and an enum
value that the parameter lists), a clause that its declaration marks `Refused` or
`NoEffect`, and a call with no `Required` clause, because it builds each call with
`semstrait-sql`'s constructors and their check (`CallError`, such as `NotALiteral`
and `EnumValue`); and a leaf that is not a link, such as a `Parameter` or a value
that is not an id's UUID text (`CompileError::StructuralViolation` on the
expression pool). An enum or a constant argument is an inline literal of the
entry's tree, so a link at such a parameter is refused (`NotALiteral`), and
reading checks a call as parse does (ADR 0089, rule 5). Each refusal is the typed
error's message in `serde_json`'s error, which places it in the text.

**There is no load path here** (ADR 0042). No module, type or function in this
crate is named for loading a manifest. `validate.rs` states what that does and does
not cover: the retired boundary is the *validating* hook the persistence seam once
carried, while `Repository::load` is a different verb that lives in
`semstrait-common` and stays — reading bytes from a store was never the boundary in
question. Deep validation is the consumer's, which is what `validate()` is for.
`LoadError` is deleted; its two variants folded into
`CompileError::{DanglingReference, StructuralViolation}`.

---

## Dependencies

| Crate | Role |
|---|---|
| `semstrait-common` | `EntityId` and the id mint, `Tree`, `Diagnostic` / `Diagnostics`, `Granularity`, the `io` transport + `Persistable` / `Repository` seam (features `serde`, `io`) |
| `semstrait-sql` | `DataType`, its type families and its name function, which `native_to_canonical` reads a catalog's type with; the tree, its calls and their constructors, the function registry, typing with the canonical type system, the fold and the tree's JSON form |
| `semstrait-ir` | `PhysicalExpr`, the plan's tree that a stored tree reassembles to; `Cardinality`, `JoinType`, `Integrity`, `CrossFilter`, `JoinsetJoinType`, `SemanticRole`, `AdditivityType`, `SemiAdditivity`, `SemiAdditivityStrategy`, `TemporalShape`, `TemporalType`, `ScdType`; `SchemaColumn`, the canonical column of the transient `SchemaIndex`; `Name::folded`, the plan's case fold, by which step 4 and the served-name check on a condition leaf compare names, and `Name::reserved_prefix`, the reserved prefix that step 4 refuses in a definition name |
| `semstrait-model` | `SemanticModel`, and the authoring vocabulary compile maps to the manifest's own matching type: `DimensionType` and `AiContext` are the manifest's own declared types (ADR 0094, rule 4); `Integrity`, `CrossFilter`, `AdditivityType`, `SemiAdditivity`, `SemiAdditivityStrategy`, `TemporalShape`, `TemporalType` and `ScdType` compile maps to `semstrait-ir`'s matching type; `Agg` lowers to the manifest's `aggregation` function name and `distinct` flag (rule 5); `UnionMode` maps to `semstrait-ir`'s `MultisetOp` |
| `semstrait-catalog` | `CatalogProvider` for the compile-time metadata fetch |
| `serde`, `serde_json` | the deterministic wire form |
| `thiserror` | `CompileError` |
| `async-trait`, `bytes` | the async persistence boundary |
| `uuid` | deterministic content-derived `EntityId` minting (UUIDv5) |
| `sha2` | SHA-256 over the canonical schema encoding (`schema_id`) |
| `percent-encoding` | decodes a `key=value` partition folder (ADR 0070) |

Dev-only: `tokio`, which drives the async `Repository` round-trips against
`InMemoryFileSystem`, and `tempfile`, for the scratch directories of the
golden generator's test.

The manifest owns its own storage types — among them the expression id
(`PhysicalExprId`) and the expression layer (`ExprLayer`), which only it writes and
reads (ADR 0090, rules 4 and 5). The authoring and IR families above are imported
from the crate that owns each, and the manifest neither copies nor re-exports them
(ADR 0090, rule 6): a manifest-local duplicate of any of them would be a second
authority.

---

## Integration tests

`tests/` holds one file per ratified behaviour, and each is the readable
statement of what its rule is:

| File | Subject |
|---|---|
| `manifest_lifecycle.rs` | the identity block's lifecycle at the `compile()` seam (ADR 0041, ADR 0055) |
| `manifest_snapshot.rs` | a committed byte-target snapshot of a compiled artifact |
| `deterministic_identity.rs` | one document yields one manifest (ADR 0051) |
| `semantic_definitions.rs` | the registry, the signatures, and the frozen bit positions |
| `semantic_sources.rs` | the flat collection at the public compile seam (ADR 0045) |
| `physical_sources.rs` | the `sources` roster and its schema provenance (ADR 0046) |
| `read_type.rs` | the read type of an unobserved column, from the semantics mapped to it (ADR 0082; ADR 0089, rule 16) |
| `model_fed_schema.rs` | the read type that compile writes into a model-fed schema: one for each physical-source column, from every semantics mapped directly to it, and none for a column that only an expression or a Filter reads (ADR 0082) |
| `column_types.rs` | a column's observed type, and the cast of an unobserved, directly mapped column to its semantics' declared type, with none under a count, and of an observed one under a `sum`, an `avg`, a `min` or a `max` where its type differs (ADR 0089, rules 3, 15 and 16) |
| `declared_type_casts.rs` | the cast of a tree to its semantics' declared type, a reference to a literal-mapped semantics among them, a tagged literal's narrowing refused, and a tree with no type (ADR 0089, rules 17 to 19) |
| `expression_typing.rs` | typing at compile: literals by their context, the refusals of typing, and a Measure's aggregate (ADR 0089, rules 11 to 13; ADR 0092, rule 4) |
| `expression_pool.rs` | the expression store, one entry per expression with its whole tree (ADR 0037; ADR 0088, rules 4 and 5) |
| `column_qualifiers.rs` | every stored column is two parts, the source alias its place names and then the name, and `validate()` refuses one that is not (ADR 0085, rule 12); compile refuses a mapped column with an empty name and a composed column with an empty part, as `CompileError::EmptyColumnPart` |
| `manifest_reading.rs` | the manifest's `dialect`, reading a manifest's JSON and its refusals, and `validate()` over the `dialect` and each call's options (ADR 0088, rules 3, 8 and 9) |
| `stored_aggregation.rs` | a Measure's `aggregation`, stored as its aggregate's canonical name, with `count_distinct` stored as `count` and a separate `distinct: true` flag, and reading's refusal of any other name (ADR 0089, rule 20) |
| `semi_additivity.rs` | a semi-additive Measure's `axes` and `by`, stored as the definitions' ids compile resolves from the authored names, `by`'s default from the time axis of the Datasets that its source reads, through a Grainset operand or branch too, and the refusals of `by` and of an axis, each with its message golden (ADR 0076, *`axes` and `by`*, *Where `by` and the axes come from*; ADR 0083, *Consequences*) |
| `expression_fold.rs` | compile folds each tree that it stores, after typing: a metadata-determined signature is stored as a literal, a CAST of a literal as the canonical CAST's value, a Unionset branch's included, and a function call and `double` arithmetic stay as typing gives them (ADR 0089, rules 21 and 23) |
| `temporal_functions.rs` | a semantic model's `date_trunc` and `extract`: the defaults of `date_trunc`'s `timezone` and `week_start`, its `'UTC'`-only timezone (`CompileError::DateTruncTimezoneNotUtc`), and the refusals that need types, each with its message golden (ADR 0089, rules 3 and 4; ADR 0087, rule 14) |
| `body_lowering.rs` | authored `expr:` bodies, lowered once per semantic source into a tree in an entry of its own, with a link to each signature of the source that it references (ADR 0088, rules 5 and 7) |
| `metric_forms.rs` | a row-level Metric's `agg:` lowered as a Measure's, a Metric's `agg:` over a formula stored as the formula alone, a row-level Metric with no `agg:` holding its `filter:` in a `CASE`, the fold that changes no Metric's kind, and the links that `reassemble` keeps (ADR 0074, *A Metric*; ADR 0088, rules 5 and 15; ADR 0089, rules 17, 20 and 23) |
| `semantic_signatures.rs` | the `semantic_signatures` collection: one per set bit, its tree, and its id |
| `composed_signatures.rs` | a composed reading surface owns one signature per set bit of its mask, projecting an operand's column or a Joinset's `Coalesce` (ADR 0074); a bodied Metric and a named filter are their bodies lowered on the composition, and a composed Measure's or Metric's `filter:` is its own view only, on a Joinset one copy for each operand, held in a row-level Metric's own `CASE` where it has no `agg:` (ADR 0065 D2; ADR 0071; ADR 0074, *A Metric*; ADR 0077) |
| `signature_filters.rs` | a Measure's or Metric's authored `filter:`, lowered into the signature's filter entry, apart from its expression entry, with a link to each Dimension it reads; a Metric linking a filtered Measure reads with that Measure's predicate on the linked aggregate instead, transitively through a Metric-referencing-Metric chain (ADR 0065 §3; ADR 0088, rules 5 and 15); an `expr:` or a `filter:` that reads a named Filter is refused with `NamedFilterReference`, except a named Filter's own body, which may read one (ADR 0071; ADR 0083 rule 8) |
| `composed_serving.rs` | a constituent serves only the names it maps: explicit mapping, `auto`, a body's or a `filter:`'s typed NULL on an explicit Unionset's branch, `UnservedComposedSemantics`, `JoinKeyNotMapped`, the null-fill setting, a Measure or Metric that no Joinset operand maps whole lowered on the Joinset, and a Grainset child's filter-read rule (ADR 0077; ADR 0084 rule 8) |
| `coverage_masks.rs` | masks as serveability statements (ADR 0047), and the three-way axis (ADR 0059) |
| `keyset_roster.rs` | the roster and its attribution (ADR 0048) |
| `path_sources.rs` | a path source is one base path read in one format: the normal form, the identity, each semantic source's selector, the fold, discovery's skips and partitions, the `partition_def` entries checked against them, and the refusals (ADR 0070; ADR 0081) |
| `dataset_reduction.rs` | the reduction and its synthesised Unionset (ADR 0049, ADR 0058, ADR 0070) |
| `relationship_resolution.rs` | the pooled join condition (ADR 0050, ADR 0052, ADR 0057) |
| `joinset_enumeration.rs` | every legal sub-composition as its own entry (ADR 0054) |
| `one_child_sugar.rs` | a one-child composition stored as its child under its name: each kind of declaration it carries, a nested one-child composition, a chain of them, the child's own temporal shape and granularity kept over any the composition states (finer, coarser or another kind), the composition's shape filling a child with none, and a Joinset child's own stated-grain refusal, and kept as authored where its parent cannot hold the child's kind |
| `grainset_composition.rs` | a Grainset end to end, and its implicit cross-grain Joinsets |
| `composed_temporal_shape.rs` | a Complex data kind's derived shape, persisted flat (ADR 0056) |
| `composed_constituents.rs` | a composition as a constituent of another, at every depth: private aliased entries, a nested composition's granularity and its three refusals, a join basis or a Joinset's Measure or row-aggregating Metric that no one child of a Grainset constituent maps, a composed Grainset child's filter-read rule, and the root keysets a nested composition serves (ADR 0026; ADR 0074; ADR 0077; ADR 0084 rule 8) |
| `model_repository_integrity.rs`, `remote_model_load.rs` | `Persistable` / `Repository` over `SemanticModel`, homed here because `save` needs `tokio` |
| `golden_generator.rs` | the message goldens' generator, which writes a golden only when the test named `generate_goldens` runs with `SEMSTRAIT_GOLDENS` set |

A refusal's message is compared with a golden under `tests/fixtures/<test>/` that only
`tests/common/golden.rs`'s generator writes:
`SEMSTRAIT_GOLDENS=1 cargo test -p semstrait-manifest generate_goldens`.
