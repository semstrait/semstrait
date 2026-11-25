---
id: 33-manifest-contract
status: ratified
version: 2.4
impl: implemented
---

# SemanticManifest Contract

This leaf owns the top-level manifest shape, metadata, SemanticBitmap registry, DataKind taxonomy, and load-time integrity validation.


## 1. Purpose and Scope

`semstrait-manifest` owns:

1. `compile`: `SemanticModel + (Option<&dyn CatalogProvider>) + filesystem -> SemanticManifest`.
2. The persisted manifest artifact (loading and saving via `Repository`).
3. The lightweight, model-aligned manifest type system — identity primitives on disk, structure rebuilt in memory by the planner runtime.

The artifact is intentionally lightweight and human-readable. It carries semantic identity primitives (the `SemanticBitmap` registry, `DataKind` entries, `SemanticInterface`, `SemanticBinding`, `Relationship`, `PhysicalSource`, expression pools) and the local coverage masks needed for first-cut path search. It does **not** persist a runtime graph, planner cache state, or per-Request derived structures.

This shape lands the 20 ratified clauses C1–C18 + CCK + CX1 (see `_research/manifest/RATIFICATION_LOG.md`).

## 2. Design Posture

- **Identity on disk, structure in memory.** Manifest persists identity primitives keyed by `EntityId`; `SemanticGraph` reconstructs nodes / edges / per-Request structures — and any compact integer traversal handles it needs — at graph build.
- **Single identity lane — `EntityId`.** Every manifest entity carries one durable `id: EntityId` and every collection is keyed by it (`BTreeMap<EntityId, _>`); every cross-reference is an `EntityId`. There are no separate per-kind compile-id newtypes (`DataKindId`, `SemanticsId`, `BindingId`, `SourceId`, `SemanticInterfaceId`, `RelationshipId`) on the manifest and no `stable_ids` side-map. Model-authored entities reuse their model `id`; compile-synthesised entities generate a deterministic `id` (§9.1). Compact integer handles used by the runtime planner are derived at graph build, never persisted.
- **Closed taxonomy mirror.** `DataKindVariant` enumerates the four ratified DataKinds (`Dataset`, `Unionset`, `Grainset`, `Joinset`) and is `#[non_exhaustive]` per I10.
- **Bitmap vs Bitmask vocabulary.**
  - **`SemanticBitmap`** = the registry, the canonical map keyed by `EntityId`, single, top-level. Holds full `SemanticDefinition` per entry plus its `bit_position` ordinal.
  - **`*Bitmask`** = a computed mask value over the bitmap (a coverage view), held on local entities. Suffix only; never standalone. The mask's only integer is `bit_position`, which maps back to an `EntityId` via the registry.
- **Public vs Nested DataKind split** mirrors spec 20: top-level `DataKind` carries an `id: EntityId`; `NestedDataKind` inlined into a parent variant carries a `structural_label` per spec 26 addressing (nested kinds have no top-level id).
- **Implicit Unionsets are top-level.** Multi-source `Dataset` auto-synthesises a Unionset (spec 21 §3.2 + spec 23 §2.1 row A) with a deterministic content-derived `id`. They live in `data_kinds`, not inside `NestedDataKind`.
- **Catalog-optional.** `compile` accepts `Option<&dyn CatalogProvider>`. When absent, model-sourced physical shape is used.
- **Deterministic input boundary.** Manifest ends at compile-time resolution + persistence. Runtime graph fragments, implicit-composition enumeration, and cache policy are planner graph-build concerns (`34`).
- **IR owns canonical expression vocabulary.** Manifest persists IR-owned `PhysicalExpr` (post-lowering), plus its `ExprLayer`, with `Serialize` / `Deserialize` derived inside `semstrait-ir` (spec 35). `UnresolvedSemanticExpr` is a transient compile input, not persisted.
- **Relationships are resolved, not re-exported.** The manifest stores a compile-resolved `ResolvedRelationship` (§8A), **not** the spec-18 authoring `Relationship`. Compile resolves endpoints to `EntityId` (`from` / `to` → `data_kinds`), applies the `optional` / `cross_filter` defaults matrix (spec 18 §2.7), derives `join_type` (spec 18 §2.9), and represents join keys as semantic-id pairs (`ResolvedJoinKey`) whose physical expression is reached through the bound `SemanticBinding.mapping`. This keeps the join contract at the semantic layer (SoC) while staying planner-complete (I5 / I8).
- **Temporal shape propagates onto leaf DataKinds.** Compile carries each leaf `Dataset`'s `TemporalShape` onto its manifest `DataKind` (§6.1) with the only resolution being identifying-Dimension `SemanticsName → EntityId`; `grain` is a generic field on the shape (mirrors the model). Planner shape-aware behaviour is deferred (spec 17 §10), but the manifest is the planner-complete record so the shape is not dropped.
- **Resolved naming policy.** Prefix `Resolved*` only for compile-complete forms; do not use it for author-level or deferred-runtime forms. `TemporalShape` carries no `Resolved*` prefix — it is propagation with a single id-resolution step, not a full compile-resolution form.

## 3. Public Crate Surface

```text
semstrait-manifest
├── manifest
│   ├── bitmap        // SemanticBitmap, SemanticDefinition, SemanticBitmask
│   ├── interface     // SemanticInterface, SemanticInterfaceBitmask
│   ├── data_kind     // DataKind, DataKindVariant, NestedDataKind, NestedDataKindVariant,
│   │                 //   DataKindOrigin
│   ├── unionset      // UnionsetBranch (used inside DataKindVariant::Unionset)
│   ├── grainset      // GrainsetLevel, RoutingUnitRef
│   ├── joinset       // JoinsetHop, HopDirection
│   ├── binding       // SemanticBinding, SemanticMappingValue, ResolvedExprMeta
│   ├── relationship  // ResolvedRelationship, ResolvedJoinKey
│   │                 //   (Cardinality, JoinType from IR/35; Integrity, Optional, CrossFilter from model/18 §2)
│   ├── temporal      // TemporalShape, TemporalType (ScdType from model/18 §3; Grain from IR/35, roster 13)
│   ├── identity      // EntityId (re-export of canonical id type from spec 31 §2.1)
│   ├── expr          // ManifestExpressions, ManifestExpression, PhysicalExprId
│   ├── source        // PhysicalSource, PhysicalSourceType,
│   │                 //   PhysicalSourceVersionRef, SourceColumn
│   └── metadata      // SemanticManifestMetadata
├── compile           // compile() entry + §3.9 pass orchestration; submodules:
│                      //   index, lower, graph, seal, auto_mapping, catalog, types
├── error             // CompileError, LoadError, Diagnose
├── load              // CX1 load-time validation (§10): SemanticManifest::validate()
│                      //   -> Result<(), LoadError> (single fail-fast pass)
├── path              // PathSignature, RelationshipPath
│                      //   (19 §3.4.5 — manifest-persisted, EntityId-based)
└── repository        // Repository + implementations
```

Primary root exports:

- `SemanticManifest`, `SemanticManifestMetadata`
- `EntityId`
- `SemanticBitmap`, `SemanticDefinition`, `SemanticBitmask`
- `SemanticInterface`, `SemanticInterfaceBitmask`
- `DataKind`, `DataKindVariant`, `NestedDataKind`, `NestedDataKindVariant`, `DataKindOrigin`
- `UnionsetBranch`
- `GrainsetLevel`, `RoutingUnitRef`
- `JoinsetHop`, `HopDirection`
- `SemanticBinding`, `SemanticMappingValue`, `ResolvedExprMeta`
- `PathSignature`, `RelationshipPath` (crate-root `path` module; `19 §3.4.5`)
- `ResolvedRelationship`, `ResolvedJoinKey`
- `TemporalShape`, `TemporalType`
- `ManifestExpressions`, `ManifestExpression`, `PhysicalExprId`, `ExprLayer` (re-export from `35`)
- `PhysicalSource`, `PhysicalSourceType`, `PhysicalSourceVersionRef`, `SourceColumn`
- `compile`, `Repository`
- `CompileError`, `LoadError`

## 4. `SemanticManifest` Contract

### 4.1 Top-level shape

```rust
#[non_exhaustive]
pub struct SemanticManifest {
    pub manifest_epoch: u64,
    pub model_hash: [u8; 32],
    pub catalog_fingerprint: Option<[u8; 32]>,
    pub model_id: EntityId,                                            // model-level identity (generated at compile)

    pub semantics:     SemanticBitmap,                                 // C4 — BTreeMap<EntityId, SemanticDefinition> inside
    pub interfaces:    BTreeMap<EntityId, SemanticInterface>,
    pub data_kinds:    BTreeMap<EntityId, DataKind>,                   // C5–C9 + CCK
    pub bindings:      BTreeMap<EntityId, SemanticBinding>,            // C2
    pub sources:       BTreeMap<EntityId, PhysicalSource>,             // C1, C3
    pub expressions:   ManifestExpressions,                            // C11, C12, C18 (expr-pool ids — §7.2)
    pub relationships: BTreeMap<EntityId, ResolvedRelationship>,       // root-scope only (§8A)

    pub metadata:      SemanticManifestMetadata,
}
```

Top-level constraints:

- Identity primitives only — no `nodes`, no `edges`, no `compositions`. `SemanticGraph` reconstructs nodes, edges, and per-Request composition structures at build time.
- **Single identity lane.** All entity collections are `EntityId`-keyed `BTreeMap`s; the key equals the value's own `id` field. Deterministic ordering follows the `EntityId` sort.
- Every persisted entity carries one `id: EntityId` (no separate compile-id newtypes, no `stable_ids` side-map). Model-authored entities reuse their model `id`; compile-synthesised entities (sources, bindings, interfaces, implicit Unionsets, synthesised fields) generate a deterministic `id` (§9.1).
- Vectors remain inside entities where ordered lists are meaningful (`branches`, `levels`, `hops`, `bindings`, etc.); their elements are `EntityId`s or carry their own `id`.
- `relationships` holds **root-scope** `ResolvedRelationship`s only (§8A) — the compile-resolved manifest form, **not** a re-export of the spec-18 authoring `Relationship`. Joinset-local (shadow) relationships per spec 18 §2.10 live inline on the Joinset variant as `ResolvedRelationship`s, not here (C7.6).
- The expression pool (`expressions.physical`) keeps its own content-dedup handle (`PhysicalExprId`, §7.2) — it addresses deduplicated expression subtrees, not named entities, and is intentionally not an `EntityId`. Each entry carries its `ExprLayer` (the applicability "extra data").

### 4.2 Top-level fingerprints

| Field | Width | Source |
|---|---|---|
| `manifest_epoch` | `u64` | Monotonic; bumps when canonical encoding of the manifest changes. |
| `model_hash` | `[u8; 32]` | SHA-256 over the source `SemanticModel`'s canonical encoding. |
| `catalog_fingerprint` | `Option<[u8; 32]>` | SHA-256 over fetched-content of catalog probe results (C14.6); `None` when catalog is absent. |

`catalog_fingerprint` is a fetched-content hash, not an identity-only hash. Provider-internal drift therefore propagates into `manifest_epoch` via canonical encoding (C18.2).

### 4.3 `SemanticManifestMetadata`

```rust
#[non_exhaustive]
pub struct SemanticManifestMetadata {
    pub created_at: SystemTime,
    pub semstrait_version: String,
    pub source_model_uri: Option<String>,
    pub annotations: BTreeMap<String, String>,
}
```

Free-form provenance for tooling; never load-bearing.

### 4.3.1 `EntityId` and `model_id` (single identity lane)

There is no `stable_ids` side-map. Identity rides on each entity: every manifest entity carries an `id: EntityId` (its map key), and every cross-reference is an `EntityId`. `EntityId` is the canonical id type defined in `semstrait-common` (`31 §2.1`) and re-exported here.

```rust
pub type EntityId = String; // canonical UUID text (lowercase, hyphenated)
```

Rules:

- `EntityId` values are canonical UUID text per the shape contract at `31 §2.1` (36 bytes, `8-4-4-4-12` lowercase hex, non-nil, version nibble unvalidated). Model-authored ids are UUIDv7 (`32`); compile-synthesised ids are UUIDv5 (name-based) derived deterministically from content (§9.1). The nibble records provenance and is not load-validated — CX1 (§10 check #6) checks shape and global uniqueness only, so a change of derivation strategy on either side needs no load-check amendment.
- Every persisted `DataKind`, `SemanticDefinition`, `SemanticInterface`, `SemanticBinding`, `PhysicalSource`, and `Relationship` carries an `id` equal to its collection key.
- `EntityId` values are globally unique across all manifest collections, plus the top-level `model_id`.
- Model-authored entities reuse the entity's model `id`; missing model ids were already resolved at parse per `IdentityProfile` (`32 §9.0.1`), so compile sees every model entity already carrying an `id`.
- `model_id` is the top-level model identity (`SemanticManifest.model_id`). It is minted deterministically from `model_hash` (§9.1) and is **not** the model-root `id` (`32 §1.4a`). Both are deterministic UUIDv5, and the difference is what each is keyed on: `model_id` is keyed on **content**, so it changes whenever the model changes — it identifies *this compilation*. The root `id` is keyed on the authored document (explicit, else derived from `name`), so it must *not* change when content changes — it identifies *which document*. Compile does not consume the root `id`, and the root `id` stays outside the canonical bytes `model_hash` covers: folding document identity into a content hash would flip `model_hash` on a rename that changed no semantics.

### 4.4 Deterministic payload consumed by graph build

`semstrait-manifest` does not build `SemanticGraph`. It persists deterministic inputs that planner runtime graph build (`34`) consumes.

| Payload slice | Deterministic intent | Used by graph build for |
|---|---|---|
| `sources: BTreeMap<EntityId, PhysicalSource>` | Stable source identity plus version/schema snapshot (`version_ref`, `schema_fingerprint`) | source-node identity, drift checks, and source-index keys |
| `data_kinds`, `relationships`, `interfaces` | `EntityId`-keyed composition primitives (`interface_id`, `coverage`, variant payloads); no runtime edges persisted | node/edge synthesis and composition candidate indexing |
| `DataKindVariant::{Joinset, Grainset, Unionset}` fields | Explicit composition hints (`anchor`, `members`, `hops`, branch/level routing — all `EntityId`) | deterministic graph topology reconstruction |
| `SemanticBitmap.entries[*].bit_position` + all `SemanticBitmask` values | Explicit `bit_position ↔ EntityId` mapping and canonical word encoding | coverage math and subset/intersection checks |
| `expressions.physical` (`ManifestExpression { expr, layer }`) + `PhysicalExprId` refs in bindings | Physical expression storage with applicability `ExprLayer` | expression placement (scalar/aggregate/post-aggregate) + node/edge lookup via `GraphExprRef` (`35`) |
| Per-entity `id: EntityId` + top-level `model_id` | Durable model identity on each primitive | cross-run identity continuity and external correlation |

Indexing note: runtime indices (`name_index`, composition-by-constituent-set, adjacency maps, and any compact integer traversal handles) are derived from these maps at graph build and are never persisted back into manifest.

## 5. `SemanticBitmap` Registry (C4)

The `SemanticBitmap` is the canonical, single, top-level registry of every semantic (`EntityId`) reachable from the model. All per-entity `*Bitmask` values are coverage views over this registry, addressed by `bit_position`.

```rust
#[non_exhaustive]
pub struct SemanticBitmap {
    pub entries: BTreeMap<EntityId, SemanticDefinition>,
}

#[non_exhaustive]
pub struct SemanticDefinition {
    pub id: EntityId,            // model id (Dimension/Measure/Metric/Key) or generated (synthesised Field); also the map key
    pub name: SemanticsName,
    pub role: SemanticRole,
    pub data_type: Option<DataType>,
    pub bit_position: u32,       // the lone integer; epoch-stable bit ordinal
}

#[non_exhaustive]
pub enum SemanticRole {
    Dimension,
    Measure,
    Metric,
    Key,
    Field,
}

#[non_exhaustive]
pub struct SemanticBitmask {
    pub words: Vec<u64>,
}
```

Rules:

- **Scope is global.** A bit position spans all semantics (`EntityId`s) in the manifest's set.
- **Wide registry.** Each entry carries the full `SemanticDefinition` (per-semantic `id`, name, role, optional `DataType`, and resolved `bit_position`). Per-semantic attributes that previously lived in a `SemanticNodePayload::Semantic` wrapper now live here.
- **Epoch-stable; cross-epoch renumber allowed.** Within one `manifest_epoch`, `bit_position` is stable. Bumping `manifest_epoch` rebuilds positions from canonical sort over `EntityId`. Value lookup at any read site is `bitmap.entries.get(&entity_id)`; the reverse index `bit_position -> EntityId` is derived at load (not persisted) so a set bit resolves to its `EntityId` and then its definition.
- **Bitmask encoding** (CCK.5): `Vec<u64>` words; bit `n` of word `n / 64` shifted by `n % 64` corresponds to the entry whose `bit_position == n`.
- **Canonical bitmask form.** `words` is little-endian by word index; bit 0 in each word is its least-significant bit. Trailing all-zero words are invalid. Empty coverage is encoded as `Vec::new()`.
- **Bitmask validity.** Any set bit whose position is absent from `SemanticBitmap.entries[*].bit_position` is a load-time integrity error.

`EntityId` never participates in bit math; `bit_position` is the only integer in the bitmap subsystem and exists solely to pack masks into words.

`SemanticInterfaceBitmask` (the per-interface mask projection) is structurally a `SemanticBitmask` and lives on `SemanticInterface`.

```rust
#[non_exhaustive]
pub struct SemanticInterface {
    pub id:         EntityId,        // generated at compile; also the map key in `interfaces`
    pub dimensions: Vec<EntityId>,
    pub measures:   Vec<EntityId>,
    pub metrics:    Vec<EntityId>,
    pub keys:       Vec<EntityId>,
    pub fields:     Vec<EntityId>,
    pub bitmask:    SemanticInterfaceBitmask,
}

pub type SemanticInterfaceBitmask = SemanticBitmask;
```

The vector-of-ids fields (each an `EntityId` into `semantics`) preserve declared shape and ordering for human inspection; `bitmask` is the membership view used by graph-build path search. An interface has no model-authored identity, so its `id` is generated deterministically at compile (§9.1).

## 6. `DataKind` and `DataKindVariant` (CCK + C5–C9)

### 6.1 Top-level `DataKind`

```rust
#[non_exhaustive]
pub struct DataKind {
    pub id: EntityId,              // model id (Explicit) or generated (Implicit); also the map key
    pub name: DataKindName,
    pub interface_id: EntityId,    // -> interfaces
    pub origin: DataKindOrigin,
    pub coverage: SemanticBitmask,
    pub temporal: Option<TemporalShape>,   // §8B — leaf-effective shape (None when shape-unclassified)
    pub variant: DataKindVariant,
}

#[non_exhaustive]
pub enum DataKindOrigin {
    Explicit,
    Implicit,
}
```

Notes:

- `coverage` is the universal union view (CCK.1) — every DataKind carries it; satisfies spec 20 invariant D4.
- `interface_id` links every `DataKind` to one canonical `SemanticInterface` entry in `interfaces` (by `EntityId`).
- `origin` distinguishes author-declared (`Explicit`) from compile-synthesised entries such as multi-source-Dataset auto-Unionsets (`Implicit`). Diagnostic-only — runtime semantics are identical (C9.5).
- `id` reuses the model `id` for `Explicit` DataKinds and is a deterministic generated id for `Implicit` ones (§9.1).
- `temporal` carries the leaf `Dataset`'s resolved `TemporalShape` (§8B). It is `Some` only when the leaf authored `extras.temporal:`; complex kinds (Unionset / Grainset / Joinset) carry `None` at this level (spec 17 §3.2 — shape cascades to leaves during compile, the composed surface derives its effective shape at graph build). The identifying-Dimension references inside it are resolved to `EntityId`; `grain` is leaf-effective (spec 18 §3.3, SR-E-6).

### 6.2 `DataKindVariant`

```rust
#[non_exhaustive]
pub enum DataKindVariant {
    Dataset {
        bindings: Vec<EntityId>,                                       // -> bindings; len >= 1
    },
    Unionset {
        mode: UnionMode,
        branches: Vec<UnionsetBranch>,                                 // len >= 2
    },
    Grainset {
        levels: Vec<GrainsetLevel>,                                    // len >= 2
    },
    Joinset {
        anchor: EntityId,                                              // -> data_kinds
        members: Vec<EntityId>,                                        // -> data_kinds; anchor in members; len == 2 (binary v1)
        hops: Vec<JoinsetHop>,                                         // len == 1 (binary v1); cumulative
        path_origin: DataKindOrigin,
        scope_local_relationships: Vec<ResolvedRelationship>,          // §2.10 shadow; resolved form (§8A); each carries its own `id`
    },
}
```

### 6.3 `NestedDataKind`

Mirrors spec 20's Public/Nested split. Inlined into a parent variant — no separate id, just a structural address (spec 26 §4) and a coverage view.

```rust
#[non_exhaustive]
pub struct NestedDataKind {
    pub structural_label: String,                  // spec 26 addressing; nested kinds have no top-level id
    pub coverage: SemanticBitmask,
    pub temporal: Option<TemporalShape>,           // §8B — leaf-effective (e.g. Grainset child grain, spec 17 §3.4)
    pub variant: NestedDataKindVariant,
}

#[non_exhaustive]
pub enum NestedDataKindVariant {
    Dataset {
        bindings: Vec<EntityId>,                   // -> bindings
    },
    Unionset {
        mode: UnionMode,
        branches: Vec<UnionsetBranch>,
    },
    Grainset {
        levels: Vec<GrainsetLevel>,
    },
}
```

`Joinset` is intentionally absent from `NestedDataKindVariant`: per spec 26 R2/R3 a Joinset never nests under another Joinset, and other parents project to top-level `data_kinds` entries by id.

### 6.4 `Dataset` variant (C5)

```rust
DataKindVariant::Dataset {
    bindings: Vec<EntityId>,                        // -> bindings; len >= 1
}
```

Rules:

- **Coverage origin.** Top-level `coverage` is the union of bindings' covered semantics. Only `Native` and `Derived` mapping coverage contribute bits (C5.1) — `NullFill` / `Metadata` mapping kinds are excluded (they don't represent real source-side coverage).
- **Cardinality.** `bindings.len() >= 1`. Empty is a compile error (cascade from spec 20 §3 and spec 15 §2.1: a leaf Dataset carries at least one Binding).
- **No special orphan-bit rule** — unresolved semantics are caught earlier by G2 (C13).

### 6.5 `Unionset` variant (C6)

```rust
DataKindVariant::Unionset {
    mode: UnionMode,
    branches: Vec<UnionsetBranch>,                 // len >= 2; name order (matches model child sequence, spec 23 §3.1)
}

#[non_exhaustive]
pub struct UnionsetBranch {
    pub kind: NestedDataKind,
    pub branch_coverage: SemanticBitmask,
}
```

Rules:

- `mode` is the `UnionMode { All, Unique }` roster ratified for v1 (spec 23 §2.1).
- `branches.len() >= 2` per spec 26 R3.
- Vector follows the model's canonical child sequence — **name order** (spec 23 §3.1; id-first rework). Renaming a branch child is a real model edit.
- `branch_coverage` records each branch's locally-covered semantics (Native/Derived bits per C5.1 cascade). The complement `top_level_coverage \ branch_coverage` is the implicit NullFill mask, derived at graph-build (spec 23 §1.3 I1) — the manifest does not persist it (C6.2).
- **Implicit Unionset placement.** Multi-source `Dataset` auto-synthesises a Unionset per spec 21 §3.2; the result is a top-level entry in `data_kinds` with `origin = Implicit` and a deterministic content-derived `id` (spec 23 §2.1 row A). It is **not** a `NestedDataKind`.

### 6.6 `Grainset` variant (C8)

```rust
DataKindVariant::Grainset {
    levels: Vec<GrainsetLevel>,                    // len >= 2; coarsest-first; distinct grains
}

#[non_exhaustive]
pub struct GrainsetLevel {
    pub grain: Grain,
    pub routing_unit: RoutingUnitRef,
    pub level_coverage: SemanticBitmask,
}

#[non_exhaustive]
pub enum RoutingUnitRef {
    Inline(NestedDataKind),                        // single same-grain child
    Synthesized(EntityId),                         // -> data_kinds (implicit-Unionset top-level; >= 2 same-grain children)
}
```

Rules:

- `levels.len() >= 2` (>= 2 unique grains per spec 22 §5.2).
- Each `level.grain` is distinct.
- Coarsest-first level ordering (spec 22 §5.2 / spec 12 §4.2).
- `RoutingUnitRef::Inline` is used when a level has a single same-grain child; `RoutingUnitRef::Synthesized` references a top-level implicit Unionset when a level has two or more same-grain children (spec 22 §3.3 same-grain pre-merge + spec 23 §2.1 row A).
- **Cross-grain JOIN-tree is not persisted** — only per-level `level_coverage` and Keys (already in the level's interface). `SemanticGraph` synthesises the JOIN-tree at build (C8.2). Cascade: spec 22 §1.3 I8 must drop "JOIN-tree shape, per-pair JOIN-key index, ComposedSemanticInterface" from `ResolvedGrainset`'s manifest contract.

### 6.7 `Joinset` variant (C7)

```rust
DataKindVariant::Joinset {
    anchor: EntityId,                              // -> data_kinds
    members: Vec<EntityId>,                        // -> data_kinds; len == 2 (binary v1); anchor in members
    hops: Vec<JoinsetHop>,                         // len == 1 (binary v1); cumulative coverage
    path_origin: DataKindOrigin,
    scope_local_relationships: Vec<ResolvedRelationship>,  // §2.10 shadow (§8A); bounded to this Joinset; each carries its own `id`
}

#[non_exhaustive]
pub struct JoinsetHop {
    pub from: EntityId,                            // -> data_kinds
    pub to: EntityId,                              // -> data_kinds
    pub relationship: EntityId,                    // -> relationships (root-scope) or scope_local_relationships
    pub direction: HopDirection,
    pub hop_coverage: SemanticBitmask,             // cumulative across hops [0..=i]
}

#[non_exhaustive]
pub enum HopDirection { Forward, Reverse }
```

Rules:

- **Cumulative coverage semantics** (C7.2). `hops[i].hop_coverage` = union of semantics reachable starting at `anchor` and walking hops `[0..=i]`. SemanticGraph's first-cut path search reads `requested_mask & hops[i].hop_coverage == requested_mask` directly.
- **Binary v1 invariants** (C7.5). `members.len() == 2`, `hops.len() == 1`. Surfaced at load via CX1; defense-in-depth twin of canonical compile errors `VALID_E_2400` / `COMP_E_2408`.
- **Anchor invariant** (C7.7). `anchor in members`. Defense-in-depth twin of canonical `VALID_E_2402`.
- **Top-level `coverage` of a Joinset DataKind** equals `hops.last().hop_coverage` (the last cumulative hop is the full Joinset coverage).
- **`path_origin`.** Diagnostic / audit tag distinguishing explicit-author vs implicit-search-derived hop sequences (spec 24 §I5). Reuses `DataKindOrigin` directly (`Explicit | Implicit`) to avoid a duplicate single-use enum.
- **Scope-local relationships persisted inline** (C7.6). Joinset-bounded shadow relationships per spec 18 §2.10 live on the variant as `ResolvedRelationship`s (§8A); root-scope `relationships` only holds non-shadow `ResolvedRelationship`s. This avoids a `relationship.scope` discriminator and keeps shadow visibility cleanly bounded.
- **No `ComposedSemanticInterface`** (C7.4). Manifest carries hops + per-hop coverage; `SemanticGraph` synthesises `UnifiedSemantics` and `FieldProvenance` at build. Cascade: spec 24 §1.4 I8 must drop "resolved ComposedSemanticInterface" from `ResolvedJoinset`'s manifest contract; §2.4 must retire the `interface: ComposedSemanticInterface { … }` pseudo-shape.

### 6.8 Implicit composition (C9)

`compositions:` is **not** a top-level field (C9.2). The `SemanticGraph` runs BFS at build time over `relationships` and per-DataKind `coverage` to resolve a `CompositionKind::Relationship` request (spec 16 §11). Cascade: spec 16 §10.4's "compile-time-enumerated compositions" framing must be aligned to the lightweight posture — that index lives in `SemanticGraph`, not in manifest.

`MAX_IMPLICIT_COMPOSITION_DEPTH = 4` (C10.1) is a compile-time-only constant in `semstrait-ir` (or the graph crate when materialized) and is **not** persisted on manifest (C10.2 / C10.4).

## 10. Load-time Integrity (CX1)

Manifest load performs a **single integrity pass**. Every cross-reference in the loaded artifact must resolve to an existing entry in its target collection. Failure surfaces as a typed `LoadError`:

```rust
#[non_exhaustive]
pub enum LoadError {
    DanglingReference {
        from: ReferenceSite,
        to_kind: ReferenceTargetKind,
        target_id: String,
    },
    StructuralViolation {
        site: ReferenceSite,
        rule: &'static str,
    },
    // ... (other variants)
}
```

Cross-references checked at load:

| Reference | From | To |
|---|---|---|
| `SemanticBinding.data_kind_id` | binding | `data_kinds` |
| `SemanticBinding.source_id` | binding | `sources` |
| `SemanticBinding.mapping[*]::Expr(id)` | binding mapping | `expressions.physical` |
| `SemanticBinding.resolved` keyset | binding | coincides with the binding's covered-semantic set (every `resolved` key is also a `mapping` key, and vice-versa) |
| `ResolvedExprMeta.path_signature` (`Some(ps)`: every `EntityId` in `ps.paths[*].0`) | binding resolved-meta | `relationships` |
| `DataKind.interface_id` | DataKind | `interfaces` |
| `DataKindVariant::Dataset.bindings[*]` | DataKind | `bindings` |
| `RoutingUnitRef::Synthesized(id)` | Grainset level | `data_kinds` |
| `JoinsetHop.relationship` | Joinset hop | `relationships` *or* `Joinset.scope_local_relationships` |
| `JoinsetHop.from` / `JoinsetHop.to` | Joinset hop | `data_kinds` |
| `Joinset.anchor`, `Joinset.members[*]` | Joinset variant | `data_kinds` |
| `ResolvedRelationship.from` / `.to` | relationship | `data_kinds` |
| `ResolvedJoinKey.from` / `.to` | relationship key | `semantics.entries` (on the `from` / `to` interface respectively) |
| `ResolvedRelationship.filter::Some(id)` | relationship | `expressions.physical` |
| `TemporalType.{occurred_at,event_time,snapshotted_at,valid_from,valid_to}` | DataKind `temporal` | `semantics.entries` (by `EntityId`) |
| `SemanticInterface.{dimensions,measures,metrics,keys,fields}[*]` | interface | `semantics.entries` (by `EntityId`) |
| `SemanticBitmask.words` (any set bit) | any bitmask | `semantics.entries` (some entry has matching `bit_position`; out-of-range bits are invalid) |
| `SemanticDefinition.bit_position` uniqueness | bitmap registry | within bitmap |
| Every entity's `id` equals its collection key | each collection | self (`data_kinds`/`semantics`/`interfaces`/`bindings`/`sources`/`relationships`) |

All cross-reference targets above are `EntityId`s. Because identity lives on each entity (no side-map), there is no separate `stable_ids` table to validate — a dangling `EntityId` reference is caught by the rows above directly.

Structural violations also surface here as a defense-in-depth twin of compile-time gates:

- `Dataset.bindings.len() >= 1`
- `Unionset.branches.len() >= 2`
- `Grainset.levels.len() >= 2`; distinct grains; coarsest-first ordering
- `Joinset.members.len() == 2` (binary v1); `Joinset.hops.len() == 1` (binary v1); `Joinset.anchor in members`
- `SemanticBitmask.words` canonical form (no trailing all-zero words; empty mask is `Vec::new()`)
- every entity `id` is canonical UUID text and equals its map key; all `id`s plus `model_id` are globally unique across collections
- every `ManifestExpression.layer` equals `classify(expr)` (`19 §3.2.6`) — the persisted layer is consistent with the expression tree
- `ResolvedRelationship.join_type` equals the derivation of `ResolvedRelationship.optional` (spec 18 §2.9) — the persisted derived value is consistent with its source
- `Joinset` top-level `coverage` equals `hops.last().hop_coverage` (I12.10) — the persisted cumulative coverage agrees with the final hop
- each `ResolvedJoinKey.from` / `.to` semantic has `role ∈ {Dimension, Field, Key}` (I12.22) — a Measure / Metric cannot be a join key
- `DataKind.temporal` / `NestedDataKind.temporal` is `Some` only on a `Dataset` variant (leaf); `Unionset` / `Grainset` / `Joinset` carry `None` (spec 17 §3.2)

This hardens C13's G1 / G2 / G4 from compile-time-only to compile-time + load-time. A hand-edited or repository-corrupted manifest cannot slip a dangling reference past load. The wire format remains an id-keyed map plus id reference (no OpenAPI-style `$ref` JSON pointers); CX1 preserves the integrity guarantee at less verbosity cost.

## 11. `Resolved*` Naming Rule

Use `Resolved*` only for compile-complete forms where uncertainty is eliminated (for example `ResolvedSemanticName`, resolved mapping, resolved schema-compatibility checks). Do **not** use `Resolved*` for author-level forms or for forms whose reconciliation is deferred to runtime / planner.

`PhysicalExpr` is not `Resolved*`-prefixed even though it is persisted post-resolution; it lives in IR and follows IR's naming policy.

## 12. Invariants

- I12.1 Manifest serialization does not contain runtime graph or planner cache state.
- I12.2 Single identity lane: top-level entity collections are `EntityId`-keyed maps whose key equals the value's own `id` (no per-kind compile-id newtypes, no `stable_ids` side-map); every persisted `DataKind`, `SemanticDefinition`, `SemanticInterface`, `SemanticBinding`, `PhysicalSource`, and `Relationship` carries one `id: EntityId`, and all `id`s plus the top-level `model_id` are canonical UUID text and globally unique within one manifest.
- I12.2b Compile-synthesised ids (sources, bindings, interfaces, implicit Unionsets, synthesised fields, `model_id`) are deterministically derived from content via UUIDv5 (per-entity-type namespace + content seed, §9.1), so identical `(model, catalog)` inputs produce byte-identical manifests (I4). Authored ids are UUIDv7; generated ids are UUIDv5.
- I12.3 `SemanticBitmap.entries[id].bit_position` is unique within a `manifest_epoch`; cross-epoch renumber (canonical sort over `EntityId`) requires an epoch bump.
- I12.4 Every set bit in any `SemanticBitmask` corresponds to some `SemanticDefinition.bit_position` within the same manifest; bitmasks use canonical encoding (no trailing all-zero words, empty mask encoded as `Vec::new()`).
- I12.5 Every `DataKind.interface_id` (an `EntityId`) references an existing `interfaces` entry. Every binding `EntityId` in a `Dataset.bindings` references an existing `bindings` entry; every binding in turn references existing `data_kinds`, `sources`, and `expressions.physical` entries.
- I12.6 `Dataset.bindings.len() >= 1`; `Unionset.branches.len() >= 2`; `Grainset.levels.len() >= 2` with distinct grains in coarsest-first order; `Joinset.members.len() == 2` and `Joinset.hops.len() == 1` (binary v1); `Joinset.anchor in Joinset.members`.
- I12.7 `Joinset.scope_local_relationships` is bounded to that Joinset's shadow scope; root-scope `relationships` does not contain shadow Relationships.
- I12.8 `data_kinds` includes implicit Unionsets (multi-source Dataset auto-synthesis) as top-level entries with a deterministic content-derived `id` and `origin = Implicit`.
- I12.9 Grainset same-grain merge — when a level has >= 2 same-grain children, `RoutingUnitRef::Synthesized` references a top-level implicit Unionset; single same-grain child uses `RoutingUnitRef::Inline`.
- I12.10 `JoinsetHop.hop_coverage` is cumulative across `[0..=i]`; `Joinset` top-level `coverage` equals `hops.last().hop_coverage`.
- I12.11 `ManifestExpressions.physical` is a single content-dedup'd pool of `ManifestExpression { expr, layer }`; no semantic pool is persisted. Every entry's `layer` equals `classify(expr)` (`19 §3.2.6`), checked at load (CX1).
- I12.12 `PhysicalSource.schema_fingerprint`, when present, equals the SHA-256 of the canonical length-prefixed encoding of `projected_schema` per §8.4.
- I12.13 `PhysicalSource.schema_fingerprint` is `None` iff `projected_schema.is_empty()`.
- I12.14 `compile`'s `catalog: Option<&dyn CatalogProvider>` parameter selects between catalog-mediated and model-as-truth fetch paths; when both supply a value for the same source, model wins (with a warning).
- I12.15 No `compositions:`, `nodes:`, or `edges:` collection appears at manifest top level; those structures are reconstructed by `SemanticGraph` at build time.
- I12.16 `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` is a code-level constant in `semstrait-ir` (or graph crate) and is not persisted on manifest.
- I12.17 Manifest load runs CX1's single-pass cross-reference and structural integrity checks; failure surfaces as `LoadError` and prevents any planner consumption of the artifact.
- I12.18 `Resolved*` prefix is used only for compile-complete forms.
- I12.19 Graph build is deterministic over manifest primitives: same manifest bytes + same builder options => same node/edge/index reconstruction.
- I12.20 Every expression reference that crosses into graph/planner flow is a `PhysicalExprId` and resolves by id, not by runtime semantic-name re-resolution.
- I12.21 `relationships` and `Joinset.scope_local_relationships` hold compile-resolved `ResolvedRelationship`s (§8A), not the spec-18 authoring `Relationship`. Each carries resolved `EntityId` endpoints (`from` / `to` → `data_kinds`), materialised `optional` / `cross_filter` (no `Option`), and a `join_type` derived from `optional` (spec 18 §2.9).
- I12.22 A `ResolvedJoinKey` side is a semantic `EntityId` (Dimension / Field / Key on the respective interface), its lowered `PhysicalExpr` reached via the bound `SemanticBinding.mapping` (see §8A.2); relationship key/filter expressions are interned in `expressions.physical` (no inline expression on the relationship).
- I12.23 `DataKind.temporal` / `NestedDataKind.temporal`, when `Some`, carries identifying-Dimension refs resolved to `EntityId` (members of `semantics`); `grain` is a generic field on `TemporalShape`. `temporal` is populated only on leaf `Dataset` variants (top-level or nested); complex kinds derive their effective shape at graph build (spec 17 §3.2).

## Testing Strategy

- SemanticManifest round-trip: serialize and deserialize preserves all structure
- EntityId single-lane uniqueness: no collisions across collections + model_id
- SemanticBitmap bit_position uniqueness and canonical bitmask encoding
- DataKindVariant per-variant cardinality and structural rules (Dataset bindings.len>=1, Unionset branches.len>=2, Grainset levels.len>=2 coarsest-first distinct, Joinset binary v1 + anchor in members)
- CX1 dangling-reference detection and structural violation checks (I12.1–I12.23)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
