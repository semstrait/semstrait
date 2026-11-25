---
id: 33-binding-expr-source
status: ratified
version: 2.1
impl: implemented
---

# Binding, Expression, and Source Shapes

This leaf owns the SemanticBinding contract, ManifestExpressions pool, PhysicalSource shape, ResolvedRelationship and ResolvedJoinKey forms, and TemporalShape propagation.

## 7. `SemanticBinding` and `ManifestExpressions` (C2 + C11 + C12)

### 7.1 `SemanticBinding`

```rust
#[non_exhaustive]
pub struct SemanticBinding {
    pub id: EntityId,                                                  // generated at compile; also the map key in `bindings`
    pub data_kind_id: EntityId,                                        // C2.4 — leaf-only direct linkage (-> data_kinds)
    pub source_id: EntityId,                                           // -> sources
    pub mapping: BTreeMap<EntityId, SemanticMappingValue>,             // keyed by semantic EntityId — *how* each covered semantic is realized
    pub resolved: BTreeMap<EntityId, ResolvedExprMeta>,               // keyed by semantic EntityId — compile-*derived facts* per (semantic, binding) (19 §3.2.4)
}

#[non_exhaustive]
pub enum SemanticMappingValue {
    Column(ColumnRef),
    Literal(Literal),
    Expr(PhysicalExprId),                                              // C11 — expr-pool handle (§7.2)
    MetadataRef(String),
}

/// Per-`(semantic, binding)` compile-derived facts — the context fields of
/// `19`'s `ResolvedExprEntry` (`19 §3.2`) that are **not** a function of the
/// expression tree alone (a shared tree bound to different sources may infer a
/// different type or traverse different relationship paths), so they cannot
/// ride the content-deduped expression pool (`§7.2`) and are persisted here
/// instead (`19 §3.2.4`).
#[non_exhaustive]
pub struct ResolvedExprMeta {
    pub inferred_type:      DataType,                // root type after boundary reconciliation (19 §3.7)
    pub referenced_columns: Vec<String>,             // flat, de-duplicated native columns read (19 §3.10)
    pub path_signature:     Option<PathSignature>,   // cross-DataKind relationship paths (19 §3.4.5); None = same-kind, no walk
}
```

Rules:

- **`mapping` vs `resolved` — two parallel views of the same covered semantics.** Both are keyed by the same semantic `EntityId`. `mapping` answers **how** a semantic is realized against the bound source (a column, a literal, a pooled `PhysicalExprId`, or a metadata key); `resolved` answers **what compile derived** about it (root type, native columns read, cross-DataKind paths). The two keysets coincide — every covered semantic has exactly one entry in each. For an `Expr`-realized semantic the pooled `PhysicalExprId` lives in `mapping` and its tree-independent context lives in `resolved`; the `PhysicalExprId` is **not** duplicated into `resolved`.
- **Seal-time split of `19`'s `ResolvedExprEntry`.** Compile's per-`(Semantics, Binding)` `ResolvedExprEntry { physical_expr, layer, inferred_type, referenced_columns, path_signature, provenance }` (`19 §3.2`) is **not** persisted whole. On seal it decomposes: `{physical_expr, layer}` interns into the content-deduped pool (`§7.2`); the pool handle lands in `mapping[sem] = Expr(id)`; `{inferred_type, referenced_columns, path_signature}` lands in `resolved[sem]`; `provenance` flows to manifest metadata (`§6.3.1`). This is the on-disk counterpart of `19 §3.2.4`'s deferred per-binding record — **ratified here**; `19` owns the field set, `33` owns this container.
- **`PathSignature` shape and `None`-for-same-kind.** Defined at `19 §3.4.5` as `PathSignature { paths: BTreeSet<RelationshipPath> }` with `RelationshipPath(Vec<EntityId>)` — a chain of relationship `EntityId`s. `path_signature` is `None` when no cross-kind walk occurred (every typed leaf resolved within the current DataKind); `Some(ps)` is populated only by `Joinset` and the implicit-composition consumer (`16 §11`) for cross-member / cross-DataKind references (`25 §<matrix>`). The `BTreeSet` dedupes identical chains deterministically (I4).
- **Leaf-only linkage** (C2.1). Bindings attach to leaf `Dataset` DataKinds; composite DataKinds (Unionset / Grainset / Joinset) traverse children to reach sources.
- **One-to-many** (C2.2). One Dataset can carry multiple bindings, each pointing to a distinct `PhysicalSource` (e.g., partitioned read, dual feed).
- **Forward only** (C2.3). Dataset variant carries `bindings: Vec<EntityId>`; binding carries `source_id`; reverse (`source_id -> [binding id]`) is derived at load.
- **Generated identity.** A binding has no model-authored id; its `id` is generated deterministically at compile from `(data_kind_id, source_id, mapping)` (§9.1).
- **`SemanticMappingValue::Expr` references `PhysicalExprId`.** Manifest persists only the post-desugar physical form referencing native columns; the semantic/authoring form is lowered away at compile and not persisted (§7.2). `PhysicalExprId` is an expression-pool handle (content-dedup), not an `EntityId` (§7.2). The pool entry it resolves to carries the expression's `ExprLayer`.

### 7.2 `ManifestExpressions` (physical-only pool + layer)

```rust
#[non_exhaustive]
pub struct ManifestExpressions {
    pub physical: BTreeMap<PhysicalExprId, ManifestExpression>,
}

/// A persisted expression plus the "extra data" the planner needs to place it:
/// its applicability layer (`35 §5.5`).
#[non_exhaustive]
pub struct ManifestExpression {
    pub expr:  PhysicalExpr,
    pub layer: ExprLayer,        // Scalar | Aggregate | PostAggregate (35 §5.5)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalExprId(pub ExprId);
```

Rules:

- **Physical-only pool** (supersedes C11's "both forms persisted"). Compile lowers every `UnresolvedSemanticExpr → PhysicalExpr` (Phase A, `19 §3`) and persists **only** the physical form. `UnresolvedSemanticExpr` is a transient compile input, not a manifest artifact; authoring round-trip and diagnostics are served by re-reading the source model (the source of truth), so a persisted semantic pool added storage without a load-bearing consumer. There is no `semantic` pool and no `SemanticExprId`.
- **Layer is the extra data** (this rework). Each entry carries its `ExprLayer` so the planner allocates the expression to the right `PlanNode` layer (scalar pushdown / aggregate / post-aggregate) without re-deriving from semantic provenance. The layer is assigned at compile (`19 §3.2.6`) and is a pure function of the tree.
- **Additivity is derived, not stored** (v1). For an `Aggregate`-layer expression, pre-/re-aggregation safety comes from the aggregate op via `AdditivitySource` (`14a §3.6.2`) — function-only in v1. Model-level additivity is the reserved extension point; it is not persisted here.
- **Sugar is desugared before persistence** (D2). The persisted `PhysicalExpr` is post-desugar canonical form referencing native columns; sugar never reaches the pool.
- **Strongly typed ID** (C12.3, narrowed). `PhysicalExprId` wraps `ExprId`; with a single pool the cross-pool-mix-up hazard disappears, but the newtype is retained so expression handles stay distinct from `EntityId`.
- **Expression ids are not `EntityId`s.** Expressions are content-deduplicated subtrees, not named/authored entities, so they keep their own pool handle (`PhysicalExprId`). This is the single deliberate exception to the manifest's one-id-lane rule; expression identity is content-addressed (C12.4), not durable model identity.
- **Compile-time dedup** (C12.4). Equivalent expressions collapse to one `PhysicalExprId`. Because `ExprLayer` is a function of the tree, dedup is layer-consistent (same tree ⇒ same layer).
- **C12.5 (no cross-pool link) is moot** — there is one pool.
- **Derived views only** (C16). Expression coverage and reverse indexes (for example semantic `EntityId -> [PhysicalExprId]`) are derived at load/graph-build time, not persisted.
- **All lowered expressions intern here.** The pool is the single home for every `PhysicalExpr` the manifest references — computed-semantic bodies, binding `Expr` mappings, **and** relationship residual `filter`s (`§8A.1`) plus the lowered bodies behind relationship join keys (reached via the synthesised-`Field` binding mapping, `§8A.2`). Relationship expressions are not stored inline on `ResolvedRelationship`; they flow through Phase A (`19 §3`) into this pool like any other expression.

### 7.3 Serialization mechanism (C18)

- **Encoding format.** JSON via `serde` (C18.1). Matches §1's readability posture.
- **Determinism.** Deterministic encoding required (C18.2) — same `Expr` value renders identical bytes across compilations. Required for `manifest_epoch` / `model_hash` stability and for content dedup. Concretely: `BTreeMap` ordering, sorted `serde` field order, length-prefixed primitives, no NaN / Infinity in literal floats unless explicitly carried by typed Literal width discriminator.
- **Ownership of derives.** `impl Serialize` / `impl Deserialize` for `Expr<L>` and the leaf families lives in `semstrait-ir` (C18.3); manifest consumes them. Manifest does **not** vendor a parallel encoder.
- **Versioning.** No per-Expr format version field (C18.4). `manifest_epoch` is the migration signal; cross-epoch encoding changes are valid because graph-build re-derivations regenerate from primitives.
- **Explicit ID** (C18.5). Each persisted `PhysicalExpr` is referenced by an explicit `PhysicalExprId`; compile-time content dedup is permitted under (C12.4).

## 8. `PhysicalSource` (C1 + C3 + C14 + C15)

### 8.1 Shape

```rust
#[non_exhaustive]
pub struct PhysicalSource {
    pub id: EntityId,                            // generated deterministically from the identity tuple below; also the map key in `sources`
    pub source_type: PhysicalSourceType,
    pub locator: String,
    pub version_ref: Option<PhysicalSourceVersionRef>,
    pub projected_schema: Vec<SourceColumn>,
    pub schema_fingerprint: Option<[u8; 32]>,
    pub provider_metadata: BTreeMap<String, String>,
}

#[non_exhaustive]
pub enum PhysicalSourceType {
    Table,
    File,
}

#[non_exhaustive]
pub enum PhysicalSourceVersionRef {
    IcebergSnapshotId(i64),
    MonotonicVersion(u64),
}

#[non_exhaustive]
pub struct SourceColumn {
    pub name: String,
    pub source_type: String,                     // native / engine-rendered string per C3.2
    pub nullable: bool,
}
```

### 8.2 Roster rules (C1)

- **Catalog-optional** (C1.1). When catalog is absent, model is the source of truth for physical shape.
- **Per identity unit** (C1.2). One `PhysicalSource` per `(table | unique file path / glob root)` plus optional version reference for invalidation.
- **Deterministic source identity.** A source's `id: EntityId` is generated deterministically from the identity tuple (`source_type`, `locator`, `version_ref`) — the same tuple that previously seeded `SourceId` — so the id is stable across runs and across model edits that don't change the source. `projected_schema` and metadata are attributes, not identity inputs. (Distinct from `semstrait-common::io::SourceId` in `31` — do not conflate.)
- **Referenced-only scope** (C1.3). Manifest is model-scoped — no catalog-wide scan; only sources reached by some binding are persisted.

### 8.3 Field set rules (C3)

- **Inline schema + fingerprint** (C3.1). `projected_schema` carries the binding-scoped column list used by mappings and `PhysicalExpr` validation; `schema_fingerprint` is a SHA-256 over its canonical encoding. Drift checks then skip column iteration on unchanged sources, and planner-cache keys can use the fingerprint directly.
- **`SourceColumn` shape** (C3.2). `name` + `source_type: String` (native engine-rendered string, e.g., `int4`, `INTEGER`) + `nullable: bool`. Canonical mapping is deferred to planner / engine registry; manifest preserves the engine string verbatim.
- **`PhysicalSourceType` is minimised** (C3.3). `Table | File` only. `ObjectStore` and `Stream` were dropped as out-of-scope for v1.
- **`PhysicalSourceVersionRef` is simplified** (C3.4). `IcebergSnapshotId(i64) | MonotonicVersion(u64)`. `ProviderToken` was dropped (its sole driver — Stream — is out of scope).
- **`locator` + `version_ref` carry source identity metadata.** Table/path identity lives in `locator`; snapshot/version identity lives in `version_ref` (for example Iceberg snapshot ids) when available.
- **Free-form provider metadata** (C3.5). `provider_metadata: BTreeMap<String, String>` carries provider-specific context not captured in typed fields (for example provider table aliases or storage-location hints) and is non-load-bearing.
- **`locator: String`** (C3.6). Provider-interpreted; semstrait does not parse it.

### 8.4 `schema_fingerprint` rules (C15)

- **Algorithm: SHA-256** (C15.1). Width-aligned with `manifest_epoch` / `model_hash` / `catalog_fingerprint` (all `[u8; 32]`).
- **Canonical encoding** (C15.2). Length-prefixed concatenation of `SourceColumn` fields, in declared order:

  ```text
  for col in projected_schema (declared order):
      hash.update(col.name.len() as u32 LE)
      hash.update(col.name.bytes)
      hash.update(col.source_type.len() as u32 LE)
      hash.update(col.source_type.bytes)
      hash.update(col.nullable as u8)
  ```

  Length prefixes prevent ambiguity (`"ab" + "c"` vs `"a" + "bc"`).
- **Preserve declared order** (C15.3). Reorder is a real drift signal; sorted-hash would silently mask reorder-only diffs. Callers wanting reorder-insensitivity sort `projected_schema` at the input layer.
- **Independent of `version_ref`** (C15.4). `schema_fingerprint` answers "did the schema shape change?"; `version_ref` answers "did the underlying data change?". Both feed independent cache keys.
- **`None` permitted when `projected_schema` is empty** (C15.5) — avoids a synthetic all-zeroes hash collision across empty-schema sources.
- **No fingerprint-algorithm-version tag** (C15.6). Per-source fingerprints feed into `manifest_epoch` / `model_hash` implicitly via C18.2 deterministic encoding; drift in any source's schema flips the canonical bytes and bumps the manifest hash.

### 8.5 Catalog metadata fetch rules (C14)

- **Eager-all when catalog provided** (C14.1). Compile walks every reachable `PhysicalSource` and fetches metadata. Required cascade from C3.1: an inline `projected_schema` claim demands metadata at compile time.
- **Hard error on missing source** (C14.2). If catalog is reachable but a specific source is *absent*, compile fails. Diagnostic: `COMP_E_CATALOG_SOURCE_MISSING { source_id, locator }` (placeholder code, finalised in spec 30 / spec 34).
- **Hard error on non-absence catalog fault** (C14.2a). A reachable catalog that fails to resolve a source for any reason *other than absence* — transport, authentication, malformed response, timeout, provider-internal error — is a distinct hard failure from C14.2, so a consumer can advise correctly ("fix credentials / retry" vs. "the table does not exist"). Diagnostic: `COMP_E_CATALOG_FETCH { source_id, locator, reason }` (placeholder code, finalised in spec 30 / spec 34); `reason` carries the provider's rendered message. The upstream `CatalogProviderErrorKind` is **not** structurally embedded (it derives neither `Eq`/`Hash` nor `std::error::Error`; structural nesting would strip `CompileError: PartialEq` and break the `Diagnose::cause` chain) — the provider message is carried as a flat `reason` string. This is the `From<CatalogProviderErrorKind>`-shaped mapping the catalog surface (spec 37 §3) anticipates, realised at the `compile` call site rather than as a blanket `From` impl (the `TableNotFound` vs. other split is compile-context-dependent, not a property of the provider error alone).
- **Hard error on multi-source Dataset partial fetch** (C14.3). Cascade from C14.2: an implicit Unionset (spec 23 §2.1 row A) over a multi-source Dataset assumes all branches are valid. Partial fetch = silently dropped branches, which is unacceptable.
- **Catalog-absent path** (C14.4). `catalog: Option<&dyn CatalogProvider>` parameter; when `None`, compile reads model's per-binding fields directly into `PhysicalSource` without any round-trip. `version_ref` is honoured if model provides it; otherwise `None`.
- **Mixed-mode override: model wins** (C14.5). When catalog is provided AND model declares overrides for specific sources, model wins. Diagnostic: `COMP_W_CATALOG_OVERRIDE { source_id, catalog_value, model_value }` (warning, not error).
- **`catalog_fingerprint` is fetched-content** (C14.6). Identity-only hashing would miss provider-internal drift. `None` when catalog is absent.

## 8A. `ResolvedRelationship` and `ResolvedJoinKey`

The manifest persists a compile-**resolved** relationship form, not the spec-18 authoring `Relationship`. The authoring struct (`18 §2.1`) carries `from` / `to` as `DataKindName`, `optional` / `cross_filter` as `Option` (default-derived at compile), join `keys` as `JoinKeyExprPair` (`UnresolvedSemanticExpr` sources), and **no** `JoinType`. Compile discharges all of that resolution; the manifest stores the result so the planner performs lookup only (I5 / I8).

### 8A.1 Shape

```rust
#[non_exhaustive]
pub struct ResolvedRelationship {
    pub id: EntityId,                    // model id (authored) — also the map key in `relationships`
    pub from: EntityId,                  // -> data_kinds (resolved from the authoring `DataKindName`)
    pub to:   EntityId,                  // -> data_kinds
    pub keys: Vec<ResolvedJoinKey>,      // equi-join pairs, semantic-id form (§8A.2)
    pub filter: Option<PhysicalExprId>,  // residual predicate; ANDed into the join condition; None = equi-only
    pub cardinality: Cardinality,        // IR planning enum (35; §8A.3) — the form PlanNode::Join consumes
    pub integrity: Integrity,            // model enum (18 §2)
    pub optional: Optional,              // resolved — defaults applied (spec 18 §2.7)
    pub cross_filter: CrossFilter,       // resolved — defaults applied (spec 18 §2.7)
    pub join_type: JoinType,             // derived at compile from `optional` (spec 18 §2.9)
    pub description: Option<String>,
}

#[non_exhaustive]
pub struct ResolvedJoinKey {
    pub from: EntityId,                  // -> semantics (Dimension | Field | Key) on the `from` interface
    pub to:   EntityId,                  // -> semantics on the `to` interface
}
```

`Cardinality` and `JoinType` are IR's planning enums (`35`) — the exact forms `PlanNode::Join` consumes (`§8A.3`), so `ResolvedRelationship` carries them without a conversion step. `Integrity`, `Optional`, `CrossFilter` are model enums (`18 §2`). The manifest re-exports all five and does not redefine them.

### 8A.2 Join keys are semantic-id pairs (SoC)

Each `ResolvedJoinKey` references a semantic on each side by `EntityId` — a `Dimension`, `Field`, or `Key` on the respective interface — **not** a `PhysicalExprId` directly. The physical expression for a side is reached through that semantic's bound `SemanticBinding.mapping[semantic_id]` (`§7.1`): a bare-name key resolves to an existing declared semantic; a computed key (e.g. `from: lower(email)`) is decomposed at compile into a synthesised implicit `Field` semantic (`role = Field`, generated id per §9.1) whose binding mapping holds the lowered `PhysicalExprId` (`SemanticMappingValue::Expr`). The decomposition is the same `auto`-mapping mechanism ratified in `19 §3.11`; it keeps the join contract expressed at the semantic layer while the physical form lives in the binding (separation of concerns) and the expression pool stays the single home for lowered `PhysicalExpr` (`§7.2`).

Consequence: a relationship key expression is **never** stored inline on the relationship. It is lowered through Phase A (`19 §3`), interned in `ManifestExpressions.physical`, and referenced indirectly via the semantic + binding. The residual `filter` is the one direct `PhysicalExprId` on the relationship (it is not a per-side semantic, but a whole-rowset predicate).

### 8A.3 Resolution responsibilities (compile)

- **Endpoints.** `from` / `to` `DataKindName`s resolve to the owning `DataKind`'s `EntityId`. A dangling endpoint is `validate.relationship-dangling-endpoint` (SR-E-5) at validate, and CX1 (`§10`) re-checks the id at load.
- **Defaults applied.** `optional` / `cross_filter` are materialised to concrete values via the spec 18 §2.7 matrix (no `Option` survives into the manifest). Symmetric-cardinality completeness (SR-E-13) and the many-to-many directional rule (SR-E-14) are already enforced at validate.
- **`join_type` derived.** From the resolved `optional` per the spec 18 §2.9 one-to-one table. `PlanNode::Join` (`35`) reads it; nothing re-derives it at plan time.
- **Scope.** Root-scope `ResolvedRelationship`s live in `SemanticManifest.relationships`; Joinset-local shadows live inline on `DataKindVariant::Joinset.scope_local_relationships` (`§6.7`, C7.6).

## 8B. `TemporalShape` and `TemporalType`

Compile propagates each leaf `Dataset`'s temporal classification onto its manifest `DataKind` (`§6.1` `temporal` field). The only resolution applied is the identifying-Dimension reference `SemanticsName → EntityId`; everything else is carried verbatim from the model (`18 §3`). It carries no `Resolved*` prefix because it is propagation, not full compile-resolution.

```rust
#[non_exhaustive]
pub struct TemporalShape {
    pub temporal_type: TemporalType,    // the shape variant; identifying dims resolved to EntityId
    pub grain: Option<Grain>,           // generic across all shapes; leaf-effective (spec 18 §3.3)
}

#[non_exhaustive]
pub enum TemporalType {
    Timeseries { occurred_at: EntityId },        // -> semantics (temporal Dimension)
    Events     { event_time:  EntityId },
    Snapshot   { snapshotted_at: EntityId },
    Scd        { scd_type: ScdType, valid_from: EntityId, valid_to: EntityId },
}
```

`ScdType` (v1 roster `{Type1, Type2}`) is a model enum (`18 §3`); `Grain` is the single canonical IR type (`35` home; roster ratified `13 §3.1`). The manifest re-exports both and does not redefine them.

Rules:

- **Leaf-only on disk.** `temporal` is `Some` only on leaf `Dataset`s — whether a top-level `DataKind` (`§6.1`) or a `NestedDataKind` `Dataset` variant (`§6.3`, e.g. a Grainset child carrying its own `grain` per spec 17 §3.4). A complex kind (`Unionset` / `Grainset` / `Joinset`, top-level or nested) carries `None`; its effective shape is derived by `SemanticGraph` at build per spec 17 §3.2 (shape cascade + composed-surface derivation are graph-build concerns, like JOIN-trees and `ComposedSemanticInterface`).
- **`grain` is generic, not per-variant.** It applies to every `TemporalType` and is leaf-effective (required on a leaf with `temporal:` authored — SR-E-6; forbidden on complex kinds — SR-E-7). Mirrors the model carrier (`18 §3.1`).
- **Identifying dims are `EntityId`.** Each variant's time-axis Dimension reference is resolved to its semantic `EntityId` (a registry entry in `semantics`), consistent with the single-id lane and matching spec 17 §1.5. CX1 (`§10`) checks each against `semantics.entries`.
- **Planner support deferred.** Shape-aware planning (rollup gates, snapshot selection, as-of joins) is deferred per spec 17 §10; the manifest still records the shape so the planner-complete contract (I8) is not violated and future shape-aware behaviour has its inputs.

## 13. Deferred Threads and Open Considerations

The following items were carried out of ratification and remain open for downstream resolution. They do **not** block the current shape.

| Thread | Where it surfaces | Notes |
|---|---|---|
| Thread A — model-as-truth posture | C1.1 cascade | Foundations / model spec edits — model authoring surface needs explicit per-binding fields for `locator`, `source_type`, `projected_schema`, optional `version_ref` when catalog is absent. Wave 2 cascade (likely spec 14 / 15). |
| Thread B — glob expansion semantics | C1.2 / C3.3 / C14 | Compile-time vs runtime expansion; whether `PhysicalSourceType::File` carries `glob_root`. Resurfaces at downstream catalog/versioning work. |
| File-payload refinement of `PhysicalSourceType::File` | C3.3 | Whether `File` carries `glob_root: Option<String>`. Deferred. |
| G5 workflow-friction watch | C13 | Strict orphan-detection-as-error may produce friction in iterative authoring. Revisit if reported. |
| Duplicate-authoring diagnostic | C18.5 | Compile-time content dedup permits canonical single `ExprId`. Whether duplicate authoring is itself a diagnostic surface = downstream call. |
| G6 — Relationship-graph cycle detection | C9 / C10 | C13 G1 scopes to `UnresolvedSemanticExpr` cycles only; cycle detection across the Relationship graph is a separate gate. |
| Per-Request cap override for `MAX_IMPLICIT_COMPOSITION_DEPTH` | C10 | Post-v1 question. |
| Request-shaped pruning of `JoinsetHop.hop_coverage` | C7 | Today coverage is request-independent (compile-time first-cut). Whether request-shaped pruning ever enters the manifest layer = downstream review. |

## 14. Cascades for Wave 2 (cross-spec edits implied by this rewrite)

The following spec amendments are **implied** by the closed clauses but live in other documents and must be applied in Wave 2:

- **Spec 14 / 15** — Add author-side per-binding fields (`locator`, `source_type`, `projected_schema`, optional `version_ref`) for the model-as-truth (catalog-absent) path (Thread A).
- **Spec 16 §9.1** — Confirm `MAX_IMPLICIT_COMPOSITION_DEPTH` wording does not imply persistence.
- **Spec 16 §10.4** — Align "compile-time-enumerated compositions" framing with the lightweight posture (index lives in `SemanticGraph`, not manifest).
- **Spec 18** — The manifest persists a compile-resolved `ResolvedRelationship` (§8A), **not** the authoring `Relationship`; it is stored both at root scope (`SemanticManifest.relationships`) and inline as Joinset-scoped shadow (`Joinset.scope_local_relationships`), with no `scope` discriminator. Re-point `18 §2.5` / `§2.9` / `§2.11`'s "manifest-layer `ResolvedRelationship` (`33 §8.1`)" references to `33 §8A` (the prior `§8.1` anchor was `PhysicalSource`). The derived `JoinType` and resolved `optional` / `cross_filter` live on `ResolvedRelationship`, not on the authoring struct. **(Applied in the relationship/temporal reconciliation pass.)**
- **Spec 17 / 18 §3** — The manifest carries temporal as `TemporalShape` / `TemporalType` on `DataKind.temporal` (§8B), not as a `ResolvedTemporalShape` on a `ResolvedDataKind`. Re-point `17 §1.5` / `§13` / `refined-by` and `18 §3.6` / frontmatter `refined-by` to `33 §8B`. Identifying-Dimension refs are resolved to `EntityId`; `grain` is the generic field. **(Applied in this pass.)**
- **Spec 20** — Reflect bitmask-coverage layer cross-references; CCK skeleton (universal `coverage`); Public/Nested split alignment with `DataKind` / `NestedDataKind`.
- **Spec 21** — Clarify multi-source Dataset auto-synthesis surfaces as a top-level implicit `Unionset` (with `origin = Implicit`), referenced by id from `RoutingUnitRef::Synthesized` when applicable.
- **Spec 22 §1.3 I8** — Drop "JOIN-tree shape, per-pair JOIN-key index, ComposedSemanticInterface" from `ResolvedGrainset`'s manifest contract (C8.2 cascade).
- **Spec 23 §2.1 row A** — Confirm implicit-Unionset top-level placement with deterministic content-derived `id` (`EntityId`).
- **Spec 24 §1.4 I8 / §2.4** — Drop "resolved ComposedSemanticInterface" from `ResolvedJoinset`'s manifest contract; retire the `interface: ComposedSemanticInterface { … }` pseudo-shape (C7.4 cascade).
- **Spec 35** — Add `impl Serialize` / `impl Deserialize` on `Expr<L>` and the leaf families with deterministic ordering (C18.3); manifest depends on these derives.

These cascades are tracked here for navigability; the actual edits occur in their owning specs.

## Testing Strategy

- SemanticBinding leaf-only linkage and one-to-many validation; mapping vs resolved keyset coincidence
- SemanticMappingValue round-trip all variants (Column, Literal, Expr, MetadataRef)
- ManifestExpression layer consistency: each entry's layer equals classify(expr)
- PhysicalExprId content-dedup: identical trees yield identical ids
- C18 deterministic encoding: same Expr value produces identical bytes
- PhysicalSource deterministic id from (source_type, locator, version_ref) tuple
- C15 schema_fingerprint canonical SHA-256 encoding (order-preserved, length-prefixed)
- ResolvedRelationship compile-resolved form: endpoints EntityId, defaults applied, join_type derived per spec 18 §2.9
- ResolvedJoinKey semantic-id-pair: from/to EntityId with role in {Dimension, Field, Key}
- TemporalShape leaf-effective propagation: identifying-Dimension EntityId resolution, leaf-only Some, complex-kind None

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
