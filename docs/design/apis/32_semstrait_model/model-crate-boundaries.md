---
id: 32-model-crate-boundaries
status: ratified
version: 2.5
impl: implemented
---

# Structural Rules, I4, Crate Boundaries, and Pointers

This leaf consolidates the structural rules (SR-*), deterministic ordering guarantees (I4), `${VAR}` substitution, crate boundaries for `semstrait-model`, feature flags, and pointers to child docs.

## 1. Structural Rules (SR-*)

Root-level invariants enforced at `parse` and the `validate` stage. Each rule maps to a typed-kind variant in `ParseErrorKind` or `ValidateError` per `30 §5`.

<!-- cross-cut: none -->

| ID        | Rule                                                                                                                                                                                                                                                                                                                                                                                                          | Kind                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **SR-1**  | Exactly one `semantic_model:` root key; `deny_unknown_fields` at root.                                                                                                                                                                                                                                                                                                                                        | `ParseErrorKind::UnknownField`                           |
| **SR-2**  | Nested data kinds MUST NOT carry `description`, `ai_context`, `dimensions`, `measures`, `metrics`, `keys`, `filters`. Enforced at the type level: `Nested`* structs wrap only a `*Body` — they have no `description`, `ai_context`, or `semantic_interface` fields — and implement `NestedDataKind` as the behavioral marker; `deny_unknown_fields` then rejects the Public-only tags at parse. | `ParseErrorKind::UnknownField`                           |
| **SR-3**  | Names are globally unique across the four top-level data-kind maps. Fired at `SemanticModelBuilder::build`.                                                                                                                                                                                                                                     | `ValidateError::DuplicateDataKindName`               |
| **SR-4**  | Same-variant self-nesting is forbidden: no grainset inside a grainset, no unionset inside a unionset, no joinset inside a joinset. Dataset leaves do not nest. Enforced at the type level by each `*Body` struct's child-field set; `deny_unknown_fields` then rejects same-variant tags at parse.                                                                                                     | `ParseErrorKind::UnknownField`                           |
| **SR-5**  | `catalog`, `storage`, `semantic_mapping` are leaf-only — they live on `LeafExtras` and have no slot in `ComplexExtras`. Authoring any of them under a Complex variant's `extras:` block is a parse error. Cascade-from-ancestor does not apply to these fields (R-6).                                                                  | `ParseErrorKind::UnknownField`                           |
| **SR-6**  | _Retired._ Per-variant required-extras rules live in `21 §7` / `22` / `23` / `24` (each chapter owns its own `VALID_E_2[1-24]xx` code band); the Grainset-child-temporal case is covered end-to-end by SR-E-8 (`18 §11`).                                                                                                                                                                                      | —                                                        |
| **SR-7**  | `deny_unknown_fields` is applied at every struct parse site (model root, data-kind blocks, extras, relationships, semantic elements).                                                                                                                                                                                                                                                                         | `ParseErrorKind::UnknownField`                           |
| **SR-8**  | Identifier rules: data-kind names and semantic-element names follow `11 §13`.                                                                                                                                                                                                                                                                                                                                 | `ParseErrorKind::InvalidIdentifier`                      |
| **SR-9**  | `${VAR}` substitution is applied before YAML decoding; unset variables are fatal parse errors (§8).                                                                                                                                                                                                                                                                                                           | `ParseErrorKind::UnsetEnvVar`                            |
| **SR-10** | Every `ComplexDataKind` (`Grainset` / `Unionset` / `Joinset`, Public or Nested) MUST have at least **2 children** across its allowed child-variant arrays. A composer with 0 or 1 children is degenerate — 0 collapses to nothing; 1 collapses to the single child's interface and should be authored as that child directly. Enforced at `validate`.                                                         | `ValidateError::ComplexDataKindInsufficientChildren` |
| **SR-11** | An `id`, when authored, MUST be **canonical UUID text**: exactly 36 bytes, `8-4-4-4-12` lowercase-hex layout (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`), re-parsing as a UUID; the nil UUID is rejected. The **version nibble is unconstrained** — it records provenance (authored v7, derived v5), it is not validated. Same rule for named entities and the model root. Full contract and rationale: `31 §2.1`. | `ParseErrorKind::InvalidEntityId`                        |
| **SR-12** | Named-entity `id` values are globally unique across the model (`data kinds + semantics + relationships`). Duplicate ids are rejected at build/validate boundary. The model-root `id` is not a collection member and is outside this scope (`32 §1.4a`).                                                                                                                                                                                                                                             | `ValidateError::DuplicateEntityId`                       |

SR-* numbering is append-only. Adding a rule is a MINOR change per `30 §2`.

Entity-level invariants (`SR-E-`*) — reference-site overrides, Semantics orphan policy, relationship cardinality, TemporalShape grain placement, filter-kind disjointness — live at `18 §11`. SR-E-* is a separate, independently-numbered series that extends the root-level SR-* roster.

## 2. Deterministic Ordering (I4)

Given the same input YAML plus the same environment, `parse` produces a byte-identical serialized `SemanticModel` under any canonical serializer.

Every entity collection is keyed by `EntityId`, but the `EntityId` is **never** the ordering axis for output. Iteration helpers and serde serialization project each map into a **name-ordered** sequence, so deterministic output does not depend on id values. This keeps I4 byte-stable even under the convenience profile, where generated **entity** ids vary across fresh parses. The model-root `id` is the one exception to that variance: it is derived deterministically, not minted from the clock (`32 §1.4a`), so it is byte-stable across parses and its serialized value does not perturb I4.

This is a load-bearing requirement on the `EntityBTreeMap<T>` wire projection, not an incidental property: the `into = "Vec<T>"` arm MUST sort the values **by name** before emitting the array. A naïve `BTreeMap::into_values()` would emit in **id order** (the map's key order) — under the convenience profile those ids are fresh per parse, so id-order output is non-deterministic and violates I4. The projection therefore sorts by `name`, not by key.

<!-- cross-cut: none -->

| Collection                                                  | Type                             | Ordering rule (iteration / serialization)                                                            |
| ----------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `SemanticModel.datasets`                                    | `BTreeMap<EntityId, Dataset>`    | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.grainsets`                                   | `BTreeMap<EntityId, Grainset>`   | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.unionsets`                                   | `BTreeMap<EntityId, Unionset>`   | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.joinsets`                                    | `BTreeMap<EntityId, Joinset>`    | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.dimensions`                                  | `BTreeMap<EntityId, Dimension>`  | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.measures`                                    | `BTreeMap<EntityId, Measure>`    | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.metrics`                                     | `BTreeMap<EntityId, Metric>`     | Alphabetical by name (name-ordered projection)                                                       |
| `SemanticModel.labels`                                      | `Vec<String>`                    | YAML author order                                                                                    |
| `SemanticModel.relationships`                               | `BTreeMap<EntityId, Relationship>` | Alphabetical by name (name-ordered projection); first-match-wins now resolves in name order — see note below and `16 §11` |
| `GrainsetBody.{datasets, unionsets, joinsets}`              | `BTreeMap<EntityId, Nested*>`    | Alphabetical by name (name-ordered projection)                                                       |
| `UnionsetBody.{datasets, grainsets, joinsets}`              | `BTreeMap<EntityId, Nested*>`    | Alphabetical by name (name-ordered projection)                                                       |
| `JoinsetBody.{datasets, grainsets, unionsets}`              | `BTreeMap<EntityId, Nested*>`    | Alphabetical by name (name-ordered projection)                                                       |
| `JoinsetBody.relationships`                                 | `BTreeMap<EntityId, Relationship>` | Alphabetical by name (name-ordered projection)                                                     |
| `iter_all` / `iter_public` / `iter_simple` / `iter_complex` | iterators                        | Alphabetical by `(variant-tag, name)`; variants in fixed order: Dataset, Grainset, Unionset, Joinset |

**Relationship ordering change.** Relationships were previously a `Vec` carrying YAML author order, and first-match-wins (`16 §11`) consumed that author order. With relationships now stored in an `EntityId`-keyed map projected in **name order**, first-match-wins resolution and cross-kind BFS neighbor iteration (`19 §3.4.3`) proceed in name order. Relationships are identified by `EntityId` (no `RelationshipId` newtype; `18 §2.1`). This is a deliberate consequence of the id-first rework (`STATUS.md` items U / U.2).

Identity determinism by profile:

- **Strict deterministic** — all named entities carry authored ids; repeated parses of identical source produce byte-identical models *including* id values.
- **Convenience** — missing ids are auto-generated as UUIDv7; name-ordered output stays deterministic, but the embedded id values vary across fresh parses unless persisted and reused from manifest (`33 §4.3.1`).

`HashMap` is banned from the entire public surface. A CI check fails on any public `HashMap<_, _>` reachable from `SemanticModel`.

## 3. `${VAR}` Substitution

Before YAML decoding, `parse` rewrites every `${IDENT}` token in the input string to the value of the environment variable `IDENT` (via `std::env::var`). Unset variables raise `ParseErrorKind::UnsetEnvVar`.

```yaml
semantic_model:
  name: analytics-${ENVIRONMENT}
  datasets:
    - name: orders
      extras:
        storage:
          paths: ["s3://${DATA_BUCKET}/orders/*.parquet"]
```

Syntax: only `${IDENT}` is recognized. Bare `$VAR` is treated as literal text. Identifier grammar follows standard Unix env-var rules (ASCII letters, digits, underscores; no leading digit).

## 4. Crate Boundaries

`semstrait-model` is the thinnest authoring-surface crate. It sits **directly above** `semstrait-common` in the workspace DAG (I7) — the public authoring surface is rank 1, depending only on `semstrait-common`. `semstrait-ir` sits *above* this crate and imports its authoring vocabulary; the prior `model → ir` edge is **inverted** to `ir → model` per `STATUS.md` item Y (which supersedes item Q):

```
semstrait-common    (leaf: Tree/Visitor/Rewriter + tree ValidateError + Diagnostic<K> / Diagnose / EntityId / constraints / io)
    ↑
semstrait-model     (DataType / Grain + shared Expr<L> incl. Window + leaves + accessors
                     + BinaryOpKind/…/Literal + WindowFn/WindowFrame/WindowFrameKind/WindowBound (A2)
                     + ColumnRef/SemanticsName + CanonicalFn-name + field-level authoring serde/schemars
                     + parse + validate + SemanticModel + ParseErrorKind + validate-stage ValidateError)
    ↑
semstrait-ir        (resolved PhysicalExpr = Expr<PhysicalLeaf> alias (A2) + PhysicalLeaf + Parameter
                     + TypeClass/Schema/SchemaColumn + FunctionRegistry/Additivity + CompileError + PlanNode)
    ↑
semstrait-manifest, semstrait-planner, semstrait-adapter, …
```

Dependencies: `semstrait-common`, `serde`, `serde_yaml`, `schemars` (V7), `tracing` (`30 §6.2`), `uuid` (v7 mint + SR-11 parse). No other `semstrait-*` crate — in particular **not** `semstrait-ir` (item Y / V1 inverts the edge). No `async`, no `arrow`, no engine-specific deps. Because this crate **owns** the shared `Expr<L>` family (item Y / V6, incl. the `Window` variant per A2), the authoring expression fields carry a bare `Expr<L>` directly with no cross-crate dependency — there is no parallel `ExprBlock` AST and no ir import (`14 §6.1`). The hand-rolled field-level authoring serde / `schemars` adapters (V7 as refined by A2; mechanism per doc 45 — moved off the deleted `ExprSource<L>` wrapper onto the fields) live here so internal tags never leak (`binary_op:` / `type:` / `content:` forbidden in authored YAML and its schema); the shared `Expr<L>`'s own derived machine serde is the cross-process transport form `ir::PhysicalExpr` aliases (distinct surface, reached only by the machine path, never the author path).

**Single schema + serde root.** `schemars` / `JsonSchema` codegen roots at `SemanticModel`: the JSON Schema a side application emits for the `semantic_model:` file is generated from `SemanticModel` itself, so the authoring contract is the schema. No `pub(crate)` decode intermediate derives `JsonSchema`, and no second public model type exists. Two categories of **hand-rolled serde are deliberate, not drift**: (a) the field-level expression adapters above (friendly tags; V7 / item Y; the friendly schema projects to a single shared `Expression` `$def` that every expression field `$ref`s); and (b) the per-enum hand-written serde on `DimensionType` / `AdditivityType` / `BucketBound` / `MetadataSource` / `DataType` (the `serde_yaml`-#370 externally-tagged-enum workaround). Both are author-surface concerns the derive cannot express; the `EntityBTreeMap<T>` wire projection is the *only* container-level custom serde and is itself derive-driven (`try_from`/`into` over `Vec<T>`), not hand-maintained JSON.

### 4.1 No direct I/O in `parse`

`parse` takes a `&str` and is synchronous. It performs no file opens and no network calls. `std::env::var` is the single allowlisted syscall (used for `${VAR}` substitution per §3).

Two syscall classes are barred from the whole crate, not just `parse`: **no writes** and **no network**. `std::net` / `std::process` never appear, and nothing here writes bytes anywhere — rendering a model to bytes is `Persistable::to_bytes` and storing them is `Repository<SemanticModel>` (`31b §2` / `§4`), both reached by the caller. Enforcement today is review plus the greps in the Testing Strategy below; the boundary is not yet a CI gate (`[TD-BOUNDARY-CI]`).

The one permitted syscall is `LocalFs`'s `std::fs::read` (`32-model-loader §1`) — the local read behind the authoring loader, confined to that single strategy impl. It is scheduled for removal in favour of the `FileSystem` primitive (`[TD-DELETE-SOURCEFS]`), after which the crate contains no filesystem call at all. Remote reads never enter: object-store schemes are fetched by the caller and handed over as bytes (`32-model-loader §4`), which is why this crate depends on no async runtime.

### 4.2 No resolution

Name resolution, reference expansion, and cross-kind path resolution are `compile`'s responsibility (`33`, `11 §7`). `parse` records identifiers verbatim. Named-entity `EntityId` values are carried inline onto the corresponding manifest entities (`33 §4.3.1`) — they are storage keys, not semantic-name resolution keys, and there is no side-map. Cross-references inside the model (`ref:`, relationship `from`/`to`) are by name and resolved at compile.

### 4.3 No planning

`Request`, `PlanNode`, `SemanticPlan`, and `SemanticManifest` types never appear in `semstrait-model`.

## 5. Authored logical-type vocabulary — `DataType` / `Grain`

`DataType` (the 14 authorable scalar variants plus the compile-transient `Unknown` sentinel) and `Grain` (7 temporal levels) are **owned by this crate** (item Y / V2). Their variant rosters and structural rules are ratified by `13` (`13 §2.1` / `13 §3.1`); `semstrait-model` is the crate-level home and carries the surface below. `semstrait-ir` imports both for resolved-IR use (`35 §4`). The non-authored `TypeClass` / `Schema` / `SchemaColumn` stay in `semstrait-ir` (`35 §4`) — they describe resolved physical shape, never authored YAML.

**`data_type:` string parsing.** A single public associated function is the workspace-wide home for turning a `data_type:` scalar string into a canonical `DataType`:

```rust
impl DataType {
    /// Parse a `data_type:` scalar into the canonical variant, accepting
    /// the full case-insensitive alias roster of `13 §2.2`. `None` for
    /// unknown strings (the caller maps that to its own diagnostic).
    pub fn parse_scalar(s: &str) -> Option<DataType>;
}
```

It normalizes case (`13 §2.2` mandates case-insensitive aliases) and resolves both bare aliases (`bigint` → `Long`) and parametrized forms (`decimal(p,s)`, `time(p)`, `timestamp(p)`, plus bare-defaults `decimal`→`(38,9)`, `time`→`(6)`, `timestamp`→`(0)`). The `Deserialize` impl and the declarative `cast.target:` parser both delegate here — there is **no** second alias table anywhere in the workspace (DRY; the roster is ratified once in `13 §2.2`). `parse_scalar` returns `Option` with no diagnostic-accumulation channel, so the `number` → `Double` legacy-alias *warning* mandated by `13 §2.2` is not emitted at this entry point (deferred — tracked in `13`).

Constructing `Decimal { precision, scale }` with out-of-range values is rejected at parse (`ParseErrorKind::InvalidDecimalParameters`; precision `1..=38`, scale `0..=precision` per `13 §2.1`); the constructor itself performs no validation.

**`Grain` coarseness.** The total coarseness order of `13 §3.2` is exposed as an accessor:

```rust
impl Grain {
    /// Selection-rank order: Minute(0) < … < Year(6). Per `13 §3.2`.
    pub fn coarseness(self) -> u8;
}
```

**Serialization.** `DataType` and `Grain` serde / `schemars` are feature-gated per §6 (`serde` on by default). `DataType` carries **hand-written** `Serialize` / `Deserialize` (not derived): the `Unknown` sentinel is refused in both directions — `Serialize` errors on it, symmetric with `Deserialize` / `parse_scalar` returning `None` for the token `unknown`. The sentinel is compile-transient (`19 §3.6`–`§3.7`): it tags a not-yet-known type during inference and is reconciled away before the manifest seals; refusing it at *write* turns a failure-to-reconcile into a loud, earliest-detection error rather than a value only a later `Deserialize` would reject (no hidden behavior, I12). No persisted expression ever carries it. The `schemars` schema mirrors the refusal — it lists the 14 authorable variants, never `unknown`. `Grain` derives both serde and `JsonSchema` normally.

## 6. Feature flags

<!-- cross-cut: none -->

| Feature    | Gates                                                                            | Default | Forwards |
| ---------- | -------------------------------------------------------------------------------- | ------- | -------- |
| `schemars` | `JsonSchema` on the authoring graph rooted at `YamlRoot`, plus the `schema` module | OFF     | —        |

`schemars` is the crate's only feature. `Serialize` / `Deserialize` are unconditional — the authoring serde *is* the crate's contract (§3), so gating it would gate the surface itself.

**No transport feature (I11).** There is no `io` / `io-aws` axis here, and none is coming. Bytes arrive already-read through `from_yaml_bytes` (`32-model-loader §4`) and leave through `Persistable::to_bytes` (`31b §2`), so no feature can add a runtime to this crate — the property §4.1 states is the one a flag would quietly undo.

## 7. Pointers to Child Docs

<!-- cross-cut: none -->

| Scope                    | Doc                                                                                    | What lives there                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Canonical entities**   | [`../../foundations/18_entities.md`](../../foundations/18_entities.md)                       | `Relationship` (identified by `id: EntityId`), `Cardinality`, `Integrity`, `Optional`, `CrossFilter`, derived `JoinType`, `TemporalShape`, `ScdType`, `Dimension` / `Measure` / `Metric`, `DimensionType` + body structs, `Additivity`, filter taxonomy, `AiContext`, `Keys`, `SemanticMappingValue` shape, root-pool reference / override grammar, `SR-E-`* entity-level rules. Authoritative for every entity struct shape embedded in 32. |
| Dataset interior         | [`../../data-kinds/21_dataset.md`](../../data-kinds/21_dataset.md)                           | Per-Dataset YAML: `dimensions:`, `measures:`, `metrics:`, `filters:`, `keys:`, leaf-only `extras` semantics                                                                                                                                                                                                                                                                                           |
| Grainset interior        | [`../../data-kinds/22_grainset.md`](../../data-kinds/22_grainset.md)                         | Per-Grainset YAML: child composition, grain-axis, `temporal:` in extras                                                                                                                                                                                                                                                                                                                               |
| Unionset interior        | [`../../data-kinds/23_unionset.md`](../../data-kinds/23_unionset.md)                         | Per-Unionset YAML: children, `mode:`, coverage                                                                                                                                                                                                                                                                                                                                                        |
| Joinset interior         | [`../../data-kinds/24_joinset.md`](../../data-kinds/24_joinset.md)                           | Per-Joinset YAML: members, `relationships:` (join graph), anchor                                                                                                                                                                                                                                                                                                                                      |
| Nesting matrix           | [`../../data-kinds/26_nesting_matrix.md`](../../data-kinds/26_nesting_matrix.md)             | Which parent variant contains which nested variants; `SR-10` + Grainset-child grain rule                                                                                                                                                                                                                                                                                                              |
| Applicability            | [`../../data-kinds/25_applicability_matrix.md`](../../data-kinds/25_applicability_matrix.md) | Per-variant × foundation-rule cross-cuts                                                                                                                                                                                                                                                                                                                                                              |
| Semantic mapping grammar | [`../../foundations/15_mapping_and_binding.md`](../../foundations/15_mapping_and_binding.md) | `SemanticMapping` values in detail; the `Binding` process                                                                                                                                                                                                                                                                                                                                             |
| Relationships (planner)  | [`../../foundations/16_composition.md`](../../foundations/16_composition.md)                 | Composition graph, implicit Joinset synthesis                                                                                                                                                                                                                                                                                                                                                         |
| Temporal shape (planner) | [`../../foundations/17_temporal_shape.md`](../../foundations/17_temporal_shape.md)           | Planner-level variant semantics, rollup matrix                                                                                                                                                                                                                                                                                                                                                        |
| Catalogs file            | [`../32b_catalogs_yaml.md`](../32b_catalogs_yaml.md)                                     | `catalogs.yaml` grammar; `CatalogRef` reference syntax                                                                                                                                                                                                                                                                                                                                                |
| SemanticManifest         | [`../33_semstrait_manifest.md`](../33_semstrait_manifest.md)                             | How the `SemanticModel` tree lowers to a `SemanticManifest`                                                                                                                                                                                                                                                                                                                                           |
| Core I/O transport       | [`../31b_semstrait_common_io.md`](../31b_semstrait_common_io.md)                             | `FileSystem` byte transport, `Persistable` / `Repository<T>`, and `IoErrorKind` — the seam a caller composes to hand this crate bytes (`32-model-loader §4`)                                                                                                                                                                                                                                                                                                                                    |

## Testing Strategy

- **SR-* rules**: every structural rule is exercised by either negative-parse fixtures (SR-1/2/4/5/7/8/9/11) or validate-stage tests (SR-3/10/12) as documented in the model-parse-validate leaf.
- **I4 deterministic ordering**: round-trip tests verify byte-identical output on repeated serialize → parse cycles under both identity profiles (name order stable; id values vary under convenience profile but don't affect ordering).
- **${VAR} substitution**: parse correctly expands `${IDENT}` tokens to env-var values and raises `ParseErrorKind::UnsetEnvVar` for unset variables — verified by substitution fixtures.
- **Crate boundaries** (§4.1): `grep -rn 'fs::write\|std::net\|std::process' crates/semstrait-model/src/` is empty and `std::fs` appears only in `source_fs.rs`; parse remains synchronous and I/O-free (except `std::env::var`). Not yet CI-enforced — `[TD-BOUNDARY-CI]`.
- **DataType / Grain vocabulary**: `DataType::parse_scalar` correctly parses all `13 §2.2` aliases (case-insensitive, bare + parametrized forms); `Grain::coarseness` returns correct total order per `13 §3.2`; `DataType::Unknown` is refused in both serde directions — verified by type-vocabulary test suite.
- **Feature flags** (§6): `--no-default-features` and `--features schemars` both build and test green; under either, this crate's own `Cargo.toml` declares no async dependency and its source has no `async fn` / `.await` — the check that fails if transport is ever gated back in. Not a `cargo tree` check: `tokio` arrives transitively through `semstrait-common`'s default-on `io` feature regardless of model's own flags (`[TD-COMMON-IO-DEFAULT]`).

Shared harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
