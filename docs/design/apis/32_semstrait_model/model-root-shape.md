---
id: 32-model-root-shape
status: ratified
version: 2.5
impl: implemented
---

# Model Root YAML Shape and SemanticModel Type

This leaf defines the `semantic_model:` root YAML envelope and the in-memory `SemanticModel` type that materializes from it.

## 1. Root YAML Shape

A model YAML file has exactly one root key: `semantic_model:`. No other root-level key is recognized (SR-1).

```yaml
semantic_model:
  name: analytics-v1
  description: "Primary analytics model for the order pipeline."
  ai_context:
    synonyms: { order: [purchase, txn] }
  labels: [analytics, prod]

  # ── Data kinds — four per-variant plural arrays ─────────────────
  datasets:   [ ... ]                  # see 21_dataset.md
  grainsets:  [ ... ]                  # see 22_grainset.md
  unionsets:  [ ... ]                  # see 23_unionset.md
  joinsets:   [ ... ]                  # see 24_joinset.md

  # ── Shared Semantics pools — one map per carrier ────────────────
  # Full entity shapes + ref/override grammar — 18 §4–§9.
  dimensions: [ ... ]                  # see 18 §4
  measures:   [ ... ]                  # see 18 §6
  metrics:    [ ... ]                  # see 18 §7

  # ── Cross-entity relationships — see 18 §2 for full grammar ────
  relationships:
    - name: orders_to_customers
      from: orders
      to:   customers
      keys: [{ from: customer_id, to: id }]
      cardinality: many_to_one          # required
      integrity: assumed                # default; alternatives: enforced | none
      optional: none                    # default per 18 §2.7; required on 1:1 / m:m
      cross_filter: left                # default per 18 §2.7; required on 1:1 / m:m
```

Every child block is optional except `name:`. An empty model — `semantic_model: { name: ... }` — parses successfully. A non-empty model with zero data kinds is a `ValidateError::EmptyModel` at the validate stage.

### 1.1 Per-variant plural arrays

Each data-kind variant is authored under its own plural tag: `datasets:`, `grainsets:`, `unionsets:`, `joinsets:`. There is no unified parent block and no `kind:` discriminator on individual entries.

### 1.2 Top-level vs nested data kinds

Only top-level data kinds — those authored directly under one of the four plural arrays — expose a `SemanticInterface` (dimensions / measures / metrics / keys / filters). Nested data kinds (authored inside a complex parent's child arrays per `26`) are structural shells: they carry `name`, `description`, `extras`, and variant-specific structural fields only. They do not author their own Semantics.

### 1.3 Catalog reference

Physical-source catalogs are authored in a sibling file, `catalogs.yaml`, per `32b`. A model references a catalog via the `catalog:` key inside an `extras:` block. The reference is a **bare alias string** — there is no namespace override or map form:

```yaml
extras:
  catalog: polaris_prod
```

`CatalogRef` is a transparent newtype around the alias; resolution against `catalogs.yaml` happens at compile. Full grammar lives in `32b §4`.

### 1.4 Optional named-entity `id`

Model authoring may include an optional `id` field on named-entity blocks:

- **the model root itself** — deterministic, distinct rules (§1.4a),
- top-level and nested DataKinds,
- shared Semantics entries (`dimensions`, `measures`, `metrics`),
- `relationships` entries.

`id` is a string and, when present, MUST be canonical UUID text — 36 bytes, `8-4-4-4-12` lowercase hex, non-nil (SR-11; full contract at `31 §2.1`). The version nibble is not validated: parse mints v7 for entities and derives v5 for the root (§1.4a), and both satisfy the same rule. When omitted, missing-id behavior is profile-controlled (see model-parse-validate leaf).

```yaml
semantic_model:
  id: "0197cb6e-a63a-7d53-7c11-9b2ae4c0f3a1"
  name: analytics-v1
  datasets:
    - id: "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e"
      name: orders
      extras: { ... }
  relationships:
    - id: "0197cb6e-a63a-7d53-9e77-17f07341da5c"
      name: orders_to_customers
      from: orders
      to: customers
      keys: [{ from: customer_id, to: id }]
      cardinality: many_to_one
```

The `id` field is part of the public payload: it is an `Option<EntityId>` field on every named-entity struct (`DataKindBase` for data kinds; `Dimension` / `Measure` / `Metric` / `Relationship` / `Key` / `Filter` per `18`). `None` is the authoring-absent shape — there is no `""` sentinel. When omitted at authoring, parse mints a UUIDv7 under the convenience profile, so after parse the field is always `Some`. The minted-or-authored inner `EntityId` is the storage key for every model collection (§2).

### 1.4a Model-root `id`

The model root carries its own `id: Option<EntityId>`. It is the **persistence key** for the model as a whole — `Persistable::entity_id` for `SemanticModel` returns it (`31b §2`), so a model can be saved and reloaded by id rather than by file path.

**Root identity is deterministic.** Unlike entity ids, the root `id` is never time-based: parsing the same model twice — in either order, on either machine, before or after a content edit — yields the same root `id`. A time-based UUIDv7 would mint a fresh key per parse, so one model would persist under many ids and `load(id)` after a re-parse would miss. Resolution order:

1. **Authored `id` wins.** Present in the YAML ⇒ used verbatim. This is the strongest form: stable across content edits *and* renames.
2. **Absent ⇒ derived from the model `name`**, as UUIDv5 (name-based, namespace `semstrait model-root`). Deterministic by construction and stable across content edits, because `name` does not change when a measure does.

Derivation is **not** content-derived. A hash-of-content id would change on every edit — exactly the regeneration this rule forbids — and each save would write a new object while orphaning the previous one. The root `id` answers *which document*, not *which version of it*.

**The derived id self-persists.** `Option<EntityId>` is `Some` after parse and always serializes, so the first `to_bytes` (`31b §2`) writes the derived value into the model's own bytes. From then on it is an authored id under rule 1, and a later rename cannot move the key. No side-car state, no id registry, no write-back step is required — the model file is the store.

**Consequences, stated:**

- Two models sharing a `name` derive the same root `id` and would clobber one another in a repository. Model `name` is the global identity axis for un-authored models; author an explicit `id` to keep same-named models distinct.
- Renaming a model that has **no** authored `id` changes its derived id, orphaning the previously stored object. Renaming one that *has* an authored id is safe — which one save-cycle guarantees.
- `StrictRequireProvided` still requires an explicitly authored root `id` (`MissingEntityId` otherwise): strict demands provenance, it does not accept derivation.

**Format.** The root `id` satisfies the same SR-11 shape rule as every other id — canonical UUID text, version nibble unvalidated (`31 §2.1`) — so a derived v5 root id round-trips with no root-specific exemption. This is the property that makes the derivation strategy above swappable: changing it (to a content hash, a different namespace, or a different UUID version) alters no validation rule.

The root `id` is **not** an entry in any collection: it identifies the model, not an entity within it, so it is exempt from SR-12's global-uniqueness scan over `data kinds + semantics + relationships` (a root id equal to a contained entity's id is legal, if pointless).

**Versus the manifest's `model_id`.** Both are deterministic, and they are not interchangeable: `model_id` is UUIDv5 over `model_hash` (`33 §9.1`) and *must* change when content changes — it identifies a compilation result. The root `id` must *not* change when content changes — it identifies the authored document. Compile does not consume the root `id`, and the root `id` stays outside the canonical bytes `model_hash` covers: folding document identity into a content hash would make `model_hash` flip on a rename that changed no semantics.

Because root identity is stable across edits, "same id, new content" is a normal model edit — and the seam obliges: `save` is a `put` for every DTO (`31b §3`), so a re-saved model overwrites rather than being refused.

## 2. `SemanticModel` Root Type

```rust
/// Canonical entity identity — defined in `semstrait-common` (`31 §2.1`)
/// as the workspace-wide id lane and re-exported here for the authoring
/// surface. At the model layer it is authored optionally; parse generates
/// a UUIDv7 per missing named entity under the convenience profile and
/// derives the root id as UUIDv5 (§1.4a). The validated contract is
/// canonical-UUID-text *shape* (SR-11), not the derivation: the version
/// nibble is provenance, so authored, minted, and derived ids all pass
/// the one rule.
pub use semstrait_common::EntityId;   // = String, defined at 31 §2.1

#[non_exhaustive]
pub struct SemanticModel {
    /// Model-root identity — the persistence key (§1.4a). Optional at
    /// authoring; parse fills it deterministically (authored, else
    /// UUIDv5 over `name`) — never time-based.
    pub id: Option<EntityId>,
    pub name: String,
    pub description: Option<String>,
    pub ai_context: Option<AiContext>,
    pub labels: Vec<String>,

    // Data kinds — per-variant maps keyed by EntityId.
    pub datasets:  BTreeMap<EntityId, Dataset>,
    pub grainsets: BTreeMap<EntityId, Grainset>,
    pub unionsets: BTreeMap<EntityId, Unionset>,
    pub joinsets:  BTreeMap<EntityId, Joinset>,

    // Shared Semantics pools — keyed by EntityId.
    pub dimensions: BTreeMap<EntityId, Dimension>,
    pub measures:   BTreeMap<EntityId, Measure>,
    pub metrics:    BTreeMap<EntityId, Metric>,

    // Cross-entity relationships — keyed by EntityId.
    pub relationships: BTreeMap<EntityId, Relationship>,
}
```

All semantic-payload fields in the public roster are `pub` so consumers can destructure without getter boilerplate. Construction outside `parse` is permitted (test harnesses, tooling).

`SemanticModel` is the **single authoring type**: the sole `serde` (`Serialize` + `Deserialize`) surface, the sole `schemars` / `JsonSchema` root, and the type `parse` materializes into. There is no parallel public DTO — the JSON Schema is generated from `SemanticModel` itself, so the authoring contract and the schema cannot drift. The collections stay `BTreeMap<EntityId, _>` in memory (O(log n) id lookup, the id-keyed lifecycle/dedup lane, and the reserved `(EntityId, version)` seam), yet author and serialize as **arrays** (`datasets: [ {name}, … ]`): each collection is held behind a `EntityBTreeMap<T>` wrapper that projects map ↔ `Vec<T>` for the wire (`#[serde(try_from = "Vec<T>", into = "Vec<T>")]`) and carries a hand-written `JsonSchema` delegating to `Vec<T>` so the schema shows an array, not an object. Any decode-only intermediate is `pub(crate)` and never on the schema.

Every collection is keyed by the entity's own `id` (the `EntityId` carried inside each value). Keying by `id` rather than by name lets the model retain *every* authored entity — including two entries that share a name — so duplicate-by-name detection runs as an explicit per-layer scan at `.build()` instead of being silently masked by map-key collision. Name lookup is therefore a scan or a caller-built index, not a direct map hit (§2.2). Iteration and serialization are name-ordered so identity-keyed storage does not perturb deterministic output.

### 2.1 Global name uniqueness

Data-kind names are globally unique across the four top-level maps: a `Dataset` named `sales` and a `Grainset` named `sales` cannot both exist. Because the maps are keyed by `EntityId`, two same-named entities do *not* collide on insert — both are retained and the clash is surfaced by the global name scan at `SemanticModelBuilder::build`: `ValidateError::DuplicateDataKindName` (SR-3) — fired uniformly over single-source, code-built, and cross-source accumulations.

Shared pools use their own namespace per carrier: a Dimension named `region` and a Measure named `region` can coexist. Duplicate names within a carrier raise `ValidateError::DuplicateSharedSemanticsName` via the same `.build()` per-carrier name scan.

Nested data kinds use parent-scoped uniqueness (addressing becomes `grainsets[sales].unionsets[regional]`); see `26 §5`.

### 2.2 Cross-variant iteration helpers

For diagnostics and walks that treat top-level data kinds uniformly, `SemanticModel` exposes iterators that yield the view enums defined in the model-datakind-hierarchy leaf.

```rust
impl SemanticModel {
    pub fn iter_all(&self)     -> impl Iterator<Item = AnyDataKindRef<'_>>;
    pub fn iter_public(&self)  -> impl Iterator<Item = PublicDataKindRef<'_>>;
    pub fn iter_simple(&self)  -> impl Iterator<Item = SimpleDataKindRef<'_>>;
    pub fn iter_complex(&self) -> impl Iterator<Item = ComplexDataKindRef<'_>>;

    pub fn find_public(&self, name: &str) -> Option<PublicDataKindRef<'_>>;
}
```

`iter_all` walks only top-level public data kinds (it does not descend into bodies); nested data kinds are reached through `ComplexDataKind::children_ref` on each visited public composer. All view enums are defined in the model-datakind-hierarchy leaf.

### 2.3 Named-entity identity is an inline field

There is no separate identity sidecar. Each named entity carries its own `id: Option<EntityId>` (canonical UUID text when present — SR-11) directly on its struct — `DataKindBase` for data kinds, and `Dimension` / `Measure` / `Metric` / `Relationship` / `Key` / `Filter` per `18`. `None` is the authoring-absent shape; parse mints it to `Some`, after which the inner `EntityId` is also the storage key of the collection the entity lives in (§2), so identity, payload, and storage key are one and the same. The `Option` is the sole identity home — there is no `DataKindId` / `RelationshipId` / sidecar newtype (concern: id is a plain field tagged `id`).

The model root follows the same inline-field discipline: it carries its own `id: Option<EntityId>` (§1.4a) — the model's persistence key, not a collection member, and filled deterministically rather than from the clock. Compile (`33`) carries each entity's `id` straight onto the corresponding manifest entity — identity rides inline, there is no `stable_ids` side-map (`33 §4.3.1`) and no path-keyed index at the model layer. The root `id` is not carried into the manifest: the manifest's `model_id` is minted from `model_hash` (`33 §9.1`).

## Testing Strategy

- **SR-1** (root-key enforcement): parse rejects YAML with `deny_unknown_fields` — verified by negative-parse test suite against `ParseErrorKind::UnknownField`.
- **SR-3** (global name uniqueness): `ValidateError::DuplicateDataKindName` fires at `.build()` when two top-level data kinds share a name — verified by duplicate-name fixture tests.
- **SR-11** (id format): authored `id` values that don't match canonical UUID text — wrong length, uppercase hex, braced/URN form, missing or misplaced hyphens, non-hex characters, the nil UUID — trigger `ParseErrorKind::InvalidEntityId`, verified by negative-parse fixtures. Ids differing **only** in version nibble (v4, v5, v7) all parse clean at both entity and root position, verified by a positive fixture per version: the test that fails if a version check is reintroduced anywhere (`31 §2.1`).
- **SR-12** (id uniqueness): `ValidateError::DuplicateEntityId` fires when two entities share an `id` — verified by id-collision test. A root `id` colliding with a contained entity's `id` does **not** fire it (§1.4a) — verified by a same-id root fixture that builds clean.
- **Root determinism** (§1.4a): parsing the same id-less model twice yields the **same** root `id` — verified by an equality assertion across two independent parses (the property a time-based mint would fail). Two models differing only in content derive the same root `id`; two differing in `name` derive different ones. An authored root `id` is preserved verbatim and takes precedence over derivation.
- **Root round-trip** (§1.4a): a derived root `id` survives `to_bytes` → `from_bytes`. A load → edit → save → reload cycle preserves it, and re-saving edited content under the same id succeeds.
- **Strict profile** (§1.4a): `StrictRequireProvided` emits `MissingEntityId` for an id-less root rather than deriving one — verified by profile fixture.
- **I4** (name-ordered serialization): round-trip tests verify that repeated serialize → parse cycles produce byte-identical output when names stay constant, even under convenience profile (id values may vary, name order does not).
- Coverage: unit tests for `iter_all` / `iter_public` / `find_public` accessors; entity-id mint under both identity profiles.

Shared harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
