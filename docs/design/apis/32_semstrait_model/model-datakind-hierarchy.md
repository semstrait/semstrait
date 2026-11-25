---
id: 32-model-datakind-hierarchy
status: ratified
version: 2.1
impl: implemented
---

# DataKind Type Hierarchy

Six layers: a common-fields struct, per-variant shared bodies, concrete types in two forms, a sealed trait hierarchy on two axes, per-concrete trait impls, and view enums for heterogeneous iteration.

## 1. Common-fields struct — `DataKindBase<E>`

```rust
pub struct DataKindBase<E> {
    pub id:     Option<EntityId>,   // None at authoring; minted to Some at parse
    pub name:   String,
    pub extras: E,
}
```

Held inside every per-variant body (§2). Carries the universal fields every data kind exposes regardless of variant or form: an `id` (`None` when authoring omits it, minted to a UUIDv7 `Some` at parse; the inner `EntityId` is the entity's storage key), a `name` (anchoring + structural label per `26 §4`), and an `extras` block parameterized over the per-axis flavor — `LeafExtras` for the leaf body and `ComplexExtras` for the three composer bodies. Because `id` lives on the base, all eight concrete DataKind types (Public + Nested) carry it through `body.base`. The `Option` is the type-level honest shape — `None` = authored-absent (no `""` sentinel transient); after parse the field is invariably `Some`.

`description`, `ai_context`, and `semantic_interface` are NOT on the base — they are Public-form-only, and live on each Public concrete type directly (§3).

## 2. Per-variant bodies

Each variant has a single `*Body` struct holding `base: DataKindBase<E>` (§1) plus variant-intrinsic structural fields. The `<E>` parameter is `LeafExtras` for the leaf body and `ComplexExtras` for the three composer bodies — the type-level expression of R-6 (`storage` / `catalog` / `semantic_mapping` are leaf-only). Public and Nested forms of the same variant wrap the same body (§3).

Self-nesting is type-level forbidden by field absence: no `grainsets:` field on `GrainsetBody`, no `unionsets:` field on `UnionsetBody`, no `joinsets:` field on `JoinsetBody`.

```rust
pub struct DatasetBody {
    pub base: DataKindBase<LeafExtras>,
}

pub struct GrainsetBody {
    pub base:      DataKindBase<ComplexExtras>,
    pub datasets:  BTreeMap<EntityId, NestedDataset>,
    pub unionsets: BTreeMap<EntityId, NestedUnionset>,
    pub joinsets:  BTreeMap<EntityId, NestedJoinset>,
}

pub struct UnionsetBody {
    pub base:      DataKindBase<ComplexExtras>,
    pub datasets:  BTreeMap<EntityId, NestedDataset>,
    pub grainsets: BTreeMap<EntityId, NestedGrainset>,
    pub joinsets:  BTreeMap<EntityId, NestedJoinset>,
    pub mode:      UnionMode,
}

pub struct JoinsetBody {
    pub base:          DataKindBase<ComplexExtras>,
    pub datasets:      BTreeMap<EntityId, NestedDataset>,
    pub grainsets:     BTreeMap<EntityId, NestedGrainset>,
    pub unionsets:     BTreeMap<EntityId, NestedUnionset>,
    /// Joinset-local relationships. Unified `Relationship` shape — same
    /// struct as `SemanticModel.relationships`. See `18 §2`.
    pub relationships: BTreeMap<EntityId, Relationship>,
}

#[non_exhaustive]
pub enum UnionMode { All, Unique }   // `All` is the default
```

The full nesting matrix is at `26 §1`; structural rules (R1 leaves don't nest; R2 no same-variant self-nesting; R3 ComplexDataKind ≥ 2 children) are at `26 §2`. `Relationship` shape is ratified at `18 §2`; `UnionMode` roster at `23 §4.1` (variant-local).

## 3. Concrete types — Public and Nested forms

Each variant has two concrete forms. Public wraps a body plus three Public-form-only fields — `description`, `ai_context`, `semantic_interface`. Nested wraps only the body.

```rust
// Public (top-level) forms — carry description + ai_context + interface.
pub struct Dataset {
    pub body:               DatasetBody,
    pub description:        Option<String>,
    pub ai_context:         Option<AiContext>,
    pub semantic_interface: SemanticInterface,
}
pub struct Grainset {
    pub body:               GrainsetBody,
    pub description:        Option<String>,
    pub ai_context:         Option<AiContext>,
    pub semantic_interface: SemanticInterface,
}
pub struct Unionset {
    pub body:               UnionsetBody,
    pub description:        Option<String>,
    pub ai_context:         Option<AiContext>,
    pub semantic_interface: SemanticInterface,
}
pub struct Joinset {
    pub body:               JoinsetBody,
    pub description:        Option<String>,
    pub ai_context:         Option<AiContext>,
    pub semantic_interface: SemanticInterface,
}

// Nested (structural) forms — body only.
pub struct NestedDataset  { pub body: DatasetBody }
pub struct NestedGrainset { pub body: GrainsetBody }
pub struct NestedUnionset { pub body: UnionsetBody }
pub struct NestedJoinset  { pub body: JoinsetBody }
```

YAML deserialization uses `#[serde(flatten)]` on the `body:` field so per-variant author YAML reads as one flat object.

## 4. Sealed trait hierarchy — `DataKind` + two axes

All traits are sealed inside the crate. The base trait `DataKind` carries only the universal name + tag accessors; two orthogonal axes of sub-traits classify *structural* shape (leaf vs composer) and *behavioral* shape (queryable top-level vs structural shell). Each axis trait owns the accessors specific to its axis.

```rust
mod sealed { pub trait Sealed {} }

pub trait DataKind: sealed::Sealed {
    fn name(&self) -> &str;
    fn variant(&self) -> DataKindVariant;
    fn form(&self) -> DataKindForm;
}

// ── Structural axis ──────────────────────────────────────────────
// Each subtype distributes its own `extras` flavor (LeafExtras vs ComplexExtras)
// per the `DataKindBase<E>` parameterization in §1 / §2.
pub trait SimpleDataKind: DataKind {
    fn extras(&self) -> &LeafExtras;
}

pub trait ComplexDataKind: DataKind {
    fn extras(&self) -> &ComplexExtras;
    fn allowed_child_variants(&self) -> &'static [DataKindVariant];
    fn child_count(&self) -> usize;
    fn children_ref(&self) -> Box<dyn Iterator<Item = NestedDataKindRef<'_>> + '_>;
}

// ── Behavioral axis ──────────────────────────────────────────────
// PublicDataKind owns the three Public-form-only accessors; NestedDataKind
// is a pure marker.
pub trait PublicDataKind: DataKind {
    fn description(&self) -> Option<&str>;
    fn ai_context(&self) -> Option<&AiContext>;
    fn semantic_interface(&self) -> &SemanticInterface;
}

pub trait NestedDataKind: DataKind {}

#[non_exhaustive]
pub enum DataKindVariant { Dataset, Grainset, Unionset, Joinset }

#[non_exhaustive]
pub enum DataKindForm { Public, Nested }
```

`NestedDataKind` is a pure marker: its contribution is the trait bound itself, which lets generic code require nested-ness without inspecting tags. The other three sub-traits carry axis-specific accessors. `SimpleDataKind::extras` and `ComplexDataKind::extras` differ in return type by design — leaf vs complex extras shapes are not interchangeable.

## 5. Trait implementation matrix

Every concrete type implements `DataKind`, exactly one trait on the structural axis, and exactly one trait on the behavioral axis.

<!-- cross-cut: none -->

| Concrete         | `DataKind` | Structural axis   | Behavioral axis  |
| ---------------- | ---------- | ----------------- | ---------------- |
| `Dataset`        | ✓          | `SimpleDataKind`  | `PublicDataKind` |
| `NestedDataset`  | ✓          | `SimpleDataKind`  | `NestedDataKind` |
| `Grainset`       | ✓          | `ComplexDataKind` | `PublicDataKind` |
| `NestedGrainset` | ✓          | `ComplexDataKind` | `NestedDataKind` |
| `Unionset`       | ✓          | `ComplexDataKind` | `PublicDataKind` |
| `NestedUnionset` | ✓          | `ComplexDataKind` | `NestedDataKind` |
| `Joinset`        | ✓          | `ComplexDataKind` | `PublicDataKind` |
| `NestedJoinset`  | ✓          | `ComplexDataKind` | `NestedDataKind` |

## 6. View enums for heterogeneous iteration

Five view-only enums provide exhaustive-match access over borrowed concrete values. Each view implements the subset of the trait hierarchy common to its members, so generic trait-bounded functions accept views as first-class `DataKind` values.

```rust
#[non_exhaustive]
pub enum AnyDataKindRef<'a> {
    Dataset(&'a Dataset),
    NestedDataset(&'a NestedDataset),
    Grainset(&'a Grainset),
    NestedGrainset(&'a NestedGrainset),
    Unionset(&'a Unionset),
    NestedUnionset(&'a NestedUnionset),
    Joinset(&'a Joinset),
    NestedJoinset(&'a NestedJoinset),
}

#[non_exhaustive]
pub enum PublicDataKindRef<'a> {
    Dataset(&'a Dataset),
    Grainset(&'a Grainset),
    Unionset(&'a Unionset),
    Joinset(&'a Joinset),
}

#[non_exhaustive]
pub enum NestedDataKindRef<'a> {
    Dataset(&'a NestedDataset),
    Grainset(&'a NestedGrainset),
    Unionset(&'a NestedUnionset),
    Joinset(&'a NestedJoinset),
}

#[non_exhaustive]
pub enum SimpleDataKindRef<'a> {
    Public(&'a Dataset),
    Nested(&'a NestedDataset),
}

#[non_exhaustive]
pub enum ComplexDataKindRef<'a> {
    Grainset(&'a Grainset),
    NestedGrainset(&'a NestedGrainset),
    Unionset(&'a Unionset),
    NestedUnionset(&'a NestedUnionset),
    Joinset(&'a Joinset),
    NestedJoinset(&'a NestedJoinset),
}
```

Trait implementations for views:

<!-- cross-cut: none -->

| View enum            | Implements                     |
| -------------------- | ------------------------------ |
| `AnyDataKindRef`     | `DataKind`                     |
| `PublicDataKindRef`  | `DataKind` + `PublicDataKind`  |
| `NestedDataKindRef`  | `DataKind` + `NestedDataKind`  |
| `SimpleDataKindRef`  | `DataKind` + `SimpleDataKind`  |
| `ComplexDataKindRef` | `DataKind` + `ComplexDataKind` |

Every view method dispatches via `match` to the underlying concrete type. Views own no data and are never persisted; they are constructed on demand by the iterators in the model-root-shape leaf and by `ComplexDataKind::children_ref` on complex bodies.

## 7. Storage layout

- `SemanticModel.{datasets, grainsets, unionsets, joinsets}` — `BTreeMap<EntityId, _>` keyed by entity `id`, holding the four **Public** concrete types.
- `SemanticModel.{dimensions, measures, metrics}` and `SemanticModel.relationships` — `BTreeMap<EntityId, _>` keyed by entity `id`.
- `*Body.{datasets, grainsets, unionsets, joinsets}` and `JoinsetBody.relationships` — `BTreeMap<EntityId, _>` holding the **Nested** concrete types (and Joinset-local `Relationship`s), constrained by each body's declared child set.
- Every one of these `BTreeMap<EntityId, _>` collections is held behind a `EntityBTreeMap<T>` wrapper that owns, in one place: the map; the wire projection map ↔ `Vec<T>` (`#[serde(try_from = "Vec<T>", into = "Vec<T>")]` — the `into` sorts **by name**, the `try_from` rejects duplicate ids, SR-12); a hand-written `impl JsonSchema` delegating to `Vec<T>` so the schema is an array (schemars does not read field-level `#[serde(with)]`, so the projection lives on the wrapper type, not a serde module); and the id-as-derived-key discipline (the key is rebuilt from the value's `id` on decode, never mutated independently). All three surfaces — serialize, deserialize, schema — derive from one `Vec<T>`-shaped projection.
- `parse` materializes decoded entities straight into these id-keyed maps via the `EntityBTreeMap` projection; there is no separate accumulation layout, handle table, name index, or `SemanticModelStorage` — duplicate-by-name detection is an explicit per-layer scan at `.build()`.

The Public / Nested split is enforced at the type level: no Public-form value can appear inside a body's child vector, and no Nested-form value can appear at the top-level map.

## Testing Strategy

- **SR-2** (nested data kinds carry no Public-form fields): type-level enforcement + `deny_unknown_fields` — verified by negative-parse tests that attempt to author `description` / `ai_context` / `semantic_interface` on a nested data kind (expect `ParseErrorKind::UnknownField`).
- **SR-4** (self-nesting forbidden): type-level enforcement by child-field absence — verified by negative-parse tests that attempt to nest a grainset inside a grainset, unionset inside unionset, or joinset inside joinset (expect `ParseErrorKind::UnknownField`).
- **SR-10** (complex data kinds ≥ 2 children): `ValidateError::ComplexDataKindInsufficientChildren` fired at `.build()` — verified by validate-stage test with 0-child and 1-child composers.
- **Trait matrix coverage**: unit tests verify that each concrete type implements exactly the expected traits (Dataset = SimpleDataKind + PublicDataKind, NestedDataset = SimpleDataKind + NestedDataKind, etc.).
- **View enum dispatch**: unit tests verify that each view enum's trait methods dispatch correctly to the underlying concrete type accessors.

Shared harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
