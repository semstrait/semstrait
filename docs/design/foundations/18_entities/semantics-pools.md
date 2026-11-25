---
id: 18-semantics-pools
status: ratified
version: 2.1
impl: implemented
---

# Semantics Pools

`18` is the canonical specification for entity payload types that populate a `SemanticModel`: shared Semantics pools, relationships, temporal shapes, dimensions / measures / metrics, filters, AI context, keys, and model-authoring `SemanticMapping` value shapes. `32` owns root YAML shape and DataKind hierarchy; `18` owns the entity payloads embedded in those structures.

Every struct in this document is `#[non_exhaustive]` and every enum is `#[non_exhaustive]` per I10, unless a specific note overrides.

### Scope note — payload shape and identity field

`18` owns canonical **entity payload shapes**. Named-entity identity is an `id: Option<EntityId>` field carried directly on each entity struct in this document (`Dimension`, `Measure`, `Metric`, `Relationship`, `Key` decls, `Filter`), mirroring the `DataKindBase.id` field for data kinds (`32 §3.1`). `None` is the authoring-absent shape (no `""` sentinel); parse mints a UUIDv7 into every `None` per `IdentityProfile` (`32 §9.0.1`), so after parse the field is invariably `Some` and the inner `EntityId` is the storage key. `EntityId` is the identity type defined in `semstrait-common` at `31 §2.1` (canonical UUID text; authored ids are UUIDv7) and re-exported through the model surface at `32 §2`.

Identity routing:

- model authoring boundary (`id` optional at authoring; strict/convenience missing-id generation profiles) — `32 §1.4` and `32 §9.0.1`;
- the entity's `id` is also its storage key in every model collection — `32 §2`, `§3.7`;
- manifest propagation — the entity's `id` is carried straight onto the corresponding manifest entity (the manifest is `EntityId`-keyed, single id lane) — `33 §4.3.1` / `§9.1`.

There is no separate identity sidecar at the model layer and no `stable_ids` side-map at the manifest layer; `id` is part of the payload throughout.

This leaf owns the shared Semantics pools (`dimensions:`, `measures:`, `metrics:`) at the root level of a `SemanticModel` and their per-DataKind reference / override grammar. See [dimension](dimension.md), [measure](measure.md), and [metric](metric.md) for the entity payload shapes.

## 1. Shared Semantics Pools & Reference Grammar

### 1.1 Two authoring locations

A Semantic (`Dimension`, `Measure`, or `Metric`) may be authored at exactly one of two locations:

- **Root pool** — under the top-level `dimensions:` / `measures:` / `metrics:` arrays on the `SemanticModel`. Root-pool entries are globally unique per carrier (`dimensions["revenue"]` and `measures["revenue"]` can coexist; two `dimensions["revenue"]` cannot).
- **Inline on a DataKind's `SemanticInterface`** — authored under the Public form of a data kind (`Dataset` / `Grainset` / `Unionset` / `Joinset`). Inline Semantics are scoped to their data kind; cross-kind visibility is via the relationship graph.

A root-pool Semantic may be **referenced** at any number of DataKinds, optionally with a local override (§1.3). A root-pool Semantic with no references and no physical binding is an orphan (§1.4).

### 1.2 Reference site grammar

A DataKind's `SemanticInterface` lists Semantics in two forms — inline or referenced. YAML:

```yaml
datasets:
  - name: orders
    dimensions:
      # Inline declaration (DataKind-local).
      - name: order_id
        data_type: string
        type: categorical

      # Reference to a root-pool Dimension.
      - ref: country

      # Reference with a local expression override.
      - ref: revenue_bucket
        expr: case_when(
                revenue > 1000, 'high',
                revenue > 100,  'medium',
                                'low'
              )
    measures:
      # Reference to a root-pool Measure with a local expression override.
      - ref: total_revenue
        expr: amount_cents * 0.01
```

The YAML shapes deserialize into two variants of each carrier:

```rust
#[non_exhaustive]
pub enum DimensionEntry {
    Inline(Dimension),
    Ref(DimensionRef),
}

#[non_exhaustive]
pub struct DimensionRef {
    /// Named-entity identity for this reference site (`32 §2.3`). A `Ref` is a
    /// distinct interface member with its own `id` — authored optionally,
    /// parse-generated when omitted. It identifies the reference-with-override
    /// site; the referenced root-pool Dimension keeps its own `id`.
    pub id: Option<EntityId>,
    pub name: SemanticsName,
    /// Local override of the root-pool expression (see §1.3).
    pub expr: Option<UnresolvedSemanticExpr>,
}
```

`MeasureEntry` / `MeasureRef` and `MetricEntry` / `MetricRef` are analogous (each `Ref` carries the same `id` field). The `Ref` variant carries only the fields that are legally overridable at the reference site (§1.3); every other attribute is read from the root-pool declaration at `compile` time.

Each entry's `id` is the storage key in its DataKind's `SemanticInterface`: the interface holds its `DimensionEntry` / `MeasureEntry` / `MetricEntry` members (and its `DataKindFilter`s, §7.1) in `EntityId`-keyed `BTreeMap`s — an `Inline` entry keys by its inner entity's `id`, a `Ref` entry by the ref-site `id`. Iteration and serialization are name-ordered per `32 §7`. Keys live in the `Keys` struct (§9).

### 1.3 Override scope at a reference site

Only a narrow set of fields may be overridden locally at a `Ref` site. Every other attribute is **immutable from the root-pool declaration**:

| Carrier | Overridable at `ref` | Immutable from root pool |
|---|---|---|
| `Dimension` | `expr` | `data_type`, `type`, `description`, `ai_context`, `constraints` (future) |
| `Measure` | `expr`, `filters` | `data_type`, `agg`, `additivity`, `description`, `ai_context`, `constraints` (future) |
| `Metric` | `expr`, `filters` | `data_type`, `agg`, `additivity`, `description`, `ai_context`, `constraints` (future) |

Attempting to author an immutable field at a `Ref` site is `validate.semantics-ref-immutable-override` (SR-E-1).

**Deferred-body pattern.** A root-pool Semantic MAY declare `data_type:`, `agg:`, `additivity:`, etc. without an `expr:` — the expression is expected to be provided per-DataKind at every `Ref` site. A root-pool Semantic with no `expr:` and a `Ref` site with no `expr:` is `validate.semantics-ref-missing-expr` (SR-E-2). Deferred-body root Metrics are how cross-DataKind metrics are authored — the root entry fixes the name / data type / aggregation shape; each DataKind's `Ref` provides the local `expr:` that binds to that DataKind's surface.

### 1.4 Orphan policy

Every Semantic — whether authored in the root pool or inline — MUST be physically bound at least once. A physical binding is any path from the Semantic to a `Dataset`'s `extras.semantic_mapping` that resolves without error:

- **Inline on a Dataset.** The Semantic is automatically bound (the containing leaf's `semantic_mapping` or its `auto` default covers it).
- **Inline on a ComplexDataKind.** Not legal — Public complex kinds flatten their Semantics only through their composed children, but direct authoring on a composer is allowed only for composer-unique Semantics (those surfaced by `Joinset`'s `ComposedSemanticInterface`). The path to physical still terminates in a `Dataset` leaf.
- **Root-pool.** The Semantic must be `Ref`-ed from at least one DataKind whose downstream tree eventually reaches a `Dataset` with a compatible `semantic_mapping`.

A Semantic with no downstream `Dataset` binding is `validate.semantics-orphan { carrier, name }` (SR-E-3). The diagnostic fires at `validate` so tooling can report orphans before `compile` tries to build physical plans.

### 1.5 Uniqueness rules

Per-carrier uniqueness (§2.1 of `32`):
- `dimensions[..]` unique by name in the root pool.
- `measures[..]` unique by name in the root pool.
- `metrics[..]` unique by name in the root pool.
- Inline Semantics on a DataKind unique by name within that DataKind's interface.
- A `Ref` site does not redeclare the name — the name is read from the root-pool entry.

A DataKind MAY shadow a root-pool name by inlining a same-named Semantic (e.g. a locally overridden `Measure`), but this is warned as `validate.semantics-shadow-root-pool` — authors who want shadowing should use `ref` + override instead.

## 12. Cross-References

- `32 §1` — root YAML shape and where the shared pools / relationships / data-kinds / semantic-mapping blocks live.
- `32 §3` — `DataKind` hierarchy; `DatasetBody` / `GrainsetBody` / `UnionsetBody` / `JoinsetBody`. `JoinsetBody.relationships: Vec<Relationship>` uses the unified struct from §2.
- `32 §4` — `LeafExtras` / `ComplexExtras` blocks; `semantic_mapping:` lives inside `LeafExtras`, value shape per §10.
- `32 §6` — root `SR-*` rules; complementary to `SR-E-*` here.
- `32b` — catalog grammar; `CatalogRef` referenced from `extras.catalog:`.
- `26` — nesting matrix; complements the ≥ 2 children structural rule.
- `11` — name / scope rules for Semantics.
- `13` — `DataType`, `Grain`, `DimensionType` vocabulary this doc embeds.
- `14` — `Expr<L>` grammar for every `expr:` field here (authoring `UnresolvedSemanticExpr` / `UnresolvedPhysicalExpr`; resolved `ir::PhysicalExpr`).
- `19 §3` — compile-time resolution of `UnresolvedSemanticExpr` at `Ref` sites + override merging.
- `15` — `Binding` process that consumes `semantic_mapping` + the entities ratified here.
- `16` — `Relationship` graph consumption; `Joinset` path synthesis.
- `17` — planner-level `TemporalShape` semantics and rollup matrices.
- `21` / `22` / `23` / `24` — per-variant YAML carriage of the entity types.
- `30 §6` — diagnostic code conventions (kebab-case primary).

---

*Cross-references use `NN §M.K` for internal sections and full relative paths for other docs.*

## Testing Strategy

- SR-E-1 (immutable-override rejection), SR-E-2 (deferred-body + missing-expr), SR-E-3 (orphan detection); reference vs inline round-trip; override merging; name-shadowing warnings.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
