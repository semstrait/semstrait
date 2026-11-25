---
id: 26-nesting-matrix
status: ratified
version: 2.1
impl: none
---


# 26. Nesting Matrix

`26` is the single source of truth for which data-kind variant may contain which variant as a nested child. Per-variant structural details beyond the matrix live in `22` (Grainset), `23` (Unionset), `24` (Joinset).

## 1. The Matrix

Each cell answers: can an instance of the Child variant appear inside a Parent variant's child arrays?

| Parent ↓ \ Child → | `dataset` | `grainset` | `unionset` | `joinset` |
|---|---|---|---|---|
| `dataset` | ✗ (R1) | ✗ (R1) | ✗ (R1) | ✗ (R1) |
| `grainset` | ✓ | ✗ (R2) | ✓ | ✓ |
| `unionset` | ✓ | ✓ | ✗ (R2) | ✓ |
| `joinset` | ✓ | ✓ | ✓ | ✗ (R2) |

Legend:
- **✓** — allowed by the matrix. Per-variant constraints may apply (see `22`–`24`). Composite-count constraint R3 (§2.3) applies universally.
- **✗ (R1)** — forbidden by R1 (leaves never contain children).
- **✗ (R2)** — forbidden by R2 (same-variant self-nesting).

The matrix is purely about *which child variants are admissible*. **How many** children a composer must have (≥ 2) is R3 (§2.3).

### 1.2 Empty arrays

Absent child arrays and empty child arrays are both legal. A complex variant with zero children is structurally valid; the downstream `compile` stage may raise a warning (`compile.empty-complex-data-kind`) unless the authoring intent is a placeholder.

---

## 2. Rules (R1, R2, R3)

### 2.1 R1 — Leaf variants never contain children

The leaf variant `dataset` has no structural child arrays. A leaf is terminal: it maps to exactly one `PhysicalSource` after the binding process (`32 §5.3`) resolves `semantic_mapping`.

`DatasetBody` (per `32 §3.2`) has no child-vector fields. Both concrete leaf forms — `Dataset` and `NestedDataset` — wrap `DatasetBody`, so neither can receive child authoring. The `SimpleDataKind` trait (`32 §3.4`) is a marker that both concrete forms implement. A YAML document that attempts to author children under a dataset fails at `deny_unknown_fields`:

```
parse.unknown-field { parent: "dataset", field: "<child-tag>" }
```

### 2.2 R2 — Same-variant self-nesting is forbidden

A `grainset` cannot contain a `grainset`; a `unionset` cannot contain a `unionset`; a `joinset` cannot contain a `joinset`.

Enforced at the type level by each `*Body` struct's child-field set (`32 §3.2`): no variant's body declares a vector of its own variant. The three diagonal `✗ (R2)` matrix cells have no Rust field to deserialize into, so any YAML document that names the own-variant plural under a complex parent fails at `deny_unknown_fields`:

```
parse.illegal-self-nesting { parent_variant, nested_variant }
```

Each body's shape (`32 §3.2` reproduced):

```rust
pub struct GrainsetBody {
    pub base:      DataKindBase<ComplexExtras>,
    pub datasets:  BTreeMap<EntityId, NestedDataset>,
    pub unionsets: BTreeMap<EntityId, NestedUnionset>,
    pub joinsets:  BTreeMap<EntityId, NestedJoinset>,
    // no `grainsets:` field
}

pub struct UnionsetBody {
    pub base:      DataKindBase<ComplexExtras>,
    pub datasets:  BTreeMap<EntityId, NestedDataset>,
    pub grainsets: BTreeMap<EntityId, NestedGrainset>,
    pub joinsets:  BTreeMap<EntityId, NestedJoinset>,
    pub mode:      UnionMode,
    // no `unionsets:` field
}

pub struct JoinsetBody {
    pub base:          DataKindBase<ComplexExtras>,
    pub datasets:      BTreeMap<EntityId, NestedDataset>,
    pub grainsets:     BTreeMap<EntityId, NestedGrainset>,
    pub unionsets:     BTreeMap<EntityId, NestedUnionset>,
    pub relationships: BTreeMap<EntityId, Relationship>,   // unified shape (`18 §2`)
    // no `joinsets:` field
}
```

### 2.3 R3 — `ComplexDataKind` requires at least 2 children

Every `ComplexDataKind` (`Grainset`, `Unionset`, `Joinset` — Public or Nested) MUST have **at least 2 children** across its allowed child-variant arrays. A composer with fewer children is ill-formed:

- **0 children** — nothing to compose; the composer has no effect and no physical grounding.
- **1 child** — the composer's interface collapses to that single child's interface; the composer is redundant and should be authored as the child directly. Unionset of one is an identity; Joinset of one is the member itself; Grainset of one is a single grain — none adds semantic value.

R3 is **not** enforced at parse (the YAML shape still accepts 0 or 1 children syntactically); it is validated post-parse against the fully-resolved `*Body` struct. Diagnostic:

```
validate.complex-data-kind-insufficient-children {
  data_kind: <addressed path per §4>,
  variant:   grainset | unionset | joinset,
  child_count: 0 | 1,
}
```

R3 corresponds to SR-10 in `32 §6`. Downstream `compile` will see only well-formed composers, so this rule is a hard validate-stage gate, not a warning.

**Example — illegal single-child Joinset:**

```yaml
joinsets:
  - name: trivial_join                     # ← validate.complex-data-kind-insufficient-children
    datasets:
      - { name: orders }                   # only 1 child, no relationships, no other members
    relationships: []
```

The author should promote `orders` to a root-level Dataset or add at least one more joinset member.

**Counting children.** The child count is the sum of lengths across all allowed child-variant arrays for the composer:

| Composer | Counted fields |
|---|---|
| `GrainsetBody` | `datasets.len() + unionsets.len() + joinsets.len()` |
| `UnionsetBody` | `datasets.len() + grainsets.len() + joinsets.len()` |
| `JoinsetBody`  | `datasets.len() + grainsets.len() + unionsets.len()` |

`JoinsetBody.relationships` does **not** count — relationships describe edges between members, not members themselves.

---

## 3. Nested-Form Structural-Only Rule

A nested data kind carries:

- `name: String` — mandatory (anchoring + structural label per §4 addressing).
- `extras` — optional. Flavor depends on the structural axis: `LeafExtras` for `NestedDataset`; `ComplexExtras` for the three nested composers (per `32 §4`).
- Variant-specific structural fields (nested child vectors; `mode:` on `NestedUnionset`; `relationships:` on `NestedJoinset`).

A nested data kind does NOT carry:

- `description:` — the human-readable description is Public-form-only (per `32 §3.3`); top-level data kinds are the queryable entry points and the only ones that surface a description on the API.
- `ai_context:` — likewise Public-form-only.
- `dimensions:` / `measures:` / `metrics:` / `keys:` / `filters:` — the entire `SemanticInterface`.
- Any aggregate interface field.

Enforced at the type level: `Nested*` structs (`32 §3.3`) wrap only a `*Body` — they have no `description`, `ai_context`, or `semantic_interface` fields — and each `Nested*` implements the `NestedDataKind` marker trait (`32 §3.4`) as the behavioral axis. A YAML author who writes any of the prohibited fields under a nested entry hits `deny_unknown_fields`:

```
parse.nested-data-kind-carries-interface {
  parent: <addressed-parent>,
  nested: <addressed-nested>,
  offending_field: <field-name>,
}
```

### 3.1 Extras on nested data kinds

`extras:` IS allowed on nested data kinds, but the field set is **type-constrained by the structural axis** (`32 §4`):

- A nested **leaf** (`NestedDataset`) carries the full `LeafExtras`: `catalog:` / `storage:` / `semantic_mapping:` / `temporal:`.
- A nested **composer** (`NestedGrainset` / `NestedUnionset` / `NestedJoinset`) carries only `ComplexExtras`: `temporal:`.

Authoring `catalog:` / `storage:` / `semantic_mapping:` under any complex data kind (top-level or nested) hits `parse.unknown-field` — these fields have no slot in `ComplexExtras` (per R-6 / `32 §4.1` SR-5). Each leaf dataset that needs a catalog or storage configuration authors it directly on its own `extras:` block; cascade-from-ancestor is **only** in effect for `temporal.<variant>:` (the shape kind), not for any leaf-only field.

```yaml
grainsets:
  - name: sales
    extras:
      temporal:                           # shape kind cascades to every leaf descendant
        timeseries:
          occurred_at: order_date
    unionsets:
      - name: regions
        # ComplexExtras here may carry only `temporal:` — no `catalog:`, no `storage:`.
        datasets:
          - name: sales_us
            extras:
              catalog: polaris_prod       # leaf-only — authored on the leaf
              storage:
                format: parquet
                paths: ["s3://.../us/*.parquet"]
              temporal:
                grain: day                # leaf-only; cascades nothing
          - name: sales_eu
            extras:
              catalog: polaris_prod
              storage:
                format: parquet
                paths: ["s3://.../eu/*.parquet"]
              temporal:
                grain: day
```

Each leaf dataset inherits the `timeseries:` shape kind from the root grainset (cascade-from-ancestor, "more-specific-overrides-default" merge per `32 §4.1`) and authors its own `catalog:` / `storage:` / `temporal.grain:` directly. Leaf-only fields do not cascade.

---

## 4. Addressing Scheme for Diagnostics

Nested data kinds are addressed in diagnostics via dotted plural-key paths:

```
grainsets[sales].unionsets[regions].datasets[sales_us]
```

Rules:

- Top-level segment is the variant's plural key (`datasets` / `grainsets` / `unionsets` / `joinsets`).
- Each subsequent segment is `<plural-key>[<child-name>]`.
- `<child-name>` is the nested data kind's `name:` field.
- Brackets are ASCII `[` / `]`; names are copied verbatim (identifiers follow `11 §4`).

Example inside a `ParseError::NestedDataKindCarriesInterface`:

```
parse.nested-data-kind-carries-interface:
  parent: grainsets[sales].unionsets[regions]
  nested: datasets[sales_us]
  offending_field: dimensions
  location: ... (YAML line/col)
```

The address is stable across renames of Rust types — it is anchored to the YAML author-level plural tags, which are part of the stable authoring surface per `30 §2`.

---

## 5. Enforcement

| Rule | Enforcement | Diagnostic |
|---|---|---|
| R1 (leaves don't nest) | Type-level — `DatasetBody` (`32 §3.2`) has no child-vector fields; both `Dataset` and `NestedDataset` wrap it and implement `SimpleDataKind` (`32 §3.4`) | `parse.unknown-field { parent: "dataset", field: … }` |
| R2 (no same-variant self-nesting) | Type-level — each `*Body` struct's child-field set (`32 §3.2`) omits its own variant | `parse.illegal-self-nesting { parent_variant, nested_variant }` |
| R3 (complex ≥ 2 children) | Validate-stage — post-parse walk over every Public / Nested complex body counts admissible children and asserts `>= 2` | `validate.complex-data-kind-insufficient-children { data_kind, variant, child_count }` |
| Nested-form structural-only | Type-level — `Nested*` structs (`32 §3.3`) omit `description` / `ai_context` / `semantic_interface`; each implements `NestedDataKind` (`32 §3.4`) as the behavioral marker | `parse.nested-data-kind-carries-interface` |
| Matrix cells (Public vs Nested) | Type-level — body child-vector element types are `Nested*` structs; top-level maps hold `Public*` types, so the form is fixed by position | — |

R1 / R2 correspond to SR-4 in `32 §6`. R3 corresponds to SR-10 in `32 §6`. The Nested-form structural-only rule is SR-2 in `32 §6`.

---

## 6. Related Entity-Level Rules (pointer)

Two adjacent rules are enforced at the **entity** layer rather than the nesting layer, but are close enough in intent to flag here so authors find them:

| Adjacent rule | Where it lives | Diagnostic |
|---|---|---|
| Grainset children MUST each author `extras.temporal.grain:` explicitly — shape can cascade, grain cannot. | `18 §3.4` (SR-E-8) | `validate.grainset-child-grain-required` |
| `TemporalShape.grain:` is forbidden at any `ComplexDataKind` level (only shape cascades, not grain). | `18 §3.3` (SR-E-7) | `validate.temporal-grain-on-complex` |
| Joinset `relationships:` use the unified `Relationship` shape (same struct as root `relationships:`) — `Cardinality` required; every Relationship is bidirectional by construction (the `directionality:` field / `Directionality` enum are retired per `18 §2`). | `18 §2` | — |

Full `SR-E-*` entity-level invariant roster is at `18 §11`.

---

*Cross-references use `NN §M.K` for internal sections and full relative paths for other docs.*

## Testing Strategy

- The three nesting rules are enforced: R1 (leaf `dataset` never contains children), R2 (no same-variant self-nesting), R3 (`ComplexDataKind` requires ≥ 2 children) — see the matrix (`§1`) and enforcement table (`§5`).
- R1/R2 are type-level via `deny_unknown_fields` (`parse.unknown-field`, `parse.illegal-self-nesting`); R3 is a validate-stage gate (`validate.complex-data-kind-insufficient-children`).
- Nested data kinds carry no `SemanticInterface` and no `ai_context` (SR-2); a violation raises `parse.nested-data-kind-carries-interface`.
- R1/R2 map to SR-4 and R3 to SR-10 in `32 §6`; the dotted `<plural-tag>[<name>]` diagnostic address (`§4`) is stable across Rust-type renames.

See the shared harness: [`../testing/50_behavioral_contract_gates.md`](../testing/50_behavioral_contract_gates.md).
