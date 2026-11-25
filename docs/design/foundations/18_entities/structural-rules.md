---
id: 18-structural-rules
status: ratified
version: 2.1
impl: implemented
---

# Structural Rules

This leaf owns the entity-level structural rules (SR-E-*) — numbered validation invariants SR-E-1 through SR-E-14. See sibling leaves for the entity types these rules constrain.

## 11. Structural Rules (SR-E-*)

Entity-level invariants. Each rule has a stable kebab-case diagnostic code per `30 §6`. Numbered independently from `32`'s root-level `SR-*` rules so additions do not perturb the other series.

| ID | Rule | Diagnostic |
|---|---|---|
| **SR-E-1** | Reference-site override MAY NOT author immutable fields (`data_type`, `type`, `agg`, `additivity`, `description`, `ai_context`). See §1.3. | `validate.semantics-ref-immutable-override` |
| **SR-E-2** | A `Ref` site missing `expr:` AND root-pool entry missing `expr:` is ill-formed. | `validate.semantics-ref-missing-expr` |
| **SR-E-3** | Every Semantic (root-pool or inline) MUST physically bind at least once. | `validate.semantics-orphan` |
| **SR-E-4** | `Relationship.cardinality:` is required at every authoring site. | `parse.relationship-missing-cardinality` |
| **SR-E-5** | `Relationship.from:` / `.to:` MUST resolve to a declared DataKind. | `validate.relationship-dangling-endpoint` |
| **SR-E-6** | A leaf `Dataset` with `extras.temporal:` authored MUST declare `temporal.grain:`. | `validate.temporal-leaf-missing-grain` |
| **SR-E-7** | A `ComplexDataKind` MUST NOT author `extras.temporal.grain:` (only shape cascades). | `validate.temporal-grain-on-complex` |
| **SR-E-8** | Every `Grainset` child MUST author its own `extras.temporal.grain:` explicitly (no inheritance). | `validate.grainset-child-grain-required` |
| **SR-E-9** | A `Measure` MUST declare `agg:` at its declaration site. | `parse.measure-missing-agg` |
| **SR-E-10** | A `Dimension` / `Measure` / `Metric` MUST declare `data_type:` at its declaration site. | `parse.semantics-missing-data-type` |
| **SR-E-11** | Filter names are not cross-referenceable between `DataKindFilter` and `AggregationFilter`. | `validate.wrong-filter-error` |
| **SR-E-12** | `data_type:` is immutable across all levels — root-pool and Ref sites must agree; local overrides are forbidden. | `validate.semantics-data-type-mismatch` |
| **SR-E-13** | `Relationship.optional:` and `Relationship.cross_filter:` are REQUIRED when `cardinality ∈ {OneToOne, ManyToMany}`. Asymmetric cardinalities (`ManyToOne`, `OneToMany`) accept defaults from `§2.7`. | `validate.relationship-symmetric-cardinality-incomplete` |
| **SR-E-14** | `Relationship.cross_filter ∈ {Left, Right}` is REJECTED when `cardinality == ManyToMany`. A many-to-many relationship has no natural "one" side; directional filter propagation is ambiguous and authors MUST declare `Both` or `None`. | `validate.relationship-many-to-many-cross-filter-directional` |

SR-E-* numbering is append-only; adding a rule is MINOR per `30 §2`.

## Testing Strategy

- SR-E-1 through SR-E-14 invariants; diagnostic-code allocation (kebab-case primary per 30 §6); append-only numbering; cross-reference to constrained entity leaves; table completeness (each rule maps to exactly one or more entity types).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
