---
number: 50
title: 'A relationship is compile input, not a manifest entity: it persists as a joinset''s join condition'
status: accepted
date: 2026-08-20
source:
- CONTEXT.md
- docs/adr/0028-relationship-edge-model.md
- docs/adr/0029-joinset-scoped-relationship-shadow.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
links:
- target: 11
  kind: amends
- target: 13
  kind: amends
- target: 28
  kind: amends
- target: 29
  kind: amends
- target: 38
  kind: amends
- target: 43
  kind: amends
- target: 66
  kind: amendedby
- target: 54
  kind: amendedby
---

# A relationship is compile input, not a manifest entity: it persists as a joinset's join condition

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): a unit is named after the one relationship that caused it, qualified by its left operand's name, applied recursively. Under this rule every unit — straight or cumulative — is caused by exactly one relationship: the one adding the new constituent.
> - [ADR 0066](0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md): "`on_expr_id` always present, `cross` as a tautology" is superseded — the signal it protected rides on the join type. Its two roots map onto the plan: `on_expr_id` → `JoinNode.on`; `filter_expr_id` → `JoinNode.post_join_filter`, or the `FilterNode` above a `CrossNode`. Its reason for keeping `cardinality` on the joinset holds with a new reader: the planner reads it from the manifest to decide fan-out, and the plan carries it in trace only.

## Context and Problem Statement

The question was whether the degenerate `ResolvedJoinKey` survives. The pair was a carrier for a distinction the authoring layer stopped being able to express when ADR 0038 made the basis a name list.

## Considered Options

- No relationship entity: the relationship persists as the produced joinset's pooled join condition (chosen).
- **A semantic-id sum type — `JoinBasis::{ Equi(Vec<EntityId>), Cross }`.** The narrower collapse, and the shape a sub-question proposed. It keeps the basis stated in semantics names, satisfies `CONTEXT.md`'s *Avoid "key pair"*, and makes the retired two-sided type-agreement obligation vacuous by construction. Refused because the manifest then carries the basis in a vocabulary no consumer executes: the joinset is realised as a view with a condition, so an id list must be re-lowered into exactly the expression this decision persists, and the intermediate form buys a check that a single shared Semantics with a mandatory `data_type` had already made unnecessary.
- **Extending `ColumnRef` with a source or side qualifier.** The direct fix for the two-sided reference problem. It forces a `semstrait-model` *and* `semstrait-ir` change — flipping ADR 0043's "the `semstrait-model` answer is no" — and it puts join-scope knowledge into a leaf that ADR 0046 deliberately resolves against one source's schema.
- **An `on` carrier of per-side expression-id pairs.** Each expression stays single-sourced, so no qualifier is needed. Refused because it reintroduces the pair one layer down, which is the shape being retired, and it makes the joinset's condition unreadable as a single predicate.
- **Keeping the pair and documenting the degeneracy at the type.** A legitimate outcome the question admitted, and the one ADR 0038 left the door open for. Refused because the relationship stops being an entity at all, so there is no type left to document it on.
- **`cross` as an absent `on_expr_id`.** Symmetrical with `filter: Option<PhysicalExprId>` and one field lighter. Refused because it makes every joinset answer "what is your join condition" identically whether it has one or not, and it puts the deliberate-cartesian case back into the shape of an absence — the exact reading ADR 0043 called a correctness defect.
- **A mandatory `filter_expr_id` encoding an absent residual as `lit(true)`.** The consistent extension of the tautology rule. Refused because the two fields answer different questions: every join has a condition, so an unconditional one is honestly written as an identity, whereas a residual predicate is genuinely absent rather than unconditional, and `lit(true)` there is noise ANDed into every joinset.
- **Keeping `optional` solely to preserve load-time check 8**, or inverting to persist `optional` and derive `join_type` at load. Both store one fact in two places; the second also moves a value the plan consumes behind a load-time derivation for no gain, since the bijection makes the check tautological either way.

## Decision Outcome

A `Relationship` has **no carrier of its own**. It is an authored declaration that compile *consumes*, and what reaches the manifest is the implicit binary joinset the relationship produced — one semantic source per traversable path, straight or cumulative, each with its own `semantic_bitmask`. On that source's polymorphic `role_definition` sit the two operands it joins and a **pooled join condition**: `on_expr_id`, always present, plus an optional `filter_expr_id`. `SemanticManifest.relationships`, `ResolvedRelationship`, `ResolvedJoinKey` and ADR 0029's `scope_local_relationships` all retire together.

So the question is answered by removing its subject rather than by weighing its degeneracy. The pair is not replaced by a narrower semantic-id carrier, because the whole relationship stops being an entity. Public and private joinsets have identical structure and logic, at every depth of the flattened DAG ADR 0045 established.

### The relationship is a process input, and that is why it has no entity

ADR 0045 already made every join path an ordinary `semantic_sources` entry with `role_definition: Joinset{from, to}`. Two consequences follow that the earlier framing did not draw. `Direction` becomes **structural** — `{from, to}` *is* the traversal direction, so the retired canonical form's `(EntityId, Direction)` tuple has nothing left to encode. And **ADR 0029's shadow precedence resolves at compile**: a Joinset-local relationship that redeclares a root-level name simply produces different implicit joinsets inside that Joinset's originator scope, so nothing resolves a relationship by name at plan time.

That removes the last standing justification for a `Relationship.name` carrier. ADR 0043 recorded the field as owed, noting that one of its two grounds — name-ordered BFS neighbour iteration — had already dissolved with plan-time path walking, and leaving scope-local shadow precedence as the survivor. The survivor dissolves here: precedence is a compile-time fact, and the relationship's name is already persisted where it matters, inside the produced unit's own name (`<originator>.<relationship>`, ADR 0045's seeding rule).

**The documentary fields land on the unit one relationship produced, and nowhere else.** `description` and `ai_context` ride the `role_definition` of a *straight* implicit joinset — the unit caused by exactly one relationship — and a **cumulative** unit carries neither, because several relationships caused it and any value would be a concatenation nobody authored. That satisfies ADR 0043's reason for carrying `ai_context` in the first place ("refusing to persist a field the parser accepts is the worst of three states") without inventing an unauthored one, and it supersedes 0043's `Relationship.name` clause on carrier grounds only.

### The basis is a pooled expression, and the operands' projections are what make it well-formed

`on` and `filter` are both `PhysicalExprId` handles into the `expressions` pool, per the base schema ("we compute implicitly new expressions for joinsets conditions ( on + filter ), so this is the only holder of expressions manifest wise"). This clause of the base schema had appeared on **neither** collision list — not the original comment's five, not the three routed on 2026-08-14 — and it collides with ADR 0038's manifest half, which is ruled here in the base schema's favour.

**The expression's referents are the operands' projected output names.** A join condition spans two sources, and the only column leaf the IR has is `PhysicalLeaf::Column(ColumnRef)` where `ColumnRef` is a bare, unqualified `String` — so `orders.id = customers.id` would lower to `Eq(Column("id"), Column("id"))`, a tautology rather than a join. Each operand is therefore an inline view with its own projection, and the condition references *those* names, unique by construction. This keeps `ColumnRef` untouched and keeps ADR 0046's single-source `Column("name")` resolution check intact; the two rejected ways out are recorded above.

The consequence is stated rather than left implicit: **the projected-naming rule is a precondition for the join condition being well-formed, not a downstream optimisation.** Which realisations a joinspec admits — stacked joins, an inline view with its own projection, whether a squash is a consumer's choice — is opened as research beside this decision and owed back to it.

**A cumulative unit carries its own condition over its two immediate operands, and never inherits or composes theirs.** That is what makes "the same structure and logic no matter what" literal: one binary join over two views, at every depth. Squashing stays a consumer optimisation rather than a second persisted shape.

### `cross` is a symmetric tautology, and an empty basis is unreachable

`on_expr_id` is **always present**. A `cross` basis persists as a literal identity — `Eq(Literal::Integer{value: 1, width: Long}, Literal::Integer{value: 1, width: Long})` — pinned to that exact form because "some tautology" is not byte-deterministic and the manifest must be deterministic, so the tree's shape is fixed. The alternative, an absent `on_expr_id`, loses a signal the machinery and any trace reader want: every joinset then answers "what is your join condition" the same way, and an unconditional join says so explicitly instead of saying nothing.

That encoding only works because **a derivation that finds nothing fails compile**. ADR 0043 recorded `JoinKeySpec`'s discriminant as owed a carrier on correctness, since "empty keys" meant both *deliberate cartesian product* and *derivation found nothing* and no consumer could tell the dangerous state from the failed one. Both halves are answered without a discriminant: the deliberate case has an explicit, positively-encoded condition, and the failed case cannot be persisted at all. Were a failed `Auto` allowed to emit the tautology, the ambiguity would be *worse* than the one 0043 named — byte-identical rather than merely undifferentiated.

The compile failure needs a diagnostic, and the code already asks for one: `compile/seal.rs`'s `relationship_resolution_error` carries a FLAG recording that `CompileError` has **no** variant dedicated to relationship resolution and that it routes through `TypeInferenceFailure` "purely so the fallible signatures are honest", with the lead asked to ratify `CompileError::RelationshipResolution { relationship, reason }`. It is ratified here. Per ADR 0044 it is stated in the relationship-resolution stage's own module documentation and joins no roster.

**`Auto` leaves no trace, and `Declared` leaves no flag.** A resolved derived basis is byte-identical to an authored one, which is ADR 0040's standing counter-precedent honoured rather than beaten — "this codebase already stores derived values without marking them". Distinguishing `cross` from an equi-condition is a distinction of *kind*, carried by the leaf types (`Literal` versus `Column`) rather than by a provenance marker, so nothing here reopens 0040.

### Order fixes the conjunction, not a vector

ADR 0038's ordered-vector rule — "entries are ordered and stored as a **vector, not a set**, because the manifest side is a positional vector and wire order is positional" — loses its manifest subject, since there is no vector left to order. Its authoring half is untouched: `on: [a, b]` is still ordered and a repeated name is still a hard error.

What replaces it is a rule on the tree: the conjunction is **left-deep in authored order** for a declared basis, and **in semantics-name order** for a derived one. Roster order is specifically refused for the derived case — roster and `BTreeSet` iteration are in content-derived-id order, while ADR 0015's ratified mechanism is name order, so a roster-ordered conjunction would reorder on an unrelated id change and move the artifact's bytes for no semantic reason.

### `optional` is dropped because it is recoverable, and that licence is conditional

The `role_definition` carries `join_type`, `cardinality`, `integrity` and `cross_filter`. `optional` is **not** carried: `derive_join_type` maps `Optional::{None, Left, Right, Both}` onto `JoinType::{Inner, Left, Right, Full}` one-to-one, so `optional` is recoverable from the value derived from it. The two were never independent facts.

`cardinality` and `integrity` stay because they are read downstream — `JoinNode.cardinality` is documented "never elided; it always reflects the authored `Relationship`", and integrity feeds the fan-out-safe rewrite — and `cross_filter` stays because filter-propagation direction is a plan-time decision that no other field determines.

**Load-time structural check 8 retires with the field, and it was checking a bijection against itself.** The check re-derives `join_type` from `optional` and rejects a mismatch; with one side of a one-to-one correspondence gone there is nothing independent left to compare. Under ADR 0042's classification this was never a corruption check over independent data. The licence is *conditional* and is recorded as such: both enums are `#[non_exhaustive]`, so if `Optional` ever gains a fifth variant without `JoinType` gaining one, recoverability breaks and the drop reopens.

### Amends

Each supersession is stated and scoped here rather than by editing the older ADR's frontmatter, per the precedent ADR 0037 set over 0031.

- **0011:** `SemanticManifest.relationships`, `ResolvedRelationship` and `ResolvedJoinKey` retire; a relationship persists as the implicit joinset it produced, carrying a pooled join condition.
- **0013:** `ResolvedJoinKey` retires; a joinset's `on` and `filter` are `PhysicalExprId` handles into the pool, and a computed key needs no synthesised `Field`.
- **0038:** supersedes the manifest half only: the `ResolvedJoinKey` semantic-id pair, the ordered-vector clause's manifest subject, and "`Field` stays reachable in the manifest for synthesised and derived cases" — which loses its relationship subject entirely, since an expression holds a computed key directly and needs no synthesised `Field`. 0038's authoring grammar (`on:` as names / `auto` / `cross`, the four join-basis rules, `on: []` as a parse error, `cross` + `filter` as the sanctioned non-equi join, cardinality conditional on the basis) is **untouched**. 0038's named rejected alternative is reversed, not beaten: 0038 rejected "retiring `JoinKeySide::Computed` along with the pair" on the ground that it is the landing site a later relationship-site expression grammar needs. The reversal is licensed by a fact that reading did not separate: `JoinKeySide` is a **private compile-internal** enum in `compile/seal.rs`, not a manifest carrier, so the persisted pair and the computed landing site were always two lanes. The landing site is not merely retired — it is made unnecessary, because a pooled expression *is* the relationship-site expression grammar's carrier. Its output type was already void besides: `resolve_key_side` returns a `SemanticMappingValue` that ADR 0037 retired, installed on a `SemanticBinding` that ADR 0043 retired.
- **0029:** supersedes the persisted half only: the two-lane layout — shadows inline as `scope_local_relationships: Vec<ResolvedRelationship>`, root scope holding non-shadows, the `relationship.scope` discriminator deliberately avoided — dissolves with the collection, and so does the argument resting it on canonical-form hashing, which ADR 0045 had already superseded; shadow precedence resolves at compile; the authoring half stands: a shadow is a *full* restatement, partial field-level override is impossible, and the same name denotes two independent declarations.
- **0028:** supersedes the clause on "the manifest's resolved relationship shape": a `Relationship` has no carrier of its own, it is an authored declaration that compile consumes, and nothing resolves a relationship by name at plan time; the clause's substance is reinforced rather than weakened — compile derives `JoinType`, an authored one is refused, declared metadata is trusted and never verified.
- **0043:** supersedes the `Relationship.name` and `JoinKeySpec`-discriminant clauses on carrier grounds, as set out in 0050's *The relationship is a process input, and that is why it has no entity* and *`cross` is a symmetric tautology, and an empty basis is unreachable*: a relationship is not a manifest entity, its `ai_context` rides the straight unit it produces, `cross` is explicit rather than an empty key vector, and a failed derivation fails compile.

### Consequences

- **`crates/` convergence is out of scope here** and is handed to the implementation effort: `ResolvedRelationship` (39 lines across 9 files), `ResolvedJoinKey` (28), `scope_local_relationships` (13), `SemanticManifest.relationships`, the private `JoinKeySide` / `ResolvedKeySide` / `ResolvedKeyPair` / `resolve_key_side` / `resolve_key_pair` group, `mint_synthesized_field_id`'s relationship use, and load's `check_relationship_refs` rows 10–11 plus structural check 8. The fifteen `#[allow(dead_code)]` sites of the unwired relationship lane in `compile/seal.rs` are resolved by deletion rather than by wiring.
- **The four `RelationshipJoinKey*` `CompileError` variants survive** and still fire at compile: `on:` is still authored as names, so unknown-on-side, not-a-Dimension, id-mismatch and duplicated are all checked before lowering produces the condition.
- **Checks this hands to the stage that assembles `semantic_sources`**, per ADR 0044, each in that stage's own module documentation: `on_expr_id` and any `filter_expr_id` resolve in `expressions`; a straight unit carries at most one relationship's `description` / `ai_context` and a cumulative unit carries none; and — owed to the join-path-realisation research rather than defined here — every name the condition references resolves in the operands' projections.
- **The persisted-shape change rides ADR 0040's single unshipped `2`.** There is nothing to sequence against: ADR 0043 dissolved the filter-carrier arity change this was once told to coordinate with, stating that "no epoch argument is owed". The field's live name is **manifest version** (ADR 0041's rename); `manifest_epoch` in code is out-of-scope convergence.
- **`CONTEXT.md` gains "Join condition"** and its **Anchor** and **Hop** entries are flagged, not edited: each describes a Joinset with a driving anchor and a hop chain, which binary operands and enumerated paths replace, and that wording belongs to the enumeration decision.

## More Information

### Open questions

- **`ColumnRef` is arguably redundant vocabulary** — `PhysicalLeaf::Column(String)` would do. Recorded as a handoff note, not ruled: it is `semstrait-model`'s type, the manifest owns no vocabulary (ADR 0046), and because `ColumnRef` is `#[serde(transparent)]` over `String` the wire form is *already* a bare scalar, so the cleanup costs no persisted bytes and carries no version consequence.
- **`COMP_E_0414`'s surviving question is not this decision's.** Its canonical-form premise died with ADR 0045's name-seeded identity, and its two apparently-unreachable differentiators — declared `keys`, Joinset-level `filter:` — bear on whether an authored Joinset duplicating an enumerable implicit one is rejected, which is a question about what compile enumerates.
