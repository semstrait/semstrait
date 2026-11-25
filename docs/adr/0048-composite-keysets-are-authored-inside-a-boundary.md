---
number: 48
title: A composite's keyset is authored inside a boundary; only a Joinset derives one
status: accepted
date: 2026-08-20
source:
- CONTEXT.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
links:
- target: 33
  kind: amends
- target: 39
  kind: amends
- target: 40
  kind: amends
- target: 43
  kind: amends
- target: 47
  kind: amends
- target: 54
  kind: amendedby
- target: 65
  kind: amendedby
- target: 74
  kind: amendedby
- target: 84
  kind: amendedby
- target: 34
  kind: supersedes
---

# A composite's keyset is authored inside a boundary; only a Joinset derives one

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): implicit-Joinset enumeration withholds nothing on cost grounds and its legality reads `integrity` alone, so the `(cardinality, integrity, preserved side)` triple is not what prunes it.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): a composite's keysets ride its own semantic signatures, not `semantic_overrides` rows, and a definition owns no keysets; a Grainset carries its authored keysets on `RoleDefinition::Grainset`.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): the fan-out advisory reads no runtime keyset: fan-out is deduplicated by the fanned operand's guaranteed keyset, and the advisory is raised only where that operand has none, and so has no runtime keyset either.
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Grain groups are dropped, and a Grainset emits no implicit Unionset*, "a Grainset's `semantic_bitmask` is the **union of its grains'**" reads: the union of its grains' and the bit of each Metric that it computes (0084's rule 8); the rest of 0048 stands (0084's *A Metric that no child maps whole*).
## Context and Problem Statement

The question is how a composite semantic source gets its keysets. Conflating the two mechanisms below is what made every earlier framing of this question wrong.

## Considered Options

- Derivation for a Joinset only, plus an authored keyset checked against a per-form boundary (chosen).
- **Derive composite keysets on read**, structurally from the operands, as ADR 0047 does for masks. It keeps the artifact smaller and needs no override rows. It puts the operand walk back into every consumer, which is what flattening exists to remove, and it leaves the earlier design's offline/runtime distinction meaningless.
- **The earlier design's X/Y containment rule alone.** It is the human's own first formulation and it is sound where it fires. It misses the many-to-one fact-to-dimension join entirely, and it drops keysets that genuinely determine a composed row.
- **Compliance as equality with a boundary element, or as a superset of one.** Equality rejects true narrower claims; superset was this session's first recommendation and inverts what a ceiling means.
- **Narrowing a boundary by the preserved side.** Physically more honest for outer joins. It makes the boundary read three more inputs to forbid a claim the model cannot actually refute, and keysets are unverified advice.
- **Grain groups with an implicit Unionset per group.** It was the plan of record and it saves the modeler real authoring work. It cannot produce a keyset, so it produces a surface nothing can join to.
- **Runtime keyset promotion as an input to derivation or basis resolution.** It would let a Grainset answer rollups it cannot answer offline. It makes the artifact's composition depend on a request, which contradicts a content-addressed manifest (ADR 0041) — hence the guaranteed/runtime split instead.
- **One arithmetic for all three forms.** Uniformity is worth a lot in a document this size. The Grainset asks a different question, and forcing union arithmetic on it would admit claims no single grain can honour.

## Decision Outcome

A composite semantic source gets its keysets by **two** mechanisms:

- **Derivation** — compile mints the keyset itself from the operands. Only a **Joinset** derives, because only a join guarantees a composed determinant.
- **Boundary** — the author declares the keyset on the composite, and the boundary is the ceiling on what may be declared. It is computable for every form, it is **subset-closed** (every subset of a boundary element is admissible), and it never derives anything: it only rejects a declaration the operands physically contradict.

The two answer different questions. Derivation asks *what do the children guarantee*. A boundary asks *what could the children never contradict*. They may disagree in both directions without inconsistency, and `Boundary ⊇ Derived` is not asserted as an invariant, since the two are computed from different inputs.

### A keyset is a determinant, and that is why the arithmetic is set algebra

Within one combination of a keyset's dimensions, nothing outside it can diverge: with keyset `[campaign_id, country]`, `age` has one value per `(campaign_id, country)` pair, and `age` could only vary if it joined the keyset. This is `CONTEXT.md`'s assumed uniqueness read from the other end — the same claim, no new fact — and it is what licenses composing keysets by union and intersection rather than by a bespoke rule per form.

### Boundaries, per form

Each form's boundary is a set of **elements**; an authored keyset complies when its members are a **subset of some element**. Compliance is subset and not equality or superset, because a boundary is an upper bound: a narrower group is a *stronger* identity claim, which the operands cannot refute, while a group reaching outside every element names a dimension no child keyset ever claimed and is therefore contradicted.

- **Unionset** — the cartesian product over **all** branches, choosing one keyset per branch; each element is the **union** of the chosen groups. Two branches may carry the same values for one branch's keyset, so identity across the stack needs both branches' determinants at once.
- **Joinset** — the same product over the two operands, each element the union of the chosen groups. Shared dimensions collapse under set union, which is correct and means a Joinset element may be narrower than a Unionset element built from groups of the same size. A Joinset is binary (ADR 0045), so three-way composition **recurses**: an inner Joinset's keysets are an outer Joinset's operand keysets. A linear chain gives the same answer either way.
- **Grainset** — the product over **all** grains, choosing one keyset per grain, each element the **intersection** of the chosen groups; an element that comes out empty is dropped. Given grains offering `[campaign_id, country, region]`, `[campaign_id, age]` and `[campaign_id, country]`, the boundary is `[campaign_id]`: the author may skip the declaration, or claim at most `campaign_id`.

**The union/intersection asymmetry is a different question, not an inconsistency.** A Unionset or Joinset keyset answers *what identifies a composed row*, so determinants accumulate. A Grainset keyset must be usable at **whichever grain answers**, so a member has to exist in every grain, and the common part is the ceiling.

**`on: cross` has no boundary at all.** With no basis there is nothing to reason from, so the author is unconstrained and nothing is derived. This is stated rather than derived: the product arithmetic would happily produce elements for a cross join, and it must not be applied there.

**A preserved side does not narrow a boundary.** On an outer join a claim drawn from the non-preserved side is false in practice once unmatched rows carry NULLs. The boundary still admits it, because a boundary states what the *model* cannot contradict and a keyset is unverified advice whose author may know their data is total where the model does not say so. The cost is disclosed rather than designed away.

### Derivation, per form

- **Unionset — nothing.** A vertical union cannot guarantee a composed determinant, so authoring within the boundary is the only route. This is why the earlier base-schema design's authored-first rule stands and ADR 0039's "the intersection of its branches'" does not: intersecting branch member sets would synthesise a group *no author ever claimed unique*, which is a fabricated identity claim rather than a derived one.
- **Grainset — nothing.** Same reason, one axis over.
- **Joinset — three rules.**
  1. **Pair rule.** For operand keysets `A` and `B`, `A ∪ B` is a keyset of the Joinset, because each composed row is exactly one distinct pair of operand rows. These are the boundary elements.
  2. **Determination rule.** If the join basis contains, **in full**, some keyset of one side, then a row of the other side matches at most one row of that side. That side is *determined by the join*, so **every keyset of the other side stands alone** as a keyset of the Joinset.
  3. **Minimality.** Prune any derived group that is a superset of another derived group; a superset is a weaker restatement of the same claim.

  Both rules assume the match happens, so `(cardinality, integrity, preserved side)` gates them: with a preserved side, derive only groups lying wholly inside the preserved side. That is the same triple that prunes implicit-Joinset enumeration; it is stated here once and cited there.

**The earlier base-schema design's X/Y rule is the special case, not the rule.** Read literally — take the keyset `x` the join columns sit inside, keep the other side's keysets that fully include `x` — it derives **nothing** for the commonest join in any model: `orders[order_id | customer_id, ordered_at]` joined many-to-one to `customers[customer_id | email]` on `[customer_id]`, where every composed row is exactly one order and `[order_id]` identifies it. The determination rule finds `[order_id]`. On that design's own example (`first_dataset[id, customer_id]` joined on `[id, customer_id]` to `second_dataset[id, customer_id, event_id][event_id, country]`) the X/Y rule keeps `y` and drops `z`; `z` is a keyset, so it determines its whole row and therefore the single matching row on the other side, so `z` identifies a composed row and dropping it discarded a true claim. When `x` *is* the basis, X/Y is the pair rule with shared names collapsed — `y = x ∪ y` because `x ⊆ y`.

### Guaranteed and runtime keysets

A **guaranteed keyset** is what the manifest holds — authored or derived — and it is the *minimum* a semantic source offers. A **runtime keyset** is a guaranteed keyset extended by what a request discovers. The split exists because one consumer wants the looser set and two consumers must have the stricter one:

- `Auto` basis resolution and a Grainset's implicit cross-grain Joinsets read **guaranteed keysets only**;
- the fan-out / cartesian **advisory** reads runtime keysets, and that is the whole of runtime promotion's remaining job.

The earlier base-schema design's runtime promotion therefore survives, narrowed to the advisory and removed from every path that shapes the artifact.

### A keyset constrains the semantic side, and never the data

ADR 0039's posture — *a keyset advises; it never constrains* — is **narrowed, not kept whole**. There is still no verification and no claim about rows: nothing reads a keyset to reject a model against data. But a keyset is load-bearing input to two artifact-shaping consumers, `Auto` basis resolution and a Grainset's implicit cross-grain Joinsets, so a wrong keyset yields a **wrong plan** rather than a rejected model. Everywhere else it stays advisory and a utility for later planning guard rails. ADR 0039's *"a relationship resolves its join basis from its own `on:` specification and never from keyness"* is also re-scoped: **a declared basis never reads keyness**, while a derived basis and a Grainset's cross-grain joins read guaranteed keysets. The ladder — how to choose when several keysets qualify — is out of scope here.

### Grain groups are dropped, and a Grainset emits no implicit Unionset

The Grainset design was to form **grain groups** (same temporal grain together; for x-axis grain types, identical dimensions on both sides) and emit one implicit Unionset per group. That is withdrawn, and the term retires with it. The reason is a keyset fact: an implicitly-formed Unionset derives no keyset, so it would have no identity for anything downstream to join to — the same limit this ADR records for authored Unionsets, reached where no author can be asked. The modeler authors Unionsets explicitly instead, with their own keysets, and places them in the Grainset; `GRAINSET_ALLOWED_CHILDREN` is `[Dataset, Unionset, Joinset]` already, so no model change is forced.

What a Grainset still emits is its grains as their own typed semantic sources, plus the **implicit cross-grain Joinsets** among them, whose enumeration and rollup-direction bound (a daily grain rolls up to monthly, not the reverse; likewise on x-axis grain types) are not ruled here. Their basis reads the **grains'** guaranteed keysets, never the Grainset's own: a keyset is a concern of a source's **parent**, so a Grainset's keyset serves whatever composes the Grainset and says nothing about what the Grainset does internally. A Grainset whose grains carry no keyset simply offers no rollup composition and each grain remains individually routable, which is a legitimate model and needs no diagnostic.

With runtime promotion confined to the advisory, ADR 0047's hedge on the Grainset mask closes: a Grainset's `semantic_bitmask` is the **union of its grains'**, a form rule like the other three.

### The carrier

Keysets are **materialized**, not derived on read: a composite's keysets are written as `semantic_overrides` rows on the composite's own `data_kind_id`. These are the first overrides whose reason for existing is keyness alone, with no `expr_id`. Deriving on read was rejected because it reinstates the operand walk that ADR 0045's flat enumeration exists to remove, and because the earlier design's offline/runtime distinction only means anything if offline means persisted. The rules apply **uniformly to implicit sources**, including every enumerated implicit Joinset: a flat list where only some sources answer "what identifies a row" forces the planner back into a walk. The volume that adds is an input to the ceiling question.

Attribution follows the base schema revision: a keyset entity is `{id, name}` with no members, `semantic_definitions[].keysets` carries the default and `semantic_overrides[].keysets` the per-appearance attribution. An override's `keysets` **replaces** the default and never unions with it — the shape `expr_id` already has — because a union would make it impossible for one appearance *not* to be a key member, which is the silent wrongness ADR 0039 exists to retire. An absent field inherits; an empty list stays refused (ADR 0039's `keysets: []` precedent).

**A derived keyset is named by ADR 0045's rule for implicit sources**: after its origin of compilation, qualified by the originator. The determination rule's output keeps the contributing keyset's name qualified by its side; the pair rule's output is the underscore-join of both qualified names in operand order. This is load-bearing rather than cosmetic, because the id is minted over `(semantic_source_id, keyset name)`: a name collision is an **id** collision, which merges two different identity claims into one entry. A derived name colliding with an authored keyset name on the same composite is therefore a hard error — the hazard ADR 0045 flagged for sources and left without a rule.

**ADR 0040 is superseded on carrier grounds, scoped to the carrier.** Its `members`, its `data_kinds` and its `(name, sorted members)` dedup all go: there are no members on the entity to seed with, and the seed the human ruled is precisely 0040's own rejected alternative *"Seeding the id per DataKind"*, whose rejected cost — no cross-source sharing, one roster entry per source per name — is now **accepted deliberately**, since a source-scoped seed makes cross-data-kind dedup impossible by construction rather than merely unimplemented. The minting mechanism itself is not ruled here. Everything else in 0040 stands: keyness advises, `KeysetName` is its own newtype, there is no DataKind bit lane.

### Diagnostics

Two hard `CompileError`s, both raised by the stage that assembles `semantic_sources` and documented in that stage's own module documentation rather than in any roster (ADR 0044):

- an authored keyset that falls outside every boundary element, naming the composite, the rejected group and the offending dimension;
- a derived keyset name colliding with an authored one on the same composite.

Neither belongs to model `validate`: both need the composed operands, which the model does not have.

### Amends

- **0033:** a Grainset emits no implicit Unionset; its cross-grain Joinsets read the grains' guaranteed keysets, and a Grainset whose grains carry no keyset is legal and "needs no diagnostic".
- **0034:** A Joinset, an enumerated one included, derives its keysets from its two operands by the pair, determination and minimality rules.
- **0039:** the three derivation clauses are all superseded: the Unionset intersection by rule, the Grainset per-level keyset and the Joinset anchor propagation by loss of subject under ADR 0045's binary carrier and ADR 0048's rules. An authored composite keyset is checked against a boundary. The posture *a keyset advises; it never constrains* is narrowed, not kept whole, and *"A relationship resolves its join basis from its own `on:` specification and never from keyness"* is re-scoped, as ADR 0048's *A keyset constrains the semantic side, and never the data* gives.
- **0040:** superseded on carrier grounds, scoped to the carrier. A keyset is `{id, name}`, minted over `(semantic_source_id, keyset name)`; its `members`, its `data_kinds` and its `(name, sorted members)` dedup all go, because there are no members on the entity to seed with. The seed the human ruled is precisely the rejected alternative *"Seeding the id per DataKind"*, whose rejected cost — no cross-source sharing, one roster entry per source per name — is now accepted deliberately, since a source-scoped seed makes cross-data-kind dedup impossible by construction rather than merely unimplemented. An authored composite keyset is boundary-checked. Everything else stands: keyness advises, `KeysetName` is its own newtype, there is no DataKind bit lane.
- **0043:** a keyset entity is `{id, name}`, minted over `(semantic_source_id, keyset name)`, so the roster is no longer deduplicated across data kinds.
- **0047:** with runtime promotion confined to the advisory, the hedge on the Grainset mask closes: a Grainset's `semantic_bitmask` is the **union of its grains'** plus the bit of each Metric that the Grainset computes, a form rule like the other three.

### Consequences

- **For an inner Joinset, derivation reaches the whole boundary and below it**, so an authored Joinset keyset is only for claims narrower than the determination rule reaches. The Unionset is the mirror image: derivation reaches nothing and authoring is the only route. The Grainset sits with the Unionset.
- **`CONTEXT.md` gains three terms** — *Keyset boundary*, *Guaranteed keyset*, *Runtime keyset* — and its **Keyset** entry gains the narrowed posture. Its **Implicit Unionset** entry loses the "same-grain Grainset children folded together" clause, which grain groups were the mechanism for.
- **ADR 0039's three derivation clauses are all superseded**: the Unionset intersection by rule, the Grainset per-level keyset and the Joinset anchor propagation by loss of subject under ADR 0045's binary carrier and this ADR's rules.
- **The earlier base-schema design's composite-keyset derivation rules are extended rather than corrected**, its authored-first rule stands, its runtime promotion is narrowed to the advisory, and its implicit-Unionset half is lost.
- **Handoff to the implementation effort.** The single-leaf guard that rejects every complex form is **legacy**: it is to be removed in favour of producing every complex data kind end to end in the manifest, with these keyset rules landing as part of that work rather than after it — a composite with an empty keyset roster is a silently wrong artifact. `mint_keysets()`, which today sits inside `seal_dataset` and scans one data kind's own dimension sites, cannot express any rule here and is replaced rather than extended. `assign_to_grain_groups` in the excluded `semstrait-planner` is the only occurrence of the retired grain-group concept and is deleted when that crate is ported.
