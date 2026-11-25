---
number: 47
title: A coverage mask states what a semantic source can serve, and null fill contributes
status: accepted
date: 2026-08-18
source:
- CONTEXT.md
- docs/adr/0036-bit-position-assignment.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
links:
- target: 40
  kind: amends
- target: 48
  kind: amendedby
- target: 49
  kind: amendedby
- target: 59
  kind: amendedby
- target: 37
  kind: amends
---

# A coverage mask states what a semantic source can serve, and null fill contributes

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0048](0048-composite-keysets-are-authored-inside-a-boundary.md): with runtime promotion confined to the advisory, the hedge on the Grainset mask closes: a Grainset's `semantic_bitmask` is the **union of its grains'** plus the bit of each Metric that the Grainset computes, a form rule like the other three.
> - [ADR 0049](0049-a-dataset-names-one-physical-source-as-a-semantic-source.md): the leaf-mask clause is superseded. "A `Dataset` source's mask is the union of what its own bindings can produce" has no subject once a leaf has one source: a reduced leaf's mask is what its single source can serve, and the union is 0047's own Unionset rule, unchanged.
> - [ADR 0059](0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md): the derivation clause only. The four-way list becomes three-way and the third test becomes a catch-all. Everything else in 0047 stands unchanged and is not reopened: a set bit is a serveability statement, null fill contributes, a Unionset's mask is the union rather than the intersection, there is exactly one bit lane, the reverse bit index is not persisted, and a keyset carries no member mask.

## Context and Problem Statement

The design inherited two readings of a set bit. The retired Native/Derived-only rule made it a statement that real source-side data exists; a literal reading of the base schema's "directly and indirectly" would make it a reachability closure over everything obtainable *from* the source.

Both readings fail, for the same reason: each asks the mask to carry a second fact. That rule asked it to carry provenance ("only `Native` and `Derived` mapping coverage contribute bits — `NullFill` / `Metadata` mapping kinds are excluded"), and the closure reading asks it to carry graph reachability. A mask that carries one fact can be tested with one bitwise operation. A mask that carries two cannot be tested at all without knowing which fact the caller meant.

## Considered Options

- The mask as a serveability statement, to which null fill contributes (chosen).
- **The retired rule as written — Native/Derived only.** Rejected because it makes the mask a provenance statement and a candidate filter at once, and the two disagree precisely where a unionset is involved. Its failure is silent: the source is skipped for a query it can answer.
- **The mask as a reachability closure over composition.** Rejected because ADR 0045's flat enumeration already materialises reachability as entities, so the closure would restate the collection's own structure in every mask, and grow monotonically with the model rather than with the source.
- **A persisted four-way provenance carrier (enum row, or one mask lane per variant).** Rejected as a second place to look for a fact ADR 0037's expression edge already determines. Reconsider only for a consumer that needs provenance *hot*.
- **A DataKind or Keyset bit lane.** Rejected: no consumer needs one, and `bit_position` "is explicitly not a persistable identifier" (ADR 0036), so a second lane doubles the renumbering surface for no gain. Kept available by 0036's lane clause.
- **Persisting the reverse bit index as the authority and dropping `bit_position` from the definition.** The honest inverse of the ruling, rejected because the definition is where every other per-semantics fact lives and a top-level name-keyed index would re-key the registry away from `EntityId` order.
- **A persisted keyset member mask.** Rejected; see below. Its subject no longer exists on the carrier.

## Decision Outcome

A set bit in a semantic source's `semantic_bitmask` asserts one thing: **this source, exactly as it stands, can serve that semantics in a query without further composition.** It is a serveability statement — the planner's candidate filter — and it is nothing else.

### "Indirectly" is satisfied by composition being inside the unit

A `Dataset` source's mask is the union of what its own bindings can produce. A composite source's mask is computed **structurally from its operands' masks**, and that is where the earlier base-schema design's "indirectly" is honoured: for a `Joinset{from, to}` the composition is already *inside* the unit, so a semantics contributed by `to` is served by the joinset directly, in the only sense the word can bear.

Reachability through paths the unit *is not* needs no mask at all, because ADR 0045 already flattened it into entities: each path within an authored Joinset's specification becomes its own `semantic_source` with its own mask. Plan-time selection is therefore a mask test over a flat list, and the enumeration — not a widened bit — is what makes indirect availability visible. This is the load-bearing consequence of ADR 0045 for this decision: **the structure absorbed the indirection, so the semantics of a bit could stay narrow.**

Per form, the rule is:

- **Joinset** — the union of the `{from, to}` operands' masks. This is the cumulative-hop rule made structural rather than checked, exactly as recorded; `JoinsetHop.hop_coverage` goes with the hop.
- **Unionset** — the union of the branches' masks (see the next section, which is why it is union and not intersection).
- **Grainset** — the union of the grains' masks, stated as the **default** rather than as a settled form rule, because grain-level availability is entangled with runtime keyset promotion (as the earlier base-schema design had it). A narrowing may be imposed later; until one does, the union stands.

### Null fill contributes a bit, and the mask therefore stops meaning "data exists"

A null-filled semantics sets its bit. A source that answers a query with NULLs has answered it, so excluding the bit would make the source a non-candidate for a query it can serve — a wrong answer produced by an omission, which is the failure mode this codebase is least able to detect. The retired rule's exclusion was correct for a mask that meant *source-side realizability*; it is wrong for a mask that means serveability, and the exclusion was never implemented anyway.

This is the clause that decides the Unionset arithmetic, and it decides it visibly: **with null fill contributing, a unionset's mask is the union of its branches' masks; without, it would be the intersection.** Nothing else in this ADR changes as many plans as that one word.

The price is stated rather than hidden. **A set bit no longer licenses the inference "real data exists here for this semantics."** A caller that needs that fact must read provenance, and the next section says where provenance is. `CONTEXT.md`'s **Coverage** entry keeps its four-way vocabulary — "natively present, null-filled, derived by expression, or supplied from metadata" — and gains no authority over the mask.

### The four-way Coverage axis gets no carrier, because the expression edge already determines it

No `CoverageVariant` enum, no per-`(source, semantics)` provenance row, and no second mask lane per variant. The axis is **derived**, and every input it needs is already persisted by ADR 0037's expression edge: for a `(semantics, source)` pair, no reachable `expr_id` is `NullFill`; a `Column` leaf is `Native`; a `MetadataRef` leaf is `Metadata`; anything composite is `Derived`.

Persisting it would create a second place to look for one fact and a load-time check to keep the two in agreement. ADR 0040 already set the precedent in this crate — the codebase "already stores derived values without marking them" — and ADR 0046 set it again for the schema fork. The axis is glossary vocabulary with a derivation, not a field. `CoverageVariant` is grep-absent from `crates/` today, so nothing retires; what this rules out is *adding* it.

### There is exactly one bit lane, and now it is ratified rather than assumed

Every persisted mask addresses the `semantic_definitions` registry. A set of sources, a set of keysets, or a set of physical sources is an `EntityId` list, never a mask.

ADR 0040 asserted this as fact — "there is no DataKind bit lane — every persisted mask is a coverage view over the **Semantics** registry (ADR 0036)" — and built two design choices on it. That assertion is **confirmed and promoted to a ruling**, so the ADR-on-ADR chain that made it load-bearing is no longer resting on an unexamined sentence. ADR 0036's lane-parameterized clause — "if per-class registries are later added, the identical rule applies independently within each lane and every mask declares the lane it addresses" — stays as dormant headroom, correct and unused. Opening a second lane later is therefore an additive change with a rule already written for it, not a redesign.

### The reverse bit index is not persisted

`semantic_definitions[].bit_position` is the sole persisted authority for a semantics' position. The base schema's top-level `semantic_bitmap_index: [{name, bit}]` is **not persisted**; the name-keyed reverse index is built at load, which is what an earlier rule held for reverse indexes generally.

The schema carried the same fact twice, keyed two different ways — by **name** at the top, while ADR 0036 assigns positions in canonical **`EntityId`** order. One home removes both problems at once: the ordering collision needs no resolution, and an entire class of integrity check (index disagrees with definitions) never has to exist. The collision would otherwise have to be resolved on a schedule, because registry ids *are* names today only as a v1 narrowing.

### A keyset carries no member mask, and the door closes with its subject

ADR 0040 left a keyset member mask "deliberately as an **additive door**, pending the coverage-mask-semantics decision" — pending this ADR. It closes, and the reason is stronger than the judgement 0040 anticipated: **the base-schema revision removed the subject.** A keyset entity is `{id, name}`; membership is attributed from the semantics side, per appearance, on `semantic_definitions[].keysets` and `semantic_overrides[].keysets`. There is no member list on the keyset for a mask to be a copy of.

So the containment test that would want a mask — the earlier base-schema design's joinset derivation, "find keysets on the second dataset whose keysets fully include `x`" — operates on a set derived by scanning attributions, and the **semantic graph** may build a mask for it at setup like any other structure the manifest declines to persist. The re-open condition is recorded so it is a test rather than an open question:

> Persist a keyset member mask only if a consumer must run containment tests **per request**, and only if that consumer cannot build the mask once at semantic-graph build.

Meeting or refusing that condition is not ruled here. All three of ADR 0040's original reasons also still hold: the ids seed identity, a mask cannot *be* an identity seed, and containment is derivable.

### Amends

- **0037:** the coverage axis is **derived**, and every input it needs is already persisted by the expression edge; the retired rule's exclusion was correct for a mask that meant source-side realizability, and it is wrong for a mask that means serveability.
- **0040:** the member-mask additive door closes, because a keyset entity has no member list for a mask to copy.

### Consequences

- **The mask stops being evidence of data.** Any consumer, diagnostic, or advisory that reads a set bit as "real data exists" is wrong under this ruling. None exists today.
- **The code is unconverged and the divergence is now named.** `SemanticBitmask::from_positions` over every semantics the seal walks happens to match this ruling's *union* shape by accident, not by design — it distinguishes nothing because it was never taught to. `DataKind.coverage`, `UnionsetBranch.branch_coverage`, `JoinsetHop.hop_coverage` and `GrainsetLevel.level_coverage` are four carriers where this ruling leaves **one**, per ADR 0043. Convergence belongs to the implementation effort.
- **Checks, placed by owning stage under ADR 0044.** A composite source's mask being the structural union of its operands' is **producer-unreachable** once the producer computes it that way — a check would restate the stage's own arithmetic. What stays is the existing pair: canonical mask form and every set bit resolving in the registry. One check is **forbidden** and worth naming so nobody adds it: "every semantics in a source's mask has an expression for that source" is *false by design* — that absence is exactly null fill.
- **One integrity check never has to be written**, because the reverse index is not persisted.
- **`CONTEXT.md` needs no new term.** **Coverage** keeps its four-way axis as vocabulary; **Bit position** already says "coverage masks address semantics by bit position", which this ADR narrows in meaning without contradicting.
