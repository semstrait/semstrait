---
number: 29
title: Joinset-scoped Relationship shadow is the sole mechanism for divergent join semantics
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/compile/relationships.rs
- crates/semstrait-model/src/validation/relationship.rs
links:
- target: 50
  kind: amendedby
- target: 54
  kind: amendedby
---

# Joinset-scoped Relationship shadow is the sole mechanism for divergent join semantics

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0050](0050-a-relationship-is-compile-input-not-a-manifest-entity.md): supersedes the persisted half only: the two-lane layout — shadows inline as `scope_local_relationships: Vec<ResolvedRelationship>`, root scope holding non-shadows, the `relationship.scope` discriminator deliberately avoided — dissolves with the collection, and so does the argument resting it on canonical-form hashing, which ADR 0045 had already superseded; shadow precedence resolves at compile; the authoring half stands: a shadow is a *full* restatement, partial field-level override is impossible, and the same name denotes two independent declarations.
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): `COMP_E_0414`'s canonical form lost its subject; an authored Joinset duplicating an enumerable implicit one is *not* rejected, because the two carry different names, so they are different entries with different ids and nothing collides.

## Context and Problem Statement

The two-lane persisted layout is manifest schema, and shadow visibility is baked into path resolution and canonical-form hashing. So the decision is fixed despite being surprising: changing one field of a join requires restating an entire Relationship, and the same Relationship *name* denotes two independent declarations with possibly different cardinality — an independent data-shape claim, not an override.

## Considered Options

- A Joinset-scoped full shadow (chosen).
- The `JoinTypeOverrides` + `HopPosition(u16)` per-hop override map on `JoinsetDataKind` (shipped, then retired in the relationship-block rebase, item K, 2026-05-12).
- Partial field-level override of a root-level Relationship.

## Decision Outcome

A `Joinset` MAY declare `Relationship`s in its own `relationships:` block, shadowing a same-named root-level entry within that Joinset's scope only. The shadow is a *full* Relationship with `cardinality`, `integrity`, `optional`, `cross_filter`, `on` and `filter` all restated, so partial field-level override is impossible; `JoinsetBody.relationships` parallels `SemanticModel.relationships` exactly.

Shadows persist inline on the Joinset variant as `scope_local_relationships: Vec<ResolvedRelationship>`, while root-scope `relationships` holds non-shadow entries only, deliberately avoiding a `relationship.scope` discriminator. A scope-local shadow is also one of the four differentiators that keep an explicit Joinset's canonical form distinct from an enumerable implicit one (`COMP_E_0414`).
