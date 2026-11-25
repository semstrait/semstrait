---
number: 28
title: Relationships are bidirectional, semantically authored, trusted, and required for cross-kind references
status: accepted
date: 2026-08-04
source:
- crates/semstrait-model/src/model/types/entities/relationship.rs
- crates/semstrait-manifest/src/compile/relationships.rs
links:
- target: 38
  kind: amendedby
- target: 50
  kind: amendedby
- target: 54
  kind: amendedby
- target: 64
  kind: amendedby
---

# Relationships are bidirectional, semantically authored, trusted, and required for cross-kind references

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0038](0038-relationship-join-basis-is-a-name-list.md): the rejected "implicit join/path inference for cross-kind names" is scoped, not superseded: it ranges over inventing an edge where none is declared, not over deriving a declared edge's basis (`on: auto`).
> - [ADR 0050](0050-a-relationship-is-compile-input-not-a-manifest-entity.md): supersedes the clause on "the manifest's resolved relationship shape": a `Relationship` has no carrier of its own, it is an authored declaration that compile consumes, and nothing resolves a relationship by name at plan time; the clause's substance is reinforced rather than weakened — compile derives `JoinType`, an authored one is refused, declared metadata is trusted and never verified.
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): a catalog that contradicts an authored `Enforced` integrity yields a warning rather than an error; with no catalog, it is trusted as authored.
> - [ADR 0064](0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md): there is no plan-time path search: a request spanning two public sources with no authored composition fails; ADR 0028 forbids inferring the edge and ADR 0054 already enumerated every legal one, so there is nothing left to discover at plan time.

## Context and Problem Statement

The authoring grammar, the derivation matrix, canonical-form hashes and the manifest's resolved relationship shape all encode this decision. Admitting an authored `join_type` or verification would create two sources of truth (or require I/O in compile) and change existing plans; relaxing the path rule would mean inventing join inference.

Accepted costs: every mainstream tool lets you write `LEFT JOIN` and put the FK on the owning entity, `from`/`to` and `ManyToOne` naming imply direction matters, and a wrong cardinality silently yields inflated sums with no warning.

## Considered Options

- Trusted, bidirectional, semantically authored Relationships (chosen).
- An authored `join_type:` field with cardinality demoted to annotation.
- A `Directionality`/`Forward` authoring enum.
- Relationship declarations nested in the `from`-side DataKind.
- Data-scan verification of cardinality/keys with fanout advisories and auto-`DISTINCT`.
- Implicit join/path inference for cross-kind names.

## Decision Outcome

`Relationship`s have two authoring sites: root-level declaration in `SemanticModel.relationships`, and a Joinset-scoped shadow bounded to that Joinset (see ADR 0029 for that mechanism).

- They are unconditionally walkable in both directions; the `Directionality` enum is removed.
- Authors declare `cardinality`/`integrity`/`optional`/`cross_filter`, and compile *derives* `JoinType`.
- Declared keys, cardinality and integrity are trusted planning metadata that semstrait never verifies against data or catalog FK metadata and that never trigger automatic `DISTINCT` or array aggregation.
- Any cross-DataKind Semantics reference is legal only if a unique relationship path exists, pre-validated at compile and walked at plan.

## More Information

### Open questions

None.
