---
number: 34
title: The composed surface is its own origin-agnostic type, and NULL-fill belongs to Unionset alone
status: superseded
date: 2026-08-04
source:
- crates/semstrait-manifest/src/manifest/coverage.rs
- crates/semstrait-manifest/src/compile/assemble.rs
- crates/semstrait-manifest/src/compile/expressions/projection.rs
links:
- target: 45
  kind: supersededby
- target: 48
  kind: supersededby
- target: 53
  kind: supersededby
- target: 54
  kind: supersededby
- target: 65
  kind: supersededby
- target: 73
  kind: supersededby
- target: 74
  kind: supersededby
---

# The composed surface is its own origin-agnostic type, and NULL-fill belongs to Unionset alone

> **Superseded by [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md).** Every reading source, a Joinset and a Unionset alike, owns one semantic signature per name it serves, and a Grainset is a routing surface that owns none.
>
> **Superseded by [ADR 0045](0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md).** An authored unit and a compile-synthesised one are the same kind of entry; `DataKindOrigin` "is deleted rather than re-homed, because implicitness is not a planner fact", and what survives is `public` plus `originators`.
>
> **Superseded by [ADR 0053](0053-authoring-direction-is-exclusive-per-root-data-kind.md).** On the `constituent.name` clause only: `constituent.name` is deleted outright, and two constituents that expose one name give one fused field, never a qualified pair, and the fused name must be pooled.
>
> **Superseded by [ADR 0048](0048-composite-keysets-are-authored-inside-a-boundary.md).** A Joinset, an enumerated one included, derives its keysets from its two operands by the pair, determination and minimality rules.
>
> **Superseded by [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md).** There is no authored anchor: "Anchor retires as a term".
>
> **Superseded by [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md).** The Strategies and the rules read the manifest only through the read model, and each lookup returns a value the planner owns, never a reference to a manifest struct, "so a later graph (ADR 0011) can use its own shapes".
>
> **Superseded by [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md).** The driver writes `NullFill` as a typed NULL for a Unionset branch that does not map a name, once the manifest lets a branch serve only the names it maps; "NULL fill on a Joinset" is a considered option 0074 did not take, and a Joinset operand carries no NULL-filled name.

## Context and Problem Statement

This is recorded because it retires an enum variant and a whole composition path from the manifest layer, and because provenance shape drives adapter NULL handling and key-dependent rewrites: unifying or re-admitting those forks every consumer.

It also reads against expectation: invariant D5 says every Complex variant exposes a composed interface yet Unionset visibly does not, the unified-surface prose is written in Unionset terms (NULL-fill, `UNION ALL`), both composition kinds produce NULLs but only one records it, and there is one merge rule per kind rather than a single uniform rule.

## Considered Options

- An origin-agnostic composed surface (chosen).
- A `SemanticInterface` enum with a `Composed` variant, or a wrapper holding `inner:`.
- A separate `CompositionKind::Relationship` for implicit walks.
- Keeping `CompositionKind::Unionset` routed through the unified-surface machinery (the Round-1/2 shape).
- Uniform name unification, or heuristic election, across all kinds.
- Empty keys for implicit Joinsets.
- Symmetrical structural `NullFill` for outer joins.

## Decision Outcome

`ComposedSemanticInterface` is a distinct struct that converges with `SemanticInterface` only through the `SemanticsView` trait. One struct plus an `Origin::{Explicit, Implicit{id}}` axis covers both author-declared and enumerated compositions, and plan-time semantics never branch on origin.

- `CompositionKind` is narrowed to `{Joinset, Grainset}`; Unionset exposes a bare `SemanticInterface` with per-child Coverage.
- Colliding names on a Joinset surface exist only as `constituent.name`, while Unionset and Grainset unify them.
- Implicit Joinsets derive keys from the anchor.
- `FieldOwnership::NullFill` is used only by Unionset; outer-join nullability is carried implicitly by `JoinType`.
