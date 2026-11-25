---
number: 26
title: The composition tree is inline-only, self-nesting-free, and enforced structurally
status: accepted
date: 2026-08-04
source:
- crates/semstrait-model/src/validation/structure.rs
- crates/semstrait-model/tests/composition_depth.rs
links:
- target: 53
  kind: amendedby
- target: 74
  kind: amendedby
---

# The composition tree is inline-only, self-nesting-free, and enforced structurally

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0053](0053-authoring-direction-is-exclusive-per-root-data-kind.md): the `DataKindRef` retirement and the "`Nested*` forms carry structural fields only" clause are reversed for constituent-first; every other clause — inline-only within interface-first, no self-nesting, the matrix as parse-time unrepresentability, `deny_unknown_fields` as the mechanism — stands as the interface-first direction's rules; the nesting matrix survives references untouched, so same-variant self-nesting stays unrepresentable when those fields hold references instead of inline bodies; the `extras.temporal.<shape>` cascade survives in interface-first and is forbidden in constituent-first.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): in *Decision Outcome*, "`grain:` is forbidden at complex levels" no longer holds for a composition that is a constituent of another composition: it may declare its granularity, which every constituent below it serves, and that granularity applies to every constituent below it, down to a constituent that declares its own; "must be repeated on every Grainset child" holds for a Dataset child, and a composed child carries the granularity that it declares or that ADR 0056 derives (0074's *Every request is planned against its grain first*).

## Context and Problem Statement

The struct/field topology, the YAML surface, the derived three-deep bound and every strategy's recursion assumptions depend on this tree. Admitting shared or self-nested children would turn scopes into a DAG requiring normalization semantics that do not exist, and inheritance changes would silently reinterpret existing models. It is deliberately counterintuitive: a modeling layer that will not let you reuse a declared Dataset in two parents, permits `unionset ⊃ grainset ⊃ unionset` but not `unionset ⊃ unionset`, forbids nested nodes from declaring semantics, and inherits one field of `temporal:` while forbidding its sibling.

## Considered Options

- An inline-only, structurally enforced tree (chosen).
- `DataKindRef`/`ref:` children referencing top-level DataKinds, which would make scopes DAG-shaped.
- Arbitrary same-variant recursion with compile-time flattening and cycle detection (the retired `TD-GRAINSET-NESTED` route).
- One uniform recursive `DataKind` type whose nested nodes may declare their own interface.
- A table-driven validate-stage nesting matrix instead of structural unrepresentability.
- Uniform cascade of the whole `extras` block, including grain.

## Decision Outcome

- Complex children are inlined `Nested*` structs. The `DataKindRef` child-reference mechanism is retired, so there is no cross-Complex sharing and the scope chain stays a strict tree.
- Same-variant self-nesting is forbidden; all cross-variant nesting is legal.
- `Nested*` forms carry structural fields only (no `description`/`ai_context`/`SemanticInterface`).
- Illegal nesting is unrepresentable and surfaces as `deny_unknown_fields` parse errors rather than through a matrix validator.
- Only `extras.temporal.<shape>` cascades to descendants; `grain:` is forbidden at complex levels and must be repeated on every Grainset child.
