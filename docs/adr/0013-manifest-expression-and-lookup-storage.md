---
number: 13
title: 'Manifest expression storage: physical pool only, structural `ExprLayer`, plan-ready denormalized lookups'
status: superseded
date: 2026-08-04
source:
- crates/semstrait-manifest/src/manifest/expression.rs
- crates/semstrait-manifest/src/manifest/expression_layer.rs
- crates/semstrait-manifest/src/compile/expressions/mod.rs
links:
- target: 37
  kind: amendedby
- target: 43
  kind: amendedby
- target: 50
  kind: amendedby
- target: 65
  kind: supersededby
---

# Manifest expression storage: physical pool only, structural `ExprLayer`, plan-ready denormalized lookups

> **Superseded by [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md).** Superseded on the pool's shape and on id derivation: the content-deduplicated flat-tree pool addressed by content becomes a node store of positional entries, and "the same `PhysicalExpr` may be stored twice" stops being an accepted anomaly and becomes the rule. Its structural `ExprLayer` survives, on the entry, computed from the subtree. One semantic signature per semantic source and definition owns exactly one complete, fully physical expression tree; the Joinset source owns its join condition or its residual; the `Field` role and its minter are removed.
>
> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0037](0037-semantics-bind-to-pooled-physical-expressions.md): `SemanticBinding.mapping` becomes `BTreeMap<EntityId, PhysicalExprId>`; every covered semantics, a bare column or literal included, interns into the pool and is referenced by handle.
> - [ADR 0043](0043-authored-primitives-get-typed-carriers.md): `SemanticBinding` retires, and `ResolvedExprMeta` with it; per-source variation lives on the expression.
> - [ADR 0050](0050-a-relationship-is-compile-input-not-a-manifest-entity.md): `ResolvedJoinKey` retires; a joinset's `on` and `filter` are `PhysicalExprId` handles into the pool, and a computed key needs no synthesised `Field`.

## Context and Problem Statement

Expression storage is both the persisted layout and the planner's hot-path lookup shape, so re-normalizing, re-adding a semantic pool or inlining join expressions would change the artifact, its load checks and every consumer.

## Considered Options

- A physical pool only, a structural `ExprLayer` and denormalized lookups (chosen).
- The original dual semantic + physical pools with a `SemanticExprId` lane.
- Deriving `ExprLayer` from the authoring `SemanticRole` (Dimension ⇒ scalar, Measure ⇒ aggregate) or re-deriving it at plan time.
- One enum-keyed mapping matched in the hot path instead of four disjoint flat maps.
- Storing join-key sides directly as `PhysicalExprId`s.

## Decision Outcome

Compile lowers every authored expression and persists only the content-deduplicated `PhysicalExpr` pool addressed by `PhysicalExprId`. Each entry carries an `ExprLayer` (`Scalar`/`Aggregate`/`PostAggregate`) computed purely from tree structure.

- Per-`(Semantics, Binding)` metadata lives on the binding.
- Mappings are denormalized into four disjoint flat maps (`columns`/`literals`/`computed`/`metadata`, duplicating pool entries).
- Relationship join keys are stored as semantic-`EntityId` pairs rather than inline expressions; computed keys become synthesized implicit `Field` semantics.

### Consequences

Deliberately accepted, even where they read as anomalies:

- the same `PhysicalExpr` may be stored twice;
- a Dimension can classify as `PostAggregate`;
- `from: lower(email)` persists as a phantom `Field` semantic;
- the pre-lowering form needed for round-trip or diagnostics is simply gone.
