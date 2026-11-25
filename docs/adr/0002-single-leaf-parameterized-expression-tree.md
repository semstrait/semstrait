---
number: 2
title: One leaf-parameterized expression tree, with two wire surfaces
status: accepted
date: 2026-08-04
source:
- crates/semstrait-sql/src/expr
- crates/semstrait-ir/src/expr.rs
- crates/semstrait-model/src/model/serde/de.rs
links:
- target: 85
  kind: amendedby
- target: 88
  kind: amendedby
---

# One leaf-parameterized expression tree, with two wire surfaces

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): in its *Decision Outcome*, the first sentence becomes: a single `Expr<L, A>` declares every structural variant once, with `Literal` and `Column(NamedReference)` as nodes of the tree, and `L` holds only a stage's own leaf, as 0085's rules 1 and 2 give them. The forbidden combinations stay unrepresentable. The rest of 0002 stands.
> - [ADR 0088](0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md): in its *Decision Outcome*, "The derived adjacently-tagged serde form is machine transport only" becomes: the tree's JSON form, which `semstrait-sql` writes and reads, in which a call's tag is its function type and a leaf is written in its own form (ADR 0088, rules 1 to 3), is machine transport only. The rest of 0002 stands.

## Context and Problem Statement

The expression tree is hard to undo and easy to misread. Every traversal, fold, planner pass and adapter renderer is written against it, and both wire surfaces are contractual: the persisted bytes and the author-facing schema. A reader would otherwise expect one concrete `Expr` enum per stage plus runtime `try_into_physical` checks, with the serde derive *being* the authoring format.

## Considered Options

- One leaf-parameterized `Expr<L>` with two wire surfaces (chosen).
- Parallel per-stage structural enums (`ir::SemanticExpr` / a re-declared `ir::PhysicalExpr`) with runtime conversion checks — duplicates every variant and demotes static guarantees to validation.
- `#[serde(tag, content)]` plus derived `JsonSchema` as the single authoring representation, along with the now-deleted `ExprSource<L>` wrapper — couples the author-facing surface to internal transport shape.

## Decision Outcome

A single `Expr<L>` declares every structural variant once and is instantiated per stage by leaf set (`UnresolvedSemanticExpr`, `UnresolvedPhysicalExpr`, `ir::PhysicalExpr = Expr<PhysicalLeaf>`). Forbidden combinations — semantic leaves inside physical trees, authored `Parameter`, unauthorable `Window` — are therefore unrepresentable rather than validated at runtime.

The derived adjacently-tagged serde form is machine transport only. The authoring YAML / JSON-Schema projection is hand-rolled per field.
