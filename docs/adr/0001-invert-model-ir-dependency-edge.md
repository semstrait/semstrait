---
number: 1
title: Invert the model↔IR crate edge and home shared machinery at the DAG root
status: accepted
date: 2026-08-04
source:
- crates/semstrait-model/Cargo.toml
- crates/semstrait-ir/Cargo.toml
- crates/semstrait-common/src/lib.rs
links:
- target: 85
  kind: amendedby
- target: 94
  kind: amendedby
---

# Invert the model↔IR crate edge and home shared machinery at the DAG root

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): in its *Decision Outcome*, the first sentence becomes: `semstrait-sql` owns the shared expression vocabulary, the types that 0085's rule 3 moves to it, and `semstrait-model`, `semstrait-ir` and `semstrait-manifest` each depend on it directly. `semstrait-model` keeps what rule 3 keeps in it, and `semstrait-ir` still depends on `semstrait-model`, for `SemanticsName` only (ADR 0090, rule 3). The rest of 0001 stands.
> - [ADR 0094](0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md): *Decision Outcome* as ADR 0085 amended it ("`semstrait-ir` still depends on `semstrait-model`, for `SemanticsName` only (ADR 0090, rule 3)"): `semstrait-ir` depends on no model crate (rule 9), and `Grain` is `semstrait-common`'s `Granularity` (rule 3).

## Context and Problem Statement

The direction of the model↔IR crate edge is effectively irreversible: it fixes the home of the most-referenced types in the system, so re-inverting it would relocate dozens of public types across six crates as a MAJOR break. It is also counter-intuitive: readers assume "IR" is the substrate the authoring model is built on, and expect `UnresolvedSemanticExpr::resolve()` to be an inherent method rather than a free function elsewhere.

## Considered Options

- `semstrait-ir` depends on `semstrait-model` (chosen).
- The earlier ratified `model → ir` edge, with the expression vocabulary (and `DataType`/`Schema`) owned by `semstrait-ir`/`semstrait-common`, plus an inherent `impl` carrying the lowering. It forced the authoring crate to depend on the planning vocabulary and gave the lowering fold a home in a crate that cannot legitimately see both layers.

## Decision Outcome

`semstrait-model` owns the shared authoring vocabulary — the `Expr<L>` family, leaf sets, support enums, `Literal`/`ColumnRef`, `DataType`/`Grain`, and the `CanonicalFn` name — and `semstrait-ir` depends *upward* on it.

Stage-agnostic machinery (`Tree`/`Visitor`/`Rewriter`, the narrow `ValidateError`) sinks to `semstrait-common`.

Cross-layer folds such as `resolve` are free functions in `semstrait-manifest`, the only crate permitted to see both sides.
