---
number: 21
title: 'Portability floor: intersection function catalog, no semstrait-owned typing'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-sql/src/functions/catalog/mod.rs
- crates/semstrait-sql/src/datatype/family.rs
- crates/semstrait-manifest/src/compile/expressions/lower.rs
links:
- target: 72
  kind: amendedby
- target: 87
  kind: amendedby
- target: 89
  kind: amendedby
- target: 91
  kind: amendedby
- target: 92
  kind: amendedby
---

# Portability floor: intersection function catalog, no semstrait-owned typing

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0072](0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md): a canonical function may take a calendar-spec argument — the request's timezone or week start — that no engine takes natively, when every engine's rules write it in that engine's own functions (ADR 0068). The intersection floor stands for every other function and argument.
> - [ADR 0087](0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md): its decision "intersection function catalog", in its title, is replaced by ADR 0087, rules 1 and 4, and the sections that only apply it read as those rules give: in *Context and Problem Statement*, "omits ubiquitous SQL functions"; in *Considered Options*, the chosen option's "Intersection floor"; in *Decision Outcome*, the first sentence and "Everything else (…) is adapter-extended"; and, as ADR 0072 amends it, "the intersection floor stands for every other function and argument". ADR 0072's calendar-spec argument is one case of ADR 0087's rule 1: the DataFusion rules write it in DataFusion's own functions. The rest of 0021 stands.
> - [ADR 0089](0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md): in its *Decision Outcome*, "a narrowing physical→declared cast is a compile error demanding an explicit author cast" becomes the cast to a semantics' declared type that ADR 0089, rules 17 and 18, give. The rest of 0021 stands.
> - [ADR 0091](0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md): in its *Decision Outcome*, "NULL propagation is delegated to the engine" becomes, for a function, what ADR 0091, rule 8, says of when its result is NULL, and "Signatures require exact `DataType` equality" becomes: a parameter takes the type its declaration names (ADR 0091, rule 4; ADR 0086, rule 8). The rest of 0021 stands.
> - [ADR 0092](0092-an-operation-node-has-a-declaration-and-checks-its-operands-by-type-family-and-a-type-system-derives-the-result-types.md): in its *Decision Outcome*, "NULL propagation is delegated to the engine" no longer holds for an operation node: its canonical declaration says when its result is NULL (ADR 0092, rule 7), and where an engine differs, its dialect declaration says so. "`BinaryOp` returns `SameAs(left)`" and "cross-operand compatibility is unvalidated" become: an operation checks each operand by its type family (ADR 0092, rule 2), and the operation's declared return rule names the result. "Signatures require exact `DataType` equality" becomes, for an operation: an operation checks its operands by type family. "Semstrait publishes no promotion or coercion lattice" becomes: `Promote` reads the widening order of ADR 0086 rule 9 to name a result type, and with a decimal operand the type system names it (ADR 0092, rule 3). Compile inserts no cast between an operation's operands. "Unification is strict equality over width-differentiated types" stands for `Case` and `coalesce`. The rest of 0021 stands.

## Context and Problem Statement

Adding a lattice or widening the catalog later changes emitted SQL and therefore the *results* of already-authored models, while narrowing either breaks them outright. The posture is also counterintuitive: a portability-focused semantic layer with its own canonical `DataType` system that nonetheless refuses to type-check arithmetic, lets `Integer` on DataFusion be `Double` on DuckDB, rejects `integer` vs `long` unification, and omits ubiquitous SQL functions.

## Considered Options

- Intersection floor, no semstrait-owned typing (chosen).
- A union function catalog with per-engine emulation and portability flags, instead of the intersection floor.
- A SQL:2016-style canonical promotion lattice with compile-inserted widening casts and mismatch errors.
- A single abstract `Integer`/`Number` type with adapter-side width inference.
- Silent narrowing/truncation on physical→declared mismatch.

## Decision Outcome

A function is canonical only if DataFusion ∩ DuckDB ∩ Spark support it natively or by trivial remap. Everything else (`initcap`, `percentile_disc`, format-string `to_date`, 2-arg `date_diff`, 4-arg `regexp_replace`) is adapter-extended, and `FunctionSpec` carries no portability flag.

Semstrait publishes no promotion or coercion lattice:

- signatures require exact `DataType` equality;
- `BinaryOp` returns `SameAs(left)`;
- cross-operand compatibility is unvalidated;
- NULL propagation is delegated to the engine;
- unification is strict equality over width-differentiated types;
- a narrowing physical→declared cast is a compile error demanding an explicit author cast.
