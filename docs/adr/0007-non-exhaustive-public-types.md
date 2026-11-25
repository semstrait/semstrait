---
number: 7
title: "`#[non_exhaustive]` public sum types and growable structs by default"
status: accepted
date: '2026-08-04'
source:
- "crates/semstrait-ir/src/plan/common.rs"
- "crates/semstrait-ir/src/expr.rs"
- "crates/semstrait-common/src/diagnostic.rs"
---

# `#[non_exhaustive]` public sum types and growable structs by default

## Context and Problem Statement

`#[non_exhaustive]` is the workspace's only non-breaking-evolution mechanism: variant and field additions are MINOR *because* of it, so retracting it would turn routine additions into MAJOR breaks. That is worth a surprising cost: consumers lose compile-time exhaustiveness on core domain enums (`JoinType`, `DataType`, `PlanNode`) and must write arms that can never be reached today.

## Considered Options

- `#[non_exhaustive]` by default (chosen).
- Plain exhaustive public enums and structs (the Rust default), trading additive evolution for consumer-side exhaustiveness checking.

## Decision Outcome

Every public classification enum and every public struct whose fields may grow is `#[non_exhaustive]` with a fixed v1 roster.

- Re-exports never relax it.
- A `pub(crate)`→`pub` promotion adds it at the same time.
- Library wildcard arms must return a `Diagnostic<K>` rather than panic.
