---
number: 5
title: Warnings survive both `Result` arms; accumulate before compile, fail fast after; policy only at the boundary
status: accepted
date: 2026-08-04
source:
- crates/semstrait-common/src/diagnostic.rs
- crates/semstrait-api/src/error.rs
- crates/semstrait-api/src/orchestrator.rs
links:
- target: 73
  kind: amendedby
- target: 64
  kind: amendedby
---

# Warnings survive both `Result` arms; accumulate before compile, fail fast after; policy only at the boundary

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0064](0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md): in *Decision Outcome*, "`Err((Diagnostic<K>, Diagnostics<K>))` for fail-fast ones" holds for `plan` in two kinds: the warnings take their own kind, `Advisory`, beside the error's kind, `PlanError`, on both arms (0064's *ADR 0018 is untouched*); the rest of 0005 stands.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): `optimize` leaves the stage list.

## Context and Problem Statement

The accumulating and fail-fast policies produce structurally different public return types: flipping a stage's class is an API break for every caller and every `?`/`From` conversion. The byte-identical diagnostic vector is a determinism guarantee that snapshot tests will pin. The design also reads wrong at first glance: doubly-nested tuple `Result`s look over-engineered, a uniform diagnostics posture looks obviously friendlier, and the same finding reported twice looks like a bug.

## Considered Options

- Warnings on both arms, policy only at the boundary (chosen).
- Flat `Result<Output, Vec<Diagnostic>>` discarding warnings on the error arm, with uniform accumulation everywhere.
- A `WarningPolicy` or warning sink threaded into each stage rather than applied at the boundary.
- `(kind, span)`-based de-duplication with severity ordering.

## Decision Outcome

Every stage entry point returns warnings on success *and* on failure: `Ok((Output, Diagnostics<K>))` plus `Err(Diagnostics<K>)` for accumulating stages, and `Err((Diagnostic<K>, Diagnostics<K>))` for fail-fast ones.

- `parse` and `validate` accumulate.
- `compile`, `plan`, `optimize` and `adapt` fail fast.

`WarningPolicy` is applied only on the API return path. The concatenated diagnostic vector is never de-duplicated or severity-sorted.
