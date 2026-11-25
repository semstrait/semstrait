---
number: 14
title: Enumerate every bounded composition eagerly at graph build; plan time is pure lookup
status: superseded
date: 2026-08-04
source:
- crates/semstrait-manifest/src/compile/assemble.rs
- crates/semstrait-manifest/tests/joinset_enumeration.rs
- crates/semstrait-model/src/validation/structure.rs
links:
- target: 54
  kind: supersededby
---

# Enumerate every bounded composition eagerly at graph build; plan time is pure lookup

> **Superseded by [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md).** Replaced rather than moved: compile emits every legal sub-composition of one authored Joinset's specification, never the model-wide relationship adjacency index, which stays forbidden. `MAX_IMPLICIT_ENUMERATION_COUNT = 2000` is re-homed from `SemanticGraph` build to the enumerating compile stage, per authored Joinset, and the no-truncation rule transfers strengthened (0054's *The bound is on units emitted, not on constituents*). The **depth** bound on implicit enumeration retires, with its subject inverted onto authored nesting: `MAX_COMPOSITION_DEPTH = 4` bounds authored nesting, and it is model validate's. Both build outcomes retire: `CompositionDepthExceeded` with the bound it reported, and `NoCompositionPath` because under eager enumeration a missing path is a plan-time absence of a candidate rather than a compile fault. 0014's posture is otherwise upheld — enumeration stays eager and exhaustive, and plan time stays a lookup. Nothing moves in code: both constants and both outcomes are grep-absent from `crates/`.

## Context and Problem Statement

The planner entry path, cost model and error taxonomy (`NoCompositionPath`, `CompositionDepthExceeded` as build outcomes) all assume the index already exists, so moving search back to plan time would relocate work across the compile/plan boundary and reshape those errors. The posture is intentionally unlike a conventional join-path planner: it refuses to search per request, may build up to 2000 compositions most requests never touch, and reports "depth exceeded" instead of searching further.

## Considered Options

- Eager bounded enumeration (chosen).
- Lazy plan-time BFS/Steiner path finding per request.
- Silent truncation when the enumeration budget is exceeded.

Both alternatives were rejected because they make plan output depend on search state and hide incomplete enumeration from authors.

## Decision Outcome

All implicit and explicit compositions are exhaustively enumerated (BFS plus Steiner trees) and indexed at `SemanticGraph` build, under `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` and `MAX_IMPLICIT_ENUMERATION_COUNT = 2000`. A cap breach is a hard error rather than truncation. Field-first resolution at plan time is therefore a set-keyed map lookup: no BFS, no Steiner walk and no on-demand synthesis.
