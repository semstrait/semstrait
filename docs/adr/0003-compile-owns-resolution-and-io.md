---
number: 3
title: 'Six-verb pipeline: compile owns all resolution and all I/O; the query path is sync and manifest-complete'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/compile/driver.rs
- crates/semstrait-planner/src/planner.rs
- crates/semstrait-catalog/src/lib.rs
- crates/semstrait-api/src/orchestrator.rs
links:
- target: 68
  kind: amendedby
- target: 73
  kind: amendedby
- target: 89
  kind: amendedby
---

# Six-verb pipeline: compile owns all resolution and all I/O; the query path is sync and manifest-complete

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0068](0068-adapters-are-layered-format-first.md): its open question, "injecting an `EngineAdapter` into `plan()`" versus "no engine influence before `adapt()`", is resolved for the second: the engine enters as rewrite rules at adapt and never at plan.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): `optimize` is not retained, and the pipeline is parse, validate, compile, plan, adapt.
> - [ADR 0089](0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md): in its *Decision Outcome*, "Name resolution, catalog fetch, glob expansion, relationship resolution and expression lowering all happen inside one async `compile`" holds for the names of semantics; parse resolves a function's name against the function catalog, which needs no I/O (ADR 0091, rule 1; ADR 0089, rule 4). The rest of 0003 stands.

## Context and Problem Statement

Which stage owns resolution and I/O? The answer looks odd on first contact — a query planner that cannot look up catalog metadata, a stage that provably does nothing, and no `ValidatedModel` typestate proving validation ran.

## Considered Options

- All resolution and I/O in one async `compile` (chosen).
- An engine-style `resolve` stage emitting a `ResolvedModel`. Splitting resolution back out would require inventing an intermediate `ResolvedModel` threaded through crate boundaries and error enums.
- An end-to-end async pipeline with lazy plan-time catalog probing (the legacy `SemanticPlanner.catalog` field) and/or `block_on` bridges. Async or catalog access at plan time would infect every strategy and adapter signature and every caller.
- A `ValidatedModel` typestate enforcing that validation ran.

## Decision Outcome

The pipeline has six verbs and two compile-time types (`SemanticModel`, `SemanticManifest`).

- Name resolution, catalog fetch, glob expansion, relationship resolution and expression lowering all happen inside one async `compile`.
- `plan`/`optimize`/`adapt` are strictly synchronous, I/O-free and take no `CatalogProvider`.
- `validate` is a pure predicate over a borrowed model.
- `optimize` is retained even when its pass registry is empty.

## More Information

### Open questions

One design document sanctions injecting an `EngineAdapter` into `plan()` to replace canonical choices via visitor hooks, while the adapter crate charter and the plan IR design forbid any engine-identity influence before `adapt()`. This is an unresolved contradiction: either the injection mode is live and needs its own ratified decision, or the first document needs a correction.
