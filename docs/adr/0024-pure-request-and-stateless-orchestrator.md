---
number: 24
title: Planner input is manifest-free with an explicit session clock; the orchestrator is stateless
status: accepted
date: 2026-08-04
source:
- crates/semstrait-planner/src/request.rs
- crates/semstrait-api/src/orchestrator.rs
links:
- target: 41
  kind: amendedby
- target: 63
  kind: amendedby
- target: 73
  kind: amendedby
---

# Planner input is manifest-free with an explicit session clock; the orchestrator is stateless

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0041](0041-recompile-is-the-callers-decision.md): "caches nothing" must be read narrowly, because the `Repository` is a cache; it remains binding on the *orchestrator*, which memoizes nothing.
> - [ADR 0063](0063-the-planner-takes-a-request-context-not-a-session.md): the `SessionContext` carrier is retired — `timezone` moves to `RequestContext`, `now` is left to the engine, feature toggles are deleted; its rulings stand that the planner reads no host clock, that `current_date` / `current_timestamp` are never folded, and that plan output is a pure function of its inputs — now `(manifest, request, context)`.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): MVP has no `optimize` stage and no `OptimizerPass`.

## Context and Problem Statement

Plan output being a pure function of `(manifest, request, toggles)` underpins golden-plan tests and downstream plan caching. An internal cache would change the freshness/drift semantics callers built their re-compile logic around, and pushing resolution back to callers is MAJOR for the API layer. The result also looks wrong at a glance: the session clock sits right there in the pass context and is deliberately not folded, and a top-level "orchestrator" memoizes nothing.

## Considered Options

- Manifest-free `Request`, explicit `SessionContext`, stateless orchestrator (chosen).
- The legacy already-resolved `ResolvedQueryRequest` / `SessionVariables` string bag as public planner input, with implicit host-clock lookup and plan-time folding of session functions.
- An internal manifest/plan cache keyed on `(yaml, catalog snapshot)`.
- A stateful session object owned by the orchestrator.

## Decision Outcome

- `Request` carries pure caller intent plus a `SessionContext` with mandatory `now`/`timezone`/`feature_toggles`.
- The manifest-contextualized `ResolvedQueryRequest` is produced internally and never appears on the `plan` signature.
- The planner never reads the host clock, and `ConstantFolding` deliberately does not fold `NOW()`/`CURRENT_DATE()`, so plans stay cacheable.
- `SemStrait` caches nothing and holds no session state, leaving manifest reuse and drift re-validation to the caller's `Arc` pattern.
