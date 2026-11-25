---
number: 63
title: The planner takes a request context, not a session
status: accepted
date: 2026-09-09
source:
- docs/adr/0024-pure-request-and-stateless-orchestrator.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0062-a-request-states-intent-never-placement.md
links:
- target: 24
  kind: amends
- target: 72
  kind: amendedby
- target: 74
  kind: amendedby
---

# The planner takes a request context, not a session

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0072](0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md): `RequestContext` carries `week_start`, Monday or Sunday, default Monday, beside `timezone`, read only when bucketing to the `week` grain; the timezone is no longer its only field.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): "requested grain" in 0063 is read as the requested granularity.

## Context and Problem Statement

ADR 0024's `SessionContext` carries mandatory `now` / `timezone` /
`feature_toggles`, and only the timezone has a reader: `now` is evaluated by the
engine because ADR 0024 already forbids folding it, nothing in the workspace
reads a feature toggle, and no session identity or authentication is planned.
ADR 0043 classes an unread carrier as a defect.

## Considered Options

- A `RequestContext` (chosen).
- Keeping an empty `SessionContext` as a placeholder for future authentication.
- A session-variable string bag referenced from model-authored filters.
- Putting `timezone` on the `Request` — the Request is *what* is asked, the
  context is *under which settings*, and a cached plan keys on both.
- Naming the carrier `QueryContext` — `CONTEXT.md` retires "query" for this
  concept.

## Decision Outcome

The planner's signature is `plan(&manifest, &request, &context)`. The third
argument is a `RequestContext`, `#[non_exhaustive]`, carrying **`timezone`** (an
IANA name, default UTC) and nothing else today. The timezone is read at one
site: bucketing a timestamp-typed temporal dimension to a requested grain. The
manifest-contextualised request never appears on the signature (ADR 0024).

The `SessionContext` is **retired**, and the one read field moves to where its
reader is. ADR 0024 is amended in that one clause; its rulings stand that the
planner reads no host clock, that `current_date` / `current_timestamp` are never
folded, and that plan output is a pure function of its inputs — now
`(manifest, request, context)`.

Caching is the caller's, keyed on compilation identity plus request plus
context. Identity, authentication and row-level restriction by model are out of
scope; a tenant restriction is a request predicate or a named filter. When an
identity arrives it arrives with a reader, and lands additively on
`RequestContext`.

### Amends

- **0024:** the `SessionContext` carrier is retired — `timezone` moves to
  `RequestContext`, `now` is left to the engine, feature toggles are deleted;
  its rulings stand that the planner reads no host clock, that `current_date` /
  `current_timestamp` are never folded, and that plan output is a pure function
  of its inputs — now `(manifest, request, context)`.
