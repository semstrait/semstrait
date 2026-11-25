---
number: 62
title: A Request states intent, never placement
status: accepted
date: 2026-09-09
source:
- CONTEXT.md
- docs/adr/0015-refuse-ambiguity-name-order-selection.md
- docs/adr/0024-pure-request-and-stateless-orchestrator.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0056-composed-temporal-shape-is-derived-at-compile.md
links:
- target: 71
  kind: amendedby
- target: 74
  kind: amendedby
---

# A Request states intent, never placement

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0071](0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md): in *Decision Outcome*, the placement bullet: "reads the expression layer of the predicate's leaves" becomes "reads the role on each leaf's definition", never the expression layer; for a before-aggregation predicate, "on the still-physical schema" becomes "on the named columns above the projection"; "a predicate mixing the two layers" becomes "a Dimension leaf beside a Measure or Metric leaf", the typed error; "the leaf replaced by the aggregate expression, never by its output alias" is read as 0071's *A predicate is applied where its semantics carry their meaning* reads it: in the plan an after-aggregation leaf reads the output column of the step that finished it, and "never by its output alias" is a rule for SQL text that a SQL format adapter keeps; "any measure or metric → applied after aggregation" holds but for a row-level Metric with no `agg:`, which is placed as a Dimension, and in the paragraph on a selection, "measures-only yields one row" holds but for such a Metric, which a request groups by (ADR 0074, *A Metric*). The rest of 0062 stands.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): "grain" for the temporal roster means the granularity, and "requested grain" is read as the requested granularity.

## Context and Problem Statement

Every SQL-literate caller of this public contract expects a `where` / `having`
split, a top-level `grain`, and one filter list mixing names and predicates —
the shape the old `RawQueryRequest` had. That shape made "total cost per
campaign, restricted to August" inexpressible: one list drove both projection
and grouping, so filtering on a date meant grouping by it.

## Considered Options

- Intent only (chosen).
- Caller-stated placement (`where` vs `having`), or a flat `{field, operator,
  value}` triple language conjoined implicitly.
- A top-level `grain` applied to an implicit time axis.
- Separate `dimensions` / `measures` / `metrics` lists in which the caller
  restates a role the manifest already holds.
- Session variables as filter values; a `*` wildcard selection; raw
  (unaggregated) row access.

## Decision Outcome

A `Request` names what the caller wants — `from` (optional, a **public** semantic
source), an ordered `select` of **selections**, one caller-authored **request
predicate** (`where`), a list of named-filter activations (`filters`),
`order_by` over selected names, `limit`, `offset` — and says nothing about how the
plan is arranged. Three consequences are the decision:

- **Placement is derived, never stated.** There is one predicate channel, not a
  `where` / `having` pair. The planner reads the expression layer of the
  predicate's leaves: all dimensions → applied before aggregation, on the
  still-physical schema; any measure or metric → applied after aggregation, with
  the leaf replaced by the aggregate expression, never by its output alias; a
  predicate mixing the two layers is a typed error. A named filter is
  model-authored and reached through `filters`; a Filter-role name as a `where`
  leaf is rejected, so each channel has one author.
- **The demanded set is `select ∪ where-leaves ∪ filters`**, and the chosen
  source must serve all of it. A dimension filtered but not projected is
  therefore legal by construction — the constraint named — and a measure
  filtered but not projected is computed in the aggregate and dropped from the
  projection. Nothing the caller demands is ever added to the grouping.
- **Grain is a modifier on a selected temporal dimension**, not a top-level
  field. A request with no temporal dimension selected has no grain. Only the
  requested grain of the source's time axis feeds Grainset routing and rollup
  legality; a bucketed non-axis temporal dimension is a projection. A grain on a
  non-temporal dimension, or finer than the source's, is a typed error.

A selection is untyped by default — `Field(name)`, role read from the manifest —
and typed (`Dimension { name, modifier }`, `Measure`, `Metric`) only when a
modifier is needed or the caller wants the role checked; a role mismatch is a
typed error. `select` is non-empty with no duplicates and fixes output column
order and names (the semantics name; no aliases, no expressions). Dimensions-only
yields distinct rows at that grain; measures-only yields one row. `from`
narrows the candidate set to the named source and the private units it
originated, and never widens past it.
