---
number: 27
title: 'Authoring surfaces are closed and explicit: one Semantics namespace, `agg:`-only aggregation, opt-in filters, closed constraints'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-model/src/yaml/retired.rs
- crates/semstrait-model/src/validation/semantics.rs
- crates/semstrait-model/src/model/types/entities/semantic_body.rs
links:
- target: 65
  kind: amendedby
- target: 39
  kind: amendedby
---

# Authoring surfaces are closed and explicit: one Semantics namespace, `agg:`-only aggregation, opt-in filters, closed constraints

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0039](0039-keyness-is-a-per-site-attribution.md): a field that exists only on the local body, with no pooled counterpart, is not an override: a local-only field on any Semantics carrier is admissible provided the pooled body cannot spell it, and the test for that is a type split rather than a review convention; `agg:` on a `MeasureRef` would be a genuine breach, because `agg` exists on the pooled `Measure` and a site value would contradict it, while `is_key:` on a `DimensionRef` is not, because the pooled `Dimension` has no `is_key`.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): superseded on its `filters` clauses only: the clause admitting *"only `expr`, plus `filters` for Measure/Metric, [to] vary at a reference site"* now reads `filter`, and the clause that *"`DataKindFilter` and `AggregationFilter` are disjoint types with no shared name table"* loses one of its two subjects (the `AggregationFilter` carrier is retired). Everything else in 0027 stands, `agg:`-only aggregation included — that clause is reinforced in 0065's *What a tree contains, and who builds it*. 0027 is **not** amended in place.

## Context and Problem Statement

The rules below are the authoring grammar and the soundness basis for global identity. Loosening unification, or admitting inline aggregates, inline predicates, authorable severities or auto-injected filters, would change the meaning and results of already-authored models.

Accepted costs: SQL-trained authors cannot write `expr: sum(amount)` though the AST has an `Aggregate` variant; there are no `ref:`/`override:` keywords, no per-type namespaces, no mandatory row-level filters and no boolean guard escape hatch; and constraints are checked before names are resolved.

## Considered Options

- Closed, explicit authoring surfaces (chosen).
- Per-element-type sub-namespaces with dot-qualified names or aliases.
- Legacy `ref:`/`override:` forms with free per-site overrides and cast-tolerant `data_type` unification.
- SQL-style inline aggregation with no `agg:` tag (with compile-side `Aggregate` wrapping).
- Always-on or `requires:`-style filter injection, and one unified `Filter` entity instead of disjoint types.
- An inline predicate DSL plus author-declared severity in `constraints:`, and constraint evaluation after target resolution.

## Decision Outcome

A Semantics name identifies exactly one logical Semantics model-wide across all element types (unqualified references only, no aliases or dotted names). It is declared through one field-driven YAML form whose fields must unify by strict equality: only `expr`, plus `filters` for Measure/Metric, may vary at a reference site.

- Aggregation is never written inside `expr:` (rejected at every site but DataKind filters); it is carried by the separate `agg:` tag, from which parse synthesizes the `Aggregate` root.
- Named Filters apply only when a Request lists them; nothing can declare an always-on filter.
- `DataKindFilter` and `AggregationFilter` are disjoint types with no shared name table.
- `constraints:` admits only closed structured kinds with hard-error-only severity, evaluated as the planner's step 0.
