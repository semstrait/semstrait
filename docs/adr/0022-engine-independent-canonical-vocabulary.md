---
number: 22
title: Canonical vocabulary is fixed independently of engine capability; adapters rewrite, warn, or refuse
status: accepted
date: 2026-08-04
source:
- crates/semstrait-common/src/dialect.rs
- crates/semstrait-adapter/src/rules.rs
- crates/semstrait-adapter/src/types.rs
links:
- target: 66
  kind: amendedby
- target: 68
  kind: amendedby
- target: 72
  kind: amendedby
- target: 86
  kind: amendedby
- target: 91
  kind: amendedby
---

# Canonical vocabulary is fixed independently of engine capability; adapters rewrite, warn, or refuse

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0066](0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md): the vocabulary mirrors the Substrait v0.103.1 relations and admits only the values semstrait writes; it is fixed independently of engine capability, and the `JoinType` roster `{Inner, Left, Right, Full}` with its Semi/Anti/Lateral exclusion stands. A cross join enters as `CrossNode`, Substrait's `CrossRel`, because the model authors it, not because an engine offers it — 0022's own criterion. Its rewrite-warn-refuse ladder is ADR 0068's to amend.
> - [ADR 0068](0068-adapters-are-layered-format-first.md): its decision "adapters rewrite, warn, or refuse" is replaced by the paragraph "The adapter adapts; it does not refuse" of 0068's *The seam*, and by 0068's *Artifacts*, and the sections that only apply it read as those give: its title's "adapters rewrite, warn, or refuse"; in *Context and Problem Statement*, "capability variants become public contract" and the bullets "adapter rewrite strategies" and "`adapt` re-derives feasibility"; in *Considered Options*, the chosen option's "with a rewrite-warn-refuse ladder" and the options "Best-effort degraded emission or emulation" and "A descriptive per-dialect capability matrix"; and in *Decision Outcome*, the sentence "An engine shortfall is answered in strict order" and the `Capability` bullet: the `Capability` *type* is retired with no reader (the planner never reads engine capability). What stands, and is 0068's mechanism: the canonical vocabulary is fixed independently of engine capability, an engine never widens it, and there is no silent emulation — a rewrite is a visible rule with a cited reason. The rest of 0022 stands.
> - [ADR 0072](0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md): a timezone-free `Timestamp` that a source supplies holds a UTC instant, and a bucketed one is a clock value in the request timezone. The type roster is unchanged.
> - [ADR 0086](0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md): in its *Decision Outcome*, in "The roster is deliberately narrower than the primary engine", the data type roster gains `array`, whose element is a canonical data type (ADR 0086, rule 4), and `varchar(n)`, `char(n)`, `binary(n)`, `uuid`, `interval_year_month` and `interval_day_second(p)` (ADR 0086, rule 3). The roster is no longer narrower than the primary engine for every type: DataFusion 55's consumer refuses a `uuid` (ADR 0086, rule 6). `Timestamp` and `Time` stay tz-naive, and `Decimal` precision stays capped at 38. The rest of 0022 stands.
> - [ADR 0091](0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md): in its *Decision Outcome*, "Per-engine mappings live in a separate "Living" `registry/` tree that canonical specs may reference but never depend on" becomes: an engine's own declarations live in `semstrait-sql`, as dialect declarations under each canonical declaration. No canonical declaration depends on them. The rest of 0022 stands.

## Context and Problem Statement

The decision is effectively permanent: every adapter must carry rewrite machinery forever, canonical variants cannot be withdrawn once authored against, and capability variants become public contract for foreign Substrait consumers. It is also surprising:

- engines are forced into `QUALIFY`/`ROW_NUMBER` and `date_part` rewrites rather than narrowing the canonical set;
- semi/anti joins, tz-aware timestamps and `Decimal256` are refused although every target engine supports them;
- adapter rewrite strategies that look exactly like capabilities are deliberately unrepresentable;
- `adapt` re-derives feasibility instead of trusting its own advertisement.

## Considered Options

- Engine-independent vocabulary, with a rewrite-warn-refuse ladder (chosen).
- A lowest-common-denominator (or per-engine) canonical variant set, with mapping tables inline in `foundations/` rather than a separate Living registry.
- Best-effort degraded emission or emulation with warnings in place of hard refusal.
- Canonicalizing Calcite's full `JoinRelType` and Substrait `SINGLE`/`MARK` join kinds.
- Engine-default tz-aware timestamps and precision-76 decimals behind adapter gates.
- A descriptive per-dialect capability matrix, and capability-gated `adapt`.

## Decision Outcome

Adding an engine must not change the canonical variant set.

- Per-engine mappings live in a separate "Living" `registry/` tree that canonical specs may reference but never depend on.
- An engine shortfall is answered in strict order: a structural rewrite preserving semantics, then a warning plus degradation, then a hard `Unsupported*` error at `adapt`. It is never answered by silent emulation, stub emission, or an adapter-inserted cast.
- The roster is deliberately narrower than the primary engine: `JoinType` = `{Inner,Left,Right,Full}` with Semi/Anti/Lateral excluded, tz-naive `Timestamp`/`Time`, `Decimal` precision capped at 38, and `Grain` a total selection-rank order rather than a divisibility lattice.
- `Capability` variants exist only for features irreducible across the Substrait handoff. They are consumed as planner/api pre-flight hints, never as a gate inside `adapt`.
