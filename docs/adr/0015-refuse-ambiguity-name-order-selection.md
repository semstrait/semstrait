---
number: 15
title: 'Refuse rather than choose: ambiguity is an error and selection uses name order, never cost'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-planner/src/resolver.rs
- crates/semstrait-planner/src/routing.rs
- crates/semstrait-planner/src/error.rs
links:
- target: 54
  kind: amendedby
- target: 64
  kind: amendedby
---

# Refuse rather than choose: ambiguity is an error and selection uses name order, never cost

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): an authored Joinset duplicating an enumerable implicit one is not rejected; the two carry different names and ids, so nothing collides.
> - [ADR 0064](0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md): superseded in two clauses, both scoped to plan-time candidate selection, in 0064 rather than by editing its frontmatter (the ADR 0037 precedent). Its selection clause ruled that where a choice must be made it is made by *"coverage plus name order, with no statistics and no cost model, so a rollup-requiring finer unit can beat a direct-grain unit alphabetically"*, on the ground that a cost function added later would silently change results for existing models. No planner and no golden plan exist yet, so the ground does not bind the selection clause, and the example it accepts is a defect that 0064's order removes (0064's *Decision Outcome*). Name order survives as the **last** tie-break, no longer the first, and no longer silent. Structural complexity is request-free and statistics-free, and granularity fit reads the request but no statistics, so 0015's refusal of a *statistical* cost model stands untouched, and the ground still holds against a statistical cost; what changes is that structure and granularity count before names do. Its refusal clause made *"equal-cost composition ties"* a hard plan error; under 0064 a tie on every term among candidates is answered and advised, never refused, so that clause too is superseded **for plan-time ties**. 0015's compile-time refusals — an explicit composition whose form equals an enumerable implicit one, and multiple shortest relationship paths — are not touched by 0064; ADR 0054 already re-ruled the first (a duplicate is a redundancy, not a fault) and removed the second's subject (there is no plan-time path search).

## Context and Problem Statement

Callers and golden plans depend on deterministic errors and deterministic emitted plans, so introducing any tie-break heuristic or cost function later would silently change results for existing models. The posture is deliberately at odds with every comparable BI/semantic layer, which picks a default join path and prefers the cheaper source: here the system refuses to answer, and where it must answer it knowingly picks the more expensive unit.

## Considered Options

- Refusal, with name-order selection (chosen).
- Heuristic tie-break by declaration or lexical order, or shortest-then-first auto-pick.
- Explicit-shadows-implicit precedence, or coexistence of the two under distinct names.
- A stats-backed cost function for driver and attached-unit selection (`TD-GRAINSET-COST-STATS`).

## Decision Outcome

Equal-cost composition ties, multiple shortest relationship/Joinset paths, and an explicit composition whose canonical form equals an enumerable implicit one are all hard compile/plan errors (`PLAN_E_0500`, `AmbiguousRelationshipPath`, `COMP_E_0414`, `COMP_E_2402`).

Where a choice must genuinely be made — Grainset driver and attached units — it is made by coverage plus **name order**, with no statistics and no cost model, so a rollup-requiring finer unit can beat a direct-grain unit alphabetically.

## More Information

### Open questions

Grainset cross-grain composition composes on declared Keys and merely *warns* about fanout, whereas composition ambiguity elsewhere is a hard error. It is not settled whether these are one posture or two; the asymmetry may be inconsistent.
