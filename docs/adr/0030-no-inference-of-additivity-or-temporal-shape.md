---
number: 30
title: 'Semantic properties are never inferred: Additivity and TemporalShape are authored and mutually independent'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/compile/temporal.rs
- crates/semstrait-ir/src/temporal.rs
- crates/semstrait-model/src/model/types/entities
links:
- target: 56
  kind: amendedby
- target: 74
  kind: amendedby
- target: 76
  kind: amendedby
- target: 77
  kind: amendedby
- target: 83
  kind: amendedby
- target: 91
  kind: amendedby
- target: 96
  kind: amendedby
---

# Semantic properties are never inferred: Additivity and TemporalShape are authored and mutually independent

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0056](0056-composed-temporal-shape-is-derived-at-compile.md): the strict-equivalence clause is superseded for Joinsets only. It stands unchanged for Unionset and Grainset. ADR 0030's operative content otherwise survives whole: no inference on a leaf, from column names or by default; composed shape derived from children; disagreement a hard error; the `MixedShapeAdvisoryChildren` advisory retired. Two ADR 0030 clauses are corrected in 0056's *A Unionset agrees; a Grainset aligns on kind and differs on grain* rather than superseded, on the precedent ADR 0052 set for ADR 0038's deferred-cardinality paragraph. *"TemporalShape is declarable only on leaf `SimpleDataKind`s"* was already contradicted before 0056 — by ADR 0026's cascade, by ADR 0053's scoping of it, and by live `ComplexExtras.temporal` — and is corrected so that a Complex kind may declare a shape that agrees (only an override is forbidden). *"Grainsets additionally require ≥2 distinct grains"* is corrected to pairwise distinctness, which is what `check_grainset_levels` enforces.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): the lossy re-aggregation advisory is raised where the grain comparison re-aggregates a non-additive aggregate over rows that are already aggregates (rows of a `Timeseries` or `Snapshot` Dataset); a semi-additive aggregate whose axis the request drops or coarsens applies its strategy instead; a Metric's aggregate node reads its function's additivity, narrowed by the effective additivity of the Measure it came from, once compile carries it, and by the Metric's own declared additivity.
> - [ADR 0076](0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md): in its *Decision Outcome*, every shape × additivity pair is legal, except where compile refuses a source that exposes a semi-additive aggregate under 0076's *Where `by` and the axes come from*, and additivity is still never derived from the shape; as ADR 0074 changed it, a semi-additive aggregate whose `by` the request drops or buckets coarser than the source holds it applies its semi-additive strategy instead of the lossy re-aggregation advisory; the rest of 0030 stands (0076's *`axes` and `by`* and *Where `by` and the axes come from*).
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in its *Decision Outcome*, as ADR 0074 amended it, a Metric's aggregate node no longer reads "the Metric's own declared additivity": a Metric's declared additivity applies to the Metric's own value, after its formula, and a node reads its function's additivity, narrowed by the effective additivity of the Measure it came from (0077's *A Metric's declared additivity applies to its own value, after its formula*).
> - [ADR 0083](0083-a-metric-declares-no-additivity.md): in its *Decision Outcome*, "`Additivity` is authored inline and defaults to `Additive` for Measures and Metrics" holds for Measures only: a Metric declares no additivity, and each `Aggregate` node of a Metric reads the effective additivity of its Measure through its link, so ADR 0074's "once compile carries it" no longer holds (0083's *Additivity belongs to an aggregate*).
> - [ADR 0091](0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md): in its *Decision Outcome*, "the *restriction* of `FunctionSpec.additivity` by the model declaration" becomes: the restriction of the aggregate function's declared additivity by the model declaration: the declared additivity of its DISTINCT form for a DISTINCT call. The rest of 0030 stands.
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *Decision Outcome*, the strict equivalence across a Grainset's children, which ADR 0056 kept, does not bar a child with no time axis: it may stand beside children with one (0096's *No time axis*); "Grainsets additionally require ≥2 distinct grains", which ADR 0056 corrected to distinctness between every two children, reads: no two children have the same grain and the same coverage mask (0096's *Shared granularity*); the rest of 0030 stands.

## Context and Problem Statement

Additivity and TemporalShape feed rollup legality and plan shape. Introducing derivation or defaults would silently change the results of every model that omitted a field, ratified rejections cannot be un-accepted without breaking authors, and promoting the advisories to errors would break working queries.

Accepted costs: the docs admit the `Additive` default is "often wrong" for ratio Metrics, obviously double-counting snapshots are permitted, a temporal Dimension with grains is *not* classified, `[Day, Day]` is rejected as degenerate, a missing join strategy errors instead of falling back to `Inner`, and arithmetically questionable SQL is knowingly emitted with a warning.

## Considered Options

- Authored, mutually independent properties (chosen).
- Deriving Additivity from `TemporalShape` (Snapshot ⇒ `SemiAdditive{snapshotted_at}`) or from `agg`.
- Author-declared additivity overriding function-level math.
- Heuristic implicit shape inference plus majority/first-branch shape selection in composition, and allowing Complex kinds to declare or override shape.
- Keeping the mixed-shape advisory with planner-side alignment (`TD-UNIONSET-SHAPE-ALIGN`).
- Silent `Inner` substitution for a missing `AsOf`, and hard refusal of lossy re-aggregation.

## Decision Outcome

`Additivity` is authored inline and defaults to `Additive` for Measures and Metrics. It is never derived from `TemporalShape`: every shape × additivity pair is legal, and suspicious pairs are Warnings only. The effective value is the *restriction* of `FunctionSpec.additivity` by the model declaration, so function-level `NonAdditive` dominates.

`TemporalShape` is declarable only on leaf `SimpleDataKind`s, never inferred from column names nor synthesized by default (absence means permanently shape-unclassified). Composed shape is derived, with strict equivalence required across children as hard errors (the `MixedShapeAdvisoryChildren` advisory is retired), and Grainsets additionally require ≥2 distinct grains.

A composition needing an unavailable `AsOf` traversal hard-fails rather than degrading, while known-lossy re-aggregation warns (`PLAN_W_2101`) and proceeds.
