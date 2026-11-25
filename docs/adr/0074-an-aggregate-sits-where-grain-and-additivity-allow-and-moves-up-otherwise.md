---
number: 74
title: An aggregate sits where the grain and additivity allow, and moves up where they do not
status: accepted
date: 2026-09-30
source:
- CONTEXT.md
- docs/adr/0027-closed-explicit-authoring-surfaces.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0033-composite-plan-shape.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0056-composed-temporal-shape-is-derived-at-compile.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0063-the-planner-takes-a-request-context-not-a-session.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- the compile-seam tests `a_composed_measure_aggregates_the_projected_column`, `a_joinset_projects_the_preserved_sides_column` and `a_joinset_projects_a_bodied_metric_and_a_named_filter_from_the_preserved_side` in `crates/semstrait-manifest/tests/composed_signatures.rs`
- '`MetricRef` in `crates/semstrait-model/src/model/types/entities/metric.rs`'
- '`BinaryOpKind` in `crates/semstrait-sql/src/functions/operation/kind.rs`'
- the old planner's Unionset and layer-planning code (reference only)
- the human's rulings of 2026-10-17
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18, and the research on a Volatile call in Calcite, Spark, DuckDB, PostgreSQL, Substrait, MetricFlow, Cube, Looker and Malloy
- the decisions of 2026-10-18, and the research on link reading in Calcite and Substrait
- the decisions of 2026-10-18 and 2026-10-19, and the research on composition constituents
- sharpened in place (a Unionset branch NULL-fills the names that its mask does not hold)
- sharpened in place (a Unionset's composed branch gives a formula node in its Measure's field where it gives partial aggregates under the node's condition)
- sharpened in place (a join condition's leaf equates a composite operand's coalesced value where the operand's own condition equates the two values and its join never NULL-fills the leaf's constituent)
links:
- target: 30
  kind: amends
- target: 33
  kind: amends
- target: 48
  kind: amends
- target: 54
  kind: amends
- target: 57
  kind: amends
- target: 59
  kind: amends
- target: 62
  kind: amends
- target: 63
  kind: amends
- target: 64
  kind: amends
- target: 65
  kind: amends
- target: 66
  kind: amends
- target: 71
  kind: amends
- target: 73
  kind: amends
- target: 34
  kind: supersedes
- target: 72
  kind: amends
- target: 56
  kind: amends
- target: 76
  kind: amendedby
- target: 77
  kind: amendedby
- target: 79
  kind: amendedby
- target: 80
  kind: amendedby
- target: 83
  kind: amendedby
- target: 84
  kind: amendedby
- target: 26
  kind: amends
- target: 94
  kind: amendedby
- target: 96
  kind: amendedby
- target: 97
  kind: amendedby
---

# An aggregate sits where the grain and additivity allow, and moves up where they do not

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0076](0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md): in *Additivity decides whether a partial aggregate may be combined again*, a semi-additive aggregate is additive except along its `by`, a partial aggregate of it is grouped by its `axes` and its `by`, and "Apart from a semi-additive strategy" no longer holds (0076's *What retires*); in *A Joinset*, the fanned operand's aggregate reads one row per row identity through a `row_number` marker, which partitions by each column of the identity as its own column, in place of the distinct-rows aggregate; in *Re-aggregating rows that are already aggregates* and *A semi-additive aggregate's strategy*, the semi-additive strategy applies when the request drops `by` or buckets it coarser than the source holds it, it keeps the rows where its `rank` marker is 1 in the one aggregate step, with no inner aggregate, no join and no `IsNotDistinctFrom`, `by` and each axis are Dimensions of the Datasets that the aggregate reads, where it reads one, and extend the demand, as 0076's *Where `by` and the axes come from* says, and `average` is not a semi-additive strategy; in *Aggregates that need different row sets* and *Advisories and refusals*, no request is refused with `MixedRowSets` or `StrategyOrderUndecided`; the rest of 0074 stands (0076's *A marker gives each aggregate its rows*, *`axes` and `by`*, *Where `by` and the axes come from* and *What retires*).
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *Additivity decides whether a partial aggregate may be combined again*, a Metric's own declared additivity no longer narrows each `Aggregate` node of its tree, and "A semi-additive Metric gives each of its nodes its axes and strategy" is replaced by 0077's *A Metric's declared additivity applies to its own value, after its formula*; in *A semi-additive aggregate's strategy*, a semi-additive Metric's semi-additive strategy runs on the Metric's value, and the Measure's runs inside it, and the planner no longer reads `first` and `last` as `earliest` and `latest`, because they leave `SemiAdditivityStrategy`; in *A Joinset*, the `Coalesce` table and the two copies of a filter read as 0077's *A filter on a composition reads semantics, never a `Coalesce`* gives (0077's *A Metric's declared additivity applies to its own value, after its formula*, *A filter on a composition reads semantics, never a `Coalesce`* and *What retires*).
> - [ADR 0079](0079-a-unique-union-also-compares-its-keyset-and-a-re-aggregated-count-is-never-null.md): in *A Unionset*, mode `unique`'s `UNION DISTINCT` compares every output field of the union, which 0079's *A unique union compares its output, its keyset included* lists, so a keyset counts; a keyset Dimension that the parent does not read goes no further than the distinct; and **No advisory** does not hold for a mode-`unique` Unionset that has no guaranteed keyset; in *Considered Options*, the keyset half of **Deduplicating mode `unique` over more than the request's rows** is the form that 0079 chooses; in *A Joinset*, **Leaving out an operand**, only a join that is built extends a Unionset's demand, so over a mode-`unique` Unionset with no keyset the plan that prunes can keep other rows than the plan with the join; in *Additivity decides whether a partial aggregate may be combined again*, as ADR 0076 changed it, `Count` combines by `Sum`, written as `Coalesce(Sum(<partial>), 0)`; in *Advisories and refusals*, the planner returns `Advisory::UniqueWithoutKeyset` beside the plan where the plan reads a mode-`unique` Unionset that has no guaranteed keyset; the rest of 0074 stands (0079's *A unique union compares its output, its keyset included*, *Only a join that is built extends the demand* and *A re-aggregated Count is never NULL*).
> - [ADR 0080](0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md): in *Decision Outcome*, *A Unionset* and its `### Amends` entry for 0033, the Unionset's rules say how its branches give the delivery that its demand fixes, as 0080's *The parent's demand fixes an operand's delivery* gives it for rows, partial aggregates and final aggregates, and disjoint branches give final aggregates with no aggregate above the union only where the demand takes final aggregates; in *A Joinset*, **Pre-aggregation**, an operand groups by each column at its value before bucketing; under **A condition is never coalesced.**, "On a row that one operand supplies alone, the other copy reads NULL, so the row fails the filter" holds for a filter that the Joinset applies above the join, and not for a copy that goes into its operand; in **Leaving out an operand**, neither of the two copies of a filter over a Dimension that both operands map keeps its operand in the plan, and a pruned operand's copy is not applied; the rest of 0074 stands (0080's *Only a frame that gives final aggregates buckets*, *The parent's demand fixes an operand's delivery*, *A predicate copy goes into its operand on every join type* and *A copy applies only on an operand that the plan reads*).
> - [ADR 0083](0083-a-metric-declares-no-additivity.md): in *Additivity decides whether a partial aggregate may be combined again*, "ADR 0030 gives a Metric an authored additivity, as it gives a Measure" and "Until then, an author narrows a Metric by declaring the Metric's own additivity" no longer hold, in the clause as ADR 0077 changed it: a Metric declares no additivity, and each `Aggregate` node of a Metric reads its Measure's effective additivity through its link (0083's rule 3), so "until compile writes that Measure's additivity onto the inlined node" no longer holds either; in *A semi-additive aggregate's strategy*, as ADRs 0076 and 0077 changed it, "a source that exposes a semi-additive Measure or Metric" and "a source that serves the Measure or Metric serves its axes" apply to a Measure only, and a Metric has no semi-additive strategy; the rest of 0074 stands (0083's *Additivity belongs to an aggregate*).
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *A Grainset*, "A Grainset stands in one of its grains; ADR 0073 keeps cross-grain units out of MVP" holds only where one child answers alone (0084's rule 1), and otherwise the Grainset drills across two or more children, a Grainset that is a constituent of another composition only where its parent takes final aggregates from it; the pick reads only children that pass the grain test for the requested granularity, as repaired in place for Week; "If no grain serves the request, the Grainset is not a candidate" holds where no child serves it and no drill-across does, and no Metric that the Grainset computes (0084's rule 8) serves it; in *Consequences*, the Neutral on the planner's subtree of two aggregate steps, "ADR 0084's rule 8 gives no Grainset signature to a Metric that reads one" reads: the Grainset's signature for a Metric that it computes keeps a link to another Metric that it computes (ADR 0088), and a Metric with its own `agg:` plans in a subtree of its own, which joins the demand's subtree `Full` on `IsNotDistinctFrom` with coalesced keys (0084's rule 8); in *A Unionset*, **NULL fill**, "`coverage_of` (ADR 0059) reads only a source's own signatures, so it says `NullFill` for each name of a Grainset branch, which owns none (ADR 0065)" reads: it says `NullFill` for each name of a Grainset branch that the Grainset does not compute, and reads the Grainset's own signature for a Metric that it computes (0084's rule 8); in *A Metric*, **Its formula**, "Each Measure it references becomes the composition's own Measure tree" and "An operand's own site for the Metric, with its body and its filter, is not read" read, for a Metric that a Grainset computes: its tree reads each name that a child serves as that name's final value, `column(<Grainset alias>, <name>)`, and the child's own signature for that name is read (0084's rule 8); in *A Metric*, "The two subtrees join with an `Inner` join" and "so they hold the same groups, and the join loses none" read, for a Metric with its own `agg:` that a Grainset computes: the two subtrees join `Full` on `IsNotDistinctFrom` with coalesced keys, because the children can hold different groups (0084's rule 8); in *A Metric*, "A request predicate on such a Metric alone sits directly above its step 2 and its cast. The inner join then drops the whole output row" reads, for the subtree of a Metric that a Grainset computes: a request predicate on that Metric sits above the `Full` joins, so it drops the whole output row (0084's rule 7); the rest of 0074 stands (0084's *Which children answer*, *A Metric that no child maps whole* and *Week rolls up to nothing coarser*).
> - [ADR 0094](0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md): *Every request is planned against its grain first*, "The code keeps its `Grain` roster for the granularity until the model, the manifest and the IR get the sharper words.": the roster is `semstrait-common`'s `Granularity` (rule 3). The next sentence, "Where existing ADRs say "grain" for the temporal roster, they mean the granularity.", stands. The same section's "Its author may declare it on that composition, as `extras.temporal.grain:`." (near 0074:359-360) and *Consequences*' Neutral listing of `crates/` changes, "a composition that is a constituent may declare `extras.temporal.grain:`" (near 0074:1658): rule 10 spells the key `granularity`.
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *A Grainset*, "Among its grains that carry a granularity, it picks the one that serves the whole demand and has the coarsest granularity equal to or finer than the requested one", and the two cases for no requested granularity that follow it, read: a Grainset sends each demanded name to the child that the child rank ranks first for it alone, and answers from one child where every name goes to it, or, where the chosen children share no guaranteed keyset, or its parent takes rows or partial aggregates from it, from the child that the child rank ranks first among those that serve the whole demand; a child with no time axis is among the children where the demand reads no time axis (0096's *The pick* and *No time axis*); in *Every request is planned against its grain first*, "A Grainset grain with no granularity is one that nothing can route to (ADR 0056), so Grainset routing reads only grains that carry one" and "A composed child always does too, because compile refuses one without it" do not hold for a child with no time axis, "A Dataset child always does, because the model refuses one without it (`GrainsetChildMissingGranularity`)" holds for a Dataset child that maps the time axis, and "a composition that is a constituent, carries no granularity, and publishes a temporal shape or is a child of a Grainset" is refused with `CompileError::ComposedConstituentMissingGranularity` where it publishes a temporal shape, and with `CompileError::ComposedConstituentMissingTemporalShape` where it is a child of a Grainset, carries a time axis and publishes no temporal shape; in the same section, under **A composition that is a constituent**, "A Grainset that is a constituent derives none, because its children differ by granularity, so its author declares one" reads: a Grainset that is a constituent derives the granularity that its children agree on, and its author declares one where they differ (0096's *Shared granularity*); wherever it says that a Grainset that is a constituent carries, picks at or fixes "the granularity that it declares" or "the granularity that the Grainset declares", in *Every request is planned against its grain first*, *A Grainset* and *Consequences*, it reads the granularity that it carries, declared or derived, and "declares a granularity that every child serves", in *Consequences*, reads: carries a granularity that every child serves; in *Every request is planned against its grain first*, "it never holds its time axis at it: no two of its children share a granularity, so one of them at least is not at it" no longer holds where its children share that granularity (0096's *Shared granularity*); in its entry for 0065, ""it holds no grain of its own" holds for a Grainset that is no constituent of another composition, and one that is a constituent holds the granularity that it declares" reads as 0096's entry for 0065 gives; in its entry for 0056, "the two refusals of a constituent's granularity" reads: those two and `ComposedConstituentMissingTemporalShape`; in *Every request is planned against its grain first*, "A composed source comes to the planner with no granularity only where it is a root composition, or a constituent that publishes no temporal shape, under a Unionset or a Joinset" also holds under a Grainset, for a child with no time axis (0096's *No time axis*); the same clauses in its `### Amends` entries for 0056 and 0064 and in *Consequences* read the same way; the rest of 0074 stands.
> - [ADR 0097](0097-a-unionset-prunes-a-branch-whose-children-cannot-serve-the-demand-with-an-advisory.md): in *A Unionset*, the Grainset-branch clause "where the union takes rows or partial aggregates from the branch, as it does from every branch that is not disjoint, and no single child maps every name that the branch serves, the plan fails with `GrainNotServable` at the branch's frame path", and in *A Grainset* "A Joinset that builds its join takes rows or partial aggregates from its operands, and so does a Unionset from a branch that is not disjoint. … where no single child serves the names in its input, the plan fails with `GrainNotServable` at the Grainset's frame path", do not apply to a Grainset branch whose children cannot serve the demand by what they map. That branch is pruned (0097's *The rule*). A branch that passes the test stays refused (0097's *Partial aggregates*). The rest of 0074 stands.

## Context and Problem Statement

ADR 0073 gives a fixed list of rules one job: to say what each operand gives its
parent. An operand gives rows, partial aggregates or final aggregates. The rules
also say how partial aggregates combine again, where a Metric's formula is
applied, and how rows are bucketed. ADR 0073 fixed who owns these rules. This
ADR says what they are.

## Considered Options

An option marked *(the first resolution, replaced)* was the first resolution,
replaced.

- Each source aggregates where the grain and additivity allow, and the
  aggregation moves up to the parent where they do not (chosen).
- **Coalescing an operand's conditions like its value** *(the first resolution,
  replaced)*. The value and the condition can come from different sides: where
  `from` gives NULL and passes and `to` gives 10 and fails, 10 is counted. A
  `CASE` that takes the condition from the value's side mends that, but it is a
  choice the author did not write. The human ruled that the author writes any
  `Coalesce` in a condition.
- **Reading the coalesced value of a shared Dimension in the composition's
  filter or in a request predicate.** On the composition that value is the
  `Coalesce`, so the filter would read a `Coalesce` that nobody wrote. The human
  ruled that both hold on each side.
- **Reading the operand's finished Metric where it is final** (disjoint
  branches, a left-out operand). The operand site's body and filter may differ
  from the composition site's (ADR 0027), so the answer can be another Metric.
- **Branch partials decided by the family** (an earlier working design, which splits `Avg` into
  `Sum` and `Count`). It gives the same numbers, but the planner writes more
  families. Also, `CONTEXT.md` **Additivity** already governs pre-aggregation.
- **Deduplicating mode `unique` over more than the request's rows**: every name
  the union serves, or a row identity from a keyset. The human ruled that
  `unique` is `UNION DISTINCT` over the rows as the request reads them, the
  columns that the request's expressions read, and that a keyset contributes
  nothing to it.
- **Branches always pre-aggregate** (ADR 0033 as written), with
  `LossyReaggregation` for the rest. That gives a wrong count distinct.
- **A contributor carrier on the Metric signature** (the earlier working design). The formula needs
  no new carrier: compile can lower the site's body on the composition.
- **Recovering a composed Metric's formula from the operands' trees.** It reads
  another source's signature, against ADR 0071, and branch trees can differ by
  boundary `Cast`s and by a site's `expr` override.
- **Fan-out as an accepted consequence** that is only reported (`CONTEXT.md`
  **Fan-out** as written).
- **Looker's hash arithmetic** (a distinct sum of hash plus value). It is
  specific to each family, and it needs a hash function.
- **A row identity over the requested Dimensions only.** Two rows that agree on
  those Dimensions collapse into one, so the answer undercounts.
- **A row identity from a keyset that the request selects in full** *(the first
  resolution, replaced)*. The distinct-rows aggregate reads the keyset's columns
  from the operand, so the request need not select them, and the inferred
  identity is needed less often.
- **A `concat` of the identity's columns into one string**, with a comma
  separator and a NULL marker *(the first resolution, replaced)*. The rows
  `("a,b", "c")` and `("a", "b,c")` give one string, and a value can equal the
  marker, so a real row is dropped. Grouping by columns keeps the ruling's
  intent, an identity for the deduplication only, without the collision.
- **A source's first keyset, in manifest order, as its x** *(the first resolution,
  replaced)*. With the keysets `(id)` and `(email)`, a request by `email` gets
  one row per group, but this rule reads a re-aggregation and can raise a false
  `LossyReaggregation`.
- **Different raw `Literal`s as disjoint branches, whatever the bucketing.** Two
  dates in one month are one group by month, so leaving out the aggregate above
  splits that group.
- **The demanded set as the aggregates to read** *(the first resolution,
  replaced)*. A request that selects only a Metric reads none, and a count
  distinct in it would give partials.
- **Leaving mixed row sets, many axes and a NULL axis undefined** *(the first
  resolution, replaced)*. A legal request then has no deterministic plan.
- **Refusing at plan time, or ruling out as a candidate, a source that lacks an
  axis.** The human ruled that an axis is a Dimension, and that a missing one is
  a compile error.
- **A plain `Eq` in the `latest` / `earliest` join.** It drops every group whose
  grouping value or axis is NULL.
- **A new `NoGrainServes` error** *(the first resolution, replaced)*. ADR 0064's
  `GrainNotServable` already names the case.
- **All its Dimensions as the x of a source that has a keyset.** It raises a
  false advisory when the keyset already determines a Dimension that the request
  drops.
- **An unknown grain for a source with no keyset**, so that every request
  re-aggregates it. The human ruled that such a source has no keyset, not no
  grain: its x is its Dimensions.
- **Comparing a composed source with no granularity through its operands.** The
  human ruled that it is compared on x alone, as any source with no granularity
  is.
- **Refusing a composed Metric that must move up, until the manifest lowers its
  formula** *(the first resolution, replaced)*. The human ruled that a composed
  Metric is not refused.
- **Rewriting each comparison on a shared name in place** into `from.x = v AND
  to.x = v`. For a filter of one comparison it is the same as the two copies.
  Under `Or` and `Not` it selects other rows than the filter held on each
  operand.
- **The preserved side alone for a name that both operands map** (ADR 0065's
  table for `Inner`, `Left` and `Right`). The human ruled one form for every
  join type, `Coalesce` in the table's order. It differs from the preserved side
  alone only on a row where that side's value is NULL and the other's is not.
- **NULL fill on a Joinset.** A coalesce only where both operands map the name
  is simpler.
- **The lossy advisory on the time axis only.** It misses a dropped Dimension.
- **Refusing lossy re-aggregation.** ADR 0030 already rejected this.

## Decision Outcome

In short:

- Planning starts from the request's **grain**.
- By default, each source aggregates with its own signature, and its parent
  combines what it gives.
- When additivity, or a Unionset's mode `unique`, does not let the parent
  combine an operand's aggregates, the aggregation **moves up** to the parent,
  after the combine, and that operand gives rows. In a Unionset, every branch
  does.
- When the request allows it, the aggregate above is left out: disjoint
  branches give final aggregates with none above the union, and a Joinset that
  leaves out an operand passes the other operand's aggregate through.
- A Metric's formula runs over the aggregates of the demand that finishes it.
  Where the formula reads, outside an aggregate, a Dimension that this demand
  does not select with no granularity, or a column, it takes the **formula
  push-down**, one `sum` beside the other aggregates, where its fixed pattern
  holds, and its **formula parts**, in two aggregate steps, otherwise (*A
  Metric*).
- A Metric's `agg:` over a formula that reads a Measure or a Metric is a second
  aggregate step at its **formula grain**. A Metric of two aggregate steps reads
  a subtree of its own, which joins the frame's subtree on `IsNotDistinctFrom`.
- A row-level Metric is planned as a Measure where it declares `agg:`, and as a
  Dimension where it does not.

**The composed tree.** Three readings of a composed tree's aggregate
root were named. This ADR takes reading (b) for a Measure: over rows, the tree is the
computation; over partial aggregates, its aggregate call only confirms the
family. A cast Measure's root is the `Cast` above that call (ADR 0089, rule
17), not the call itself. The placement comes from the role, the additivity
and the kind of combine. A composed Metric's tree becomes its formula once
compile lowers it on the composition. This ratifies the earlier working design's R1. Its
forms (rows, partials, finals) stay; its `Avg` decomposition, its contributor
carrier and `ConflictingDelivery` do not.

The nouns *partial aggregate*, *final aggregate*, *row identity*,
*distinct-rows aggregate*, *moves up* and *disjoint branches*, and the advisory
and error names below, are working names.
A later decision fixes them. `crates/`
has no `Advisory` type yet.

### Every request is planned against its grain first

A **grain** is a pair (x, y):

- x is a set of Dimensions, taken from the Dimensions of an interface.
- y is the **granularity** of each Dimension that has one. Today only a temporal
  Dimension has one, from the roster Minute to Year.

The **request's grain** is its selected Dimensions, each with its requested
granularity (ADR 0062). Only the granularity on the source's time axis decides
whether the source serves the request: ADR 0062 gives routing and rollup
legality to the time axis alone, and there a bucketed temporal Dimension that is
not the time axis is a projection. Every bucketing counts when the planner asks
whether the source re-aggregates, because ADR 0072 groups by the bucketed value.
A request that does not select the time axis, or selects it without a
granularity, has no requested granularity on the time axis. It may still bucket
another temporal Dimension, and that bucketing counts for re-aggregation.

A **source's grain** is built from its schema and its granularity, the same way
for every kind of source:

- x is a guaranteed keyset of the source. A source with more than one guaranteed
  keyset has more than one x, and the comparison reads each. A source with no
  guaranteed keyset has one x, all its Dimensions: it has no keyset, but its
  Dimensions are still known.
- y is the granularity of its temporal shape. ADR 0056 derives the shape of a
  composed source, and it leaves the granularity absent where the source
  guarantees no single grain. A composition that is a constituent of another
  composition carries the granularity that its author declares on it, or else
  the one that ADR 0056 derives. That granularity is its y, and it applies to
  every constituent below it, down to a constituent that declares its own
  (**A composition that is a constituent**, below).

The planner compares the two grains:

- **The source serves the request** when the request's Dimensions are among the
  source's Dimensions, and the requested granularity on the time axis, if there
  is one, is equal to or coarser than the source's. A source at Week serves no
  requested granularity of Month, Quarter or Year, because a week can span two
  of them. This is the grain half of ADR 0064's candidate test. The mask test
  over the demanded set stays as ADR 0064 says.
- **The source re-aggregates** unless the request's x contains an x of the
  source and buckets none of that x's Dimensions coarser than the source holds
  them. The source holds its time axis at its granularity, and any other
  temporal Dimension at its own values. So a source keyed by
  `(order_day, ship_day)`, asked by `order_day` and by `ship_day` at Month,
  re-aggregates. The order of the keysets in the manifest does not change this.
  A composition holds its time axis at its granularity only where each value of
  that time axis in its rows comes from a Dataset that has the Dimension on its
  own time axis at that granularity. Otherwise the composition holds its time
  axis at its constituents' own values, and a request at its granularity
  buckets them and re-aggregates, as for a source with no granularity. A
  Dataset that does not map the Dimension gives no value. Each branch of a
  Unionset gives its values. A Grainset gives the values of every child for
  the grain test and for a semi-additive strategy, and of the children that
  its decision reads for the fan-out marker (**Fan-out**; ADR 0076, **Where
  the marker has no place**, rule 1), whose check runs once the plan has
  decided the frames below. So a Grainset that is a constituent carries the
  granularity that it declares. For the grain test and a semi-additive
  strategy, it never holds its time axis at it: no two of its children share
  a granularity, so one of them at least is not at it. For the fan-out
  marker, it holds its time axis at it where each child that its decision
  reads is at it. A Joinset takes a Dimension that both operands map through
  its `Coalesce` (*A Joinset*). Where its condition equates the values of
  that name that its two operands give, it takes the values of `from` under a
  left join, of `to` under a right join, of either operand under an inner
  join, and of both under a full join. A leaf of the condition names a
  constituent, never a composite operand (ADR 0057, *The qualifier names the
  constituent, not the immediate operand*, as ADR 0075 amends it), so a leaf
  naming a constituent whose value the operand coalesces with another
  constituent's equates the operand's values only where that `Coalesce` is the
  constituent's value in each row: where the operand's own condition equates
  the two values and its join never NULL-fills that constituent. For any other
  name, it takes the values of both. What a composition serves, and how it
  ranks by **Granularity fit**, read the granularity that it carries. Take,
  under a Grainset, a Joinset that declares Month: `spend` (Day, keyed by
  `(campaign_id, date)`, with A's clicks 5 on 2026-03-10 and 4 on 2026-03-20)
  joined many-to-one to `targets` (Month). For
  `select date at Month, campaign_id, clicks`, the Joinset re-aggregates, and
  March gives (A, 2026-03, 9). If the test read the Joinset's rows as held at
  Month, no aggregate would run, and March would give (A, 2026-03, 5) and (A,
  2026-03, 4).
- **A source with no granularity** has no y, so its grain is compared on x
  alone. This is the part of the Grainset's rule that reads Dimensions (A
  Grainset). The source serves every requested granularity. It holds its time
  axis at its own values, as it holds any other temporal Dimension, so a request
  that buckets the time axis re-aggregates. This holds for a Dataset and for a
  composed source alike. A composed source comes to the planner with no
  granularity only where it is a root composition, or a constituent that
  publishes no temporal shape, under a Unionset or a Joinset (**A composition
  that is a constituent**, below).

  A Grainset grain with no granularity is one that nothing can route to (ADR
  0056), so Grainset routing reads only grains that carry one. A Dataset child
  always does, because the model refuses one without it
  (`GrainsetChildMissingGranularity`). A composed child always does too, because
  compile refuses one without it (below).
- **The rows an aggregate reads are already aggregates** when a Dataset that its
  arguments read has the temporal shape `Timeseries` or `Snapshot`. The rows of
  an `Events` or `Scd` Dataset, or of a Dataset with no shape, are not. A
  composed source's derived shape (ADR 0056) does not decide it: in `orders
  (Events) ⋈ balances (Snapshot)`, an aggregate over `balance` reads rows that
  are already aggregates.

Why x is a keyset: take a daily table with the Dimensions `(campaign_id,
campaign_name, date)` and the keyset `(campaign_id, date)`. The request `avg_cpc
by campaign_id, date` does not select `campaign_name`. The keyset says that each
group is one row, so no re-aggregation happens. Why every keyset: a source with
the keysets `(id)` and `(email)` gives one row per group to a request by
`email`, whichever keyset comes first.

**A composition that is a constituent** of another composition carries a
granularity. Its author may declare it on that composition, as
`extras.temporal.grain:`. Otherwise it is the granularity that ADR 0056
derives: the one that every constituent below it, at every depth, that carries
a granularity agrees on. The derivation of the granularity stops at a
constituent that declares one, because that declaration stands for the
constituent's whole subtree; the derivation of the temporal shape in ADR 0056
does not stop there. A
derived granularity is absent where no constituent below carries one, and
where they disagree. Compile refuses:

- a composition that is a constituent, carries no granularity, and publishes a
  temporal shape or is a child of a Grainset, with
  `CompileError::ComposedConstituentMissingGranularity { composite, constituent }`.
  A composition with no temporal shape under a Unionset or a Joinset carries
  none and takes no part in the derivation, as a Dataset with no granularity
  does. A Grainset that is a constituent derives none, because its children
  differ by granularity, so its author declares one.
- a declared granularity that a constituent below the composition does not
  serve by the grain test above, with `CompileError::StatedGranularityNotServed {
  composite, stated, constituents }`: one finer than that constituent's
  granularity, or Month, Quarter or Year over Week. Every child of a Grainset
  is such a constituent, so a Grainset of a Day and a Month child declares
  Month or coarser, and one of a Week and a Month child has no granularity that
  both serve. The check reads each constituent below, down to a constituent
  that declares its own granularity, and there it reads that declaration and
  not the constituent's subtree, which that constituent's own check reads. A
  constituent with no granularity serves every granularity. So a declaration
  never makes the declared claim of a constituent false.

Compile stores the declared or derived granularity as the composition's own
`temporal.grain`, ADR 0056's slot, and changes no entry of a constituent. The
planner reads it as the composition's y. That granularity applies to every
constituent below the composition, down to a constituent that declares its
own: there a constituent serves only the requested granularities that a source
at that granularity serves, whatever its own granularity is. A constituent
that declares its own granularity reads it for its own subtree, and a
granularity declared above it bounds only what it serves to the composition
above. So a Day Dataset under a Unionset that declares Month serves Month,
Quarter and Year, and not Day, and a Grainset that declares Day under a
Joinset that declares Month serves the Joinset Month and coarser and picks its
child at Day. A Grainset that is a constituent picks its child at the
granularity that it declares, whatever granularity the request asks, and
drills across (ADR 0084) only where its parent takes final aggregates from it
(*A Grainset*). Every other
constituent reads its time axis at its own values, and the bucket stays where
ADR 0080 puts it. A Dataset and a root composition do not change otherwise: a
root composition's granularity is only derived, because model validation
refuses `temporal.grain:` on it (`TemporalGranularityOnComplex`), it holds its time
axis at that granularity only as **The source re-aggregates** says, and a
source with no granularity is compared on x alone, as above.

The code keeps its `Grain` roster for the granularity until the model, the
manifest and the IR get the sharper words. Where existing ADRs say "grain" for
the temporal roster, they mean the granularity.

### Additivity decides whether a partial aggregate may be combined again

Additivity is the effective one of ADR 0030: the family's value, narrowed by the
author. Its two inputs stay apart. The aggregate function's additivity, from its
declaration (ADR 0091, rule 3), says whether a partial aggregate of it may be
combined again. The Measure's declared additivity, on its definition, says how
its rows are read, and it narrows the function's: `non` combines no partial
aggregate again, and a semi-additive strategy keeps the combine except along its
`by` and keeps the rows that the function then combines (ADR 0076).

- **Additive** families combine again. `Sum`, `Min` and `Max` combine by
  themselves, and so do `any_value`, `bool_and` and `bool_or`. `Count` combines
  by `Sum`, and `regr_count`, which counts rows and is never NULL, combines as
  `Count` does (ADR 0079). Apart from a semi-additive strategy
  (below), those are the places where the planner writes a family other than
  the definition's. The formula push-down (*A Metric*) also writes a `count`
  as a `sum`.
- **Non-additive** families: `Avg`, count distinct (`Count` with `distinct`),
  `Median`, `StdDev` and `Variance`. A Measure that its author declares
  non-additive is non-additive too.
- **Semi-additive**: additive except along the aggregate's axes. A partial
  aggregate of a semi-additive aggregate is grouped by its axes as well, so that
  its strategy can apply after the combine.

The rules below read the additivity of every aggregate that the request needs:

- each demanded Measure, by its effective additivity;
- each `Aggregate` node in the tree of a demanded Metric, by its function's
  additivity, narrowed by the effective additivity of the Measure it came from
  (below) and by the Metric's own declared additivity. ADR 0030 gives a Metric
  an authored additivity, as it gives a Measure. A semi-additive Metric gives
  each of its nodes its axes and strategy.
- not a Metric's own `agg:` over a formula that reads a Measure or a Metric:
  it aggregates the formula's values in its step 2, and no additivity judges it
  (*A Metric*).

An inlined node carries no link to the Measure it came from (ADR 0071). So that
Measure's own narrowing is not read inside a Metric until compile writes that
Measure's additivity onto the inlined node. Until then, an author narrows a
Metric by declaring the Metric's own additivity.

The demanded set (`CONTEXT.md`) names definitions, and it does not expand a
Metric. So a request that selects only a Metric that reads a Measure still has
aggregates to read.
Once compile lowers a composed Metric's formula, a Metric over a count distinct
makes the branches of a union give rows.

`Avg` is not split into `Sum` and `Count`. The answer is exact either way (rows
and one aggregate above), and the planner writes no further family.

### A Unionset

- **Partial aggregates.** When the mode is `all`, every aggregate that the
  request needs is additive or semi-additive, and none is a formula push-down's
  `sum` (*A Metric*), each branch gives partial aggregates. The union
  re-aggregates them above.
- **Rows.** In every other case, unless the branches are disjoint (below), each
  branch gives rows, and one aggregate sits above the union. This includes any
  non-additive aggregate, and every request in mode `unique` (the human's
  ruling).
  A union's output has one form, so one non-additive aggregate changes the form
  for the whole request, and a Metric's subtree of two aggregate steps, which
  reads the union again (*A Metric*), reads it in that form (**One form for
  every read**). A branch that gives rows outputs, for each Measure
  that its parent demands, the Measure's row argument under the Measure's name,
  and its aggregation condition as a boolean column.
- **Mode `unique` is `UNION DISTINCT` over the request's rows.** It compares the
  rows as the request reads them: the columns that the request's expressions
  read (the union's output, ADR 0073), and no others. A keyset adds nothing to
  it. So two rows that agree on those columns count once, as the author's choice
  of mode says.
- **Disjoint branches: no aggregate of the frame's read above.** A selected
  Dimension may fold to a `Literal` on every branch, by the fold with which
  ADR 0073 prunes a branch,
  with every value non-NULL and different. A Joinset on the way folds a name
  that both operands map to a literal only where both operands give that same
  literal, and a Unionset or a Grainset only where every branch or child gives
  that same literal.
  Where the
  request buckets that Dimension, the values must still differ after the
  bucketing (ADR 0072): the dates `2026-09-01` and `2026-09-02` are one group by
  month. Then no group spans two branches. Each branch gives final aggregates of
  every family, and no aggregate of the frame's read sits above the union. This
  applies in mode `all` only, and where each Metric of two aggregate steps
  groups its step 1 by the demand's grouping alone: an `agg:` whose formula
  reads no name outside its aggregate calls that the demand does not group by,
  and that links no Metric of two aggregate steps. Each such subtree's step 2
  sits above the union (**One form for every read**). Where a step 1 groups
  finer, the branches give the partial aggregates or the rows of the rules
  above. A value that reaches an operand that its join null-fills is not a
  non-NULL literal. It is ADR 0033's "provable disjointness predicate". It
  replaces the old planner's check (`has_distinguishing_known_values`), which compares raw
  values, does not bucket, and reads a missing value as `''`. A Unionset that
  reads one branch reads it as disjoint only where that branch's frame drills
  across; such a branch needs no Dimension that folds to a literal.
- **One form for every read.** The frame's read and each subtree of two
  aggregate steps, a linked one included, read the union in one form, as one
  delivery of a Joinset's operand serves both reads (*A Metric*, **Nesting**).
  - Where it is partial aggregates, each branch also groups them by each
    Dimension and each column under `auto` that a subtree's formula grain or
    part key reads, at its value before bucketing, never by the key. It gives
    each aggregate of the subtree's step 1 beside the frame's own. The frame's
    read gives each aggregate that a reader of a re-read Metric reads beside
    the link; the link is the re-read's. The frame's aggregate step and each
    step 1 re-aggregate them.
  - Where it is final aggregates (**Disjoint branches**), each branch also
    gives each aggregate of step 1, under its own `rank` marker where it reads
    one (ADR 0076). Step 1 reads each one as a field, with no aggregate over
    it, and the subtree's formula and step 2 sit above the union.
  - Where one form cannot serve both reads, the subtree reads the union again
    in its own form at step 1's grouping, by the rules above. There are two
    such cases. In one, a Grainset branch serves the frame's read only by a
    drill-across, and a step 1 groups finer than the demand. In the other, a
    composed branch's `row_number` marker is refused by ADR 0076's rule 1 at
    that finer grouping. A subtree that takes the parts reads the union again
    at the demand and the part key, which each branch computes on its own rows
    under final aggregates. The subtrees join `Inner` over the same read,
    `Left` onto a frame that has aggregates of its own, and `Full` with
    coalesced keys between two reads again. After a `Full`, a later subtree
    over the left read joins `Left`.
- **NULL fill.** For a branch that does not map a name, the driver writes a
  typed `Literal(NULL)` for it in that branch's row projection. A branch that
  NULL-fills the Measure of a Metric's aggregate node gives that node the type
  of the node's call over the Measure's typed NULL (ADR 0089, rule 19), the
  type that the node's call gives over the rows of a branch that maps the
  Measure. The Unionset Strategy reads the branch's mask: a branch maps the
  names that its mask holds. `coverage_of` (ADR 0059) reads only a source's own signatures, so
  it says `NullFill` for each name of a Grainset branch, which owns none (ADR
  0065). NULL fill needs compile to let a branch serve only the names it maps.
  A Grainset branch keeps the names that it NULL-fills out of the demand that
  its pick and ADR 0084's rules 1, 3 and 6 read, and out of a drill-across's
  grouping and join. The child that answers
  alone writes them in its row projection; under a drill-across, the Grainset's
  frame writes them in its projection (ADR 0084, rule 7). A name that a
  Grainset branch's mask holds is never NULL-filled. Every child of the branch
  serves the granularity that the branch declares (*Every request is planned
  against its grain first*); where the union takes rows or partial aggregates
  from the branch, as it does from every branch that is not disjoint, and no
  single child maps every name that the branch serves, the plan fails with
  `GrainNotServable` at the branch's frame path (*A Grainset*).
- **No advisory.** Each of these answers is exact. The union's `semantic_meta`
  records the form the branches gave, as trace.

### A Joinset

**The projection.** A name that only one operand maps reads that operand's
column, as ADR 0065 already says. A name that both operands map reads
`Coalesce`, in this order (ADR 0065's table changes to match):

| Join type | A name both operands map |
|---|---|
| Inner | `Coalesce(<from>.<name>, <to>.<name>)` |
| Left | `Coalesce(<from>.<name>, <to>.<name>)` |
| Right | `Coalesce(<to>.<name>, <from>.<name>)` |
| Full | `Coalesce(<from>.<name>, <to>.<name>)` |

**A condition is never coalesced.** The planner writes `Coalesce` for a value
only. It never writes one into a join condition, or into an aggregation
condition (ADR 0071), because a value that one side lacks, taken from the other
side, can admit a row that its own side rejects. Where the author wants a
`Coalesce` in a condition, the author writes it in the model.

- A name that only one operand maps reads that operand's condition column as it
  is. On a row that the other operand supplies alone, the condition is NULL, and
  the aggregate skips the row.
- A name that both operands map reads no operand's condition. Its condition is
  the composition's own filter for it (ADR 0071): its site's `filter:`, else the
  pooled one (ADR 0065 D2). If an operand's signature carries a condition for
  that name and the composition has no own filter for it, the request is refused
  with `PlanError::AmbiguousOperandCondition { semantics, semantic_source }`, so
  that no operand's condition is dropped unseen.
- **A Dimension that both operands map is read on each operand in the
  composition's filter and in a request predicate.** The planner makes two
  copies of the whole filter. In the first, every such name reads `from`; in the
  second, every such name reads `to`. A name that one operand maps reads that
  operand in both copies. The two copies are joined with `And`, so the filter
  holds on each operand. So `status = 'paid'` becomes
  `from.status = 'paid' AND to.status = 'paid'`, and
  `status = 'paid' OR region = 'EU'` becomes
  `(from.status = 'paid' OR from.region = 'EU') AND (to.status = 'paid' OR
  to.region = 'EU')`. The filter never reads the coalesced value of that name.
  Each copy of a request predicate is placed as ADR 0071 places a predicate.
- On a row that one operand supplies alone, the other copy reads NULL, so the
  row fails the filter. An author or a request that wants such rows writes the
  NULL case into the filter, for example `coalesce(status, 'paid') = 'paid'`,
  and it is then applied on each side as written.

The operands' own conditions are still never joined with `And` (ADR 0071). The
`And` above joins the author's one filter, read on each side.

**Pre-aggregation.** An operand pre-aggregates when every aggregate it gives is
additive or semi-additive. It groups by these columns:

- the columns of the join condition;
- every column the parent reads from it: the grouping values, the columns of the
  predicates placed above the join, and the columns of the relationship filter.

First it applies the predicates that ADR 0071 places inside it. Then the Joinset
re-aggregates. An operand with a non-additive aggregate gives rows.

**Fan-out.** The declared cardinality may let one row of an operand match many
rows of the other.

- If the aggregates are additive, pre-aggregation removes the fan-out, provided
  the many side becomes one row per join key.
- Otherwise the fanned operand's aggregates are **deduplicated by its row
  identity**. This covers non-additive aggregates, and a many side that still
  has more than one row per key. First a distinct-rows aggregate (ADR 0066: an
  aggregate step with no aggregates) runs over the request's grouping, the
  identity, and the aggregate's arguments and condition. Then the family's
  aggregate runs above it.
- `Min`, `Max` and count distinct skip the deduplication, because duplicate rows
  do not change them.

With a guaranteed keyset, each row of the fanned operand then counts once per
group, the answer that Looker's symmetric aggregates give. With an inferred
identity (below), the answer can differ from the true one.

**A Measure lowered on a Joinset** (ADR 0077, *A constituent serves only the
names it maps*) reads the joined rows, and its value can read two operands. So
its fanned operand is found for each result of its argument: each result of the
argument's `case`, nested ones included, or the whole argument where it has no
`case`. Each value leaf of a result walks down its qualifier chain to a Dataset,
by the walk of *A Metric*, *On a Joinset*:

- a result with a value leaf that reaches a Dataset that no Joinset on its path
  repeats, or with no value leaf that reads a column, reads the joined rows as
  they are: each joined row is one row of that Dataset;
- a result whose every value leaf reaches a Dataset that its path repeats is an
  aggregate of those Datasets. It reads one row per row identity of each of
  them, in each group and for each result, through ADR 0076's `row_number`
  marker, whose partition also holds the position of the result that the row
  selects. Where one result would read two such markers, the planner refuses
  it (ADR 0076, **Where the marker has no place**, rule 3). `Min`, `Max` and
  count distinct skip it, as above.

On `orders ⋈ order_lines`, one-to-many on `order_id`, with the lines (l1, o1,
`gift`), (l2, o1, `gift`), (l3, o2, `std`), (l4, o3, `gift`) and (l5, o4,
`std`), `gift_amount = sum(CASE WHEN kind = 'gift' THEN amount END)` reads o1
once, and by region it gives eu 100 and us 200, where the joined rows give eu
200. `count(line_id)` beside it, in the same aggregate step, reads every joined
row. On `orders ⋈ currencies`, many-to-one, `sum(amount * rate)` reads `amount`
from `orders`, which the join does not repeat, so it reads the joined rows: each
order once.

**The row identity** is a list of columns. The distinct-rows aggregate groups by
each of them as its own column.

- The columns are the Dimensions of the fanned operand's first guaranteed
  keyset, in manifest order. The request need not select them: the distinct-rows
  aggregate reads them from the operand. A Grainset's `keysets` field holds no
  members (ADR 0065), and compile mints a keyset's id from its source and its
  name (`mint_keyset_id`), so for a fanned Grainset operand the planner
  resolves the keyset's name to the keyset of that name on the child that
  answers, and the columns are that keyset's Dimensions. One child
  answers it, because the Joinset that repeats its rows builds its join and
  takes rows or partial aggregates from it (*A Grainset*).
- With no guaranteed keyset, the columns are every Dimension that the fanned
  operand exposes. That is an inference: two rows that are equal in every
  exposed Dimension and in the aggregate's arguments merge into one. The planner
  returns `Advisory::FanOut`, which says so, beside the plan.
- Grouping compares each column by itself: two rows share an identity only when
  every column is equal. NULL groups with NULL and never with `''`, so a NULL
  and an empty string are two identities. No column is cast, joined into one
  string or hashed.

**Leaving out an operand.** The Joinset's Strategy may leave an operand out, and
the driver then builds no node for it, when all three of these hold:

1. every demanded name is read from the other operand;
2. the join drops no read row: the read operand is the preserved side, or the
   join is `Inner` and the relationship claims integrity (`enforced`, or
   `assumed`, which is the default). Integrity claims that the composition loses
   rows from neither side (`CONTEXT.md` **Integrity**). It is not referential
   integrity, which speaks about one side only;
3. the relationship has no filter.

Cardinality is not a condition, because reading the operand alone counts each
read row once per group, the answer that the deduplication above aims at. This
answers ADR 0054's handoff: the rule reads the join type, and the deduplication
takes the place of cardinality. A `Full` or cross join never leaves out an
operand. A bypass into a sub-composition (ADR 0064) must satisfy the same rule
for every constituent that it leaves out. The operand that is read gives what
the parent asks for, and the Joinset passes it through. There is no advisory:
the plan records the left-out operand as trace.

For example, in `orders INNER customers` with `integrity: none`, the Joinset's
answer does not include an order that has no customer. Reading `orders` alone
would add it, so condition 2 keeps the join.

### A Grainset

A Grainset stands in one of its grains; ADR 0073 keeps cross-grain units out of
MVP. Among its grains that carry a granularity, it picks the one that serves the
whole demand and has the coarsest granularity equal to or finer than the
requested one. With no requested granularity on the time axis:

- if the request does not select the time axis, it picks the grain with the
  coarsest granularity that serves the whole demand;
- if the request selects the time axis without a granularity, it picks such a
  grain with the finest granularity, whose values are the least bucketed.

If no grain serves the request, the Grainset is not a candidate (ADR 0064).
Field-first resolution then picks among the other candidates, and it fails only
as ADR 0064 says: `NoCandidate` when no source qualifies, and `GrainNotServable`
when `from` names this Grainset. The grain buckets to the requested granularity
at its own aggregate. Where the rows that an aggregate reads in the grain are
already aggregates (above), the rule on re-aggregating aggregates (below)
applies to it.

A Grainset that is a constituent of another composition runs this pick, and
the pick of each name in ADR 0084's drill-across, among the children that pass
the grain test at the granularity that the Grainset declares, whatever
granularity the request asks, and it fixes that granularity where the request
asks none (*Every request is planned against its grain first*). Every child
serves that granularity, so none is coarser than it. A request at a coarser
granularity is bucketed at the requested granularity, where ADR 0080 puts the
bucket: in a drill-across, each chosen child buckets at it in its own
aggregate (ADR 0084, rule 5).

It drills across only where the delivery that its parent's demand fixes for it
is final aggregates (ADR 0080, *The parent's demand fixes an operand's
delivery*), so that no aggregate step above it combines its rows again: for
example as a disjoint branch of a Unionset whose demand takes final
aggregates, or as the operand that a Joinset keeps alone, whose aggregate the
Joinset passes through (**Leaving out an operand**), where the Joinset's own
demand takes final aggregates. A Joinset that builds its join takes rows or partial aggregates
from its operands, and so does a Unionset from a branch that is not disjoint.
There a drill-across's final aggregates would be combined again or joined, so
where no single child serves the names in its input, the plan fails with
`GrainNotServable` at the Grainset's frame path. Take `G` {`g_day` (Day; maps
clicks), `g_month` (Month; maps budget)}, keyed by `(campaign_id, country,
date)` and declaring Month, joined many-to-one to `campaigns` on
`campaign_id`, where `g_day` has A's clicks 10 and 20 in SE and 30 in NO and
B's 40, `g_month` has A's budgets 100 in SE and 50 in NO and B's 70, and both
campaigns are in EU. `select region, avg_clicks, budget` is refused: an
average per group from the drill-across, averaged again above the join, gives
30.0 or 28.33 where `g_day`'s rows give 25.0. `select region, avg_clicks` reads
`g_day`'s rows and gives 25.0. `select campaign_id, avg_clicks, budget` reads
no name of `campaigns`, so the Joinset keeps `G` alone, and the drill-across
gives A (20.0, 150) and B (40.0, 70).

### A Metric

**Its formula.** A composed Metric's signature becomes the body that the
composition's own site gives it, lowered on the composition itself as on a
Dataset. That body is the site's `expr` where its `MetricRef` overrides it (ADR
0027), and the pooled body otherwise. Each Measure it references becomes the
composition's own Measure tree. The composition's own filter for the Metric is
joined with `And` onto each inlined aggregate, as ADR 0071 does for a Dataset
Metric. An operand's own site for the Metric, with its body and its filter, is
not read, because the composition reads the operand's Measures, not its Metric.

This works for two reasons:

- Compile refuses a Metric that references a name its source does not expose
  (`CompileError::UnexposedReference`), so each referenced Measure has a
  composed tree.
- The planner then places each aggregate node of the formula like any Measure:
  in the operands, as partial or final aggregates, or above, over rows. It
  applies the formula above the last of them, or inside one `sum` where the
  push-down applies (below). An operand that gives a node gives that node's
  call below the root cast of its own tree, where that tree has one, in an
  output field of its own under a name that the planner makes (ADR 0075), one
  field for each Measure and condition (ADR 0084, rule 9). A Unionset's
  composed branch that gives partial aggregates, and whose demand holds the
  node's condition for the Measure, gives the node in the Measure's field
  instead. The formula above reads the node's field under the rest of the
  composition's own tree, its root cast included.

**A composed Metric is not refused.** Its signature today follows the
Joinset's shared-name rule (above): `Column(<operand>.<name>)` where the join
has a preserved side, and `Coalesce` of both operands' columns on a `Full`
join. Either form is an operand's finished value, under that operand's own
site's body and filter. Compile does not lower the composition site's own `expr`
or filter for it (ADR 0065 D3), so that signature gives the planner no formula.
Planning a composed Metric therefore needs compile to lower the composition
site's body, and it ships with that change.

The rest of this section reads the semantic model `shop`. Its Dataset `orders`
reads `warehouse.orders`:

| order_id | customer_id | region | channel | amount | ordered_at |
|---|---|---|---|---|---|
| o1 | c1 | eu | web | 100 | 2026-10-05 |
| o2 | c1 | eu | app | 50 | 2026-10-20 |
| o3 | c1 | us | web | 200 | 2026-10-10 |
| o4 | c2 | us | NULL | 10 | 2026-10-25 |

Its Dataset `refunds` has two rows, each with `amount` 30. Its Joinset
`orders_customers` joins `orders_j` (`warehouse.orders`) many-to-one to
`customers_j` (`warehouse.customers`: c1 is `vip`, c2 is `basic`) on
`customer_id`. `ordered_at` is a temporal Dimension, and only `customers_j`
maps the Dimension `segment`. `revenue`, `refund_total` and each Metric that
reads `revenue`, but `regions_with_revenue`, declare `data_type: double`, so
compile stores `revenue` as `CAST(sum(amount) AS double)` (ADR 0089, rule 17).

| Name | Role | `agg:` | Expression |
|---|---|---|---|
| `revenue` | Measure | `sum` | `amount` |
| `order_count` | Measure | `count` | `order_id` |
| `buyers` | Measure | `count_distinct` | `customer_id` |
| `vip_orders` | Measure | `count` | `CASE WHEN segment = 'vip' THEN order_id END`, on `orders_customers` |
| `revenue_adj` | Metric | none | `CASE WHEN region = 'eu' THEN revenue * 1.2 ELSE revenue END` |
| `revenue_eu_only` | Metric | none | `CASE WHEN region = 'eu' THEN revenue * 1.2 END` |
| `revenue_eu_or_zero` | Metric | none | `CASE WHEN region = 'eu' THEN revenue * 1.2 ELSE 0 END` |
| `revenue_eu_bonus` | Metric | none | `CASE WHEN region = 'eu' THEN revenue + 100 ELSE revenue END` |
| `avg_order_value_adj` | Metric | none | `CASE WHEN region = 'eu' THEN revenue / order_count * 1.2 ELSE revenue / order_count END` |
| `revenue_early_adj` | Metric | none | `CASE WHEN ordered_at < '2026-10-15' THEN revenue * 0.9 ELSE revenue END` |
| `vip_weighted_revenue` | Metric | none | `CASE WHEN segment = 'vip' THEN revenue * 2 ELSE revenue END`, on `orders_customers` |
| `refund_total` | Metric | `sum` | `amount`, on `refunds` |
| `regions_with_revenue` | Metric | `count` | the formula of `revenue_adj`, with `data_type: bigint` |
| `revenue_per_region` | Metric | none | `revenue / regions_with_revenue` |
| `avg_customer_revenue` | Metric | `avg` | `revenue` |
| `region_label` | Metric | none | `CASE WHEN region = 'eu' THEN 'EU' ELSE 'other' END` |

**A Metric's `agg:`.** A Metric is row-level where its formula reads no Measure
and no Metric, down each link, a link to a row-level Metric with no `agg:` read
through that Metric's formula. Compile decides this on the formula as written,
and the fold keeps it: where the fold of a Metric's formula would leave no link
to a Measure or a Metric, compile stores the formula as written (ADR 0089, rule
23). So `CASE WHEN FALSE THEN order_count ELSE 1000 END` stays a formula over
the aggregates, not a row-level Metric. The stored tree of a Metric with `agg:`
is the call of its `agg:` where the Metric is row-level, and its formula, with
no such call, where it is not. Compile lowers a row-level Metric's `agg:` as a
Measure's: its tree is `Aggregate(<agg:>, <formula>)`, cast to `data_type` where
the types differ (ADR 0089, rule 17), and its `filter:` is a second root, which
goes on the call's `filter` clause (ADR 0071, *Aggregation-level conditions*).
The planner plans it as a Measure, and its aggregate reads its function's
additivity only (ADR 0083, rule 1). Its formula over Dimensions is one
expression, as a Measure's `expr:` is, not a mapping followed by an expression.
`select refund_total` gives 60, as a Measure that sums the same column does.

**A row-level Metric with no `agg:`** is planned as a Dimension whose `expr:`
is its formula. A request that selects it groups by its value, a request
predicate on it applies before aggregation (ADR 0071, *Placement is read from
the role*), and a formula that links it reads a Dimension. So in the formula
grain, the Grouped test, the push-down's pattern and the parts, a link to it
reads one Dimension, the Metric's own value, as a Dimension with the same
`expr:` reads, and reads no Metric. The row-level test and each test of a
Volatile call read the link through the Metric's formula, down each link:
`region_flag2 = region_flag * 2`, over a row-level `region_flag`, is row-level
too, and a Metric with `agg: count` over a constant counts rows, as `count(1)`
does. Its
`filter:` gives it the value NULL on a row where the filter does not hold:
compile stores its tree as `CASE WHEN <filter> THEN <formula> END`, with no
second root, so each reader of it reads the filter. A constant formula is one
value on every row. A row-level body that reads it is
still refused, as one that reads any Metric is
(`CompileError::AggregateReferenceInRowLevelBody`). `select channel, region_label`
gives (web, EU), (web, other), (app, EU) and (NULL, other), and `select
region_label, revenue` gives EU 150 and other 210.

**A Metric's `agg:` over a post-aggregate formula.** Where the formula reads a
Measure or a Metric, down each link, the `agg:` aggregates the formula's values
at the Metric's **formula grain**: the demand that finishes the Metric,
together with each Dimension and each column under `auto` that the formula
reads outside an aggregate, at its value before bucketing. The Metric is
computed in two aggregate steps, in a subtree of its own (*Two aggregate steps*,
below):

- **step 1** groups the rows by the formula grain and computes each aggregate of
  the formula. The Metric's `filter:` narrows each of them (ADR 0071,
  *Aggregation-level conditions*), not the groups: a group of step 1 is a group
  of the rows;
- a projection computes the formula on each group of step 1;
- **step 2** aggregates those values with the Metric's `agg:`, grouped by the
  demand.

Step 2 is the Metric's own aggregate over the formula's values, as its author
declares, and not a combine of the partials of one aggregate (`CONTEXT.md`
**Re-aggregation**), so no additivity judges it.
`agg:` stays in the Metric's definition and step 2 is the planner's node, so the
tree holds no aggregate call over an aggregate call (`semstrait-sql`'s
`CallError::AggregateInAggregate`). Compile stores the formula with no root
cast, and the planner casts step 2's value to `data_type` directly above step
2, where the two types differ (ADR 0089, rule 19). A temporal Dimension that the demand also selects
at a granularity is in step 1's grouping twice: at the bucketed value that the
demand selects, and at its value before bucketing, an output field that the
planner creates, which carries a reserved name (ADR 0075). A formula grain with
no grouping column has one group, over no rows too, as a total has.

`select channel, regions_with_revenue` has the formula grain (channel, region).
Step 1 gives:

| channel | region | `revenue` | formula |
|---|---|---|---|
| web | eu | 100 | 120 |
| web | us | 200 | 200 |
| app | eu | 50 | 60 |
| NULL | us | 10 | 10 |

Step 2 counts the formula's values: web 2, app 1 and NULL 1. A formula that
reads no Dimension and no column outside an aggregate has the demand as its
formula grain, so each group of step 2 holds one value: `select
avg_customer_revenue` gives 360, the `avg` of one value. An average over the
customers needs a formula that reads `customer_id` outside an aggregate: with
`agg: avg`, `CASE WHEN customer_id IS NOT NULL THEN revenue END` gives 180, the
average of c1's 350 and c2's 10. With `agg: sum`, the formula of
`revenue_early_adj` gives 330 for October at Month: step 1 groups by the month
and by each order's day.

A formula that links such a Metric reads the value that its step 2 finishes,
never its formula: reading the link puts no tree in its place (ADR 0088, rule
15). The linked Metric is finished at the grouping of the step that reads it.
Its subtree joins the reader's subtree on that grouping: on the demand's, or,
where the reader takes its parts or has an `agg:` over a formula, on the
grouping of the reader's step 1, directly above that step. By channel,
`revenue_per_region` gives web 150, app 50 and NULL 10. No fold reads the link
as a literal (ADR 0089, rule 23). The reader's `filter:`, and each filter
inherited above it, narrows the step-1 aggregates of the linked subtree after
the linked Metric's own, and removes none of its groups, as the Metric's own
`filter:` does. So the linked Metric under a reader's filter is a subtree apart
from the linked Metric alone: by channel, `eu_regions = regions_with_revenue`
with `filter: region = 'eu'` gives web 1, app 1 and NULL 0, beside
`regions_with_revenue`'s web 2, app 1 and NULL 1.

**A Metric with no `agg:`** whose formula reads a Measure or a Metric may also
read, down every Metric that it inlines, a Dimension or a column under `auto`,
outside an aggregate. Such a name extends the demand of the frame, as each name
that a tree reads does (ADR 0073, *One decision shape, checked per body*). It
never extends the grouping of the output, so ADR 0062's "Nothing the caller
demands is ever added to the grouping" holds: where it groups, it groups step 1
of a subtree of the Metric's own, and step 2 drops it (*The parts*, below).

**The reference value.** Such a formula has a reference value, R, in each group
of the demand that finishes the Metric. R splits the group's rows into
**parts**. On a row, the formula selects one result of each `case` that it
reaches whose conditions read no Measure and no Metric: the first result whose
condition is TRUE, else the `else`, so a row whose condition is NULL goes to the
next result. A row that reaches such a `case` reaches each of its conditions
where no earlier condition is TRUE, and a `case` inside that condition selects
on each row that reaches the condition. A **piece** is a largest subtree outside
those conditions that reads no Measure and no Metric, reads a Dimension or a
column, and holds no `case`. Two rows are in one part where they select the same
results and hold the same value, NULL included, of each piece that those results
read. A `case` whose condition reads
a Measure or a Metric selects nothing on a row: it is computed on each part, and
the pieces of its condition split the rows as other pieces do. R gives the
formula's value on each part, and adds those values as a `sum` adds rows: it
skips NULLs, and a group with no part, or with NULL parts only, is NULL. A root cast to the Metric's `data_type` applies once, after the
parts are added. By channel, `revenue_adj` splits web into the eu part, o1, and
the `else` part, o3, where the formula gives 120 and 200, so R is 320; app is 60
and NULL is 10. An extra row o5 (c3, region NULL, web, 40, 2026-10-12) would go
to the `else` part, and web would be 360. Where a formula has R, each case
below that plans it gives R. A formula that reads no Dimension and no column
outside an aggregate has no R. Its value is the projection over the final
aggregates.

**The cases.** The planner tries these cases in order:

- **No such name.** The formula reads no Dimension and no column outside an
  aggregate. It runs as one projection over the final aggregates.
- **Grouped.** The demand that finishes the Metric selects each Dimension that
  the formula reads outside an aggregate, with no granularity. The formula runs
  as one projection over the final aggregates, and any formula passes. A
  Dimension that the demand selects at a granularity, Day included, is not
  grouped: the formula reads its value before bucketing (ADR 0080, *Every
  filter reads the value before bucketing*). A column under `auto` is never
  grouped, because a request selects no column. A request predicate `D =
  literal` does not group by `D`.
- **The formula push-down.** The formula is in the formula push-down's pattern.
  The planner writes it into one `sum` in the frame's one aggregate step.
- **The parts.** Otherwise the planner computes R on its parts, in a subtree of
  two aggregate steps (*The parts*, below). Where the formula's type is not
  numeric, its parts cannot be added, and the planner refuses the request with
  `PlanError::UngroupedMetricFormula { metric, name }`. `name` is the first
  name, in the stored tree's order, that the formula reads outside an aggregate
  and the demand does not select with no granularity. Compile accepts the
  Metric, because the outcome depends on the request.

In each case, a request predicate on the Metric sits above the node that
finishes it (ADR 0083, rule 5). By region, `revenue_adj` is grouped and gives
eu 180 and us 210, and so is `revenue_eu_bonus`, with eu 250 and us 210. The
refusal's message names its fix: group the request by `name`.

**A formula over an aggregate step reads that step's output fields.** Where a
projection computes a formula over an aggregate step, each name that the
formula reads outside an aggregate and that the step groups by is read as the
output field that the step gives for that grouping, and no tree is put in its
place: a Dimension, a row-level Metric with no `agg:`, and a column under
`auto`. The planner finds that field by the link, or by the column leaf of a
column under `auto`, never by equal trees: two Dimensions with equal trees are
two semantics. Where the step groups a temporal Dimension twice, the formula
reads the field of its value before bucketing (ADR 0080, *Every filter reads
the value before bucketing*). The leaf names that output field (ADR 0075),
never a column of the scan (`CONTEXT.md` **Column**). This holds in the Grouped
case, in the projection that computes the formula on each group of step 1, and
in the parts, where each piece reads its column. It holds above the
re-aggregation of a frame that gives partial aggregates (ADR 0071, *A tree
gives the computation, and the role gives the placement*), and above the join
of two subtrees (ADR 0073, stage 5c). A link to a Measure stays a copy: its
aggregate node is one entry of the step (ADR 0071), and the projection reads
that entry under the rest of the Measure's tree, its root cast included. Below
the aggregate step, in the row-level projection and in the push-down's per-row
terms, each link reads as a copy of the linked tree (ADR 0088, rule 15). By
region, `revenue_adj` reads the step's field `region`, and gives eu 180 and us
210.

**The formula push-down** writes the formula into one `sum` over the rows of the
frame, in the frame's one aggregate step, beside the request's other aggregates.
The rewrite has three parts, and the paragraphs below give each:

- each link becomes its row argument, cast per row, and a `count` link becomes
  `CASE WHEN x IS NOT NULL THEN 1 ELSE 0 END` over its argument `x`;
- the formula below its root cast becomes the argument of one `sum`, with the
  final cast above that `sum`;
- the Metric's `filter:`, with the one `filter:` that the links share, is that
  `sum`'s FILTER clause.

`select channel, revenue_adj, buyers` is then `sum(CASE WHEN region = 'eu' THEN
CAST(amount AS double) * 1.2 ELSE CAST(amount AS double) END)` and
`count(DISTINCT customer_id)`, in one aggregate step grouped by `channel`: web
320 / 1, app 60 / 1 and NULL 10 / 1. By channel, `revenue_eu_only` gives web
120, app 60 and NULL NULL: the NULL channel's one row is us, and a missing
`else` adds nothing. `revenue_eu_or_zero` gives web 120, app 60 and NULL 0.
`select ordered_at at Month, revenue_early_adj` gives 330 for October: each row
reads its own day, so o1 and o3 take the `then` branch. The bucketed value
2026-10-01 would give 360 × 0.9 = 324.

The push-down equals R. In one part of R, each factor `t` and each condition is
constant, so the `sum` of `t * x` over the part's rows is `t` times the part's
value of the Measure, which is R's value for the part. A `0` result adds 0 in
both, and a NULL result or a missing `else` adds nothing in both. This holds
where each operation that the push-down moves into the `sum` is exact (below).
A non-zero constant is added once per row, and not once per part: in total, the
parts of `revenue_eu_bonus` give 460 (eu 250, us 210), and one `sum` would give
560, so that formula takes the parts.

**The pattern.** The formula, below its root cast to `data_type`, is a term, or
a `case` whose every result is a term, `NULL`, the literal `0` (where no
filter applies, *Filters*), or a missing `else`. No result is a `case`. A term is `M`, `-term`, `term * t`, `t * term`
or `term / t`, where:

- `M` links a Measure, or a row-level Metric with `agg:`, whose aggregate is
  `sum` or a `count` with no DISTINCT, and which declares no narrower
  additivity;
- `t`, and each condition of the `case`, read no Measure and no Metric. Each
  may be any expression of the row, a `case` included;
- `t` calls no Volatile function (ADR 0091), down each link: one `sum` calls it
  once for each row, and R calls it once for each part where it is outside each
  piece and each condition of a `case` that selects, so `CASE WHEN region = 'eu'
  THEN revenue * random() ELSE revenue END` takes the parts.

A link to any other Metric does not match. A formula outside the pattern takes
the parts. In `shop`, `revenue + 100`, `revenue / order_count` and `buyers * 2`
are not terms.

**Exactness.** The push-down equals R exactly where each operation that it
moves into the `sum` is exact in its type, and up to floating-point rounding
where that type is floating: there an intermediate value past the type's range
gives an infinity on a row where R's part may not. So a negation and a `*` pass
on floating types, and on decimals, where a `*` passes only where p1 + p2 + 1 ≤
38, so that its type holds every product of its operands. Above that, ADR 0092,
rule 3, caps the precision at 38 and lowers the scale, to no less than 6, so a
product of two rows can overflow where their part's product does not: over the
`decimal(38,6)` rows 6 × 10^31 and −6 × 10^31 of one part, a `decimal(1,0)`
rate of 2 gives 0 for the part, and the first row overflows. A `/` passes only
where its type is floating. A negation, a `*` or a `/` on an integer type, and
any other `*` or `/`, makes the formula take the parts. An integer `/` cuts on each row: over the rows 1, 1 and 4
of one part, a bigint `revenue / 2` gives 0 + 0 + 2 = 2 per row, and 6 / 2 = 3
for the part. An integer `*` or negation can overflow on a row where its part
does not, and ADR 0092 makes integer overflow an error: over the bigint rows
2^62 and −2^62 of one part, `m * 2` gives 0 for the part, and the first row
overflows.

**Casts.** Each link becomes its row argument, cast per row to the Measure's
value type, which is the type of the Measure's stored tree: `revenue`'s link
becomes `CAST(amount AS double)`. That cast must be a no-op, a cast to a
floating type, or a widening (ADR 0086, rule 9) whose target holds every value
of the argument's type: an integer to a wider integer, an integer to a
`decimal(p,s)` where p − s holds its digits (5 for `smallint`, 10 for
`integer`, 19 for `bigint`), or a decimal to one of more precision at the same
scale. A `bigint` cast to `decimal(2,0)` is a widening by rule 9, and it fails
on a row of 100 where its part, 100 and −100, sums to 0. Any other cast makes
the formula take the parts, because a narrowing cast can fail or cut on a row
where R's part does not. So a `revenue` that declares `decimal(18,2)` passes
over a `decimal(18,2)` column, where its cast is a no-op, and takes the parts
over a `decimal(30,2)` column.

The final cast, `Cast(sum(<per-row term>), <the Metric's value type>)`, sits
above the `sum` where the two types differ. The Metric's value type is the type
of its stored tree, root cast included. Where compile stored a root cast, the
final cast is that cast, moved above the `sum`. Where it stored none, the
planner writes one, because a `sum` over `decimal(p,s)` gives `decimal(38,s)`
(ADR 0087, rule 17). These are the casts that the planner writes (ADR 0071, *A
tree gives the computation, and the role gives the placement*; ADR 0089, rule
19).

**Filters.** The Metric's own `filter:` is the FILTER clause of the one `sum`,
as it is the condition of each aggregate node of the Metric (ADR 0071,
*Aggregation-level conditions*). Each link carries the same `filter:`, by equal
stored trees, or none does. That shared `filter:` joins the FILTER clause with
`And`, and over the rows that it keeps the push-down is again R's sum. Where
the links' filters differ, the formula takes the parts. Over a Unionset, each
Measure's aggregation condition is its own column on each branch (*A
Unionset*, **Rows**), so the calls of two Measures share no FILTER clause, and
the formula takes the parts in both passes (ADR 0073, *Two passes*). A result
`0` and a count term give 0 for a part whose rows the filter excludes, so a formula with
either one and a FILTER clause takes the parts too: in a group where no row passes the filter, R gives that 0, and the
filtered `sum` gives NULL. With `revenue` filtered by `channel = 'tv'`,
`revenue_eu_or_zero` in total gives 0 under R, and NULL as one filtered `sum`;
with `order_count` filtered so, `CASE WHEN region = 'eu' THEN order_count END`
gives the same 0 and NULL.

**A count.** A `count` link becomes `CASE WHEN x IS NOT NULL THEN 1 ELSE 0 END`
over its argument `x`, inside its term, where no FILTER clause applies
(*Filters*). So a selected row whose `x` is NULL counts 0, as in R. A `sum`
over no rows is NULL (ADR 0087, rule 17), so over an
empty input a formula with a count term gives NULL, where the count alone gives
0. The planner writes no `COALESCE` (*Consequences*).

**The parts** compute R as it is written:

- the row-level projection (ADR 0073, stage 2) gives the **part key**: one
  column for each `case` that selects, a `case` over its conditions whose
  results are the positions of its results, and NULL on a row that does not
  reach it; and one column for each piece, its value on a row whose selected
  results read it, and NULL on another. Each reads its Dimensions before
  bucketing;
- **step 1** groups the rows by the demand and the part key, and computes each
  aggregate of the formula under its own conditions (ADR 0071,
  *Aggregation-level conditions*);
- a projection computes the formula on each part: each such `case` takes its
  result by the key, and each piece reads its column;
- **step 2** is a `sum` of the parts' values, grouped by the demand, and cast
  to the Metric's value type where the two types differ, as the push-down's
  `sum` is (*Casts*).

Each part of R is one group of step 1, so the parts give R on every input, and
no operation moves into or out of an aggregate. They serve each formula that
the push-down does not: a non-additive Measure, a constant, a quotient of two
Measures, a nested `case`, an inexact operation, a narrowing cast, and Measures
with different filters. By channel, `avg_order_value_adj` gives web 320, app 60
and NULL 10. In total, its eu part gives 150 / 2 × 1.2 = 90 and its `else` part
210 / 2 = 105, so it gives 195, and `revenue_eu_bonus` gives 460. For a
non-additive Measure under two results that are not NULL, R adds the parts'
values (*Consequences*): in total, `CASE WHEN channel = 'web' THEN buyers ELSE
buyers END` gives 3, c1 in the web part and c1 and c2 in the other, where
`buyers` gives 2.

**Two aggregate steps.** A Metric whose `agg:` aggregates a post-aggregate
formula, and a Metric that takes the parts, read a subtree of their own. The
frame that finishes the Metric reads its sources a second time, under the same
pre-aggregate predicates, and builds step 1, the formula and step 2 over them.
Each other aggregate of the request stays in the frame's subtree, in its one
aggregate step. Inside the Metric's subtree, each rule that reads the grouping
reads step 1's: a marker's partition (ADR 0076), the partials that an operand
or a branch gives, and the grain test. The two subtrees join with an `Inner`
join whose condition is `IsNotDistinctFrom` on each grouping column of the step
that reads the Metric, the demand's or a reader's step 1 (above), or with a
cross join where that grouping has no column, where both read the same source
rows. They read them under the same predicates, so they hold the same groups,
and the join loses none: a plain `Eq` would drop the NULL channel's row. That
holds where no grouping column of the join, no join condition or relationship filter
of a Joinset that both subtrees read, and no pre-aggregate predicate calls a
Volatile function, down each link (`CONTEXT.md` **Volatility**). Each read
draws such a call apart. Each read also draws apart a Volatile call in an
argument or an aggregation-level condition that both subtrees compute, down each
link (*Consequences*). The planner then returns `Advisory::VolatileInTwoReads`
beside the plan. A subtree that reads the union of a Unionset again joins as *A
Unionset*, **One form for every read**, gives. Two such Metrics
whose step 1 has the same grouping and the same filters share one subtree, and a
request whose
aggregates all sit in one subtree builds no join.

- A request predicate on such a Metric alone sits directly above its step 2 and
  its cast. The inner join then drops the whole output row, as a predicate on
  an aggregate drops its group. Where the subtree reads the union again and
  joins `Left` or `Full` (*A Unionset*, **One form for every read**), the
  predicate also sits above the joins, so it drops the whole output row.
- A formula or a predicate that reads two subtrees sits above the join.
- **The names of the join.** Both subtrees carry each grouping column of the
  join under one name. So the subtree that joins another writes
  each of its grouping columns as an output field that the planner creates,
  under a reserved name, and the subtree that it joins keeps its names (ADR
  0075, *Names are unique, and the planner's own names are reserved*): a
  Metric's own subtree joins the frame's subtree, and a linked Metric's subtree
  joins its reader's subtree (above): the frame's, or the reader's step 1.
  Where no frame's subtree is built, each other subtree joins the first one,
  in the order of the request, and the subtrees that one tree links in the
  order of its arguments. Two subtrees that join one write different
  `<purpose>` parts. A Metric's own subtree keeps the name of its value, so a
  request predicate on it binds above its step 2. Only a kept link reads the
  value of a linked Metric's subtree, so that value is an output field that
  the planner creates, under a reserved name, and the kept link names that
  field (ADR 0088, rule 4). Two linked subtrees of one Metric, under two
  filters or at two groupings, write different `<purpose>` parts. By channel,
  a Metric with `expr: regions_with_revenue - eu_regions` reads two linked
  subtrees of `regions_with_revenue`, one of them under `region = 'eu'`, and
  gives web 1, app 0 and NULL 1. A cross join compares no column and renames
  none.
- `BinaryOpKind` gains `IsNotDistinctFrom`, with a declaration (ADR 0092, rule
  1). It lowers to Substrait's `is_not_distinct_from`, and DataFusion 55 plans
  it as `HashJoinExec … NullsEqual: true`.
- Over a mode-`unique` Unionset with no guaranteed keyset, each subtree's
  distinct compares the fields that it reads (ADR 0079), so the rows that each
  subtree keeps can differ, under `Advisory::UniqueWithoutKeyset`. Their groups
  are the same.

`select channel, regions_with_revenue, buyers` reads two subtrees: the one of
step 1 and step 2 above, and one with `buyers` by `channel`. It gives web 2 / 1,
app 1 / 1 and NULL 1 / 1. With `where regions_with_revenue > 1`, it gives web
2 / 1 only. `select channel, revenue_eu_bonus, buyers` gives web 400 / 1 (eu
part 200, `else` part 200), app 150 / 1 and NULL 10 / 1.

**On a Joinset.** A Metric's `expr:` can read names that no operand maps alone.
Compile then lowers the Metric on the Joinset, and each operand gives the names
of the body that it maps (ADR 0077, *A constituent serves only the names it
maps*). So the planner reads both operands and builds the join, and it plans
the Metric above the join as this section says. A Measure's `expr:` reads the
same way. `vip_weighted_revenue` on `orders_customers` reads `segment` from
`customers_j` and `revenue` from `orders_j`. By region it gives eu 300 and us
410, through the push-down above the join.

On a Joinset, the push-down needs each link's Measure to read the joined rows as
they are (*A Joinset*, **Fan-out**). Each value leaf of the link's row argument
walks down its qualifier chain to a Dataset. At each Joinset on the path, the
walk reads that Joinset's declared cardinality for the operand that the leaf
enters: `many_to_one` and `one_to_many` repeat the "one" side, `many_to_many`
and an omitted cardinality, which resolves to it, repeat both, and `one_to_one`
repeats neither. The walk enters each branch of a Unionset. A Grainset on the
path gives aggregates, not rows, so the link's Measure does not read the joined
rows as they are. A `Coalesce` walks both of its chains. Only value leaves walk: a condition and a factor `t` may read any
operand. A value that reads no column, such as the literal of `count(1)`, reads
no operand, so it counts the joined rows. Where a link's Measure does not read
the joined rows as they are, the formula takes the parts, in which each Measure
reads the Joinset's rows as *A Joinset* says. A Measure lowered on a Joinset
reads the same walk (ADR 0077).

`vip_weighted_revenue` passes, because `segment` is a condition and `amount`
comes from `orders_j`, the many side, and so does the Measure `vip_orders`,
which gives eu 2 and us 1 by region. `revenue * rate`, with `rate` on a table
that `orders` joins many-to-one, passes too: `rate` is a factor. On
`orders ⋈ order_lines` (*A Joinset*, **Fan-out**), `CASE WHEN kind = 'gift'
THEN revenue END` takes the parts: one `sum` over the joined rows would count an
order once for each of its lines, and step 1 reads each order once in each part.
By region it gives eu 100 and us 200, where one `sum` would give eu 200.

**Nesting.** The push-down's `sum` is an Additive aggregate, placed by the rules
above. A Joinset operand pre-aggregates it where its argument reads that operand
alone and the join does not null-extend that operand, and otherwise the `sum`
reads the joined rows above the join. A null-extended row reads NULL for that
operand, so a result `0` or a count term gives it 0 above the join, where no
partial of it exists below: on `customers` left-joined to `orders`, a segment
whose customers have no orders gets 0 from `CASE WHEN region = 'eu' THEN
revenue ELSE 0 END` above the join, and NULL from partials of `orders`. Its
partials combine again by plain `Sum`, a count term's too. `select segment,
revenue_adj` on `orders_customers` gives vip 380 and basic 10: `orders_j`
pre-aggregates the `sum` by `customer_id`, and the Joinset sums the partials
above the join. On a Unionset whose branches are not disjoint, in mode `all`
and mode `unique` alike, each branch gives rows, and the one `sum` sits above
the union (*A Unionset*): a
branch outputs each linked Measure's row argument, never the formula's value on
its row. On a Unionset of two Datasets that split the rows of `orders`, `select
channel, revenue_adj` gives web 320, app 60 and NULL 10, as on `orders`. A
subtree of two aggregate steps nests as the frame's subtree does: its step 1 is
placed by the rules above at step 1's grouping, and its step 2 sits above it.
On a Joinset, the subtree's row-level projection sits above the join (ADR 0073,
stages 1 and 2), so the part key reads the joined rows, as R does: an operand
that pre-aggregates step 1 groups by the columns that the key reads, never by
the key. On `customers` left-joined to `orders`, a segment whose customers have
no orders selects the `else` of `CASE WHEN region = 'eu' THEN revenue /
order_count ELSE order_count END`, and gets 0.

### Re-aggregating rows that are already aggregates

The rows an aggregate reads may already be aggregates (above), while the grain
comparison says the plan re-aggregates them. Then each aggregate that the
request needs is handled by its additivity:

- **Additive:** no advisory.
- **Non-additive:** `Advisory::LossyReaggregation`, and the plan proceeds (ADR
  0030). Example: `avg_cpc by campaign` over a daily table keyed by
  `(campaign, country, date)` drops `country` and `date`.
- **Semi-additive**, where the request drops or coarsens one of the aggregate's
  axes: the strategy applies (next section).

### A semi-additive aggregate's strategy

This applies on any source. When the request drops an axis of the aggregate, or
coarsens its granularity, the planner:

1. aggregates by the family within each axis value, grouped by the request's
   grain plus the axis: the time axis at the source's granularity, or at its own
   values where the source has none, and any other axis at its own values;
2. then applies the strategy across the axis:
   - `latest` / `earliest`: the inner aggregate is joined to the `Max` / `Min`
     of the axis per request group, and it keeps the rows at that value;
   - `average`: `Avg` over the inner aggregate.

These steps, the join included, are the aggregate steps of ADR 0073's stage 4.

**Each axis is a Dimension that the source serves.** The human ruled that a
source that exposes a semi-additive Measure or Metric and does not serve each of
its axes is a compile error. Compile does not check this today. The planner
relies on it: a source that serves the Measure or Metric serves its axes, so the
candidate test needs nothing more.

**An axis that the request does not select extends the demand.** The planner
adds the axis to the names that the source's subtree must serve, as ADR 0073
carries predicate leaves. The scan reads the columns that the axis's signature
names, and the row-level projection (ADR 0073's stage 2) gives the axis in its
output. The steps above read it to apply the strategy. The unit that applies the
strategy drops it in its projection to what its parent asked for (ADR 0073's
stage 7), so the output holds only what the request selects.

**The `latest` / `earliest` join compares with a NULL-safe equality**, on the
grouping columns and on the axis. So a group whose grouping value is NULL keeps
its rows. `Max` and `Min` skip NULL, so a group whose axis is NULL in every row
keeps its rows at NULL. The expression vocabulary grows only by ADR (ADR 0066),
and this ADR adds the comparison `IsNotDistinctFrom` to `BinaryOpKind`. It
lowers to Substrait's `is_not_distinct_from`.

`first` is the same as `earliest`, and `last` is the same as `latest`, so the
planner reads them as those two. A Snapshot's Measures are expected to carry a
strategy along the snapshot's time axis (`Snapshot { snapshotted_at }`).

The strategy reads one order: that of one temporal axis. When the request drops
or coarsens more than one axis of one aggregate, or `latest` / `earliest` would
apply across an axis that is not temporal, no order is decided. The request is
then refused with `PlanError::StrategyOrderUndecided { semantics }`.
A later decision decides the order.

### Aggregates that need different row sets

The rules above give each aggregate the rows it reads: the rows as the operands
give them, the rows deduplicated by one row identity, or the rows that a
strategy keeps at an axis value. When the aggregates of one request need more
than one of these row sets, the request is refused with `PlanError::MixedRowSets
{ semantics }`. Examples:

- a semi-additive `balance` with `latest` on `date`, beside `cost`, by month;
- the `Avg` of a fanned operand's column, deduplicated by its row identity,
  beside the `Avg` of a column of the many side, which reads every row.

How the planner answers such a request is
not ruled here. The refusal stays
until it decides.

### Advisories and refusals

- **Advisories:**
  - `Advisory::LossyReaggregation`;
  - `Advisory::FanOut`, only when the row identity is inferred;
  - `Advisory::VolatileInTwoReads`, where the two reads of a Metric of two
    aggregate steps draw a Volatile call apart (*A Metric*, **Two aggregate
    steps**);
  - `Advisory::VolatileCopied { semantics }`, where one frame computes the tree
    of one entry that calls a Volatile function, down each link, in two or more
    of its expressions, so two readers of the semantics on a row read two draws
    (ADR 0088, rule 15). The push-down's shared `filter:` is an expression of
    each link's filter entry;
  - the path ambiguity advisory of ADR 0064.
- **No advisory for a change of form that stays exact.** Examples are rows in
  place of partial aggregates, and a left-out operand. The plan's
  `semantic_meta` records it as trace.
- **Removed** from the earlier working design's roster: `PreAggregationSkipped`, `JoinElided` and
  `ConflictingDelivery`.
- **Refusals**, each where its section says:
  - ADR 0064's `NoCandidate` and `GrainNotServable` (A Grainset);
  - `PlanError::AmbiguousOperandCondition` (A Joinset);
  - `PlanError::StrategyOrderUndecided` (A semi-additive aggregate's strategy);
  - `PlanError::MixedRowSets` (Aggregates that need different row sets);
  - `PlanError::UngroupedMetricFormula` (A Metric);
  - `PlanError::FanOutMarkerNotPlaceable` (ADR 0076, **Where the marker has no
    place**).

### Amends

- **0026:** in *Decision Outcome*, "`grain:` is forbidden at complex levels" no
  longer holds for a composition that is a constituent of another composition:
  it may declare its granularity, which every constituent below it serves, and
  that granularity applies to every constituent below it, down to a constituent
  that declares its own; "must be repeated on every Grainset child" holds for a
  Dataset child, and a composed child carries the granularity that it declares
  or that ADR 0056 derives (0074's *Every request is planned against its grain
  first*).
- **0030:** the lossy re-aggregation advisory is raised where the grain
  comparison re-aggregates a non-additive aggregate over rows that are already
  aggregates (rows of a `Timeseries` or `Snapshot` Dataset); a semi-additive
  aggregate whose axis the request drops or coarsens applies its strategy
  instead; a Metric's aggregate node reads its function's additivity, narrowed
  by the effective additivity of the Measure it came from, once compile carries
  it, and by the Metric's own declared additivity.
- **0033:** a Unionset pre-aggregates per branch only in mode `all` and only
  when every aggregate the request needs (each demanded Measure, and each
  aggregate node of a demanded Metric) is additive or semi-additive; otherwise
  the branches give rows and one aggregate sits above the union, the form 0033
  rejected; the disjointness elision is a selected Dimension that folds to a
  distinct non-NULL `Literal` on every branch, through a composed branch's
  operands as ADR 0073 folds them to prune a branch, a Joinset on the way giving
  a literal for a name that both operands map only where both give the same
  one, and a Unionset or a Grainset only where every branch or child gives the
  same one, still distinct after the request's bucketing, in mode `all` only,
  where each Metric of two aggregate steps groups its step 1 by the demand's
  grouping alone.
- **0034:** The driver writes `NullFill` as a typed NULL for a Unionset branch
  that does not map a name, once the manifest lets a branch serve only the names
  it maps; "NULL fill on a Joinset" is a considered option 0074 did not take,
  and a Joinset operand carries no NULL-filled name.
- **0048:** the fan-out advisory reads no runtime keyset: fan-out is
  deduplicated by the fanned operand's guaranteed keyset, and the advisory is
  raised only where that operand has none, and so has no runtime keyset either.
- **0054:** answers the planner handoff: the rule for leaving out an operand
  reads the join type, and the deduplication takes the place of cardinality.
- **0056:** where existing ADRs say "grain" for the temporal roster, they mean
  the granularity; in *Three rules, three variants, two sites each*, rule A's
  "full shape unanimity (kind body *and* grain)" reads "full shape unanimity
  (kind body)": the granularity is no part of a Unionset's agreement, as *A
  Unionset agrees; a Grainset aligns on kind and differs on grain* says, and "A
  Joinset raises nothing", there, and "a **Joinset** requires nothing", in
  *Decision Outcome*, hold only for a Joinset that is no constituent of
  another composition, since the two refusals of a constituent's granularity
  join the three rules, each with its own variant and two sites; "forbidding
  only `temporal.grain:` on it", in *A Unionset agrees; a Grainset aligns on
  kind and differs on grain*, and "re-scoped to authoring", in *The grain is
  derived where it is unanimous, …*, no longer hold for a composition that is
  a constituent of another composition: its author may declare its
  granularity, which it carries in place of a derived one, and the derivation
  of the granularity, not of the temporal shape, stops at a constituent that
  declares one, so "absent otherwise", there, in
  *Decision Outcome* and in its `### Amends` entry for 0049, and "`grain` is
  empty wherever the constituents do not agree on one", in *Absence has one
  meaning per field*, hold only for a composition that declares none (0074's
  *Every request is planned against its grain first*); in *Consequences*,
  `GrainsetChildMissingGranularity` is neither re-cut nor retired, and stays for a
  Dataset child of a Grainset: compile refuses, with
  `CompileError::ComposedConstituentMissingGranularity`, a composition that is a
  constituent, carries no granularity, and publishes a temporal shape or is a
  child of a Grainset (0074's *Every request is planned against its grain
  first*).
- **0057:** every join type reads `Coalesce` for a name both operands map, and
  `Right` reads it `to` then `from`, so the fixed `from`-then-`to` order no
  longer holds for `Right`.
- **0059:** the driver writes `NullFill` as a typed NULL for a Unionset branch
  that does not map a name, once the manifest lets a branch serve only the names
  it maps.
- **0062:** "grain" for the temporal roster means the granularity, and
  "requested grain" is read as the requested granularity.
- **0063:** "requested grain" in 0063 is read as the requested granularity.
- **0064:** a stand-in sub-composition must satisfy the rule for leaving out an
  operand: every demanded name read from what remains, no read row dropped, no
  relationship filter; among its grains that carry a granularity, a Grainset
  picks the one that serves the whole demand and has the coarsest
  granularity equal to or finer than the requested one; with no requested
  granularity on the time axis, it picks the grain with the coarsest granularity
  that serves the whole demand if the request does not select the time
  axis, and such a grain with the finest granularity, whose values are the least
  bucketed, if the request selects the time axis without a granularity;
  "requested grain" is read as the requested granularity.
- **0065:** in D2 and the §4d rule, a name that both operands of a Joinset
  map reads no operand's condition, only the composition's own filter, and is
  refused with `PlanError::AmbiguousOperandCondition` where the composition has
  no own filter for it and an operand's signature carries one; in D3 and the
  §3 rule, a composed Metric's signature is its site's body lowered on the
  composition, with the composition's own filter, so a composed Metric
  re-lowers its body; an inlined `Aggregate` node carries the effective
  additivity of the Measure it came from, with its axes and strategy; in the
  per-join-type table, where both operands map a name, `Inner`, `Left` and
  `Full` read `Coalesce` from then to, and `Right` to then from; a constituent
  serves only the names it maps; in *A Grainset routes and does not read*, "it
  holds no grain of its own" holds for a Grainset that is no constituent of
  another composition, and one that is a constituent holds the granularity that
  it declares and still reads no rows (0074's *Every request is planned
  against its grain first*).
- **0066:** in ruling 7, `Avg` is non-additive and is not split into `Sum` and
  `Count`; in ruling 4, a `JoinNode` also joins two subtrees of one frame, on
  the condition that 0074's *A Metric* writes (`IsNotDistinctFrom` on each
  grouping column), with no relationship and no cardinality.
- **0071:** a subtree's aggregation may take more than one aggregate step (a
  distinct-rows aggregate below a deduplicated aggregate, an inner aggregate
  below a semi-additive strategy); a Unionset in mode `all` re-aggregates above
  the union only where its branches give partial aggregates; once a constituent
  serves only the names it maps, a `Left` or `Right` join reads a name that only
  the other side maps from its null-filled side, and a name that both map
  through `Coalesce`; a composed Metric's tree is its formula once the manifest
  lowers it, with the composition's own filter joined onto each inlined
  aggregate as a Dataset Metric's is; no condition is coalesced, and a name that
  both operands of a Joinset map reads no operand's condition, only the
  composition's own filter, in which a Dimension that both operands map is read
  in two copies of the whole filter, one on each operand, joined with `And`, and
  so is a request predicate over such a Dimension, each copy placed as 0071
  places a predicate; how a subtree re-aggregates partials is 0074's.
- **0072:** "requested grain" is read as the requested granularity, and "grain"
  for the temporal roster means the granularity.
- **0073:** each Unionset branch that is read carries the names it NULL-fills
  beside the names it serves: a name that its mask does not hold; the driver
  writes each such name as a typed NULL in that branch's own row projection,
  and for a Grainset branch, which keeps those names out of its pick and its
  drill-across, in the row projection of the child that answers alone or in the
  projection of its drill-across frame (0084's rule 7). An operand of a Joinset
  carries no NULL-filled name. In *The build order*, stage 5b's "`step_join`
  with `Inner`" reads, for a subtree that reads the union of a Unionset again:
  the join that *A Unionset*, **One form for every read**, gives (`Inner` over
  the same read, `Left` onto a frame that has aggregates of its own, `Full`
  with coalesced keys between two reads again, and `Left` for a later subtree
  over the left read after a `Full`).

The 0073 amendment adds only the NULL-fill names to its decision shape, where a
branch had no place to learn them.

### Consequences

- **The naming decision:**
  - *grain* now names the pair (x, y) and also a Grainset's child, so that
    decision rules
    on one of the two;
  - that decision still rules whether *Effective routing unit*, *Rollup*, *Lossy
    re-aggregation* and *Fan-out* stay; this ADR only brings their text in line
    with its rules;
  - the glossary has no word for the human's "time spine": the time axis that a
    temporal shape links (`occurred_at`, `snapshotted_at`).
- **The "requested grain" of ADRs 0062, 0063, 0064, 0071 and 0072** is read
  as the requested granularity.
- Bad, because over no rows a formula with a count term gives NULL, where the
  count alone gives 0. A request with no grouping whose rows are all excluded,
  as by `where channel = 'tv'` on `orders`, gets NULL from `CASE WHEN region =
  'eu' THEN order_count END` and 0 from `order_count`.
  `ELSE 0` gives NULL too, because a `sum` over no rows is NULL. The gain is one
  `sum` with no `COALESCE`: a `COALESCE` would also turn the NULL that
  `revenue_eu_only` gives the NULL channel into 0 (*A Metric*).
- Bad, because the push-down fails out of range on fewer inputs than R. R
  fails where one part's value of a Measure passes the Measure's type: a
  `revenue` that declares `decimal(18,2)` fails on a part whose sum passes
  10^16, and so does the grouped plan. The push-down casts each row, which does
  not fail on a column of the Measure's own type, and fails only where the
  finished value passes the Metric's type. The gain is the push-down for the
  common money Measure, a `sum` that declares its column's `decimal(18,2)` (*A
  Metric*).
- Neutral: the push-down is one more place where the planner writes a family
  other than the definition's: it writes a `count` as a `sum` (*Additivity
  decides whether a partial aggregate may be combined again*).
- Bad, because a Metric of two aggregate steps beside other aggregates reads
  the frame's sources twice and joins the two subtrees (*A Metric*, *Two
  aggregate steps*). The gain: each formula that reads a Dimension that the
  request does not group by, and each `agg:` over a formula, has a plan whose
  value is defined. On DataFusion 55 the plan holds two independent reads and a
  join with `NullsEqual: true`, and it keeps the NULL group.
- Bad, because a Unionset's union has one form for every read, so the
  subtree of a Metric of two aggregate steps takes the frame's form, even where
  a form of its own would read fewer rows. On a Unionset of two Datasets that
  split the rows of `orders`, beside a count distinct, `select channel,
  revenue_eu_bonus, buyers` reads the four order rows in both reads, where the
  subtree's read could read four partial rows; both forms give web 400 / 1, app
  150 / 1 and NULL 10 / 1. Where the two Datasets split `orders` by `region`
  and each maps it as a literal, the branches are disjoint, and a step 1 that
  groups finer than the demand makes the frame's read re-aggregate partials
  above the union: `select region, revenue_web_bonus`, which is
  `revenue_eu_bonus` with `channel = 'web'` for `region = 'eu'`, re-aggregates
  four partial rows (eu 250, us 310), where final aggregates per branch would
  leave no aggregate of the frame above. The gain: one decided branch set per
  Unionset frame, no second decide pass, and both reads hold the same groups,
  which the join of the subtrees needs (*A Metric*, **Two aggregate steps**).
- Bad, because the two reads draw a Volatile call apart (`CONTEXT.md`
  **Volatility**). Where a grouping column of the join (the demand's, or a
  reader's step 1), a join condition or relationship filter of a Joinset that
  both subtrees read, or a pre-aggregate predicate calls one, down each link,
  the two subtrees hold different groups or different rows. With a Dimension
  `draw` whose `expr:` is `random()`, `select draw, buyers,
  avg_customer_revenue` gives 0 rows, where one read gives 4. With a Dimension
  `sampled` whose `expr:` is `random() < 0.5`, `select channel,
  regions_with_revenue, buyers where sampled` gives a channel's two values from
  two samples, or loses the channel. Where an argument or an aggregation-level
  condition that both subtrees compute calls one, down each link, the two
  aggregates of one output row read two draws. The planner returns
  `Advisory::VolatileInTwoReads` beside the plan.
  DataFusion 55 shares no read between two subtrees: its Substrait consumer
  reads one relation tree and no `ReferenceRel`, and it copies a SQL CTE into
  each of its uses. The gain: one plan for each Metric of two aggregate steps,
  with no grouping sets (ADR 0066, rule 7), where `random()` is the one
  Volatile function of the catalog (ADR 0087).
- Bad, because for a non-additive Measure under two results that are not NULL,
  R adds the parts' values: in total, `CASE WHEN channel = 'web' THEN buyers
  ELSE buyers END` gives 3, where `buyers` gives 2. The gain: one value at every
  grain, which is the grouped value where the request groups by the formula's
  Dimensions. A Measure's `expr:` gives the Measure over the rows of each
  result.
- Bad, because R splits the rows by the result that a `case` selects, and by
  the value of a Dimension, of a row-level Metric with no `agg:` or of a column
  under `auto`. So in total `x * buyers` gives 5, where `x` is `CASE WHEN
  region = 'eu' THEN 2 WHEN channel = 'web' THEN 2 ELSE 1 END`, and `weight *
  buyers` gives 3 over a Dimension `weight` whose `expr:` is `x`, as `x` with
  the one condition `region = 'eu' OR channel = 'web'` does. The gain: one rule
  for every `case`, so R does not change where an author moves the factor into
  each result or `x` into a condition: `CASE WHEN x = 2 THEN 2 * buyers ELSE
  buyers END` gives 5.
- Bad, because the formula grain is implicit: `avg_customer_revenue` is the
  `avg` of one value in each group, and an average over the customers needs a
  formula that reads `customer_id` outside an aggregate. The gain: no change to
  the semantic model. A grouping that the Metric declares, as Cube's
  `grain.include` does, changes the semantic model.
- Bad, because a row-level Metric with no `agg:` adds its values to the rows of
  a request, as a Dimension does: `select channel, region_label` gives four
  rows, and over no rows it gives none. The gain: the value that each tool gives
  the same expression as a dimension.
- Bad, because a formula whose type is not numeric, in a request that does not
  group by each Dimension that it reads, has no plan: its parts cannot be
  added, so the planner refuses it with `PlanError::UngroupedMetricFormula`.
- Bad, because a Grainset that is a constituent of another composition
  declares a granularity that every child serves, so it serves no granularity
  finer than its coarsest child, and it answers only at that granularity:
  through one child, or through a drill-across only where its parent takes
  final aggregates from it. A Grainset of a Day and a Month child serves no
  request at Day to its parent, and a request that needs both children where
  a Joinset builds its join over it is refused with `GrainNotServable`:
  `select region, avg_clicks, budget` over `G ⋈ campaigns` (*A Grainset*). The
  refusal also holds where every aggregate is additive and the final aggregates
  would combine again exactly, as for `select date at Month, channel, clicks,
  budget` over a mode-`all` Unionset of such a Grainset and a Day Dataset. The
  gain: the granularity of its pick is fixed at compile, it needs no rule for
  the parent's demand, and no drill-across's final aggregates are combined
  again or joined.
- Neutral: `crates/` changes only for *A Metric*, *A Joinset*, *A Unionset*, *A
  Grainset* and *Every request is planned against its grain first*:
  - the planner gains the formula push-down, with its pattern, its rewrite,
    and its per-row and final casts;
  - the planner gains the subtree of two aggregate steps: the part key, the
    formula grain, step 1, step 2 and the cast above it, and the join of two
    subtrees; `BinaryOpKind` gains `IsNotDistinctFrom`;
  - the planner gains `PlanError::UngroupedMetricFormula { metric, name }`;
  - the planner gains `Advisory::VolatileInTwoReads`;
  - the planner gains `Advisory::VolatileCopied { semantics }`;
  - the planner deduplicates a result of a Measure lowered on a Joinset whose
    every value leaf reaches a repeated Dataset, through a `row_number` marker
    that also partitions by the result's position (*A Joinset*, **Fan-out**);
  - the planner plans a row-level Metric with no `agg:` as a Dimension;
  - compile gains `CompileError::NamedFilterReference { semantics, filter }`,
    lowers a row-level Metric's `agg:` as a Measure's, stores the formula of a
    Metric whose `agg:` aggregates a post-aggregate formula with no root cast,
    and stores a row-level Metric with no `agg:` and a `filter:` as a `case`
    over its formula;
  - compile lowers a Measure or a Metric on a Joinset where no operand maps
    every name that its `expr:` reads, and it refuses such a Measure with
    `CompileError::UnservedComposedSemantics` only where a value leaf's walk
    reaches a Grainset of which no single child, read at every depth below it,
    maps every name that the Measure's value and condition leaves read through
    it (ADR 0077);
  - reading a manifest keeps a link to a Metric whose `agg:` aggregates a
    post-aggregate formula (ADR 0088, rule 15), types the link as that Metric's
    value (ADR 0088, rule 5), and ADR 0084's rule 8 gives no Grainset signature
    to a Metric that reads one;
  - reading a manifest keeps each link to a Dimension, or to a row-level Metric
    with no `agg:`, that a projection above an aggregate step reads where the
    step groups by it (ADR 0088, rule 15);
  - the test `a_filter_that_reads_a_named_filter_links_it` in
    `crates/semstrait-manifest/tests/signature_filters.rs` becomes a refusal
    test;
  - compile accepts a composition as a constituent of another: its guard
    goes, and the planner keeps the names that a Grainset branch NULL-fills
    out of its pick and its drill-across;
  - in `semstrait-model`, `check_temporal`'s `NestedUnionset`, `NestedJoinset`
    and `NestedGrainset` arms stop raising `TemporalGranularityOnComplex`, so a
    composition that is a constituent may declare `extras.temporal.grain:`;
    its root arms keep it;
  - in `semstrait-manifest`, `composed_shape` reads a declared granularity,
    its granularity fold (`unanimous_grain`) stops below a composition that
    declares one while its type rank (`highest_rank_type`) still reads every
    claim that `reachable_claims` gives, the check
    of a declared granularity reads every constituent below, a Grainset's
    children included, down to one that declares its own, and
    compile gains `CompileError::ComposedConstituentMissingGranularity { composite,
    constituent }` and `CompileError::StatedGranularityNotServed { composite, stated,
    constituents }`, each raised in the temporal stage and restated in
    `validate()` (ADR 0056: one rule, one variant, two sites). Model validation's
    `GrainsetChildMissingGranularity`, which compile never raises, stays for a
    Dataset child of a Grainset;
  - the planner lets every constituent below a composition, down to one that
    declares its own granularity, serve only what a source at the
    composition's granularity serves, holds a composition's time axis, a root
    composition's too, at its granularity only where each of its values comes
    from a Dataset that holds it there (a Joinset by its `Coalesce` and the
    equalities of its condition, a Grainset by every child, and for the
    fan-out marker by the children that its decision reads), runs the pick of
    a Grainset that is a constituent at the granularity that it declares,
    runs its drill-across only where its parent takes final
    aggregates from it, and otherwise, where no single child serves its input,
    fails with `GrainNotServable` (*A Grainset*).

## More Information

Decided 2026-09-30, and on 2026-10-17 for the two inputs
of additivity and how the partials of each Additive aggregate combine, and on
2026-10-17 and 2026-10-18 for a Metric whose formula reads a Dimension or
a column, and a Metric's own `agg:`, and on 2026-10-18 and 2026-10-19
for a composed Grainset child, a composed Unionset branch and the granularity
of a composition that is a constituent. The decisions of 2026-10-18 on
the Metric formula read the research on
MetricFlow, Cube, Looker, Malloy and
Calcite, and an oracle on DuckDB and DataFusion 55. The decisions of 2026-10-18
on the Volatile call cover a Volatile
call in a value that both reads of a Metric of two aggregate steps compute.
They read the research on Calcite,
Spark, DuckDB, PostgreSQL, Substrait, MetricFlow, Cube, Looker and Malloy, and
an oracle on DataFusion 55 and DuckDB. The decisions of 2026-10-18 on
the link reading cover how a formula
over an aggregate step reads a name that the step groups by, how a formula
reads a link to a row-level Metric with no `agg:`, the names in the join of two
subtrees, and `Advisory::VolatileCopied`. They read the research on Calcite's conversion of a grouped query and its
`RexProgram`, Substrait's `ProjectRel`, and probes on DataFusion 55, with the
research of the Volatile call.

The grain test in *Every request is planned against its grain first* was
corrected on 2026-10-06 on the decision. It read Week as finer
than Month by the roster's order, so a weekly source rolled up to Month put a
week that spans two months into the month of its first day. It now says that a
source at Week serves no requested granularity of Month, Quarter or Year, as
Cube and MetricFlow treat Week. It was repaired in place rather than in a later
ADR, on the human's explicit call ("it is more of sharpening existing adr, then
reversing something"), so the rule is read once and in one place.
