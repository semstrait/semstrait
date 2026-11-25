---
number: 80
title: Only a frame that gives final aggregates buckets, and every filter reads the value before bucketing
status: accepted
date: 2026-10-05
source:
- CONTEXT.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md
- docs/adr/0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md
- docs/adr/0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md
- docs/adr/0079-a-unique-union-also-compares-its-keyset-and-a-re-aggregated-count-is-never-null.md
- the research on aggregation across joins (Yan & Larson, VLDB 1995; MetricFlow; Cube)
- the earlier bucket rules of ADR 0075 (since removed)
- the human's direction and rulings
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18 and 2026-10-19, and the research on composition constituents
links:
- target: 71
  kind: amends
- target: 73
  kind: amends
- target: 74
  kind: amends
- target: 75
  kind: amends
- target: 77
  kind: amends
- target: 72
  kind: amends
- target: 83
  kind: amendedby
- target: 84
  kind: amendedby
- target: 96
  kind: amendedby
---

# Only a frame that gives final aggregates buckets, and every filter reads the value before bucketing

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0083](0083-a-metric-declares-no-additivity.md): in *Every filter reads the value before bucketing*, "For a semi-additive Metric, the predicate sits above the node that applies the Metric's semi-additive strategy; that node is not ruled here" has no subject: the row "above the node that finishes the value" covers every Metric; in *Amends*, "a request predicate on a semi-additive Measure or Metric" applies to a Measure only; the rest of 0080 stands (0083's *Additivity belongs to an aggregate*).
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Only a frame that gives final aggregates buckets*, "a Grainset's picked child, and a stand-in" also covers each child of a drill-across, and "A frame that passes those final aggregates through, and a Unionset over disjoint branches" also covers a Grainset's drill-across; in *A predicate copy goes into its operand on every join type* and *A named filter is placed as a request predicate*, a predicate over a Dimension and an activated named filter go into each chosen child of a drill-across, as 0084's rule 7 says, and the child reads its own signature for the filter; the rest of 0080 stands (0084's *How they are joined*).
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *A named filter is placed as a request predicate*, an activated named filter on a Grainset also goes into the child that answers alone, which reads its own signature for it (0096's *A named filter on a Grainset*); the rest of 0080 stands.

## Context and Problem Statement

A request may ask for a temporal Dimension at a granularity (ADR 0062). ADR
0072 (*The function*) writes the bucket "at the subtree that groups the temporal
item, as part of the projection before that subtree's aggregate step", and a
request predicate reads the value before bucketing (ADR 0071). Below a join or a
union, more than one subtree groups: an operand or a branch that gives partial
aggregates groups too (ADR 0074). No ADR says which of them buckets, or which
value a join, a filter or a distinct above them reads. In *Ruled elsewhere, not
here*, ADR 0075 leaves to this ADR
where the planner buckets, and where each filter that reads the bucketed
Dimension is applied. ADR 0077 leaves to it where the planner applies an
activated named filter, and a request predicate on a semi-additive Measure or
Metric.

The example below runs through this ADR. `spend` and `bookings` are joined
`Inner` on `(campaign_id, date)`, one-to-one, with integrity `assumed`. Both are
daily:

- `spend`: (A, 03-10, cost 10), (A, 03-20, cost 5);
- `bookings`: (A, 03-10, revenue 100, channel `app`), (A, 03-20, revenue 50,
  channel `web`).

`select date at Month, cost, revenue` gives March's cost 15 and revenue 150.
Where a subtree below the join buckets, other requests give wrong answers:

- **A join that compares buckets.** `select date at Month, channel, cost`: if
  `spend` gives one row per `(campaign_id, month)`, that row matches both
  `bookings` rows of March, and each channel gets cost 15. The answer is 10 for
  `app` and 5 for `web`.
- **One side gives rows.** With `avg_cost`, `spend` gives rows, because `Avg` is
  not additive (ADR 0074). If the join compares months, the March row of
  `bookings` matches both `spend` rows, and revenue is 300.
- **A filter above the join.** With the relationship filter
  `date >= '2026-03-15'`, a `post_join_filter` that reads the bucket
  `2026-03-01` drops March.
- **A distinct.** In mode `unique`, a distinct over buckets merges two rows that
  the keyset tells apart (ADR 0079).

The research on aggregation across joins found no product, no
engine and no paper that joins rows or partial aggregates on a bucket: the
grouping `{campaign_id, month}` does not determine `date`, so the eager
aggregation theorem does not hold (Yan & Larson, VLDB 1995, Thm 1). It found
partial aggregates below a join exact under the three conditions that ADR 0074
already states. Where a product aggregates each side before a join (MetricFlow
with several metrics, Cube with several facts), each side gives final aggregates
at the request's grain.

Several kinds of filter also read the Dimension, and each has its own place: a
relationship filter, a request predicate, an activated named filter, a
Measure's or Metric's own `filter:` (an aggregation-level condition), and a
request predicate on a Measure or a Metric.

## Considered Options

- Only a frame that gives final aggregates buckets (chosen).
- **Each operand buckets below the join, whether it gives rows or partial
  aggregates.** The join then compares
  buckets: revenue 300 with `avg_cost`, and cost 15 for each channel.
- **A frame buckets where it groups by the bucket, unless a node above it reads
  the value before bucketing.** The answers are correct, and fewer partial rows
  reach the combine. But the rule holds a list of the nodes that read the value
  (a join condition on the Dimension, a filter above a combine, a mode-`unique`
  distinct, a semi-additive `by`), and each new combine must be checked against
  it.
- **No partial aggregates below a join**, as Looker and Malloy plan: each
  operand gives rows, and one aggregate sits above the join. A fanned operand's
  aggregates are then always deduplicated by its row identity, which is an
  inference where the operand has no guaranteed keyset (ADR 0074, *A Joinset*,
  **Fan-out** and **The row identity**).
  DataFusion 55 does not move an aggregate below a join by itself.
- **A Unionset's own rules choose its delivery** (ADR 0074 as written). Below a
  Joinset that reads an `Avg`, the disjoint-branch rule gives final aggregates
  to a parent that aggregates again (*The parent's demand fixes an operand's
  delivery*).
- **A relationship filter moved into each operand, below the bucket.** A
  relationship filter is the join's `post_join_filter`, applied after the join
  (ADRs 0066 and 0075).
- **A filter that reads the bucket.** `date >= '2026-03-15'` reads `2026-03-01`
  and drops March.
- **A predicate copy kept above a join, for a side that the join null-fills**
  (ADRs 0071 and 0074 as written). The human ruled that the planner acts on what
  semantic sources and relationships declare, never on which rows are stored.
- **A named filter on a composition as one boolean output field above the
  join** (ADR 0071 as written). It applies above the join a filter that each
  operand can apply.
- **A copy that keeps its operand in the plan.** A request that reads one
  operand would join the other only to check that operand's copy.

## Decision Outcome

In short:

- Only a frame whose own aggregate step gives final aggregates buckets, once. A
  frame that gives rows or partial aggregates gives each Dimension at the value
  before bucketing, so every node below the bucket reads that value. A Metric's
  subtree of two aggregate steps buckets in its step 1, and its step 2 reads
  the bucketed value (ADR 0074, *A Metric*).
- Partial aggregates stay below a join and in a Unionset's branches, as an
  intermediate step that the parent combines again.
- The parent's demand fixes an operand's delivery. A Unionset's rules say only
  how its branches give it.
- Each copy of a request predicate or of a named filter that reads only one
  operand's names goes into that operand, on every join type. The two copies of
  a filter over a Dimension that both operands map apply only on an operand that
  the plan reads.

### Only a frame that gives final aggregates buckets

ADR 0072's "subtree that groups the temporal item" (*The function*) is the
frame whose own aggregate step gives final aggregates (`CONTEXT.md`
**Delivery**). Where its demand takes final aggregates, that frame is:

- the scope root's frame, where it aggregates;
- the one operand that a Joinset keeps, whose aggregate the Joinset passes
  through (ADR 0074, **Leaving out an operand**);
- each of the disjoint branches (*The parent's demand fixes an operand's
  delivery*);
- a Grainset's picked child, and a stand-in, whose subtree answers for the frame
  (ADRs 0073 and 0074).

That frame buckets in its aggregate stage, after its pre-aggregate predicates
(ADR 0073, *The build order*, stages 3 and 4), and its aggregate groups by the
bucket. A frame that passes those final aggregates through, and a Unionset over
disjoint branches, has no aggregate step of its own, so it does not bucket
again: the bucket happens once (ADR 0072, *What a bucketed timestamp means*).

A frame that gives rows or partial aggregates does not bucket. Its demand still
carries the requested granularity (ADR 0073, *One decision shape, checked per
body*), which the grain comparison reads (ADR 0074, *Every request is planned
against its grain first*). Its partial aggregates are grouped by the value
before bucketing: by the day, where the request asks for the month. So every
node between the physical read and the bucket reads the value before bucketing:

- a join's condition and its `post_join_filter`;
- a `FilterNode` above a join or above a `CrossNode`;
- a Unionset's stack, and the distinct of mode `unique`, which compares the
  values at which a keyset identifies a row (ADR 0079).

A semi-additive aggregate's `by` keeps its value before bucketing too: its
partial aggregates are grouped by that value, and its marker, which sits after
the bucketing, orders by it (ADR 0076, *A marker gives each aggregate its
rows*).

A bucket is then only grouped by and output, as ADR 0072 says (*What a bucketed
timestamp means*). Rows and partial
aggregates never carry a bucket, so a join compares a bucket only between final
aggregates. Whether a Joinset may ask its operands for final aggregates is not ruled here.

In the example:

- `select date at Month, cost, revenue`: each side gives partial aggregates per
  `(campaign_id, date)`, the join compares days, and the scope root's frame
  buckets: March's cost is 15 and its revenue 150.
- `select date at Month, channel, cost`: `bookings` gives partial aggregates per
  `(campaign_id, date, channel)`, the join compares days, and March's cost is 10
  for `app` and 5 for `web`.
- `select date at Month, avg_cost, revenue`: `spend` gives rows, `bookings`
  gives partial aggregates per day, and March's `avg_cost` is 7.5 and its
  revenue 150.
- A mode-`unique` Unionset with the keyset `(campaign_id, date)` and the rows
  (A, 03-01, 10) and (A, 03-02, 10): the distinct keeps two rows, and March's
  cost is 20.

### Partial aggregates are an intermediate step

ADR 0074's pre-aggregation stays, in a Joinset's operands and in a Unionset's
branches. A partial aggregate is grouped by values before bucketing, and the
parent's aggregate combines it again. It never gives the request its value. The
research found partial aggregates below a join exact under three conditions, and
ADR 0074 states each:

1. the grouping holds the columns of the join condition and every column read
   above the join (*A Joinset*, **Pre-aggregation**);
2. each aggregate combines again (*Additivity decides whether a partial
   aggregate may be combined again*);
3. each partial group meets one row of the other operand, or the fanned
   operand's aggregates read one row per row identity (*A Joinset*,
   **Fan-out**, as ADR 0076 changed it); and, in both cases, where a join on
   the way repeats rows, every name of the grouping that is not in the
   request's grain, that the request buckets coarser, or that a Joinset above
   coalesces, is a name that the fanned row determines (ADR 0076, **Where the
   marker has no place**, rule 1).

Where the many side gives one row per value of the join condition's columns,
its partial aggregates remove the fan-out without a row identity (ADR 0074,
**Fan-out**). In
`campaigns ⋈ spend`, one-to-many on `campaign_id`,
`select campaign_name, budget, cost` reads one `spend` row per `campaign_id`,
so each campaign's budget counts once with no `row_number` marker (ADR 0076).

### The parent's demand fixes an operand's delivery

`CONTEXT.md` **Delivery** says: "The parent's demand fixes the delivery, through
the shared rules". ADR 0073 gives every Strategy call "the form the parent wants"
(*One input for every Strategy call*). ADR 0074's rules for a Unionset choose from the request instead. They now say
only how the branches give the delivery that the Unionset's demand fixes:

- **Rows:** every branch gives rows, and no aggregate sits above the union. In
  mode `unique` the distinct stays.
- **Partial aggregates:** in mode `all`, where every aggregate is additive or
  semi-additive and none is a formula push-down's `sum` (ADR 0074, *A
  Metric*), the branches give partial aggregates grouped by the grouping
  that the parent demands, and by each Dimension and each column under `auto`
  beyond it that a Metric's formula grain or part key reads (ADR 0074, *A
  Unionset*, **One form for every read**). The
  Unionset re-aggregates them in its own aggregate (ADR 0073, *The build
  order*, stage 4), so it gives one row per group. In mode
  `unique`, or where an aggregate is a formula push-down's `sum`, the branches
  give rows, and the aggregate above the distinct, or above the union, gives the
  partial aggregates.
- **Final aggregates:** ADR 0074's rules, as ADR 0079 changed them, the
  disjoint-branch rule included.

Take `campaigns` (A and B, both named `Summer`, budgets 100 and 200), joined
`Inner`, one-to-many, on `campaign_id` to the Unionset `spend_all`. Its branch
`web` has the costs (A, 10), (A, 20) and (B, 60), its branch `app` has (A, 40),
and `channel` is a literal in each branch.

- `select channel, campaign_name, avg_cost`: `Avg` is not additive, so the
  Joinset demands rows of `spend_all` (ADR 0074, *A Joinset*,
  **Pre-aggregation**). Every
  branch gives rows: `web` is 30 and `app` 40. The disjoint-branch rule, read
  alone, gives an `Avg` per branch and `campaign_id`, and the `Avg` above the
  join makes `web` 37.5.
- `select campaign_name, budget, cost`: `spend_all` gives partial aggregates
  per `campaign_id`, one row per campaign, so the budget is 300 and the cost 130.

This rule reads the demand, not the parent's source role, so it holds for each
composition that is a constituent of another.

### Every filter reads the value before bucketing

| Filter | What it narrows | Where it applies |
|---|---|---|
| A relationship filter | the joined rows | `post_join_filter`, or a `FilterNode` above a `CrossNode` (ADRs 0066 and 0075) |
| A request predicate on Dimensions, or on a row-level Metric with no `agg:` | every row | each copy that reads only one operand's names in that operand, and a copy that reads two operands above the join (ADR 0071, and *A predicate copy goes into its operand on every join type*) |
| An activated named filter | every row | as a request predicate on Dimensions (*A named filter is placed as a request predicate*) |
| A Measure's or Metric's own `filter:` | the rows of one aggregate | that aggregate's condition, computed in the row-level projection (ADRs 0071 and 0073); on a Joinset, over the rows that it combines (ADR 0071, as ADRs 0074 and 0077 changed it); on a Unionset, in each branch, whose own signature carries it (ADR 0065, D2, and ADR 0077), and over partial aggregates the Unionset's root only confirms the family (ADR 0074, *The composed tree*); on a Unionset whose branches give partial or final aggregates, each branch's aggregate for a Measure reads its branch's own signature filter joined with `And` onto the Unionset's own filter for that Measure (ADR 0065, D2) |
| A request predicate on a Measure or another Metric | the groups | above the node that finishes the value (ADR 0071) |

The first four sit below the bucket, so each reads the value before bucketing
(*Only a frame that gives final aggregates buckets*). With the relationship
filter `date >= '2026-03-15'`, both sides give partial aggregates per day,
`post_join_filter` keeps 03-20, and March's cost is 5 and its revenue 50.

**A Metric's formula reads a Dimension at its value before bucketing.** A
request that selects the Dimension at a granularity does not group by that
value in the grouping that finishes the Metric (ADR 0083, rule 4). The formula
push-down reads the value on each row, the part key of a Metric's parts reads
it on each row, and a Metric's formula grain groups step 1 by it beside the
bucketed value, under a reserved name (ADR 0074, *A Metric*). On ADR 0074's `shop`, `revenue_early_adj` has no
`agg:` and is `CASE WHEN ordered_at < DATE '2026-10-15' THEN revenue * 0.9 ELSE
revenue END`. Over the orders of 10-05 (100), 10-10 (200), 10-20 (50) and
10-25 (10), `select ordered_at at Month, revenue_early_adj` gives 330 for
October: 90 + 180 + 50 + 10. Reading the bucketed value 2026-10-01 on every
row would give 324. The same formula with `agg: sum` gives 330 too.

For a row-level Metric with no `agg:`, the node that finishes the value is the
row-level projection (ADR 0083, rule 5). So a request predicate on it narrows
every row, as a predicate on Dimensions does, and it sits below the bucket too
(ADR 0071, *Placement is read from the role*).

**A predicate on a Measure or a Metric reads the finished value.** A Measure's
semi-additive strategy runs inside its aggregate step (ADR 0076, *A marker gives
each aggregate its rows*), so a predicate directly above that step reads the
value after the strategy. With `balance` `latest` by `date`, 60 on day 1 and 40
on day 2, `where balance > 50` by month reads 40 and excludes the month. For a
semi-additive Metric, the predicate sits above the node that applies the
Metric's semi-additive strategy; that node is
not ruled here.

### A predicate copy goes into its operand on every join type

ADR 0074 writes a request predicate over a Dimension that both operands of a
Joinset map as two copies, one per operand. Each copy that reads only one
operand's names goes into that operand, below the join, whatever the join type.
So on `spend LEFT bookings`, `where date >= '2026-03-15'` goes into `spend` and
into `bookings`, and March's cost is 5 and its revenue 50. The human: "we cannot
assume what is really stored in data, we action on basis of what semantc
sources declare and their relation … what matter is that we have  filter which
by dimension mapping exposed for both side, which means that it must be applied
for both side".

Where that operand is a Unionset, the copy is a predicate on its rows, so it can
exclude a branch (ADR 0073, *A Unionset prunes a branch its predicate excludes*),
on the side that a `Left` join null-fills too.

ADR 0071's "It never goes below a join into a side the join null-fills"
(*Decision Outcome*) still holds for a predicate that reads no Dimension that both operands map: on
`orders LEFT customers`, `where country = 'SE'`, where only `customers` maps
`country`, stays above the join.

### A named filter is placed as a request predicate

An activated named filter on a composed semantic source is its body lowered on
the composition (ADR 0077). Where it reads a Dimension that both operands of a
Joinset map, compile stores two copies, one per operand; otherwise it stores one
copy. Each copy goes where a request predicate on Dimensions goes: into the
operand whose names it reads, or above the join where it reads two operands. On
a Unionset the filter goes into every branch. An operand gives each output field
that a copy above it reads, as it gives a predicate's leaves (ADR 0073). The
human, in round 1,
as recorded: "for each
choosen data kind filter pushed-down or located where field is accesible and
filter can be defined. … we need to put such filter for each side which is
involved into query".

### A copy applies only on an operand that the plan reads

The two copies of a filter on a Joinset over a Dimension that both operands map
(of a request predicate, of a named filter, or of a Measure's or Metric's own
`filter:`) apply only on an operand that the plan reads. A copy does not keep
its operand in the plan: ADR 0074's conditions for pruning an operand read the
request's names, not the copies. Where the Joinset prunes an operand, that
operand's copy is not applied, and the other copy applies in the operand that is
read. A filter that reads a name that only one operand maps reads that operand,
so ADR 0074's first condition for pruning (every demanded name is read from the
other operand) keeps it. The human: "query activates one side or both sides
filters - if query demands only bookings from this relationship - then only that
side activates filter".

The Joinset declares `late_revenue`, the sum of `bookings`' revenue with
`filter: date >= '2026-03-15'`. Compile stores it as
`spend.date >= '2026-03-15' AND bookings.date >= '2026-03-15'` (ADR 0077), where
`spend` and `bookings` stand for the operands' aliases:

- `select date at Month, late_revenue` reads only `bookings`, so the Joinset
  prunes `spend`. `bookings` is the operand that the Joinset keeps: its copy is
  its aggregate's condition, it gives final aggregates and buckets, and March's
  `late_revenue` is 50.
- `select date at Month, cost, late_revenue` reads both operands. The condition
  reads the rows that the Joinset combines, both sides give days, and the scope
  root's frame buckets: March's cost is 15 and its `late_revenue` 50.

### Amends

- **0071:** in *Decision Outcome* and in the table of *A predicate is applied
  where its semantics carry their meaning*, "It never goes below a join into a
  side the join null-fills" holds only for a predicate that reads no Dimension
  that both operands of a Joinset map: each copy of a request predicate over
  such a Dimension that reads only one operand's names goes into that operand on
  every join type; in the same table, an activated named filter on a composed
  semantic source is placed copy by copy as a request predicate on Dimensions,
  not as one boolean column applied at the lowest node where it is attached; the
  rest of 0071 stands (0080's *A predicate copy goes into its operand on every
  join type* and *A named filter is placed as a request predicate*).
- **0072:** in *The function*, "the subtree that groups the temporal item" is
  the frame whose own aggregate step gives final aggregates; a frame that gives
  rows or partial
  aggregates gives the temporal item at its value before bucketing; the rest of
  0072 stands (0080's *Only a frame that gives final aggregates buckets*).
- **0073:** in *One decision shape, checked per body*, the leaves of either of
  the two copies of a filter over a Dimension that both operands map are among
  the names that an operand serves only where the operand is read, and they do
  not make the Strategy read it; in *The build order*, stage 4
  buckets only in a frame whose own aggregate step gives final aggregates, and
  "never below a join
  into a null-filled side" holds only for a predicate that reads no Dimension
  that both operands map; in *A Unionset prunes a branch its predicate excludes*,
  a copy that goes into a side that the join null-fills is placed on that side's
  rows, so it may exclude a branch; the rest of 0073 stands (0080's *Only a
  frame that gives final aggregates buckets*, *A predicate copy goes into its
  operand on every join type* and *A copy applies only on an operand that the
  plan reads*).
- **0074:** in *Decision Outcome*, *A Unionset* and its `### Amends` entry for
  0033, the Unionset's rules say how its branches give the delivery that its
  demand fixes, as 0080's *The parent's demand fixes an operand's delivery*
  gives it for rows, partial aggregates and final aggregates, and disjoint
  branches give final aggregates with no aggregate above the union only where
  the demand takes final aggregates; in *A Joinset*, **Pre-aggregation**, an
  operand groups by each column at its value before bucketing; under **A
  condition is never coalesced.**, "On a row that one operand supplies alone,
  the other copy reads NULL, so the row fails the filter" holds for a filter
  that the Joinset applies above the join, and not for a copy that goes into its
  operand; in **Leaving out an operand**, neither of the two copies of a filter
  over a Dimension that both operands map keeps its operand in the plan, and a
  pruned operand's copy is not applied; the rest of 0074 stands (0080's *Only a
  frame that gives final aggregates buckets*, *The parent's demand fixes an
  operand's delivery*, *A predicate copy goes into its operand on every join
  type* and *A copy applies only on an operand that the plan reads*).
- **0075:** in *Ruled elsewhere, not here*, where the planner buckets a
  Dimension that the request asks for at a granularity, and where each filter
  that reads that Dimension is applied, is 0080's; the rest of 0075 stands.
- **0077:** in *Ruled elsewhere, not here*, where the planner applies an
  activated named filter, and a request predicate on a semi-additive Measure or
  Metric, is 0080's *A named filter is placed as a request predicate* and
  *Every filter reads the value before bucketing*; the rest of 0077 stands.

### Consequences

- Good, because one rule places every bucket: every join of rows or partial
  aggregates, every filter, every distinct and every `by` reads the value before
  bucketing, so no list of node kinds has to grow with each new combine, and ADR
  0072's "a bucket is only grouped by and output" (*What a bucketed timestamp
  means*) holds on every source role.
- Good, because a join of rows or partial aggregates never compares a bucket, so
  the declared cardinality describes every such join, and
  an earlier case does not
  occur.
- Good, because an operand gives the delivery that its parent demands, so a
  Unionset never gives final aggregates to a parent that aggregates again.
- Bad, because partial aggregates below a combine are grouped by the value
  before bucketing, so more rows reach the union or the join: one per day, not
  one per month. The physical read and the first aggregate do not change.
- Compile needs no change: ADR 0077 already has compile write the copies.
  `crates/` is not changed by this decision.

## More Information

Decided 2026-10-05, and on
2026-10-18 for the
value that a Metric's formula reads, and the bucket of a Metric's two aggregate
steps. The research that it reads is on aggregation across joins (Yan & Larson, VLDB 1995; MetricFlow; Cube). The decisions of 2026-10-18 read the research on
MetricFlow, Cube, Looker, Malloy and
Calcite, and an oracle on DuckDB and DataFusion 55.
