---
number: 83
title: A Metric declares no additivity
status: accepted
date: 2026-10-06
source:
- CONTEXT.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md
- docs/adr/0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md
- docs/adr/0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md
- '`Metric`, `MetricWire` and the `authoring_body!` key list in `crates/semstrait-model/src/model/types/entities/metric.rs`, and `reject_unknown_keys` in `crates/semstrait-model/src/model/types/entities/semantic_body.rs`'
- '`SemanticRoleDefinition::Metric` in `crates/semstrait-manifest/src/manifest/semantic_definition.rs`'
- the Metric `profit_margin` in `crates/semstrait-model/schemas/reference.yaml`
- a prototype run through Substrait into DataFusion 55, against a SQL oracle
- the human's ruling of 2026-10-06
- for where a Metric's aggregate reads its Measure's additivity, the human's rulings of 2026-10-17
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
links:
- target: 30
  kind: amends
- target: 74
  kind: amends
- target: 76
  kind: amends
- target: 77
  kind: amends
- target: 80
  kind: amends
- target: 65
  kind: amends
- target: 84
  kind: amendedby
---

# A Metric declares no additivity

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Additivity belongs to an aggregate*, rule 5, "and the predicate sits directly above them, in stage 5a" does not hold for a Metric that a Grainset computes: a request predicate on that Metric sits above the `Full` joins of the Grainset's frame, which null-fill every child that lacks a group (0084's rule 7); the rest of 0083 stands (0084's *How they are joined*).

## Context and Problem Statement

ADR 0030 gives a Metric an authored additivity: "`Additivity` is authored inline
and defaults to `Additive` for Measures and Metrics" (*Decision Outcome*). ADR
0077 rules how it applies: "A Metric's declared additivity is a fact about the
Metric's own value. It applies after the formula, and it narrows no node"
(*Decision Outcome*). For a semi-additive Metric, "the Metric's semi-additive
strategy runs across the Metric's values" (*A Metric's declared additivity
applies to its own value, after its formula*). ADR 0077 left the plan shape for
that step to a later decision, because
ADR 0076's marker cannot write it. ADR 0077 says why: "That marker is read by an
aggregate's condition, below the formula" (*Ruled elsewhere, not here*).

A prototype ran each plan shape through Substrait into DataFusion 55,
against a SQL oracle. A step above the formula is either inexact for the other
aggregates of the request, or reads the physical source twice, or needs window
aggregates that DataFusion's Substrait consumer reads wrongly. The Metric's
marker on each of its nodes keeps ADR 0076's one aggregate step. Where the
Metric's declared additivity holds, it gives ADR 0077's value. But then a
Metric's value is not its formula over the values of its Measures in the same
row.

Take the Dataset `ledger`. A later load restates the balance of a day:

| account | date | loaded_at | balance | fees |
|---|---|---|---|---|
| X | 10-30 | 10-30 08:00 | 100 | 1 |
| X | 10-31 | 10-31 08:00 | 110 | 2 |
| X | 10-31 | 10-31 20:00 | 115 | 0 |
| Y | 10-20 | 10-20 09:00 | 190 | 4 |
| Y | 10-31 | 10-31 09:00 | 200 | 3 |

`balance` is a `sum`, `latest` by `loaded_at` inside `[account]`. `fees` is a
`sum`. `select month, balance, fees, net` for October gives `balance` 315 and
`fees` 10. With `net = balance - fees` declared `latest` by `date` inside
`[account]`, `net` is 310 under ADR 0077, beside 315 − 10 = 305 in the same row.

## Considered Options

- **A Metric declares no additivity** (chosen).
- **The Metric's semi-additive strategy as a marker on each of its nodes.** Each
  `Aggregate` node reads `rank() over (partition by <grouping>, <Metric axes>
  order by <Metric by>) = 1`, in ADR 0076's one aggregate step. It gives ADR 0077's value only where the Metric's declared
  additivity holds: for `m = sum(a) / sum(b)` declared `latest` by `date` inside
  `[account]`, October is 0.667 where ADR 0077 gives 1.25. And on `ledger`, `net`
  is 310 beside 315 − 10 in the same row.
- **A second aggregate step above the formula**: the nodes grouped by the
  request's grouping, the Metric's axes and `by`; the formula; a marker over the
  Metric's values; `Sum` across the axes. Every other aggregate of the request
  must pass through the first step. In the prototype, October's `avg(cost)` is
  3.125 where it is 3.0, and `count(distinct usr)` is 5 where it is 3. ADR 0076
  refused this shape for the same reason: "a non-additive aggregate beside it
  cannot pass through the step per date" (*Considered Options*).
- **The Metric in its own subtree, joined back on the grouping.** It is exact,
  but the plan reads the physical source twice: DataFusion 55's consumer refuses a
  `ReferenceRel` ("Substrait plan with more than 1 relation trees not
  supported"). The join needs `IsNotDistinctFrom`. ADR 0074 builds this shape,
  with two independent reads, for a Metric's own second aggregate step, where a
  second read is the cost of a value that no single step gives.
- **Window aggregates below the one aggregate step** (`sum(a) over (partition
  by …)`). The expression vocabulary gains aggregate window functions, and it
  grows only by ADR (ADR 0066). DataFusion 55's Substrait consumer reads a
  windowed `count(distinct …)` as a plain count, with no error.

## Decision Outcome

In short:

- Additivity belongs to an aggregate of rows. A Metric declares no additivity,
  with or without `agg:`. The `agg:` of a row-level Metric aggregates rows as a
  Measure's does, and reads its function's additivity only. The `agg:` of a
  Metric whose formula reads a Measure or a Metric aggregates the formula's
  values in a second aggregate step, and no additivity judges it. A row-level
  Metric with no `agg:` is planned as a Dimension.
- No declared additivity and no semi-additive strategy combines the values of a
  Metric. For a Metric with no `agg:` and a formula over Measures and Metrics
  only, down every Metric that it inlines, the planner combines the nodes, and
  then the formula runs over the final aggregates.
- So that Metric, with no `filter:`, gives its formula over their values in the
  same row.
- A formula that also reads a Dimension or a column outside an aggregate runs
  over the final aggregates where it reads no column under `auto` there and the
  demand that finishes the Metric selects each such Dimension with no
  granularity, runs as one `sum` through the formula push-down where its pattern
  holds, and otherwise runs on its parts, in two aggregate steps (rule 4).

### Additivity belongs to an aggregate

1. Additivity belongs to an aggregate of rows: a Measure, an `Aggregate` node in
   a Metric's tree, or the `agg:` of a row-level Metric. A Metric declares no
   additivity, with or without `agg:`. A Metric is row-level where its formula
   reads no Measure and no Metric, down each link, a link to a row-level Metric
   with no `agg:` read through that Metric's formula. Compile lowers a row-level
   Metric's `agg:` as a Measure's (ADR 0074, *A Metric*), and its aggregate
   reads its function's additivity only (ADR 0091, rule 3): no Measure narrows
   it. Where the formula reads a Measure or a Metric, the `agg:` aggregates the
   formula's values at the formula grain, in a second aggregate step that no
   additivity judges (ADR 0074, *A Metric*).
2. A Metric does not declare additivity. A semantic model whose Metric declares
   it fails to parse with the ordinary unknown-key refusal, as for any key that a
   body does not accept (`ParseErrorKind::UnknownField`).
3. Each `Aggregate` node of a Metric reads the effective additivity of its
   Measure, with that Measure's `axes`, `by` and semi-additive strategy, from two
   inputs that stay apart. The function's additivity, from the aggregate
   function's declaration (ADR 0091, rule 3), says whether a partial aggregate of
   it may be combined again. The Measure's declared additivity, from the
   definition of the Measure whose entry the Metric's link names (ADR 0088, rule
   5), narrows the function's: `non` combines no partial aggregate again, and a
   semi-additive strategy keeps the combine except along its `by` and keeps the
   rows that the function then combines (ADR 0076). The planner reads the Measure's definition as it follows the link,
   and nothing is written on the node. So a node of a semi-additive Measure reads
   its Measure's marker as ADR 0076 says (*A marker gives each aggregate its
   rows*), and no marker of the Metric. The human: "addivity is measure
   level signature attribute"; later: the Measure's additivity "declares
   strategy of how each tuple must handled on read", and the function's is a
   "hint on pre-aggregation capability".
4. No declared additivity and no semi-additive strategy combines the values of a
   Metric. Take a Metric with no `agg:` whose formula reads only Measures and
   Metrics, where each Metric that it inlines, at any depth, meets the same two
   conditions. For it, the planner combines the nodes. Then the formula runs
   over the final aggregates, in stage 5 (ADR 0073, *The build order*).

   A formula that reads a Measure or a Metric may also read, down every Metric
   that it inlines, a Dimension or a column under `auto`, outside an
   aggregate. Such a name extends the demand of the frame, and it adds no
   grouping to the output: where it groups, it groups step 1 of the Metric's own
   subtree (ADR 0074, *A Metric*). For a Metric with no `agg:`, the planner tries these cases in
   order (ADR 0074, *A Metric*):

   - **No such name.** The formula reads no Dimension and no column outside an
     aggregate. It runs as one projection over the final aggregates.
   - **Grouped.** The demand that finishes the Metric selects each Dimension
     that the formula reads outside an aggregate, with no granularity. The
     formula runs as one projection over the final aggregates, and any formula
     passes. A Dimension that the demand selects at a granularity, Day
     included, is not grouped: the formula reads its value before bucketing
     (ADR 0080, *Every filter reads the value before bucketing*). A column
     under `auto` is never grouped, because a request selects no column. A
     request predicate `D = literal` does not group by `D`.
   - **The formula push-down.** The formula is in the formula push-down's
     pattern. The planner writes it into one `sum` in the frame's one
     aggregate step.
   - **The parts.** Otherwise the planner computes the formula on its parts, in
     a subtree of two aggregate steps, and adds the parts' values. Where the
     formula's type is not numeric, the parts cannot be added, and the planner
     refuses the request with `PlanError::UngroupedMetricFormula { metric, name
     }`. `name` is the first name, in the stored tree's order, that the formula
     reads outside an aggregate and the demand does not select with no
     granularity. Compile accepts the Metric, because the outcome depends on
     the request.

   A row-level Metric with no `agg:` (rule 1), a constant included, is planned as
   a Dimension whose `expr:` is its formula (ADR 0074, *A Metric*).
5. A request predicate on a Metric sits above the node that finishes the
   Metric's value. This is true for every Metric. It is ADR 0080's row "A
   request predicate on a Measure or a Metric", "above the node that finishes
   the value (ADR 0071)" (*Every filter reads the value before bucketing*). For
   a Metric of rule 4, that node is the formula's projection or the formula
   push-down's `sum` (ADR 0074, *A Metric*), so the predicate is a
   post-aggregate predicate, in stage 6. For a Metric that takes its parts, and
   for a Metric whose `agg:` aggregates a formula that reads a Measure or a
   Metric, that node is its step 2 and its cast, and the predicate sits directly
   above them, in stage 5a (ADR 0073, *The build order*). For a row-level Metric with no `agg:`, that node is
   the row-level projection, so the predicate is a pre-aggregate predicate, in
   stage 3, as on a Dimension (ADR 0071, *Placement is read from the role*).
6. A Metric may need a value that its Measure does not give. Then the author
   declares a second Measure with that additivity.
7. Thus a Metric of rule 4 with no `filter:`, whose formula reads no column
   under `auto` outside an aggregate, in a request that selects each Dimension
   that its formula reads outside an aggregate, with no granularity, gives its formula over the values of its Measures and Metrics in the same
   row. A Metric's own `filter:` narrows the rows of each node (ADR 0071,
   *Aggregation-level conditions*; ADR 0074, *A Metric*).
8. Compile accepts a Metric whose formula reads a Dimension or a column that
   `auto` maps, with or without `agg:`. It also accepts a row-level Metric with
   no `agg:`.
   Under an explicit `semantic_mapping`, `ColumnInSemanticExprUnderManualMapping`
   still refuses a column. Compile refuses a Metric whose `expr:` or `filter:`
   reads a named Filter, with `CompileError::NamedFilterReference { semantics,
   filter }` (ADR 0071, *Placement is read from the role*).

On `ledger`, October by month:

- `balance` gives 315 and `fees` gives 10, so `net = balance - fees` gives 305.
- The value at the last date is another Metric. `fees_closing` is a `sum`,
  `latest` by `date` inside `[account]`, and gives 5. `closing_net = balance -
  fees_closing` gives 310.
- By day, `net` gives 310 on 10-31, 99 on 10-30 and 186 on 10-20.

Take the Dataset `orders`, with the Measures `revenue = sum(amount)`, which
declares `data_type: double`, `order_count = count(order_id)` and `buyers =
count(DISTINCT customer_id)`:

| order_id | customer_id | region | channel | amount |
|---|---|---|---|---|
| o1 | c1 | eu | web | 100 |
| o2 | c1 | eu | app | 50 |
| o3 | c1 | us | web | 200 |
| o4 | c2 | us | NULL | 10 |

Each Metric below has no `agg:` (rule 4), but `regions_with_revenue`:

| Metric | Request | Plan | Value |
|---|---|---|---|
| `revenue_adj = CASE WHEN region = 'eu' THEN revenue * 1.2 ELSE revenue END` | by `region` | grouped | eu 180, us 210 |
| `revenue_adj` | by `channel` | the formula push-down | web 320, app 60, NULL 10 |
| `avg_order_value_adj`: the same `CASE` over `revenue / order_count` | in total | the parts: `revenue / order_count` is not a term | eu part 150 / 2 × 1.2 = 90, `else` part 210 / 2 = 105: 195 |
| `revenue_eu_bonus = CASE WHEN region = 'eu' THEN revenue + 100 ELSE revenue END` | in total | the parts: `revenue + 100` is not a term | 460, where one `sum` would give 560 |
| `revenue_eu_bonus` | by `region` | grouped: any formula passes | eu 250, us 210 |
| `buyers_eu_weighted = CASE WHEN region = 'eu' THEN buyers * 2 ELSE buyers END` | in total | the parts: a count distinct is not `sum` or `count` | eu part 1 × 2, `else` part 2: 4 |
| `region_label = CASE WHEN region = 'eu' THEN 'EU' ELSE 'other' END` | by `channel` | as a Dimension | (web, EU), (web, other), (app, EU), (NULL, other) |
| `regions_with_revenue`, `agg: count` over the formula of `revenue_adj` | by `channel` | step 2 over the formula grain (channel, region) | web 2, app 1, NULL 1 |

### A Measure over rows that are already aggregates

9. The Dataset states it. The temporal shape `Timeseries` or `Snapshot` says
   that the rows are already aggregates, and the keyset says which Dimensions
   each row holds (ADR 0074, *Every request is planned against its grain
   first*).
10. The Measure's `agg` is the function that combines the stored values. It is
    not the function that made them.

| Stored column | Measure | Request coarser than the keyset |
|---|---|---|
| a sum (`clicks`) | `agg: sum` | exact |
| a row count (`orders_n`) | `agg: sum` over `orders_n`; never `count(*)`, which counts the stored groups | exact |
| a min or a max | `agg: min` or `agg: max` | exact |
| an average | store the sum and the count, and declare the Metric sum / count; else `agg: avg`, which gives the average of the stored averages | exact; else `LossyReaggregation` |
| a distinct count (`users`) | `agg: sum`, `additivity: non` | `LossyReaggregation`: the sum counts a user once for each row |
| a balance (Snapshot) | `agg: sum`, `additivity: { semi: { axes, strategy: latest } }`; `by` defaults to `snapshotted_at` | the semi-additive strategy applies where the request drops `by` or buckets it coarser (ADR 0076) |

11. Sometimes a column of a physical source already holds a ratio, for example
    `ctr` in a Timeseries Dataset keyed by `(campaign, date)`.
    - Where the physical source has the parts, the semantic model has the parts: `clicks` and
      `impressions` are Measures with `agg: sum`, and `ctr = clicks /
      impressions` is a Metric. The value is exact at every grain.
    - Where it has only the ratio, the semantic model declares a Measure
      `ctr_reported` with `agg: avg`. `Avg` is a non-additive family (ADR 0074,
      *Additivity decides whether a partial aggregate may be combined again*).
      At the keyset's grain each group is one row, so the value is exact. A
      coarser request gets `Advisory::LossyReaggregation`.

### Amends

- **0030:** in its *Decision Outcome*, "`Additivity` is authored inline and
  defaults to `Additive` for Measures and Metrics" holds for Measures only: a
  Metric declares no additivity, and each `Aggregate` node of a Metric reads the
  effective additivity of its Measure through its link, so ADR 0074's "once
  compile carries it" no longer holds (0083's *Additivity belongs to an
  aggregate*).
- **0065:** in *A definition is the contract; a signature is the
  implementation*, "a metric's optional aggregation and optional additivity"
  reads "a metric's optional aggregation"; in *A role disagreement stays
  `SemanticKindMismatch`, the other four axes fold to `SemanticShapeMismatch`
  (D1)*, additivity is an axis for a Measure only; in *A composed Metric
  carries no filter (D3)*, *Consequences* and *Open questions*, a Metric's
  "own `agg:` lowering … is out of scope" reads: a row-level formula's `agg:`
  is built as a Measure's, and a post-aggregate formula's `agg:` is the
  planner's second aggregate step over the formula's values (0083's rules 1
  and 8); in *What a tree contains, and who builds it*,
  as ADR 0088 changed it, a Metric whose formula reads no Measure and no Metric
  and that declares `agg:` has a measure's tree, `Aggregate(<its agg:>, <its
  row-level tree>)` (0083's rule 1); the rest of 0065 stands (0083's
  *Additivity belongs to an aggregate*).
- **0074:** in *Additivity decides whether a partial aggregate may be combined
  again*, "ADR 0030 gives a Metric an authored additivity, as it gives a
  Measure" and "Until then, an author narrows a Metric by declaring the Metric's
  own additivity" no longer hold, in the clause as ADR 0077 changed it: a Metric declares no additivity, and each
  `Aggregate` node of a Metric reads its Measure's effective additivity through
  its link (0083's rule 3), so "until compile writes that Measure's additivity
  onto the inlined node" no longer holds either; in *A semi-additive
  aggregate's strategy*, as ADRs 0076 and 0077 changed it, "a source that
  exposes a semi-additive Measure or Metric" and "a source that serves the
  Measure or Metric serves its axes" apply to a Measure only, and a Metric has
  no semi-additive strategy;
  the rest of 0074 stands (0083's *Additivity belongs to an aggregate*).
- **0076:** in *`axes` and `by`*, "A semi-additive Measure or Metric declares"
  applies to a Measure only; in *Where `by` and the axes come from*, "The
  Datasets that a semi-additive Metric reads are those that its aggregate nodes
  read" has no subject, and "Compile refuses a source that exposes a
  semi-additive Measure or Metric" applies to a Measure only; the rest of 0076
  stands (0083's *Additivity belongs to an aggregate*).
- **0077:** *A Metric's declared additivity applies to its own value, after its
  formula*, and the first bullet of *Decision Outcome*'s "In short", no longer
  hold; only "Each node reads its function's additivity, narrowed by the
  effective additivity of the Measure it came from" stays; in *Ruled elsewhere,
  not here*, the sentence that begins "That marker is read by an aggregate's
  condition, below the formula" and the sentence after it have no subject, and a
  node's marker partitions by the `<grouping>` that ADR 0076 gives it, which no
  Metric's additivity extends; in *Consequences*, `SemiAdditiveMetricInlined` is
  not a compile refusal; in *Amends*, the entries for 0030 and 0074 no longer
  hold where they say that "a Metric's declared additivity applies to the
  Metric's own value, after its formula" and that "a semi-additive Metric's
  semi-additive strategy runs on the Metric's value"; in *Ruled elsewhere, not
  here*, "the effective additivity on each `Aggregate` node" is not a manifest
  shape: each node reads it through its link (0083's rule 3); the rest of 0077
  stands (0083's *Additivity belongs to an aggregate*).
- **0080:** in *Every filter reads the value before bucketing*, "For a
  semi-additive Metric, the predicate sits above the node that applies the
  Metric's semi-additive strategy; that node is not ruled here" has no subject: the row
  "above the node that finishes the value" covers every Metric; in *Amends*,
  "a request predicate on a semi-additive Measure or Metric" applies to a
  Measure only; the rest of 0080
  stands (0083's *Additivity belongs to an aggregate*).

### Consequences

- Good, because a Metric of rule 4 with no `filter:` gives its formula over the
  values of its Measures and Metrics in the same row, where its formula reads
  no column under `auto` outside an aggregate, in every request that selects
  each Dimension that its formula reads outside an aggregate, with no
  granularity (*Additivity belongs to an aggregate*, rule 7).
- Good, because a Metric's additivity adds nothing to the plan: no marker for a
  Metric, no step above the formula, and no function that combines a Metric's
  values.
- Bad, because a model that declares a Metric's additivity stops parsing. The
  reference model's `profit_margin` declares `additivity: full`.
- Bad, because a Metric's value at the last date needs a second Measure, such as
  `fees_closing`.
- The semantic model, compile and the manifest change:
  - `additivity` leaves `Metric`, `MetricWire`, the builder and its setters, and
    the Metric's `authoring_body!` key list, so `reject_unknown_keys` refuses the
    key. The reference model's `profit_margin` drops it.
  - `additivity` leaves `SemanticRoleDefinition::Metric`, and compile no longer
    copies it or compares it across sites.
  - `CompileError::SemiAdditiveMetricInlined` (ADR 0077) is not built. The checks
    of a semi-additive axis and of `by` (ADR
    0076) apply to a Measure only.
- Good, because each Metric that compile accepts has a plan: a row-level
  Metric with no `agg:` as a Dimension, a Metric's `agg:` over a formula in a
  second aggregate step, and a formula that reads a Dimension that the request
  does not group by on its parts. The planner refuses only such a formula whose
  type is not numeric (rule 4), and compile refuses on a Grainset only a Metric
  that no child serves (ADR 0084, rule 8).
- Bad, because a model whose semantics reads a named Filter stops compiling,
  with `CompileError::NamedFilterReference { semantics, filter }` (rule 8). No
  corpus model has one; the test `a_filter_that_reads_a_named_filter_links_it`
  becomes a refusal test.
- Neutral: compile lowers a row-level Metric's `agg:` as a Measure's (rule 1).
  ADR 0074's *Consequences* lists the changes to `crates/`.

## More Information

Decided 2026-10-06, and on 2026-10-17 for where
a Metric's aggregate reads its Measure's additivity, and on
2026-10-17 and
2026-10-18 for a Metric whose formula reads a Dimension or a column, and a
Metric's own `agg:`. The human: "a Metric declares no additivity", and "With
this ruling, a Metric has no semi-additive strategy." The decisions of
2026-10-18 read the
research on MetricFlow, Cube, Looker,
Malloy and Calcite, and an oracle on DuckDB and DataFusion 55.

The prototype runs each
plan shape in *Considered Options* through the `substrait` crate 0.63 into
DataFusion 55, against a SQL oracle.
