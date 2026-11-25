---
number: 76
title: A window marker gives each aggregate its rows, and latest and earliest order by one Dimension inside their axes
status: accepted
date: 2026-10-04
source:
- CONTEXT.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md
- '`SemiAdditivity` and `SemiAdditivityStrategy` in `crates/semstrait-model/src/model/types/entities/measure.rs`'
- '`WindowCall` in `crates/semstrait-sql/src/expr/call/mod.rs`, and `SortDir` in `crates/semstrait-sql/src/expr/sort.rs`'
- '`TemporalShapeKind` in `crates/semstrait-model/src/model/types/entities/temporal.rs`'
- a prototype run on DataFusion 55
- the human's direction recorded in the body, and the human's rulings
- the decisions of 2026-10-17 and 2026-10-18
- the decisions of 2026-10-19
links:
- target: 74
  kind: amends
- target: 71
  kind: amends
- target: 30
  kind: amends
- target: 75
  kind: amends
- target: 77
  kind: amendedby
- target: 83
  kind: amendedby
- target: 85
  kind: amendedby
- target: 90
  kind: amendedby
---

# A window marker gives each aggregate its rows, and latest and earliest order by one Dimension inside their axes

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *`axes` and `by`*, "ADR 0074 reads `first` and `last` as `earliest` and `latest`" no longer holds: they leave `SemiAdditivityStrategy`, and their removal, which 0076 leaves to a later decision, is 0077's *What retires*.
> - [ADR 0083](0083-a-metric-declares-no-additivity.md): in *`axes` and `by`*, "A semi-additive Measure or Metric declares" applies to a Measure only; in *Where `by` and the axes come from*, "The Datasets that a semi-additive Metric reads are those that its aggregate nodes read" has no subject, and "Compile refuses a source that exposes a semi-additive Measure or Metric" applies to a Measure only; the rest of 0076 stands (0083's *Additivity belongs to an aggregate*).
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): in *A window's order key carries its direction*, `SortDir` and `NullOrdering` move to `semstrait-sql`, not to `semstrait-model`, because `Expr` is now `semstrait-sql`'s. The rest of 0076 stands.
> - [ADR 0090](0090-semstrait-ir-keeps-the-plan-and-its-parameter-and-the-manifest-owns-the-expression-id-and-the-expression-layer.md): in *A window's order key carries its direction*, "`semstrait-ir` re-exports them" becomes: `semstrait-ir` imports `SortDir` and `NullOrdering` from `semstrait-sql` (ADR 0090, rule 6). The rest of 0076 stands.

## Context and Problem Statement

ADR 0074 refuses two kinds of request in the MVP. `PlanError::MixedRowSets`
refuses aggregates that need different rows: `balance` with `latest` on `date`
reads each month's last day, beside `cost`, which reads every day. Also the
`Avg` of a fanned operand's column reads each of its rows once, beside the `Avg`
of the many side's column, which reads every row.
`PlanError::StrategyOrderUndecided` refuses a semi-additive strategy with no
defined order: `latest` across two axes, or across `axes: [region]`.

The human directed window functions ordered by the time axis, and a definition
`axes: … by …`, in which `axes` partitions and `by` orders.

## Considered Options

- The marker form in every request, not only in a request that mixes sets of
  rows; `axes` and `by` on the semi-additive definition; and the time axis as
  the default `by` (chosen).
- **`first_value` over the source's rows.** It keeps one row: with
  `(A, 09-30, 110)` and `(B, 09-30, 210)`, September's `balance` is 110 or 210,
  not 320.
- **`first_value` over ADR 0074's inner aggregate per (month, date).** It keeps
  two aggregate steps, and a non-additive aggregate beside it cannot pass
  through the step per date.
- **A marker only where a request's aggregates need different rows.** The plan
  of one aggregate would change with the other aggregates of its request.
- **`axes` as the Dimensions along which the aggregate does not add, and `by` as
  its order only.** The rows that `axes: [date, warehouse] by date` keeps still
  span warehouses that the aggregate does not add.
- **`by` required.** The time axis already gives the order.
- **The request's temporal Dimension as the order, and then no order.** One
  Measure would mean other things in other requests, and no order picks an
  arbitrary row.
- **The time axis of the source that answers.** Over `orders` (Events) joined
  with `balances` (Snapshot), `balance` would be ordered by the order date.
- **A sort type for windows only.** Substrait carries one `SortField` for both.
- **Keeping `average`.** It needs more aggregate steps, and a refusal beside a
  non-additive aggregate. Nobody defined it: it came with the first
  `SemiAdditivityStrategy`, and ADR 0074 gave it its only meaning. An average of
  daily totals is the Metric `sum(balance) / count_distinct(date)`.

## Decision Outcome

In short:

- One aggregate step reads every set of rows that a request needs, but in a
  Metric's subtree of two aggregate steps (ADR 0074, *A Metric*). A **marker**,
  a window function in a projection below the step, gives each aggregate its
  rows.
- `latest` and `earliest` take their rows by one Dimension, `by`, inside the
  aggregate's `axes`. `by` is the time axis unless the author writes one.
- A request is refused for its sets of rows only where a `row_number` marker
  has no place that keeps one row per row identity in each group of the
  request (**Where the marker has no place**), and never for its order.

### A marker gives each aggregate its rows

An aggregate reads the rows as the operands give them, the rows deduplicated by
a row identity (ADR 0074, *A Joinset*), or the rows that `latest` or `earliest`
keeps. For the last two, a projection below the subtree's aggregate step adds a
marker:

| Rows | Marker | Over |
|---|---|---|
| one per row identity | `row_number() over (partition by <grouping>, <row identity>, <arguments>, <condition>)` | the joined rows |
| `latest` / `earliest` | `rank() over (partition by <grouping>, <axes> order by <by> desc / asc, NULLs last)` | the rows or partial aggregates that the semi-additive strategy reads |

A `rank` marker is added only where the semi-additive strategy applies (below).
The aggregate reads its rows through its aggregation-level condition: its own
condition (ADR 0071, *Aggregation-level conditions*, as ADR 0074 changed it)
joined with `And` onto `<marker> = 1`. An aggregate that needs both markers
joins both.

- For a Measure lowered on a Joinset, the `row_number` partition also holds
  the position of the result of its argument that the row selects, and only a
  result that ADR 0074 deduplicates reads the marker (ADR 0074, *A Joinset*,
  **Fan-out**).
- In a Metric's subtree of two aggregate steps, `<grouping>` is the grouping of
  step 1 (ADR 0074, *A Metric*).
- The `row_number` partition is the grouping of ADR 0074's distinct-rows
  aggregate, so ADR 0074's row identity stands, and `Min`, `Max` and count
  distinct read no `row_number` marker, as they skip the deduplication in ADR
  0074.
- The `rank` reads every row of its partition, whatever the aggregate's own
  condition says. With NULLs last, `rank = 1` keeps a group's rows at its latest
  (for `earliest`, its earliest) non-NULL `by`, or all its rows where `by` is
  NULL in every row.
- A semi-additive partial aggregate is grouped by its `axes` and its `by`, at
  the values that the source holds `by` at. Its marker and its condition apply
  above the combine.
- The marker projection is part of the aggregate steps that ADR 0074, as
  amended here, gives in ADR 0073's stage 4. It sits after the bucketing,
  because it partitions by the bucketed values, and before the aggregate step.
- Where the aggregate reads a Dataset, a marker reads each of `by` and the
  axes from the Dataset of the aggregate that has it, never from the `Coalesce`
  of a name that both operands of a Joinset map. Where two or more of the
  aggregate's Datasets map it, it reads the one whose time axis it is. Where the
  aggregate reads no Dataset, it reads the value that the source serves.
- A marker, and a Dimension that a marker or a semi-additive partial aggregate
  reads at values other than the bucketed value that the request selects,
  beside that value, are output fields that the planner creates, so they carry
  reserved names (ADR 0075).

So a marker adds no aggregate step. A subtree has one, and a Metric's subtree
of two aggregate steps has two (ADR 0074, *A Metric*). With the rows
`(A, 10-30, 120)`,
`(A, 10-31, 130)` and `(B, 10-30, 230)`, `select month, balance, cost` reads
`sum(balance) filter (rank = 1)` and `sum(cost)`, and October's `balance` is
130.

**Where the marker has no place.** A `row_number` marker sits before the
aggregate step that first reads the rows of the join that repeats them (ADR
0074, *A Joinset*, **Fan-out**), and it keeps one row per row identity in each
group of that step. The planner refuses a request with
`PlanError::FanOutMarkerNotPlaceable { semantics, path }`, where `path` is the
frame path of that join, when an aggregate reads such a marker and one of
these holds:

1. The frame of that step gives its parent partial aggregates, grouped by a
   name that the fanned row does not determine, and that the request's grain
   does not hold, or holds only at a coarser granularity, or that a Joinset
   above the step coalesces, because both of its operands map it (ADR 0074, *A
   Joinset*): the `Coalesce` can give two groups of the step one value. The
   fanned row
   determines a name that only the fanned operand maps, a name of the join's
   condition, and a name of an operand of which one fanned row meets at most
   one row, by the declared cardinality of each join between them. Otherwise
   one fanned row can fall in two groups of that step, and the parent combines
   them into one group of the request.
2. The rows reach that step through a Unionset, at any depth, or the step is a
   Unionset's, above its union. There the marker also partitions the rows of
   the other branches, which NULL-fill the row identity. Rule 2 also holds
   where the walk of ADR 0077 takes a value leaf of a Measure lowered on a
   Joinset into the branches of a Unionset that a join on the way repeats, and
   reaches Datasets in two or more of those branches. Each branch NULL-fills
   the row identities of the others, and NULL groups with NULL (ADR 0074, *A
   Joinset*, **The row identity**). So a row of one branch whose identity is
   NULL in every column has the same identity as such a row of another
   branch, and the marker merges the two rows. Where the identities share a
   Dimension, two rows with the same value of it also merge. With branches
   whose identities are `a_id` and `b_id`, each with one row of amount 10 and
   a NULL identity, joined many-to-many to two `gift` lines that each meet
   both rows, `select line_id, gift_amount` (ADR 0074's `sum(CASE WHEN kind =
   'gift' THEN amount END)`) gives 10 for each line, where the answer is 20.
3. One result of the aggregate would read two or more `row_number` markers.
   For a Measure lowered on a Joinset, the walk of ADR 0077 (**A value whose
   rows a Joinset repeats**) gives one marker for each Dataset that the value
   leaves of one result reach through a join that repeats it, however many
   joins on the way repeat that Dataset. For any other aggregate, ADR 0074
   gives one marker for each join on the way whose fanned operand serves it
   (*A Joinset*, **Fan-out**). A `row_number` marker is joined with a `rank`
   marker only, never with another `row_number` marker.

Every other aggregate reads its marker as this section says. So the marker
stays where the join's own aggregate step gives the final aggregates, so that
no aggregate step above combines them again: for example in the scope root's
frame, and, where their demand takes final aggregates, in the operand that a
Joinset keeps, in each disjoint branch, and in a Grainset's child that answers
alone, or that a drill-across reads: a Grainset drills across only where its
demand takes final aggregates (ADR 0084, rule 1). It also stays where that
step gives partial aggregates that rule 1 allows, and where the join gives its rows to Joinsets above, through a
Grainset or not. `Min`, `Max` and count distinct read no `row_number` marker,
so these rules do not apply to them.

With `all_sales` {`orders INNER order_lines` (`one_to_many`, it repeats
`orders`; o1 100 with two lines, o2 50 with one), `store_sales` (s1 30, s2
30)}, `select kind, avg_amount` is refused by rule 2: the marker above the
union, where `store_sales` NULL-fills `order_id`, gives `gift` 60.0, and each
order and each sale counted once gives 52.5. On
`(orders ⋈ order_lines) ⋈ ads`, with o1 100 on lines to `ad1` and `ad2`, o2 40
on `ad2`, and both ads `search`, `select ad_type, total` is refused by rule 1:
the inner Joinset's partial aggregates by `ad_id` count o1 once per ad, 240,
where the answer is 140. On the same model, `select ad_id, total` gives `ad1`
100 and `ad2` 140, and `select ad_type, avg_amount` gives 70.0. Both are
planned.

### `axes` and `by`

A semi-additive Measure or Metric declares `SemiAdditivity { axes, by, strategy }`:

- `by` names one Dimension, and `latest` and `earliest` order by it.
- `axes` names the Dimensions inside which the latest or earliest rows are
  taken. It is empty where the author writes none.
- The family then adds across every other Dimension, so the aggregate is
  additive except along `by`.
- The semi-additive strategy applies when the request drops `by`, or buckets it
  coarser than the source holds it.
- The semi-additive strategies are `latest` and `earliest`. `average` is not
  one, and the semantic model drops it. ADR 0074 reads `first` and `last` as
  `earliest` and `latest`, and their removal is not ruled here.

With the rows above, where a row is written when a balance changes (the Events
shape), October's `balance` is 130 with `axes: [] by date` and 130 + 230 = 360
with `axes: [account] by date`.

### Where `by` and the axes come from

The Datasets that an aggregate reads are the Datasets that its arguments read.
For an aggregate whose arguments read no column, such as `count(*)`, they are
the source itself where the source is a Dataset, and none on a composed source.
The Datasets that a semi-additive Metric reads are those that its aggregate
nodes read.

- `by`, written or default, and each axis are Dimensions of those Datasets,
  where the aggregate reads one. So all copies of a fanned row carry the same
  `by` and the same axes.
- Every source that exposes the aggregate serves its `by`, as ADR 0074 rules
  for each axis.
- `by` is never one of the `axes`.
- `by` and each axis that the request does not select extend the demand, as ADR
  0074 says for an axis.

With no `by`, `by` is the time axis (`CONTEXT.md` **Time axis**) of the
Datasets that the aggregate reads: `occurred_at` of a Timeseries, `event_time`
of an Events and `snapshotted_at` of a Snapshot Dataset. Where they give exactly
one time axis, that axis is the default; an Scd validity range is not one time
axis. A composed source's derived shape does not decide it.

Compile refuses a source that exposes a semi-additive Measure or Metric:

- with no `by`, where the Datasets that it reads give no single time axis:
  none has a Timeseries, Events or Snapshot shape, they give two or more
  different time axes, or it reads no Dataset;
- whose `by` is not a temporal Dimension, or is one of its `axes`;
- whose `by` or one of whose axes is not a Dimension of the Datasets that it
  reads, where it reads one;
- whose `by` or one of whose axes two or more of the Datasets that it reads
  map, unless it is the time axis of exactly one of them;
- whose `by` it does not serve.

So every shape × additivity pair stays legal, except where compile refuses a
source that exposes a semi-additive aggregate as above, and additivity is still
never derived from the shape (ADR 0030). The planner applies the same default,
so the manifest gains only the `by` field.

### A window's order key carries its direction

A window call is a `Function(Call)` with `over` (ADR 0091, rule 1). Each
order key of its `over` carries a `SortDir`. Because `Expr` is
model-owned and `semstrait-ir` depends on `semstrait-model` (ADR 0001),
`SortDir` and `NullOrdering` move to `semstrait-model`; `semstrait-ir`
re-exports them, and `SortNode` continues to use those same carriers. The
planner writes NULLs last on each window it builds, never
`NullOrdering::Unspecified`, and no window frame on `rank` or `row_number`.
This grows the expression vocabulary, which grows only by ADR (ADR 0066,
ruling 2). No window function is added.

### What retires

- `PlanError::MixedRowSets` and `PlanError::StrategyOrderUndecided` are not
  raised. Each of their cases has a plan, or compile refuses it, or the
  planner refuses it with `PlanError::FanOutMarkerNotPlaceable` (**Where the
  marker has no place**).
- The `latest` / `earliest` join retires, and with it that join's
  `IsNotDistinctFrom`. `BinaryOpKind` gains `IsNotDistinctFrom` for the join of
  a Metric's subtree of two aggregate steps (ADR 0074, *A Metric*).
- A semi-additive strategy writes no aggregate family other than the
  definition's, so `Count` combining by `Sum`, and the formula push-down's
  `sum` for a `count` (ADR 0074, *A Metric*), are the places where the planner
  writes another family.

### Amends

- **0074:** in *Additivity decides whether a partial aggregate may be combined
  again*, a semi-additive aggregate is additive except along its `by`, a partial
  aggregate of it is grouped by its `axes` and its `by`, and "Apart from a
  semi-additive strategy" no longer holds (0076's *What retires*); in *A
  Joinset*, the fanned operand's aggregate reads one row per row identity
  through a `row_number` marker, which partitions by each column of the identity
  as its own column, in place of the distinct-rows aggregate; in *Re-aggregating
  rows that are already aggregates* and *A semi-additive aggregate's strategy*,
  the semi-additive strategy applies when the request drops `by` or buckets it
  coarser than the source holds it, it keeps the rows where its `rank` marker is
  1 in the one aggregate step, with no inner aggregate, no join and no
  `IsNotDistinctFrom`, `by` and each axis are Dimensions of the Datasets that
  the aggregate reads, where it reads one, and extend the demand, as 0076's
  *Where `by` and the axes come from* says, and `average` is not a
  semi-additive strategy; in
  *Aggregates that need different row sets* and *Advisories and refusals*, no
  request is refused with `MixedRowSets` or `StrategyOrderUndecided`; the rest
  of 0074 stands (0076's *A marker gives each aggregate its rows*, *`axes` and
  `by`*, *Where `by` and the axes come from* and *What retires*).
- **0071:** in *A tree gives the computation, and the role gives the placement*,
  as ADR 0074 changed it, a subtree's aggregation is one aggregate step, and two
  in a Metric's subtree of two aggregate steps (ADR 0074, *A Metric*), and a
  projection below it may add a marker that gives an aggregate its rows; in
  *Aggregation-level conditions*, as ADR 0074 changed it, an aggregate that
  reads a marker has its condition joined with `And` onto `<marker> = 1`; the
  rest of 0071 stands (0076's *A marker gives each aggregate its rows*).
- **0030:** in its *Decision Outcome*, every shape × additivity pair is legal,
  except where compile refuses a source that exposes a semi-additive aggregate
  under 0076's *Where `by` and the axes come from*, and additivity is still
  never derived from the shape; as ADR 0074 changed it, a semi-additive
  aggregate whose `by` the request drops or buckets coarser than the source
  holds it applies its semi-additive strategy instead of the lossy
  re-aggregation advisory; the rest of 0030 stands (0076's *`axes` and `by`* and
  *Where `by` and the axes come from*).
- **0075:** in *Names are unique, and the planner's own names are reserved*, the
  bullet on the `latest` / `earliest` join has no subject, and the cases of one
  definition giving two output fields in one output schema are a Dimension that
  a marker or a semi-additive partial aggregate reads at values other than the
  bucketed value that the request selects, beside that value, and in the bullet
  *One definition can give two output fields in one output schema*, the `Max` /
  `Min` side, the two join inputs and the inner-aggregate example have no
  subject; a marker is an output field that the planner creates and carries a
  reserved name; in *Ruled elsewhere, not here*, the `on` of the `latest` /
  `earliest` join has no subject; the rest of 0075 stands (0076's *A marker
  gives each aggregate its rows*).

### Consequences

- **`crates/`**: the planner gains `PlanError::FanOutMarkerNotPlaceable {
  semantics, path }` (**Where the marker has no place**); the rest of this
  decision changes nothing in `crates/`.
- Bad, because a request is refused with `PlanError::FanOutMarkerNotPlaceable`
  where its `row_number` marker would read rows that a Unionset passes up,
  partial aggregates grouped by a name that the fanned row does not
  determine, or rows for which one result would read two `row_number` markers
  (**Where the marker has no place**). In `test_data/`, `order_details` in `comprehensive_ecommerce.yaml`
  admits such requests: `customer_count` with `product_name` or
  `product_category`. Gain: no plan gives them wrong rows, such as 60.0 for
  52.5, 240 for 140, or 3 customers for 2.

## More Information

Decided 2026-10-04, and on
2026-10-18 for the
formula push-down's `sum` for a `count`, the marker of a Measure lowered on a
Joinset, and the join of a Metric's subtrees, and on
2026-10-19 for the
requests whose `row_number` marker has no place. A prototype
runs each marker through the `substrait`
crate 0.63 into DataFusion 55, against a SQL oracle.
