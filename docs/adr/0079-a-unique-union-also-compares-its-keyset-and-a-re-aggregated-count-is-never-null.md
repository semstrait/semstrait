---
number: 79
title: A unique union also compares its keyset, and a re-aggregated Count is never NULL
status: accepted
date: 2026-10-05
source:
- CONTEXT.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md
- docs/adr/0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md
- '`guard_composer_constituents` in `crates/semstrait-manifest/src/compile/relationships.rs`'
- a prototype run on DataFusion 55, by SQL and through Substrait
- the human's rulings (2026-10-17)
- the decisions of 2026-10-18
links:
- target: 74
  kind: amends
- target: 75
  kind: amends
---

# A unique union also compares its keyset, and a re-aggregated Count is never NULL

## Context and Problem Statement

ADR 0074 rules two things that this ADR changes.

**A mode-`unique` Unionset.** ADR 0074 (*A Unionset*) says that its
`UNION DISTINCT` compares "the columns that the request's expressions read (the
union's output, ADR 0073), and no others. A keyset adds nothing to it." ADR
0075 (*What an operand gives for a join condition*) says that "the constituent
outputs that field even where the request does not select its semantics". So
where a Unionset is a Joinset constituent, its output holds more than the
request reads, and two faithful plans of one request can disagree:

- `all_sales` is a mode-`unique` Unionset. `web_sales` gives (`c1`, 10), and
  `store_sales` gives (`c2`, 10).
- `sales_with_customers` is `all_sales INNER customers`, on `customer_id`, with
  integrity `assumed`. So the join drops no row, and ADR 0074 (*A Joinset*,
  **Leaving out an operand**) lets the Joinset prune `customers`.
- The request is `select revenue`, and `revenue` sums `amount`. With the join,
  the distinct compares (`customer_id`, `amount`), and `revenue` is 20. With
  `customers` pruned, the distinct compares `amount`, and `revenue` is 10.

At the root, the same rule merges two different rows: two sales of 10 count
once.

A Unionset under a Joinset compiles, and these rules
hold for it.

**A re-aggregated Count.** ADR 0074 (*Additivity decides whether a partial
aggregate may be combined again*, as ADR 0076 restates it) re-aggregates
`Count` partial aggregates by `Sum`. `Sum` over no rows gives NULL, and `Count`
over no rows gives 0. Two plans give a request with no grouping no partial
aggregate:

- Every branch of a Unionset is pruned, and ADR 0073 (*A Unionset prunes a
  branch its predicate excludes*, **Every branch excluded**) writes the empty
  read, "so a request with no grouping still returns its one row of
  aggregates". `select order_count where region = 'apac'` gives NULL.
- A Joinset operand pre-aggregates by the fields that the join condition reads,
  and the join gives no row. Over `orders INNER customers`, with no `vip`
  customer, `select order_count where segment = 'vip'` gives NULL.

A Count over the same rows gives 0 in both cases. The prototype runs both plans
on DataFusion 55, by SQL and through Substrait.

## Considered Options

- The keyset, and the fields that a join that is built reads, in the distinct;
  and `Coalesce(Sum(<partial>), 0)` for every re-aggregated Count (chosen).
- **The distinct over what the request's expressions read only** (ADR 0074 as
  written). Two different rows that agree there count once, and under a
  Joinset the answer changes with pruning.
- **No pruning above a mode-`unique` Unionset.** The human ruled that where the
  Joinset prunes, the Unionset serves what the request asks for: "what is
  requested from query, is actually the scope".
- **The `on` fields kept in the Unionset's demand after pruning.** The human
  ruled that only a join that is built extends the demand.
- **A typed refusal of a mode-`unique` Unionset under a Joinset.** It refuses a
  valid request.
- **No rule while compile refuses the form.** The human ruled that the
  planner's rules do not depend on what compile produces today.
- **The empty read as rows**, so that one `Count` reads it. It fixes the
  Unionset case only. The Joinset case still gives NULL.
- **Substrait's `sum0`**, whose sum of zero elements is zero. DataFusion 55 has
  no `sum0`, and it refuses the plan ("Aggregate function sum0 is not
  supported").
- **The `Coalesce` only where the re-aggregation has no grouping.** A
  semi-additive partial's marker and condition apply above the combine (ADR
  0076, *A marker gives each aggregate its rows*), so in a grouped
  re-aggregation, too, no partial aggregate of a group can pass the condition.
  The rule would also read the grouping as well as the family.
- **NULL for a re-aggregated Count over no rows.** `CONTEXT.md` **Additivity**
  recombines parts "without loss", and NULL in place of 0 is a loss.

## Decision Outcome

In short:

- A mode-`unique` Unionset's distinct compares its whole output. That output
  also holds the Dimensions of its first guaranteed keyset, and the fields that
  a join built above it reads from it.
- Where a Joinset prunes its other operand, no join is built, so nothing above
  extends the Unionset's demand.
- A re-aggregated Count is `Coalesce(Sum(<partial>), 0)`, so it is never NULL.

These rules hold for a Unionset under a Joinset, in either mode, whatever
compile produces today.

### A unique union compares its output, its keyset included

The distinct compares every output field of the union. The union's output
holds:

- the names of the Unionset's demand (ADR 0073). These include the fields that
  the request's expressions read (ADR 0074), the `by` and each axis of a
  semi-additive aggregate (ADR 0076, *Where `by` and the axes come from*), and,
  where a Joinset above builds its join, each field that the join reads from
  the Unionset: its `on` and its relationship filter (ADR 0075), and the row
  identity where the join fans out the Unionset (ADR 0074, *A Joinset*, **The
  row identity**);
- the Dimensions of the Unionset's first guaranteed keyset, in manifest order,
  which the Unionset adds to the names it sends each branch.

Above the distinct, a keyset Dimension that the parent does not read goes no
further. The aggregate above the union does not group by it, and the
projection to what the parent asked for (ADR 0073, *The build order*, stage 7)
drops it. Every branch gives rows, as ADR 0074 says for mode `unique`.

A Unionset derives no keyset, so its guaranteed keysets are the ones that its
author declares inside its boundary (ADR 0048, *Derivation, per form*). Under
the keyset claim, "nothing outside it can diverge" (ADR 0048, *A keyset is a
determinant, and that is why the arithmetic is set algebra*). So every
guaranteed keyset keeps the same rows, and the first one in manifest order
gives one plan. ADR 0074's row identity makes the same pick (*A Joinset*,
**The row identity**).

So a mode-`unique` union drops each copy of one row and keeps different rows.
With the keyset `sale_id`, `web_sales` gives (`s1`, 10), and `store_sales` gives
(`s2`, 10) and (`s1`, 10). `select revenue` compares (`sale_id`, `amount`),
keeps (`s1`, 10) and (`s2`, 10), and gives 20.

### Only a join that is built extends the demand

Where the Joinset prunes its other operand (ADR 0074, *A Joinset*, **Leaving
out an operand**), it builds no join, and the Unionset gives what the parent
asks for, with its keyset. ADR 0074's three conditions for pruning do not
change.

- With a keyset, the plan with the join and the plan without it keep the same
  rows, because the keyset fixes every other field of a row.
- With no keyset, they can keep different rows, so the answer depends on
  whether the Joinset prunes. In the example in the context, `revenue` is 10
  with integrity `assumed`, where `customers` is pruned, and 20 with integrity
  `none`, where the join is built.

The planner returns `Advisory::UniqueWithoutKeyset { semantic_source }` beside
the plan where the plan reads a mode-`unique` Unionset that has no guaranteed
keyset, under a Joinset or not: its rows can merge where they agree on every field that the distinct
compares. ADR 0074 has the same pattern for `Advisory::FanOut`, which is raised
only where no keyset gives the row identity (*Advisories and refusals*).

### A re-aggregated Count is never NULL

The planner writes each re-aggregation of `Count` partial aggregates as
`Coalesce(Sum(<partial>), 0)`: above a union and above a join, with a grouping
or without one. It writes the re-aggregation of `regr_count` partial aggregates
the same way, because `regr_count` also counts rows and is never NULL (ADR 0087,
rule 17).

- The `Coalesce` changes a value only where the re-aggregation of a group reads
  no partial aggregate that passes its condition: with no grouping, as in the
  context, or where the condition rejects every partial aggregate that a
  semi-additive marker keeps (ADR 0076, *A marker gives each aggregate its
  rows*).
- A Metric's formula reads the coalesced value.
- `coalesce` is a function in the function catalog (ADR 0092, rule 6).
- ADR 0073's empty read does not change.

### Amends

- **0074:** in *A Unionset*, mode `unique`'s `UNION DISTINCT` compares every
  output field of the union, which 0079's *A unique union compares its output,
  its keyset included* lists, so a keyset counts; a keyset Dimension that the
  parent does not read goes no further than the distinct; and **No advisory**
  does not hold for a mode-`unique` Unionset that has no guaranteed keyset; in
  *Considered Options*, the keyset half of **Deduplicating mode `unique` over
  more than the request's rows** is the form that 0079 chooses; in *A Joinset*,
  **Leaving out an operand**, only a join that is built extends a Unionset's
  demand, so over a mode-`unique` Unionset with no keyset the plan that prunes
  can keep other rows than the plan with the join; in
  *Additivity decides whether a partial aggregate may be combined again*, as
  ADR 0076 changed it, `Count` combines by `Sum`, written as
  `Coalesce(Sum(<partial>), 0)`; in *Advisories and refusals*, the planner
  returns `Advisory::UniqueWithoutKeyset` beside the plan where the plan reads a
  mode-`unique` Unionset that has no guaranteed keyset; the rest of 0074 stands (0079's *A unique union
  compares its output, its keyset included*, *Only a join that is built extends
  the demand* and *A re-aggregated Count is never NULL*).
- **0075:** in *Ruled elsewhere, not here*, which output fields a mode-`unique`
  Unionset's distinct compares when it passes one up for a join condition is
  0079's *A unique union compares its output, its keyset included*; the rest of
  0075 stands.

### Consequences

- Good, because a mode-`unique` Unionset with a keyset drops the copies of one
  row and keeps different rows, and pruning does not change its answer.
- Bad, because with no keyset, the answer of a request over a Joinset depends
  on whether the Joinset prunes (*Only a join that is built extends the
  demand*). The advisory says so.
- Bad, because every plan that re-aggregates a Count carries a `Coalesce`, also
  where it changes no value.
- `CONTEXT.md` **Keyset** lists a mode-`unique` Unionset's distinct among the
  uses of a keyset.
- `crates/` is not changed by this decision.

## More Information

Decided 2026-10-05, and on
2026-10-17 for `regr_count`. A prototype
runs each re-aggregation by SQL and through the `substrait` crate 0.63 into
DataFusion 55. Item 2, the `on` of the `latest` / `earliest` join,
lapsed with ADR 0076.
