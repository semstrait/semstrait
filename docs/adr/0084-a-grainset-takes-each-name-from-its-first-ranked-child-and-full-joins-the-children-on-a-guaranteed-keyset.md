---
number: 84
title: A Grainset takes each name from its first-ranked child and full-joins the children on a guaranteed keyset
status: accepted
date: 2026-10-06
source:
- CONTEXT.md
- docs/adr/0033-composite-plan-shape.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0056-composed-temporal-shape-is-derived-at-compile.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md
- docs/adr/0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md
- docs/adr/0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md
- docs/adr/0083-a-metric-declares-no-additivity.md
- '`resolve_cross_grain` and `equi_condition` in `crates/semstrait-manifest/src/compile/relationships.rs`, and `plan_grainset` in `crates/semstrait-manifest/src/compile/assemble.rs`, and `GrainsetStrategy` in `crates/semstrait-planner/src/grainset_strategy.rs`'
- the research on how MetricFlow and Cube join aggregated results
- the human's ruling of 2026-10-06
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18 and 2026-10-19, and the research on composition constituents
- the decision recorded (a Metric is computed inside one child, for a Metric that a child maps whole)
- the decisions recorded (2026-11-06 and 2026-11-09), and `crates/semstrait-planner/src/grainset_metrics.rs`
- the decisions of 2026-11-10 and 2026-11-11
links:
- target: 33
  kind: amends
- target: 48
  kind: amends
- target: 65
  kind: amends
- target: 73
  kind: amends
- target: 74
  kind: amends
- target: 77
  kind: amends
- target: 80
  kind: amends
- target: 64
  kind: amends
- target: 71
  kind: amends
- target: 75
  kind: amends
- target: 83
  kind: amends
- target: 96
  kind: amendedby
- target: 97
  kind: amendedby
---

# A Grainset takes each name from its first-ranked child and full-joins the children on a guaranteed keyset

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *How they are joined*, rule 6 orders the left-deep joins by the child rank with the granularity fit first, then fewest Dimensions, then name order, so that the first child is at the requested granularity, or, where no chosen child carries it, the chosen child closest to it; and the joins are `Full` (0096's *Drill-across at one granularity*); in rule 3, the child that each name goes to is the child that the child rank ranks first for it alone (0096's *The pick*); in *Which children answer*, rule 2, "a Grainset that is a constituent of another composition fixes the granularity that it declares" and "runs rule 3's pick among the children that pass the grain test at the granularity that it declares" read the granularity that it carries, declared or derived (0096's *Shared granularity*); in *Which children answer*, rule 1, after "as ADR 0074 picks it." the child that answers alone takes the whole demand, each predicate and each activated named Filter included, and it reads its own signature for the filter (rule 8) (0096's *A named filter on a Grainset*); the rest of 0084 stands.
> - [ADR 0097](0097-a-unionset-prunes-a-branch-whose-children-cannot-serve-the-demand-with-an-advisory.md): in *Which children answer*, rule 1, "Where its parent takes rows or partial aggregates from it, or rule 4 leaves it no drill-across, and no child serves the whole demand, the plan fails with `GrainNotServable` at its frame path" does not apply to a Grainset branch of a Unionset whose children cannot serve by what they map: that branch is pruned (0097's *The rule*); a branch that passes the test but fails only because its parent takes partial aggregates from it stays refused with `GrainNotServable` (0097's *Partial aggregates*); the rest of 0084 stands.

## Context and Problem Statement

ADR 0074 lets a Grainset answer from one child only: "A Grainset stands in one
of its grains; ADR 0073 keeps cross-grain units out of MVP" (*A Grainset*).
ADR 0073 gave the reason: "every grain of one Grainset maps the same interface"
(*One decision shape, checked per body*). ADR 0077 removed that reason: "A
Grainset's child serves only the names it maps", and "A demand that only two or
more children serve together is the cross-grain case that ADR 0073 keeps out of
MVP, and this ADR does not decide it" (*A constituent serves only the names it
maps*). A Grainset's mask is "the **union of its grains'**" (ADR 0048, *Grain
groups are dropped, and a Grainset emits no implicit Unionset*), so such a
demand passes the mask test, and no child serves it.

Compile already makes one Joinset per ordered pair of children, finer to
coarser (ADR 0054, *Consequences*): `Left`, `ManyToMany`, `Integrity::None`,
`CrossFilter::None`. Its condition is one equality per keyset name over each
child's own column (`equi_condition`). It carries no granularity, because its
children disagree (ADR 0056).

Take the Grainset `spend`, whose children carry the guaranteed keyset
`(date, campaign)`, with the time axis `date`:

`spend_daily` (granularity Day; maps clicks):

| date | campaign | clicks |
|---|---|---|
| 2026-03-01 | A | 10 |
| 2026-03-02 | A | 20 |
| 2026-04-01 | A | 30 |
| 2026-03-15 | B | 40 |

`spend_monthly` (granularity Month; maps budget):

| date | campaign | budget |
|---|---|---|
| 2026-03-01 | A | 100 |
| 2026-04-01 | A | 100 |
| 2026-03-01 | B | 50 |

`select date at Month, campaign, clicks, budget` needs both children. Through
compile's Joinset, `spend_daily.date = spend_monthly.date` holds only on the day
that is the month's value, so B's March budget is NULL where it is 50. A request
at Day goes to that Joinset too, since a source with no granularity "serves
every requested granularity" (ADR 0074, *Every request is planned against its
grain first*), though `budget` has no value per day.

## Considered Options

- **Each child gives final aggregates at the demand's grain, and the Grainset
  drills across them; a guaranteed keyset allows it** (chosen).
- **The Grainset stands in compile's cross-grain Joinset.** Its condition
  compares a day with a month, and with no granularity it serves a request at
  Day, as above. The human ruled that the planner decides these joins at plan
  time, and that planning them at compile is a later offline mode.
- **The Grainset joins its children's rows.** The joins start from the finest
  child; each coarser child is joined `Left` on the keyset, with the finer child's time
  axis truncated to the coarser child's granularity; the join is many-to-one,
  and a marker gives each coarser child's aggregates one row per row identity.
  It is exact, and it answers a Dimension that only the finer child maps by
  repeating the coarser value in each group. But its condition compares a
  bucket of rows, which ADR 0080 rules out: "Rows and partial aggregates never
  carry a bucket, so a join compares a bucket only between final aggregates"
  (*Only a frame that gives final aggregates buckets*). And for data at
  different grains, the products join aggregates: MetricFlow joins daily and
  monthly bookings `ON a.ds = b.ds__month`; Cube: "the facts are stitched
  together at the grain you group by"; Kimball: "A BI application must never
  issue SQL that joins two fact tables together across the fact table's foreign
  keys".
- **The keyset as the join condition.** Each child is grouped by the keyset's
  Dimensions, with the time axis at the requested granularity, and the Grainset
  aggregates again above the join. A non-additive Measure cannot give partial
  aggregates, so for `avg_budget = avg(budget)` the monthly child gives its rows
  bucketed to Year: two rows for A in 2026. Each daily group then meets two
  monthly rows, and `select date at Year, campaign, clicks, avg_budget` gives
  A's `clicks` as 120 where it is 60.
- **Pairs only, as compile makes them.** A demand that three children serve
  together has no plan.
- **`Full` joins, with each grouping Dimension coalesced over the children and a
  NULL-safe key** (chosen), as MetricFlow and Cube join aggregated results. It
  keeps a group that only one child holds. The key `IsNotDistinctFrom` matches
  a NULL group to the NULL group of another child, so that group gives one row.
- **`Left` joins from the child at the requested granularity, or the closest to
  it, with a plain `Eq` key.** A group that only an attached child holds is not in the answer,
  and a NULL group keeps the values of one child only.
- **A sentinel key, `coalesce(a, default) = coalesce(b, default)`.** A real value
  that equals the default joins the NULL group.
- **The key `coalesce(a, b) = coalesce(b, a)`.** It matches a NULL group with
  every group.
- **The combine `join`, with the delivery rule reading the source role.** The
  operands of a Joinset that builds its join give rows or partial aggregates
  (ADR 0074, *A Joinset*, **Pre-aggregation**; ADR 0080, *Partial aggregates are
  an intermediate step*), so one combine value would give two deliveries. And
  ADR 0073's rule list "never reads which body combined them" (*Three parts*).
- **A Grainset signature for a Metric that no child maps whole**, computed over
  the drill-across's final aggregates, each Measure under the Metric's filters
  (the design: "measures and metrics can be served partially, from
  different grains if there is compatiblity and keysets") (chosen, rule 8). The
  tree reads each name that a child maps whole as a final value in the
  drill-across output, and each Measure under a condition is its own item of the
  pick (rule 9). So `web_cpc2 = web_budget / clicks`, over `web_budget = budget`
  with `filter: channel = 'web'`, gives 60 / 30 = 2, not 100 / 30. The Grainset
  stores the formula and its filters only; the plan decides the children and the
  join for each request.
- **A filter after the join.** The Grainset would apply a Metric's `filter:` to
  the joined values. A non-additive Measure cannot take it there: count distinct
  `users` under `channel in (web, app)`, with web u1, u2 and app u2, u3, gives 3
  in the child and 2 + 2 = 4 after the join.
- **A Grainset signature for every exposed name**, in the Unionset's form. The
  Grainset would store entries that no child needs: a child that maps a name
  whole already holds its signature, and its own filter governs.
- **The Week rule as an amendment of ADR 0074 in this ADR.** The human ruled
  that ADR 0074's grain test is sharpened in place (*Week rolls up to nothing
  coarser*).

## Decision Outcome

In short:

- Each demanded name goes to the child that the Child rank ranks first for that
  name alone. Where every name goes to one child, that child answers alone.
- Otherwise the Grainset drills across the children that the names go to, a
  Grainset that is a constituent of another composition only where its demand
  takes final aggregates: each gives final aggregates at the grain of the
  Grainset's demand, and `Full` joins, left-deep, compare the demand's grouping Dimensions with `IsNotDistinctFrom`,
  one-to-one. The joins follow the Child rank with the granularity fit first,
  then fewest Dimensions, then name order, so the first child is at the
  requested granularity, or, where none carries it, the closest to it. Each grouping Dimension of the output
  is the `coalesce` of that Dimension over the children.
- The children must share a guaranteed keyset. Without one, the Grainset drills
  across no children.
- A Metric that one child maps whole is served by that child alone, with its
  own signature, body and `filter:`. A Metric that no child maps whole and that
  the children compute together is computed by the Grainset: its formula runs
  above the joins, on each joined row of the final aggregates of the children
  that serve the names it reads. Where it takes ADR 0074's parts or has its own
  `agg:`, a step above the formula sums the parts or applies the `agg:`.
  Compile refuses a Metric of a public Grainset that its
  children cannot compute. A composition above the Grainset still lowers its
  own Metric over the Grainset's Measures where the Grainset does not compute it
  (*A Metric that no child maps whole*).
- A source at Week serves no requested granularity of Month, Quarter or Year
  (*Week rolls up to nothing coarser*).

### Which children answer

1. Where every demanded name goes to one child (rule 3), that child answers
   alone, in its own frame and with its own signatures. Rules 4 to 7 apply only
   where the names go to two or more children, and rule 9 says when a child
   serves a Measure under a condition. A Grainset that is
   a constituent of another composition drills across only where the delivery
   that its parent's demand fixes for it is final aggregates (ADR 0080, *The
   parent's demand fixes an operand's delivery*), so that no aggregate step
   above it combines them again (ADR 0074, *A Grainset*). Where the chosen
   children share no guaranteed keyset (rule 4), or its parent takes rows or
   partial aggregates from it, one child that serves the whole demand answers
   alone, as ADR 0074 picks it. Where its parent takes rows or partial
   aggregates from it, or rule 4 leaves it no drill-across, and no child serves
   the whole demand, the plan fails with `GrainNotServable` at its frame path
   (ADR 0074, *A Grainset*).
2. Where the demand selects the time axis with no requested granularity, the
   Grainset first fixes one: a Grainset that is a constituent of another
   composition fixes the granularity that it declares (ADR 0074, *A
   Grainset*) for the time axis of its own output. A time axis that only the
   formula grain of a Metric that it computes reads (rule 8) never reaches its
   output. Any other Grainset fixes the coarsest granularity that ADR
   0074's pick gives for any demanded name alone. A demanded name includes each
   name that the tree of a Metric that the Grainset computes reads (rule 8).
   Rules 3 to 9 read the fixed
   granularity as the requested granularity, and the Grainset gives it to each
   chosen child. On every Grainset, the formula grain of rule 8 is the
   exception: it never reaches the output, and rule 8 gives the granularity at
   which it reads each Dimension. A Grainset that is a constituent runs rule 3's pick among the
   children that pass the grain test at the granularity that it declares,
   whatever granularity the demand asks; each chosen child still buckets at
   the requested granularity (rule 5).
3. Each demanded Measure and Metric goes to the child that the Child rank (ADR
   0096, *The pick*) ranks first for it alone. The rank runs among the children
   that map the name, that pass
   ADR 0074's grain test for the requested granularity, and that map every
   Dimension and named Filter that the demand reads: selected, in the request
   predicate, or activated. A Metric that a child maps whole goes to a child
   that has a signature for it (rule 8). A Metric that the Grainset computes
   goes to no child: each name that its tree reads is demanded and goes to its
   rank-first child (rule 8). A Metric's `filter:` reads no named Filter (ADR 0071,
   *Placement is read from the role*). A child that no name goes to is pruned.
4. The chosen children share a guaranteed keyset. ADR 0048 reads "the
   **grains'** guaranteed keysets, never the Grainset's own", and says "A
   Grainset whose grains carry no keyset simply offers no rollup composition and
   each grain remains individually routable" (*Grain groups are dropped, and a
   Grainset emits no implicit Unionset*). So without one, the Grainset drills
   across no children. Where no child serves the whole demand either (rule 1),
   it is not a candidate (ADR 0064): `NoCandidate`, or `GrainNotServable` where
   `from` names it.

### How they are joined

5. Each chosen child gives final aggregates at the grain of the Grainset's
   demand, or, for a Metric with its own `agg:` that the Grainset computes, at
   that Metric's formula grain (rule 8), and it buckets in its own aggregate
   (ADR 0080, *Only a frame that gives final aggregates buckets*). For a Metric
   that the Grainset computes with ADR 0074's parts (rule 8), each child that
   gives a value that its formula reads computes the columns of the part key on
   its own rows, and groups by them as well as by the demand.
6. The Grainset's combine is `drill-across`. Its frame reads the chosen children
   as its operands and joins their final aggregates in binary joins, left-deep.
   The joins follow the Child rank with the granularity fit first, then fewest
   Dimensions, then name order: the form that the pick already uses where the
   demand selects the time axis without a granularity. So the first child is at
   the requested granularity, or, where no chosen child carries it, the chosen
   child closest to it. The order keeps the plan
   bytes stable, and the rows do not depend on it. Each chosen child passes the grain test (rule
   3), so it is at the requested granularity or finer. So at Month,
   `spend_monthly` joins first and `spend_daily` second; at Year, with a name
   for each child, `((spend_yearly ⋈ spend_monthly) ⋈ spend_daily)`; at Day,
   with children at Minute, Hour and Month and a name for each of the Minute and
   Hour children, the Hour child joins first and the Minute child second, and
   the Month child serves no request at Day.
   Each join is `Full`. Its condition compares each grouping Dimension of the
   demand, and each column of a part key (rule 5), with `IsNotDistinctFrom`,
   between the child that the join adds and the earlier children. The first
   join compares the keys of the first two children. Each later join compares its child's keys with the `coalesce` of
   the earlier children's keys. Each side has one row per group, so the join is
   one-to-one and no marker is written. Every group that any chosen child holds
   is in the answer. A NULL grouping value matches the NULL group of another
   child, so that group gives one row. A dialect whose engine lacks
   `IS NOT DISTINCT FROM` declares a lowering to
   `(a = b) OR (a IS NULL AND b IS NULL)` (ADR 0092, rule 1).
   The driver builds each join with `step_join`, or with `step_cross` where the
   demand has no grouping Dimension and no part key, and each child gives one
   row. The delivery rule reads the combine `drill-across` as final aggregates.
7. Each child writes its output fields under its own alias, as an operand of a
   Joinset does (ADR 0075, *Every output name is the one its parent reads*).
   Above the joins, the Grainset's frame projects each name once under the
   Grainset's own alias: each grouping Dimension as the `coalesce` of that
   Dimension over the children, each column of a part key as the `coalesce` of
   that column over the children that group by it (rule 5), each Measure and
   Metric from the child that it went to (a Metric that the Grainset computes
   is not among them: the Compute frame of rule 8 computes it above the
   drill-across), and each name that a Grainset branch NULL-fills as a typed NULL.
   Where its reader carries another qualifier, as a Unionset does whose
   disjoint branch it is (rule 1), the frame's last projection, which that
   reader reads, writes each name
   under the reader's qualifier (ADR 0075, *Every output name is the one its
   parent reads*). The drill-across
   frame of the Grainset has no aggregate step, so it does not bucket again.
   The steps of rule 8 sit above it: a `sum` adds the parts of a Metric that
   takes ADR 0074's parts, and a Metric's own `agg:` aggregates its formula's
   values up to the demand's grain. Where the formula grain of that Metric is
   the demand's grain, its `agg:` aggregates the joined rows in place, and it
   reads no child again. A predicate on a
   Dimension, and an activated named Filter, goes into each chosen child, which
   maps its names (rule 3): the human's reason on ADR 0080, "filter which by
   dimension mapping exposed for both side, which means that it must be applied
   for both side" (*A predicate copy goes into its operand on every join type*).
   A predicate on a Measure or a Metric sits above the joins. ADR 0071's table
   keeps it above the joins wherever they null-fill the child that gives it, and
   a `Full` join null-fills every child that lacks a group. Inside one child, the
   predicate would remove the group from that child only, and the other children
   would still hold it.

With a third child `spend_yearly` (granularity Year; maps budget), with A's
2026 budget 200 and B's 50:

| request | children joined, first child first | answer |
|---|---|---|
| `date at Month, campaign, clicks, budget` | monthly, daily | (2026-03, A, 30, 100), (2026-03, B, 40, 50), (2026-04, A, 30, 100) |
| `date at Year, campaign, clicks, budget` | yearly, daily | (2026, A, 60, 200), (2026, B, 40, 50) |
| `date, campaign, clicks, budget` | Month first (rule 2), then monthly, daily | as at Month |
| `clicks, budget` | yearly, daily, crossed | (100, 250) |
| `date at Day, clicks, budget` | none: no child gives `budget` at Day | `NoCandidate` |

### A Metric that no child maps whole

8. A Metric that one child maps whole is served by that child alone, in that
   child's own frame, with its own signature, body and `filter:` (ADR 0074, *A
   Metric*). On a Grainset's child, a semantics has a signature only where the
   child maps every name that its body and its `filter:` read: a Grainset
   applies no row-level filter above its children's final aggregates. A
   row-level Metric with no `agg:` is planned as a Dimension (ADR 0074, *A
   Metric*), and it is a Dimension in rules 3 and 6 to 8.
   - **Which Metric.** The Grainset computes a Metric that no child maps whole
     and that its children can compute together: each name that its body and its
     `filter:` read is mapped by a child, or is a Metric that the Grainset
     computes (`cpm = cpc * 1000` reads `cpc`), and its body reads at least one
     Measure or a Metric that is not row-level. The Metric also passes the
     checks of **Refusals**: where one fails on a nested Grainset, the Grainset
     does not compute it (**Nested**).
   - **Signature.** The Grainset owns a signature for each Metric that it
     computes and for no other name. Its tree reads each name that a child maps
     whole as `column(<Grainset alias>, <name>)`: that name's final value in the
     drill-across output. A link to another Metric that it computes stays a
     link (ADR 0088). Its `filter:` is the signature's filter entry. The tree is
     stored at `layer: post_aggregate`: a column of the Grainset's own alias that
     names a Measure or a Metric that is not row-level counts as an aggregate.
     A signature stores only the formula and its filters; the plan decides the
     children and the join for each request. Named Filters get no Grainset
     signature.
   - **Plan.** Each name that the tree reads is a demanded name for rules 2 and
     3. The children drill across with rules 1 to 7, and each gives final
     aggregates at the demand's grain. The formula runs over the joined values
     in one step above the joins, or above the one child where one child answers
     the whole demand (rule 1). Where the demand holds a Metric whose stored
     tree reads the frame's own final values (a Metric that the Grainset
     computes), the driver's Shared rule for a Metric's formula makes the frame
     a Compute frame. A Compute frame is a frame kind of
     its own, outside the closed set of combines: a Strategy only decides (ADR
     0073), and no Strategy returns it, so ADR 0073's table in *One decision
     shape, checked per body* gains no value for it. It reads one frame for each
     formula group. A formula group is a subtree of its own, which this rule
     gives below, or the group of the other values that the demand and the
     formulas read. The Grainset's Strategy decides the frame of each group by
     rules 1 to 7. Each group's frame has a path of its own,
     `<grainset>#group<n>`, with n counting from 0, so an error or an advisory
     inside a group names that group (ADR 0073: a decision is named by its
     path): on a Grainset `g`, the child `c_cost` of the second group is at
     `g / g#group1 / g.c_cost`. A Metric with its
     own `agg:` whose formula grain is finer than the demand's grain plans in a
     subtree of its own (ADR 0074), also beside a Metric without one; one at the
     demand's grain aggregates the joined rows in place (rule 7). In that subtree:
     the Dimensions that its formula reads outside an aggregate
     join the demand of that subtree only, the children drill across at that
     formula grain, the formula runs on each joined row, and its `agg:`
     aggregates the values up to the demand's grain. Such a Dimension is
     demanded with no granularity (ADR 0083, rule 4). The formula grain reads it
     at the finest granularity at which a child that gives a value maps it; a
     declared granularity does not apply to it. The subtree joins the demand's
     subtree `Full` on `IsNotDistinctFrom` with coalesced keys. The `agg:` step groups by the demanded Dimensions at their
     requested granularity: `early_cpc_avg`, with `case when date < 2026-03-02
     …`, at Month runs per day, as on one Dataset, and gives (A, 2026-03-01,
     3.0). A Metric that reads a Metric with its own `agg:` reads that Metric's
     value from its subtree: `avg_cpc2 = avg_cpc * 2` gives (A, 4.0), as on one
     Dataset. A Metric that the Grainset computes, with no `agg:`, whose formula
     reads a Dimension that the request does not group by, takes ADR 0074's
     parts over the drill-across, in a subtree of its own (ADR 0074, *A
     Metric*). A Dimension that the request selects at a granularity is not
     grouped for the formula (ADR 0083, rule 4): the formula reads its value
     before bucketing, through the part key. Step 1 is in each child that gives
     a Measure that the formula reads: the child computes the part key on its
     own rows, and aggregates its Measures grouped by the demand and the key.
     The children join `Full` on the demand and the key, with `IsNotDistinctFrom`.
     The formula runs on each part, and step 2 sums the parts to the demand.
     The Dimensions that the key reads group step 1 only, and never reach the
     output. In the oracle case on `spend_named_filter`'s `spend`,
     `date@month, web_only_cpc`, with `case when channel = 'web' then budget /
     clicks else 0.5 end`, gives 3.5, as on one Dataset: 150 / 50 = 3.0 for the
     web part, and 0.5 once for the `else` part of app and store, where a sum
     per channel gives 4.0. On its `spend_days`, whose two children are at Day,
     `date@month, early_cpc`, with `case when date < 2026-03-03 …`, reads
     `date` per day and gives (2026-03-01, 3.5), as on one Dataset. Where the
     formula is not numeric, the plan refuses the request with
     `PlanError::UngroupedMetricFormula`, as on one Dataset (ADR 0074, *A
     Metric*).
   - **Refusals.** Where the Grainset is public and exposes the Metric itself,
     compile refuses:
     - a name that the Metric reads and no child maps:
       `CompileError::GrainsetMetricReadsUnmapped { grainset, metric, name }`;
     - a `filter:` Dimension that no child maps together with the Measure, or
       with the Metric that a child maps whole, that the filter conditions
       (`web_cpc_in_eu`: no child maps `web_budget` together with `region`):
       `CompileError::GrainsetMetricFilterUnmapped { grainset, metric, measure,
       dimension }`;
     - for a Metric with no `agg:`, a child that cannot form the part key
       (**Plan**) and gives a value that the formula reads, a Measure or a
       Metric that the child maps whole: the child does not map a Dimension
       that the key reads, or maps it at a granularity coarser than the key
       reads, which is the finest granularity at which a child that gives such
       a value maps it (`web_only_cpc` where `spend_monthly`, which gives
       `budget`, does not map `channel`; `early_cpc`, with `case when date <
       2026-03-05 …`, where `spend_monthly` gives `budget` by month and
       `spend_daily` gives `clicks` by day):
       `CompileError::GrainsetMetricPartKeyUnformed { grainset, metric, measure,
       dimension, child }`, where `child` is the child's full name, such as
       `spend.spend_monthly`;
     - children that serve its names and share no guaranteed keyset (rule 4), or
       none at its formula grain for a Metric with its own `agg:`, or, for such a
       Metric, a child that gives a value maps a Dimension of the formula grain
       coarser than the finest granularity at which such a child maps it:
       `CompileError::GrainsetMetricWithoutKeyset { grainset, metric }`. A
       row-level Metric that the formula reads counts as a Dimension of the
       formula grain.
   - **Row-level.** A row-level Metric that no child maps whole is no Metric
     that the Grainset computes (**Which Metric**): on a public Grainset it is
     `CompileError::UnservedComposedSemantics { semantics, semantic_source }`,
     and on a nested one that error stays for it where it is exposed and no
     child, at any depth, has a signature for it (**Nested**).
   - **Nested.** Where a check fails on a Grainset that is nested, the Grainset
     does not compute the Metric, with no error. A composition above the
     Grainset still lowers its own Metric over the Grainset's Measures (ADR
     0074, *A Metric*), and those Measures follow rules 1 to 7 and rule 9; a
     request for a Metric that nothing computes fails at plan with
     `GrainNotServable`, and the semantic model's other names plan.
     `CompileError::UnservedComposedSemantics { semantics, semantic_source }`
     (ADR 0077) stays for an exposed name that no child, at any depth, has a
     signature for and that the Grainset does not compute. The plan refuses what
     depends on the request with `GrainNotServable` and its `nearest` list.
9. Where the Grainset's demand holds a Measure under a condition that a
   composition above lowers onto it (a Metric's `filter:`, or the
   composition's own filter for a Measure), or that a Metric which the Grainset
   computes puts with its own `filter:` on a value that its tree reads, a child
   serves that value, under that condition, only where it also maps every
   Dimension that the condition reads, in rule 1 as in rules 2 and 3. The value
   is a Measure; a Metric that one child maps whole, which that child gives with
   the condition joined by `And` after its own `filter:`, in each aggregate of
   its tree for that Metric; or a Metric that the Grainset computes with its own
   `agg:`, where the condition reaches that Metric's subtree, and the same
   Metric read with and without the condition in one request gets one subtree
   for each. The child
   gives it with the condition joined by `And` after its own, as ADR 0074 joins a
   composition's own filter onto each inlined aggregate (*A Metric*), as an
   output field of its own under a name that the planner makes (ADR 0075, *Names
   are unique, and the planner's own names are reserved*), one field for each
   Measure and condition. The Grainset's decision carries the condition into the
   child's demand. The condition is part of the child's aggregate,
   `agg(x) FILTER (cond)`, and never applies after the join: a non-additive
   Measure under a filter stays exact only there.
10. `validate()` accepts a Grainset signature only for a Metric that no child
    maps whole, and refuses any other.

For `web_cpc = budget / clicks` with `filter: channel = 'web'`, where the
children carry the keyset `(date, campaign, channel)`, `spend_daily` has A's
March clicks 10 on `web` and 20 on `app`, and `spend_monthly` has A's March
budget 60 on `web` and 40 on `app`: no child maps both `budget` and `clicks`,
so the Grainset computes `web_cpc`. Its signature stores only
`divide(column(spend, budget), column(spend, clicks))` and the `filter:`.
`select date at Month, campaign, web_cpc` sends `budget` under `channel = 'web'`
to `spend_monthly` and `clicks` under `channel = 'web'` to `spend_daily`, joins
them, and gives A's March as 60 / 10 = 6.

For `web_cpc2 = web_budget / clicks`, over `web_budget = budget` with `filter:
channel = 'web'`, the signature stores only
`divide(column(spend, web_budget), column(spend, clicks))`. `web_budget`'s
`channel = 'web'` stays in `spend_monthly`'s own signature, and `web_cpc2`
gives 60 / 30 = 2.

For `web_cpc_of_a = web_budget / clicks` with `filter: campaign = 'A'`, the
condition reaches `web_budget`, so `spend_monthly` gives
`sum(budget) FILTER (channel = 'web' AND campaign = 'A')`, and `web_cpc_of_a`
gives 60 / 30 = 2.0.

### Compile's cross-grain Joinsets

11. The planner does not read the cross-grain Joinsets that compile makes (ADR
    0054), and compile keeps making them unchanged. ADR 0073's "ADR 0054's
    cross-grain Joinsets are not among them in MVP" stands: the Grainset drills
    across its children itself, at plan time.

### Week rolls up to nothing coarser

A week can span two months, so a weekly value has no one month, quarter or
year. Cube's roll-up table puts no Week under Month, Quarter or Year, and
MetricFlow: "WEEK does not roll up to the larger granularities cleanly". On the
human's ruling, ADR 0074's grain test now says that a source at Week serves no
requested granularity of Month, Quarter or Year. The repair is in place, and
ADR 0074's *More Information* records it. Rule 3's pick, and so every child that
a drill-across reads, passes that test.

### Amends

- **0033:** in *Decision Outcome*, "when no unit covers the Request, builds a
  LEFT OUTER JOIN tree equi-joined on Keys declared on both interfaces. Missing
  or disagreeing Keys are hard compile errors", as ADRs 0048 and 0073 changed
  it, reads: where the demanded names go to two or more children, the Grainset
  drills across them with `Full` joins on the demand's grouping Dimensions,
  compared NULL-safe, as 0084's rules 4 to 6 give, and children that carry no
  shared guaranteed keyset are no compile error: the Grainset then drills across
  no children (0084's rule 4); the rest of 0033 stands (0084's *Which children
  answer* and *How they are joined*).
- **0048:** in *Grain groups are dropped, and a Grainset emits no implicit
  Unionset*, "a Grainset's `semantic_bitmask` is the **union of its grains'**"
  reads: the union of its grains' and the bit of each Metric that it computes
  (0084's rule 8); the rest of 0048 stands (0084's *A Metric that no child maps
  whole*).
- **0064:** in *Decision Outcome*, "a composite's mask is the union of its
  operands' (ADR 0047), so the root serves the demanded set exactly when some
  unit in its scope does" holds for a Grainset only where one child serves the
  demand: a Grainset also serves a demand that it drills across two or more
  children for, and the mask of a Grainset is the union of its children's and
  the bit of each Metric that it computes (0084's rule 8); in *Decision
  Outcome*, "the child that its drill-across joins
  from (ADR 0084, rule 6)" is the first child of rule 6's left-deep joins; the
  rest of 0064 stands (0084's *Which children answer*).
- **0065:** in *ADR 0057's qualifier rule is scoped: a condition names the
  constituent, a projection names the immediate operand*, "the operand is
  itself a reading surface owning exactly one signature per set bit of its own
  mask" holds for an operand that is a Dataset, a Joinset or a Unionset, and a
  Grainset operand owns a signature only for each Metric that it computes
  (0084's rule 8), so a projection's `Column(<Grainset alias>.<name>)` reads the
  output field that the Grainset's routing fills, or the Grainset's own
  signature for a Metric that it computes; in *A Grainset routes and does not
  read*, "`semantic_signature_ids` is **empty by rule**" reads: it lists one
  signature for each Metric that the Grainset computes and none for any other
  name, "reads no rows, so there is no source for a signature's tree to name"
  reads: the tree of a signature that it lists names the Grainset's own alias
  (0084's rule 8), "`validate()` rejects a Grainset that lists a signature"
  reads as 0084's rule 10 gives, "stays the folded union of its grains" reads:
  stays the folded union of its grains and the bit of each Metric that it
  computes (0084's rule 8), and "a Grainset owns none" and the rejected
  alternative "Giving a Grainset signatures to hang the attribution off
  contradicts the rule above" concern the carrier of the keysets, which stays
  the `keysets` field of `RoleDefinition::Grainset`; in the 0034 entry of *Amends*, "a
  Grainset is a routing surface that owns none" reads: a Grainset owns a
  signature only for each Metric that it computes (0084's rule 8); the rest of
  0065 stands (0084's *How they are joined* and *A Metric that no child maps
  whole*).
- **0071:** in *Each source's subtree reads only its own signature*, "Each
  subtree reads the signatures of the source it reads, and no other source's"
  holds except that a child of a Grainset's drill-across also applies each
  condition that a composition above lowers onto a Measure that it gives, and
  each condition that a Metric the Grainset computes puts on a value that its
  tree reads (0084's rule 9); the rest of 0071 stands (0084's *A Metric that no
  child maps whole*).
- **0073:** in *One decision shape, checked per body*, the table's Grainset row
  reads: who answers, a child or the Grainset over two or more children;
  operand not read, a child that 0084's rule 3 sends no name to; combine,
  pass-through or drill-across; "A Grainset owns no signatures (ADR 0065), so it
  has no "answer as itself" choice" holds only for a name that a child maps
  whole, and in *Through a composed branch*, "a Grainset owns none (ADR 0065)"
  reads: a Grainset owns a signature only for a Metric that it computes (0084's
  rule 8); "ADR 0054's
  cross-grain Joinsets are not among them in MVP" stands (0084's rule 11), and "The rule for cross-grain units
  stays open" no longer holds; "the requested granularity
  comes from the parent's input" holds for a child of a drill-across only where
  that input has one, and otherwise the child takes the granularity of 0084's
  rule 2; the combines in that section, the
  rule in *Three parts* for the form in which each operand gives an item, and
  stage 1 in *The build order* gain `drill-across` as 0084's rule 6 gives it;
  for a Metric whose stored tree reads the frame's own final values (a Metric
  that a Grainset computes, 0084's rule 8), the driver's Shared rule in *Three
  parts* for where a Metric's formula is applied makes a Compute frame, a frame
  kind of its own that no Strategy decides, outside the closed
  set of combines, so the table gains no value for it; in *The build order*,
  stage 5b's "`step_join` with `Inner`" reads, for the subtree of a Metric with
  its own `agg:` that a Grainset computes: a `Full` join on `IsNotDistinctFrom`
  with coalesced keys (0084's rule 8), and, for the subtree of a Metric that a
  Grainset computes, stage 5a's "A request predicate on that Metric alone sits
  directly above it" and the next paragraph's "or stage 5a for a Metric's step
  2" do not hold: a request predicate on that Metric sits above the joins
  (0084's rule 7); in *Two passes*, under
  "A decision is named by its **path** from the root", each formula group's
  frame below a Compute frame has a path of its own, `<grainset>#group<n>`,
  with n counting from 0 (0084's rule 8); in
  *One input for every Strategy call*, the names that a child of a drill-across
  must serve include each name that the tree of a Metric that the Grainset
  computes reads (0084's rule 8) and each Measure under the condition of 0084's
  rule 9, and "a
  Grainset answers only through a grain" holds only where one child answers alone (0084's rule 1); the
  rest of 0073 stands (0084's *How they are joined*, *A Metric that no child
  maps whole* and *Compile's cross-grain Joinsets*).
- **0074:** in *A Grainset*, "A Grainset stands in one of its grains; ADR 0073
  keeps cross-grain units out of MVP" holds only where one child answers alone
  (0084's rule 1), and otherwise the Grainset drills across two or more children, a
  Grainset that is a constituent of another composition only where its parent
  takes final aggregates from it; the pick reads only children that pass the
  grain test for the requested granularity, as repaired in place for Week; "If
  no grain serves the request,
  the Grainset is not a candidate" holds where no child serves it and no
  drill-across does, and no Metric that the Grainset computes (0084's rule 8)
  serves it; in *Consequences*, the Neutral on the planner's subtree of two
  aggregate steps, "ADR 0084's rule 8 gives no Grainset signature to a Metric
  that reads one" reads: the Grainset's signature for a Metric that it computes
  keeps a link to another Metric that it computes (ADR 0088), and a Metric with
  its own `agg:` plans in a subtree of its own, which joins the demand's subtree
  `Full` on `IsNotDistinctFrom` with coalesced keys (0084's rule 8); in *A Unionset*,
  **NULL fill**, "`coverage_of` (ADR 0059) reads only a source's own signatures, so it
  says `NullFill` for each name of a Grainset branch, which owns none (ADR
  0065)" reads: it says `NullFill` for each name of a Grainset branch that the
  Grainset does not compute, and reads the Grainset's own signature for a Metric
  that it computes (0084's rule 8); in *A Metric*, **Its formula**, "Each
  Measure it references becomes the composition's own Measure tree" and "An
  operand's own site for the Metric, with its body and its filter, is not read"
  read, for a Metric that a Grainset computes: its tree reads each name that a
  child serves as that name's final value, `column(<Grainset alias>, <name>)`,
  and the child's own signature for that name is read (0084's rule 8); in *A Metric*, "The two subtrees join with an
  `Inner` join" and "so they hold the same groups, and the join loses none"
  read, for a Metric with its own `agg:` that a Grainset computes: the two
  subtrees join `Full` on `IsNotDistinctFrom` with coalesced keys, because the children can hold different groups (0084's rule 8); in *A Metric*, "A request predicate on such a Metric alone sits
  directly above its step 2 and its cast. The inner join then drops the whole
  output row" reads, for the subtree of a Metric that a Grainset computes: a
  request predicate on that Metric sits above the `Full` joins, so it drops the
  whole output row (0084's rule 7); the rest of 0074 stands (0084's *Which
  children answer*, *A Metric that no child maps whole* and *Week rolls up to
  nothing coarser*).
- **0075:** in *Every output name is the one its parent reads*, the qualifiers
  that a projection writes gain one case: under a Grainset's drill-across, each
  child writes its own alias, and the Grainset's frame writes the Grainset's
  alias for each name that it projects, and, where its reader carries another
  qualifier, its last projection writes the reader's; the rest of 0075 stands
  (0084's *How they are joined*).
- **0077:** in *A constituent serves only the names it maps*, "A demand that
  only two or more children serve together is the cross-grain case that ADR 0073
  keeps out of MVP, and this ADR does not decide it" no longer holds: the
  Grainset drills across its children, and a Metric that it exposes is served
  by a child that maps it whole, or computed by the Grainset over its
  drill-across (0084's rule 8); in **A value whose rows a Joinset repeats**,
  "as ADR 0084 rule 8 refuses a Metric that no child serves" reads: rule 8
  computes such a Metric where the children can compute it together, and
  refuses it only where they cannot; in **A semantics
  with an `expr:`
  body**, "a `filter:` that reads a name of another operand is applied at the
  composition, as its own filter" does not hold on a Grainset's child, which has
  a signature only where it maps every name that the body and the `filter:`
  read (0084's rule 8); the rest of 0077 stands (0084's *How they are joined*
  and *A Metric that no child maps whole*).
- **0080:** in *Only a frame that gives final aggregates buckets*, "a Grainset's
  picked child, and a stand-in" also covers each child of a drill-across, and "A
  frame that passes those final aggregates through, and a Unionset over disjoint
  branches" also covers a Grainset's drill-across; in *A predicate copy goes
  into its operand on every join type* and *A named filter is placed as a
  request predicate*, a predicate over a Dimension and an activated named filter
  go into each chosen child of a drill-across, as 0084's rule 7 says, and the
  child reads its own signature for the filter; the rest of 0080 stands (0084's
  *How they are joined*).
- **0083:** in *Additivity belongs to an aggregate*, rule 5, "and the predicate
  sits directly above them, in stage 5a" does not hold for a Metric that a
  Grainset computes: a request predicate on that Metric sits above the `Full`
  joins of the Grainset's frame, which null-fill every child that lacks a group
  (0084's rule 7); the rest of 0083 stands (0084's *How they are joined*).

### Consequences

- Good, because a demand whose names two or more children serve together is
  answered, for any number of children, exactly for every additivity, with no
  marker, for each group that any chosen child holds, wherever the
  Grainset's demand takes final aggregates (rule 1).
- Good, because a Metric over the Measures of two children has a plan, and the
  stored signature holds only the formula and its filter. A Metric that one child
  maps whole is served by that child, under its own filters.
- Good, because every join compares final aggregates, which ADR 0080 allows: "a
  join compares a bucket only between final aggregates".
- Bad, because a request that reads a Dimension that a chosen child does not map
  has no plan: `select date at Month, channel, clicks, budget` finds no
  candidate where only `spend_daily` maps `channel`. A join of rows would repeat
  the coarser value in each group.
- Bad, because a Grainset whose children carry no guaranteed keyset serves no
  such demand. No Grainset of the oracle corpus declares one today.
- Bad, because the manifest keeps compile's cross-grain Joinsets, which nothing
  reads, and their condition still compares a day with a month.
- Bad, because ADR 0074's grain test is edited in place, a departure from this
  repo's route, where the most recent ADR governs and the earlier text stays.
  The human ruled it, as ADR 0058 records for ADR 0054's threshold.
- Bad, a trade-off: one child could answer a request alone, but the request
  reads two children and joins them where they rank first for different names;
  gain: a name's value does not depend on the other names in the request.
- Bad, a trade-off: one row can hold values from two reports, and a ratio mixes
  them where the reports disagree; gain: each value comes from the most
  complete report.
- Bad, a trade-off: a Metric that one child maps whole is answered by that
  child, even where other children rank first for the Measures it reads. On
  `marketing_all_spend`'s `ads`, `campaign, cpc` gives 30 / 10 = 3.0 from
  `by_gender`, while `campaign, cost, clicks` gives 40 and 20; gain: a Grainset
  owns a signature only where no child can compute the Metric, and the child's
  own filter governs.
- Bad, a trade-off: a nested Grainset whose children cannot compute a Metric
  leaves it to the composition above, and a request for it fails at plan, not at
  compile; gain: the semantic model's other names still plan (`campaign_clicks` over
  `spend_eu`: `date@month, clicks` gives (2026-03-01, 2), and `date@month,
  web_click_days` fails at plan).
- Bad, a trade-off: on an engine without `IS NOT DISTINCT FROM`, the `OR` form
  can stop a hash join; gain: correct NULL groups on every engine. The cost is
  not measured.
- Neutral: the rows of the oracle cases for a NULL group at a tie and for a NULL
  campaign group change, and so do the plan-shape tests that pin `Left` or a
  predicate inside its child.
- Neutral: the glossary term *Preserved child* retires. A group that only one
  child holds, and a NULL group, give one row in the answer.
- Neutral: the case that refused a Grainset Metric that no child maps whole now
  compiles where the serving children share a guaranteed keyset (rule 8).
- Neutral: compile refuses a Metric with no `agg:` where a child that gives a
  value that it reads cannot form the part key, with
  `GrainsetMetricPartKeyUnformed` (rule 8). No relation over the children
  gives its value: two sets of source rows that give the same children give
  it different values.
- Compile changes: a Grainset child's signature needs every name that the body
  and the `filter:` read (rule 8). A public Grainset's Metric that its children
  cannot compute is refused with `GrainsetMetricReadsUnmapped`,
  `GrainsetMetricFilterUnmapped`, `GrainsetMetricPartKeyUnformed` or
  `GrainsetMetricWithoutKeyset` (rule 8);
  `UnservedComposedSemantics` stays for an exposed name that no child, at any
  depth, has a signature for and that the Grainset does not compute.
  `validate()` accepts a Grainset signature only for a Metric that no child maps whole
  (rule 10).

## More Information

Decided 2026-10-06. The human: "I
think all this we need to keep runtime planner decision and respective
strategy - to build up chains of relations out of grainset childs contract. In
future we could go in ofline mode - to decide how we want to pre-compile this
rulings"; "keysets is somethign that makes cross-grain joins possible, otherwise
we cannot serve partial results"; and, on joining final aggregates, "we need
rollup in any case to requested grain which makes it aggregate, or at least
partial aggregate".

The product sources quoted above come from the research on MetricFlow and Cube,
and each plan shape compared here was run as SQL.

Rule 3's sentence on a Metric's `filter:` follows the decision of 2026-10-17, and rule 8's
sentence on a row-level Metric with no `agg:` follows the decisions of
2026-10-18 there, which read the research on
MetricFlow, Cube, Looker, Malloy and Calcite. Rules 8 to 10 follow the decisions recorded. The decision recorded
that a Metric is
computed inside one child holds for a Metric that a child maps whole.

The `Full` joins, the NULL-safe key, the pick of the rank-first child for each
name and the placement of a predicate on a Measure follow the decision.
