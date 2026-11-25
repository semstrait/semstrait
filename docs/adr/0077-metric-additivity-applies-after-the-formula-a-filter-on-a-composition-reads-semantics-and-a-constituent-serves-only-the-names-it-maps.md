---
number: 77
title: Metric additivity applies after the formula, a filter on a composition reads semantics, and a constituent serves only the names it maps
status: accepted
date: 2026-10-04
source:
- CONTEXT.md
- docs/adr/0027-closed-explicit-authoring-surfaces.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md
- docs/adr/0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md
- docs/adr/0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md
- '`seal_leaf` and `unmapped_exposed_semantics` in `crates/semstrait-manifest/src/compile/assemble.rs`'
- '`join_projection` and `seal_composed_projection` in `crates/semstrait-manifest/src/compile/expressions/projection.rs`'
- '`filter_view` in `crates/semstrait-manifest/src/compile/expressions/bodies.rs`'
- '`ResolvedSource` in `crates/semstrait-manifest/src/compile/sources.rs`'
- '`SemiAdditivityStrategy` in `crates/semstrait-model/src/model/types/entities/measure.rs`'
- the compile-seam tests `a_joinset_projects_a_bodied_metric_and_a_named_filter_from_the_preserved_side` and `a_composed_measure_conjoins_its_own_view_with_its_constituents` in `crates/semstrait-manifest/tests/composed_signatures.rs`
- the corpus models `test_data/acme_eu_ad_platform_v3.yaml` and `test_data/e2e_full_coverage.yaml`
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18 and 2026-10-19
links:
- target: 30
  kind: amends
- target: 53
  kind: amends
- target: 65
  kind: amends
- target: 71
  kind: amends
- target: 74
  kind: amends
- target: 73
  kind: amends
- target: 59
  kind: amends
- target: 76
  kind: amends
- target: 80
  kind: amendedby
- target: 83
  kind: amendedby
- target: 84
  kind: amendedby
---

# Metric additivity applies after the formula, a filter on a composition reads semantics, and a constituent serves only the names it maps

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0080](0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md): in *Ruled elsewhere, not here*, where the planner applies an activated named filter, and a request predicate on a semi-additive Measure or Metric, is 0080's *A named filter is placed as a request predicate* and *Every filter reads the value before bucketing*; the rest of 0077 stands.
> - [ADR 0083](0083-a-metric-declares-no-additivity.md): *A Metric's declared additivity applies to its own value, after its formula*, and the first bullet of *Decision Outcome*'s "In short", no longer hold; only "Each node reads its function's additivity, narrowed by the effective additivity of the Measure it came from" stays; in *Ruled elsewhere, not here*, the sentence that begins "That marker is read by an aggregate's condition, below the formula" and the sentence after it have no subject, and a node's marker partitions by the `<grouping>` that ADR 0076 gives it, which no Metric's additivity extends; in *Consequences*, `SemiAdditiveMetricInlined` is not a compile refusal; in *Amends*, the entries for 0030 and 0074 no longer hold where they say that "a Metric's declared additivity applies to the Metric's own value, after its formula" and that "a semi-additive Metric's semi-additive strategy runs on the Metric's value"; in *Ruled elsewhere, not here*, "the effective additivity on each `Aggregate` node" is not a manifest shape: each node reads it through its link (0083's rule 3); the rest of 0077 stands (0083's *Additivity belongs to an aggregate*).
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *A constituent serves only the names it maps*, "A demand that only two or more children serve together is the cross-grain case that ADR 0073 keeps out of MVP, and this ADR does not decide it" no longer holds: the Grainset drills across its children, and a Metric that it exposes is served by a child that maps it whole, or computed by the Grainset over its drill-across (0084's rule 8); in **A value whose rows a Joinset repeats**, "as ADR 0084 rule 8 refuses a Metric that no child serves" reads: rule 8 computes such a Metric where the children can compute it together, and refuses it only where they cannot; in **A semantics with an `expr:` body**, "a `filter:` that reads a name of another operand is applied at the composition, as its own filter" does not hold on a Grainset's child, which has a signature only where it maps every name that the body and the `filter:` read (0084's rule 8); the rest of 0077 stands (0084's *How they are joined* and *A Metric that no child maps whole*).

## Context and Problem Statement

ADR 0074 rules what the planner reads.
A separate decision rules what compile
writes so that the planner can read it. Three of those changes need a decision
that no ADR states:

1. **How a Metric's declared additivity is read.** ADR 0074 narrows each
   `Aggregate` node of a Metric's tree by it: "A semi-additive Metric gives each
   of its nodes its axes and strategy" (*Additivity decides whether a partial
   aggregate may be combined again*). So a semi-additive Measure inside a
   semi-additive Metric gives one node two semi-additive strategies, in no
   defined order.
2. **What a filter on a composition reads.** ADR 0074's table reads `Coalesce`
   for a name that both operands of a Joinset map, and ADR 0074 reads a
   Dimension that both operands map "on each operand", never through that
   `Coalesce` (*A Joinset*). But compile lowers a composition's own filter over
   the columns that the composition projects, so with the table that filter
   reads the coalesced value. A named filter on a composition is the column
   that an operand projects (ADR 0065), so it can be applied only above that
   operand.
3. **How a constituent maps a name.** ADR 0074 rules that a constituent serves
   only the names it maps (*A Unionset*, **NULL fill**). Today every constituent
   realises the whole interface (`seal_leaf`). An explicit `semantic_mapping`
   must list every exposed name, and `auto` maps every name to a column of that
   name without a schema read. The refusal says why: "Whether an unmapped
   exposed name falls back to its identity column or is null-filled is a rule
   no ADR states" (`unmapped_exposed_semantics`).

## Considered Options

- A Metric's additivity applies after its formula, a filter on a composition
  reads semantics, and a constituent serves only the names it maps (chosen).
- **Narrowing each node by the Metric's declaration** (ADR 0074 as written). A
  Metric declared `non` makes every node non-additive, so an exact
  re-aggregation of two sums raises `LossyReaggregation`. A semi-additive
  Measure inside a semi-additive Metric gets two semi-additive strategies on one
  node.
- **Refusing two semi-additive declarations on one node**
  (`ConflictingSemiAdditivity`). The human ruled that each declaration applies
  to its own value, so the two never meet on one node.
- **A carrier on the signature for a semi-additive Metric that another Metric
  inlines**: the root node of each inlined copy, with its semi-additive
  strategy. The planner reads trees, and it keeps node ids for diagnostics only
  (ADR 0071), so it would not read the carrier.
- **`Coalesce` for a named filter that both operands map** (ADR 0074's table as
  written). The human ruled that a named filter reads semantics, and that
  `Coalesce` is for the output.
- **`And` of the operands' finished named-filter columns.** The planner could
  then apply the filter only above the operands that output the columns, not
  where its semantics are.
- **One stored copy over the coalesced value, with ADR 0074's two copies for a
  request predicate only.** The stored filter then reads the coalesced value
  that ADR 0074 forbids, and the planner cannot split it into copies: it cannot
  tell compile's `Coalesce` from an author's `coalesce(a, b)` over two names
  that each one side maps.
- **One copy for each constituent that maps the name, in a nested Joinset.** A
  Joinset is binary (ADR 0045), and the human ruled that each level has two
  operands, while the lower level gives its own value.
- **An omitted explicit entry falls back to its identity column.** A
  constituent then claims a column that the author did not list.
- **`auto` maps every exposed name on a constituent** (ADR 0053's "`auto` keeps
  its meaning — 1:1 to a same-named physical column — in both", in *`semantic_mapping`
  stays Dataset-only; only its key-space moves*, read without a schema). With
  ADR 0074's table, every name of an `auto` Joinset then reads `Coalesce` over a
  column that one table does not have. A catalog source refuses it, and a path
  source fails at the read.
- **`auto` maps every name on a constituent that has no reported schema**,
  unchecked. DataFusion fails at the read in place of a compile refusal.
- **Null fill in an Implicit Unionset.** ADR 0049 rules "**No null fill,
  ever.**" for it (*The implicit Unionset is a narrowing of the authored one*),
  and ADR 0058's carry-over relies on its identical branches. The human ruled
  that it stays a Unionset without null fill.
- **An Implicit Unionset that maps a name only where every one of its physical
  sources has the column.** A name then leaves the Dataset in silence.
- **A body that reads a name its constituent does not map is refused**
  (`UnexposedReference`). It refuses models whose composed answer is defined.
- **The semantics never has a signature where its body reads an unmapped
  name.** On
  an explicit Unionset, a `case` whose taken branch reads only mapped names then
  gives NULL. In `acme_eu_ad_platform_v3.yaml`, `market` on Bing does.
- **A name of a declared join basis that one constituent does not map is
  dropped from the condition.** The join then compares fewer names than the
  author declared.

## Decision Outcome

In short:

- A Metric's declared additivity is a fact about the Metric's own value. It
  applies after the formula, and it narrows no node.
- A filter on a composition reads semantics. Compile adds no `Coalesce` to a
  filter. Where both operands of a Joinset map a Dimension that a filter reads,
  each operand gets its own copy of the filter.
- A constituent serves only the names it maps, and null fill happens on an
  explicit Unionset only.

### A Metric's declared additivity applies to its own value, after its formula

A Metric's tree is its formula over `Aggregate` nodes (ADR 0071). Each node
reads its function's additivity, narrowed by the effective additivity of the
Measure it came from. The Metric's own declaration does not narrow the nodes. It
applies to the Metric's value:

1. Each node is planned as ADR 0074 plans a Measure, with its Measure's
   semi-additive strategy where it has one.
2. The formula runs over those aggregates.
3. Where the request drops or coarsens what a semi-additive Metric's strategy
   reads, as ADR 0074 says for a Measure (*A semi-additive aggregate's
   strategy*), the Metric's semi-additive strategy runs across the Metric's
   values.

**The two semi-additive strategies nest, so their order is defined.** The
Measure's semi-additive strategy runs inside the Metric's value, and the
Metric's runs on that value.

**A Metric's semi-additive strategy extends the demand of its nodes.** The
Dimensions that it reads are added to the names that its nodes' subtree must
serve, as ADR 0074 does for an axis that the request does not select. Where a
node's own semi-additive strategy reads the same Dimension, that Dimension stays
grouped below the formula, so only the Metric's semi-additive strategy applies
across it.

**A Metric declared `full` or `non` changes no plan.** The planner applies the
formula above the last aggregate and never combines a Metric's values, except
under a semi-additive strategy. Example: `ctr = clicks / impressions`, declared
`non`, over a `Timeseries` Dataset, by month. `clicks` and `impressions`
re-aggregate as sums, exactly, so no `LossyReaggregation` is raised.

**A semi-additive Metric that another Metric inlines is refused.** In
`B = A * 2`, with `A` semi-additive, `A`'s strategy must run on `A`'s value
inside `B`. The inlined copy of `A` is an ordinary subtree of `B`'s tree, and
nothing marks where it starts. A body can differ per site (ADR 0027), so compile
refuses it per semantic source:
`CompileError::SemiAdditiveMetricInlined { metric, inlined, semantic_source }`.

### A filter on a composition reads semantics, never a `Coalesce`

**A named filter is lowered on the composition.** A named filter's signature on
a composed semantic source is its body lowered on that semantic source, as ADR
0074 lowers a composed Metric (*A Metric*), with the copies below. It is no
longer the column that an operand projects.

**Each operand of a Joinset gets its own copy of a filter.** A filter that
compile lowers on a Joinset may read a Dimension that both operands map. That
filter is a composition's own filter for a Measure (ADR 0065 D2) or for a Metric
(ADR 0074, *A Metric*), or a named filter's body. Compile writes the whole
filter once for each operand: in the first copy every such Dimension reads
`from`, and in the second it reads `to`. A name that only one operand maps reads
that operand in both copies. The copies are joined with `And`. These are ADR
0074's two copies, written by compile into the stored tree.

For example, on the Joinset `sales` (`orders ⋈ customers`), where both operands
map `region`, `region = 'eu'` is stored as:

```text
And( Eq(Column(sales_orders.region), 'eu') , Eq(Column(sales_customers.region), 'eu') )
     └ the copy for from                     └ the copy for to
```

A filter that reads no such Dimension has one copy. A copy reads its immediate
operand (ADR 0065, *ADR 0057's qualifier rule is scoped: a condition names the
constituent, a projection names the immediate operand*). In a nested Joinset,
the copy for an operand that is itself a Joinset reads that operand's own value,
which may be the `Coalesce` that ADR 0074's table gives inside that operand. So
no stored filter reads the coalesced value of its own composition.

**On a Unionset, a Measure's own filter does not change.** A Unionset has no
`Coalesce`, and each branch's own signature already filters the rows it stacks
(ADR 0065 D2). A named filter on a Unionset is lowered over the union's
columns, as above.

**ADR 0074's `Coalesce` table is for a value only**: a Dimension, and the
argument of a Measure's aggregate. It never applies to a named filter.

Compile writes the copies, not the planner, because the manifest stores
resolved trees, and the planner cannot tell compile's `Coalesce` from an
author's `coalesce(a, b)`. A request predicate is not stored. The planner
copies it as ADR 0074 says.

### A constituent serves only the names it maps

A constituent has a signature for a name only where it maps the name. The
composition then reads the name from the operands that have one (ADR 0065, ADR
0074).

- **An explicit `semantic_mapping`.** On a constituent, an exposed name with no
  entry is not mapped. A Dataset that is a root data kind (ADR 0053) keeps
  today's refusal, because nothing above it null-fills. Where no operand or
  child of an explicit composition maps a name that its interface exposes, and
  a Joinset does not lower its `expr:` body on itself (*A semantics with an
  `expr:` body*, below), compile refuses with
  `CompileError::UnservedComposedSemantics { semantics, semantic_source }`, in
  place of today's reused `TypeInferenceFailure`. A sub-composition that compile
  enumerates (ADR 0054) serves the names that its operands serve.
- **The keysets of a nested composition.** Under interface-first, a nested
  composition guarantees each keyset of the root's interface whose every member
  it serves, checked against its own boundary (ADR 0048); a nested Joinset also
  derives its own, and a nested Grainset holds them on its `keysets` field. A
  keyset whose member a unit does not serve does not reach that unit, since a
  keyset is a claim about the unit's rows.
- **`auto` with an observed schema.** On a constituent, `auto` maps a name with
  no `expr:` body where the schema that the catalog observes (ADR 0046) has a
  column of that name. Compile has that schema before it seals a semantic
  source (`ResolvedSource`). A root Dataset keeps today's rule: every exposed
  name maps, and a column that its table does not have is
  `CompileError::ColumnNotInSourceSchema`.
- **`auto` with no observed schema**: a path source (ADR 0061), or a table
  compiled with no catalog. On a constituent, compile refuses `auto` with
  `CompileError::AutoMappingWithoutSchema { dataset }`, and the author maps
  explicitly. A root Dataset keeps `auto` as today. Once compile observes the
  columns of such a source, the rule above applies to it.
- **A semantics with an `expr:` body**, and a Measure's or Metric's `filter:`,
  read the names they reference on the constituent. On a constituent of an
  explicit Unionset, a name that the constituent does not map reads as a typed
  NULL, which is the value the Unionset gives for that name on that branch. On
  every other constituent, the semantics has no signature there unless the
  constituent maps every name that its body reads, and a `filter:` that reads a
  name of another operand is applied at the composition, as its own filter (ADR
  0071). Where no operand of a Joinset maps every name that the body reads,
  compile lowers the semantics on the Joinset itself, a Measure and a Metric
  alike. Each operand then gives the names of the body that it maps, so the
  planner reads both operands and builds the join. A Measure's body holds, for
  each name, the tree of the Joinset's own signature for it, with no link, and a
  Metric's body links those signatures (ADR 0088, rules 5 and 14). On
  `orders_j ⋈ customers_j`, `vip_orders = count(CASE WHEN segment = 'vip' THEN
  order_id END)` reads
  `segment` from `customers_j` and `order_id` from `orders_j`, and by region it
  gives eu 2 and us 1. For a name that the interface does not expose, ADR 0065's
  `UnexposedReference` still refuses.
- **A value whose rows a Joinset repeats.** A Measure that compile lowers on a
  Joinset reads the joined rows. Where each value leaf of a result of its
  argument reads an operand whose rows the declared cardinality repeats, the
  planner reads one row per row identity of each such Dataset for that result
  (ADR 0074, *A Joinset*, **Fan-out**), and refuses a result that would read
  two `row_number` markers (ADR 0076, **Where the marker has no place**, rule
  3). A Metric whose `agg:` aggregates rows is
  lowered as a Measure (ADR 0074, *A Metric*), so it reads the same way. A leaf
  of a `case` condition, or of the Measure's `filter:`, is a condition, not a
  value. The declared cardinality decides, on `on: cross` too: `many_to_one`
  and `one_to_many` repeat the rows of the "one" side, `many_to_many` repeats
  the rows of both sides, and `one_to_one` repeats none. An omitted cardinality
  resolves to `many_to_many`. A value that reads no column, as the `1` of
  `count(1)`, reads no operand: it counts the joined rows. `min`, `max` and
  `count distinct` read every joined row, because repeated rows do not change
  them.

  The operands of a value are found by a walk. Each value leaf follows its
  qualifier, which names the immediate operand (ADR 0065), down to a Dataset.
  At each Joinset on that path, the walk reads that Joinset's cardinality for
  the operand that it enters, and the leaf reaches a repeated Dataset where the
  cardinality repeats that operand. On a Unionset, the walk enters each branch.
  Where the path reaches a Grainset operand, compile refuses the Measure with
  `UnservedComposedSemantics` only where no single child of that Grainset maps
  every name that the Measure's value and condition leaves read through it, as
  ADR 0084 rule 8 refuses a Metric that no child serves. The check reads every
  depth: in an operand that is a Unionset, each branch that is a Grainset must
  map, through one child, those of the names that its mask holds, and the check
  applies again at each composition below. Otherwise the walk stops at the
  Grainset at compile; in the plan it continues through the child that
  answers, as through any operand: the child that ADR 0074's pick takes at
  the granularity that the Grainset declares. No drill-across answers there,
  because the Joinset whose rows the walk follows builds its join and takes
  rows or partial aggregates from the Grainset (ADR 0074, *A Grainset*). A
  `Coalesce` leaf, for a name that both operands map
  (ADR 0074, *A Joinset*), walks both chains. Only value leaves walk.

  `orders_j ⋈ customers_j` is many-to-one, so it repeats the rows of
  `customers_j`: c1 meets o1, o2 and o3. `vip_orders` reads `segment` only in
  its `case` condition, and its value `order_id` from `orders_j`, so it reads
  the joined rows, and by region it gives eu 2 and us 1. `orders ⋈ order_lines`
  is one-to-many on `order_id`, with the lines (l1, o1, `gift`), (l2, o1,
  `gift`), (l3, o2, `std`), (l4, o3, `gift`) and (l5, o4, `std`), so it repeats
  the rows of `orders`. The Measure `gift_amount = sum(CASE WHEN kind = 'gift'
  THEN amount END)` reads its value `amount` from `orders`, so it reads o1 once:
  eu gives 100, where the joined rows would give 200.

  The planner reads the same walk for each Measure link of a Metric's formula
  push-down, at plan time (ADR 0074, *A Metric*). A link whose Measure does not
  read the joined rows as they are makes the formula take its parts. On
  `orders_j ⋈ customers_j`, `vip_weighted_revenue = CASE WHEN segment = 'vip'
  THEN revenue * 2 ELSE revenue END` reads `segment` in a condition, and the
  argument of `revenue` from `orders_j`, so the formula push-down applies, and
  by region it gives eu 300 and us 410.
- **A declared join basis.** Each constituent that a relationship joins must
  map each name of the join basis that the relationship declares
  (`on: [names]`), else compile refuses
  with `CompileError::JoinKeyNotMapped { relationship, key, semantic_source }`.
  A Dataset maps a name that its mask holds. A Grainset maps the basis only
  where one child maps every name of it. A Unionset maps it where its mask holds
  each name and each branch that is a Grainset maps, through one child, the
  names of the basis that its mask holds; the check applies again at each
  composition below.
- **Null fill happens on an explicit Unionset only.** The planner writes a typed
  NULL for a constituent of an explicit Unionset that does not map a name (ADR
  0074, *A Unionset*, **NULL fill**). A Joinset reads the name from the other
  operand.
- **A Grainset's child serves only the names it maps**, as every constituent
  does. The Grainset routes as ADR 0064 and ADR 0074 say (*A Grainset*), and
  this ADR does not change that rule. A demand that only two or more children
  serve together is the cross-grain case that ADR 0073 keeps out of MVP, and
  this ADR does not decide it.
- **Each Unionset has a null-fill setting, on its source role in the manifest.**
  Compile sets it on for an explicit Unionset, and off for an Implicit Unionset
  (ADR 0049). With it off, the Implicit Unionset maps the names that its
  Dataset maps, by the rule for that Dataset: a root Dataset maps every exposed
  name, and a constituent maps by the rules above, for any of its physical
  sources. Every reduction leaf must then have each mapped column: a leaf that
  does not is refused (`ColumnNotInSourceSchema` for a table). `validate()` refuses an Implicit
  Unionset whose reduction leaf has no signature for a name that a sibling has.

For example, in `acme_eu_ad_platform_v3.yaml` the Bing constituent of
the explicit Unionset has no `country` entry. So Bing has no signature for
`country`, and the Unionset null-fills `country` on Bing's rows. The Dimension
`market` reads `country` only in its Facebook branch, so on Bing it reads a
typed NULL there and keeps its value from `campaign`. In `e2e_full_coverage.yaml`,
each constituent of the Joinset `txn_details` lists only its own names (in the
older `column_mapping:` key, which the corpus migration renames).

### What retires

- `first` and `last` leave `SemiAdditivityStrategy`. They are the same as
  `earliest` and `latest` (ADR 0074, *A semi-additive aggregate's strategy*), so
  the semantic model drops them. A semantic model whose semi-additive `strategy`
  is `first` or `last` fails to parse with the ordinary unknown-variant error,
  with no second spelling and no dedicated message, as ADR 0065 rejects the
  retired `filters:` key (*One `filter:` predicate, and no
  `AggregationFilter`*).

### Ruled elsewhere, not here

- How the planner writes a Measure's semi-additive strategy at a node is ADR
  0076's *A marker gives each aggregate its rows*, with its `<grouping>` read as
  the node grouping that this ADR extends. That marker is read by an aggregate's
  condition, below the formula, so it does not write a Metric's semi-additive
  strategy on the Metric's value. The plan shape for that is not ruled here. The removal of
  `average`, which
  Dimensions a semi-additive strategy reads and how it orders them (*`axes` and
  `by`*), and aggregates that need different row sets are ADR 0076's too.
- Where the planner applies an activated named filter, and a request predicate
  on a semi-additive Measure or Metric, are not ruled here.
- Observing the columns of a path source is not ruled here.
- The manifest shapes that ADR 0074 already decides are recorded in the
  resolution: a composed Metric's signature and filter, the effective
  additivity on each `Aggregate` node, and the checks of a semi-additive axis.

### Amends

- **0030:** in its *Decision Outcome*, as ADR 0074 amended it, a Metric's
  aggregate node no longer reads "the Metric's own declared additivity": a
  Metric's declared additivity applies to the Metric's own value, after its
  formula, and a node reads its function's additivity, narrowed by the effective
  additivity of the Measure it came from (0077's *A Metric's declared additivity
  applies to its own value, after its formula*).
- **0053:** in *`semantic_mapping` stays Dataset-only; only its key-space
  moves*, "`auto` keeps its meaning — 1:1 to a same-named physical column — in
  both" holds for a root Dataset only; on a constituent, `auto` and an explicit
  mapping map a name as 0077 rules (0077's *A constituent serves only the names
  it maps*).
- **0059:** in *Coverage is read against a composition, and null fill is what
  forces it*, a semantics is null-filled at a constituent only on an explicit
  Unionset; a Joinset reads a name that one operand does not map from the other
  operand (0077's *A constituent serves only the names it maps*).
- **0065:** in *What a tree contains, and who builds it*, a body or a `filter:`
  that reads a name its constituent exposes and does not map reads a typed NULL
  on an explicit Unionset, and elsewhere the semantics has no signature on that
  constituent; in *A composed source's signature names the constituent's
  projected column*, a named filter's signature on a composed semantic source is
  its body lowered on that semantic source, not "the same qualified column with
  no root above it"; in the same section, "A measure on a composed source keeps
  its aggregate root over the qualified column" holds but for a semantics whose
  body reads names that no operand of a Joinset maps alone: compile lowers that
  body on the Joinset itself (0077's *A constituent serves only the names it
  maps*); in *ADR 0057's qualifier rule is scoped: a condition names
  the constituent, a projection names the immediate operand*, *A composed
  Measure's filter is the conjunction of the two views (D2)*, *A composition's
  own `filter:` reads only what it exposes (D4)* and *One `filter:` predicate,
  and no `AggregationFilter`*, the composition's own view, where it reads a
  Dimension that both operands of a Joinset map, is written as 0077's two
  copies, and each copy reads its operand's column, not the column that the
  composition projects (0077's *A filter on a composition reads semantics, never
  a `Coalesce`* and *A constituent serves only the names it maps*).
- **0071:** in *Placement is read from the role* and *A tree gives the
  computation, and the role gives the placement*, a composed named filter is its
  body lowered on the composition, not "the column the composition carries for
  it", and "No advertised bit is null-filled under interface-first authoring" no
  longer holds: a constituent of an explicit Unionset that does not map a name
  is null-filled for it; in *A predicate is applied where its semantics carry
  their meaning*, the tree of an activated named filter on a composed semantic
  source is that body (0077's *A filter on a composition reads semantics, never
  a `Coalesce`* and *A constituent serves only the names it maps*).
- **0073:** in *One decision shape, checked per body*, "every grain of one
  Grainset maps the same interface" no longer holds, because a child serves only
  the names it maps; in *Details fixed with the seam*, "No model that can be authored today
  produces a null-filled bit" no longer holds (0077's *A constituent serves only
  the names it maps*).
- **0074:** in *Additivity decides whether a partial aggregate may be combined
  again*, a Metric's own declared additivity no longer narrows each `Aggregate`
  node of its tree, and "A semi-additive Metric gives each of its nodes its axes
  and strategy" is replaced by 0077's *A Metric's declared additivity applies to
  its own value, after its formula*; in *A semi-additive aggregate's strategy*,
  a semi-additive Metric's semi-additive strategy runs on the Metric's value,
  and the Measure's runs inside it, and the planner no longer reads `first` and
  `last` as `earliest` and `latest`, because they leave
  `SemiAdditivityStrategy`; in *A Joinset*, the `Coalesce` table and the two
  copies of a filter read as 0077's *A filter on a composition reads semantics,
  never a `Coalesce`* gives (0077's *A Metric's declared additivity applies to
  its own value, after its formula*, *A filter on a composition reads semantics,
  never a `Coalesce`* and *What retires*).
- **0076:** in *`axes` and `by`*, "ADR 0074 reads `first` and `last` as
  `earliest` and `latest`" no longer holds: they leave `SemiAdditivityStrategy`,
  and their removal, which 0076 leaves open, is 0077's *What retires*.

### Consequences

- Compile refusals: `UnservedComposedSemantics`, `AutoMappingWithoutSchema`,
  `JoinKeyNotMapped` and `SemiAdditiveMetricInlined`. The refusal
  `unmapped_exposed_semantics` stays for a root Dataset only.
- The manifest gains the null-fill setting on a Unionset's source role, and
  `validate()` checks it.
- A model may author an explicit Unionset whose constituents map different
  names, and a Joinset whose constituents list only their own names, as the
  corpus already does.

## More Information

Decided 2026-10-04, and on 2026-10-17 for a semantics
on a Joinset whose `expr:` reads names that no operand maps alone, and on
2026-10-18 for a Grainset constituent and the keysets of a nested composition.
The reading of a value whose rows a Joinset repeats, and its walk, follow the
decisions of 2026-10-18, which read the research on
MetricFlow, Cube, Looker, Malloy and Calcite, and
an oracle on DuckDB and DataFusion 55.
