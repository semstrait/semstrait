---
number: 71
title: A predicate is applied where its semantics carry their meaning, and each source's subtree reads only its own signature
status: accepted
date: 2026-09-29
source:
- CONTEXT.md
- docs/adr/0033-composite-plan-shape.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0063-the-planner-takes-a-request-context-not-a-session.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- crates/semstrait-planner/src/frame.rs
- crates/semstrait-planner/src/predicate.rs
- the compile-seam test `the_composed_filter_names_the_semantics_and_the_operands_own_filter_its_column` in `crates/semstrait-manifest/tests/composed_signatures.rs`
- the human's rulings of 2026-10-17
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18
links:
- target: 37
  kind: amends
- target: 62
  kind: amends
- target: 65
  kind: amends
- target: 74
  kind: amendedby
- target: 75
  kind: amendedby
- target: 76
  kind: amendedby
- target: 77
  kind: amendedby
- target: 80
  kind: amendedby
- target: 84
  kind: amendedby
- target: 88
  kind: amendedby
---

# A predicate is applied where its semantics carry their meaning, and each source's subtree reads only its own signature

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): a subtree's aggregation may take more than one aggregate step (a distinct-rows aggregate below a deduplicated aggregate, an inner aggregate below a semi-additive strategy); a Unionset in mode `all` re-aggregates above the union only where its branches give partial aggregates; once a constituent serves only the names it maps, a `Left` or `Right` join reads a name that only the other side maps from its null-filled side, and a name that both map through `Coalesce`; a composed Metric's tree is its formula once the manifest lowers it, with the composition's own filter joined onto each inlined aggregate as a Dataset Metric's is; no condition is coalesced, and a name that both operands of a Joinset map reads no operand's condition, only the composition's own filter, in which a Dimension that both operands map is read in two copies of the whole filter, one on each operand, joined with `And`, and so is a request predicate over such a Dimension, each copy placed as 0071 places a predicate; how a subtree re-aggregates partials is 0074's.
> - [ADR 0075](0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md): in *A tree gives the computation, and the role gives the placement*, how a qualified leaf binds to an output field, which 0071 leaves open, is 0075's *Every output name is the one its parent reads* and *Names are unique, and the planner's own names are reserved*; in *A predicate is applied where its semantics carry their meaning*, a request predicate at a join is a `FilterNode` above the `JoinNode` and never merges into `post_join_filter` (0075's *A request predicate at a join stays above it*); the rest of 0071 stands.
> - [ADR 0076](0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md): in *A tree gives the computation, and the role gives the placement*, as ADR 0074 changed it, a subtree's aggregation is one aggregate step, and two in a Metric's subtree of two aggregate steps (ADR 0074, *A Metric*), and a projection below it may add a marker that gives an aggregate its rows; in *Aggregation-level conditions*, as ADR 0074 changed it, an aggregate that reads a marker has its condition joined with `And` onto `<marker> = 1`; the rest of 0071 stands (0076's *A marker gives each aggregate its rows*).
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *Placement is read from the role* and *A tree gives the computation, and the role gives the placement*, a composed named filter is its body lowered on the composition, not "the column the composition carries for it", and "No advertised bit is null-filled under interface-first authoring" no longer holds: a constituent of an explicit Unionset that does not map a name is null-filled for it; in *A predicate is applied where its semantics carry their meaning*, the tree of an activated named filter on a composed semantic source is that body (0077's *A filter on a composition reads semantics, never a `Coalesce`* and *A constituent serves only the names it maps*).
> - [ADR 0080](0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md): in *Decision Outcome* and in the table of *A predicate is applied where its semantics carry their meaning*, "It never goes below a join into a side the join null-fills" holds only for a predicate that reads no Dimension that both operands of a Joinset map: each copy of a request predicate over such a Dimension that reads only one operand's names goes into that operand on every join type; in the same table, an activated named filter on a composed semantic source is placed copy by copy as a request predicate on Dimensions, not as one boolean column applied at the lowest node where it is attached; the rest of 0071 stands (0080's *A predicate copy goes into its operand on every join type* and *A named filter is placed as a request predicate*).
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Each source's subtree reads only its own signature*, "Each subtree reads the signatures of the source it reads, and no other source's" holds except that a child of a Grainset's drill-across also applies each condition that a composition above lowers onto a Measure that it gives, and each condition that a Metric the Grainset computes puts on a value that its tree reads (0084's rule 9); the rest of 0071 stands (0084's *A Metric that no child maps whole*).
> - [ADR 0088](0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md): in *Placement is read from the role*, "`ExprLayer` stays on every stored node" becomes: `ExprLayer` stays on every entry (ADR 0088, rule 6). In *A tree gives the computation, and the role gives the placement*, "Every signature tree is materialised once per request with `ManifestExpressions::reassemble`" holds, and `reassemble` reads each link as ADR 0088 rule 15 says. In *Aggregation-level conditions*, "A condition already on an inlined `Aggregate` node" becomes: the filter of a Measure or a Metric that a Metric links is joined where the link is read (ADR 0088, rule 15), never dropped and never applied twice; and "Compile applies the same rule when another Metric inlines it" becomes: reading applies the same rule where another Metric links it (ADR 0088, rule 15). The rest of 0071 stands.

## Context and Problem Statement

The resolver has fixed a scope root, and the root's Strategy has picked the unit
it plans against (ADR 0064). The planner then reads semantic signatures of that
unit and of every operand beneath it, and writes plan expressions from their
trees. This ADR rules four things about that step:

1. what decides whether a request predicate applies before or after aggregation;
2. where in the plan a request predicate goes;
3. how a tree that holds an aggregate becomes plan nodes;
4. which signature each part of the plan reads.

"Subtree" below names the part of the plan that reads one semantic source —
the planner's *frame*. The noun is provisional.

## Considered Options

- Placement read from the role, where the semantics carry their meaning; a tree
  gives the computation; each source's subtree reads only its own signature
  (chosen).
- **Placement by the expression layer** (ADR 0062 as written), and **the role
  checked against the layer**. Both misplace or refuse a composed Metric and a
  cast Measure.
- **Substituting signature trees into the predicate and pushing the physical
  conjuncts into `ScanNode.filter`**. It applies a predicate below
  the node that gives its semantics their meaning.
- **A predicate at the leaf whose rows it reads**, unconditionally (an earlier
  draft). Under an outer join it changes the answer.
- **An aggregate rebuilt from the definition's family** (`row_body` /
  `aggregate_of`). It drops a boundary `Cast` and has no answer for a Metric.
- **Carrying an operand's physical columns through its projection** so a
  composed subtree evaluates the whole D2 linkage. `CONTEXT.md` **Column** puts a
  physical column below the projection only, and a nested operand's copy names
  columns several levels down.

## Decision Outcome

> A request predicate is applied at the lowest node where every semantics it
> reads is a named column with its meaning attached — directly above the
> projection that names and casts it, or above the aggregation that finishes
> it. It never goes below that node. It never goes below a join into a side
> the join null-fills. A predicate that reads two operands stays at the join.

### Placement is read from the role

A request predicate's placement is read from the **role** on the semantic
definition of each of its leaves, never from the expression layer of any tree:

- every leaf is a Dimension → the predicate applies before aggregation;
- any leaf is a Measure or a Metric → the predicate applies after aggregation;
- a Dimension leaf beside a Measure or Metric leaf → `PlanError::MixedPlacement`.

A row-level Metric with no `agg:` is planned as a Dimension whose `expr:` is its
formula (ADR 0074, *A Metric*), so its leaf is placed as a Dimension leaf is, in
the three rules above. The planner reads that the Metric is row-level from the
roles of the names that its formula reads, down each link, never from the stored
`ExprLayer`.

An activated named filter is placed by its own role, Filter, which applies before
aggregation. Compile refuses a named filter body that reads a Measure or a Metric
(`CompileError::AggregateReferenceInRowLevelBody`), and a Filter-role name as a
leaf of the request predicate is refused (ADR 0062). So a named filter never
mixes with an after-aggregation leaf.

No semantics reads a named Filter. Compile refuses a Dimension's, a Measure's or
a Metric's `expr:`, and a Measure's or a Metric's `filter:`, that reads a named
Filter (`CompileError::NamedFilterReference { semantics, filter }`). A named
Filter's body may read semantics, and it may read another named Filter. On a
Dataset `orders` whose rows o1 to o4 hold (`region`, `channel`) as (eu, web),
(eu, app), (us, web) and (us, NULL), the named Filters `is_eu: region = 'eu'`
and `eu_web: is_eu AND channel = 'web'` are legal. A request that names `eu_web`
keeps one row, o1. A named Filter is a filter at the semantic source's level,
and no rule lets a semantics read one.

The role is one value per name and does not change with the source. The layer is
a fact of one tree on one source, and on `main` it gives wrong answers in two
places:

- a composed Metric's tree and a composed named filter's tree read a column the
  composition carries, under no aggregate root (ADR 0065): `Column(<alias>.<name>)`,
  or on a `Full` join that both sides serve, a `Coalesce` of the two sides'
  columns. They classify `Scalar`;
- a Measure whose declared `data_type` differs from its inferred type is
  `Cast(Aggregate(…), <declared>)` (`compile::expressions::lower::reconcile_boundary`),
  and it classifies `PostAggregate`, not `Aggregate`.

`ExprLayer` stays on every stored node. The planner does not read it.

This rules request predicates only. Whether a composed tree's aggregate root
decides where a composition aggregates is not ruled here.

### A predicate is applied where its semantics carry their meaning

The predicate reads **named columns**. The planner does not substitute
signature trees into it: each leaf reads the column that the subtree below
outputs for that semantics.

| Case | Where the predicate goes |
|---|---|
| Dimension, on a Dataset | above the Dataset's row projection, below its aggregate step |
| Dimension, on a Unionset whose branches serve it | into every branch, above the branch's projection, before the union — in mode `all` and in mode `unique` alike, since a row predicate commutes with both |
| Dimension read from the null-filled side: `where country = 'SE'` on `orders LEFT customers`, where the join reads `country` from `customers` | above the join, as a `FilterNode` over the `JoinNode`. Inside `customers` it would keep every order and null the customer |
| Dimension on either side of `Inner`, or on the preserved side of `Left` / `Right` | into that side |
| A predicate reading two operands (`a.x = 1 OR b.y = 2`) | above the join; each leaf is routed to its operand as a row item |
| Measure: `where revenue > 100` | directly above the aggregate step and its `Cast` — the position SQL calls `HAVING` — reading the column `revenue` |
| Metric | above the node that finishes it: the projection that computes its formula, the formula push-down's `sum`, or step 2 of its own subtree and its cast (ADR 0074, *A Metric*) — inside the operand that delivers it finished, unless the join null-fills that operand's side, where it stays above the join. A row-level Metric with no `agg:`: as in the Dimension rows |
| Measure on a Unionset | above the step that finishes it above the union: in mode `all` the refold, since a branch's partial result does not carry the Measure's meaning yet. Where a Unionset aggregates is not ruled here |
| Temporal Dimension with a requested granularity | on the value before bucketing; the bucketed value belongs to the selection |
| Activated named filter | as a Dimension: its tree is output as a boolean column, applied at the lowest node where that column is attached, then dropped from the output |

Under interface-first authoring every operand serves every name, so a `Left` or
`Right` join reads each name from its preserved side. Today only a `Full` join,
whose `Coalesce` reads both sides, reaches null-filled rows; the rule above holds for
any join that reads a null-filled side.

A demanded name that is not selected is still computed where its meaning is
attached, and dropped from the final projection (ADR 0062). Whether a predicate
at a join may be merged into the join's `post_join_filter` is not ruled here.

**In MVP the planner writes no `ScanNode.filter`.** A predicate sits directly
above the node that makes its columns, and pushing it further down is the
consuming engine's work. A partition key is filtered like any other column.

**ADR 0062's "leaf replaced by the aggregate expression, never by its output
alias" is read as follows.** In the plan, an after-aggregation leaf reads the
output column of the aggregate step or projection that finished it. The plan
cannot restate the aggregate inside the predicate: a Substrait filter relation
holds no aggregate function, and ADR 0066 makes every plan node a Substrait
relation. The clause is a rule for SQL text: a SQL format adapter that writes
`HAVING` restates the aggregate there.

### A tree gives the computation, and the role gives the placement

Every signature tree is materialised once per request with
`ManifestExpressions::reassemble`. A `None` is `PlanError::BrokenTree`. Plan
nodes and the manifest share one `PhysicalExpr` type, so node ids are kept for
diagnostics only.

Where a subtree aggregates the rows it reads:

- each `Aggregate` node in a demanded Measure's or Metric's tree becomes **one
  entry of the subtree's one aggregate step**, with that node's own function,
  arguments, `distinct` flag and filter;
- the part of the tree above those nodes becomes **one projection** over the
  aggregate step's outputs — empty for a plain Measure, the `Cast` for a cast
  Measure, the Metric's expression for a Metric. That projection reads each
  name that the step groups by as the step's output field for it (ADR 0074, *A
  Metric*). In a frame that gives partial aggregates, the projection is empty: a partial aggregate never gives the
  request its value (ADR 0080, *Partial aggregates are an intermediate step*),
  so the `Cast` and the Metric's expression run above the re-aggregation;
- every demanded item of the subtree shares that one aggregate step. An
  aggregate function never gets a node of its own. A Metric whose `agg:`
  aggregates a formula that reads a Measure or a Metric, and a Metric that takes
  its parts, read a subtree of their own with two aggregate steps, joined to the
  frame's subtree (ADR 0074, *A Metric*, *Two aggregate steps*).

The planner never rebuilds an aggregate from the definition's family over a tree
with its root removed. A cast Measure has no aggregate root to remove, and
dropping its `Cast` changes the output type away from the declared `data_type`;
a Metric's inlined aggregates have no definition in hand at all.

The exceptions to these rules are the formula push-down and step 2 of a
subtree of two aggregate steps (ADR 0074, *A Metric*). Step 2 aggregates the
values of a Metric's formula with the Metric's own `agg:`, or adds its parts with
a `sum`, and a cast above it gives the Metric's value type. The formula
push-down reads each Measure that a Metric's formula links as the argument of
the Measure's aggregate, cast on each row to the Measure's value type, and it
writes one `sum` over the formula's per-row terms, in the subtree's one
aggregate step. Where the `sum`'s type differs from the Metric's value type, a
cast above the `sum` gives the Metric's type. So no output type changes.

Subtrees nest. A Unionset in mode `all` has one aggregate step in each branch's
subtree and a refold above the union. Each branch's projection above its
aggregate step is empty, and the `Cast` and the Metric's expression sit above the
refold: a cast of each branch's partial `sum` to `decimal(18,2)` fails where one
branch's sum is outside the type's range and the union's sum is inside it. Where a subtree combines partial results
instead of rows, how it refolds them is not ruled here, and the composed tree's aggregate
root is not that step's computation.

What each kind of tree reads:

- **A Dataset's leaf** is a name path (ADR 0085) whose last part is the
  physical column name, read exactly as written; a dot inside that part is
  just part of the name, since the manifest already holds it apart from the
  `ps` alias. It binds to the scan's schema by that last part, matched
  exactly. Above an aggregate step, a column under `auto` that the step groups
  by is read as the step's output field for it, through a leaf that names that
  field (ADR 0074, *A Metric*).
- **A composed tree is read one level at a time** (ADR 0065). Each operand is its
  own subtree, and the composed tree is used as written over the combined rows.
  A composed Metric or named filter reads the column the composition carries for
  it, which its operands output finished.
  - On a **Joinset**, a leaf's first part names the operand whose output it
    reads, and routing reads only that part (ADR 0085).
  - On a **Unionset**, every leaf is qualified by the Unionset's own alias
    (ADR 0065) and reads the union's output column, which each branch supplies by
    name below the union. It routes to no branch.

  How a qualified leaf binds to an output column is not ruled here.
- **A `Parameter` leaf** is refused as `PlanError::UnboundParameter`: nothing in
  the planner binds one. Compile writes none today. `current_date` and
  `current_timestamp` pass through unfolded (ADR 0063).
- **Types.** ADR 0066 ruling 5 types the leaves. The planner infers the types it
  creates — an aggregate step's outputs, a condition column — by reading each
  call's declaration (ADR 0089, rule 12), and it checks no coercion again: the
  tree already carries compile's casts. The exceptions are the formula
  push-down, which writes its own casts: each linked Measure's per-row cast to
  the Measure's value type, and the cast of its `sum` to the Metric's value type
  where the two differ; and the cast above step 2 of a subtree of two aggregate
  steps (ADR 0074, *A Metric*).
- **Coverage** (ADR 0059) is read only to NULL-fill a name that a branch does
  not map (ADR 0074, *A Unionset*). No advertised bit is null-filled under
  interface-first authoring.

### Aggregation-level conditions

`CONTEXT.md` **Filter** separates a named filter from an aggregation-level
condition, which narrows which rows feed one aggregation. Three origins of an
aggregation-level condition reach an aggregate step:

- **A Measure's own `filter_expr_id`** goes on that Measure's aggregate call's
  `filter` clause (ADR 0090, rule 2). It is never a `WHERE`, which would narrow
  every other item's rows.
- **A condition already on an inlined `Aggregate` node** — ADR 0065's splice of a
  filtered Measure or Metric into a Metric that references it — passes through
  unchanged: never dropped, never applied twice.
- **A Dataset Metric's own `filter_expr_id`** is joined with `And`, on the right,
  onto every `Aggregate` node of that Metric's tree. Compile applies the same rule
  when another Metric inlines it (ADR 0065, the transitive rule), so a Metric
  planned directly and a Metric planned as an inlined copy agree. A composed
  Metric has none (ADR 0065 D3); its operand applies its own. A Metric whose
  `agg:` aggregates rows is lowered as a Measure (ADR 0074, *A Metric*), so its
  own condition goes on its `agg:` call's `filter` clause, as a Measure's does.

Where the aggregate step sits above the operand whose rows it reads, the operand
outputs the condition as a boolean column beside its rows, and the aggregate step
reads that column.

### Each source's subtree reads only its own signature

Each subtree reads the signatures of the source it reads, and no other source's.

ADR 0065 D2 stores a Joinset's composed Measure's filter as `And(own, c′)`, where
`c′` is a copy of the filter on the signature of each operand the projection
reads. A Unionset's holds its own view only: each branch's own signature already
filters the rows it stacks, and a copy would filter them twice (ADR 0065 D2).

For a concrete operand the copy's leaves read the operand's physical columns. The
compile-seam test named in `source` asserts the filter of `revenue` on `sales`, a
Joinset over the Dataset `sales.orders` (alias `sales_orders`), whose mapping
reads `region` from the physical column `region_code`:

```text
revenue @ sales   filter  And( sales_orders.region = 'eu' , sales_orders.region_code = 'eu' )
                               └ sales' own view            └ copy of the operand's filter
```

`sales_orders.region_code` exists only below the operand's projection, so the
`sales` subtree cannot read it. The operand's subtree already applies that
filter, `region_code = 'eu'`, from its own signature. The copy holds nothing that
signature does not: compile builds it by re-minting `(operand, definition)` and
adding the operand's alias
(`compile::expressions::projection::ComposedProjection::constituent_filter`).
Outside compile, only `validate()` reads a composed filter.

**Ruled: the copy goes.** A Joinset's composed Measure's `filter_expr_id` holds
the composition's own view only, as a Unionset's already does. The meaning is
unchanged: a composed Measure's value is filtered by the composition's own
condition and by the own condition of every operand it reads, and each condition
is read from its own source's signature. The manifest change is the manifest
effort's (below).

**Until it lands**, the planner takes a Joinset's linkage apart. Compile folds it
as `And(And(own, from′), to′)`: own view first, then each read operand's copy in
from-then-to order. So the planner removes the copies from the right end, **last
read operand first** — `to′`, then `from′` on a `Full` join that reads both —
each prefixed with that operand's alias for a concrete operand and unchanged for
a composite one, and applies what remains at the composed subtree. Nothing may
remain, where compile left out the composition's own view (§4d). A right
end that does not match is `PlanError::FilterLinkageMismatch`.

### Handoff to the manifest effort

- A Joinset's composed Measure's `filter_expr_id` stops carrying the operands'
  filters, as a Unionset's already does. This amends ADR 0065 D2, whose rationale
  it keeps: an operand's condition narrows the rows the operand emits, the
  composition's narrows the rows it reads out of them. The removal follows ADR
  0065's own argument against copying a constituent's tree into a projection: a
  copy duplicates a decision that already has one home.

### Amends

- **0037:** `ExprLayer` stays on every stored node, and the planner does not
  read it; a request predicate's placement is read from the role on the
  semantic definition of each of its leaves, never from the expression layer of
  any tree.
- **0062:** in *Decision Outcome*, the placement bullet: "reads the expression
  layer of the predicate's leaves" becomes "reads the role on each leaf's
  definition", never the expression layer; for a before-aggregation predicate,
  "on the still-physical schema" becomes "on the named columns above the
  projection"; "a predicate mixing the two layers" becomes "a Dimension leaf
  beside a Measure or Metric leaf", the typed error; "the leaf replaced by the
  aggregate expression, never by its output alias" is read as 0071's *A
  predicate is applied where its semantics carry their meaning* reads it: in
  the plan an after-aggregation leaf reads the output column of the step that
  finished it, and "never by its output alias" is a rule for SQL text that a
  SQL format adapter keeps; "any measure or metric → applied after
  aggregation" holds but for a row-level Metric with no `agg:`, which is placed
  as a Dimension, and in the paragraph on a selection, "measures-only yields
  one row" holds but for such a Metric, which a request groups by (ADR 0074, *A
  Metric*). The rest of 0062 stands.
- **0065:** in D2, a Joinset's composed Measure's filter holds the composition's
  own view only, as a Unionset's already does, and each operand's condition is
  read from its own signature. The manifest change is the manifest effort's.

### Consequences

- Planner errors this ADR adds to the planner's error roster: `MixedPlacement`,
  `UnboundParameter`, and `FilterLinkageMismatch` until the handoff lands.
  `BrokenTree`, `NoSignature` and `UnresolvedType` (ADR 0066 ruling 5) stand.
- Recorded and not
  ruled here: a Unionset pre-aggregates per branch only in mode `all` (the
  human's ruling, which narrows ADR 0033's "a Union always pre-aggregates
  per branch"); and on a `Full` join a composed Measure's condition
  `own AND from′ AND to′` excludes every row that one side alone supplies,
  whenever a side's condition rejects NULL.

## More Information

Decided 2026-09-29, and on 2026-10-17 for the
projection of a frame that gives partial aggregates, and for the names
that may not read a named Filter and the condition of a Metric's own `agg:`
(2026-10-17), and for the formula push-down's casts, the placement of a
row-level Metric with no `agg:` and a Metric's subtree of two aggregate steps
(2026-10-18).

The decisions of 2026-10-18 read the research on
MetricFlow, Cube, Looker, Malloy and
Calcite, and an oracle on DuckDB and DataFusion 55. The decisions of 2026-10-18
cover how a
Metric's expression over an aggregate step reads a name that the step groups
by.
