---
number: 75
title: A join condition reads each constituent's semantics, and every output name is the one its parent reads
status: accepted
date: 2026-09-30
source:
- CONTEXT.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0067-the-scan-carries-its-read-identity.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- '`Relationship` and `JoinKeySpec` in `crates/semstrait-model/src/model/types/entities/relationship.rs`'
- '`Projections::term` and `residual_leaf` in `crates/semstrait-manifest/src/compile/relationships.rs`'
- the compile-seam test `a_cumulative_unit_joins_its_two_immediate_operands_on_its_own_condition` in `crates/semstrait-manifest/tests/joinset_enumeration.rs`
- the prototypes of the Substrait round trip through DataFusion and of the output names
- the human's rulings
- the decisions of 2026-10-18 and 2026-10-19, and the research on composition constituents
links:
- target: 57
  kind: amends
- target: 65
  kind: amends
- target: 66
  kind: amends
- target: 71
  kind: amends
- target: 73
  kind: amends
- target: 76
  kind: amendedby
- target: 78
  kind: amendedby
- target: 79
  kind: amendedby
- target: 80
  kind: amendedby
- target: 84
  kind: amendedby
---

# A join condition reads each constituent's semantics, and every output name is the one its parent reads

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0076](0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md): in *Names are unique, and the planner's own names are reserved*, the bullet on the `latest` / `earliest` join has no subject, and the cases of one definition giving two output fields in one output schema are a Dimension that a marker or a semi-additive partial aggregate reads at values other than the bucketed value that the request selects, beside that value, and in the bullet *One definition can give two output fields in one output schema*, the `Max` / `Min` side, the two join inputs and the inner-aggregate example have no subject; a marker is an output field that the planner creates and carries a reserved name; in *Ruled elsewhere, not here*, the `on` of the `latest` / `earliest` join has no subject; the rest of 0075 stands (0076's *A marker gives each aggregate its rows*).
> - [ADR 0078](0078-a-source-alias-is-its-source-roles-letter-and-its-index-in-the-manifest.md): in *Names are unique, and the planner's own names are reserved*, source aliases are unique by construction (0078's *An alias is `ss`, its source role's letter and its index*), so "ADR 0057 mints an alias with no dot and intends aliases to be disjoint" reads 0078's rule, and the sentence that begins "Compile's minting does not check its suffixed aliases again" and "How uniqueness is guaranteed is not ruled here" no longer hold; "Unique aliases are a precondition of these rules, not a result of them" stands; in *Considered Options*, the option "An earlier working design's `<name>$filter`, or a counter" stays refused on both its grounds, and "which ADR 0057 refused for aliases" no longer holds. The rest of 0075 stands.
> - [ADR 0079](0079-a-unique-union-also-compares-its-keyset-and-a-re-aggregated-count-is-never-null.md): in *Ruled elsewhere, not here*, which output fields a mode-`unique` Unionset's distinct compares when it passes one up for a join condition is 0079's *A unique union compares its output, its keyset included*; the rest of 0075 stands.
> - [ADR 0080](0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md): in *Ruled elsewhere, not here*, where the planner buckets a Dimension that the request asks for at a granularity, and where each filter that reads that Dimension is applied, is 0080's; the rest of 0075 stands.
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Every output name is the one its parent reads*, the qualifiers that a projection writes gain one case: under a Grainset's drill-across, each child writes its own alias, and the Grainset's frame writes the Grainset's alias for each name that it projects, and, where its reader carries another qualifier, its last projection writes the reader's; the rest of 0075 stands (0084's *How they are joined*).

## Context and Problem Statement

ADR 0066 fixed the join nodes, `JoinNode { join_type, on, post_join_filter }`
and `CrossNode`. This ADR rules four questions that ADRs 0066, 0071 and 0073
and a prototype left open:

1. what a leaf of `on` reads from a constituent, and which output fields an
   operand must give for it (ADR 0066; ADR 0073);
2. how a qualified leaf binds to an output field of its input (ADR 0071);
3. which name each output field has, and what happens when two names collide;
4. whether a request predicate at a join may merge into `post_join_filter`
   (ADR 0071).

## Considered Options

- Semantic condition leaves, whole output names, and a request predicate at a
  join kept out of `post_join_filter` (chosen).
- **Carrying the constituent's physical column up** through each projection, so
  that today's leaves bind. `CONTEXT.md` **Column** puts a physical column below
  the projection only, ADR 0071 rejected the same move for the filter copy, and
  the join would compute a copy of the constituent's tree.
- **Rewriting a condition leaf to the immediate operand's output field**
  (`sales_orders.region` → `sales_r1.region`). It needs the inverse from
  physical column to semantics, which ADR 0065 says does not exist (*A composed
  Measure's filter is the conjunction of the two views (D2)*). It also changes
  the answer: a name that both operands map reads `Coalesce` (ADR 0074, *A
  Joinset*, the per-join-type table), so `sales_r1.region` is the `Coalesce` of
  `orders` and `customers`, and an order with a NULL region joins on its
  customer's region. A prototype, which applies that table, returns
  `(United States, SMB, 65)` where the oracle returns 25.
- **Bare definition names, with a name per unit for qualifiers to select**
  (Spark's `SubqueryAlias`). The unit name needs a node field of semstrait's
  own, and its only Substrait carrier is `hint.alias`, which ADR 0066 keeps at
  `None`. The leaf `sales_orders.region` then binds to nothing, because the unit
  name `sales_r1` requalifies every output field of `sales.r1`. Mending that
  mints a forwarded name and rewrites the leaf, against ADR 0066's
  "alias-qualified leaves intact" (ruling 4, the join).
- **Bare names with stacked joins**, where the sub-composition is not wrapped.
  The condition leaf binds, but every compiled `sales_r1.*` leaf must be
  replaced by the tree of `sales.r1`. That ends ADR 0065's rule to read one
  level at a time (*ADR 0057's qualifier rule is scoped: a condition names the
  constituent, a projection names the immediate operand*).
- **A qualifier and a name as two parts of each output field**, as DataFusion's
  qualified schema has. It cuts the qualifier to bind, and it adds a field of
  semstrait's own.
- **Comparing names and binding leaves with case kept.** A case-only difference
  then passes `validate` and collides where an engine folds identifiers, which
  is why the manifest's own checks are asked to ignore case (ADR 0057, *The
  check ADR 0050 owed is now definable*).
- **An earlier working design's `<name>$filter`, or a counter, for the planner's own names.** A
  suffix can equal a legal definition name. A counter changes when an unrelated
  item joins the request, which ADR 0057 refused for aliases.
- **Merging a request predicate into `post_join_filter`.** It puts two origins
  in one slot and saves nothing at execution.
- **No interim**, the planner lowering no Joinset until compile changes. The
  carrier keeps its type and only its leaf strings change, so the planner can
  read today's artifact and refuse, typed, what it cannot bind.

## Decision Outcome

The first five sections below answer the four questions in order. *Until
compile writes semantic leaves* is the interim.

### A join condition reads each constituent's semantics

The semantic model states a relationship in semantics. `Relationship.on` is
`JoinKeySpec::Declared(Vec<SemanticsName>)`; `Auto`, which compile derives from
the semantics that both sides share; or `Cross`, a cross join, which has no join
condition. `Relationship.filter` is an optional
`RowExpr<UnresolvedSemanticLeaf>`, and compile refuses a physical `Column` leaf
in it ("name the semantics instead"). The glossary agrees: a **Relationship** is
"stated in semantics names", and a **Join basis** is "never in physical
columns".

Compile on `main` writes something else. For each side, it copies the
constituent's own signature tree into the condition (`Projections`: "the
condition *copies* it and qualifies its column leaves"). So a leaf names a
physical column. With `orders` mapping `customer_id ← cust_key`:

```text
on today:   Eq(sales_orders.cust_key,    sales_customers.customer_id)
on ruled:   Eq(sales_orders.customer_id, sales_customers.customer_id)
```

**Every `Column` leaf of a Joinset's join condition and of a relationship filter
is `Column(<constituent alias>.<semantics name>)`.** The constituent computes
the value from its own signature, and the join compares the output fields of the
constituents. This applies ADR 0071's rule to conditions: each source's subtree
reads only its own signature.

- **The qualifier stays the constituent's** (ADR 0057, *The qualifier names the
  constituent, not the immediate operand*, which ADR 0065 scopes to condition
  leaves in *ADR 0057's qualifier rule is scoped: a condition names the
  constituent, a projection names the immediate operand*). Only the part after
  the dot changes: it is a semantics name, not a physical column.
- **A constituent may itself be a composition.** Where the constituent is a
  Unionset or a Grainset, the leaf is `Column(<that constituent's
  alias>.<semantics name>)` and names no branch or child. Compile's qualifier
  check for a join condition and a relationship filter admits the alias of each
  constituent that the Joinset reaches through its Joinset operands: a Dataset,
  a Unionset or a Grainset. A relationship end names one of the Joinset's own
  constituents, never a branch or child inside one, and `on: auto` reads the
  constituent's guaranteed keysets. `validate()` checks a semantic leaf against
  the names that the constituent serves, which for a Grainset is its mask.
  A cross-grain Joinset (ADR 0054) has no relationship, and its basis is a
  keyset that each child guarantees, so its constituents are its two children,
  each a Dataset, a Unionset or a Joinset: its leaf is `Column(<child
  alias>.<keyset member>)`, and the qualifier check admits those two aliases
  there and no source inside them. For a Joinset child, the leaf reads the value
  that the child's projection computes, `Coalesce` where both of its operands
  map the name (ADR 0074, *A Joinset*), and the condition holds no `Coalesce`.
  The column of a constituent inside the child is not that value where the
  child's join null-fills that constituent.
- **ADR 0050 is applied, not amended.** Its decision reads: "The expression's
  referents are the operands' projected output names … the condition references
  *those* names". ADR 0057 read that clause as the physical column: a leaf "is
  written `<alias>.<physical column>`" (its *Decision Outcome*), because the
  clause "does not say the projection **renames**" (its *Amends*). This ADR
  withdraws that reading.
- **A computed mapping stays in the constituent.** For `region_key =
  concat(region, customer_id)`, compile today writes
  `concat(sales_orders.region_code, sales_orders.cust_key)` into `on`. Under
  this rule the leaf is `sales_orders.region_key`, and `orders` computes it.
- **The side of a relationship filter leaf stays compile's**: a leaf reads the
  side that serves its semantics, `from` first (`relationship_filter_leaf`).

### What an operand gives for a join condition

An operand passes up, unchanged and under the leaf's own name, each output field
that a join condition or a relationship filter above it reads from one of its
constituents. The constituent outputs that field even where the request does
not select its semantics. A passed-up field must survive the operand's own
aggregation, so where an operand aggregates below the join, each such field is
one of its groupings. Whether and where an operand aggregates is ADR
0074's (*A Joinset*, **Pre-aggregation**).

In the fixture `chain_model` (`joinset_enumeration.rs`), `sales = (sales.r1,
regions)` joins on `sales_orders.region = sales_regions.region`, and `orders`
sits inside `sales.r1`:

```text
orders    writes   sales_orders.region, sales_orders.customer_id, …
sales.r1  writes   sales_r1.region, sales_r1.revenue, …  and passes up sales_orders.region
sales     on       sales_orders.region = sales_regions.region
```

### Every output name is the one its parent reads

Only two kinds of name reach the format: the root's, as `RelRoot.names` from
`SemanticPlan.output_names` (ADR 0068), and a scan's, as the read's base schema
(ADR 0067). A Dataset's physical read, its empty read included, gives physical
column names there, and the empty read of a Unionset (ADR 0073) gives the
union's output names. The names between them do not:

- Substrait reads a field by position (`StructField.field`). `ProjectRel` and
  `JoinRel` carry no names.
- Elsewhere a name exists only in `RelCommon.hint.alias` and
  `hint.output_names`. A hint "should not impact correctness", and ADR 0066
  keeps `hint` at `None`.
- The producer does not write `output_schema` (ADR 0068). substrait-explain
  prints positions, and DataFusion's consumer names its own fields (the research and
  the prototype).

So the names in an intermediate output schema serve two readers: `validate`,
and the producer's lookup from name to position.

- **A projection writes each semantics as `<qualifier>.<definition name>`.** The
  qualifier is the one that its reader's leaf carries:
  - under a Joinset, the operand's own alias: `sales_r1.region`; a field that
    the operand passes up for a join condition or a relationship filter keeps
    its constituent's alias: `sales_orders.region`;
  - under a Unionset, the Unionset's alias: every branch writes
    `<union alias>.region`;
  - a Grainset operand or branch that one child answers has no projection of
    its own, so that child writes these names under the qualifier that the
    Grainset's reader carries; under a drill-across, which a Grainset that is
    a constituent runs only as a disjoint branch or as the operand that a
    Joinset keeps alone (ADR 0084, rule 1), the Grainset's frame projects them
    under its own alias, and its last projection writes them under that
    qualifier (ADR 0084, rule 7); the child that answers a Grainset operand of
    a Joinset that builds its join is the one that ADR 0074's pick takes at the
    granularity that the Grainset declares;
  - at the root, none: the root writes the request's bare names (ADR 0062),
    which are `SemanticPlan.output_names` (ADR 0066).

  The parent's input says which names the operand writes: each operand "carries
  the names it must serve" (ADR 0073, *One decision shape, checked per body*). A
  Dataset's physical read keeps the physical column names, and a Dataset leaf
  binds to them as written (ADR 0071). The empty read of a Unionset, which ADR
  0073 types as the union's output, writes the union's output names.
- **A leaf is a name path (ADR 0085). A Dataset's leaf names a physical column,
  and it binds to the scan's schema by its last part (ADR 0071). Every other
  leaf names an output field, and it binds to the one output field of its input
  whose one-identifier name equals `<first>.<last>`, the leaf's own parts
  joined.** No string is cut to bind it, because the
  name path holds its parts apart: routing reads only
  the first part (ADRs 0071 and 0073). A plan node carries no name of its own
  for a qualifier to select, and `hint` stays at `None`.
- **Where each leaf binds.** `on` binds against the two inputs, `left ++ right`.
  `post_join_filter` binds against the join's natural output. A cross join's
  filter binds against the output of the `CrossNode`, which is its
  `FilterNode`'s input. Each binds before any emit form (ADR 0066).
- **A name is one identifier.** A consumer that rebuilds the plan by name reads
  and quotes each name whole, and never splits it at a dot. In the
  prototype, DataFusion's `col("sales_r1.region")` splits the name and fails
  ("No field named sales_r1.region. Did you mean '"sales_r1.region"'?").
  `ident` and a quoted name resolve. DataFusion's SQL unparser quotes the whole
  name, and its SQL re-runs with equal rows.

### Names are unique, and the planner's own names are reserved

- **`validate` refuses a repeated name** in every output schema except a
  Dataset's physical read, including an emit form that maps two positions to
  one name. In those output schemas, names are compared, and qualified leaves
  are bound, **without regard to case**, as is asked of the manifest's own
  checks (ADR 0057). A Dataset's physical read, its empty read included (ADR
  0073), and a Dataset leaf keep ADR 0071's exact match. The empty read of a
  Unionset is not a Dataset's physical read: a leaf binds to it as it binds to
  the union's output when the branches are read.
- **A Joinset's join repeats no name while source aliases are unique.** ADR 0057
  mints an alias with no dot and intends aliases to be disjoint, a constituent
  sits on one side of a join only, and a relationship from a data kind to itself
  is refused (ADR 0057). Unique aliases are a precondition of these rules, not a
  result of them: routing reads a leaf's qualifier (ADR 0073), so a qualifier
  must name one semantic source. Compile's minting does not check its suffixed
  aliases again, so two aliases can still collide, and the repeated-name rule
  catches only a collision that repeats a full name. How uniqueness is
  guaranteed is not ruled here.
- **An output field that the planner creates carries a reserved name**,
  `<qualifier>.__plan_<purpose>_<definition name>`. This form replaces an earlier working design's
  suffix form `<name>$filter`, which a legal definition name can hold. The
  prefixes `__semstrait_`, `__plan_` and `__agg_` are the ones that `Name`
  reserves (`crates/semstrait-ir/src/primitives.rs`).
- **The `latest` / `earliest` join** (ADR 0074, *A semi-additive aggregate's
  strategy*) compares its two sides on the groupings and on the axis,
  which both sides carry, so its two inputs would carry the same names. Every
  output field of its `Max` / `Min` side is one that the planner creates, so it
  carries a reserved name, and the other side keeps its names. For `balance`,
  `latest` on `date`, requested by `region`, the `Max` side writes
  `<qualifier>.__plan_<purpose>_region` and `<qualifier>.__plan_<purpose>_date`.
- **One definition can give two output fields in one output schema.** ADR 0074's
  semi-additive inner aggregate is "grouped by the request's grain plus the
  axis: the time axis at the source's granularity, or at its own values where
  the source has none, and any other axis at its own values" (*A semi-additive
  aggregate's strategy*), and its distinct-rows aggregate "runs over the
  request's grouping, the identity, and the aggregate's arguments and condition"
  (*A Joinset*, **Fan-out**). So a Dimension can sit beside itself at another
  granularity. Outside the `Max` / `Min` side, the value that the request
  selects keeps `<qualifier>.<definition name>`, and the other value is an
  output field that the planner creates, so it carries a reserved name. Two
  output fields that the planner creates for one definition carry different
  `<purpose>` parts where they meet in one output schema, as the two inputs of a
  join do. For `balance`, `latest` on `date`, requested by `date` at Month over
  a daily semantic source, the inner aggregate writes `<qualifier>.date` for the
  month and `<qualifier>.__plan_<purpose>_date` for the day.
- **Field order.** Below the root, a projection orders its output fields by
  semantics name (ADR 0057, *Column order follows the semantics name*), and two
  output fields of one semantics by their whole names, which differ, so they
  have one order in every branch of a union. At the root, `select` fixes the
  order (ADR 0062).
- **A root output name is a definition name**, so `validate` refuses one that
  starts with a reserved prefix. Two definition names that are equal without
  regard to case are refused at the root by the repeated-name rule. A root name
  carries no qualifier, and the planner knows the output fields it creates, so
  no qualified name is read for its prefix.
- **In an intermediate output schema, a reserved definition name can collide.**
  Until compile refuses a definition name with a reserved prefix, a demanded but
  unselected definition named `__plan_filter_revenue` writes
  `<qualifier>.__plan_filter_revenue`, the same name as an output field that the
  planner creates for `revenue` with the purpose `filter`. The repeated-name
  rule then refuses the plan, typed; no field is read for the other.

### A request predicate at a join stays above it

A request predicate that ADR 0071 places at a join is a `FilterNode` above the
`JoinNode`. It is never merged into `post_join_filter`, which holds the
relationship filter only. Both forms give the same rows, because
`post_join_filter` applies after the join type. One slot keeps one origin, so
the trace and the explain text show the relationship's filter apart from the
caller's. The merge saves no work: the DataFusion rules already rewrite
`post_join_filter` to a `FilterRel` above the join (ADR 0068).

### Until compile writes semantic leaves

Until then, a leaf of a join condition or of a relationship filter is
`<constituent alias>.<physical column>`, and no operand outputs a physical
column. The planner accepts such a leaf only where the constituent has a
signature for a definition of the same name, compared without regard to case,
whose tree is exactly `Column(<that column>)`. Then both readings name one
value, and the leaf binds, as ruled, to the output field that the constituent
writes for that definition. Every other leaf is refused as
`PlanError::PhysicalConditionLeaf { semantic_source, column }`.

Compile casts a column that it did not observe to its semantics' declared type
(ADR 0089, rule 16), so that definition's tree is
`Cast(Column(<that column>), <type>)`, never exactly `Column(<that column>)`.
The interim then matches no leaf of a constituent with no catalog. This
interim is not on the e2e oracle's path: the oracle builds semantic condition
leaves with the planner, with no `PhysicalConditionLeaf` gate (per a resolution).

With `orders` mapping `customer_id ← cust_key` and `region ← region_code`, and
`regions` mapping `region ← reg`, `sales_customers.customer_id` (automatic
mapping) plans, and `sales_orders.cust_key`, `sales_orders.region_code` and
`sales_regions.reg` are refused. Every compile-seam fixture on `main` maps each
semantics of a join basis to a column of its own name.

This follows the precedent for a carrier whose type stays and whose
content changes: the planner reads what compile builds into the decided meaning,
and refuses the rest typed (ADR 0066's interim cross marker, ADR 0071's
`FilterLinkageMismatch`).

### Ruled elsewhere, not here

- A join's cardinality rides in its `semantic_meta` (ADR 0066).
- Integrity reaches no node. ADR 0074's rule for leaving out an operand reads it
  (*A Joinset*, **Leaving out an operand**).
- The weakest tuple of a cross-grain unit (`ManyToMany`, `Integrity::None`,
  `CrossFilter::None`) belongs to cross-grain units, which ADR 0073 keeps out of
  MVP.
- A cross join's relationship filter is a `FilterNode` above the `CrossNode`
  (ADR 0066).
- Where the planner buckets a Dimension that the request asks for at a
  granularity, and where each filter that reads that Dimension is applied, is
  not ruled here.
- Which output fields a mode-`unique` Unionset's distinct compares when it
  passes one up for a join condition, and the `on` of the `latest` / `earliest`
  join, are not ruled here.

### Amends

- **0057:** in its *Decision Outcome*, a column leaf of a Joinset's join
  condition or of a relationship filter names the constituent's semantics, not
  "`<alias>.<physical column>`"; 0057's reading of ADR 0050's referents clause
  is withdrawn, where ADR 0057's *Amends* says the clause "does not say the
  projection **renames**"; in *The check ADR 0050 owed is now definable*, the
  name it checks after the dot is a semantics name once compile writes semantic
  leaves, and the check changes with that compile change; in *The attachment
  says whether a leaf is qualified, because a dot cannot*, a
  composition-attached expression carries the output fields that its
  constituents give, not the columns their mappings produced; its consequence
  "The cut before quoting is an adapter obligation with no adapter behind it" is
  replaced by 0075's **A name is one identifier**; in *Column order follows the
  semantics name*, below the root two output fields of one semantics are ordered
  by their whole names; in *The qualifier names the constituent, not the
  immediate operand*, "A constituent is concrete and a composite is not" reads:
  a constituent that a condition leaf names is a Dataset, a Unionset or a
  Grainset, or a child of a cross-grain Joinset, a Joinset included, and a
  composite, the Joinset that a cumulative unit has as an immediate operand, is
  never a constituent; a leaf names a Unionset or a Grainset constituent, and a
  child of a cross-grain Joinset, by its own alias and a semantics name, never
  by a branch, a child or a constituent inside it; the constituent
  qualifier stands (0075's *A join condition reads each constituent's
  semantics*, *Until compile writes semantic leaves*, *Every output name is the
  one its parent reads* and *Names are unique, and the planner's own names are
  reserved*).
- **0065:** in *ADR 0057's qualifier rule is scoped: a condition names the
  constituent, a projection names the immediate operand*, a condition names the
  constituent's output field for a semantics, not a physical column, so "only a
  concrete constituent holds one — a composite holds none" no longer limits
  which constituent a condition names: a constituent may be a Unionset, a
  Grainset or a Joinset that is a child of a cross-grain Joinset; none holds a
  physical column, and each is named by its own alias; the
  constituent qualifier for a condition and the immediate operand's qualifier
  for a projection stand (0075's *A join condition reads each constituent's
  semantics*).
- **0066:** in ruling 4 (the join) and in *Consequences*, the binding rule
  is 0075's *Every output name is the one its parent reads* and
  *Names are unique, and the planner's own names are reserved*; the rest of
  0066 stands.
- **0071:** in *A tree gives the computation, and the role gives the
  placement*, how a qualified leaf binds to an output field, which 0071 leaves
  open, is 0075's *Every output name is the one its parent reads* and *Names
  are unique, and the planner's own names are reserved*; in *A predicate is
  applied where its semantics carry their meaning*, a request
  predicate at a join is a `FilterNode` above the `JoinNode` and never merges
  into `post_join_filter` (0075's *A request predicate at a join stays above
  it*); the rest of 0071 stands.
- **0073:** in *One decision shape, checked per body*, the output fields an
  operand also gives for a join condition, which 0073 leaves open, are those
  that 0075's *What an operand gives for a join condition* rules; the rest of
  0073 stands.

### Consequences

- Planner errors: `PhysicalConditionLeaf`, until compile writes semantic
  leaves; not on the e2e oracle's path (per a resolution).
- `validate` gains the binding, repeated-name and reserved-prefix rules of
  *Every output name is the one its parent reads* and *Names are unique, and the
  planner's own names are reserved*.
- The open question of the round-trip prototype ("Should planning guarantee unique column names at every
  intermediate schema, or must the adapter disambiguate duplicate names
  positionally?") is answered: planning guarantees them.
- Glossary: **Operand**, **Column**, **Column qualifier** and **Physical
  expression** no longer say that a join condition reads a physical column;
  **Emit form**, **Selection** and **Format artifact** say output fields or
  names. **Output schema** and **Output field** are added.

## More Information

Decided 2026-09-30, 2026-10-01 and 2026-10-04, and on 2026-10-18
for a constituent that is a composition.

A prototype runs `select region_name,
segment, revenue where segment = 'SMB'` over `((orders LEFT customers) INNER
regions)` through Substrait, substrait-explain and DataFusion 55 under each
naming scheme, against a hand-written SQL oracle.
