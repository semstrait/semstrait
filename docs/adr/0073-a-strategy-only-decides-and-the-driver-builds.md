---
number: 73
title: A Strategy only decides; the driver builds with a fixed rule list and one-node steps
status: accepted
date: 2026-09-30
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0005-warning-preserving-stage-error-policy.md
- docs/adr/0008-unsealed-extension-traits.md
- docs/adr/0011-thin-manifest-in-memory-graph.md
- docs/adr/0018-canonical-plan-ir-contract.md
- docs/adr/0031-metadata-as-compile-stratum.md
- docs/adr/0033-composite-plan-shape.md
- docs/adr/0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0063-the-planner-takes-a-request-context-not-a-session.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0067-the-scan-carries-its-read-identity.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0070-a-path-source-is-one-base-path-read-in-one-format.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- the human's rulings of 2026-10-17
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18, and a probe of a `ProjectRel`'s emit on DataFusion 55
- the decisions of 2026-10-18 and 2026-10-19, and the research on composition constituents
- sharpened in place (a Unionset branch NULL-fills the names that its mask does not hold; a composed branch is folded with the literals that its branches or children share)
- sharpened in place (the fold of a Unionset branch replaces a leaf that the branch NULL-fills by its typed NULL)
links:
- target: 24
  kind: amends
- target: 3
  kind: amends
- target: 5
  kind: amends
- target: 8
  kind: amends
- target: 42
  kind: amends
- target: 64
  kind: amends
- target: 66
  kind: amends
- target: 67
  kind: amends
- target: 68
  kind: amends
- target: 70
  kind: amends
- target: 74
  kind: amendedby
- target: 18
  kind: supersedes
- target: 34
  kind: supersedes
- target: 33
  kind: amends
- target: 75
  kind: amendedby
- target: 77
  kind: amendedby
- target: 80
  kind: amendedby
- target: 81
  kind: amendedby
- target: 84
  kind: amendedby
- target: 89
  kind: amendedby
- target: 96
  kind: amendedby
- target: 97
  kind: amendedby
---

# A Strategy only decides; the driver builds with a fixed rule list and one-node steps

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): each Unionset branch that is read carries the names it NULL-fills beside the names it serves: a name that its mask does not hold; the driver writes each such name as a typed NULL in that branch's own row projection, and for a Grainset branch, which keeps those names out of its pick and its drill-across, in the row projection of the child that answers alone or in the projection of its drill-across frame (0084's rule 7). An operand of a Joinset carries no NULL-filled name. In *The build order*, stage 5b's "`step_join` with `Inner`" reads, for a subtree that reads the union of a Unionset again: the join that *A Unionset*, **One form for every read**, gives (`Inner` over the same read, `Left` onto a frame that has aggregates of its own, `Full` with coalesced keys between two reads again, and `Left` for a later subtree over the left read after a `Full`).
> - [ADR 0075](0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md): in *One decision shape, checked per body*, the output fields an operand also gives for a join condition, which 0073 leaves open, are those that 0075's *What an operand gives for a join condition* rules; the rest of 0073 stands.
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *One decision shape, checked per body*, "every grain of one Grainset maps the same interface" no longer holds, because a child serves only the names it maps; in *Details fixed with the seam*, "No model that can be authored today produces a null-filled bit" no longer holds (0077's *A constituent serves only the names it maps*).
> - [ADR 0080](0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md): in *One decision shape, checked per body*, the leaves of either of the two copies of a filter over a Dimension that both operands map are among the names that an operand serves only where the operand is read, and they do not make the Strategy read it; in *The build order*, stage 4 buckets only in a frame whose own aggregate step gives final aggregates, and "never below a join into a null-filled side" holds only for a predicate that reads no Dimension that both operands map; in *A Unionset prunes a branch its predicate excludes*, a copy that goes into a side that the join null-fills is placed on that side's rows, so it may exclude a branch; the rest of 0073 stands (0080's *Only a frame that gives final aggregates buckets*, *A predicate copy goes into its operand on every join type* and *A copy applies only on an operand that the plan reads*).
> - [ADR 0081](0081-a-partition-key-is-a-column-and-partition-def-names-the-dimension-that-maps-it.md): in *The physical read below a Dataset*, the consumer that "reads the files in the `key=value` folders under `uri_folder`" also does what 0081's *The consumer reads `__HIVE_DEFAULT_PARTITION__` as NULL* lists; in *Consequences*, "The planner needs a partition-fed Dimension to be readable as a column, named by its key" is met as 0081's *A partition key is a column* rules. The rest of 0073 stands.
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *One decision shape, checked per body*, the table's Grainset row reads: who answers, a child or the Grainset over two or more children; operand not read, a child that 0084's rule 3 sends no name to; combine, pass-through or drill-across; "A Grainset owns no signatures (ADR 0065), so it has no "answer as itself" choice" holds only for a name that a child maps whole, and in *Through a composed branch*, "a Grainset owns none (ADR 0065)" reads: a Grainset owns a signature only for a Metric that it computes (0084's rule 8); "ADR 0054's cross-grain Joinsets are not among them in MVP" stands (0084's rule 11), and "the rule for cross-grain units stays open until a narrower grain can be authored" no longer holds; "the requested granularity comes from the parent's input" holds for a child of a drill-across only where that input has one, and otherwise the child takes the granularity of 0084's rule 2; the combines in that section, the rule in *Three parts* for the form in which each operand gives an item, and stage 1 in *The build order* gain `drill-across` as 0084's rule 6 gives it; for a Metric whose stored tree reads the frame's own final values (a Metric that a Grainset computes, 0084's rule 8), the driver's Shared rule in *Three parts* for where a Metric's formula is applied makes a Compute frame, a frame kind of its own that no Strategy decides, outside the closed set of combines, so the table gains no value for it; in *The build order*, stage 5b's "`step_join` with `Inner`" reads, for the subtree of a Metric with its own `agg:` that a Grainset computes: a `Full` join on `IsNotDistinctFrom` with coalesced keys (0084's rule 8), and, for the subtree of a Metric that a Grainset computes, stage 5a's "A request predicate on that Metric alone sits directly above it" and the next paragraph's "or stage 5a for a Metric's step 2" do not hold: a request predicate on that Metric sits above the joins (0084's rule 7); in *Two passes*, under "A decision is named by its **path** from the root", each formula group's frame below a Compute frame has a path of its own, `<grainset>#group<n>`, with n counting from 0 (0084's rule 8); in *One input for every Strategy call*, the names that a child of a drill-across must serve include each name that the tree of a Metric that the Grainset computes reads (0084's rule 8) and each Measure under the condition of 0084's rule 9, and "a Grainset answers only through a grain" holds only where one child answers alone (0084's rule 1); the rest of 0073 stands (0084's *How they are joined*, *A Metric that no child maps whole* and *Compile's cross-grain Joinsets*).
> - [ADR 0089](0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md): in *A Unionset prunes a branch its predicate excludes*, "an authored literal body is a `Literal` too" holds also for a tagged literal whose type differs from its semantics' declared type: compile casts it to the declared type, and the fold stores the cast's value as a literal (ADR 0089, rules 17, 21 and 23); where the canonical CAST gives that literal no value (ADR 0086, rule 10), the fold leaves the cast, so its leaf is not replaced, and a predicate that only that leaf decides keeps the branch; "The fold is a shared helper. It is a pure function over expressions" becomes the fold of ADR 0089, rules 21 to 23. The rest of 0073 stands.
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *One input for every Strategy call*, "A Grainset that is an operand of another composition picks its child at the granularity that it declares" reads: it picks its child at the granularity that it carries, declared or derived (0096's *Shared granularity*); the rest of 0073 stands.
> - [ADR 0097](0097-a-unionset-prunes-a-branch-whose-children-cannot-serve-the-demand-with-an-advisory.md): in *A Unionset prunes a branch its predicate excludes*, a Grainset branch is also pruned where its children cannot serve the demand by what they map (0097's *The rule*); "No advisory. A pruned branch loses no rows" holds for a branch that the fold prunes, and a branch pruned for this reason raises `BranchNotServable` (0097's *The advisory*); the empty read of "Every branch excluded" is not the answer where every branch is pruned and the test of 0097's *The rule* prunes at least one of them: the request is refused, and where the fold alone excludes every branch the empty read stands (0097's *The rule*); in *One decision shape, checked per body*, the Unionset row's cell "Operand not read", "a branch the predicate excludes", also holds a Grainset branch whose children cannot serve the demand by what they map (0097's *The rule*); the rest of 0073 stands.

## Context and Problem Statement

A unit's semantic source has a composition body: Dataset, Joinset, Unionset or
Grainset. Some planning depends on the body (who answers, which operands are
read, how their rows combine), and some does not (the form each operand gives,
Metric formulas, bucketing, predicate placement). This ADR decides who owns each
part, who builds the plan nodes, and whether an `optimize` stage stays.

## Considered Options

- A Strategy that only decides, a fixed rule list that the driver runs, and
  steps that build one node each (chosen).
- **Each Strategy applies the rules itself, through shared helpers.** Four
  bodies would hold four copies. Also, the form an operand gives chains across
  bodies: each Strategy would have to understand its parent's request in the
  parent's own terms.
- **The rules as a second open trait.** This is a second ADR 0008 promise that
  no caller needs.
- **The `Resolution` for the root and a second, smaller shape for operands.**
  With two shapes, the one dispatch site would need two signatures.
- **One decision type per body.** The bodies share most values ("not read" in
  Joinset and Unionset, a stand-in in Joinset and Grainset), and the driver
  needs the three-part shape to build anyway.
- **A `combine` method on the Strategy**. The combine values are
  closed, so a second method would only repeat the driver's mapping.
- **Each rule choosing its own place in the build.** Two rules could each pick a
  place and disagree, and golden plans would lose their fixed shape.
- **One branch per partition planned by the planner** (ADR 0067 as written),
  and **key values found by splitting file paths**. The first puts a union in
  every partitioned plan. The second is the string convention that ADR 0067
  refused.
- **Keeping the `optimize` stage with an open trait and no pass** (ADR 0003 as
  written), or as an identity stage with no trait. See "No `optimize` stage"
  below.
- **A Unionset pruning that asks each composed branch's own Strategy.** The
  Unionset decides before its branches do, so it would need a branch's decision
  before its own. The Unionset folds its predicate through a composed branch's
  operands instead (*A Unionset prunes a branch its predicate excludes*), which
  needs no decision of the branch.
- **Folding a name that both operands of a Joinset map through its
  `Coalesce`.** A predicate never reads that value: it reads the name on each
  operand that the plan reads (ADR 0077, ADR 0080). Where the operands give
  different literals, the coalesced value can exclude a branch that gives
  rows. Pruning by each operand's own copy needs the Joinset's decision of
  which operands it reads.
- **Keeping one branch, or refusing the request, when every branch is
  excluded.** The first reads data for nothing and picks a branch by name order.
  The second refuses a valid request.

## Decision Outcome

The planner has three parts:

- A **Strategy**, one per variant of a semantic source's body. It only
  decides: who answers, which operands are read and which names each serves,
  and how they combine.
- A fixed list of **rules**, which the planner's driver runs for every unit.
  They decide what does not depend on the body.
- **Steps**, which build one plan node each and decide nothing.

Planning runs in two passes: first decide, top-down, then build, bottom-up. The
Strategy has one method, and the driver builds every plan node. MVP has no
`optimize` stage and no `OptimizerPass`.

Where the working draft differs, this ADR governs. The draft's
nouns (*frame*, *delivery*, *refold*, *step*, …) are provisional. The words below
are plain descriptions, and
a later ruling names them. Here
*driver* means the planner's own loop over the decision tree, not the
glossary's **Driver**, a Grainset routing unit, which a later ruling also covers.

### Two passes

1. **Decide**, top-down from the scope root. The dispatch site calls the
   Strategy for the unit's body, and the Strategy returns a decision. The rules
   then fix what each operand that is read must give, and which of the unit's
   predicates go into it (ADR 0071's placement). The driver repeats this for
   each such operand. So a unit's predicates are placed before its Strategy is
   called, and its operands' predicates after. The result is a **decision
   tree** of plain values, with no plan nodes.
2. **Build**, bottom-up from the leaves. The driver turns each decision into
   plan nodes by calling steps.

Either pass can fail with a typed error. A decision is named by its **path**
from the root, never by an entity id, because one Dataset can be the operand of
two units in one plan. Every error and every advisory names a path.

### Three parts

**Strategy.** The variant of the semantic source's composition body chooses the
Strategy, and exactly one exists per variant. A compile-synthesized semantic
source is planned by the same rule as an authored one.

**Rules.** The rules are one fixed list that the driver runs for every unit. A
rule reads how rows combine, the item's role and aggregation, and the facts
that ADR 0074 names (a combine's mode and relationship, keysets, temporal
shape). It never reads which body combined them. The list covers:

- the form in which each operand gives an item: rows, partial aggregates or
  final aggregates;
- how partials combine again above a stack or a join;
- where a Metric's formula is applied over its aggregates or its formula parts,
  and where a Metric's own `agg:` aggregates the formula's values;
- bucketing to the requested granularity;
- where each predicate is applied.

What each rule says is ADR 0071's (predicates) and ADR 0074's (aggregation). This ADR fixes only
who owns the rules, and it states none of them. A rule may apply where only one
body goes today: the disjoint-branch rule applies above a stack, and today only
a Unionset stacks. The rule still belongs to the list, so a later body that
stacks gets it unchanged.

**Steps.** A step builds one plan node and writes its node-common block. It
decides nothing, and it reads neither the manifest nor the read model. It takes
values that are already resolved: typed columns, reassembled expressions, and
the reader copied from the physical source. Some steps have one caller today
(`step_cross`, `step_union`). The step signatures are those in ADR 0066's
handoff. `step_distinct` is `step_agg(<the columns it keeps>, [])`, as that
handoff says; for a dimensions-only request, those columns are the selected
dimensions (ADR 0066 ruling 6).

### One input for every Strategy call

Every call gets the same input shape: the semantic source, the names it must
serve, the requested granularity, the units that may stand in for it, the form
the parent wants, and the predicates placed on its rows. For the scope root, the
driver builds this input from ADR 0064's `Resolution`, the resolver's result,
and every request predicate is placed on the root's rows. For an operand, the
parent's decision builds it, and its list of stand-ins is empty. A Dataset, a
Joinset and a Unionset operand can each answer as itself. Compile also produces
a Grainset operand, and a Grainset
answers only through a grain. A Grainset that is an operand of another
composition picks its child at the granularity that it declares, and answers
through the child that ADR 0074's pick takes, or, only where the form the
parent wants is final aggregates, through ADR 0084's drill-across (ADR 0074,
*A Grainset*). A later resolver
that finds stand-ins below the root fills that same field, and no signature
changes.

### One decision shape, checked per body

Every decision has three parts:

1. **Who answers**: the unit itself, or a stand-in (a *bypass*).
2. **Which operands are read, and which names each serves.** Each operand
   that is read carries the names it must serve, predicate leaves included.
   The names that each tree of the frame reads, down each link, are among
   them: those of each `expr:`, each `filter:`, each predicate and each join
   condition. The driver builds the operand's input from them: the requested
   granularity comes from the parent's input; the form, the predicates and any
   grouping that the operand adds come from the rules; and the list of
   stand-ins is empty. A name that a Metric's formula reads outside an
   aggregate adds no grouping to the output. The demand that finishes the
   Metric selects it with no granularity (ADR 0083, rule 4), or the formula
   push-down reads it on each row, or it groups step 1 of the Metric's own
   subtree, which step 2 drops (ADR 0074, *A Metric*). An operand that is not
   read gets no plan nodes.
3. **How the operands that are read combine**: leaf, join, cross, stack or
   pass-through.

The Strategy routes each name by its own signature. A composed signature's leaf
is a name path (ADR 0085) whose first part names the immediate operand
(`Column(<operand alias>.<name>)`, ADR 0065), and routing reads only that part
(ADR 0071). So a `Coalesce` sends a name to both sides. A Unionset sends every
name to every branch it reads; ADR 0074 says how a branch that does not map a
name gives it. A stand-in gets the unit's names unchanged. The physical read
below a Dataset reads the columns that the demanded signatures' leaves name.
Which columns an operand must also give for a join condition is the binding
of `on` against the join inputs.

All bodies share this one type. The driver checks each decision against the
table below, and it refuses with a typed error a value that the body may not
use:

| Body | Who answers | Operand not read | Combine |
|---|---|---|---|
| Dataset | itself | — | leaf over its physical read |
| Joinset | itself, or a sub-composition | an operand that ADR 0074 lets it leave out | join, cross, or pass-through |
| Unionset | itself only | a branch the predicate excludes | stack |
| Grainset | a grain only | — | pass-through |

A Grainset owns no signatures (ADR 0065), so it has no "answer as itself"
choice, and ADR 0064's "on a tie the scope root comes first" does not apply to
it. No step reads `RoleDefinition::Grainset::keysets`. ADR 0074 says when
leaving out a Joinset operand is faithful.

A Grainset's stand-ins are its grains (ADR 0064), and ADR 0054's cross-grain
Joinsets are not among them in MVP. That loses no answer: interface-first is the
only authoring direction (ADR 0053), and under it every grain of one Grainset
maps the same interface (`crates/semstrait-manifest/tests/grainset_composition.rs`).
So a cross-grain unit never serves a demand that no grain serves. The rule for
cross-grain units stays open until a narrower grain can be
authored.

The combine values are a closed set, and the driver maps each value to its
step. So the **`Strategy` trait has one required method, `decide`**, and the
build pass belongs to the driver alone. A bypass builds no node of its own: the
stand-in's subtree is the result.

### The build order

The driver builds each unit in this order. Any stage can be empty.

1. Combine: `step_scan` for a Dataset's physical read; `step_join` or
   `step_cross` for a Joinset (a cross with a relationship filter is followed
   by `step_filter`); `step_union` for a stack, in the Unionset's mode
   (`UnionAll` or `UnionDistinct`, ADR 0066); nothing for a pass-through.
2. Row-level projection: each demanded semantics under its name, the arguments
   below each aggregate, and each aggregation condition as a boolean column.
   The row-level projection has an emit form, and it outputs only the fields
   that a later stage of the frame or its parent reads. Each semantics among
   them is written under its name (ADR 0075): a name of the frame's demand, a
   predicate's leaf, a row identity's Dimension, a marker's `by` and axes, and
   an activated named filter's boolean column included (*One decision shape,
   checked per body*). Each field that the planner creates is written under a
   reserved name: each aggregate's argument, each aggregation condition, the
   part key and each piece of the parts, and each column under `auto` that a
   formula reads outside an aggregate; a column under `auto` takes its column's
   name in the place of the definition name (ADR 0075). No column of the
   physical read passes above it (`CONTEXT.md` **Column**).
3. Pre-aggregate predicates.
4. Aggregate: bucketing to the requested granularity, the aggregate steps
   that ADR 0074 gives, each aggregate with its own filter, and the
   re-aggregation of partials. In a Metric's subtree of two aggregate steps,
   this aggregate step is step 1 (ADR 0074, *A Metric*).
5. Derivation: the projection above the aggregate (Metric formulas, casts). A
   frame that gives partial aggregates has none (ADR 0071).
   - 5a. In a Metric's subtree of two aggregate steps: step 2, the Metric's
     `agg:` or the `sum` of its parts, grouped by the demand, or by its
     reader's step 1 where another such Metric links it (below), then the cast
     of its value. A request predicate on that Metric alone sits directly above it
     (ADR 0074, *A Metric*).
   - 5b. The join of the subtrees: `step_join` with `Inner` and
     `IsNotDistinctFrom` on each grouping column of the demand, or `step_cross`
     where the demand has none (ADR 0074, *A Metric*).
   - 5c. The derivation of each formula that reads two subtrees.
6. Post-aggregate predicates.
7. Projection to what the parent asked for.
8. At the root only: order, then offset and limit.

This is the standard SQL order. Research
found that DataFusion keeps `WHERE` below the projection and puts `HAVING` on
the aggregate. ADR 0071 places each predicate and fixes stage 2's contents: a
predicate goes directly above the projection that names it (stage 3) or above
the aggregation that finishes it (stage 6, or stage 5a for a Metric's step 2),
and never below a join into a null-filled side. This ADR fixes only the order.

A Metric's subtree of two aggregate steps builds stages 1 to 5 over its own read
of the frame's sources, then 5a. The frame's own subtree builds stages 1 to 5.
Stages 5b and 5c then join them and derive over them, and stages 6 to 8 follow.
Where the formula of such a subtree links another Metric of two aggregate steps,
that Metric's subtree joins it directly above its step 1, on step 1's grouping,
before its stage 5 (ADR 0074, *A Metric*).

### The physical read below a Dataset

Every Dataset has a **physical read** below it. It is not a data kind, and the
dispatch site never matches on it, so ADR 0018's single dispatch site holds. The
Dataset Strategy creates it. The physical read owns the reader, the columns
read, their types (ADR 0066 ruling 5, as ADR 0082 amends it: the catalog's
type, else, in a model-fed schema, the type that compile writes from the
semantics mapped to the column, else `UnresolvedType`) and the scan's filter
slot. That slot stays empty in MVP,
because ADR 0071 puts no predicate into `ScanNode.filter`. The Dataset owns the
row-level expressions over those columns.

**A physical source with discovered partitions is one read of its base path.**
When a semantic source reads such a physical source through a `Prefix`
selection (ADR 0070), the planner writes one `ScanNode`:

- The `FilesScan` has the base path, no `pattern` and no `paths`.
- The scan's schema lists the key columns the demand reads, like any other
  column.

The format adapter's faithful default writes this scan as one `LocalFiles`
item, `uri_folder` = the base path. ADR 0068's default of one `uri_file` per
path would write an empty read, and no rows would come back.

The engine discovers the partitions under the base path and supplies the key
columns. It skips partitions by the request's predicates on those columns,
wherever ADR 0071 places them, because pushing a filter down into its own read
is the engine's work. The planner writes no branch per partition, no literal
and no union. From `partition_def` it reads only the keys, never the listing. A
read without partitions stays as ADR 0067 and ADR 0068 say: `paths` is
compile's listing.

This is the simpler form, and the implementation risk is lower. Reads per
partition written by the adapter, and partitions skipped at plan time, can come
later.

**The folder read is not portable Substrait.** The key columns are not in the
files, and no Substrait field carries a partition value: `FileOrFiles` has
none, and `partition_index` numbers a work split (ADR 0067). So the read relies
on a convention outside Substrait: the consumer reads the files in the
`key=value` folders under `uri_folder` and supplies the key columns from the
folder names. A consumer that does not do this reads no key columns, or fails
the read. That is the price of the simpler form. A portable form, with one read
per partition and its key values written as literals, is the open question
"Partition reads beyond the folder read".

**A read without partitions that selects no object is the empty read.** The
planner writes it with `step_empty`, typed by the scan's schema, and it writes
no `FilesScan`. So the engine reads no object that compile did not list, and a
`FilesScan` with no paths is always a read with discovered partitions.

### A Unionset prunes a branch its predicate excludes

Pruning a branch is how the Unionset Strategy behaves, not an optimization. A
branch is **not read** when a pre-aggregate predicate on the Unionset's input
folds to `false` on it. To fold, each leaf whose signature *on that branch* is a
`Literal` is replaced by that literal, and a leaf that the branch NULL-fills is
replaced by its typed NULL, the type that the NULL fill gives the branch's
column; the expression is then folded. Compile writes a metadata Dimension's
signature as a broadcast literal, and an authored literal body is a `Literal`
too. A NULL result counts as `false`, as in a SQL `WHERE`.

- **Through a composed branch.** A composed branch's own signature is never a
  literal: a Joinset's is `Column(<operand alias>.<name>)`, or ADR 0074's
  `Coalesce` for a name that both operands map, a Unionset's is
  `Column(<union alias>.<name>)`, and a Grainset owns none (ADR 0065). So the
  fold reads on through its operands' signatures, down to a Dataset's. It
  stops, and the leaf stays, at an operand that its join null-fills (`Left`:
  `to`; `Right`: `from`; `Full`: both), because pruning below a null-filled
  side would change which rows are null-filled. A name that both operands of a
  Joinset map folds to a literal only where both operands fold it to that same
  literal; otherwise the leaf stays for this fold, at that Joinset and on each
  branch or child below it. The fold never evaluates the `Coalesce`: the plan reads such
  a name in a predicate on each operand that it reads (ADR 0077), and the
  Joinset leaves an operand out by its demand, whatever that operand's copy
  says (ADR 0080), so only a literal that both operands give holds on every
  copy that the plan applies. At a Unionset or a Grainset that the fold
  reaches, as the branch or below it, a leaf folds to a literal only where
  every branch or child folds it to that same literal, and the predicate is
  folded with those shared literals only. So a predicate that is `false` on
  each branch or child through different literals does not exclude the
  composed branch: the branch is read, and takes the predicate into its rows.
  Each Unionset at or below a branch that is read prunes the branches that a
  predicate placed on its own rows excludes, when the driver decides its frame.
  A Grainset there prunes only a child that no name goes to (ADR 0084, rule 3),
  so a predicate that is `false` on a child does not prune that child. The fold
  needs no decision of the branch's own Strategy.
- **Pre-aggregate only.** A predicate is pre-aggregate when it reads no
  Measure and no Metric, because the role gives the placement (ADR 0071). So
  the Strategy knows this without a placement result. A post-aggregate
  predicate never excludes a branch.
- **Only the predicates placed on its rows.** A predicate that stays above a
  join is not on the Unionset's input, so it never excludes a branch. Such a
  predicate reads two operands, or a side that the join null-fills. Pruning a
  branch below a null-filled side would change which rows are null-filled.
- **The substitution serves the decision only.** The predicate the plan
  applies in each branch that is read still reads named columns, as ADR 0071
  says.
- **The fold is a shared helper.** It is a pure function over expressions. It
  builds no node, reads no manifest, and is neither a step nor a rule.
- **No advisory.** A pruned branch loses no rows. The pruned branches are
  named, as trace, in the `semantic_meta` of the root node of the Unionset's
  built subtree. That holds with a `MultisetNode`, with one branch left, and
  with the empty read.
- **One branch left:** no `MultisetNode`, since `validate` requires two or
  more inputs (ADR 0066). In `All` mode the branch is used as it is. In
  `Unique` mode the branch is read under `step_distinct` over every column of
  the union's output, because `UNION DISTINCT` removes duplicates from one
  input too.
- **Every branch excluded:** the read is an empty
  `ScanNode { reader: VirtualTable { expressions: [] } }`, typed as the
  union's output and written by `step_empty`. The later stages still sit
  above it, so a request with no grouping still returns its one row of
  aggregates.

### The dispatch site and a new body

The dispatch site matches `SemanticSource.role_definition` once. It gives each
Strategy its own arm's fields, so no Strategy matches the enum again.
`RoleDefinition` is `#[non_exhaustive]`, so the planner's match has a wildcard
arm. That arm fails with `PlanError::StrategyNotFoundForSemanticSource
{ path, semantic_source, role_definition }`. The path is needed because one
semantic source can occur at more than one path.

A new body needs:

- a `RoleDefinition` arm in the manifest;
- a Strategy;
- one arm in the dispatch site.

A new manifest arm still compiles in the planner, and it fails only when a
request reaches it. If the body combines rows in a new way, it also needs:

- a combine value;
- a step;
- a plan node, which ADR 0066 admits only with the code that writes it;
- a rule-list entry where the rows it gives need one.

**Window readiness** is this recipe applied. A window-bearing body adds a window
combine value, `step_window` and a window node once ADR 0066's next pass admits
window relations. Its operand gives rows.

Only the workspace can add a body. So ADR 0008's open `Strategy` trait lets a
caller *replace* a built-in Strategy, and nothing more. The planner is a value
that holds one Strategy per body variant, and `plan` is a method on it that
keeps ADR 0063's three arguments.

### Details fixed with the seam

- **Anchors.** Steps take no counter. One walk at the end of `plan` numbers the
  nodes 1, 2, 3… in pre-order, so the root is 1. ADR 0066 fixes the numbers,
  and this fixes the order.
- **Read model.** The lookups are ADR 0064's six: `bit_of`, `name_of`,
  `public_sources`, `originated_by`, `operand_closure` and `signature`. Six more
  are added: `definition`, `source_kind`, `alias`, `granularity`, `expr_tree` and
  `physical_source`. Each lookup returns a value the planner owns, never a
  reference to a manifest struct, so a later graph (ADR 0011) can use its own
  shapes. The Strategies and the rules read the manifest only through the read
  model. The rules of ADR 0074 also read keysets and a relationship's join
  type, cardinality, integrity and filter, and those lookups come with them.
- **NULL fill is ADR 0074's.** No model that can be authored today produces a
  null-filled bit. ADR 0074 says which names a branch NULL-fills: the names
  that its mask does not hold.

### No `optimize` stage

The resolution pipeline is parse, validate, compile, plan, adapt. Every rewrite
named so far has an owner:

- elision of a branch or an operand: the Strategy;
- the form an operand gives: a rule;
- a plan-node rewrite: the engine (ADR 0068).

An open `OptimizerPass` with no pass would be the one-way promise that ADR 0064
refused for `Complexity`, because it "buys nothing today". No pass chain runs
between the two planning passes either. A stage and a trait can come back
later, together with the pass that needs them. That reverses an earlier
"the optimizer module survives".

### Amends

- **0003:** `optimize` is not retained, and the pipeline is parse, validate,
  compile, plan, adapt.
- **0005:** `optimize` leaves the stage list.
- **0008:** `OptimizerPass` leaves the unsealed roster. `Strategy` stays
  unsealed; it buys per-body substitution only, and its one required method is
  `decide`.
- **0018:** The dispatch site is the single match site: it matches
  `SemanticSource.role_definition` once, and its wildcard arm fails with
  `PlanError::StrategyNotFoundForSemanticSource`; the driver, not a Strategy,
  builds every plan node.
- **0024:** MVP has no `optimize` stage and no `OptimizerPass`.
- **0033:** in *Decision Outcome*, a Grainset is answered by a grain only, with
  a pass-through combine; its stand-ins are its grains (ADR 0064), and ADR
  0054's cross-grain Joinsets are not among them in MVP, which loses no answer:
  interface-first is the only authoring direction (ADR 0053), and under it every
  grain of one Grainset maps the same interface, so a cross-grain unit never
  serves a demand that no grain serves; the rule for cross-grain units stays open
  until a narrower grain can be authored.
- **0034:** The Strategies and the rules read the manifest only through the
  read model, and each lookup returns a value the planner owns, never a
  reference to a manifest struct, "so a later graph (ADR 0011) can use its own
  shapes".
- **0042:** the pipeline is parse, validate, compile, plan, adapt, not
  `parse → compile → resolve → plan`.
- **0064:** every Strategy call takes one input shape, built from the
  `Resolution` for the root and from the parent's decision for an operand; the
  driver checks the per-body table of stand-ins; "on a tie the scope root comes
  first" does not apply to a Grainset.
- **0066:** anchors are numbered in pre-order by one walk at the end of `plan`,
  the root 1; `step_empty` writes an empty `VirtualTable` when a Unionset's
  predicate excludes every branch, or when a read without partitions selects no
  object.
- **0067:** a read with discovered partitions is one `FilesScan` over its base
  path with no pattern and no paths; the engine discovers and skips partitions
  and supplies the key columns; the Partitions section's branches are not
  planned, and `paths` is the truth only for a read without partitions; a read
  without partitions that selects no object is the empty read, not a
  `FilesScan`.
- **0068:** its "as it already is for `OptimizerPass`" no longer applies. The
  faithful default and the DataFusion Files rule write a `FilesScan` with no
  paths as one `uri_folder` item, the base path. That default is faithful only
  for a consumer that reads the `key=value` folders under it and supplies the
  key columns, since no Substrait field carries them.
- **0070:** "compile's listing is the only discovery" holds for reads without
  partitions. For discovered partitions, the engine discovers them and supplies
  the key column, and the planner writes no literal per partition.

### Consequences

- Bad, because all bodies share one decision type, so a replacement Strategy
  that breaks the table fails at plan time, not at compile time.
- Bad, because if an engine's partition discovery differs from ADR 0070's
  rules, the engine's discovery decides which rows are read.
- Bad, because a Substrait consumer without the folder convention reads no key
  columns from a folder read. The DataFusion consumer needs an override.
- **DataFusion consumer override:** DataFusion's consumer reads only one `uri_file`, so the e2e
  harness's `consume_read` override must read a `uri_folder` read with its Hive
  partition columns.
- **ADR 0071 (decided first):** this ADR follows it. There is no
  `ScanNode.filter` in MVP, a placed predicate reads named columns, and the
  build order has ADR 0071's row-level projection as stage 2. ADR 0071's
  placement runs top-down in the decide pass, and each Strategy input carries
  the predicates placed on that unit's rows. The Unionset's pruning folds only those,
  and it substitutes a branch's `Literal` signatures only to decide.
- **ADR 0074:**
  it says what each rule is, including the old planner's skip of re-aggregation
  when every branch has its own value for a selected dimension
  (`unionset.rs:67`) and when a Joinset operand is left out. This ADR states
  none of them.
- **Glossary:**
  - the glossary's **Resolver** entry calls the resolver "the first step of
    planning", which clashes with *step*;
  - the glossary has no word for a semantic source's composition body, for "an
    operand not read", or for the Strategy's input ("a slice of a request");
  - the Strategy has no `combine`.
- **Manifest effort:**
  - For a source with discovered partitions, the planner reads the base path,
    the format and the partition keys, and not the listing.
  - The planner needs a partition-fed Dimension to be readable as a column,
    named by its key. That carrier is the manifest effort's, as ADR 0070 already says.
- `crates/` is not changed by this decision. The implementation effort deletes
  the old planner's `optimizer` module along with the old planner.

## More Information

Decided 2026-09-30, and on 2026-10-17 for the
derivation of a frame that gives partial aggregates, and for the demand
of a tree's names (2026-10-17), and for a name that a Metric's formula reads
outside an aggregate and the stages of a Metric's two aggregate steps
(2026-10-18), and on 2026-10-18 for a composed branch and a Grainset
operand.

The decisions of 2026-10-18 read the research on
MetricFlow, Cube, Looker, Malloy and
Calcite, and an oracle on DuckDB and DataFusion 55. The decisions of 2026-10-18
cover the fields
that the row-level projection outputs. They read a probe of a `ProjectRel`'s emit on
DataFusion 55.
