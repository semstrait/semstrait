---
number: 65
title: A semantics has one global definition and one signature per semantic source, and the expression store is a forest of positional nodes
status: accepted
date: 2026-09-21
source:
- CONTEXT.md
- docs/adr/0013-manifest-expression-and-lookup-storage.md
- docs/adr/0027-closed-explicit-authoring-surfaces.md
- docs/adr/0036-bit-position-assignment.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0055-manifest-provenance-is-caller-supplied-and-reverse-indexes-are-not-persisted.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0058-a-reduction-carries-the-authors-keysets-to-the-public-face.md
- docs/adr/0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md
links:
- target: 27
  kind: amends
- target: 36
  kind: amends
- target: 37
  kind: amends
- target: 40
  kind: amends
- target: 43
  kind: amends
- target: 44
  kind: amends
- target: 48
  kind: amends
- target: 49
  kind: amends
- target: 55
  kind: amends
- target: 57
  kind: amends
- target: 58
  kind: amends
- target: 60
  kind: amends
- target: 71
  kind: amendedby
- target: 74
  kind: amendedby
- target: 13
  kind: supersedes
- target: 34
  kind: supersedes
- target: 75
  kind: amendedby
- target: 77
  kind: amendedby
- target: 83
  kind: amendedby
- target: 84
  kind: amendedby
- target: 85
  kind: amendedby
- target: 88
  kind: amendedby
- target: 96
  kind: amendedby
---

# A semantics has one global definition and one signature per semantic source, and the expression store is a forest of positional nodes

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0071](0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md): in D2, a Joinset's composed Measure's filter holds the composition's own view only, as a Unionset's already does, and each operand's condition is read from its own signature. The manifest change is still to build.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): in D2 and the §4d rule, a name that both operands of a Joinset map reads no operand's condition, only the composition's own filter, and is refused with `PlanError::AmbiguousOperandCondition` where the composition has no own filter for it and an operand's signature carries one; in D3 and the §3 rule, a composed Metric's signature is its site's body lowered on the composition, with the composition's own filter, so a composed Metric re-lowers its body; an inlined `Aggregate` node carries the effective additivity of the Measure it came from, with its axes and strategy; in the per-join-type table, where both operands map a name, `Inner`, `Left` and `Full` read `Coalesce` from then to, and `Right` to then from; a constituent serves only the names it maps; in *A Grainset routes and does not read*, "it holds no grain of its own" holds for a Grainset that is no constituent of another composition, and one that is a constituent holds the granularity that it declares and still reads no rows (0074's *Every request is planned against its grain first*).
> - [ADR 0075](0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md): in *ADR 0057's qualifier rule is scoped: a condition names the constituent, a projection names the immediate operand*, a condition names the constituent's output field for a semantics, not a physical column, so "only a concrete constituent holds one — a composite holds none" no longer limits which constituent a condition names: a constituent may be a Unionset, a Grainset or a Joinset that is a child of a cross-grain Joinset; none holds a physical column, and each is named by its own alias; the constituent qualifier for a condition and the immediate operand's qualifier for a projection stand (0075's *A join condition reads each constituent's semantics*).
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *What a tree contains, and who builds it*, a body or a `filter:` that reads a name its constituent exposes and does not map reads a typed NULL on an explicit Unionset, and elsewhere the semantics has no signature on that constituent; in *A composed source's signature names the constituent's projected column*, a named filter's signature on a composed semantic source is its body lowered on that semantic source, not "the same qualified column with no root above it"; in the same section, "A measure on a composed source keeps its aggregate root over the qualified column" holds but for a semantics whose body reads names that no operand of a Joinset maps alone: compile lowers that body on the Joinset itself (0077's *A constituent serves only the names it maps*); in *ADR 0057's qualifier rule is scoped: a condition names the constituent, a projection names the immediate operand*, *A composed Measure's filter is the conjunction of the two views (D2)*, *A composition's own `filter:` reads only what it exposes (D4)* and *One `filter:` predicate, and no `AggregationFilter`*, the composition's own view, where it reads a Dimension that both operands of a Joinset map, is written as 0077's two copies, and each copy reads its operand's column, not the column that the composition projects (0077's *A filter on a composition reads semantics, never a `Coalesce`* and *A constituent serves only the names it maps*).
> - [ADR 0083](0083-a-metric-declares-no-additivity.md): in *A definition is the contract; a signature is the implementation*, "a metric's optional aggregation and optional additivity" reads "a metric's optional aggregation"; in *A role disagreement stays `SemanticKindMismatch`, the other four axes fold to `SemanticShapeMismatch` (D1)*, additivity is an axis for a Measure only; in *A composed Metric carries no filter (D3)*, *Consequences* and *Open questions*, a Metric's "own `agg:` lowering … is out of scope" reads: a row-level formula's `agg:` is built as a Measure's, and a post-aggregate formula's `agg:` is the planner's second aggregate step over the formula's values (0083's rules 1 and 8); in *What a tree contains, and who builds it*, as ADR 0088 changed it, a Metric whose formula reads no Measure and no Metric and that declares `agg:` has a measure's tree, `Aggregate(<its agg:>, <its row-level tree>)` (0083's rule 1); the rest of 0065 stands (0083's *Additivity belongs to an aggregate*).
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *ADR 0057's qualifier rule is scoped: a condition names the constituent, a projection names the immediate operand*, "the operand is itself a reading surface owning exactly one signature per set bit of its own mask" holds for an operand that is a Dataset, a Joinset or a Unionset, and a Grainset operand owns a signature only for each Metric that it computes (0084's rule 8), so a projection's `Column(<Grainset alias>.<name>)` reads the output field that the Grainset's routing fills, or the Grainset's own signature for a Metric that it computes; in *A Grainset routes and does not read*, "`semantic_signature_ids` is **empty by rule**" reads: it lists one signature for each Metric that the Grainset computes and none for any other name, "reads no rows, so there is no source for a signature's tree to name" reads: the tree of a signature that it lists names the Grainset's own alias (0084's rule 8), "`validate()` rejects a Grainset that lists a signature" reads as 0084's rule 10 gives, "stays the folded union of its grains" reads: stays the folded union of its grains and the bit of each Metric that it computes (0084's rule 8), and "a Grainset owns none" and the rejected alternative "Giving a Grainset signatures to hang the attribution off contradicts the rule above" concern the carrier of the keysets, which stays the `keysets` field of `RoleDefinition::Grainset`; in the 0034 entry of *Amends*, "a Grainset is a routing surface that owns none" reads: a Grainset owns a signature only for each Metric that it computes (0084's rule 8); the rest of 0065 stands (0084's *How they are joined* and *A Metric that no child maps whole*).
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): in *A composed source's signature names the constituent's projected column*, the qualified column is the two-part `NamedReference`, not the string `alias.column`. The rest of 0065 stands.
> - [ADR 0088](0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md): its decision "the expression store is a forest of positional nodes", in its title, is replaced by ADR 0088, rules 4 to 6 and 13 to 15, where no id is minted from content, and the sections that only apply it read as those rules and ADR 0088's entry for 0055 give: in *Context and Problem Statement*, "what shape does the persisted expression store take" and "the shape of every expression entry"; in *Considered Options*, the chosen option's "a forest of positional nodes" and the options "**Keeping ADR 0013's flat-tree, content-addressed pool**" and "**A DAG pool with shared subtrees**"; in *Decision Outcome*, the paragraph on the **node store**; the whole of *The pool is a node-store forest with positional ids*; in *A composed Measure's filter is the conjunction of the two views (D2)*, "so it stays disjoint from the projection it filters (positional ids under `(owner, marker, path)`)"; in *A Metric inlining a filtered Measure carries that Measure's predicate on the inlined aggregate (§3)*, the "disjoint" of "the disjoint, row-level second root"; in *One `filter:` predicate, and no `AggregationFilter`*, "the two trees stay disjoint"; in *What `validate()` gains, and the version move*, "the store being a forest — every child id resolves, no node has two parents, no node is an orphan"; in its own *Amends*, the ADR 0013 entry's "a node store of positional entries" and "computed from the subtree", the ADR 0037 entry's "the expressions are persisted as", the ADR 0055 entry's sentence on the premise that "no closure is needed at all", and the ADR 0044 entry's "nothing is interned" and "the orphan sweep closes over child ids"; and in *Consequences*, "**The artifact grows.**", where two sources over one model duplicate every entry, "two `Column` entries", which are two entries, each with its own `Column`, and "a positional id". In its decision "A semantics has one global definition and one signature per semantic source", in *Decision Outcome*, "exactly one complete, fully physical expression tree, an optional row-level filter tree" becomes: exactly one expression entry, whose tree is complete and fully physical once each link in it is followed, and an optional row-level filter entry (ADR 0088, rules 5 and 15). In *Considered Options* and in *A definition is the contract; a signature is the implementation* (**Why not a deferred choice**), the reason given against a handle leaf, that it hides the aggregates beneath it, does not hold for a link: the linked entry's `layer` shows them, and compile binds each link to an entry of its own source, so a link defers no resolution to a request (ADR 0088, rules 5 and 6). In *What a tree contains, and who builds it*, "a **metric**'s tree is its body with each referenced measure or metric substituted" becomes ADR 0088 rule 5's reading: the body with a link to *that same source's* entry for each reference where that rule writes one; and "it is what makes inlining (ADR 0037) cheaper than it looks: the walk happens once per compile, not once per request" becomes: compile walks each reference once, to a link, and a request follows each link by id (ADR 0088, rule 15). In *A Metric inlining a filtered Measure carries that Measure's predicate on the inlined aggregate (§3)*, the predicate is joined where the link is read, not spliced onto a copy (ADR 0088, rule 15). In *What `validate()` gains, and the version move*, "a filter tree that is row-level and disjoint from its expression tree" reads as ADR 0088 rule 14 gives. In its own *Amends*, in the ADR 0037 entry, "Its inlining ruling and its complete-tree requirement are the reason this shape works" becomes: its complete-tree requirement holds once each link is followed (ADR 0088, rule 15), and its inlining ruling gives way to links, as ADR 0088's entry for 0037 gives; and "no second lineage to keep" holds for a tree that links its dependencies, because each link is that edge, inside the tree. The rest of 0065 stands.
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *A Grainset routes and does not read*, the premise "it holds no grain of its own" reads: a Grainset that declares no granularity carries the granularity that its children agree on, and none where they differ (0096's *Shared granularity*), and it still reads no rows; "It groups children holding one subject at different grains so a request can be routed to the grain that serves it" reads: it groups children holding one subject that differ by grain or coverage mask, so that a request can be routed to the child that serves it (0096's *Shared granularity*); the rest of 0065 stands.

## Context and Problem Statement

Where does a semantics' expression live when each semantic source may implement
it differently, and what shape does the persisted expression store take?

The three-part test holds on all
three counts, which is why this is recorded rather than left to the code:

- **Hard to reverse.** Both the collection roster and the shape of every
  expression entry are persisted. `manifest_version` moves 2 → 3 and no
  backward compatibility is owed, so a reversal is a second wire break, not a
  refactor.
- **Surprising without context.** A signature is the concrete thing and a
  definition the abstract one, which inverts the reading a name like
  "definition" invites: the thing a request names is the one that computes
  nothing, and the thing a planner computes from is the one no request can
  name. A reader who has not been told will look for the expression on the
  definition.
- **A real trade-off with a nameable rejected alternative.** The direction
  competed with one global expression tree per semantics carrying a plan-time
  semantic leaf (`Sem(name)`, or a `PhysicalLeaf::ExprRef` handle), resolved per
  source at request time. That is a smaller artifact and a live proposal — see
  the unmerged branch note below — and it was rejected on the glossary's own
  definition of the artifact.

## Considered Options

- A global definition, one signature per semantic source, and a forest of
  positional nodes (chosen).
- **One global expression tree per semantics with a plan-time semantic leaf**
  (`Sem(name)`), resolved per source at request time. The smallest artifact, and
  it defers a resolution the glossary defines this artifact as having finished.
- **A `PhysicalLeaf::ExprRef(id)` handle leaf** as the same idea inside the
  physical roster. ADR 0037 already rejected it on `classify`: aggregates beneath
  a handle are invisible, so a Metric over Measures misclassifies as `Scalar`.
- **The direction on the unmerged `adr/0062-…` branch** — a pooled expression
  that names a semantics, bound per data kind by the override row. It is the
  first alternative with a carrier for the binding, and the carrier is the one
  this decision removes.
- **Category, family or additivity on the signature**, varying per source. It
  breaks shape lock: those are identity-bearing fields, so only *how* a value is
  obtained may vary per source.
- **Definitions only on Dataset leaves**, with a composed source resolving
  through its constituents. Every read of a composite then walks to a leaf, and a
  name two constituents serve has two answers where ADR 0047 gives the composite
  one bit.
- **A definition id minted from tree content.** A definition owns no tree, and
  under this shape the tree differs per source — so the id would differ per
  source for a thing there is one of.
- **`source_id` on the signature beside the owning source's list.** Two
  authorities for one fact, and an integrity check owed forever to keep them
  agreeing.
- **Keeping ADR 0013's flat-tree, content-addressed pool** and adding signatures
  above it. It keeps the migration small and keeps ownership ambiguous: a shared
  entry has no owner to attribute a diagnostic to.
- **A DAG pool with shared subtrees** — content ids plus explicit sharing.
  Strictly smaller than a forest, and it makes every per-node fact
  ("which source reads this", "which tree is this in") a search rather than a
  read.

## Decision Outcome

The manifest holds two carriers for one semantics where it used to hold a
definition and an override row. A **semantic definition** is the global,
request-facing contract — name, data type, role with its shape-locked families,
`bit_position`, documentation — and owns no expression and no keysets. A
**semantic signature** is one concrete implementation of that contract on one
semantic source: it owns exactly one complete, fully physical expression tree, an
optional row-level filter tree, and that source's keyset attribution. Every
reading source lists one signature per set bit of its mask and none for an unset
bit, and the mask is derived from the signatures rather than held beside them.

The expressions those signatures name are persisted as a **node store**: one
entry per operator or leaf, children addressed by expression id, every tree
disjoint, and ids minted from position under an owner rather than from content.

### A definition is the contract; a signature is the implementation

`SemanticDefinition` is one entry per name: `data_type` (absent for a named
filter), the role with its shape-locked families — a dimension's category, a
measure's mandatory aggregation and optional additivity, a metric's optional
aggregation and optional additivity, a filter's none — `bit_position` under
ADR 0036 unchanged, and `description` / `ai_context`. It owns no expression, no
keysets, and no source reference. The `Field` role and its minter are removed:
the roster is Dimension | Measure | Metric | Filter, which is the queryable
vocabulary `CONTEXT.md` **Semantics** names and nothing else.

`SemanticSignature` is `{ id, definition_id, expression_id, filter_expr_id,
keysets }`. `expression_id` is **mandatory** — every covered semantics emits, a
bare column and a broadcast literal no less than an aggregate (ADR 0037) — and
`filter_expr_id` is present only for a Measure or Metric that authored a
`filter:`. The id is minted over `(semantic source id, definition id)`, in the
namespace shape the override row's mint used, so the migration changes what the
pair means and not how a pair becomes an id.

**A signature carries no `source_id`.** The owning `SemanticSource`'s
`semantic_signature_ids` is the one authority for ownership, and a field on the
signature would be a second one to hold in agreement forever — the same argument
ADR 0055 makes against a persisted reverse index. `validate()` therefore states
the rule from the list's side: exactly one source lists any signature.

**The invariant is a biconditional per source**: exactly one signature per set
bit of the source's `semantic_bitmask`, and none for an unset bit. Masks are
computed from the signatures after the seal loop, so a set bit cannot outlive the
implementation that justified it — which is what ADR 0047's serveability reading
of a bit asserts, now with a carrier that proves it. Every **reading** source
owns signatures: a Dataset, a Joinset including ADR 0054's implicit
sub-compositions and the cross-grain units a Grainset emits, and a Unionset.

**Why not a deferred choice.** `CONTEXT.md` **Semantic manifest** defines the
artifact as the one "in which every name, source, and expression has been
resolved and denormalized, so planning needs no further lookup or I/O". A leaf
that names a semantics and is bound per source at request time is therefore a
category error in this artifact and not merely a slower shape: it moves a
resolution the manifest exists to have finished into the consumer, and it does it
in the one place ADR 0037 already showed the cost of — an opaque handle leaf
hides the aggregates beneath it, so `classify` stops being a total function of the
tree.

### What a tree contains, and who builds it

Compile builds the aggregate root from the definition's family, so the shape of a
tree is a function of the role and never of what the author typed inside `expr:`:

- a **measure**'s tree is `Aggregate(family, <row-level tree>)`;
- a **dimension**'s tree is its row-level tree;
- a **metric**'s tree is its body with each referenced measure or metric
  substituted by *that same source's* tree for the referenced name.

Authors express aggregation only through `agg:`. ADR 0027's `agg:`-only clause
stands and is reinforced rather than touched: the aggregate root now has a single
builder, so an inline aggregate inside `expr:` would produce a second one.

A body referencing a name the source does not expose is a **compile error**
naming the semantics and the source, never a null fill — null fill is the absence
of a realization (ADR 0059) and has no authored subject here. Reference cycles
are refused **per source**, because a reference site may defer its own `expr:`
and so two sources exposing the same names need not have the same graph.

Compile does the whole dependency walk; the planner does id lookups only. That
split is the point of the artifact, and it is what makes inlining (ADR 0037)
cheaper than it looks: the walk happens once per compile, not once per request.

### The pool is a node-store forest with positional ids

One entry is one operator or leaf node. Children are expression ids, and a
signature's `expression_id` is its root node, so a tree is a root plus the
closure over child ids. `layer` stays on the entry, computed over that node's
subtree, and `physical_sources` stays. The leaf roster stays Column | Literal |
Parameter. `depends_on` is removed. There is no interning and no dedup anywhere.

**Every tree is disjoint.** An expression id belongs to exactly one owner — a
signature, or the Joinset source for a join condition or its residual; two owners
with byte-identical trees get two sets of entries; and a subtree repeated within
one tree is one entry per position. Ids are minted from `(owner id, root marker,
position path from the root)` and never from content, so the same bytes under two
owners are two ids and an unchanged model still compiles byte-identically. The
root marker is what lets one owner hold several roots whose paths would otherwise
collide.

Worked example, from the RFC:

```text
signature gross_margin @ eu   expression_id = #0
  #0  Subtract(#1, #3)
  #1  Aggregate(Sum, #2)
  #2  Column(amount)
  #3  Aggregate(Sum, #4)
  #4  Column(expense)

signature gross_margin @ us   expression_id = #5
  #5  Subtract(#6, #8)
  #6  Aggregate(Sum, #7)
  #7  Column(total)
  #8  Aggregate(Sum, #9)
  #9  Column(cost)
```

(The RFC prints the two trees side by side; they are stacked here only to keep
the line width, and no node, id or order differs.) Had `us` also read a column
named `amount`, `#7` would still be distinct from `#2`.

**A signature's filter tree is its own disjoint set of nodes**, minted under the
same signature with a distinct root marker; it shares no entry with the
expression tree even where the same column appears in both. Join conditions and
residuals are emitted the same way, under the Joinset source.

**Why positional and not content-addressed.** Content ids buy sharing, and
sharing is what makes a node's owner ambiguous: two signatures reaching one entry
leave no answer to "which source is this node read for", so a per-node diagnostic
has no single home and the orphan sweep cannot say which referrer kept a node
alive. Positional ids make ownership a property of the id itself, and the price —
a larger artifact, with no sharing at all — is accepted.

### A composed source's signature names the constituent's projected column

A Joinset's or a Unionset's signature names the constituent's **projected
column**, qualified by its source alias in ADR 0057's wire form `alias.column`,
and never a copy of the constituent's own tree. Copying would duplicate a
decision that already has one home, and would put a physical column above the
projection that turns a source into its semantic interface — which `CONTEXT.md`
**Column** forbids.

Where one side exposes the name, the tree is `Column(<that alias>.<name>)`.
Where both do:

| Join type | Tree |
|---|---|
| Inner | `Column(<from alias>.<name>)` |
| Left  | `Column(<from alias>.<name>)` |
| Right | `Column(<to alias>.<name>)` |
| Full  | `Coalesce(<from alias>.<name>, <to alias>.<name>)` |

A Unionset's is `Column(<union alias>.<name>)`. A measure on a composed source
keeps its aggregate root over the qualified column, because the family is the
definition's and does not change with the surface. A **Metric** and a **named
filter** are the same qualified column with no root above it: each one's body was
lowered on the constituent that exposes it, so the composition reads the column
the constituent projects — the same rule as for a Dimension, and not a second one.

This supersedes ADR 0057's section *"A Joinset coalesces a semantics two
constituents serve"* on one clause only: *"Where one constituent serves the
semantics, nothing is minted. Absence means resolve through that constituent,
which keeps an override optional exactly as a self-contained `semantic_definition`
allows."* Absence is no longer representable — every set bit has a signature — so
there is nothing for a consumer to resolve through. The `coalesce(from, to)`
shape that section rules, its fixed `from`-then-`to` argument order, and its
reasoning about agreement between two sides are all untouched.

### ADR 0057's qualifier rule is scoped: a condition names the constituent, a projection names the immediate operand

ADR 0057's section *"The qualifier names the constituent, not the immediate
operand"* governs **join-condition leaves** and nothing else. A condition names a
physical column, and only a concrete constituent holds one — a composite holds
none, so qualifying a condition leaf by the immediate operand would name a column
no schema has.

A composed source's **signature projection** is qualified by the source alias of
the **immediate operand**. The ambiguity ADR 0057 argues from — that a reference
into `(A, B)` has two answers where `A` and `B` both serve one name — does not
arise above this pass: the operand is itself a reading surface owning exactly one
signature per set bit of its own mask, so `(A, B).customer_id` has exactly one
answer, and the planner resolves it one level at a time by id lookup rather than
descending into the constituent. Qualifying a projection by a constituent instead would need a rule saying
which constituent inside `(A, B)` serves a name both expose, and there is none.

The same rule qualifies the column leaves of the **own** half of a composed
signature's **filter** tree: a composition's own predicate reads what that
composition projects. The constituent half of that linkage is qualified one level
lower, by the source that owns the columns its leaves name (D2 below).

#### A composed Measure's filter is the conjunction of the two views (D2)

A composed **Measure**'s signature carries the `And` linkage of two views, the
composition's own on the left:

- the composition's **own** view — its interface-site `filter:` where it states
  one, else the pooled predicate, lowered over the columns the composition
  projects;
- the **constituent's** — the filter tree of each operand whose column the
  projection reads, rebuilt from the pool and qualified so every column leaf names
  the relation whose rows that predicate selects.

A view absent leaves the rest alone; all absent leaves `filter_expr_id: None`.

**Which operands contribute a half.** The ones the projection rule already picked:
the preserved side of an `Inner`, `Left` or `Right` join, or the single side that
serves the name. A **`Full`** join over two serving sides preserves both and its
projection coalesces both columns, so **both** contribute and the linkage is
`own AND from' AND to'` in the join's own from-then-to order — dropping either
would persist a predicate weaker than that side states, and picking one would be a
claim about which side a NULL came from that the coalesce exists to refuse. A
**Unionset** contributes **none**: a stack projects its own alias rather than a
branch's, and every branch's own signature already carries that branch's
predicate, so the rows the stack reads are each branch's *filtered* rows and a
restatement here would select them twice. Rejected for the stack: re-qualifying
every branch's predicate into a disjunction over the stack's own columns, which is
a claim about which branch a row came from that a stack's single alias cannot
make.

**Every leaf of a constituent's half is qualified exactly once, by the level that
owns the columns it names.** The half is the operand's tree as the operand wrote
it, not the operand's predicate re-lowered over the composition's columns:

- a **concrete** operand (a `Dataset`) holds a physical-source-attached tree whose
  leaves are bare physical column names, and they gain that operand's source alias
  — so a leaf reads `<operand alias>.<physical column>`, `orders.region_code`
  where an explicit `semantic_mapping` maps `region` onto `region_code`. That is
  the join condition's leaf reading (ADR 0057) and this is its second producer, so
  **the two halves of one linkage may spell one semantics two ways on purpose**:
  the left names what the composition projects, the right names the columns the
  operand's own rows are selected by;
- a **composite** operand holds a composition-attached tree already qualified by
  the source whose columns its leaves name, so it is conjoined **unchanged**.

Two alternatives are recorded because each is the reading a reader expects.
Re-lowering the constituent's predicate over the composition's projected
semantics columns, so both halves read one convention: rejected because it needs
the physical→semantics inverse of the leaf's own lowering, and there is none — a
realisation may be a computed body, and a predicate may read a column no exposed
name names. Adding the immediate operand's alias on top of a composite operand's
already-qualified leaves, giving `<outer>.<inner>.<name>`: rejected because
`split_qualifier` cuts at the first dot (ADR 0057) and reads that as the column
name `<inner>.<name>`, which no relation projects.

**The conjunction is redundant today.** Under the interface-first direction a
nested constituent carries no interface of its own and is planned with the
composition's, so both views lower the same authored predicate and every composed
Measure filter reads `P AND P'` — one predicate over two vocabularies, which the
compile-seam test `a_composed_measure_conjoins_its_own_view_with_its_constituents`
asserts literally. The linkage is ruled all the same because it is load-bearing
the moment a constituent can state a filter the composition does not; until then
it adds no information, and a reader should not take it to mean the rule itself
is redundant.

The conjunction is the reading because the predicates select rows of different
relations: an operand's narrows the rows it emits, and the composition's narrows
the rows it reads out of them. A **divergence** between the views is therefore
not an authoring fault and not a narrowing to detect — it is the conjunction, and
compile compares nothing across the two views.

The linked tree is emitted under the `SignatureFilter` marker, so it stays
disjoint from the projection it filters (positional ids under `(owner, marker,
path)`), and the re-qualification reuses the join condition's own leaf walk rather
than a second structural walk.

#### A composed Metric carries no filter (D3)

A composed **Metric**'s signature carries `filter_expr_id: None`, whether the
`filter:` is authored on the pooled declaration or on the composition's own
reference site. A Metric's realisation is a lowered body one level down and its
own `agg:` lowering is out of scope, so there is no fold here for a predicate to
select the rows of. The predicate is therefore skipped rather than lowered and
discarded, so no composed Metric is refused over a predicate no artifact would
carry.

This ruling is about the *composition's own* predicate and nothing else. What a
Metric's *inlined Measures* carry is a separate matter, ruled next.

#### A Metric inlining a filtered Measure carries that Measure's predicate on the inlined aggregate (§3)

A Measure's `filter:` is a second root on the Measure's *own* signature, not a
slot on the `Aggregate` node its expression tree carries. Splicing "that same
source's tree for the referenced name" therefore starts from a `filter: None`
aggregate, and doing nothing more would leave the predicate behind: `revenue`
declared `sum` filtered to `status = 'paid'` would reach a metric
`revenue + revenue` as
`Add(Aggregate(Sum, Column(revenue)), Aggregate(Sum, Column(revenue)))`, both folds
unfiltered, with `depends_on` removed and so no id lookup to recover the predicate
by either.

**The fix is to splice the referenced Measure's own predicate onto the inlined
`Aggregate.filter` slot, once per reference, at the point of substitution** — not
to attach one combined predicate to the Metric's signature, and not to refuse the
model (the two named alternatives). Each `Aggregate` node a Metric's body composes
keeps the predicate of the Measure it came from: `revenue` above reaches
`Add(Aggregate(Sum, Column(revenue), filter: status = 'paid'), Aggregate(Sum,
Column(revenue), filter: status = 'paid'))`, and a Metric composing `paid + pending`
— two Measures, two different `filter:`s — keeps each at its own fold rather than
collapsing them: the two predicates belong to two different aggregates, and
collapsing them changes what either one selects. A Metric composing an
*unfiltered* Measure copies a `filter: None` aggregate exactly as before this
ruling.

The Measure's *own* signature is untouched by this: its `filter_expr_id` stays the
disjoint, row-level second root the rest of this ADR states — what changes is only
the *copy* a referencing Metric's splice produces. And a composed Metric never
re-lowers a body (D3 above) — it projects the constituent's already-sealed column
— so this ruling has one implementation site, the leaf body walk
(`compile::expressions::bodies::lower_source`,
`compile::expressions::lower::substitute_semantic_leaf`), and a composed Metric
inherits it for free through the column it reads.

**The rule is transitive through a Metric-referencing-Metric chain.** A Metric `B`
may reference another Metric `A` (`bodies.rs` keeps `A` before `B` in the same
topological order a Measure reference relies on), and `A` may itself carry a
lowered `filter:` for this source. Inlining `A`'s tree into `B` carries `A`'s own
predicate too, by the same fix and for the same reason: `A`'s tree is a copy,
`A`'s `filter:` is a second root, and doing nothing would leave `A`'s predicate
behind exactly as doing nothing left a Measure's behind above. The splice conjoins
`A`'s predicate onto **every** `Aggregate` node the inlined copy of `A` carries,
`And`-ed onto whatever the slot already holds rather than replacing it — a slot a
Measure's own splice already filled (the case above, one level down) keeps that
predicate and gains `A`'s beside it; an empty slot gets `A`'s predicate alone.
`A`'s own signature is untouched by `B`'s splice, exactly as a Measure's is: only
the copy changes. Ruled.

#### A composition's own `filter:` reads only what it exposes (D4)

A composed source's own `filter:` may reference only names the composition's
interface exposes; anything else is `CompileError::UnexposedReference`, per
source, exactly as on a leaf. The composition's predicate is lowered over the
columns it projects, and it projects one column per exposed name — a reference to
an unexposed name has no column to splice.

#### A composed source's `filter:` may not author a physical column, refused rather than defined (§4)

A composed source's own `filter:` names a **semantics**, exactly as its expression
tree does — a bare `column: x` leaf is refused as
`CompileError::ColumnInComposedFilter` rather than lowered. A composition owns no
physical realisation of its own (the previous section's rule already implies this:
what the predicate splices is a *projected* column, never a source's own), so a
`Column` leaf names no operand this ADR's projection rule can qualify.

**The refusal scopes to the composition's own *authored* `filter:` — never to a
pooled predicate it only inherits (§4d).** D2 above re-lowers the pooled
Measure declaration's predicate on *every* composition exposing that Measure,
whether or not the composition's own ref site states a `filter:` at all — and a
bare `Column` leaf is perfectly legal on the Dataset the pooled predicate was
written for. Refusing that inherited half would make an author's Dataset-only
predicate uncompilable through every composition exposing the Measure, though the
composition's author wrote no `filter:` of their own. So the refusal fires only
for a name the composition's own ref site states a `filter:` for; a pooled
predicate's bare `Column` leaf is **omitted** instead — that Measure's own half is
simply absent from the composition's linkage (D2's `own AND constituents'` fold
then carries only the constituents' halves, or nothing at all if there are none,
e.g. a Unionset). This loses no predicate: every constituent leaf source's own
signature already carries the pooled predicate lowered over *that* source's
physical columns (D2), so the rows it selects are already excluded by the
constituent halves' own conjunction.

**The workaround, when a composition needs its own predicate over a physical
column the pool cannot express relative to it:** author a `filter:` on the
composition's own ref site. `filter_view`'s site-vs-pool split makes a ref-site
`filter:` *replace* the pooled predicate for that source (D4's rule, restated: "a
ref site's `filter:` replaces the pooled predicate for that source only"), so a
site-authored `filter:` is under this section's refusal exactly as before — there
is no pooled half left to fall back to once the site states its own, and a bare
`Column` leaf in it still names no operand this ADR's projection rule can qualify.

**Rejected: defining a meaning instead of refusing.** A bare column would need an
operand to read it from, and nothing here picks one — not the immediate operand (a
bare name authors no semantics, so there is no `ProjectedColumn` a projection rule
could consult) and not a fixed side (a `Full` join has two, and coalescing or
picking one is a new rule this ADR does not otherwise need). Assigning either
reading would be qualification inferred from where the leaf appears rather than
from the name itself, which the ADR 0057 qualifier rule already forbids for a
condition leaf and this ADR never relaxes for a projection. Refusing the
composition's own authored leaf is also the cheaper of the two by every measure:
no new rule for `Full`, and no change to what a schema check reads — the one field
the lowering context gains (`LoweringCtx::composed_filter_unit`) is what buys the
§4d scoping itself, not the refusal's existence.

#### A role disagreement stays `SemanticKindMismatch`, the other four axes fold to `SemanticShapeMismatch` (D1)

Two occurrences of one semantics name disagreeing on **role** — one site's
`measure("x")` against another's registered Dimension — keeps raising the named
error `CompileError::SemanticKindMismatch { authored_kind, registered_kind,
name }`, unfolded into `SemanticShapeMismatch`'s `axis` vocabulary (ruled). The other four axes
— `data_type`, a Dimension's category, a Measure's or Metric's aggregation
family, and additivity — raise `CompileError::SemanticShapeMismatch { axis, ..
}`, one error shape carrying which axis disagreed.

Role is not a fifth `ShapeAxis` variant because the two errors already differ in
what they name: a shape mismatch is one declaration re-stating itself
differently on an axis both occurrences agree the definition *has*, while a role
disagreement means the two occurrences do not agree the definition has that axis
at all — a Dimension has no aggregation family to compare against a Measure's.
Folding role into `axis` would need a placeholder value on the side that has
none, which restates the same asymmetry `SemanticKindMismatch`'s own two typed
fields already carry for free.

### A Grainset routes and does not read

A Grainset's `semantic_signature_ids` is **empty by rule**. It groups children
holding one subject at different grains so a request can be routed to the grain
that serves it (`CONTEXT.md` **Grainset**); it holds no grain of its own and
reads no rows, so there is no source for a signature's tree to name. Its
`semantic_bitmask` stays the folded union of its grains, which is exactly
ADR 0047's serveability claim read for routing, and `validate()` rejects a
Grainset that lists a signature.

**It carries its own keysets all the same** (D5, ruled). The Grainset semantic
source holds

```rust
keysets: Vec<EntityId>
```

on `RoleDefinition::Grainset` — ids into `manifest.keysets`, the same carrier
shape as `SemanticSignature::keysets`, sorted and deduplicated. Compile populates
it from the author's declaration after ADR 0048's intersection-boundary check and
from nothing else — a Grainset derives no keyset (ADR 0048): the ids are exactly
what the keysets stage guaranteed for the Grainset's own source (`KeysetPlan::
guaranteed`), which for a Grainset are exactly the authored groups since it
derives nothing. `validate()` checks two things over it: every id resolves in
the roster, and the list is sorted and deduplicated.

The reason there is a field at all is that ADR 0048 gives a composite's keysets
one carrier — the composite's own signatures — and a Grainset owns none. Its
author's declaration was boundary-checked and then reached no artifact, so a fact
the model states was silently dropped for exactly one form. The shape is a bare
id list rather than `SemanticSignature::keysets`' per-definition map, and that is
a knowing loss: **membership is not recoverable from this carrier.** A bare id
says which keysets the family carries, not which Dimension attributed which
id, so "the members of keyset K" stops being derivable for a Grainset the way it
is for every reading source, and ADR 0040's role-Dimension and served-bit checks
have no subject to run against — `validate()` asks only roster resolution and
set-ness over this carrier.

Two alternatives were rejected. The per-definition `BTreeMap<EntityId,
Vec<EntityId>>` this ADR carried between R2 and D5 keeps membership
derivable, at the cost of a second per-appearance shape for a surface that owns no
signature and therefore no per-appearance site to key it by — the map's key was
always the family's own interface, restating one fact `semantic_bitmask` already
carries, for no reader this codebase has. Giving a Grainset signatures to hang the
attribution off contradicts the rule above — it would make a routing surface a
reading one — and ADR 0048's Grainset clause, which has the family declare
keysets without reading rows.

### One `filter:` predicate, and no `AggregationFilter`

Measure, Metric and their reference sites carry `filter: Option<Expr>`. Two
conditions are authored conjoined. `filters: Vec<AggregationFilter>` and the
`AggregationFilter` carrier itself are retired, `description` / `ai_context`
included: a filter is a bare predicate and carries no documentation of its own,
because documentation belongs to the measure or metric it qualifies (ruled). The retired
`filters:` key is rejected by the ordinary unknown-field diagnostic; there is no
dedicated "retired key" message, so the authoring surface gains no vocabulary for
its own history.

It lowers to `signature.filter_expr_id`, and the tree is **row-level**: no
aggregate node and no measure or metric reference anywhere in it. A site filter
**replaces** the pooled one for that source rather than conjoining with it, on
ADR 0039's reading of what a site states.

A **composed** source's Measure signature carries the `And` linkage of the
composition's own predicate — its site `filter:` where it states one, else the
pooled one, over the composition's *projected* columns — with the filter tree of
every operand whose column the projection reads, each qualified exactly once
(D2 above). The projected `<alias>.<name>` leaves its **own** half's semantic
references splice are therefore qualified by the immediate operand exactly as the
signature's expression tree is (an authored bare `Column` leaf in that half names no
operand this projection rule can qualify, and is refused rather than qualified, per
the §4 section above), while a concrete operand's half names that operand's physical
columns; the two trees stay disjoint because the predicate is emitted under the
`SignatureFilter` marker. A composed Metric's signature carries none (D3 above). A
predicate that references a name the composition's interface does not expose is
refused as `UnexposedReference`, per source, as it is on a leaf (D4 above).

### What `validate()` gains, and the version move

`validate()` gains the artifact-shape half of the invariants above, for the
subject no stage can reach — decoded bytes, or an artifact whose `pub` fields
were mutated after compile:

- set bits ⇔ listed signatures per source;
- every signature listed by exactly one source;
- every definition id and expression id resolving;
- a filter tree that is row-level and disjoint from its expression tree;
- the store being a forest — every child id resolves, no node has two parents,
  no node is an orphan;
- a Dataset's signature trees — expression and filter alike — naming that
  Dataset's own physical relation among their consuming sources, since a leaf's
  realisation is lowered against the one relation the leaf reads.

The override-row checks are gone with their subject. `manifest_version` moves
from 2 to 3, and no backward compatibility is owed.

### Amends

- **0013:** Superseded on the pool's shape and on id derivation: the
  content-deduplicated flat-tree pool addressed by content becomes a node store
  of positional entries, and "the same `PhysicalExpr` may be stored twice" stops
  being an accepted anomaly and becomes the rule. Its structural `ExprLayer`
  survives, on the entry, computed from the subtree. One semantic signature per
  semantic source and definition owns exactly one complete, fully physical
  expression tree; the Joinset source owns its join condition or its residual;
  the `Field` role and its minter are removed.
- **0037:** superseded on the `depends_on` half only. Its inlining ruling and
  its complete-tree requirement are the reason this shape works, and they and
  its `ExprLayer`-on-the-entry decision stand; the recorded semantic edges go,
  because a tree that inlines its dependencies and is owned by one source has no
  second lineage to keep, and the expressions are persisted as a node store with
  "no interning and no dedup anywhere".
- **0055:** the `depends_on` direction clause is **void**, its subject removed.
  So is the premise that "no closure is needed at all" for the orphan sweep,
  which rested on compile interning only roots: the sweep now closes over child
  ids, because a stored node is a position inside a tree and not a root. The
  reverse-index rule — no persisted name → bit and no persisted expression →
  consumers — stands, and ADR 0065's refusal of a `source_id` on the signature
  applies it.
- **0049:** the successor invariant *"exactly one override row per
  `(semantic, data_kind)`"* is restated as **one signature per
  `(semantic source, definition)`**, and made total: the optional case it
  allowed is gone.
- **0057:** its successor invariant, ADR 0049's *"exactly one override row per
  `(semantic, data_kind)`"*, is restated as **one signature per
  `(semantic source, definition)`**, and made total: the optional case it
  allowed is gone, so "nothing is minted" is superseded; the constituent
  qualifier governs join-condition leaves only, and a signature projection is
  qualified by the immediate operand.
- **0058:** ADR 0049's successor invariant *"exactly one override row per
  `(semantic, data_kind)`"* is restated as one signature per
  `(semantic source, definition)`, and made total; the carry-over invariant
  reads against signatures for the same reason, and its rule is unchanged.
- **0027:** superseded on its `filters` clauses only: the clause admitting
  *"only `expr`, plus `filters` for Measure/Metric, [to] vary at a reference
  site"* now reads `filter`, and the clause that *"`DataKindFilter` and
  `AggregationFilter` are disjoint types with no shared name table"* loses one
  of its two subjects (the `AggregationFilter` carrier is retired). Everything
  else in 0027 stands, `agg:`-only aggregation included — that clause is
  reinforced in 0065's *What a tree contains, and who builds it*. 0027 is
  **not** amended in place.
- **0036:** the `Field` role and its minter are removed (the roster is
  Dimension | Measure | Metric | Filter), so `Field` will not join it either;
  `bit_position` stays under 0036 unchanged, and 0036's bit-position assignment
  gains a carrier that proves what it asserts.
- **0034:** Every reading source, a Joinset and a Unionset alike, owns one
  semantic signature per name it serves, and a Grainset is a routing surface
  that owns none.
- **0040:** `manifest_version` moves 2 → 3, and no backward compatibility is
  owed.
- **0043:** the override row becomes a signature, one per
  `(semantic source, definition)`:
  `{ id, definition_id, expression_id, filter_expr_id, keysets }`, with no
  `additivity` or `grains`; a `filter:` lowers to its own row-level tree, and
  `depends_on` is removed; `manifest_version` moves from 2 to 3, and no backward
  compatibility is owed.
- **0044:** a definition owns no expression and a signature replaces the
  override row; nothing is interned, `depends_on` is removed, and the orphan
  sweep closes over child ids.
- **0048:** a composite's keysets ride its own semantic signatures, not
  `semantic_overrides` rows, and a definition owns no keysets; a Grainset
  carries its authored keysets on `RoleDefinition::Grainset`.
- **0060:** the override row gives way to a signature carrying `expression_id`,
  `filter_expr_id` and `keysets`; 0065's D1 checks shape lock at compile — a
  role disagreement keeps raising `SemanticKindMismatch`, and a disagreement on
  `data_type`, a Dimension's category, a Measure's or Metric's aggregation
  family, or additivity raises `SemanticShapeMismatch`.

**ADR 0039** — unchanged as a rule. Keyness stays a per-site attribution; only
its carrier moves, from the override row to the signature.

**ADR 0047** (the mask is a serveability statement) and **ADR 0059** (the
three-way coverage axis, read against a composition) are **unchanged**. The
first gains a carrier that proves what it asserts; the second is why an
unexposed reference is an error rather than a null fill.

**A draft ADR on semantic references in the expression pool**
proposes the opposite direction — a pooled expression naming a semantics, bound
per data kind through the override row — and is **superseded before merge**:
its carrier no longer exists and its leaf is the first rejected alternative
above. It is to be adapted or withdrawn by its author; nothing here edits or
closes that draft. Merged `docs/adr/0062-a-request-states-intent-never-placement.md`
is a different ADR on a different subject and is untouched.

### Consequences

- The planner's read path becomes `request → bitmask → source → signature ids →
  trees`, by lookup alone. That is what the artifact enables; consuming it is
  downstream work in `semstrait-planner`, which does not build today.
- **The artifact grows.** No sharing means one entry per position per owner, and
  two sources over one model duplicate every node. Accepted: the artifact is
  written once per compile and read many times, and ownership is worth more than
  bytes.
- `coverage_of` reads the named source's **own** signature for the semantics and
  follows it into the store, so the derivation is total over the artifact without
  descending into any constituent.
- A metric's own `agg:` lowering — an aggregate over aggregates — is out of scope
  and unbuilt (see Open questions).
- Two sources mapping one name to the same physical column yield two `Column`
  entries. That is the intended reading of a positional id, not a leak.

## More Information

### Open questions

Each is recorded here because the code has a live behaviour
for it and no ADR ratifies that behaviour. None is decided by this ADR.

- **A Metric's own `agg:` lowering** — an aggregate over aggregates — is out of
  scope and unbuilt.

### History of the §4 refusal

The §4 refusal was implemented once
and reverted
because the refusal was enforcing a ruling this ADR did not yet state — not because
refusing was the wrong call. The §4 section is that ruling, corrected once more (§4d)
after a first pass (§4)
refused the pooled half too and made every composition exposing a
Dataset-filtered Measure uncompilable though its own author wrote no `filter:` at all.
