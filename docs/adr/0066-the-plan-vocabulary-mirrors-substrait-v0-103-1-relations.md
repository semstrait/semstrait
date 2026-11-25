---
number: 66
title: The plan vocabulary mirrors Substrait v0.103.1 relations
status: accepted
date: 2026-09-23
source:
- CONTEXT.md
- docs/adr/0015-refuse-ambiguity-name-order-selection.md
- docs/adr/0018-canonical-plan-ir-contract.md
- docs/adr/0019-substrait-outbound-only.md
- docs/adr/0022-engine-independent-canonical-vocabulary.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0052-compile-gates-the-cardinality-it-materialises.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0067-the-scan-carries-its-read-identity.md
- docs/adr/0068-adapters-are-layered-format-first.md
- Substrait `proto/substrait/algebra.proto` at tag v0.103.1
- 'evidence: gap matrix, DataFusion consumer, round-trip, aggregation phase'
links:
- target: 22
  kind: amends
- target: 38
  kind: amends
- target: 50
  kind: amends
- target: 52
  kind: amends
- target: 57
  kind: amends
- target: 61
  kind: amends
- target: 73
  kind: amendedby
- target: 74
  kind: amendedby
- target: 18
  kind: supersedes
- target: 75
  kind: amendedby
- target: 82
  kind: amendedby
- target: 87
  kind: amendedby
- target: 90
  kind: amendedby
---

# The plan vocabulary mirrors Substrait v0.103.1 relations

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): anchors are numbered in pre-order by one walk at the end of `plan`, the root 1; `step_empty` writes an empty `VirtualTable` when a Unionset's predicate excludes every branch, or when a read without partitions selects no object.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): in ruling 7, `Avg` is non-additive and is not split into `Sum` and `Count`; in ruling 4, a `JoinNode` also joins two subtrees of one frame, on the condition that 0074's *A Metric* writes (`IsNotDistinctFrom` on each grouping column), with no relationship and no cardinality.
> - [ADR 0075](0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md): in ruling 4 (the join) and in *Consequences*, the binding rule is 0075's *Every output name is the one its parent reads* and *Names are unique, and the planner's own names are reserved*; the rest of 0066 stands.
> - [ADR 0082](0082-a-physical-read-types-an-unobserved-column-by-the-semantics-mapped-to-it-and-compile-writes-that-type-into-the-schema.md): on a model-fed schema, ruling 5's branch "else the definition's `data_type` where the Dimension's tree *is* a bare `Column` (result and leaf are one value)" reads: the column's type in the schema, which compile writes from the semantics mapped to the column as 0082's *Which semantics types the column* gives, whatever their roles; `PlanError::UnresolvedType` remains for a column that no semantics with a `data_type` is mapped to. In *Amends*, the entry for ADR 0061 reads with 0082's entry for 0061. The rest of 0066 stands.
> - [ADR 0087](0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md): in its ruling 2, "grows only by ADR, as before", said of the expression vocabulary, becomes: its node variants grow only by ADR, as before, and a function enters by its declaration under ADR 0087's rule 1. The rest of 0066 stands.
> - [ADR 0090](0090-semstrait-ir-keeps-the-plan-and-its-parameter-and-the-manifest-owns-the-expression-id-and-the-expression-layer.md): in *Rulings*, ruling 5's "`SchemaColumn.data_type` / `nullable` and `AggregateExpr.inferred_type` are mandatory, on every `output_schema` and on `ScanNode.schema`" becomes: `SchemaColumn.data_type` / `nullable` are mandatory, on every `output_schema` and on `ScanNode.schema`, and an aggregate's type is its column on the `AggregateNode`'s `output_schema` (ADR 0090, rule 2). Ruling 8's "`AggregateExpr` keeps its `filter` slot as the mirror of `AggregateRel.Measure { measure, filter }`" becomes: an `AggregateNode` entry's call keeps its `filter` clause as the mirror of `AggregateRel.Measure { measure, filter }` (ADR 0090, rule 2). The rest of 0066 stands.

## Context and Problem Statement

Semstrait plans at the logical level, in nodes of its own. Which Substrait
release does each plan node mirror, and who owns the gap to the release an
engine reads?

## Considered Options

- A mirror of v0.103.1's relations (chosen).
- **A full mirror** — every Substrait value and slot now (twelve join types,
  eight set operations, grouping sets). Refused: a value no planner step
  writes is a hole a consumer falls into, and the mirrored layout lets it enter
  later with no reshape (ruling 2).
- **Variants only where a semstrait concept maps onto them**, with node shapes
  of semstrait's own. Refused by the human: the nodes are semstrait's
  implementation of Substrait's relations, and a mirrored slot has no
  trade-off to record.
- **Mirroring the release the stack reads (v0.85) or the newest a Rust
  producer can write (v0.102).** Refused: the IR is hand-written and links no
  Substrait crate, so it mirrors the latest release at no dependency cost, and
  a release gap is the adapter's.
- **A cross join as `JoinType::Cross`** on a `JoinNode` with no condition.
  Refused: Substrait writes it as `CrossRel`; one relation, one node.
- **An optional `JoinNode.on`.** Refused: beside `CrossNode`,
  `JoinNode { Inner, on: None }` is a second spelling of a cross join, and no
  manifest concept writes a conditionless outer join.
- **`JoinNode.cardinality`.** Refused: `JoinRel` has no slot for it, and
  nothing that reads the plan needs it.
- **A relationship filter as a `FilterNode` above a keyed join.** Refused:
  Substrait carries it on the join as `post_join_filter`; an engine that
  rejects the slot rewrites it in its rules (ADR 0068).
- **One node per set operation** (`UnionNode`, `IntersectNode`,
  `ExceptNode`). Refused: one relation, one node, and `op` maps by table.
- **Expression-valued `FetchNode` bounds** checked by `validate` for integer
  literals. Refused: the type blocks at compile time what `validate` would
  catch, and the meaning is `FetchRel`'s.
- **`u64` `FetchNode` bounds.** Refused: Substrait's integer literal is
  `i64`, so a bound above `i64::MAX` has no encoding, and the adapter does not
  refuse a plan (ADR 0068).
- **`TopNNode` held back** until a request carries "with ties". Refused by the
  human: admitted now, refused by `validate`, lowered later — the smallest
  blast radius.
- **Only `emit_form` in `NodeCommon` now.** Refused: `RelCommon` is common to
  every relation, and its other fields are ignorable by contract.
- **Two ids** (`NodeId` beside `anchor: Option<u32>`), or
  `enum Anchor { Seq(u32), Uuid(u128) }`. Refused: one idea, one id, one type.
- **No `output_schema`**, the producer deriving schemas by walking the tree.
  Refused: name-based references need the input's names at every node, and a
  type failure would move into an adapter that does not refuse.
- **`RootNode { input, fields }`** in place of `SemanticPlan`. Refused: the
  glossary's *Semantic plan* names the whole plan for one request, and the
  rename changes no meaning.
- **`PlanNode::Distinct`**; **`ValuesNode` beside `VirtualTable`** — two homes
  for one set of inline rows.

## Decision Outcome

Each plan node is semstrait's implementation of one Substrait relation at the
**reference release, Substrait v0.103.1** (2026-09-20): it has that relation's
slots, and each slot means what it means in Substrait. A node admits only the
values semstrait writes. The gap between the reference release and the release
an engine reads — DataFusion 55.1 reads v0.85.0, through crate 0.63 — and the
differences between engines are the adapter's
(ADR 0068), never the plan's. This ADR changes no code; the implementation
effort makes the carrier changes.

### Rulings

1. **Mirror.** A node has its relation's slots, with the relation's meaning,
   and admits only the values in use, listed per node in ruling 4. Names are
   semstrait's where the glossary has a term — `ScanNode` for `ReadRel`,
   `JoinType::Full` for `OUTER`, `aggregates` for `measures` (*Measure* is a
   semstrait role), `order_by` for `sorts`, `limit` for `count` — and
   Substrait's where it has none (`VirtualTable`, `NamedTable`). The producer's
   mapping table holds each pair (ADR 0068).
2. **Growth.** A mirrored slot, value or node enters in one of two ways: with
   the code change that writes it, or by a human's ruling recorded in an ADR
   (the common fields, ruling 3; `TopNNode`, ruling 4; the text read options,
   ADR 0067). A field with no Substrait slot — a **semstrait-only field** —
   enters only by amending an ADR. The semstrait-only fields are
   `NodeCommon.output_schema` and `NodeCommon.semantic_meta` (ruling 3), and on
   the scan's readers the files' `pattern` and `base_path` and a table's
   reported format (ADR 0067).
   The reference release moves only by amending this ADR, so the vocabulary
   never changes silently under a new Substrait release. `#[non_exhaustive]` is
   a Rust-evolution device, not a licence. This ADR mirrors relations only: the
   expression vocabulary is semstrait's own, lowers through the producer's
   function mapping (ADR 0068), and grows only by ADR, as before. The word is
   *vocabulary*; *catalog* is the physical catalog.
3. **The node-common block.** Every node composes one `NodeCommon`, the mirror
   of `RelCommon` at v0.103.1, and today's `NodeMeta` grown:

   ```
   NodeCommon {
     anchor: Anchor,                   // rel_anchor
     emit_form: Option<EmitForm>,      // emit_kind: None = Direct, Some = Emit { output_mapping }
     hint: Option<Hint>,               // stats { row_count, record_size }, constraint, alias,
                                       // output_names, saved and loaded computations with their type
     output_schema,                    // semstrait-only: every output column's name and type
     semantic_meta,                    // semstrait-only: trace
   }
   ```

   Every mirrored field but `anchor` is optional. `Some` is the planner's
   instruction to use the field; `None` is its absence, and an absent common
   field changes no result. A field the target release lacks (`rel_anchor`
   before v0.89) stays in the IR, and the producer drops or rewrites it for
   that release or engine (ADR 0068). `RelCommon.advanced_extension` is not
   admitted (ruling 7).
   - **`anchor`** is one id, `Anchor(u128)`, and replaces `NodeId`. The planner
     mints 1, 2, 3… per plan, so every run gives the same ids; a UUID also fits
     the type. It is unique per plan. The producer writes it as `rel_anchor`
     where the target release has the field, and renumbers in tree order only
     if a value does not fit `rel_anchor`'s `u32`.
   - **`emit_form`** selects and orders a node's natural output columns; a
     `ProjectNode` computes new ones. `validate` checks that `output_schema` is
     the emit form applied to the natural output.
   - **`hint`** is `None` today: the planner reads no statistics (ADRs 0015 and
     0064) and writes none. `hint.stats` is a Substrait hint, not a semstrait
     cost.
   - **`output_schema`** holds each output column's name and type. Semstrait
     expressions name columns (`orders.amount`); the producer turns a name
     into a position against the input's `output_schema`, and `validate` and
     ruling 5 read it.
   - **`semantic_meta`** is the trace — today's `NodeMeta.annotations`: the
     data kind, dataset and source behind a subtree, strategy boundaries, and
     a join's cardinality. No consumer executes it; ADR 0068 writes it only
     into the explain copy of a plan.
4. **Nodes.** Each node beside its `common`:

   | Node | Relation | Fields | Admitted values |
   |---|---|---|---|
   | `ScanNode` | `ReadRel` | `schema`, `filter: Option<PhysicalExpr>`, `reader` | readers `VirtualTable`, `FilesScan`, `TableScan::NamedTable` (ADR 0067) |
   | `FilterNode` | `FilterRel` | `input`, `filter` | |
   | `ProjectNode` | `ProjectRel` | `input`, `expressions` | output = input columns, then `expressions`, then `emit_form` |
   | `AggregateNode` | `AggregateRel` | `input`, `groupings: Vec<PhysicalExpr>`, `aggregates` | one grouping set; each aggregate a function with an optional `filter`, as `Measure { measure, filter }` (ruling 8) |
   | `JoinNode` | `JoinRel` | `left`, `right`, `join_type`, `on: PhysicalExpr`, `post_join_filter: Option<PhysicalExpr>` | `Inner`, `Left`, `Right`, `Full` |
   | `CrossNode` | `CrossRel` | `left`, `right` | |
   | `MultisetNode` | `SetRel` | `inputs`, `op` | `UnionAll`, `UnionDistinct` |
   | `SortNode` | `SortRel` | `input`, `order_by` | each sort field an expression with its direction and null order |
   | `FetchNode` | `FetchRel` | `input`, `offset: i64`, `limit: Option<i64>` | each bound non-negative |
   | `TopNNode` | `TopNRel` | `input`, `order_by`, `offset`, `limit`, `fetch_mode` | `RowsOnly`, `WithTies` |

   - **Join.** `on` is `on_expr_id` reassembled, alias-qualified leaves intact
     (ADR 0057), and is required: `CrossNode` is the only cross join.
     `KeyPair` retires; the planner never decomposes the condition.
     `post_join_filter` is `filter_expr_id` reassembled, applied to the joined
     rows after the join type — Substrait's meaning; on an outer join it may
     remove null-extended rows, unlike SQL `ON`. `validate` resolves `on`
     against the two join inputs under the binding rule, and
     `post_join_filter` against the join output; the outer-join nullability
     rule is unchanged. `JoinNode` has no cardinality field: `JoinRel` has no
     slot for one and nothing that reads the plan needs it. The planner reads
     the resolved cardinality from the manifest when it decides fan-out, and
     writes it into the join's `semantic_meta`.
   - **Cross.** `CrossNode` is a cartesian product. A joinset whose join type
     is cross (the handoff below) plans as a `CrossNode`, and its relationship
     filter as a `FilterNode` immediately above it — ADR 0038's theta join,
     inner by construction, since `CrossRel` has no post-join filter. Its
     cardinality rides in its `semantic_meta` like any join's: the resolved
     value, `ManyToMany` from the defaults matrix for a bare `on: cross`.
   - **Multiset.** `UnionNode` becomes `MultisetNode`: `distinct = false` is
     `UnionAll`, `true` is `UnionDistinct`. `validate` keeps the arity of two
     or more.
   - **Fetch.** `offset` 0 means no offset and `limit: None` means all rows,
     Substrait's meaning. The producer writes each as an integer literal in
     `offset_expr` / `count_expr` — the only slots at v0.103.1, and read by
     v0.85 too. Both bounds are `i64`, the type Substrait recommends for the
     two slots, so every bound the node holds encodes as a literal. A negative
     bound is an error in Substrait, and `validate` refuses one.
   - **TopN** is admitted by the human's ruling. It is not in the query API
     and no planner step writes it; `validate` refuses a `TopNNode` until its
     adapter path lands, so the adapter's never-refuse rule (ADR 0068) holds.
     `limit` is required, as `TopNRel`'s count is. The adapter path, for a
     consumer without `TopNRel`: `RowsOnly` is `FetchRel ← SortRel`;
     `WithTies` is `FetchRel(offset) ← SortRel ← FilterRel(r <= offset +
     count) ← ProjectRel(r = rank() over order_by)`, with `r` dropped by the
     emit form. A row's `rank()` is one plus the number of rows before it, so
     `r <= offset + count` keeps the first `offset + count` rows and every row
     tied with the last of them; the fetch then skips `offset` rows. Not
     `row_number()`, which cuts a tie group. The window is a `Function(Call)`
     with `over` (ADR 0091, rule 1), in the `ProjectNode`: on the wire a
     `WindowFunction` expression in a `ProjectRel`, which DataFusion 55 reads
     (`project_rel.rs:51-72`), not the window relation ruling 7 leaves out.
   - **Values.** `ValuesNode` retires into `ScanNode { reader: VirtualTable }`:
     its rows are the reader's `expressions` (v0.103.1 removed `values`;
     DataFusion 55 reads `expressions`), its schema the scan's `schema`.
   - **Root.** `SemanticPlan { root, output_names }` is the mirror of
     `RelRoot { input, names }`. It is not a `PlanNode` variant.
5. **Plan schemas stay total.** `SchemaColumn.data_type` / `nullable` and
   `AggregateExpr.inferred_type` are mandatory, on every `output_schema` and on
   `ScanNode.schema`. The planner's physical leaf is ADR 0061's one
   conversion: catalog-observed type and nullability; else the definition's
   `data_type` where the Dimension's tree *is* a bare `Column` (result and leaf
   are one value); else the planner's own `PlanError::UnresolvedType` (a
   planned planner variant, not an IR one) naming relation and column.
   Unobserved nullability is decided `true` there — the one value
   DataFusion's `ensure_field_compatibility` accepts on no evidence — as this
   ADR's rule, not a per-plan default. An optional type on the plan was
   refused: it would move the failure to a format adapter, which cannot see an
   engine's tables and does not refuse a plan (ADR 0068).
6. **Distinct rows** (ADR 0062, dimensions-only = no Measure or Metric in the
   demanded set) are `AggregateNode { groupings: <selected dimensions>,
   aggregates: [] }`. A `where`-only dimension lowers below it as a
   pre-aggregate predicate; a Measure in `where` makes the request
   measure-bearing. No `Distinct` node: Substrait has no
   `DistinctRel`, DataFusion's producer emits this `AggregateRel`, and its
   `Distinct` reaches execution only through the `ReplaceDistinctWithAggregate`
   rule.
7. **Not admitted** (each may enter under ruling 2):
   - the other join types (semi, anti, single, mark) and `LateralJoinRel`;
   - the other set operations (minus, intersection);
   - grouping sets — `AggregateNode` holds one;
   - `AggregateExpr.phase`: DataFusion 55 ignores `phase`, so a partial,
     where chosen, is by decomposition — `sum` / `min` / `max` refold as
     themselves, non-distinct `count` as `sum`, `avg` as `sum` / `count`;
     `count(distinct)`, `median`, `stddev`, `variance` are holistic and are
     not refolded. Whether and where the planner decomposes is not ruled here;
   - window relations (`ConsistentPartitionWindowRel`) — the next pass; the
     window call, `Function(Call)` with `over` (ADR 0091, rule 1), stays,
     compile refuses authored windows, and DataFusion 55 cannot consume the
     relation;
   - `ExpandRel`, `ReferenceRel`, and the physical join and exchange relations;
   - `ReadRel.best_effort_filter` and the projection mask (the emit form
     selects), and the reader kinds and file fields ADR 0067 leaves out;
   - advanced extensions, extension relations and extension tables: on the
     wire an extension is either an optimization, which a consumer may ignore,
     or an enhancement, which a consumer that does not know it must refuse;
   - write, DDL and update relations (ADR 0019), and distributed compute;
   - a computed `FetchNode` offset or limit — no planner step writes one;
   - the subquery, lambda, nested-type, switch and multi-or-list expression
     forms.
8. **Confirmed unchanged.** `Expr::Case` exists for its use. `coalesce` is a
   function call in the function catalog, not a node (ADR 0092, rule 6).
   `AggregateExpr` keeps its `filter` slot as the mirror of
   `AggregateRel.Measure { measure, filter }`: an aggregate function with an
   optional filter. The plan accepts a filtered aggregation in either shape
   the manifest writes: lowered into the aggregate itself, with `filter`
   `None`; or carried in `filter`. A carried filter is the adapter's to write
   or rewrite (ADR 0068).

### Handoff to the manifest effort

The artifact has no cross join type: `derive_join_type` maps `Optional` onto
four types and `on: cross` persists as the pinned identity `Eq(1, 1)` under
`on_expr_id`. Asked: (a) a unit's `join_type` is `Cross` iff **every**
relationship closing at it (ADR 0054) has a cross basis; otherwise identities
drop as neutral terms and the equi terms stand; (b) `on_expr_id` becomes
`Option<PhysicalExprId>`, `None` iff `Cross`; `cross_identity()` retires;
`filter_expr_id` unchanged; (c) `optional:` other than `none` on `on: cross` is
refused at validate — `Optional` maps onto the four non-`Cross` types, `Cross`
comes from the basis; (d) `MANIFEST_VERSION` advances. Until it lands, the
identity is the interim cross marker. The planner lowers a `Cross` joinset to
a `CrossNode`, so the plan's `JoinType` has no cross value; the manifest
persists `semstrait_ir::JoinType` today, and how its cross value is carried
without reaching `JoinNode` is the implementation's choice.

### Amends

- **0018:** each plan node is semstrait's implementation of one Substrait
  v0.103.1 relation, and "growth only inside the spec" becomes 0066's ruling 2.
  Its "no cost, statistics or physical properties" is narrowed: every node
  carries the optional Substrait common fields, the statistics hint among them,
  which the planner does not fill (0066's ruling 3); and `TopNNode`, a Substrait
  physical relation, is admitted and refused by `validate` (0066's ruling 4).
  `NodeMeta` becomes `NodeCommon`, `NodeId` becomes `Anchor`. Its `ScanNode`
  clause is ADR 0067's.
- **0022:** the vocabulary mirrors the Substrait v0.103.1 relations and admits
  only the values semstrait writes; it is fixed independently of engine
  capability, and the `JoinType` roster `{Inner, Left, Right, Full}` with its
  Semi/Anti/Lateral exclusion stands. A cross join enters as `CrossNode`,
  Substrait's `CrossRel`, because the model authors it, not because an engine
  offers it — 0022's own criterion. Its rewrite-warn-refuse ladder is ADR 0068's
  to amend.
- **0038:** the absent-cardinality row derives a cross join, which plans as a
  `CrossNode`, not inner; a preserved side on `on: cross` is no longer
  authorable (a LEFT theta join is unauthorable rather than unplannable); its
  "authorable and un-lowerable" disclosure is answered. All four cardinalities
  stay authorable beside `on: cross`, and an explicit symmetric one still owes
  `optional:` and `cross_filter:` — the owed `optional` is `none`, which is a
  value, so the completeness rule is satisfied without a preserved side. Its
  deferred `JoinNode.residual` landing site is `JoinNode.post_join_filter` under
  a keyed basis and a `FilterNode` above the `CrossNode` under `on: cross`.
  `JoinNode` has no cardinality field: the planner reads the resolved
  cardinality from the manifest to decide fan-out and writes it into the join's
  `semantic_meta`, in trace only.
- **0050:** "`on_expr_id` always present, `cross` as a tautology" is superseded
  — the signal it protected rides on the join type. Its two roots map onto the
  plan: `on_expr_id` → `JoinNode.on`; `filter_expr_id` →
  `JoinNode.post_join_filter`, or the `FilterNode` above a `CrossNode`. Its
  reason for keeping `cardinality` on the joinset holds with a new reader: the
  planner reads it from the manifest to decide fan-out, and the plan carries it
  in trace only.
- **0052:** its statement that ADR 0038's absent-cardinality row stands
  untouched is narrowed. The row still resolves cardinality to `ManyToMany`, and
  ADR 0052's cardinality gates stand; only its derived join type changes from
  `Inner` to cross.
- **0057:** ADR 0050's cross tautology is superseded; a cross joinset plans as a
  `CrossNode`, and the two-literal identity is only an interim cross marker.
- **0061:** reader-contract clause 2 admits the bare-`Column` Dimension
  exception; clause 3's "decide at the conversion" is met by 0066's ruling 5.

### Consequences

- Glossary: **Canonical vocabulary** states the mirror and the growth rule;
  **Node-common block** and **Emit form** are added.
- Handoffs: discharged by ADRs 0067 and 0068;
  `step_join(left, right, join_type, on, post_join_filter)`,
  `step_cross(left, right)` followed by `step_filter` when `filter_expr_id` is
  present, `step_scan` writing the reader (ADR 0067) and types to the output
  schema, `step_distinct = step_agg(all, [])`, window readiness; bind
  `on` against the join inputs, `post_join_filter` against the join output,
  and a cross join's filter against the cross output; `inferred_type`
  and the filter splice; `distinct` timing and decomposition;
  typed fixtures only; the implementation effort — `validate` refuses
  `TopNNode` and checks `output_schema` against `emit_form`.

## More Information

Rewritten in place 2026-09-23 by the deciding human's ruling.
**This engages the bound that a merged ADR is closed to in-place
edits, and the exception is deliberate and narrow**: the number stays, the text
above replaces the merged text whole, and no correction note is left in it (git
history holds the retired text). Carried over in substance: the typed-schema
rule (ruling 5), distinct rows (ruling 6), the aggregation-phase evidence
(ruling 7) and the four asks of the manifest handoff.
