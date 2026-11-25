# semstrait-ir

The plan and graph vocabulary. The planner
(`semstrait-planner`) produces the `SemanticPlan` defined here; every adapter
(`semstrait-adapter`) consumes it. Each plan node is semstrait's implementation
of one Substrait v0.103.1 relation: it has that relation's slots, with their
meaning, and admits only the values semstrait writes (ADR 0066). The vocabulary
is engine-neutral: no dialect tag on any plan node, and the statistics hint is
admitted but never written.

Substrait is a **derived, outbound-only** target, not an interchange format
(ADR 0019) — see [Substrait is outbound only](#substrait-is-outbound-only).

This file is current-state reference only. Decisions live in
[`docs/adr/`](../../docs/adr/); terminology lives in
[`CONTEXT.md`](../../CONTEXT.md).

---

## Dependency direction

This crate defines no expression, function or data type: the tree, the
aggregate call and the data types are `semstrait-sql`'s; it depends on no
model crate and takes its names from `semstrait-common` (ADR 0094, rule 9),
and only `FileFormat` from `semstrait-catalog` (ADR 0067 refuses a
format roster of the plan's own). It imports each type it uses from the crate
that owns it, and re-exports none of them (ADR 0090, rule 6); see
[Dependencies](#dependencies). A crate that uses `Granularity` imports
it from `semstrait-common`. ADR 0001 also rules that a cross-layer lowering such
as `resolve` belongs in `semstrait-manifest` — the only crate permitted to see
both sides — as a free function rather than an inherent method.

---

## Module map

The modules are private: each type has one import path, the crate root
(`semstrait_ir::PlanNode`, `semstrait_ir::Parameter`).

| Module | Owns |
|---|---|
| [`expr`](src/expr.rs) | The plan's leaf, `Parameter` / `ParameterKey`, and its tree, `PhysicalExpr` (an alias of `semstrait-sql`'s `Expr<Parameter>`). The expression id and the expression layer are `semstrait-manifest`'s (ADR 0090, rules 4 and 5) |
| [`types`](src/types.rs) | `Schema` / `SchemaColumn` over `semstrait-sql`'s `DataType`, a leaf's parts as the dotted name that an output field carries (`field_name`), the leaf binding rule (`Schema::position`), and typing an expression over a schema's fields |
| [`primitives`](src/primitives.rs) | `Name`, `JoinType`, `Cardinality`, `Integrity`, `CrossFilter`, `JoinsetJoinType` |
| [`semantic`](src/semantic.rs) | `SemanticRole`, `AdditivityType`, `SemiAdditivity`, `SemiAdditivityStrategy` |
| [`temporal`](src/temporal.rs) | `TemporalShape`, `TemporalType`, `ScdType` |
| [`plan`](src/plan/) | `PlanNode` and its node structs with their constructors, the node-common block (`NodeCommon`, `Anchor`, `EmitForm`, `Hint`), the scan's readers, the trace (`SemAnnotation`), the `Tree` traversal impl, and `SemanticPlan` with `validate` |
| [`semantic_graph`](src/semantic_graph.rs) | Immutable graph value types and read traits (`SemanticNode`, `SemanticEdge`, `SemanticGraph`, `SemanticGraphRead`, …). No graph library, no mutation, no traversal policy. No crate reads them: the planner's lookups walk the manifest, and a later semantic graph, built at setup, replaces them (ADR 0064, *The seam a cached graph replaces*) |
| [`error`](src/error.rs) | `IrErrorKind` |

---

## SemanticPlan

The mirror of `RelRoot`: the plan for one request.

```rust
#[non_exhaustive]
pub struct SemanticPlan {
    pub root: PlanNode,
    pub output_names: Vec<Name>,
}
```

`output_names` are the request's bare names, which the root outputs (ADR 0075).
`SemanticPlan::new(root, output_names)` numbers every node's anchor 1, 2, 3… by
one pre-order walk, the root 1 (ADR 0073, *Details fixed with the seam*); no
constructor writes one.

`SemanticPlan::validate()` refuses a plan that no engine can take, so no
adapter meets one (ADR 0068). It returns the first violation as a
`Diagnostic<IrErrorKind>` whose note names the node by its anchor and type
(`node 3 (FilterNode)`):

1. an anchor that repeats, in pre-order;
2. each node after its inputs: its constructor's rules again over its input and
   parts, with the `IrErrorKind` that the constructor gives for the same fault;
   an output schema that is not the emit form over the natural output; an
   aggregate whose name or type is not its field's (ADR 0090, rule 2); a name
   that the output schema repeats, case ignored, but in a scan; then what no
   engine can take yet: a `TopNNode`, a `Parameter`, a literal whose value does
   not fit its type (`check_fit()`, ADR 0085, rule 5), an aggregate call
   anywhere but as an aggregate node's entry, a call that carries a null
   treatment, a window call outside a projection's expressions, each place its
   own variant, a join's `on` and post-join filter among them, and a window
   frame that Substrait cannot hold (GROUPS, or an offset past `i64`);
3. output names that are not the root's field names, and a root name with a
   reserved prefix (`__semstrait_`, `__plan_`, `__agg_`).

---

## PlanNode

Ten `#[non_exhaustive]` variants (ADR 0007), each wrapping a
`#[non_exhaustive]` struct, so another crate builds a node only through its
constructor. Each constructor binds every leaf, types every new expression
with `semstrait-sql`'s `type_tree` against its input's output schema in the
canonical type system, and derives the node's output schema; the output schema
is never passed in. A constructor refuses a node with the `IrErrorKind` of its
fault.

| Variant | Relation | Constructor | Natural output |
|---|---|---|---|
| `Scan` | `ReadRel` | `ScanNode::new(reader, schema)`, each `VirtualTable` row of the schema's arity and types, a value that can be NULL only in a nullable field; `with_filter` sets the `filter` slot, `None` in MVP | `schema`, the read set |
| `Filter` | `FilterRel` | `FilterNode::new(input, filter)`, a condition | the input's fields |
| `Project` | `ProjectRel` | `ProjectNode::new(input, [(Name, PhysicalExpr)])`; the names go to the output schema only. A window call is typed by its declaration (`rank` and `row_number` give `bigint`, never NULL), and its order keys take a NULL order other than `Unspecified` | the input's fields, then one per expression |
| `Aggregate` | `AggregateRel` | `AggregateNode::new(input, [(Name, grouping)], [(Name, AggregateCall)])`, one grouping set; each call's `filter` clause is `Measure.filter`, and its ORDER BY keys take a NULL order other than `Unspecified`. With no aggregate it gives distinct rows (ADR 0066, ruling 6) | the groupings, then the aggregates |
| `Sort` | `SortRel` | `SortNode::new(input, order_by)`, each a `semstrait-sql` `SortKey` with a NULL order other than `Unspecified` | the input's fields |
| `Fetch` | `FetchRel` | `FetchNode::new(input, offset, limit)`, non-negative `i64`s; `offset` 0 is no offset, `limit` `None` every row | the input's fields |
| `TopN` | `TopNRel` | `TopNNode::new(input, order_by, offset, limit, fetch_mode)`, admitted and not yet lowered | the input's fields |
| `Join` | `JoinRel` | `JoinNode::new(left, right, join_type, on, post_join_filter)`, `Inner`, `Left`, `Right` or `Full`; `on` a condition over `left ++ right`, `post_join_filter` one over the natural output, both before any emit form | the left input's fields, then the right's, the outer side's nullable |
| `Cross` | `CrossRel` | `CrossNode::new(left, right)`; its relationship filter is a `FilterNode` above it | the left input's fields, then the right's |
| `Multiset` | `SetRel` | `MultisetNode::new(inputs, op)`, two or more inputs, `UnionAll` or `UnionDistinct`; each branch has the first branch's names, case ignored, and types at each position | the first branch's fields, each nullable where any branch's is |

A scan's `reader` is one of `VirtualTable { expressions }` (inline rows; with
none, the **empty read**), `FilesScan { format, format_options, pattern,
base_path, paths }` (the listing, or with no paths and no pattern the **folder
read** of `base_path`), and `TableScan::NamedTable { names, format }`, whose
three names are catalog (`""` for none), namespace and name (ADR 0067, as ADRs
0070 and 0073 amend it). A reader's format is `semstrait-catalog`'s
`FileFormat`.

**Binding.** A leaf whose first part is a physical source alias `ps<i>` binds
to its input's field by its last part, exactly; every other leaf binds to the
one field whose name equals its parts joined by `.`, case ignored. No string is
cut (ADR 0075, *Every output name is the one its parent reads*; ADR 0085,
rules 11 and 12; `Schema::position`).

**Names.** A constructor refuses a name that the names it is given repeat, case
ignored; `with_emit_form` and `validate` refuse an output schema that repeats
one, but a scan's, a physical read (ADR 0075).

`PlanNode` implements `semstrait-common`'s `Tree`: replacing a node's input
rebuilds it through its constructor and its emit form, so its output schema is
derived again, and keeps its anchor, hint and trace, and each computed field's
name, read over the old input.

### NodeCommon

```rust
#[non_exhaustive]
pub struct NodeCommon {
    pub anchor: Anchor,                // rel_anchor
    pub emit_form: Option<EmitForm>,   // emit_kind: None = Direct, Some = Emit { output_mapping }
    pub hint: Option<Hint>,            // RelCommon.hint; never written
    pub output_schema: Schema,         // semstrait-only
    pub semantic_meta: Vec<SemAnnotation>, // semstrait-only: the trace
}
```

`PlanNode::with_emit_form(emit_form)` sets an emit form on any node and derives
its output schema again: the fields it selects from the natural output, in its
order. It computes nothing, and it refuses a form that leaves out a field that
the node computes (a projection's expressions; an aggregate's groupings and
aggregates); a node above leaves such a field out by its own emit form.
`PlanNode::with_trace(trace)` writes the node's trace.

---

## Trace

`SemAnnotation` holds two line kinds (ADR 0095), each sorted by variant then
by value: `Rank`, the scope root's rank on a line of its own — its
`scope_root` (`DataKindName`), its `compositions` count, its
`granularity_fit` (`GranularityGroup`: `Requested`, `Selected`,
`NotSelected`), and an optional `granularity`; and `SemanticSource`, one line
per semantic source a node plans — its `semantic_source` (`DataKindName`),
its `source_role` (`SourceRole`: `Dataset`, `Joinset`, `Unionset`,
`Grainset`), its `originators` (the manifest's recorded originators, empty
when it recorded none), an optional `physical_source`, a join's `cardinality`
(a cross's resolved value, or `ManyToMany` for a bare `on: cross`; none on
the join of two subtrees, ADR 0074), the `delivery` a union's branches gave,
a stand-in's `answered_by`, and the `pruned` operands it chose against.
A plan's nodes are read in anchor pre-order. No consumer executes the trace;
only the explain copy of a plan carries it (ADR 0068).

---

## Substrait is outbound only

Per **ADR 0019**, the canonical form is the in-memory `SemanticPlan` /
`PhysicalExpr`. Substrait is a co-equal *output* derived from it:

- The mapping is **unidirectional**. Ingesting an externally produced
  `substrait::proto::Plan` is out of scope.
- `semstrait-adapter`'s `SubstraitProducer` writes it, and no conversion
  code lives here (ADR 0068).
- This crate links no wire-format crate and holds no artifact type: the
  producer's output is `substrait::proto::Plan`, and no engine identity lives
  here (ADR 0068, *Artifacts*).

---

## Serde posture

Serde derivation is **partial, and on purpose**:

- Derived: `types` (`Schema`, `SchemaColumn`), every `primitives` type, and
  `Parameter` / `ParameterKey`. `semstrait-manifest` persists schemas, join
  types and cardinalities, so their encoding must be deterministic — the same
  value encodes to the same bytes every time, which is what makes a stable
  manifest version possible. No stored expression holds a `Parameter` (ADR
  0088, rule 4).
- Not derived: the plan tree (`SemanticPlan`, `PlanNode`, `NodeCommon`,
  `SemAnnotation`) and `semantic_graph`. Graph structures are
  rebuilt in memory at graph-build time and never persisted — identity on
  disk, structure in memory.

`semstrait-sql`'s `Expr` and its calls take no serde derive (ADR 0088, rule
1), so neither `PhysicalExpr` nor an aggregation node's `AggregateCall` has
one. Where a `Deserialize` half exists it is for same-version internal
transport and test fixtures only — it is not a contract for inbound conversion
(ADR 0019).

---

## Errors

| Type | Raised by |
|---|---|
| `IrErrorKind` | A plan node's constructor, `with_emit_form` and `SemanticPlan::validate`, the same variant for the same fault: one variant per fault — an unbound leaf, a type refused, a field or sort key whose type is not valid (`Unknown` among them), a filter that is no condition, a `VirtualTable` row of another arity or type, or with a value that can be NULL in a field that is not nullable, an emit form index out of range or a computed field left out, a repeated name, a negative fetch bound, an unspecified NULL order on a sort key, a window's order key or an aggregate's ORDER BY key, a union of fewer than two inputs or with a branch that is not its first branch at a field; and `validate`'s: an output schema or an aggregate field that its node does not derive, an unnamed computed field, a repeated anchor, output names that are not the root's, a reserved root name, and each form that no engine can take yet |
| `ValidateError` | `Tree::with_new_children` (`ChildCountMismatch`, and `NodeRefused`, whose `reason` is the message of the rebuilt node's `IrErrorKind`: its constructor's or `with_emit_form`'s refusal over the new inputs, or a computed field that the node's output schema does not hold or name, as `OutputSchemaMismatch`, `EmitFormLeavesOutComputedField` or `UnnamedField`), `Name::new`; `semstrait-common`'s |

A call that does not fit its function's declaration is refused by
`semstrait-sql`'s constructor with a `CallError`, before it reaches a plan
node (ADR 0091, rule 2).

Diagnostic primitives (`Diagnostic`, `Diagnostics`, `Severity`) live in
`semstrait-common`. Error kinds are typed variants, not numeric codes
(ADR 0004).

---

## Dependencies

- `semstrait-sql` — the tree `Expr`, the aggregate call `AggregateCall`,
  `DataType` (a schema column's type), `SortKey` (a sort's key) and typing
  (`type_tree`).
- `semstrait-common` (with `serde`) — `Tree` for `PlanNode`,
  `ValidateError`, the diagnostics, `EntityId` for the semantic graph, and
  `DataKindName` and `Granularity`, in a plan node's trace (`SemAnnotation`;
  ADR 0094, rules 3 and 9).
- `semstrait-catalog` — `FileFormat` only, on a scan's reader (ADR 0067).
  It brings the catalog crate's async dependencies into this crate's build.
- `serde` — see [Serde posture](#serde-posture).
- `thiserror` — error plumbing.

Dev-only: `serde_json`, for the serde round-trip tests.

The crate declares **no Cargo features** and **no `uuid` dependency**: an
`Anchor` is a `u128` that `SemanticPlan::new` numbers.
