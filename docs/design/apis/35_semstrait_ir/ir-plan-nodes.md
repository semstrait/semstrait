---
id: 35-ir-plan-nodes
status: ratified
version: 2.1
impl: implemented
---

# IR Plan Nodes

The canonical, engine-agnostic query plan tree. `SemanticPlan` is the root; `PlanNode` is an 8-variant sum type mirroring Substrait's `Rel` family with name-based column references.

## `SemanticPlan` Root

```rust
/// The canonical, engine-agnostic query plan tree. Output of the planner
/// (`34`), input of every adapter (`36`). Per `00 §4.1`.
///
/// A `SemanticPlan` is a single rooted tree of `PlanNode`s plus
/// plan-wide metadata: the names in the final projection order, and any
/// warning-severity diagnostics the planner wishes to surface alongside
/// the plan (errors abort planning per `10 §3.4`).
#[non_exhaustive]
pub struct SemanticPlan {
    /// The root of the plan tree. Never empty; planning produces at
    /// least a `Scan` leaf.
    pub root: PlanNode,

    /// Output column names in the order they appear in `root`'s
    /// output schema. Length equals `root.meta().output_schema.len()`.
    /// Each entry is the user-visible column label the adapter emits
    /// (e.g. the SELECT-list alias in generated SQL).
    pub output_names: Vec<Name>,

    /// Warning-severity diagnostics surfaced by `plan` + `optimize`.
    /// Per `30 §7.3` and `34 §13.2`, warnings flow through BOTH the
    /// planner result tuple's `Diagnostics<PlanErrorKind>` second element
    /// AND this artifact-side `Vec<PlanDiagnostic>`. Contains only
    /// `Severity::Warning` entries per `30 §5.2`; errors abort planning.
    pub diagnostics: Vec<PlanDiagnostic>,
}
```

### Invariants

A `SemanticPlan` is **well-formed** when:

1. `output_names.len() == root.meta().output_schema.len()` — every output column has a user-visible name.
2. Every `Name` in `output_names` is a valid identifier (see [ir-plan-primitives.md](ir-plan-primitives.md) §11.4).
3. `root` and every descendant satisfy the tree invariants of §13 below.
4. Every `PlanDiagnostic` in `diagnostics` has `Severity::Warning`.

Construction does **not** re-check invariants 1–3 at runtime (planning established them). An optional `SemanticPlan::validate()` method (§14 below) walks the tree and reports violations for debugging.

### Serde

`SemanticPlan` derives `Serialize` / `Deserialize` under the crate-level `serde` feature. The wire form is the direct struct shape; no intermediate envelope. `PhysicalExpr` inside child nodes serializes through the shared `Expr<L>`'s derived serde (see [ir-serde-substrait.md](ir-serde-substrait.md)). `PlanNode` (`#[non_exhaustive]`) requires a tagged wire form carrying an explicit discriminator (`"scan" | "filter" | ...`); the concrete tagging form is fixed with the `PlanNode` serde deliverable (adjacently tagged `type` / `content` is the recommended default).

A serialized `SemanticPlan` is a format-stable portable plan artifact: two processes with the same compiled SemanticManifest can exchange a `SemanticPlan` and get identical adapter output.

### Construction Patterns

Planners build a `SemanticPlan` from the bottom up:

```rust
let scan = PlanNode::Scan(ScanNode {
    meta: NodeMeta::new(output_schema),
    source: source_ref,
    columns: scan_columns,
    filters_pushdown: Vec::new(),
});

let filter = PlanNode::Filter(FilterNode {
    meta: NodeMeta::new_shared(scan.meta().output_schema.clone()),
    input: Box::new(scan),
    predicate: physical_expr_from_request(req)?,
});

let plan = SemanticPlan {
    root: filter,
    output_names: vec![Name::new("id")?, Name::new("amount")?],
    diagnostics: Vec::new(),
};
```

Builders in `semstrait-planner` (`34`) wrap these constructions in a higher-level fluent API; `35` exposes only the struct-literal form.

### Cloning

`SemanticPlan: Clone`. Every child node is `Box`-owned; `Clone` is a deep clone. For cheap structural sharing inside optimizer passes, use `walk_post` with in-place `transform` (§14) rather than successive `clone`s.

## `PlanNode` Sum Type

```rust
/// A single node within a `SemanticPlan`. The variant set forms the
/// canonical operator catalog borrowed (structurally only) from
/// DataFusion's `LogicalPlan`, Calcite's `RelNode`, and Substrait's
/// `Rel`. Per `00 §4.1` and §1.4 (see [ir-crate-charter.md](ir-crate-charter.md)).
///
/// `#[non_exhaustive]` per I10: adding a variant is MINOR.
#[non_exhaustive]
pub enum PlanNode {
    Scan    (ScanNode),
    Filter  (FilterNode),
    Project (ProjectNode),
    Agg     (AggNode),
    Join    (JoinNode),
    Union   (UnionNode),
    Sort    (SortNode),
    Fetch   (FetchNode),
}

impl PlanNode {
    /// Shared accessor for the `NodeMeta` that every variant carries.
    pub fn meta(&self) -> &NodeMeta;
    pub fn meta_mut(&mut self) -> &mut NodeMeta;

    /// Shared accessor for the set of child `PlanNode`s (0, 1, or 2+).
    pub fn children(&self) -> Vec<&PlanNode>;
    pub fn children_mut(&mut self) -> Vec<&mut PlanNode>;
}
```

Eight variants in v1. Every variant wraps a struct (not tuple / record form) so field additions inside a variant are MINOR per `30 §4.2`.

## Per-Variant Shapes

### `Scan` — Leaf Source

```rust
/// Reads a resolved source. The only `PlanNode` variant without child
/// nodes. Each `ScanNode` corresponds to one engine-level LogicalRelation
/// (Substrait `ReadRel`, DataFusion `TableScan`, SQL `FROM`).
#[non_exhaustive]
pub struct ScanNode {
    pub meta: NodeMeta,

    /// Opaque handle into the SemanticManifest. Resolves to a
    /// `ResolvedPhysicalSource` per `15 §7.1`. Adapters consume the
    /// SemanticManifest + this handle to learn the on-engine table / path /
    /// format. `35` never stores the expanded path.
    pub source: SourceRef,

    /// The projected columns in the order the `Scan` outputs them.
    /// Each `ResolvedColumn` carries `{ name, data_type, nullable,
    /// ordinal }` per `15 §4.2`. Non-empty.
    pub columns: Vec<ResolvedColumn>,

    /// Push-down predicates resolved at plan time. Each `PhysicalExpr`
    /// references only columns in `columns`. Empty when no pushdown is
    /// applicable or the adapter does not support it.
    pub filters_pushdown: Vec<PhysicalExpr>,
}
```

`ScanNode` carries **no raw path, no URL, no dialect**. Resolution from `SourceRef` to on-engine identity happens in the adapter via the SemanticManifest (I1).

**Partition info — manifest-side, never on `ScanNode`.** `ScanNode` carries **no partition columns, no partition transforms, no `partition_def` declarations, and no per-source partition values**. All partition-related metadata, when present, is reachable on the SemanticManifest via `ScanNode.source` — specifically `15 §3.4 PartitionColumn` and `15 §3.5.4 partition_def`. Engine consumers handle partition pruning from `filters_pushdown` predicates. Adding partition fields to `ScanNode` would diverge from every primary consumer's expected shape (Substrait `ReadRel`, DataFusion `TableScan`, Spark `LogicalRelation`, SQL `FROM` — none carry partition info on the rel).

### `Filter` — Predicate

```rust
#[non_exhaustive]
pub struct FilterNode {
    pub meta: NodeMeta,
    pub input: Box<PlanNode>,
    /// Boolean-valued predicate. Operand type must be `Boolean`.
    /// Columns referenced must exist in `input.meta().output_schema`.
    pub predicate: PhysicalExpr,
}
```

Pass-through schema: `FilterNode.meta.output_schema` equals `input.meta().output_schema` (enforced at construction; adapters rely on this).

### `Project` — Column List

```rust
#[non_exhaustive]
pub struct ProjectNode {
    pub meta: NodeMeta,
    pub input: Box<PlanNode>,

    /// Ordered list of `(output_name, expression)` pairs. The result
    /// schema's field at ordinal `i` has name `projections[i].0` and
    /// `data_type` follows by inference from `projections[i].1`'s
    /// leaf-level `ExprLeaf::inferred_type`. Empty list is
    /// rejected at construction (trivial `Project` collapses to `input`).
    pub projections: Vec<(Name, PhysicalExpr)>,
}
```

### `Agg` — Group-By + Aggregates

```rust
#[non_exhaustive]
pub struct AggNode {
    pub meta: NodeMeta,
    pub input: Box<PlanNode>,

    /// Group-by keys by output-schema column name. Each `Name` must
    /// resolve to a column in `input.meta().output_schema`.
    /// An empty `group_by` is a grand-total aggregation.
    pub group_by: Vec<Name>,

    /// Aggregate expressions paired with their output-column names.
    /// Each `AggregateExpr` is an `Expr::Aggregate` kernel lifted out
    /// of the input `PhysicalExpr` by Phase B per `19 §7`.
    pub aggregates: Vec<(Name, AggregateExpr)>,
}
```

### `Join` — Binary Composition

```rust
#[non_exhaustive]
pub struct JoinNode {
    pub meta: NodeMeta,
    pub left: Box<PlanNode>,
    pub right: Box<PlanNode>,

    /// Join kind. Per `16 §5.2` and `00 §4.1`.
    pub join_type: JoinType,

    /// Cardinality annotation carried from the Relationship graph
    /// (`16 §5.1`). Planners fill this from the `Relationship`
    /// ratified in `16`; optimizers / adapters MAY use it for
    /// optimization hints.
    pub cardinality: Cardinality,

    /// Join keys. Each `KeyPair` names one column from `left`'s
    /// schema and one from `right`'s. Non-empty for equi-joins
    /// (`Inner`, `Left`, `Right`, `Full`) in v1; cross-join is not a
    /// v1 variant (see §17.1).
    pub on: Vec<KeyPair>,
}
```

Non-equi-join predicates (range joins, inequality joins) are deferred — a future `JoinNode.residual: Option<PhysicalExpr>` field is MINOR.

### `Union` — N-Ary Stack

```rust
#[non_exhaustive]
pub struct UnionNode {
    pub meta: NodeMeta,

    /// Two or more inputs. Every input's `output_schema` must be
    /// structurally compatible (same arity, same element types, same
    /// nullability).
    pub inputs: Vec<PlanNode>,

    /// Whether to apply `DISTINCT` deduplication after the union.
    /// `false` = `UNION ALL` (bag semantics); `true` = `UNION` (set
    /// semantics). Defaults to `false`.
    pub distinct: bool,
}
```

### `Sort` — Ordering

```rust
#[non_exhaustive]
pub struct SortNode {
    pub meta: NodeMeta,
    pub input: Box<PlanNode>,

    /// Sort keys in priority order (first entry is the primary key).
    /// Each `Name` resolves to a column in `input`'s schema.
    /// Each `SortDir` carries ascending/descending + null-ordering.
    pub order: Vec<(Name, SortDir)>,
}
```

### `Fetch` — Limit / Offset

```rust
#[non_exhaustive]
pub struct FetchNode {
    pub meta: NodeMeta,
    pub input: Box<PlanNode>,

    /// Row limit. `None` = unlimited. `Some(0)` is well-formed;
    /// adapters may short-circuit emission.
    pub limit: Option<u64>,

    /// Row offset. `None` = no offset. `Some(0)` is equivalent to
    /// `None` and kept for Substrait-roundtrip fidelity.
    pub offset: Option<u64>,
}
```

Unsigned integer (`u64`) deliberately rejects negative values at the type boundary; the `Option` shape keeps "no limit / no offset" distinct from "limit 0 / offset 0". The rare `u64 > i64::MAX` case is rejected at adapter-emit time with `IrErrorKind::FetchValueOutOfRange` (see [ir-errors.md](ir-errors.md)).

## Tree Invariants (§13)

A **well-formed** `SemanticPlan` satisfies every invariant below. The planner (`34`) is the canonical producer. `SemanticPlan::validate()` (§14) is an optional post-hoc walker that reports violations.

### Expression-Wrapper Invariants

- Every predicate-valued expression on a `PlanNode` is the resolved `ir::PhysicalExpr` — never an authoring `model::UnresolvedSemanticExpr`. The leaf-set boundary makes this a type-level invariant — `ir::PhysicalExpr` is built over `PhysicalLeaf` and literally cannot contain `Field` / `Dimension` / `Measure` / `Metric` / `Key`.
- No `PhysicalExpr` stored on a `FilterNode.predicate`, `ProjectNode.projections[*].1`, `ScanNode.filters_pushdown[*]`, or future `JoinNode` residual carries an `Expr::Aggregate` node — aggregation is lifted into `AggNode.aggregates` as `AggregateExpr` by the planner's Phase B aggregate-lift per `19 §7`.
- No `PhysicalExpr` reaching a `PlanNode` carries an `Expr::Window` node directly authored — `Window` is compile-emitted only, entering the tree exclusively through sugar-accessor elimination at compile.

### Type-Resolution Invariants

- Every `PhysicalExpr` stored on every `PlanNode` is fully type-resolved: every leaf returns `Some(_)` from `ExprLeaf::inferred_type()`.
- Every `AggregateExpr.inferred_type` is populated per the aggregate-typing rules implied by the registered aggregate signatures in `FunctionRegistry`.

### Scan-Schema Invariants

- `ScanNode.columns[*]` references actual columns of the resolved source.
- `ScanNode.meta.output_schema.len() == ScanNode.columns.len()`.
- `ScanNode.meta.output_schema.fields[i].name == ScanNode.columns[i].name` for all `i`.

### Push-Down Invariants

- Every `PhysicalExpr` in `ScanNode.filters_pushdown` references only columns in `ScanNode.columns`.
- `filters_pushdown` does not change `meta.output_schema` — it narrows row count, not column shape.

### Join Invariants

- `JoinNode.on` is non-empty. (Cross-joins deferred.)
- For each `KeyPair`, `left` resolves to a column in `left.meta().output_schema` and `right` resolves to a column in `right.meta().output_schema`.
- For each `KeyPair`, `left`'s column `data_type` matches `right`'s (modulo nullability). A mismatch is `IrErrorKind::JoinKeyTypeMismatch` (see [ir-errors.md](ir-errors.md)).
- `JoinNode.meta.output_schema` = structural concatenation of `left.meta().output_schema` and `right.meta().output_schema`, with nullability widened on the outer side per `join_type`.

### Union Invariants

- `UnionNode.inputs.len() >= 2`.
- All inputs have structurally compatible output schemas: same arity; same `DataType` at each ordinal; same nullability at each ordinal (after upward widening).
- `UnionNode.meta.output_schema` = first input's schema with nullability widened across inputs.

### Agg Invariants

- Every `Name` in `AggNode.group_by` resolves to a column in `input.meta().output_schema`.
- Every `(Name, AggregateExpr)` in `AggNode.aggregates` has a unique output `Name`. Duplicate output-name is `IrErrorKind::DuplicateAggName`.
- Every `args[i]` of each `AggregateExpr` references only columns in `input.meta().output_schema`.
- `AggNode.meta.output_schema` = one column per `group_by` entry (in that order) followed by one column per `aggregates` entry (in that order).

### Sort Invariants

- Every `Name` in `SortNode.order[*].0` resolves to a column in `input.meta().output_schema`.
- Pass-through schema: `SortNode.meta.output_schema == input.meta().output_schema` (cheap via `Arc` share).

### Fetch Invariants

- If `FetchNode.limit == Some(0)`, the adapter MAY short-circuit to an empty-relation emission; this is an adapter choice, not a plan-tree invariant.
- Pass-through schema: `FetchNode.meta.output_schema == input.meta().output_schema`.

### Filter Invariants

- The `FilterNode.predicate`'s inferred type is `DataType::Boolean`. A non-Boolean predicate is `IrErrorKind::FilterPredicateNotBoolean`.
- Pass-through schema: `FilterNode.meta.output_schema == input.meta().output_schema`.

## Visitor / Traversal API (§14)

### `PlanVisitor`

```rust
/// Tree walker over a `SemanticPlan`. Implementations provide node
/// handlers; `PlanNode::walk_pre` / `walk_post` dispatch.
///
/// Equivalent to `Visitor<PlanNode>` per the universal trait surface in
/// `semstrait-common` (`14 §3.1`); preserved as a named alias for
/// plan-tree consumer ergonomics. `PlanNode` also implements `Tree` so
/// the generic helpers (`apply`, `transform`) work over plan trees.
pub trait PlanVisitor {
    type Output;
    fn visit(&mut self, node: &PlanNode) -> Self::Output;
}

/// Mutating variant for in-place analysis.
pub trait PlanVisitorMut {
    type Output;
    fn visit_mut(&mut self, node: &mut PlanNode) -> Self::Output;
}
```

### Walk / Transform Methods

```rust
impl PlanNode {
    /// Pre-order traversal: visitor sees each node before its children.
    pub fn walk_pre<V: PlanVisitor>(&self, v: &mut V) -> V::Output;

    /// Post-order traversal: visitor sees each node after all children.
    pub fn walk_post<V: PlanVisitor>(&self, v: &mut V) -> V::Output;

    /// Bottom-up rewrite: each node is rewritten after its children.
    /// Propagates `Err` if the rewrite function fails on any node.
    pub fn transform<F>(self, f: F) -> Result<PlanNode, IrErrorKind>
    where F: FnMut(PlanNode) -> Result<PlanNode, IrErrorKind>;

    /// Iterator-style child access; used by generic tree algorithms.
    pub fn children(&self) -> impl Iterator<Item = &PlanNode>;
    pub fn children_mut(&mut self) -> impl Iterator<Item = &mut PlanNode>;
}
```

Wrapper-level delegation: `SemanticPlan::walk_pre` / `::walk_post` / `::transform` call the corresponding `PlanNode` method on `root`, returning any diagnostic the rewrite surfaces alongside the transformed tree.

### `SemanticPlan::validate`

```rust
impl SemanticPlan {
    /// Full tree walk; re-checks every invariant in §13. Returns the
    /// first violation as a boxed `Diagnostic<IrErrorKind>`; `Ok(())`
    /// on well-formedness.
    ///
    /// Intended use: debug / test harnesses, planner-regression
    /// catches, audit tools. Production callers rely on the planner's
    /// well-formedness guarantee and SHOULD NOT validate on every plan.
    pub fn validate(&self) -> Result<(), Box<Diagnostic<IrErrorKind>>>;
}
```

The error is **boxed**. `validate()` is the one fail-fast Result in the IR surface that returns a single `Diagnostic<IrErrorKind>` *by value* — a ~136-byte rich envelope. Validation succeeds on the overwhelming majority of plans, so boxing keeps the hot `Ok` path pointer-sized.

### Typical Usage Patterns

**Count nodes of a variant.** Implement `PlanVisitor` with `Output = ()` and a counter field.

**Extract all `Scan`'s sources.** Implement `PlanVisitor` collecting `&SourceRef` from every `PlanNode::Scan(ScanNode { source, .. })`.

**Push-down rewrite.** Implement `transform` with a closure that matches `Filter { input: box Scan { .. }, predicate }` and rebuilds the subtree with the predicate in `filters_pushdown`.

**Schema re-check.** Implement `PlanVisitor<Output = Result<(), Diagnostic<IrErrorKind>>>` that recomputes `output_schema` for each variant and compares to `meta().output_schema`.

## Testing Strategy

This leaf owns `SemanticPlan`, `PlanNode`, the 8 variant structs, tree invariants, and traversal API. Tests:

- **Well-formedness invariants (§13):** `SemanticPlan::validate()` exercises all invariant checks; unit tests verify each `IrErrorKind` variant (see [ir-errors.md](ir-errors.md)) is raised for the corresponding violation (e.g., `FilterPredicateNotBoolean` when `predicate.inferred_type() != DataType::Boolean`).
- **Pass-through schema enforcement:** Unit tests verify `Filter` / `Sort` / `Fetch` nodes preserve `input.meta().output_schema` (cheap via `Arc` share).
- **Serde roundtrip:** Feature-gated tests verify `SemanticPlan` serializes to deterministic JSON and deserializes to a structurally-equal plan (see [ir-serde-substrait.md](ir-serde-substrait.md)).
- **PlanVisitor / traversal:** Unit tests verify `walk_pre` / `walk_post` visit every node in the expected order; `transform` applies bottom-up rewrite correctly.
- **Partition info exclusion:** Integration tests verify `ScanNode` carries no partition fields; all partition metadata is manifest-side (reachable via `SourceRef`).

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
