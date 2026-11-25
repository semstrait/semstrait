---
id: 35-ir-plan-primitives
status: ratified
version: 2.1
impl: implemented
---

# IR Plan Primitives

Shared primitives referenced across `PlanNode` variants. These types are plan-level vocabulary; `PlanNode` references every primitive but not vice-versa, so primitives can be consumed without the full `PlanNode` surface.

## `NodeMeta`

```rust
/// Metadata attached to every `PlanNode`. Per `15 §7` and `17 §3`.
#[non_exhaustive]
pub struct NodeMeta {
    /// Per-process opaque identifier for this node, scoped to a single
    /// `SemanticPlan` lifetime. Used by the optimizer for rule-engine
    /// source tracking. NOT a content hash — two structurally-equal
    /// plans MAY have different `NodeId`s, and a `transform` that rebuilds
    /// the same shape MAY produce a fresh `NodeId`. Comparing `NodeId`s
    /// across processes or serialize-rehydrate cycles is undefined.
    pub node_id: NodeId,

    /// Output schema after this node. `Arc` allows pass-through
    /// nodes (Filter, Sort, Fetch) to share the parent schema without
    /// deep-cloning.
    pub output_schema: std::sync::Arc<Schema>,

    /// Plan-level annotations. Non-computational metadata only — `34`
    /// and `36` MUST NOT make planning or emission decisions based on
    /// `annotations` content. Two classes: **TRACE** (descriptive
    /// provenance) and **PLAN** (advisory hints for tools).
    pub annotations: Vec<SemAnnotation>,
}
```

`NodeId` is a newtype over `Uuid::new_v4()` in v1; external consumers should treat it as opaque. `Schema` is the canonical structural schema owned at [ir-type-vocabulary.md](ir-type-vocabulary.md) — `{ columns: Vec<SchemaColumn> }`.

### `SemAnnotation` — Variant Inventory and Classification

`SemAnnotation` is an additive `#[non_exhaustive]` sum. Variants split into two classes per S5 (see [ir-crate-charter.md](ir-crate-charter.md) §1.5):

- **PLAN** class — read by `34` / `36` as part of the IR-to-consumer contract (advisory hints, never dispatch). Renaming or removing a PLAN variant is MAJOR per `30 §2.1`.
- **TRACE** class — descriptive only; never read by `34` / `36`. Adding, renaming, or removing a TRACE variant is MINOR.

```rust
/// Per-node annotation. Per S5.
///
/// Variants are classified TRACE or PLAN. See per-variant doc-comments.
#[non_exhaustive]
pub enum SemAnnotation {
    // ───── PLAN class ─────

    /// Role of an aggregate node in additivity resolution.
    /// PLAN — read by `34`'s additivity reasoning.
    AggregateRole(AggregateRole),

    /// Source of a filter — which authoring site introduced it.
    /// PLAN — read by `34`'s `19 §9.1` filter-placement logic.
    FilterSource(FilterSource),

    /// Effective-additivity hint for a measure subtree.
    /// PLAN — read by `34`'s lossy-reaggregation advisory channel
    /// per `19 §6.5`.
    Additivity(AdditivityAnnotation),

    // ───── TRACE class ─────

    /// DataKind that contributed this subtree.
    /// TRACE — for tools, debuggers, SQL pretty-printers.
    DataKindRef(SemanticsName),

    /// Boundary marker emitted by a planner strategy when one DataKind's
    /// strategy expansion stops contributing nodes and another begins.
    /// TRACE — for tools that visualize plan-tree composition.
    StrategyBoundary {
        type_: SemanticsName,
        position: BoundaryPosition,
    },
}

/// Position of a `StrategyBoundary` annotation relative to the strategy's
/// emitted subtree.
#[non_exhaustive]
pub enum BoundaryPosition {
    Entry,  // First node a strategy emitted
    Exit,   // Last node a strategy emitted
}
```

PLAN-class variants `AggregateRole` / `FilterSource` / `Additivity` (and their support enums) are ratified in `34`'s planner contract; `35` carries the structural shape only. TRACE-class variants `DataKindRef` / `StrategyBoundary` are emitted by planner strategies to mark subtree origin.

## `SourceRef`

```rust
/// Opaque handle to a source in the SemanticManifest. Constructed by
/// the planner; resolved by the adapter against the SemanticManifest
/// it was handed alongside the `SemanticPlan`. No path, URL, catalog
/// name, or file format leaks into the plan tree. I1 guarantee.
///
/// Newtype-over-stable exception per `30 §4.3`: no `#[non_exhaustive]`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SourceRef(/* crate-private */);
```

## `ResolvedColumn`

```rust
/// A column as projected by a `Scan`. Per `15 §4.2`.
#[non_exhaustive]
pub struct ResolvedColumn {
    pub name: Name,
    pub data_type: DataType,
    pub nullable: bool,
    /// Ordinal in the underlying source's native schema order. Adapters
    /// consume this when emitting stable column references.
    pub ordinal: u32,
}
```

## `Name`

```rust
/// Identifier used for output-column names, group-by keys, sort keys,
/// and projection aliases. A plan-level newtype over `String` with a
/// construction boundary that enforces identifier well-formedness:
///
/// - Non-empty.
/// - UTF-8 (guaranteed by `String`).
/// - Not a reserved plan-tree tag (see below).
///
/// `Name` is **not** normalized (case-folding, whitespace-collapsing)
/// — two distinct `Name`s may differ only in case, matching
/// case-preserving-but-case-sensitive semantics from `11 §5`.
///
/// Newtype-over-stable exception per `30 §4.3`: no `#[non_exhaustive]`.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Name(String);

impl Name {
    /// Validates the identifier; returns `IrErrorKind::InvalidName` on
    /// violation. Bare-kind shape per `31 §3.1` construction-site convention.
    pub fn new(s: impl Into<String>) -> Result<Self, IrErrorKind>;

    pub fn as_str(&self) -> &str;
    pub fn into_string(self) -> String;
}
```

### Reserved Plan-Tree Tags

Substrait-roundtrip fidelity reserves a small set of identifier prefixes for semstrait's own use: `__semstrait_`, `__plan_`, `__agg_`. Constructing a `Name` with one of these prefixes raises `IrErrorKind::ReservedName` (see [ir-errors.md](ir-errors.md)). The reserved-prefix set is additive; adding new prefixes is MINOR.

## `KeyPair`

```rust
/// One join-key pair on a `JoinNode.on`. Per `16 §5.1`.
///
/// Both `left` and `right` are column names resolving against the
/// join's corresponding child's `output_schema`. Column types must
/// match per §13.5 (see [ir-plan-nodes.md](ir-plan-nodes.md)); a mismatch
/// is `IrErrorKind::JoinKeyTypeMismatch`.
#[non_exhaustive]
pub struct KeyPair {
    pub left: Name,
    pub right: Name,
}
```

## `SortDir` / `NullOrdering`

```rust
/// Sort direction + null-ordering bundle. Per Substrait's
/// `SortField.direction`.
#[non_exhaustive]
pub enum SortDir {
    /// Ascending order.
    Asc  { nulls: NullOrdering },
    /// Descending order.
    Desc { nulls: NullOrdering },
}

/// Where to place `NULL` values in a sort.
#[non_exhaustive]
pub enum NullOrdering {
    /// Place `NULL` values first.
    First,
    /// Place `NULL` values last.
    Last,
    /// Let the adapter choose (SQL default: ASC NULLS LAST, DESC NULLS
    /// FIRST — most engines; adapters MAY narrow).
    Unspecified,
}

impl SortDir {
    pub const ASC_NULLS_FIRST:  SortDir = SortDir::Asc { nulls: NullOrdering::First };
    pub const ASC_NULLS_LAST:   SortDir = SortDir::Asc { nulls: NullOrdering::Last  };
    pub const DESC_NULLS_FIRST: SortDir = SortDir::Desc { nulls: NullOrdering::First };
    pub const DESC_NULLS_LAST:  SortDir = SortDir::Desc { nulls: NullOrdering::Last  };
}
```

Keeping null-ordering on `SortDir` (rather than a separate field) keeps the per-key shape tight and matches Substrait's wire form.

## `AggregateExpr`

```rust
/// Plan-level carrier for an aggregate kernel on `AggNode.aggregates`,
/// hoisted from a `PhysicalExpr` by Phase B aggregate-lift per `19 §7`.
/// No `Expr::Aggregate` survives inside a predicate/projection slot (§13.1
/// in [ir-plan-nodes.md](ir-plan-nodes.md)).
#[non_exhaustive]
pub struct AggregateExpr {
    pub aggregation:   AggregateKind,    // matches Expr::Aggregate.op (14 §3.3)
    pub args:          Vec<PhysicalExpr>, // inner arguments; n-ary to admit extension aggregates
    pub distinct:      bool,             // matches Expr::Aggregate.distinct
    pub filter:        Option<PhysicalExpr>,  // FILTER (WHERE ...) clause
    pub inferred_type: DataType,         // populated by Phase B
}
```

## Invariants Enforced at Construction

Each `PlanNode` variant's struct is constructed directly (no hidden builder); the variant's field combination is the contract. Schema invariants (§13 in [ir-plan-nodes.md](ir-plan-nodes.md)) are *not* checked at construction — consumers rely on the planner to produce well-formed trees and rely on `SemanticPlan::validate()` for a debug-only full re-check.

## Testing Strategy

This leaf owns plan-level shared primitives (`NodeMeta`, `SourceRef`, `ResolvedColumn`, `Name`, `KeyPair`, `SortDir`, `NullOrdering`, `AggregateExpr`, `SemAnnotation`). Tests:

- **`Name` validation:** Unit tests verify `Name::new` rejects empty strings and reserved prefixes (`__semstrait_`, `__plan_`, `__agg_`); raises `IrErrorKind::InvalidName` / `IrErrorKind::ReservedName` (see [ir-errors.md](ir-errors.md)).
- **`SourceRef` opacity:** Integration tests verify `SourceRef` is opaque (no path/URL/format accessible on the plan tree); resolution happens adapter-side via SemanticManifest lookup.
- **`ResolvedColumn` ordinal stability:** Unit tests verify `ordinal` field matches the source's native schema order (Parquet footer, catalog fetch).
- **`SortDir` null-ordering:** Unit tests verify `ASC_NULLS_FIRST` / `ASC_NULLS_LAST` / `DESC_NULLS_FIRST` / `DESC_NULLS_LAST` const values match the expected `SortDir { nulls }` combinations.
- **`AggregateExpr` lift postcondition:** Integration tests verify Phase B aggregate-lift (see `19 §7`) extracts `Expr::Aggregate` nodes into `AggNode.aggregates` as `AggregateExpr`; no `Expr::Aggregate` survives in predicate/projection slots (§13.1 in [ir-plan-nodes.md](ir-plan-nodes.md)).
- **`SemAnnotation` classification:** Unit tests verify PLAN-class variants (`AggregateRole`, `FilterSource`, `Additivity`) are read by `34` (planner contract); TRACE-class variants (`DataKindRef`, `StrategyBoundary`) are emitted but never read by `34` / `36`.

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
