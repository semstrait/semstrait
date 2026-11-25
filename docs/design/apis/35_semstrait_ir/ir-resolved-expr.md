---
id: 35-ir-resolved-expr
status: ratified
version: 2.1
impl: implemented
---

# IR Resolved Expression

The resolved IR expression type `ir::PhysicalExpr` is the **alias** `Expr<PhysicalLeaf>` over the model-owned shared `Expr<L>` (item Y / V3 as amended by A2). There is no resolved-only structural enum; the shared tree carries every variant (including `Window`), and `35` specializes it to the ir-owned `PhysicalLeaf` set that admits `Parameter`.

## Where the Type Architecture Lives

The structural shape of canonical expressions is ratified by `14 §3`. Under item Y the **homes** split three ways:

- `14 §3.1` ratifies the universal `Tree` trait and its `Visitor` / `Rewriter` companions — homed in **`semstrait-common`** (V5), imported here.
- `14 §3.2` ratifies the `ExprLeaf` trait — homed in **`semstrait-model`** (V5, it names `DataType`), imported here.
- `14 §3.3` ratifies the structural-variant catalog — every variant (`BinaryOp`, `UnaryOp`, `FunctionCall`, `Cast`, `Case`, `InList`, `Between`, `Like`, `IsNull`, `Coalesce`, `NullIf`, `Aggregate`, **and `Window`** under item Y / V3 as amended by A2) on the single shared `Expr<L>`, homed in **`semstrait-model`** (V3 / V6).

`35` is the crate that holds the **resolved** expression type `ir::PhysicalExpr` — the **alias** `Expr<PhysicalLeaf>` over the `PhysicalLeaf` set that admits `Parameter`. Per `14 §9.2`, the shared `Expr<L>` family relocated to `semstrait-model` and `35` now **imports** it; the `model → ir` edge inverted to `ir → model` (V1). `35` does not re-ratify any variant catalog; catalogs are `14 §3.3`'s contract.

## The `Tree` / `Visitor` / `Rewriter` / `ExprLeaf` Trait Surface

Per `14 §3.1` / `§3.2` and item-Y placement in `14 §9.1`, the universal-traversal trait family is owned by **`semstrait-common`** (V5) and `ExprLeaf` by **`semstrait-model`** (V5). `35` imports all four:

```rust
/// Universal traversal contract. Implemented by Expr<L> and PlanNode.
/// Stage-agnostic. Per `14 §3.1`.
pub trait Tree: Sized {
    fn children(&self) -> Vec<&Self>;
    fn with_new_children(self, new_children: Vec<Self>) -> Result<Self, ValidateError>;
}

pub trait TreeExt: Tree {
    fn apply<V: Visitor<Self>>(&self, v: &mut V) -> V::Output { /* default body */ }
    fn transform<F>(self, f: F) -> Result<Self, ValidateError>
    where F: FnMut(Self) -> Result<Self, ValidateError> { /* default body */ }
}

impl<T: Tree> TreeExt for T {}

pub trait Visitor<N> {
    type Output;
    fn f_down(&mut self, node: &N) -> ControlFlow<Self::Output>;
    fn f_up(&mut self,   node: &N) -> ControlFlow<Self::Output>;
}

pub trait Rewriter<N> {
    fn f_down(&mut self, node: N) -> Result<N, ValidateError>;
    fn f_up(&mut self,   node: N) -> Result<N, ValidateError>;
}

/// Per-leaf-set metadata contract. Implemented by PhysicalLeaf and UnresolvedSemanticLeaf.
/// Per `14 §3.2`.
pub trait ExprLeaf: Sized + Clone + Debug {
    fn inferred_type(&self) -> Option<&DataType>;
}
```

Consumers write `use semstrait_common::{Tree, TreeExt}`. The single trait surface lets one generic algorithm operate on both expression trees and plan trees.

`ValidateError` (the `with_new_children` / `Rewriter<N>::f_*` return type) is owned by **`semstrait-common`** (item Y / V5) — it is **not** an `ir`-owned error. `35` imports it; the only ir-emitted error kind is `CompileError` (see [ir-errors.md](ir-errors.md)).

## The Resolved `ir::PhysicalExpr` Definition

The full variant catalog is per `14 §3.3`. Under item Y / V3 **as amended by A2**, there is exactly **one** structural enum — the shared `Expr<L>`, owned by `semstrait-model`, carrying every variant (the `Window` included). `35` declares no expression enum of its own; `ir::PhysicalExpr` is the **alias** that pins that shared tree to the ir-owned `PhysicalLeaf`:

```rust
use semstrait_model::Expr;

/// Resolved canonical expression tree. NOT a re-declared enum — an alias for
/// the shared `Expr<L>` (owned by `semstrait-model`, full catalog per `14 §3.3`,
/// `Window` included) pinned to the ir-owned `PhysicalLeaf` set (which admits
/// `Parameter`). Compile lowers `model::UnresolvedSemanticExpr` into this form
/// (`19 §3`); `#[non_exhaustive]` lives on the shared enum (I10).
pub type PhysicalExpr = Expr<PhysicalLeaf>;
```

A2 collapses the duplication V3 had weighed: one structural enum, one `Tree` impl (`impl<L: ExprLeaf> Tree for Expr<L>`, owned with the enum in model), parameterized over `L` — `35` re-uses both by aliasing, declaring nothing. The `BinaryOpKind` / `UnaryOpKind` / `CanonicalFn` / `CastFailure` / `LikeKind` / `AggregateKind` operands, the `DataType` cast target, the `WindowFn` / `WindowFrame` operands, and the `PhysicalLeaf::Literal` carrier are all **imported from `semstrait-model`** (item Y / V2 / V4 / V10 / A2). Per `14 §3.3`:

- `Window` is a variant on the shared `Expr<L>` but is **compile-emitted only** — author-facing parsers reject window syntax (parse/dump guard, `14 §7`); `Window` nodes enter a resolved tree exclusively through sugar-accessor elimination during compile.
- Engine-specific operators do not add `PhysicalExpr` variants. They land as `FunctionCall` entries via `FunctionRegistry` extensions per `14a §7`.
- `Aggregate`'s `filter` carries the canonical `agg(expr) FILTER (WHERE p)` shape; adapter compensation for engines without native `FILTER` is the adapter's concern (`36`), not part of the canonical IR.

## Support Enums — All Imported

The shared `Expr<L>`'s variants reference support enums and identifier carriers. Under item Y / V10 **as amended by A2**, **every** support enum is **owned by `semstrait-model`** and imported here — including the four Window enums, which followed the `Window` variant onto the shared tree (A2). `35` owns no support enum. Rosters per `14 §3.3`.

```rust
// ── All imported from semstrait-model (item Y / V10 / V4 / A2) ──
// Authoring support enums + carriers: BinaryOpKind, UnaryOpKind, AggregateKind,
// AggregationOp, LikeKind, CastFailure, Literal, IntegerWidth, FloatWidth,
// ColumnRef, SemanticsName, CanonicalFn (name).
// Window enums (A2 — follow the Window variant onto the shared Expr<L>):
// WindowFn, WindowFrame, WindowFrameKind, WindowBound.
use semstrait_model::{
    BinaryOpKind, UnaryOpKind, AggregateKind, AggregationOp, LikeKind, CastFailure,
    Literal, IntegerWidth, FloatWidth, ColumnRef, SemanticsName, CanonicalFn,
    WindowFn, WindowFrame, WindowFrameKind, WindowBound,
};
```

For reference, the Window enums (rosters ratified by `14 §3.3`, now homed in `32`):

```rust
// semstrait-model, carried by the shared Expr<L>::Window variant.
// Window nodes are compile-emitted only via sugar-accessor elimination (14 §4.2).
#[non_exhaustive]
pub enum WindowFn { Lag, Lead, FirstValue, LastValue, RowNumber, Rank, DenseRank }

#[non_exhaustive]
pub struct WindowFrame { pub kind: WindowFrameKind, pub start: WindowBound, pub end: WindowBound }

#[non_exhaustive]
pub enum WindowFrameKind { Rows, Range, Groups }

#[non_exhaustive]
pub enum WindowBound {
    UnboundedPreceding, Preceding(u64), CurrentRow, Following(u64), UnboundedFollowing,
}
```

The authored `DataType` / `Grain` are also imported from `semstrait-model` (item Y / V2); the non-authored `TypeClass` / `Schema` / `SchemaColumn` are owned at [ir-type-vocabulary.md](ir-type-vocabulary.md). `DataType` flows through both the structural variants (`Cast::target: DataType`) and the imported support enums.

**Construction-time validation.** `Literal` parses bare YAML scalars into typed shape at parse time in `semstrait-model` (`32 §5`); `Decimal { precision, scale }` range validation runs at construction in the owning crate.

## `PhysicalLeaf` — Owned Here

Canonical-IR leaf set per `14 §3.4`. The full variant catalog and per-variant invariants are owned by `14`; `35` carries the implementation.

```rust
/// Canonical-IR leaf set. Carries exactly what the planner and adapters
/// need. Variant catalog per `14 §3.4`.
#[non_exhaustive]
pub enum PhysicalLeaf {
    /// Physical column reference (binding-resolved).
    Column(ColumnRef),

    /// Typed literal value.
    Literal(Literal),

    /// Compile-emitted, plan-bound parameter placeholder. Replaced with a
    /// concrete value during Phase B planning per `14 §5.3` / `19 §2.1`.
    Parameter(Parameter),
}

impl ExprLeaf for PhysicalLeaf { /* per-variant inferred_type */ }
```

Invariants on `PhysicalExpr` (per `14 §3.4`):

- No `Field` / `Dimension` / `Measure` / `Metric` / `Key` — semantic references are eliminated during compile per `19 §3`.
- No sugar accessors — typed-leaf-with-accessor leaves are eliminated during compile (lowered to `Window`-rooted subtrees per `14 §4.2`).
- `Parameter` leaves are the only non-resolved state the canonical IR carries; they MUST be substituted before adapt time (`14 §5.3` postcondition).

## `UnresolvedSemanticLeaf` — Imported

```rust
use semstrait_model::UnresolvedSemanticLeaf;
```

The per-kind typed leaf set is **owned by `semstrait-model`** (item Y / V6; full body + per-variant invariants at `32`, catalog ratified by `14 §3.5`). `35` imports it only where the compile-lowering signature names it (`model::UnresolvedSemanticExpr → ir::PhysicalExpr`). Its variant tags (`Literal`, `Column`, `Field`, kind-pinned `Dimension` / `Measure` / `Metric` / `Key` with `Option<…Accessor>`) and the manual-vs-auto `Column` rule are author-surface concerns; the resolved IR never persists a `UnresolvedSemanticLeaf` value — compile eliminates it (`19 §3`). `UnresolvedSemanticLeaf` carries **no `Parameter`** — parameters are compile-emitted and live only in the ir-owned `PhysicalLeaf`.

## Type Names

The resolved IR exposes exactly one expression type name:

```rust
pub type PhysicalExpr = Expr<PhysicalLeaf>;  // alias
```

`PhysicalExpr` **is** the `Expr<PhysicalLeaf>` alias. The authoring spellings are **imported** from `semstrait-model`:

```rust
use semstrait_model::{Expr, UnresolvedSemanticExpr, UnresolvedPhysicalExpr};
// model: Expr<L>                  = the one shared structural enum (incl. Window)
//        UnresolvedSemanticExpr   = Expr<UnresolvedSemanticLeaf>
//        UnresolvedPhysicalExpr   = Expr<UnresolvedPhysicalLeaf>
```

There is **no `ir::SemanticExpr`** (item Y / V6 deletes it — the authoring semantic form is `model::UnresolvedSemanticExpr`). Downstream docs and downstream-crate APIs spell the resolved type `ir::PhysicalExpr` and the authoring form `model::UnresolvedSemanticExpr`; both are aliases of the same model-owned `Expr<L>` at different leaf sets.

### Expression-Pool ID Newtype

The manifest persists a **single, physical-only** expression pool (`33 §7.2`); the authoring `model::UnresolvedSemanticExpr` is lowered away at compile and not persisted. The pool handle is one newtype:

```rust
/// Identifier of an entry in the manifest's physical expression pool
/// (keys `BTreeMap<PhysicalExprId, ManifestExpression>` per `33 §7.2`).
/// Newtype-over-stable per `30 §4.3`: no `#[non_exhaustive]`; `.0` access intentional.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PhysicalExprId(pub ExprId);
```

`ExprId` is the underlying content-keyed identifier carrier; `PhysicalExprId` wraps it without altering its representation. With a single pool there is no cross-pool mix-up hazard; the earlier `SemanticExprId` (and the `semantic` pool) are retired (supersedes C12.3's split-pool framing).

### Expression Resolvability Path

The expression lifecycle is intentionally two-stage and lookup-only after compile:

1. **Authoring.** Parse produces `model::UnresolvedSemanticExpr` at semantic sites (or `model::UnresolvedPhysicalExpr` for physical-mapping sites) per `14` / `19`.
2. **Compile.** `semstrait-manifest::compile` resolves semantic leaves, runs gates G1-G5, lowers each `model::UnresolvedSemanticExpr → ir::PhysicalExpr`, classifies its `ExprLayer`, and emits physical-pool entries (`ManifestExpression { expr, layer }`) keyed by `PhysicalExprId` (`33 §7.2`). The semantic form is not persisted.
3. **Graph build.** `SemanticGraph` references pool entries through `GraphExprRef` (physical-only); unresolved ids fail graph build and no untyped `ExprId` dispatch is allowed.
4. **Planning.** Planner consumes graph/manifest references by lookup only; no re-resolution pass mutates expression identity.

## Type-Enforced Forbidden Combinations

Per `14 §3.7`, the leaf-set boundary makes several invariants type-level:

- `ir::PhysicalExpr` cannot contain `Field` / `Dimension` / `Measure` / `Metric` / `Key` — those variants do not exist in `PhysicalLeaf`. The static type system, not a runtime check, upholds this.
- `model::UnresolvedSemanticExpr` cannot contain `Parameter` — `Parameter` is `PhysicalLeaf`-only (and the authoring `UnresolvedPhysicalLeaf` lacks it too); parameters are compile-emitted into the resolved `ir::PhysicalExpr` exclusively.
- A `Dimension`-tagged leaf cannot carry a `MeasureAccessor` — the variant signature `Dimension { name, accessor: Option<DimensionAccessor> }` enforces kind agreement at construction.

There is no `try_into_physical` runtime check, no defensive `panic!` for "Field found in `ir::PhysicalExpr`". `UnresolvedSemanticLeaf::Column` is type-admissible but context-validated by compile per the manual-vs-auto rule.

## Expression Layer (`ExprLayer`)

A lowered `PhysicalExpr` is "just a tree", but **where its result applies in the relational pipeline differs**: a grouping-key scalar belongs *below* the GROUP BY (pushdown-eligible), a Measure aggregation *is* the GROUP BY, and a ratio Metric's outer formula belongs *above* it (after any union/join re-aggregation). The planner needs this placement to allocate expressions optimally across `PlanNode`s and across complex-DataKind composition. `ExprLayer` is that classification — the canonical home for the "what layer" metadata persisted on every manifest expression (`33 §7.2`).

```rust
/// The relational-pipeline layer at which an expression's RESULT applies.
/// Determined at compile from the expression's structure (presence and
/// position of `Aggregate` / `Window` nodes), not from the authoring kind.
#[non_exhaustive]
pub enum ExprLayer {
    /// Row-level: the tree contains no aggregation. Evaluated before grouping;
    /// pushdown-eligible (Scan / pre-Agg Project). Grouping-key Dimensions, Keys.
    Scalar,
    /// Grouped aggregation: after Phase-B aggregate-lift (`19 §7`), the result
    /// is produced by an `Aggregate` over row-level inputs. Measures.
    /// Pre-/re-aggregation safety is governed by the aggregation's `Additivity`
    /// (`14a §3.6`), derived from the aggregate op in v1.
    Aggregate,
    /// Computed over already-aggregated inputs: one or more `Aggregate` nodes
    /// sit strictly below a scalar root. Ratio / derived Metrics, and any
    /// Dimension whose `expr` references a Measure. Evaluated after the final
    /// aggregation (and after any union/join re-aggregation).
    PostAggregate,
}
```

Design notes:

- **Tree-derived, not role-derived.** `ExprLayer` is a pure function of the expression's structure, decoupled from the authoring `SemanticRole` (`33 §5`). A bucketed Dimension `case_when(revenue > 1000, …)` over a Measure `revenue` classifies as `PostAggregate` even though its role is `Dimension`.
- **Computed once at compile.** Phase A assigns the layer when it lowers each `model::UnresolvedSemanticExpr → ir::PhysicalExpr` (`19 §3`); it is persisted on the manifest expression record (`33 §7.2`) and load-validated against the tree (CX1).
- **Dedup-safe.** Because the layer is a function of the tree, content-deduplicated pool entries (same tree) always carry the same layer.
- **Window classification (v1).** Window-bearing expressions (per-entity accessor sugar) classify by their relationship to aggregation: a window over the aggregated grain (e.g. `measure.previous`) is `PostAggregate`; a window over raw rows is `Scalar`. A dedicated `Windowed` layer is a reserved `#[non_exhaustive]` extension if windowed pushdown needs finer placement.

## Testing Strategy

This leaf owns the resolved `ir::PhysicalExpr` (alias Expr<PhysicalLeaf>), `PhysicalLeaf` set, `PhysicalExprId`, and `ExprLayer`. Tests:

- **Type-level forbidden-combination gates:** The static type system enforces `PhysicalExpr` cannot contain `Field` / semantic leaves — no runtime check needed. Integration-test verifies `Parameter` exists only on `PhysicalLeaf`, not on authoring leaf sets.
- **ExprLayer classification:** Unit tests per `19 §3.2.6` verify `Scalar` / `Aggregate` / `PostAggregate` assignment from tree structure (no aggregation → Scalar; single top `Aggregate` → Aggregate; aggregation strictly below scalar root → PostAggregate).
- **PhysicalExprId pool resolution:** Load-time integrity check CX1 (see [ir-serde-substrait.md](ir-serde-substrait.md) §15.4) verifies every `PhysicalExprId` referenced outside `ManifestExpressions` resolves to an existing pool entry.
- **`Tree` impl for `Expr<PhysicalLeaf>`:** Unit tests verify `children()` and `with_new_children` follow `14 §3.1` contract; a malformed `with_new_children` raising `ValidateError` is the expected behavior (see [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)).
