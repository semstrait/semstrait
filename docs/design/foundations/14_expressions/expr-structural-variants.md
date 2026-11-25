---
id: 14-expr-structural-variants
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: Window variant exists per A2 (author-rejected, compile-emitted); semstrait-model owns all variants
---

# Structural-Variant Catalog

This leaf defines the structural-variant taxonomy of `Expr<L>` — the recursive operators (BinaryOp, UnaryOp, FunctionCall, Cast, Case, InList, Between, Like, IsNull, Coalesce, NullIf, Aggregate, Window) shared across every instantiation. The `Window` variant is author-rejected but present on the shared `Expr<L>` to support compile-time sugar elimination. See [yaml-authoring.md](yaml-authoring.md) for reserved-tag mappings and [crate-placement.md](crate-placement.md) for ownership.

<!-- cross-cut: none -->

### 3.3 `Expr<L>` — structural-variant catalog

The structural variants are declared once and parameterised over `L`. Every structural variant is `#[non_exhaustive]` per I10. The variants, shared by every instantiation: `Leaf(L)`, `BinaryOp`, `UnaryOp`, `FunctionCall`, `Cast`, `Case`, `InList`, `Between`, `Like`, `IsNull`, `Coalesce`, `NullIf`, `Aggregate`, `Window`. The `Window` variant is **author-rejected, not absent** (item Y / V3 as amended by A2): it exists on the shared `Expr<L>` — so the compile sugar-elimination fold can construct it over the *semantic* leaf set (§4.2) — but no author-facing parser accepts window syntax (§3.3 note, §7). Crate-of-record: the shared `Expr<L>` (all instantiations) is `[32](../../apis/32_semstrait_model.md)`; the resolved alias `ir::PhysicalExpr = Expr<ir::PhysicalLeaf>` is `[35 §3.3](../../apis/35_semstrait_ir.md)`.

```rust
#[non_exhaustive]
pub enum Expr<L: ExprLeaf> {
    Leaf(L),

    BinaryOp     { op: BinaryOpKind, left: Box<Self>, right: Box<Self> },
    UnaryOp      { op: UnaryOpKind,  operand: Box<Self> },
    FunctionCall { name: CanonicalFn, args: Vec<Self> },
    Cast         { input: Box<Self>, target: DataType, on_failure: CastFailure },
    Case         { whens: Vec<(Self, Self)>, else_: Option<Box<Self>> },
    InList       { value: Box<Self>, list: Vec<Self>, negated: bool },
    Between      { value: Box<Self>, low: Box<Self>, high: Box<Self>, negated: bool },
    Like         { value: Box<Self>, pattern: Box<Self>, kind: LikeKind },
    IsNull(Box<Self>),
    Coalesce(Vec<Self>),
    NullIf       { left: Box<Self>, right: Box<Self> },
    Aggregate    { op: AggregateKind, args: Vec<Self>, distinct: bool, filter: Option<Box<Self>> },
    Window       { function: WindowFn, args: Vec<Self>, partition_by: Vec<Self>, order_by: Vec<Self>, frame: Option<WindowFrame> },
    // `Window` is author-rejected (§3.3 note, §7) — compile-emitted via §4.2 sugar lowering, never parsed.
}
```

**`Expr<L>` is a recursive tree, homogeneous in `L`.** Every non-`Leaf` variant nests its operands as `Box<Self>` / `Vec<Self>` — i.e. `Expr<L>` over the **same leaf set `L`** (the `Box` heap-breaks the recursion at each branch). A node never nests a *different* leaf family: an `UnresolvedSemanticExpr` (`Expr<UnresolvedSemanticLeaf>`) contains only `UnresolvedSemanticExpr` children, and an `UnresolvedPhysicalExpr` (`Expr<UnresolvedPhysicalLeaf>`) only `UnresolvedPhysicalExpr` children — the two authoring families never interleave inside one tree. Leaves terminate the recursion; everything above a leaf is structure. This is the property the `Tree` traversal contract (§3.1) relies on, and it holds identically for the resolved alias `ir::PhysicalExpr` (`[35 §3.3](../../apis/35_semstrait_ir.md)`), which is the same `Expr<L>` instantiated at `ir::PhysicalLeaf`.

**Notes on the structural catalog:**

- `BinaryOpKind` covers the canonical arithmetic / comparison / logical operators (`Add`, `Subtract`, `Multiply`, `Divide`, `SafeDivide`, `Mod`, `Eq`, `NotEq`, `Lt`, `LtEq`, `Gt`, `GtEq`, `And`, `Or`). `UnaryOpKind` covers `Negate` and `Not`. Both enums are `#[non_exhaustive]`.
- `CanonicalFn` is the stable canonical function identity per `[14a](../14a_function_catalog.md)`; the registry is the single source of truth for which names exist.
- `Aggregate`'s `filter` field carries the canonical `agg(expr) FILTER (WHERE p)` shape; adapter compensation for engines without native `FILTER` is the adapter's concern (not part of the canonical IR).
- `Window` is in the structural catalog (rather than being a leaf) because its inner operands recurse over `Self`. In practice, `Window` is **compile-emitted only** — author-facing parsers do not accept window syntax; `Window` nodes enter the tree exclusively through sugar-accessor elimination during compile (§4.2 — typed leaves with `accessor: Some(_)` lower to `Window`-rooted subtrees).
- Engine-specific operators or function-shaped predicates do **not** add `Expr<L>` variants. They land as `FunctionCall` entries via `FunctionRegistry` extensions per `[14a](../14a_function_catalog.md)`.

## Testing Strategy

- **Structural completeness:** Every structural variant in §3.3 exists exactly once on the shared `Expr<L>` in `semstrait-model`.
- **Non-exhaustive enforcement:** Every structural variant and support enum carries `#[non_exhaustive]` per I10.
- **Window parse rejection:** Author-facing parsers reject window syntax (author-rejected gate). A `Window` node is constructible only via Rust DSL or compile-time sugar elimination.
- **Compile construction:** Typed leaves with `accessor: Some(_)` successfully lower to `Window`-rooted subtrees (§4.2 sugar-elimination test per [accessor-sugar.md](accessor-sugar.md)).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
