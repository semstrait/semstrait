---
id: 14-tree-trait
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: semstrait-common per item Y / V5; orphan-safe per-type impl; ValidateError co-moved
---

# Tree Traversal Trait

This leaf defines the universal traversal contract — the `Tree` trait for children/with_new_children/visitor/rewriter and the `ExprLeaf` trait for leaf metadata. These contracts apply family-agnostically across `Expr<L>` (any leaf set) and also to `ir::PlanNode`. See [crate-placement.md](crate-placement.md) for ownership (both traits live in `semstrait-common`).

## 3. Type Architecture

### 3.1 The `Tree` trait — universal traversal

The traversal contract is **node-shape-agnostic** — it works for `Expr<L>` regardless of leaf set, and may also be implemented for plan-tree nodes per `[35](../../apis/35_semstrait_ir.md)`. The trait is therefore stage-agnostic.

```rust
pub trait Tree: Sized {
    fn children(&self) -> Vec<&Self>;
    fn with_new_children(self, new_children: Vec<Self>) -> Result<Self, ValidateError>;
}
```

Default-derived helpers on `Tree` (no separate trait method required):

```rust
pub trait TreeExt: Tree {
    /// Pre-order read-only walk.
    fn apply<V: Visitor<Self>>(&self, v: &mut V) -> V::Output { /* default body */ }

    /// Bottom-up rewrite.
    fn transform<F>(self, f: F) -> Result<Self, ValidateError>
    where
        F: FnMut(Self) -> Result<Self, ValidateError>,
    {
        /* default body */
    }
}

impl<T: Tree> TreeExt for T {}
```

`Visitor` / `Rewriter` companion traits follow the `f_down` / `f_up` shape (compatible with the pattern documented for canonical tree-traversal libraries):

```rust
pub trait Visitor<N> {
    type Output;
    fn f_down(&mut self, node: &N) -> ControlFlow<Self::Output>;
    fn f_up(&mut self,   node: &N) -> ControlFlow<Self::Output>;
}

pub trait Rewriter<N> {
    fn f_down(&mut self, node: N) -> Result<N, ValidateError>;
    fn f_up(&mut self,   node: N) -> Result<N, ValidateError>;
}
```

The trait surface stays small. All algorithmic work in the rest of the pipeline (compile, plan, adapt) composes these primitives.

### 3.2 The `ExprLeaf` trait

Each leaf set implements a small trait that exposes the per-leaf metadata the structural layer needs:

```rust
pub trait ExprLeaf: Sized + Clone + Debug {
    /// Canonical logical type carried (or inferred) by this leaf.
    /// Returns `None` only when type cannot be determined locally
    /// (e.g. untyped `Null`, untyped `Field` before resolution).
    fn inferred_type(&self) -> Option<&DataType>;
}
```

`ExprLeaf` is intentionally minimal — leaf-set–specific behaviour (e.g., semantic-ref resolution per `[19 §3](../19_expression_flow.md)`, `Parameter` binding at plan time) lives at the site that operates on the leaf, not as a trait method. This keeps the trait surface stable across leaf-set evolution.

## Testing Strategy

- **Trait contract round-trip:** `children` / `with_new_children` / `apply` / `transform` compose to identity on any `Expr<L>` tree and `ir::PlanNode`.
- **Visitor/Rewriter control flow:** `ControlFlow::Break` short-circuits; `ControlFlow::Continue` exhausts the tree.
- **ExprLeaf parity:** `inferred_type` agrees with the type system across all three leaf sets (`UnresolvedSemanticLeaf`, `UnresolvedPhysicalLeaf`, `ir::PhysicalLeaf`).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
