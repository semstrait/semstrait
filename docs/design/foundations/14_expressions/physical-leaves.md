---
id: 14-physical-leaves
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: model owns UnresolvedPhysicalLeaf (no Parameter); ir owns PhysicalLeaf (with Parameter)
---

# Physical Leaf Sets

This leaf defines the two physical leaf sets — `model::UnresolvedPhysicalLeaf` (author-facing, no `Parameter`) and `ir::PhysicalLeaf` (resolved-IR, admits `Parameter`). The split enforces that authors cannot write `Parameter` leaves; they are compile-emitted only and live exclusively in the resolved-IR form. See [parameter.md](parameter.md) for the `Parameter` placeholder shape and [crate-placement.md](crate-placement.md) for ownership.

<!-- cross-cut: none -->

### 3.4 Physical leaf sets — two, by `Parameter` admissibility

Because `Parameter` is compile-emitted only, the physical leaf is split into **two distinct types** (item Y / V3): an author-facing set with no `Parameter`, and the resolved-IR set that admits it. This is the leaf-level mirror of the `model::Expr` / `ir::PhysicalExpr` enum split.

**`model::UnresolvedPhysicalLeaf`** — the author-facing physical-mapping leaf (crate-of-record `[32](../../apis/32_semstrait_model.md)`):

```rust
#[non_exhaustive]
pub enum UnresolvedPhysicalLeaf {
    /// Physical column reference (binding-resolved at compile).
    Column(ColumnRef),

    /// Typed literal value.
    Literal(Literal),
    // No Parameter — an author can never write one.
}
```

**`ir::PhysicalLeaf`** — the resolved-IR leaf set, what the planner and adapters need (crate-of-record `[35 §3.4](../../apis/35_semstrait_ir.md)`):

```rust
#[non_exhaustive]
pub enum PhysicalLeaf {
    Column(ColumnRef),
    Literal(Literal),

    /// Compile-emitted, plan-bound parameter placeholder.
    /// Replaced with a concrete value during planning (§5).
    Parameter(Parameter),
}
```

`ir::PhysicalExpr` is the **canonical IR form**. Adapters render from it. The `SemanticManifest` stores `PhysicalExpr` per-`(Semantics, Binding)` pair per `[19 §3.2](../19_expression_flow.md)`.

Notable invariants on `PhysicalExpr`:

- No `Field` / `Dimension` / `Measure` / `Metric` / `Key` — semantic references are eliminated during compile.
- No sugar accessors — typed-leaf-with-accessor leaves are eliminated during compile (lowered to `Window`-rooted subtrees per §4.2).
- `Parameter` leaves are the only non-resolved state the canonical IR carries; they MUST be substituted before adapt time (§5.3 postcondition).

## Testing Strategy

- **Type safety:** `Parameter` is admitted only in `ir::PhysicalLeaf`, not in `model::UnresolvedPhysicalLeaf`. Parse construction of `UnresolvedPhysicalExpr` cannot produce a `Parameter` leaf.
- **Compile identity preservation:** `UnresolvedPhysicalExpr` → `ir::PhysicalExpr` identity is preserved modulo `Parameter` injection (compile-emitted only at sugar-accessor lowering sites, per §4.2).
- **Canonical IR postcondition:** Every `ir::PhysicalExpr` leaving `compile` carries no `Field` / `Dimension` / `Measure` / `Metric` / `Key` leaves, no sugar accessors.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
