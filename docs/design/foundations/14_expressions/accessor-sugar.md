---
id: 14-accessor-sugar
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: four accessor enums; Window lowering via compile sugar elimination; fixpoint iteration per 19 §3.1
---

# Per-Entity Accessor Sugar

This leaf defines the four per-kind accessor enums (`DimensionAccessor`, `MeasureAccessor`, `MetricAccessor`, `KeyAccessor`) and the sugar-elimination contract that lowers typed leaves with `accessor: Some(_)` to `Window`-rooted subtrees. The accessors are carried as `Option` fields directly on the typed semantic leaves. See [semantic-leaves.md](semantic-leaves.md) for the leaf-carrying structure and [crate-placement.md](crate-placement.md) for ownership.

## 4. Per-Entity Accessor Sugar

### 4.1 The per-kind accessor enums

Per-entity sugar lets authors write shorthand like `measure("revenue").previous()` or `metric("conv_rate").delta()`. The mechanism is a kind-specific accessor enum carried as an `Option<…>` field on the typed semantic leaves (§3.5):

```rust
#[non_exhaustive]
pub enum DimensionAccessor {
    First,
    Last,
    Lag(u32),
    Lead(u32),
}

#[non_exhaustive]
pub enum MeasureAccessor {
    Previous,
    Next,
    Lag(u32),
    Lead(u32),
    Delta,
    PercentChange,
}

#[non_exhaustive]
pub enum MetricAccessor {
    Previous,
    Next,
    Lag(u32),
    Lead(u32),
    Delta,
    PercentChange,
}

#[non_exhaustive]
pub enum KeyAccessor {
    First,
    Last,
    Lag(u32),
    Lead(u32),
}
```

Two structural pairings emerge:

- **`MetricAccessor` mirrors `MeasureAccessor` 1:1**. A Metric is a per-group already-aggregated value at access time, structurally identical to a Measure at the output projection stage.
- **`KeyAccessor` mirrors `DimensionAccessor` 1:1**. A Key is a Dimension-shaped entity for sugar purposes; the windowed accessor surface is symmetric.

There is **no outer `Accessor` wrapping enum**. Each per-kind accessor enum is carried directly on the matching typed leaf (`UnresolvedSemanticLeaf::Dimension { accessor: Option<DimensionAccessor> }`, etc.). The type system enforces kind agreement at construction: a `Dimension` leaf simply has no way to carry a `MeasureAccessor`.

The `Field` leaf carries no accessor — it is the untyped semantic reference whose kind is resolved at compile. To apply sugar, authors use the typed accessor for the matching kind (`measure("x").delta()`, not `field("x").delta()`).

### 4.2 Sugar elimination shape

A typed leaf with `accessor: Some(_)` lowers at compile to a canonical `Window`-rooted subtree:

```text
UnresolvedSemanticLeaf::Measure { name: "revenue", accessor: Some(MeasureAccessor::Previous) }
  ─→ Expr::Window {
       function: <derived from accessor>,
       args:         [<resolved Measure "revenue" expr>],
       partition_by: [Parameter(RequestDimensionsMinusTemporal)],
       order_by:     [Parameter(RequestTemporalAxis)],
       frame:        Some(<derived from accessor>),
     }
```

A typed leaf with `accessor: None` lowers to whatever the registered semantic at `name` resolves to (its own `expr` tree under the binding's `SemanticMapping`), unwrapped — no `Window` is emitted.

Compile substitutes the entity reference (the `name` field) inside `args` via the binding's `SemanticMapping` per `[15](../15_mapping_and_binding.md)`. The `partition_by` / `order_by` slots emit `Parameter` leaves whose `ParameterKey`s are bound at plan time (§5).

**Sugar-on-sugar handling.** Some accessors lower to compositions that still contain typed leaves with `accessor: Some(_)` — for example, `Delta` lowers to `operand - operand.Previous`, where `operand.Previous` is still a typed leaf with `Some(Previous)` accessor. The compile pipeline runs sugar elimination **to fixpoint** so that every typed leaf with a non-`None` accessor is eliminated before downstream substeps begin. Detailed substep ordering lives in `[19 §3.1](../19_expression_flow.md)`.

## Testing Strategy

- **Structural mirroring:** `MetricAccessor` variants exactly match `MeasureAccessor` variants (1:1). `KeyAccessor` variants exactly match `DimensionAccessor` variants (1:1).
- **Sugar elimination to Window:** Every typed leaf with `accessor: Some(_)` successfully lowers to a `Window`-rooted subtree. The lowered `args` contain the resolved semantic expression; `partition_by` / `order_by` contain `Parameter` leaves.
- **Fixpoint iteration:** Accessors that lower to sugar-on-sugar compositions (`Delta` → `operand - operand.Previous`) are eliminated to fixpoint — no `accessor: Some(_)` survives into the final `ir::PhysicalExpr`.
- **Compile postcondition:** No `accessor: Some(_)` survives into `PhysicalExpr`. Every accessor-bearing leaf is eliminated during compile.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
