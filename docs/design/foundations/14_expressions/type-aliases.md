---
id: 14-type-aliases
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: model owns UnresolvedSemanticExpr/UnresolvedPhysicalExpr; ir::PhysicalExpr = Expr<ir::PhysicalLeaf> alias
---

# Type Aliases

This leaf defines the authoring and resolved type aliases — `model::UnresolvedSemanticExpr`, `model::UnresolvedPhysicalExpr`, and `ir::PhysicalExpr` — each an instantiation of the shared `Expr<L>` at a different leaf set. The aliases are not re-declared enums; forbidden combinations are type-enforced by leaf-set structure. See [layered-model.md](layered-model.md) for the layered model and [crate-placement.md](crate-placement.md) for ownership.

### 3.6 Type aliases

```rust
// model (authoring, unresolved) — semstrait-model
pub type UnresolvedSemanticExpr  = Expr<UnresolvedSemanticLeaf>;
pub type UnresolvedPhysicalExpr  = Expr<UnresolvedPhysicalLeaf>;

// ir (resolved) — semstrait-ir
//   PhysicalExpr is the same shared Expr<L> at ir::PhysicalLeaf (admits Parameter);
//   the Window variant is shared structure (item Y / A2).
pub type PhysicalExpr = Expr<ir::PhysicalLeaf>;   // alias, not a re-declared enum
```

`UnresolvedSemanticExpr` is the authoring form of an expression at any semantic site; `UnresolvedPhysicalExpr` the authoring form at a physical-mapping site. Both live in `semstrait-model` (item Y / V6). The resolved `ir::PhysicalExpr` is produced by the manifest fold (§9.4) and is the form every post-compile stage operates on. The generic `Expr<L>` appears in trait bounds and shared algorithmic code over the authoring families.

> **Naming note (item Y).** Earlier revisions used a bare `SemanticExpr` alias homed in `semstrait-ir`. Item Y renames the type `model::UnresolvedSemanticExpr` and relocates it to model; `semstrait-ir` no longer defines a semantic-expression alias. Downstream prose (other docs) that named `ir::SemanticExpr` as the lowering input now reads `model::UnresolvedSemanticExpr`.
>
> **Shorthand license (this chapter).** For readability, the conceptual prose below uses the short names **`SemanticExpr`** and **`PhysicalExpr`** for the two authoring/resolved forms — `SemanticExpr` ≡ `model::UnresolvedSemanticExpr`, `PhysicalExpr` ≡ the resolved `ir::PhysicalExpr`. The fully-qualified names are authoritative; the short forms are this chapter's local shorthand only.

### 3.7 Forbidden combinations are type-enforced

Because the leaf sets differ structurally:

- `PhysicalExpr` **cannot contain** `Field` / `Dimension` / `Measure` / `Metric` / `Key` — those variants do not exist in `PhysicalLeaf`. All semantic references are eliminated during compile.
- `SemanticExpr` **cannot contain** `Parameter` — `Parameter` is a `PhysicalLeaf`-only variant.
- A `Dimension`-tagged leaf cannot carry a `MeasureAccessor` (or any non-Dimension accessor) — the variant signature `Dimension { name, accessor: Option<DimensionAccessor> }` enforces kind agreement at construction. The same holds for `Measure` / `Metric` / `Key`.

These invariants are upheld at the type level, not by runtime assertion. There is no `try_into_physical` runtime check, no defensive `panic!` for "Field found in PhysicalExpr".

**`UnresolvedSemanticLeaf::Column` is type-admissible but context-validated.** The leaf can be constructed (via `col(name)` or via bare identifier at a physical-mapping site that re-uses the same leaf set), but compile rejects it under manual mapping. The full rule is in §8.

The remaining structural invariants (e.g., `Aggregate` admitted only in aggregate-admitting sites, `Window` author-rejected) are construction-boundary checks; see §7.

### 3.8 Expression layer (applicability)

The two leaf-set layers (`SemanticExpr` / `PhysicalExpr`, §3.6) describe *what vocabulary* an expression uses. A second, orthogonal axis describes *where in the relational pipeline an expression's result applies* — its **layer**. After lowering, a `PhysicalExpr` is "just a tree", but its result lands at a specific stage:

- **Scalar** — row-level; evaluated before grouping; a pushdown candidate (grouping-key Dimensions, Keys).
- **Aggregate** — produced by the grouped aggregation (Measures); pre-/re-aggregation governed by `Additivity` (`14a §3.6`).
- **PostAggregate** — computed over already-aggregated inputs (ratio Metrics; any Dimension whose `expr` references a Measure); evaluated after the final aggregation and after any union/join re-aggregation.

The layer is a **pure function of the expression's structure** (presence and position of `Aggregate` nodes; an aggregation-relative `Window` — one whose argument contains an `Aggregate` — falls under `PostAggregate`, a raw-row `Window` under `Scalar`), computed at compile via `classify(tree)` (`19 §3.2.6`). It is deliberately **decoupled from the authoring `SemanticRole`** — a Dimension computed over a Measure is `PostAggregate`, not `Scalar` — so planning logic depends on evaluation placement, not on the semantic taxonomy. The type is ratified in `[apis/35_semstrait_ir.md §5.5](../../apis/35_semstrait_ir.md)` (`ExprLayer`); compile assigns it (`19 §3`); the manifest persists it per expression (`33 §7.2`); the planner reads it for placement (`19 §6`).

## Testing Strategy

- **Type safety:** `ir::PhysicalExpr` is a type alias, not a re-declared enum. `semstrait-ir` imports `Expr<L>` from `semstrait-model` and instantiates it at `ir::PhysicalLeaf`.
- **Forbidden combination enforcement:** Compile rejects `UnresolvedSemanticExpr` into physical-mapping sites. No `Parameter` is constructible in authoring expressions. No typed leaf carries a mismatched accessor (e.g., `Dimension` + `MeasureAccessor`).
- **Expression layer classification:** `classify(tree)` correctly assigns `Scalar` / `Aggregate` / `PostAggregate` based on `Aggregate` node presence and position; aggregation-relative `Window` is `PostAggregate`, raw-row `Window` is `Scalar`.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
