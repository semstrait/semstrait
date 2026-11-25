---
id: 14-semantic-leaves
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: semstrait-model owns UnresolvedSemanticLeaf; per-kind typed; accessor Option<XxxAccessor> carriers
---

# Semantic Leaf Set

This leaf defines `UnresolvedSemanticLeaf` — the per-kind typed leaf set (`Literal`, `Column`, `Field`, `Dimension`, `Measure`, `Metric`, `Key`) enabled by semantic declarations. Each typed leaf optionally carries a kind-specific sugar accessor as an `Option<XxxAccessor>` field. The `Column` leaf is type-admissible but context-validated (legal only under `semantic_mapping: auto`). See [accessor-sugar.md](accessor-sugar.md) for accessor enums and [site-shape-gates.md](site-shape-gates.md) for `Column`-under-auto-mapping rules.

<!-- cross-cut: none -->

### 3.5 `UnresolvedSemanticLeaf` — per-kind typed leaf set

The semantic leaf set carries **per-kind typed leaves**, eliminating the bare-identifier ambiguity that previously surfaced under `semantic_mapping: auto` name collisions. Each typed leaf optionally carries a kind-specific sugar accessor (§4).

```rust
#[non_exhaustive]
pub enum UnresolvedSemanticLeaf {
    /// Typed literal value.
    Literal(Literal),

    /// Physical column reference. Type-admissible inside SemanticExpr, but
    /// LEGAL only when the owning binding uses `semantic_mapping: auto`.
    /// Compile rejects this leaf under manual mapping (see §8).
    /// Authored as `col(name)` or via bare identifier in physical-mapping sites.
    Column(ColumnRef),

    /// Untyped semantic reference. Kind resolved at compile by name lookup
    /// against the semantic registry. Authored as `field(name)` or via bare
    /// identifier in semantic sites.
    Field(SemanticsName),

    /// Typed Dimension reference, optionally with sugar accessor.
    /// Authored as `dim(name)` or `dim(name).first()` / `.lag(2)` / etc.
    Dimension { name: SemanticsName, accessor: Option<DimensionAccessor> },

    /// Typed Measure reference, optionally with sugar accessor.
    /// Authored as `measure(name)` or `measure(name).previous()` / `.delta()` / etc.
    Measure { name: SemanticsName, accessor: Option<MeasureAccessor> },

    /// Typed Metric reference, optionally with sugar accessor.
    Metric { name: SemanticsName, accessor: Option<MetricAccessor> },

    /// Typed Key reference, optionally with sugar accessor.
    Key { name: SemanticsName, accessor: Option<KeyAccessor> },
}

impl ExprLeaf for UnresolvedSemanticLeaf { /* per-variant inferred_type — None for unresolved Field */ }
```

`UnresolvedSemanticExpr = Expr<UnresolvedSemanticLeaf>` (in `semstrait-model`) is the **authoring form** of an expression inside any semantic site. (This chapter's prose calls it `SemanticExpr` for short per §3.6.)

Notable properties:

- **No `EntityRef` wrapper, no `Access` wrapper, no `Accessor` outer enum.** Every semantic reference is a typed leaf whose variant tag already encodes the entity kind. The per-kind accessor enums (§4) sit as `Option<…>` fields on the typed leaves.
- **`Field` is the untyped fallback.** When the author writes a bare identifier or explicit `field(name)`, the leaf carries no kind hint; compile resolves the kind by registry lookup.
- **`Dimension` / `Measure` / `Metric` / `Key` are kind-pinned.** Compile fails fast if the registered semantic at `name` has a different kind than the authored leaf variant.
- **`Column` is conditionally legal.** Type-admissible (so the parser can construct it), but compile rejects it under manual mapping (§8). Under `semantic_mapping: auto`, compile synthesizes `SemanticMapping` entries for `Column` leaves and the rest of resolution proceeds as with manual mapping.
- **Sugar accessor agreement is type-enforced.** A `Dimension` leaf can only carry a `DimensionAccessor`, not a `MeasureAccessor` — the variant signature prevents mismatched pairings at construction time.
- **No `Parameter`.** Parameters are exclusively compile-emitted and live only in `PhysicalLeaf`.

## Testing Strategy

- **Type safety:** Per-kind accessor type agreement — `Dimension` carries `DimensionAccessor`, `Measure` carries `MeasureAccessor`, `Metric` carries `MetricAccessor`, `Key` carries `KeyAccessor`. No mismatched pairings are constructible.
- **Column admissibility:** `Column` leaf is legal only under `semantic_mapping: auto`. Compile rejects `Column` under manual mapping with `ColumnInManualMapping` error.
- **Compile transience:** Every `Field` / `Dimension` / `Measure` / `Metric` / `Key` leaf is eliminated during compile. The resolved `ir::PhysicalExpr` contains no semantic references.
- **ExprLeaf::inferred_type:** `Field` returns `None`; typed leaves return `Some(&DataType)` consistent with the registered semantic's type.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
