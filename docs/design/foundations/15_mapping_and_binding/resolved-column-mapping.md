---
id: 15-resolved-column-mapping
status: ratified
version: 2.1
impl: implemented
---

# Resolved Column Mapping

The SemanticManifest-layer ResolvedColumnMapping structure (four flat HashMaps per value category) provides planner-speed O(1) access to physical column names and metadata values. This document defines ResolvedBinding/ResolvedPhysicalSource envelope structures, the planner-speed contract, and the relation to 19's ResolvedExprTable. See [semantic-mapping](semantic-mapping.md) for the model-layer SemanticMapping and [coverage](coverage.md) for Coverage.

## 7. SemanticManifest-Layer Counterpart: `ResolvedColumnMapping`

### 7.1 Motivation

The Model-layer `SemanticMapping` is a single `BTreeMap<SemanticsName, SemanticMappingValue>`. At plan time, the planner's hot-loop lookup pattern is "given a Semantics name, jump straight to the physical recipe." Matching the sum-type at every lookup is wasteful when the per-variant shape is known; a flat per-variant HashMap is faster and avoids the planner carrying an enum match in its hottest inner loop.

Per I8 and the `Resolved*` prefix convention (`00 §4.1`), the SemanticManifest stores `ResolvedColumnMapping`, a denormalized / pre-indexed form:

### 7.2 Structure

```rust
pub struct ResolvedColumnMapping {
    pub columns: HashMap<SemanticsName, ColumnName>,
    pub literals: HashMap<SemanticsName, ResolvedLiteral>,
    pub computed: HashMap<SemanticsName, PhysicalExpr>,
    pub metadata: HashMap<SemanticsName, MetadataDimensionRecipe>,
    pub source_coverage: HashMap<CoverageKey, CoverageVariant>,
}

pub struct ResolvedLiteral {
    pub value: LiteralValue,
    pub data_type: DataType,
}
```

The four top-level HashMaps are **disjoint**: a given `SemanticsName` appears in exactly one of them. The completeness rule (§5.6) is preserved: the union of their key sets equals the `SemanticInterface`'s name set.

`source_coverage` is the SemanticManifest-layer form of the §6 `Coverage` — same key/value shape, just promoted from an `Option<Coverage>` to a bare field (the `None` case is represented as an empty map; the per-source default is still `Native`).

`metadata: HashMap<SemanticsName, MetadataDimensionRecipe>` is the **per-Binding recipe table** for metadata-bound Semantics (`MetadataDimensionRecipe` shape per `18 §10.4`). The recipe is **global to the Binding** (extraction kind + declared `data_type:`); per-source resolved `LiteralValue`s live separately on each `ResolvedPhysicalSource.metadata_values` (§7.6) — the planner reads the recipe to know the kind, and the per-source map for the actual value of the source it is processing.

### 7.3 Construction

The compile stage produces `ResolvedColumnMapping` from the Model-layer `SemanticMapping` (the v1 4-variant roster per `18 §10`) via a single pass — each variant routes to its dedicated flat map:

```
for (semantics, value) in model.semantic_mapping.entries:
    match value:
        Column(name)        → resolved.columns.insert(semantics, name)
        Literal(lit)        → resolved.literals.insert(semantics, ResolvedLiteral::from(lit))
        Expr(expr)          → resolved.computed.insert(semantics, expr)
        Metadata(recipe)    → resolved.metadata.insert(semantics, recipe)
```

`expr` is the already-compiled `PhysicalExpr` (with `inferred_type` populated per §9.2) — no further expression work happens here. `recipe` is the already-synthesized `MetadataDimensionRecipe` from §5.5 / §10.4.0 (compile pass, before the completeness check). The per-source `LiteralValue`s for metadata Semantics are populated into each `ResolvedPhysicalSource.metadata_values` during the Coverage-derivation pass (§10.5), not here — `ResolvedColumnMapping.metadata` only carries the recipe.

**Cast-wrapped Column entries (`§9.1`)** flow into `resolved.computed` (not `resolved.columns`) because the cast wraps the `Column` reference into a `PhysicalExpr` — the model-layer `SemanticMapping` carries the rewritten `Expr` entry by the time §7.3's pass runs (§10.4 step 4 performs the rewrite before §10.6's materialization).

### 7.4 Planner access pattern

Plan-time per-Semantics lookup is per-(Binding, source) — the planner already knows which source it is processing (Unionset branch, Grainset child, etc.):

```rust
fn resolve_semantics<'a>(
    rb: &'a ResolvedBinding,
    src_idx: usize,
    s: &SemanticsName,
) -> SemanticsRecipe<'a> {
    if let Some(col)    = rb.column_mapping.columns.get(s)  { return SemanticsRecipe::Column(col); }
    if let Some(lit)    = rb.column_mapping.literals.get(s) { return SemanticsRecipe::Literal(lit); }
    if let Some(exp)    = rb.column_mapping.computed.get(s) { return SemanticsRecipe::Computed(exp); }
    if let Some(recipe) = rb.column_mapping.metadata.get(s) {
        // Per-source resolved value lives on the source itself.
        let value = &rb.sources[src_idx].metadata_values[s];
        return SemanticsRecipe::Metadata { recipe, value };
    }
    // Invariant: one of the above must hit, by compile-time completeness (§5.6).
    unreachable_by_invariant!("completeness guaranteed at compile")
}
```

Each branch is an O(1) HashMap probe. The metadata branch performs a second O(1) probe into the source's `metadata_values` map. The sum-type match on the Model-layer enum is never paid at plan time.

### 7.5 Relation to `19`'s `ResolvedExprTable`

`19 §3.2`'s `ResolvedExprTable` is a **SemanticManifest-global** map from `(SemanticsName, EntityId)` to `PhysicalExpr`. `ResolvedColumnMapping.computed` is a **per-Binding denormalization** of that table, filtered to the Binding's own entries. Both exist:

- The global table supports cross-Binding planner work (e.g. `19 §3.4.5` Relationship-path composition via `PathSignature`, where an expression is shared across a Joinset's members).
- The per-Binding HashMap supports single-Binding planner work (per-`Scan` expression lookup) without the extra `EntityId` in the key.

Whether the two share storage (the `ResolvedColumnMapping.computed` values are pointers into the global table) or are duplicated is a `33`-owned implementation choice. From `15`'s contract surface, both are populated and both are plan-readable; `33` will ratify the storage strategy. **Proposed (Round 1):** duplicate storage by default; the memory overhead is a small constant per binding-semantics pair, and the planner is free of aliasing concerns. See `questions/open/15_questions.md` Q-MAP-006.

### 7.6 `ResolvedBinding` envelope

The Binding's SemanticManifest-layer counterpart is:

```rust
pub struct ResolvedBinding {
    pub binding_id: EntityId,
    pub sources: Vec<ResolvedPhysicalSource>,
    pub column_mapping: ResolvedColumnMapping,
}

#[non_exhaustive]
pub struct ResolvedPhysicalSource {
    /// The resolved physical target. Same variants as `PhysicalSource` (§3.1).
    pub source: PhysicalSource,

    /// Per-source resolved metadata literals — one entry per metadata-bound
    /// Semantic in the Binding's interface. Populated during Coverage
    /// derivation (`§10.5`) by running the layer-3 mechanic (`§8`) plus
    /// `Cast` to the recipe's declared `data_type` (§5.5). Empty when the
    /// Binding has no metadata-typed Semantics. Keys are exactly the keys
    /// of `ResolvedColumnMapping.metadata`.
    pub metadata_values: HashMap<SemanticsName, LiteralValue>,
}
```

`ResolvedPhysicalSource` wraps the SemanticManifest-layer `PhysicalSource` (§3.1) and adds the per-source `metadata_values` map. The map is sized by the number of metadata-bound Semantics in the Binding's interface — `0` for Bindings that declare no metadata Dimensions; `N` for Bindings with `N` such Dimensions. **Values may differ across sources** (e.g. for a glob spanning `year=2024/...` and `year=2025/...` paths, the metadata-bound `source_year` Semantic resolves to different `LiteralValue`s on each source). The recipe in `ResolvedColumnMapping.metadata[name]` is constant across the Binding; only the resolved value varies.

Future manifest-layer optimizations (e.g. pre-computed partition pruning indices) evolve as MINOR per `30 §4`; the `#[non_exhaustive]` on `ResolvedPhysicalSource` allows additive growth.

## Testing Strategy

- ResolvedColumnMapping construction: §7.3 4-way split (dimensions, keys, timestamp, measures) into per-category HashMaps
- Planner O(1) access: §7.4 per-variant HashMap probe for column name lookup
- Relation to ResolvedExprTable: §7.5 per-Binding flat maps vs global ResolvedExprTable for Expr-backed semantics
- ResolvedPhysicalSource metadata_values map: §7.6 per-source MetadataDimension → LiteralValue map population
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
