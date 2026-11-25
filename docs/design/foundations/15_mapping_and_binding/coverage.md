---
id: 15-coverage
status: ratified
version: 2.1
impl: implemented
---

# Coverage

Coverage at the Binding level (Native, NullFill, Derived, Metadata) declares how each Semantics is materialized across physical sources. This document defines the four CoverageVariant values, per-source × per-Semantics truth table, computation at compile, and the scope boundary with 16's composition-level Coverage. See [metadata-extraction](metadata-extraction.md) for MetadataDimension Coverage mechanics. <!-- cross-cut: none -->

## 6. `Coverage` at Binding Level

### 6.1 Structure

When a Binding's `sources: Vec<PhysicalSource>` has cardinality > 1, not every source necessarily provides every Semantics. `Coverage` is the per-source × per-Semantics truth table that tells the planner "what to do for source *i*, Semantics *s*":

```rust
pub struct Coverage {
    pub entries: HashMap<CoverageKey, CoverageVariant>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CoverageKey {
    pub source_index: usize,
    pub semantics: SemanticsName,
}

#[non_exhaustive]
pub enum CoverageVariant {
    Native,
    NullFill,
    Derived,
    Metadata,
}
```

- `source_index` indexes into `Binding.sources` (0-based).
- `semantics` is the `SemanticsName` in the Binding's `SemanticMapping`.
- **Default when absent:** `Native`. The `HashMap` stores only the non-default entries (`NullFill`, `Derived`, `Metadata`). A missing entry means "this source provides this Semantics via the direct-column mechanism of the `SemanticMappingValue`". This is the common case and the storage is sparse.

Variants:

- `**Native`** — the source has a physical column (for `Column` / `Expr` variants whose referenced columns are all present) or the Semantics is a `Literal` (uniformly applicable across sources).
- `**NullFill**` — the source does NOT have the physical column(s) that `SemanticMappingValue::Column` / `::Expr` requires. At plan time, the planner emits a `Project[Cast(Null, declared_type) AS semantics]` at the branch covering this source (Unionset pattern; `23`). For non-Unionset consumers, `NullFill` on any source is a `CompileError::UnusableNullFillInNonUnionContext { binding_id, source_index, semantics }` (COMP_E_0310) — the v1 behavior: NullFill is meaningful only when the binding is consumed by a Unionset or Grainset constituent that tolerates it; in a bare Dataset / Joinset consumer, a NullFill source is a Model bug.
- `**Derived**` — the source has the *upstream* columns that an `Expr` expression needs, but not necessarily the direct Semantics column. Used for `Expr` entries whose expression's `referenced_columns` are all present on this source. (For simple `Column`-valued Semantics, `Derived` is indistinguishable from `Native` and is not used — the planner short-circuits to `Native`.)
- `**Metadata*`* — the Semantics's value comes from the source's metadata, eagerly resolved at compile via the layer-3 mechanic (§8). The actual `LiteralValue` lives on `ResolvedPhysicalSource.metadata_values[semantics]` (§7.6); the planner reads from that map rather than a column scan or expression evaluation. Used exclusively for `SemanticMappingValue::Metadata`-bound Semantics. Distinct from `Native` because the read path is not a column scan, distinct from `Derived` because no expression evaluation is involved, and distinct from `NullFill` because a value is present (compile fails fast if the metadata cannot be resolved — see §8.1.2 / §11). Used by the planner to know "skip column scan, read constant from per-source metadata map" — see `34 §5` for pushdown reasoning.

### 6.2 Computation at compile time

`Coverage` is populated during §10 step 5 of the resolution flow. The algorithm per `(source_index, semantics)`:

1. Look up `SemanticMapping.entries[semantics]` → `SemanticMappingValue`.
2. Dispatch on variant:
  - `Column(name)` — check `Binding.sources[source_index].schema.columns` for `name`. Present → `Native`. Absent → `NullFill`.
  - `Literal(_)` — always `Native`. (Literals do not depend on source schemas.)
  - `Expr(expr)` (author-written compute) — check every name in `expr.referenced_columns()` against the source's schema. All present → `Derived` (since the Semantics is computed, not directly present). Any missing → `NullFill`.
  - `Metadata(recipe)` — dispatch on `recipe.extraction`:
    - `Path { token }` — applicable iff `Binding.sources[source_index]` is `PhysicalSource::File`. For `Table` / `Snapshot` sources, fail-fast at compile with `COMP_E_0312 MetadataTokenOnNonFileSource` (path-token has no defined value for non-file sources in v1; this is a Model bug, not a per-source NullFill). On `File`: run the layer-3 `path_token` mechanic (§8.1) — out-of-range fails with `COMP_E_0311`, empty segment fails with `COMP_E_0320`, cast-to-`recipe.data_type` failure fails with `COMP_E_0321`. On success: emit `Metadata` and write the resolved `LiteralValue` into `ResolvedPhysicalSource.metadata_values[semantics]` (§7.6).
3. Persist the resulting `CoverageVariant` in `Coverage.entries` iff it is not `Native` (to keep storage sparse).

### 6.3 Derived-cross-source rule

When a Semantics maps to `Computed`, its `Coverage` on a specific source depends on whether the expression's column references are present on that source. The `Derived` variant encodes: "the upstream columns are here, the planner computes the Semantics on this source branch." The `NullFill` variant encodes: "the upstream columns are not here, the planner emits a NULL-filled constant of the declared Semantics type on this source branch."

**Proposed (Round 1):** `Derived` is a distinct variant from `Native` (rather than collapsing into `Native`) because consumers that care about provenance — notably the `16` composition layer building a `ComposedSemanticInterface` coverage map — need to distinguish "this is physically present as a column" (`Native` on a `Column`-valued Semantics) from "this is computed from upstream columns that happen to be present" (`Derived`). The distinction matters for pushdown reasoning in `34 §5`: Native reads are always pushdownable, Derived reads require pushing the computation. See `questions/closed/15_questions.md` Q-MAP-005 (closed).

### 6.4 Scope boundary with `16`

`15` owns `Coverage` at the **Binding** level — one Coverage per Binding, keyed on `(source_index, semantics)`. `16` owns `Coverage` at the **ComposedSemanticInterface** level — one Coverage per composition, keyed on `(constituent_datakind, composed_field)`.

The two are orthogonal. A Unionset of three Simple branches has:

- One `Coverage` per branch Binding (from `15`).
- One composition-level `Coverage` on the Unionset itself (from `16`), which records which constituent provides which field of the unified surface.

`15` does not speak about composition-level coverage; `16` uses `15`'s Binding coverage as an input when it builds the composition coverage (for each composed field, look up which constituent Binding has `Native` / `NullFill` for the underlying Semantics; this feeds the planner's per-branch NULL-fill emission in `23`).

## Testing Strategy

- Coverage computation: §6.2 per-variant derivation (Native from Column, NullFill from Literal, Derived from Expr, Metadata from Metadata)
- Derived vs Native distinction: §6.3 Expr-backed even when only one source → Derived (not Native)
- UnusableNullFillInNonUnionContext: COMP_E_0310 when NullFill appears in non-union Dataset
- CoverageVariant roster: all four variants (Native, NullFill, Derived, Metadata) appear in corpus
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
