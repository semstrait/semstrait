---
id: 15-semantic-mapping
status: ratified
version: 2.1
impl: implemented
---

# Semantic Mapping

The SemanticMapping structure (BTreeMap keyed on SemanticsName) binds every semantic field in the Dataset to a physical column or derived value. This document defines the four SemanticMappingValue variants (Column, Literal, Expr, Metadata), per-variant compile semantics, and the completeness rule: every Semantics must be covered exactly once. See [expression-strata](expression-strata.md) for the three-stratum expression model and [metadata-extraction](metadata-extraction.md) for MetadataDimension layer-3 mechanics.

## 5. `SemanticMapping`

### 5.1 Structure

The `SemanticMappingValue` enum roster, its `LiteralValue` payload, the `MetadataDimensionRecipe` / `MetadataExtraction` shapes, and the `auto` default are ratified in `[18 §10](../18_entities.md#10-semanticmapping-value-shape)` — v1 roster `{Column(String), Literal(LiteralValue), Expr(PhysicalExpr), Metadata(MetadataDimensionRecipe)}` (4 variants). `15` owns the container `SemanticMapping` (the per-Binding Semantics-name-keyed recipe table) and the compile-time semantics of each variant:

```rust
pub struct SemanticMapping {
    pub entries: BTreeMap<SemanticsName, SemanticMappingValue>,
}
// SemanticMappingValue: owned by `18 §10`.
```

`BTreeMap` keying is deliberate: it gives the Model-layer shape a deterministic iteration order (alphabetical on `SemanticsName`) which feeds straight into the `ResolvedExprTable` ordering ratified by `19 §3.4`. The SemanticManifest-layer `ResolvedColumnMapping` (§7) splits entries into per-category flat maps for O(1) lookup — the ordering fence is paid at SemanticManifest-construction time, not at plan time.

### 5.2 `SemanticMappingValue::Column` — compile semantics

The most common case: the Semantics value is a direct physical column reference. The variant carries a bare `String` (the physical column name) per `18 §10`:

- The string MUST resolve to a column in **every** `PhysicalSource` in the Binding whose `Coverage` for this Semantics is `Native` (§6). Sources with `NullFill` do not require the column.
- No per-source divergence in spelling: the column name is a single value; cross-source name-mapping is not supported at the Binding layer (it is a `Coverage` / NullFill question).
- The compile stage wraps a `Column(name)` whose physical `DataType` does not exactly match the Semantics's declared `data_type` with a `Cast` — the wrapped form is re-homed into `ResolvedColumnMapping.computed` per §9.1 rather than `ResolvedColumnMapping.columns`.

### 5.3 `SemanticMappingValue::Literal` — compile semantics

A typed constant per `18 §10.2`. The `LiteralValue` enum carries the type tag implicitly via its discriminant (`Null`, `Bool`, `Int`, `Float`, `Decimal`, `String`, `Date`, `Timestamp`).

**Literal validation at compile.** The `LiteralValue`'s discriminant is validated against the Semantics's declared `DataType` for representability:

- `LiteralValue::Int(i)` against a Semantics declared as `Byte` → validate `i ∈ [-128, 127]`; else `CompileError::LiteralOverflow { name, value, data_type }` (COMP_E_0305).
- `LiteralValue::Null` against a non-nullable declared Semantics → `CompileError::NullLiteralForNonNullableSemantics { name }` (COMP_E_0306).
- `LiteralValue::String(s)` against a Semantics declared as `Date`/`Timestamp` → requires RFC 3339 parse success; else `CompileError::LiteralParseFailed { name, value, target_type }` (COMP_E_0307).

### 5.4 `SemanticMappingValue::Expr` — compile semantics

An author-written expression, compiled to `PhysicalExpr` by `19 §3.3`:

- The wrapped `PhysicalExpr` honors the wrapper invariants from `14 §3.3`: no `EntityRef`, no `Aggregate`, `Column` allowed. Type inference (`14 §5`) has already run; `expr.inferred_type()` is populated.
- `expr.referenced_columns()` — every column name it reads must exist in every `PhysicalSource` in the Binding whose `Coverage` for this Semantics is `Derived` (§6). Sources with `NullFill` do not require the columns.

The YAML-to-`PhysicalExpr` compilation pathway for `Expr` entries:

```
YAML authoring expression   (14 §6; bare Expr<L> field, friendly field-level projection)
   → model::UnresolvedSemanticExpr  (authors may write @other_semantics; 19 §3.3 substitutes them)
   → ir::PhysicalExpr        (columns resolved against this Binding's PhysicalSource schemas)
```

The compile stage invokes `model::UnresolvedSemanticExpr::resolve` per `[19 §3.1](../19_expression_flow.md)` per `Expr` entry. The binding context supplies (a) the cross-source reconciled schema over which `Column` identifiers resolve, (b) the `FunctionRegistry` (via `14a`), (c) the substitution map for typed-leaf identifiers into same-Binding `Expr` / `Column` entries. The output is a `PhysicalExpr` with `inferred_type` set; §9.1 then compares `inferred_type` against the declared Semantics `DataType` and emits a `Cast` at the Semantics boundary if needed.

**Cross-reference to `19 §3.5` cycle detection.** `Expr`-entries within a single `SemanticMapping` can refer to other Semantics via typed semantic leaves. The `19 §3.5` Tarjan-SCC pass runs over the Binding's `Expr` entries and detects same-Binding cycles (`e1 → e2 → e1`); the failure is `CompileError::CyclicReference` per `[19 §8.1](../19_expression_flow.md)`.

### 5.5 `SemanticMappingValue::Metadata` — compile semantics

The 4th variant. v1 scope is **path-token extraction only**; partition-level extraction described in `13 §4.7` is deferred to v2 (see §8.0 v1-scope banner and §1's resolved-`[TD-MAP-METADATA-FOLD]` note).

**Authoring surface — Dimension type, not `semantic_mapping:`.** The author writes the recipe on the Dimension itself, in its `type:` block (per `13 §4.7` / `18 §4.2`):

```yaml
dimensions:
  - name: source_year
    data_type: string
    type:
      metadata:
        source:
          path:
            token: 1
```

There is **no author-side `semantic_mapping:` entry** for a metadata Dimension. The binding-resolution pass (`§10.4` step 4.0) detects each Dimension whose `type` is `Metadata(...)`, reads the `MetadataDimensionBody.source` shape (`18 §4`), and synthesizes the corresponding `SemanticMapping.entries[name] = SemanticMappingValue::Metadata(MetadataDimensionRecipe { extraction, data_type })` entry **before the completeness check (§5.6) runs**. A `semantic_mapping:` block that *does* contain an explicit entry for a metadata Dimension is rejected as a `SpuriousBindingEntry`-class error (or, equivalently, an authoring-layer parse error if `32` chooses to catch it earlier).

**Compile-time mechanic, not an expression.** Metadata extraction is layer 3 of the three-stratum model (banner, §1). It is **not** a `PhysicalExpr` variant, **not** a function in the `14a` registry, and **not** subject to `19 §3`'s substitution / cycle-detection passes. The recipe (`MetadataDimensionRecipe`) is a compile-output struct that pairs the extraction kind with the declared `data_type:`; its evaluation runs inside `compile` to produce the per-source `LiteralValue`s that feed `ResolvedPhysicalSource.metadata_values` (§7.6).

**Per-source resolution flow** (one pass per `(source_index, Metadata-bound semantics)` pair, executed during the Coverage-derivation pass §10.5):

1. Read the `MetadataDimensionRecipe.extraction` kind. v1: `Path { token }`.
2. Run the layer-3 mechanic for the source variant:
  - `PhysicalSource::File { path, .. }` → `path_token(path, token)` (§8.1) → returns a `String` (the raw segment) on success, or an error on out-of-range / empty (§8.1.2).
  - `PhysicalSource::Table { .. }` / `PhysicalSource::Snapshot { .. }` → not applicable for path extraction; coverage records `Metadata` (per §6) but no value is produced and the source is rejected at compile per the v1 fail-fast policy (`COMP_E_0312 MetadataTokenOnNonFileSource`).
3. Cast the extracted `String` to the declared `data_type:` per `MetadataDimensionRecipe.data_type`. Cast failure is a compile error (`COMP_E_0321 MetadataCastFailed`).
4. Wrap the cast result in the appropriate `LiteralValue` discriminant; insert into the source's `metadata_values: HashMap<SemanticsName, LiteralValue>` (§7.6).

**Cross-source value divergence is allowed.** The recipe is global to the Binding, but the **values may differ across sources** — that is the whole point of metadata extraction. The SemanticManifest stores one resolved `LiteralValue` per source (per metadata Dim); the planner reads the value for the source it is currently processing.

**Constant folding feed.** Layer-1 (`UnresolvedSemanticExpr`) consumers that reference a metadata-bound Semantic are eligible for partial evaluation at the `UnresolvedSemanticExpr → PhysicalExpr` lowering boundary: the metadata `LiteralValue` is known per source at compile time, so a `CASE WHEN @dataset_name = 'asd' THEN @col1 ELSE @col2 END` can be flattened to a per-source `Column(col1)` / `Column(col2)` projection. The partial-evaluation language is owned by `[19 §4.1](../19_expression_flow.md)`; `15`'s contract is to make the per-source resolved literals available on `ResolvedPhysicalSource.metadata_values`.

Detailed runtime mechanics for `path_token`, exhaustively described in §8.1.

### 5.6 Completeness: coverage of the `SemanticInterface`

Per `11 §6`, a `SimpleDataKind`'s `SemanticInterface` is the complete named surface (Dimensions, Measures, Metrics, Filters, Keys). `15` ratifies the rule:

**Every Semantics name in the `SemanticInterface` MUST appear exactly once as a key in `SemanticMapping.entries`.**

- A name in the interface but missing from `entries` → `CompileError::MissingBindingEntry { semantics, binding_id }` (COMP_E_0308). Fail-fast.
- A name in `entries` but not in the interface → `CompileError::SpuriousBindingEntry { name, binding_id }` (COMP_E_0309). Fail-fast.
- A name duplicated in `entries` — the YAML parser rejects duplicate keys at a lower layer (`32`) via standard YAML duplicate-key handling; at the `15` level, `BTreeMap` is an unambiguous map and duplication is structurally impossible.

**Edge case: Semantics with a `Constraint` that derives its value (e.g. `Measure(Count, Key)` per `11 §8.4`).** Per `11 §8.4`, a count-like Measure declared with `Constraint::DerivesFrom(Key)` does not require a physical column; it counts the Key's rows. `SemanticMapping` still includes an entry for that Semantics — proposal: `SemanticMappingValue::Expr { expr: PhysicalExpr(Count(Column(<key_column>))) }` where the key column is the `SemanticMapping`'s entry for the referenced Key. The `compile` stage synthesizes the `Computed` entry from the `Constraint::DerivesFrom` spec — authors do not need to re-declare it. **Proposed (Round 1):** this is a compile-stage synthesis, not a YAML-surface convenience; the Model's authored `SemanticMapping` can omit the key-derived Measure entry, and the compile stage fills it in before the completeness check runs. See `questions/open/15_questions.md` Q-MAP-003.

**Edge case: `ComputedDimension` (per `14 §1.2`).** These ALWAYS map to `SemanticMappingValue::Expr`; they never have a `Column`-valued entry. The YAML parse enforces this at `32`.

**Edge case: Name case.** `SemanticsName` preserves the author's case (`14 §6.5`); the parser does no case folding. Mapping-key mismatches due to case errors (`"customer_id"` declared, `"CustomerID"` mapped) → `SpuriousBindingEntry` + `MissingBindingEntry` pair.

### 5.7 Shape constraints

- `SemanticMappingValue::Column` is the **common path**; it should account for the majority of entries in any real Model. Complex variants exist for the edge cases.
- `SemanticMappingValue::Expr` is for author-declared computed Semantics (`14 §1.2`) AND for the synthesized Measures from §5.6 and for the `Cast`-wrapped Column cases from §9.1. The cardinality of Computed entries should be bounded by the sum of authored-computed-Semantics + cast-wrapped Semantics in the `SemanticInterface`.
- `SemanticMappingValue::Metadata` is **always compile-synthesized** (§5.5) from a Dimension whose `type:` block carries a `metadata:` recipe (`13 §4.7` / `18 §4`); it has no author-facing `semantic_mapping:` YAML and is never authored directly. Cardinality is bounded by the number of metadata-typed Dimensions in the `SemanticInterface`. v1 scope: path-token only.
- `SemanticMappingValue::Literal` is a rarely-used fallback — it captures NullFill-style "this source does not have this column, use a constant" patterns, but the more common NullFill pattern uses `Coverage::NullFill` (§6) which does not need a Literal entry (the planner emits a `NULL` cast itself).

## Testing Strategy

- Completeness: §5.6 MissingBindingEntry (COMP_E_0308) when Semantics uncovered, SpuriousBindingEntry (COMP_E_0309) when extra entry present
- Per-variant compile semantics: Column (§5.2), Literal (§5.3), Expr (§5.4), Metadata (§5.5) all validated
- Literal validation: COMP_E_0305 (InvalidLiteralForDataType), COMP_E_0306 (UnrepresentableLiteralType), COMP_E_0307 (MalformedLiteralValue)
- Synthesized entries: derived Measures and metadata dimensions correctly added to SemanticMapping
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
