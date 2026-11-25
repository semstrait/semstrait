---
id: 18-semantic-mapping-value
status: ratified
version: 2.1
impl: implemented
---

# Semantic Mapping Value

This leaf owns the `SemanticMappingValue` enum — four variants (Column/Literal/Expr/Metadata) for binding Semantics to physical sources. See `15` for the Binding process.

## 10. `SemanticMapping` Value Shape

A `semantic_mapping:` entry carries one of four variants, resolved at `compile` during the Binding process (`15`):

```rust
#[non_exhaustive]
pub enum SemanticMappingValue {
    /// Bare physical column name — the Semantic is 1:1 to `Column(name)`.
    Column(String),

    /// A literal broadcast over every row — useful for per-source constants.
    Literal(LiteralValue),

    /// An authored physical-expression tree (`UnresolvedPhysicalExpr` =
    /// `Expr<UnresolvedPhysicalLeaf>`, item Y / V3) — anything from a simple
    /// cast to a multi-column compute. References physical column names, not
    /// Semantic names; `Binding` resolves it to `ir::PhysicalExpr` at compile.
    Expr(UnresolvedPhysicalExpr),

    /// A metadata-extraction recipe (§10.4). Compile-synthesized from the
    /// Dimension's `type: metadata` block (per `13 §4.7` / `15 §5.5`); never
    /// authored under `semantic_mapping:`. Distinct from `Expr` because the
    /// extraction is not an expression but a compile-time mechanic that
    /// resolves to a per-source `LiteralValue` stored on
    /// `ResolvedPhysicalSource.metadata_values` (`15 §7.6`).
    Metadata(MetadataDimensionRecipe),
}
```

### 10.1 YAML

```yaml
extras:
  semantic_mapping:
    # Variant 1 — bare column.
    revenue: net_revenue_cents

    # Variant 2 — literal broadcast.
    currency: { literal: "USD" }

    # Variant 3 — physical expression.
    hour_bucket:
      expr:
        trunc:
          column: event_ts
          unit: hour
```

Single-string values dispatch to `Column`; mapping values with `literal:` / `expr:` keys dispatch to `Literal` / `Expr`. The `Metadata` variant has **no author-facing YAML under `semantic_mapping:`** — it is exclusively compile-synthesized from the Dimension's `type: { metadata: { path: { token: N } } }` block authored on the Dimension itself (per `13 §4.7` and the YAML in §4.2 above). The compile stage's binding-resolution pass (`15 §5.5` / `15 §10.4`) produces the `SemanticMappingValue::Metadata(...)` entry before the completeness check runs.

### 10.2 `LiteralValue`

```rust
#[non_exhaustive]
pub enum LiteralValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Decimal(String),
    String(String),
    Date(String),
    Timestamp(String),
}
```

Each literal carries its `DataType`-equivalent kind tag; the compile-time binding validates that the literal's kind is assignable to the Semantic's declared `data_type:`.

### 10.3 `auto` vs explicit

An absent `semantic_mapping:` block on a `Dataset`'s `extras` is equivalent to `semantic_mapping: auto`. Explicit entries narrow that default: every Semantic named in the explicit map receives the declared value; every other Semantic on the `Dataset`'s interface is resolved per `auto` (name-identical physical column).

### 10.4 `MetadataDimensionRecipe`

Compile-synthesized payload of `SemanticMappingValue::Metadata`. The recipe pairs the extraction kind with the Dimension's declared `data_type:`; per-source resolved `LiteralValue`s live on each `ResolvedPhysicalSource.metadata_values` (`15 §7.6`), not on the recipe.

```rust
#[non_exhaustive]
pub struct MetadataDimensionRecipe {
    /// The extraction kind. v1: path-token only.
    pub extraction: MetadataExtraction,

    /// The declared Dimension `data_type:`. Extraction always returns
    /// `String` at the layer-3 mechanic (`15 §8`); this field is the
    /// target of the post-extraction `Cast` invoked during compile.
    pub data_type: DataType,
}

#[non_exhaustive]
pub enum MetadataExtraction {
    /// Extract token at 0-indexed (scheme-stripped) segment position.
    /// Runtime mechanic in `15 §8.1`.
    Path { token: u32 },
    // Future: Partition { level: u32 } — deferred to v2 per `15 §8.0`.
}
```

`MetadataExtraction` is `#[non_exhaustive]` so v2 partition-level extraction (or other compile-time-resolvable metadata kinds) can grow as MINOR per `30 §2`. The v1 roster is **path-only**; partition extraction described in `13 §4.7` is non-goal in v1 and explicitly deferred (no `Partition` variant in v1's `MetadataExtraction`).

The recipe is **never serialized at the author surface** — it is a compile-output struct that lives on the SemanticManifest. The `13 §4.7` `MetadataDimensionBody.source: MetadataSource` shape (the Dimension-type author surface) is what the author writes; the binding-resolution pass converts the author-side body into a recipe for `15`'s consumption.

## Testing Strategy

- Column/Literal/Expr/Metadata variant round-trip; LiteralValue roster; MetadataDimensionRecipe (compile-synthesized, never authored); MetadataExtraction::Path only (v1); auto vs explicit mapping merge; YAML dispatch (bare string → Column, literal:/expr: → Literal/Expr).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
