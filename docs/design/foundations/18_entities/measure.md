---
id: 18-measure
status: ratified
version: 2.1
impl: implemented
---

# Measure

This leaf owns the `Measure` struct, `AggregationType` roster, and `AdditivityType` classification. See [semantics-pools](semantics-pools.md) for root-pool reference grammar and [filter-taxonomy](filter-taxonomy.md) for `AggregationFilter`.

## 5. `Measure`

```rust
#[non_exhaustive]
pub struct Measure {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in the model's `measures` map and
    /// in a DataKind's `SemanticInterface`.
    pub id: Option<EntityId>,

    pub name: SemanticsName,

    pub data_type: DataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    /// REQUIRED on Measure declarations. Selects the aggregation family.
    pub agg: AggregationType,

    /// Optional horizontal-only transform applied before aggregation.
    /// E.g. `expr: amount_cents * 0.01` + `agg: sum` gives `sum(amount_cents * 0.01)`.
    /// `None` means the aggregation is applied directly to the Semantics named
    /// by the Measure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<UnresolvedSemanticExpr>,

    /// Optional additivity classification per `AdditivityType`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additivity: Option<AdditivityType>,

    /// Measure-level conditional-aggregation filters (see §7), keyed by EntityId.
    /// Each filter wraps the Measure in a `CASE WHEN ... THEN expr ELSE NULL END`
    /// at compile time. Multiple filters AND together regardless of order.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub filters: BTreeMap<EntityId, AggregationFilter>,
}
```

### 5.1 `AggregationType` roster

```rust
#[non_exhaustive]
pub enum AggregationType {
    Sum,
    Avg,
    Count,
    CountDistinct,
    Min,
    Max,
    Median,
    StdDev,
    Variance,
}
```

`Count` and `CountDistinct` are separate variants. Internally, function-level modifiers (distinct over specific columns, percentile variants) may be richer, but the model-level `agg:` surface carries only this roster.

### 5.2 `AdditivityType` roster

**Two-source SoC.** This is the **model-level** Additivity declaration carried per-Measure / per-Metric. The function-level Additivity (carried on the canonical aggregate function itself) lives on `FunctionSpec.additivity` per `[14a §3.6](../14a_function_catalog.md)`. The composition rule that turns the two sources into an **effective** additivity consumed by the planner is owned by `[19 §6.5](../19_expression_flow.md)`. Variants map 1:1 to the unified shape in `14a §3.6` (`Full → Additive`, `Semi(SemiAdditivity) → SemiAdditive { axes }`, `Non → NonAdditive`); the rename to the canonical names happens during the planner-doc rebase. The model-level form retains the per-Semantics-name `axes` precision (a Measure can declare semi-additivity against a specific Dimension like `snapshotted_at`) plus the `strategy` field below — both are extensions beyond the function-level form.

```rust
#[non_exhaustive]
pub enum AdditivityType {
    Full,
    Semi(SemiAdditivity),
    Non,
}

#[non_exhaustive]
pub struct SemiAdditivity {
    /// The Dimension axes along which this Measure is semi-additive.
    pub axes: Vec<SemanticsName>,
    /// Rollup strategy for the non-additive axes.
    pub strategy: SemiAdditivityStrategy,
}

#[non_exhaustive]
pub enum SemiAdditivityStrategy {
    Latest,
    Earliest,
    Average,
    First,
    Last,
}
```

### 5.3 YAML

```yaml
measures:
  # Full-additive Measure with a horizontal transform.
  - name: gross_revenue
    data_type: decimal(18, 2)
    agg: sum
    expr: amount_cents * 0.01
    additivity: full

  # Semi-additive — inventory is additive across warehouses, snapshot-latest over time.
  - name: inventory_on_hand
    data_type: long
    agg: sum
    additivity:
      semi:
        axes: [snapshotted_at]
        strategy: latest

  # Non-additive — a ratio precomputed at source.
  - name: margin_rate
    data_type: double
    agg: avg
    additivity: non
```

### 5.4 Requirements

- `agg:` is REQUIRED on every Measure declaration (root or inline). Missing `agg:` is `parse.measure-missing-agg` (SR-E-9).
- `expr:` is OPTIONAL; when absent, the aggregation applies to the Semantic whose name the Measure declares (via `semantic_mapping` resolution at binding time).
- `additivity:` is optional; `None` is treated as `Full` for planning purposes, with a `plan.additivity-unspecified` advisory.
- `data_type:` is mandatory at declaration (SR-E-10).

## Testing Strategy

- SR-E-9 (agg required), SR-E-10 (data_type required); AggregationType roster round-trip; AdditivityType (Full/Semi/Non + SemiAdditivity axes/strategy); filters field (AggregationFilter keyed by EntityId); expr optional vs binding.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
