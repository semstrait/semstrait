---
id: 18-metric
status: ratified
version: 2.1
impl: implemented
---

# Metric

This leaf owns the `Metric` struct and cross-DataKind reference resolution. See [semantics-pools](semantics-pools.md) for root-pool reference grammar and deferred-body pattern.

## 6. `Metric`

```rust
#[non_exhaustive]
pub struct Metric {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in the model's `metrics` map and
    /// in a DataKind's `SemanticInterface`.
    pub id: Option<EntityId>,

    pub name: SemanticsName,

    pub data_type: DataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    /// OPTIONAL on Metric — the expression itself carries the aggregation
    /// intent when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agg: Option<AggregationType>,

    /// REQUIRED on Metric — the derivation expression over Measures /
    /// Dimensions. May be deferred-body on a root-pool Metric and
    /// supplied at every Ref site (see §1.3).
    pub expr: UnresolvedSemanticExpr,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additivity: Option<AdditivityType>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub filters: BTreeMap<EntityId, AggregationFilter>,
}
```

### 6.1 Requirements

- `expr:` is REQUIRED (either inline on a declaration or at every `Ref` site for a deferred-body root Metric — §1.3).
- `agg:` is OPTIONAL — a Metric whose `expr:` is itself an aggregation (e.g. `sum(amount) / count(*)`) carries its aggregation in the expression; `agg:` is the field authors set when they want to state a post-aggregation wrapper explicitly.
- `data_type:` is mandatory at declaration (SR-E-10).

### 6.2 Cross-DataKind references

A root-pool Metric MAY reference Measures / Dimensions from different DataKinds. Resolution rules:

1. **Same-level references.** All names in the Metric's `expr:` are defined at the root pool (as `Measure` / `Dimension` entries). The planner synthesizes the join path via the relationship graph at Request time.
2. **Deferred-body.** The root Metric declares `data_type:`, `agg:` (optional), `additivity:`, etc. but omits `expr:`. Each DataKind that wants to expose the Metric writes `- ref: metric_name` with its local `expr:` (bound to that DataKind's interface).

A root Metric whose `expr:` names Semantics that span multiple DataKinds without a resolvable relationship path is `compile.metric-path-unresolvable { metric, endpoints }` — same diagnostic as a bare Request whose field set spans without a path.

### 6.3 YAML

```yaml
metrics:
  # Inline root-pool Metric — same-level refs against root-pool Measures.
  - name: revenue_per_order
    data_type: double
    expr: total_revenue / order_count
    additivity: non

  # Deferred-body root-pool Metric — expression is supplied at Ref sites.
  - name: conversion_rate
    data_type: double
    agg: avg
    # No expr here — required at every Ref site.

datasets:
  - name: orders
    metrics:
      - ref: conversion_rate
        expr: sum(case_when(is_converted, 1, 0)) / count(*)

  - name: leads
    metrics:
      - ref: conversion_rate
        expr: sum(case_when(lead_converted, 1, 0)) / count(*)
```

## Testing Strategy

- SR-E-10 (data_type required); expr required (inline or deferred-body per-Ref); agg optional; cross-DataKind reference resolution; filters field (AggregationFilter); deferred-body round-trip.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
