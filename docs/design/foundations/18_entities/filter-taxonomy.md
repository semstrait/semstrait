---
id: 18-filter-taxonomy
status: ratified
version: 2.1
impl: implemented
---

# Filter Taxonomy

This leaf owns the two distinct filter types — `DataKindFilter` (DataKind-level predicate) and `AggregationFilter` (Measure/Metric-level conditional aggregation). They are disjoint namespaces and may not cross-reference.

## 7. Filter Taxonomy

There are **two distinct filter types** in the model, sharing no common supertype. They serve different functional roles and may not cross-reference each other.

### 7.1 `DataKindFilter` — DataKind-level, user-facing predicate

A DataKind-level filter narrows the rowset a DataKind exposes. Authored under a DataKind's `SemanticInterface.filters:` and resolved against the DataKind's Dimension / Measure surface.

```rust
#[non_exhaustive]
pub struct DataKindFilter {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in the DataKind's
    /// `SemanticInterface` filter map.
    pub id: Option<EntityId>,
    pub name: FilterName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,
    pub expr: UnresolvedSemanticExpr,
}
```

YAML:

```yaml
datasets:
  - name: orders
    filters:
      - name: high_value_only
        expr: amount_cents >= 10000
        description: "Only orders ≥ $100."
```

A Request may name a DataKindFilter to activate it at query time:

```yaml
request:
  from: orders
  filters: [high_value_only]
```

Functional role: **gate the DataKind's rowset before any aggregation**. Equivalent to a WHERE clause at the semantic level.

### 7.2 `AggregationFilter` — Measure / Metric-level, conditional aggregation

A Measure / Metric-level filter specifies a conditional inside the aggregation. Authored under a Measure's or Metric's `filters:` list.

```rust
#[non_exhaustive]
pub struct AggregationFilter {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in a Measure's / Metric's `filters` map.
    pub id: Option<EntityId>,
    pub name: FilterName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub expr: UnresolvedSemanticExpr,
}
```

YAML:

```yaml
measures:
  - name: revenue_usd_only
    data_type: decimal(18, 2)
    agg: sum
    expr: amount_cents * 0.01
    filters:
      - name: usd_only
        expr: currency == 'USD'
```

Compile lowering: the aggregation is wrapped in `CASE WHEN <filter.expr> THEN <measure.expr> ELSE NULL END`; the aggregation is then applied to that wrapped expression. Multiple filters on a single Measure AND together.

Functional role: **conditionally contribute rows to the aggregation**. Equivalent to `SUM(CASE WHEN ... THEN amount ELSE NULL END)` at the adapter level.

### 7.3 No cross-referencing

The two filter types are disjoint: a `DataKindFilter` cannot be named from a Measure / Metric's `filters:` list (and vice versa). The namespace boundary is the authoring site:

- A Request's `filters: [name]` resolves only against `DataKindFilter` names.
- A Measure's / Metric's `filters:` entries are local declarations (inline) — they have no global name table and are not addressable by Request filters.

Attempting a cross-reference is `validate.wrong-filter-error { name, expected, actual }` (SR-E-11).

## Testing Strategy

- SR-E-11 (cross-referencing rejection); DataKindFilter struct (id/name/expr); AggregationFilter struct (id/name/expr, no ai_context); namespace disjoint; CASE WHEN lowering contract; resolution-site distinction.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
