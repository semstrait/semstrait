# semstrait

> Compile semantic models to Substrait compute plans

**⚠️ This project is under active development — API may change.**

## What is semstrait?

semstrait is a Rust library that transforms YAML-based semantic model definitions into [Substrait](https://substrait.io/) compute plans, enabling engine-agnostic analytics.

```
YAML Schema → semstrait → Substrait Plan → Any Engine
                                              ├── DataFusion
                                              ├── DuckDB
                                              ├── Velox
                                              └── ...
```

## Features

- **Semantic Modeling** — Define dimensions, measures, metrics, and joins in YAML
- **Query Resolution** — Resolve business queries against the semantic model  
- **Logical Planning** — Generate optimized relational algebra plans
- **Substrait Output** — Emit portable compute plans for any Substrait-compatible engine
- **Degenerate Dimensions** — Support for fact table columns as dimension attributes
- **Metrics** — Derived calculations from measures (e.g., `revenue / quantity`)
- **Lightweight** — Pure Rust library, no runtime server required

## Example

```rust
use semstrait::{parser, resolve_query, plan_query, emit_plan, QueryRequest};

// Parse your semantic model
let schema = parser::parse_file("model.yaml")?;

// Build a query request
let request = QueryRequest {
    model: "sales".to_string(),
    rows: Some(vec!["dates.year".to_string(), "markets.country".to_string()]),
    columns: None,
    measures: Some(vec!["revenue".to_string()]),
    metrics: Some(vec!["avg_unit_price".to_string()]),
    filter: None,
    dimensions: None,
};

// Resolve → Plan → Emit
let resolved = resolve_query(&schema, &request)?;
let plan = plan_query(&resolved)?;
let substrait = emit_plan(&plan, Some(resolved.output_names()))?;

// Execute on DataFusion, DuckDB, or any Substrait consumer
```

## YAML Schema Example

```yaml
models:
  - name: sales
    table: warehouse.order_fact
    columns:
      - name: quantity
        type: i32
      - name: price
        type: f64
    dimensions:
      - name: dates
        join:
          leftKey: date_id
          rightKey: date_id
      - name: products
        join:
          leftKey: product_id
          rightKey: product_id
    measures:
      - name: revenue
        aggregation: sum
        expr: price
      - name: quantity
        aggregation: sum
        expr: quantity
    metrics:
      - name: avg_unit_price
        label: Average Unit Price
        expr:
          divide: [revenue, quantity]

dimensions:
  - name: dates
    table: warehouse.dates
    attributes:
      - name: year
        column: year_num
        type: i32
      - name: quarter
        column: quarter_name
  - name: products
    table: warehouse.products
    attributes:
      - name: category
        column: product_category
      - name: name
        column: product_name
```

## Supported Data Types

semstrait uses a type-safe `DataType` enum for all column and attribute types:

| Type | YAML | Description |
|------|------|-------------|
| `I8` | `i8` | 8-bit signed integer |
| `I16` | `i16` | 16-bit signed integer |
| `I32` | `i32`, `int`, `integer` | 32-bit signed integer |
| `I64` | `i64`, `long`, `bigint` | 64-bit signed integer |
| `F32` | `f32`, `float` | 32-bit floating point |
| `F64` | `f64`, `double` | 64-bit floating point |
| `Bool` | `bool`, `boolean` | Boolean |
| `String` | `string`, `text`, `varchar` | Variable-length string |
| `Date` | `date` | Date (days since Unix epoch) |
| `Timestamp` | `timestamp`, `datetime` | Timestamp (microseconds since epoch) |
| `Decimal` | `decimal(p, s)` | Fixed-point decimal with precision and scale |

Example usage in YAML:

```yaml
columns:
  - name: quantity
    type: i32
  - name: price
    type: decimal(18, 2)
  - name: created_at
    type: timestamp
```

## Supported Aggregations

The `Aggregation` enum defines available aggregate functions for measures:

| Aggregation | YAML | Description |
|-------------|------|-------------|
| `Sum` | `sum` | Sum of values |
| `Avg` | `avg`, `average` | Average of values |
| `Count` | `count` | Count of rows |
| `CountDistinct` | `count_distinct`, `countdistinct`, `distinct_count` | Count of distinct values |
| `Min` | `min`, `minimum` | Minimum value |
| `Max` | `max`, `maximum` | Maximum value |

Example usage in YAML:

```yaml
measures:
  - name: total_revenue
    aggregation: sum
    expr: price
  - name: unique_customers
    aggregation: count_distinct
    expr: customer_id
```

## LLM-Friendly Metadata

semstrait supports metadata fields designed for AI/LLM consumption:

| Field | Available On | Purpose |
|-------|--------------|---------|
| `description` | Dimension, Attribute, Measure, Metric | Human-readable explanation |
| `synonyms` | Measure, Metric | Alternative names for query understanding |
| `examples` | Attribute | Sample values to help LLMs validate queries |

Example:

```yaml
measures:
  - name: revenue
    label: Revenue
    description: "Total revenue from completed orders, excluding refunds"
    synonyms: [sales, total sales, income]
    aggregation: sum
    expr: totalprice

dimensions:
  - name: geography
    description: "Customer location hierarchy"
    attributes:
      - name: country
        label: Country
        examples: ["USA", "UK", "Germany", "Japan"]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        semstrait                            │
├─────────────────────────────────────────────────────────────┤
│  model/     │ Schema, Model, Dimension, Measure, Metric     │
│  query/     │ QueryRequest, DataFilter                      │
│  parser/    │ YAML → Schema                                 │
│  resolver/  │ Schema + QueryRequest → ResolvedQuery         │
│  planner/   │ ResolvedQuery → PlanNode (relational algebra) │
│  emitter/   │ PlanNode → Substrait Plan                     │
└─────────────────────────────────────────────────────────────┘
```

## Status

🚧 **Early Development** — Core functionality works but API may change.

- ✅ YAML schema parsing
- ✅ Query resolution with dimension/measure/metric support
- ✅ Logical plan generation with joins, filters, aggregations
- ✅ Substrait plan emission
- ✅ Degenerate dimensions (fact table columns)
- ✅ Metric calculations (derived from measures)
- ✅ Type-safe data types (`DataType` enum)
- ✅ Type-safe aggregations (`Aggregation` enum)
- ✅ LLM-friendly metadata (description, synonyms, examples)
- 🔲 Schema validation
- 🔲 LookML parser support
- 🔲 More aggregation functions

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
