# semstrait - Semantic Model Specification

**Version:** 1.0

## Goals

- **Engine-Agnostic**: Compile semantic models to Substrait compute plans executable on any compatible engine (DataFusion, DuckDB, Velox, etc.)
- **Multi-Source**: Support multiple data sources (tableGroups) in a single model with automatic UNION handling
- **Aggregate Awareness**: Automatically select optimal pre-aggregated tables based on query requirements
- **Conformed Dimensions**: Query dimensions across tableGroups with simple two-part paths
- **Metrics-First API**: Expose metrics as the public interface; measures are internal implementation details

## Table of Contents

1. [Enumerations](#enumerations)
2. [Semantic Model](#semantic-model)
3. [Dimensions](#dimensions)
4. [Table Groups](#table-groups)
5. [Measures](#measures)
6. [Metrics](#metrics)
7. [Query Request](#query-request)
8. [Examples](#examples)

---

## Enumerations

### Data Types

Supported data types for columns and attributes.

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

### Aggregations

Supported aggregation functions for measures.

| Aggregation | YAML | Description |
|-------------|------|-------------|
| `Sum` | `sum` | Sum of values |
| `Avg` | `avg`, `average` | Average of values |
| `Count` | `count` | Count of rows |
| `CountDistinct` | `count_distinct`, `distinct_count` | Count of distinct values |
| `Min` | `min`, `minimum` | Minimum value |
| `Max` | `max`, `maximum` | Maximum value |

### Expression Operators

Supported operators in measure and metric expressions.

| Operator | Description |
|----------|-------------|
| `add` | Addition of two values |
| `subtract` | Subtraction |
| `multiply` | Multiplication |
| `divide` | Division |
| `case` | Conditional (CASE WHEN) |
| `column` | Column reference |
| `literal` | Literal value |

---

## Semantic Model

The top-level container representing a complete semantic model.

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the model |
| `namespace` | string | No | Organization/tenant identifier |
| `dimensions` | array | No | Model-level dimensions (queryable with 2-part paths) |
| `metrics` | array | No | Derived calculations (model-level) |
| `tableGroups` | array | Yes | Groups of tables sharing field definitions |
| `dataFilter` | array | No | Row-level security filters |

### Example

```yaml
semantic_models:
  - name: sales_analytics
    namespace: "tenant-123"
    dimensions: []
    metrics: []
    tableGroups: []
```

---

## Dimensions

Dimensions represent business entities used for grouping and filtering. They can be:
- **Joined dimensions**: Separate tables joined to fact tables
- **Degenerate dimensions**: Columns directly on fact tables
- **Virtual dimensions**: No physical table, emit literal values (e.g., `_table` metadata)

### Dimension Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the dimension |
| `table` | string | No | Physical table name (not required for virtual) |
| `source` | object | No | Data source configuration |
| `virtual` | boolean | No | If true, dimension has no physical table |
| `label` | string | No | Human-readable display name |
| `description` | string | No | Human-readable description |
| `alias` | string | No | Alias for column qualification |
| `attributes` | array | Yes | Row-level attributes |

### Attribute Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the attribute |
| `column` | string | No | Physical column name (defaults to `name`) |
| `type` | DataType | Yes | Data type of the attribute |
| `label` | string | No | Human-readable display name |
| `description` | string | No | Human-readable description |
| `examples` | array | No | Sample values for AI/LLM context |

### Source Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Source type (`parquet`, etc.) |
| `path` | string | Yes | Path or template to data file |

### Path Template Variables

| Variable | Description |
|----------|-------------|
| `{model.name}` | Model name |
| `{model.namespace}` | Model namespace |
| `{dimension.name}` | Dimension name |
| `{dimension.table}` | Dimension table name |

### Example

```yaml
dimensions:
  # Regular joined dimension
  - name: dates
    table: dates
    source:
      type: parquet
      path: "{model.namespace}/dimensions/{dimension.name}.parquet"
    label: Time
    attributes:
      - name: date
        column: time_id
        type: date
      - name: year
        column: year_id
        type: i32
  
  # Virtual dimension (metadata)
  - name: _table
    virtual: true
    label: Table Metadata
    attributes:
      - name: model
        type: string
      - name: tableGroup
        type: string
      - name: table
        type: string
```

---

## Dimension Path Types

The location where a dimension is defined determines how it can be queried:

| Defined At | Path Format | Example | Behavior |
|------------|-------------|---------|----------|
| Model-level `dimensions` | Two-part | `dates.year` | UNION across all tableGroups |
| Inline in tableGroup | Three-part | `adwords.campaign.name` | Single tableGroup only |
| Virtual (model-level) | Two-part | `_table.tableGroup` | Literal values across all tableGroups |

**Model-level dimensions** are shared concepts that can be queried across tableGroups. The planner automatically UNIONs results from all tableGroups that reference the dimension.

**Inline dimensions** are tableGroup-specific and must be queried with the three-part path `tableGroup.dimension.attribute`.

---

## Table Groups

A tableGroup defines a collection of tables sharing dimension and measure definitions. Multiple tableGroups enable multi-source analytics (e.g., Google Ads + Facebook Ads).

### TableGroup Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier for the tableGroup |
| `dimensions` | array | Yes | Dimension references available in this group |
| `measures` | array | Yes | Measure definitions shared by tables |
| `tables` | array | Yes | Physical tables in this group |

### TableGroup Dimension Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Reference to model-level dimension |
| `join` | object | No | Join specification (omit for degenerate) |
| `attributes` | array | No | Inline attributes (for degenerate dimensions) |

### Join Specification

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `leftKey` | string | Yes | Column on fact table |
| `rightKey` | string | Yes | Column on dimension table |
| `rightAlias` | string | No | Alias for dimension table in query |

### Table Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `table` | string | Yes | Physical table name |
| `source` | object | Yes | Data source configuration |
| `uuid` | string | No | Unique identifier (e.g., Iceberg UUID) |
| `properties` | map | No | Custom key-value metadata |
| `dimensions` | map | Yes | Dimension → attributes mapping |
| `measures` | array | Yes | Available measures |
| `rowFilter` | object | No | Partition filter for this table |

### Example

```yaml
tableGroups:
  - name: orders
    dimensions:
      - name: dates
        join:
          leftKey: time_id
          rightKey: time_id
      - name: flags
        attributes:
          - name: is_premium
            column: is_premium_order
            type: bool
    
    measures:
      - name: sales
        aggregation: sum
        expr: totalprice
        type: f64
    
    tables:
      - table: orderfact
        source:
          type: parquet
          path: "{model.namespace}/{table.name}.parquet"
        uuid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        dimensions:
          dates: [date, month, year]
          flags: [is_premium]
        measures: [sales]
```

---

## Measures

Measures are aggregated calculations defined at the tableGroup level. They are internal implementation details—users query metrics, not measures directly.

### Measure Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier |
| `aggregation` | Aggregation | Yes | Aggregation function |
| `expr` | string/object | Yes | Column reference or expression |
| `type` | DataType | No | Result data type |
| `label` | string | No | Human-readable display name |
| `description` | string | No | Human-readable description |
| `synonyms` | array | No | Alternative names for AI/LLM |

### Expression Types

**Simple Column:**
```yaml
expr: totalprice
```

**Computed:**
```yaml
expr:
  multiply: [quantity, price]
```

**Conditional (CASE WHEN):**
```yaml
expr:
  case:
    when:
      - condition:
          gt: [priceeach, 100]
        then: totalprice
    else: 0
```

---

## Metrics

Metrics are the public query interface. They can be pass-through (exposing a measure) or derived (calculations from multiple measures).

### Metric Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique identifier |
| `expr` | string/object | Yes | Measure reference or expression |
| `type` | DataType | No | Result data type |
| `label` | string | No | Human-readable display name |
| `description` | string | No | Human-readable description |
| `synonyms` | array | No | Alternative names for AI/LLM |

### Examples

**Pass-through Metric:**
```yaml
- name: revenue
  expr: revenue    # References "revenue" measure
  type: f64
```

**Derived Metric:**
```yaml
- name: margin
  expr:
    divide:
      - subtract: [revenue, cost]
      - revenue
  type: f64
```

**Cross-TableGroup Metric:**
```yaml
- name: unified_cost
  type: f64
  expr:
    case:
      when:
        - condition:
            eq: [tableGroup.name, "google_ads"]
          then: ad_cost
        - condition:
            eq: [tableGroup.name, "meta_ads"]
          then: media_spend
      else: 0
```

---

## Query Request

Queries are expressed in terms of dimensions and metrics.

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Target model name |
| `rows` | array | No | Dimension attributes for row grouping |
| `columns` | array | No | Dimension attributes for column pivoting |
| `metrics` | array | No | Metrics to compute |
| `filter` | array | No | Filters to apply |

### Filter Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `field` | string | Yes | Dimension attribute path |
| `operator` | string | No | Comparison operator (defaults based on value) |
| `value` | any | Yes | Filter value(s) |

### Example

```yaml
model: "sales"
rows: 
  - "dates.year"
  - "markets.territory"
metrics: 
  - "revenue"
  - "margin"
filter:
  - field: dates.year
    value: [2024, 2025]
```

---

## Complete Example

```yaml
semantic_models:
  - name: marketing
    namespace: "tenant-123"
    
    dimensions:
      - name: dates
        table: dates
        source:
          type: parquet
          path: "{model.namespace}/dimensions/dates.parquet"
        attributes:
          - { name: date, type: date }
          - { name: year, type: i32 }
      
      - name: _table
        virtual: true
        attributes:
          - { name: tableGroup, type: string }
    
    metrics:
      - name: total_cost
        expr: cost
        type: f64
    
    tableGroups:
      - name: adwords
        dimensions:
          - name: dates
            join:
              leftKey: date_id
              rightKey: date
          - name: campaign
            attributes:
              - { name: id, type: string }
              - { name: name, type: string }
        
        measures:
          - name: cost
            aggregation: sum
            expr: spend
            type: f64
        
        tables:
          - table: adwords_daily
            source:
              type: parquet
              path: "{model.namespace}/adwords/daily.parquet"
            dimensions:
              dates: [date, year]
              campaign: [id, name]
            measures: [cost]
      
      - name: facebookads
        dimensions:
          - name: dates
            join:
              leftKey: date_id
              rightKey: date
          - name: campaign
            attributes:
              - { name: id, type: string }
              - { name: name, type: string }
        
        measures:
          - name: cost
            aggregation: sum
            expr: amount_spent
            type: f64
        
        tables:
          - table: facebook_daily
            source:
              type: parquet
              path: "{model.namespace}/facebook/daily.parquet"
            dimensions:
              dates: [date, year]
              campaign: [id, name]
            measures: [cost]
```

---

## AI/LLM Context

semstrait supports metadata fields for AI consumption:

| Field | Available On | Purpose |
|-------|--------------|---------|
| `description` | Dimension, Attribute, Measure, Metric | Human-readable explanation |
| `synonyms` | Measure, Metric | Alternative names for query understanding |
| `examples` | Attribute | Sample values to validate queries |
| `label` | All | Human-readable display name |

---

## Version History

- **1.0** (2026-01): Initial specification
  - Core semantic model structure
  - Table groups with aggregate awareness
  - Conformed and virtual dimensions
  - TableGroup-qualified dimension paths
  - Cross-tableGroup UNION with typed NULLs
  - Metrics as public API

---

## License

Licensed under the Apache License, Version 2.0.
