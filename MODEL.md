# MODEL.md - Semantic Model YAML Format Design

This document describes the target YAML format for semstrait semantic models.

## Overview

The semantic model supports:
- **Multiple tables** serving the same semantic model (aggregate awareness)
- **Table groups** that share dimension and measure definitions
- **Automatic table selection** based on query requirements (scoped to tableGroup)
- **Union of partitioned tables** with disjoint rows
- **Cross-tableGroup metrics** for unified reporting across data sources
- **Metrics-only queries** - measures are internal, metrics are the public API

## Query Interface

Queries are expressed in terms of **dimensions** and **metrics**:

```yaml
# Query request
model: "sales"
rows: ["dates.year", "products.category"]
metrics: ["revenue", "margin", "avg_unit_price"]
filter:
  - field: dates.year
    value: [2024, 2025]
```

**Metrics** are the public query interface. **Measures** are internal implementation details used to define metrics.

### Pass-Through Metrics

To expose a measure directly as a metric, use a simple reference:

```yaml
metrics:
  # Pass-through: directly exposes the underlying measure
  - name: revenue
    expr: revenue    # References the "revenue" measure
    type: f64
  
  # Derived: calculation from multiple measures  
  - name: margin
    expr:
      divide:
        - subtract: [revenue, cost]
        - revenue
    type: f64
```

This design ensures:
- Users don't need to know about tableGroups or physical table structure
- No ambiguity about which source data comes from
- Clear separation between internal implementation (measures) and public API (metrics)

## High-Level Structure

```yaml
models:
  - name: <model_name>
    namespace: <namespace>          # Optional namespace/organization identifier
    dimensions: [...]               # Model-level dimensions (queryable with 2-part paths)
    metrics: [...]                  # Derived calculations (model-level)
    tableGroups: [...]              # Groups of tables sharing field definitions
    dataFilter: [...]               # Row-level security (optional)
```

### Dimension Definition

Dimensions are defined under each model with a `source` configuration:

```yaml
dimensions:
  - name: <dim_name>
    source:
      type: parquet
      path: <path_or_template>   # Supports {model.namespace}, {dimension.name}, etc.
    table: <physical_table>
    label: <display_label>
    attributes: [...]
```

## Dimension Path Types

The location where a dimension is defined determines how it can be queried:

| Defined At | Path Format | Example | Behavior |
|------------|-------------|---------|----------|
| Model-level `dimensions` | Two-part | `dates.year` | UNION across all tableGroups |
| Inline in tableGroup | Three-part | `adwords.campaign.name` | Single tableGroup only |
| Virtual (model-level) | Two-part | `_table.tableGroup` | Literal values across all tableGroups |

### Model-Level Dimensions (Two-Part Path)

Dimensions defined at the model level are shared concepts that can be queried across tableGroups. The planner automatically UNIONs results from all tableGroups that reference the dimension.

```yaml
rows:
  - "dates.date"    # Queries dates.date from ALL tableGroups
metrics: ["revenue"]
```

### Inline Dimensions (Three-Part Path)

Dimensions defined inline within a tableGroup are tableGroup-specific and must be queried with the three-part path `tableGroup.dimension.attribute`.

```yaml
rows:
  - "adwords.campaign.name"      # Only adwords campaigns
  - "facebookads.campaign.name"  # Only facebookads campaigns
metrics: ["revenue"]
```

### Virtual Dimensions (Two-Part Path)

Virtual dimensions (like `_table`) are defined at model level with `virtual: true`. They have no physical table and emit literal values.

```yaml
rows:
  - "_table.tableGroup"  # Shows which tableGroup each row came from
metrics: ["revenue"]
```

## Mixed Dimension Queries

You can combine model-level, inline, and virtual dimensions in the same query. The planner builds a UNION where:

- **Model-level dimensions**: Have values in all rows
- **Inline dimensions**: Have values only for their tableGroup, NULL for others
- **Virtual dimensions**: Have tableGroup-specific literal values in all rows

### Example

Query:
```yaml
rows:
  - "dates.date"              # Model-level (2-part)
  - "adwords.dates.date"      # Inline (3-part)
  - "facebookads.dates.date"  # Inline (3-part)
  - "_table.tableGroup"       # Virtual (2-part)
metrics: ["fun-cost"]
```

Result:
```
| dates.date | adwords.dates.date | facebookads.dates.date | _table.tableGroup | fun-cost |
|------------|--------------------|-----------------------|-------------------|----------|
| 2024-01-01 | 2024-01-01         | null                  | adwords           | 1500     |
| 2024-01-02 | 2024-01-02         | null                  | adwords           | 1800     |
| 2024-01-01 | null               | 2024-01-01            | facebookads       | 2300     |
| 2024-01-02 | null               | 2024-01-02            | facebookads       | 2100     |
```

This enables:
- Seeing the shared dimension value (`dates.date`) for sorting/grouping
- Identifying which tableGroup each row came from via inline columns or `_table.tableGroup`
- Client-side post-processing to compute cross-tableGroup totals

## Table Groups

A `tableGroup` defines:
1. **Dimensions** - all dimension references available to tables in the group
2. **Measures** - all measure definitions shared by tables in the group
3. **Tables** - physical tables, each declaring which subset of fields it has

### Why Table Groups?

- **Partitioned tables**: Multiple tables with identical schemas but disjoint rows (e.g., `orders_2023`, `orders_2024`) can be UNIONed
- **Aggregate tables**: Pre-aggregated tables with fewer dimensions sit alongside detail tables
- **DRY**: Define dimensions and measures once, reference in multiple tables

## Complete Example

```yaml
models:
  - name: steelwheels
    namespace: "a908ff91-c951-4d65-b054-d246d2e8cae1"  # tenant id
    
    # ============================================
    # DIMENSIONS - Shared dimension tables
    # ============================================
    dimensions:
      - name: dates
        source:
          type: parquet
          path: "{model.namespace}/dimensions/{dimension.name}.parquet"
        label: Time
        table: dates
        attributes:
          - name: date
            column: time_id
            label: Date
            type: date
          - name: year
            column: year_id
            label: Year
            type: i32
          - name: quarter
            column: qtr_id
            label: Quarter
            type: i32
          - name: month
            column: month_name
            label: Month
            type: string
      
      - name: markets
        source:
          type: parquet
          path: "{model.namespace}/dimensions/{dimension.name}.parquet"
        label: Geography
        table: customer_w_ter
        attributes:
          - name: customer
            column: customernumber
            label: Customer Number
            type: i32
          - name: territory
            label: Territory
            type: string
          - name: country
            label: Country
            type: string
          - name: state
            label: State Province
            type: string
          - name: city
            label: City
            type: string
    
    # ============================================
    # METRICS - The public query interface
    # ============================================
    metrics:
      # Pass-through metrics (expose measures)
      - name: sales
        label: Total Sales
        expr: sales
        type: f64
      
      - name: quantity
        label: Total Quantity
        expr: quantity
        type: i32
      
      # Derived metric (calculation from measures)
      - name: avg_unit_price
        label: Average Unit Price
        expr:
          divide: [sales, quantity]
        type: f64
    
    # ============================================
    # TABLE GROUPS
    # ============================================
    tableGroups:
      - name: orders
        # ------------------------------------------
        # DIMENSIONS available in this group
        # ------------------------------------------
        dimensions:
          # Joined dimension - references model-level dimension
          - name: dates
            join:
              leftKey: time_id
              rightKey: time_id
          
          # Joined dimension
          - name: markets
            join:
              leftKey: customernumber
              rightKey: customernumber
          
          # Degenerate dimension (columns on fact table, no join)
          - name: flags
            attributes:
              - name: is_premium
                column: is_premium_order
                type: bool
              - name: status
                column: order_status
                type: string
        
        # ------------------------------------------
        # MEASURES shared by all tables in group
        # ------------------------------------------
        measures:
          - name: sales
            label: Sales
            aggregation: sum
            expr: totalprice
            type: f64
          
          - name: quantity
            label: Quantity
            aggregation: sum
            expr: quantityordered
            type: i32
          
          - name: revenue
            label: Revenue (computed)
            aggregation: sum
            expr:
              multiply: [quantityordered, priceeach]
            type: f64
          
          - name: order_count
            aggregation: count
            expr: order_id
        
        # ------------------------------------------
        # TABLES: Each declares its field subset
        # ------------------------------------------
        # Join detection is based on attribute inclusion:
        # - dates: 'date' attr has column: time_id (the key attr)
        # - markets: 'customer' attr has column: customernumber (the key attr)
        # Including the key attribute → JOIN; excluding it → denormalized
        tables:
          # Detail fact table - includes key attrs, needs joins
          - table: orderfact
            source:
              type: parquet
              path: "{model.namespace}/{model.name}/{table.name}.parquet"
            uuid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            properties:
              connectorType: "jdbc"
              sourceSystem: "pentaho"
            dimensions:
              dates: [date, month, quarter, year]    # 'date' present → JOIN
              markets: [customer, territory, country, state, city]  # 'customer' present → JOIN
              flags: [is_premium, status]            # Degenerate, always on fact table
            measures: [sales, quantity, revenue, order_count]
          
          # Denormalized aggregate - excludes key attrs, no joins
          - table: daily_territory_agg
            source:
              type: parquet
              path: "{model.namespace}/{model.name}/{table.name}.parquet"
            dimensions:
              dates: [date, month, year]    # 'date' present but no 'customer' → partial join
              markets: [territory, country]  # No 'customer' → denormalized
            measures: [sales, quantity]
          
          # More aggregated - no key attrs at all
          - table: monthly_summary
            source:
              type: parquet
              path: "{model.namespace}/{model.name}/{table.name}.parquet"
            dimensions:
              dates: [month, year]           # No 'date' → denormalized
            measures: [sales, quantity]
          
          # Most aggregated
          - table: yearly_totals
            source:
              type: parquet
              path: "{model.namespace}/{model.name}/{table.name}.parquet"
            dimensions:
              dates: [year]                  # No 'date' → denormalized
            measures: [sales]
```

## Field Subset Notation

### Dimensions

Tables declare which dimensions/attributes they have:

```yaml
dimensions:
  dates: [date, month, quarter, year]   # Explicit list of attributes
  markets: [territory, country]          # Only these specific attributes
  flags: [is_premium]                    # Single attribute
```

All attribute lists are explicit - no shorthand notation.

### Columns (Optional)

The `columns` section is **optional**. Column types are inferred from:

- **Join key columns** → inferred from dimension's key attribute type
- **Measure columns** → inferred from measure expressions and their types
- **Degenerate dimension columns** → inferred from dimension attribute definitions

You can explicitly specify `columns` for documentation or to override inferred types, but it's not required.

### Join vs Denormalized Detection

The system determines whether to JOIN or use denormalized columns based on **attribute inclusion** in the table's dimension list:

1. **Key Attribute**: Each joined dimension has a "key attribute" - the attribute whose column matches the join's `rightKey`. For the `dates` dimension with `rightKey: time_id`, the key attribute is `date` (which has `column: time_id`).

2. **Detection Logic**:
   - **If table's attribute list includes the key attribute** → JOIN to dimension table
   - **If table's attribute list excludes the key attribute** → attributes are denormalized columns
   - **If dimension has no `join` spec** → always denormalized (degenerate dimension)

Example:
```yaml
# dates dimension: 'date' attribute has column: time_id (the join key)
# markets dimension: 'customer' attribute has column: customernumber (the join key)

# This table includes 'date' and 'customer' → will JOIN both dimensions
- table: steelwheels.orderfact
  dimensions:
    dates: [date, month, year]    # 'date' = key attr → JOIN
    markets: [customer, country]  # 'customer' = key attr → JOIN
    flags: [is_premium]           # Degenerate, no join

# This table excludes 'date' → dates attributes are denormalized
- table: steelwheels.monthly_summary
  dimensions:
    dates: [month, year]          # No 'date' → denormalized columns

# This table excludes both key attributes → fully denormalized
- table: steelwheels.yearly_totals
  dimensions:
    dates: [year]                 # No 'date' → denormalized
    markets: [territory]          # No 'customer' → denormalized
```

This approach eliminates redundancy: the attribute list already implies whether joins are needed based on granularity.

### Measures

Tables declare which measures they support by referencing group-level measure names:

```yaml
measures: [sales, quantity]       # References to group measures
```

## Table Selection (Aggregate Awareness)

### Algorithm

1. **Filter to feasible tables**: Table must have all dimensions and measures required by the query
2. **Select optimal table**: Prefer table with fewest total attributes (most aggregated = likely smallest)
3. **Handle partitioned tables**: If multiple tables with same schema have `rowFilter`, UNION them if query spans multiple filters

### Feasibility Check

```
Query: GROUP BY dates.year, markets.territory WITH measures [sales]

Table: yearly_totals
  - dates: [year] ✓
  - markets: ✗ (not available)
  → NOT FEASIBLE

Table: daily_territory_agg
  - dates: [date, month, year] → has year ✓
  - markets: [territory, country] → has territory ✓
  - measures: [sales, quantity] → has sales ✓
  → FEASIBLE

Table: orderfact
  - dates: [date, month, quarter, year] ✓
  - markets: [customer, territory, country, state, city] ✓
  - measures: [sales, quantity, revenue, order_count] ✓
  → FEASIBLE

Selection: daily_territory_agg (fewer attributes = more aggregated)
```

### Union for Partitioned Tables

```
Query: dates.year IN [2023, 2024], measures [sales]

Tables with rowFilter:
  - orders_2023: rowFilter: { dates.year: 2023 }
  - orders_2024: rowFilter: { dates.year: 2024 }

Both filters match query → UNION both tables
```

### Cross-TableGroup Metrics

Metrics can span multiple tableGroups using `tableGroup.name` conditions. This enables unified metrics across data sources with different measure names:

```yaml
models:
  - name: marketing
    tableGroups:
      - name: google_ads
        measures:
          - name: ad_cost
            aggregation: sum
            expr: cost
        # ...
      
      - name: meta_ads
        measures:
          - name: media_spend
            aggregation: sum
            expr: spend
        # ...
    
    metrics:
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

The planner detects `tableGroup.name` conditions and generates a UNION plan:
1. Query each tableGroup for its mapped measure
2. Project to a common schema
3. Union the results
4. Re-aggregate by the requested dimensions

Use `plan_cross_table_group_query()` for explicit cross-tableGroup planning.

## rowFilter (Partitioning)

Tables can declare a `rowFilter` indicating they contain a subset of data:

```yaml
- table: orders_2023
  rowFilter:
    dates.year: 2023

- table: orders_emea_2023
  rowFilter:
    dates.year: 2023
    markets.territory: "EMEA"
```

The selector uses `rowFilter` to:
1. Determine if a table is relevant for a query's filters
2. Identify tables that can be UNIONed (same schema, disjoint row filters)

## Degenerate Dimensions

Dimensions whose attributes live directly on the fact table (not in a separate dimension table). Defined at the tableGroup level without a `join`:

```yaml
dimensions:
  - name: flags
    attributes:
      - name: is_premium
        column: is_premium_order
        type: bool
      - name: status
        column: order_status
        type: string
```

No `join` = degenerate dimension. Attributes are columns on the fact table itself.

## Model Metadata

Models can have optional metadata:

```yaml
models:
  - name: sales_analytics
    namespace: "com.example.analytics"  # Organization/project identifier
```

| Field | Type | Description |
|-------|------|-------------|
| `namespace` | string | Optional namespace for the model (e.g., organization, project, or domain identifier) |

## Table Metadata

Tables require a `source` configuration and can have optional metadata:

```yaml
tables:
  - table: orderfact
    source:
      type: parquet
      path: "{model.namespace}/{model.name}/{table.name}.parquet"
    uuid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    properties:
      connectorType: "jdbc"
      sourceSystem: "pentaho"
      dataOwner: "analytics-team"
```

| Field | Type | Description |
|-------|------|-------------|
| `source` | object | **Required.** Data source configuration |
| `uuid` | string | Unique identifier for the table (e.g., Iceberg table UUID) |
| `properties` | map<string, string> | Custom key-value properties for connector metadata, lineage, etc. |

## Source Configuration

Both tables and dimensions require a `source` configuration specifying how to access the data:

```yaml
source:
  type: parquet
  path: "/data/{model.namespace}/{table.name}.parquet"
```

Currently supported types: `parquet` (Iceberg support planned)

### Path Template Variables

Path strings support variable substitution:

**For Tables:**

| Variable | Description |
|----------|-------------|
| `{model.name}` | Model name |
| `{model.namespace}` | Model namespace (errors if not set) |
| `{tableGroup.name}` | Table group name |
| `{table.name}` | Physical table name |
| `{table.uuid}` | Table UUID (errors if not set) |

**For Dimensions:**

| Variable | Description |
|----------|-------------|
| `{model.name}` | Model name |
| `{model.namespace}` | Model namespace (errors if not set) |
| `{dimension.name}` | Dimension name |
| `{dimension.table}` | Dimension table name |

Example:
```yaml
# Table source path
source:
  type: parquet
  path: "{model.namespace}/{model.name}/{table.name}.parquet"
# Resolves to: "tenant-uuid/steelwheels/orderfact.parquet"

# Dimension source path
source:
  type: parquet
  path: "{model.namespace}/dimensions/{dimension.name}.parquet"
# Resolves to: "tenant-uuid/dimensions/dates.parquet"
```

## Data Types

Supported types: `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `bool`, `string`, `date`, `timestamp`, `decimal(p,s)`

## Aggregation Functions

Supported: `sum`, `avg`, `count`, `count_distinct`, `min`, `max`

## Measure Expressions

Simple column reference:
```yaml
expr: totalprice
```

Computed expression:
```yaml
expr:
  multiply: [quantity, price]
```

Conditional (CASE WHEN):
```yaml
expr:
  case:
    when:
      - condition:
          gt: [priceeach, 100]
        then: totalprice
    else: 0
```

## Metric Expressions

Metrics operate on aggregated measures:

```yaml
metrics:
  - name: avg_unit_price
    expr:
      divide: [sales, quantity]
```

Supported: `add`, `subtract`, `multiply`, `divide`

## Virtual `_table` Dimension

The `_table` dimension is a virtual dimension that provides access to table and model metadata as queryable attributes. These values are emitted as constant literals in the query output.

Unlike regular dimensions, virtual dimensions have no physical table - they are marked with `virtual: true` and their attribute values come from the model configuration rather than data.

### Declaring the `_table` Dimension

The `_table` dimension must be explicitly declared in the model's `dimensions` list:

```yaml
models:
  - name: steelwheels
    namespace: "tenant-123"
    
    dimensions:
      # Regular dimension (has physical table)
      - name: dates
        table: steelwheels.dates
        source:
          type: parquet
          path: /data/dates.parquet
        attributes:
          - { name: year, column: year_id, type: i32 }
      
      # Virtual dimension (no physical table)
      - name: _table
        virtual: true
        label: Table Metadata
        description: Metadata about the table and model
        attributes:
          - name: model
            label: Model Name
            type: string
          - name: namespace
            label: Model Namespace
            type: string
          - name: tableGroup
            label: Table Group
            type: string
          - name: table
            label: Physical Table
            type: string
          - name: uuid
            label: Table UUID
            type: string
          # Custom properties (synced from Iceberg catalog, etc.)
          - name: sourceSystem
            label: Source System
            type: string
          - name: dataOwner
            label: Data Owner
            type: string
```

### Referencing in TableGroups and Tables

Like regular dimensions, `_table` must be referenced in the tableGroup and each table must declare which attributes it provides:

```yaml
tableGroups:
  - name: orders
    dimensions:
      - name: dates
        join:
          leftKey: time_id
          rightKey: time_id
      # Reference the virtual dimension (no join needed)
      - name: _table
    
    tables:
      - table: steelwheels.orderfact
        uuid: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        properties:
          sourceSystem: "pentaho"
          dataOwner: "analytics-team"
        dimensions:
          dates: [year, quarter, month]
          # Declare which _table attributes this table provides
          _table: [model, namespace, tableGroup, table, uuid, sourceSystem, dataOwner]
        measures: [sales, quantity]
```

### Usage in Queries

Include `_table.*` attributes in `rows` or `columns` just like regular dimension attributes:

```yaml
model: "steelwheels"
rows:
  - "dates.year"
  - "_table.tableGroup"
  - "_table.sourceSystem"
metrics: ["sales"]
```

Result:
```
| dates.year | _table.tableGroup | _table.sourceSystem | sales    |
|------------|-------------------|---------------------|----------|
| 2023       | orders            | pentaho             | 500000   |
| 2024       | orders            | pentaho             | 650000   |
```

### Built-in vs Custom Attributes

| Attribute | Value Source | Description |
|-----------|--------------|-------------|
| `_table.model` | `model.name` | Model name |
| `_table.namespace` | `model.namespace` | Model namespace (required if declared) |
| `_table.tableGroup` | `tableGroup.name` | TableGroup name |
| `_table.table` | `table.table` | Physical table name |
| `_table.uuid` | `table.uuid` | Table UUID (required if declared) |
| `_table.{key}` | `table.properties[key]` | Custom property from `properties` map |

### Cross-TableGroup Queries

When used with cross-tableGroup metrics, each UNION branch gets its own metadata values:

```yaml
model: "marketing"
rows:
  - "dates.date"
  - "_table.tableGroup"
metrics: ["unified_cost"]

# Result:
# | dates.date | _table.tableGroup | unified_cost |
# |------------|-------------------|--------------|
# | 2024-01-01 | google_ads        | 1500.00      |
# | 2024-01-01 | meta_ads          | 2300.00      |
```

### Behavior

- **Self-documenting schema**: UI/LLM can introspect the model to discover available `_table` attributes
- **Explicit declaration**: Only declared attributes can be queried - unknown attributes return an error
- **Not in GROUP BY**: Meta attributes are constant values and are not included in the SQL GROUP BY clause
- **Projected as literals**: The values are emitted as string literals in the Substrait ProjectRel
- **Error on missing value**: If an attribute is declared but the source value isn't set (e.g., `uuid` when table has no UUID), an error is returned
