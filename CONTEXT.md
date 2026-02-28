# CONTEXT.md

Context for AI coding assistants (Cursor, Claude Code, etc.) working on semstrait.

## Project Overview

semstrait is a semantic layer for Rust analytics applications. It compiles semantic models to Substrait compute plans, providing both schema context for UIs/LLMs and engine-portable query execution.

## Purpose

semstrait serves two primary use cases:

1. **Semantic context for applications** - Frontend/UX (web or conversational) can consume the semantic model to:
   - Display available dimensions, measures, metrics with labels
   - Generate query builders, filters, visualizations
   - Provide context for LLM-powered interfaces

2. **Engine-agnostic compute plans** - Compile semantic queries to Substrait, which can execute on:
   - DataFusion
   - DuckDB
   - Velox
   - Any Substrait-compatible engine

The semantic model is the single source of truth: define once, use for both UX generation and query execution.

## Architecture

**Noun modules** (data structures):
- `semantic_model/` - Semantic IR: Schema, SemanticModel, Dimension, Measure, Metric (format-agnostic)
- `query/` - Query request types (what the user wants to compute)
- `plan/` - Relational algebra: PlanNode, Expr, Column (close to Substrait)

**Verb modules** (transformations):
- `parser/` - Input format → `semantic_model::Schema` (YAML built-in, extensible)
- `selector/` - Selects optimal dataset from dataset_groups based on query requirements
- `resolver/` - Validates query against schema, resolves attribute references
- `planner/` - Semantic query → relational algebra plan
- `emitter/` - Plan → Substrait protobuf

## Key Design Decisions

- **`semantic_model/` IS the semantic IR** - all input parsers produce these types
- **`plan/` is relational algebra** - not semantic concepts, close to Substrait
- **Type-safe enums** - `DataType`, `Aggregation` validate at parse time, not runtime
- **Enums shared across layers** - e.g., `semantic_model::Aggregation` used directly in `plan::AggregateExpr`
- **Serde with aliases** - YAML strings like "count_distinct" deserialize to enum variants
- **Virtual dimensions** - Dimensions with `virtual: true` have no physical table and emit constant literal values (e.g., `_dataset` for metadata)
- **Model-level dimensions** - Defined at model level, queryable with two-part paths, UNION across dataset_groups
- **Inline dimensions** - Defined in dataset_groups, queryable only with three-part paths (`datasetGroup.dimension.attribute`)
- **Typed NULLs in UNION** - When combining dimensions from different dataset_groups, NULLs carry the correct type for schema compatibility
- **Source types are declarative** - `Source` enum supports `Parquet` (file path) and `Iceberg` (table identifier). Catalog/connection resolution is the service layer's responsibility, not the semantic model's

## Expressions

Measure and metric expressions are structured YAML that deserialize directly to Rust enums:

```yaml
expr:
  multiply: [quantity, priceeach]
```

→ `ExprNode::Multiply(vec![...])` in `semantic_model/measure.rs`

Supported: `add`, `subtract`, `multiply`, `divide`, `case` (CASE WHEN), `column`, `literal`

To add a new expression type: add variant to `ExprNode` enum, serde handles deserialization automatically.

## Code Style

- Validation at parse time via serde, not runtime checks
- Tests in same file: `#[cfg(test)] mod tests { ... }`
- Custom error types with `Display` impl
- Prefer pattern matching over if-let chains

## Common Tasks

| Task | Location |
|------|----------|
| Add data type | `model/types.rs` - add to `DataType` enum, update `FromStr` |
| Add aggregation | `model/types.rs` - add to `Aggregation` enum, update `FromStr` |
| Add input format | Create parser crate that produces `model::Schema` |
| Add plan node | `plan/node.rs` + update `planner/` and `emitter/` |
| Add source type | `semantic_model/datasetgroup.rs` - add variant to `Source` enum, update accessors on `GroupDataset` and `Dimension`, update `semstrait-js` `Source` type |
| Add `_dataset` attribute | 1) Add to model's `_dataset` dimension attributes, 2) Add to dataset's `_dataset` list |
| Add dataset property | 1) Add to `dataset.properties`, 2) Declare in `_dataset` dimension, 3) Add to dataset's `_dataset` list |
| Add shared dimension | Define at model-level `dimensions` (queryable with 2-part path) |
| Add datasetGroup-specific dimension | Define inline in datasetGroup (queryable with 3-part path) |

## Source Types

The `Source` enum (`semantic_model/datasetgroup.rs`) describes where physical data lives:

| Type | YAML | Fields | Notes |
|------|------|--------|-------|
| Parquet | `type: parquet` | `path` | Local or remote file path, supports template variables |
| Iceberg | `type: iceberg` | `table` | Table identifier (e.g., `warehouse.orderfact`). Catalog resolution is the service layer's responsibility |

The planner and emitter never inspect `Source` -- they use `dataset.dataset` and `dimension.table` for Substrait `NamedTable` references. `Source` is consumed by the service layer to register tables in the query engine.

## Dimension Path Types

| Defined At | Path Format | Example | Behavior |
|------------|-------------|---------|----------|
| Model-level | Two-part | `dates.year` | UNION across all dataset_groups |
| Inline in datasetGroup | Three-part | `adwords.campaign.name` | Single datasetGroup only |
| Virtual (model-level) | Two-part | `_dataset.datasetGroup` | Literal values across all dataset_groups |

## Testing

```bash
cargo test           # Run all tests
cargo test types     # Run tests matching "types"
```
