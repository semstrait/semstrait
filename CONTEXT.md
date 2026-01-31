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
- `model/` - Semantic IR: Schema, Model, Dimension, Measure, Metric (format-agnostic)
- `query/` - Query request types (what the user wants to compute)
- `plan/` - Relational algebra: PlanNode, Expr, Column (close to Substrait)

**Verb modules** (transformations):
- `parser/` - Input format → `model::Schema` (YAML built-in, extensible)
- `selector/` - Selects optimal table from tableGroups based on query requirements
- `resolver/` - Validates query against schema, resolves attribute references
- `planner/` - Semantic query → relational algebra plan
- `emitter/` - Plan → Substrait protobuf

## Key Design Decisions

- **`model/` IS the semantic IR** - all input parsers produce these types
- **`plan/` is relational algebra** - not semantic concepts, close to Substrait
- **Type-safe enums** - `DataType`, `Aggregation` validate at parse time, not runtime
- **Enums shared across layers** - e.g., `model::Aggregation` used directly in `plan::AggregateExpr`
- **Serde with aliases** - YAML strings like "count_distinct" deserialize to enum variants
- **Virtual dimensions** - Dimensions with `virtual: true` have no physical table and emit constant literal values (e.g., `_table` for metadata)

## Expressions

Measure and metric expressions are structured YAML that deserialize directly to Rust enums:

```yaml
expr:
  multiply: [quantity, priceeach]
```

→ `ExprNode::Multiply(vec![...])` in `model/measure.rs`

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
| Add `_table` attribute | 1) Add to model's `_table` dimension attributes, 2) Add to table's `_table` list |
| Add table property | 1) Add to `table.properties`, 2) Declare in `_table` dimension, 3) Add to table's `_table` list |

## Testing

```bash
cargo test           # Run all tests
cargo test types     # Run tests matching "types"
```
