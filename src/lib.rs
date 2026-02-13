//! semstrait - Compile semantic models to Substrait compute plans
//!
//! This library provides:
//! - Schema definition types (SemanticModel, TableGroup, Dimension, Measure, etc.)
//! - Schema parsing from YAML
//! - Table selection (aggregate awareness)
//! - Query resolution
//! - Logical plan generation
//! - Substrait plan emission
//!
//! # Architecture
//!
//! **Noun modules** (data structures):
//! - `semantic_model/` - domain concepts (Schema, SemanticModel, TableGroup, Dimension, Measure)
//! - `query/` - query request types (QueryRequest, DataFilter)
//! - `plan/` - logical plan types (PlanNode, Expr, Column)
//!
//! **Verb modules** (transformations):
//! - `parser/` - YAML → Schema
//! - `selector/` - SemanticModel + Query → Selected tables (aggregate awareness)
//! - `resolver/` - Schema + QueryRequest + Tables → ResolvedQuery
//! - `planner/` - ResolvedQuery → PlanNode
//! - `emitter/` - PlanNode → Substrait
//!
//! # Example
//!
//! ```ignore
//! use semstrait::{parser, select_tables, resolve_query, plan_query, emit_plan, QueryRequest};
//!
//! let schema = parser::parse_file("schema.yaml")?;
//! let request = QueryRequest { model: "sales".into(), ..Default::default() };
//! let model = schema.get_model(&request.model).unwrap();
//! let tables = select_tables(&schema, model, &dims, &measures)?;
//! let resolved = resolve_query(&schema, &request, &tables[0])?;
//! let plan = plan_query(&resolved)?;
//! let substrait = emit_plan(&plan)?;
//! ```

pub mod semantic_model;
pub mod query;
pub mod selector;
pub mod resolver;
pub mod plan;
pub mod planner;
pub mod emitter;
pub mod parser;
pub mod error;

// Re-export commonly used types
pub use semantic_model::{Schema, SemanticModel, TableGroup, GroupTable, TableGroupDimension, DataType, Aggregation, resolve_path_template, resolve_dimension_path_template};
pub use query::{QueryRequest, DataFilter};
pub use selector::{select_tables, SelectedTable, SelectError};
pub use resolver::{resolve_query, ResolvedQuery, ResolveError};
pub use plan::{PlanNode, Expr, Column, AggregateExpr};
pub use planner::{plan_query, plan_semantic_query, PlanError};
pub use emitter::{emit_plan, EmitError};
pub use error::ParseError;
