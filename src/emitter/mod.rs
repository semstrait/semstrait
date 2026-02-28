//! Emitters (verb modules)
//!
//! Transforms a PlanNode into an output format.
//!
//! - `substrait` – Substrait protobuf Plan
//! - `sql` – ANSI SQL string

mod substrait;
mod sql;
mod error;

pub use substrait::emit_plan;
pub use sql::emit_sql;
pub use error::EmitError;

