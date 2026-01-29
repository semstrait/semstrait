//! Substrait emitter (verb module)
//!
//! Transforms a PlanNode into a Substrait Plan.

mod emit;
mod error;

pub use emit::emit_plan;
pub use error::EmitError;

