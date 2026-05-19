//! API error types.

use thiserror::Error;

/// Errors from the SemstraitEngine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("compile error: {0}")]
    Compile(#[from] semstrait_manifest::CompileError),

    #[error("plan error: {0}")]
    Plan(#[from] semstrait_planner::PlannerError),

    #[error("emit error: {0}")]
    Emit(#[from] semstrait_adapter::sql::EmitError),

    #[error("adapt error: {0}")]
    Adapt(#[from] semstrait_adapter::AdaptError),

    #[error("not configured: {0}")]
    NotConfigured(String),

    #[error("internal error: {0}")]
    Internal(String),
}

/// Errors from request parsing.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("entity not found: {0}")]
    EntityNotFound(String),

    #[error("dimension not found: {name} in entity {entity}")]
    DimensionNotFound { entity: String, name: String },

    #[error("measure not found: {name} in entity {entity}")]
    MeasureNotFound { entity: String, name: String },

    #[error("unknown select name: {name} in entity {entity}")]
    UnknownSelectName { entity: String, name: String },

    #[error("named filter not found: {name} in entity {entity}")]
    FilterNotFound { entity: String, name: String },

    #[error("raw filter operator '{operator}' is not recognized")]
    RawFilterInvalidOperator { operator: String },

    #[error("raw filter on field '{field}' has invalid value: {message}")]
    RawFilterInvalidValue { field: String, message: String },

    #[error(
        "raw filter field '{name}' names a DataKindFilter declared on entity '{entity}'; \
         use the `filters` request field to activate a named filter"
    )]
    RawFilterNamesNamedFilter { entity: String, name: String },

    #[error("invalid grain: {0}")]
    InvalidGrain(String),

    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    #[error("validation error: {0}")]
    Validation(String),
}
