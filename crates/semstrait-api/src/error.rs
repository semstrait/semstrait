//! API error types.

use semstrait_planner::inline_filter::InlineFilterError;
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

    /// Inline raw filter references a field that is not in the resolved entity's interface.
    /// Per `docs/design/foundations/11_names_and_scopes.md §6.4.2`, the field must resolve
    /// to a known Dimension / Measure / Metric / Key in the request's scope.
    #[error("inline filter field not found: {field} in entity {entity}")]
    RawFilterFieldNotFound { entity: String, field: String },

    /// Inline raw filter uses an operator outside the canonical v1 set.
    /// Accepted: eq, ne, lt, le, gt, ge, in, like (plus common symbolic aliases).
    #[error("invalid inline filter operator '{operator}' for field {field}")]
    RawFilterOperatorInvalid { field: String, operator: String },

    /// Inline raw filter `value` failed type-check against the field's `DataType`.
    #[error("inline filter value type mismatch on field {field}: expected {expected}, got {got}")]
    RawFilterValueTypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    #[error("invalid grain: {0}")]
    InvalidGrain(String),

    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    #[error("validation error: {0}")]
    Validation(String),
}

/// Map planner-side inline-filter lowering errors back to API-layer variants.
/// Used in the explicit-`from` path where the parser lowers raw filters
/// against the named entity's interface via
/// `semstrait_planner::inline_filter::lower_inline_filter`.
impl From<InlineFilterError> for ParseError {
    fn from(err: InlineFilterError) -> Self {
        match err {
            InlineFilterError::FieldNotFound { entity, field } => {
                ParseError::RawFilterFieldNotFound { entity, field }
            }
            InlineFilterError::OperatorInvalid { field, operator } => {
                ParseError::RawFilterOperatorInvalid { field, operator }
            }
            InlineFilterError::ValueTypeMismatch {
                field,
                expected,
                got,
            } => ParseError::RawFilterValueTypeMismatch {
                field,
                expected,
                got,
            },
            // FieldOnNoEntity is multi-entity ad-hoc only; never surfaces in
            // the explicit-`from` parser path. Map defensively to a flat
            // ValueTypeMismatch shape so the API doesn't silently swallow it.
            InlineFilterError::FieldOnNoEntity { field, candidates } => {
                ParseError::RawFilterFieldNotFound {
                    entity: candidates.join(", "),
                    field,
                }
            }
        }
    }
}
