//! Planner error types.

use thiserror::Error;

use crate::inline_filter::InlineFilterError;

/// Errors that can occur during query planning.
#[derive(Debug, Error)]
pub enum PlannerError {
    /// Constraint violation (step 0, pre-resolution).
    #[error("constraint violation on {entity}: {message}")]
    ConstraintViolation {
        entity: String,
        message: String,
    },

    /// Kind not found in the manifest.
    #[error("kind not found: {0}")]
    KindNotFound(String),

    /// Dimension not found in the kind.
    #[error("dimension '{dimension}' not found in kind '{kind}'")]
    DimensionNotFound {
        kind: String,
        dimension: String,
    },

    /// Measure not found in the kind.
    #[error("measure '{measure}' not found in kind '{kind}'")]
    MeasureNotFound {
        kind: String,
        measure: String,
    },

    /// No dataset can cover the requested dimensions and measures.
    #[error("no covering dataset for kind '{kind}': {reason}")]
    NoCoveringDataset {
        kind: String,
        reason: String,
    },

    /// Unsupported kind type for planning.
    #[error("unsupported kind type: {0}")]
    UnsupportedKindType(String),

    /// Internal planner error (bug).
    #[error("internal planner error: {0}")]
    Internal(String),

    /// Optimizer pass failed.
    #[error("optimizer pass '{pass}' failed: {reason}")]
    OptimizerError {
        pass: String,
        reason: String,
    },

    /// Inline raw filter lowering failed in ad-hoc mode (where `from` was
    /// omitted and the lowering is deferred until entity resolution).
    /// Carries the typed cause from `inline_filter::lower_inline_filter`.
    #[error("inline filter resolution failed: {0}")]
    InlineFilterResolution(#[from] InlineFilterError),
}
