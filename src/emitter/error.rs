//! Emitter errors

use std::fmt;

#[derive(Debug)]
pub enum EmitError {
    /// Unsupported plan node type
    UnsupportedNode(String),
    /// Unsupported expression type
    UnsupportedExpression(String),
    /// Missing required field
    MissingField(String),
    /// Column not found in schema context
    ColumnNotFound(String),
    /// Invalid plan structure
    InvalidPlan(String),
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmitError::UnsupportedNode(node) => {
                write!(f, "Unsupported plan node: {}", node)
            }
            EmitError::UnsupportedExpression(expr) => {
                write!(f, "Unsupported expression: {}", expr)
            }
            EmitError::MissingField(field) => {
                write!(f, "Missing required field: {}", field)
            }
            EmitError::ColumnNotFound(col) => {
                write!(f, "Column not found in schema: {}", col)
            }
            EmitError::InvalidPlan(msg) => {
                write!(f, "Invalid plan: {}", msg)
            }
        }
    }
}

impl std::error::Error for EmitError {}

