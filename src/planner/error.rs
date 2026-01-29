//! Planner errors

use std::fmt;

#[derive(Debug)]
pub enum PlanError {
    /// No measures or dimensions specified
    EmptyQuery,
    /// Invalid query configuration
    InvalidQuery(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::EmptyQuery => {
                write!(f, "Query must have at least one measure or dimension")
            }
            PlanError::InvalidQuery(msg) => {
                write!(f, "Invalid query: {}", msg)
            }
        }
    }
}

impl std::error::Error for PlanError {}
