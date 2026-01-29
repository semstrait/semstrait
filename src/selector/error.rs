//! Selector error types

use std::fmt;

/// Errors that can occur during table selection
#[derive(Debug)]
pub enum SelectError {
    /// No table can serve the requested query
    NoFeasibleTable {
        model: String,
        reason: String,
    },
    /// Model has no tables defined
    NoTablesInModel {
        model: String,
    },
    /// Multiple tableGroups can serve the query - ambiguous
    /// Use a cross-tableGroup metric to disambiguate
    AmbiguousTableGroup {
        model: String,
        table_groups: Vec<String>,
    },
}

impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFeasibleTable { model, reason } => {
                write!(f, "No table in model '{}' can serve the query: {}", model, reason)
            }
            Self::NoTablesInModel { model } => {
                write!(f, "Model '{}' has no tables defined", model)
            }
            Self::AmbiguousTableGroup { model, table_groups } => {
                write!(
                    f, 
                    "Query for model '{}' matches multiple tableGroups: [{}]. Use a cross-tableGroup metric to combine data from multiple sources.",
                    model,
                    table_groups.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for SelectError {}
