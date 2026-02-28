//! Selector error types

use std::fmt;

/// Errors that can occur during dataset selection
#[derive(Debug)]
pub enum SelectError {
    /// No dataset can serve the requested query
    NoFeasibleDataset {
        model: String,
        reason: String,
    },
    /// Model has no datasets defined
    NoDatasetsInModel {
        model: String,
    },
    /// Multiple datasetGroups can serve the query - ambiguous
    /// Use a cross-datasetGroup metric to disambiguate
    AmbiguousDatasetGroup {
        model: String,
        dataset_groups: Vec<String>,
    },
}

impl fmt::Display for SelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFeasibleDataset { model, reason } => {
                write!(f, "No dataset in model '{}' can serve the query: {}", model, reason)
            }
            Self::NoDatasetsInModel { model } => {
                write!(f, "Model '{}' has no datasets defined", model)
            }
            Self::AmbiguousDatasetGroup { model, dataset_groups } => {
                write!(
                    f, 
                    "Query for model '{}' matches multiple datasetGroups: [{}]. Use a cross-datasetGroup metric to combine data from multiple sources.",
                    model,
                    dataset_groups.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for SelectError {}
