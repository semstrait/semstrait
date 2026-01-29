use std::fmt;

/// Errors that can occur during query resolution
#[derive(Debug)]
pub enum ResolveError {
    ModelNotFound(String),
    DimensionNotFound(String),
    AttributeNotFound { dimension: String, attribute: String },
    MeasureNotFound(String),
    MetricNotFound(String),
    InvalidAttributeFormat(String),
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolveError::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            ResolveError::DimensionNotFound(name) => write!(f, "Dimension '{}' not found", name),
            ResolveError::AttributeNotFound { dimension, attribute } => {
                write!(f, "Attribute '{}' not found in dimension '{}'", attribute, dimension)
            }
            ResolveError::MeasureNotFound(name) => write!(f, "Measure '{}' not found", name),
            ResolveError::MetricNotFound(name) => write!(f, "Metric '{}' not found", name),
            ResolveError::InvalidAttributeFormat(s) => {
                write!(f, "Invalid attribute format '{}', expected 'dimension.attribute'", s)
            }
        }
    }
}

impl std::error::Error for ResolveError {}
