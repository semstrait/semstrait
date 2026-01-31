use std::fmt;

/// Errors that can occur during query resolution
#[derive(Debug)]
pub enum ResolveError {
    ModelNotFound(String),
    DimensionNotFound(String),
    TableGroupNotFound(String),
    AttributeNotFound { dimension: String, attribute: String },
    MeasureNotFound(String),
    MetricNotFound(String),
    InvalidAttributeFormat(String),
    InvalidQuery(String),
    /// A virtual _table metadata attribute was requested but not available
    MetaAttributeNotFound(String),
    /// A virtual _table metadata attribute requires a value that isn't set
    /// (e.g., table.uuid when uuid is None)
    MetaAttributeNotSet { attribute: String, reason: String },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolveError::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            ResolveError::DimensionNotFound(name) => write!(f, "Dimension '{}' not found", name),
            ResolveError::TableGroupNotFound(name) => write!(f, "TableGroup '{}' not found", name),
            ResolveError::AttributeNotFound { dimension, attribute } => {
                write!(f, "Attribute '{}' not found in dimension '{}'", attribute, dimension)
            }
            ResolveError::MeasureNotFound(name) => write!(f, "Measure '{}' not found", name),
            ResolveError::MetricNotFound(name) => write!(f, "Metric '{}' not found", name),
            ResolveError::InvalidAttributeFormat(s) => {
                write!(f, "Invalid attribute format '{}', expected 'dimension.attribute' or 'tableGroup.dimension.attribute'", s)
            }
            ResolveError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
            ResolveError::MetaAttributeNotFound(name) => {
                write!(f, "Unknown _table attribute '{}'. Available: model, namespace, tableGroup, table, uuid, or any table property key", name)
            }
            ResolveError::MetaAttributeNotSet { attribute, reason } => {
                write!(f, "_table.{} is not available: {}", attribute, reason)
            }
        }
    }
}

impl std::error::Error for ResolveError {}
