use serde::Deserialize;

/// Filter for analytics queries
#[derive(Debug, Deserialize, Clone)]
pub struct DataFilter {
    pub field: String,
    /// Optional operator, defaults to "in" for array values or "eq" for single values
    #[serde(default)]
    pub operator: Option<String>,
    pub value: serde_json::Value,
}

/// Request body for analytics queries
/// 
/// Queries are expressed in terms of dimensions (for grouping) and metrics (for values).
/// Metrics are the public API - measures are internal implementation details.
#[derive(Debug, Deserialize, Default)]
pub struct QueryRequest {
    #[serde(default)]
    pub model: String,
    pub dimensions: Option<Vec<String>>,
    pub rows: Option<Vec<String>>,
    pub columns: Option<Vec<String>>,
    /// Metrics to compute - derived calculations from measures
    pub metrics: Option<Vec<String>>,
    pub filter: Option<Vec<DataFilter>>,
}
