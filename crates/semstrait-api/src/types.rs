//! API request/response types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Raw query request from external clients (JSON/CLI/gRPC).
/// Parsed into ResolvedQueryRequest by RequestParser.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawQueryRequest {
    /// Semantic model source (file path for CLI, inline YAML/JSON for REST/gRPC).
    #[serde(default)]
    pub model: Option<String>,
    /// Entity to query: a kind name or a dataset name. If None, planner resolves from select.
    #[serde(default)]
    pub from: Option<String>,
    /// Semantic names to select — system classifies into dimensions/measures/metrics.
    /// Use `["*"]` to select all columns from the entity.
    #[serde(default)]
    pub select: Vec<String>,
    /// Named filters from the manifest.
    #[serde(default)]
    pub filters: Vec<String>,
    /// Inline filter expressions — anonymous, request-scope predicates that share
    /// the named-filter engine. See `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
    /// Each `RawFilter { field, operator, value }` is translated at request
    /// resolution into a canonical boolean `Expr` and injected at the scan layer.
    #[serde(default)]
    pub raw_filters: Vec<RawFilter>,
    pub grain: Option<String>,
    pub limit: Option<u64>,
    #[serde(default)]
    pub order_by: Vec<RawOrderBy>,
    #[serde(default)]
    pub session: HashMap<String, String>,
    /// Engine to use for plan generation (e.g., "datafusion", "duckdb").
    /// If not set, uses the default engine from the connector.
    #[serde(default)]
    pub engine: Option<String>,
}

/// Convenience alias — same as RawQueryRequest for now.
pub type QueryRequest = RawQueryRequest;

/// An inline filter on the raw query request.
///
/// Translated by `RequestParser::to_resolved` into a canonical boolean `Expr`
/// and carried on the resolved request as a request-scope, anonymous
/// `CompiledFilter`. Rides the same scan-layer injection engine as named
/// DataKind filters per `docs/design/foundations/11_names_and_scopes.md §6.4.2`
/// and `docs/design/foundations/19_expression_flow.md §7.1`.
///
/// - `field`: a semantics name resolved against the entity's interface
///   (Dimension / Measure / Metric / Key per SR-E-11).
/// - `operator`: one of `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`, `like`
///   (plus common symbolic aliases like `=`, `!=`, `<`, etc.).
/// - `value`: a JSON literal that type-checks against the field's `DataType`,
///   or an array of literals for `in`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawFilter {
    pub field: String,
    pub operator: String,
    pub value: serde_json::Value,
}

/// An order-by clause in the raw query request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawOrderBy {
    pub field: String,
    #[serde(default = "default_asc")]
    pub direction: String,
}

fn default_asc() -> String {
    "asc".to_string()
}

/// Result of an explain operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainResult {
    /// SQL string (if SQL emitter was used)
    pub sql: Option<String>,
    /// Human-readable plan tree (indented, similar to DataFusion EXPLAIN)
    pub plan_text: String,
}

/// Result of a validation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}
