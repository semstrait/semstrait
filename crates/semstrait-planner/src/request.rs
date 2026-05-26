//! ResolvedQueryRequest — the planner's input type.
//!
//! Produced by the API layer's RequestParser from a raw QueryRequest.
//! Contains only resolved names (no raw SQL, no unresolved references).

use semstrait_core::Grain;
use semstrait_manifest::CompiledFilter;
use std::collections::HashMap;

/// The resolved query request — input to `SemanticPlanner::plan()`.
#[derive(Debug, Clone)]
pub struct ResolvedQueryRequest {
    /// The entity to query (kind name, or dataset name for implicit kinds).
    /// Empty string triggers ad-hoc resolution from requested fields.
    pub entity_name: String,
    /// Semantic dimension names to include in GROUP BY.
    pub dimensions: Vec<String>,
    /// Semantic measure/metric names to include.
    pub measures: Vec<String>,
    /// User-supplied filter predicates.
    pub filters: Vec<QueryFilter>,
    /// Inline request-time filters (anonymous, request-scope boolean predicates).
    ///
    /// Translated by the API layer from `RawFilter { field, operator, value }`
    /// triples into canonical boolean `Expr`s. These ride the same scan-layer
    /// injection engine as `CompiledInterface.filters` (named DataKind filters),
    /// per `docs/design/foundations/11_names_and_scopes.md §6.4.2` and
    /// `docs/design/foundations/19_expression_flow.md §7.1`.
    ///
    /// Each carries a synthetic `__inline_filter_<N>` name; they are not
    /// addressable by `Request.filters: [name]`.
    pub inline_filters: Vec<CompiledFilter>,
    /// Inline request-time filters carried in unresolved form because the
    /// entity scope wasn't known at parse time (ad-hoc mode, `from` omitted).
    ///
    /// These are lowered to `CompiledFilter`s by the planner after entity
    /// resolution — by `plan_ad_hoc` in the single-entity case, or by
    /// `build_ad_hoc_join_plan` in the multi-entity case (where each filter's
    /// `field` is attributed to its owning `MatchedEntity`). Once lowered they
    /// ride the same scan-layer engine as `inline_filters`.
    ///
    /// Empty in the explicit-`from` path: the parser lowers raw filters
    /// directly into `inline_filters` when the entity is known.
    pub pending_inline_filters: Vec<PendingInlineFilter>,
    /// Temporal grain for date grouping.
    pub grain: Option<Grain>,
    /// Maximum number of rows to return.
    pub limit: Option<u64>,
    /// ORDER BY clauses.
    pub order_by: Vec<OrderByClause>,
    /// Runtime session variables (tenant_id, user_id, etc.).
    pub session_variables: SessionVariables,
}

/// An inline request-time filter carried in unresolved form.
///
/// Planner-side mirror of the API-layer `RawFilter`, used when entity
/// resolution is deferred to the planner (ad-hoc mode). The planner lowers
/// these into `CompiledFilter`s against the resolved interface; see
/// `crate::inline_filter::lower_inline_filter`.
#[derive(Debug, Clone)]
pub struct PendingInlineFilter {
    /// Semantic name to look up on the resolved entity's `CompiledInterface`
    /// (Dimension / Measure / Metric / Key per `§6.5`).
    pub field: String,
    /// Operator wire token: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`, `like`
    /// (plus the symbolic aliases `=`, `!=`, `<>`, `<`, `<=`, `>`, `>=`).
    pub operator: String,
    /// JSON value type-checked against `field`'s `DataType` at lowering time.
    /// Arrays accepted for `in`; null is admissible on any type.
    pub value: serde_json::Value,
}

/// A user-supplied filter predicate.
#[derive(Debug, Clone)]
pub struct QueryFilter {
    /// The dimension or measure name being filtered.
    pub field: String,
    /// The filter operator.
    pub operator: FilterOperator,
    /// The value(s) to compare against.
    pub values: Vec<FilterValue>,
}

/// Filter operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterOperator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    In,
    NotIn,
    Between,
    IsNull,
    IsNotNull,
}

/// A filter value (typed).
#[derive(Debug, Clone)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}

/// ORDER BY clause.
#[derive(Debug, Clone)]
pub struct OrderByClause {
    /// Field name (dimension or measure).
    pub field: String,
    /// Sort direction.
    pub direction: SortDirection,
}

/// Sort direction for ORDER BY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Runtime session variables provided by the API layer.
pub type SessionVariables = HashMap<String, String>;
