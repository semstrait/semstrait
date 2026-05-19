//! RequestParser — converts RawQueryRequest → validated/resolved request.

use crate::error::ParseError;
use crate::types::{RawFilter, RawQueryRequest};
use semstrait_manifest::{CompiledManifest, CompiledInterface};
use semstrait_planner::request::{
    FilterOperator, FilterValue, OrderByClause, QueryFilter, ResolvedQueryRequest,
    SortDirection,
};

/// Parses raw query requests against a compiled manifest.
pub struct RequestParser;

impl RequestParser {
    /// Basic structural validation (no manifest needed).
    ///
    /// Inline `raw_filters` are structurally validated (operator/value shape) but
    /// the cross-reference check against `DataKindFilter` names lives in
    /// `to_resolved()` where the manifest is available.
    pub fn parse(raw: &RawQueryRequest) -> Result<ValidatedRequest, ParseError> {
        if raw.select.is_empty() {
            return Err(ParseError::Validation(
                "select must contain at least one column name or \"*\"".to_string(),
            ));
        }

        // Structural validation of any inline raw_filters (op + value arity).
        for rf in &raw.raw_filters {
            let _ = lower_raw_filter(rf)?;
        }

        Ok(ValidatedRequest {
            entity_name: raw.from.clone(),
            select: raw.select.clone(),
            filters: raw.filters.clone(),
            grain: raw.grain.clone(),
            limit: raw.limit,
        })
    }

    /// Convert a RawQueryRequest into a fully resolved planner request,
    /// validating names against the compiled manifest.
    pub fn to_resolved(
        raw: &RawQueryRequest,
        manifest: &CompiledManifest,
    ) -> Result<ResolvedQueryRequest, ParseError> {
        // Basic validation.
        let _ = Self::parse(raw)?;

        // Convert order_by (entity-independent).
        let order_by = raw
            .order_by
            .iter()
            .map(|ob| OrderByClause {
                field: ob.field.clone(),
                direction: if ob.direction.to_lowercase() == "desc" {
                    SortDirection::Descending
                } else {
                    SortDirection::Ascending
                },
            })
            .collect();

        // If `from` is None, pass through to planner for ad-hoc resolution.
        // Select names are passed as-is — the planner classifies them.
        let Some(ref from) = raw.from else {
            // Without a manifest entity we still lower raw_filters structurally; the
            // cross-reference check against DataKindFilter names is skipped because
            // there is no kind to compare against until ad-hoc resolution succeeds.
            let raw_filters_lowered: Vec<QueryFilter> = raw
                .raw_filters
                .iter()
                .map(lower_raw_filter)
                .collect::<Result<_, _>>()?;

            return Ok(ResolvedQueryRequest {
                entity_name: String::new(),
                dimensions: raw.select.clone(), // planner will reclassify
                measures: vec![],
                filters: raw_filters_lowered,
                grain: None,
                limit: raw.limit,
                order_by,
                session_variables: raw.session.clone(),
            });
        };

        // Resolve entity via CompiledDataKind.
        let data_kind = manifest
            .resolve(from)
            .ok_or_else(|| ParseError::EntityNotFound(from.clone()))?;
        let kind = data_kind.interface();

        // Expand "*" and classify select names.
        let select_names = expand_select(&raw.select, kind);
        let (dimensions, measures) = classify_select(&select_names, kind, from)?;

        // Resolve named filters against kind-level filters.
        for filter_name in &raw.filters {
            if !kind.filters.iter().any(|f| f.name == *filter_name) {
                return Err(ParseError::FilterNotFound {
                    entity: from.clone(),
                    name: filter_name.clone(),
                });
            }
        }

        // Lower raw_filters into QueryFilter triples. Cross-reference invariant:
        // a raw_filter MUST NOT name a DataKindFilter declared on the kind —
        // those belong on `RawQueryRequest.filters` instead.
        let mut filters: Vec<QueryFilter> = Vec::with_capacity(raw.raw_filters.len());
        for rf in &raw.raw_filters {
            if kind.filters.iter().any(|f| f.name == rf.field) {
                return Err(ParseError::RawFilterNamesNamedFilter {
                    entity: from.clone(),
                    name: rf.field.clone(),
                });
            }
            filters.push(lower_raw_filter(rf)?);
        }

        Ok(ResolvedQueryRequest {
            entity_name: from.clone(),
            dimensions,
            measures,
            filters,
            grain: None,
            limit: raw.limit,
            order_by,
            session_variables: raw.session.clone(),
        })
    }
}

fn lower_raw_filter(rf: &RawFilter) -> Result<QueryFilter, ParseError> {
    let operator = parse_operator(&rf.operator)?;
    let values = lower_value(&rf.field, &operator, &rf.value)?;
    Ok(QueryFilter {
        field: rf.field.clone(),
        operator,
        values,
    })
}


fn parse_operator(s: &str) -> Result<FilterOperator, ParseError> {
    let normalised = s.trim().to_ascii_lowercase();
    let op = match normalised.as_str() {
        "=" | "==" | "eq" => FilterOperator::Eq,
        "!=" | "<>" | "ne" | "neq" | "not_eq" | "noteq" => FilterOperator::NotEq,
        "<" | "lt" => FilterOperator::Lt,
        "<=" | "le" | "lte" => FilterOperator::LtEq,
        ">" | "gt" => FilterOperator::Gt,
        ">=" | "ge" | "gte" => FilterOperator::GtEq,
        "in" => FilterOperator::In,
        "not_in" | "notin" | "nin" => FilterOperator::NotIn,
        "between" => FilterOperator::Between,
        "is_null" | "isnull" | "null" => FilterOperator::IsNull,
        "is_not_null" | "isnotnull" | "not_null" => FilterOperator::IsNotNull,
        _ => {
            return Err(ParseError::RawFilterInvalidOperator {
                operator: s.to_string(),
            });
        }
    };
    Ok(op)
}

fn lower_value(
    field: &str,
    operator: &FilterOperator,
    value: &serde_json::Value,
) -> Result<Vec<FilterValue>, ParseError> {
    match operator {
        FilterOperator::IsNull | FilterOperator::IsNotNull => {
            Ok(Vec::new())
        }
        FilterOperator::Between => {
            let arr = value.as_array().ok_or_else(|| ParseError::RawFilterInvalidValue {
                field: field.to_string(),
                message: "BETWEEN requires a 2-element array [lo, hi]".to_string(),
            })?;
            if arr.len() != 2 {
                return Err(ParseError::RawFilterInvalidValue {
                    field: field.to_string(),
                    message: format!(
                        "BETWEEN requires exactly 2 values, got {}",
                        arr.len()
                    ),
                });
            }
            arr.iter()
                .map(|v| json_to_filter_value(field, v))
                .collect()
        }
        FilterOperator::In | FilterOperator::NotIn => {
            let arr = value.as_array().ok_or_else(|| ParseError::RawFilterInvalidValue {
                field: field.to_string(),
                message: "IN / NOT IN require an array of values".to_string(),
            })?;
            if arr.is_empty() {
                return Err(ParseError::RawFilterInvalidValue {
                    field: field.to_string(),
                    message: "IN / NOT IN require at least one value".to_string(),
                });
            }
            arr.iter()
                .map(|v| json_to_filter_value(field, v))
                .collect()
        }
        FilterOperator::Eq
        | FilterOperator::NotEq
        | FilterOperator::Lt
        | FilterOperator::LtEq
        | FilterOperator::Gt
        | FilterOperator::GtEq => {
            // Accept either a scalar or a single-element array for convenience.
            if let Some(arr) = value.as_array() {
                if arr.len() != 1 {
                    return Err(ParseError::RawFilterInvalidValue {
                        field: field.to_string(),
                        message: format!(
                            "{:?} requires exactly 1 value, got {}",
                            operator,
                            arr.len()
                        ),
                    });
                }
                Ok(vec![json_to_filter_value(field, &arr[0])?])
            } else {
                Ok(vec![json_to_filter_value(field, value)?])
            }
        }
    }
}

/// Convert a JSON scalar into a typed `FilterValue`.
fn json_to_filter_value(
    field: &str,
    value: &serde_json::Value,
) -> Result<FilterValue, ParseError> {
    match value {
        serde_json::Value::String(s) => Ok(FilterValue::String(s.clone())),
        serde_json::Value::Bool(b) => Ok(FilterValue::Bool(*b)),
        serde_json::Value::Number(n) => n.as_f64().map(FilterValue::Number).ok_or_else(|| {
            ParseError::RawFilterInvalidValue {
                field: field.to_string(),
                message: format!("number '{}' is not representable as f64", n),
            }
        }),
        serde_json::Value::Null => Ok(FilterValue::Null),
        other => Err(ParseError::RawFilterInvalidValue {
            field: field.to_string(),
            message: format!("unsupported value shape: {}", other),
        }),
    }
}

/// Expand `["*"]` into all dimension + measure + metric + key names from the entity.
/// If no `*` is present, returns the select list as-is.
fn expand_select(select: &[String], kind: &CompiledInterface) -> Vec<String> {
    if select.len() == 1 && select[0] == "*" {
        let mut names: Vec<String> = Vec::new();
        names.extend(kind.dimensions.keys().cloned());
        // Include key columns not already in dimensions.
        if let Some(ref keys) = kind.keys {
            for key_col in keys.all_column_names() {
                if !kind.dimensions.contains_key(&key_col) {
                    names.push(key_col);
                }
            }
        }
        names.extend(kind.measures.keys().cloned());
        names.extend(kind.metrics.keys().cloned());
        names
    } else {
        select.to_vec()
    }
}

/// Classify select names into dimensions and measures/metrics.
/// Returns `(dimensions, measures)` where measures includes both measures and metrics.
fn classify_select(
    names: &[String],
    kind: &CompiledInterface,
    entity_name: &str,
) -> Result<(Vec<String>, Vec<String>), ParseError> {
    let mut dimensions = Vec::new();
    let mut measures = Vec::new();

    // Collect key column names for classification.
    let key_names: std::collections::HashSet<String> = kind
        .keys
        .as_ref()
        .map(|k| k.all_column_names().into_iter().collect())
        .unwrap_or_default();

    for name in names {
        if kind.dimensions.contains_key(name) {
            dimensions.push(name.clone());
        } else if key_names.contains(name) {
            // Keys contribute to GROUP BY — classify as dimension.
            dimensions.push(name.clone());
        } else if kind.measures.contains_key(name) || kind.metrics.contains_key(name) {
            measures.push(name.clone());
        } else {
            return Err(ParseError::UnknownSelectName {
                entity: entity_name.to_string(),
                name: name.clone(),
            });
        }
    }

    Ok((dimensions, measures))
}

/// A validated query request (structural validation only, no manifest).
#[derive(Debug, Clone)]
pub struct ValidatedRequest {
    pub entity_name: Option<String>,
    pub select: Vec<String>,
    pub filters: Vec<String>,
    pub grain: Option<String>,
    pub limit: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_request() {
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["region".to_string(), "revenue".to_string()],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.entity_name, Some("sales".to_string()));
        assert_eq!(req.select, vec!["region", "revenue"]);
    }

    #[test]
    fn test_parse_star_select() {
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["*".to_string()],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_no_from() {
        let raw = RawQueryRequest {
            from: None,
            select: vec!["revenue".to_string()],
            ..Default::default()
        };

        // None passes structural validation — entity resolution happens in planner.
        let result = RequestParser::parse(&raw);
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.entity_name, None);
    }

    #[test]
    fn test_parse_empty_select() {
        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(matches!(result, Err(ParseError::Validation(_))));
    }

    #[test]
    fn test_raw_filters_accepted_structurally() {
        use crate::types::RawFilter;

        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["revenue".to_string()],
            raw_filters: vec![RawFilter {
                field: "region".to_string(),
                operator: "=".to_string(),
                value: serde_json::json!("US"),
            }],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(
            result.is_ok(),
            "parse() should accept structurally-valid raw_filters: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_raw_filter_unknown_operator() {
        use crate::types::RawFilter;

        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["revenue".to_string()],
            raw_filters: vec![RawFilter {
                field: "region".to_string(),
                operator: "bogus".to_string(),
                value: serde_json::json!("US"),
            }],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(matches!(
            result,
            Err(ParseError::RawFilterInvalidOperator { .. })
        ));
    }

    #[test]
    fn test_raw_filter_between_arity() {
        use crate::types::RawFilter;

        let raw = RawQueryRequest {
            from: Some("sales".to_string()),
            select: vec!["revenue".to_string()],
            raw_filters: vec![RawFilter {
                field: "revenue".to_string(),
                operator: "between".to_string(),
                value: serde_json::json!([1, 2, 3]),
            }],
            ..Default::default()
        };

        let result = RequestParser::parse(&raw);
        assert!(matches!(
            result,
            Err(ParseError::RawFilterInvalidValue { .. })
        ));
    }

    #[test]
    fn test_lower_raw_filter_eq() {
        use crate::types::RawFilter;

        let rf = RawFilter {
            field: "region".to_string(),
            operator: "=".to_string(),
            value: serde_json::json!("US"),
        };
        let q = lower_raw_filter(&rf).expect("lowering should succeed");
        assert_eq!(q.field, "region");
        assert_eq!(q.operator, FilterOperator::Eq);
        assert_eq!(q.values.len(), 1);
        match &q.values[0] {
            FilterValue::String(s) => assert_eq!(s, "US"),
            other => panic!("expected String value, got {:?}", other),
        }
    }

    #[test]
    fn test_lower_raw_filter_in() {
        use crate::types::RawFilter;

        let rf = RawFilter {
            field: "region".to_string(),
            operator: "in".to_string(),
            value: serde_json::json!(["US", "EU"]),
        };
        let q = lower_raw_filter(&rf).expect("lowering should succeed");
        assert_eq!(q.operator, FilterOperator::In);
        assert_eq!(q.values.len(), 2);
    }

    #[test]
    fn test_lower_raw_filter_is_null() {
        use crate::types::RawFilter;

        let rf = RawFilter {
            field: "region".to_string(),
            operator: "is_null".to_string(),
            value: serde_json::Value::Null,
        };
        let q = lower_raw_filter(&rf).expect("lowering should succeed");
        assert_eq!(q.operator, FilterOperator::IsNull);
        assert!(q.values.is_empty());
    }
}
