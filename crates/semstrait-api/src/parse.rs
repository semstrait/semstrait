//! RequestParser — converts RawQueryRequest → validated/resolved request.

use crate::error::ParseError;
use crate::types::{RawFilter, RawQueryRequest};
use semstrait_core::{DataType, Expr};
use semstrait_manifest::{CompiledFilter, CompiledInterface, CompiledManifest};
use semstrait_planner::request::{
    OrderByClause, ResolvedQueryRequest, SortDirection,
};

/// Parses raw query requests against a compiled manifest.
pub struct RequestParser;

impl RequestParser {
    /// Basic structural validation (no manifest needed).
    ///
    /// Inline `raw_filters` are accepted here but not validated — full
    /// validation (field existence, operator admissibility, value type-check)
    /// happens in `to_resolved` against the live manifest.
    pub fn parse(raw: &RawQueryRequest) -> Result<ValidatedRequest, ParseError> {
        if raw.select.is_empty() {
            return Err(ParseError::Validation(
                "select must contain at least one column name or \"*\"".to_string(),
            ));
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
        // Inline raw filters require an explicit entity to validate against —
        // ad-hoc mode cannot type-check fields, so we reject them up front
        // rather than letting the planner produce confusing errors later.
        let Some(ref from) = raw.from else {
            if !raw.raw_filters.is_empty() {
                return Err(ParseError::RawFiltersRequireEntity);
            }
            return Ok(ResolvedQueryRequest {
                entity_name: String::new(),
                dimensions: raw.select.clone(), // planner will reclassify
                measures: vec![],
                filters: vec![],
                inline_filters: vec![],
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

        // Translate inline raw filters into CompiledFilters. Each becomes a
        // request-scope, anonymous filter with a synthetic `__inline_filter_N`
        // name; they ride the same scan-layer engine as named DataKind filters
        // per `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
        let inline_filters: Vec<CompiledFilter> = raw
            .raw_filters
            .iter()
            .enumerate()
            .map(|(i, rf)| resolve_raw_filter(rf, kind, from, i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ResolvedQueryRequest {
            entity_name: from.clone(),
            dimensions,
            measures,
            filters: vec![],
            inline_filters,
            grain: None,
            limit: raw.limit,
            order_by,
            session_variables: raw.session.clone(),
        })
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

/// Translate a `RawFilter { field, operator, value }` into a request-scope
/// `CompiledFilter`. Validates `field` against the kind interface, maps
/// `operator` to a canonical comparison/set op, and coerces `value` against
/// `field`'s `DataType`. Synthetic name is `__inline_filter_<index>`.
fn resolve_raw_filter(
    raw: &RawFilter,
    kind: &CompiledInterface,
    entity_name: &str,
    index: usize,
) -> Result<CompiledFilter, ParseError> {
    // 1. Resolve field → DataType.
    let field_type = lookup_field_type(&raw.field, kind).ok_or_else(|| {
        ParseError::RawFilterFieldNotFound {
            entity: entity_name.to_string(),
            field: raw.field.clone(),
        }
    })?;

    // 2. Build the field-side Expr. Use EntityRef to match the shape that
    //    the named-filter DSL parser produces — this routes through the
    //    same PhysicalResolver path during scan-layer lowering.
    let field_expr = Expr::entity_ref(raw.field.clone());

    // 3. Build the predicate from operator + value.
    let op_canonical = canonicalize_operator(&raw.operator).ok_or_else(|| {
        ParseError::RawFilterOperatorInvalid {
            field: raw.field.clone(),
            operator: raw.operator.clone(),
        }
    })?;

    let predicate = match op_canonical {
        CanonicalOp::Eq => Expr::eq(field_expr, coerce_value(&raw.value, &field_type, &raw.field)?),
        CanonicalOp::Ne => Expr::ne(field_expr, coerce_value(&raw.value, &field_type, &raw.field)?),
        CanonicalOp::Lt => Expr::lt(field_expr, coerce_value(&raw.value, &field_type, &raw.field)?),
        CanonicalOp::Le => Expr::lte(field_expr, coerce_value(&raw.value, &field_type, &raw.field)?),
        CanonicalOp::Gt => Expr::gt(field_expr, coerce_value(&raw.value, &field_type, &raw.field)?),
        CanonicalOp::Ge => Expr::gte(field_expr, coerce_value(&raw.value, &field_type, &raw.field)?),
        CanonicalOp::In => {
            let items: Vec<Expr> = match &raw.value {
                serde_json::Value::Array(arr) if !arr.is_empty() => arr
                    .iter()
                    .map(|v| coerce_value(v, &field_type, &raw.field))
                    .collect::<Result<Vec<_>, _>>()?,
                serde_json::Value::Array(_) => {
                    return Err(ParseError::RawFilterValueTypeMismatch {
                        field: raw.field.clone(),
                        expected: "non-empty array".to_string(),
                        got: "empty array".to_string(),
                    });
                }
                other => {
                    // Allow single-value shorthand for `in`.
                    vec![coerce_value(other, &field_type, &raw.field)?]
                }
            };
            Expr::in_list(field_expr, items)
        }
        CanonicalOp::Like => {
            let pattern = match &raw.value {
                serde_json::Value::String(s) => Expr::string(s.clone()),
                other => {
                    return Err(ParseError::RawFilterValueTypeMismatch {
                        field: raw.field.clone(),
                        expected: "string pattern".to_string(),
                        got: describe_json_value(other),
                    });
                }
            };
            Expr::like(field_expr, pattern)
        }
    };

    Ok(CompiledFilter {
        name: format!("__inline_filter_{}", index),
        expr: predicate,
        expr_source: format!(
            "{} {} {}",
            raw.field,
            raw.operator,
            json_value_for_source(&raw.value)
        ),
    })
}

/// Look up the `DataType` of a semantic name in the kind's interface.
/// Checks dimensions, measures, then metrics. Keys are dimensions per `§6.5`.
fn lookup_field_type(field: &str, kind: &CompiledInterface) -> Option<DataType> {
    if let Some(d) = kind.dimensions.get(field) {
        return Some(d.data_type.clone());
    }
    if let Some(m) = kind.measures.get(field) {
        return Some(m.data_type.clone());
    }
    if let Some(m) = kind.metrics.get(field) {
        return Some(m.data_type.clone());
    }
    None
}

/// Canonical filter operators admissible on `RawFilter.operator`.
/// See `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
#[derive(Debug, Clone, Copy)]
enum CanonicalOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    Like,
}

/// Map a wire `operator` string to its canonical form. Accepts both name
/// tokens (`eq`, `ne`, ...) and common symbolic aliases (`=`, `!=`, ...).
fn canonicalize_operator(op: &str) -> Option<CanonicalOp> {
    match op.to_lowercase().as_str() {
        "eq" | "=" | "==" => Some(CanonicalOp::Eq),
        "ne" | "!=" | "<>" => Some(CanonicalOp::Ne),
        "lt" | "<" => Some(CanonicalOp::Lt),
        "le" | "lte" | "<=" => Some(CanonicalOp::Le),
        "gt" | ">" => Some(CanonicalOp::Gt),
        "ge" | "gte" | ">=" => Some(CanonicalOp::Ge),
        "in" => Some(CanonicalOp::In),
        "like" => Some(CanonicalOp::Like),
        _ => None,
    }
}

/// Coerce a JSON value into a canonical literal `Expr` matching the field's
/// `DataType`. Strict shape match: Number→numeric, String→string-like, etc.
/// Null is always admissible.
fn coerce_value(
    value: &serde_json::Value,
    field_type: &DataType,
    field: &str,
) -> Result<Expr, ParseError> {
    use serde_json::Value as JV;
    let mismatch = |expected: &str, got: &str| ParseError::RawFilterValueTypeMismatch {
        field: field.to_string(),
        expected: expected.to_string(),
        got: got.to_string(),
    };

    if matches!(value, JV::Null) {
        return Ok(Expr::null());
    }

    match field_type {
        DataType::Integer => match value {
            JV::Number(n) => n
                .as_i64()
                .map(Expr::int)
                .ok_or_else(|| mismatch("integer", &describe_json_value(value))),
            _ => Err(mismatch("integer", &describe_json_value(value))),
        },
        DataType::Number | DataType::Decimal { .. } => match value {
            JV::Number(n) => n
                .as_f64()
                .map(Expr::float)
                .ok_or_else(|| mismatch("number", &describe_json_value(value))),
            _ => Err(mismatch("number", &describe_json_value(value))),
        },
        DataType::String => match value {
            JV::String(s) => Ok(Expr::string(s.clone())),
            _ => Err(mismatch("string", &describe_json_value(value))),
        },
        DataType::Boolean => match value {
            JV::Bool(b) => Ok(Expr::boolean(*b)),
            _ => Err(mismatch("boolean", &describe_json_value(value))),
        },
        // Date / Timestamp / Binary accept ISO-style or base64 strings; engine casts.
        DataType::Date | DataType::Timestamp { .. } | DataType::Binary => match value {
            JV::String(s) => Ok(Expr::string(s.clone())),
            _ => Err(mismatch("string literal", &describe_json_value(value))),
        },
    }
}

fn describe_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(_) => "boolean".to_string(),
        serde_json::Value::Number(_) => "number".to_string(),
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Array(_) => "array".to_string(),
        serde_json::Value::Object(_) => "object".to_string(),
    }
}

fn json_value_for_source(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable>".to_string())
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
    fn test_parse_accepts_raw_filters() {
        // `parse()` is structural and should not reject raw_filters; full
        // validation happens at `to_resolved` time against the manifest.
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
        assert!(result.is_ok(), "parse should accept raw_filters: {:?}", result.err());
    }

    #[test]
    fn test_canonicalize_operator() {
        assert!(matches!(canonicalize_operator("eq"), Some(CanonicalOp::Eq)));
        assert!(matches!(canonicalize_operator("="), Some(CanonicalOp::Eq)));
        assert!(matches!(canonicalize_operator("EQ"), Some(CanonicalOp::Eq)));
        assert!(matches!(canonicalize_operator("ne"), Some(CanonicalOp::Ne)));
        assert!(matches!(canonicalize_operator("!="), Some(CanonicalOp::Ne)));
        assert!(matches!(canonicalize_operator("<>"), Some(CanonicalOp::Ne)));
        assert!(matches!(canonicalize_operator("lt"), Some(CanonicalOp::Lt)));
        assert!(matches!(canonicalize_operator("<"), Some(CanonicalOp::Lt)));
        assert!(matches!(canonicalize_operator("le"), Some(CanonicalOp::Le)));
        assert!(matches!(canonicalize_operator("lte"), Some(CanonicalOp::Le)));
        assert!(matches!(canonicalize_operator("gt"), Some(CanonicalOp::Gt)));
        assert!(matches!(canonicalize_operator("ge"), Some(CanonicalOp::Ge)));
        assert!(matches!(canonicalize_operator("in"), Some(CanonicalOp::In)));
        assert!(matches!(canonicalize_operator("like"), Some(CanonicalOp::Like)));
        assert!(canonicalize_operator("regex").is_none());
        assert!(canonicalize_operator("between").is_none());
    }

    #[test]
    fn test_coerce_value_strict_typing() {
        // String field accepts string only.
        assert!(matches!(
            coerce_value(&serde_json::json!("hi"), &DataType::String, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::json!(42), &DataType::String, "f"),
            Err(ParseError::RawFilterValueTypeMismatch { .. })
        ));

        // Integer field accepts integer only.
        assert!(matches!(
            coerce_value(&serde_json::json!(7), &DataType::Integer, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::json!("7"), &DataType::Integer, "f"),
            Err(ParseError::RawFilterValueTypeMismatch { .. })
        ));

        // Boolean field accepts boolean only.
        assert!(matches!(
            coerce_value(&serde_json::json!(true), &DataType::Boolean, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::json!("true"), &DataType::Boolean, "f"),
            Err(ParseError::RawFilterValueTypeMismatch { .. })
        ));

        // Number field accepts number (i64 and f64).
        assert!(matches!(
            coerce_value(&serde_json::json!(2.5), &DataType::Number, "f"),
            Ok(Expr::Literal(_))
        ));

        // Null is admissible on any type.
        assert!(matches!(
            coerce_value(&serde_json::Value::Null, &DataType::String, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::Value::Null, &DataType::Integer, "f"),
            Ok(Expr::Literal(_))
        ));

        // Date accepts ISO string.
        assert!(matches!(
            coerce_value(&serde_json::json!("2024-01-01"), &DataType::Date, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::json!(20240101), &DataType::Date, "f"),
            Err(ParseError::RawFilterValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn test_resolve_raw_filter_unknown_field() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "nonexistent".to_string(),
            operator: "=".to_string(),
            value: serde_json::json!("X"),
        };
        let err = resolve_raw_filter(&raw, &kind, "sales", 0).unwrap_err();
        assert!(matches!(err, ParseError::RawFilterFieldNotFound { .. }));
    }

    #[test]
    fn test_resolve_raw_filter_invalid_operator() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "region".to_string(),
            operator: "regex".to_string(),
            value: serde_json::json!("US"),
        };
        let err = resolve_raw_filter(&raw, &kind, "sales", 0).unwrap_err();
        assert!(matches!(err, ParseError::RawFilterOperatorInvalid { .. }));
    }

    #[test]
    fn test_resolve_raw_filter_type_mismatch() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "revenue".to_string(),
            operator: ">".to_string(),
            value: serde_json::json!("not_a_number"),
        };
        let err = resolve_raw_filter(&raw, &kind, "sales", 0).unwrap_err();
        assert!(matches!(err, ParseError::RawFilterValueTypeMismatch { .. }));
    }

    #[test]
    fn test_resolve_raw_filter_eq_string() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "region".to_string(),
            operator: "eq".to_string(),
            value: serde_json::json!("US"),
        };
        let cf = resolve_raw_filter(&raw, &kind, "sales", 0).unwrap();
        assert_eq!(cf.name, "__inline_filter_0");
        // Predicate shape: BinaryOp(EntityRef("region"), Eq, String("US")).
        assert!(matches!(&cf.expr, Expr::BinaryOp(_)));
    }

    #[test]
    fn test_resolve_raw_filter_in_list() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "region".to_string(),
            operator: "in".to_string(),
            value: serde_json::json!(["US", "EU", "APAC"]),
        };
        let cf = resolve_raw_filter(&raw, &kind, "sales", 2).unwrap();
        assert_eq!(cf.name, "__inline_filter_2");
        assert!(matches!(&cf.expr, Expr::InList(_)));
    }

    #[test]
    fn test_resolve_raw_filter_in_empty_array_rejected() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "region".to_string(),
            operator: "in".to_string(),
            value: serde_json::json!([]),
        };
        let err = resolve_raw_filter(&raw, &kind, "sales", 0).unwrap_err();
        assert!(matches!(err, ParseError::RawFilterValueTypeMismatch { .. }));
    }

    #[test]
    fn test_resolve_raw_filter_like_requires_string() {
        let kind = make_minimal_interface();
        let raw = RawFilter {
            field: "region".to_string(),
            operator: "like".to_string(),
            value: serde_json::json!(42),
        };
        let err = resolve_raw_filter(&raw, &kind, "sales", 0).unwrap_err();
        assert!(matches!(err, ParseError::RawFilterValueTypeMismatch { .. }));
    }

    // -- helpers ----------------------------------------------------------

    fn make_minimal_interface() -> CompiledInterface {
        use indexmap::IndexMap;
        use semstrait_manifest::{CategoricalDimension, CompiledDimension, CompiledMeasure, DimensionType};

        let mut dimensions = IndexMap::new();
        dimensions.insert(
            "region".to_string(),
            CompiledDimension {
                name: "region".to_string(),
                description: None,
                data_type: DataType::String,
                dim_type: DimensionType::Categorical(CategoricalDimension { enum_values: None }),
                expr: None,
                expr_source: None,
            },
        );

        let mut measures = IndexMap::new();
        measures.insert(
            "revenue".to_string(),
            CompiledMeasure {
                name: "revenue".to_string(),
                description: None,
                data_type: DataType::Number,
                agg: semstrait_core::Aggregation::Sum,
                expr: Expr::entity_ref("amount"),
                expr_source: "amount".to_string(),
                additivity: None,
                constraints: None,
                filters: vec![],
            },
        );

        CompiledInterface {
            name: "sales".to_string(),
            description: None,
            dimensions,
            measures,
            metrics: IndexMap::new(),
            keys: None,
            filters: vec![],
            temporal_dim: None,
        }
    }
}
