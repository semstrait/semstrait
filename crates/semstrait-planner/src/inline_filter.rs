//! Inline raw-filter lowering — translates `PendingInlineFilter` (wire-shape
//! `{ field, operator, value }` triple) into a typed `CompiledFilter` that
//! rides the same scan-layer injection engine as named `CompiledInterface.filters`.
//!
//! Per `docs/design/foundations/11_names_and_scopes.md §6.4.2` and
//! `docs/design/foundations/19_expression_flow.md §7.1`, an inline filter is
//! a request-scope, anonymous boolean predicate carried on the `Request` that
//! is normalised into a canonical boolean `Expr` and injected at the scan
//! layer alongside DataKind filters — making the two indistinguishable
//! downstream.
//!
//! Used in two places:
//! 1. `semstrait-api/parse.rs` (explicit-`from`): lowers immediately against
//!    the named entity's `CompiledInterface`, mapping errors through
//!    `From<InlineFilterError> for ParseError`.
//! 2. `semstrait-planner::planner::plan_ad_hoc` (ad-hoc): defers lowering
//!    until `find_covering_entities` resolves the target scope, then lowers
//!    against the resolved interface(s).

use semstrait_core::{DataType, Expr};
use semstrait_manifest::{CompiledFilter, CompiledInterface};
use thiserror::Error;

use crate::request::PendingInlineFilter;

/// Errors emitted by `lower_inline_filter`.
///
/// The API layer converts these into `ParseError::RawFilter*` variants via
/// `From<InlineFilterError>`; the planner wraps them in
/// `PlannerError::InlineFilterResolution`.
#[derive(Debug, Error)]
pub enum InlineFilterError {
    /// `field` does not name any Dimension / Measure / Metric / Key on the
    /// resolved entity's `CompiledInterface`.
    #[error("inline filter field not found: {field} in entity {entity}")]
    FieldNotFound { entity: String, field: String },

    /// `operator` is outside the canonical v1 set
    /// (`eq, ne, lt, le, gt, ge, in, like` plus symbolic aliases).
    #[error("invalid inline filter operator '{operator}' for field {field}")]
    OperatorInvalid { field: String, operator: String },

    /// `value` failed strict type-check against `field`'s `DataType`.
    #[error("inline filter value type mismatch on field {field}: expected {expected}, got {got}")]
    ValueTypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    /// (Multi-entity ad-hoc only) `field` is not present on any of the
    /// entities chosen by `find_covering_entities`. The user picked a name
    /// that none of the covering scans can satisfy.
    #[error(
        "inline filter field '{field}' not found on any covering entity: [{}]",
        candidates.join(", ")
    )]
    FieldOnNoEntity {
        field: String,
        candidates: Vec<String>,
    },
}

/// Translate a `PendingInlineFilter { field, operator, value }` into a
/// request-scope `CompiledFilter` against `iface`. Validates `field` against
/// the kind interface, maps `operator` to a canonical comparison / set op,
/// and coerces `value` against `field`'s `DataType`. Synthetic name is
/// `__inline_filter_<index>`.
pub fn lower_inline_filter(
    pending: &PendingInlineFilter,
    iface: &CompiledInterface,
    entity_name: &str,
    index: usize,
) -> Result<CompiledFilter, InlineFilterError> {
    let field_type =
        lookup_field_type(&pending.field, iface).ok_or_else(|| InlineFilterError::FieldNotFound {
            entity: entity_name.to_string(),
            field: pending.field.clone(),
        })?;

    // Field-side Expr uses EntityRef to route through PhysicalResolver during
    // scan-layer lowering — same shape as the named-filter DSL parser.
    let field_expr = Expr::entity_ref(pending.field.clone());

    let op_canonical =
        canonicalize_operator(&pending.operator).ok_or_else(|| InlineFilterError::OperatorInvalid {
            field: pending.field.clone(),
            operator: pending.operator.clone(),
        })?;

    let predicate = match op_canonical {
        CanonicalOp::Eq => Expr::eq(
            field_expr,
            coerce_value(&pending.value, &field_type, &pending.field)?,
        ),
        CanonicalOp::Ne => Expr::ne(
            field_expr,
            coerce_value(&pending.value, &field_type, &pending.field)?,
        ),
        CanonicalOp::Lt => Expr::lt(
            field_expr,
            coerce_value(&pending.value, &field_type, &pending.field)?,
        ),
        CanonicalOp::Le => Expr::lte(
            field_expr,
            coerce_value(&pending.value, &field_type, &pending.field)?,
        ),
        CanonicalOp::Gt => Expr::gt(
            field_expr,
            coerce_value(&pending.value, &field_type, &pending.field)?,
        ),
        CanonicalOp::Ge => Expr::gte(
            field_expr,
            coerce_value(&pending.value, &field_type, &pending.field)?,
        ),
        CanonicalOp::In => {
            let items: Vec<Expr> = match &pending.value {
                serde_json::Value::Array(arr) if !arr.is_empty() => arr
                    .iter()
                    .map(|v| coerce_value(v, &field_type, &pending.field))
                    .collect::<Result<Vec<_>, _>>()?,
                serde_json::Value::Array(_) => {
                    return Err(InlineFilterError::ValueTypeMismatch {
                        field: pending.field.clone(),
                        expected: "non-empty array".to_string(),
                        got: "empty array".to_string(),
                    });
                }
                other => {
                    // Single-value shorthand for `in`.
                    vec![coerce_value(other, &field_type, &pending.field)?]
                }
            };
            Expr::in_list(field_expr, items)
        }
        CanonicalOp::Like => {
            let pattern = match &pending.value {
                serde_json::Value::String(s) => Expr::string(s.clone()),
                other => {
                    return Err(InlineFilterError::ValueTypeMismatch {
                        field: pending.field.clone(),
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
            pending.field,
            pending.operator,
            json_value_for_source(&pending.value)
        ),
    })
}

/// Look up the `DataType` of a semantic name in the kind's interface.
/// Checks dimensions, measures, then metrics. Keys are dimensions per `§6.5`.
pub fn lookup_field_type(field: &str, iface: &CompiledInterface) -> Option<DataType> {
    if let Some(d) = iface.dimensions.get(field) {
        return Some(d.data_type.clone());
    }
    if let Some(m) = iface.measures.get(field) {
        return Some(m.data_type.clone());
    }
    if let Some(m) = iface.metrics.get(field) {
        return Some(m.data_type.clone());
    }
    None
}

/// Canonical filter operators admissible on inline-filter `operator`.
/// See `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
#[derive(Debug, Clone, Copy)]
pub enum CanonicalOp {
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
pub fn canonicalize_operator(op: &str) -> Option<CanonicalOp> {
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
pub fn coerce_value(
    value: &serde_json::Value,
    field_type: &DataType,
    field: &str,
) -> Result<Expr, InlineFilterError> {
    use serde_json::Value as JV;
    let mismatch = |expected: &str, got: &str| InlineFilterError::ValueTypeMismatch {
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

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use semstrait_manifest::{
        CategoricalDimension, CompiledDimension, CompiledMeasure, DimensionType,
    };

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
        assert!(matches!(
            canonicalize_operator("lte"),
            Some(CanonicalOp::Le)
        ));
        assert!(matches!(canonicalize_operator("<="), Some(CanonicalOp::Le)));
        assert!(matches!(canonicalize_operator("gt"), Some(CanonicalOp::Gt)));
        assert!(matches!(canonicalize_operator(">"), Some(CanonicalOp::Gt)));
        assert!(matches!(canonicalize_operator("ge"), Some(CanonicalOp::Ge)));
        assert!(matches!(
            canonicalize_operator("gte"),
            Some(CanonicalOp::Ge)
        ));
        assert!(matches!(canonicalize_operator(">="), Some(CanonicalOp::Ge)));
        assert!(matches!(canonicalize_operator("in"), Some(CanonicalOp::In)));
        assert!(matches!(
            canonicalize_operator("like"),
            Some(CanonicalOp::Like)
        ));
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
            Err(InlineFilterError::ValueTypeMismatch { .. })
        ));

        // Integer field accepts integer only.
        assert!(matches!(
            coerce_value(&serde_json::json!(7), &DataType::Integer, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::json!("7"), &DataType::Integer, "f"),
            Err(InlineFilterError::ValueTypeMismatch { .. })
        ));

        // Boolean field accepts boolean only.
        assert!(matches!(
            coerce_value(&serde_json::json!(true), &DataType::Boolean, "f"),
            Ok(Expr::Literal(_))
        ));
        assert!(matches!(
            coerce_value(&serde_json::json!("true"), &DataType::Boolean, "f"),
            Err(InlineFilterError::ValueTypeMismatch { .. })
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
            Err(InlineFilterError::ValueTypeMismatch { .. })
        ));
    }

    #[test]
    fn test_lower_inline_filter_unknown_field() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "nonexistent".to_string(),
            operator: "=".to_string(),
            value: serde_json::json!("X"),
        };
        let err = lower_inline_filter(&pending, &iface, "sales", 0).unwrap_err();
        assert!(matches!(err, InlineFilterError::FieldNotFound { .. }));
    }

    #[test]
    fn test_lower_inline_filter_invalid_operator() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "region".to_string(),
            operator: "regex".to_string(),
            value: serde_json::json!("US"),
        };
        let err = lower_inline_filter(&pending, &iface, "sales", 0).unwrap_err();
        assert!(matches!(err, InlineFilterError::OperatorInvalid { .. }));
    }

    #[test]
    fn test_lower_inline_filter_type_mismatch() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "revenue".to_string(),
            operator: ">".to_string(),
            value: serde_json::json!("not_a_number"),
        };
        let err = lower_inline_filter(&pending, &iface, "sales", 0).unwrap_err();
        assert!(matches!(err, InlineFilterError::ValueTypeMismatch { .. }));
    }

    #[test]
    fn test_lower_inline_filter_eq_string() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "region".to_string(),
            operator: "eq".to_string(),
            value: serde_json::json!("US"),
        };
        let cf = lower_inline_filter(&pending, &iface, "sales", 0).unwrap();
        assert_eq!(cf.name, "__inline_filter_0");
        assert!(matches!(&cf.expr, Expr::BinaryOp(_)));
    }

    #[test]
    fn test_lower_inline_filter_in_list() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "region".to_string(),
            operator: "in".to_string(),
            value: serde_json::json!(["US", "EU", "APAC"]),
        };
        let cf = lower_inline_filter(&pending, &iface, "sales", 2).unwrap();
        assert_eq!(cf.name, "__inline_filter_2");
        assert!(matches!(&cf.expr, Expr::InList(_)));
    }

    #[test]
    fn test_lower_inline_filter_in_empty_array_rejected() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "region".to_string(),
            operator: "in".to_string(),
            value: serde_json::json!([]),
        };
        let err = lower_inline_filter(&pending, &iface, "sales", 0).unwrap_err();
        assert!(matches!(err, InlineFilterError::ValueTypeMismatch { .. }));
    }

    #[test]
    fn test_lower_inline_filter_like_requires_string() {
        let iface = make_minimal_interface();
        let pending = PendingInlineFilter {
            field: "region".to_string(),
            operator: "like".to_string(),
            value: serde_json::json!(42),
        };
        let err = lower_inline_filter(&pending, &iface, "sales", 0).unwrap_err();
        assert!(matches!(err, InlineFilterError::ValueTypeMismatch { .. }));
    }

    fn make_minimal_interface() -> CompiledInterface {
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
