use super::error::ParseError;
use super::literal::{self, LIT};
use super::{PhysicalExpr, Unresolved};
use crate::model::types::entities::mapping::SemanticMappingValue;
use semstrait_sql::expr::UntypedLiteral;
use serde_yaml::Value;

pub(crate) const EXPR_KEY: &str = "expr";

pub fn deserialize_mapping_value(value: &Value) -> Result<SemanticMappingValue, ParseError> {
    match value {
        Value::String(s) if s.is_empty() => Err(ParseError::InvalidValue {
            field: "semantic_mapping",
            reason: r#"a column name cannot be empty; the empty string is written `{lit: ""}`"#
                .into(),
        }),
        Value::String(s) => Ok(SemanticMappingValue::Column(s.clone())),
        Value::Mapping(map) => {
            let lit = map.get(Value::String(LIT.into()));
            let expr = map.get(Value::String(EXPR_KEY.into()));
            match (lit, expr, map.len()) {
                (Some(body), None, 1) => {
                    literal::lit(body, SemanticMappingValue::Literal, |literal| {
                        SemanticMappingValue::Untyped(UntypedLiteral(literal))
                    })
                }
                (None, Some(body), 1) => Unresolved::<PhysicalExpr>::deserialize(body)
                    .map(SemanticMappingValue::Expr)
                    .map_err(|e| ParseError::InvalidMappingValue(format!("`expr:` body: {e}"))),
                _ => Err(ParseError::InvalidMappingValue(
                    "expected bare column string or `{lit: ...}` / `{expr: ...}` map".into(),
                )),
            }
        }
        other => Err(ParseError::InvalidMappingValue(format!(
            "unexpected YAML shape {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::physical_col;
    use semstrait_sql::expr::{Expr, Literal, PhysicalLeaf, UntypedLiteral};

    fn yaml(s: &str) -> Value {
        serde_yaml::from_str(s).expect("valid YAML")
    }

    #[test]
    fn bare_string_is_column() {
        let v = yaml("amount_cents");
        let out = deserialize_mapping_value(&v).expect("parse ok");
        assert_eq!(out, SemanticMappingValue::Column("amount_cents".into()));
    }

    #[test]
    fn expr_block_function_call() {
        let v = yaml(r#"{ expr: { upper: { col: amount } } }"#);
        let out = deserialize_mapping_value(&v).expect("parse ok");
        assert_eq!(
            out,
            SemanticMappingValue::Expr(super::super::test_call(
                "upper",
                vec![physical_col("amount").expect("a name")]
            ))
        );
    }

    #[test]
    fn expr_bare_column_via_col_tag() {
        let v = yaml("{ expr: { col: amount } }");
        let out = deserialize_mapping_value(&v).expect("parse ok");
        assert_eq!(
            out,
            SemanticMappingValue::Expr(physical_col("amount").expect("a name"))
        );
    }

    #[test]
    fn expr_lit_inside_expr_arm() {
        let v = yaml("{ expr: { lit: 42 } }");
        let out = deserialize_mapping_value(&v).expect("parse ok");
        match out {
            SemanticMappingValue::Expr(Expr::Leaf(PhysicalLeaf::Untyped(UntypedLiteral(
                Literal::BigInt(42),
            )))) => {}
            other => panic!("expected lit leaf, got {other:?}"),
        }
    }

    #[test]
    fn expr_rejects_semantic_tag_at_physical_site() {
        let v = yaml("{ expr: { field: revenue } }");
        let err = deserialize_mapping_value(&v).expect_err("should reject");
        match err {
            ParseError::InvalidMappingValue(msg) => {
                assert!(
                    msg.contains("physical-mapping") || msg.contains("field"),
                    "expected site-rejection message, got: {msg}"
                );
            }
            other => panic!("expected InvalidMappingValue, got {other:?}"),
        }
    }

    #[test]
    fn unknown_top_key_rejects() {
        let v = yaml("{ wat: 1 }");
        let err = deserialize_mapping_value(&v).expect_err("should reject");
        assert!(matches!(err, ParseError::InvalidMappingValue(_)));
    }

    #[test]
    fn both_lit_and_expr_in_one_arm_rejects() {
        let v = yaml("{ lit: 0, expr: { col: x } }");
        let err = deserialize_mapping_value(&v).expect_err("should reject");
        assert!(matches!(err, ParseError::InvalidMappingValue(_)));
    }

    #[test]
    fn legacy_literal_keyword_rejects() {
        let v = yaml("{ literal: { string: USD } }");
        let err = deserialize_mapping_value(&v).expect_err("should reject");
        assert!(matches!(err, ParseError::InvalidMappingValue(_)));
    }

    #[test]
    fn sequence_at_top_level_rejects() {
        let v = yaml("[ 1, 2 ]");
        let err = deserialize_mapping_value(&v).expect_err("should reject");
        assert!(matches!(err, ParseError::InvalidMappingValue(_)));
    }

    #[test]
    fn round_trip_literal_uses_lit_keyword() {
        let v = SemanticMappingValue::Literal(Literal::String("Paid Search".into()));
        let yaml_str = serde_yaml::to_string(&v).expect("serialise ok");
        assert!(
            yaml_str.contains("lit:"),
            "expected `lit:` in output, got: {yaml_str}"
        );
        assert!(
            !yaml_str.contains("literal:"),
            "should not emit legacy `literal:`, got: {yaml_str}"
        );
        let parsed: SemanticMappingValue = serde_yaml::from_str(&yaml_str).expect("round-trip ok");
        assert_eq!(parsed, v);
    }

    #[test]
    fn round_trip_column_is_bare_scalar() {
        let v = SemanticMappingValue::Column("amount_cents".into());
        let yaml_str = serde_yaml::to_string(&v).expect("serialise ok");
        let parsed: SemanticMappingValue = serde_yaml::from_str(&yaml_str).expect("round-trip ok");
        assert_eq!(parsed, v);
    }
}
