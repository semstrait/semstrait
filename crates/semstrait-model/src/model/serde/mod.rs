pub(crate) mod data_type;
pub mod de;
pub(crate) mod digits;
pub mod error;
pub mod leaf;
pub(crate) mod literal;
pub mod mapping;
pub mod ser;
pub mod token;
mod unresolved;
pub(crate) mod value;

pub use de::parse_expr;
pub use error::ParseError;
pub use leaf::LeafResolver;
pub use mapping::deserialize_mapping_value;
pub use token::{tokenize_leaf, LeafToken};
#[cfg(feature = "schemars")]
pub(crate) use unresolved::object;
pub use unresolved::{PhysicalExpr, SemanticExpr, Unresolved};

#[cfg(test)]
pub(crate) fn test_call<L: semstrait_sql::expr::Leaf>(
    name: &str,
    args: Vec<semstrait_sql::expr::RowExpr<L>>,
) -> semstrait_sql::expr::RowExpr<L> {
    use semstrait_sql::expr::{Call, Expr, FunctionRef};
    use semstrait_sql::functions::registry;

    let function = registry().get(name).expect("a catalog function");
    let function = FunctionRef::try_from(function).expect("no aggregate function");
    Expr::Function(Call::new(function, args).expect("a call that fits its declaration"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{col, dim, field, measure, metric, physical_col};
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{
        BinaryOpKind, CastFailure, DimensionAccessor, Expr, LikeKind, Literal, MeasureAccessor,
        MetricAccessor, NamedReference, PhysicalLeaf, RowExpr, SemanticLeaf, UnaryOpKind,
    };
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct SemCarrier {
        #[serde(with = "Unresolved::<SemanticExpr>")]
        expr: RowExpr<SemanticLeaf>,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct PhysCarrier {
        #[serde(with = "Unresolved::<PhysicalExpr>")]
        expr: RowExpr<PhysicalLeaf>,
    }

    fn slit(lit: Literal) -> RowExpr<SemanticLeaf> {
        Expr::Literal(lit)
    }
    fn sint(v: i64) -> RowExpr<SemanticLeaf> {
        slit(Literal::BigInt(v))
    }

    #[track_caller]
    fn rt_sem(expr: RowExpr<SemanticLeaf>) {
        let c = SemCarrier { expr };
        let yaml = serde_yaml::to_string(&c).expect("serialize");
        assert!(!yaml.contains('!'), "no !-tags:\n{yaml}");
        for machine in ["type:", "content:", "binary_op:", "aggregate:"] {
            assert!(
                !yaml.contains(machine),
                "machine tag {machine:?} leaked into authoring YAML:\n{yaml}"
            );
        }
        let back: SemCarrier = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(c, back, "round-trip mismatch via:\n{yaml}");
    }

    #[track_caller]
    fn rt_phys(expr: RowExpr<PhysicalLeaf>) {
        let c = PhysCarrier { expr };
        let yaml = serde_yaml::to_string(&c).expect("serialize");
        assert!(!yaml.contains('!'), "no !-tags:\n{yaml}");
        for machine in ["type:", "content:", "binary_op:"] {
            assert!(
                !yaml.contains(machine),
                "machine tag {machine:?} leaked into authoring YAML:\n{yaml}"
            );
        }
        let back: PhysCarrier = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(c, back, "round-trip mismatch via:\n{yaml}");
    }

    #[test]
    fn round_trips_all_semantic_leaves() {
        let cases: Vec<RowExpr<SemanticLeaf>> = vec![
            field("revenue").expect("a name"),
            col("amount").expect("a name"),
            dim("region").expect("a name"),
            Expr::Leaf(SemanticLeaf::Dimension(
                NamedReference::unqualified("region").expect("a name"),
                Some(DimensionAccessor::First),
            )),
            Expr::Leaf(SemanticLeaf::Dimension(
                NamedReference::unqualified("region").expect("a name"),
                Some(DimensionAccessor::Last),
            )),
            Expr::Leaf(SemanticLeaf::Dimension(
                NamedReference::unqualified("region").expect("a name"),
                Some(DimensionAccessor::Lag(2)),
            )),
            Expr::Leaf(SemanticLeaf::Dimension(
                NamedReference::unqualified("region").expect("a name"),
                Some(DimensionAccessor::Lead(3)),
            )),
            measure("revenue").expect("a name"),
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::Previous),
            )),
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::Next),
            )),
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::Lag(2)),
            )),
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::Lead(3)),
            )),
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::Delta),
            )),
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::PercentChange),
            )),
            metric("conv").expect("a name"),
            Expr::Leaf(SemanticLeaf::Metric(
                NamedReference::unqualified("conv").expect("a name"),
                Some(MetricAccessor::Previous),
            )),
            Expr::Leaf(SemanticLeaf::Metric(
                NamedReference::unqualified("conv").expect("a name"),
                Some(MetricAccessor::Next),
            )),
            Expr::Leaf(SemanticLeaf::Metric(
                NamedReference::unqualified("conv").expect("a name"),
                Some(MetricAccessor::Lag(1)),
            )),
            Expr::Leaf(SemanticLeaf::Metric(
                NamedReference::unqualified("conv").expect("a name"),
                Some(MetricAccessor::Lead(2)),
            )),
            Expr::Leaf(SemanticLeaf::Metric(
                NamedReference::unqualified("conv").expect("a name"),
                Some(MetricAccessor::Delta),
            )),
            Expr::Leaf(SemanticLeaf::Metric(
                NamedReference::unqualified("conv").expect("a name"),
                Some(MetricAccessor::PercentChange),
            )),
        ];
        for e in cases {
            rt_sem(e);
        }
    }

    #[test]
    fn round_trips_all_structural_variants() {
        let mut cases: Vec<RowExpr<SemanticLeaf>> = Vec::new();

        for op in [
            BinaryOpKind::Add,
            BinaryOpKind::Subtract,
            BinaryOpKind::Multiply,
            BinaryOpKind::Divide,
            BinaryOpKind::SafeDivide,
            BinaryOpKind::Mod,
            BinaryOpKind::Eq,
            BinaryOpKind::NotEq,
            BinaryOpKind::Lt,
            BinaryOpKind::LtEq,
            BinaryOpKind::Gt,
            BinaryOpKind::GtEq,
            BinaryOpKind::IsNotDistinctFrom,
            BinaryOpKind::And,
            BinaryOpKind::Or,
        ] {
            cases.push(Expr::BinaryOp {
                op,
                left: Box::new(field("a").expect("a name")),
                right: Box::new(field("b").expect("a name")),
            });
        }

        cases.push(Expr::UnaryOp {
            op: UnaryOpKind::Negate,
            operand: Box::new(sint(5)),
        });
        cases.push(Expr::UnaryOp {
            op: UnaryOpKind::Not,
            operand: Box::new(Expr::BinaryOp {
                op: BinaryOpKind::Eq,
                left: Box::new(field("a").expect("a name")),
                right: Box::new(field("b").expect("a name")),
            }),
        });

        cases.push(test_call("upper", vec![field("name").expect("a name")]));
        cases.push(test_call(
            "concat",
            vec![
                field("a").expect("a name"),
                field("b").expect("a name"),
                field("c").expect("a name"),
            ],
        ));
        cases.push(test_call(
            "coalesce",
            vec![field("a").expect("a name"), sint(0)],
        ));
        cases.push(test_call(
            "nullif",
            vec![field("a").expect("a name"), sint(0)],
        ));

        cases.push(Expr::Cast {
            input: Box::new(field("x").expect("a name")),
            target: DataType::String,
            on_failure: CastFailure::Error,
        });
        cases.push(Expr::Cast {
            input: Box::new(field("x").expect("a name")),
            target: DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            on_failure: CastFailure::Null,
        });

        cases.push(Expr::Case {
            whens: vec![(
                Expr::BinaryOp {
                    op: BinaryOpKind::Gt,
                    left: Box::new(field("revenue").expect("a name")),
                    right: Box::new(sint(0)),
                },
                sint(1),
            )],
            else_: Some(Box::new(sint(0))),
        });
        cases.push(Expr::Case {
            whens: vec![
                (field("flag").expect("a name"), sint(1)),
                (field("other").expect("a name"), sint(2)),
            ],
            else_: None,
        });

        for negated in [false, true] {
            cases.push(Expr::InList {
                value: Box::new(field("country").expect("a name")),
                list: vec![
                    slit(Literal::String("DE".into())),
                    slit(Literal::String("FR".into())),
                ],
                negated,
            });
            cases.push(Expr::Between {
                value: Box::new(field("x").expect("a name")),
                low: Box::new(sint(1)),
                high: Box::new(sint(10)),
                negated,
            });
        }

        for kind in [
            LikeKind::Like,
            LikeKind::ILike,
            LikeKind::NotLike,
            LikeKind::NotILike,
        ] {
            cases.push(Expr::Like {
                value: Box::new(field("name").expect("a name")),
                pattern: Box::new(slit(Literal::String("%a%".into()))),
                kind,
            });
        }

        cases.push(Expr::IsNull(Box::new(field("revenue").expect("a name"))));

        for e in cases {
            rt_sem(e);
        }
    }

    #[test]
    fn round_trips_all_physical_leaves() {
        rt_phys(physical_col("amount").expect("a name"));
        rt_phys(Expr::Literal(Literal::BigInt(0)));
        rt_phys(Expr::Literal(Literal::String("USD".into())));
        rt_phys(Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(physical_col("a").expect("a name")),
            right: Box::new(Expr::Literal(Literal::BigInt(1))),
        });
    }

    #[test]
    fn compound_string_field_is_rejected() {
        let err = serde_yaml::from_str::<SemCarrier>("expr: \"a / b\"").unwrap_err();
        assert!(
            err.to_string().contains("inline expression DSL"),
            "expected inline-DSL rejection, got: {err}"
        );
    }
}
