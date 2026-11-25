use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Call, Expr, OperationKind, RowExpr, UnaryOpKind};

use serde_yaml::Value;

use super::de::{CAST_FAILURES, LIST};
use super::error::ParseError;
use super::leaf::{authored_name, SerializeLeaf};
use super::literal::tagged_to_value;
use super::value::{map, seq, tag};

pub(crate) fn expr_to_value<L: SerializeLeaf>(expr: &RowExpr<L>) -> Result<Value, ParseError> {
    match expr {
        Expr::Leaf(l) => l.leaf_to_value(),

        Expr::Literal(literal) => tagged_to_value(literal),

        Expr::Column(column) => Ok(tag("col", authored_name(column)?)),

        Expr::BinaryOp { op, left, right } => Ok(tag(
            op.operation().canonical.name,
            seq(vec![expr_to_value(left)?, expr_to_value(right)?]),
        )),

        Expr::UnaryOp { op, operand } => {
            Ok(tag(op.operation().canonical.name, expr_to_value(operand)?))
        }

        Expr::Function(call) => {
            let clause = match call {
                Call::Scalar(_) => false,
                Call::Window(call) => call.nulls().is_some() || !call.over().is_empty(),
                _ => true,
            };
            if clause || !call.options().is_empty() {
                return Err(ParseError::NotAuthorRepresentable(
                    "a call that sets an option or writes a clause",
                ));
            }
            Ok(tag(call.name(), all_to_value(call.args())?))
        }

        Expr::Cast {
            input,
            target,
            on_failure,
        } => {
            let word = CAST_FAILURES
                .iter()
                .find_map(|&(word, failure)| (failure == *on_failure).then_some(word))
                .ok_or(ParseError::NotAuthorRepresentable(
                    "a cast failure that `on_failure` has no word for",
                ))?;
            let body = map(vec![
                ("input", expr_to_value(input)?),
                ("target", data_type_to_value(target)?),
                ("on_failure", Value::String(word.to_owned())),
            ]);
            Ok(tag("cast", body))
        }

        Expr::Case { whens, else_ } => {
            let mut whens_seq = Vec::with_capacity(whens.len());
            for (when, then) in whens {
                whens_seq.push(map(vec![
                    ("when", expr_to_value(when)?),
                    ("then", expr_to_value(then)?),
                ]));
            }
            let mut body = vec![("whens", seq(whens_seq))];
            if let Some(e) = else_ {
                body.push(("else", expr_to_value(e)?));
            }
            Ok(tag("case", map(body)))
        }

        Expr::InList {
            value,
            list,
            negated,
        } => {
            let inner = tag(
                OperationKind::InList.operation().canonical.name,
                map(vec![
                    ("value", expr_to_value(value)?),
                    (LIST, all_to_value(list)?),
                ]),
            );
            Ok(maybe_negate(*negated, inner))
        }

        Expr::Between {
            value,
            low,
            high,
            negated,
        } => {
            let inner = tag(
                OperationKind::Between.operation().canonical.name,
                map(vec![
                    ("value", expr_to_value(value)?),
                    ("low", expr_to_value(low)?),
                    ("high", expr_to_value(high)?),
                ]),
            );
            Ok(maybe_negate(*negated, inner))
        }

        Expr::Like {
            value,
            pattern,
            kind,
        } => Ok(tag(
            kind.operation().canonical.name,
            map(vec![
                ("value", expr_to_value(value)?),
                ("pattern", expr_to_value(pattern)?),
            ]),
        )),

        Expr::IsNull(inner) => Ok(tag(
            OperationKind::IsNull.operation().canonical.name,
            expr_to_value(inner)?,
        )),

        _ => Err(ParseError::NotAuthorRepresentable(
            "a node that the authoring grammar has no form for",
        )),
    }
}

fn all_to_value<L: SerializeLeaf>(exprs: &[RowExpr<L>]) -> Result<Value, ParseError> {
    exprs
        .iter()
        .map(expr_to_value)
        .collect::<Result<_, _>>()
        .map(seq)
}

pub(crate) fn data_type_to_value(dt: &DataType) -> Result<Value, ParseError> {
    if !dt.is_valid() {
        return Err(ParseError::NotAuthorRepresentable(
            "a data type that is not valid, `Unknown` among them",
        ));
    }
    Ok(Value::String(dt.to_string()))
}

fn maybe_negate(negated: bool, inner: Value) -> Value {
    if negated {
        tag(UnaryOpKind::Not.operation().canonical.name, inner)
    } else {
        inner
    }
}

#[cfg(test)]
mod tests {
    use super::super::de::parse_expr;
    use super::super::error::ParseError;
    use super::super::test_call as call;
    use super::super::{SemanticExpr, Unresolved};
    use crate::expr::{col, dim, field, physical_col};
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{
        BinaryOpKind, Call, CastFailure, DimensionAccessor, Expr, FunctionRef, LikeKind, Literal,
        MeasureAccessor, NamedReference, Over, PhysicalLeaf, RowExpr, SemanticLeaf, UnaryOpKind,
        UntypedLiteral, WindowCall,
    };
    use semstrait_sql::functions::registry;
    use serde_yaml::Value;

    #[track_caller]
    fn round_trip_semantic(expr: RowExpr<SemanticLeaf>) {
        let value = super::expr_to_value(&expr).expect("serialize");
        let yaml = serde_yaml::to_string(&value).expect("yaml");
        assert!(
            !yaml.contains('!'),
            "serialized authoring YAML must not carry `!`-tags:\n{yaml}"
        );
        let reparsed: Value = serde_yaml::from_str(&yaml).expect("yaml round-trip");
        let back: RowExpr<SemanticLeaf> = parse_expr(&reparsed).expect("deserialize/parse_expr");
        assert_eq!(expr, back, "round-trip mismatch via:\n{yaml}");
    }

    #[track_caller]
    fn round_trip_physical(expr: RowExpr<PhysicalLeaf>) {
        let value = super::expr_to_value(&expr).expect("serialize");
        let yaml = serde_yaml::to_string(&value).expect("yaml");
        assert!(!yaml.contains('!'), "no `!`-tags:\n{yaml}");
        let reparsed: Value = serde_yaml::from_str(&yaml).expect("yaml round-trip");
        let back: RowExpr<PhysicalLeaf> = parse_expr(&reparsed).expect("parse_expr");
        assert_eq!(expr, back, "round-trip mismatch via:\n{yaml}");
    }

    fn slit_i(v: i64) -> RowExpr<SemanticLeaf> {
        Expr::Literal(Literal::BigInt(v))
    }

    #[test]
    fn round_trips_every_literal_tagged_and_untyped() {
        let text = |value: &str| value.to_owned();
        let tagged = vec![
            Literal::Boolean(true),
            Literal::TinyInt(-7),
            Literal::SmallInt(7),
            Literal::Integer(7),
            Literal::BigInt(-7),
            Literal::Real(0.1),
            Literal::Double(5.0),
            Literal::Decimal {
                value: text("1.23"),
                precision: 4,
                scale: 2,
            },
            Literal::String(text("42")),
            Literal::VarChar {
                value: text("abc"),
                length: 10,
            },
            Literal::FixedChar {
                value: text("ab"),
                length: 2,
            },
            Literal::Binary(vec![0xDE, 0xAD]),
            Literal::FixedBinary {
                value: vec![0xBE, 0xEF],
                length: 2,
            },
            Literal::Date(text("2026-01-01")),
            Literal::Time {
                value: text("12:34:56"),
                precision: 3,
            },
            Literal::Timestamp {
                value: text("2026-01-01T00:00:00"),
                precision: 6,
            },
            Literal::Interval(text("P1D")),
            Literal::IntervalYearMonth(text("P1Y")),
            Literal::IntervalDaySecond {
                value: text("P1D"),
                precision: 6,
            },
            Literal::Uuid(text("550e8400-e29b-41d4-a716-446655440000")),
            Literal::Array {
                element: DataType::Integer,
                values: vec![Literal::Integer(1), Literal::Null(DataType::Integer)],
            },
            Literal::Null(DataType::BigInt),
        ];
        for lit in tagged {
            round_trip_semantic(Expr::Literal(lit));
        }
        let untyped = vec![
            Literal::Boolean(true),
            Literal::BigInt(-7),
            Literal::Double(1.5),
            Literal::Double(5.0),
            Literal::String(text("42")),
            Literal::String(text("true")),
            Literal::Null(DataType::String),
        ];
        for lit in untyped {
            round_trip_semantic(Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(lit))));
        }
    }

    #[test]
    fn round_trips_leaves() {
        round_trip_semantic(field("revenue").expect("a name"));
        round_trip_semantic(col("amount").expect("a name"));
        round_trip_semantic(dim("dt").expect("a name"));
        round_trip_semantic(Expr::Leaf(SemanticLeaf::Dimension(
            NamedReference::unqualified("dt").expect("a name"),
            Some(DimensionAccessor::Lag(2)),
        )));
        round_trip_semantic(Expr::Leaf(SemanticLeaf::Measure(
            NamedReference::unqualified("rev").expect("a name"),
            Some(MeasureAccessor::Previous),
        )));
        round_trip_physical(physical_col("amount").expect("a name"));
    }

    #[test]
    fn round_trips_structural_variants() {
        round_trip_semantic(Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(field("revenue").expect("a name")),
            right: Box::new(field("cost").expect("a name")),
        });
        round_trip_semantic(Expr::UnaryOp {
            op: UnaryOpKind::Negate,
            operand: Box::new(slit_i(5)),
        });
        round_trip_semantic(Expr::UnaryOp {
            op: UnaryOpKind::Not,
            operand: Box::new(Expr::BinaryOp {
                op: BinaryOpKind::Eq,
                left: Box::new(field("a").expect("a name")),
                right: Box::new(field("b").expect("a name")),
            }),
        });
        round_trip_semantic(call("upper", vec![field("name").expect("a name")]));
        round_trip_semantic(call(
            "concat",
            vec![
                field("a").expect("a name"),
                field("b").expect("a name"),
                field("c").expect("a name"),
            ],
        ));
        round_trip_semantic(call("current_date", vec![]));
        round_trip_semantic(call("row_number", vec![]));
        round_trip_semantic(Expr::Cast {
            input: Box::new(field("x").expect("a name")),
            target: DataType::String,
            on_failure: CastFailure::Error,
        });
        round_trip_semantic(Expr::Cast {
            input: Box::new(field("x").expect("a name")),
            target: DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            on_failure: CastFailure::Null,
        });
        round_trip_semantic(Expr::Case {
            whens: vec![(
                Expr::BinaryOp {
                    op: BinaryOpKind::Gt,
                    left: Box::new(field("revenue").expect("a name")),
                    right: Box::new(slit_i(0)),
                },
                slit_i(1),
            )],
            else_: Some(Box::new(slit_i(0))),
        });
        for negated in [false, true] {
            round_trip_semantic(Expr::InList {
                value: Box::new(field("country").expect("a name")),
                list: vec![
                    Expr::Literal(Literal::String("DE".into())),
                    Expr::Literal(Literal::String("FR".into())),
                ],
                negated,
            });
            round_trip_semantic(Expr::Between {
                value: Box::new(field("x").expect("a name")),
                low: Box::new(slit_i(1)),
                high: Box::new(slit_i(10)),
                negated,
            });
        }
        for kind in [
            LikeKind::Like,
            LikeKind::ILike,
            LikeKind::NotLike,
            LikeKind::NotILike,
        ] {
            round_trip_semantic(Expr::Like {
                value: Box::new(field("name").expect("a name")),
                pattern: Box::new(Expr::Literal(Literal::String("%a%".into()))),
                kind,
            });
        }
        round_trip_semantic(Expr::IsNull(Box::new(field("revenue").expect("a name"))));
    }

    #[test]
    fn unrepresentable_nodes_error_not_panic() {
        let Some(FunctionRef::Window(row_number)) = registry().get("row_number") else {
            panic!("`row_number` is a window function");
        };
        let over = Over {
            partition_by: vec![physical_col("r").expect("a name")],
            order_by: Vec::new(),
            window_frame: None,
        };
        let window = WindowCall::builder(row_number, vec![])
            .over(over)
            .build()
            .expect("`row_number` takes PARTITION BY");
        let win: RowExpr<PhysicalLeaf> = Expr::Function(Call::Window(window));
        assert!(matches!(
            super::expr_to_value(&win),
            Err(ParseError::NotAuthorRepresentable(_))
        ));

        let qualified: RowExpr<SemanticLeaf> =
            Expr::Column(NamedReference::qualified("ps0", "amount").expect("an alias and a name"));
        assert!(matches!(
            super::expr_to_value(&qualified),
            Err(ParseError::NotAuthorRepresentable(_))
        ));

        let narrow: RowExpr<SemanticLeaf> =
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::Integer(1))));
        assert!(matches!(
            super::expr_to_value(&narrow),
            Err(ParseError::NotAuthorRepresentable(_))
        ));

        let mut buf = Vec::new();
        let mut ser = serde_yaml::Serializer::new(&mut buf);
        assert!(Unresolved::<SemanticExpr>::serialize(&narrow, &mut ser).is_err());
    }
}
