use std::convert::identity;

use semstrait_ir::PhysicalExpr;
use semstrait_manifest::CompileError;
use semstrait_model::expr::{dim, lit};
use semstrait_model::{Dimension, Measure, SemanticModel, UnresolvedSemanticExpr};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, LiteralError, TypeError};

mod common;
use common::*;

const COLUMNS: [(&str, DataType); 3] = [
    (
        "amount",
        DataType::Decimal {
            precision: 18,
            scale: 2,
        },
    ),
    ("qty", DataType::Integer),
    ("units", DataType::BigInt),
];

fn model(dimensions: Vec<Dimension>, measures: Vec<Measure>) -> SemanticModel {
    let columns = COLUMNS.map(|(name, data_type)| dimension(name, data_type));
    orders_model(
        columns.into_iter().chain(dimensions).collect(),
        measures,
        identity,
    )
}

fn decimal(value: &str) -> Literal {
    Literal::Decimal {
        value: value.to_owned(),
        precision: 18,
        scale: 2,
    }
}

async fn stored(model: SemanticModel, name: &str) -> PhysicalExpr {
    let manifest = compile_model(model).await;
    tree(&manifest, signature(&manifest, "orders", name))
}

fn mapped(name: &str) -> PhysicalExpr {
    let (_, declared) = COLUMNS
        .into_iter()
        .find(|(column, _)| *column == name)
        .unwrap_or_else(|| panic!("`{name}` is one of COLUMNS"));
    cast(orders_column(name), declared)
}

fn literal(literal: Literal) -> PhysicalExpr {
    Expr::Literal(literal)
}

#[tokio::test]
async fn an_untyped_literal_takes_its_contexts_type() {
    let cases = [
        (
            dim("amount")
                .expect("a name")
                .gt(untyped(Literal::Double(1.5))),
            DataType::Boolean,
            mapped("amount").gt(literal(decimal("1.5"))),
        ),
        (
            dim("qty").expect("a name") + untyped(Literal::BigInt(1)),
            DataType::Integer,
            mapped("qty") + literal(Literal::Integer(1)),
        ),
        (
            call(
                "coalesce",
                vec![dim("qty").expect("a name"), untyped(Literal::BigInt(0))],
            ),
            DataType::Integer,
            physical_call(
                "coalesce",
                vec![mapped("qty"), literal(Literal::Integer(0))],
            ),
        ),
        (
            dim("amount")
                .expect("a name")
                .gt(untyped(Literal::String("5".to_owned()))),
            DataType::Boolean,
            mapped("amount").gt(literal(decimal("5"))),
        ),
        (
            untyped(Literal::BigInt(1)) + untyped(Literal::BigInt(2)),
            DataType::Integer,
            literal(Literal::Integer(3)),
        ),
    ];
    for (expr, declared, expected) in cases {
        let model = model(vec![bodied("derived", declared, expr)], Vec::new());
        assert_eq!(stored(model, "derived").await, expected);
    }
}

async fn type_error(expr: UnresolvedSemanticExpr, declared: DataType) -> TypeError {
    let model = model(vec![bodied("derived", declared, expr)], Vec::new());
    match compile_error(model).await {
        CompileError::TypeError { node, error } => {
            assert_eq!(node, "derived");
            error
        }
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

#[tokio::test]
async fn a_literal_its_context_cannot_take_is_refused_with_the_type_error() {
    let cut = type_error(
        dim("amount")
            .expect("a name")
            .gt(untyped(Literal::Double(1.125))),
        DataType::Boolean,
    )
    .await;
    assert!(
        matches!(&cut, TypeError::UntypedLiteral { context, .. } if context == &COLUMNS[0].1),
        "{cut:?}"
    );
    let string = type_error(
        dim("amount")
            .expect("a name")
            .gt(lit(Literal::String("5".to_owned()))),
        DataType::Boolean,
    )
    .await;
    assert!(
        matches!(string, TypeError::Arguments { name: "gt", .. }),
        "{string:?}"
    );
}

#[tokio::test]
async fn a_dimensions_literal_takes_its_declared_type_or_is_cast_to_it() {
    let day = lit(Literal::String("2026-05-19".to_owned()));
    assert_eq!(
        stored(
            model(vec![bodied("day", DataType::Date, day)], Vec::new()),
            "day"
        )
        .await,
        literal(Literal::Date("2026-05-19".to_owned()))
    );

    let string = |value: &str| Literal::String(value.to_owned());
    let cases = [
        (
            Literal::Double(1.5),
            COLUMNS[0].1.clone(),
            Ok(decimal("1.5")),
        ),
        (
            string("2026-05-19"),
            DataType::Date,
            Ok(Literal::Date("2026-05-19".to_owned())),
        ),
        (
            Literal::Null(DataType::String),
            DataType::BigInt,
            Ok(Literal::Null(DataType::BigInt)),
        ),
        (
            Literal::Double(1.5),
            DataType::Integer,
            Err(LiteralError::CutsADigit),
        ),
        (
            string("2026-05-19T10:00:00+02:00"),
            DataType::Timestamp { precision: 6 },
            Err(LiteralError::Zoned),
        ),
    ];
    for (authored, declared, manifest) in cases {
        match manifest {
            Ok(stored_literal) => {
                let model = model(
                    vec![bodied("derived", declared, untyped(authored))],
                    Vec::new(),
                );
                assert_eq!(stored(model, "derived").await, literal(stored_literal));
            }
            Err(error) => assert_eq!(
                type_error(untyped(authored.clone()), declared.clone()).await,
                TypeError::UntypedLiteral {
                    literal: authored,
                    context: declared,
                    error,
                }
            ),
        }
    }
}

#[tokio::test]
async fn a_tree_that_typing_refuses_is_refused_whatever_its_fold_gives() {
    let upper = || call("upper", vec![dim("amount").expect("a name")]);
    let cases = [
        (upper(), DataType::String),
        (
            upper().eq(lit(Literal::String("X".to_owned()))) & lit(Literal::Boolean(false)),
            DataType::Boolean,
        ),
        (
            Expr::Case {
                whens: vec![(lit(Literal::Boolean(true)), lit(Literal::Integer(1)))],
                else_: Some(Box::new(upper())),
            },
            DataType::Integer,
        ),
    ];
    for (expr, declared) in cases {
        let error = type_error(expr, declared).await;
        assert!(
            matches!(error, TypeError::Argument { name: "upper", .. }),
            "`upper` refuses a decimal: {error:?}"
        );
    }
}

#[tokio::test]
async fn a_measures_literal_is_typed_as_its_aggregates_operand() {
    let cases = [
        (
            untyped(Literal::BigInt(1)),
            DataType::BigInt,
            literal(Literal::BigInt(1)),
        ),
        (
            dim("amount").expect("a name") * untyped(Literal::Double(1.5)),
            DataType::Decimal {
                precision: 38,
                scale: 4,
            },
            mapped("amount") * literal(decimal("1.5")),
        ),
    ];
    for (expr, declared, operand) in cases {
        let measure = Measure::builder("total")
            .data_type(declared)
            .agg(agg("sum"))
            .expr(expr)
            .build();
        let model = model(Vec::new(), vec![measure]);
        assert_eq!(
            stored(model, "total").await,
            aggregate("sum", false, operand, None)
        );
    }
}

#[tokio::test]
async fn an_arithmetic_operations_type_does_not_depend_on_its_operands_order() {
    for (left, right) in [("qty", "units"), ("units", "qty")] {
        let model = model(
            vec![bodied(
                "sum_of_two",
                DataType::BigInt,
                dim(left).expect("a name") + dim(right).expect("a name"),
            )],
            Vec::new(),
        );
        assert_eq!(
            stored(model, "sum_of_two").await,
            mapped(left) + mapped(right),
            "{left} + {right}"
        );
    }
}

#[tokio::test]
async fn a_count_distinct_measure_stores_count_with_distinct() {
    let model = model(
        Vec::new(),
        vec![measure("customers", DataType::BigInt, "count_distinct")],
    );
    assert_eq!(
        stored(model, "customers").await,
        aggregate("count", true, orders_column("customers"), None)
    );
}

#[tokio::test]
async fn an_authored_window_call_is_refused() {
    let model = model(
        vec![bodied(
            "previous_amount",
            COLUMNS[0].1.clone(),
            lag(dim("amount").expect("a name"), dim("qty").expect("a name")),
        )],
        Vec::new(),
    );
    match compile_error(model).await {
        CompileError::TypeInferenceFailure { node, reason } => {
            assert!(
                node.contains("previous_amount"),
                "the semantics is named: {node}"
            );
            assert!(
                reason.contains("window"),
                "the reason names the window: {reason}"
            );
        }
        other => panic!("expected the authored window's refusal, got {other:?}"),
    }
}
