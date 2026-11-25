use std::convert::identity;

use semstrait_catalog::{Schema as NativeSchema, SchemaColumn as NativeColumn};
use semstrait_manifest::CompileError;
use semstrait_model::expr::{col, dim, lit};
use semstrait_model::{
    DataKindFilter, DatasetBuilder, Dimension, Measure, SemanticMapping, SemanticModel,
};
use semstrait_sql::datatype::{DataType, TypeFamily};
use semstrait_sql::expr::{CastFailure, Expr, ExprOperationExt, Literal, TypeError};
use semstrait_sql::functions::ParameterType;

mod common;
use common::*;

fn model(day: Literal, day_type: DataType, measures: Vec<Measure>) -> SemanticModel {
    let mapping = measures
        .iter()
        .map(|measure| measure.name().as_str())
        .fold(
            SemanticMapping::builder().literal("day", day),
            |mapping, name| mapping.column(name, name),
        )
        .build();
    orders_model(vec![dimension("day", day_type)], measures, |orders| {
        orders.semantic_mapping(mapping)
    })
}

fn literal_day(day: Literal, declared: DataType) -> SemanticModel {
    model(day, declared, Vec::new())
}

#[tokio::test]
async fn a_tree_whose_type_is_not_its_declared_type_is_cast_to_it() {
    let model = model(
        Literal::String("2026-05-19".to_owned()),
        DataType::Date,
        vec![
            measure("cost", decimal_18_2(), "sum"),
            measure("quantity", DataType::Integer, "sum"),
            measure("avg_price", decimal_18_2(), "avg"),
            measure("units", DataType::Integer, "sum"),
        ],
    );
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![
            NativeColumn::new("cost", "decimal(18,2)", true),
            NativeColumn::new("quantity", "int", true),
            NativeColumn::new("avg_price", "decimal(18,2)", true),
            NativeColumn::new("units", "double", true),
        ]),
    );
    let manifest = compile_model_with(model, Some(&catalog)).await;
    let over = |aggregate_name: &str, name: &str, declared: DataType| {
        let call = aggregate(aggregate_name, false, orders_column(name), None);
        (name.to_owned(), cast(call, declared))
    };
    for (name, expected) in [
        over("sum", "cost", decimal_18_2()),
        over("sum", "quantity", DataType::Integer),
        over("avg", "avg_price", decimal_18_2()),
        (
            "units".to_owned(),
            cast(
                aggregate(
                    "sum",
                    false,
                    cast(orders_column("units"), DataType::Integer),
                    None,
                ),
                DataType::Integer,
            ),
        ),
        (
            "day".to_owned(),
            Expr::Literal(Literal::Date("2026-05-19".to_owned())),
        ),
    ] {
        assert_eq!(
            tree(&manifest, signature(&manifest, "orders", &name)),
            expected,
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_tree_whose_type_does_not_cast_to_its_declared_type_is_refused() {
    let model = literal_day(Literal::Boolean(true), DataType::Date);
    match compile_error(model).await {
        CompileError::TypeError { node, error } => {
            assert_eq!(node, "day");
            assert_eq!(
                error,
                TypeError::Cast {
                    from: DataType::Boolean.into(),
                    to: DataType::Date,
                }
            );
        }
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

#[tokio::test]
async fn an_aggregate_whose_result_does_not_cast_to_its_declared_type_is_refused() {
    let model = orders_model(
        Vec::new(),
        vec![measure("units", DataType::Boolean, "count")],
        identity,
    );
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![NativeColumn::new("units", "int", true)]),
    );
    match compile_error_with(model, Some(&catalog)).await {
        CompileError::TypeError { node, error } => {
            assert_eq!(node, "units");
            assert_eq!(
                error,
                TypeError::Cast {
                    from: DataType::BigInt.into(),
                    to: DataType::Boolean,
                }
            );
        }
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

#[tokio::test]
async fn an_authored_cast_that_the_cast_table_refuses_is_refused() {
    let authored = Expr::Cast {
        input: Box::new(Expr::Literal(Literal::Boolean(true))),
        target: DataType::Date,
        on_failure: CastFailure::Error,
    };
    let model = orders_model(
        vec![bodied("day", DataType::String, authored)],
        Vec::new(),
        identity,
    );
    match compile_error(model).await {
        CompileError::TypeError { node, error } => {
            assert_eq!(node, "day");
            assert_eq!(
                error,
                TypeError::Cast {
                    from: DataType::Boolean.into(),
                    to: DataType::Date,
                }
            );
        }
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

#[tokio::test]
async fn a_tagged_literals_narrowing_cast_is_refused_whatever_its_value() {
    for value in [1.5, 2.0] {
        let literal = Literal::Double(value);
        let authored = orders_model(
            vec![bodied(
                "day",
                DataType::Integer,
                Expr::Literal(literal.clone()),
            )],
            Vec::new(),
            identity,
        );
        for model in [literal_day(literal.clone(), DataType::Integer), authored] {
            match compile_error(model).await {
                CompileError::NarrowingLiteral {
                    node,
                    literal: refused,
                    declared,
                } => {
                    assert_eq!(node, "day");
                    assert_eq!(refused, literal);
                    assert_eq!(declared, DataType::Integer);
                }
                other => panic!("expected the narrowing literal's refusal, got {other:?}"),
            }
        }
    }
}

#[tokio::test]
async fn a_reference_to_a_literal_mapped_semantics_is_cast_to_its_declared_type() {
    for mapping in [
        SemanticMapping::builder().untyped("a", Literal::BigInt(7)),
        SemanticMapping::builder().literal("a", Literal::BigInt(7)),
    ] {
        let model = orders_model(
            vec![
                dimension("a", DataType::BigInt),
                bodied("b", DataType::Integer, dim("a").expect("a name")),
            ],
            Vec::new(),
            |orders| orders.semantic_mapping(mapping.build()),
        );
        let manifest = compile_model(model).await;
        assert_eq!(
            tree(&manifest, signature(&manifest, "orders", "b")),
            Expr::Literal(Literal::Integer(7))
        );
    }
}

fn scored(
    derived: Option<Dimension>,
    measures: Vec<Measure>,
    filter: Option<DataKindFilter>,
) -> SemanticModel {
    let operands = [
        dimension("score", DataType::Double),
        dimension("price", decimal_18_2()),
    ];
    orders_model(
        operands.into_iter().chain(derived).collect(),
        measures,
        |orders| filter.into_iter().fold(orders, DatasetBuilder::filter),
    )
}

#[tokio::test]
async fn a_tree_with_no_type_gets_the_declared_types_cast_or_is_refused_where_none_is_declared() {
    let margin = bodied(
        "margin",
        DataType::Double,
        dim("score").expect("a name") * dim("price").expect("a name"),
    );
    let total = Measure::builder("total")
        .data_type(DataType::Double)
        .agg(agg("sum"))
        .expr(dim("score").expect("a name") * dim("price").expect("a name"))
        .build();
    let manifest = compile_model(scored(Some(margin), vec![total], None)).await;
    let product = || {
        cast(orders_column("score"), DataType::Double)
            * cast(orders_column("price"), decimal_18_2())
    };
    for (name, typed) in [
        ("margin", product()),
        ("total", aggregate("sum", false, product(), None)),
    ] {
        assert_eq!(
            tree(&manifest, signature(&manifest, "orders", name)),
            cast(typed, DataType::Double),
            "{name}"
        );
    }

    let filter = DataKindFilter::builder("weighted")
        .expr(dim("score").expect("a name") * dim("price").expect("a name"))
        .build();
    match compile_error(scored(None, Vec::new(), Some(filter))).await {
        CompileError::TypeInferenceFailure { node, .. } => assert_eq!(node, "weighted"),
        other => panic!("expected the refusal of a tree with no type, got {other:?}"),
    }
}

#[tokio::test]
async fn a_measure_condition_that_is_not_a_boolean_is_refused_by_its_type_or_its_family() {
    for (filter, found) in [
        (dim("score").expect("a name") + lit(1), "double"),
        (
            dim("score").expect("a name") * dim("price").expect("a name"),
            "a value of the number family with no type",
        ),
    ] {
        let total = Measure::builder("total")
            .data_type(DataType::Double)
            .agg(agg("sum"))
            .expr(dim("score").expect("a name"))
            .filter(filter)
            .build();
        match compile_error(scored(None, vec![total], None)).await {
            CompileError::TypeInferenceFailure { node, reason } => {
                assert_eq!(node, "total filter");
                assert_eq!(reason, format!("expected Boolean, found {found}"));
            }
            other => {
                panic!("expected the refusal of a condition that is no boolean, got {other:?}")
            }
        }
    }
}

#[tokio::test]
async fn a_measure_condition_with_no_type_and_no_family_compiles() {
    let total = Measure::builder("total")
        .data_type(DataType::Double)
        .agg(agg("sum"))
        .expr(dim("score").expect("a name"))
        .filter(col("flag").expect("a name"))
        .build();
    let manifest = compile_model(scored(None, vec![total], None)).await;
    let root = (signature(&manifest, "orders", "total").filter_expr_id)
        .expect("the Measure keeps its condition");
    assert_eq!(stored_tree(&manifest, root), orders_column("flag"));
}

async fn refused_cast_to_a_date(model: SemanticModel, node: &str) {
    match compile_error(model).await {
        CompileError::TypeError {
            node: refused,
            error,
        } => {
            assert_eq!(refused, node);
            assert_eq!(
                error,
                TypeError::Cast {
                    from: TypeFamily::Number.into(),
                    to: DataType::Date,
                }
            );
        }
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

#[tokio::test]
async fn an_authored_cast_of_a_tree_with_no_type_is_checked_by_its_family() {
    let authored = Expr::Cast {
        input: Box::new(dim("score").expect("a name") * dim("price").expect("a name")),
        target: DataType::Date,
        on_failure: CastFailure::Error,
    };
    let bad = bodied("bad", DataType::Date, authored);
    refused_cast_to_a_date(scored(Some(bad), Vec::new(), None), "bad").await;
}

#[tokio::test]
async fn a_declared_type_cast_of_a_tree_with_no_type_is_checked_by_its_family() {
    let bad = bodied(
        "bad_declared",
        DataType::Date,
        dim("score").expect("a name") * dim("price").expect("a name"),
    );
    refused_cast_to_a_date(scored(Some(bad), Vec::new(), None), "bad_declared").await;

    let margin = bodied(
        "margin",
        DataType::Double,
        dim("score").expect("a name") * dim("price").expect("a name"),
    );
    let manifest = compile_model(scored(Some(margin), Vec::new(), None)).await;
    let product = cast(orders_column("score"), DataType::Double)
        * cast(orders_column("price"), decimal_18_2());
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "margin")),
        cast(product, DataType::Double)
    );

    let mystery = bodied(
        "day",
        DataType::Date,
        col("mystery").expect("a name") * lit(Literal::BigInt(2)),
    );
    let manifest = compile_model(scored(Some(mystery), Vec::new(), None)).await;
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "day")),
        cast(
            orders_column("mystery") * Expr::Literal(Literal::BigInt(2)),
            DataType::Date
        )
    );
}

#[tokio::test]
async fn a_tree_with_no_type_is_checked_by_its_operands_family() {
    let product = || dim("score").expect("a name") * dim("price").expect("a name");
    let label = bodied("label", DataType::String, call("upper", vec![product()]));
    let abc = DataKindFilter::builder("abc")
        .expr(product().eq(lit(Literal::String("abc".to_owned()))))
        .build();
    for (model, refused, expected) in [
        (
            scored(Some(label), Vec::new(), None),
            "label",
            TypeError::Argument {
                name: "upper",
                parameter: "value",
                takes: &ParameterType::Family(&TypeFamily::String),
                got: TypeFamily::Number.into(),
            },
        ),
        (
            scored(None, Vec::new(), Some(abc)),
            "abc",
            TypeError::Arguments {
                name: "eq",
                types: vec![TypeFamily::Number.into(), DataType::String.into()],
            },
        ),
    ] {
        match compile_error(model).await {
            CompileError::TypeError { node, error } => {
                assert_eq!(node, refused);
                assert_eq!(error, expected);
            }
            other => panic!("expected the TypeError variant, got {other:?}"),
        }
    }
}
