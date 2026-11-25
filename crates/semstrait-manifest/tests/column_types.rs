use semstrait_catalog::CatalogProvider;
use semstrait_ir::PhysicalExpr;
use semstrait_manifest::CompileError;
use semstrait_model::expr::col;
use semstrait_model::{
    DataKindFilter, DatasetBuilder, SemanticMapping, SemanticModel, UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::{DataType, TypeOrFamily};
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, TypeError};

mod common;
use common::*;

fn mapped(mapping: SemanticMapping) -> impl FnOnce(DatasetBuilder) -> DatasetBuilder {
    |orders| orders.semantic_mapping(mapping)
}

fn auto(orders: DatasetBuilder) -> DatasetBuilder {
    orders
}

async fn stored<const N: usize>(model: SemanticModel, names: [&str; N]) -> [PhysicalExpr; N] {
    stored_with(model, None, "orders", names).await
}

async fn type_error(model: SemanticModel, catalog: Option<&dyn CatalogProvider>) -> TypeError {
    match compile_error_with(model, catalog).await {
        CompileError::TypeError { error, .. } => error,
        other => panic!("expected the TypeError variant, got {other:?}"),
    }
}

fn is_5(column: &str) -> UnresolvedSemanticExpr {
    col(column).expect("a name").eq(untyped(Literal::BigInt(5)))
}

fn filter(name: &str, expr: UnresolvedSemanticExpr) -> DataKindFilter {
    DataKindFilter::builder(name).expr(expr).build()
}

#[tokio::test]
async fn region_is_5_is_refused_where_the_catalog_gives_region_as_a_string() {
    let model = orders_model(
        vec![bodied("is_five", DataType::Boolean, is_5("region"))],
        Vec::new(),
        auto,
    );
    let error = type_error(model, Some(&catalog(&[("region", "string")]))).await;
    assert!(
        matches!(
            &error,
            TypeError::UntypedLiteral {
                context: DataType::String,
                ..
            }
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn amount_above_1_5_stores_a_decimal_where_the_catalog_gives_amount_as_a_decimal() {
    let model = orders_model(
        vec![bodied(
            "is_big",
            DataType::Boolean,
            col("amount")
                .expect("a name")
                .gt(untyped(Literal::Double(1.5))),
        )],
        Vec::new(),
        auto,
    );
    let catalog = catalog(&[("amount", "decimal(18,2)")]);
    let decimal = Literal::Decimal {
        value: "1.5".to_owned(),
        precision: 18,
        scale: 2,
    };
    assert_eq!(
        stored_with(model, Some(&catalog), "orders", ["is_big"]).await,
        [orders_column("amount").gt(Expr::Literal(decimal))]
    );
}

#[tokio::test]
async fn region_is_5_in_a_filter_is_refused_where_a_string_dimension_maps_region() {
    let model = orders_model(
        vec![dimension("region", DataType::String)],
        Vec::new(),
        |orders| orders.filter(filter("region_five", is_5("region"))),
    );
    let error = type_error(model, None).await;
    assert!(
        matches!(
            &error,
            TypeError::UntypedLiteral {
                context: DataType::String,
                ..
            }
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn channel_is_5_in_a_filter_compiles_where_no_semantics_maps_channel() {
    let model = orders_model(
        vec![dimension("region", DataType::String)],
        Vec::new(),
        |orders| orders.filter(filter("channel_five", is_5("channel"))),
    );
    assert_eq!(
        stored(model, ["channel_five"]).await,
        [orders_column("channel").eq(Expr::Literal(Literal::BigInt(5)))]
    );
}

#[tokio::test]
async fn upper_of_amount_is_refused_where_the_catalog_gives_amount_as_a_decimal() {
    let model = orders_model(
        vec![bodied(
            "loud",
            DataType::String,
            call("upper", vec![col("amount").expect("a name")]),
        )],
        Vec::new(),
        auto,
    );
    let error = type_error(model, Some(&catalog(&[("amount", "decimal(18,2)")]))).await;
    assert!(
        matches!(
            &error,
            TypeError::Argument {
                name: "upper",
                got: TypeOrFamily::Type(got),
                ..
            } if got == &decimal_18_2()
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn a_directly_mapped_column_is_cast_to_its_dimensions_declared_type() {
    let mapping = SemanticMapping::builder()
        .column("country", "adwords-country")
        .build();
    let model = orders_model(
        vec![dimension("country", DataType::String)],
        Vec::new(),
        mapped(mapping),
    );
    assert_eq!(
        stored(model, ["country"]).await,
        [cast(orders_column("adwords-country"), DataType::String)]
    );
}

#[tokio::test]
async fn a_measures_column_is_cast_under_its_aggregate_and_the_result_to_its_declared_type() {
    let mapping = SemanticMapping::builder()
        .column("cost", "adwords-cost")
        .build();
    let model = orders_model(
        Vec::new(),
        vec![measure("cost", decimal_18_2(), "sum")],
        mapped(mapping),
    );
    let column = cast(orders_column("adwords-cost"), decimal_18_2());
    assert_eq!(
        stored(model, ["cost"]).await,
        [cast(aggregate("sum", false, column, None), decimal_18_2())]
    );
}

#[tokio::test]
async fn a_counts_column_gets_no_cast_and_its_result_gets_the_declared_type() {
    let mapping = SemanticMapping::builder()
        .column("order_count", "order_id")
        .build();
    let model = orders_model(
        Vec::new(),
        vec![measure("order_count", DataType::Integer, "count")],
        mapped(mapping),
    );
    let count = aggregate("count", false, orders_column("order_id"), None);
    assert_eq!(
        stored(model, ["order_count"]).await,
        [cast(count, DataType::Integer)]
    );
}

#[tokio::test]
async fn an_approx_count_distincts_column_gets_no_cast() {
    let mapping = SemanticMapping::builder()
        .column("users", "user_id")
        .build();
    let model = orders_model(
        Vec::new(),
        vec![measure("users", DataType::BigInt, "approx_count_distinct")],
        mapped(mapping),
    );
    let users = aggregate(
        "approx_count_distinct",
        false,
        orders_column("user_id"),
        None,
    );
    assert_eq!(stored(model, ["users"]).await, [users]);
}

#[tokio::test]
async fn an_observed_column_is_cast_under_sum_avg_min_and_max_where_its_type_differs() {
    let mapping = SemanticMapping::builder()
        .column("top", "x")
        .column("total", "y")
        .column("lowest", "x")
        .column("mean", "x")
        .build();
    let model = orders_model(
        Vec::new(),
        vec![
            measure("top", DataType::Integer, "max"),
            measure("total", DataType::Integer, "sum"),
            measure("lowest", DataType::Integer, "min"),
            measure("mean", DataType::Double, "avg"),
        ],
        mapped(mapping),
    );
    let catalog = catalog(&[("x", "string"), ("y", "double")]);
    let cast_column = |name, to| cast(orders_column(name), to);
    let over = |function, column, to| aggregate(function, false, cast_column(column, to), None);
    let names = ["top", "total", "lowest", "mean"];
    let observed = stored_with(model.clone(), Some(&catalog), "orders", names).await;
    assert_eq!(
        observed,
        [
            over("max", "x", DataType::Integer),
            cast(over("sum", "y", DataType::Integer), DataType::Integer),
            over("min", "x", DataType::Integer),
            over("avg", "x", DataType::Double),
        ]
    );
    assert_eq!(stored(model, names).await, observed, "with no catalog");
}

#[tokio::test]
async fn a_count_casts_nothing_on_an_observed_column() {
    let mapping = SemanticMapping::builder()
        .column("orders", "x")
        .column("customers", "x")
        .column("visitors", "x")
        .build();
    let model = orders_model(
        Vec::new(),
        vec![
            measure("orders", DataType::Integer, "count"),
            measure("customers", DataType::BigInt, "count_distinct"),
            measure("visitors", DataType::BigInt, "approx_count_distinct"),
        ],
        mapped(mapping),
    );
    let catalog = catalog(&[("x", "string")]);
    let x = || orders_column("x");
    assert_eq!(
        stored_with(
            model,
            Some(&catalog),
            "orders",
            ["orders", "customers", "visitors"]
        )
        .await,
        [
            cast(aggregate("count", false, x(), None), DataType::Integer),
            aggregate("count", true, x(), None),
            aggregate("approx_count_distinct", false, x(), None),
        ]
    );
}

#[tokio::test]
async fn an_expression_over_a_mapped_column_reads_it_with_no_cast() {
    let model = orders_model(
        vec![
            dimension("name", DataType::String),
            bodied(
                "name_length",
                DataType::Integer,
                call("length", vec![col("name").expect("a name")]),
            ),
        ],
        Vec::new(),
        auto,
    );
    assert_eq!(
        stored(model, ["name_length"]).await,
        [physical_call("length", vec![orders_column("name")])]
    );
}

#[tokio::test]
async fn two_semantics_on_one_column_each_cast_it_to_their_own_type() {
    let mapping = SemanticMapping::builder()
        .column("order_date", "created_at")
        .column("order_ts", "created_at")
        .build();
    let timestamp = DataType::Timestamp { precision: 6 };
    let model = orders_model(
        vec![
            dimension("order_date", DataType::Date),
            dimension("order_ts", timestamp.clone()),
        ],
        Vec::new(),
        mapped(mapping),
    );
    assert_eq!(
        stored(model, ["order_date", "order_ts"]).await,
        [
            cast(orders_column("created_at"), DataType::Date),
            cast(orders_column("created_at"), timestamp),
        ]
    );
}
