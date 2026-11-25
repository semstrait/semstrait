mod common;

use common::{agg, call};
use semstrait_common::Granularity;
use semstrait_model::expr::{dim, field, lit, measure, metric, physical_col};
use semstrait_model::model::serde::deserialize_mapping_value;
use semstrait_model::{
    DataKindFilter, Dimension, DimensionSite, Measure, MeasureSite, Metric, MetricSite, ParseError,
    Reference, Relationship, SemanticMapping, SemanticMappingValue, UnresolvedPhysicalExpr,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, PhysicalLeaf, UntypedLiteral};

fn body() -> UnresolvedSemanticExpr {
    (measure("revenue").expect("a name") - lit(1_i64))
        / (field("orders").expect("a name") + lit(2_i64))
}

fn predicate() -> UnresolvedSemanticExpr {
    dim("region").expect("a name").eq(lit("EU"))
}

fn assert_serializes<T: serde::Serialize>(what: &str, value: &T) {
    serde_json::to_value(value).unwrap_or_else(|e| panic!("{what} must serialize to JSON: {e}"));
    serde_yaml::to_string(value).unwrap_or_else(|e| panic!("{what} must serialize to YAML: {e}"));
}

#[test]
fn dimension_expr_setter_takes_a_row_expr() {
    let d = Dimension::builder("region_norm")
        .data_type(DataType::String)
        .categorical()
        .expr(body())
        .build();
    assert!(d.body.body.expr.is_some());
    assert_serializes("Dimension.expr", &d);
}

#[test]
fn measure_expr_and_filter_setters_take_a_row_expr() {
    let m = Measure::builder("net_revenue")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .expr(body())
        .filter(predicate())
        .build();
    assert!(m.body.body.expr.is_some());
    assert!(m.filter.is_some());
    assert_serializes("Measure.expr / Measure.filter", &m);
}

#[test]
fn metric_expr_and_filter_setters_take_a_row_expr() {
    let m = Metric::builder("conv_rate")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .expr(measure("conversions").expect("a name") / measure("sessions").expect("a name"))
        .filter(predicate())
        .build();
    assert!(m.body.body.expr.is_some());
    assert!(m.filter.is_some());
    assert_serializes("Metric.expr / Metric.filter", &m);
}

#[test]
fn site_and_reference_constructors_take_a_row_expr() {
    let d = DimensionSite::ref_with_expr("region", dim("region").expect("a name").is_null());
    let m = MeasureSite::ref_with_expr("revenue", body());
    let mm = MetricSite::ref_with_expr(
        "conv_rate",
        metric("conv_rate").expect("a name") + lit(0_i64),
    );
    let r = Reference::with_expr("revenue", predicate());
    assert!(r.body.expr.is_some());
    assert_serializes("DimensionSite::ref_with_expr", &d);
    assert_serializes("MeasureSite::ref_with_expr", &m);
    assert_serializes("MetricSite::ref_with_expr", &mm);
    assert_serializes("Reference::with_expr", &r);
}

#[test]
fn data_kind_filter_expr_setter_takes_a_row_expr() {
    let f = DataKindFilter::builder("recent").expr(predicate()).build();
    assert_serializes("DataKindFilter.expr", &f);
}

#[test]
fn relationship_filter_setter_takes_a_row_expr() {
    let r = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on_key("customer_id")
        .many_to_one()
        .filter(predicate())
        .build()
        .expect("relationship builds");
    assert!(r.filter.is_some());
    assert_serializes("Relationship.filter", &r);
}

#[test]
fn semantic_mapping_expr_setter_takes_a_physical_row_expr() {
    let mapping = SemanticMapping::builder()
        .expr(
            "amount",
            physical_col("amount_cents").expect("a name") / physical_col("rate").expect("a name"),
        )
        .build();
    let SemanticMapping::Explicit(entries) = &mapping else {
        panic!("an explicit entry was inserted");
    };
    let value = entries.values().next().expect("one entry");
    let SemanticMappingValue::Expr(expr) = value else {
        panic!("the `expr:` arm was inserted, got {value:?}");
    };
    let _: &UnresolvedPhysicalExpr = expr;
    assert_serializes("SemanticMappingValue::Expr", &mapping);
}

#[test]
fn a_semantic_mapping_s_expr_reads_every_node_of_the_tree() {
    let read = |body: &str| {
        let value = serde_yaml::from_str(&format!("{{expr: {body}}}")).expect("the arm is YAML");
        deserialize_mapping_value(&value)
    };
    let campaign = || physical_col("campaign").expect("a name");
    for (body, expected) in [
        ("{col: campaign}", campaign()),
        ("{lit: {bigint: 5}}", Expr::Literal(Literal::BigInt(5))),
        (
            "42",
            Expr::Leaf(PhysicalLeaf::Untyped(UntypedLiteral(Literal::BigInt(42)))),
        ),
        ("{upper: {col: campaign}}", call("upper", vec![campaign()])),
    ] {
        let read = read(body).unwrap_or_else(|error| panic!("{body}: {error}"));
        assert_eq!(read, SemanticMappingValue::Expr(expected), "{body}");
    }
    assert!(matches!(
        read("{field: revenue}"),
        Err(ParseError::InvalidMappingValue(message))
            if message.contains("tag `field` not allowed at physical-mapping site")
    ));
}

#[test]
fn a_model_built_through_the_authoring_setters_serializes() {
    use semstrait_model::{
        ComplexExtras, Dataset, Grainset, LeafExtras, NestedDataset, SemanticInterface,
        SemanticModel, TemporalShape,
    };

    let order_ts = Dimension::builder("order_ts")
        .data_type(DataType::Timestamp { precision: 6 })
        .temporal([Granularity::Day])
        .build();
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .filter(predicate())
        .build();
    let region = Dimension::builder("region")
        .data_type(DataType::String)
        .categorical()
        .build();

    let interface = SemanticInterface::builder()
        .dimensions(vec![
            DimensionSite::r#ref("order_ts").with_key(),
            DimensionSite::r#ref("region"),
        ])
        .measures(vec![MeasureSite::r#ref("revenue")])
        .build();

    let extras = LeafExtras::builder()
        .temporal(TemporalShape::events("order_ts", Some(Granularity::Minute)))
        .build();
    let orders = Dataset::builder("orders")
        .extras(extras.clone())
        .semantic_interface(interface.clone())
        .filter(DataKindFilter::builder("recent").expr(predicate()).build())
        .build();
    let events = Grainset::builder("order_events")
        .extras(ComplexExtras::default())
        .dataset(
            NestedDataset::builder("returns")
                .extras(extras.clone())
                .build(),
        )
        .dataset(NestedDataset::builder("refunds").extras(extras).build())
        .semantic_interface(interface)
        .build();

    let (model, _diags) = SemanticModel::builder()
        .name("row-expr-authoring")
        .dataset(orders)
        .grainset(events)
        .dimension(order_ts)
        .dimension(region)
        .measure(revenue)
        .build()
        .expect("the model validates");

    assert_serializes("SemanticModel", &model);
}
