mod common;

use common::agg;
use semstrait_common::Granularity;
use semstrait_model::{
    AdditivityType, ComplexExtras, Dataset, Dimension, DimensionSite, DimensionType, Grainset,
    LeafExtras, Measure, MeasureSite, NestedDataset, Relationship, SemanticInterface,
    SemanticMapping, SemanticMappingValue, SemanticModel, TemporalShape,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::Literal;

#[test]
fn builds_minimal_model_with_dataset_and_grainset() {
    let order_ts = Dimension::builder("order_ts")
        .data_type(DataType::Timestamp { precision: 6 })
        .dim_type(DimensionType::temporal([
            Granularity::Minute,
            Granularity::Hour,
            Granularity::Day,
        ]))
        .build();

    let revenue = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("sum"))
        .additivity(AdditivityType::Full)
        .build();

    let extras = LeafExtras::builder()
        .semantic_mapping(
            SemanticMapping::builder()
                .column("revenue", "amount_cents")
                .build(),
        )
        .temporal(TemporalShape::events("order_ts", Some(Granularity::Minute)))
        .build();

    let interface = SemanticInterface::builder()
        .dimensions(vec![DimensionSite::r#ref("order_ts").with_key()])
        .measures(vec![MeasureSite::r#ref("revenue")])
        .build();

    let orders = Dataset::builder("orders")
        .extras(extras.clone())
        .description("Order-line fact dataset.")
        .semantic_interface(interface.clone())
        .build();

    let returns = NestedDataset::builder("returns")
        .extras(extras.clone())
        .build();
    let refunds = NestedDataset::builder("refunds").extras(extras).build();

    let order_events = Grainset::builder("order_events")
        .extras(ComplexExtras::default())
        .dataset(returns)
        .dataset(refunds)
        .description("Roll-up of order-side events.")
        .semantic_interface(interface.clone())
        .build();

    let orders_to_events = Relationship::builder()
        .name("orders_to_events")
        .from("orders")
        .to("order_events")
        .on_key("order_id")
        .cardinality(semstrait_model::Cardinality::ManyToOne)
        .build()
        .expect("relationship build");

    let (model, _diags) = SemanticModel::builder()
        .name("analytics-v1")
        .description("Smoke-test model.")
        .dataset(orders)
        .grainset(order_events)
        .dimension(order_ts)
        .measure(revenue)
        .relationship(orders_to_events)
        .build()
        .expect("validate clean");

    assert_eq!(model.name, "analytics-v1");
    assert_eq!(model.datasets.len(), 1);
    assert_eq!(model.grainsets.len(), 1);
    assert_eq!(model.relationships.len(), 1);
    assert!(model.find_public("orders").is_some());
    assert!(model.find_public("order_events").is_some());
}

#[test]
fn relationship_symmetric_cardinality_requires_optional_and_cross_filter() {
    let r = Relationship::builder()
        .name("one_to_one_missing")
        .from("a")
        .to("b")
        .on_key("k")
        .cardinality(semstrait_model::Cardinality::OneToOne)
        .build();

    assert!(r.is_err());
}

#[test]
fn loader_inmemory_round_trip_with_minimal_yaml() {
    let yaml = r#"
semantic_model:
  name: smoke-test
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#;

    let mut fs = semstrait_model::InMemoryFs::new();
    fs.insert("model.yaml", yaml);

    let (model, _diags) = SemanticModel::loader()
        .with_fs(fs)
        .from_yaml_file("model.yaml")
        .build()
        .expect("loader build clean");

    assert_eq!(model.name, "smoke-test");
    assert!(model.find_public("orders").is_some());
}

#[test]
fn loader_no_source_returns_no_source_diagnostic() {
    let result = SemanticModel::loader().build();
    assert!(result.is_err());
    let diags = result.unwrap_err();
    assert!(matches!(
        diags[0].kind,
        semstrait_model::ModelBuildErrorKind::NoSource
    ));
}

#[test]
fn semantic_mapping_with_semantic_builds_explicit_map() {
    let mapping = SemanticMapping::builder()
        .with_semantic(
            "revenue",
            SemanticMappingValue::Column("amount_cents".into()),
        )
        .with_semantic(
            "currency",
            SemanticMappingValue::Literal(Literal::String("USD".into())),
        )
        .build();

    let SemanticMapping::Explicit(entries) = mapping else {
        panic!("expected Explicit mapping, got {mapping:?}");
    };

    assert_eq!(entries.len(), 2);

    let mut iter = entries.iter();
    let (name_a, value_a) = iter.next().expect("first entry");
    assert_eq!(name_a.as_str(), "revenue");
    assert!(matches!(
        value_a,
        SemanticMappingValue::Column(col) if col == "amount_cents"
    ));

    let (name_b, value_b) = iter.next().expect("second entry");
    assert_eq!(name_b.as_str(), "currency");
    assert!(matches!(
        value_b,
        SemanticMappingValue::Literal(Literal::String(s)) if s == "USD"
    ));
}
