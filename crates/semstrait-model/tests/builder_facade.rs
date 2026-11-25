mod common;

use common::agg;
use semstrait_common::Granularity;
use semstrait_model::expr::field;
use semstrait_model::*;
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, Literal, UnaryOpKind};

#[test]
fn dimension_facade_temporal() {
    let d = Dimension::builder("ts")
        .data_type(DataType::Timestamp { precision: 6 })
        .temporal([Granularity::Day, Granularity::Hour])
        .build();
    assert!(matches!(d.dim_type, DimensionType::Temporal(_)));
}

#[test]
fn dimension_facade_categorical() {
    let d = Dimension::builder("country")
        .data_type(DataType::String)
        .categorical()
        .build();
    assert!(matches!(d.dim_type, DimensionType::Categorical));
}

#[test]
fn measure_facade_sum_full() {
    let m = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("sum"))
        .full()
        .build();
    assert_eq!(m.agg, agg("sum"));
    assert!(matches!(m.additivity, Some(AdditivityType::Full)));
}

#[test]
fn measure_facade_filter_predicate() {
    let m = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("count"))
        .filter(Expr::UnaryOp {
            op: UnaryOpKind::Not,
            operand: Box::new(Expr::IsNull(Box::new(field("amount").expect("a name")))),
        })
        .build();
    assert!(m.filter.is_some());
}

#[test]
fn metric_facade_sum_some_wrapper() {
    let m = Metric::builder("revenue_ytd")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .expr(field("revenue").expect("a name"))
        .agg(agg("sum"))
        .build();
    assert_eq!(m.agg, Some(agg("sum")));
}

#[test]
fn relationship_facade_on_key_and_many_to_one() {
    let r = Relationship::builder()
        .name("o_to_c")
        .from("orders")
        .to("customers")
        .on_key("customer_id")
        .many_to_one()
        .build()
        .expect("build clean");
    assert_eq!(r.on.declared().expect("declared basis").len(), 1);
    assert!(matches!(r.cardinality, Some(Cardinality::ManyToOne)));
}

#[test]
fn semantic_interface_per_item_inserters() {
    let iface = SemanticInterface::builder()
        .dimension(DimensionSite::r#ref("dim_a"))
        .measure(MeasureSite::r#ref("m_a"))
        .metric(MetricSite::r#ref("met_a"))
        .build();
    assert_eq!(iface.dimensions.len(), 1);
    assert_eq!(iface.measures.len(), 1);
    assert_eq!(iface.metrics.len(), 1);
}

#[test]
fn semantic_interface_projects_its_keysets_from_the_sites() {
    let iface = SemanticInterface::builder()
        .dimension(DimensionSite::r#ref("id").with_key())
        .dimension(
            DimensionSite::r#ref("sku")
                .with_keysets(["business"])
                .expect("named keysets"),
        )
        .build();
    let keysets = iface.keysets();
    assert_eq!(keysets["default"].len(), 1);
    assert_eq!(keysets["business"].len(), 1);
}

#[test]
fn semantic_mapping_entries_bulk_insert() {
    let m = SemanticMapping::builder()
        .entries([
            ("revenue", SemanticMappingValue::Column("amount".into())),
            (
                "currency",
                SemanticMappingValue::Literal(Literal::String("USD".into())),
            ),
        ])
        .build();
    let SemanticMapping::Explicit(e) = m else {
        panic!("expected Explicit")
    };
    assert_eq!(e.len(), 2);
}

#[test]
fn dataset_facade_path_and_format_rmw() {
    let d = Dataset::builder("orders")
        .catalog("polaris")
        .format(StorageFormat::Parquet)
        .path("s3://bucket/orders/")
        .build();
    let storage = d.body.base.extras.storage.as_ref().expect("storage set");
    assert_eq!(storage.format, Some(StorageFormat::Parquet));
    assert_eq!(storage.paths.len(), 1);
    assert_eq!(
        d.body
            .base
            .extras
            .catalog
            .as_ref()
            .expect("catalog set")
            .alias,
        "polaris",
    );
}

#[test]
fn dataset_facade_extras_then_path_rmw_preserves_other_fields() {
    let pre = LeafExtras::builder()
        .catalog(CatalogRef::new("polaris"))
        .build();
    let d = Dataset::builder("orders")
        .extras(pre)
        .path("s3://bucket/orders/")
        .build();
    assert_eq!(
        d.body
            .base
            .extras
            .catalog
            .as_ref()
            .expect("catalog preserved")
            .alias,
        "polaris",
    );
    assert_eq!(
        d.body
            .base
            .extras
            .storage
            .as_ref()
            .expect("storage set")
            .paths
            .len(),
        1,
    );
}

#[test]
fn dataset_facade_semantic_interface_then_dimension_rmw_preserves_other_fields() {
    let pre = SemanticInterface::builder()
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    let d = Dataset::builder("orders")
        .semantic_interface(pre)
        .dimension(DimensionSite::r#ref("order_ts"))
        .build();
    assert_eq!(
        d.semantic_interface.measures.len(),
        1,
        "pre.measures preserved"
    );
    assert_eq!(
        d.semantic_interface.dimensions.len(),
        1,
        "facade dimension added"
    );
}

#[test]
fn dataset_facade_dimension_and_keyness() {
    let d = Dataset::builder("orders")
        .dimension(DimensionSite::r#ref("order_ts").with_key())
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    assert_eq!(d.semantic_interface.dimensions.len(), 1);
    assert_eq!(d.semantic_interface.measures.len(), 1);
    assert_eq!(
        d.semantic_interface.keysets()["default"]
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["order_ts"]
    );
}

#[test]
fn nested_dataset_facade_format_and_paths() {
    let n = NestedDataset::builder("returns")
        .format(StorageFormat::Parquet)
        .paths(["s3://bucket/returns/2026/", "s3://bucket/returns/2025/"])
        .build();
    let storage = n.body.base.extras.storage.as_ref().expect("storage set");
    assert_eq!(storage.format, Some(StorageFormat::Parquet));
    assert_eq!(storage.paths.len(), 2);
}

#[test]
fn grainset_facade_temporal_and_dimension() {
    let nested_a = NestedDataset::builder("a").build();
    let nested_b = NestedDataset::builder("b").build();
    let g = Grainset::builder("evt")
        .dataset(nested_a)
        .dataset(nested_b)
        .temporal(TemporalShape::events("ts", Some(Granularity::Day)))
        .dimension(DimensionSite::r#ref("ts"))
        .build();
    assert!(g.body.base.extras.temporal.is_some());
    assert_eq!(g.semantic_interface.dimensions.len(), 1);
    assert_eq!(g.body.datasets.len(), 2);
}

#[test]
fn unionset_facade_union_all_and_union_unique() {
    let nested_a = NestedDataset::builder("a").build();
    let nested_b = NestedDataset::builder("b").build();
    let u_all = Unionset::builder("u_all")
        .dataset(nested_a.clone())
        .dataset(nested_b.clone())
        .union_all()
        .build();
    assert_eq!(u_all.body.mode, UnionMode::All);
    let u_uniq = Unionset::builder("u_uniq")
        .dataset(nested_a)
        .dataset(nested_b)
        .union_unique()
        .build();
    assert_eq!(u_uniq.body.mode, UnionMode::Unique);
}

#[test]
fn joinset_facade_dimension_and_temporal() {
    let nested_a = NestedDataset::builder("a").build();
    let nested_b = NestedDataset::builder("b").build();
    let j = Joinset::builder("j")
        .dataset(nested_a)
        .dataset(nested_b)
        .temporal(TemporalShape::events("ts", Some(Granularity::Day)))
        .dimension(DimensionSite::r#ref("ts"))
        .build();
    assert!(j.body.base.extras.temporal.is_some());
    assert_eq!(j.semantic_interface.dimensions.len(), 1);
}

#[test]
fn nested_grainset_facade_temporal_only() {
    let inner_a = NestedDataset::builder("a").build();
    let inner_b = NestedDataset::builder("b").build();
    let n = NestedGrainset::builder("ng")
        .dataset(inner_a)
        .dataset(inner_b)
        .temporal(TemporalShape::events("ts", Some(Granularity::Hour)))
        .build();
    assert!(n.body.base.extras.temporal.is_some());
    assert_eq!(n.body.datasets.len(), 2);
}

#[test]
fn nested_unionset_facade_union_all() {
    let inner_a = NestedDataset::builder("a").build();
    let inner_b = NestedDataset::builder("b").build();
    let n = NestedUnionset::builder("nu")
        .dataset(inner_a)
        .dataset(inner_b)
        .union_all()
        .build();
    assert_eq!(n.body.mode, UnionMode::All);
}

#[test]
fn nested_joinset_relationships_plural_inserter() {
    let r1 = Relationship::builder()
        .name("r1")
        .from("a")
        .to("b")
        .on_key("k")
        .many_to_one()
        .build()
        .expect("build clean");
    let r2 = Relationship::builder()
        .name("r2")
        .from("c")
        .to("d")
        .on_key("k")
        .many_to_one()
        .build()
        .expect("build clean");
    let inner_a = NestedDataset::builder("a").build();
    let inner_b = NestedDataset::builder("b").build();
    let nested = NestedJoinset::builder("j")
        .datasets([inner_a, inner_b])
        .relationships([r1, r2])
        .build();
    assert_eq!(nested.body.datasets.len(), 2);
    assert_eq!(nested.body.relationships.len(), 2);
}

#[test]
fn semantic_model_plural_inserters() {
    let d1 = Dimension::builder("d1")
        .data_type(DataType::String)
        .categorical()
        .build();
    let d2 = Dimension::builder("d2")
        .data_type(DataType::String)
        .categorical()
        .build();
    let m1 = Measure::builder("m1")
        .data_type(DataType::Integer)
        .agg(agg("count"))
        .build();
    let m2 = Measure::builder("m2")
        .data_type(DataType::Integer)
        .agg(agg("count"))
        .build();
    let ds1 = Dataset::builder("ds1")
        .dimension(DimensionSite::r#ref("d1"))
        .measure(MeasureSite::r#ref("m1"))
        .build();
    let ds2 = Dataset::builder("ds2")
        .dimension(DimensionSite::r#ref("d2"))
        .measure(MeasureSite::r#ref("m2"))
        .build();
    let (model, _) = SemanticModel::builder()
        .name("syms")
        .datasets([ds1, ds2])
        .dimensions([d1, d2])
        .measures([m1, m2])
        .build()
        .expect("build clean");
    assert_eq!(model.datasets.len(), 2);
    assert_eq!(model.dimensions.len(), 2);
    assert_eq!(model.measures.len(), 2);
}
