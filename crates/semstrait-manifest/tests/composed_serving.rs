mod common;
use common::golden::{self, Golden};
use common::{
    agg, aggregate, bodied, by_name, catalog, column, compile_diagnostic, compile_diagnostic_with,
    compile_model, compile_model_with, constituent, definition_id, dimension, filter_tree,
    has_signature, identity_mapping, lake_folder, link, link_free_tree, listed, measure,
    orders_model, refusal_golden, semantic_alias, signature, stored_expr, string, test_provenance,
    tree, warehouse_catalog, MapCatalog,
};

use std::collections::BTreeSet;

use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::Granularity;
use semstrait_manifest::{compile, mint_table_source_id, CompileError, ExprLayer};
use semstrait_model::expr::{col, dim, lit, measure as measure_leaf, metric as metric_leaf};
use semstrait_model::{
    Dataset, DimensionSite, Grainset, GrainsetBuilder, Joinset, Measure, MeasureSite, Metric,
    MetricSite, NestedDataset, NestedGrainset, NestedJoinset, NestedUnionset, Relationship,
    SemanticModel, SemanticModelBuilder, StorageFormat, TemporalShape, Unionset,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, Literal};

fn on_customer_id(from: &str, to: &str) -> Relationship {
    Relationship::builder()
        .name(format!("{from}_to_{to}"))
        .from(from)
        .to(to)
        .on(["customer_id"])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

fn txn_details_pools(extra: &str) -> SemanticModelBuilder {
    SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .dimension(dimension(extra, DataType::String))
        .measure(measure("revenue", DataType::Double, "sum"))
}

fn txn_details_model(extra: &str, joinset: Joinset) -> SemanticModel {
    txn_details_pools(extra)
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

fn auto_mapped(name: &str) -> NestedDataset {
    NestedDataset::builder(name)
        .table(format!("warehouse.{name}"))
        .build()
}

fn auto_unobservable(name: &str) -> NestedDataset {
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .build()
}

fn rollup_pools() -> SemanticModelBuilder {
    SemanticModel::builder()
        .name("sales")
        .dimension(dimension("region", DataType::String))
        .dimension(dimension("channel", DataType::String))
        .measure(measure("revenue", DataType::Double, "sum"))
}

fn rollup_model(unionset: Unionset) -> SemanticModel {
    rollup_pools()
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_dimension_mapped_on_one_operand_only_reads_from_that_operand() {
    let joinset = Joinset::builder("txn_details")
        .dataset(constituent(
            "orders_j",
            &["customer_id", "region", "revenue"],
        ))
        .dataset(constituent("customers_j", &["customer_id"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();

    let manifest = compile_model(txn_details_model("region", joinset)).await;
    signature(&manifest, "txn_details.orders_j", "region");
    let orders_j_alias = by_name(&manifest, "txn_details.orders_j").alias.as_str();
    assert_eq!(
        tree(&manifest, signature(&manifest, "txn_details", "region")),
        Expr::Column(
            semstrait_sql::expr::NamedReference::qualified(orders_j_alias, "region")
                .expect("a qualified name")
        )
    );
    assert!(
        !has_signature(&manifest, "txn_details.customers_j", "region"),
        "`customers_j` does not map `region`"
    );
}

#[tokio::test]
async fn a_composed_name_no_operand_maps_is_refused() {
    composed_name_unserved_refusal().await.assert_committed();
}

#[tokio::test]
async fn a_composed_metric_no_operand_maps_whole_is_lowered_on_the_joinset() {
    let manifest = compile_model(composed_metric_model()).await;
    let revenue = signature(&manifest, "txn_details_metric", "revenue").expression_id;
    let customers = signature(&manifest, "txn_details_metric", "customers").expression_id;

    assert_eq!(
        stored_expr(
            &manifest,
            signature(&manifest, "txn_details_metric", "revenue_per_customer").expression_id,
        ),
        &(link(revenue) / link(customers)),
        "the Metric links the Joinset's own `revenue` and `customers`"
    );
    for operand in [
        "txn_details_metric.orders_j",
        "txn_details_metric.customers_j",
    ] {
        assert!(
            !has_signature(&manifest, operand, "revenue_per_customer"),
            "`{operand}` does not map every name the body reads"
        );
    }
}

fn composed_metric_model() -> SemanticModel {
    let joinset = Joinset::builder("txn_details_metric")
        .dataset(constituent("orders_j", &["customer_id", "revenue"]))
        .dataset(constituent("customers_j", &["customer_id", "customers"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .measure(MeasureSite::r#ref("revenue"))
        .measure(MeasureSite::r#ref("customers"))
        .metric(MetricSite::r#ref("revenue_per_customer"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();
    SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .measure(measure("revenue", DataType::Double, "sum"))
        .measure(measure("customers", DataType::Double, "sum"))
        .metric(
            Metric::builder("revenue_per_customer")
                .data_type(DataType::Double)
                .expr(
                    measure_leaf("revenue").expect("a name")
                        / measure_leaf("customers").expect("a name"),
                )
                .build(),
        )
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_measure_no_operand_maps_whole_is_lowered_on_the_joinset() {
    let manifest = compile_model(vip_orders_model()).await;
    let orders = semantic_alias(&manifest, "orders_customers.orders_j");
    let customers = semantic_alias(&manifest, "orders_customers.customers_j");
    let vip = Expr::Case {
        whens: vec![(
            column(customers, "segment").eq(string("vip")),
            column(orders, "order_id"),
        )],
        else_: None,
    };

    assert_eq!(
        link_free_tree(
            &manifest,
            signature(&manifest, "orders_customers", "vip_orders").expression_id,
        ),
        aggregate("count", false, vip, None),
        "`segment` reads `customers_j`'s output field and `order_id` `orders_j`'s"
    );
    for operand in ["orders_customers.orders_j", "orders_customers.customers_j"] {
        assert!(
            !has_signature(&manifest, operand, "vip_orders"),
            "`{operand}` does not map every name the body reads"
        );
    }
}

#[tokio::test]
async fn a_metric_links_a_measure_lowered_on_the_joinset() {
    let manifest = compile_model(vip_orders_model()).await;
    let vip_orders = signature(&manifest, "orders_customers", "vip_orders").expression_id;

    assert_eq!(
        stored_expr(
            &manifest,
            signature(&manifest, "orders_customers", "vip_orders_twice").expression_id,
        ),
        &(link(vip_orders) + link(vip_orders)),
        "the Metric links the Joinset's own `vip_orders`"
    );
}

fn vip_orders_model() -> SemanticModel {
    let joinset = Joinset::builder("orders_customers")
        .dataset(constituent("orders_j", &["customer_id", "order_id"]))
        .dataset(constituent("customers_j", &["customer_id", "segment"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("order_id"))
        .dimension(DimensionSite::r#ref("segment"))
        .measure(MeasureSite::r#ref("vip_orders"))
        .metric(MetricSite::r#ref("vip_orders_twice"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();
    let vip_orders = || measure_leaf("vip_orders").expect("a name");
    let vip = Expr::Case {
        whens: vec![(
            dim("segment").expect("a name").eq(lit("vip")),
            dim("order_id").expect("a name"),
        )],
        else_: None,
    };
    SemanticModel::builder()
        .name("sales")
        .dimension(dimension("customer_id", DataType::String))
        .dimension(dimension("order_id", DataType::String))
        .dimension(dimension("segment", DataType::String))
        .measure(
            Measure::builder("vip_orders")
                .data_type(DataType::BigInt)
                .agg(agg("count"))
                .expr(vip)
                .build(),
        )
        .metric(
            Metric::builder("vip_orders_twice")
                .data_type(DataType::BigInt)
                .expr(vip_orders() + vip_orders())
                .build(),
        )
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

async fn unserved_refusal(
    case: &str,
    model: SemanticModel,
    semantics: &str,
    semantic_source: &str,
) -> Golden {
    let diagnostic = compile_diagnostic(model).await;
    match &diagnostic.kind {
        CompileError::UnservedComposedSemantics {
            semantics: got_semantics,
            semantic_source: got_source,
        } => {
            assert_eq!(got_semantics, semantics);
            assert_eq!(got_source, semantic_source);
        }
        other => panic!("expected UnservedComposedSemantics, got {other:?}"),
    }
    Golden::message(case, diagnostic.to_string())
}

async fn composed_name_unserved_refusal() -> Golden {
    let joinset = Joinset::builder("txn_details")
        .dataset(constituent("orders_j", &["customer_id", "revenue"]))
        .dataset(constituent("customers_j", &["customer_id"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();

    unserved_refusal(
        "a_composed_name_no_operand_maps_is_refused",
        txn_details_model("channel", joinset),
        "channel",
        "txn_details",
    )
    .await
}

#[tokio::test]
async fn a_join_basis_neither_operand_maps_is_refused() {
    join_basis_not_mapped_refusal().await.assert_committed();
}

async fn join_basis_not_mapped_refusal() -> Golden {
    let joinset = Joinset::builder("txn_details")
        .dataset(constituent("orders_j", &["revenue"]))
        .dataset(constituent("customers_j", &["region"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();

    let diagnostic = compile_diagnostic(txn_details_model("region", joinset)).await;
    match &diagnostic.kind {
        CompileError::JoinKeyNotMapped {
            relationship,
            key,
            semantic_source,
        } => {
            assert_eq!(relationship, "orders_j_to_customers_j");
            assert_eq!(key, "customer_id");
            assert_eq!(semantic_source, "txn_details.orders_j");
        }
        other => panic!("expected JoinKeyNotMapped, got {other:?}"),
    }
    Golden::message(
        "a_join_basis_neither_operand_maps_is_refused",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn a_join_basis_a_nested_operand_does_not_map_names_the_operand() {
    let txn_details = NestedJoinset::builder("txn_details")
        .dataset(constituent("orders_j", &["revenue"]))
        .dataset(constituent("customers_j", &["region"]))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();
    let all_txn = Unionset::builder("all_txn")
        .union_all()
        .dataset(constituent(
            "store_txn",
            &["customer_id", "region", "revenue"],
        ))
        .joinset(txn_details)
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    let model = txn_details_pools("region")
        .unionset(all_txn)
        .build()
        .expect("the fixture model must build")
        .0;

    let diagnostic = compile_diagnostic(model).await;

    assert_eq!(
        diagnostic.kind,
        CompileError::JoinKeyNotMapped {
            relationship: "orders_j_to_customers_j".to_owned(),
            key: "customer_id".to_owned(),
            semantic_source: "all_txn.txn_details.orders_j".to_owned(),
        }
    );
}

#[tokio::test]
async fn a_unionsets_exposed_dimension_no_branch_maps_is_refused() {
    unionset_dimension_unserved_refusal()
        .await
        .assert_committed();
}

async fn unionset_dimension_unserved_refusal() -> Golden {
    unionset_unserved_refusal(
        "a_unionsets_exposed_dimension_no_branch_maps_is_refused",
        &["region", "revenue"],
        "channel",
    )
    .await
}

#[tokio::test]
async fn a_unionsets_exposed_measure_no_branch_maps_is_refused() {
    unionset_measure_unserved_refusal().await.assert_committed();
}

async fn unionset_measure_unserved_refusal() -> Golden {
    unionset_unserved_refusal(
        "a_unionsets_exposed_measure_no_branch_maps_is_refused",
        &["region", "channel"],
        "revenue",
    )
    .await
}

async fn unionset_unserved_refusal(case: &str, mapped: &[&str], semantics: &str) -> Golden {
    let unionset = Unionset::builder("stacked")
        .dataset(constituent("east", mapped))
        .dataset(constituent("west", mapped))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .build();
    unserved_refusal(case, rollup_model(unionset), semantics, "stacked").await
}

#[tokio::test]
async fn an_implicit_unionsets_wrapper_invents_no_signature_for_a_name_no_table_maps() {
    let catalog = warehouse_catalog(&[
        ("east_a", &[("region", "string"), ("revenue", "double")]),
        ("east_b", &[("region", "string"), ("revenue", "double")]),
        (
            "west",
            &[
                ("region", "string"),
                ("channel", "string"),
                ("revenue", "double"),
            ],
        ),
    ]);
    let unionset = Unionset::builder("stacked")
        .dataset(
            NestedDataset::builder("east")
                .tables(["warehouse.east_a", "warehouse.east_b"])
                .build(),
        )
        .dataset(auto_mapped("west"))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .build();

    let manifest = compile_model_with(rollup_model(unionset), Some(&catalog)).await;
    assert!(
        !has_signature(&manifest, "stacked.east", "channel"),
        "neither of `east`'s tables maps `channel`"
    );
    signature(&manifest, "stacked", "channel");
}

#[tokio::test]
async fn a_generated_joinset_sub_composition_does_not_refuse_a_name_a_later_operand_maps() {
    let joinset = Joinset::builder("abc")
        .dataset(constituent("a", &["customer_id", "revenue"]))
        .dataset(constituent("b", &["customer_id"]))
        .dataset(constituent("c", &["customer_id", "channel"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(on_customer_id("a", "b"))
        .relationship(on_customer_id("b", "c"))
        .build();
    let model = txn_details_pools("channel")
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model(model).await;
    signature(&manifest, "abc", "channel");
}

#[tokio::test]
async fn auto_on_an_explicit_unionsets_constituent_missing_a_column_is_null_filled() {
    let catalog = warehouse_catalog(&[
        (
            "east",
            &[
                ("region", "string"),
                ("channel", "string"),
                ("revenue", "double"),
            ],
        ),
        ("west", &[("region", "string"), ("revenue", "double")]),
    ]);
    let unionset = Unionset::builder("stacked")
        .dataset(auto_mapped("east"))
        .dataset(auto_mapped("west"))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .build();

    let manifest = compile_model_with(rollup_model(unionset), Some(&catalog)).await;
    assert!(
        !has_signature(&manifest, "stacked.west", "channel"),
        "`west`'s catalog has no `channel`, so `auto` leaves it unmapped there"
    );
    assert!(
        has_signature(&manifest, "stacked.east", "channel"),
        "`east`'s catalog observes `channel`, so `auto` maps it"
    );
    signature(&manifest, "stacked", "channel");
}

#[tokio::test]
async fn auto_on_a_joinset_operand_missing_a_column_reads_from_the_other_operand() {
    let catalog = warehouse_catalog(&[
        (
            "orders_j",
            &[
                ("customer_id", "string"),
                ("channel", "string"),
                ("revenue", "double"),
            ],
        ),
        ("customers_j", &[("customer_id", "string")]),
    ]);
    let joinset = Joinset::builder("txn_details")
        .dataset(auto_mapped("orders_j"))
        .dataset(auto_mapped("customers_j"))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();

    let manifest = compile_model_with(txn_details_model("channel", joinset), Some(&catalog)).await;
    signature(&manifest, "txn_details.orders_j", "channel");
    signature(&manifest, "txn_details", "channel");
    assert!(
        !has_signature(&manifest, "txn_details.customers_j", "channel"),
        "`customers_j`'s catalog has no `channel`, so `auto` leaves it unmapped there"
    );
}

#[tokio::test]
async fn auto_on_a_root_dataset_missing_a_column_is_refused() {
    root_dataset_missing_column_refusal()
        .await
        .assert_committed();
}

async fn root_dataset_missing_column_refusal() -> Golden {
    let catalog = catalog(&[("region", "string"), ("revenue", "double")]);
    let model = orders_model(
        vec![
            dimension("region", DataType::String),
            dimension("channel", DataType::String),
        ],
        vec![measure("revenue", DataType::Double, "sum")],
        |b| b,
    );
    let diagnostic = compile_diagnostic_with(model, Some(&catalog)).await;
    match &diagnostic.kind {
        CompileError::ColumnNotInSourceSchema {
            column,
            physical_source_id,
        } => {
            assert_eq!(column, "channel");
            assert_eq!(
                *physical_source_id,
                mint_table_source_id("warehouse.orders")
            );
        }
        other => panic!("expected ColumnNotInSourceSchema, got {other:?}"),
    }
    Golden::message(
        "auto_on_a_root_dataset_missing_a_column_is_refused",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn auto_on_a_constituent_with_no_observed_schema_is_refused() {
    auto_without_schema_refusal().await.assert_committed();
}

fn no_schema_refusal(case: &str, diagnostic: Diagnostic<CompileError>, dataset: &str) -> Golden {
    match &diagnostic.kind {
        CompileError::AutoMappingWithoutSchema { dataset: got } => {
            assert_eq!(got, dataset);
        }
        other => panic!("expected AutoMappingWithoutSchema, got {other:?}"),
    }
    Golden::message(case, diagnostic.to_string())
}

async fn auto_without_schema_refusal() -> Golden {
    let joinset = Joinset::builder("txn_details")
        .dataset(constituent(
            "orders_j",
            &["customer_id", "region", "revenue"],
        ))
        .dataset(auto_unobservable("customers_j"))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(on_customer_id("orders_j", "customers_j"))
        .build();

    let diagnostic = compile_diagnostic(txn_details_model("region", joinset)).await;
    no_schema_refusal(
        "auto_on_a_constituent_with_no_observed_schema_is_refused",
        diagnostic,
        "txn_details.customers_j",
    )
}

#[tokio::test]
async fn a_no_schema_constituents_body_reading_a_column_is_refused() {
    no_schema_body_refusal().await.assert_committed();
}

async fn no_schema_body_refusal() -> Golden {
    let model = no_schema_unionset_model("region_from_column", col("region").expect("a name"));
    let diagnostic = compile_diagnostic_with(model, Some(&no_schema_catalog())).await;
    no_schema_refusal(
        "a_no_schema_constituents_body_reading_a_column_is_refused",
        diagnostic,
        "stacked.west",
    )
}

#[tokio::test]
async fn a_no_schema_constituent_is_refused_even_reading_no_column() {
    no_schema_literal_refusal().await.assert_committed();
}

async fn no_schema_literal_refusal() -> Golden {
    let model = no_schema_unionset_model("constant_dimension", lit("constant"));
    let diagnostic = compile_diagnostic_with(model, Some(&no_schema_catalog())).await;
    no_schema_refusal(
        "a_no_schema_constituent_is_refused_even_reading_no_column",
        diagnostic,
        "stacked.west",
    )
}

fn no_schema_catalog() -> MapCatalog {
    warehouse_catalog(&[("east", &[("region", "string")])])
}

fn no_schema_unionset_model(name: &str, expr: UnresolvedSemanticExpr) -> SemanticModel {
    let unionset = Unionset::builder("stacked")
        .dataset(auto_mapped("east"))
        .dataset(auto_unobservable("west"))
        .dimension(DimensionSite::r#ref(name))
        .union_all()
        .build();
    SemanticModel::builder()
        .name("sales")
        .dimension(bodied(name, DataType::String, expr))
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn an_implicit_unionsets_second_table_missing_a_mapped_column_is_refused() {
    implicit_unionset_missing_column_refusal()
        .await
        .assert_committed();
}

async fn implicit_unionset_missing_column_refusal() -> Golden {
    let (model, catalog) = two_table_orders(&[("region", "string")]);
    let diagnostic = compile_diagnostic_with(model, Some(&catalog)).await;
    match &diagnostic.kind {
        CompileError::ColumnNotInSourceSchema {
            column,
            physical_source_id,
        } => {
            assert_eq!(column, "revenue");
            assert_eq!(
                *physical_source_id,
                mint_table_source_id("warehouse.orders_2")
            );
        }
        other => panic!("expected ColumnNotInSourceSchema, got {other:?}"),
    }
    Golden::message(
        "an_implicit_unionsets_second_table_missing_a_mapped_column_is_refused",
        diagnostic.to_string(),
    )
}

fn two_table_orders(orders_2: &[(&str, &str)]) -> (SemanticModel, MapCatalog) {
    let catalog = warehouse_catalog(&[
        ("orders_1", &[("region", "string"), ("revenue", "double")]),
        ("orders_2", orders_2),
    ]);
    let dataset = Dataset::builder("orders")
        .tables(["warehouse.orders_1", "warehouse.orders_2"])
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(dimension("region", DataType::String))
        .measure(measure("revenue", DataType::Double, "sum"))
        .dataset(dataset)
        .build()
        .expect("the fixture model must build")
        .0;
    (model, catalog)
}

#[tokio::test]
async fn a_constituents_implicit_unionsets_reduction_leaf_missing_a_mapped_column_is_refused() {
    constituent_implicit_unionset_missing_column_refusal()
        .await
        .assert_committed();
}

async fn constituent_implicit_unionset_missing_column_refusal() -> Golden {
    let catalog = warehouse_catalog(&[
        (
            "east_a",
            &[
                ("region", "string"),
                ("channel", "string"),
                ("revenue", "double"),
            ],
        ),
        ("east_b", &[("region", "string"), ("revenue", "double")]),
        (
            "west",
            &[
                ("region", "string"),
                ("channel", "string"),
                ("revenue", "double"),
            ],
        ),
    ]);
    let unionset = Unionset::builder("stacked")
        .dataset(
            NestedDataset::builder("east")
                .tables(["warehouse.east_a", "warehouse.east_b"])
                .build(),
        )
        .dataset(auto_mapped("west"))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .build();

    let diagnostic = compile_diagnostic_with(rollup_model(unionset), Some(&catalog)).await;
    match &diagnostic.kind {
        CompileError::ColumnNotInSourceSchema {
            column,
            physical_source_id,
        } => {
            assert_eq!(column, "channel");
            assert_eq!(
                *physical_source_id,
                mint_table_source_id("warehouse.east_b")
            );
        }
        other => panic!("expected ColumnNotInSourceSchema, got {other:?}"),
    }
    Golden::message(
        "a_constituents_implicit_unionsets_reduction_leaf_missing_a_mapped_column_is_refused",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn validate_refuses_a_compiled_implicit_unionsets_reduction_leaf_missing_a_signature() {
    let (model, catalog) = two_table_orders(&[("region", "string"), ("revenue", "double")]);
    let mut manifest = compile_model_with(model, Some(&catalog)).await;

    let region = definition_id(&manifest, "region");
    let revenue_bit = manifest
        .semantic_definitions
        .by_name("revenue")
        .expect("the fixture declares `revenue`")
        .bit_position;
    let unionset_id = by_name(&manifest, "orders").id.clone();
    let leaf_id = by_name(&manifest, "orders_orders_1").id.clone();
    let signature_id = semstrait_manifest::mint_signature_id(leaf_id.as_str(), region.as_str());
    let leaf = manifest
        .semantic_sources
        .get_mut(&leaf_id)
        .expect("the compiled artifact holds `orders_orders_1`");
    leaf.semantic_signature_ids.retain(|id| *id != signature_id);
    leaf.semantic_bitmask = semstrait_manifest::SemanticBitmask::from_positions([revenue_bit]);
    let dropped = manifest
        .semantic_signatures
        .remove(&signature_id)
        .expect("the compiled artifact holds the signature");
    manifest.expressions.physical.remove(&dropped.expression_id);

    let faults = manifest
        .validate()
        .expect_err("the mutated artifact disagrees between reduction leaves");
    assert_eq!(faults.len(), 1, "expected exactly one fault: {faults:?}");
    assert!(
        matches!(
            &faults[0].kind,
            CompileError::ReductionLeafMissingSignature { semantic_source, leaf, semantics }
                if *semantic_source == unionset_id
                    && *leaf == leaf_id
                    && semantics == "region"
        ),
        "expected ReductionLeafMissingSignature, got {:?}",
        faults[0]
    );
}

fn stacked_model() -> SemanticModel {
    let is_web = bodied(
        "is_web",
        DataType::Boolean,
        ExprOperationExt::eq(dim("channel").expect("a name"), lit("web")),
    );
    let web_revenue = Measure::builder("web_revenue")
        .data_type(DataType::Double)
        .agg(agg("sum"))
        .filter(ExprOperationExt::eq(
            dim("channel").expect("a name"),
            lit("web"),
        ))
        .build();
    let unionset = Unionset::builder("stacked")
        .dataset(constituent(
            "east",
            &["region", "channel", "revenue", "web_revenue"],
        ))
        .dataset(constituent("west", &["region", "revenue", "web_revenue"]))
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("channel"))
        .dimension(DimensionSite::r#ref("is_web"))
        .measure(MeasureSite::r#ref("revenue"))
        .measure(MeasureSite::r#ref("web_revenue"))
        .union_all()
        .build();
    rollup_pools()
        .dimension(is_web)
        .measure(web_revenue)
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_body_reading_a_name_an_explicit_unionsets_branch_does_not_map_reads_a_typed_null() {
    let manifest = compile_model(stacked_model()).await;
    let west_is_web = tree(&manifest, signature(&manifest, "stacked.west", "is_web"));
    assert_eq!(
        west_is_web,
        Expr::Literal(Literal::Null(DataType::Boolean)),
        "`channel = 'web'` with `channel` a typed NULL folds to a NULL Boolean"
    );
    let east_is_web = tree(&manifest, signature(&manifest, "stacked.east", "is_web"));
    assert_ne!(
        east_is_web,
        Expr::Literal(Literal::Null(DataType::Boolean)),
        "`east` maps `channel`, so its `is_web` reads the column, not a NULL"
    );
}

#[tokio::test]
async fn a_filter_reading_a_name_an_explicit_unionsets_branch_does_not_map_reads_a_typed_null() {
    let manifest = compile_model(stacked_model()).await;
    let west_filter = filter_tree(
        &manifest,
        signature(&manifest, "stacked.west", "web_revenue"),
    );
    assert_eq!(
        west_filter,
        Expr::Literal(Literal::Null(DataType::Boolean)),
        "`channel = 'web'` with `channel` a typed NULL folds to a NULL Boolean"
    );
    let east_filter = filter_tree(
        &manifest,
        signature(&manifest, "stacked.east", "web_revenue"),
    );
    assert_ne!(
        east_filter,
        Expr::Literal(Literal::Null(DataType::Boolean)),
        "`east` maps `channel`, so its filter reads the column, not a NULL"
    );
}

fn grain_dataset(
    name: &str,
    event_time: &str,
    granularity: Granularity,
    names: &[&str],
) -> NestedDataset {
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(names))
        .temporal(TemporalShape::events(event_time, Some(granularity)))
        .build()
}

fn txn_by_grain_pools() -> SemanticModelBuilder {
    SemanticModel::builder()
        .name("sales")
        .dimension(dimension("event_date", DataType::Date))
        .dimension(dimension("country", DataType::String))
        .measure(measure("quantity", DataType::Double, "sum"))
        .measure(measure("revenue_target", DataType::Double, "sum"))
}

fn txn_by_grain(children: impl IntoIterator<Item = NestedDataset>) -> GrainsetBuilder {
    txn_by_grain_keyed(&["event_date", "country"], children)
}

fn txn_by_grain_keyed(
    keys: &[&str],
    children: impl IntoIterator<Item = NestedDataset>,
) -> GrainsetBuilder {
    let site = |name: &str| match keys.contains(&name) {
        true => DimensionSite::r#ref(name).with_key(),
        false => DimensionSite::r#ref(name),
    };
    (children.into_iter()).fold(
        Grainset::builder("txn_by_grain")
            .dimension(site("event_date"))
            .dimension(site("country"))
            .measure(MeasureSite::r#ref("quantity"))
            .measure(MeasureSite::r#ref("revenue_target")),
        |grainset, child| grainset.dataset(child),
    )
}

fn txn_by_grain_children() -> [NestedDataset; 2] {
    [
        grain_dataset(
            "transactions",
            "event_date",
            Granularity::Day,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset(
            "txn_monthly_agg",
            "event_date",
            Granularity::Month,
            &["event_date", "country", "revenue_target"],
        ),
    ]
}

fn txn_by_grain_model(metric_name: &str, metric: Metric) -> SemanticModel {
    txn_by_grain_model_over(txn_by_grain(txn_by_grain_children()), metric_name, metric)
}

fn txn_by_grain_model_over(
    grainset: GrainsetBuilder,
    metric_name: &str,
    metric: Metric,
) -> SemanticModel {
    let grainset = grainset.metric(MetricSite::r#ref(metric_name)).build();
    txn_by_grain_pools()
        .metric(metric)
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0
}

fn quantity_over_target() -> UnresolvedSemanticExpr {
    measure_leaf("quantity").expect("a name") / measure_leaf("revenue_target").expect("a name")
}

fn quantity_per_target() -> Metric {
    Metric::builder("quantity_per_target")
        .data_type(DataType::Double)
        .expr(quantity_over_target())
        .build()
}

fn quantity_per_target_us() -> Metric {
    Metric::builder("quantity_per_target_us")
        .data_type(DataType::Double)
        .expr(quantity_over_target())
        .filter(ExprOperationExt::eq(
            dim("country").expect("a name"),
            lit("US"),
        ))
        .build()
}

fn quantity_avg(name: &str, us: UnresolvedSemanticExpr) -> Metric {
    Metric::builder(name)
        .data_type(DataType::Double)
        .agg(agg("avg"))
        .expr(Expr::Case {
            whens: vec![(us, quantity_over_target() * lit(1.2))],
            else_: Some(Box::new(quantity_over_target())),
        })
        .build()
}

fn country_quantity_avg() -> Metric {
    quantity_avg(
        "country_quantity_avg",
        dim("country").expect("a name").eq(lit("US")),
    )
}

fn is_us() -> Metric {
    Metric::builder("is_us")
        .data_type(DataType::Boolean)
        .expr(dim("country").expect("a name").eq(lit("US")))
        .build()
}

fn is_us_quantity_avg() -> Metric {
    quantity_avg("is_us_quantity_avg", metric_leaf("is_us").expect("a name"))
}

fn txn_by_grain_model_reading_is_us(monthly_maps_country: bool) -> SemanticModel {
    let grainset = txn_by_grain_keyed(
        &["event_date"],
        txn_children_mapping_country(monthly_maps_country),
    )
    .metric(MetricSite::r#ref("is_us"))
    .metric(MetricSite::r#ref("is_us_quantity_avg"))
    .build();
    txn_by_grain_pools()
        .metric(is_us())
        .metric(is_us_quantity_avg())
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0
}

fn txn_children_mapping_country(monthly_maps_country: bool) -> [NestedDataset; 2] {
    let monthly: &[&str] = match monthly_maps_country {
        true => &["event_date", "country", "revenue_target"],
        false => &["event_date", "revenue_target"],
    };
    [
        grain_dataset(
            "transactions",
            "event_date",
            Granularity::Day,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset("txn_monthly_agg", "event_date", Granularity::Month, monthly),
    ]
}

#[tokio::test]
async fn a_grainset_metric_reading_a_name_no_child_maps_is_refused() {
    grainset_metric_reads_unmapped_refusal()
        .await
        .assert_committed();
}

async fn grainset_metric_reads_unmapped_refusal() -> Golden {
    let children = [
        grain_dataset(
            "transactions",
            "event_date",
            Granularity::Day,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset(
            "txn_weekly_agg",
            "event_date",
            Granularity::Week,
            &["event_date", "country", "quantity"],
        ),
    ];
    refusal_golden(
        "a_grainset_metric_reading_a_name_no_child_maps_is_refused",
        txn_by_grain_model_over(
            txn_by_grain(children),
            "quantity_per_target",
            quantity_per_target(),
        ),
        CompileError::GrainsetMetricReadsUnmapped {
            grainset: "txn_by_grain".to_owned(),
            metric: "quantity_per_target".to_owned(),
            name: "revenue_target".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_grainset_metric_over_children_that_share_no_keyset_is_refused() {
    grainset_metric_without_keyset_refusal()
        .await
        .assert_committed();
}

async fn grainset_metric_without_keyset_refusal() -> Golden {
    refusal_golden(
        "a_grainset_metric_over_children_that_share_no_keyset_is_refused",
        txn_by_grain_model_over(
            txn_by_grain_keyed(&[], txn_by_grain_children()),
            "quantity_per_target",
            quantity_per_target(),
        ),
        CompileError::GrainsetMetricWithoutKeyset {
            grainset: "txn_by_grain".to_owned(),
            metric: "quantity_per_target".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_grainset_metric_whose_formula_grain_one_serving_child_does_not_map_is_refused() {
    grainset_formula_grain_unmapped_refusal()
        .await
        .assert_committed();
}

async fn grainset_formula_grain_unmapped_refusal() -> Golden {
    refusal_golden(
        "a_grainset_metric_whose_formula_grain_one_serving_child_does_not_map_is_refused",
        txn_by_grain_model_over(
            txn_by_grain_keyed(&["event_date"], txn_children_mapping_country(false)),
            "country_quantity_avg",
            country_quantity_avg(),
        ),
        CompileError::GrainsetMetricWithoutKeyset {
            grainset: "txn_by_grain".to_owned(),
            metric: "country_quantity_avg".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_grainset_metric_whose_formula_grain_reads_a_row_level_metric_one_serving_child_does_not_map_is_refused(
) {
    grainset_formula_grain_row_level_unmapped_refusal()
        .await
        .assert_committed();
}

async fn grainset_formula_grain_row_level_unmapped_refusal() -> Golden {
    refusal_golden(
        "a_grainset_metric_whose_formula_grain_reads_a_row_level_metric_one_serving_child_does_not_map_is_refused",
        txn_by_grain_model_reading_is_us(false),
        CompileError::GrainsetMetricWithoutKeyset {
            grainset: "txn_by_grain".to_owned(),
            metric: "is_us_quantity_avg".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_grainset_metric_whose_formula_grain_each_serving_child_maps_is_computed() {
    let manifest = compile_model(txn_by_grain_model_over(
        txn_by_grain_keyed(&["event_date"], txn_children_mapping_country(true)),
        "country_quantity_avg",
        country_quantity_avg(),
    ))
    .await;
    let is_us = compile_model(txn_by_grain_model_reading_is_us(true)).await;

    signature(&manifest, "txn_by_grain", "country_quantity_avg");
    signature(&is_us, "txn_by_grain", "is_us_quantity_avg");
}

#[tokio::test]
async fn a_grainsets_children_of_one_grain_stay_refused_beside_a_computed_metric() {
    let children = [
        grain_dataset(
            "daily_a",
            "event_date",
            Granularity::Day,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset(
            "daily_b",
            "event_date",
            Granularity::Day,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset(
            "monthly",
            "event_date",
            Granularity::Month,
            &["event_date", "country", "revenue_target"],
        ),
    ];
    let model = txn_by_grain_model_over(
        txn_by_grain(children),
        "quantity_per_target",
        quantity_per_target(),
    );
    assert!(matches!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetGrainRepeated { .. }
    ));
}

#[tokio::test]
async fn a_grainsets_cross_grain_unit_does_not_refuse_a_name_a_third_child_maps() {
    let grainset = txn_by_grain([
        grain_dataset(
            "daily",
            "event_date",
            Granularity::Day,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset(
            "weekly",
            "event_date",
            Granularity::Week,
            &["event_date", "country", "quantity"],
        ),
        grain_dataset(
            "monthly",
            "event_date",
            Granularity::Month,
            &["event_date", "country", "revenue_target"],
        ),
    ])
    .build();
    let model = txn_by_grain_pools()
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;

    let manifest = compile_model(model).await;
    assert!(
        !has_signature(&manifest, "txn_by_grain.daily_weekly", "revenue_target"),
        "neither `daily` nor `weekly` maps `revenue_target`"
    );
    assert!(
        has_signature(&manifest, "txn_by_grain.daily_monthly", "revenue_target"),
        "`monthly` maps `revenue_target`"
    );
}

#[tokio::test]
async fn a_grainset_computes_a_metric_no_child_maps_whole_over_its_childrens_final_values() {
    let manifest = compile_model(txn_by_grain_model(
        "quantity_per_target",
        quantity_per_target(),
    ))
    .await;
    let grainset = by_name(&manifest, "txn_by_grain");
    let alias = grainset.alias.as_str();
    let computed = signature(&manifest, "txn_by_grain", "quantity_per_target");

    assert_eq!(
        tree(&manifest, computed),
        column(alias, "quantity") / column(alias, "revenue_target"),
        "the formula reads each child-served name as the Grainset's final value"
    );
    assert_eq!(
        manifest
            .expressions
            .get(&computed.expression_id)
            .map(|entry| entry.layer),
        Some(ExprLayer::PostAggregate),
        "a formula over final aggregates is post-aggregate"
    );
    assert_eq!(
        listed(&manifest, grainset).count(),
        1,
        "the Grainset lists a signature for the Metric it computes and for no other name"
    );
}

#[tokio::test]
async fn a_grainsets_exposed_dimension_no_child_maps_is_refused_naming_the_grainset() {
    grainset_dimension_unserved_refusal()
        .await
        .assert_committed();
}

async fn grainset_dimension_unserved_refusal() -> Golden {
    let grainset = txn_by_grain(txn_by_grain_children())
        .dimension(DimensionSite::r#ref("channel"))
        .build();
    let model = txn_by_grain_pools()
        .dimension(dimension("channel", DataType::String))
        .grainset(grainset)
        .build()
        .expect("the fixture model must build")
        .0;

    unserved_refusal(
        "a_grainsets_exposed_dimension_no_child_maps_is_refused_naming_the_grainset",
        model,
        "channel",
        "txn_by_grain",
    )
    .await
}

#[tokio::test]
async fn a_grainsets_computed_metrics_own_filter_is_stored_over_final_values() {
    let manifest = compile_model(txn_by_grain_model(
        "quantity_per_target_us",
        quantity_per_target_us(),
    ))
    .await;
    let alias = semantic_alias(&manifest, "txn_by_grain");
    let computed = signature(&manifest, "txn_by_grain", "quantity_per_target_us");

    assert_eq!(
        filter_tree(&manifest, computed),
        ExprOperationExt::eq(column(alias, "country"), string("US")),
        "the Metric's own filter reads the Grainset's final value of `country`"
    );
    assert_eq!(
        manifest
            .expressions
            .get(&computed.filter_expr_id.expect("a filter"))
            .map(|entry| entry.layer),
        Some(ExprLayer::Scalar),
        "a filter stays row-level"
    );
}

fn spend_pools() -> SemanticModelBuilder {
    spend_pools_with_budget_filter(None)
}

fn spend_pools_with_budget_filter(filter: Option<UnresolvedSemanticExpr>) -> SemanticModelBuilder {
    let budget = match filter {
        Some(filter) => Measure::builder("budget")
            .data_type(DataType::Double)
            .agg(agg("sum"))
            .filter(filter)
            .build(),
        None => measure("budget", DataType::Double, "sum"),
    };
    SemanticModel::builder()
        .name("ads")
        .dimension(dimension("date", DataType::Date))
        .dimension(dimension("campaign", DataType::String))
        .dimension(dimension("channel", DataType::String))
        .measure(measure("clicks", DataType::Double, "sum"))
        .measure(budget)
}

fn web_budget() -> Metric {
    Metric::builder("web_budget")
        .data_type(DataType::Double)
        .expr(measure_leaf("budget").expect("a name"))
        .filter(ExprOperationExt::eq(
            dim("channel").expect("a name"),
            lit("web"),
        ))
        .build()
}

fn cpc() -> Metric {
    Metric::builder("cpc")
        .data_type(DataType::Double)
        .expr(measure_leaf("budget").expect("a name") / measure_leaf("clicks").expect("a name"))
        .build()
}

fn cpm() -> Metric {
    Metric::builder("cpm")
        .data_type(DataType::Double)
        .expr(metric_leaf("cpc").expect("a name") * lit(1000.0))
        .build()
}

fn avg_cpc() -> Metric {
    Metric::builder("avg_cpc")
        .data_type(DataType::Double)
        .agg(agg("avg"))
        .expr(measure_leaf("budget").expect("a name") / measure_leaf("clicks").expect("a name"))
        .build()
}

fn web_cpc2() -> Metric {
    Metric::builder("web_cpc2")
        .data_type(DataType::Double)
        .expr(metric_leaf("web_budget").expect("a name") / measure_leaf("clicks").expect("a name"))
        .build()
}

fn spend_children(monthly_maps_channel: bool) -> [NestedDataset; 2] {
    let mut monthly_names = vec!["date", "campaign", "budget"];
    if monthly_maps_channel {
        monthly_names.push("channel");
    }
    [
        grain_dataset(
            "spend_daily",
            "date",
            Granularity::Day,
            &["date", "campaign", "clicks", "channel"],
        ),
        grain_dataset("spend_monthly", "date", Granularity::Month, &monthly_names),
    ]
}

fn spend_with(
    region: bool,
    metrics: Vec<Metric>,
    children: impl IntoIterator<Item = NestedDataset>,
) -> SemanticModelBuilder {
    spend_over(region, metrics, children, Vec::new())
}

fn spend_over(
    region: bool,
    metrics: Vec<Metric>,
    children: impl IntoIterator<Item = NestedDataset>,
    unionsets: Vec<NestedUnionset>,
) -> SemanticModelBuilder {
    let mut grainset = Grainset::builder("spend")
        .dimension(DimensionSite::r#ref("date").with_key())
        .dimension(DimensionSite::r#ref("campaign").with_key())
        .dimension(DimensionSite::r#ref("channel"));
    let mut pools = spend_pools();
    if region {
        grainset = grainset.dimension(DimensionSite::r#ref("region"));
        pools = pools.dimension(dimension("region", DataType::String));
    }
    grainset = grainset
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"));
    for metric in &metrics {
        grainset = grainset.metric(MetricSite::r#ref(metric.body.name.as_str()));
    }
    (metrics.into_iter())
        .fold(pools, SemanticModelBuilder::metric)
        .grainset(grainset.datasets(children).unionsets(unionsets).build())
}

fn spend_joinset(monthly_maps_channel: bool) -> Joinset {
    Joinset::builder("j")
        .grainset(
            NestedGrainset::builder("spend")
                .temporal(TemporalShape::events("date", Some(Granularity::Month)))
                .datasets(spend_children(monthly_maps_channel))
                .build(),
        )
        .dataset(constituent("campaigns", &["campaign"]))
        .dimension(DimensionSite::r#ref("campaign").with_key())
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .metric(MetricSite::r#ref("web_budget"))
        .metric(MetricSite::r#ref("web_cpc2"))
        .relationship(
            Relationship::builder()
                .name("spend_to_campaigns")
                .from("spend")
                .to("campaigns")
                .on(["campaign"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build()
}

fn spend_model(monthly_maps_channel: bool) -> SemanticModel {
    let metrics = vec![web_budget(), cpc(), cpm(), web_cpc2(), avg_cpc()];
    spend_with(false, metrics, spend_children(monthly_maps_channel))
        .joinset(spend_joinset(monthly_maps_channel))
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_grainsets_child_signature_also_reads_the_names_its_filter_reads() {
    grainset_child_filter_unserved_refusal()
        .await
        .assert_committed();
}

async fn grainset_child_filter_unserved_refusal() -> Golden {
    refusal_golden(
        "a_grainsets_child_signature_also_reads_the_names_its_filter_reads",
        spend_model(false),
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend".to_owned(),
            metric: "web_budget".to_owned(),
            measure: "budget".to_owned(),
            dimension: "channel".to_owned(),
        },
    )
    .await
}

fn web_cpc_in_eu() -> Metric {
    Metric::builder("web_cpc_in_eu")
        .data_type(DataType::Double)
        .expr(metric_leaf("web_budget").expect("a name") / measure_leaf("clicks").expect("a name"))
        .filter(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("EU"),
        ))
        .build()
}

fn spend_model_with_a_region_filter(monthly_maps_region: bool) -> SemanticModel {
    let mut monthly_names = vec!["date", "campaign", "budget", "channel"];
    if monthly_maps_region {
        monthly_names.push("region");
    }
    let children = [
        grain_dataset(
            "spend_daily",
            "date",
            Granularity::Day,
            &["date", "campaign", "clicks", "channel", "region"],
        ),
        grain_dataset("spend_monthly", "date", Granularity::Month, &monthly_names),
    ];
    spend_with(true, vec![web_budget(), web_cpc_in_eu()], children)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_metric_that_one_child_maps_whole_is_checked_under_the_filter_of_the_metric_that_reads_it(
) {
    metric_child_whole_filter_unserved_refusal()
        .await
        .assert_committed();
}

async fn metric_child_whole_filter_unserved_refusal() -> Golden {
    refusal_golden(
        "a_metric_that_one_child_maps_whole_is_checked_under_the_filter_of_the_metric_that_reads_it",
        spend_model_with_a_region_filter(false),
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend".to_owned(),
            metric: "web_cpc_in_eu".to_owned(),
            measure: "web_budget".to_owned(),
            dimension: "region".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_metric_that_one_child_maps_whole_compiles_where_that_child_maps_the_filter_too() {
    let manifest = compile_model(spend_model_with_a_region_filter(true)).await;
    let listed_names: BTreeSet<&str> = (listed(&manifest, by_name(&manifest, "spend")))
        .map(|signature| manifest.semantic_definitions.get(&signature.definition_id))
        .map(|definition| definition.expect("a definition").name.as_str())
        .collect();
    assert_eq!(listed_names, BTreeSet::from(["web_cpc_in_eu"]));
}

fn web_eu_budget(first: (&str, &str), second: (&str, &str)) -> Metric {
    let reads = |(dimension, value): (&str, &str)| {
        ExprOperationExt::eq(dim(dimension).expect("a name"), lit(value))
    };
    Metric::builder("web_eu_budget")
        .data_type(DataType::Double)
        .expr(measure_leaf("budget").expect("a name"))
        .filter(Expr::BinaryOp {
            op: BinaryOpKind::And,
            left: Box::new(reads(first)),
            right: Box::new(reads(second)),
        })
        .build()
}

fn spend_model_with_a_filter_on_two_dimensions(
    first: (&str, &str),
    second: (&str, &str),
) -> SemanticModel {
    let children = [
        grain_dataset(
            "spend_monthly",
            "date",
            Granularity::Month,
            &["date", "campaign", "clicks", "budget", "channel"],
        ),
        grain_dataset(
            "spend_weekly",
            "date",
            Granularity::Week,
            &["date", "campaign", "budget", "region"],
        ),
    ];
    spend_with(true, vec![web_eu_budget(first, second)], children)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_filter_on_two_dimensions_that_no_child_maps_together_is_refused_naming_the_first_unmapped(
) {
    filter_dimensions_unmapped_together_refusal()
        .await
        .assert_committed();
}

async fn filter_dimensions_unmapped_together_refusal() -> Golden {
    refusal_golden(
        "a_filter_on_two_dimensions_that_no_child_maps_together_is_refused_naming_the_first_unmapped",
        spend_model_with_a_filter_on_two_dimensions(("channel", "web"), ("region", "eu")),
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend".to_owned(),
            metric: "web_eu_budget".to_owned(),
            measure: "budget".to_owned(),
            dimension: "region".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn the_dimension_a_filter_refusal_names_follows_the_filters_order_not_the_alphabet() {
    filter_dimensions_reversed_refusal()
        .await
        .assert_committed();
}

async fn filter_dimensions_reversed_refusal() -> Golden {
    refusal_golden(
        "the_dimension_a_filter_refusal_names_follows_the_filters_order_not_the_alphabet",
        spend_model_with_a_filter_on_two_dimensions(("region", "eu"), ("channel", "web")),
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend".to_owned(),
            metric: "web_eu_budget".to_owned(),
            measure: "budget".to_owned(),
            dimension: "channel".to_owned(),
        },
    )
    .await
}

fn cpc_case(when: UnresolvedSemanticExpr) -> UnresolvedSemanticExpr {
    Expr::Case {
        whens: vec![(
            when,
            measure_leaf("budget").expect("a name") / measure_leaf("clicks").expect("a name"),
        )],
        else_: Some(Box::new(lit(0.5))),
    }
}

fn cpc_where(when: UnresolvedSemanticExpr, name: &str) -> Metric {
    Metric::builder(name)
        .data_type(DataType::Double)
        .expr(cpc_case(when))
        .build()
}

fn avg_cpc_where(when: UnresolvedSemanticExpr, name: &str) -> Metric {
    Metric::builder(name)
        .data_type(DataType::Double)
        .agg(agg("avg"))
        .expr(cpc_case(when))
        .build()
}

fn avg_web_only_cpc() -> Metric {
    let web = ExprOperationExt::eq(dim("channel").expect("a name"), lit("web"));
    avg_cpc_where(web, "avg_web_only_cpc")
}

fn avg_is_web_cpc() -> Metric {
    avg_cpc_where(metric_leaf("is_web").expect("a name"), "avg_is_web_cpc")
}

fn web_only_cpc() -> Metric {
    let web = ExprOperationExt::eq(dim("channel").expect("a name"), lit("web"));
    cpc_where(web, "web_only_cpc")
}

fn before_mid_march() -> UnresolvedSemanticExpr {
    (dim("date").expect("a name")).lt(Expr::Literal(Literal::Date("2026-03-05".to_owned())))
}

fn early_cpc() -> Metric {
    cpc_where(before_mid_march(), "early_cpc")
}

fn is_early() -> Metric {
    Metric::builder("is_early")
        .data_type(DataType::Boolean)
        .expr(before_mid_march())
        .build()
}

fn is_early_cpc() -> Metric {
    cpc_where(metric_leaf("is_early").expect("a name"), "is_early_cpc")
}

fn avg_early_cpc() -> Metric {
    avg_cpc_where(before_mid_march(), "avg_early_cpc")
}

fn avg_is_early_cpc() -> Metric {
    avg_cpc_where(metric_leaf("is_early").expect("a name"), "avg_is_early_cpc")
}

fn spend_children_at_month() -> [NestedDataset; 2] {
    let [_, monthly] = spend_children(true);
    let clicks = grain_dataset(
        "spend_clicks",
        "date",
        Granularity::Month,
        &["date", "campaign", "clicks", "channel"],
    );
    [clicks, monthly]
}

fn spend_with_a_part_key(metrics: Vec<Metric>, children: [NestedDataset; 2]) -> SemanticModel {
    spend_with(false, metrics, children)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_part_key_that_reads_a_dimension_the_child_of_a_measure_does_not_map_is_refused() {
    part_key_unmapped_refusal().await.assert_committed();
}

async fn part_key_unmapped_refusal() -> Golden {
    refusal_golden(
        "a_part_key_that_reads_a_dimension_the_child_of_a_measure_does_not_map_is_refused",
        spend_with_a_part_key(vec![web_only_cpc()], spend_children(false)),
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "web_only_cpc".to_owned(),
            measure: "budget".to_owned(),
            dimension: "channel".to_owned(),
            child: "spend.spend_monthly".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_part_key_that_reads_the_time_axis_finer_than_the_child_of_a_measure_maps_it_is_refused()
{
    part_key_coarser_refusal().await.assert_committed();
}

async fn part_key_coarser_refusal() -> Golden {
    refusal_golden(
        "a_part_key_that_reads_the_time_axis_finer_than_the_child_of_a_measure_maps_it_is_refused",
        spend_with_a_part_key(vec![early_cpc()], spend_children(true)),
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "early_cpc".to_owned(),
            measure: "budget".to_owned(),
            dimension: "date".to_owned(),
            child: "spend.spend_monthly".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_part_key_whose_row_level_metric_reads_the_time_axis_finer_than_the_child_of_a_measure_maps_it_is_refused(
) {
    part_key_row_level_coarser_refusal()
        .await
        .assert_committed();
}

async fn part_key_row_level_coarser_refusal() -> Golden {
    refusal_golden(
        "a_part_key_whose_row_level_metric_reads_the_time_axis_finer_than_the_child_of_a_measure_maps_it_is_refused",
        spend_with_a_part_key(vec![is_early(), is_early_cpc()], spend_children(true)),
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "is_early_cpc".to_owned(),
            measure: "budget".to_owned(),
            dimension: "date".to_owned(),
            child: "spend.spend_monthly".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_part_key_that_each_child_of_a_measure_forms_is_computed() {
    let daily_budget = grain_dataset(
        "spend_budget",
        "date",
        Granularity::Day,
        &["date", "campaign", "budget", "channel"],
    );
    let [daily, _] = spend_children(true);
    let daily_children = || [daily.clone(), daily_budget.clone()];
    let early = compile_model(spend_with_a_part_key(vec![early_cpc()], daily_children())).await;
    let is_early = compile_model(spend_with_a_part_key(
        vec![is_early(), is_early_cpc()],
        daily_children(),
    ))
    .await;
    let web = compile_model(spend_with_a_part_key(
        vec![web_only_cpc()],
        spend_children(true),
    ))
    .await;

    signature(&early, "spend", "early_cpc");
    signature(&is_early, "spend", "is_early_cpc");
    signature(&web, "spend", "web_only_cpc");
}

#[tokio::test]
async fn a_grainset_metric_whose_formula_grain_reads_the_time_axis_finer_than_the_child_of_a_measure_maps_it_is_refused(
) {
    formula_grain_coarser_refusal().await.assert_committed();
    assert_eq!(
        compile_diagnostic(spend_with_a_part_key(
            vec![is_early(), avg_is_early_cpc()],
            spend_children(true),
        ))
        .await
        .kind,
        CompileError::GrainsetMetricWithoutKeyset {
            grainset: "spend".to_owned(),
            metric: "avg_is_early_cpc".to_owned(),
        },
        "`is_early` is row-level, so the formula grain reads `date` through it"
    );
}

async fn formula_grain_coarser_refusal() -> Golden {
    refusal_golden(
        "a_grainset_metric_whose_formula_grain_reads_the_time_axis_finer_than_the_child_of_a_measure_maps_it_is_refused",
        spend_with_a_part_key(vec![avg_early_cpc()], spend_children(true)),
        CompileError::GrainsetMetricWithoutKeyset {
            grainset: "spend".to_owned(),
            metric: "avg_early_cpc".to_owned(),
        },
    )
    .await
}

#[tokio::test]
async fn a_grainset_metric_whose_formula_grain_reads_the_time_axis_that_each_child_of_a_measure_maps_at_month_is_computed(
) {
    let manifest = compile_model(spend_with_a_part_key(
        vec![is_early(), avg_early_cpc(), avg_is_early_cpc()],
        spend_children_at_month(),
    ))
    .await;

    signature(&manifest, "spend", "avg_early_cpc");
    signature(&manifest, "spend", "avg_is_early_cpc");
}

fn monthly_unionset(temporal: Option<TemporalShape>, unique: bool) -> NestedUnionset {
    let branch = |name: &str| {
        grain_dataset(
            name,
            "date",
            Granularity::Month,
            &["date", "campaign", "budget", "channel"],
        )
    };
    let unionset = NestedUnionset::builder("spend_monthly")
        .dataset(branch("monthly_a"))
        .dataset(branch("monthly_b"));
    let unionset = match temporal {
        Some(temporal) => unionset.temporal(temporal),
        None => unionset,
    };
    match unique {
        true => unionset.union_unique().build(),
        false => unionset.union_all().build(),
    }
}

type SpendMonthlyForm = (String, Vec<NestedDataset>, Vec<NestedUnionset>);

fn spend_monthly_forms() -> Vec<SpendMonthlyForm> {
    let [daily, monthly] = spend_children(true);
    let mut forms = vec![(
        "a `spend_monthly` Dataset at Month".to_owned(),
        vec![daily.clone(), monthly],
        Vec::new(),
    )];
    let stated = [
        (
            "Month",
            Some(TemporalShape::events("date", Some(Granularity::Month))),
        ),
        (
            "the time axis without a granularity",
            Some(TemporalShape::events("date", None)),
        ),
        ("no temporal shape", None),
    ];
    for (states, temporal) in stated {
        for (mode, unique) in [("all", false), ("unique", true)] {
            forms.push((
                format!("a `spend_monthly` Unionset of union {mode} that states {states}"),
                vec![daily.clone()],
                vec![monthly_unionset(temporal.clone(), unique)],
            ));
        }
    }
    forms
}

type SpendMonthlyOutcome = (String, Option<CompileError>, Result<bool, CompileError>);

async fn over_each_spend_monthly_form(
    metrics: fn() -> Vec<Metric>,
    name: &str,
) -> Vec<SpendMonthlyOutcome> {
    let mut outcomes = Vec::new();
    for (form, children, unionsets) in spend_monthly_forms() {
        let top = spend_over(false, metrics(), children.clone(), unionsets.clone())
            .build()
            .expect("the fixture model must build")
            .0;
        let nested = spend_beside_j_over(metrics(), spend_children_at_month(), children, unionsets);
        let fs = InMemoryFileSystem::new();
        let refused = (compile(top, None, &fs, test_provenance()).await)
            .err()
            .map(|(diagnostic, _)| diagnostic.kind);
        let nested_computes = (compile(nested, None, &fs, test_provenance()).await)
            .map(|(manifest, _)| has_signature(&manifest, "j.spend", name))
            .map_err(|(diagnostic, _)| diagnostic.kind);
        outcomes.push((form, refused, nested_computes));
    }
    outcomes
}

#[tokio::test]
async fn a_part_key_reads_the_time_axis_of_a_composed_child_of_a_measure_at_the_granularity_it_derives(
) {
    let outcomes = over_each_spend_monthly_form(|| vec![early_cpc()], "early_cpc").await;
    let refusal = CompileError::GrainsetMetricPartKeyUnformed {
        grainset: "spend".to_owned(),
        metric: "early_cpc".to_owned(),
        measure: "budget".to_owned(),
        dimension: "date".to_owned(),
        child: "spend.spend_monthly".to_owned(),
    };
    let expected: Vec<SpendMonthlyOutcome> = (outcomes.iter())
        .map(|(form, ..)| (form.clone(), Some(refusal.clone()), Ok(false)))
        .collect();

    assert_eq!(
        outcomes, expected,
        "each `spend_monthly` gives `budget` with `date` at Month and `spend_daily` gives \
         `clicks` with `date` at Day, so `spend` refuses the part key `date` and the nested \
         `spend` computes nothing"
    );
}

#[tokio::test]
async fn a_formula_grain_reads_the_time_axis_of_a_composed_child_of_a_measure_at_the_granularity_it_derives(
) {
    let outcomes = over_each_spend_monthly_form(|| vec![avg_early_cpc()], "avg_early_cpc").await;
    let refusal = CompileError::GrainsetMetricWithoutKeyset {
        grainset: "spend".to_owned(),
        metric: "avg_early_cpc".to_owned(),
    };
    let expected: Vec<SpendMonthlyOutcome> = (outcomes.iter())
        .map(|(form, ..)| (form.clone(), Some(refusal.clone()), Ok(false)))
        .collect();

    assert_eq!(
        outcomes, expected,
        "each `spend_monthly` gives `budget` with `date` at Month and `spend_daily` gives \
         `clicks` with `date` at Day, so `spend` refuses the formula grain `date` and the \
         nested `spend` computes nothing"
    );
}

fn in_eu() -> UnresolvedSemanticExpr {
    ExprOperationExt::eq(dim("region").expect("a name"), lit("EU"))
}

fn eu_cpc() -> Metric {
    cpc_where(in_eu(), "eu_cpc")
}

fn early_eu_cpc() -> Metric {
    let early_in_eu = Expr::BinaryOp {
        op: BinaryOpKind::And,
        left: Box::new(in_eu()),
        right: Box::new(before_mid_march()),
    };
    cpc_where(early_in_eu, "early_eu_cpc")
}

fn web_only_cpc_in_eu() -> Metric {
    let web = ExprOperationExt::eq(dim("channel").expect("a name"), lit("web"));
    Metric::builder("web_only_cpc_in_eu")
        .data_type(DataType::Double)
        .expr(cpc_case(web))
        .filter(in_eu())
        .build()
}

fn regional_clicks_children() -> [NestedDataset; 3] {
    [
        grain_dataset(
            "spend_daily",
            "date",
            Granularity::Day,
            &["date", "campaign", "clicks", "channel"],
        ),
        grain_dataset(
            "spend_clicks",
            "date",
            Granularity::Month,
            &["date", "campaign", "clicks", "region"],
        ),
        grain_dataset(
            "spend_monthly",
            "date",
            Granularity::Month,
            &["date", "campaign", "budget", "channel", "region"],
        ),
    ]
}

fn spend_with_a_regional_clicks_child(metrics: Vec<Metric>) -> SemanticModel {
    spend_with(true, metrics, regional_clicks_children())
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_part_key_that_one_child_of_a_measure_cannot_form_is_computed_from_the_child_that_forms_it(
) {
    let manifest = compile_model(spend_with_a_regional_clicks_child(vec![eu_cpc()])).await;
    signature(&manifest, "spend", "eu_cpc");
}

#[tokio::test]
async fn a_part_keys_grain_comes_only_from_the_children_that_form_the_key() {
    let manifest = compile_model(spend_with_a_regional_clicks_child(vec![early_eu_cpc()])).await;
    signature(&manifest, "spend", "early_eu_cpc");
}

#[tokio::test]
async fn a_part_key_refusal_names_the_child_that_maps_the_value_under_its_filter_but_not_the_key() {
    part_key_unmapped_under_filter_refusal()
        .await
        .assert_committed();
}

async fn part_key_unmapped_under_filter_refusal() -> Golden {
    refusal_golden(
        "a_part_key_refusal_names_the_child_that_maps_the_value_under_its_filter_but_not_the_key",
        spend_with_a_regional_clicks_child(vec![web_only_cpc_in_eu()]),
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "web_only_cpc_in_eu".to_owned(),
            measure: "clicks".to_owned(),
            dimension: "channel".to_owned(),
            child: "spend.spend_clicks".to_owned(),
        },
    )
    .await
}

fn is_web_cpc_in_eu() -> Metric {
    Metric::builder("is_web_cpc_in_eu")
        .data_type(DataType::Double)
        .expr(cpc_case(metric_leaf("is_web").expect("a name")))
        .filter(in_eu())
        .build()
}

fn spend_with_a_regional_budget_child(metrics: Vec<Metric>) -> SemanticModel {
    let children = [
        grain_dataset(
            "spend_daily",
            "date",
            Granularity::Day,
            &["date", "campaign", "clicks", "channel"],
        ),
        grain_dataset(
            "spend_regional",
            "date",
            Granularity::Month,
            &["date", "campaign", "budget", "clicks", "region"],
        ),
    ];
    spend_with(true, metrics, children)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_part_key_that_reads_a_row_level_metric_under_a_filter_names_the_measure_whose_child_cannot_form_it(
) {
    part_key_row_level_under_filter_refusal()
        .await
        .assert_committed();
    assert_eq!(
        compile_diagnostic(spend_with_a_regional_budget_child(vec![
            web_only_cpc_in_eu()
        ]))
        .await
        .kind,
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "web_only_cpc_in_eu".to_owned(),
            measure: "budget".to_owned(),
            dimension: "channel".to_owned(),
            child: "spend.spend_regional".to_owned(),
        },
        "the same Metric reading `channel` gets the same refusal"
    );
}

async fn part_key_row_level_under_filter_refusal() -> Golden {
    refusal_golden(
        "a_part_key_that_reads_a_row_level_metric_under_a_filter_names_the_measure_whose_child_cannot_form_it",
        spend_with_a_regional_budget_child(vec![is_web(), is_web_cpc_in_eu()]),
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "is_web_cpc_in_eu".to_owned(),
            measure: "budget".to_owned(),
            dimension: "is_web".to_owned(),
            child: "spend.spend_regional".to_owned(),
        },
    )
    .await
}

fn web_cpc() -> Metric {
    Metric::builder("web_cpc")
        .data_type(DataType::Double)
        .expr(metric_leaf("cpc").expect("a name"))
        .filter(ExprOperationExt::eq(
            dim("channel").expect("a name"),
            lit("web"),
        ))
        .build()
}

fn cpc_beside_web_cpc() -> Metric {
    Metric::builder("cpc_beside_web_cpc")
        .data_type(DataType::Double)
        .expr(metric_leaf("cpc").expect("a name") + metric_leaf("web_cpc").expect("a name"))
        .build()
}

fn cpc_beside_web_cpc_in_eu() -> Metric {
    Metric::builder("cpc_beside_web_cpc_in_eu")
        .data_type(DataType::Double)
        .expr(metric_leaf("cpc_beside_web_cpc").expect("a name"))
        .filter(in_eu())
        .build()
}

#[tokio::test]
async fn a_metric_reached_twice_is_checked_under_each_filter_that_reaches_it() {
    let model = spend_with_a_regional_clicks_child(vec![
        cpc(),
        web_cpc(),
        cpc_beside_web_cpc(),
        cpc_beside_web_cpc_in_eu(),
    ]);
    assert_eq!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend".to_owned(),
            metric: "cpc_beside_web_cpc_in_eu".to_owned(),
            measure: "clicks".to_owned(),
            dimension: "channel".to_owned(),
        },
        "`cpc` is reached plain and under `web_cpc`'s filter, and no child maps `clicks` \
         with `region` and `channel`"
    );
}

fn cpc_beside_web_cpc_per_target_in_eu() -> Metric {
    Metric::builder("cpc_beside_web_cpc_per_target_in_eu")
        .data_type(DataType::Double)
        .expr(
            metric_leaf("cpc_beside_web_cpc").expect("a name")
                / measure_leaf("target").expect("a name"),
        )
        .filter(in_eu())
        .build()
}

fn regional_spend_plan(metrics: Vec<Metric>, outer: Metric) -> SemanticModel {
    regional_spend_plan_over(regional_clicks_children(), &[], metrics, outer)
}

fn regional_spend_plan_over(
    children: impl IntoIterator<Item = NestedDataset>,
    measures: &[&str],
    metrics: Vec<Metric>,
    outer: Metric,
) -> SemanticModel {
    let mut spend_plan = Grainset::builder("spend_plan")
        .dimension(DimensionSite::r#ref("campaign").with_key())
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .measure(MeasureSite::r#ref("target"));
    let mut pools = spend_with(true, Vec::new(), regional_clicks_children()).measure(measure(
        "target",
        DataType::Double,
        "sum",
    ));
    for name in measures {
        spend_plan = spend_plan.measure(MeasureSite::r#ref(*name));
        pools = pools.measure(measure(name, DataType::Double, "sum"));
    }
    for metric in metrics.iter().chain([&outer]) {
        spend_plan = spend_plan.metric(MetricSite::r#ref(metric.body.name.as_str()));
    }
    let spend_plan = spend_plan
        .dataset(constituent("targets", &["campaign", "target", "region"]))
        .joinset(jp_over(children))
        .build();
    (metrics.into_iter().chain([outer]))
        .fold(pools, SemanticModelBuilder::metric)
        .grainset(spend_plan)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_nested_grainsets_metric_reached_twice_is_read_under_each_filter_that_reaches_it() {
    let metrics = vec![cpc(), web_cpc(), cpc_beside_web_cpc()];
    let model = regional_spend_plan(metrics, cpc_beside_web_cpc_per_target_in_eu());
    assert_eq!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend_plan".to_owned(),
            metric: "cpc_beside_web_cpc_per_target_in_eu".to_owned(),
            measure: "cpc_beside_web_cpc".to_owned(),
            dimension: "region".to_owned(),
        },
        "`jp.spend` reads `cpc` plain and under `web_cpc`'s filter, and none of its children \
         maps `clicks` with `channel` and `region`"
    );
}

fn web_only_cpc_per_target_in_eu() -> Metric {
    Metric::builder("web_only_cpc_per_target_in_eu")
        .data_type(DataType::Double)
        .expr(
            metric_leaf("web_only_cpc").expect("a name") / measure_leaf("target").expect("a name"),
        )
        .filter(in_eu())
        .build()
}

#[tokio::test]
async fn a_nested_grainsets_parts_metric_is_read_with_its_part_key_under_the_filter_that_reaches_it(
) {
    let model = regional_spend_plan(vec![web_only_cpc()], web_only_cpc_per_target_in_eu());
    assert_eq!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend_plan".to_owned(),
            metric: "web_only_cpc_per_target_in_eu".to_owned(),
            measure: "web_only_cpc".to_owned(),
            dimension: "region".to_owned(),
        },
        "`jp.spend` keys the parts of `web_only_cpc` on `channel`, and none of its children \
         maps `clicks` with `channel` and `region`"
    );
}

fn web_budget_per_impression() -> Metric {
    let web = ExprOperationExt::eq(dim("channel").expect("a name"), lit("web"));
    Metric::builder("web_budget_per_impression")
        .data_type(DataType::Double)
        .expr(Expr::Case {
            whens: vec![(
                web,
                measure_leaf("budget").expect("a name")
                    / measure_leaf("impressions").expect("a name"),
            )],
            else_: Some(Box::new(lit(0.5))),
        })
        .build()
}

fn web_budget_per_impression_per_click() -> Metric {
    Metric::builder("web_budget_per_impression_per_click")
        .data_type(DataType::Double)
        .expr(
            metric_leaf("web_budget_per_impression").expect("a name")
                / measure_leaf("clicks").expect("a name"),
        )
        .build()
}

fn web_budget_per_impression_per_click_per_target_in_eu() -> Metric {
    Metric::builder("web_budget_per_impression_per_click_per_target_in_eu")
        .data_type(DataType::Double)
        .expr(
            metric_leaf("web_budget_per_impression_per_click").expect("a name")
                / measure_leaf("target").expect("a name"),
        )
        .filter(in_eu())
        .build()
}

#[tokio::test]
async fn a_nested_grainsets_parts_metric_reads_the_measure_beside_a_link_with_that_links_part_key()
{
    let mut children = Vec::from(regional_clicks_children());
    children.push(grain_dataset(
        "spend_reach",
        "date",
        Granularity::Month,
        &["date", "campaign", "impressions", "channel", "region"],
    ));
    let model = regional_spend_plan_over(
        children,
        &["impressions"],
        vec![
            web_budget_per_impression(),
            web_budget_per_impression_per_click(),
        ],
        web_budget_per_impression_per_click_per_target_in_eu(),
    );
    assert_eq!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend_plan".to_owned(),
            metric: "web_budget_per_impression_per_click_per_target_in_eu".to_owned(),
            measure: "web_budget_per_impression_per_click".to_owned(),
            dimension: "region".to_owned(),
        },
        "`jp.spend` keys the parts of `web_budget_per_impression_per_click` on \
         `web_budget_per_impression`'s `channel`, and none of its children maps `clicks` with \
         `channel` and `region`"
    );
}

fn is_web() -> Metric {
    Metric::builder("is_web")
        .data_type(DataType::Boolean)
        .expr(ExprOperationExt::eq(
            dim("channel").expect("a name"),
            lit("web"),
        ))
        .build()
}

fn is_web_cpc() -> Metric {
    cpc_where(metric_leaf("is_web").expect("a name"), "is_web_cpc")
}

fn is_web_cpc_per_target_in_eu() -> Metric {
    Metric::builder("is_web_cpc_per_target_in_eu")
        .data_type(DataType::Double)
        .expr(metric_leaf("is_web_cpc").expect("a name") / measure_leaf("target").expect("a name"))
        .filter(in_eu())
        .build()
}

#[tokio::test]
async fn a_nested_grainsets_parts_metric_reads_a_row_level_metric_as_its_part_key() {
    let model = regional_spend_plan(vec![is_web(), is_web_cpc()], is_web_cpc_per_target_in_eu());
    assert_eq!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend_plan".to_owned(),
            metric: "is_web_cpc_per_target_in_eu".to_owned(),
            measure: "is_web_cpc".to_owned(),
            dimension: "region".to_owned(),
        },
        "`jp.spend` keys the parts of `is_web_cpc` on `is_web`, which reads `channel`, and none \
         of its children maps `clicks` with `channel` and `region`"
    );
}

fn per_target_in_eu(name: &str, metric: &str) -> Metric {
    Metric::builder(name)
        .data_type(DataType::Double)
        .expr(metric_leaf(metric).expect("a name") / measure_leaf("target").expect("a name"))
        .filter(in_eu())
        .build()
}

#[tokio::test]
async fn a_nested_grainsets_own_agg_metric_reads_a_row_level_metric_in_its_formula_grain() {
    let web = regional_spend_plan(
        vec![avg_web_only_cpc()],
        per_target_in_eu("avg_web_only_cpc_per_target_in_eu", "avg_web_only_cpc"),
    );
    let is_web = regional_spend_plan(
        vec![is_web(), avg_is_web_cpc()],
        per_target_in_eu("avg_is_web_cpc_per_target_in_eu", "avg_is_web_cpc"),
    );
    assert_eq!(
        compile_diagnostic(web).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend_plan".to_owned(),
            metric: "avg_web_only_cpc_per_target_in_eu".to_owned(),
            measure: "avg_web_only_cpc".to_owned(),
            dimension: "region".to_owned(),
        },
        "`jp.spend` reads `clicks` at the formula grain `channel`, and none of its children \
         maps `clicks` with `channel` and `region`"
    );
    assert_eq!(
        compile_diagnostic(is_web).await.kind,
        CompileError::GrainsetMetricFilterUnmapped {
            grainset: "spend_plan".to_owned(),
            metric: "avg_is_web_cpc_per_target_in_eu".to_owned(),
            measure: "avg_is_web_cpc".to_owned(),
            dimension: "region".to_owned(),
        },
        "`is_web` is row-level, so `jp.spend` reads `clicks` at the formula grain `is_web` as \
         it does at `channel`"
    );
}

fn web_only_cpc_of_eu() -> Metric {
    Metric::builder("web_only_cpc_of_eu")
        .data_type(DataType::Double)
        .expr(metric_leaf("web_only_cpc").expect("a name"))
        .filter(in_eu())
        .build()
}

#[tokio::test]
async fn a_part_key_read_through_a_linked_metric_without_agg_reads_that_metrics_key() {
    let model = spend_with_a_regional_clicks_child(vec![web_only_cpc(), web_only_cpc_of_eu()]);
    assert_eq!(
        compile_diagnostic(model).await.kind,
        CompileError::GrainsetMetricPartKeyUnformed {
            grainset: "spend".to_owned(),
            metric: "web_only_cpc_of_eu".to_owned(),
            measure: "clicks".to_owned(),
            dimension: "channel".to_owned(),
            child: "spend.spend_clicks".to_owned(),
        },
        "`web_only_cpc` keys its parts on `channel`, which `spend_clicks` does not map"
    );
}

fn avg_of_web_only_cpc(name: &str, filter: Option<UnresolvedSemanticExpr>) -> Metric {
    let metric = Metric::builder(name)
        .data_type(DataType::Double)
        .agg(agg("avg"))
        .expr(metric_leaf("web_only_cpc").expect("a name"));
    match filter {
        Some(filter) => metric.filter(filter).build(),
        None => metric.build(),
    }
}

#[tokio::test]
async fn a_formula_grain_read_through_a_linked_metric_without_agg_reads_that_metrics_dimensions() {
    let in_eu = spend_with_a_regional_clicks_child(vec![
        web_only_cpc(),
        avg_of_web_only_cpc("avg_of_web_only_cpc_in_eu", Some(in_eu())),
    ]);
    assert_eq!(
        compile_diagnostic(in_eu).await.kind,
        CompileError::GrainsetMetricWithoutKeyset {
            grainset: "spend".to_owned(),
            metric: "avg_of_web_only_cpc_in_eu".to_owned(),
        },
        "`web_only_cpc` reads `channel` on each row, so the formula grain of its reader holds \
         `channel`, and no child maps `clicks` with `channel` and `region`"
    );
    let manifest = compile_model(spend_with_a_regional_clicks_child(vec![
        web_only_cpc(),
        avg_of_web_only_cpc("avg_of_web_only_cpc", None),
    ]))
    .await;
    signature(&manifest, "spend", "avg_of_web_only_cpc");
}

#[tokio::test]
async fn a_nested_grainset_whose_formula_grain_reads_a_linked_metrics_dimension_that_no_child_maps_under_the_filter_computes_nothing(
) {
    let model = regional_spend_plan(
        vec![web_only_cpc()],
        avg_of_web_only_cpc("avg_of_web_only_cpc_in_eu", Some(in_eu())),
    );

    let manifest = compile_model(model).await;

    assert!(
        !has_signature(
            &manifest,
            "spend_plan.jp.spend",
            "avg_of_web_only_cpc_in_eu"
        ),
        "the formula grain of `avg_of_web_only_cpc_in_eu` holds the `channel` that \
         `web_only_cpc` reads, and no child of `jp.spend` maps `clicks` with `channel` and \
         `region`, so `jp.spend` computes nothing for it"
    );
    assert!(
        has_signature(&manifest, "spend_plan.jp", "avg_of_web_only_cpc_in_eu"),
        "`jp` keeps the Metric as on `main`"
    );
    signature(&manifest, "spend_plan.jp.spend", "web_only_cpc");
}

fn spend_beside_j_over_a_nested_spend(
    metrics: Vec<Metric>,
    children: [NestedDataset; 2],
    nested: [NestedDataset; 2],
) -> SemanticModel {
    spend_beside_j_over(metrics, children, nested, Vec::new())
}

fn spend_beside_j_over(
    metrics: Vec<Metric>,
    children: [NestedDataset; 2],
    nested: impl IntoIterator<Item = NestedDataset>,
    nested_unionsets: Vec<NestedUnionset>,
) -> SemanticModel {
    let sites = (metrics.iter()).map(|metric| MetricSite::r#ref(metric.body.name.as_str()));
    let joinset = Joinset::builder("j")
        .grainset(
            NestedGrainset::builder("spend")
                .temporal(TemporalShape::events("date", Some(Granularity::Month)))
                .datasets(nested)
                .unionsets(nested_unionsets)
                .build(),
        )
        .dataset(constituent("campaigns", &["campaign"]))
        .dimension(DimensionSite::r#ref("campaign").with_key())
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .metrics(sites)
        .relationship(
            Relationship::builder()
                .name("spend_to_campaigns")
                .from("spend")
                .to("campaigns")
                .on(["campaign"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    spend_with(false, metrics, children)
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[derive(Clone, Copy, Debug)]
enum Place {
    JoinsetOperand,
    UnionsetBranch,
    GrainsetChild,
}

const PLACES: [Place; 3] = [
    Place::JoinsetOperand,
    Place::UnionsetBranch,
    Place::GrainsetChild,
];

impl Place {
    fn parent(self) -> &'static str {
        match self {
            Place::JoinsetOperand => "j",
            Place::UnionsetBranch => "u",
            Place::GrainsetChild => "spend_plan.jp",
        }
    }

    fn nested_spend(self) -> String {
        format!("{}.spend", self.parent())
    }

    fn model(
        self,
        metrics: Vec<Metric>,
        children: [NestedDataset; 2],
        nested: [NestedDataset; 2],
    ) -> SemanticModel {
        self.keyed_model(true, metrics, children, nested)
    }

    fn keyed_model(
        self,
        keyed: bool,
        metrics: Vec<Metric>,
        children: [NestedDataset; 2],
        nested: [NestedDataset; 2],
    ) -> SemanticModel {
        let campaign = match keyed {
            true => DimensionSite::r#ref("campaign").with_key(),
            false => DimensionSite::r#ref("campaign"),
        };
        match self {
            Place::JoinsetOperand if keyed => {
                spend_beside_j_over_a_nested_spend(metrics, children, nested)
            }
            Place::JoinsetOperand => spend_beside_j_keyless(metrics, children, nested),
            Place::UnionsetBranch => {
                spend_beside_u_over_a_nested_spend(campaign, metrics, children, nested)
            }
            Place::GrainsetChild => {
                spend_plan_over_a_nested_spend(campaign, metrics, children, nested)
            }
        }
    }
}

fn nested_spend_of(nested: impl IntoIterator<Item = NestedDataset>) -> NestedGrainset {
    NestedGrainset::builder("spend")
        .temporal(TemporalShape::events("date", Some(Granularity::Month)))
        .datasets(nested)
        .build()
}

fn spend_beside_j_keyless(
    metrics: Vec<Metric>,
    children: [NestedDataset; 2],
    nested: [NestedDataset; 2],
) -> SemanticModel {
    let sites = (metrics.iter()).map(|metric| MetricSite::r#ref(metric.body.name.as_str()));
    let joinset = Joinset::builder("j")
        .grainset(nested_spend_of(nested))
        .dataset(constituent("campaigns", &["campaign"]))
        .dimension(DimensionSite::r#ref("campaign"))
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .metrics(sites)
        .relationship(
            Relationship::builder()
                .name("spend_to_campaigns")
                .from("spend")
                .to("campaigns")
                .on(["campaign"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    spend_with(false, metrics, children)
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

fn spend_beside_u_over_a_nested_spend(
    campaign: DimensionSite,
    metrics: Vec<Metric>,
    children: [NestedDataset; 2],
    nested: [NestedDataset; 2],
) -> SemanticModel {
    let sites = (metrics.iter()).map(|metric| MetricSite::r#ref(metric.body.name.as_str()));
    let unionset = Unionset::builder("u")
        .grainset(nested_spend_of(nested))
        .dataset(grain_dataset(
            "spend_extra",
            "date",
            Granularity::Month,
            &["date", "campaign", "clicks", "budget", "channel"],
        ))
        .dimension(campaign)
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .metrics(sites)
        .build();
    spend_with(false, metrics, children)
        .unionset(unionset)
        .build()
        .expect("the fixture model must build")
        .0
}

fn spend_plan_over_a_nested_spend(
    campaign: DimensionSite,
    metrics: Vec<Metric>,
    children: [NestedDataset; 2],
    nested: [NestedDataset; 2],
) -> SemanticModel {
    let sites = (metrics.iter()).map(|metric| MetricSite::r#ref(metric.body.name.as_str()));
    let spend_plan = Grainset::builder("spend_plan")
        .dimension(campaign)
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .measure(MeasureSite::r#ref("target"))
        .metrics(sites)
        .dataset(constituent("targets", &["campaign", "target"]))
        .joinset(jp_over(nested))
        .build();
    spend_with(false, metrics, children)
        .measure(measure("target", DataType::Double, "sum"))
        .grainset(spend_plan)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_nested_grainset_whose_child_cannot_form_the_part_key_computes_nothing() {
    for place in PLACES {
        let model = place.model(
            vec![web_only_cpc()],
            spend_children(true),
            spend_children(false),
        );

        let manifest = compile_model(model).await;

        assert!(
            !has_signature(&manifest, &place.nested_spend(), "web_only_cpc"),
            "{place:?}: the nested `spend` cannot form the part key in `spend_monthly`, so it \
             computes nothing"
        );
        assert!(
            has_signature(&manifest, place.parent(), "web_only_cpc"),
            "{place:?}: the parent keeps the Metric as on `main`"
        );
        signature(&manifest, "spend", "web_only_cpc");
    }
}

#[tokio::test]
async fn a_nested_grainset_whose_metric_filter_no_child_of_a_measure_maps_computes_nothing() {
    for place in PLACES {
        let model = place.model(
            vec![web_budget()],
            spend_children(true),
            spend_children(false),
        );

        let manifest = compile_model(model).await;

        assert!(
            !has_signature(&manifest, &place.nested_spend(), "web_budget"),
            "{place:?}: no child of the nested `spend` maps `budget` with `channel`, so it \
             computes nothing"
        );
    }
}

#[tokio::test]
async fn a_nested_grainset_whose_children_share_no_keyset_computes_nothing() {
    for place in PLACES {
        let model = place.keyed_model(
            false,
            vec![cpc()],
            spend_children(true),
            spend_children(true),
        );

        let manifest = compile_model(model).await;

        assert!(
            !has_signature(&manifest, &place.nested_spend(), "cpc"),
            "{place:?}: the parent marks no Dimension a key, so the children of the nested `spend` \
             share no keyset for `cpc`"
        );
        signature(&manifest, "spend", "cpc");
    }
}

#[tokio::test]
async fn a_nested_grainset_whose_formula_grain_one_serving_child_does_not_map_computes_nothing() {
    for place in PLACES {
        let model = place.model(
            vec![avg_cpc(), is_web(), avg_web_only_cpc(), avg_is_web_cpc()],
            spend_children(true),
            spend_children(false),
        );

        let manifest = compile_model(model).await;

        assert!(
            has_signature(&manifest, &place.nested_spend(), "avg_cpc"),
            "{place:?}: the nested `spend` computes an own-`agg:` Metric whose formula grain \
             reads no Dimension"
        );
        for name in ["avg_web_only_cpc", "avg_is_web_cpc"] {
            assert!(
                !has_signature(&manifest, &place.nested_spend(), name),
                "{place:?}: `spend_monthly` gives `budget` but does not map `channel`, which the \
                 formula grain of `{name}` reads"
            );
            signature(&manifest, "spend", name);
        }
    }
}

#[tokio::test]
async fn a_nested_grainset_whose_formula_grain_reads_the_time_axis_finer_than_a_child_of_a_measure_maps_it_computes_nothing(
) {
    for place in PLACES {
        let mixed = compile_model(place.model(
            vec![is_early(), avg_early_cpc(), avg_is_early_cpc()],
            spend_children_at_month(),
            spend_children(true),
        ))
        .await;
        let monthly = compile_model(place.model(
            vec![is_early(), avg_early_cpc(), avg_is_early_cpc()],
            spend_children_at_month(),
            spend_children_at_month(),
        ))
        .await;

        for name in ["avg_early_cpc", "avg_is_early_cpc"] {
            assert!(
                !has_signature(&mixed, &place.nested_spend(), name),
                "{place:?}: `spend_monthly` gives `budget` with `date` at Month, and \
                 `spend_daily` maps `date` at Day, so the nested `spend` does not compute `{name}`"
            );
            assert!(
                has_signature(&monthly, &place.nested_spend(), name),
                "{place:?}: each child that gives a value maps `date` at Month, so the nested \
                 `spend` computes `{name}`"
            );
            signature(&mixed, "spend", name);
        }
    }
}

#[tokio::test]
async fn a_grainset_lists_a_signature_for_each_metric_it_computes_and_for_no_other_name() {
    let manifest = compile_model(spend_model(true)).await;
    let grainset = by_name(&manifest, "spend");
    let alias = grainset.alias.as_str();
    let listed_names: BTreeSet<&str> = (listed(&manifest, grainset))
        .map(|signature| manifest.semantic_definitions.get(&signature.definition_id))
        .map(|definition| definition.expect("a definition").name.as_str())
        .collect();
    assert_eq!(
        listed_names,
        BTreeSet::from(["avg_cpc", "cpc", "cpm", "web_cpc2"]),
        "`web_budget` is mapped whole by `spend_monthly`, so the Grainset computes none of it"
    );

    let cpc = signature(&manifest, "spend", "cpc");
    assert_eq!(
        tree(&manifest, cpc),
        column(alias, "budget") / column(alias, "clicks")
    );
    assert_eq!(
        stored_expr(
            &manifest,
            signature(&manifest, "spend", "cpm").expression_id
        ),
        &(link(cpc.expression_id) * Expr::Literal(Literal::Double(1000.0))),
        "a Metric over a computed Metric links it"
    );
    assert_eq!(
        tree(&manifest, signature(&manifest, "spend", "web_cpc2")),
        column(alias, "web_budget") / column(alias, "clicks"),
        "a Metric over a Metric that one child maps whole reads that Metric's final value"
    );
    assert_eq!(
        tree(&manifest, signature(&manifest, "spend", "avg_cpc")),
        column(alias, "budget") / column(alias, "clicks"),
        "a Metric with its own aggregation stores its formula over final values, not the aggregate"
    );
    for name in ["avg_cpc", "cpc", "cpm", "web_cpc2"] {
        let expression = signature(&manifest, "spend", name).expression_id;
        assert_eq!(
            manifest
                .expressions
                .get(&expression)
                .map(|entry| entry.layer),
            Some(ExprLayer::PostAggregate),
            "{name} is stored post-aggregate"
        );
    }
}

#[tokio::test]
async fn a_joinset_over_a_grainset_that_computes_its_metric_compiles() {
    let manifest = compile_model(spend_model(true)).await;
    let nested: BTreeSet<&str> = (listed(&manifest, by_name(&manifest, "j.spend")))
        .map(|signature| manifest.semantic_definitions.get(&signature.definition_id))
        .map(|definition| definition.expect("a definition").name.as_str())
        .collect();
    assert_eq!(
        nested,
        BTreeSet::from(["web_cpc2"]),
        "the nested `spend` computes `web_cpc2` and no other name"
    );
    signature(&manifest, "j", "web_cpc2");
}

fn web_cpc2_per_target(name: &str, filter: Option<UnresolvedSemanticExpr>) -> Metric {
    let metric = Metric::builder(name)
        .data_type(DataType::Double)
        .expr(metric_leaf("web_cpc2").expect("a name") / measure_leaf("target").expect("a name"));
    match filter {
        Some(filter) => metric.filter(filter).build(),
        None => metric.build(),
    }
}

fn jp_over(children: impl IntoIterator<Item = NestedDataset>) -> NestedJoinset {
    NestedJoinset::builder("jp")
        .grainset(
            NestedGrainset::builder("spend")
                .temporal(TemporalShape::events("date", Some(Granularity::Month)))
                .datasets(children)
                .build(),
        )
        .dataset(constituent("campaigns", &["campaign"]))
        .relationship(
            Relationship::builder()
                .name("spend_to_campaigns")
                .from("spend")
                .to("campaigns")
                .on(["campaign"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build()
}

fn spend_plan_model() -> SemanticModel {
    let spend_plan = Grainset::builder("spend_plan")
        .dimension(DimensionSite::r#ref("campaign").with_key())
        .dimension(DimensionSite::r#ref("date"))
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .measure(MeasureSite::r#ref("target"))
        .metric(MetricSite::r#ref("web_budget"))
        .metric(MetricSite::r#ref("web_cpc2"))
        .metric(MetricSite::r#ref("web_cpc2_per_target"))
        .metric(MetricSite::r#ref("web_cpc2_per_target_of_a"))
        .dataset(constituent("targets", &["campaign", "target"]))
        .joinset(jp_over(spend_children(true)))
        .build();
    let of_a = ExprOperationExt::eq(dim("campaign").expect("a name"), lit("A"));
    let metrics = vec![web_budget(), cpc(), cpm(), web_cpc2(), avg_cpc()];
    spend_with(false, metrics, spend_children(true))
        .measure(measure("target", DataType::Double, "sum"))
        .metric(web_cpc2_per_target("web_cpc2_per_target", None))
        .metric(web_cpc2_per_target("web_cpc2_per_target_of_a", Some(of_a)))
        .grainset(spend_plan)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_grainset_computes_a_metric_over_a_value_that_a_nested_grainset_computes() {
    let manifest = compile_model(spend_plan_model()).await;
    let grainset = by_name(&manifest, "spend_plan");
    let alias = grainset.alias.as_str();
    let listed_names: BTreeSet<&str> = (listed(&manifest, grainset))
        .map(|signature| manifest.semantic_definitions.get(&signature.definition_id))
        .map(|definition| definition.expect("a definition").name.as_str())
        .collect();
    assert_eq!(
        listed_names,
        BTreeSet::from(["web_cpc2_per_target", "web_cpc2_per_target_of_a"]),
        "`jp.spend` computes `web_cpc2` and maps each name it reads with `campaign`, so \
         `jp` serves it at the keyset `campaign` that `targets` also carries"
    );
    for name in ["web_cpc2_per_target", "web_cpc2_per_target_of_a"] {
        assert_eq!(
            tree(&manifest, signature(&manifest, "spend_plan", name)),
            column(alias, "web_cpc2") / column(alias, "target"),
            "{name} reads the final values of `web_cpc2` and `target`"
        );
    }
    signature(&manifest, "spend_plan.jp.spend", "web_cpc2");
    manifest
        .validate()
        .expect("the compiled artifact passes validate()");
}

#[tokio::test]
async fn a_grainsets_child_mapping_both_names_its_filter_reads_serves_the_metric() {
    let manifest = compile_model(spend_model(true)).await;
    signature(&manifest, "spend.spend_monthly", "web_budget");
    assert!(
        !has_signature(&manifest, "spend.spend_daily", "web_budget"),
        "`spend_daily` still does not map `budget`"
    );
    assert!(
        !has_signature(&manifest, "spend", "web_budget"),
        "a Grainset owns no signature for a name a child maps (ADR 0084 rule 10)"
    );
}

#[tokio::test]
async fn validate_reports_a_grainset_among_its_own_children_beside_its_computed_metric_as_a_cycle()
{
    let mut manifest = compile_model(spend_model(true)).await;
    let id = by_name(&manifest, "spend").id.clone();
    let entry = manifest
        .semantic_sources
        .get_mut(&id)
        .expect("the compiled artifact holds `spend`");
    let semstrait_manifest::RoleDefinition::Grainset { grains, .. } = &mut entry.role_definition
    else {
        panic!("`spend` is a Grainset");
    };
    grains.insert(0, id.clone());

    let faults = manifest
        .validate()
        .expect_err("a Grainset among its own children is a cycle");
    assert!(
        faults.iter().any(|fault| matches!(
            &fault.kind,
            CompileError::CyclicSemanticSource { cycle } if cycle.contains(&id)
        )),
        "expected CyclicSemanticSource naming `spend` among {faults:?}"
    );
}

#[tokio::test]
async fn a_body_linking_a_dropped_name_is_dropped_too() {
    let web_budget_body = Metric::builder("web_budget_body")
        .data_type(DataType::Double)
        .expr(measure_leaf("budget").expect("a name"))
        .build();
    let grainset = Grainset::builder("spend")
        .dimension(DimensionSite::r#ref("date").with_key())
        .dimension(DimensionSite::r#ref("campaign").with_key())
        .dimension(DimensionSite::r#ref("channel"))
        .measure(MeasureSite::r#ref("clicks"))
        .measure(MeasureSite::r#ref("budget"))
        .metric(MetricSite::r#ref("web_budget_body"))
        .dataset(grain_dataset(
            "spend_daily",
            "date",
            Granularity::Day,
            &["date", "campaign", "clicks", "channel", "budget"],
        ))
        .dataset(grain_dataset(
            "spend_monthly",
            "date",
            Granularity::Month,
            &["date", "campaign", "clicks", "budget"],
        ))
        .build();
    let model = spend_pools_with_budget_filter(Some(ExprOperationExt::eq(
        dim("channel").expect("a name"),
        lit("web"),
    )))
    .metric(web_budget_body)
    .grainset(grainset)
    .build()
    .expect("the fixture model must build")
    .0;

    let manifest = compile_model(model).await;
    signature(&manifest, "spend.spend_daily", "budget");
    signature(&manifest, "spend.spend_daily", "web_budget_body");
    assert!(
        !has_signature(&manifest, "spend.spend_monthly", "budget"),
        "`spend_monthly` does not map `channel`, so `budget`'s own filter drops it"
    );
    assert!(
        !has_signature(&manifest, "spend.spend_monthly", "web_budget_body"),
        "`web_budget_body` links `budget`, which `spend_monthly` has no entry for"
    );
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            composed_name_unserved_refusal().await,
            join_basis_not_mapped_refusal().await,
            unionset_dimension_unserved_refusal().await,
            unionset_measure_unserved_refusal().await,
            root_dataset_missing_column_refusal().await,
            auto_without_schema_refusal().await,
            no_schema_body_refusal().await,
            no_schema_literal_refusal().await,
            implicit_unionset_missing_column_refusal().await,
            constituent_implicit_unionset_missing_column_refusal().await,
            grainset_dimension_unserved_refusal().await,
            grainset_child_filter_unserved_refusal().await,
            grainset_metric_reads_unmapped_refusal().await,
            grainset_metric_without_keyset_refusal().await,
            grainset_formula_grain_unmapped_refusal().await,
            grainset_formula_grain_row_level_unmapped_refusal().await,
            metric_child_whole_filter_unserved_refusal().await,
            filter_dimensions_unmapped_together_refusal().await,
            filter_dimensions_reversed_refusal().await,
            part_key_unmapped_refusal().await,
            part_key_coarser_refusal().await,
            part_key_row_level_coarser_refusal().await,
            part_key_unmapped_under_filter_refusal().await,
            part_key_row_level_under_filter_refusal().await,
            formula_grain_coarser_refusal().await,
        ]
    })
    .await;
}
