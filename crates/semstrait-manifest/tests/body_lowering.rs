use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, CastFailure, Expr};

use semstrait_manifest::{CompileError, ExprLayer, SemanticManifest, SemanticSignature};
use semstrait_model::expr::{col, dim, lit, measure, metric};
use semstrait_model::{
    Dataset, Dimension, DimensionSite, Measure, MeasureSite, Metric, MetricSite, SemanticMapping,
    SemanticModel,
};

mod common;
use common::*;

fn reads_column(manifest: &SemanticManifest, signature: &SemanticSignature, name: &str) -> bool {
    manifest
        .expressions
        .columns(&signature.expression_id)
        .iter()
        .any(|(_, column)| column.name() == name)
}

fn concat_names() -> semstrait_model::UnresolvedSemanticExpr {
    call(
        "concat",
        vec![
            dim("first_name").expect("a name"),
            dim("last_name").expect("a name"),
        ],
    )
}

fn two_datasets_over_one_bodied_dimension() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "people".to_owned();
    for name in ["first_name", "last_name"] {
        insert_dimension(
            &mut model,
            Dimension::builder(name)
                .data_type(DataType::String)
                .categorical()
                .build(),
        );
    }
    insert_dimension(
        &mut model,
        Dimension::builder("full_name")
            .data_type(DataType::String)
            .categorical()
            .expr(concat_names())
            .build(),
    );
    for (dataset, first, last) in [
        ("contacts", "contact_first", "contact_last"),
        ("users", "given_name", "surname"),
    ] {
        insert_dataset(
            &mut model,
            Dataset::builder(dataset)
                .table(format!("warehouse.{dataset}"))
                .dimension(DimensionSite::r#ref("first_name"))
                .dimension(DimensionSite::r#ref("last_name"))
                .dimension(DimensionSite::r#ref("full_name"))
                .semantic_mapping(
                    SemanticMapping::builder()
                        .column("first_name", first)
                        .column("last_name", last)
                        .build(),
                )
                .build(),
        );
    }
    model
}

#[tokio::test]
async fn one_bodied_dimension_realises_once_per_source_over_that_sources_columns() {
    let manifest = compile_model(two_datasets_over_one_bodied_dimension()).await;

    let named: Vec<_> = manifest
        .semantic_definitions
        .entries
        .values()
        .filter(|definition| definition.name.as_str() == "full_name")
        .collect();
    assert_eq!(named.len(), 1, "one name is one definition, got {named:?}");

    let contacts = signature(&manifest, "contacts", "full_name");
    let users = signature(&manifest, "users", "full_name");
    assert_ne!(contacts.id, users.id);

    let expected = |dataset: &str, first: &str, last: &str| {
        let alias = source_alias(&manifest, dataset);
        let mapped = |name| cast(column(&alias, name), DataType::String);
        physical_call("concat", vec![mapped(first), mapped(last)])
    };
    assert_eq!(
        tree(&manifest, contacts),
        expected("contacts", "contact_first", "contact_last"),
        "the body is lowered over this source's mapped columns"
    );
    assert_eq!(
        tree(&manifest, users),
        expected("users", "given_name", "surname"),
        "and over the other source's, which are different columns"
    );

    assert_ne!(
        contacts.expression_id, users.expression_id,
        "two sources' trees are two entries"
    );
}

#[tokio::test]
async fn a_dimension_over_a_dimension_links_its_entry() {
    let manifest = compile_model(zone_model()).await;
    let region = signature(&manifest, "orders", "region").expression_id;
    let zone = signature(&manifest, "orders", "zone");
    assert_eq!(
        stored_expr(&manifest, zone.expression_id),
        &common::zone(link(region))
    );
    assert_eq!(
        manifest.expressions.len(),
        2,
        "`region`'s entry and `zone`'s"
    );
    assert_eq!(
        manifest.expressions.physical[&zone.expression_id].layer,
        ExprLayer::Scalar
    );
    assert_eq!(
        tree(&manifest, zone),
        common::zone(cast(orders_column("region"), DataType::String))
    );
}

#[tokio::test]
async fn a_metric_over_two_measures_is_one_entry_with_two_links() {
    let mut model = orders_model(
        Vec::new(),
        vec![
            common::measure("item_count", DataType::Integer, "sum"),
            common::measure("short_tally", DataType::SmallInt, "sum"),
        ],
        |orders| orders.metric(MetricSite::r#ref("items_net")),
    );
    insert_metric(
        &mut model,
        Metric::builder("items_net")
            .data_type(DataType::Integer)
            .expr(measure("item_count").expect("a name") - measure("short_tally").expect("a name"))
            .build(),
    );
    let manifest = compile_model(model).await;
    let entry =
        &manifest.expressions.physical[&signature(&manifest, "orders", "items_net").expression_id];
    assert_eq!(
        serde_json::to_string(&entry.expr).expect("a tree writes"),
        r#"{"type":"binary_op","content":{"op":"subtract","left":"15184182-69e5-55a9-9292-b09b5731f128","right":"09712050-49b8-534f-98f1-34d64cf76e3c"}}"#
    );
    assert_eq!(entry.layer, ExprLayer::PostAggregate);
}

#[tokio::test]
async fn a_measure_over_a_dimension_links_its_entry() {
    let units = Measure::builder("units")
        .data_type(DataType::BigInt)
        .agg(agg("sum"))
        .expr(dim("qty").expect("a name"))
        .build();
    let manifest = compile_model(orders_model(
        vec![dimension("qty", DataType::Integer)],
        vec![units],
        |orders| orders,
    ))
    .await;
    let qty = signature(&manifest, "orders", "qty").expression_id;
    let units = signature(&manifest, "orders", "units");
    assert_eq!(
        stored_expr(&manifest, units.expression_id),
        &aggregate("sum", false, link(qty), None)
    );
    assert_eq!(
        tree(&manifest, units),
        aggregate(
            "sum",
            false,
            cast(orders_column("qty"), DataType::Integer),
            None
        )
    );
}

#[tokio::test]
async fn an_alias_dimension_and_an_alias_metric_store_a_bare_link() {
    let mut model = orders_model(
        vec![
            dimension("region", DataType::String),
            bodied("zone2", DataType::String, dim("region").expect("a name")),
        ],
        vec![
            common::measure("revenue", DataType::Integer, "sum"),
            common::measure("refunds", DataType::Integer, "sum"),
        ],
        |orders| {
            orders
                .metric(MetricSite::r#ref("net"))
                .metric(MetricSite::r#ref("net2"))
        },
    );
    for (name, expr) in [
        (
            "net",
            measure("revenue").expect("a name") - measure("refunds").expect("a name"),
        ),
        ("net2", metric("net").expect("a name")),
    ] {
        insert_metric(
            &mut model,
            Metric::builder(name)
                .data_type(DataType::Integer)
                .expr(expr)
                .build(),
        );
    }
    let manifest = compile_model(model).await;
    for (alias, linked) in [("zone2", "region"), ("net2", "net")] {
        let [alias, linked] = [alias, linked].map(|name| signature(&manifest, "orders", name));
        assert_eq!(
            stored_expr(&manifest, alias.expression_id),
            &link(linked.expression_id)
        );
        assert_eq!(tree(&manifest, alias), tree(&manifest, linked));
    }
}

fn two_datasets_over_one_bodied_metric() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "finance".to_owned();
    for name in ["revenue", "cost"] {
        insert_measure(
            &mut model,
            Measure::builder(name)
                .data_type(DataType::Integer)
                .agg(agg("sum"))
                .build(),
        );
    }
    insert_metric(
        &mut model,
        Metric::builder("gross_margin")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name") - measure("cost").expect("a name"))
            .build(),
    );
    for (dataset, revenue) in [("orders", "amount"), ("invoices", "total")] {
        insert_dataset(
            &mut model,
            Dataset::builder(dataset)
                .table(format!("warehouse.{dataset}"))
                .measure(MeasureSite::r#ref("revenue"))
                .measure(MeasureSite::r#ref("cost"))
                .metric(MetricSite::r#ref("gross_margin"))
                .semantic_mapping(
                    SemanticMapping::builder()
                        .column("revenue", revenue)
                        .column("cost", "shared_cost")
                        .build(),
                )
                .build(),
        );
    }
    model
}

#[tokio::test]
async fn a_metric_reads_each_measures_aggregate_once_per_source() {
    let manifest = compile_model(two_datasets_over_one_bodied_metric()).await;

    let orders = signature(&manifest, "orders", "gross_margin");
    let invoices = signature(&manifest, "invoices", "gross_margin");

    let expected = |dataset: &str, revenue: &str| {
        let alias = source_alias(&manifest, dataset);
        Expr::BinaryOp {
            op: BinaryOpKind::Subtract,
            left: Box::new(integer_sum(&alias, revenue)),
            right: Box::new(integer_sum(&alias, "shared_cost")),
        }
    };
    assert_eq!(
        tree(&manifest, orders),
        expected("orders", "amount"),
        "each referenced measure's aggregate reads in its link's place"
    );
    assert_eq!(tree(&manifest, invoices), expected("invoices", "total"));

    assert_ne!(orders.expression_id, invoices.expression_id);
    let (orders_cost, invoices_cost) = (
        signature(&manifest, "orders", "cost"),
        signature(&manifest, "invoices", "cost"),
    );
    assert_ne!(orders_cost.expression_id, invoices_cost.expression_id);
    assert!(
        reads_column(&manifest, orders_cost, "shared_cost")
            && reads_column(&manifest, invoices_cost, "shared_cost"),
        "each source's `cost` entry reads `shared_cost` in its own tree"
    );
}

#[tokio::test]
async fn gross_margins_entry_has_the_stored_form() {
    let manifest = compile_model(two_datasets_over_one_bodied_metric()).await;
    let gross_margin = signature(&manifest, "orders", "gross_margin");
    let [revenue, cost] =
        ["revenue", "cost"].map(|name| signature(&manifest, "orders", name).expression_id);
    let id = gross_margin.expression_id;
    let entry = &manifest.expressions.physical[&id];
    let sources = serde_json::to_string(&entry.physical_sources).expect("a set of ids writes");
    assert_eq!(
        serde_json::to_string(entry).expect("an entry writes"),
        format!(
            r#"{{"id":"{id}","expr":{{"type":"binary_op","content":{{"op":"subtract","left":"{revenue}","right":"{cost}"}}}},"layer":"post_aggregate","physical_sources":{sources}}}"#
        )
    );
}

#[tokio::test]
async fn a_measures_own_entry_is_apart_from_the_metric_that_links_it() {
    let manifest = compile_model(two_datasets_over_one_bodied_metric()).await;
    let alias = source_alias(&manifest, "orders");
    let metric = signature(&manifest, "orders", "gross_margin");
    let revenue = signature(&manifest, "orders", "revenue");
    assert_ne!(metric.expression_id, revenue.expression_id);
    assert_eq!(
        tree(&manifest, revenue),
        integer_sum(&alias, "amount"),
        "the cast of the Aggregate over the cast of one Column, whole in one entry"
    );
}

#[tokio::test]
async fn a_body_referencing_a_name_the_source_does_not_expose_is_refused() {
    let mut model = SemanticModel::default();
    model.name = "finance".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("cost")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("gross_margin")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name") - measure("cost").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("gross_margin"))
            .build(),
    );

    match compile_error(model).await {
        CompileError::UnexposedReference {
            semantics,
            referenced,
            semantic_source,
        } => {
            assert_eq!(semantics, "gross_margin");
            assert_eq!(referenced, "cost");
            assert_eq!(semantic_source, "orders");
        }
        other => panic!("expected UnexposedReference, got {other:?}"),
    }
}

#[tokio::test]
async fn a_dimension_body_referencing_a_measure_is_refused() {
    let mut model = SemanticModel::default();
    model.name = "finance".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_dimension(
        &mut model,
        Dimension::builder("band")
            .data_type(DataType::Integer)
            .categorical()
            .expr(measure("revenue").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .dimension(DimensionSite::r#ref("band"))
            .build(),
    );

    match compile_error(model).await {
        CompileError::AggregateReferenceInRowLevelBody {
            semantics,
            referenced,
        } => {
            assert_eq!(semantics, "band");
            assert_eq!(referenced, "revenue");
        }
        other => panic!("expected AggregateReferenceInRowLevelBody, got {other:?}"),
    }
}

fn mutually_referential_model(overrides: bool) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "loops".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("a")
            .data_type(DataType::BigInt)
            .categorical()
            .expr(dim("b").expect("a name") + lit(1_i64))
            .build(),
    );
    insert_dimension(
        &mut model,
        Dimension::builder("b")
            .data_type(DataType::BigInt)
            .categorical()
            .expr(dim("a").expect("a name") + lit(1_i64))
            .build(),
    );
    let sites = if overrides {
        [
            DimensionSite::ref_with_expr("a", col("a_raw").expect("a name")),
            DimensionSite::ref_with_expr("b", col("b_raw").expect("a name")),
        ]
    } else {
        [DimensionSite::r#ref("a"), DimensionSite::r#ref("b")]
    };
    insert_dataset(
        &mut model,
        Dataset::builder("rows")
            .table("warehouse.rows")
            .dimensions(sites)
            .build(),
    );
    model
}

#[tokio::test]
async fn mutually_referential_bodies_are_refused_on_the_source_that_cycles() {
    match compile_error(mutually_referential_model(false)).await {
        CompileError::CyclicReference { cycle } => {
            assert!(
                cycle.contains(&"a".to_owned()) && cycle.contains(&"b".to_owned()),
                "the members are named: {cycle:?}"
            );
        }
        other => panic!("expected CyclicReference, got {other:?}"),
    }
}

#[tokio::test]
async fn the_same_two_names_compile_where_their_bodies_do_not_cycle() {
    let manifest = compile_model(mutually_referential_model(true)).await;
    assert_eq!(
        tree(&manifest, signature(&manifest, "rows", "a")),
        Expr::Cast {
            input: Box::new(column("ps0", "a_raw")),
            target: DataType::BigInt,
            on_failure: CastFailure::Error,
        },
        "a bare column's type is pinned at the boundary by the declared type"
    );
}

#[tokio::test]
async fn a_reference_sites_deferred_body_lowers_the_same_way() {
    let mut model = SemanticModel::default();
    model.name = "finance".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("doubled")
            .data_type(DataType::Integer)
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::ref_with_expr(
                "doubled",
                measure("revenue").expect("a name") + measure("revenue").expect("a name"),
            ))
            .build(),
    );

    let manifest = compile_model(model).await;
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "doubled")),
        Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(integer_sum("ps0", "revenue")),
            right: Box::new(integer_sum("ps0", "revenue")),
        },
        "the deferred body is this source's body"
    );
}

#[tokio::test]
async fn an_exposed_metric_with_no_site_or_pooled_expression_is_refused() {
    let mut model = SemanticModel::default();
    model.name = "finance".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("doubled")
            .data_type(DataType::Integer)
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("doubled"))
            .build(),
    );

    match compile_error(model).await {
        CompileError::MetricWithoutExpression {
            semantics,
            semantic_source,
        } => {
            assert_eq!(semantics, "doubled");
            assert_eq!(semantic_source, "orders");
        }
        other => panic!("expected MetricWithoutExpression, got {other:?}"),
    }
}

#[tokio::test]
async fn an_inline_metric_with_no_expression_is_refused() {
    let mut model = SemanticModel::default();
    model.name = "finance".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::inline(
                Metric::builder("doubled")
                    .data_type(DataType::Integer)
                    .build(),
            ))
            .build(),
    );

    match compile_error(model).await {
        CompileError::MetricWithoutExpression {
            semantics,
            semantic_source,
        } => {
            assert_eq!(semantics, "doubled");
            assert_eq!(semantic_source, "orders");
        }
        other => panic!("expected MetricWithoutExpression, got {other:?}"),
    }
}

#[tokio::test]
async fn a_bodied_model_compiles_to_the_same_ids_twice() {
    let one = compile_model(two_datasets_over_one_bodied_metric()).await;
    let two = compile_model(two_datasets_over_one_bodied_metric()).await;
    assert_eq!(
        one.semantic_signatures.keys().collect::<Vec<_>>(),
        two.semantic_signatures.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        one.expressions.physical.keys().collect::<Vec<_>>(),
        two.expressions.physical.keys().collect::<Vec<_>>()
    );
    for (id, signature) in &one.semantic_signatures {
        assert_eq!(
            signature.expression_id, two.semantic_signatures[id].expression_id,
            "`{id}` roots the same tree on both compiles"
        );
    }
}
