use std::collections::BTreeSet;

use semstrait_common::io::InMemoryFileSystem;
use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{
    compile, mint_expr_id, CompileError, ExprLayer, ExprRoot, Link, ManifestExpression,
    PhysicalExprId, SemanticManifest, StoredExpr,
};
use semstrait_model::expr::{dim, field, lit, measure, metric};
use semstrait_model::{
    Dataset, Dimension, DimensionSite, Measure, Metric, SemanticMapping, SemanticModel,
    StorageFormat,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, TypeError};

mod common;
use common::{
    agg, bodied, cast, column, compile_error, insert_dataset, insert_dimension as insert,
    insert_measure, integer_sum, integer_sum_filtered, is, link, path_token, revenue_and_refunds,
    signature, source_alias, source_signatures, stored_tree, string, test_provenance, zone_model,
};

const LOCATION: &str = "s3://bucket/warehouse/orders/year=2024/";

fn fixture_model() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();

    insert(
        &mut model,
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert(
        &mut model,
        Dimension::builder("net.revenue")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert(
        &mut model,
        Dimension::builder("load_partition")
            .data_type(DataType::String)
            .metadata(path_token(3))
            .build(),
    );

    let revenue = Measure::builder("revenue")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .build();
    let id = revenue.entity_id().expect("the mint is eager").to_owned();
    model.measures.insert(id, revenue);

    let orders = Dataset::builder("orders")
        .path(LOCATION)
        .format(StorageFormat::Parquet)
        .dimensions(
            ["region", "net.revenue", "load_partition"].map(semstrait_model::DimensionSite::r#ref),
        )
        .measure(semstrait_model::MeasureSite::r#ref("revenue"))
        .build();
    let id = orders.body.base.id.clone().expect("the mint is eager");
    model.datasets.insert(id, orders);

    model
}

async fn compile_model(model: SemanticModel) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile clean");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

fn realisation<'m>(manifest: &'m SemanticManifest, name: &str) -> &'m ManifestExpression {
    let definition = manifest
        .semantic_definitions
        .by_name(name)
        .unwrap_or_else(|| panic!("{name} is not registered"));
    let expr_id = manifest
        .semantic_sources
        .values()
        .flat_map(|source| &source.semantic_signature_ids)
        .filter_map(|id| manifest.semantic_signatures.get(id))
        .find(|signature| signature.definition_id == definition.id)
        .map(|signature| signature.expression_id)
        .unwrap_or_else(|| panic!("{name} has no pooled realisation"));
    manifest
        .expressions
        .get(&expr_id)
        .unwrap_or_else(|| panic!("{name}'s expression id resolves in the store"))
}

fn realised_tree(manifest: &SemanticManifest, name: &str) -> semstrait_ir::PhysicalExpr {
    stored_tree(manifest, realisation(manifest, name).id)
}

#[tokio::test]
async fn every_covered_semantics_emits_with_its_layer() {
    let manifest = compile_model(fixture_model()).await;

    assert_eq!(manifest.semantic_definitions.len(), 4);
    assert_eq!(manifest.expressions.len(), 4, "one entry per realisation");

    let region = realisation(&manifest, "region");
    assert_eq!(
        realised_tree(&manifest, "region"),
        cast(column("ps0", "region"), DataType::String)
    );
    assert_eq!(
        region.layer,
        ExprLayer::Scalar,
        "a column semantics carries its layer"
    );

    let revenue = realisation(&manifest, "revenue");
    assert_eq!(
        realised_tree(&manifest, "revenue"),
        integer_sum("ps0", "revenue"),
        "a call of `sum` over the Measure's column"
    );
    assert_eq!(revenue.layer, ExprLayer::PostAggregate);
}

#[tokio::test]
async fn a_pool_entry_carries_its_own_id() {
    let manifest = compile_model(fixture_model()).await;
    for (key, entry) in &manifest.expressions.physical {
        assert_eq!(&entry.id, key, "an entry's id is its map key");
    }
}

#[tokio::test]
async fn physical_sources_names_the_consuming_source() {
    let manifest = compile_model(fixture_model()).await;
    let source_id = manifest
        .sources
        .keys()
        .next()
        .expect("one file-backed source")
        .clone();
    for entry in manifest.expressions.physical.values() {
        assert_eq!(
            entry.physical_sources,
            BTreeSet::from([source_id.clone()]),
            "every entry of a realisation names the source its data kind reads"
        );
    }
}

#[tokio::test]
async fn a_dataset_column_named_with_a_dot_is_one_name_under_its_alias() {
    let manifest = compile_model(fixture_model()).await;
    assert_eq!(
        realised_tree(&manifest, "net.revenue"),
        cast(column("ps0", "net.revenue"), DataType::String),
        "a Dataset's tree holds the column name whole, after its source's alias"
    );
}

#[tokio::test]
async fn a_metadata_dimension_is_an_ordinary_pooled_literal() {
    let manifest = compile_model(fixture_model()).await;
    let entry = realisation(&manifest, "load_partition");
    assert_eq!(
        realised_tree(&manifest, "load_partition"),
        Expr::Literal(Literal::String("year=2024".into())),
        "the whole raw segment, never split on `=`"
    );
    assert_eq!(entry.layer, ExprLayer::Scalar);

    let definition = manifest
        .semantic_definitions
        .by_name("load_partition")
        .expect("`load_partition` is registered");
    let stored = serde_json::to_value(&definition.role_definition).expect("serialize Dimension");
    assert_eq!(
        stored,
        serde_json::json!({
            "type": "dimension",
            "content": { "category": { "metadata": { "source": { "path": { "token": 3 } } } } }
        }),
        "the manifest's own `DimensionType::Metadata` still stores `path`/`token` unchanged: {stored}"
    );
}

fn metadata_only_model(location: Option<&str>, token: u32, data_type: DataType) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert(
        &mut model,
        Dimension::builder("load_partition")
            .data_type(data_type)
            .metadata(path_token(token))
            .build(),
    );
    let dataset = Dataset::builder("orders");
    let dataset = match location {
        Some(location) => dataset.path(location).format(StorageFormat::Parquet),
        None => dataset.table("warehouse.orders"),
    };
    let orders = dataset
        .dimension(semstrait_model::DimensionSite::r#ref("load_partition"))
        .build();
    let id = orders.body.base.id.clone().expect("the mint is eager");
    model.datasets.insert(id, orders);
    model
}

#[tokio::test]
async fn a_metadata_segment_the_location_lacks_fails_fast() {
    match compile_error(metadata_only_model(Some(LOCATION), 42, DataType::String)).await {
        CompileError::MetadataSegmentUnavailable {
            semantic,
            locator,
            token,
            ..
        } => {
            assert_eq!(semantic, "load_partition");
            assert_eq!(token, 42);
            assert_eq!(locator, LOCATION, "names the base path");
        }
        other => panic!("expected MetadataSegmentUnavailable, got {other:?}"),
    }
}

#[tokio::test]
async fn a_metadata_dimension_over_a_table_source_fails_fast() {
    assert!(matches!(
        compile_error(metadata_only_model(None, 0, DataType::String)).await,
        CompileError::MetadataSegmentUnavailable { .. }
    ));
}

#[tokio::test]
async fn a_metadata_value_reads_as_its_declared_type() {
    let model = metadata_only_model(Some("s3://bucket/2024/"), 1, DataType::Integer);
    let manifest = compile_model(model).await;
    assert_eq!(
        realised_tree(&manifest, "load_partition"),
        Expr::Literal(Literal::Integer(2024))
    );
    manifest.validate().expect("the compiled artifact is valid");
}

#[tokio::test]
async fn a_metadata_value_that_does_not_read_as_its_declared_type_is_refused() {
    match compile_error(metadata_only_model(Some(LOCATION), 3, DataType::Integer)).await {
        CompileError::TypeError {
            node,
            error: TypeError::UntypedLiteral {
                literal, context, ..
            },
        } => {
            assert_eq!(node, "load_partition");
            assert_eq!(literal, Literal::String("year=2024".into()));
            assert_eq!(context, DataType::Integer);
        }
        other => panic!("expected the untyped literal's TypeError, got {other:?}"),
    }
}

#[tokio::test]
async fn the_compiled_artifact_is_valid() {
    let manifest = compile_model(fixture_model()).await;
    match manifest.validate() {
        Ok(warnings) => assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}"),
        Err(faults) => panic!("a compiled manifest is valid by construction: {faults:?}"),
    }

    let mut named = BTreeSet::new();
    for signature in manifest.semantic_signatures.values() {
        let id = signature.expression_id;
        assert!(
            manifest.expressions.get(&id).is_some(),
            "{id:?} resolves in the store"
        );
        assert!(
            named.insert(id),
            "{id:?} is named by more than one signature"
        );
    }
}

#[tokio::test]
async fn two_sources_mapping_one_semantics_to_one_column_name_produce_two_entries() {
    let mut model = SemanticModel::default();
    model.name = "people".to_owned();
    insert(
        &mut model,
        Dimension::builder("last_name")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    for name in ["contacts", "users"] {
        let dataset = Dataset::builder(name)
            .table(format!("warehouse.{name}"))
            .dimension(semstrait_model::DimensionSite::r#ref("last_name"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("last_name", "surname")
                    .build(),
            )
            .build();
        let id = dataset.body.base.id.clone().expect("the mint is eager");
        model.datasets.insert(id, dataset);
    }

    let manifest = compile_model(model).await;
    assert_eq!(manifest.semantic_definitions.len(), 1, "one definition");

    let signatures: Vec<_> = manifest
        .semantic_sources
        .values()
        .map(|source| {
            assert_eq!(
                source.semantic_signature_ids.len(),
                1,
                "one signature per source"
            );
            let signature = manifest
                .semantic_signatures
                .get(&source.semantic_signature_ids[0])
                .expect("owned signature resolves");
            (source_alias(&manifest, source.name.as_str()), signature)
        })
        .collect();
    assert_eq!(
        signatures.len(),
        2,
        "one signature owned by each of the two sources"
    );
    assert_ne!(
        signatures[0].1.expression_id, signatures[1].1.expression_id,
        "the two owners' ids differ for the same column name"
    );
    for (alias, signature) in &signatures {
        assert_eq!(
            stored_tree(&manifest, signature.expression_id),
            cast(column(alias, "surname"), DataType::String),
            "each names its own physical source's alias"
        );
    }

    let surname_entries = manifest
        .expressions
        .physical
        .keys()
        .filter(|id| {
            (manifest.expressions.columns(id).iter()).any(|(_, column)| column.name() == "surname")
        })
        .count();
    assert_eq!(
        surname_entries, 2,
        "the store holds two entries, each with its own Column(surname)"
    );
}

#[tokio::test]
async fn a_measures_entry_holds_its_whole_tree() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    let dataset = Dataset::builder("orders")
        .table("warehouse.orders")
        .measure(semstrait_model::MeasureSite::r#ref("revenue"))
        .semantic_mapping(
            SemanticMapping::builder()
                .column("revenue", "amount")
                .build(),
        )
        .build();
    let id = dataset.body.base.id.clone().expect("the mint is eager");
    model.datasets.insert(id, dataset);

    let manifest = compile_model(model).await;
    let source = manifest.semantic_sources.values().next().expect("one leaf");
    assert_eq!(source.semantic_signature_ids.len(), 1);
    let signature = manifest
        .semantic_signatures
        .get(&source.semantic_signature_ids[0])
        .expect("owned signature resolves");

    let entry = manifest
        .expressions
        .get(&signature.expression_id)
        .expect("the entry resolves");
    assert!(
        matches!(entry.expr.expr(), Expr::Cast { .. }),
        "expected the cast to the declared type at the root, got {:?}",
        entry.expr
    );
    assert_eq!(manifest.expressions.len(), 1, "the Measure's one entry");
    assert_eq!(
        stored_tree(&manifest, signature.expression_id),
        integer_sum("ps0", "amount")
    );
    let consumer = manifest.sources.keys().next().expect("one source");
    assert_eq!(
        entry.physical_sources,
        BTreeSet::from([consumer.clone()]),
        "the entry names the source that consumes it"
    );
}

fn expression_id(manifest: &SemanticManifest, name: &str) -> PhysicalExprId {
    realisation(manifest, name).id
}

fn store(manifest: &mut SemanticManifest, id: PhysicalExprId, expr: Expr<Link>) {
    manifest
        .expressions
        .physical
        .get_mut(&id)
        .expect("the entry is stored")
        .expr = StoredExpr::from(expr);
}

fn read(manifest: &SemanticManifest, id: PhysicalExprId) -> Option<PhysicalExpr> {
    manifest
        .expressions
        .reassemble(&id, &source_signatures(manifest, id), &[])
}

#[tokio::test]
async fn reading_a_link_gives_the_linked_entrys_tree() {
    let mut manifest = compile_model(zone_model()).await;
    let (region, zone) = (
        expression_id(&manifest, "region"),
        expression_id(&manifest, "zone"),
    );
    let copy = stored_tree(&manifest, zone);
    store(&mut manifest, zone, common::zone(link(region)));
    assert_eq!(read(&manifest, zone), Some(copy));
}

#[tokio::test]
async fn reading_a_dangling_link_or_a_cycle_gives_no_tree() {
    let compiled = compile_model(zone_model()).await;
    let (region, zone) = (
        expression_id(&compiled, "region"),
        expression_id(&compiled, "zone"),
    );
    let absent = mint_expr_id("absent", ExprRoot::SignatureExpression);
    for links in [
        vec![(zone, absent)],
        vec![(zone, zone)],
        vec![(zone, region), (region, zone)],
    ] {
        let mut manifest = compiled.clone();
        for &(entry, linked) in &links {
            store(&mut manifest, entry, link(linked));
        }
        assert_eq!(read(&manifest, zone), None, "the links {links:?}");
    }
}

#[tokio::test]
async fn reading_a_link_to_a_filtered_measure_joins_its_filter_onto_its_aggregate_call() {
    let net = Metric::builder("net")
        .data_type(DataType::Integer)
        .expr(measure("revenue").expect("a name") - measure("refunds").expect("a name"))
        .build();
    let mut manifest = compile_model(revenue_and_refunds(vec![net])).await;
    let [status, revenue, refunds, net] =
        ["status", "revenue", "refunds", "net"].map(|name| expression_id(&manifest, name));
    let paid = (signature(&manifest, "orders", "revenue").filter_expr_id)
        .expect("`revenue`'s filter is its own entry");
    let copy = stored_tree(&manifest, net);
    assert_eq!(
        copy,
        integer_sum_filtered("ps0", "revenue", Some(is("status", "paid")))
            - integer_sum("ps0", "refunds")
    );
    store(
        &mut manifest,
        paid,
        ExprOperationExt::eq(link(status), string("paid")),
    );
    store(&mut manifest, net, link(revenue) - link(refunds));
    assert_eq!(read(&manifest, net), Some(copy));
    assert_eq!(
        read(&manifest, revenue),
        Some(integer_sum("ps0", "revenue"))
    );
}

#[tokio::test]
async fn reading_a_link_to_a_filtered_metric_joins_its_filter_after_any_filter_there() {
    let m1 = Metric::builder("m1")
        .data_type(DataType::Integer)
        .expr(measure("revenue").expect("a name") + measure("refunds").expect("a name"))
        .filter(ExprOperationExt::eq(
            dim("channel").expect("a name"),
            lit("web"),
        ))
        .build();
    let m2 = Metric::builder("m2")
        .data_type(DataType::Integer)
        .expr(metric("m1").expect("a name") * lit(2))
        .build();
    let mut manifest = compile_model(revenue_and_refunds(vec![m1, m2])).await;
    let [revenue, refunds, m1, m2] =
        ["revenue", "refunds", "m1", "m2"].map(|name| expression_id(&manifest, name));
    let copy = stored_tree(&manifest, m2);
    assert_eq!(
        copy,
        (integer_sum_filtered(
            "ps0",
            "revenue",
            Some(is("status", "paid") & is("channel", "web"))
        ) + integer_sum_filtered("ps0", "refunds", Some(is("channel", "web"))))
            * Expr::Literal(Literal::Integer(2))
    );
    store(&mut manifest, m1, link(revenue) + link(refunds));
    store(
        &mut manifest,
        m2,
        link(m1) * Expr::Literal(Literal::Integer(2)),
    );
    assert_eq!(read(&manifest, m2), Some(copy));
}

fn dataset_name_model() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "ads".to_owned();
    insert(
        &mut model,
        Dimension::builder("dataset_name")
            .data_type(DataType::String)
            .metadata(path_token(1))
            .build(),
    );
    insert(
        &mut model,
        bodied(
            "ds2",
            DataType::String,
            field("dataset_name").expect("a name"),
        ),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .path("s3://bucket/google_ads/")
            .format(StorageFormat::Parquet)
            .dimensions(["dataset_name", "ds2"].map(DimensionSite::r#ref))
            .build(),
    );
    model
}

#[tokio::test]
async fn reading_a_bare_link_to_a_literal_entry_gives_the_literal() {
    let mut manifest = compile_model(dataset_name_model()).await;
    let (dataset_name, ds2) = (
        expression_id(&manifest, "dataset_name"),
        expression_id(&manifest, "ds2"),
    );
    assert_eq!(stored_tree(&manifest, ds2), string("google_ads"));
    store(&mut manifest, ds2, link(dataset_name));
    assert_eq!(read(&manifest, ds2), Some(string("google_ads")));
}

#[tokio::test]
async fn a_metric_named_before_the_measures_it_links_compiles() {
    let a_net = Metric::builder("a_net")
        .data_type(DataType::Integer)
        .expr(measure("revenue").expect("a name") - measure("refunds").expect("a name"))
        .build();
    let manifest = common::compile_model(revenue_and_refunds(vec![a_net])).await;
    let [revenue, refunds, a_net] =
        ["revenue", "refunds", "a_net"].map(|name| expression_id(&manifest, name));
    assert_eq!(
        manifest.expressions.physical[&a_net]
            .expr
            .links()
            .collect::<Vec<_>>(),
        [revenue, refunds]
    );
}

#[tokio::test]
async fn the_store_is_identical_across_two_compiles() {
    let first = compile_model(fixture_model()).await;
    let second = compile_model(fixture_model()).await;
    assert_eq!(
        first.expressions, second.expressions,
        "the whole store, not just the signatures that root into it, is identical"
    );
}
