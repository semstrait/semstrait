mod common;

use common::{agg, by_name, column, seeded_fs, signature, test_provenance, tree};

use semstrait_catalog::FileFormat;
use semstrait_common::io::{FileSystem, InMemoryFileSystem};
use semstrait_common::Granularity;
use semstrait_ir::MultisetOp;
use semstrait_manifest::{
    compile, mint_path_source_id, mint_reduction_leaf_id, mint_semantic_source_id,
    mint_table_source_id, RoleDefinition, SemanticManifest,
};
use semstrait_model::expr::col;
use semstrait_model::{
    parse, DataKindFilter, Dataset, DatasetBuilder, Dimension, DimensionSite, Measure, MeasureSite,
    SemanticModel, StorageFormat, TemporalShape,
};
use semstrait_sql::datatype::DataType;

fn leaf(name: &str) -> DatasetBuilder {
    Dataset::builder(name)
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
}

fn model_with(dataset: Dataset) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();

    let region = Dimension::builder("region")
        .data_type(DataType::String)
        .categorical()
        .build();
    model
        .dimensions
        .insert(region.name().as_str().to_owned(), region);

    let revenue = Measure::builder("revenue")
        .data_type(DataType::Double)
        .agg(agg("sum"))
        .build();
    model
        .measures
        .insert(revenue.name().as_str().to_owned(), revenue);

    model
        .datasets
        .insert(dataset.body.base.name.clone(), dataset);
    model
}

async fn compiled(model: SemanticModel, fs: &dyn FileSystem) -> SemanticManifest {
    let (manifest, warnings) = compile(model, None, fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

fn names(manifest: &SemanticManifest) -> Vec<&str> {
    let mut out: Vec<&str> = manifest
        .semantic_sources
        .values()
        .map(|entry| entry.name.as_str())
        .collect();
    out.sort_unstable();
    out
}

fn reads<'m>(manifest: &'m SemanticManifest, leaf_name: &str) -> &'m str {
    let RoleDefinition::Dataset { source_id, .. } = &by_name(manifest, leaf_name).role_definition
    else {
        panic!("`{leaf_name}` must be a leaf `Dataset`");
    };
    assert!(
        manifest.sources.contains_key(source_id),
        "a leaf's single reference resolves in `physical_sources`"
    );
    source_id.as_str()
}

const BY_REGION: &str = "s3://lake/orders/*/*.parquet";

async fn two_regions() -> InMemoryFileSystem {
    seeded_fs([
        "s3://lake/orders/us/part-0.parquet",
        "s3://lake/orders/eu/part-0.parquet",
    ])
    .await
}

fn by_region(dataset: DatasetBuilder) -> SemanticModel {
    model_with(
        dataset
            .path(BY_REGION)
            .format(StorageFormat::Parquet)
            .build(),
    )
}

#[tokio::test]
async fn a_multi_source_dataset_becomes_one_leaf_per_source_under_a_unionset() {
    let manifest = compiled(by_region(leaf("orders")), &two_regions().await).await;

    assert_eq!(
        names(&manifest),
        vec!["orders", "orders_eu", "orders_us"],
        "one leaf per physical source, plus the composer under the authored name"
    );

    let composer = by_name(&manifest, "orders");
    let RoleDefinition::Unionset { branches, .. } = &composer.role_definition else {
        panic!("a multi-source Dataset's public face is a Unionset");
    };
    let expected: Vec<String> = ["eu", "us"]
        .map(|region| {
            let source = mint_path_source_id(
                &format!("s3://lake/orders/{region}/"),
                Some(FileFormat::Parquet),
            );
            mint_reduction_leaf_id(&composer.id, &source)
        })
        .into();
    assert_eq!(
        branches, &expected,
        "the branches are the reduction leaves' ids, in name order"
    );
    assert_eq!(
        composer.id,
        mint_semantic_source_id("orders"),
        "the composer holds the authored name, so it holds the authored name's id"
    );

    let first = reads(&manifest, "orders_eu");
    let second = reads(&manifest, "orders_us");
    assert_ne!(
        first, second,
        "a leaf reads one source, and not the same one"
    );
    assert_eq!(manifest.sources.len(), 2);
    let mut read: Vec<&str> = vec![first, second];
    read.sort_unstable();
    let mut present: Vec<&str> = manifest.sources.keys().map(String::as_str).collect();
    present.sort_unstable();
    assert_eq!(
        read, present,
        "every expanded source is read by exactly one leaf"
    );

    assert!(
        manifest.validate().is_ok(),
        "a compiled manifest must satisfy its own validator"
    );
}

#[tokio::test]
async fn a_nested_branch_reduces_on_the_same_terms() {
    let (model, warnings) = parse(
        r"semantic_model:
  name: sales
  dimensions:
    - name: region
      data_type: string
      type: categorical
  unionsets:
    - name: all_orders
      mode: all
      dimensions:
        - ref: region
      datasets:
        - name: eu
          extras:
            semantic_mapping: {region: region}
            storage:
              tables: [warehouse.orders_north, warehouse.orders_south]
        - name: us
          extras:
            semantic_mapping: {region: region}
            storage:
              tables: [warehouse.orders_us]
",
    )
    .expect("the document parses")
    .build()
    .expect("the document builds");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    let fs = InMemoryFileSystem::new();
    let manifest = compiled(model, &fs).await;

    assert_eq!(
        names(&manifest),
        vec![
            "all_orders",
            "all_orders.eu",
            "all_orders.eu_orders_north",
            "all_orders.eu_orders_south",
            "all_orders.us",
        ],
        "the reducing branch keeps its authored nested name and gains two leaves"
    );

    let branch = by_name(&manifest, "all_orders.eu");
    assert!(matches!(
        branch.role_definition,
        RoleDefinition::Unionset { .. }
    ));
    assert!(!branch.public, "an authored nested unit stays private");
    assert_eq!(
        branch.originators,
        vec![mint_semantic_source_id("all_orders")],
        "one originator, the nearest authored ancestor"
    );
    for deep in ["all_orders.eu_orders_north", "all_orders.eu_orders_south"] {
        assert_eq!(
            by_name(&manifest, deep).originators,
            vec![mint_semantic_source_id("all_orders.eu")],
            "a reduction leaf's originator is the unit that was reduced"
        );
    }
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn a_single_source_dataset_stays_a_leaf_with_the_authored_name() {
    let fs = InMemoryFileSystem::new();
    let manifest = compiled(
        model_with(leaf("orders").table("warehouse.orders").build()),
        &fs,
    )
    .await;

    assert_eq!(names(&manifest), vec!["orders"], "no wrapper at one source");
    let entry = by_name(&manifest, "orders");
    assert!(matches!(
        entry.role_definition,
        RoleDefinition::Dataset { .. }
    ));
    assert!(entry.public, "the authored name is what a request may name");
    assert_eq!(entry.id, mint_semantic_source_id("orders"));
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn a_reductions_unionset_takes_u_and_each_reduction_leaf_d() {
    let fs = seeded_fs([
        "s3://lake/orders/eu/part-0.parquet",
        "s3://lake/orders/uk/part-0.parquet",
        "s3://lake/orders/us/part-0.parquet",
    ])
    .await;
    let manifest = compiled(by_region(leaf("orders")), &fs).await;

    let table: Vec<(&str, &str, &str)> = manifest
        .semantic_sources
        .values()
        .map(|entry| (&entry.id[..8], entry.name.as_str(), entry.alias.as_str()))
        .collect();
    assert_eq!(
        table,
        vec![
            ("752274fc", "orders_uk", "ssd0"),
            ("9751804e", "orders_eu", "ssd1"),
            ("af3fc250", "orders_us", "ssd2"),
            ("cd3e4c16", "orders", "ssu3"),
        ],
        "the Unionset's `u` and each leaf's `d`, indexed in `EntityId` order"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn the_synthesised_unionset_is_all_mode_and_no_wider_than_its_leaves() {
    let manifest = compiled(by_region(leaf("orders")), &two_regions().await).await;

    let composer = by_name(&manifest, "orders");
    let RoleDefinition::Unionset { mode, .. } = &composer.role_definition else {
        panic!("a multi-source Dataset's public face is a Unionset");
    };
    assert_eq!(
        *mode,
        MultisetOp::UnionAll,
        "never `Unique`: folding would delete rows"
    );

    for name in ["orders_eu", "orders_us"] {
        assert_eq!(
            composer.semantic_bitmask,
            by_name(&manifest, name).semantic_bitmask,
            "the produced unit cannot be wider than the leaf that produced it"
        );
    }
    assert_eq!(
        composer.semantic_bitmask.count_ones(),
        2,
        "and it is not narrower either: both exposed semantics are servable"
    );
}

#[tokio::test]
async fn the_synthesised_unionsets_named_filter_reads_its_leaves_column() {
    let flagged = DataKindFilter::builder("flagged")
        .expr(!col("flag").expect("a name"))
        .build();
    let manifest = compiled(
        by_region(leaf("orders").filter(flagged)),
        &two_regions().await,
    )
    .await;

    let composer = by_name(&manifest, "orders");
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "flagged")),
        column(composer.alias.as_str(), "flagged"),
        "the stack's own column, which each leaf's signature gives"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn only_the_unionset_holding_the_authored_name_is_public() {
    let manifest = compiled(by_region(leaf("orders")), &two_regions().await).await;

    assert!(
        by_name(&manifest, "orders").public,
        "the entry the authored name resolves to is public"
    );
    for name in ["orders_eu", "orders_us"] {
        let branch = by_name(&manifest, name);
        assert!(!branch.public, "a derived name is private");
        assert_eq!(
            branch.originators,
            vec![mint_semantic_source_id("orders")],
            "the nearest authored ancestor, and exactly one of them"
        );
    }
}

#[tokio::test]
async fn a_folder_wildcard_names_its_leaves_for_the_folders_it_matches() {
    let fs = seeded_fs(["s3://lake/eu/orders.parquet", "s3://lake/us/orders.parquet"]).await;
    let manifest = compiled(
        model_with(
            leaf("orders")
                .path("s3://lake/*/orders.parquet")
                .format(StorageFormat::Parquet)
                .build(),
        ),
        &fs,
    )
    .await;

    assert_eq!(names(&manifest), vec!["orders", "orders_eu", "orders_us"]);
}

#[tokio::test]
async fn literal_paths_name_their_leaves_for_their_base_paths() {
    let paths = ["s3://lake/eu/orders.parquet", "s3://lake/us/orders.parquet"];
    let fs = seeded_fs(paths).await;
    let manifest = compiled(
        model_with(
            leaf("orders")
                .paths(paths)
                .format(StorageFormat::Parquet)
                .build(),
        ),
        &fs,
    )
    .await;

    assert_eq!(names(&manifest), vec!["orders", "orders_eu", "orders_us"]);
}

#[tokio::test]
async fn tables_name_their_leaves_for_the_table_name() {
    let fs = InMemoryFileSystem::new();
    let manifest = compiled(
        model_with(
            leaf("orders")
                .tables(["warehouse.orders_eu", "warehouse.orders_us"])
                .build(),
        ),
        &fs,
    )
    .await;

    assert_eq!(
        names(&manifest),
        vec!["orders", "orders_orders_eu", "orders_orders_us"]
    );
}

#[tokio::test]
async fn two_tables_of_one_name_are_two_leaves_of_one_label() {
    let fs = InMemoryFileSystem::new();
    let manifest = compiled(
        model_with(leaf("orders").tables(["eu.orders", "us.orders"]).build()),
        &fs,
    )
    .await;

    assert_eq!(
        names(&manifest),
        vec!["orders", "orders_orders", "orders_orders"]
    );
    let dataset = mint_semantic_source_id("orders");
    let mut expected = ["eu.orders", "us.orders"]
        .map(|table| mint_reduction_leaf_id(&dataset, &mint_table_source_id(table)));
    expected.sort();
    let RoleDefinition::Unionset { branches, .. } = &by_name(&manifest, "orders").role_definition
    else {
        panic!("a multi-source Dataset's public face is a Unionset");
    };
    assert_eq!(branches, &expected, "name order, then id order");
    assert!(manifest.validate().is_ok());
}

async fn keyed_reduction() -> SemanticManifest {
    compiled(
        by_region(
            Dataset::builder("orders")
                .dimension(DimensionSite::r#ref("region").with_key())
                .measure(MeasureSite::r#ref("revenue")),
        ),
        &two_regions().await,
    )
    .await
}

#[tokio::test]
async fn every_dataset_signature_tree_names_the_relation_its_owner_reads() {
    let manifest = keyed_reduction().await;

    let mut leaf_roots = 0usize;
    let mut composed_roots = 0usize;
    for entry in manifest.semantic_sources.values() {
        let roots = entry
            .semantic_signature_ids
            .iter()
            .filter_map(|id| manifest.semantic_signatures.get(id))
            .flat_map(|signature| {
                [Some(signature.expression_id), signature.filter_expr_id]
                    .into_iter()
                    .flatten()
            })
            .map(|root| {
                manifest
                    .expressions
                    .get(&root)
                    .expect("a signature's root is answered for by the pool")
            });
        match &entry.role_definition {
            RoleDefinition::Dataset { source_id, .. } => {
                for pooled in roots {
                    assert!(
                        pooled.physical_sources.contains(source_id),
                        "`{}` reads `{source_id}`, so its tree must name it as a consumer; \
                         names {:?}",
                        entry.id,
                        pooled.physical_sources
                    );
                    leaf_roots += 1;
                }
            }
            _ => {
                for pooled in roots {
                    assert!(
                        pooled.physical_sources.is_empty(),
                        "a composed projection consumes no relation of its own; `{}` names {:?}",
                        entry.id,
                        pooled.physical_sources
                    );
                    composed_roots += 1;
                }
            }
        }
    }
    assert!(
        leaf_roots >= 4,
        "two leaves over two semantics is at least four checked roots, got {leaf_roots}"
    );
    assert!(
        composed_roots > 0,
        "the stacked surface owns signatures too, so the exemption is exercised"
    );
    let validated = manifest.validate();
    assert!(
        validated.is_ok(),
        "a composer's empty consumer set is not a violation: {:?}",
        validated.err()
    );
}

fn attributing_sources(manifest: &SemanticManifest) -> Vec<String> {
    let mut asserting: Vec<String> = manifest
        .semantic_sources
        .values()
        .filter(|source| {
            source
                .semantic_signature_ids
                .iter()
                .filter_map(|id| manifest.semantic_signatures.get(id))
                .any(|signature| !signature.keysets.is_empty())
        })
        .map(|source| source.id.clone())
        .collect();
    asserting.sort();
    asserting.dedup();
    asserting
}

#[tokio::test]
async fn authored_keysets_land_on_every_leaf() {
    let manifest = keyed_reduction().await;

    for leaf in ["orders_eu", "orders_us"] {
        assert!(
            attributing_sources(&manifest).contains(&by_name(&manifest, leaf).id),
            "one authored shape, attributed on {leaf}"
        );
    }
    assert_eq!(
        manifest.keysets.len(),
        3,
        "one authored name over three sources is three entities (ADR 0048)"
    );
}

#[tokio::test]
async fn authored_keysets_land_on_the_unionset_too() {
    let manifest = keyed_reduction().await;
    assert!(
        attributing_sources(&manifest).contains(&mint_semantic_source_id("orders")),
        "the composer holding the authored name attributes the authored keyset"
    );
}

#[tokio::test]
async fn an_absent_keyset_declaration_mints_nothing() {
    let manifest = compiled(by_region(leaf("orders")), &two_regions().await).await;

    assert!(
        manifest.keysets.is_empty(),
        "no `is_key:` anywhere means no keyset entity, on any unit of the reduction"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn an_authored_grain_survives_the_reduction() {
    let mut model = by_region(
        Dataset::builder("orders")
            .dimension(DimensionSite::r#ref("region"))
            .dimension(DimensionSite::r#ref("occurred_at"))
            .measure(MeasureSite::r#ref("revenue"))
            .temporal(TemporalShape::timeseries(
                "occurred_at",
                Some(Granularity::Day),
            )),
    );
    let occurred_at = Dimension::builder("occurred_at")
        .data_type(DataType::Date)
        .categorical()
        .build();
    model
        .dimensions
        .insert(occurred_at.name().as_str().to_owned(), occurred_at);
    let manifest = compiled(model, &two_regions().await).await;

    for name in ["orders_eu", "orders_us"] {
        let shape = by_name(&manifest, name)
            .temporal
            .as_ref()
            .unwrap_or_else(|| panic!("`{name}` must carry the authored shape"));
        assert_eq!(
            shape.granularity,
            Some(Granularity::Day),
            "the authored granularity survives"
        );
        let occurred_at = manifest
            .semantic_definitions
            .by_name("occurred_at")
            .expect("the identifying dimension is registered");
        assert_eq!(
            shape.temporal_type,
            semstrait_ir::TemporalType::timeseries(occurred_at.id.clone()),
            "the identifying dimension is resolved to its minted id"
        );
    }
    assert!(manifest.validate().is_ok());
}
