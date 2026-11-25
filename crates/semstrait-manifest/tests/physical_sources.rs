mod common;

use common::golden::{self, Golden};
use common::{agg, seeded_fs, test_provenance, warehouse_orders, MapCatalog};

use bytes::Bytes;
use semstrait_catalog::{
    FileFormat, NoopCatalogProvider, Partition, PartitionTransform, Schema as NativeSchema,
    SchemaColumn as NativeColumn,
};
use semstrait_common::io::{FileEntry, FileSystem, FileSystemErrorKind, InMemoryFileSystem, Path};
use semstrait_common::{Diagnostic, Dialect};
use semstrait_manifest::{
    compile, CompileError, SchemaProvenance, SemanticManifest, SourceAlias, StorageTarget, Uri,
};
use semstrait_model::{
    Dataset, DatasetBuilder, Dimension, DimensionSite, Measure, MeasureSite, SemanticModel,
    StorageFormat,
};
use semstrait_sql::datatype::DataType;

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

fn leaf(name: &str) -> DatasetBuilder {
    Dataset::builder(name)
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
}

#[tokio::test]
async fn paths_and_tables_in_one_dataset_is_refused_with_the_reason() {
    let model = model_with(
        leaf("orders")
            .path("s3://lake/orders/part-0.parquet")
            .table("warehouse.orders")
            .format(semstrait_model::StorageFormat::Parquet)
            .build(),
    );

    let fs = seeded_fs(["s3://lake/orders/part-0.parquet"]).await;
    let (error, _warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect_err("a storage block naming both paths and tables must not compile");

    match error.kind {
        CompileError::StorageMixesPathsAndTables {
            ref dataset,
            ref reason,
        } => {
            assert_eq!(dataset, "orders", "the refusal names the Dataset");
            assert!(
                !reason.is_empty(),
                "the refusal states a reason rather than leaving it to be inferred"
            );
        }
        ref other => panic!("expected StorageMixesPathsAndTables, got {other:?}"),
    }
}

fn table_leaf_model() -> SemanticModel {
    model_with(leaf("orders").table("warehouse.orders").build())
}

fn orders_model_reading(columns: &[(&str, DataType)]) -> SemanticModel {
    common::orders_model(
        columns
            .iter()
            .cloned()
            .map(|(name, data_type)| common::dimension(name, data_type))
            .collect(),
        Vec::new(),
        |orders| orders,
    )
}

fn one_column_catalog(column: &str, native: &str, dialect: Option<Dialect>) -> MapCatalog {
    let catalog = common::catalog(&[(column, native)]);
    match dialect {
        Some(dialect) => catalog.with_dialect(dialect),
        None => catalog,
    }
}

async fn unmappable_refusal(
    column: &str,
    native: &str,
    dialect: Option<Dialect>,
) -> Diagnostic<CompileError> {
    let catalog = one_column_catalog(column, native, dialect);
    let fs = InMemoryFileSystem::new();
    let (diagnostic, warnings) = compile(
        orders_model_reading(&[(column, DataType::String)]),
        Some(&catalog),
        &fs,
        test_provenance(),
    )
    .await
    .expect_err(native);
    assert!(
        warnings.is_empty(),
        "{native}: the refusal is no warning too"
    );
    match &diagnostic.kind {
        CompileError::UnmappableNativeType {
            column: got_column,
            native_type,
            expr_id,
            physical_source_id,
        } => {
            assert_eq!(got_column, column, "{native}");
            assert_eq!(native_type, native, "{native}");
            assert!(
                expr_id.is_some(),
                "{native}: the reading expression is known here"
            );
            assert!(
                !physical_source_id.is_empty(),
                "{native}: the offending source is named"
            );
        }
        other => panic!("{native}: expected UnmappableNativeType, got {other:?}"),
    }
    diagnostic
}

async fn compile_model_fed() -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, _warnings) = compile(table_leaf_model(), None, &fs, test_provenance())
        .await
        .expect("the single-leaf fixture is the happy path and must compile clean");
    manifest
}

fn only_source(manifest: &SemanticManifest) -> &semstrait_manifest::PhysicalSource {
    assert_eq!(
        manifest.sources.len(),
        1,
        "the single-leaf fixture reads exactly one physical source"
    );
    manifest.sources.values().next().expect("one source")
}

#[tokio::test]
async fn a_source_with_no_catalog_is_model_fed_with_its_read_types() {
    let manifest = compile_model_fed().await;
    let source = only_source(&manifest);

    let SchemaProvenance::ModelFed { schema } = &source.schema_provenance else {
        panic!(
            "no catalog was supplied, so the schema is assumed and the arm must say so; \
             got {:?}",
            source.schema_provenance
        );
    };

    let columns: Vec<(&str, Option<&DataType>, Option<bool>)> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c.data_type.as_ref(), c.nullable))
        .collect();
    assert_eq!(
        columns,
        vec![
            ("region", Some(&DataType::String), None),
            ("revenue", Some(&DataType::Double), None)
        ],
        "the column set is the referenced one and complete; each column holds the type \
         of the semantics that maps it, and no nullability is claimed"
    );
}

#[tokio::test]
async fn a_table_source_carries_a_typed_table_identity() {
    let manifest = compile_model_fed().await;
    let source = only_source(&manifest);

    let StorageTarget::Table { table, .. } = &source.storage.target else {
        panic!(
            "a `tables:` entry is a table target; got {:?}",
            source.storage.target
        );
    };
    assert_eq!(
        table.name, "orders",
        "the trailing segment is the table name"
    );
    assert_eq!(
        table.namespace.dotted(),
        "warehouse",
        "the leading segments are the namespace path"
    );
    assert_eq!(
        table.fully_qualified(),
        "warehouse.orders",
        "the typed identity renders back to what the author wrote"
    );
}

#[tokio::test]
async fn a_source_with_a_catalog_is_catalog_derived_and_keeps_the_native_spelling() {
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![
            NativeColumn::new("region", "string", false),
            NativeColumn::new("revenue", "decimal(18,2)", true),
        ]),
    );
    let fs = InMemoryFileSystem::new();
    let (manifest, _warnings) = compile(table_leaf_model(), Some(&catalog), &fs, test_provenance())
        .await
        .expect("the catalog resolves the one table the leaf names");

    let source = only_source(&manifest);
    let SchemaProvenance::CatalogDerived { schema, native } = &source.schema_provenance else {
        panic!(
            "a catalog was supplied and resolved, so the schema is observed; got {:?}",
            source.schema_provenance
        );
    };

    let canonical: Vec<(&str, Option<&DataType>, Option<bool>)> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c.data_type.as_ref(), c.nullable))
        .collect();
    assert_eq!(
        canonical,
        vec![
            ("region", Some(&DataType::String), Some(false)),
            (
                "revenue",
                Some(&DataType::Decimal {
                    precision: 18,
                    scale: 2
                }),
                Some(true)
            ),
        ],
        "observed columns are canonical and keep the observed nullability"
    );

    assert_eq!(
        native
            .columns
            .iter()
            .map(|c| c.data_type.as_str())
            .collect::<Vec<_>>(),
        vec!["string", "decimal(18,2)"],
        "the native spelling is carried beside the canonical one, unchanged"
    );
}

#[tokio::test]
async fn a_native_type_is_read_through_the_canonical_names_and_their_aliases() {
    let roster = [
        DataType::Boolean,
        DataType::TinyInt,
        DataType::SmallInt,
        DataType::Integer,
        DataType::BigInt,
        DataType::Real,
        DataType::Double,
        DataType::Decimal {
            precision: 18,
            scale: 2,
        },
        DataType::String,
        DataType::VarChar { length: 10 },
        DataType::FixedChar { length: 5 },
        DataType::Binary,
        DataType::FixedBinary { length: 16 },
        DataType::Date,
        DataType::Time { precision: 3 },
        DataType::Timestamp { precision: 6 },
        DataType::Interval,
        DataType::IntervalYearMonth,
        DataType::IntervalDaySecond { precision: 3 },
        DataType::Uuid,
        DataType::Array(Box::new(DataType::String)),
    ];
    let spellings = [
        ("int", DataType::Integer),
        ("long", DataType::BigInt),
        ("float", DataType::Real),
        (
            "decimal(38, 9)",
            DataType::Decimal {
                precision: 38,
                scale: 9,
            },
        ),
    ];
    let natives: Vec<(String, DataType)> = roster
        .map(|data_type| (data_type.to_string(), data_type))
        .into_iter()
        .chain(spellings.map(|(native, data_type)| (native.to_owned(), data_type)))
        .collect();
    let names: Vec<String> = (0..natives.len())
        .map(|index| format!("c{index}"))
        .collect();
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(
            names
                .iter()
                .zip(&natives)
                .map(|(name, (native, _))| NativeColumn::new(name, native, true))
                .collect(),
        ),
    );
    let columns: Vec<(&str, DataType)> = names
        .iter()
        .zip(&natives)
        .map(|(name, (_, data_type))| (name.as_str(), data_type.clone()))
        .collect();
    let model = orders_model_reading(&columns);
    let manifest = common::compile_model_with(model, Some(&catalog)).await;

    let SchemaProvenance::CatalogDerived { schema, native } =
        &only_source(&manifest).schema_provenance
    else {
        panic!("a catalog answered");
    };
    let read: Vec<(String, Option<DataType>)> = native
        .columns
        .iter()
        .zip(&schema.columns)
        .map(|(native, column)| (native.data_type.clone(), column.data_type.clone()))
        .collect();
    assert_eq!(
        read,
        natives
            .into_iter()
            .map(|(native, data_type)| (native, Some(data_type)))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_native_type_that_maps_to_no_type_fails_the_compile_when_referenced() {
    for native in [
        "Long",
        "BIGINT",
        "decimal",
        "numeric",
        "timestamp",
        "datetime",
        "time",
        "interval_day_second",
        "array<decimal>",
        "unknown",
        "timestamptz",
        "fixed[16]",
        "int4",
        "TIMESTAMP_NTZ",
        "number",
        "decimal(40,2)",
        "timestamp(4)",
    ] {
        unmappable_refusal("amount", native, None).await;
    }
}

#[tokio::test]
async fn the_observed_storage_block_persists_including_partitions() {
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![
            NativeColumn::new("region", "string", true),
            NativeColumn::new("revenue", "double", true),
        ]),
    )
    .with_partitions(vec![Partition::new(
        1,
        "day",
        "occurred_at",
        "timestamp",
        PartitionTransform::Day,
    )]);
    let fs = InMemoryFileSystem::new();
    let (manifest, _warnings) = compile(table_leaf_model(), Some(&catalog), &fs, test_provenance())
        .await
        .expect("compiles against the catalog");

    let storage = &only_source(&manifest).storage;
    assert_eq!(storage.format, Some(FileFormat::Iceberg), "format persists");
    assert_eq!(storage.partitions.len(), 1, "partitions persist");
    assert_eq!(storage.partitions[0].name, "day");
    assert_eq!(
        storage.partitions[0].transform,
        PartitionTransform::Day,
        "the transform is the canonicalised partition spec, not a value"
    );
}

#[tokio::test]
async fn an_unmappable_native_type_fails_the_compile_with_its_spelling() {
    unmappable_native_type_refusal().await.assert_committed();
}

async fn unmappable_native_type_refusal() -> Golden {
    let diagnostic = unmappable_refusal("region", "variant", None).await;
    Golden::message(
        "an_unmappable_native_type_fails_the_compile_with_its_spelling",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn iceberg_spellings_type_referenced_columns_end_to_end() {
    let spellings = [
        ("c0", "time", DataType::Time { precision: 6 }),
        ("c1", "timestamp", DataType::Timestamp { precision: 6 }),
        ("c2", "timestamptz", DataType::Timestamp { precision: 6 }),
        ("c3", "timestamp_ns", DataType::Timestamp { precision: 9 }),
        ("c4", "timestamptz_ns", DataType::Timestamp { precision: 9 }),
        ("c5", "fixed[16]", DataType::FixedBinary { length: 16 }),
        ("c6", "fixed[ 16 ]", DataType::FixedBinary { length: 16 }),
        ("c7", "boolean", DataType::Boolean),
        ("c8", "int", DataType::Integer),
        ("c9", "long", DataType::BigInt),
        ("c10", "float", DataType::Real),
        ("c11", "double", DataType::Double),
        (
            "c12",
            "decimal(9,2)",
            DataType::Decimal {
                precision: 9,
                scale: 2,
            },
        ),
        ("c13", "date", DataType::Date),
        ("c14", "string", DataType::String),
        ("c15", "uuid", DataType::Uuid),
        ("c16", "binary", DataType::Binary),
    ];
    let catalog = common::catalog(
        &spellings
            .each_ref()
            .map(|(name, native, _)| (*name, *native)),
    )
    .with_dialect(Dialect::Iceberg);
    let columns: Vec<(&str, DataType)> = spellings
        .iter()
        .map(|(name, _, data_type)| (*name, data_type.clone()))
        .collect();
    let manifest = common::compile_model_with(orders_model_reading(&columns), Some(&catalog)).await;

    let SchemaProvenance::CatalogDerived { schema, .. } = &only_source(&manifest).schema_provenance
    else {
        panic!("a catalog answered");
    };
    let types: Vec<Option<&DataType>> = schema
        .columns
        .iter()
        .map(|column| column.data_type.as_ref())
        .collect();
    assert_eq!(
        types,
        spellings
            .iter()
            .map(|(_, _, data_type)| Some(data_type))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn a_typing_read_goes_through_the_dialect_too() {
    let catalog = one_column_catalog("occurred_at", "timestamptz", Some(Dialect::Iceberg));
    let declared = DataType::Timestamp { precision: 6 };
    let model = orders_model_reading(&[("occurred_at", declared)]);
    let manifest = common::compile_model_with(model, Some(&catalog)).await;
    assert_eq!(
        common::tree(
            &manifest,
            common::signature(&manifest, "orders", "occurred_at")
        ),
        common::orders_column("occurred_at"),
        "the dialect's observed `timestamp(6)` equals the declared type, so no cast is built"
    );
}

#[tokio::test]
async fn an_exotic_iceberg_column_fails_when_referenced_and_is_silent_otherwise() {
    for golden in exotic_iceberg_column_refusals().await {
        golden.assert_committed();
    }

    for native in ["variant", "struct", "list"] {
        let catalog = common::catalog(&[("payload", native), ("amount", "double")])
            .with_dialect(Dialect::Iceberg);
        let manifest = common::compile_model_with(
            orders_model_reading(&[("amount", DataType::Double)]),
            Some(&catalog),
        )
        .await;

        let SchemaProvenance::CatalogDerived { schema, .. } =
            &only_source(&manifest).schema_provenance
        else {
            panic!("{native}: a catalog answered");
        };
        let names: Vec<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["amount"],
            "{native}: the unreferenced exotic column is neither read nor reported"
        );
    }
}

async fn exotic_iceberg_column_refusals() -> Vec<Golden> {
    let mut goldens = Vec::new();
    for native in ["variant", "struct", "list"] {
        let diagnostic = unmappable_refusal("payload", native, Some(Dialect::Iceberg)).await;
        goldens.push(Golden::message(
            &format!(
                "an_exotic_iceberg_column_fails_when_referenced_and_is_silent_otherwise_{native}"
            ),
            diagnostic.to_string(),
        ));
    }
    goldens
}

#[tokio::test]
async fn a_mapped_column_reported_as_variant_gives_a_typed_compile_error() {
    mapped_variant_column_refusal().await.assert_committed();
}

async fn mapped_variant_column_refusal() -> Golden {
    let diagnostic = unmappable_refusal("created_at", "variant", Some(Dialect::Iceberg)).await;
    Golden::message(
        "a_mapped_column_reported_as_variant_gives_a_typed_compile_error",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn an_unreferenced_exotic_column_is_neither_read_nor_reported() {
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![
            NativeColumn::new("region", "string", true),
            NativeColumn::new("revenue", "double", true),
            NativeColumn::new("payload", "fixed[16]", true),
            NativeColumn::new("ingested_at", "timestamptz", true),
        ]),
    );
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(table_leaf_model(), Some(&catalog), &fs, test_provenance())
        .await
        .expect("two unreadable columns nobody references must not fail the compile");

    assert!(
        warnings.is_empty(),
        "an unreferenced exotic column produces no diagnostic in a compile that does not \
         read it: {warnings:?}"
    );

    let source = only_source(&manifest);
    let names: Vec<&str> = source
        .schema()
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["region", "revenue"],
        "the canonical schema is the referenced projection, not the whole table"
    );

    let SchemaProvenance::CatalogDerived { native, .. } = &source.schema_provenance else {
        panic!("a catalog answered");
    };
    assert_eq!(
        native.columns.len(),
        4,
        "the native schema is provenance and is not projected"
    );
}

#[tokio::test]
async fn a_referenced_column_absent_from_the_observed_schema_is_refused() {
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![NativeColumn::new("region", "string", true)]),
    );
    let fs = InMemoryFileSystem::new();
    let (error, _warnings) = compile(table_leaf_model(), Some(&catalog), &fs, test_provenance())
        .await
        .expect_err("a referenced column the provider lacks must not compile");

    match error.kind {
        CompileError::ColumnNotInSourceSchema { ref column, .. } => {
            assert_eq!(column, "revenue");
        }
        ref other => panic!("expected ColumnNotInSourceSchema, got {other:?}"),
    }
}

#[tokio::test]
async fn the_projection_preserves_relative_native_column_order() {
    let catalog = MapCatalog::new(
        warehouse_orders(),
        NativeSchema::new(vec![
            NativeColumn::new("revenue", "double", true),
            NativeColumn::new("region", "string", true),
        ]),
    );
    let fs = InMemoryFileSystem::new();
    let (manifest, _warnings) = compile(table_leaf_model(), Some(&catalog), &fs, test_provenance())
        .await
        .expect("compiles against the catalog");

    let names: Vec<&str> = only_source(&manifest)
        .schema()
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["revenue", "region"],
        "the provider's order is preserved, not normalised into name order"
    );
}

#[tokio::test]
async fn a_table_the_catalog_lacks_is_a_hard_error() {
    let catalog = NoopCatalogProvider::default();
    let fs = InMemoryFileSystem::new();
    let (error, _warnings) = compile(table_leaf_model(), Some(&catalog), &fs, test_provenance())
        .await
        .expect_err("the noop catalog reports every table missing");

    match error.kind {
        CompileError::CatalogSourceMissing { ref locator, .. } => {
            assert_eq!(
                locator, "warehouse.orders",
                "the diagnostic shows the locator the author wrote, rendered from the \
                 typed target rather than read off a persisted string"
            );
        }
        ref other => panic!("expected CatalogSourceMissing, got {other:?}"),
    }
}

fn path_leaf_model(pattern: &str) -> SemanticModel {
    model_with(
        leaf("orders")
            .path(pattern)
            .format(StorageFormat::Parquet)
            .build(),
    )
}

fn only_path_source(manifest: &SemanticManifest) -> (&str, Vec<&str>) {
    let StorageTarget::Path {
        base_path,
        listings,
        ..
    } = &only_source(manifest).storage.target
    else {
        panic!("a `paths:` entry is a path target");
    };
    (
        base_path.as_str(),
        listings.iter().map(Uri::as_str).collect(),
    )
}

#[tokio::test]
async fn a_glob_in_one_folder_is_one_physical_source_with_a_sorted_listing() {
    let fs = seeded_fs([
        "s3://lake/orders/part-2.parquet",
        "s3://lake/orders/part-0.parquet",
        "s3://lake/orders/part-1.parquet",
    ])
    .await;
    let (manifest, _warnings) = compile(
        path_leaf_model("s3://lake/orders/*.parquet"),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("three matched objects compile");

    assert_eq!(
        only_path_source(&manifest),
        (
            "s3://lake/orders/",
            vec![
                "s3://lake/orders/part-0.parquet",
                "s3://lake/orders/part-1.parquet",
                "s3://lake/orders/part-2.parquet",
            ]
        )
    );
}

#[tokio::test]
async fn a_pattern_selects_no_object_below_its_base_path() {
    let fs = seeded_fs([
        "s3://lake/orders/part-0.parquet",
        "s3://lake/orders/2024/part-9.parquet",
    ])
    .await;
    let (manifest, _warnings) = compile(
        path_leaf_model("s3://lake/orders/*.parquet"),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("the one same-level match compiles");

    assert_eq!(
        only_path_source(&manifest).1,
        vec!["s3://lake/orders/part-0.parquet"],
        "the nested object must not be swallowed by the wildcard"
    );
}

#[tokio::test]
async fn a_pattern_matching_nothing_names_the_dataset_and_the_pattern() {
    let fs = seeded_fs(["s3://lake/orders/part-0.parquet"]).await;
    let (error, _warnings) = compile(
        path_leaf_model("s3://lake/ordrs/*.parquet"),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect_err("a pattern matching nothing must not compile to an empty roster");

    match error.kind {
        CompileError::SourcePatternMatchedNothing {
            ref dataset,
            ref pattern,
        } => {
            assert_eq!(dataset, "orders", "names the Dataset that wrote it");
            assert_eq!(
                pattern, "s3://lake/ordrs/*.parquet",
                "names the pattern verbatim"
            );
        }
        ref other => panic!("expected SourcePatternMatchedNothing, got {other:?}"),
    }
}

#[tokio::test]
async fn a_transport_that_cannot_list_is_refused_as_a_listing_failure() {
    for golden in listing_failure_refusals().await {
        golden.assert_committed();
    }
}

async fn listing_failure_refusals() -> Vec<Golden> {
    #[derive(Debug)]
    struct UnreachableStore;

    #[async_trait::async_trait]
    impl FileSystem for UnreachableStore {
        async fn list(&self, _prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
            Err(FileSystemErrorKind::ConnectionFailed {
                msg: "the endpoint refused the connection".to_owned(),
            })
        }
        async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind> {
            Err(FileSystemErrorKind::NotFound { path: path.clone() })
        }
        async fn write(&self, _path: &Path, _data: Bytes) -> Result<(), FileSystemErrorKind> {
            Ok(())
        }
        async fn exists(&self, _path: &Path) -> Result<bool, FileSystemErrorKind> {
            Ok(false)
        }
    }

    let mut goldens = Vec::new();
    for (kind, written) in [
        ("glob", "s3://lake/orders/*.parquet"),
        ("folder", "s3://lake/orders/"),
        ("named_object", "s3://lake/orders/a.parquet"),
    ] {
        let (error, _warnings) = compile(
            path_leaf_model(written),
            None,
            &UnreachableStore,
            test_provenance(),
        )
        .await
        .expect_err("a transport that cannot list must not compile to an empty roster");

        let CompileError::SourceListingFailed {
            ref dataset,
            ref entry,
            ref reason,
        } = error.kind
        else {
            panic!("{kind}: expected SourceListingFailed, got {:?}", error.kind);
        };
        assert_eq!(dataset, "orders", "{kind}: names the Dataset that wrote it");
        assert_eq!(entry, written, "{kind}: names the entry verbatim");
        assert!(
            reason.contains("refused the connection"),
            "{kind}: carries the transport's own reason: {reason}"
        );
        goldens.push(Golden::message(
            &format!("a_transport_that_cannot_list_is_refused_as_a_listing_failure_{kind}"),
            error.to_string(),
        ));
    }
    goldens
}

#[tokio::test]
async fn a_literal_path_lists_only_the_object_it_names() {
    #[derive(Debug, Default)]
    struct Recording {
        objects: InMemoryFileSystem,
        listed: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl FileSystem for Recording {
        async fn list(&self, prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
            if let Ok(mut listed) = self.listed.lock() {
                listed.push(prefix.as_str().to_owned());
            }
            self.objects.list(prefix).await
        }
        async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind> {
            self.objects.read(path).await
        }
        async fn write(&self, path: &Path, data: Bytes) -> Result<(), FileSystemErrorKind> {
            self.objects.write(path, data).await
        }
        async fn exists(&self, path: &Path) -> Result<bool, FileSystemErrorKind> {
            self.objects.exists(path).await
        }
    }

    let fs = Recording::default();
    fs.write(
        &Path::new("s3://lake/orders/part-0.parquet"),
        Bytes::from_static(b"PAR1"),
    )
    .await
    .expect("seeding the in-memory filesystem");
    let (manifest, _warnings) = compile(
        path_leaf_model("s3://lake/orders/part-0.parquet"),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("a literal path that names an object compiles");

    assert_eq!(
        only_path_source(&manifest),
        ("s3://lake/orders/", vec!["s3://lake/orders/part-0.parquet"])
    );
    assert_eq!(
        *fs.listed.lock().expect("the recording"),
        vec!["s3://lake/orders/part-0.parquet"]
    );
}

#[tokio::test]
async fn two_leaves_naming_one_relation_share_one_entry() {
    let mut model = table_leaf_model();
    let twin = leaf("orders_copy").table("warehouse.orders").build();
    model.datasets.insert("orders_copy".to_owned(), twin);

    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("two leaves over one relation compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    assert_eq!(manifest.semantic_sources.len(), 2);
    assert_eq!(manifest.sources.len(), 1, "one relation is one entry");
    let shared = manifest.sources.keys().next().expect("one source").clone();
    for entry in manifest.semantic_sources.values() {
        let semstrait_manifest::RoleDefinition::Dataset { source_id, .. } = &entry.role_definition
        else {
            panic!("both leaves are `Dataset` role definitions");
        };
        assert_eq!(
            source_id, &shared,
            "the reverse edge from a source to its readers is 1:N by design"
        );
    }
    assert_eq!(
        shared,
        semstrait_manifest::mint_table_source_id("warehouse.orders"),
        "the seed is `(source_type, locator)` and nothing else"
    );
    assert!(manifest.validate().is_ok());
}

async fn two_relation_manifest() -> SemanticManifest {
    let model = model_with(
        leaf("orders")
            .table("warehouse.orders")
            .table("warehouse.returns")
            .build(),
    );
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("a Dataset over two relations compiles");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

#[tokio::test]
async fn two_physical_sources_take_ps0_and_ps1_in_entity_id_order() {
    let manifest = two_relation_manifest().await;

    let table: Vec<(&str, String, &str)> = manifest
        .sources
        .values()
        .map(|source| (&source.id[..8], source.locator(), source.alias.as_str()))
        .collect();
    assert_eq!(
        table,
        vec![
            ("5d33f8f9", "warehouse.returns".to_owned(), "ps0"),
            ("bf575591", "warehouse.orders".to_owned(), "ps1"),
        ],
        "in `EntityId` order"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn validate_refuses_a_ps_alias_its_index_does_not_give() {
    let manifest = two_relation_manifest().await;
    assert!(manifest.validate().is_ok(), "the unedited manifest passes");

    let edited = semstrait_manifest::mint_table_source_id("warehouse.orders");
    for (found, case) in [
        ("ps0", "a repeated alias"),
        ("ps.1", "an alias with a dot"),
        ("PS1", "an alias that differs only in case"),
        ("", "an empty alias"),
        ("ssd1", "an alias in a semantic source's form"),
    ] {
        let mut decoded = manifest.clone();
        if let Some(source) = decoded.sources.get_mut(&edited) {
            source.alias = SourceAlias::from(found);
        }
        let faults: Vec<CompileError> = decoded
            .validate()
            .expect_err(case)
            .into_iter()
            .map(|fault| fault.kind)
            .collect();
        assert_eq!(
            faults,
            vec![CompileError::PhysicalSourceAliasMismatch {
                physical_source: edited.clone(),
                expected: SourceAlias::from("ps1"),
                found: SourceAlias::from(found),
            }],
            "{case}"
        );
        assert_eq!(
            faults[0].to_string(),
            format!(
                "physical source `{edited}` carries the alias `{found}`; its index gives `ps1`"
            ),
            "ADR 0078's message, with no source role"
        );
    }
}

#[tokio::test]
async fn validate_refuses_a_catalog_derived_column_with_no_data_type() {
    untyped_catalog_column_refusal().await.assert_committed();
}

async fn untyped_catalog_column_refusal() -> Golden {
    let catalog = one_column_catalog("amount", "double", Some(Dialect::Iceberg));
    let manifest = common::compile_model_with(
        orders_model_reading(&[("amount", DataType::Double)]),
        Some(&catalog),
    )
    .await;

    let source_id = only_source(&manifest).id.clone();
    let mut decoded = manifest;
    let source = decoded
        .sources
        .get_mut(&source_id)
        .expect("the one physical source");
    let SchemaProvenance::CatalogDerived { schema, .. } = &mut source.schema_provenance else {
        panic!("a catalog answered");
    };
    schema.columns[0].data_type = None;

    let faults: Vec<CompileError> = decoded
        .validate()
        .expect_err("an untyped catalog column is refused")
        .into_iter()
        .map(|fault| fault.kind)
        .collect();
    assert_eq!(
        faults,
        vec![CompileError::UntypedCatalogColumn {
            physical_source_id: source_id.clone(),
            column: "amount".to_owned(),
        }]
    );
    Golden::message(
        "validate_refuses_a_catalog_derived_column_with_no_data_type",
        faults[0].to_string(),
    )
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        let mut goldens = vec![
            unmappable_native_type_refusal().await,
            mapped_variant_column_refusal().await,
            untyped_catalog_column_refusal().await,
        ];
        goldens.extend(exotic_iceberg_column_refusals().await);
        goldens.extend(listing_failure_refusals().await);
        goldens
    })
    .await;
}
