mod common;

use common::golden::{self, Golden};
use common::{
    agg, bodied, cast, column, compile_diagnostic_with_on, compile_model_on, dimension,
    identity_mapping, insert_dataset, insert_dimension, insert_measure, measure, refusal_golden_on,
    refusal_golden_with_on, seeded_fs, signature, source_alias, tree, untyped, warehouse_orders,
    MapCatalog,
};

use bytes::Bytes;
use semstrait_catalog::{FileFormat, Schema as NativeSchema};
use semstrait_common::io::{FileEntry, FileSystem, FileSystemErrorKind, InMemoryFileSystem, Path};
use semstrait_manifest::{
    mint_path_source_id, mint_reduction_leaf_id, mint_semantic_source_id, CompileError,
    HivePartition, PhysicalSource, ReferenceSite, RoleDefinition, Selector, SemanticManifest,
    StorageTarget, Uri,
};
use semstrait_model::{
    Dataset, DatasetBuilder, Dimension, DimensionSite, Measure, MeasureSite, NestedDataset,
    PartitionDef, SemanticMapping, SemanticModel, StorageFormat, Unionset,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::Literal;
use serde_json::{json, Value};

fn leaf(name: &str) -> DatasetBuilder {
    Dataset::builder(name)
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .format(StorageFormat::Parquet)
}

fn model_of(datasets: impl IntoIterator<Item = DatasetBuilder>) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(&mut model, dimension("region", DataType::String));
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Double)
        .agg(agg("sum"))
        .build();
    let id = revenue.entity_id().expect("the mint is eager").to_owned();
    model.measures.insert(id, revenue);
    for dataset in datasets {
        insert_dataset(&mut model, dataset.build());
    }
    model
}

fn stored(manifest: &SemanticManifest) -> Value {
    serde_json::to_value(manifest).expect("a manifest serializes")
}

fn role_definition(json: &Value, name: &str) -> Value {
    let sources = json["semantic_sources"].as_object().expect("an object");
    let source = (sources.values())
        .find(|source| source["name"] == name)
        .unwrap_or_else(|| panic!("no semantic source named `{name}`"));
    source["role_definition"].clone()
}

fn path_source_id(base_path: &str) -> String {
    mint_path_source_id(base_path, Some(FileFormat::Parquet))
}

async fn compiled_twice(model: impl Fn() -> SemanticModel, fs: &dyn FileSystem) -> Value {
    let first = stored(&compile_model_on(model(), fs).await);
    assert_eq!(first, stored(&compile_model_on(model(), fs).await));
    first
}

#[tokio::test]
async fn uris_in_either_order_read_one_physical_source_through_one_sorted_set() {
    let (a, b) = ("s3://lake/orders/a.parquet", "s3://lake/orders/b.parquet");
    let fs = seeded_fs([a, b]).await;
    let json = compiled_twice(
        || {
            model_of([
                leaf("forward").paths([a, b]),
                leaf("backward").paths([b, a]),
            ])
        },
        &fs,
    )
    .await;

    let id = path_source_id("s3://lake/orders/");
    let sources = json["sources"].as_object().expect("an object");
    assert_eq!(sources.keys().collect::<Vec<_>>(), [&id]);
    assert_eq!(sources[&id]["name"], "orders");
    assert_eq!(
        sources[&id]["storage"]["target"],
        json!({
            "type": "path",
            "content": {"base_path": "s3://lake/orders/", "listings": [a, b], "partition_def": null},
        })
    );
    for dataset in ["forward", "backward"] {
        assert_eq!(
            role_definition(&json, dataset),
            json!({
                "type": "dataset",
                "content": {"source_id": id, "selector": {"type": "uris", "content": [a, b]}},
            }),
            "{dataset}"
        );
    }
}

fn named<'j>(json: &'j Value, name: &str) -> Vec<&'j Value> {
    let sources = json["semantic_sources"].as_object().expect("an object");
    (sources.values())
        .filter(|source| source["name"] == name)
        .collect()
}

async fn sales_by_region() -> Value {
    let fs = seeded_fs([
        "s3://lake/us/orders/a.parquet",
        "s3://lake/eu/orders/a.parquet",
        "s3://lake/eu/returns/a.parquet",
    ])
    .await;
    compiled_twice(
        || model_of([leaf("sales").path("s3://lake/\x2a/orders/\x2a.parquet")]),
        &fs,
    )
    .await
}

#[tokio::test]
async fn a_folder_wildcard_gives_one_physical_source_for_each_folder_it_matches() {
    let json = sales_by_region().await;

    let sources = json["sources"].as_object().expect("an object");
    let mut expected =
        ["eu", "us"].map(|region| path_source_id(&format!("s3://lake/{region}/orders/")));
    expected.sort();
    assert_eq!(
        sources.keys().collect::<Vec<_>>(),
        expected.iter().collect::<Vec<_>>()
    );
    for region in ["eu", "us"] {
        let source = &sources[&path_source_id(&format!("s3://lake/{region}/orders/"))];
        assert_eq!(source["name"], "orders");
        assert_eq!(
            source["storage"]["target"]["content"]["listings"],
            json!([format!("s3://lake/{region}/orders/a.parquet")])
        );
    }
    let mut selectors: Vec<Value> = (named(&json, "sales_orders").into_iter())
        .map(|leaf| leaf["role_definition"]["content"]["selector"].clone())
        .collect();
    selectors.sort_by_key(Value::to_string);
    assert_eq!(
        selectors,
        ["eu", "us"].map(|region| {
            json!({"type": "pattern", "content": format!("s3://lake/{region}/orders/\x2a.parquet")})
        })
    );
    assert_eq!(role_definition(&json, "sales")["type"], "unionset");
}

#[tokio::test]
async fn each_reduction_leaf_is_minted_from_its_dataset_and_its_physical_source() {
    let json = sales_by_region().await;

    let sales = mint_semantic_source_id("sales");
    let mut expected: Vec<String> = ["eu", "us"]
        .map(|region| {
            let source = path_source_id(&format!("s3://lake/{region}/orders/"));
            mint_reduction_leaf_id(&sales, &source)
        })
        .into();
    expected.sort();
    let mut leaves: Vec<&str> = (named(&json, "sales_orders").into_iter())
        .map(|leaf| leaf["id"].as_str().expect("an id"))
        .collect();
    leaves.sort_unstable();
    assert_eq!(leaves, expected);
    assert_eq!(
        role_definition(&json, "sales")["content"]["branches"],
        json!(expected),
        "the Implicit Unionset's branches are its two leaves"
    );
}

fn column_names(source: &Value) -> Vec<&Value> {
    (source["schema_provenance"]["content"]["schema"]["columns"]
        .as_array()
        .expect("an array")
        .iter())
    .map(|column| &column["name"])
    .collect()
}

#[tokio::test]
async fn three_selectors_over_one_folder_fold_into_one_physical_source() {
    let objects = ["s3://lake/orders/a.parquet", "s3://lake/orders/b.parquet"];
    let fs = seeded_fs(objects).await;
    let model = || {
        let mut model = model_of([
            Dataset::builder("x")
                .dimension(DimensionSite::r#ref("region"))
                .format(StorageFormat::Parquet)
                .path("s3://lake/orders/"),
            Dataset::builder("y")
                .measure(MeasureSite::r#ref("revenue"))
                .format(StorageFormat::Parquet)
                .path("s3://lake/orders/\x2a.parquet"),
            Dataset::builder("z")
                .dimension(DimensionSite::r#ref("channel"))
                .format(StorageFormat::Parquet)
                .path("s3://lake/orders/a.parquet"),
        ]);
        insert_dimension(&mut model, dimension("channel", DataType::String));
        model
    };
    let json = compiled_twice(model, &fs).await;

    let id = path_source_id("s3://lake/orders/");
    let sources = json["sources"].as_object().expect("an object");
    assert_eq!(sources.keys().collect::<Vec<_>>(), [&id]);
    let source = &sources[&id];
    assert_eq!(
        source["storage"]["target"]["content"]["listings"],
        json!(objects)
    );
    assert_eq!(column_names(source), ["channel", "region", "revenue"]);
    for (dataset, selector) in [
        ("x", json!({"type": "prefix"})),
        (
            "y",
            json!({"type": "pattern", "content": "s3://lake/orders/\x2a.parquet"}),
        ),
        (
            "z",
            json!({"type": "uris", "content": ["s3://lake/orders/a.parquet"]}),
        ),
    ] {
        assert_eq!(
            role_definition(&json, dataset)["content"],
            json!({"source_id": id, "selector": selector}),
            "{dataset}"
        );
    }
}

type Edit = fn(&mut PhysicalSource, &mut Selector);

#[tokio::test]
async fn validate_refuses_each_catalog_stage_rule_the_stored_manifest_decides() {
    let (a, b) = ("s3://lake/orders/a.parquet", "s3://lake/orders/b.parquet");
    let fs = seeded_fs([a, b]).await;
    let manifest = compile_model_on(model_of([leaf("orders").paths([a, b])]), &fs).await;
    let source_id = path_source_id("s3://lake/orders/");
    let dataset_id = mint_semantic_source_id("orders");
    let base_path_rule = "every object a path source lists lies beneath its base path, and a \
                          Pattern selector's glob names objects directly under it";

    let cases: [(&str, Edit, ReferenceSite, &str); 8] = [
        (
            "a selector paired with the wrong target",
            |_, selector| *selector = Selector::Table,
            ReferenceSite::SemanticSource(dataset_id.clone()),
            "a Table selector reads a table source, and a Prefix, Pattern or Uris selector a \
             path source",
        ),
        (
            "a list out of its fixed order",
            |source, _| {
                if let StorageTarget::Path { listings, .. } = &mut source.storage.target {
                    listings.reverse();
                }
            },
            ReferenceSite::Source(source_id.clone()),
            "a listing and a Uris selector are sets sorted by URI",
        ),
        (
            "a URI not in normal form",
            |source, selector| {
                if let StorageTarget::Path {
                    base_path,
                    listings,
                    ..
                } = &mut source.storage.target
                {
                    *base_path = Uri::from("s3a://lake/orders/");
                    *listings = ["s3a://lake/orders/a.parquet", "s3a://lake/orders/b.parquet"]
                        .map(Uri::from)
                        .into();
                }
                *selector = Selector::Prefix;
            },
            ReferenceSite::Source(source_id.clone()),
            "every URI a path source or a selector stores is in normal form, and a base path \
             names a folder",
        ),
        (
            "a path source not read as Parquet",
            |source, _| source.storage.format = Some(FileFormat::Csv),
            ReferenceSite::Source(source_id.clone()),
            "a path source is read as Parquet, and a table source is an Iceberg table",
        ),
        (
            "a URI that names no listed object",
            |_, selector| {
                *selector = Selector::Uris(vec![
                    Uri::from("s3://lake/orders/a.parquet"),
                    Uri::from("s3://lake/orders/c.parquet"),
                ]);
            },
            ReferenceSite::SemanticSource(dataset_id.clone()),
            "each URI of a Uris selector is an object of its physical source's listing",
        ),
        (
            "an object listed outside its base path",
            |source, _| {
                if let StorageTarget::Path { listings, .. } = &mut source.storage.target {
                    listings.push(Uri::from("s3://lake/returns/a.parquet"));
                }
            },
            ReferenceSite::Source(source_id.clone()),
            base_path_rule,
        ),
        (
            "a Pattern over another folder",
            |_, selector| {
                *selector = Selector::Pattern(Uri::from("s3://lake/returns/\x2a.parquet"))
            },
            ReferenceSite::SemanticSource(dataset_id.clone()),
            base_path_rule,
        ),
        (
            "a Pattern below its base path",
            |_, selector| {
                *selector = Selector::Pattern(Uri::from("s3://lake/orders/2024/\x2a.parquet"));
            },
            ReferenceSite::SemanticSource(dataset_id.clone()),
            base_path_rule,
        ),
    ];
    for (case, edit, site, rule) in cases {
        let faults = faults_after(&manifest, case, |decoded| {
            let source = decoded
                .sources
                .get_mut(&source_id)
                .expect("the path source");
            let dataset = decoded
                .semantic_sources
                .get_mut(&dataset_id)
                .expect("the Dataset");
            let RoleDefinition::Dataset { selector, .. } = &mut dataset.role_definition else {
                panic!("`orders` is a Dataset");
            };
            edit(source, selector);
        });
        assert_eq!(
            faults,
            [CompileError::StructuralViolation { site, rule }],
            "{case}"
        );
    }
}

fn faults_after(
    manifest: &SemanticManifest,
    case: &str,
    edit: impl FnOnce(&mut SemanticManifest),
) -> Vec<CompileError> {
    let mut decoded = manifest.clone();
    edit(&mut decoded);
    (decoded.validate().expect_err(case).into_iter())
        .map(|fault| fault.kind)
        .collect()
}

#[tokio::test]
async fn a_prefix_over_a_folder_with_nothing_to_read_compiles_with_no_objects() {
    let elsewhere = "s3://lake/returns/a.parquet";
    let empty = seeded_fs([elsewhere]).await;
    let skipped = seeded_fs([
        elsewhere,
        "s3://lake/orders/_SUCCESS",
        "s3://lake/orders/.part-0.crc",
        "s3://lake/orders/_temporary/0/a.parquet",
        "s3://lake/orders/a.parquet._COPYING_",
    ])
    .await;
    skipped
        .write(&Path::new("s3://lake/orders/b.parquet"), Bytes::new())
        .await
        .expect("seeding the in-memory filesystem");

    let id = path_source_id("s3://lake/orders/");
    for fs in [&empty, &skipped] {
        let json =
            compiled_twice(|| model_of([leaf("orders").path("s3://lake/orders/")]), fs).await;
        assert_eq!(
            json["sources"][&id]["storage"]["target"],
            json!({
                "type": "path",
                "content": {"base_path": "s3://lake/orders/", "listings": [], "partition_def": null},
            })
        );
    }
}

fn order_years(dataset: impl FnOnce(DatasetBuilder) -> DatasetBuilder) -> SemanticModel {
    exposing(dimension("order_year", DataType::Integer), dataset)
}

fn exposing(
    order_year: Dimension,
    dataset: impl FnOnce(DatasetBuilder) -> DatasetBuilder,
) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(&mut model, order_year);
    let orders = Dataset::builder("orders")
        .dimension(DimensionSite::r#ref("order_year"))
        .format(StorageFormat::Parquet);
    insert_dataset(&mut model, dataset(orders).build());
    model
}

fn partitioned_over<'p>(
    paths: impl IntoIterator<Item = &'p str>,
    entries: impl IntoIterator<Item = PartitionDef>,
) -> SemanticModel {
    let mapping = SemanticMapping::builder()
        .column("order_year", "year")
        .build();
    order_years(|orders| {
        (entries.into_iter()).fold(
            orders.paths(paths).semantic_mapping(mapping),
            DatasetBuilder::partition_def,
        )
    })
}

fn partitioned(entries: impl IntoIterator<Item = PartitionDef>) -> SemanticModel {
    partitioned_over(["s3://lake/orders/"], entries)
}

fn range(dimension: &str) -> PartitionDef {
    PartitionDef::Range {
        dimension: dimension.to_owned(),
    }
}

fn list(dimension: &str, values: &[&str]) -> PartitionDef {
    PartitionDef::List {
        dimension: dimension.to_owned(),
        values: values.iter().map(|value| (*value).to_owned()).collect(),
    }
}

async fn year_month_folders(key: &str) -> InMemoryFileSystem {
    seeded_fs([
        format!("s3://lake/orders/{key}=2026/month=02/part-0.parquet"),
        format!("s3://lake/orders/{key}=2026/month=01/part-1.parquet"),
        format!("s3://lake/orders/{key}=2025/month=12/part-0.parquet"),
        format!("s3://lake/orders/{key}=2026/month=01/part-0.parquet"),
    ])
    .await
}

fn year_month_partitions(key: &str) -> Value {
    let partition = |year: &str, month: &str, parts: &[&str]| {
        let listing: Vec<String> = (parts.iter())
            .map(|part| format!("s3://lake/orders/{key}={year}/month={month}/{part}.parquet"))
            .collect();
        json!({
            "key_values": [{"key": key, "value": year}, {"key": "month", "value": month}],
            "listing": listing,
        })
    };
    json!({"partitions": [
        partition("2025", "12", &["part-0"]),
        partition("2026", "01", &["part-0", "part-1"]),
        partition("2026", "02", &["part-0"]),
    ]})
}

async fn reads_the_partition_column(entries: &[PartitionDef], fs: &dyn FileSystem) -> Value {
    let model = || partitioned(entries.to_vec());
    let json = compiled_twice(model, fs).await;
    let manifest = compile_model_on(model(), fs).await;
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "order_year")),
        cast(
            column(&source_alias(&manifest, "orders"), "year"),
            DataType::Integer
        )
    );
    let source = &json["sources"][path_source_id("s3://lake/orders/")];
    assert_eq!(column_names(source), ["year"]);
    source["storage"]["target"].clone()
}

#[tokio::test]
async fn a_dimension_named_by_an_entry_reads_its_partition_column() {
    let entries = [range("order_year")];
    let target = reads_the_partition_column(&entries, &year_month_folders("year").await).await;
    assert_eq!(
        target["content"]["partition_def"],
        year_month_partitions("year")
    );
}

#[tokio::test]
async fn a_dimension_that_maps_a_discovered_key_reads_its_partition_column_with_no_entry() {
    let target = reads_the_partition_column(&[], &year_month_folders("year").await).await;
    assert_eq!(
        target["content"]["partition_def"],
        year_month_partitions("year")
    );
}

#[tokio::test]
async fn a_mapping_matches_a_key_without_regard_to_ascii_case() {
    let target = reads_the_partition_column(&[], &year_month_folders("Year").await).await;
    assert_eq!(
        target["content"]["partition_def"],
        year_month_partitions("Year")
    );
}

#[tokio::test]
async fn partitions_sort_by_their_folder_as_found() {
    let (month, year) = (
        "s3://lake/orders/date=2026-01/a.parquet",
        "s3://lake/orders/date=2026/a.parquet",
    );
    let fs = seeded_fs([year, month]).await;
    let json = compiled_twice(|| model_of([leaf("orders").path("s3://lake/orders/")]), &fs).await;
    let target = &json["sources"][path_source_id("s3://lake/orders/")]["storage"]["target"];
    assert_eq!(
        target["content"]["partition_def"],
        json!({"partitions": [
            {"key_values": [{"key": "date", "value": "2026-01"}], "listing": [month]},
            {"key_values": [{"key": "date", "value": "2026"}], "listing": [year]},
        ]})
    );
}

fn partitions_of(source: &mut PhysicalSource) -> &mut Vec<HivePartition> {
    let StorageTarget::Path {
        partition_def: Some(found),
        ..
    } = &mut source.storage.target
    else {
        panic!("a partitioned path source");
    };
    &mut found.partitions
}

fn respell(source: &mut PhysicalSource, from: &str, to: &str) {
    let respelled = |uri: &mut Uri| *uri = Uri::from(uri.as_str().replace(from, to));
    for partition in partitions_of(source) {
        partition.listing.iter_mut().for_each(respelled);
    }
    if let StorageTarget::Path { listings, .. } = &mut source.storage.target {
        listings.iter_mut().for_each(respelled);
    }
}

type PartitionEdit = fn(&mut PhysicalSource);

#[tokio::test]
async fn validate_refuses_each_partition_rule_the_stored_manifest_decides() {
    let fs = year_month_folders("year").await;
    let manifest = compile_model_on(partitioned([]), &fs).await;
    let source_id = path_source_id("s3://lake/orders/");
    let listing_rule = "a partition's listing is a set sorted by URI of objects of its path \
                        source's listing, each in the partition folder";
    let key_values_rule = "a partition's key_values are its folder's keys and values, in \
                           folder order, each key once and neither side empty";
    let direct_objects_rule = "objects directly under a base path are in no partition, and a \
                               stored partition_def holds at least one";
    let cases: [(&str, PartitionEdit, &str); 10] = [
        (
            "partitions out of folder order",
            |source| partitions_of(source).reverse(),
            "partitions are sorted by partition folder, the parent folder of the objects of each",
        ),
        (
            "a partition's listing out of its order",
            |source| partitions_of(source)[1].listing.reverse(),
            listing_rule,
        ),
        (
            "an object outside its partition folder",
            |source| {
                (partitions_of(source)[0].listing).push(Uri::from(
                    "s3://lake/orders/year=2026/month=01/part-1.parquet",
                ));
            },
            listing_rule,
        ),
        (
            "an object that its path source does not list",
            |source| {
                (partitions_of(source)[2].listing).push(Uri::from(
                    "s3://lake/orders/year=2026/month=02/part-9.parquet",
                ));
            },
            listing_rule,
        ),
        (
            "key_values out of folder order",
            |source| {
                for partition in partitions_of(source) {
                    partition.key_values.reverse();
                }
            },
            key_values_rule,
        ),
        (
            "a partition folder with an empty value",
            |source| {
                respell(source, "year=2025/", "year=/");
                partitions_of(source)[0].key_values[0].value = Some(String::new());
            },
            key_values_rule,
        ),
        (
            "one key twice in one folder path",
            |source| {
                respell(source, "/month=", "/YEAR=");
                for partition in partitions_of(source) {
                    partition.key_values[1].key = "YEAR".to_owned();
                }
            },
            key_values_rule,
        ),
        (
            "a partition directly under the base path",
            |source| {
                respell(source, "year=2025/month=12/", "");
                let partitions = partitions_of(source);
                partitions.truncate(1);
                partitions[0].key_values.clear();
            },
            direct_objects_rule,
        ),
        (
            "a stored partition_def with no partition",
            |source| partitions_of(source).clear(),
            direct_objects_rule,
        ),
        (
            "two key sequences in one source",
            |source| {
                respell(source, "year=2026/month=02/", "year=2027/");
                let last = partitions_of(source).last_mut().expect("three partitions");
                last.key_values.truncate(1);
                last.key_values[0].value = Some("2027".to_owned());
            },
            "every partition of a path source has one key sequence",
        ),
    ];
    for (case, edit, rule) in cases {
        let faults = faults_after(&manifest, case, |decoded| {
            edit(
                decoded
                    .sources
                    .get_mut(&source_id)
                    .expect("the path source"),
            );
        });
        assert_eq!(
            faults,
            [CompileError::StructuralViolation {
                site: ReferenceSite::Source(source_id.clone()),
                rule
            }],
            "{case}"
        );
    }
}

#[tokio::test]
async fn a_partition_folder_is_split_at_the_first_equals_and_decoded() {
    let (sao_paulo, default) = (
        "s3://lake/orders/sale%20city=S%C3%A3o%20Paulo=SP/a.parquet",
        "s3://lake/orders/sale%20city=__HIVE_DEFAULT_PARTITION__/a.parquet",
    );
    let fs = seeded_fs([default, sao_paulo]).await;
    let json = compiled_twice(|| model_of([leaf("orders").path("s3://lake/orders/")]), &fs).await;
    assert_eq!(
        json["sources"][path_source_id("s3://lake/orders/")]["storage"]["target"]["content"]
            ["partition_def"],
        json!({"partitions": [
            {"key_values": [{"key": "sale city", "value": "São Paulo=SP"}], "listing": [sao_paulo]},
            {"key_values": [{"key": "sale city", "value": null}], "listing": [default]},
        ]})
    );
}

#[derive(Debug)]
struct Unlistable;

#[async_trait::async_trait]
impl FileSystem for Unlistable {
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

async fn a_uri_written_twice_in_one_list() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/a.parquet"]).await;
    let model = model_of([
        leaf("orders").paths(["s3://lake/orders/a.parquet", "s3a://lake/orders/a.parquet"])
    ]);
    let expected = CompileError::DuplicateStorageUri {
        dataset: "orders".to_owned(),
        uri: "s3://lake/orders/a.parquet".to_owned(),
    };
    refusal_golden_on(
        "a_uri_written_twice_in_one_list_is_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_uri_written_twice_in_one_list_is_refused() {
    a_uri_written_twice_in_one_list().await.assert_committed();
}

async fn a_relative_path() -> Golden {
    let model = model_of([leaf("orders").paths(["s3://lake/orders/", "orders/a.parquet"])]);
    let expected = CompileError::RelativeStoragePath {
        dataset: "orders".to_owned(),
        entry: "orders/a.parquet".to_owned(),
    };
    refusal_golden_on(
        "a_relative_path_is_refused_before_anything_is_listed",
        model,
        &Unlistable,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_relative_path_is_refused_before_anything_is_listed() {
    a_relative_path().await.assert_committed();
}

async fn a_scheme_compile_does_not_read() -> Golden {
    let model = model_of([leaf("orders").path("gs://lake/orders/")]);
    let expected = CompileError::UnsupportedStorageScheme {
        dataset: "orders".to_owned(),
        entry: "gs://lake/orders/".to_owned(),
        scheme: "gs".to_owned(),
    };
    refusal_golden_on(
        "a_scheme_compile_does_not_read_is_refused",
        model,
        &Unlistable,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_scheme_compile_does_not_read_is_refused() {
    a_scheme_compile_does_not_read().await.assert_committed();
}

async fn a_folder_wildcard_over_an_empty_segment() -> Golden {
    let fs = seeded_fs(["s3://lake/orders//a.parquet"]).await;
    let model = model_of([leaf("orders").path("s3://lake/orders/\x2a/\x2a.parquet")]);
    let expected = CompileError::BasePathNotNormal {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/\x2a/\x2a.parquet".to_owned(),
        base_path: "s3://lake/orders//".to_owned(),
    };
    refusal_golden_on(
        "a_folder_wildcard_over_an_empty_segment_is_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_folder_wildcard_over_an_empty_segment_is_refused() {
    a_folder_wildcard_over_an_empty_segment()
        .await
        .assert_committed();
}

async fn a_uri_that_names_no_object() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/a.parquet"]).await;
    let model = model_of([leaf("orders").path("s3a://lake/orders/b.parquet")]);
    let expected = CompileError::StorageObjectMissing {
        dataset: "orders".to_owned(),
        entry: "s3a://lake/orders/b.parquet".to_owned(),
        uri: "s3://lake/orders/b.parquet".to_owned(),
    };
    refusal_golden_on(
        "a_uri_that_names_no_object_is_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_uri_that_names_no_object_is_refused() {
    a_uri_that_names_no_object().await.assert_committed();
}

async fn a_uri_that_names_an_empty_object() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/a.parquet"]).await;
    fs.write(&Path::new("s3://lake/orders/empty.parquet"), Bytes::new())
        .await
        .expect("seeding the in-memory filesystem");
    let model = model_of([leaf("orders").path("s3://lake/orders/empty.parquet")]);
    let expected = CompileError::StorageObjectEmpty {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/empty.parquet".to_owned(),
        uri: "s3://lake/orders/empty.parquet".to_owned(),
    };
    refusal_golden_on(
        "a_uri_that_names_an_empty_object_is_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_uri_that_names_an_empty_object_is_refused() {
    a_uri_that_names_an_empty_object().await.assert_committed();
}

async fn a_path_source_not_read_as_parquet() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/a.csv"]).await;
    let model = model_of([leaf("orders")
        .path("s3://lake/orders/")
        .format(StorageFormat::Csv)]);
    let expected = CompileError::UnsupportedSourceFormat {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        format: Some(FileFormat::Csv),
    };
    refusal_golden_on(
        "a_path_source_not_read_as_parquet_is_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_path_source_not_read_as_parquet_is_refused() {
    a_path_source_not_read_as_parquet().await.assert_committed();
}

async fn a_table_source_that_is_not_iceberg() -> Golden {
    let catalog = MapCatalog::new(warehouse_orders(), NativeSchema::new(Vec::new()))
        .with_format(FileFormat::Delta);
    let model = model_of([Dataset::builder("orders")
        .dimension(DimensionSite::r#ref("region"))
        .table("warehouse.orders")]);
    refusal_golden_with_on(
        "a_table_source_that_is_not_iceberg_is_refused",
        model,
        Some(&catalog),
        &Unlistable,
        CompileError::UnsupportedSourceFormat {
            dataset: "orders".to_owned(),
            entry: "warehouse.orders".to_owned(),
            format: Some(FileFormat::Delta),
        },
    )
    .await
}

#[tokio::test]
async fn a_table_source_that_is_not_iceberg_is_refused() {
    a_table_source_that_is_not_iceberg()
        .await
        .assert_committed();
}

#[tokio::test]
async fn a_table_source_with_an_authored_format_and_no_catalog_compiles() {
    let model = model_of([leaf("orders").table("warehouse.orders")]);
    let json = stored(&compile_model_on(model, &Unlistable).await);
    let sources = json["sources"].as_object().expect("an object");
    let formats: Vec<&Value> = (sources.values())
        .map(|source| &source["storage"]["format"])
        .collect();
    assert_eq!(formats, [&Value::Null]);
}

async fn a_glob_in_a_table_entry() -> Golden {
    let model = model_of([Dataset::builder("orders")
        .dimension(DimensionSite::r#ref("region"))
        .table("warehouse.orders_*")]);
    let expected = CompileError::TableGlob {
        dataset: "orders".to_owned(),
        entry: "warehouse.orders_*".to_owned(),
    };
    refusal_golden_on(
        "a_glob_in_a_table_entry_is_refused",
        model,
        &Unlistable,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_glob_in_a_table_entry_is_refused() {
    a_glob_in_a_table_entry().await.assert_committed();
}

async fn two_entries_on_one_base_path() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/a.parquet", "s3://lake/orders/b.parquet"]).await;
    let model = model_of([
        leaf("orders").paths(["s3://lake/orders/a*.parquet", "s3://lake/orders/b*.parquet"])
    ]);
    let expected = CompileError::StorageReadTwice {
        dataset: "orders".to_owned(),
        first: "s3://lake/orders/a*.parquet".to_owned(),
        second: "s3://lake/orders/b*.parquet".to_owned(),
    };
    refusal_golden_on(
        "two_entries_on_one_base_path_are_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn two_entries_on_one_base_path_are_refused() {
    two_entries_on_one_base_path().await.assert_committed();
}

async fn two_entries_whose_listings_overlap() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/year=2026/a.parquet"]).await;
    let model =
        model_of([leaf("orders").paths(["s3://lake/orders/", "s3://lake/orders/year=2026/"])]);
    let expected = CompileError::StorageReadTwice {
        dataset: "orders".to_owned(),
        first: "s3://lake/orders/".to_owned(),
        second: "s3://lake/orders/year=2026/".to_owned(),
    };
    refusal_golden_on(
        "two_entries_whose_listings_overlap_are_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn two_entries_whose_listings_overlap_are_refused() {
    two_entries_whose_listings_overlap()
        .await
        .assert_committed();
}

const READ_ONCE_RULE: &str =
    "the sources of one Dataset have distinct base paths, and no object is selected by two of \
     them";

fn read_twice(name: &str) -> CompileError {
    CompileError::StructuralViolation {
        site: ReferenceSite::SemanticSource(mint_semantic_source_id(name)),
        rule: READ_ONCE_RULE,
    }
}

fn relist(manifest: &mut SemanticManifest, source: &str, base_path: &str, listing: &[&str]) {
    let source = (manifest.sources.get_mut(&path_source_id(source))).expect("the path source");
    let StorageTarget::Path {
        base_path: stored,
        listings,
        ..
    } = &mut source.storage.target
    else {
        panic!("a path source");
    };
    *stored = Uri::from(base_path);
    *listings = listing.iter().copied().map(Uri::from).collect();
}

async fn returns_relisted(case: &str, base_path: &str, listing: &[&str]) -> Vec<CompileError> {
    let objects = [
        "s3://lake/orders/year=2026/a.parquet",
        "s3://lake/returns/r.parquet",
    ];
    let model =
        model_of([leaf("orders").paths(["s3://lake/orders/year=2026/", "s3://lake/returns/"])]);
    let manifest = compile_model_on(model, &seeded_fs(objects).await).await;
    faults_after(&manifest, case, |decoded| {
        relist(decoded, "s3://lake/returns/", base_path, listing);
    })
}

async fn a_dataset_that_reads_one_object_twice() -> Golden {
    let case = "a_dataset_that_reads_one_object_twice_is_refused";
    let faults = returns_relisted(
        case,
        "s3://lake/orders/",
        &["s3://lake/orders/year=2026/a.parquet"],
    )
    .await;
    assert_eq!(faults, [read_twice("orders")]);
    Golden::message(case, faults[0].to_string())
}

#[tokio::test]
async fn validate_refuses_a_dataset_whose_sources_read_one_object_twice() {
    a_dataset_that_reads_one_object_twice()
        .await
        .assert_committed();
    let faults = returns_relisted(
        "two sources on one base path",
        "s3://lake/orders/year=2026/",
        &[],
    )
    .await;
    assert_eq!(faults, [read_twice("orders")]);
}

fn all_orders(nested: &[(&str, &[&str])]) -> SemanticModel {
    let mut all_orders = (Unionset::builder("all_orders"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all();
    for (name, paths) in nested {
        all_orders = all_orders.dataset(
            NestedDataset::builder(*name)
                .paths(paths.iter().copied())
                .format(StorageFormat::Parquet)
                .semantic_mapping(identity_mapping(&["region", "revenue"]))
                .build(),
        );
    }
    let all_orders = all_orders.build();
    let mut model = model_of([]);
    let id = all_orders.body.base.id.clone().expect("the mint is eager");
    model.unionsets.insert(id, all_orders);
    model
}

#[tokio::test]
async fn validate_refuses_a_nested_dataset_whose_sources_read_one_object_twice() {
    let objects = [
        "s3://lake/eu/2025/a.parquet",
        "s3://lake/eu/2026/b.parquet",
        "s3://lake/us/c.parquet",
    ];
    let model = all_orders(&[
        ("eu", &["s3://lake/eu/2025/", "s3://lake/eu/2026/"]),
        ("us", &["s3://lake/us/"]),
    ]);
    let manifest = compile_model_on(model, &seeded_fs(objects).await).await;
    let cases: [(&str, &str, &[&str]); 2] = [
        (
            "two sources that select one object",
            "s3://lake/eu/",
            &["s3://lake/eu/2025/a.parquet"],
        ),
        ("two sources on one base path", "s3://lake/eu/2025/", &[]),
    ];
    for (case, base_path, listing) in cases {
        let faults = faults_after(&manifest, case, |decoded| {
            relist(decoded, "s3://lake/eu/2026/", base_path, listing);
        });
        assert_eq!(faults, [read_twice("all_orders.eu")], "{case}");
    }
}

#[tokio::test]
async fn a_dataset_whose_sources_read_no_object_twice_validates() {
    let cases: [(&str, &[&str], SemanticModel); 5] = [
        (
            "`orders2/b` is not beneath `orders/`: a base path ends in `/`",
            &["s3://lake/orders/a.parquet", "s3://lake/orders2/b.parquet"],
            model_of([leaf("orders").paths(["s3://lake/orders/", "s3://lake/orders2/"])]),
        ),
        (
            "nested base paths, and no object twice",
            &["s3://lake/orders/year=2026/a.parquet"],
            model_of([leaf("orders").paths(["s3://lake/orders/", "s3://lake/orders/year=2027/"])]),
        ),
        (
            "`orders/` skips `_tmp` as debris",
            &[
                "s3://lake/orders/a.parquet",
                "s3://lake/orders/_tmp/x.parquet",
            ],
            model_of([leaf("orders").paths(["s3://lake/orders/", "s3://lake/orders/_tmp/"])]),
        ),
        (
            "one object for each folder that a folder wildcard matches",
            &[
                "s3://lake/eu/orders/a.parquet",
                "s3://lake/us/orders/b.parquet",
            ],
            model_of([leaf("orders").path("s3://lake/\x2a/orders/\x2a.parquet")]),
        ),
        (
            "the listing of `orders/` also holds `year=2026/a.parquet`, for `e`",
            &[
                "s3://lake/orders/_x.parquet",
                "s3://lake/orders/year=2026/a.parquet",
            ],
            model_of([
                leaf("orders")
                    .paths(["s3://lake/orders/_x.parquet", "s3://lake/orders/year=2026/"]),
                leaf("e").path("s3://lake/orders/"),
            ]),
        ),
    ];
    for (case, objects, model) in cases {
        let fs = seeded_fs(objects.iter().copied()).await;
        let json = stored(&compile_model_on(model, &fs).await);
        assert_eq!(
            role_definition(&json, "orders")["type"],
            "unionset",
            "{case}"
        );
    }
}

#[tokio::test]
async fn two_datasets_may_read_one_object() {
    let object = "s3://lake/orders/year=2026/a.parquet";
    let model = all_orders(&[
        ("eu", &["s3://lake/orders/"]),
        ("recent", &["s3://lake/orders/year=2026/"]),
    ]);
    let json = stored(&compile_model_on(model, &seeded_fs([object]).await).await);
    assert_eq!(
        role_definition(&json, "all_orders")["content"]["null_fill"],
        true
    );
    for base_path in ["s3://lake/orders/", "s3://lake/orders/year=2026/"] {
        assert_eq!(
            json["sources"][&path_source_id(base_path)]["storage"]["target"]["content"]["listings"],
            json!([object]),
            "{base_path}"
        );
    }
}

async fn a_pattern_whose_every_match_is_skipped() -> Golden {
    let fs = seeded_fs(["s3://lake/orders/_meta.parquet"]).await;
    let model = model_of([leaf("orders").path("s3://lake/orders/_*.parquet")]);
    let expected = CompileError::SourcePatternMatchedNothing {
        dataset: "orders".to_owned(),
        pattern: "s3://lake/orders/_*.parquet".to_owned(),
    };
    refusal_golden_on(
        "a_pattern_whose_every_match_is_skipped_is_refused",
        model,
        &fs,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_pattern_whose_every_match_is_skipped_is_refused() {
    a_pattern_whose_every_match_is_skipped()
        .await
        .assert_committed();
}

async fn layout_refusal(case: &str, objects: &[&str], expected: CompileError) -> Golden {
    let fs = seeded_fs(objects.iter().copied()).await;
    let model = model_of([leaf("orders").path("s3://lake/orders/")]);
    refusal_golden_on(case, model, &fs, expected).await
}

async fn a_folder_without_a_key_beside_partition_folders() -> Golden {
    let expected = CompileError::PartitionFolderWithoutKey {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        folder: "s3://lake/orders/misc/".to_owned(),
    };
    layout_refusal(
        "a_folder_without_a_key_beside_partition_folders_is_refused",
        &[
            "s3://lake/orders/year=2026/a.parquet",
            "s3://lake/orders/misc/b.parquet",
        ],
        expected,
    )
    .await
}

#[tokio::test]
async fn a_folder_without_a_key_beside_partition_folders_is_refused() {
    a_folder_without_a_key_beside_partition_folders()
        .await
        .assert_committed();
}

async fn an_object_beside_partition_folders() -> Golden {
    let expected = CompileError::ObjectBesidePartitions {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        object: "s3://lake/orders/a.parquet".to_owned(),
    };
    layout_refusal(
        "an_object_beside_partition_folders_is_refused",
        &[
            "s3://lake/orders/year=2026/b.parquet",
            "s3://lake/orders/a.parquet",
        ],
        expected,
    )
    .await
}

#[tokio::test]
async fn an_object_beside_partition_folders_is_refused() {
    an_object_beside_partition_folders()
        .await
        .assert_committed();
}

async fn partitions_whose_key_sequences_differ() -> Golden {
    let expected = CompileError::PartitionKeySequencesDiffer {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        folder: "s3://lake/orders/year=2026/".to_owned(),
        keys: vec!["year".to_owned()],
        first: vec!["region".to_owned()],
    };
    layout_refusal(
        "partitions_whose_key_sequences_differ_are_refused",
        &[
            "s3://lake/orders/year=2026/a.parquet",
            "s3://lake/orders/region=eu/a.parquet",
        ],
        expected,
    )
    .await
}

#[tokio::test]
async fn partitions_whose_key_sequences_differ_are_refused() {
    partitions_whose_key_sequences_differ()
        .await
        .assert_committed();
}

async fn a_folder_without_a_key_below_the_last_key() -> Golden {
    let expected = CompileError::PartitionFolderWithoutKey {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        folder: "s3://lake/orders/year=2026/data/".to_owned(),
    };
    layout_refusal(
        "a_folder_without_a_key_below_the_last_key_is_refused",
        &["s3://lake/orders/year=2026/data/a.parquet"],
        expected,
    )
    .await
}

#[tokio::test]
async fn a_folder_without_a_key_below_the_last_key_is_refused() {
    a_folder_without_a_key_below_the_last_key()
        .await
        .assert_committed();
}

async fn a_partition_folder_with_an_empty_key() -> Golden {
    let expected = CompileError::PartitionKeyValueEmpty {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        folder: "s3://lake/orders/=2026/".to_owned(),
    };
    layout_refusal(
        "a_partition_folder_with_an_empty_key_is_refused",
        &["s3://lake/orders/=2026/a.parquet"],
        expected,
    )
    .await
}

#[tokio::test]
async fn a_partition_folder_with_an_empty_key_is_refused() {
    a_partition_folder_with_an_empty_key()
        .await
        .assert_committed();
}

async fn a_partition_folder_with_an_empty_value() -> Golden {
    let expected = CompileError::PartitionKeyValueEmpty {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        folder: "s3://lake/orders/year=2026/month=/".to_owned(),
    };
    layout_refusal(
        "a_partition_folder_with_an_empty_value_is_refused",
        &["s3://lake/orders/year=2026/month=/a.parquet"],
        expected,
    )
    .await
}

#[tokio::test]
async fn a_partition_folder_with_an_empty_value_is_refused() {
    a_partition_folder_with_an_empty_value()
        .await
        .assert_committed();
}

async fn two_spellings_of_one_key_in_one_source() -> Golden {
    let expected = CompileError::PartitionKeySpelledTwice {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        first: "Year".to_owned(),
        second: "year".to_owned(),
    };
    layout_refusal(
        "two_spellings_of_one_key_in_one_source_are_refused",
        &[
            "s3://lake/orders/year=2026/a.parquet",
            "s3://lake/orders/Year=2025/a.parquet",
        ],
        expected,
    )
    .await
}

#[tokio::test]
async fn two_spellings_of_one_key_in_one_source_are_refused() {
    two_spellings_of_one_key_in_one_source()
        .await
        .assert_committed();
}

async fn one_key_twice_in_one_folder_path() -> Golden {
    let expected = CompileError::PartitionKeyRepeated {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        folder: "s3://lake/orders/year=2026/YEAR=01/".to_owned(),
        key: "year".to_owned(),
    };
    layout_refusal(
        "one_key_twice_in_one_folder_path_is_refused",
        &["s3://lake/orders/year=2026/YEAR=01/a.parquet"],
        expected,
    )
    .await
}

#[tokio::test]
async fn one_key_twice_in_one_folder_path_is_refused() {
    one_key_twice_in_one_folder_path().await.assert_committed();
}

async fn an_entry_whose_dimension_maps_no_column() -> Golden {
    let mapping = SemanticMapping::builder()
        .literal("order_year", Literal::Integer(2026))
        .build();
    let model = order_years(|orders| {
        (orders.path("s3://lake/orders/"))
            .semantic_mapping(mapping)
            .partition_def(range("order_year"))
    });
    let expected = CompileError::PartitionEntryNotColumn {
        dataset: "orders".to_owned(),
        dimension: "order_year".to_owned(),
    };
    refusal_golden_on(
        "an_entry_whose_dimension_maps_no_column_is_refused",
        model,
        &year_month_folders("year").await,
        expected,
    )
    .await
}

#[tokio::test]
async fn an_entry_whose_dimension_maps_no_column_is_refused() {
    an_entry_whose_dimension_maps_no_column()
        .await
        .assert_committed();
}

async fn an_entry_whose_dimension_authors_an_expression() -> Golden {
    let order_year = bodied(
        "order_year",
        DataType::Integer,
        untyped(Literal::Integer(2026)),
    );
    let model = exposing(order_year, |orders| {
        (orders.path("s3://lake/orders/")).partition_def(range("order_year"))
    });
    let expected = CompileError::PartitionEntryNotColumn {
        dataset: "orders".to_owned(),
        dimension: "order_year".to_owned(),
    };
    refusal_golden_on(
        "an_entry_whose_dimension_authors_an_expression_is_refused",
        model,
        &year_month_folders("year").await,
        expected,
    )
    .await
}

#[tokio::test]
async fn an_entry_whose_dimension_authors_an_expression_is_refused() {
    an_entry_whose_dimension_authors_an_expression()
        .await
        .assert_committed();
}

async fn an_entry_naming_a_dimension_the_dataset_lacks() -> Golden {
    let expected = CompileError::PartitionEntryUnknownDimension {
        dataset: "orders".to_owned(),
        dimension: "order_month".to_owned(),
    };
    refusal_golden_on(
        "an_entry_naming_a_dimension_the_dataset_lacks_is_refused",
        partitioned([range("order_month")]),
        &year_month_folders("year").await,
        expected,
    )
    .await
}

#[tokio::test]
async fn an_entry_naming_a_dimension_the_dataset_lacks_is_refused() {
    an_entry_naming_a_dimension_the_dataset_lacks()
        .await
        .assert_committed();
}

async fn a_dimension_that_two_entries_name() -> Golden {
    let expected = CompileError::PartitionEntryRepeated {
        dataset: "orders".to_owned(),
        dimension: "order_year".to_owned(),
    };
    refusal_golden_on(
        "a_dimension_that_two_entries_name_is_refused",
        partitioned([range("order_year"), list("order_year", &["2025", "2026"])]),
        &year_month_folders("year").await,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_dimension_that_two_entries_name_is_refused() {
    a_dimension_that_two_entries_name().await.assert_committed();
}

fn inapplicable(entry: &str, keys: &[&str]) -> CompileError {
    CompileError::PartitionEntryInapplicable {
        dataset: "orders".to_owned(),
        entry: entry.to_owned(),
        dimension: "order_year".to_owned(),
        key: "year".to_owned(),
        keys: keys.iter().map(|key| (*key).to_owned()).collect(),
    }
}

async fn an_entry_whose_key_one_source_lacks() -> Golden {
    let fs = seeded_fs([
        "s3://lake/eu/year=2026/month=01/a.parquet",
        "s3://lake/us/month=01/a.parquet",
    ])
    .await;
    let model = partitioned_over(["s3://lake/eu/", "s3://lake/us/"], [range("order_year")]);
    refusal_golden_on(
        "an_entry_whose_key_one_source_lacks_is_refused",
        model,
        &fs,
        inapplicable("s3://lake/us/", &["month"]),
    )
    .await
}

#[tokio::test]
async fn an_entry_whose_key_one_source_lacks_is_refused() {
    an_entry_whose_key_one_source_lacks()
        .await
        .assert_committed();
}

async fn an_entry_whose_column_is_no_discovered_key() -> Golden {
    let fs = seeded_fs([
        "s3://lake/orders/region=eu/a.parquet",
        "s3://lake/orders/region=us/a.parquet",
    ])
    .await;
    refusal_golden_on(
        "an_entry_whose_column_is_no_discovered_key_is_refused",
        partitioned([range("order_year")]),
        &fs,
        inapplicable("s3://lake/orders/", &["region"]),
    )
    .await
}

#[tokio::test]
async fn an_entry_whose_column_is_no_discovered_key_is_refused() {
    an_entry_whose_column_is_no_discovered_key()
        .await
        .assert_committed();
}

async fn an_entry_over_two_sources_whose_keys_differ() -> Golden {
    let fs = seeded_fs([
        "s3://lake/eu/year=2026/a.parquet",
        "s3://lake/us/region=us/a.parquet",
    ])
    .await;
    let model = partitioned_over(["s3://lake/eu/", "s3://lake/us/"], [range("order_year")]);
    refusal_golden_on(
        "an_entry_over_two_sources_whose_keys_differ_is_refused",
        model,
        &fs,
        inapplicable("s3://lake/us/", &["region"]),
    )
    .await
}

#[tokio::test]
async fn an_entry_over_two_sources_whose_keys_differ_is_refused() {
    an_entry_over_two_sources_whose_keys_differ()
        .await
        .assert_committed();
}

#[tokio::test]
async fn a_refusal_of_a_globbed_leaf_notes_its_base_path_and_format() {
    let fs = seeded_fs([
        "s3://lake/eu/orders/year=2026/a.parquet",
        "s3://lake/us/orders/region=us/a.parquet",
    ])
    .await;
    let model = partitioned_over(["s3://lake/\x2a/orders/"], [range("order_year")]);

    let diagnostic = compile_diagnostic_with_on(model, None, &fs).await;

    assert_eq!(
        diagnostic.kind,
        inapplicable("s3://lake/\x2a/orders/", &["region"])
    );
    assert_eq!(
        diagnostic.notes,
        ["`orders_orders` reads `s3://lake/us/orders/` as parquet"]
    );
}

#[tokio::test]
async fn an_entry_finds_its_key_by_name_not_by_depth() {
    let fs = seeded_fs([
        "s3://lake/eu/year=2026/month=01/a.parquet",
        "s3://lake/us/month=01/year=2026/a.parquet",
    ])
    .await;
    let model = || partitioned_over(["s3://lake/eu/", "s3://lake/us/"], [range("order_year")]);
    let json = compiled_twice(model, &fs).await;
    let key_values = |base_path: &str| {
        json["sources"][path_source_id(base_path)]["storage"]["target"]["content"]["partition_def"]
            ["partitions"][0]["key_values"]
            .clone()
    };
    let (year, month) = (
        json!({"key": "year", "value": "2026"}),
        json!({"key": "month", "value": "01"}),
    );
    assert_eq!(key_values("s3://lake/eu/"), json!([year, month]));
    assert_eq!(key_values("s3://lake/us/"), json!([month, year]));
}

#[tokio::test]
async fn every_mapping_that_names_the_key_reads_the_folder_value() {
    let model = || {
        let mapping = SemanticMapping::builder()
            .column("order_year", "year")
            .column("latest_year", "year")
            .build();
        let mut model = order_years(|orders| {
            (orders.measure(MeasureSite::r#ref("latest_year")))
                .path("s3://lake/orders/")
                .semantic_mapping(mapping)
                .partition_def(range("order_year"))
        });
        insert_measure(&mut model, measure("latest_year", DataType::Integer, "max"));
        model
    };
    let json = compiled_twice(model, &year_month_folders("year").await).await;
    assert_eq!(
        column_names(&json["sources"][path_source_id("s3://lake/orders/")]),
        ["year"]
    );
}

async fn a_discovered_value_the_list_entry_lacks() -> Golden {
    let expected = CompileError::PartitionValueNotListed {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        dimension: "order_year".to_owned(),
        key: "year".to_owned(),
        value: "2026".to_owned(),
    };
    refusal_golden_on(
        "a_discovered_value_the_list_entry_lacks_is_refused",
        partitioned([list("order_year", &["2025"])]),
        &year_month_folders("year").await,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_discovered_value_the_list_entry_lacks_is_refused() {
    a_discovered_value_the_list_entry_lacks()
        .await
        .assert_committed();
}

async fn a_null_partition() -> InMemoryFileSystem {
    seeded_fs([
        "s3://lake/orders/year=2026/a.parquet",
        "s3://lake/orders/year=__HIVE_DEFAULT_PARTITION__/a.parquet",
    ])
    .await
}

async fn a_null_partition_under_a_list_entry() -> Golden {
    let expected = CompileError::PartitionValueNotListed {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        dimension: "order_year".to_owned(),
        key: "year".to_owned(),
        value: "__HIVE_DEFAULT_PARTITION__".to_owned(),
    };
    refusal_golden_on(
        "a_null_partition_under_a_list_entry_is_refused",
        partitioned([list("order_year", &["2026"])]),
        &a_null_partition().await,
        expected,
    )
    .await
}

#[tokio::test]
async fn a_null_partition_under_a_list_entry_is_refused() {
    a_null_partition_under_a_list_entry()
        .await
        .assert_committed();
}

#[tokio::test]
async fn a_null_partition_under_a_list_entry_that_lists_it_compiles() {
    let entries = [list("order_year", &["2026", "__HIVE_DEFAULT_PARTITION__"])];
    compiled_twice(|| partitioned(entries.clone()), &a_null_partition().await).await;
}

#[tokio::test]
async fn a_listed_value_with_no_folder_compiles() {
    let entries = [list("order_year", &["2024", "2025", "2026"])];
    compiled_twice(
        || partitioned(entries.clone()),
        &year_month_folders("year").await,
    )
    .await;
}

async fn an_entry_over_a_folder_with_no_partitions() -> Golden {
    refusal_golden_on(
        "an_entry_over_a_folder_with_no_partitions_is_refused",
        partitioned([range("order_year")]),
        &seeded_fs(["s3://lake/orders/a.parquet"]).await,
        inapplicable("s3://lake/orders/", &[]),
    )
    .await
}

#[tokio::test]
async fn an_entry_over_a_folder_with_no_partitions_is_refused() {
    an_entry_over_a_folder_with_no_partitions()
        .await
        .assert_committed();
}

async fn a_value_that_does_not_cast(
    case: &str,
    value: &str,
    entries: impl IntoIterator<Item = PartitionDef>,
) -> Golden {
    let expected = CompileError::PartitionValueNotCast {
        dataset: "orders".to_owned(),
        entry: "s3://lake/orders/".to_owned(),
        dimension: "order_year".to_owned(),
        value: value.to_owned(),
        data_type: DataType::Integer,
    };
    let fs = seeded_fs([format!("s3://lake/orders/year={value}/a.parquet")]).await;
    refusal_golden_on(case, partitioned(entries), &fs, expected).await
}

async fn a_word_under_an_entry() -> Golden {
    a_value_that_does_not_cast(
        "a_word_under_an_entry_for_an_integer_is_refused",
        "abc",
        [range("order_year")],
    )
    .await
}

#[tokio::test]
async fn a_word_under_an_entry_for_an_integer_is_refused() {
    a_word_under_an_entry().await.assert_committed();
}

async fn a_word_under_an_inferred_link() -> Golden {
    a_value_that_does_not_cast(
        "a_word_under_an_inferred_link_to_an_integer_is_refused",
        "abc",
        [],
    )
    .await
}

#[tokio::test]
async fn a_word_under_an_inferred_link_to_an_integer_is_refused() {
    a_word_under_an_inferred_link().await.assert_committed();
}

async fn a_fraction_under_an_entry() -> Golden {
    a_value_that_does_not_cast(
        "a_fraction_under_an_entry_for_an_integer_is_refused",
        "1.5",
        [range("order_year")],
    )
    .await
}

#[tokio::test]
async fn a_fraction_under_an_entry_for_an_integer_is_refused() {
    a_fraction_under_an_entry().await.assert_committed();
}

async fn a_fraction_under_an_inferred_link() -> Golden {
    a_value_that_does_not_cast(
        "a_fraction_under_an_inferred_link_to_an_integer_is_refused",
        "1.5",
        [],
    )
    .await
}

#[tokio::test]
async fn a_fraction_under_an_inferred_link_to_an_integer_is_refused() {
    a_fraction_under_an_inferred_link().await.assert_committed();
}

async fn an_entry_on_a_pattern() -> Golden {
    let pattern = "s3://lake/orders/year=2026/month=01/\x2a.parquet";
    refusal_golden_on(
        "an_entry_on_a_pattern_is_refused",
        partitioned_over([pattern], [range("order_year")]),
        &year_month_folders("year").await,
        inapplicable(pattern, &[]),
    )
    .await
}

#[tokio::test]
async fn an_entry_on_a_pattern_is_refused() {
    an_entry_on_a_pattern().await.assert_committed();
}

async fn an_entry_on_uris() -> Golden {
    let uri = "s3://lake/orders/year=2026/month=01/part-0.parquet";
    refusal_golden_on(
        "an_entry_on_uris_is_refused",
        partitioned_over([uri], [range("order_year")]),
        &year_month_folders("year").await,
        inapplicable(uri, &[]),
    )
    .await
}

#[tokio::test]
async fn an_entry_on_uris_is_refused() {
    an_entry_on_uris().await.assert_committed();
}

async fn an_entry_on_a_table() -> Golden {
    let mapping = SemanticMapping::builder()
        .column("order_year", "year")
        .build();
    let model = order_years(|orders| {
        (orders.table("warehouse.orders"))
            .semantic_mapping(mapping)
            .partition_def(range("order_year"))
    });
    refusal_golden_on(
        "an_entry_on_a_table_is_refused",
        model,
        &Unlistable,
        inapplicable("warehouse.orders", &[]),
    )
    .await
}

#[tokio::test]
async fn an_entry_on_a_table_is_refused() {
    an_entry_on_a_table().await.assert_committed();
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            a_uri_written_twice_in_one_list().await,
            a_relative_path().await,
            a_scheme_compile_does_not_read().await,
            a_folder_wildcard_over_an_empty_segment().await,
            a_uri_that_names_no_object().await,
            a_uri_that_names_an_empty_object().await,
            a_path_source_not_read_as_parquet().await,
            a_table_source_that_is_not_iceberg().await,
            a_glob_in_a_table_entry().await,
            two_entries_on_one_base_path().await,
            two_entries_whose_listings_overlap().await,
            a_dataset_that_reads_one_object_twice().await,
            a_pattern_whose_every_match_is_skipped().await,
            a_folder_without_a_key_beside_partition_folders().await,
            an_object_beside_partition_folders().await,
            partitions_whose_key_sequences_differ().await,
            a_folder_without_a_key_below_the_last_key().await,
            a_partition_folder_with_an_empty_key().await,
            a_partition_folder_with_an_empty_value().await,
            two_spellings_of_one_key_in_one_source().await,
            one_key_twice_in_one_folder_path().await,
            an_entry_whose_dimension_maps_no_column().await,
            an_entry_whose_dimension_authors_an_expression().await,
            an_entry_naming_a_dimension_the_dataset_lacks().await,
            a_dimension_that_two_entries_name().await,
            an_entry_whose_key_one_source_lacks().await,
            an_entry_whose_column_is_no_discovered_key().await,
            an_entry_over_two_sources_whose_keys_differ().await,
            a_discovered_value_the_list_entry_lacks().await,
            a_null_partition_under_a_list_entry().await,
            an_entry_over_a_folder_with_no_partitions().await,
            a_word_under_an_entry().await,
            a_word_under_an_inferred_link().await,
            a_fraction_under_an_entry().await,
            a_fraction_under_an_inferred_link().await,
            an_entry_on_a_pattern().await,
            an_entry_on_uris().await,
            an_entry_on_a_table().await,
        ]
    })
    .await;
}
