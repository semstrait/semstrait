mod common;

use common::agg;
use common::golden::{self, Golden};
use std::collections::BTreeMap;
use std::time::{Duration, UNIX_EPOCH};

use semstrait_common::io::InMemoryFileSystem;
use semstrait_ir::SemanticRole;
use semstrait_manifest::{
    compile, mint_keyset_id, mint_semantic_id, mint_semantic_source_id, ExprLayer,
    ManifestProvenance, SemanticManifest, SemanticManifestMetadata,
};
use semstrait_model::{Dataset, Dimension, DimensionSite, Measure, MeasureSite, SemanticModel};
use semstrait_sql::datatype::DataType;

const FIXTURE: &str = include_str!("fixtures/happy_manifest.snapshot.json");

fn fixture_model() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();

    let dimensions = [
        ("is_active", DataType::Boolean),
        ("region", DataType::String),
        ("payload", DataType::Binary),
        ("order_date", DataType::Date),
        ("order_time", DataType::Time { precision: 3 }),
        ("created_at", DataType::Timestamp { precision: 6 }),
        ("billing_period", DataType::Interval),
    ];
    for (name, data_type) in dimensions {
        let dimension = Dimension::builder(name)
            .data_type(data_type)
            .categorical()
            .build();
        let id = dimension.entity_id().expect("the mint is eager").to_owned();
        model.dimensions.insert(id, dimension);
    }

    let tiny_count = Measure::builder("tiny_count")
        .data_type(DataType::TinyInt)
        .agg(agg("count"))
        .build();
    let short_tally = Measure::builder("short_tally")
        .data_type(DataType::SmallInt)
        .agg(agg("sum"))
        .build();
    let item_count = Measure::builder("item_count")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .build();
    let row_count = Measure::builder("row_count")
        .data_type(DataType::BigInt)
        .agg(agg("count_distinct"))
        .build();
    let conversion_rate = Measure::builder("conversion_rate")
        .data_type(DataType::Real)
        .agg(agg("avg"))
        .build();
    let avg_basket = Measure::builder("avg_basket")
        .data_type(DataType::Double)
        .agg(agg("avg"))
        .build();
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("sum"))
        .build();
    for measure in [
        tiny_count,
        short_tally,
        item_count,
        row_count,
        conversion_rate,
        avg_basket,
        revenue,
    ] {
        let id = measure.entity_id().expect("the mint is eager").to_owned();
        model.measures.insert(id, measure);
    }

    let keyed = [
        DimensionSite::r#ref("order_date")
            .with_keysets(["natural", "surrogate"])
            .expect("named keysets"),
        DimensionSite::r#ref("region")
            .with_keysets(["natural"])
            .expect("named keysets"),
        DimensionSite::r#ref("is_active").with_key(),
    ];
    let plain = ["payload", "order_time", "created_at", "billing_period"].map(DimensionSite::r#ref);
    let measures = [
        "tiny_count",
        "short_tally",
        "item_count",
        "row_count",
        "conversion_rate",
        "avg_basket",
        "revenue",
    ]
    .map(MeasureSite::r#ref);

    let orders = Dataset::builder("orders")
        .table("warehouse.orders")
        .dimensions(keyed.into_iter().chain(plain))
        .measures(measures)
        .build();
    let id = orders.body.base.id.clone().expect("the mint is eager");
    model.datasets.insert(id, orders);

    model
}

fn golden_provenance() -> ManifestProvenance {
    let pinned = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    ManifestProvenance::new(pinned, pinned)
}

async fn compile_fixture() -> SemanticManifest {
    compile_fixture_at(golden_provenance()).await
}

async fn compile_fixture_at(provenance: ManifestProvenance) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(fixture_model(), None, &fs, provenance)
        .await
        .expect("the representative fixture model must compile clean (happy path)");
    assert!(
        warnings.is_empty(),
        "a cast to a declared type raises no warning; got {warnings:?}"
    );
    manifest
}

async fn snapshot() -> Golden {
    let manifest = compile_fixture().await;
    Golden::fixture(
        "happy_manifest.snapshot.json",
        serde_json::to_string_pretty(&manifest).expect("serialize the compiled manifest"),
    )
}

#[tokio::test]
async fn compiled_manifest_matches_committed_snapshot() {
    snapshot().await.assert_committed();
}

fn without_provenance(mut manifest: SemanticManifest) -> SemanticManifest {
    manifest.created_at = UNIX_EPOCH;
    manifest.updated_at = UNIX_EPOCH;
    manifest.metadata = SemanticManifestMetadata::new("", None, BTreeMap::new());
    manifest
}

#[tokio::test]
async fn determinism_same_model_twice() {
    let first = without_provenance(compile_fixture().await);
    let second = without_provenance(compile_fixture().await);
    assert_eq!(
        first, second,
        "identical inputs must yield equal load-bearing state"
    );
    let a = serde_json::to_string_pretty(&first).expect("serialize first");
    let b = serde_json::to_string_pretty(&second).expect("serialize second");
    assert_eq!(a, b, "identical inputs must serialize to identical bytes");
}

#[tokio::test]
async fn moving_the_injected_instants_moves_no_load_bearing_state() {
    let early = compile_fixture_at(ManifestProvenance::new(
        UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        UNIX_EPOCH + Duration::from_secs(1_700_000_000),
    ))
    .await;
    let late = compile_fixture_at(ManifestProvenance::new(
        UNIX_EPOCH + Duration::from_secs(1_900_000_001),
        UNIX_EPOCH + Duration::from_secs(1_900_000_002),
    ))
    .await;

    assert_ne!(
        (early.created_at, early.updated_at),
        (late.created_at, late.updated_at),
        "precondition: the two passes were handed different instants"
    );
    assert_eq!(
        without_provenance(early),
        without_provenance(late),
        "an artifact's load-bearing state is a function of the model, the \
         catalog and the filesystem — never of the clock"
    );
}

#[tokio::test]
async fn committed_snapshot_round_trips_and_equals_compiled() {
    let compiled = compile_fixture().await;
    let from_fixture: SemanticManifest =
        serde_json::from_str(FIXTURE).expect("the committed snapshot must deserialize");
    assert!(
        from_fixture.validate().is_ok(),
        "the snapshot manifest must satisfy its own validator"
    );
    assert_eq!(
        without_provenance(from_fixture),
        without_provenance(compiled),
        "the deserialized golden must equal the freshly-compiled manifest over \
         load-bearing state"
    );
}

#[tokio::test]
async fn fixture_is_not_vocabulary_empty() {
    let manifest = compile_fixture().await;

    assert_eq!(
        manifest.semantic_definitions.len(),
        14,
        "fixture should register the full DataType roster (7 dims + 7 measures)"
    );
    let mut typed = 0usize;
    for def in manifest.semantic_definitions.entries.values() {
        assert!(
            def.data_type.is_some(),
            "every fixture semantic declares a concrete data_type"
        );
        typed += 1;
    }
    assert_eq!(typed, 14);

    assert_eq!(manifest.semantic_sources.len(), 1);
    assert_eq!(manifest.semantic_signatures.len(), 14);
    assert_eq!(
        manifest
            .semantic_sources
            .values()
            .next()
            .expect("one leaf")
            .semantic_signature_ids
            .len(),
        14
    );
    let source_id = manifest.sources.keys().next().expect("one source").clone();
    let layers = [
        ExprLayer::Scalar,
        ExprLayer::Aggregate,
        ExprLayer::PostAggregate,
    ]
    .map(|layer| {
        (manifest.semantic_signatures.values())
            .filter(|signature| {
                let entry = manifest
                    .expressions
                    .get(&signature.expression_id)
                    .expect("signature's root resolves in the store");
                assert_eq!(
                    entry.physical_sources.iter().collect::<Vec<_>>(),
                    vec![&source_id]
                );
                entry.layer == layer
            })
            .count()
    });
    assert_eq!(
        layers,
        [7, 2, 5],
        "Scalar, Aggregate and PostAggregate roots"
    );

    assert_eq!(
        manifest.expressions.len(),
        14,
        "one entry per signature's expression"
    );
    assert!(
        !FIXTURE.contains("depends_on"),
        "the expression store carries no `depends_on` edge"
    );

    assert!(
        FIXTURE.contains("\"decimal(18,2)\""),
        "the golden must embed the parametric Decimal type token"
    );
    assert!(
        FIXTURE.contains("\"timestamp(6)\"") && FIXTURE.contains("\"time(3)\""),
        "the golden must embed the parametric temporal type tokens"
    );
    assert!(
        FIXTURE.contains("\"interval\"") && FIXTURE.contains("\"boolean\""),
        "the golden must embed the scalar type tokens"
    );
    assert!(
        FIXTURE.contains("\"type\": \"column\""),
        "the golden must embed at least one column"
    );
    assert_eq!(manifest.keysets.len(), 3);
    let orders_id = mint_semantic_source_id("orders");
    let mut rendered: Vec<(String, String)> = manifest
        .keysets
        .values()
        .map(|keyset| {
            assert_eq!(
                keyset.id,
                mint_keyset_id(&orders_id, keyset.name.as_str()),
                "every persisted keyset id must be its own seed mint"
            );
            (keyset.name.as_str().to_owned(), keyset.id.clone())
        })
        .collect();
    rendered.sort();
    assert_eq!(
        rendered.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        vec!["default", "natural", "surrogate"]
    );

    let dim = |name: &str| mint_semantic_id(SemanticRole::Dimension, name);
    let members_of = |keyset_id: &str| -> Vec<String> {
        let mut members: Vec<String> = manifest
            .semantic_signatures
            .values()
            .filter(|signature| signature.keysets.iter().any(|id| id == keyset_id))
            .map(|signature| signature.definition_id.clone())
            .collect();
        members.sort();
        members
    };
    let mut natural = vec![dim("order_date"), dim("region")];
    natural.sort();
    assert_eq!(
        members_of(&mint_keyset_id(&orders_id, "default")),
        vec![dim("is_active")]
    );
    assert_eq!(members_of(&mint_keyset_id(&orders_id, "natural")), natural);
    assert_eq!(
        members_of(&mint_keyset_id(&orders_id, "surrogate")),
        vec![dim("order_date")]
    );
    assert!(
        FIXTURE.contains("\"keysets\""),
        "the golden must embed the keyset roster"
    );
    assert_eq!(manifest.manifest_version, 4);

    assert!(
        FIXTURE.contains("\"type\": \"aggregate\""),
        "the golden must embed at least one aggregate call"
    );
}

#[test]
fn the_golden_commits_the_identity_block_and_nothing_retired() {
    let golden: serde_json::Value =
        serde_json::from_str(FIXTURE).expect("the committed snapshot must deserialize");
    let block = golden.as_object().expect("the artifact is a JSON object");

    for field in [
        "manifest_version",
        "created_at",
        "updated_at",
        "model_hash",
        "model_id",
        "dialect",
    ] {
        assert!(
            block.contains_key(field),
            "the identity block must carry `{field}`"
        );
    }
    for retired in [
        "manifest_epoch",
        "catalog_fingerprint",
        "model_checksum",
        "semantic_bitmap_index",
    ] {
        assert!(
            !block.contains_key(retired),
            "`{retired}` must not reach the artifact"
        );
    }
}

#[tokio::test]
async fn each_signature_s_entry_holds_its_whole_tree_under_its_minted_id() {
    let manifest = compile_fixture().await;
    let expression_id = |semantic| common::signature(&manifest, "orders", semantic).expression_id;

    let revenue = expression_id("revenue");
    assert_eq!(
        serde_json::to_string(&manifest.expressions.physical[&revenue]).expect("an entry writes"),
        r#"{"id":"253641c7-819a-5b60-a3dc-c276bfceb36d","expr":{"type":"cast","content":{"input":{"type":"aggregate","content":{"name":"sum","args":[{"type":"cast","content":{"input":{"type":"column","content":["ps0","revenue"]},"target":"decimal(18,2)","on_failure":"error"}}]}},"target":"decimal(18,2)","on_failure":"error"}},"layer":"post_aggregate","physical_sources":["bf575591-d8c6-59c7-b3aa-5fbdf5673b12"]}"#
    );
    for (semantic, id) in [
        ("region", "cb8b6c6b-b311-5484-9a64-5966edb7851d"),
        ("item_count", "15184182-69e5-55a9-9292-b09b5731f128"),
        ("short_tally", "09712050-49b8-534f-98f1-34d64cf76e3c"),
    ] {
        assert_eq!(
            expression_id(semantic).to_string(),
            id,
            "{semantic}'s entry"
        );
    }
}

#[test]
fn the_golden_writes_each_expression_id_as_uuid_text() {
    let golden: serde_json::Value =
        serde_json::from_str(FIXTURE).expect("the committed snapshot must deserialize");
    let signatures = golden["semantic_signatures"]
        .as_object()
        .expect("the signatures are a JSON object");
    for (id, signature) in signatures {
        let expression_id = signature["expression_id"].as_str();
        assert_eq!(
            expression_id.map(str::len),
            Some(36),
            "signature {id}'s `expression_id` is UUID text: {expression_id:?}"
        );
    }
    let physical = golden["expressions"]["physical"]
        .as_object()
        .expect("the store is a JSON object");
    for key in physical.keys() {
        assert_eq!(key.len(), 36, "the store's key {key} is UUID text");
    }
    assert_eq!(physical.len(), 14);
}

#[test]
fn the_golden_pins_manifest_version_four() {
    let golden: serde_json::Value =
        serde_json::from_str(FIXTURE).expect("the committed snapshot must deserialize");
    assert_eq!(golden["manifest_version"], serde_json::json!(4));
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async { vec![snapshot().await] }).await;
}
