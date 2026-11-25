mod common;

use common::{
    agg, bodied, by_name, column, compile_model_with, fixture_fs, identity_mapping, lake_folder,
    link, listed, physical_call, signature, stored_expr, stored_tree, test_provenance,
    warehouse_orders, MapCatalog,
};
use semstrait_catalog::{Schema as NativeSchema, SchemaColumn as NativeColumn};
use semstrait_common::Granularity;
use semstrait_model::expr::dim;
use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_manifest::{compile, Coverage, RoleDefinition, SemanticManifest, SemanticSource};
use semstrait_model::{
    Dataset, Dimension, DimensionSite, Grainset, Joinset, Measure, MeasureSite, NestedDataset,
    Relationship, SemanticModel, StorageFormat, TemporalShape, Unionset,
};
use semstrait_sql::datatype::DataType;

const FILES: [&str; 8] = [
    "orders", "people", "lines", "shops", "east", "west", "fine", "coarse",
];

fn leaf(name: &str, names: &[&str], shape: Option<TemporalShape>) -> NestedDataset {
    let mut built = NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(names));
    if let Some(shape) = shape {
        built = built.temporal(shape);
    }
    built.build()
}

fn fixture_model() -> SemanticModel {
    let orders = Dataset::builder("orders")
        .path(lake_folder("orders"))
        .format(StorageFormat::Parquet)
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    let people = Dataset::builder("people")
        .path(lake_folder("people"))
        .format(StorageFormat::Parquet)
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("channel"))
        .build();

    let basket = Joinset::builder("basket")
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(leaf("lines", &["region", "revenue"], None))
        .dataset(leaf("shops", &["region", "revenue"], None))
        .relationship(
            Relationship::builder()
                .name("lines_to_shops")
                .from("lines")
                .to("shops")
                .on(["region"])
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();

    let stacked = Unionset::builder("stacked")
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"))
        .union_all()
        .dataset(leaf("east", &["region", "revenue"], None))
        .dataset(leaf("west", &["region", "revenue"], None))
        .build();

    let family = Grainset::builder("family")
        .dimension(DimensionSite::r#ref("region").with_key())
        .dimension(DimensionSite::r#ref("order_ts"))
        .measure(MeasureSite::r#ref("revenue"))
        .dataset(leaf(
            "fine",
            &["region", "order_ts", "revenue"],
            Some(TemporalShape::events("order_ts", Some(Granularity::Day))),
        ))
        .dataset(leaf(
            "coarse",
            &["region", "order_ts", "revenue"],
            Some(TemporalShape::events("order_ts", Some(Granularity::Month))),
        ))
        .build();

    SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("region")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("channel")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("order_ts")
                .data_type(DataType::Date)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
        .dataset(orders)
        .dataset(people)
        .joinset(basket)
        .unionset(stacked)
        .grainset(family)
        .build()
        .expect("the fixture model must build")
        .0
}

async fn compiled() -> SemanticManifest {
    let fs = fixture_fs(&FILES).await;
    let (manifest, warnings) = compile(fixture_model(), None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    assert!(
        manifest.validate().is_ok(),
        "compile must not emit an artifact its own validator refuses: {:?}",
        manifest.validate()
    );
    manifest
}

async fn observed_region() -> SemanticManifest {
    let orders = Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(DimensionSite::r#ref("region"))
        .dimension(DimensionSite::r#ref("zone2"))
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("region")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(bodied(
            "zone2",
            DataType::String,
            dim("region").expect("a name"),
        ))
        .dataset(orders)
        .build()
        .expect("the observed model must build")
        .0;
    let schema = NativeSchema::new(vec![NativeColumn::new("region", "string", true)]);
    let catalog = MapCatalog::new(warehouse_orders(), schema);
    compile_model_with(model, Some(&catalog)).await
}

fn artifact_json(manifest: &SemanticManifest) -> serde_json::Value {
    serde_json::from_str(&serde_json::to_string(manifest).expect("the artifact must serialise"))
        .expect("the artifact's own encoding must parse")
}

fn semantics(manifest: &SemanticManifest, name: &str) -> EntityId {
    manifest
        .semantic_definitions
        .by_name(name)
        .unwrap_or_else(|| panic!("`{name}` must be registered"))
        .id
        .clone()
}

fn served(manifest: &SemanticManifest, entry: &SemanticSource) -> Vec<String> {
    let by_position: BTreeMap<u32, &str> = manifest
        .semantic_definitions
        .entries
        .values()
        .map(|def| (def.bit_position, def.name.as_str()))
        .collect();
    let mut names: Vec<String> = entry
        .semantic_bitmask
        .set_positions()
        .into_iter()
        .map(|position| {
            by_position
                .get(&position)
                .unwrap_or_else(|| panic!("bit {position} addresses no registry entry"))
                .to_string()
        })
        .collect();
    names.sort();
    names
}

fn served_by(manifest: &SemanticManifest, name: &str) -> Vec<String> {
    served(manifest, by_name(manifest, name))
}

#[tokio::test]
async fn a_set_bit_says_this_source_serves_that_semantics_as_it_stands() {
    let manifest = compiled().await;

    assert_eq!(served_by(&manifest, "orders"), vec!["region", "revenue"]);
    assert_eq!(served_by(&manifest, "people"), vec!["channel", "region"]);

    let channel = manifest
        .semantic_definitions
        .get(semantics(&manifest, "channel").as_str())
        .expect("`channel` is registered")
        .bit_position;
    assert!(
        !by_name(&manifest, "orders")
            .semantic_bitmask
            .contains(channel),
        "`orders` cannot serve `channel`, so its bit must be clear"
    );
    assert!(
        by_name(&manifest, "people")
            .semantic_bitmask
            .contains(channel),
        "`people` can serve `channel` as it stands"
    );
}

#[tokio::test]
async fn a_leafs_mask_is_what_its_one_source_can_serve() {
    let manifest = compiled().await;

    assert_eq!(manifest.semantic_definitions.len(), 4);
    for name in ["orders", "people", "basket.lines", "stacked.east"] {
        let entry = by_name(&manifest, name);
        assert!(
            entry.semantic_bitmask.count_ones() < 4,
            "`{name}` must serve only what it realises, not the registry"
        );
        assert!(
            entry.semantic_bitmask.is_canonical(),
            "`{name}`'s mask must be canonical"
        );
    }
    assert_eq!(
        served_by(&manifest, "basket.lines"),
        vec!["region", "revenue"]
    );
}

#[tokio::test]
async fn a_joinsets_mask_is_the_union_of_its_two_operands() {
    let manifest = compiled().await;

    assert_eq!(served_by(&manifest, "basket"), vec!["region", "revenue"]);
    let mut union: BTreeSet<String> = BTreeSet::new();
    for operand in ["basket.lines", "basket.shops"] {
        union.extend(served_by(&manifest, operand));
        assert!(
            by_name(&manifest, operand)
                .semantic_bitmask
                .is_subset_of(&by_name(&manifest, "basket").semantic_bitmask),
            "`{operand}`'s coverage must reach the composition above it"
        );
    }
    assert_eq!(
        served_by(&manifest, "basket"),
        union.into_iter().collect::<Vec<String>>()
    );
}

#[tokio::test]
async fn a_unionsets_mask_is_the_union_of_its_branches() {
    let manifest = compiled().await;

    assert_eq!(served_by(&manifest, "stacked"), vec!["region", "revenue"]);
    let mut union: BTreeSet<String> = BTreeSet::new();
    for branch in ["stacked.east", "stacked.west"] {
        union.extend(served_by(&manifest, branch));
        assert!(
            by_name(&manifest, branch)
                .semantic_bitmask
                .is_subset_of(&by_name(&manifest, "stacked").semantic_bitmask),
            "`{branch}`'s coverage must reach the stack above it"
        );
    }
    assert_eq!(
        served_by(&manifest, "stacked"),
        union.into_iter().collect::<Vec<String>>()
    );
}

#[tokio::test]
async fn a_grainsets_mask_is_the_union_of_its_grains() {
    let manifest = compiled().await;

    let expected = vec![
        "order_ts".to_owned(),
        "region".to_owned(),
        "revenue".to_owned(),
    ];
    assert_eq!(served_by(&manifest, "family"), expected);
    let mut union: BTreeSet<String> = BTreeSet::new();
    for grain in ["family.fine", "family.coarse"] {
        union.extend(served_by(&manifest, grain));
        assert!(
            by_name(&manifest, grain)
                .semantic_bitmask
                .is_subset_of(&by_name(&manifest, "family").semantic_bitmask),
            "`{grain}`'s coverage must reach the family above it"
        );
    }
    assert_eq!(
        served_by(&manifest, "family"),
        union.into_iter().collect::<Vec<String>>()
    );
    assert_eq!(served_by(&manifest, "family.fine_coarse"), expected);
}

#[tokio::test]
async fn exactly_one_bit_lane_addresses_the_definitions_registry() {
    let manifest = compiled().await;
    let json: serde_json::Value = artifact_json(&manifest);

    let mut lanes: Vec<String> = Vec::new();
    collect_word_vectors(&json, "", &mut lanes);
    assert!(
        !lanes.is_empty(),
        "the fixture must carry at least one mask, or this asserts nothing"
    );
    for lane in &lanes {
        assert!(
            lane.ends_with("semantic_bitmask.words"),
            "the only bit lane is `semantic_bitmask`; found one at `{lane}`"
        );
    }

    for entry in manifest.semantic_sources.values() {
        for position in entry.semantic_bitmask.set_positions() {
            assert!(
                manifest
                    .semantic_definitions
                    .entries
                    .values()
                    .any(|def| def.bit_position == position),
                "`{}` sets bit {position}, which addresses no registered semantics",
                entry.name.as_str()
            );
        }
    }

    let mut memberships = 0usize;
    for signature in manifest.semantic_signatures.values() {
        for keyset_id in &signature.keysets {
            assert!(
                manifest.keysets.contains_key(keyset_id),
                "keyset membership is an id into the roster, not a bit"
            );
            memberships += 1;
        }
    }
    assert!(
        memberships > 0,
        "the fixture must carry keyset membership, or this asserts nothing"
    );
    for source in manifest.sources.values() {
        assert!(
            json["sources"][source.id.as_str()]
                .get("semantic_bitmask")
                .is_none(),
            "a physical source carries no mask"
        );
    }
}

fn collect_word_vectors(value: &serde_json::Value, path: &str, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let next = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                if key == "words" {
                    out.push(next.clone());
                }
                collect_word_vectors(child, &next, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_word_vectors(item, &format!("{path}[]"), out);
            }
        }
        _ => {}
    }
}

#[tokio::test]
async fn no_reverse_index_of_any_kind_is_persisted() {
    let manifest = compiled().await;
    let json: serde_json::Value = artifact_json(&manifest);

    let mut keys: BTreeSet<String> = BTreeSet::new();
    collect_keys(&json, &mut keys);
    for forbidden in [
        "semantic_bitmap_index",
        "bit_index",
        "by_bit",
        "by_name",
        "by_alias",
        "alias_index",
        "consumers",
        "consumed_by",
    ] {
        assert!(
            !keys.contains(forbidden),
            "`{forbidden}` is a persisted reverse index (ADR 0055)"
        );
    }
    for key in &keys {
        assert!(
            !key.ends_with("_index"),
            "`{key}` reads as a persisted index (ADR 0055)"
        );
    }

    assert_eq!(
        manifest
            .semantic_definitions
            .by_name("channel")
            .map(|def| def.bit_position),
        Some(
            manifest
                .semantic_definitions
                .get(semantics(&manifest, "channel").as_str())
                .expect("`channel` is registered")
                .bit_position
        )
    );
}

fn collect_keys(value: &serde_json::Value, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                out.insert(key.clone());
                collect_keys(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_keys(item, out);
            }
        }
        _ => {}
    }
}

#[tokio::test]
async fn the_coverage_axis_is_total_and_three_way() {
    let manifest = compiled().await;

    let orders = by_name(&manifest, "orders").id.clone();
    let people = by_name(&manifest, "people").id.clone();

    assert_eq!(
        manifest.coverage_of(&semantics(&manifest, "region"), &orders),
        Coverage::Derived,
        "a column cast to the declared type is not a bare column leaf"
    );
    let observed = observed_region().await;
    assert_eq!(
        observed.coverage_of(
            &semantics(&observed, "region"),
            &by_name(&observed, "orders").id
        ),
        Coverage::Native,
        "a bare column leaf is `Native`"
    );
    assert_eq!(
        manifest.coverage_of(&semantics(&manifest, "revenue"), &orders),
        Coverage::Derived,
        "an aggregate is not a bare column leaf, so it is `Derived`"
    );
    assert_eq!(
        manifest.coverage_of(&semantics(&manifest, "channel"), &orders),
        Coverage::NullFill,
        "no realisation exists for the pair"
    );
    assert_eq!(
        manifest.coverage_of(&semantics(&manifest, "revenue"), &people),
        Coverage::NullFill,
    );

    for entry in manifest.semantic_sources.values() {
        for definition in manifest.semantic_definitions.entries.values() {
            let _: Coverage = manifest.coverage_of(&definition.id, &entry.id);
        }
    }
}

#[tokio::test]
async fn an_alias_over_a_native_column_is_native() {
    let observed = observed_region().await;
    let orders = &by_name(&observed, "orders").id;
    let zone2 = signature(&observed, "orders", "zone2").expression_id;
    assert_eq!(
        stored_expr(&observed, zone2),
        &link(signature(&observed, "orders", "region").expression_id)
    );
    assert_eq!(
        observed.coverage_of(&semantics(&observed, "zone2"), orders),
        Coverage::Native
    );
}

#[tokio::test]
async fn a_composition_holds_a_realisation_for_what_its_mask_advertises() {
    let manifest = compiled().await;
    let region = semantics(&manifest, "region");
    let basket = by_name(&manifest, "basket");

    assert!(
        served(&manifest, basket).contains(&"region".to_owned()),
        "the fold advertises `region` on the composition, because both constituents serve it"
    );
    assert_eq!(
        manifest.coverage_of(&region, &basket.id),
        Coverage::Derived,
        "the composition holds a realisation for the semantics it advertises"
    );

    let signature = listed(&manifest, basket)
        .find(|signature| signature.definition_id == region)
        .expect("the composition owns a signature for `region`");
    let side = |operand| column(by_name(&manifest, operand).alias.as_str(), "region");
    assert_eq!(
        stored_tree(&manifest, signature.expression_id),
        physical_call("coalesce", vec![side("basket.lines"), side("basket.shops")]),
        "each operand's projected column, qualified by that operand's alias"
    );
}

#[tokio::test]
async fn a_leafs_mask_never_carries_a_null_filled_bit() {
    let manifest = compiled().await;

    let mut leaves = 0usize;
    for entry in manifest.semantic_sources.values() {
        if !matches!(entry.role_definition, RoleDefinition::Dataset { .. }) {
            continue;
        }
        leaves += 1;
        for definition in manifest.semantic_definitions.entries.values() {
            if manifest.coverage_of(&definition.id, &entry.id) == Coverage::NullFill {
                assert!(
                    !entry.semantic_bitmask.contains(definition.bit_position),
                    "leaf `{}` sets the bit for `{}`, which it has no realisation for",
                    entry.name.as_str(),
                    definition.name.as_str()
                );
            }
        }
    }
    assert!(
        leaves >= 6,
        "the fixture must hold several leaves, got {leaves}"
    );
}

#[tokio::test]
async fn every_set_bit_addresses_the_position_the_freeze_assigned() {
    let manifest = compiled().await;

    let positions: BTreeSet<u32> = manifest
        .semantic_definitions
        .entries
        .values()
        .map(|def| def.bit_position)
        .collect();
    assert_eq!(
        positions,
        (0..u32::try_from(manifest.semantic_definitions.len()).expect("a small registry"))
            .collect(),
        "the freeze assigns a dense position set"
    );
    for entry in manifest.semantic_sources.values() {
        for position in entry.semantic_bitmask.set_positions() {
            assert!(
                positions.contains(&position),
                "`{}` sets bit {position}, outside the frozen position set",
                entry.name.as_str()
            );
        }
    }
}
