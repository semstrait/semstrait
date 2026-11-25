mod common;

use common::golden::{self, Golden};
use common::{agg, constituent, fixture_fs, model_from_yaml, stored_tree, test_provenance};
use std::collections::BTreeMap;

use semstrait_common::io::InMemoryFileSystem;
use semstrait_ir::SemanticRole;
use semstrait_manifest::{
    compile, mint_keyset_id, mint_semantic_source_id, CompileError, SemanticManifest,
};
use semstrait_model::{
    Dimension, DimensionSite, Joinset, Measure, MeasureSite, Relationship, SemanticModel,
};
use semstrait_sql::datatype::DataType;

const ORDER_NAMES: [&str; 3] = ["customer_id", "order_id", "revenue"];

fn model_yaml(pooled: &[&str], interface: &str) -> String {
    let pool: String = pooled
        .iter()
        .map(|name| {
            format!("    - name: {name}\n      data_type: string\n      type: categorical\n")
        })
        .collect();
    format!(
        "semantic_model:\n  name: keysets\n  dimensions:\n{pool}  datasets:\n    \
         - name: orders\n      extras:\n        semantic_mapping: auto\n        storage:\n          \
         tables: [warehouse.orders]\n{interface}"
    )
}

fn model(pooled: &[&str], interface: &str) -> SemanticModel {
    model_from_yaml(&model_yaml(pooled, interface))
}

async fn compile_interface(pooled: &[&str], interface: &str) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(model(pooled, interface), None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

async fn compile_yaml(yaml: &str) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(model_from_yaml(yaml), None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

fn roster(manifest: &SemanticManifest) -> Vec<(String, String, Vec<String>)> {
    let semantic_name = |id: &str| -> String {
        manifest
            .semantic_definitions
            .entries
            .get(id)
            .map_or_else(|| id.to_owned(), |def| def.name.as_str().to_owned())
    };

    let mut owner: BTreeMap<String, String> = BTreeMap::new();
    for source in manifest.semantic_sources.values() {
        for keyset in manifest.keysets.values() {
            if mint_keyset_id(&source.id, keyset.name.as_str()) == keyset.id {
                owner.insert(keyset.id.clone(), source.name.as_str().to_owned());
            }
        }
    }

    let mut membership: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for source in manifest.semantic_sources.values() {
        for signature_id in &source.semantic_signature_ids {
            let Some(signature) = manifest.semantic_signatures.get(signature_id) else {
                continue;
            };
            for keyset_id in &signature.keysets {
                membership
                    .entry(keyset_id.as_str())
                    .or_default()
                    .push(semantic_name(&signature.definition_id));
            }
        }
    }

    let mut out: Vec<(String, String, Vec<String>)> = manifest
        .keysets
        .values()
        .map(|keyset| {
            let mut members = membership
                .get(keyset.id.as_str())
                .cloned()
                .unwrap_or_default();
            members.sort();
            (
                owner.get(&keyset.id).cloned().unwrap_or_default(),
                keyset.name.as_str().to_owned(),
                members,
            )
        })
        .collect();
    out.sort();
    out
}

fn entry(source: &str, name: &str, members: &[&str]) -> (String, String, Vec<String>) {
    let mut members: Vec<String> = members.iter().map(|m| (*m).to_owned()).collect();
    members.sort();
    (source.to_owned(), name.to_owned(), members)
}

#[tokio::test]
async fn bare_is_key_materialises_the_default_keyset() {
    let manifest = compile_interface(
        &["customer_id"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n",
    )
    .await;
    assert_eq!(
        roster(&manifest),
        vec![entry("orders", "default", &["customer_id"])]
    );
}

#[tokio::test]
async fn the_id_seeds_the_owning_source_and_the_name_and_nothing_else() {
    let manifest = compile_interface(
        &["customer_id", "order_ts"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n          \
         keysets: [natural]\n        - ref: order_ts\n          is_key: true\n          \
         keysets: [natural, surrogate]\n",
    )
    .await;
    let orders = mint_semantic_source_id("orders");
    let mut ids: Vec<&String> = manifest.keysets.keys().collect();
    ids.sort();
    let mut expected = [
        mint_keyset_id(&orders, "natural"),
        mint_keyset_id(&orders, "surrogate"),
    ];
    expected.sort();
    assert_eq!(ids, expected.iter().collect::<Vec<_>>());
    assert_ne!(expected[0], expected[1]);
}

async fn two_table_reduction() -> SemanticManifest {
    compile_yaml(
        r"semantic_model:
  name: keysets
  dimensions:
    - name: customer_id
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders_eu, warehouse.orders_us]
      dimensions:
        - ref: customer_id
          is_key: true
",
    )
    .await
}

#[tokio::test]
async fn one_authored_name_is_one_entity_per_source() {
    let manifest = two_table_reduction().await;

    assert_eq!(
        roster(&manifest),
        vec![
            entry("orders", "default", &["customer_id"]),
            entry("orders_orders_eu", "default", &["customer_id"]),
            entry("orders_orders_us", "default", &["customer_id"]),
        ],
        "one name over three sources is three entities, not one deduplicated entry"
    );
}

#[tokio::test]
async fn the_composers_default_keyset_carries_the_authored_member() {
    let manifest = two_table_reduction().await;
    assert!(
        roster(&manifest).contains(&entry("orders", "default", &["customer_id"])),
        "the composer's own entry carries the member the author declared"
    );
}

#[tokio::test]
async fn two_datasets_declaring_one_shape_are_two_entities() {
    let manifest = compile_yaml(
        r"semantic_model:
  name: keysets
  dimensions:
    - name: customer_id
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
      dimensions:
        - ref: customer_id
          is_key: true
    - name: returns
      extras:
        storage:
          tables: [warehouse.returns]
      dimensions:
        - ref: customer_id
          is_key: true
",
    )
    .await;

    assert_eq!(
        roster(&manifest),
        vec![
            entry("orders", "default", &["customer_id"]),
            entry("returns", "default", &["customer_id"]),
        ]
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn the_singular_spelling_merges_into_the_plural() {
    let singular = compile_interface(
        &["customer_id"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n          \
         keyset: natural\n",
    )
    .await;
    let plural = compile_interface(
        &["customer_id"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n          \
         keysets: [natural]\n",
    )
    .await;
    assert_eq!(
        roster(&singular),
        vec![entry("orders", "natural", &["customer_id"])]
    );
    assert_eq!(roster(&singular), roster(&plural));
    assert_eq!(
        singular.keysets, plural.keysets,
        "one spelling of one fact ⇒ one entry under one id"
    );
}

#[tokio::test]
async fn cover_arity_puts_one_site_in_every_keyset() {
    let manifest = compile_interface(
        &["customer_id", "order_ts", "tenant"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n          \
         keysets: [natural]\n        - ref: order_ts\n          is_key: true\n          \
         keysets: [surrogate]\n        - ref: tenant\n          is_key: true\n          \
         keysets: [natural, surrogate]\n",
    )
    .await;
    assert_eq!(
        roster(&manifest),
        vec![
            entry("orders", "natural", &["customer_id", "tenant"]),
            entry("orders", "surrogate", &["order_ts", "tenant"]),
        ]
    );
}

#[tokio::test]
async fn identity_is_stable_under_member_reordering() {
    let pooled = ["customer_id", "order_ts"];
    let one = compile_interface(
        &pooled,
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n        \
         - ref: order_ts\n          is_key: true\n",
    )
    .await;
    let reordered = compile_interface(
        &pooled,
        "      dimensions:\n        - ref: order_ts\n          is_key: true\n        \
         - ref: customer_id\n          is_key: true\n",
    )
    .await;
    let expected = mint_keyset_id(&mint_semantic_source_id("orders"), "default");
    assert_eq!(one.keysets.keys().collect::<Vec<_>>(), vec![&expected]);
    assert_eq!(
        reordered.keysets.keys().collect::<Vec<_>>(),
        vec![&expected]
    );
    assert_eq!(one.keysets, reordered.keysets);
    assert_eq!(roster(&one), roster(&reordered));
}

#[tokio::test]
async fn declaring_no_keyset_mints_nothing_and_warns_about_nothing() {
    let manifest = compile_interface(
        &["customer_id"],
        "      dimensions:\n        - ref: customer_id\n",
    )
    .await;
    assert!(manifest.keysets.is_empty());
    assert!(
        manifest
            .semantic_signatures
            .values()
            .all(|signature| signature.keysets.is_empty()),
        "an absent declaration leaves the list empty, which is its one spelling"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn the_manifest_version_reads_four() {
    let manifest = compile_interface(
        &["customer_id"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n",
    )
    .await;
    assert_eq!(manifest.manifest_version, 4);
}

#[tokio::test]
async fn the_roster_round_trips_and_re_encodes_byte_identically() {
    let manifest = compile_interface(
        &["customer_id", "order_ts"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n          \
         keysets: [natural, surrogate]\n        - ref: order_ts\n          is_key: true\n          \
         keysets: [surrogate]\n",
    )
    .await;
    assert_eq!(
        manifest.keysets.len(),
        2,
        "precondition: a populated roster"
    );

    let first = serde_json::to_string(&manifest).expect("serialize");
    let back: SemanticManifest = serde_json::from_str(&first).expect("deserialize");
    assert_eq!(back, manifest);
    let second = serde_json::to_string(&back).expect("re-serialize");
    assert_eq!(first, second, "a decode must re-encode byte-identically");
    assert!(
        back.validate().is_ok(),
        "a compiled roster is valid by construction"
    );
}

#[tokio::test]
async fn a_compiled_roster_is_valid_by_construction() {
    let manifest = compile_interface(
        &["customer_id", "order_ts"],
        "      dimensions:\n        - ref: customer_id\n          is_key: true\n        \
         - ref: order_ts\n          is_key: true\n          keysets: [surrogate]\n",
    )
    .await;
    assert_eq!(manifest.keysets.len(), 2);
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn an_inline_only_keyed_site_reaches_the_registry_and_is_keyed() {
    let manifest = compile_interface(
        &["customer_id"],
        "      dimensions:\n        - ref: customer_id\n        - name: region\n          \
         data_type: string\n          type: categorical\n          is_key: true\n",
    )
    .await;
    assert_eq!(
        roster(&manifest),
        vec![entry("orders", "default", &["region"])],
        "the inline-only site is the `default` keyset's sole member"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn a_keyed_site_resolving_to_a_measure_is_refused_naming_the_role() {
    keyed_measure_refusal().await.assert_committed();
}

async fn keyed_measure_refusal() -> Golden {
    let yaml = r"semantic_model:
  name: keysets
  dimensions:
    - name: customer_id
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
        storage:
          tables: [warehouse.orders]
      dimensions:
        - ref: customer_id
        - ref: revenue
          is_key: true
          keysets: [natural]
      measures:
        - ref: revenue
";
    let fs = InMemoryFileSystem::new();
    let (error, _warnings) = compile(model_from_yaml(yaml), None, &fs, test_provenance())
        .await
        .expect_err("a Measure cannot be a keyset member");
    match &error.kind {
        CompileError::KeysetMemberNotADimension {
            data_kind,
            keyset,
            site,
            role,
        } => {
            assert_eq!(*data_kind, mint_semantic_source_id("orders"));
            assert_eq!(keyset.as_str(), "natural");
            assert_eq!(site, "revenue");
            assert_eq!(*role, SemanticRole::Measure);
        }
        other => panic!("expected KeysetMemberNotADimension, got {other:?}"),
    }
    Golden::message(
        "a_keyed_site_resolving_to_a_measure_is_refused_naming_the_role",
        error.to_string(),
    )
}

async fn joinset_fs() -> InMemoryFileSystem {
    fixture_fs(&["customers", "orders", "regions"]).await
}

fn on_customer_id() -> Relationship {
    Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(["customer_id"])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

fn joinset_model(natural_keysets: &[&str], order_id_keysets: &[&str]) -> SemanticModel {
    joinset_model_joined(natural_keysets, order_id_keysets, [on_customer_id()])
}

fn joinset_model_joined(
    natural_keysets: &[&str],
    order_id_keysets: &[&str],
    relationships: impl IntoIterator<Item = Relationship>,
) -> SemanticModel {
    let joinset = Joinset::builder("sales")
        .dataset(constituent("customers", &ORDER_NAMES))
        .dataset(constituent("orders", &ORDER_NAMES))
        .dimension(
            DimensionSite::r#ref("customer_id")
                .with_key()
                .with_keysets(natural_keysets.iter().copied())
                .expect("a non-empty keyset list builds"),
        )
        .dimension(
            DimensionSite::r#ref("order_id")
                .with_key()
                .with_keysets(order_id_keysets.iter().copied())
                .expect("a non-empty keyset list builds"),
        )
        .measure(MeasureSite::r#ref("revenue"))
        .relationships(relationships)
        .build();
    SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("customer_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("order_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0
}

#[tokio::test]
async fn a_joinsets_keysets_are_derived_from_its_operands() {
    let fs = joinset_fs().await;
    let (manifest, warnings) = compile(
        joinset_model(&["natural"], &["pk"]),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    assert_eq!(
        names_on(&manifest, "sales"),
        vec![
            "natural".to_owned(),
            "pk".to_owned(),
            "sales.customers.natural".to_owned(),
            "sales.customers.pk".to_owned(),
        ],
        "no `{{customer_id, order_id}}` group survives minimality"
    );
    assert!(manifest.validate().is_ok(), "{:?}", manifest.validate());
}

#[tokio::test]
async fn a_joinsets_derived_keysets_carry_their_members() {
    let fs = joinset_fs().await;
    let (manifest, _warnings) = compile(
        joinset_model(&["natural"], &["pk"]),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("the fixture model must compile");

    let mut on_sales: Vec<(String, Vec<String>)> = roster(&manifest)
        .into_iter()
        .filter(|(source, _, _)| source == "sales")
        .map(|(_, name, members)| (name, members))
        .collect();
    on_sales.sort();
    assert_eq!(
        on_sales,
        vec![
            ("natural".to_owned(), vec!["customer_id".to_owned()]),
            ("pk".to_owned(), vec!["order_id".to_owned()]),
            (
                "sales.customers.natural".to_owned(),
                vec!["customer_id".to_owned()]
            ),
            ("sales.customers.pk".to_owned(), vec!["order_id".to_owned()]),
        ],
        "every group on the composite carries the members it was derived from"
    );
}

fn names_on(manifest: &SemanticManifest, composite: &str) -> Vec<String> {
    roster(manifest)
        .into_iter()
        .filter(|(source, _, _)| source == composite)
        .map(|(_, name, _)| name)
        .collect()
}

#[tokio::test]
async fn a_cross_joinset_derives_nothing_and_rejects_nothing() {
    let crossed = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on_cross()
        .many_to_many()
        .optional(semstrait_model::Optional::None)
        .cross_filter(semstrait_model::CrossFilter::None)
        .build()
        .expect("an `on: cross` relationship builds");
    let fs = joinset_fs().await;
    let (manifest, _warnings) = compile(
        joinset_model_joined(&["natural"], &["pk"], [crossed]),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("a cross Joinset compiles");

    assert_eq!(
        names_on(&manifest, "sales"),
        vec!["natural", "pk"],
        "the author's two claims and no derived name at all"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn a_cross_basis_beside_a_keyed_one_derives_from_the_keyed_one() {
    let crossed = Relationship::builder()
        .name("customers_cross_orders")
        .from("customers")
        .to("orders")
        .on_cross()
        .build()
        .expect("an `on: cross` relationship builds");
    let fs = joinset_fs().await;
    let (manifest, warnings) = compile(
        joinset_model_joined(&["natural"], &["pk"], [crossed, on_customer_id()]),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("a mixed-basis Joinset compiles");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    assert_eq!(
        names_on(&manifest, "sales"),
        vec![
            "natural",
            "pk",
            "sales.customers.natural",
            "sales.customers.pk"
        ],
        "the keyed join derives both determination-rule outputs"
    );
    assert!(manifest.validate().is_ok(), "{:?}", manifest.validate());
}

#[tokio::test]
async fn a_preserved_side_gates_derivation_without_narrowing_the_boundary() {
    let joined = |optional| {
        Relationship::builder()
            .name("orders_to_customers")
            .from("orders")
            .to("customers")
            .on(["customer_id"])
            .many_to_one()
            .optional(optional)
            .build()
            .expect("the fixture relationship must build")
    };
    let fs = joinset_fs().await;

    let (inner, _) = compile(
        joinset_model_joined(
            &["natural"],
            &["pk"],
            [joined(semstrait_model::Optional::None)],
        ),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("an inner join compiles");
    assert_eq!(
        names_on(&inner, "sales"),
        vec![
            "natural",
            "pk",
            "sales.customers.natural",
            "sales.customers.pk"
        ],
        "an inner join derives both determination-rule outputs"
    );

    let (full, _) = compile(
        joinset_model_joined(
            &["natural"],
            &["pk"],
            [joined(semstrait_model::Optional::Both)],
        ),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect("a full outer join compiles");
    assert_eq!(
        names_on(&full, "sales"),
        vec!["natural", "pk"],
        "a full outer join derives nothing, and the author's claims are still admitted"
    );
    assert!(full.validate().is_ok());
}

#[tokio::test]
async fn three_way_composition_recurses_into_the_cumulative_unit() {
    let related = |name: &str, from: &str, to: &str, key: &str| {
        Relationship::builder()
            .name(name)
            .from(from)
            .to(to)
            .on([key])
            .many_to_one()
            .build()
            .expect("the fixture relationship must build")
    };
    let joinset = Joinset::builder("sales")
        .dataset(constituent("orders", &ORDER_NAMES))
        .dataset(constituent("customers", &ORDER_NAMES))
        .dataset(constituent("regions", &ORDER_NAMES))
        .dimension(DimensionSite::r#ref("customer_id").with_key())
        .dimension(DimensionSite::r#ref("order_id"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(related("r1", "orders", "customers", "customer_id"))
        .relationship(related("r2", "customers", "regions", "customer_id"))
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("customer_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("order_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;

    let fs = joinset_fs().await;
    let (manifest, _warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("a three-constituent Joinset compiles");

    assert_eq!(
        names_on(&manifest, "sales"),
        vec!["default", "sales.r1.default"],
        "`sales.r1` is a derived unit, and the outer unit's keyset is qualified by it"
    );
    assert!(manifest.validate().is_ok(), "{:?}", manifest.validate());
}

#[tokio::test]
async fn a_derived_name_colliding_with_an_authored_one_is_refused() {
    let fs = joinset_fs().await;
    let (error, _warnings) = compile(
        joinset_model(&["natural"], &["sales.customers.natural"]),
        None,
        &fs,
        test_provenance(),
    )
    .await
    .expect_err("one name is one entity, so the collision must be refused");
    match error.kind {
        CompileError::DerivedKeysetNameCollision { composite, keyset } => {
            assert_eq!(composite, mint_semantic_source_id("sales"));
            assert_eq!(keyset.as_str(), "sales.customers.natural");
        }
        other => panic!("expected DerivedKeysetNameCollision, got {other:?}"),
    }
}

#[tokio::test]
async fn a_unionset_derives_nothing_and_only_carries_what_was_authored() {
    let manifest = named_reduction().await;

    let names: Vec<String> = roster(&manifest)
        .into_iter()
        .map(|(_, name, _)| name)
        .collect();
    assert_eq!(
        names,
        vec!["natural", "natural", "natural"],
        "one authored name per source, and no derived name anywhere"
    );
    assert!(
        roster(&manifest)
            .iter()
            .filter(|(source, _, _)| source != "orders")
            .all(|(_, _, members)| members == &vec!["customer_id".to_owned()]),
        "`order_ts` is not keyed, so no group invents it"
    );
}

async fn named_reduction() -> SemanticManifest {
    compile_yaml(
        r"semantic_model:
  name: keysets
  dimensions:
    - name: customer_id
      data_type: string
      type: categorical
    - name: order_ts
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders_eu, warehouse.orders_us]
      dimensions:
        - ref: customer_id
          is_key: true
          keysets: [natural]
        - ref: order_ts
",
    )
    .await
}

#[tokio::test]
async fn a_unionsets_carried_keyset_holds_the_authored_member() {
    let manifest = named_reduction().await;
    let on_orders: Vec<_> = roster(&manifest)
        .into_iter()
        .filter(|(source, _, _)| source == "orders")
        .collect();
    assert!(
        !on_orders.is_empty(),
        "the composer's own roster entry exists"
    );
    assert!(
        on_orders
            .iter()
            .all(|(_, _, members)| members == &vec!["customer_id".to_owned()]),
        "the carried keyset holds the member the author keyed"
    );
}

#[tokio::test]
async fn on_auto_resolves_its_basis_from_guaranteed_keysets_only() {
    let joinset = Joinset::builder("sales")
        .dataset(constituent("customers", &ORDER_NAMES))
        .dataset(constituent("orders", &ORDER_NAMES))
        .dimension(DimensionSite::r#ref("customer_id").with_key())
        .dimension(DimensionSite::r#ref("order_id"))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(
            Relationship::builder()
                .name("orders_to_customers")
                .from("orders")
                .to("customers")
                .on_auto()
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("customer_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("order_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;

    let fs = joinset_fs().await;
    let (manifest, _warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("a keyed side gives `on: auto` a basis to derive");

    let sales = manifest
        .semantic_sources
        .values()
        .find(|entry| entry.name.as_str() == "sales")
        .expect("the authored Joinset is an entry");
    let semstrait_manifest::RoleDefinition::Joinset {
        on_expr_id: Some(on_expr_id),
        ..
    } = sales.role_definition
    else {
        panic!("the authored Joinset carries a keyed Joinset body");
    };
    let rendered = format!("{:?}", stored_tree(&manifest, on_expr_id));
    assert!(
        rendered.contains("customer_id"),
        "the derived basis is the keyset's member: {rendered}"
    );
    assert!(
        !rendered.contains("order_id"),
        "an unkeyed Dimension both sides serve is NOT a derived basis any more: {rendered}"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async { vec![keyed_measure_refusal().await] }).await;
}
