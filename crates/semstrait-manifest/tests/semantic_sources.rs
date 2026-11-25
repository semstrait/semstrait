use semstrait_common::io::InMemoryFileSystem;
use semstrait_ir::MultisetOp;
use semstrait_manifest::{
    compile, mint_semantic_source_id, CompileError, RoleDefinition, SemanticManifest, SourceAlias,
};
use semstrait_model::parse;

mod common;
use common::{by_name, model_from_yaml, test_provenance};

async fn compiled(yaml: &str) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, warnings) = compile(model_from_yaml(yaml), None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

const AUTHORED_UNIONSET: &str = r"semantic_model:
  name: stacking
  dimensions:
    - name: region
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
  unionsets:
    - name: all_orders
      mode: all
      dimensions:
        - ref: region
      measures:
        - ref: revenue
      datasets:
        - name: eu
          extras:
            semantic_mapping: {region: region, revenue: revenue}
            storage:
              tables: [warehouse.orders_eu]
        - name: us
          extras:
            semantic_mapping: {region: region, revenue: revenue}
            storage:
              tables: [warehouse.orders_us]
";

#[tokio::test]
async fn an_authored_unionset_compiles_to_three_ordinary_entries() {
    let manifest = compiled(AUTHORED_UNIONSET).await;

    assert_eq!(
        manifest.semantic_sources.len(),
        3,
        "the composer and both branches are ordinary entries"
    );

    let composer = by_name(&manifest, "all_orders");
    let RoleDefinition::Unionset {
        branches,
        mode,
        null_fill,
    } = &composer.role_definition
    else {
        panic!("an authored Unionset carries the Unionset arm");
    };
    assert_eq!(*mode, MultisetOp::UnionAll, "`mode` reaches the artifact");
    assert!(
        *null_fill,
        "an authored (explicit) Unionset NULL-fills (ADR 0077)"
    );
    let stored = serde_json::to_value(&composer.role_definition).expect("serialize Unionset");
    assert_eq!(
        stored["content"]["mode"], "all",
        "the manifest's own `MultisetOp` still serializes the `mode` key as `all`/`unique`: {stored}"
    );

    let expected: Vec<String> = ["all_orders.eu", "all_orders.us"]
        .into_iter()
        .map(mint_semantic_source_id)
        .collect();
    assert_eq!(branches, &expected, "branches are entry ids in name order");
    for branch in branches {
        assert!(
            manifest.semantic_sources.contains_key(branch),
            "every operand resolves in `semantic_sources`"
        );
    }

    assert_eq!(manifest.sources.len(), 2);
    for name in ["all_orders.eu", "all_orders.us"] {
        let RoleDefinition::Dataset { source_id, .. } = &by_name(&manifest, name).role_definition
        else {
            panic!("a branch is a leaf `Dataset`");
        };
        assert!(
            manifest.sources.contains_key(source_id),
            "the `Dataset` arm's single reference resolves in `physical_sources`"
        );
        assert!(
            !manifest.semantic_sources.contains_key(source_id),
            "and it is NOT an id into `semantic_sources` — the one arm asymmetry"
        );
    }

    let served = |name: &str| by_name(&manifest, name).semantic_bitmask.clone();
    assert_eq!(
        composer.semantic_bitmask,
        served("all_orders.eu").union(&served("all_orders.us"))
    );
    assert!(
        composer.semantic_bitmask.count_ones() == 2,
        "both root-pool semantics are servable from the stack"
    );

    assert!(
        manifest.validate().is_ok(),
        "a compiled manifest must satisfy its own validator"
    );
}

const SPLIT_ORDERS_DATASET: &str = r"  datasets:
    - name: split_orders
      dimensions:
        - ref: region
      measures:
        - ref: revenue
      extras:
        semantic_mapping: {region: region, revenue: revenue}
        storage:
          tables: [warehouse.split_orders_1, warehouse.split_orders_2]
";

#[tokio::test]
async fn the_null_fill_setting_reads_from_the_serialized_manifest() {
    let manifest = compiled(&format!("{AUTHORED_UNIONSET}{SPLIT_ORDERS_DATASET}")).await;
    assert!(
        manifest.validate().is_ok(),
        "a compiled manifest must satisfy its own validator"
    );

    let json = serde_json::to_value(&manifest).expect("the manifest serializes");
    let null_fill_of = |source_name: &str| {
        let id = &by_name(&manifest, source_name).id;
        json["semantic_sources"][id.as_str()]["role_definition"]["content"]["null_fill"]
            .as_bool()
            .unwrap_or_else(|| {
                panic!("`{source_name}`'s serialized role_definition carries a `null_fill` field")
            })
    };
    assert!(
        null_fill_of("all_orders"),
        "the authored Unionset's serialized field is `true`"
    );
    assert!(
        !null_fill_of("split_orders"),
        "the Implicit Unionset's serialized field is `false`"
    );
}

#[tokio::test]
async fn a_nested_unit_is_private_and_names_its_originator() {
    let manifest = compiled(AUTHORED_UNIONSET).await;

    let composer = by_name(&manifest, "all_orders");
    assert!(
        composer.public,
        "an authored top-level unit is a query interface"
    );
    assert!(
        composer.originators.is_empty(),
        "empty means authored top-level, never `unknown`"
    );
    assert_eq!(
        composer.id,
        mint_semantic_source_id("all_orders"),
        "a unit's id is minted from its own name"
    );

    for name in ["all_orders.eu", "all_orders.us"] {
        let branch = by_name(&manifest, name);
        assert!(!branch.public, "an authored nested unit is private");
        assert_eq!(
            branch.originators,
            vec![composer.id.clone()],
            "the nearest authored ancestor, and exactly one of them"
        );
        assert_eq!(
            branch.id,
            mint_semantic_source_id(name),
            "a nested unit's id seeds on `<parent>.<name>`"
        );
    }
}

#[tokio::test]
async fn validate_refuses_an_alias_its_source_role_and_index_do_not_give() {
    let manifest = compiled(AUTHORED_UNIONSET).await;
    assert!(manifest.validate().is_ok(), "the unedited manifest passes");
    let json = serde_json::to_string(&manifest).expect("serialize the manifest");
    let aliases: Vec<(&str, &str)> = manifest
        .semantic_sources
        .values()
        .map(|entry| (entry.name.as_str(), entry.alias.as_str()))
        .inspect(|(_, alias)| {
            let field = format!("\"alias\":\"{alias}\"");
            assert_eq!(json.matches(&field).count(), 1, "`{alias}` is one entry's");
            let key = format!("\"{alias}\":");
            assert!(
                !json.contains(&key),
                "`{alias}` keys no map, so no reverse index"
            );
        })
        .collect();
    assert_eq!(
        aliases,
        vec![
            ("all_orders.eu", "ssd0"),
            ("all_orders.us", "ssd1"),
            ("all_orders", "ssu2"),
        ],
        "in `EntityId` order"
    );

    let edited = by_name(&manifest, "all_orders.us");
    for (found, case) in [
        ("ssd0", "a repeated alias"),
        ("ssd.1", "an alias with a dot"),
        ("SSD1", "an alias that differs only in case"),
        ("", "an empty alias"),
    ] {
        let mut decoded = manifest.clone();
        if let Some(entry) = decoded.semantic_sources.get_mut(&edited.id) {
            entry.alias = SourceAlias::from(found);
        }
        let faults: Vec<CompileError> = decoded
            .validate()
            .expect_err(case)
            .into_iter()
            .map(|fault| fault.kind)
            .collect();
        assert_eq!(
            faults,
            vec![CompileError::SourceAliasMismatch {
                semantic_source: edited.id.clone(),
                expected: SourceAlias::from("ssd1"),
                found: SourceAlias::from(found),
            }],
            "{case}"
        );
        assert_eq!(
            faults[0].to_string(),
            format!(
                "semantic source `{}` carries the alias `{found}`; its source role and \
                 index give `ssd1`",
                edited.id
            ),
            "ADR 0078's message"
        );
    }
}

#[tokio::test]
async fn two_top_level_datasets_are_two_entries() {
    let manifest = compiled(
        r"semantic_model:
  name: two_leaves
  dimensions:
    - name: region
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
  datasets:
    - name: orders
      dimensions:
        - ref: region
      measures:
        - ref: revenue
      extras:
        storage:
          tables: [warehouse.orders]
    - name: returns
      dimensions:
        - ref: region
      extras:
        storage:
          tables: [warehouse.returns]
",
    )
    .await;

    assert_eq!(manifest.semantic_sources.len(), 2);
    for name in ["orders", "returns"] {
        let entry = by_name(&manifest, name);
        assert!(entry.public);
        assert!(entry.originators.is_empty());
        assert!(matches!(
            entry.role_definition,
            RoleDefinition::Dataset { .. }
        ));
    }

    assert_eq!(
        by_name(&manifest, "orders").semantic_bitmask.count_ones(),
        2
    );
    assert_eq!(
        by_name(&manifest, "returns").semantic_bitmask.count_ones(),
        1
    );
    assert!(manifest.validate().is_ok());
}

#[test]
fn two_sites_for_one_name_in_one_collection_are_reported() {
    let yaml = r"semantic_model:
  name: duplicated
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: orders
      dimensions:
        - ref: region
        - ref: region
      extras:
        storage:
          tables: [warehouse.orders]
";
    let diagnostics = parse(yaml)
        .expect("the document parses")
        .build()
        .expect_err("one name twice in one collection must be reported");
    let rendered = format!("{diagnostics:?}");
    assert!(
        rendered.contains("InterfaceMemberPathNotUnique"),
        "expected the ADR 0060 diagnostic, got {rendered}"
    );
    assert!(
        rendered.contains("region") && rendered.contains("dimensions"),
        "the diagnostic names the collection and the name: {rendered}"
    );
}
