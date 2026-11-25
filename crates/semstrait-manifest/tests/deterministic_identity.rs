use semstrait_common::diagnostic::SourceId;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_manifest::{compile, SemanticManifest};
use semstrait_model::SemanticModel;

mod common;
use common::test_provenance;

const MODEL_YAML: &str = r#"
semantic_model:
  name: sales
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
      extras:
        storage:
          tables:
            - warehouse.fact_rows
        semantic_mapping: auto
      dimensions:
        - ref: region
          is_key: true
      measures:
        - ref: revenue
"#;

fn parse(yaml: &str) -> SemanticModel {
    let (model, _warnings) = SemanticModel::loader()
        .from_yaml_bytes(yaml.as_bytes(), SourceId::named("identity-fixture.yaml"))
        .build()
        .expect("the fixture document must parse and validate clean");
    model
}

async fn compile_yaml(yaml: &str) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, _warnings) = compile(parse(yaml), None, &fs, test_provenance())
        .await
        .expect("the fixture document must compile clean");
    manifest
}

fn bit_positions(manifest: &SemanticManifest) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = manifest
        .semantic_definitions
        .entries
        .values()
        .map(|def| (def.name.as_str().to_owned(), def.bit_position))
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn one_document_parsed_twice_yields_one_manifest() {
    let first = compile_yaml(MODEL_YAML).await;
    let second = compile_yaml(MODEL_YAML).await;

    assert_eq!(
        first.model_hash, second.model_hash,
        "one document must produce one model_hash (ADR 0041's content-addressed gate)"
    );
    assert_eq!(
        first.model_id, second.model_id,
        "model_id is UUIDv5 over model_hash, so it moves only when the hash does"
    );
    assert_eq!(
        bit_positions(&first),
        bit_positions(&second),
        "one document must produce one bit-position assignment (ADR 0036 rule 1)"
    );
    assert_eq!(
        first.sources, second.sources,
        "every minted entity id must reproduce, not just the top-level three"
    );
    assert_eq!(first.semantic_signatures, second.semantic_signatures);
    assert_eq!(first.keysets, second.keysets);
    assert_eq!(first.semantic_sources, second.semantic_sources);
}

#[tokio::test]
async fn renaming_a_data_kind_re_keys_its_semantic_source() {
    let before = compile_yaml(MODEL_YAML).await;
    let after = compile_yaml(&MODEL_YAML.replace("name: orders", "name: purchases")).await;

    let name_of = |m: &semstrait_manifest::SemanticManifest| -> Vec<String> {
        m.semantic_sources
            .values()
            .map(|entry| entry.name.as_str().to_owned())
            .collect()
    };
    assert_eq!(name_of(&before), vec!["orders".to_owned()]);
    assert_eq!(name_of(&after), vec!["purchases".to_owned()]);
    assert_ne!(
        before.semantic_sources.keys().collect::<Vec<_>>(),
        after.semantic_sources.keys().collect::<Vec<_>>(),
        "the name seeds the id, so a rename re-keys the entry (ADR 0045)"
    );
}

#[tokio::test]
async fn renaming_an_entity_moves_the_artifact() {
    let renamed = MODEL_YAML.replace("name: region", "name: territory");
    let before = compile_yaml(MODEL_YAML).await;
    let after = compile_yaml(&renamed.replace("ref: region", "ref: territory")).await;

    assert_ne!(
        before.model_hash, after.model_hash,
        "a rename is content, so it moves the fingerprint (ADR 0041)"
    );
    assert_ne!(
        before.model_id, after.model_id,
        "model_id is UUIDv5 over model_hash"
    );
}
