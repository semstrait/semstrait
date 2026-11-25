use std::time::{Duration, SystemTime, UNIX_EPOCH};

use semstrait_common::diagnostic::SourceId;
use semstrait_common::io::{Fingerprintable, InMemoryFileSystem};
use semstrait_manifest::{compile, ManifestProvenance, SemanticManifest};
use semstrait_model::SemanticModel;

const MODEL_YAML: &str = r#"
semantic_model:
  name: sales
  dimensions:
    - name: region
      data_type: string
      type: categorical
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
"#;

fn first_compiled_at() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_700_000_000)
}

fn last_written_at() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_800_000_000)
}

fn parse(yaml: &str) -> SemanticModel {
    let (model, _warnings) = SemanticModel::loader()
        .from_yaml_bytes(yaml.as_bytes(), SourceId::named("lifecycle-fixture.yaml"))
        .build()
        .expect("the fixture document must parse and validate clean");
    model
}

async fn compile_with(provenance: ManifestProvenance) -> SemanticManifest {
    let fs = InMemoryFileSystem::new();
    let (manifest, _warnings) = compile(parse(MODEL_YAML), None, &fs, provenance)
        .await
        .expect("the fixture document must compile clean");
    manifest
}

#[tokio::test]
async fn both_provenance_instants_are_the_callers_own() {
    let manifest = compile_with(ManifestProvenance::new(
        first_compiled_at(),
        last_written_at(),
    ))
    .await;

    assert_eq!(manifest.created_at, first_compiled_at());
    assert_eq!(manifest.updated_at, last_written_at());
}

#[tokio::test]
async fn the_artifact_carries_the_models_own_fingerprint() {
    let model = parse(MODEL_YAML);
    let expected = model.fingerprint();

    let manifest = compile_with(ManifestProvenance::new(
        first_compiled_at(),
        last_written_at(),
    ))
    .await;

    assert_eq!(manifest.model_hash, expected);
}
