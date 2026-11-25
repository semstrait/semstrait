use std::sync::Arc;

use semstrait_common::diagnostic::{Diagnostics, SourceId};
use semstrait_common::io::{FileSystem, FileSystemErrorKind, InMemoryFileSystem, Path};
use semstrait_model::{ModelBuildErrorKind, SemanticModel};

const MODEL_YAML: &str = r#"
semantic_model:
  name: remote-model
  dimensions:
    - name: country
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: country
"#;

const BAD_IDENT_YAML: &str = r#"
semantic_model:
  name: bad-ident-remote
  dimensions:
    - name: bad-name
      data_type: string
      type: categorical
"#;

#[derive(Debug)]
enum LoadFailure {
    Transport(FileSystemErrorKind),
    Parse(Diagnostics<ModelBuildErrorKind>),
}

async fn load_from(fs: &dyn FileSystem, path: &Path) -> Result<SemanticModel, LoadFailure> {
    let bytes = fs.read(path).await.map_err(LoadFailure::Transport)?;
    let (model, _warnings) = SemanticModel::loader()
        .from_yaml_bytes(&bytes, SourceId::named(path.as_str()))
        .build()
        .map_err(LoadFailure::Parse)?;
    Ok(model)
}

async fn store_with(key: &str, yaml: &str) -> (InMemoryFileSystem, Path) {
    let fs = InMemoryFileSystem::default();
    let path = Path::parse(key).expect("fixture key must be a valid path");
    fs.write(&path, yaml.as_bytes().to_vec().into())
        .await
        .expect("seed write");
    (fs, path)
}

#[tokio::test]
async fn model_loads_from_a_remote_store() {
    let (fs, path) = store_with("s3://bucket/models/remote-model.yaml", MODEL_YAML).await;

    let model = load_from(&fs, &path).await.expect("remote load");

    assert_eq!(model.name, "remote-model");
    assert_eq!(model.dimensions.len(), 1);
    assert_eq!(model.datasets.len(), 1);
}

#[tokio::test]
async fn remote_load_equals_direct_parse() {
    let (fs, path) = store_with("s3://bucket/models/remote-model.yaml", MODEL_YAML).await;

    let remote = load_from(&fs, &path).await.expect("remote load");
    let (direct, _) = SemanticModel::loader()
        .from_yaml_bytes(MODEL_YAML.as_bytes(), SourceId::named(path.as_str()))
        .build()
        .expect("direct build");

    assert_eq!(remote, direct);
}

#[tokio::test]
async fn transport_returns_the_stored_bytes_verbatim() {
    let (fs, path) = store_with("s3://bucket/models/remote-model.yaml", MODEL_YAML).await;

    let fetched = fs.read(&path).await.expect("read back");

    assert_eq!(fetched.as_ref(), MODEL_YAML.as_bytes());
}

#[tokio::test]
async fn parse_diagnostics_are_attributed_to_the_object_key() {
    let key = "s3://bucket/models/bad-ident-remote.yaml";
    let (fs, path) = store_with(key, BAD_IDENT_YAML).await;

    let Err(LoadFailure::Parse(diags)) = load_from(&fs, &path).await else {
        panic!("an SR-8-invalid identifier must fail the parse, not the transport");
    };

    let sources: Vec<&str> = diags
        .iter()
        .filter_map(|d| d.location.as_ref())
        .map(|loc| loc.source.as_str())
        .collect();
    assert!(
        sources.contains(&key),
        "expected a diagnostic located at {key}, got {sources:?}"
    );
}

#[tokio::test]
async fn absent_object_fails_in_transport_not_parse() {
    let fs = InMemoryFileSystem::default();
    let path = Path::parse("s3://bucket/models/absent.yaml").expect("valid key");

    let err = load_from(&fs, &path).await.expect_err("nothing is stored");

    assert!(
        matches!(
            err,
            LoadFailure::Transport(FileSystemErrorKind::NotFound { .. })
        ),
        "expected a transport NotFound, got {err:?}"
    );
}

#[tokio::test]
async fn one_caller_body_serves_any_backend() {
    let (fs, path) = store_with("models/remote-model.yaml", MODEL_YAML).await;
    let erased: Arc<dyn FileSystem> = Arc::new(fs);

    let model = load_from(erased.as_ref(), &path)
        .await
        .expect("load through a trait object");

    assert_eq!(model.name, "remote-model");
}
