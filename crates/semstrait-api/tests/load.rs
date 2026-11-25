mod common;

use semstrait_api::{load, Input, InputErrorKind, SemstraitError};
use semstrait_common::io::{Persistable, RepositoryErrorKind};
use semstrait_common::{FileSystemErrorKind, IoErrorKind};
use semstrait_manifest::{CompileError, ReferenceSite, SemanticManifest};
use tempfile::tempdir;

use common::{only_retired_keys, sales, sales_model, semstrait, written, SALES};

async fn refusal_of(input: Input) -> (SemstraitError, Option<String>) {
    let (diagnostic, _) = load(&semstrait(), input)
        .await
        .expect_err("the input is refused");
    let at = diagnostic
        .location
        .map(|location| location.source.to_string());
    (diagnostic.kind, at)
}

fn rekeyed(mut manifest: SemanticManifest) -> SemanticManifest {
    let (_, source) = manifest
        .sources
        .pop_first()
        .expect("the manifest holds a source");
    manifest.sources.insert("rekeyed".into(), source);
    manifest
}

fn is_the_rekeyed_source(kind: &SemstraitError) -> bool {
    matches!(
        kind,
        SemstraitError::Compile(CompileError::StructuralViolation {
            site: ReferenceSite::Source(id),
            rule: "entity id must equal its map key",
        }) if id == "rekeyed"
    )
}

#[tokio::test]
async fn each_input_of_one_semantic_model_gives_the_same_manifest() {
    let (semstrait, dir) = (semstrait(), tempdir().expect("a temporary directory"));
    let model_path = written(&dir, "sales.yaml", SALES);
    let (manifest, _) = load(&semstrait, Input::ModelPath(model_path))
        .await
        .expect("the model file compiles");
    let manifest_path = written(&dir, "sales.json", manifest.to_bytes().expect("it encodes"));

    for input in [
        Input::ModelYaml(SALES.into()),
        Input::Manifest(manifest.clone()),
        Input::ManifestPath(manifest_path),
    ] {
        let (loaded, _) = load(&semstrait, input).await.expect("the input loads");

        assert_eq!(loaded, manifest);
    }
}

#[tokio::test]
async fn save_manifest_then_load_manifest_gives_back_the_same_manifest() {
    let (semstrait, manifest) = (semstrait(), sales().await);

    semstrait
        .save_manifest(&manifest)
        .await
        .expect("the manifest is stored");
    let (loaded, warnings) = semstrait
        .load_manifest(&sales_model())
        .await
        .expect("the stored manifest loads");

    assert_eq!(loaded, manifest);
    assert!(warnings.is_empty());
}

#[tokio::test]
async fn load_manifest_refuses_a_stored_manifest_that_validate_refuses_with_its_compile_error() {
    let semstrait = semstrait();
    semstrait
        .save_manifest(&rekeyed(sales().await))
        .await
        .expect("the manifest is stored");

    let refusal = semstrait.load_manifest(&sales_model()).await;

    assert!(matches!(refusal, Err((diagnostic, _)) if is_the_rekeyed_source(&diagnostic.kind)));
}

#[tokio::test]
async fn a_second_load_of_one_model_yaml_is_a_store_hit_that_returns_the_stored_manifest() {
    let (semstrait, model) = (semstrait(), sales_model());
    assert!(!semstrait.exists(&model).await.expect("the store answers"));

    let (first, first_warnings) = load(&semstrait, Input::ModelYaml(SALES.into()))
        .await
        .expect("the model compiles");
    assert!(semstrait.exists(&model).await.expect("the store answers"));
    let (second, second_warnings) = load(&semstrait, Input::ModelYaml(SALES.into()))
        .await
        .expect("the stored manifest loads");

    assert_eq!(second, first);
    assert!(only_retired_keys(first_warnings) && only_retired_keys(second_warnings));
}

#[tokio::test]
async fn load_refuses_a_manifest_or_a_manifest_file_that_validate_refuses_with_its_compile_error() {
    let (dir, manifest) = (
        tempdir().expect("a temporary directory"),
        rekeyed(sales().await),
    );
    let path = written(
        &dir,
        "sales.json",
        manifest.to_bytes().expect("the manifest encodes"),
    );

    for input in [Input::Manifest(manifest), Input::ManifestPath(path)] {
        let (kind, _) = refusal_of(input).await;

        assert!(is_the_rekeyed_source(&kind));
    }
}

#[tokio::test]
async fn a_path_input_that_does_not_load_gives_the_kind_of_its_failure_at_its_path() {
    let dir = tempdir().expect("a temporary directory");
    let inputs = [
        Input::ModelPath(dir.path().join("absent.yaml").display().to_string()),
        Input::ModelPath(written(&dir, "latin1.yaml", [0xff])),
        Input::ManifestPath(written(&dir, "sales.json", SALES)),
    ];

    let mut kinds = Vec::new();
    for (input, file) in inputs
        .into_iter()
        .zip(["absent.yaml", "latin1.yaml", "sales.json"])
    {
        let (kind, at) = refusal_of(input).await;
        assert_eq!(at, Some(dir.path().join(file).display().to_string()));
        kinds.push(kind);
    }

    assert!(
        matches!(
            kinds.as_slice(),
            [
                SemstraitError::FileSystem(FileSystemErrorKind::NotFound { .. }),
                SemstraitError::Io(IoErrorKind::Malformed { .. }),
                SemstraitError::Input(InputErrorKind::UndecodableManifest { .. }),
            ]
        ),
        "{kinds:?}"
    );
}

#[cfg(not(feature = "aws"))]
#[tokio::test]
async fn an_s3_path_without_the_aws_feature_gives_the_input_error_at_its_path() {
    let uri = "s3://lake/models/sales.yaml";

    for input in [
        Input::ModelPath(uri.into()),
        Input::ManifestPath(uri.into()),
    ] {
        let (kind, at) = refusal_of(input).await;

        assert!(matches!(
            kind,
            SemstraitError::Input(InputErrorKind::S3WithoutAws)
        ));
        assert_eq!(at.as_deref(), Some(uri));
    }
}

#[tokio::test]
async fn a_bare_file_name_is_read_from_the_working_directory() {
    let (kind, _) = refusal_of(Input::ManifestPath("Cargo.toml".into())).await;

    assert!(matches!(
        kind,
        SemstraitError::Input(InputErrorKind::UndecodableManifest { .. })
    ));
}

#[tokio::test]
async fn load_manifest_of_a_semantic_model_with_no_stored_manifest_gives_the_repository_error() {
    let refusal = semstrait().load_manifest(&sales_model()).await;

    assert!(matches!(
        refusal,
        Err((diagnostic, _))
            if matches!(diagnostic.kind, SemstraitError::Repository(RepositoryErrorKind::Io(IoErrorKind::NotFound { .. })))
    ));
}

const ONE_CHILD_UNION: &str = r#"
semantic_model:
  name: one_child
  dimensions:
    - name: region
      data_type: string
      type: categorical
  unionsets:
    - name: ads
      mode: all
      dimensions: [{ref: region}]
      datasets:
        - name: only
          extras:
            storage: {tables: [warehouse.only]}
            semantic_mapping: auto
"#;

#[tokio::test]
async fn a_saved_model_with_a_one_child_composition_is_found_by_exists_and_load_manifest() {
    let semstrait = semstrait();
    let (model, _) = semstrait_model::parse(ONE_CHILD_UNION)
        .expect("the model parses")
        .build()
        .expect("the model builds");
    let (manifest, _) = semstrait
        .compile_from_model(model.clone())
        .await
        .expect("the model compiles");
    semstrait
        .save_manifest(&manifest)
        .await
        .expect("the manifest is stored");

    assert!(semstrait.exists(&model).await.expect("the store answers"));
    let (loaded, _) = semstrait
        .load_manifest(&model)
        .await
        .expect("the stored manifest loads");
    assert_eq!(loaded, manifest);
}
