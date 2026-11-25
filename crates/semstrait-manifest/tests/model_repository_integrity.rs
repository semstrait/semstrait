use std::sync::Arc;

use semstrait_common::io::{
    FileSystemRepository, InMemoryFileSystem, Path, Repository, RepositoryErrorKind,
};
use semstrait_model::SemanticModel;

const MODEL_YAML: &str = r#"
semantic_model:
  name: analytics-v1
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#;

fn built_model() -> SemanticModel {
    semstrait_model::parse(MODEL_YAML)
        .expect("parse ok")
        .build()
        .expect("build ok")
        .0
}

fn repo() -> FileSystemRepository<SemanticModel> {
    FileSystemRepository::new(Arc::new(InMemoryFileSystem::new()), Path::new("mem:"))
}

#[tokio::test]
async fn saving_a_model_whose_root_id_was_cleared_after_build_is_refused_not_a_panic() {
    let mut model = built_model();
    model.id = None;

    let err = repo()
        .save(&model)
        .await
        .expect_err("a model with no root id must be refused");

    assert!(
        matches!(err, RepositoryErrorKind::IntegrityCheckFailed { .. }),
        "expected IntegrityCheckFailed, got {err:?}"
    );
}

#[tokio::test]
async fn saving_a_well_formed_model_still_succeeds() {
    let model = built_model();

    repo().save(&model).await.expect("well-formed model saves");
}
