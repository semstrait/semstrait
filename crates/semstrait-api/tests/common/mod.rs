#![allow(dead_code, reason = "each test file uses only some of these helpers")]

use std::sync::Arc;

use semstrait_api::{Semstrait, SemstraitBuilder, SemstraitError};
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::Diagnostics;
use semstrait_manifest::SemanticManifest;
use semstrait_model::{parse, ModelBuildErrorKind, ParseErrorKind, SemanticModel};
use tempfile::TempDir;

pub const SALES: &str = include_str!("../fixtures/sales.yaml");

pub const SPEND_BOOKINGS: &str = include_str!("../fixtures/spend_bookings.yaml");

pub fn semstrait() -> Semstrait {
    SemstraitBuilder::default()
        .with_filesystem(Arc::new(InMemoryFileSystem::new()))
        .build()
}

pub fn sales_model() -> SemanticModel {
    let (model, _) = parse(SALES)
        .expect("the model parses")
        .build()
        .expect("the model builds");
    model
}

pub async fn compiled(yaml: &str) -> SemanticManifest {
    let (manifest, _) = semstrait()
        .compile_from_yaml(yaml)
        .await
        .expect("the model compiles");
    manifest
}

pub async fn sales() -> SemanticManifest {
    compiled(SALES).await
}

pub fn kinds(diagnostics: Diagnostics<SemstraitError>) -> Vec<SemstraitError> {
    diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.kind)
        .collect()
}

pub fn only_retired_keys(warnings: Diagnostics<SemstraitError>) -> bool {
    let retired = ModelBuildErrorKind::Parse(ParseErrorKind::DataKindKeysBlockRetired {
        data_kind: "orders".into(),
    });
    matches!(kinds(warnings).as_slice(), [SemstraitError::Build(kind)] if *kind == retired)
}

pub fn written(dir: &TempDir, name: &str, contents: impl AsRef<[u8]>) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, contents).expect("the file is written");
    path.display().to_string()
}
