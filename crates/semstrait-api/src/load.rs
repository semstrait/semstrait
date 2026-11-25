use semstrait_common::io::Persistable;
use semstrait_manifest::SemanticManifest;

use crate::error::{beside, InputErrorKind, Outcome, SemstraitError};
use crate::input::Input;
use crate::orchestrator::{built, validated, Semstrait};
use crate::reader::{at, read};

pub async fn load(semstrait: &Semstrait, input: Input) -> Outcome<SemanticManifest> {
    match input {
        Input::ModelYaml(yaml) => stored_or_compiled(semstrait, &yaml).await,
        Input::ModelPath(path) => stored_or_compiled(semstrait, &read(&path).await?).await,
        Input::Manifest(manifest) => validated(manifest),
        Input::ManifestPath(path) => {
            let json = read(&path).await?;
            let decoded = SemanticManifest::from_bytes(json.as_bytes()).map_err(|reason| {
                let undecodable = InputErrorKind::UndecodableManifest { reason };
                at(&path, SemstraitError::Input(undecodable))
            })?;
            validated(decoded)
        }
    }
}

async fn stored_or_compiled(semstrait: &Semstrait, yaml: &str) -> Outcome<SemanticManifest> {
    let (model, warnings) = built(yaml)?;
    let asked = semstrait.exists(&model).await.map(|hit| (hit, Vec::new()));
    let (hit, warnings) = beside(warnings, asked)?;
    if hit {
        return beside(warnings, semstrait.load_manifest(&model).await);
    }
    let (manifest, warnings) = semstrait.compiled(model, warnings).await?;
    let saved = semstrait.save_manifest(&manifest).await;
    beside(warnings, saved.map(|()| (manifest, Vec::new())))
}
