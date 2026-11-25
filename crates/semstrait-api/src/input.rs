use semstrait_manifest::SemanticManifest;

#[non_exhaustive]
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "an Input is moved once into load, which takes the manifest by value"
)]
pub enum Input {
    ModelYaml(String),
    ModelPath(String),
    Manifest(SemanticManifest),
    ManifestPath(String),
}
