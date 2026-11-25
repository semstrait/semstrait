#[derive(schemars::JsonSchema)]
#[cfg_attr(
    feature = "schemars",
    schemars(description = "A semstrait model document: the `semantic_model:` key and its body.")
)]
#[serde(deny_unknown_fields)]
struct SemanticModelDocument {
    #[allow(dead_code)]
    semantic_model: crate::model::SemanticModel,
}

pub fn semantic_model_json_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(SemanticModelDocument))
        .expect("schema serializes to JSON")
}

pub fn catalogs_json_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(crate::catalog::CatalogsConfig))
        .expect("schema serializes to JSON")
}
