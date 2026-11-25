use semstrait_model::{parse, SemanticModel};

const REFERENCE_YAML: &str = include_str!("../schemas/reference.yaml");

fn reference_model() -> SemanticModel {
    let (model, _warnings) = parse(REFERENCE_YAML)
        .expect("reference.yaml parses")
        .build()
        .expect("reference.yaml builds");
    model
}

#[test]
fn semantic_model_round_trips_through_json() {
    let model = reference_model();
    let json = serde_json::to_string_pretty(&model).expect("serialize to JSON");
    let back: SemanticModel = serde_json::from_str(&json).expect("deserialize from JSON");
    assert_eq!(model, back, "JSON round-trip changed the model");
}

#[test]
fn semantic_model_round_trips_through_yaml() {
    let model = reference_model();
    let yaml = serde_yaml::to_string(&model).expect("serialize to YAML");
    let back: SemanticModel = serde_yaml::from_str(&yaml).expect("deserialize from YAML");
    assert_eq!(model, back, "YAML round-trip changed the model");
}

#[test]
fn convenience_minted_model_is_byte_stable_across_reserialize() {
    let model = reference_model();
    let first = serde_json::to_string(&model).expect("serialize once");
    let back: SemanticModel = serde_json::from_str(&first).expect("deserialize");
    let second = serde_json::to_string(&back).expect("serialize twice");
    assert_eq!(first, second, "reserialize was not byte-identical");
}

#[test]
fn json_and_yaml_dumps_carry_no_serde_tags() {
    let model = reference_model();
    let yaml = serde_yaml::to_string(&model).expect("serialize to YAML");
    assert!(!yaml.contains('!'), "YAML dump leaked a `!`-tag:\n{yaml}");
}
