use semstrait_model::{parse, parse_catalogs};

const REFERENCE_YAML: &str = include_str!("../schemas/reference.yaml");
const CATALOGS_YAML: &str = include_str!("../schemas/catalogs_reference.yaml");

#[cfg(feature = "schemars")]
fn yaml_to_json(yaml: &str) -> serde_json::Value {
    serde_yaml::from_str(yaml).expect("yaml must parse as a json-compatible value")
}

#[cfg(feature = "schemars")]
fn validate_against_schema(schema: &serde_json::Value, instance: &serde_json::Value) {
    let validator = jsonschema::validator_for(schema).expect("generated schema must compile");
    let messages: Vec<String> = validator
        .iter_errors(instance)
        .map(|err| format!("schema error: {err} (at {})", err.instance_path()))
        .collect();
    if !messages.is_empty() {
        for m in &messages {
            eprintln!("{m}");
        }
        panic!(
            "instance does not validate against the GENERATED schema ({} errors)",
            messages.len()
        );
    }
}

#[cfg(feature = "schemars")]
#[test]
fn reference_yaml_validates_against_generated_schema() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let instance = yaml_to_json(REFERENCE_YAML);
    validate_against_schema(&schema, &instance);
}

#[test]
fn reference_yaml_parses_and_validates_clean() {
    let builder = match parse(REFERENCE_YAML) {
        Ok(b) => b,
        Err(diags) => {
            for d in &diags {
                eprintln!("parse: {:?}", d);
            }
            panic!("reference.yaml must parse clean");
        }
    };

    match builder.build() {
        Ok((_model, warnings)) => {
            assert!(
                warnings.is_empty(),
                "no warnings expected, got {:?}",
                warnings
            );
        }
        Err(diags) => {
            for d in &diags {
                eprintln!("build: {:?}", d);
            }
            panic!("reference.yaml must validate clean");
        }
    }
}

#[cfg(feature = "schemars")]
#[test]
fn catalogs_reference_yaml_validates_against_generated_schema() {
    let schema = semstrait_model::schema::catalogs_json_schema();
    let instance = yaml_to_json(CATALOGS_YAML);
    validate_against_schema(&schema, &instance);
}

#[cfg(feature = "schemars")]
#[test]
fn generated_catalogs_schema_rejects_invalid_documents() {
    let schema = semstrait_model::schema::catalogs_json_schema();
    let validator =
        jsonschema::validator_for(&schema).expect("generated catalogs schema must compile");

    let missing_required = serde_json::json!({
        "catalogs": { "c": { "type": "polaris" } }
    });
    assert!(
        !validator.is_valid(&missing_required),
        "schema must require a catalog entry's `name`/`url`/`auth`"
    );

    let wrong_type = serde_json::json!({
        "catalogs": { "c": {
            "type": "polaris",
            "name": "w",
            "url": 123,
            "auth": { "method": "bearer", "token": "t" }
        } }
    });
    assert!(
        !validator.is_valid(&wrong_type),
        "schema must type-check a catalog entry's `url` as a string"
    );

    let bad_auth = serde_json::json!({
        "catalogs": { "c": {
            "type": "polaris",
            "name": "w",
            "url": "https://example.com",
            "auth": { "method": "no_such_method" }
        } }
    });
    assert!(
        !validator.is_valid(&bad_auth),
        "schema must reject an unmodelled auth `method`"
    );
}

#[test]
fn catalogs_reference_yaml_parses_clean_with_env() {
    std::env::set_var("POLARIS_CLIENT_SECRET", "test-secret");
    std::env::set_var("ICEBERG_DEV_TOKEN", "test-token");

    let (cfg, warnings) = parse_catalogs(CATALOGS_YAML).expect("catalogs reference must parse");
    assert!(warnings.is_empty(), "no warnings expected");
    assert_eq!(cfg.catalogs.len(), 3, "three catalog entries expected");
    assert!(cfg.catalogs.contains_key("polaris_prod"));
    assert!(cfg.catalogs.contains_key("iceberg_dev"));
    assert!(cfg.catalogs.contains_key("unity_prod"));

    std::env::remove_var("POLARIS_CLIENT_SECRET");
    std::env::remove_var("ICEBERG_DEV_TOKEN");
}
