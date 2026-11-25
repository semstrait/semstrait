#![cfg(feature = "schemars")]

const REFERENCE_YAML: &str = include_str!("../schemas/reference.yaml");

fn yaml_to_json(yaml: &str) -> serde_json::Value {
    serde_yaml::from_str(yaml).expect("reference.yaml must parse as a json-compatible value")
}

#[test]
fn generated_schema_compiles() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    jsonschema::validator_for(&schema).expect("generated schema must compile");
}

#[test]
fn reference_yaml_validates_against_generated_schema() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("generated schema must compile");

    let instance = yaml_to_json(REFERENCE_YAML);

    let messages: Vec<String> = validator
        .iter_errors(&instance)
        .map(|err| format!("schema error: {err} (at {})", err.instance_path()))
        .collect();

    if !messages.is_empty() {
        for m in &messages {
            eprintln!("{m}");
        }
        panic!(
            "reference.yaml does not validate against the GENERATED schema ({} errors)",
            messages.len()
        );
    }
}

#[test]
fn generated_schema_rejects_invalid_documents() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("generated schema must compile");

    let empty = serde_json::json!({});
    assert!(
        !validator.is_valid(&empty),
        "schema must require the `semantic_model` root key"
    );

    let no_name = serde_json::json!({ "semantic_model": { "measures": [] } });
    assert!(
        !validator.is_valid(&no_name),
        "schema must require `semantic_model.name`"
    );

    let wrong_type = serde_json::json!({ "semantic_model": { "name": 123 } });
    assert!(
        !validator.is_valid(&wrong_type),
        "schema must type-check `semantic_model.name` as a string"
    );
}

fn properties<'a>(value: &'a serde_json::Value, out: &mut Vec<(&'a str, &'a serde_json::Value)>) {
    if let Some(props) = value.get("properties").and_then(|p| p.as_object()) {
        out.extend(props.iter().map(|(key, property)| (key.as_str(), property)));
    }
    match value {
        serde_json::Value::Object(map) => map.values().for_each(|v| properties(v, out)),
        serde_json::Value::Array(items) => items.iter().for_each(|v| properties(v, out)),
        _ => {}
    }
}

fn property_keys(value: &serde_json::Value) -> Vec<&str> {
    let mut found = Vec::new();
    properties(value, &mut found);
    found.into_iter().map(|(key, _)| key).collect()
}

#[test]
fn each_expression_field_refers_to_the_definition_of_its_leaf_type() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let defs = schema["$defs"].as_object().expect("a `$defs` table");
    let mut fields = Vec::new();
    for (name, def) in defs {
        let mut found = Vec::new();
        properties(def, &mut found);
        fields.extend(found.into_iter().filter_map(|(key, property)| {
            let target = property["$ref"].as_str()?.strip_prefix("#/$defs/")?;
            target
                .ends_with("Expr")
                .then_some([name.as_str(), key, target])
        }));
    }
    fields.sort();
    fields.dedup();
    let semantic = "UnresolvedSemanticExpr";
    let expected = [
        ["DataKindFilter", "expr", semantic],
        ["Dimension", "expr", semantic],
        ["DimensionDecl", "expr", semantic],
        ["DimensionRef", "expr", semantic],
        ["Measure", "expr", semantic],
        ["Measure", "filter", semantic],
        ["MeasureRef", "expr", semantic],
        ["MeasureRef", "filter", semantic],
        ["Metric", "expr", semantic],
        ["Metric", "filter", semantic],
        ["MetricRef", "expr", semantic],
        ["MetricRef", "filter", semantic],
        ["Relationship", "filter", semantic],
        ["SemanticMappingValue", "expr", "UnresolvedPhysicalExpr"],
    ];
    assert_eq!(fields, expected);
    let defined = [
        "Expression",
        "UnresolvedSemanticExpr",
        "UnresolvedPhysicalExpr",
    ];
    assert_eq!(
        defined.map(|name| defs.contains_key(name)),
        [false, true, true]
    );
}

#[test]
fn a_semantics_data_type_refers_to_the_unresolved_data_type_definition() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let defs = schema["$defs"].as_object().expect("a `$defs` table");
    let refs = ["Dimension", "Measure", "Metric"]
        .map(|semantics| &defs[semantics]["properties"]["data_type"]["$ref"]);
    assert!(defs.contains_key("UnresolvedDataType"));
    assert!(
        refs.iter().all(|r| *r == "#/$defs/UnresolvedDataType"),
        "{refs:?}"
    );
}

#[test]
fn no_machine_transport_tags_in_authored_schema() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let keys = property_keys(&schema);
    for machine in ["binary_op", "unary_op", "aggregate", "content"] {
        assert!(
            !keys.contains(&machine),
            "machine transport tag {machine:?} leaked into the authored schema as a property"
        );
    }
}

#[test]
fn generated_schema_encodes_the_three_join_basis_forms() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("generated schema must compile");

    let document = |on: serde_json::Value| {
        serde_json::json!({
            "semantic_model": {
                "name": "ana",
                "relationships": [{
                    "name": "orders_to_customers",
                    "from": "orders",
                    "to": "customers",
                    "cardinality": "many_to_one",
                    "on": on,
                }],
            }
        })
    };

    for accepted in [
        serde_json::json!("auto"),
        serde_json::json!("cross"),
        serde_json::json!(["customer_id"]),
        serde_json::json!(["region_id", "customer_id"]),
    ] {
        assert!(
            validator.is_valid(&document(accepted.clone())),
            "schema must accept `on: {accepted}`"
        );
    }
    for rejected in [
        serde_json::json!([]),
        serde_json::json!("crossed"),
        serde_json::json!([{ "from": "customer_id", "to": "id" }]),
    ] {
        assert!(
            !validator.is_valid(&document(rejected.clone())),
            "schema must reject `on: {rejected}`"
        );
    }
}

#[test]
fn generated_schema_takes_a_bucket_bound_as_a_tagged_literal_or_null() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("generated schema must compile");

    let document = |lower: serde_json::Value| {
        serde_json::json!({
            "semantic_model": {
                "name": "ana",
                "dimensions": [{
                    "name": "bucket",
                    "data_type": "string",
                    "type": { "bucketed": { "buckets": [{ "name": "only", "lower": lower }] } },
                }],
            }
        })
    };

    for accepted in [
        serde_json::json!({ "bigint": 100 }),
        serde_json::Value::Null,
    ] {
        assert!(
            validator.is_valid(&document(accepted.clone())),
            "schema must accept `lower: {accepted}`"
        );
    }
    assert!(
        !validator.is_valid(&document(serde_json::json!(100))),
        "schema must reject the untyped `lower: 100`"
    );
}

#[test]
fn the_generated_schema_carries_the_join_basis_key() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let keys = property_keys(&schema);
    assert!(
        keys.contains(&"on"),
        "`on` absent from the generated schema's declared properties: {keys:?}"
    );
    assert!(
        !keys.contains(&"keys"),
        "the retired join-basis spelling `keys` is still a declared property: {keys:?}"
    );
}

fn required_sets_declaring_cardinality(value: &serde_json::Value, out: &mut Vec<Vec<String>>) {
    match value {
        serde_json::Value::Object(map) => {
            let declares = map
                .get("properties")
                .and_then(|p| p.as_object())
                .is_some_and(|props| props.contains_key("cardinality"));
            if declares {
                out.push(
                    map.get("required")
                        .and_then(|r| r.as_array())
                        .map(|entries| {
                            entries
                                .iter()
                                .filter_map(|e| e.as_str().map(str::to_owned))
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            }
            for v in map.values() {
                required_sets_declaring_cardinality(v, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                required_sets_declaring_cardinality(item, out);
            }
        }
        _ => {}
    }
}

#[test]
fn the_generated_schema_does_not_require_cardinality() {
    let schema = semstrait_model::schema::semantic_model_json_schema();

    let mut required = Vec::new();
    required_sets_declaring_cardinality(&schema, &mut required);
    assert!(
        !required.is_empty(),
        "no subschema declares a `cardinality` property, so this guard proves nothing"
    );
    for set in &required {
        assert!(
            !set.iter().any(|k| k == "cardinality"),
            "`cardinality` is still in a `required` set: {set:?}"
        );
        assert!(
            set.iter().any(|k| k == "from") && set.iter().any(|k| k == "to"),
            "the relationship subschema must still require its endpoints: {set:?}"
        );
    }

    let validator = jsonschema::validator_for(&schema).expect("generated schema must compile");
    let bare_cross = serde_json::json!({
        "semantic_model": {
            "name": "ana",
            "relationships": [{
                "name": "orders_x_promos",
                "from": "orders",
                "to": "promos",
                "on": "cross",
            }],
        }
    });
    let messages: Vec<String> = validator
        .iter_errors(&bare_cross)
        .map(|err| format!("{err} (at {})", err.instance_path()))
        .collect();
    assert!(
        messages.is_empty(),
        "schema must accept a bare `on: cross` relationship: {messages:?}"
    );
}

#[cfg(feature = "schemars")]
#[test]
fn the_generated_schema_carries_the_keyness_keys() {
    let text = serde_json::to_string(&semstrait_model::schema::semantic_model_json_schema())
        .expect("schema serializes");
    for key in ["\"is_key\"", "\"keyset\"", "\"keysets\""] {
        assert!(text.contains(key), "{key} absent from the generated schema");
    }
}

#[test]
fn the_generated_schema_accepts_the_singular_keyset_spelling() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("generated schema must compile");

    let document = |dimensions: serde_json::Value| {
        serde_json::json!({
            "semantic_model": {
                "name": "ana",
                "dimensions": [
                    { "name": "country", "data_type": "string", "type": "categorical" }
                ],
                "datasets": [{ "name": "orders", "dimensions": dimensions }],
            }
        })
    };

    for accepted in [
        serde_json::json!([{ "ref": "country", "is_key": true, "keyset": "natural" }]),
        serde_json::json!([{ "ref": "country", "is_key": true, "keysets": ["natural"] }]),
        serde_json::json!([{ "ref": "country", "is_key": true }]),
        serde_json::json!([{
            "name": "region", "data_type": "string", "type": "categorical",
            "is_key": true, "keyset": "natural",
        }]),
    ] {
        let messages: Vec<String> = validator
            .iter_errors(&document(accepted.clone()))
            .map(|err| format!("{err} (at {})", err.instance_path()))
            .collect();
        assert!(
            messages.is_empty(),
            "schema must accept the keyness keys on {accepted}: {messages:?}"
        );
    }

    for rejected in [
        serde_json::json!([{ "ref": "country", "is_key": "yes" }]),
        serde_json::json!([{ "ref": "country", "is_key": true, "keysets": "natural" }]),
    ] {
        assert!(
            !validator.is_valid(&document(rejected.clone())),
            "schema must reject {rejected}"
        );
    }
}
