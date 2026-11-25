use semstrait_common::Dialect;
use semstrait_sql::datatype::{DataType, WrittenDataType};
use semstrait_sql::DocumentDialect;
use serde_json::{json, Value};

fn written(value: &Value) -> WrittenDataType {
    match value {
        Value::String(spelling) => WrittenDataType::Spelling(spelling.clone()),
        Value::Number(number) => {
            WrittenDataType::Number(number.as_u64().expect("a natural number"))
        }
        Value::Object(entries) => WrittenDataType::Object(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), written(value)))
                .collect(),
        ),
        other => panic!("`{other}` is no written type"),
    }
}

fn objects() -> [(Value, &'static str); 11] {
    [
        (
            json!({"decimal": {"precision": 18, "scale": 4}}),
            "decimal(18,4)",
        ),
        (
            json!({"numeric": {"precision": 18, "scale": 4}}),
            "numeric(18,4)",
        ),
        (
            json!({"NUMERIC": {"scale": 4, "precision": 18}}),
            "decimal(18,4)",
        ),
        (json!({"varchar": {"length": 10}}), "varchar(10)"),
        (json!({"VarChar": {"length": 10}}), "varchar(10)"),
        (json!({"timestamp": {"precision": 6}}), "timestamp(6)"),
        (json!({"DateTime": {"precision": 3}}), "timestamp(3)"),
        (json!({"binary": {"length": 16}}), "binary(16)"),
        (
            json!({"array": {"element": {"decimal": {"precision": 18, "scale": 2}}}}),
            "array<decimal(18,2)>",
        ),
        (
            json!({"ARRAY": {"element": "decimal(18,2)"}}),
            "array<decimal(18,2)>",
        ),
        (
            json!({"array": {"element": {"array": {"element": "long"}}}}),
            "array<array<bigint>>",
        ),
    ]
}

#[test]
fn an_object_resolves_to_the_type_of_its_string_form() {
    for (object, spelling) in objects() {
        let resolved = DataType::resolve_written(&written(&object), DocumentDialect::Canonical);
        assert!(resolved.is_some(), "`{object}`");
        assert_eq!(
            resolved,
            DataType::resolve(spelling, DocumentDialect::Canonical),
            "`{object}`"
        );
    }
}

#[test]
fn a_spelling_resolves_by_its_canonical_name_or_an_alias_in_any_case() {
    for (spelling, expected) in [
        ("long", DataType::BigInt),
        ("LONG", DataType::BigInt),
        ("Double", DataType::Double),
        (
            "DECIMAL(18,4)",
            DataType::Decimal {
                precision: 18,
                scale: 4,
            },
        ),
        ("NuMeRiC", DataType::decimal()),
        ("TimeStamp(3)", DataType::Timestamp { precision: 3 }),
        (
            "Array<VarChar(10)>",
            DataType::Array(Box::new(DataType::VarChar { length: 10 })),
        ),
    ] {
        assert_eq!(
            DataType::resolve(spelling, DocumentDialect::Canonical),
            Some(expected),
            "`{spelling}`"
        );
    }
}

fn refused_objects() -> Vec<Value> {
    vec![
        json!({"decimal": {"PRECISION": 18, "scale": 4}}),
        json!({"decimal": {"precision": 18}}),
        json!({"decimal": {"precision": 18, "scale": 4, "length": 1}}),
        json!({"varchar": {"size": 10}}),
        json!({"decimal": {"precision": 39, "scale": 2}}),
        json!({"decimal": {"precision": 0, "scale": 0}}),
        json!({"timestamp": {"precision": 4}}),
        json!({"varchar": {"length": 0}}),
        json!({"varchar": {"length": 2_147_483_648_u64}}),
        json!({"decimal": {"precision": "18", "scale": 4}}),
        json!({"decimal": 18}),
        json!({"bigint": {}}),
        json!({"uuid": {"length": 1}}),
        json!({"frobnicate": {"length": 1}}),
        json!({"decimal": {"precision": 18, "scale": 4}, "varchar": {"length": 1}}),
        json!({}),
        json!({"array": {"element": "varchar"}}),
        json!({"array": {"Element": "bigint"}}),
        json!({"array": {"element": "bigint", "length": 1}}),
        json!({"array": {}}),
        json!({"array": "bigint"}),
        json!(5),
    ]
}

fn only_resolution_refuses() -> [Value; 2] {
    let deepest = format!("{}bigint{}", "array<".repeat(128), ">".repeat(128));
    [
        json!({"decimal": {"precision": 10, "scale": 12}}),
        json!({"array": {"element": deepest}}),
    ]
}

#[test]
fn an_object_that_names_no_valid_type_resolves_to_none() {
    for object in refused_objects()
        .into_iter()
        .chain(only_resolution_refuses())
    {
        assert_eq!(
            DataType::resolve_written(&written(&object), DocumentDialect::Canonical),
            None,
            "`{object}`"
        );
    }
}

#[test]
fn a_spelling_that_names_no_valid_type_resolves_to_none() {
    for spelling in [
        "varchar",
        "frobnicate",
        "decimal(99,2)",
        "timestamp(4)",
        "decimal(10,12)",
    ] {
        assert_eq!(
            DataType::resolve(spelling, DocumentDialect::Canonical),
            None,
            "`{spelling}`"
        );
    }
}

#[test]
fn a_spelling_resolves_through_the_set_of_declarations_that_it_is_given() {
    let iceberg = DocumentDialect::Dialect(Dialect::Iceberg);
    for (spelling, canonical, on_iceberg) in [
        (
            "Timestamp(9)",
            Some(DataType::Timestamp { precision: 9 }),
            Some(DataType::Timestamp { precision: 9 }),
        ),
        (
            "timestamp(3)",
            Some(DataType::Timestamp { precision: 3 }),
            None,
        ),
        ("tinyint", Some(DataType::TinyInt), None),
    ] {
        assert_eq!(
            DataType::resolve(spelling, DocumentDialect::Canonical),
            canonical,
            "`{spelling}`"
        );
        assert_eq!(
            DataType::resolve(spelling, iceberg),
            on_iceberg,
            "`{spelling}`"
        );
    }
}

#[cfg(feature = "schemars")]
fn schema(dialect: DocumentDialect) -> Value {
    let element = schemars::Schema::new_ref("#".to_owned());
    serde_json::to_value(DataType::resolution_schema(dialect, element)).expect("a JSON schema")
}

#[cfg(feature = "schemars")]
#[test]
fn the_schema_of_a_set_of_declarations_gives_the_verdict_of_its_resolution() {
    let iceberg = DocumentDialect::Dialect(Dialect::Iceberg);
    let validator = jsonschema::validator_for(&schema(iceberg)).expect("a valid JSON schema");
    for (written_type, accepted) in [
        (json!("Timestamp(9)"), true),
        (json!("timestamp(3)"), false),
        (json!("tinyint"), false),
        (json!({"TimeStamp": {"precision": 9}}), true),
        (json!({"timestamp": {"precision": 3}}), false),
        (json!({"list": {"element": "int"}}), true),
        (json!({"array": {"element": "int"}}), false),
    ] {
        assert_eq!(
            validator.is_valid(&written_type),
            accepted,
            "`{written_type}`"
        );
        assert_eq!(
            DataType::resolve_written(&written(&written_type), iceberg).is_some(),
            accepted,
            "`{written_type}`"
        );
    }
}

#[cfg(feature = "schemars")]
#[test]
fn the_schema_gives_the_resolutions_verdict_on_each_object() {
    let validator = jsonschema::validator_for(&schema(DocumentDialect::Canonical))
        .expect("a valid JSON schema");
    let accepted = objects().map(|(object, _)| object);
    for object in accepted.into_iter().chain(refused_objects()) {
        assert_eq!(
            validator.is_valid(&object),
            DataType::resolve_written(&written(&object), DocumentDialect::Canonical).is_some(),
            "`{object}`"
        );
    }
    for object in only_resolution_refuses() {
        assert!(
            validator.is_valid(&object),
            "`{object}` is an expected difference"
        );
    }
}

#[cfg(feature = "schemars")]
fn titles<'a>(schema: &'a Value, kind: &str) -> Vec<&'a str> {
    schema["oneOf"]
        .as_array()
        .expect("one branch for each type")
        .iter()
        .filter(|branch| kind.is_empty() || branch["type"] == kind)
        .map(|branch| branch["title"].as_str().expect("a titled branch"))
        .collect()
}

#[cfg(feature = "schemars")]
#[test]
fn each_canonical_type_has_a_branch_for_each_form_titled_with_its_spelling() {
    assert_eq!(
        titles(&schema(DocumentDialect::Canonical), ""),
        [
            "boolean",
            "tinyint",
            "smallint",
            "integer",
            "bigint",
            "real",
            "double",
            "decimal(p,s)",
            "{decimal: {precision: p, scale: s}}",
            "string",
            "varchar(n)",
            "{varchar: {length: n}}",
            "char(n)",
            "{char: {length: n}}",
            "binary",
            "binary(n)",
            "{binary: {length: n}}",
            "date",
            "time(p)",
            "{time: {precision: p}}",
            "timestamp(p)",
            "{timestamp: {precision: p}}",
            "interval",
            "interval_year_month",
            "interval_day_second(p)",
            "{interval_day_second: {precision: p}}",
            "uuid",
            "array<T>",
            "{array: {element: T}}",
        ]
    );
}

#[cfg(feature = "schemars")]
#[test]
fn each_canonical_type_has_one_string_branch_titled_with_its_spelling_in_both_schemas() {
    let stored = serde_json::to_value(schemars::schema_for!(DataType)).expect("a JSON schema");
    assert_eq!(
        titles(&stored, ""),
        titles(&schema(DocumentDialect::Canonical), "string")
    );
}
