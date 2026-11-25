mod common;

use common::{built, expression_model, parsed_expression, refused, refused_expression};
use semstrait_common::diagnostic::Diagnose;
use semstrait_model::{ParseErrorKind, SemanticModel};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, Literal};

fn spellings() -> Vec<(&'static str, DataType)> {
    let bare_decimal = DataType::Decimal {
        precision: 38,
        scale: 9,
    };
    let decimal_18_2 = DataType::Decimal {
        precision: 18,
        scale: 2,
    };
    vec![
        ("boolean", DataType::Boolean),
        ("bool", DataType::Boolean),
        ("tinyint", DataType::TinyInt),
        ("byte", DataType::TinyInt),
        ("i8", DataType::TinyInt),
        ("smallint", DataType::SmallInt),
        ("short", DataType::SmallInt),
        ("int16", DataType::SmallInt),
        ("i16", DataType::SmallInt),
        ("integer", DataType::Integer),
        ("int", DataType::Integer),
        ("int32", DataType::Integer),
        ("i32", DataType::Integer),
        ("bigint", DataType::BigInt),
        ("long", DataType::BigInt),
        ("int64", DataType::BigInt),
        ("i64", DataType::BigInt),
        ("real", DataType::Real),
        ("float", DataType::Real),
        ("float32", DataType::Real),
        ("f32", DataType::Real),
        ("double", DataType::Double),
        ("float64", DataType::Double),
        ("f64", DataType::Double),
        ("decimal", bare_decimal.clone()),
        ("numeric", bare_decimal),
        ("decimal(18,2)", decimal_18_2.clone()),
        ("numeric(18,2)", decimal_18_2),
        ("string", DataType::String),
        ("text", DataType::String),
        ("utf8", DataType::String),
        ("large_utf8", DataType::String),
        ("binary", DataType::Binary),
        ("blob", DataType::Binary),
        ("bytes", DataType::Binary),
        ("varbinary", DataType::Binary),
        ("date", DataType::Date),
        ("date32", DataType::Date),
        ("date64", DataType::Date),
        ("time", DataType::Time { precision: 9 }),
        ("timestamp", DataType::Timestamp { precision: 6 }),
        ("datetime", DataType::Timestamp { precision: 6 }),
        ("timestamp(3)", DataType::Timestamp { precision: 3 }),
        ("array<string>", DataType::Array(Box::new(DataType::String))),
        ("varchar(10)", DataType::VarChar { length: 10 }),
        ("char(2)", DataType::FixedChar { length: 2 }),
        ("binary(16)", DataType::FixedBinary { length: 16 }),
        ("uuid", DataType::Uuid),
        ("interval", DataType::Interval),
        ("interval_year_month", DataType::IntervalYearMonth),
        (
            "interval_day_second(6)",
            DataType::IntervalDaySecond { precision: 6 },
        ),
        (
            "interval_day_second",
            DataType::IntervalDaySecond { precision: 9 },
        ),
    ]
}

const REFUSED: [&str; 12] = [
    "int8",
    "number",
    "number(10,2)",
    "varchar",
    "char",
    "timestamp_ms",
    "timestamp_us",
    "timestamp_ns",
    "timestamp(4)",
    "decimal(40,2)",
    "decimal(99,2)",
    "frobnicate",
];

const OBJECTS: [(&str, &str); 12] = [
    ("{decimal: {precision: 18, scale: 4}}", "decimal(18,4)"),
    ("{numeric: {precision: 18, scale: 4}}", "numeric(18,4)"),
    ("{NUMERIC: {scale: 4, precision: 18}}", "decimal(18,4)"),
    ("{varchar: {length: 10}}", "varchar(10)"),
    ("{timestamp: {precision: 6}}", "timestamp(6)"),
    ("{DateTime: {precision: 3}}", "timestamp(3)"),
    ("{binary: {length: 16}}", "binary(16)"),
    ("{char: {length: 2}}", "char(2)"),
    (
        "{interval_day_second: {precision: 6}}",
        "interval_day_second(6)",
    ),
    (
        "{array: {element: {decimal: {precision: 18, scale: 2}}}}",
        "array<decimal(18,2)>",
    ),
    (
        r#"{Array: {element: "decimal(18,2)"}}"#,
        "array<decimal(18,2)>",
    ),
    (
        "{array: {element: {ARRAY: {element: long}}}}",
        "array<array<bigint>>",
    ),
];

const REFUSED_OBJECTS: [&str; 14] = [
    "{decimal: {PRECISION: 18, scale: 4}}",
    "{decimal: {precision: 18}}",
    "{decimal: {precision: 18, scale: 4, length: 1}}",
    "{varchar: {size: 10}}",
    "{decimal: {precision: 39, scale: 2}}",
    "{decimal: {precision: -1, scale: 2}}",
    "{timestamp: {precision: 4}}",
    "{varchar: {length: 0}}",
    "{decimal: 18}",
    "{bigint: {}}",
    "{frobnicate: {length: 1}}",
    "{decimal: {precision: 18, scale: 4}, varchar: {length: 1}}",
    "{array: {element: varchar}}",
    "{array: {Element: bigint}}",
];

const SITES: [&str; 3] = [
    "{cast: {input: amount, target: TYPE}}",
    "{lit: {null: TYPE}}",
    "{lit: {array: {element: TYPE, values: []}}}",
];

fn model(data_type: &str) -> String {
    expression_model("amount").replace("data_type: string", &format!("data_type: {data_type}"))
}

fn quoted(spelling: &str) -> String {
    format!("\"{spelling}\"")
}

fn declared(data_type: &str) -> Option<DataType> {
    let model = built(&model(data_type));
    let dimension = model.dimensions.values().next().expect("one Dimension");
    dimension.data_type().resolve()
}

fn cases(spelling: &str) -> [String; 3] {
    let mixed = spelling
        .chars()
        .enumerate()
        .map(|(at, c)| {
            if at % 2 == 0 {
                c.to_ascii_uppercase()
            } else {
                c
            }
        })
        .collect();
    [spelling.to_owned(), spelling.to_ascii_uppercase(), mixed]
}

fn tree_type(expr: &str) -> DataType {
    match parsed_expression(expr) {
        Expr::Cast { target, .. } => target,
        Expr::Literal(Literal::Null(data_type)) => data_type,
        Expr::Literal(Literal::Array { element, .. }) => element,
        other => panic!("`{expr}` names no type: {other:?}"),
    }
}

#[test]
fn each_canonical_name_and_alias_reads_in_any_case_at_each_site() {
    for (spelling, expected) in spellings() {
        for written in cases(spelling) {
            let model = built(&model(&quoted(&written)));
            let declared = model.dimensions.values().next().expect("one Dimension");
            assert_eq!(declared.data_type().spelling(), written);
            assert_eq!(*declared.data_type(), expected, "`{written}`");
            for site in SITES {
                let expr = site.replace("TYPE", &quoted(&written));
                assert_eq!(tree_type(&expr), expected, "`{expr}`");
            }
        }
    }
}

#[test]
fn an_object_reads_as_its_string_form_at_data_type_and_at_each_site() {
    for (object, spelling) in OBJECTS {
        assert_eq!(declared(object), declared(&quoted(spelling)), "`{object}`");
        for site in SITES {
            assert_eq!(
                tree_type(&site.replace("TYPE", object)),
                tree_type(&site.replace("TYPE", &quoted(spelling))),
                "`{site}` with `{object}`"
            );
        }
    }
}

#[test]
fn a_written_type_that_names_no_valid_type_is_refused_at_each_site() {
    let spellings = REFUSED.map(|spelling| (quoted(spelling), spelling));
    let objects = REFUSED_OBJECTS.map(|object| (object.to_owned(), object));
    for (written, spelling) in spellings.into_iter().chain(objects) {
        let expected = vec![ParseErrorKind::InvalidDataType {
            spelling: spelling.to_owned(),
        }];
        let at_sites = SITES.map(|site| refused_expression(&site.replace("TYPE", &written)));
        for kinds in [refused(&model(&written))].into_iter().chain(at_sites) {
            assert_eq!(kinds, expected, "`{written}`");
            assert!(kinds[0].message().contains(&format!("`{spelling}`")));
        }
    }
}

#[test]
fn a_model_is_written_back_with_the_canonical_string() {
    for (written, canonical, gone) in [
        ("long", "bigint", "long"),
        (
            "{NUMERIC: {precision: 18, scale: 4}}",
            "decimal(18,4)",
            "precision",
        ),
    ] {
        let model = built(&model(written));
        let yaml = serde_yaml::to_string(&model).expect("a written model");
        assert!(
            yaml.contains(&format!("data_type: {canonical}")) && !yaml.contains(gone),
            "{yaml}"
        );
        let back: SemanticModel = serde_yaml::from_str(&yaml).expect("a read model");
        assert_eq!(back, model);
    }
}

#[test]
fn a_tagged_literals_tag_is_never_an_object() {
    assert!(matches!(
        &refused_expression(r#"{lit: {{decimal: {precision: 18, scale: 2}}: "1.5"}}"#)[..],
        [ParseErrorKind::YamlSyntax { message }] if message.contains("a canonical type name")
    ));
}

#[cfg(feature = "schemars")]
fn verdicts(validator: &jsonschema::Validator, data_type: &str) -> (bool, bool) {
    let verdict = |yaml: &str| {
        let instance: serde_json::Value = serde_yaml::from_str(yaml).expect("a YAML document");
        (
            validator.is_valid(&instance),
            semstrait_model::parse(yaml).is_ok(),
        )
    };
    let at_data_type = verdict(&model(data_type));
    for site in SITES {
        let yaml = expression_model(&site.replace("TYPE", data_type));
        assert_eq!(verdict(&yaml), at_data_type, "{yaml}");
    }
    at_data_type
}

#[cfg(feature = "schemars")]
#[test]
fn the_schema_gives_parses_verdict_on_each_spelling_at_data_type_and_at_each_site() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("a valid JSON schema");
    let deep = format!("{}bigint{}", "array<".repeat(129), ">".repeat(129));
    let only_parse_refuses = [
        "decimal(10,12)",
        "array<array<bigint>",
        "array<bigint>>",
        &deep,
    ];
    let accepted = spellings()
        .into_iter()
        .map(|(spelling, _)| spelling)
        .chain([
            "decimal(38,38)",
            "decimal(30,0)",
            "decimal( 18 , 2 )",
            "varchar(1)",
            "varchar(999999999)",
            "varchar(1000000000)",
            "varchar(2147483640)",
            "varchar(2147483647)",
            "time(0)",
            "interval_day_second(0)",
            "array< array<long> >",
            "Array<Decimal(18,2)>",
        ]);
    let refused = REFUSED.into_iter().chain([
        "decimal(0,0)",
        "decimal(39,2)",
        "decimal(40,0)",
        "decimal(38,39)",
        "decimal(18)",
        "decimal(18,2,1)",
        "varchar(0)",
        "varchar(2147483648)",
        "varchar(2147483650)",
        "varchar(3000000000)",
        "time(1)",
        "interval_day_second(10)",
        "bigint(19)",
        " bigint",
        "decimal (18,2)",
        "array<>",
        "array<varchar>",
    ]);
    for spelling in accepted.chain(refused).chain(only_parse_refuses) {
        for written in cases(spelling) {
            let (valid, parses) = verdicts(&validator, &quoted(&written));
            match only_parse_refuses.contains(&spelling) {
                true => assert!(valid && !parses, "`{written}` is an expected difference"),
                false => assert_eq!(valid, parses, "`{written}`"),
            }
        }
    }
}

#[cfg(feature = "schemars")]
#[test]
fn the_schema_gives_parses_verdict_on_each_object_at_data_type_and_at_each_site() {
    let schema = semstrait_model::schema::semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("a valid JSON schema");
    let objects = OBJECTS.map(|(object, _)| object);
    for object in objects.into_iter().chain(REFUSED_OBJECTS) {
        let (valid, parses) = verdicts(&validator, object);
        assert_eq!(valid, parses, "`{object}`");
    }
    let deepest = format!("{}bigint{}", "array<".repeat(128), ">".repeat(128));
    for object in [
        "{decimal: {precision: 10, scale: 12}}".to_owned(),
        "{decimal: {precision: 18.0, scale: 4}}".to_owned(),
        format!("{{array: {{element: \"{deepest}\"}}}}"),
    ] {
        assert_eq!(
            verdicts(&validator, &object),
            (true, false),
            "`{object}` is an expected difference"
        );
    }
}
