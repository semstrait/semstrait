mod common;

use common::{built, expression, expression_model, parsed_expression, refused, untyped};
use semstrait_common::io::{Fingerprintable, Persistable};
use semstrait_model::expr::field;
use semstrait_model::{
    DimensionType, ParseErrorKind, SemanticMapping, SemanticMappingValue, SemanticModel,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, Literal, LiteralError, UntypedLiteral};
use std::collections::HashSet;

fn arm_model(dimension: &str, arm: &str) -> String {
    format!(
        r#"
semantic_model:
  name: literals
  dimensions:
    - name: {dimension}
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        semantic_mapping:
          {dimension}: {arm}
      dimensions:
        - ref: {dimension}
"#
    )
}

fn mapping_model(body: &str) -> String {
    arm_model("written", &format!("{{lit: {body}}}"))
}

fn synonyms_model(synonyms: &str) -> String {
    expression_model("{lit: 5}").replace(
        "type: categorical",
        &format!("type: categorical\n      ai_context: {{synonyms: {synonyms}}}"),
    )
}

fn mapping(model: &SemanticModel) -> SemanticMappingValue {
    let dataset = model.datasets.values().next().expect("one Dataset");
    let SemanticMapping::Explicit(entries) = &dataset.body.base.extras.semantic_mapping else {
        panic!("the Dataset maps its Dimension");
    };
    entries.values().next().expect("one entry").clone()
}

fn parsed_mapping(body: &str) -> SemanticMappingValue {
    mapping(&built(&mapping_model(body)))
}

#[test]
fn an_untyped_literal_holds_its_parse_default() {
    for (authored, expected) in [
        ("{lit: abc}", Literal::String("abc".to_owned())),
        ("{lit: \"5\"}", Literal::String("5".to_owned())),
        ("{lit: 5}", Literal::BigInt(5)),
        ("{lit: 1.5}", Literal::Double(1.5)),
        ("{lit: 1e3}", Literal::Double(1000.0)),
        ("{lit: null}", Literal::Null(DataType::String)),
        ("{lit: true}", Literal::Boolean(true)),
    ] {
        assert_eq!(
            parsed_expression(authored),
            untyped(expected),
            "`{authored}`"
        );
    }
}

#[test]
fn an_untyped_whole_number_past_bigint_is_refused_with_advice_to_declare_its_type() {
    let nines_38 = "9".repeat(38);
    for digits in [
        "9223372036854775808",
        "-9223372036854775809",
        "99999999999999999999",
        "-99999999999999999999",
        &nines_38,
    ] {
        for model in [
            expression_model(&format!("{{lit: {digits}}}")),
            expression_model(&format!("{{abs: {digits}}}")),
            expression_model(&format!("\"{digits}\"")),
            mapping_model(digits),
        ] {
            let refusal = refused(&model);
            assert!(
                matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
                    if message.contains(&format!("`{{decimal: \"{digits}\"}}`"))),
                "{model}\n{refusal:?}"
            );
        }
    }
}

fn outcome(yaml: &str) -> Result<SemanticModel, Vec<ParseErrorKind>> {
    match semstrait_model::parse(yaml) {
        Ok(_) => Ok(built(yaml)),
        Err(_) => Err(refused(yaml)),
    }
}

fn refused_reason(yaml: &str) -> String {
    match &refused(yaml)[..] {
        [ParseErrorKind::InvalidLiteral { reason, .. }] => reason.clone(),
        other => panic!("not a literal refusal:\n{yaml}\n{other:?}"),
    }
}

#[test]
fn a_tags_body_is_read_by_its_type_as_a_number_or_a_string() {
    let nines_38 = "9".repeat(38);
    let bodies: [fn(&str) -> String; 4] = [
        |body| {
            expression_model(&format!(
                "{{lit: {{decimal: {{value: {body}, precision: 38}}}}}}"
            ))
        },
        |body| mapping_model(&format!("{{decimal: {body}}}")),
        |body| {
            expression_model(&format!(
                "{{lit: {{array: {{element: decimal, values: [{body}]}}}}}}"
            ))
        },
        |body| bucket_model(&format!("{{decimal: {body}}}"), "null"),
    ];
    for digits in [
        "9223372036854775808",
        "99999999999999999999",
        "-99999999999999999999",
        &nines_38,
    ] {
        let quoted = format!("\"{digits}\"");
        for body in [digits, &quoted] {
            assert_eq!(
                parsed_expression(&format!("{{lit: {{decimal: {body}}}}}")),
                tagged(decimal(digits, 38, 0)),
                "`{body}`"
            );
        }
        for model in bodies {
            assert_eq!(
                outcome(&model(digits)),
                outcome(&model(&quoted)),
                "`{digits}`"
            );
        }
        for tag in ["bigint", "double"] {
            let model = |body: &str| expression_model(&format!("{{lit: {{{tag}: {body}}}}}"));
            assert_eq!(
                outcome(&model(digits)),
                outcome(&model(&quoted)),
                "`{tag}`: `{digits}`"
            );
        }
        for tag in ["string", "varchar"] {
            let model = |body: &str| expression_model(&format!("{{lit: {{{tag}: {body}}}}}"));
            assert_eq!(
                refused_reason(&model(digits)),
                refused_reason(&model("5")),
                "`{tag}`: `{digits}`"
            );
        }
    }
}

#[test]
fn a_yaml_integer_of_more_than_38_digits_is_refused() {
    let in_u128 = "123456789012345678901234567890123456789";
    let past_u128 = "1234567890123456789012345678901234567890";
    for (digits, float) in [
        (in_u128.to_owned(), "1.2345678901234568e38"),
        (format!("-{in_u128}"), "-1.2345678901234568e38"),
    ] {
        for model in [
            expression_model(&format!("{{lit: {digits}}}")),
            expression_model(&digits),
            mapping_model(&digits),
        ] {
            let refusal = refused(&model);
            assert!(
                matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
                    if message.contains("no decimal holds it")
                        && message.contains(&format!("`{{double: {float}}}`"))),
                "`{digits}`: {refusal:?}"
            );
        }
        assert_eq!(
            refused(&expression_model(&format!(
                "{{lit: {{decimal: {digits}}}}}"
            ))),
            refused_literal("decimal(38,0)", LiteralError::TooManyDigits),
            "`{digits}`"
        );
    }
    for (authored, float) in [
        (past_u128.to_owned(), "1.2345678901234568e39"),
        (format!("-{past_u128}"), "-1.2345678901234568e39"),
        ("1.0e39".to_owned(), "1e39"),
    ] {
        for model in [
            expression_model(&format!("{{lit: {authored}}}")),
            expression_model(&authored),
            mapping_model(&authored),
        ] {
            let refusal = refused(&model);
            assert!(
                matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
                    if message.contains(&format!("`{{double: {float}}}`"))),
                "`{authored}`: {refusal:?}"
            );
        }
    }
    for authored in ["{lit: 2.5e38}", "2.5e38"] {
        assert_eq!(
            parsed_expression(authored),
            untyped(Literal::Double(2.5e38)),
            "`{authored}`"
        );
    }
    assert_eq!(
        parsed_expression("{lit: {double: 1.0e39}}"),
        tagged(Literal::Double(1e39))
    );
}

#[test]
fn the_float_band_is_refused_at_or_past_2_128_and_at_or_below_minus_2_127() {
    for (model, float) in [
        (expression_model("{lit: -2e38}"), "-2e38"),
        (expression_model("\"1e39\""), "1e39"),
        (expression_model("\"-2e38\""), "-2e38"),
        (expression_model("{abs: \"1e39\"}"), "1e39"),
        (expression_model("{abs: \"-2e38\"}"), "-2e38"),
        (arm_model("written", "{expr: \"1e39\"}"), "1e39"),
    ] {
        let refusal = refused(&model);
        assert!(
            matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
                if message.contains(&format!("`{{double: {float}}}`"))),
            "{model}\n{refusal:?}"
        );
    }
    for (authored, float) in [
        ("{lit: 2e38}", 2e38),
        ("{lit: -1.5e38}", -1.5e38),
        ("\"2e38\"", 2e38),
    ] {
        assert_eq!(
            parsed_expression(authored),
            untyped(Literal::Double(float)),
            "`{authored}`"
        );
    }
    assert_eq!(
        parsed_expression("{lit: {double: -2e38}}"),
        tagged(Literal::Double(-2e38))
    );
}

#[test]
fn the_writer_refuses_an_untyped_double_that_parse_refuses() {
    let mut model = built(&expression_model("{lit: 5}"));
    for float in [1e39, -2e38, 2e38] {
        let dimension = model.dimensions.values_mut().next().expect("one Dimension");
        dimension.body.body.expr = Some(untyped(Literal::Double(float)));
        if float == 2e38 {
            let yaml = serde_yaml::to_string(&model).expect("the writer writes YAML");
            let json = serde_json::to_string(&model).expect("the writer writes JSON");
            let from_yaml: SemanticModel =
                serde_yaml::from_str(&yaml).expect("the YAML reads back");
            let from_json: SemanticModel =
                serde_json::from_str(&json).expect("the JSON reads back");
            assert_eq!((from_yaml, from_json), (model.clone(), model.clone()));
        } else {
            assert_the_writer_refuses(&model);
        }
    }
}

fn assert_refused_as_an_integer(yaml: &str, digits: &str) {
    let refusal = refused(yaml);
    assert!(
        matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
            if message.contains(&format!("integer `{digits}`"))
                && !message.contains(" map")
                && !message.contains("tag:")),
        "{yaml}\n{refusal:?}"
    );
}

#[test]
fn a_refusal_names_an_integer_past_64_bits_as_written() {
    let wide = "99999999999999999999";
    for yaml in [
        expression_model("{lit: 5}").replace("type: categorical", &format!("type: {wide}")),
        expression_model(&format!(
            "{{lit: {{decimal: {{value: '1', precision: {wide}}}}}}}"
        )),
        expression_model(&format!("{{dim: {{name: written, accessor: {wide}}}}}")),
        expression_model(&format!(
            "{{dim: {{name: written, accessor: {{lag: {wide}}}}}}}"
        )),
        bucket_model(wide, "null"),
        synonyms_model(&format!("{{lit: {wide}}}")),
        arm_model("lit", wide),
    ] {
        assert_refused_as_an_integer(&yaml, wide);
    }
}

#[test]
fn a_yaml_integer_past_64_bits_is_refused_where_its_64_bit_twin_is() {
    let wide = "99999999999999999999";
    let read: [fn(&str) -> String; 8] = [
        |integer| expression_model(&format!("{{lit: {{string: {integer}}}}}")),
        |integer| mapping_model(&format!("{{string: {integer}}}")),
        |integer| {
            expression_model(&format!(
                "{{lit: {{varchar: {{value: {integer}, length: 20}}}}}}"
            ))
        },
        |integer| {
            expression_model(&format!(
                "{{lit: {{array: {{element: string, values: [{integer}]}}}}}}"
            ))
        },
        |integer| expression_model(&format!("{{lit: {{array: {integer}}}}}")),
        |integer| expression_model(&format!("{{lit: {{lit: {integer}}}}}")),
        |integer| mapping_model(&format!("{{lit: {integer}}}")),
        |integer| expression_model(&format!("{{col: {integer}}}")),
    ];
    for model in read {
        assert_eq!(
            refused(&model(wide)),
            refused(&model("5")),
            "{}",
            model(wide)
        );
    }
    let unread: [fn(&str) -> String; 2] = [
        |integer| arm_model("written", integer),
        |integer| synonyms_model(&format!("{{lit: [{integer}]}}")),
    ];
    for model in unread {
        refused(&model("5"));
        assert_refused_as_an_integer(&model(wide), wide);
    }
}

const INTEGER_TAG: &str = "tag:yaml.org,2002:int";

const DECIMAL_5_JSON: &str = r#"{"decimal":{"value":"5","precision":38,"scale":0}}"#;

#[test]
fn a_lit_value_keyed_by_yamls_integer_tag_is_an_unknown_tag() {
    let unknown = |tag: String| vec![ParseErrorKind::UnknownTag { tag }];
    let lit_tag = format!("lit.{INTEGER_TAG}");
    for value in ["\"5\"", "\"abc\"", "5"] {
        let body = format!("{{\"{INTEGER_TAG}\": {value}}}");
        for model in [
            expression_model(&format!("{{lit: {body}}}")),
            mapping_model(&body),
            expression_model(&format!("{{lit: {{decimal: {body}}}}}")),
            expression_model(&format!(
                "{{lit: {{decimal: {{value: {body}, precision: 38}}}}}}"
            )),
            expression_model(&format!(
                "{{lit: {{array: {{element: decimal, values: [{body}]}}}}}}"
            )),
            bucket_model(&format!("{{decimal: {body}}}"), "null"),
        ] {
            assert_eq!(refused(&model), unknown(lit_tag.clone()), "{model}");
        }
        assert_eq!(
            refused(&expression_model(&format!("{{abs: {body}}}"))),
            unknown(INTEGER_TAG.to_owned())
        );
    }
    let keyed = format!(r#"{{"{INTEGER_TAG}":"5"}}"#);
    for (yaml, written, keyed) in [
        (
            expression_model(r#"{lit: {decimal: "5"}}"#),
            DECIMAL_5_JSON,
            format!(r#"{{"decimal":{keyed}}}"#),
        ),
        (
            expression_model(r#"{lit: {decimal: "5"}}"#),
            r#""value":"5""#,
            format!(r#""value":{keyed}"#),
        ),
        (
            expression_model(r#"{lit: {array: {element: decimal, values: ["5"]}}}"#),
            r#""values":["5"]"#,
            format!(r#""values":[{keyed}]"#),
        ),
        (
            bucket_model(r#"{decimal: "5"}"#, "null"),
            DECIMAL_5_JSON,
            format!(r#"{{"decimal":{keyed}}}"#),
        ),
    ] {
        let json = serde_json::to_string(&built(&yaml)).expect("the writer writes JSON");
        assert_eq!(json.matches(written).count(), 1, "`{written}` in {json}");
        let refusal = serde_json::from_str::<SemanticModel>(&json.replace(written, &keyed))
            .expect_err(&keyed)
            .to_string();
        assert!(refusal.contains(&lit_tag), "{keyed}: {refusal}");
    }
}

const COLLECTIONS_MODEL: &str = r#"
semantic_model:
  name: collections
  dimensions:
    - {name: region, data_type: string, type: categorical, expr: {lit: 5}}
  measures:
    - {name: revenue, data_type: double, agg: sum, filter: {eq: [region, 6]}}
  metrics:
    - {name: margin, data_type: double, expr: revenue, filter: {eq: [region, 7]}}
  relationships:
    - {name: to_customers, from: orders, to: customers, on: cross, filter: {eq: [region, 8]}}
  datasets:
    - name: orders
      extras: {semantic_mapping: {region: {lit: {decimal: 55555555555555555555}}}}
      dimensions: [{ref: region}]
      measures: [{ref: revenue}]
      metrics: [{ref: margin}]
    - {name: customers}
  grainsets:
    - name: rolled
      extras: {}
      datasets:
        - name: daily
          extras:
            semantic_mapping: {region: {lit: {decimal: 22222222222222222222}}}
            temporal: {events: {event_time: region}, granularity: day}
        - {name: hourly, extras: {temporal: {events: {event_time: region}, granularity: hour}}}
  unionsets:
    - name: all_rows
      extras: {}
      mode: all
      datasets:
        - {name: first, extras: {semantic_mapping: {region: {lit: {decimal: 33333333333333333333}}}}}
        - {name: second}
  joinsets:
    - name: joined
      extras: {}
      datasets:
        - {name: left, extras: {semantic_mapping: {region: {lit: {decimal: 44444444444444444444}}}}}
        - {name: right}
"#;

#[test]
fn the_stored_model_and_the_json_form_read_each_collection_through_the_models_reader() {
    let model = built(COLLECTIONS_MODEL);
    let stored = model.to_bytes().expect("the writer writes the model");
    assert_eq!(
        SemanticModel::from_bytes(&stored).expect("the stored model reads back"),
        model
    );
    let json = serde_json::to_string(&model).expect("the writer writes JSON");
    let decimal = |integer: &str| {
        format!(r#"{{"lit":{{"decimal":{{"value":"{integer}","precision":38,"scale":0}}}}}}"#)
    };
    let untyped = |integer: &str| format!(r#"{{"lit":{integer}}}"#);
    let data_kinds = [
        ("datasets", "55555555555555555555"),
        ("grainsets", "22222222222222222222"),
        ("unionsets", "33333333333333333333"),
        ("joinsets", "44444444444444444444"),
    ]
    .map(|(collection, integer)| (collection, integer, decimal(integer)));
    let semantics = [
        ("dimensions", "5"),
        ("measures", "6"),
        ("metrics", "7"),
        ("relationships", "8"),
    ]
    .map(|(collection, integer)| (collection, integer, untyped(integer)));
    for (collection, integer, written) in data_kinds.into_iter().chain(semantics) {
        assert_eq!(json.matches(&written).count(), 1, "{collection}: {json}");
        let keyed = json.replace(
            &written,
            &format!(r#"{{"lit":{{"{INTEGER_TAG}":"{integer}"}}}}"#),
        );
        let refusal = serde_json::from_str::<SemanticModel>(&keyed)
            .expect_err(collection)
            .to_string();
        assert!(
            refusal.contains(&format!("lit.{INTEGER_TAG}")),
            "{collection}: {refusal}"
        );
    }
}

fn tagged(literal: Literal) -> UnresolvedSemanticExpr {
    Expr::Literal(literal)
}

fn decimal(value: &str, precision: u8, scale: i8) -> Literal {
    Literal::Decimal {
        value: value.to_owned(),
        precision,
        scale,
    }
}

#[test]
fn a_tagged_literal_keeps_its_tags_type_with_the_defaults_of_rule_5() {
    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    for (authored, expected) in [
        ("{lit: {integer: 5}}".to_owned(), Literal::Integer(5)),
        (
            "{lit: {decimal: \"1.5\"}}".to_owned(),
            decimal("1.5", 38, 1),
        ),
        ("{lit: {decimal: 1.5}}".to_owned(), decimal("1.5", 38, 1)),
        (
            "{lit: {decimal: {value: \"1.5\", precision: 10}}}".to_owned(),
            decimal("1.5", 10, 1),
        ),
        (
            "{lit: {time: \"12:34:56\"}}".to_owned(),
            Literal::Time {
                value: "12:34:56".to_owned(),
                precision: 9,
            },
        ),
        (
            "{lit: {timestamp: \"2026-05-19T10:00:00.123\"}}".to_owned(),
            Literal::Timestamp {
                value: "2026-05-19T10:00:00.123".to_owned(),
                precision: 6,
            },
        ),
        (
            "{lit: {timestamp: \"2026-05-19T10:00:00.1234567\"}}".to_owned(),
            Literal::Timestamp {
                value: "2026-05-19T10:00:00.1234567".to_owned(),
                precision: 9,
            },
        ),
        (
            "{lit: {varchar: {value: abc, length: 10}}}".to_owned(),
            Literal::VarChar {
                value: "abc".to_owned(),
                length: 10,
            },
        ),
        (
            "{lit: {varchar: abc}}".to_owned(),
            Literal::VarChar {
                value: "abc".to_owned(),
                length: 3,
            },
        ),
        (
            "{lit: {char: ab}}".to_owned(),
            Literal::FixedChar {
                value: "ab".to_owned(),
                length: 2,
            },
        ),
        (
            "{lit: {binary: {value: [1, 2], length: 2}}}".to_owned(),
            Literal::FixedBinary {
                value: vec![1, 2],
                length: 2,
            },
        ),
        (
            format!("{{lit: {{uuid: \"{uuid}\"}}}}"),
            Literal::Uuid(uuid.to_owned()),
        ),
        (
            "{lit: {interval_day_second: {value: P1DT2H, precision: 6}}}".to_owned(),
            Literal::IntervalDaySecond {
                value: "P1DT2H".to_owned(),
                precision: 6,
            },
        ),
        (
            "{lit: {interval_day_second: P1DT2H}}".to_owned(),
            Literal::IntervalDaySecond {
                value: "P1DT2H".to_owned(),
                precision: 9,
            },
        ),
    ] {
        assert_eq!(
            parsed_expression(&authored),
            tagged(expected),
            "`{authored}`"
        );
    }
}

fn refused_literal(data_type: &str, reason: LiteralError) -> Vec<ParseErrorKind> {
    vec![ParseErrorKind::InvalidLiteral {
        data_type: data_type.to_owned(),
        reason: reason.to_string(),
    }]
}

#[test]
fn parse_refuses_a_literal_that_its_type_does_not_hold() {
    let digits_39 = "123456789012345678901234567890123456789";
    for (authored, data_type, reason) in [
        (
            format!("{{lit: {{decimal: \"{digits_39}\"}}}}"),
            "decimal(38,0)",
            LiteralError::TooManyDigits,
        ),
        (
            "{lit: {decimal: {value: \"123.45\", precision: 4}}}".to_owned(),
            "decimal(4,2)",
            LiteralError::CutsADigit,
        ),
        (
            "{lit: {timestamp: {value: \"9999-12-31\", precision: 9}}}".to_owned(),
            "timestamp(9)",
            LiteralError::OutOfRange,
        ),
        (
            "{lit: {varchar: {value: abcdef, length: 3}}}".to_owned(),
            "varchar(3)",
            LiteralError::TooLong,
        ),
        (
            "{lit: {timestamp: \"2026-05-19T10:00:00+02:00\"}}".to_owned(),
            "timestamp(6)",
            LiteralError::Zoned,
        ),
    ] {
        assert_eq!(
            refused(&expression_model(&authored)),
            refused_literal(data_type, reason),
            "`{authored}`"
        );
    }
}

#[test]
fn a_double_that_does_not_hold_its_value_is_refused() {
    let beyond = format!("\"{}.5\"", "9".repeat(400));
    for (authored, reason) in [
        ("\"0.123456789012345678\"", LiteralError::CutsADigit),
        (
            "{add: [1, \"0.123456789012345678\"]}",
            LiteralError::CutsADigit,
        ),
        (&beyond, LiteralError::OutOfRange),
        ("\"1e400\"", LiteralError::OutOfRange),
        ("{lit: .inf}", LiteralError::Malformed),
        ("{lit: {double: -.inf}}", LiteralError::Malformed),
        ("{lit: .nan}", LiteralError::Malformed),
    ] {
        assert_eq!(
            refused(&expression_model(authored)),
            refused_literal("double", reason),
            "`{authored}`"
        );
    }
    assert_eq!(
        refused(&mapping_model(".inf")),
        refused_literal("double", LiteralError::Malformed)
    );
    for (body, data_type, reason) in [
        ("{real: .inf}", "real", LiteralError::Malformed),
        ("{decimal: .nan}", "decimal(38,0)", LiteralError::Malformed),
        ("{string: .inf}", "string", LiteralError::NoReading),
        (
            "{array: {element: real, values: [.nan]}}",
            "real",
            LiteralError::Malformed,
        ),
    ] {
        for model in [
            expression_model(&format!("{{lit: {body}}}")),
            mapping_model(body),
        ] {
            assert_eq!(
                refused(&model),
                refused_literal(data_type, reason),
                "`{body}`"
            );
        }
    }
}

fn plus_one(left: UnresolvedSemanticExpr) -> UnresolvedSemanticExpr {
    Expr::BinaryOp {
        op: BinaryOpKind::Add,
        left: Box::new(left),
        right: Box::new(untyped(Literal::BigInt(1))),
    }
}

#[test]
fn a_bare_word_is_a_name_even_where_it_names_a_float() {
    for name in ["nan", "NaN", "inf", "infinity", "Infinity"] {
        assert_eq!(
            parsed_expression(&format!("{{add: [{name}, 1]}}")),
            plus_one(field(name).expect("a name")),
            "`{name}`"
        );
    }
}

#[test]
fn a_number_with_an_exponent_reads_as_a_double() {
    for (authored, expected) in [
        (
            "{add: [\"1e3\", 1]}",
            plus_one(untyped(Literal::Double(1000.0))),
        ),
        ("\"1E3\"", untyped(Literal::Double(1000.0))),
        ("\"-2.5e-3\"", untyped(Literal::Double(-0.0025))),
        ("\"0.15e+2\"", untyped(Literal::Double(15.0))),
        ("{lit: {double: \"1e3\"}}", tagged(Literal::Double(1000.0))),
        ("{lit: {real: \"1.5E2\"}}", tagged(Literal::Real(150.0))),
    ] {
        assert_eq!(parsed_expression(authored), expected, "`{authored}`");
    }
}

#[test]
fn a_value_that_the_tags_type_does_not_take_is_refused_by_its_name() {
    for (body, value) in [
        ("{integer: {value: 5, length: 3}}", "length"),
        ("{date: {value: \"2026-01-01\", precision: 3}}", "precision"),
        ("{varchar: {value: abc, scale: 1}}", "scale"),
    ] {
        let refusal = refused(&expression_model(&format!("{{lit: {body}}}")));
        assert!(
            matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
                if message.contains(&format!("invalid value for field `{value}`"))
                    && !message.contains("unknown reserved tag")),
            "`{body}`: {refusal:?}"
        );
    }
}

#[test]
fn each_tag_gives_one_literal_in_an_expression_and_in_a_mapping() {
    let tagged_cases = [
        ("{boolean: true}", Literal::Boolean(true)),
        ("{tinyint: 5}", Literal::TinyInt(5)),
        ("{smallint: 5}", Literal::SmallInt(5)),
        ("{integer: 5}", Literal::Integer(5)),
        ("{bigint: 5}", Literal::BigInt(5)),
        ("{real: 1.5}", Literal::Real(1.5)),
        ("{double: 1.5}", Literal::Double(1.5)),
        ("{decimal: \"1.25\"}", decimal("1.25", 38, 2)),
        ("{string: USD}", Literal::String("USD".to_owned())),
        (
            "{date: \"2026-05-19\"}",
            Literal::Date("2026-05-19".to_owned()),
        ),
        (
            "{timestamp: \"2026-05-19T10:00:00\"}",
            Literal::Timestamp {
                value: "2026-05-19T10:00:00".to_owned(),
                precision: 6,
            },
        ),
        (
            "{time: \"10:00:00\"}",
            Literal::Time {
                value: "10:00:00".to_owned(),
                precision: 9,
            },
        ),
        ("{interval: P1D}", Literal::Interval("P1D".to_owned())),
        ("{binary: [1, 2]}", Literal::Binary(vec![1, 2])),
        (
            "{varchar: abc}",
            Literal::VarChar {
                value: "abc".to_owned(),
                length: 3,
            },
        ),
        (
            "{char: ab}",
            Literal::FixedChar {
                value: "ab".to_owned(),
                length: 2,
            },
        ),
        (
            "{uuid: \"550e8400-e29b-41d4-a716-446655440000\"}",
            Literal::Uuid("550e8400-e29b-41d4-a716-446655440000".to_owned()),
        ),
        (
            "{interval_year_month: P1Y2M}",
            Literal::IntervalYearMonth("P1Y2M".to_owned()),
        ),
        (
            "{interval_day_second: P1D}",
            Literal::IntervalDaySecond {
                value: "P1D".to_owned(),
                precision: 9,
            },
        ),
        ("{null: bigint}", Literal::Null(DataType::BigInt)),
    ];
    for (body, expected) in tagged_cases {
        assert_eq!(
            parsed_expression(&format!("{{lit: {body}}}")),
            tagged(expected.clone()),
            "expression `{body}`"
        );
        assert_eq!(
            parsed_mapping(body),
            SemanticMappingValue::Literal(expected),
            "mapping `{body}`"
        );
    }
    for (body, expected) in [
        ("null", Literal::Null(DataType::String)),
        ("\"null\"", Literal::String("null".to_owned())),
        ("0", Literal::BigInt(0)),
    ] {
        assert_eq!(
            parsed_expression(&format!("{{lit: {body}}}")),
            untyped(expected.clone()),
            "expression `{body}`"
        );
        assert_eq!(
            parsed_mapping(body),
            SemanticMappingValue::Untyped(UntypedLiteral(expected)),
            "mapping `{body}`"
        );
    }
}

#[test]
fn an_alias_is_no_tag() {
    for body in [
        "{int: 0}",
        "{bool: true}",
        "{float: 1.5}",
        "{long: 0}",
        "{numeric: \"1.5\"}",
    ] {
        for yaml in [
            expression_model(&format!("{{lit: {body}}}")),
            mapping_model(body),
        ] {
            let refusal = refused(&yaml);
            assert!(
                matches!(&refusal[..], [ParseErrorKind::UnknownTag { tag }] if tag.starts_with("lit.")),
                "`{body}`: {refusal:?}"
            );
        }
    }
}

#[test]
fn an_array_literal_reads_each_value_by_its_elements_grammar() {
    let strings = Literal::Array {
        element: DataType::String,
        values: vec![
            Literal::String("a".to_owned()),
            Literal::String("b".to_owned()),
        ],
    };
    let body = "{array: {element: string, values: [\"a\", \"b\"]}}";
    assert_eq!(
        parsed_expression(&format!("{{lit: {body}}}")),
        tagged(strings.clone())
    );
    assert_eq!(parsed_mapping(body), SemanticMappingValue::Literal(strings));

    let decimals = Literal::Array {
        element: DataType::Decimal {
            precision: 10,
            scale: 2,
        },
        values: vec![
            decimal("1.5", 10, 2),
            Literal::Null(DataType::Decimal {
                precision: 10,
                scale: 2,
            }),
        ],
    };
    assert_eq!(
        parsed_expression("{lit: {array: {element: \"decimal(10,2)\", values: [1.5, null]}}}"),
        tagged(decimals)
    );

    assert_eq!(
        refused(&expression_model(
            "{lit: {array: {element: integer, values: [1, 1.5]}}}"
        )),
        refused_literal("integer", LiteralError::CutsADigit)
    );
    for yaml in [
        expression_model("{lit: [\"a\", \"b\"]}"),
        mapping_model("[\"a\", \"b\"]"),
    ] {
        let refusal = refused(&yaml);
        assert!(
            matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }] if message.contains("untyped list")),
            "{refusal:?}"
        );
    }
}

fn integer_tokens(json: &str) -> Vec<String> {
    let (mut tokens, mut token) = (Vec::new(), String::new());
    let (mut in_string, mut escaped) = (false, false);
    for character in json.chars() {
        if in_string {
            in_string = escaped || character != '"';
            escaped = !escaped && character == '\\';
        } else if character.is_ascii_digit() || matches!(character, '-' | '+' | '.' | 'e' | 'E') {
            token.push(character);
            continue;
        } else {
            in_string = character == '"';
        }
        tokens.push(std::mem::take(&mut token));
    }
    tokens.push(token);
    tokens.retain(|token| token.contains(|digit: char| digit.is_ascii_digit()));
    tokens.retain(|token| !token.contains(['.', 'e', 'E']));
    tokens
}

#[test]
fn a_literal_keeps_its_type_through_yaml_and_json() {
    assert_eq!(
        integer_tokens(r#"{"a":[1e20,-5,99999999999999999999,2.5],"b":"7 \"8\"","c":true}"#),
        ["-5", "99999999999999999999"]
    );
    let wide = tagged(decimal("99999999999999999999", 38, 0));
    for (authored, expected) in [
        ("{lit: 5}", untyped(Literal::BigInt(5))),
        ("{lit: 1.0e20}", untyped(Literal::Double(1e20))),
        ("{lit: {decimal: \"99999999999999999999\"}}", wide.clone()),
        ("{lit: {decimal: 99999999999999999999}}", wide.clone()),
        (
            "{lit: {decimal: {value: \"99999999999999999999\", precision: 38}}}",
            wide.clone(),
        ),
    ] {
        let model = built(&expression_model(authored));
        assert_eq!(expression(&model), expected, "`{authored}`");
        let stored = model.to_bytes().expect("the writer writes the model");
        assert_eq!(
            SemanticModel::from_bytes(&stored).expect("the stored model reads back"),
            model,
            "`{authored}`"
        );
        let json = serde_json::to_string(&model).expect("the writer writes JSON");
        assert_eq!(
            serde_json::from_str::<SemanticModel>(&json).expect("the JSON reads back"),
            model,
            "`{authored}`"
        );
        let past_i64: Vec<String> = (integer_tokens(&json).into_iter())
            .filter(|token| token.parse::<i64>().is_err())
            .collect();
        assert!(past_i64.is_empty(), "`{authored}`: {past_i64:?} in {json}");
    }
}

#[test]
fn the_writer_round_trips_an_untyped_and_a_tagged_literal_each_in_its_own_form() {
    let wide = "123456789012345678901234567890";
    let below_bigint = "-9223372036854775809";
    let yaml = r#"
semantic_model:
  name: literals
  dimensions:
    - name: untyped
      data_type: bigint
      type: categorical
      expr: {add: [{lit: 5}, 2.5]}
    - name: tagged
      data_type: integer
      type: categorical
      expr: {add: [{lit: {integer: 5}}, {lit: {decimal: "1.25"}}]}
    - name: wide
      data_type: decimal
      type: categorical
      expr: {add: [{lit: {decimal: "123456789012345678901234567890"}}, {lit: {decimal: -9223372036854775809}}]}
    - name: floats
      data_type: double
      type: categorical
      expr: {add: [{lit: 1.0715660391465826e-75}, {lit: {real: 1e-45}}]}
  datasets:
    - name: orders
      extras:
        semantic_mapping:
          untyped: {lit: 0}
          tagged: {lit: {bigint: 0}}
      dimensions:
        - ref: untyped
        - ref: tagged
        - ref: wide
        - ref: floats
"#;
    let model = built(yaml);
    let written = serde_yaml::to_string(&model).expect("the writer writes the model");
    for form in [
        "lit: 5",
        "lit: 2.5",
        "integer: 5",
        "precision: 38",
        "lit: 0",
        "bigint: 0",
        &format!("value: '{wide}'"),
        &format!("value: '{below_bigint}'"),
    ] {
        assert!(written.contains(form), "`{form}` is not in:\n{written}");
    }
    let from_yaml: semstrait_model::SemanticModel =
        serde_yaml::from_str(&written).expect("the written YAML reads back");
    assert_eq!(from_yaml, model);
    let json = serde_json::to_string(&model).expect("the writer writes JSON");
    let from_json: semstrait_model::SemanticModel =
        serde_json::from_str(&json).expect("the written JSON reads back");
    assert_eq!(from_json, model);
}

#[test]
fn a_mappings_wide_decimal_is_written_with_its_tag_and_its_digits_as_a_string_and_has_its_own_fingerprint(
) {
    let wide = [
        "99999999999999999999",
        "22222222222222222222",
        "99999999999999999998",
        "-99999999999999999999",
    ];
    let models: Vec<SemanticModel> = wide
        .iter()
        .map(|digits| built(&mapping_model(&format!("{{decimal: {digits}}}"))))
        .collect();
    for (digits, model) in wide.iter().zip(&models) {
        let stored = model.to_bytes().expect("the writer writes the model");
        let written = String::from_utf8_lossy(&stored);
        for form in ["decimal:".to_owned(), format!("value: '{digits}'\n")] {
            assert!(written.contains(&form), "`{form}` in:\n{written}");
        }
        assert_eq!(
            &SemanticModel::from_bytes(&stored).expect("the stored model reads back"),
            model
        );
    }
    let fingerprints: HashSet<[u8; 32]> = models.iter().map(Fingerprintable::fingerprint).collect();
    assert_eq!(fingerprints.len(), wide.len());
}

#[test]
fn a_mappings_wide_decimal_reads_back_from_json_as_the_same_decimal() {
    let digits = "99999999999999999999";
    let json = serde_json::to_string(&built(&mapping_model(&format!("{{decimal: {digits}}}"))))
        .expect("the writer writes JSON");
    assert!(json.contains(&format!("\"value\":\"{digits}\"")), "{json}");
    let from_json: SemanticModel = serde_json::from_str(&json).expect("the JSON reads back");
    assert_eq!(
        mapping(&from_json),
        SemanticMappingValue::Literal(decimal(digits, 38, 0))
    );
}

fn assert_the_writer_refuses(model: &SemanticModel) {
    for written in [
        serde_yaml::to_string(model).map_err(|error| error.to_string()),
        serde_json::to_string(model).map_err(|error| error.to_string()),
    ] {
        assert!(
            written
                .as_ref()
                .is_err_and(|message| message.contains("not representable")),
            "{written:?}"
        );
    }
}

#[test]
fn the_writer_refuses_a_literal_that_would_not_read_back() {
    let past_bigint = decimal("9223372036854775808", 38, 0);
    let mut model = built(&expression_model("{lit: 5}"));
    for expr in [
        tagged(Literal::VarChar {
            value: "abcdef".to_owned(),
            length: 3,
        }),
        tagged(decimal("1.5", 50, 1)),
        tagged(decimal("1.255", 4, 2)),
        tagged(Literal::Timestamp {
            value: "2026-01-01T00:00:00".to_owned(),
            precision: 12,
        }),
        tagged(Literal::Date("2026-02-30".to_owned())),
        tagged(Literal::Array {
            element: DataType::Integer,
            values: vec![Literal::BigInt(1)],
        }),
        tagged(Literal::Double(f64::INFINITY)),
        tagged(Literal::Real(f32::NAN)),
        untyped(Literal::Double(f64::NEG_INFINITY)),
        untyped(Literal::Double(f64::NAN)),
        untyped(past_bigint.clone()),
    ] {
        let dimension = model.dimensions.values_mut().next().expect("one Dimension");
        dimension.body.body.expr = Some(expr);
        assert_the_writer_refuses(&model);
    }
    let mut mapped = built(&mapping_model("0"));
    let dataset = mapped.datasets.values_mut().next().expect("one Dataset");
    let SemanticMapping::Explicit(entries) = &mut dataset.body.base.extras.semantic_mapping else {
        panic!("the Dataset maps its Dimension");
    };
    *entries.values_mut().next().expect("one entry") =
        SemanticMappingValue::Untyped(UntypedLiteral(past_bigint));
    assert_the_writer_refuses(&mapped);
}

fn bucket_model(lower: &str, upper: &str) -> String {
    format!(
        r#"
semantic_model:
  name: literals
  dimensions:
    - name: bucket
      data_type: string
      type:
        bucketed:
          buckets:
            - name: only
              lower: {lower}
              upper: {upper}
  datasets:
    - name: orders
      dimensions:
        - ref: bucket
"#
    )
}

#[test]
fn a_bucket_bound_is_a_tagged_literal_or_no_bound() {
    for (lower, upper, expected) in [
        (
            "{decimal: \"1.5\"}",
            "{date: \"2026-01-01\"}",
            (
                Some(decimal("1.5", 38, 1)),
                Some(Literal::Date("2026-01-01".to_owned())),
            ),
        ),
        (
            "null",
            "{timestamp: \"2026-01-01T00:00:00\"}",
            (
                None,
                Some(Literal::Timestamp {
                    value: "2026-01-01T00:00:00".to_owned(),
                    precision: 6,
                }),
            ),
        ),
    ] {
        let model = built(&bucket_model(lower, upper));
        let dimension = model.dimensions.values().next().expect("one Dimension");
        let DimensionType::Bucketed(body) = &dimension.dim_type else {
            panic!("the Dimension is bucketed");
        };
        let bucket = body.buckets.first().expect("one bucket");
        assert_eq!(
            (bucket.lower.clone(), bucket.upper.clone()),
            expected,
            "`{lower}` to `{upper}`"
        );
        let written = serde_yaml::to_string(&model).expect("the writer writes the model");
        let from_yaml: semstrait_model::SemanticModel =
            serde_yaml::from_str(&written).expect("the written YAML reads back");
        assert_eq!(from_yaml, model, "via:\n{written}");
    }
    let refusal = refused(&bucket_model("{bigint: 0}", "100"));
    assert!(
        matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }] if message.contains("tagged literal")),
        "`100`: {refusal:?}"
    );
    let refusal = refused(&bucket_model("{bigint: 0}", "{int: 100}"));
    assert_eq!(
        refusal,
        [ParseErrorKind::UnknownTag {
            tag: "lit.int".to_owned()
        }],
        "`{{int: 100}}`"
    );
}
