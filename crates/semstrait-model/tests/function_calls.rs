mod common;

use common::{built, call, expression_model, parsed_expression, refused_expression, untyped};
use semstrait_common::diagnostic::Diagnose;
use semstrait_model::expr::{col, field};
use semstrait_model::{ParseErrorKind, SemanticModel, UnresolvedSemanticExpr};
use semstrait_sql::expr::{Expr, FunctionRef, Literal, NoAggregate};
use semstrait_sql::functions::registry;

fn text(value: &str) -> UnresolvedSemanticExpr {
    untyped(Literal::String(value.to_owned()))
}

#[test]
fn each_catalog_function_parses_as_a_call_of_its_declaration() {
    let pattern = "^([A-Z]{2})_";
    let cases = [
        (
            "{left: [name, {lit: 3}]}",
            call(
                "left",
                vec![field("name").expect("a name"), untyped(Literal::BigInt(3))],
            ),
        ),
        (
            "{concat_ws: [{lit: '-'}, a, b]}",
            call(
                "concat_ws",
                vec![
                    text("-"),
                    field("a").expect("a name"),
                    field("b").expect("a name"),
                ],
            ),
        ),
        (
            "{regexp_like: [{col: campaign}, {lit: '^UK_'}]}",
            call(
                "regexp_like",
                vec![col("campaign").expect("a name"), text("^UK_")],
            ),
        ),
        (
            "{regexp_match: [{col: campaign}, {lit: '^([A-Z]{2})_'}]}",
            call(
                "regexp_match",
                vec![col("campaign").expect("a name"), text(pattern)],
            ),
        ),
        (
            "{regexp_extract: [{col: campaign}, {lit: '^([A-Z]{2})_'}, {lit: 1}]}",
            call(
                "regexp_extract",
                vec![
                    col("campaign").expect("a name"),
                    text(pattern),
                    untyped(Literal::BigInt(1)),
                ],
            ),
        ),
        (
            "{date_trunc: [{lit: month}, {col: ordered_at}]}",
            call(
                "date_trunc",
                vec![text("month"), col("ordered_at").expect("a name")],
            ),
        ),
        (
            "{extract: [{lit: year}, {col: ordered_at}]}",
            call(
                "extract",
                vec![text("year"), col("ordered_at").expect("a name")],
            ),
        ),
    ];
    for (expr, tree) in cases {
        assert_eq!(parsed_expression(expr), tree, "`{expr}`");
    }
}

#[test]
fn an_enum_argument_is_its_value_written_bare_or_as_a_string_literal() {
    for (bare, literal, tree) in [
        (
            "{extract: [year, {col: order_date}]}",
            "{extract: [{lit: year}, {col: order_date}]}",
            call(
                "extract",
                vec![text("year"), col("order_date").expect("a name")],
            ),
        ),
        (
            "{extract: [day_of_week, order_date]}",
            "{extract: [{lit: day_of_week}, order_date]}",
            call(
                "extract",
                vec![text("day_of_week"), field("order_date").expect("a name")],
            ),
        ),
        (
            "{date_trunc: [month, {col: order_date}]}",
            "{date_trunc: [{lit: month}, {col: order_date}]}",
            call(
                "date_trunc",
                vec![text("month"), col("order_date").expect("a name")],
            ),
        ),
        (
            "{date_trunc: [week, {col: order_date}, sunday]}",
            "{date_trunc: [{lit: week}, {col: order_date}, {lit: sunday}]}",
            call(
                "date_trunc",
                vec![
                    text("week"),
                    col("order_date").expect("a name"),
                    text("sunday"),
                ],
            ),
        ),
        (
            "{date_diff: [day, {col: ordered_at}, {col: shipped_at}]}",
            "{date_diff: [{lit: day}, {col: ordered_at}, {col: shipped_at}]}",
            call(
                "date_diff",
                vec![
                    text("day"),
                    col("ordered_at").expect("a name"),
                    col("shipped_at").expect("a name"),
                ],
            ),
        ),
    ] {
        assert_eq!(parsed_expression(bare), tree, "`{bare}`");
        assert_eq!(parsed_expression(literal), tree, "`{literal}`");
    }
}

#[test]
fn a_bare_value_at_a_constant_parameter_is_a_literal_and_at_a_value_parameter_a_leaf() {
    let one = || untyped(Literal::BigInt(1));
    let flags = call(
        "regexp_instr",
        vec![
            col("campaign").expect("a name"),
            text("^uk_"),
            one(),
            one(),
            text("i"),
        ],
    );
    for expr in [
        "{regexp_instr: [{col: campaign}, {lit: '^uk_'}, 1, 1, i]}",
        "{regexp_instr: [{col: campaign}, {lit: '^uk_'}, 1, 1, {lit: i}]}",
    ] {
        assert_eq!(parsed_expression(expr), flags, "`{expr}`");
    }
    assert_eq!(
        parsed_expression("{regexp_like: [{col: campaign}, pattern]}"),
        call(
            "regexp_like",
            vec![
                col("campaign").expect("a name"),
                field("pattern").expect("a name")
            ]
        )
    );
}

#[test]
fn an_enum_value_that_the_parameter_does_not_list_is_refused_at_parse() {
    for (expr, function, reason) in [
        (
            "{extract: [yeer, {col: order_date}]}",
            "extract",
            "`extract`'s `component` has no value 'yeer'",
        ),
        (
            "{extract: [{lit: yeer}, {col: order_date}]}",
            "extract",
            "`extract`'s `component` has no value 'yeer'",
        ),
        (
            "{date_trunc: [mnth, {col: order_date}]}",
            "date_trunc",
            "`date_trunc`'s `granularity` has no value 'mnth'",
        ),
        (
            "{date_diff: [dy, {col: ordered_at}, {col: shipped_at}]}",
            "date_diff",
            "`date_diff`'s `part` has no value 'dy'",
        ),
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: function.to_owned(),
                reason: reason.to_owned(),
            }],
            "`{expr}`"
        );
    }
}

#[test]
fn a_regexp_extract_group_above_its_literal_patterns_groups_is_refused_at_parse() {
    let above_two = "`regexp_extract`'s `pattern` has no group 3: its last group is 2";
    for (expr, reason) in [
        (
            "{regexp_extract: [v, {lit: '(a)(b)'}, {lit: 3}]}",
            above_two,
        ),
        (
            "{regexp_extract: [v, {lit: {string: '(a)(b)'}}, {lit: {integer: 3}}]}",
            above_two,
        ),
        (
            "{regexp_extract: [v, {lit: '(?:a)(b)'}, {lit: 2}]}",
            "`regexp_extract`'s `pattern` has no group 2: its last group is 1",
        ),
        (
            "{regexp_extract: [v, {lit: ab}, {lit: 1}]}",
            "`regexp_extract`'s `pattern` has no group 1: its last group is 0",
        ),
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: "regexp_extract".to_owned(),
                reason: reason.to_owned(),
            }],
            "`{expr}`"
        );
    }
}

#[test]
fn a_regexp_extract_group_below_0_is_refused_at_parse_with_any_pattern() {
    for expr in [
        "{regexp_extract: [v, {lit: '(a)'}, {lit: -1}]}",
        "{regexp_extract: [v, {lit: '(a)'}, {lit: {integer: -1}}]}",
        "{regexp_extract: [v, {col: pattern}, {lit: -1}]}",
        "{regexp_extract: [v, pattern, {lit: -1}]}",
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: "regexp_extract".to_owned(),
                reason: "`regexp_extract` has no group -1: a group is not below 0".to_owned(),
            }],
            "`{expr}`"
        );
    }
}

#[test]
fn a_regexp_extract_group_of_its_pattern_or_of_a_pattern_that_is_no_literal_parses() {
    let group = |group| untyped(Literal::BigInt(group));
    for (expr, tree) in [
        (
            "{regexp_extract: [v, {lit: '(a)(b)'}, {lit: 2}]}",
            call(
                "regexp_extract",
                vec![field("v").expect("a name"), text("(a)(b)"), group(2)],
            ),
        ),
        (
            "{regexp_extract: [v, {lit: '(a)(b)'}, {lit: 0}]}",
            call(
                "regexp_extract",
                vec![field("v").expect("a name"), text("(a)(b)"), group(0)],
            ),
        ),
        (
            "{regexp_extract: [v, {col: pattern}, {lit: 7}]}",
            call(
                "regexp_extract",
                vec![
                    field("v").expect("a name"),
                    col("pattern").expect("a name"),
                    group(7),
                ],
            ),
        ),
        (
            "{regexp_extract: [v, pattern, {lit: 7}]}",
            call(
                "regexp_extract",
                vec![
                    field("v").expect("a name"),
                    field("pattern").expect("a name"),
                    group(7),
                ],
            ),
        ),
    ] {
        assert_eq!(parsed_expression(expr), tree, "`{expr}`");
    }
}

#[test]
fn a_call_with_a_wrong_number_of_arguments_is_refused_at_parse() {
    for (expr, function, reason) in [
        (
            "{upper: [name, 2]}",
            "upper",
            "`upper` does not take 2 arguments",
        ),
        ("{left: [name]}", "left", "`left` takes two arguments"),
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: function.to_owned(),
                reason: reason.to_owned(),
            }],
            "`{expr}`"
        );
    }
}

#[test]
fn a_call_of_the_fewest_arguments_parses_and_one_fewer_is_refused() {
    for (name, fewest, takes) in [
        ("greatest", 1, "one argument or more"),
        ("least", 1, "one argument or more"),
        ("coalesce", 2, "two arguments or more"),
        ("concat", 2, "two arguments or more"),
        ("concat_ws", 2, "two arguments or more"),
    ] {
        let args = |count: usize| &["a", "b"][..count];
        assert_eq!(
            parsed_expression(&format!("{{{name}: [{}]}}", args(fewest).join(", "))),
            call(
                name,
                (args(fewest).iter())
                    .map(|arg| field(*arg).expect("a name"))
                    .collect()
            ),
            "{name}"
        );
        assert_eq!(
            refused_expression(&format!("{{{name}: [{}]}}", args(fewest - 1).join(", "))),
            [ParseErrorKind::CallRefused {
                function: name.to_owned(),
                reason: format!("`{name}` takes {takes}"),
            }],
            "{name}"
        );
    }
}

#[test]
fn a_misspelt_function_is_refused_at_parse() {
    assert_eq!(
        refused_expression("{uper: name}"),
        [ParseErrorKind::UnknownTag {
            tag: "uper".to_owned()
        }]
    );
}

#[test]
fn an_aggregate_function_in_a_dimension_expression_is_refused() {
    for tag in ["sum", "count", "every"] {
        assert_eq!(
            refused_expression(&format!("{{{tag}: amount}}")),
            [ParseErrorKind::AggregateInRowExpr {
                function: tag.to_owned()
            }],
            "`{tag}`"
        );
    }
}

#[test]
fn a_call_that_holds_a_call_that_it_cannot_hold_is_refused_at_parse() {
    let window_in_window = || ParseErrorKind::CallRefused {
        function: "first_value".to_owned(),
        reason: "`first_value` holds the window call `first_value`, and a window call holds \
                 no window call"
            .to_owned(),
    };
    let window_message = "the call of `first_value` is refused: `first_value` holds the window \
                          call `first_value`, and a window call holds no window call";
    for (expr, kind, message) in [
        (
            "{first_value: {first_value: {col: x}}}",
            window_in_window(),
            window_message,
        ),
        (
            "{first_value: {abs: {first_value: {col: x}}}}",
            window_in_window(),
            window_message,
        ),
        (
            "{sum: {sum: {col: x}}}",
            ParseErrorKind::AggregateInRowExpr {
                function: "sum".to_owned(),
            },
            "aggregate tag `sum` is not allowed in an expression body; aggregation is stated \
             through `agg:`",
        ),
    ] {
        let refusal = refused_expression(expr);
        assert_eq!(refusal, [kind], "`{expr}`");
        assert_eq!(refusal[0].message(), message, "`{expr}`");
    }
}

#[test]
fn coalesce_and_null_if_parse_as_calls_of_coalesce_and_nullif() {
    let zero = || untyped(Literal::BigInt(0));
    let nullif = call("nullif", vec![field("a").expect("a name"), zero()]);
    for (expr, tree) in [
        (
            "{coalesce: [a, {lit: 0}]}",
            call("coalesce", vec![field("a").expect("a name"), zero()]),
        ),
        ("{nullif: [a, {lit: 0}]}", nullif.clone()),
        ("{null_if: [a, {lit: 0}]}", nullif.clone()),
        ("{null_if: {left: a, right: {lit: 0}}}", nullif),
    ] {
        assert_eq!(parsed_expression(expr), tree, "`{expr}`");
    }
}

#[test]
fn a_mapping_key_that_names_no_parameter_is_refused() {
    let nullif = "no parameter is named `bogus`; the parameters are `left`, `right`";
    for (expr, function, reason) in [
        (
            "{regexp_instr: {value: {col: c}, pattern: {lit: 'x'}, grop: 1}}",
            "regexp_instr",
            "no parameter is named `grop`; the parameters are `value`, `pattern`, `start`, \
             `occurrence`, `flags`, `group`",
        ),
        ("{null_if: {left: a, bogus: b}}", "null_if", nullif),
        (
            "{null_if: {left: a, right: b, bogus: c}}",
            "null_if",
            nullif,
        ),
        (
            "{null_if: {1: a, 2: b}}",
            "null_if",
            "no parameter is named `1`; the parameters are `left`, `right`",
        ),
        (
            "{pi: {a: {lit: 1}, b: {lit: 2}}}",
            "pi",
            "no parameter is named `a`; the function has none",
        ),
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: function.to_owned(),
                reason: reason.to_owned(),
            }],
            "`{expr}`"
        );
    }
}

#[test]
fn a_one_entry_mapping_names_the_parameter_that_its_key_names() {
    let upper = call("upper", vec![field("campaign").expect("a name")]);
    for expr in ["{upper: {value: campaign}}", "{upper: [campaign]}"] {
        assert_eq!(parsed_expression(expr), upper, "`{expr}`");
    }
    assert_eq!(
        refused_expression("{round: {scale: 2}}"),
        [ParseErrorKind::CallRefused {
            function: "round".to_owned(),
            reason: "the mapping leaves out `value`, which the call writes".to_owned(),
        }]
    );
}

#[test]
fn a_one_entry_mapping_whose_key_is_a_tag_or_a_function_is_one_argument() {
    assert_eq!(
        parsed_expression("{upper: {col: campaign}}"),
        call("upper", vec![col("campaign").expect("a name")])
    );
    assert_eq!(
        refused_expression("{null_if: {left: a}}"),
        [ParseErrorKind::CallRefused {
            function: "left".to_owned(),
            reason: "`left` takes two arguments".to_owned(),
        }]
    );
}

#[test]
fn a_mapping_names_parameters_in_any_order() {
    for (mapping, list) in [
        (
            "{regexp_instr: {pattern: {lit: 'x'}, value: {col: c}}}",
            "{regexp_instr: [{col: c}, {lit: 'x'}]}",
        ),
        ("{null_if: {right: b, left: a}}", "{null_if: [a, b]}"),
    ] {
        assert_eq!(
            parsed_expression(mapping),
            parsed_expression(list),
            "`{mapping}`"
        );
    }
}

#[test]
fn a_skipped_optional_takes_its_declared_default() {
    for (mapping, list) in [
        (
            "{regexp_instr: {value: {col: c}, pattern: {lit: 'x'}, group: 1}}",
            "{regexp_instr: [{col: c}, {lit: 'x'}, {lit: {integer: 1}}, {lit: {integer: 1}}, \
             {lit: {string: ''}}, 1]}",
        ),
        (
            "{regexp_instr: {value: {col: c}, pattern: {lit: '^uk_'}, flags: i}}",
            "{regexp_instr: [{col: c}, {lit: '^uk_'}, {lit: {integer: 1}}, {lit: {integer: 1}}, \
             i]}",
        ),
    ] {
        assert_eq!(
            parsed_expression(mapping),
            parsed_expression(list),
            "`{mapping}`"
        );
    }
    let mapping = "{regexp_instr: {value: {col: c}, pattern: {lit: 'x'}, group: 1}}";
    assert_eq!(
        parsed_expression(mapping),
        call(
            "regexp_instr",
            vec![
                col("c").expect("a name"),
                text("x"),
                Expr::Literal(Literal::Integer(1)),
                Expr::Literal(Literal::Integer(1)),
                Expr::Literal(Literal::String(String::new())),
                untyped(Literal::BigInt(1)),
            ]
        )
    );
    let model = built(&expression_model(mapping));
    let written = serde_yaml::to_string(&model).expect("the writer writes the model");
    let back: SemanticModel = serde_yaml::from_str(&written).expect("the YAML reads back");
    assert_eq!(back, model);
}

#[test]
fn a_mapping_that_skips_lags_offset_is_refused_as_its_list_form_is() {
    let refusal = refused_expression("{lag: {value: {col: x}, default: 0}}");
    assert_eq!(
        refusal,
        refused_expression("{lag: [{col: x}, {lit: {integer: 1}}, 0]}")
    );
    assert_eq!(
        refusal,
        [ParseErrorKind::CallRefused {
            function: "lag".to_owned(),
            reason: "`lag` requires its ORDER BY clause".to_owned(),
        }]
    );
}

#[test]
fn a_mapping_that_leaves_a_required_tail_without_an_argument_is_refused_as_its_list_form_is() {
    let refusal = refused_expression("{concat_ws: {separator: {lit: '-'}}}");
    assert_eq!(refusal, refused_expression("{concat_ws: [{lit: '-'}]}"));
    assert_eq!(
        refusal,
        [ParseErrorKind::CallRefused {
            function: "concat_ws".to_owned(),
            reason: "`concat_ws` takes two arguments or more".to_owned(),
        }]
    );
}

#[test]
fn a_named_parameter_that_an_earlier_argument_rules_out_is_refused() {
    for expr in [
        "{date_trunc: {granularity: day, value: {col: ts}, week_start: {lit: monday}}}",
        "{date_trunc: {granularity: day, value: {col: ts}, timezone: {lit: UTC}, week_start: {lit: \
         monday}}}",
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: "date_trunc".to_owned(),
                reason: "the mapping names `week_start`, which the call does not write".to_owned(),
            }],
            "`{expr}`"
        );
    }
    let date_trunc = registry().get("date_trunc").expect("a catalog function");
    let date_trunc = FunctionRef::<NoAggregate>::try_from(date_trunc).expect("a scalar function");
    for (mapping, list, parameter, value) in [
        (
            "{date_trunc: {granularity: week, value: {col: ts}, week_start: {lit: monday}}}",
            "{date_trunc: [week, {col: ts}, {lit: monday}]}",
            "week_start",
            "monday",
        ),
        (
            "{date_trunc: {granularity: day, value: {col: ts}, timezone: {lit: UTC}}}",
            "{date_trunc: [day, {col: ts}, {lit: UTC}]}",
            "timezone",
            "UTC",
        ),
    ] {
        let tree = parsed_expression(mapping);
        assert_eq!(tree, parsed_expression(list), "`{mapping}`");
        let Expr::Function(call) = &tree else {
            panic!("a call: {tree:?}");
        };
        let bound = date_trunc
            .parameters(call.args())
            .expect("the arguments bind");
        let names: Vec<_> = bound.iter().map(|parameter| parameter.name).collect();
        assert_eq!(names, ["granularity", "value", parameter], "`{mapping}`");
        assert_eq!(call.args().last(), Some(&text(value)), "`{mapping}`");
    }
}

#[test]
fn a_week_bucket_that_leaves_its_week_start_out_takes_monday() {
    let monday = || Expr::Literal(Literal::String("monday".to_owned()));
    let ts = || col("ts").expect("a name");
    for (expr, args) in [
        (
            "{date_trunc: [week, {col: ts}]}",
            vec![text("week"), ts(), monday()],
        ),
        (
            "{date_trunc: {granularity: week, value: {col: ts}}}",
            vec![text("week"), ts(), monday()],
        ),
        (
            "{date_trunc: {granularity: week, value: {col: ts}, timezone: {lit: UTC}}}",
            vec![text("week"), ts(), text("UTC"), monday()],
        ),
        (
            "{date_trunc: [month, {col: ts}]}",
            vec![text("month"), ts()],
        ),
    ] {
        assert_eq!(
            parsed_expression(expr),
            call("date_trunc", args),
            "`{expr}`"
        );
    }
    let model = built(&expression_model("{date_trunc: [week, {col: ts}]}"));
    let written = serde_yaml::to_string(&model).expect("the writer writes the model");
    let back: SemanticModel = serde_yaml::from_str(&written).expect("the YAML reads back");
    assert_eq!(back, model);
}

#[test]
fn a_named_parameter_that_binds_to_an_earlier_one_is_refused() {
    assert_eq!(
        refused_expression(
            "{date_trunc: {granularity: {col: g}, value: {col: ts}, week_start: {lit: monday}}}"
        ),
        [ParseErrorKind::CallRefused {
            function: "date_trunc".to_owned(),
            reason: "the mapping names `week_start`, which the call does not write".to_owned(),
        }]
    );
}

#[test]
fn a_trailing_optional_that_no_key_names_stays_absent() {
    assert_eq!(
        parsed_expression("{round: {value: {col: x}}}"),
        call("round", vec![col("x").expect("a name")])
    );
    assert_eq!(
        parsed_expression("{regexp_instr: {value: {col: c}, pattern: {lit: 'x'}}}"),
        call("regexp_instr", vec![col("c").expect("a name"), text("x")])
    );
}

#[test]
fn a_mapping_that_leaves_out_a_required_parameter_is_refused() {
    for (expr, function, parameter) in [
        (
            "{regexp_instr: {pattern: {lit: 'x'}, group: 1}}",
            "regexp_instr",
            "value",
        ),
        (
            "{substring: {value: a, length: {lit: 3}}}",
            "substring",
            "start",
        ),
    ] {
        assert_eq!(
            refused_expression(expr),
            [ParseErrorKind::CallRefused {
                function: function.to_owned(),
                reason: format!("the mapping leaves out `{parameter}`, which the call writes"),
            }],
            "`{expr}`"
        );
    }
}

#[test]
fn a_mapping_that_names_a_parameter_twice_is_refused() {
    let refusal = refused_expression("{null_if: {left: a, right: b, left: c}}");
    assert!(
        matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }] if message.contains("duplicate entry")),
        "{refusal:?}"
    );
}

#[test]
fn a_window_functions_tag_builds_its_call() {
    assert_eq!(
        parsed_expression("{row_number: []}"),
        call("row_number", vec![])
    );
}

#[test]
fn the_writer_writes_each_declarations_name() {
    for (expr, name, alias) in [
        ("{ceiling: amount}", "ceil:", "ceiling"),
        ("{char_length: name}", "length:", "char_length"),
        (
            "{null_if: {left: a, right: {lit: 0}}}",
            "nullif:",
            "null_if",
        ),
    ] {
        let model = built(&expression_model(expr));
        let written = serde_yaml::to_string(&model).expect("the writer writes the model");
        assert!(
            written.contains(name) && !written.contains(alias),
            "`{expr}`: {written}"
        );
        let back: SemanticModel = serde_yaml::from_str(&written).expect("the YAML reads back");
        assert_eq!(back, model, "`{expr}`");
    }
}
