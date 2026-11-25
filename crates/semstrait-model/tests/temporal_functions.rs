mod common;

use common::expression_model;
use common::golden::{self, Golden};
use semstrait_model::{parse, ParseErrorKind};

const PARSE_REFUSALS: [(&str, &str, &str, &str); 5] = [
    (
        "date_trunc_s_grain_as_a_column_is_refused_at_parse",
        "{date_trunc: [{col: g}, {col: ts}]}",
        "date_trunc",
        "`date_trunc`'s `granularity` takes a literal",
    ),
    (
        "date_trunc_to_the_second_is_refused_at_parse",
        "{date_trunc: [second, {col: ts}]}",
        "date_trunc",
        "`date_trunc`'s `granularity` has no value 'second'",
    ),
    (
        "extract_s_component_as_a_column_is_refused_at_parse",
        "{extract: [{col: c}, {col: ts}]}",
        "extract",
        "`extract`'s `component` takes a literal",
    ),
    (
        "extract_of_the_epoch_is_refused_at_parse",
        "{extract: [epoch, {col: ts}]}",
        "extract",
        "`extract`'s `component` has no value 'epoch'",
    ),
    (
        "extract_of_three_arguments_is_refused_at_parse",
        "{extract: [year, {col: ts}, {col: d}]}",
        "extract",
        "`extract` does not take 3 arguments",
    ),
];

fn refusal_goldens() -> Vec<Golden> {
    (PARSE_REFUSALS.iter())
        .map(|(case, expr, function, reason)| {
            let diagnostics = parse(&expression_model(expr)).expect_err("parse refuses the call");
            let kinds: Vec<_> = diagnostics.iter().map(|d| d.kind.clone()).collect();
            assert_eq!(
                kinds,
                [ParseErrorKind::CallRefused {
                    function: (*function).to_owned(),
                    reason: (*reason).to_owned(),
                }],
                "{case}"
            );
            Golden::message(case, &diagnostics)
        })
        .collect()
}

#[test]
fn each_fault_that_the_declaration_decides_is_refused_at_parse() {
    for golden in refusal_goldens() {
        golden.assert_committed();
    }
}

#[test]
fn generate_goldens() {
    golden::generate(refusal_goldens);
}
