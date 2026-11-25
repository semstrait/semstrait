mod common;

use common::golden::{self, Golden};
use semstrait_common::diagnostic::Diagnostics;
use semstrait_model::{parse, ParseErrorKind};

const PROFIT_MARGIN_WITH_ADDITIVITY: &str = "semantic_model:\n  name: m\n  metrics:\n    \
     - name: profit_margin\n      data_type: decimal(18,4)\n      expr: revenue\n      \
     additivity: full\n";

fn parse_errors(yaml: &str) -> Diagnostics<ParseErrorKind> {
    parse(yaml)
        .err()
        .unwrap_or_else(|| panic!("parse must fail for:\n{yaml}"))
}

#[test]
fn a_metric_declaring_additivity_fails_to_parse() {
    additivity_on_metric_refusal().assert_committed();
}

fn additivity_on_metric_refusal() -> Golden {
    let errors = parse_errors(PROFIT_MARGIN_WITH_ADDITIVITY);
    assert!(
        errors.iter().any(|d| matches!(
            &d.kind,
            ParseErrorKind::UnknownField { field, parent }
                if field == "additivity" && parent == "Metric"
        )),
        "expected UnknownField {{ field: \"additivity\", parent: \"Metric\" }}, got {errors:?}"
    );
    Golden::message("a_metric_declaring_additivity_fails_to_parse", &errors)
}

#[test]
fn generate_goldens() {
    golden::generate(|| vec![additivity_on_metric_refusal()]);
}
