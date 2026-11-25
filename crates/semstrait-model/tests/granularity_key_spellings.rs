mod common;

use common::golden::{self, Golden};
use semstrait_common::diagnostic::Diagnostics;
use semstrait_model::{parse, ParseErrorKind};

const TEMPORAL_SHAPE_WITH_OLD_GRAIN_KEY: &str = "\
semantic_model:\n  name: m\n  dimensions:\n    - name: day\n      data_type: date\n      \
     type: categorical\n  datasets:\n    - name: sessions\n      extras:\n        \
     semantic_mapping: auto\n        temporal: {timeseries: {occurred_at: day}, grain: day}\n      \
     dimensions:\n        - ref: day\n";

const TEMPORAL_DIMENSION_WITH_OLD_GRAINS_KEY: &str = "\
semantic_model:\n  name: m\n  dimensions:\n    - name: day\n      data_type: date\n      \
     type: {temporal: {grains: [day, month]}}\n  datasets:\n    - name: sessions\n      \
     extras:\n        semantic_mapping: auto\n      dimensions:\n        - ref: day\n";

fn parse_errors(yaml: &str) -> Diagnostics<ParseErrorKind> {
    parse(yaml)
        .err()
        .unwrap_or_else(|| panic!("parse must fail for:\n{yaml}"))
}

#[test]
fn a_temporal_shape_with_old_grain_key_fails_to_parse() {
    temporal_shape_old_grain_key_refusal().assert_committed();
}

fn temporal_shape_old_grain_key_refusal() -> Golden {
    let errors = parse_errors(TEMPORAL_SHAPE_WITH_OLD_GRAIN_KEY);
    assert!(
        errors.iter().any(
            |d| matches!(&d.kind, ParseErrorKind::UnknownField { field, .. } if field == "grain")
        ),
        "expected UnknownField {{ field: \"grain\", .. }}, got {errors:?}"
    );
    Golden::message(
        "a_temporal_shape_with_old_grain_key_fails_to_parse",
        &errors,
    )
}

#[test]
fn a_temporal_dimension_with_old_grains_key_fails_to_parse() {
    temporal_dimension_old_grains_key_refusal().assert_committed();
}

fn temporal_dimension_old_grains_key_refusal() -> Golden {
    let errors = parse_errors(TEMPORAL_DIMENSION_WITH_OLD_GRAINS_KEY);
    assert!(
        errors.iter().any(
            |d| matches!(&d.kind, ParseErrorKind::UnknownField { field, .. } if field == "grains")
        ),
        "expected UnknownField {{ field: \"grains\", .. }}, got {errors:?}"
    );
    Golden::message(
        "a_temporal_dimension_with_old_grains_key_fails_to_parse",
        &errors,
    )
}

#[test]
fn generate_goldens() {
    golden::generate(|| {
        vec![
            temporal_shape_old_grain_key_refusal(),
            temporal_dimension_old_grains_key_refusal(),
        ]
    });
}
