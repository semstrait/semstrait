mod common;

use common::{built, refused};
use semstrait_model::{ParseErrorKind, SemanticModel};

fn model(agg: &str, metric_agg: Option<&str>) -> String {
    let metric_agg = metric_agg.map_or_else(String::new, |agg| format!("\n      agg: {agg}"));
    format!(
        r#"
semantic_model:
  name: aggregates
  measures:
    - name: spend
      data_type: double
      agg: {agg}
  metrics:
    - name: spend_total
      data_type: double
      expr: spend{metric_agg}
  datasets:
    - name: orders
      measures:
        - ref: spend
      metrics:
        - ref: spend_total
"#
    )
}

fn written_measure_agg(model: &SemanticModel) -> serde_json::Value {
    let measure = model.measures.values().next().expect("one Measure");
    serde_json::to_value(measure.agg).expect("the writer writes `agg:`")
}

#[test]
fn an_agg_that_names_a_catalog_aggregate_parses() {
    for (agg, written) in [
        ("sum", "sum"),
        ("count", "count"),
        ("every", "bool_and"),
        ("approx_count_distinct", "approx_count_distinct"),
        ("stddev_samp", "stddev_samp"),
    ] {
        let model = built(&model(agg, None));
        assert_eq!(written_measure_agg(&model), written, "`agg: {agg}`");
    }
}

#[test]
fn an_agg_whose_simplest_call_does_not_fit_its_declaration_is_refused() {
    for (agg, reason) in [
        ("corr", "`corr` takes two arguments"),
        ("string_agg", "`string_agg` takes two arguments"),
        (
            "percentile_cont",
            "`percentile_cont`'s `fraction` takes a literal",
        ),
    ] {
        assert_eq!(
            refused(&model(agg, None)),
            [ParseErrorKind::CallRefused {
                function: agg.to_owned(),
                reason: reason.to_owned(),
            }],
            "`agg: {agg}`"
        );
    }
}

#[test]
fn an_agg_that_names_no_catalog_aggregate_is_refused() {
    for agg in ["std_dev", "variance", "upper", "nope"] {
        assert_eq!(
            refused(&model(agg, None)),
            [ParseErrorKind::UnknownAggregate {
                name: agg.to_owned(),
            }],
            "`agg: {agg}`"
        );
    }
}

#[test]
fn a_metric_agg_reads_as_a_measure_agg() {
    let every = built(&model("sum", Some("every")));
    let metric = every.metrics.values().next().expect("one Metric");
    assert_eq!(
        serde_json::to_value(metric.agg).expect("the writer writes `agg:`"),
        "bool_and"
    );
    for (agg, kind) in [
        (
            "corr",
            ParseErrorKind::CallRefused {
                function: "corr".to_owned(),
                reason: "`corr` takes two arguments".to_owned(),
            },
        ),
        (
            "std_dev",
            ParseErrorKind::UnknownAggregate {
                name: "std_dev".to_owned(),
            },
        ),
    ] {
        assert_eq!(
            refused(&model("sum", Some(agg))),
            [kind],
            "the Metric's `agg: {agg}`"
        );
    }
}

#[test]
fn an_alias_is_written_as_its_canonical_name_and_round_trips() {
    let every = built(&model("every", Some("every")));
    let yaml = serde_yaml::to_string(&every).expect("the writer writes YAML");
    assert_eq!(yaml.matches("agg: bool_and").count(), 2, "{yaml}");
    let from_yaml: SemanticModel = serde_yaml::from_str(&yaml).expect("the YAML reads back");
    let json = serde_json::to_string(&every).expect("the writer writes JSON");
    let from_json: SemanticModel = serde_json::from_str(&json).expect("the JSON reads back");
    assert_eq!((&from_yaml, &from_json), (&every, &every));
}

#[test]
fn count_distinct_is_count_with_distinct() {
    let model = built(&model("count_distinct", None));
    let agg = model.measures.values().next().expect("one Measure").agg;
    assert_eq!(
        (agg.function().declarations.canonical.name, agg.distinct()),
        ("count", true)
    );
    assert_eq!(written_measure_agg(&model), "count_distinct");
}
