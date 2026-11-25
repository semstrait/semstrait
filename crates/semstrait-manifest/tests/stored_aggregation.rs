mod common;

use common::compile_model;
use semstrait_manifest::SemanticManifest;
use semstrait_model::{parse, SemanticModel};

fn model(agg: &str, data_type: &str) -> SemanticModel {
    let yaml = format!(
        r#"
semantic_model:
  name: aggregates
  measures:
    - name: spend
      data_type: {data_type}
      agg: {agg}
  datasets:
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
      measures:
        - ref: spend
"#
    );
    parse(&yaml)
        .unwrap_or_else(|diagnostics| panic!("parse refused `agg: {agg}`: {diagnostics:?}"))
        .build()
        .unwrap_or_else(|diagnostics| panic!("build refused `agg: {agg}`: {diagnostics:?}"))
        .0
}

fn stored_content(manifest: &SemanticManifest) -> serde_json::Value {
    let definition = manifest
        .semantic_definitions
        .by_name("spend")
        .expect("the Measure is registered");
    serde_json::to_value(&definition.role_definition).expect("the role body serializes")["content"]
        .clone()
}

fn stored_aggregation(manifest: &SemanticManifest) -> serde_json::Value {
    stored_content(manifest)["aggregation"].clone()
}

#[tokio::test]
async fn each_of_todays_aggregations_stores_todays_bytes() {
    for agg in ["sum", "avg", "count", "min", "max", "median"] {
        let manifest = compile_model(model(agg, "double")).await;
        assert_eq!(stored_aggregation(&manifest), agg, "`agg: {agg}`");
        assert_eq!(
            stored_content(&manifest).get("distinct"),
            None,
            "`agg: {agg}` leaves distinct out when false"
        );
        let json = serde_json::to_string(&manifest).expect("the manifest serializes");
        let back: SemanticManifest = serde_json::from_str(&json).expect("the manifest reads");
        assert_eq!(back, manifest, "`agg: {agg}` reads back");
    }

    let manifest = compile_model(model("count_distinct", "double")).await;
    assert_eq!(
        stored_aggregation(&manifest),
        "count",
        "`agg: count_distinct` lowers to the function `count`"
    );
    assert_eq!(
        stored_content(&manifest)["distinct"],
        serde_json::Value::Bool(true),
        "`agg: count_distinct` sets the distinct flag"
    );
    let json = serde_json::to_string(&manifest).expect("the manifest serializes");
    assert!(
        json.contains(r#""aggregation":"count","distinct":true"#),
        "got {json}"
    );
    let back: SemanticManifest = serde_json::from_str(&json).expect("the manifest reads");
    assert_eq!(back, manifest, "`agg: count_distinct` reads back");
}

#[tokio::test]
async fn a_catalog_aggregate_is_stored_by_its_canonical_name() {
    for (agg, data_type, stored) in [
        ("every", "boolean", "bool_and"),
        ("stddev_samp", "double", "stddev_samp"),
    ] {
        let manifest = compile_model(model(agg, data_type)).await;
        assert_eq!(stored_aggregation(&manifest), stored, "`agg: {agg}`");
    }
}

#[tokio::test]
async fn reading_refuses_an_aggregation_that_is_no_canonical_name() {
    let manifest = compile_model(model("sum", "double")).await;
    let json = serde_json::to_string(&manifest).expect("the manifest serializes");
    for name in ["nope", "every"] {
        let stored = json.replace(
            r#""aggregation":"sum""#,
            &format!(r#""aggregation":"{name}""#),
        );
        assert_ne!(stored, json, "the fixture stores `sum`");
        let error = serde_json::from_str::<SemanticManifest>(&stored)
            .expect_err("reading refuses the name");
        assert!(error.to_string().contains(name), "`{name}`: {error}");
    }
}

#[tokio::test]
async fn reading_takes_every_catalog_aggregate_s_canonical_name() {
    let manifest = compile_model(model("sum", "double")).await;
    let json = serde_json::to_string(&manifest).expect("the manifest serializes");
    for name in ["corr", "string_agg", "percentile_cont"] {
        let stored = json.replace(
            r#""aggregation":"sum""#,
            &format!(r#""aggregation":"{name}""#),
        );
        let back = serde_json::from_str::<SemanticManifest>(&stored)
            .unwrap_or_else(|error| panic!("reading refuses `{name}`: {error}"));
        assert_eq!(stored_aggregation(&back), name, "`{name}`");
    }
}
