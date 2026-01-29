//! Integration tests for metric expressions
//!
//! Tests that metrics (derived calculations) are correctly compiled.

mod common;

use common::{load_fixture, run_pipeline};
use semstrait::QueryRequest;

#[test]
fn test_simple_metric_reference() {
    let schema = load_fixture("metrics.yaml");

    // Query a metric that just references a measure
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec!["total_revenue".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

#[test]
fn test_arithmetic_metric() {
    let schema = load_fixture("metrics.yaml");

    // Query a metric with division: avg_unit_price = revenue / quantity
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec!["avg_unit_price".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

#[test]
fn test_nested_metric() {
    let schema = load_fixture("metrics.yaml");

    // Query margin = (revenue - cost) / revenue (nested expressions)
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec!["products.category".to_string()]),
        metrics: Some(vec!["margin".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

#[test]
fn test_multiple_metrics_with_shared_dependency() {
    let schema = load_fixture("metrics.yaml");

    // Query both a pass-through metric and a derived metric
    // Both depend on the underlying revenue measure
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec!["revenue".to_string(), "profit".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

#[test]
fn test_multiple_metrics() {
    let schema = load_fixture("metrics.yaml");

    // Query multiple metrics at once
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec![
            "profit".to_string(),
            "margin".to_string(),
            "avg_unit_price".to_string(),
        ]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

// TODO: Add tests for:
// - Verify the actual expressions in the Substrait plan
// - CASE WHEN expressions in metrics
// - Metrics that reference other metrics (if supported)
// - Error handling for undefined measure references
