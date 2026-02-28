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

#[test]
fn test_meta_attributes_with_metrics() {
    let schema = load_fixture("metrics.yaml");

    // Query with virtual _dataset attributes alongside regular dimensions and metrics
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec![
            "dates.year".to_string(),
            "_dataset.datasetGroup".to_string(),
        ]),
        metrics: Some(vec!["total_revenue".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Meta attributes with metrics should succeed");
    
    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

#[test]
fn test_meta_attributes_only_with_metrics() {
    let schema = load_fixture("metrics.yaml");

    // Query with only _dataset attributes (no physical dimensions) and metrics
    let request = QueryRequest {
        model: "financial".to_string(),
        rows: Some(vec![
            "_dataset.model".to_string(),
            "_dataset.datasetGroup".to_string(),
        ]),
        metrics: Some(vec!["revenue".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Meta-only with metrics should succeed");
    
    // Plan should compile successfully
    assert!(!plan.relations.is_empty());
}

#[test]
fn test_meta_with_metrics_multiple_datasetgroups_requires_model_metric() {
    // Test model with virtual _dataset dimension at model level
    // Virtual dimensions are implicitly conformed, so querying _dataset.datasetGroup
    // triggers the UNION path, which requires model-level metrics
    let schema = load_fixture("marketing.yaml");

    // Query with _dataset.datasetGroup + clicks metric
    // "clicks" exists as a measure in datasetGroups but NOT as a model-level metric
    let request = QueryRequest {
        model: "-ObDoDFVQGxxCGa5vw_Z".to_string(),
        rows: Some(vec![
            "_dataset.datasetGroup".to_string(),
        ]),
        metrics: Some(vec!["clicks".to_string()]),
        ..Default::default()
    };

    let result = run_pipeline(&schema, &request);
    
    // Should fail because:
    // 1. Virtual _dataset is implicitly conformed → takes UNION path
    // 2. UNION path requires model-level metrics
    // 3. "clicks" is not a model-level metric (only a datasetGroup measure)
    assert!(result.is_err(), "Expected error when metric not defined at model level");
    let err = result.unwrap_err();
    assert!(
        err.contains("MetricNotFound") || err.contains("clicks"),
        "Expected MetricNotFound error, got: {}", 
        err
    );
}
