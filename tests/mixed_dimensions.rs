//! Integration tests for mixed dimension scenarios
//!
//! Tests that the system correctly handles models where some dimensions
//! need JOINs and others are denormalized.

mod common;

use common::{count_joins, count_scans, has_join, load_fixture, run_pipeline};
use semstrait::QueryRequest;

#[test]
fn test_denormalized_only_no_join() {
    let schema = load_fixture("mixed_dimensions.yaml");

    // Query only denormalized dimensions (dates)
    let request = QueryRequest {
        model: "orders".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec!["order_total".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Denormalized dimension should NOT produce a JOIN
    assert!(
        !has_join(&plan),
        "Plan should NOT contain JOINs for denormalized dates"
    );
    assert_eq!(count_scans(&plan), 1, "Should scan only fact table");
}

#[test]
fn test_joined_only_produces_join() {
    let schema = load_fixture("mixed_dimensions.yaml");

    // Query only joined dimensions (customers.name)
    let request = QueryRequest {
        model: "orders".to_string(),
        rows: Some(vec!["customers.name".to_string()]),
        metrics: Some(vec!["order_total".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // customers.name requires a JOIN
    assert!(has_join(&plan), "Plan should contain a JOIN for customers");
    assert_eq!(count_joins(&plan), 1, "Should have exactly 1 JOIN");
    assert_eq!(
        count_scans(&plan),
        2,
        "Should scan fact table + customers dimension"
    );
}

#[test]
fn test_mixed_query() {
    let schema = load_fixture("mixed_dimensions.yaml");

    // Query both denormalized (dates.year) and joined (customers.name)
    let request = QueryRequest {
        model: "orders".to_string(),
        rows: Some(vec![
            "dates.year".to_string(),
            "customers.name".to_string(),
        ]),
        metrics: Some(vec!["order_total".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Should have exactly 1 JOIN (only for customers, not dates)
    assert!(has_join(&plan), "Plan should contain a JOIN");
    assert_eq!(
        count_joins(&plan),
        1,
        "Should have exactly 1 JOIN (customers only)"
    );
    assert_eq!(
        count_scans(&plan),
        2,
        "Should scan fact + customers (dates is denormalized)"
    );
}

// TODO: Add tests for:
// - Attribute-based join detection (key attribute in/out)
// - Same dimension partially denormalized, partially joined
