//! Integration tests for denormalized (flat) tables
//!
//! Tests that degenerate dimensions (no join spec) do NOT produce JOINs.

mod common;

use common::{count_scans, has_join, load_fixture, run_pipeline};
use semstrait::QueryRequest;

#[test]
fn test_degenerate_dimension_no_join() {
    let schema = load_fixture("denormalized.yaml");

    let request = QueryRequest {
        model: "events".to_string(),
        rows: Some(vec!["dates.event_year".to_string()]),
        metrics: Some(vec!["event_count".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Degenerate dimension should NOT produce a JOIN
    assert!(
        !has_join(&plan),
        "Plan should NOT contain JOINs for degenerate dimensions"
    );

    // Should scan only the fact table
    assert_eq!(count_scans(&plan), 1, "Should scan only fact table");
}

#[test]
fn test_multiple_degenerate_dimensions() {
    let schema = load_fixture("denormalized.yaml");

    let request = QueryRequest {
        model: "events".to_string(),
        rows: Some(vec![
            "dates.event_year".to_string(),
            "user.user_country".to_string(),
            "event.event_type".to_string(),
        ]),
        metrics: Some(vec!["event_count".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Multiple degenerate dimensions should still NOT produce JOINs
    assert!(
        !has_join(&plan),
        "Plan should NOT contain JOINs for degenerate dimensions"
    );

    // Should scan only the fact table
    assert_eq!(count_scans(&plan), 1, "Should scan only fact table");
}

#[test]
fn test_column_aliasing() {
    // Test that column: overrides work correctly
    let schema = load_fixture("denormalized.yaml");

    let request = QueryRequest {
        model: "events".to_string(),
        rows: Some(vec![
            "dates.event_year".to_string(), // column: year
            "dates.event_month".to_string(), // column: month
        ]),
        metrics: Some(vec!["total_value".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Plan should compile without errors (column mapping works)
    assert!(!has_join(&plan));
}

// TODO: Add tests for:
// - Verify column names in the generated Substrait plan
// - Filter on degenerate dimension
// - Complex expressions with denormalized columns
