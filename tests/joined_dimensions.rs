//! Integration tests for star schema with JOINed dimensions
//!
//! Tests that queries against joined dimensions produce correct JOINs in the plan.

mod common;

use common::{count_joins, count_scans, has_join, load_fixture, run_pipeline};
use semstrait::QueryRequest;

#[test]
fn test_single_joined_dimension() {
    let schema = load_fixture("star_schema.yaml");

    let request = QueryRequest {
        model: "sales".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec!["revenue".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Should have exactly 1 JOIN (to dates dimension)
    assert!(has_join(&plan), "Plan should contain a JOIN");
    assert_eq!(count_joins(&plan), 1, "Should have exactly 1 JOIN");

    // Should scan 2 tables: fact + dates dimension
    assert_eq!(count_scans(&plan), 2, "Should scan fact table + 1 dimension");
}

#[test]
fn test_multiple_joined_dimensions() {
    let schema = load_fixture("star_schema.yaml");

    let request = QueryRequest {
        model: "sales".to_string(),
        rows: Some(vec![
            "dates.year".to_string(),
            "products.category".to_string(),
        ]),
        metrics: Some(vec!["revenue".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // Should have 2 JOINs (dates + products)
    assert!(has_join(&plan), "Plan should contain JOINs");
    assert_eq!(count_joins(&plan), 2, "Should have exactly 2 JOINs");

    // Should scan 3 tables: fact + 2 dimensions
    assert_eq!(
        count_scans(&plan),
        3,
        "Should scan fact table + 2 dimensions"
    );
}

#[test]
fn test_measures_only_no_dimensions() {
    let schema = load_fixture("star_schema.yaml");

    let request = QueryRequest {
        model: "sales".to_string(),
        rows: None,
        metrics: Some(vec!["revenue".to_string(), "quantity".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request).expect("Pipeline should succeed");

    // No dimensions requested = no JOINs needed
    assert!(!has_join(&plan), "Plan should NOT contain JOINs");

    // Should scan only the fact table
    assert_eq!(count_scans(&plan), 1, "Should scan only fact table");
}

// TODO: Add tests for:
// - Filter on joined dimension attribute
// - Multiple attributes from same dimension
// - Sort order on dimension attribute
