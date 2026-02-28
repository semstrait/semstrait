//! Integration tests for dataset selection (aggregate awareness)
//!
//! Tests that the selector picks the most appropriate dataset based on query requirements.

mod common;

use common::load_fixture;
use semstrait::select_datasets;

#[test]
fn test_select_yearly_for_year_only() {
    let schema = load_fixture("multi_table.yaml");
    let model = schema.get_model("sales_multi").unwrap();

    // Query only needs year - should select yearly_summary
    let dims = vec!["dates.year".to_string()];
    let measures = vec!["revenue".to_string()];

    let selected = select_datasets(&schema, model, &dims, &measures).expect("Selection should succeed");

    assert!(!selected.is_empty(), "Should select at least one dataset");
    assert_eq!(
        selected[0].dataset.dataset, "agg.yearly_summary",
        "Should select the yearly aggregate dataset"
    );
}

#[test]
fn test_select_monthly_for_month_query() {
    let schema = load_fixture("multi_table.yaml");
    let model = schema.get_model("sales_multi").unwrap();

    // Query needs month - should select monthly_summary
    let dims = vec!["dates.year".to_string(), "dates.month".to_string()];
    let measures = vec!["revenue".to_string()];

    let selected = select_datasets(&schema, model, &dims, &measures).expect("Selection should succeed");

    assert!(!selected.is_empty(), "Should select at least one dataset");
    assert_eq!(
        selected[0].dataset.dataset, "agg.monthly_summary",
        "Should select the monthly aggregate dataset"
    );
}

#[test]
fn test_select_daily_for_day_query() {
    let schema = load_fixture("multi_table.yaml");
    let model = schema.get_model("sales_multi").unwrap();

    // Query needs day - should select daily_detail
    let dims = vec!["dates.day".to_string()];
    let measures = vec!["revenue".to_string()];

    let selected = select_datasets(&schema, model, &dims, &measures).expect("Selection should succeed");

    assert!(!selected.is_empty(), "Should select at least one dataset");
    assert_eq!(
        selected[0].dataset.dataset, "agg.daily_detail",
        "Should select the daily detail dataset"
    );
}

#[test]
fn test_select_most_aggregated_when_possible() {
    let schema = load_fixture("multi_table.yaml");
    let model = schema.get_model("sales_multi").unwrap();

    // Query only needs region - all datasets have it, should pick most aggregated
    let dims = vec!["region.region_name".to_string()];
    let measures = vec!["revenue".to_string()];

    let selected = select_datasets(&schema, model, &dims, &measures).expect("Selection should succeed");

    assert!(!selected.is_empty(), "Should select at least one dataset");
    assert_eq!(
        selected[0].dataset.dataset, "agg.yearly_summary",
        "Should select the most aggregated dataset"
    );
}
