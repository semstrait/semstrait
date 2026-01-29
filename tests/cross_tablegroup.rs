//! Integration tests for cross-tableGroup queries
//!
//! Tests that cross-tableGroup metrics (using tableGroup.name conditions)
//! generate correct UNION plans.

mod common;

use common::{has_union, load_fixture};
use semstrait::{parser, planner::plan_cross_table_group_query};

#[test]
fn test_cross_tablegroup_metric_detection() {
    let schema = load_fixture("cross_tablegroup.yaml");
    let model = schema.get_model("marketing").unwrap();

    // Get the unified_cost metric
    let metric = model.get_metric("unified_cost").unwrap();

    // Should be detected as cross-tableGroup
    assert!(
        metric.is_cross_table_group(),
        "unified_cost should be detected as cross-tableGroup metric"
    );

    // Should have mappings for both tableGroups
    let mappings = metric.table_group_measures();
    assert_eq!(mappings.len(), 2, "Should have 2 tableGroup mappings");

    assert!(
        mappings.iter().any(|(tg, m)| tg == "google_ads" && m == "ad_cost"),
        "Should map google_ads → ad_cost"
    );
    assert!(
        mappings.iter().any(|(tg, m)| tg == "meta_ads" && m == "media_spend"),
        "Should map meta_ads → media_spend"
    );
}

#[test]
fn test_cross_tablegroup_union_plan() {
    let schema = load_fixture("cross_tablegroup.yaml");
    let model = schema.get_model("marketing").unwrap();
    let metric = model.get_metric("unified_cost").unwrap();

    // Plan a cross-tableGroup query
    let plan_node = plan_cross_table_group_query(
        &schema,
        model,
        metric,
        &["dates.year".to_string()],
    )
    .expect("Cross-tableGroup planning should succeed");

    // Convert to Substrait to verify structure
    let substrait = semstrait::emit_plan(&plan_node, None).expect("Emission should succeed");

    // Should contain a UNION
    assert!(
        has_union(&substrait),
        "Cross-tableGroup query should produce a UNION plan"
    );
}

#[test]
fn test_single_tablegroup_metric_not_cross() {
    // Verify that normal metrics are NOT detected as cross-tableGroup
    let schema = parser::parse_file("test_data/steelwheels.yaml").unwrap();
    let model = schema.get_model("steelwheels").unwrap();

    if let Some(metric) = model.get_metric("avg_unit_price") {
        assert!(
            !metric.is_cross_table_group(),
            "Normal metric should NOT be cross-tableGroup"
        );
    }
}

// TODO: Add tests for:
// - Cross-tableGroup with multiple dimension attributes
// - Verify the re-aggregation logic
// - Error handling for missing tableGroups
// - Cross-tableGroup with incompatible schemas
