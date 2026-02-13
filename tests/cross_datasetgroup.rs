//! Integration tests for cross-datasetGroup queries
//!
//! Tests that cross-datasetGroup metrics (using datasetGroup.name conditions)
//! generate correct UNION plans.

mod common;

use common::{has_union, load_fixture};
use semstrait::{parser, planner::plan_cross_dataset_group_query};

#[test]
fn test_cross_datasetgroup_metric_detection() {
    let schema = load_fixture("cross_tablegroup.yaml");
    let model = schema.get_model("marketing").unwrap();

    // Get the unified_cost metric
    let metric = model.get_metric("unified_cost").unwrap();

    // Should be detected as cross-datasetGroup
    assert!(
        metric.is_cross_dataset_group(),
        "unified_cost should be detected as cross-datasetGroup metric"
    );

    // Should have mappings for both datasetGroups
    let mappings = metric.dataset_group_measures();
    assert_eq!(mappings.len(), 2, "Should have 2 datasetGroup mappings");

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
fn test_cross_datasetgroup_union_plan() {
    let schema = load_fixture("cross_tablegroup.yaml");
    let model = schema.get_model("marketing").unwrap();
    let metric = model.get_metric("unified_cost").unwrap();

    // Plan a cross-datasetGroup query
    let plan_node = plan_cross_dataset_group_query(
        &schema,
        model,
        metric,
        &["dates.year".to_string()],
    )
    .expect("Cross-datasetGroup planning should succeed");

    // Convert to Substrait to verify structure
    let substrait = semstrait::emit_plan(&plan_node, None).expect("Emission should succeed");

    // Should contain a UNION
    assert!(
        has_union(&substrait),
        "Cross-datasetGroup query should produce a UNION plan"
    );
}

#[test]
fn test_single_datasetgroup_metric_not_cross() {
    // Verify that normal metrics are NOT detected as cross-datasetGroup
    let schema = parser::parse_file("test_data/steelwheels.yaml").unwrap();
    let model = schema.get_model("steelwheels").unwrap();

    if let Some(metric) = model.get_metric("avg_unit_price") {
        assert!(
            !metric.is_cross_dataset_group(),
            "Normal metric should NOT be cross-datasetGroup"
        );
    }
}

// =============================================================================
// Model-Level Dimension Tests
// =============================================================================

#[test]
fn test_conformed_dimension_detection() {
    let schema = load_fixture("cross_tablegroup.yaml");
    let model = schema.get_model("marketing").unwrap();

    // dates is at model level - all attributes are conformed
    assert!(model.is_conformed("dates", "year"), "dates.year should be conformed (model-level)");
    assert!(model.is_conformed("dates", "date"), "dates.date should be conformed (model-level)");
    
    // _table is at model level (virtual) - all attributes are conformed
    assert!(model.is_conformed("_table", "tableGroup"), "_table.tableGroup should be conformed (virtual)");
    
    // campaign is NOT at model level (inline only) - NOT conformed
    assert!(!model.is_conformed("campaign", "campaign_id"), "campaign.campaign_id should NOT be conformed (inline only)");
    assert!(!model.is_conformed("campaign", "campaign_name"), "campaign.campaign_name should NOT be conformed (inline only)");
    
    // Non-existent dimensions are not conformed
    assert!(!model.is_conformed("other", "attr"), "non-existent dimension should NOT be conformed");
}

#[test]
fn test_conformed_query_detection() {
    let schema = load_fixture("cross_tablegroup.yaml");
    let model = schema.get_model("marketing").unwrap();

    // Query with only model-level dimensions
    let conformed_query = vec!["dates.year".to_string(), "_table.tableGroup".to_string()];
    assert!(model.is_conformed_query(&conformed_query), "Query with dates.year and _table should be conformed");
    
    // Query with inline-only dimension (not at model level)
    let non_conformed_query = vec!["campaign.campaign_name".to_string()];
    assert!(!model.is_conformed_query(&non_conformed_query), "Query with inline dimension should NOT be conformed");
    
    // Query with mix of model-level and inline dimensions
    let mixed_query = vec!["dates.year".to_string(), "campaign.campaign_name".to_string()];
    assert!(!model.is_conformed_query(&mixed_query), "Mixed query should NOT be conformed");
}

#[test]
fn test_conformed_dimension_union_plan() {
    use common::run_pipeline;
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");

    // Query conformed dimension with a metric that exists in both datasetGroups
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec!["dates.year".to_string()]),
        metrics: Some(vec!["clicks".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request)
        .expect("Conformed dimension query should succeed");
    
    // Should produce a UNION plan (querying across both datasetGroups)
    assert!(
        has_union(&plan),
        "Conformed dimension query should produce a UNION plan"
    );
}

#[test]
fn test_conformed_dimension_with_table_metadata() {
    use common::run_pipeline;
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");

    // Query conformed dimension + _table.tableGroup + metric
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "dates.year".to_string(),
            "_table.tableGroup".to_string(),
        ]),
        metrics: Some(vec!["clicks".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request)
        .expect("Conformed dimension + _table query should succeed");
    
    // Should produce a UNION plan
    assert!(
        has_union(&plan),
        "Conformed dimension + _table query should produce a UNION plan"
    );
}

#[test]
fn test_virtual_dimension_implicitly_conformed() {
    use common::run_pipeline;
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");

    // Query ONLY _table.tableGroup (virtual dimension) + metric
    // Virtual dimensions should be implicitly conformed
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "_table.tableGroup".to_string(),
        ]),
        metrics: Some(vec!["clicks".to_string()]),
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request)
        .expect("Virtual dimension only query should succeed (implicitly conformed)");
    
    // Should produce a UNION plan (querying across both datasetGroups)
    assert!(
        has_union(&plan),
        "Virtual dimension query should produce a UNION plan"
    );
}

#[test]
fn test_virtual_only_query_no_table_scan() {
    use common::run_pipeline;
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");

    // Query ONLY _table.tableGroup (virtual dimension) with NO metrics
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "_table.tableGroup".to_string(),
        ]),
        metrics: None,
        ..Default::default()
    };

    let plan = run_pipeline(&schema, &request)
        .expect("Virtual-only query should succeed without table scans");
    
    // Should NOT produce a UNION - should be a VirtualTable
    assert!(
        !has_union(&plan),
        "Virtual-only query should NOT produce a UNION plan"
    );
}

// =============================================================================
// DatasetGroup-Qualified Dimension Tests
// =============================================================================

#[test]
fn test_datasetgroup_qualified_dimension_parsing() {
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");
    
    // Query with datasetGroup-qualified dimension
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "google_ads.dates.year".to_string(),
        ]),
        metrics: Some(vec!["unified_cost".to_string()]),
        ..Default::default()
    };

    let result = common::run_pipeline(&schema, &request);
    assert!(
        result.is_ok(),
        "DatasetGroup-qualified dimension query should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_datasetgroup_qualified_dimension_cross_datasetgroup_metric() {
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");
    
    // Query with datasetGroup-qualified dimensions from BOTH datasetGroups
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "google_ads.dates.year".to_string(),
            "meta_ads.dates.year".to_string(),
        ]),
        metrics: Some(vec!["unified_cost".to_string()]),
        ..Default::default()
    };

    let result = common::run_pipeline(&schema, &request);
    assert!(
        result.is_ok(),
        "Query with qualified dimensions from both datasetGroups should succeed: {:?}",
        result.err()
    );
    
    let plan = result.unwrap();
    
    // Should produce a UNION plan
    assert!(
        has_union(&plan),
        "Query with qualified dimensions should produce a UNION plan"
    );
}

#[test]
fn test_datasetgroup_qualified_with_virtual_dimension() {
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");
    
    // Query with datasetGroup-qualified dimension + virtual _table dimension
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "google_ads.dates.year".to_string(),
            "_table.tableGroup".to_string(),
        ]),
        metrics: Some(vec!["unified_cost".to_string()]),
        ..Default::default()
    };

    let result = common::run_pipeline(&schema, &request);
    assert!(
        result.is_ok(),
        "Query with qualified + virtual dimensions should succeed: {:?}",
        result.err()
    );
    
    let plan = result.unwrap();
    assert!(
        has_union(&plan),
        "Query with qualified + virtual dimensions should produce a UNION plan"
    );
}

#[test]
fn test_invalid_datasetgroup_qualifier_fails() {
    use semstrait::QueryRequest;
    
    let schema = load_fixture("cross_tablegroup.yaml");
    
    // Query with non-existent datasetGroup qualifier should fail
    let request = QueryRequest {
        model: "marketing".to_string(),
        rows: Some(vec![
            "nonexistent_tg.dates.year".to_string(),
        ]),
        metrics: Some(vec!["unified_cost".to_string()]),
        ..Default::default()
    };

    let result = common::run_pipeline(&schema, &request);
    assert!(
        result.is_err(),
        "Query with invalid datasetGroup qualifier should fail"
    );
}

// ============================================================================
// Multiple Cross-DatasetGroup Metrics Tests
// ============================================================================

#[test]
fn test_multiple_cross_datasetgroup_metrics_detection() {
    let schema = parser::parse_file("test_data/marketing.yaml").unwrap();
    let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();

    // Both metrics should be detected as cross-datasetGroup
    let cost_metric = model.get_metric("fun-cost").unwrap();
    let impressions_metric = model.get_metric("fun-impressions").unwrap();

    assert!(cost_metric.is_cross_dataset_group(), "fun-cost should be cross-datasetGroup");
    assert!(impressions_metric.is_cross_dataset_group(), "fun-impressions should be cross-datasetGroup");
}

#[test]
fn test_multiple_cross_datasetgroup_metrics_planning() {
    use semstrait::planner::plan_semantic_query;
    use semstrait::query::QueryRequest;

    let schema = parser::parse_file("test_data/marketing.yaml").unwrap();
    let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();

    // Query with BOTH cross-datasetGroup metrics
    let request = QueryRequest {
        model: "-ObDoDFVQGxxCGa5vw_Z".to_string(),
        dimensions: None,
        rows: Some(vec!["dates.date".to_string()]),
        columns: None,
        metrics: Some(vec!["fun-cost".to_string(), "fun-impressions".to_string()]),
        filter: None,
    };

    let plan = plan_semantic_query(&schema, model, &request);
    assert!(plan.is_ok(), "Multiple cross-datasetGroup metrics should be supported: {:?}", plan.err());

    let plan_node = plan.unwrap();
    let substrait = semstrait::emit_plan(&plan_node, None).expect("Emission should succeed");
    
    assert!(has_union(&substrait), "Multiple cross-datasetGroup metrics should produce UNION plan");
}

#[test]
fn test_multiple_cross_datasetgroup_metrics_union_structure() {
    use semstrait::planner::plan_semantic_query;
    use semstrait::query::QueryRequest;
    use semstrait::plan::PlanNode;

    let schema = parser::parse_file("test_data/marketing.yaml").unwrap();
    let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();

    let request = QueryRequest {
        model: "-ObDoDFVQGxxCGa5vw_Z".to_string(),
        dimensions: None,
        rows: Some(vec!["dates.date".to_string()]),
        columns: None,
        metrics: Some(vec!["fun-cost".to_string(), "fun-impressions".to_string()]),
        filter: None,
    };

    let plan = plan_semantic_query(&schema, model, &request).unwrap();

    // The plan should be: Sort(Aggregate(Union([branch1, branch2])))
    match plan {
        PlanNode::Sort(sort) => {
            match *sort.input {
                PlanNode::Aggregate(agg) => {
                    assert_eq!(agg.aggregates.len(), 2, "Should have 2 aggregates (one per metric)");
                    
                    let aliases: Vec<&str> = agg.aggregates.iter()
                        .map(|a| a.alias.as_str())
                        .collect();
                    assert!(aliases.contains(&"fun-cost"), "Should have fun-cost aggregate");
                    assert!(aliases.contains(&"fun-impressions"), "Should have fun-impressions aggregate");
                    
                    match *agg.input {
                        PlanNode::Union(union) => {
                            assert_eq!(union.inputs.len(), 2, "Union should have 2 branches");
                        }
                        _ => panic!("Expected Union as input to Aggregate"),
                    }
                }
                _ => panic!("Expected Aggregate inside Sort"),
            }
        }
        _ => panic!("Expected Sort at top level"),
    }
}

#[test]
fn test_single_cross_datasetgroup_metric_still_works() {
    use semstrait::planner::plan_semantic_query;
    use semstrait::query::QueryRequest;

    let schema = parser::parse_file("test_data/marketing.yaml").unwrap();
    let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();

    let request = QueryRequest {
        model: "-ObDoDFVQGxxCGa5vw_Z".to_string(),
        dimensions: None,
        rows: Some(vec!["dates.date".to_string()]),
        columns: None,
        metrics: Some(vec!["fun-cost".to_string()]),
        filter: None,
    };

    let plan = plan_semantic_query(&schema, model, &request);
    assert!(plan.is_ok(), "Single cross-datasetGroup metric should work: {:?}", plan.err());
}
