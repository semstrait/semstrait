//! Integration tests for multi-dataset JOIN within same datasetGroup
//!
//! Tests the scenario where a query requires measures from multiple datasets
//! that share common dimensions, resulting in a JOIN plan.

use semstrait::semantic_model::Schema;
use semstrait::selector::{select_datasets, select_datasets_for_join};
use semstrait::planner::plan_semantic_query;
use semstrait::query::QueryRequest;
use semstrait::plan::PlanNode;

fn load_schema() -> Schema {
    Schema::from_file("test_data/multi_table_join.yaml").unwrap()
}

#[test]
fn test_single_dataset_selection_when_possible() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for impressions + clicks - both on campaign_summary
    let result = select_datasets(
        &schema,
        model,
        &["campaign.name".to_string(), "dates.date".to_string()],
        &["impressions".to_string(), "clicks".to_string()],
    );
    
    // Should succeed with single dataset selection
    assert!(result.is_ok());
    let datasets = result.unwrap();
    assert_eq!(datasets.len(), 1);
    assert_eq!(datasets[0].dataset.name, "campaign_summary");
}

#[test]
fn test_single_dataset_selection_fails_for_cross_dataset_measures() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for clicks + cost - clicks on summary, cost on details
    let result = select_datasets(
        &schema,
        model,
        &["campaign.name".to_string()],
        &["clicks".to_string(), "cost".to_string()],
    );
    
    // Should fail - no single dataset has both measures
    assert!(result.is_err());
}

#[test]
fn test_multi_dataset_selection_for_cross_dataset_measures() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for clicks + cost - clicks on summary, cost on details
    let result = select_datasets_for_join(
        &schema,
        model,
        &["campaign.name".to_string()],
        &["clicks".to_string(), "cost".to_string()],
    );
    
    // Should succeed with multi-dataset selection
    assert!(result.is_ok());
    let selection = result.unwrap();
    assert_eq!(selection.datasets.len(), 2);
    
    // Verify measure assignments (smallest dataset first wins)
    let summary_dataset = selection.datasets.iter()
        .find(|t| t.dataset.name == "campaign_summary");
    let details_dataset = selection.datasets.iter()
        .find(|t| t.dataset.name == "campaign_details");
    
    assert!(summary_dataset.is_some());
    assert!(details_dataset.is_some());
    
    // Verify measure assignments
    assert!(summary_dataset.unwrap().measures.contains(&"clicks".to_string()));
    assert!(details_dataset.unwrap().measures.contains(&"cost".to_string()));
}

#[test]
fn test_multi_dataset_join_plan_structure() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    let request = QueryRequest {
        model: "multi-table-test".to_string(),
        dimensions: None,
        rows: Some(vec!["campaign.name".to_string()]),
        columns: None,
        metrics: Some(vec!["clicks".to_string(), "cost".to_string()]),
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // The plan should be: Sort(Project(Join(...)))
    match &plan {
        PlanNode::Sort(sort) => {
            match sort.input.as_ref() {
                PlanNode::Project(proj) => {
                    // Should project campaign.name, clicks, cost
                    assert!(proj.expressions.len() >= 2);
                    
                    // Input should be a Join
                    match proj.input.as_ref() {
                        PlanNode::Join(join) => {
                            // Should be FULL OUTER JOIN
                            assert_eq!(join.join_type, semstrait::plan::JoinType::Full);
                        }
                        _ => panic!("Expected Join node as input to Project"),
                    }
                }
                _ => panic!("Expected Project node inside Sort"),
            }
        }
        _ => panic!("Expected Sort node at top level"),
    }
}

#[test]
fn test_smallest_dataset_first_assignment() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for impressions + cost
    let result = select_datasets_for_join(
        &schema,
        model,
        &["campaign.name".to_string()],
        &["impressions".to_string(), "cost".to_string()],
    );
    
    let selection = result.unwrap();
    
    // campaign_summary should come first (smaller dataset)
    assert_eq!(selection.datasets[0].dataset.name, "campaign_summary");
    assert!(selection.datasets[0].measures.contains(&"impressions".to_string()));
    
    // campaign_details should come second (larger dataset)  
    assert_eq!(selection.datasets[1].dataset.name, "campaign_details");
    assert!(selection.datasets[1].measures.contains(&"cost".to_string()));
}

#[test]
fn test_cross_join_no_dimensions() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Two metrics from different tables, no dimensions at all
    let request = QueryRequest {
        model: "multi-table-test".to_string(),
        dimensions: None,
        rows: None,
        columns: None,
        metrics: Some(vec!["clicks".to_string(), "cost".to_string()]),
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // Should be: Project(CrossJoin(...))
    match &plan {
        PlanNode::Project(proj) => {
            assert_eq!(proj.expressions.len(), 2);
            match proj.input.as_ref() {
                PlanNode::CrossJoin(_) => {}
                other => panic!("Expected CrossJoin, got {:?}", std::mem::discriminant(other)),
            }
        }
        other => panic!("Expected Project at top level, got {:?}", std::mem::discriminant(other)),
    }
}

#[test]
fn test_cross_join_with_virtual_dimension_only() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Two metrics from different tables, only a virtual dimension
    let request = QueryRequest {
        model: "multi-table-test".to_string(),
        dimensions: None,
        rows: Some(vec!["_dataset.datasetGroup".to_string()]),
        columns: None,
        metrics: Some(vec!["clicks".to_string(), "cost".to_string()]),
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // Plan should succeed and contain a CrossJoin (virtual dims don't create join keys)
    fn has_cross_join(node: &PlanNode) -> bool {
        match node {
            PlanNode::CrossJoin(_) => true,
            PlanNode::Project(p) => has_cross_join(&p.input),
            PlanNode::Sort(s) => has_cross_join(&s.input),
            _ => false,
        }
    }
    assert!(has_cross_join(&plan), "Plan should contain a CrossJoin:\n{}", plan.display_indent());
}
