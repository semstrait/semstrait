//! Integration tests for multi-table JOIN within same tableGroup
//!
//! Tests the scenario where a query requires measures from multiple tables
//! that share common dimensions, resulting in a JOIN plan.

use semstrait::model::Schema;
use semstrait::selector::{select_tables, select_tables_for_join};
use semstrait::planner::plan_semantic_query;
use semstrait::query::QueryRequest;
use semstrait::plan::PlanNode;

fn load_schema() -> Schema {
    Schema::from_file("test_data/multi_table_join.yaml").unwrap()
}

#[test]
fn test_single_table_selection_when_possible() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for impressions + clicks - both on campaign_summary
    let result = select_tables(
        &schema,
        model,
        &["campaign.name".to_string(), "dates.date".to_string()],
        &["impressions".to_string(), "clicks".to_string()],
    );
    
    // Should succeed with single table selection
    assert!(result.is_ok());
    let tables = result.unwrap();
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].table.table, "campaign_summary");
}

#[test]
fn test_single_table_selection_fails_for_cross_table_measures() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for clicks + cost - clicks on summary, cost on details
    let result = select_tables(
        &schema,
        model,
        &["campaign.name".to_string()],
        &["clicks".to_string(), "cost".to_string()],
    );
    
    // Should fail - no single table has both measures
    assert!(result.is_err());
}

#[test]
fn test_multi_table_selection_for_cross_table_measures() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for clicks + cost - clicks on summary, cost on details
    let result = select_tables_for_join(
        &schema,
        model,
        &["campaign.name".to_string()],
        &["clicks".to_string(), "cost".to_string()],
    );
    
    // Should succeed with multi-table selection
    assert!(result.is_ok());
    let selection = result.unwrap();
    assert_eq!(selection.tables.len(), 2);
    
    // Verify measure assignments (smallest table first wins)
    // campaign_summary has fewer dimensions (attribute_count), so clicks should be assigned to it
    let summary_table = selection.tables.iter()
        .find(|t| t.table.table == "campaign_summary");
    let details_table = selection.tables.iter()
        .find(|t| t.table.table == "campaign_details");
    
    assert!(summary_table.is_some());
    assert!(details_table.is_some());
    
    // Verify measure assignments
    assert!(summary_table.unwrap().measures.contains(&"clicks".to_string()));
    assert!(details_table.unwrap().measures.contains(&"cost".to_string()));
}

#[test]
fn test_multi_table_join_plan_structure() {
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
fn test_smallest_table_first_assignment() {
    let schema = load_schema();
    let model = schema.get_model("multi-table-test").unwrap();
    
    // Query for impressions + cost
    // impressions is on campaign_summary (smaller)
    // cost is on campaign_details (larger)
    let result = select_tables_for_join(
        &schema,
        model,
        &["campaign.name".to_string()],
        &["impressions".to_string(), "cost".to_string()],
    );
    
    let selection = result.unwrap();
    
    // campaign_summary should come first (smaller table)
    assert_eq!(selection.tables[0].table.table, "campaign_summary");
    assert!(selection.tables[0].measures.contains(&"impressions".to_string()));
    
    // campaign_details should come second (larger table)  
    assert_eq!(selection.tables[1].table.table, "campaign_details");
    assert!(selection.tables[1].measures.contains(&"cost".to_string()));
}
