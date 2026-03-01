//! Integration tests for partitioned datasets
//!
//! Tests the scenario where a dataset group contains multiple partitioned datasets
//! (e.g., same Facebook Ads data split by ad account) that should be combined
//! with UNION ALL.

use semstrait::semantic_model::Schema;
use semstrait::selector::select_datasets;
use semstrait::planner::{plan_semantic_query, plan_cross_dataset_group_query};
use semstrait::query::QueryRequest;
use semstrait::plan::PlanNode;

fn load_schema() -> Schema {
    Schema::from_file("test_data/partitioned.yaml").unwrap()
}

#[test]
fn test_selector_returns_all_partitions() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let datasets = select_datasets(
        &schema,
        model,
        &["dates.date".to_string()],
        &["spend".to_string()],
    ).unwrap();
    
    // Should return all 3 partitioned datasets
    assert_eq!(datasets.len(), 3);
    assert_eq!(datasets[0].group.name, "facebookads");
    
    // All should have partition values
    let partitions: Vec<&str> = datasets.iter()
        .map(|s| s.dataset.partition.as_deref().unwrap())
        .collect();
    assert!(partitions.contains(&"111"));
    assert!(partitions.contains(&"222"));
    assert!(partitions.contains(&"333"));
}

#[test]
fn test_non_partitioned_group_returns_single() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    // adwords has no partitions - should return single best
    let datasets = select_datasets(
        &schema,
        model,
        &["dates.date".to_string()],
        &["cost".to_string()],
    ).unwrap();
    
    assert_eq!(datasets.len(), 1);
    assert_eq!(datasets[0].group.name, "adwords");
    assert!(datasets[0].dataset.partition.is_none());
}

#[test]
fn test_partition_label_on_group() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let fb_group = model.get_dataset_group("facebookads").unwrap();
    assert_eq!(fb_group.partition_label.as_deref(), Some("Account ID"));
    
    let adwords_group = model.get_dataset_group("adwords").unwrap();
    assert!(adwords_group.partition_label.is_none());
}

#[test]
fn test_has_partitions() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let fb_group = model.get_dataset_group("facebookads").unwrap();
    assert!(fb_group.has_partitions());
    
    let adwords_group = model.get_dataset_group("adwords").unwrap();
    assert!(!adwords_group.has_partitions());
}

#[test]
fn test_partitioned_query_produces_union_plan() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let request = QueryRequest {
        model: "partitioned_ads".to_string(),
        dimensions: None,
        rows: Some(vec![
            "dates.date".to_string(),
            "_dataset.partition".to_string(),
        ]),
        columns: None,
        metrics: Some(vec!["spend".to_string()]),
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // The plan should be a Union with 3 branches (one per partition)
    match &plan {
        PlanNode::Union(union) => {
            assert_eq!(union.inputs.len(), 3, "Expected 3 union branches for 3 partitions");
        }
        other => panic!("Expected Union plan, got: {:?}", std::mem::discriminant(other)),
    }
}

#[test]
fn test_non_partitioned_query_no_union() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let request = QueryRequest {
        model: "partitioned_ads".to_string(),
        dimensions: None,
        rows: Some(vec!["dates.date".to_string()]),
        columns: None,
        metrics: Some(vec!["cost".to_string()]),
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // adwords is not partitioned, so no Union
    match &plan {
        PlanNode::Union(_) => panic!("Expected non-Union plan for non-partitioned group"),
        _ => {} // Any other plan type is fine
    }
}

#[test]
fn test_partition_value_in_resolver() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let datasets = select_datasets(
        &schema,
        model,
        &["dates.date".to_string(), "_dataset.partition".to_string()],
        &["spend".to_string()],
    ).unwrap();
    
    // Resolve query for each partition to verify _dataset.partition meta value
    for selected in &datasets {
        let request = QueryRequest {
            model: "partitioned_ads".to_string(),
            dimensions: None,
            rows: Some(vec![
                "dates.date".to_string(),
                "_dataset.partition".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["spend".to_string()]),
            filter: None,
        };
        
        let resolved = semstrait::resolver::resolve_query(&schema, &request, selected).unwrap();
        
        // Find the _dataset.partition attribute
        let partition_attr = resolved.row_attributes.iter()
            .find(|a| a.dimension_name() == "_dataset" && a.attribute_name() == "partition")
            .expect("Should have _dataset.partition attribute");
        
        assert!(partition_attr.is_meta());
        let meta_value = partition_attr.meta_value().unwrap();
        assert!(
            ["111", "222", "333"].contains(&meta_value),
            "Unexpected partition value: {}",
            meta_value
        );
    }
}

#[test]
fn test_cross_dataset_group_with_partitions() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let metric = model.get_metric("total_clicks").unwrap();
    
    let plan = plan_cross_dataset_group_query(
        &schema,
        model,
        metric,
        &["_dataset.partition".to_string()],
    ).unwrap();
    
    // The plan should contain a Union. For facebookads (3 partitions) + adwords (1 dataset),
    // we expect 4 branches total.
    fn count_union_inputs(node: &PlanNode) -> Option<usize> {
        match node {
            PlanNode::Union(u) => Some(u.inputs.len()),
            PlanNode::Sort(s) => count_union_inputs(&s.input),
            PlanNode::Aggregate(a) => count_union_inputs(&a.input),
            PlanNode::Project(p) => count_union_inputs(&p.input),
            _ => None,
        }
    }
    
    let branch_count = count_union_inputs(&plan)
        .expect("Expected a Union somewhere in the plan");
    assert_eq!(branch_count, 4, "Expected 4 union branches: 3 fb partitions + 1 adwords");
}

#[test]
fn test_cross_dataset_group_partition_literals_not_null() {
    use semstrait::plan::{Literal, Expr};
    
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let metric = model.get_metric("total_clicks").unwrap();
    
    let plan = plan_cross_dataset_group_query(
        &schema,
        model,
        metric,
        &["_dataset.partition".to_string()],
    ).unwrap();
    
    // Walk to the Union node and check projection literals
    fn find_union(node: &PlanNode) -> Option<&PlanNode> {
        match node {
            PlanNode::Union(_) => Some(node),
            PlanNode::Sort(s) => find_union(&s.input),
            PlanNode::Aggregate(a) => find_union(&a.input),
            PlanNode::Project(p) => find_union(&p.input),
            _ => None,
        }
    }
    
    fn extract_partition_literal(node: &PlanNode) -> Option<String> {
        match node {
            PlanNode::Project(p) => {
                for pe in &p.expressions {
                    if pe.alias == "_dataset.partition" {
                        match &pe.expr {
                            Expr::Literal(Literal::String(s)) => return Some(s.clone()),
                            Expr::Literal(Literal::Null(_)) => return Some("NULL".to_string()),
                            _ => {}
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
    
    let union_node = find_union(&plan).expect("Should have a Union node");
    if let PlanNode::Union(u) = union_node {
        let mut partition_values: Vec<String> = Vec::new();
        for branch in &u.inputs {
            if let Some(val) = extract_partition_literal(branch) {
                partition_values.push(val);
            }
        }
        
        // Should have 4 values: "111", "222", "333" from fb partitions, NULL from adwords
        assert_eq!(partition_values.len(), 4);
        assert!(partition_values.contains(&"111".to_string()));
        assert!(partition_values.contains(&"222".to_string()));
        assert!(partition_values.contains(&"333".to_string()));
        assert!(partition_values.contains(&"NULL".to_string()));
    }
}

#[test]
fn test_virtual_only_partition_query() {
    use semstrait::plan::LiteralValue;
    
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let request = QueryRequest {
        model: "partitioned_ads".to_string(),
        dimensions: None,
        rows: Some(vec!["_dataset.partition".to_string()]),
        columns: None,
        metrics: None,
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // Should be a VirtualTable with one row per partition + one for adwords
    match &plan {
        PlanNode::VirtualTable(vt) => {
            // 3 fb partitions + 1 adwords = 4 rows
            assert_eq!(vt.rows.len(), 4, "Expected 4 rows (3 partitions + 1 non-partitioned)");
            
            let partition_values: Vec<String> = vt.rows.iter()
                .map(|row| match &row[0] {
                    LiteralValue::String(s) => s.clone(),
                    LiteralValue::Null => "NULL".to_string(),
                    other => format!("{:?}", other),
                })
                .collect();
            
            assert!(partition_values.contains(&"111".to_string()));
            assert!(partition_values.contains(&"222".to_string()));
            assert!(partition_values.contains(&"333".to_string()));
            assert!(partition_values.contains(&"NULL".to_string()));
        }
        other => panic!("Expected VirtualTable, got: {:?}", std::mem::discriminant(other)),
    }
}

#[test]
fn test_conformed_dimension_with_partition_no_metrics() {
    let schema = load_schema();
    let model = schema.get_model("partitioned_ads").unwrap();
    
    let request = QueryRequest {
        model: "partitioned_ads".to_string(),
        dimensions: None,
        rows: Some(vec![
            "dates.date".to_string(),
            "_dataset.partition".to_string(),
        ]),
        columns: None,
        metrics: None,
        filter: None,
    };
    
    let plan = plan_semantic_query(&schema, model, &request).unwrap();
    
    // Should produce a Union with 4 branches (3 fb partitions + 1 adwords)
    fn count_union_inputs(node: &PlanNode) -> Option<usize> {
        match node {
            PlanNode::Union(u) => Some(u.inputs.len()),
            PlanNode::Sort(s) => count_union_inputs(&s.input),
            PlanNode::Aggregate(a) => count_union_inputs(&a.input),
            PlanNode::Project(p) => count_union_inputs(&p.input),
            _ => None,
        }
    }
    
    let branch_count = count_union_inputs(&plan)
        .expect("Expected a Union in the plan");
    assert_eq!(branch_count, 4, "Expected 4 union branches: 3 fb partitions + 1 adwords");
}
