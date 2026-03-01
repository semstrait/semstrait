//! Top-level query planning entry point
//!
//! `plan_semantic_query` is the main router that classifies queries and
//! dispatches to the appropriate planner.

use std::collections::HashSet;
use crate::semantic_model::{MetricExpr, Schema, SemanticModel, Metric};
use crate::plan::PlanNode;
use crate::resolver::resolve_query;
use crate::selector::{select_datasets, select_datasets_for_join, SelectedDataset};
use crate::query::QueryRequest;
use super::error::PlanError;
use super::table::plan_query;
use super::cross::{plan_cross_dataset_group_query, plan_multi_cross_dataset_group_query};
use super::union::{plan_partitioned_union, plan_conformed_query, plan_multi_tablegroup_query, plan_single_tablegroup_query};
use super::join::plan_same_tablegroup_join;

/// Plan a semantic query, automatically handling all query types.
///
/// This is the main entry point for query planning. It:
/// 1. Analyzes the requested metrics to detect cross-tableGroup metrics
/// 2. Routes to the appropriate planner based on query characteristics
pub fn plan_semantic_query(
    schema: &Schema,
    model: &SemanticModel,
    request: &QueryRequest,
) -> Result<PlanNode, PlanError> {
    let mut dimension_attrs: Vec<String> = Vec::new();
    if let Some(ref rows) = request.rows {
        dimension_attrs.extend(rows.clone());
    }
    if let Some(ref cols) = request.columns {
        dimension_attrs.extend(cols.clone());
    }

    let cross_dataset_metrics: Vec<&Metric> = request.metrics
        .as_ref()
        .map(|names| {
            names.iter()
                .filter_map(|name| model.get_metric(name))
                .filter(|m| m.is_cross_dataset_group())
                .collect()
        })
        .unwrap_or_default();

    let qualified_groups: HashSet<&str> = dimension_attrs.iter()
        .filter_map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 { Some(parts[0]) } else { None }
        })
        .collect();

    let is_conformed = model.is_conformed_query(&dimension_attrs);

    if cross_dataset_metrics.len() == 1 {
        plan_cross_dataset_group_query(schema, model, cross_dataset_metrics[0], &dimension_attrs)
    } else if cross_dataset_metrics.len() > 1 {
        plan_multi_cross_dataset_group_query(schema, model, &cross_dataset_metrics, &dimension_attrs)
    } else if qualified_groups.len() > 1 {
        plan_multi_tablegroup_query(schema, model, request, &dimension_attrs, &qualified_groups)
    } else if qualified_groups.len() == 1 {
        let target_group = *qualified_groups.iter().next().unwrap();
        plan_single_tablegroup_query(schema, model, request, &dimension_attrs, target_group)
    } else {
        let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
        let required_measures: Vec<String> = metric_names
            .iter()
            .filter_map(|metric_name| {
                model.get_metric(metric_name).and_then(|m| {
                    match &m.expr {
                        MetricExpr::MeasureRef(name) => Some(name.clone()),
                        MetricExpr::Structured(_) => None,
                    }
                })
            })
            .collect();

        match select_datasets(schema, model, &dimension_attrs, &required_measures) {
            Ok(selected_tables) => {
                if selected_tables.len() > 1 && selected_tables[0].dataset.partition.is_some() {
                    plan_partitioned_union(schema, model, request, &selected_tables)
                } else if is_conformed && model.dataset_groups.len() > 1 {
                    plan_conformed_query(schema, model, request, &dimension_attrs)
                } else {
                    let selected = selected_tables.into_iter().next()
                        .ok_or_else(|| PlanError::InvalidQuery("No feasible table found for query".to_string()))?;
                    let resolved = resolve_query(schema, request, &selected)
                        .map_err(|e| PlanError::InvalidQuery(format!("Query resolution error: {:?}", e)))?;
                    plan_query(&resolved)
                }
            }
            Err(select_err) => {
                if is_conformed && model.dataset_groups.len() > 1 {
                    plan_conformed_query(schema, model, request, &dimension_attrs)
                } else {
                    let multi_selection = select_datasets_for_join(schema, model, &dimension_attrs, &required_measures)
                        .map_err(|_| PlanError::InvalidQuery(format!("Dataset selection error: {:?}", select_err)))?;

                    if multi_selection.datasets.len() == 1 {
                        let selected = SelectedDataset {
                            group: multi_selection.group,
                            dataset: multi_selection.datasets[0].dataset,
                        };
                        let resolved = resolve_query(schema, request, &selected)
                            .map_err(|e| PlanError::InvalidQuery(format!("Query resolution error: {:?}", e)))?;
                        plan_query(&resolved)
                    } else {
                        plan_same_tablegroup_join(
                            model,
                            &multi_selection,
                            &dimension_attrs,
                            &metric_names,
                        )
                    }
                }
            }
        }
    }
}
