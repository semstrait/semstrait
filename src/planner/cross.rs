//! Cross-datasetGroup query planning
//!
//! Handles metrics that span multiple datasetGroups. These produce UNION plans
//! where each branch aggregates its own datasetGroup, then a re-aggregation
//! combines the results.

use std::collections::{HashMap, HashSet};
use crate::semantic_model::{MeasureExpr, Dataset, DatasetGroup, Schema, SemanticModel, Metric, Measure, Aggregation};
use crate::plan::{
    Aggregate, AggregateExpr, Column, Expr, Join, JoinType,
    Literal, PlanNode, Scan, Project, ProjectExpr, Sort, SortKey, SortDirection, Union,
    LiteralValue as PlanLiteralValue,
};
use super::error::PlanError;
use super::util::{needs_join_for_dimension, ParsedDimensionAttr, get_virtual_attribute_value_with_dataset};
use super::expr::convert_measure_expr;
use super::table::{build_tablegroup_branch, build_tablegroup_branch_for_dataset};

/// A branch in a cross-datasetGroup query
#[derive(Debug)]
pub struct CrossDatasetGroupBranch<'a> {
    pub dataset_group: &'a DatasetGroup,
    pub measure: &'a Measure,
    pub table: &'a Dataset,
}

/// Plan a cross-tableGroup query for a single metric.
pub fn plan_cross_dataset_group_query<'a>(
    _schema: &'a Schema,
    model: &'a SemanticModel,
    metric: &'a Metric,
    dimension_attrs: &[String],
) -> Result<PlanNode, PlanError> {
    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        if parts.len() == 3 {
            let tg_name = parts[0];
            if model.get_dataset_group(tg_name).is_none() {
                return Err(PlanError::InvalidQuery(
                    format!("DatasetGroup '{}' not found in qualified dimension '{}'", tg_name, attr_path)
                ));
            }
        }
    }

    let mappings = metric.dataset_group_measures();
    if mappings.is_empty() {
        return Err(PlanError::InvalidQuery(
            format!("Metric '{}' is not a cross-tableGroup metric", metric.name)
        ));
    }

    let mut branches: Vec<PlanNode> = Vec::new();

    for (tg_name, measure_name) in &mappings {
        let dataset_group = model.get_dataset_group(tg_name)
            .ok_or_else(|| PlanError::InvalidQuery(
                format!("DatasetGroup '{}' not found", tg_name)
            ))?;
        let measure = dataset_group.get_measure(measure_name)
            .ok_or_else(|| PlanError::InvalidQuery(
                format!("Measure '{}' not found in tableGroup '{}'", measure_name, tg_name)
            ))?;

        if dataset_group.has_partitions() {
            let partitioned: Vec<&Dataset> = dataset_group.datasets.iter()
                .filter(|d| d.partition.is_some() && d.has_measure(measure_name))
                .collect();
            for table in partitioned {
                let branch = build_cross_dataset_group_branch(
                    model, dataset_group, table, measure, dimension_attrs, &metric.name,
                )?;
                branches.push(branch);
            }
        } else {
            let table = dataset_group.datasets.iter()
                .find(|t| t.has_measure(measure_name))
                .ok_or_else(|| PlanError::InvalidQuery(
                    format!("No table in tableGroup '{}' has measure '{}'", tg_name, measure_name)
                ))?;
            let branch = build_cross_dataset_group_branch(
                model, dataset_group, table, measure, dimension_attrs, &metric.name,
            )?;
            branches.push(branch);
        }
    }

    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }

    let union = PlanNode::Union(Union { inputs: branches });

    let group_by: Vec<Column> = dimension_attrs.iter()
        .map(|attr| Column::unqualified(attr))
        .collect();
    let aggregates = vec![
        AggregateExpr {
            func: Aggregation::Sum,
            expr: Expr::Column(Column::unqualified(&metric.name)),
            alias: metric.name.clone(),
        }
    ];

    let plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(union),
        group_by: group_by.clone(),
        aggregates,
    });

    let sort_keys: Vec<SortKey> = dimension_attrs.iter()
        .map(|attr| SortKey {
            column: attr.clone(),
            direction: SortDirection::Ascending,
        })
        .collect();

    if !sort_keys.is_empty() {
        Ok(PlanNode::Sort(Sort {
            input: Box::new(plan),
            sort_keys,
        }))
    } else {
        Ok(plan)
    }
}

/// Plan a cross-tableGroup query for multiple metrics.
pub fn plan_multi_cross_dataset_group_query<'a>(
    _schema: &'a Schema,
    model: &'a SemanticModel,
    metrics: &[&'a Metric],
    dimension_attrs: &[String],
) -> Result<PlanNode, PlanError> {
    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        if parts.len() == 3 {
            let tg_name = parts[0];
            if model.get_dataset_group(tg_name).is_none() {
                return Err(PlanError::InvalidQuery(
                    format!("DatasetGroup '{}' not found in qualified dimension '{}'", tg_name, attr_path)
                ));
            }
        }
    }

    let metric_tg_measures: Vec<(String, Vec<(String, String)>)> = metrics.iter()
        .map(|metric| {
            let mappings = metric.dataset_group_measures();
            (metric.name.clone(), mappings)
        })
        .collect();

    for (metric_name, mappings) in &metric_tg_measures {
        if mappings.is_empty() {
            return Err(PlanError::InvalidQuery(
                format!("Metric '{}' is not a cross-tableGroup metric", metric_name)
            ));
        }
    }

    plan_cross_tablegroup_union(model, dimension_attrs, &metric_tg_measures)
}

/// Unified cross-tableGroup UNION planner.
///
/// 1. Build a branch per tableGroup using build_tablegroup_branch
/// 2. Project each branch to common schema (NULLs for missing columns)
/// 3. UNION all branches
/// 4. Re-aggregate to combine rows
pub fn plan_cross_tablegroup_union(
    model: &SemanticModel,
    dimension_attrs: &[String],
    metric_tg_measures: &[(String, Vec<(String, String)>)],
) -> Result<PlanNode, PlanError> {
    let metric_names: Vec<&str> = metric_tg_measures.iter()
        .map(|(name, _)| name.as_str())
        .collect();

    let mut tg_to_metric_measures: HashMap<String, Vec<(String, String)>> = HashMap::new();

    for (metric_name, tg_measures) in metric_tg_measures {
        for (tg_name, measure_name) in tg_measures {
            tg_to_metric_measures
                .entry(tg_name.clone())
                .or_default()
                .push((metric_name.clone(), measure_name.clone()));
        }
    }

    let mut branches: Vec<PlanNode> = Vec::new();

    for (tg_name, metric_measure_pairs) in &tg_to_metric_measures {
        let dataset_group = model.get_dataset_group(tg_name)
            .ok_or_else(|| PlanError::InvalidQuery(
                format!("DatasetGroup '{}' not found", tg_name)
            ))?;

        let measure_aliases: Vec<(String, String)> = metric_measure_pairs.iter()
            .map(|(metric, measure)| (metric.clone(), measure.clone()))
            .collect();

        let measure_names_for_check: Vec<&str> = measure_aliases.iter()
            .map(|(_, m)| m.as_str())
            .collect();

        if dataset_group.has_partitions() {
            let partitioned_datasets: Vec<&Dataset> = dataset_group.datasets.iter()
                .filter(|d| d.partition.is_some() && measure_names_for_check.iter().all(|m| d.has_measure(m)))
                .collect();
            for dataset in partitioned_datasets {
                let branch = build_tablegroup_branch_for_dataset(
                    model, dataset_group, Some(dataset), dimension_attrs, &measure_aliases
                )?;
                let projected = project_branch_for_union(
                    model, dataset_group, Some(dataset), branch,
                    dimension_attrs, &metric_names, metric_measure_pairs,
                )?;
                branches.push(projected);
            }
        } else {
            let branch = build_tablegroup_branch(model, dataset_group, dimension_attrs, &measure_aliases)?;
            let projected = project_branch_for_union(
                model, dataset_group, None, branch,
                dimension_attrs, &metric_names, metric_measure_pairs,
            )?;
            branches.push(projected);
        }
    }

    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }

    let union = PlanNode::Union(Union { inputs: branches });

    let group_by: Vec<Column> = dimension_attrs.iter()
        .map(|attr| Column::unqualified(attr))
        .collect();

    let aggregates: Vec<AggregateExpr> = metric_names.iter()
        .map(|name| AggregateExpr {
            func: Aggregation::Sum,
            expr: Expr::Column(Column::unqualified(*name)),
            alias: name.to_string(),
        })
        .collect();

    let plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(union),
        group_by: group_by.clone(),
        aggregates,
    });

    let sort_keys: Vec<SortKey> = dimension_attrs.iter()
        .map(|attr| SortKey {
            column: attr.clone(),
            direction: SortDirection::Ascending,
        })
        .collect();

    if !sort_keys.is_empty() {
        Ok(PlanNode::Sort(Sort {
            input: Box::new(plan),
            sort_keys,
        }))
    } else {
        Ok(plan)
    }
}

/// Project a tableGroup branch to the common UNION schema.
fn project_branch_for_union(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    dataset: Option<&Dataset>,
    input: PlanNode,
    dimension_attrs: &[String],
    all_metric_names: &[&str],
    tg_metrics: &[(String, String)],
) -> Result<PlanNode, PlanError> {
    let tg_metric_set: HashSet<&str> = tg_metrics.iter()
        .map(|(m, _)| m.as_str())
        .collect();

    let mut projections = Vec::new();

    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        let (tg_qualifier, dim_name, attr_name) = match parts.len() {
            2 => (None, parts[0], parts[1]),
            3 => (Some(parts[0]), parts[1], parts[2]),
            _ => continue,
        };

        if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
            let value = get_virtual_attribute_value_with_dataset(model, dataset_group, dataset, dim_name, attr_name);
            let expr = match value {
                PlanLiteralValue::String(s) => Expr::Literal(Literal::String(s)),
                PlanLiteralValue::Int64(i) => Expr::Literal(Literal::Int(i)),
                PlanLiteralValue::Float64(f) => Expr::Literal(Literal::Float(f)),
                PlanLiteralValue::Bool(b) => Expr::Literal(Literal::Bool(b)),
                _ => Expr::Literal(Literal::Null("string".to_string())),
            };
            projections.push(ProjectExpr {
                expr,
                alias: attr_path.clone(),
            });
        } else if let Some(qualifier) = tg_qualifier {
            if qualifier == dataset_group.name {
                let semantic_name = format!("{}.{}", dim_name, attr_name);
                projections.push(ProjectExpr {
                    expr: Expr::Column(Column::unqualified(&semantic_name)),
                    alias: attr_path.clone(),
                });
            } else {
                let data_type = model.get_dimension(dim_name)
                    .and_then(|d| d.get_attribute(attr_name))
                    .map(|a| a.data_type.to_string())
                    .unwrap_or_else(|| "string".to_string());
                projections.push(ProjectExpr {
                    expr: Expr::Literal(Literal::Null(data_type)),
                    alias: attr_path.clone(),
                });
            }
        } else {
            let semantic_name = format!("{}.{}", dim_name, attr_name);
            projections.push(ProjectExpr {
                expr: Expr::Column(Column::unqualified(&semantic_name)),
                alias: attr_path.clone(),
            });
        }
    }

    for metric_name in all_metric_names {
        let expr = if tg_metric_set.contains(metric_name) {
            Expr::Column(Column::unqualified(*metric_name))
        } else {
            Expr::Literal(Literal::Null("f64".to_string()))
        };
        projections.push(ProjectExpr {
            expr,
            alias: metric_name.to_string(),
        });
    }

    Ok(PlanNode::Project(Project {
        input: Box::new(input),
        expressions: projections,
    }))
}

/// Build a single branch of a cross-tableGroup query for one measure/tableGroup.
fn build_cross_dataset_group_branch(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    table: &Dataset,
    measure: &Measure,
    dimension_attrs: &[String],
    output_alias: &str,
) -> Result<PlanNode, PlanError> {
    let parsed_attrs: Vec<(String, ParsedDimensionAttr)> = dimension_attrs.iter()
        .map(|attr_path| (attr_path.clone(), ParsedDimensionAttr::parse(attr_path, model)))
        .collect();

    let physical_attrs: Vec<&(String, ParsedDimensionAttr)> = parsed_attrs.iter()
        .filter(|(_, parsed)| {
            !parsed.is_virtual() && parsed.belongs_to_dataset_group(&dataset_group.name)
        })
        .collect();

    let mut unique_dim_attrs: Vec<(String, String)> = Vec::new();
    let mut dim_attr_to_group_idx: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();

    for (_, parsed) in &physical_attrs {
        let key = (parsed.dim_name().to_string(), parsed.attr_name().to_string());
        if !dim_attr_to_group_idx.contains_key(&key) {
            let idx = unique_dim_attrs.len();
            unique_dim_attrs.push(key.clone());
            dim_attr_to_group_idx.insert(key, idx);
        }
    }

    let fact_alias: &str = &table.name;
    let mut columns = Vec::new();
    let mut types = Vec::new();
    let mut joined_dimensions: HashSet<String> = HashSet::new();

    for (dim_name, attr_name) in &unique_dim_attrs {
        if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
            if group_dim.is_degenerate() {
                if let Some(attr) = group_dim.get_attribute(attr_name) {
                    columns.push(attr.column_name().to_string());
                    types.push(attr.data_type.to_string());
                }
            }
        }
    }

    if let MeasureExpr::Column(col) = &measure.expr {
        columns.push(col.clone());
        types.push(measure.data_type().to_string());
    }

    let mut plan = PlanNode::Scan(
        Scan::new(&table.name)
            .with_alias(fact_alias)
            .with_columns(columns, types)
    );

    for (dim_name, _) in &unique_dim_attrs {
        if joined_dimensions.contains(dim_name) {
            continue;
        }
        if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
            if let Some(join_spec) = &group_dim.join {
                if let Some(dimension) = model.get_dimension(dim_name) {
                    if needs_join_for_dimension(table, group_dim, dimension) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        let dim_cols: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.column_name().to_string())
                            .collect();
                        let dim_types: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.data_type.to_string())
                            .collect();
                        let dim_table = dimension.table.as_ref()
                            .expect("Non-virtual dimension must have a table");
                        let dim_scan = PlanNode::Scan(
                            Scan::new(dim_table)
                                .with_alias(dim_alias)
                                .with_columns(dim_cols, dim_types)
                        );
                        let left_key = Column::new(fact_alias, &join_spec.left_key);
                        let right_key = Column::new(
                            join_spec.right_alias.as_deref().unwrap_or(dim_alias),
                            &join_spec.right_key,
                        );
                        plan = PlanNode::Join(Join {
                            left: Box::new(plan),
                            right: Box::new(dim_scan),
                            join_type: JoinType::Left,
                            left_key,
                            right_key,
                        });
                        joined_dimensions.insert(dim_name.clone());
                    }
                }
            }
        }
    }

    let group_by: Vec<Column> = unique_dim_attrs.iter()
        .filter_map(|(dim_name, attr_name)| {
            if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
                if group_dim.is_degenerate() {
                    if let Some(attr) = group_dim.get_attribute(attr_name) {
                        return Some(Column::new(fact_alias, attr.column_name()));
                    }
                } else if let Some(dimension) = model.get_dimension(dim_name) {
                    if let Some(attr) = dimension.get_attribute(attr_name) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        return Some(Column::new(dim_alias, attr.column_name()));
                    }
                }
            }
            None
        })
        .collect();

    let aggregates = vec![
        AggregateExpr {
            func: measure.aggregation,
            expr: convert_measure_expr(&measure.expr),
            alias: output_alias.to_string(),
        }
    ];

    plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(plan),
        group_by: group_by.clone(),
        aggregates,
    });

    let mut projections = Vec::new();

    for (attr_path, parsed) in &parsed_attrs {
        let expr = if parsed.is_virtual() {
            let dim_name = parsed.dim_name();
            let attr_name = parsed.attr_name();
            let value = get_virtual_attribute_value_with_dataset(model, dataset_group, Some(table), dim_name, attr_name);
            match value {
                PlanLiteralValue::String(s) => Expr::Literal(Literal::String(s)),
                PlanLiteralValue::Int64(i) => Expr::Literal(Literal::Int(i)),
                PlanLiteralValue::Float64(f) => Expr::Literal(Literal::Float(f)),
                PlanLiteralValue::Bool(b) => Expr::Literal(Literal::Bool(b)),
                PlanLiteralValue::Null => Expr::Literal(Literal::Null("string".to_string())),
                _ => Expr::Literal(Literal::Null("string".to_string())),
            }
        } else if parsed.belongs_to_dataset_group(&dataset_group.name) {
            let key = (parsed.dim_name().to_string(), parsed.attr_name().to_string());
            if let Some(&idx) = dim_attr_to_group_idx.get(&key) {
                let col = group_by.get(idx).cloned()
                    .unwrap_or_else(|| Column::unqualified(attr_path));
                Expr::Column(col)
            } else {
                let data_type = parsed.get_data_type(model);
                Expr::Literal(Literal::Null(data_type))
            }
        } else {
            let data_type = parsed.get_data_type(model);
            Expr::Literal(Literal::Null(data_type))
        };

        projections.push(ProjectExpr {
            expr,
            alias: attr_path.clone(),
        });
    }

    projections.push(ProjectExpr {
        expr: Expr::Column(Column::unqualified(output_alias)),
        alias: output_alias.to_string(),
    });

    Ok(PlanNode::Project(Project {
        input: Box::new(plan),
        expressions: projections,
    }))
}
