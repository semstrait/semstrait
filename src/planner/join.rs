//! Same-datasetGroup multi-table JOIN planning
//!
//! Used when no single table has all required measures but multiple tables
//! together can satisfy the query. Tables are JOINed on their common dimensions.

use crate::semantic_model::{MeasureExpr, MetricExpr, Dataset, DatasetGroup, SemanticModel};
use crate::plan::{
    Aggregate, AggregateExpr, Column, CrossJoin, Expr, Join, JoinType,
    Literal, PlanNode, Scan, Project, ProjectExpr, Sort, SortKey, SortDirection,
    LiteralValue as PlanLiteralValue,
};
use crate::selector::MultiDatasetSelection;
use super::error::PlanError;
use super::util::{get_virtual_attribute_value, get_dimension_column_name};
use super::expr::convert_measure_expr;

/// Plan a query that JOINs multiple tables within the same tableGroup.
pub fn plan_same_tablegroup_join(
    model: &SemanticModel,
    selection: &MultiDatasetSelection<'_>,
    dimension_attrs: &[String],
    metric_names: &[String],
) -> Result<PlanNode, PlanError> {
    let dataset_group = selection.group;

    let physical_dims: Vec<(String, String)> = dimension_attrs.iter()
        .filter_map(|attr_path| {
            let parts: Vec<&str> = attr_path.split('.').collect();
            if parts.len() == 2 {
                let dim_name = parts[0];
                if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                    return None;
                }
                Some((dim_name.to_string(), parts[1].to_string()))
            } else if parts.len() == 3 {
                let dim_name = parts[1];
                if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                    return None;
                }
                Some((dim_name.to_string(), parts[2].to_string()))
            } else {
                None
            }
        })
        .collect();

    if selection.datasets.is_empty() {
        return Err(PlanError::InvalidQuery("No tables selected for JOIN".to_string()));
    }

    let mut sub_queries: Vec<(PlanNode, String)> = Vec::new();

    for (idx, table_with_measures) in selection.datasets.iter().enumerate() {
        let table = table_with_measures.dataset;
        let measures = &table_with_measures.measures;
        let table_alias = format!("t{}", idx);
        let sub_plan = build_table_subquery(
            model, dataset_group, table, &physical_dims, measures, &table_alias,
        )?;
        sub_queries.push((sub_plan, table_alias));
    }

    let (first_plan, first_alias) = sub_queries.remove(0);
    let mut joined_plan = first_plan;
    let mut left_alias = first_alias;

    for (right_plan, right_alias) in sub_queries {
        if let Some((dim_name, attr_name)) = physical_dims.first() {
            let col_name = get_dimension_column_name(dataset_group, dim_name, attr_name);
            let left_key = Column::new(&left_alias, &col_name);
            let right_key = Column::new(&right_alias, &col_name);
            joined_plan = PlanNode::Join(Join {
                left: Box::new(joined_plan),
                right: Box::new(right_plan),
                join_type: JoinType::Full,
                left_key,
                right_key,
            });
            left_alias = format!("{}_{}", left_alias, right_alias);
        } else {
            joined_plan = PlanNode::CrossJoin(CrossJoin {
                left: Box::new(joined_plan),
                right: Box::new(right_plan),
            });
            left_alias = format!("{}_{}", left_alias, right_alias);
        }
    }

    let mut projections = Vec::new();

    for (dim_name, attr_name) in &physical_dims {
        let col_name = get_dimension_column_name(dataset_group, dim_name, attr_name);
        let semantic_name = format!("{}.{}", dim_name, attr_name);
        let coalesce_args: Vec<Expr> = selection.datasets.iter().enumerate()
            .map(|(idx, _)| Expr::Column(Column::new(&format!("t{}", idx), &col_name)))
            .collect();
        let expr = if coalesce_args.len() == 1 {
            coalesce_args.into_iter().next().unwrap()
        } else {
            Expr::Coalesce(coalesce_args)
        };
        projections.push(ProjectExpr {
            expr,
            alias: semantic_name,
        });
    }

    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        if parts.len() == 2 {
            let dim_name = parts[0];
            let attr_name = parts[1];
            if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                let value = get_virtual_attribute_value(model, dataset_group, dim_name, attr_name);
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
            }
        }
    }

    for metric_name in metric_names {
        let table_idx = selection.datasets.iter().position(|twm| {
            twm.measures.iter().any(|m| {
                if let Some(metric) = model.get_metric(metric_name) {
                    match &metric.expr {
                        MetricExpr::MeasureRef(measure_name) => measure_name == m,
                        MetricExpr::Structured(_) => false,
                    }
                } else {
                    false
                }
            })
        });
        if let Some(idx) = table_idx {
            projections.push(ProjectExpr {
                expr: Expr::Column(Column::new(&format!("t{}", idx), metric_name)),
                alias: metric_name.clone(),
            });
        }
    }

    let final_plan = PlanNode::Project(Project {
        input: Box::new(joined_plan),
        expressions: projections,
    });

    let sort_keys: Vec<SortKey> = physical_dims.iter()
        .map(|(dim_name, attr_name)| SortKey {
            column: format!("{}.{}", dim_name, attr_name),
            direction: SortDirection::Ascending,
        })
        .collect();

    if sort_keys.is_empty() {
        Ok(final_plan)
    } else {
        Ok(PlanNode::Sort(Sort {
            input: Box::new(final_plan),
            sort_keys,
        }))
    }
}

/// Build a sub-query for one table in a multi-table JOIN.
fn build_table_subquery(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    table: &Dataset,
    physical_dims: &[(String, String)],
    measures: &[String],
    alias: &str,
) -> Result<PlanNode, PlanError> {
    let mut columns = Vec::new();
    let mut types = Vec::new();

    for (dim_name, attr_name) in physical_dims {
        if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
            if group_dim.is_degenerate() {
                if let Some(attr) = group_dim.get_attribute(attr_name) {
                    columns.push(attr.column_name().to_string());
                    types.push(attr.data_type.to_string());
                }
            }
        }
    }

    for measure_name in measures {
        if let Some(measure) = dataset_group.get_measure(measure_name) {
            if let MeasureExpr::Column(col) = &measure.expr {
                columns.push(col.clone());
                types.push(measure.data_type().to_string());
            }
        }
    }

    let scan = PlanNode::Scan(
        Scan::new(&table.name)
            .with_alias(alias)
            .with_columns(columns, types)
    );

    let group_by: Vec<Column> = physical_dims.iter()
        .filter_map(|(dim_name, attr_name)| {
            if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
                if group_dim.is_degenerate() {
                    if let Some(attr) = group_dim.get_attribute(attr_name) {
                        return Some(Column::new(alias, attr.column_name()));
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

    let aggregates: Vec<AggregateExpr> = measures.iter()
        .filter_map(|measure_name| {
            dataset_group.get_measure(measure_name).map(|measure| {
                let metric_name = model.metrics.as_ref()
                    .and_then(|metrics| {
                        metrics.iter().find(|m| {
                            matches!(&m.expr, MetricExpr::MeasureRef(name) if name == measure_name)
                        })
                    })
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| measure_name.clone());
                AggregateExpr {
                    func: measure.aggregation,
                    expr: convert_measure_expr(&measure.expr),
                    alias: metric_name,
                }
            })
        })
        .collect();

    let mut projections = Vec::new();

    for (dim_name, attr_name) in physical_dims {
        let col_name = get_dimension_column_name(dataset_group, dim_name, attr_name);
        projections.push(ProjectExpr {
            expr: Expr::Column(Column::unqualified(&col_name)),
            alias: col_name.clone(),
        });
    }

    for agg in &aggregates {
        projections.push(ProjectExpr {
            expr: Expr::Column(Column::unqualified(&agg.alias)),
            alias: agg.alias.clone(),
        });
    }

    let plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(scan),
        group_by,
        aggregates,
    });

    Ok(PlanNode::Project(Project {
        input: Box::new(plan),
        expressions: projections,
    }))
}
