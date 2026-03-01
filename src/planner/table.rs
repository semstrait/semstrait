//! Resolver-based single-table planning and the unified tableGroup branch builder
//!
//! Contains:
//! - `plan_query` — builds a plan from a ResolvedQuery (single dataset)
//! - `build_tablegroup_branch` / `build_tablegroup_branch_for_dataset` — the unified
//!   builder for within-datasetGroup plans (auto-selects single or multi-table)
//! - `build_single_table_aggregate` / `build_multi_table_aggregate` — lower-level builders

use std::collections::{HashMap, HashSet};
use crate::semantic_model::{MeasureExpr, Dataset, DatasetGroup, SemanticModel};
use crate::plan::{
    Aggregate, AggregateExpr, Column, CrossJoin, Expr, Filter, Join, JoinType,
    PlanNode, Scan, Project, ProjectExpr, Sort,
};
use crate::resolver::{ResolvedQuery, ResolvedDimension};
use super::error::PlanError;
use super::util::{needs_join_for_dimension, build_column, build_attribute_expr, build_sort_keys, collect_required_columns, extract_physical_dims};
use super::expr::{convert_measure_expr, convert_metric_expr, build_filter_expr};

/// Build a logical plan from a resolved query (single-dataset path).
pub fn plan_query(resolved: &ResolvedQuery<'_>) -> Result<PlanNode, PlanError> {
    let (fact_columns, fact_types, dimension_columns) = collect_required_columns(resolved);

    let fact_table = &resolved.dataset.name;
    let fact_alias = &resolved.dataset.name;

    let mut plan: PlanNode = PlanNode::Scan(
        Scan::new(fact_table)
            .with_alias(fact_alias)
            .with_columns(fact_columns, fact_types)
    );

    for dim in &resolved.dimensions {
        if let ResolvedDimension::Joined { group_dim, dimension } = dim {
            if needs_join_for_dimension(resolved.dataset, group_dim, dimension) {
                if let Some(join_spec) = &group_dim.join {
                    let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                    let (dim_cols, dim_types) = dimension_columns
                        .get(&dimension.name)
                        .cloned()
                        .unwrap_or_default();
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
                }
            }
        }
    }

    if !resolved.filters.is_empty() {
        let filter_exprs: Vec<Expr> = resolved.filters
            .iter()
            .map(|f| build_filter_expr(f, fact_alias))
            .collect();
        let predicate = if filter_exprs.len() == 1 {
            filter_exprs.into_iter().next().unwrap()
        } else {
            Expr::And(filter_exprs)
        };
        plan = PlanNode::Filter(Filter {
            input: Box::new(plan),
            predicate,
        });
    }

    let group_by: Vec<Column> = resolved.row_attributes.iter()
        .chain(resolved.column_attributes.iter())
        .filter(|attr| !attr.is_meta())
        .map(|attr| build_column(attr, resolved.dataset, fact_alias))
        .collect();

    let aggregates: Vec<AggregateExpr> = resolved.measures
        .iter()
        .map(|measure| AggregateExpr {
            func: measure.aggregation,
            expr: convert_measure_expr(&measure.expr),
            alias: measure.name.clone(),
        })
        .collect();

    if !group_by.is_empty() || !aggregates.is_empty() {
        plan = PlanNode::Aggregate(Aggregate {
            input: Box::new(plan),
            group_by: group_by.clone(),
            aggregates,
        });
    }

    let has_meta_attrs = resolved.row_attributes.iter()
        .chain(resolved.column_attributes.iter())
        .any(|attr| attr.is_meta());

    if !resolved.metrics.is_empty() || has_meta_attrs {
        let mut projections = Vec::new();
        for attr in resolved.row_attributes.iter().chain(resolved.column_attributes.iter()) {
            let semantic_name = format!("{}.{}", attr.dimension_name(), attr.attribute_name());
            let expr = build_attribute_expr(attr, resolved.dataset, fact_alias);
            projections.push(ProjectExpr {
                expr,
                alias: semantic_name,
            });
        }
        for metric in &resolved.metrics {
            projections.push(ProjectExpr {
                expr: convert_metric_expr(&metric.expr),
                alias: metric.name.clone(),
            });
        }
        plan = PlanNode::Project(Project {
            input: Box::new(plan),
            expressions: projections,
        });
    }

    let sort_keys = build_sort_keys(resolved);
    if !sort_keys.is_empty() {
        plan = PlanNode::Sort(Sort {
            input: Box::new(plan),
            sort_keys,
        });
    }

    Ok(plan)
}

// ============================================================================
// UNIFIED TABLEGROUP BRANCH BUILDER
// ============================================================================

/// Build an aggregated plan for a single tableGroup.
///
/// Handles single table selection, multi-table JOIN, dimension JOINs, and aggregation.
pub fn build_tablegroup_branch(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    dimension_attrs: &[String],
    measure_aliases: &[(String, String)],
) -> Result<PlanNode, PlanError> {
    build_tablegroup_branch_for_dataset(model, dataset_group, None, dimension_attrs, measure_aliases)
}

/// Build an aggregated branch, optionally targeting a specific dataset.
pub fn build_tablegroup_branch_for_dataset(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    target_dataset: Option<&Dataset>,
    dimension_attrs: &[String],
    measure_aliases: &[(String, String)],
) -> Result<PlanNode, PlanError> {
    let physical_dims = extract_physical_dims(dimension_attrs, model);

    let measure_names: Vec<&str> = measure_aliases.iter()
        .map(|(_, m)| m.as_str())
        .collect();

    if let Some(table) = target_dataset {
        build_single_table_aggregate(model, dataset_group, table, &physical_dims, measure_aliases, None)
    } else {
        let single_table = dataset_group.datasets.iter()
            .find(|t| measure_names.iter().all(|m| t.has_measure(m)));

        if let Some(table) = single_table {
            build_single_table_aggregate(model, dataset_group, table, &physical_dims, measure_aliases, None)
        } else {
            build_multi_table_aggregate(model, dataset_group, &physical_dims, measure_aliases)
        }
    }
}

/// Build an aggregated plan from a single table.
pub fn build_single_table_aggregate(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    table: &Dataset,
    physical_dims: &[(String, String)],
    measure_aliases: &[(String, String)],
    output_prefix: Option<&str>,
) -> Result<PlanNode, PlanError> {
    let fact_alias = &table.name;
    let mut columns = Vec::new();
    let mut types = Vec::new();
    let mut joined_dimensions: HashSet<String> = HashSet::new();

    for (dim_name, attr_name) in physical_dims {
        if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
            if group_dim.is_degenerate() {
                if let Some(attr) = group_dim.get_attribute(attr_name) {
                    let col_name = attr.column_name().to_string();
                    if !columns.contains(&col_name) {
                        columns.push(col_name);
                        types.push(attr.data_type.to_string());
                    }
                }
            }
        }
    }

    for (_, measure_name) in measure_aliases {
        if let Some(measure) = dataset_group.get_measure(measure_name) {
            if let MeasureExpr::Column(col) = &measure.expr {
                if !columns.contains(col) {
                    columns.push(col.clone());
                    types.push(measure.data_type().to_string());
                }
            }
        }
    }

    let mut plan = PlanNode::Scan(
        Scan::new(&table.name)
            .with_alias(fact_alias)
            .with_columns(columns, types)
    );

    for (dim_name, _) in physical_dims {
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

    let group_by: Vec<Column> = physical_dims.iter()
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

    let aggregates: Vec<AggregateExpr> = measure_aliases.iter()
        .filter_map(|(alias, measure_name)| {
            dataset_group.get_measure(measure_name).map(|measure| {
                AggregateExpr {
                    func: measure.aggregation,
                    expr: convert_measure_expr(&measure.expr),
                    alias: alias.clone(),
                }
            })
        })
        .collect();

    let agg_plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(plan),
        group_by: group_by.clone(),
        aggregates,
    });

    let mut projections = Vec::new();

    for (idx, (dim_name, attr_name)) in physical_dims.iter().enumerate() {
        let semantic_name = format!("{}.{}", dim_name, attr_name);
        let output_name = if let Some(prefix) = output_prefix {
            format!("{}.{}", prefix, semantic_name)
        } else {
            semantic_name
        };
        if let Some(col) = group_by.get(idx) {
            projections.push(ProjectExpr {
                expr: Expr::Column(col.clone()),
                alias: output_name,
            });
        }
    }

    for (alias, _) in measure_aliases {
        let output_name = if let Some(prefix) = output_prefix {
            format!("{}.{}", prefix, alias)
        } else {
            alias.clone()
        };
        projections.push(ProjectExpr {
            expr: Expr::Column(Column::unqualified(alias)),
            alias: output_name,
        });
    }

    Ok(PlanNode::Project(Project {
        input: Box::new(agg_plan),
        expressions: projections,
    }))
}

/// Build an aggregated plan by JOINing multiple tables within one datasetGroup.
fn build_multi_table_aggregate(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    physical_dims: &[(String, String)],
    measure_aliases: &[(String, String)],
) -> Result<PlanNode, PlanError> {
    let mut tables_sorted: Vec<&Dataset> = dataset_group.datasets.iter().collect();
    tables_sorted.sort_by_key(|t| t.attribute_count());

    let mut table_measures: HashMap<usize, Vec<(String, String)>> = HashMap::new();
    let mut assigned_measures: HashSet<&str> = HashSet::new();

    for (alias, measure_name) in measure_aliases {
        if assigned_measures.contains(measure_name.as_str()) {
            continue;
        }
        for (idx, table) in tables_sorted.iter().enumerate() {
            if table.has_measure(measure_name) {
                table_measures.entry(idx)
                    .or_default()
                    .push((alias.clone(), measure_name.clone()));
                assigned_measures.insert(measure_name.as_str());
                break;
            }
        }
    }

    if table_measures.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No tables found for required measures".to_string()
        ));
    }

    let mut sub_queries: Vec<(PlanNode, String)> = Vec::new();

    for (idx, measures) in &table_measures {
        let table = tables_sorted[*idx];
        let table_alias = format!("t{}", idx);
        let sub_plan = build_single_table_aggregate(
            model, dataset_group, table, physical_dims, measures, Some(&table_alias),
        )?;
        sub_queries.push((sub_plan, table_alias));
    }

    if sub_queries.len() == 1 {
        let (plan, prefix) = sub_queries.into_iter().next().unwrap();
        let mut projections = Vec::new();
        for (dim_name, attr_name) in physical_dims {
            let prefixed = format!("{}.{}.{}", prefix, dim_name, attr_name);
            let semantic_name = format!("{}.{}", dim_name, attr_name);
            projections.push(ProjectExpr {
                expr: Expr::Column(Column::unqualified(&prefixed)),
                alias: semantic_name,
            });
        }
        for (alias, _) in measure_aliases {
            let prefixed = format!("{}.{}", prefix, alias);
            projections.push(ProjectExpr {
                expr: Expr::Column(Column::unqualified(&prefixed)),
                alias: alias.clone(),
            });
        }
        return Ok(PlanNode::Project(Project {
            input: Box::new(plan),
            expressions: projections,
        }));
    }

    let (first_plan, first_alias) = sub_queries.remove(0);
    let mut joined_plan = first_plan;
    let mut left_alias = first_alias;

    for (right_plan, right_alias) in sub_queries {
        if let Some((dim_name, attr_name)) = physical_dims.first() {
            let semantic_name = format!("{}.{}", dim_name, attr_name);
            let left_col_name = format!("{}.{}", left_alias, semantic_name);
            let right_col_name = format!("{}.{}", right_alias, semantic_name);
            let left_key = Column::unqualified(&left_col_name);
            let right_key = Column::unqualified(&right_col_name);
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

    for (dim_name, attr_name) in physical_dims {
        let semantic_name = format!("{}.{}", dim_name, attr_name);
        let table_indices: Vec<usize> = table_measures.keys().cloned().collect();
        let coalesce_args: Vec<Expr> = table_indices.iter()
            .map(|idx| {
                let col_name = format!("t{}.{}", idx, semantic_name);
                Expr::Column(Column::unqualified(&col_name))
            })
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

    for (alias, measure_name) in measure_aliases {
        let table_idx = table_measures.iter()
            .find(|(_, measures)| measures.iter().any(|(_, m)| m == measure_name))
            .map(|(idx, _)| *idx);
        if let Some(idx) = table_idx {
            let col_name = format!("t{}.{}", idx, alias);
            projections.push(ProjectExpr {
                expr: Expr::Column(Column::unqualified(&col_name)),
                alias: alias.clone(),
            });
        }
    }

    Ok(PlanNode::Project(Project {
        input: Box::new(joined_plan),
        expressions: projections,
    }))
}
