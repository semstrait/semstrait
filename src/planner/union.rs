//! Union-based planning: conformed, qualified, partitioned, and virtual-only queries

use std::collections::HashSet;
use crate::semantic_model::{MeasureExpr, MetricExpr, Dataset, DatasetGroup, Schema, SemanticModel};
use crate::plan::{
    Aggregate, AggregateExpr, Column, Expr, Join, JoinType,
    Literal, PlanNode, Scan, Project, ProjectExpr, Union,
    VirtualTable, LiteralValue as PlanLiteralValue,
};
use crate::resolver::resolve_query;
use crate::selector::SelectedDataset;
use crate::query::QueryRequest;
use super::error::PlanError;
use super::util::{needs_join_for_dimension, ParsedDimensionAttr, get_virtual_attribute_value, get_virtual_attribute_value_with_dataset};
use super::expr::convert_measure_expr;
use super::table::plan_query;

/// Plan a UNION ALL query across partitioned datasets within the same datasetGroup.
pub fn plan_partitioned_union(
    schema: &Schema,
    _model: &SemanticModel,
    request: &QueryRequest,
    partitions: &[SelectedDataset],
) -> Result<PlanNode, PlanError> {
    if partitions.is_empty() {
        return Err(PlanError::InvalidQuery("No partitioned datasets to union".to_string()));
    }
    if partitions.len() == 1 {
        let resolved = resolve_query(schema, request, &partitions[0])
            .map_err(|e| PlanError::InvalidQuery(format!("Query resolution error: {:?}", e)))?;
        return plan_query(&resolved);
    }

    let mut branch_plans: Vec<PlanNode> = Vec::new();
    for partition in partitions {
        let resolved = resolve_query(schema, request, partition)
            .map_err(|e| PlanError::InvalidQuery(format!("Query resolution error: {:?}", e)))?;
        let plan = plan_query(&resolved)?;
        branch_plans.push(plan);
    }

    Ok(PlanNode::Union(Union {
        inputs: branch_plans,
    }))
}

/// Plan a query on conformed dimensions across multiple tableGroups.
///
/// Triggered when all queried dimensions are conformed and there are multiple tableGroups.
/// Special case: virtual-only queries produce a VirtualTable instead.
pub fn plan_conformed_query(
    schema: &Schema,
    model: &SemanticModel,
    request: &QueryRequest,
    dimension_attrs: &[String],
) -> Result<PlanNode, PlanError> {
    let physical_dims: Vec<String> = dimension_attrs.iter()
        .filter(|d| {
            let parts: Vec<&str> = d.split('.').collect();
            if parts.len() != 2 { return true; }
            !model.get_dimension(parts[0])
                .map(|dim| dim.is_virtual())
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    let virtual_dims: Vec<String> = dimension_attrs.iter()
        .filter(|d| {
            let parts: Vec<&str> = d.split('.').collect();
            if parts.len() != 2 { return false; }
            model.get_dimension(parts[0])
                .map(|dim| dim.is_virtual())
                .unwrap_or(false)
        })
        .cloned()
        .collect();

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

    if physical_dims.is_empty() && metric_names.is_empty() && !virtual_dims.is_empty() {
        return plan_virtual_only_query(model, &virtual_dims);
    }

    let mut branches: Vec<PlanNode> = Vec::new();

    let is_feasible_table = |dataset_group: &DatasetGroup, table: &Dataset| -> bool {
        for dim_attr in &physical_dims {
            let parts: Vec<&str> = dim_attr.split('.').collect();
            if parts.len() != 2 { return false; }
            let (dim_name, attr_name) = (parts[0], parts[1]);
            if let Some(attrs) = table.get_dimension_attributes(dim_name) {
                if !attrs.iter().any(|a| a == attr_name) { return false; }
            } else {
                return false;
            }
        }
        for measure_name in &required_measures {
            if dataset_group.get_measure(measure_name).is_none() { return false; }
            if !table.has_measure(measure_name) { return false; }
        }
        true
    };

    for dataset_group in &model.dataset_groups {
        if dataset_group.has_partitions() {
            let feasible_partitions: Vec<&Dataset> = dataset_group.datasets.iter()
                .filter(|t| t.partition.is_some() && is_feasible_table(dataset_group, t))
                .collect();
            for table in feasible_partitions {
                let selected = SelectedDataset { group: dataset_group, dataset: table };
                let resolved = resolve_query(schema, request, &selected)
                    .map_err(|e| PlanError::InvalidQuery(format!(
                        "Query resolution error for tableGroup '{}' partition '{}': {:?}",
                        dataset_group.name, table.partition.as_deref().unwrap_or("?"), e
                    )))?;
                let branch = plan_query(&resolved)?;
                branches.push(branch);
            }
        } else {
            let feasible_table = dataset_group.datasets.iter()
                .find(|t| is_feasible_table(dataset_group, t));
            let Some(table) = feasible_table else { continue; };
            let selected = SelectedDataset { group: dataset_group, dataset: table };
            let resolved = resolve_query(schema, request, &selected)
                .map_err(|e| PlanError::InvalidQuery(format!(
                    "Query resolution error for tableGroup '{}': {:?}",
                    dataset_group.name, e
                )))?;
            let branch = plan_query(&resolved)?;
            branches.push(branch);
        }
    }

    if branches.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No tableGroup can serve this conformed dimension query".to_string()
        ));
    }
    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }

    Ok(PlanNode::Union(Union { inputs: branches }))
}

/// Plan a query with dimensions qualified for multiple different tableGroups.
pub fn plan_multi_tablegroup_query(
    _schema: &Schema,
    model: &SemanticModel,
    request: &QueryRequest,
    dimension_attrs: &[String],
    qualified_groups: &HashSet<&str>,
) -> Result<PlanNode, PlanError> {
    let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
    let mut branches: Vec<PlanNode> = Vec::new();

    for dataset_group in &model.dataset_groups {
        if !qualified_groups.contains(dataset_group.name.as_str()) {
            continue;
        }
        let feasible_table = find_feasible_table_for_qualified(
            model, dataset_group, dimension_attrs, &metric_names
        );
        let Some(table) = feasible_table else {
            return Err(PlanError::InvalidQuery(format!(
                "No table in tableGroup '{}' can serve the qualified dimension query",
                dataset_group.name
            )));
        };
        let branch = build_union_branch(
            model, dataset_group, table, dimension_attrs, &metric_names,
        )?;
        branches.push(branch);
    }

    if branches.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No tableGroup can serve this qualified dimension query".to_string()
        ));
    }
    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }

    Ok(PlanNode::Union(Union { inputs: branches }))
}

/// Plan a query constrained to a single tableGroup (via qualified dimension).
pub fn plan_single_tablegroup_query(
    schema: &Schema,
    model: &SemanticModel,
    request: &QueryRequest,
    dimension_attrs: &[String],
    target_group: &str,
) -> Result<PlanNode, PlanError> {
    let dataset_group = model.dataset_groups.iter()
        .find(|tg| tg.name == target_group)
        .ok_or_else(|| PlanError::InvalidQuery(format!(
            "DatasetGroup '{}' not found in model", target_group
        )))?;

    let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
    let feasible_table = find_feasible_table_for_qualified(
        model, dataset_group, dimension_attrs, &metric_names
    );
    let Some(table) = feasible_table else {
        return Err(PlanError::InvalidQuery(format!(
            "No table in tableGroup '{}' can serve the qualified dimension query",
            target_group
        )));
    };

    let normalized_dims: Vec<String> = dimension_attrs.iter()
        .map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 && parts[0] == target_group {
                format!("{}.{}", parts[1], parts[2])
            } else {
                path.clone()
            }
        })
        .collect();

    let normalized_request = QueryRequest {
        model: request.model.clone(),
        dimensions: None,
        rows: Some(normalized_dims.iter()
            .filter(|d| request.rows.as_ref().map(|r| r.iter().any(|rd| {
                let parts: Vec<&str> = rd.split('.').collect();
                if parts.len() == 3 {
                    format!("{}.{}", parts[1], parts[2]) == **d
                } else {
                    rd == *d
                }
            })).unwrap_or(false))
            .cloned()
            .collect()),
        columns: request.columns.as_ref().map(|cols| {
            cols.iter()
                .map(|c| {
                    let parts: Vec<&str> = c.split('.').collect();
                    if parts.len() == 3 && parts[0] == target_group {
                        format!("{}.{}", parts[1], parts[2])
                    } else {
                        c.clone()
                    }
                })
                .collect()
        }),
        metrics: request.metrics.clone(),
        filter: request.filter.clone(),
    };

    let selected = SelectedDataset { group: dataset_group, dataset: table };
    let resolved = resolve_query(schema, &normalized_request, &selected)
        .map_err(|e| PlanError::InvalidQuery(format!(
            "Query resolution error for tableGroup '{}': {:?}",
            target_group, e
        )))?;

    plan_query(&resolved)
}

/// Find a feasible table in a tableGroup for qualified dimension queries.
pub fn find_feasible_table_for_qualified<'a>(
    model: &SemanticModel,
    dataset_group: &'a DatasetGroup,
    dimension_attrs: &[String],
    metric_names: &[String],
) -> Option<&'a Dataset> {
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

    dataset_group.datasets.iter().find(|table| {
        for dim_attr in dimension_attrs {
            let parts: Vec<&str> = dim_attr.split('.').collect();
            if parts.len() == 3 {
                let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
                if tg_qualifier != dataset_group.name { continue; }
                if let Some(attrs) = table.get_dimension_attributes(dim_name) {
                    if !attrs.iter().any(|a| a == attr_name) { return false; }
                } else {
                    return false;
                }
            } else if parts.len() == 2 {
                let (dim_name, attr_name) = (parts[0], parts[1]);
                if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                    continue;
                }
                if let Some(attrs) = table.get_dimension_attributes(dim_name) {
                    if !attrs.iter().any(|a| a == attr_name) { return false; }
                } else {
                    return false;
                }
            }
        }
        for measure_name in &required_measures {
            if dataset_group.get_measure(measure_name).is_none() { return false; }
            if !table.has_measure(measure_name) { return false; }
        }
        true
    })
}

/// Build a single branch for a multi-tableGroup UNION query with NULL projection.
fn build_union_branch(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    table: &Dataset,
    dimension_attrs: &[String],
    metric_names: &[String],
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

    let measures_to_aggregate: Vec<(&str, &crate::semantic_model::Measure)> = metric_names.iter()
        .filter_map(|metric_name| {
            model.get_metric(metric_name).and_then(|m| {
                match &m.expr {
                    MetricExpr::MeasureRef(measure_name) => {
                        dataset_group.get_measure(measure_name)
                            .map(|measure| (metric_name.as_str(), measure))
                    }
                    MetricExpr::Structured(_) => None,
                }
            })
        })
        .collect();

    for (_, measure) in &measures_to_aggregate {
        if let MeasureExpr::Column(col) = &measure.expr {
            columns.push(col.clone());
            types.push(measure.data_type().to_string());
        }
    }

    let mut plan = PlanNode::Scan(
        Scan::new(&table.name)
            .with_alias(fact_alias)
            .with_columns(columns, types)
    );

    for (dim_name, _) in &unique_dim_attrs {
        if joined_dimensions.contains(dim_name) { continue; }
        if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
            if let Some(join_spec) = &group_dim.join {
                if let Some(dimension) = model.get_dimension(dim_name) {
                    if needs_join_for_dimension(table, group_dim, dimension) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        let dim_cols: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.column_name().to_string()).collect();
                        let dim_types: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.data_type.to_string()).collect();
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

    let aggregates: Vec<AggregateExpr> = measures_to_aggregate.iter()
        .map(|(metric_name, measure)| {
            AggregateExpr {
                func: measure.aggregation,
                expr: convert_measure_expr(&measure.expr),
                alias: metric_name.to_string(),
            }
        })
        .collect();

    if !group_by.is_empty() || !aggregates.is_empty() {
        plan = PlanNode::Aggregate(Aggregate {
            input: Box::new(plan),
            group_by: group_by.clone(),
            aggregates,
        });
    }

    let mut projections = Vec::new();

    for (attr_path, parsed) in &parsed_attrs {
        let expr = if parsed.is_virtual() {
            let dim_name = parsed.dim_name();
            let attr_name = parsed.attr_name();
            let value = get_virtual_attribute_value(model, dataset_group, dim_name, attr_name);
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

    for metric_name in metric_names {
        projections.push(ProjectExpr {
            expr: Expr::Column(Column::unqualified(metric_name)),
            alias: metric_name.clone(),
        });
    }

    Ok(PlanNode::Project(Project {
        input: Box::new(plan),
        expressions: projections,
    }))
}

/// Plan a virtual-only query that doesn't need table scans.
pub fn plan_virtual_only_query(
    model: &SemanticModel,
    virtual_dims: &[String],
) -> Result<PlanNode, PlanError> {
    let attrs: Vec<(&str, &str)> = virtual_dims.iter()
        .filter_map(|d| {
            let parts: Vec<&str> = d.split('.').collect();
            if parts.len() == 2 { Some((parts[0], parts[1])) } else { None }
        })
        .collect();

    if attrs.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No valid virtual dimension attributes in query".to_string()
        ));
    }

    let columns: Vec<String> = virtual_dims.iter().cloned().collect();
    let column_types: Vec<String> = attrs.iter().map(|_| "string".to_string()).collect();

    let mut rows: Vec<Vec<PlanLiteralValue>> = Vec::new();

    for dataset_group in &model.dataset_groups {
        if dataset_group.has_partitions() {
            for dataset in &dataset_group.datasets {
                if dataset.partition.is_some() {
                    let row: Vec<PlanLiteralValue> = attrs.iter()
                        .map(|(dim_name, attr_name)| {
                            get_virtual_attribute_value_with_dataset(model, dataset_group, Some(dataset), dim_name, attr_name)
                        })
                        .collect();
                    rows.push(row);
                }
            }
        } else {
            let row: Vec<PlanLiteralValue> = attrs.iter()
                .map(|(dim_name, attr_name)| {
                    get_virtual_attribute_value(model, dataset_group, dim_name, attr_name)
                })
                .collect();
            rows.push(row);
        }
    }

    Ok(PlanNode::VirtualTable(VirtualTable {
        columns,
        column_types,
        rows,
    }))
}
