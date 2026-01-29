use std::collections::HashSet;
use crate::model::{Schema, Model, TableGroup, GroupTable, Metric, MetricExpr, MetricExprNode, MetricExprArg};
use crate::query::{DataFilter, QueryRequest};
use crate::selector::SelectedTable;
use super::error::ResolveError;
use super::types::{AttributeRef, ResolvedDimension, ResolvedFilter, ResolvedQuery};

/// Resolve an analytics query request against a schema and selected table
/// 
/// This turns string references (model name, metric names, "dimension.attribute" strings)
/// into actual schema object references.
/// 
/// # Arguments
/// * `schema` - The schema containing dimension definitions
/// * `request` - The query request with model name, metrics, dimensions, etc.
/// * `selected` - The selected table (from selector) containing both group and table
pub fn resolve_query<'a>(
    schema: &'a Schema,
    request: &QueryRequest,
    selected: &SelectedTable<'a>,
) -> Result<ResolvedQuery<'a>, ResolveError> {
    let group = selected.group;
    let table = selected.table;
    
    // 1. Resolve the model (for metrics which are model-level)
    let model = schema
        .get_model(&request.model)
        .ok_or_else(|| ResolveError::ModelNotFound(request.model.clone()))?;

    // 2. Resolve row attributes (e.g., ["dates.year", "markets.country"])
    let row_attributes = resolve_attributes(model, group, table, request.rows.as_ref())?;

    // 3. Resolve column attributes
    let column_attributes = resolve_attributes(model, group, table, request.columns.as_ref())?;

    // 4. Resolve metrics (from model - metrics are model-level)
    let metrics = resolve_metrics(model, request.metrics.as_ref())?;

    // 5. Collect all measures needed by metrics
    let measures = collect_metric_measures(group, table, &metrics)?;

    // 6. Resolve filters
    let filters = resolve_filters(model, group, table, request.filter.as_ref())?;

    // 7. Collect all unique dimensions needed for this query (including filter dimensions)
    let dimensions = collect_dimensions(&row_attributes, &column_attributes, &filters);

    Ok(ResolvedQuery {
        model,
        table_group: group,
        table,
        dimensions,
        row_attributes,
        column_attributes,
        measures,
        metrics,
        filters,
    })
}

/// Resolve filters to ResolvedFilter objects
fn resolve_filters<'a>(
    model: &'a Model,
    group: &'a TableGroup,
    table: &'a GroupTable,
    filters: Option<&Vec<DataFilter>>,
) -> Result<Vec<ResolvedFilter<'a>>, ResolveError> {
    let Some(filter_list) = filters else {
        return Ok(vec![]);
    };

    filter_list
        .iter()
        .map(|f| {
            let attribute = resolve_attribute(model, group, table, &f.field)?;
            // Default operator: "in" for arrays, "eq" for single values
            let operator = f.operator.clone().unwrap_or_else(|| {
                if f.value.is_array() { "in".to_string() } else { "eq".to_string() }
            });
            Ok(ResolvedFilter {
                attribute,
                operator,
                value: f.value.clone(),
            })
        })
        .collect()
}

/// Resolve a list of "dimension.attribute" strings to AttributeRef objects
fn resolve_attributes<'a>(
    model: &'a Model,
    group: &'a TableGroup,
    table: &'a GroupTable,
    attributes: Option<&Vec<String>>,
) -> Result<Vec<AttributeRef<'a>>, ResolveError> {
    let Some(attrs) = attributes else {
        return Ok(vec![]);
    };

    attrs
        .iter()
        .map(|attr_str| resolve_attribute(model, group, table, attr_str))
        .collect()
}

/// Resolve a single "dimension.attribute" string
fn resolve_attribute<'a>(
    model: &'a Model,
    group: &'a TableGroup,
    table: &'a GroupTable,
    attr_str: &str,
) -> Result<AttributeRef<'a>, ResolveError> {
    // Parse "dimension.attribute" format
    let parts: Vec<&str> = attr_str.split('.').collect();
    if parts.len() != 2 {
        return Err(ResolveError::InvalidAttributeFormat(attr_str.to_string()));
    }
    let (dim_name, attr_name) = (parts[0], parts[1]);

    // Get the dimension from the table group
    let group_dim = group
        .get_dimension(dim_name)
        .ok_or_else(|| ResolveError::DimensionNotFound(dim_name.to_string()))?;

    // Check that the table has this dimension and attribute
    let table_attrs = table
        .get_dimension_attributes(dim_name)
        .ok_or_else(|| ResolveError::DimensionNotFound(dim_name.to_string()))?;
    
    if !table_attrs.iter().any(|a| a == attr_name) {
        return Err(ResolveError::AttributeNotFound {
            dimension: dim_name.to_string(),
            attribute: attr_name.to_string(),
        });
    }

    // Check for degenerate dimension (inline attributes on TableGroupDimension)
    if group_dim.is_degenerate() {
        let attribute = group_dim
            .get_attribute(attr_name)
            .ok_or_else(|| ResolveError::AttributeNotFound {
                dimension: dim_name.to_string(),
                attribute: attr_name.to_string(),
            })?;

        return Ok(AttributeRef::Degenerate {
            group_dim,
            attribute,
        });
    }

    // Regular: lookup dimension from model
    let dimension = model
        .get_dimension(dim_name)
        .ok_or_else(|| ResolveError::DimensionNotFound(dim_name.to_string()))?;

    // Get the attribute from the dimension
    let attribute = dimension.get_attribute(attr_name).ok_or_else(|| {
        ResolveError::AttributeNotFound {
            dimension: dim_name.to_string(),
            attribute: attr_name.to_string(),
        }
    })?;

    Ok(AttributeRef::Joined {
        group_dim,
        dimension,
        attribute,
    })
}

/// Resolve metric names to Metric objects
fn resolve_metrics<'a>(
    model: &'a crate::model::Model,
    metrics: Option<&Vec<String>>,
) -> Result<Vec<&'a Metric>, ResolveError> {
    let Some(metric_names) = metrics else {
        return Ok(vec![]);
    };

    metric_names
        .iter()
        .map(|name| {
            model.get_metric(name)
                .ok_or_else(|| ResolveError::MetricNotFound(name.clone()))
        })
        .collect()
}

/// Collect all measures required by metrics
fn collect_metric_measures<'a>(
    group: &'a TableGroup,
    table: &'a GroupTable,
    metrics: &[&'a Metric],
) -> Result<Vec<&'a crate::model::Measure>, ResolveError> {
    let mut measure_names: HashSet<&str> = HashSet::new();
    
    // Extract measures required by metrics
    for metric in metrics {
        collect_metric_measure_deps(&metric.expr, &mut measure_names);
    }
    
    // Resolve all measure names to Measure objects (from the table group)
    measure_names
        .into_iter()
        .map(|name| {
            // Check that the table supports this measure
            if !table.has_measure(name) {
                return Err(ResolveError::MeasureNotFound(name.to_string()));
            }
            group.get_measure(name)
                .ok_or_else(|| ResolveError::MeasureNotFound(name.to_string()))
        })
        .collect()
}

/// Extract measure names referenced in a metric expression
fn collect_metric_measure_deps<'a>(expr: &'a MetricExpr, measures: &mut HashSet<&'a str>) {
    match expr {
        MetricExpr::MeasureRef(name) => {
            measures.insert(name);
        }
        MetricExpr::Structured(node) => {
            collect_node_measure_deps(node, measures);
        }
    }
}

/// Extract measure names from an expression node
fn collect_node_measure_deps<'a>(node: &'a MetricExprNode, measures: &mut HashSet<&'a str>) {
    match node {
        MetricExprNode::Measure(name) => {
            measures.insert(name);
        }
        MetricExprNode::Literal(_) => {}
        MetricExprNode::Add(args) 
        | MetricExprNode::Subtract(args) 
        | MetricExprNode::Multiply(args) 
        | MetricExprNode::Divide(args) => {
            for arg in args {
                collect_arg_measure_deps(arg, measures);
            }
        }
        MetricExprNode::Case(case_expr) => {
            // Collect measure dependencies from all CASE branches
            for when_branch in &case_expr.when {
                collect_arg_measure_deps(&when_branch.then, measures);
            }
            if let Some(else_val) = &case_expr.else_value {
                collect_arg_measure_deps(else_val, measures);
            }
        }
    }
}

/// Extract measure names from an expression argument
fn collect_arg_measure_deps<'a>(arg: &'a MetricExprArg, measures: &mut HashSet<&'a str>) {
    match arg {
        MetricExprArg::MeasureName(name) => {
            measures.insert(name);
        }
        MetricExprArg::LiteralNumber(_) => {
            // Literal numbers don't reference measures
        }
        MetricExprArg::Node(node) => {
            collect_node_measure_deps(node, measures);
        }
    }
}

/// Collect unique dimensions needed for this query
fn collect_dimensions<'a>(
    row_attrs: &[AttributeRef<'a>],
    col_attrs: &[AttributeRef<'a>],
    filters: &[ResolvedFilter<'a>],
) -> Vec<ResolvedDimension<'a>> {
    let mut seen: Vec<&str> = vec![];
    let mut dimensions = vec![];

    // Helper to convert AttributeRef to ResolvedDimension
    fn attr_to_resolved_dim<'a>(attr: &AttributeRef<'a>) -> ResolvedDimension<'a> {
        match attr {
            AttributeRef::Degenerate { group_dim, .. } => {
                ResolvedDimension::Degenerate { group_dim }
            }
            AttributeRef::Joined { group_dim, dimension, .. } => {
                ResolvedDimension::Joined { group_dim, dimension }
            }
        }
    }

    // Collect from row and column attributes
    for attr in row_attrs.iter().chain(col_attrs.iter()) {
        let dim_name = attr.dimension_name();
        if !seen.contains(&dim_name) {
            seen.push(dim_name);
            dimensions.push(attr_to_resolved_dim(attr));
        }
    }

    // Also collect from filter attributes
    for filter in filters {
        let dim_name = filter.attribute.dimension_name();
        if !seen.contains(&dim_name) {
            seen.push(dim_name);
            dimensions.push(attr_to_resolved_dim(&filter.attribute));
        }
    }

    dimensions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selector::select_tables;

    fn load_test_schema() -> Schema {
        Schema::from_file("test_data/steelwheels.yaml").unwrap()
    }
    
    /// Helper to get the first selected table from the model
    fn get_first_selected<'a>(schema: &'a Schema, model_name: &str) -> SelectedTable<'a> {
        let model = schema.get_model(model_name).unwrap();
        select_tables(schema, model, &[], &[]).unwrap().into_iter().next().unwrap()
    }

    #[test]
    fn test_resolve_simple_query() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["dates.year".to_string()]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        assert_eq!(resolved.model.name, "steelwheels");
        assert_eq!(resolved.row_attributes.len(), 1);
        assert_eq!(resolved.row_attributes[0].dimension_name(), "dates");
        assert_eq!(resolved.row_attributes[0].attribute().name, "year");
        // sales metric depends on sales measure
        assert_eq!(resolved.measures.len(), 1);
        assert_eq!(resolved.measures[0].name, "sales");
        assert_eq!(resolved.dimensions.len(), 1);
    }

    #[test]
    fn test_resolve_multi_dimension_query() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["dates.year".to_string(), "dates.quarter".to_string()]),
            columns: Some(vec!["markets.country".to_string()]),
            metrics: Some(vec!["sales".to_string(), "quantity".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        assert_eq!(resolved.row_attributes.len(), 2);
        assert_eq!(resolved.column_attributes.len(), 1);
        assert_eq!(resolved.measures.len(), 2);
        // dates and markets = 2 unique dimensions
        assert_eq!(resolved.dimensions.len(), 2);
    }

    #[test]
    fn test_resolve_model_not_found() {
        let schema = load_test_schema();
        // Use a valid table for the call, but the model name in request is invalid
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "nonexistent".to_string(),
            dimensions: None,
            rows: None,
            columns: None,
            metrics: None,
            filter: None,
        };

        let err = resolve_query(&schema, &request, &selected).unwrap_err();
        assert!(matches!(err, ResolveError::ModelNotFound(_)));
    }

    #[test]
    fn test_resolve_invalid_attribute_format() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["invalid_format".to_string()]), // Missing dot
            columns: None,
            metrics: None,
            filter: None,
        };

        let err = resolve_query(&schema, &request, &selected).unwrap_err();
        assert!(matches!(err, ResolveError::InvalidAttributeFormat(_)));
    }

    #[test]
    fn test_resolve_attribute_not_found() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["dates.nonexistent".to_string()]),
            columns: None,
            metrics: None,
            filter: None,
        };

        let err = resolve_query(&schema, &request, &selected).unwrap_err();
        assert!(matches!(err, ResolveError::AttributeNotFound { .. }));
    }

    #[test]
    fn test_resolve_metric() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["dates.year".to_string()]),
            columns: None,
            metrics: Some(vec!["avg_unit_price".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        // Metric should be resolved
        assert_eq!(resolved.metrics.len(), 1);
        assert_eq!(resolved.metrics[0].name, "avg_unit_price");
        
        // Dependent measures (sales, quantity) should be auto-included
        assert_eq!(resolved.measures.len(), 2);
        let measure_names: Vec<&str> = resolved.measures.iter().map(|m| m.name.as_str()).collect();
        assert!(measure_names.contains(&"sales"));
        assert!(measure_names.contains(&"quantity"));
        
        // Output should only include the metric, not the underlying measures
        let output_names = resolved.output_names();
        assert!(output_names.contains(&"dates.year".to_string()));
        assert!(output_names.contains(&"avg_unit_price".to_string()));
        assert!(!output_names.contains(&"sales".to_string()));
        assert!(!output_names.contains(&"quantity".to_string()));
    }

    #[test]
    fn test_resolve_multiple_metrics() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: None,
            columns: None,
            metrics: Some(vec!["sales".to_string(), "avg_unit_price".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        // Output should include both metrics
        let output_names = resolved.output_names();
        assert!(output_names.contains(&"sales".to_string()));
        assert!(output_names.contains(&"avg_unit_price".to_string()));
        // quantity is a dependency of avg_unit_price, not directly requested
        assert!(!output_names.contains(&"quantity".to_string()));
    }

    #[test]
    fn test_resolve_metric_not_found() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: None,
            columns: None,
            metrics: Some(vec!["nonexistent_metric".to_string()]),
            filter: None,
        };

        let err = resolve_query(&schema, &request, &selected).unwrap_err();
        assert!(matches!(err, ResolveError::MetricNotFound(_)));
    }

    #[test]
    fn test_resolve_degenerate_dimension() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["flags.is_premium".to_string()]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        // Should have one row attribute from degenerate dimension
        assert_eq!(resolved.row_attributes.len(), 1);
        assert!(resolved.row_attributes[0].is_degenerate());
        assert_eq!(resolved.row_attributes[0].dimension_name(), "flags");
        assert_eq!(resolved.row_attributes[0].attribute().name, "is_premium");
        
        // The degenerate dimension should be in the dimensions list
        assert_eq!(resolved.dimensions.len(), 1);
        assert!(resolved.dimensions[0].is_degenerate());
    }

    #[test]
    fn test_resolve_mixed_dimensions() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "dates.year".to_string(),
                "flags.status".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        // Should have two row attributes
        assert_eq!(resolved.row_attributes.len(), 2);
        
        // First is joined dimension
        assert!(!resolved.row_attributes[0].is_degenerate());
        assert_eq!(resolved.row_attributes[0].dimension_name(), "dates");
        
        // Second is degenerate dimension
        assert!(resolved.row_attributes[1].is_degenerate());
        assert_eq!(resolved.row_attributes[1].dimension_name(), "flags");
        
        // Should have 2 dimensions (dates and flags)
        assert_eq!(resolved.dimensions.len(), 2);
    }
}
