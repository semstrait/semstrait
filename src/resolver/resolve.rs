use std::collections::HashSet;
use crate::semantic_model::{Schema, SemanticModel, DatasetGroup, GroupDataset, Metric, MetricExpr, MetricExprNode, MetricExprArg};
use crate::query::{DataFilter, QueryRequest};
use crate::selector::SelectedDataset;
use super::error::ResolveError;
use super::types::{AttributeRef, ResolvedDimension, ResolvedFilter, ResolvedQuery};

/// Resolve an analytics query request against a schema and selected dataset
/// 
/// This turns string references (model name, metric names, "dimension.attribute" strings)
/// into actual schema object references.
/// 
/// # Arguments
/// * `schema` - The schema containing dimension definitions
/// * `request` - The query request with model name, metrics, dimensions, etc.
/// * `selected` - The selected dataset (from selector) containing both group and dataset
pub fn resolve_query<'a>(
    schema: &'a Schema,
    request: &QueryRequest,
    selected: &SelectedDataset<'a>,
) -> Result<ResolvedQuery<'a>, ResolveError> {
    let group = selected.group;
    let dataset = selected.dataset;
    
    // 1. Resolve the model (for metrics which are model-level)
    let model = schema
        .get_model(&request.model)
        .ok_or_else(|| ResolveError::ModelNotFound(request.model.clone()))?;

    // 2. Resolve row attributes (e.g., ["dates.year", "markets.country"])
    let row_attributes = resolve_attributes(model, group, dataset, request.rows.as_ref())?;

    // 3. Resolve column attributes
    let column_attributes = resolve_attributes(model, group, dataset, request.columns.as_ref())?;

    // 4. Resolve metrics (from model - metrics are model-level)
    let metrics = resolve_metrics(model, request.metrics.as_ref())?;

    // 5. Collect all measures needed by metrics
    let measures = collect_metric_measures(group, dataset, &metrics)?;

    // 6. Resolve filters
    let filters = resolve_filters(model, group, dataset, request.filter.as_ref())?;

    // 7. Collect all unique dimensions needed for this query (including filter dimensions)
    let dimensions = collect_dimensions(&row_attributes, &column_attributes, &filters);

    Ok(ResolvedQuery {
        model,
        dataset_group: group,
        dataset,
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
    model: &'a SemanticModel,
    group: &'a DatasetGroup,
    dataset: &'a GroupDataset,
    filters: Option<&Vec<DataFilter>>,
) -> Result<Vec<ResolvedFilter<'a>>, ResolveError> {
    let Some(filter_list) = filters else {
        return Ok(vec![]);
    };

    filter_list
        .iter()
        .map(|f| {
            let attribute = resolve_attribute(model, group, dataset, &f.field)?;
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
    model: &'a SemanticModel,
    group: &'a DatasetGroup,
    dataset: &'a GroupDataset,
    attributes: Option<&Vec<String>>,
) -> Result<Vec<AttributeRef<'a>>, ResolveError> {
    let Some(attrs) = attributes else {
        return Ok(vec![]);
    };

    attrs
        .iter()
        .map(|attr_str| resolve_attribute(model, group, dataset, attr_str))
        .collect()
}

/// Resolve a single attribute string
/// 
/// Supports two formats:
/// - "dimension.attribute" - resolved against the selected datasetGroup
/// - "datasetGroup.dimension.attribute" - resolved against a specific datasetGroup
fn resolve_attribute<'a>(
    model: &'a SemanticModel,
    group: &'a DatasetGroup,
    dataset: &'a GroupDataset,
    attr_str: &str,
) -> Result<AttributeRef<'a>, ResolveError> {
    // Parse the attribute string
    let parts: Vec<&str> = attr_str.split('.').collect();
    
    match parts.len() {
        2 => {
            // Standard format: "dimension.attribute"
            let (dim_name, attr_name) = (parts[0], parts[1]);
            resolve_attribute_in_group(model, group, dataset, dim_name, attr_name, None)
        }
        3 => {
            // DatasetGroup-qualified format: "datasetGroup.dimension.attribute"
            let (tg_name, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            resolve_dataset_group_qualified_attribute(model, tg_name, dim_name, attr_name)
        }
        _ => Err(ResolveError::InvalidAttributeFormat(attr_str.to_string())),
    }
}

/// Resolve a datasetGroup-qualified attribute (e.g., "adwords.campaign.name")
fn resolve_dataset_group_qualified_attribute<'a>(
    model: &'a SemanticModel,
    tg_name: &str,
    dim_name: &str,
    attr_name: &str,
) -> Result<AttributeRef<'a>, ResolveError> {
    // Get the specified datasetGroup
    let target_group = model
        .get_dataset_group(tg_name)
        .ok_or_else(|| ResolveError::DatasetGroupNotFound(tg_name.to_string()))?;
    
    // Find a dataset in this datasetGroup (use the first one for validation)
    let target_dataset = target_group.datasets.first()
        .ok_or_else(|| ResolveError::InvalidQuery(
            format!("DatasetGroup '{}' has no datasets", tg_name)
        ))?;
    
    // Resolve the attribute within that datasetGroup
    resolve_attribute_in_group(
        model, 
        target_group, 
        target_dataset, 
        dim_name, 
        attr_name, 
        Some(tg_name.to_string())
    )
}

/// Resolve an attribute within a specific datasetGroup
fn resolve_attribute_in_group<'a>(
    model: &'a SemanticModel,
    group: &'a DatasetGroup,
    dataset: &'a GroupDataset,
    dim_name: &str,
    attr_name: &str,
    dataset_group_qualifier: Option<String>,
) -> Result<AttributeRef<'a>, ResolveError> {
    // Handle virtual _table dimension for metadata attributes
    if dim_name == "_table" {
        return resolve_meta_attribute(model, group, dataset, attr_name);
    }

    // Get the dimension from the dataset group
    let group_dim = group
        .get_dimension(dim_name)
        .ok_or_else(|| ResolveError::DimensionNotFound(dim_name.to_string()))?;

    // Check that the dataset has this dimension and attribute
    let dataset_attrs = dataset
        .get_dimension_attributes(dim_name)
        .ok_or_else(|| ResolveError::DimensionNotFound(dim_name.to_string()))?;
    
    if !dataset_attrs.iter().any(|a| a == attr_name) {
        return Err(ResolveError::AttributeNotFound {
            dimension: dim_name.to_string(),
            attribute: attr_name.to_string(),
        });
    }

    // Check for degenerate dimension (inline attributes on DatasetGroupDimension)
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
            dataset_group_qualifier,
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
        dataset_group_qualifier,
    })
}

/// Resolve a virtual _table metadata attribute
/// 
/// The attribute must be:
/// 1. Declared in the model's `_table` dimension (if explicit _table dimension exists)
/// 2. Declared as available on the dataset (in dataset.dimensions._table)
/// 
/// Supported built-in attributes:
/// - `_table.model` - Model name
/// - `_table.namespace` - Model namespace  
/// - `_table.datasetGroup` - DatasetGroup name (also supports legacy `_table.tableGroup`)
/// - `_table.dataset` - Physical dataset name (also supports legacy `_table.table`)
/// - `_table.uuid` - Dataset UUID (e.g., from Iceberg catalog)
/// - `_table.{key}` - Any key from dataset.properties
fn resolve_meta_attribute<'a>(
    model: &'a SemanticModel,
    group: &'a DatasetGroup,
    dataset: &'a GroupDataset,
    attr_name: &str,
) -> Result<AttributeRef<'a>, ResolveError> {
    // 1. Check if _table dimension is declared in the model
    let table_dim = model.get_dimension("_table");
    
    // 2. If declared, validate the attribute exists in the dimension definition
    if let Some(dim) = table_dim {
        if dim.get_attribute(attr_name).is_none() {
            return Err(ResolveError::AttributeNotFound {
                dimension: "_table".to_string(),
                attribute: attr_name.to_string(),
            });
        }
    }
    
    // 3. Check if this dataset declares this _table attribute as available
    if let Some(dataset_attrs) = dataset.get_dimension_attributes("_table") {
        if !dataset_attrs.iter().any(|a| a == attr_name) {
            return Err(ResolveError::AttributeNotFound {
                dimension: "_table".to_string(),
                attribute: attr_name.to_string(),
            });
        }
    } else if table_dim.is_some() {
        // _table dimension declared but dataset doesn't list any _table attributes
        return Err(ResolveError::DimensionNotFound("_table".to_string()));
    }
    
    // 4. Resolve the actual value
    let value = resolve_meta_value(model, group, dataset, attr_name)?;

    Ok(AttributeRef::Meta {
        name: attr_name.to_string(),
        value,
    })
}

/// Get the actual value for a _table metadata attribute
fn resolve_meta_value(
    model: &SemanticModel,
    group: &DatasetGroup,
    dataset: &GroupDataset,
    attr_name: &str,
) -> Result<String, ResolveError> {
    match attr_name {
        "model" => Ok(model.name.clone()),
        "namespace" => model.namespace.clone().ok_or_else(|| {
            ResolveError::MetaAttributeNotSet {
                attribute: "namespace".to_string(),
                reason: "model does not have a namespace defined".to_string(),
            }
        }),
        // Support both new and legacy names
        "datasetGroup" | "tableGroup" => Ok(group.name.clone()),
        "dataset" | "table" => Ok(dataset.dataset.clone()),
        "uuid" => dataset.uuid.clone().ok_or_else(|| {
            ResolveError::MetaAttributeNotSet {
                attribute: "uuid".to_string(),
                reason: "dataset does not have a uuid defined".to_string(),
            }
        }),
        key => {
            // Look in dataset.properties
            dataset.properties
                .as_ref()
                .and_then(|props| props.get(key))
                .cloned()
                .ok_or_else(|| ResolveError::MetaAttributeNotSet {
                    attribute: key.to_string(),
                    reason: format!("dataset does not have property '{}' defined", key),
                })
        }
    }
}

/// Resolve metric names to Metric objects
fn resolve_metrics<'a>(
    model: &'a crate::semantic_model::SemanticModel,
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
    group: &'a DatasetGroup,
    dataset: &'a GroupDataset,
    metrics: &[&'a Metric],
) -> Result<Vec<&'a crate::semantic_model::Measure>, ResolveError> {
    let mut measure_names: HashSet<&str> = HashSet::new();
    
    // Extract measures required by metrics
    for metric in metrics {
        collect_metric_measure_deps(&metric.expr, &mut measure_names);
    }
    
    // Resolve all measure names to Measure objects (from the dataset group)
    measure_names
        .into_iter()
        .map(|name| {
            // Check that the dataset supports this measure
            if !dataset.has_measure(name) {
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
            AttributeRef::Meta { .. } => {
                ResolvedDimension::Meta
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
    use crate::selector::select_datasets;

    fn load_test_schema() -> Schema {
        Schema::from_file("test_data/steelwheels.yaml").unwrap()
    }
    
    /// Helper to get the first selected dataset from the model
    fn get_first_selected<'a>(schema: &'a Schema, model_name: &str) -> SelectedDataset<'a> {
        let model = schema.get_model(model_name).unwrap();
        select_datasets(schema, model, &[], &[]).unwrap().into_iter().next().unwrap()
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
        // Use a valid dataset for the call, but the model name in request is invalid
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

    #[test]
    fn test_resolve_meta_dataset_group() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "dates.year".to_string(),
                "_table.tableGroup".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        // Should have two row attributes
        assert_eq!(resolved.row_attributes.len(), 2);
        
        // First is regular dimension
        assert_eq!(resolved.row_attributes[0].dimension_name(), "dates");
        assert!(!resolved.row_attributes[0].is_meta());
        
        // Second is meta attribute
        assert_eq!(resolved.row_attributes[1].dimension_name(), "_table");
        assert!(resolved.row_attributes[1].is_meta());
        assert_eq!(resolved.row_attributes[1].attribute_name(), "tableGroup");
        assert_eq!(resolved.row_attributes[1].meta_value(), Some("orders"));
        
        // Output names should include both
        let output_names = resolved.output_names();
        assert!(output_names.contains(&"dates.year".to_string()));
        assert!(output_names.contains(&"_table.tableGroup".to_string()));
    }

    #[test]
    fn test_resolve_meta_model_and_uuid() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "_table.model".to_string(),
                "_table.namespace".to_string(),
                "_table.uuid".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        assert_eq!(resolved.row_attributes.len(), 3);
        
        // Model name
        assert!(resolved.row_attributes[0].is_meta());
        assert_eq!(resolved.row_attributes[0].meta_value(), Some("steelwheels"));
        
        // Namespace
        assert!(resolved.row_attributes[1].is_meta());
        assert_eq!(resolved.row_attributes[1].meta_value(), Some("a908ff91-c951-4d65-b054-d246d2e8cae1"));
        
        // UUID
        assert!(resolved.row_attributes[2].is_meta());
        assert_eq!(resolved.row_attributes[2].meta_value(), Some("a1b2c3d4-e5f6-7890-abcd-ef1234567890"));
    }

    #[test]
    fn test_resolve_meta_properties() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "_table.sourceSystem".to_string(),
                "_table.dataOwner".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        assert_eq!(resolved.row_attributes.len(), 2);
        
        // sourceSystem property
        assert!(resolved.row_attributes[0].is_meta());
        assert_eq!(resolved.row_attributes[0].attribute_name(), "sourceSystem");
        assert_eq!(resolved.row_attributes[0].meta_value(), Some("pentaho"));
        
        // dataOwner property
        assert!(resolved.row_attributes[1].is_meta());
        assert_eq!(resolved.row_attributes[1].attribute_name(), "dataOwner");
        assert_eq!(resolved.row_attributes[1].meta_value(), Some("analytics-team"));
    }

    #[test]
    fn test_resolve_meta_not_found() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["_table.nonexistent".to_string()]),
            columns: None,
            metrics: None,
            filter: None,
        };

        let err = resolve_query(&schema, &request, &selected).unwrap_err();
        // With explicit _table dimension, unknown attributes return AttributeNotFound
        assert!(matches!(err, ResolveError::AttributeNotFound { dimension, .. } if dimension == "_table"));
    }

    #[test]
    fn test_resolve_meta_dimension_in_output() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "dates.year".to_string(),
                "_table.tableGroup".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();

        // Should have 2 dimensions: dates and _table (Meta)
        assert_eq!(resolved.dimensions.len(), 2);
        
        // First dimension is regular
        assert!(!resolved.dimensions[0].is_meta());
        
        // Second dimension is Meta
        assert!(resolved.dimensions[1].is_meta());
        assert_eq!(resolved.dimensions[1].name(), "_table");
    }
}
