//! Dataset group types for aggregate awareness
//!
//! A DatasetGroup defines a set of datasets that share dimension and measure definitions.
//! Datasets within a group declare which subset of fields they have, enabling
//! automatic dataset selection (aggregate awareness).

use serde::Deserialize;
use std::collections::HashMap;
use super::column::Column;
use super::dimension::{Attribute, Join};
use super::measure::Measure;

/// Data source configuration for a dataset
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Source {
    /// Parquet file source
    #[serde(rename = "parquet")]
    Parquet { path: String },
    /// Iceberg table source (catalog resolution is the service layer's responsibility)
    #[serde(rename = "iceberg")]
    Iceberg { table: String },
}

/// Resolve template variables in a path string
/// 
/// Supports the following variables:
/// - `{model.name}` - Model name
/// - `{model.namespace}` - Model namespace (errors if not set)
/// - `{datasetGroup.name}` - Dataset group name
/// - `{dataset.name}` - Physical dataset name
/// - `{dataset.uuid}` - Dataset UUID (errors if not set)
/// 
/// # Example
/// ```ignore
/// let path = resolve_path_template(
///     "{model.namespace}/{dataset.uuid}/data.parquet",
///     "sales",
///     Some("tenant-123"),
///     "orders",
///     "warehouse.orderfact",
///     Some("abc-123"),
/// )?;
/// // Returns: "tenant-123/abc-123/data.parquet"
/// ```
pub fn resolve_path_template(
    template: &str,
    model_name: &str,
    model_namespace: Option<&str>,
    dataset_group_name: &str,
    dataset_name: &str,
    dataset_uuid: Option<&str>,
) -> Result<String, String> {
    let mut path = template.to_string();
    
    // Required variables (always available)
    path = path.replace("{model.name}", model_name);
    path = path.replace("{datasetGroup.name}", dataset_group_name);
    path = path.replace("{dataset.name}", dataset_name);
    
    
    // Optional variables - error if used but not present
    if path.contains("{model.namespace}") {
        match model_namespace {
            Some(ns) => path = path.replace("{model.namespace}", ns),
            None => return Err(format!(
                "Path template uses {{model.namespace}} but model '{}' has no namespace defined",
                model_name
            )),
        }
    }
    
    if path.contains("{dataset.uuid}") {
        match dataset_uuid {
            Some(uuid) => path = path.replace("{dataset.uuid}", uuid),
            None => return Err(format!(
                "Path template uses {{dataset.uuid}} but dataset '{}' has no uuid defined",
                dataset_name
            )),
        }
    }
    
    // Check for unresolved variables
    if let Some(start) = path.find('{') {
        if let Some(end) = path[start..].find('}') {
            let var = &path[start..start + end + 1];
            return Err(format!("Unknown variable in path template: {}", var));
        }
    }
    
    Ok(path)
}

/// Resolve template variables in a dimension path string
/// 
/// Supports the following variables:
/// - `{model.name}` - Model name
/// - `{model.namespace}` - Model namespace (errors if not set)
/// - `{dimension.name}` - Dimension name
/// - `{dimension.table}` - Dimension table name
/// 
/// # Example
/// ```ignore
/// let path = resolve_dimension_path_template(
///     "{model.namespace}/dimensions/{dimension.name}.parquet",
///     "sales",
///     Some("tenant-123"),
///     "dates",
///     "warehouse.dates",
/// )?;
/// // Returns: "tenant-123/dimensions/dates.parquet"
/// ```
pub fn resolve_dimension_path_template(
    template: &str,
    model_name: &str,
    model_namespace: Option<&str>,
    dimension_name: &str,
    dimension_table: &str,
) -> Result<String, String> {
    let mut path = template.to_string();
    
    // Required variables (always available)
    path = path.replace("{model.name}", model_name);
    path = path.replace("{dimension.name}", dimension_name);
    path = path.replace("{dimension.table}", dimension_table);
    
    // Optional variables - error if used but not present
    if path.contains("{model.namespace}") {
        match model_namespace {
            Some(ns) => path = path.replace("{model.namespace}", ns),
            None => return Err(format!(
                "Path template uses {{model.namespace}} but model '{}' has no namespace defined",
                model_name
            )),
        }
    }
    
    // Check for unresolved variables
    if let Some(start) = path.find('{') {
        if let Some(end) = path[start..].find('}') {
            let var = &path[start..start + end + 1];
            return Err(format!("Unknown variable in dimension path template: {}", var));
        }
    }
    
    Ok(path)
}

/// A dataset group - datasets sharing dimension and measure definitions
#[derive(Debug, Deserialize)]
pub struct DatasetGroup {
    pub name: String,
    pub label: Option<String>,
    /// Human-readable description for UIs and LLMs
    pub description: Option<String>,
    /// Dimensions available to datasets in this group
    pub dimensions: Vec<DatasetGroupDimension>,
    /// Measures shared by all datasets in this group
    pub measures: Vec<Measure>,
    /// Physical datasets, each declaring which subset of fields it has
    pub datasets: Vec<Dataset>,
}

/// A dimension reference within a dataset group
/// 
/// Can be either:
/// - A reference to a top-level dimension (has join)
/// - A degenerate dimension (no join, has inline attributes)
#[derive(Debug, Deserialize, Clone)]
pub struct DatasetGroupDimension {
    pub name: String,
    pub label: Option<String>,
    /// Join specification - if None, this is a degenerate dimension
    pub join: Option<Join>,
    /// Inline attributes for degenerate dimensions
    pub attributes: Option<Vec<Attribute>>,
}

/// A physical dataset within a dataset group
#[derive(Debug, Deserialize)]
pub struct Dataset {
    /// Physical dataset name (e.g., "warehouse.orderfact")
    pub name: String,
    pub label: Option<String>,
    /// Human-readable description for UIs and LLMs
    pub description: Option<String>,
    /// Data source configuration (parquet path, iceberg table, etc.)
    pub source: Source,
    /// Unique identifier for this dataset (e.g., Iceberg table UUID)
    pub uuid: Option<String>,
    /// Custom key-value properties (e.g., connectorType, sourceSystem)
    pub properties: Option<HashMap<String, String>>,
    /// Column definitions - optional, used for explicit schema documentation
    /// Join detection is now based on dimension attribute inclusion, not column presence
    #[serde(default)]
    pub columns: Option<Vec<Column>>,
    /// Dimension attributes available on this dataset
    /// Map from dimension name to list of attribute names
    pub dimensions: HashMap<String, Vec<String>>,
    /// Measure names available on this dataset (references group-level measures)
    pub measures: Vec<String>,
    /// Row filter for partitioned datasets
    /// e.g., { "dates.year": 2023 } means this dataset only contains 2023 data
    pub row_filter: Option<HashMap<String, serde_yaml::Value>>,
}

impl DatasetGroup {
    /// Get a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<&DatasetGroupDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Get a measure by name
    pub fn get_measure(&self, name: &str) -> Option<&Measure> {
        self.measures.iter().find(|m| m.name == name)
    }

    /// Get a dataset by physical dataset name
    pub fn get_dataset(&self, dataset_name: &str) -> Option<&Dataset> {
        self.datasets.iter().find(|t| t.name == dataset_name)
    }

    /// Get all unique measure names
    pub fn measure_names(&self) -> Vec<&str> {
        self.measures.iter().map(|m| m.name.as_str()).collect()
    }
}

impl DatasetGroupDimension {
    /// Returns true if this is a degenerate dimension (no join, has inline attributes)
    pub fn is_degenerate(&self) -> bool {
        self.join.is_none()
    }

    /// Returns true if this references a top-level dimension (has join)
    pub fn is_reference(&self) -> bool {
        self.join.is_some()
    }

    /// Get an inline attribute by name (for degenerate dimensions)
    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.as_ref()?.iter().find(|a| a.name == name)
    }

    /// Get the join key (left key) if this is a joined dimension
    pub fn join_key(&self) -> Option<&str> {
        self.join.as_ref().map(|j| j.left_key.as_str())
    }
}

impl Dataset {
    /// Get the parquet path if source is Parquet
    pub fn parquet_path(&self) -> Option<&str> {
        match &self.source {
            Source::Parquet { path } => Some(path),
            _ => None,
        }
    }

    /// Get the Iceberg table identifier if source is Iceberg
    pub fn iceberg_table(&self) -> Option<&str> {
        match &self.source {
            Source::Iceberg { table } => Some(table),
            _ => None,
        }
    }

    /// Get the primary source identifier regardless of source type.
    ///
    /// Returns the parquet path for Parquet sources, or the table identifier
    /// for Iceberg sources.
    pub fn source_ref(&self) -> &str {
        match &self.source {
            Source::Parquet { path } => path,
            Source::Iceberg { table } => table,
        }
    }

    /// Get a column definition by name (from optional columns list)
    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.as_ref()?.iter().find(|c| c.name == name)
    }

    /// Check if this dataset has a specific column in the explicit columns list
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.as_ref()
            .map(|cols| cols.iter().any(|c| c.name == name))
            .unwrap_or(false)
    }

    /// Get the list of attributes for a dimension
    pub fn get_dimension_attributes(&self, dim_name: &str) -> Option<&Vec<String>> {
        self.dimensions.get(dim_name)
    }

    /// Check if this dataset has a dimension
    pub fn has_dimension(&self, name: &str) -> bool {
        self.dimensions.contains_key(name)
    }

    /// Check if this dataset has a measure (by name)
    pub fn has_measure(&self, name: &str) -> bool {
        self.measures.iter().any(|m| m == name)
    }

    /// Count total available attributes across all dimensions
    pub fn attribute_count(&self) -> usize {
        self.dimensions.values().map(|attrs| attrs.len()).sum()
    }

    /// Check if a dimension needs a join on this dataset (legacy method)
    /// 
    /// If the join key column exists on this dataset, a join is needed.
    /// If the join key is absent, assume attributes are denormalized.
    #[deprecated(note = "Use needs_join_for_dimension instead, which uses attribute-based detection")]
    pub fn needs_join(&self, dim: &DatasetGroupDimension) -> bool {
        match dim.join_key() {
            Some(key) => self.has_column(key),
            None => false, // Degenerate dimensions never need joins
        }
    }

    /// Check if this dataset has a specific attribute for a dimension
    pub fn has_dimension_attribute(&self, dim_name: &str, attr_name: &str) -> bool {
        self.dimensions
            .get(dim_name)
            .map(|attrs| attrs.iter().any(|a| a == attr_name))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataset_group() -> DatasetGroup {
        let yaml = r#"
name: orders
dimensions:
  - name: dates
    join:
      left_key: time_id
      right_key: time_id
  - name: flags
    attributes:
      - name: is_premium
        column: is_premium_order
        type: bool
measures:
  - name: sales
    aggregation: sum
    expr: totalprice
    type: f64
datasets:
  - name: warehouse.orderfact
    source:
      type: parquet
      path: /data/warehouse/orderfact.parquet
    dimensions:
      dates: [year, month]
      flags: [is_premium]
    measures: [sales]
"#;
        serde_yaml::from_str(yaml).unwrap()
    }

    fn sample_dataset_group_with_columns() -> DatasetGroup {
        let yaml = r#"
name: orders
dimensions:
  - name: dates
    join:
      left_key: time_id
      right_key: time_id
  - name: flags
    attributes:
      - name: is_premium
        column: is_premium_order
        type: bool
measures:
  - name: sales
    aggregation: sum
    expr: totalprice
    type: f64
datasets:
  - name: warehouse.orderfact
    source:
      type: parquet
      path: /data/warehouse/orderfact.parquet
    columns:
      - name: time_id
        type: i32
      - name: totalprice
        type: f64
      - name: is_premium_order
        type: bool
    dimensions:
      dates: [year, month]
      flags: [is_premium]
    measures: [sales]
"#;
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn test_parse_dataset_group() {
        let group = sample_dataset_group();
        assert_eq!(group.name, "orders");
        assert_eq!(group.dimensions.len(), 2);
        assert_eq!(group.measures.len(), 1);
        assert_eq!(group.datasets.len(), 1);
    }

    #[test]
    fn test_parse_dataset_group_without_columns() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        // columns should be None when not specified
        assert!(dataset.columns.is_none());
    }

    #[test]
    fn test_parse_dataset_group_with_columns() {
        let group = sample_dataset_group_with_columns();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        // columns should be Some when specified
        assert!(dataset.columns.is_some());
        assert_eq!(dataset.columns.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_dimension_types() {
        let group = sample_dataset_group();
        
        let dates = group.get_dimension("dates").unwrap();
        assert!(dates.is_reference());
        assert!(!dates.is_degenerate());
        assert_eq!(dates.join_key(), Some("time_id"));
        
        let flags = group.get_dimension("flags").unwrap();
        assert!(!flags.is_reference());
        assert!(flags.is_degenerate());
        assert_eq!(flags.join_key(), None);
    }

    #[test]
    fn test_dataset_columns_optional() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        // Without explicit columns, has_column returns false
        assert!(!dataset.has_column("time_id"));
        assert!(!dataset.has_column("totalprice"));
        assert!(!dataset.has_column("nonexistent"));
    }

    #[test]
    fn test_dataset_columns_explicit() {
        let group = sample_dataset_group_with_columns();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        // With explicit columns, has_column works
        assert!(dataset.has_column("time_id"));
        assert!(dataset.has_column("totalprice"));
        assert!(!dataset.has_column("nonexistent"));
    }

    #[test]
    fn test_dataset_dimensions() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        assert!(dataset.has_dimension("dates"));
        assert!(dataset.has_dimension("flags"));
        assert!(!dataset.has_dimension("markets"));
        
        let attrs = dataset.get_dimension_attributes("dates").unwrap();
        assert_eq!(attrs, &vec!["year".to_string(), "month".to_string()]);
    }

    #[test]
    fn test_has_dimension_attribute() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        assert!(dataset.has_dimension_attribute("dates", "year"));
        assert!(dataset.has_dimension_attribute("dates", "month"));
        assert!(!dataset.has_dimension_attribute("dates", "quarter"));
        assert!(dataset.has_dimension_attribute("flags", "is_premium"));
        assert!(!dataset.has_dimension_attribute("nonexistent", "attr"));
    }

    #[test]
    fn test_dataset_measures() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        assert!(dataset.has_measure("sales"));
        assert!(!dataset.has_measure("quantity"));
    }

    #[test]
    fn test_attribute_count() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();
        
        // dates: [year, month] = 2, flags: [is_premium] = 1
        assert_eq!(dataset.attribute_count(), 3);
    }

    #[test]
    fn test_resolve_path_template_all_variables() {
        let result = resolve_path_template(
            "{model.namespace}/{dataset.uuid}/data.parquet",
            "sales",
            Some("tenant-123"),
            "orders",
            "warehouse.orderfact",
            Some("abc-def-123"),
        );
        assert_eq!(result.unwrap(), "tenant-123/abc-def-123/data.parquet");
    }

    #[test]
    fn test_resolve_path_template_required_only() {
        let result = resolve_path_template(
            "/data/{model.name}/{datasetGroup.name}/{dataset.name}.parquet",
            "sales",
            None,
            "orders",
            "warehouse.orderfact",
            None,
        );
        assert_eq!(result.unwrap(), "/data/sales/orders/warehouse.orderfact.parquet");
    }

    #[test]
    fn test_resolve_path_template_no_variables() {
        let result = resolve_path_template(
            "/static/path/data.parquet",
            "sales",
            Some("tenant"),
            "orders",
            "dataset",
            Some("uuid"),
        );
        assert_eq!(result.unwrap(), "/static/path/data.parquet");
    }

    #[test]
    fn test_resolve_path_template_missing_namespace() {
        let result = resolve_path_template(
            "{model.namespace}/data.parquet",
            "sales",
            None,  // namespace not set
            "orders",
            "dataset",
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("model.namespace"));
    }

    #[test]
    fn test_resolve_path_template_missing_uuid() {
        let result = resolve_path_template(
            "{dataset.uuid}/data.parquet",
            "sales",
            None,
            "orders",
            "dataset",
            None,  // uuid not set
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dataset.uuid"));
    }

    #[test]
    fn test_resolve_path_template_unknown_variable() {
        let result = resolve_path_template(
            "{unknown.var}/data.parquet",
            "sales",
            None,
            "orders",
            "dataset",
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown variable"));
    }

    #[test]
    fn test_parse_iceberg_source() {
        let yaml = r#"
name: orders
dimensions:
  - name: dates
    join:
      left_key: time_id
      right_key: time_id
measures:
  - name: sales
    aggregation: sum
    expr: totalprice
    type: f64
datasets:
  - name: warehouse.orderfact
    source:
      type: iceberg
      table: warehouse.orderfact
    dimensions:
      dates: [year, month]
    measures: [sales]
"#;
        let group: DatasetGroup = serde_yaml::from_str(yaml).unwrap();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();

        assert!(dataset.parquet_path().is_none());
        assert_eq!(dataset.iceberg_table(), Some("warehouse.orderfact"));
        assert_eq!(dataset.source_ref(), "warehouse.orderfact");
    }

    #[test]
    fn test_source_ref_parquet() {
        let group = sample_dataset_group();
        let dataset = group.get_dataset("warehouse.orderfact").unwrap();

        assert_eq!(dataset.source_ref(), "/data/warehouse/orderfact.parquet");
        assert_eq!(dataset.parquet_path(), Some("/data/warehouse/orderfact.parquet"));
        assert!(dataset.iceberg_table().is_none());
    }

}
