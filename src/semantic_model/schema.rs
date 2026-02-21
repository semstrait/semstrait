//! Root schema definition

use serde::Deserialize;
use std::path::Path;
use super::dimension::Dimension;
use super::measure::Measure;
use super::metric::Metric;
use super::datasetgroup::{DatasetGroup, GroupDataset};
use crate::error::ParseError;

/// The root semantic schema containing semantic models
#[derive(Debug, Deserialize)]
pub struct Schema {
    pub semantic_models: Vec<SemanticModel>,
}

/// A semantic model - the queryable business entity
/// 
/// Contains one or more dataset groups that share dimension and measure definitions.
/// The selector picks the optimal dataset based on query requirements.
#[derive(Debug, Deserialize)]
pub struct SemanticModel {
    pub name: String,
    /// Namespace for the model (e.g., organization or project identifier)
    pub namespace: Option<String>,
    /// Model-level dimensions - queryable with 2-part paths across all dataset groups
    #[serde(default)]
    pub dimensions: Vec<Dimension>,
    /// Dataset groups - each group contains datasets that share field definitions
    pub dataset_groups: Vec<DatasetGroup>,
    /// Metrics - derived calculations from measures (model-level, shared across dataset groups)
    pub metrics: Option<Vec<Metric>>,
    /// Row-level security filter
    #[serde(rename = "dataFilter")]
    pub data_filter: Option<Vec<DataFilter>>,
}

/// Row-level security filter
#[derive(Debug, Deserialize)]
pub struct DataFilter {
    pub field: String,
    #[serde(rename = "userAttribute")]
    pub user_attribute: Option<String>,
}

impl Schema {
    /// Load a schema from a YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ParseError> {
        let path_str = path.as_ref().display().to_string();
        let contents = std::fs::read_to_string(&path).map_err(|e| ParseError::Io {
            path: path_str,
            source: e,
        })?;
        serde_yaml::from_str(&contents).map_err(ParseError::from)
    }

    /// Get a semantic model by name
    pub fn get_model(&self, name: &str) -> Option<&SemanticModel> {
        self.semantic_models.iter().find(|m| m.name == name)
    }

    /// Get all unique dataset names referenced in the schema.
    /// 
    /// Returns fully qualified dataset names (e.g., "warehouse.orderfact")
    /// from both models (fact datasets) and dimensions.
    pub fn datasets(&self) -> Vec<String> {
        let mut datasets = Vec::new();

        for model in &self.semantic_models {
            // Fact datasets from dataset groups
            for group in &model.dataset_groups {
                for dataset in &group.datasets {
                    datasets.push(dataset.dataset.clone());
                }
            }
            
            // Dimension tables (non-virtual only)
            for dim in &model.dimensions {
                if let Some(table) = &dim.table {
                    datasets.push(table.clone());
                }
            }
        }

        // Deduplicate and sort
        datasets.sort();
        datasets.dedup();
        datasets
    }

    /// Get all datasets across all models and dataset groups
    /// 
    /// Returns references to GroupDataset structs with full source configuration.
    pub fn all_datasets(&self) -> Vec<&GroupDataset> {
        self.semantic_models
            .iter()
            .flat_map(|m| m.dataset_groups.iter())
            .flat_map(|g| g.datasets.iter())
            .collect()
    }
}

impl SemanticModel {
    /// Get a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<&Dimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }
    
    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<&Metric> {
        self.metrics.as_ref()?.iter().find(|m| m.name == name)
    }
    
    /// Get a dataset group by name
    pub fn get_dataset_group(&self, name: &str) -> Option<&DatasetGroup> {
        self.dataset_groups.iter().find(|g| g.name == name)
    }
    
    /// Get the first dataset group (convenience for single-group models)
    pub fn first_dataset_group(&self) -> Option<&DatasetGroup> {
        self.dataset_groups.first()
    }
    
    /// Get a dataset by physical dataset name (searches all groups)
    pub fn get_dataset(&self, dataset_name: &str) -> Option<&GroupDataset> {
        self.dataset_groups
            .iter()
            .flat_map(|g| g.datasets.iter())
            .find(|t| t.dataset == dataset_name)
    }
    
    /// Get a measure by name (searches all groups)
    pub fn get_measure(&self, name: &str) -> Option<&Measure> {
        self.dataset_groups
            .iter()
            .flat_map(|g| g.measures.iter())
            .find(|m| m.name == name)
    }
    
    /// Check if a measure exists in any dataset group
    pub fn has_measure(&self, name: &str) -> bool {
        self.dataset_groups.iter().any(|g| g.get_measure(name).is_some())
    }
    
    /// Get all unique measure names across all dataset groups
    pub fn measure_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.dataset_groups
            .iter()
            .flat_map(|g| g.measures.iter().map(|m| m.name.as_str()))
            .collect();
        names.sort();
        names.dedup();
        names
    }
    
    /// Get all datasets across all groups
    pub fn all_datasets(&self) -> Vec<&GroupDataset> {
        self.dataset_groups
            .iter()
            .flat_map(|g| g.datasets.iter())
            .collect()
    }
    
    /// Check if a dimension is defined at model level (can be queried with 2-part path)
    /// 
    /// Model-level dimensions are queryable across all dataset groups that reference them.
    /// The attr_name parameter is kept for API compatibility but not used in the check.
    pub fn is_conformed(&self, dim_name: &str, _attr_name: &str) -> bool {
        self.dimensions.iter().any(|d| d.name == dim_name)
    }
    
    /// Check if all dimension attributes in a query can use the cross-datasetGroup UNION path
    /// 
    /// Returns true if all dimensions are either:
    /// - Virtual dimensions (like `_dataset`) - implicitly work across dataset groups
    /// - Model-level dimensions - defined at model.dimensions, queryable with 2-part paths
    pub fn is_conformed_query(&self, dimension_attrs: &[String]) -> bool {
        if dimension_attrs.is_empty() {
            return false;
        }
        
        dimension_attrs.iter().all(|dim_attr| {
            let parts: Vec<&str> = dim_attr.split('.').collect();
            if parts.len() != 2 {
                return false;
            }
            // Check if dimension exists at model level (includes virtual dimensions)
            self.get_dimension(parts[0]).is_some()
        })
    }
}
