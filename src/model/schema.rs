//! Root schema definition

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use std::fmt;
use std::path::Path;
use super::dimension::Dimension;
use super::measure::Measure;
use super::metric::Metric;
use super::tablegroup::{TableGroup, GroupTable};
use crate::error::ParseError;

/// The root semantic schema containing models
#[derive(Debug, Deserialize)]
pub struct Schema {
    pub models: Vec<Model>,
}

/// A conformed dimension - can be queried across multiple tableGroups
/// 
/// Supports two YAML forms:
/// - Simple: `- dates` (all attributes are conformed)
/// - Detailed: `- campaign: [country, market]` (only listed attributes are conformed)
#[derive(Debug, Clone)]
pub struct ConformedDimension {
    /// Name of the dimension
    pub name: String,
    /// Specific attributes that are conformed, or None if all attributes are conformed
    pub attributes: Option<Vec<String>>,
}

impl ConformedDimension {
    /// Check if a specific attribute is conformed for this dimension
    pub fn is_attribute_conformed(&self, attr_name: &str) -> bool {
        match &self.attributes {
            None => true, // All attributes are conformed
            Some(attrs) => attrs.iter().any(|a| a == attr_name),
        }
    }
}

// Custom deserializer for ConformedDimension to support both:
// - `- dates` (string)
// - `- campaign: [country, market]` (map)
impl<'de> Deserialize<'de> for ConformedDimension {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ConformedDimensionVisitor;

        impl<'de> Visitor<'de> for ConformedDimensionVisitor {
            type Value = ConformedDimension;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string or a map with dimension name and attributes")
            }

            // Simple form: `- dates`
            fn visit_str<E>(self, value: &str) -> Result<ConformedDimension, E>
            where
                E: de::Error,
            {
                Ok(ConformedDimension {
                    name: value.to_string(),
                    attributes: None,
                })
            }

            // Detailed form: `- campaign: [country, market]`
            fn visit_map<M>(self, mut map: M) -> Result<ConformedDimension, M::Error>
            where
                M: MapAccess<'de>,
            {
                let entry: Option<(String, Vec<String>)> = map.next_entry()?;
                match entry {
                    Some((name, attributes)) => Ok(ConformedDimension {
                        name,
                        attributes: Some(attributes),
                    }),
                    None => Err(de::Error::custom("expected dimension name and attributes")),
                }
            }
        }

        deserializer.deserialize_any(ConformedDimensionVisitor)
    }
}

/// A model - the queryable business entity
/// 
/// Contains one or more table groups that share dimension and measure definitions.
/// The selector picks the optimal table based on query requirements.
#[derive(Debug, Deserialize)]
pub struct Model {
    pub name: String,
    /// Namespace for the model (e.g., organization or project identifier)
    pub namespace: Option<String>,
    /// Shared dimensions (with physical tables) available to table groups
    #[serde(default)]
    pub dimensions: Vec<Dimension>,
    /// Conformed dimensions - can be queried across multiple tableGroups
    /// Queries on conformed dimensions automatically UNION across all tableGroups that have them
    #[serde(rename = "conformedDimensions", default)]
    pub conformed_dimensions: Vec<ConformedDimension>,
    /// Table groups - each group contains tables that share field definitions
    #[serde(rename = "tableGroups")]
    pub table_groups: Vec<TableGroup>,
    /// Metrics - derived calculations from measures (model-level, shared across table groups)
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

    /// Get a model by name
    pub fn get_model(&self, name: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.name == name)
    }

    /// Get all unique table names referenced in the schema.
    /// 
    /// Returns fully qualified table names (e.g., "warehouse.orderfact")
    /// from both models (fact tables) and dimensions.
    pub fn tables(&self) -> Vec<String> {
        let mut tables = Vec::new();

        for model in &self.models {
            // Fact tables from table groups
            for group in &model.table_groups {
                for table in &group.tables {
                    tables.push(table.table.clone());
                }
            }
            
            // Dimension tables (non-virtual only)
            for dim in &model.dimensions {
                if let Some(table) = &dim.table {
                    tables.push(table.clone());
                }
            }
        }

        // Deduplicate and sort
        tables.sort();
        tables.dedup();
        tables
    }

    /// Get all tables across all models and table groups
    /// 
    /// Returns references to GroupTable structs with full source configuration.
    pub fn all_tables(&self) -> Vec<&GroupTable> {
        self.models
            .iter()
            .flat_map(|m| m.table_groups.iter())
            .flat_map(|g| g.tables.iter())
            .collect()
    }
}

impl Model {
    /// Get a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<&Dimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }
    
    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<&Metric> {
        self.metrics.as_ref()?.iter().find(|m| m.name == name)
    }
    
    /// Get a table group by name
    pub fn get_table_group(&self, name: &str) -> Option<&TableGroup> {
        self.table_groups.iter().find(|g| g.name == name)
    }
    
    /// Get the first table group (convenience for single-group models)
    pub fn first_table_group(&self) -> Option<&TableGroup> {
        self.table_groups.first()
    }
    
    /// Get a table by physical table name (searches all groups)
    pub fn get_table(&self, table_name: &str) -> Option<&GroupTable> {
        self.table_groups
            .iter()
            .flat_map(|g| g.tables.iter())
            .find(|t| t.table == table_name)
    }
    
    /// Get a measure by name (searches all groups)
    pub fn get_measure(&self, name: &str) -> Option<&Measure> {
        self.table_groups
            .iter()
            .flat_map(|g| g.measures.iter())
            .find(|m| m.name == name)
    }
    
    /// Check if a measure exists in any table group
    pub fn has_measure(&self, name: &str) -> bool {
        self.table_groups.iter().any(|g| g.get_measure(name).is_some())
    }
    
    /// Get all unique measure names across all table groups
    pub fn measure_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.table_groups
            .iter()
            .flat_map(|g| g.measures.iter().map(|m| m.name.as_str()))
            .collect();
        names.sort();
        names.dedup();
        names
    }
    
    /// Get all tables across all groups
    pub fn all_tables(&self) -> Vec<&GroupTable> {
        self.table_groups
            .iter()
            .flat_map(|g| g.tables.iter())
            .collect()
    }
    
    /// Check if a dimension.attribute is conformed (can be queried across tableGroups)
    /// 
    /// Returns true if:
    /// - The dimension is listed in conformedDimensions with no attribute restrictions, OR
    /// - The dimension is listed with this specific attribute
    pub fn is_conformed(&self, dim_name: &str, attr_name: &str) -> bool {
        self.conformed_dimensions
            .iter()
            .find(|cd| cd.name == dim_name)
            .map(|cd| cd.is_attribute_conformed(attr_name))
            .unwrap_or(false)
    }
    
    /// Check if all dimension attributes in a query are conformed
    /// 
    /// Virtual dimensions (like `_table`) are implicitly conformed - they don't need
    /// to be listed in conformedDimensions.
    /// 
    /// Returns true if:
    /// - All dimensions in the query are virtual (implicitly conformed), OR
    /// - All non-virtual dimensions are explicitly listed in conformedDimensions
    pub fn is_conformed_query(&self, dimension_attrs: &[String]) -> bool {
        // Separate virtual and non-virtual dimensions
        let (virtual_dims, physical_dims): (Vec<_>, Vec<_>) = dimension_attrs.iter()
            .partition(|dim_attr| {
                let parts: Vec<&str> = dim_attr.split('.').collect();
                if parts.len() != 2 {
                    return false;
                }
                // Check if this dimension is virtual (at model level)
                self.get_dimension(parts[0])
                    .map(|d| d.is_virtual())
                    .unwrap_or(false)
            });
        
        // If there are only virtual dimensions, allow UNION path
        // (virtual dimensions are implicitly conformed)
        if physical_dims.is_empty() && !virtual_dims.is_empty() {
            return true;
        }
        
        // If there are physical dimensions, they must all be conformed
        if !physical_dims.is_empty() {
            // Need conformedDimensions to be defined
            if self.conformed_dimensions.is_empty() {
                return false;
            }
            
            // Check all physical dimensions are conformed
            return physical_dims.iter().all(|dim_attr| {
                let parts: Vec<&str> = dim_attr.split('.').collect();
                if parts.len() != 2 {
                    return false;
                }
                self.is_conformed(parts[0], parts[1])
            });
        }
        
        // Empty query - not conformed
        false
    }
}
