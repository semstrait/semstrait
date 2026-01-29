//! Dimension and attribute types

use serde::Deserialize;
use super::types::DataType;
use super::tablegroup::Source;

/// A dimension definition with its attributes
#[derive(Debug, Deserialize)]
pub struct Dimension {
    pub name: String,
    /// Data source configuration (parquet path, iceberg table, etc.)
    pub source: Source,
    pub table: String,
    pub alias: Option<String>,
    pub label: Option<String>,
    /// Human-readable description for UIs and LLMs
    pub description: Option<String>,
    pub attributes: Vec<Attribute>,
}

/// Join specification between fact and dimension tables
#[derive(Debug, Deserialize, Clone)]
pub struct Join {
    #[serde(rename = "leftKey")]
    pub left_key: String,
    #[serde(rename = "rightKey")]
    pub right_key: String,
    #[serde(rename = "rightAlias")]
    pub right_alias: Option<String>,
}

/// An attribute (column) within a dimension
#[derive(Debug, Deserialize, Clone)]
pub struct Attribute {
    pub name: String,
    pub column: Option<String>,
    pub label: Option<String>,
    /// Human-readable description for UIs and LLMs
    pub description: Option<String>,
    /// Sample values (helps LLMs understand valid inputs)
    pub examples: Option<Vec<String>>,
    /// Data type. Defaults to String if not specified.
    #[serde(rename = "type", default)]
    pub data_type: DataType,
}

impl Dimension {
    /// Get the parquet path if source is Parquet
    pub fn parquet_path(&self) -> Option<&str> {
        match &self.source {
            Source::Parquet { path } => Some(path),
        }
    }
    
    /// Get an attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Find the attribute that serves as the join key
    /// 
    /// The key attribute is the one whose column matches the given key column name.
    /// This is used to determine if a table needs a join - if the table's attribute
    /// list includes the key attribute, it needs to join to get other attributes.
    pub fn key_attribute(&self, key_column: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.column_name() == key_column)
    }

    /// Get all attribute names
    pub fn attribute_names(&self) -> Vec<&str> {
        self.attributes.iter().map(|a| a.name.as_str()).collect()
    }
}

impl Attribute {
    /// Get the column name, defaulting to attribute name if not specified
    pub fn column_name(&self) -> &str {
        self.column.as_deref().unwrap_or(&self.name)
    }

    /// Get the data type
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
}
