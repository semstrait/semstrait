//! Types for resolved query components

use crate::model::{Attribute, Dimension, Measure, Metric, Model, TableGroup, TableGroupDimension, GroupTable};

/// A resolved attribute reference (dimension.attribute)
/// 
/// Can be either from a joined dimension table or a degenerate dimension
/// where the columns live directly on the fact table.
#[derive(Debug, Clone)]
pub enum AttributeRef<'a> {
    /// Attribute from a joined dimension table
    Joined {
        group_dim: &'a TableGroupDimension,
        dimension: &'a Dimension,
        attribute: &'a Attribute,
    },
    /// Attribute from a degenerate dimension (columns on fact table)
    Degenerate {
        group_dim: &'a TableGroupDimension,
        attribute: &'a Attribute,
    },
}

impl<'a> AttributeRef<'a> {
    /// Get the table group dimension reference
    pub fn group_dim(&self) -> &'a TableGroupDimension {
        match self {
            Self::Joined { group_dim, .. } => group_dim,
            Self::Degenerate { group_dim, .. } => group_dim,
        }
    }
    
    /// Get the attribute
    pub fn attribute(&self) -> &'a Attribute {
        match self {
            Self::Joined { attribute, .. } => attribute,
            Self::Degenerate { attribute, .. } => attribute,
        }
    }
    
    /// Get the dimension name
    pub fn dimension_name(&self) -> &str {
        match self {
            Self::Joined { dimension, .. } => &dimension.name,
            Self::Degenerate { group_dim, .. } => &group_dim.name,
        }
    }
    
    /// Get the full dimension (only for Joined)
    pub fn dimension(&self) -> Option<&'a Dimension> {
        match self {
            Self::Joined { dimension, .. } => Some(dimension),
            Self::Degenerate { .. } => None,
        }
    }
    
    /// Is this a degenerate dimension?
    pub fn is_degenerate(&self) -> bool {
        matches!(self, Self::Degenerate { .. })
    }
    
    /// Is this an inline attribute (no join needed)?
    pub fn is_inline(&self) -> bool {
        match self {
            Self::Degenerate { .. } => true,
            Self::Joined { group_dim, .. } => group_dim.join.is_none(),
        }
    }
}

/// A resolved filter with attribute reference
#[derive(Debug, Clone)]
pub struct ResolvedFilter<'a> {
    pub attribute: AttributeRef<'a>,
    pub operator: String,
    pub value: serde_json::Value,
}

/// The result of resolving an analytics query against a schema
#[derive(Debug)]
pub struct ResolvedQuery<'a> {
    /// The model (for metrics and model-level config)
    pub model: &'a Model,
    /// The selected table group (for dimensions and measures)
    pub table_group: &'a TableGroup,
    /// The selected table (for physical table name and columns)
    pub table: &'a GroupTable,
    /// Dimensions needed for this query (from rows, columns, and filters)
    pub dimensions: Vec<ResolvedDimension<'a>>,
    /// Attributes for rows (GROUP BY, first axis)
    pub row_attributes: Vec<AttributeRef<'a>>,
    /// Attributes for columns (GROUP BY, second axis)  
    pub column_attributes: Vec<AttributeRef<'a>>,
    /// Measures to aggregate (internal - derived from metric dependencies)
    pub measures: Vec<&'a Measure>,
    /// Metrics - the public query interface
    pub metrics: Vec<&'a Metric>,
    /// Filters to apply (WHERE clause)
    pub filters: Vec<ResolvedFilter<'a>>,
}

impl<'a> ResolvedQuery<'a> {
    /// Build semantic output column names for the query result.
    /// Returns names in order: row_attributes, column_attributes, metrics
    pub fn output_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        
        // Row attributes: "dimension.attribute"
        for attr in &self.row_attributes {
            names.push(format!("{}.{}", attr.dimension_name(), attr.attribute().name));
        }
        
        // Column attributes: "dimension.attribute"
        for attr in &self.column_attributes {
            names.push(format!("{}.{}", attr.dimension_name(), attr.attribute().name));
        }
        
        // Metrics: the public query interface
        for metric in &self.metrics {
            names.push(metric.name.clone());
        }
        
        names
    }
}

/// A resolved dimension with its reference and definition
#[derive(Debug)]
pub enum ResolvedDimension<'a> {
    /// Dimension from a joined table
    Joined {
        group_dim: &'a TableGroupDimension,
        dimension: &'a Dimension,
    },
    /// Degenerate dimension (columns on fact table)
    Degenerate {
        group_dim: &'a TableGroupDimension,
    },
}

impl<'a> ResolvedDimension<'a> {
    /// Get the table group dimension reference
    pub fn group_dim(&self) -> &'a TableGroupDimension {
        match self {
            Self::Joined { group_dim, .. } => group_dim,
            Self::Degenerate { group_dim } => group_dim,
        }
    }
    
    /// Get the dimension name
    pub fn name(&self) -> &str {
        &self.group_dim().name
    }
    
    /// Is this a degenerate dimension?
    pub fn is_degenerate(&self) -> bool {
        matches!(self, Self::Degenerate { .. })
    }
    
    /// Returns true if this dimension doesn't require a join
    pub fn is_inline(&self) -> bool {
        self.group_dim().join.is_none()
    }
    
    /// Get the full dimension (only for Joined)
    pub fn dimension(&self) -> Option<&'a Dimension> {
        match self {
            Self::Joined { dimension, .. } => Some(dimension),
            Self::Degenerate { .. } => None,
        }
    }
}
