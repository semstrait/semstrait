//! Types for resolved query components

use crate::model::{Attribute, Dimension, Measure, Metric, Model, TableGroup, TableGroupDimension, GroupTable};

/// A resolved attribute reference (dimension.attribute)
/// 
/// Can be either from a joined dimension table, a degenerate dimension
/// where the columns live directly on the fact table, or a virtual
/// metadata attribute from the `_table` dimension.
#[derive(Debug, Clone)]
pub enum AttributeRef<'a> {
    /// Attribute from a joined dimension table
    Joined {
        group_dim: &'a TableGroupDimension,
        dimension: &'a Dimension,
        attribute: &'a Attribute,
        /// Optional tableGroup qualifier (e.g., "adwords" in "adwords.campaign.name")
        /// When set, this attribute is explicitly scoped to a specific tableGroup
        table_group_qualifier: Option<String>,
    },
    /// Attribute from a degenerate dimension (columns on fact table)
    Degenerate {
        group_dim: &'a TableGroupDimension,
        attribute: &'a Attribute,
        /// Optional tableGroup qualifier (e.g., "adwords" in "adwords.campaign.name")
        /// When set, this attribute is explicitly scoped to a specific tableGroup
        table_group_qualifier: Option<String>,
    },
    /// Virtual metadata attribute from the `_table` dimension
    /// Emits as a constant literal value, not a column reference
    Meta {
        /// The attribute name (e.g., "tableGroup", "uuid", or a property key)
        name: String,
        /// The resolved value to emit as a literal
        value: String,
    },
}

impl<'a> AttributeRef<'a> {
    /// Get the table group dimension reference (None for Meta)
    pub fn group_dim(&self) -> Option<&'a TableGroupDimension> {
        match self {
            Self::Joined { group_dim, .. } => Some(group_dim),
            Self::Degenerate { group_dim, .. } => Some(group_dim),
            Self::Meta { .. } => None,
        }
    }
    
    /// Get the attribute (panics for Meta - use attribute_opt instead)
    pub fn attribute(&self) -> &'a Attribute {
        match self {
            Self::Joined { attribute, .. } => attribute,
            Self::Degenerate { attribute, .. } => attribute,
            Self::Meta { .. } => panic!("Meta attributes don't have an Attribute reference"),
        }
    }
    
    /// Get the attribute if available (None for Meta)
    pub fn attribute_opt(&self) -> Option<&'a Attribute> {
        match self {
            Self::Joined { attribute, .. } => Some(attribute),
            Self::Degenerate { attribute, .. } => Some(attribute),
            Self::Meta { .. } => None,
        }
    }
    
    /// Get the dimension name
    pub fn dimension_name(&self) -> &str {
        match self {
            Self::Joined { dimension, .. } => &dimension.name,
            Self::Degenerate { group_dim, .. } => &group_dim.name,
            Self::Meta { .. } => "_table",
        }
    }
    
    /// Get the attribute name
    pub fn attribute_name(&self) -> &str {
        match self {
            Self::Joined { attribute, .. } => &attribute.name,
            Self::Degenerate { attribute, .. } => &attribute.name,
            Self::Meta { name, .. } => name,
        }
    }
    
    /// Get the full dimension (only for Joined)
    pub fn dimension(&self) -> Option<&'a Dimension> {
        match self {
            Self::Joined { dimension, .. } => Some(dimension),
            Self::Degenerate { .. } => None,
            Self::Meta { .. } => None,
        }
    }
    
    /// Is this a degenerate dimension?
    pub fn is_degenerate(&self) -> bool {
        matches!(self, Self::Degenerate { .. })
    }
    
    /// Is this a virtual metadata attribute?
    pub fn is_meta(&self) -> bool {
        matches!(self, Self::Meta { .. })
    }
    
    /// Is this an inline attribute (no join needed)?
    /// Meta attributes are always inline (they're literals).
    pub fn is_inline(&self) -> bool {
        match self {
            Self::Degenerate { .. } => true,
            Self::Joined { group_dim, .. } => group_dim.join.is_none(),
            Self::Meta { .. } => true,
        }
    }
    
    /// Get the meta value (only for Meta variant)
    pub fn meta_value(&self) -> Option<&str> {
        match self {
            Self::Meta { value, .. } => Some(value),
            _ => None,
        }
    }
    
    /// Get the tableGroup qualifier if this attribute is explicitly scoped
    pub fn table_group_qualifier(&self) -> Option<&str> {
        match self {
            Self::Joined { table_group_qualifier, .. } => table_group_qualifier.as_deref(),
            Self::Degenerate { table_group_qualifier, .. } => table_group_qualifier.as_deref(),
            Self::Meta { .. } => None,
        }
    }
    
    /// Is this attribute qualified with a specific tableGroup?
    pub fn is_table_group_qualified(&self) -> bool {
        self.table_group_qualifier().is_some()
    }
    
    /// Get the semantic name for this attribute
    /// Returns "tableGroup.dimension.attribute" if qualified, "dimension.attribute" otherwise
    pub fn semantic_name(&self) -> String {
        match self.table_group_qualifier() {
            Some(tg) => format!("{}.{}.{}", tg, self.dimension_name(), self.attribute_name()),
            None => format!("{}.{}", self.dimension_name(), self.attribute_name()),
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
        
        // Row attributes: "dimension.attribute", "_table.attribute", or "tableGroup.dimension.attribute"
        for attr in &self.row_attributes {
            names.push(attr.semantic_name());
        }
        
        // Column attributes: "dimension.attribute", "_table.attribute", or "tableGroup.dimension.attribute"
        for attr in &self.column_attributes {
            names.push(attr.semantic_name());
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
    /// Virtual `_table` metadata dimension
    /// No physical table - all attributes are constant literals
    Meta,
}

impl<'a> ResolvedDimension<'a> {
    /// Get the table group dimension reference (None for Meta)
    pub fn group_dim(&self) -> Option<&'a TableGroupDimension> {
        match self {
            Self::Joined { group_dim, .. } => Some(group_dim),
            Self::Degenerate { group_dim } => Some(group_dim),
            Self::Meta => None,
        }
    }
    
    /// Get the dimension name
    pub fn name(&self) -> &str {
        match self {
            Self::Joined { group_dim, .. } => &group_dim.name,
            Self::Degenerate { group_dim } => &group_dim.name,
            Self::Meta => "_table",
        }
    }
    
    /// Is this a degenerate dimension?
    pub fn is_degenerate(&self) -> bool {
        matches!(self, Self::Degenerate { .. })
    }
    
    /// Is this the virtual metadata dimension?
    pub fn is_meta(&self) -> bool {
        matches!(self, Self::Meta)
    }
    
    /// Returns true if this dimension doesn't require a join
    /// Meta dimensions are always inline (they're literals).
    pub fn is_inline(&self) -> bool {
        match self {
            Self::Joined { group_dim, .. } => group_dim.join.is_none(),
            Self::Degenerate { .. } => true,
            Self::Meta => true,
        }
    }
    
    /// Get the full dimension (only for Joined)
    pub fn dimension(&self) -> Option<&'a Dimension> {
        match self {
            Self::Joined { dimension, .. } => Some(dimension),
            Self::Degenerate { .. } => None,
            Self::Meta => None,
        }
    }
}
