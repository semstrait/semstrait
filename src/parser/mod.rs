//! Schema parser (verb module)
//!
//! Transforms YAML files into model types.

use std::path::Path;
use crate::error::ParseError;
use crate::model::Schema;

/// Parse a schema from a YAML file
pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Schema, ParseError> {
    let path_str = path.as_ref().display().to_string();
    let contents = std::fs::read_to_string(&path).map_err(|e| ParseError::Io {
        path: path_str,
        source: e,
    })?;
    parse_str(&contents)
}

/// Parse a schema from a YAML string
pub fn parse_str(yaml: &str) -> Result<Schema, ParseError> {
    serde_yaml::from_str(yaml).map_err(ParseError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MeasureExpr, MetricExpr, ExprNode, Aggregation};

    #[test]
    fn test_parse_steelwheels() {
        let schema = parse_file("test_data/steelwheels.yaml").unwrap();
        
        // Check models
        assert_eq!(schema.models.len(), 1);
        let model = schema.get_model("steelwheels").unwrap();
        
        // Check table groups
        assert_eq!(model.table_groups.len(), 1);
        let group = model.first_table_group().unwrap();
        assert_eq!(group.name, "orders");
        
        // Check tables within group
        assert_eq!(group.tables.len(), 1);
        let table = group.get_table("steelwheels.orderfact").unwrap();
        assert_eq!(table.table, "steelwheels.orderfact");
        
        // Check dimensions (now on model) - includes _table virtual dimension
        assert_eq!(model.dimensions.len(), 3);
        let dates = model.get_dimension("dates").unwrap();
        assert_eq!(dates.table, Some("steelwheels.dates".to_string()));
        assert!(!dates.is_virtual());
        assert_eq!(dates.attributes.len(), 4);
        
        // Check _table virtual dimension
        let table_dim = model.get_dimension("_table").unwrap();
        assert!(table_dim.is_virtual());
        assert!(table_dim.table.is_none());
        
        // Check key attribute detection for dates dimension
        // The join key is time_id, which maps to the 'date' attribute
        let date_key_attr = dates.key_attribute("time_id").unwrap();
        assert_eq!(date_key_attr.name, "date");
        
        // Check measures (now on group, not table)
        assert_eq!(group.measures.len(), 4);
        let sales = group.get_measure("sales").unwrap();
        assert_eq!(sales.aggregation, Aggregation::Sum);
        assert!(matches!(&sales.expr, MeasureExpr::Column(s) if s == "totalprice"));
        
        // Check structured expression measure
        let revenue = group.get_measure("revenue").unwrap();
        assert!(matches!(&revenue.expr, MeasureExpr::Structured(_)));
        
        // Check CASE WHEN measure
        let premium = group.get_measure("premium_sales").unwrap();
        assert!(matches!(&premium.expr, MeasureExpr::Structured(ExprNode::Case(_))));
        
        // Check metrics (still on model)
        let metric = model.get_metric("avg_unit_price").unwrap();
        assert_eq!(metric.label.as_deref(), Some("Average Unit Price"));
        assert!(matches!(&metric.expr, MetricExpr::Structured(_)));
        
        // Columns are now optional - join detection is based on attribute inclusion
        // The table should NOT have explicit columns defined (they're inferred)
        assert!(table.columns.is_none());
        
        // Check table's dimension and measure references
        assert!(table.has_dimension("dates"));
        assert!(table.has_dimension("markets"));
        assert!(table.has_dimension("flags"));
        assert!(table.has_measure("sales"));
        assert!(table.has_measure("quantity"));
        
        // Check attribute-based join detection:
        // - dates: [date, year, quarter, month] includes 'date' (key attr) → needs join
        // - markets: [customer, ...] includes 'customer' (key attr) → needs join
        assert!(table.has_dimension_attribute("dates", "date"));  // Key attr present → JOIN
        assert!(table.has_dimension_attribute("markets", "customer"));  // Key attr present → JOIN
    }

    #[test]
    fn test_parse_invalid_yaml() {
        let result = parse_str("not: [valid: yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_marketing_cross_table_group_metric() {
        let schema = parse_file("test_data/marketing.yaml").unwrap();
        
        // Check models
        assert_eq!(schema.models.len(), 1);
        let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();
        
        // Check table groups (should have adwords and facebookads)
        assert_eq!(model.table_groups.len(), 2);
        
        // Check adwords group
        let adwords = model.get_table_group("adwords").unwrap();
        assert!(adwords.get_measure("cost").is_some());
        
        // Check facebookads group
        let facebookads = model.get_table_group("facebookads").unwrap();
        assert!(facebookads.get_measure("spend").is_some());
        
        // Check cross-tableGroup metric
        let fun_cost = model.get_metric("fun-cost").unwrap();
        
        // Verify it's detected as a cross-tableGroup metric
        assert!(fun_cost.is_cross_table_group());
        
        // Verify tableGroup-to-measure mappings
        let mappings = fun_cost.table_group_measures();
        assert_eq!(mappings.len(), 2);
        assert!(mappings.iter().any(|(tg, m)| tg == "adwords" && m == "cost"));
        assert!(mappings.iter().any(|(tg, m)| tg == "facebookads" && m == "spend"));
    }
}
