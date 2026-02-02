//! Table selection logic
//!
//! Selects the optimal table(s) from a model to serve a query based on:
//! - Dimension/attribute availability (feasibility)
//! - Measure availability (feasibility)
//! - Table size heuristic: fewer attributes = likely smaller/more aggregated (selection)
//!
//! Aggregate awareness is scoped to a single tableGroup. If multiple tableGroups
//! can serve the query, an error is returned - use a cross-tableGroup metric instead.
//!
//! Multi-table JOIN support:
//! When no single table has all required measures, multiple tables can be selected
//! and joined on their common dimensions. Uses "smallest table first" heuristic
//! to assign measures to the most aggregated tables.

use std::collections::{HashMap, HashSet};
use crate::model::{Model, TableGroup, GroupTable, Schema};
use super::error::SelectError;

/// Result of table selection - includes both the group and table
#[derive(Debug, Clone)]
pub struct SelectedTable<'a> {
    /// The table group containing the selected table
    pub group: &'a TableGroup,
    /// The selected table
    pub table: &'a GroupTable,
}

/// Result of multi-table selection for JOIN scenarios
/// 
/// When no single table has all required measures, multiple tables are selected
/// and joined on common dimensions. Each table is assigned specific measures.
#[derive(Debug)]
pub struct MultiTableSelection<'a> {
    /// The table group containing all selected tables
    pub group: &'a TableGroup,
    /// Selected tables with their assigned measures
    pub tables: Vec<TableWithMeasures<'a>>,
}

/// A selected table with its assigned measures
#[derive(Debug, Clone)]
pub struct TableWithMeasures<'a> {
    /// The selected table
    pub table: &'a GroupTable,
    /// Measures assigned to this table (using "first smallest wins" strategy)
    pub measures: Vec<String>,
}

/// Select the optimal table(s) to serve a query
/// 
/// Aggregate awareness is scoped to a single tableGroup:
/// - Finds which tableGroup(s) can serve the query
/// - If exactly one tableGroup matches, selects the optimal table within it
/// - If multiple tableGroups match, returns an error (use cross-tableGroup metric)
/// 
/// # Arguments
/// * `schema` - The schema containing dimension definitions
/// * `model` - The model to select tables from
/// * `required_dimensions` - Dimension.attribute paths needed (e.g., "dates.year")
/// * `required_measures` - Measure names needed
/// 
/// # Returns
/// A vector of SelectedTable (usually 1, but may be multiple
/// for partitioned/sharded tables in the future)
pub fn select_tables<'a>(
    schema: &'a Schema,
    model: &'a Model,
    required_dimensions: &[String],
    required_measures: &[String],
) -> Result<Vec<SelectedTable<'a>>, SelectError> {
    if model.table_groups.is_empty() {
        return Err(SelectError::NoTablesInModel {
            model: model.name.clone(),
        });
    }
    
    // Extract tableGroup qualifiers from three-part dimension paths
    // e.g., "adwords.dates.year" -> "adwords"
    let qualified_groups: HashSet<&str> = required_dimensions.iter()
        .filter_map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 { Some(parts[0]) } else { None }
        })
        .collect();
    
    // If there are qualified dimensions, only consider those specific tableGroups
    // Otherwise, consider all tableGroups
    let groups_to_check: Vec<&TableGroup> = if qualified_groups.is_empty() {
        model.table_groups.iter().collect()
    } else {
        model.table_groups.iter()
            .filter(|g| qualified_groups.contains(g.name.as_str()))
            .collect()
    };
    
    // Find all feasible tables, grouped by their tableGroup
    let mut feasible_by_group: HashMap<&str, Vec<SelectedTable>> = HashMap::new();
    
    for group in groups_to_check {
        for table in &group.tables {
            if is_feasible(model, group, table, required_dimensions, required_measures) {
                feasible_by_group
                    .entry(&group.name)
                    .or_default()
                    .push(SelectedTable { group, table });
            }
        }
    }
    
    if feasible_by_group.is_empty() {
        let missing = find_missing_requirements(schema, model, required_dimensions, required_measures);
        return Err(SelectError::NoFeasibleTable {
            model: model.name.clone(),
            reason: missing,
        });
    }
    
    // Check if multiple tableGroups can serve the query
    if feasible_by_group.len() > 1 {
        let group_names: Vec<String> = feasible_by_group.keys().map(|s| s.to_string()).collect();
        return Err(SelectError::AmbiguousTableGroup {
            model: model.name.clone(),
            table_groups: group_names,
        });
    }
    
    // Exactly one tableGroup - apply aggregate awareness within it
    let (_, feasible) = feasible_by_group.into_iter().next().unwrap();
    
    // Select the best table (fewest dimensions = likely more aggregated = smaller)
    // For now, return single best. Future: return multiple for partitioned tables.
    let best = feasible
        .into_iter()
        .min_by_key(|st| st.table.attribute_count())
        .unwrap();
    
    Ok(vec![best])
}

/// Select multiple tables for a JOIN when no single table has all measures
/// 
/// This is used when:
/// 1. Query requires measures that exist in different tables within the same tableGroup
/// 2. All tables share the required common dimensions (JOIN keys)
/// 
/// Uses "smallest table first" strategy: measures are assigned to the smallest
/// (most aggregated) table that has them. This minimizes data scanned.
/// 
/// # Arguments
/// * `schema` - The schema containing dimension definitions
/// * `model` - The model to select tables from
/// * `required_dimensions` - Dimension.attribute paths needed for JOIN keys
/// * `required_measures` - Measure names needed (may span multiple tables)
/// 
/// # Returns
/// A `MultiTableSelection` with tables and their assigned measures, or an error
pub fn select_tables_for_join<'a>(
    _schema: &'a Schema,
    model: &'a Model,
    required_dimensions: &[String],
    required_measures: &[String],
) -> Result<MultiTableSelection<'a>, SelectError> {
    if model.table_groups.is_empty() {
        return Err(SelectError::NoTablesInModel {
            model: model.name.clone(),
        });
    }
    
    // Extract tableGroup qualifiers from three-part dimension paths
    let qualified_groups: HashSet<&str> = required_dimensions.iter()
        .filter_map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 { Some(parts[0]) } else { None }
        })
        .collect();
    
    // Determine which tableGroup to use
    // If qualified, use that specific one; otherwise find the one with all measures
    let target_group = if qualified_groups.len() == 1 {
        let group_name = *qualified_groups.iter().next().unwrap();
        model.table_groups.iter()
            .find(|g| g.name == group_name)
            .ok_or_else(|| SelectError::NoFeasibleTable {
                model: model.name.clone(),
                reason: format!("TableGroup '{}' not found", group_name),
            })?
    } else if qualified_groups.len() > 1 {
        return Err(SelectError::AmbiguousTableGroup {
            model: model.name.clone(),
            table_groups: qualified_groups.iter().map(|s| s.to_string()).collect(),
        });
    } else {
        // Find tableGroup that has all required measures (across any of its tables)
        model.table_groups.iter()
            .find(|g| {
                required_measures.iter().all(|m| {
                    g.get_measure(m).is_some() && g.tables.iter().any(|t| t.has_measure(m))
                })
            })
            .ok_or_else(|| SelectError::NoFeasibleTable {
                model: model.name.clone(),
                reason: "No tableGroup has all required measures".to_string(),
            })?
    };
    
    // Find all tables that have the required dimensions (can participate in JOIN)
    let dimension_feasible: Vec<&GroupTable> = target_group.tables.iter()
        .filter(|table| has_all_dimensions(model, target_group, table, required_dimensions))
        .collect();
    
    if dimension_feasible.is_empty() {
        return Err(SelectError::NoFeasibleTable {
            model: model.name.clone(),
            reason: "No table has all required dimensions".to_string(),
        });
    }
    
    // Sort tables by attribute count (smallest/most aggregated first)
    let mut sorted_tables: Vec<&GroupTable> = dimension_feasible;
    sorted_tables.sort_by_key(|t| t.attribute_count());
    
    // Assign measures to tables using "first smallest wins" strategy
    let mut measure_assignments: HashMap<String, &GroupTable> = HashMap::new();
    let mut tables_used: HashSet<String> = HashSet::new();
    
    for measure_name in required_measures {
        // Find the smallest table that has this measure
        if let Some(table) = sorted_tables.iter()
            .find(|t| t.has_measure(measure_name))
        {
            measure_assignments.insert(measure_name.clone(), *table);
            tables_used.insert(table.table.clone());
        } else {
            return Err(SelectError::NoFeasibleTable {
                model: model.name.clone(),
                reason: format!("No table has measure '{}'", measure_name),
            });
        }
    }
    
    // Build the result: group tables by table, preserving smallest-first order
    let mut tables_with_measures: Vec<TableWithMeasures> = Vec::new();
    
    for table in &sorted_tables {
        if tables_used.contains(&table.table) {
            let measures: Vec<String> = measure_assignments.iter()
                .filter(|(_, t)| t.table == table.table)
                .map(|(m, _)| m.clone())
                .collect();
            
            if !measures.is_empty() {
                tables_with_measures.push(TableWithMeasures {
                    table,
                    measures,
                });
            }
        }
    }
    
    Ok(MultiTableSelection {
        group: target_group,
        tables: tables_with_measures,
    })
}

/// Check if a table has all required dimensions (for JOIN participation)
fn has_all_dimensions(
    model: &Model,
    group: &TableGroup,
    table: &GroupTable,
    required_dimensions: &[String],
) -> bool {
    for dim_attr in required_dimensions {
        // Skip virtual _table dimension
        if dim_attr.starts_with("_table.") {
            continue;
        }
        
        let parts: Vec<&str> = dim_attr.split('.').collect();
        if parts.len() == 3 {
            // Three-part: tableGroup.dimension.attribute
            let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            if tg_qualifier != group.name {
                continue; // Different tableGroup, not required from this group
            }
            let two_part = format!("{}.{}", dim_name, attr_name);
            if !table_has_attribute(model, group, table, &two_part) {
                return false;
            }
        } else if parts.len() == 2 {
            // Two-part: dimension.attribute
            if !table_has_attribute(model, group, table, dim_attr) {
                return false;
            }
        }
    }
    true
}

/// Check if a table can serve a query with the given requirements
fn is_feasible(
    model: &Model,
    group: &TableGroup,
    table: &GroupTable,
    required_dimensions: &[String],
    required_measures: &[String],
) -> bool {
    // Check all required dimension.attribute paths exist
    for dim_attr in required_dimensions {
        // Skip virtual _table dimension - it's available on all tables
        // and shouldn't affect table selection
        if dim_attr.starts_with("_table.") {
            continue;
        }
        
        // Handle tableGroup-qualified paths (e.g., "adwords.dates.year")
        let parts: Vec<&str> = dim_attr.split('.').collect();
        if parts.len() == 3 {
            let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            // If this dimension is qualified for a DIFFERENT tableGroup, skip it
            // (it's not required from this group)
            if tg_qualifier != group.name {
                continue;
            }
            // Check if this table has the dimension.attribute
            let two_part = format!("{}.{}", dim_name, attr_name);
            if !table_has_attribute(model, group, table, &two_part) {
                return false;
            }
        } else {
            // Two-part path: check normally
            if !table_has_attribute(model, group, table, dim_attr) {
                return false;
            }
        }
    }
    
    // Check all required measures exist in the group and are available on this table
    for measure_name in required_measures {
        // Measure must be defined in the group
        if group.get_measure(measure_name).is_none() {
            return false;
        }
        // Table must support this measure
        if !table.has_measure(measure_name) {
            return false;
        }
    }
    
    true
}

/// Check if a table has access to a dimension.attribute path
fn table_has_attribute(
    model: &Model,
    group: &TableGroup,
    table: &GroupTable,
    dim_attr_path: &str,
) -> bool {
    let parts: Vec<&str> = dim_attr_path.split('.').collect();
    if parts.len() != 2 {
        return false;
    }
    let (dim_name, attr_name) = (parts[0], parts[1]);
    
    // Check if table has this dimension
    let Some(table_attrs) = table.get_dimension_attributes(dim_name) else {
        return false;
    };
    
    // Check if the attribute is in the table's list
    if !table_attrs.iter().any(|a| a == attr_name) {
        return false;
    }
    
    // Verify the attribute exists in either:
    // 1. Degenerate dimension (defined in group with inline attributes)
    // 2. Joined dimension (defined in model)
    
    let Some(group_dim) = group.get_dimension(dim_name) else {
        return false;
    };
    
    if group_dim.is_degenerate() {
        // Degenerate: check inline attributes on the group dimension
        group_dim.get_attribute(attr_name).is_some()
    } else {
        // Joined: check model dimension
        model.get_dimension(dim_name)
            .and_then(|d| d.get_attribute(attr_name))
            .is_some()
    }
}

/// Build a helpful error message about what's missing
fn find_missing_requirements(
    _schema: &Schema,
    model: &Model,
    required_dimensions: &[String],
    required_measures: &[String],
) -> String {
    let mut missing = Vec::new();
    
    for dim_attr in required_dimensions {
        // Skip virtual _table dimension - it's always available
        if dim_attr.starts_with("_table.") {
            continue;
        }
        
        // Handle tableGroup-qualified paths (e.g., "adwords.dates.year")
        let parts: Vec<&str> = dim_attr.split('.').collect();
        let available_in_any = if parts.len() == 3 {
            let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            let two_part = format!("{}.{}", dim_name, attr_name);
            // Only check the specified tableGroup
            model.table_groups.iter()
                .filter(|group| group.name == tg_qualifier)
                .any(|group| {
                    group.tables.iter().any(|table| {
                        table_has_attribute(model, group, table, &two_part)
                    })
                })
        } else {
            // Two-part path: check all tableGroups
            model.table_groups.iter().any(|group| {
                group.tables.iter().any(|table| {
                    table_has_attribute(model, group, table, dim_attr)
                })
            })
        };
        
        if !available_in_any {
            missing.push(format!("dimension '{}' not available in any table", dim_attr));
        }
    }
    
    for measure_name in required_measures {
        let available_in_any = model.table_groups.iter().any(|group| {
            group.get_measure(measure_name).is_some() &&
            group.tables.iter().any(|table| table.has_measure(measure_name))
        });
        if !available_in_any {
            missing.push(format!("measure '{}' not available in any table", measure_name));
        }
    }
    
    if missing.is_empty() {
        "no single table has all required dimensions and measures".to_string()
    } else {
        missing.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn load_test_schema() -> Schema {
        Schema::from_file("test_data/steelwheels.yaml").unwrap()
    }
    
    fn load_marketing_schema() -> Schema {
        Schema::from_file("test_data/marketing.yaml").unwrap()
    }
    
    #[test]
    fn test_select_single_table() {
        let schema = load_test_schema();
        let model = schema.get_model("steelwheels").unwrap();
        
        let tables = select_tables(
            &schema,
            model,
            &["dates.year".to_string()],
            &["sales".to_string()],
        ).unwrap();
        
        assert_eq!(tables.len(), 1);
    }
    
    #[test]
    fn test_select_missing_dimension() {
        let schema = load_test_schema();
        let model = schema.get_model("steelwheels").unwrap();
        
        let result = select_tables(
            &schema,
            model,
            &["nonexistent.attr".to_string()],
            &["sales".to_string()],
        );
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_select_missing_measure() {
        let schema = load_test_schema();
        let model = schema.get_model("steelwheels").unwrap();
        
        let result = select_tables(
            &schema,
            model,
            &["dates.year".to_string()],
            &["nonexistent_measure".to_string()],
        );
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_ambiguous_measure_across_tablegroups() {
        // marketing.yaml has both adwords and facebookads tableGroups
        // Both have "clicks" and "impressions" measures
        let schema = load_marketing_schema();
        let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();
        
        // Query for "clicks" which exists in both tableGroups
        let result = select_tables(
            &schema,
            model,
            &["dates.date".to_string()],
            &["clicks".to_string()],
        );
        
        // Should error because multiple tableGroups can serve this query
        assert!(result.is_err());
        match result.unwrap_err() {
            SelectError::AmbiguousTableGroup { table_groups, .. } => {
                assert_eq!(table_groups.len(), 2);
                assert!(table_groups.contains(&"adwords".to_string()));
                assert!(table_groups.contains(&"facebookads".to_string()));
            }
            other => panic!("Expected AmbiguousTableGroup error, got: {:?}", other),
        }
    }
    
    #[test]
    fn test_unique_measure_selects_correct_tablegroup() {
        // marketing.yaml: "cost" only exists in adwords, "spend" only in facebookads
        let schema = load_marketing_schema();
        let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();
        
        // Query for "cost" which only exists in adwords
        let tables = select_tables(
            &schema,
            model,
            &["dates.date".to_string()],
            &["cost".to_string()],
        ).unwrap();
        
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].group.name, "adwords");
        
        // Query for "spend" which only exists in facebookads
        let tables = select_tables(
            &schema,
            model,
            &["dates.date".to_string()],
            &["spend".to_string()],
        ).unwrap();
        
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].group.name, "facebookads");
    }
}
