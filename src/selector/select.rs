//! Dataset selection logic
//!
//! Selects the optimal dataset(s) from a model to serve a query based on:
//! - Dimension/attribute availability (feasibility)
//! - Measure availability (feasibility)
//! - Dataset size heuristic: fewer attributes = likely smaller/more aggregated (selection)
//!
//! Aggregate awareness is scoped to a single datasetGroup. If multiple datasetGroups
//! can serve the query, an error is returned - use a cross-datasetGroup metric instead.
//!
//! Multi-dataset JOIN support:
//! When no single dataset has all required measures, multiple datasets can be selected
//! and joined on their common dimensions. Uses "smallest dataset first" heuristic
//! to assign measures to the most aggregated datasets.

use std::collections::{HashMap, HashSet};
use crate::semantic_model::{SemanticModel, DatasetGroup, GroupDataset, Schema};
use super::error::SelectError;

/// Result of dataset selection - includes both the group and dataset
#[derive(Debug, Clone)]
pub struct SelectedDataset<'a> {
    /// The dataset group containing the selected dataset
    pub group: &'a DatasetGroup,
    /// The selected dataset
    pub dataset: &'a GroupDataset,
}

/// Result of multi-dataset selection for JOIN scenarios
/// 
/// When no single dataset has all required measures, multiple datasets are selected
/// and joined on common dimensions. Each dataset is assigned specific measures.
#[derive(Debug)]
pub struct MultiDatasetSelection<'a> {
    /// The dataset group containing all selected datasets
    pub group: &'a DatasetGroup,
    /// Selected datasets with their assigned measures
    pub datasets: Vec<DatasetWithMeasures<'a>>,
}

/// A selected dataset with its assigned measures
#[derive(Debug, Clone)]
pub struct DatasetWithMeasures<'a> {
    /// The selected dataset
    pub dataset: &'a GroupDataset,
    /// Measures assigned to this dataset (using "first smallest wins" strategy)
    pub measures: Vec<String>,
}

/// Select the optimal dataset(s) to serve a query
/// 
/// Aggregate awareness is scoped to a single datasetGroup:
/// - Finds which datasetGroup(s) can serve the query
/// - If exactly one datasetGroup matches, selects the optimal dataset within it
/// - If multiple datasetGroups match, returns an error (use cross-datasetGroup metric)
/// 
/// # Arguments
/// * `schema` - The schema containing dimension definitions
/// * `model` - The model to select datasets from
/// * `required_dimensions` - Dimension.attribute paths needed (e.g., "dates.year")
/// * `required_measures` - Measure names needed
/// 
/// # Returns
/// A vector of SelectedDataset (usually 1, but may be multiple
/// for partitioned/sharded datasets in the future)
pub fn select_datasets<'a>(
    schema: &'a Schema,
    model: &'a SemanticModel,
    required_dimensions: &[String],
    required_measures: &[String],
) -> Result<Vec<SelectedDataset<'a>>, SelectError> {
    if model.dataset_groups.is_empty() {
        return Err(SelectError::NoDatasetsInModel {
            model: model.name.clone(),
        });
    }
    
    // Extract datasetGroup qualifiers from three-part dimension paths
    // e.g., "adwords.dates.year" -> "adwords"
    let qualified_groups: HashSet<&str> = required_dimensions.iter()
        .filter_map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 { Some(parts[0]) } else { None }
        })
        .collect();
    
    // If there are qualified dimensions, only consider those specific datasetGroups
    // Otherwise, consider all datasetGroups
    let groups_to_check: Vec<&DatasetGroup> = if qualified_groups.is_empty() {
        model.dataset_groups.iter().collect()
    } else {
        model.dataset_groups.iter()
            .filter(|g| qualified_groups.contains(g.name.as_str()))
            .collect()
    };
    
    // Find all feasible datasets, grouped by their datasetGroup
    let mut feasible_by_group: HashMap<&str, Vec<SelectedDataset>> = HashMap::new();
    
    for group in groups_to_check {
        for dataset in &group.datasets {
            if is_feasible(model, group, dataset, required_dimensions, required_measures) {
                feasible_by_group
                    .entry(&group.name)
                    .or_default()
                    .push(SelectedDataset { group, dataset });
            }
        }
    }
    
    if feasible_by_group.is_empty() {
        let missing = find_missing_requirements(schema, model, required_dimensions, required_measures);
        return Err(SelectError::NoFeasibleDataset {
            model: model.name.clone(),
            reason: missing,
        });
    }
    
    // Check if multiple datasetGroups can serve the query
    if feasible_by_group.len() > 1 {
        let group_names: Vec<String> = feasible_by_group.keys().map(|s| s.to_string()).collect();
        return Err(SelectError::AmbiguousDatasetGroup {
            model: model.name.clone(),
            dataset_groups: group_names,
        });
    }
    
    // Exactly one datasetGroup - apply aggregate awareness within it
    let (_, feasible) = feasible_by_group.into_iter().next().unwrap();
    
    // Select the best dataset (fewest dimensions = likely more aggregated = smaller)
    // For now, return single best. Future: return multiple for partitioned datasets.
    let best = feasible
        .into_iter()
        .min_by_key(|st| st.dataset.attribute_count())
        .unwrap();
    
    Ok(vec![best])
}

/// Select multiple datasets for a JOIN when no single dataset has all measures
/// 
/// This is used when:
/// 1. Query requires measures that exist in different datasets within the same datasetGroup
/// 2. All datasets share the required common dimensions (JOIN keys)
/// 
/// Uses "smallest dataset first" strategy: measures are assigned to the smallest
/// (most aggregated) dataset that has them. This minimizes data scanned.
/// 
/// # Arguments
/// * `schema` - The schema containing dimension definitions
/// * `model` - The model to select datasets from
/// * `required_dimensions` - Dimension.attribute paths needed for JOIN keys
/// * `required_measures` - Measure names needed (may span multiple datasets)
/// 
/// # Returns
/// A `MultiDatasetSelection` with datasets and their assigned measures, or an error
pub fn select_datasets_for_join<'a>(
    _schema: &'a Schema,
    model: &'a SemanticModel,
    required_dimensions: &[String],
    required_measures: &[String],
) -> Result<MultiDatasetSelection<'a>, SelectError> {
    if model.dataset_groups.is_empty() {
        return Err(SelectError::NoDatasetsInModel {
            model: model.name.clone(),
        });
    }
    
    // Extract datasetGroup qualifiers from three-part dimension paths
    let qualified_groups: HashSet<&str> = required_dimensions.iter()
        .filter_map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 { Some(parts[0]) } else { None }
        })
        .collect();
    
    // Determine which datasetGroup to use
    // If qualified, use that specific one; otherwise find the one with all measures
    let target_group = if qualified_groups.len() == 1 {
        let group_name = *qualified_groups.iter().next().unwrap();
        model.dataset_groups.iter()
            .find(|g| g.name == group_name)
            .ok_or_else(|| SelectError::NoFeasibleDataset {
                model: model.name.clone(),
                reason: format!("DatasetGroup '{}' not found", group_name),
            })?
    } else if qualified_groups.len() > 1 {
        return Err(SelectError::AmbiguousDatasetGroup {
            model: model.name.clone(),
            dataset_groups: qualified_groups.iter().map(|s| s.to_string()).collect(),
        });
    } else {
        // Find datasetGroup that has all required measures (across any of its datasets)
        model.dataset_groups.iter()
            .find(|g| {
                required_measures.iter().all(|m| {
                    g.get_measure(m).is_some() && g.datasets.iter().any(|t| t.has_measure(m))
                })
            })
            .ok_or_else(|| SelectError::NoFeasibleDataset {
                model: model.name.clone(),
                reason: "No datasetGroup has all required measures".to_string(),
            })?
    };
    
    // Find all datasets that have the required dimensions (can participate in JOIN)
    let dimension_feasible: Vec<&GroupDataset> = target_group.datasets.iter()
        .filter(|dataset| has_all_dimensions(model, target_group, dataset, required_dimensions))
        .collect();
    
    if dimension_feasible.is_empty() {
        return Err(SelectError::NoFeasibleDataset {
            model: model.name.clone(),
            reason: "No dataset has all required dimensions".to_string(),
        });
    }
    
    // Sort datasets by attribute count (smallest/most aggregated first)
    let mut sorted_datasets: Vec<&GroupDataset> = dimension_feasible;
    sorted_datasets.sort_by_key(|t| t.attribute_count());
    
    // Assign measures to datasets using "first smallest wins" strategy
    let mut measure_assignments: HashMap<String, &GroupDataset> = HashMap::new();
    let mut datasets_used: HashSet<String> = HashSet::new();
    
    for measure_name in required_measures {
        // Find the smallest dataset that has this measure
        if let Some(dataset) = sorted_datasets.iter()
            .find(|t| t.has_measure(measure_name))
        {
            measure_assignments.insert(measure_name.clone(), *dataset);
            datasets_used.insert(dataset.dataset.clone());
        } else {
            return Err(SelectError::NoFeasibleDataset {
                model: model.name.clone(),
                reason: format!("No dataset has measure '{}'", measure_name),
            });
        }
    }
    
    // Build the result: group datasets by dataset, preserving smallest-first order
    let mut datasets_with_measures: Vec<DatasetWithMeasures> = Vec::new();
    
    for dataset in &sorted_datasets {
        if datasets_used.contains(&dataset.dataset) {
            let measures: Vec<String> = measure_assignments.iter()
                .filter(|(_, t)| t.dataset == dataset.dataset)
                .map(|(m, _)| m.clone())
                .collect();
            
            if !measures.is_empty() {
                datasets_with_measures.push(DatasetWithMeasures {
                    dataset,
                    measures,
                });
            }
        }
    }
    
    Ok(MultiDatasetSelection {
        group: target_group,
        datasets: datasets_with_measures,
    })
}

/// Check if a dataset has all required dimensions (for JOIN participation)
fn has_all_dimensions(
    model: &SemanticModel,
    group: &DatasetGroup,
    dataset: &GroupDataset,
    required_dimensions: &[String],
) -> bool {
    for dim_attr in required_dimensions {
        // Skip virtual _dataset dimension
        if dim_attr.starts_with("_dataset.") {
            continue;
        }
        
        let parts: Vec<&str> = dim_attr.split('.').collect();
        if parts.len() == 3 {
            // Three-part: datasetGroup.dimension.attribute
            let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            if tg_qualifier != group.name {
                continue; // Different datasetGroup, not required from this group
            }
            let two_part = format!("{}.{}", dim_name, attr_name);
            if !dataset_has_attribute(model, group, dataset, &two_part) {
                return false;
            }
        } else if parts.len() == 2 {
            // Two-part: dimension.attribute
            if !dataset_has_attribute(model, group, dataset, dim_attr) {
                return false;
            }
        }
    }
    true
}

/// Check if a dataset can serve a query with the given requirements
fn is_feasible(
    model: &SemanticModel,
    group: &DatasetGroup,
    dataset: &GroupDataset,
    required_dimensions: &[String],
    required_measures: &[String],
) -> bool {
    // Check all required dimension.attribute paths exist
    for dim_attr in required_dimensions {
        // Skip virtual _dataset dimension - it's available on all datasets
        // and shouldn't affect dataset selection
        if dim_attr.starts_with("_dataset.") {
            continue;
        }
        
        // Handle datasetGroup-qualified paths (e.g., "adwords.dates.year")
        let parts: Vec<&str> = dim_attr.split('.').collect();
        if parts.len() == 3 {
            let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            // If this dimension is qualified for a DIFFERENT datasetGroup, skip it
            // (it's not required from this group)
            if tg_qualifier != group.name {
                continue;
            }
            // Check if this dataset has the dimension.attribute
            let two_part = format!("{}.{}", dim_name, attr_name);
            if !dataset_has_attribute(model, group, dataset, &two_part) {
                return false;
            }
        } else {
            // Two-part path: check normally
            if !dataset_has_attribute(model, group, dataset, dim_attr) {
                return false;
            }
        }
    }
    
    // Check all required measures exist in the group and are available on this dataset
    for measure_name in required_measures {
        // Measure must be defined in the group
        if group.get_measure(measure_name).is_none() {
            return false;
        }
        // Dataset must support this measure
        if !dataset.has_measure(measure_name) {
            return false;
        }
    }
    
    true
}

/// Check if a dataset has access to a dimension.attribute path
fn dataset_has_attribute(
    model: &SemanticModel,
    group: &DatasetGroup,
    dataset: &GroupDataset,
    dim_attr_path: &str,
) -> bool {
    let parts: Vec<&str> = dim_attr_path.split('.').collect();
    if parts.len() != 2 {
        return false;
    }
    let (dim_name, attr_name) = (parts[0], parts[1]);
    
    // Check if dataset has this dimension
    let Some(dataset_attrs) = dataset.get_dimension_attributes(dim_name) else {
        return false;
    };
    
    // Check if the attribute is in the dataset's list
    if !dataset_attrs.iter().any(|a| a == attr_name) {
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
    model: &SemanticModel,
    required_dimensions: &[String],
    required_measures: &[String],
) -> String {
    let mut missing = Vec::new();
    
    for dim_attr in required_dimensions {
        // Skip virtual _dataset dimension - it's always available
        if dim_attr.starts_with("_dataset.") {
            continue;
        }
        
        // Handle datasetGroup-qualified paths (e.g., "adwords.dates.year")
        let parts: Vec<&str> = dim_attr.split('.').collect();
        let available_in_any = if parts.len() == 3 {
            let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
            let two_part = format!("{}.{}", dim_name, attr_name);
            // Only check the specified datasetGroup
            model.dataset_groups.iter()
                .filter(|group| group.name == tg_qualifier)
                .any(|group| {
                    group.datasets.iter().any(|dataset| {
                        dataset_has_attribute(model, group, dataset, &two_part)
                    })
                })
        } else {
            // Two-part path: check all datasetGroups
            model.dataset_groups.iter().any(|group| {
                group.datasets.iter().any(|dataset| {
                    dataset_has_attribute(model, group, dataset, dim_attr)
                })
            })
        };
        
        if !available_in_any {
            missing.push(format!("dimension '{}' not available in any dataset", dim_attr));
        }
    }
    
    for measure_name in required_measures {
        let available_in_any = model.dataset_groups.iter().any(|group| {
            group.get_measure(measure_name).is_some() &&
            group.datasets.iter().any(|dataset| dataset.has_measure(measure_name))
        });
        if !available_in_any {
            missing.push(format!("measure '{}' not available in any dataset", measure_name));
        }
    }
    
    if missing.is_empty() {
        "no single dataset has all required dimensions and measures".to_string()
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
    fn test_select_single_dataset() {
        let schema = load_test_schema();
        let model = schema.get_model("steelwheels").unwrap();
        
        let datasets = select_datasets(
            &schema,
            model,
            &["dates.year".to_string()],
            &["sales".to_string()],
        ).unwrap();
        
        assert_eq!(datasets.len(), 1);
    }
    
    #[test]
    fn test_select_missing_dimension() {
        let schema = load_test_schema();
        let model = schema.get_model("steelwheels").unwrap();
        
        let result = select_datasets(
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
        
        let result = select_datasets(
            &schema,
            model,
            &["dates.year".to_string()],
            &["nonexistent_measure".to_string()],
        );
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_ambiguous_measure_across_datasetgroups() {
        // marketing.yaml has both adwords and facebookads datasetGroups
        // Both have "clicks" and "impressions" measures
        let schema = load_marketing_schema();
        let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();
        
        // Query for "clicks" which exists in both datasetGroups
        let result = select_datasets(
            &schema,
            model,
            &["dates.date".to_string()],
            &["clicks".to_string()],
        );
        
        // Should error because multiple datasetGroups can serve this query
        assert!(result.is_err());
        match result.unwrap_err() {
            SelectError::AmbiguousDatasetGroup { dataset_groups, .. } => {
                assert_eq!(dataset_groups.len(), 2);
                assert!(dataset_groups.contains(&"adwords".to_string()));
                assert!(dataset_groups.contains(&"facebookads".to_string()));
            }
            other => panic!("Expected AmbiguousDatasetGroup error, got: {:?}", other),
        }
    }
    
    #[test]
    fn test_unique_measure_selects_correct_datasetgroup() {
        // marketing.yaml: "cost" only exists in adwords, "spend" only in facebookads
        let schema = load_marketing_schema();
        let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();
        
        // Query for "cost" which only exists in adwords
        let datasets = select_datasets(
            &schema,
            model,
            &["dates.date".to_string()],
            &["cost".to_string()],
        ).unwrap();
        
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].group.name, "adwords");
        
        // Query for "spend" which only exists in facebookads
        let datasets = select_datasets(
            &schema,
            model,
            &["dates.date".to_string()],
            &["spend".to_string()],
        ).unwrap();
        
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].group.name, "facebookads");
    }
}
