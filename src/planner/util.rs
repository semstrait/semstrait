//! Shared utility types and helpers for the planner

use std::collections::HashMap;
use crate::semantic_model::{MeasureExpr, ExprNode, ExprArg, ConditionExpr, DataType, Dataset, DatasetGroup, DatasetGroupDimension, Dimension, SemanticModel};
use crate::plan::{Column, Expr, Literal, LiteralValue as PlanLiteralValue};
use crate::resolver::{AttributeRef, ResolvedDimension, ResolvedQuery};

/// Determine if a table needs a join for a given dimension
///
/// - If the datasetGroup dimension has no join spec -> no join (degenerate dimension)
/// - If the table's attribute list includes the "key attribute" -> needs join
/// - If the table's attribute list excludes the key attribute -> denormalized, no join
pub fn needs_join_for_dimension(
    table: &Dataset,
    group_dim: &DatasetGroupDimension,
    dimension: &Dimension,
) -> bool {
    let Some(join) = &group_dim.join else {
        return false;
    };
    let Some(key_attr) = dimension.key_attribute(&join.right_key) else {
        return false;
    };
    table.has_dimension_attribute(&group_dim.name, &key_attr.name)
}

/// Build a Column from an AttributeRef.
/// Panics if called with a Meta attribute (use build_attribute_expr instead).
pub fn build_column(attr: &AttributeRef<'_>, table: &Dataset, fact_alias: &str) -> Column {
    match attr {
        AttributeRef::Degenerate { attribute, .. } => {
            Column::new(fact_alias, attribute.column_name())
        }
        AttributeRef::Joined { group_dim, dimension, attribute, .. } => {
            if needs_join_for_dimension(table, group_dim, dimension) {
                let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                Column::new(dim_alias, attribute.column_name())
            } else {
                Column::new(fact_alias, &attribute.name)
            }
        }
        AttributeRef::Meta { .. } => {
            panic!("Meta attributes should use build_attribute_expr, not build_column")
        }
    }
}

/// Build an Expr from an AttributeRef.
/// Returns a Column reference for regular attributes, or a Literal for Meta attributes.
pub fn build_attribute_expr(attr: &AttributeRef<'_>, table: &Dataset, fact_alias: &str) -> Expr {
    match attr {
        AttributeRef::Meta { value, .. } => {
            if value.is_empty() {
                Expr::Literal(Literal::Null("string".to_string()))
            } else {
                Expr::Literal(Literal::String(value.clone()))
            }
        }
        _ => {
            Expr::Column(build_column(attr, table, fact_alias))
        }
    }
}

/// Build sort keys from resolved query attributes (row attrs, then col attrs).
pub fn build_sort_keys(resolved: &ResolvedQuery<'_>) -> Vec<crate::plan::SortKey> {
    let mut keys = Vec::new();
    for attr in &resolved.row_attributes {
        keys.push(crate::plan::SortKey {
            column: format!("{}.{}", attr.dimension_name(), attr.attribute_name()),
            direction: crate::plan::SortDirection::Ascending,
        });
    }
    for attr in &resolved.column_attributes {
        keys.push(crate::plan::SortKey {
            column: format!("{}.{}", attr.dimension_name(), attr.attribute_name()),
            direction: crate::plan::SortDirection::Ascending,
        });
    }
    keys
}

/// Collect required columns for each table based on the resolved query.
/// Returns (fact_columns, fact_types, dimension_columns_by_name with types).
pub fn collect_required_columns(
    resolved: &ResolvedQuery<'_>
) -> (Vec<String>, Vec<String>, HashMap<String, (Vec<String>, Vec<String>)>) {
    let mut fact_columns: HashMap<String, String> = HashMap::new();
    let mut dimension_columns: HashMap<String, HashMap<String, String>> = HashMap::new();

    for dim in &resolved.dimensions {
        if let ResolvedDimension::Joined { group_dim, dimension } = dim {
            if needs_join_for_dimension(resolved.dataset, group_dim, dimension) {
                if let Some(join) = &group_dim.join {
                    let right_type = dimension.key_attribute(&join.right_key)
                        .map(|a| a.data_type().to_string())
                        .unwrap_or_else(|| DataType::I32.to_string());
                    fact_columns.entry(join.left_key.clone()).or_insert(right_type.clone());
                    dimension_columns
                        .entry(dimension.name.clone())
                        .or_default()
                        .insert(join.right_key.clone(), right_type);
                }
            }
        }
    }

    for attr in &resolved.row_attributes {
        add_attribute_column_with_type(attr, resolved.dataset, &mut fact_columns, &mut dimension_columns);
    }
    for attr in &resolved.column_attributes {
        add_attribute_column_with_type(attr, resolved.dataset, &mut fact_columns, &mut dimension_columns);
    }
    for filter in &resolved.filters {
        add_attribute_column_with_type(&filter.attribute, resolved.dataset, &mut fact_columns, &mut dimension_columns);
    }
    for measure in &resolved.measures {
        collect_measure_columns(&measure.expr, &measure.data_type(), resolved.dataset, resolved.dataset_group, &mut fact_columns);
    }

    let (fact_cols, fact_types): (Vec<String>, Vec<String>) = fact_columns.into_iter().unzip();
    let dim_cols: HashMap<String, (Vec<String>, Vec<String>)> = dimension_columns
        .into_iter()
        .map(|(k, v)| {
            let (cols, types): (Vec<String>, Vec<String>) = v.into_iter().unzip();
            (k, (cols, types))
        })
        .collect();

    (fact_cols, fact_types, dim_cols)
}

/// Add column from an attribute reference to the appropriate collection.
fn add_attribute_column_with_type(
    attr: &AttributeRef<'_>,
    table: &Dataset,
    fact_columns: &mut HashMap<String, String>,
    dimension_columns: &mut HashMap<String, HashMap<String, String>>,
) {
    match attr {
        AttributeRef::Degenerate { attribute, .. } => {
            let data_type = attribute.data_type().to_string();
            fact_columns
                .entry(attribute.column_name().to_string())
                .or_insert(data_type);
        }
        AttributeRef::Joined { group_dim, dimension, attribute, .. } => {
            let data_type = attribute.data_type().to_string();
            if needs_join_for_dimension(table, group_dim, dimension) {
                dimension_columns
                    .entry(dimension.name.clone())
                    .or_default()
                    .entry(attribute.column_name().to_string())
                    .or_insert(data_type);
            } else {
                fact_columns
                    .entry(attribute.name.clone())
                    .or_insert(data_type);
            }
        }
        AttributeRef::Meta { .. } => {}
    }
}

/// Look up the type of a column from explicit columns, degenerate dimension attributes, or fallback.
pub fn lookup_column_type(name: &str, table: &Dataset, dataset_group: &DatasetGroup, fallback_type: &DataType) -> String {
    if let Some(col) = table.get_column(name) {
        return col.data_type().to_string();
    }
    for dim in &dataset_group.dimensions {
        if dim.is_degenerate() {
            if let Some(attrs) = &dim.attributes {
                for attr in attrs {
                    if attr.column_name() == name {
                        return attr.data_type().to_string();
                    }
                }
            }
        }
    }
    fallback_type.to_string()
}

pub fn collect_measure_columns(
    expr: &MeasureExpr,
    fallback_type: &DataType,
    table: &Dataset,
    dataset_group: &DatasetGroup,
    columns: &mut HashMap<String, String>,
) {
    match expr {
        MeasureExpr::Column(name) => {
            let col_type = lookup_column_type(name, table, dataset_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
        MeasureExpr::Structured(node) => collect_node_columns(node, fallback_type, table, dataset_group, columns),
    }
}

fn collect_node_columns(
    node: &ExprNode,
    fallback_type: &DataType,
    table: &Dataset,
    dataset_group: &DatasetGroup,
    columns: &mut HashMap<String, String>,
) {
    match node {
        ExprNode::Column(name) => {
            let col_type = lookup_column_type(name, table, dataset_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
        ExprNode::Literal(_) => {}
        ExprNode::Add(args) | ExprNode::Subtract(args) | ExprNode::Multiply(args) | ExprNode::Divide(args) => {
            for arg in args {
                collect_arg_columns(arg, fallback_type, table, dataset_group, columns);
            }
        }
        ExprNode::Case(case_expr) => {
            for when_branch in &case_expr.when {
                collect_condition_columns(&when_branch.condition, fallback_type, table, dataset_group, columns);
                collect_arg_columns(&when_branch.then, fallback_type, table, dataset_group, columns);
            }
            if let Some(else_val) = &case_expr.else_value {
                collect_arg_columns(else_val, fallback_type, table, dataset_group, columns);
            }
        }
    }
}

fn collect_condition_columns(
    cond: &ConditionExpr,
    fallback_type: &DataType,
    table: &Dataset,
    dataset_group: &DatasetGroup,
    columns: &mut HashMap<String, String>,
) {
    match cond {
        ConditionExpr::Eq(args) | ConditionExpr::Ne(args) |
        ConditionExpr::Gt(args) | ConditionExpr::Gte(args) |
        ConditionExpr::Lt(args) | ConditionExpr::Lte(args) => {
            for arg in args {
                collect_arg_columns(arg, fallback_type, table, dataset_group, columns);
            }
        }
        ConditionExpr::And(conds) | ConditionExpr::Or(conds) => {
            for c in conds {
                collect_condition_columns(c, fallback_type, table, dataset_group, columns);
            }
        }
        ConditionExpr::IsNull(name) | ConditionExpr::IsNotNull(name) => {
            let col_type = lookup_column_type(name, table, dataset_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
    }
}

fn collect_arg_columns(
    arg: &ExprArg,
    fallback_type: &DataType,
    table: &Dataset,
    dataset_group: &DatasetGroup,
    columns: &mut HashMap<String, String>,
) {
    match arg {
        ExprArg::LiteralInt(_) | ExprArg::LiteralFloat(_) => {}
        ExprArg::ColumnName(name) => {
            let col_type = lookup_column_type(name, table, dataset_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
        ExprArg::Node(node) => collect_node_columns(node, fallback_type, table, dataset_group, columns),
    }
}

/// Get the literal value for a virtual dimension attribute.
pub fn get_virtual_attribute_value(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    dim_name: &str,
    attr_name: &str,
) -> PlanLiteralValue {
    get_virtual_attribute_value_with_dataset(model, dataset_group, None, dim_name, attr_name)
}

pub fn get_virtual_attribute_value_with_dataset(
    model: &SemanticModel,
    dataset_group: &DatasetGroup,
    dataset: Option<&Dataset>,
    dim_name: &str,
    attr_name: &str,
) -> PlanLiteralValue {
    if dim_name == "_dataset" {
        match attr_name {
            "datasetGroup" => PlanLiteralValue::String(dataset_group.name.clone()),
            "model" => PlanLiteralValue::String(model.name.clone()),
            "namespace" => model.namespace.as_ref()
                .map(|ns| PlanLiteralValue::String(ns.clone()))
                .unwrap_or(PlanLiteralValue::Null),
            "dataset" => dataset
                .map(|d| PlanLiteralValue::String(d.name.clone()))
                .unwrap_or(PlanLiteralValue::Null),
            "partition" => dataset
                .and_then(|d| d.partition.as_ref())
                .map(|p| PlanLiteralValue::String(p.clone()))
                .unwrap_or(PlanLiteralValue::Null),
            _ => PlanLiteralValue::Null,
        }
    } else {
        PlanLiteralValue::Null
    }
}

/// Parsed dimension attribute for cross-tableGroup queries.
#[derive(Debug, Clone)]
pub enum ParsedDimensionAttr {
    Standard { dim_name: String, attr_name: String },
    Qualified { tg_name: String, dim_name: String, attr_name: String },
    Virtual { dim_name: String, attr_name: String },
}

impl ParsedDimensionAttr {
    pub fn parse(attr_path: &str, model: &SemanticModel) -> Self {
        let parts: Vec<&str> = attr_path.split('.').collect();
        match parts.len() {
            2 => {
                let (dim_name, attr_name) = (parts[0], parts[1]);
                if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                    ParsedDimensionAttr::Virtual {
                        dim_name: dim_name.to_string(),
                        attr_name: attr_name.to_string(),
                    }
                } else {
                    ParsedDimensionAttr::Standard {
                        dim_name: dim_name.to_string(),
                        attr_name: attr_name.to_string(),
                    }
                }
            }
            3 => {
                let (tg_name, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
                ParsedDimensionAttr::Qualified {
                    tg_name: tg_name.to_string(),
                    dim_name: dim_name.to_string(),
                    attr_name: attr_name.to_string(),
                }
            }
            _ => {
                ParsedDimensionAttr::Standard {
                    dim_name: attr_path.to_string(),
                    attr_name: String::new(),
                }
            }
        }
    }

    pub fn belongs_to_dataset_group(&self, tg_name: &str) -> bool {
        match self {
            ParsedDimensionAttr::Qualified { tg_name: qualified_tg, .. } => qualified_tg == tg_name,
            ParsedDimensionAttr::Standard { .. } => true,
            ParsedDimensionAttr::Virtual { .. } => true,
        }
    }

    pub fn is_virtual(&self) -> bool {
        matches!(self, ParsedDimensionAttr::Virtual { .. })
    }

    pub fn dim_name(&self) -> &str {
        match self {
            ParsedDimensionAttr::Standard { dim_name, .. } => dim_name,
            ParsedDimensionAttr::Qualified { dim_name, .. } => dim_name,
            ParsedDimensionAttr::Virtual { dim_name, .. } => dim_name,
        }
    }

    pub fn attr_name(&self) -> &str {
        match self {
            ParsedDimensionAttr::Standard { attr_name, .. } => attr_name,
            ParsedDimensionAttr::Qualified { attr_name, .. } => attr_name,
            ParsedDimensionAttr::Virtual { attr_name, .. } => attr_name,
        }
    }

    pub fn get_data_type(&self, model: &SemanticModel) -> String {
        if self.is_virtual() {
            return "string".to_string();
        }
        if let Some(dimension) = model.get_dimension(self.dim_name()) {
            if let Some(attr) = dimension.get_attribute(self.attr_name()) {
                return attr.data_type.to_string();
            }
        }
        "string".to_string()
    }
}

/// Get the physical column name for a dimension attribute.
pub fn get_dimension_column_name(
    dataset_group: &DatasetGroup,
    dim_name: &str,
    attr_name: &str,
) -> String {
    if let Some(group_dim) = dataset_group.get_dimension(dim_name) {
        if group_dim.is_degenerate() {
            if let Some(attr) = group_dim.get_attribute(attr_name) {
                return attr.column_name().to_string();
            }
        }
    }
    attr_name.to_string()
}

/// Extract physical dimension pairs from dimension paths, skipping virtual dimensions.
pub fn extract_physical_dims(dimension_attrs: &[String], model: &SemanticModel) -> Vec<(String, String)> {
    dimension_attrs.iter()
        .filter_map(|attr_path| {
            let parts: Vec<&str> = attr_path.split('.').collect();
            match parts.len() {
                2 => {
                    let dim_name = parts[0];
                    if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                        return None;
                    }
                    Some((dim_name.to_string(), parts[1].to_string()))
                }
                3 => {
                    let dim_name = parts[1];
                    if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                        return None;
                    }
                    Some((dim_name.to_string(), parts[2].to_string()))
                }
                _ => None,
            }
        })
        .collect()
}
