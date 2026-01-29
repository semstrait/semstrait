//! Plan building logic

use std::collections::HashMap;
use crate::model::{MeasureExpr, ExprNode, ExprArg, LiteralValue, MetricExpr, MetricExprNode, MetricExprArg, CaseExpr, ConditionExpr, DataType, GroupTable, Dimension, TableGroupDimension, TableGroup, Schema, Model, Metric, Aggregation, Measure};
use crate::plan::{
    Aggregate, AggregateExpr, Column, Expr, Filter, Join, JoinType, 
    Literal, PlanNode, Scan, BinaryOperator, Project, ProjectExpr, Sort, SortKey, SortDirection, Union,
};
use crate::resolver::{ResolvedQuery, AttributeRef, ResolvedFilter, ResolvedDimension, resolve_query};
use crate::selector::select_tables;
use crate::query::QueryRequest;
use super::error::PlanError;

/// Determine if a table needs a join for a given dimension
/// 
/// New logic based on attribute inclusion:
/// - If the tableGroup dimension has no join spec → no join (degenerate dimension)
/// - If the table's attribute list includes the "key attribute" → needs join
/// - If the table's attribute list excludes the key attribute → denormalized, no join
/// 
/// The "key attribute" is the dimension attribute whose column matches the join's rightKey.
fn needs_join_for_dimension(
    table: &GroupTable,
    group_dim: &TableGroupDimension,
    dimension: &Dimension,
) -> bool {
    // No join spec = degenerate dimension, never needs join
    let Some(join) = &group_dim.join else {
        return false;
    };

    // Find the key attribute in the dimension (attribute whose column matches rightKey)
    let Some(key_attr) = dimension.key_attribute(&join.right_key) else {
        // Can't find key attribute - shouldn't happen with valid model
        return false;
    };

    // Check if table's attribute list for this dimension includes the key attribute
    table.has_dimension_attribute(&group_dim.name, &key_attr.name)
}

/// Build a logical plan from a resolved query
pub fn plan_query(resolved: &ResolvedQuery<'_>) -> Result<PlanNode, PlanError> {
    // Collect columns and types needed for each table
    let (fact_columns, fact_types, dimension_columns) = collect_required_columns(resolved);
    
    // Start with the fact table scan (from the selected table)
    let fact_table = &resolved.table.table;
    let fact_alias = "fact"; // GroupTable doesn't have alias, use default
    
    let mut plan: PlanNode = PlanNode::Scan(
        Scan::new(fact_table)
            .with_alias(fact_alias)
            .with_columns(fact_columns, fact_types)
    );

    // Add joins for non-degenerate dimensions
    for dim in &resolved.dimensions {
        // Only join if this is a Joined dimension (not degenerate) and needs a join
        if let ResolvedDimension::Joined { group_dim, dimension } = dim {
            // Check if join is needed based on attribute inclusion
            if needs_join_for_dimension(resolved.table, group_dim, dimension) {
                if let Some(join_spec) = &group_dim.join {
                    let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);

                    // Get columns and types for this dimension table
                    let (dim_cols, dim_types) = dimension_columns
                        .get(&dimension.name)
                        .cloned()
                        .unwrap_or_default();

                    let dim_scan = PlanNode::Scan(
                        Scan::new(&dimension.table)
                            .with_alias(dim_alias)
                            .with_columns(dim_cols, dim_types)
                    );

                    let left_key = Column::new(fact_alias, &join_spec.left_key);
                    let right_key = Column::new(
                        join_spec.right_alias.as_deref().unwrap_or(dim_alias),
                        &join_spec.right_key,
                    );

                    plan = PlanNode::Join(Join {
                        left: Box::new(plan),
                        right: Box::new(dim_scan),
                        join_type: JoinType::Left,
                        left_key,
                        right_key,
                    });
                }
            }
            // If table doesn't include key attribute, attributes are denormalized - no join needed
        }
        // Degenerate dimensions don't need joins - columns are on fact table
    }

    // Add filters
    if !resolved.filters.is_empty() {
        let filter_exprs: Vec<Expr> = resolved.filters
            .iter()
            .map(|f| build_filter_expr(f, fact_alias))
            .collect();

        let predicate = if filter_exprs.len() == 1 {
            filter_exprs.into_iter().next().unwrap()
        } else {
            Expr::And(filter_exprs)
        };

        plan = PlanNode::Filter(Filter {
            input: Box::new(plan),
            predicate,
        });
    }

    // Build GROUP BY columns from row and column attributes
    let group_by: Vec<Column> = resolved.row_attributes.iter()
        .chain(resolved.column_attributes.iter())
        .map(|attr| build_column(attr, resolved.table, fact_alias))
        .collect();

    // Build aggregate expressions from measures
    let aggregates: Vec<AggregateExpr> = resolved.measures
        .iter()
        .map(|measure| AggregateExpr {
            func: measure.aggregation,
            expr: convert_measure_expr(&measure.expr),
            alias: measure.name.clone(),
        })
        .collect();

    // Only add Aggregate node if we have something to aggregate
    if !group_by.is_empty() || !aggregates.is_empty() {
        plan = PlanNode::Aggregate(Aggregate {
            input: Box::new(plan),
            group_by: group_by.clone(),
            aggregates,
        });
    }

    // Add Project node if there are metrics
    if !resolved.metrics.is_empty() {
        let mut projections = Vec::new();
        
        // Pass through GROUP BY columns with semantic names
        // Row attributes first, then column attributes
        for attr in resolved.row_attributes.iter().chain(resolved.column_attributes.iter()) {
            let col = build_column(attr, resolved.table, fact_alias);
            let semantic_name = format!("{}.{}", attr.dimension_name(), attr.attribute().name);
            projections.push(ProjectExpr {
                expr: Expr::Column(col),
                alias: semantic_name,
            });
        }
        
        // Add metric calculations
        for metric in &resolved.metrics {
            projections.push(ProjectExpr {
                expr: convert_metric_expr(&metric.expr),
                alias: metric.name.clone(),
            });
        }
        
        plan = PlanNode::Project(Project {
            input: Box::new(plan),
            expressions: projections,
        });
    }

    // Add default sort by dimension columns (row attrs, then col attrs)
    let sort_keys = build_sort_keys(resolved);
    if !sort_keys.is_empty() {
        plan = PlanNode::Sort(Sort {
            input: Box::new(plan),
            sort_keys,
        });
    }

    Ok(plan)
}

/// Plan a semantic query, automatically handling both single-tableGroup and cross-tableGroup cases
/// 
/// This is the main entry point for query planning. It:
/// 1. Analyzes the requested metrics to detect cross-tableGroup metrics
/// 2. Routes to `plan_cross_table_group_query` for cross-tableGroup metrics
/// 3. Routes to normal select → resolve → plan flow for single-tableGroup queries
/// 
/// # Arguments
/// * `schema` - The schema containing models and dimensions
/// * `model` - The model to query
/// * `request` - The query request with dimensions, metrics, and filters
/// 
/// # Returns
/// A `PlanNode` that can be emitted to Substrait
pub fn plan_semantic_query(
    schema: &Schema,
    model: &Model,
    request: &QueryRequest,
) -> Result<PlanNode, PlanError> {
    // Check if any requested metric is cross-tableGroup
    let cross_table_metrics: Vec<&Metric> = request.metrics
        .as_ref()
        .map(|names| {
            names.iter()
                .filter_map(|name| model.get_metric(name))
                .filter(|m| m.is_cross_table_group())
                .collect()
        })
        .unwrap_or_default();
    
    if !cross_table_metrics.is_empty() {
        // Cross-tableGroup query path
        // For now, support single cross-tableGroup metric per query
        if cross_table_metrics.len() > 1 {
            return Err(PlanError::InvalidQuery(
                "Multiple cross-tableGroup metrics in a single query not yet supported".to_string()
            ));
        }
        
        let metric = cross_table_metrics[0];
        
        // Collect dimension attributes from rows and columns
        let mut dimension_attrs: Vec<String> = Vec::new();
        if let Some(ref rows) = request.rows {
            dimension_attrs.extend(rows.clone());
        }
        if let Some(ref cols) = request.columns {
            dimension_attrs.extend(cols.clone());
        }
        
        plan_cross_table_group_query(schema, model, metric, &dimension_attrs)
    } else {
        // Normal single-tableGroup query path
        
        // Build required dimensions from rows + columns
        let mut required_dims: Vec<String> = Vec::new();
        if let Some(ref rows) = request.rows {
            required_dims.extend(rows.clone());
        }
        if let Some(ref cols) = request.columns {
            required_dims.extend(cols.clone());
        }
        
        // Extract measure dependencies from metrics
        let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
        let required_measures: Vec<String> = metric_names
            .iter()
            .filter_map(|metric_name| {
                model.get_metric(metric_name).and_then(|m| {
                    // For pass-through metrics (expr is just a measure name), extract the measure
                    match &m.expr {
                        MetricExpr::MeasureRef(name) => Some(name.clone()),
                        MetricExpr::Structured(_) => None, // Complex metrics - resolver will handle
                    }
                })
            })
            .collect();
        
        // Select optimal table
        let selected_tables = select_tables(schema, model, &required_dims, &required_measures)
            .map_err(|e| PlanError::InvalidQuery(format!("Table selection error: {:?}", e)))?;
        
        let selected = selected_tables.into_iter().next()
            .ok_or_else(|| PlanError::InvalidQuery("No feasible table found for query".to_string()))?;
        
        // Resolve the query
        let resolved = resolve_query(schema, request, &selected)
            .map_err(|e| PlanError::InvalidQuery(format!("Query resolution error: {:?}", e)))?;
        
        // Build the plan
        plan_query(&resolved)
    }
}

/// Build sort keys from resolved query attributes (row attrs, then col attrs)
fn build_sort_keys(resolved: &ResolvedQuery<'_>) -> Vec<SortKey> {
    let mut keys = Vec::new();
    
    // First: row attributes
    for attr in &resolved.row_attributes {
        keys.push(SortKey {
            column: format!("{}.{}", attr.dimension_name(), attr.attribute().name),
            direction: SortDirection::Ascending,
        });
    }
    
    // Then: column attributes
    for attr in &resolved.column_attributes {
        keys.push(SortKey {
            column: format!("{}.{}", attr.dimension_name(), attr.attribute().name),
            direction: SortDirection::Ascending,
        });
    }
    
    keys
}

/// Build a Column from an AttributeRef
/// 
/// Considers whether the table needs a join (based on key attribute inclusion) or has denormalized columns
fn build_column(attr: &AttributeRef<'_>, table: &GroupTable, fact_alias: &str) -> Column {
    match attr {
        AttributeRef::Degenerate { attribute, .. } => {
            // Degenerate dimension: column is directly on fact table
            Column::new(fact_alias, attribute.column_name())
        }
        AttributeRef::Joined { group_dim, dimension, attribute } => {
            // Check if this is a join or denormalized using attribute-based detection
            if needs_join_for_dimension(table, group_dim, dimension) {
                // Table needs join - use joined dimension table
                let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                Column::new(dim_alias, attribute.column_name())
            } else {
                // Table doesn't need join - attribute is denormalized on fact table
                // Use the attribute name directly as the column name
                Column::new(fact_alias, &attribute.name)
            }
        }
    }
}

/// Build a filter expression from a ResolvedFilter
fn build_filter_expr(filter: &ResolvedFilter<'_>, fact_alias: &str) -> Expr {
    // For filters, we need to determine the correct column reference
    // This is simplified - in a full implementation we'd pass the table too
    let column = match &filter.attribute {
        AttributeRef::Degenerate { attribute, .. } => {
            Column::new(fact_alias, attribute.column_name())
        }
        AttributeRef::Joined { group_dim, dimension, attribute } => {
            if group_dim.join.is_some() {
                let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                Column::new(dim_alias, attribute.column_name())
            } else {
                Column::new(fact_alias, attribute.column_name())
            }
        }
    };
    let column_expr = Expr::Column(column);

    match filter.operator.as_str() {
        "in" => {
            let values = match &filter.value {
                serde_json::Value::Array(arr) => arr.iter().map(json_to_literal).collect(),
                v => vec![json_to_literal(v)],
            };
            Expr::In {
                expr: Box::new(column_expr),
                values,
            }
        }
        "eq" | "=" => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::Eq,
            right: Box::new(json_to_literal(&filter.value)),
        },
        "neq" | "!=" => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::NotEq,
            right: Box::new(json_to_literal(&filter.value)),
        },
        "lt" | "<" => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::Lt,
            right: Box::new(json_to_literal(&filter.value)),
        },
        "lte" | "<=" => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::LtEq,
            right: Box::new(json_to_literal(&filter.value)),
        },
        "gt" | ">" => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::Gt,
            right: Box::new(json_to_literal(&filter.value)),
        },
        "gte" | ">=" => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::GtEq,
            right: Box::new(json_to_literal(&filter.value)),
        },
        // Default to equality
        _ => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::Eq,
            right: Box::new(json_to_literal(&filter.value)),
        },
    }
}

/// Convert a JSON value to a Literal expression
fn json_to_literal(value: &serde_json::Value) -> Expr {
    let lit = match value {
        serde_json::Value::Null => Literal::Null,
        serde_json::Value::Bool(b) => Literal::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Literal::Int(i)
            } else if let Some(f) = n.as_f64() {
                Literal::Float(f)
            } else {
                Literal::Null
            }
        }
        serde_json::Value::String(s) => Literal::String(s.clone()),
        // Arrays and objects become null (shouldn't happen in filters)
        _ => Literal::Null,
    };
    Expr::Literal(lit)
}

/// Collect required columns for each table based on the resolved query
/// Returns (fact_columns, fact_types, dimension_columns_by_name with types)
fn collect_required_columns(
    resolved: &ResolvedQuery<'_>
) -> (Vec<String>, Vec<String>, HashMap<String, (Vec<String>, Vec<String>)>) {
    let mut fact_columns: HashMap<String, String> = HashMap::new(); // name -> type
    let mut dimension_columns: HashMap<String, HashMap<String, String>> = HashMap::new(); // dim_name -> (col_name -> type)

    // Collect join keys for fact and dimension tables (only for joined dimensions that need joins)
    for dim in &resolved.dimensions {
        if let ResolvedDimension::Joined { group_dim, dimension } = dim {
            // Check if join is needed based on attribute inclusion
            if needs_join_for_dimension(resolved.table, group_dim, dimension) {
                if let Some(join) = &group_dim.join {
                    // Right key is on dimension table - find its type from the key attribute
                    let right_type = dimension.key_attribute(&join.right_key)
                        .map(|a| a.data_type().to_string())
                        .unwrap_or_else(|| DataType::I32.to_string());
                    
                    // Left key is on fact table - use same type as right key (they must match for join)
                    fact_columns.entry(join.left_key.clone()).or_insert(right_type.clone());
                    
                    dimension_columns
                        .entry(dimension.name.clone())
                        .or_default()
                        .insert(join.right_key.clone(), right_type);
                }
            }
        }
        // Degenerate dimensions don't have join keys
    }

    // Collect columns from row attributes
    for attr in &resolved.row_attributes {
        add_attribute_column_with_type(attr, resolved.table, &mut fact_columns, &mut dimension_columns);
    }

    // Collect columns from column attributes
    for attr in &resolved.column_attributes {
        add_attribute_column_with_type(attr, resolved.table, &mut fact_columns, &mut dimension_columns);
    }

    // Collect columns from filters
    for filter in &resolved.filters {
        add_attribute_column_with_type(&filter.attribute, resolved.table, &mut fact_columns, &mut dimension_columns);
    }

    // Collect columns from measures
    // Use column type from table.columns or degenerate dimension attributes, fall back to measure type
    for measure in &resolved.measures {
        collect_measure_columns(&measure.expr, &measure.data_type(), resolved.table, resolved.table_group, &mut fact_columns);
    }

    // Convert to Vec for stable ordering
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

/// Convert a MeasureExpr to a plan Expr
fn convert_measure_expr(expr: &MeasureExpr) -> Expr {
    match expr {
        MeasureExpr::Column(name) => Expr::Sql(name.clone()),
        MeasureExpr::Structured(node) => convert_expr_node(node),
    }
}

/// Convert an ExprNode to a plan Expr
fn convert_expr_node(node: &ExprNode) -> Expr {
    match node {
        ExprNode::Column(name) => Expr::Sql(name.clone()),
        ExprNode::Literal(lit) => Expr::Literal(convert_literal(lit)),
        ExprNode::Add(args) => {
            let (left, right) = binary_args(args);
            Expr::Add(Box::new(left), Box::new(right))
        }
        ExprNode::Subtract(args) => {
            let (left, right) = binary_args(args);
            Expr::Subtract(Box::new(left), Box::new(right))
        }
        ExprNode::Multiply(args) => {
            let (left, right) = binary_args(args);
            Expr::Multiply(Box::new(left), Box::new(right))
        }
        ExprNode::Divide(args) => {
            let (left, right) = binary_args(args);
            Expr::Divide(Box::new(left), Box::new(right))
        }
        ExprNode::Case(case_expr) => convert_case_expr(case_expr),
    }
}

/// Convert a CaseExpr to a plan Expr
fn convert_case_expr(case: &CaseExpr) -> Expr {
    let when_then: Vec<(Expr, Expr)> = case.when
        .iter()
        .map(|w| {
            let condition = convert_condition_expr(&w.condition);
            let then = convert_expr_arg(&w.then);
            (condition, then)
        })
        .collect();
    
    let else_result = case.else_value
        .as_ref()
        .map(|e| Box::new(convert_expr_arg(e)));
    
    Expr::Case { when_then, else_result }
}

/// Convert a ConditionExpr to a plan Expr
fn convert_condition_expr(cond: &ConditionExpr) -> Expr {
    match cond {
        ConditionExpr::Eq(args) => {
            let (left, right) = binary_args(args);
            Expr::BinaryOp { left: Box::new(left), op: BinaryOperator::Eq, right: Box::new(right) }
        }
        ConditionExpr::Ne(args) => {
            let (left, right) = binary_args(args);
            Expr::BinaryOp { left: Box::new(left), op: BinaryOperator::NotEq, right: Box::new(right) }
        }
        ConditionExpr::Gt(args) => {
            let (left, right) = binary_args(args);
            Expr::BinaryOp { left: Box::new(left), op: BinaryOperator::Gt, right: Box::new(right) }
        }
        ConditionExpr::Gte(args) => {
            let (left, right) = binary_args(args);
            Expr::BinaryOp { left: Box::new(left), op: BinaryOperator::GtEq, right: Box::new(right) }
        }
        ConditionExpr::Lt(args) => {
            let (left, right) = binary_args(args);
            Expr::BinaryOp { left: Box::new(left), op: BinaryOperator::Lt, right: Box::new(right) }
        }
        ConditionExpr::Lte(args) => {
            let (left, right) = binary_args(args);
            Expr::BinaryOp { left: Box::new(left), op: BinaryOperator::LtEq, right: Box::new(right) }
        }
        ConditionExpr::And(conditions) => {
            let exprs: Vec<Expr> = conditions.iter().map(convert_condition_expr).collect();
            Expr::And(exprs)
        }
        ConditionExpr::Or(conditions) => {
            let exprs: Vec<Expr> = conditions.iter().map(convert_condition_expr).collect();
            Expr::Or(exprs)
        }
        ConditionExpr::IsNull(col) => {
            Expr::IsNull(Box::new(Expr::Sql(col.clone())))
        }
        ConditionExpr::IsNotNull(col) => {
            Expr::IsNotNull(Box::new(Expr::Sql(col.clone())))
        }
    }
}

/// Convert an ExprArg to a plan Expr
fn convert_expr_arg(arg: &ExprArg) -> Expr {
    match arg {
        ExprArg::LiteralInt(i) => Expr::Literal(Literal::Int(*i)),
        ExprArg::LiteralFloat(f) => Expr::Literal(Literal::Float(*f)),
        ExprArg::ColumnName(name) => Expr::Sql(name.clone()),
        ExprArg::Node(node) => convert_expr_node(node),
    }
}

/// Extract two arguments from a binary operation
fn binary_args(args: &[ExprArg]) -> (Expr, Expr) {
    let left = args.get(0).map(convert_expr_arg).unwrap_or(Expr::Literal(Literal::Null));
    let right = args.get(1).map(convert_expr_arg).unwrap_or(Expr::Literal(Literal::Null));
    (left, right)
}

/// Convert a LiteralValue to a plan Literal
fn convert_literal(lit: &LiteralValue) -> Literal {
    match lit {
        LiteralValue::Int(i) => Literal::Int(*i),
        LiteralValue::Float(f) => Literal::Float(*f),
        LiteralValue::String(s) => Literal::String(s.clone()),
        LiteralValue::Bool(b) => Literal::Bool(*b),
    }
}

/// Convert a MetricExpr to a plan Expr
/// Metric expressions reference aggregated measure columns (post-aggregation)
fn convert_metric_expr(expr: &MetricExpr) -> Expr {
    match expr {
        MetricExpr::MeasureRef(name) => {
            // Simple measure reference becomes a column reference
            Expr::Column(Column::unqualified(name))
        }
        MetricExpr::Structured(node) => convert_metric_node(node),
    }
}

/// Convert a MetricExprNode to a plan Expr
fn convert_metric_node(node: &MetricExprNode) -> Expr {
    match node {
        MetricExprNode::Measure(name) => {
            Expr::Column(Column::unqualified(name))
        }
        MetricExprNode::Literal(f) => Expr::Literal(Literal::Float(*f)),
        MetricExprNode::Add(args) => {
            let (left, right) = metric_binary_args(args);
            Expr::Add(Box::new(left), Box::new(right))
        }
        MetricExprNode::Subtract(args) => {
            let (left, right) = metric_binary_args(args);
            Expr::Subtract(Box::new(left), Box::new(right))
        }
        MetricExprNode::Multiply(args) => {
            let (left, right) = metric_binary_args(args);
            Expr::Multiply(Box::new(left), Box::new(right))
        }
        MetricExprNode::Divide(args) => {
            let (left, right) = metric_binary_args(args);
            Expr::Divide(Box::new(left), Box::new(right))
        }
        MetricExprNode::Case(_) => {
            // Cross-tableGroup metrics with CASE expressions should be resolved 
            // at a higher level (in plan_cross_table_group_query) before reaching here.
            // If we get here, it means the metric should have been handled specially.
            panic!("Metric CASE expressions should be resolved before planning. Use plan_cross_table_group_query for cross-tableGroup metrics.")
        }
    }
}

/// Convert a MetricExprArg to a plan Expr
fn convert_metric_arg(arg: &MetricExprArg) -> Expr {
    match arg {
        MetricExprArg::MeasureName(name) => {
            Expr::Column(Column::unqualified(name))
        }
        MetricExprArg::LiteralNumber(f) => {
            Expr::Literal(Literal::Float(*f))
        }
        MetricExprArg::Node(node) => convert_metric_node(node),
    }
}

/// Extract two arguments from a metric binary operation
fn metric_binary_args(args: &[MetricExprArg]) -> (Expr, Expr) {
    let left = args.get(0).map(convert_metric_arg).unwrap_or(Expr::Literal(Literal::Null));
    let right = args.get(1).map(convert_metric_arg).unwrap_or(Expr::Literal(Literal::Null));
    (left, right)
}

/// Collect column names referenced in a measure expression
/// Uses column type from table.columns if defined, otherwise falls back to measure's data_type
/// Look up the type of a column from:
/// 1. Explicit table.columns (if defined)
/// 2. Degenerate dimension attributes in the table group
/// 3. Fall back to the provided default type
fn lookup_column_type(name: &str, table: &GroupTable, table_group: &TableGroup, fallback_type: &DataType) -> String {
    // First, try explicit columns
    if let Some(col) = table.get_column(name) {
        return col.data_type().to_string();
    }
    
    // Second, check degenerate dimension attributes
    // Look through all degenerate dimensions in the table group
    for dim in &table_group.dimensions {
        if dim.is_degenerate() {
            if let Some(attrs) = &dim.attributes {
                for attr in attrs {
                    // Check if the column name matches the attribute's column (or name)
                    if attr.column_name() == name {
                        return attr.data_type().to_string();
                    }
                }
            }
        }
    }
    
    // Fall back to provided default type
    fallback_type.to_string()
}

fn collect_measure_columns(
    expr: &MeasureExpr, 
    fallback_type: &DataType, 
    table: &GroupTable,
    table_group: &TableGroup,
    columns: &mut HashMap<String, String>
) {
    match expr {
        MeasureExpr::Column(name) => {
            let col_type = lookup_column_type(name, table, table_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
        MeasureExpr::Structured(node) => collect_node_columns(node, fallback_type, table, table_group, columns),
    }
}

/// Collect column names from an expression node
fn collect_node_columns(
    node: &ExprNode, 
    fallback_type: &DataType, 
    table: &GroupTable,
    table_group: &TableGroup,
    columns: &mut HashMap<String, String>
) {
    match node {
        ExprNode::Column(name) => {
            let col_type = lookup_column_type(name, table, table_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
        ExprNode::Literal(_) => {}
        ExprNode::Add(args) | ExprNode::Subtract(args) | ExprNode::Multiply(args) | ExprNode::Divide(args) => {
            for arg in args {
                collect_arg_columns(arg, fallback_type, table, table_group, columns);
            }
        }
        ExprNode::Case(case_expr) => {
            // Collect columns from all WHEN branches
            for when_branch in &case_expr.when {
                collect_condition_columns(&when_branch.condition, fallback_type, table, table_group, columns);
                collect_arg_columns(&when_branch.then, fallback_type, table, table_group, columns);
            }
            // Collect columns from ELSE if present
            if let Some(else_val) = &case_expr.else_value {
                collect_arg_columns(else_val, fallback_type, table, table_group, columns);
            }
        }
    }
}

/// Collect column names from a condition expression
fn collect_condition_columns(
    cond: &ConditionExpr, 
    fallback_type: &DataType, 
    table: &GroupTable,
    table_group: &TableGroup,
    columns: &mut HashMap<String, String>
) {
    match cond {
        ConditionExpr::Eq(args) | ConditionExpr::Ne(args) | 
        ConditionExpr::Gt(args) | ConditionExpr::Gte(args) |
        ConditionExpr::Lt(args) | ConditionExpr::Lte(args) => {
            for arg in args {
                collect_arg_columns(arg, fallback_type, table, table_group, columns);
            }
        }
        ConditionExpr::And(conds) | ConditionExpr::Or(conds) => {
            for c in conds {
                collect_condition_columns(c, fallback_type, table, table_group, columns);
            }
        }
        ConditionExpr::IsNull(name) | ConditionExpr::IsNotNull(name) => {
            let col_type = lookup_column_type(name, table, table_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
    }
}

/// Collect column names from an expression argument
fn collect_arg_columns(
    arg: &ExprArg, 
    fallback_type: &DataType, 
    table: &GroupTable,
    table_group: &TableGroup,
    columns: &mut HashMap<String, String>
) {
    match arg {
        ExprArg::LiteralInt(_) | ExprArg::LiteralFloat(_) => {
            // Literals don't reference columns
        }
        ExprArg::ColumnName(name) => {
            let col_type = lookup_column_type(name, table, table_group, fallback_type);
            columns.entry(name.clone()).or_insert(col_type);
        }
        ExprArg::Node(node) => collect_node_columns(node, fallback_type, table, table_group, columns),
    }
}

/// Add column from an attribute reference to the appropriate collection
fn add_attribute_column_with_type(
    attr: &AttributeRef<'_>,
    table: &GroupTable,
    fact_columns: &mut HashMap<String, String>,
    dimension_columns: &mut HashMap<String, HashMap<String, String>>,
) {
    let data_type = attr.attribute().data_type().to_string();
    
    match attr {
        AttributeRef::Degenerate { attribute, .. } => {
            // Degenerate: column is directly on fact table
            fact_columns
                .entry(attribute.column_name().to_string())
                .or_insert(data_type);
        }
        AttributeRef::Joined { group_dim, dimension, attribute } => {
            // Check if join is needed based on attribute inclusion
            if needs_join_for_dimension(table, group_dim, dimension) {
                // Table needs join - column is in dimension table
                dimension_columns
                    .entry(dimension.name.clone())
                    .or_default()
                    .entry(attribute.column_name().to_string())
                    .or_insert(data_type);
            } else {
                // Table doesn't need join - attribute is denormalized on fact table
                fact_columns
                    .entry(attribute.name.clone())
                    .or_insert(data_type);
            }
        }
    }
}

// ============================================================================
// Cross-TableGroup Query Planning
// ============================================================================

/// A branch in a cross-tableGroup query
/// Represents one tableGroup's contribution to the union
#[derive(Debug)]
pub struct CrossTableGroupBranch<'a> {
    pub table_group: &'a TableGroup,
    pub measure: &'a Measure,
    pub table: &'a GroupTable,
}

/// Plan a cross-tableGroup query for a metric that spans multiple tableGroups
/// 
/// This generates a Union plan that:
/// 1. Queries each tableGroup for the appropriate measure
/// 2. Projects each branch to have the same schema (dimension columns + "value" column)
/// 3. Unions the results
/// 4. Re-aggregates by the dimension columns
/// 
/// # Arguments
/// * `schema` - The schema (unused, kept for API compatibility)
/// * `model` - The model containing the tableGroups and dimensions
/// * `metric` - The cross-tableGroup metric (must have tableGroup.name conditions)
/// * `dimension_attrs` - The dimension.attribute paths to GROUP BY (e.g., ["dates.date"])
pub fn plan_cross_table_group_query<'a>(
    _schema: &'a Schema,
    model: &'a Model,
    metric: &'a Metric,
    dimension_attrs: &[String],
) -> Result<PlanNode, PlanError> {
    // Get the tableGroup-to-measure mappings from the metric
    let mappings = metric.table_group_measures();
    if mappings.is_empty() {
        return Err(PlanError::InvalidQuery(
            format!("Metric '{}' is not a cross-tableGroup metric", metric.name)
        ));
    }

    // Build a branch for each tableGroup
    let mut branches: Vec<PlanNode> = Vec::new();
    
    for (tg_name, measure_name) in &mappings {
        // Get the tableGroup
        let table_group = model.get_table_group(tg_name)
            .ok_or_else(|| PlanError::InvalidQuery(
                format!("TableGroup '{}' not found", tg_name)
            ))?;
        
        // Get the measure from this tableGroup
        let measure = table_group.get_measure(measure_name)
            .ok_or_else(|| PlanError::InvalidQuery(
                format!("Measure '{}' not found in tableGroup '{}'", measure_name, tg_name)
            ))?;
        
        // Find a suitable table in this tableGroup
        // For now, pick the first table that has the measure
        let table = table_group.tables.iter()
            .find(|t| t.has_measure(measure_name))
            .ok_or_else(|| PlanError::InvalidQuery(
                format!("No table in tableGroup '{}' has measure '{}'", tg_name, measure_name)
            ))?;
        
        // Build a sub-plan for this branch
        let branch = build_cross_table_group_branch(
            model,
            table_group,
            table,
            measure,
            dimension_attrs,
            &metric.name,
        )?;
        
        branches.push(branch);
    }

    // If only one branch, return it directly (no union needed)
    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }

    // Create the Union of all branches
    let union = PlanNode::Union(Union { inputs: branches });

    // Re-aggregate the union by the dimension columns
    // This sums up the "value" column from each branch
    let group_by: Vec<Column> = dimension_attrs.iter()
        .map(|attr| Column::unqualified(attr))
        .collect();
    
    let aggregates = vec![
        AggregateExpr {
            func: Aggregation::Sum,
            expr: Expr::Column(Column::unqualified(&metric.name)),
            alias: metric.name.clone(),
        }
    ];

    let plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(union),
        group_by: group_by.clone(),
        aggregates,
    });

    // Add sort by dimension columns
    let sort_keys: Vec<SortKey> = dimension_attrs.iter()
        .map(|attr| SortKey {
            column: attr.clone(),
            direction: SortDirection::Ascending,
        })
        .collect();

    if !sort_keys.is_empty() {
        Ok(PlanNode::Sort(Sort {
            input: Box::new(plan),
            sort_keys,
        }))
    } else {
        Ok(plan)
    }
}

/// Build a single branch of a cross-tableGroup query
fn build_cross_table_group_branch(
    model: &Model,
    table_group: &TableGroup,
    table: &GroupTable,
    measure: &Measure,
    dimension_attrs: &[String],
    output_alias: &str,
) -> Result<PlanNode, PlanError> {
    // Build the scan
    let fact_alias = "fact";
    let mut columns = Vec::new();
    let mut types = Vec::new();
    
    // Add dimension columns (for degenerate dimensions)
    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        if parts.len() != 2 {
            continue;
        }
        let (dim_name, attr_name) = (parts[0], parts[1]);
        
        if let Some(group_dim) = table_group.get_dimension(dim_name) {
            if group_dim.is_degenerate() {
                if let Some(attr) = group_dim.get_attribute(attr_name) {
                    columns.push(attr.column_name().to_string());
                    types.push(attr.data_type.to_string());
                }
            }
        }
    }
    
    // Add measure column
    if let MeasureExpr::Column(col) = &measure.expr {
        columns.push(col.clone());
        types.push(measure.data_type().to_string());
    }
    
    let mut plan = PlanNode::Scan(
        Scan::new(&table.table)
            .with_alias(fact_alias)
            .with_columns(columns, types)
    );
    
    // Add joins for non-degenerate dimensions
    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        if parts.len() != 2 {
            continue;
        }
        let (dim_name, _attr_name) = (parts[0], parts[1]);
        
        if let Some(group_dim) = table_group.get_dimension(dim_name) {
            if let Some(join_spec) = &group_dim.join {
                if let Some(dimension) = model.get_dimension(dim_name) {
                    if needs_join_for_dimension(table, group_dim, dimension) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        
                        // Build dimension column list
                        let dim_cols: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.column_name().to_string())
                            .collect();
                        let dim_types: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.data_type.to_string())
                            .collect();
                        
                        let dim_scan = PlanNode::Scan(
                            Scan::new(&dimension.table)
                                .with_alias(dim_alias)
                                .with_columns(dim_cols, dim_types)
                        );
                        
                        let left_key = Column::new(fact_alias, &join_spec.left_key);
                        let right_key = Column::new(
                            join_spec.right_alias.as_deref().unwrap_or(dim_alias),
                            &join_spec.right_key,
                        );
                        
                        plan = PlanNode::Join(Join {
                            left: Box::new(plan),
                            right: Box::new(dim_scan),
                            join_type: JoinType::Left,
                            left_key,
                            right_key,
                        });
                    }
                }
            }
        }
    }
    
    // Build GROUP BY columns
    let group_by: Vec<Column> = dimension_attrs.iter()
        .map(|attr_path| {
            let parts: Vec<&str> = attr_path.split('.').collect();
            if parts.len() != 2 {
                return Column::unqualified(attr_path);
            }
            let (dim_name, attr_name) = (parts[0], parts[1]);
            
            if let Some(group_dim) = table_group.get_dimension(dim_name) {
                if group_dim.is_degenerate() {
                    if let Some(attr) = group_dim.get_attribute(attr_name) {
                        return Column::new(fact_alias, attr.column_name());
                    }
                } else if let Some(dimension) = model.get_dimension(dim_name) {
                    if let Some(attr) = dimension.get_attribute(attr_name) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        return Column::new(dim_alias, attr.column_name());
                    }
                }
            }
            Column::unqualified(attr_path)
        })
        .collect();
    
    // Build aggregate expression from measure
    let aggregates = vec![
        AggregateExpr {
            func: measure.aggregation,
            expr: convert_measure_expr(&measure.expr),
            alias: output_alias.to_string(),
        }
    ];
    
    plan = PlanNode::Aggregate(Aggregate {
        input: Box::new(plan),
        group_by: group_by.clone(),
        aggregates,
    });
    
    // Project to standardized output schema: dimension columns + metric value
    let mut projections = Vec::new();
    
    // Add dimension columns with standardized names
    for (i, attr_path) in dimension_attrs.iter().enumerate() {
        projections.push(ProjectExpr {
            expr: Expr::Column(group_by[i].clone()),
            alias: attr_path.clone(),
        });
    }
    
    // Add the measure value with the metric name
    projections.push(ProjectExpr {
        expr: Expr::Column(Column::unqualified(output_alias)),
        alias: output_alias.to_string(),
    });
    
    Ok(PlanNode::Project(Project {
        input: Box::new(plan),
        expressions: projections,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Schema;
    use crate::query::QueryRequest;
    use crate::resolver::resolve_query;
    use crate::selector::{select_tables, SelectedTable};

    fn load_test_schema() -> Schema {
        Schema::from_file("test_data/steelwheels.yaml").unwrap()
    }
    
    /// Helper to get the first selected table from the model
    fn get_first_selected<'a>(schema: &'a Schema, model_name: &str) -> SelectedTable<'a> {
        let model = schema.get_model(model_name).unwrap();
        select_tables(schema, model, &[], &[]).unwrap().into_iter().next().unwrap()
    }

    #[test]
    fn test_plan_simple_query() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["dates.year".to_string()]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Should be: Sort(Project(Aggregate(Join(Scan(fact), Scan(dates)))))
        let proj = match plan {
            PlanNode::Sort(sort) => {
                assert_eq!(sort.sort_keys.len(), 1);
                assert_eq!(sort.sort_keys[0].column, "dates.year");
                match *sort.input {
                    PlanNode::Project(proj) => proj,
                    _ => panic!("Expected Project node inside Sort"),
                }
            }
            _ => panic!("Expected Sort node at top level"),
        };
        
        // Project should have dimensions + metrics
        assert_eq!(proj.expressions.len(), 2);
        assert_eq!(proj.expressions[0].alias, "dates.year");
        assert_eq!(proj.expressions[1].alias, "sales");
        
        match proj.input.as_ref() {
            PlanNode::Aggregate(agg) => {
                assert_eq!(agg.group_by.len(), 1);
                assert_eq!(agg.aggregates.len(), 1);
                assert_eq!(agg.aggregates[0].alias, "sales");
                
                match agg.input.as_ref() {
                    PlanNode::Join(join) => {
                        assert_eq!(join.join_type, JoinType::Left);
                    }
                    _ => panic!("Expected Join node"),
                }
            }
            _ => panic!("Expected Aggregate node"),
        }
    }

    #[test]
    fn test_plan_metrics_only() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: None,
            columns: None,
            metrics: Some(vec!["sales".to_string(), "quantity".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Project(Aggregate(...)) since we have metrics
        match plan {
            PlanNode::Project(proj) => {
                assert_eq!(proj.expressions.len(), 2);
                match proj.input.as_ref() {
                    PlanNode::Aggregate(agg) => {
                        assert!(agg.group_by.is_empty());
                        assert_eq!(agg.aggregates.len(), 2);
                    }
                    _ => panic!("Expected Aggregate node"),
                }
            }
            _ => panic!("Expected Project node"),
        }
    }

    #[test]
    fn test_plan_structured_expression() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: None,
            columns: None,
            metrics: Some(vec!["revenue".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        match plan {
            PlanNode::Project(proj) => {
                assert_eq!(proj.expressions.len(), 1);
                assert_eq!(proj.expressions[0].alias, "revenue");
                
                match proj.input.as_ref() {
                    PlanNode::Aggregate(agg) => {
                        assert_eq!(agg.aggregates.len(), 1);
                        assert_eq!(agg.aggregates[0].alias, "revenue");
                        // The expression should be a Multiply
                        match &agg.aggregates[0].expr {
                            Expr::Multiply(_, _) => {}
                            other => panic!("Expected Multiply expression, got {:?}", other),
                        }
                    }
                    _ => panic!("Expected Aggregate node"),
                }
            }
            _ => panic!("Expected Project node"),
        }
    }

    #[test]
    fn test_plan_metric() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["dates.year".to_string()]),
            columns: None,
            metrics: Some(vec!["avg_unit_price".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Should be: Sort(Project(Aggregate(...)))
        let proj = match plan {
            PlanNode::Sort(sort) => {
                assert_eq!(sort.sort_keys.len(), 1);
                assert_eq!(sort.sort_keys[0].column, "dates.year");
                match *sort.input {
                    PlanNode::Project(proj) => proj,
                    _ => panic!("Expected Project node inside Sort"),
                }
            }
            _ => panic!("Expected Sort node at top level for metric query"),
        };
        
        // Project should have: dates.year, avg_unit_price (metric calculation)
        assert_eq!(proj.expressions.len(), 2);
        assert_eq!(proj.expressions[0].alias, "dates.year");
        assert_eq!(proj.expressions[1].alias, "avg_unit_price");
        
        // Metric expression should be a Divide
        match &proj.expressions[1].expr {
            Expr::Divide(_, _) => {}
            other => panic!("Expected Divide expression for metric, got {:?}", other),
        }
        
        // Input should be Aggregate with the underlying measures
        match proj.input.as_ref() {
            PlanNode::Aggregate(agg) => {
                // Should have sales and quantity measures (dependencies of the metric)
                assert_eq!(agg.aggregates.len(), 2);
                let aliases: Vec<&str> = agg.aggregates.iter().map(|a| a.alias.as_str()).collect();
                assert!(aliases.contains(&"sales"));
                assert!(aliases.contains(&"quantity"));
            }
            _ => panic!("Expected Aggregate node as input to Project"),
        }
    }

    #[test]
    fn test_plan_multiple_metrics() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: None,
            columns: None,
            metrics: Some(vec!["sales".to_string(), "avg_unit_price".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Should be: Project(Aggregate(...)) - no Sort since no dimensions
        match plan {
            PlanNode::Project(proj) => {
                // Project should have: sales, avg_unit_price
                assert_eq!(proj.expressions.len(), 2);
                assert_eq!(proj.expressions[0].alias, "sales");
                assert_eq!(proj.expressions[1].alias, "avg_unit_price");
            }
            _ => panic!("Expected Project node at top level"),
        }
    }

    #[test]
    fn test_plan_degenerate_dimension() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec!["flags.is_premium".to_string()]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Should be: Sort(Project(Aggregate(Scan(fact)))) - NO join for degenerate dimension
        let proj = match plan {
            PlanNode::Sort(sort) => {
                assert_eq!(sort.sort_keys.len(), 1);
                assert_eq!(sort.sort_keys[0].column, "flags.is_premium");
                match *sort.input {
                    PlanNode::Project(proj) => proj,
                    _ => panic!("Expected Project node inside Sort"),
                }
            }
            _ => panic!("Expected Sort node at top level"),
        };

        match proj.input.as_ref() {
            PlanNode::Aggregate(agg) => {
                // Aggregate should have 1 group by column
                assert_eq!(agg.group_by.len(), 1);
                // The column should be from the fact table (degenerate dimension)
                assert_eq!(agg.group_by[0].table, "fact");
                assert_eq!(agg.group_by[0].name, "is_premium_order"); // Uses the column name from attribute

                // Input should be just a Scan (no Join for degenerate dimension)
                match agg.input.as_ref() {
                    PlanNode::Scan(scan) => {
                        assert_eq!(scan.table, "steelwheels.orderfact");
                    }
                    _ => panic!("Expected Scan node (no join for degenerate dimension)"),
                }
            }
            _ => panic!("Expected Aggregate node"),
        }
    }

    #[test]
    fn test_plan_mixed_dimensions() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "dates.year".to_string(),
                "flags.status".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Should be: Sort(Project(Aggregate(Join(Scan(fact), Scan(dates)))))
        // Note: only one join for dates, flags is degenerate
        let proj = match plan {
            PlanNode::Sort(sort) => {
                assert_eq!(sort.sort_keys.len(), 2);
                match *sort.input {
                    PlanNode::Project(proj) => proj,
                    _ => panic!("Expected Project node inside Sort"),
                }
            }
            _ => panic!("Expected Sort node at top level"),
        };

        match proj.input.as_ref() {
            PlanNode::Aggregate(agg) => {
                // Should have 2 group by columns
                assert_eq!(agg.group_by.len(), 2);

                // Input should be a Join (for dates dimension only)
                match agg.input.as_ref() {
                    PlanNode::Join(join) => {
                        assert_eq!(join.join_type, JoinType::Left);
                        // Left side is fact table, right side is dates dimension
                    }
                    _ => panic!("Expected Join node (for dates dimension)"),
                }
            }
            _ => panic!("Expected Aggregate node"),
        }
    }

    #[test]
    fn test_plan_cross_table_group_query() {
        let schema = Schema::from_file("test_data/marketing.yaml").unwrap();
        let model = schema.get_model("-ObDoDFVQGxxCGa5vw_Z").unwrap();
        let metric = model.get_metric("fun-cost").unwrap();
        
        // Verify this is a cross-tableGroup metric
        assert!(metric.is_cross_table_group());
        
        // Plan a cross-tableGroup query for dates.date
        let plan = plan_cross_table_group_query(
            &schema,
            model,
            metric,
            &["dates.date".to_string()],
        ).unwrap();
        
        // Should be: Sort(Aggregate(Union([Project(Aggregate(Scan)), Project(Aggregate(Scan))])))
        // The outer aggregate re-aggregates the union, the inner aggregates are per-tableGroup
        match plan {
            PlanNode::Sort(sort) => {
                assert_eq!(sort.sort_keys.len(), 1);
                assert_eq!(sort.sort_keys[0].column, "dates.date");
                
                match *sort.input {
                    PlanNode::Aggregate(agg) => {
                        // The re-aggregation should GROUP BY dates.date and SUM the metric
                        assert_eq!(agg.group_by.len(), 1);
                        assert_eq!(agg.aggregates.len(), 1);
                        assert_eq!(agg.aggregates[0].alias, "fun-cost");
                        
                        // Input should be a Union
                        match *agg.input {
                            PlanNode::Union(union) => {
                                // Should have 2 branches (adwords and facebookads)
                                assert_eq!(union.inputs.len(), 2);
                                
                                // Each branch should be a Project
                                for branch in &union.inputs {
                                    match branch {
                                        PlanNode::Project(proj) => {
                                            // Should project dates.date and fun-cost
                                            assert_eq!(proj.expressions.len(), 2);
                                        }
                                        _ => panic!("Expected Project node in union branch"),
                                    }
                                }
                            }
                            _ => panic!("Expected Union node as input to aggregate"),
                        }
                    }
                    _ => panic!("Expected Aggregate node inside Sort"),
                }
            }
            _ => panic!("Expected Sort node at top level"),
        }
    }
}
