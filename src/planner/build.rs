//! Plan building logic

use std::collections::{HashMap, HashSet};
use crate::model::{MeasureExpr, ExprNode, ExprArg, LiteralValue, MetricExpr, MetricExprNode, MetricExprArg, CaseExpr, ConditionExpr, DataType, GroupTable, Dimension, TableGroupDimension, TableGroup, Schema, Model, Metric, Aggregation, Measure};
use crate::plan::{
    Aggregate, AggregateExpr, Column, Expr, Filter, Join, JoinType, 
    Literal, PlanNode, Scan, BinaryOperator, Project, ProjectExpr, Sort, SortKey, SortDirection, Union,
    VirtualTable, LiteralValue as PlanLiteralValue,
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

                    // Virtual dimensions don't have physical tables
                    let dim_table = dimension.table.as_ref()
                        .expect("Non-virtual dimension must have a table");
                    
                    let dim_scan = PlanNode::Scan(
                        Scan::new(dim_table)
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
    // Skip Meta attributes - they're constants, not columns to group by
    let group_by: Vec<Column> = resolved.row_attributes.iter()
        .chain(resolved.column_attributes.iter())
        .filter(|attr| !attr.is_meta())
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

    // Add Project node if there are metrics or meta attributes
    let has_meta_attrs = resolved.row_attributes.iter()
        .chain(resolved.column_attributes.iter())
        .any(|attr| attr.is_meta());
    
    if !resolved.metrics.is_empty() || has_meta_attrs {
        let mut projections = Vec::new();
        
        // Pass through attributes with semantic names
        // Row attributes first, then column attributes
        for attr in resolved.row_attributes.iter().chain(resolved.column_attributes.iter()) {
            let semantic_name = format!("{}.{}", attr.dimension_name(), attr.attribute_name());
            let expr = build_attribute_expr(attr, resolved.table, fact_alias);
            projections.push(ProjectExpr {
                expr,
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
    // Build dimension list from rows + columns
    let mut dimension_attrs: Vec<String> = Vec::new();
    if let Some(ref rows) = request.rows {
        dimension_attrs.extend(rows.clone());
    }
    if let Some(ref cols) = request.columns {
        dimension_attrs.extend(cols.clone());
    }
    
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
    
    // Extract qualified tableGroups from 3-part dimension paths (e.g., "adwords.dates.year")
    let qualified_groups: HashSet<&str> = dimension_attrs.iter()
        .filter_map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 { Some(parts[0]) } else { None }
        })
        .collect();
    
    let is_conformed = model.is_conformed_query(&dimension_attrs);
    
    if !cross_table_metrics.is_empty() {
        // Cross-tableGroup metric query path
        // For now, support single cross-tableGroup metric per query
        if cross_table_metrics.len() > 1 {
            return Err(PlanError::InvalidQuery(
                "Multiple cross-tableGroup metrics in a single query not yet supported".to_string()
            ));
        }
        
        let metric = cross_table_metrics[0];
        
        plan_cross_table_group_query(schema, model, metric, &dimension_attrs)
    } else if qualified_groups.len() > 1 {
        // Multi-tableGroup qualified dimensions - UNION across the specified tableGroups
        // e.g., "adwords.dates.year" + "facebookads.dates.year" → UNION with NULL projection
        plan_multi_tablegroup_query(schema, model, request, &dimension_attrs, &qualified_groups)
    } else if qualified_groups.len() == 1 {
        // Single tableGroup qualified - constrain to that tableGroup
        let target_group = *qualified_groups.iter().next().unwrap();
        plan_single_tablegroup_query(schema, model, request, &dimension_attrs, target_group)
    } else if is_conformed && model.table_groups.len() > 1 {
        // Conformed dimension query - UNION across all feasible tableGroups
        plan_conformed_query(schema, model, request, &dimension_attrs)
    } else {
        // Normal single-tableGroup query path
        
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
        let selected_tables = select_tables(schema, model, &dimension_attrs, &required_measures)
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
/// Meta attributes are included (constant values sort consistently)
fn build_sort_keys(resolved: &ResolvedQuery<'_>) -> Vec<SortKey> {
    let mut keys = Vec::new();
    
    // First: row attributes
    for attr in &resolved.row_attributes {
        keys.push(SortKey {
            column: format!("{}.{}", attr.dimension_name(), attr.attribute_name()),
            direction: SortDirection::Ascending,
        });
    }
    
    // Then: column attributes
    for attr in &resolved.column_attributes {
        keys.push(SortKey {
            column: format!("{}.{}", attr.dimension_name(), attr.attribute_name()),
            direction: SortDirection::Ascending,
        });
    }
    
    keys
}

/// Build a Column from an AttributeRef
/// 
/// Considers whether the table needs a join (based on key attribute inclusion) or has denormalized columns.
/// Panics if called with a Meta attribute (use build_attribute_expr instead).
fn build_column(attr: &AttributeRef<'_>, table: &GroupTable, fact_alias: &str) -> Column {
    match attr {
        AttributeRef::Degenerate { attribute, .. } => {
            // Degenerate dimension: column is directly on fact table
            Column::new(fact_alias, attribute.column_name())
        }
        AttributeRef::Joined { group_dim, dimension, attribute, .. } => {
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
        AttributeRef::Meta { .. } => {
            panic!("Meta attributes should use build_attribute_expr, not build_column")
        }
    }
}

/// Build an Expr from an AttributeRef
/// 
/// Returns a Column reference for regular attributes, or a Literal for Meta attributes.
fn build_attribute_expr(attr: &AttributeRef<'_>, table: &GroupTable, fact_alias: &str) -> Expr {
    match attr {
        AttributeRef::Meta { value, .. } => {
            // Meta attributes are constant literal values
            Expr::Literal(Literal::String(value.clone()))
        }
        _ => {
            // Regular attributes become column references
            Expr::Column(build_column(attr, table, fact_alias))
        }
    }
}

/// Build a filter expression from a ResolvedFilter
fn build_filter_expr(filter: &ResolvedFilter<'_>, fact_alias: &str) -> Expr {
    // For filters, we need to determine the correct column/value reference
    // Meta attributes are constant literals
    let base_expr = match &filter.attribute {
        AttributeRef::Degenerate { attribute, .. } => {
            Expr::Column(Column::new(fact_alias, attribute.column_name()))
        }
        AttributeRef::Joined { group_dim, dimension, attribute, .. } => {
            if group_dim.join.is_some() {
                let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                Expr::Column(Column::new(dim_alias, attribute.column_name()))
            } else {
                Expr::Column(Column::new(fact_alias, attribute.column_name()))
            }
        }
        AttributeRef::Meta { value, .. } => {
            // Meta attributes are constants - filtering on them will be evaluated statically
            Expr::Literal(Literal::String(value.clone()))
        }
    };
    let column_expr = base_expr;

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
        serde_json::Value::Null => Literal::Null("string".to_string()),
        serde_json::Value::Bool(b) => Literal::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Literal::Int(i)
            } else if let Some(f) = n.as_f64() {
                Literal::Float(f)
            } else {
                Literal::Null("f64".to_string())
            }
        }
        serde_json::Value::String(s) => Literal::String(s.clone()),
        // Arrays and objects become null (shouldn't happen in filters)
        _ => Literal::Null("string".to_string()),
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
    let left = args.get(0).map(convert_expr_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
    let right = args.get(1).map(convert_expr_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
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
    let left = args.get(0).map(convert_metric_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
    let right = args.get(1).map(convert_metric_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
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
    match attr {
        AttributeRef::Degenerate { attribute, .. } => {
            // Degenerate: column is directly on fact table
            let data_type = attribute.data_type().to_string();
            fact_columns
                .entry(attribute.column_name().to_string())
                .or_insert(data_type);
        }
        AttributeRef::Joined { group_dim, dimension, attribute, .. } => {
            let data_type = attribute.data_type().to_string();
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
        AttributeRef::Meta { .. } => {
            // Meta attributes are literal values, no columns to collect
        }
    }
}

// ============================================================================
// Cross-TableGroup Query Planning
// ============================================================================

/// Plan a query on conformed dimensions across multiple tableGroups
/// 
/// This is triggered when:
/// 1. All queried dimensions are marked as conformed in the model
/// 2. There are multiple tableGroups that could serve the query
/// 
/// The function builds a UNION plan across all tableGroups that can serve the query.
/// 
/// Special case: If the query contains ONLY virtual dimensions (like `_table.*`) and
/// NO metrics, we generate a VirtualTable (VALUES clause) instead of scanning tables.
fn plan_conformed_query(
    schema: &Schema,
    model: &Model,
    request: &QueryRequest,
    dimension_attrs: &[String],
) -> Result<PlanNode, PlanError> {
    use crate::selector::SelectedTable;
    
    // Get physical dimensions (exclude virtual dimensions like _table.*)
    let physical_dims: Vec<String> = dimension_attrs.iter()
        .filter(|d| {
            let parts: Vec<&str> = d.split('.').collect();
            if parts.len() != 2 {
                return true; // Keep malformed, let it fail later
            }
            // Check if dimension is virtual
            !model.get_dimension(parts[0])
                .map(|dim| dim.is_virtual())
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    
    // Get virtual dimensions
    let virtual_dims: Vec<String> = dimension_attrs.iter()
        .filter(|d| {
            let parts: Vec<&str> = d.split('.').collect();
            if parts.len() != 2 {
                return false;
            }
            model.get_dimension(parts[0])
                .map(|dim| dim.is_virtual())
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    
    // Get measure dependencies from metrics
    let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
    let required_measures: Vec<String> = metric_names
        .iter()
        .filter_map(|metric_name| {
            model.get_metric(metric_name).and_then(|m| {
                match &m.expr {
                    MetricExpr::MeasureRef(name) => Some(name.clone()),
                    MetricExpr::Structured(_) => None,
                }
            })
        })
        .collect();
    
    // Special case: Virtual-only query (no physical dimensions, no metrics)
    // Generate a VirtualTable instead of scanning tables
    if physical_dims.is_empty() && metric_names.is_empty() && !virtual_dims.is_empty() {
        return plan_virtual_only_query(model, &virtual_dims);
    }
    
    let mut branches: Vec<PlanNode> = Vec::new();
    
    // Build a branch for each tableGroup that can serve the query
    for table_group in &model.table_groups {
        // Find a table in this tableGroup that has all required dimensions and measures
        let feasible_table = table_group.tables.iter().find(|table| {
            // Check all required physical dimensions
            for dim_attr in &physical_dims {
                let parts: Vec<&str> = dim_attr.split('.').collect();
                if parts.len() != 2 {
                    return false;
                }
                let (dim_name, attr_name) = (parts[0], parts[1]);
                
                if let Some(attrs) = table.get_dimension_attributes(dim_name) {
                    if !attrs.iter().any(|a| a == attr_name) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            
            // Check all required measures exist in this tableGroup and table
            for measure_name in &required_measures {
                if table_group.get_measure(measure_name).is_none() {
                    return false;
                }
                if !table.has_measure(measure_name) {
                    return false;
                }
            }
            
            true
        });
        
        let Some(table) = feasible_table else {
            // This tableGroup can't serve the query - skip it
            continue;
        };
        
        // Create a SelectedTable for this branch
        let selected = SelectedTable {
            group: table_group,
            table,
        };
        
        // Resolve and plan this branch
        let resolved = resolve_query(schema, request, &selected)
            .map_err(|e| PlanError::InvalidQuery(format!(
                "Query resolution error for tableGroup '{}': {:?}", 
                table_group.name, e
            )))?;
        
        let branch = plan_query(&resolved)?;
        branches.push(branch);
    }
    
    if branches.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No tableGroup can serve this conformed dimension query".to_string()
        ));
    }
    
    // If only one tableGroup can serve the query, return it directly
    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }
    
    // Create UNION of all branches
    Ok(PlanNode::Union(Union { inputs: branches }))
}

/// Plan a query with dimensions qualified for multiple different tableGroups.
/// 
/// This handles queries like "adwords.dates.year + facebookads.dates.year" where
/// dimensions are explicitly scoped to specific tableGroups. The result is a UNION
/// where each branch projects values for its tableGroup and NULLs for columns
/// belonging to other tableGroups.
/// 
/// Key difference from conformed queries:
/// - Conformed: All branches project actual values for all dimensions
/// - Multi-TG qualified: Each branch projects NULLs for other TG's qualified dimensions
fn plan_multi_tablegroup_query(
    _schema: &Schema,
    model: &Model,
    request: &QueryRequest,
    dimension_attrs: &[String],
    qualified_groups: &HashSet<&str>,
) -> Result<PlanNode, PlanError> {
    let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
    
    let mut branches: Vec<PlanNode> = Vec::new();
    
    // Build a branch for each qualified tableGroup
    for table_group in &model.table_groups {
        // Only process tableGroups mentioned in the qualified dimensions
        if !qualified_groups.contains(table_group.name.as_str()) {
            continue;
        }
        
        // Find a feasible table in this tableGroup
        let feasible_table = find_feasible_table_for_qualified(
            model, table_group, dimension_attrs, &metric_names
        );
        
        let Some(table) = feasible_table else {
            return Err(PlanError::InvalidQuery(format!(
                "No table in tableGroup '{}' can serve the qualified dimension query",
                table_group.name
            )));
        };
        
        // Build the branch with NULL projection for other TG's qualified dimensions
        let branch = build_union_branch(
            model,
            table_group,
            table,
            dimension_attrs,
            &metric_names,
        )?;
        
        branches.push(branch);
    }
    
    if branches.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No tableGroup can serve this qualified dimension query".to_string()
        ));
    }
    
    // If only one tableGroup, return directly (no UNION needed)
    if branches.len() == 1 {
        return Ok(branches.into_iter().next().unwrap());
    }
    
    // Create UNION of all branches
    Ok(PlanNode::Union(Union { inputs: branches }))
}

/// Plan a query constrained to a single tableGroup (via qualified dimension).
/// 
/// This handles queries like "adwords.dates.year" where all qualified dimensions
/// target the same tableGroup.
fn plan_single_tablegroup_query(
    schema: &Schema,
    model: &Model,
    request: &QueryRequest,
    dimension_attrs: &[String],
    target_group: &str,
) -> Result<PlanNode, PlanError> {
    use crate::selector::SelectedTable;
    
    // Find the target tableGroup
    let table_group = model.table_groups.iter()
        .find(|tg| tg.name == target_group)
        .ok_or_else(|| PlanError::InvalidQuery(format!(
            "TableGroup '{}' not found in model", target_group
        )))?;
    
    let metric_names: Vec<String> = request.metrics.clone().unwrap_or_default();
    
    // Find a feasible table
    let feasible_table = find_feasible_table_for_qualified(
        model, table_group, dimension_attrs, &metric_names
    );
    
    let Some(table) = feasible_table else {
        return Err(PlanError::InvalidQuery(format!(
            "No table in tableGroup '{}' can serve the qualified dimension query",
            target_group
        )));
    };
    
    // Convert 3-part paths to 2-part for the resolver
    // e.g., "adwords.dates.year" → "dates.year"
    let normalized_dims: Vec<String> = dimension_attrs.iter()
        .map(|path| {
            let parts: Vec<&str> = path.split('.').collect();
            if parts.len() == 3 && parts[0] == target_group {
                format!("{}.{}", parts[1], parts[2])
            } else {
                path.clone()
            }
        })
        .collect();
    
    // Create a modified request with normalized dimensions
    let normalized_request = QueryRequest {
        model: request.model.clone(),
        dimensions: None,
        rows: Some(normalized_dims.iter()
            .filter(|d| request.rows.as_ref().map(|r| r.iter().any(|rd| {
                // Check if original path matches this normalized dim
                let parts: Vec<&str> = rd.split('.').collect();
                if parts.len() == 3 {
                    format!("{}.{}", parts[1], parts[2]) == **d
                } else {
                    rd == *d
                }
            })).unwrap_or(false))
            .cloned()
            .collect()),
        columns: request.columns.as_ref().map(|cols| {
            cols.iter()
                .map(|c| {
                    let parts: Vec<&str> = c.split('.').collect();
                    if parts.len() == 3 && parts[0] == target_group {
                        format!("{}.{}", parts[1], parts[2])
                    } else {
                        c.clone()
                    }
                })
                .collect()
        }),
        metrics: request.metrics.clone(),
        filter: request.filter.clone(),
    };
    
    // Use standard resolve + plan path
    let selected = SelectedTable { group: table_group, table };
    
    let resolved = resolve_query(schema, &normalized_request, &selected)
        .map_err(|e| PlanError::InvalidQuery(format!(
            "Query resolution error for tableGroup '{}': {:?}",
            target_group, e
        )))?;
    
    plan_query(&resolved)
}

/// Find a feasible table in a tableGroup for qualified dimension queries.
fn find_feasible_table_for_qualified<'a>(
    model: &Model,
    table_group: &'a TableGroup,
    dimension_attrs: &[String],
    metric_names: &[String],
) -> Option<&'a GroupTable> {
    // Get required measures from metrics
    let required_measures: Vec<String> = metric_names
        .iter()
        .filter_map(|metric_name| {
            model.get_metric(metric_name).and_then(|m| {
                match &m.expr {
                    MetricExpr::MeasureRef(name) => Some(name.clone()),
                    MetricExpr::Structured(_) => None,
                }
            })
        })
        .collect();
    
    table_group.tables.iter().find(|table| {
        // Check dimensions qualified for THIS tableGroup
        for dim_attr in dimension_attrs {
            let parts: Vec<&str> = dim_attr.split('.').collect();
            
            if parts.len() == 3 {
                // Three-part: tableGroup.dimension.attribute
                let (tg_qualifier, dim_name, attr_name) = (parts[0], parts[1], parts[2]);
                
                // Skip if this dimension is for a different tableGroup
                if tg_qualifier != table_group.name {
                    continue;
                }
                
                // Check if this table has the dimension.attribute
                if let Some(attrs) = table.get_dimension_attributes(dim_name) {
                    if !attrs.iter().any(|a| a == attr_name) {
                        return false;
                    }
                } else {
                    return false;
                }
            } else if parts.len() == 2 {
                // Two-part: dimension.attribute (conformed or virtual)
                let (dim_name, attr_name) = (parts[0], parts[1]);
                
                // Skip virtual dimensions
                if model.get_dimension(dim_name).map(|d| d.is_virtual()).unwrap_or(false) {
                    continue;
                }
                
                // Check if this table has the dimension.attribute
                if let Some(attrs) = table.get_dimension_attributes(dim_name) {
                    if !attrs.iter().any(|a| a == attr_name) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        
        // Check all required measures
        for measure_name in &required_measures {
            if table_group.get_measure(measure_name).is_none() {
                return false;
            }
            if !table.has_measure(measure_name) {
                return false;
            }
        }
        
        true
    })
}

/// Build a single branch for a multi-tableGroup UNION query.
/// 
/// This function creates a plan branch for one tableGroup that:
/// - Scans and aggregates data for dimensions belonging to this tableGroup
/// - Projects actual values for dimensions this TG owns
/// - Projects NULLs for qualified dimensions belonging to other tableGroups
/// - Projects virtual dimension values as literals
/// - Optionally aggregates metrics
fn build_union_branch(
    model: &Model,
    table_group: &TableGroup,
    table: &GroupTable,
    dimension_attrs: &[String],
    metric_names: &[String],
) -> Result<PlanNode, PlanError> {
    // Parse all dimension attributes
    let parsed_attrs: Vec<(String, ParsedDimensionAttr)> = dimension_attrs.iter()
        .map(|attr_path| (attr_path.clone(), ParsedDimensionAttr::parse(attr_path, model)))
        .collect();
    
    // Physical attrs that belong to this tableGroup (include in scan/group)
    // Multiple parsed paths might reference the same physical column (e.g., dates.date and adwords.dates.date)
    let physical_attrs: Vec<&(String, ParsedDimensionAttr)> = parsed_attrs.iter()
        .filter(|(_, parsed)| {
            !parsed.is_virtual() && parsed.belongs_to_table_group(&table_group.name)
        })
        .collect();
    
    // Build unique physical columns for scan/group (deduplicate by dim_name + attr_name)
    // Also build a mapping from (dim_name, attr_name) -> group_by index
    let mut unique_dim_attrs: Vec<(String, String)> = Vec::new();
    let mut dim_attr_to_group_idx: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();
    
    for (_, parsed) in &physical_attrs {
        let key = (parsed.dim_name().to_string(), parsed.attr_name().to_string());
        if !dim_attr_to_group_idx.contains_key(&key) {
            let idx = unique_dim_attrs.len();
            unique_dim_attrs.push(key.clone());
            dim_attr_to_group_idx.insert(key, idx);
        }
    }
    
    // Build the scan
    let fact_alias = "fact";
    let mut columns = Vec::new();
    let mut types = Vec::new();
    
    // Track which dimensions need joins (deduplicated)
    let mut joined_dimensions: HashSet<String> = HashSet::new();
    
    // Add dimension columns (for degenerate dimensions) - use unique_dim_attrs
    for (dim_name, attr_name) in &unique_dim_attrs {
        if let Some(group_dim) = table_group.get_dimension(dim_name) {
            if group_dim.is_degenerate() {
                if let Some(attr) = group_dim.get_attribute(attr_name) {
                    columns.push(attr.column_name().to_string());
                    types.push(attr.data_type.to_string());
                }
            }
        }
    }
    
    // Add measure columns for metrics
    let measures_to_aggregate: Vec<(&str, &Measure)> = metric_names.iter()
        .filter_map(|metric_name| {
            model.get_metric(metric_name).and_then(|m| {
                match &m.expr {
                    MetricExpr::MeasureRef(measure_name) => {
                        table_group.get_measure(measure_name)
                            .map(|measure| (metric_name.as_str(), measure))
                    }
                    MetricExpr::Structured(_) => None,
                }
            })
        })
        .collect();
    
    for (_, measure) in &measures_to_aggregate {
        if let MeasureExpr::Column(col) = &measure.expr {
            columns.push(col.clone());
            types.push(measure.data_type().to_string());
        }
    }
    
    let mut plan = PlanNode::Scan(
        Scan::new(&table.table)
            .with_alias(fact_alias)
            .with_columns(columns, types)
    );
    
    // Add joins for non-degenerate dimensions (using unique_dim_attrs to avoid duplicate joins)
    for (dim_name, _) in &unique_dim_attrs {
        // Skip if we've already joined this dimension
        if joined_dimensions.contains(dim_name) {
            continue;
        }
        
        if let Some(group_dim) = table_group.get_dimension(dim_name) {
            if let Some(join_spec) = &group_dim.join {
                if let Some(dimension) = model.get_dimension(dim_name) {
                    if needs_join_for_dimension(table, group_dim, dimension) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        
                        let dim_cols: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.column_name().to_string())
                            .collect();
                        let dim_types: Vec<String> = dimension.attributes.iter()
                            .map(|a| a.data_type.to_string())
                            .collect();
                        
                        let dim_table = dimension.table.as_ref()
                            .expect("Non-virtual dimension must have a table");
                        
                        let dim_scan = PlanNode::Scan(
                            Scan::new(dim_table)
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
                        
                        joined_dimensions.insert(dim_name.clone());
                    }
                }
            }
        }
    }
    
    // Build GROUP BY columns (using unique_dim_attrs)
    let group_by: Vec<Column> = unique_dim_attrs.iter()
        .filter_map(|(dim_name, attr_name)| {
            if let Some(group_dim) = table_group.get_dimension(dim_name) {
                if group_dim.is_degenerate() {
                    if let Some(attr) = group_dim.get_attribute(attr_name) {
                        return Some(Column::new(fact_alias, attr.column_name()));
                    }
                } else if let Some(dimension) = model.get_dimension(dim_name) {
                    if let Some(attr) = dimension.get_attribute(attr_name) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        return Some(Column::new(dim_alias, attr.column_name()));
                    }
                }
            }
            None
        })
        .collect();
    
    // Build aggregate expressions for metrics
    let aggregates: Vec<AggregateExpr> = measures_to_aggregate.iter()
        .map(|(metric_name, measure)| {
            AggregateExpr {
                func: measure.aggregation,
                expr: convert_measure_expr(&measure.expr),
                alias: metric_name.to_string(),
            }
        })
        .collect();
    
    // Only add aggregate node if we have GROUP BY or aggregates
    if !group_by.is_empty() || !aggregates.is_empty() {
        plan = PlanNode::Aggregate(Aggregate {
            input: Box::new(plan),
            group_by: group_by.clone(),
            aggregates,
        });
    }
    
    // Project to standardized output schema
    // All branches MUST have the same schema
    let mut projections = Vec::new();
    
    for (attr_path, parsed) in &parsed_attrs {
        let expr = if parsed.is_virtual() {
            // Virtual dimension: project as literal
            let dim_name = parsed.dim_name();
            let attr_name = parsed.attr_name();
            let value = get_virtual_attribute_value(model, table_group, dim_name, attr_name);
            match value {
                PlanLiteralValue::String(s) => Expr::Literal(Literal::String(s)),
                PlanLiteralValue::Int64(i) => Expr::Literal(Literal::Int(i)),
                PlanLiteralValue::Float64(f) => Expr::Literal(Literal::Float(f)),
                PlanLiteralValue::Bool(b) => Expr::Literal(Literal::Bool(b)),
                PlanLiteralValue::Null => Expr::Literal(Literal::Null("string".to_string())),
                _ => Expr::Literal(Literal::Null("string".to_string())),
            }
        } else if parsed.belongs_to_table_group(&table_group.name) {
            // Physical dimension that belongs to this tableGroup
            // Look up the GROUP BY index using the (dim_name, attr_name) mapping
            let key = (parsed.dim_name().to_string(), parsed.attr_name().to_string());
            if let Some(&idx) = dim_attr_to_group_idx.get(&key) {
                let col = group_by.get(idx).cloned()
                    .unwrap_or_else(|| Column::unqualified(attr_path));
                Expr::Column(col)
            } else {
                // Fallback - shouldn't happen if logic is correct
                let data_type = parsed.get_data_type(model);
                Expr::Literal(Literal::Null(data_type))
            }
        } else {
            // Qualified dimension for another tableGroup: project typed NULL
            let data_type = parsed.get_data_type(model);
            Expr::Literal(Literal::Null(data_type))
        };
        
        projections.push(ProjectExpr {
            expr,
            alias: attr_path.clone(),
        });
    }
    
    // Add metric projections
    for metric_name in metric_names {
        projections.push(ProjectExpr {
            expr: Expr::Column(Column::unqualified(metric_name)),
            alias: metric_name.clone(),
        });
    }
    
    Ok(PlanNode::Project(Project {
        input: Box::new(plan),
        expressions: projections,
    }))
}

/// Plan a virtual-only query that doesn't need table scans.
/// 
/// For queries that only request virtual dimension attributes (like `_table.tableGroup`)
/// and no metrics, we generate a VirtualTable with one row per tableGroup containing
/// the literal metadata values.
fn plan_virtual_only_query(
    model: &Model,
    virtual_dims: &[String],
) -> Result<PlanNode, PlanError> {
    // Parse virtual dimension attributes
    let attrs: Vec<(&str, &str)> = virtual_dims.iter()
        .filter_map(|d| {
            let parts: Vec<&str> = d.split('.').collect();
            if parts.len() == 2 {
                Some((parts[0], parts[1]))
            } else {
                None
            }
        })
        .collect();
    
    if attrs.is_empty() {
        return Err(PlanError::InvalidQuery(
            "No valid virtual dimension attributes in query".to_string()
        ));
    }
    
    // Build column names (semantic names like "dimension.attribute")
    let columns: Vec<String> = virtual_dims.iter().cloned().collect();
    
    // All virtual attributes are strings for now
    let column_types: Vec<String> = attrs.iter().map(|_| "string".to_string()).collect();
    
    // Build one row per tableGroup with the metadata values
    let mut rows: Vec<Vec<PlanLiteralValue>> = Vec::new();
    
    for table_group in &model.table_groups {
        let row: Vec<PlanLiteralValue> = attrs.iter()
            .map(|(dim_name, attr_name)| {
                get_virtual_attribute_value(model, table_group, dim_name, attr_name)
            })
            .collect();
        rows.push(row);
    }
    
    Ok(PlanNode::VirtualTable(VirtualTable {
        columns,
        column_types,
        rows,
    }))
}

/// Get the literal value for a virtual dimension attribute
fn get_virtual_attribute_value(
    model: &Model,
    table_group: &TableGroup,
    dim_name: &str,
    attr_name: &str,
) -> PlanLiteralValue {
    // Currently we only support the _table virtual dimension
    if dim_name == "_table" {
        match attr_name {
            "tableGroup" => PlanLiteralValue::String(table_group.name.clone()),
            "model" => PlanLiteralValue::String(model.name.clone()),
            "namespace" => model.namespace.as_ref()
                .map(|ns| PlanLiteralValue::String(ns.clone()))
                .unwrap_or(PlanLiteralValue::Null),
            // For "table" attribute, we don't have a specific table context here
            // In virtual-only queries, we just return the tableGroup name
            "table" => PlanLiteralValue::Null,
            _ => PlanLiteralValue::Null,
        }
    } else {
        PlanLiteralValue::Null
    }
}

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
    // Validate tableGroup-qualified dimensions
    for attr_path in dimension_attrs {
        let parts: Vec<&str> = attr_path.split('.').collect();
        if parts.len() == 3 {
            let tg_name = parts[0];
            if model.get_table_group(tg_name).is_none() {
                return Err(PlanError::InvalidQuery(
                    format!("TableGroup '{}' not found in qualified dimension '{}'", tg_name, attr_path)
                ));
            }
        }
    }
    
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

/// Parsed dimension attribute for cross-tableGroup queries
#[derive(Debug, Clone)]
enum ParsedDimensionAttr {
    /// Standard two-part: dimension.attribute
    Standard { dim_name: String, attr_name: String },
    /// TableGroup-qualified three-part: tableGroup.dimension.attribute
    Qualified { tg_name: String, dim_name: String, attr_name: String },
    /// Virtual dimension (like _table)
    Virtual { dim_name: String, attr_name: String },
}

impl ParsedDimensionAttr {
    fn parse(attr_path: &str, model: &Model) -> Self {
        let parts: Vec<&str> = attr_path.split('.').collect();
        
        match parts.len() {
            2 => {
                let (dim_name, attr_name) = (parts[0], parts[1]);
                // Check if this is a virtual dimension
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
                // Fallback - treat as standard
                ParsedDimensionAttr::Standard {
                    dim_name: attr_path.to_string(),
                    attr_name: String::new(),
                }
            }
        }
    }
    
    /// Check if this attribute belongs to the given tableGroup
    fn belongs_to_table_group(&self, tg_name: &str) -> bool {
        match self {
            ParsedDimensionAttr::Qualified { tg_name: qualified_tg, .. } => qualified_tg == tg_name,
            ParsedDimensionAttr::Standard { .. } => true, // Standard attrs belong to all
            ParsedDimensionAttr::Virtual { .. } => true,   // Virtual attrs belong to all
        }
    }
    
    fn is_virtual(&self) -> bool {
        matches!(self, ParsedDimensionAttr::Virtual { .. })
    }
    
    fn dim_name(&self) -> &str {
        match self {
            ParsedDimensionAttr::Standard { dim_name, .. } => dim_name,
            ParsedDimensionAttr::Qualified { dim_name, .. } => dim_name,
            ParsedDimensionAttr::Virtual { dim_name, .. } => dim_name,
        }
    }
    
    fn attr_name(&self) -> &str {
        match self {
            ParsedDimensionAttr::Standard { attr_name, .. } => attr_name,
            ParsedDimensionAttr::Qualified { attr_name, .. } => attr_name,
            ParsedDimensionAttr::Virtual { attr_name, .. } => attr_name,
        }
    }
    
    /// Get the data type of this dimension attribute from the model
    fn get_data_type(&self, model: &Model) -> String {
        // For virtual dimensions, return string
        if self.is_virtual() {
            return "string".to_string();
        }
        
        // Look up the dimension in the model
        if let Some(dimension) = model.get_dimension(self.dim_name()) {
            if let Some(attr) = dimension.get_attribute(self.attr_name()) {
                return attr.data_type.to_string();
            }
        }
        
        // Fallback to string
        "string".to_string()
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
    // Parse all dimension attributes
    let parsed_attrs: Vec<(String, ParsedDimensionAttr)> = dimension_attrs.iter()
        .map(|attr_path| (attr_path.clone(), ParsedDimensionAttr::parse(attr_path, model)))
        .collect();
    
    // Separate into categories based on this tableGroup
    // - Physical attrs that belong to this tableGroup (include in scan/group)
    // - Virtual attrs (project as literals)
    // - Qualified attrs for OTHER tableGroups (project as NULL)
    let physical_attrs: Vec<&(String, ParsedDimensionAttr)> = parsed_attrs.iter()
        .filter(|(_, parsed)| {
            !parsed.is_virtual() && parsed.belongs_to_table_group(&table_group.name)
        })
        .collect();
    
    // Build unique physical columns for scan/group (deduplicate by dim_name + attr_name)
    // Also build a mapping from (dim_name, attr_name) -> group_by index
    let mut unique_dim_attrs: Vec<(String, String)> = Vec::new();
    let mut dim_attr_to_group_idx: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();
    
    for (_, parsed) in &physical_attrs {
        let key = (parsed.dim_name().to_string(), parsed.attr_name().to_string());
        if !dim_attr_to_group_idx.contains_key(&key) {
            let idx = unique_dim_attrs.len();
            unique_dim_attrs.push(key.clone());
            dim_attr_to_group_idx.insert(key, idx);
        }
    }
    
    // Build the scan
    let fact_alias = "fact";
    let mut columns = Vec::new();
    let mut types = Vec::new();
    
    // Track which dimensions need joins (deduplicated)
    let mut joined_dimensions: HashSet<String> = HashSet::new();
    
    // Add dimension columns (for degenerate dimensions) - use unique_dim_attrs
    for (dim_name, attr_name) in &unique_dim_attrs {
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
    
    // Add joins for non-degenerate dimensions (using unique_dim_attrs to avoid duplicate joins)
    for (dim_name, _) in &unique_dim_attrs {
        // Skip if we've already joined this dimension
        if joined_dimensions.contains(dim_name) {
            continue;
        }
        
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
                        
                        // Virtual dimensions don't have physical tables
                        let dim_table = dimension.table.as_ref()
                            .expect("Non-virtual dimension must have a table");
                        
                        let dim_scan = PlanNode::Scan(
                            Scan::new(dim_table)
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
                        
                        joined_dimensions.insert(dim_name.clone());
                    }
                }
            }
        }
    }
    
    // Build GROUP BY columns (using unique_dim_attrs)
    let group_by: Vec<Column> = unique_dim_attrs.iter()
        .filter_map(|(dim_name, attr_name)| {
            if let Some(group_dim) = table_group.get_dimension(dim_name) {
                if group_dim.is_degenerate() {
                    if let Some(attr) = group_dim.get_attribute(attr_name) {
                        return Some(Column::new(fact_alias, attr.column_name()));
                    }
                } else if let Some(dimension) = model.get_dimension(dim_name) {
                    if let Some(attr) = dimension.get_attribute(attr_name) {
                        let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                        return Some(Column::new(dim_alias, attr.column_name()));
                    }
                }
            }
            None
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
    
    // Project to standardized output schema
    // All branches MUST have the same schema, so we iterate ALL dimension_attrs in order
    let mut projections = Vec::new();
    
    for (attr_path, parsed) in &parsed_attrs {
        let expr = if parsed.is_virtual() {
            // Virtual dimension: project as literal
            let dim_name = parsed.dim_name();
            let attr_name = parsed.attr_name();
            let value = get_virtual_attribute_value(model, table_group, dim_name, attr_name);
            match value {
                PlanLiteralValue::String(s) => Expr::Literal(Literal::String(s)),
                PlanLiteralValue::Int64(i) => Expr::Literal(Literal::Int(i)),
                PlanLiteralValue::Float64(f) => Expr::Literal(Literal::Float(f)),
                PlanLiteralValue::Bool(b) => Expr::Literal(Literal::Bool(b)),
                PlanLiteralValue::Null => Expr::Literal(Literal::Null("string".to_string())),
                _ => Expr::Literal(Literal::Null("string".to_string())),
            }
        } else if parsed.belongs_to_table_group(&table_group.name) {
            // Physical dimension that belongs to this tableGroup
            // Look up the GROUP BY index using the (dim_name, attr_name) mapping
            let key = (parsed.dim_name().to_string(), parsed.attr_name().to_string());
            if let Some(&idx) = dim_attr_to_group_idx.get(&key) {
                let col = group_by.get(idx).cloned()
                    .unwrap_or_else(|| Column::unqualified(attr_path));
                Expr::Column(col)
            } else {
                // Fallback - shouldn't happen if logic is correct
                let data_type = parsed.get_data_type(model);
                Expr::Literal(Literal::Null(data_type))
            }
        } else {
            // Qualified dimension for another tableGroup: project typed NULL
            let data_type = parsed.get_data_type(model);
            Expr::Literal(Literal::Null(data_type))
        };
        
        projections.push(ProjectExpr {
            expr,
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

    #[test]
    fn test_plan_with_meta_attributes() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "dates.year".to_string(),
                "_table.tableGroup".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // Should be: Sort(Project(Aggregate(...)))
        // Meta attributes should be in Project as literals, not in GROUP BY
        let proj = match plan {
            PlanNode::Sort(sort) => {
                assert_eq!(sort.sort_keys.len(), 2);
                assert_eq!(sort.sort_keys[0].column, "dates.year");
                assert_eq!(sort.sort_keys[1].column, "_table.tableGroup");
                match *sort.input {
                    PlanNode::Project(proj) => proj,
                    _ => panic!("Expected Project node inside Sort"),
                }
            }
            _ => panic!("Expected Sort node at top level"),
        };
        
        // Project should have 3 expressions: dates.year, _table.tableGroup, sales
        assert_eq!(proj.expressions.len(), 3);
        assert_eq!(proj.expressions[0].alias, "dates.year");
        assert_eq!(proj.expressions[1].alias, "_table.tableGroup");
        assert_eq!(proj.expressions[2].alias, "sales");
        
        // The _table.tableGroup should be a literal
        match &proj.expressions[1].expr {
            Expr::Literal(Literal::String(value)) => {
                assert_eq!(value, "orders");
            }
            other => panic!("Expected Literal(String) for _table.tableGroup, got {:?}", other),
        }
        
        // Check that GROUP BY only has 1 column (dates.year), not the meta attribute
        match proj.input.as_ref() {
            PlanNode::Aggregate(agg) => {
                assert_eq!(agg.group_by.len(), 1);
                assert_eq!(agg.group_by[0].name, "year_id");
            }
            _ => panic!("Expected Aggregate node"),
        }
    }

    #[test]
    fn test_plan_meta_only() {
        let schema = load_test_schema();
        let selected = get_first_selected(&schema, "steelwheels");
        let request = QueryRequest {
            model: "steelwheels".to_string(),
            dimensions: None,
            rows: Some(vec![
                "_table.model".to_string(),
                "_table.tableGroup".to_string(),
            ]),
            columns: None,
            metrics: Some(vec!["sales".to_string()]),
            filter: None,
        };

        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan = plan_query(&resolved).unwrap();

        // With only meta attributes (no real dimensions), GROUP BY should be empty
        // but we still have aggregation for the metric
        match plan {
            PlanNode::Sort(sort) => {
                match *sort.input {
                    PlanNode::Project(proj) => {
                        assert_eq!(proj.expressions.len(), 3);
                        
                        // Both should be literals
                        match &proj.expressions[0].expr {
                            Expr::Literal(Literal::String(v)) => assert_eq!(v, "steelwheels"),
                            _ => panic!("Expected literal for _table.model"),
                        }
                        match &proj.expressions[1].expr {
                            Expr::Literal(Literal::String(v)) => assert_eq!(v, "orders"),
                            _ => panic!("Expected literal for _table.tableGroup"),
                        }
                        
                        // GROUP BY should be empty since only meta attributes
                        match proj.input.as_ref() {
                            PlanNode::Aggregate(agg) => {
                                assert!(agg.group_by.is_empty());
                            }
                            _ => panic!("Expected Aggregate node"),
                        }
                    }
                    _ => panic!("Expected Project node"),
                }
            }
            _ => panic!("Expected Sort node"),
        }
    }
}
