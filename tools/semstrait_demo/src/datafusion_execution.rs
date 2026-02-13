use datafusion::prelude::*;
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::{col, Expr as DFExpr, LogicalPlanBuilder};
use datafusion::common::Column as DFColumn;
use datafusion::functions_aggregate::expr_fn::*;
use semstrait::plan::{PlanNode, Expr, AggregateExpr, Column, BinaryOperator};
use semstrait::semantic_model::Aggregation;
use std::collections::HashMap;
use futures::future::BoxFuture;
use futures::FutureExt;

/// Execute a semstrait PlanNode using DataFusion
pub async fn execute_plan_node(
    ctx: &SessionContext,
    plan_node: &PlanNode,
    table_paths: &HashMap<String, String>,
) -> anyhow::Result<DataFrame> {
    let logical_plan = plan_node_to_logical_plan(ctx, plan_node, table_paths).await?;
    Ok(ctx.execute_logical_plan(logical_plan).await?)
}

/// Convert semstrait PlanNode to DataFusion LogicalPlan
fn plan_node_to_logical_plan<'a>(
    ctx: &'a SessionContext,
    plan_node: &'a PlanNode,
    table_paths: &'a HashMap<String, String>,
) -> BoxFuture<'a, anyhow::Result<datafusion::logical_expr::LogicalPlan>> {
    async move {
        match plan_node {
        PlanNode::Scan(scan) => {
            // Register the parquet file as a table
            let table_path = table_paths.get(&scan.table)
                .ok_or_else(|| anyhow::anyhow!("Table {} not found in paths", scan.table))?;

            let df = ctx.read_parquet(table_path, Default::default()).await?;
            let mut builder = LogicalPlanBuilder::from(df.logical_plan().clone());

            // Apply column selection if specified
            if !scan.columns.is_empty() {
                let projections: Vec<DFExpr> = scan.columns.iter()
                    .map(|col_name| col(col_name))
                    .collect();
                builder = builder.project(projections)?;
            }

            Ok(builder.build()?)
        }
        PlanNode::Project(project) => {
            let input_plan = plan_node_to_logical_plan(ctx, &project.input, table_paths).await?;
            let mut builder = LogicalPlanBuilder::from(input_plan);

            let projections: Vec<DFExpr> = project.expressions.iter()
                .map(|expr| {
                    let df_expr = expr_to_df_expr(&expr.expr)?;
                    Ok::<DFExpr, anyhow::Error>(df_expr.alias(&expr.alias))
                })
                .collect::<Result<Vec<_>, _>>()?;

            builder = builder.project(projections)?;
            Ok(builder.build()?)
        }
        PlanNode::Filter(filter) => {
            let input_plan = plan_node_to_logical_plan(ctx, &filter.input, table_paths).await?;
            let mut builder = LogicalPlanBuilder::from(input_plan);

            let predicate = expr_to_df_expr(&filter.predicate)?;
            builder = builder.filter(predicate)?;

            Ok(builder.build()?)
        }
        PlanNode::Aggregate(agg) => {
            let input_plan = plan_node_to_logical_plan(ctx, &agg.input, table_paths).await?;
            let mut builder = LogicalPlanBuilder::from(input_plan);

            // Group by expressions
            let group_exprs: Vec<DFExpr> = agg.group_by.iter()
                .map(|col| col_to_df_expr(col))
                .collect();

            // Aggregate expressions
            let agg_exprs: Vec<DFExpr> = agg.aggregates.iter()
                .map(|agg_expr| aggregate_expr_to_df_expr(agg_expr))
                .collect::<Result<Vec<_>, _>>()?;

            builder = builder.aggregate(group_exprs, agg_exprs)?;
            Ok(builder.build()?)
        }
        PlanNode::Union(union) => {
            if union.inputs.is_empty() {
                return Err(anyhow::anyhow!("Union with no inputs"));
            }

            let mut input_plans: Vec<datafusion::logical_expr::LogicalPlan> = Vec::new();
            for input in &union.inputs {
                input_plans.push(plan_node_to_logical_plan(ctx, input, table_paths).await?);
            }

            // Start with first input
            let mut builder = LogicalPlanBuilder::from(input_plans[0].clone());

            // Union with remaining inputs
            for input_plan in input_plans.into_iter().skip(1) {
                builder = builder.union(input_plan)?;
            }

            Ok(builder.build()?)
        }
        PlanNode::Sort(sort) => {
            let input_plan = plan_node_to_logical_plan(ctx, &sort.input, table_paths).await?;
            let mut builder = LogicalPlanBuilder::from(input_plan);

            use datafusion::logical_expr::SortExpr;
            let sort_exprs: Vec<SortExpr> = sort.sort_keys.iter()
                .map(|sort_key| {
                    let col_expr = col(&sort_key.column);
                    match sort_key.direction {
                        semstrait::plan::SortDirection::Ascending => SortExpr::new(col_expr, true, false),
                        semstrait::plan::SortDirection::Descending => SortExpr::new(col_expr, false, false),
                    }
                })
                .collect();

            builder = builder.sort(sort_exprs)?;
            Ok(builder.build()?)
        }
        PlanNode::Join(join) => {
            let left_plan = plan_node_to_logical_plan(ctx, &join.left, table_paths).await?;
            let right_plan = plan_node_to_logical_plan(ctx, &join.right, table_paths).await?;

            let left_col_expr = col_to_df_expr(&join.left_key);
            let right_col_expr = col_to_df_expr(&join.right_key);

            // Extract column references from expressions
            let left_col = match &left_col_expr {
                DFExpr::Column(col) => col.clone(),
                _ => return Err(anyhow::anyhow!("Join key must be a column reference")),
            };
            let right_col = match &right_col_expr {
                DFExpr::Column(col) => col.clone(),
                _ => return Err(anyhow::anyhow!("Join key must be a column reference")),
            };

            let join_type = match join.join_type {
                semstrait::plan::JoinType::Inner => datafusion::logical_expr::JoinType::Inner,
                semstrait::plan::JoinType::Left => datafusion::logical_expr::JoinType::Left,
                semstrait::plan::JoinType::Right => datafusion::logical_expr::JoinType::Right,
                semstrait::plan::JoinType::Full => datafusion::logical_expr::JoinType::Full,
            };

            let builder = LogicalPlanBuilder::from(left_plan)
                .join(right_plan, join_type, (vec![left_col], vec![right_col]), None)?;

            Ok(builder.build()?)
        }
        PlanNode::VirtualTable(_vt) => {
            // Virtual tables are complex to implement with DataFusion - for now, return an error
            // This would require creating a VALUES expression or similar
            Err(anyhow::anyhow!("VirtualTable execution not yet implemented"))
        }
    }
    }.boxed()
}

/// Convert semstrait Expr to DataFusion Expr
fn expr_to_df_expr(expr: &Expr) -> anyhow::Result<DFExpr> {
    match expr {
        Expr::Column(col) => Ok(col_to_df_expr(col)),
        Expr::Literal(lit) => Ok(literal_to_df_expr(lit)),
        Expr::BinaryOp { left, op, right } => {
            let left_expr = expr_to_df_expr(left)?;
            let right_expr = expr_to_df_expr(right)?;
            let operator = binary_op_to_df_op(op)?;
            use datafusion::logical_expr::BinaryExpr;
            Ok(DFExpr::BinaryExpr(BinaryExpr::new(
                Box::new(left_expr),
                operator,
                Box::new(right_expr),
            )))
        }
        Expr::Add(left, right) => {
            let left_expr = expr_to_df_expr(left)?;
            let right_expr = expr_to_df_expr(right)?;
            Ok(left_expr + right_expr)
        }
        Expr::Subtract(left, right) => {
            let left_expr = expr_to_df_expr(left)?;
            let right_expr = expr_to_df_expr(right)?;
            Ok(left_expr - right_expr)
        }
        Expr::Multiply(left, right) => {
            let left_expr = expr_to_df_expr(left)?;
            let right_expr = expr_to_df_expr(right)?;
            Ok(left_expr * right_expr)
        }
        Expr::Divide(left, right) => {
            let left_expr = expr_to_df_expr(left)?;
            let right_expr = expr_to_df_expr(right)?;
            Ok(left_expr / right_expr)
        }
        Expr::And(exprs) => {
            let mut result = None;
            for expr in exprs {
                let df_expr = expr_to_df_expr(expr)?;
                match result {
                    None => result = Some(df_expr),
                    Some(prev) => result = Some(prev.and(df_expr)),
                }
            }
            result.ok_or_else(|| anyhow::anyhow!("Empty AND expression"))
        }
        Expr::Or(exprs) => {
            let mut result = None;
            for expr in exprs {
                let df_expr = expr_to_df_expr(expr)?;
                match result {
                    None => result = Some(df_expr),
                    Some(prev) => result = Some(prev.or(df_expr)),
                }
            }
            result.ok_or_else(|| anyhow::anyhow!("Empty OR expression"))
        }
        Expr::IsNull(expr) => {
            let df_expr = expr_to_df_expr(expr)?;
            Ok(df_expr.is_null())
        }
        Expr::IsNotNull(expr) => {
            let df_expr = expr_to_df_expr(expr)?;
            Ok(df_expr.is_not_null())
        }
        Expr::Case { when_then, else_result } => {
            let mut cases = Vec::new();
            for (condition, result) in when_then {
                let cond = expr_to_df_expr(condition)?;
                let res = expr_to_df_expr(result)?;
                cases.push((Box::new(cond), Box::new(res)));
            }
            let else_expr = else_result.as_ref()
                .map(|expr| expr_to_df_expr(expr))
                .transpose()?;
            use datafusion::logical_expr::Case;
            Ok(DFExpr::Case(Case::new(None, cases, else_expr.map(Box::new))))
        }
        Expr::Coalesce(exprs) => {
            let df_exprs: Vec<DFExpr> = exprs.iter()
                .map(|expr| expr_to_df_expr(expr))
                .collect::<Result<Vec<_>, _>>()?;
            use datafusion::functions::expr_fn::coalesce;
            Ok(coalesce(df_exprs))
        }
        Expr::In { expr, values } => {
            let expr = expr_to_df_expr(expr)?;
            let values: Vec<DFExpr> = values.iter()
                .map(|val| expr_to_df_expr(val))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(expr.in_list(values, false))
        }
        Expr::Sql(sql) => {
            // For simple column references in SQL, treat as column
            // This is a temporary fix - the plan generation should be fixed to use proper Expr::Column
            if !sql.contains(' ') && !sql.contains('(') && !sql.contains(')') {
                // Simple column name
                Ok(DFExpr::Column(DFColumn::from_name(sql)))
            } else {
                // For SQL expressions, we'll need to parse them - simplified for now
                Err(anyhow::anyhow!("SQL expressions not yet supported: {}", sql))
            }
        }
    }
}

/// Convert semstrait Column to DataFusion Expr
fn col_to_df_expr(col: &Column) -> DFExpr {
    if col.table.is_empty() {
        DFExpr::Column(DFColumn::from_name(&col.name))
    } else {
        DFExpr::Column(DFColumn::new(Some(&col.table), &col.name))
    }
}

/// Convert semstrait Literal to DataFusion Expr
fn literal_to_df_expr(lit: &semstrait::plan::Literal) -> DFExpr {
    match lit {
        semstrait::plan::Literal::Null(_) => DFExpr::Literal(datafusion::scalar::ScalarValue::Null, None),
        semstrait::plan::Literal::Bool(b) => DFExpr::Literal(datafusion::scalar::ScalarValue::Boolean(Some(*b)), None),
        semstrait::plan::Literal::Int(i) => DFExpr::Literal(datafusion::scalar::ScalarValue::Int64(Some(*i)), None),
        semstrait::plan::Literal::Float(f) => DFExpr::Literal(datafusion::scalar::ScalarValue::Float64(Some(*f)), None),
        semstrait::plan::Literal::String(s) => DFExpr::Literal(datafusion::scalar::ScalarValue::Utf8(Some(s.clone())), None),
    }
}

/// Convert semstrait BinaryOperator to DataFusion Operator
fn binary_op_to_df_op(op: &BinaryOperator) -> anyhow::Result<datafusion::logical_expr::Operator> {
    match op {
        BinaryOperator::Eq => Ok(datafusion::logical_expr::Operator::Eq),
        BinaryOperator::NotEq => Ok(datafusion::logical_expr::Operator::NotEq),
        BinaryOperator::Lt => Ok(datafusion::logical_expr::Operator::Lt),
        BinaryOperator::LtEq => Ok(datafusion::logical_expr::Operator::LtEq),
        BinaryOperator::Gt => Ok(datafusion::logical_expr::Operator::Gt),
        BinaryOperator::GtEq => Ok(datafusion::logical_expr::Operator::GtEq),
    }
}

/// Convert semstrait AggregateExpr to DataFusion Expr
fn aggregate_expr_to_df_expr(agg_expr: &AggregateExpr) -> anyhow::Result<DFExpr> {
    let input_expr = expr_to_df_expr(&agg_expr.expr)?;

    let df_agg_expr = match &agg_expr.func {
        Aggregation::Sum => sum(input_expr),
        Aggregation::Avg => avg(input_expr),
        Aggregation::Count => count(input_expr),
        Aggregation::CountDistinct => count_distinct(input_expr),
        Aggregation::Min => min(input_expr),
        Aggregation::Max => max(input_expr),
    };

    Ok(df_agg_expr.alias(&agg_expr.alias))
}


/// Convert string type name to Arrow DataType
fn string_to_arrow_type(type_name: &str) -> anyhow::Result<datafusion::arrow::datatypes::DataType> {
    use datafusion::arrow::datatypes::DataType;
    match type_name {
        "string" => Ok(DataType::Utf8),
        "i32" => Ok(DataType::Int32),
        "i64" => Ok(DataType::Int64),
        "f64" => Ok(DataType::Float64),
        "bool" => Ok(DataType::Boolean),
        "date" => Ok(DataType::Date32),
        "timestamp" => Ok(DataType::Timestamp(datafusion::arrow::datatypes::TimeUnit::Microsecond, None)),
        _ => Err(anyhow::anyhow!("Unsupported type: {}", type_name)),
    }
}