//! Expression conversion: semantic model expressions -> plan expressions

use crate::semantic_model::{MeasureExpr, ExprNode, ExprArg, LiteralValue, MetricExpr, MetricExprNode, MetricExprArg, CaseExpr, ConditionExpr};
use crate::plan::{BinaryOperator, Column, Expr, Literal};
use crate::resolver::ResolvedFilter;

/// Convert a MeasureExpr to a plan Expr
pub fn convert_measure_expr(expr: &MeasureExpr) -> Expr {
    match expr {
        MeasureExpr::Column(name) => Expr::Sql(name.clone()),
        MeasureExpr::Structured(node) => convert_expr_node(node),
    }
}

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

fn convert_expr_arg(arg: &ExprArg) -> Expr {
    match arg {
        ExprArg::LiteralInt(i) => Expr::Literal(Literal::Int(*i)),
        ExprArg::LiteralFloat(f) => Expr::Literal(Literal::Float(*f)),
        ExprArg::ColumnName(name) => Expr::Sql(name.clone()),
        ExprArg::Node(node) => convert_expr_node(node),
    }
}

fn binary_args(args: &[ExprArg]) -> (Expr, Expr) {
    let left = args.get(0).map(convert_expr_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
    let right = args.get(1).map(convert_expr_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
    (left, right)
}

fn convert_literal(lit: &LiteralValue) -> Literal {
    match lit {
        LiteralValue::Int(i) => Literal::Int(*i),
        LiteralValue::Float(f) => Literal::Float(*f),
        LiteralValue::String(s) => Literal::String(s.clone()),
        LiteralValue::Bool(b) => Literal::Bool(*b),
    }
}

/// Convert a MetricExpr to a plan Expr (post-aggregation references)
pub fn convert_metric_expr(expr: &MetricExpr) -> Expr {
    match expr {
        MetricExpr::MeasureRef(name) => {
            Expr::Column(Column::unqualified(name))
        }
        MetricExpr::Structured(node) => convert_metric_node(node),
    }
}

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
            panic!("Metric CASE expressions should be resolved before planning. Use plan_cross_dataset_group_query for cross-tableGroup metrics.")
        }
    }
}

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

fn metric_binary_args(args: &[MetricExprArg]) -> (Expr, Expr) {
    let left = args.get(0).map(convert_metric_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
    let right = args.get(1).map(convert_metric_arg).unwrap_or(Expr::Literal(Literal::Null("f64".to_string())));
    (left, right)
}

/// Build a filter expression from a ResolvedFilter
pub fn build_filter_expr(filter: &ResolvedFilter<'_>, fact_alias: &str) -> Expr {
    let base_expr = match &filter.attribute {
        crate::resolver::AttributeRef::Degenerate { attribute, .. } => {
            Expr::Column(Column::new(fact_alias, attribute.column_name()))
        }
        crate::resolver::AttributeRef::Joined { group_dim, dimension, attribute, .. } => {
            if group_dim.join.is_some() {
                let dim_alias = dimension.alias.as_deref().unwrap_or(&dimension.name);
                Expr::Column(Column::new(dim_alias, attribute.column_name()))
            } else {
                Expr::Column(Column::new(fact_alias, attribute.column_name()))
            }
        }
        crate::resolver::AttributeRef::Meta { value, .. } => {
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
        _ => Expr::BinaryOp {
            left: Box::new(column_expr),
            op: BinaryOperator::Eq,
            right: Box::new(json_to_literal(&filter.value)),
        },
    }
}

/// Convert a JSON value to a Literal expression
pub fn json_to_literal(value: &serde_json::Value) -> Expr {
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
        _ => Literal::Null("string".to_string()),
    };
    Expr::Literal(lit)
}
