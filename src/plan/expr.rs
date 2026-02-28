//! Expression types for the logical plan

use std::fmt;
use crate::semantic_model::Aggregation;

/// A column reference
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    /// Table name or alias
    pub table: String,
    /// Column name
    pub name: String,
}

impl Column {
    pub fn new(table: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            name: name.into(),
        }
    }

    /// Create an unqualified column reference (no table prefix)
    pub fn unqualified(name: impl Into<String>) -> Self {
        Self {
            table: String::new(),
            name: name.into(),
        }
    }

    /// Fully qualified name: table.column
    pub fn qualified_name(&self) -> String {
        if self.table.is_empty() {
            self.name.clone()
        } else {
        format!("{}.{}", self.table, self.name)
        }
    }
}

/// Scalar expressions
#[derive(Debug, Clone)]
pub enum Expr {
    /// Column reference
    Column(Column),
    /// Literal value
    Literal(Literal),
    /// Binary operation (e.g., a = b, a > 5)
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    /// IN expression (column IN (values))
    In {
        expr: Box<Expr>,
        values: Vec<Expr>,
    },
    /// AND of multiple expressions
    And(Vec<Expr>),
    /// OR of multiple expressions
    Or(Vec<Expr>),
    /// Raw SQL expression (for complex metric expressions)
    Sql(String),
    /// Addition: a + b
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: a - b
    Subtract(Box<Expr>, Box<Expr>),
    /// Multiplication: a * b
    Multiply(Box<Expr>, Box<Expr>),
    /// Division: a / b
    Divide(Box<Expr>, Box<Expr>),
    /// IS NULL check
    IsNull(Box<Expr>),
    /// IS NOT NULL check
    IsNotNull(Box<Expr>),
    /// CASE WHEN expression
    Case {
        /// List of (condition, result) pairs
        when_then: Vec<(Expr, Expr)>,
        /// Optional ELSE result
        else_result: Option<Box<Expr>>,
    },
    /// COALESCE expression: returns first non-NULL value
    Coalesce(Vec<Expr>),
}

/// Literal values
#[derive(Debug, Clone)]
pub enum Literal {
    /// NULL with a specific data type (e.g., "date", "i64", "string")
    Null(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null(t) => write!(f, "NULL::{}", t),
            Literal::Bool(b) => write!(f, "{}", b),
            Literal::Int(i) => write!(f, "{}", i),
            Literal::Float(v) => write!(f, "{}", v),
            Literal::String(s) => write!(f, "'{}'", s),
        }
    }
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl BinaryOperator {
    pub fn as_str(&self) -> &'static str {
        match self {
            BinaryOperator::Eq => "=",
            BinaryOperator::NotEq => "!=",
            BinaryOperator::Lt => "<",
            BinaryOperator::LtEq => "<=",
            BinaryOperator::Gt => ">",
            BinaryOperator::GtEq => ">=",
        }
    }
}

/// An aggregate expression: func(expr) AS alias
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    pub func: Aggregation,
    pub expr: Expr,
    pub alias: String,
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(col) => write!(f, "{}", col.qualified_name()),
            Expr::Literal(lit) => write!(f, "{}", lit),
            Expr::BinaryOp { left, op, right } =>
                write!(f, "{} {} {}", left, op.as_str(), right),
            Expr::In { expr, values } => {
                let vals: Vec<_> = values.iter().map(|v| v.to_string()).collect();
                write!(f, "{} IN ({})", expr, vals.join(", "))
            }
            Expr::And(exprs) => {
                let parts: Vec<_> = exprs.iter().map(|e| e.to_string()).collect();
                write!(f, "({})", parts.join(" AND "))
            }
            Expr::Or(exprs) => {
                let parts: Vec<_> = exprs.iter().map(|e| e.to_string()).collect();
                write!(f, "({})", parts.join(" OR "))
            }
            Expr::Sql(s) => write!(f, "{}", s),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Subtract(a, b) => write!(f, "({} - {})", a, b),
            Expr::Multiply(a, b) => write!(f, "({} * {})", a, b),
            Expr::Divide(a, b) => write!(f, "({} / {})", a, b),
            Expr::IsNull(e) => write!(f, "{} IS NULL", e),
            Expr::IsNotNull(e) => write!(f, "{} IS NOT NULL", e),
            Expr::Case { when_then, else_result } => {
                write!(f, "CASE")?;
                for (cond, result) in when_then {
                    write!(f, " WHEN {} THEN {}", cond, result)?;
                }
                if let Some(else_r) = else_result {
                    write!(f, " ELSE {}", else_r)?;
                }
                write!(f, " END")
            }
            Expr::Coalesce(exprs) => {
                let parts: Vec<_> = exprs.iter().map(|e| e.to_string()).collect();
                write!(f, "coalesce({})", parts.join(", "))
            }
        }
    }
}

impl fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}) AS {}", self.func, self.expr, self.alias)
    }
}

