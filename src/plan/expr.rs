//! Expression types for the logical plan

use crate::model::Aggregation;

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
}

/// Literal values
#[derive(Debug, Clone)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
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

