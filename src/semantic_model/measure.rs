//! Measure types

use serde::Deserialize;
use super::types::{DataType, Aggregation};

/// Measure expression - either simple column name or structured expression
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MeasureExpr {
    /// Simple column reference: "quantity"
    Column(String),
    /// Structured expression tree
    Structured(ExprNode),
}

/// Structured expression node (AST)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExprNode {
    /// Column reference
    Column(String),
    /// Literal value
    Literal(LiteralValue),
    /// Addition: add: [a, b]
    Add(Vec<ExprArg>),
    /// Subtraction: subtract: [a, b]
    Subtract(Vec<ExprArg>),
    /// Multiplication: multiply: [a, b]
    Multiply(Vec<ExprArg>),
    /// Division: divide: [a, b]
    Divide(Vec<ExprArg>),
    /// CASE WHEN expression
    Case(CaseExpr),
}

/// CASE WHEN expression
#[derive(Debug, Clone, Deserialize)]
pub struct CaseExpr {
    /// List of WHEN...THEN branches
    pub when: Vec<CaseWhen>,
    /// Optional ELSE value
    #[serde(rename = "else")]
    pub else_value: Option<Box<ExprArg>>,
}

/// A single WHEN...THEN branch
#[derive(Debug, Clone, Deserialize)]
pub struct CaseWhen {
    /// The condition to evaluate
    pub condition: ConditionExpr,
    /// The value if condition is true
    pub then: ExprArg,
}

/// Condition expression for CASE WHEN and filters
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConditionExpr {
    /// Equal: eq: [a, b]
    Eq(Vec<ExprArg>),
    /// Not equal: ne: [a, b]
    Ne(Vec<ExprArg>),
    /// Greater than: gt: [a, b]
    Gt(Vec<ExprArg>),
    /// Greater or equal: gte: [a, b]
    Gte(Vec<ExprArg>),
    /// Less than: lt: [a, b]
    Lt(Vec<ExprArg>),
    /// Less or equal: lte: [a, b]
    Lte(Vec<ExprArg>),
    /// AND: and: [cond1, cond2, ...]
    And(Vec<ConditionExpr>),
    /// OR: or: [cond1, cond2, ...]
    Or(Vec<ConditionExpr>),
    /// IS NULL: is_null: column_name
    #[serde(rename = "is_null")]
    IsNull(String),
    /// IS NOT NULL: is_not_null: column_name
    #[serde(rename = "is_not_null")]
    IsNotNull(String),
}

/// Expression argument - can be column name shorthand, literal value, or nested node
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ExprArg {
    /// Literal integer value
    LiteralInt(i64),
    /// Literal float value
    LiteralFloat(f64),
    /// Shorthand: just a column name string
    ColumnName(String),
    /// Nested expression node
    Node(ExprNode),
}

/// Literal values in expressions
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

/// A measure definition with aggregation
#[derive(Debug, Deserialize)]
pub struct Measure {
    pub name: String,
    pub label: Option<String>,
    /// Human-readable description for UIs and LLMs
    pub description: Option<String>,
    /// Alternative names (for LLM query understanding)
    pub synonyms: Option<Vec<String>>,
    pub hidden: Option<bool>,
    pub format: Option<String>,
    /// Aggregation function (sum, avg, count, count_distinct, min, max)
    pub aggregation: Aggregation,
    pub expr: MeasureExpr,
    /// Result data type. Defaults to I64 for count, F64 for others.
    #[serde(rename = "type")]
    pub data_type: Option<DataType>,
    pub data_filter: Option<Vec<MeasureFilter>>,
}

impl Measure {
    /// Get the result data type, with smart defaults based on aggregation
    pub fn data_type(&self) -> DataType {
        if let Some(ref t) = self.data_type {
            return t.clone();
        }
        // Default types based on aggregation
        match self.aggregation {
            Aggregation::Count | Aggregation::CountDistinct => DataType::I64,
            _ => DataType::F64,
        }
    }
}

/// Filter that applies to a specific measure
#[derive(Debug, Deserialize)]
pub struct MeasureFilter {
    pub field: String,
    pub user_attribute: Option<String>,
}
