//! Plan node types

use super::expr::{AggregateExpr, Column, Expr};

/// A node in the logical plan tree
#[derive(Debug)]
pub enum PlanNode {
    /// Scan a table
    Scan(Scan),
    /// Join two relations
    Join(Join),
    /// Filter rows
    Filter(Filter),
    /// Aggregate (GROUP BY)
    Aggregate(Aggregate),
    /// Project columns
    Project(Project),
    /// Sort rows (ORDER BY)
    Sort(Sort),
    /// Union multiple relations (UNION ALL)
    Union(Union),
    /// Virtual table with literal values (like SQL VALUES clause)
    /// Used for metadata-only queries that don't need table scans
    VirtualTable(VirtualTable),
}

/// Scan a table
#[derive(Debug)]
pub struct Scan {
    /// Table name (schema.table)
    pub table: String,
    /// Alias for the table
    pub alias: Option<String>,
    /// Column names in the table (for Substrait schema)
    pub columns: Vec<String>,
    /// Column types (e.g., "i32", "i64", "f64", "string", etc.)
    pub column_types: Vec<String>,
}

impl Scan {
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            alias: None,
            columns: Vec::new(),
            column_types: Vec::new(),
        }
    }

    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    pub fn with_columns(mut self, columns: Vec<String>, types: Vec<String>) -> Self {
        self.columns = columns;
        self.column_types = types;
        self
    }

    /// Get the name to use for column references (alias if set, otherwise table)
    pub fn reference_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.table)
    }

    /// Get the index of a column by name
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Get the type of a column by index
    pub fn column_type(&self, index: usize) -> &str {
        self.column_types.get(index).map(|s| s.as_str()).unwrap_or("string")
    }
}

/// Join type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Join two relations
#[derive(Debug)]
pub struct Join {
    /// Left input
    pub left: Box<PlanNode>,
    /// Right input
    pub right: Box<PlanNode>,
    /// Join type
    pub join_type: JoinType,
    /// Left key column
    pub left_key: Column,
    /// Right key column
    pub right_key: Column,
}

/// Filter rows (WHERE clause)
#[derive(Debug)]
pub struct Filter {
    /// Input relation
    pub input: Box<PlanNode>,
    /// Filter predicate
    pub predicate: Expr,
}

/// Aggregate (GROUP BY)
#[derive(Debug)]
pub struct Aggregate {
    /// Input relation
    pub input: Box<PlanNode>,
    /// GROUP BY columns
    pub group_by: Vec<Column>,
    /// Aggregate expressions
    pub aggregates: Vec<AggregateExpr>,
}

/// Project specific columns or computed expressions
#[derive(Debug)]
pub struct Project {
    /// Input relation
    pub input: Box<PlanNode>,
    /// Expressions to project with their aliases
    pub expressions: Vec<ProjectExpr>,
}

/// A projected expression with its output alias
#[derive(Debug, Clone)]
pub struct ProjectExpr {
    /// The expression to compute
    pub expr: Expr,
    /// Output column name
    pub alias: String,
}

/// Sort rows (ORDER BY)
#[derive(Debug)]
pub struct Sort {
    /// Input relation
    pub input: Box<PlanNode>,
    /// Sort keys with direction
    pub sort_keys: Vec<SortKey>,
}

/// A sort key with direction
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column name to sort by
    pub column: String,
    /// Sort direction
    pub direction: SortDirection,
}

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Union multiple relations (UNION ALL)
/// 
/// All inputs must have compatible schemas (same number and types of columns).
/// Used for combining results from multiple table groups or partitioned tables.
#[derive(Debug)]
pub struct Union {
    /// Input relations to union (must have at least 2)
    pub inputs: Vec<PlanNode>,
}

/// Virtual table with literal values (like SQL VALUES clause)
/// 
/// Used for metadata-only queries (e.g., querying only `_dataset` attributes)
/// where no actual table scan is needed. Each row is a set of literal values.
#[derive(Debug)]
pub struct VirtualTable {
    /// Column names
    pub columns: Vec<String>,
    /// Column types (e.g., "string", "i32", etc.)
    pub column_types: Vec<String>,
    /// Rows of literal values (each inner Vec is one row)
    pub rows: Vec<Vec<LiteralValue>>,
}

/// A literal value for VirtualTable rows
#[derive(Debug, Clone)]
pub enum LiteralValue {
    String(String),
    Int32(i32),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Null,
}
