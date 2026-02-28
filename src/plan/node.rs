//! Plan node types

use std::fmt;
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

// =============================================================================
// Display - indented plan tree (DataFusion-style)
// =============================================================================

impl PlanNode {
    /// Return a display wrapper that renders the plan as an indented tree.
    ///
    /// Mirrors DataFusion's `LogicalPlan::display_indent()` style:
    /// ```text
    /// Projection: fact.year AS dates.year, sum(fact.amount) AS revenue
    ///   Sort: fact.year ASC
    ///     Aggregate: groupBy=[[fact.year]], aggr=[[sum(fact.amount) AS revenue]]
    ///       InnerJoin: fact.date_id = dates.date_id
    ///         TableScan: fact projection=[date_id, amount]
    ///         TableScan: dates projection=[date_id, year]
    /// ```
    pub fn display_indent(&self) -> impl fmt::Display + '_ {
        struct IndentDisplay<'a>(&'a PlanNode);
        impl fmt::Display for IndentDisplay<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt_indent(f, 0)
            }
        }
        IndentDisplay(self)
    }

    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        match self {
            PlanNode::Scan(scan) => {
                let name = scan.alias.as_deref().unwrap_or(&scan.table);
                write!(f, "{}TableScan: {} projection=[{}]",
                    pad, name, scan.columns.join(", "))
            }
            PlanNode::Join(join) => {
                let jt = match join.join_type {
                    JoinType::Inner => "Inner",
                    JoinType::Left => "Left",
                    JoinType::Right => "Right",
                    JoinType::Full => "Full",
                };
                writeln!(f, "{}{}Join: {} = {}",
                    pad, jt,
                    join.left_key.qualified_name(),
                    join.right_key.qualified_name())?;
                join.left.fmt_indent(f, indent + 1)?;
                writeln!(f)?;
                join.right.fmt_indent(f, indent + 1)
            }
            PlanNode::Filter(filter) => {
                writeln!(f, "{}Filter: {}", pad, filter.predicate)?;
                filter.input.fmt_indent(f, indent + 1)
            }
            PlanNode::Aggregate(agg) => {
                let groups: Vec<_> = agg.group_by.iter()
                    .map(|c| c.qualified_name()).collect();
                let aggrs: Vec<_> = agg.aggregates.iter()
                    .map(|a| a.to_string()).collect();
                writeln!(f, "{}Aggregate: groupBy=[[{}]], aggr=[[{}]]",
                    pad, groups.join(", "), aggrs.join(", "))?;
                agg.input.fmt_indent(f, indent + 1)
            }
            PlanNode::Project(proj) => {
                let exprs: Vec<_> = proj.expressions.iter()
                    .map(|pe| {
                        if pe.expr.to_string() == pe.alias {
                            pe.alias.clone()
                        } else {
                            format!("{} AS {}", pe.expr, pe.alias)
                        }
                    })
                    .collect();
                writeln!(f, "{}Projection: {}", pad, exprs.join(", "))?;
                proj.input.fmt_indent(f, indent + 1)
            }
            PlanNode::Sort(sort) => {
                let keys: Vec<_> = sort.sort_keys.iter()
                    .map(|k| {
                        let dir = match k.direction {
                            SortDirection::Ascending => "ASC",
                            SortDirection::Descending => "DESC",
                        };
                        format!("{} {}", k.column, dir)
                    })
                    .collect();
                writeln!(f, "{}Sort: {}", pad, keys.join(", "))?;
                sort.input.fmt_indent(f, indent + 1)
            }
            PlanNode::Union(union) => {
                writeln!(f, "{}Union", pad)?;
                for (i, input) in union.inputs.iter().enumerate() {
                    if i > 0 { writeln!(f)?; }
                    input.fmt_indent(f, indent + 1)?;
                }
                Ok(())
            }
            PlanNode::VirtualTable(vt) => {
                write!(f, "{}VirtualTable: [{}] ({} rows)",
                    pad, vt.columns.join(", "), vt.rows.len())
            }
        }
    }
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, 0)
    }
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JoinType::Inner => write!(f, "Inner"),
            JoinType::Left => write!(f, "Left"),
            JoinType::Right => write!(f, "Right"),
            JoinType::Full => write!(f, "Full"),
        }
    }
}
