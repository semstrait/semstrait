//! Logical plan types (noun module)
//!
//! Represents a relational algebra tree that can be translated to Substrait.

mod node;
mod expr;

pub use node::{PlanNode, Scan, Join, JoinType, Filter, Aggregate, Project, ProjectExpr, Sort, SortKey, SortDirection, Union, VirtualTable, LiteralValue};
pub use expr::{Expr, AggregateExpr, Column, Literal, BinaryOperator};

