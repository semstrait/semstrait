//! Query planner (verb module)
//!
//! Transforms a ResolvedQuery into a logical plan (PlanNode).
//!
//! ## Module structure
//!
//! - `plan` — top-level entry point (`plan_semantic_query`)
//! - `table` — resolver-based single-table planning (`plan_query`) and unified
//!    tableGroup branch builders
//! - `cross` — cross-datasetGroup metric planning (UNION + re-aggregate)
//! - `union` — conformed, qualified, partitioned, and virtual-only UNION queries
//! - `join` — same-datasetGroup multi-table JOIN planning
//! - `expr` — semantic model expression → plan expression conversion
//! - `util` — shared helpers (column builders, dimension parsing, virtual values)
//! - `error` — `PlanError` type

mod plan;
mod table;
mod cross;
mod union;
mod join;
mod expr;
mod util;
mod error;

pub use plan::plan_semantic_query;
pub use table::plan_query;
pub use cross::{plan_cross_dataset_group_query, plan_multi_cross_dataset_group_query, CrossDatasetGroupBranch};
pub use error::PlanError;
