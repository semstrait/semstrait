//! Query planner (verb module)
//!
//! Transforms a ResolvedQuery into a logical plan (PlanNode).

mod build;
mod error;

pub use build::{plan_query, plan_semantic_query, plan_cross_dataset_group_query, CrossDatasetGroupBranch};
pub use error::PlanError;
