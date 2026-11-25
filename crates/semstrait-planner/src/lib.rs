#![allow(
    clippy::result_large_err,
    reason = "every stage returns its PlanError unboxed, as ADR 0064's fail-fast form does"
)]

mod advisory;
mod binding;
mod build;
mod context;
mod dataset_strategy;
mod decision;
mod error;
mod fan_out;
mod formula;
mod frame;
mod frame_path;
mod grain;
mod grainset_metrics;
mod grainset_strategy;
mod joinset_strategy;
mod parts;
mod planner;
mod predicate;
mod read_model;
mod request;
mod resolver;
mod routing;
mod semi_additive;
mod steps;
mod strategy;
mod tree;
mod unionset_strategy;

pub use advisory::Advisory;
pub use binding::BoundName;
pub use context::{RequestContext, Timezone, UnknownTimezone, Weekday};
pub use decision::{Combine, Decision, DecisionValue, Operand, PhysicalRead};
pub use error::{LookedIn, PlanError, UnbuildableState};
pub use frame_path::FramePath;
pub use planner::Planner;
pub use predicate::{Placement, Predicate};
pub use read_model::{PhysicalSource, ReadModel, Relationship, TimeAxis};
pub use request::{DimensionModifier, Direction, OrderKey, Request, Selection};
pub use strategy::{DatasetArm, FormulaNode, GrainsetArm, Strategy, StrategyInput, UnionsetArm};
