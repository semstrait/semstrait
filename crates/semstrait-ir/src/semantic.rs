use semstrait_common::EntityId;
use serde::{Deserialize, Serialize};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRole {
    Dimension,
    Measure,
    Metric,
    Filter,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdditivityType {
    Full,
    Semi(SemiAdditivity),
    Non,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemiAdditivity {
    #[serde(default)]
    pub axes: Vec<EntityId>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<EntityId>,

    pub strategy: SemiAdditivityStrategy,
}

impl SemiAdditivity {
    #[must_use]
    pub fn new(
        axes: Vec<EntityId>,
        by: Option<EntityId>,
        strategy: SemiAdditivityStrategy,
    ) -> Self {
        Self { axes, by, strategy }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemiAdditivityStrategy {
    Latest,
    Earliest,
}
