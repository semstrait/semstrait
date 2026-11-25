use crate::model::types::data_kinds::base::{DataKindBase, LeafExtras};
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_interface::SemanticInterface;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct DatasetBody {
    #[serde(flatten)]
    pub base: DataKindBase<LeafExtras>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "A Dataset: a leaf data kind, with its own description, AI hints and interface."
    )
)]
#[non_exhaustive]
pub struct Dataset {
    #[serde(flatten)]
    pub body: DatasetBody,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    #[serde(flatten, default)]
    pub semantic_interface: SemanticInterface,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "A Dataset nested inside a complex data kind, without its own description, AI hints or interface."
    )
)]
#[non_exhaustive]
pub struct NestedDataset {
    #[serde(flatten)]
    pub body: DatasetBody,
}
