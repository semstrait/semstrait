use crate::model::types::data_kinds::base::{ComplexExtras, DataKindBase};
use crate::model::types::data_kinds::dataset::NestedDataset;
use crate::model::types::data_kinds::grainset::NestedGrainset;
use crate::model::types::data_kinds::joinset::NestedJoinset;
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_interface::SemanticInterface;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(description = "How a Unionset combines its children's rows: `all` or `unique`.")
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UnionMode {
    #[default]
    All,
    Unique,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct UnionsetBody {
    #[serde(flatten)]
    pub base: DataKindBase<ComplexExtras>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "How the Unionset combines its children; defaults to `all` when omitted."
        )
    )]
    #[serde(default)]
    pub mode: UnionMode,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<NestedDataset>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grainsets: Vec<NestedGrainset>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joinsets: Vec<NestedJoinset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Unionset {
    #[serde(flatten)]
    pub body: UnionsetBody,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    #[serde(flatten, default)]
    pub semantic_interface: SemanticInterface,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct NestedUnionset {
    #[serde(flatten)]
    pub body: UnionsetBody,
}
