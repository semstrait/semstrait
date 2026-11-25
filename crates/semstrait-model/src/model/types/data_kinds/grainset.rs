use crate::model::types::data_kinds::base::{ComplexExtras, DataKindBase};
use crate::model::types::data_kinds::dataset::NestedDataset;
use crate::model::types::data_kinds::joinset::NestedJoinset;
use crate::model::types::data_kinds::unionset::NestedUnionset;
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_interface::SemanticInterface;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct GrainsetBody {
    #[serde(flatten)]
    pub base: DataKindBase<ComplexExtras>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The nested Datasets this Grainset rolls up.")
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<NestedDataset>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The nested Unionsets this Grainset rolls up.")
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unionsets: Vec<NestedUnionset>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joinsets: Vec<NestedJoinset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Grainset {
    #[serde(flatten)]
    pub body: GrainsetBody,

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
pub struct NestedGrainset {
    #[serde(flatten)]
    pub body: GrainsetBody,
}
