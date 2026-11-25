use crate::model::types::data_kinds::base::{ComplexExtras, DataKindBase};
use crate::model::types::data_kinds::dataset::NestedDataset;
use crate::model::types::data_kinds::grainset::NestedGrainset;
use crate::model::types::data_kinds::unionset::NestedUnionset;
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::relationship::Relationship;
use crate::model::types::entities::semantic_interface::SemanticInterface;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct JoinsetBody {
    #[serde(flatten)]
    pub base: DataKindBase<ComplexExtras>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The Joinset's own relationships, in the same shape as the root `relationships:` pool."
        )
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relationships: Vec<Relationship>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<NestedDataset>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub grainsets: Vec<NestedGrainset>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unionsets: Vec<NestedUnionset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Joinset {
    #[serde(flatten)]
    pub body: JoinsetBody,

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
pub struct NestedJoinset {
    #[serde(flatten)]
    pub body: JoinsetBody,
}
