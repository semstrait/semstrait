use crate::model::entity_btree_map::NamedEntity;
use crate::model::serde::{SemanticExpr, Unresolved};
use crate::model::types::entities::ai::AiContext;
use crate::UnresolvedSemanticExpr;
use bon::Builder;
use semstrait_common::EntityId;
use semstrait_common::SemanticsName;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
#[cfg_attr(
    feature = "schemars",
    schemars(description = "A named filter that narrows the rows a data kind exposes.")
)]
pub struct DataKindFilter {
    #[builder(start_fn, into)]
    pub name: SemanticsName,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(skip)]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The entity's id, which keeps its identity across a rename.")
    )]
    pub id: Option<EntityId>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    #[serde(with = "Unresolved::<SemanticExpr>")]
    #[builder(into)]
    pub expr: UnresolvedSemanticExpr,
}

impl NamedEntity for DataKindFilter {
    fn entity_id(&self) -> Option<&str> {
        self.id.as_deref()
    }
    fn entity_name(&self) -> &str {
        self.name.as_str()
    }
}
