use bon::Builder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "LLM-facing hints for this entity: synonyms, an extra description, examples and a unit."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
pub struct AiContext {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "Synonyms an LLM may use for this entity: each key is an alias, its values are canonical forms."
        )
    )]
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    #[builder(default)]
    pub synonyms: BTreeMap<String, Vec<String>>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "A plain-language description an LLM may show the user; freer than the entity's `description:`."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub description: Option<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Example queries or phrasings an LLM may emit.")
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default)]
    pub examples: Vec<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The unit of measurement for a numeric entity, e.g. `usd`, `percent` or `events_per_minute`."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub unit: Option<String>,
}

impl AiContext {
    pub fn is_empty(&self) -> bool {
        self.synonyms.is_empty()
            && self.description.is_none()
            && self.examples.is_empty()
            && self.unit.is_none()
    }
}
