use crate::error::build::ModelBuildErrorKind;
use crate::error::serde_message;
use crate::error::validate::ValidateErrorKind;
use crate::model::serde::{SemanticExpr, Unresolved};
use crate::model::types::entities::ai::AiContext;
use crate::UnresolvedSemanticExpr;
use bon::Builder;
use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::{DataKindName, EntityId, SemanticsName};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(
    start_fn = builder,
    finish_fn = finalize,
    state_mod(name = relationship_builder, vis = "pub"),
)]
pub struct Relationship {
    #[serde(default, skip_serializing_if = "JoinKeySpec::is_auto")]
    #[builder(field)]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The join basis: a list of names, `auto` (the default) or `cross`."
        )
    )]
    pub on: JoinKeySpec,

    #[builder(into)]
    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(skip)]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The entity's id, which keeps its identity across a rename.")
    )]
    pub id: Option<EntityId>,

    #[builder(into)]
    pub from: DataKindName,
    #[builder(into)]
    pub to: DataKindName,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    #[builder(into)]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "A predicate on the joined rows; with `on: cross`, a non-equi join."
        )
    )]
    pub filter: Option<UnresolvedSemanticExpr>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "How many rows each side may match; absent only under `on: cross`."
        )
    )]
    pub cardinality: Option<Cardinality>,

    #[serde(default)]
    #[builder(default)]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The asserted referential integrity; `assumed` by default.")
    )]
    pub integrity: Integrity,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Which side keeps its unmatched rows; derived when absent.")
    )]
    pub optional: Option<Optional>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "Which side receives filters through the join; derived when absent."
        )
    )]
    pub cross_filter: Option<CrossFilter>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub description: Option<String>,
}

impl<S: relationship_builder::IsComplete> RelationshipBuilder<S> {
    pub fn build(self) -> Result<Relationship, Diagnostic<ModelBuildErrorKind>> {
        let r = self.finalize();
        validate_relationship_build(&r).map_err(Diagnostic::new)?;
        Ok(r)
    }
}

impl<S: relationship_builder::State> RelationshipBuilder<S> {
    pub fn on_key(mut self, name: impl Into<SemanticsName>) -> Self {
        match &mut self.on {
            JoinKeySpec::Declared(names) => names.push(name.into()),
            _ => self.on = JoinKeySpec::Declared(vec![name.into()]),
        }
        self
    }

    pub fn on(mut self, names: impl IntoIterator<Item = impl Into<SemanticsName>>) -> Self {
        self.on = JoinKeySpec::Declared(names.into_iter().map(Into::into).collect());
        self
    }

    pub fn on_auto(mut self) -> Self {
        self.on = JoinKeySpec::Auto;
        self
    }

    pub fn on_cross(mut self) -> Self {
        self.on = JoinKeySpec::Cross;
        self
    }
}

impl<S: relationship_builder::State> RelationshipBuilder<S>
where
    S::Cardinality: relationship_builder::IsUnset,
{
    pub fn one_to_one(self) -> RelationshipBuilder<relationship_builder::SetCardinality<S>> {
        self.cardinality(Cardinality::OneToOne)
    }
    pub fn one_to_many(self) -> RelationshipBuilder<relationship_builder::SetCardinality<S>> {
        self.cardinality(Cardinality::OneToMany)
    }
    pub fn many_to_one(self) -> RelationshipBuilder<relationship_builder::SetCardinality<S>> {
        self.cardinality(Cardinality::ManyToOne)
    }
    pub fn many_to_many(self) -> RelationshipBuilder<relationship_builder::SetCardinality<S>> {
        self.cardinality(Cardinality::ManyToMany)
    }
}

fn validate_relationship_build(r: &Relationship) -> Result<(), ModelBuildErrorKind> {
    use Cardinality::*;
    if let JoinKeySpec::Declared(names) = &r.on {
        if names.is_empty() {
            return Err(ModelBuildErrorKind::Validate(
                ValidateErrorKind::RelationshipJoinBasisEmpty {
                    relationship: r.name.clone(),
                },
            ));
        }
        if let Some(dup) = first_repeat(names) {
            return Err(ModelBuildErrorKind::Validate(
                ValidateErrorKind::RelationshipJoinKeyDuplicated {
                    relationship: r.name.clone(),
                    key: dup.as_str().to_owned(),
                },
            ));
        }
    }
    if r.cardinality.is_none() && !matches!(r.on, JoinKeySpec::Cross) {
        return Err(ModelBuildErrorKind::Validate(
            ValidateErrorKind::RelationshipMissingCardinality {
                relationship: r.name.clone(),
            },
        ));
    }
    if let Some(kind) = cross_with_preserved_side(r) {
        return Err(ModelBuildErrorKind::Validate(kind));
    }
    let symmetric = matches!(r.cardinality, Some(OneToOne | ManyToMany));
    if symmetric {
        if r.optional.is_none() {
            return Err(ModelBuildErrorKind::Validate(
                ValidateErrorKind::RelationshipSymmetricCardinalityIncomplete {
                    relationship: r.name.clone(),
                    missing: "optional".to_string(),
                },
            ));
        }
        if r.cross_filter.is_none() {
            return Err(ModelBuildErrorKind::Validate(
                ValidateErrorKind::RelationshipSymmetricCardinalityIncomplete {
                    relationship: r.name.clone(),
                    missing: "cross_filter".to_string(),
                },
            ));
        }
    }
    if matches!(r.cardinality, Some(ManyToMany)) {
        if let Some(cf) = r.cross_filter {
            if matches!(cf, CrossFilter::Left | CrossFilter::Right) {
                return Err(ModelBuildErrorKind::Validate(
                    ValidateErrorKind::RelationshipManyToManyCrossFilterDirectional {
                        relationship: r.name.clone(),
                    },
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn cross_with_preserved_side(r: &Relationship) -> Option<ValidateErrorKind> {
    match r.optional {
        Some(optional) if matches!(r.on, JoinKeySpec::Cross) && optional != Optional::None => {
            Some(ValidateErrorKind::RelationshipCrossWithPreservedSide {
                relationship: r.name.clone(),
                optional,
            })
        }
        _ => None,
    }
}

fn first_repeat(names: &[SemanticsName]) -> Option<&SemanticsName> {
    names
        .iter()
        .enumerate()
        .find(|(i, name)| names[..*i].contains(name))
        .map(|(_, name)| name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Cardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Integrity {
    Enforced,
    #[default]
    Assumed,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Optional {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Keep neither side's unmatched rows: an inner join.")
    )]
    None,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Keep the `from` side's unmatched rows: a left join.")
    )]
    Left,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Keep the `to` side's unmatched rows: a right join.")
    )]
    Right,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Keep both sides' unmatched rows: a full join.")
    )]
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CrossFilter {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "No filter flows between the sides.")
    )]
    None,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The `from` side receives filters from the `to` side.")
    )]
    Left,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The `to` side receives filters from the `from` side.")
    )]
    Right,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Each side receives filters from the other.")
    )]
    Both,
}

const AUTO: &str = "auto";
const CROSS: &str = "cross";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum JoinKeySpec {
    #[default]
    Auto,
    Cross,
    Declared(Vec<SemanticsName>),
}

impl JoinKeySpec {
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    pub fn declared(&self) -> Option<&[SemanticsName]> {
        match self {
            Self::Declared(names) => Some(names),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for JoinKeySpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;

        match Value::deserialize(deserializer)? {
            Value::String(s) if s == AUTO => Ok(Self::Auto),
            Value::String(s) if s == CROSS => Ok(Self::Cross),
            Value::String(s) => Err(D::Error::custom(
                serde_message::join_basis_unknown_spelling(&s),
            )),
            Value::Sequence(items) => {
                if items.is_empty() {
                    return Err(D::Error::custom(serde_message::join_basis_empty()));
                }
                let mut names: Vec<SemanticsName> = Vec::with_capacity(items.len());
                for item in items {
                    let Value::String(name) = item else {
                        return Err(D::Error::custom("on: every entry is a bare Semantics name"));
                    };
                    let name = SemanticsName(name);
                    if names.contains(&name) {
                        return Err(D::Error::custom(serde_message::join_key_duplicated(
                            name.as_str(),
                        )));
                    }
                    names.push(name);
                }
                Ok(Self::Declared(names))
            }
            other => Err(D::Error::custom(format!(
                "on: expected a list of names, `{AUTO}`, or `{CROSS}`, got {other:?}"
            ))),
        }
    }
}

impl Serialize for JoinKeySpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str(AUTO),
            Self::Cross => serializer.serialize_str(CROSS),
            Self::Declared(names) => names.serialize(serializer),
        }
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for JoinKeySpec {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "JoinKeySpec".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        serde_json::from_value(serde_json::json!({
            "oneOf": [
                { "type": "string", "enum": [AUTO, CROSS] },
                { "type": "array", "items": { "type": "string" }, "minItems": 1 }
            ]
        }))
        .expect("JoinKeySpec JSON Schema is valid JSON")
    }
}
