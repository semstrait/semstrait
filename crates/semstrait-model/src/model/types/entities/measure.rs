use crate::model::serde::{SemanticExpr, Unresolved};
use crate::model::types::entities::agg::Agg;
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_body::{
    authoring_body, decode_gated, AuthoringBody, NamedSemanticBody, Reference, SemanticBody,
    SemanticSite,
};
use crate::yaml::tagged::{serialize_single_key, single_key_map};
use crate::UnresolvedDataType;
use crate::UnresolvedSemanticExpr;
use bon::{bon, Builder};
use semstrait_common::identity::{mint_named_entity_id, EntityNamespace};
use semstrait_common::SemanticsName;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "A Measure of the root `measures:` pool, or one declared inline at a site."
    )
)]
#[non_exhaustive]
pub struct Measure {
    #[serde(flatten)]
    pub body: NamedSemanticBody,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The aggregate function of the Measure, which a reference site cannot override."
        )
    )]
    pub agg: Agg,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "How the Measure adds up across Dimensions: `full`, `non` or `semi`."
        )
    )]
    pub additivity: Option<AdditivityType>,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The predicate that keeps a row in the Measure's aggregate.")
    )]
    pub filter: Option<UnresolvedSemanticExpr>,
}

#[derive(Deserialize)]
struct MeasureWire {
    #[serde(flatten)]
    body: NamedSemanticBody,
    agg: Agg,
    #[serde(default)]
    additivity: Option<AdditivityType>,
    #[serde(default, with = "Unresolved::<SemanticExpr>")]
    filter: Option<UnresolvedSemanticExpr>,
}

impl<'de> Deserialize<'de> for Measure {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let MeasureWire {
            body,
            agg,
            additivity,
            filter,
        } = decode_gated(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self {
            body,
            agg,
            additivity,
            filter,
        })
    }
}

#[bon]
impl Measure {
    #[builder(
        builder_type(name = MeasureBuilder, vis = "pub"),
        state_mod(name = measure_builder, vis = "pub"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: SemanticsName,
        #[builder(into)] data_type: UnresolvedDataType,
        agg: Agg,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
        #[builder(into)] expr: Option<UnresolvedSemanticExpr>,
        additivity: Option<AdditivityType>,
        #[builder(into)] filter: Option<UnresolvedSemanticExpr>,
    ) -> Self {
        Measure {
            body: NamedSemanticBody {
                body: SemanticBody {
                    id: Some(mint_named_entity_id(
                        EntityNamespace::Measure,
                        name.as_str(),
                    )),
                    expr,
                },
                name,
                data_type,
                description,
                ai_context,
            },
            agg,
            additivity,
            filter,
        }
    }
}

authoring_body!(named Measure, "Measure", ["agg", "additivity", "filter"], body);

impl<S: measure_builder::State> MeasureBuilder<S>
where
    S::Additivity: measure_builder::IsUnset,
{
    pub fn full(self) -> MeasureBuilder<measure_builder::SetAdditivity<S>> {
        self.additivity(AdditivityType::Full)
    }
    pub fn semi(self, s: SemiAdditivity) -> MeasureBuilder<measure_builder::SetAdditivity<S>> {
        self.additivity(AdditivityType::Semi(s))
    }
    pub fn non(self) -> MeasureBuilder<measure_builder::SetAdditivity<S>> {
        self.additivity(AdditivityType::Non)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AdditivityType {
    Full,
    Semi(SemiAdditivity),
    Non,
}

impl Serialize for AdditivityType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Full => serializer.serialize_str("full"),
            Self::Non => serializer.serialize_str("non"),
            Self::Semi(body) => serialize_single_key(serializer, "semi", body),
        }
    }
}

impl<'de> Deserialize<'de> for AdditivityType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => match s.as_str() {
                "full" => Ok(Self::Full),
                "non" => Ok(Self::Non),
                "semi" => Err(serde::de::Error::custom(
                    "AdditivityType::semi requires a body (`{ semi: { axes, strategy } }`)",
                )),
                other => Err(serde::de::Error::custom(format!(
                    "AdditivityType: unknown variant `{other}`"
                ))),
            },
            Value::Mapping(_) => {
                let (key, body) = single_key_map::<D::Error>(value, "AdditivityType")?;
                match key.as_str() {
                    "semi" => {
                        let s: SemiAdditivity =
                            serde_yaml::from_value(body).map_err(serde::de::Error::custom)?;
                        Ok(Self::Semi(s))
                    }
                    "full" | "non" => Err(serde::de::Error::custom(format!(
                        "AdditivityType::{key} is body-less; use the bare-string form"
                    ))),
                    other => Err(serde::de::Error::custom(format!(
                        "AdditivityType: unknown variant `{other}`"
                    ))),
                }
            }
            other => Err(serde::de::Error::custom(format!(
                "AdditivityType: expected string or mapping, got {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
pub struct SemiAdditivity {
    #[serde(default)]
    #[builder(default)]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The Dimensions inside which the latest or earliest rows are taken."
        )
    )]
    pub axes: Vec<SemanticsName>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The one Dimension that `latest` and `earliest` order by.")
    )]
    pub by: Option<SemanticsName>,
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The rollup strategy for the non-additive axis.")
    )]
    pub strategy: SemiAdditivityStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SemiAdditivityStrategy {
    Latest,
    Earliest,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "A `ref:` to a pooled Measure, with the `expr` and `filter` that a site may override."
    )
)]
#[non_exhaustive]
pub struct MeasureRef {
    #[serde(flatten)]
    pub body: Reference,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    pub filter: Option<UnresolvedSemanticExpr>,
}

#[derive(Deserialize)]
struct MeasureRefWire {
    #[serde(flatten)]
    body: Reference,
    #[serde(default, with = "Unresolved::<SemanticExpr>")]
    filter: Option<UnresolvedSemanticExpr>,
}

impl<'de> Deserialize<'de> for MeasureRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let MeasureRefWire { body, filter } =
            decode_gated(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self { body, filter })
    }
}

authoring_body!(reference MeasureRef, "Measure reference site", ["filter"], body);

pub type MeasureSite = SemanticSite<Measure, MeasureRef>;

impl MeasureSite {
    pub fn r#ref(name: impl Into<SemanticsName>) -> Self {
        Self::Reference(MeasureRef {
            body: Reference::new(name),
            filter: None,
        })
    }

    pub fn ref_with_expr(name: impl Into<SemanticsName>, expr: UnresolvedSemanticExpr) -> Self {
        Self::Reference(MeasureRef {
            body: Reference::with_expr(name, expr),
            filter: None,
        })
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for AdditivityType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "AdditivityType".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let semi = serde_json::to_value(generator.subschema_for::<SemiAdditivity>())
            .expect("subschema to value");
        serde_json::from_value(serde_json::json!({
            "oneOf": [
                { "type": "string", "enum": ["full", "non"] },
                { "type": "object", "additionalProperties": false, "required": ["semi"],
                  "properties": { "semi": semi } }
            ]
        }))
        .expect("AdditivityType JSON Schema is valid JSON")
    }
}

#[cfg(test)]
mod authoring_serialize_tests {
    use super::*;

    #[track_caller]
    fn round_trip(v: AdditivityType) {
        let yaml = serde_yaml::to_string(&v).expect("serialize");
        assert!(!yaml.contains('!'), "no `!`-tag leak:\n{yaml}");
        let back: AdditivityType = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(v, back, "round-trip mismatch via:\n{yaml}");
    }

    #[test]
    fn additivity_type_round_trips_all_variants() {
        round_trip(AdditivityType::Full);
        round_trip(AdditivityType::Non);
        round_trip(AdditivityType::Semi(SemiAdditivity {
            axes: vec![SemanticsName("region".into())],
            by: None,
            strategy: SemiAdditivityStrategy::Latest,
        }));
    }
}
