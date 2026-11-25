use crate::error::build::ModelBuildErrorKind;
use crate::UnresolvedDataType;
use semstrait_common::{Granularity, SemanticsName};
use semstrait_sql::expr::Literal;

use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::keyness::{decode_keyed_site, Keyness};
use crate::model::types::entities::semantic_body::{
    authoring_body, decode_gated, AuthoringBody, NamedSemanticBody, Reference, SemanticBody,
    SemanticSite,
};
use crate::yaml::tagged::{serialize_single_key, single_key_map};
use crate::UnresolvedSemanticExpr;
use bon::{bon, Builder};
use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::identity::{mint_named_entity_id, EntityNamespace};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_yaml::Value;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "A Dimension of the root `dimensions:` pool, or one declared inline at a site."
    )
)]
#[non_exhaustive]
pub struct Dimension {
    #[serde(flatten)]
    pub body: NamedSemanticBody,

    #[serde(rename = "type")]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The Dimension's category, which drives planner and adapter behavior."
        )
    )]
    pub dim_type: DimensionType,
}

#[derive(Deserialize)]
struct DimensionWire {
    #[serde(flatten)]
    body: NamedSemanticBody,
    #[serde(rename = "type")]
    dim_type: DimensionType,
}

impl<'de> Deserialize<'de> for Dimension {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let DimensionWire { body, dim_type } =
            decode_gated(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self { body, dim_type })
    }
}

#[bon]
impl Dimension {
    #[builder(
        builder_type(name = DimensionBuilder, vis = "pub"),
        state_mod(name = dimension_builder, vis = "pub"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: SemanticsName,
        #[builder(into)] data_type: UnresolvedDataType,
        dim_type: DimensionType,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
        #[builder(into)] expr: Option<UnresolvedSemanticExpr>,
    ) -> Self {
        Dimension {
            body: NamedSemanticBody {
                body: SemanticBody {
                    id: Some(mint_named_entity_id(
                        EntityNamespace::Dimension,
                        name.as_str(),
                    )),
                    expr,
                },
                name,
                data_type,
                description,
                ai_context,
            },
            dim_type,
        }
    }
}

authoring_body!(named Dimension, "Dimension", ["type"], body);

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DimensionType {
    Temporal(TemporalDimensionBody),
    Categorical,
    Binary,
    Geo,
    Bucketed(BucketedDimensionBody),
    Metadata(MetadataDimensionBody),
}

impl Serialize for DimensionType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Categorical => serializer.serialize_str("categorical"),
            Self::Binary => serializer.serialize_str("binary"),
            Self::Geo => serializer.serialize_str("geo"),
            Self::Temporal(body) => serialize_single_key(serializer, "temporal", body),
            Self::Bucketed(body) => serialize_single_key(serializer, "bucketed", body),
            Self::Metadata(body) => serialize_single_key(serializer, "metadata", body),
        }
    }
}

impl DimensionType {
    pub fn temporal(granularities: impl IntoIterator<Item = Granularity>) -> Self {
        Self::Temporal(TemporalDimensionBody {
            granularities: granularities.into_iter().collect(),
        })
    }

    pub fn categorical() -> Self {
        Self::Categorical
    }

    pub fn binary() -> Self {
        Self::Binary
    }

    pub fn geo() -> Self {
        Self::Geo
    }

    pub fn bucketed(buckets: impl IntoIterator<Item = BucketSpec>) -> Self {
        Self::Bucketed(BucketedDimensionBody {
            buckets: buckets.into_iter().collect(),
        })
    }

    pub fn metadata(source: MetadataSource) -> Self {
        Self::Metadata(MetadataDimensionBody { source })
    }
}

impl<S: dimension_builder::State> DimensionBuilder<S>
where
    S::DimType: dimension_builder::IsUnset,
{
    pub fn temporal(
        self,
        granularities: impl IntoIterator<Item = Granularity>,
    ) -> DimensionBuilder<dimension_builder::SetDimType<S>> {
        self.dim_type(DimensionType::temporal(granularities))
    }

    pub fn categorical(self) -> DimensionBuilder<dimension_builder::SetDimType<S>> {
        self.dim_type(DimensionType::categorical())
    }

    pub fn binary(self) -> DimensionBuilder<dimension_builder::SetDimType<S>> {
        self.dim_type(DimensionType::binary())
    }

    pub fn geo(self) -> DimensionBuilder<dimension_builder::SetDimType<S>> {
        self.dim_type(DimensionType::geo())
    }

    pub fn bucketed(
        self,
        buckets: impl IntoIterator<Item = BucketSpec>,
    ) -> DimensionBuilder<dimension_builder::SetDimType<S>> {
        self.dim_type(DimensionType::bucketed(buckets))
    }

    pub fn metadata(
        self,
        source: MetadataSource,
    ) -> DimensionBuilder<dimension_builder::SetDimType<S>> {
        self.dim_type(DimensionType::metadata(source))
    }
}

impl<'de> Deserialize<'de> for DimensionType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => match s.as_str() {
                "categorical" => Ok(Self::Categorical),
                "binary" => Ok(Self::Binary),
                "geo" => Ok(Self::Geo),
                "temporal" => Ok(Self::Temporal(TemporalDimensionBody::default())),
                "bucketed" => Err(serde::de::Error::custom(
                    "DimensionType::bucketed requires a body (buckets list)",
                )),
                "metadata" => Err(serde::de::Error::custom(
                    "DimensionType::metadata requires a body (source spec)",
                )),
                other => Err(serde::de::Error::custom(format!(
                    "DimensionType: unknown variant `{}`",
                    other
                ))),
            },
            Value::Mapping(_) => {
                let (key, body) = single_key_map::<D::Error>(value, "DimensionType")?;
                match key.as_str() {
                    "temporal" => {
                        let b: TemporalDimensionBody =
                            serde_yaml::from_value(body).map_err(serde::de::Error::custom)?;
                        Ok(Self::Temporal(b))
                    }
                    "bucketed" => {
                        let b: BucketedDimensionBody =
                            serde_yaml::from_value(body).map_err(serde::de::Error::custom)?;
                        Ok(Self::Bucketed(b))
                    }
                    "metadata" => {
                        let b: MetadataDimensionBody =
                            serde_yaml::from_value(body).map_err(serde::de::Error::custom)?;
                        Ok(Self::Metadata(b))
                    }
                    "categorical" | "binary" | "geo" => Err(serde::de::Error::custom(format!(
                        "DimensionType::{key} is body-less; use the bare-string form (`type: {key}`)"
                    ))),
                    other => Err(serde::de::Error::custom(format!(
                        "DimensionType: unknown variant `{other}`"
                    ))),
                }
            }
            other => Err(serde::de::Error::custom(format!(
                "DimensionType: expected string or mapping, got {:?}",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TemporalDimensionBody {
    #[serde(default)]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The granularities this Dimension rolls up to; empty where only the source granularity applies."
        )
    )]
    pub granularities: Vec<Granularity>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct BucketedDimensionBody {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The buckets, each from its inclusive lower bound to its exclusive upper bound."
        )
    )]
    pub buckets: Vec<BucketSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
pub struct BucketSpec {
    #[builder(start_fn, into)]
    pub name: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::model::serde::literal::bound"
    )]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            schema_with = "crate::model::serde::literal::bound::schema",
            description = "The bucket's inclusive lower bound, a tagged literal such as `{bigint: 100}`."
        )
    )]
    pub lower: Option<Literal>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::model::serde::literal::bound"
    )]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            schema_with = "crate::model::serde::literal::bound::schema",
            description = "The bucket's exclusive upper bound, a tagged literal."
        )
    )]
    pub upper: Option<Literal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct MetadataDimensionBody {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Where each physical source gives the value: a path token.")
    )]
    pub source: MetadataSource,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MetadataSource {
    Path(PathTokenRef),
}

impl Serialize for MetadataSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Path(body) => serialize_single_key(serializer, "path", body),
        }
    }
}

impl<'de> Deserialize<'de> for MetadataSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let (key, body) = single_key_map::<D::Error>(value, "MetadataSource")?;
        match key.as_str() {
            "path" => {
                let r: PathTokenRef =
                    serde_yaml::from_value(body).map_err(serde::de::Error::custom)?;
                Ok(Self::Path(r))
            }
            other => Err(serde::de::Error::custom(format!(
                "MetadataSource: unknown variant `{other}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct PathTokenRef {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The 0-indexed segment of the path that holds the value.")
    )]
    pub token: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(description = "A Dimension declared inline at a site, with the site's keyness.")
)]
#[non_exhaustive]
pub struct DimensionDecl {
    #[serde(flatten)]
    pub body: Dimension,

    #[serde(flatten)]
    pub keyness: Keyness,
}

impl From<Dimension> for DimensionDecl {
    fn from(body: Dimension) -> Self {
        Self {
            body,
            keyness: Keyness::default(),
        }
    }
}

impl<'de> Deserialize<'de> for DimensionDecl {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (keyness, body) = decode_keyed_site(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self { body, keyness })
    }
}

authoring_body!(
    named DimensionDecl, "Dimension",
    ["type", "is_key", "keyset", "keysets"], body.body
);

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "A `ref:` to a pooled Dimension, with the fields that a site may override and its keyness."
    )
)]
#[non_exhaustive]
pub struct DimensionRef {
    #[serde(flatten)]
    pub body: Reference,

    #[serde(flatten)]
    pub keyness: Keyness,
}

impl<'de> Deserialize<'de> for DimensionRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (keyness, body) = decode_keyed_site(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self { body, keyness })
    }
}

authoring_body!(
    reference DimensionRef, "Dimension reference site",
    ["is_key", "keyset", "keysets"], body
);

pub type DimensionSite = SemanticSite<DimensionDecl, DimensionRef>;

impl DimensionSite {
    pub fn r#ref(name: impl Into<SemanticsName>) -> Self {
        Self::Reference(DimensionRef {
            body: Reference::new(name),
            keyness: Keyness::default(),
        })
    }

    pub fn ref_with_expr(name: impl Into<SemanticsName>, expr: UnresolvedSemanticExpr) -> Self {
        Self::Reference(DimensionRef {
            body: Reference::with_expr(name, expr),
            keyness: Keyness::default(),
        })
    }

    pub fn keyness(&self) -> &Keyness {
        match self {
            Self::Inline(i) => &i.keyness,
            Self::Reference(r) => &r.keyness,
        }
    }

    fn keyness_mut(&mut self) -> &mut Keyness {
        match self {
            Self::Inline(i) => &mut i.keyness,
            Self::Reference(r) => &mut r.keyness,
        }
    }

    pub fn with_key(mut self) -> Self {
        *self.keyness_mut() = Keyness::key();
        self
    }

    pub fn with_keysets(
        mut self,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, Diagnostic<ModelBuildErrorKind>> {
        *self.keyness_mut() = Keyness::keysets_named(names)?;
        Ok(self)
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for DimensionType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DimensionType".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let temporal = serde_json::to_value(generator.subschema_for::<TemporalDimensionBody>())
            .expect("subschema to value");
        let bucketed = serde_json::to_value(generator.subschema_for::<BucketedDimensionBody>())
            .expect("subschema to value");
        let metadata = serde_json::to_value(generator.subschema_for::<MetadataDimensionBody>())
            .expect("subschema to value");
        serde_json::from_value(serde_json::json!({
            "oneOf": [
                { "type": "string", "enum": ["categorical", "binary", "geo"] },
                { "type": "object", "additionalProperties": false, "required": ["temporal"],
                  "properties": { "temporal": temporal } },
                { "type": "object", "additionalProperties": false, "required": ["bucketed"],
                  "properties": { "bucketed": bucketed } },
                { "type": "object", "additionalProperties": false, "required": ["metadata"],
                  "properties": { "metadata": metadata } }
            ]
        }))
        .expect("DimensionType JSON Schema is valid JSON")
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for MetadataSource {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "MetadataSource".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let path =
            serde_json::to_value(generator.subschema_for::<PathTokenRef>()).expect("subschema");
        serde_json::from_value(serde_json::json!({
            "type": "object", "additionalProperties": false, "required": ["path"],
            "properties": { "path": path }
        }))
        .expect("MetadataSource JSON Schema is valid JSON")
    }
}

#[cfg(test)]
mod authoring_serialize_tests {
    use super::*;

    #[track_caller]
    fn round_trip_dim_type(v: DimensionType) {
        let yaml = serde_yaml::to_string(&v).expect("serialize");
        assert!(!yaml.contains('!'), "no `!`-tag leak:\n{yaml}");
        let back: DimensionType = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(v, back, "round-trip mismatch via:\n{yaml}");
    }

    #[test]
    fn dimension_type_round_trips_all_variants() {
        round_trip_dim_type(DimensionType::categorical());
        round_trip_dim_type(DimensionType::binary());
        round_trip_dim_type(DimensionType::geo());
        round_trip_dim_type(DimensionType::temporal([
            Granularity::Day,
            Granularity::Month,
        ]));
        round_trip_dim_type(DimensionType::temporal([]));
        round_trip_dim_type(DimensionType::bucketed([
            BucketSpec {
                name: "low".into(),
                lower: None,
                upper: Some(Literal::BigInt(100)),
            },
            BucketSpec {
                name: "high".into(),
                lower: Some(Literal::Double(1.5)),
                upper: None,
            },
        ]));
        round_trip_dim_type(DimensionType::metadata(MetadataSource::Path(
            PathTokenRef { token: 2 },
        )));
    }

    #[test]
    fn metadata_source_round_trips() {
        let s = MetadataSource::Path(PathTokenRef { token: 0 });
        let yaml = serde_yaml::to_string(&s).expect("serialize");
        assert!(!yaml.contains('!'), "no `!`-tag leak:\n{yaml}");
        let back: MetadataSource = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(s, back, "round-trip mismatch via:\n{yaml}");
    }
}
