use crate::error::serde_message;

use crate::model::entity_btree_map::NamedEntity;
use crate::model::serde::data_type::UnresolvedDataType;
use crate::model::serde::{SemanticExpr, Unresolved};
use crate::model::types::entities::ai::AiContext;
use crate::UnresolvedSemanticExpr;
use semstrait_common::EntityId;
use semstrait_common::SemanticsName;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_yaml::Value;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct SemanticBody {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The entity's id, which keeps its identity across a rename.")
    )]
    pub id: Option<EntityId>,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The expression that derives this semantics; without one, it binds from `semantic_mapping`."
        )
    )]
    pub expr: Option<UnresolvedSemanticExpr>,
}

impl SemanticBody {
    pub fn entity_id(&self) -> Option<&str> {
        self.id.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct NamedSemanticBody {
    #[serde(flatten)]
    pub body: SemanticBody,

    pub name: SemanticsName,

    pub data_type: UnresolvedDataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,
}

impl NamedSemanticBody {
    pub fn new(name: impl Into<SemanticsName>, data_type: impl Into<UnresolvedDataType>) -> Self {
        Self {
            body: SemanticBody::default(),
            name: name.into(),
            data_type: data_type.into(),
            description: None,
            ai_context: None,
        }
    }

    pub fn entity_name(&self) -> &SemanticsName {
        &self.name
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Reference {
    #[serde(rename = "ref")]
    pub name: SemanticsName,

    #[serde(flatten)]
    pub body: SemanticBody,
}

impl Reference {
    pub fn new(name: impl Into<SemanticsName>) -> Self {
        Self {
            name: name.into(),
            body: SemanticBody::default(),
        }
    }

    pub fn with_expr(name: impl Into<SemanticsName>, expr: UnresolvedSemanticExpr) -> Self {
        Self {
            name: name.into(),
            body: SemanticBody {
                id: None,
                expr: Some(expr),
            },
        }
    }

    pub fn entity_name(&self) -> &SemanticsName {
        &self.name
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "An inline declaration with `name:`, or a `ref:` to a semantics declared at the model's root."
    )
)]
#[serde(untagged)]
#[non_exhaustive]
pub enum SemanticSite<I, R> {
    Inline(I),
    Reference(R),
}

impl<I, R> SemanticSite<I, R> {
    pub fn inline(body: impl Into<I>) -> Self {
        Self::Inline(body.into())
    }
}

impl<I: AuthoringBody, R: AuthoringBody> SemanticSite<I, R> {
    pub fn name(&self) -> &SemanticsName {
        match self {
            Self::Inline(i) => i.semantics_name(),
            Self::Reference(r) => r.semantics_name(),
        }
    }

    pub fn body(&self) -> &SemanticBody {
        match self {
            Self::Inline(i) => i.body(),
            Self::Reference(r) => r.body(),
        }
    }

    pub fn body_mut(&mut self) -> &mut SemanticBody {
        match self {
            Self::Inline(i) => i.body_mut(),
            Self::Reference(r) => r.body_mut(),
        }
    }
}

impl<I: AuthoringBody, R: AuthoringBody> NamedEntity for SemanticSite<I, R> {
    fn entity_id(&self) -> Option<&str> {
        self.body().id.as_deref()
    }
    fn entity_name(&self) -> &str {
        self.name().as_str()
    }
}

impl<'de, I: AuthoringBody, R: AuthoringBody> Deserialize<'de> for SemanticSite<I, R> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;

        let value = Value::deserialize(deserializer)?;
        let Value::Mapping(map) = &value else {
            return Err(D::Error::custom(
                "a semantics site must be a mapping (`name:` … or `ref:` …)",
            ));
        };
        let has_ref = map.contains_key(Value::String("ref".into()));
        let has_name = map.contains_key(Value::String("name".into()));
        if has_ref && has_name {
            return Err(D::Error::custom(serde_message::site_name_with_ref(
                I::CARRIER,
            )));
        }

        if has_ref {
            Ok(Self::Reference(
                serde_yaml::from_value(value).map_err(D::Error::custom)?,
            ))
        } else {
            Ok(Self::Inline(
                serde_yaml::from_value(value).map_err(D::Error::custom)?,
            ))
        }
    }
}

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub trait AuthoringBody: DeserializeOwned + sealed::Sealed {
    const CARRIER: &'static str;
    const WIRE_FIELDS: &'static [&'static str];

    fn body(&self) -> &SemanticBody;
    fn body_mut(&mut self) -> &mut SemanticBody;
    fn semantics_name(&self) -> &SemanticsName;
}

pub(crate) fn decode_gated<'de, D, T>(
    deserializer: D,
    carrier: &str,
    fields: &'static [&'static str],
) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    use serde::de::Error as _;

    let value = gated_value::<D>(deserializer, carrier, fields)?;
    serde_yaml::from_value(value).map_err(D::Error::custom)
}

pub(crate) fn gated_value<'de, D: Deserializer<'de>>(
    deserializer: D,
    carrier: &str,
    fields: &'static [&'static str],
) -> Result<Value, D::Error> {
    let value = Value::deserialize(deserializer)?;
    reject_unknown_keys::<D::Error>(&value, carrier, fields)?;
    Ok(value)
}

pub(crate) fn reject_unknown_keys<E: serde::de::Error>(
    value: &Value,
    carrier: &str,
    fields: &'static [&'static str],
) -> Result<(), E> {
    let Value::Mapping(map) = value else {
        return Err(E::custom(format!("{carrier} must be a mapping")));
    };
    for key in map.keys() {
        let Some(key) = key.as_str() else {
            return Err(E::custom(format!("{carrier}: keys must be strings")));
        };
        if !fields.contains(&key) {
            return Err(E::custom(serde_message::unknown_field(
                key, carrier, fields,
            )));
        }
    }
    Ok(())
}

macro_rules! authoring_body {
    (named $t:ty, $carrier:literal, [$($extra:literal),* $(,)?], $($base:ident).+) => {
        $crate::model::types::entities::semantic_body::authoring_body!(
            @impl $t, $carrier,
            ["id", "expr", "name", "data_type", "description", "ai_context" $(, $extra)*],
            $($base).+
        );

        impl $t {
            pub fn data_type(&self) -> &$crate::UnresolvedDataType {
                &self.$($base).+.data_type
            }
        }
    };
    (reference $t:ty, $carrier:literal, [$($extra:literal),* $(,)?], $($base:ident).+) => {
        $crate::model::types::entities::semantic_body::authoring_body!(
            @impl $t, $carrier, ["ref", "id", "expr" $(, $extra)*], $($base).+
        );
    };
    (@impl $t:ty, $carrier:literal, [$($field:literal),+ $(,)?], $($base:ident).+) => {
        impl $t {
            pub fn name(&self) -> &::semstrait_common::SemanticsName {
                self.$($base).+.entity_name()
            }

            pub fn entity_id(&self) -> Option<&str> {
                self.$($base).+.body.id.as_deref()
            }

            pub fn id_mut(&mut self) -> &mut Option<::semstrait_common::EntityId> {
                &mut self.$($base).+.body.id
            }

            pub fn expr(
                &self,
            ) -> Option<&$crate::UnresolvedSemanticExpr> {
                self.$($base).+.body.expr.as_ref()
            }
        }

        impl $crate::model::types::entities::semantic_body::AuthoringBody for $t {
            const CARRIER: &'static str = $carrier;
            const WIRE_FIELDS: &'static [&'static str] = &[$($field),+];

            fn body(&self) -> &$crate::model::types::entities::semantic_body::SemanticBody {
                &self.$($base).+.body
            }
            fn body_mut(
                &mut self,
            ) -> &mut $crate::model::types::entities::semantic_body::SemanticBody {
                &mut self.$($base).+.body
            }
            fn semantics_name(&self) -> &::semstrait_common::SemanticsName {
                self.$($base).+.entity_name()
            }
        }

        impl $crate::model::types::entities::semantic_body::sealed::Sealed for $t {}

        impl $crate::model::entity_btree_map::NamedEntity for $t {
            fn entity_id(&self) -> Option<&str> {
                self.$($base).+.body.entity_id()
            }
            fn entity_name(&self) -> &str {
                self.$($base).+.entity_name().as_str()
            }
        }
    };
}
pub(crate) use authoring_body;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::dim;
    use crate::model::types::entities::agg::Agg;
    use crate::model::types::entities::dimension::{
        Dimension, DimensionDecl, DimensionRef, DimensionType,
    };
    use crate::model::types::entities::measure::{Measure, MeasureRef};
    use crate::model::types::entities::metric::{Metric, MetricRef};
    use std::collections::BTreeSet;

    fn populated_base() -> NamedSemanticBody {
        NamedSemanticBody {
            body: SemanticBody {
                id: Some("id".to_owned()),
                expr: dim("other"),
            },
            name: SemanticsName("full".into()),
            data_type: semstrait_sql::datatype::DataType::String.into(),
            description: Some("d".to_owned()),
            ai_context: Some(AiContext::default()),
        }
    }

    fn populated_ref() -> Reference {
        Reference {
            name: SemanticsName("target".into()),
            body: SemanticBody {
                id: Some("id".to_owned()),
                expr: dim("other"),
            },
        }
    }

    fn keyness() -> crate::model::types::entities::keyness::Keyness {
        crate::model::types::entities::keyness::Keyness::keysets_named(["natural"])
            .expect("named keysets")
    }

    #[track_caller]
    fn assert_wire_fields<T: AuthoringBody + Serialize>(fully_populated: &T, accept_only: &[&str]) {
        let value = serde_yaml::to_value(fully_populated).expect("serialize");
        let mut keys: BTreeSet<String> = value
            .as_mapping()
            .expect("body serializes as a mapping")
            .keys()
            .map(|k| k.as_str().expect("string key").to_owned())
            .collect();
        keys.extend(accept_only.iter().map(|f| (*f).to_owned()));
        let declared: BTreeSet<String> = T::WIRE_FIELDS.iter().map(|f| (*f).to_owned()).collect();
        assert_eq!(
            keys,
            declared,
            "{}: WIRE_FIELDS drifted from the serialized key set",
            T::CARRIER
        );
    }

    #[track_caller]
    fn assert_rejects_bogus_key<T: DeserializeOwned>(label: &str, json: &str) {
        match serde_json::from_str::<T>(json) {
            Ok(_) => panic!("{label}: the unknown key was accepted"),
            Err(e) => assert!(
                e.to_string().contains("unknown field `bogus`"),
                "{label}: expected an unknown-field rejection, got: {e}"
            ),
        }
    }

    #[test]
    fn the_unknown_key_gate_holds_at_the_raw_decode_seam() {
        const DIMENSION: &str =
            r#"{"name":"region","data_type":"string","type":"categorical","bogus":1}"#;
        const MEASURE: &str = r#"{"name":"revenue","data_type":"double","agg":"sum","bogus":1}"#;
        assert_rejects_bogus_key::<Dimension>("bare Dimension", DIMENSION);
        assert_rejects_bogus_key::<Measure>("bare Measure", MEASURE);
        assert_rejects_bogus_key::<crate::SemanticModel>(
            "pooled Dimension in a SemanticModel",
            &format!(r#"{{"name":"m","dimensions":[{DIMENSION}]}}"#),
        );
        assert_rejects_bogus_key::<crate::SemanticModel>(
            "pooled Measure in a SemanticModel",
            &format!(r#"{{"name":"m","measures":[{MEASURE}]}}"#),
        );
    }

    #[test]
    fn wire_fields_match_the_serialized_key_set() {
        let dim = Dimension {
            body: populated_base(),
            dim_type: DimensionType::Categorical,
        };
        assert_wire_fields(&dim, &[]);
        assert_wire_fields(
            &DimensionDecl {
                body: dim.clone(),
                keyness: keyness(),
            },
            &["keyset"],
        );
        assert_wire_fields(
            &DimensionRef {
                body: populated_ref(),
                keyness: keyness(),
            },
            &["keyset"],
        );
        assert_wire_fields(
            &Measure {
                body: populated_base(),
                agg: Agg::of_name("sum").expect("a catalog aggregate"),
                additivity: Some(crate::model::types::entities::measure::AdditivityType::Full),
                filter: crate::expr::dim("other"),
            },
            &[],
        );
        assert_wire_fields(
            &MeasureRef {
                body: populated_ref(),
                filter: crate::expr::dim("other"),
            },
            &[],
        );
        assert_wire_fields(
            &Metric {
                body: populated_base(),
                agg: Some(Agg::of_name("sum").expect("a catalog aggregate")),
                filter: crate::expr::dim("other"),
            },
            &[],
        );
        assert_wire_fields(
            &MetricRef {
                body: populated_ref(),
                filter: crate::expr::dim("other"),
            },
            &[],
        );
    }
}
