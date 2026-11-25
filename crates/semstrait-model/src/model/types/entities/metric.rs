use crate::model::serde::{SemanticExpr, Unresolved};
use crate::model::types::entities::agg::Agg;
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::semantic_body::{
    authoring_body, decode_gated, AuthoringBody, NamedSemanticBody, Reference, SemanticBody,
    SemanticSite,
};
use crate::UnresolvedDataType;
use crate::UnresolvedSemanticExpr;
use bon::bon;
use semstrait_common::identity::{mint_named_entity_id, EntityNamespace};
use semstrait_common::SemanticsName;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "A Metric of the root `metrics:` pool, or one declared inline at a site."
    )
)]
#[non_exhaustive]
pub struct Metric {
    #[serde(flatten)]
    pub body: NamedSemanticBody,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The aggregate function of the Metric; without one, its expression carries the aggregation."
        )
    )]
    pub agg: Option<Agg>,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "The predicate that keeps a row in the Metric's aggregate.")
    )]
    pub filter: Option<UnresolvedSemanticExpr>,
}

#[derive(Deserialize)]
struct MetricWire {
    #[serde(flatten)]
    body: NamedSemanticBody,
    #[serde(default)]
    agg: Option<Agg>,
    #[serde(default, with = "Unresolved::<SemanticExpr>")]
    filter: Option<UnresolvedSemanticExpr>,
}

impl<'de> Deserialize<'de> for Metric {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let MetricWire { body, agg, filter } =
            decode_gated(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self { body, agg, filter })
    }
}

#[bon]
impl Metric {
    #[builder(
        builder_type(name = MetricBuilder, vis = "pub"),
        state_mod(name = metric_builder, vis = "pub"),
        finish_fn = build,
    )]
    pub fn new(
        #[builder(start_fn, into)] name: SemanticsName,
        #[builder(into)] data_type: UnresolvedDataType,
        agg: Option<Agg>,
        #[builder(into)] description: Option<String>,
        ai_context: Option<AiContext>,
        #[builder(into)] expr: Option<UnresolvedSemanticExpr>,
        #[builder(into)] filter: Option<UnresolvedSemanticExpr>,
    ) -> Self {
        Metric {
            body: NamedSemanticBody {
                body: SemanticBody {
                    id: Some(mint_named_entity_id(EntityNamespace::Metric, name.as_str())),
                    expr,
                },
                name,
                data_type,
                description,
                ai_context,
            },
            agg,
            filter,
        }
    }
}

authoring_body!(named Metric, "Metric", ["agg", "filter"], body);

#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(
    feature = "schemars",
    derive(schemars::JsonSchema),
    schemars(
        description = "A `ref:` to a pooled Metric, with the `expr` and `filter` that a site may override."
    )
)]
#[non_exhaustive]
pub struct MetricRef {
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
struct MetricRefWire {
    #[serde(flatten)]
    body: Reference,
    #[serde(default, with = "Unresolved::<SemanticExpr>")]
    filter: Option<UnresolvedSemanticExpr>,
}

impl<'de> Deserialize<'de> for MetricRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let MetricRefWire { body, filter } =
            decode_gated(deserializer, Self::CARRIER, Self::WIRE_FIELDS)?;
        Ok(Self { body, filter })
    }
}

authoring_body!(reference MetricRef, "Metric reference site", ["filter"], body);

pub type MetricSite = SemanticSite<Metric, MetricRef>;

impl MetricSite {
    pub fn r#ref(name: impl Into<SemanticsName>) -> Self {
        Self::Reference(MetricRef {
            body: Reference::new(name),
            filter: None,
        })
    }

    pub fn ref_with_expr(name: impl Into<SemanticsName>, expr: UnresolvedSemanticExpr) -> Self {
        Self::Reference(MetricRef {
            body: Reference::with_expr(name, expr),
            filter: None,
        })
    }
}
