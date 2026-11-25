use crate::model::builder::SemanticModelBuilder;
use crate::model::types::entities::{AiContext, Dimension, Measure, Metric, Relationship};
use crate::yaml::data_kinds::{YamlDataset, YamlGrainset, YamlJoinset, YamlUnionset};
use semstrait_common::diagnostic::{Location, SourceId, Span};
use semstrait_common::EntityId;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct YamlRoot {
    pub(crate) semantic_model: YamlSemanticModel,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct YamlSemanticModel {
    #[serde(default)]
    pub(crate) id: Option<EntityId>,

    pub(crate) name: String,

    #[serde(default)]
    pub(crate) description: Option<String>,

    #[serde(default)]
    pub(crate) ai_context: Option<AiContext>,

    #[serde(default)]
    pub(crate) labels: Vec<String>,

    #[serde(default)]
    pub(crate) datasets: Vec<YamlDataset>,

    #[serde(default)]
    pub(crate) grainsets: Vec<YamlGrainset>,

    #[serde(default)]
    pub(crate) unionsets: Vec<YamlUnionset>,

    #[serde(default)]
    pub(crate) joinsets: Vec<YamlJoinset>,

    #[serde(default)]
    pub(crate) dimensions: Vec<Dimension>,

    #[serde(default)]
    pub(crate) measures: Vec<Measure>,

    #[serde(default)]
    pub(crate) metrics: Vec<Metric>,

    #[serde(default)]
    pub(crate) relationships: Vec<Relationship>,
}

impl YamlRoot {
    pub(crate) fn lower_into(
        self,
        source: &str,
        mut builder: SemanticModelBuilder,
    ) -> SemanticModelBuilder {
        let YamlRoot {
            semantic_model:
                YamlSemanticModel {
                    id,
                    name,
                    description,
                    ai_context,
                    labels,
                    datasets,
                    grainsets,
                    unionsets,
                    joinsets,
                    dimensions,
                    measures,
                    metrics,
                    relationships,
                },
        } = self;

        builder = builder.name(name);
        if let Some(id) = id {
            builder = builder.id(id);
        }
        if let Some(d) = description {
            builder = builder.description(d);
        }
        if let Some(a) = ai_context {
            builder = builder.ai_context(a);
        }
        for label in labels {
            builder = builder.label(label);
        }

        let loc = Location::new(SourceId::named(source), Span::unknown());

        for d in datasets {
            builder = builder.dataset_at(d, loc.clone());
        }
        for g in grainsets {
            builder = builder.grainset_at(g, loc.clone());
        }
        for u in unionsets {
            builder = builder.unionset_at(u, loc.clone());
        }
        for j in joinsets {
            builder = builder.joinset_at(j, loc.clone());
        }

        for d in dimensions {
            builder = builder.dimension_at(d, loc.clone());
        }
        for m in measures {
            builder = builder.measure_at(m, loc.clone());
        }
        for m in metrics {
            builder = builder.metric_at(m, loc.clone());
        }

        for r in relationships {
            builder = builder.relationship_at(r, loc.clone());
        }

        builder
    }
}
