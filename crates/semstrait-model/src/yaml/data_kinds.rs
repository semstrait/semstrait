use crate::model::entity_btree_map::{reject_duplicate_authored_ids, NamedEntity};
use crate::model::identity::{IdentityProfile, MintLedger};
use crate::model::types::data_kinds::{
    Dataset, DatasetBody, Grainset, GrainsetBody, Joinset, JoinsetBody, Unionset, UnionsetBody,
};
use crate::model::types::entities::ai::AiContext;
use crate::model::types::entities::dimension::DimensionSite;
use crate::model::types::entities::filter::DataKindFilter;
use crate::model::types::entities::measure::MeasureSite;
use crate::model::types::entities::metric::MetricSite;
use crate::model::types::entities::semantic_interface::{resolve_collection, SemanticInterface};
use semstrait_common::diagnostic::Location;
use serde::{Deserialize, Deserializer};

#[derive(Debug, Default, Deserialize)]
pub(crate) struct YamlSemanticInterface {
    #[serde(default, deserialize_with = "deserialize_unminted")]
    dimensions: Vec<DimensionSite>,

    #[serde(default, deserialize_with = "deserialize_unminted")]
    measures: Vec<MeasureSite>,

    #[serde(default, deserialize_with = "deserialize_unminted")]
    metrics: Vec<MetricSite>,

    #[serde(default, deserialize_with = "deserialize_unminted")]
    filters: Vec<DataKindFilter>,
}

impl YamlSemanticInterface {
    fn resolve(
        self,
        data_kind: &str,
        profile: IdentityProfile,
        location: &Location,
        ledger: &mut MintLedger,
    ) -> SemanticInterface {
        let Self {
            dimensions,
            measures,
            metrics,
            filters,
        } = self;
        SemanticInterface {
            dimensions: resolve_collection(
                dimensions,
                data_kind,
                "dimensions",
                profile,
                location,
                ledger,
            ),
            measures: resolve_collection(
                measures, data_kind, "measures", profile, location, ledger,
            ),
            metrics: resolve_collection(metrics, data_kind, "metrics", profile, location, ledger),
            filters: resolve_collection(filters, data_kind, "filters", profile, location, ledger),
        }
    }
}

fn deserialize_unminted<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + NamedEntity,
{
    let items = Vec::<T>::deserialize(deserializer)?;
    reject_duplicate_authored_ids::<T, D::Error>(&items)?;
    Ok(items)
}

macro_rules! yaml_public_data_kind {
    ($wire:ident => $kind:ident, $body:ty) => {
        #[derive(Debug, Deserialize)]
        pub(crate) struct $wire {
            #[serde(flatten)]
            body: $body,

            #[serde(default)]
            description: Option<String>,

            #[serde(default)]
            ai_context: Option<AiContext>,

            #[serde(flatten, default)]
            interface: YamlSemanticInterface,
        }

        impl $wire {
            pub(crate) fn resolve(
                self,
                profile: IdentityProfile,
                location: &Location,
                ledger: &mut MintLedger,
            ) -> $kind {
                let Self {
                    body,
                    description,
                    ai_context,
                    interface,
                } = self;
                let semantic_interface =
                    interface.resolve(&body.base.name, profile, location, ledger);
                $kind {
                    body,
                    description,
                    ai_context,
                    semantic_interface,
                }
            }
        }
    };
}

yaml_public_data_kind!(YamlDataset => Dataset, DatasetBody);
yaml_public_data_kind!(YamlGrainset => Grainset, GrainsetBody);
yaml_public_data_kind!(YamlUnionset => Unionset, UnionsetBody);
yaml_public_data_kind!(YamlJoinset => Joinset, JoinsetBody);
