pub mod builder;
pub mod entity_btree_map;
pub mod identity;
pub mod serde;
pub mod types;

use crate::model::entity_btree_map::{EntityBTreeMap, NamedEntity};
use crate::model::types::data_kinds::{
    AnyDataKindRef, ComplexDataKindRef, DataKindVariant, Dataset, Grainset, Joinset,
    PublicDataKindRef, SimpleDataKindRef, Unionset,
};
use crate::model::types::entities::{AiContext, Dimension, Measure, Metric, Relationship};
use ::serde::{Deserialize, Serialize};
use semstrait_common::EntityId;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "A semstrait model: its data kinds, Dimensions, Measures, Metrics and relationships."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SemanticModel {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The model's id, which keeps its identity across a rename; derived from `name` if absent."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<EntityId>,

    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub datasets: EntityBTreeMap<Dataset>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub grainsets: EntityBTreeMap<Grainset>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub unionsets: EntityBTreeMap<Unionset>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub joinsets: EntityBTreeMap<Joinset>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub dimensions: EntityBTreeMap<Dimension>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub measures: EntityBTreeMap<Measure>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub metrics: EntityBTreeMap<Metric>,

    #[serde(
        default,
        skip_serializing_if = "EntityBTreeMap::is_empty",
        deserialize_with = "crate::model::serde::digits::deserialize"
    )]
    pub relationships: EntityBTreeMap<Relationship>,
}

impl SemanticModel {
    pub fn public_data_kind_count(&self) -> usize {
        self.datasets.len() + self.grainsets.len() + self.unionsets.len() + self.joinsets.len()
    }

    pub fn is_empty_model(&self) -> bool {
        self.public_data_kind_count() == 0
    }

    pub fn iter_all(&self) -> impl Iterator<Item = AnyDataKindRef<'_>> {
        let datasets = self.datasets_by_name().map(AnyDataKindRef::Dataset);
        let grainsets = self.grainsets_by_name().map(AnyDataKindRef::Grainset);
        let unionsets = self.unionsets_by_name().map(AnyDataKindRef::Unionset);
        let joinsets = self.joinsets_by_name().map(AnyDataKindRef::Joinset);
        datasets.chain(grainsets).chain(unionsets).chain(joinsets)
    }

    pub fn iter_public(&self) -> impl Iterator<Item = PublicDataKindRef<'_>> {
        let datasets = self.datasets_by_name().map(PublicDataKindRef::Dataset);
        let grainsets = self.grainsets_by_name().map(PublicDataKindRef::Grainset);
        let unionsets = self.unionsets_by_name().map(PublicDataKindRef::Unionset);
        let joinsets = self.joinsets_by_name().map(PublicDataKindRef::Joinset);
        datasets.chain(grainsets).chain(unionsets).chain(joinsets)
    }

    pub fn iter_simple(&self) -> impl Iterator<Item = SimpleDataKindRef<'_>> {
        self.datasets_by_name().map(SimpleDataKindRef::Public)
    }

    pub fn iter_complex(&self) -> impl Iterator<Item = ComplexDataKindRef<'_>> {
        let grainsets = self.grainsets_by_name().map(ComplexDataKindRef::Grainset);
        let unionsets = self.unionsets_by_name().map(ComplexDataKindRef::Unionset);
        let joinsets = self.joinsets_by_name().map(ComplexDataKindRef::Joinset);
        grainsets.chain(unionsets).chain(joinsets)
    }

    pub fn find_public(&self, name: &str) -> Option<PublicDataKindRef<'_>> {
        if let Some(d) = self.datasets.values().find(|d| d.body.base.name == name) {
            return Some(PublicDataKindRef::Dataset(d));
        }
        if let Some(g) = self.grainsets.values().find(|g| g.body.base.name == name) {
            return Some(PublicDataKindRef::Grainset(g));
        }
        if let Some(u) = self.unionsets.values().find(|u| u.body.base.name == name) {
            return Some(PublicDataKindRef::Unionset(u));
        }
        if let Some(j) = self.joinsets.values().find(|j| j.body.base.name == name) {
            return Some(PublicDataKindRef::Joinset(j));
        }
        None
    }

    pub fn variant_of(&self, name: &str) -> Option<DataKindVariant> {
        self.find_public(name).map(|r| match r {
            PublicDataKindRef::Dataset(_) => DataKindVariant::Dataset,
            PublicDataKindRef::Grainset(_) => DataKindVariant::Grainset,
            PublicDataKindRef::Unionset(_) => DataKindVariant::Unionset,
            PublicDataKindRef::Joinset(_) => DataKindVariant::Joinset,
        })
    }

    fn datasets_by_name(&self) -> impl Iterator<Item = &Dataset> {
        by_name(&self.datasets)
    }
    fn grainsets_by_name(&self) -> impl Iterator<Item = &Grainset> {
        by_name(&self.grainsets)
    }
    fn unionsets_by_name(&self) -> impl Iterator<Item = &Unionset> {
        by_name(&self.unionsets)
    }
    fn joinsets_by_name(&self) -> impl Iterator<Item = &Joinset> {
        by_name(&self.joinsets)
    }
}

fn by_name<T: NamedEntity>(map: &EntityBTreeMap<T>) -> impl Iterator<Item = &T> {
    let mut values: Vec<&T> = map.values().collect();
    values.sort_by(|a, b| a.entity_name().cmp(b.entity_name()));
    values.into_iter()
}

macro_rules! named_via_base {
    ($($t:ty),+ $(,)?) => {$(
        impl NamedEntity for $t {
            fn entity_id(&self) -> Option<&str> { self.body.base.id.as_deref() }
            fn entity_name(&self) -> &str { &self.body.base.name }
        }
    )+};
}
named_via_base!(Dataset, Grainset, Unionset, Joinset);

impl NamedEntity for Relationship {
    fn entity_id(&self) -> Option<&str> {
        self.id.as_deref()
    }
    fn entity_name(&self) -> &str {
        &self.name
    }
}
