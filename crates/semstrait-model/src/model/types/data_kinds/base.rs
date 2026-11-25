use crate::model::types::data_kinds::storage::{
    CatalogRef, PartitionDef, StorageConfig, StorageFormat,
};
use crate::model::types::entities::mapping::SemanticMapping;
use crate::model::types::entities::temporal::TemporalShape;
use bon::Builder;
use semstrait_common::identity::{mint_named_entity_id, EntityNamespace};
use semstrait_common::EntityId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "A data kind: its `name:`, its `extras:` and the fields its kind adds."
    )
)]
#[non_exhaustive]
pub struct DataKindBase<E: ExtrasFlavor> {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The data kind's id, which keeps its identity across a rename; generated when omitted."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<EntityId>,

    pub name: String,

    #[serde(
        default = "<E as Default>::default",
        skip_serializing_if = "ExtrasFlavor::is_default"
    )]
    pub extras: E,
}

impl<E: ExtrasFlavor> DataKindBase<E> {
    #[must_use]
    pub fn rooted(name: String, extras: E) -> Self {
        Self {
            id: Some(mint_named_entity_id(EntityNamespace::DataKind, &name)),
            name,
            extras,
        }
    }

    #[must_use]
    pub fn nested(name: String, extras: E) -> Self {
        Self {
            id: None,
            name,
            extras,
        }
    }
}

pub trait ExtrasFlavor: Default + PartialEq {
    fn is_default(&self) -> bool;
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "The `extras:` block for a Dataset: catalog or storage source, `semantic_mapping:` and temporal shape."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
pub struct LeafExtras {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog: Option<CatalogRef>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageConfig>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "How semantic names map to physical columns; `auto` when omitted.")
    )]
    #[serde(default, skip_serializing_if = "is_default_mapping")]
    #[builder(default)]
    pub semantic_mapping: SemanticMapping,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalShape>,
}

fn is_default_mapping(m: &SemanticMapping) -> bool {
    m.is_auto()
}

impl ExtrasFlavor for LeafExtras {
    fn is_default(&self) -> bool {
        self.catalog.is_none()
            && self.storage.is_none()
            && self.semantic_mapping.is_auto()
            && self.temporal.is_none()
    }
}

impl LeafExtras {
    pub fn with_catalog(mut self, c: impl Into<String>) -> Self {
        self.catalog = Some(CatalogRef::new(c));
        self
    }
    pub fn with_storage(mut self, s: StorageConfig) -> Self {
        self.storage = Some(s);
        self
    }
    pub fn with_format(mut self, f: StorageFormat) -> Self {
        self.storage.get_or_insert_with(Default::default).format = Some(f);
        self
    }
    pub fn with_path(mut self, p: impl Into<String>) -> Self {
        self.storage
            .get_or_insert_with(Default::default)
            .paths
            .push(p.into());
        self
    }
    pub fn with_paths(mut self, items: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.storage
            .get_or_insert_with(Default::default)
            .paths
            .extend(items.into_iter().map(Into::into));
        self
    }
    pub fn with_table(mut self, t: impl Into<String>) -> Self {
        self.storage
            .get_or_insert_with(Default::default)
            .tables
            .push(t.into());
        self
    }
    pub fn with_tables(mut self, items: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.storage
            .get_or_insert_with(Default::default)
            .tables
            .extend(items.into_iter().map(Into::into));
        self
    }
    pub fn with_partition_def(mut self, entry: PartitionDef) -> Self {
        self.storage
            .get_or_insert_with(Default::default)
            .partition_def
            .push(entry);
        self
    }
    pub fn with_semantic_mapping(mut self, m: SemanticMapping) -> Self {
        self.semantic_mapping = m;
        self
    }
    pub fn with_temporal(mut self, t: TemporalShape) -> Self {
        self.temporal = Some(t);
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "The `extras:` block for a Grainset, Unionset or Joinset: only a temporal shape, no storage fields."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
pub struct ComplexExtras {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalShape>,
}

impl ExtrasFlavor for ComplexExtras {
    fn is_default(&self) -> bool {
        self.temporal.is_none()
    }
}

impl ComplexExtras {
    pub fn with_temporal(mut self, t: TemporalShape) -> Self {
        self.temporal = Some(t);
        self
    }
}
