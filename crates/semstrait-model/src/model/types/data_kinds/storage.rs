use bon::Builder;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(description = "A reference to a catalog entry, authored as a bare alias string.")
)]
#[serde(transparent)]
#[non_exhaustive]
pub struct CatalogRef {
    pub alias: String,
}

impl CatalogRef {
    pub fn new(alias: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "The file format for path-based sources; omitted when the source resolves through a catalog."
    )
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StorageFormat {
    Parquet,
    Csv,
    Json,
    Orc,
    Avro,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Builder)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "The `storage:` block: file, folder or glob URIs, catalog tables, and an optional partition layout."
    )
)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
#[builder(start_fn = builder, finish_fn = build)]
pub struct StorageConfig {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The storage format for path-based sources; required when `paths` is set, ignored for `tables`."
        )
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<StorageFormat>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "File, folder or glob URIs for the source data.")
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default)]
    pub paths: Vec<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Catalog table names or name globs for the source data.")
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default)]
    pub tables: Vec<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "The partition claims, in any order: each names the Dimension mapping one partition key."
        )
    )]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default)]
    pub partition_def: Vec<PartitionDef>,
}

impl StorageConfig {
    pub fn is_empty(&self) -> bool {
        self.format.is_none()
            && self.paths.is_empty()
            && self.tables.is_empty()
            && self.partition_def.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        description = "One partition claim: the Dimension a partition key maps to, and how that key partitions the data."
    )
)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum PartitionDef {
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "Range-partitioned: the claim names no values.")
    )]
    Range { dimension: String },
    #[cfg_attr(
        feature = "schemars",
        schemars(description = "List-partitioned: each discovered value must be one of `values`.")
    )]
    List {
        dimension: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        values: Vec<String>,
    },
}

impl PartitionDef {
    pub fn dimension(&self) -> &str {
        match self {
            Self::Range { dimension } | Self::List { dimension, .. } => dimension,
        }
    }

    pub fn values(&self) -> Option<&[String]> {
        match self {
            Self::Range { .. } => None,
            Self::List { values, .. } => Some(values),
        }
    }
}
