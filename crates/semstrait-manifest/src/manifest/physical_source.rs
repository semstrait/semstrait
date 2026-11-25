use semstrait_catalog::{FileFormat, Partition, TableRef};
use semstrait_common::io::Path;
use semstrait_common::str_newtype;
use semstrait_common::EntityId;
use semstrait_sql::datatype::DataType;
use serde::{Deserialize, Serialize};

use super::source_alias::SourceAlias;
use crate::identity::{mint_path_source_id, mint_table_source_id};

str_newtype!(Uri);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NotNormal {
    Relative,
    Scheme(String),
}

const SCHEMES: [(&str, &str); 4] = [("s3", "s3"), ("s3a", "s3"), ("s3n", "s3"), ("file", "file")];

const SCHEME_SEPARATOR: &str = "://";

impl Uri {
    pub(crate) fn normal(authored: &str) -> Result<Self, NotNormal> {
        let (scheme, rest) = match authored.split_once(SCHEME_SEPARATOR) {
            Some((written, rest)) => {
                let normal = SCHEMES.iter().find(|(scheme, _)| *scheme == written);
                let (_, normal) = normal.ok_or_else(|| NotNormal::Scheme(written.to_owned()))?;
                (*normal, rest)
            }
            None if authored.starts_with('/') => ("file", authored),
            None => return Err(NotNormal::Relative),
        };
        let body = rest.trim_end_matches('/');
        let folder = if body.len() < rest.len() { "/" } else { "" };
        Ok(Self(format!("{scheme}{SCHEME_SEPARATOR}{body}{folder}")))
    }

    #[must_use]
    pub fn is_normal(&self) -> bool {
        Self::normal(self.as_str()).is_ok_and(|normal| normal == *self)
    }

    pub fn segments(&self) -> impl Iterator<Item = &str> {
        let rest = (self.0.split_once(SCHEME_SEPARATOR)).map_or(self.as_str(), |(_, rest)| rest);
        rest.split('/').filter(|segment| !segment.is_empty())
    }

    pub(crate) fn beneath(&self, base_path: &Uri) -> Option<&str> {
        (self.as_str().strip_prefix(base_path.as_str())).filter(|rest| !rest.is_empty())
    }
}

str_newtype!(SchemaId);

mod opt_path_serde {
    use semstrait_common::io::Path;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(path: &Option<Path>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match path {
            Some(path) => serializer.serialize_some(path.as_str()),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<Path>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<String>::deserialize(deserializer)?.map(Path::new))
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum StorageTarget {
    Path {
        base_path: Uri,
        listings: Vec<Uri>,
        partition_def: Option<HivePartitions>,
    },
    Table {
        table: TableRef,
        #[serde(with = "opt_path_serde")]
        location: Option<Path>,
    },
}

impl StorageTarget {
    #[must_use]
    pub fn path(base_path: Uri, listings: Vec<Uri>) -> Self {
        Self::Path {
            base_path,
            listings,
            partition_def: None,
        }
    }

    #[must_use]
    pub fn table(table: TableRef) -> Self {
        Self::Table {
            table,
            location: None,
        }
    }

    #[must_use]
    pub fn table_at(table: TableRef, location: Path) -> Self {
        Self::Table {
            table,
            location: Some(location),
        }
    }

    #[must_use]
    pub fn table_ref(&self) -> Option<&TableRef> {
        match self {
            Self::Table { table, .. } => Some(table),
            Self::Path { .. } => None,
        }
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Path { base_path, .. } => base_path.segments().last().unwrap_or_else(|| {
                let path = base_path.as_str();
                path.split_inclusive(SCHEME_SEPARATOR)
                    .next()
                    .unwrap_or(path)
            }),
            Self::Table { table, .. } => &table.name,
        }
    }

    #[must_use]
    pub fn locator(&self) -> String {
        match self {
            Self::Path { base_path, .. } => base_path.as_str().to_owned(),
            Self::Table { table, .. } => table.fully_qualified(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HivePartitions {
    pub partitions: Vec<HivePartition>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HivePartition {
    pub key_values: Vec<KeyValue>,
    pub listing: Vec<Uri>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: Option<String>,
}

impl std::fmt::Display for StorageTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.locator())
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageProperties {
    pub target: StorageTarget,
    pub format: Option<FileFormat>,
    pub partitions: Vec<Partition>,
}

impl StorageProperties {
    #[must_use]
    pub fn new(
        target: StorageTarget,
        format: Option<FileFormat>,
        partitions: Vec<Partition>,
    ) -> Self {
        Self {
            target,
            format,
            partitions,
        }
    }

    #[must_use]
    pub fn bare(target: StorageTarget) -> Self {
        Self::new(target, None, Vec::new())
    }

    #[must_use]
    pub fn locator(&self) -> String {
        self.target.locator()
    }

    pub(crate) fn source_id(&self) -> EntityId {
        match &self.target {
            StorageTarget::Path { base_path, .. } => {
                mint_path_source_id(base_path.as_str(), self.format)
            }
            StorageTarget::Table { table, .. } => mint_table_source_id(&table.fully_qualified()),
        }
    }

    #[must_use]
    pub fn reads_a_supported_format(&self) -> bool {
        match self.target {
            StorageTarget::Path { .. } => self.format == Some(FileFormat::Parquet),
            StorageTarget::Table { .. } => matches!(self.format, None | Some(FileFormat::Iceberg)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSchema {
    pub columns: Vec<SourceColumn>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceColumn {
    pub name: String,
    pub data_type: Option<DataType>,
    pub nullable: Option<bool>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum SchemaProvenance {
    CatalogDerived {
        schema: SourceSchema,
        native: semstrait_catalog::Schema,
    },
    ModelFed {
        schema: SourceSchema,
    },
}

impl SchemaProvenance {
    #[must_use]
    pub fn catalog_derived(schema: SourceSchema, native: semstrait_catalog::Schema) -> Self {
        Self::CatalogDerived { schema, native }
    }

    #[must_use]
    pub fn model_fed(schema: SourceSchema) -> Self {
        Self::ModelFed { schema }
    }

    #[must_use]
    pub fn schema(&self) -> &SourceSchema {
        match self {
            Self::CatalogDerived { schema, .. } | Self::ModelFed { schema } => schema,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalSource {
    pub id: EntityId,
    pub name: String,
    pub alias: SourceAlias,
    pub storage: StorageProperties,
    pub schema_provenance: SchemaProvenance,
    pub schema_id: SchemaId,
}

impl PhysicalSource {
    #[must_use]
    pub fn new(
        id: impl Into<EntityId>,
        name: impl Into<String>,
        alias: SourceAlias,
        storage: StorageProperties,
        schema_provenance: SchemaProvenance,
        schema_id: SchemaId,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            alias,
            storage,
            schema_provenance,
            schema_id,
        }
    }

    #[must_use]
    pub fn schema(&self) -> &SourceSchema {
        self.schema_provenance.schema()
    }

    #[must_use]
    pub fn locator(&self) -> String {
        self.storage.locator()
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum PhysicalSourceVersionRef {
    IcebergSnapshotId(i64),
    MonotonicVersion(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    use semstrait_catalog::{NamespaceRef, PartitionTransform, SchemaColumn as NativeColumn};
    use semstrait_sql::datatype::DataType;

    fn table_ref() -> TableRef {
        TableRef::new(NamespaceRef::new(["warehouse"]), "orders")
    }

    fn canonical() -> SourceSchema {
        SourceSchema {
            columns: vec![SourceColumn {
                name: "amount".to_owned(),
                data_type: Some(DataType::Decimal {
                    precision: 18,
                    scale: 2,
                }),
                nullable: Some(true),
            }],
        }
    }

    fn native() -> semstrait_catalog::Schema {
        semstrait_catalog::Schema::new(vec![NativeColumn::new("amount", "DECIMAL(18,2)", true)])
    }

    fn table_source() -> PhysicalSource {
        PhysicalSource::new(
            "src-orders",
            "orders",
            SourceAlias::from("ps0"),
            StorageProperties::new(
                StorageTarget::table_at(table_ref(), Path::new("s3://wh/db/orders")),
                Some(FileFormat::Iceberg),
                vec![Partition::new(
                    1,
                    "day",
                    "ts",
                    "timestamp",
                    PartitionTransform::Day,
                )],
            ),
            SchemaProvenance::catalog_derived(canonical(), native()),
            SchemaId::from("schema-1"),
        )
    }

    fn base_path(uri: &str) -> StorageTarget {
        StorageTarget::path(Uri::from(uri), Vec::new())
    }

    #[test]
    fn a_table_target_carries_its_observed_location() {
        let observed = StorageTarget::table_at(table_ref(), Path::new("s3://wh/db/orders"));
        let StorageTarget::Table { location, .. } = &observed else {
            panic!("built a table target");
        };
        assert_eq!(
            location.as_ref().map(Path::as_str),
            Some("s3://wh/db/orders")
        );
        assert_eq!(observed.locator(), "warehouse.orders");
    }

    #[test]
    fn a_locator_is_rendered_from_the_typed_target() {
        assert_eq!(
            StorageTarget::table(table_ref()).locator(),
            "warehouse.orders"
        );
        assert_eq!(
            base_path("s3://lake/orders/").locator(),
            "s3://lake/orders/"
        );
        assert!(base_path("s3://lake/orders/").table_ref().is_none());
    }

    #[test]
    fn every_location_has_one_normal_form() {
        for (authored, normal) in [
            ("s3a://lake/orders/", "s3://lake/orders/"),
            ("s3n://lake/orders/", "s3://lake/orders/"),
            ("s3://lake/Orders//", "s3://lake/Orders/"),
            ("/data/orders/a.parquet", "file:///data/orders/a.parquet"),
            ("file:///data/orders/", "file:///data/orders/"),
            ("file:///", "file:///"),
        ] {
            assert_eq!(Uri::normal(authored), Ok(Uri::from(normal)), "{authored}");
            assert!(Uri::from(normal).is_normal(), "{normal}");
        }
        assert_eq!(Uri::normal("orders/a.parquet"), Err(NotNormal::Relative));
        assert_eq!(
            Uri::normal("gs://lake/orders/"),
            Err(NotNormal::Scheme("gs".to_owned()))
        );
        assert!(!Uri::from("s3a://lake/orders/").is_normal());
    }

    #[test]
    fn a_physical_source_is_named_by_a_label() {
        for (uri, name) in [
            ("s3://lake/eu/orders/", "orders"),
            ("s3://lake/", "lake"),
            ("file:///", "file://"),
        ] {
            assert_eq!(base_path(uri).name(), name, "{uri}");
        }
        assert_eq!(StorageTarget::table(table_ref()).name(), "orders");
    }

    #[test]
    fn the_canonical_schema_reads_through_one_accessor_on_both_arms() {
        let observed = SchemaProvenance::catalog_derived(canonical(), native());
        let assumed = SchemaProvenance::model_fed(canonical());
        assert_eq!(observed.schema(), assumed.schema());
        assert_eq!(observed.schema().columns[0].name, "amount");
    }

    #[test]
    fn the_native_spelling_rides_the_catalog_arm_only() {
        let SchemaProvenance::CatalogDerived { native, .. } =
            SchemaProvenance::catalog_derived(canonical(), native())
        else {
            panic!("constructed the catalog-derived arm");
        };
        assert_eq!(
            native.columns[0].data_type, "DECIMAL(18,2)",
            "the provider's own spelling is kept verbatim"
        );
        assert!(matches!(
            SchemaProvenance::model_fed(canonical()),
            SchemaProvenance::ModelFed { .. }
        ));
    }

    #[test]
    fn source_round_trips_through_json() {
        let s = table_source();
        let json = serde_json::to_string(&s).expect("serialize PhysicalSource");
        let back: PhysicalSource = serde_json::from_str(&json).expect("deserialize PhysicalSource");
        assert_eq!(back, s);
    }

    #[test]
    fn provenance_uses_the_adjacent_tag_wire_form() {
        let json = serde_json::to_string(&SchemaProvenance::model_fed(SourceSchema {
            columns: Vec::new(),
        }))
        .expect("serialize");
        assert!(
            json.starts_with(r#"{"type":"model_fed","content":"#),
            "provenance is tagged in the artifact, got {json}"
        );
    }

    #[test]
    fn storage_target_uses_the_adjacent_tag_wire_form() {
        let json = serde_json::to_string(&StorageTarget::table(table_ref())).expect("serialize");
        assert!(
            json.starts_with(r#"{"type":"table","content":"#),
            "got {json}"
        );
    }
}
