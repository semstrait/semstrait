use semstrait_common::io::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CatalogId(String);

impl CatalogId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CatalogId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for CatalogId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for CatalogId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamespaceRef {
    pub parts: Vec<String>,
}

impl NamespaceRef {
    pub fn new(parts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            parts: parts.into_iter().map(Into::into).collect(),
        }
    }

    pub fn dotted(&self) -> String {
        self.parts.join(".")
    }
}

impl std::fmt::Display for NamespaceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.dotted())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableRef {
    pub catalog: Option<CatalogId>,
    pub namespace: NamespaceRef,
    pub name: String,
}

impl TableRef {
    pub fn new(namespace: NamespaceRef, name: impl Into<String>) -> Self {
        Self {
            catalog: None,
            namespace,
            name: name.into(),
        }
    }

    pub fn with_catalog(
        catalog: impl Into<CatalogId>,
        namespace: NamespaceRef,
        name: impl Into<String>,
    ) -> Self {
        Self {
            catalog: Some(catalog.into()),
            namespace,
            name: name.into(),
        }
    }

    pub fn fully_qualified(&self) -> String {
        match &self.catalog {
            Some(c) => format!("{}.{}.{}", c, self.namespace.dotted(), self.name),
            None => format!("{}.{}", self.namespace.dotted(), self.name),
        }
    }
}

impl std::fmt::Display for TableRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.fully_qualified())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

impl SchemaColumn {
    pub fn new(name: impl Into<String>, data_type: impl Into<String>, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            nullable,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    pub columns: Vec<SchemaColumn>,
}

impl Schema {
    pub fn new(columns: Vec<SchemaColumn>) -> Self {
        Self { columns }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    Parquet,
    Orc,
    Avro,
    Csv,
    Json,
    Delta,
    Iceberg,
}

impl std::fmt::Display for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Parquet => "parquet",
            Self::Orc => "orc",
            Self::Avro => "avro",
            Self::Csv => "csv",
            Self::Json => "json",
            Self::Delta => "delta",
            Self::Iceberg => "iceberg",
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub position: usize,
    pub name: String,
    pub source_column: String,
    pub data_type: String,
    pub transform: PartitionTransform,
}

impl Partition {
    pub fn new(
        position: usize,
        name: impl Into<String>,
        source_column: impl Into<String>,
        data_type: impl Into<String>,
        transform: PartitionTransform,
    ) -> Self {
        Self {
            position,
            name: name.into(),
            source_column: source_column.into(),
            data_type: data_type.into(),
            transform,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionTransform {
    Identity,
    Year,
    Month,
    Day,
    Hour,
    Bucket(u32),
    Truncate(u32),
    Void,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotVersion {
    Current,
    Id(i64),
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub version: SnapshotVersion,
    pub snapshot_id: Option<i64>,
}

impl SnapshotMetadata {
    pub fn new(version: SnapshotVersion, snapshot_id: Option<i64>) -> Self {
        Self {
            version,
            snapshot_id,
        }
    }

    pub fn current_unpinned() -> Self {
        Self {
            version: SnapshotVersion::Current,
            snapshot_id: None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedTable {
    pub table: TableRef,
    #[serde(with = "path_serde")]
    pub location: Path,
    pub format: FileFormat,
    pub schema: Schema,
    pub partitions: Vec<Partition>,
    pub snapshot: SnapshotMetadata,
}

impl ResolvedTable {
    pub fn new(
        table: TableRef,
        location: Path,
        format: FileFormat,
        schema: Schema,
        partitions: Vec<Partition>,
        snapshot: SnapshotMetadata,
    ) -> Self {
        Self {
            table,
            location,
            format,
            schema,
            partitions,
            snapshot,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriftReport {
    pub table: TableRef,
    pub status: DriftStatus,
    pub details: Vec<DriftKind>,
}

impl DriftReport {
    pub fn new(table: TableRef, status: DriftStatus, details: Vec<DriftKind>) -> Self {
        Self {
            table,
            status,
            details,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftStatus {
    Unchanged,
    Compatible,
    Breaking,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftKind {
    ColumnAdded {
        name: String,
    },
    ColumnRemoved {
        name: String,
    },
    ColumnRetyped {
        name: String,
        expected: String,
        actual: String,
    },
    ColumnRenamed {
        expected: String,
        actual: String,
    },
    NullabilityTightened {
        name: String,
    },
    NullabilityRelaxed {
        name: String,
    },
    PartitionChanged {
        details: String,
    },
}

impl DriftKind {
    pub fn describe(&self) -> String {
        match self {
            Self::ColumnAdded { name } => format!("column added: {name}"),
            Self::ColumnRemoved { name } => format!("column removed: {name}"),
            Self::ColumnRetyped {
                name,
                expected,
                actual,
            } => format!("column retyped: {name} ({expected} -> {actual})"),
            Self::ColumnRenamed { expected, actual } => {
                format!("column renamed: {expected} -> {actual}")
            }
            Self::NullabilityTightened { name } => format!("nullability tightened: {name}"),
            Self::NullabilityRelaxed { name } => format!("nullability relaxed: {name}"),
            Self::PartitionChanged { details } => format!("partition spec changed: {details}"),
        }
    }
}

mod path_serde {
    use super::Path;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(path: &Path, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(path.as_str())
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Path, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Path::new(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_dotted_joins_parts() {
        let ns = NamespaceRef::new(["sales", "prod"]);
        assert_eq!(ns.dotted(), "sales.prod");
        assert_eq!(ns.to_string(), "sales.prod");
    }

    #[test]
    fn table_ref_fully_qualified_with_and_without_catalog() {
        let ns = NamespaceRef::new(["sales"]);
        let t = TableRef::new(ns.clone(), "orders");
        assert_eq!(t.fully_qualified(), "sales.orders");

        let t = TableRef::with_catalog("prod", ns, "orders");
        assert_eq!(t.fully_qualified(), "prod.sales.orders");
    }

    #[test]
    fn catalog_id_round_trips() {
        let id = CatalogId::from("polaris_prod");
        assert_eq!(id.as_str(), "polaris_prod");
        assert_eq!(id.to_string(), "polaris_prod");
    }

    #[test]
    fn schema_column_carries_native_type_string() {
        let c = SchemaColumn::new("ts", "timestamptz", true);
        assert_eq!(c.data_type, "timestamptz");
        assert!(c.nullable);
    }

    #[test]
    fn schema_empty_is_empty() {
        assert!(Schema::empty().is_empty());
        assert!(!Schema::new(vec![SchemaColumn::new("a", "int", false)]).is_empty());
    }

    #[test]
    fn drift_kind_describe_retyped_shows_native_strings() {
        let d = DriftKind::ColumnRetyped {
            name: "amount".to_owned(),
            expected: "DECIMAL(18,2)".to_owned(),
            actual: "DOUBLE".to_owned(),
        };
        assert_eq!(
            d.describe(),
            "column retyped: amount (DECIMAL(18,2) -> DOUBLE)"
        );
    }

    #[test]
    fn schema_serde_round_trip_preserves_native_types() {
        let s = Schema::new(vec![
            SchemaColumn::new("id", "BIGINT", false),
            SchemaColumn::new("amount", "DECIMAL(18,2)", true),
        ]);
        let json = serde_json::to_string(&s).expect("serialize");
        let back: Schema = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, s);
    }

    #[test]
    fn resolved_table_serde_round_trip() {
        let ns = NamespaceRef::new(["sales"]);
        let rt = ResolvedTable::new(
            TableRef::new(ns, "orders"),
            Path::new("s3://wh/db/orders"),
            FileFormat::Parquet,
            Schema::new(vec![SchemaColumn::new("id", "BIGINT", false)]),
            vec![Partition::new(
                1,
                "order_date_day",
                "order_date",
                "date",
                PartitionTransform::Day,
            )],
            SnapshotMetadata::new(SnapshotVersion::Id(42), Some(42)),
        );
        let json = serde_json::to_string(&rt).expect("serialize");
        let back: ResolvedTable = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, rt);
    }

    #[test]
    fn partition_transform_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&PartitionTransform::Identity).expect("serialize"),
            r#""identity""#
        );
        assert_eq!(
            serde_json::to_string(&PartitionTransform::Bucket(8)).expect("serialize"),
            r#"{"bucket":8}"#
        );
    }

    #[test]
    fn file_format_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&FileFormat::Parquet).expect("serialize"),
            r#""parquet""#
        );
    }

    #[test]
    fn a_file_format_displays_its_serde_name() {
        for format in [
            FileFormat::Parquet,
            FileFormat::Orc,
            FileFormat::Avro,
            FileFormat::Csv,
            FileFormat::Json,
            FileFormat::Delta,
            FileFormat::Iceberg,
        ] {
            assert_eq!(
                serde_json::to_value(format).expect("serialize"),
                format.to_string()
            );
        }
    }

    #[test]
    fn drift_report_serde_round_trip() {
        let report = DriftReport::new(
            TableRef::new(NamespaceRef::new(["s"]), "t"),
            DriftStatus::Breaking,
            vec![DriftKind::ColumnRemoved {
                name: "x".to_owned(),
            }],
        );
        let json = serde_json::to_string(&report).expect("serialize");
        let back: DriftReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);
    }
}
