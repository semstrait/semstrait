use std::collections::BTreeMap;
use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType as ArrowType, Field, Fields, Schema as ArrowSchema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::object_store::memory::InMemory;
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use semstrait_catalog::{
    CatalogId, CatalogProvider, CatalogProviderErrorKind, DriftReport, FileFormat, NamespaceRef,
    Partition, ResolvedTable, Schema, SchemaColumn, SnapshotMetadata, SnapshotVersion, TableRef,
};
use semstrait_common::io::Path;
use semstrait_common::Dialect;
use semstrait_manifest::{SemanticManifest, StorageTarget};
use semstrait_sql::datatype::DataType;
use serde::Deserialize;
use serde_yaml::Value;

use super::data::{arrow_type, put, row, session, Bucket, BUCKET};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declaration {
    dialect: Option<Dialect>,
    #[serde(default)]
    tables: Vec<Table>,
    #[serde(default)]
    objects: Vec<Object>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    catalog: Option<String>,
    namespace: String,
    name: String,
    columns: Vec<Column>,
    #[serde(default)]
    rows: Vec<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Object {
    uri: String,
    columns: Vec<Column>,
    #[serde(default)]
    rows: Vec<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Column {
    name: String,
    #[serde(rename = "type")]
    data_type: DataType,
    #[serde(default = "nullable")]
    nullable: bool,
    native: Option<String>,
    zone: Option<String>,
    read: Option<DataType>,
}

fn nullable() -> bool {
    true
}

impl Column {
    fn arrow_type(&self) -> Result<ArrowType, Box<dyn Error>> {
        match (arrow_type(&self.data_type)?, &self.zone) {
            (data_type, None) => Ok(data_type),
            (ArrowType::Timestamp(unit, None), Some(zone)) => {
                Ok(ArrowType::Timestamp(unit, Some(zone.as_str().into())))
            }
            (data_type, Some(zone)) => Err(format!("no zone `{zone}` on `{data_type}`").into()),
        }
    }
}

#[derive(Debug)]
pub struct Declared {
    pub catalog: Catalog,
    pub bucket: Bucket,
    tables: Vec<([String; 3], RecordBatch)>,
}

impl Declared {
    pub fn session(&self) -> Result<SessionContext, Box<dyn Error>> {
        session(Arc::clone(&self.bucket.0), &self.tables)
    }
}

impl Table {
    fn names(&self) -> [String; 3] {
        let catalog = self.catalog.clone().unwrap_or_default();
        [catalog, self.namespace.clone(), self.name.clone()]
    }

    fn table_ref(&self) -> TableRef {
        let namespace = NamespaceRef::new(self.namespace.split('.'));
        match &self.catalog {
            Some(catalog) => TableRef::with_catalog(catalog.as_str(), namespace, &self.name),
            None => TableRef::new(namespace, &self.name),
        }
    }

    fn resolved(&self) -> ResolvedTable {
        let table = self.table_ref();
        let columns = (self.columns.iter()).map(|column| {
            let native = (column.native.clone()).unwrap_or_else(|| column.data_type.to_string());
            SchemaColumn::new(&column.name, native, column.nullable)
        });
        ResolvedTable::new(
            table.clone(),
            Path::new(format!("{BUCKET}{table}")),
            FileFormat::Iceberg,
            Schema::new(columns.collect()),
            Vec::new(),
            SnapshotMetadata::new(SnapshotVersion::Current, None),
        )
    }

    fn matches(&self, key: &str) -> bool {
        let plain = format!("{}.{}", self.namespace, self.name);
        if key == plain {
            return true;
        }
        match &self.catalog {
            Some(catalog) if !catalog.is_empty() => key == format!("{catalog}.{plain}"),
            _ => false,
        }
    }
}

impl Declaration {
    pub fn read(path: &std::path::Path) -> Result<Self, Box<dyn Error>> {
        Ok(serde_yaml::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn apply_input_data(
        &mut self,
        data: &BTreeMap<String, Vec<Vec<Value>>>,
    ) -> Result<(), Box<dyn Error>> {
        for (key, rows) in data {
            let mut matched = false;
            for table in &mut self.tables {
                if table.matches(key) {
                    table.rows = rows.clone();
                    matched = true;
                }
            }
            if !matched {
                if let Some(object) = self.objects.iter_mut().find(|object| object.uri == *key) {
                    object.rows = rows.clone();
                    matched = true;
                }
            }
            if !matched {
                return Err(format!("the input data names no table or object `{key}`").into());
            }
        }
        Ok(())
    }

    pub async fn build(&self) -> Result<Declared, Box<dyn Error>> {
        let store = Arc::new(InMemory::new());
        for object in &self.objects {
            put(&store, &object.uri, &batch(&object.columns, &object.rows)?).await?;
        }
        let tables = (self.tables.iter())
            .map(|table| Ok((table.names(), batch(&table.columns, &table.rows)?)))
            .collect::<Result<_, Box<dyn Error>>>()?;
        Ok(Declared {
            catalog: Catalog {
                dialect: self.dialect,
                tables: self.tables.iter().map(Table::resolved).collect(),
            },
            bucket: Bucket(store),
            tables,
        })
    }

    pub fn check(&self, manifest: &SemanticManifest) -> Result<(), Box<dyn Error>> {
        for source in manifest.sources.values() {
            for declared in self.read_by(&source.storage.target)? {
                for column in &source.schema().columns {
                    let given = declared.iter().find(|given| given.name == column.name);
                    let (Some(data_type), Some(given)) = (&column.data_type, given) else {
                        continue;
                    };
                    let read = given.read.as_ref().unwrap_or(&given.data_type);
                    if read != data_type {
                        return Err(format!(
                            "the column `{}` of `{}` is {data_type} in the manifest and {read} \
                             in the declaration",
                            column.name, source.name
                        )
                        .into());
                    }
                }
            }
        }
        Ok(())
    }

    fn read_by(&self, target: &StorageTarget) -> Result<Vec<&[Column]>, Box<dyn Error>> {
        Ok(match target {
            StorageTarget::Table { table, .. } => (self.tables.iter())
                .filter(|declared| declared.table_ref() == *table)
                .map(|declared| declared.columns.as_slice())
                .collect(),
            StorageTarget::Path { listings, .. } => (self.objects.iter())
                .filter(|declared| listings.iter().any(|uri| uri.as_str() == declared.uri))
                .map(|declared| declared.columns.as_slice())
                .collect(),
            _ => return Err(format!("the type check reads no target {target}").into()),
        })
    }
}

fn batch(columns: &[Column], rows: &[Vec<Value>]) -> Result<RecordBatch, Box<dyn Error>> {
    let fields = (columns.iter())
        .map(|column| {
            let data_type = column.arrow_type()?;
            Ok(Field::new(&column.name, data_type, column.nullable))
        })
        .collect::<Result<Fields, Box<dyn Error>>>()?;
    let rows = (rows.iter())
        .map(|values| row(values, &fields))
        .collect::<Result<Vec<_>, _>>()?;
    let schema = Arc::new(ArrowSchema::new(fields));
    if rows.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }
    let arrays = (0..schema.fields().len())
        .map(|index| ScalarValue::iter_to_array(rows.iter().map(|row| row[index].clone())))
        .collect::<Result<_, _>>()?;
    Ok(RecordBatch::try_new(schema, arrays)?)
}

#[derive(Debug)]
pub struct Catalog {
    dialect: Option<Dialect>,
    tables: Vec<ResolvedTable>,
}

impl Catalog {
    fn resolved(&self, table: &TableRef) -> Result<&ResolvedTable, CatalogProviderErrorKind> {
        (self.tables.iter())
            .find(|resolved| &resolved.table == table)
            .ok_or_else(|| CatalogProviderErrorKind::TableNotFound {
                fqn: table.fully_qualified(),
            })
    }
}

#[async_trait]
impl CatalogProvider for Catalog {
    fn id(&self) -> CatalogId {
        CatalogId::new("declaration")
    }

    fn dialect(&self) -> Option<Dialect> {
        self.dialect
    }

    async fn list_tables(
        &self,
        namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind> {
        let tables = self.tables.iter().map(|resolved| &resolved.table);
        Ok(tables
            .filter(|table| &table.namespace == namespace)
            .cloned()
            .collect())
    }

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind> {
        self.resolved(table).cloned()
    }

    async fn get_schema(&self, table: &TableRef) -> Result<Schema, CatalogProviderErrorKind> {
        Ok(self.resolved(table)?.schema.clone())
    }

    async fn get_partitions(
        &self,
        table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind> {
        Ok(self.resolved(table)?.partitions.clone())
    }

    async fn get_snapshot(
        &self,
        table: &TableRef,
        _version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind> {
        Ok(self.resolved(table)?.snapshot.clone())
    }

    async fn check_schema_drift(
        &self,
        _table: &TableRef,
        _expected: &Schema,
    ) -> Result<DriftReport, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::NotAvailable {
            msg: "a declaration reports no drift".to_owned(),
        })
    }
}
