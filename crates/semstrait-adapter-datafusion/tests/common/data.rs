use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::datatypes::{DataType as ArrowType, Fields, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemTable, MemoryCatalogProvider, MemorySchemaProvider};
use datafusion::common::TableReference;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::memory::InMemory;
use datafusion::object_store::path::Path as Key;
use datafusion::object_store::{Error as StoreError, ObjectStore, ObjectStoreExt};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::prelude::SessionContext;
use datafusion::scalar::ScalarValue;
use futures_util::TryStreamExt;
use semstrait_common::io::{FileEntry, FileSystem, FileSystemErrorKind, Path};
use semstrait_common::Diagnose;
use semstrait_sql::datatype::DataType;
use serde_yaml::Value;

use super::Row;

pub const BUCKET: &str = "s3://semstrait-e2e/";

pub fn session(
    store: Arc<InMemory>,
    tables: &[([String; 3], RecordBatch)],
) -> Result<SessionContext, Box<dyn Error>> {
    let session = SessionContext::new();
    session.register_object_store(ObjectStoreUrl::parse(BUCKET)?.as_ref(), store);
    for ([catalog, namespace, name], batch) in tables {
        let catalog = if catalog.is_empty() {
            "datafusion"
        } else {
            catalog
        };
        if session.catalog(catalog).is_none() {
            session.register_catalog(catalog, Arc::new(MemoryCatalogProvider::new()));
        }
        let provider = session.catalog(catalog).ok_or("a registered catalog")?;
        if provider.schema(namespace).is_none() {
            provider.register_schema(namespace, Arc::new(MemorySchemaProvider::new()))?;
        }
        let table = MemTable::try_new(batch.schema(), vec![vec![batch.clone()]])?;
        let names = TableReference::full(catalog, namespace.as_str(), name.as_str());
        session.register_table(names, Arc::new(table))?;
    }
    Ok(session)
}

pub async fn put(store: &InMemory, uri: &str, batch: &RecordBatch) -> Result<(), Box<dyn Error>> {
    let key = key(&Path::new(uri)).map_err(|error| error.message())?;
    let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), None)?;
    writer.write(batch)?;
    store.put(&key, writer.into_inner()?.into()).await?;
    Ok(())
}

#[derive(Debug)]
pub struct Bucket(pub Arc<InMemory>);

fn key(path: &Path) -> Result<Key, FileSystemErrorKind> {
    let key = path.as_str().strip_prefix(BUCKET);
    key.map(Key::from)
        .ok_or_else(|| FileSystemErrorKind::InvalidPath {
            path: path.clone(),
            msg: format!("not under {BUCKET}"),
        })
}

fn refused(path: &Path, error: StoreError) -> FileSystemErrorKind {
    match error {
        StoreError::NotFound { .. } => FileSystemErrorKind::NotFound { path: path.clone() },
        error => FileSystemErrorKind::Internal {
            msg: error.to_string(),
        },
    }
}

#[async_trait]
impl FileSystem for Bucket {
    async fn list(&self, prefix: &Path) -> Result<Vec<FileEntry>, FileSystemErrorKind> {
        let objects: Vec<_> =
            (self.0.list(None).try_collect().await).map_err(|error| refused(prefix, error))?;
        let entries = objects.into_iter().filter_map(|object| {
            let uri = format!("{BUCKET}{}", object.location);
            uri.starts_with(prefix.as_str())
                .then(|| FileEntry::new(Path::new(uri), object.size, None))
        });
        Ok(entries.collect())
    }

    async fn read(&self, path: &Path) -> Result<Bytes, FileSystemErrorKind> {
        let object = (self.0.get(&key(path)?).await).map_err(|error| refused(path, error))?;
        object.bytes().await.map_err(|error| refused(path, error))
    }

    async fn write(&self, path: &Path, data: Bytes) -> Result<(), FileSystemErrorKind> {
        let written = self.0.put(&key(path)?, data.into()).await;
        written.map(drop).map_err(|error| refused(path, error))
    }

    async fn exists(&self, path: &Path) -> Result<bool, FileSystemErrorKind> {
        match self.0.head(&key(path)?).await {
            Ok(_) => Ok(true),
            Err(StoreError::NotFound { .. }) => Ok(false),
            Err(error) => Err(refused(path, error)),
        }
    }
}

pub fn arrow_type(data_type: &DataType) -> Result<ArrowType, Box<dyn Error>> {
    let unit = |precision: &u8| match precision {
        0 => Ok(TimeUnit::Second),
        3 => Ok(TimeUnit::Millisecond),
        6 => Ok(TimeUnit::Microsecond),
        9 => Ok(TimeUnit::Nanosecond),
        _ => Err(format!("no Arrow time unit for `{data_type}`")),
    };
    Ok(match data_type {
        DataType::Boolean => ArrowType::Boolean,
        DataType::TinyInt => ArrowType::Int8,
        DataType::SmallInt => ArrowType::Int16,
        DataType::Integer => ArrowType::Int32,
        DataType::BigInt => ArrowType::Int64,
        DataType::Real => ArrowType::Float32,
        DataType::Double => ArrowType::Float64,
        DataType::Decimal { precision, scale } => ArrowType::Decimal128(*precision, *scale),
        DataType::String | DataType::VarChar { .. } | DataType::FixedChar { .. } => ArrowType::Utf8,
        DataType::Binary => ArrowType::Binary,
        DataType::Date => ArrowType::Date32,
        DataType::Timestamp { precision } => ArrowType::Timestamp(unit(precision)?, None),
        _ => return Err(format!("no Arrow type for `{data_type}`").into()),
    })
}

pub fn row(values: &[Value], fields: &Fields) -> Result<Row, Box<dyn Error>> {
    if values.len() != fields.len() {
        let failure = format!(
            "a row of {} values under {} fields: {values:?}",
            values.len(),
            fields.len()
        );
        return Err(failure.into());
    }
    (values.iter().zip(fields.iter()))
        .map(|(value, field)| cell(value, field.data_type()))
        .collect()
}

fn cell(value: &Value, arrow_type: &ArrowType) -> Result<ScalarValue, Box<dyn Error>> {
    let text = match value {
        Value::Null => return Ok(ScalarValue::try_from(arrow_type)?),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        _ => return Err(format!("no literal for {value:?}").into()),
    };
    Ok(ScalarValue::try_from_string(text, arrow_type)?)
}
