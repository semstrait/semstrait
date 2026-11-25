use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Fields, Schema};
use datafusion::catalog::TableProvider;
use datafusion::common::{not_impl_err, substrait_err, TableReference};
use datafusion::datasource::file_format::options::ReadOptions;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::error::Result;
use datafusion::execution::{FunctionRegistry, SessionState};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{
    cast, ident, lit, nullif, ParquetReadOptions as DataFusionParquetOptions, SessionContext,
};
use datafusion_substrait::extensions::Extensions as DataFusionExtensions;
use datafusion_substrait::logical_plan::consumer::{
    from_read_rel, from_substrait_named_struct, DefaultSubstraitConsumer, SubstraitConsumer,
};
use datafusion_substrait::substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, PathType,
};
use datafusion_substrait::substrait::proto::read_rel::ReadType;
use datafusion_substrait::substrait::proto::ReadRel;

pub struct FilesConsumer<'a> {
    stock: DefaultSubstraitConsumer<'a>,
    state: &'a SessionState,
    session: &'a SessionContext,
}

impl<'a> FilesConsumer<'a> {
    pub fn new(
        extensions: &'a DataFusionExtensions,
        state: &'a SessionState,
        session: &'a SessionContext,
    ) -> Self {
        Self {
            stock: DefaultSubstraitConsumer::new(extensions, state),
            state,
            session,
        }
    }
}

#[async_trait]
impl SubstraitConsumer for FilesConsumer<'_> {
    async fn resolve_table_ref(
        &self,
        table_ref: &TableReference,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        self.stock.resolve_table_ref(table_ref).await
    }

    fn get_extensions(&self) -> &DataFusionExtensions {
        self.stock.get_extensions()
    }

    fn get_function_registry(&self) -> &impl FunctionRegistry {
        self.stock.get_function_registry()
    }

    async fn consume_read(&self, read: &ReadRel) -> Result<LogicalPlan> {
        let Some(ReadType::LocalFiles(files)) = &read.read_type else {
            return from_read_rel(self, read).await;
        };
        let Some(base_schema) = &read.base_schema else {
            return substrait_err!("a read with no base schema");
        };
        let schema = from_substrait_named_struct(self, base_schema)?;
        let frame = match files.items.as_slice() {
            [item] => match (&item.path_type, &item.file_format) {
                (Some(PathType::UriFile(uri)), Some(FileFormat::Parquet(_))) => {
                    let options = DataFusionParquetOptions::default().schema(schema.as_arrow());
                    self.session.read_parquet(uri.as_str(), options).await?
                }
                (Some(PathType::UriFolder(uri)), Some(FileFormat::Parquet(_))) => {
                    let state = self.state;
                    let keys = (DataFusionParquetOptions::default())
                        .to_listing_options(state.config(), state.default_table_options())
                        .infer_partitions(state, &ListingTableUrl::parse(uri)?)
                        .await?;
                    let key =
                        |column: &str| keys.iter().find(|key| key.eq_ignore_ascii_case(column));
                    let unkeyed: Fields = (schema.fields().iter())
                        .filter(|field| key(field.name()).is_none())
                        .cloned()
                        .collect();
                    let unkeyed = Schema::new(unkeyed);
                    let partitions = (keys.iter()).map(|key| (key.clone(), DataType::Utf8));
                    let options = (DataFusionParquetOptions::default())
                        .schema(&unkeyed)
                        .table_partition_cols(partitions.collect());
                    let frame = self.session.read_parquet(uri.as_str(), options).await?;
                    let columns = (schema.fields().iter()).map(|field| match key(field.name()) {
                        Some(key) => {
                            let value = nullif(ident(key), lit("__HIVE_DEFAULT_PARTITION__"));
                            cast(value, field.data_type().clone()).alias(field.name())
                        }
                        None => ident(field.name()),
                    });
                    frame.select(columns)?
                }
                _ => return not_impl_err!("a read of {item:?}"),
            },
            items => return not_impl_err!("a read of {} items", items.len()),
        };
        let frame = match &read.filter {
            Some(filter) => {
                let condition = self.consume_expression(filter, frame.schema()).await?;
                frame.filter(condition)?
            }
            None => frame,
        };
        Ok(frame.into_unoptimized_plan())
    }
}
