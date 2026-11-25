pub mod error;
pub mod noop_provider;
pub mod types;

#[cfg(any(feature = "iceberg", feature = "unity"))]
pub(crate) mod drift;

#[cfg(feature = "iceberg")]
pub mod iceberg;

#[cfg(feature = "unity")]
pub mod unity;

#[cfg(feature = "aws")]
pub mod secrets;

use async_trait::async_trait;
use semstrait_common::Dialect;

pub use error::CatalogProviderErrorKind;
pub use noop_provider::NoopCatalogProvider;
pub use types::{
    CatalogId, DriftKind, DriftReport, DriftStatus, FileFormat, NamespaceRef, Partition,
    PartitionTransform, ResolvedTable, Schema, SchemaColumn, SnapshotMetadata, SnapshotVersion,
    TableRef,
};

#[cfg(feature = "iceberg")]
pub use iceberg::IcebergRestCatalogProvider;

#[cfg(feature = "unity")]
pub use unity::UnityCatalogProvider;

#[allow(clippy::double_must_use)]
#[async_trait]
pub trait CatalogProvider: Send + Sync {
    fn id(&self) -> CatalogId;

    fn dialect(&self) -> Option<Dialect> {
        None
    }

    async fn list_tables(
        &self,
        namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind>;

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind>;

    async fn get_schema(&self, table: &TableRef) -> Result<Schema, CatalogProviderErrorKind>;

    async fn get_partitions(
        &self,
        table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind>;

    async fn get_snapshot(
        &self,
        table: &TableRef,
        version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind>;

    async fn check_schema_drift(
        &self,
        table: &TableRef,
        expected_schema: &Schema,
    ) -> Result<DriftReport, CatalogProviderErrorKind>;

    async fn table_exists(&self, table: &TableRef) -> Result<bool, CatalogProviderErrorKind> {
        match self.resolve_table(table).await {
            Ok(_) => Ok(true),
            Err(CatalogProviderErrorKind::TableNotFound { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use super::*;

    fn _assert_object_safe(_: &dyn CatalogProvider) {}

    #[test]
    fn provider_is_object_safe_as_box() {
        let _: Box<dyn CatalogProvider> = Box::new(NoopCatalogProvider::default());
    }

    struct Probe<T: ?Sized>(PhantomData<T>);

    trait NotDebug {
        const DEBUG: bool = false;
    }

    impl<T: ?Sized> NotDebug for Probe<T> {}

    impl<T: ?Sized + Debug> Probe<T> {
        const DEBUG: bool = true;
    }

    #[test]
    fn only_the_types_that_hold_no_credential_are_debug() {
        let debug: Vec<&str> = [
            ("NoopCatalogProvider", Probe::<NoopCatalogProvider>::DEBUG),
            ("dyn CatalogProvider", Probe::<dyn CatalogProvider>::DEBUG),
            #[cfg(feature = "iceberg")]
            ("AuthConfig", Probe::<iceberg::AuthConfig>::DEBUG),
            #[cfg(feature = "iceberg")]
            (
                "IcebergRestCatalogProvider",
                Probe::<IcebergRestCatalogProvider>::DEBUG,
            ),
            #[cfg(feature = "iceberg")]
            ("TokenResponse", Probe::<iceberg::TokenResponse>::DEBUG),
            #[cfg(feature = "unity")]
            ("UnityAuth", Probe::<unity::UnityAuth>::DEBUG),
            #[cfg(feature = "unity")]
            ("UnityCatalogProvider", Probe::<UnityCatalogProvider>::DEBUG),
            #[cfg(feature = "aws")]
            ("PolarisSecret", Probe::<secrets::PolarisSecret>::DEBUG),
        ]
        .into_iter()
        .filter_map(|(name, debug)| debug.then_some(name))
        .collect();

        assert_eq!(debug, ["NoopCatalogProvider"]);
    }
}
