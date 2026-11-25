use async_trait::async_trait;

use crate::{
    CatalogId, CatalogProvider, CatalogProviderErrorKind, DriftReport, NamespaceRef, Partition,
    Schema, SnapshotMetadata, SnapshotVersion, TableRef,
};

#[derive(Debug, Clone)]
pub struct NoopCatalogProvider {
    id: CatalogId,
}

impl NoopCatalogProvider {
    pub fn new(id: impl Into<CatalogId>) -> Self {
        Self { id: id.into() }
    }
}

impl Default for NoopCatalogProvider {
    fn default() -> Self {
        Self {
            id: CatalogId::new("noop"),
        }
    }
}

#[async_trait]
impl CatalogProvider for NoopCatalogProvider {
    fn id(&self) -> CatalogId {
        self.id.clone()
    }

    async fn list_tables(
        &self,
        _namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind> {
        Ok(Vec::new())
    }

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<crate::ResolvedTable, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::TableNotFound {
            fqn: table.fully_qualified(),
        })
    }

    async fn get_schema(&self, table: &TableRef) -> Result<Schema, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::TableNotFound {
            fqn: table.fully_qualified(),
        })
    }

    async fn get_partitions(
        &self,
        _table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind> {
        Ok(Vec::new())
    }

    async fn get_snapshot(
        &self,
        table: &TableRef,
        _version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::TableNotFound {
            fqn: table.fully_qualified(),
        })
    }

    async fn check_schema_drift(
        &self,
        table: &TableRef,
        _expected_schema: &Schema,
    ) -> Result<DriftReport, CatalogProviderErrorKind> {
        Err(CatalogProviderErrorKind::TableNotFound {
            fqn: table.fully_qualified(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> TableRef {
        TableRef::new(NamespaceRef::new(["ns"]), "t")
    }

    #[test]
    fn noop_names_no_dialect() {
        assert_eq!(NoopCatalogProvider::default().dialect(), None);
    }

    #[test]
    fn default_id_is_noop() {
        assert_eq!(NoopCatalogProvider::default().id().as_str(), "noop");
    }

    #[test]
    fn explicit_id_is_preserved() {
        let p = NoopCatalogProvider::new("custom");
        assert_eq!(p.id().as_str(), "custom");
    }

    #[tokio::test]
    async fn list_tables_is_empty() {
        let p = NoopCatalogProvider::default();
        let ns = NamespaceRef::new(["any"]);
        assert!(p.list_tables(&ns).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn get_partitions_is_empty() {
        let p = NoopCatalogProvider::default();
        assert!(p.get_partitions(&table()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn resolve_table_is_not_found() {
        let p = NoopCatalogProvider::default();
        let err = p.resolve_table(&table()).await.unwrap_err();
        assert!(matches!(
            err,
            CatalogProviderErrorKind::TableNotFound { .. }
        ));
    }

    #[tokio::test]
    async fn table_exists_is_false_via_default_impl() {
        let p = NoopCatalogProvider::default();
        assert!(!p.table_exists(&table()).await.unwrap());
    }

    #[tokio::test]
    async fn usable_as_dyn_object() {
        let provider: &dyn CatalogProvider = &NoopCatalogProvider::default();
        assert_eq!(provider.id().as_str(), "noop");
        assert!(provider
            .list_tables(&NamespaceRef::new(["x"]))
            .await
            .unwrap()
            .is_empty());
    }
}
