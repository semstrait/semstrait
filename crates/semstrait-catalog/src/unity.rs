use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use semstrait_common::io::Path;

use crate::drift::diff_schema;
use crate::{
    CatalogId, CatalogProvider, CatalogProviderErrorKind, DriftReport, FileFormat, NamespaceRef,
    Partition, ResolvedTable, Schema, SchemaColumn, SnapshotMetadata, SnapshotVersion, TableRef,
};

#[derive(Clone)]
pub struct UnityCatalogProvider {
    id: CatalogId,
    host: String,
    catalog_name: String,
    client: Client,
    auth: UnityAuth,
}

#[derive(Clone)]
pub enum UnityAuth {
    Pat(String),
    BearerToken(String),
}

impl UnityCatalogProvider {
    pub fn new(
        id: impl Into<CatalogId>,
        host: impl Into<String>,
        catalog_name: impl Into<String>,
        auth: UnityAuth,
    ) -> Self {
        Self {
            id: id.into(),
            host: host.into(),
            catalog_name: catalog_name.into(),
            client: Client::new(),
            auth,
        }
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn catalog_name(&self) -> &str {
        &self.catalog_name
    }

    fn schema_of(namespace: &NamespaceRef) -> &str {
        namespace.parts.last().map(String::as_str).unwrap_or("")
    }

    async fn get(
        &self,
        url: &str,
        query: &[(&str, &str)],
    ) -> Result<reqwest::Response, CatalogProviderErrorKind> {
        let token = match &self.auth {
            UnityAuth::Pat(pat) => pat.clone(),
            UnityAuth::BearerToken(token) => token.clone(),
        };

        let resp = self
            .client
            .get(url)
            .query(query)
            .bearer_auth(token)
            .send()
            .await
            .map_err(|e| CatalogProviderErrorKind::ConnectionFailed { msg: e.to_string() })?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(CatalogProviderErrorKind::TableNotFound {
                fqn: url.to_string(),
            });
        }
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(CatalogProviderErrorKind::AuthFailed);
        }
        if resp.status() == reqwest::StatusCode::FORBIDDEN {
            return Err(CatalogProviderErrorKind::AuthDenied {
                resource: url.to_string(),
            });
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CatalogProviderErrorKind::ConnectionFailed {
                msg: format!("HTTP {status}: {body}"),
            });
        }

        Ok(resp)
    }

    async fn list_tables_raw(
        &self,
        schema_name: &str,
    ) -> Result<Vec<UnityTableInfo>, CatalogProviderErrorKind> {
        let url = format!("{}/api/2.1/unity-catalog/tables", self.host);

        let mut all_tables = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut query: Vec<(&str, &str)> = vec![
                ("catalog_name", &self.catalog_name),
                ("schema_name", schema_name),
            ];
            if let Some(token) = &page_token {
                query.push(("page_token", token));
            }

            let resp = self.get(&url, &query).await?;
            let body: ListTablesResponse =
                resp.json()
                    .await
                    .map_err(|e| CatalogProviderErrorKind::MalformedResponse {
                        msg: format!("failed to parse tables: {e}"),
                    })?;

            all_tables.extend(body.tables.unwrap_or_default());

            match body.next_page_token {
                Some(token) if !token.is_empty() => page_token = Some(token),
                _ => break,
            }
        }

        Ok(all_tables)
    }

    async fn get_table(
        &self,
        schema_name: &str,
        table_name: &str,
    ) -> Result<UnityTableInfo, CatalogProviderErrorKind> {
        let full_name = format!("{}.{}.{}", self.catalog_name, schema_name, table_name);
        let url = format!("{}/api/2.1/unity-catalog/tables/{}", self.host, full_name);

        let resp = self.get(&url, &[]).await?;
        resp.json()
            .await
            .map_err(|e| CatalogProviderErrorKind::MalformedResponse {
                msg: format!("failed to parse table: {e}"),
            })
    }
}

fn schema_from(info: &UnityTableInfo) -> Schema {
    let columns = info
        .columns
        .iter()
        .flatten()
        .map(|c| {
            SchemaColumn::new(
                c.name.clone(),
                c.type_name.clone(),
                c.nullable.unwrap_or(true),
            )
        })
        .collect();
    Schema::new(columns)
}

#[async_trait]
impl CatalogProvider for UnityCatalogProvider {
    fn id(&self) -> CatalogId {
        self.id.clone()
    }

    async fn list_tables(
        &self,
        namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind> {
        let tables = self.list_tables_raw(Self::schema_of(namespace)).await?;
        Ok(tables
            .into_iter()
            .map(|t| {
                TableRef::with_catalog(
                    self.catalog_name.as_str(),
                    NamespaceRef::new([t.schema_name]),
                    t.name,
                )
            })
            .collect())
    }

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind> {
        let info = self
            .get_table(Self::schema_of(&table.namespace), &table.name)
            .await?;
        let location = info.storage_location.clone().ok_or_else(|| {
            CatalogProviderErrorKind::MalformedResponse {
                msg: format!("table {} has no storage_location", table.fully_qualified()),
            }
        })?;
        Ok(ResolvedTable::new(
            table.clone(),
            Path::new(location),
            unity_format(info.data_source_format.as_deref()),
            schema_from(&info),
            Vec::new(),
            SnapshotMetadata::current_unpinned(),
        ))
    }

    async fn get_schema(&self, table: &TableRef) -> Result<Schema, CatalogProviderErrorKind> {
        let info = self
            .get_table(Self::schema_of(&table.namespace), &table.name)
            .await?;
        Ok(schema_from(&info))
    }

    async fn get_partitions(
        &self,
        _table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind> {
        Ok(Vec::new())
    }

    async fn get_snapshot(
        &self,
        _table: &TableRef,
        _version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind> {
        Ok(SnapshotMetadata::current_unpinned())
    }

    async fn check_schema_drift(
        &self,
        table: &TableRef,
        expected_schema: &Schema,
    ) -> Result<DriftReport, CatalogProviderErrorKind> {
        let current = self.get_schema(table).await?;
        Ok(diff_schema(table.clone(), expected_schema, &current))
    }
}

#[derive(Debug, Deserialize)]
struct ListTablesResponse {
    tables: Option<Vec<UnityTableInfo>>,
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnityTableInfo {
    name: String,
    schema_name: String,
    #[serde(default)]
    storage_location: Option<String>,
    #[serde(default)]
    data_source_format: Option<String>,
    #[serde(default)]
    columns: Option<Vec<UnityColumn>>,
}

#[derive(Debug, Deserialize)]
struct UnityColumn {
    name: String,
    type_name: String,
    #[serde(default)]
    nullable: Option<bool>,
}

fn unity_format(format: Option<&str>) -> FileFormat {
    match format.map(str::to_ascii_uppercase).as_deref() {
        Some("DELTA") => FileFormat::Delta,
        Some("CSV") => FileFormat::Csv,
        Some("JSON") => FileFormat::Json,
        Some("ORC") => FileFormat::Orc,
        Some("AVRO") => FileFormat::Avro,
        _ => FileFormat::Parquet,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_names_no_dialect() {
        let provider = UnityCatalogProvider::new(
            "test",
            "https://example.test",
            "main",
            UnityAuth::Pat("token".to_owned()),
        );
        assert_eq!(provider.dialect(), None);
    }

    #[test]
    fn schema_from_captures_native_type_strings() {
        let info: UnityTableInfo = serde_json::from_value(serde_json::json!({
            "name": "orders",
            "schema_name": "sales",
            "columns": [
                {"name": "id", "type_name": "LONG", "nullable": false},
                {"name": "amount", "type_name": "DECIMAL(10,2)", "nullable": true}
            ]
        }))
        .unwrap();
        let schema = schema_from(&info);
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].data_type, "LONG");
        assert!(!schema.columns[0].nullable);
        assert_eq!(schema.columns[1].data_type, "DECIMAL(10,2)");
        assert!(schema.columns[1].nullable);
    }

    #[test]
    fn unity_format_maps_known_and_defaults() {
        assert_eq!(unity_format(Some("DELTA")), FileFormat::Delta);
        assert_eq!(unity_format(Some("parquet")), FileFormat::Parquet);
        assert_eq!(unity_format(Some("csv")), FileFormat::Csv);
        assert_eq!(unity_format(None), FileFormat::Parquet);
        assert_eq!(unity_format(Some("weird")), FileFormat::Parquet);
    }

    #[test]
    fn schema_of_uses_last_namespace_segment() {
        assert_eq!(
            UnityCatalogProvider::schema_of(&NamespaceRef::new(["main", "sales"])),
            "sales"
        );
        assert_eq!(
            UnityCatalogProvider::schema_of(&NamespaceRef::new(["sales"])),
            "sales"
        );
    }

    #[test]
    fn test_builder() {
        let catalog = UnityCatalogProvider::new(
            "unity_main",
            "https://dbc-xxx.cloud.databricks.com",
            "main",
            UnityAuth::Pat("dapi123".to_string()),
        );
        assert_eq!(catalog.id().as_str(), "unity_main");
        assert_eq!(catalog.host(), "https://dbc-xxx.cloud.databricks.com");
        assert_eq!(catalog.catalog_name(), "main");
    }

    #[test]
    fn test_parse_list_tables_response() {
        let json = serde_json::json!({
            "tables": [
                {
                    "name": "orders",
                    "catalog_name": "main",
                    "schema_name": "sales",
                    "table_type": "MANAGED",
                    "columns": [
                        {"name": "id", "type_name": "LONG", "nullable": false},
                        {"name": "amount", "type_name": "DOUBLE", "nullable": true}
                    ]
                },
                {
                    "name": "customers",
                    "catalog_name": "main",
                    "schema_name": "sales",
                    "table_type": "MANAGED"
                }
            ]
        });
        let resp: ListTablesResponse = serde_json::from_value(json).unwrap();
        let tables = resp.tables.unwrap();
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].name, "orders");
        assert_eq!(tables[0].columns.as_ref().unwrap().len(), 2);
        assert!(tables[1].columns.is_none());
    }

    #[test]
    fn test_parse_table_info_with_columns_and_location() {
        let json = serde_json::json!({
            "name": "orders",
            "catalog_name": "main",
            "schema_name": "sales",
            "table_type": "MANAGED",
            "storage_location": "s3://bucket/sales/orders",
            "data_source_format": "DELTA",
            "columns": [
                {"name": "id", "type_name": "LONG", "nullable": false},
                {"name": "amount", "type_name": "DECIMAL(10,2)", "nullable": true}
            ]
        });
        let info: UnityTableInfo = serde_json::from_value(json).unwrap();
        assert_eq!(
            info.storage_location.as_deref(),
            Some("s3://bucket/sales/orders")
        );
        assert_eq!(info.data_source_format.as_deref(), Some("DELTA"));
        assert_eq!(info.columns.unwrap().len(), 2);
    }

    #[test]
    fn test_parse_empty_list_response() {
        let json = serde_json::json!({});
        let resp: ListTablesResponse = serde_json::from_value(json).unwrap();
        assert!(resp.tables.is_none());
        assert!(resp.next_page_token.is_none());
    }
}
