use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::RwLock;

use semstrait_common::io::Path;
use semstrait_common::Dialect;

use crate::drift::diff_schema;
use crate::{
    CatalogId, CatalogProvider, CatalogProviderErrorKind, DriftReport, FileFormat, NamespaceRef,
    Partition, PartitionTransform, ResolvedTable, Schema, SchemaColumn, SnapshotMetadata,
    SnapshotVersion, TableRef,
};

#[derive(Clone)]
pub enum AuthConfig {
    None,
    BearerToken(String),
    OAuth2 {
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    },
}

#[derive(Clone)]
pub struct IcebergRestCatalogProvider {
    id: CatalogId,
    base_url: String,
    warehouse: Option<String>,
    prefix: Option<String>,
    client: Client,
    auth: AuthConfig,
    custom_headers: HashMap<String, String>,
    #[allow(clippy::type_complexity)]
    token_cache: Arc<RwLock<Option<(String, Option<Instant>)>>>,
}

impl IcebergRestCatalogProvider {
    pub fn new(id: impl Into<CatalogId>, base_url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into(),
            warehouse: None,
            prefix: None,
            client: Client::new(),
            auth: AuthConfig::None,
            custom_headers: HashMap::new(),
            token_cache: Arc::new(RwLock::new(None)),
        }
    }

    pub fn with_warehouse(mut self, warehouse: impl Into<String>) -> Self {
        self.warehouse = Some(warehouse.into());
        self
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth = AuthConfig::BearerToken(token.into());
        self
    }

    pub fn with_oauth2(
        mut self,
        token_url: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        scope: Option<String>,
    ) -> Self {
        self.auth = AuthConfig::OAuth2 {
            token_url: token_url.into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            scope,
        };
        self
    }

    pub fn with_custom_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers.insert(key.into(), value.into());
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn api_path(&self) -> String {
        match &self.prefix {
            Some(prefix) => format!("{}/v1/{}", self.base_url, prefix),
            None => format!("{}/v1", self.base_url),
        }
    }

    async fn get_token(&self) -> Result<Option<String>, CatalogProviderErrorKind> {
        match &self.auth {
            AuthConfig::None => Ok(None),
            AuthConfig::BearerToken(token) => Ok(Some(token.clone())),
            AuthConfig::OAuth2 {
                token_url,
                client_id,
                client_secret,
                scope,
            } => {
                {
                    let cached = self.token_cache.read().await;
                    if let Some((token, expiry)) = cached.as_ref() {
                        if expiry.is_none_or(|exp| Instant::now() < exp) {
                            return Ok(Some(token.clone()));
                        }
                        tracing::debug!("OAuth2 token expired, refreshing");
                    }
                }

                tracing::debug!("fetching OAuth2 token from {}", token_url);
                let mut params = vec![
                    ("grant_type", "client_credentials"),
                    ("client_id", client_id.as_str()),
                    ("client_secret", client_secret.as_str()),
                ];
                if let Some(s) = scope {
                    params.push(("scope", s.as_str()));
                }

                let mut token_req = self.client.post(token_url).form(&params);
                for (k, v) in &self.custom_headers {
                    token_req = token_req.header(k.as_str(), v.as_str());
                }
                let resp = token_req.send().await.map_err(|e| {
                    CatalogProviderErrorKind::ConnectionFailed { msg: e.to_string() }
                })?;

                if !resp.status().is_success() {
                    return Err(CatalogProviderErrorKind::AuthFailed);
                }

                let body: TokenResponse =
                    resp.json()
                        .await
                        .map_err(|e| CatalogProviderErrorKind::MalformedResponse {
                            msg: format!("failed to parse token: {e}"),
                        })?;

                let token = body.access_token;
                let expiry = body
                    .expires_in
                    .map(|secs| Instant::now() + Duration::from_secs(secs.saturating_sub(30)));
                *self.token_cache.write().await = Some((token.clone(), expiry));
                Ok(Some(token))
            }
        }
    }

    async fn get(&self, url: &str) -> Result<reqwest::Response, CatalogProviderErrorKind> {
        let mut req = self.client.get(url);

        if let Some(token) = self.get_token().await? {
            req = req.bearer_auth(token);
        }
        if let Some(wh) = &self.warehouse {
            req = req.query(&[("warehouse", wh.as_str())]);
        }
        for (k, v) in &self.custom_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let resp = req
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
        namespace: &str,
    ) -> Result<Vec<TableIdentifier>, CatalogProviderErrorKind> {
        let url = format!(
            "{}/namespaces/{}/tables",
            self.api_path(),
            encode_namespace(namespace)
        );
        let resp = self.get(&url).await?;
        let body: ListTablesResponse =
            resp.json()
                .await
                .map_err(|e| CatalogProviderErrorKind::MalformedResponse {
                    msg: format!("failed to parse tables: {e}"),
                })?;
        Ok(body.identifiers)
    }

    async fn load_table(
        &self,
        namespace: &str,
        table: &str,
    ) -> Result<TableMetadata, CatalogProviderErrorKind> {
        let url = format!(
            "{}/namespaces/{}/tables/{}",
            self.api_path(),
            encode_namespace(namespace),
            table
        );
        let resp = self.get(&url).await?;
        let parsed: LoadTableResponse =
            resp.json()
                .await
                .map_err(|e| CatalogProviderErrorKind::MalformedResponse {
                    msg: format!("failed to parse table metadata: {e}"),
                })?;
        Ok(parsed.metadata)
    }

    fn current_schema(meta: &TableMetadata, fqn: &str) -> Result<Schema, CatalogProviderErrorKind> {
        let schema = meta
            .schemas
            .iter()
            .find(|s| s.schema_id == meta.current_schema_id)
            .ok_or_else(|| CatalogProviderErrorKind::MalformedResponse {
                msg: format!("no current schema found for {fqn}"),
            })?;

        let columns = schema
            .fields
            .iter()
            .map(|f| SchemaColumn::new(f.name.clone(), native_type(&f.r#type), !f.required))
            .collect();
        Ok(Schema::new(columns))
    }

    fn partitions(meta: &TableMetadata) -> Vec<Partition> {
        let field_map: HashMap<i32, (&str, String)> = meta
            .schemas
            .iter()
            .find(|s| s.schema_id == meta.current_schema_id)
            .map(|s| {
                s.fields
                    .iter()
                    .filter_map(|f| {
                        f.id.map(|id| (id, (f.name.as_str(), native_type(&f.r#type))))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let Some(spec) = meta
            .partition_specs
            .iter()
            .find(|s| Some(s.spec_id) == meta.default_spec_id)
            .or_else(|| meta.partition_specs.first())
        else {
            return Vec::new();
        };

        spec.fields
            .iter()
            .enumerate()
            .filter_map(|(idx, pf)| {
                let (source_column, data_type) = field_map.get(&pf.source_id)?;
                Some(Partition::new(
                    idx + 1,
                    pf.name.clone(),
                    *source_column,
                    data_type.clone(),
                    parse_transform(&pf.transform),
                ))
            })
            .collect()
    }

    fn snapshot(meta: &TableMetadata) -> SnapshotMetadata {
        match meta.current_snapshot_id {
            Some(id) => SnapshotMetadata::new(SnapshotVersion::Id(id), Some(id)),
            None => SnapshotMetadata::current_unpinned(),
        }
    }
}

#[async_trait]
impl CatalogProvider for IcebergRestCatalogProvider {
    fn id(&self) -> CatalogId {
        self.id.clone()
    }

    fn dialect(&self) -> Option<Dialect> {
        Some(Dialect::Iceberg)
    }

    async fn list_tables(
        &self,
        namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind> {
        let identifiers = self.list_tables_raw(&namespace.dotted()).await?;
        Ok(identifiers
            .into_iter()
            .map(|id| TableRef::new(NamespaceRef::new(id.namespace), id.name))
            .collect())
    }

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind> {
        let meta = self
            .load_table(&table.namespace.dotted(), &table.name)
            .await?;
        let fqn = table.fully_qualified();
        let schema = Self::current_schema(&meta, &fqn)?;
        let location =
            meta.location
                .clone()
                .ok_or_else(|| CatalogProviderErrorKind::MalformedResponse {
                    msg: format!("table metadata for {fqn} has no location"),
                })?;
        Ok(ResolvedTable::new(
            table.clone(),
            Path::new(location),
            FileFormat::Iceberg,
            schema,
            Self::partitions(&meta),
            Self::snapshot(&meta),
        ))
    }

    async fn get_schema(&self, table: &TableRef) -> Result<Schema, CatalogProviderErrorKind> {
        let meta = self
            .load_table(&table.namespace.dotted(), &table.name)
            .await?;
        Self::current_schema(&meta, &table.fully_qualified())
    }

    async fn get_partitions(
        &self,
        table: &TableRef,
    ) -> Result<Vec<Partition>, CatalogProviderErrorKind> {
        let meta = self
            .load_table(&table.namespace.dotted(), &table.name)
            .await?;
        Ok(Self::partitions(&meta))
    }

    // TODO: honor a pinned `SnapshotVersion::Id`; this returns the current snapshot.
    // It needs a metadata read through the Iceberg REST `snapshot-id` selector.
    async fn get_snapshot(
        &self,
        table: &TableRef,
        _version: SnapshotVersion,
    ) -> Result<SnapshotMetadata, CatalogProviderErrorKind> {
        let meta = self
            .load_table(&table.namespace.dotted(), &table.name)
            .await?;
        Ok(Self::snapshot(&meta))
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

#[derive(Deserialize)]
pub(crate) struct TokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ListTablesResponse {
    identifiers: Vec<TableIdentifier>,
}

#[derive(Debug, Deserialize)]
struct TableIdentifier {
    namespace: Vec<String>,
    name: String,
}

#[derive(Debug, Deserialize)]
struct LoadTableResponse {
    metadata: TableMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct TableMetadata {
    current_schema_id: i32,
    #[serde(default)]
    schemas: Vec<IcebergSchema>,
    #[serde(default)]
    current_snapshot_id: Option<i64>,
    #[serde(default)]
    partition_specs: Vec<IcebergPartitionSpec>,
    #[serde(default)]
    default_spec_id: Option<i32>,
    #[serde(default)]
    location: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct IcebergSchema {
    schema_id: i32,
    fields: Vec<IcebergField>,
}

#[derive(Debug, Deserialize)]
struct IcebergField {
    #[serde(default)]
    id: Option<i32>,
    name: String,
    r#type: serde_json::Value,
    required: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct IcebergPartitionSpec {
    spec_id: i32,
    fields: Vec<IcebergPartitionField>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
struct IcebergPartitionField {
    source_id: i32,
    name: String,
    transform: String,
}

fn encode_namespace(namespace: &str) -> String {
    namespace.replace('.', "%1F")
}

fn native_type(type_val: &serde_json::Value) -> String {
    match type_val {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn parse_transform(transform: &str) -> PartitionTransform {
    let lower = transform.to_ascii_lowercase();
    match lower.as_str() {
        "identity" => PartitionTransform::Identity,
        "year" => PartitionTransform::Year,
        "month" => PartitionTransform::Month,
        "day" => PartitionTransform::Day,
        "hour" => PartitionTransform::Hour,
        "void" => PartitionTransform::Void,
        _ => parse_parametric_transform(&lower).unwrap_or(PartitionTransform::Identity),
    }
}

fn parse_parametric_transform(lower: &str) -> Option<PartitionTransform> {
    let (kind, rest) = lower.split_once('[')?;
    let n: u32 = rest.strip_suffix(']')?.parse().ok()?;
    match kind {
        "bucket" => Some(PartitionTransform::Bucket(n)),
        "truncate" => Some(PartitionTransform::Truncate(n)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iceberg_rest_names_the_iceberg_dialect() {
        let provider = IcebergRestCatalogProvider::new("test", "https://example.test");
        assert_eq!(provider.dialect(), Some(Dialect::Iceberg));
    }

    #[test]
    fn test_encode_namespace() {
        assert_eq!(encode_namespace("sales"), "sales");
        assert_eq!(encode_namespace("prod.sales"), "prod%1Fsales");
        assert_eq!(
            encode_namespace("catalog.db.schema"),
            "catalog%1Fdb%1Fschema"
        );
    }

    #[test]
    fn native_type_captures_primitive_verbatim() {
        assert_eq!(native_type(&serde_json::json!("long")), "long");
        assert_eq!(
            native_type(&serde_json::json!("timestamptz")),
            "timestamptz"
        );
        assert_eq!(
            native_type(&serde_json::json!("decimal(18,2)")),
            "decimal(18,2)"
        );
    }

    #[test]
    fn native_type_serializes_nested_verbatim() {
        let nested = serde_json::json!({"type": "list", "element": "long"});
        let result: serde_json::Value = serde_json::from_str(&native_type(&nested)).unwrap();
        assert_eq!(result, nested);
    }

    #[test]
    fn parse_transform_handles_named_and_parametric() {
        assert_eq!(parse_transform("identity"), PartitionTransform::Identity);
        assert_eq!(parse_transform("DAY"), PartitionTransform::Day);
        assert_eq!(
            parse_transform("bucket[16]"),
            PartitionTransform::Bucket(16)
        );
        assert_eq!(
            parse_transform("truncate[10]"),
            PartitionTransform::Truncate(10)
        );
        assert_eq!(parse_transform("weirdo"), PartitionTransform::Identity);
    }

    #[test]
    fn test_builder_pattern() {
        let catalog = IcebergRestCatalogProvider::new("ice", "https://polaris.example.com")
            .with_warehouse("my_warehouse")
            .with_prefix("prefix")
            .with_bearer_token("token123");

        assert_eq!(catalog.id().as_str(), "ice");
        assert_eq!(catalog.base_url(), "https://polaris.example.com");
        assert_eq!(catalog.warehouse.as_deref(), Some("my_warehouse"));
        assert_eq!(catalog.prefix.as_deref(), Some("prefix"));
        assert!(matches!(catalog.auth, AuthConfig::BearerToken(_)));
    }

    #[test]
    fn test_api_path() {
        let catalog = IcebergRestCatalogProvider::new("ice", "https://host:8181");
        assert_eq!(catalog.api_path(), "https://host:8181/v1");

        let catalog =
            IcebergRestCatalogProvider::new("ice", "https://host:8181").with_prefix("my-catalog");
        assert_eq!(catalog.api_path(), "https://host:8181/v1/my-catalog");
    }

    #[test]
    fn test_parse_list_tables_response() {
        let json = serde_json::json!({
            "identifiers": [
                {"namespace": ["default"], "name": "orders"},
                {"namespace": ["default"], "name": "customers"}
            ]
        });
        let resp: ListTablesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.identifiers.len(), 2);
        assert_eq!(resp.identifiers[0].name, "orders");
    }

    #[test]
    fn current_schema_captures_native_type_strings() {
        let json = serde_json::json!({
            "metadata": {
                "current-schema-id": 0,
                "schemas": [{
                    "schema-id": 0,
                    "fields": [
                        {"id": 1, "name": "id", "type": "long", "required": true},
                        {"id": 2, "name": "name", "type": "string", "required": false},
                        {"id": 3, "name": "amount", "type": "decimal(18,2)", "required": false}
                    ]
                }]
            }
        });
        let resp: LoadTableResponse = serde_json::from_value(json).unwrap();
        let schema = IcebergRestCatalogProvider::current_schema(&resp.metadata, "ns.t").unwrap();
        assert_eq!(schema.columns.len(), 3);
        assert_eq!(schema.columns[0].data_type, "long");
        assert!(!schema.columns[0].nullable);
        assert_eq!(schema.columns[2].data_type, "decimal(18,2)");
        assert!(schema.columns[2].nullable);
    }

    #[test]
    fn partitions_resolve_source_column_and_transform() {
        let json = serde_json::json!({
            "metadata": {
                "current-schema-id": 0,
                "schemas": [{
                    "schema-id": 0,
                    "fields": [
                        {"id": 1, "name": "id", "type": "long", "required": true},
                        {"id": 2, "name": "order_date", "type": "timestamp", "required": true},
                        {"id": 4, "name": "region", "type": "string", "required": false}
                    ]
                }],
                "current-snapshot-id": 3497810964824022504_i64,
                "default-spec-id": 0,
                "partition-specs": [{
                    "spec-id": 0,
                    "fields": [
                        {"source-id": 2, "field-id": 1000, "name": "order_date_day", "transform": "day"},
                        {"source-id": 4, "field-id": 1001, "name": "region_identity", "transform": "identity"}
                    ]
                }],
                "location": "s3://warehouse/db/orders"
            }
        });
        let resp: LoadTableResponse = serde_json::from_value(json).unwrap();
        let meta = &resp.metadata;
        let parts = IcebergRestCatalogProvider::partitions(meta);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].position, 1);
        assert_eq!(parts[0].name, "order_date_day");
        assert_eq!(parts[0].source_column, "order_date");
        assert_eq!(parts[0].data_type, "timestamp");
        assert_eq!(parts[0].transform, PartitionTransform::Day);
        assert_eq!(parts[1].transform, PartitionTransform::Identity);

        let snap = IcebergRestCatalogProvider::snapshot(meta);
        assert_eq!(snap.snapshot_id, Some(3497810964824022504));
        assert_eq!(snap.version, SnapshotVersion::Id(3497810964824022504));
    }

    #[test]
    fn snapshot_is_unpinned_when_absent() {
        let json = serde_json::json!({
            "metadata": {
                "current-schema-id": 0,
                "schemas": [{
                    "schema-id": 0,
                    "fields": [{"id": 1, "name": "id", "type": "long", "required": true}]
                }]
            }
        });
        let resp: LoadTableResponse = serde_json::from_value(json).unwrap();
        let snap = IcebergRestCatalogProvider::snapshot(&resp.metadata);
        assert_eq!(snap.snapshot_id, None);
        assert_eq!(snap.version, SnapshotVersion::Current);
        assert!(IcebergRestCatalogProvider::partitions(&resp.metadata).is_empty());
    }
}
