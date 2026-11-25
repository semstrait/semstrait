# semstrait-catalog

Structured catalog-metadata access for the semstrait toolchain.

Provides the object-safe `CatalogProvider` trait for abstracting catalog
metadata access (namespaces, tables, schemas, partitions, snapshots, drift).
Consumed as `Option<&dyn CatalogProvider>` by manifest compilation and for
query-time schema-drift detection.

This crate depends ONLY on `semstrait-common`, the DAG root: no
`semstrait-ir`, no `semstrait-model`, no `semstrait-manifest`. Byte-level I/O is
NOT here — that is `semstrait_common::io::FileSystem`.

---

## Native-type carrier

Column and partition *types* are carried as the provider's **native engine type
string** (e.g. `"timestamptz"`, `"DECIMAL(18,2)"`), never a canonical enum. This
crate performs no native→canonical conversion; that happens in
`semstrait-manifest`'s catalog compile stage, which reads a spelling in two
steps — the canonical names and aliases first, then the dialect that
`CatalogProvider::dialect` names — and fails the compile for a referenced
column whose spelling names no type under either step (ADR 0086, rule 7). This
is the only home of the native spelling: it used to be
described as mirroring `semstrait-manifest`'s `SourceColumn.source_type: String`,
which ADR 0046 deleted; the `SourceColumn` that crate declares today (ADR 0061)
holds an *optional canonical* type and no spelling, so it is not a counterpart.

---

## CatalogProvider Trait

Object-safe (`Send + Sync`, `#[async_trait]`), so `Box<dyn
CatalogProvider>` / `&dyn CatalogProvider` are usable. Every method returns a
bare `CatalogProviderErrorKind`.

```rust
#[async_trait]
pub trait CatalogProvider: Send + Sync {
    fn id(&self) -> CatalogId;
    fn dialect(&self) -> Option<Dialect> { None } // default: names no dialect
    async fn list_tables(&self, namespace: &NamespaceRef)
        -> Result<Vec<TableRef>, CatalogProviderErrorKind>;
    async fn resolve_table(&self, table: &TableRef)
        -> Result<ResolvedTable, CatalogProviderErrorKind>;
    async fn get_schema(&self, table: &TableRef)
        -> Result<Schema, CatalogProviderErrorKind>;
    async fn get_partitions(&self, table: &TableRef)
        -> Result<Vec<Partition>, CatalogProviderErrorKind>;
    async fn get_snapshot(&self, table: &TableRef, version: SnapshotVersion)
        -> Result<SnapshotMetadata, CatalogProviderErrorKind>;
    async fn check_schema_drift(&self, table: &TableRef, expected_schema: &Schema)
        -> Result<DriftReport, CatalogProviderErrorKind>;

    // Default impl over `resolve_table`.
    async fn table_exists(&self, table: &TableRef)
        -> Result<bool, CatalogProviderErrorKind> { /* ... */ }
}
```

`list_tables` carries NO glob parameter — pattern listing is a client concern.
Ordering is provider-defined; callers needing determinism sort client-side.

`dialect` names the dialect for typing that this provider's native types are
spelled in — `Dialect::Iceberg` for `IcebergRestCatalogProvider`, `None` for
`UnityCatalogProvider` and `NoopCatalogProvider` (ADR 0086, rule 6). It names a
vocabulary, not a provider; `None` means the provider names no dialect, so its
native types are read by the canonical names and aliases alone (ADR 0086, rule
7).

---

## Catalog Provider Implementations

| Provider | Feature | Description |
|----------|---------|-------------|
| `NoopCatalogProvider` | *(always)* | Test double: list/partitions return empty, the rest return `TableNotFound`. |
| `IcebergRestCatalogProvider` | `iceberg` | Iceberg REST API client. OAuth2/Bearer auth, partition specs, snapshots. |
| `UnityCatalogProvider` | `unity` | Databricks Unity Catalog REST API. PAT/Bearer auth, pagination. |

Concrete `FileSystem`-backed providers (local / S3) are deferred; this
crate's own tests exercise the provider logic directly — parsing, builder
patterns and diffing — with no network or filesystem transport in the loop.

Of the provider and credential types, only `NoopCatalogProvider` is `Debug`: no
type that holds a credential — `IcebergRestCatalogProvider`,
`UnityCatalogProvider`, `AuthConfig`, `TokenResponse`, `UnityAuth`,
`PolarisSecret` — implements it, as the test
`only_the_types_that_hold_no_credential_are_debug` in `src/lib.rs` asserts.

### NoopCatalogProvider

```rust
use semstrait_catalog::{CatalogProvider, NoopCatalogProvider, NamespaceRef};

let catalog = NoopCatalogProvider::default();
let tables = catalog.list_tables(&NamespaceRef::new(["my_namespace"])).await?;
assert!(tables.is_empty());
```

---

## Key Types

| Type | Description |
|------|-------------|
| `CatalogId` | Human-meaningful provider-instance id. |
| `NamespaceRef` | Ordered namespace path (`["sales", "prod"]` for `sales.prod`). |
| `TableRef` | Fully-qualified table reference: `(catalog?, namespace, name)`. |
| `SchemaColumn` / `Schema` | Column (name, native type string, nullability) and its collection. |
| `FileFormat` | Storage-format discriminator (Parquet / Delta / Iceberg / …). |
| `Partition` / `PartitionTransform` | Partition-spec descriptor and its transform. |
| `SnapshotVersion` / `SnapshotMetadata` | Snapshot selector and resolved metadata. |
| `ResolvedTable` | Aggregate: location, format, schema, partitions, snapshot. |
| `DriftReport` / `DriftStatus` / `DriftKind` | Schema-drift comparison result. |
| `CatalogProviderErrorKind` | Typed-kind error enum implementing `Diagnose`. |

The byte-transport value types (`FileSystem`, `Path`, `FileEntry`) are
`semstrait_common::io`'s, and a caller imports them from there (ADR 0090, rule 6).

---

## Error model

`CatalogProviderErrorKind` is a `#[non_exhaustive]` typed-kind enum (variant
identity, no string codes) implementing `semstrait_common::diagnostic::Diagnose`
(`message()` + `severity_default()`) — the same posture as
`FileSystemErrorKind`. It does NOT implement `std::error::Error`; consumers wrap
it into their own typed-kind via `From` at the call site.

---

## Dependencies

- `semstrait-common` -- `Path`, `Diagnose`
- `async-trait` -- object-safe async trait support
- `tokio` -- the `RwLock` over the Iceberg provider's cached OAuth2 token, and the runtime of the async tests
- `serde` -- value-type (de)serialization
- `tracing` -- the Iceberg provider's debug events when it fetches or refreshes an OAuth2 token
- `reqwest` (optional, behind `iceberg`/`unity`) -- HTTP client
- `serde_json` (optional, behind `iceberg`/`unity`) -- JSON parsing
- `aws-config` / `aws-credential-types` / `aws-sdk-secretsmanager` (optional, behind `aws`) -- Polaris credential resolution via AWS Secrets Manager
