---
id: 37-catalog-crate-surface
status: ratified
version: 2.3
impl: in-progress
---

# Semstrait-Catalog Crate Surface

This leaf owns the public API roster for `semstrait-catalog`: trait definitions (`CatalogProvider`, `FileSystem`), built-in provider implementations, error-kind enums, structural value types, module layout, caller responsibilities, stability posture, and crate boundaries.

---

## 1. Purpose, scope, layering

### 1.1 Purpose

`semstrait-catalog` provides the single narrow waist through which the semstrait toolchain reaches external metadata systems and byte-level storage. It exposes two independent abstractions:

- `CatalogProvider` — structured metadata access (namespaces, tables, schemas, partitions, snapshots) against a catalog service (Iceberg REST, Unity Catalog, or a filesystem-derived catalog).
- `FileSystem` — format-agnostic byte-level I/O against an object store or local disk (list, read, write, exists).

Both traits are `async` because every real-world implementation performs network I/O. Both are **open** per `30 §10` — third parties MAY implement them to plug additional metadata sources or storage backends into the toolchain without modifying any crate in the `3x` map.

### 1.2 Scope — what this crate owns

- Trait definitions (`CatalogProvider`, `FileSystem`) and their structural value types (`CatalogId`, `Path`, `TableRef`, `NamespaceRef`, `ResolvedTable`, `Schema`, `SchemaColumn`, `Partition`, `PartitionTransform`, `FileFormat`, `SnapshotVersion`, `SnapshotMetadata`, `DriftReport`, `DriftStatus`, `DriftKind`, `FileEntry`).
- Built-in `CatalogProvider` implementations: `NoopCatalogProvider`, `IcebergRestCatalogProvider`, `UnityCatalogProvider`, `FilesystemCatalogProvider`.
- Built-in `FileSystem` implementations: `LocalFileSystem`, `S3FileSystem`, `AzureFileSystem`, `GcsFileSystem`.
- Error types (`CatalogProviderErrorKind`, `FileSystemErrorKind`) — typed-kind enums per `30 §5` / `31 §3`; identification by variant identity. Legacy `CAT_E_*` / `FS_E_*` codes are retired.
- Shared glob-expansion utility (`expand_glob`) that composes `FileSystem` with a client-supplied pattern.
- The schema-drift gated I/O contract invoked by `semstrait-manifest` under I11b, and no other query-time I/O entry point.

### 1.3 Scope — what this crate does NOT own

- **No planning.** Has no awareness of `SemanticPlan`, `PlanNode`, `PhysicalExpr`, or any tree node type from `semstrait-ir` (`35`).
- **No SQL emission.** Does not render SQL, nor any engine-specific string. SQL generation is exclusively an adapter concern (`36`).
- **No format-header schema reading.** `FileSystem::read` returns raw bytes; it does NOT parse Parquet footers, ORC stripes, CSV headers, Avro schemas, etc. Schema knowledge comes from `CatalogProvider` (for tables) or from the manifest author (for glob-expanded file sets — see `15 §6.3`).
- **No catalog-specific branching.** Per `30 §3` / `00 §9 I3` (metadata-source-independence), canonical crates MUST route every catalog interaction through `CatalogProvider` without inspecting its concrete type.
- **No expression evaluation, no caching policy, no retry machinery, no schema inference from data.** All such concerns are callers' or adapters' responsibilities.

### 1.4 Layering

```
caller (semstrait-manifest at compile time, semstrait-manifest at I11b gate, tests)
    ↓ depends on traits only
CatalogProvider trait            FileSystem trait          (defined here)
    ↓                                ↓
IcebergRestCatalogProvider       LocalFileSystem
UnityCatalogProvider             S3FileSystem
FilesystemCatalogProvider ───── uses ─────→ FileSystem (composition)
NoopCatalogProvider              AzureFileSystem
                                 GcsFileSystem
```

`semstrait-catalog` depends only on `semstrait-common` (`31`). It does NOT depend on `semstrait-manifest`, `semstrait-ir`, `semstrait-planner`, or any adapter crate. This keeps metadata-source concerns behind a single narrow dependency edge and preserves the I3 independence axis.

---

## 2. Public crate surface

### 2.1 Roster

| Name                            | Kind           | `#[non_exhaustive]`? | Stability (v1) |
|---------------------------------|----------------|----------------------|----------------|
| `CatalogProvider`               | trait          | n/a (trait sealing)  | **Open**       |
| `FileSystem`                    | trait          | n/a                  | **Open**       |
| `CatalogProviderErrorKind`      | enum           | yes                  | Provisional    |
| `FileSystemErrorKind`           | enum           | yes                  | Provisional    |
| `CatalogId`                     | newtype struct | no (field-stable)    | Stable         |
| `Path`                          | newtype struct | no (field-stable)    | Stable         |
| `TableRef`                      | struct         | no (field-stable)    | Provisional    |
| `NamespaceRef`                  | struct         | no (field-stable)    | Provisional    |
| `ResolvedTable`                 | struct         | yes                  | Provisional    |
| `Schema`                        | struct         | no                   | Provisional    |
| `SchemaColumn`                  | struct         | no                   | Provisional    |
| `Partition`                     | struct         | yes                  | Provisional    |
| `PartitionTransform`            | enum           | yes                  | Provisional    |
| `FileFormat`                    | enum           | yes                  | Provisional    |
| `SnapshotVersion`               | enum           | yes                  | Provisional    |
| `SnapshotMetadata`              | struct         | yes                  | Provisional    |
| `DriftReport`                   | struct         | yes                  | Provisional    |
| `DriftStatus`                   | enum           | yes                  | Provisional    |
| `DriftKind`                     | enum           | yes                  | Provisional    |
| `FileEntry`                     | struct         | yes                  | Provisional    |
| `NoopCatalogProvider`           | struct         | no                   | Stable         |
| `IcebergRestCatalogProvider`    | struct         | no                   | Stable         |
| `UnityCatalogProvider`          | struct         | no                   | Stable         |
| `FilesystemCatalogProvider`     | struct         | no                   | Stable         |
| `LocalFileSystem`               | struct         | no                   | Stable         |
| `S3FileSystem`                  | struct         | no                   | Stable         |
| `AzureFileSystem`               | struct         | no                   | Stable         |
| `GcsFileSystem`                 | struct         | no                   | Stable         |
| `expand_glob`                   | free function  | n/a                  | Planned — unimplemented |

"Stability (v1)" follows `30 §8`. "Provisional" means field or variant additions permitted in MINOR under documented migration; breaking changes only in MAJOR. "Planned — unimplemented" means the signature is specified (`filesystem.md §7.1`) but no code exists; its error variant was deleted with it (`[TD-GLOB-VARIANT-DEBT]`).

### 2.2 Module layout (informative)

```rust
pub mod traits;             // CatalogProvider, FileSystem
pub mod types;              // CatalogId, Path, TableRef, NamespaceRef,
                            //   Schema, Partition, FileEntry, ResolvedTable, ...
pub mod error;              // CatalogProviderErrorKind, FileSystemErrorKind
pub mod glob;               // expand_glob
pub mod providers;          // NoopCatalogProvider, IcebergRestCatalogProvider, ...
pub mod filesystems;        // LocalFileSystem, S3FileSystem, ...

pub use traits::{CatalogProvider, FileSystem};
pub use types::*;
pub use error::{CatalogProviderErrorKind, FileSystemErrorKind};
pub use glob::expand_glob;
```

Public items are re-exported at the crate root per `30 §4.3`.

### 2.3 Structural-type sketches

Minimal shape per public value type; full field rosters live alongside the traits that consume them (`catalog-provider.md §3`, `filesystem.md §5`, `catalog-provider.md §9`).

```rust
/// Human-meaningful identifier for a CatalogProvider instance. Opaque newtype.
/// In v1, `compile()` takes a single `Option<&dyn CatalogProvider>` (`33 §9.1`);
/// a multi-catalog `CatalogRegistry` keyed on this id is a deferred mechanism
/// whose ownership is the open question Q-CAT-012.
pub struct CatalogId(String);

/// URI-shaped path accepted by every `FileSystem` method. Validated by the
/// concrete `FileSystem` impl at call boundary — MUST parse to one of:
///   - `file:///...` or a local absolute/relative path
///   - `s3://bucket/key`
///   - `gs://bucket/key`
///   - `abfss://container@account.dfs.core.windows.net/key`
/// Custom schemes ship via third-party `FileSystem` impls (per `30 §10`).
pub struct Path(String);

/// Fully-qualified table reference: `(catalog_id?, namespace, table)`.
#[derive(Debug, Clone)]
pub struct TableRef {
    pub catalog: Option<CatalogId>,
    pub namespace: NamespaceRef,
    pub name: String,
}

/// Ordered namespace path (`["sales", "prod"]` for `sales.prod`).
#[derive(Debug, Clone)]
pub struct NamespaceRef {
    pub parts: Vec<String>,
}

/// Returned by `CatalogProvider::resolve_table`. Captures enough metadata
/// for manifest compilation to build a `PhysicalSource` per `15 §3`.
#[non_exhaustive]
pub struct ResolvedTable {
    pub table: TableRef,
    pub location: Path,
    pub format: FileFormat,
    pub schema: Schema,
    pub partitions: Vec<Partition>,
    pub snapshot: SnapshotMetadata,
}
```

The `FileEntry` and `Partition` types are defined below:

```rust
/// Entry returned by `FileSystem::list`. Size and modified-at are hints —
/// callers MUST NOT rely on cross-provider byte-precise agreement.
#[non_exhaustive]
pub struct FileEntry {
    pub path: Path,
    pub size: u64,
    pub modified_at: Option<std::time::SystemTime>,
}

/// One partition field in a table's partition spec. Matches the shape of
/// `15 §3.4`'s `PartitionColumn` but lives on the catalog side of the
/// boundary; `semstrait-manifest` converts between the two at compile time.
#[non_exhaustive]
pub struct Partition {
    pub position: usize,                  // 1-indexed per 15 §3.4
    pub name: String,
    pub source_column: String,            // column this transform applies to
    pub data_type: DataType,              // result type after transform
    pub transform: PartitionTransform,
}

#[non_exhaustive]
pub enum PartitionTransform {
    Identity,
    Year, Month, Day, Hour,
    Bucket(u32),
    Truncate(u32),
    Void,
}
```

`DataType` and `Diagnostic<K>` / `Diagnose` are re-exported from `semstrait-common` per `31 §3` / `§4`; no duplicate definitions live here. `Schema` / `SchemaColumn` shapes follow `15 §3.2`; the fully-qualified field list is deferred pending `Q-CAT-008`.

---

## 8. `CatalogProviderErrorKind` / `FileSystemErrorKind`

> **Migration note.** Prior drafts of this document used `CatalogError` and
> `FileSystemError` enums with embedded `diagnostic: Diagnostic` fields and
> stable `CAT_E_*` / `FS_E_*` numeric codes. Those shapes are **retired**
> per `30 §5` and replaced by typed-kind enums (`*ErrorKind`) per `31 §3`:
> identification is by variant identity, severity comes from
> `Diagnose::severity()`, and source location is carried in the wrapping
> `Diagnostic<K>` envelope when the consumer wraps the bare kind. Body
> prose may still cite legacy `CAT_E_NNNN` / `FS_E_NNNN` strings as
> transitional anchors; read those as shorthand for the corresponding
> `CatalogProviderErrorKind::*` / `FileSystemErrorKind::*` variants.

### 8.1 `CatalogProviderErrorKind`

```rust
/// Typed error-kind for the `CatalogProvider` trait surface.
/// Identification by variant identity per `30 §5`; numeric `CAT_E_*`
/// codes are retired. Severity is conveyed via `Diagnose::severity()`;
/// every v1 variant is `Severity::Error` (no warning-severity catalog
/// kinds in v1). Source location (where applicable) lives in the
/// wrapping `Diagnostic<K>` envelope at the consumer's wrap site.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CatalogProviderErrorKind {
    // -- Availability / resource-presence --
    /// Catalog connection / handshake established but the catalog
    /// itself is not in a usable state (e.g. service degraded).
    NotAvailable        { msg: String },

    /// The requested table does not exist in the catalog.
    TableNotFound       { fqn: String },

    /// The requested namespace does not exist in the catalog.
    NamespaceNotFound   { ns: String },

    /// The requested snapshot version of an existing table does not
    /// exist (e.g. expired snapshot, retention pruned).
    SnapshotNotFound    { table: String, version: String },

    // -- Transport / auth --
    /// Network connect / TLS handshake / DNS failure reaching the
    /// catalog endpoint.
    ConnectionFailed    { msg: String },

    /// Authentication credentials were rejected.
    AuthFailed,

    /// Authentication succeeded but the principal is not authorized to
    /// access the named resource.
    AuthDenied          { resource: String },

    /// The catalog call exceeded its deadline.
    Timeout             { millis: u64 },
```

The protocol and contract error variants follow:

```rust
    // -- Protocol / contract --
    /// `check_schema_drift` returned `DriftStatus::Breaking`.
    /// The accompanying `DriftKind` carries the structural detail.
    SchemaDrift         { kind: DriftKind },

    /// Partition-spec metadata was structurally malformed
    /// (transform unknown, source field absent, name collision).
    MalformedPartition  { msg: String },

    /// The catalog response violated its declared contract (e.g.
    /// missing required fields, type mismatch, schema version drift).
    MalformedResponse   { msg: String },

    /// Catch-all internal error from a `CatalogProvider`
    /// implementation (mapping fault, panic safety net, etc.).
    Internal            { msg: String },
}

impl semstrait_common::diagnostic::Diagnose for CatalogProviderErrorKind {
    fn message(&self) -> std::borrow::Cow<'_, str>;
    fn severity(&self) -> semstrait_common::Severity {
        semstrait_common::Severity::Error
    }
}
```

Variant identity is the stable contract. Renaming a variant is MAJOR;
adding a variant is MINOR (`#[non_exhaustive]`); refining
`Diagnose::message()` text is PATCH. Consumers wrap into their own
typed-kind via `From<CatalogProviderErrorKind>` (e.g.
`CompileError` in `33 §10.1`).

### 8.2 `FileSystemErrorKind`

Seven variants. `InvalidGlob` and `Timeout` were removed per `47 §4a` **R5**: measured
across the whole tree, neither had a constructor, and a real timeout never reached
`Timeout` because `object_store` reports one as `Error::Generic`, which the backend
folds to `ConnectionFailed`. `InvalidGlob`'s only specified producer is `expand_glob`
(`filesystem.md §7.1`), which is **unimplemented** — when it lands it re-adds the
variant additively per `§11.3`, tracked as `[TD-GLOB-VARIANT-DEBT]`.

```rust
/// Typed error-kind for the `FileSystem` trait surface.
/// Identification by variant identity per `30 §5`; numeric `FS_E_*`
/// codes are retired. Severity is conveyed via `Diagnose::severity()`;
/// every v1 variant is `Severity::Error`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FileSystemErrorKind {
    // -- Input / resource-presence --
    /// Object does not exist at the supplied path.
    NotFound            { path: Path },

    /// The supplied URI scheme is not handled by this `FileSystem`
    /// implementation (e.g. `s3://` passed to `LocalFileSystem`).
    UnsupportedScheme   { scheme: String },

    /// The supplied path is structurally invalid (e.g. malformed URI,
    /// non-UTF-8 bytes where UTF-8 is required, scheme-internal
    /// validation failure).
    InvalidPath         { path: Path, msg: String },

    // -- Transport / auth --
    /// Network connect / TLS handshake / DNS failure reaching the
    /// storage endpoint, including a timed-out call.
    ConnectionFailed    { msg: String },
```

The remaining `FileSystemErrorKind` variants:

```rust
    /// Authentication credentials were rejected.
    AuthFailed,

    /// Authentication succeeded but the principal is not authorized to
    /// read / write the path.
    PermissionDenied    { path: Path },

    // -- Catch-all --
    /// Catch-all internal error from a `FileSystem` implementation.
    Internal            { msg: String },
}

impl semstrait_common::diagnostic::Diagnose for FileSystemErrorKind {
    fn message(&self) -> std::borrow::Cow<'_, str>;
    fn severity(&self) -> semstrait_common::Severity {
        semstrait_common::Severity::Error
    }
}
```

`FileSystemErrorKind` follows the same SemVer posture as
`CatalogProviderErrorKind` (renames MAJOR; additions MINOR; message
refinement PATCH).

### 8.3 Cross-crate wrapping

`CatalogProviderErrorKind` and `FileSystemErrorKind` are produced by the
transport traits. Stage entry-points that consume them wrap into their
own typed-kinds via `From` impls — for example `33 §10.1`'s
`CompileError::CatalogResolutionFailed { source: CatalogProviderErrorKind }`
and `RepositoryErrorKind::Io` in `31b-io-repository §5` (which supersedes the
`FromIoBytes` chain this once cited — deleted by `47 §4a` **R6**).
The wrap site attaches the
stage-relevant `Location` to the resulting `Diagnostic<K>` envelope; the
transport layer carries no `Location` field.

### 8.4 Code range registration — closed

Round-1 drafting proposed registering `CAT_E_0100`–`0399` and
`FS_E_0100`–`0199` ranges in `30 §6.2`. Both proposals are **closed without
action** as a consequence of the workspace-wide stable-code retirement
(`30 §5` typed-kind discipline). No subsystem-prefix table amendment is
required; consumers route on variant identity. See `Q-CAT-001` (closed).

---

## 10. Caller responsibilities

### 10.1 Construction

Catalog and filesystem providers are constructed at *program startup* (or at tool-invocation time for a CLI), not lazily inside the compile/plan pipeline. Typical wiring:

```rust
let fs: Arc<dyn FileSystem> = Arc::new(S3FileSystem::new(s3_config)?);
let catalog: Arc<dyn CatalogProvider> = Arc::new(
    IcebergRestCatalogProvider::builder()
        .id("polaris_prod")
        .base_url("https://polaris.example.com")
        .auth(token)
        .build()?,
);

// Compile-time (v1 ratified signature, see `33 §9.1`):
let manifest =
    semstrait_manifest::compile(model, Some(catalog.as_ref()), fs.as_ref()).await?;
```

In v1, `compile()` takes a single `Option<&dyn CatalogProvider>` plus a `&dyn FileSystem` (`33 §9.1`); the ratified signature has no registry parameter. Multi-catalog routing via a `CatalogRegistry` is therefore deferred past the v1 single-provider signature; its ownership (`semstrait-manifest` vs. `semstrait-catalog`) is the **open** question Q-CAT-012. Were it adopted, `semstrait-catalog` would expose only the *pieces* from which such a registry is built.

### 10.2 Dependency injection pattern

Every consumer takes `&dyn CatalogProvider` / `&dyn FileSystem` (or `Arc<dyn …>`), never a concrete type. This preserves I3 and lets tests substitute `NoopCatalogProvider` or an in-memory `FileSystem` without conditional compilation.

### 10.3 Thread safety

`CatalogProvider: Send + Sync` and `FileSystem: Send + Sync` are required by the trait bounds. Implementations MUST be safe to share across async tasks behind an `Arc`. Internal mutability (e.g. connection-pool state) MUST use interior-mutable synchronization (`Mutex`, `RwLock`, `tokio::sync::*`, atomic); exterior `&self` on every method is a hard requirement.

### 10.4 Error propagation

Callers propagate `CatalogProviderErrorKind` and `FileSystemErrorKind` through `?` into their own typed-kind enums (typically `CompileError` at compile time per `33 §10.1`; `33 §16.5`'s manifest I/O chain at I11b) via `From` impls. The wrapping site attaches stage-relevant location and emits a `Diagnostic<CompileError>` (or whatever the consumer's outer kind is). Variant identity remains stable across the wrap.

### 10.5 Async runtime

The trait is `async fn` via `async_trait::async_trait` in v1. Callers MUST drive futures on a `tokio`-compatible runtime (per `30 §9`). A move to native `async fn` in traits (stable as of Rust 1.75+) is a possible future simplification — see `questions/open/37 Q-CAT-007`.

---

## 11. Stability

### 11.1 Trait stability posture

Both `CatalogProvider` and `FileSystem` are **open** traits per `30 §10`:

- Third-party crates MAY implement them to plug new catalog services or storage backends.
- Method-set growth is permitted only in MAJOR releases, OR in MINOR via a default-implemented method (which does NOT break existing impls).
- Method signature changes are MAJOR-only.

To avoid the "default method lie" — where a default silently returns `None` or an empty value and a caller mistakes absence-of-support for absence-of-data — MINOR-added methods MUST have defaults that either:

1. Return an error with a dedicated typed-kind variant (e.g. `CatalogProviderErrorKind::MethodNotSupported` / `FileSystemErrorKind::MethodNotSupported` added under `#[non_exhaustive]`), OR
2. Explicitly document that the default response is semantically meaningful (e.g. "returns empty when the method is genuinely not applicable, not when the provider simply lacks support").

### 11.2 Built-in implementation stability

`NoopCatalogProvider`, `IcebergRestCatalogProvider`, `UnityCatalogProvider`, `FilesystemCatalogProvider`, `LocalFileSystem`, `S3FileSystem`, `AzureFileSystem`, `GcsFileSystem` are **Stable** in v1: their struct identities, their `new`/`builder` constructors, and their observable behavior do not break in MINOR. Field additions on their configuration types (e.g. `S3Config`) are permitted additively.

### 11.3 Error-variant stability

`CatalogProviderErrorKind` and `FileSystemErrorKind` are `#[non_exhaustive]`. MINOR releases MAY add variants. Existing variants MUST NOT change shape (variant rename or field rename / removal is MAJOR per `30 §5`'s typed-kind SemVer rules). `Diagnose::message()` text refinement is PATCH; consumers route on variant identity, not on rendered message strings.

### 11.4 Value-type stability

`CatalogId`, `Path`, `TableRef`, `NamespaceRef`, `Schema`, `SchemaColumn` are field-stable (not `#[non_exhaustive]`) — these are the shared vocabulary between this crate and its callers, and shielding them behind non-exhaustive marking would require builder boilerplate that outweighs the benefit. `ResolvedTable`, `Partition`, `PartitionTransform`, `FileFormat`, `SnapshotVersion`, `SnapshotMetadata`, `DriftReport`, `DriftStatus`, `DriftKind`, `FileEntry` are `#[non_exhaustive]` — these are more likely to grow as catalog ecosystems expose new metadata shapes.

### 11.5 Method-set growth mechanism

Method-set growth on `CatalogProvider` or `FileSystem` follows this ordered procedure:

1. Propose in an `questions/open/37_*.md` entry citing the need.
2. Draft signature + default behavior.
3. Ratify in a MINOR release with a default impl that errors on absence-of-support.
4. Built-in impls override the default in the same MINOR; third-party impls migrate at their own pace.

This procedure is the MINOR-safe path; any growth that cannot be fit into it becomes a MAJOR candidate.

---

## 12. Crate boundaries

| Boundary                                     | Status                                                                 |
|----------------------------------------------|------------------------------------------------------------------------|
| Planning (`SemanticPlan`, `PlanNode`)        | **NO.** `semstrait-catalog` has zero `semstrait-ir` / `semstrait-planner` imports. |
| SQL emission / engine-specific string gen    | **NO.** Every string produced is a URI, an identifier, or a diagnostic message — never an SQL fragment. |
| Format-header schema reading                 | **NO.** `FileSystem::read` returns raw `Bytes`. The legacy `StorageProvider::read_schema(path, DataFormat)` path is removed. Schemas come from `CatalogProvider` or from the manifest author. |
| Expression evaluation                        | **NO.** No `Expr<L>`, no evaluator. The authoring expression vocabulary (`Expr<L>`, leaf sets, support enums) lives in `semstrait-model` (`32`); the resolved `PhysicalExpr` lives in `semstrait-ir` (`35`). Catalog never sees expression trees. |
| Caching policy                               | **NO.** Callers wrap. |
| Retry / backoff machinery                    | **NO.** Individual implementations MAY retry internally as a quality-of-implementation detail, but the trait surface exposes no retry knobs. |
| Transaction scoping                          | **NO.** Each call is independent. |
| Mutation of catalog state                    | **NO in v1.** Read-only trait surface. Future: see `Q-CAT-006`. |
| Credential management                        | **NO public surface.** Credentials are internal config of each concrete impl. |
| Scheme-dispatch across FileSystems           | **NO.** Callers compose. See `Q-CAT-004`. |
| Format enumeration (`FileFormat`)            | **YES.** Owned here because `FileFormat` is a metadata attribute of resolved tables and files. Adapters and the manifest consume it as opaque metadata. |
| Partition-transform enumeration (`PartitionTransform`) | **YES (v1 limited).** `Identity`, `Year`, `Month`, `Day`, `Hour`, `Bucket(u32)`, `Truncate(u32)`, `Void` (per `§2.3`). Matches Iceberg REST v2 spec. New transforms land in MINOR via `#[non_exhaustive]`. See `Q-CAT-010`. |
| Glob semantics                               | **Delegated.** `expand_glob` uses `semstrait-common`'s glob predicate; this crate owns only the `FileSystem::list`-driven prefix-then-filter dance. |

---

## 13. Cross-references

- Overview: `00 §4 The Public Surface`, `00 §5 Layer 3 — Runtime Integration`.
- Invariants: `00 §9 I3, I10, I11 (incl. I11b), I12`.
- API contracts: `30 §3 (Open/sealed traits)`, `30 §5 (Typed-kind discipline)`, `30 §7 (Result shapes)`, `30 §8 (Stability)`, `30 §9 (Async posture)`, `30 §10 (Per-crate async table)`.
- Compile-time consumers: `15 §5 (Compile-Time Resolution)`, `15 §6 (Source resolution: paths, tables, globs)`.
- Query-time consumers: `33 (semstrait-manifest, I11b gate)` — drafted adjacent to this doc.
- Sibling crate: `31 (semstrait-common)` — shared primitives (`ColumnName`, `DataType`, `Span`, `Diagnostic<K>`, `Diagnose`, `GlobPattern`).
- Downstream: `35 (semstrait-ir)`, `34 (semstrait-planner)`, `36 (semstrait-adapter)` — do NOT import from this crate.

---

## Testing Strategy

- Public crate surface roster matches §2.1 (27 items: 2 traits, 2 error enums, 17 types, 4 per-catalog providers, 4 per-filesystem backends, expand_glob)
- Structural-type sketches field shapes (§2.3: CatalogId, Path, TableRef, ResolvedTable, FileEntry, Partition, PartitionTransform)
- CatalogProviderErrorKind variant roster (§8.1: NotAvailable, TableNotFound, NamespaceNotFound, SnapshotNotFound, ConnectionFailed, AuthFailed, AuthDenied, Timeout, SchemaDrift, MalformedPartition, MalformedResponse, Internal) and FileSystemErrorKind (§8.2: NotFound, UnsupportedScheme, InvalidPath, ConnectionFailed, AuthFailed, PermissionDenied, Internal)
- Cross-crate wrapping via From (§8.3: CatalogProviderErrorKind → CompileError, FileSystemErrorKind → CompileError, no Location in transport layer)
- Caller DI pattern Arc<dyn CatalogProvider> + Arc<dyn FileSystem> (§10.2) and thread-safety Send+Sync bounds (§10.3)
- Stability posture per-item (§11: Open traits, Stable built-in implementations, Provisional error variants, field-stable value types)
- Crate boundaries delegation table (§12: NO planning/SQL/format-header/expr/cache/retry/transaction/mutation/credentials/scheme-dispatch; YES FileFormat/PartitionTransform/glob-delegated)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
