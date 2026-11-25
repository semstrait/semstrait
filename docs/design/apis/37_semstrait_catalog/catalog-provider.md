---
id: 37-catalog-provider
status: ratified
version: 2.1
impl: in-progress
---

# CatalogProvider Trait

This leaf owns the `CatalogProvider` trait definition, method contracts, invariants, non-goals, per-provider v1 implementations (Noop/IcebergRest/Unity/Filesystem), and schema-drift gated I/O (I11b boundary).

## 3. `CatalogProvider` trait

### 3.1 Purpose

`CatalogProvider` abstracts one catalog *instance*. A single compiler invocation MAY inject multiple `CatalogProvider`s (e.g. `polaris_prod` + `polaris_dev` + `unity_main`), each keyed by its `id`. The trait is the sole interface through which manifest compilation acquires structured metadata (namespaces, tables, schemas, partitions, snapshots).

### 3.2 Trait definition

```rust
use async_trait::async_trait;

#[async_trait]
pub trait CatalogProvider: Send + Sync + std::fmt::Debug {
    fn id(&self) -> CatalogId;

    async fn list_tables(
        &self,
        namespace: &NamespaceRef,
    ) -> Result<Vec<TableRef>, CatalogProviderErrorKind>;

    async fn resolve_table(
        &self,
        table: &TableRef,
    ) -> Result<ResolvedTable, CatalogProviderErrorKind>;

    async fn get_schema(
        &self,
        table: &TableRef,
    ) -> Result<Schema, CatalogProviderErrorKind>;

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
}
```

Each method returns a **bare** `CatalogProviderErrorKind` per `31 §3.1`'s
construction-site convention — `CatalogProvider` is a transport trait, not a
stage entry-point. Consumers (notably `compile` in `33`) wrap into the
caller-side typed-kind via `From<CatalogProviderErrorKind>` for their own
kind enum, attaching a `Location` at the wrapping point if relevant.

All methods are `async`. `Send + Sync` is required so `Arc<dyn CatalogProvider>` is safely shareable across the async runtime; `Debug` supports diagnostic logging without forcing `Display`.

### 3.3 Method contracts

- **`id`** — Returns a `CatalogId` by value (cheap clone of an interned string; see `catalog-crate-surface.md §2.3`). Stable and human-meaningful (e.g. `CatalogId::from("polaris_prod")`). Used by `semstrait-manifest` to key its catalog registry and by diagnostics to cite the source of resolution. MUST NOT collide across providers within one registry — collision handling is a caller concern.

- **`list_tables`** — Returns every table directly under `namespace`. Nested namespaces are NOT traversed. Ordering is provider-defined but MUST be stable within a single snapshot of the catalog state; callers that need deterministic output (`00 §9 I4`) sort client-side. The method signature carries NO glob parameter — pattern-style listing is a client concern driven through `expand_glob` (`filesystem.md §7`) or via client-side filtering of a full `list_tables` result.

- **`resolve_table`** — Returns a `ResolvedTable` aggregating the table's `Path` location, format, schema, partition spec, and current snapshot in a single call. This is the canonical entry point for manifest compilation (`15 §5 Compile-Time Resolution`). Implementations MAY issue multiple network calls internally but MUST present the result as if captured at a single logical instant.

- **`get_schema`** — Returns the table's *current* schema. Called when only schema is needed (e.g. the caller already has location and partitions from a prior `resolve_table`). The returned `Schema` MUST be structurally identical to `resolve_table(table).await?.schema` for the same instant.

- **`get_partitions`** — Returns the list of `Partition` descriptors (position 1-indexed per `15 §3.4`, name, source column, result type, transform). Tables without partitioning return `Ok(vec![])`. The returned `Partition`s describe the **partition spec** of the table — not the set of concrete partition values or their row counts. Callers that need 15's `PartitionColumn` shape convert at the manifest boundary (`33`).

- **`get_snapshot`** — Returns `SnapshotMetadata` for a given version (or `SnapshotVersion::Current`). Snapshot pinning — the mechanism by which a compiled manifest captures a reproducible view of the catalog — goes through this call. See `15 §5.4` and `§9` below.

- **`check_schema_drift`** — The I11b-gated query-time entry point. Compares `expected_schema` (captured in the manifest at compile time) with the catalog's current schema and returns a `DriftReport`. Deterministic: same inputs → same output within a single catalog snapshot. See `§9` for the full contract.

### 3.4 Invariants

| Ref | Invariant |
|-----|-----------|
| I3  | No downstream crate branches on the concrete `CatalogProvider` type. Canonical crates see only `&dyn CatalogProvider`. |
| I11 | Compile-time async entries MAY be called during manifest compilation. The only query-time async entry is `check_schema_drift`. No other method is reachable during planning or execution. |
| I10 | `CatalogProviderErrorKind`, `Partition`, `PartitionTransform`, `FileFormat`, `ResolvedTable`, `SnapshotVersion`, `SnapshotMetadata`, `DriftReport`, `DriftStatus`, `DriftKind`, `FileEntry` are `#[non_exhaustive]`. |
| I12 | `CatalogProviderErrorKind` is identified by **variant identity** per `30 §5`; numeric `CAT_E_*` codes are retired alongside the workspace-wide stable-code retirement. Severity is conveyed via `Diagnose::severity()` per `31 §3`. |

### 3.5 Non-goals

- No `create_table`, `drop_table`, `write_snapshot`, or any mutation. The v1 surface is strictly read-oriented. Mutation lands in a future MINOR under a separate companion trait (see `questions/open/37 Q-CAT-006`).
- No transaction scoping. Each call is independent.
- No caching inside the trait. Callers may wrap with a cache layer but the trait is not the cache surface.

---

## 4. Per-provider implementations (v1 roster)

### 4.1 `NoopCatalogProvider`

| Aspect                  | Detail                                                                 |
|-------------------------|------------------------------------------------------------------------|
| `id()`                  | User-supplied via `NoopCatalogProvider::new(id)`; defaults to `"noop"`.|
| Auth model              | None.                                                                  |
| Metadata source         | In-memory; always empty.                                               |
| Namespace support       | None (every `list_tables` returns empty).                              |
| Snapshot / partition support | None.                                                             |
| Use cases               | Unit tests, stateless compilation paths, integration harnesses that only exercise file-based bindings. |

Every method returns either `Ok(empty)` or `Err(CatalogProviderErrorKind::TableNotFound { .. })` (for `resolve_table` / `get_schema` / `get_partitions` / `get_snapshot` / `check_schema_drift`). This mirrors the legacy `NullCatalogProvider` behavior and preserves the "catalog-absent" graceful-degradation path.

### 4.2 `IcebergRestCatalogProvider`

| Aspect                  | Detail                                                                 |
|-------------------------|------------------------------------------------------------------------|
| `id()`                  | User-supplied at construction.                                         |
| Auth model              | OAuth2 bearer token (Polaris default); configurable credential provider. |
| Metadata source         | Iceberg REST catalog API (`GET /v1/namespaces`, `GET /v1/namespaces/{ns}/tables`, `GET /v1/namespaces/{ns}/tables/{table}`). |
| Namespace support       | Multi-level (dotted); `NamespaceRef` round-trips with Iceberg's `namespace` list form. |
| Snapshot / partition support | Full — snapshot IDs, partition specs (with transforms: `identity`, `year`, `month`, `day`, `hour`, `bucket[N]`, `truncate[N]`, `void`), format version, location, properties. |
| Supported formats       | Parquet (primary), ORC, Avro (as reported by catalog metadata).        |
| Use cases               | Production deployments with Apache Polaris, Tabular, Iceberg REST Catalog service. |

Snapshot pinning captures `metadata.current-snapshot-id`.

Limitations: REST API only — no direct metadata-file reads. Table data files are NOT listed by this provider (that is a `FileSystem` concern via the `location` returned in `ResolvedTable`).

### 4.3 `UnityCatalogProvider`

| Aspect                  | Detail                                                                 |
|-------------------------|------------------------------------------------------------------------|
| `id()`                  | User-supplied at construction.                                         |
| Auth model              | OAuth2 personal-access-token or Databricks-issued bearer.              |
| Metadata source         | Unity Catalog REST API (`GET /api/2.1/unity-catalog/tables`, `GET /api/2.1/unity-catalog/schemas`). |
| Namespace support       | Three-level: `catalog.schema.table`. `NamespaceRef` with exactly two segments (`catalog`, `schema`). |
| Snapshot / partition support | Partition columns only (Unity does not expose snapshot IDs in v1 of this provider). `get_snapshot` returns `SnapshotMetadata` with `version: SnapshotVersion::Current` and no pin id. |
| Supported formats       | Parquet, Delta (reported as format).                                   |
| Use cases               | Databricks-hosted data; platform-agnostic Unity Catalog deployments.   |

Limitations: no snapshot pinning (versioning on Delta tables lives in the Delta log, not the Unity API). `check_schema_drift` still functions — it compares current schema to expected — but without snapshot-ID correlation. See `questions/open/37 Q-CAT-003`.

### 4.4 `FilesystemCatalogProvider`

| Aspect                  | Detail                                                                 |
|-------------------------|------------------------------------------------------------------------|
| `id()`                  | User-supplied at construction.                                         |
| Auth model              | Delegated entirely to the composed `FileSystem`.                       |
| Metadata source         | File listings under a configured root; tables are *directories*, namespaces are *parent directories*. |
| Namespace support       | Arbitrary depth, derived from directory nesting.                       |
| Snapshot / partition support | Partition columns inferred from Hive-style `key=value` subdirectories (per `15 §6.5`); no snapshot IDs. |
| Supported formats       | Parquet, CSV, JSON, ORC, Avro — inferred from file extension.          |
| Composition             | Constructed with a `Arc<dyn FileSystem>`; all metadata calls ultimately issue `FileSystem::list` / `exists`. |
| Use cases               | Data lakes without a catalog service; lake-house pre-adoption; dev/test. |

This provider is the canonical bridge: it shows that `FileSystem` alone is sufficient to stand up a minimal catalog-independent compile path, and it demonstrates the composition pattern (inject a `FileSystem` into a `CatalogProvider`) that third-party implementations may follow.

Limitations: no schema reading. `get_schema` and the `schema` field of `ResolvedTable` return `Ok(Schema::empty())` — the manifest author is expected to supply the schema explicitly or accept downstream validation against observed columns. (The legacy `StorageProvider::read_schema` path, which parsed Parquet footers, is **removed** in this design — see `catalog-crate-surface.md §12`.)

### 4.5 Roster summary

| Provider                    | Snapshot pinning | Partition metadata | Schema source | Namespace depth |
|-----------------------------|------------------|--------------------|---------------|-----------------|
| `NoopCatalogProvider`       | —                | —                  | —             | flat (empty)    |
| `IcebergRestCatalogProvider`| yes              | full (all transforms) | API       | arbitrary       |
| `UnityCatalogProvider`      | current only     | columns only       | API           | exactly 2       |
| `FilesystemCatalogProvider` | —                | Hive-style         | manifest-declared | arbitrary |

---

## 9. Schema-drift gated I/O (I11b)

### 9.1 Signature

```rust
async fn check_schema_drift(
    &self,
    table: &TableRef,
    expected_schema: &Schema,
) -> Result<DriftReport, CatalogProviderErrorKind>;
```

Where:

```rust
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DriftReport {
    pub table: TableRef,
    pub status: DriftStatus,
    pub details: Vec<DriftKind>,
    pub checked_at: std::time::SystemTime,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    Unchanged,
    Compatible,
    Breaking,
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum DriftKind {
    ColumnAdded { name: String },
    ColumnRemoved { name: String },
    ColumnRetyped { name: String, expected: DataType, actual: DataType },
    ColumnRenamed { expected: String, actual: String },
    NullabilityTightened { name: String },
    NullabilityRelaxed { name: String },
    PartitionChanged { details: String },
}
```

### 9.2 Deterministic contract

- **Idempotent per snapshot.** Two calls with identical `(table, expected_schema)` against the same underlying catalog snapshot MUST return identical `DriftReport` values except for `checked_at`. `status` and `details` (excluding ordering of additive fields) are deterministic.
- **Pure over inputs.** The method performs catalog I/O but MUST NOT mutate external state.
- **No planner involvement.** `check_schema_drift` does NOT touch `semstrait-ir`, does NOT accept or return `SemanticPlan` / `PlanNode`, and does NOT take a diagnostic sink. Failures surface as `CatalogProviderErrorKind`.

### 9.3 Caller policy

The caller (`semstrait-manifest`'s `SemanticManifest::verify_against_catalog` or equivalent) maps `DriftReport` to one of:

| `DriftStatus` | Caller response                                                                 |
|---------------|---------------------------------------------------------------------------------|
| `Unchanged`   | Proceed with execution using the compiled plan as-is.                           |
| `Compatible`  | Proceed; optionally emit an advisory diagnostic (`Severity::Info`).             |
| `Breaking`    | Abort execution; return `CatalogProviderErrorKind::SchemaDrift { kind }` (with the `DriftKind` from `DriftReport.details`) wrapped at the consumer in their own typed-kind. |

The caller MAY skip the gate entirely (e.g. dev/test mode) — this is a *caller* policy, not a trait requirement. The trait only promises to answer the question when asked.

### 9.4 I11b boundary

`check_schema_drift` is the ONLY `CatalogProvider` method that a well-behaved query-time path invokes. All other methods (`resolve_table`, `get_schema`, `get_partitions`, `get_snapshot`, `list_tables`) are compile-time only. A runtime path that invokes any non-drift method is a layering violation and SHOULD be caught in review.

---

## Testing Strategy

- CatalogProvider trait definition (§3.2: resolve_table signature returns ResolvedTable feeding 33 §8 PhysicalSource)
- Method contracts (§3.3: id returns CatalogId, list_tables stable ordering, resolve_table single logical instant, get_schema matches resolve_table, get_partitions 1-indexed position, get_snapshot per 15 §5.4, check_schema_drift deterministic per §9.2)
- Invariants (§3.4: I3 no concrete-type branching, I11 compile-time+I11b only, I10 non_exhaustive, I12 variant identity)
- Per-provider roster (§4: NoopCatalogProvider always empty/TableNotFound, IcebergRestCatalogProvider full snapshot+partition, UnityCatalogProvider 3-level namespace+current-only, FilesystemCatalogProvider Hive-style+Schema::empty)
- Schema-drift I11b (§9.1 check_schema_drift signature, §9.2 deterministic contract idempotent per snapshot+pure over inputs+no planner, §9.3 caller policy Unchanged/Compatible/Breaking→abort, §9.4 I11b boundary only query-time method)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
