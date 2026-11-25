---
id: 15-physical-source
status: ratified
version: 2.1
impl: implemented
---

# Physical Source

The PhysicalSource sum type (File, Table, Snapshot) represents a resolved data source at runtime. This document defines the three variants, Schema/PartitionColumn/CatalogRef shapes, source resolution (glob expansion, catalog fetch), and deterministic ordering. See [file-format](file-format.md) for FileFormat details and [binding-process](binding-process.md) for the compile-time resolution flow.

## 3. The `PhysicalSource`

### 3.1 Sum-type shape

A `PhysicalSource` is a compile-time-resolved physical target. The variants span the practical v1 axis:

```rust
#[non_exhaustive]
pub enum PhysicalSource {
    File {
        path: String,
        format: FileFormat,
        schema: Schema,
        partitions: Vec<PartitionColumn>,
    },
    Table {
        catalog_ref: CatalogRef,
        schema: Schema,
        partitions: Vec<PartitionColumn>,
    },
    Snapshot {
        catalog_ref: CatalogRef,
        snapshot_id: SnapshotId,
        schema: Schema,
        partitions: Vec<PartitionColumn>,
    },
}
```

Every variant carries a resolved `schema: Schema` and a resolved `partitions: Vec<PartitionColumn>`. The tree of open `#[non_exhaustive]` types propagates: adding a variant (e.g. `PhysicalSource::Stream { ... }` for a future Kafka integration) is MINOR, and adding a field to any existing variant (e.g. `File { ... , read_options: ReadOptions }`) is MINOR provided the variant is `#[non_exhaustive]` — which it is at the variant level per Rust's rules when the enum itself is `#[non_exhaustive]`.

### 3.2 `Schema` shape

The schema is the compile-time snapshot of the physical columns exposed by the source:

```rust
pub struct Schema {
    pub columns: Vec<SchemaColumn>,
}

pub struct SchemaColumn {
    pub name: ColumnName,
    pub data_type: DataType,
    pub nullable: bool,
}

pub struct ColumnName(pub String);
```

Ordering in `Schema.columns` is the source's native order (Parquet field order from the footer, Iceberg table schema field order, CSV header order, etc.). The compile stage preserves this ordering for determinism (I4) but the planner does not semantically depend on it; column lookup is by `ColumnName` through `ResolvedColumnMapping`.

`DataType` here is **logical** (I2) — the `13 §2` canonical set. Conversion from a physical-source type system (Arrow `DataType`, Iceberg type, CSV-inferred scalar) to the canonical `DataType` happens inside each `FileFormat`'s schema-resolution strategy (§4) or inside the relevant `CatalogProvider` implementation (`37`). By the time `Schema` exists, every type is logical.

`**nullable` source of truth.** Logical nullability is read from the physical source's metadata when available (Parquet: the `FieldType.required`/`optional` markers; Iceberg: `required: bool` per schema field; CSV/JSON: always `true` unless a declared schema overrides). It is not inferred from the data.

### 3.3 `CatalogRef`

```rust
pub struct CatalogRef {
    pub catalog_id: CatalogId,
    pub fqn: Fqn,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CatalogId(pub u32);

pub struct Fqn(pub String);
```

`CatalogRef` is an **opaque handle** back into the catalog registry owned by `semstrait-catalog` (`37`). The compile-time `CatalogProvider` interaction (fetch schema, fetch snapshot-id, list tables under glob) uses `catalog_id` to route to the right provider and `fqn` to address the specific table. The planner never dereferences a `CatalogRef`; it appears on `PhysicalSource::Table` / `Snapshot` so adapters (`36`) can carry the catalog identity into their engine-registration call.

`Fqn` is the dot-separated fully-qualified name in the catalog's own namespace convention (e.g. `"ns.sales.transactions"`). The `fqn`'s canonical spelling is what the catalog reports; semstrait does not normalize case or whitespace.

### 3.4 `PartitionColumn`

```rust
pub struct PartitionColumn {
    pub name: ColumnName,
    pub position: usize,
    pub data_type: DataType,
    pub nullable: bool,
}
```

`PartitionColumn.position` is **1-indexed** (matches `MetadataDimension.partition.level` per `13 §4.7` and §8 below). `position == 1` is the coarsest-declared partition column; `position == N` is the innermost. For File sources with Hive-style path-encoded partitioning (`year=*/month=*/day=*/*.parquet`), `PartitionColumn` is populated during glob expansion (§3.5) from the path components. For Table / Snapshot sources backed by Iceberg, it is populated from the partition spec that Iceberg REST returns (per `37` / legacy `CATALOG_RESOLUTION.md §4`).

**Grain inference is NOT done in `15`.** The Iceberg partition-transform vocabulary (`year`, `month`, `day`, `hour`) may carry a grain hint; consuming that hint is the planner's job (`22` Grainset, `17` TemporalShape). `15`'s `PartitionColumn` records only the resolved name/type/position; the transform, if any, is captured upstream in `37`'s `TableMetadataResponse` and attached to the `CatalogRef` / Binding via the compile-stage step that walks the catalog response. The exact placement of that transform record is a `37`-owned question.

### 3.5 Source resolution

The Model's YAML surface places a Binding's sources under `extras.storage.paths:` and `extras.storage.tables:` (per `32 §4`). Each entry resolves at compile to one or more `PhysicalSource`s; the Binding's `sources: Vec<PhysicalSource>` is the concatenation of every entry's resolution in author order across both fields. Per `21 §3.2 / §4.5`, multiple `PhysicalSource`s in one Binding compose under `Union ALL` with optional per-source pre-aggregation as a planner optimization.

A `PhysicalSource` is an **engine-level LogicalRelation** — one Substrait `ReadRel`, one DataFusion `TableScan`, one Spark `LogicalRelation`, one SQL `FROM` reference. Each resolved `PhysicalSource` carries a concrete identifier (path / FQN); engine-internal mechanics (Hive partition discovery, multi-file consolidation, schema merge for path sources; partition-spec consultation for catalog tables) live below the `PhysicalSource` boundary and are the adapter / engine's responsibility per `35 §5.2.1`.

#### 3.5.1 `paths:` resolution — `PhysicalSource::File`

- **Concrete path** (`"s3://bucket/orders/2024-01/customers.parquet"`, `"s3://bucket/orders/"`) → produces **one** `PhysicalSource::File` with the path stored verbatim. The engine treats the source as a single LogicalRelation and handles file consolidation, schema merge, and Hive-partition discovery internally.
- **Wildcard path** (`"s3://bucket/*/orders/"`, `"s3://lake/year=*/sales/"`) → compile enumerates each resolved variation via `FileSystem::expand_glob` → produces **one `PhysicalSource::File` per resolved variation**. Each resulting source has a concrete (wildcard-free) path string; downstream metadata extraction (`§8`'s `path_token`) operates on that concrete path per source.

#### 3.5.2 `tables:` resolution — `PhysicalSource::Table` / `Snapshot`

- **Concrete table FQN** (`"iceberg.sales.transactions"`) → produces **one** `PhysicalSource::Table`, or `Snapshot` if the spec carries `at: { snapshot_id: ... }`.
- **Table-name glob** (`"iceberg.sales.*_transactions"`) → compile-expanded via `CatalogProvider::list_tables` → produces **one `PhysicalSource::Table` per resolved FQN**.

#### 3.5.4 `partition_def` carriage — manifest-side, runtime-dormant in v1

`extras.storage.partition_def:` (per `32 §4`) is the canonical catalog-less partition declaration for file sources, in v1 form `Range { column }` / `List { column }`. The compile pass parses, schema-validates, and carries it verbatim onto each `PhysicalSource::File` it produces from a `paths:` entry. **No v1 plan-time logic consumes it** — adapters defer partition pruning to engine-side discovery from filter predicates per `35 §5.2.1`. The declaration is forward-compat for v2+ consumers (per-partition extraction per `Q-MAP-009`; partition-aware grain inference per `17`; planner pruning hints). This is a closure clause of `Q-MAP-002`.

#### 3.5.6 Expansion algorithm

The deterministic algorithm (I4) is:

1. **Classify each source spec** by which YAML key carried it (`paths:` → File; `tables:` → Table). `Snapshot` is produced only when a `tables:` entry carries an `at:` subkey.
2. **If the entry contains a glob metacharacter** (`*`, `?`, `[`), call the respective provider to enumerate:
  - `paths:` entries → `FileSystem::expand_glob(pattern) → Vec<String>` (ordered).
  - `tables:` entries → `CatalogProvider::list_tables(namespace, pattern) → Vec<Fqn>` (ordered).
3. **Sort the returned list lexicographically** by the full resolved identifier (absolute file path for `File`, `Fqn` for `Table`). This is the `15`-mandated determinism fence: provider ordering is not trusted to be stable across calls; `compile` sorts it explicitly.
4. **For each resolved identifier, produce one `PhysicalSource`:**
  - File → fetch format (inferred from extension when `extras.storage.format` is absent, else taken verbatim; §4) → fetch schema per §4's per-format strategy → extract `PartitionColumn`s from any Hive-style `key=value` segments in the resolved path → carry `partition_def` from the storage block (§3.5.4) → emit `File { ... }`.
  - Table → `CatalogProvider::load_table_metadata(fqn)` → emit `Table { ... }` (or `Snapshot { ... }` if an `at:` was present).
5. **Check the resolved list is non-empty per entry.** Empty → `CompileError::NoSourcesMatched { binding_id, pattern }` (§11).

Ordering is fully specified: step 3's lexical sort makes the `sources: Vec<PhysicalSource>` field of the Binding a deterministic function of `(pattern, catalog/filesystem snapshot)`. Re-running `compile` against the same pattern and the same underlying set yields the same order, byte-identical SemanticManifest (I4).

#### 3.5.7 Error model

- `CompileError::NoSourcesMatched { binding_id, pattern }` — the pattern produced zero matches. Fail-fast per `10 §3.3` / `30 §7`. Proposed code: `COMP_E_0301`.
- `CompileError::GlobExpansionFailed { binding_id, pattern, cause }` — the filesystem or catalog raised an I/O error during expansion. Surface the upstream error as `cause` (an `IntoDiagnostic`-compatible trait object). Proposed code: `COMP_E_0302`. Lives in the `COMP_E_0200-0299` sub-range (catalog/source resolution per `30 §6.2`) since the failure is at source-resolution time, not schema-assembly time — §11 maps these carefully.
- `CompileError::CatalogUnavailable { catalog_id, cause }` — the catalog whose `CatalogId` was named in the binding spec is not registered, not reachable, or returned an unexpected error outside the per-table fetch path. Proposed code: `COMP_E_0203`.

#### 3.5.8 Cross-source schema agreement within one Binding

When source resolution produces multiple `PhysicalSource`s (multiple author entries across `paths:` / `tables:`, or a wildcard expansion under one entry), their schemas MUST agree on every column that a Semantics references (full cross-source type-agreement rule is in §9.3). Soft agreement — a column exists in some sources but not all — is a `Coverage` question (per-source `Native` vs `NullFill`; §6). Hard agreement — a column exists in all sources but with different logical `DataType`s — is a compile error (`CompileError::CrossSourceTypeDisagreement`; §11).

This is one of the `15`-specific pitfalls that peers handle inconsistently: metricflow essentially forbids the pattern (one `data_source` is one table); Cube.js leaves it to the author via `rollup_join`; Iceberg handles schema evolution at the catalog level. `15`'s answer: the Binding is the atomic unit, so every source in it either provides the column (Native) or is explicitly missing it (NullFill); mixed types at the same name across sources is a Model bug.

### 3.6 Ordering and stability within the Binding

`Binding.sources` preserves the §3.5 step-3 lexical order. The planner reads sources by index when Unionset-style per-source branching is in play (`23`); the ordering is stable across any number of re-reads of the same SemanticManifest. The per-source `Coverage` is keyed on this index.

## Testing Strategy

- Glob expansion determinism: §3.5.6 step 3 lexical sort produces stable ordering; I4 byte-identical SemanticManifest
- NoSourcesMatched: COMP_E_0301 when glob matches no files
- GlobExpansionFailed: COMP_E_0302 when filesystem access fails
- Cross-source schema agreement: §3.5.8 / COMP_E_0317 when schemas differ across sources
- PhysicalSource variant construction: File, Table, Snapshot all appear in corpus
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
