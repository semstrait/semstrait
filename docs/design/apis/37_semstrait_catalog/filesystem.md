---
id: 37-filesystem
status: ratified
version: 2.10
impl: in-progress
---

# FileSystem Trait

This leaf owns the `FileSystem` trait definition, method contracts, invariants, non-goals, per-filesystem v1 implementations (Local/S3/Azure/Gcs), and glob expansion (shared utility, composition with CatalogProvider, determinism).

## 5. `FileSystem` trait

### 5.1 Purpose

`FileSystem` abstracts byte-level I/O over a remote or local blob/object store. It is **strictly** format-agnostic and schema-unaware: it moves bytes and lists entries. Format interpretation (Parquet, CSV, JSON, …) is an adapter concern; schema knowledge is a `CatalogProvider` concern.

### 5.2 Trait definition

```rust
use async_trait::async_trait;
use bytes::Bytes;

#[async_trait]
pub trait FileSystem: Send + Sync + std::fmt::Debug {
    async fn list(
        &self,
        prefix: &Path,
    ) -> Result<Vec<FileEntry>, FileSystemErrorKind>;

    async fn read(
        &self,
        path: &Path,
    ) -> Result<Bytes, FileSystemErrorKind>;

    async fn write(
        &self,
        path: &Path,
        data: Bytes,
    ) -> Result<(), FileSystemErrorKind>;

    async fn exists(
        &self,
        path: &Path,
    ) -> Result<bool, FileSystemErrorKind>;
}
```

`Path` is the URI-shaped newtype defined in `catalog-crate-surface.md §2.3` (e.g. `s3://bucket/key`, `file:///abs/path`, `gs://bucket/key`, `abfss://container@account.dfs.core.windows.net/path`). Each concrete implementation validates scheme at call boundary and rejects non-matching URIs with `FileSystemErrorKind::UnsupportedScheme` (`catalog-crate-surface.md §8.2`). Accepting a typed `&Path` rather than a raw `&str` keeps accidental string-path confusion out of the trait surface and gives `FileSystem` impls a stable value to pattern-match.

Each method returns a **bare** `FileSystemErrorKind` per `31 §3.1`'s
construction-site convention — `FileSystem` is a transport trait. Consumers
wrap into their own typed-kind via `From<FileSystemErrorKind>` impls (e.g.
the `expand_glob` helper in `§7.1` returns `FileSystemErrorKind` directly;
`33 §16.5` consumers wrap into `CompileError`).

### 5.3 Method contracts

- **`list`** — Returns every object whose key begins with `prefix`. Non-recursive semantics are NOT implied; implementations return all descendants. Results are NOT sorted — callers sort client-side when determinism is required.
- **`read`** — Returns the full object body as `Bytes`. Streaming reads are not in v1; large-object streaming is a MINOR extension (see `questions/open/37 Q-CAT-005`).
- **`write`** — Writes `data` as a single object at `path`. Overwrites by default. Atomicity is provider-defined — typically atomic on S3/Azure/GCS, non-atomic on local disk unless the implementation stages to a temp file.
- **`exists`** — Returns `true` iff an object exists at exactly `path`. Does NOT test prefix existence.

### 5.4 Invariants

| Ref | Invariant |
|-----|-----------|
| I11 | `FileSystem` is compile-time and artifact-output async. Adapter-hot-path I/O goes through engine-native reads (not this trait). |
| I10 | `FileSystemErrorKind`, `FileEntry` are `#[non_exhaustive]`. |
| I12 | `FileSystemErrorKind` is identified by **variant identity** per `30 §5`; numeric `FS_E_*` codes are retired alongside the workspace-wide stable-code retirement. Severity is conveyed via `Diagnose::severity()` per `31 §3`. |
| —   | No format-aware logic. `read` returns bytes; interpretation is not this trait's responsibility. |
| —   | No schema-aware logic. `FileSystem` MUST NOT parse Parquet footers, CSV headers, JSON objects, etc. |

### 5.5 Non-goals

- No range reads, no multipart uploads, no copy/move/rename as first-class operations in v1. (Move = read + write + delete; delete itself is also deferred — see `Q-CAT-005`.)
- No credential management surface on the trait. Credentials are internal configuration of the concrete implementation.
- No caching. Callers wrap if needed.

---

## 6. Per-filesystem implementations (v1 roster)

Each concrete `FileSystem` (except `InMemoryFileSystem`) is a **thin wrapper over
`object_store`** (Apache Arrow — the chosen byte-transport engine). `object_store`
supplies the atomic writes, credential chains, and retry behavior; the wrapper's
only job is scheme validation at the call boundary, `Path` translation, and
folding `object_store::Error` into `FileSystemErrorKind`. **`object_store` is an
internal detail (the facade rule): its `ObjectStore` / `Path` / error types
never appear on a public signature** — the wrapper surface is `FileSystem`-only,
enforced by code review + API audit. No `FileSystem` impl carries an opt-in
escape hatch. (One existed elsewhere in `common` on
`S3SourceBuilder::with_object_store_builder` — a builder that was not a
`FileSystem` and reached none of these impls. It was deleted with the Lane-A
transport, `47 §4a` **R6**, so no hatch survives anywhere.)
A backend `object_store` does not cover (e.g. a document/KV store) is not a
`FileSystem` at all; it becomes its own `Repository<T>` impl over its best
library (`31b-io-repository §6`).

| Impl | Internal wrapping (`object_store`) | Feature flag |
|------|-------------------------------------|--------------|
| `LocalFileSystem`  | `object_store::local::LocalFileSystem`  | always under `io` |
| `S3FileSystem`     | `object_store::aws::AmazonS3`           | `io-aws` |
| `GcsFileSystem`    | `object_store::gcp::GoogleCloudStorage` | `io-gcs` |
| `AzureFileSystem`  | `object_store::azure::MicrosoftAzure`   | `io-azure` |
| `InMemoryFileSystem` | none — process-local `HashMap` (or `object_store::memory::InMemory`) | always under `io` |

Adding an `object_store`-backed backend (GCS/Azure/HTTP) is ~30 LOC of trait
delegation plus a feature flag: the same `object_store` delegation shape these
impls carry (atomic temp-file+rename, credential chains, per-call client build).

### 6.1 `LocalFileSystem`

| Aspect             | Detail                                                                 |
|--------------------|------------------------------------------------------------------------|
| Accepted schemes   | `file://`, and relative/absolute unadorned paths.                      |
| Auth model         | OS filesystem permissions.                                             |
| Use cases          | Tests, local development, one-box deployments.                         |

### 6.2 `S3FileSystem`

| Aspect             | Detail                                                                 |
|--------------------|------------------------------------------------------------------------|
| Accepted schemes   | `s3://`, `s3a://`                                                      |
| Auth model         | AWS credential chain (env / profile / instance role / IMDS).           |
| Endpoint           | Configurable (AWS, R2, MinIO, Ceph-S3).                                |
| Consistency        | Read-after-write consistent per AWS S3 strong-consistency guarantees. |
| Client lifetime    | The `object_store` client is built **per call**, not cached: `new_with_prefix` canonicalizes and so fails on an absent root, and deferring it keeps `new` infallible. Cost is one build per operation (~85 ms and up to three IMDS round-trips when credentials come from an instance role). Accepted for v1 — no v1 caller is on a hot path, and caching would need rulings on credential expiry and on whether a failed build poisons the cache. |

### 6.3 `AzureFileSystem`

| Aspect             | Detail                                                                 |
|--------------------|------------------------------------------------------------------------|
| Accepted schemes   | `abfss://`, `abfs://`                                                  |
| Auth model         | Azure AD / managed identity / SAS token / account key.                 |
| Use cases          | Azure Data Lake Storage Gen2 (hierarchical namespace supported).       |

### 6.4 `GcsFileSystem`

| Aspect             | Detail                                                                 |
|--------------------|------------------------------------------------------------------------|
| Accepted schemes   | `gs://`                                                                |
| Auth model         | Application default credentials / service-account JSON key.            |

### 6.5 `InMemoryFileSystem`

| Aspect             | Detail                                                                 |
|--------------------|------------------------------------------------------------------------|
| Accepted schemes   | Scheme-free in-process paths (no URI dispatch).                        |
| Auth model         | None — process-local `HashMap<Path, Bytes>` behind an async lock.      |
| Use cases          | Tests, fixtures, one-box runs; the in-memory transport under `FileSystemRepository<T>` for warning-free persistence round-trips (`31b-io-repository §7`). |

`InMemoryFileSystem` is `Clone + Debug + Default` and `Arc`-shareable; a clone shares the same backing store. It implements all four `FileSystem` methods and is the in-memory persistence substrate for **every** DTO — no per-DTO in-memory backend is needed (`31b-io-repository §7`).

It is not a faithful double in one measured respect: it keys on the URI **verbatim**, so a path bearing a `..` or `.` segment is stored and listed at that unresolved key, while every `object_store`-backed impl above routes through `object_store::path::Path::parse` and refuses it (`Io(Malformed)`, nothing written). The in-memory transport is therefore the *less* strict of the two, and a suite that only exercises it cannot see a path a real backend would reject (`31b-io-repository §4.1` pins both halves).

Verbatim keying diverges on normalization too, not only on refusal: `"root"` and `"file:///root"` are **one** location on an `object_store`-backed impl and **two** distinct keys in memory (measured; `"root"`/`"root/"` and `""`/`"/"` collapse on both). So an in-memory test asserting isolation between a bare prefix and its scheme-qualified form passes while being false in production. Both divergences are tracked as `TD-021`; until that is ruled on, a seam contract that depends on key identity is pinned over a real backend as well.

### 6.6 Roster summary

Implementation selection is caller-side: the caller constructs the concrete `FileSystem` (or a `Vec<Arc<dyn FileSystem>>` dispatched by scheme) and injects it into the `CatalogProvider`. No **multi-filesystem** scheme-dispatch facility is exposed by the crate root in v1; if routing one call across several backends is needed, callers compose it themselves. See `questions/deferred/37 Q-CAT-004`.

Selecting a *single* backend from a *single* base URI is a different thing and is permitted: `FileSystemRepository::open(base)` does exactly that in `common` (`47 §4a` **R4**). What this section forbids is a crate-root router over a backend set; what it protects is that each `FileSystem` impl still rejects a foreign scheme at its call boundary (`§5.2`).

---

## 7. Glob expansion

### 7.1 Shared utility

**Unimplemented in v1.** Measured: no `expand_glob` exists in code, and `common::glob`
ships the `GlobPattern` newtype with no matching logic. Its error variant
`FileSystemErrorKind::InvalidGlob` was consequently deleted (`47 §4a` **R5**) and is
owed back when this section is built — see `[TD-GLOB-VARIANT-DEBT]`.

```rust
pub async fn expand_glob(
    fs: &dyn FileSystem,
    pattern: &GlobPattern,
) -> Result<Vec<Path>, FileSystemErrorKind>;
```

`GlobPattern` is re-exported from `semstrait-common` (`31 §14.4`-adjacent per `Q-CAT-002`). `expand_glob` is the single public glob-expansion entry point. It:

1. Parses `pattern` into a fixed prefix (everything up to the first glob metacharacter) and a suffix pattern.
2. Wraps the prefix in a `Path` and calls `fs.list(&prefix)` to enumerate candidates.
3. Filters candidate `FileEntry.path` values against the suffix using glob semantics defined in `31` (via `semstrait_common::glob_match` or equivalent — see `questions/open/37 Q-CAT-002`).
4. Returns the lexicographically-sorted list of matching `Path`s.

Globbing supports `*`, `**`, `?`, and character classes `[abc]` as defined by `semstrait-common`'s glob module. `**` matches any number of path segments; `*` matches within a single segment.

### 7.2 Composition with `CatalogProvider`

`FilesystemCatalogProvider` internally uses `expand_glob` when resolving a table reference of the form `root/namespace/table_glob`. Other `CatalogProvider`s do NOT use `expand_glob` — they expand glob patterns through their native catalog APIs (e.g. `GET /v1/namespaces/{ns}/tables` followed by client-side filtering).

SemanticManifest compilation also calls `expand_glob` directly for glob-bound `Source::Path { glob: ... }` bindings (`15 §6.3`), bypassing the catalog.

### 7.3 Determinism

`expand_glob` MUST return results sorted lexicographically. This aligns with `00 §9 I4` (manifest determinism) — a compiled manifest that includes glob-expanded paths must be byte-identical across compiler runs against the same filesystem state.

---

## Testing Strategy

- FileSystem trait definition (§5.2: list/read/write/exists method signatures, Send+Sync+Debug bounds, bare FileSystemErrorKind results)
- Method contracts (§5.3: list returns all descendants, read returns full Bytes, write overwrites, exists tests exact path) and invariants (§5.4: I11 compile-time+artifact-output async, I10 non_exhaustive, I12 variant identity, no format/schema logic)
- Per-filesystem roster (§6: LocalFileSystem file://+OS permissions, S3FileSystem s3://+AWS cred chain, AzureFileSystem abfss://+Azure AD, GcsFileSystem gs://+ADC, InMemoryFileSystem scheme-free in-process store §6.5, caller-side implementation selection per §6.6). The in-memory double's verbatim keying is asserted against a real backend on both divergences (§6.5): the refusal of the same `..`-bearing path, and the collapse of `"root"` with `"file:///root"` into one location — so the fidelity gap stays visible rather than being discovered through a suite that passes only in memory.
- object_store delegation (§6: each concrete FileSystem thin-wraps the named object_store backend; the facade rule — no object_store type on a public signature, no exception; API audit confirms the wrapper surface is `FileSystem`-only). S3FileSystem builds its client per call rather than caching it (§6.2)
- Glob expansion shared utility (§7.1: expand_glob signature, parse prefix+suffix, fs.list+filter, lexicographic sort, supports */\*\*/?/[abc])
- Composition with CatalogProvider (§7.2: FilesystemCatalogProvider uses expand_glob, other providers native API, SemanticManifest direct expand_glob for Source::Path glob)
- Determinism (§7.3: lexicographically sorted results per I4, byte-identical manifest across runs)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
