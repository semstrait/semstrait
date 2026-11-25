---
id: 32-model-loader
status: ratified
version: 2.3
impl: drift
---

# Fluent Loader API

`parse` / `parse_with_profile` and `validate` are the primitives — pure inputs, pure outputs, with explicit profile control only for missing-id policy. Most callers want **parse + validate** together, returning a `SemanticModel` ready for `compile` consumption. The fluent loader is the ergonomic entry that fuses the two **plus** filesystem reads.

The loader is parametrized by a **filesystem strategy** (not a phase marker). The type parameter `F: SourceFs` selects how the loader reads source bytes — `LocalFs` for production, `InMemoryFs` for tests. This matches Rust idiom for I/O-strategy parametrization (see `figment`, `config-rs`, `tower`). Compile-time enforcement of "did the caller attach a source?" is dropped in favour of a runtime `ModelBuildErrorKind::NoSource` diagnostic; the cost (one extra runtime check) buys real testability via `InMemoryFs`.

## 1. Entry points

The loader is parametrized by a filesystem strategy and exposes fluent setters for attaching sources and controlling parse/validate behavior.

```rust
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_model::{SemanticModel, CatalogsConfig};

/// Filesystem-strategy trait. Implementors decide how a logical path
/// resolves to bytes. Sync only — remote transport is composed outside
/// this crate (§4).
pub trait SourceFs {
    fn read(&self, path: &str) -> Result<Vec<u8>, std::io::Error>;
}

/// Production strategy — delegates to `std::fs::read`.
pub struct LocalFs;
impl SourceFs for LocalFs { /* std::fs::read */ }

/// Test strategy — `HashMap<path, bytes>` lookup, miss => `ErrorKind::NotFound`.
pub struct InMemoryFs { /* ... */ }
impl InMemoryFs {
    pub fn new() -> Self;
    pub fn insert(&mut self, path: impl Into<String>, contents: impl Into<Vec<u8>>);
}
impl SourceFs for InMemoryFs { /* HashMap lookup */ }
```

The loader type itself provides fluent configuration methods and a terminating `build()`.

```rust
/// Loader for `SemanticModel`. Configured via fluent setters, terminated by `build`.
pub struct SemanticModelLoader<F: SourceFs = LocalFs> { /* fields pub(crate) */ }

impl SemanticModelLoader<LocalFs> {
    /// Default-strategy entry. Equivalent to `SemanticModel::loader()`.
    pub fn new() -> Self;
}

impl<F: SourceFs> SemanticModelLoader<F> {
    /// Swap the filesystem strategy. Returns a loader with the new `F2`.
    pub fn with_fs<F2: SourceFs>(self, fs: F2) -> SemanticModelLoader<F2>;

    /// Attach an in-memory YAML payload tagged with a logical `SourceId`.
    /// Multiple calls accumulate; sources are merged in append order.
    pub fn from_yaml_str(self, yaml: impl Into<String>, source: SourceId) -> Self;

    /// Attach an already-read byte payload — the seam for bytes this crate
    /// did not fetch (§4). UTF-8 is validated here; a non-UTF-8 payload
    /// stages `SourceIo { path: source.to_string(), error: InvalidData }`
    /// and surfaces at `build()`.
    pub fn from_yaml_bytes(self, bytes: &[u8], source: SourceId) -> Self;

    /// Resolve `path` via `self.fs.read(path)` at `build()` time.
    /// `SourceId` defaults to `path` (string form).
    pub fn from_yaml_file(self, path: impl Into<String>) -> Self;

    /// Attach a pre-parsed catalogs config. Mutually exclusive with
    /// `from_catalogs_yaml_*` (calls overwrite, last-write-wins).
    pub fn with_catalogs(self, c: CatalogsConfig) -> Self;
    pub fn from_catalogs_yaml_str(self, yaml: impl Into<String>, source: SourceId) -> Self;
    pub fn from_catalogs_yaml_file(self, path: impl Into<String>) -> Self;
}
```

The remaining setters tune the pipeline rather than feed it, and `build` terminates.

```rust
impl<F: SourceFs> SemanticModelLoader<F> {
    /// Skip the validate pass (default: validate runs). The returned
    /// model is parsed-only; SR-* / SR-E-* are not enforced. Intended
    /// for inspector / round-trip tooling.
    pub fn skip_validate(self) -> Self;

    /// Configure missing-id policy for named entities (see model-parse-validate leaf §1.1).
    /// Default is `IdentityProfile::ConvenienceGenerateMissing`.
    pub fn with_identity_profile(self, profile: IdentityProfile) -> Self;

    /// Run the configured pipeline.
    pub fn build(self) -> Result<
        (SemanticModel, Diagnostics<ModelBuildErrorKind>),
        Diagnostics<ModelBuildErrorKind>,
    >;
}

impl SemanticModel {
    /// Convenience: `SemanticModelLoader::<LocalFs>::new()`.
    pub fn loader() -> SemanticModelLoader<LocalFs> { SemanticModelLoader::new() }
}
```

The loader's error kind aggregates all stage-specific errors into a single fused sum.

```rust
/// Fused per-stage kind for the loader pipeline. Implements `Diagnose`
/// by delegating to the wrapped stage kind. Per `30 §5.6` the model crate
/// owns this fused sum because the loader composes stages whose kinds
/// live in this same crate (`ParseErrorKind`, `ValidateError`,
/// `CatalogsParseErrorKind`) plus loader-internal kinds.
#[non_exhaustive]
pub enum ModelBuildErrorKind {
    /// No source attached — `build()` was called on an empty loader.
    NoSource,
    /// `self.fs.read(path)` failed.
    SourceIo { path: String, error: std::io::ErrorKind },
    Parse(ParseErrorKind),
    CatalogsParse(CatalogsParseErrorKind),
    Validate(ValidateError),
    /// Per-field builder-internal error (e.g. invalid newtype payload).
    BuilderField { struct_name: &'static str, field: &'static str, message: String },
}

impl From<ParseErrorKind>             for ModelBuildErrorKind { /* … */ }
impl From<ValidateError>          for ModelBuildErrorKind { /* … */ }
impl From<CatalogsParseErrorKind>     for ModelBuildErrorKind { /* … */ }

impl Diagnose for ModelBuildErrorKind { /* delegates to wrapped variant */ }
```

**Stages composed by `build`** (in order, fail-fast across stages, accumulating within each stage):

1. **Read** — for each `from_yaml_file`/`from_catalogs_yaml_file` source, call `self.fs.read(path)`. `Err` ⇒ `ModelBuildErrorKind::SourceIo`.
2. **Parse model** — `parse_with_profile(&yaml, self.identity_profile)` per attached model source(s). `Err` ⇒ `ModelBuildErrorKind::Parse(_)`.
3. **Parse catalogs** — when present, `parse_catalogs(&yaml, source)`. `Err` ⇒ `ModelBuildErrorKind::CatalogsParse(_)`.
4. **Identity finalize** — confirm every named entity carries an `id` (authored or generated) and enforce profile-specific constraints via the per-layer scans (`DuplicateEntityId`, strict `MissingEntityId`).
5. **Validate** — when `skip_validate()` was NOT called, run the structural-precondition pass. `Err` ⇒ `ModelBuildErrorKind::Validate(_)`.

Within a stage, every diagnostic the stage produces is collected (parse / validate are accumulating per `30 §7.1`); across stages, the loader halts at the first stage whose Err arm fires and lifts that stage's accumulated set into `ModelBuildErrorKind`. Warnings from earlier stages that completed successfully ride through on the failing stage's Err vector — never silently dropped per `30 §7.3`.

## 2. Usage

```rust
// Production
let (model, diags) = SemanticModel::loader()
    .from_yaml_file("model.yaml")
    .from_catalogs_yaml_file("catalogs.yaml")
    .build()?;

// Testing
let mut fs = InMemoryFs::new();
fs.insert("model.yaml", REFERENCE_YAML);
let (model, _) = SemanticModelLoader::<InMemoryFs>::new()
    .with_fs(fs)
    .from_yaml_file("model.yaml")
    .build()?;
```

**FS-strategy contract.** Source attachment is runtime-checked; empty loader build returns `ModelBuildErrorKind::NoSource`. `SourceFs` is the filesystem abstraction, with `LocalFs` and `InMemoryFs` as built-ins.

**Stage API contract.** `parse` / `parse_with_profile` and `validate` remain primary stage APIs. The fluent loader is an additive wrapper that composes them without changing stage semantics.

**Composition with `semstrait-api`.** `SemStrait::compile_from_yaml` (`38 §3.3`) is the parallel API at the orchestration layer (it adds compile on top). The two lanes don't compete: in-process callers reach for `SemanticModel::loader()` to obtain a `SemanticModel` they can pass to a separate compile call (e.g. for caching mid-pipeline); end-to-end callers reach for `SemStrait::compile_from_yaml` to skip the intermediate handle. The fused `SemStraitErrorKind` (`30 §5.6`) at `semstrait-api` is parallel to `ModelBuildErrorKind` here — same cross-stage aggregation pattern, broader scope.

## 3. What the loader does NOT do

- **No remote transport.** `SourceFs` reads the local filesystem. `s3://` and every other object-store scheme are reached by the caller through `FileSystem::read` (`37`) and handed over as bytes via `from_yaml_bytes` (§4) — the sync loader never bridges to async transport, so `semstrait-model` pulls in no runtime (I11).
- **No multi-file loading.** Each `from_yaml_*` call attaches one payload. Directory walks, `$include` expansion, and cross-file merges are out of scope permanently (`Q-IO-003`, closed). Callers needing aggregation enumerate blobs on their own side and attach each one.
- **No implicit format detection.** Payloads are `semantic_model:` YAML; a non-model document fails with `ModelBuildErrorKind::Parse(_)`. YAML is the only authoring format, forever (`Q-IO-H`, resolved). JSON appears only as the manifest's machine codec (`33 §4.1`).
- **No authoring dumper.** There is no `dump_model` / `DumpMode` counterpart on this surface. Rendering a model back to bytes is `Persistable::to_bytes` (`31b §2`) — canonical, name-ordered, one form — and storing it is `Repository<SemanticModel>` (`31b §4`). The dump-time `NotRoundTrippable` guard that the retired surface carried does not move with it: every property it re-checked at write is enforced at parse instead (identifier grammar by SR-8, id canonicality by SR-11, YAML-safe scalars by the serializer), so a model that reaches `to_bytes` has already passed them and the encoder fails only on encoder error.

## 4. Composing a remote load

Bytes the loader did not fetch enter through `from_yaml_bytes`. The read is the caller's, on `FileSystem` (`37`); the parse is the loader's:

```rust
let bytes = fs.read(&Path::new("s3://bucket/models/prod.yaml")).await?;
let (model, diags) = SemanticModel::loader()
    .from_yaml_bytes(&bytes, "s3://bucket/models/prod.yaml".into())
    .build()?;
```

Local, in-memory, and remote payloads therefore converge on one parse path — `from_yaml_file` is the local-read convenience over the same seam, and all three produce the same `SemanticModel` and the same diagnostics for the same bytes.

`from_yaml_bytes` (authoring parse — human YAML, warnings) and `Persistable::from_bytes` (canonical decode — machine blob, no warnings) are deliberately distinct entries on `SemanticModel`: same type, different contract (`31b §2`).

## 5. Stability

- `SemanticModelLoader<F>`, the `SourceFs` trait, `LocalFs`, `InMemoryFs`, the `SemanticModel::loader()` entry, the `from_yaml_*` / `from_catalogs_yaml_*` / `with_fs` / `with_catalogs` / `with_identity_profile` / `skip_validate` setters, and `build` are **Stable in v1**.
- `ModelBuildErrorKind` is `#[non_exhaustive]`; new stage variants land as MINOR per `30 §2.2`. Removing or renaming a variant is MAJOR per `30 §2.1`.
- The relationship "loader composes `parse_with_profile + parse_catalogs + validate`" is a public contract — `build` MUST NOT alter parse / validate semantics; any divergence is a v1 bug.
- `LocalFs` is the documented default and is intended to remain so. Adding new built-in `SourceFs` implementations is MINOR. `SourceFs` stays local-only: remote schemes arrive as bytes (§4), so no back-end is ever added to this trait.

## Testing Strategy

Ergonomic loader testing is covered by integration tests that exercise the full pipeline (read → parse → validate) with both `LocalFs` and `InMemoryFs` strategies, verify stage composition (parse errors surface as `ModelBuildErrorKind::Parse`, validate errors as `ModelBuildErrorKind::Validate`), and confirm that warnings from earlier stages ride through on failing stage's Err arm.

- **Source parity** (§1): the same YAML attached via `from_yaml_str`, `from_yaml_bytes`, and `from_yaml_file` yields the same `SemanticModel` and the same diagnostics — verified by a three-way parity test over one fixture.
- **UTF-8 validation** (§1): a non-UTF-8 byte payload surfaces `ModelBuildErrorKind::SourceIo` with `std::io::ErrorKind::InvalidData` at `build()` and does not panic — asserted on the variant **and** its `path`, which carries the caller's `SourceId` verbatim (an empty `path` renders as ``failed to read ``: InvalidData`` and loses the only attribution a byte payload has); valid UTF-8 that is invalid YAML still yields `Parse(_)`, so the two failure classes stay distinguishable.
- **Remote composition** (§4): a model read from a `FileSystem` back-end and attached via `from_yaml_bytes` loads identically to the same bytes read locally — verified over `InMemoryFileSystem`, which keeps the test free of credentials and network.
- **No runtime dependency** (§3): this crate declares no async dependency of its own (`grep -cE '^(tokio|async-trait|futures)' Cargo.toml` == 0) and its source contains no `async fn` / `.await` — the pair that fails if transport is ever bridged in. A whole-tree `cargo tree` check would NOT work: `tokio` is already a normal transitive dep via `semstrait-common`'s default-on `io` feature, so the tree count is 3 on a green baseline (`[TD-COMMON-IO-DEFAULT]`).
