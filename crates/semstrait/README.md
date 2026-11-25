# semstrait

The facade crate: the dependency a library caller takes for `run`, the orchestrator and the Request wire form. It re-exports the types a caller names from the crates that own them, and has one free function, `run` (ADR 0009). It owns no type.

---

## run

```rust
pub async fn run(
    input: Input,
    request: &Request,
    context: &RequestContext,
    dialect: impl Into<DialectRelease>,
) -> Result<
    (substrait::proto::Plan, Diagnostics<SemstraitError>),
    (Diagnostic<SemstraitError>, Diagnostics<SemstraitError>),
>;
```

`run` builds a `Semstrait` with no catalog and the default filesystem, gives `input` to `semstrait_api::load`, then plans the Request in `dialect`. `dialect` is `impl Into<DialectRelease>`: a bare `Dialect` reads at its reference release, and a `DialectRelease` reads at the release it names, written `name=release` (a bare name is the dialect's reference release; DataFusion's is 55, its oldest 53). The warnings of both steps come back beside the plan or the refusal. The plan is a `substrait::proto::Plan`, so a caller that reads it also depends on `substrait`.

```rust
use semstrait::{run, Dialect, Input, RequestPart};

let (request, context) =
    RequestPart::read_yaml("select: [region]\ncontext: {timezone: Europe/Stockholm}")?;
let planned = run(
    Input::ModelPath("model.yaml".into()),
    &request,
    &context,
    Dialect::Substrait,
)
.await;
```

`Input` is `ModelYaml(String)`, `ModelPath(String)`, `Manifest(SemanticManifest)` or `ManifestPath(String)`; a path is local or, with the `aws` feature, an `s3://` URI. A caller with a catalog builds its own `Semstrait` with `SemstraitBuilder`: `with_catalog` takes a `CatalogProvider`, and `with_catalogs` takes a `CatalogsConfig` that `semstrait_model::parse_catalogs` reads, so that caller also depends on `semstrait-model`. `crates/semstrait-api/README.md` gives the verbs, the Request wire form, the CLI and REST.

---

## load

```rust
pub async fn load(
    semstrait: &Semstrait,
    input: Input,
) -> Result<
    (SemanticManifest, Diagnostics<SemstraitError>),
    (Diagnostic<SemstraitError>, Diagnostics<SemstraitError>),
>;
```

`load` is `run`'s first step, re-exported so a caller can hold onto the `Semstrait` it loads into. `Semstrait` is `Send + Sync`, so one long-lived instance can be shared behind an `Arc` across calls. `load` keeps every manifest it compiles in that instance's in-memory store, keyed by the model's hash, so a second `load` of an unchanged model is a store hit, not a recompile; a changed model hashes differently and compiles again. `run` builds a new `Semstrait` per call, so its store is always empty, and it never reuses a manifest across calls.

```rust
use std::sync::Arc;

use semstrait::{load, Input, SemstraitBuilder};

let semstrait = Arc::new(SemstraitBuilder::default().build());
let (manifest, _) = load(&semstrait, Input::ModelPath("model.yaml".into())).await?;
```

---

## Re-exports

| From | Types |
|---|---|
| `semstrait-api` | `Semstrait`, `SemstraitBuilder`, `SemstraitError`, `Input`, and the Request wire form: `RequestPart`, `Selected`, `Key`, `Order`, `ContextPart`, `WeekStart`, `WireError` |
| `semstrait-planner` | `Request`, `Selection`, `DimensionModifier`, `OrderKey`, `Direction`, `RequestContext`, `Timezone`, `Weekday` |
| `semstrait-model` | `UnresolvedSemanticExpr` |
| `semstrait-manifest` | `SemanticManifest` |
| `semstrait-catalog` | `CatalogProvider` |
| `semstrait-common` | `Dialect`, `DialectRelease`, `Release`, `Diagnostic`, `Diagnostics`, `FileSystem`, `LocalFileSystem`, `SemanticsName`, `DataKindName`, `Granularity` |

---

## Feature Flags

| Feature | Adds |
|---------|------|
| `datafusion` | `DataFusionRules` for `Dialect::DataFusion` (forwards to `semstrait-api`'s `datafusion`) |
| `catalog-iceberg` | Iceberg REST catalog connections, Polaris included (forwards to `semstrait-api`'s `iceberg`) |
| `aws` | `s3://` model and manifest paths for `run`, and `aws_secrets` catalog auth (forwards to `semstrait-api`'s `aws`) |
| `api-rest` | the REST routes via axum (forwards to `semstrait-api`'s `rest`) |
| `api-cli` | the CLI's parser and run entry via clap (forwards to `semstrait-api`'s `cli`) |

No feature is on by default. The facade re-exports neither `router` nor `cli`: a caller names `semstrait_api::router` and `semstrait_api::cli` through its own `semstrait-api` dependency.
