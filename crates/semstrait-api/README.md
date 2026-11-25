# semstrait-api

The semstrait orchestrator: one `Semstrait` compiles a model, keeps its manifest in a store, and validates, explains and plans a Request through the stages the workspace builds. Beside it sit `load`, the Request wire form, catalog connections, and the REST and CLI transports, each transport behind its feature.

Behind `rest`, the crate names one `semstrait-ir` type, `SemanticRole`, which `/schema` groups each source's dimensions, measures and metrics by; otherwise the semantic plan passes from the planner to the adapter by inference. It links no engine.

---

## Semstrait

`SemstraitBuilder::default().build()` gives a `Semstrait` that holds one optional `CatalogProvider`, one `FileSystem` and one in-memory manifest store (ADR 0017; see [Input, load and the manifest store](#input-load-and-the-manifest-store)). It holds no per-caller state and memoizes nothing (ADR 0024); the store is the cache, at the persistence seam (ADR 0041).

| Builder method | Sets |
|---|---|
| `with_catalog(Arc<dyn CatalogProvider>)` | the catalog provider that compile reads |
| `with_catalogs(&CatalogsConfig, Option<&str>)` | the provider of one `catalogs.yaml` entry; async, and refused as `Err((refusal, warnings))` with `SemstraitError::Catalog` (see [Catalog connections](#catalog-connections)) |
| `with_filesystem(Arc<dyn FileSystem>)` | the filesystem that compile reads through; by default a `LocalFileSystem` at `/`, so a `file:///abs/x` path reads `/abs/x` |

| Verb | Steps | Returns |
|---|---|---|
| `compile_from_yaml(&str)` | `parse` → `build` (which runs model `validate`) → `compile` | the manifest, with every warning |
| `compile_from_model(SemanticModel)` | model `validate` → `compile` | the manifest, with every warning |
| `exists(&SemanticModel)`, `save_manifest(&SemanticManifest)`, `load_manifest(&SemanticModel)` | the store's `exists`, `save` and `load`; `load_manifest` then runs `SemanticManifest::validate()` (ADR 0042) | a bool; nothing; the manifest |
| `validate(&manifest, &request, &context)` | `Planner::plan`; the plan is dropped | the advisories |
| `explain(&manifest, &request, &context, dialect)` | plan, then `semstrait_adapter::explain` | the explain text, with the advisories |
| `plan(&manifest, &request, &context, dialect)` | plan, then `SubstraitProducer::produce` | `substrait::proto::Plan`, with the advisories |

- Compile and the store verbs are async; `validate`, `explain` and `plan` are sync (ADR 0003).
- The verbs keep ADR 0005's shapes. The compile verbs, `load_manifest`, `explain` and `plan` give `Ok((output, warnings))`; `validate` gives `Ok(advisories)`, `exists` `Ok(bool)` and `save_manifest` `Ok(())`. Each refuses with `Err((refusal, warnings))`, with the warnings gathered before the refusal. Compile and plan fail fast.
- The compile's provenance is its own instant, as both `created_at` and `updated_at` (ADR 0055).
- `catalog()` gives the catalog provider the `Semstrait` holds.

---

## Input, load and the manifest store

`load(&semstrait, input)` gives a manifest that `SemanticManifest::validate()` accepted. `Input` is model YAML, a model path, a manifest or a manifest path; a path is local or, with the `aws` feature, an `s3://` URI:

- model YAML or a model path: `exists` → `load_manifest` on a hit; `compile_from_yaml` → `save_manifest` on a miss
- a manifest: `SemanticManifest::validate()`
- a manifest path: decoded through `Persistable`, then `SemanticManifest::validate()`

Each `Semstrait` holds one in-memory `FileSystemRepository<SemanticManifest>`. `exists`, `save_manifest` and `load_manifest` key it by the compilation identity that `mint_model_id` gives for the model fingerprint. The compile verbs still compile every time (ADR 0041): the gate is `load`'s. The key is the model fingerprint, so a catalog change inside one `Semstrait` is not seen.

A path input that does not load is refused at its path: `SemstraitError::FileSystem`, `SemstraitError::Io` for text that is not UTF-8, or `SemstraitError::Input` for a manifest file that does not decode and for an `s3://` path without the `aws` feature.

---

## The Request wire form

`RequestPart::read_yaml` and `RequestPart::read_json` read one document into a `Request` and its `RequestContext`:

```yaml
from: orders
select:
  - region
  - {dimension: order_date, granularity: month}
  - {measure: revenue}
  - {metric: margin}
where: {eq: [channel, {lit: web}]}
filters: [paid_only]
order_by: [{name: revenue, direction: desc}]
limit: 100
offset: 0
context: {timezone: Europe/Stockholm, week_start: sunday}
```

- Every key but `select` is optional.
- A bare name in `select` is `Selection::Field`. `{dimension, granularity}` is `Selection::Dimension` at a requested granularity (`DimensionModifier::Granularity`); `{measure}` and `{metric}` select a Measure and a Metric.
- `where` is the request predicate, a semantic expression that reads through `Unresolved::<SemanticExpr>`, as a model's `filter:` does.
- `context` gives the `RequestContext`: `timezone` defaults to UTC, and `week_start` (`monday` or `sunday`) to Monday (ADR 0063; ADR 0072).
- An unknown key is refused, in the document, a typed selection, an `order_by` entry and `context`.

A refused document is a `WireError`: `Document` for a document that does not read, a refused `where` among them, whose message gives the line and column of the refusal, and `Timezone` for a timezone that is not an IANA name. Each wire type (`RequestPart`, `Selected`, `Key`, `Order`, `ContextPart`, `WeekStart`) also writes itself back through serde.

---

## Dialect

`Semstrait::plan` and `Semstrait::explain` take `impl Into<semstrait_common::DialectRelease>`, so a caller may still pass a bare `Dialect` (it reads at its reference release) or a `DialectRelease` at a chosen release, as `datafusion=53`. A dialect carries no adapter identity (ADR 0091, rule 6), so the API maps each dialect to the rules the Substrait producer is built with:

- `Substrait` — every `SubstraitRules` default
- `DataFusion` — `DataFusionRules::new(release)` from `semstrait-adapter-datafusion`, at the given release or the reference release when none is given, with the `datafusion` feature; without it, `SemstraitError::NoEngineAdapter(NoEngineAdapter::DataFusionFeatureOff)`
- `Iceberg` — `SemstraitError::NoEngineAdapter(NoEngineAdapter::NoRules(Dialect::Iceberg))`: a dialect for typing, with no rules (ADR 0086, rule 6)

The `datafusion` feature brings in rules only: the library of `semstrait-adapter-datafusion` links no engine.

---

## Catalog connections

`SemstraitBuilder::with_catalogs(&catalogs, alias)` connects one entry of a `catalogs.yaml` that `semstrait_model::parse_catalogs` read: the only entry, or the entry whose catalog alias is `alias`. One compile reads one catalog provider (ADR 0017). An entry of `type: polaris` or `type: iceberg_rest` becomes an `IcebergRestCatalogProvider`, with the `iceberg` feature:

| `auth: {method: …}` | Connects with |
|---|---|
| `aws_secrets` | the client id and secret from Secrets Manager, then OAuth2 (`secrets::build_polaris_catalog`, with the `aws` feature) |
| `oauth2` | `with_oauth2` |
| `bearer` | `with_bearer_token` |

A refused connection is `SemstraitError::Catalog`, whose `CatalogConnectionError` is `AliasNotGiven` (no alias, and not exactly one entry), `AliasNotFound`, `UnknownType` (a `type` other than the two), `UnknownAuth`, `IcebergFeatureOff`, `AwsFeatureOff` or `SecretFetch`.

---

## Transports

### CLI (`feature = "cli"`)

Command-line interface via `clap`. Binary: `semstrait`; the `cli` feature turns on `rest` for `serve`.

```
semstrait compile  --input model.yaml [--catalogs catalogs.yaml] [--catalog <alias>] [--output manifest.json]
semstrait validate --model model.yaml <request>
semstrait explain  --model model.yaml <request> [--json]
semstrait plan     --model model.yaml <request> --output plan.pb
semstrait serve    --model model.yaml [--host 0.0.0.0] [--port 8080]
```

| Command | Gives |
|---|---|
| `compile` | the manifest JSON on stdout, or in the `--output` file |
| `validate` | `valid` on stdout |
| `explain` | the explain text, or with `--json` the `/explain` body `{"text", "advisories"}` |
| `plan` | the Substrait plan's protobuf bytes in the `--output` file |
| `serve` | the REST router below, over the model it loads once at start |

- `validate`, `explain`, `plan` and `serve` load `--model` through `load`, and each takes `[--catalogs catalogs.yaml] [--catalog <alias>] [--dialect name=release]`, parsed as a `semstrait_common::DialectRelease` (`substrait`, `datafusion`, `datafusion=53`, `iceberg`); the dialect defaults to `substrait`. A bare name is the dialect's reference release (DataFusion 55); DataFusion's oldest is 53. A refused value gives clap's usual error, naming the value.
- `<request>` is `--request <file | ->`, a wire form document in YAML or JSON read from the file or stdin, or the quick flags `[--from <data kind>] --select <name>… [--filters <name>…]`; a typed selection comes only through `--request`.
- `--catalogs` reads `catalogs.yaml` from a local path or an `s3://` URI, and `--catalog` names the entry when it has several.
- `serve` refuses at start, with exit code 1, a `--dialect` that this build has no engine adapter for.
- Every warning, advisory and error goes to stderr. The exit code is 0 on success, 1 on a refusal at any stage, and `clap`'s own code on an argument error.

### REST (`feature = "rest"`)

`router(semstrait, manifest, dialect)` gives an axum `Router` over an `Arc<Semstrait>` and an `Arc<SemanticManifest>` compiled at start (ADR 0024). `dialect` is `impl Into<DialectRelease>`, set once as the server's when the router is built. No path prefix — routes are mounted at root:

| Route | Body | Response |
|---|---|---|
| `GET /health` | — | `{"status": "ok"}` |
| `GET /schema` | — | each public semantic source: `[{"name", "dimensions", "measures", "metrics"}]` |
| `POST /compile` | model YAML | the manifest JSON |
| `POST /validate` | the wire form, as JSON | `{"valid", "errors", "advisories"}`, 400 for a refused Request |
| `POST /explain` | the wire form, as JSON | `{"text", "advisories"}` |
| `POST /plan` | the wire form, as JSON | the Substrait plan, as `application/x-protobuf` bytes |

Each error and advisory is `{"message", "severity"}`. A refused body or Request on `/compile`, `/explain` or `/plan` answers 400 with `{"errors", "advisories"}`, and so does a body that is not UTF-8 (`SemstraitError::Io`). A producer failure (`SemstraitError::Adapt`) and a dialect of the server with no engine adapter (`SemstraitError::NoEngineAdapter`) answer 500.

`/compile` expands no environment variable: before any parse, it refuses a body that contains `${` anywhere, also in a quoted string or a comment, with 400 and `InputErrorKind::EnvVarInPostedModel`. `compile_from_yaml` itself replaces each `${NAME}` with the value of the environment variable `NAME` of its process, so a library caller that passes it untrusted YAML should refuse `${` itself.

---

## Errors

Every diagnostic is a `Diagnostic<SemstraitError>`. `SemstraitError` (`#[non_exhaustive]`, `Diagnose`) wraps each stage's own kind (ADR 0004):

| Variant | Wraps |
|---|---|
| `Parse`, `Build`, `Validate` | `semstrait-model`'s `ParseErrorKind`, `ModelBuildErrorKind` and `ValidateErrorKind` |
| `Compile` | `semstrait-manifest`'s `CompileError` |
| `Plan`, `Advisory` | `semstrait-planner`'s `PlanError` and `Advisory` |
| `Adapt` | `semstrait-adapter`'s `InvalidRewrite` |
| `Repository`, `FileSystem`, `Io` | `semstrait-common`'s `RepositoryErrorKind`, `FileSystemErrorKind` and `IoErrorKind` |
| `Input` | `InputErrorKind`: `UndecodableManifest`, `S3WithoutAws`, `EnvVarInPostedModel` |
| `Catalog` | `CatalogConnectionError` |
| `NoEngineAdapter` | `NoEngineAdapter`: `NoRules(Dialect)` for a dialect with no rules, `DataFusionFeatureOff` for `Dialect::DataFusion` without the `datafusion` feature |
| `Wire` | `WireError` |

---

## Feature Flags

| Feature | Adds |
|---------|------|
| `cli` | the `semstrait` binary via clap (turns on `rest`) |
| `rest` | `router`, the REST routes via axum |
| `datafusion` | `DataFusionRules` for `Dialect::DataFusion` |
| `iceberg` | Iceberg REST catalog connections (`semstrait-catalog/iceberg`) |
| `aws` | `s3://` model and manifest paths through `S3FileSystem`, and `aws_secrets` catalog auth (turns on `iceberg`, `semstrait-catalog/aws` and `semstrait-common/io-aws`) |

No feature is on by default.

---

## Dependencies

- `semstrait-common` — `Dialect`, `DialectRelease`, `Release`, `Diagnostic`, `FileSystem`, the manifest store's `FileSystemRepository`
- `semstrait-model` — `parse`, `validate`, `SemanticModel`, `parse_catalogs`, `CatalogsConfig`, the expression adapter `Unresolved` and its `SemanticExpr`
- `semstrait-manifest` — `compile`, `SemanticManifest`, `mint_model_id`
- `semstrait-catalog` — `CatalogProvider`, `IcebergRestCatalogProvider`
- `semstrait-planner` — `Planner`, `Request`, `RequestContext`
- `semstrait-adapter` — `SubstraitProducer`, `SubstraitRules`, `explain`
- `semstrait-adapter-datafusion` (optional, `datafusion`) — `DataFusionRules`
- `semstrait-ir` (optional, `rest`) — `SemanticRole`
- `prost` — `Message`, encoding the Substrait `Plan` to protobuf bytes for `plan` and `/plan`
- `serde` — `Serialize` and `Deserialize`, which every wire type derives
- `serde_json` — `RequestPart::read_json`, and the REST responses' and errors' JSON
- `serde_yaml` — `RequestPart::read_yaml`
- `thiserror` — `Error`, for every error kind this crate defines
- `tokio` — the async runtime `serve` binds its `TcpListener` on, via `#[tokio::main]`
- `axum` (optional, `rest`) — `Router`, the REST transport's routing and extraction
