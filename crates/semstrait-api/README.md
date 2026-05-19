# semstrait-api

API layer providing CLI, REST, and gRPC entry points for semstrait.

All transports share the `SemstraitEngine` orchestrator and `RequestParser` for consistent request handling. Each transport is feature-gated.

---

## SemstraitEngine

The central orchestrator that coordinates manifest compilation, query planning, and artifact generation:

```rust
pub struct SemstraitEngine { .. }

impl SemstraitEngine {
    // Construction
    pub fn new() -> Self;                                         // no manifest
    pub fn with_manifest(manifest: CompiledManifest) -> Self;     // pre-compiled
    pub fn with_adapter(manifest: CompiledManifest, adapter: Arc<dyn EngineAdapter>) -> Self;
    pub async fn with_model(yaml: &str) -> Result<Self, EngineError>;

    // Operations (all synchronous except schema drift)
    pub fn validate(&self, raw: &RawQueryRequest) -> ValidationResult;
    pub async fn explain(&self, raw: &RawQueryRequest) -> Result<ExplainResult, EngineError>;
    pub async fn plan(&self, raw: &RawQueryRequest) -> Result<PlanArtifact, EngineError>;

    // Observability
    pub async fn check_schema_drift(&self, catalog: &dyn CatalogProvider, namespace: &str)
        -> Vec<PlannerWarning>;
}
```

### explain() flow

1. Parse `RawQueryRequest` into `ResolvedQueryRequest`
2. Plan via `SemanticPlanner::plan()`
3. If an adapter is configured, use `adapter.debug_sql()` for the SQL representation
4. Otherwise, fall back to ANSI SQL emission

### plan() flow

1. Parse and plan (same as explain)
2. If an adapter is configured, `adapter.adapt()` produces a `PlanArtifact` (SQL or Substrait depending on engine)
3. Otherwise, falls back to ANSI SQL as `PlanArtifact::Sql`

---

## Unified Query API

All transports accept the same `RawQueryRequest`:

```rust
pub struct RawQueryRequest {
    pub model: Option<String>,      // semantic model source (file path or inline YAML/JSON)
    pub from: Option<String>,       // entity to query (None = resolve from select fields)
    pub select: Vec<String>,        // semantic names — auto-classified into dims/measures/metrics
    pub filters: Vec<String>,       // named-filter activations (DataKindFilter names declared on the kind)
    pub raw_filters: Vec<RawFilter>, // inline operator/value triples — lowered into ResolvedQueryRequest.filters
    pub grain: Option<String>,      // temporal grain override
    pub limit: Option<u64>,
    pub order_by: Vec<RawOrderBy>,
    pub session: HashMap<String, String>,
    pub engine: Option<String>,     // engine for plan generation (e.g., "datafusion", "ansi")
}
```

### Filter surfaces

The two filter fields target distinct authoring shapes:

- `filters: Vec<String>` — activates pre-declared `DataKindFilter`s by name (see `docs/design/foundations/18_entities.md` §7.1).
- `raw_filters: Vec<RawFilter>` — inline `{ field, operator, value }` triples (per `docs/design/apis/34_semstrait_planner.md` §3.5).

Both lower into the single `ResolvedQueryRequest.filters` list and share the same predicate-construction pipeline downstream (see semstrait-planner's _Unified filter pipeline_ section). Cross-reference invariant: `raw_filters[i].field` MUST NOT name a `DataKindFilter` — that's what `filters` is for. Violations are rejected at parse with `ParseError::RawFilterNamesNamedFilter`.

```rust
pub struct RawFilter {
    pub field: String,                 // dimension / measure / metric name
    pub operator: String,              // "=", "!=", "<", "<=", ">", ">=", "in", "not_in", "between", "is_null", "is_not_null"
    pub value: serde_json::Value,      // scalar or array, shape depends on operator arity
}
```

---

## Transports

### CLI (`feature = "cli"`)

Command-line interface via `clap`. Binary: `semstrait`.

```
semstrait compile  --input model.yaml [--output manifest.json] [--catalogs catalogs.yaml]
semstrait explain  --model model.yaml [--from orders] --select region revenue [--engine datafusion]
semstrait validate --model model.yaml [--from orders] --select region revenue
semstrait serve    --model model.yaml [--port 8080] [--engine datafusion]
```

Explain supports `--output plan|sql` (default: both) and `--json` for structured output.

### REST (`feature = "rest"`)

HTTP API via `axum`. No path prefix — routes are mounted at root:

```
POST /explain     { "from": "orders", "select": ["region", "revenue"] }
POST /plan        { "from": "orders", "select": ["region", "revenue"] }
POST /validate    { "from": "orders", "select": ["region", "revenue"] }
POST /compile     <raw YAML body>
GET  /schema      → manifest introspection (entities, dims, measures, metrics)
GET  /health      → { "status": "ok" }
```

### gRPC (`feature = "grpc"`)

gRPC service via `tonic`. Proto definition in `proto/service.proto`. Started via `semstrait serve --grpc-port <port>`.

---

## Engine Resolution

`resolve_adapter(engine: Option<&str>)` maps engine names to adapters:

- `None` / `"ansi"` / `"canonical"` — no adapter, ANSI SQL output
- `"datafusion"` — DataFusion adapter (requires `datafusion` feature)

The engine field is validated per-request: mismatches between requested engine and configured adapter produce `EngineError::NotConfigured`.

---

## Error Handling

`EngineError` wraps errors from each pipeline stage:

- `Parse` — from `ParseError` (request validation)
- `Compile` — from `semstrait-manifest`
- `Plan` — from `semstrait-planner`
- `Emit` — from `semstrait-adapter::sql::EmitError`
- `Adapt` — from `semstrait-adapter::AdaptError`
- `NotConfigured` — missing manifest or engine mismatch
- `Internal` — unexpected failures

---

## Feature Flags

| Feature | Adds |
|---------|------|
| `cli` (default) | CLI transport via clap |
| `rest` (default) | REST transport via axum |
| `grpc` | gRPC transport via tonic |
| `datafusion` | DataFusion adapter |
| `duckdb` | DuckDB adapter |
| `spark` | Spark adapter (structural) |
| `iceberg` | Iceberg REST catalog |
| `unity` | Unity catalog |
| `aws` | AWS Secrets Manager auth (implies `iceberg`) |

---

## Dependencies

- `semstrait-core` — shared primitives
- `semstrait-ir` — `LogicalPlan`, `PlanArtifact`, `PlannerWarning`
- `semstrait-planner` — `SemanticPlanner`, `ResolvedQueryRequest`
- `semstrait-manifest` — `ManifestCompiler`, `CompiledManifest`
- `semstrait-adapter` — `EngineAdapter`, `AdaptError`, `SqlEmitter`
- `semstrait-catalog` — `CatalogProvider`, `CatalogRegistry`
- `semstrait-model` — `parse_catalogs`, `CatalogEntry`
