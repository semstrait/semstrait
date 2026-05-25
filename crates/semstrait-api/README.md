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
    pub filters: Vec<String>,       // named filters from the manifest
    pub raw_filters: Vec<RawFilter>, // inline filter expressions (see "Inline raw filters" below)
    pub grain: Option<String>,      // temporal grain override
    pub limit: Option<u64>,
    pub order_by: Vec<RawOrderBy>,
    pub session: HashMap<String, String>,
    pub engine: Option<String>,     // engine for plan generation (e.g., "datafusion", "ansi")
}
```

### Inline raw filters

`raw_filters` carries anonymous, request-scope `{ field, operator, value }` predicates that are translated into canonical boolean `Expr`s by `RequestParser::to_resolved` and injected into the plan at the **scan layer** — the same engine that lifts named DataKind filters. See [`docs/design/foundations/11_names_and_scopes.md §6.4.2`](../../docs/design/foundations/11_names_and_scopes.md) and [`docs/design/foundations/19_expression_flow.md §7.1`](../../docs/design/foundations/19_expression_flow.md).

```rust
let raw = RawQueryRequest {
    from: Some("orders".to_string()),
    select: vec!["date".into(), "revenue".into()],
    raw_filters: vec![RawFilter {
        field: "region".into(),
        operator: "eq".into(),         // also: ne, lt, le, gt, ge, in, like (+ symbolic aliases)
        value: serde_json::json!("US"),
    }],
    ..Default::default()
};
```

Validation runs at request resolution time against the live `CompiledManifest`:

- Unknown field → `ParseError::RawFilterFieldNotFound`
- Operator outside the canonical set → `ParseError::RawFilterOperatorInvalid`
- Value that fails type-check against the field's `DataType` → `ParseError::RawFilterValueTypeMismatch`
- Missing `from` (ad-hoc mode) + non-empty `raw_filters` → `ParseError::RawFiltersRequireEntity`

Inline filters do not enter the manifest; they have no name and are not addressable by `Request.filters: [name]`. Each becomes a `CompiledFilter` with a synthetic `__inline_filter_<N>` name internally.

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
