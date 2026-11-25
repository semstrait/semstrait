---
id: 34-planner-crate-charter
status: ratified
version: 2.1
impl: in-progress
---

# Semstrait-Planner Crate Charter

This leaf owns the semstrait-planner crate charter, layering, scope boundary, invariants I5-I12, runtime contract (manifest→graph→planning→plan), crate boundaries, and dependency posture.

> **Status:** ratified (Round 1). `34` nails down the public surface of
> `semstrait-planner` — the crate that owns the `plan` stage (`10 §3.4`)
> and the `optimize` stage (`10 §3.5`) — against `20`'s Strategy-per-
> variant taxonomy, `21`–`24`'s per-variant strategy bodies, `33`'s
> SemanticManifest consumer contract, `35`'s `SemanticPlan` shape, and `30`'s
> stability / diagnostics policy. All types the surface touches are
> ratified upstream; `34` adds the crate-level wiring — the `plan` /
> `optimize` entry points, the `Request` / `SessionContext` /
> `ResolvedQueryRequest` value objects, the `Strategy` and
> `OptimizerPass` trait shapes, and the `PlanErrorKind` / `OptimizeErrorKind`
> typed-kind enums (per `30 §5`) that flow across the stage boundary.
> Round-1 open items parked in `questions/open/34_questions.md`.
>
> **Scope note (2026-05-27).** Planner runtime graph lifecycle is intentionally marked TODO/provisional in this document. Canonical graph types live in ir (`35`), manifest seeds live in `33`, and finalized planner runtime contracts will land in a dedicated planner pass. Runtime DAG backend target for that pass is `daggy`; legacy `petgraph` usage in manifest internals is transitional and non-authoritative for planner runtime contracts.

## 1. Purpose, scope, layering

### 1.1 What `semstrait-planner` OWNS

- The `plan` free function (§6) that turns a `&SemanticManifest` + a `Request` into a `SemanticPlan`.
- The `optimize` free function (§11) that takes a `SemanticPlan` and returns an equivalent, canonicalized `SemanticPlan`.
- The `Request` (§3), `SessionContext` (§4), and `ResolvedQueryRequest` (§5) value types.
- The `Strategy` trait (§8) — the dispatch surface ratified structurally in `20 §5.2` and concretized here.
- The four built-in strategies: `SimpleStrategy`, `GrainsetStrategy`, `UnionsetStrategy`, `JoinsetStrategy` (§9) — crate-public wrappers whose algorithms are ratified in `21`–`24`.
- Field-first resolution (§10) — the planner-side realization of the algorithm ratified in `16 §11`.
- TODO/provisional boundary contract for runtime graph lifecycle (builder/store/drift/lowering details deferred).
- The `OptimizerPass` trait (§12) — the pluggable v1 optimizer interface.
- The `PlanErrorKind` and `OptimizeErrorKind` typed-kind enums (§13), their `Diagnose` impls per `30 §5.4`, and the variant rosters wrapping per-DataKind `Simple` / `Grainset` / `Unionset` / `Joinset` errors.
- The `StrategyRegistry` and `StrategyContext` (§8.3 / §8.4) — internal wiring types exposed to adapter-level extensions and test doubles only.

### 1.2 What `semstrait-planner` does NOT own

- **Expression and graph type vocabulary.** `PhysicalExpr`, `SemanticGraph`, `SemanticNode`, `SemanticEdge`, and `SegmentKey` live in `semstrait-ir` (`35`); `Expr`, `DataType`, and `Grain` live in `semstrait-model` (`32`); diagnostics primitives live in `semstrait-common` (`31`). `34` consumes them.
- **Plan-tree shape.** `SemanticPlan`, `PlanNode`, `NodeMeta`, `SourceRef`, `Name` all live in `semstrait-ir` (`35`). `34` emits and consumes them.
- **SemanticManifest shape.** `SemanticManifest` seed/index contracts live in `semstrait-manifest` (`33`). `34` reads them, builds runtime segments, and never mutates persisted artifacts.
- **YAML parsing / structural validation.** Lives in `semstrait-model` (`32`). The `Request` the planner accepts is already a Rust value — the API layer (`semstrait-api`) is responsible for converting user-facing JSON / gRPC / protobuf into `Request`.
- **Catalog / filesystem ownership.** `CatalogProvider` / `FileSystem` trait ownership remains in `37`; planner may consume explicit drift probes/results but does not own provider contracts.
- **SQL / engine emission.** `semstrait-adapter` (`36`) consumes the `SemanticPlan` and produces engine-specific artifacts; no planner API touches a dialect.

### 1.3 Design posture — sync-only dispatch crate

The planner is the workspace's widest dispatch site — the `plan` entry point fans out across every `ResolvedDataKind` variant — but it is **not** the crate with the most runtime weight. The hot path is a single tree walk over pre-resolved SemanticManifest indices (§7). Three properties shape the design:

- **Synchronous end-to-end.** Per I6, there is no `async fn` anywhere on the public surface. `plan` and `optimize` are ordinary fallible functions; every strategy method is sync. The `async` wrapper on `compile` (`10 §3.3`) is the last async boundary in the pipeline; everything the planner reads has already been resolved.
- **Manifest-seeded planning.** Per I5 / I8, the planner does no semantic name resolution. It performs O(1)/O(log n) lookups on manifest indices and lowers to `PlanNode`s. Runtime graph lifecycle details are TODO/provisional in this revision.
- **Strategy dispatch is the one variant-match site.** Per `20 §5.3`, the only place the planner branches on a `ResolvedDataKind` variant tag is the dispatch function (§8.5). Every other planner site consumes `&dyn Strategy`. Adding a new `Complex` variant per I10 is a MINOR change that forces a new match arm in one place, not a scatter of edits.

### 1.4 Invariants upheld by the crate


| Invariant                                  | `semstrait-planner` guarantee                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I5** — name resolution is compile-time   | The planner performs **lookup only** (`19 §3.2.3`, `33 §4`). No `resolve`_* method walks names to Semantics or columns; every such walk was performed at `compile`. A CI lint forbids `EntityRef` leaves in any `PhysicalExpr` a plan node carries.                                                                                                            |
| **I6** — plan hot path is synchronous      | **No `pub async fn` exists on `semstrait-planner`.** The `Strategy` trait's `plan` method is sync; so is every `OptimizerPass::apply`. A CI audit (`cargo clippy -- -D clippy::async_fn_in_trait`) enforces.                                                                                                                                                    |
| **I8** — planner-complete SemanticManifest | Every lookup needed for candidate selection and segment build is present in manifest seeds/indices (`33 §3`). Missing ids/edges/seeds trigger `PlanErrorKind::SemanticManifestIndexInconsistent` before lowering.           |
| **I10** — non-exhaustive public sum types  | `Request`, `SessionContext`, `ResolvedQueryRequest`, `PlanErrorKind`, `OptimizeErrorKind`, `StrategyId` are all `#[non_exhaustive]`. An integration test over `cargo public-api` enforces.                                                                                                                                                                      |
| **I11** — no I/O in hot path               | No `std::fs`, no `std::net`, no `tokio`, no `reqwest` in the crate's dependency graph. The `Cargo.toml` audit (§16.2) is CI-enforced.                                                                                                                                                                                                                           |
| **I12** — first-class diagnostics          | `PlanErrorKind` and `OptimizeErrorKind` implement `Diagnose` per `30 §5.4`; identification is by variant identity per `30 §5.4` (no string-code surface). Warnings accumulate through both success and failure arms per `30 §7.2` / `§7.3`. The `tracing` channel (`30 §6`) carries library-internal observability events orthogonally to returned diagnostics. |


### 1.4A Runtime contract: `manifest -> graph -> planning -> plan`

Status: **TODO / provisional**.

This section is intentionally a boundary marker, not a finalized planner implementation contract.

Planned runtime path:

1. read candidates from manifest indices (`33`);
2. materialize/lookup runtime graph fragments using canonical IR graph shapes (`35`);
3. resolve runtime physical expressions for touched fragment scope;
4. apply source drift policy (`Strict` / `Warn` / `TrustManifest`);
5. lower fragment into canonical `SemanticPlan`.

Current contract at this stage:

- `33` is authoritative for persisted manifest seed/index shape.
- `35` is authoritative for canonical graph and expression types.
- Graph-fragment admission validates DAG-ness and typed expression-reference resolvability (`GraphExprRef`) before planning/lowering.
- planner runtime store/build/eviction/drift interfaces are pending dedicated planner design pass.
- planner runtime DAG backend target is `daggy`; backend-specific types do not leak through `33`/`35` public contracts.

### 1.5 Constraint validation precedes Strategy dispatch

Per `11 §8.6`, realized-carrier Constraint validation runs as the **planner's first action — step 0, pre-resolution — before any other sub-step of the `plan` pipeline**. The rationale is twofold:

1. Constraints express author-declared admissibility rules on Request shape (`11 §8.4`). A Request that violates a Measure's `dimensions: { one_of: [...] }` rule cannot be planned without lying about author intent; it is cheaper and clearer to reject before any SemanticManifest index work begins.
2. The v1 `ConstraintValidator::check()` (`11 §8.6`) reads only the Request and the SemanticManifest's name indices — work the planner must do anyway in step 1. Reordering would duplicate lookups.

§7.1 ratifies step 0 as the planner's entry action. The seven-step pipeline is fixed: violating the order is a crate-internal bug, not a surface the caller can influence.

## 2. Public crate surface

Every `pub` symbol below carries a doc comment, is listed in this document, and has documented invariants. Crate-internal helpers (`DataKindPlannerRegistry`, `PlanFragment`, `PrunedView`, etc. from the legacy code) are `pub(crate)` and are not part of `34`'s surface. The target module layout is:

```
semstrait-planner
├── request              // Request, SessionContext, ResolvedQueryRequest,
│                        //   Filter, TemporalRequest (forward-ref to 17)
├── plan                 // pub fn plan, PlanErrorKind, plan-pipeline internals
├── optimize             // pub fn optimize, OptimizerPass, OptimizeErrorKind,
│                        //   canonical v1 passes
├── graph_runtime        // TODO/provisional: runtime graph lifecycle interfaces
├── strategy             // Strategy trait, StrategyId, StrategyContext,
│                        //   StrategyRegistry, dispatch_strategy
├── strategies           // SimpleStrategy, GrainsetStrategy, UnionsetStrategy,
│                        //   JoinsetStrategy
└── resolution           // field-first resolution entry point; Request-lookup
                         //   helpers; relationship-traversal wrappers
```

Crate-root re-exports expose the stable convenience surface. Non-root re-exports are forbidden — consumers either import `semstrait_planner::plan` or `semstrait_planner::plan::plan`, never both.

**Surface roster (one line per item; full shapes in later sections):**


| Module       | Item                                                                                                                                              | Kind                               | Section |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------- |
| (crate root) | `pub fn plan(...) -> Result<(SemanticPlan, Diagnostics<PlanErrorKind>), (Diagnostic<PlanErrorKind>, Diagnostics<PlanErrorKind>)>`                 | free fn                            | §6      |
| (crate root) | `pub fn optimize(...) -> Result<(SemanticPlan, Diagnostics<OptimizeErrorKind>), (Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>)>` | free fn                            | §11     |
| `graph_runtime` | TODO/provisional runtime graph lifecycle interfaces                                                                                              | deferred surface                   | §1.4A   |
| `request`    | `pub struct Request`                                                                                                                              | value type                         | §3      |
| `request`    | `pub struct SessionContext`                                                                                                                       | value type                         | §4      |
| `request`    | `pub struct ResolvedQueryRequest`                                                                                                                 | value type                         | §5      |
| `request`    | `pub struct Filter`                                                                                                                               | value type                         | §3.5    |
| `request`    | `pub enum FilterOperator`                                                                                                                         | sum type                           | §3.5    |
| `request`    | `pub enum FilterValue`                                                                                                                            | sum type                           | §3.5    |
| `request`    | `pub enum SortDir`                                                                                                                                | sum type                           | §3.6    |
| `request`    | `pub struct DataKindRef`                                                                                                                          | newtype                            | §3.8    |
| `request`    | `pub struct TemporalRequest`                                                                                                                      | value type (reserved)              | §3.9    |
| `plan`       | `pub enum PlanErrorKind`                                                                                                                          | typed-kind enum (impls `Diagnose`) | §13     |
| `optimize`   | `pub enum OptimizeErrorKind`                                                                                                                      | typed-kind enum (impls `Diagnose`) | §13.3   |
| `optimize`   | `pub trait OptimizerPass`                                                                                                                         | pluggable pass                     | §12     |
| `optimize`   | `pub struct Optimizer`                                                                                                                            | pass chain                         | §12.4   |
| `optimize`   | `pub struct OptimizerBuilder`                                                                                                                     | builder                            | §12.5   |
| `strategy`   | `pub trait Strategy`                                                                                                                              | dispatch surface                   | §8      |
| `strategy`   | `pub struct StrategyId`                                                                                                                           | newtype                            | §8.2    |
| `strategy`   | `pub struct StrategyContext<'a>`                                                                                                                  | per-invocation context             | §8.4    |
| `strategy`   | `pub struct StrategyRegistry`                                                                                                                     | dispatch table                     | §8.3    |
| `strategy`   | `pub fn dispatch_strategy(...) -> Result<&dyn Strategy, PlanErrorKind>`                                                                          | free fn                            | §8.5    |
| `strategies` | `pub struct SimpleStrategy`                                                                                                                       | v1 strategy                        | §9.1    |
| `strategies` | `pub struct GrainsetStrategy`                                                                                                                     | v1 strategy                        | §9.2    |
| `strategies` | `pub struct UnionsetStrategy`                                                                                                                     | v1 strategy                        | §9.3    |
| `strategies` | `pub struct JoinsetStrategy`                                                                                                                      | v1 strategy                        | §9.4    |


> **Legacy detail note.** Sections `§3+` retain extensive pre-split planner details. The split contract ratified in `§1.4A` (runtime graph lifecycle) and `§2` (public surface roster) is authoritative when names/flows differ (for example legacy `expr_table` references vs `ManifestExpressions` + runtime realization).

## 16. Crate boundaries

### 16.1 NO I/O

No `std::fs`, no `std::net`, no `tokio`, no `reqwest`, no `aws-sdk-`*, no `object_store` in the crate's dependency graph. A `plan` or `optimize` invocation performs zero syscalls on the hot path; every datum consulted is already in the `SemanticManifest` (I8 / I11). `tracing::debug!` is permissible as instrumentation — `tracing` is a no-op when no subscriber is installed.

### 16.2 NO SQL emission

The planner emits `PlanNode`s carrying `PhysicalExpr` trees (`35 §5`). No SQL string is produced and no dialect-aware operator is chosen; SQL emission is strictly `semstrait-adapter`'s concern (`36`) per I1 / I3. Filter placement names `PlanNode::Filter` (not `WHERE` / `HAVING`); aggregation emission names `PlanNode::Agg` with `Aggregation` variants (not `SUM(...)` strings).

### 16.3 NO YAML parsing

YAML parsing lives in `semstrait-model` (`32`). The `Request` the planner accepts is already a Rust value — constructed by `semstrait-api` or any direct Rust caller. No `serde_yaml` / `yaml-rust` in the dependency graph.

### 16.4 NO catalog access

`CatalogProvider` / `FileSystem` (`37`) are strictly a `compile`-stage concern. The planner takes a `&SemanticManifest` that already encodes every piece of catalog data it needs; no `&dyn CatalogProvider` appears on any planner surface. The legacy `SemanticPlanner.catalog: Option<Arc<dyn CatalogProvider>>` field is a migration item (`[TD-PLANNER-NO-CATALOG]`) retained only for the handful of ad-hoc-join paths pending the field-first-resolution refactor (§9.5); the ratified surface drops it entirely.

### 16.5 Dependency posture

A canonical `Cargo.toml` target (matching `31 §12.1`'s discipline):

```toml
[dependencies]
semstrait-common     = { workspace = true }
semstrait-ir       = { workspace = true }
semstrait-manifest = { workspace = true }

thiserror  = "^"    # error enum derivations
tracing    = "^"    # instrumentation-only; no I/O

[dependencies.serde]
version = "^"
optional = true
features = ["derive"]

[features]
default = []
serde = ["dep:serde", "semstrait-common/serde", "semstrait-ir/serde", "semstrait-manifest/serde"]
```

**No runtime async dependencies.** No `tokio`, `async-trait`, `futures`, `reqwest`.

**No engine-identity dependencies.** No `datafusion`, `arrow`, `spark-`*, `duckdb`, `substrait` — these live in `semstrait-adapter`.

**Zero `semstrait-adapter` / `semstrait-catalog` / `semstrait-model` dependencies.** The planner sits strictly above the first four workspace crates and strictly below the adapter / API crates per I7's DAG.

### 16.6 CI enforcement

Per `31 §13`, concrete CI checks guard each boundary:

- `cargo deny` enforces §16.1 and §16.5's dependency bans.
- `cargo clippy -- -D clippy::async_fn_in_trait` enforces I6.
- `cargo public-api` snapshot test enforces the surface in §2.
- An integration test asserts that every exported `pub enum` / `pub struct` (minus the stable-newtype exception set) carries `#[non_exhaustive]`.
- A grep-based CI lint rejects `String`-typed SQL literals inside planner source (the `EXPRESSION_INCLUDES_SQL` regex).

## Testing Strategy

- cargo deny enforces I/O and dependency bans (§16.1, §16.5)
- clippy::async_fn_in_trait enforces I6 sync discipline
- cargo public-api snapshot enforces surface roster (§2)
- integration test enforces #[non_exhaustive] discipline on public sum types
- grep-based lint rejects SQL literals in planner source (EXPRESSION_INCLUDES_SQL regex)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
