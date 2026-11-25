---
id: 38-api-orchestrator
status: ratified
version: 2.1
impl: planned
---

# SemStrait Orchestrator

This leaf owns the `SemStrait` orchestrator handle: its purpose, scope, layering, public crate surface (roster, re-exports, module layout), struct shape (field roster with catalog_provider, file_system, repository, function_registry, optimizer_passes, warning_policy), method roster with signatures (compile_from_yaml, compile_from_model, plan, adapt, compile_and_plan_and_adapt multi-stage fused, save_manifest, load_manifest, validate_manifest I11b out-of-band), method contracts (parse→validate→compile→plan→optimize→adapt orchestration per §3.4), invariants (I6, I7, I10, I11, I12), and tracing observability (span-per-stage, stable fields, severity levels, CLI verbosity mapping, warning-policy-independent mirroring).

> Round-1 draft. Ratifies the public surface of the `semstrait-api` crate: the single orchestration entry point that stitches `parse` → `validate` → `compile` → `plan` → `adapt` into a caller-friendly flow. This crate owns zero domain logic; it owns one builder-configured orchestrator, one warning-propagation policy, and one unified error enum. Every stage-specific decision continues to live in its owning `3x` crate; `38` is the seam that makes those decisions addressable through one handle.

---

## 1. Purpose, scope, layering

### 1.1 Purpose

`semstrait-api` is the top-level orchestration crate for the semstrait toolchain. Its reason for existence is ergonomic: a direct caller of `semstrait-model`, `semstrait-manifest`, `semstrait-planner`, and `semstrait-adapter` must thread catalog providers, filesystems, function registries, optimizer-pass vectors, and warning sinks through five successive call sites, each with its own error type and its own async posture. `semstrait-api` collapses that into a single `SemStrait` handle with a `SemStraitBuilder` for setup.

`semstrait-facade` (`39`) sits one layer above `38`, exposing a still-narrower "one-shot-no-configuration" entry point for users who want defaults everywhere. This crate (`38`) is the configurable layer; `39` is the zero-config layer.

### 1.2 Scope — what this crate owns

- **Orchestration.** The `SemStrait` struct and its method roster (`§3`), composing per-stage crates in canonical pipeline order.
- **Configuration carrier.** The `SemStraitBuilder` (`§4`), collecting `CatalogProvider` / `FileSystem` / `Repository` handles, the function-registry handle, optimizer-pass pipeline, and `WarningPolicy`.
- **Warning-propagation policy.** `WarningPolicy` (`§5`) and the accumulate / escalate / strict rules at stage boundaries.
- **Unified error type.** `SemStraitErrorKind` (`§6`), a typed-kind enum that wraps each stage's `*ErrorKind` per `30 §5`. Identification by variant identity; no numeric subsystem prefix.
- **One-shot convenience.** `compile_and_plan_and_adapt` (`§7`).
- **Streaming / incremental support.** Compile-once / plan-many lifecycle (`§8`).
- **I11b plumbing.** `save_manifest` / `load_manifest` (thin-wraps `Repository::save` / `load`, `33 §11`) and `validate_manifest` (thin-wraps `CatalogProvider::check_schema_drift`, `37 §9`).

### 1.3 Scope — what this crate does NOT own

- **No new domain types.** Every exposed struct is a re-export from `31`–`37` or a small orchestration helper. No new `Resolved*`, `PlanNode`, `EngineArtifact` variant, or `DataType`.
- **No I/O beyond injected providers.** Every catalog / byte / persistence access flows through `&dyn CatalogProvider` / `&dyn FileSystem` / `&dyn Repository`. No network or disk code lives in this crate.
- **No SQL emission.** SQL is an adapter concern (`36 §8`); callers reach it via `EngineAdapter::emit` on the injected adapter, not through `SemStrait`.
- **No planner / adapter logic.** `plan` and `adapt` are thin wrappers over `34` and `36 §3`; they add warning-policy plumbing and error-enum wrapping only.
- **No function-registry construction.** The process-global `function_registry()` (`31 §5`) is the sole source; `SemStrait` carries a `&'static FunctionRegistry` handle.
- **No session state.** Every runtime input (`request`, `adapter`) is passed explicitly per call; configuration is frozen at build-time; the handle is `Send + Sync + Clone`.

### 1.4 Layering

```
caller → SemStrait (configured by SemStraitBuilder)
           ↓
    semstrait-model    semstrait-manifest    semstrait-planner
    semstrait-ir       semstrait-adapter     semstrait-catalog
```

Methods group by I-invariant: `compile_from_yaml` / `compile_from_model` are I11a (async); `plan` / `adapt` are I6 (sync); `compile_and_plan_and_adapt` is async because it straddles compile; `save_manifest` / `load_manifest` / `validate_manifest` are I11b (async out-of-band). Dependency edges flow strictly down and across per I7. `semstrait-api` re-exports the minimum set of types needed to drive the pipeline end-to-end (`§2.2`); no private internals, no crate-boundary-crossing hooks.

---

## 2. Public crate surface

### 2.1 Roster

| Name                         | Kind          | `#[non_exhaustive]`? | Stability (v1) |
|------------------------------|---------------|----------------------|----------------|
| `SemStrait`                  | struct        | no (opaque; new ctor only via builder) | Stable |
| `SemStraitBuilder`           | struct        | no (builder; `Default` + chain methods) | Stable |
| `WarningPolicy`              | enum          | yes                  | Stable         |
| `SemStraitErrorKind`         | enum          | yes                  | Stable         |
| `StageOrigin`                | enum          | yes                  | Stable         |
| `PipelineOutcome`            | struct        | yes                  | Provisional    |

Plus the re-exports (`§2.2`).

"Stability (v1)" follows `30 §13`. `semstrait-api` is **Stable in v1**; every struct and enum in its roster inherits that tier. `PipelineOutcome` is `Provisional` because its exact shape depends on ratification of Q-API-002 (one-shot return shape) during the next docs cycle.

### 2.2 Re-exports

The minimum set of types a caller needs to drive the pipeline end-to-end:

- From `31`: `Diagnostic` (i.e. `Diagnostic<K>`), `Diagnostics<K>`, `Diagnose`, `Severity`.
- From `32`: `SemanticModel`, `ParseErrorKind`, `ValidateError`, `ModelBuildErrorKind`.
- From `33`: `SemanticManifest`, `SemanticManifestId`, `SemanticManifestMetadata`, `CompileError`, `Repository`, `RepositoryErrorKind`, `SemanticManifestLoadErrorKind`, `SemanticManifestDumpErrorKind`.
- From `34`: `Request`, `SessionContext`, `PlanErrorKind`, `OptimizeErrorKind`.
- From `35`: `SemanticPlan`, `EngineArtifact`, `SqlArtifact`, `EnginePlan`, `IrErrorKind`.
- From `36`: `EngineAdapter`, `AdapterId`, `AdaptErrorKind`.
- From `37`: `CatalogProvider`, `FileSystem`, `DriftReport`, `CatalogProviderErrorKind`, `FileSystemErrorKind`.

Re-exports are additive (adding one is MINOR per `30 §2`); removing one is MAJOR. Consumer code SHOULD import through `semstrait_api::…` unless it has a reason to depend directly on an inner crate.

### 2.3 Module layout (informative)

```rust
pub mod builder;   // SemStrait, SemStraitBuilder
pub mod policy;    // WarningPolicy
pub mod error;     // SemStraitErrorKind, StageOrigin
pub mod outcome;   // PipelineOutcome

pub use builder::{SemStrait, SemStraitBuilder};
pub use policy::WarningPolicy;
pub use error::{SemStraitErrorKind, StageOrigin};
pub use outcome::PipelineOutcome;

pub use semstrait_common::{
    diagnostic::{Diagnostic, Diagnostics, Diagnose},
    Severity,
};
pub use semstrait_model::{
    SemanticModel, ParseErrorKind, ValidateError, ModelBuildErrorKind,
};
pub use semstrait_manifest::{
    SemanticManifest, SemanticManifestId, SemanticManifestMetadata,
    CompileError, Repository, RepositoryErrorKind,
};
pub use semstrait_planner::{Request, SessionContext, PlanErrorKind, OptimizeErrorKind};
pub use semstrait_ir::{SemanticPlan, EngineArtifact, SqlArtifact, EnginePlan, IrErrorKind};
pub use semstrait_adapter::{EngineAdapter, AdapterId, AdaptErrorKind};
pub use semstrait_catalog::{
    CatalogProvider, FileSystem, DriftReport,
    CatalogProviderErrorKind, FileSystemErrorKind,
};
```

Re-exports at the crate root per `30 §3`. Module names are informative; the exact layout may shift in refactors without MINOR impact so long as the crate-root re-exports hold.

---

## 3. `SemStrait` orchestrator

### 3.1 Purpose

`SemStrait` is a configured, thread-safe handle over the pipeline. It owns `Arc`-shared provider handles and process-global function-registry access; every method consumes `&self`. Two `SemStrait` values constructed with identical builder input are **observationally equivalent**: calling the same method sequence on either produces the same output modulo catalog-snapshot drift.

### 3.2 Struct shape

```rust
pub struct SemStrait {
    catalog_provider: Arc<dyn CatalogProvider>,
    file_system: Arc<dyn FileSystem>,
    repository: Option<Arc<dyn Repository>>,
    function_registry: &'static FunctionRegistry,
    optimizer_passes: Arc<[Box<dyn OptimizerPass>]>,
    warning_policy: WarningPolicy,
}
```

Fields are `pub(crate)`; construction goes through `SemStraitBuilder::build` (`§4.7`). There is no `Default` for `SemStrait` — every usable configuration requires at least one caller decision (the `CatalogProvider` injection), and surfacing a `Default` would force a hidden `NoopCatalogProvider` choice that a caller might not want.

`Arc<[Box<dyn OptimizerPass>]>` is chosen over `Vec<…>` so cloning the `SemStrait` (e.g. into worker tasks) does not re-allocate the pass vector.

### 3.3 Method roster and signatures

```rust
impl SemStrait {
    pub fn builder() -> SemStraitBuilder;

    // Method-level error kind selection rule (Option A in §6.1):
    //   - Single-stage methods PASSTHROUGH the underlying crate's typed kind.
    //   - Multi-stage methods (those that orchestrate two or more stage entries)
    //     RETURN `SemStraitErrorKind` — the unified kind whose variants wrap the
    //     upstream `*ErrorKind`s via `From` impls (§6.3).
    // Callers that prefer the underlying surface for a multi-stage method
    // destructure `SemStraitErrorKind::Compile(CompileError::…)` etc.

    pub async fn compile_from_yaml(&self, yaml: &str)               // multi-stage: parse + validate + compile
        -> Result<
            (SemanticManifest, Diagnostics<SemStraitErrorKind>),
            (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
        >;
    pub async fn compile_from_model(&self, model: SemanticModel)    // multi-stage: validate + compile
        -> Result<
            (SemanticManifest, Diagnostics<SemStraitErrorKind>),
            (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
        >;

    pub fn plan(&self, manifest: &SemanticManifest, request: Request)  // multi-stage: plan + optimize
        -> Result<
            (SemanticPlan, Diagnostics<SemStraitErrorKind>),
            (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
        >;
    pub fn adapt(&self, adapter: &dyn EngineAdapter,                  // single-stage: passthrough
                 plan: &SemanticPlan, manifest: &SemanticManifest)
        -> Result<
            (EngineArtifact, Diagnostics<AdaptErrorKind>),
            (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>),
        >;

    // The fused helper is the single site that unifies all per-stage kinds into
    // `SemStraitErrorKind` (§6). Warnings concatenate across stages on success
    // (§10.2). On failure, the originating stage's typed-kind is wrapped via
    // `From` impls (§6.3) into `SemStraitErrorKind`, with warnings observed up
    // to that stage preserved in the `Diagnostics<SemStraitErrorKind>` slot.
    pub async fn compile_and_plan_and_adapt(
        &self, yaml: &str, request: Request, adapter: &dyn EngineAdapter,
    ) -> Result<
        (EngineArtifact, Diagnostics<SemStraitErrorKind>),
        (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
    >;
```

The I11b out-of-band methods complete the signature block.

```rust
    // I11b — out-of-band gated entries. Repository / catalog-provider transports
    // bring in their own typed kinds that fold into `SemStraitErrorKind` via
    // `From` impls (§6.3); `validate_manifest` aggregates drift across many
    // catalog sources and returns the unified kind for the same reason.
    pub async fn save_manifest(&self, manifest: &SemanticManifest)
        -> Result<
            (SemanticManifestId, Diagnostics<SemStraitErrorKind>),
            (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
        >;
    pub async fn load_manifest(&self, id: SemanticManifestId)
        -> Result<
            (SemanticManifest, Diagnostics<SemStraitErrorKind>),
            (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
        >;
    pub async fn validate_manifest(&self, manifest: &SemanticManifest)
        -> Result<DriftReport, Diagnostic<SemStraitErrorKind>>;

    pub fn warning_policy(&self) -> WarningPolicy;
    pub fn function_registry(&self) -> &'static FunctionRegistry;
    pub fn adapter_id_for<'a>(&'a self, adapter: &'a dyn EngineAdapter) -> AdapterId;
}
```

`adapt` takes `&SemanticManifest` because adapters reach into resolved bindings (`36 §3.1`); `compile_and_plan_and_adapt` is async only so it can straddle the compile I/O boundary. `save_manifest` / `load_manifest` require `self.repository.is_some()` — otherwise they fail-fast with a `Diagnostic<SemStraitErrorKind>` whose kind is `NoRepositoryConfigured` (§6.2).

### 3.4 Method contracts

- **`compile_from_yaml`** — `parse` (`32 §9.1`) → `validate` (`32 §9.4`) → `compile` (`33 §9`). Errors short-circuit per `30 §7`. Each per-stage failure is wrapped into `SemStraitErrorKind` via the `From<ParseErrorKind> | From<ValidateError> | From<CompileError>` impls in §6.3, preserving the inner `*ErrorKind` variant verbatim. Stage-internal warnings are re-keyed via the same `From` impls and accumulated in source-stage order. The first stage to fail short-circuits the chain; warnings observed up to that point ride alongside the fatal in the `Err((fatal, warnings))` tuple.
- **`compile_from_model`** — Skips parse; still runs validate + compile. For callers that synthesize `SemanticModel`s (or apply custom post-parse transforms) and want the API crate to perform validation before compile. Same wrapping discipline as `compile_from_yaml`, minus the `Parse(...)` variant.
- **`plan`** — Runs `semstrait_planner::plan(&SemanticManifest, Request)` (canonical 2-arg signature per `34 §6.1`), then applies optimization via `34 §12.5`'s `OptimizerBuilder` (`OptimizerBuilder::new().with(self.optimizer_passes.clone()).build().apply(...)` when the builder slot is non-empty; otherwise `Optimizer::with_v1_passes()` per `34 §11.2`). The function-registry handle is consumed by the planner via the process-global `function_registry()` (`31 §5.2`) — not threaded as a per-call argument. `self.warning_policy` is applied to the returned warnings on success. `PlanErrorKind` and `OptimizeErrorKind` are wrapped into `SemStraitErrorKind` via `From` impls (§6.3); the optimizer pass that produced an `OptimizeErrorKind` is identifiable via the inner kind's payload.
- **`adapt`** — Single-stage delegate over `adapter.adapt(plan, manifest)` (`36 §3.1`). Returns `Diagnostic<AdaptErrorKind>` directly — same surface a caller would observe using `semstrait-adapter` as a free crate. The fused helper (§7) re-wraps via `From<AdaptErrorKind> for SemStraitErrorKind` (§6.3) when needed.
- **`compile_and_plan_and_adapt`** — Runs compile → plan → adapt. Every per-stage failure folds into `SemStraitErrorKind` via the §6.3 `From` impls. Warnings concatenate across stages in source order (`§10.2`). On error, the first failing stage's typed-kind is wrapped into a `Diagnostic<SemStraitErrorKind>` with warnings up to that point preserved in the `Diagnostics<SemStraitErrorKind>` slot of the `Err` tuple.
- **`save_manifest` / `load_manifest`** — Wrap `Repository::save` / `Repository::load` (`33 §11`). Require `self.repository.is_some()`; otherwise return `Err((Diagnostic::error(SemStraitErrorKind::NoRepositoryConfigured), Diagnostics::empty()))`. Underlying `RepositoryErrorKind`s fold into `SemStraitErrorKind::Repository(...)` via §6.3.
- **`validate_manifest`** — Iterates resolved physical sources (`33 §8`), calls `CatalogProvider::check_schema_drift` (`37 §9`) per source, aggregates under `Unchanged < Compatible < Breaking`. Per-source `DriftKind` details concatenate. Returns the simple `Result<DriftReport, Diagnostic<SemStraitErrorKind>>` shape since the call produces no advisory warnings; `CatalogProviderErrorKind`s fold into `SemStraitErrorKind::CatalogProvider(...)` via §6.3.

### 3.5 Invariants

| Ref | Invariant |
|-----|-----------|
| I6  | `plan`, `adapt`, and every accessor on `SemStrait` are synchronous. No `.await` point is reachable from these methods. |
| I7  | Depends on `31`–`37`; depended on by `39`. No cycles. |
| I10 | `WarningPolicy`, `SemStraitErrorKind`, `StageOrigin`, `PipelineOutcome` are `#[non_exhaustive]`. |
| I11 | `compile_*` is async (I11a); `save_manifest`, `load_manifest`, `validate_manifest` are async (I11b); no other method is async. |
| I12 | Errors are typed-kind enums per `30 §5` / `31 §3` (`ParseErrorKind`, `ValidateError`, `CompileError`, `PlanErrorKind`, `OptimizeErrorKind`, `AdaptErrorKind`, `RepositoryErrorKind`, `CatalogProviderErrorKind`, `FileSystemErrorKind`, `IrErrorKind`). The fused helper unifies these via `From` impls into `SemStraitErrorKind`; no numeric subsystem prefix exists in v1. Per-stage observability flows through `tracing` per `30 §6`; library code never writes to stdout/stderr. |

### 3.6 Observability via `tracing`

`SemStrait`'s methods emit structured `tracing` events and spans for the
work they orchestrate; informational signals never reach the
`Diagnostic<K>` channel (`30 §6`).

**What this crate emits.** The orchestrator opens one `tracing::span!` per
public-method invocation and a child span per delegated stage (parse,
validate, compile, plan, optimize, adapt). Stage spans carry stable
fields: `stage` (one of `parse | validate | compile | plan | optimize |
adapt | repository | catalog_provider | file_system`), `model_revision`
when reachable from the inputs, and `request_id` when the `Request`
carries one (`31 §14`). Inside each stage span the owning crate emits
its own events — `semstrait-api` adds none of its own beyond span
boundaries.

**Levels.**

| Level     | What rides on it                                                              |
|-----------|--------------------------------------------------------------------------------|
| `error`   | `Severity::Error` diagnostics that were just produced, mirrored as a tracing event with the kind variant in a structured field. The diagnostic itself is still returned in the `Result`; `tracing` is for the embedder's logging pipeline. |
| `warn`    | `Severity::Warning` diagnostics, mirrored as above. Mirroring is opt-out at the embedder via filter. |
| `info`    | Stage transitions: "compile starting", "compile finished in 12 ms", "loaded manifest from repository". One event per stage entry / exit. |
| `debug`   | Per-step granularity inside each stage: "resolving dataset `orders`", "binding column `customer_id`". Most of these come from `33`/`34`/`36`, not from `38` itself. |
| `trace`   | Fine-grained traversal: per-row, per-binding, per-rewrite-rule events. Reserved for diagnosing pipeline internals. |

**Tracing is opt-in.** No subscriber is installed by `semstrait-api`.
Embedders configure `tracing-subscriber` to route events to their
preferred sink. The library never writes to stdout / stderr; the only
writers are the embedder's subscriber and the `Diagnostic<K>` carriers
returned in `Result`. This matches the discipline established in
`30 §6` and the per-crate I12 invariants.

**CLI convention.** When a binary embedder (e.g. `semstrait-cli`,
explorer harnesses, examples) maps verbosity flags onto `tracing` levels,
the canonical mapping is:

| Flag          | `RUST_LOG` equivalent | Stages emit                                  |
|---------------|------------------------|-----------------------------------------------|
| (none)        | `warn`                 | `error` + `warn` (default; quiet operation)   |
| `--info`      | `info`                 | `error` + `warn` + `info` (stage transitions) |
| `--debug`     | `debug`                | adds per-step events                          |
| `--trace`     | `trace`                | adds fine-grained per-binding / per-row events|

The flag set is documented as a recommendation, not an enforced
contract: `semstrait-api` is a library and does not provide a CLI.
Embedders are free to expose their own knobs, but ratifying this
mapping at the spec level keeps multiple front-ends consistent.

**Relation to `WarningPolicy`.** Tracing mirroring is independent of
`WarningPolicy`. The diagnostic vector is the canonical record;
`tracing` is a side-channel for embedder visibility. Setting
`WarningPolicy::FailOnWarning` does not suppress the `warn`-level
mirror — the embedder still sees the mirrored event, alongside the
`Err((fatal, warnings))` tuple returned to the caller.

---

## Testing Strategy

- SemStrait struct shape matches §3.2 (field roster: catalog_provider, file_system, repository, function_registry, optimizer_passes, warning_policy)
- Method roster signatures match §3.3 (compile_from_yaml, compile_from_model async; plan, adapt sync; compile_and_plan_and_adapt async; save_manifest, load_manifest, validate_manifest async)
- Method contracts per §3.4 (compile_from_yaml parse→validate→compile error short-circuit and warning accumulation; plan wraps planner and optimizer; adapt passthrough AdaptErrorKind; compile_and_plan_and_adapt fused multi-stage; save/load require repository.is_some())
- I6 invariant (no .await in plan/adapt)
- I11 invariant (compile* async I11a; save/load/validate async I11b)
- I12 invariant (typed-kind enums and tracing)
- Tracing span-per-stage with stable fields (stage, model_revision, request_id)
- Send+Sync+Clone fanout
- Re-exports roster matches §2.2

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
