---
id: 38-api-lifecycle-boundaries
status: ratified
version: 2.1
impl: planned
---

# Lifecycle Patterns and Boundaries

This leaf owns lifecycle patterns and boundaries: one-shot convenience (§7 compile_and_plan_and_adapt fused helper semantics, success-arm warning concatenation, error-arm first-failing-stage wrapping, when NOT to use), streaming/incremental use (§8 compile-once plan-many Arc<SemanticManifest> pattern, lifecycle expectations manifest-freshness/no-implicit-caching/per-request-Request/fan-out, compile-and-persist save/load round-trip, out-of-v1-scope incremental-mutation/streaming-iteration), async boundaries (§9 per-method async posture table, no-.await-in-hot-path enforcement I6, runtime-independence, fused-helper async→sync transition, blocking-in-callers Q-API-007), diagnostic propagation (§10 stage-internal source-order, cross-stage concatenation, no-deduplication rationale, severity-ordering-not-enforced, warning-policy escalation), stability (§11 crate-level Stable-in-v1, per-item stability SemStrait/SemStraitBuilder/WarningPolicy/SemStraitErrorKind Stable and PipelineOutcome Provisional, trait-object surfaces MINOR-cascading, const-fn factories, auto-trait posture Send+Sync+Clone), and crate boundaries (§12 delegation table NO new-domain-types/new-*ErrorKind/SQL-emission/planner-adapter-logic/function-registry-construction/session-state/catalog-branching/cross-stage-dependency-reasoning plus YES orchestration/warning-policy/builder/unified-error).

## 7. One-shot convenience

### 7.1 `compile_and_plan_and_adapt`

The fused pipeline entry:

```rust
pub async fn compile_and_plan_and_adapt(
    &self,
    yaml: &str,
    request: Request,
    adapter: &dyn EngineAdapter,
) -> Result<
    (EngineArtifact, Diagnostics<SemStraitErrorKind>),
    (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>),
>;
```

**Semantics.** Runs the full pipeline:

1. Parse+validate+compile (`compile_from_yaml` logic).
2. Plan+optimize (`plan` logic, internally synchronous).
3. Adapt (`adapt` logic, internally synchronous).

Async because step 1 is async (I11a); steps 2 and 3 execute synchronously inside the same future, polled to completion before the future yields. There is no `.await` point between step 2 and step 3 — callers observe this as a single async call that internally switches from async (I/O-bearing) to sync (hot-path) execution.

**Success arm.** Returns the final `EngineArtifact` and the accumulated warning vector in source-stage order (parse → validate → compile → plan → adapt). Empty vector if every stage was clean.

**Error arm.** Returns `Err((Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>))`. The fatal is the first stage's failure, with its kind wrapped via `From` into `SemStraitErrorKind` (§6.3); warnings produced up to that point live in the `Diagnostics<SemStraitErrorKind>` slot per `30 §7`. The outer `SemStraitErrorKind` variant identifies which stage failed; destructuring the inner `*ErrorKind` identifies what exactly went wrong.

### 7.3 When NOT to use the fused helper

- **Streaming / incremental.** Callers reusing a `SemanticManifest` across many requests prefer the compile-once pattern (`§8`).
- **Per-stage inspection.** Callers that need to inspect the `SemanticManifest` (e.g. to write it to a cache, to log the schema, to emit a dataflow diagram) must use the per-stage methods.
- **Custom error routing.** Callers that want to route parse errors differently from adapt errors inspect the `SemStraitErrorKind` variant on `Err`, which is identical cost to using per-stage methods.

---

## 8. Streaming / incremental use

### 8.1 Compile-once, plan-many

The expected pattern for services answering many requests against a stable model:

```rust
// Once at startup:
let (manifest, _build_warnings) = semstrait.compile_from_yaml(&yaml).await?;
let manifest: Arc<SemanticManifest> = Arc::new(manifest);

// Per request:
let (plan, _plan_warnings) = semstrait.plan(&manifest, request)?;
let (artifact, _adapt_warnings) = semstrait.adapt(adapter, &plan, &manifest)?;
```

`SemanticManifest` is `Send + Sync` and cheap to `Arc`-share (`33 §7.6`). The `plan` + `adapt` hot path is synchronous (I6), so a service that holds an `Arc<SemanticManifest>` answers requests on many threads without re-entering async.

### 8.2 Lifecycle expectations

- **SemanticManifest freshness.** A `SemanticManifest` captures a catalog snapshot at compile time (`33 §13.2`). Long-lived manifests SHOULD be periodically re-validated via `SemStrait::validate_manifest` (I11b); a `Breaking` `DriftStatus` means re-compile.
- **No implicit caching.** `SemStrait` does NOT cache manifests, plans, or artifacts. Callers that want caching hold `Arc<SemanticManifest>` themselves.
- **Per-request construction.** `Request` (`31 §14`) is a value type; `SessionContext` may shift per request without forcing re-compile.
- **Fan-out.** `SemStrait` is `Send + Sync + Clone` (cheap `Arc`-clone); fan out via `Arc<SemStrait>` into worker tasks.

### 8.3 Compile-and-persist

```rust
let (manifest, _w_compile) = semstrait.compile_from_yaml(&yaml).await?;
let (id, _w_save)          = semstrait.save_manifest(&manifest).await?;
// Later, in a different process:
let (manifest, _w_load)    = semstrait.load_manifest(id).await?;
let (plan, _w_plan)        = semstrait.plan(&manifest, request)?;
```

`SemanticManifestId` round-trips; identical `(model, catalog snapshot)` pairs produce the same id (`33 §3.5`), so save/load is content-addressable.

### 8.4 Out of v1 scope

Incremental model mutation (re-compile on every change; `Q-API-005`) and streaming request iteration (caller-side loop in v1) are not supported by `SemStrait` directly.

---

## 9. Async boundaries

### 9.1 Per-method async posture

| Method                        | Async? | Why |
|-------------------------------|--------|-----|
| `builder()`                   | sync   | pure construction |
| `compile_from_yaml`           | async  | I11a — awaits `CatalogProvider` / `FileSystem` via `semstrait_manifest::compile` |
| `compile_from_model`          | async  | I11a — same as above, parse skipped |
| `plan`                        | sync   | I6 — planner hot path |
| `adapt`                       | sync   | I6 — adapter hot path |
| `compile_and_plan_and_adapt`  | async  | I11a — straddles compile; `.await` only inside the compile segment |
| `save_manifest`               | async  | I11b — awaits `Repository::save` |
| `load_manifest`               | async  | I11b — awaits `Repository::load` |
| `validate_manifest`           | async  | I11b — awaits `CatalogProvider::check_schema_drift` |
| accessors / utilities         | sync   | pure |

This matches `30 §9`'s per-crate posture row for `semstrait-api` (async at compile-time entry; sync at plan-time entry). The fused helper `compile_and_plan_and_adapt` is the only async method that internally contains synchronous stages; all other async methods are async end-to-end.

### 9.2 No `.await` in the hot path

`plan` and `adapt` are synchronous by signature and synchronous by implementation. No transitive `.await` point is reachable from them. This is enforced by:

- Method signatures (no `async fn` / no `-> impl Future<…>` / no `futures::executor::block_on`).
- `34`'s planner crate is entirely synchronous (`30 §9`).
- `36`'s adapter crate is entirely synchronous (`30 §9`).
- `semstrait-api` introduces no new async dependency inside the `plan` / `adapt` method bodies.

### 9.3 Runtime independence

Per `30 §9`, async surfaces are executor-agnostic. `semstrait-api`'s async methods do not pin a runtime; callers drive them on whatever executor they prefer. Bundled providers (`S3FileSystem`, `IcebergRestCatalogProvider`, `FileSystemRepository`) use `tokio` by convention, so most callers will have a `tokio` runtime in scope.

### 9.4 Fused helper: async → sync transition

Inside `compile_and_plan_and_adapt`, the last `.await` is in the compile step; `plan` and `adapt` execute synchronously on the same future. From the caller's perspective, a single async call internally transitions from I/O-bearing execution to pure CPU work once the SemanticManifest lands. This is the I6 "hot path is synchronous" discipline surfaced at the fused-helper level.

### 9.5 Blocking in callers

Callers that cannot block a thread drive the async future on an appropriate executor. `semstrait-api` provides no `sync_*` variants; callers who need a sync-over-async bridge wrap with their runtime's `block_on`. See `Q-API-007`.

---

## 10. Diagnostic propagation

### 10.1 Stage-internal ordering

Within a single stage, diagnostics appear in **source order**:

- Parse / validate diagnostics are produced by visiting the YAML AST in lexical order (`32 §10`).
- Compile diagnostics are produced by visiting the `SemanticModel` in canonical BTreeMap order (`33 §13`) and, within each data-kind, in the order defined by the owning `20`–`25` spec.
- Plan / adapt diagnostics inherit their ordering from the planner's and adapter's internal traversal order, which is deterministic per I4.

### 10.2 Cross-stage ordering

The accumulated warning vector in `compile_and_plan_and_adapt`'s success arm is **stage-concatenated**:

```
[ parse_diags ... , validate_diags ... , compile_diags ... , plan_diags ... , adapt_diags ... ]
```

Within each stage block, source-order. Across stage blocks, pipeline order. No interleaving. This guarantees that two identical runs produce byte-identical diagnostic vectors (subject to non-determinism in the catalog — which `33 §13.2` requires to be stabilized into the SemanticManifest).

### 10.3 De-duplication

`semstrait-api` does NOT de-duplicate diagnostics. Two reasons:

1. Duplication across stages is **informative**: a missing column reported once at validate and again at compile is a signal that the author saw the early error message but did not act on it, and the redundant later report helps downstream tooling present a consolidated view.
2. De-duplication requires a deterministic equivalence relation on `Diagnostic<K>` (what makes two diagnostics equal? same kind variant? same kind + location? same kind + location + rendered message?) that has not been ratified in `30` and would need its own `Q-API-*` to land.

Callers that want de-duplicated output apply their own predicate to the returned `Diagnostics<SemStraitErrorKind>`. A conservative default (matching `kind` variant + matching `location.span()` = same diagnostic) is sketched in `Q-API-004`.

### 10.4 Severity ordering is not enforced

The accumulated vector may interleave `Severity::Warning` and `Severity::Error` diagnostics in any order — `Severity` is reduced to `{Error, Warning}` in `30 §5.2`, with informational signals on `tracing` per `30 §6`. Callers that want to surface errors first apply `.sort_by_key(|d| d.severity().sort_rank())` or an equivalent. `Severity` does not implement `Ord` in `31` (per `30 §5.2`, the enum is `#[non_exhaustive]`); a caller-supplied rank function is the idiomatic pattern.

### 10.5 Warning-policy escalation

When `WarningPolicy ∈ {FailOnWarning, Strict}` escalates a warning into a stage-fatal, the escalated `Diagnostic<K>` (variant identity preserved) lands in the `Err` tuple's fatal slot. Prior warnings in the same stage are preserved in the same tuple's warnings slot. Warnings from prior stages travel forward in the helper's `Diagnostics<SemStraitErrorKind>` slot (re-keyed via the `From` impls of §6.3), keeping the cross-stage history intact. See `§5.3` for the escalation mechanics.

## 11. Stability

### 11.1 Crate-level stability

Per `30 §13`, `semstrait-api` is **Stable in v1**. The `SemStrait` / `SemStraitBuilder` method rosters, the `WarningPolicy` variants, the `SemStraitErrorKind` variant set, and the re-export list (`§2.2`) all participate in the workspace's MAJOR / MINOR / PATCH discipline.

### 11.2 Per-item stability

All `SemStrait` methods, `SemStraitBuilder` chain methods, `WarningPolicy`, `SemStraitErrorKind`, and `StageOrigin` are **Stable** in v1. `PipelineOutcome` is **Provisional** pending `Q-API-002`. Adding a new `SemStrait` method is MINOR (struct, not trait); adding a `SemStraitBuilder` field with a default is MINOR; a new required field is MAJOR. `#[non_exhaustive]` enums (`WarningPolicy`, `SemStraitErrorKind`, `StageOrigin`) admit MINOR variant additions, with the same MINOR/MAJOR rules cascading from upstream `*ErrorKind`s reachable via the wrapping variants.

### 11.3 Trait-object surfaces

The builder exposes `Arc<dyn CatalogProvider>`, `Arc<dyn FileSystem>`, `Arc<dyn Repository>`, `Box<dyn OptimizerPass>`. These traits are owned by `37` / `33` / `34`; MINOR additions to those traits (via default methods per `30 §11`) stay MINOR here. MAJORs there cascade to MAJOR here.

### 11.4 Const-fn factories

`SemStraitBuilder::new` is `const fn`. `SemStrait::builder` cannot be `const fn` (builder's `Vec::new()` and `Arc` usage are non-const); callers needing a compile-time builder hold a `SemStraitBuilder` in `const` directly.

### 11.5 Auto-trait posture

`SemStrait: Send + Sync + Clone` (`Arc::clone` on every field). `SemStraitBuilder: Send + Sync`, not `Clone` (consumed by `build`; multiple builders for multiple `SemStrait`s).

---

## 12. Crate boundaries

| Boundary                                         | Status                                                                 |
|--------------------------------------------------|------------------------------------------------------------------------|
| New domain types (`Resolved*`, `PlanNode`, `EngineArtifact` variants) | **NO.** Zero new types cross the boundary. |
| New per-crate `*ErrorKind`                       | **NO.** `SemStraitErrorKind` is a thin sum over upstream kinds plus two intrinsic API-only variants (`BuilderInvalid`, `NoRepositoryConfigured`); no parallel kind taxonomy. |
| SQL emission                                     | **NO.** `adapt` returns `EngineArtifact`; SQL comes from `EngineAdapter::emit` on the adapter. |
| Parse / validate / compile / plan / adapt algorithms | **NO.** All delegated to `32` / `33` / `34` / `36`. |
| Direct I/O                                       | **NO.** Every I/O call routes through `Arc<dyn CatalogProvider>`, `Arc<dyn FileSystem>`, `Arc<dyn Repository>`. No network, disk, or process-call lives inside `semstrait-api`. |
| Catalog-type branching (I3)                      | **NO.** Every dispatch over `&dyn CatalogProvider`; concrete type is not inspected. |
| Function-registry construction                   | **NO.** `function_registry()` from `31` is the sole source of the global registry; the builder's field accepts a pre-built `&'static FunctionRegistry` only. |
| SemanticManifest mutation                                | **NO.** `SemanticManifest` is sealed per `33 §7.4`. `SemStrait::save_manifest` takes `&SemanticManifest` and produces a `SemanticManifestId`; it does not mutate. |
| Plan mutation                                    | **NO.** `SemanticPlan` is `#[non_exhaustive]` but treated as immutable downstream of the planner. |
| Async runtime pinning                            | **NO.** `tokio`-neutral per `30 §9`. |
| Warning-policy enforcement at stages              | **NO.** Stages are unaware of `WarningPolicy`; enforcement is strictly at `semstrait-api`'s return boundary (`§5.4`). |
| Cross-stage dependency reasoning                 | **NO.** `SemStrait::plan` does not inspect the `SemanticManifest`'s internal structure; it hands the `SemanticManifest` + `Request` to the planner and consumes the result. |
| One-shot convenience                             | **YES.** `compile_and_plan_and_adapt` is owned here. |
| Warning aggregation at stage boundaries          | **YES.** `WarningPolicy` + diagnostic concatenation live here. |
| Builder ergonomics                               | **YES.** `SemStraitBuilder` is owned here. |
| Unified error kind                               | **YES.** `SemStraitErrorKind` (sum + intrinsic variants) is owned here, with `Diagnose` and `From` impls. |

The rightmost "YES" row is narrow: orchestration, warning policy, builder, unified error. Everything else is delegation.

---

## 14. Cross-references

- Overview: `00 §4.2 verbs`, `00 §5 pipeline`.
- Invariants: `00 §9 I3, I6, I7, I10, I11 (I11a + I11b), I12`.
- API contracts: `30 §4 (non-exhaustive policy)`, `30 §5–§6 (Diagnostic + error codes)`, `30 §7 (stage return shapes)`, `30 §8 (trait rules)`, `30 §9 (per-crate async matrix)`, `30 §10 (feature-flag policy)`, `30 §13 (stability tiers)`.
- Shared primitives: `31 §5 (FunctionRegistry)`, `31 §7 (Diagnostic)`, `31 §14 (Request / SessionContext)`.
- Parse / validate: `32 §9 (parse signature)`, `32 §11 (validate signature)`.
- Compile: `33 §9 (compile signature)`, `33 §10 (CompileError / RepositoryErrorKind)`, `33 §11 (Repository)`, `33 §12 (I11b drift-check caller)`.
- Plan / optimize: `34 §10 (PlanErrorKind / OptimizeErrorKind)`.
- IR types: `35 §3 (SemanticPlan)`, `35 §6 (EngineArtifact)`, `35 §11 (IrErrorKind)`.
- Adapt: `36 §3 (EngineAdapter::adapt)`, `36 §9 (EngineArtifact / SqlArtifact shape)`, `36 §10 (AdaptErrorKind)`.
- Catalog / FileSystem: `37 §3 (CatalogProvider)`, `37 §5 (FileSystem)`, `37 §8 (CatalogProviderErrorKind / FileSystemErrorKind)`, `37 §9 (I11b drift gate)`.
- Downstream: `39 (semstrait-facade)` — zero-config wrapper over this crate.

---

## Testing Strategy

- One-shot convenience (compile_and_plan_and_adapt success warning concatenation in parse+validate+compile+plan+adapt source-stage order; error first-failing-stage wrapping with warnings-up-to-that-point preserved; when NOT to use for streaming, per-stage-inspection, custom-error-routing)
- Streaming/incremental (compile-once plan-many Arc<SemanticManifest> lifecycle; manifest-freshness validate_manifest I11b; no-implicit-caching; per-request Request construction; fan-out Send+Sync+Clone)
- Compile-and-persist (save_manifest, load_manifest SemanticManifestId round-trip content-addressable)
- Async boundaries (per-method async posture: compile*, save, load, validate async; plan, adapt sync; no-.await-in-hot-path I6; runtime-independence; fused-helper async→sync transition; blocking-in-callers Q-API-007)
- Diagnostic propagation (stage-internal source-order; cross-stage concatenation; no-deduplication rationale; severity-ordering-not-enforced; warning-policy escalation)
- Stability (crate-level Stable-in-v1; per-item stability SemStrait/SemStraitBuilder/WarningPolicy/SemStraitErrorKind Stable and PipelineOutcome Provisional; trait-object surfaces MINOR-cascading; const-fn factories; auto-trait posture Send+Sync+Clone)
- Crate boundaries (delegation-table NO new-domain-types/new-*ErrorKind/SQL-emission/etc plus YES orchestration/warning-policy/builder/unified-error)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
