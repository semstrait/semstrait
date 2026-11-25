---
id: 30-async-posture
status: ratified
version: 2.1
impl: implemented
---

# Async Posture

This leaf owns the per-crate sync/async posture — which crates are sync-only, which have compile-time async, runtime choice, and no .await inside the hot path.

## 9. Async Posture per Crate

Per I6 (sync hot path) and I11 (gated I/O), most crates are sync-only. The exceptions are the crates that orchestrate compile-time I/O or that bridge to async metadata / storage providers.

| Crate | Sync / async | I/O allowed? | Rationale |
|---|---|---|---|
| `semstrait-common` | Sync only | No | Pure primitives (`Diagnostic`, `Span`, tree traversal, `ValidateError`); no I/O surface. `DataType` / `CanonicalFn` live in `semstrait-model`. |
| `semstrait-model` | Sync only | No | `parse` and `validate` are pure transformations over in-memory YAML. |
| `semstrait-manifest` | Compile-time async; plan-time sync | Compile-time via providers | The `compile` entry point is `async fn` (awaits `CatalogProvider` / `FileSystem`). The `SemanticManifest` is then consumed synchronously; no `async fn` at plan time. |
| `semstrait-planner` | Sync only | No | `plan` and `optimize` are the I6 hot path. |
| `semstrait-ir` | Sync only | No | Canonical IR types; no I/O. |
| `semstrait-adapter` | Sync only | No | `adapt` is the I6 hot path. Per-engine adapter crates inherit the posture. |
| `semstrait-catalog` | Trait surface async; impls per-provider | Yes (sole I/O home) | `CatalogProvider` and `FileSystem` are `async fn`-in-trait. Individual impls may be sync-over-async or genuinely async. |
| `semstrait-api` | Async at compile-time entry; sync at plan-time entry | Via manifest + catalog | Bundles the compile-time async path and the query-time sync path under a single crate surface. |
| `semstrait-facade` | Same as `-api` | Same | Thin re-export + one-shot-use convenience over `semstrait-api`. |

**Runtime choice.** Async surfaces are executor-agnostic. `semstrait-manifest`'s `compile` and `semstrait-catalog`'s trait methods are `async fn` but do not pin a specific runtime. Bundled impls (e.g. `FileSystemRepository`) use `tokio` by convention; the API surface accepts any executor. Per-crate docs specify the runtime dependency stance.

**No `.await` inside the hot path.** The synchronous crates above MUST NOT introduce `.await` points, even via third-party dependencies. Dependency choice is a per-crate concern but is audited against I6 at review time.

## Testing Strategy

- Per-crate async posture matches §9 table (sync-only vs compile-time async)
- Sync crates have zero .await points (hot-path invariant)
- Compile-time async surfaces are executor-agnostic (no tokio pinning in trait signatures)
- Hot-path crates (planner, adapter) remain sync-only

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
