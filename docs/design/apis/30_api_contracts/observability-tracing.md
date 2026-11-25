---
id: 30-observability-tracing
status: ratified
version: 2.1
impl: implemented
---

# Observability and Tracing

This leaf owns the workspace observability policy — tracing adoption, no stdout/stderr from library code, canonical field names, and Diagnostic ↔ tracing orthogonality.

## 6. Observability and Tracing

> **§6 was previously "Stable Error-Code Format" — retired.** The numeric `{SUBSYSTEM}_{SEVERITY}_{NUMBER}` allocation table, reserved-prefix list, and per-subsystem ranges are gone. Diagnostic identification is now by **per-stage typed-kind variant identity** per §5; rename = MAJOR, additive variant inside `#[non_exhaustive]` = MINOR. This slot now hosts the workspace observability policy, the parallel channel to the `Result`-borne diagnostics of §5.

### 6.1 No direct stdout / stderr from library code

No `semstrait-*` library crate writes to `stdout` or `stderr`. There is no `println!`, `eprintln!`, `print!`, `eprint!`, or any equivalent on any library code path. Diagnostics returned via `Result` are the output channel for caller-actionable findings. An embedder (the eventual `semstrait-cli` binary, an LSP server, a web service, a test harness) configures its own destination for any visible text.

This is a workspace-level invariant, enforced by review and (optionally) by a `clippy` lint configuration that bans `print_stdout` / `print_stderr` for `semstrait-*` crates.

### 6.2 `tracing` is the observability channel

Every `semstrait-*` library crate uses [`tracing`](https://docs.rs/tracing) to emit structured spans and events. No other observability dependency is permitted (no `log`, no custom event channel, no embedder-coupled logger).

Every `pub` stage entry-point carries `#[tracing::instrument]`:

```rust
#[tracing::instrument(level = "info", skip(model), fields(stage = "compile"))]
pub fn compile(model: &SemanticModel) -> Result<…> { /* … */ }
```

The macro propagates spans across `.await` automatically (relevant for `compile` and other compile-time async boundaries — no manual `Instrument` wrapping required). Internal helpers below the public boundary inherit the parent span and need no instrumentation of their own unless an author wants finer-grained sub-spans.

### 6.3 Level discipline

| Level | Purpose | Examples |
|---|---|---|
| `error` | Library refused to proceed and is bubbling | reserved (most errors are returned via `Result`; `error!` is for the "this is an internal bug" class) |
| `warn` | Library-internal advisory not surfaced as a Diagnostic | retrying a transient I/O failure; falling back to a default config value; deprecation usage at boundary |
| `info` | "What is happening" signal a CLI user might want | "parsed 200 entities", "resolved 14 sources (8 cached, 6 fetched)", "compile completed in 240ms" |
| `debug` | Developer-targeting state visibility | scope graph constructed (N nodes, M edges); chosen adapter `Snowflake@v6` |
| `trace` | Deep inspection | every expression normalisation step; every binding resolution |

`tracing` short-circuits at the macro site when no subscriber filters the level in; off-by-default levels (`debug`, `trace`) carry near-zero overhead in production builds.

### 6.4 Diagnostic ↔ tracing orthogonality

A `Diagnostic<K>` returned via `Result` is **never also** emitted via `tracing::warn!` (or any other level) by the library itself. The two channels carry orthogonal information:

- A `Diagnostic` surfaces something *about the user's model* — the LSP / IDE / CLI renders it on the source. Caller MUST handle it; it is part of the API contract.
- A `tracing` event surfaces something *about library behavior* — operators / developers see it in logs. Caller MAY observe it; it is not part of the API contract.

Embedders that want diagnostics in their tracing stream convert in their own code:

```rust
for diag in diagnostics {
    match diag.severity {
        Severity::Error   => tracing::error!(?diag, "diagnostic"),
        Severity::Warning => tracing::warn!(?diag, "diagnostic"),
    }
}
```

Library-internal warnings (a fallback applied, a transient retry, a deprecated entry-point used) flow only through `tracing::warn!`. They are not `Diagnostic`s — they describe library behavior, not the user's model.

### 6.5 Canonical field-name vocabulary

Subscribers can rely on these field names being used consistently across every stage span / event in the workspace. When an author emits an event with information matching one of these, the canonical field name MUST be used; additional non-canonical fields MAY be added freely.

| Field | Type | Semantics |
|---|---|---|
| `stage` | `&'static str` | One of `parse`, `validate`, `compile`, `plan`, `optimize`, `adapt`, `catalog.resolve`, `fs.read`, `adapter.adapt`, `repository.load`, `repository.save`. Set on every stage span. |
| `entities` | `usize` | Entity count for any stage iterating over the model entity set. |
| `sources` | `usize` | Source-document or catalog-source count. |
| `duration_ms` | `u64` | Stage wall-time in milliseconds. Emitted on the closing `info` event of long stages. |
| `engine` | `&'static str` | Adapter target identifier (`snowflake`, `bigquery`, `duckdb`, …). |
| `adapter` | `&'static str` | Adapter crate identifier (`semstrait-adapter-snowflake`, …). |
| `kind` | `&'static str` | DataKind variant name when relevant (`Dataset`, `Grainset`, `Unionset`, `Joinset`). |

Canonical field names extend MINOR-additively: a new reserved field name lands with a MINOR bump and a migration-notes entry.

### 6.6 Subscriber configuration is the embedder's responsibility

`semstrait-*` library crates do not install a global `tracing` subscriber and do not declare a transitive dependency on a subscriber crate (`tracing-subscriber`, `tracing-bunyan-formatter`, etc.). Embedders configure their own.

The eventual `semstrait-cli` binary will configure a `tracing-subscriber::fmt` layer with verbosity controlled by `--info` / `--debug` / `--trace` flags; the API spec for that binary is out of scope for `30`. Any other embedder (LSP server, web service, test harness) is free to choose its own subscriber stack.

### 6.7 Async span propagation

Stages that run `async fn` bodies use `#[tracing::instrument]` on their public entry point. The macro hooks `.await` to keep spans entered/exited correctly across suspension points. No manual span management is required at the crate-author level. If a crate adds a manually-opened span around an `async` block, it MUST use `Instrument::instrument` / `.in_scope` correctly per the `tracing` documentation.

### 6.8 Banned patterns on the observability channel

- **`tracing::error!` at level `error` for caller-actionable conditions.** Caller-actionable failures flow through `Result<…, Diagnostic<K>>`. `error!` is reserved for bugs — conditions a `Diagnostic` would never describe.
- **Direct `stdout` / `stderr` writes** (re-stating §6.1).
- **Logging via `log` crate.** `tracing` re-exports a `log` compatibility layer; the workspace's `Cargo.toml` does not depend on `log` directly.
- **Subscriber registration in library code.** Only embedders register subscribers.

## Testing Strategy

- No stdout/stderr from library code via clippy lint banning print_stdout/print_stderr (§6.1 prohibited patterns)
- #[tracing::instrument] present on every pub stage entry-point (§6.2 instrumentation coverage)
- Canonical field-name usage matches §6.5 vocabulary (stage/entities/sources/duration_ms/engine/adapter/kind)
- Subscriber configuration deferred to embedder, no global subscriber in library crates (§6.6)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
