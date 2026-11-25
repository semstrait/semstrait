---
id: 30-diagnostic-structure
status: ratified
version: 2.1
impl: implemented
---

# Diagnostic Structure

This leaf owns the generic Diagnostic<K> envelope, Severity, Location, Span, SourceId, Diagnose trait, per-stage typed-kind discipline, and cross-stage aggregation patterns.

## 5. Diagnostic Structure

The diagnostic surface is **typed** and **per-stage**. Every consumer crate declares its own kind enum (`ParseErrorKind`, `ValidateError`, `CompileError`, …) implementing the `Diagnose` trait; the generic `Diagnostic<K>` envelope wraps a kind with severity and source location. Public entry points that can fail return either `Result<…, Diagnostic<K>>` (fail-fast) or `Result<…, Diagnostics<K>>` (accumulating). Raw `String` errors, `anyhow::Error`, and `Box<dyn Error>` are banned on public APIs (§5.5).

There is no central error-code allocation table. Stable identification is by **variant identity** of the per-stage kind — renaming a variant is MAJOR per §2; adding one inside a `#[non_exhaustive]` enum is MINOR. The prior code-allocation framework (§6 of earlier drafts) is retired; §6 now hosts the workspace observability policy.

### 5.1 Generic `Diagnostic<K>` envelope

```rust
#[non_exhaustive]
pub struct Diagnostic<K: Diagnose> {
    /// The typed payload describing what failed. Each crate declares
    /// its own `K`; consumers `match` on variants for compile-time
    /// exhaustiveness over the kind enum.
    pub kind: K,

    /// Severity class; see §5.2.
    pub severity: Severity,

    /// Optional source-level location. `None` for context-free errors
    /// (e.g. a `CatalogUnavailable` that originates from a network
    /// failure with no document anchor).
    pub location: Option<Location>,

    /// Optional supplementary remarks attached to the primary kind —
    /// short free-text strings ("declared here at line 42", "see also
    /// the alias entry at …") that callers render alongside the primary
    /// message. Not recursive; not a structured cross-reference.
    pub notes: Vec<String>,
}

pub type Diagnostics<K> = Vec<Diagnostic<K>>;
```

`Diagnostic<K>` lives in `semstrait-common` (ratified in `31 §7`). The `#[non_exhaustive]` annotation permits adding fields (e.g. a `related: Vec<Location>` cross-reference) in a MINOR release. `Diagnostics<K>` is a transparent type alias — callers may use either form interchangeably.

Construction sites are crate-local; callers do not construct `Diagnostic<K>` by hand. Each per-stage helper in the owning crate builds a kind, sets severity (defaulting to `K::severity_default()`), attaches optional location / notes, and wraps.

### 5.2 `Severity`

```rust
#[non_exhaustive]
pub enum Severity {
    /// Fatal. In fail-fast stages, raises Err immediately. In
    /// accumulating stages, contributes to the final Err vector.
    Error,

    /// Advisory. Returned alongside Output on the success arm of
    /// either stage class. MUST NOT be silently dropped at API
    /// boundaries.
    Warning,
}
```

Severity carries **message intent only**. Control flow (accumulating vs fail-fast) is a property of the function's return signature (§7), not the diagnostic. `#[non_exhaustive]` leaves room for future gradations (e.g. `Note`) without breaking matcher code. The earlier `Info` variant retired in this ratification; informational signals belong on the `tracing` channel (§6), not the diagnostic channel.

### 5.3 `Location`, `Span`, `SourceId`

```rust
/// Source-level location.
#[non_exhaustive]
pub struct Location {
    pub source: SourceId,
    pub span: Span,
}

/// Half-open byte range into the source document.
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Opaque source identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct SourceId(/* crate-private */);
```

`Location`, `Span`, `SourceId` live in `semstrait-common`. `SourceId` is opaque — its variant set is private; constructors live on the producing crate (`semstrait-model` for YAML-file sources, etc.). Public surface exposes `SourceId::unknown()`, `as_str()`, plus `Eq` / `Hash` / `Display`.

**Retired.** `ContextLine` is retired. The earlier "supplementary line with annotated pointer" role is covered either by (a) `notes: Vec<String>` on `Diagnostic<K>` for short remarks, or (b) richer typed location information embedded directly in the kind variant when structurally meaningful (e.g., a `ShapeFieldConflict` variant carrying `occurrences: Vec<Location>`).

### 5.4 `Diagnose` trait

Each per-stage kind implements:

```rust
pub trait Diagnose {
    /// Human-readable rendering. Powers the Diagnostic's Display impl;
    /// must not include line breaks (callers add their own framing).
    fn message(&self) -> String;

    /// Default severity for this variant. Construction sites may
    /// override; most callers accept the default.
    fn severity_default(&self) -> Severity;

    /// Foreign-error chain for std::error::Error interop. Default None.
    /// Variants wrapping foreign errors (e.g. ParseErrorKind::Yaml(serde_yaml::Error))
    /// override this to return Some(&inner).
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
```

`semstrait-common` provides blanket impls:

```rust
impl<K: Diagnose> std::fmt::Display for Diagnostic<K> { /* delegates to K::message() */ }
impl<K: Diagnose + std::fmt::Debug> std::error::Error for Diagnostic<K> { /* chains via K::cause() */ }
```

Foreign-error wrapping is **variant-side**: a per-stage kind enum has typed variants for the foreign errors it actually wraps (`ParseErrorKind::Yaml(serde_yaml::Error)`, `FileSystemErrorKind::Io(std::io::Error)`); the wrapped error participates in the `std::error::Error` chain via the `cause()` override. There is no struct-side `source: Box<dyn Error>` field on `Diagnostic<K>` — the kind is the source of truth.

### 5.5 Banned patterns

The following patterns are forbidden on any `pub` entry point in the workspace:

- Returning `Result<T, String>`, `Result<T, anyhow::Error>`, `Result<T, Box<dyn Error>>`, or any untyped error.
- Returning `Result<T, Diagnostic>` without the type parameter — every `Diagnostic` is `Diagnostic<K>` for some specific `K`.
- Panicking on caller-reachable input (malformed YAML, missing catalog entries, unsupported engine features). A panic is a semstrait bug.
- Writing diagnostics directly to `stdout` / `stderr`. Diagnostics flow through `Result`; the embedder decides where to render them. See §6 for the parallel observability channel.

Internal-only APIs (`pub(crate)`) may use kind enums directly (without the `Diagnostic<K>` envelope) where construction-site location is implicit.

### 5.6 Cross-stage aggregation

When multiple stage kinds need to surface through a single helper (e.g. `compile_and_plan_and_adapt` in `semstrait-api`), the fused helper declares a sum-typed kind:

```rust
#[non_exhaustive]
pub enum SemStraitErrorKind {
    Parse(semstrait_model::ParseErrorKind),
    Validate(semstrait_model::ValidateError),
    Compile(semstrait_manifest::CompileError),
    Plan(semstrait_planner::PlanErrorKind),
    Adapt(semstrait_adapter::AdaptErrorKind),
}

impl From<semstrait_model::ParseErrorKind>     for SemStraitErrorKind { /* … */ }
impl From<semstrait_model::ValidateError>  for SemStraitErrorKind { /* … */ }
impl From<semstrait_manifest::CompileError> for SemStraitErrorKind { /* … */ }
// ...
```

The fused helper returns `Diagnostic<SemStraitErrorKind>` (or `Diagnostics<…>`); per-stage results lift via `From`. Cross-crate kind-nesting is permitted but not mandatory — a stage's kind enum MAY embed an upstream stage's kind directly (e.g. `CompileError::Parse(ParseErrorKind)`) when convenient; the fused sum at `38` is the canonical location for cross-stage aggregation.

## Testing Strategy

- Diagnostic<K> envelope construction with all field combinations (§5.1 struct fields)
- Diagnose trait blanket impls for Display and Error (§5.4 blanket impls)
- Cross-stage aggregation via From impls preserving kind identity (§5.6 SemStraitErrorKind)
- Banned-pattern detection via CI check for Result<T,String> (§5.5 prohibited patterns)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
