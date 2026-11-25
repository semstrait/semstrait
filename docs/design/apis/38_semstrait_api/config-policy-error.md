---
id: 38-api-config-policy-error
status: ratified
version: 2.1
impl: planned
---

# Configuration, Policy, and Error Types

This leaf owns configuration, policy, and error types: `SemStraitBuilder` configuration surface (chain methods with_catalog_provider, with_file_system, with_repository, with_function_registry, with_optimizer_pass, with_optimizer_passes, with_warning_policy, required-field validation at build, Default), `WarningPolicy` enum (Accumulate/FailOnWarning/Strict variants, escalation rules §5.3, application at stage boundary §5.4, interaction with fused helper §5.5, invariants §5.6), `SemStraitErrorKind` unified typed-error enum (Parse/Validate/Compile/Plan/Optimize/Adapt/Repository/CatalogProvider/FileSystem/BuilderInvalid/NoRepositoryConfigured variants, From<*ErrorKind> impls §6.3, Diagnose impl §6.5, variant-to-origin map §6.4), `StageOrigin` helper enum (Parse/Validate/Compile/Plan/Optimize/Adapt/Repository/CatalogProvider/FileSystem/Builder), and stability (#[non_exhaustive] MINOR-additive variant additions §6.7).

## 4. `SemStraitBuilder`

### 4.1 Purpose

`SemStraitBuilder` is the single construction path for `SemStrait`. Chain-style configuration methods populate its fields; `build` produces a validated `SemStrait` (or fails with a `Diagnostic<SemStraitErrorKind>` whose kind is `BuilderInvalid`, §6.2).

### 4.2 Struct shape

```rust
pub struct SemStraitBuilder {
    catalog_provider: Option<Arc<dyn CatalogProvider>>,
    file_system: Option<Arc<dyn FileSystem>>,
    repository: Option<Arc<dyn Repository>>,
    function_registry: Option<&'static FunctionRegistry>,
    optimizer_passes: Vec<Box<dyn OptimizerPass>>,
    warning_policy: WarningPolicy,
}
```

The builder itself is `pub`, but fields are `pub(crate)`; configuration flows through chain methods.

### 4.3 Required fields

| Field               | Required? | Default if unset                                        |
|---------------------|-----------|----------------------------------------------------------|
| `catalog_provider`  | yes       | — (build fails with `SemStraitErrorKind::BuilderInvalid`)                   |
| `file_system`       | yes       | — (build fails with `SemStraitErrorKind::BuilderInvalid`)                   |
| `repository`        | no        | `None` — save/load methods then return `SemStraitErrorKind::NoRepositoryConfigured` |
| `function_registry` | no        | `semstrait_common::function_registry()` (the process-global instance) |
| `optimizer_passes`  | no        | the canonical-pass list from `34 §*` (`Vec::new()` semantically means "use defaults", not "no passes") |
| `warning_policy`    | no        | `WarningPolicy::Accumulate` (`§5.2`)                    |

`catalog_provider` and `file_system` are required because every non-trivial compile invocation needs both, and silently defaulting either to `NoopCatalogProvider` / `LocalFileSystem` would mask configuration errors. Callers who want a truly zero-I/O build (unit tests, model-only work) construct an explicit `NoopCatalogProvider` + an explicit `LocalFileSystem` pointing at an empty root.

### 4.4 Chain methods

```rust
impl SemStraitBuilder {
    pub fn new() -> Self;            // all None / empty / default
    pub fn with_catalog_provider(self, cp: Arc<dyn CatalogProvider>) -> Self;
    pub fn with_file_system(self, fs: Arc<dyn FileSystem>) -> Self;
    pub fn with_repository(self, repo: Arc<dyn Repository>) -> Self;
    pub fn with_function_registry(self, reg: &'static FunctionRegistry) -> Self;
    pub fn with_optimizer_pass(self, pass: Box<dyn OptimizerPass>) -> Self;
    pub fn with_optimizer_passes(self, passes: Vec<Box<dyn OptimizerPass>>) -> Self;
    pub fn with_warning_policy(self, policy: WarningPolicy) -> Self;
    pub fn build(self) -> Result<SemStrait, Diagnostic<SemStraitErrorKind>>;
}

impl Default for SemStraitBuilder {
    fn default() -> Self { Self::new() }
}
```

Setters use the Rust idiomatic `with_*` prefix to clearly signal "consume self, return modified self" — the chain-builder convention shared with `reqwest::ClientBuilder` / `tokio::runtime::Builder` / `clap::Command`. `with_optimizer_pass` appends; `with_optimizer_passes` replaces. The field names on `SemStraitBuilder` and `SemStrait` themselves stay un-prefixed (`catalog_provider`, `file_system`, …) — `with_` is a setter convention, not a field convention.

### 4.6 Validation at `build`

`build` checks required fields (`catalog_provider`, `file_system`) and fills unset optional fields with their defaults (`§4.3`). No I/O runs at build time: a builder with an unreachable catalog builds successfully and the first `compile_*` call surfaces the connectivity failure.

```rust
pub fn build(self) -> Result<SemStrait, Diagnostic<SemStraitErrorKind>>;
```

On missing required fields: `Err(Diagnostic::error(SemStraitErrorKind::BuilderInvalid { missing }))`. `build` is a one-shot configuration check, not a stage entry-point — it doesn't produce warnings, so the simpler `Result<_, Diagnostic<K>>` shape is preferable to the fail-fast tuple.

### 4.7 `const fn` constructors

`SemStraitBuilder::new` is `const fn` (all fields initialize to `None` / `Vec::new()` / `WarningPolicy::Accumulate`), letting callers declare builders in `static` / `const` items. Chain methods accepting `Arc<dyn …>` cannot be `const`; `with_warning_policy` could in principle be made `const fn` in a future MINOR.

---

## 5. `WarningPolicy`

### 5.1 Purpose

The semstrait pipeline produces two diagnostic severities: `Severity::Error` (always fatal at its owning stage per `30 §7`) and `Severity::Warning` (advisory, preserved alongside success per `30 §7`). Informational signals — progress, telemetry, debug detail — flow through `tracing` (`30 §6`) and are not on the `Diagnostic<K>` channel. `WarningPolicy` is the knob for callers that want to escalate the advisory channel into a fatal at the API boundary.

### 5.2 Variants

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningPolicy {
    Accumulate,    // default — Severity::Warning passes through to the success arm's warnings slot
    FailOnWarning, // first Severity::Warning at any stage escalates to that stage's fatal
    Strict,        // reserved alias of FailOnWarning until Severity grows a third variant (see §5.3)
}
```

`#[non_exhaustive]` so future policies (e.g. `FailOnWarningAfter(Stage)`, `FilterByKind(&[&'static str])`) extend via MINOR (`30 §4`). `Default::default()` returns `Accumulate`.

### 5.3 Escalation rules

- **`Accumulate`.** No-op. Every `Severity::Warning` flows through to the success arm's `Diagnostics<K>` slot. `Severity::Error` is handled by the owning stage's fail-fast policy (`30 §7`); warnings produced up to the failure still accumulate in the `Err((fatal, warnings))` tuple. `Severity::Info` is not on the diagnostic channel — informational signals flow through `tracing` per `30 §6` (see §3.6 below).

- **`FailOnWarning`.** Any `Diagnostic<K> { severity: Warning, .. }` produced by a stage is re-emitted as the stage's fail-fast `Err((fatal, warnings))` tuple — the warning that triggered escalation lands in the `fatal` slot (kind preserved); warnings observed up to that point ride alongside in the `warnings` slot. Variant identity of the underlying `*ErrorKind` is preserved across the escalation: pattern-matching on the kind variant remains a stable predicate for machine triage.

- **`Strict`.** Same as `FailOnWarning`. There is no separate `Severity::Info` to trigger on — `Severity` was reduced to `{Error, Warning}` per `30 §5.2` and informational signals moved to `tracing`. Retained as a name for forward compatibility (it currently behaves identically to `FailOnWarning`); a future `Severity::Note` variant could differentiate.

### 5.4 Where the policy is applied

Warning-policy application is the LAST step on each stage's return path inside `SemStrait`: the stage-specific call returns its `Result<Output, StageErrors>`; `SemStrait` inspects the warnings slot; if the policy escalates, re-wraps into the stage's error carrier and returns `Err`; otherwise passes the original `Result` through unchanged. Stage implementations are unaware of `WarningPolicy` — the policy is strictly a boundary concern at `semstrait-api`.

### 5.5 Interaction with `compile_and_plan_and_adapt`

In the fused helper (`§7`), `WarningPolicy` applies at every stage boundary. If compile, plan, or adapt produces an escalation-eligible diagnostic, the helper returns `Err((Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>))` — the originating stage's typed-kind is wrapped into `SemStraitErrorKind` via `From` (§6.4); warnings from prior stages are preserved in the `Diagnostics<SemStraitErrorKind>` slot in source-stage order. All stages clean → `Ok((artifact, w1 ++ w2 ++ w3))` concatenated in source-stage order.

### 5.6 Invariants

No policy changes a diagnostic's kind variant, severity, or rendered message. `FailOnWarning` escalating a `CompileError::SchemaInferenceClamped` (Severity::Warning) into the fail-fast `fatal` slot keeps the `kind` variant unchanged; the only difference is the slot it occupies in the `Result` tuple. Pattern-matching on the underlying `*ErrorKind` variant remains a stable predicate for distinguishing escalated warnings from intrinsic errors — consumers may additionally inspect `Diagnose::severity()` to confirm the original severity was Warning.

---

## 6. `SemStraitErrorKind`

> **Migration note.** Prior drafts of this document used a `SemStraitError`
> enum that wrapped per-stage *Errors* carriers (`ParseErrors`, `ValidateErrors`,
> `CompileErrors`, `PlanErrors`, `AdaptError`, `RepositoryError`, `CatalogError`).
> Those carriers and the wrapping enum are **retired** as part of the
> workspace-wide typed-kind transition (`30 §5` / `31 §3`). The replacement
> is `SemStraitErrorKind`: a typed-kind enum whose variants carry the
> upstream `*ErrorKind` enums directly, with `From` impls for ergonomic `?`
> propagation. The unified shape is keyed by variant identity, not by a
> numeric subsystem prefix; consumers route on variant matching.

### 6.1 Purpose

`SemStraitErrorKind` is the unified typed-error kind for every
`SemStrait` method whose body crosses two or more stage boundaries —
`compile_from_yaml`, `compile_from_model`, `plan` (planner + optimizer),
`compile_and_plan_and_adapt`, `save_manifest`, `load_manifest`,
`validate_manifest`. The single-stage delegate `SemStrait::adapt`
preserves the native `AdaptErrorKind` so callers using just adapt see
the same surface they would using `semstrait-adapter` directly.

This **multi-stage methods unify; single-stage methods passthrough**
rule is the §3.3 method-roster contract. The `From` impls in §6.3 are
the wrapping mechanism — every wrapping site is a single `?` at a stage
boundary inside the orchestrator. No per-stage helper carriers
(retired with the migration note above) and no upstream `*ErrorKind`
mutation: each stage's kind stays owned by its crate, and `38` only
adds a sum on top.

### 6.2 Enum shape

```rust
/// Unified typed-error kind for `semstrait-api` orchestration entry points.
/// Identification by variant identity per `30 §5`. Variants wrap upstream
/// `*ErrorKind` enums; severity propagates through `Diagnose::severity()`
/// of the inner kind, except for the configuration variants
/// (`BuilderInvalid`, `NoRepositoryConfigured`) which are intrinsic to `38`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SemStraitErrorKind {
    // -- Stage kinds (wrappers over upstream *ErrorKind) --
    Parse              (ParseErrorKind),
    Validate           (ValidateError),
    Compile            (CompileError),
    Plan               (PlanErrorKind),
    Optimize           (OptimizeErrorKind),
    Adapt              (AdaptErrorKind),

    // -- Transport kinds (also wrappers; surfaced at I11b call sites) --
    Repository         (RepositoryErrorKind),
    CatalogProvider    (CatalogProviderErrorKind),
    FileSystem         (FileSystemErrorKind),

    // -- Configuration kinds (intrinsic to `38`) --
    /// `SemStraitBuilder::build` was called without a required field;
    /// `missing` is the field name (matching the chain-method name minus
    /// `with_`).
    BuilderInvalid     { missing: &'static str },

    /// A repository-bound method (`save_manifest` / `load_manifest`) was
    /// called on a `SemStrait` whose builder did not configure a
    /// `Repository`.
    NoRepositoryConfigured,
}
```

The `Diagnose` impl follows.

```rust
impl semstrait_common::diagnostic::Diagnose for SemStraitErrorKind {
    fn message(&self) -> std::borrow::Cow<'_, str> {
        use SemStraitErrorKind::*;
        match self {
            Parse(k)            => k.message(),
            Validate(k)         => k.message(),
            Compile(k)          => k.message(),
            Plan(k)             => k.message(),
            Optimize(k)         => k.message(),
            Adapt(k)            => k.message(),
            Repository(k)       => k.message(),
            CatalogProvider(k)  => k.message(),
            FileSystem(k)       => k.message(),
            BuilderInvalid { missing } =>
                format!("builder invalid: missing required field {missing}").into(),
            NoRepositoryConfigured =>
                "no repository configured on this SemStrait".into(),
        }
    }
    fn severity(&self) -> semstrait_common::Severity {
        use SemStraitErrorKind::*;
        match self {
            Parse(k)            => k.severity(),
            Validate(k)         => k.severity(),
            Compile(k)          => k.severity(),
            Plan(k)             => k.severity(),
            Optimize(k)         => k.severity(),
            Adapt(k)            => k.severity(),
            Repository(k)       => k.severity(),
            CatalogProvider(k)  => k.severity(),
            FileSystem(k)       => k.severity(),
            BuilderInvalid { .. } | NoRepositoryConfigured =>
                semstrait_common::Severity::Error,
        }
    }
}
```

### 6.3 `From` impls

```rust
impl From<ParseErrorKind>            for SemStraitErrorKind { /* Parse(k)            */ }
impl From<ValidateError>         for SemStraitErrorKind { /* Validate(k)         */ }
impl From<CompileError>          for SemStraitErrorKind { /* Compile(k)          */ }
impl From<PlanErrorKind>             for SemStraitErrorKind { /* Plan(k)             */ }
impl From<OptimizeErrorKind>         for SemStraitErrorKind { /* Optimize(k)         */ }
impl From<AdaptErrorKind>            for SemStraitErrorKind { /* Adapt(k)            */ }
impl From<RepositoryErrorKind>       for SemStraitErrorKind { /* Repository(k)       */ }
impl From<CatalogProviderErrorKind>  for SemStraitErrorKind { /* CatalogProvider(k)  */ }
impl From<FileSystemErrorKind>       for SemStraitErrorKind { /* FileSystem(k)       */ }
```

The fused helper relies on these impls so per-stage `?`-propagation lands
on the right `SemStraitErrorKind` variant without explicit conversion at
each call site. Each `From` is **lossless**: variant identity of the inner
kind is preserved; consumers that need the inner kind for fine-grained
matching destructure `SemStraitErrorKind::Compile(CompileError::…)`.

### 6.4 Variant-to-origin map

<!-- cross-cut: none -->

| Variant                  | Origin doc | Inner `*ErrorKind`            | Notes |
|--------------------------|------------|-------------------------------|-------|
| `Parse`                  | `32`       | `ParseErrorKind`              | YAML / expression-DSL parse errors |
| `Validate`               | `32`       | `ValidateError`           | structural-preconditions failures |
| `Compile`                | `33`       | `CompileError`            | name / catalog / schema / binding resolution |
| `Plan`                   | `34`       | `PlanErrorKind`               | planner failures |
| `Optimize`               | `34`       | `OptimizeErrorKind`           | optimizer-pass failures |
| `Adapt`                  | `36`       | `AdaptErrorKind`              | adapter / emission failures |
| `Repository`             | `33`       | `RepositoryErrorKind`         | persistence I/O |
| `CatalogProvider`        | `37`       | `CatalogProviderErrorKind`    | catalog-provider transport / drift |
| `FileSystem`             | `37`       | `FileSystemErrorKind`         | filesystem transport |
| `BuilderInvalid`         | `38`       | (none — configuration)        | builder missing required field |
| `NoRepositoryConfigured` | `38`       | (none — configuration)        | I11b call without `Repository` injected |

A caller writing a log-router pattern-matches first on the outer
`SemStraitErrorKind` variant (origin) and then, where needed, on the inner
`*ErrorKind` variant (specific failure). Neither alone is sufficient.

### 6.5 `Diagnose` impl semantics

`SemStraitErrorKind` implements `Diagnose` per `31 §3`. The wrapper variants
delegate `message()` and `severity()` to the inner kind so the rendered
diagnostic is identical regardless of whether a caller observes the inner
kind directly (per-stage method) or unified through `SemStraitErrorKind`
(fused helper). Configuration variants (`BuilderInvalid`,
`NoRepositoryConfigured`) render their own messages and report
`Severity::Error`.

The retired `IntoDiagnostic` trait of earlier drafts is not used; the
typed-kind discipline emits `Diagnostic<K>` directly via the construction
helpers in `31 §3.4` (e.g. `Diagnostic::error(kind)`,
`Diagnostic::warning(kind).with_location(loc)`).

### 6.6 `StageOrigin`

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageOrigin {
    Parse, Validate, Compile, Plan, Optimize, Adapt,
    Repository, CatalogProvider, FileSystem, Builder,
}

impl SemStraitErrorKind {
    pub fn origin(&self) -> StageOrigin;
}
```

Provided so callers can branch on origin without pattern-matching the
whole enum (typical use: error messaging, log routing). `Optimize` /
`CatalogProvider` / `FileSystem` are added to match the variant set above.

### 6.7 Stability

`SemStraitErrorKind` and `StageOrigin` are `#[non_exhaustive]`; variant
additions are MINOR (`30 §2.1`). Inner kind shapes inherit stability from
their owning docs (renames MAJOR; field additions on `#[non_exhaustive]`
variant payloads MINOR). `BuilderInvalid.missing` carries a `&'static str`
matching a field name in `§4.3`; a new required field without default is
MAJOR; with a default is MINOR.

---

## Testing Strategy

- SemStraitBuilder required-field validation (missing catalog_provider or file_system returns BuilderInvalid)
- Chain-method ergonomics (with_* consume-self return-modified-self)
- Default::default returns SemStraitBuilder with defaults per §4.3
- WarningPolicy escalation rules (Accumulate passthrough; FailOnWarning escalate-first-warning-to-fatal; Strict alias)
- WarningPolicy invariants (kind-variant-preserved, severity-preserved)
- SemStraitErrorKind From<*ErrorKind> lossless wrapping (variant identity preserved)
- SemStraitErrorKind::Diagnose impl delegation (message() and severity() delegate to inner kind)
- StageOrigin helper (origin() branch on variant)
- #[non_exhaustive] MINOR-additive (variant additions MINOR per 30 §2.1)
- const fn factories (SemStraitBuilder::new const fn)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
