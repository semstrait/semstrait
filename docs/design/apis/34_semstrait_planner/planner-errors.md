---
id: 34-planner-errors
status: ratified
version: 2.1
impl: in-progress
---

# PlanErrorKind and OptimizeErrorKind

This leaf owns PlanErrorKind typed-kind enum (step 0-6 variants + per-DataKind passthrough Simple/Grainset/Unionset/Joinset), OptimizeErrorKind typed-kind enum, Diagnose impls, variant identity stability per 30 §5.4, non-error diagnostic emission policy.

## 13. `PlanErrorKind` / `OptimizeErrorKind`

> **Migration note.** Body sections `planner-plan.md §6`–`§10` and the cross-doc-fix table in `§17` retain references to legacy `PLAN_E_`* / `OPT_E_`* codes (e.g. `PLAN_E_0500 ConstraintViolation`). Those codes are **retired** per `30 §5`; the public-API surface identifies errors by `PlanErrorKind` / `OptimizeErrorKind` variant identity. The legacy code prefixes remain in body prose during the migration as cross-reference anchors and will be stripped in a follow-up doc pass. Read `PLAN_E_NNNN VariantName` in the body as shorthand for `PlanErrorKind::VariantName`.

### 13.1 `PlanErrorKind`

```rust
use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_common::{DataType, Location};

/// Typed-kind enum for the `plan` stage. Per `30 §5`. Identification is
/// by variant identity (`matches!`); there is no string-code accessor.
///
/// `#[non_exhaustive]` per I10: adding a variant is MINOR per
/// `30 §2.2`; renaming or removing a variant is MAJOR per `30 §2.1`.
#[non_exhaustive]
pub enum PlanErrorKind {
    // -- step 0: constraint validation (§7.1 / 11 §8.7) --
    ConstraintViolation { entity: String, message: String },

    // -- step 1: request lookup (§7.2) --
    UnknownSemantics         { name: SemanticsName },
    DuplicateRequestedName   { name: SemanticsName },
    EmptyRequest,
    RequestFieldTypeMismatch { name: SemanticsName, placed_in: SemanticElement, resolved: SemanticElement },
    OrderByUnknownName       { name: Name },
    FilterArityMismatch      { field: SemanticsName, operator: FilterOperator, expected: usize, got: usize },
    FilterValueTypeMismatch  { field: SemanticsName, resolved_type: DataType, value: String },
```

The enum continues with the routing, dispatch, per-DataKind passthrough, and temporal variant groups:

```rust
    // -- step 2/3: dataset routing and relationship traversal (§7.3 / §7.4) --
    // Payload shapes match `16 §14.3`'s canonical `PlannerError` table.
    AmbiguousImplicitComposition  { constituent_set: Vec<DataKindRef>, candidates: Vec<DataKindRef> },
    NoCompositionPath             { from: DataKindRef, to: DataKindRef },
    CompositionDepthExceeded      { from_kinds: Vec<DataKindRef>, max_depth: usize },
    CrossCompositionForbidden     { relationship_id: EntityId, attempted_direction: String },
    AmbiguousCompositionReference { name: SemanticsName, candidates: Vec<DataKindRef> },
    CompositionAggregationConflict { name: SemanticsName, aggregations: Vec<String> },
    SemanticsNotOnSurface         { name: SemanticsName, surface: DataKindRef },

    // -- step 4/5: dispatch + construction (§7.5 / §7.6) --
    PlanNodeConstructionFailed       { strategy: StrategyId, reason: String },
    UnsupportedRequestShape          { reason: String },
    PostConstructionInvariantViolated { underlying: String },

    // -- DataKind-specific (§7.5 delegate) --
    DataKindNotInSemanticManifest    { name: DataKindRef },
    StrategyDispatchFailed           { kind_variant: String },
    StrategyMissingForVariant        { kind_variant: String },
    SemanticManifestIndexInconsistent { detail: String },

    // -- per-variant error passthrough (21–24) — kind-nesting per `30 §5.6` --
    Simple   (SimpleErrorKind),
    Grainset (GrainsetErrorKind),
    Unionset (UnionsetErrorKind),
    Joinset  (JoinsetErrorKind),

    // -- temporal (17, DEFERRED) --
    TemporalDeferred { code: TemporalDeferredCode },
}
```

The `impl` blocks for this enum follow:

```rust
impl Diagnose for PlanErrorKind {
    fn message(&self) -> String { /* per-variant human text */ }
    fn severity_default(&self) -> Severity { Severity::Error }
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}

impl From<SimpleErrorKind>   for PlanErrorKind { /* … */ }
impl From<GrainsetErrorKind> for PlanErrorKind { /* … */ }
impl From<UnionsetErrorKind> for PlanErrorKind { /* … */ }
impl From<JoinsetErrorKind>  for PlanErrorKind { /* … */ }
```

**Per-variant location.** The primary source span of an error lives on the wrapping `Diagnostic<PlanErrorKind>` envelope (per `30 §5.1`'s `location: Option<Location>` field), not on the variant. Variants carry semantic payload only.

**Variant rename and identity.** Renaming a variant of `PlanErrorKind` is MAJOR per `30 §2.1` because variant identity is the public-API surface for caller pattern-matching. Adding a variant inside `#[non_exhaustive]` is MINOR per `30 §2.2`.

**Per-DataKind passthrough.** The four `Simple` / `Grainset` / `Unionset` / `Joinset` variants embed the per-DataKind kind enums (`SimpleErrorKind` per `21 §7`, `GrainsetErrorKind` per `22 §8`, `UnionsetErrorKind` per `23 §10`, `JoinsetErrorKind` per `24 §10`) using the cross-crate kind-nesting pattern from `30 §5.6`. Each `From<XxxErrorKind>` impl lifts a per-DataKind kind into the parent `PlanErrorKind` enum.

### 13.2 Non-error diagnostic emission

Strategies emit non-error diagnostics via `ctx.emit(diag)` (§8.4). The per-plan-call `DiagnosticSink` accumulates them in emission order. At step 6 the sink drains into `SemanticPlan.diagnostics` (success) and into the Ok-tuple's `Diagnostics<PlanErrorKind>` second element on success, or the Err-tuple's `Diagnostics<PlanErrorKind>` second element on failure. Warnings flow through both arms — no silent drops (per `30 §7.3`).

### 13.3 `OptimizeErrorKind`

```rust
#[non_exhaustive]
pub enum OptimizeErrorKind {
    PassFailed          { pass: String, reason: String },
    InvalidRewrite      { pass: String, detail: String },
    PassRequiresContext { pass: String, required: &'static str },
    /// Non-fatal ConstantFold result; emitted at `Severity::Warning`.
    /// Carries the reason the pass declined to fold (overflow, precision loss, …).
    ConstantFoldSkipped { reason: String },
}

impl Diagnose for OptimizeErrorKind {
    fn message(&self) -> String { /* per-variant human text */ }
    fn severity_default(&self) -> Severity {
        match self {
            OptimizeErrorKind::ConstantFoldSkipped { .. } => Severity::Warning,
            _ => Severity::Error,
        }
    }
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}
```

Canonical v1 passes are specified to NOT error under well-formed inputs; the `Error`-severity variants fire in practice only for third-party passes with soundness bugs. `ConstantFoldSkipped` is the one v1-emitted `Warning`-severity variant — riding the warnings vector of either Ok or Err arm.

### 13.4 `Diagnose`-conversion

Both `PlanErrorKind` and `OptimizeErrorKind` implement `Diagnose` per `30 §5.4`. The blanket `Display` and `std::error::Error` impls on `Diagnostic<K>` that ride from `Diagnose` (per `31 §10`) make `Diagnostic<PlanErrorKind>` / `Diagnostic<OptimizeErrorKind>` directly usable as `std::error::Error` values without any `IntoDiagnostic`-style conversion. `Diagnostic`s never carry a `code: &'static str` field — identification is the variant tag.

## Testing Strategy

- PlanErrorKind variant roster (step 0-6 variants + per-DataKind passthrough Simple/Grainset/Unionset/Joinset)
- OptimizeErrorKind variant roster (PassFailed/InvalidRewrite/PassRequiresContext/ConstantFoldSkipped)
- Diagnose impls (message per-variant human text; severity_default Error except ConstantFoldSkipped→Warning; cause None)
- From<XxxErrorKind> impl lifts per-DataKind kind into parent PlanErrorKind (Simple/Grainset/Unionset/Joinset)
- per-variant location on Diagnostic<K> envelope per 30 §5.1 (not on variant payload)
- non-error diagnostic emission via ctx.emit accumulates in DiagnosticSink
- warnings flow through both Ok/Err arms per 30 §7.3 (SemanticPlan.diagnostics + Diagnostics<K> second element)
- variant identity frozen at first release (add=MINOR; rename/remove=MAJOR per 30 §2.1)
- #[non_exhaustive] per I10 (adding variant is MINOR per 30 §2.2)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
