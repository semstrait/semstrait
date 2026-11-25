---
id: 30-error-emission-policy
status: ratified
version: 2.1
impl: implemented
---

# Error Emission Policy

This leaf owns the per-stage error-emission policy at API boundaries — accumulating vs fail-fast, return shape per stage, and warnings never silently dropped.

## 7. Error-Emission Policy per Stage

The per-stage policy is ratified in `10 §5`; `30` carries it forward to the crate-public-surface level. Two patterns bind every public stage entry-point:

- **Accumulating** stages process the whole input and collect every independent finding into one vector before deciding success vs failure. Used where a single pass can find many independent errors (multi-document YAML, structural well-formedness across the whole model).
- **Fail-fast** stages stop at the first hard inconsistency and return immediately. Used where downstream work cannot meaningfully proceed past the first failure (compile, plan, adapt — sequential transformations).

Both patterns carry warnings (`Severity::Warning` diagnostics) on the success arm. Fail-fast stages additionally preserve any warnings encountered before the fatal error on the failure arm — warnings are never silently dropped (§7.3).

### 7.1 Per-stage classification and return shape

The accumulating-vs-fail-fast classification is fixed at the spec level — switching a stage between classes is a MAJOR change.

| Stage | Crate | Class | Public-API return shape |
|---|---|---|---|
| `parse` | `32` (model) | accumulating | `Result<(SemanticModel, Diagnostics<ParseErrorKind>), Diagnostics<ParseErrorKind>>` |
| `validate` | `32` (model) | accumulating | `Result<Diagnostics<ValidateError>, Diagnostics<ValidateError>>` |
| `compile` | `33` (manifest) | fail-fast | `Result<(SemanticManifest, Diagnostics<CompileError>), (Diagnostic<CompileError>, Diagnostics<CompileError>)>` |
| `plan` | `34` (planner) | fail-fast | `Result<(SemanticPlan, Diagnostics<PlanErrorKind>), (Diagnostic<PlanErrorKind>, Diagnostics<PlanErrorKind>)>` |
| `optimize` | `34` (planner) | fail-fast | `Result<(SemanticPlan, Diagnostics<OptimizeErrorKind>), (Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>)>` |
| `adapt` | `36` (adapter) | fail-fast | `Result<(EngineArtifact, Diagnostics<AdaptErrorKind>), (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>)>` |
| `CatalogProvider::resolve` | `37` (catalog) | fail-fast | `Result<(ResolvedSource, Diagnostics<CatalogProviderErrorKind>), (Diagnostic<CatalogProviderErrorKind>, Diagnostics<CatalogProviderErrorKind>)>` |
| `FileSystem::read` | `37` (catalog) | fail-fast | `Result<(Bytes, Diagnostics<FileSystemErrorKind>), (Diagnostic<FileSystemErrorKind>, Diagnostics<FileSystemErrorKind>)>` |
| `Repository::{load,save}` | `33` (manifest) | fail-fast | analogous to `compile`, with `RepositoryErrorKind` |
| Fused helpers (`compile_and_plan_and_adapt`, …) | `38` (api) | fail-fast | as their owning stages, with `SemStraitErrorKind` (§5.6) as the kind |

**`validate` special case.** Because `validate` produces no value on success, the success arm carries only the warnings vector (no tuple wrapper). This is the one exception to the uniform `(Output, Diagnostics<K>)` Ok shape.

### 7.2 Pattern shapes — stated abstractly

```rust
// Accumulating stage:
//   Ok  = (Output, Diagnostics<K>)   — Output + warnings collected during the pass
//   Err = Diagnostics<K>             — every diagnostic collected (errors + any warnings)
fn accumulating_stage(input: Input)
    -> Result<(Output, Diagnostics<K>), Diagnostics<K>>;

// Fail-fast stage:
//   Ok  = (Output, Diagnostics<K>)   — Output + warnings emitted before completion
//   Err = (Diagnostic<K>, Diagnostics<K>)
//                                    — the fatal diagnostic + any warnings preceding it
fn fail_fast_stage(input: Input)
    -> Result<(Output, Diagnostics<K>), (Diagnostic<K>, Diagnostics<K>)>;
```

The `Diagnostics<K>` warnings vector on the success arm contains **only `Severity::Warning`** diagnostics (errors would have routed the call to the Err arm). The `Diagnostics<K>` on an accumulating stage's Err arm is **mixed** (errors + warnings). The `Diagnostics<K>` on a fail-fast stage's Err-tuple second element is **warnings only** (the single fatal error lives in the tuple's first element).

### 7.3 Warnings are never silently dropped

Both arms of every stage signature carry warnings. A library implementation that accumulates warnings internally and discards them when raising an Err is a bug. Conversion helpers (`?` propagation, `From` impls) MUST preserve the warnings vector across stage boundaries.

### 7.4 Stage-ownership of kinds

A diagnostic's stage of origin is determined by the kind type (`ParseErrorKind`, `CompileError`, …), not by a substring of a code. Cross-crate kind-nesting is permitted (`CompileError::Parse(ParseErrorKind)` when compile internally re-parses a fragment); see §5.6 for the canonical aggregation pattern at fused helpers.

### 7.5 Accumulation limits

`parse` and `validate` accumulate without an intrinsic limit at the `30` level — every independent error is reported. In practice, large malformed models can produce hundreds of errors; per-crate docs (`32`) MAY introduce a soft cap with a sentinel kind variant ("further errors suppressed") to protect caller UX. No cap is imposed at the `30` level.

### 7.6 Panic-freedom

Public entry points never panic on caller-reachable input. Internal `unreachable!()` / `panic!("invariant ...")` calls are permitted only where the invariant is genuinely impossible to violate without a semstrait bug (e.g. a `SemanticManifest` field the compile pass is sworn to populate). A caller-reachable panic is a semstrait bug and is fixed as such.

## Testing Strategy

- Per-stage classification fixed to table in §7.1 (parse/validate accumulating, others fail-fast)
- Return shape matches §7.1 table for every public stage entry-point
- Warnings preserved on both success and failure arms (§7.3 no silent drop)
- No caller-reachable panics (§7.6 panic-freedom on all pub entry points)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
