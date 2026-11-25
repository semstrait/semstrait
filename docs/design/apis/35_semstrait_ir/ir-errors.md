---
id: 35-ir-errors
status: ratified
version: 2.1
impl: implemented
---

# IR Errors

The error kinds owned by `semstrait-ir`. Three categories: (1) `ValidateError` — imported from `semstrait-common`, raised by the `Tree` family it serves; (2) `CompileError` — owned here, raised by function resolution + `ReturnTypeRule::Custom` callbacks; (3) `IrErrorKind` — owned here, raised by plan-tree construction, validation, and artifact serialization.

## `ValidateError` — Imported from `semstrait-common`

```rust
use semstrait_common::ValidateError;  // tree-machinery error (item Y / V5)
```

The construction-boundary diagnostic raised by `Tree::with_new_children` and `Rewriter<N>::f_down` / `f_up` is **owned by `semstrait-common`** (item Y / V5), co-located with the `Tree` family it serves — it cannot stay above the trait whose return type it is, and it is primitive-only (`usize` / `String` / `&'static str`; no `DataType` / `CanonicalFn` / `Name` payload), so it sits cleanly in the DAG leaf. Body + variants (`StructuralArityMismatch`, `AggregateInScalarContext`, `RewriteInvariantViolated`) + `Diagnose` impl at `31 §3.1`, shape ratified by `14 §3.1`. `35` imports it as the return type of its `impl Tree for PhysicalExpr` / `impl Tree for PlanNode`; it is **no longer an ir-owned error**.

Downstream stages embed via D.ii kind-nesting (`30 §7.4`): `model::ValidateError` carries a `Tree(common::ValidateError)` variant for invariants raised during parse-time tree construction (the `ir::ValidateError` intermediary is gone — `model → common` is the only hop, item Y / V1 / V5).

## `CompileError` — Function Resolution + `FunctionSpec` Machinery

```rust
/// Function-resolution diagnostic raised while a `FunctionCall` is resolved
/// against the sealed registry, by `ReturnTypeRule::Custom` callbacks wired
/// into `FunctionSpec`, and by registry self-consistency checks.
/// Hosts the `14a §8` call-resolution roster. Per `14a §8` / `14a §3.5`.
///
/// Identification is by variant identity per `30 §5.4`.
#[non_exhaustive]
pub enum CompileError {
    /// `FunctionCall.name` not found in the sealed registry (`14a §2.1`).
    UnknownFunction { name: CanonicalFn },

    /// Call-site arity outside the function's declared range; `expected`
    /// renders the accepted arity, `got` is the actual argument count.
    FunctionArityMismatch { name: CanonicalFn, expected: String, got: usize },

    /// No registered overload matched the call-site argument types
    /// (`14 §5.4`: exact per-arg match, no implicit coercion — author
    /// must `CAST`). `tried_signatures` pre-renders the full overload set.
    NoMatchingSignature { name: CanonicalFn, arg_types: Vec<DataType>, tried_signatures: Vec<String> },

    /// `ReturnTypeRule::Custom` callback declined to produce a return
    /// type for the supplied argument types.
    CustomRuleRejected { fn_name: CanonicalFn, args: Vec<DataType>, reason: String },

    /// A registered `FunctionSpec` failed its own internal consistency
    /// check (e.g. signature overlaps another in the same registry).
    SpecInconsistent   { fn_name: CanonicalFn, reason: String },
}

impl Diagnose for CompileError { /* per-variant message / severity */ }
```

`CompileError` is the function-resolution diagnostic — it hosts the `14a §8` call-resolution roster (`UnknownFunction`, `FunctionArityMismatch`, `NoMatchingSignature`) alongside the return-type-rule and registry-consistency faults. These are *raised* downstream in the manifest compile pipeline (resolution runs there per `19 §3.6`, not in this crate) and embed into `manifest::CompileError` as `Ir(ir::CompileError)` via D.ii kind-nesting (`30 §7.4`). The wider *non-function* compile surface — unknown semantic references, ambiguous relationship paths, reference cycles, boundary type-inference failures — lives in `semstrait-manifest::CompileError` per `33`'s compile/error boundary, not here.

The adapter-registration collision roster (`14a §8`: `ReservedTagCollision`, `AdapterFunctionShadowsCore`, `AdapterFunctionCollision`) is **not** carried here — those surface as panics at `function_registry()` initialization (`14a §7`), not as recoverable compile diagnostics.

## `IrErrorKind`

```rust
/// Typed-kind enum for `semstrait-ir`'s own operations: plan-tree
/// construction (`Name` validation), plan walking (`transform`
/// failures), plan validation (`validate`), and adapter-artifact
/// serialization (`EnginePlan::to_bytes` / `to_json`).
///
/// Per `30 §5`. Identification is by variant identity (`matches!`);
/// there is no string-code accessor.
#[non_exhaustive]
pub enum IrErrorKind {
    /// `Name::new` was called with an empty or invalid identifier.
    InvalidName        { supplied: String, reason: String },

    /// `Name::new` was called with a reserved plan-tree prefix (§11.4.1
    /// in [ir-plan-primitives.md](ir-plan-primitives.md)).
    ReservedName       { supplied: String, prefix: String },

    /// A `PhysicalExpr` reaching the plan tree lacks `inferred_type`.
    /// Only reported by `SemanticPlan::validate`.
    UnresolvedType     { location: String, expr_sketch: String },

    /// Two `KeyPair` columns have incompatible types on a `JoinNode`.
    /// Only reported by `SemanticPlan::validate`.
    JoinKeyTypeMismatch{ pair: KeyPair, left_ty: DataType, right_ty: DataType },

    /// `AggNode.aggregates` contains a duplicate output name.
    /// Only reported by `SemanticPlan::validate`.
    DuplicateAggName   { name: Name },

    /// `FilterNode.predicate` has non-Boolean type.
    /// Only reported by `SemanticPlan::validate`.
    FilterPredicateNotBoolean { actual: DataType },

    /// A `PlanNode`'s `meta.output_schema` disagrees with the schema
    /// computed from its children. Only reported by
    /// `SemanticPlan::validate`.
    SchemaMismatch     { node_kind: &'static str, expected: String, got: String },

    /// `UnionNode.inputs` schemas are not structurally compatible.
    UnionSchemaMismatch{ input_ix: usize, expected: String, got: String },

    /// `UnionNode.inputs.len() < 2`.
    UnionArityTooLow   { arity: usize },
```

Remaining `IrErrorKind` variants covering column-reference resolution, range-validation failures, and visitor machinery errors.

```rust
    /// A `Name` referenced by `group_by` / `order` / `on` does not
    /// resolve to a column in the input schema.
    UnresolvedColumnRef{ name: Name, available: Vec<Name> },

    /// A `FetchNode.limit` / `FetchNode.offset` value is out of the
    /// adapter's representable range (typically `i64::MAX` for Substrait).
    FetchValueOutOfRange { field: &'static str, value: u64 },

    /// A `transform` / `walk` callback returned an error.
    TransformFailure   { reason: String },

    /// `EnginePlan::to_bytes` / `to_json` failed; wraps the underlying
    /// `prost::EncodeError` / `serde_json::Error` context as a string.
    ArtifactSerializationFailed { reason: String },

    /// A visitor-side invariant was violated (e.g. a transform produced
    /// a structurally invalid subtree that immediate post-check caught).
    TransformInvariantViolated  { reason: String },
}
```

The `Diagnose` trait provides human-readable messages and severity classification for all `IrErrorKind` variants.

```rust
impl Diagnose for IrErrorKind {
    fn message(&self) -> String { /* per-variant human text */ }
    fn severity_default(&self) -> Severity { Severity::Error }
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}
```

`IrErrorKind` is owned by `semstrait-ir`. It is distinct from `ir::CompileError` (above, raised by `FunctionSpec` machinery), the imported `common::ValidateError` (raised by the `Tree` / `Rewriter` machinery `ir::PhysicalExpr` impls — owned by `semstrait-common` per item Y / V5), and the downstream `manifest::CompileError` / `PlanErrorKind` / `AdaptErrorKind` — each has a different production site and a different lifecycle. A planner-side failure producing a malformed `SemanticPlan` is a `PlanErrorKind` (`34`); that same malformed plan caught by `SemanticPlan::validate()` on the consumer side becomes an `IrErrorKind`. All four surface as `Diagnostic<K>` envelopes for the appropriate `K` per `30 §5.1`.

## Variant Identity, Not Codes

The retired `IR_E_*` numeric range from earlier drafts is gone. Identification is by variant identity per `30 §5.4`; renaming or removing a variant is MAJOR per `30 §2.1`; adding a variant inside `#[non_exhaustive]` is MINOR per `30 §2.2`. The error module co-locates the ir-owned `CompileError` and `IrErrorKind` next to their production sites; the tree-machinery `ValidateError` is imported from `semstrait-common` (item Y / V5), not declared here.

## Warning Posture

`semstrait-ir` itself, being pure data + validation, has no warning-emitting operation in v1. Warnings surfaced by planner or optimizer ride `PlanErrorKind` / `OptimizeErrorKind` envelopes per `34`; adapter warnings ride `AdaptErrorKind` per `36`. If a future v2 walker emits `Severity::Warning` `Diagnostic<IrErrorKind>` (e.g. an "unused PlanNode" advisory), it lands additively under `#[non_exhaustive]` per `30 §2.2`.

## Testing Strategy

This leaf owns `CompileError` and `IrErrorKind`. Tests:

- **`CompileError` variant coverage:** Unit tests verify each variant (`UnknownFunction`, `FunctionArityMismatch`, `NoMatchingSignature`, `CustomRuleRejected`, `SpecInconsistent`) is raised for the corresponding function-resolution failure (see [ir-function-registry.md](ir-function-registry.md)).
- **`IrErrorKind` variant coverage:** Unit tests verify each validation-failure variant (`InvalidName`, `ReservedName`, `UnresolvedType`, `JoinKeyTypeMismatch`, `DuplicateAggName`, `FilterPredicateNotBoolean`, `SchemaMismatch`, `UnionSchemaMismatch`, `UnionArityTooLow`, `UnresolvedColumnRef`, `FetchValueOutOfRange`) is raised by `SemanticPlan::validate()` for the corresponding invariant violation (see [ir-plan-nodes.md](ir-plan-nodes.md) §13).
- **`TransformFailure` / `TransformInvariantViolated`:** Unit tests verify these are raised when a `transform` callback returns an error or produces a structurally invalid subtree.
- **`ArtifactSerializationFailed`:** Unit tests verify `EnginePlan::to_bytes` / `to_json` raise this variant when the underlying `prost::EncodeError` / `serde_json::Error` occurs (see [ir-adapter-artifacts.md](ir-adapter-artifacts.md)).
- **`ValidateError` import correctness:** Integration tests verify `impl Tree for PhysicalExpr` / `impl Tree for PlanNode` return `ValidateError` (imported from `semstrait-common`, not declared here) when `with_new_children` / rewriter callbacks fail.
- **Variant identity identification:** Unit tests verify `matches!` works on every variant (no string-code accessor exists per `30 §5.4`).
- **`Diagnose` impl correctness:** Unit tests verify each variant's `message()` method returns human-readable text; `severity_default()` returns `Severity::Error` for all variants.

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
