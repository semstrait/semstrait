---
id: 34-planner-optimize
status: ratified
version: 2.1
impl: in-progress
---

# Optimize Function and OptimizerPass Trait

This leaf owns the optimize() free function signature, OptimizerPass trait surface, Optimizer/OptimizerBuilder, canonical v1 passes (ConstantFolding/MetadataDimensionSubstitution/PredicateSimplification/IdentityProjectElimination), pass ordering, OptimizePassContext, OptimizeErrorKind, externally-implementable pass trait.

## 11. The `optimize` function

### 11.1 Signature

```rust
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};

pub fn optimize(
    plan: SemanticPlan,
) -> Result<
    (SemanticPlan, Diagnostics<OptimizeErrorKind>),
    (Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>),
>;
```

Optimizer entry point per `10 §3.5`. Consumes a `SemanticPlan` and returns an equivalent plan (same observable results) with canonical-form rewrites applied, plus warnings on the success arm. Sync (I6), no I/O (I11), fail-fast per `30 §7.1`. Return-shape mirrors `30 §7.2`'s fail-fast pattern: success `(SemanticPlan, Diagnostics<OptimizeErrorKind>)`, failure `(Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>)`. Passes are in-place rewrites over the tree (`PlanNode::transform` per `35 §9`); ownership is consumed because retaining the caller's copy has no utility.

The free function applies `Optimizer::with_v1_passes()`, bundling the four canonical passes (§11.2). A caller wishing to skip optimization simply does not call `optimize` — there is no bypass argument. Callers composing custom pass chains use `OptimizerBuilder` (§12.5) and `Optimizer::apply` directly.

### 11.2 Canonical v1 passes


| Pass | Name                            | Purpose                                                                                                                                   | Failure variant                                                                | Section |
| ---- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------- |
| 1    | `ConstantFolding`               | Fold constant `PhysicalExpr` subtrees into `Expr::Literal` leaves. Operates on predicates, Project expressions, and Agg expressions.      | `OptimizeErrorKind::PassFailed { pass: "constant_folding", … }`                | §11.3   |
| 2    | `MetadataDimensionSubstitution` | Substitute metadata-source Dimensions (per `13 §4.7` / SR-10) with their declared metadata expression.                                    | `OptimizeErrorKind::PassFailed { pass: "metadata_dimension_substitution", … }` | §11.4   |
| 3    | `PredicateSimplification`       | Simplify predicates: `true AND x` → `x`, `false OR x` → `x`, `NOT NOT x` → `x`, range fusion (`x >= a AND x <= b` → `x BETWEEN a AND b`). | `OptimizeErrorKind::PassFailed { pass: "predicate_simplification", … }`        | §11.5   |
| 4    | `IdentityProjectElimination`    | Remove `PlanNode::Project` nodes whose projection list is a 1:1 identity on the input schema.                                             | `OptimizeErrorKind::PassFailed { pass: "identity_project_elimination", … }`    | §11.6   |


Every pass is deterministic, shape-preserving, and pure over the plan tree. No pass introduces or removes a `PlanNode` variant — the tree's variant distribution is stable under the v1 pass chain. (Pass 4 removes a `Project` node, which is variant-count change but not variant-introduction.)

### 11.3 `ConstantFolding`

**Algorithm.** Post-order walk over every `PhysicalExpr` the plan carries. Fold `BinaryOp`, `Negate` / `Not`, `Cast`, `Coalesce`, `NullIf`, and `Case` nodes whose operands are all `Literal` into a single `Literal` per the semantics in `31 §3.1` / `13 §4`. Non-constant subtrees pass through unchanged. Overflow, precision loss, or divide-by-zero is non-fatal — the pass returns the original expression and emits a `Severity::Warning` `Diagnostic<OptimizeErrorKind>` with kind `ConstantFoldSkipped { reason }`.

**Interaction with `SessionContext`.** The pass does NOT fold session-dependent functions (`NOW()`, `CURRENT_DATE()`) even though `session.now` is available — folding them would prevent planner-level caching of the `SemanticPlan` across invocations with different `SessionContext.now`. Session-dependent folds happen at adapter time per `36`.

### 11.4 `MetadataDimensionSubstitution`

**Algorithm.** Per `13 §4.7` (SR-10), a Dimension may be declared `metadata_source: true`, meaning its runtime value is pinned at plan time rather than projected from a Binding. The pass walks every `PhysicalExpr` in the plan and substitutes every `Column { name }` referring to a metadata Dimension with the Dimension's declared metadata expression (typically a `Literal`). The substitution source is `manifest.metadata_index: BTreeMap<SemanticsName, PhysicalExpr>`, populated by `compile` per `19 §3.2.3`. The pass short-circuits if `metadata_index.is_empty()`.

Lifting the substitution to a uniform pass keeps per-strategy code free of metadata-aware bookkeeping.

### 11.5 `PredicateSimplification`

**Algorithm.** Post-order walk over every `PlanNode::Filter.predicate` and every `PhysicalExpr` child of Agg / Project nodes, applying a fixed rewrite-rule set: `Literal(Bool(true)) AND x` → `x`; `Literal(Bool(false)) AND x` → `false` (and symmetric `OR` rules); `NOT NOT x` → `x`; `NOT Literal(Bool(b))` → `Literal(Bool(!b))`; `x IS NULL AND x IS NULL` → `x IS NULL`; range fusion (`x >= a AND x <= b` → `x BETWEEN a AND b`) for same-type literal bounds; De Morgan only when it strictly shortens the expression. Every rewrite strictly shrinks the expression, so a single pass converges.

### 11.6 `IdentityProjectElimination`

**Algorithm.** Walk the tree. For every `PlanNode::Project { input, expressions, .. }`, replace the node with `input` if and only if (a) every expression is `PhysicalExpr::Column { name }`, (b) the name sequence equals the input's column-name sequence exactly, and (c) the output `Name` for each expression equals the column name (no rename). O(n) in node count.

The outermost Request-level `Project` emitted at step 5 (§7.6 item 4) pins `SemanticPlan.output_names` — the identity check includes the output-order constraint and the Project survives unless its input already presents the required order.

### 11.7 Pass ordering

Passes run in order 1 → 2 → 3 → 4. Pass 2 must run before pass 3 so that metadata-pinned predicates become literals pass 3 can eliminate; pass 1 must run before pass 3 for the same reason; pass 4 must run last because earlier passes MAY create identity-shaped `Project` nodes. Third-party passes MAY be inserted at any position via `OptimizerBuilder::with(...)` (§12.5); the v1 built-in order is fixed.

## 12. `OptimizerPass` trait

### 12.1 Trait surface

```rust
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};

pub trait OptimizerPass: Send + Sync {
    fn name(&self) -> &str;
    fn apply(
        &self,
        plan: SemanticPlan,
        ctx: &OptimizePassContext,
    ) -> Result<
        (SemanticPlan, Diagnostics<OptimizeErrorKind>),
        (Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>),
    >;
    fn is_applicable(&self, _plan: &SemanticPlan) -> bool { true }
}
```

Passes are pure, sync, deterministic (`10 §3.5`). `name` is stable per pass for diagnostics (`"constant_folding"`, `"identity_project"`, …); `apply` returns the rewritten plan plus warnings on success or a fail-fast `(Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>)` pair on failure (per `30 §7.2`); `is_applicable` allows skipping a pass silently (default: always applicable). A pass that cannot produce an equivalent plan (e.g. an adapter-specific rewrite requiring dialect information) MUST NOT implement `OptimizerPass` — it belongs in the adapter's `adapt` stage per `36`.

### 12.2 `OptimizePassContext`

```rust
pub struct OptimizePassContext<'a> {
    pub manifest:    Option<&'a SemanticManifest>,
    pub session:     Option<&'a SessionContext>,
    pub diagnostics: &'a DiagnosticSink,
}
```

`manifest` and `session` are `Option` because a `SemanticPlan` passed through `optimize` does not necessarily carry its producing SemanticManifest / SessionContext — a plan deserialized from a wire form (`35 §3.3`) may arrive without either. Passes that require them MUST emit a fatal `Diagnostic<OptimizeErrorKind>` whose kind is `PassRequiresContext { pass, required }` when invoked without the needed reference.

### 12.3 Externally-implementable

`OptimizerPass` is **non-sealed** per `30 §4.6`: new passes are a primary extensibility vector. Adapter crates in particular MAY expose plan-level passes for engine-independent rewrites they've identified. Soundness (semantic equivalence) cannot be statically enforced by the trait; it relies on the author's discipline. A test harness in `semstrait-planner::tests::optimize` cross-checks every built-in pass against a golden-plan corpus.

### 12.4 `Optimizer`

```rust
pub struct Optimizer { passes: Vec<Box<dyn OptimizerPass>> }

impl Optimizer {
    pub fn empty() -> Self;
    pub fn with_v1_passes() -> Self;
    pub fn apply(
        &self,
        plan: SemanticPlan,
        ctx: &OptimizePassContext,
    ) -> Result<
        (SemanticPlan, Diagnostics<OptimizeErrorKind>),
        (Diagnostic<OptimizeErrorKind>, Diagnostics<OptimizeErrorKind>),
    >;
    pub fn pass_count(&self) -> usize;
}
```

`apply` runs the passes in registered order, fail-fast on first error; warnings accumulate across passes via the context's diagnostic sink and ride out on the success-tuple's second element or the failure-tuple's second element. `with_v1_passes` bundles the four canonical passes in order.

### 12.5 `OptimizerBuilder`

```rust
pub struct OptimizerBuilder { passes: Vec<Box<dyn OptimizerPass>> }

impl OptimizerBuilder {
    pub fn new() -> Self;
    pub fn with(self, pass: impl OptimizerPass + 'static) -> Self;
    pub fn with_v1_passes(self) -> Self;
    pub fn build(self) -> Optimizer;
}
```

The free function `optimize(plan)` is equivalent to `OptimizerBuilder::new().with_v1_passes().build().apply(plan, &default_ctx)`. Callers wishing to insert custom passes before or after the v1 chain compose explicitly via the builder.

## Testing Strategy

- optimize() signature shape per 30 §7.2 (consumes SemanticPlan; returns equivalent plan + warnings on success; fail-fast on error)
- OptimizerPass trait (name/apply/is_applicable; Send+Sync bounds)
- Optimizer with_v1_passes bundles 4 canonical passes in fixed order (1→2→3→4)
- OptimizerBuilder with/with_v1_passes/build (custom pass insertion before/after v1 chain)
- ConstantFolding (post-order walk; fold BinaryOp/Negate/Not/Cast/Coalesce/NullIf/Case; overflow/precision-loss/divide-by-zero → warning ConstantFoldSkipped; no session-dependent fold)
- MetadataDimensionSubstitution (manifest.metadata_index lookup; Column→Dimension's metadata expression; short-circuit on empty index)
- PredicateSimplification (true AND x → x; false OR x → x; NOT NOT x → x; range fusion x >= a AND x <= b → BETWEEN; De Morgan strictly-shortens-only)
- IdentityProjectElimination (remove identity-shaped Project; outermost Request-level Project survives unless input order matches)
- pass ordering 1→2→3→4 fixed (pass 2/1 before pass 3; pass 4 last)
- fail-fast on first Error; warnings accumulate
- non-sealed trait per 30 §4.6 (adapter-exposed engine-independent rewrites)
- idempotence of re-optimize (v1 canonical passes reach fixed point in single run)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
