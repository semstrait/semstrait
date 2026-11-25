---
id: 34-planner-strategy
status: ratified
version: 2.1
impl: in-progress
---

# Strategy Trait and Built-in Strategies

This leaf owns Strategy trait surface (id/supports/plan), StrategyId enum, StrategyRegistry, StrategyContext, dispatch_strategy function, the four built-in strategies (SimpleStrategy/GrainsetStrategy/UnionsetStrategy/JoinsetStrategy) with algorithm pointers to 21-24, stability (open trait per 30 §4.6).

## 8. The `Strategy` trait

### 8.1 Trait surface

Per `20 §5.2`, the `Strategy` trait is the planner-side contract every per-variant strategy implements. `20 §5.2` sketches the shape; `34` ratifies the concrete signature:

```rust
pub trait Strategy: Send + Sync {
    fn id(&self) -> StrategyId;
    fn supports(&self, kind: &ResolvedDataKind) -> bool;
    fn plan(
        &self,
        ctx: &StrategyContext,
        request: &ResolvedQueryRequest,
        datakind: &ResolvedDataKind,
    ) -> Result<PlanNode, PlanError>;
}
```

`**id`.** Stable identifier for diagnostic / debug / registry use — one of `StrategyId::{Simple, Grainset, Unionset, Joinset}`. Third-party strategies (if Q-PLAN-002 opens the trait) add variants via `#[non_exhaustive]`.

`**supports`.** Called by `dispatch_strategy` as a defense-in-depth check AFTER the variant-tag match has chosen the candidate strategy. Mismatch raises `PLAN_E_2051 StrategyMissingForVariant`. Built-in strategies return `true` for exactly one variant tag.

`**plan`.** Plans the Request against the given DataKind, returning the root `PlanNode` of the subtree (not a full `SemanticPlan` — Request-level wrapping is applied by step 5, §7.6, after the strategy returns). `ctx` carries session, registry for recursive dispatch, and the diagnostic sink; `request` is the resolved form from step 1; `datakind` is the variant this call is planning against (consistent with `request.target`).

**Borrow shapes.** `&StrategyContext` (not `&mut`) — mutation flows through interior-mutable fields so `&dyn Strategy` is shareable across threads. `&ResolvedQueryRequest` — constructed once per `plan` call and consumed read-only. Passing `&ResolvedDataKind` separately from `request.target` lets strategies access variant-specific fields (`ResolvedJoinset.path`, `ResolvedGrainset.levels`) without re-matching.

Per I6, no `async` anywhere in the surface; per I12, every failure returns a typed `PlanError` with a stable `PLAN_E_`* code.

### 8.2 `StrategyId`

```rust
#[non_exhaustive]
pub enum StrategyId { Simple, Grainset, Unionset, Joinset }

impl StrategyId {
    pub fn as_str(&self) -> &'static str; // "SimpleStrategy" …
}
```

Used in `Diagnostic` messages and structured logs only; never in user-facing error codes (those use the stable `PLAN_E_*` constants per §13).

### 8.3 `StrategyRegistry`

```rust
pub struct StrategyRegistry {
    simple:   Box<dyn Strategy>,
    grainset: Box<dyn Strategy>,
    unionset: Box<dyn Strategy>,
    joinset:  Box<dyn Strategy>,
}

impl StrategyRegistry {
    pub fn default_v1() -> Self;
    pub fn simple(&self)   -> &dyn Strategy;
    pub fn grainset(&self) -> &dyn Strategy;
    pub fn unionset(&self) -> &dyn Strategy;
    pub fn joinset(&self)  -> &dyn Strategy;
    pub fn with(
        simple:   impl Strategy + 'static,
        grainset: impl Strategy + 'static,
        unionset: impl Strategy + 'static,
        joinset:  impl Strategy + 'static,
    ) -> Self;
}
```

Holds exactly one strategy per variant; replacement is allowed at construction time only (no mutate-in-place API). Registry is `Send + Sync` — a single instance backs every `plan` invocation in a multi-threaded server. `with` is the test-doubles constructor.

### 8.4 `StrategyContext`

```rust
pub struct StrategyContext<'a> {
    pub manifest: &'a SemanticManifest,
    pub session:  &'a SessionContext,
    pub registry: &'a StrategyRegistry,
    pub diagnostics: &'a DiagnosticSink,
    pub plan_builder: &'a dyn PlanBuilder,
}

impl<'a> StrategyContext<'a> {
    pub fn plan_datakind(
        &self,
        request: &ResolvedQueryRequest,
        datakind: &ResolvedDataKind,
    ) -> Result<PlanNode, PlanError>;
    pub fn emit(&self, diagnostic: Diagnostic);
}
```

`plan_datakind` is the recursive-dispatch entry point for Complex strategies (Unionset / Grainset / Joinset) planning child subplans. `emit` adds a non-error `Diagnostic` to the accumulator; at step 6 the accumulator is drained into `SemanticPlan.diagnostics` (success) or `PlanErrors.warnings` (failure).

`DiagnosticSink` is an interior-mutable container (`RefCell` / `Cell`); every field of `StrategyContext` is either `&T: Sync` or an interior-mutable wrapper over a `Send + Sync` type. `PlanBuilder` is the `10 §6` injection-mode hook for adapter-provided deterministic rewrites; `DefaultPlanBuilder` (v1) applies none.

### 8.5 `dispatch_strategy`

```rust
pub fn dispatch_strategy<'r>(
    kind: &ResolvedDataKind,
    registry: &'r StrategyRegistry,
) -> Result<&'r dyn Strategy, PlanErrorKind> {
    match kind {
        ResolvedDataKind::Simple(_) => Ok(registry.simple()),
        ResolvedDataKind::Complex(ResolvedComplexDataKind::Unionset(_)) => Ok(registry.unionset()),
        ResolvedDataKind::Complex(ResolvedComplexDataKind::Grainset(_)) => Ok(registry.grainset()),
        ResolvedDataKind::Complex(ResolvedComplexDataKind::Joinset(_))  => Ok(registry.joinset()),
        _ => Err(PlanErrorKind::StrategyMissingForVariant),
    }
}
```

The single variant-tag match site in the planner's hot path (per `20 §5.3`). `#[non_exhaustive]` on `ResolvedDataKind` / `ResolvedComplexDataKind` requires a defensive fallback arm for forward-compatible builds. O(1) for known variants; every other planner site consumes the returned `&dyn Strategy` from here or from `ctx.plan_datakind` recursion.

## 9. Built-in strategies (v1 roster)

Each v1 strategy is a zero-sized, stateless struct constructed via `::new()` and held as `Box<dyn Strategy>` in the registry (§8.3). All four share the same `impl Strategy` shape — `id` returns the fixed `StrategyId`, `supports` pattern-matches on the variant tag, `plan` delegates to the algorithm in the cited data-kind spec. The signatures below omit the repetitive `impl Strategy` block; the authoritative algorithm lives in the cited sections.

### 9.1 `SimpleStrategy`

**Variant.** `ResolvedDataKind::Simple(ResolvedSimpleDataKind)`.

```rust
pub struct SimpleStrategy;
impl SimpleStrategy { pub fn new() -> Self; }
// impl Strategy per §8.1 — dispatches to the `21 §4` algorithm.
```

**Algorithm pointer.** `21 §4.1`–§4.7: L1 `Scan` per-source per `15 §3.6`; L2 `Rename` Semantics → physical columns; L3 `Expression` materializes Measure / Metric / Dimension expressions from the manifest physical pool (`ManifestExpression { expr, layer }`, `33 §7.2`); L4 `Agg` aggregates per-Measure; L5 `Project` final-column projection. The Strategy reads each expression's `ExprLayer` to choose the L-layer placement (`Scalar` → L2/L3 pre-agg, `Aggregate` → L4, `PostAggregate` → above L4) per `19 §6.0`; pre-/re-aggregation safety from function-derived `Additivity` (`14a §3.6.2`). Single-source vs multi-source fan-out per `21 §4.2`; filter interleaving per `21 §4.6`.

**Errors.** `PLAN_E_21xx` per `21 §7`; shared `PLAN_E_0600`, `PLAN_E_0601`.

### 9.2 `GrainsetStrategy`

**Variant.** `ResolvedDataKind::Complex(ResolvedComplexDataKind::Grainset(ResolvedGrainset))`.

```rust
pub struct GrainsetStrategy;
impl GrainsetStrategy { pub fn new() -> Self; }
// impl Strategy per §8.1 — dispatches to the `22 §4` algorithm.
```

**Algorithm pointer.** `22 §4.1`–§4.5: derive `request_grain` from the Request's Dimensions; filter children via the eligibility predicate; apply the rollup-legality gate (cross-ref `17 §8`); pick the winner via the cost function + deterministic tie-break; recursively dispatch into the winner's strategy via `ctx.plan_datakind`; optionally wrap with rollup nodes per `22 §10.1`. The Grainset is a **planner strategy**, not a `PlanNode` variant — the emitted tree rooted at the Grainset's subtree is identical to what would be produced if the Request had targeted the winning child directly, plus the optional rollup wrapper.

**Errors.** `PLAN_E_22xx` per `22 §8`; notably `PLAN_E_2201 NoEligibleChild`, `PLAN_E_2205 SnapshotRollupWithoutPin`.

### 9.3 `UnionsetStrategy`

**Variant.** `ResolvedDataKind::Complex(ResolvedComplexDataKind::Unionset(ResolvedUnionset))`.

```rust
pub struct UnionsetStrategy;
impl UnionsetStrategy { pub fn new() -> Self; }
// impl Strategy per §8.1 — dispatches to the `23 §4` algorithm.
```

**Algorithm pointer.** `23 §4.1`–§4.6: per-branch Coverage narrowing (`23 §4.4`); recursive dispatch into each branch's strategy via `ctx.plan_datakind`; NULL-fill / type-reconciliation seam (`23 §4.3`); NullFill-only branch pruning (`23 §4.6`; advisory `PLAN_W_2301`; fully-NullFilled requests raise `PLAN_E_2303`); emit `PlanNode::Union` over surviving branches; optional terminal re-aggregation (`23 §4.5`).

**Errors.** `PLAN_E_23xx` per `23 §10`.

### 9.4 `JoinsetStrategy`

**Variant.** `ResolvedDataKind::Complex(ResolvedComplexDataKind::Joinset(ResolvedJoinset))`.

```rust
pub struct JoinsetStrategy;
impl JoinsetStrategy { pub fn new() -> Self; }
// impl Strategy per §8.1 — dispatches to the `24 §5` algorithm.
```

**Algorithm pointer.** `24 §5.1`–§5.5: scan-plan the anchor via `ctx.plan_datakind`; for each hop in `ResolvedJoinset.path`, scan-plan the hop target and emit a `PlanNode::Join` with per-hop `JoinType` / `Cardinality` (`24 §5.2`, `16 §4`); after every hop, emit a `Project` reconciling the `UnifiedSemantics` (`16 §6`). `AsOf` join emission is DEFERRED per `17 §5` / `24 §5.4` — the v1 planner emits `PLAN_W_2401 AsOfNotYetActive` and falls back to an inner join on the non-temporal key.

**Errors.** `PLAN_E_24xx` per `24 §10`.

## 15. Stability

### 15.1 `Strategy` trait — open for adapter extension

The `Strategy` trait is **non-sealed** (`30 §4.6`). Third-party crates MAY implement it. This is a deliberate extensibility point: adapter authors who need a custom plan-tree shape for a novel `Complex` DataKind variant (added under I10 per `20 §5.3`) contribute a new `Strategy` impl and register it into a custom `StrategyRegistry`.

**Caveat.** The built-in variant dispatch (`dispatch_strategy`, §8.5) matches on the ratified `ResolvedDataKind` variant set — a third-party strategy is useful only when paired with a third-party `ResolvedDataKind` variant (also under I10 per `20 §5.1`). The v1 built-in registry holds exactly the four built-in strategies; a custom strategy requires building a custom registry.

Whether the trait should be **sealed** (restricting impls to the workspace) is tracked as `Q-PLAN-002` in open questions. The Round-1 default is non-sealed per `Q-KIND-001` (in `[questions/open/20_questions.md](../../questions/open/20_questions.md)`) pending resolution.

### 15.2 Built-in strategies — stable

`SimpleStrategy`, `GrainsetStrategy`, `UnionsetStrategy`, `JoinsetStrategy` are **stable types** in the crate's public API. Renaming any of them is MAJOR per `30 §2.1`. Behavioral changes to their `plan` methods are MINOR if they preserve the ratified algorithm (`21 §4`, `22 §4`–§5, `23 §4`, `24 §5`) and MAJOR if they alter the observable shape of the emitted plan tree.

### 15.3 `OptimizerPass` trait — open

The `OptimizerPass` trait is **non-sealed** per §12.3. Third-party passes compose via `OptimizerBuilder::with(...)`. The v1 canonical passes are stable types; the pass-chain order within `Optimizer::with_v1_passes()` is stable per §11.7.

### 15.4 `#[non_exhaustive]` discipline

Every public `pub enum` and every public `pub struct` exposed by `34` carries `#[non_exhaustive]` with the following exceptions (newtype-over-stable per `30 §3.5`):

- `DataKindRef` — newtype over `DataKindName`.
- `StrategyRegistry` — stable shape by construction.
- `Optimizer` / `OptimizerBuilder` — stable internal vector.

Adding a new variant to any `#[non_exhaustive]` enum (e.g. a new `PlanErrorKind` variant for a new error condition, a new `StrategyId` variant for an externally-contributed strategy) is MINOR per `30 §2.2`.

### 15.5 Variant identity stability

Per `30 §2.1` / `30 §5.4`:

- A published `PlanErrorKind` / `OptimizeErrorKind` variant's identity (its name + payload shape) is frozen at its first release.
- Adding a new variant inside `#[non_exhaustive]` is MINOR.
- Renaming or removing a variant is MAJOR. Deprecation rides via `#[deprecated(since, note)]` and is MINOR per `30 §12.3`.

The `PLAN_E_0500` aliasing referenced in `planner-errors.md §13.1`'s legacy code allocation table is no longer applicable: variant identity is the surface, and `ConstraintViolation` and `AmbiguousImplicitComposition` are distinct variants. `§17`'s Q-PLAN-003 closes accordingly.

## Testing Strategy

- Strategy trait signature (id/supports/plan; Send+Sync bounds)
- StrategyId enum (Simple/Grainset/Unionset/Joinset variants + as_str())
- StrategyRegistry default_v1 + with constructor (simple/grainset/unionset/joinset accessors)
- StrategyContext plan_datakind recursive-dispatch + emit non-error diagnostic (manifest/session/registry/diagnostics/plan_builder fields)
- dispatch_strategy single variant-match site (ResolvedDataKind → &dyn Strategy; StrategyMissingForVariant fallback)
- SimpleStrategy (21 §4 5-layer Scan/Rename/Expression/Agg/Project; filter interleaving; PLAN_E_21xx errors)
- GrainsetStrategy (22 §4 child-selection + rollup wrap; PLAN_E_22xx errors; NoEligibleChild/SnapshotRollupWithoutPin)
- UnionsetStrategy (23 §4 per-branch dispatch + NULL-fill + optional re-agg; PLAN_E_23xx errors)
- JoinsetStrategy (24 §5 anchor-outward path walk + Join sequences; PLAN_E_24xx errors; AsOfNotYetActive warning)
- non-sealed trait per 30 §4.6
- built-in strategies stable types per 30 §2.1

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
