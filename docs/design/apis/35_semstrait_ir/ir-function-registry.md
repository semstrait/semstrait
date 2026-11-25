---
id: 35-ir-function-registry
status: ratified
version: 2.1
impl: implemented
---

# IR Function Registry

The canonical function **identity** (`CanonicalFn`), the registry shape (`FunctionRegistry`), the specification shape (`FunctionSpec`, `FnSignature`, `ParamType`, `ReturnTypeRule`, `FunctionCategory`), and the extension hook (`RegistryExtension`) are all ratified in `14a`. Under item Y / V4 the homes split:

- **`CanonicalFn` name identity → `semstrait-model`** (imported here). It is an author-writable token; the `FunctionCall` variant on the authoring `model::Expr<L>` carries it.
- **Registry + spec family + extension hook → `semstrait-ir`** (owned here). The registry resolves the model-owned `CanonicalFn` name to a `FunctionSpec`; signature matching and `ReturnTypeRule` evaluation are resolved-IR concerns that name `DataType` and emit `CompileError`.

This supersedes the earlier "registry moved common→ir, adjacent to `Expr<L>`" rationale (`14 §9.2`): `Expr<L>` is now model-owned, but the *registry* stays ir because resolution — not authoring — is what consumes it.

## Public Surface

`CanonicalFn` is **imported** from `semstrait-model` (the name-identity newtype + its validating `new(...) -> Result<Self, common::ValidateError>` constructor live at `32`); `35` owns everything below it.

```rust
use semstrait_model::CanonicalFn;  // name identity (item Y / V4); body at `32`

/// The single canonical function catalog. Per `14a §2.2 / §2.3`.
///
/// Sealed at startup by every crate that registers signatures
/// (`semstrait-ir` bootstraps the built-ins; adapter crates contribute
/// engine-specific overloads via `RegistryExtension` per `14a §7`).
pub struct FunctionRegistry { /* crate-private */ }

impl FunctionRegistry {
    pub fn lookup(&self, name: &CanonicalFn) -> Option<&FunctionSpec>;
    pub fn contains(&self, name: &CanonicalFn) -> bool;
    pub fn iter(&self) -> impl Iterator<Item = (&CanonicalFn, &FunctionSpec)>;
    pub fn is_sealed(&self) -> bool;
}

/// Process-wide accessor returning the sealed registry. Per `14a §2`.
pub fn function_registry() -> &'static FunctionRegistry;
```

Specification structures define per-function signatures, parameter-type constraints, return-type rules, and function categorization.

```rust
/// Per-function specification — name, signature overloads, category,
/// return-type rule. Per `14a §3`.
#[non_exhaustive]
pub struct FunctionSpec {
    pub name:        CanonicalFn,
    pub signatures:  Vec<FnSignature>,
    pub category:    FunctionCategory,
    pub return_type: ReturnTypeRule,
}

/// One signature overload — argument types + arity discipline.
/// Per `14a §3.1`.
#[non_exhaustive]
pub struct FnSignature {
    pub params:        Vec<ParamType>,
    pub variadic_tail: Option<ParamType>,
}
```

The parameter-type and return-type carriers, the category axis, and the adapter extension hook follow.

```rust
/// Parameter-type carrier. Per `14a §3.2`.
#[non_exhaustive]
pub enum ParamType {
    Concrete(DataType),
    AnyOf(Vec<DataType>),
    NumericFamily,
    StringFamily,
    TemporalFamily,
    Any,
}

/// Return-type computation rule. Per `14a §3.3`.
#[non_exhaustive]
pub enum ReturnTypeRule {
    Fixed(DataType),
    SameAsFirstArg,
    SameAsArg(u32),
    Custom(fn(&[DataType]) -> Result<DataType, CompileError>),
}

/// Function-category axis (scalar / aggregate / window). Per `14a §3.4`.
#[non_exhaustive]
pub enum FunctionCategory {
    Scalar,
    Aggregate,
    Window,
}

/// Adapter / downstream-crate extension hook. Per `14a §7`.
///
/// Declarative: implementers expose a static spec list and an adapter
/// identity. `function_registry()` enumerates linked impls at startup
/// and folds their `FUNCTIONS` into the sealed registry.
///
/// Not sealed — adapter crates outside the workspace MAY contribute
/// their own overloads.
pub trait RegistryExtension {
    const ADAPTER_ID: &'static str;
    const FUNCTIONS: &'static [FunctionSpec];
}
```

`CompileError` is owned here in `semstrait-ir::error` (see [ir-errors.md](ir-errors.md)) — raised by function resolution against the sealed registry plus the registry callbacks (`ReturnTypeRule::Custom`). `ValidateError` (raised by the trait machinery `Tree::with_new_children` / `Rewriter<N>::f_*`) is **owned by `semstrait-common`** (item Y / V5) and imported; it is no longer an ir error.

## Per-Kind Accessor Enums — Imported

```rust
use semstrait_model::{DimensionAccessor, MeasureAccessor, MetricAccessor, KeyAccessor};
```

Per-entity sugar (`measure("revenue").previous()`, `metric("conv_rate").delta()`) is an **authoring** concern: each accessor enum rides as an `Option<…>` field on the matching `UnresolvedSemanticLeaf` variant, so it is **owned by `semstrait-model`** alongside that leaf (item Y / V6; bodies + variant catalogs at `32`, ratified by `14 §4.1`). `35` imports them only because the imported `UnresolvedSemanticLeaf` names them. The structural pairings (`MetricAccessor` ≡ `MeasureAccessor`, `KeyAccessor` ≡ `DimensionAccessor`), the no-outer-`Accessor`-wrapper rule, and the accessor-less `Field` leaf are all model-surface properties per `14 §4.1`. The resolved IR never carries an accessor — compile eliminates sugar into `Window`-rooted `ir::PhysicalExpr` subtrees (`14 §4.2`).

## `Parameter` and `ParameterKey`

Compile-emitted, plan-bound placeholder per `14 §5`:

```rust
/// Plan-bound parameter placeholder. Substituted by the planner during
/// Phase B per `19 §2.1`. Per `14 §5.1`.
pub struct Parameter {
    pub key: ParameterKey,
    pub data_type: DataType,
}

/// Closed set of typed parameter keys. Internal to the canonical pipeline —
/// not author-extensible. v1 carries the two keys needed by sugar-accessor
/// elimination (`14 §4.2`); future keys land additively per I10.
#[non_exhaustive]
pub enum ParameterKey {
    RequestDimensionsMinusTemporal,
    RequestTemporalAxis,
}
```

`Parameter` lives only in `PhysicalLeaf`; `UnresolvedSemanticLeaf` carries no `Parameter`. The plan-time binding postcondition is per `14 §5.3`: no `Parameter` survives into adapt time. Phase B substitution mechanics are per `19 §6`; a `Parameter` reaching an adapter is a hard error owned by the planner (`PlanErrorKind`), not by `35`.

## Authoring-Surface DSL

The Rust authoring DSL (`expr_fn` free constructors, `std::ops` operator overloads, the `ExprFunctionExt` builder-sugar trait) mirrors the YAML reserved-tag catalog of `14 §6.4.1` exactly — `dim("region")` in Rust corresponds to `{ dim: region }` in YAML. Under item Y it constructs the **model-owned authoring vocabulary**: `model::Expr<L>` over the author-facing leaf sets, **never** the resolved `ir::PhysicalExpr`. It is **opt-in** ergonomic sugar — every value it produces is also constructible via direct struct / enum literal.

Contracts the DSL must honor (rosters ratified by `14 §6.4.1` / `14 §3.3` / `14 §4.1`, surface bodies at `32`):

- **`col` is dual-site.** It constructs either authoring form per the call-site type (`UnresolvedPhysicalExpr` at physical sites; `UnresolvedSemanticExpr` under `semantic_mapping: auto` per `14 §8`).
- **`std::ops` overloads** (`Add`/`Sub`/`Mul`/`Div`/`Rem`, `BitAnd`/`BitOr` as SQL `AND`/`OR`, `Neg`/`Not`) ride generic `model::Expr<L>`, producing `Expr::BinaryOp` / `Expr::UnaryOp`.
- **`ExprFunctionExt`** supplies the predicate / cast / aggregate / accessor sugar that operators cannot express; `aggregate(...)` is admitted only at aggregate-admitting sites (`14 §7`).
- **Accessor sugar lowers to `Window` at compile** (`14 §4.2`): DSL methods produce author leaves with `accessor: Some(_)`, never `Window` nodes; `Window` is not author-constructible (`14 §6.4.1`).

**Crate home — resolved to `semstrait-model` (OQ-D closed, option (a), 2026-06-15).** The earlier "DSL lives in `semstrait-ir`" placement (Q4.F) is superseded by item Y's relocation of the authoring vocabulary to `semstrait-model`: an authoring DSL belongs with the authoring types it builds. **OQ-D** ratified the DSL lands **wholly in `semstrait-model`** (not split) — see `questions/closed/35_questions.md`. `35` therefore does not own this surface; this note records the contract for context.

## Testing Strategy

This leaf owns `FunctionRegistry`, `FunctionSpec`, `FnSignature`, `ParamType`, `ReturnTypeRule`, `FunctionCategory`, `RegistryExtension`, `function_registry()`, `Parameter`, and `ParameterKey`. Tests:

- **Registry sealing:** Integration test verifies `function_registry()` is sealed at startup; calling it twice returns the same `&'static` reference; `is_sealed()` returns `true`. Bootstrap built-ins are present in the sealed registry.
- **Function lookup:** Unit tests verify `lookup(&CanonicalFn)` returns the correct `FunctionSpec` for registered functions; `None` for unregistered names. `contains` matches `lookup(...).is_some()`.
- **Signature overload matching:** Unit tests per `14a §3.1` verify call-site arity + argument types match exactly one overload (no implicit coercion). `FunctionArityMismatch` / `NoMatchingSignature` errors (see [ir-errors.md](ir-errors.md)) raised when no overload matches.
- **ReturnTypeRule evaluation:** Unit tests verify `Fixed` / `SameAsFirstArg` / `SameAsArg(ix)` / `Custom` return-type rules produce the correct `DataType` for matched signatures. `CustomRuleRejected` error raised when a `Custom` callback declines.
- **RegistryExtension hook:** Integration test verifies adapter crates can impl `RegistryExtension` and contribute overloads via `FUNCTIONS` static array; the sealed registry folds them in at startup.
- **Parameter substitution postcondition:** Unit tests verify `PhysicalExpr` containing `Parameter` leaves are substituted by the planner during Phase B (see `19 §6`). A `Parameter` reaching adapt time is a hard `PlanErrorKind` error (planner contract, not ir contract).
- **ParameterKey closed set:** Unit tests verify v1 carries exactly `RequestDimensionsMinusTemporal` and `RequestTemporalAxis` per `14 §5.1`; adding new keys is MINOR per I10.

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
