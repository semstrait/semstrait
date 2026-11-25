---
id: 30-breaking-change-governance
status: ratified
version: 2.1
impl: implemented
---

# Breaking Change Governance

This leaf owns the breaking-change governance and deprecation policy — additivity escape hatch, non-additive changes require MAJOR changelog + migration notes + deprecation window, and the deprecation lifecycle.

## 11. Breaking-Change Governance

### 11.1 The additivity escape hatch

Per §2, `#[non_exhaustive]` variant addition and `#[non_exhaustive]` struct-field addition are MINOR. This is the workspace's primary "non-breaking evolution" mechanism. A new `JoinType::Semi` variant, a new `DataType::Json` variant, a new `Additivity::Partial` category — all ship in MINOR releases with a migration-notes entry describing caller-visible behavior (if any) and an updated match-recommendation table.

### 11.2 Non-additive changes

Any change to a `pub` type / function / trait that is not additive requires:

1. A **MAJOR changelog entry** in the release notes, calling out the break.
2. A **migration-notes entry** with before-and-after examples and replacement guidance.
3. A **deprecation window** of at least one MINOR cycle where feasible (§12).
4. A **`cargo-semver-checks` lint** green-light (or a documented waiver) before publish.

### 11.3 Cross-crate breaks

When a break propagates across crates (e.g. a `semstrait-ir` variant removal forces `semstrait-adapter` impls to re-match), every affected crate's migration-notes entry cross-references the others. The workspace's coordinated-release (§2) ensures no version pair of crates ever exposes a mid-break state.

### 11.4 Behavior-preserving refactors

Internal refactors that preserve every observable output (same `SemanticManifest` bytes, same `SemanticPlan` tree, same `EngineArtifact` text) are PATCH. Determinism (I4) makes this bit-comparable for `SemanticManifest` and `SemanticPlan`; adapter output is compared at the `SqlArtifact::text` / `EnginePlan` serialization level. A refactor that produces equivalent but not byte-identical output is MINOR (callers comparing artifacts byte-for-byte, e.g. for content-addressable caching, see the change).

## 12. Deprecation Policy

### 12.1 `#[deprecated]` lifecycle

A symbol slated for removal passes through three states:

1. **Active** — fully supported, documented, used in examples.
2. **Deprecated** — `#[deprecated(since = "VERSION", note = "use X instead; removed in ...")]` attribute present. Still compiled and callable; callers receive a rustc deprecation warning. Lives for at least one MINOR cycle.
3. **Removed** — the symbol is deleted in a MAJOR bump. The matching migration-notes entry references the earlier deprecation entry.

### 12.2 Deprecation register

Every deprecation — the moment a `#[deprecated]` attribute lands — is recorded in the deprecation register with:

- The symbol's fully-qualified path.
- The `since` version.
- The suggested replacement.
- The target removal version (best estimate).

When the symbol is removed, the register entry moves to the relevant migration-note entry; the register retains tombstones for at least one MAJOR after removal.

### 12.3 Deprecation ≠ removal

A **deprecated** symbol (per-stage kind variant, public function, public type) is still part of the surface and remains callable / matchable; rustc emits a deprecation warning at the use-site. A **removed** symbol is gone from the public surface entirely. Removal is MAJOR; deprecation is MINOR. The deprecation window MUST cover at least one MINOR cycle.

For per-stage kind variants specifically: deprecating a variant means the library MAY continue to produce it for one MINOR cycle, then either stop producing it (replacing emissions with a successor variant) or remove it (in the next MAJOR). The replacement-variant transition is the period in which both old and new variants may surface for the same condition; callers SHOULD match the new variant first.

### 12.4 Minimum window

At least one full MINOR cycle between `#[deprecated]` and removal. Longer for widely-used symbols — per-crate `3x` docs may specify extended windows (e.g. two MINOR cycles for a core `CatalogProvider` method).

## 13. Stability Table

v1 per-crate maturity markers. These lock at the v1.0 cut and evolve per the semver rules of §2.

| Crate | Stability | Notes |
|---|---|---|
| `semstrait-common` | Stable in v1 | Stage-agnostic infrastructure after the item Y inversion (`STATUS.md`): the `Tree` / `Visitor` / `Rewriter` traversal family + the narrow tree-machinery `ValidateError` (V5), `Diagnostic`, `Diagnostics`, `Severity`, `Location`, `Span`, `SourceId`, `Diagnose`, constraint DSL, `io` transport. Carries **no** type vocabulary, expression tree, or plan tree — the authored `DataType` / `Grain` + `Expr<L>` live in `semstrait-model`, the resolved `PhysicalExpr` + `TypeClass` / `Schema` / `SchemaColumn` in `semstrait-ir`. Breaking changes require a workspace-wide MAJOR. |
| `semstrait-model` | Stable in v1 | `SemanticModel`, the shared `Expr<L>` family (V6; **incl. the `Window` variant** — A2; `ir::PhysicalExpr` is its `Expr<PhysicalLeaf>` alias; authoring expression fields are bare `Expr<L>` via `Unresolved{Semantic,Physical}Expr` — the `ExprSource` wrapper is removed per doc 45, friendly projection now in field-level adapters) with leaf sets `UnresolvedSemanticLeaf` / `UnresolvedPhysicalLeaf` + `ExprLeaf` + accessors, support enums (`BinaryOpKind`, `UnaryOpKind`, `AggregationOp`, `AggregateKind`, `LikeKind`, `CastFailure`) + the Window support enums (`WindowFn` / `WindowFrame` / `WindowFrameKind` / `WindowBound` — A2) + `Literal`, `ColumnRef`, `SemanticsName`, the `CanonicalFn` name (V4), the authored type vocabulary `DataType` / `Grain` (V2), `ParseErrorKind`, `ValidateError` (validate-stage roster, embedding `Tree(common::ValidateError)` per D.ii — `14 §9.1` / V5), YAML grammar. The author-facing YAML shape extends non-exhaustively (new keys, new variants) in MINOR. |
| `semstrait-ir` | Stable in v1 | The resolved `PhysicalExpr` (the `Expr<ir::PhysicalLeaf>` alias — item Y / A2; admits the `Parameter` leaf), `PhysicalLeaf`, the **non-authored** type vocabulary `TypeClass` / `Schema` / `SchemaColumn`, `FunctionRegistry` / `FunctionSpec` / `Additivity`, narrow `CompileError` (`ReturnTypeRule::Custom`) / `IrErrorKind`, `SemanticPlan`, `PlanNode`, `EngineArtifact`, `SqlArtifact`, `EnginePlan`. Imports the authoring vocabulary (`Expr<L>` family incl. the `Window` variant, leaves, support enums incl. the Window support enums `WindowFn` / `WindowFrame` / `WindowFrameKind` / `WindowBound` (A2), `Literal`, `ColumnRef`, `SemanticsName`, `CanonicalFn`-name, `DataType` / `Grain`) from `semstrait-model` and the `Tree` family + tree `ValidateError` from `semstrait-common` (item Y / V1–V10, A2). Variant and field additions are MINOR via `#[non_exhaustive]`. |
| `semstrait-manifest` | Stable in v1 | `SemanticManifest`, `Resolved*` family, `CompileError` (embedding `Ir(ir::CompileError)` per D.ii — `35 §16.2`), `RepositoryErrorKind`, `Repository` trait. **Internal serialization format (SemanticManifest on-disk bytes) is NOT a public API** — callers round-trip through `Repository::save` / `Repository::load`, not through direct byte access. |
| `semstrait-planner` | Stable in v1 | `plan`, `optimize`, `PlanError`, `OptimizeError`. Per-DataKind strategy dispatch is internal. `PlanNode` variants are I10 `#[non_exhaustive]` and defined in `semstrait-ir`. |
| `semstrait-adapter` | Provisional | `EngineAdapter` trait stable; `AdaptError` stable; `DialectId` extends in MINOR. Per-engine adapter crates (`semstrait-adapter-datafusion`, `semstrait-adapter-duckdb`, `semstrait-adapter-spark`, `semstrait-adapter-substrait`) are **versioned independently** and may carry their own stability tier in their own `3x` appendix. |
| `semstrait-catalog` | Provisional | `CatalogProvider`, `FileSystem`, local-fs impl stable. Per-provider impls (`semstrait-catalog-iceberg`, `semstrait-catalog-unity`) are **versioned independently**; their stability follows their own maturity. |
| `semstrait-api` | Stable in v1 | Unified entry point wrapping the `parse → … → adapt` pipeline. Re-exports the minimum `semstrait-*` types required to use the pipeline end-to-end. |
| `semstrait-facade` | Stable in v1 | Facade over `semstrait-api` for one-shot use (single compile, single plan, single adapt). Default features enable the minimum useful adapter bundle; extension is via `semstrait-api`. |

**Provisional** crates may introduce non-additive changes in MINOR cycles (pre-1.0 semver rules continue for these crates past workspace-v1.0 if they are not yet promoted). Every provisional change still carries a migration note.

## Testing Strategy

- cargo-semver-checks lint green-light before publish (§11.2 breaking-change gates)
- migration-notes entry for every MAJOR (§11.2 documentation requirement)
- deprecation-register entry for every deprecation (§12.2)
- Deprecation window spans at least one MINOR (§12.3 minimum lifecycle)
- Cross-crate breaks documented (§11.3 coordinated-release cross-references)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
