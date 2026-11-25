---
id: 16-composition-overview
status: ratified
version: 2.1
impl: implemented
---

# Composition Overview

> **Struct ownership (2026-04-17 consolidation; relationship-block rebase 2026-05-12).** The `Relationship` struct (identified by its `id: EntityId`), `Cardinality`, `Integrity`, `Optional`, `CrossFilter`, (derived) `JoinType`, and `JoinKeyExprPair` are ratified in [`18_entities.md §2`](../18_entities.md#2-relationship). This doc owns the *composition semantics* on top — placement, scope, traversal, fanout analysis, `ComposedSemanticInterface` construction, field-first resolution. The struct-shape lands in `18`; what the planner does with it lands here.

This navigational leaf opens the composition design space. `semstrait` composes data along two axes: vertically (grouping equivalent-shape `DataKind`s into `Unionset` or `Grainset`) and horizontally (declaring `Relationship`s between complementary-Semantics kinds, named as `Joinset` or walked implicitly).

The composition machinery closes three open items from `00 §4.1`:

- **(i) `ComposedSemanticInterface` structural shape** — ratified in [composed-surface-type](composed-surface-type.md).
- **(ii) Materialization policy** — ratified in [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md): all compositions (explicit and implicit) are materialized at `SemanticGraph` build time; the SemanticManifest carries only primitives, no `compositions:` field.
- **(iii) Explicit vs implicit composition scope** — ratified in [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md): implicit composition is bounded by declared `Relationship`s, depth-limited to 4 hops, and count-capped at 2000 per Model. Clash with explicit → reject at compile.

**Status.** Round 2 ratified 2026-04-29; lightweight-manifest cascade 2026-05-28. Unified Joinset model, eager materialization (rebased to `SemanticGraph` build time), and implicit-explicit-clash rejection all closed. Round-1 ratifications for explicit-only composition shape preserved.

## 1. Purpose and Scope

`16` is the authoritative specification for the horizontal axis's **core type machinery** (`Relationship`, derived `JoinType`, `ComposedSemanticInterface`, `Origin`, `ImplicitId`, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`), for the **boundary** between `Origin::Explicit` and `Origin::Implicit` compositions, the **eager-materialization policy** (rebased 2026-05-28 — manifest carries primitives only; `SemanticGraph` reconstructs the composition index at planner setup), and the **field-first resolution algorithm** (pure lookup at plan time).

Per-`DataKind` materialization strategies (`Unionset`, `Grainset`, `Joinset` bodies), the YAML authoring surface, and the SemanticManifest / Planner IR are refined in the `refined-by` docs.

### 1.1 What `16` ratifies (index)

`16` ratifies: the `Relationship` struct + `KeyPair` + `Integrity` + `Optional` + `CrossFilter` (see [relationship-edge-model](relationship-edge-model.md)); `Cardinality` (see [relationship-edge-model](relationship-edge-model.md) §3); derived `JoinType` + `PlanNode::Join` carriage (see [relationship-edge-model](relationship-edge-model.md) §4); `ComposedSemanticInterface` + `CompositionKind` (2 variants V1: `Joinset` and `Grainset`) + `Origin` axis + `ImplicitId` + `SemanticsView` trait (see [composed-surface-type](composed-surface-type.md), **resolves (i)**); `UnifiedSemantics` merge logic (see [composed-surface-contents](composed-surface-contents.md) §6); `FieldProvenance` + `FieldOwnership` (see [composed-surface-contents](composed-surface-contents.md) §7); `CompositionCoverage` extending `15 §6` (see [composed-surface-contents](composed-surface-contents.md) §8); the explicit-vs-implicit composition boundary, including transparent unfolding through composed surfaces (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §9, **resolves (iii)**); the materialization policy — eager enumeration at `SemanticGraph` build with cap + canonical-ID + clash-reject (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10, **resolves (ii)**); the field-first resolution algorithm as pure lookup (see [field-first-resolution](field-first-resolution.md) §11); `Relationship` graph well-formedness preconditions (see [composition-validation-and-errors](composition-validation-and-errors.md) §12); `Joinset`'s explicit and implicit forms under the unified model (see [joinset-model](joinset-model.md) §13); and new `CompileError` / `ValidateError` / `PlannerError` variants with stable codes in the `COMP_E_04xx` / `PLAN_E_05xx` / `PLAN_W_05xx` ranges (see [composition-validation-and-errors](composition-validation-and-errors.md) §14).

### 1.2 What `16` does NOT ratify

- **Per-`DataKind` planning strategies** — Scan / Join / Aggregate / Project lowering lives in `20`–`25` and `34`; `16` ratifies the type-of-surface the strategies plan against.
- **YAML authoring syntax** — `relationships:` and `joinsets:` block shapes and defaults live in `32`.
- **`SemanticManifest` serialization** — on-disk shape of `Relationship` (root + Joinset shadow) and the closed `DataKindVariant` taxonomy mirror lives in `33`. The `ResolvedComplexDataKind` and `RelationshipGraph` adjacency index are NOT persisted (per C7.4 / C8.2 / C9.6 — `SemanticGraph` reconstructs them at planner setup from manifest primitives).
- **Fanout-safe-rewrite algorithm shape** — the rewrite's plan-tree transformation lives in `20` / `34`; `16` ratifies the `Cardinality × JoinType` matrix that triggers it.
- **`JoinType::AsOf`** — deferred; gated on `17 TemporalShape`.
- **Physical predicate lowering** — `34` / `36` for adapter rendering.
- **Diagnostic rendering** — `30 §5` owns the Diagnostic shape; `16` ratifies the stable codes and structured payloads.

### 1.3 Design posture

Five stances govern:

1. **Name, not column.** `Relationship.keys` pair `SemanticsName`s; per I1 physical resolution is `15`'s responsibility.
2. **Declare the edges, let the compiler enumerate the walks.** Authors declare pairwise `Relationship`s; compile enumerates every implicit `Joinset` (and implicit `Unionset` for coverage overlap) within the depth + count bounds. Authors who want to override defaults declare an explicit `Joinset`.
3. **Materialize everything.** Both explicit and implicit compositions are materialized as `ResolvedComplexDataKind` shapes inside `SemanticGraph` at planner-setup build time (rebased 2026-05-28 per C9.2 / C10.2 — manifest carries primitives only). Plan-time is a pure lookup over `SemanticGraph`'s build-time composition index. The eager-enumeration cap (`MAX_IMPLICIT_ENUMERATION_COUNT = 2000`) protects against pathological models.
4. **One canonical form per composition.** An explicit `Joinset` whose canonical form matches an enumerable implicit `Joinset` is rejected at compile (`COMP_E_0414`). Authors differentiate via per-leg overrides, filters, or `keys`.
5. **Fail fast, disambiguate up.** Path ambiguity (multiple shortest paths between same constituents) errors at plan time; authors disambiguate by declaring a differentiated explicit `Joinset`.

### 1.4 Guardrails upheld

- **I1 (canonical layer).** `Relationship.keys` are `SemanticsName`s; physical resolution is `15`'s responsibility.
- **I4 (determinism).** Ambiguous implicit-composition paths error (`PLAN_E_0500`); no heuristic tie-break.
- **I5 (compile-time resolution).** Name indices and the `RelationshipGraph` are pre-built at `compile`; the planner's walk is lookup, not resolution.
- **I8 (SemanticManifest is planner-complete).** The planner reads primitives from the SemanticManifest; `SemanticGraph` reconstructs the composition index from those primitives once at planner setup. No catalog fetch, no re-parse, no plan-time BFS.
- **I10 (non-exhaustive public sums).** `Cardinality`, `Integrity`, `Optional`, `CrossFilter`, `JoinType`, `CompositionKind`, and `FieldOwnership` all carry `#[non_exhaustive]`.
- **I12 (fail-fast).** `CompileError::*` composition variants abort `compile`; `PlannerError::*` composition variants abort `plan`.

## Navigation map to sibling leaves

- **[Relationship / Cardinality / JoinType](relationship-edge-model.md)** — The `Relationship` struct (§2), `Cardinality` (§3), derived `JoinType` (§4). Use when working with relationship declarations, join-type semantics, cardinality-based fanout, bidirectional traversal.
- **[ComposedSemanticInterface type](composed-surface-type.md)** — `ComposedSemanticInterface` struct (§5), `CompositionKind`, `Origin`, `ImplicitId`, `SemanticsView` trait. Use when working with the unified queryable surface, explicit vs implicit axis, canonical-form hashing.
- **[UnifiedSemantics / FieldProvenance / CompositionCoverage](composed-surface-contents.md)** — `UnifiedSemantics` (§6), `FieldProvenance` (§7), `CompositionCoverage` (§8). Use when working with namespace-aware field merging, per-field ownership, composition-level coverage.
- **[Explicit vs Implicit boundary](implicit-enumeration-and-materialization.md)** — Explicit vs Implicit composition (§9), materialization policy (§10). Use when working with implicit-Joinset enumeration, implicit-Unionset enumeration, canonical-form clash, depth and count bounds.
- **[Field-first resolution](field-first-resolution.md)** — The planner's field-first lookup algorithm (§11). Use when working with plan-time composition selection, `Request.from = None` path, multi-kind lookup.
- **[Joinset model](joinset-model.md)** — `Joinset` explicit and implicit forms (§13). Use when working with Joinset authoring, scope-local Relationship shadow, canonical-form differentiators, traversed paths. Refines `24` (authoritative home for Joinset).
- **[Validation and error model](composition-validation-and-errors.md)** — `Relationship` graph well-formedness (§12), new error variants (§14). Use when working with duplicate Relationship detection, key type agreement, self-reference rejection, error codes `COMP_E_04xx` / `PLAN_E_05xx`.

## Testing Strategy

Navigational leaf — covered by `doc-space-check` structural gates. Behavioral tests for the composition machinery are distributed across the sibling leaves' testing strategies and the shared harness [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
