---
id: 16-field-first-resolution
status: ratified
version: 2.1
impl: implemented
---

# Field-first Resolution Algorithm

This leaf ratifies the planner's field-first resolution algorithm — the entry point when `Request.from = None`. Per `10 §3.4`, the `plan` stage consumes the `SemanticGraph` (built once from the SemanticManifest at planner setup; I8 — no I/O, no re-resolution at plan time) and produces a `SemanticPlan`. Field-first resolution runs before plan tree construction to decide *which composition surface* the plan will be built against.

Under the unified Joinset model (2026-04-29), every viable composition is pre-materialized inside `SemanticGraph` at build time (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md)). Manifest carries the primitives — `semantics: SemanticBitmap`, per-DataKind `coverage`, `relationships:`, the closed `DataKindVariant` taxonomy mirror (`33 §4.1`, §6) — and `SemanticGraph` reconstructs the composition index from them (C9.2 / C9.6). The plan-time algorithm therefore reduces to a **lookup over `SemanticGraph`'s build-time composition index** — no BFS, no synthesis, no ambiguity resolution at plan time except the documented path-ambiguity error (§11.4).

## 11. Field-first Resolution Algorithm

### 11.1 Inputs

- `graph: &SemanticGraph` — built from `&SemanticManifest` at planner setup; carrying the in-memory build-time composition index reconstructed from manifest primitives (per C9.2 / C9.6 cascade — manifest itself has no `compositions:` field, cf. `33 §4.1` / §6.8 / I12.15):
  - `name_index: BTreeMap<SemanticsName, Vec<DataKindRef>>` — all kinds (Simple + Complex, including implicit Joinsets and Unionsets indexed under their synthetic names) that declare or expose the name. Built from the manifest's `semantics: SemanticBitmap` (C4) and per-DataKind `coverage` masks (CCK / C5–C8).
  - `composition_index: BTreeMap<DataKindName, ResolvedComplexDataKind>` — every composition (explicit + implicit) synthesized in [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10. Explicit compositions project from manifest `data_kinds:` entries (the `DataKindVariant` per `33 §6.5–§6.7`); implicit compositions are enumerated by the graph from `relationships:` + per-DataKind `coverage` (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4 / §10.5).
  - `composition_by_canonical: BTreeMap<ImplicitId, DataKindName>` — reverse index from `ImplicitId` to the synthetic name; derived from the same enumeration pass.
  - `composition_by_constituent_set: BTreeMap<BTreeSet<DataKindRef>, Vec<DataKindName>>` — for each unordered constituent set, the list of compositions covering exactly that set (used by step §11.4 to detect path ambiguity); derived from the same enumeration pass.
- `request: &Request` — with `request.from = None` and `request.select: Vec<SemanticsName>` of length ≥ 1.

### 11.2 Step — Map each selected name to its owning kinds

For each `name` in `request.select`:

```text
let owning = graph.name_index.get(&name)
    .ok_or(PlannerError::UnknownSemantics { name })?;
```

`owning` is `Vec<DataKindRef>`. The list may include implicit compositions (under their synthetic names) when the implicit composition's `UnifiedSemantics` exposes the name. A name with zero entries is an error (`PLAN_E_0508 UnknownSemantics`).

Let `T = ⋃ owning` for all selected names — the set of **candidate owning kinds** for the Request, restricted to top-level kinds (we filter out the synthetic implicit-composition names at this step; they re-enter at §11.4 via the constituent-set lookup).

### 11.3 Step — Single-kind fast path

If `|T| == 1`: the Request is satisfiable from a single top-level kind. The planner takes that `DataKindRef` and plans against its `SemanticInterface` (Simple) or its pre-materialized `ComposedSemanticInterface` (explicit Complex). The `Request` is treated as if `from: Some(T[0])` had been declared.

This fast path dominates well-authored Models where most Requests target a single fact-like `DataKind` with co-located dimensions.

### 11.4 Step — Composition lookup (multi-kind path)

> **Phase 3 amendment (2026-05-28; cascade from C9.6 / C10).** The field-first algorithm reads from the **`SemanticGraph`'s build-time composition index**, NOT from a `compositions:` field on manifest (which does not exist per `33 §4.1` and I12.15). The graph builds that index from manifest primitives — `semantics: SemanticBitmap` (per C4), per-DataKind `coverage` masks (per CCK), and the root-scope `relationships:` map (plus Joinset-local shadow Relationships per `33 §6.7`) — by running [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4's BFS over the open Relationship graph at graph-build time. The pseudocode below uses `graph.*` indices directly; they are reconstructed by `SemanticGraph` at build time from manifest primitives and are never top-level manifest fields.

If `|T| >= 2`: look up the implicit / explicit composition that covers `T`.

```text
let candidates = graph.composition_by_constituent_set
    .get(&BTreeSet::from(T))
    .cloned()
    .unwrap_or_default();
```

**Outcome cases:**

- **`|candidates| == 1`** — exactly one composition (explicit or implicit) covers `T`. The planner uses it as the resolved target.
- **`|candidates| == 0`** — no composition covers exactly `T`. Two sub-cases:
  - **Connectable but not enumerated** — the kinds are connected by `Relationship`s but no implicit Joinset was enumerated for this exact `T` (e.g. depth bound at compile dropped it). Error: `PlannerError::NoCompositionPath` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0501`) — author declares an explicit Joinset to escape the depth bound.
  - **Disconnected coverage** — no implicit Unionset was enumerated either (e.g. coverage groups did not include `T` exactly because one of the kinds is a partial-coverage outlier). Same error: `PLAN_E_0501` cites the kinds with no shared composition path.
- **`|candidates| >= 2`** — multiple compositions of equal canonical cost cover `T` (path ambiguity, e.g. billing-vs-shipping address). Error: `PlannerError::AmbiguousImplicitComposition` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0500`) cites every candidate's `DataKindName` and, for implicit candidates, the `ImplicitId` and the `Relationship`-traversal differences. Authors disambiguate by declaring an explicit `Joinset` with at least one differentiator (per [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6's `candidate_differentiators` pattern), which makes the explicit form's canonical distinct from the ambiguous implicit ones — the implicit Joinset against which the explicit shadowed form was disambiguated remains in the index for Requests that *don't* select fields requiring the differentiator.

(Path ambiguity at plan time is the v1 surface for `Q-COMP-002`'s "error-on-tie" decision; the implicit enumeration at compile produces both candidates as distinct, deferring the ambiguity to plan-time field-first lookup.)

**Sub-step — depth-bound exception path.** If a Request's `T` would require a hop count greater than `MAX_IMPLICIT_COMPOSITION_DEPTH` (reached only via a path no implicit Joinset was enumerated for), the planner detects this from the absence of the constituent set in `composition_by_constituent_set` and the *presence* of the constituents in the `RelationshipGraph` connected component distance > bound. Error: `PlannerError::CompositionDepthExceeded` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0502`). Authors declare an explicit Joinset to escape.

### 11.5 Step — Final selection and surface handoff

After lookup yields exactly one composition (Explicit or Implicit), the planner:

1. Retrieves the `ResolvedComplexDataKind` from `graph.composition_index` (the `SemanticGraph`'s build-time index per §11.1; not a manifest field, cf. C9.2 / `33 §6.8`).
2. Performs a selected-name membership check: every `name` in `request.select` must exist on the resolved composed surface (`SemanticsView`). Missing → `PLAN_E_0507 SemanticsNotOnSurface`.
3. Hands the surface (and `traversed_paths` for Joinsets) to the strategy dispatcher (`34 §7`).

No `ComposedSemanticInterface` synthesis occurs at plan — the resolved surface is directly the one synthesized by `SemanticGraph` at planner setup (per C7.4 / C8.2 cascade; the manifest carried only primitives).

### 11.6 Step — `Request.from = Some(DataKindRef)` path

When `Request.from` is specified, field-first resolution does NOT run. Instead:

1. Look up the named `DataKindRef`. Top-level explicit kinds (Simple and Complex) resolve against the manifest's `data_kinds:` map (per `33 §4.1`); synthetic implicit-composition names resolve against `SemanticGraph`'s build-time composition index (§11.1). Implicit compositions are NOT persisted on manifest (C9.2 / `33 §6.8`).
2. If the kind is `Simple`, plan against its `SemanticInterface`.
3. If the kind is `Complex`, plan against its pre-materialized `ComposedSemanticInterface` (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.1).
4. **Selected-name membership check.** Every `name` in `request.select` must exist on the resolved (possibly composed) surface. A name not present triggers `PlannerError::SemanticsNotOnSurface` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0507`). The check is per-name against `SemanticsView` on the resolved interface.

Authors using explicit `from:` have opted into a fixed surface; the planner honours it. Synthetic implicit-composition names are addressable and, under the id-first rework, **stable across recompiles** (per [composed-surface-type](composed-surface-type.md) §5.7, since they derive from stable relationship `EntityId`s) — but authors who want an explicitly-controlled name should still declare an explicit `Joinset` with at least one differentiator.

### 11.7 Interaction with `19`'s cross-kind path resolution

`19 §3.4`'s BFS runs at `compile` time to produce `PathSignature` entries on the `ResolvedExprTable` — one per cross-kind reference inside any declared `expr:`. The implicit-Joinset enumeration in [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4 is structurally similar (same `RelationshipGraph`, same neighbor-iteration order, same depth bound, same canonicalization), but it explores **all** paths up to the depth bound — not just the ones referenced by an expression.

Both share: the neighbour-iteration order, the tie-break-by-error policy, the depth bound, and the `RelationshipGraph` infrastructure (built once at `compile`).

Plan-time field-first resolution (§11) is a pure lookup over the pre-built indices — no BFS, no synthesis, no graph traversal. The heavy work is moved to `SemanticGraph` build per [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10 (cascade 2026-05-28: enumeration was previously framed as compile-time; under the lightweight-manifest posture per C9.2 / C10.2 the manifest carries only primitives — `relationships:`, per-DataKind `coverage` — and `SemanticGraph` runs the BFS at build time before plan-time lookup).

## Testing Strategy

- **Invariants:** I4 (determinism — path ambiguity errors, no heuristic tie-break), I5 (compile-time resolution — field-first is pure lookup, no BFS at plan), I8 (manifest carries primitives; graph reconstructs composition index at build).
- **Single-kind fast path:** Tests that `|T| == 1` takes the direct `SemanticInterface` path, that `Request` is treated as `from: Some(T[0])`.
- **Multi-kind lookup:** Tests that `|T| >= 2` triggers constituent-set lookup, that `|candidates| == 1` resolves successfully, that `|candidates| == 0` fires `PLAN_E_0501` (no composition path), that `|candidates| >= 2` fires `PLAN_E_0500` (ambiguous implicit composition), that depth-bound exception fires `PLAN_E_0502` (depth exceeded).
- **Selected-name membership:** Tests that missing names fire `PLAN_E_0507` (SemanticsNotOnSurface), that qualified names (e.g. `orders.total`) resolve correctly on composed surfaces.
- **`Request.from` path:** Tests that explicit `from:` skips field-first, that synthetic implicit-composition names are addressable, that Simple → `SemanticInterface` and Complex → pre-materialized `ComposedSemanticInterface`.
- **Error codes:** `PLAN_E_0500`, `PLAN_E_0501`, `PLAN_E_0502`, `PLAN_E_0507`, `PLAN_E_0508` — see [composition-validation-and-errors](composition-validation-and-errors.md).
- **Behavioral gates:** Tests that field-first resolution produces identical results for explicit and implicit compositions with the same constituent set + canonical form, that `SemanticGraph` build-time enumeration is complete (all ≤4-hop paths enumerated), that plan-time lookup is O(1) against `composition_by_constituent_set`.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
