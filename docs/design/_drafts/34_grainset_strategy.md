---
status: Pending relocation
purpose: Transitional sidecar holding the `GrainsetStrategy` algorithm body extracted from `data-kinds/22_grainset.md §4` (pre-rebase) and extended with the new cross-grain LEFT OUTER JOIN-tree composition mechanism ratified at G-2 (2026-05-03). Pending integration into `apis/34_semstrait_planner.md §<GrainsetStrategy>` when the planner doc opens its Strategy chapter.
extracted-from: data-kinds/22_grainset.md §4 (pre-rebase) — refactored under the post-thirteenth-pass cascade discipline (effective routing units; same-grain pre-merge produces implicit Unionsets; cross-grain composition produces LEFT OUTER JOIN trees via `ComposedSemanticInterface`)
destination: apis/34_semstrait_planner.md §<GrainsetStrategy>
---

# `GrainsetStrategy` algorithm body — extract pending relocation to `34`

> **Transitional document.** This file holds the `GrainsetStrategy` algorithm body extracted from `data-kinds/22_grainset.md §4` (pre-rebase) during the post-thirteenth-pass cascade rebase (2026-05-03), and extended with the new cross-grain LEFT OUTER JOIN-tree composition mechanism ratified at G-2.
>
> Per the architectural boundary ratified in this thread, algorithm bodies belong in the planner doc (`apis/34_semstrait_planner.md`), not in variant chapters (`21`–`24`). Variant chapters carry only conceptual / observable invariants.
>
> When `34` opens its Strategy chapter, this content lifts directly into `34 §<GrainsetStrategy>`. Until then, this sidecar is the canonical reference. Cross-references in `22` (and elsewhere) point at `34 §<GrainsetStrategy>` (forthcoming) — readers needing the algorithm body until that section lands consult this file.
>
> **Vocabulary note.** Content was extracted with refactoring per the post-thirteenth-pass cascade: `ColumnMapping` → `SemanticMapping`, runtime `column_mapping` → `semantic_mapping`, YAML `binding:` block dissolved into `extras.storage:` / `extras.semantic_mapping:` per `32 §4`, `DataKindRef`-based child references retired (children are inlined `Nested*` structs per `32 §3.2`), `RollupPolicy` retained as internal-only planner concept (per G-4 ratification — not authored in V1 YAML). **Cross-grain mechanism is NEW** in this rebase per G-2 ratification.

## 1. Overview — algorithm shape

`GrainsetStrategy` is the planner's resolution strategy for any DataKind whose resolved kind is `Grainset`. The strategy operates on **effective routing units** (one per distinct grain across the Grainset's children — see §2 for assembly) and produces a sub-tree rooted at one of two shapes:

- **Single-unit case** (the common path) — a delegation to the chosen routing unit's own Strategy, optionally wrapped in a rollup transformation (`Project (DATE_TRUNC) + Agg`) when the unit's grain is finer than the Request's.
- **Cross-grain JOIN case** (when no single unit covers the full Request) — a `PlanNode::Join { LeftOuter }` tree assembled from a driver unit and one or more attached units, equi-joined on shared Keys (per `18 §2.5`).

Canonical observable shapes:

```text
# Single-unit case
PlanNode::Project              (final shape — Request fields)
  PlanNode::Agg                (Request's group_by + aggregates; conditional per Measures)
    PlanNode::Filter           (Request filters)
      [PlanNode::Project (DATE_TRUNC) + Agg if grain coarsening needed]
        <chosen unit's Strategy output>

# Cross-grain JOIN case
PlanNode::Project              (final shape — Request fields)
  PlanNode::Agg                (Request's group_by + aggregates)
    PlanNode::Filter           (Request filters above the join)
      PlanNode::Join { LeftOuter, on: <equi-join on shared Keys> }
        <driver unit's Strategy output, rolled up to Request grain if needed>
        <attached unit 1's Strategy output, rolled up to Request grain if needed>
      [+ additional joins for further attached units]
```

`<unit's Strategy output>` is produced by the unit's own Strategy: a single-child unit runs the child's own Strategy (`SimpleStrategy` for `NestedDataset`, `UnionsetStrategy` for `NestedUnionset`, `JoinsetStrategy` for `NestedJoinset`); a multi-child same-grain unit runs `UnionsetStrategy` against the implicit Unionset assembled at compile (per §2). `GrainsetStrategy` does not reach inside unit subplans; it composes them at the top via single-unit delegation or cross-grain JOIN tree.

**Filter placement** is interleaved between layers per the planner's filter-pushdown sub-pass (`34 §7.6`): row-level filters on a Semantics that all units share land below per-unit aggregation; aggregation-referencing filters land above the final `Agg`; cross-grain JOIN filters land above the JOIN tree (push-down into individual unit subplans only when the filter Semantics is unit-local). This sidecar does not pin exact placement.

## 2. Effective routing units — assembly at compile

The Grainset's children are normalized at compile into a list of **effective routing units**, one per distinct `extras.temporal.grain:` value across the child set:

1. **Group children by `grain`**: bucket the Grainset's `body.{datasets, unionsets, joinsets}` by each child's `extras.temporal.grain:` value.
2. **Single-child grain** → the child IS the effective routing unit. Its Coverage of the Grainset's surface is the child's own resolved Coverage (Native / Derived / Metadata / NullFill per `15 §6` for Simple; folded per the child's own resolution for Complex).
3. **Multi-child grain** (≥ 2 children at the same grain) → fold them into an **implicit Unionset** (`mode: All`, **non-strict** Coverage discipline per `23 §3.2`). The implicit Unionset's identity is a content-derived hash-id per `23 §2.1` row A (the id rule, minted per `33 §9.1`; manifest-resident shape and ordering per `33 §6.5`). The unit's Coverage is the per-Semantics fold over the Unionset members (any one provides → unit provides; non-providers NullFill at the Unionset's seam per `23 §4.3`).

The effective routing unit list is sorted by grain coarseness ascending (per `13 §3.2`) to support fast eligibility filtering at plan time. The original child declaration order is preserved for tie-break purposes (G-2b / G-2c).

**Per-unit grain** is the (unique) grain value that defines the bucket. **Per-unit Coverage** is a `BTreeMap<SemanticsName, CoverageVariant>` projecting onto the Grainset's surface.

## 3. `ComposedSemanticInterface` construction — for cross-grain composition

For each Grainset, the planner constructs (at compile) a `ComposedSemanticInterface` per `16 §5` to support cross-grain LEFT OUTER JOIN composition:

1. **`composition_kind`** = `CompositionKind::Grainset` (per `16 §5.3`).
2. **`origin`** = `Origin::Explicit` (Grainset is always author-declared per `16 §5.1`).
3. **`constituents`** = the effective routing unit list (in canonical order per §2).
4. **`interface`** = `UnifiedSemantics` lifted from the Grainset's own `SemanticInterface` per `16 §6`. (The Grainset's surface is the canonical surface; `UnifiedSemantics` is its composition-shape adapter.)
5. **`provenance`** = `FieldProvenance` per-Semantics ownership across units per `16 §7`. For each surface Semantics, the `FieldOwnership` is one of:
   - `Native(unit_ref)` if exactly one unit covers it.
   - `Shared(Vec<unit_ref>)` if multiple units cover it (the planner picks one at routing time per most-covering rule).
   - `NullFill(Vec<unit_ref>)` if no unit covers it (compile error `COMP_E_2202` per `22 §8` — Grainset coverage-completeness check).
6. **`coverage`** = `CompositionCoverage` per-(unit, Semantics) entries per `16 §8`.
7. **`traversed_paths`** = empty (Grainset cross-grain composition uses Keys, not Relationships; cross-grain JOIN-tree details live in `ResolvedGrainset` per `33`, not on `ComposedSemanticInterface`).

Additionally, the planner builds a **JOIN-key index**: for each pair of effective routing units, the set of shared Keys (per `18 §2.5`) that can serve as equi-join conditions. Type compatibility under `13 §7`'s widening rules is verified at compile (`COMP_E_2205` per `22 §8` on mismatch).

## 4. Per-Request resolution — inputs

`GrainsetStrategy::resolve(request, resolved_grainset, manifest)` consumes:

- `request: Request` — the canonical-layer Request with `from`, `select`, `filters`, `group_by`, `aggregations`.
- `resolved_grainset: ResolvedGrainset` — the compile-built artifact carrying:
  - `effective_units: Vec<EffectiveRoutingUnit>` (per §2)
  - `composed_interface: ComposedSemanticInterface` (per §3)
  - `join_key_index: BTreeMap<(UnitId, UnitId), Vec<KeyName>>` (per §3 step 8)
  - `surface: SemanticInterface` (the Grainset's own authored surface)
- `manifest: &SemanticManifest` — for delegating to constituent Strategies.

## 5. Step — extract Request grain

Determine the Request's effective grain (`request_grain: Option<Grain>`):

```text
EXTRACT_REQUEST_GRAIN(request, surface):
  # (a) Explicit grain selector on a temporal Dimension in group_by
  for grouped_dim in request.group_by:
    if grouped_dim has explicit grain selector g:
      return Some(g)
  # (b) Temporal filter narrowing the grain implicitly (DATE_TRUNC literal arg)
  for filter in request.filters:
    if filter references a temporal Dimension via DATE_TRUNC(_, g_literal):
      return Some(g_literal)
  # (c) Bare temporal grouping → finest grain on the surface's temporal Dimension
  for grouped_dim in request.group_by:
    if grouped_dim is a temporal Dimension on surface:
      return Some(finest_grain(grouped_dim))
  # (d) No temporal grouping → no grain constraint
  return None
```

When `request_grain` is `None`, the eligibility filter (§6 step 1) admits all units; cost-based selection (§6 step 4) defaults to the coarsest unit (smallest scan).

## 6. Step — eligibility, single-unit-vs-cross-grain dispatch

```text
GrainsetStrategy::resolve(request, resolved_grainset, manifest):
  request_grain = EXTRACT_REQUEST_GRAIN(request, resolved_grainset.surface)
  request_semantics = extract_referenced_semantics(request)

  # Step 1 — grain eligibility
  grain_eligible_units = [
    u for u in resolved_grainset.effective_units
    if request_grain is None or u.grain ≤ request_grain
  ]
  if grain_eligible_units.is_empty():
    return Err(PLAN_E_2201 GrainsetNoMatchingUnitByGrain)

  # Step 2 — coverage check per unit (Native/Derived only)
  per_unit_covers = {
    u: subset of request_semantics that u covers (Native or Derived)
    for u in grain_eligible_units
  }

  # Step 3 — single-unit fast path
  fully_covering = [u for u in grain_eligible_units if per_unit_covers[u] == request_semantics]
  if !fully_covering.is_empty():
    chosen = pick_most_covering_with_decl_order_tiebreak(fully_covering, per_unit_covers)
    sub_request = narrow_request_for_unit(request, chosen)
    sub_plan = dispatch_strategy(manifest, chosen).resolve(sub_request, ...)
    return Ok(wrap_for_rollup(sub_plan, chosen.grain, request_grain, chosen.shape))

  # Step 4 — cross-grain JOIN composition
  return resolve_cross_grain(request, request_grain, request_semantics,
                             grain_eligible_units, per_unit_covers, resolved_grainset, manifest)
```

`pick_most_covering_with_decl_order_tiebreak` returns the unit with the largest `|per_unit_covers[u]|`; ties break by smallest declaration-order index (per G-2b). When all `per_unit_covers[u]` are equal in the fully-covering set, the first-declared unit wins.

## 7. Step — cross-grain JOIN composition

```text
resolve_cross_grain(request, request_grain, request_semantics,
                    grain_eligible_units, per_unit_covers, resolved_grainset, manifest):
  # Step 1 — driver selection: most-covering with name-order tie-break (G-2b; id-first rework)
  driver = pick_most_covering_with_name_order_tiebreak(grain_eligible_units, per_unit_covers)
  covered = per_unit_covers[driver].clone()
  in_tree_units = [driver]

  # Step 2 — greedy attached selection in declaration order
  for u in grain_eligible_units (in declaration order, skipping driver):
    needed = per_unit_covers[u] - covered
    if needed.is_empty():
      continue                                # u contributes nothing new
    # u must share at least one Key with some unit already in the tree
    join_key = find_join_key(u, in_tree_units, resolved_grainset.join_key_index)
    if join_key is None:
      # u contributes new Semantics but no equi-join path exists
      # — defer error decision to step 3 (may be uncoverable)
      continue
    in_tree_units.push((u, join_key))
    covered = covered ∪ per_unit_covers[u]
    if covered == request_semantics:
      break                                   # all Semantics covered

  # Step 3 — uncovered check
  uncovered = request_semantics - covered
  if !uncovered.is_empty():
    return Err(PLAN_E_2202 GrainsetSemanticsNotCoverableByJoin {
      grainset, request_semantics, uncovered,
    })

  # Step 4 — build the JOIN tree
  driver_subplan = build_unit_subplan(driver, request, request_grain, manifest)
  current_tree = driver_subplan
  for (attached, join_key) in in_tree_units[1..]:
    attached_subplan = build_unit_subplan(attached, request, request_grain, manifest)
    current_tree = PlanNode::Join {
      join_type: LeftOuter,
      left: current_tree,
      right: attached_subplan,
      on: build_equi_join_predicate(join_key, current_tree, attached_subplan),
    }

  # Step 5 — wrap with Filter + Agg + Project per Request shape
  return Ok(wrap_with_request_pipeline(current_tree, request))
```

`build_unit_subplan(unit, request, request_grain, manifest)`:

1. Narrow the Request to only the Semantics the unit covers (drops uncovered ones; the JOIN tree picks them up from other units).
2. Dispatch to the unit's own Strategy (`SimpleStrategy` for single-Dataset units, `UnionsetStrategy` for same-grain implicit Unionsets, `UnionsetStrategy` / `JoinsetStrategy` for `NestedUnionset` / `NestedJoinset` children).
3. Wrap with rollup transformation (`Project (DATE_TRUNC) + Agg`) if `unit.grain < request_grain` AND the rollup is shape-legal per `17 §4`.

`build_equi_join_predicate(join_key, left_subtree, right_subtree)` constructs `PhysicalExpr::BinaryOp { op: Eq, left: Column(join_key on left), right: Column(join_key on right) }`. Multi-Key shared joins use a conjunction.

`wrap_with_request_pipeline(tree, request)` adds the standard final pipeline: `Filter` (Request filters), then `Agg` (Request group_by + aggregates), then `Project` (Request select shape).

## 8. Step — rollup wrapper emission

```text
wrap_for_rollup(sub_plan, unit_grain, request_grain, unit_shape):
  if request_grain is None or unit_grain == request_grain:
    return sub_plan          # no rollup needed
  if unit_grain > request_grain:
    panic!("invariant violation: ineligible unit reached rollup wrapper")
  # unit_grain < request_grain — rollup needed
  legal = SHAPE_ROLLUP_LEGAL(unit_shape, unit_grain, request_grain)
  if !legal.is_legal():
    return Err(legal.error_code())   # PLAN_E_2203 SnapshotRollupWithoutPin or PLAN_E_2204 SCDRollupWithoutAsOf
  return PlanNode::Agg {
    group_by: rollup_group_by(request, request_grain),
    aggregates: rollup_aggregates(request),
    input: PlanNode::Project {
      expressions: rollup_projections(request, request_grain),
      input: sub_plan,
    },
  }
```

`SHAPE_ROLLUP_LEGAL(shape, from_grain, to_grain)` consults `17`'s shape × grain rollup matrix:

- `Timeseries` / `Events` — always legal when `from_grain ≤ to_grain`. Rollup via DATE_TRUNC + Agg.
- `Snapshot` — legal only when (a) a pin policy is declared per `17`, OR (b) `from_grain == to_grain` (no rollup needed). The internal `RollupPolicy::PinOnly` (per G-4) is the default for `Snapshot` units when no pin policy is declared. Without pin policy AND `from_grain < to_grain` → `PLAN_E_2203 GrainsetSnapshotRollupWithoutPin`.
- `Scd` — legal only when the Request carries an as-of anchor. Without as-of AND rollup needed → `PLAN_E_2204 GrainsetSCDRollupWithoutAsOf`.

## 9. Internal `RollupPolicy` (per G-4 ratification, 2026-05-03)

```rust
#[non_exhaustive]
pub enum RollupPolicy {
    /// Default for shapes that roll freely (Timeseries / Events).
    /// Behavior derived from `17`'s shape rules.
    ShapeDefault,

    /// Default for `Snapshot` and `Scd` shapes. Forbids rollup unless
    /// `17`'s pin policy or as-of anchor mechanism applies.
    PinOnly,

    /// Reserved for future fine-tuning. Forces selection of the
    /// finest-grain unit even when a coarser unit would be cheaper.
    /// Useful for data-quality workflows. Not the default for any shape.
    PreferFinest,
}
```

V1 carries this enum internally; `RollupPolicy` is **not authored in YAML** and not surfaced to users. Per-unit policy is derived at compile from the unit's `extras.temporal.kind:`:

- `Timeseries` / `Events` → `ShapeDefault`.
- `Snapshot` / `Scd` → `PinOnly` (default; pin policies / as-of anchors per `17` may relax).

Future `34` extension may surface `RollupPolicy` as an opt-in fine-tuning hook for advanced authors.

## 10. Algorithm sketch — full pseudocode

```text
GrainsetStrategy::resolve(request, resolved_grainset, manifest):
  request_grain = EXTRACT_REQUEST_GRAIN(request, resolved_grainset.surface)        # §5
  request_semantics = extract_referenced_semantics(request)

  grain_eligible_units = filter_grain_eligible(resolved_grainset.effective_units, request_grain)
  if grain_eligible_units.is_empty():
    return Err(PLAN_E_2201)

  per_unit_covers = compute_per_unit_coverage(grain_eligible_units, request_semantics)

  fully_covering = [u for u in grain_eligible_units if per_unit_covers[u] == request_semantics]
  if !fully_covering.is_empty():
    # Single-unit fast path (§4.2 of `22`)
    chosen = pick_most_covering_with_decl_order_tiebreak(fully_covering, per_unit_covers)
    sub_request = narrow_request_for_unit(request, chosen)
    sub_plan = dispatch_strategy(manifest, chosen).resolve(sub_request, ...)
    return Ok(wrap_for_rollup(sub_plan, chosen.grain, request_grain, chosen.shape))

  # Cross-grain JOIN path (§4.3 of `22`; §7 here)
  return resolve_cross_grain(request, request_grain, request_semantics,
                             grain_eligible_units, per_unit_covers, resolved_grainset, manifest)
```

## 11. Determinism guarantees

- Effective routing unit ordering is canonical (sorted by grain coarseness ascending; child declaration order within same grain).
- Driver selection is deterministic: most-covering with name-order tie-break (G-2b; id-first rework, STATUS item U).
- Attached unit ordering is declaration order (G-2c) — strict; no greedy-by-coverage-delta in V1.
- Same-grain pre-merge produces implicit Unionsets in declaration order per `23 §3.1`.
- All compile-time index lookups (`semantics_to_covering_units`, `join_key_index`) are deterministic builds from declaration-ordered inputs.

Identical `(SemanticManifest, Request)` → identical chosen routing path → identical `PlanNode` tree (per I4).

---

**End of sidecar.** This document retires when `apis/34_semstrait_planner.md §<GrainsetStrategy>` lifts the content into the planner-doc canon. At that point this file is deleted; `22`'s cross-refs update from `_drafts/34_grainset_strategy.md` to `34 §<GrainsetStrategy>`.
