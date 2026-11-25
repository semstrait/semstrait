---
id: 22-grainset-routing
status: ratified
version: 2.1
impl: drift
---

# Grainset — Plan-Time Routing

## 4. Plan-Time Observable Behavior

A Grainset's plan-time observable behavior is the realization of `GrainsetStrategy::resolve` (algorithm body in `34 §<GrainsetStrategy>`, parked in `_drafts/34_grainset_strategy.md` pending `34` drafting). This section ratifies the observable invariants an author or consumer can rely on; the algorithm body is referenced only for derivation.

### 4.1 Effective routing units

An **effective routing unit** is the planner's routing primitive — one per distinct child `extras.temporal.grain:` value:

- **Single-child case** — exactly one child sits at that grain → the unit is that child (NestedDataset / NestedUnionset / NestedJoinset, resolved per its own Strategy).
- **Multi-child case** — two or more children share that grain → the unit is the implicit Unionset folded at compile per §3.3. The unit's coverage is the per-Semantics fold over the Unionset members (any one provides → unit provides; non-providers NullFill at the seam per `23 §4.3`).

The Grainset's effective routing units are **one per distinct grain** value across the child set (V1 ≥ 2 per §5.2). Routing decisions (eligibility, driver selection, JOIN-tree construction) operate at the routing-unit granularity, not at the raw-child granularity.

### 4.2 Single-unit delegation (simple case)

When a Request's Semantics set is fully covered by exactly one effective routing unit (after grain-eligibility filtering), `GrainsetStrategy` delegates to that unit's own Strategy and wraps the result in a rollup wrapper if needed:

1. Identify grain-eligible units (units whose `grain ≤ request.grain` per `13 §3.2`'s coarseness order).
2. Filter to units that fully cover the Request's Semantics set (Coverage `Native` / `Derived` per §3.2).
3. If exactly one such unit (or multiple with the most-covering equal — name-order tie-break per G-2b) → that's the chosen unit.
4. Run the chosen unit's Strategy on a unit-narrowed Request.
5. Wrap with `Project (DATE_TRUNC) + Agg` if `chosen_unit.grain` is finer than `request.grain` AND the rollup is shape-legal per `17 §4`.

The "splice" pattern from pre-cascade `22 §10.5` is preserved: there is no `PlanNode::Grainset`. The chosen unit's subplan is the Grainset's contribution; the rollup wrapper (when needed) is built from shared `PlanNode::{Project, Agg}` nodes.

### 4.3 Cross-grain JOIN delegation (partial-coverage case)

When no single grain-eligible unit fully covers the Request's Semantics set, the planner builds a LEFT OUTER JOIN tree using the `ComposedSemanticInterface` (§3.4). Per G-2a / G-2b / G-2c (2026-05-03 ratification).

**Algorithm**:

1. Filter to grain-eligible units.
2. Identify the **driver** = unit covering the largest subset of the Request's Semantics (Native / Derived). Tie-break: name order (G-2b).
3. Greedily add **attached** units in name order (G-2c — deterministic over greedy-by-coverage-delta). Each attached unit must:
   - Cover ≥ 1 Semantics not yet covered by driver-or-already-attached units.
   - Share at least one Key (per `18 §2.5`) with the driver (or with an already-attached unit reachable through the JOIN tree).
4. After all attached units are added, every Request Semantics must be covered. If any remains uncovered → `PLAN_E_2202 GrainsetSemanticsNotCoverableByJoin` (§9).
5. Emit the JOIN tree:
   - Driver as the LEFT side.
   - Each attached unit added via `PlanNode::Join { join_type: LeftOuter }` (per G-2a) on the equi-join condition over shared Keys.
6. Above the JOIN tree, emit the Request's `Project + Agg + Filter` per the standard observable pipeline.

Expression placement at each level and at the top follows the `ExprLayer` (`19 §6.0`): `Scalar` Dimensions/Keys materialise pre-`Agg` (and feed `GROUP BY`); `Aggregate` Measures lift into the per-unit `Agg` and re-aggregate at the top; `PostAggregate` Metric residuals land above the final `Agg`. Rollup (`DATE_TRUNC + Agg`) and cross-grain re-aggregation are gated by **function-derived** `Additivity` in v1 (`19 §6.5`, `14a §3.6.2`).

**Observable plan shape** (worked example in §10.4):

```
PlanNode::Project       (final shape — Request fields)
  PlanNode::Agg         (Request's group_by + aggregates; conditional per Measures)
    PlanNode::Filter    (Request filters above the join)
      PlanNode::Join { LeftOuter, on: campaign_id = campaign_id }
        <driver unit's subplan, rolled up to Request grain if needed>
        <attached unit 1's subplan, rolled up to Request grain if needed>
      [+ additional joins for further attached units]
```

Each unit's subplan is produced by its own Strategy (the same Strategy invoked in the single-unit case). For implicit-Unionset units (§4.1 multi-child case), that's `UnionsetStrategy` per `_drafts/34_unionset_strategy.md`.

### 4.4 Rollup mechanics — shape-gated per `17`; `RollupPolicy` is internal-only

When a Request's grain is coarser than a routing unit's grain, the chosen unit's subplan is wrapped with a rollup transformation:

- **`Timeseries` / `Events` shapes** — bucket-then-aggregate via `Project (DATE_TRUNC(axis, request_grain))` + `Agg`. Always legal.
- **`Snapshot` shape** — pin policy required (per `17`); without a pin policy, rolling a snapshot up across periods is semantically unsafe. Internal `RollupPolicy::PinOnly` is the planner's default for `Snapshot` units when no pin policy is declared.
- **`SCD` shape** — as-of anchoring required (per `17`); the Request must carry an as-of timestamp. Without one, rollup fails.

The `RollupPolicy { ShapeDefault, PinOnly, PreferFinest }` enum is implemented in the planner but NOT authored in V1 YAML (per G-4). Default behavior:

- `Snapshot` units → `PinOnly` (no rollup unless pin policy declared per `17`).
- `Timeseries` / `Events` units → `ShapeDefault` (free rollup via DATE_TRUNC + Agg).
- `SCD` units → `PinOnly` semantically (pin to as-of anchor).

Future fine-tuning hooks (e.g. exposing `PreferFinest` for data-quality workflows where the finer source is preferred even at higher cost) may surface in `34`.

### 4.5 Coverage-driven pruning

Routing units that NullFill every Semantics named in the Request contribute nothing and are pruned before JOIN-tree construction. Pruning collapses the candidate set for both single-unit delegation (§4.2) and cross-grain composition (§4.3); no advisory is emitted (the prune is transparent because Grainset's routing model is "pick covering units", not "merge all units").

## 6. Grain Interaction

### 6.1 Per-child grain authoring (mandatory per SR-E-8)

`grain` lives on each leaf descendant's `LeafExtras` (per `32 §4`); Grainset itself does not author `grain` (SR-E-7). Per SR-E-8 (`18 §3.4`), every Grainset child MUST author its own `extras.temporal.grain:` explicitly — there is no inheritance-from-ancestor mechanism. Validated post-parse at validate stage; diagnostic `validate.grainset-child-grain-required` per `26 §6`.

The Grainset's "grain axis" is the **set** of distinct grain values across children (≥ 2 per §5.2). It is not an authored field on the Grainset; it is derived from children's `extras.temporal.grain:` declarations.

### 6.2 Plan-time grain consultation

Two observable invariants involve `grain`:

- **Grain eligibility** (§4.1). For a Request at grain `G`, an effective routing unit is grain-eligible iff its grain `≤ G` per `13 §3.2`'s coarseness order.
- **Rollup wrapping** (§4.4). When a chosen unit's grain is finer than the Request's grain, the planner emits a rollup wrapper (`Project (DATE_TRUNC) + Agg`) shape-gated per `17 §4`.

If no routing unit has grain ≤ Request grain → `PLAN_E_2201 GrainsetNoMatchingUnitByGrain` (§9). Authors should add a coarser routing unit or restructure the Request.

### 6.3 Scope boundary with `13` and `17`

- `13 §5` ratifies the `Grain` enum and its coarseness order.
- `17` ratifies the `TemporalShape × Grain` legality matrix (which rollups are shape-legal).
- `17` ratifies the per-shape pin / anchor mechanics.

## Testing Strategy

- Effective routing units = one per distinct grain; single-unit delegation + rollup wrapper (shape-gated per 17 §4); cross-grain JOIN tree emission; Coverage-driven pruning.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
