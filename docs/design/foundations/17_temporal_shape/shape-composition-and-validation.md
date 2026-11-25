---
id: 17-shape-composition-and-validation
status: partially ratified
version: 1.0
impl: partially implemented
---

# Shape Composition and Validation

This document specifies shape-gated composition rules for Unionset, Grainset, and implicit composition (rollup vs as-of anchoring), along with the complete validate/compile/plan Precondition catalog (VALID_E_17xx, COMP_E_17xx, PLAN_E_17xx, PLAN_W_17xx) and the DEFERRED-items roster.

## 8. Shape-gated composition rules

`16 §11` ratifies the field-first implicit-composition algorithm. `17 §8` refines that algorithm with shape-aware preconditions: which shape combinations admit implicit composition, which require explicit anchoring, which admit only as-of composition.

### 8.1 Shape compatibility across `Unionset` branches

A `Unionset`'s branches must agree on `TemporalShape`. Compatibility matrix:

| Branch A shape | Branch B shape | Compatible? | Notes |
|---|---|---|---|
| `Timeseries { grain: G_A }` | `Timeseries { grain: G_B }` | Iff `G_A == G_B` | Different grains inside a Unionset are semantically incoherent — the union produces a mixed-grain stream. Different grains mean use a `Grainset` instead. |
| `Events` | `Events` | Yes | `occurred_at_dim` must unify (global identity per `11 §3`); the union inherits it. |
| `Snapshot { cadence: C_A }` | `Snapshot { cadence: C_B }` | Iff `C_A == C_B` (or both `None`) | Divergent cadences inside a union produce unaligned snapshots. |
| `Scd { subtype: T }` | `Scd { subtype: T' }` | Iff `T == T'` | Subtype mismatch = structurally different history encoding; cannot be unioned mechanically. |
| Any shape A | shape-unclassified | No | `COMP_E_1710 UnionsetShapeConflict`. Every branch must agree. |
| Shape A | Shape B (different variants) | No | Same code. |

The only exception: a `Unionset` whose branches are **all** shape-unclassified is a shape-unclassified `Unionset`. No warnings, no shape-aware behavior, vanilla composition per `16`.

### 8.2 Shape constraints on `Grainset` levels

A `Grainset` composes children at different grains per `22 §…`. Shape constraints:

- All child shapes must be of the **same variant family**. Specifically: all `Timeseries`, or all `Snapshot`, or all shape-unclassified. Mixing variants within a `Grainset` is `COMP_E_1711 GrainsetShapeMixed`.
- `Timeseries` children within one `Grainset` must have **distinct `grain` values** that each appear in the `Grainset`'s declared `levels:` per `22 §…`.
- `Snapshot` children within one `Grainset` must agree on cadence or have nested cadences forming a coarseness chain (e.g. daily snapshots at one level and monthly snapshots at another).
- `Events` children in a `Grainset` are forbidden — `Events` have no fixed grain per §4.1. Use a pre-aggregated `Timeseries` for each rollup level instead. `COMP_E_1712 EventsInGrainset`.
- `Scd` children in a `Grainset` are forbidden — `Scd` is not a grain-layered concept. `COMP_E_1712 EventsInGrainset` (same variant, different bound carrier — the error struct discriminates via `actual_shape`).

### 8.3 Shape constraints on implicit composition (`16 §11`)

`16 §11.4`'s `RELATIONSHIP_BFS` discovers a path connecting owning DataKinds. `17 §8.3` adds per-edge shape-preconditions:

1. **Edges between two shape-unclassified DataKinds**: pass through to `16`'s rules unchanged.
2. **Edges between one shape-classified and one shape-unclassified DataKind**: pass through; the classified side contributes its shape-aware behavior, the unclassified side does not.
3. **Edges that would require `AsOf` traversal** (per §5.2 matrix): in Round 1, `AsOf` is DEFERRED (§10). The planner **does not** silently synthesize an `Inner` join in place of a needed `AsOf`; the Request fails with `PLAN_E_1751 AsOfNotAvailableInRound1`. Authors needing cross-shape composition in Round 1 must either:
   - Pin the `as_of` via a `filter` on the SCD side (`filter: is_current = TRUE`, effectively degrading `Scd::Type2` to a `Type1`-style current-row view); or
   - Declare an explicit `Joinset` with a Round-1-supported `JoinType` (`Inner / Left / Right / Full`), accepting that the join will be cartesian across SCD validity windows without the anchoring the Request needs.
4. **Edges between two `Scd` kinds with different subtypes**: no automatic as-of propagation; requires explicit `Joinset`. `PLAN_E_1752 MultiScdImplicitCompositionForbidden`.
5. **Multi-hop paths that mix shape-dependent and shape-neutral edges**: the shape-dependent edges gate the path's overall legality; shape-neutral edges pass through.

### 8.4 Rollup vs as-of anchoring across composition

The principle: **rollup** (coarsening along `Grain`) and **as-of anchoring** (pinning to one row per entity via `Request.temporal.as_of`) are distinct operations that apply to different shapes:

- `Timeseries` / `Events`: rollup applies; as-of anchoring does not (they are naturally time-series).
- `Snapshot`: as-of anchoring applies (pick one snapshot); rollup applies *after* anchoring (compose snapshots across a range).
- `Scd`: as-of anchoring applies (pick one row per entity per the §6.2 rules); rollup does not apply at the SCD layer (SCD is grain-free per §4.1).

A composition that mixes shapes needs **all** their anchoring / rollup operations satisfied. The planner's shape-aware composition pass (DEFERRED per §10) runs:

1. For each shape-classified constituent, compute its required anchoring (from `Request.temporal` per §6) and rollup (from the Request's `group_by` per `11 §6`).
2. For each edge, verify the per-edge shape legality per §8.3.
3. For each constituent, apply anchoring before the composition-level aggregate; apply rollup after.
4. Emit the composed `PlanNode` tree.

This pass is ratified at the algorithm-level here; implementation is DEFERRED.

### 8.5 `Scd` require as-of anchoring before wide composition

A specific instance worth calling out: `Scd::Type2 / Type4 / Type5 / Type6` DataKinds must be **anchored** (either via `Request.temporal.as_of` or via a filter pinning to `current_flag_dim = TRUE`) before composition widens their surface via implicit `Relationship` traversal. Without anchoring, each SCD row is one validity-window snapshot of an entity, and the composed surface would carry per-entity fanout equal to the number of validity windows. The planner emits `PLAN_W_1750 ScdWideCompositionWithoutAnchor` when it detects this shape; it proceeds (the query is legal; it may simply not be what the author intended).

In Round 2 (planner implementation), the rule hardens: implicit composition over a history-preserving SCD kind **without** an anchor (as-of or filter) will be an `Error` (`PLAN_E_1753 ScdWideCompositionWithoutAnchor`). The Round-1 advisory is the softer precursor, giving authors the chance to observe the shape before the rule tightens.

---

## 9. `validate` / `compile` / `plan` Preconditions

All error codes are in the `17NN` block per the [CONTRADICTION-FOUND] block and Q-TEMPORAL-001.

### 9.1 `validate` Preconditions (structural; accumulate; no catalog access)

Run by `validate` per `10 §3.3`. These check the static shape of `temporal_shape:` blocks independent of name resolution.

| ID | Code | Rule | Failure condition |
|---|---|---|---|
| TS-V1 | `VALID_E_1700` | `TemporalShape` authored on a `SimpleDataKind`, not a Complex one. | `temporal_shape:` appears inside a `Unionset` / `Grainset` / `Joinset` declaration. Emits `TemporalShapeOnComplexDataKind { data_kind }`. |
| TS-V2 | `VALID_E_1701` | `TemporalShape` variant discriminator is one of the ratified names. | Unknown variant key (e.g. `temporal_shape: { bitemporal: { ... } }`). Per-subtype check runs at compile. |
| TS-V3 | `VALID_E_1702` | `ScdType` discriminator is one of `type_0`–`type_6`. | Unknown subtype key. |
| TS-V4 | `VALID_E_1703` | Each variant's **required** payload fields are present. | `Timeseries` without `occurred_at_dim` or `grain`; `Events` without `occurred_at_dim`; `Snapshot` without `snapshotted_at_dim`; `Scd::Type2` without `valid_from_dim` or `valid_to_dim`; etc. |
| TS-V5 | `VALID_E_1704` | Disallowed fields for the declared variant are absent. | `Timeseries { occurred_at_dim, grain, snapshotted_at_dim }` mixes keys across variants. |
| TS-V6 | `VALID_E_1705` | `Timeseries.grain` is a valid `Grain` variant (`minute`–`year`). | Unknown token. |
| TS-V7 | `VALID_E_1706` | Only one `temporal_shape:` per `SimpleDataKind`. | Duplicate declaration (typically a YAML authoring mistake). |

### 9.2 `compile` Preconditions (catalog / registry / cross-reference; fail-fast)

Run by `compile` per `10 §3.4`. These require resolved Semantics names and type context.

| ID | Code | Rule | Failure condition |
|---|---|---|---|
| TS-C1 | `COMP_E_1701` | Identifying Dimension exists on the declaring DataKind's interface. | `occurred_at_dim` / `snapshotted_at_dim` / SCD window Dim not in the DataKind's declared interface. |
| TS-C2 | `COMP_E_1702` | Identifying Dimension has `DimensionType::Temporal`. | Dimension exists but is Categorical / Metadata / etc. |
| TS-C3 | `COMP_E_1703` | Identifying Dimension's `DataType` is time-typed (`Date`, `Time`, `Timestamp`, `Interval`). | Dim has `String` / `Long` / ... |
| TS-C4 | `COMP_E_1704` | `Timeseries.grain ∈ occurred_at_dim.grains:`. | Grain not in the Dimension's declared grain list. |
| TS-C5 | `COMP_E_1705` | `valid_from_dim.DataType == valid_to_dim.DataType` for SCD `Type2` / `Type5` / `Type6`. | Strict type mismatch. |
| TS-C6 | `COMP_E_1706` | `current_flag_dim.DataType == Boolean`. | Non-boolean flag Dimension. |
| TS-C7 | `COMP_E_1707` | SCD `Type4.history_data_kind_ref` / `Type5.mini_dim_ref` resolves to a declared top-level `DataKind`. | Unknown reference. |
| TS-C8 | `COMP_E_1708` | `Snapshot.cadence`, when declared, is a `Grain` member of `snapshotted_at_dim.grains:`. | Cadence not in the Dimension's grain list. |
| TS-C9 | `COMP_E_1709` | Composed-surface shape compatibility (§8.1, §8.2). | Unionset branches disagree; Grainset mixes shapes; Events / Scd in a Grainset; etc. |
| TS-C10 | `COMP_E_1710` | `UnionsetShapeConflict` (specific shape per §8.1). | See §8.1 matrix. |
| TS-C11 | `COMP_E_1711` | `GrainsetShapeMixed` (specific to §8.2). | See §8.2. |
| TS-C12 | `COMP_E_1712` | `EventsInGrainset` / SCD-in-Grainset. | §8.2. |

### 9.3 `plan` Preconditions (Request / Session-Context scoped; fail-fast)

Run by `plan` per `10 §3.5`. These depend on the Request plus Session state.

| ID | Code | Rule | Failure condition |
|---|---|---|---|
| TS-P1 | `PLAN_E_1710` | `TimeseriesSubGrainRequested`. | Request group-by grain is finer than the `Timeseries.grain`. |
| TS-P2 | `PLAN_E_1711` | `SnapshotSubCadenceRequested`. | Request group-by grain finer than `Snapshot.cadence`. |
| TS-P3 | `PLAN_E_1712` | `GrainsetDowngradeRequested`. | Request asks for grain finer than any child's declared grain. |
| TS-P4 | `PLAN_E_1720` | `AsOfShapePairIllegal`. | §5.2 matrix violation. |
| TS-P5 | `PLAN_E_1721` | `AsOfShapeUnclassified`. | One or both sides of the requested `AsOf` traversal is shape-unclassified. |
| TS-P6 | `PLAN_E_1722` | `AsOfReverseTraversalForbidden`. | §5.3. |
| TS-P7 | `PLAN_E_1730` | `SnapshotAsOfNoCoveringSnapshot`. | `Request.temporal.as_of` precedes the earliest available snapshot. |
| TS-P8 | `PLAN_E_1751` | `AsOfNotAvailableInRound1`. | Implicit composition requires `AsOf` but the planner DEFERRED state forbids it. |
| TS-P9 | `PLAN_E_1752` | `MultiScdImplicitCompositionForbidden`. | §8.3 bullet 4. |
| TS-P10 | `PLAN_E_1753` | `ScdWideCompositionWithoutAnchor` (Round-2 promotion of `PLAN_W_1750`). | §8.5 Round-2 rule. **Not active in Round 1** — reserved. |

### 9.4 `plan` advisories (warnings; planning proceeds)

| Code | Variant | Fires when |
|---|---|---|
| `PLAN_W_1701` | `ScdHistoryShapeUnexpected` | `Scd::Type4.history_data_kind_ref` resolves to a kind whose own shape is not `Scd::Type2 { .. }` (§2.4 rule 8). |
| `PLAN_W_1702` | `ShapeAwareRequestOnUnclassifiedDataKind` | `Request.temporal` block provided targeting a shape-unclassified DataKind (§3.3). |
| `PLAN_W_1730` | `ScdAsOfIgnored` | `Request.temporal.as_of` on a `Scd::Type0 / Type1 / Type3` kind — the subtype has no history to pick from. |
| `PLAN_W_1731` | `ScdCurrentRowHeuristic` | Default-current lookup on an SCD kind without a ratified current-row signal (§6.3). |
| `PLAN_W_1732` | `ScdRangeMultipleRows` | `Request.temporal.range` on an SCD kind without anchor; multiple rows per entity possible (§6.4). |
| `PLAN_W_1740`–`PLAN_W_1744` | `AdditiveOnSnapshot` / `AdditiveOnScdHistoryPreserving` / `SemiAdditiveUnsafeAxisAbsentFromShape` / `NonAdditiveOnTimeseriesGrainMismatch` / `SemiAdditiveOnEvents` | §7.3 roster. |
| `PLAN_W_1750` | `ScdWideCompositionWithoutAnchor` | §8.5 — history-preserving SCD in an implicit composition without anchor. |

### 9.5 Error-severity summary

All `VALID_E_17xx`, `COMP_E_17xx`, `PLAN_E_17xx` are `Severity::Error`; they fail their respective stages per `30 §7`. All `PLAN_W_17xx` are `Severity::Warning`; planning proceeds, advisories flow through to the `Diagnostic` list. No `Severity::Info` codes are reserved for `17` in Round 1.

### 9.6 Code-allocation governance

Per `30 §6.4` and the [CONTRADICTION-FOUND] block:

1. New `17`-subsystem variants append to the next free number in the 17NN block.
2. The 17NN block is reserved for temporal-shape concerns only; unrelated subsystems do not encroach.
3. If Option 2 reconciliation (§[CONTRADICTION-FOUND]) is adopted, every `*_E_17NN` / `*_W_17NN` reference in this doc is re-homed to Option 2 codes via a single find-and-replace; error semantics are preserved.

---

## 10. DEFERRED items — explicit roster

The closed list of behaviors whose **vocabulary and model surface** `17` ratifies but whose **implementation** lands in a later milestone.

| ID | Item | Ratified in | Implementation deferred to |
|---|---|---|---|
| D1 | `JoinType::AsOf` variant and `AsOfAnchor` payload shape | §5.1 | Planner / adapter milestone; requires `PlanNode::Join` extension in `35`, per-adapter emission in `registry/temporal_shape_mapping.md`. |
| D2 | `AsOf` anchor selection per shape pair (§5.2 matrix) | §5.2 | Same as D1. |
| D3 | `Joinset` per-traversal `AsOf` override | §5.5 | Q-TEMPORAL-003 closed (Option B, 2026-04-28); admitted as a post-implicit additive MINOR after milestone-1 ships implicit synthesis. |
| D4 | `Request.temporal` block (vocabulary + per-shape semantics) | §6.1–§6.4 | Planner milestone; `34 §…`. |
| D5 | `Request.temporal` across composed surfaces with heterogeneous shapes | §6.5 | Q-TEMPORAL-004; gated on composed-shape-resolution algorithm in `34`. |
| D6 | `PLAN_E_1753 ScdWideCompositionWithoutAnchor` (hard error version) | §8.5 | Round-2; currently `PLAN_W_1750` advisory. |
| D7 | `Scd::Type4` as-of join traversal | §5.2 `[TD-SCD-TYPE4-ASOF]` | Gated on composition-surface integration in `16`; requires a structural extension to treat `history_data_kind_ref` as a composition hint. |
| D8 | Snapshot cadence-rollup reducer policies (first / last / average / configurable) | §4.1 | `25 §…` per-strategy catalog. |
| D9 | `Scd::Type5` mini-dim outrigger planner behavior (join onto `mini_dim_ref` with as-of anchor) | §2.2 | Requires both D1 AsOf and D7 coordination; same milestone. |
| D10 | `Scd::Type6` current-value duplication detection (`current_value_dim` reconciliation with history rows) | §2.2 | Planner milestone; advisory emission similar to `PLAN_W_1731`. |
| D11 | Adapter-side SQL emission for `AsOf` joins (Spark `AS OF JOIN`, DuckDB `LATERAL`, DataFusion `RANGE JOIN`, etc.) | §5.4 | `registry/temporal_shape_mapping.md` + per-adapter `36` docs. |
| D12 | `additivity_confirmed: true` suppression field for §7.3 advisories | §7.4 | `11 §7.3` extension; cross-doc. |
| D13 | Canonical resolution of default-current when `current_flag_dim` is absent on an SCD kind (i.e. ratified replacement for the §6.3 heuristic) | §6.3 | Q-TEMPORAL-005; planner milestone. |
| D14 | Non-temporal `TemporalShape` variants (e.g. bi-temporal with `valid_time` + `system_time`) | §1.2 | Post-Round-1; tracked as `[TD-BITEMPORAL]`. |

**DEFERRED scope boundary.** Items outside this roster are **not deferred**: they are either (a) ratified-and-implemented in the existing codebase (all the existing `TemporalHistorization` parse / validate behavior continues to work; rename to `TemporalShape` is a pure refactor), or (b) out of scope per `00 §10` (live manifest hot-reload, row-level security, etc.).

---

## 13. Cross-References

- `00 §4.1` — `TemporalShape` row (ratified vocabulary referenced here); `AsOf` `JoinType` row (deferred variant ratified here); `Additivity` row (independent-axes interaction per §7).
- `00 §9` — I1 (canonical layer), I4 (determinism), I5 (compile-time resolution), I8 (SemanticManifest planner-complete), I10 (non-exhaustive), I12 (diagnostics).
- `11 §5` — shape-vs-resolution-variant boundary; `temporal_shape:` is not a Semantics shape field but a DataKind-level property.
- `11 §6.1` — Dimension authoring; temporal Dimensions' `grains:` list per `13 §4.2` is what this doc's identifying-Dimension Precondition TS-C4 checks against.
- `11 §7.2, §7.3` — `Additivity`'s default-`Additive` rule and the rationale for keeping it independent of `TemporalShape`.
- `13 §2.1` — `DataType` variants; `Date` / `Time` / `Timestamp` / `Interval` are the time-typed subset.
- `13 §3.2` — `Grain` total coarseness order; referenced by §4.1 / §4.3 / §8.2.
- `13 §4.2` — `DimensionType::Temporal` and its `grains:` list; TS-C2 / TS-C4.
- `16 §3.5` — `Cardinality × Additivity` matrix sharpened by §7.2.
- `16 §4.1`, `§4.4.2` — `JoinType` enum; `AsOf` gating discharged here.
- `16 §5`, `§5.3` — `ComposedSemanticInterface` and `CompositionKind` surface that §3.2 builds atop.
- `16 §11` — implicit composition algorithm; §8.3 adds shape-preconditions per-edge.
- `16 §13` — `Joinset` override surface; §5.5 records the `AsOf` override DEFERRAL.
- `16 §14` — error-code allocations; `16`'s `COMP_E_04xx` / `VALID_E_04xx` / `PLAN_E_05xx` / `PLAN_W_05xx` ranges are **not** disturbed by `17`'s 17NN allocation.
- `20`–`25` — per-DataKind strategy catalog; consumes `TemporalShape` for rollup, anchoring, and snapshot selection.
- `30 §6.2` — subsystem code-range allocation; `[CONTRADICTION-FOUND]` at head of doc records the 17NN coordination.
- `32` — YAML surface for `temporal_shape:` block and subtype discriminators.
- `33 §8B` — `TemporalShape` / `TemporalType` on `DataKind.temporal`; SemanticManifest-layer propagation of this doc's model-layer types (identifying-Dimension refs resolved to `EntityId`; `grain` generic; leaf-only on disk).
- `34` — planner's shape-aware strategy dispatch; `Request.temporal` consumption; `AsOf` emission.
- `35` — `PlanNode::Join` extension for `JoinType::AsOf(anchor)` payload.
- `36` — per-adapter emission rules for `AsOf` joins / snapshot selection / SCD window predicates.
- `registry/temporal_shape_mapping.md` — per-engine SCD / Events / Snapshot emission catalog.
- `questions/open/17_questions.md` (Q-TEMPORAL-001) · `questions/closed/17_questions.md` (Q-TEMPORAL-002 / -003 / -005 / -006 / -007 / -008) · `questions/deferred/17_questions.md` (Q-TEMPORAL-004).

## Testing Strategy

- Unionset shape compatibility validation (COMP_E_1710 UnionsetShapeConflict — per §8.1 matrix)
- Grainset shape constraints (COMP_E_1711 GrainsetShapeMixed, COMP_E_1712 EventsInGrainset/ScdInGrainset — per §8.2)
- Implicit composition shape-preconditions (PLAN_E_1751 AsOfNotAvailableInRound1, PLAN_E_1752 MultiScdImplicitCompositionForbidden, PLAN_W_1750 ScdWideCompositionWithoutAnchor — per §8.3-§8.5)
- Full Precondition catalog coverage — validate TS-V1 through TS-V7; compile TS-C1 through TS-C12; plan TS-P1 through TS-P10; advisory PLAN_W_1701/1702/1730-1732/1750 (per §9)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
