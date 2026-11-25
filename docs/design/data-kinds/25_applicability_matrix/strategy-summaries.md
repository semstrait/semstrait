---
id: 25-strategy-summaries
status: ratified
version: 2.1
impl: none
---

# Applicability — Per-Variant Strategy Summaries

## 3. Per-Variant Planner-Strategy Summary

One subsection per variant. Each summary is three-to-five bullets naming the strategy's key moves + a pointer into the authoritative section. Strategy semantics themselves are NOT ratified here; `§3` is a reader's convenience.

Across all four strategies, four discipline points hold uniformly (per `20 §5.2` `Strategy` trait + `10 §3.4` `plan` stage + `10 §8`):

- **Synchronous hot path** per I6. No `.await` in `Strategy::resolve`; no blocking I/O; no `PhysicalSource` probe.
- **SemanticManifest-complete input** per I8. Every datum the strategy needs (interface, binding, coverage, provenance, relationships, temporal shape) is already in the `SemanticManifest` by the time `plan` runs.
- **Deterministic output** per I4. Given identical `(SemanticManifest, Request, SessionContext)`, `Strategy::resolve` produces an identical `PlanNode` tree.
- **Emits diagnostics, not panics** per I12. Unreachable / invariant-violation states emit a `PLAN_E_*` diagnostic; panics are reserved for implementer-bug assertions.

Every strategy's ratification doc repeats these points in its own voice; `§3` does not re-state them per-strategy.

### 3.1 `SimpleStrategy` — `Simple` / Dataset

- **Fast path.** `Simple` is the only variant that owns a `Binding`; its strategy emits a single `PlanNode::Scan` subtree (plus optional `Project` / `Agg` / `Project` layers per the 5-layer plan shape of `21 §4.1`).
- **5-layer canonical shape.** L1 `Scan` → L2 `Rename` → L3 `Expression` → L4 `Aggregate` → L5 `Project`. Every layer except L1 is optional per the skip rules in `21 §4.7`.
- **Multi-`PhysicalSource` fan-out.** One `Scan` per source; union at L1 when glob expansion yields ≥ 2 sources with uniform Coverage (`21 §3.2` / `§4.2` step 3).
- **No composition machinery.** `SimpleStrategy` never walks `Relationship` / `Cardinality` / `CompositionCoverage`; those belong to Complex strategies. Simple participates in composition as a **constituent**, not as a composer.
- **Filter placement.** Filter clauses (`11 §6.4`) are placed between L1 (`Scan`) and L2 (`Rename`) when they reference only pre-rename column names, or between L3 (`Expression`) and L4 (`Aggregate`) when they reference computed expressions (`21 §4.1` / `§4.4`).
- **Authoritative section:** `21 §4` (with the skip-rule table in `21 §4.7` and the worked example in `21 §10`).

### 3.2 `GrainsetStrategy` — `Grainset`

- **Effective routing units.** Grainset is a **router**, not a composer of rows. At compile, children are normalized into one effective routing unit per distinct grain; same-grain children fold into an implicit Unionset (`mode: All`, **non-strict** NullFilling per `23 §3.2`) per `22 §3.3`. Routing decisions operate at the unit granularity, not the raw-child granularity (`22 §4.1`).
- **Two routing modes.** **Single-unit delegation** (the common case, `22 §4.2`): when one effective routing unit covers the full Request, the planner delegates to that unit's own Strategy and wraps with a rollup transformation if the unit's grain is finer than the Request's. **Cross-grain JOIN composition** (`22 §4.3`, new in this rebase): when no single unit covers the full Request, the planner builds a LEFT OUTER JOIN tree mediated by a `ComposedSemanticInterface` (per `16 §5`) — driver = most-covering grain-eligible unit (name-order tie-break per G-2b); attached units added in name order (G-2c) and equi-joined on shared `Key`s per `18 §2.5`.
- **Shape-gated rollup legality.** Per `17 §4`'s matrix: `Timeseries`/`Events` roll freely; `Snapshot` requires pin policy or rejects via `PLAN_E_2203 GrainsetSnapshotRollupWithoutPin`; `SCD` requires as-of anchor or rejects via `PLAN_E_2204 GrainsetSCDRollupWithoutAsOf` (`22 §9`).
- **Strict V1 shape equivalence.** All children's `TemporalShape.kind` (incl. SCD subtype) must match — mismatch is `COMP_E_2201 GrainsetChildShapeKindMismatch` (hard error, not advisory). The pre-rebase `MixedShapeAdvisoryChildren` (`PLAN_W_2202`) was retired.
- **Plan shape.** Single-unit case: chosen unit's subplan + optional rollup wrapper (`Project (DATE_TRUNC) + Agg`); there is no `PlanNode::Grainset`. Cross-grain case: `PlanNode::Join { LeftOuter, on: <equi-join on shared Keys> }` over driver + attached subtrees, all wrapped by the standard final pipeline (`Filter → Agg → Project`) per `22 §4.3`.
- **Internal `RollupPolicy`.** `RollupPolicy { ShapeDefault, PinOnly, PreferFinest }` is implemented internally as a planner knob (per G-4 ratification, 2026-05-03). NOT authored in V1 YAML; default behavior follows shape rules in `17`.
- **Authoritative sections:** `22 §3.4` (cross-grain `ComposedSemanticInterface` construction) + `22 §4` (plan-time observable behavior) + `22 §5` (`TemporalShape` interaction) + `22 §10` (worked example). Algorithm body in [`../_drafts/34_grainset_strategy.md`](../../_drafts/34_grainset_strategy.md) pending lift into `34 §<GrainsetStrategy>`.

### 3.3 `UnionsetStrategy` — `Unionset`

- **UNION ALL / UNION DISTINCT over children.** Emits `PlanNode::Union { distinct, inputs }` where `inputs[i]` is the `i`-th child's strategy output wrapped by a per-child `Project` seam (`23 §4.1`).
- **NULL-fill at the seam.** For every composed-surface field a child does NOT provide, the per-child `Project` emits `PhysicalExpr::Cast(Null, unified_type)` (`23 §4.3`); `unified_type` derives from the LUB across contributing children (`23 §4.4`).
- **Delegates per-child.** The strategy never reaches into a child's subplan construction — Simple children run `SimpleStrategy`, Grainset children run `GrainsetStrategy`, Joinset children run `JoinsetStrategy` (`23 §4.2`).
- **Coverage-driven branch pruning.** A child whose every Request-selected field is `NullFill` is pruned from the Union with advisory `PLAN_W_2301 UnionsetBranchPrunable`; exceptions for `COUNT(*)`-like queries (`23 §4.6`).
- **Terminal re-aggregation.** Post-Union `PlanNode::Aggregate` re-groups by requested Dimensions and re-aggregates Measures; skipped when no Measures requested, when metadata-Dimension values are source-distinguishing, or when branch-pruning collapsed the Union to a single branch (`23 §4.5`).
- **Authoritative section:** `23 §4` (strategy) + `23 §5` (Coverage semantics) + `23 §11` (worked example).

### 3.4 `JoinsetStrategy` — `Joinset`

- **Anchor-outward join emission.** The anchor's `Scan` is the plan-tree left of hop 0; each subsequent hop joins onto the running tree's right (`24 §5.2`). v1 is binary (`12 §5.2` / `24 §2.5`), so one hop.
- **Path resolution at compile.** Implicit path runs `RELATIONSHIP_BFS_ANCHORED`; explicit path validates `JoinHop` by `JoinHop`; both produce `Vec<ResolvedJoinHop>` before plan (`24 §4`).
- **`JoinType` derivation.** Per-hop `JoinType` derived from each traversed `Relationship.optional` field (`18 §2.9`). There is no per-hop override surface in v1; a Joinset needing divergent join semantics declares a scope-local `Relationship` shadow (`18 §2.10`, `24 §5.3.2`).
- **Shape-gated `AsOf` activation.** Per-hop shape pairs consult `17 §5`'s activation matrix. `Events ↔ Snapshot` / `Events ↔ SCD` mandate `AsOf`; `Snapshot ↔ Snapshot` forbids it. Advisory `PLAN_W_2404` on silent activation (`24 §7`).
- **Post-join unified `Project`.** Consumes `UnifiedSemantics` (`16 §6`) for rename / namespacing and `FieldProvenance` (`16 §7`) for per-field ownership. Joinset-level Filter applies atop the Project (`24 §5.5`).
- **Authoritative section:** `24 §5` (strategy) + `24 §7` (`AsOf` integration) + `24 §8` (`ComposedSemanticInterface` shape) + `24 §12` (worked examples).

### 3.5 Cross-variant composition — worked pointer

A Joinset over a Grainset member is the canonical example of cross-variant composition (permitted by `12 §2` and `12 §5.1`). Intended plan emission (authoritative in the owning variants' plan-shape sections):

1. `RESOLVE_JOINSET_PATH` runs per `24 §4` — anchor and target members are resolved to their `ResolvedDataKind` entries.
2. For each Grainset member, `GrainsetStrategy` runs first to pick a winning child for the requested grain (`22 §4`).
3. The winning routing unit's sub-plan is spliced in place of the Grainset (`22 §4.2`); the Joinset's `PlanNode::Join` then composes over the spliced subplans.
4. `AsOf` activation consults the spliced unit's `TemporalShape` (not the Grainset's — Grainset has no top-level shape per `§2.9`).

`24 §12` and `22 §10` each carry worked versions of this composition from their side.

### 3.6 Strategy interaction table

A compact reading of how the four strategies interact along the axes that matter for cross-variant composition:

| Axis | `SimpleStrategy` | `GrainsetStrategy` | `UnionsetStrategy` | `JoinsetStrategy` |
|---|---|---|---|---|
| Delegates to child strategies? | No. | Yes — one (winner). | Yes — N (all branches). | No (delegates via inline Scan per member, no separate child strategy invocation in v1). |
| Emits a `PlanNode` of its own kind? | No — only shared nodes (`Scan` / `Project` / `Aggregate`). | Single-unit case: No — splices chosen unit's subplan (`22 §4.2`). Cross-grain case: Yes — `PlanNode::Join { LeftOuter }` per `22 §4.3`. | Yes — `PlanNode::Union` (`35 §*`). | Yes — `PlanNode::Join` (`35 §*`). |
| Requires per-member Coverage at plan? | No — single Binding (`15 §6`). | Yes — `CompositionCoverage` drives eligibility. | Yes — drives NULL-fill projection. | Yes — drives `FieldProvenance` at Project time. |
| Consumes `Relationship` at plan? | No. | No. | No. | Yes — per hop. |
| Consumes `TemporalShape` at plan? | Advisory only. | Yes — gates child eligibility. | Advisory (heterogeneity) + error (post-Union grain incompat). | Yes — gates `AsOf` activation. |
| Emits advisories (`PLAN_W_*`)? | Yes (`21 §9`). | Yes (`22 §9` — `PLAN_W_2201` / `PLAN_W_2202`). | Yes (`23 §10.2`). | Yes (`24 §11.2`). |

---

## Testing Strategy

- Strategy bullet-list completeness per variant; cross-variant composition table (Joinset × Grainset example); authoritative-pointer validity.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
