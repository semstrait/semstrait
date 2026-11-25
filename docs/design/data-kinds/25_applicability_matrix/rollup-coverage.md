---
id: 25-rollup-coverage
status: ratified
version: 2.1
impl: none
---

# Applicability — Rollup & Coverage

## 4. Per-Variant Rollup-Legality Summary

"Rollup" means transforming rows at one `Grain` into rows at a coarser `Grain` via bucket-then-aggregate (`DATE_TRUNC` + `GROUP BY` + additive aggregation). Rollup legality depends on `Grain` coarseness order (`13 §3.2`), `TemporalShape` (`17 §4`), and `Additivity` (`11 §7` / `17 §7`).

| Variant | Native rollup? | Rollup driver | Requires anchor? | Emits advisories? |
|---|---|---|---|---|
| `Simple` / Dataset | **Yes** — at `SimpleStrategy` L4 `Aggregate`. | Simple's declared `grain:` (`21 §6`) compared against Request's requested grain. | No. | `PLAN_W_2101 LossyMultiSourceReaggregation`; `PLAN_W_2102 ShapeAdditivityMismatch`. |
| `Grainset` | **Yes, at the strategy level** — `Grainset` **picks** a child whose grain rolls up to the Request's grain per `17 §4`'s matrix; the chosen child's own strategy then handles the rollup. | `GrainsetDataKind.grain_axis` + per-child `grain:` declarations. | **Yes** — implicit: the Grainset's `grain_axis` anchors the decision (`22 §2.1`). `Snapshot` children with `PinOnly` never roll up (`22 §5`). | `PLAN_W_2201 GrainsetUnitPrunable`; `PLAN_W_2202 GrainsetCrossGrainJoinFanout` (`22 §9`). |
| `Unionset` | **Indirectly** — the terminal `Aggregate` above `PlanNode::Union` performs the rollup. The Union itself has no grain; grain is determined by the common-coarsest of contributing branches per `23 §7`. | Children's grains reconciled at compile; Request grain compared against common-coarsest. | No in v1 (union mode is set-like, not anchor-biased). | `COMP_W_2308 UnionsetGrainDivergent`; `PLAN_E_2302 UnionsetGrainIncompatibleWithRequest`. |
| `Joinset` | **Inherits anchor grain** — the Joinset's output grain is the anchor's grain; target-side rollup is the target member's own concern (`24 §5.4`). | `Joinset.anchor`'s grain + per-member grain + `17 §5`'s `AsOf` activation. | **Yes** — the `Joinset.anchor` IS the rollup anchor (`24 §3.1` / `§3.4`). | `PLAN_W_2400 JoinsetFanoutAdvisory`; `PLAN_W_2402 ManyToManyHopAdvisory`; `PLAN_W_2403 MultiFanoutAdvisory`; `PLAN_W_2404 AsOfActivation`. |

Cross-refs: `17 §4` (shape × grain matrix), `22 §4.4` (rollup mechanics; shape-gated per `17`; internal `RollupPolicy`), `23 §7` (Unionset post-Union rollup), `24 §7` (`AsOf`-gated Joinset hops).

### 4.1 Worked-pointer table

For a reader who prefers an example-first traversal, the authoritative rollup walk-throughs are:

| Variant | Worked example | Rollup path shape |
|---|---|---|
| `Simple` | `21 §10` multi-source Dataset rollup. | `Scan → Aggregate(group_by = Request.dimensions, agg = Measures)`. |
| `Grainset` | `22 §10` paid-media rollups (single-unit case in §10.2; cross-grain JOIN case in §10.3). | Single-unit case: delegates to winning routing unit; no `PlanNode::Grainset`. Cross-grain case: emits `PlanNode::Join { LeftOuter }` over driver + attached. |
| `Unionset` | `23 §11` heterogeneous branches with NULL-fill. | `Union → Aggregate`. |
| `Joinset` | `24 §12` two-Simple anchor + target. | `Scan(anchor) → Join(target) → (optional Aggregate over anchor's grain)`. |

### 4.2 Shape-gated rollup legality — cross-reference

Per `17 §4.1`'s matrix (referenced by every variant that rolls up):

| Shape | Rollup discipline | Owning variant |
|---|---|---|
| `Timeseries` | Roll freely along the Grain axis (`13 §3.2`). | Consumed by `SimpleStrategy` L4 (`21 §4.5`) and `GrainsetStrategy` rollup wrapper (`22 §4.4`). |
| `Events` | Roll to coarser buckets via `DATE_TRUNC`; additive only. | Same consumers. |
| `Snapshot` | `PinOnly` — no rollup across snapshots without a pin policy (`22 §5`). | `GrainsetStrategy` rejects (`PLAN_E_2203 GrainsetSnapshotRollupWithoutPin`). |
| `SCD` | Rollup requires as-of anchoring (`17 §5`). | `GrainsetStrategy` rejects without as-of pin (`PLAN_E_2204 GrainsetSCDRollupWithoutAsOf`). |

---

## 5. Per-Variant Coverage & NULL-Fill Summary

`Coverage` appears at two levels (per `15 §1.3` / `15 §6.4` / `16 §8.1`):

- **Binding-level** `Coverage` — the X-axis of source selection inside a `Simple` (`15 §6`). Records per-`(Semantics, PhysicalSource)` whether the source is `Native`, `Derived`, or `NullFill`.
- **Composition-level** `CompositionCoverage` — the per-`(ConstituentRef, UnifiedName)` fold over constituents' Binding-level `Coverage`s (`16 §8`). Records per-child / per-member coverage of the composed surface.

NULL-fill appears in two distinct axes:

- **`CoverageVariant::NullFill`** at the `Coverage` / `CompositionCoverage` layer (`15 §6.1` / `16 §8.4`): a source / constituent that does not serve a given Semantics.
- **`FieldOwnership::NullFill`** at the `FieldProvenance` layer (`16 §7.3.3`): a composition-level record of which constituents provide a field and which do not. **Per `16 §7.3.3`, only `Unionset` emits `FieldOwnership::NullFill`.** Joinset NULL-fill is carried by `JoinType` outer-join semantics at the plan tree, not by `FieldProvenance`.

| Variant | Coverage surface | NULL-fill in `FieldProvenance`? | NULL-fill in plan? |
|---|---|---|---|
| `Simple` / Dataset | **Binding-level** per-`(Semantics, PhysicalSource)` per `15 §6`. Heterogeneous `NullFill` across sources in a bare Simple is a compile error (`COMP_E_0310 UnusableNullFillInNonUnionContext`); author must wrap in a Unionset (`21 §3.2`). | `n/a` — bare `SemanticInterface` has no `FieldProvenance`. | `n/a` under bare Simple; `NullFill` is rejected before plan. |
| `Grainset` | **Per-effective-routing-unit Coverage** — projected from each unit's interface (`22 §3.2`). For cross-grain JOIN composition (`22 §3.4`), folded into a `CompositionCoverage` per `16 §8`. `NullFill` Coverage is legal (a unit may lack a Semantics the Request does not name); a Semantics that NO unit provides is a compile error (`COMP_E_2202`). | No `FieldOwnership::NullFill` — Grainset's two routing modes (single-unit delegation per `22 §4.2`, cross-grain LEFT OUTER JOIN per `22 §4.3`) emit NULLs via JOIN semantics, not via `FieldOwnership`. Same-grain pre-merge inside the Grainset (per `22 §3.3`) produces an implicit Unionset whose `FieldOwnership::NullFill` lives at the Unionset level (per `23 §5.5`), not at the Grainset's. | **Conditional**. Single-unit delegation: chosen unit's rows flow directly with rollup if grain coarsens. Cross-grain LEFT OUTER JOIN: NULL-fill emerges from `JoinType::LeftOuter` semantics on attached units (per `22 §4.3`). |
| `Unionset` | **Per-child bare `Coverage`** — inferred at compile from each child's own interface + Binding-level `Coverage` (Simple) or resolved composed interface (Complex); no `CompositionCoverage` post-rebase (`23 §3.2`). | **Yes** — `FieldOwnership::NullFill(providers)` records which children DO cover each field; non-providers inferred by set-difference (`23 §5.5`, `16 §7.3.3`). | **Yes** — per-child `Project` emits `Cast(Null, unified_type)` at the seam for every `NullFill` field (`23 §4.3`). The only variant that materializes structural NULL-fill in `PlanNode`s. |
| `Joinset` | **Per-member `CompositionCoverage`** — fold per `24 §8.4`; every member is either `Native` or `NullFill` on each composed-surface Semantics. Most Joinset coverage rows are `Native` on one side and `NullFill` on the other. | **No** — `16 §7.3.3` reserves `FieldOwnership::NullFill` for Unionset; Joinset-side outer-join NULL-fill is carried by `JoinType` semantics at plan time (`24 §8.3` / `§5.5` step 3). | NULL-fill is emitted by the `PlanNode::Join`'s outer-join semantics (`Left` / `Right` / `Full`), not by a typed `Cast(Null, _)` projection. |

Q-24-08 in `questions/closed/24_questions.md` ratified the Round-1 position: Joinset does NOT gain structural `NullFill` records for outer joins (closed).

Cross-refs: `15 §6` (Binding-level Coverage), `16 §7.3.3` (per-variant `FieldOwnership::NullFill` policy), `16 §8.4` (`CompositionCoverage` fold), `21 §3.2` (Simple + multi-source), `22 §6` (Grainset), `23 §5` (Unionset — the canonical NULL-fill case), `24 §8.3` / `§8.4` (Joinset).

### 5.1 Coverage-vs-Provenance axis contrast

The two terms both talk about "where a field comes from" but operate at different layers. `25` reads the two axes explicitly rather than assuming readers remember which is which:

| Question | Answered by `Coverage` / `CompositionCoverage` | Answered by `FieldProvenance` |
|---|---|---|
| Does this source / constituent serve this field? | Yes — `CoverageVariant` enumerates `Native` / `Derived` / `NullFill` / `Metadata`. | No — provenance is about composition ownership, not source provision. |
| Who owns this field on the composed surface? | No. | Yes — `FieldOwnership::Native(src)` / `Shared(srcs)` / `Derived(expr)` / `NullFill(providers)`. |
| Does this branch contribute a row? | Indirectly — `NullFill` coverage + strategy semantics imply pruning (`23 §4.6`). | No. |
| Does this variant emit `NullFill` rows at plan? | Only Unionset (`23 §4.3`). | Only Unionset (`16 §7.3.3`). Joinset outer-join NULL-fill is `PlanNode::Join` semantics (`24 §8.3`). |

### 5.2 NULL-fill origin routing

Which mechanism produces a runtime NULL depends on the composition kind:

| Source of the runtime NULL | `Simple` | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| Binding-level `CoverageVariant::NullFill` (`15 §6`). | Rejected at `compile` in bare Simple (`21 §3.2`); legal only inside a Unionset branch. | n/a — child Bindings are resolved per Simple's rules. | Legal; drives `FieldOwnership::NullFill` at composition. | n/a. |
| Unionset per-child `Project` with `Cast(Null, unified_type)` (`23 §4.3`). | n/a. | n/a. | Yes — primary emission site. | n/a. |
| Outer-join (`JoinType::Left` / `Right` / `Full`) emission in `PlanNode::Join` (`24 §5.5` step 3). | n/a. | n/a. | n/a (except inside a Joinset child of a Unionset). | Yes — primary emission site. |
| SQL engine's intrinsic NULL handling (expression over NULL, missing row). | Engine concern; not emitted structurally. | Same. | Same. | Same. |

### 5.3 Coverage lift — from `Binding` to `CompositionCoverage`

The lift operation that promotes Binding-level Coverage to CompositionCoverage is cross-referenced from each Complex doc but authored once in `16 §8.4`. The per-variant specialization:

| Variant | Lift input | Lift output | Authoritative ref |
|---|---|---|---|
| `Grainset` | Per-effective-routing-unit Coverage projection (each unit derives from its child's `Binding`-level Coverage for Simple children, or from the child's resolution for Complex children — including same-grain pre-merge implicit Unionsets per `22 §3.3`). | Per-unit `BTreeMap<SemanticsName, CoverageVariant>`; lifted to `CompositionCoverage` keyed by `(unit-ref, unified-name)` only when cross-grain JOIN composition activates per `22 §3.4`. | `22 §3.2`. |
| `Unionset` | Per-child `Coverage` inferred at compile from each child's own interface + Binding-level `Coverage` (Simple) or resolved composed interface (Complex) (`23 §3.2`). | Same keying shape. | `23 §5`. |
| `Joinset` | Per-member `Coverage`. | Same keying shape; anchor side typically `Native`, target side typically `Native` or `NullFill` per outer-join. | `24 §8.4`. |

No variant re-implements the lift algorithm; all three reuse the `16 §8.4` fold.

---

## Testing Strategy

- Rollup table completeness (4 variants × 4 shape kinds); Coverage-vs-Provenance axis contrast; NULL-fill origin routing per variant.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
