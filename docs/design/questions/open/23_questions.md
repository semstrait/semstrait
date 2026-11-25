---
doc: design/questions/open/23_questions
status: Living
purpose: Parked unresolved questions surfaced while drafting `data-kinds/23_unionset.md`
depends-on:
  - data-kinds/23_unionset.md
  - data-kinds/20_taxonomy.md
  - data-kinds/22_grainset.md
  - data-kinds/24_joinset.md
  - data-kinds/25_applicability_matrix.md
  - data-kinds/26_nesting_matrix.md
  - foundations/13_types_and_grain.md
  - foundations/14_expressions.md
  - foundations/15_mapping_and_binding.md
  - foundations/16_composition.md
  - foundations/17_temporal_shape.md
  - foundations/18_entities.md
  - apis/30_api_contracts.md
  - apis/32_semstrait_model.md
  - apis/33_semstrait_manifest.md
  - apis/34_semstrait_planner.md
  - apis/35_semstrait_ir.md
  - apis/36_semstrait_adapter.md
---

# Open Questions — `data-kinds/23_unionset.md`

> Five questions remain open after the post-thirteenth-pass cascade rebase (2026-05-03): Q-UNI-004, -006, -012, -013, -014. Six items closed in this rebase: Q-UNI-003 (no per-child override in v1; inference-only), Q-UNI-005 (moot under V1 strict equivalence), Q-UNI-007 (moot under V1 strict equivalence), Q-UNI-008 (`{All, Unique}` v1 roster ratified at C6), Q-UNI-010 (re-scoped under `CompositionKind` retirement), Q-UNI-011 (collapsed via Q-UNI-003). All closed items including ratified resolutions are in [`../closed/23_questions.md`](../closed/23_questions.md). Each remaining entry restates the question, lists its ratified references against the post-rebase `23` section numbering, and records the Round-1 default `23` currently uses. Entries migrate out as later docs (`17`, `24`, `25`, `30`, `34`–`36`) make decisions that confirm or amend `23`'s defaults.

---

## Q-UNI-004 — `Avg` re-aggregation: default `Error` (`PLAN_E_2304`) or `Lossy` warning (`PLAN_W_2302`)?

**Question.** `23 §4.5`'s re-aggregation function table marks `Avg` as "not decomposable," directing authors to restructure as an explicit Metric with `Sum(num) / Sum(den)`. Round-1 emits `PLAN_E_2304 UnionsetReAggregationInfeasible` for an `Avg` Measure across a Unionset. Should the default be an error, or a `Lossy`-warning with a fallback to "sum the averages and re-average" (which is mathematically wrong on heterogeneous row counts, but provides a result)?

**Refs.**
- `23 §4.5` — Round-1 default: `PLAN_E_2304` for `Avg`.
- `23 §9` (`PLAN_E_2304`) — error roster.
- `_drafts/34_unionset_strategy.md §6.2` — re-aggregation function table.
- `22 §?` — parallel Grainset `Avg` decision (forthcoming); if the two docs diverge, cross-source consistency is undermined.
- `14a` — `FunctionRegistry`; `Avg` signature.
- `18 §2` — Measure's `Aggregation` variant roster.

**Arguments for error (Round-1 default).**
- `Avg` across Unioned children is almost always mathematically wrong (the average of averages ≠ the true average unless row counts are equal). Failing loudly is safer than silent wrong answers.
- Authors can opt into a `Sum(num) / Sum(den)` Metric declaration if they want correct semantics; the error message points at this remediation.

**Arguments for warning-with-fallback.**
- Some adapters (e.g. OLAP engines) compute `AVG` in a way that's meaningful over pre-aggregated rows (carrying a count-weight); the engine-side fallback may be correct in those specific cases.
- Matches SQL's permissive behavior: a `SELECT AVG(x) FROM (unioned)` in SQL doesn't fail; authors may expect the same.

**Current position in `23`.** Error (`PLAN_E_2304`).

**Next step.** Keep as error; promote `Avg` → `Metric` pattern in example documentation. If adapter-side weighted-average becomes a thing, revisit as `[TD-UNIONSET-AVG-WEIGHTED]` and introduce a `PLAN_W_23xx` variant.

---

## Q-UNI-006 — Post-prune single-child collapse: emit `PlanNode::Union` or skip?

**Question.** `23 §4.6` notes that Coverage-driven pruning may reduce the surviving child set to 1. In that case, should the emitted plan (a) retain a single-input `PlanNode::Union` for structural regularity, or (b) skip the `Union` node entirely and flow the surviving child's subplan directly into the (conditional) final aggregation?

**Refs.**
- `_drafts/34_unionset_strategy.md §7` ("Single-branch post-prune") — Round-1 default: skip the Union.
- `23 §4.6` — Coverage-driven branch pruning predicate.
- `35` (pending) — `PlanNode::Union` field roster; may allow `inputs: Vec<PlanNode>` with `len == 1` or may require `len >= 2`.

**Arguments for skip (Round-1 default).**
- A single-input UNION is a no-op at the IR level and produces unnecessary plan tree depth.
- Downstream adapters may not generate SQL like `SELECT ... UNION ALL ... (empty second branch)`; skipping avoids adapter complexity.

**Arguments for retain.**
- Structural regularity: every Unionset emits `PlanNode::Union` unconditionally; downstream plan-introspection tools don't need a special case.
- Makes the re-expansion of pruned branches (if a future optimizer re-admits them) more structurally uniform.

**Current position in `23`.** Skip the Union for single-child-post-prune cases.

**Next step.** Confirm at `35` ratification. If `35`'s `PlanNode::Union` requires `|inputs| >= 2`, the skip is mandated. If `35` allows `len == 1`, keep the Round-1 skip as an optimization and document the alternative retain path in `34`'s planner trait surface.

---

## Q-UNI-012 — `PlanNode::Union` mode-flag projection

**Question.** `23 §4.7` and `_drafts/34_unionset_strategy.md §7` emit `PlanNode::Union { mode: UnionMode, inputs: Vec<PlanNode> }`. `35` ratifies the node's exact roster. Should the field be a strict `mode: UnionMode` (matching `UnionsetBody.mode`'s authoring surface), or a richer discriminator (`UnionKind` enum) that anticipates non-Unionset planner-level union emission (e.g. a future Substrait-style `SetRel`)?

**Refs.**
- `23 §4.1, §4.7` — Round-1: emit `mode: UnionMode` with `#[non_exhaustive]` per `32 §3.2`.
- `35` (pending) — `PlanNode::Union` field roster.
- `36` — adapter trait; SQL generation per engine.
- `37` — catalog-adapter; not involved here.
- Engine compatibility: DataFusion has `LogicalPlan::Union`; Substrait has `SetRel`; DuckDB / Postgres have ANSI `UNION` / `UNION ALL`; some older warehouses have dialectal differences.

**Current position in `23`.** Emit `mode: UnionMode`; adapter responsibility to translate.

**Next step.** Confirm at `35`-drafting. If `35` prefers a richer discriminator (`UnionKind` enum with more variants or a `distinct: bool` flag), `23 §4.7` aligns.

---

## Q-UNI-013 — Re-aggregation wrapper: single-level or nested?

**Question.** `_drafts/34_unionset_strategy.md §6` emits a single final `PlanNode::Agg` above the `PlanNode::Union` (when not elided). For deep nesting (Unionset → Grainset → SimpleStrategy), each level may emit its own Agg, producing nested `Agg(Union(Agg(Scan), Agg(Agg(Scan))))`-style plans. Is nested aggregation acceptable, or should the planner collapse redundant levels?

**Refs.**
- `_drafts/34_unionset_strategy.md §4, §6` — Round-1: per-branch sub-Agg always (when Measures requested); final Agg conditional on disjointness; no collapsing pass at the `23` level.
- `23 §4.5` — re-aggregation policy.
- `35` (pending) — whether `PlanNode::Agg(Agg(...))` nesting is admitted.
- `34` (pending) — whether the planner has an optimizer pass for aggregate collapsing.

**Current position in `23`.** No collapsing pass at the `23` level; per-child aggregates are the child-Strategy's concern.

**Next step.** If `34` introduces an optimizer pass for aggregate collapsing, `23` may benefit (reducing redundant work). Track as `[TD-UNIONSET-AGG-COLLAPSE]`.

---

## Q-UNI-014 — Adapter-level engine-specific UNION semantics

**Question.** Some engines treat `UNION ALL` with mismatched nullabilities differently (e.g. Spark ANSI mode rejects; Spark non-ANSI widens; DuckDB always widens; Postgres depends on column types). Should `23` ratify a normalized nullability-widening behavior that all adapters must match, or leave it adapter-dependent?

**Refs.**
- `23 §4.4` — Round-1: nullability widens if any contributor is nullable; advisory `COMP_W_2301`.
- `36` — adapter trait.
- I2 — logical types only in the canonical layer; `13`'s `DataType` does not carry nullability as a top-level axis (nullability is per-field on the SemanticInterface).

**Current position in `23`.** Canonical-layer rule: unified column nullability widens if any contributor's column is nullable or NULL-filled. Adapters translate to engine-specific nullability.

**Next step.** Confirm at `36`-drafting that the canonical-layer widening rule is adapter-emulable on every target engine. If an engine cannot express the widened type, adapters must add explicit `CAST` or issue a runtime diagnostic.

---
