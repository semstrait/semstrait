---
prereqs: [00]
status: superseded
---

# 34_semstrait_planner (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`34_semstrait_planner/`](34_semstrait_planner/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [34-planner-crate-charter](34_semstrait_planner/planner-crate-charter.md) — The semstrait-planner crate charter, layering, scope boundary, invariants I5-I12, runtime contract (manifest→graph→planning→plan), crate boundaries, and dependency posture
- [34-planner-request](34_semstrait_planner/planner-request.md) — Request, SessionContext, ResolvedQueryRequest, Filter/FilterOperator/FilterValue, SortDir, DataKindRef, TemporalRequest (reserved), RequestDimensionRef (rollup-aware carrier, reserved)
- [34-planner-plan](34_semstrait_planner/planner-plan.md) — The plan() free function signature, 7-step pipeline (step 0 constraint validation, step 1 Request lookup, step 2 Dataset routing, step 3 composition consistency check, step 4 Strategy dispatch, step 5 PlanNode construction, step 6 post-construction validation), field-first resolution (§10), diagnostics accumulation (fail-fast policy)
- [34-planner-strategy](34_semstrait_planner/planner-strategy.md) — Strategy trait surface (id/supports/plan), StrategyId enum, StrategyRegistry, StrategyContext, dispatch_strategy function, the four built-in strategies (SimpleStrategy/GrainsetStrategy/UnionsetStrategy/JoinsetStrategy) with algorithm pointers to 21-24, stability (open trait per 30 §4.6)
- [34-planner-optimize](34_semstrait_planner/planner-optimize.md) — The optimize() free function signature, OptimizerPass trait surface, Optimizer/OptimizerBuilder, canonical v1 passes (ConstantFolding/MetadataDimensionSubstitution/PredicateSimplification/IdentityProjectElimination), pass ordering, OptimizePassContext, OptimizeErrorKind, externally-implementable pass trait
- [34-planner-errors](34_semstrait_planner/planner-errors.md) — PlanErrorKind typed-kind enum (step 0-6 variants + per-DataKind passthrough Simple/Grainset/Unionset/Joinset), OptimizeErrorKind typed-kind enum, Diagnose impls, variant identity stability per 30 §5.4, non-error diagnostic emission policy
