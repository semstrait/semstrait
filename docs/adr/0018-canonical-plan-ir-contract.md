---
number: 18
title: 'Canonical plan IR: closed catalog, engine/cost/source-free payloads, composites spliced into generic nodes'
status: superseded
date: 2026-08-04
source:
- crates/semstrait-ir/src/plan
- crates/semstrait-planner/src/strategy.rs
- crates/semstrait-adapter/src
links:
- target: 66
  kind: supersededby
- target: 67
  kind: supersededby
- target: 68
  kind: supersededby
- target: 73
  kind: supersededby
---

# Canonical plan IR: closed catalog, engine/cost/source-free payloads, composites spliced into generic nodes

> **Superseded by [ADR 0066](0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md).** Each plan node is semstrait's implementation of one Substrait v0.103.1 relation, and "growth only inside the spec" becomes 0066's ruling 2. Its "no cost, statistics or physical properties" is narrowed: every node carries the optional Substrait common fields, the statistics hint among them, which the planner does not fill (0066's ruling 3); and `TopNNode`, a Substrait physical relation, is admitted and refused by `validate` (0066's ruling 4). `NodeMeta` becomes `NodeCommon`, `NodeId` becomes `Anchor`. Its `ScanNode` clause is ADR 0067's.
>
> **Superseded by [ADR 0067](0067-the-scan-carries-its-read-identity.md).** Its `ScanNode` clause, "holds only an opaque `SourceRef` resolved adapter-side against the manifest", becomes: the scan carries its reader, the physical source's read information, copied by the planner from the manifest's physical source, and `SourceRef` retires. Its "no partition metadata" and its rejection of expanded physical sources and partition columns on `ScanNode` are narrowed: a scan carries its listing and names its partition in trace, and partition columns are planned as literals above the scan.
>
> **Superseded by [ADR 0068](0068-adapters-are-layered-format-first.md).** Its artifact clause named `SqlArtifact.dialect`, `Dialect::ID` and `Capability` as the only homes of engine identity; `semstrait-ir::artifact` (`EngineArtifact`, `EnginePlan`, `SqlArtifact`, `DialectId`, `Dialect`, `Capability`) is deleted (0068's *Artifacts*), and engine identity's only home is an `impl SubstraitRules`, which lives in the engine adapter's own crate and never in `semstrait-adapter`. The rule that no canonical crate branches on engine identity is unchanged. Its open question, "injecting an `EngineAdapter` into `plan()`" versus "no engine influence before `adapt()`", is resolved for the second: the engine enters as rules at adapt and never at plan, and no engine reaches `plan`.
>
> **Superseded by [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md).** The dispatch site is the single match site: it matches `SemanticSource.role_definition` once, and its wildcard arm fails with `PlanError::StrategyNotFoundForSemanticSource`; the driver, not a Strategy, builds every plan node.

## Context and Problem Statement

The node catalog is the contract every adapter, optimizer pass and serialized plan is written against, so adding cost, dialect, source identity or composite variants would touch all of them and break the purity invariants CI enforces. It surprises readers: structures that look exactly like DataFusion `LogicalPlan`/Substrait `Rel` have nowhere to put a cost, a table name or a target dialect, `ResolvedDataKind::Complex` variants have no matching plan nodes, and a Grainset vanishes from the emitted tree.

## Considered Options

- Closed, engine-free catalog with composites spliced (chosen).
- A DataFusion-style logical plan with per-node cost/statistics and physical properties, dialect-aware variants (dbt-style `adapter_type` pushed upstream), and expanded physical sources/partition columns on `ScanNode`.
- Dedicated `PlanNode::{Grainset,Unionset,Joinset}` variants with per-site variant matching (the legacy `DataKindPlannerRegistry`).
- Open node/expression catalogs extended by adapters.

## Decision Outcome

- The `PlanNode`/`Expr<L>` variant catalogs are closed (growth only inside the spec). They carry no cost, statistics, physical or distribution properties, dialect tag, vendor rel kinds or partition metadata.
- `ScanNode` holds only an opaque `SourceRef`, resolved adapter-side against the manifest.
- No canonical crate may branch on engine identity: `DialectId` appears only on `SqlArtifact.dialect`/`Dialect::ID`/`Capability`.
- Complex DataKinds have no plan-node variants. `dispatch_strategy` is the single `ResolvedDataKind` match site, and each `Strategy` splices generic `Project`/`Agg`/`Join`/`Union` nodes.
- Engine-specific capability arrives only as `FunctionRegistry` entries.

## More Information

### Open questions

`foundations/10_resolution_pipeline.md` sanctions injecting an `EngineAdapter` into `plan()`, while `apis/36_semstrait_adapter`'s charter and `apis/35_semstrait_ir` forbid any engine influence before `adapt()`. This contradiction is unresolved in the design tree.
