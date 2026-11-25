---
prereqs: [00]
status: superseded
---

# 35_semstrait_ir (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`35_semstrait_ir/`](35_semstrait_ir/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [35-ir-crate-charter](35_semstrait_ir/ir-crate-charter.md) — The semstrait-ir crate charter, module layout, and canonical semantic_graph surface
- [35-ir-resolved-expr](35_semstrait_ir/ir-resolved-expr.md) — The resolved ir::PhysicalExpr type (alias Expr<PhysicalLeaf>), PhysicalLeaf set, and ExprLayer classification
- [35-ir-type-vocabulary](35_semstrait_ir/ir-type-vocabulary.md) — Canonical type vocabulary — TypeClass, Schema, SchemaColumn (non-authored; DataType/Grain imported from model)
- [35-ir-function-registry](35_semstrait_ir/ir-function-registry.md) — FunctionRegistry shape, FunctionSpec, Parameter/ParameterKey, and per-kind accessor enums (imported)
- [35-ir-plan-nodes](35_semstrait_ir/ir-plan-nodes.md) — SemanticPlan root, PlanNode sum type (8 variants + per-variant shapes), tree invariants, and visitor/traversal API
- [35-ir-plan-primitives](35_semstrait_ir/ir-plan-primitives.md) — Plan-level shared primitives — SourceRef, ResolvedColumn, Name, KeyPair, SortDir, NullOrdering, AggregateExpr, NodeMeta
- [35-ir-adapter-artifacts](35_semstrait_ir/ir-adapter-artifacts.md) — Adapter artifact family — EngineArtifact, EnginePlan, SqlArtifact, DialectId, Dialect trait, Capability
- [35-ir-serde-substrait](35_semstrait_ir/ir-serde-substrait.md) — Serde posture for SemanticPlan/PhysicalExpr, Substrait mapping table, and load-time integrity contract
- [35-ir-errors](35_semstrait_ir/ir-errors.md) — IR error types — ValidateError (imported), CompileError (function resolution), IrErrorKind (plan-shape diagnostics)
