---
prereqs: [00]
status: superseded
---

# 19_expression_flow (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`19_expression_flow/`](19_expression_flow/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [19-two-phase-pipeline](19_expression_flow/two-phase-pipeline.md) — The two-phase expression pipeline architecture and phase boundary contract
- [19-phase-a-resolution](19_expression_flow/phase-a-resolution.md) — Phase A compile-time resolution algorithm — resolve entry point, ResolvedExprTable, substitution, path resolution, cycle detection, type inference, reconciliation, sub-pass ordering
- [19-sugar-contract](19_expression_flow/sugar-contract.md) — Sugar contract — Family A constant folding / partial evaluation and Family B per-kind accessor lowering
- [19-expression-gates](19_expression_flow/expression-gates.md) — Per-site expression shape gates (scalar / Boolean / aggregate admission)
- [19-phase-b-placement](19_expression_flow/phase-b-placement.md) — Phase B plan-time placement rules — filter placement, group_by handoff, computed Dimension placement, Metric semantics, Additivity axis, advisory channel, aggregation handling
- [19-error-model](19_expression_flow/error-model.md) — Expression compile-pipeline error model — CompileError variants, resolution-stage errors, function-resolution errors, boundary-reconciliation warning, fail-fast policy
- [19-naming-and-scope](19_expression_flow/naming-and-scope.md) — Naming discipline and v1 scope boundaries
