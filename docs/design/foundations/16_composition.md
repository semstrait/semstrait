---
prereqs: [00]
status: superseded
---

# 16_composition (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`16_composition/`](16_composition/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [16-composition-overview](16_composition/composition-overview.md) — Navigation and scope overview for the composition design space
- [16-relationship-edge-model](16_composition/relationship-edge-model.md) — Composition semantics for Relationship edges — placement, scope, bidirectional traversal, fanout analysis
- [16-composed-surface-type](16_composition/composed-surface-type.md) — The ComposedSemanticInterface struct, CompositionKind, Origin axis, ImplicitId, and SemanticsView trait
- [16-composed-surface-contents](16_composition/composed-surface-contents.md) — UnifiedSemantics namespace-aware merge, FieldProvenance per-field ownership, and CompositionCoverage fold
- [16-implicit-enumeration-and-materialization](16_composition/implicit-enumeration-and-materialization.md) — Explicit vs implicit composition boundary, eager materialization at SemanticGraph build, depth and count bounds, canonical-form clash rejection
- [16-field-first-resolution](16_composition/field-first-resolution.md) — The planner's field-first resolution algorithm — pure lookup over SemanticGraph's build-time composition index
- [16-joinset-model](16_composition/joinset-model.md) — Joinset explicit and implicit forms, scope-local Relationship shadow, canonical-form differentiators, and traversed paths
- [16-composition-validation-and-errors](16_composition/composition-validation-and-errors.md) — Relationship graph well-formedness preconditions and composition error model (CompileError, ValidateError, PlannerError variants)
