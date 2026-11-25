---
prereqs: [00]
status: superseded
---

# 24_joinset (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`24_joinset/`](24_joinset/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [24-joinset-canonical-shape](24_joinset/canonical-shape.md) — JoinsetDataKind struct, ExplicitPath, JoinHop, anchor specification, and binary-arity v1 constraint
- [24-joinset-path-resolution](24_joinset/path-resolution.md) — Join-path specification — implicit path (anchor-rooted BFS) vs explicit path (author-pinned hop sequence), validation, scope-local Relationship shadow
- [24-joinset-strategy](24_joinset/strategy.md) — JoinsetStrategy planner contract — anchor-outward PlanNode::Join emission, per-hop JoinType derivation (18 §2.9), Cardinality propagation, fanout advisories
- [24-joinset-interface](24_joinset/interface.md) — Joinset's ComposedSemanticInterface — UnifiedSemantics merge, FieldProvenance, CompositionCoverage, anchor-first constituents ordering
- [24-joinset-errors](24_joinset/errors.md) — Joinset-specific error rosters — VALID_E_2400–2499, COMP_E_2400–2499, PLAN_E_2400–2499, PLAN_W_2400–2499
