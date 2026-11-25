---
prereqs: [00]
status: superseded
---

# 23_unionset (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`23_unionset/`](23_unionset/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [23-unionset-authoring](23_unionset/authoring.md) — Unionset authoring surface — explicit vs implicit fields, UnionMode roster (All/Unique), Public vs Nested form, per-child Coverage inference (v1 inference-only, no authored override)
- [23-unionset-plan](23_unionset/plan.md) — Plan-time fan-out mechanism — UnionMode projection, per-source pre-aggregation (always when Measures requested), NULL-fill projection at seam, column-type reconciliation, conditional final-aggregation, branch pruning
- [23-unionset-shape](23_unionset/shape.md) — TemporalShape and Grain interaction — authoring placement, cascade-up from children, V1 strict equivalence rule (equal kind/ScdType/grain across all children), multi-as-of Snapshot advisories
- [23-unionset-errors](23_unionset/errors.md) — Unionset-specific error rosters — VALID_E_2300–2399, COMP_E_2300–2399, PLAN_E_2300–2399
