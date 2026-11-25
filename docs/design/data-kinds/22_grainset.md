---
prereqs: [00]
status: superseded
---

# 22_grainset (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`22_grainset/`](22_grainset/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [22-grainset-authoring](22_grainset/authoring.md) — Grainset authoring surface — explicit vs implicit fields, Public vs Nested form, per-child grain mandatory (SR-E-8), ≥2 unique grains requirement
- [22-grainset-same-grain](22_grainset/same-grain.md) — Same-grain pre-merge mechanism — implicit Unionset assembly (mode: All, non-strict NullFilling per 23 §3.2) for ≥2 children at same grain
- [22-grainset-cross-grain](22_grainset/cross-grain.md) — Cross-grain composition mechanism — ComposedSemanticInterface + LEFT OUTER JOIN tree on shared Keys (18 §2.5) when no single routing unit covers Request
- [22-grainset-routing](22_grainset/routing.md) — Plan-time routing mechanics — effective routing units (one per distinct grain), single-unit delegation (common case), cross-grain JOIN delegation (partial-coverage case), rollup wrapping (shape-gated per 17)
- [22-grainset-shape](22_grainset/shape.md) — TemporalShape interaction — authoring placement, cascade-up of kind only (grain stays per-child), V1 strict equivalence rule (equal kind/ScdType/grain across children)
- [22-grainset-errors](22_grainset/errors.md) — Grainset-specific error rosters — VALID_E_2200–2299, COMP_E_2200–2299, PLAN_E_2200–2299
