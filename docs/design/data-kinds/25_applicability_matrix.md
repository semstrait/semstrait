---
prereqs: [00]
status: superseded
---

# 25_applicability_matrix (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`25_applicability_matrix/`](25_applicability_matrix/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [25-applicability-matrix-core](25_applicability_matrix/core.md) — The master cross-cut matrix — foundation rule (10–17) × DataKind variant (Simple/Grainset/Unionset/Joinset) applicability
- [25-strategy-summaries](25_applicability_matrix/strategy-summaries.md) — Per-variant planner-strategy summaries — SimpleStrategy/GrainsetStrategy/UnionsetStrategy/JoinsetStrategy bullet-list contracts + cross-variant composition table
- [25-rollup-coverage](25_applicability_matrix/rollup-coverage.md) — Per-variant rollup-legality summary (Grain × TemporalShape × strategy) and Coverage/NULL-fill summary (Binding-level vs CompositionCoverage-level)
- [25-error-bands](25_applicability_matrix/error-bands.md) — Per-variant error-code-band index (VALID_E/COMP_E/PLAN_E 2000–2599 allocation) and cross-variant reservations (2500–2599)
