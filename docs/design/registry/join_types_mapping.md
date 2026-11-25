---
prereqs: [00]
status: superseded
---

# join_types_mapping (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`join_types_mapping/`](join_types_mapping/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [registry-join-types-mapping-intro-basic](join_types_mapping/intro-basic.md) — Join-types mapping intro, scope, engine roster, plus four non-temporal canonical JoinType variants (Inner/Left/Right/Full) — universal per-engine emission
- [registry-join-types-mapping-asof](join_types_mapping/asof-reserved.md) — AsOf join variant emission (ScdWindow / SnapshotLatestAtOrBefore anchors), rewrite-tier table (First-class / Structural / Unsupported per anchor×engine), reserved future variants (Semi/Anti/Lateral) and Round-1 alternatives
- [registry-join-types-mapping-cardinality](join_types_mapping/cardinality-gaps.md) — Cardinality-informed emission (DISTINCT/array-aggregation hints for OneToMany/ManyToMany), cross-engine idioms (windowed-ASOF emulation, lateral-join correlated subquery, FULL OUTER emulation, DISTINCT vs GROUP BY), per-engine gap catalog, TD index, versioning, interaction
