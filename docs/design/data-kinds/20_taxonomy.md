---
prereqs: [00]
status: superseded
---

# 20_taxonomy (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`20_taxonomy/`](20_taxonomy/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [20-datakind-taxonomy](20_taxonomy/datakind-taxonomy.md) — DataKind sealed trait hierarchy, four-variant roster (Dataset/Grainset/Unionset/Joinset), Simple vs Complex split, shared invariants D1-D9, and the variant-level contract surface
- [20-strategy-lifecycle](20_taxonomy/strategy-lifecycle.md) — Strategy trait surface, dispatch mechanism, per-stage responsibility skeleton (parse/validate/compile/plan/optimize/adapt), and the shared error-code roster (VALID_E_2000-2099, COMP_E_2000-2099, PLAN_E_2040-2099)
