---
doc: design/questions/open/36_questions
status: Living (focused v1 backlog)
purpose: Active architecture-impacting questions for `apis/36_semstrait_adapter.md`
depends-on:
  - apis/36_semstrait_adapter.md
  - apis/35_semstrait_ir.md
  - apis/34_semstrait_planner.md
  - apis/30_api_contracts.md
---

# Open Questions — `apis/36_semstrait_adapter.md`

Active set is narrowed to architecture-impacting items needed before v1 implementation planning.

Closed:
- [`../closed/36_questions.md`](../closed/36_questions.md)

Deferred (non-blocking operational depth):
- [`../deferred/36_questions.md`](../deferred/36_questions.md)

---

## Q-ADAPT-003 — `Dialect` vs `DialectEmit` split ownership

Should the split remain (`Dialect` structural in `35`, `DialectEmit` operational in `36`) or collapse into one trait?

Current default: keep split to preserve `35` purity and keep emission mechanics in adapter layer.

---

## Q-ADAPT-005 — Adapter registry lifecycle

Should `AdapterRegistry` remain process-global `OnceLock`, or move to caller/session-supplied registries?

Current default: global registry with optional API-layer allowlists.

---

## Q-ADAPT-007 — `Capability` roster ownership split

Should enum placement in `35` + roster authority in `36` remain, or converge to one owner?

Current default: keep split (`35` enum, `36` roster authority).

---

## Q-ADAPT-009 — Unsupported-feature error shape

Single top-level unsupported variant with sub-classifier vs separate top-level variants per unsupported family.

Current default: single variant + `UnsupportedFeatureKind`.

