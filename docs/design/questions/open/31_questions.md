---
doc: design/questions/open/31_questions
status: Living (focused v1 backlog)
purpose: Active architecture-impacting questions for `apis/31_semstrait_common.md`
depends-on:
  - apis/31_semstrait_common.md
  - foundations/14_expressions.md
---

# Open Questions — `apis/31_semstrait_common.md`

Closed:
- [`../closed/31_questions.md`](../closed/31_questions.md)

Deferred:
- [`../deferred/31_questions.md`](../deferred/31_questions.md)

---

## Q2 — `ContextLine` placement

Should rich context rendering data live directly on common diagnostic primitives or remain presentation-layer concern?

Current default: not exposed in common public primitive surface.

---

## Q6 — `SourceId` opacity surface (`as_str` / `Display`)

Should both `as_str()` and `Display` remain, or should one be removed for stricter opacity?

Current default: opaque newtype with both convenience surfaces.

---

## Q8 — `is_reserved_tag` source-of-truth

Which source should drive reserved-tag table synchronization (static list vs enum-derived vs fixture-driven)?

Current default: not ratified yet.

