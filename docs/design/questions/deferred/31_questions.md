---
doc: design/questions/deferred/31_questions
status: Deferred (post-v1 API surface refinements)
purpose: Deferred non-blocking questions moved from `open/31_questions.md`
depends-on:
  - apis/31_semstrait_common.md
---

# Deferred Questions — `apis/31_semstrait_common.md`

| ID | Topic | Last known default |
|---|---|---|
| Q5 | visitor shape depth (single method vs enter/exit) | keep single-method for now |
| Q7 | `#[non_exhaustive]` on expression wrappers | keep wrappers exhaustive, inner enum non-exhaustive |

Re-open when expression-resolution implementation requires trait-shape or wrapper-stability changes.

