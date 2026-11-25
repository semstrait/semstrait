---
doc: design/questions/deferred/36_questions
status: Deferred (post-v1 adapter operational depth)
purpose: Deferred questions moved from `open/36_questions.md`
depends-on:
  - apis/36_semstrait_adapter.md
---

# Deferred Questions — `apis/36_semstrait_adapter.md`

Moved from active v1 backlog because these items are non-blocking for architecture closure.

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-ADAPT-004 | `debug_sql` method vs free function | free function direction |
| Q-ADAPT-006 | per-adapter crate version pin policy | unresolved; per-adapter ratification |
| Q-ADAPT-008 | Substrait anchor mechanism | defer until divergence appears |
| Q-ADAPT-010 | audit seam placement | keep in-crate by default |

## Re-open trigger

Re-open when adapter implementation or release process makes one of these choices release-gating.

