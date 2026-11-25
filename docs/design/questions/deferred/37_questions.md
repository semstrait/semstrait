---
doc: design/questions/deferred/37_questions
status: Deferred (post-v1 catalog operational depth)
purpose: Deferred questions moved from `open/37_questions.md`
depends-on:
  - apis/37_semstrait_catalog.md
---

# Deferred Questions — `apis/37_semstrait_catalog.md`

Moved from active v1 backlog because these items are non-blocking for architecture closure.

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-CAT-004 | scheme-dispatching filesystem utility | caller-composed in v1 |
| Q-CAT-005 | streaming read/delete filesystem methods | omit in v1 |
| Q-CAT-006 | mutation trait posture | separate future trait |
| Q-CAT-007 | `async_trait` vs native async traits | keep `async_trait` in v1 |
| Q-CAT-009 | `expand_glob` return shape | keep `Vec<Path>` |
| Q-CAT-010 | partition transform vocabulary breadth | keep Iceberg-exact + non-exhaustive |
| Q-CAT-011 | filesystem provider schema source strategy | keep empty schema default |

## Re-open trigger

Re-open when catalog implementation needs one of these decisions for behavior that directly affects v1 compile/manifests.

