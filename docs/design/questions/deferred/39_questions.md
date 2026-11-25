---
doc: design/questions/deferred/39_questions
status: Deferred (post-v1 ergonomics and packaging)
purpose: Deferred questions originally authored in `open/39_questions.md`
depends-on:
  - apis/39_semstrait_facade.md
  - apis/38_semstrait_api.md
---

# Deferred Questions — `apis/39_semstrait_facade.md`

Moved from active backlog during structure-optimization pass.

These items are valid but non-blocking for v1 implementation planning.

---

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-FAC-001 | Default feature composition (`ansi-sql` default) | keep `default = [\"ansi-sql\"]` |
| Q-FAC-002 | Prelude membership of `ir::Name` | keep in prelude for now |
| Q-FAC-004 | `run` catalog wiring shape | keep hard-coded `NoopCatalogProvider` |
| Q-FAC-005 | Exact-version pin policy for sub-crates | keep exact pin |
| Q-FAC-006 | Reserved feature-name namespace policy | keep reserved list with caveat |
| Q-FAC-007 | Prelude growth budget policy | no hard cap, principle-based |
| Q-FAC-008 | `semstrait::VERSION` constant posture | keep facade constant |

---

## Re-open trigger

Re-open when one of these occurs:
- facade API expansion blocks v1 adoption;
- packaging/versioning decisions become release-gating;
- user-facing ergonomics become implementation-critical for `39`.

