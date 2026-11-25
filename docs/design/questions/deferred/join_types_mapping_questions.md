---
doc: design/questions/deferred/join_types_mapping_questions
status: Deferred (post-v1 mapping style and adapter-depth items)
purpose: Deferred questions originally authored in `open/join_types_mapping_questions.md`
depends-on:
  - registry/join_types_mapping.md
  - foundations/16_composition.md
---

# Deferred Questions — `registry/join_types_mapping.md`

These questions are important for adapter polish but not blocking for v1 architecture closure.

---

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-JOIN-MAP-001 | explicit `INNER` vs bare `JOIN` | explicit `INNER JOIN` |
| Q-JOIN-MAP-002 | `ON` vs `USING` | always `ON` |
| Q-JOIN-MAP-003 | auto-rewrite `RIGHT` to `LEFT` | preserve author orientation |
| Q-JOIN-MAP-004 | `AsOf` rewrite-tier authority split | temporal mapping table authoritative; join table mirrors |
| Q-JOIN-MAP-005 | canonical `LATERAL` posture | adapter-extended only |
| Q-JOIN-MAP-006 | cardinality-driven auto `DISTINCT` / hints | no automatic emission |
| Q-JOIN-MAP-007 | `FULL OUTER` emulation for historical dialects | pattern documented, no active engine row |

---

## Re-open trigger

Re-open when adapter implementation requires these style/policy choices for shipping behavior, or when non-v1 engines/dialects join the active roster.

