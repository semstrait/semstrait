---
doc: design/questions/closed/14b_questions
status: Closed
purpose: Resolved questions originally raised against the (retired) `foundations/14b_expression_resolution.md`
---

# Closed Questions — `foundations/14b_expression_resolution.md` (retired)

> The `14b` chapter has been merged into `[../../foundations/19_expression_flow.md](../../foundations/19_expression_flow.md)` per `STATUS.md` item N. Historical Round-1 ratifications retained below for traceability; the surviving forwarding pointer for previously-open items lives at `[../open/14b_questions.md](../open/14b_questions.md)`.

## Q8 — Cycle-detection algorithm choice

**Status: CLOSED.** Tarjan SCC ratified by `[19 §3.5.2](../../foundations/19_expression_flow.md)`. The decision is stable.

**Question.** For cycle detection over the reference DAG, the Round-1 design chose Tarjan's strongly-connected-components algorithm. A lighter alternative existed: a linear DFS with a "gray / black" visited set that aborts on a back-edge.

**Round-1 decision.** Tarjan SCC.

**Rationale.** Single pass detects every cycle; the topological order is a free side-product reused by `19 §3.6`'s bottom-up type-inference pass (no fixpoint needed); stable order is easy to pin down per `00 §9` **I4**.
