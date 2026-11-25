---
doc: design/questions/open/14b_questions
status: Retired
purpose: Round-1 questions originally raised against `foundations/14b_expression_resolution.md`; all settled by the merged `foundations/19_expression_flow.md`
---

# 14b — Open Questions — RETIRED

> **Retired 2026-05-18 as part of `STATUS.md` item N (expression compile-pipeline cascade).** The doc `foundations/14b_expression_resolution.md` was merged into `[19_expression_flow.md](../../foundations/19_expression_flow.md)`. Every Round-1 open question previously listed here has been settled by `19`'s ratification.

## Round-1 question outcomes

| OQ  | Topic                                  | Settled in `19`                                          |
| --- | -------------------------------------- | -------------------------------------------------------- |
| OQ-1 | `PhysicalExpr` interning              | `[19 §3.2.4](../../foundations/19_expression_flow.md)` — no interning; entries store `PhysicalExpr` inline. |
| OQ-2 | Multi-leaf path composition           | `[19 §3.4.5](../../foundations/19_expression_flow.md)` — distinct paths in `PathSignature.paths`; join-subgraph canonicalization is `16`'s concern. |
| OQ-5 | `Provenance` entry-level vs node-level | `[33 §6.3.1](../../apis/33_semstrait_manifest.md)` — entry-level only. |
| OQ-6 | Batch-error mode for resolution       | `[19 §8.4](../../foundations/19_expression_flow.md)` — fail-fast; batch mode tracked as future extension. |
| OQ-7 | `BindingId` stability                 | MOOTED — `BindingId` eliminated (id-first rework, STATUS item U.2). Bindings identified by deterministic `EntityId`; see `[closed/15 Q-MAP-001](../closed/15_questions.md)`. |
| OQ-9 | Split join-key columns in `referenced_columns` | `[19 §3.10](../../foundations/19_expression_flow.md)` — inline; no separate field. |

This file remains as a forwarding pointer; the closed sibling `[../closed/14b_questions.md](../closed/14b_questions.md)` retains the historical Round-1 ratifications.
