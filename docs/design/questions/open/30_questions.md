---
doc: design/questions/open/30_questions
status: Living (focused v1 backlog)
purpose: Active architecture-impacting questions for `apis/30_api_contracts.md`
depends-on:
  - apis/30_api_contracts.md
  - apis/31_semstrait_common.md
  - apis/33_semstrait_manifest.md
  - apis/34_semstrait_planner.md
  - apis/36_semstrait_adapter.md
  - apis/37_semstrait_catalog.md
---

# Open Questions — `apis/30_api_contracts.md`

Active set is narrowed to architecture-impacting items required for v1 closure.

Closed:
- [`../closed/30_questions.md`](../closed/30_questions.md)

Deferred:
- [`../deferred/30_questions.md`](../deferred/30_questions.md)

---

## Q-API-002 — Warning propagation across fail-fast stages

When a fail-fast stage returns an error, should warnings observed before the failure always be preserved in the error arm?

Current default: preserve warnings in both success and failure tuple arms.

---

## Q-API-004 — `Span` authoritative ownership (`core` vs `model`)

Should `Span`/`SourceId`/`ContextLine` stay owned in `semstrait-common`, with model-specific variants layered by `semstrait-model`?

Current default: core-owned primitives to preserve DAG and reuse.

---

## Q-API-007 — Adapter/catalog architecture framing

Should adapter/catalog integrations remain separate-crate posture or migrate to single-crate feature-gated modules (item C reconciliation)?

Current default in this sidecar: separate-crate posture; status tracked as active reconciliation item in `STATUS.md`.

