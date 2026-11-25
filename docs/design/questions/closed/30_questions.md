---
doc: design/questions/closed/30_questions
status: Closed
purpose: Resolved questions originally raised against `apis/30_api_contracts.md`
depends-on:
  - apis/30_api_contracts.md
---

# Closed Questions — `apis/30_api_contracts.md`

---

## Q-API-001 — Reconcile `10 §5.1` diagnostic sketch with `30`

**CLOSED (structure-optimization pass, 2026-05-03).** Superseded by typed-kind discipline already ratified in `30` and propagated in `31`-`39`.

**Resolution.** `30` remains authoritative for diagnostic shape and behavior; legacy numeric-code wording is treated as historical context only.

---

## Q-API-005 — Error-code retirement mechanics

**CLOSED (structure-optimization pass, 2026-05-03).** Numeric-code retirement mechanics are not v1-blocking under typed-kind-first policy.

**Resolution.** Keep this topic as historical/reference-only. Active v1 decisions key on typed kinds and documented API behavior rather than subsystem code tables.

