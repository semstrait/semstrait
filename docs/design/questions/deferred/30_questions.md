---
doc: design/questions/deferred/30_questions
status: Deferred (post-v1 policy depth)
purpose: Deferred non-blocking questions moved from `open/30_questions.md`
depends-on:
  - apis/30_api_contracts.md
---

# Deferred Questions — `apis/30_api_contracts.md`

## Deferred set

| ID | Topic | Last known default |
|---|---|---|
| Q-API-003 | optimizer error-vs-warning posture depth | reserve optimizer errors for pass failures |
| Q-API-006 | `#[non_exhaustive]` matrix granularity for resolved structs | public MAY-grow structs non-exhaustive; internals hidden |
| Q-API-008 | async posture nuance for manifest surfaces | compile-time async only, plan-time sync |
| Q-API-009 | `RegistryExtension` sealing policy depth | open trait by default |
| Q-API-010 | stability tier naming | two tiers (`Stable`/`Provisional`) sufficient for v1 |

Re-open when any of these becomes release-gating for v1 API contracts.

