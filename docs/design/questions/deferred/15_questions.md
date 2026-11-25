---
doc: design/questions/deferred/15_questions
status: Deferred
purpose: Mapping-and-binding questions parked for post-v1 ratification
---

# Deferred Questions — `foundations/15_mapping_and_binding.md`

> Items deferred to v2 (or later) ratification. Live items are in `../open/15_questions.md`; closed items are in `../closed/15_questions.md`.

---

## Q-MAP-009 — Hive-style partition value type — DEFERRED to v2 (2026-04-27)

**Status: DEFERRED to v2.** Partition extraction (`partition.level: N`) is non-goal in v1 — the v1 metadata extraction surface is **path-only** per `15 §8.0` / `13 §4.7` v1-scope banner / `15 §13 R47`. The compile pass guards `partition: Some(_)` with `COMP_E_0322 MetadataPartitionDeferredV2`. The result-type contract for Hive-style partitions (raw segment vs value-after-`=`, declared override grammar, type-inference fallback) reactivates when v2 ratifies the partition arm.

**Refs.**

- `15 §8.0` — v1 scope: path-only.
- `15 §8.2 / §8.2.1` — v2 design parking with the original options preserved for future reference.
- `15 §11.1 COMP_E_0322` — v1 compile-time guard.
- `15 §13 R31` — entry confirmed `DEFERRED v2`.
- `13 §4.7` — author-side body retained for forward-compat with v1-scope banner.

**Original options (preserved for v2 ratification).**

- **A:** raw value post-`=`, typed `String`. Authors cast downstream via `Expr`.
- **B:** raw value, typed per author-declared partition-column type (new YAML surface).
- **C:** auto-detect from first encountered value.

**Next step.** Address during the v2 partition-extraction ratification pass; the Hive-style result-type decision rolls into the broader partition-arm design.

---

## Q-MAP-D02 — Thread A: model-as-truth posture for binding fields — DEFERRED

**Status: DEFERRED.** Tabled at the 2026-05-28 manifest ratification cascade. Thread A (the broader question of how the SemanticManifest relates to the authored Model — copy-by-value of every Model field vs reference-into-Model vs partial-projection) is scoped at the model-layer and does not block manifest shape ratifications C1–C18 / CCK / CX1. v1 ratification posture: manifest copies primitive fields by value (per C17(d) / C18 / CX1); the Round-1 alternative — "manifest holds an `Arc<SemanticModel>` and projects fields lazily" — is parked.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — Thread A (model-layer concern, deferred from manifest cascade).
- `15 §2.2` — `Binding` ownership discipline.
- `15 §7.5` — per-Binding mapping persistence.
- See sibling: [`32_questions.md`](32_questions.md) Q-MODEL-D03 — model-layer counterpart.

**Open axes (preserved for v2).**

- **Manifest-as-projection vs manifest-as-copy.** Projection avoids duplication but couples manifest-load to Model availability; copy is self-contained but bloats manifest size for shared fields.
- **Selective sharing.** Some fields (`Binding.format`, `Binding.partition_def`) are large and rarely re-read post-compile; lazy projection could pay off there.

**Next step.** Address during a v2 manifest-vs-Model layering ratification pass when manifest size in real-world deployments becomes a measured concern. Migration MAJOR if the public `SemanticManifest` field shape changes.
