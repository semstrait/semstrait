---
doc: design/questions/closed/37_questions
status: Closed
purpose: Resolved questions originally raised against `apis/37_semstrait_catalog.md`
---

# Closed Questions — `apis/37_semstrait_catalog.md`

> Historical record of ratified catalog decisions. Live items are in [`../open/37_questions.md`](../open/37_questions.md).

---

## Q-CAT-001 — `CAT_E_*` / `FS_E_*` subsystem-prefix registration in `30 §6.2`  *[Closed — superseded by typed-kind transition]*

**Status.** Closed. The eleventh-pass retirement of the stable string-code subsystem at `30 §6` (2026-04-29) makes prefix registration moot. Catalog and filesystem failures are identified by typed-variant identity on `CatalogProviderErrorKind` and `FileSystemErrorKind` respectively (per `37 §8.1` / §8.2 rewrites); no `CAT_E_*` or `FS_E_*` numeric strings exist anywhere in the public surface. The `[TD-CAT-CODE-TABLE-AMEND]` tech-debt item that this question created retires alongside the string-code surface.

**Original framing (preserved).** `37 §8.3` formerly proposed registering two new subsystems in `30 §6.2`'s reserved-ranges table: `CAT` (catalog, range `0100`–`0399`) and `FS` (filesystem, range `0100`–`0199`). `30 §6.2` did not list either at the time of drafting. Three sub-questions:

- **(a)** Should `CAT` and `FS` be added to `30 §6.2` as distinct subsystems, or share one subsystem (e.g. `IO`) with internal sub-ranges?
- **(b)** Are the proposed 300-wide (`CAT`) and 100-wide (`FS`) ranges sized appropriately, or should both be 1000-wide for consistency with other subsystems?
- **(c)** Should the range start at `0100` (per `30 §6.2` convention for most subsystems) or at `0001` (per `PARSE`, `EXPR`)?

Round-1 defaulted to two separate subsystems with the proposed ranges. The `[TD-CAT-CODE-TABLE-AMEND]` tech-debt item tracked the pending `30 §6.2` amendment. The decision was gated on `30`'s next amendment pass alongside `Q-IR-001` (the analogous IR prefix question).

**Resolution.** No prefix registration occurs because no prefix table exists. `CatalogProviderErrorKind` and `FileSystemErrorKind` are independent typed enums each implementing `Diagnose`; identification is by enum-variant identity, not by a `{SUBSYSTEM}_{SEVERITY}_{NUMBER}` string. Sub-questions (a) / (b) / (c) all dissolve — there is no "subsystem" concept and no "range" concept in the typed-kind discipline. The `[TD-CAT-CODE-TABLE-AMEND]` tech-debt item retires.
