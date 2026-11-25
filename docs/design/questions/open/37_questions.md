---
doc: design/questions/open/37_questions
status: Living (focused v1 backlog)
purpose: Active architecture-impacting questions for `apis/37_semstrait_catalog.md`
depends-on:
  - apis/37_semstrait_catalog.md
  - apis/33_semstrait_manifest.md
  - foundations/15_mapping_and_binding.md
---

# Open Questions — `apis/37_semstrait_catalog.md`

Active set is narrowed to architecture-impacting items needed before v1 implementation planning.

Closed:
- [`../closed/37_questions.md`](../closed/37_questions.md)

Deferred (non-blocking operational depth):
- [`../deferred/37_questions.md`](../deferred/37_questions.md)

---

## Q-CAT-002 — Glob semantics ownership

Should glob predicate semantics stay in `semstrait-common` with catalog-layer orchestration, or move fully into catalog crate?

Current default: keep predicate in core, orchestration in catalog.

---

## Q-CAT-003 — Snapshot pinning on providers without snapshot IDs

How should determinism and drift signaling behave when provider APIs only support `current` snapshots?

Current default: allow `SnapshotVersion::Current` with explicit limitation signaling.

---

## Q-CAT-008 — `Schema` type ownership (`catalog` vs `core`)

Where should shared schema vocabulary types live for long-term layering clarity?

Current default: owned in `semstrait-catalog`, imported by manifest.

---

## Q-CAT-012 — `CatalogRegistry` ownership

Should provider-registry composition live in `semstrait-manifest` or `semstrait-catalog`?

Current default: `semstrait-manifest` owns `CatalogRegistry`.

