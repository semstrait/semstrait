---
doc: design/questions/closed/21_questions
status: Closed
purpose: Resolved questions originally raised against `data-kinds/21_dataset.md`
---

# Closed Questions — `data-kinds/21_dataset.md`

> Historical record of ratified Dataset decisions. Live items are in `[../open/21_questions.md](../open/21_questions.md)`.

---

## Q-DS-001 — Structural label for nested Simple under a Complex

**CLOSED (post-thirteenth-pass cascade, 2026-04-30).** Ratified at Round-1 working default (Option A): at nested scope, the structural anchor is `body.base.name` on `DataKindBase<LeafExtras>` (per `32 §3.1` + `32 §4`); `NestedDataset` carries no separate label field (`32 §3.3`). The post-thirteenth-pass sealed trait hierarchy in `20 §2` does not introduce a label-related contract on either `DataKind`, `SimpleDataKind`, or `PublicDataKind` / `NestedDataKind`. Re-opening is reserved for a future `23` / `24` ratification surfacing a per-member label orthogonal to the child's DataKind name (no such need has emerged).

**Question.** When a `SimpleDataKind` is nested inline inside a `ComplexDataKind` (per `12 §2`'s matrix — e.g. a Unionset branch, a Grainset level, a Joinset member), it participates in the parent's nested-kind scope (`11 §2.1`). `11 §10` allows nested kinds to carry a non-Semantics structural label distinct from their `name:`. Does `SimpleDataKind`'s Rust struct need a separate `label: Option<StructuralLabel>` field, or should `name:` double as the label at nested scope?

**Refs.**

- `21 §2.2` — pre-thirteenth-pass struct roster (`{name, interface, binding, temporal_shape, grain}`); superseded by `32 §3.1` (`DataKindBase<E>`) + `32 §3.3` (`NestedDataset`).
- `21 §2.5` — nesting posture.
- `11 §2.1` — nested-kind scope.
- `11 §10` — structural labels for nested kinds.
- `12 §2` — nesting matrix.
- `20 §`* — sealed trait hierarchy and mandatory trait surface (now ratified at `20 §2`).

**Arguments for (A) — ratified.**

- Matches LookML / Cube / dbt's parsing conventions: the `name:` of a nested model IS the member label.
- Minimizes shape churn — no extra field on `DataKindBase<E>` and no extra method on the sealed trait surface.
- `11 §10` handles the "when the label is not a Semantics name" case via scope rules.

**Arguments for (B) — rejected.**

- Decouples DataKind identity from structural role in the parent — useful for advanced composition where the same Simple is referenced by multiple Complexes. But `11 §2` forbids referenced children, so this case does not arise.

**Arguments for (C) — rejected.**

- Cleanest separation but forces Complexes to carry a parallel list of labels, duplicating name information for the common case.

**Current position.** **CLOSED at (A).** Round-1 default confirmed against the post-thirteenth-pass architecture. Re-open via `23` / `24` only if a concrete label-orthogonal-to-name use case arises.