---
doc: design/questions/closed/32_questions
status: Closed
purpose: Resolved questions originally raised against `apis/32_semstrait_model.md`
---

# Closed Questions — `apis/32_semstrait_model.md`

> Historical record of ratified model decisions. Live items are in [`../open/32_questions.md`](../open/32_questions.md); deferred items in [`../deferred/32_questions.md`](../deferred/32_questions.md).

---

## Session-close note — 2026-04-17 (canonical-entity types end-to-end closure)

The late-session Q&A sweep ratified 18 model-level decisions, consolidated into the new [`foundations/18_entities.md`](../../foundations/18_entities.md) (originally landed as `apis/32c_entities.md`, promoted to the foundations layer in the same consolidation pass) and cascaded into `32`, `32b`, `26`. Stale items retired in that pass:

- `Q-MODEL-003` (`kind:` discriminator spelling) — superseded by per-variant plural tags (already retired).
- `Q-MODEL-004` (`DataKindRef::Inline` hoisting cadence) — superseded by the 6-layer `Public*` / `Nested*` hierarchy (already retired).

---

## Q-MODEL-001 — Multi-file / directory loader helper — **CLOSED (PARTIALLY RESOLVED)**

**Question.** Should `semstrait-model` expose a helper for loading YAML from external storage, and if so, does it also cover multi-file / directory aggregation?

**Resolution (single-file).** `semstrait-model::io::load_model` / `load_catalogs` are ratified as the async single-source wrappers over `semstrait-common::io::Source` — see `32 §10.4`, `32b §5.4`, `31b §1 / §5`. The wrappers accept exactly one `Source` yielding a single YAML blob; directory / multi-blob aggregation is **not** in their contract.

**Resolution (multi-file).** Out of scope for the `model` crate, **forever**. `Q-IO-003` was ratified as **CLOSED (out-of-scope)** in [`questions/closed/31b_io_questions.md`](31b_io_questions.md): multi-file walks are caller territory (CLI / LSP / pipeline orchestrator), not a `core::io` back-end concern and not a `model::io` wrapper concern. `[TD-MODEL-DIR-LOADER]` is therefore retired — there is no `model`-side feature pending; callers compose `Source::List(_)` walks themselves and feed each blob through `load_model` independently.

**Refs.**

- `32 §10.4` — ratified `load_model` / `dump_model` / `load_catalogs` / `dump_catalogs` single-source wrappers.
- `32b §5.4` — shared catalog async wrappers.
- `32b §4.1` — catalog resolution precedence now collapses to `CatalogEntry.default_namespace > provider default` (no reference-site override per the 2026-04-17 canonical-entity closure).
- `31b §1 / §5` — `Source` / `Sink` transport primitives consumed by §10.4.
- `32 §10.3` — multi-file loading explicitly out of scope for v1.
- [`questions/closed/31b_io_questions.md` `Q-IO-003`](31b_io_questions.md) — multi-file walks ratified as caller-owned (out of scope for `core::io`).
- `00 §9` I11 — domain wrappers over `core::io` are permitted in model / manifest; CLI-level aggregation stays outside.

---

## Q-MODEL-002 — Primary error-code shape — **CLOSED (kebab-case primary)**

**Question.** `32 §11` documents `ParseError` variants with a numeric-code grouping (`PARSE_E_01xx`, `PARSE_E_02xx`, ...) in comments but `code()` returns kebab-case per `31 §8.3`. Which is the primary, authoritative code shape — kebab-case (`"parse.duplicate-data-kind"`) or numeric (`"PARSE_E_0201"`)?

**Resolution (2026-04-17, canonical-entity closure).** Kebab-case is now locked in as primary. The `18`-entity ratification pass landed 12 new `SR-E-*` rules — every diagnostic code ships as kebab-case (`validate.semantics-ref-immutable-override`, `parse.measure-missing-agg`, …) with no numeric counterpart. That closes any lingering ambiguity about which form "wins" when the two disagree.

**Refs.**

- `30 §6` — subsystem prefix allocation (`PARSE_E`, `VALID_E`, `COMP_E`, `PLAN_E`, `PLAN_W`, `ADAPT_E`) — now secondary / historical.
- `31 §8.3` — `code()` returns kebab-case; numeric form preserved as a `const LEGACY_CODE: &'static str` associated value per variant.
- `31 §15` R-?? — "migration stance only, pending a ratification pass" — `31`'s position is kebab-case primary, numeric secondary.
- `14 §7` — the historical numeric form (`EXPR_E_####`) ratified there.
- `00 §9` I12 — first-class diagnostics with stable documented codes.
- `18 §11` — 12 new `SR-E-*` codes, all kebab-case, zero numeric counterparts.

**Remaining sub-questions (routing-only).** (a) should the numeric form be exposed via `ParseError::legacy_code(&self) -> Option<&'static str>` per `31`'s scheme? (b) should `30 §6`'s allocation table be rewritten to use kebab-case ranges or retired entirely?

---

## Q-MODEL-003 — `kind:` discriminator spelling: `simple` vs `dataset` — **RETIRED (STALE)**

**Status.** Superseded by the ratified YAML surface in `32 §2.1` and the per-variant plural tag convention (`datasets:` / `unionsets:` / `grainsets:` / `joinsets:`). There is no single `kind:` discriminator in the ratified model; every variant has its own block-level plural tag, so the "which spelling" question no longer applies.

**Cross-reference.** See `32 §2.1` (top-level YAML roster) and `32 §3.1` — `DataKindBase<E>` common-field struct + per-variant `*Body` structs replace the earlier `DataKind { kind: ... }` sketch.

---

## Q-MODEL-004 — `DataKindRef::Inline` hoisting cadence — **RETIRED (STALE)**

**Status.** Superseded by the ratified 6-layer DataKind hierarchy (`32 §3`). Inline vs. by-name declarations are no longer a `DataKindRef` enum concern — `Public*` and `Nested*` concrete types carry their own structural shape, and nesting is enforced by the per-variant `*Body` struct layout (`32 §3.2` + `26`). The `DataKindRef::Inline` variant no longer exists in the ratified model.

**Cross-reference.** See `32 §3.2 / §3.3` for the ratified type hierarchy and `26` for the nesting matrix.

---

## Q-MODEL-009 — `SemanticsName` duplication collapse — **CLOSED (resolved by item Y / V10)**

**Origin.** Raised as OQ-C during the item-Y vocabulary ratification (2026-06-15), then briefly carried as an open sidecar. Closed in the same ratification session: V10 already **decides** it, so it is not an open design question — only an execution step.

**Resolution.** Two `SemanticsName` definitions existed: the rich model copy (`semstrait-model`, `types::names::SemanticsName` — `Ord`, `new` / `as_str`, `From`, `Display`, derived `JsonSchema`) and a bare ir newtype (`semstrait-ir`). **V10 ratifies collapse to the model copy**; `semstrait-ir` imports it. One definition ⇒ one `JsonSchema` `$def` (eliminates the double-registration hazard the V10 Schemars note flags); the model copy is strictly richer, so ir consumers gain capability and lose nothing structural; and the carrier rides with the authoring vocabulary per item Y. The spec docs already reflect the end state (model owns it, ir imports — `35 §20.10`).

**Execution caveat (carried, not open).** The collapse lands during the item-Y crate-code phase, sequenced with the V10 relocation of the leaf/identifier module to `semstrait-model` (avoids a transient dangling import). At execution, confirm no persisted-IR ordering invariant depends on the previously-non-`Ord` ir type — expected nil (the ir copy was used as a value, not a `BTreeMap` key). Tracked as `[TD-MODEL-SEMANTICSNAME-COLLAPSE]`.

**Refs.**

- `_research/vocabulary/RATIFICATION_LOG.md` §V10 — placement decision (`SemanticsName` (ir copy) → **delete → use model's**) + Schemars note (one `JsonSchema` impl).
- `32 §4.3` — model is the crate-of-record for the identifier carriers under item Y / V10.
- `35 §20.10` — `semstrait-ir` crate-root re-exports `SemanticsName` from `semstrait-model` (imported, not owned).
