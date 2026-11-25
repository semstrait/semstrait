---
doc: design/questions/closed/17_questions
status: Closed
purpose: Resolved questions originally raised against `foundations/17_temporal_shape.md`
---

# Closed Questions — `foundations/17_temporal_shape.md`

> Historical record of ratified temporal-shape decisions. Live items are in [`../open/17_questions.md`](../open/17_questions.md); deferred items in [`../deferred/17_questions.md`](../deferred/17_questions.md).

---

## Q-TEMPORAL-001 — `30 §6.2` code-range reconciliation for the 17NN block

**CLOSED (structure-optimization pass, 2026-05-03).** Superseded by the typed-kind diagnostic policy ratified in `30` and cascaded through `31`-`39`. Numeric code-range governance is no longer a v1 gating surface for stage diagnostics; variant identity on typed `*ErrorKind` enums is authoritative.

**Resolution.** Keep historical 17NN allocation discussion as archival context only. Active v1 work references typed-kind diagnostics and stage-owned enums rather than numeric-range reconciliation.

---

## Q-TEMPORAL-002 — `Scd` payload shape: per-subtype vs flat-fields

**CLOSED (Phase-3 cascade, 2026-04-17).** Ratified via `foundations/18_entities.md §3.3`: **Option B (flat fields)** is the v1 shape, combined with the v1 `ScdType` roster trim to **`{Type1, Type2}`**. `ScdBody` carries `{ scd_type: ScdType, valid_from: SemanticsName, valid_to: SemanticsName }`. There is no longer a per-subtype payload divergence to model, because the four subtypes that had distinct payloads (`Type3` / `Type4` / `Type5` / `Type6`) are descoped from v1. Option A (per-subtype) and Option C (hybrid window-struct) are post-v1 concerns — if the roster re-expands to include history-preserving subtypes with divergent shapes, the question reopens.

**Question.** `17 §2.3` adopts a nested `Scd { subtype: ScdSubtype }` form where `ScdSubtype` carries per-subtype payloads (e.g. `Type2 { valid_from_dim, valid_to_dim, current_flag_dim: Option<_> }`, `Type3 { prior_value_dim }`, `Type4 { history_data_kind_ref }`, etc.). The alternative — a **flat** `Scd { subtype: ScdSubtype, valid_from_dim: Option<SemanticsName>, valid_to_dim: Option<SemanticsName>, prior_value_dim: Option<SemanticsName>, history_data_kind_ref: Option<DataKindRef>, current_flag_dim: Option<SemanticsName>, ... }` where fields are present-or-absent per subtype — was mentioned in the authoring brief. Which is canonical?

**Refs.**

- `17 §2.3` — chosen per-subtype form with rationale.
- `crates/semstrait-model/src/types/temporal.rs` — existing implementation uses a per-subtype form (with `ScdVersionedColumns` as a helper struct shared between `Type2 / Type5 / Type6`).
- `00 §9` I10 — non-exhaustive extensibility rule.

**Options.**

- **A. Per-subtype payload (current).** Each `ScdSubtype` variant carries exactly the fields its semantics need. Type-system refuses nonsense combinations (`Type0` with `valid_from_dim` fails to compile). Matches the existing crate code. Adding a new subtype adds a variant with its own payload.
- **B. Flat fields.** Single `Scd { subtype, ...optional_fields }` struct with every possible SCD field as `Option<_>`. Validation at compile time enforces subtype-specific presence rules (e.g. `Type2 → valid_from_dim.is_some()`). YAML authoring is arguably friendlier (one block with named fields rather than nested by subtype key).
- **C. Hybrid — shared window-struct for the common case.** Keep per-subtype payload for structurally divergent subtypes (`Type3`, `Type4`, `Type5`, `Type6`) but share a `ScdWindow { valid_from_dim, valid_to_dim, current_flag_dim }` struct across `Type2 / Type5 / Type6`. Matches the existing crate's `ScdVersionedColumns`.

**Current position.** **CLOSED — Option B (flat fields)** ratified via `18 §3.3`. Body retained as historical resolution context.

**Next step.** None. Reactivates only on a future `ScdType` roster expansion that introduces shape-divergent subtypes.

---

## Q-TEMPORAL-003 — `Joinset` `JoinType::AsOf` override: ratify pre- or post-implicit-`AsOf`? — CLOSED (2026-04-28)

**Status: CLOSED.** Option B (implicit-first) ratified as the post-v1 milestone-sequencing baseline. When planner-side `AsOf` lands, **milestone 1** ships matrix-driven implicit synthesis per `17 §5.2` — Joinset traversals automatically receive `AsOf` per `24 §7.2.1` without YAML override admission. **Milestone 2** (or a later additive MINOR) extends the `Joinset` YAML override surface to accept `AsOf`, opening the narrowing-or-forcing escape hatch documented in `24 §5.3 / §7.2.2`. The milestone team retains freedom to revisit this sequencing if early implementation surfaces an unexpected coupling. Round-1 framing retained for historical reference.

**Question.** `16 §13.3` ratifies per-traversal `JoinType` overrides inside a `Joinset` declaration. Round-1 overrides are limited to `Inner / Left / Right / Full`. Should the `Joinset` YAML surface admit `AsOf` overrides **before** the planner supports implicit `AsOf` synthesis (i.e. author-forced `AsOf` as the path to adoption), **after** the planner supports implicit synthesis (so the Joinset override is a narrowing choice), or **never** (the Joinset always forbids `AsOf`; authors must rely on implicit synthesis only)?

**Refs.**

- `16 §13.3` — Joinset override surface.
- `17 §5.1` — `JoinType::AsOf` ratified, implementation DEFERRED.
- `17 §5.5` — records the `Joinset` `AsOf` override DEFERRAL.
- `17 §10` D1, D3 — DEFERRED items.

**Options.**

- **A. Author-forced-first.** Admit `AsOf` in the `Joinset` YAML now; planner handles it on Joinset traversals before implicit synthesis lands.
- **B. Implicit-first.** Planner implements implicit `AsOf` synthesis first; Joinset-level override adds later as a narrowing mechanism.
- **C. Never.** `Joinset` overrides stay `Inner / Left / Right / Full` permanently.

**Current position.** Joinset `AsOf` override is DEFERRED; Option B is the implicit default. No `AsOf` admitted in `Joinset` until the implicit-synthesis algorithm ships.

---

## Q-TEMPORAL-005 — Default-current row semantics for SCD without `current_flag_dim`

**CLOSED (Phase-3 cascade, 2026-04-17).** Ratified via `foundations/18_entities.md §3.3`: v1 `ScdBody` does not carry a `current_flag_dim` field at all — the v1 roster `{Type1, Type2}` uses `valid_from` / `valid_to` exclusively. Default-current selection on `Type2` is the `valid_to IS NULL` convention (the open-ended-window signal); max-`valid_from` per entity is retained as a secondary fallback heuristic with `PLAN_W_1731` advisory. `current_flag_dim` re-enters the spec only if a future roster extension (e.g., `Type6`) reintroduces it. The sentinel-aware Option B and refuse-without-signal Option C stay deferred.

**Question.** `17 §6.3` specifies that default-current selection on an `Scd::Type2 / Type5 / Type6` kind looks for `current_flag_dim = TRUE` when the flag Dim is declared, else falls back to the `valid_to_dim IS NULL` (open-ended) convention. When neither signal is available — the author declared a `Type2` SCD with no `current_flag_dim` and uses a sentinel value for `valid_to` — the planner emits `PLAN_W_1731 ScdCurrentRowHeuristic` and picks the row with the maximum `valid_from` per entity. Is this heuristic ratified semantics, or a placeholder?

**Refs.**

- `17 §6.3` — the heuristic.
- `17 §10` D13 — DEFERRED.
- `registry/temporal_shape_mapping.md` (pending) — per-engine sentinel conventions.

**Options.**

- **A. Heuristic is ratified.** Max-`valid_from` per entity is the canonical "current row" when no flag is declared.
- **B. Explicit sentinel-aware ratification.** The author declares `valid_to_sentinel: "9999-12-31"` (or `NULL`) on the SCD shape.
- **C. Refuse without current-signal.** Treat "author declared an SCD history-preserving subtype without `current_flag_dim` or sentinel" as a compile-time error.

**Current position.** Option A as Round-1 default with `PLAN_W_1731` advisory. Option B likely lands in a future MINOR once adapter sentinel conventions are ratified.

---

## Q-TEMPORAL-006 — Append-only enforcement for `Scd::Type0`

**CLOSED (Phase-3 cascade, 2026-04-17).** Ratified via `foundations/18_entities.md §3.3`: **`Scd::Type0` is not in the v1 roster**. The v1 `ScdType` roster is trimmed to `{Type1, Type2}`, `#[non_exhaustive]`. The append-only-enforcement question is moot for v1; it re-enters the spec only if a future roster extension re-includes `Type0`. The "out of scope for the semantic layer" disposition (Option A) remains the design guidance for any future reintroduction.

**Question.** `Scd::Type0` — "retain original; no updates after insert" — is a runtime-behavior promise: once a row is written, it is never re-written. Should semstrait validate this at query time, at ingest time (out of scope for the semantic layer), or neither?

**Refs.**

- `17 §2.2` — `Type0` definition.
- `17 §10` — DEFERRED roster; `Type0` runtime enforcement not listed.

**Options.**

- **A. Neither. Runtime-invariant out of scope.**
- **B. Advisory at query time.**
- **C. Advisory at compile time on the DataKind declaration.**

**Current position.** Option A. `Type0` carries vocabulary meaning only; no runtime / query-time enforcement in Round 1.

---

## Q-TEMPORAL-007 — Hoisting `TemporalShape` to `ComplexDataKind`? — CLOSED (2026-04-28)

**Status: CLOSED.** Option A (no hoisting) ratified. `ComplexDataKind` shape propagates via `16 §8` composition rules; no `temporal_shape:` block on the complex variants in v1. Shape hoisting is MINOR per I10 and can land later. Round-1 framing retained for historical reference.

**Question.** `17 §3.2` ratifies that `ComplexDataKind` (`Unionset`, `Grainset`, `Joinset`) does not carry its own `temporal_shape:` block; shape propagates via §8's composition rules. But a `Joinset` with a single root child (say the root is a `Timeseries { grain: Day }`) could reasonably inherit its root's shape as a first-class property — the `Joinset`'s observation-cadence is the root's cadence. Should `17` ratify shape hoisting for these cases?

**Refs.**

- `17 §3.2` — `ComplexDataKind` shape stance.
- `16 §5.3` — `CompositionKind` hierarchy.
- `22 §…` (pending) — Grainset root; `24 §…` (pending) — Joinset root.

**Options.**

- **A. No hoisting (current).**
- **B. Hoist for Joinset when root is shape-classified.**
- **C. Hoist universally.**

**Current position.** Option A. Hoisting is MINOR per I10 and can land as an optimization later.

---

## Q-TEMPORAL-008 — `AsOfAnchor` shape: per-family enum vs tagged struct — CLOSED (2026-04-28)

**Status: CLOSED.** Option A (per-family enum) ratified — `AsOfAnchor::ScdWindow { .. }` / `AsOfAnchor::SnapshotLatestAtOrBefore { .. }`. Matches `TemporalShape`'s own per-variant payload style; exhaustiveness-checking on `match` lets `35` / `36` downstream emitters catch unhandled anchor families at compile time. Round-1 framing retained for historical reference.

**Question.** `17 §5.1` specifies `AsOfAnchor` as a `#[non_exhaustive]` enum with two variants (`ScdWindow { ... }`, `SnapshotLatestAtOrBefore { ... }`). An alternative is a tagged struct with optional fields. Which is canonical?

**Refs.**

- `17 §5.1` — current enum form.
- Q-TEMPORAL-002 — a parallel Rust-model-shape choice for SCD payloads.

**Options.**

- **A. Per-family enum (current).**
- **B. Tagged struct.**

**Current position.** Option A ratified.
