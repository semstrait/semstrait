---
doc: design/questions/deferred/32_questions
status: Deferred
purpose: Items deferred to v2 (or later) ratification, originally raised against `apis/32_semstrait_model.md`
---

# Deferred Questions — `apis/32_semstrait_model.md`

> Items deferred to v2 (or later) ratification. Live items are in [`../open/32_questions.md`](../open/32_questions.md); closed items in [`../closed/32_questions.md`](../closed/32_questions.md).

---

## Q-MODEL-006 — `AggregationConstraints.allowed` / `.prohibited` ordering — DEFERRED

**Status: DEFERRED.** Parked: blocks on the constraints session (see [`../deferred/11_questions.md`](11_questions.md)). The question resolves itself if (a) `31 §6.3` matching becomes order-sensitive (then sorting is wrong), or (b) the content-hashing use case materializes (then sort — at the hashing boundary, not at parse). Neither is a v1 concern.

**Question.** `31 §6.3` exposes `AggregationConstraints { allowed: Vec<String>, prohibited: Vec<String> }`. `32 §12.2` commits to preserving YAML author order in both vectors for I4. `31`'s matching algorithm is token-based and order-insensitive. Is preserving author order correct, or should `32` sort the vectors for a stronger canonical form?

**Refs.**

- `31 §6.3` — token-based matching; order does not affect semantics.
- `32 §12.2` — YAML author order preserved for I4 determinism.
- `30 §2` — stability rules for public fields.
- `11 §8.4.1` — constraint DSL YAML shape.

**Arguments for preserving author order.**

- Authors may have ordered the list intentionally (e.g. preferred-first policy for a future ordering-sensitive feature).
- Matches the `serde_yaml` round-trip — sorting at parse changes the on-disk form when re-serialized.
- I4 is satisfied either way (both author order and sort order are deterministic); author order matches the input byte sequence.

**Arguments for sorting.**

- Canonicalisation: `{ allowed: [SUM, MIN] }` and `{ allowed: [MIN, SUM] }` become byte-identical, which enables content-hashed caching at a lower level than the full `SemanticModel`.
- Diagnostic messages citing the constraint can be position-stable.
- Removes a trap: if `31 §6.3`'s matching ever becomes order-sensitive (e.g. "first match wins" for conflict resolution), the doc says "order doesn't matter" but code behaviour would depend on it.

**Current position in `32`.** YAML author order preserved per §12.2. The vectors are not sorted at parse.

---

## Q-MODEL-007 — YAML crate choice (`serde_yaml` vs `yaml-rust2` / `saphyr`) — DEFERRED

**Status: DEFERRED.** Tracked as `[TD-MODEL-YAML-CRATE]` in `32 §15`; pre-v1 migration not blocking. Migration lands post-v1 as a transparent internal swap — the public surface (`ParseError`, `parse`) is stable.

**Question.** Current code uses `serde_yaml` (upstream archived March 2024). `32`'s dependency posture (`§13.4`) assumes `serde_yaml`. Should the crate migrate to a maintained alternative (`yaml-rust2`, `saphyr`) before v1?

**Refs.**

- `32 §13.4` — dependency posture table lists `serde_yaml`.
- `crates/semstrait-model/src/parse.rs` — current parser uses `serde_yaml` throughout.
- Upstream: `serde_yaml` archived by maintainer; `yaml-rust2` is active; `saphyr` is the emerging alternative.

**Arguments for migration (pre-v1).**

- Unmaintained dependencies are a supply-chain risk.
- Error-quality improvements in `yaml-rust2` / `saphyr` (span tracking, incremental parse) would strengthen `ParseError.location: Option<Location>` (§11.1).
- Migrating post-v1 is a breaking change if any `ParseError` variant message embeds `serde_yaml`-specific strings.

**Arguments against migration (pre-v1).**

- `serde_yaml`'s API is `serde`-idiomatic; alternatives require bespoke deserialization plumbing.
- The crate is functional and stable. "Archived" ≠ "broken."
- Migration cost: non-trivial; the current parse code uses `serde_yaml` idioms throughout (`#[derive(Deserialize)]`, `serde_yaml::Value`, etc.).
- v1 shipping is the priority; migrations can land as `[TD-MODEL-YAML-CRATE]` in v1.x.

**Current position in `32`.** `serde_yaml` remains the v1 choice. Tracked as `[TD-MODEL-YAML-CRATE]` in §15.

**Next step.** Monitor. If a concrete parse-error-quality blocker emerges (e.g. "line / column tracking on `YamlSyntax` errors is too poor"), re-open and spike a `yaml-rust2` / `saphyr` adapter.

---

## Q-MODEL-D03 — Thread A: model-as-truth posture for manifest projection — DEFERRED

**Status: DEFERRED.** Tabled at the 2026-05-28 manifest ratification cascade. Thread A (the broader question of how `SemanticManifest` should relate to the authored `SemanticModel`) is a model-layer concern that does not block the manifest ratifications themselves. v1 posture: `SemanticManifest` is a self-contained artifact that copies primitive fields by value from the Model (per C17(d) / C18 / CX1), supporting content-addressable caching without Model availability. The alternative — `SemanticManifest` holds an `Arc<SemanticModel>` and projects fields lazily — is parked.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — Thread A (model-layer concern, deferred from manifest cascade).
- `32 §...` — `SemanticModel` ownership discipline (parse returns owned).
- `33 §3.4` — `SemanticManifest` field roster (post-C17 cascade).
- See sibling: [`15_questions.md`](15_questions.md) Q-MAP-D02 — binding-layer counterpart.

**Open axes (preserved for v2).**

- **Lifecycle decoupling.** `SemanticManifest` content-hash today depends on copied bytes; an `Arc<SemanticModel>` projection would let manifest IDs match across Models that share core structure.
- **Disk artifact size.** A manifest persisted to disk needs to round-trip without the Model; lazy projection complicates persistence.
- **Cross-Model sharing.** If two Models share most of their semantic graph, can manifest fragments be shared? (Out of v1 scope; future tooling concern.)

**Next step.** Address at v2 manifest-vs-Model layering ratification pass. Migration MAJOR if `SemanticManifest` public field shape changes.

---

## Q-MODEL-D04 — OpenAPI/JSON Pointer/decorator in core model contract — DEFERRED

**Status: DEFERRED.** Tabled in the 2026-05-28 semstrait-model round after scope review. The proposed direction ("core model accepts OpenAPI-style refs/decorators now") materially expands parse grammar, validation, and compatibility surface beyond the current internal-refactor scope.

**Question.** Should `apis/32_semstrait_model.md` directly adopt OpenAPI-style `$ref` / JSON Pointer and decorator semantics in the core authoring contract, or should web-API interoperability remain a separate projection/sidecar concern?

**Refs.**

- `32 §9` / `§10` — parse/build and crate-boundary posture.
- `33` CX1 — manifest/IR load-time integrity posture and wire-shape constraints.
- `30 §2` — stability and MAJOR/MINOR classification for contract changes.

**Current position in `32`.** Deferred. The v1 model contract carries an `id: EntityId` on each named entity and keys its collections by that `id` (name-ordered output) per the id-first rework (`32 §2`, STATUS item U); no OpenAPI-specific `$ref` / decorator grammar is added to the core shape.

**Deferred axes (v2 candidate).**

- Model-level normalization strategy (accept OpenAPI-like authoring, lower to canonical model before compile).
- Decorator namespace policy and deterministic canonicalization rules.
- Compatibility envelope with manifest/IR wire contracts (no leakage of unresolved OpenAPI refs).

**Next step.** Re-open in a dedicated interoperability session with explicit boundary pick (core normalization vs sidecar projection), migration plan, and compatibility tests.
