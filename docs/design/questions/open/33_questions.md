---
doc: design/questions/open/33_questions
status: Living
purpose: Round-1 open items raised against `apis/33_semstrait_manifest.md`
---

# 33 — Open Questions

Unresolved items arising while drafting `docs/design/apis/33_semstrait_manifest.md`.

Closed:
- [`../closed/33_questions.md`](../closed/33_questions.md)

Deferred:
- [`../deferred/33_questions.md`](../deferred/33_questions.md)

---

## Q1 — `compile` input: by value or by reference

Should `compile` accept `SemanticModel` by value (consuming it) or by `&SemanticModel`?

Why: by-value matches the stage boundary and avoids a `Clone` bound; by-ref would let A/B-compile against two catalog snapshots skip the clone.

Current default: by-value per `33 §9.1`/`§9.2`.

Trigger to reopen: if A/B-compile becomes common in `semstrait-api`, add an `&Arc<SemanticModel>` overload (do not change the primary signature).

---

## Q2 — `CompileError` as unified enum vs split

Should `manifest::CompileError` stay one enum embedding `ir::CompileError` via D.ii kind-nesting, or split into separate `IrCompileError` + `ManifestCompileError`?

Why: unified lets callers match one enum (`30 §7.4` standard pattern); split gives clearer crate-boundary semantics.

Current default: unified per `33 §10.1`. Tracked as `[TD-33-ERROR-UNIFY]`.

Trigger to reopen: if a third compile-like stage introduces a third error bucket clearer as a separate enum.

---

## Q3 — `canonical_bytes()` encoder choice

Should `canonical_bytes()` use bincode-with-sorted-fields or `serde_json` with `preserve_order`?

Why: bincode is smaller/faster (bytes are hashed, not read); JSON is human-diffable for debugging determinism failures.

Current default: bincode per `33 §14.3`. Tracked as `[TD-33-CANONICAL-JSON]`.

Trigger to reopen: after the first content-addressable caching benchmarks — switch to JSON if the byte-size delta is negligible.

---

## Q4 — `Repository::save` content-hash pre-condition

Should `Repository::save` verify the incoming manifest's derived `SemanticManifestId` matches its content before persisting?

Why: catches corrupt/hand-crafted manifests at save-time; cost is extra hashing per save (non-trivial at scale).

Current default: no pre-condition in v1 (`33 §11.1`; `IntegrityViolation` covers the case at load).

Trigger to reopen: if `IntegrityViolation` reports become frequent — add `Repository::save_checked(manifest, expected_id)` as a MINOR.

---

## Q5 — `SemanticManifestEncoding::Bincode` enable-in-v1

Should `SemanticManifestEncoding::Bincode` be exposed in v1 alongside `MessagePack` and `Json`?

Why: fastest/smallest for the single-machine case; but non-self-describing (a schema migration needs a side-channel version tag).

Current default: deferred per `33 §14.2`. Tracked as `[TD-33-BINCODE]`.

Trigger to reopen: if a concrete consumer needs bincode's characteristics — MINOR addition any time.

---

## Q6 — `FileSystemRepository` locking / atomic-write discipline

Should `FileSystemRepository` carry locking / tempfile+rename to protect against concurrent writers and partial-write-on-crash?

Why: a crash mid-write leaves a half-written file (decode error on load); fix is well-known (`tempfile::persist`) but advisory locks vary by filesystem.

Current default: no locks in v1 (`33 §11.4`; concurrent saves of the same id are idempotent no-ops).

Trigger to reopen: add tempfile-and-rename in a PATCH; MINOR only if the trait signature changes.

---

## Q7 — `check_schema_drift` trait placement

Should `check_schema_drift` live on `CatalogProvider` or on a separate `DriftChecker` trait?

Why: single trait keeps the I/O surface narrow; split would let a repository-only caller skip a full `CatalogProvider`.

Current default: single-trait per the `37` ratification; `33 §12` defers via forward-ref.

Trigger to reopen: decided as part of `37`; `33` updates the §12 forward-ref to match.

---

## Q8 — `SemanticManifest::datakind` / `relationship` return shape

Should `SemanticManifest::datakind(name)` / `relationship(id)` return `Option<&T>` or `Result<&T, SemanticManifestLookupError>`?

Why: `Option` matches `BTreeMap::get` idioms; `Result` makes the missing-entry path impossible to ignore and routes through Diagnostic.

Current default: `Option` per `33 §3.4`.

Trigger to reopen: if error-path ergonomics hurt — add `datakind_or_err` helpers as MINOR.

---

## Q9 — `ResolvedRelationshipGraph` public-field vs accessor — CLOSED (2026-05-28)

> **Moved to [`../closed/33_questions.md`](../closed/33_questions.md).** Dissolved by C17(d) + C9.6 (Manifest Ratification Log, 2026-05-28): manifest carries primitives only; relationship-graph topology is constructed in the planner runtime (daggy per `34 §1.4A`). No graph object on the manifest, hence no field-vs-accessor choice.

---

## Q10 — Compile catalog-error accumulation

Should `compile` accumulate all catalog-related errors (e.g. every `SourceNotFound`) in one pass, or stay strictly fail-fast per `10 §5`?

Why: catalog errors are pre-expression-resolution and don't cascade, so accumulating gives a complete picture; but it adds an exception to a uniform policy.

Current default: strict fail-fast per `33 §10.4`.

Trigger to reopen: if real-world traces show users hitting multiple catalog errors per compile routinely — add a `compile_collecting_catalog_errors` variant as MINOR.
