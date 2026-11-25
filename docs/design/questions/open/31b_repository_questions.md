---
doc: design/questions/open/31b_repository_questions
status: Living
purpose: Open items raised against the proposed `apis/31b_semstrait_common_io/repository.md` persistence seam
---

# 31b Repository Seam — Open Questions

Items arising while drafting the `31b-io-repository` leaf (`Persistable` + `Repository<T>` + `FileSystemRepository<T>`). Each entry restates the question, lists ratified references, and either resolves inline or proposes a lean default for a later decision pass.

**Status (2026-07 simplification pass).** `Q-IO-K` and `Q-IO-M` are **resolved**: the codec stays opaque with **no `CodecError` type** (fold into the existing `EncodingMismatch`), and there is **no `Persistable::validate` hook** (load-time integrity is the one shallow id-vs-key check plus a consumer-owned deep pass). These trim the earlier draft's speculative generality so the seam is the exact generalization of the shipping manifest `RepositoryErrorKind`. **`Q-IO-L` (error home in `common`) is now also resolved** to its own lean default, since `47` T5 implements that shape. `Q-IO-N` (key collision) remains open with a lean default below.

The closed `Q-IO-001..005` decisions (`closed/31b_io_questions.md`) are **not** reopened here: canonical-only dump, v1 backend freeze, no multi-file load, no WASM, and last-writer-wins all carry forward onto the reconciled seam unchanged.

---

## Q-IO-K — `Persistable` codec: bound to serde, or codec-agnostic? — **RESOLVED (opaque, no codec type)**

**Question.** Should `Persistable::to_bytes` / `from_bytes` be defined against a fixed serde codec (JSON today, MessagePack later) or stay opaque (each DTO owns its bytes, no shared codec type)?
**Why.** A shared codec type lets the repository layer log/negotiate encoding and lets `manifest_epoch`-style fingerprints hash a known canonical form; an opaque codec keeps `common` free of any format dependency and lets manifest use binary while model uses YAML/JSON without a common codec enum.
**Ratified references.** `33 §4.1` manifest is JSON-on-disk + `manifest_epoch` over canonical encoding; `32-model-io` is YAML; `31b Q-IO-001` canonical-only dump.
**Resolution (2026-07).** Opaque per-DTO codec: each `Persistable` owns `to_bytes`/`from_bytes` and returns a bare failure **reason** (`Cow<'static, str>`) on encode/decode error. There is **no shared codec enum and no `CodecError` type** in `common` — the earlier draft's dedicated `CodecError` is dropped; the repository folds a codec-failure reason into the existing `RepositoryErrorKind::EncodingMismatch` (`31b-io-repository §5`), the same variant the shipping manifest code already uses (`33`). Encoding is a DTO concern, matching the two DTOs already using different formats. Rationale: a separate codec error type was speculative generality over an error surface that already exists.
**Reopen.** If a caller needs to negotiate encoding at the repository boundary (e.g. content-type on an HTTP backend), add a `Persistable::CONTENT_TYPE: &'static str` associated const (MINOR) rather than a codec enum.

---

## Q-IO-L — `RepositoryErrorKind` home: `common` vs per-DTO crate — **RESOLVED (single home in `common`; `33` re-exports)**

**Question.** The proposed leaf moves `RepositoryErrorKind` into `semstrait-common`. The concrete manifest Repository (`33`) currently defines its own in `semstrait-manifest`. Should the common definition be the single home, with `33` re-exporting it?
**Why.** One taxonomy keeps `38 §6`'s `SemStraitErrorKind::Repository(RepositoryErrorKind)` wrapper and every DTO's repository sharing one error surface; a per-crate error would force `38` to choose one or sum several.
**Ratified references.** `38 §6.2` wraps a single `RepositoryErrorKind`; `33 §3` exports a `Repository` module; `[TD-GENERIFY-REPOSITORY]`.
**Resolution (2026-07).** The lean default stands, adopted as-is: **single home in `common`**, `33` re-exports for source compatibility, and the manifest impl folds `FileSystemErrorKind` into it. Grounded in `38 §6.2`, which already ratifies a single-wrapper shape — a per-crate error would force that wrapper to choose one taxonomy or sum several. Resolved (rather than left as a default) because `47` T5 implements exactly this shape, so it is no longer a deferred choice.
**Reopen.** Only if a DTO needs a persistence error `common` cannot express; extend the `#[non_exhaustive]` enum (MINOR) before forking it.

---

## Q-IO-M — Load-time integrity placement (`Repository::load` vs DTO) — **RESOLVED (no validate hook; caller runs deep checks)**

**Question.** Should `Repository::load` run a DTO's load-time integrity pass (e.g. manifest CX1, `33 §10`) and surface `RepositoryErrorKind::IntegrityCheckFailed`, or does the caller invoke `validate()` after `from_bytes`?
**Why.** Running it inside `load` makes a repository-sourced artifact safe by construction (a hand-edited/corrupted blob cannot escape the load boundary); leaving it to the caller keeps `Repository` a pure transport+codec seam and avoids a `Persistable::validate` hook.
**Ratified references.** `33 §10` CX1 single-pass integrity; `33 §12` I12.17 (load runs CX1, failure blocks planner consumption).
**Resolution (2026-07).** **No `Persistable::validate` hook.** Load-time integrity is fail-fast and shallow: `FileSystemRepository::load` runs exactly the one cheap **id-vs-key** check the shipping manifest code already runs — the decoded value's own id must equal the requested id, else `RepositoryErrorKind::IntegrityCheckFailed` (`31b-io-repository §5`). Reading by id yields the object filed under that id or a `NotFound`; there is no repository-injected deep-validation pass. A DTO's own deep integrity (e.g. manifest CX1) is the **consumer's post-load step**, not a framework hook — this keeps `Repository` a transport+codec seam and matches the user directive that load-time integrity is not a repository concern in the common case. Rationale: a default-on `validate` hook was overreach — it pushed a DTO-specific concern into the generic seam and duplicated a check consumers already own.
**Reopen.** Only if a concrete caller demonstrates that safe-by-construction load (corrupt blob cannot escape the boundary) is worth a hook; prefer a DTO-side `T::from_bytes` that decodes-and-checks over a separate `Repository`-driven `validate`.

---

## Q-IO-N — `kind_prefix` collision across DTOs — **RESOLVED (`base` alone is the namespace; `kind_prefix` deleted)**

**Question.** Is the `"{base}/{kind_prefix}/{id}.json"` key enough, given `EntityId`s are globally unique UUID text (`33 §4.3.1`) — or should the repository namespace by DTO type to prevent a model id and a manifest id ever colliding on one backend?
**Why.** The key path namespaced on two axes: the caller-supplied **`base`** path (`31b-io-repository §4` — one tenant/workspace root separable from another) and `kind_prefix` ("manifests" vs "models", separating DTO kinds under one root); a single-table document backend (Dynamo, §6) keyed only by id would collide.
**Ratified references.** `31b-io-repository §4` key derivation (`base` ⧺ `id` ⧺ `extension`); `33 §4.3.1` UUID-text global uniqueness.
**Resolution (2026-07).** The question is **dissolved, not decided**: `47 §4a` **R2** deleted `kind_prefix()` on the ground that one `Repository<T>` instance is monomorphic in `T`, so there is no second DTO kind under a given repository's root to separate from. Cross-DTO collision needs two kinds sharing one namespace, which the type no longer permits — a caller who wants two kinds co-located constructs two repositories and gives them two `base` values. `base` alone is the namespace; the per-DTO `extension()` incidentally distinguishes the two anyway when bases *are* shared. No global cross-DTO id registry.
**Reopen.** If a `Repository` impl ever becomes polymorphic over its DTO (a single handle serving several `T`), the separation this question asked about must be re-established — and then on the *key*, not on a deleted trait method.

---

## Q-IO-O — traversal at the persistence seam: who defines the contract?

**Question.** A key bearing a `..` or `.` segment is **accepted and stored** by `InMemoryFileSystem` and **refused** by every `object_store`-backed backend. Which of the two defines the seam's contract — and does `common` validate keys itself, or keep delegating?
**Why.** Today the seam's guarantee is "whatever `object_store::path::Path::parse` refuses, plus nothing for the in-memory transport" (`31b-io-repository §4.1`), which means a suite that only exercises the in-memory double cannot see a save a real backend would reject. **Two** caller-controlled doors compose such a key, both measured: `extension()` is `""` with the id `"."`; and `base`, which `key_for` never inspects. It was three until `47 §4a` **R2** deleted `kind_prefix()` (which could return `".."`); the count is restated rather than silently edited, because a completeness claim quietly shrinking is the defect this enumeration exists to catch. `key_for`'s existing guard is about **injectivity** (a separator makes id→key non-injective), not containment, so nothing here is a regression — but nothing here is a sandbox either, and the leaf now says so rather than implying otherwise.
**Ratified references.** `31b-io-repository §4.1` (the guard is a collision guard; the seam's actual guarantee); `37 §6.5` (`InMemoryFileSystem` keys verbatim and is the less strict transport); `37 §5.3` (`write` overwrites, atomicity is provider-defined).
**Default.** Keep delegating, and keep both halves pinned by test — the current shape. Rationale is a survey of the callers, not a guarantee of the types: every shipped `extension()` returns a literal (`"json"` for the manifest, `"yaml"` for the model), and no production `FileSystemRepository::new` derives `base` from data. Note what this is *not*: `fn extension() -> &'static str` is a function, so `&'static str` bounds the lifetime, not the origin — a conforming impl may return runtime-computed, environment-derived text (measured on the since-deleted `kind_prefix()`, whose signature was identical: `Box::leak(std::env::var(…))` returns `".."` and the key composes). The convention holds in v1; the type system does not enforce it.
**Reopen.** If a caller ever composes `base` or an `extension` from untrusted input, or if a backend lands whose native keys tolerate `..` (a document/KV store per `§6` would), promote containment to a `common`-side check at `key_for` — and decide then whether the in-memory double becomes stricter to match, since a divergent double is what would hide the regression.
