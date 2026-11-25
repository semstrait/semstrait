---
title: Model Loading + Persistence — Implementation Plan
status: to do
effort: layer model's loader/codec over common's io primitives (read/write + Repository + Persistable in common); bless model-root loader.rs (retire doc-32 §5 io-submodule); build LocalFileSystem/S3FileSystem; generify manifest Repository; impl Persistable for SemanticModel (LWW, no IdCollision guard); add the deterministic model-root id that keys it (M1/T7 — authored else derived from name, never time-based)
integration_branch: created at execution off feature/semstrait-mvp (design-only this round — no branch yet)
worktree: .claude/worktrees/docs+io-persistence-seam (authoring home)
spec: docs/design/apis/31b_semstrait_common_io/repository.md; docs/design/apis/32_semstrait_model/model-io-submodule.md (§5 RETIRED by T1); docs/design/apis/37_semstrait_catalog/filesystem.md; docs/design/apis/33_semstrait_manifest/*.md (generify source); questions Q-IO-H / Q-IO-K / Q-IO-005
sibling_baseline: STATUS.md item G (I/O transport — Ratified) + Deferred (c); [TD-BUILD-MODEL-IO]; [TD-GENERIFY-REPOSITORY]
session_ruling: layering ratified 2026-07-24 — common owns lower/reusable io (read/write, Repository, Persistable); model owns higher-level abstractions (loader, parse, serialization). Model never touches the disk. Design-only; each task is approve-then-execute (SPEC_DRIVEN).
---

# 47. Model Loading + Persistence — Implementation Plan

## 1) Layering ruling

Two levels, one direction of dependency (`semstrait-model` already depends on
`semstrait-common` — no new DAG edge needed):

| Level | Crate | Owns |
|---|---|---|
| **Lower / reusable** | `semstrait-common::io` | byte read/write (`FileSystem`), `Repository<T>`, the `Persistable` trait *tag*, error taxonomy |
| **Higher / DTO-specific** | `semstrait-model` | `SemanticModel` loader, parse + `Diagnostics`, serialization (its `Persistable` **impl**) |

**Rules this fixes:**

1. **All disk/object-store contact is common's.** The model crate performs **no**
   filesystem or network I/O — not in the loader, not in save. It produces and
   consumes *bytes*; common moves them.
2. **Encoding is the DTO's, transport is common's.** `Persistable::to_bytes` /
   `from_bytes` for `SemanticModel` live in the model crate (its serde, its
   canonical form). The `write` that puts those bytes anywhere is
   `Repository`/`FileSystem` in common.
3. **The trait tag is common's; the impl is the DTO's.** `trait Persistable` is
   declared in common (so `Repository<T>` can be generic over it); `impl
   Persistable for SemanticModel` lives in model. No DAG inversion: common never
   names `SemanticModel` (`31b §8`).
4. **No `semstrait-model::io` submodule** and **no generic `Loader<T>` in common.**
   Load is DTO-specific (diagnostics, sugar expansion, id-profile minting) — it
   cannot be expressed as `T: DeserializeOwned` without hoisting model-owned
   behavior into common. Loading stays concrete, in `model/loader.rs`.

**The asymmetry, stated once:** *persist* is uniform (canonical bytes keyed by id)
→ generic, in common. *Load* is DTO-specific (warnings, sugar, profiles) →
concrete, in model. Same serde library, two surfaces.

**Boundary correction this ruling forces (verified defect).** The shipped manifest
repository does encoding **and** writing inside one method — `serde_json::to_vec`
at `semstrait-manifest/src/repository.rs:242`, inside `save`. Under rule 2 the
encode step becomes `Persistable::to_bytes` on the DTO and `save` keeps only the
`fs.write`. T5 performs that split; it is behavior-neutral for the manifest.

**Remote load = common fetches bytes, model parses them:**

```rust
let bytes = fs.read(&path).await?;              // common: FileSystem (S3 backend)
let (model, warnings) = SemanticModel::loader() // model: serde, sync, transport-free
    .from_yaml_bytes(&bytes, source)
    .build()?;
```

YAML only (`Q-IO-H`): JSON is not an authoring format anywhere; it is solely the
*manifest's* persistence codec.

## 2) Verified baseline (2026-07-24, post-pull)

Checked against code at this branch's **starting** HEAD — **not** assumed. This table
is a dated snapshot of the pre-implementation state, kept as the record of what was
verified before work began; **three rows are now historical, superseded by §4a**:
`Persistable`/`Repository<T>` exist and `io/` no longer holds `source.rs` / `sink.rs` /
`location.rs` / `convert.rs` (deleted by **R6**); Local and S3 `FileSystem` impls both
shipped; and `io/backends/s3.rs` no longer exists. Read §4a and the code for current
state, never this table.

| Claim | Verified | Evidence |
|---|---|---|
| `model` already depends on `common` | **yes** | `semstrait-model/Cargo.toml:8` |
| `Persistable` / generic `Repository<T>` exist in common | **no — absent** | no hits in `semstrait-common/src/`; `io/` holds `fs.rs`, `source.rs`, `sink.rs`, `backends/`, `location.rs`, `convert.rs`, `error.rs` |
| model has any write/save path | **no — none** | zero `fn save` / `fs::write` / `fn dump` in `semstrait-model/src/` |
| manifest `save` conflates encode + write | **yes** | `manifest/src/repository.rs:242` (`serde_json::to_vec` inside `save`) |
| loader is a **builder** over `SourceFs` | **yes** | `model/src/loader.rs:60,79,89,128`; `from_yaml_file(path: impl Into<String>)` |
| `SemanticModel` has a top-level `EntityId` | **no — absent in code; now specified** | `model/src/model/mod.rs:37` has `name`/`description`/`ai_context`/`labels` + entity maps, no `id`. **M1 ruled** — `32 §1.4a` added, code owes the field (T7) |
| only `InMemoryFileSystem` implements `FileSystem` | **yes** | `common/src/io/fs.rs:232`; no Local/S3 impl |
| `object_store` S3 wiring exists but on retired traits | **yes** | `io/backends/s3.rs` impls `Source`/`Sink` (Lane A, retired) |

## 3) Scenario → task map

| # | Scenario | Today | Task |
|---|---|---|---|
| a | local file **by path** | **DONE** — `loader().from_yaml_file(..).build()` (sync, `LocalFs`) | T1 blesses; no code |
| b | **string** | **DONE** — `parse` / `from_yaml_str` (sync) | T1 blesses; no code |
| c | **S3 / object-store path** | **ABSENT** | T3 + T4 |
| d | **consistent modification (save state)** | **ABSENT for models** | T5 (+T6 wiring) |

## 4) Tasks

Each task carries a **verification** step (how we prove it, not just that it
compiles). `V:` = verification command/criterion.

| # | Crate · file | Adds | Done when | Deps |
|---|---|---|---|---|
| **T1** | docs 32 / 47 | retire §5 io-submodule; bless `loader.rs` | doc 32 load-half `impl: drift` cleared; no `Source`/`Sink` in model-load spec | — |
| **T2** | model · `loader.rs` | `from_yaml_bytes` builder step; reconcile load error roster | bytes step green; non-UTF-8 ⇒ `Io(Malformed)` at `build()`; **no async dep added to model** | T1 |
| **T3** | common · `io/backends/` | `LocalFileSystem` + `S3FileSystem impl FileSystem` | `fs.read(file://)` green; `fs.read(s3://)` green under `io-aws` | — |
| **T4** | api / caller | remote load = `fs.read` → `loader().from_yaml_bytes(..).build()` | integration test loads from `InMemoryFileSystem` (+S3 under `io-aws`) | T2, T3 |
| **T5** | common (trait+repo) · model (impl) | `Persistable` + generic `Repository<T>`; split encode out of `save`; `impl Persistable for SemanticModel` | model save→load round-trip byte-equal; **manifest tests unchanged**; no `fs::write` in model | T3 for real storage |
| **T6** | api | wire `Repository<SemanticModel>` + loader at the compile-from-YAML boundary | end-to-end | T2, T5 |
| **T7** | model · `model/mod.rs`, `yaml/root.rs`, `identity.rs`, `parse.rs` | deterministic model-root `id: Option<EntityId>` (M1) + SR-11 restated as a shape rule (M2) | two parses ⇒ same id; derived v5 round-trips; any-version id parses, nil/uppercase/wrong-length rejected; strict flags it; 4 leaves leave `impl: drift` | — |

**Order:** T1→T2; T3 independent; T4 needs T2+T3; T5 needs T3 only for *real*
storage (works on `InMemoryFileSystem` without it); T6 needs T2+T5. **T7 gates T5's
model half** (no root id ⇒ no `Persistable` impl) and is otherwise independent.

### T1 — Retire the io-submodule; bless `loader.rs` (spec-only)

Supersede doc 32 §5 (`load_model<S: Source>` / `dump_model<S: Sink>`). Load surface
= model-root `loader.rs`; the persist direction becomes `impl Persistable` (T5),
**not** an authoring dumper.

**V:** `grep -rn 'Source\|Sink' docs/design/apis/32_semstrait_model/` returns no
model-load transport types; `doc-space-check.sh` exit 0.

### T2 — `loader.rs` entry point (model, sync)

The live loader is a **builder over `SourceFs`**, not flat associated functions:

```rust
SemanticModel::loader()             // -> SemanticModelLoader<LocalFs>
    .from_yaml_str(yaml, source)    // live
    .build();                       // -> (SemanticModel, Diagnostics<ModelBuildErrorKind>)

SemanticModel::loader()
    .from_yaml_file(path)           // live; path: impl Into<String>
    .build();

// NEW step — UTF-8 validate then stage. Steps return Self; errors defer to build()
// (the live builder pattern), so a non-UTF-8 payload stages as Io(Malformed).
pub fn from_yaml_bytes(mut self, bytes: &[u8], source: SourceId) -> Self;
```

Delta = the one `from_yaml_bytes` step. `SourceFs` stays the *local* reader; remote
transport is composed outside (T4), never bridged into the sync trait.

**Error roster:** `build()` yields `ModelBuildErrorKind` in code; doc 32 §2 names
`ModelLoadErrorKind` on the (retired) submodule. Recommended: keep the live
`ModelBuildErrorKind` and update doc 32's roster to it, rather than add a second
type. `from_yaml_bytes` (authoring parse) and `Persistable::from_bytes` (canonical
decode, T5) are deliberately distinct — different semantics, both on `SemanticModel`.

**V:** unit tests for str/bytes/file parity (same model, same diagnostics);
malformed-UTF-8 fixture ⇒ `Io(Malformed)`; `cargo tree -p semstrait-model | grep -c
tokio` **stays 0** (proves rule 1 — no transport pulled into model).

### T3 — `FileSystem` backends (common)

```rust
// semstrait-common/src/io/backends/
pub struct LocalFileSystem { /* … */ }  // impl FileSystem
pub struct S3FileSystem    { /* … */ }  // impl FileSystem, behind `io-aws`
```

Lift the `object_store` wiring from the retired `Source`/`Sink` backends
(`local.rs`, `s3.rs`) — the engine is proven, only the trait surface changes. The
scheme validator (`fs.rs:92`) already admits `file` + `s3`.

**Gates every remote lane (T4) and all real-storage persistence (T5)** — today only
`InMemoryFileSystem` implements `FileSystem`, so nothing reaches disk or S3.

**V:** round-trip write→read→exists per backend; `file://` + relative paths on
Local; `s3://` under `io-aws`; `gs://`/`abfss://` ⇒ `UnsupportedScheme`.

### T4 — remote load composition (caller, not model)

No new model API. The caller composes common transport + model parser (the §1
snippet). Model stays transport-free.

**V:** integration test loads a model from `InMemoryFileSystem`; behind `io-aws`,
from S3. `gs://`/`abfss://` remain rejected (no impl, no feature).

### T5 — persistence + consistent modification (scenario d)

**Common** gets the trait tag + generic repository (`31b §2/§3/§4`):

```rust
// semstrait-common/src/io/repository.rs (NEW)
pub trait Persistable: Sized {
    fn entity_id(&self) -> &EntityId;
    // `kind_prefix()` was here; struck by §4a R2 (one instance, one type).
    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>>;
}

#[async_trait]
pub trait Repository<T: Persistable>: Send + Sync {
    async fn save(&self, value: &T) -> Result<(), RepositoryErrorKind>;
    async fn load(&self, id: &EntityId) -> Result<T, RepositoryErrorKind>;
    async fn exists(&self, id: &EntityId) -> Result<bool, RepositoryErrorKind>;
}
```

`FileSystemRepository<T>` over `Arc<dyn FileSystem>` + base `Path`. Generifying the
shipped manifest repository (`ManifestIdCollision` → `IdCollision`) is
`[TD-GENERIFY-REPOSITORY]`; it also performs the **encode/write split** (§1) so
`save` no longer calls `serde_json::to_vec` inline.

**Model** gets only the impl (its serde, no I/O):

```rust
// semstrait-model
impl Persistable for SemanticModel {
    fn entity_id(&self) -> &EntityId { /* self.id — post-mint; see M1 */ }
    // no `kind_prefix` — struck by §4a R2; `extension()` is "yaml"

    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>>;      // canonical YAML, name-ordered (I4)
    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>>; // canonical decode, NO diagnostics
}
```

> **M1 — RULED (2026-07-29): add a model-root `id`, deterministic.** The user chose
> option (i), then corrected the derivation: the root id must **not** be regenerated
> per parse. `SemanticModel` gains `id: Option<EntityId>`, resolved as *authored, else
> derived from `name` via UUIDv5* — never from the clock. It is the model's
> persistence key, so `Persistable::entity_id` returns it.
>
> **Why time-based was wrong (the user's correction).** A UUIDv7 root id is minted
> fresh on every parse, so the same model file persists under a different key each
> time it is loaded: `save` writes a new object per session and `load(id)` after a
> re-parse misses. Stability across *content edits* (which UUIDv7 does give, once
> minted) is not the property persistence needs — it needs stability across *parses*,
> which only derivation gives.
>
> **No extra state needed.** The user flagged that this "might require storing
> additional state". It does not: `id` always serializes, so the first `to_bytes`
> writes the derived value into the model's own bytes, and every later load reads it
> as an authored id. The model file is the store — no side-car, no id registry, no
> write-back step.
>
> **Cost of the derivation, accepted and documented** (`32 §1.4a`): models sharing a
> `name` derive the same key and would clobber each other, and renaming a model that
> has no authored `id` orphans its stored object. One save-cycle removes the second
> hazard by materializing the id. Authoring an explicit `id` removes both.
>
> **This reverses a ratified sentence.** `32 §2.3` read *"The model root itself is
> addressed by `name`; it carries no separate `id`"*. Spec amended: new `32 §1.4a`
> (root-id rules), root struct field, §2.3 rewritten, SR-11 restated per M2, SR-12
> explicitly scoped to exclude it, and the minting stage
> (`32-model-parse-validate` step 3 + both profiles) extended to the root. The
> touched leaves move to `impl: drift` — the field is specified, not yet in code
> (T7).
>
> **`Option`, not bare `EntityId`.** `31b:40` types `entity_id(&self) -> &EntityId`
> infallibly, which argues for a bare field; but authoring must allow the id to be
> omitted, and every sibling entity already uses `Option<EntityId>` with parse
> minting it to `Some`. Keeping `Option` preserves that one uniform identity shape
> (`32 §2.3`: *"the `Option` is the sole identity home"*) and matches `31b §4`'s call
> order, which invokes `to_bytes` before `entity_id`. Resolution: `entity_id` stays
> infallible over a post-mint invariant, exactly as `EntityBTreeMap` already treats a
> residual `None` as unreachable-by-construction
> (`model/entity_btree_map.rs:103,228`). A hand-built model that never went through
> parse is the one unminted path — T7 fixes its shape at construction, not by making
> the trait fallible.
>
> **M2 — RULED (2026-07-29): SR-11 constrains id *shape*, not UUID version.** Raised
> by the user against M1's root-only v5 exemption: constrain byte-length/text-shape
> instead, so the id scheme can move — including to content-derived hashes for every
> entity — without amending a ratified rule. Adopted, and it is a convergence rather
> than a change of policy: `31 §2.1` (the `EntityId` owning site) and `33`
> `compile-driver:41` already state that *"the version nibble records provenance but
> is informational — consumers validate canonical-UUID-text shape, not the specific
> version"*. `32` SR-11 was the only leaf asserting a version rule and
> `model/identity.rs:37-40` the only production code enforcing one (the manifest's two
> are `#[cfg(test)]`). SR-11 is now: **36 bytes, `8-4-4-4-12` lowercase hex, non-nil,
> re-parses as a UUID; version unvalidated** — one rule for root and entities, so M1's
> root exemption is deleted rather than widened.
>
> **What it buys.** The version nibble was doing no work: ids are never the ordering
> axis (`32 §2`, `33` — name-ordered projections), so nothing decodes the v7 timestamp,
> and nothing routes on the version. What ids *are* used for is storage keys —
> `31b §4`'s `key_for` interpolates one straight into `"{base}/{id}.{extension}"` (the
> `kind_prefix` segment this once named was struck by §4a **R2**)
> — so fixed length and a hyphen-plus-lowercase-hex charset are the properties that
> actually hold the system up. Pinning the nibble coupled every consumer to the current
> derivation: the rule had already been amended once for that reason (v7-only → v5 for
> generated ids, `closed/33` Q-MAN-D12) and M1 was about to force a second amendment.
>
> **What it adds.** Rejecting the **nil** UUID, which the old version check caught
> incidentally (nil is version 0). Nil passes every layout property but is the
> canonical *absent* value, and `Option<EntityId>` already owns absence — two
> encodings of "no id" is the one gap a pure-layout rule would open.
>
> **Bound, not open-ended.** Ids stay canonical UUID text; this does not admit
> arbitrary strings. A future content-addressed scheme must render its digest into
> that shape (e.g. UUIDv5/v8 over the content), which is exactly what the manifest
> already does at `33 §9.1`. Shape stays fixed; derivation is free.
> **Not carried into the manifest, and excluded from `model_hash`.** The manifest's
> `model_id` is UUIDv5 over `model_hash` (`33 §9.1`, `identity.rs:167`) and must
> change with content. The root `id` is also UUIDv5 but keyed on the *document*, so it
> must not. Keeping it out of the hashed canonical bytes is what stops a rename from
> flipping `model_hash` with no semantic change. The two ids answer different
> questions: which *document* vs which *compilation*.
>
> **SR-11 must become a shape rule, or the id fails to round-trip.**
> `model/identity.rs:37-40` rejects any UUID whose version ≠ 7, so a derived UUIDv5
> root id would serialize on save and then be rejected as `InvalidEntityId` on load —
> a silent round-trip break, not a compile error, and the sharpest edge in T7. The fix
> is **M2** below (restate SR-11 as a shape rule), not a root-only exemption.

**Consistent modification** = `load` → mutate in memory → `save` (whole-object
atomic replace; **no CAS in v1**, `Q-IO-005`; no append/range-patch — `object_store`
offers only whole-object replace). Model `save` is **plain last-writer-wins**.

**Do NOT wire the `IdCollision` guard for `SemanticModel`.** The model-root id is
document-keyed (authored, else derived from `name` — `32 §1.4a`), so it deliberately
survives content edits: "same id, new content" is a *normal edit*. The manifest's
guard exists for **content**-keyed ids (`33 §9.1`), where changed content must change
the id and a clash therefore means corruption; applied to a model it would reject
legitimate edits. `31b §3` already scopes the guard to content-derived DTOs — the
model is exactly the DTO that must not have it. Note both ids are UUIDv5: what
separates them is the seed, not the version.

**V:** `save`→`load` byte-equal on `InMemoryFileSystem`; **two successive saves of
an edited model under the same id both succeed** (proves no IdCollision guard);
manifest repository tests pass unchanged (proves the encode/write split is
behavior-neutral); `grep -rn 'fs::write' crates/semstrait-model/src/` stays empty
(proves rule 1).

### T6 — wire at the API boundary

`Repository<SemanticModel>` + loader at the compile-from-YAML boundary
(`38 §4.2 with_repository`, currently manifest-only). **Constraint:**
`semstrait-api` is excluded from the workspace build (`engine.rs:71` holds the only
end-to-end compile-from-YAML), so this lands when that crate re-enters the build or
the boundary moves.

**V:** end-to-end test: load model from a `FileSystem` → compile → persist manifest
→ persist model → reload both by id.

### T7 — model-root `id` (M1 ruling, code half)

Spec landed (`32 §1.4a`); this is the code that clears the resulting drift.

```rust
// model/src/model/mod.rs — SemanticModel, first field
#[serde(default, skip_serializing_if = "Option::is_none")]
pub id: Option<EntityId>,

// model/src/yaml/root.rs — YamlSemanticModel decode mirror
#[serde(default)]
pub(crate) id: Option<EntityId>,
```

Four sites, in order:

1. **The two structs above.** The root is `deny_unknown_fields` on *both*
   (`model/mod.rs:35`, `yaml/root.rs:39`), so a `semantic_model.id:` key is rejected
   until the `pub(crate)` mirror also gains the field — editing only `model/mod.rs`
   fails confusingly.
2. **A root-id derivation fn** next to `validate_authored_id`
   (`model/identity.rs`): UUIDv5 over a fixed root namespace + the model `name`.
   Deterministic, no clock, no `Uuid::now_v7`.
3. **`parse`** resolves the root id in the same pass that mints entity ids
   (`32-model-parse-validate` step 3): authored ⇒ verbatim; absent ⇒ derived;
   `StrictRequireProvided` ⇒ `MissingEntityId`.
4. **SR-11 as a shape rule** — delete the `version != 7` rejection at
   `model/identity.rs:37-40`; keep the canonical-form re-render check at `:44` and add
   a nil-UUID rejection. This *removes* code rather than branching it: one rule for
   root and entities alike, matching `31 §2.1`. The `uuid` dep stays (v7 mint + parse);
   only `get_version_num` goes.

Three care points. Site 4 is the one that bites: skip it and a derived root id
serializes fine, then fails to load — a silent round-trip break, not a compile
error. `#[derive(Default)]` yields `id: None`, so a hand-built model bypasses
resolution; T5's `Persistable` impl is the only consumer requiring `Some`, which is
why T7 precedes it. And the root `id` now serializes into every model's bytes, so
existing fixtures change: **25 YAML fixtures + 36 inline-YAML sites in `crates/`**
mention `semantic_model:`; any that assert exact serialized output need the root id
line. That count is the real size of T7, and it is why T7 is a code task rather than
a two-line edit.

**V:** two independent parses of the same id-less model yield the **same** root `id`
(the assertion a time-based mint fails); a derived root `id` survives `to_bytes` →
`from_bytes` (proves site 4); authored root `id` round-trips verbatim and wins over
derivation; `StrictRequireProvided` emits `MissingEntityId` for an id-less root; a
v4 root `id` **parses clean** (proves site 4 — the version nibble is unvalidated)
while a nil, uppercase, braced, and wrong-length id each yield
`ParseErrorKind::InvalidEntityId`; a root `id` equal to a contained entity's id
builds clean (SR-12 exclusion); `model_hash` is unchanged by adding the field (proves
the exclusion from canonical bytes); the amended leaves return to
`impl: implemented` with the gate green.

## 4a) Simplification rulings (2026-07-31)

**Nine** rulings. Seven from the cross-check pass — resolving the one blocking item
(`R1`) and converting four open questions into decided shape — plus **R8** and **R9**
from the later user rounds. Each is the governing clause for its task; the leaf edits
they imply are listed per ruling. The count is restated rather than edited in place,
because a "seven rulings" header above nine clauses is how a plan starts lying about
its own scope. **R8** carries code (`W5`); **R9** carries none — it ratifies the
shipped shape and defers its second half out of this doc.

> **R1 — RULED: ids are lowercased on the way to a key, making the seam
> case-insensitive.** Closes the cross-check's only blocking item. Measured hazard:
> saving under `id` and `ID` on a case-insensitive filesystem leaves **one** object,
> and `load(id)` then fails `IntegrityCheckFailed` because the surviving object
> carries the other id — a silent data loss whose mechanism is the *filesystem*, not
> `key_for` (control: two case-differing files in a bare tempdir yield one entry).
>
> **Ruling:** `key_for` lowercases the id before composing the key, so `id` and `ID`
> address one object on every transport. This is a *fold*, not a rejection: no new
> error, and the id space stays `String`. Case is then not an identity axis anywhere
> in the seam.
>
> **Consequences, deliberate.** Two ids differing only in case become the same
> object on *all* backends, including S3 and the in-memory double — which is the
> point: the platform-dependent divergence is what the loss came from. Canonical
> SR-11 ids are already lowercase hex (`32` SR-11 as restated by **M2**), so for
> conforming ids this is identity.
>
> **Three legs, not one — and the third is not closable at this seam.** An earlier
> draft of this ruling said the load-leg compare must fold case "or a lowercased key
> would load a value whose stored id is upper-case and then reject it". That is only
> one of three outcomes and states it unconditionally; it is **struck**. Measured on
> a probe worktree at `df31efc` with *both* legs folded (`key_for` lowercasing +
> `entity_id().eq_ignore_ascii_case(id)`), using a letter-bearing id — note an
> all-digit id like the suite's `ID_A` makes any case probe vacuous:
>
> - **(a) cross-case load** — `save("AAAA…")` then `load("aaaa…")`: without the load-leg
>   fold this is `IntegrityCheckFailed`; with it, `Ok`. The fold closes this.
> - **(b) same-case load** — `save("AAAA…")` then `load("AAAA…")`: `Ok` either way. So a
>   worker reproducing the struck symptom with matching case sees green and would
>   wrongly conclude the load-leg change is unnecessary. It is still needed, for (a),
>   and because the returned DTO's `entity_id()` keeps the *written* case, which then
>   propagates into the caller's own keying.
> - **(c) `RefuseDifferentValue` read-back** — `save` lower then `save` upper on `Twin`:
>   **`Err(IdCollision { kind: "twins", id: "AAAAAAAA-BBBB-5CCC-8DDD-EEEEEEEEEEEE" })`**,
>   *with both legs already folded*. Cause: the guard at `repository.rs:271` compares
>   with `PartialEq` on `T`, and the id is a field of `T`. **The seam cannot fold case
>   inside a DTO's own `PartialEq`**, so (c) is out of reach of any `key_for`/load-leg
>   change.
>
> **Ruling on (c): leave it, and state it.** It is not a data-loss path — the write is
> *refused*, loudly and typed. It is reachable only by a caller that hand-builds an
> id in non-canonical case, which SR-11 already forbids: `validate_authored_id`
> (`model/identity.rs:52`) rejects uppercase at the parse boundary (`parse.rs:319`),
> and `SemanticManifest`'s id is compiler-derived lowercase. So on both shipped DTOs
> (c) is unreachable in practice; only a non-conforming hand-built id finds it. R1's
> scope is therefore **legs (a) and (b)** — `key_for` and the load compare, both or
> neither — and (c) is a documented residue of `EntityId = String` being unvalidated,
> tracked with `TD-022`'s id-shape residue rather than fixed here.
>
> **Not adopted: rejecting non-canonical ids.** Considered and set aside for v1 — it
> would make `key_for` a validator (the `Q-IO-M` posture just ruled out) and would
> fail hand-built ids that work today. Lowercasing fixes the loss without adding a
> gate. `[TD-SEAM-ID-SHAPE]` (`TD-022`) tracks the residue: a junk id like
> `"TOTALLY-NOT-A-UUID"` still files (measured, on disk), because no seam door
> checks id *shape*. Non-blocking — it is a caller precondition, now stated.

> **R2 — RULED: drop `kind_prefix`; one repository instance stores exactly one
> type, rooted at its own `base`.** Resolves `Q-IO-N` and `Q-IO-M`'s namespace half.
> The key becomes `"{base}/{id}.{extension}"`.
>
> **Why.** `FileSystemRepository<T>` is already monomorphic in `T`, so the prefix
> encodes at runtime what the type parameter fixes at compile time. `base` is
> required (`31b §4`) and already separates roots, so the second axis buys nothing:
> two kinds under one root is a *caller* wiring choice — give them two bases. A tree
> is not the seam's concern.
>
> **What this costs, accepted.** `RepositoryErrorKind::IdCollision` carries
> `kind: &'static str` sourced from `kind_prefix()`; with the method gone the
> variant loses that field, so the message names the id and the base rather than the
> DTO kind. Callers that want the kind in a message have the static type at the call
> site. Stored-layout change: existing objects at `{base}/manifests/{id}.json` are
> not migrated — nothing durable exists yet (no released persisted store), so this
> is a pre-v1 layout choice, not a migration.
>
> **`31b §4.1`'s traversal table loses a door — restate the count, don't just delete
> the row.** Measured: the table has **four** data rows and its framing sentence
> reads *"All four key legs are caller-controlled, and each reaches a parse-refused
> key on its own."* Row 3 is the `kind_prefix()` door (`".."`, `"."`, `""`,
> `"a/../b"`). R2 removes that leg entirely, so the table becomes **three** rows and
> the framing sentence must say *three*, not four. The closing completeness claim —
> the seam's guarantee is *"exactly whatever `object_store::path::Path::parse`
> refuses"* — must be re-derived over the surviving three legs (`extension()` empty
> with id `"."`; `extension()` containing `/`; `base` bearing a `..`/`.` segment) and
> its accept-side examples re-measured, not re-reasoned.
>
> This is a **reduction in attack surface**: one fewer caller-controlled leg into a
> traversal key, and it is the only one of the four that no caller could reach
> without writing a deliberately hostile `Persistable` impl. State it that way. The
> reason this consequence gets its own paragraph is that a "four doors" completeness
> claim silently becoming three is precisely the defect class this branch has already
> been bitten by; the count is load-bearing, so it gets restated explicitly.

> **R8 — RULED (2026-07-31): `save` is a map `put`. Same id, later write wins —
> always.** *"Common sense of working with collisions in saving states into
> Repository — is to re-write state for same Id (think of hashmap `put`
> behavior)."* This **supersedes** the `IdCollision`-shape question (was `47 §4a`
> CONCERN 1) rather than answering it: the field debate is moot once the variant
> itself goes.
>
> **What it removes.** `SaveGuard` (both variants), `Persistable::save_guard()`,
> `RepositoryErrorKind::IdCollision`, and the read-compare-refuse block in `save`
> — which today costs an `exists` **plus** a full `read` **plus** a decode on
> every manifest save, purely to decide whether to refuse. `save` becomes
> integrity-check, key, encode, write. Measured footprint: **60 references across
> 5 files**, concentrated in `common/src/io/repository.rs` (37).
>
> **Footprint re-measured at W5 dispatch — the count above is stale, and the
> direction it moved matters.** At `914e9b7`: **71 occurrences of
> `SaveGuard`/`save_guard`/`IdCollision` across 6 files** — `common/src/io/repository.rs`
> 45, `common/tests/repository_over_local_fs.rs` 10, `manifest/src/repository.rs` 8,
> `model/src/repository.rs` 6, `io/mod.rs` 1 (the re-export), `model/Cargo.toml` 1 (a
> feature doc-comment naming `SaveGuard`, easy to miss because it is not Rust). The
> R1/R2/R5 waves that landed between the ruling and this dispatch each added sites, so
> a worker briefed on "60 across 5" would have under-swept by 11 and missed a whole
> file. Grep is the inventory; the clause is not.
>
> **Why this is the right default, not merely the simpler one.** A `Repository`
> is a keyed store; `put` semantics are what every caller already predicts from
> the name, and a seam that silently diverges from its namesake is a trap. The
> refusal also could not deliver what it promised: it is **advisory, not atomic**
> (no compare-and-swap on the local backend, `Q-IO-005` closed), so a concurrent
> writer interposes between the check and the write regardless. It bought a
> partial guarantee at the cost of a read-decode on the hot path and a whole
> error variant.
>
> **What is genuinely lost, stated plainly.** The manifest's id is
> content-derived (`33 §9.1`), so a *differing* value under an equal id means the
> id derivation is broken — and `RefuseDifferentValue` was the tripwire for
> exactly that defect. Under R8 that write silently succeeds and the evidence is
> overwritten. This is accepted: a persistence seam is the wrong altitude for an
> identity-derivation invariant. If that check is wanted it belongs where the id
> is *minted* (a compile-time assertion that `derive → id` is injective), which
> fails at the defect rather than at some later save that happens to collide.
> Verified before ruling: **no production code depends on the refusal** — the
> only `IdCollision` consumers outside the seam are two manifest test sites and
> one hand-constructed `message()` case.
>
> **Re-measured at W5 dispatch: the conclusion holds, the enumeration does not.**
> "No production code depends on the refusal" is CONFIRMED — every `IdCollision`
> site outside `common/src/io/repository.rs` is inside a `#[cfg(test)]` module
> (`manifest/src/repository.rs`'s `mod tests` opens at `:52`, `model`'s at `:82`) or
> in an integration test. But the enumeration omits a whole file: `common/tests/
> repository_over_local_fs.rs` holds **three** more consumer sites (`:140`, `:142`,
> `:191`), so it is five test sites across two files plus the `message()` case, not
> three sites in one. A worker briefed on the short list would compile-break on the
> file it never opened.
>
> **Consequence for R1/R2's landed work.** The three tests that pinned collision
> behavior lose their subject and are deleted, not adapted — including
> `id_collision_carries_the_repositorys_own_base`, added one wave ago to close a
> real coverage gap. Its replacement is the `put` property itself: a differing
> value under an equal id **overwrites**, on both DTOs. Per §4a.2's own rule, a
> test whose name couples a doomed subject to a surviving one must be **split**,
> not deleted — so check each of the three for a surviving half before removing
> it, rather than deleting by name.
>
> **Amended at W5 dispatch — it is not three tests, it is thirteen sites, and
> "delete" is the right disposition for only five of them.** The clause told a worker
> to check for a surviving half but named three tests, so a worker following it
> literally would delete three and leave ten broken compiles — or worse, delete by
> grep and take the survivors with them. Measured inventory at `914e9b7`, with the
> disposition each site actually earns:
>
> **Delete outright — the whole subject is the refusal (5):**
> `refusing_dto_reports_a_collision_and_keeps_the_stored_value`,
> `id_collision_carries_the_repositorys_own_base`,
> `a_collision_message_names_the_id_and_the_base` (all `common/src/io/repository.rs`),
> `diagnose_message_collision_names_id` (`manifest/src/repository.rs`),
> `the_collision_guard_still_sees_its_own_write_over_a_real_backend`
> (`common/tests/…local_fs.rs`). The last one also asserted that a real backend
> normalizes a key the same way on read as on write — but R8 removes the read from the
> save path entirely, so the property has no mechanism left here; confirm before
> deleting that `every_base_form_round_trips_over_a_real_backend` still covers
> write-then-read over the same key.
>
> **Plus one the clause never named:** the test DTO `RefusingButUnencodable` and its
> test `an_unencodable_value_fails_even_when_an_equal_value_is_stored` exist *only* to
> reach the guard's equal-value branch and pin encode-before-guard ordering. With no
> guard there is no ordering. `Unencodable` (`LastWriterWins`) already covers
> codec-failure → `EncodingMismatch`, so the codec path keeps its coverage.
>
> **Invert, do not delete (2):** `a_case_differing_resave_still_collides_via_the_dtos_own_partial_eq`
> and `saving_different_manifest_under_same_id_is_collision` both name behaviour that
> **reverses** rather than disappears. Each becomes the `put` property on its own DTO —
> a differing value under an equal id overwrites, and the second read observes the
> second write. These two ARE the replacement coverage the clause promised; deleting
> them would leave `put` semantics unpinned on both shipped DTOs.
>
> **Split — a doomed assert bundled with live ones (2):**
> `persistable_members_are_callable_without_a_store` also asserts `to_bytes`,
> `from_bytes`, `entity_id` and `extension` (drop only the `save_guard` line), and
> `content_edit_under_a_stable_id_is_a_normal_save_not_a_refused_collision` also pins
> that `SemanticModel`'s id is name-derived so a content edit leaves it stable (keep
> both asserts, drop the `save_guard` line, and rename — "not a refused collision" is
> vacuous once nothing refuses).
>
> **Narrow the roster, keep it exhaustive (2):** `every_variant_renders_its_payload_in_its_message`
> and `severity_is_error_for_every_variant` enumerate all four `RepositoryErrorKind`
> variants. Drop the `IdCollision` arm from each; both must still cover **every**
> surviving variant, because an exhaustive-roster test that silently stops being
> exhaustive is the `#[non_exhaustive]` blind spot these tests exist to cover.
>
> **Re-witness, do not delete (1) — and this is the trap in the wave.**
> `a_scheme_qualified_base_and_its_bare_form_are_one_location_here_and_two_in_memory`
> (`common/tests/…local_fs.rs:191`) is named for `TD-021` and pins the *second measured
> transport divergence*: `"root"` and `"file:///root"` are one location on a real
> backend and two in memory. It merely **borrows** `IdCollision` as the observation
> instrument — the refusal is what made the collapse visible without a second read.
> The property is entirely independent of R8 and must survive it; only the instrument
> dies. Re-witness it the way `put` allows: save value 1 through the bare base, save
> value 2 through the `file:///` base, then **load through the bare base and assert it
> reads 2** — one location, so the second write overwrote the first. The in-memory half
> already needs no change; it asserts two objects, which is still true. Deleting this
> test would silently drop `TD-021`'s only coverage, and its docstring warns in so many
> words that a same-shaped in-memory-only assertion "would pass in memory and be false
> in production."
>
> **Two non-Rust sites are easy to miss:** the `SaveGuard` re-export in
> `common/src/io/mod.rs`, and a feature doc-comment in `model/Cargo.toml:8` naming
> `SaveGuard` — grep for the identifier across the workspace, not just across `.rs`.
>
> Also note `common/tests/…local_fs.rs` has two local `Persistable` impls declaring
> `save_guard` (`:43` `RefuseDifferentValue`, `:231` `LastWriterWins`) — the second
> belongs to `a_dotdot_key_the_memory_double_accepts_is_refused_by_a_real_backend`,
> which is about traversal, not clobber policy, and survives with the method removed.
>
> **Wave placement: W5, after R4, before R7.** It deletes code and comments R7
> would otherwise trim, and it touches `repository.rs`, which R4 does not.

> **R3 — RULED: each state owns its serialization format; the repository never
> names one.** Confirms `Q-IO-H` and `Q-IO-K` as already-shipped shape, and removes
> the last place the seam presumes a format: `EncodingMismatch { expected }` is
> currently filled with `T::extension()`, which reports a *file suffix* as if it
> were an encoding. `expected` goes away; the variant carries the DTO's own failure
> reason. `extension()` stays — it names the stored object's suffix, which is a
> key-composition concern, not a codec one.
>
> **`extension()`'s no-default justification needs re-framing with this.** `31b §4`
> argues `extension()` has no default because *"a hardcoded `json` would file YAML
> bytes under a `.json` key"* — a **codec**-framed argument, i.e. the very conflation
> R3 removes. Re-frame it on key composition: the suffix is part of the object's
> name, and only the DTO knows its own name. Same conclusion, non-contradictory
> reason.
>
> **Breaking-change posture for R2 and R3 (`30 §2`).** Both are non-additive changes
> to public surface: R2 deletes a trait method the leaf declares *"Stable across
> versions; a rename is a MAJOR change"* (`31b §4`), and R3 removes a field from a
> `#[non_exhaustive]` enum's struct variant (`30 §2.2`). Permitted here under
> `30 §2.3` — *"All crates in this workspace are pre-1.0 … MINOR bumps may carry
> breaking changes"* — which waives the version **bump**, not the note: the same
> clause requires *"every MINOR merits the same migration-note discipline MAJOR
> does."* Measured: this repo has no migration-notes file, so the obligation has no
> home. It is discharged **here**, in these two rulings, which state what was
> removed and what a caller does instead (`IdCollision.kind` → the static type at the
> call site; `EncodingMismatch.expected` → the DTO's own `actual` reason). No
> separate migration-notes artifact is created for a pre-v1 seam with no external
> consumers; if one is ever needed, these clauses are its source.

> **R4 — RULED: `FileSystemRepository` derives its backend from the base path's
> scheme.** Resolves `Q-IO-O`'s "who owns the contract" by removing the caller's
> need to know. Today a caller constructs `Arc<dyn FileSystem>` itself and hands it
> in beside a `base`; the scheme in `base` and the backend's identity can disagree,
> and the caller carries `object_store`'s taxonomy.
>
> **Ruling:** add `FileSystemRepository::open(base) -> Result<Self, …>` that parses
> the scheme off `base` and constructs the backend: `s3://` / `s3a://` → S3 (behind
> `io-aws`), `file://` and every scheme-less path → local. (An earlier draft added
> `memory://` → the in-memory double; that is **struck** — see the in-memory
> paragraph below.) The vocabulary is **`object_store::ObjectStoreScheme`'s**,
> measured at 0.13.2 — note `s3s://` is **not** in it (`("s3" | "s3a", Some(_))` are
> the S3 arms), so it is not implemented; `gs`/`az`/`abfss`/`http` parse in upstream
> but have no `FileSystem` backend here and must return `UnsupportedScheme`, not a
> panic.
>
> **A Windows drive letter is not a scheme — dispatch must not treat it as one.**
> Measured: `C:\Users\x` and `C:/Users/x` both parse with scheme `c` under ordinary
> URL parsing, so a naive "split on `:`" sends a local Windows path to
> `UnsupportedScheme`. Two rules, in this order: a single-ASCII-letter scheme is a
> drive letter and dispatches **local**, and a scheme is only a scheme when followed
> by `://`. Anything else scheme-less is local. This must be pinned by test on both
> `C:\` and `C:/` forms; the test runs on every platform because it exercises the
> *dispatch*, not the filesystem.
>
> **`new(fs, base)` stays** as the injection door — tests and callers with a
> pre-built or custom `FileSystem` need it, and it is what every current test uses.
> `open` is the convenience path, not a replacement.
>
> **What `base` means in `open(base)` — one value doing the job of two.** Measured:
> `new(fs, base)` takes **two** independent values today, and every test wires them
> separately (`FileSystemRepository::new(Arc::new(LocalFileSystem::new(dir)),
> Path::new(base))` at `tests/repository_over_local_fs.rs:62-64`, where `dir` is the
> filesystem **root** and `base` is a **key prefix inside it**). `open(base)` has one
> value and must therefore rule which it is. **Ruled: `base` is the whole location —
> the backend's root — and the repository's key prefix is empty**, so the key is
> `"{base}/{id}.{extension}"` with `base` doing exactly the job R2 gives it. No
> root/prefix split: splitting would need a rule for *where*, and that is the tree
> R2 just refused to build.
>
> **Amended while dispatching W3 — "the backend's root" is not one thing, because the
> two backends take different values.** Measured constructor shapes:
> `LocalFileSystem::new(root: impl Into<PathBuf>)` takes a **filesystem root**, while
> `S3FileSystem::new(bucket: impl Into<String>)` takes **only a bucket** — it has no
> prefix field, and its `strip_bucket` removes the literal `s3://{bucket}/` from a
> composed key. So "base is the backend's root, key prefix empty" cannot be applied
> verbatim to both: for a base of `s3://bucket/prefix`, an empty key prefix would
> **discard `prefix` entirely**, silently writing to the bucket root. The two legs
> therefore split, and the split is in *which value the repository keeps as `base`*,
> not in re-introducing a root/prefix axis:
>
> - **S3** — backend gets the extracted bucket; the repository keeps the **whole**
>   original base, so `key_for` composes `s3://bucket/prefix/<id>.<ext>` and
>   `strip_bucket` reduces it to `prefix/<id>.<ext>`. Prefix preserved.
> - **Local** — backend root gets the base with any `file://` stripped; the
>   repository's `base` is **empty**, so `key_for` composes `<id>.<ext>` as a key
>   *under* that root. Keeping the directory in both places would double-apply it
>   (`/data/models/data/models/<id>.yaml`), and a rooted `object_store` cannot serve
>   an absolute key anyway (`local_fs.rs:44-49`).
>
> This is a correction to the ruling's wording, not a reversal of it: `base` is still
> one value the caller supplies and still the whole location. What changes is the
> admission that mapping it onto a backend is per-backend, because the backends do not
> accept the same shape. The W3 worker was dispatched with this flagged as the one open
> question and instructed to verify both legs by round-trip rather than accept it.
>
> **`s3a` normalizes to canonical `s3` in the kept base.** Follows from the S3 leg
> above, and measured: `strip_bucket` (`s3_fs.rs:39`) strips the **literal**
> `format!("s3://{}/", self.bucket)`, and `reject_foreign_scheme` accepts only
> `Some(("s3", _)) | None`. So an `s3a://bucket/prefix` base kept verbatim would both
> miss `strip_bucket` *and* be rejected by its own backend — dispatch would succeed and
> every key would then fail. Ruled: when `open` dispatches an `s3a://` base to S3, the
> base it **keeps** is rewritten to `s3://bucket/prefix`. `s3a` is an addressing alias
> for the same store, not a distinct one, so normalizing at the door is the narrow fix;
> teaching `strip_bucket` and `reject_foreign_scheme` a second spelling would spread one
> alias across two more sites. Pinned by test: an `s3a://` base composes a key
> **identical** to its `s3://` equivalent.
>
> Consequence to accept explicitly: `LocalFileSystem::store()` calls
> `new_with_prefix`, which **canonicalizes and fails on an absent root** (measured,
> and documented at `local_fs.rs:27-30` — the reason `new` is deliberately
> infallible so a caller can `mkdir` after wiring). Since `open` returns `Result`
> already, it may surface that failure — but it must not do so *eagerly at
> construction*, because that would silently drop the create-after-wiring affordance
> `new` exists to preserve. `open` therefore builds the backend handle without
> touching the filesystem; an absent root still surfaces on first `save`/`load`, as
> it does today. `open`'s `Err` is for an **unsupported or unparseable scheme**, not
> for a missing directory.
>
> **`Path`'s scheme vocabulary must widen in the same change, and `memory://` is not
> the shipped convention.** Measured: `is_known_scheme` (`fs.rs:82-84`) is
> `matches!(scheme, "file" | "s3")` and `Path::parse` rejects everything else as
> `UnsupportedScheme`, pinned by `path_parse_rejects_unknown_scheme` (`ftp`) and
> `path_parse_rejects_descoped_schemes` (`gs`/`abfss`/`abfs`). So `s3a://` — which
> this ruling dispatches to S3 — **would not survive `Path::parse` today**.
> `is_known_scheme` widens to `file | s3 | s3a`, the `Path` doc-comment's stated
> vocabulary (`fs.rs:17-25`) follows, and `37-filesystem`'s roster follows.
>
> **Widen by exactly one scheme — `gs`/`abfss` stay refused.** Measured while scoping
> this: `37 catalog-crate-surface §2.3`'s `Path` doc-block says a path **"MUST parse
> to one of"** four forms including `gs://` and `abfss://`, while code rejects both,
> deliberately and by test (`path_parse_rejects_descoped_schemes`, *"descoped
> 2026-06-05"*). `37 §6` likewise rosters `GcsFileSystem`/`AzureFileSystem` under
> `io-gcs`/`io-azure` features that `Cargo.toml` does not declare and types that do
> not exist. That is a pre-existing spec/code tense mismatch, **not R4's to fix** —
> it is `TD-024` (`[TD-37-ROSTER-VS-SHIPPED]`). R4 adds `s3a` because R4's dispatch
> produces it; it must **not** widen to `gs`/`abfss` on the strength of `§2.3`'s
> MUST, which would hand `Path` a scheme with no backing `FileSystem` and defeat the
> refusal test. If a worker finds `§2.3` and the code disagreeing, the code is the
> ruled side.
>
> On the in-memory leg the ruling is corrected: **there is no `memory://` scheme and
> `open` does not introduce one.** Measured: zero `memory://` occurrences in the
> tree; the shipped in-memory convention is the **scheme-less** prefix `mem:` (27
> occurrences across the seam tests), and `InMemoryFileSystem` keys the URI verbatim
> with no scheme concept at all — `fs.rs:22-24` states it normatively (*"in-memory
> objects use scheme-less keys"*). Adding a `memory` scheme would either break that
> or fork the in-memory keyspace against ~27 hard-coded `mem:` keys. So: the
> in-memory double is reachable **only** through `new(fs, base)`, which is what every
> in-memory test already uses; `open` covers `file`/scheme-less → local and
> `s3`/`s3a` → S3, and nothing else. The earlier `memory://` clause is **struck**.
>
> **This does not reopen `Q-CAT-004`, and the apparent conflict with `37 §6.6` is a
> category difference.** Three sites say the crate exposes no scheme dispatch:
> `37-filesystem §6.6`, `37 §12`'s boundary table, and `Q-CAT-004`. Measured, they
> are about a **different facility**: dispatching one call across a *set* of
> filesystems (`Vec<Arc<dyn FileSystem>>`, routing each path to whichever backend
> claims its scheme). `open(base)` selects **one** backend, once, from **one** base at
> construction time, and lives on `FileSystemRepository` in `common` — not on the
> catalog crate root those sites govern. `FileSystem` implementations keep rejecting
> foreign schemes at the call boundary (`37 §5.2`), which is the invariant `§6.6`
> exists to protect.
>
> Two drift notes measured while checking this, both to fix in the `§6.6`
> convergence: `Q-CAT-004` is in `questions/deferred/`, not `questions/open/`, so
> `37-filesystem:160`'s `questions/open/37 Q-CAT-004` pointer is stale; and `§6.6`
> offers `expand_glob` as an injection target, which **R5** established does not
> exist.

> **R5 — RULED: the error surface covers the common cases only.**
> `FileSystemErrorKind` has 9 variants. Measured across the whole tree — including
> the excluded crates and root tests — `InvalidGlob` and `Timeout` have **zero**
> constructors: all 9 grep hits are the variant declaration, its `Diagnose` arm, its
> `From` fold arm, and test cases that construct the variant only to assert its own
> fold. Nothing routes on either. Delete `InvalidGlob`; fold `Timeout` into
> `ConnectionFailed`, which is where a real timeout already arrives — `object_store`
> reports one as `Error::Generic`, and `object_store_fs.rs:40` folds `Generic` to
> `ConnectionFailed`, so the `Timeout` variant was unreachable through the live
> transport, not merely unused. Both fold to the same `IoErrorKind::Network` shape,
> so no observable classification changes. Keep the 7 that backends construct.
> `IoErrorKind`'s 5 stay: each has a live producer.
>
> **`InvalidGlob`'s deletion is a spec change, not a spec alignment.** `37 §8.2`
> documents it as "Surfaces from `expand_glob` (§7.1)", and `§7.1` specs
> `expand_glob` as a **crate-provided** `pub async fn` returning
> `FileSystemErrorKind` — not a consumer concern. (An earlier draft of this ruling
> justified the delete by calling expansion consumer-side; that misread `§7.1` and is
> struck.) The real justification is that `expand_glob` is **unimplemented**: zero
> code hits outside a `common/src/glob.rs` doc-comment, and `common::glob` ships the
> `GlobPattern` newtype with no matching logic. So the variant's only specified
> producer does not exist. `37 §11.3` (*Error-variant stability*) makes variant
> removal MAJOR — permitted here under `30 §2.3` because the crate is pre-1.0 and the
> variant has no producer to break. (Cite `37 §11.3` with its crate number: the
> clause numbered `30 §11.3` is *Cross-crate breaks*, a different rule.) When
> `expand_glob` lands, it re-adds `InvalidGlob` additively under `#[non_exhaustive]`;
> `[TD-023]` records that reciprocal obligation so the delete is not read as a ruling
> that glob errors are unwanted.

> **R6 — RULED: delete the dead Lane-A transport.** `[TD-STRIP-LANE-A-FROM-31]`'s
> trigger has fired — the `FileSystem` backends landed. Measured: `Source`, `Sink`,
> `Location`, `FromIoBytes`, `IntoIoBytes` and `backends::{local, memory, s3}` have
> **zero** consumers outside `common/src/io/` across the whole tree (including the
> excluded crates and root tests); the sole external reference is the `lib.rs:27`
> re-export. 1302 lines, 7 files. Delete them and the `31` sections that document
> them in the same commit (DOC-I6: remove, do not retain) — `31 §5`'s io module
> tree, `§7`'s export list, `§8`'s feature roster, the I11 row, and Testing
> Strategy, per `[TD-STRIP-LANE-A-FROM-31]`. Keep `Path`'s scheme vocabulary, which
> currently cites `Location` as its lockstep partner — after the delete, **R4**'s
> dispatch is the partner.
>
> **Two different `Location`s — delete only the io one.** `common` declares
> `io::Location` (the `Source`/`Sink` backend selector, `31b §6`, deleted here) and
> `diagnostic::Location` (a `SourceId` + `Span` pair, `31 §4.2`, ratified and heavily
> used: `parse.rs`, `builder/`, `yaml/root.rs`). They share a name and nothing else.
> The diagnostic one stays, and so does every `Location::new` call site — a sweep by
> bare name would delete the wrong type.

> **R7 — RULED: comment budget is 1–2 lines, few more only where genuinely
> complex.** Restates `CODE_DESIGN.md` ("comments earn their space", a comment
> longer than its code is a smell) as an enforced gate on this branch's files.
> Measured at `03bccc3`, the commit this ruling was drafted against:
> `io/repository.rs` 203/1424 comment lines with 15 blocks ≥4 lines (longest 9),
> `io/fs.rs` 109/503 with 9 blocks ≥4 lines, the longest being the 16-line `Path`
> doc and the second the 13-line module header. **Re-measured at `df31efc`** (R5+R6
> deleted code and comments in different ratios): `io/repository.rs` **203/1407**,
> `io/fs.rs` **97/472**. Rationale moves to
> the leaf behind a one-line `see …` pointer; the *why* stays, the essay does not.
>
> **Amended at W4 scoping — the pinned figures are now stale, and the original "or it
> is reading a different tree" gate would misfire on a correct worker.** Re-measured at
> `389f21a` (this branch's HEAD before W3/W5 land), with the counting rule stated so it
> is reproducible — a comment line is one matching `^\s*(///|//!|//)`:
>
> | file | comment / total | at `df31efc` | blocks ≥4 | longest |
> |---|---|---|---|---|
> | `io/repository.rs` | **210 / 1375** | 203 / 1407 | 17 | 10 |
> | `io/fs.rs` | **97 / 472** | 97 / 472 | 9 | 10 |
>
> `fs.rs` is unchanged; `repository.rs` moved in **both** directions at once — comments
> **up 7**, code **down 32** — because R1/R3/T5e replaced code with rationale. So a
> worker that measured correctly and compared against the pinned figure would conclude
> it had the wrong tree and stop. The ratio, not the absolute count, is the signal:
> re-measure at your own HEAD with the rule above and report the number; do not gate on
> a figure from three waves ago. **W3 and W5 both land in `repository.rs` before W4, so
> its count will move again — measure last, not from this table.**
>
> **The clause counted 2 files; the seam is 11.** Never measured, and in scope for the
> same budget: `io/error.rs` 5/248, `io/mod.rs` 8/21, `backends/local_fs.rs` 53/390,
> `backends/s3_fs.rs` 31/226, `backends/object_store_fs.rs` 33/244, `backends/mod.rs`
> 2/9, `tests/repository_over_local_fs.rs` 61/270 (**longest block 21** — the module
> header, and the single largest comment block anywhere in the seam), `manifest/src/repository.rs`
> 47/311, `model/src/repository.rs` 57/288. Note that 21-line header is *protected class
> (ii)*: it carries the mutation-measured "three of these six catch something the unit
> suite cannot" plus `TD-021`. Longest ≠ worst — the budget is about comments that do not
> earn their space, not about length alone.
>
> **Two protected classes, not one.** This is a trim of prose, never of (i) a `///`
> contract sentence or a test assertion, and never of (ii) a **comment recording a
> measurement that cannot be re-derived by reading the code**. Class (ii) is the one
> a naive trim destroys, because it looks like prose. Enumerated, all in
> `common/src/io/`: `repository.rs`'s `key_for` doc (the non-injectivity measurement
> `a/b`/`a//b`/`/a/b` → one object, plus its explicit "does NOT guard traversal"
> scope limit); the "no wildcard arm — `#[non_exhaustive]` does not bind within the
> defining crate" note, whose deletion invites someone to "fix" the exhaustive match
> with a `_ =>`; `save`'s four ordering rationales (integrity-before-`entity_id`,
> encode-before-guard, the advisory/non-atomic disclaimer, "a corrupt blob must not
> wedge the id"); `load`'s guard rationale including the measured "not reachable
> through either shipped DTO today"; the `PhantomData<fn() -> T>` variance note;
> `local_fs.rs`'s `new_with_prefix`-canonicalizes-and-fails note (load-bearing for
> **R4**), the `ObjectPath::from("file:///tmp/x")` → `file:/tmp/x` measurement, and
> the why-the-scheme-check-must-precede-conversion note; `object_store_fs.rs`'s
> `fold` non-exhaustive note and `list_literal`'s widen-then-refilter rationale; and
> `tests/repository_over_local_fs.rs`'s module header carrying the mutation-measured
> "three of these six catch something the unit suite cannot" plus `TD-021`. A
> measurement deleted is a measurement that must be re-run to recover.
>
> **The actual defect inventory, measured — it is 6 sites, not a ratio.** Ratios locate
> nothing; `CODE_DESIGN.md`'s rule is *a comment longer than the code it explains*, so
> that is what was counted. Across all 11 seam files at `442dc3e`, the inline (`//`, not
> `///`) blocks of ≥3 lines whose comment count **exceeds** the contiguous code that
> follows them:
>
> | site | cmt / code | disposition |
> |---|---|---|
> | `manifest/src/repository.rs:284-291` | 8 / **0** | **W4 target.** A tombstone for a *deleted* test (`fold_fs_error_maps_each_variant_into_io`), explaining why it is gone and where the property now lives. Zero code. The forward pointer is worth one line; the paragraph is not. |
> | `common/src/io/repository.rs:278-283` | 6 / 3 | **PROTECTED (ii)** — `load`'s guard rationale, incl. the measured "not reachable through either shipped DTO today". Already enumerated above. |
> | `common/src/io/repository.rs:253-256` | 4 / 1 | **W5 deletes it** — the advisory/non-atomic guard disclaimer. |
> | `common/src/io/repository.rs:248-250` | 3 / 1 | **W5 rewrites it** — "encode before the guard" justifies ordering against a guard R8 removes. |
> | `common/src/io/repository.rs:315-317` | 3 / **0** | **W4 target, and W5 makes it false.** Narrates why there are two test DTOs: *"every property that matters here is one where the two must differ: extension and clobber policy."* After R8 there is no clobber policy to differ on, so half the stated reason evaporates. Re-write to name the surviving axis (extension), or drop. |
> | `common/src/io/repository.rs:627-629` | 3 / 1 | **PROTECTED (ii)-adjacent** — "`IntegrityCheckFailed`, not `Io`: nothing was asked of the transport". That is a why-not, exactly the comment class the rule preserves. Keep. |
>
> So W4's own inline footprint is **2 sites** (`manifest:284-291`, `repository.rs:315-317`);
> W5 consumes three more and one is protected. This is why R8 lands first: measuring W4
> before it would inventory work that no longer exists. **W4 re-measures at its own HEAD
> and reports this table refreshed** — the counting script is the definition, not the
> table.
>
> **EXECUTED (W4, 2026-08-03) — the refreshed table, measured at `ab98857`.** Ratios
> across all 11 seam files: **569 / 3826** total, `io/repository.rs` 189/1396 (16 blocks
> ≥4, longest 10), `io/fs.rs` 101/513, `tests/repository_over_local_fs.rs` 58/237
> (longest 21, still the seam's largest block and still protected (ii)). The inline
> inventory refreshed **6 sites → 3**, and of those 3 only **one** was actionable:
>
> | site at `ab98857` | cmt / code | outcome |
> |---|---|---|
> | `manifest/src/repository.rs:261-268` | 8 / **0** | **TRIMMED** to a 2-line forward pointer. Both of its factual claims were verified before compressing, not after: `every_filesystem_error_variant_folds_into_io` exists at `common/src/io/repository.rs:996`, and `fold_fs_error` is gone from the manifest crate (the comment was its only remaining mention). |
> | `common/src/io/repository.rs:301-306` | 6 / 3 | **KEPT — protected (ii).** `load`'s guard rationale, incl. the measured "not reachable through either shipped DTO today". |
> | `common/src/io/repository.rs:612-614` | 3 / 1 | **KEPT — protected (ii).** "`IntegrityCheckFailed`, not `Io`: nothing was asked of the transport." |
>
> The clause's other two W4/W5 targets are **consumed, not skipped**: `repository.rs:315-317`
> (the two-DTO clobber narration) no longer exists after R8 — the only surviving `clobber`
> token in the seam is a one-line test-section header — and `:253-256` / `:248-250` went
> with `SaveGuard`. The line numbers in the pinned table are all stale by construction;
> the three sites above are the same defects renumbered plus two deletions.
>
> **The inline-only scope was itself a blind spot, and it hid one site.** The inventory
> counts `//` blocks, which exempts every `///` doc — including the ones documenting
> *tests*, where "longer than the code it explains" applies just as well because no
> external caller ever reads them. Measured as a subclass: **1 site**,
> `model/src/repository.rs:120-122`, a 3-line tombstone over a 1-line body naming a
> deleted test and a `kind_prefix()` that exists nowhere else in the workspace. Trimmed
> to 2 lines citing `R2`, the file's own convention. Both subclasses now measure **0**
> unprotected sites. Gates on the merged tree: **1127 / 0 / 1** (unchanged — a
> comment-only wave must not move the count), non-aws 0, clippy `-D warnings` 0, fmt 0,
> doc gate 0.

> **R9 — RULED (2026-07-31): `SemanticModel::id` stays `Option`, and the root keeps
> its name-derived id. The `<data_kind>.<name>` proposal is ACCEPTED in principle
> but NOT scheduled here — it is an entity-identity change, not a root-optionality
> one, and it lands only after a scoping gap is closed.** The user's message
> contains two separable halves and states indifference on the first ("I'm ok with
> both"), so this clause decides the first and scopes the second.
>
> **Half 1 — optionality: keep `Option<EntityId>`, already code-filled.** The
> behavior the user asked for is *shipped*: `build()` resolves an absent root id by
> deriving it (`builder/model.rs:376-382`), and `derive_root_id(name)` is UUIDv5
> over a frozen `ROOT_NAMESPACE` + `name` (`model/identity.rs:43-45`) — deterministic
> across parses, machines, and time. Measured: parsing the `alpinestars` fixture
> yields root `32af57bc-219c-5a1c-8b55-9929d97a4318` (a v5 id — note the `5` in
> position 13), stable across runs. So "id could be filled by us on parsing stage"
> is the status quo, not a change.
>
> **But the user named two doors, and only one derives — stated, because the clause
> read as if both did.** The message says "when we parsing input yaml, **or** we
> deserliaze from yaml". Measured, those are different code paths with opposite
> behavior: the authoring door (`SemanticModelLoader` → `parse` → `build()`) derives
> an absent root id, while the persistence door
> (`Persistable::from_bytes`, `crates/semstrait-model/src/repository.rs:69-77`) uses **direct serde** and
> **rejects** id-less bytes with "persisted model bytes must carry a resolved root
> `id`". Deriving there instead would be wrong, and this is the reason: persisted
> bytes are keyed *by* the id the repository was asked for, so silently minting one
> from `name` would let a renamed model decode under a key it does not match, turning
> a corrupt-artifact error into a wrong-object success — and `load`'s id-vs-key check
> is what would then fire, one layer too late to say why. So: derive on the authoring
> path, refuse on the persistence path. The user's "or" is satisfied by *one* door
> deriving; the asymmetry is deliberate and is now on the record rather than implied.
> Choosing `Option` over non-optional is therefore not a coin-flip: making the field
> required would (i) force every authored YAML *or* every code-built model to carry
> an id the compiler cannot derive at field-init time, (ii) delete the
> `IdentityProfile::StrictRequireProvided` distinction, whose entire content is
> *which* absent ids are reported rather than filled, and (iii) buy nothing at the
> persistence seam, which already guards the field. Scope that third argument
> precisely, because the guards do not all cover the same thing: `integrity_check`
> (`crates/semstrait-model/src/repository.rs:38-45`) and `from_bytes` (`:69-77`) both
> reject an absent id on **the root only** — they never walk entities. A third guard
> the earlier draft did not name covers the maps: `EntityBTreeMap`'s hand-written
> `Deserialize` (`entity_btree_map.rs:111`) rejects a residual `None` id *and*
> duplicate ids, but only for pool and top-level maps — not for entities declared
> inside a `SemanticInterface`. So the honest form of (iii) is: making the *root*
> field required buys nothing, because three guards already cover it; it says nothing
> about entity ids, which are a separate question answered in Half 2. The `Option` is
> the parse-time-vs-post-build distinction made explicit in the type; the invariant
> "post-`build()` the root is always `Some`" is enforced at the places that matter.
>
> **Half 2 — `<data_kind>.<name>` derivation: right idea, wrong scope, blocked on a
> measured gap.** The user's third sentence describes what the *root* already does
> and proposes extending it to entities. Today entities do the opposite: `Dimension`,
> `Measure`, `Metric`, `Relationship` and `DataKindBase` all mint **time-based
> UUIDv7** via `generate_entity_id()` = `Uuid::now_v7()`, so their ids vary across
> fresh parses — which `32` ratifies (`model-crate-boundaries.md:69`). I4 survives
> only because output is name-ordered, never id-ordered (`32 §7`,
> `entity_btree_map.rs`). Deterministic entity ids would make that reasoning
> *unnecessary* rather than wrong, which is a real gain: two parses of one file would
> become byte-identical including ids, and a re-parse would stop re-keying every
> entity.
>
> **The blocker is that `<data_kind>.<name>` is not yet a key.** Measured on the
> `alpinestars` fixture at `6f8f44a`, each mutation verified to have actually landed
> before crediting its result:
>
> - **Entities are declared in two places, and the two authoring doors disagree about
>   minting them.** A `SemanticInterface` may declare an entity **inline**
>   (`DimensionEntry::Inline`). On the **parse** door, `ensure_id` is called from
>   exactly the 8 top-level map inserts (`builder/model.rs:265,275,285,295,305,315,325,335`)
>   — never into an interface — so the fixture's inline dimension `market` emerges from
>   `parse` → `build` with **`id: None`**, while every pool entity is `Some`. On the
>   **code** door the opposite holds: 12 further eager-mint sites (`#[builder(skip = …)]`
>   on `dimension.rs:33`, `measure.rs:41`, `metric.rs:44`, `relationship.rs:39`, plus 8
>   `DataKindBase` constructors in `builder/data_kind.rs`) mint unconditionally, and
>   because `DimensionEntry::Inline` wraps a whole `Dimension`, a code-built inline
>   entity **does** carry an id — measured: `Dimension::builder("market")…build()` yields
>   `Some(019f…)` and `DimensionEntry::inline` preserves it. So "no minting site" is true
>   of the **parse door only**; the scheme cannot be applied uniformly until it is ruled
>   whether inline entities are identity-bearing, because today the answer depends on
>   which door built the model.
> - **`StrictRequireProvided` does not see inline entities either.** On the same fixture
>   it produces 25 diagnostics, of which **24** are `MissingEntityId` (the 25th is an
>   unrelated `SemanticsShadowRootPool { carrier: "Dimension", name: "country" }`) — all
>   for pool and data-kind entities, and **0** naming `market`. The strict profile's
>   completeness claim is therefore scoped to the minted set, not to all named entities.
>   *(The earlier draft said "25 `MissingEntityId`", counting the total diagnostic
>   set and mislabelling all of it as one variant.)*
> - **The `<data_kind>` half is load-bearing, and measurement shows it is *necessary*,
>   not merely prudent.** One name may live in two carriers at once: a pool Dimension
>   `cost` alongside a pool Measure `cost` raises **no duplicate-name diagnostic**
>   (measured on a bare model: only `EmptyModel` + two `OrphanSharedSemantics`, neither
>   about duplication). `dedup.rs::collect_duplicate_shared_semantics` calls
>   `collect_pool` three times, each building a *fresh* index (`dedup.rs:113`), so there
>   is no cross-carrier name scan by construction. Measured directly:
>   `derive_root_id("cost")` is necessarily equal to itself across carriers, whereas
>   `derive_root_id("dimension.cost")` = `b4060538-…` ≠ `derive_root_id("measure.cost")`
>   = `9907b3db-…`. Bare-name derivation would therefore **invent** an id collision
>   between two entities that are legal today; the carrier-qualified namespace the user
>   proposed is what prevents it. This part of the proposal is confirmed correct.
> - **The real blocker: name-derivation collapses the name checks into the id check,
>   and loses data doing it.** This supersedes the precondition the earlier draft
>   stated. Measured by mutating the claimed cause — substituting
>   `derive_root_id(name)` for `generate_entity_id()` at the single mint site
>   (`builder/model.rs:86`), guarded to exactly one match — two datasets named `orders`
>   no longer produce `DuplicateDataKindName` at all. They produce
>   `DuplicateEntityId` **instead**, and `builder.datasets.len()` drops from **2 to 1**:
>   the first dataset's body is *silently discarded* (`desc: "FIRST"` gone, `"SECOND"`
>   surviving under the shared key). The mechanism, verified rather than inferred:
>   `collect_duplicate_data_kinds` iterates the **id-keyed** maps (`dedup.rs:31-52`), so
>   when derivation makes two same-named entities share one key the second `insert`
>   overwrites the first and the name scan sees a single entry — it goes *blind* rather
>   than losing a race. SR-12 still fires only because `id_locations` accumulates with
>   `or_default().push` (`model.rs:271` et al.) and so retains both witnesses. The
>   same holds for the shared pool: two measures named `cost` collapse 2 → 1 with
>   `DuplicateEntityId` substituting for `DuplicateSharedSemanticsName`. Control run:
>   reverting to `generate_entity_id()` restores `len() == 2` with two distinct keys and
>   one occurrence each, so the mutation is the cause. Re-measured under the user's
>   exact `<data_kind>.<name>` form at all 8 sites: the namespace changes the UUIDs but
>   **not** the collapse — duplicates are same-kind, so the qualifier cannot separate
>   them. Blast radius: exactly the 3 tests the reviewer named
>   (`convenience_profile_mints_valid_uuidv7_for_idless_entity`,
>   `single_file_duplicate_still_carries_original_location`,
>   `loader_emits_duplicate_when_two_files_share_dataset_name`) fail; a 4th failure seen
>   in the first namespaced attempt was an artifact of the prototype, not the proposal —
>   `ensure_id`'s `name` parameter feeds *both* derivation and the `MissingEntityId`
>   diagnostic path, so namespacing it renamed the diagnostic `"country"` →
>   `"dimension.country"`. Splitting that parameter into `name` (diagnostics) +
>   `derive_key` (derivation) returned the failure set to exactly 3. That split is a
>   design requirement any real implementation inherits.
> - **Duplicate names inside one interface survive.** Three inline dimensions all
>   named `market` in a single interface build green — nothing scans interface-local
>   names for duplicates. `<carrier>.<data_kind>.<name>` would mint **one** id for
>   all three — the same collapse as above, one scope further in.
>
> **A latent defect found while measuring, independent of this ruling.** SR-12 claims
> `id` global uniqueness across "data kinds + semantics + relationships"
> (`model-crate-boundaries.md:31`). Its witness is `builder.id_locations`, populated
> only at those same 8 insert paths. Measured: authoring a pool Dimension and an
> inline dimension with the **same** id (`11111111-…`) builds **green** — SR-12 does
> not see the collision, because the inline entity never reaches the index. Recorded
> as a tracked item rather than fixed inside an io wave; it is a `32`-crate
> validation gap, not a persistence-seam one, and it is *pre-existing* on this
> branch.
>
> **Ruling.** (a) `SemanticModel::id` keeps `Option<EntityId>` with code-fill at
> `build()` — no code change owed, so W1–W5 are unaffected. (b) The
> `<data_kind>.<name>` entity-derivation proposal is accepted as direction and
> **deferred out of doc 47**: it changes ratified `32` surface (entity minting, the
> `IdentityProfile` roster, the I4 rationale, the SR-12 scan) and belongs to the
> model round, not the io round the user sequenced first. Its **binding** precondition
> is not the inline-identity ruling the earlier draft named — that one is real but
> secondary (it decides whether `<carrier>.<data_kind>.<name>` is a *total* function on
> named entities). The blocking precondition is a ruling on **what a duplicate name
> means once names determine ids**: under derivation, two same-named entities become
> one map key, so `DuplicateDataKindName` / `DuplicateSharedSemanticsName` stop firing,
> `DuplicateEntityId` reports in their place, and the first entity's body is silently
> dropped — measured, with a control. Deterministic ids are only safe once duplicate
> names are rejected **before** the id-keyed insert that would swallow one of them; the
> name scan cannot keep living downstream of a map it no longer has two entries in.
> A conforming implementation therefore owes: (i) name-uniqueness validation moved
> ahead of insertion, or run over the authored sequence rather than the map; (ii)
> `ensure_id`'s conflated `name` parameter split into `name` (diagnostics) +
> `derive_key` (derivation); (iii) a decision on inline-entity identity, which the two
> doors currently answer differently. Neither half blocks the io waves.

### 4a.1 Execution order — the waves are serialized on `io/fs.rs`, not disjoint

Four rulings touch **one** file, so "parallel workers on disjoint files" is false for
`io/fs.rs` and must not be assumed: **R5** edits its enum, `Diagnose` arms and tests
(and `repository.rs`'s `From` arms); **R6** edits its doc-comments and `io/mod.rs`;
**R4** must widen `is_known_scheme` and the `Path` doc there; **R7** trims comments
across it. `repository.rs` is likewise shared by R1, R2, R3 and R7.

Ruling: these run **sequentially**, one writer per file at a time, each on its own
worktree branched from the previous merge — never as concurrent file-disjoint workers
on a shared worktree, where a tree-wide `git restore` by one worker clobbers another's
edits regardless of file disjointness. Landed order: **W1** = R6 then R5 (both merged);
**W2** = R1+R2+R3 as a single task (all three land in `repository.rs`); **W3** = R4;
**W5** = R8 (`save` becomes a map `put`); **W4** = R7 last, because it trims comments
the earlier waves rewrite. R8 lands **before** R7 for that same reason — it deletes
code and comments R7 would otherwise spend effort trimming — so the executed order is
W1, W2, W3, W5, W4.

`git stash` is prohibited in any of these worktrees: `refs/stash` is shared across all
worktrees of a repository, so a stash here can capture or drop another worktree's work.

### 4a.2 Spec convergence owed by R1–R4

`31b_semstrait_common_io/repository.md`: `kind_prefix` at `:44, :73, :104, :149, :156,
:170, :204, :226, :263`; `expected` at `:121, :197, :207`; the `§4.1` traversal table
`:164-175` (row 3 dies, count restated per **R2**); the `§5` variant table `:193-198`
and its code block `:200-212`; Testing Strategy `:263`, which pins the literal key
`"{base}/{kind_prefix}/{id}.{extension}"` plus `manifests/<id>.json` and
`models/<id>.yaml` examples. Every traversal-table cell must be **re-measured** over
both transports, not re-reasoned from the old row.

This plan itself contradicts **R2** in three places until converged: `§4`'s T5 trait
sketch (two sites) and the `"{base}/{kind_prefix}/{id}.json"` key literal.
**Discharged:** R1–R3's spec convergence landed with the leaf at `version: 2.0`
(frontmatter and `_manifest.json` both, gate exit 0), and all three plan sites are
struck in place rather than deleted. The traversal table was re-measured over both
transports on a scratch harness — every surviving cell, the four accept-side examples
and the slash-free defusal reproduce; the row-3 vectors were re-pointed
(`base/deep/docs/..` → `base/deep/..`), so the property held while the door closed.
The `use_when` routing string in `_manifest.json` also named `kind_prefix`; the gate
does not read it, so it was fixed by hand.

Code, **R2**: 14 `fn kind_prefix` definition sites and ~12 key-literal assertions across
`common/src/io/repository.rs`, `common/tests/repository_over_local_fs.rs`,
`manifest/src/repository.rs`, `model/src/repository.rs`. Two tests lose their subject
entirely and are deleted, not adapted: `a_collision_names_the_colliding_id_and_the_dtos_own_kind`
and `extension_and_kind_prefix_match_the_model_contract`. **R3** likewise deletes
`the_expected_encoding_is_derived_from_the_dto_not_hardcoded`, which exists solely to
defend the field being removed.

> **Correction — that deletion instruction was over-broad, and cost coverage.**
> `extension_and_kind_prefix_match_the_model_contract` asserted **two** things:
> `kind_prefix() == "models"` (dead under R2) and `extension() == "yaml"` (still
> live). Deleting the whole test dropped the second. Measured after the fact:
> mutating `SemanticModel::extension()` to `"json"` leaves the **entire workspace
> suite green**, while the same mutation on the manifest fails 3 tests — an
> asymmetry, not a convention. A test whose name couples a doomed subject to a
> surviving one must be **split**, not deleted; the instruction should have said so.
>
> A second regression came from the same wave: `IdCollision` traded `kind` for
> `base`, and the test it replaced had explicitly pinned that *both* payload fields
> be **carried, not known**. Mutating the construction site to
> `base: Path::new("WRONG-NOT-THE-BASE")` also leaves the whole suite green — every
> surviving assertion destructures `IdCollision { id, .. }`. Both are fixed in the
> R1 wave, and both are now pinned as a durable requirement in the leaf's Testing
> Strategy so the gap cannot reopen silently.
>
> **The `extension()` asymmetry was mis-explained, and the correction widens the
> fix.** "The same mutation on the manifest fails 3 tests" is measured, but the
> mechanism is not an extension assertion: the manifest crate contains **no**
> `extension()` reference outside its own definition. The 3 failures come from
> `expected_key`'s hardcoded `.json` literal disagreeing with the real `key_for` —
> a key-agreement pin that catches this mutation incidentally. So **neither** DTO
> asserted its extension by value, and the leaf requirement ("each shipped
> `extension()` is asserted, per DTO and by value") was unmet on both, not one.
> `key_scheme_is_deterministic_and_extension_qualified` compounded it by comparing
> `expected_key`'s literal against another copy of the same literal — a tautology
> that cannot fail. Both halves are fixed: the model gained `extension_is_yaml`,
> and the manifest's helper now composes from `SemanticManifest::extension()` so
> the pin flows through the real associated function.

**R2's stale sites beyond the leaf, converged.** `kind_prefix` outlived R2 in five
more files, split by kind. Live text corrected: `TECH_DEBT.md:307` (which carried the
four-vs-three key-leg count in a **second** home, so the leaf's restatement alone left
a contradicting copy), TD-013's remediation prose, `31-semstrait-common.md:114`'s key
literal (plus an R1 note — the charset claim is now what the key *carries*, not merely
what callers pass), and `Q-IO-N`, which the leaf claimed R2 resolved while the register
still listed it open with a `kind_prefix`-based default. `Q-IO-N` is marked **RESOLVED
by dissolution**: R2 removed the second DTO kind rather than deciding how to separate
it. `Q-IO-O`'s door enumeration went 3 → 2 with the reduction restated, and its
`&'static str`-bounds-lifetime-not-origin note re-pointed onto `extension()`, whose
signature is identical to the method it was measured on.

`COVERAGE_MATRIX.md` needed a different treatment, because it is the **standard review
runs against** ("verifies against *this*"), so stale text there misdirects review
rather than merely aging. Its one live instruction — *"The generic message should name
the kind (`T::kind_prefix()`) … do not accept a bare `id collision: {id}`"* — had
**inverted**: R2 makes the shape it forbids the correct one. Marked SUPERSEDED with the
replacement named. Past measurements (`M1`'s mutation, the 6-method `Persistable` set,
the second traversal door) are faithful records of what *was* true, so they carry
one-line markers and keep their original text; the `git show` transcript and the merge
record are left verbatim, being literal quotations.

> **CONCERN 1 — SUPERSEDED by R8, not answered.** The concern was that R2 mandated
> *removing* `kind` from `IdCollision` while the *addition* of `base` was my
> inference, and that `IdCollision { id }` was arguably the shape the "errors must
> be straightforward" directive pointed at. Put to the user as a two-way choice; the
> answer reframed it — `save` is a map `put`, so the whole variant goes and the field
> question dissolves. Recorded because the *question* was mis-framed, which is worth
> keeping: I asked which payload to carry when the live issue was whether the error
> should exist. `base` stays carried and pinned until R8 lands.

`37-filesystem` + `catalog-crate-surface.md`, **R4**: the scheme roster and `Path`
vocabulary, plus `fs.rs`'s `is_known_scheme`, the `Path` doc-comment, and both
`path_parse_rejects_*` tests.

### 4a.3 Spec convergence owed by R8 — measured, and one grep trap

R8 deletes a trait method, a public enum, and an error variant, so its doc footprint is
larger than its code footprint. Measured at `98aba17`, all in
`31b_semstrait_common_io/repository.md` (269 lines) unless stated. **Moderator-owned:
workers must not touch `docs/`.**

| site | what dies | disposition |
|---|---|---|
| `:48-50` | `save_guard()` in the `Persistable` listing + its no-default comment | delete both |
| `:70` | prose making `extension` **and** `save_guard` associated fns "so a repository can compute a key from an `EntityId` alone" | rewrite — the reason survives for `extension` alone |
| `:72` | `integrity_check`'s default-`Ok` contrast, *"unlike `extension` and `save_guard`"* | rewrite to `extension` alone |
| `:87` | `save`'s doc-comment citing `T::save_guard()` and `Q-IO-005` | rewrite — `Q-IO-005` (no CAS in v1) stays true and is now the *whole* reason |
| `:103-118` | the **"Clobber policy is the DTO's"** bullet: prose, the `SaveGuard` code block, and the content-derived-vs-authored rationale | **delete the bullet entirely** — §3's remaining bullets stand |
| `:194` | `IdCollision` row of the `§5` variant table | delete row |
| `:202-204` | `IdCollision` + its two-line R2 comment in the `§5` code block | delete |
| `:189`, `:251` | *"the **four**-variant `RepositoryErrorKind`"* — twice | **three** after R8 |
| `:260-261` | Testing Strategy: `§4` key-derivation bullet is fine; `§5`'s *"a collision raised through a real `save` carries the repository's actual `base`"* loses its subject | rewrite the `§5` bullet around `EncodingMismatch` only |
| `:264` | Testing Strategy **"Clobber policy (§3)"** bullet | replace with the `put` property: a differing value under an equal id **overwrites**, on both DTOs |
| `32-model-root-shape.md:124` | *"model persistence is plain last-writer-wins and does **not** layer the `IdCollision` guard that `31b §3` permits"* | rewrite — after R8 there is no guard to opt out of, so the sentence states a distinction that no longer exists |
| `32-model-root-shape.md:210` | *"re-saving edited content under the same id succeeds (no `IdCollision` guard)"* | keep the assertion, drop the parenthetical |

**The grep trap — `:159` says "collision" and must NOT be touched.** *"This is a
`collision` guard, not a sandbox"* is about **`§4.1` key injectivity** (`key_for`
refusing a separator-bearing id as `IntegrityCheckFailed`), an entirely different sense
of the word from clobber policy. It survives R8 untouched. A `grep -i collision`
disposition pass would delete a live invariant; this is the same failure class §4a.2
records for `kind_prefix`, so the sites above are listed by line and by *what dies*,
never by identifier alone.

Out of scope, already tracked: `io-error-kind.md:103`'s block still shows
`ManifestIdCollision { id: String }` and `EncodingMismatch { expected, actual }` — a
pre-R3 shape, and `ManifestIdCollision` no longer exists anywhere in `crates/`. That
leaf is already `impl: drift`; R8 does not widen its debt and must not silently absorb
it.

#### 4a.3a Re-measured at `46d348a` — four defects in the table above

The table was measured at `98aba17`; the R1–R4 waves and their doc convergence landed
after it, so it is re-measured here before execution. **The table's line numbers are
stale and the amended anchors below supersede them.** `repository.md` is now 295 lines,
not 269. Four defects, each measured:

1. **Every `§5` line number is off by ~26.** The `IdCollision` table row is `:220`, not
   `:194`; the enum variant is `:230`, not `:202-204`; the Testing Strategy bullets are
   `:287` and `:290`, not `:260-261` and `:264`. The grep trap's own anchor moved too —
   *"a `collision` guard, not a sandbox"* is at `:288`, not `:159`, and it is now a
   Testing Strategy bullet rather than the `§4.1` prose the table describes. §4a.2's
   R1–R4 convergence and §4a.4's R4 pass both edited this file between the two
   measurements. The **dispositions all still hold**; only the addresses moved.

2. **`:189`/`:251` "four-variant" is a FALSE disposition — do not change it to
   "three".** Both surviving sites (now `:215` and `:277`) say the common kind is *"the
   exact generalization of the **four-variant** `RepositoryErrorKind` the manifest crate
   already ships"*. That counts the **manifest's historical** kind, not common's. R8
   removes a variant from *common*; it cannot retroactively change how many variants the
   manifest crate shipped before `TD-013`. Worse, the count is already historical in a
   second way: `grep -n 'enum RepositoryErrorKind' crates/` returns **exactly one** hit
   (`common/src/io/repository.rs:68`) — the manifest's own kind is already gone, and
   `manifest/src/repository.rs:15` now `pub use`s common's. Both sentences are
   past-tense narration of a completed lift. What *would* be wrong after R8 is a claim
   that *common* has four variants.

   **Amended during execution — "leave both untouched" was too broad, and one site does
   make the wrong claim.** The *count* holds: the pre-lift manifest kind is measured at
   `67d9b7a^:crates/semstrait-manifest/src/repository.rs:36-57` as exactly four variants
   (`Io`, `ManifestIdCollision`, `EncodingMismatch`, `IntegrityCheckFailed`). Three
   surrounding claims do not:

   - `:196` said common is *"the **exact generalization** of the four-variant"* kind. After
     R8 it generalizes **three of the four** — `ManifestIdCollision` has no target. "Exact
     generalization" of a four-variant source by a three-variant type is false arithmetic,
     not stale tense.
   - `:254` said *"the same four-variant taxonomy **this leaf reconciles to**"* — which
     *is* the forbidden claim that common has four variants, made obliquely by "the same".
     Now reads *"a four-variant taxonomy this leaf reconciles into three"*.
   - `:254` also carried the mapping `(`ManifestIdCollision` → `IdCollision`)`, naming a
     deleted target. It is the third instance of TD-013's dead clause (with
     `TECH_DEBT.md:217` and `COVERAGE_MATRIX.md:178`), and the R8 brief listed only two.

   The lesson is the one `§4a.3` was written to prevent, arriving from the opposite side: a
   grep-trap exclusion can be **over**-broad as well as under-broad. `:254` appeared in the
   executor's own dead-vocabulary grep and was skipped because a blanket "leave `:215`/`:277`
   alone" instruction covered the line. State exclusions per *claim*, not per line.

3. **Two sites the table omits entirely, both outside `repository.md`.**
   `apis/_manifest.json` carries `31b-io-repository`'s search metadata, which names the
   dying vocabulary twice: `:1038` `"use_when"` lists `save_guard` in the `Persistable`
   surface, `SaveGuard clobber policy (LastWriterWins/RefuseDifferentValue)`, and
   `IdCollision` in the `RepositoryErrorKind` roster; `:1052` is a `key_concepts` entry
   reading *"the `SaveGuard` clobber policy"*. The gate checks frontmatter-vs-manifest
   **version** parity, not vocabulary, so it will stay green while the index advertises
   a deleted enum — the same class of silent desync §4a.4 hit from the other side.
   Both must be updated, and the leaf version bumped in **both** places.

4. **`TECH_DEBT.md:217` (TD-013) names the mapping R8 deletes.** Its Remediation says
   *"map the manifest's `ManifestIdCollision` onto the common `IdCollision`"* and closes
   *"the common `RepositoryErrorKind` is the exact generalization of the four-variant
   kind it already ships"*. TD-013 is `Status: tracked … blocked on ratification`, but
   defect 2's measurement shows the lift **already happened** — so the entry is stale
   independently of R8. Narrowest correct edit: strike the `ManifestIdCollision` →
   `IdCollision` clause (that target variant will not exist), leave the rest, and do
   **not** re-adjudicate TD-013's status here — that is its own item, and R8 must not
   quietly close a tracked debt as a side effect.

One further drift, non-blocking and **not** R8's to fix: `:230`'s block types
`IntegrityCheckFailed { reason: String }` while the code has
`reason: Cow<'static, str>` (`repository.rs:86`). Pre-existing, unrelated to R8, and
correcting it inside this pass would make the diff claim a convergence it did not
measure. Recorded here so the next `31b` pass has it.

**Net:** 10 of the table's 12 rows stand with corrected anchors; 1 row (`four-variant`)
is struck as a false disposition **and then partly re-instated** — the count survives, the
"exact generalization" and "the same … taxonomy" claims around it do not (see the amendment
in defect 2); 3 sites are added (`_manifest.json` ×2, `TECH_DEBT.md` ×1), plus `:254`'s
`ManifestIdCollision` → `IdCollision` mapping found during execution. The grep trap stands,
at `:288`.

#### 4a.3b Two more sites the `docs/`-scoped grep could not see

Both were found only by widening the sweep past `docs/`, and by grepping for the
*consequence* rather than the deleted identifier. They are recorded separately because
each defeats a different assumption above.

5. **`COVERAGE_MATRIX.md` — 24 sites, at the repo root, outside `docs/`.** It is
   **tracked**, it is the ledger the moderator verifies against ("Moderator verifies
   against **this**, not against agent prose"), and it is **not** in `doc-space-check`'s
   scope, so no gate will ever flag it. A `docs/`-only sweep — the natural reading of
   "spec convergence" — misses the file that defines the verification standard.

   Its 24 sites are **not** uniformly live. Classified: **2 open checkboxes** (`:178`
   `ManifestIdCollision` → `IdCollision`; `:215` *"two successive saves … proves NO
   `IdCollision` guard"*), **8 table rows**, **14 prose**. The prose is deliberate
   historical narration — Q-IO-005's closure argument, the T5a spec-inversion record, the
   two spec↔code drifts, the CF trace — and the file's **own established idiom** for an
   R-wave change is to *append* a superseding parenthetical, never to rewrite the record:
   `:1186` already reads *"MATCH (now **5** on both sides — `kind_prefix` struck by `47
   §4a` R2)"*, and `:253` stacks R2-then-R8 supersessions onto the original instruction
   without deleting it. **Follow that idiom.** Rewriting the history would destroy the
   audit trail this ledger exists to keep, and `:253` already anticipates R8 in exactly
   the right form.

   So the disposition is narrow: annotate the **2 open checkboxes** (both describe
   obligations R8 dissolves — the first's target variant will not exist, the second's
   parenthetical rationale no longer names anything real) and the **shape-diff rows**
   at `:1186-1189`, which claim a *current* spec↔code MATCH and would otherwise assert
   a `SaveGuard` row and a `PartialEq` bound that are both gone. Leave all 14 prose
   sites and the historical mutation tables (`:673`, `:1156`) untouched: a mutation that
   *was* KILLED in T5a was killed, and deleting that row would falsify the ledger.

6. **The `PartialEq` bound removal has a doc footprint the `SaveGuard` grep cannot
   find.** R8's code side drops `PartialEq` from
   `impl<T: Persistable + PartialEq + Send + Sync> Repository<T> for
   FileSystemRepository<T>` (and from the `key_of` test helper), because `&stored !=
   value` in the deleted guard was its only consumer. `grep 'Persistable + PartialEq'`
   over `docs/` + `COVERAGE_MATRIX.md` returns **exactly one** site —
   `COVERAGE_MATRIX.md:1189`, an impl-bounds row asserting `MATCH` against the spec —
   and `31b/repository.md` never declares the bound at all, so the *spec* needs no edit
   for it. That asymmetry is the finding: the ledger is stricter than the leaf here, and
   the bound would have gone stale in the one place nothing checks. Its `line 244`
   citation is also stale independently of R8.

### 4a.4 Spec convergence owed by R4 — sites, plus two defects it exposes

§4a.2's last paragraph names R4's *files*; these are the measured *sites*, at `98aba17`.
**Moderator-owned.**

`31b_semstrait_common_io/repository.md` — `§4` documents **only** `new` (`:142`), so
`open` is undocumented new public API and must be added to the `impl` block beside it,
carrying the per-backend base mapping and the "no filesystem access at construction"
property. `§4`'s prose already covers what `base` *means*, so `open` needs its
dispatch table and the `s3a`→`s3` normalization, not a restatement of rooting.

`37_semstrait_catalog/filesystem.md` — `§6.2`'s roster row `:124` reads
`| Accepted schemes | `s3://` |` and must become `s3://`, `s3a://`. `§6.6`'s `:163`
already anticipates `open(base)` correctly (single-backend-from-single-URI permitted,
crate-root router still forbidden) and needs **no** change — it was written for this.

`fs.rs`'s prose is **three** sites, not the two an earlier draft of this clause implied:
`:20-23` (the `Path` type doc's form bullets **and** its "recognized scheme vocabulary
is `file` and `s3`" sentence), **`:39`** (inside `Path::parse`'s own doc-comment: "any
scheme outside the recognized set (`file`, `s3`)"), and `:76` (`is_known_scheme`'s doc).
`:120` also names `s3://` but as an example of a *foreign-scheme mismatch* on
`UnsupportedScheme` — correct as written, leave it. The W3 worker was dispatched with
two and sent the third mid-flight.

**Two pre-existing defects R4 exposes but does not own.** Both are in
`31b/repository.md` and both predate this wave:

1. **`:139` offers `gs://…` and `abfss://…` as example `base` values** in `new`'s
   doc-comment. `Path::parse` refuses both, deliberately and by test
   (`path_parse_rejects_descoped_schemes`), and after R4 `open` refuses them too. The
   examples are aspirational in exactly the way `TD-024` records for `37`'s roster, so
   this is the same debt surfacing in a second leaf — fold it into `TD-024` rather than
   opening a new item.
2. **`:234` and `:238` call `Path::from("mem:")`, which does not exist.** Measured:
   `Path` derives `Debug, Clone, PartialEq, Eq, Hash` and implements `Display`; there is
   **no** `From` impl reaching it anywhere in `crates/` (the only `::from` hits are
   `object_store`'s `ObjectPath`). The shipped constructors are `Path::new` (unvalidated)
   and `Path::parse` (validated). `§7`'s example therefore would not compile; it should
   read `Path::new("mem:")`, since a nominal in-memory root is precisely the trusted
   input `new` exists for. Worth fixing while the file is open for R8's `§3` deletion —
   `§7` is untouched by either clause, so this is opportunistic, not owed.

### 4a.5 R4 remediation — the drive-letter rule shipped half-implemented (measured at `2606125`)

W3 landed `open` with **one** of R4's two ordered dispatch rules. The clause requires
*"two rules, in this order: a single-ASCII-letter scheme is a drive letter and dispatches
**local**, and a scheme is only a scheme when followed by `://`"* (`§4a`, R4). The shipped
code implements only the second, and its comment claims the second alone is sufficient:

> `// Only "://" counts as a scheme separator, so a Windows drive letter`
> `// ("C:\...", "C:/...") never matches here and falls to the local leg.`

True for those two forms, **false for a third**. `C://Users/x` is a legitimate local
Windows path — Windows tolerates a doubled separator — and it *does* contain `://`, so
rule (b) reads its scheme as `C`. Measured, by temporary probe against the shipped code:

| input | `open` result | correct |
|---|---|---|
| `C:\Users\x` | local (base `""`) | yes |
| `C:/Users/x` | local (base `""`) | yes |
| `C://Users/x` | `Io(Unsupported{describe:"C"})` | **no** |

The eight committed tests all pass because **none exercises the doubled-separator form**;
`open_windows_drive_letter_dispatches_local_both_forms` pins exactly the two forms the
comment names. This is the `§4a.2` failure mode again — a comment asserting a property
the code does not have, with tests chosen to match the comment rather than the ruling.

**The gap is two sites, not one.** `Path::parse` (`fs.rs:52`) uses the same
`split_once("://")` shape and is *upstream* of dispatch, so it refuses the same path
before `open` ever sees it. Measured on `Path::parse` directly: `C:\Users\x` and
`C:/Users/x` → `Ok`; `C://Users/x`, `c://data`, `D://x/y` → `UnsupportedScheme{scheme:
"C"|"c"|"D"}`. Any fix that touches only `repository.rs` leaves a validated-construction
door still shut.

**Ruled:** implement rule (a) as the clause orders it — before the `://` test, a scheme of
exactly one ASCII letter is a drive letter and takes the local leg — at **both** sites,
with the single-letter test shared rather than written twice. Pin `C://Users/x` and a
lower-case `c://data` by test at both sites, and correct the `open` comment to state the
rule it actually implements. Case matters: Windows drive letters are case-insensitive, and
the probe shows lower-case `c` reaching the same error.

**Why one letter is the right cut, not a `known_scheme` fallback.** Every scheme in the v1
vocabulary (`file`, `s3`, `s3a`) is ≥2 characters, and `object_store::ObjectStoreScheme`
has no single-letter scheme either, so "one ASCII letter ⇒ not a scheme" costs no
reachable vocabulary. Dispatching unknown *multi*-letter schemes to local instead would
silently swallow `gs://`, which `§4a`'s R4 explicitly requires to return
`UnsupportedScheme` — so the fallback must stay an error and only the single-letter case
may be reinterpreted.

### 4a.6 R4 remediation 2 — `s3a` normalization is missing from `new`, and 4 coverage gaps

A second adversarial review of R4 returned 7 findings. Six survive re-measurement at
`afa74f0`; one is rejected as measured against a superseded tree. Each disposition below
was re-derived by probe, not inherited from the report.

**F1 — `s3a` is blessed by `Path::parse` and refused by `S3FileSystem`, so `new` is a
broken door.** Measured by temporary integration probe under `--features io-aws`:

| door | result |
|---|---|
| `Path::parse("s3a://bucket/key")` | `Ok("s3a://bucket/key")` — blessed verbatim |
| `S3FileSystem::read` / `exists` on that `Path` | `UnsupportedScheme { scheme: "s3a" }` |
| `new(S3FileSystem, Path::new("s3a://bucket/prefix")).save` | `Io(Unsupported { describe: "s3a" })` |
| `open(Path::new("s3a://bucket/prefix")).save` | reaches real S3 (region error) — works |

So the alias is reachable and broken through exactly one door: `new`. R4 already ruled the
shape of the fix and ruled *against* the obvious one — *"teaching `strip_bucket` and
`reject_foreign_scheme` a second spelling would spread one alias across two more sites"*
(above). That rationale is unchanged by this finding; what the clause missed is that
**`new` is also a door**, and it kept `base` verbatim.

**Ruled (user, this session):** normalize in `new`, not in the backend. `new` rewrites an
`s3a://` base to `s3://` exactly as `open` already does, so the alias dies at both
constructors and `strip_bucket` / `reject_foreign_scheme` stay single-spelling as ratified.
This **extends** R4's narrow-fix posture to the second constructor rather than reversing
it — no clause is overturned. The alternative (an `s3a` arm in each backend helper) was
put and declined. Pin by test at the `new` door, not only through `open`: the existing
`open_s3_and_s3a_compose_identical_keys` cannot see this, because `open` normalizes before
`new` is reached.

Two narrowings of the report's own account, both measured. (a) It located the gap as *"one
arm each in `reject_foreign_scheme` and `strip_bucket`"*; `reject_foreign_scheme` fires
first in all four `FileSystem` methods, so the `strip_bucket` miss is **unreachable** today
and only becomes reachable if the rejection is relaxed. (b) The severity is not
seam-breaking: the sole failing path is `new` with a hand-built `S3FileSystem`, which no
shipped caller takes — `open` is the convenience door and it is correct.

**F4 — the absent-directory test is inert.** `open_does_not_touch_the_filesystem_on_an_
absent_directory` asserts only `repo.is_ok()`, which holds whether or not `open` touched
disk; the deferred-failure half is already pinned by the test immediately after it. **Ruled
(user):** replace the assertion with the property the name claims — the directory still
does not exist after `open` returns — so an eager `create_dir_all` or canonicalize would
kill it.

**F2/F3/F5/F7 — confirmed, unchanged from the report:** the `#[cfg(not(feature =
"io-aws"))]` arms in `open` duplicate what the `Some(other)` fallback already produces
(measured: identical `Unsupported { describe }` payload), so ~10 lines are a tautology;
`extract_bucket` and `strip_bucket` have no direct test at the `open` boundary; the
`file://`-stripping leg of `open` is unpinned (`:228`, distinct from the `file:///` **base**
tests at `:809`-`:851`, which go through `new`); and a bucketless `s3://` base deserves one
sentence of documented behavior.

**F6 — REJECTED as stale.** It claimed the ordered drive-letter rule was unimplemented.
It was measured against `/tmp/w-r4` at `2606125`; `§4a.5`'s remediation landed at
`fs.rs:90` (`is_drive_letter`) and is called from both sites, with `C://Users/x` and
`c://data` pinned at each. Found by `git branch --contains`. A review's base commit is part
of its evidence.

**EXECUTED — `2346422`, merged `0f3405d`; spec converged in `5355a9c`.** All six acted-on
findings landed in `io/repository.rs` alone; `backends/s3_fs.rs` is untouched, which is the
ruling's own constraint. `canonical_base` is a free function called by `new`, deliberately
**not** `#[cfg(feature = "io-aws")]` — `new` compiles unconditionally, so gating it would
break the non-aws build. `open`'s `s3a` arm now derives only the bucket and hands `base`
through, since `new` normalizes.

Five mutations, all killing, each applied and reversed by the moderator on the merged tree:
`new` keeping `base` (kills the new `new`-door pin **and** `open_s3_and_s3a_compose_identical_keys`,
because Edit 3 moved the rewrite so `open` now normalizes *through* `new` — one normalizer,
two doors); eager `create_dir_all` in `open`; `file://` strip removed; `extract_bucket`
taking the last segment; and the non-aws fallback misreporting `s3a` as `s3`. That last one
is what makes F2's deletion safe rather than merely plausible — the two deleted arms are
only redundant if the fallback names **both** spellings, so the non-aws test was widened
from `s3` alone to both before the arms came out.

Two records worth keeping. (a) **One equivalent mutant:** stripping `file:` instead of
`file://` leaves `//var/…`, which POSIX resolves to the same file, so the suite stays green
and no assertion can distinguish it. Not a gap in `open_strips_the_file_scheme_off_the_backend_root`.
(b) **A bad mutation of mine, corrected:** my first `extract_bucket` mutant was
`rsplit('/').next_back()`, which is *identical* to the original (Rust's `rsplit` iterates
from the end, so `next_back()` returns the **first** segment). It survived, and for a moment
that read as a real F3 coverage gap. Reproducing both functions over all five rows settled
it; the genuine last-segment mutant then killed the test on `"s3://bucket/"`. **A surviving
mutant is evidence only once the mutant is shown to be non-equivalent** — the same
verify-the-mechanism discipline this plan applies to agents applies to my own probes.

Counts: 1124 → **1127 passed / 0 failed / 1 ignored** (+3; F4 replaces and F2 widens in
place). Non-aws suite 0, clippy `-D warnings` 0, fmt 0, doc gate 0 with census unchanged.

## 5) Cross-references

- **Leaves:** `31b-io-repository` (persistence seam — T5 target), `32-model-io-submodule`
  (§5 retired by T1), `37-filesystem` (T3 roster), `33-manifest-contract` /
  `compile-driver` (generify source; UUIDv5-vs-UUIDv7 identity contrast).
  **Amended by M1 + M2** (all four now `impl: drift`, cleared by T7):
  `32-model-root-shape` (new §1.4a, root struct field, §2.3 rewritten, SR-11/SR-12
  scope), `32-model-parse-validate` (root derivation in step 3 + both profiles +
  cross-run-stability caveat scoped to entity ids), `32-model-crate-boundaries`
  (SR-11 restated as shape, SR-12 scoped, I4 variance note corrected),
  `31-semstrait-common` (§2.1 states the `EntityId` shape contract normatively —
  the rule SR-11 now defers to; no code change, the alias is unchanged).
  `33-manifest-contract` records the `model_id` ↔ root-`id` contrast (content-keyed
  vs document-keyed); behavior unchanged, so it stays `impl: implemented`.
- **Questions:** `Q-IO-H` (YAML-only), `Q-IO-K` (per-DTO codec — model YAML vs
  manifest JSON), `Q-IO-005` (LWW, no CAS).
- **Tech debt:** `[TD-BUILD-MODEL-IO]` (T1–T4, T6), `[TD-GENERIFY-REPOSITORY]` (T5).
- **STATUS.md:** item G (I/O transport — Ratified); Deferred (c) (async model::io
  wrappers — superseded by this plan's model-root loader + common-fetch shape).
- **Authoring/maintenance rules:** `.agents/skills/semstrait-doc/` (governance was
  centralized there; `docs/design/CONVENTIONS.md` no longer exists).
