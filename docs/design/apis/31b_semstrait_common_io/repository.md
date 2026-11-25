---
id: 31b-io-repository
status: proposed
version: 2.4
impl: implemented
---

# Repository Persistence Seam

This leaf owns the `Persistable` trait a DTO implements to be stored, the generic `Repository<T>` persistence trait, the `FileSystemRepository<T>` blanket implementation, the `RepositoryErrorKind` taxonomy, and the backend-extension open seam that lets a new persistence backend be added without touching callers.

## 1. Purpose and posture

`semstrait-common` owns the **persistence seam**: the abstraction that tracks, loads, and saves a workspace DTO (`SemanticManifest`, `SemanticModel`, and — post-v1 — `SemanticGraph`) by its `EntityId`. The seam is three layers:

1. **`Persistable`** — the trait a DTO implements to be stored: it declares the DTO's durable identity and its byte codec. It lives **on the DTO**, not on the repository or the transport. Defined here; implemented downstream (`impl Persistable for SemanticManifest`, `… for SemanticModel`).
2. **`Repository<T>`** — the load/save/exists contract keyed by `EntityId`, generic over any `Persistable`. Its operations are expressed **in terms of `Persistable`**: `save` **accepts** a `&T: Persistable` to write, `load` **returns** the concrete `T: Persistable` it read. The repository is the persister; the `Persistable` DTO is the thing persisted.
3. **`FileSystemRepository<T>`** — the one v1 implementation, a blanket impl over any `FileSystem` byte transport (`37-filesystem`) plus the DTO's own codec. Constructed with a base **path** — an object-store prefix (S3/GCS/Azure) or a local directory — that roots every key it writes (§4); in-memory persistence needs none (§7).

Design posture:

- **Persistence, not authoring.** `Repository<T>::load` returns a `T` from a canonical, machine-written blob keyed by id. It carries **no `Diagnostics` channel** — a persisted artifact round-trips without warnings. Reading and parsing *human-authored* YAML (which does emit warnings) is the separate authoring door in `32-model-loader` §1, layered over the `FileSystem` byte read, not over this seam.
- **Codec is the DTO's, transport is the backend's.** `Persistable` owns byte encoding; the `Repository` impl owns byte movement. Neither knows the other's internals.
- **A stable facade over the byte-transport engine.** `Persistable` / `Repository<T>` / `FileSystem` are an anti-corruption layer: callers depend on *our* surface, never on the underlying byte-transport library. v1 backends thin-wrap `object_store` (Apache Arrow — the chosen engine, `37 §6`), which keeps `object_store` types off every public signature. Changing that engine, or adding a backend `object_store` does not cover (e.g. DynamoDB via `aws-sdk-dynamodb`, §6), changes no caller.
- **Fail-fast.** Every method returns `Result<_, RepositoryErrorKind>`; a read/write/decode failure surfaces the error straight to the caller. No retries, no silent recovery, no repository-injected deep-validation pass (§5).
- **Open seam.** A new backend is a new `impl<T: Persistable> Repository<T> for MyBackend` — no change to any caller that holds `Arc<dyn Repository<T>>` (§6).
- **Cycle-free (I7).** Every type this seam names — `EntityId`, `Bytes`, `FileSystem`, `IoErrorKind` — already lives in `semstrait-common`. The seam therefore sits in `common` with no DAG inversion; `common` never names `SemanticModel` / `SemanticManifest` (§8).

## 2. `Persistable`

The trait a DTO implements **to be stored** — it is the "persistable thing" a `Repository` reads and writes, not a persister itself. A DTO becomes storable by declaring its durable identity and its byte codec. Pure and synchronous — no I/O, no transport knowledge: those belong to the `Repository` (§3), whose `save` accepts a `Persistable` and whose `load` returns one.

```rust
use std::borrow::Cow;
use bytes::Bytes;
use crate::EntityId;

pub trait Persistable: Sized {
    /// The durable identity this value is keyed by in a repository.
    fn entity_id(&self) -> &EntityId;

    /// The key's file extension, without the dot — part of the stored
    /// object's *name*. No default: only the DTO knows what its own object
    /// is called, so adding a `Persistable` cannot silently inherit a
    /// wrong one.
    fn extension() -> &'static str;

    /// Refuse a value whose own invariants are broken, before the repository
    /// calls the infallible `entity_id` on it. Defaults to `Ok`: override
    /// where a well-formed value is not guaranteed by construction.
    fn integrity_check(&self) -> Result<(), Cow<'static, str>> {
        Ok(())
    }

    /// Encode to the canonical wire bytes. Pure — MUST NOT perform I/O.
    /// A failure is a bare reason the repository folds into
    /// `RepositoryErrorKind::EncodingMismatch` (§5) — no dedicated codec type.
    fn to_bytes(&self) -> Result<Bytes, Cow<'static, str>>;

    /// Decode from wire bytes. Pure — MUST NOT perform I/O. Reason-on-failure
    /// as above.
    fn from_bytes(bytes: &[u8]) -> Result<Self, Cow<'static, str>>;
}
```

`extension` is an associated (type-level) function so a repository can compute a key from an `EntityId` alone (for `load` / `exists`, where no value exists yet). `entity_id` is an instance method used by `save`. The codec stays **opaque and per-DTO** (`Q-IO-K`, `47 §4a` **R3**): the manifest encodes JSON, the model YAML, with no shared codec enum in `common` — and the repository never names an encoding, not even to report a mismatch. `extension` carries no default because the suffix is part of the object's *name*, and only the DTO knows its own name; it is a key-composition concern, not a codec one. (`extension() == "json"` on the manifest and `"yaml"` on the model do coincide with their codecs, which is why the two must not be conflated: the suffix is a naming choice that happens to agree.)

`integrity_check` exists because `entity_id` is **infallible by design** — it returns `&EntityId`, so a DTO whose id is not yet resolved has nothing to return and must panic. Rather than make every DTO's identity accessor fallible for the sake of the one whose id is optional until sealed (`SemanticModel`, `32 §1.4a`), the fallibility moves to a separate method the repository calls *first*, on both the `save` and `load` legs. It defaults to `Ok` — unlike `extension`, a missing override is safe here, because a DTO whose invariants hold by construction has nothing to check. `SemanticModel` overrides it; `SemanticManifest`, whose id is a plain field, does not.

**Authoring-shaped bytes.** A DTO that is also human-authored encodes the *same document shape* through both doors. For `SemanticModel` that means `to_bytes` emits the `semantic_model:` root wrapper (`32 §1`), so a saved model is a loadable model file — the property `32 §1.4a` relies on when it states the model file is the store, with no side-car state and no write-back. A DTO whose in-memory `Serialize` is unwrapped therefore wraps in its `Persistable` impl — and `from_bytes` unwraps symmetrically — rather than shipping two mutually unreadable file shapes.

## 3. `Repository<T>`

The persistence contract, generic over the DTO. Object-safe so callers hold `Arc<dyn Repository<T>>`.

```rust
use async_trait::async_trait;
use crate::EntityId;

#[async_trait]
pub trait Repository<T: Persistable>: Send + Sync {
    /// Persist a `Persistable` value under `value.entity_id()`, overwriting
    /// whatever is there (no compare-and-swap in v1 — `31b Q-IO-005`, closed).
    async fn save(&self, value: &T) -> Result<(), RepositoryErrorKind>;

    /// Load the `Persistable` value keyed by `id`, returned as the concrete
    /// `T`. `RepositoryErrorKind::Io(NotFound)` when absent.
    async fn load(&self, id: &EntityId) -> Result<T, RepositoryErrorKind>;

    /// Whether a value is stored under `id`. Does not decode.
    async fn exists(&self, id: &EntityId) -> Result<bool, RepositoryErrorKind>;
}
```

Contract:

- **Identity keying.** `save(v)` stores at the key derived from the repository's base path + `v.entity_id()`; `load(id)` / `exists(id)` derive the same key from `id`. `save` then `load` of the same id yields a value byte-equal under `to_bytes` (round-trip, absent concurrent mutation). Ids are case-folded on the way into a key (`47 §4a` **R1**), so ids differing only in case name one object.
- **Update is whole-object.** There is no separate `update` method: `save` **is** the update — it replaces the entire stored object under the key. A caller mutating a persisted DTO does read-modify-rewrite (`load` → mutate in memory → `save`), never an in-place patch. The engine (`object_store`, `37 §6`) exposes only whole-object atomic replace; see the non-goal below.
- **Clobber policy is the seam's, and it is `put`.** `save` overwrites unconditionally: the same id, written twice, keeps the later value — one write, no read-before-write (`47 §4a` **R8**). v1 has no conditional (compare-and-swap) writes (`Q-IO-005`, closed), so a seam-level refusal could not be atomic anyway. A DTO does not choose: an id whose derivation is broken is an identity defect, and this seam is the wrong altitude to detect one.
- **No diagnostics.** The return is a bare `Result<_, RepositoryErrorKind>` — persistence is not a warning-producing stage.
- **No listing / query in v1.** Enumeration, prefix scan, and secondary-index query are out of scope; callers that need them compose over the backend's native surface.
- **No byte-level partial / fragmented update (non-goal).** Appending to, range-patching, or fragment-rewriting a stored object is **out of scope**. The forcing constraint is the engine: `object_store` provides only whole-object atomic replace — no append, no in-place patch, and conditional (compare-and-swap) put is `NotImplemented` on the local backend and off by default on S3. A caller that needs sub-object granularity models it as multiple `Persistable` DTOs (each its own key), not as a partial write.

## 4. `FileSystemRepository<T>`

The sole v1 implementation: a `Persistable` codec bridged onto any `FileSystem` byte transport, rooted at a caller-supplied base **path**.

```rust
use std::{marker::PhantomData, sync::Arc};
use crate::io::{FileSystem, Path};

pub struct FileSystemRepository<T> {
    fs: Arc<dyn FileSystem>,
    base: Path,
    _t: PhantomData<fn() -> T>,
}

impl<T: Persistable> FileSystemRepository<T> {
    /// `base` roots every key this repository reads or writes: an object-store
    /// prefix (`s3://bucket/tenants/acme`, `s3a://…`) or a local directory
    /// (`file:///var/lib/semstrait`, `./state`). An `s3a://` base is
    /// canonicalized to `s3://` here (§4.2). A repository over an
    /// `InMemoryFileSystem` ignores `base` (the in-memory store is flat — §7).
    pub fn new(fs: Arc<dyn FileSystem>, base: Path) -> Self {
        Self { fs, base: canonical_base(base), _t: PhantomData }
    }

    /// Derives the backend from `base`'s scheme, so a caller needs no
    /// `object_store` vocabulary (§4.2). Performs no I/O: an absent local
    /// directory surfaces on the first `save`/`load`, not here.
    pub fn open(base: Path) -> Result<Self, RepositoryErrorKind> { /* … */ }

    /// "{base}/{id}.{extension}" — deterministic, path-rooted, id-addressed,
    /// with the id case-folded (`R1`). Fallible: an id that would break the
    /// id→key mapping is rejected rather than filed (§4.1).
    fn key_for(&self, id: &EntityId) -> Result<Path, RepositoryErrorKind> { /* … */ }
}
```

**Base path — object-store prefix or local directory.** `base` is one caller-chosen root under which all of a repository's state is filed. On S3/GCS/Azure that root is an object-store **prefix** — those stores expose only key prefixes, not directories; on a local disk it is a **directory path**. Both are the same rooting operation in a URI, carried as the single `Path` newtype (`37 §5.2`). `key_for` composes `base` ⧺ `id` ⧺ `extension`, and that is the whole key: one repository instance stores exactly one type, so there is no second, kind-level namespace axis (`47 §4a` **R2**, resolving `Q-IO-N` and `Q-IO-M`). Two roots never collide on equal ids; two DTO kinds under **one** root is a caller wiring choice — give them two bases. Only their extensions would otherwise separate them.

### 4.1 The id→key mapping must stay injective

`EntityId` is `pub type EntityId = String` (`31 §2.1`) — an unvalidated alias, and a public field on every DTO — so nothing upstream stops a separator reaching this seam. Composing such an id into a key is not injective: `a/b`, `a//b` and `/a/b` all name one object, so a `save` under one id would silently overwrite a *different* entity stored under another. `key_for` therefore **rejects** an id that is empty or contains `/` or `\`, as `IntegrityCheckFailed` — which is why it returns `Result`, and why `save`, `load` and `exists` can each fail before touching the store.

This is a *collision* guard, not a sandbox, and **traversal is not guarded anywhere at this seam**. The refusal that exists is a property of one dependency, not of this crate: `object_store::path::Path::parse` rejects `..`, `.`, and `./`-forms outright (`Io(Malformed)`, "illegal character sequence", nothing written), and all four plumbing functions in `backends/object_store_fs.rs` route through it. So every `object_store`-backed backend refuses a traversal key — while `InMemoryFileSystem`, a `HashMap` keyed on the URI verbatim, **stores it**: `write("base/docs/../escaped.bin")` returns `Ok`, `list` reports that exact key, and it round-trips at the unresolved key only. The two shipped backends therefore disagree on whether such a save succeeds, and the in-memory double is the *less* strict one — a fidelity gap for any suite that treats it as a stand-in.

All three key legs are caller-controlled, and each reaches a parse-refused key on its own. Measured over both transports, `base = "base"` unless the row says otherwise:

| door | key composed | in-memory | `object_store`-backed |
| --- | --- | --- | --- |
| `extension()` is `""`, id is `"."` | `base/deep/..` (base `base/deep`) | `Ok`, stored | refused |
| `extension()` contains `/` (`"/x"`, `"bin/.."`) | `base/../x`, `base/abc.bin/..` | `Ok`, stored | refused |
| `base` bears a `..` or `.` segment | not inspected by this guard at all | `Ok`, stored | refused |

**Three, not four — R2 removed a door.** A fourth leg used to exist: `kind_prefix()` returning `".."`, `"."`, `""` or `"a/../b"` composed a traversal segment of its own. `47 §4a` **R2** deleted that method, so the leg is gone with it — a **reduction** in attack surface, and the only one of the four no caller could reach without writing a deliberately hostile `Persistable` impl. The count is load-bearing: a "four doors" completeness claim silently becoming three is exactly the defect this table is here to prevent, so it is restated rather than quietly edited.

The id leg is the non-obvious one: the separator guard tests for `/` and `\`, and `"."` has neither. A **slash-free** `extension()` defuses it — the interposed `.` makes id `"."` file the name `..bin`, which parses — but non-emptiness is not the property, as row 2 shows.

The accept side is per-segment-exact, not "`.` anywhere": `base/...` (id `".."`, empty extension), `base/abc..` (id `"abc."`, empty extension), and `base/.../y` (id `"."`, extension `"./y"`) are all stored by *both* transports, because a `...` segment and a trailing-dot filename are legal. All re-measured over both transports after the row-3 deletion, not carried over from it. So the seam's guarantee is narrow and delegated: **exactly whatever `object_store::path::Path::parse` refuses**, plus nothing at all for the in-memory transport. Containment is not a property of this seam (`Q-IO-O`).

Length is likewise not checked here — an id the OS cannot name surfaces as a backend error on write (`37 §6`), not as a rejection at this seam.

The impl folds the two error surfaces it bridges into `RepositoryErrorKind` (§5):

- `save` → `value.integrity_check()?` (§2 — ahead of `key_for`, because `entity_id` is infallible and would otherwise panic on a value that has no id to key by), then `T::to_bytes(value)?`, then `fs.write(self.key_for(value.entity_id()), bytes)`; a `FileSystemErrorKind` folds via `From` into `RepositoryErrorKind::Io`.
- `load` → `fs.read(self.key_for(id))` then `T::from_bytes(&bytes)?`; a codec-failure reason folds into `RepositoryErrorKind::EncodingMismatch`. The decoded value then passes the **same** `integrity_check` as on the way in — a DTO whose codec is laxer than its accessor would otherwise be dereferenced rather than refused, having been cleanly rejected on `save`. The impl also runs one cheap **id-vs-key** check — the decoded value's own `entity_id()` must equal the requested `id` — surfaced as `RepositoryErrorKind::IntegrityCheckFailed`; DTO-specific *deep* validation is not run here (§5).
- `exists` → `fs.exists(self.key_for(id))`.

Because `FileSystem` is scheme-dispatched (`37 §5.2`), one `FileSystemRepository<T>` serves local, S3, Azure, GCS, and in-memory transport (§7) without per-backend repository code — the backend variety lives in the injected `Arc<dyn FileSystem>` and the root in `base`. Each of those `FileSystem` backends thin-wraps `object_store` (`37 §6`), so the repository inherits its atomic writes and credential chains without ever naming `object_store` on this surface (`37 §6`, the facade rule).

### 4.2 `open(base)` — deriving the backend from the scheme

`new(fs, base)` takes **two** values that can disagree: the scheme written in `base` and the identity of the injected backend. `open(base)` takes one and derives the other, so the disagreement cannot be expressed (`47 §4a` **R4**, resolving `Q-IO-O`). `new` **stays** as the injection door — tests and callers holding a pre-built or custom `FileSystem` need it; `open` is the convenience path, not a replacement.

| `base` scheme | backend | what the repository keeps as `base` |
| --- | --- | --- |
| `s3://bucket/prefix` | `S3FileSystem::new(bucket)`, behind `io-aws` | the whole base, verbatim |
| `s3a://bucket/prefix` | `S3FileSystem::new(bucket)`, behind `io-aws` | the base **normalized to `s3://`** (in `new`, so `new` alone does it too) |
| `file:///dir` | `LocalFileSystem::new(dir)` | empty |
| no scheme (`./state`, `/var/lib/x`) | `LocalFileSystem::new(base)` | empty |
| single ASCII letter (`C:\…`, `C:/…`, `C://…`) | `LocalFileSystem::new(base)` | empty |
| anything else (`gs://`, `http://`) | — | `UnsupportedScheme`, never a panic |

**The two legs keep different values, and that asymmetry is the ruling, not an accident.** `LocalFileSystem::new` takes a filesystem **root** while `S3FileSystem::new` takes **only a bucket** — it has no prefix field, and its `strip_bucket` removes the literal `s3://{bucket}/` from a composed key. So for the local leg the whole base is consumed by the backend and the repository's prefix is empty, while for S3 the backend takes the bucket alone and the repository must retain the full base — an empty prefix there would discard `prefix` and silently write to the bucket root.

**`s3a` normalizes to canonical `s3` in the retained base, and it normalizes in `new`.** `strip_bucket` strips the literal `s3://{bucket}/`, so an `s3a://` base kept verbatim would dispatch correctly and then fail on every composed key. The rewrite therefore sits in `new`, not in `open` and not in the backend: `open` reaches `base` only through `new`, so one normalizer covers **both** constructors, while a caller who injects a hand-built `S3FileSystem` — the whole point of keeping `new` as the injection door — would otherwise get an `s3a` base the backend refuses (`47 §4a.6` **F1**). Teaching `strip_bucket` and `reject_foreign_scheme` a second spelling was the rejected alternative: the alias is *addressing*, so it dies at the door rather than propagating into every backend that later accepts it.

Pinned by **two** tests, because one door is invisible from the other: an `s3://` and an `s3a://` base composing the *identical* key through `open`, and the same equality through `new` over a hand-built backend. The `open` test alone cannot see a `new` that keeps `s3a`, since `open` normalized before `new` was reached even before the fix.

**A Windows drive letter is not a scheme.** `C://Users/x` is a local path Windows accepts, and it contains `://` — so "a scheme is only a scheme when followed by `://`" is not sufficient on its own. Two rules apply **in order**: a scheme of exactly one ASCII letter is a drive letter and dispatches local; only then does the `://` test decide. Both rules live at **two** sites — `open` here and `Path::parse` in `37 §5.2`, which is upstream of dispatch and would otherwise refuse the path before `open` saw it. Every ordinary scheme is ≥2 characters, so reinterpreting the single-letter case costs no reachable vocabulary; unknown *multi*-letter schemes must still error, which is why the fallback is not "unknown means local".

`open` performs **no I/O**. An absent local directory is not detected at construction — it surfaces on the first `save` or `load`. This keeps `open` cheap and its failure modes limited to scheme parsing, and it matches `S3FileSystem`'s per-call client build (`37 §6.2`).

## 5. `RepositoryErrorKind`

<!-- cross-cut: none -->

`RepositoryErrorKind` is the crate-neutral persistence error, keyed by variant identity (`30 §5`), `#[non_exhaustive]` (I10). It is defined **here in common** so every DTO's repository shares one taxonomy; the concrete manifest Repository (`33`) re-uses it. It generalizes **three of the four** variants of the `RepositoryErrorKind` the manifest crate shipped (`33`): codec failure folds into the existing `EncodingMismatch` (no dedicated codec error type), and there is no repository-injected `validate` hook (`Q-IO-M`). The fourth, `ManifestIdCollision`, has no target here — `save` is a `put` (`47 §4a` **R8**).

| Variant | Carries | Raised when |
|---|---|---|
| `Io(IoErrorKind)` | transport error (`31b-io-error-kind`) | backend read/write/exists fails; `FileSystemErrorKind` folds in via `From` |
| `EncodingMismatch { actual }` | the DTO's own failure reason | `Persistable::to_bytes` / `from_bytes` fails, or stored bytes do not decode under the DTO's canonical encoding |
| `IntegrityCheckFailed { reason }` | why | three sites, all cheap and inline: the id would not compose an injective key (§4.1); `Persistable::integrity_check` refuses the value on the way in or out (§2); or the decoded value's own id disagrees with the key it was filed under |

```rust
#[non_exhaustive]
pub enum RepositoryErrorKind {
    Io(IoErrorKind),
    /// `actual` is the DTO's own reason. The repository names no encoding (R3).
    EncodingMismatch { actual: std::borrow::Cow<'static, str> },
    IntegrityCheckFailed { reason: std::borrow::Cow<'static, str> },
}
```

`RepositoryErrorKind` implements `Diagnose` (`31 §3`) and provides `From<IoErrorKind>` and `From<FileSystemErrorKind>` (folding transport errors at the impl edge). A `Persistable` codec failure is a bare reason (`Cow<'static, str>`) the repository folds into `EncodingMismatch` — there is **no** separate codec error type. Load-time integrity is limited to the DTO's own `integrity_check` plus the inline id-vs-key check (both `IntegrityCheckFailed`): reading by id yields the object filed under that id or a `NotFound`, so a DTO's own deep-validation pass (e.g. manifest CX1, `33 §10`) is the consumer's post-load step, not a repository-injected hook. The variant table constrains no external type leaf — hence the cross-cut waiver above.

## 6. Backend-extension open seam

Adding a persistence backend is additive and touches no caller: implement `Repository<T>` for the new type. Blob backends (fs / S3 / Azure / GCS / memory) are already served by `FileSystemRepository<T>` over `object_store` (`37 §6`); a **document / KV** backend that does not unify at the byte-transport layer (no path, no prefix-list) becomes its own `Repository<T>` impl over *its* best library.

```rust
// Post-v1, illustrative — NOT a v1 deliverable.
pub struct DynamoDbRepository { /* aws-sdk-dynamodb client, table */ }

#[async_trait]
impl<T: Persistable> Repository<T> for DynamoDbRepository {
    // key = (partition = the repository's own base, sort = id); body = to_bytes(value).
    // Caveat: DynamoDB's 400 KB item cap requires overflow handling (chunk to S3
    // + pointer) for large manifests — a backend-internal concern, invisible to
    // callers holding `Arc<dyn Repository<T>>`.
}
```

The seam guarantee: a caller wired with `Arc<dyn Repository<SemanticManifest>>` is source-unchanged whether the concrete type is `FileSystemRepository` (over an `object_store` transport) or `DynamoDbRepository` (over `aws-sdk-dynamodb`) — the §1 facade guarantee, generalized to any `T` and any backend.

## 7. In-memory persistence

In-memory persistence is preserved for tests and one-box runs **at this layer**, not as a bespoke backend: an `InMemoryFileSystem` behind `FileSystemRepository` yields a fully in-memory repository for **every** `T` — the in-memory `FileSystem` (`37 §6`) supplies the transport, and the generic repository supplies save/load/exists. The `base` path is inert here: the in-memory store is a flat keyspace, so a caller passes a nominal root (e.g. `Path::new("mem:")`) and the value is ignored beyond key composition.

```rust
let repo: FileSystemRepository<SemanticManifest> =
    FileSystemRepository::new(Arc::new(InMemoryFileSystem::default()), Path::new("mem:"));
repo.save(&manifest).await?;
let round_tripped = repo.load(manifest.entity_id()).await?;
```

No separate `InMemoryRepository` type is needed — in-memory is one `FileSystem` choice behind the single generic repository.

## 8. DAG safety and layering

`Persistable`, `Repository<T>`, `FileSystemRepository<T>`, and `RepositoryErrorKind` all live in `semstrait-common`. Every type they reference (`EntityId` — `31 §2.1`; `Bytes`; `FileSystem` / `IoErrorKind` — `37 §5`, `31b-io-error-kind`) is already a `common` type, so the seam introduces **no** dependency inversion (I7): `common` never names `SemanticModel` or `SemanticManifest`. Downstream crates import `Persistable` to tag their DTOs — `impl Persistable for SemanticManifest` in `semstrait-manifest`, `impl Persistable for SemanticModel` in `semstrait-model` — and inject a concrete repository at the API boundary (`38 §4.2`, `with_repository`).

## 9. Relationship to the concrete manifest Repository (33) and model authoring (32)

- **`33-manifest-contract`** already ships a **non-generic** `Repository` (save/load/exists for `SemanticManifest`) and its own `RepositoryErrorKind` in code, persisting over an `Arc<dyn FileSystem>` (object_store-backed, `37 §6`) with a four-variant taxonomy this leaf reconciles into three. This leaf is the **generic target**: `33`'s concrete impl becomes `impl Persistable for SemanticManifest` + `FileSystemRepository<SemanticManifest>`, and its error type is lifted into the common `RepositoryErrorKind` here (`ManifestIdCollision` alone has no target — `save` is a `put`, `47 §4a` **R8**). Because that is a behavior-neutral lift-into-common of a ratified/implemented surface, it is tracked as `[TD-GENERIFY-REPOSITORY]` and executed by the implementation plan (`47` T5), not by this leaf. One consequence is **not** behavior-neutral and is called out there: `base` becomes required and the kind segment goes away (`47 §4a` **R2**), so the manifest's on-disk key changes shape twice over — `manifests/{id}.json` → `{base}/{id}.json`.
- **`32-model-loader`** owns the **authoring door** (`SemanticModel::loader()` = read or accept bytes, parse human YAML, emit warnings). That door keeps its `Diagnostics` channel and takes remote bytes through `from_yaml_bytes` (`32-model-loader §4`) rather than bridging to transport itself, so it layers over the `FileSystem` byte read (`37`), not over this seam. Canonical model persistence (a machine-written model blob, no warnings) is `Repository<SemanticModel>` from this leaf.

## Testing Strategy

- **Round-trip contract (§3):** `save(v)` then `load(v.entity_id())` yields a value byte-equal under `to_bytes` — proven against `FileSystemRepository` over `InMemoryFileSystem` (§7), so the contract test needs no disk or network.
- **Whole-object update (§3):** a second `save` of a mutated value under the same id replaces the stored object; a subsequent `load` returns the mutated value (no partial-write path exists — the update is the full rewrite).
- **`Persistable` purity (§2):** `to_bytes` / `from_bytes` perform no I/O — a `Persistable` impl compiled against a no-I/O feature set still builds.
- **Authoring-shape parity (§2):** for a human-authored DTO, bytes from `to_bytes` load cleanly through that DTO's authoring door — `SemanticModel`'s save output parses via `from_yaml_bytes` (`32-model-loader §1`), proving one file shape serves both doors.
- **Key derivation (§4):** `key_for(id)` is `"{base}/{id}.{extension}"` and deterministic, carrying each DTO's own extension (`<base>/<id>.json` for the manifest, `<base>/<id>.yaml` for the model) — never one hardcoded by the repository; **each** shipped `extension()` is asserted, per DTO and by value, since a wrong suffix files the object under a name nothing reads back, and asserted bare (no leading dot, no separator), since `".yaml"` would compose `<id>..yaml`; ids differing only in case resolve to one key and one loadable value (**R1**, asserted on a hex-bearing id — an all-digit id makes a case probe vacuous); `save` under `entity_id()` and `load(id)` resolve to the same key; two repositories over the same `fs` with **different** `base` roots do not collide on equal ids.
- **Error payloads are carried, not known (§5):** an `EncodingMismatch` raised through a real `save` or `load` carries the DTO's own reason — asserted against a value a hardcoded literal could not satisfy. Hand-constructing a variant tests `message()`, not the construction site, so it does not discharge this.
- **Traversal is the backend's call, not this seam's (§4.1):** the id `"."` under an empty `extension()` composes a literal `..` segment past the separator guard — asserted as the composed key, so the guard is not mistaken for a sandbox. The same key is *accepted and stored* by `InMemoryFileSystem` and *refused* as `Io(_)` by a real `object_store`-backed backend; both halves are pinned, so the asymmetry cannot silently close or widen.
- **`Persistable::integrity_check` (§2):** a DTO whose id is unresolved is refused on `save` *and* on `load`, both as `IntegrityCheckFailed`, rather than reaching the infallible `entity_id`. The default `Ok` body is asserted live, not assumed: mutating it to `Err` must break the DTOs that do not override it. Pinned on both legs, because a guard on `save` alone leaves a DTO whose `from_bytes` is laxer than its accessor panicking on the way out.
- **Clobber policy (§3):** a *differing* value saved under an equal id overwrites, asserted on both shipped DTOs — the manifest (content-derived id) and the model (authored id) behave identically, since the policy is the seam's and not the DTO's. Corrupt stored bytes do not wedge the id: `save` never reads, so a re-save repairs it.
- **Error folding (§5):** a backend `FileSystemErrorKind::NotFound` surfaces as `RepositoryErrorKind::Io(IoErrorKind::NotFound)` on `load` of an absent id; a codec failure surfaces as `RepositoryErrorKind::EncodingMismatch`; an id-vs-key disagreement, a non-injective id, and a `Persistable::integrity_check` refusal on either leg all surface as `IntegrityCheckFailed`; `RepositoryErrorKind` variant identity is preserved across `From` folds.
- **Open-seam substitutability (§6):** a caller generic over `Arc<dyn Repository<T>>` compiles and passes the round-trip contract against a second, hand-rolled `Repository<T>` impl (a test double), proving no caller coupling to `FileSystemRepository`.
- **Scheme dispatch (§4.2):** every row of the dispatch table is asserted on the value it actually keeps — the S3 legs on the composed key (pure string work, so no credentials and no request), the local and `file://` legs on an empty repository `base` **and** on a byte round-trip that lands under the real directory rather than one named `file:`. `s3a`-normalization is pinned at **both** constructors, since `open` normalizes before `new` is reached and so cannot witness a `new` that kept the alias. Bucket extraction is asserted across prefix depths (bucketless, trailing-slash, one segment, many), because a bucket that swallowed a prefix segment still composes a plausible-looking key. Unknown multi-letter schemes error rather than panic, asserted on the scheme name they report — under a feature set without `io-aws`, `s3` **and** `s3a` are among them, so the generic fallback arm is what proves those two need no arms of their own.
- **`open` performs no I/O (§4.2):** asserted as *absence after the call* — a root that did not exist before `open` still does not exist after it, so an eager `create_dir_all` or canonicalize inside `open` fails the test. `Ok`-ness alone does not discharge this: it holds whether or not the directory was touched. The deferred failure is pinned separately, on the first `save`.
- **DAG (§8):** `semstrait-common` builds with no dependency on `semstrait-model` / `semstrait-manifest` (I7) — enforced by the workspace build graph.

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
