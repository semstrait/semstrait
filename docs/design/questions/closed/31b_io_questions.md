---
doc: design/questions/closed/31b_io_questions
status: Closed
purpose: Resolved I/O-layer questions originally raised against `apis/31b_semstrait_common_io.md`
---

# Closed Questions — `apis/31b_semstrait_common_io.md`

Ratification pass on 2026-04-17 closed every open item in this file. This document is retained as a historical record of the decisions; no live items remain. New questions arising in post-v1 work go into a separate file.

Cross-references in this document are by section (e.g. `31b §3.2`, `32 §10.4`). No code-path references are used, per `00 §8`.

---

## Q-IO-001 — Round-trip-preserving `DumpMode` — **RETIRED (CANONICAL ONLY)**

**Resolution.** v1 ships `DumpMode::Canonical` only. A future `Faithful` variant — comment-preserving, anchor-preserving, author-order-preserving — does not land in v1 and is not planned for any follow-on MINOR. It would require a comment-preserving YAML parser (`saphyr` / `yaml-rust2` with custom adapter) *and* round-trip-able AST retention on `SemanticModel`; both are non-trivial and solve a problem no current tooling has asked for. `DumpMode` is `#[non_exhaustive]`, so if a concrete tooling need ever materializes, adding `Faithful` remains a MINOR change. No TD tracking required.

---

## Q-IO-002 — Additional transport back-ends (HTTP / GCS / Azure) — **RETIRED (v1 FROZEN AT 3)**

**Resolution.** v1 ships with `Local`, `InMemory`, `S3` only. `object_store` (the chosen back-end library, `31b §1.4`) already implements `HttpStore`, `GoogleCloudStorage`, `MicrosoftAzure` — each would be ~30 LOC of trait delegation + one feature flag (`io-http`, `io-gcs`, `io-azure`) in `semstrait-common`. Each lands as an additive MINOR when a concrete caller asks; neither we nor adopters pay the compile cost until opt-in. Priority order when demand surfaces: HTTP (public YAML mirrors, dev servers) > GCS > Azure.

---

## Q-IO-003 — Multi-file / directory loader — **CLOSED (OUT OF SCOPE FOREVER)**

**Resolution.** `core::io` ships single-blob `Source` / `Sink` primitives. Multi-file aggregation is **permanently out of scope** for `core::io`, `model::io`, `manifest::io`, and every other domain wrapper. Callers that want to walk a directory tree (CLI preview, LSP re-parse-on-change, build orchestrators) perform the enumeration in their own code path using `std::fs::read_dir`, `object_store::ObjectStore::list`, or a higher-level CLI helper, and call `load_model` on each resulting blob independently. This preserves the single-responsibility of transport primitives and avoids baking product-level merge policy (cross-file collision handling, `$include` directive semantics, deduplication rules) into `core::io`.

---

## Q-IO-004 — WASM back-end constraints — **CLOSED (NO WASM IN v1)**

**Resolution.** v1 does not support the `wasm32-unknown-unknown` target. `semstrait-common::io` is not compiled for wasm; `cfg` guards in the back-end layer prevent wasm builds from ever exercising `tokio::fs` / S3 paths. Adding wasm support (`InMemory` only, `fetch`-backed HTTP, IndexedDB-backed `Sink`, or any combination) is not planned for any follow-on MINOR unless a concrete tooling need surfaces. Browser-hosted tools that want `semstrait` must preload model bytes into their runtime and call sync-path parse functions (`semstrait_model::parse(&str)`), bypassing `io` entirely.

---

## Q-IO-005 — Atomic compare-and-swap on `Sink` — **RETIRED (LWW IN v1)**

**Resolution.** v1 ships last-writer-wins semantics per `31b §4.3`, matching `object_store`'s native behaviour. Two concurrent writers on the same path produce whichever completes last. If a concrete pipeline hits the clobber problem, a `ConditionalSink` extension trait with `store_if_absent(bytes) -> Result<(), ConditionalStoreError>` can land as a MINOR (opt-in; default-implementable as `store` + pre-check on back-ends that lack native CAS). Not tracked as TD because v1 use cases do not exhibit concurrent dumps on the same path.

---

## Q-IO-A — Byte-level vs text-level transport — **RESOLVED (BYTES + `FromIoBytes`)**

**Resolution.** `Source::read_raw` returns `Bytes`; a generic default method `Source::read<T: FromIoBytes>` materializes any target type (`Bytes`, `Vec<u8>`, `String`, …). `Sink::write_raw(Bytes)` + `Sink::write<B: IntoIoBytes>` symmetric. UTF-8 validation is centralized in `FromIoBytes for String`, emitting `IoError::Malformed`. Domain crates pick their target type: `model::io::load_model` reads `String` (YAML is text); `manifest::io::load_manifest` reads `Bytes` (manifest is binary-encoded JSON / MessagePack). Specified in `31b §3` / `§4` / `§5`.

---

## Q-IO-B — Cancellation / timeout plumbing — **RESOLVED (PURE TOKIO IDIOM)**

**Resolution.** No `CancellationToken` / deadline parameter in the trait. Callers enforce cancellation by dropping the future; timeouts via `tokio::time::timeout(duration, src.read_raw())`. Matches `object_store`, `tokio::fs`, `reqwest` conventions. Back-ends must leave no partial state on cancellation (delegated to `object_store`). Specified in `31b §3.2`.

---

## Q-IO-C — Size limits — **RESOLVED (NO LIMIT AT TRANSPORT)**

**Resolution.** `core::io` imposes no upper bound on payload size. Callers that need a cap enforce it in their own layer. Matches `object_store`'s default posture; avoids baking a number into the trait. Specified in `31b §3.4`.

---

## Q-IO-D — `Location` ↔ back-end identity and connection reuse — **RESOLVED (CACHED DISPATCH)**

**Resolution.** The `Location::from_str` dispatch path consults a process-global `OnceLock<DashMap<ClientKey, Arc<object_store::aws::AmazonS3>>>` keyed by `(region, endpoint_url)`. First call constructs the `AmazonS3` client; subsequent calls reuse. No eviction in v1. Callers that construct `S3Source` explicitly via `S3SourceBuilder` bypass the cache (they own and reuse the client themselves). Specified in `31b §6.2`.

---

## Q-IO-E — S3 credential override path — **RESOLVED (`S3SourceBuilder` + `object_store` escape)**

**Resolution.** `S3SourceBuilder` exposes a thin opinionated API over `object_store::aws::AmazonS3Builder`: `with_region`, `with_endpoint`, `with_credentials`, `with_session_token`, `with_allow_http`, `build`. Advanced configuration (retry policy, proxy, custom HTTP client) uses the escape hatch `with_object_store_builder(AmazonS3Builder)`. Specified in `31b §8.3`.

---

## Q-IO-F — Error-type composition across domain wrappers — **RESOLVED (`#[non_exhaustive]` ENUM WRAPPER)**

**Resolution.** `IoError` is `#[non_exhaustive]`. Domain wrappers (`ModelLoadError`, `ModelDumpError`, `CatalogsLoadError`, `CatalogsDumpError`, `ManifestLoadError`, `ManifestDumpError`) are `#[non_exhaustive]` enums with variants like `Io(IoError)`, `Parse(ParseErrors)`, `Decode(DecodeError)`. Adding any variant at any layer is a MINOR change because consumers must have `_ => ...` catch-all arms. Ergonomic `match` at the call site; stable evolution through the dependency chain. Specified in `31b §7` (last paragraph).

---

## Q-IO-G — Concurrent-write semantics on `Sink` — **RESOLVED (LWW, DOCUMENTED)**

**Resolution.** Merged with Q-IO-005 above. Last-writer-wins at the transport level. Callers that need coordination serialize at the domain wrapper layer (e.g. a `tokio::sync::Mutex` in their own code). Specified in `31b §4.3`.

---

## Q-IO-H — Format-agnosticism of `load_model` — **RESOLVED (YAML ONLY, COMMITTED)**

**Resolution.** `load_model` is YAML-only, forever. No `Format` parameter, no content sniffing, no format suffix variants. If a JSON / TOML / alternative encoding is ever needed, it lands as a distinct function (`load_model_json`) in a MINOR. Input shapes are unchanged:

- `semstrait_model::parse(&str)` — sync, UTF-8 in-memory string, pre-existing.
- `semstrait_model::io::load_model(&impl Source)` — async, any `Source` (local / S3 / in-memory).
- `load_model(&Location::from_str("s3://…")?)` — one-liner via `Location`.

Specified in `32 §10.4`.

---

## Q-IO-I — `describe()` contract — **RESOLVED (STABLE CONTENT-ADDRESSABLE IDENTITY)**

**Resolution.** `describe()` is a stable content-addressable identity, not a free-form log string. Equal `describe()` ⇒ equal bytes (absent concurrent mutation). Per-back-end consequences: `LocalFile` uses absolute path; `S3Source` uses `s3://bucket/key`; `InMemory` requires a user-picked `name` at construction (`InMemory::new(name, bytes)`). Must never emit secrets. Safe to log at any level. Specified in `31b §3.5` and SR-IO-3.

---

## Q-IO-J — Ownership of `load_catalogs` / `dump_catalogs` — **RESOLVED (`model::io` OWNS)**

**Resolution.** `semstrait-model::io` owns `load_model`, `dump_model`, `load_catalogs`, `dump_catalogs`. `catalogs.yaml` is a sibling file parsed in the same layer; `CatalogsConfig` resolves catalog `$ref`s in the model, and the manifest compilation pass consumes the resolved config to fetch metadata from the external catalog (Polaris / Unity / etc.). No separate `semstrait-catalog::io` load / dump surface. Specified in `32 §10.4`.

---

*All items in this file are now resolved or closed. This file is retained as a historical record.*
