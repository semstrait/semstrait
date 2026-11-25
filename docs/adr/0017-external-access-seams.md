---
number: 17
title: "External access seams: two orthogonal traits, no data-derived schema, `object_store` hidden, whole-object repository"
status: accepted
date: '2026-08-04'
source:
- "crates/semstrait-catalog/src/lib.rs"
- "crates/semstrait-common/src/io/fs.rs"
- "crates/semstrait-common/src/io/repository.rs"
---

# External access seams: two orthogonal traits, no data-derived schema, `object_store` hidden, whole-object repository

## Context and Problem Statement

Adding a diagnostics channel, a listing method, a registry parameter or an escape hatch breaks every impl and caller, and re-admitting data-derived schema would change manifest inputs everywhere. The result reads oddly on purpose: a filesystem-backed catalog that cannot report the schema of Parquet it can read, hand-written `Path`/`FileEntry` duplicating the dependency's vocabulary, `CatalogId` as a first-class newtype nothing routes on, and a test double documented as *less* strict than production.

## Considered Options

- Orthogonal traits, whole-object repository (chosen).
- The legacy unified `StorageProvider::read_schema(path, format)` parsing format headers.
- Re-exporting `object_store` or per-backend SDK types with an opt-in escape hatch (`S3SourceBuilder::with_object_store_builder`).
- One unified load path returning `(T, Diagnostics)` with a repository-injected `validate` hook, a bespoke `InMemoryRepository`, and seam-level path sandboxing.
- A `CatalogRegistry` keyed on `CatalogId` plus a scheme-dispatch `FileSystem` router.
- A closed `CatalogType` enum.

## Decision Outcome

- `CatalogProvider` (structured metadata) and `FileSystem` (raw bytes) are independent traits. `read` returns `Bytes` and MUST NOT parse Parquet footers, ORC stripes or CSV headers, so schema comes only from a catalog or the author.
- Every non-memory `FileSystem` thin-wraps Arrow `object_store`, with no `object_store` type on any signature and no escape hatch.
- `Repository<T>`/`Persistable` is a fail-fast, id-keyed, whole-object seam with no diagnostics channel, no listing and no validation hook. Its sole in-memory substrate is one knowingly-divergent `InMemoryFileSystem`, and key containment is delegated to the transport rather than guaranteed at the seam.
- v1 injects one provider and one filesystem instead of a registry.
- `provider_type` stays an open string, validated only in `semstrait-catalog`.
- These surfaces deliberately omit retry, streaming, credentials and validation because the dependency or a layer above owns them.
