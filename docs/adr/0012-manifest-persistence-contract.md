---
number: 12
title: 'Manifest persistence contract: deterministic JSON, epoch-only versioning, full load-time re-validation, bitmask coverage'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/manifest/top.rs
- crates/semstrait-manifest/src/compile/driver.rs
- crates/semstrait-manifest/src/manifest/expression.rs
- crates/semstrait-manifest/src/manifest/bitmap.rs
links:
- target: 36
  kind: amendedby
- target: 41
  kind: amendedby
- target: 42
  kind: amendedby
- target: 46
  kind: amendedby
---

# Manifest persistence contract: deterministic JSON, epoch-only versioning, full load-time re-validation, bitmask coverage

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0036](0036-bit-position-assignment.md): `bit_position` is a dense index scoped to one manifest artifact, not a persistable identifier, and no consumer may persist a bit or apply a mask from one manifest to another.
> - [ADR 0041](0041-recompile-is-the-callers-decision.md): `manifest_epoch` is renamed `manifest_version`.
> - [ADR 0042](0042-manifest-validity-is-a-compile-stage-property.md): supersedes the load mechanism and upholds its guarantee, which is relocated above the crate: `semstrait-manifest` has no load path, and `validate()` applies compile's gates to a decoded manifest, raising `CompileError` and accumulating by default with a fail-fast flag; a caller can reach past the composing layer to the generic `Repository<SemanticManifest>`, deserialize, and consume the result unvalidated — nothing prevents it.
> - [ADR 0046](0046-physical-source-is-flat-with-explicit-schema-provenance.md): `schema_id` replaces `schema_fingerprint`.

## Context and Problem Statement

The deterministic JSON bytes seed `model_hash`, `schema_fingerprint` and content dedup, so any change to format, determinism or coverage representation invalidates every persisted manifest and cache. The combination is deliberately unusual — a machine-consumed compiled artifact in pretty JSON, with no embedded format versions, fully re-validated on read (derived fields both stored *and* re-derived), plus a second integer id lane in a design that otherwise bans integers.

## Considered Options

- Deterministic JSON, re-validated on load, with bitmask coverage (chosen).
- A compact or versioned binary encoding (protobuf/bincode) with per-record `format_version` fields.
- Trusting compile output at load, with gates G1/G2/G4 enforced only at compile time.
- OpenAPI-style `$ref` JSON-pointer indirection.
- `BTreeSet<EntityId>` coverage sets instead of bitmasks.

## Decision Outcome

- **Encoding.** The manifest is deterministic, human-readable JSON (BTreeMap ordering, sorted fields, canonical floats, byte-identical re-encoding).
- **Versioning.** `manifest_epoch` is the only migration signal — no per-record or per-`Expr` `format_version`.
- **Load.** `Repository::load` runs one fail-fast pass that re-checks every cross-reference and every derived value, and the load guarantee is contractual for all consumers.
- **Coverage.** Coverage everywhere is a `SemanticBitmask` over epoch-scoped `bit_position` ordinals held in a global registry.
