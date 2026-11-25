---
number: 10
title: One untyped `EntityId` lane, with content-derived deterministic ids for everything unauthored
status: accepted
date: 2026-08-04
source:
- crates/semstrait-common/src/identity.rs
- crates/semstrait-manifest/src/identity.rs
- crates/semstrait-model/src/model/identity.rs
links:
- target: 45
  kind: amendedby
- target: 51
  kind: amendedby
---

# One untyped `EntityId` lane, with content-derived deterministic ids for everything unauthored

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0045](0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md): the BLAKE3 clause is superseded, because a BLAKE3 digest is not canonical UUID text; a semantic source without an authored id, implicit units included, mints UUIDv5 over its name, not its content.
> - [ADR 0051](0051-entity-ids-are-derived-from-the-natural-key.md): the authored-lane tolerance, and only that: the clause admitting v7 into the `EntityId` lane — "a bare `String` mixes v7 and v5 UUIDs in one field" — retires; an entity that omits `id:` mints `UUIDv5(namespace_for_its_class, scope_qualified_name)`, so the lane now holds v5 throughout. Everything else in 0010 stands and is extended rather than weakened: one untyped `EntityId` lane, no per-kind newtypes, no `stable_ids` sidecar, `EntityId`-keyed `BTreeMap`s, deterministic content-derived ids for everything unauthored. Its rejected alternative "time-ordered UUIDv7 … for synthesized entities" now covers the authoring mint too, which is where it always pointed.

## Context and Problem Statement

Ids are persisted keys for stored objects, coverage bitmaps, expression tables and cross-manifest comparison, so changing the lane or the derivation orphans every saved artifact and cache. The cost is that idiomatic Rust would newtype each id kind, whereas here a bare `String` mixes v7 and v5 UUIDs in one field with the version unchecked, and a list already returned as "ordered" is re-sorted anyway.

## Considered Options

- One untyped `EntityId` lane with content-derived ids (chosen).
- Per-kind compile-id newtypes (`DataKindId`, `BindingId`, `RelationshipId`, …) plus a `stable_ids` side-map, and a validated `EntityId(Uuid)` newtype.
- Time-ordered UUIDv7 or per-compile counters for synthesized entities.
- Trusting provider enumeration order instead of re-sorting.

## Decision Outcome

`type EntityId = String` (canonical UUID text, shape checked at minting sites, version nibble deliberately unvalidated) is the single identity lane: no per-kind newtypes, no `stable_ids` sidecar, `EntityId`-keyed `BTreeMap`s everywhere.

Every unauthored entity gets a deterministic content-derived id:

- UUIDv5 over a per-type namespace plus content seed for sources, bindings, interfaces, implicit unionsets and `model_id`;
- UUIDv5 over `name` for the model root;
- BLAKE3 over the canonical form for implicit compositions.

Provider enumeration is re-sorted inside compile as a determinism fence.
