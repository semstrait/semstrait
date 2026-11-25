---
number: 25
title: '`DataKind` is a sealed two-axis trait hierarchy in the model with a flat enum mirror in the manifest'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-model/src/model/types/data_kinds/mod.rs
- crates/semstrait-model/src/model/types/data_kinds/variants.rs
- crates/semstrait-manifest/src/manifest/semantic_source.rs
links:
- target: 45
  kind: amendedby
- target: 53
  kind: amendedby
---

# `DataKind` is a sealed two-axis trait hierarchy in the model with a flat enum mirror in the manifest

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0045](0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md): the manifest half is superseded — `DataKindVariant` gives way to the polymorphic `role_definition` and `DataKindOrigin` goes; the model's trait hierarchy is unaffected.
> - [ADR 0053](0053-authoring-direction-is-exclusive-per-root-data-kind.md): the behavioural axis (`Public` / `Nested`) is superseded by one form per variant carrying `semantic_interface: Option<SemanticInterface>` and `public`; the structural axis, sealing and read-only accessors stand.

## Context and Problem Statement

The two-axis split is the type-level carrier of invariants D1/D5 and is mirrored into manifest types: collapsing it rewrites both surfaces, and the enum is persisted schema that must stay in sync forever. It also contradicts expectations: Rust's obvious encoding for four closed variants is a single enum (and tag enums exist anyway), the same taxonomy is duplicated in two incompatible encodings, and `data_kind.validate()` does not exist.

## Considered Options

- Sealed two-axis traits plus a manifest enum (chosen).
- A public `enum DataKind { Dataset, Grainset, Unionset, Joinset }` with runtime capability flags as the authoring representation.
- Carrying one representation end-to-end instead of accepting the model/manifest dual.
- Polymorphic lifecycle methods (validate/compile/strategy) on the trait hierarchy.

## Decision Outcome

The authoring `DataKind` is a sealed base trait plus one structural axis (`Simple`/`Complex`) and one behavioral axis (`Public`/`Nested`), exposing read-only accessors only; validate, compile and strategy selection stay stage-owned outside the hierarchy.

The manifest deliberately carries a second representation: a flat closed `DataKindVariant` enum plus `DataKindOrigin`, accepted as an intentional dual for serialization and id-keyed storage.
