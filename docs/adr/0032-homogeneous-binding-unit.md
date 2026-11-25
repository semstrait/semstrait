---
number: 32
title: The Binding is the atomic homogeneous unit; text-format inference qualifies determinism
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/manifest/physical_source.rs
- crates/semstrait-manifest/src/compile/sources.rs
- crates/semstrait-manifest/src/compile/catalog.rs
- crates/semstrait-manifest/src/compile/definitions/mod.rs
links:
- target: 43
  kind: amendedby
- target: 49
  kind: amendedby
---

# The Binding is the atomic homogeneous unit; text-format inference qualifies determinism

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0043](0043-authored-primitives-get-typed-carriers.md): `SemanticBinding` retires and per-source variation lives on the expression; "`Binding` retires as a glossary term", since there is no single join-point entity.
> - [ADR 0049](0049-a-dataset-names-one-physical-source-as-a-semantic-source.md): with the binding retired, the remedy reads as *separate Datasets composed by an authored Unionset*; "one physical column name per Semantics across all its sources" is now a property of the single shared `semantic_mapping` rather than a rule to check, and one `FileFormat` per unit follows from `format` being one field plus the restriction that `paths` and `tables` may not both be non-empty on one Dataset; cross-source *type* disagreement (`COMP_E_0317`) still has a subject inside `tables` alone, where two catalog tables may type the same column differently, and becomes a cross-branch reconciliation question on the synthesised Unionset, whose successor rule is not settled.

## Context and Problem Statement

Schema reconciliation, Coverage and every strategy rely on the atomic-Binding assumption. Admitting per-source naming or mixed formats would add fields and new coverage semantics throughout; forbidding inference later would break existing models; and the qualified invariant must stay consistent tree-wide.

It is a surprising boundary: a Binding already carries N sources with per-source Coverage (so per-source renaming looks supported), the compiler already emits reconciliation casts yet refuses to cast divergent per-source types, and a headline "byte-identical manifest" invariant is given a sanctioned non-reproducible path.

## Considered Options

- An atomic homogeneous Binding, with inference for text formats (chosen).
- Per-source name mapping and mixed-format Bindings (the Cube.js-style escape hatch), with plan-time per-source cast reconciliation.
- Requiring a declared schema for text formats, i.e. forbidding compile-time inference outright: rejected because it makes the common CSV/JSON case unusable, at the cost of qualifying the determinism invariant.

## Decision Outcome

One Binding resolves to a single `FileFormat`, one physical column name per Semantics across all its sources, and one type per Semantics. Cross-source type disagreement is `COMP_E_0317` at compile, not per-source casting at plan time. So heterogeneity must be expressed as separate Bindings composed by a Unionset, or fixed upstream.

Schemaless CSV/JSON sources are inferred at compile with `COMP_W_0301`, which explicitly redefines the determinism invariant as "relative to a given catalog + filesystem snapshot".
