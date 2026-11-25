---
number: 31
title: Metadata-derived Dimensions are a compile-time third stratum, not expressions
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/manifest/semantic_definition.rs
- crates/semstrait-manifest/src/compile/definitions
- crates/semstrait-manifest/src/manifest/coverage.rs
links:
- target: 37
  kind: amendedby
- target: 70
  kind: amendedby
- target: 81
  kind: amendedby
---

# Metadata-derived Dimensions are a compile-time third stratum, not expressions

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0037](0037-semantics-bind-to-pooled-physical-expressions.md): its decision "a `#[non_exhaustive]` public mapping variant plus a companion per-source storage lane in the persisted manifest" is replaced, on carrier shape only: metadata remains a compile-time third stratum, never a registry function and never a `PhysicalExpr` *variant*, resolved eagerly and fail-fast — but it lands as an ordinary pooled `Literal` leaf. The sections that only apply that decision read so: in *Context and Problem Statement*, that phrase and "an enum variant no parser can produce"; and in *Decision Outcome*, "a distinct `SemanticMappingValue::Metadata(recipe)` variant" and "onto `ResolvedPhysicalSource.metadata_values`". So the mandated variant and the lane are both retired, along with the earlier design's `manifest.metadata_index` (the planner now finds the constant already in `mapping`). The rest of 0031 stands.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): the `partition` form is admitted, the form that the model reserves: its value is the percent-decoded text after the first `=` of a discovered `key=value` folder, and a metadata value is constant per source or per partition; `path_token` still returns the raw segment.
> - [ADR 0081](0081-a-partition-key-is-a-column-and-partition-def-names-the-dimension-that-maps-it.md): the `partition` form is not a metadata form, and a metadata value is constant per source; ADR 0070's clause "the `partition` form is admitted" no longer holds. `path_token` and the rest of 0031 stand.

## Context and Problem Statement

This is recorded because it is a `#[non_exhaustive]` public mapping variant plus a companion per-source storage lane in the persisted manifest, and the partial-evaluation boundary (metadata as compile-known constants) depends on its fail-fast strictness.

It is also counter-intuitive: an enum variant no parser can produce, synthesis that runs before validation and can override an author-written entry, a "give me the partition value" helper that returns the key-prefixed token, and a `NullFill` coverage variant that deliberately is not used here.

## Considered Options

- Metadata as a compile-time third stratum (chosen).
- Registering `path_token` as a `14a` registry function / `PhysicalExpr` variant authored under `semantic_mapping.expr` (`TD-MAP-METADATA-FOLD`, reversed 2026-04-27): this would push a compile-known constant into the runtime expression language.
- Splitting on `=` inside the mechanic, rather than returning the raw segment.
- The previous lenient per-source NULL fallback for non-applicable sources or failed casts.

## Decision Outcome

Metadata extraction is layer-3 compile mechanics — never a `PhysicalExpr` variant and never a registry function. The v1 inhabitant `path_token` returns the whole raw path segment, e.g. `"year=2024"`.

Compile synthesizes a distinct `SemanticMappingValue::Metadata(recipe)` variant, with no author-facing `semantic_mapping:` form, before the completeness check. It resolves per-source `LiteralValue`s onto `ResolvedPhysicalSource.metadata_values`, treating any non-applicable source or failed cast as a fail-fast compile error rather than a NULL fill.
