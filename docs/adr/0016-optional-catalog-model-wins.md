---
number: 16
title: The catalog is optional, fetched eagerly, and the model wins on conflict
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/compile/driver.rs
- crates/semstrait-manifest/src/compile/catalog.rs
links:
- target: 41
  kind: amendedby
---

# The catalog is optional, fetched eagerly, and the model wins on conflict

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0041](0041-recompile-is-the-callers-decision.md): `catalog_fingerprint` is retired, so nothing hashes the catalog as a whole, by identity or by fetched content; drift detection reads per-source data.

## Context and Problem Statement

The rule determines the `compile` signature, the authoring surface, and the trust model that drift/fingerprint invalidation is built on, so inverting precedence would change compiled manifests for models already authored. It is deliberately counter-intuitive: the system that owns physical metadata is overridden by hand-authored claims with a warning rather than an error, and metadata is walked eagerly rather than on demand.

## Considered Options

- Optional eager catalog; model wins (chosen).
- A mandatory, authoritative catalog (catalog-wins, conflict = error) with lazy per-source resolution, plus identity-only `catalog_fingerprint` instead of fetched-content hashing.

## Decision Outcome

`compile` takes `catalog: Option<&dyn CatalogProvider>`.

- With no catalog the model is the source of physical truth, via optional `locator`/`source_type`/`projected_schema`/`version_ref` on bindings.
- With a catalog, every reachable source is fetched eagerly, and absence or fault is a hard error.
- When both supply a value, the **model wins**, with only `COMP_W_CATALOG_OVERRIDE`.
