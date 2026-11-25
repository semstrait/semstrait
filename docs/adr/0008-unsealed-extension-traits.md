---
number: 8
title: Extension traits are unsealed, and defaults may not lie
status: accepted
date: 2026-08-04
source:
- crates/semstrait-planner/src/strategy.rs
- crates/semstrait-adapter/src/rules.rs
- crates/semstrait-catalog/src/lib.rs
- crates/semstrait-common/src/io/fs.rs
links:
- target: 68
  kind: amendedby
- target: 73
  kind: amendedby
- target: 87
  kind: amendedby
---

# Extension traits are unsealed, and defaults may not lie

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0068](0068-adapters-are-layered-format-first.md): in its roster, `EngineAdapter` and `DialectEmit` are replaced by the per-format rewrite-rules traits, `SubstraitRules` today and a SQL counterpart when SQL emission returns. The unsealing promise transfers whole. Its defaults-may-not-lie rule is met differently for these traits: a rewrite hook's default is the identity and a mapping hook's default is the faithful format answer, and a MINOR-added method defaults the same way rather than to a `MethodNotSupported` error — an honest identity is not a lie, and a refusing default would be the refusal that 0068's *The seam* removes. `InvalidRewrite` is these traits' bound on an unverifiable third-party rewrite, as it already is for `OptimizerPass`.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): `OptimizerPass` leaves the unsealed roster. `Strategy` stays unsealed; it buys per-body substitution only, and its one required method is `decide`.
> - [ADR 0087](0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md): in its *Decision Outcome*, `RegistryExtension` leaves the list of traits open to out-of-workspace implementors: a function enters by its declaration in the function catalog (ADR 0087, rule 3). The rest of 0008 stands.

## Context and Problem Statement

Unsealing is a one-way public promise: sealing later breaks every external implementor, and the method set is frozen against them. It is deliberately surprising that a correctness-critical rewrite stage admits arbitrary external plan mutation, and that an internal-looking abstraction is left open, inside an otherwise CI-locked, closed-catalog design.

## Considered Options

- Unsealed extension traits whose defaults may not lie (chosen).
- Sealed traits via a private supertype, limiting strategies, passes, adapters and providers to the workspace, combined with convenience default methods returning empty results.

## Decision Outcome

`Strategy`, `OptimizerPass`, `EngineAdapter`, `DialectEmit`, `CatalogProvider`, `FileSystem` and `RegistryExtension` are explicitly open to out-of-workspace implementors with no private escape hatch. Consequently:

- adding a required method is MAJOR;
- built-in dialect rendering is snapshot-frozen within a MINOR;
- unverifiable third-party rewrites are absorbed by `PassFailed`/`InvalidRewrite` variants;
- any MINOR-added method must default to an explicit `MethodNotSupported`-style error rather than an empty or `None` answer.
