---
number: 23
title: 'Adapter emission architecture: central orchestrator, split dialect traits, dialect-blind rewrites, mandatory quoting'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-adapter/src
- crates/semstrait-adapter-datafusion/src/rewrite.rs
- crates/semstrait-sql/src/functions
links:
- target: 68
  kind: amendedby
---

# Adapter emission architecture: central orchestrator, split dialect traits, dialect-blind rewrites, mandatory quoting

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0068](0068-adapters-are-layered-format-first.md): its shape (a central orchestrator owns the walk; per-variant hooks contribute rendering only; adapt is pure, sync, byte-deterministic and touches no catalog and no running engine — an engine's *rules* are an input to adapt, an engine's runtime never is) is generalised from SQL to every format and is the seam in 0068's *The seam*, with one widening: a hook may rewrite a subtree before lowering, not only render it; `SubstraitProducer` owns the Substrait walk. Its SQL-specific clauses — the `emit_sql`/`emit_substrait` pair, the `Dialect`/`DialectEmit` split across two crates, dialect-blind `Expr→Expr` rewrites, mandatory quoting — describe the SQL format adapter, which is out of scope with SQL emission, and are dormant, not repealed. `semstrait-ir`'s `Dialect`, `DialectId` and `Capability` types are deleted, and `EngineAdapter` and `DialectEmit` are replaced by the per-format rewrite-rules traits, `SubstraitRules` today and a SQL counterpart when SQL emission returns.

## Context and Problem Statement

The trait method surface is the public extension contract that third-party adapters and release audits are built on: moving traversal ownership, merging the two traits, or relaxing quoting selectively breaks every implementor or the soundness guarantee. The shape also reads oddly: two traits in two crates both describing "a dialect", engine crates unable to call the walker they extend, a rewrite forbidden from knowing its dialect, and catalog-resolved table names quoted as if untrusted.

## Considered Options

- Central orchestrator with split dialect traits (chosen).
- One `Dialect` trait carrying both capabilities and rendering.
- Per-adapter independent emitters walking the plan themselves.
- A single dialect-aware pass that reshapes while printing.
- A per-adapter known-safe-name fast path skipping quoting.
- An async adapter performing catalog/drift lookups during emission.

## Decision Outcome

- `emit_sql`/`emit_substrait` are `pub(crate)` orchestrators that own the plan walk, CTE structuring and rewrite dispatch.
- `semstrait-ir` owns a structural `Dialect` (id plus capabilities), while `semstrait-adapter` adds an operational `DialectEmit` contributing only per-variant rendering hooks with ANSI defaults.
- `Expr→Expr` rewrites are dialect-blind (`EngineAdapter::rewrite_expr`), and each function belongs to exactly one layer.
- Every identifier and string literal passes through `quote_identifier`/`escape_string_literal`, with no trusted-catalog fast path.
- `adapt` is pure, sync, byte-deterministic, and never touches a catalog or an engine.
