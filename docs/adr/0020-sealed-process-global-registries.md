---
number: 20
title: "Sealed process-global registries that panic on registration collisions"
status: accepted
date: '2026-08-04'
source:
- "crates/semstrait-sql/src/functions/registry.rs"
- "crates/semstrait-sql/src/functions/error.rs"
- "crates/semstrait-sql/src/functions/catalog/mod.rs"
links:
- target: 87
  kind: amendedby
---

# Sealed process-global registries that panic on registration collisions

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0087](0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md): in its *Decision Outcome*, "`function_registry()` … sealed at first access by folding in every linked `RegistryExtension`" becomes: `semstrait-sql`'s `registry()` is sealed at first access over the function catalog's declarations only, and its faults are ADR 0087's seal faults (rule 9). The rest of 0020 stands.

## Context and Problem Statement

`&'static` removes registry handles and lifetimes from every resolution and compile signature, so per-request registries would change every call site and the entire testing strategy. It is a deliberate exception: crates that ban I/O, async and global state keep a process-wide singleton that aborts the process on a misconfigured extension, and sessions restrict adapters without owning a registry.

## Considered Options

- Sealed process-global registries (chosen).
- Per-compile or per-session injectable registry instances with runtime `register`/`deregister` and collisions reported as recoverable `CompileError` diagnostics under a shadowing precedence.

## Decision Outcome

`function_registry()` and `AdapterRegistry` are `OnceLock`-backed `&'static` tables, sealed at first access by folding in every linked `RegistryExtension`/`AdapterContribution`.

- No mutation API, no per-Model or per-request configurability, and no tier precedence.
- Registration faults (reserved-tag collision, core shadowing, adapter-vs-adapter clash) are **panics** at initialization.
- Per-session allow-lists layer on top.
