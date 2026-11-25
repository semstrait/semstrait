---
number: 6
title: "`tracing` is the only observability channel and is orthogonal to diagnostics"
status: accepted
date: '2026-08-04'
source:
- "crates/semstrait-catalog/src/iceberg.rs"
- "crates/semstrait-common/src/diagnostic.rs"
---

# `tracing` is the only observability channel and is orthogonal to diagnostics

## Context and Problem Statement

The observability channel is both a technology lock-in and a behavioral contract that embedders build rendering and log pipelines against — dual emission or a bundled subscriber would duplicate or reorder everyone's output, and re-admitting an `Info` severity would change what every stage is expected to emit — so it is fixed up front despite the surprise that the library logs nothing about the failure it just returned and that error kinds have nowhere to record which stage produced them.

## Considered Options

- `tracing` only, orthogonal to diagnostics (chosen).
- The `log` crate, or an embedder-coupled logger, with diagnostics mirrored onto the log channel.
- A library-registered tracing subscriber.
- An `Info` variant on `Severity`.

## Decision Outcome

`tracing` is the sole observability dependency: no `log`, no library-installed subscriber, and no `println!`/`eprintln!` anywhere in library code.

- A `Diagnostic<K>` returned via `Result` is never also emitted as a tracing event.
- `Severity` carries only `Error` and `Warning`.
- Informational output plus stage-of-origin move to span fields.
