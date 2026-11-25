---
number: 4
title: "Typed per-stage error kinds inside `Diagnostic<K>`; no error-code registry, no untyped errors"
status: accepted
date: '2026-08-04'
source:
- "crates/semstrait-common/src/diagnostic.rs"
- "crates/semstrait-common/src/io/error.rs"
- "crates/semstrait-planner/src/error.rs"
---

# Typed per-stage error kinds inside `Diagnostic<K>`; no error-code registry, no untyped errors

## Context and Problem Statement

Variant identity becomes the published semver contract of every crate's error surface: reinstating codes, or a central registry that downstream tooling depends on, is a workspace-wide MAJOR rewrite. The choice also cuts against the expectation that a compiler-like tool exposes greppable stable codes.

## Considered Options

- Typed per-stage kinds inside `Diagnostic<K>`, with no codes (chosen).
- The earlier ratified stable-code framework: reserved subsystem prefixes and per-stage numeric bands with `code: &'static str` on `Diagnostic`.
- One workspace-wide `anyhow`-style boundary error.

## Decision Outcome

Stable diagnostic identity is the *variant identity* of a per-stage `#[non_exhaustive]` kind enum implementing `Diagnose`, carried in a generic `Diagnostic<K>` envelope.

- The `{SUBSYSTEM}_{SEVERITY}_{NUMBER}` allocation table and all `*_E_NNNN` codes are retired.
- `String`, `anyhow` and `Box<dyn Error>` are banned on public APIs.
- Transport traits return bare kinds.
- `semstrait-api` unifies kinds only for multi-stage methods.
- Non-`Eq` upstream causes are flattened to strings.

## More Information

### Open questions

Body prose across `apis/34_semstrait_planner/*`, `apis/36_semstrait_adapter/*` and `apis/37_semstrait_catalog/*` still quotes numeric codes (`PLAN_E_*`, `ADAPT_E_*`, `CAT_E_*`). This is documentation debt against a fully retired code registry, not partial retention of it; the prose should be scrubbed.
