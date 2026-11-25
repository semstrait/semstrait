---
number: 9
title: Ship engines and providers as separate crates, released lock-step behind a logic-free facade
status: accepted
date: 2026-08-04
source:
- crates/semstrait/src/lib.rs
- crates/semstrait-adapter/Cargo.toml
- crates/semstrait-adapter-datafusion/Cargo.toml
- Cargo.toml
links:
- target: 68
  kind: amendedby
---

# Ship engines and providers as separate crates, released lock-step behind a logic-free facade

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0068](0068-adapters-are-layered-format-first.md): its rule is about engine *linkage*, and that rule stands: no engine dependency and no `cfg(feature)` engine gating in `semstrait-adapter`, and an engine adapter that links its engine is its own crate. Amended in one clause: in *Decision Outcome*, "Each engine adapter … is its own independently-tiered crate" reads: an engine adapter is its own crate that depends on `semstrait-adapter`. Its library links no engine, and its engine is a dev-dependency of its tests only: the tests that run a plan on the engine need a crate that takes the engine as a dev-dependency, and the consumer stays out of `semstrait-adapter` (0068's *Decision Outcome*).

## Context and Problem Statement

Crate names, exact pins and the "no facade-owned types" rule are published artifacts: collapsing back to features or loosening pins rewrites the dependency graph and silently permits type-identity divergence. The choice knowingly rejects the obvious Rust idiom (one crate, per-engine features, caret ranges), mandates exact pins that are usually an anti-pattern, and leaves the namesake crate holding under 200 lines of `pub use`.

## Considered Options

- Separate crates, lock-step releases, a logic-free facade (chosen).
- A monolithic `semstrait-adapter`/`semstrait-catalog` with per-engine Cargo features and `default = ["every-adapter"]`.
- Independent per-crate semver with caret version ranges.
- A convenience facade owning wrapper types and one-shot helpers (`semstrait::run_sql`, `semstrait::Diag`).

## Decision Outcome

Each engine adapter and catalog provider is its own independently-tiered crate — no `cfg(feature)` engine gating and no engine dependencies in `semstrait-adapter`.

All core crates publish at one coordinated version.

The `semstrait` facade is re-exports plus exactly one free function. It pins every sub-crate at `=X.Y.Z` and surfaces pipeline transitions as one extension trait per hop (`Compile`/`Query`/`Adapt`).
