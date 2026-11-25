---
number: 19
title: "Substrait is a derived, outbound-only target; the Rust types are canonical"
status: accepted
date: '2026-08-04'
source:
- "crates/semstrait-ir/src/plan"
- "crates/semstrait-adapter/src/producer.rs"
---

# Substrait is a derived, outbound-only target; the Rust types are canonical

## Context and Problem Statement

`PlanNode` mirrors `Rel` so closely — and every IR type derives `Serialize` + `Deserialize` next to a Substrait mapping table — that it reads as a bidirectional interchange format, so the direction needs recording. Adopting proto as canonical, or admitting inbound plans, would rewrite every producer/consumer signature and force the vocabulary to become total and version-tolerant over foreign constructs.

## Considered Options

- Canonical Rust types, outbound-only Substrait (chosen).
- Substrait protobuf as the canonical/persisted plan representation (a proto-first IR) with bidirectional interop.

## Decision Outcome

Substrait inspires the structure and is a co-equal *output*, but the canonical form is the in-memory `SemanticPlan`/`PhysicalExpr`.

- The mapping table is unidirectional, and conversion code lives in `semstrait-adapter`.
- Ingesting an externally-produced `substrait::proto::Plan` is out of scope; `Deserialize` exists only for same-version internal transport and fixtures.
