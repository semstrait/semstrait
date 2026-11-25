---
number: 35
title: "Behavioral contract gates as ratifiable Gherkin with a one-way `@target → @now` ratchet"
status: accepted
date: '2026-08-04'
source:
- "tests/features"
---

# Behavioral contract gates as ratifiable Gherkin with a one-way `@target → @now` ratchet

## Context and Problem Statement

This is recorded because it adds a BDD dependency and a step-library/`World` architecture bound to the public pipeline entry points, the feature files become ratified spec surface that abandoning the approach would discard, and the `@now` fraction becomes the project's progress meter.

It is also unusual on its face: a Rust workspace carrying Gherkin files, a test crate that refuses to assert anything not written in a design doc, and a CI suite that deliberately contains failing tests.

## Considered Options

- Ratifiable Gherkin gates with a one-way ratchet (chosen).
- Plain Rust integration tests (the retired-in-place root `tests/` MVP harness) asserting current behavior, with unearned gates skipped or omitted until the excluded crates return: rejected because it gives up the ratchet and the design-doc citation requirement.

## Decision Outcome

The wide-blast-radius test tier is a `cucumber-rs` suite whose `.feature` files, kept in `tests/features`, are the ratifiable artifact.

- Every scenario must cite a ratified design clause, and a gate with nothing to cite is a spec defect.
- Scenarios are tagged `@now` (hard gate) or `@target` (allowed-to-fail xfail whose *passing* must be promoted to `@now` in the same PR).

## More Information

### Open questions

The source design document was still `status: proposed`, so this decision is recorded from a pre-ratification doc: either the doc gets ratified or this ADR should be downgraded to proposed.
