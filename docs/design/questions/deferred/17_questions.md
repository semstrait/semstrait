---
doc: design/questions/deferred/17_questions
status: Deferred
purpose: Temporal-shape questions parked for post-v1 ratification
---

# Deferred Questions — `foundations/17_temporal_shape.md`

> Items deferred to v2 (or later) ratification. Live items are in [`../open/17_questions.md`](../open/17_questions.md); closed items in [`../closed/17_questions.md`](../closed/17_questions.md).

---

## Q-TEMPORAL-004 — Multi-shape heterogeneous `Request.temporal` resolution

**Question.** `17 §6.5` notes that a `Request` spanning a composed surface with heterogeneous constituent shapes (e.g. one `Scd::Type2` + one `Snapshot` + one `Events`) needs a well-defined resolution of `Request.temporal.as_of` across all shapes. What is the ratified algorithm?

**Refs.**

- `17 §6.5` — illustrative cases and DEFERRED statement.
- `17 §8.4` — shape-aware composition pass (DEFERRED) that runs anchoring + rollup per constituent.
- `16 §11` — implicit composition algorithm; doesn't ratify temporal-composition semantics.
- `34 §…` (pending) — planner strategies.

**Options.**

- **A. Per-constituent as-of independence.** Each shape-classified constituent interprets `Request.temporal.as_of` through its own lens (§6.2 rules); no cross-constituent consistency enforcement. The composed result is the natural join / union of per-constituent anchored views.
- **B. Dominant-shape-driven.** One shape in the composition is nominated "dominant" (typically the `from:` clause's root kind) and drives the as-of; other constituents follow.
- **C. Per-Request explicit anchor declaration.** Author declares `Request.temporal.per_data_kind: { "orders": as_of_A, "customers_scd": as_of_B }` when multi-shape.

**Arguments for A.**

- Simplest ratification. Each shape already has well-defined per-shape as-of semantics from §6.2.
- No new vocabulary. `Request.temporal` stays a single `as_of`.
- Degenerate case when all constituents agree on shape = identity (which is what Round 1 happens).

**Arguments for B.**

- Authors often *intend* a single anchor moment ("as of end-of-quarter 2024-12-31"); Option A's per-constituent independence could surface as surprising when e.g. a Snapshot aligns to a cadence boundary but the SCD doesn't.
- Matches how dbt MetricFlow and similar tools treat a "metric_time" — one anchor for the whole query.

**Arguments for C.**

- Ratifies the escape hatch upfront; authors can opt in to per-constituent control when they need it.
- Complements Option A (use Option A as default, Option C as explicit override).

**Current position in `17`.** DEFERRED; Option A looks likely but the `34` algorithm ratification is where this settles.

**Next step.** `34 §…` ratifies. Possibly as A + C (default + explicit override).
