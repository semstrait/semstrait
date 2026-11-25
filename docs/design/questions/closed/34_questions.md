---
doc: design/questions/closed/34_questions
status: Closed
purpose: Resolved questions originally raised against `apis/34_semstrait_planner.md`
---

# Closed Questions — `apis/34_semstrait_planner.md`

> Historical record of ratified planner decisions. Live items are in [`../open/34_questions.md`](../open/34_questions.md).

---

## Q-PLAN-008 — Field-first depth bound (`MAX_IMPLICIT_COMPOSITION_DEPTH`) — CLOSED (2026-04-28)

**Status: CLOSED.** Mirrored from Q-COMP-001 (closed 2026-04-28 at value `4`). `34 §10.4` constant updated from `3` → `4` to align with `16 §9.1`'s ratified value; `semstrait.plan.implicit_depth_max` feature toggle remains an off-by-default escape hatch unaffected by the constant. Q-COMP-001 owns the canonical depth-bound decision; this entry is `34`'s sibling restatement and tracks any post-v1 reconsideration through the same review trigger (`34` drafting + early-usage telemetry; raising to `6` is MINOR if `PLAN_E_0502 CompositionDepthExceeded` fires on legitimate models). Round-1 framing retained for historical reference.

**Question.** `34 §10.4` sets the implicit-composition depth bound at 3 hops. Is 3 the right default? (See also `16` Q-COMP-001.)

**Refs.**

- `34 §10.4` — current constant.
- `16 §9.1` — "depth-limited" rationale.
- `16` Q-COMP-001 — sibling question in the composition doc.
- `19 §3.4` — compile-time cross-kind path resolution (same bound).

**Arguments pro 3.**

- Covers 95%+ of realistic star-schema / snowflake / hub-and-spoke models where field-first resolution is ergonomic.
- Keeps the Steiner-tree search tractable (worst-case `E^3`).
- Authors who need deeper paths declare an explicit Joinset (`24`) — cleaner intent.

**Arguments pro higher (e.g. 5).**

- Complex healthcare / pharma / supply-chain models have deep chains.
- A tighter bound forces Joinset declarations that may not match authorial intent.

**Current position.** 4 hops (mirrored from Q-COMP-001). `semstrait.plan.implicit_depth_max` feature toggle remains off-by-default.

---

## Q-PLAN-003 — `PLAN_E_0500` allocation conflict  *[Closed — superseded by typed-kind transition]*

**Status.** Closed. The eleventh-pass retirement of the stable string-code subsystem at `30 §6` (2026-04-29) makes the allocation conflict moot. `ConstraintViolation` and `AmbiguousImplicitComposition` no longer share a numeric identifier — they are distinct typed variants on `PlanErrorKind` (per `34 §13`'s rewritten error roster), each identified by enum-variant identity. The `[TD-PLAN-E-0500-REALLOC]` tech-debt item retires alongside the string-code surface.

**Original framing (preserved).** `PLAN_E_0500` was referenced by two distinct error conditions:

- `ConstraintViolation` per `11 §8.7` (step-0 constraint validation).
- `AmbiguousImplicitComposition` per `16 §14.3` (step-2 field-first resolution).

Both could not share the same stable code; the open question was which one moves. Proposal A (move `AmbiguousImplicitComposition` to `PLAN_E_0506`) was the Round-1 default, with Proposal B (relocate `ConstraintViolation` to `PLAN_E_0580`) as the alternative. `34 §13.1` flagged this as a pre-release blocker per `30 §6.2`.

**Resolution.** With typed-kind discipline (`30 §5`, eleventh pass), `PlanErrorKind::ConstraintViolation` and `PlanErrorKind::AmbiguousImplicitComposition` are independent enum variants; no string allocation is involved. Both `11 §8.7` and `16 §14.3` reference variants by typed identity, not by code. The conflict cannot recur. `34 §13` no longer carries a `PLAN_E_05xx` allocation table; the prior placeholder language is gone.
