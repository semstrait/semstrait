---
doc: design/questions/open/34_questions
status: Living
purpose: Round-1 open items raised against `apis/34_semstrait_planner.md`
---

# 34 — Open Questions

Unresolved items arising while drafting `docs/design/apis/34_semstrait_planner.md`. Each entry restates the question, lists the relevant ratified references, and proposes a lean next step so a later decision pass can resolve without re-reading the whole doc.

> Eight questions remain open: Q-PLAN-001, -002, -004 through -007, Q-PLAN-009, Q-PLAN-010. Closed items moved to [`../closed/34_questions.md`](../closed/34_questions.md).

---

## Q-PLAN-001 — `Request.from` shape: `DataKindRef` scalar vs `DataKindPath`

**Question.** Should `Request.from: Option<DataKindRef>` (`34 §3.1`/§3.8, a single top-level DataKind name) widen to a `DataKindPath` so a Request can target an interior arm of a nested Complex?
**Why.** A flat `DataKindName` cannot pin a nested branch (e.g. `23 §3` Unionset-of-Unionsets).
**Default.** Scalar `DataKindRef` per `34 §3.1`.
**Reopen.** If users need interior targeting, add a sibling `Request.from_path` (MINOR per `30 §2.2`) rather than widening `from`.

---

## Q-PLAN-002 — `Strategy` trait openness: sealed vs non-sealed

**Question.** Should the `Strategy` trait (`34 §8.1`) be sealed (crate-private impls) or open to third-party implementers?
**Why.** Sealing collapses the strategy roster to the four DataKinds and lets `dispatch_strategy` stay a closed match; openness lets adapters add plan shapes for novel `Complex` variants (I10).
**Default.** Non-sealed per `34 §8.1`/§15.1, gated by `Q-KIND-001` in [`20_questions.md`](20_questions.md) (`ResolvedDataKind` openness).
**Reopen.** Revisit jointly with Q-KIND-001 before v1.0; follow whatever `ResolvedDataKind` does — seal both or keep both open.

---

## Q-PLAN-003 — `PLAN_E_0500` allocation conflict — CLOSED

> **Moved to [`../closed/34_questions.md`](../closed/34_questions.md).** The typed-kind transition at `30 §6` retired the stable string-code subsystem; `PlanErrorKind::ConstraintViolation` and `PlanErrorKind::AmbiguousImplicitComposition` are now distinct typed variants with no numeric identifier collision.

---

## Q-PLAN-004 — `TemporalRequest` vocabulary: expose now vs defer

**Question.** Should `Request.temporal: TemporalRequest` (`34 §3.9`) be visible in v1, or stay a placeholder, given planner consumption is DEFERRED per `17 §10`?
**Why.** Exposing lets `semstrait-api` populate forward-compatibly; but a type the planner ignores is a foot-gun.
**Default.** Exposed as a reserved shape with advisory `PLAN_W_1702` on any populated consumption attempt.
**Reopen.** If `semstrait-api` populates it before `17`'s planner milestone, escalate the advisory to a hard error — `[TD-TEMPORAL-GATE]`.

---

## Q-PLAN-005 — `SessionContext.feature_toggles` typing: free-form vs typed catalog

**Question.** Should `feature_toggles: BTreeMap<String, FeatureToggleValue>` (`34 §4.1`) be a closed/open enum (typed catalog) instead of a free-form map?
**Why.** Free-form lets third-party passes register toggles without a crate bump; a typed catalog turns typos into compile errors and documents semantics in one place.
**Default.** Free-form map; the `semstrait.*` prefix is reserved for built-in toggles.
**Reopen.** If the toggle surface exceeds ~10 stable entries, add a typed read-through view alongside (not replacing) the map.

---

## Q-PLAN-006 — `OptimizerPass` idempotence: proof obligation vs convention

**Question.** Should `OptimizerPass` (`34 §14.5`) ratify idempotence as a proof obligation (e.g. `verify_idempotent`) or is convention enough?
**Why.** A proof obligation makes re-optimize safe by construction; but idempotence is not statically checkable in Rust, so any check is test-only or a runtime pass-cost.
**Default.** Convention only (matches the `34 §12.3` soundness posture).
**Reopen.** Add an integration test applying the default pass chain twice and asserting byte-identical output; keep convention for third-party passes.

---

## Q-PLAN-007 — `ResolvedQueryRequest` visibility: `pub` vs `pub(crate)`

**Question.** Should `ResolvedQueryRequest` (`34 §5`) be `pub(crate)` instead of `pub`, given its only consumer is `Strategy::plan`?
**Why.** `pub(crate)` shrinks the public surface and prevents callers building malformed values; but Q-PLAN-002's open-trait posture requires its parameter types to be `pub`.
**Default.** `pub`, with `pub(crate)` constructors (§5.1).
**Reopen.** Resolve jointly with Q-PLAN-002 — if `Strategy` is sealed, move to `pub(crate)`; if open, keep `pub`.

---

## Q-PLAN-008 — Field-first depth bound — CLOSED (2026-04-28)

> **Moved to [`../closed/34_questions.md`](../closed/34_questions.md).** Mirrored from Q-COMP-001 at value `4`.

---

## Q-PLAN-009 — `StrategyRegistry` construction vs. mutation

**Question.** Should `StrategyRegistry` (`34 §8.3`) stay construction-time-only, or become mutable to support live A/B-testing of a replacement strategy?
**Why.** Construction-only keeps the plan a pure function of registry construction (I4 determinism) and simplifies thread-safety; mutability enables A/B testing without a server restart.
**Default.** Construction-only.
**Reopen.** For A/B testing, spin up a second registry and route a traffic fraction at the API layer — the registry is cheap to construct.

---

## Q-PLAN-010 — `PlanError` per-variant wrapper vs flat enum

**Question.** Should `PlanError` (`34 §13.1`) keep per-variant wrappers for `21`–`24` (`Simple(SimpleError)`, `Grainset(GrainsetError)`, etc.) or flatten them into one enum?
**Why.** Wrappers keep each DataKind's error taxonomy self-contained (plan-stage rosters `21 §9` / `22 §9` / `23 §9` / `24 §11`) so a new variant does not touch `34`; flat lets pattern-match users see all errors in one enum.
**Default.** Wrapper per variant.
**Reopen.** Revisit if consumers complain about match-nesting; a `PlanError::all_codes()` iterator can paper over the cost.

---

## Cross-doc fixes flagged while drafting `34`

Tracked out of band; none block `34`'s ratification but should be resolved before v1.0.

| ID | Location | Fix |
|---|---|---|
| CDF-30-02 | `30 §6.2` `PLAN_E` row | Spell out the `PLAN_E_05xx` sub-bands explicitly (`0500` constraint, `0501`–`0509` composition, `0510`–`0519` request-shape, `0520`–`0529` filter-shape) once Q-PLAN-003 resolves. |
| CDF-21-01 | `21 §7` | Cross-reference `34 §13` as the aggregation surface (`PlanError::Simple(SimpleError)` wraps `PLAN_E_21xx`). |
| CDF-22-01 | `22 §8` | Same as CDF-21-01 for `PLAN_E_22xx` and `PlanError::Grainset(GrainsetError)`. |
| CDF-23-01 | `23 §9` | Same as CDF-21-01 for `PLAN_E_23xx` and `PlanError::Unionset(UnionsetError)`. |
| CDF-24-01 | `24 §11` | Same as CDF-21-01 for `PLAN_E_24xx` and `PlanError::Joinset(JoinsetError)`. |

## Deferred / known-gap migration items

Tracked as deferred migration items (not open questions):

- `[TD-PLANNER-SHAPE]` — rename `SemanticPlanner` struct → free `plan` / `optimize` functions at the crate root (per `34 §6` / §11).
- `[TD-SESSION-CONTEXT]` — legacy `SessionVariables: HashMap<String, String>` → typed `SessionContext` (per `34 §4`).
- `[TD-REQUEST-SHAPE]` — legacy `ResolvedQueryRequest` folds Request + SessionContext + partial resolution; v1 splits into `Request` (caller surface) + `ResolvedQueryRequest` (internal post-lookup form) per `34 §3` / §5.
- `[TD-ADHOC-INTO-FIELD-FIRST]` — legacy `AdHocJoin` dispatch path subsumed by field-first resolution (per `34 §9.5`).
- `[TD-PLANNER-NO-CATALOG]` — drop the optional `SemanticPlanner.catalog` field once ad-hoc-join paths migrate to field-first resolution (per `34 §16.4`).
- `[TD-CONSTRAINT-ERROR-FANOUT]` — v1 `ConstraintValidator` short-circuits on first violation; future work may accumulate multiple violations in one pass (per `34 §14.2`, originally flagged in `11 §8.7`).
