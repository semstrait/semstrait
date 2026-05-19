---
doc: design/questions/closed/34_questions
status: Closed
purpose: Resolved questions originally raised against `apis/34_semstrait_planner.md`
---

# Closed Questions — `apis/34_semstrait_planner.md`

> Historical record of ratified planner decisions. Live items are in [`../open/34_questions.md`](../open/34_questions.md).

---

## Q-RAW-FILTER — Engine unification for named and raw filters — CLOSED (2026-05-19)

**Status: CLOSED.** Ratified that named filters (`DataKindFilter` activations) and raw filters (inline operator/value triples) share the same predicate-construction engine at the `SemanticExpr` level. No canonical type changes required in `34 §3`, `§3.5`, or `§5`; the spec already accommodates the design. Closure is paper-trail for the implementation pass that lifted the `RawFiltersNotImplemented` stub in `semstrait-api` and unified the planner-side lowering path in `semstrait-planner`.

**Question.** Two implementation gaps coexisted: (1) `semstrait-api::RequestParser::parse` rejected any `raw_filters` with `RawFiltersNotImplemented`; (2) `semstrait-planner::query_filter_to_expr` lowered user filters via `Expr::column(field)` directly, bypassing `ExprResolver` — diverging from the `phys_resolver.resolve_expr` path used by `DataKindFilter` and `AggregationFilter` bodies. Should raw filters share the engine with named filters, and if so at what layer?

**Refs.**

- `34 §3` — `Request` shape (`filters`, `raw_filters` already two-field at API surface).
- `34 §3.5` — `Filter { field, operator, values }` — the canonical raw-filter shape; matches `RequestParser`'s output element.
- `34 §5` — `ResolvedQueryRequest.filters: Vec<ResolvedFilter>` — single unified landing site.
- `19 §7.1` — Phase A → Phase B filter placement.
- `14_expressions.md` — `SemanticExpr` / `EntityRef` as the U-1 unification surface.

**Ratification.**

- **(a) Raw-filter shape unchanged.** Raw filter = the existing `34 §3.5 Filter { field, operator, values }` struct. No new type.
- **(b) Request shape unchanged.** `34 §3.1 Request` and `RawQueryRequest` keep their two-field API surface: `filters: Vec<String>` for named activations, `raw_filters: Vec<RawFilter>` for inline triples.
- **(c) Single landing site.** Both lower into `ResolvedQueryRequest.filters` (`34 §5`). For raw filters, the API parser lowers `{ field, operator: String, value: JSON }` → `QueryFilter { field, operator: FilterOperator, values: Vec<FilterValue> }`. Named-filter activation continues to be validated for existence at the parser; application remains via `iface.filters` at scan level (no behavioural change in this pass).
- **(d) Engine unification at `SemanticExpr` (U-1).** Both forms produce a `SemanticExpr` predicate that walks `ExprResolver::resolve_expr`. Field references are authored as `EntityRef`, NOT `Column` — the resolver translates `EntityRef → Column` against the appropriate `ResolvedColumnMapping.physical` (scan-level for named bodies; identity for user-filter root placement). Same downstream pipeline both sides (Phase A resolve → per-Binding `PhysicalExpr` → Phase B placement).
- **(e) Cross-reference invariant.** `raw_filters[i].field` MUST NOT name a `DataKindFilter` declared on the kind; that's what `filters: Vec<String>` is for. Violations surface as `ParseError::RawFilterNamesNamedFilter { entity, name }` at parse time. Keeps the two API surfaces clean.
- **(f) `In` / `NotIn` v1 lowering.** OR-chain / AND-chain of equalities over the same `EntityRef`. Avoids introducing a new `14a` registry entry. Future canonical multi-arity tracked under `[TD-RAW-IN-LOWERING]`.

**Explicit out-of-scope (deferred).**

- Retiring the `Filter.field`-overloads-to-`DataKindFilter`-name overload mentioned in `34 §3.3` prose. Remains valid for direct canonical callers; the API path simply doesn't use it.
- Pinning the `ResolvedFilter` struct shape in `34 §5`. Implementation-internal shape (`QueryFilter`) is sufficient.
- Re-positioning user filters in the plan tree (currently post-rename root; named filter bodies remain at scan level). Layer choice is independent of the engine-unification ratified here.
- Opt-in vs unconditional application of `iface.filters` per kind. Tracked separately under the spec/legacy reconciliation work on `11 §6.4.1`.

**Implementation pointers.**

- `crates/semstrait-api/src/parse.rs::RequestParser::to_resolved` — JSON-string-op + JSON-value lowering for `RawFilter`; cross-reference rejection.
- `crates/semstrait-planner/src/planner.rs::query_filter_to_semantic_expr` — `EntityRef`-based `SemanticExpr` construction; routed through `ExprResolver::resolve_expr` in `inject_user_filters`.

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
