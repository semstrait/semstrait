---
doc: design/questions/open/16_questions
status: Living
purpose: Open questions surfaced while drafting `foundations/16_composition.md`
depends-on:
  - foundations/16_composition.md
  - foundations/11_names_and_scopes.md
  - foundations/12_nesting_policy.md
  - foundations/19_expression_flow.md
  - foundations/15_mapping_and_binding.md
  - foundations/17_temporal_shape.md
  - data-kinds/23_joinset.md
  - apis/32_semstrait_model.md
  - apis/33_semstrait_manifest.md
  - apis/34_semstrait_planner.md
---

# Open Questions — `foundations/16_composition.md`

> **Status (manifest ratification cascade, 2026-05-28).** Five framework-level questions remain open after the manifest ratification cascade: Q-COMP-006 (post-rebase Unionset-retirement structural cleanup of §9.3 / §10.5 / §13 + downstream cross-refs), Q-COMP-009 (composite-key shape alternatives), Q-COMP-014 (PLAN_E_0505 candidate suggestions), Q-COMP-015 (FieldOwnership::Derived distinctness), Q-COMP-016 (ManyToMany reject-by-default — narrowed by SR-E-14 in Round 3). Closed 2026-05-28 with manifest cascade: Q-COMP-010 (CompositionCoverage serialization shape — settled by CCK.3 tuple-keyed ratification). Earlier 2026-05-12 closures (item K): Q-COMP-007 / Q-COMP-008 / Q-COMP-017. Closed items moved to [`../closed/16_questions.md`](../closed/16_questions.md). None of the items below block the headline ratifications in `16 §16` (note: §16 itself may need re-numbering when Q-COMP-006 lands).

---

## Q-COMP-006 — Post-rebase Unionset-retirement structural cleanup

**Question.** After `CompositionKind` shrank to `{Joinset, Grainset}` (Unionset uses its own bare `SemanticInterface` per `23 §3.2`), should the now-inert pre-rebase Unionset material in `16 §9.3` / `§10.5` / `§13` (+ `CompositionKind::Unionset` cross-refs in §10 / §11 / §14) be deleted, banner-tagged "RETIRED", or restructured into a generic implicit-composition enumeration framework?
**Why.** Vestigial enumeration sketches no longer feed `ComposedSemanticInterface`; V1 enumeration lives variant-internal (`21 §3.2`, `22 §3.3`).
**Current position.** `16 §5` ratification block flags V1 scope; deeper §9.3/§10.5/§13 text retained as-is (banner-equivalent, lightweight).
**Trigger.** Resolve before Round-4 `16` framework cleanup; align with whether `34` Strategy prefers per-variant or framework-level enumeration. (`§16` may need re-numbering when this lands.)

---

## Q-COMP-009 — Composite keys: positional pairing vs named pairing

**Question.** `16 §2.3` ratifies composite keys as multiple positional `KeyPair` entries (ordering significant); should it instead be a single `KeyPair` whose `left` / `right` are `Vec<SemanticsName>` (symmetric, order-free)?
**Why.** Positional list relies on ordering; single-entry makes the composite cardinality explicit.
**Current position.** Positional pairs (one `KeyPair` per column) — matches FK declaration style, natural YAML surface in `32`.
**Trigger.** Revisit at `32` YAML drafting if the shape proves unergonomic; migration would be MAJOR.

---

## Q-COMP-010 — `CompositionCoverage` — serialize per-constituent or collapsed? — CLOSED (2026-05-28)

> **Moved to [`../closed/16_questions.md`](../closed/16_questions.md).** Settled by manifest ratification clause CCK.3 (Manifest Ratification Log, 2026-05-28): per-constituent local masks live on the variant struct (not top-level DataKind); symmetric with `CompositionCoverage` keyed by `(ConstituentRef, SemanticsName)` — tuple-keyed shape confirmed.

---

## Q-COMP-014 — Should `PLAN_E_0505 AmbiguousCompositionReference` include suggested qualifications?

**Question.** `16 §14.3` `PLAN_E_0505` carries `(name, candidates)`; should the Diagnostic also surface the exact qualified-name forms the author can use (e.g. `orders.total`, `returns.total`)?
**Why.** Classic "did you mean X?" UX; `30 §5.3` ratifies `ContextLine` for exactly this.
**Current position.** Yes — `context: Vec<ContextLine>` carries one `ContextLine` per candidate qualification with a "use this form" label.
**Trigger.** Effectively settled (implementation detail, no doc change pending); reopen only if `ContextLine` shape changes.

---

## Q-COMP-015 — `FieldOwnership::Derived`: how common, and does it warrant its own variant?

**Question.** Should `16 §7.3.4`'s `FieldOwnership::Derived(PhysicalExpr)` (fields existing only on the composed surface) stay a dedicated variant, or fold into `Native` via a synthetic composition `DataKindRef`?
**Why.** `Derived` carries a `PhysicalExpr` payload that has no home on `Native` and only makes sense in the composed scope.
**Current position.** Distinct variant — lets the planner skip constituent scans for `Derived` fields.
**Trigger.** Revisit at `34` if planner dispatch on `FieldOwnership` rarely branches on `Derived`.

---

## Q-COMP-016 — `Cardinality::ManyToMany` — permit or reject by default?

**Question.** `16 §3.3.4` permits `ManyToMany`; should v1 reject it outright and force junction-table modeling?
**Why.** Some legitimate Models need genuine many-to-many (e.g. tags/articles with no modeled junction), but it raises fanout-correctness surprises.
**Current position.** Permit, narrowed in Round 3 (item K): directional `cross_filter ∈ {Left, Right}` rejected on `ManyToMany` (SR-E-14), authors MUST declare `Both`/`None`; `optional` required (SR-E-13).
**Trigger.** Revisit during a model-discipline pass if real-world models reveal pervasive misuse.

---

