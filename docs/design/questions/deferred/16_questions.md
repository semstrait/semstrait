---
doc: design/questions/deferred/16_questions
status: Deferred
purpose: Composition-side questions parked for post-v1 ratification
---

# Deferred Questions — `foundations/16_composition.md`

> Items deferred to v2 (or later) ratification. Live items are in [`../open/16_questions.md`](../open/16_questions.md); closed items in [`../closed/16_questions.md`](../closed/16_questions.md).

---

## Q-COMP-D01 — Composition-side cycle detection — DEFERRED

**Status: DEFERRED.** Sibling of Q-MAN-D03 ([`33_questions.md`](33_questions.md)) at the composition layer. The 2026-05-28 manifest ratification cascade ratified validation gates G1–G5 in C13 but parked G6 (cycle detection on the relationship graph). At the composition layer, the question of cycle detection during implicit-composition enumeration (`16 §10`) is similarly deferred — implicit Steiner-tree enumeration over the relationship graph already prunes deeper-than-needed walks via `MAX_IMPLICIT_COMPOSITION_DEPTH` (Q-COMP-001 closed at depth 4), so the "cycle creates infinite walk" hazard is bounded; nonetheless, an explicit cycle-detection pass at composition-build time is parked for v2.

**Refs.**

- See sibling: [`33_questions.md`](33_questions.md) Q-MAN-D03 — manifest-side governance.
- `_research/manifest/RATIFICATION_LOG.md` — C13 G6 (deferred).
- `16 §10.4` — implicit-composition Steiner-tree enumeration.
- `closed/16_questions.md` Q-COMP-001 — `MAX_IMPLICIT_COMPOSITION_DEPTH = 4`.
- `closed/16_questions.md` Q-COMP-006 (override) — transparent unfolding through composed surfaces.

**Open axes.**

- **Detection scope.** Among explicit `Relationship` declarations only, or extended through implicit-composition synthesis.
- **Cycle definition.** Undirected (any back-edge in the unfolded graph) vs directed-under-`cross_filter`.
- **Diagnostic shape.** `COMP_E_...-RELATIONSHIP-CYCLE` reporting traversal-order nodes.

**Next step.** Reactivate when Q-MAN-D03 (manifest-side G6) reactivates; the two should land together for layering consistency.

---

## Q-COMP-D02 — Phase 2 Target D: Hop-depth caps on implicit composition — DEFERRED

**Status: DEFERRED — Phase 2 research candidate.** Targets C10 (per-Joinset hop-depth limit). The 2026-05-28 cascade ratified the per-Joinset hop-coverage shape (C7) and confirms the existing `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` global cap (Q-COMP-001 closed). The Phase 2 question is whether v2 should refine the cap from a global constant to a **per-Joinset** or **per-anchor-DataKind** value, recognizing that some legitimate Models genuinely need deeper walks for specific anchors (e.g., hierarchical org-chart traversal anchored on the leaf-employee kind) while others should stay shallow.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C7 (per-Joinset hop coverage); C10 (hop-depth governance, deferred).
- `closed/16_questions.md` Q-COMP-001 — global cap `4` ratified for v1.
- `closed/34_questions.md` Q-PLAN-008 — sibling restatement of the global cap.
- `16 §9.1` — depth-limit boundary.

**Open axes (Phase 2 scope).**

- **Per-anchor configurability.** A YAML field on `DataKind` (e.g. `implicit_depth_limit: 6`) overriding the global cap for that anchor.
- **Per-Joinset override.** An explicit Joinset declaration could declare a deeper walk on its own constituent set without raising the global cap.
- **Telemetry-driven.** Profile real Models, raise the global cap to the 95th percentile observed depth.
- **Algorithmic implications.** Per-anchor caps change the Steiner-tree search shape; brute-force enumeration (Q-COMP-003 closed) stays viable but the constant factor shifts.

**Next step.** Phase 2 research dossier; output feeds back into `16 §9.1` cap discipline and `34 §10.4` planner constant.
