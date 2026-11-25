---
doc: design/questions/closed/22_questions
status: Closed
purpose: Resolved questions originally raised against `data-kinds/22_grainset.md`
---

# Closed Questions — `data-kinds/22_grainset.md`

> Historical record of ratified Grainset decisions. Live items are in [`../open/22_questions.md`](../open/22_questions.md); deferred items in [`../deferred/22_questions.md`](../deferred/22_questions.md).

---

## Q-GRN-001 — Inheritance default for child `grain`: finest vs declared

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Moot under SR-E-8 (`18 §3.4`): every Grainset child MUST author `extras.temporal.grain:` explicitly. There is no inheritance mechanism in V1 — the question of "finest vs declared" never arises because both options require an inference pathway, and SR-E-8 forbids inference from the parent. Validated post-parse via `validate.grainset-child-grain-required` per `26 §6`.

The pre-rebase `22 §3.2` finest-grain inheritance rule is retired; `VALID_E_2204 GrainsetChildGrainUnresolvable` is retired. The mandatory-explicit policy is the V1 ratification.

**Question.** When a `GrainsetChild` omits `grain:` in YAML, `22 §3.2` inherits from the child's own temporal-Dimension `grains:` list and picks **the finest** grain. Is "finest" the right default, or should inheritance require an explicit pick (e.g. "the first entry in the list" / "error if more than one grain declared")?

**Refs.**

- `22 §3.2` — pre-rebase Round-1 default: finest. Retired.
- `13 §4.2` — `TemporalDimension { grains: Vec<Grain> }`; the `grains:` list is "which coarser grains this dimension can be rolled to."
- `11 §6` — Semantics-level declarations.
- `18 §3.4` (SR-E-8) — every Grainset child MUST author `extras.temporal.grain:` explicitly.
- `32 §4` — `LeafExtras { temporal: ..., grain: ... }`; cascade table forbids `grain` on `ComplexExtras`.

**Resolution rationale.** The author's surface is now strictly explicit: SR-E-8 makes per-child grain authoring mandatory, eliminating the inheritance / inference question. Authoring cost (one extra YAML line per child) is acceptable for the determinism gain.

---

## Q-GRN-002 — Cross-child partial coverage: error in v1, or split-and-delegate?

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Ratified at G-2 (2026-05-03): partial-coverage Requests across grains route through **cross-grain LEFT OUTER JOIN composition** mediated by a `ComposedSemanticInterface` (per `16 §5` / `22 §3.4`). The driver is the most-covering grain-eligible effective routing unit (declaration-order tie-break per G-2b); attached units are added in declaration order (G-2c) and equi-joined on shared `Key`s per `18 §2.5`. No shared `Key`s between two units the planner needs to join is a hard compile error (`COMP_E_2204 GrainsetCrossGrainKeysAbsent`, per G-2d).

> **Superseded ordering basis (id-first rework, STATUS item U / U.2, 2026-05-29).** The G-2b/G-2c tie-break basis changed from **declaration order** to **name order**: model child collections are `EntityId`-keyed and projected name-ordered (`32 §7`), so there is no author/declaration order to tie-break on. The authoritative `22 §4.2`/`§4.3` now reads "name-order tie-break." This historical record is retained as the G-2 ratification; only the ordering axis is superseded.

This supersedes Round-1's "error via `PLAN_E_2201 NoEligibleChild`". The pre-rebase `PLAN_E_2208 GrainsetPartialCoverageNotSupported` reservation is retired.

**Question.** When a Request names Semantics that no **single** child of a Grainset covers Natively/Derivedly — but the **union** of children's Coverage does — should the planner split the Request into per-child sub-Requests and combine the results, or report `PLAN_E_2201 NoEligibleChild` (the Round-1 default)?

**Refs.**

- `22 §3.4` — cross-grain `ComposedSemanticInterface` construction.
- `22 §4.3` — cross-grain JOIN delegation algorithm (driver + attached LEFT OUTER on Keys).
- `22 §8` — `COMP_E_2204` / `COMP_E_2205` / `COMP_E_2206` (Keys-related compile errors).
- `22 §9` — `PLAN_E_2202 GrainsetSemanticsNotCoverableByJoin` (residual plan-stage error when Keys are insufficient).
- `16 §5` — broadened `ComposedSemanticInterface` shape (V1 covers Joinset and Grainset; Unionset retired).
- `18 §2.5` — `Key` shape.
- G-2a / G-2b / G-2c / G-2d — 2026-05-03 ratifications (LEFT OUTER, most-covering driver with declaration-order tie-break, declaration-order attached, hard error on missing Keys).

**Resolution rationale.** The user's vision (G-2): "Cross grain combination is resolved through compatibility evaluation at compile time — we basically check how `ComposedSemanticInterface` can be built using keys specification (that will help us to build equi join). Driving side is everytime chosen by most covering 'data kind grain', less covering everytime attached, if keys binding is allowing that. ... most probably join is correct way." Confirmed via four sub-questions:

- **G-2a** — LEFT OUTER JOIN (preserves driver row set; attached contributes Measures via equi-join).
- **G-2b** — Most-covering driver with declaration-order tie-break (deterministic).
- **G-2c** — Attached units in declaration order (deterministic over greedy-by-coverage-delta).
- **G-2d** — Hard compile error on missing shared Keys (rather than runtime fallback or implicit relationship-walk).

---

## Q-GRN-003 — Cost function pluggability hook site

(Tracked in [`../deferred/22_questions.md`](../deferred/22_questions.md). Status unchanged in this rebase.)

---

## Q-GRN-004 — Grainset-of-Grainset nesting

**CLOSED (Phase-3 cascade, 2026-04-17).** Ratified via `data-kinds/26_nesting_matrix.md §R2`: no same-variant self-nesting at any depth. Grainset-of-Grainset is forbidden structurally (not just via `COMP_E_2207`). `[TD-GRAINSET-NESTED]` is retired — admitting nested Grainsets becomes a post-v1 matrix relaxation, not a Round-2 item.

**Question.** `22 §3.4` / `COMP_E_2207` currently forbids a `Grainset` as a child of another `Grainset` (`[TD-GRAINSET-NESTED]`). Should Round 2 admit nested Grainsets? What is the semantic?

**Refs.**

- `22 §3.4` — pre-rebase deferred; `COMP_E_2207` retired.
- `26 §R2` — structural type-level absence of `grainsets:` on `GrainsetBody`.
- `12 §2` — pre-rebase nesting matrix.

**Current position.** Forbidden structurally via `26 §R2`. Body retained as historical resolution context.

---

## Q-GRN-005 — Mixed-shape Grainsets: warning vs error

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Ratified via the V1 strict TemporalShape kind equivalence rule (`22 §5.2`): all children of a Grainset MUST have equal `TemporalShape.kind` (incl. SCD subtype). Mixed shapes is a hard compile error (`COMP_E_2201 GrainsetChildShapeKindMismatch`), not an advisory.

This supersedes Round-1's "warning" position. The pre-rebase `PLAN_W_2202 MixedShapeAdvisoryChildren` advisory code is retired.

**Question.** `22 §5` ratifies mixed `TemporalShape`s across children as a **warning** (`PLAN_W_2202 MixedShapeAdvisoryChildren`), not an error. Should Round 2 promote to error, or relax further (silent)?

**Refs.**

- `22 §5.2` — V1 strict equivalence rule.
- `22 §8` — `COMP_E_2201` (single consolidated code; covers kind mismatches and SCD subtype mismatches).
- `[TD-GRAINSET-SHAPE-MIX]` — smart alignment of non-equivalent shapes (e.g. `Events + Snapshot`); deferred post-V1.
- 2026-05-03 user ratification: "Unionset and Grainset — require symmetric definition for temporal shape type."

**Resolution rationale.** Promote-to-error is the V1 ratification. Smart alignment of mixed shapes (e.g. `Events + Snapshot` mixing) is a future-research item; until then, hard error is the safest stance.

---

## Q-GRN-006 — Single-child Grainset degeneracy: lint or accept?

**CLOSED (Phase-3 cascade, 2026-04-17).** Ratified via `data-kinds/26_nesting_matrix.md §R3`: every `ComplexDataKind` (including `Grainset`) REQUIRES ≥ 2 children. Single-child is now a structural rejection, not a silent-accept. This unifies the policy across `Unionset`, `Grainset`, and `Joinset` (all three require ≥ 2 children). `[TD-GRAINSET-SINGLE-CHILD]` and the parallel `[TD-UNIONSET-SINGLE-CHILD]` are retired.

**Question.** A Grainset with exactly one child is structurally valid per `22 §2` but semantically degenerate — it is a one-child wrapper that adds nothing over the underlying DataKind. Should Round 1 accept silently, emit a lint, or reject?

**Refs.**

- `26 §R3` — structural rejection at validate stage.
- Similar pattern: a Unionset with one branch, a Joinset with one member — same shape of question for `23` / `24` (also closed via `26 §R3`).

**Current position.** **CLOSED — single-child Grainset is rejected structurally** via `26 §R3`. Body retained as historical resolution context.
