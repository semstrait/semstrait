---
doc: design/questions/open/20_questions
status: Living
purpose: Parked unresolved questions surfaced while drafting `data-kinds/20_taxonomy.md`
depends-on:
  - data-kinds/20_taxonomy.md
  - foundations/10_resolution_pipeline.md
  - foundations/11_names_and_scopes.md
  - foundations/12_nesting_policy.md
  - foundations/15_mapping_and_binding.md
  - foundations/16_composition.md
  - foundations/17_temporal_shape.md
  - apis/30_api_contracts.md
  - apis/34_semstrait_planner.md
---

# Open Questions — `data-kinds/20_taxonomy.md`

> Items surfaced during Round-1 drafting of the DataKind-taxonomy doc. Each entry restates the question, lists its ratified references, documents the Round-1 default `20` currently uses, and names the expected resolution point. Entries migrate out of this file when `21`–`25`, `33`, or `34` make a decision that confirms or amends `20`'s default. None of these items block the four shared-invariant ratifications (D1–D9) in `20 §2`–`§5`.

---

## Q-KIND-001. `Strategy` trait openness — sealed vs open for third-party implementers

Should the public `Strategy` trait (`20 §5.2`) be sealed to `semstrait-planner`, or open for third-party crates to implement and register new variant strategies?

Why: the trait carries I4 determinism / I5 / I6 / I8; an open trait makes variant-to-strategy binding runtime rather than compile-time-total.

Current default: sealed (strategies live in `semstrait-planner`; new variants extend in-tree); non-blocking — either choice ratifiable without amending `20`.

Expected resolution point: `34 §`* (public `Strategy` trait surface).

---

## Q-KIND-002. Subsystem-prefix allocation within the `2000`–`2099` range

> **Moved to `[../closed/20_questions.md](../closed/20_questions.md#q-kind-002-subsystem-prefix-allocation-within-the-20002099-range)`.** Superseded by typed-kind diagnostic discipline; numeric subsystem-prefix governance is not a v1-blocking decision surface.

---

## Q-KIND-003. Interface exposure for a single-DataKind Request — bare vs degenerate-composed at planner entry

Should a `SimpleStrategy` receive the bare `SemanticInterface` (literal D5), a degenerate `ComposedSemanticInterface`, or a shared trait view, given D5 (`§4.4`) maps Simple→bare and Complex→composed?

Why: the planner dispatch site (`§5.3`) is only uniform if every strategy consumes one interface type; otherwise it branches on interface type, not just variant tag.

Current default: hybrid via the shared `SemanticsView` trait already ratified in `16 §5.1` (no allocation, no branching inside strategies; D5 preserved); non-blocking — `20 §5.2` accesses the interface via `ctx`/`manifest`, not directly.

Expected resolution point: `34 §`* (when `PlannerCtx` field shapes are pinned) and `22` / `24` (concrete interface shape their strategies consume).

---

## Q-KIND-004. Shared-vs-per-variant partition of structural Preconditions

How should `20 §6.2` partition structural `validate`-stage Preconditions with per-variant exceptions (e.g. "≥2 children" for Unionset/Grainset only, structural-label uniqueness for Complex only) between `20` and `22`–`24`?

Why: a strict "shared-only-if-all-four" split duplicates parameter-only-differing diagnostic text across `22`/`23`/`24`.

Current default: hybrid — shared diagnostic code in `20 §8.2`, per-variant rule text in `22`/`23`/`24`; add `VALID_E_2006 InsufficientChildCount` to `20 §8.2`, keep `VALID_E_2002 InterfaceTypeMismatch` (all-four trait check) and `VALID_E_2004 StructuralLabelCollision` (shared-Complex) in `20`. Non-blocking; current option-A roster is already correct.

Expected resolution point: first draft of `22` or `23` — whichever ratifies "≥2 children" first decides whether `VALID_E_2006` moves out of reserved.

---

## Not Opened as Questions

Items considered during drafting but not elevated to open-question status. Each already has a ratified answer cited inline in `20`:

- **Two-level `DataKind` sum type structure.** Ratified in `§2.1`. `00 §4.1` already pairs `SimpleDataKind` / `ComplexDataKind` explicitly; `20` just mirrors the vocabulary.
- `**Binding` uniqueness on Simple.** Ratified in `15 §2.1` and restated as Invariant D1.
- **Same-variant self-nesting ban.** Ratified in `12 §2.1`.
- **Nested-kind scope has no interface.** Ratified in `11 §2` / `§10`.
- `**ComposedSemanticInterface` is a distinct type from `SemanticInterface`.** Ratified in `16 §5.1`.
- `**Joinset` is the explicit composition mechanism; implicit composition is planner-synthesized.** Ratified in `16 §9` / `§10`.
- **Strategies are per-variant, not per-aspect or per-something-else.** Ratified in `§5.1` with Invariant D9.

---

**End of `20_questions.md`.**