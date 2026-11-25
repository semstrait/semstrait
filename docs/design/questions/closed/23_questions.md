---
doc: design/questions/closed/23_questions
status: Closed
purpose: Resolved questions originally raised against `data-kinds/23_unionset.md`
---

# Closed Questions — `data-kinds/23_unionset.md`

> Historical record of ratified Unionset decisions. Live items are in [`../open/23_questions.md`](../open/23_questions.md).

---

## Q-UNI-001 — Error-code allocation: `*_E_23NN` per doc vs `30 §6.2` cross-subsystem ranges

**CLOSED (structure-optimization pass, 2026-05-03).** Superseded by typed-kind diagnostic discipline ratified in `30` and cascaded through `31`-`39`.

**Resolution.** Numeric code-range reconciliation is retained as historical context only. Active v1 diagnostic identity is stage-owned typed `*ErrorKind` variant identity.

---

## Q-UNI-002 — `UnionMode::Distinct` in v1 or deferred?

**CLOSED (Phase-3 cascade, 2026-04-17; re-confirmed at C6 ratification, 2026-05-03).** Ratified via `foundations/18_entities.md §2` (adjacency reference) and re-confirmed in the post-thirteenth-pass cascade rebase: v1 `UnionMode` roster is **`{All, Unique}`**, `#[non_exhaustive]`, default `All`. The previous spelling `Distinct` is renamed to `Unique` (authors who need SQL-`DISTINCT` semantics at Union-level select `Unique`; the three-valued-logic NULL-collision caveat documented in `23 §4.3` still applies as `PLAN_W_2303`). The "defer Unique to post-v1" alternative is rejected; `Unique` is supported in v1.

**Question.** `23 §2.1` ratifies a `UnionMode` enum with two variants: `All` (default) and `Distinct`. Distinct semantics in the presence of NULL-fill are subtle (per `23 §4.3`'s three-valued-logic note): rows from different children differing only in NULL-fill positions do NOT dedupe under `DISTINCT`. Should Round 1 ship `Distinct` at all, or defer it to a later milestone?

**Refs.**
- `23 §2.1, §4.1, §4.3` — Round-1: `UnionMode::Distinct` supported, with advisory `PLAN_W_2303` for the NULL × NULL non-collision case.
- `UNIONSET.md` (legacy) — distinguishes `UNION ALL` vs. `UNION DISTINCT`.
- Cube.js `unionAll` — does not expose a DISTINCT mode at the cube level.
- dbt MetricFlow — does not have a direct Unionset analog.
- `[TD-UNIONSET-DISTINCT-SEMANTICS]` — deferred subsection under §4.3.

**Current position.** Both modes supported in v1; `UnionMode::All` is the default. Body retained as historical resolution context.

---

## Q-UNI-003 — Per-child Coverage override shape: whitelist `provides` vs per-Semantics tri-variant

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** V1 carries **no authored per-child Coverage override**. Coverage is **inference-only**, folded once at compile from each child's interface plus Binding-level Coverage (for Simple children) or the child's resolved interface (for Complex children) per `23 §3.2`. The `ChildCoverageOverride { provides }` Rust shape, the YAML `coverage:` block on child entries, and the per-(child, name) override mechanism are retired pre-cascade artifacts not authored by the user. The whitelist-vs-tri-variant axis is moot until v2 reopens explicit overrides; tracked as `[TD-UNIONSET-COVERAGE-OVERRIDE]` for future re-evaluation.

**Question.** `23 §3.2` ratifies `ChildCoverageOverride { provides: BTreeSet<SemanticsName> }` — an opt-in whitelist. Any name NOT in `provides` that the Binding-level fold would cover is forced to `FieldOwnership::NullFill` at the composition level. Is this the right shape, or should authors be able to distinguish per-Semantics between `Native` / `Derived` / `NullFill` overrides?

**Current position.** No override mechanism in v1. Body retained as historical context.

---

## Q-UNI-005 — Strict-mode posture for `TemporalShape`-mismatch advisories

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Moot under V1 **strict equivalence rule** (`23 §5.2`). Children's `TemporalShape` (kind including SCD subtype + grain) MUST be equivalent; mismatch is `COMP_E_2301 UnionsetChildShapeMismatch` — a single consolidated hard-error code per C1 ratification (2026-05-03). No advisory-vs-error toggle needed; the rule is always strict in V1. Future smart alignment of non-equivalent shapes (e.g. `Scd + Snapshot` projecting `Snapshot` as a degenerate `Scd(Type1)`) is post-v1 and tracked as `[TD-UNIONSET-SHAPE-ALIGN]`.

**Question.** `23 §6.1` lists a matrix of cross-child `TemporalShape` combinations and emits warnings (`COMP_W_2302`–`2305`) for every mismatch. Should there be a strict-mode flag (e.g. `--strict-unionset-shapes`) that promotes these warnings to errors?

**Current position.** Strict by construction in V1; advisories survive only for multi-as-of `Snapshot` (`COMP_W_2302` / `COMP_W_2303` per C2). Body retained as historical context.

---

## Q-UNI-007 — Interaction with `17`'s as-of / snapshot-selection when children have heterogeneous shapes

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Moot under V1 strict equivalence rule (`23 §5.2`). Children **cannot** have heterogeneous shapes in V1 — `Scd(Type2)` + `Timeseries` is `COMP_E_2301`. As-of routing in mixed-shape Unionsets is post-V1; tracked as `[TD-UNIONSET-SHAPE-PLANNING]`. Multi-as-of `Snapshot` advisories (`COMP_W_2302` / `COMP_W_2303` per C2 ratification) cover the only V1-legal heterogeneity (two `Snapshot` children with equivalent shape but different as-of timestamps).

**Question.** When a Unionset has an `SCD(Type2)` child and a `Timeseries` child, and a Request carries an as-of timestamp, how should the planner route the as-of filter to each child?

**Current position.** Mixed-shape Unionsets are not V1-legal. Body retained as historical context.

---

## Q-UNI-008 — Non-exhaustive `UnionMode`: future variants

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** V1 roster ratified at C6: **`UnionMode::{All, Unique}`**, `#[non_exhaustive]`, default `All`. Future variants tracked as `[TD-UNIONSET-FUTURE-MODES]` for MINOR additions per `30 §6.3`:

- `UnionMode::ByName` — name-keyed alignment rather than positional.
- `UnionMode::DistinctOnKeys(Vec<SemanticsName>)` — dedupe on a specific key subset rather than the full row.

The suggestion to lift `Intersect` / `Except` into separate `ComplexDataKind` variants (rather than overloading `UnionMode`) is carried forward as a backlog architectural consideration — these are set operations with substantively different planner semantics (NULL handling, every-row-must-exist-in-every-child) and likely deserve their own `Intersectset` / `Exceptset` variants.

**Question.** `UnionMode` is `#[non_exhaustive]` per I10. Which additional variants are plausibly in the MINOR-addition space?

**Current position.** `{All, Unique}` only; future variants deferred to backlog. Body retained as historical context.

---

## Q-UNI-009 — Single-child Unionset acceptance: reject (Round-1) vs silent accept vs warning

**CLOSED (Phase-3 cascade, 2026-04-17).** Ratified via `data-kinds/26_nesting_matrix.md §R3`: every `ComplexDataKind` (including `Unionset`) REQUIRES ≥ 2 children. Rejection stands (via R3 + `validate.complex-data-kind-insufficient-children` per `26 §2.3`); the "silent accept" alternative from `22 Q-GRN-006` (now also closed by R3) no longer creates asymmetry. All three Complex variants (`Unionset`, `Grainset`, `Joinset`) share the ≥ 2 children rule. `[TD-UNIONSET-SINGLE-CHILD]` is retired.

**Question.** `23 §8.1` fires `VALID_E_2302 UnionsetSingleChild` on a Unionset with exactly one child. The rationale: "semantically the child itself; authors should replace." Is rejection right, or should Round 1 be more permissive (silent accept or warning)?

**Current position.** Rejection (now structurally enforced via `26 §R3`). Body retained as historical resolution context.

---

## Q-UNI-010 — Composition-level `Derived` expressions: who declares them?

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Re-scoped under `CompositionKind` retirement. `CompositionKind` was a pre-cascade artifact not authored by the user; removed entirely from the v1 architecture. `ComposedSemanticInterface` survives as a **Joinset-only** per-hop join-path-search artifact (per `24`'s scope) — it has no role in Unionset or Grainset.

For Unionset, composition-level computed Semantics are authored directly on the Unionset's own `SemanticInterface` via Computed Dimensions / Metrics per `19 §3` — identical to Dataset's authoring surface (the `SemanticInterface` is a flat set of Dimensions / Measures / Metrics / Keys / Filters per `18 §1.1`). No `23`-specific declaration mechanism is needed; the question's premise (a separate `ComposedSemanticInterface` carrier on Unionset) does not hold in the post-thirteenth-pass architecture.

**Question.** When a composed-surface Semantics is `Derived` at the Unionset level (computed from other composed-surface fields, not from any child), where does the author declare the derivation?

**Current position.** On the Unionset's own `SemanticInterface` per `19 §3`, like any Computed Dimension / Metric. Body retained as historical context.

---

## Q-UNI-011 — `CompositionCoverage` override: override-before-fold or override-after-fold?

**CLOSED (post-thirteenth-pass cascade rebase, 2026-05-03).** Moot — collapsed via Q-UNI-003. V1 has no per-child Coverage override; the override-vs-fold ordering question evaporates. Coverage is **pure inference** per `23 §3.2` — folded from each child's interface plus Binding-level Coverage (Simple children) or the child's resolved interface (Complex children). When v2 reopens explicit overrides per `[TD-UNIONSET-COVERAGE-OVERRIDE]`, this question reopens with it.

**Question.** `23 §5.4` specifies the override composes with the fold: `override` acts as a post-fold mask. An alternative is that `override.provides` REPLACES the fold. Which semantic is right?

**Current position.** No override mechanism in v1. Body retained as historical context.

---
