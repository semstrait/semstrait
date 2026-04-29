# Constraints — Deferred Session Snapshot (CLOSED)

**Status**: **CLOSED — all four open ratification items (Q-R4.3a … Q-R4.3d) resolved as part of the 2026-04-27 categories+constraints expansion (tenth pass).** This file is retained as a historical record of the deferred-session snapshot that the resolutions resumed from. The resolutions and their consequences are baked into:

- [`../foundations/11_names_and_scopes.md §8`](../foundations/11_names_and_scopes.md) — full rewrite of the Constraint section against ratified Q-R4.3a … Q-R4.3d; carrier roster trimmed to `{Measure, Metric}`; explicit `constraints:` redefined as the refinement layer over category-implicit rules.
- [`../foundations/19_categories.md`](../foundations/19_categories.md) — the new category axis (`MeasureCategory` / `MetricCategory`) that subsumes most explicit-constraint scaffolding the deferred session was reaching for.
- [`STATUS.md`](../STATUS.md) §2 item E (now resolved) and §5 tenth-pass checkpoint.

**Resolution summary (per-Q-ID)**:

| Q-ID | Resolution | Lands in |
|---|---|---|
| **Q-R4.3a** (`aggregation:` semantics) | **Reading 1, narrowed** — `aggregation:` constrains downstream re-aggregation; default is implied by the carrier's category (`additivity = Non` ⇒ no further agg legal); the block is the override carrier when a category default is too strict / too loose. | `11 §8.4`, `19 §3.3` / §4.3 |
| **Q-R4.3b** (DSL spelling + rename) | Outer key **`aggregation`** (singular). Third dimensions key **`all_of`** (symmetric with `one_of` / `none_of`); rename of code's `all` ⇒ `[TD-CONSTRAINT-RENAME-ALL]`. Sub-blocks **AND-combined** when multiple are present. Type **`MeasureConstraints` → `Constraints`** (shared across Measure / Metric carriers); `[TD-CONSTRAINT-RENAME]` resolved. | `11 §8.2`, `11 §8.3` |
| **Q-R4.3c** (`constraints.filter:` inside Measure / Metric) | **(c1) — no sub-block.** Filter intent already lives in category bodies (`SimpleMetricBody.filter`, `RatioMetricBody.filter`) and in the carrier's `filters: Vec<AggregationFilter>` whitelist. | `11 §8.2`, `19 §4` |
| **Q-R4.3d** (Filter entity-level `constraints:`) | **(d3) — Filter does not carry `constraints:` in v1.** Reachability + admissibility deferred to `[TD-FILTER-REACHABILITY]`. | `11 §8.1` (carrier roster), `[TD-CONSTRAINT-CARRIER-EXT]` (single rolled-up note covering Filter / Key / Dimension / DataKind future carriers) |

**Carrier-roster movement (relative to the snapshot below)**: the §1.2 table previously listed `Filter` as "to-be-ratified"; Q-R4.3d closed that as **out-of-scope for v1**. Reserved future carriers (Dimension, Key, DataKind) — previously contemplated as future scaffolding — are folded into a single `[TD-CONSTRAINT-CARRIER-EXT]` tech-debt note rather than carrying section-level scaffolding.

The remaining body (below) is the original deferred-session snapshot, frozen for historical fidelity. **Do not edit**; new questions land in [`questions/open/19_questions.md`](../open/19_questions.md) (category-extension items) or in a fresh `11_questions.md` sidecar if a follow-up emerges that doesn't fit the category axis.

---

**Parent document**: [`../foundations/11_names_and_scopes.md`](../foundations/11_names_and_scopes.md) §8 (Constraints).

**Related docs**:
- [`../foundations/10_resolution_pipeline.md`](../foundations/10_resolution_pipeline.md) §3.4 — step-0 validation framing
- [`../foundations/13_types_and_grain.md`](../foundations/13_types_and_grain.md) §5.3 — Key participation
- [`../apis/32_semstrait_model.md`](../apis/32_semstrait_model.md) — YAML shape of the `constraints:` block

---

## 1. Ratified framing

Both points below are **ratified** — not open. They are the fixed axis the deferred session resumes on.

### 1.1 Two-axis constraint model

Constraints in `semstrait` are conceptually generic and split along **one axis**:

| Kind | Source | Surface | Authored? |
|---|---|---|---|
| **Implicit** | Functional role of the entity (what it IS) | Compiler / planner invariants | No — never appears in YAML |
| **Explicit** | Authored `constraints:` block on a carrier | Model YAML | Yes |

**Implicit examples** (not authoritative — authoritative definitions live in the doc owning that role):
- "Cannot derive `SUM` from a Dimension or Key" — fact about Dimension / Key functional role
- "Cannot nest Grainset inside Grainset" — fact about nesting matrix (`12_nesting_policy.md`)
- "Request must include the rollup dim set for a measure without an `agg:`" — fact about request shape
- "`SemanticExpr` cannot reference a physical column; `PhysicalExpr` cannot reference a Semantics name" — Semantics boundary rule

Implicit constraints are enforced by code invariants (validators, nesting matrix, type-state). `11 §8` surveys them with pointers but is not their authoritative home.

### 1.2 Explicit constraint carriers in v1

| Carrier | Code state | Spec state |
|---|---|---|
| `Measure` | In code — `MeasureConstraints { dimensions, aggregations }` | Realized, documented |
| `Metric` | In code — same `MeasureConstraints` type reused | Realized, documented |
| `Filter` | Not in code | **To-be-ratified in the deferred session** |
| Any other element | Not in code, not in scope | Future extension behind a named TD, not a section header |

The structural scope of `11 §8` is these three carriers. No generic "reserved future carriers" scaffolding beyond a single `[TD-CONSTRAINT-CARRIER-EXT]` note.

---

## 2. Open ratification items

These are the concrete questions the deferred session will resolve. Answers land in `11 §8` after ratification.

### Q-R4.3a — `aggregation:` sub-block semantics

Two plausible readings of `Measure.constraints.aggregations.{allowed, prohibited}` (and same on Metric):

- **Reading 1 — downstream derivation.** `allowed` / `prohibited` restricts which aggregation functions may be applied to this Measure / Metric when it's consumed downstream (e.g. rollups, further aggregation over a pre-aggregated metric). A request that tries to `SUM(<metric with agg: avg, aggregations.prohibited: [sum]>)` fails at step 0.
- **Reading 2 — replaced by measure/metric "kinds".** Drop the sub-block entirely; capture the intent by extending the kind of aggregation function the measure / metric carries (non-additive marker, distinct-only marker, ratio-typed, etc.), and let implicit rules enforce downstream restrictions.
- **Hybrid** — Reading 2 as the primary mechanism, Reading 1 as an escape hatch for cases the kind system cannot express.

**User note from the in-flight thread**: "it seems like it must only affect derivation, or can be replaced by metric/measure types which implicitly implementing those constraints". Strongly suggests Reading 1 or Hybrid.

### Q-R4.3b — DSL spelling and rename

Four specific spellings to ratify:

1. Outer key: `aggregation` (singular, used in the in-flight example) vs `aggregations` (plural, current code). Consistency argument: `dimensions` (plural) in the same block.
2. Third dimensions key: `all_of` (in-flight example, symmetric with `one_of` / `none_of`) vs `all` (current code).
3. Whether `one_of` / `none_of` / `all_of` are **mutually exclusive** in a single constraints block, or whether multiple sub-blocks can compose via AND.
4. `MeasureConstraints` rename: with Filter as a third carrier, the type name is legacy. Candidates: `Constraints` (shared across all three carriers), or per-carrier concrete types (`MeasureConstraints`, `MetricConstraints`, `FilterConstraints`) with a shared trait. Connects to `[TD-CONSTRAINT-RENAME]`.

### Q-R4.3c — `constraints.filter:` sub-block inside Measure / Metric

Does `Measure.constraints` / `Metric.constraints` carry a `filter:` sub-block at all?

- **(c1) No sub-block.** The existing `filters: Vec<MeasureFilter>` on the carrier is the whitelist. Nothing to add.
- **(c2) Sub-block as explicit denylist.** `filter: { prohibited: [name_a, name_b] }` — declares specific filters that may be defined on the carrier but should not apply in certain request contexts.
- **(c3) Sub-block constrains request-side filter composition.** Filter authored elsewhere (top-level or other entity) is forbidden from composing with this measure.

**User note from the in-flight thread**: "model author is specifying the interface, and everything non covered by model is not allowed. I'm might be wrong." Reads as lean toward **(c1)**.

### Q-R4.3d — Filter entity-level `constraints:`

Separate from Q-R4.3c — this is whether `Filter` entities themselves carry a `constraints:` field.

- **(d1) Same shape as Measure / Metric.** Filter gets `constraints: { dimensions, aggregations }` with the same semantics. A Filter with `dimensions.one_of: [date]` cannot be applied unless the request includes `date`.
- **(d2) Expression-form predicate.** `constraints:` on a Filter is an `ExprSource` over request / compose context — a boolean predicate that must hold for the filter to apply. More expressive; introduces a new expression evaluation context (request-time) that must be spec'd.
- **(d3) No Filter-level field — the "Filter carrier" mention was about the sub-block in Q-R4.3c, not entity-level.** Only Measure + Metric carry authored constraints in v1; Filter is not a third carrier.

---

## 3. Concrete DSL example (in-flight, not ratified)

Surfaced during the thread as the user's working shape for `Measure.constraints:` — kept here as the starting point for ratification. This is **not** the spec; it is the example the deferred session starts from.

```yaml
constraints:
  dimensions:          # exactly one of one_of / none_of / all_of
    one_of:            # at least one of these must be present in the request
    none_of:           # none of these may be present in the request
    all_of:            # all of these must be present in the request
  aggregation:         # semantics to ratify (Q-R4.3a)
    allowed:           # list of allowed aggregation functions
    prohibited:        # list of disallowed aggregation functions
  filter:              # existence and shape to ratify (Q-R4.3c)
    ...
```

---

## 4. Ground-truth snapshot

Code reality at the time of deferral (`crates/semstrait-model/src/types/measure.rs` + `metric.rs`):

```rust
pub struct Measure {
    // ...
    pub constraints: Option<MeasureConstraints>,
    // ...
}

pub struct Metric {
    // ...
    pub constraints: Option<MeasureConstraints>,  // same type reused
    // ...
}

pub struct MeasureConstraints {
    pub dimensions: Option<DimensionConstraints>,
    pub aggregations: Option<AggregationConstraints>,  // note: plural
}

pub struct DimensionConstraints {
    pub one_of: Option<Vec<String>>,
    pub none_of: Option<Vec<String>>,
    pub all: Option<Vec<String>>,                      // note: bare "all"
}

pub struct AggregationConstraints {
    pub allowed: Option<Vec<String>>,
    pub prohibited: Option<Vec<String>>,
}
```

No `FilterConstraints` type. No `Filter.constraints` field.

---

## 5. Resumption checklist for the deferred session

1. Read `00_overview.md` and `STATUS.md` (mandatory).
2. Read this file in full.
3. Resolve Q-R4.3a (`aggregation:` semantics — pick Reading 1 / Reading 2 / Hybrid).
4. Resolve Q-R4.3b (spelling — `aggregation` vs `aggregations`, `all` vs `all_of`, mutual exclusion, rename).
5. Resolve Q-R4.3c (sub-block on Measure / Metric — c1 / c2 / c3).
6. Resolve Q-R4.3d (Filter entity-level — d1 / d2 / d3).
7. Write the fourth rewrite of `11 §8` against the ratified decisions.
8. Cascade cross-doc updates: `10 §3.4` (step-0 framing), `13 §5.3` (Key participation cross-ref), `32` (YAML shape), `41_deprecations.md` (rename schedule if applicable), `42_migration_notes.md` (migration recipes).
9. Update `STATUS.md` §§1, 2, 3, 5 to mark item E resolved.

---

## 6. Do not

- Begin writing a fourth rewrite of `11 §8` without ratifying Q-R4.3a–d first.
- Cascade cross-doc constraint updates (`10`, `13`, `32`, `41`, `42`) before `11` itself is rewritten.
- Re-open the ratified axis in §1 — that decision is final.
