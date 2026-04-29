---
doc: 19_categories
status: open
---

# 19 — Categories — Open Questions

> Sidecar for [`../foundations/19_categories.md`](../foundations/19_categories.md). Carries post-v1 category-extension items that did not block the tenth-pass ratification of the v1 roster (`DimensionType`, `MeasureCategory`, `MetricCategory`) and the v1 expandability invariants (`SR-CAT-FWD`, `SR-CAT-CLOSED`, `SR-E-19`).

**Authored**: 2026-04-27 (tenth pass — categories+constraints expansion).

**v1 ratification scope (closed in `19`)**:

- `DimensionType` enum roster `{Temporal, Categorical, Binary, Geo, Bucketed, Metadata}` — pre-existing in `18 §4.1`; `19 §2` ratifies its **category-axis** role (implicit-constraint table, planner / adapter contract).
- `MeasureCategory` enum roster `{Additive, MinMax, Average, Distinct, Statistical, Boolean, Snapshot, Custom}` + per-variant body structs (`DistinctMeasureBody`, `StatisticalMeasureBody`, `SnapshotMeasureBody`).
- `MetricCategory` enum roster `{Simple, Ratio, Derived}`; `Cumulative` and `Conversion` are reserved post-v1 variants documented in `19 §4` as commented-out enum arms.
- Implicit-vs-explicit constraint contract (`19 §5`); SR-E-13 … SR-E-19 (`19 §6`); collapsed-wrapper YAML grammar (`19 §7`); growth recipe (`19 §8`).
- Expandability invariants: Layer 1 `SR-CAT-FWD` (Rust forward-compat: `#[non_exhaustive]` + serde defaults + exhaustive matches); Layer 2 `SR-E-19 validate.unknown-category` (strict reject on YAML); Layer 3 `SR-CAT-CLOSED` (closed spec-owned enum; no manifest-level registry); Layer 4 `19 §8` growth recipe (doc-discipline).

**Out of scope for this sidecar**: items already CLOSED in [`../closed/11_constraints_deferred.md`](../closed/11_constraints_deferred.md) (Q-R4.3a … Q-R4.3d on the constraint axis); per-engine adapter cascades (those land under the `registry/` catalogs as concrete demand surfaces).

---

## 1. Status summary

| Q-ID | Theme | Status | Resolution scope |
|---|---|---|---|
| Q-CAT-001 | `Identifier` Dimension category | **Open — deferred to follow-up ratification round** | v1 enum frozen at six variants per `18 §4.1` / `19 §2`; `Identifier` adoption requires its own implicit-constraint contract row + planner-routing decision. |
| Q-CAT-002 | `Snapshot` measure category vs `AdditivityType::Semi` authoring discipline | **Resolved (advisory)** — `Snapshot` subsumes explicit `additivity: semi` authoring; `AdditivityType` stays public for the planner contract. Listed here for traceability of the discipline decision. | `19 §3.3`; `18 §5.2`; `INDEX.md` duplication-guard row for `AdditivityType`. |
| Q-CAT-003 | `Cumulative` / `Conversion` metric categories — enum carriage in v1 source | **Resolved** — kept as commented-out reserved variants in the `MetricCategory` enum source (visible to readers, not authoring-legal), per `00 §10` post-v1 deferral. | `19 §4`; future ratification rounds for the variant bodies. |
| Q-CAT-004 | Author-extensible category registry (`[TD-CAT-REGISTRY]`) | **Open — deferred indefinitely** — closed-enum model holds for v1; registry pattern (mirroring `[14a §2 FunctionRegistry](../foundations/14a_function_catalog.md)`) only when concrete author demand surfaces. | `19 §1` (Layer 3); `[TD-CAT-REGISTRY]` tech-debt entry. |
| Q-CAT-005 | Lenient unknown-category downgrade (`[TD-CAT-LENIENT]`) | **Open — deferred to a future major version** — v1 enforces strict reject (Layer 2). Lenient mode requires a `plan.unknown-category-degraded` warning channel and `Custom` fallback semantics that drop unknown body fields. | `19 §1` (Layer 2); `[TD-CAT-LENIENT]` tech-debt entry. |
| Q-CAT-006 | Per-category planner routing — when does category change SQL emission? | **Open — pending adapter-cascade pass** | `19 §3.3` / §4.2 implicit-constraint tables surface the contracts; concrete per-engine SQL templates land in `registry/functions_mapping.md` once adapter framing closes. |
| Q-CAT-007 | Downstream-aggregation policy edge cases (SR-E-16 perimeter) | **Open** | `19 §3.3` / §4.2 + `11 §8.4` cover the v1 envelope; corner cases (e.g. wrapping a `Ratio` metric in `Avg` for cohort dashboards; double-count guards across nested `Joinset` children) need a worked-example pass. |
| Q-CAT-008 | `Balance` as naming alias for `Snapshot` | **Open — deferred indefinitely** — composition (`category: snapshot` + `constraints.aggregation:`) already covers the use case (worked example in `11 §8.4.1`); promotion to a first-class variant would be sugar-only, deferred until concrete naming-clarity demand surfaces. | `11 §8.4.1` (worked example); `19 §3.3` (Snapshot row). |

---

## 2. Q-CAT-001 — `Identifier` Dimension category

**State**: **Open — deferred for ratification.** The tenth-pass plan §5 surfaced `Identifier` as a candidate seventh `DimensionType` variant; the user's response chose to defer rather than land it inline.

**Motivation captured during the tenth pass**:

> *High-cardinality bare ID (foreign-key column exposed as a Dimension); planner avoids it as a default group-by; adapters may forbid it from `SELECT DISTINCT` constraints. Distinguishes ID-like Dimensions from low-cardinality `Categorical`.*

**Why deferred**:

1. The five existing variants (`Categorical`, `Binary`, `Geo`, `Bucketed`, `Metadata`) plus `Temporal` cover every dimension shape the current code treats meaningfully. `Identifier` is a **planner-hint refinement** of `Categorical`, not a new shape.
2. The implicit-constraint contract for `Identifier` (avoid as default group-by; planner-warning on `SELECT DISTINCT`; adapter-emission rule "don't pull this into a high-cardinality `GROUP BY` unless explicitly requested") is non-trivial and overlaps with the **Keys** axis (`18 §9`). Adopting `Identifier` without first resolving the Keys-vs-Dimension role-assignment story would lock in a half-baked contract.
3. The **closed-enum model** (Layer 3, `SR-CAT-CLOSED`) means adding `Identifier` is a MINOR release with mechanical adapter / registry / `25 §2.11` cascade. Cheap to add later under the growth recipe (`19 §8`); expensive to retract if the contract is wrong.

**Resume from**: a worked example showing two queries against the same model — one that reasonably groups by an `Identifier` (e.g. "top-10 customers by revenue") and one that does not (e.g. "revenue by region"), and the planner's expected behavior in each case. With the example, decide whether `Identifier` warrants its own variant or whether a `categorical_hint:` field on `CategoricalDimensionBody` (with `cardinality_class ∈ {low, mid, high, identifier}`) is the right shape.

**Tech-debt tag**: `[TD-CAT-IDENTIFIER]`.

---

## 3. Q-CAT-002 — `Snapshot` measure category vs `AdditivityType::Semi` authoring (informational)

**State**: **Resolved — recorded for traceability.** The tenth-pass plan §5 surfaced this as a discipline question; the user's decision: keep `AdditivityType` public (for the planner contract — `Manifest` consumers, optimizer rules in `34`, IR encoding in `35`) but **discourage** explicit `additivity: semi` authoring in favor of `category: snapshot` with an inline body.

**Outcome**:

- `MeasureCategory::Snapshot(SnapshotMeasureBody)` is the preferred authoring surface (`19 §3.3`).
- `AdditivityType::Semi { axes, strategy }` is **derived** from the category body at validation time — author-stated `additivity: semi` is accepted only if it agrees with the category-derived value (else SR-E-13).
- `INDEX.md` duplication-guard row for `AdditivityType` documents the subsumption: "`MeasureCategory::Snapshot` synthesizes `AdditivityType::Semi`".

**Why not delete `AdditivityType::Semi` from the authoring surface entirely**: legacy manifests (Phase-3 migration window) may still carry explicit `additivity: semi`. Strict-reject would break them; the agreement-check (SR-E-13) lets them parse and validates them positively. A future major version may flip the discipline to "`AdditivityType::Semi` is internal-only"; tracked as a Phase-4 cleanup item, not a Q-CAT.

**No further work required** — left in this sidecar so future readers can audit the decision trail without spelunking through STATUS.md.

---

## 4. Q-CAT-003 — `Cumulative` / `Conversion` metric category enum carriage (informational)

**State**: **Resolved — recorded for traceability.** The tenth-pass plan §5 surfaced this; the user's decision: keep `Cumulative` and `Conversion` as **commented-out reserved variants** in the `MetricCategory` enum source (visible in `19 §4` to readers as a forward-reference) rather than omitting them entirely.

**Outcome**:

- `MetricCategory` v1 roster is `{Simple, Ratio, Derived}`; `Cumulative(CumulativeMetricBody)` and `Conversion(ConversionMetricBody)` are present as commented-out arms with brief descriptions and a `// POST-V1 (00 §10 deferred)` marker.
- This signals the design intent (these will land) without making them authoring-legal in v1.
- `00 §10` already defers them; `19 §4` re-documents the deferral with a one-paragraph forward-reference describing the variant intent.

**Why not omit them entirely**: discoverability. A reader landing on `19 §4` should see the full design intent for the metric-category space, not just v1. The commented-out arms + the §10 deferral note give the right reading: "these exist in the roadmap, not in v1."

**No further work required** — left here for traceability; promotion to authoring-legal happens when their bodies land (a separate ratification round per category, following the `19 §8` growth recipe).

---

## 5. Q-CAT-004 — Author-extensible category registry (`[TD-CAT-REGISTRY]`)

**State**: **Open — deferred indefinitely.** Tracked as `[TD-CAT-REGISTRY]` tech-debt entry; surfaces only when concrete author demand exists.

**Sketch of the deferred design**:

- Manifests would declare project-local categories via a top-level `category_definitions:` block:
  ```yaml
  category_definitions:
    measures:
      revenue_per_customer:
        derives_from: ratio
        implicit_constraints:
          numerator: { measure_category: additive }
          denominator: { measure_category: distinct }
        agg_derivation: avg
        additivity: non
        adapter_emission_template: "SUM({numerator}) / COUNT(DISTINCT {denominator})"
  ```
- The registry mirrors `[14a §2 FunctionRegistry](../foundations/14a_function_catalog.md)`: per-project catalog with a stable extension surface; planner / adapter consume the registry the same way they consume the spec-owned enums; SR-E-* rules generalize from "category mismatch" to "category mismatch — implicit-constraint table from registry entry `<name>`".
- The escape hatch (`Custom`) stays anonymous and pre-registered; the registry adds named, project-shareable categories.

**Why deferred**:

1. The spec-owned roster (8 measure categories, 3 metric categories) covers the entire dbt + Cube + LookML mental model. Author demand for project-local categories has not surfaced.
2. Implicit-constraint tables are subtle; getting them wrong silently breaks query plans. The closed-enum model lets the spec maintainer review every contract row before it ships. A registry pattern shifts that responsibility to manifest authors, who may not have the planner expertise.
3. Layer 3 (`SR-CAT-CLOSED`) is the **safe-then-loosen** posture; opening up a registry is a one-way door. Better to wait for concrete author pain.

**Trigger to revisit**: a concrete user request showing their need for a category whose implicit-constraint contract differs from any of the 11 v1 variants AND from `Custom`. Until then, the closed-enum model holds.

**Tech-debt tag**: `[TD-CAT-REGISTRY]`.

---

## 6. Q-CAT-005 — Lenient unknown-category downgrade (`[TD-CAT-LENIENT]`)

**State**: **Open — deferred to a future major version.** Tracked as `[TD-CAT-LENIENT]` tech-debt entry.

**Sketch of the deferred behavior**:

- When the binary encounters a `category:` value it does not recognize:
  - **v1 (current)** — strict reject with `validate.unknown-category` (SR-E-19) and a "spec version older than manifest" hint.
  - **lenient mode (deferred)** — downgrade the unknown category to `Custom`, emit `plan.unknown-category-degraded` warning with the original category name, drop unknown body fields silently, and require explicit `agg` + `additivity` declarations on the carrier (else fall through to the standard `Custom` validation path).

**Why deferred**:

1. **Strict-then-loosen is the safe direction.** Adding lenient mode later is a MINOR; retracting it is a breaking change. v1 should be strict.
2. The warning channel (`plan.unknown-category-degraded`) needs a structured shape that integrates with `30 §6` diagnostics — and that integration depends on `34`'s warning-propagation contract, which has its own open Q&A.
3. The legitimate use-case (forward-read compatibility: read a manifest authored against a newer spec on an older binary) only emerges after **two** consecutive MAJOR versions ship, since v1 manifests are the only manifests in existence right now.

**Trigger to revisit**: when a downstream tooling project (e.g. a mid-tier query optimizer that wraps `semstrait`) reports that its build pipeline would benefit from forward-read compatibility — that is, when v2 manifests start to circulate while v1 binaries are still in production. Until then, strict reject holds.

**Tech-debt tag**: `[TD-CAT-LENIENT]`.

---

## 7. Q-CAT-006 — Per-category planner routing (adapter cascade)

**State**: **Open — pending adapter-cascade pass.** This question rolls up everything in the implicit-constraint tables (`19 §3.3` for measures; `19 §4.2` for metrics) into per-engine SQL templates.

**Scope**:

- Per (`MeasureCategory`, `engine`) pair: the canonical SQL emission template (e.g. `Snapshot` on DataFusion ⇒ "use `LAST_VALUE(... ORDER BY ...) OVER (PARTITION BY ... )` for the non-additive axes; aggregate normally for additive axes").
- Per (`MetricCategory`, `engine`) pair: how the metric category shapes the planner's join graph (e.g. `Ratio` ⇒ planner forces both `numerator` and `denominator` to share a `GROUP BY` envelope; `Derived` ⇒ each input may aggregate at its own grain before composition).
- Cross-cutting interactions: `Snapshot` measure inside a `Ratio` metric (non-additivity-of-axes propagates to the metric); `Distinct` measure inside a `Derived` metric (the `CountDistinct` cannot be pulled out into the outer expression).

**Why deferred**:

- The implicit-constraint tables in `19 §3.3` / §4.2 ratify the **planner contract** (what each category guarantees about `agg` / `additivity` / `expr` shape); concrete per-engine SQL templates are an **adapter** concern that follows once Item C (adapter / catalog framing) ratifies.
- Premature SQL templating before the adapter framing closes risks codifying engine-specific behavior in the wrong layer.

**Trigger to revisit**: alongside Item C ratification (per `STATUS.md §2`). The result lands in `registry/functions_mapping.md` (per-engine canonical-fn rewrite tier rows extended with category-aware variants) plus a new `registry/categories_mapping.md` if the matrix is large enough to warrant its own catalog.

**Tech-debt tag**: `[TD-CATEGORIES-MIGRATE]` (existing — covers code migration; SQL-template work is part of the same cascade).

---

## 8. Q-CAT-007 — Downstream-aggregation policy edge cases (SR-E-16 perimeter)

**State**: **Open** — needs a worked-example pass.

**Context**: SR-E-16 (`validate.downstream-aggregation-violation`) fires when a request tries to wrap a Measure or Metric with an aggregation function that the carrier's category disallows (e.g. `SUM(<measure with category: average>)`). The basic envelope is clear from the implicit-constraint tables. The corner cases are not.

**Open corners**:

- **Cohort dashboards** — wrapping a `Ratio` metric in `Avg` for "average conversion rate across cohorts." Does the `Ratio` category's `additivity = Non` block this universally, or is `Avg` allowed under the `aggregation: { allowed: [avg] }` explicit-narrowing block (`11 §8.4`)?
- **Nested `Joinset` children with double-count guards** — a `Distinct` measure declared on a `Dataset` that participates in a `Joinset` with a fanout-inducing relationship: is the double-count guard a category-implicit constraint (SR-E-16 fires automatically) or an explicit `aggregation: { all_of: { joinset_role: [primary] } }` block authored by the user?
- **Statistical kinds and rollup** — `StdDev` and `Variance` are `additivity = Non` mechanically, but `StdDev(StdDev(...))` is a meaningful question in some statistical contexts (pooled variance). v1 says no; document the narrow escape hatch (`category: custom` + manual additivity statement) for the future cases.

**Why deferred**:

- Each corner case is a real query a user might write. Resolving them in the abstract risks codifying the wrong default. A worked-example pass — three to five concrete queries with their expected planner behavior — gives the contract grounding.
- The corner cases don't block v1 ratification because the basic SR-E-16 envelope is correct (and conservative — false-positive rejections can be loosened by an explicit `constraints:` block; false-negative acceptances cannot be retracted without a breaking change).

**Trigger to revisit**: when the constraints session §3.1 (now closed) gets a follow-up "real-world examples" round — typically driven by the first two or three production-weight manifests authored against the v1 spec.

**Tech-debt tag**: `[TD-CAT-DOWNSTREAM-EDGES]`.

---

## 9. Q-CAT-008 — `Balance` as a naming alias for `Snapshot`

**State**: **Open — deferred indefinitely.**

**Context**: The balance use case (banking-balance, inventory-level, debt-position) is fully covered by the existing composition `category: snapshot` + `constraints.aggregation:` — worked example in [`../foundations/11_names_and_scopes.md §8.4.1`](../foundations/11_names_and_scopes.md). A `MeasureCategory::Balance` variant would offer **no new mechanism**: same body shape (`non_additive_axes`, `strategy`), same implicit-constraint contract, identical planner / adapter routing. The only differentiator is the *spelling* (`category: balance` vs `category: { snapshot: { ... } }`).

**Why deferred**:

1. **No mechanism gap.** The composition expresses everything balance needs; a `Balance` variant would be sugar over `Snapshot + constraints.aggregation:`. Per [`../STATUS.md §7 L13`](../STATUS.md), labeled invariants earn their place when they carry a distinct contract — aliases without contract divergence are second-naming for the same invariant.
2. **Doc-cascade overhead.** Per `19 §8`, every new `MeasureCategory` variant runs the full 8-section growth recipe (variant + body shape, implicit-constraint table row, planner routing, adapter emission, SR-E-* additions, manifest YAML examples, peer-system lineage, TD entries). For a sugar-only variant the cost / benefit is heavily negative.
3. **Strict-then-loosen.** Adding `Balance` later is a MINOR (per `SR-CAT-FWD` Layer 1 — `#[non_exhaustive]` on `MeasureCategory`); retracting it would break authoring contracts. The safe direction is to keep the closed roster minimal until concrete demand surfaces.

**Trigger to revisit**: banking / inventory / position-tracking authors report that `category: snapshot` reads as semantically misleading for stock-balance measures and would prefer a domain-flavored spelling. A concrete request showing the spelling causes real authoring confusion would justify the recipe pass.

**Tech-debt tag**: `[TD-CAT-BALANCE-ALIAS]`.

**Cross-references**:

- [`../foundations/11_names_and_scopes.md §8.4.1`](../foundations/11_names_and_scopes.md) — worked example demonstrating the composition; section "The 'Balance' idiom — composition, not a separate category".
- [`../foundations/19_categories.md §3.3`](../foundations/19_categories.md#33-implicit-constraint-contract-per-measure-category) — Snapshot row in the implicit-constraint contract table.
- [`../STATUS.md §7 L13`](../STATUS.md) — labeled-invariant lesson; aliases without contract divergence are second-naming.

---

## 10. Cross-references

- [`../foundations/19_categories.md`](../foundations/19_categories.md) — parent doc.
- [`../foundations/18_entities.md §4.1`](../foundations/18_entities.md) / [§5](../foundations/18_entities.md) / [§6](../foundations/18_entities.md) — `DimensionType` / `Measure.category` / `Metric.category` field carriers.
- [`../foundations/11_names_and_scopes.md §8`](../foundations/11_names_and_scopes.md) — `Constraint` consumer of the category-implicit contract.
- [`../closed/11_constraints_deferred.md`](../closed/11_constraints_deferred.md) — frozen historical record of the deferred constraints session that closed alongside this sidecar's authoring.
- [`../STATUS.md §2 item I`](../STATUS.md) — categories ratification entry; §3.1 (now closed); §5 tenth-pass checkpoint.
- [`../INDEX.md`](../INDEX.md) — alphabetical and duplication-guard rows for `MeasureCategory` / `MetricCategory` / categories growth recipe.
