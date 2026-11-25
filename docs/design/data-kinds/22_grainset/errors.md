---
id: 22-grainset-errors
status: ratified
version: 2.1
impl: drift
---

# Grainset — Error Rosters

## 7. Validation Preconditions — `VALID_E_2200`–`2299`

Structural well-formedness for Grainsets is partially upstreamed to:

- **R1 / R2** (parse-stage type-level absence per `26 §2.1` / `§2.2`; diagnostics `parse.unknown-field` and `parse.illegal-self-nesting`).
- **R3** (validate-stage walk per `26 §2.3`; diagnostic `validate.complex-data-kind-insufficient-children`).
- **SR-E-7** (no `grain` on Complex; type-level absence on `ComplexExtras`).
- **SR-E-8** (Grainset-child grain required; diagnostic `validate.grainset-child-grain-required` per `26 §6`).
- **SR-3** name uniqueness, **SR-4** / **SR-5** field validity, **SR-7** unknown-field rejection — all in `32 §6`.

The Grainset-specific structural checks below run at validate stage:

| Code | Variant | Trigger |
|---|---|---|
| `VALID_E_2201` | `GrainsetDuplicateNestedChildName { grainset, name, indices }` | Two nested children of the same Grainset (across `datasets:` / `unionsets:` / `joinsets:`) declare the same `body.base.name`. |
| `VALID_E_2202` | `GrainsetInsufficientUniqueGrains { grainset, distinct_grains }` | Children's `extras.temporal.grain:` values include fewer than 2 distinct values (per §5.2 V1 strict). |

**Extensibility.** Codes `2203`–`2299` reserved; MINOR per `30 §6.3`.

## 8. Compile Preconditions — `COMP_E_2200`–`2299`

Compile-stage checks run after `validate` has passed and child resolution has produced per-child interface + Coverage data + `ComposedSemanticInterface` construction.

| Code | Variant | Trigger |
|---|---|---|
| `COMP_E_2201` | `GrainsetChildShapeKindMismatch { grainset, children: Vec<(NestedDataKindName, TemporalShape)> }` | One or more children's `TemporalShape.kind` (or SCD subtype) diverges from the others. Single consolidated code; covers kind mismatches and SCD subtype mismatches per §5.2. Fail-fast. |
| `COMP_E_2202` | `GrainsetCoverageIncomplete { grainset, semantics }` | A Semantics declared on the Grainset's own `SemanticInterface` is provided by NO child (every child's Coverage is `NullFill`). The Semantics is unanswerable at any grain — authoring mistake. Fail-fast. |
| `COMP_E_2203` | `GrainsetTemporalShapeKindMismatchWithGrainset { grainset, grainset_kind, child_kinds }` | The Grainset author wrote `extras.temporal:` at the Grainset level AND the kind disagrees with the (unanimous) children's kind. Cascade-up rule per §5.1 requires agreement. |
| `COMP_E_2204` | `GrainsetCrossGrainKeysAbsent { grainset, driver_unit, attached_unit, missing_via }` | The cross-grain JOIN-tree construction (§3.4) needs to attach `attached_unit` to the JOIN tree, but no shared Keys exist between `attached_unit` and any unit already in the tree (the driver or an earlier attached unit). Per G-2d ratification: hard compile error. |
| `COMP_E_2205` | `GrainsetCrossGrainKeyTypeMismatch { grainset, key_name, units_with_types: Vec<(NestedDataKindName, DataType)> }` | Two units both declare a Key with the same name but with `DataType`s that cannot be reconciled under `13 §7`'s widening rules. The equi-join would be type-unsafe. |
| `COMP_E_2206` | `GrainsetCrossGrainNoEquiJoinPath { grainset, request_semantics, driver_unit, missing_units }` | Compile-stage check: for at least one structurally-feasible Request (one that touches Semantics in `request_semantics`), the cross-grain JOIN tree cannot be assembled because some required attached unit has no Key reachable from the driver. (Subset of `COMP_E_2204`; surfaced for static analysis.) |

**Extensibility.** Codes `2207`–`2299` reserved; MINOR per `30 §6.3`.

## 9. Plan-Stage Rules — `PLAN_E_2200`–`2299`

Plan-stage checks run against a specific `Request` and the resolved Grainset.

| Code | Severity | Variant | Trigger |
|---|---|---|---|
| `PLAN_E_2201` | Error | `GrainsetNoMatchingUnitByGrain { grainset, request_grain, unit_grains }` | No effective routing unit has `unit.grain ≤ request_grain`. Every unit is strictly coarser than the Request. Authors should add a coarser-grain unit or restructure the Request. |
| `PLAN_E_2202` | Error | `GrainsetSemanticsNotCoverableByJoin { grainset, request_semantics, uncovered }` | After cross-grain JOIN-tree construction (§4.3), some Semantics in the Request's `select` / `group_by` / `filters` remain uncovered. Authors fix by adding a unit covering the missing Semantics or removing them from the Request. |
| `PLAN_E_2203` | Error | `GrainsetSnapshotRollupWithoutPin { grainset, unit, unit_grain, request_grain }` | A `Snapshot`-kind unit was selected for a Request requiring rollup but `17`'s pin policy did not declare a coarser-grain aggregation rule. (Internal `RollupPolicy::PinOnly` rejects.) |
| `PLAN_E_2204` | Error | `GrainsetSCDRollupWithoutAsOf { grainset, unit, unit_grain }` | An `Scd`-kind unit was selected for a Request lacking an as-of anchor. Per `17`'s SCD mechanics. |
| `PLAN_W_2201` | Warning | `GrainsetUnitPrunable { grainset, unit, reason }` | A routing unit was eliminated from candidacy (Coverage-pruned per §4.5, or grain-ineligible). Advisory; informs authors of unused units. |
| `PLAN_W_2202` | Warning | `GrainsetCrossGrainJoinFanout { grainset, attached_unit, expected_fanout }` | Cross-grain JOIN may fanout (e.g. monthly-attached joined onto daily-driver introduces a 1-to-N relationship via the equi-join Key). Advisory; authors verify the join doesn't multiplicatively inflate row counts. |

**Extensibility.** Codes `2205`–`2299` (errors) and `2203`–`2299` (warnings) reserved; MINOR per `30 §6.3`.

**Codes retired in this rebase:**

- Pre-cascade `PLAN_W_2202 MixedShapeAdvisoryChildren` — now `COMP_E_2201` (mixed shapes is a hard compile error per V1 strict equivalence).
- Pre-cascade `PLAN_E_2200 GrainsetNoMatchingChildByGrain` — renamed to `PLAN_E_2201 GrainsetNoMatchingUnitByGrain` (operates at the routing-unit level, not the raw-child level).

## 10. Worked Example

### 10.1 YAML

(Re-uses the §2.3 YAML — `paid_media_rollups` Grainset with two children at `Day` and `Month` grains, both `kind: snapshot`.)

### 10.2 Request A — single-unit case

```
Request {
  from: paid_media_rollups,
  fields:  [ report_date (rollup Month), campaign_id, cost ],
  filters: [ paid_only ],
}
```

Effective routing units: `Day` (single child `paid_media_daily`), `Month` (single child `paid_media_monthly`). Both are grain-eligible at `Month` (Day rolls up; Month direct).

Coverage: `cost` is provided by both. `Day` unit covers `cost` Natively; `Month` unit covers `cost` Natively.

Most-covering at Month: tie (both cover {report_date, campaign_id, cost}). Tie-break: name order → `Day` (`paid_media_daily` sorts alphabetically before `paid_media_monthly`).

Wait — but `Day` requires rollup; `Month` is direct. The "most-covering" tie-break breaks before the cost-of-rollup is consulted. V1 picks `Day` (name order). Future cost-aware extensions may flip this.

Single-unit delegation (§4.2): `Day` chosen; rollup wrapper added.

```
Project          (final shape — report_date (month), campaign_id, cost)
  Agg            (group_by: [DateTrunc(snapshot_date, Month), campaign_pk], aggregates: [Sum(daily_cost) AS cost])
    Filter: paid_only
      Project    (rename per Day unit's seam)
        Scan     (s3://b/daily/*.parquet)
```

### 10.3 Request B — cross-grain JOIN case

```
Request {
  from: paid_media_rollups,
  fields:  [ report_date (rollup Month), campaign_id, cost, clicks ],
  filters: [],
}
```

Coverage:

- `Day` unit (`paid_media_daily`): covers `cost`, `clicks`, `campaign_id`, `report_date`. (4/4 — full)
- `Month` unit (`paid_media_monthly`): covers `cost`, `campaign_id`, `report_date`. Does NOT cover `clicks`. (3/4)

Single-unit case: `Day` covers all 4 → §4.2 single-unit delegation, no JOIN needed. Plan shape similar to §10.2 but projecting `clicks` too.

To illustrate the cross-grain JOIN case (§4.3), assume `paid_media_daily` lacks `clicks` but instead has it on a hypothetical `paid_media_quarterly_clicks` unit (kind `snapshot`, grain `Quarter`, only carries `clicks` + `campaign_id`):

Coverage:

- `Day` unit (`paid_media_daily`): covers `cost`, `campaign_id`, `report_date`. (3/4)
- `Month` unit (`paid_media_monthly`): covers `cost`, `campaign_id`, `report_date`. (3/4)
- `Quarter` unit (`paid_media_quarterly_clicks`): covers `clicks`, `campaign_id`, `report_date`. (Quarterly Coverage; rolls up to Month? Month is finer than Quarter → Quarter ineligible at Month grain. Restructure: assume the Request is at grain `Quarter`.)

Reframed Request at `Quarter`:

- All three units grain-eligible (Day → Quarter, Month → Quarter, Quarter → Quarter).
- Most-covering at Quarter: `Day` and `Month` tied at 3 (both cover {cost, campaign_id, report_date}). Name-order tie-break (`paid_media_daily` < `paid_media_monthly`) → `Day` is driver.
- `clicks` uncovered by driver → need attached unit. Greedy name order (`paid_media_monthly` < `paid_media_quarterly_clicks`): `Month` is next, but doesn't cover `clicks`. Skip to `Quarter`: covers `clicks` AND shares `campaign_id` Key with `Day`. Attach.
- After driver + 1 attached, all 4 Semantics covered.

Observable plan shape:

```
Project           (final shape — report_date (quarter), campaign_id, cost, clicks)
  Agg             (group_by: [DateTrunc(report_date, Quarter), campaign_id])
    Join { LeftOuter, on: campaign_id = campaign_id }
      <Day unit's subplan, rolled up to Quarter via DateTrunc + Agg>
      <Quarter unit's subplan, no rollup needed>
```

### 10.4 Reading key

- Effective routing units = one per distinct grain. Same-grain children would pre-merge into an implicit Unionset (`mode: All`, non-strict NullFill per §3.3); the Unionset would be the routing unit.
- §4.2 single-unit delegation is the common case. Cross-grain JOIN (§4.3) fires only when no single unit covers the full Request.
- LEFT OUTER JOIN preserves driver's row set. Attached units contribute additional Measures via equi-join on shared Keys (per `18 §2.5`). Per G-2a ratification.
- Driver = most-covering at the Request grain; name-order tie-break (G-2b). Attached added in name order (G-2c — deterministic).
- `RollupPolicy` is internal-only in V1 (per G-4). Default rollup behavior follows shape rules in `17`.
- Algorithm body — exact column ordering, `DateTrunc` placement, partial-aggregate staging, JOIN-key index lookup — lives in `34 §<GrainsetStrategy>` (parked in [`../_drafts/34_grainset_strategy.md`](../../_drafts/34_grainset_strategy.md)).

## 11. Open Items

Round-2 review (post-thirteenth-pass cascade rebase, 2026-05-03) closed three previously-open items and re-scoped a fourth; details in [`../questions/closed/22_questions.md`](../../questions/closed/22_questions.md). Items remaining open in [`../questions/open/22_questions.md`](../../questions/open/22_questions.md):

- (none in V1; all Round-1 questions closed or moved to deferred per the rebase)

Closed in this rebase:

- **Q-GRN-001** (inheritance default for child grain) — moot under SR-E-8 (every Grainset child MUST author `extras.temporal.grain:` explicitly; no inheritance mechanism in V1).
- **Q-GRN-002** (cross-child partial coverage) — ratified at G-2 as cross-grain LEFT OUTER JOIN composition via `ComposedSemanticInterface` + Keys per `18 §2.5`.
- **Q-GRN-005** (mixed-shape warning vs. error) — moot under V1 strict equivalence rule (§5.2); mixed shapes are now `COMP_E_2201` hard error.

Deferred (post-V1):

- **Q-GRN-003** — Cost-function pluggability hook site (planner trait vs. adapter trait vs. separate `CostEstimator`). Tracked in [`../questions/deferred/22_questions.md`](../../questions/deferred/22_questions.md).
- `[TD-GRAINSET-SHAPE-MIX]` — smart alignment of non-equivalent shapes (e.g. `Events + Snapshot`).
- `[TD-GRAINSET-COST-STATS]` — stats-backed cost function for driver / attached selection.
- `[TD-GRAINSET-NESTED]` — Grainset-of-Grainset nesting (closed via R2 in `26 §2.2`; future matrix relaxation post-V1).

---

**End of document.** Ratified decisions are inline throughout §§1–9; closed items in [`../questions/closed/22_questions.md`](../../questions/closed/22_questions.md); deferred items in [`../questions/deferred/22_questions.md`](../../questions/deferred/22_questions.md).

## Testing Strategy

- VALID_E_2201/2202; COMP_E_2201/2202/2204; PLAN_E_2201/2203; full roster round-trip per 30 §6.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
