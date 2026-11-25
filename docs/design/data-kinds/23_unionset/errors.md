---
id: 23-unionset-errors
status: ratified
version: 2.1
impl: drift
---

# Unionset — Error Rosters

## 7. Validation Preconditions — `VALID_E_2300`–`2399`

Structural well-formedness for Unionsets is upstreamed to:

- **R1 / R2** (parse-stage type-level absence per `26 §2.1` / `§2.2`; diagnostics `parse.unknown-field` and `parse.illegal-self-nesting`).
- **R3** (validate-stage walk per `26 §2.3`; diagnostic `validate.complex-data-kind-insufficient-children`).
- **SR-3** name uniqueness, **SR-4 / SR-5** field validity, **SR-7** unknown-field rejection — all in `32 §6`.

Consequently the `VALID_E_2300`–`2399` band is **largely empty** in v1. The single Unionset-specific structural check that does not fit cleanly upstream:


| Code           | Variant                                                        | Trigger                                                                                                                                                                                                                                                            |
| -------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `VALID_E_2301` | `UnionsetDuplicateNestedChildName { unionset, name, indices }` | Two nested children of the same Unionset (across `datasets:` / `grainsets:` / `joinsets:`) declare the same `body.base.name`. The `26 §4` addressing scheme (`unionsets[X].datasets[Y]`) requires nested-name uniqueness within the parent's combined child scope. |


**Extensibility.** Codes `2302`–`2399` reserved; MINOR per `30 §6.3`.

## 8. Compile Preconditions — `COMP_E_2300`–`2399`

Compile-stage checks run after `validate` has passed and child resolution has produced per-child interface + Coverage data.


| Code          | Variant                                                                                                           | Trigger                                                                                                                                                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COMP_E_2301` | `UnionsetChildShapeMismatch { unionset, children: Vec<(NestedDataKindName, TemporalShape)> }`                     | One or more children's `TemporalShape` (kind, SCD subtype, or grain) diverges from the others. Single consolidated code per C1 (2026-05-03); covers every equivalence violation (kind, subtype, grain). Fail-fast.                       |
| `COMP_E_2302` | `UnionsetCoverageIncomplete { unionset, semantics }`                                                              | A composed-surface Semantics declared on the Unionset's own `SemanticInterface` is provided by NO child (every child NullFills). The column would be always-NULL across every branch — almost certainly an authoring mistake. Fail-fast. |
| `COMP_E_2303` | `UnionsetCrossChildTypeDisagreement { unionset, semantics, children_types: Vec<(NestedDataKindName, DataType)> }` | Two or more children both provide a Semantics with logically incompatible `DataType`s that cannot be reconciled by `13 §7`'s widening rules. Fail-fast.                                                                                  |
| `COMP_E_2304` | `UnionsetUniqueModeIncompatibleType { unionset, semantics, data_type }`                                           | `mode: unique` declared but the Unionset's surface carries a Semantics whose `DataType` is non-comparable (e.g. `Binary`); the engine cannot dedupe. Fail-fast.                                                                          |


**Compile-stage warnings.**


| Code          | Variant                                                                         | Trigger                                                                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `COMP_W_2301` | `UnionsetNullabilityWidened { unionset, semantics }`                            | Per §4.4: at least one child reports `s` as non-nullable but another contributor (or NullFill) forces nullability; the unified column is nullable. Survives equivalence rule per C2. |
| `COMP_W_2302` | `UnionsetSnapshotMultipleAsOf { unionset, snapshot_identifier }`                | Per §5.3: two `Snapshot` children with equivalent shape but different as-of timestamps; `snapshotted_at` is on the surface (rows distinguishable). Survives equivalence rule per C2. |
| `COMP_W_2303` | `UnionsetSnapshotMultipleAsOfNoDiscriminator { unionset, snapshot_identifier }` | Per §5.3: same as `COMP_W_2302` but `snapshotted_at` is NOT on the surface (rows indistinguishable). Survives equivalence rule per C2.                                               |


**Extensibility.** Codes `2305`–`2399` (errors) and `2304`–`2399` (warnings) reserved; MINOR per `30 §6.3`.

## 9. Plan-Stage Rules — `PLAN_E_2300`–`2399`

Plan-stage checks run against a specific `Request` and the resolved Unionset.


| Code          | Severity | Variant                                                          | Trigger                                                                                                                                                                                                                                              |
| ------------- | -------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PLAN_E_2301` | Error    | `UnionsetRequestFieldNotCovered { unionset, semantics }`         | A Request references a Semantics on the Unionset's surface whose composition-level Coverage resolves to NullFill from every child. Edge case; `COMP_E_2302` catches most instances at compile, but pathological Request field sets may slip through. |
| `PLAN_E_2303` | Error    | `UnionsetRequestTotallyNullFilled { unionset, request_fields }`  | Coverage-driven branch pruning (§4.6) collapsed every branch; zero branches survive. Indicates a Request / manifest inconsistency.                                                                                                                   |
| `PLAN_E_2304` | Error    | `UnionsetReAggregationInfeasible { unionset, measure }`          | A requested Measure's re-aggregation is not decomposable (e.g. `Avg` per §4.5's table). Author restructures as a Metric.                                                                                                                             |
| `PLAN_W_2301` | Warning  | `UnionsetBranchPrunable { unionset, child_index, reason }`       | Per §4.6: a child's contribution to the Request's selected fields is pure NullFill; branch pruned (subject to exceptions). Advisory.                                                                                                                 |
| `PLAN_W_2302` | Warning  | `UnionsetReAggregationLossy { unionset, measure, reason }`       | Per §4.5's table: the requested Measure's re-aggregation function is `Lossy` (e.g. `CountDistinct` summed across branches). Advisory; result may overcount.                                                                                          |
| `PLAN_W_2303` | Warning  | `UnionsetUniqueThreeValuedNullCollision { unionset, semantics }` | Per §4.3's `Unique` mode note: rows differing only in NullFill positions are NOT deduplicated under three-valued logic. Advisory for authors expecting full dedup.                                                                                   |


**Extensibility.** Codes `2305`–`2399` (errors) and `2304`–`2399` (warnings) reserved; MINOR per `30 §6.3`.

**Note.** `PLAN_E_2302` (mixed-grain rollup error from pre-cascade `23`) is removed under V1 strict equivalence — mixed grains across children fail at compile (`COMP_E_2301`) and never reach plan stage.

## 10. Worked Example

### 10.1 YAML

(Re-uses the §2.3 YAML — full SemanticInterface surface + per-source-distinguishing literals.)

### 10.2 Request

```
Request {
  from: paid_media,
  fields:  [ ordered_at (rollup Month), source_platform, cost ],
  filters: [ paid_only ],
}
```

### 10.3 Observable plan shape

Both branches' `source_platform` is a per-source literal (`"adwords"` in branch 0, `"facebook"` in branch 1). The disjointness predicate (§4.5) recognizes the literals are distinct → final aggregation **elided**. Per-branch sub-aggregation always emitted because `cost` is a Measure.

```
Union { mode: All }                       (final aggregation elided — disjoint literals)
├─ Project                                (branch 0 — adwords; rename + literal projection)
│  ├─ ordered_at AS ordered_at,
│  │  Literal("adwords") AS source_platform,
│  │  cost AS cost
│  └─ Agg                                 (per-branch sub-aggregation; SimpleStrategy under)
│     ├─ group_by: [DateTrunc(order_date, Month)],
│     ├─ aggregates: [Sum(total_cost) AS cost]
│     └─ Filter: paid_only
│        └─ Project (rename) + Scan       (adwords/*.parquet)
└─ Project                                (branch 1 — facebook; same shape)
   ├─ ordered_at AS ordered_at,
   │  Literal("facebook") AS source_platform,
   │  cost AS cost
   └─ Agg
      ├─ group_by: [DateTrunc(ts, Month)],
      ├─ aggregates: [Sum(spend_cents) AS cost]
      └─ Filter: paid_only
         └─ Project (rename) + Scan       (facebook/*.parquet)
```

### 10.4 Reading key

- The literals `"adwords"` / `"facebook"` projected at each branch's `Project` make `source_platform` source-distinguishing; per §4.5 the disjointness predicate elides the final aggregation. Branches' partial aggregates are already correct because no branch contributes a row at the same `(month, platform)` key as another.
- Per-branch `Agg` runs unconditionally because `cost` is a Measure (per §4.2 — pre-aggregation is the canonical path).
- `Filter: paid_only` is pushed below per-branch `Agg` because it references no aggregate; placement is `34`'s concern.
- Per-branch `Project + Scan` is the child `SimpleStrategy`'s output (per `_drafts/34_simple_strategy.md`); `UnionsetStrategy` does not reach into the child.
- `mode: All` projects to the engine's `UNION ALL`; the (absent) final aggregation would have been the Unionset-level merge layer if literals were not source-distinguishing.

### 10.5 Implicit Unionset reference

The implicit-Unionset case (multi-source `Dataset`) is structurally identical — see `[21 §10](../21_dataset.md)`. Only the children are different (per-source pseudo-leaves vs. user-authored nested DataKinds) and the discipline is strict (`21 §3.2`).

## 11. Open Items

Round-2 review (post-thirteenth-pass cascade, 2026-05-03) closed six previously-open items and re-scoped a seventh; details in `[../questions/closed/23_questions.md](../../questions/closed/23_questions.md)`. Five items remain open in `[../questions/open/23_questions.md](../../questions/open/23_questions.md)`:

- **Q-UNI-004** — `Avg` re-aggregation: hard error stands; promote `Sum(num) / Sum(den)` Metric pattern in docs.
- **Q-UNI-006** — Single-child collapse after pruning: skip the `PlanNode::Union`; final shape per `35`'s `PlanNode::Union { inputs: Vec<_> }` validation rules.
- **Q-UNI-012** — `PlanNode::Union` mode-flag projection: defer to `35` for the exact field shape (`mode: UnionMode` vs `distinct: bool` vs richer enum).
- **Q-UNI-013** — Nested aggregation collapse: defer to `34`'s optimizer pass; `23` does no collapsing.
- **Q-UNI-014** — Engine-specific UNION nullability handling: defer to `36`'s adapter trait.

Closed in this rebase: Q-UNI-003 (no per-child override in v1; inference-only), Q-UNI-005 (moot under V1 strict equivalence), Q-UNI-007 (moot under V1 strict equivalence), Q-UNI-008 (`{All, Unique}` v1 roster ratified; future variants tracked as `[TD-UNIONSET-FUTURE-MODES]`), Q-UNI-010 (re-scoped — `CompositionKind` retired; Unionset-level computed Semantics author per `19 §3` like Dataset), Q-UNI-011 (collapsed via Q-UNI-003).

---

**End of document.** Ratified decisions are inline throughout §§1–9; open items live in `[../questions/open/23_questions.md](../../questions/open/23_questions.md)`; historical resolutions in `[../questions/closed/23_questions.md](../../questions/closed/23_questions.md)`.

## Testing Strategy

- VALID_E_2301; COMP_E_2301/2302/2303; COMP_W_2301; PLAN_E_2301/2304; PLAN_W_2302; full roster round-trip per 30 §6.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
