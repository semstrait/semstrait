---
id: 23-unionset-plan
status: ratified
version: 2.1
impl: drift
---

# Unionset — Plan-Time Fan-Out

## 4. Plan-Time Observable Behavior

A Unionset's plan-time observable behavior is the realization of `UnionsetStrategy::resolve` (algorithm body in `34 §<UnionsetStrategy>`, parked in `_drafts/34_unionset_strategy.md` pending `34` drafting). This section ratifies the observable invariants an author or consumer can rely on; the algorithm body is referenced only for derivation.

### 4.1 `UnionMode { All, Unique }`

The variant-local enum, deferred from `32 §3.2`:

```rust
#[non_exhaustive]
pub enum UnionMode {
    /// Preserve duplicates across branches.
    /// Engine rendering: `UNION ALL`.
    All,

    /// Deduplicate across branches via the engine's natural sort-merge or
    /// hash-aggregate mechanism. Engine rendering: `UNION` (without `ALL`).
    /// Three-valued-logic NULL semantics apply: rows differing only in
    /// NullFilled positions do NOT dedupe (NULL ≠ NULL); see §4.3.
    Unique,
}
```

`All` is the default — declared via `32 §3.2`'s field comment and surfaced in §2.1 row E. `Unique` corresponds to the engine's natural `UNION` default (sort-merge / hash-aggregate dedupe per the engine survey in `00_overview.md`). `#[non_exhaustive]` per I10; future variants (`Intersect`, `ByName`, etc.) tracked as MINOR additions per `30 §6.3` and Q-UNI-008.

### 4.2 Per-source pre-aggregation — always emitted when Measures requested

When a Request names a Measure or Metric on the Unionset's surface, `UnionsetStrategy` emits **per-branch sub-aggregation** under every branch's `Project` (the seam wrapper). This is unconditional: pre-aggregation is the canonical path, not an optimizer judgment, because unioning raw rows and then aggregating wastes compute relative to per-branch partial-aggregate then merge. The contract holds for both explicit Unionsets and implicit Unionsets (`21 §3.2`'s multi-source `Dataset` case).

The aggregate expressions are placed by their `ExprLayer` (`19 §6.0`): `Aggregate`-layer Measures lift into the per-branch `Agg`; a `PostAggregate`-layer Metric's residual stays above the final aggregation (its constituent aggregates pre-aggregate per branch). Pre-aggregate eligibility and the re-aggregation operator come from **function-derived** `Additivity` in v1 (`19 §6.5`, `14a §3.6.2`) — `Additive` Measures (`Sum`/`Count`/`Min`/`Max`) pre-aggregate then re-aggregate; `NonAdditive` (`Avg`) does not.

A Request that names only Dimensions (no Measures, no Metrics) skips per-branch aggregation — the branches' `Project`s flow directly into the `PlanNode::Union`, with a final post-Union de-duplication implicit in `UnionMode::Unique` or absent in `UnionMode::All`.

### 4.3 NULL-fill projection at the seam

For each composed-surface Semantics `s` that a child does NOT provide (per §3.2), the child's branch `Project` (the wrap-for-union seam) emits `PhysicalExpr::Cast(PhysicalExpr::Literal(LiteralValue::Null), unified_type(s))` at the column position. The `unified_type(s)` is the column-type-reconciled type per §4.4. The typed NULL is carried as a `PhysicalExpr` tree (no SQL string per I1); adapter rendering (`36`) translates to the engine's NULL idiom.

**Branch projection ordering.** The wrap-for-union step orders columns according to a deterministic walk over the Unionset's own `SemanticInterface` (author-declared Semantics order). Every branch projects in the same order — required for `PlanNode::Union` to align inputs positionally.

`**UnionMode::Unique` interaction.** Two branches' rows that agree on every Native column but differ in NullFilled positions are NOT deduplicated under `Unique`, because three-valued-logic treats `NULL = NULL` as UNKNOWN (not TRUE) and `Unique` mode dedupes only when comparison is TRUE. Authors expecting full deduplication across partial-coverage branches see advisory `PLAN_W_2303 UnionsetUniqueThreeValuedNullCollision` (§9).

### 4.4 Column-type reconciliation

Per composed-surface Semantics `s`, the unified branch column type is derived once at compile:

- **Pass-through fast path** (per `14 §5.4`) — if every contributing child's `DataType` for `s` is identical, the unified type is that shared type; no `Cast` needed.
- **Widening** (per `13 §7`) — if contributors' types differ but are pairwise cast-compatible under `13 §7`'s widening rules, the unified type is the LUB. `UnionsetStrategy` wraps non-LUB-typed contributors in `Cast(<col>, <lub>)` inside the seam `Project`. LUB selection follows `14`'s promotion lattice.
- **Incompatible** — if any pair is not cast-compatible under `13 §7`, `COMP_E_2303 UnionsetCrossChildTypeDisagreement` fires at compile (§8).
- **Single contributor** — if exactly one child provides `s`, the unified type is that child's type. **No contributor** — caught earlier as `COMP_E_2302 UnionsetCoverageIncomplete` per §3.2.

**Nullability.** If any contributor reports the column as nullable OR the column is NullFilled in any branch, the unified column is nullable. Authors see advisory `COMP_W_2301 UnionsetNullabilityWidened` (§8) when widening occurs.

### 4.5 Re-aggregation policy — final aggregation conditional

After the per-branch sub-aggregation (§4.2) and the `PlanNode::Union`, a **final aggregation** above the Union is emitted **conditionally**:

- **Required** when the Request names Measures / Metrics AND the planner cannot prove from literals + per-source `Metadata` that branches' GROUP BY keys are disjoint.
- **Elided** when disjointness is provable — per-branch partial aggregates are already correct. Provability requires every branch to project a literal or per-source `Metadata` Dimension whose value is unique across branches (e.g. `source_platform: { literal: "adwords" }` in one branch and `{ literal: "facebook" }` in another).

For Dimensions-only Requests, no aggregation is emitted at any layer (no per-branch sub-agg, no final agg).

**Re-aggregation function inference.** Per-Measure aggregation function over the union'd partials follows the table:


| Per-branch `agg` | Final agg          | Correctness                                                                                           |
| ---------------- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| `Sum`            | `Sum`              | Exact                                                                                                 |
| `Count`          | `Sum`              | Exact (sum of partial counts)                                                                         |
| `Min`            | `Min`              | Exact                                                                                                 |
| `Max`            | `Max`              | Exact                                                                                                 |
| `CountDistinct`  | `Sum`              | **Lossy** (overcounts when branches' row spaces overlap); `PLAN_W_2302` advisory                      |
| `Avg`            | (not decomposable) | `PLAN_E_2304 UnionsetReAggregationInfeasible` — author restructures as a Metric `Sum(num) / Sum(den)` |


The lossy `CountDistinct` case fires `PLAN_W_2302 UnionsetReAggregationLossy` (§9). The `Avg` case is a hard error directing authors to the Metric idiom; mathematically correct cross-branch averaging requires explicit numerator / denominator aggregates.

### 4.6 Coverage-driven branch pruning (advisory)

When Coverage analysis (§3.2) shows that a child contributes only NullFills for every Request-selected Semantics, that branch's contribution is rows of typed NULLs; the final `PlanNode::Union` would inflate row counts with all-NULL rows. `UnionsetStrategy` prunes the branch and emits advisory `PLAN_W_2301 UnionsetBranchPrunable` (§9). The pruned branch's subplan is not constructed.

**Exceptions.**

- A Request whose only Measure is `Count(*)` (i.e. row-counting) is sensitive to NULL rows; pruning would change the result. Pruning suppressed; the all-NULL branch is preserved.
- For `UnionMode::Unique`, pruning is always safe — an all-NULL branch collapses under deduplication to at most one row, which may be removed without altering the dedup result.

If pruning collapses the surviving branch set to zero, `PLAN_E_2303 UnionsetRequestTotallyNullFilled` (§9). This is exceptional; the Coverage-completeness check (§3.2) catches most instances at compile.

### 4.7 Implicit Unionset fan-out — `21 §3.2` cross-link

The mechanism described in §§4.2–4.6 is identical for implicit Unionsets created by multi-source `Dataset` resolution (`21 §3.2`). The differences from the explicit case:

- Children are per-source pseudo-leaves (one per `PhysicalSource` resolved from `extras.storage.paths:` / `extras.storage.tables:`), not user-authored `Nested*` structs.
- Mode is hard-coded `All` (the implicit `mode: all`).
- Coverage discipline is **strict** (`21 §3.2`'s `COMP_E_2106` rather than `23 §8`'s NullFill-tolerant inference).
- Identity is a content-derived hash-id (per `33 §<implicit-unionset-id>`, forthcoming) rather than an author-declared `name`.

The plan shape is structurally identical: per-source `Project` (with literals + per-source `Metadata` driving disjointness), `PlanNode::Union { mode: All }`, and conditional final-aggregation. `21 §10` carries the worked example for the implicit case.

## Testing Strategy

- UnionMode projection to UNION/UNION ALL; per-branch pre-agg when Measures requested; NULL-fill Cast(Null, unified_type); LUB widening per 13 §7; PLAN_W_2301 Coverage-driven branch pruning.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
