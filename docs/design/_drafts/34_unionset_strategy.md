---
status: Pending relocation
purpose: Transitional sidecar holding the `UnionsetStrategy` algorithm body extracted from `data-kinds/23_unionset.md §4` during the post-thirteenth-pass cascade rebase (2026-05-03). Pending integration into `apis/34_semstrait_planner.md §<UnionsetStrategy>` when the planner doc opens its Strategy chapter.
extracted-from: data-kinds/23_unionset.md §4 (pre-rebase) — refactored under the post-thirteenth-pass discipline (per-branch sub-aggregation as canonical path; v1 inference-only Coverage; CompositionKind / ChildCoverageOverride retired)
destination: apis/34_semstrait_planner.md §<UnionsetStrategy>
---

# `UnionsetStrategy` algorithm body — extract pending relocation to `34`

> **Transitional document.** This file holds the `UnionsetStrategy` algorithm body extracted from `data-kinds/23_unionset.md §4` during the post-thirteenth-pass cascade rebase (2026-05-03). Per the architectural boundary ratified in this thread, algorithm bodies belong in the planner doc (`apis/34_semstrait_planner.md`), not in variant chapters (`21`–`24`). Variant chapters carry only conceptual / observable invariants.
>
> When `34` opens its Strategy chapter, this content lifts directly into `34 §<UnionsetStrategy>`. Until then, this sidecar is the canonical reference. Cross-references in `23` (and elsewhere) point at `34 §<UnionsetStrategy>` (forthcoming) — readers needing the algorithm body until that section lands consult this file.
>
> **Vocabulary note.** Content was extracted with refactoring per the post-thirteenth-pass cascade: `ColumnMapping` → `SemanticMapping`, `ColumnMappingValue` → `SemanticMappingValue`, `ResolvedColumnMapping` → `ResolvedSemanticMapping`, runtime `column_mapping` → `semantic_mapping`, YAML `binding:` block dissolved into `extras.storage:` / `extras.semantic_mapping:` / `extras.catalog:` per `32 §4`. **Retired concepts**: `CompositionKind` (artifact, not authored); `ComposedSemanticInterface` as a Unionset-level concept (survives Joinset-only per `24`); `ChildCoverageOverride { provides }` and the YAML `coverage:` block (no per-child override in v1; Coverage is inference-only); `DataKindRef`-based child references (children are inlined `Nested*` structs per `32 §3.2`). **Reframed**: per-branch sub-aggregation is the canonical path (always emitted when Measures requested), not an optimizer pushdown; final aggregation above the Union is conditional on disjointness inference (was: terminal Aggregate the canonical, skipped in 3 cases). `UnionMode::Distinct` renamed to `UnionMode::Unique`. References to `23 §<X>` point at the post-rebase `23` sections.

## 1. Overview — algorithm shape

`UnionsetStrategy` is the planner's resolution strategy for any DataKind whose resolved kind is `Unionset` — both **explicit** Unionsets (user-authored `unionsets:` block) and **implicit** Unionsets (planner-created from multi-source `Dataset` per `21 §3.2`). The algorithm produces a sub-tree rooted at `PlanNode::Union { mode: UnionMode, inputs: Vec<PlanNode> }`, with per-child branches inside each `inputs` entry and an optional `PlanNode::Agg` wrapper above the Union for conditional final-aggregation.

Canonical observable shape (when Measures are requested):

```text
PlanNode::Agg                  (final aggregation; conditional — see §6)
  PlanNode::Union              (mode: All | Unique)
  ├─ PlanNode::Project         (branch 0 — wrap_for_union: rename + cast + NULL-fill + per-source literals)
  │  └─ PlanNode::Agg          (branch 0 — per-branch sub-aggregation; ALWAYS emitted when Measures requested)
  │     └─ <child 0's Strategy output>
  ├─ PlanNode::Project         (branch 1 — same shape)
  │  └─ PlanNode::Agg
  │     └─ <child 1's Strategy output>
  └─ ...
```

When the Request names only Dimensions (no Measures, no Metrics), the per-branch `Agg`s and the final-aggregation `Agg` are both elided; the shape is `Union > Project > <child Strategy output>` per branch.

`<child N's Strategy output>` is produced by the child's own Strategy: a `NestedDataset` runs `SimpleStrategy` (per `_drafts/34_simple_strategy.md`); a `NestedGrainset` runs `GrainsetStrategy` (per `22 §<GrainsetStrategy>`, forthcoming); a `NestedJoinset` runs `JoinsetStrategy` (per `24 §<JoinsetStrategy>`, forthcoming). `UnionsetStrategy` does not reach inside child plans; it wraps each child's output at the seam `Project`.

**Filter placement** is interleaved between layers per the planner's filter-pushdown sub-pass (`34 §7.6`): row-level filters on a Semantics that all branches share land below per-branch `Agg`; aggregation-referencing filters land above the final `Agg`; per-branch-specific filters land inside the relevant branch only. This sidecar does not pin exact placement.

## 2. Per-child Request narrowing

The Unionset-level Request specifies fields and filters in terms of the Unionset's own `SemanticInterface`. For each child `i`, `UnionsetStrategy` rewrites the Request into the child's own scope:

1. **Filter-on-coverage.** Fields the Request `select`s that child `i` does not provide (per the Coverage inference of `23 §3.2`) are dropped from the child's narrowed Request. The child's Strategy does not scan for them; NULL-fill is injected at the wrap-for-union seam (§3).
2. **Preserve grouping.** Any Dimension the Request groups by (or rolls up) is either:
   - covered by child `i` — requested from the child, with the child responsible for projecting / aggregating; or
   - NullFilled by child `i` — the child is not asked for it; the seam projects `Cast(Null, T)`; the per-branch sub-`Agg` (§4) groups on the NULL value (semantically: "this branch contributes rows whose dimension value is unknown").
3. **Filter routing on NullFilled fields.** A Request-level filter referencing a Semantics NullFilled by child `i` is logically unsatisfiable on that child's rows (NULL three-valued logic). Two policies:
   - If the filter is the SOLE reason the child contributes nothing → branch pruned (§7).
   - Otherwise → `Filter(WHERE FALSE)` injected into the narrowed child Request (the child's Strategy may further optimize).

The narrowed Requests for all branches are computed once at the start of `UnionsetStrategy::resolve`; per-branch resolution proceeds in the canonical child sequence (per `23 §3.1`: `datasets:` → `grainsets:` → `joinsets:`, each in YAML order).

## 3. `wrap_for_union` — seam projection

For each child branch `i`, after the child's Strategy produces its subplan, `UnionsetStrategy` wraps the output in a `PlanNode::Project` (the **seam**). The seam emits one column per Semantics on the Unionset's own `SemanticInterface`, in the deterministic walk order ratified in `23 §4.3`. Per Semantics `s`:

| Coverage of `(child_i, s)` | Seam projection expression |
|---|---|
| Provided Natively (child column `c` mapped to `s` via the child's `extras.semantic_mapping`) | `PhysicalExpr::Column("c")` aliased AS `s` |
| Provided Natively with type widening (per §5) | `PhysicalExpr::Cast(Column("c"), unified_type(s))` aliased AS `s` |
| Provided as a literal in the child's mapping (`{ literal: V }` per `15 §10.4` `SemanticMappingValue::Literal`) | `PhysicalExpr::Literal(V)` aliased AS `s` (drives disjointness — see §6) |
| Provided as per-source `Metadata` (via the child's Dimension `type: { metadata: ... }` recipe per `15 §10.4` step 4.0) | `PhysicalExpr::Literal(per_source_value)` aliased AS `s` (drives disjointness) |
| Provided as a Computed expression in the child's mapping (`SemanticMappingValue::Computed(PhysicalExpr)`) | The child's Strategy emits the Computed at L3 (per `_drafts/34_simple_strategy.md §4`); the seam pass-throughs the resulting column as `Column("s")` |
| **NullFilled** (child does NOT provide `s`) | `PhysicalExpr::Cast(Literal(Null), unified_type(s))` aliased AS `s` |

The typed NULL is carried as a `PhysicalExpr` tree (no SQL string per I1). Adapter rendering (`36`) translates to engine-native NULL idioms (`CAST(NULL AS INTEGER)` in ANSI; `lit(null).cast("int")` in DataFusion; etc.).

**Column ordering in the seam.** Every branch emits columns in the same order — required for `PlanNode::Union` to align inputs positionally. The order is the Unionset's own `SemanticInterface` author-declared Semantics order (per `23 §4.3`).

**Per-source variation.** The seam is the only place per-branch values diverge below the `PlanNode::Union` — literals and per-source `Metadata` produce different scalar broadcasts per branch; below the seam (inside each child's Strategy output), columns are uniformly named per the child's own SemanticInterface.

## 4. Per-branch sub-aggregation — always when Measures requested

When the Request names a Measure or Metric on the Unionset's surface, `UnionsetStrategy` emits a `PlanNode::Agg` **between** the child's Strategy output and the seam `Project`. This is unconditional (per `23 §4.2`'s "per-source pre-aggregation always emitted when Measures requested" contract).

**Per-branch `GROUP BY`**: every Dimension in the Request's `select` (or `group_by`) that child `i` provides Natively or via Computed/Literal/Metadata. NullFilled Dimensions do NOT appear in the per-branch `GROUP BY` (they would group every row under NULL; a no-op).

**Per-branch aggregates**: per Measure `M` in the Request:

- `M` declares `agg: Sum + expr: c` → `Aggregate { aggregation: Sum, expr: Column(c), distinct: false }` named after `M`.
- `M` declares `agg: Count` → `Aggregate { aggregation: Count, expr: Literal(1), distinct: false }`.
- `M` carries a measure-level filter (`filter: expr_F` per `18 §2`) → conditional aggregate via `Case`: `Aggregate { aggregation: Sum, expr: Case { when: [{ condition: expr_F, result: Column(c) }], else_expr: None }, distinct: false }`.

**Metric decomposition**: a requested Metric `X` is recursively expanded into its constituent Measures per `18 §2`'s Metric composition. Each constituent Measure surfaces as its own per-branch aggregate; the Metric's composing expression is materialized at the final-aggregation layer (§6) — or, when final-aggregation is elided, at a post-Union `Project`.

**Rationale for unconditional emission.** Unioning raw rows and aggregating later spends compute on per-source rows that will collapse anyway; per-branch sub-agg guarantees the union's input row count is bounded by `branches × |distinct group keys|` rather than `branches × |raw rows|`. The disjointness-elision optimization (§6) keys off this bound: when per-branch aggregates are already disjoint, the final aggregation is unnecessary.

## 5. Column-type reconciliation

The unified column type for each composed-surface Semantics `s` is computed once at compile (§5.1 algorithm body); plan-time consults the pre-computed type.

### 5.1 `unified_type(s)` derivation

1. Compute `contribs(s) = { (i, t_i) | child i provides s with `DataType` `t_i` }`.
2. **No contributors** (`|contribs(s)| == 0`) — caught earlier by `COMP_E_2302 UnionsetCoverageIncomplete` (`23 §8`); never reaches plan-time.
3. **Single contributor** — unified type is `t_0`.
4. **Multiple contributors**:
   - **Pass-through fast path (per `14 §5.4`)** — if all `t_i` are identical, unified type is the shared type; no `Cast` needed.
   - **Widening (per `13 §7`)** — if all `t_i` are pairwise cast-compatible, unified type is the LUB. Promotion lattice per `14`. Tie-breaker: first contributor's type wins.
   - **Incompatible** — `COMP_E_2303 UnionsetCrossChildTypeDisagreement` at compile.

**Nullability widening.** If any contributor's column is nullable OR any branch NullFills `s`, the unified column is nullable. Compile emits `COMP_W_2301 UnionsetNullabilityWidened` advisory.

### 5.2 Cast emission at the seam

When `unified_type(s) ≠ t_i` for child `i` providing `s` Natively, the seam `Project` (§3) wraps the column read in `Cast(Column(c), unified_type(s))`. The single seam `Project` carries both casts and NULL-fills uniformly.

## 6. Final aggregation — conditional on disjointness

After per-branch sub-aggregation (§4) and seam projection (§3), `UnionsetStrategy` emits the `PlanNode::Union` (§7). Above the Union, a **final `PlanNode::Agg`** is emitted **conditionally**:

### 6.1 Disjointness predicate

For each Dimension `d` in the Request's `GROUP BY`, `is_source_distinguishing(d)` returns true if every contributing branch projects a value for `d` that is unique across branches. The predicate is satisfied when, for every pair of branches `(i, j)`, the value of `d` in branch `i` differs from the value in branch `j`. V1 scope:

- **`Literal { value }`** in a child's `extras.semantic_mapping` → branch's value is `value`; comparison is straightforward equality (per `15 §7.4`'s `SemanticMappingValue::Literal`).
- **`Metadata(MetadataDimensionRecipe)`** with path-token extraction (per `15 §8.1` v1 scope) → branch's value is the compile-resolved `LiteralValue` from `ResolvedPhysicalSource.metadata_values` (per `15 §10.5`).

V1 does NOT consider Computed Dimensions as source-distinguishing, even when their expression is a pure function of literals/metadata (see `Q-UNI-005` deferral history; `34 §<implicit-union>` may extend post-v1 per the question now closed in `23`).

If `is_source_distinguishing(d)` is true for AT LEAST ONE Dimension in the `GROUP BY`, the entire branch product is provably disjoint — no two branches contribute rows sharing a `GROUP BY` key. Per-branch partial aggregates are already correct; **final aggregation is elided**.

### 6.2 Required final aggregation

When no `GROUP BY` Dimension is source-distinguishing, branches MAY share `GROUP BY` keys. Per-branch partial aggregates need to be merged via the re-aggregation function table (per `23 §4.5`):

| Per-branch `agg` | Final `agg` | Correctness |
|---|---|---|
| `Sum` | `Sum` | Exact |
| `Count` | `Sum` | Exact |
| `Min` | `Min` | Exact |
| `Max` | `Max` | Exact |
| `CountDistinct` | `Sum` | Lossy (`PLAN_W_2302`) — overcounts when branches' raw-row spaces overlap |
| `Avg` | (not decomposable) | `PLAN_E_2304 UnionsetReAggregationInfeasible` — author restructures as Metric `Sum(num) / Sum(den)` |

The final `Agg`'s `group_by` is the Request's full Dimension set (Native + NullFill positions); aggregates are the re-aggregation entries for each requested Measure / Metric-constituent.

### 6.3 Metric materialization

Requested Metrics are decomposed into constituent Measures at per-branch sub-agg (§4). When final aggregation is REQUIRED, the Metric's composing expression (e.g. `revenue / user_count` for a ratio Metric) is materialized at a `PlanNode::Project` ABOVE the final `Agg`. When final aggregation is ELIDED, the Metric's composing expression is materialized at a post-Union `Project` below where the elided final `Agg` would have been.

## 7. `PlanNode::Union` emission

The outermost shape of `UnionsetStrategy`'s output:

```rust
PlanNode::Union {
    mode: UnionMode,           // from UnionsetBody.mode (per 32 §3.2)
    inputs: Vec<PlanNode>,     // per-branch wrapped subplans, in canonical child sequence
}
```

The exact field roster of `PlanNode::Union` is `35`'s ratification (`Q-UNI-012` deferred to `35`). `UnionsetStrategy` populates `mode` directly from the resolved `UnionsetBody.mode`; for implicit Unionsets (per `21 §3.2`), `mode = UnionMode::All` is hard-coded.

**Single-branch post-prune.** When Coverage-driven branch pruning (§7) reduces `inputs.len()` to 1, the planner skips the `PlanNode::Union` entirely and the surviving branch's wrapped subplan flows directly into the (conditional) final `Agg` or output. (`Q-UNI-006` Round-1 default per `23`.)

**Zero-branch post-prune.** When pruning collapses all branches, `PLAN_E_2303 UnionsetRequestTotallyNullFilled` (per `23 §9`).

## 8. Coverage-driven branch pruning (advisory)

For each branch `i`, after computing the narrowed Request (§2): if every Semantics in the narrowed `select` is NullFilled by child `i`, the branch's only contribution to the Union is rows of typed NULLs. `UnionsetStrategy` emits `PLAN_W_2301 UnionsetBranchPrunable` (per `23 §9`) as advisory and prunes the branch — its subplan is not constructed.

**Exceptions (pruning suppressed)**:

- The Request's only Measure is `Count(*)` or equivalent (row-counting); NULL rows DO contribute to row count. Pruning would change result semantics.
- For `UnionMode::Unique`, pruning is always safe — the all-NULL branch collapses to ≤ 1 row under deduplication; advisory still emitted.

## 9. Algorithm sketch — pseudocode

```text
UnionsetStrategy::resolve(request, resolved_unionset, manifest):
    # §2 — per-child Request narrowing
    coverage = resolved_unionset.coverage_index             # compile-built per `23 §3.2`
    per_child_requests = [
        narrow_request(request, child_i, coverage) for child_i in resolved_unionset.children
    ]

    # §8 — Coverage-driven branch pruning
    surviving_indices = [
        i for i in 0..N if not is_pruneable(per_child_requests[i], request, mode)
    ]
    if surviving_indices.is_empty():
        return Err(PLAN_E_2303 UnionsetRequestTotallyNullFilled)

    # Per-branch resolution + seam wrapping
    branches = []
    for i in surviving_indices:
        child_strategy = dispatch_strategy(manifest, resolved_unionset.children[i])
        child_subplan = child_strategy.resolve(per_child_requests[i], ...)

        # §4 — per-branch sub-aggregation (always when Measures requested)
        if request.has_measures():
            child_subplan = wrap_per_branch_aggregate(child_subplan, per_child_requests[i])

        # §3 — seam projection (rename, cast, NULL-fill, literals)
        wrapped = wrap_for_union(child_subplan, i, resolved_unionset, request)
        branches.push(wrapped)

    # §7 — single-branch shortcut
    if branches.len() == 1:
        union_node = branches[0]
    else:
        union_node = PlanNode::Union {
            mode: resolved_unionset.body.mode,            # `UnionMode::All` for implicit
            inputs: branches,
        }

    # §6 — conditional final-aggregation
    if request.has_measures() and !is_disjoint(request.group_by, branches):
        return Ok(emit_final_agg(union_node, request, resolved_unionset))
    elif request.has_metrics_post_aggregation():
        return Ok(emit_metric_project(union_node, request))
    else:
        return Ok(union_node)
```

`is_disjoint` is the §6.1 predicate; `wrap_per_branch_aggregate` builds the per-branch `PlanNode::Agg` per §4; `wrap_for_union` emits the seam `Project` per §3; `emit_final_agg` builds the final `PlanNode::Agg` per §6.2 + Metric materialization per §6.3.

---

**End of sidecar.** This document retires when `apis/34_semstrait_planner.md §<UnionsetStrategy>` lifts the content into the planner-doc canon. At that point this file is deleted; `23`'s cross-refs update from `_drafts/34_unionset_strategy.md` to `34 §<UnionsetStrategy>`.
