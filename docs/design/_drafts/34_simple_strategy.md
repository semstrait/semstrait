---
status: Pending relocation
purpose: Transitional sidecar holding the `SimpleStrategy` L1–L5 algorithm body extracted from `data-kinds/21_dataset.md §4` during the post-thirteenth-pass cascade rebase (2026-04-30). Pending integration into `apis/34_semstrait_planner.md §<SimpleStrategy>` when the planner doc opens its Strategy chapter.
extracted-from: data-kinds/21_dataset.md §4 (pre-rebase, lines 239–420)
destination: apis/34_semstrait_planner.md §
---

# `SimpleStrategy` L1–L5 plan emission — extract pending relocation to `34`

> **Transitional document.** This file holds the `SimpleStrategy` algorithm body (the layered L1–L5 plan emission contract) extracted from `data-kinds/21_dataset.md §4` during the post-thirteenth-pass cascade rebase (2026-04-30). Per the architectural boundary ratified in this thread, algorithm bodies belong in the planner doc (`apis/34_semstrait_planner.md`), not in variant chapters (`21`–`24`). Variant chapters carry only conceptual / observable invariants.
>
> When `34` opens its Strategy chapter, this content lifts directly into `34 §<SimpleStrategy>`. Until then, this sidecar is the canonical reference. Cross-references in `21` (and elsewhere) point at `34 §<SimpleStrategy>` (forthcoming) — readers needing the algorithm body until that section lands consult this file.
>
> **Vocabulary note.** Content was extracted with minimal edits. The post-thirteenth-pass cascade renamed types (`ColumnMapping` → `SemanticMapping`, `ColumnMappingValue` → `SemanticMappingValue`, `Binding.column_mapping` → `Binding.semantic_mapping` in runtime code, the YAML-level `binding:` block dissolved into `extras.storage:` / `extras.semantic_mapping:` / `extras.catalog:` per `32 §4`); references in the body are updated accordingly. Layer-by-layer logic and `PlanNode` emission rules are unchanged from `21 §4` (pre-rebase). Cross-section references (`§4.X`) have been re-anchored to this sidecar's own §1–§7. References to `21 §2.3` (old `ResolvedSimpleDataKind`) point forward to `33 §<ResolvedDataset>` (forthcoming).

## 1. Overview

`SimpleStrategy` is the planner's resolution strategy for `Dataset` (`SimpleDataKind`). Its output is a canonical `PlanNode` tree with up to five layers:

```
L5  Project          Final output shape; skipped when identity with L4.
L4  Aggregate        GROUP BY semantic Dimensions + declarative Measure
                     decomposition; skipped when Request has no aggregation.
L3  Expression       Computed Dimension / Measure evaluation
                     (model::UnresolvedSemanticExpr
                     substituted per 19 §3.3); skipped when no computed fields.
L2  Rename           Physical -> semantic rename, literal / metadata
                     injection, boundary CAST per 14 §6.4 / 15 §9.1.
L1  Scan             Physical column scan(s) from `extras.storage.paths:` /
                     `extras.storage.tables:` resolved sources.
```

Every Dataset plan is a `Project(Agg(Expression(Rename(Scan(...)))))`-shaped tree with optional layers elided per §7. The tree reads top-to-bottom as "final shape on top, physical scan at the bottom". In ASCII form:

```
PlanNode::Project         (L5, optional)
  PlanNode::Agg           (L4, optional)
    PlanNode::Project     (L3 as Project — computed columns, optional)
      PlanNode::Project   (L2 — rename / cast / literal / metadata)
        PlanNode::Scan    (L1, required; may itself be a Union of N Scans)
```

Filter placement is **in between** layers, not a numbered layer itself. A Request's filter list is decomposed into per-layer predicates by `SimpleStrategy`'s filter-pushdown sub-pass (not detailed here; ratified in `34 §5`): Semantics-referencing filters land above L2 (after rename), column-level filters sink into L1 where possible, aggregation-referencing filters (HAVING-equivalents) land above L4. `PlanNode::Filter` nodes are inserted where appropriate. This sidecar does not pin the exact placement; it only notes that `Filter` is a layer-agnostic node that `SimpleStrategy` interleaves.

**This shape is canonical.** Every Complex kind (`22`–`24`) composes `SimpleStrategy` sub-plans at specific layer boundaries; a Grainset's grain-routing decision is made above L4 of the chosen level's SimpleStrategy; a Unionset's branch assembly wraps each branch's SimpleStrategy at L2 or L3; a Joinset's join emission wraps each member's SimpleStrategy at L4 or L5. This composition discipline is why §1's shape is ratified — downstream composition relies on its regularity.

## 2. L1 — Scan

**Role.** Emit the minimal set of physical columns the Request's downstream layers need, from the Binding's `ResolvedPhysicalSource` list.

**Inputs.**

- `rb: &ResolvedBinding` — the Dataset's resolved binding (cross-ref `33 §<ResolvedDataset>` forthcoming).
- `needed_columns: Vec<ColumnName>` — the set of physical column names required downstream (see below).

**Algorithm.**

1. Compute `needed_columns` as the union of:
  - For every requested physical Dimension (`SemanticMappingValue::Column`): the mapped column name.
  - For every requested computed Dimension (`SemanticMappingValue::Computed`): `PhysicalExpr::referenced_columns` (per `14`'s compile-enriched field).
  - For every requested Measure's expression: its `PhysicalExpr::referenced_columns`, recursively through Metric expansion (Metric composition ratified in `11 §6.3`).
  - For every filter in the Request: its `PhysicalExpr::referenced_columns` (for column-level filter pushdown).
2. For each `src ∈ rb.sources`, emit a `PlanNode::Scan { source_ref: src, projected_columns: needed_columns }`.
3. If `rb.sources.len() > 1`, combine the per-source scans with `PlanNode::Union { branches: Vec<PlanNode::Scan> }`. (Ordering follows `15 §3.6`'s lexical resolution order; I4.)

**Output shape.** A `PlanNode` whose schema is the `needed_columns` physical-type-preserved projection of the Binding's physical surface. No semantic renaming has happened yet — columns carry their physical names.

**Metadata dimension columns.** Metadata-typed Semantics (per `15 §8`) do NOT contribute to `needed_columns` — their values are extracted from `PhysicalSource` metadata at L2 (Rename), not scanned. This is the sidecar-specific rule: the L1 scan is minimal; the L2 layer is where injected columns appear.

**Literal dimension columns.** Likewise not in L1; literals are injected at L2.

**Interaction with `22` Grainset.** When a Dataset is a Grainset level, the Grainset's grain-routing decision picks a single child; that child's `SimpleStrategy` runs a full L1 — no cross-level fan-out at L1. Ratified in `22`.

**Interaction with `23` Unionset.** A Unionset branch is a `Dataset` (or nested Complex). Each branch runs its own L1 independently; the Unionset's union-all happens at L2 / L3 / L4 per the branch's Coverage. Ratified in `23`.

## 3. L2 — Rename (Project)

**Role.** Transform the physical-named L1 output into a semantic-named row shape; inject literal and metadata values; emit boundary `Cast`s where the physical `DataType` disagrees with the declared Semantics type.

**Inputs.**

- L1 output (physical-named).
- `rb.semantic_mapping: ResolvedSemanticMapping` — per `15 §7.2`'s four HashMaps + per-source `Coverage`.

**Algorithm.** Emit `PlanNode::Project { expressions: Vec<NamedPhysicalExpr> }` where each entry in `expressions` is produced per the `ResolvedSemanticMapping` lookup (`15 §7.4`'s `resolve_semantics`):


| `SemanticMappingValue`                           | L2 emission                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Column { name }`                                | `(name_semantic, PhysicalExpr::Column(name_physical))` — a rename.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `Column { name }` with boundary cast (`15 §9.1`) | `(name_semantic, PhysicalExpr::Cast(Column(name_physical), declared_type))` — the `Cast` was wrapped at compile and lives in `ResolvedSemanticMapping.computed` per `15 §7.2`; L2 reads from `computed` for this Semantics.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `Literal { value, data_type }`                   | `(name_semantic, PhysicalExpr::Literal(value))` — materialized as a scalar broadcast.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `Metadata(MetadataDimensionRecipe)`              | `(name_semantic, PhysicalExpr::Literal(per_source_value))` — the **extraction is performed at compile time**, not at plan time (per `15 §5.5` / §10.5). The planner reads each source's pre-resolved `LiteralValue` from `ResolvedPhysicalSource.metadata_values[name_semantic]` (`15 §7.6`) and emits it as a scalar broadcast. In a multi-source scan (§2 step 3), each source's Rename project emits its own per-source literal because the `metadata_values` map's value can differ across sources (the recipe is global to the Binding; the resolved `LiteralValue` is per-source). v1 scope is path-token only (`15 §8.0`). |
| `Computed { expr }`                              | Deferred to L3. L2 passes through the columns `expr.referenced_columns` needs unchanged.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |


**Output shape.** Every non-Computed Semantics is named and typed per the `SemanticInterface`. Computed Semantics are not yet materialized — their `referenced_columns` are carried through as physical columns.

**Per-source variation in multi-source scans.** When L1 emits a `Union` of N `Scan`s, L2's Rename project is emitted per-branch (inside the Union) so metadata values, which vary per source, are per-branch literal constants. The final tree shape:

```
Union(
  Project[rename_source_0](Scan[source_0]),
  Project[rename_source_1](Scan[source_1]),
  ...
)
```

This is the one place in `SimpleStrategy` where per-source divergence bleeds into the plan shape; below L2, the union'd rows are semantic-named and uniform.

**Narrowing-cast warning at compile.** Any `Cast` emitted at L2 was already diagnosed at compile (`COMP_W_0302 ImplicitNarrowingCast` per `15 §9.1`). L2 does not re-emit the warning.

## 4. L3 — Expression (Project)

**Role.** Materialize computed Dimensions and computed Measure-base expressions; `model::UnresolvedSemanticExpr` substitution has already happened at compile (`19 §3.3`), so L3 projects already-resolved `PhysicalExpr` trees.

**Inputs.**

- L2 output (semantic-named row shape, minus computed Semantics).
- `rb.semantic_mapping.computed: HashMap<SemanticsName, PhysicalExpr>` — per `15 §7.2`.

**Algorithm.** For every requested Semantics that maps to `SemanticMappingValue::Computed`:

1. Look up its `PhysicalExpr` in `rb.semantic_mapping.computed[name]` (per `15 §7.4`). The expression carries only `PhysicalLeaf` variants (`19 §3` has substituted every typed semantic leaf during compile per the `14 §3.7` postcondition) and `inferred_type`-annotated.
2. Emit a `PlanNode::Project` entry `(name_semantic, expr)` alongside pass-through of every L2-produced column that downstream layers (L4 / L5 / filters) need.

The resulting `Project` has the shape `(all_pass_through_columns ++ computed_columns)`.

**Cross-reference to `19 §3.3`.** The substitution algorithm is specified there; this sidecar reads the output. The `PhysicalExpr` stored in `rb.semantic_mapping.computed[name]` is semantically equivalent to "inline the computation as of the SemanticInterface's definition at the time of `compile`." No plan-time recomputation.

**Skip rule.** If the Request references no computed Semantics on this Dataset, L3 is elided. See §7.

## 5. L4 — Aggregate

**Role.** Apply `GROUP BY` over the requested semantic Dimensions and evaluate the requested Measures and Metrics via declarative decomposition.

**Inputs.**

- L3 output (or L2 output when L3 is skipped).
- `interface: &SemanticInterface` — to look up per-Measure aggregation shape (`agg: Sum` / `Count` / `Avg` / etc. per `14 §3.2`'s `Aggregation` enum) and per-Metric composition.

**Algorithm.**

1. **GROUP BY clause.** The `GROUP BY` keys are the requested Dimensions (their semantic names). Includes both physical-mapped and computed Dimensions — both are semantic columns by L4's input. For temporal Dimensions with a Request-level grain rollup (e.g. "rollup `ordered_at` to Week"), wrap in `DateTrunc(<sem_col>, Grain::Week)` per `14 §3.2`'s `DateTrunc` variant. The legality of the rollup is shape-gated (cross-ref `21 §6`; full matrix in `17`).
2. **Aggregate expressions.** For each requested Measure `M`:
  - `M` declares `agg: Sum` + `expr: amount_cents` (a physical Measure over one column). L4 emits `PhysicalExpr::Aggregate { aggregation: Sum, expr: Column(amount_cents), distinct: false }` under the name `M`.
  - `M` declares `agg: Count` + `expr` omitted (Count-star-like). L4 emits `Aggregate { aggregation: Count, expr: Literal(1), distinct: false }`.
  - `M` carries a measure-level filter (`filter: expr_F` per `11 §6.2`). L4 emits `Aggregate { aggregation: Sum, expr: Case { when: [{condition: expr_F, result: Column(amount_cents)}], else_expr: None }, distinct: false }` — conditional aggregation via `Case`, equivalent to `SUM(CASE WHEN ... THEN ... END)` at the adapter level.
3. **Metric decomposition.** A requested Metric `X` is recursively expanded into its constituent Measures per `11 §6.3`'s Metric composition rule. Every Measure surfaces as its own L4 aggregate expression; the Metric's composing expression lives in L5 as a post-aggregation Project term.
4. **Re-aggregation over multi-source Scans.** If L1 emitted a Union of N sources and L4's `GROUP BY` keys include a metadata Dimension, the Request may be satisfiable with no re-aggregation — see §5.1 below. In the general case, L4 runs twice conceptually: once per-source (as a per-source partial aggregate, pushed down inside the Union's branches) and once as a "merge aggregate" above the Union. The per-source-partial pushdown is an optimizer decision (`34 §5`); this sidecar ratifies the shape but not the pushdown predicate.

**Output shape.** A `PlanNode::Agg { group_by: Vec<PhysicalExpr>, aggregates: Vec<NamedAggregate> }` whose schema is `(group_by_columns ++ aggregate_columns)`.

**Skip rule.** If the Request asks for no Measures or Metrics (a Dimensions-only Request), L4 is elided — the plan returns the L3 (or L2) Project unchanged. See §7.

### 5.1 Re-aggregation skip when metadata is source-distinguishing

When a multi-source `Binding` is consumed at L4 and the `GROUP BY` includes a metadata Dimension whose values are **distinct per source** (e.g. `path.token: 0` extracting `year_dir = "year=2024"` on source 0 vs `"year=2025"` on source 1, per `15 §8.1`; v1 path-only scope per `15 §8.0`), the per-source partial aggregation is already "complete" — no two sources contribute rows that share a `GROUP BY` key. In that case, the re-aggregation above the Union can be skipped, and the plan shape simplifies to:

```
Union(
  Agg[partial_source_0](... L3 source_0 ...),
  Agg[partial_source_1](... L3 source_1 ...),
  ...
)
```

`SimpleStrategy` checks this predicate via a sub-pass (functional name proposed: `has_source_distinguishing_metadata`). The predicate is **conservative** — if any metadata Dimension in `GROUP BY` yields the same value across two sources, the re-aggregation is retained. Determinism (I4) is preserved: the predicate is a pure function of the `ResolvedBinding`'s per-source metadata literals.

**Lossy re-aggregation warning.** When re-aggregation is not skippable and the per-source partial aggregate uses `COUNT_DISTINCT` or `AVG`, the merge-aggregate cannot recover the exact value without the underlying rows. `SimpleStrategy` falls back to emitting the full aggregation only at the merge layer (no per-source partial). A `PLAN_W_2101 LossyMultiSourceReaggregation` advisory is emitted when this fallback fires and the Binding has multiple sources.

## 6. L5 — Project

**Role.** Produce the final output row shape the Request expects — typically a reordering, a Metric's post-aggregation expression, or an identity pass-through.

**Inputs.**

- L4 output (or the layer below if L4 was skipped).

**Algorithm.**

1. For every requested field (Dimension, Measure, or Metric) in the Request's output list, emit a `PhysicalExpr` under its semantic name:
  - **Dimension** — `PhysicalExpr::Column(sem_name)` (pass-through from L4's `GROUP BY`).
  - **Measure** — `PhysicalExpr::Column(sem_name)` (pass-through from L4's aggregate output).
  - **Metric** — the Metric's composing expression, built over the Measures that L4 computed. E.g. a ratio Metric `revenue_per_user = revenue / user_count` becomes `PhysicalExpr::BinaryOp { op: Divide, left: Column("revenue"), right: Column("user_count") }`.
2. Honor the Request's ordering of output fields.

**Output shape.** `PlanNode::Project { expressions: Vec<NamedPhysicalExpr> }` whose schema matches the Request's expected output row.

**Skip rule (identity elision).** If every L5 projection expression is `PhysicalExpr::Column(name)` and the `name` sequence matches the L4 output schema field-for-field (same names, same order), L5 is elided. The resulting plan ends at L4 (or wherever the prior-layer elision left it).

## 7. Skip rules

Consolidated per-layer skip predicates. Applied top-to-bottom after the logical plan is built; each layer's emission is conditioned on its predicate.


| Layer      | Skip predicate                                                                                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| L5 Project | Every projection expression is `Column(name)` AND `name` sequence matches L4 output schema exactly.                                                                                                    |
| L4 Agg     | Request asks for zero Measures AND zero Metrics (pure Dimensions query).                                                                                                                               |
| L3 Expr    | Request references zero computed Semantics on this Dataset (zero `Computed` entries in `rb.semantic_mapping.computed`).                                                                                |
| L2 Rename  | Never skipped. Even when every Semantics maps to `Column { name }` with no `Cast` and no renaming, L2 is emitted — renaming from physical `ColumnName` to semantic `SemanticsName` is always required. |
| L1 Scan    | Never skipped.                                                                                                                                                                                         |


**Interaction with multi-source fan-out.** The multi-source case (cross-ref `21 §3.2`) does not change skip rules; each per-source branch applies the same predicates uniformly.

**Interaction with filter pushdown.** `PlanNode::Filter` nodes (interleaved between layers per §1) are skipped when the Request carries no filter. When they exist, they are additionally subject to their own pushdown placement — column-level filters sink to the right of L2; aggregation-referencing filters surface above L4. Ratified in `34 §5`.

---

**End of sidecar.** This document retires when `apis/34_semstrait_planner.md §<SimpleStrategy>` lifts the content into the planner-doc canon. At that point this file is deleted; `21`'s cross-refs update from `_drafts/34_simple_strategy.md` to `34 §<SimpleStrategy>`.