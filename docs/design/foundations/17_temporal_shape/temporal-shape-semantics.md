---
id: 17-temporal-shape-semantics
status: partially ratified
version: 1.0
impl: partially implemented
---

# Temporal Shape Semantics

> **Struct ownership (2026-04-17 consolidation).** The v1 authoring-layer `TemporalShape` struct, `TemporalShapeKind` enum, per-variant `*Body` structs (`TimeseriesBody`, `EventsBody`, `SnapshotBody`, `ScdBody`), and the trimmed `ScdType` v1 roster `{Type1, Type2}` are ratified in [`18_entities.md §3`](../18_entities.md#3-temporalshape). This doc owns the *planner-level semantics* on top — shape-propagation rules, grain interactions, additivity advisories, shape-gated composition, and the post-v1 `AsOf` forward-reference design. Key deltas the reader should keep in mind while reading body prose:
>
> - **`TemporalShape` is a struct** (`{ kind: TemporalShapeKind, grain: Option<Grain> }`), not a bare enum. Per-variant payload lives in `*Body` structs (see `18 §3.1`). Body prose below that treats `TemporalShape` as a flat enum is pre-consolidation vocabulary.
> - **`ScdType` v1 roster is `{Type1, Type2}`** per `18 §3.3`. The Kimball `Type0`–`Type6` material in §2 below is retained as post-v1 forward-reference; Type0 / Type3 / Type4 / Type5 / Type6 are NOT in the v1 authoring surface.
> - **`ScdBody` is flat** (`{ scd_type, valid_from, valid_to }`) — no per-subtype sub-struct, no `current_flag_dim` / `prior_value_dim` / `history_data_kind_ref` in v1.
> - **YAML shape** — `extras.temporal.<variant>: { <fields> }` + sibling `grain:` per `18 §3.2`.
> - **`grain:` rules** — required on leaf `Dataset`, forbidden on `ComplexDataKind` (SR-E-6 / SR-E-7 in `18 §11`); Grainset children each author their own (SR-E-8).
> - **`JoinType::AsOf`** — descoped for v1; `18 §2.3` roster is `{Inner, Left, Right, Full}`. The `AsOf` design in §5 below remains the forward-reference spec.
>
> **Status:** partially ratified — §3 / §4 shape-propagation rules, §7 additivity interactions, §8 shape-gated composition, and §9 diagnostic catalog are authoritative. §2 SCD taxonomy (full Kimball treatment), §5 AsOf design, and §6 Request.temporal are forward-reference / post-v1. Round-1 drafting open items live in `questions/open/17_questions.md`.

This document defines the TemporalShape classification vocabulary (Timeseries, Events, Snapshot, SCD) and its declaration site, along with interaction rules for Grain, JoinType::AsOf (post-v1 deferred), Request.temporal blocks, and Additivity warnings. For shape-gated composition constraints and the validate/compile/plan Precondition catalog, see [shape-composition-and-validation.md](shape-composition-and-validation.md).

## [CONTRADICTION-FOUND] — code-range coordination with `30 §6.2`

This document assigns error codes in the doc-aligned range `*_E_1700`–`*_E_1799` (and `*_W_1700`–`*_W_1799`) per the authoring-time brief. `30 §6.2` as currently ratified allocates subsystem-level ranges that **do not extend past 0999** — `VALID_E` tops out at `0999`, `COMP_E` at `0499`, `PLAN_E` / `PLAN_W` at `0699`. The 17NN allocation sits outside every current subsystem range.

**Two reconciliations are possible** (both are MINOR per `30 §2`):

1. **Extend `30 §6.2`'s subsystem ranges.** Add a doc-aligned sub-range convention (`NN00`–`NN99` reserved for the doc that ratifies the error, where `NN` is the doc number), and widen `VALID_E` / `COMP_E` / `PLAN_E` / `PLAN_W` overall ranges to `0001`–`9999` to accommodate. Bumps `30 §6.2`'s table.
2. **Re-allocate within `30 §6.2`'s current ranges.** `16` used `0400`–`0499` / `0500`–`0599`; the next free 100-ranges in the ratified allocation are `VALID_E_0500`–`0999` (structurally claimed for keys + internal but most unused), `COMP_E` has no free 100-block left (0400-0499 taken), `PLAN_E_0600`–`0699`, `PLAN_W_0600`–`0699`. Under this reconciliation, `17` would claim `VALID_E_0500`–`0599` and `PLAN_E_0600`–`0699` and `PLAN_W_0600`–`0699`; `COMP_E` would need to either borrow from `VALID_E_0400`–`0499` unused slots or extend.

This doc adopts **Option 1** (doc-aligned 17NN allocation) on the authoring-time brief, and records the coordination task as Q-TEMPORAL-001 in `questions/open/17_questions.md`. If `30 §6.2` is instead revised to adopt Option 2, every `*_E_17NN` reference in this doc is re-homed to its Option-2 code; error semantics are unchanged.

`16 §14`'s `COMP_E_04xx` / `VALID_E_04xx` / `PLAN_E_05xx` / `PLAN_W_05xx` allocations are **not affected** — they remain authoritative for composition.

---

## 1. Purpose and Scope

### 1.1 What `17` ratifies

`17` is the canonical specification for the **historization axis** of a `DataKind`: *how* the data records time, *which* Dimension carries the time axis, and *what* the planner is permitted (or required) to infer from that classification. It is the eighth and final foundations document in the `1x` series; every `2x` data-kind doc consumes `17`'s vocabulary (`20 §…` — `25 §…`).

Concretely, `17` ratifies:

- **§2** — The `TemporalShape` classification vocabulary. Four top-level variants — `Timeseries`, `Events`, `Snapshot`, `Scd` — with per-variant payloads identifying the time axis and auxiliary fields. SCD subtypes; the v1 enum roster is `{Type1, Type2}` (per `18 §3`), with Type0 / Type3–Type6 documented as post-v1 forward-reference in §2.2.
- **§3** — The declaration site. `TemporalShape` is authored on a `SimpleDataKind` alongside its `Binding` and its declared `Grain`; `ComplexDataKind`s do not carry their own `TemporalShape` but may constrain or inherit their constituents' shapes per §8.
- **§4** — The `TemporalShape × Grain` interaction matrix. Which shapes roll up freely across `Grain`, which have a fixed source grain (`Snapshot`), which have no intrinsic grain (`Scd`).
- **§5** — The `JoinType::AsOf` variant ratification. Shape, semantics, and the per-shape-pair legality matrix gating which combinations admit `AsOf` traversal. Implementation is DEFERRED (planner does not yet emit `AsOf` joins); the variant is ratified at the IR vocabulary level per `16 §4.4.2`.
- **§6** — The `Request.temporal` block. As-of timestamp, time-range override, default-current behavior. Vocabulary ratified; planner consumption DEFERRED.
- **§7** — The `TemporalShape × Additivity` interaction. Two-axis independence (per `11 §7.2`) plus the concrete advisory-warning roster when the two appear inconsistent.
- **§8** — Shape-gated composition rules refining `16 §11`. Which shape combinations admit implicit composition, which require explicit `AsOf` anchoring, which forbid composition entirely in Round 1.
- **§9** — The structural-Precondition catalog by stage (`validate`, `compile`, `plan`), with code allocations in `VALID_E_1700`–`1799` / `COMP_E_1700`–`1799` / `PLAN_E_1700`–`1799` / `PLAN_W_1700`–`1799` per the [CONTRADICTION-FOUND] block above.
- **§10** — The DEFERRED-items roster: the closed list of planner behaviors whose vocabulary and model surface this doc ratifies but whose implementation lands in a later milestone.

### 1.2 What `17` does NOT ratify (forward-refs)

- The YAML surface for `temporal_shape:` authoring (subtype discriminator keys, alias policy) — `32`.
- The per-engine emission of `AsOf` / snapshot-selection / SCD-window predicates — `registry/temporal_shape_mapping.md` and per-adapter docs (`36`).
- Per-DataKind-variant planner strategies that consume `TemporalShape` — `20`–`25`, particularly `22` (Grainset cadence alignment) and `24` (Joinset as-of anchors).
- Non-temporal grain extensions (geographic, entity rollup) — deferred per `13 §3.4` (`TD-GRAIN-NON-TEMPORAL`); `17`'s vocabulary is temporal-only.
- The full `Request.temporal` planner algorithm (how default-current resolves when multiple constituents disagree on "current") — deferred per §6.5, gated on `34`.

### 1.3 Green-field vs existing-code stance

Per I9, the design tree is the "after" state. The existing crate `semstrait-model::types::temporal` already carries a `TemporalHistorization` enum with `Timeseries` / `Events` / `Snapshot` / `Scd` variants and an `ScdType` enum covering `Type1` / `Type2(ScdVersionedColumns)` / `Type3` / `Type4` / `Type5(ScdVersionedColumns)` / `Type6(ScdVersionedColumns)`. This document supersedes that symbol in three ways:

1. **Rename** `TemporalHistorization` → `TemporalShape` per `00 §4.3`'s banned-terms table. The rename is tracked in `docs/TECH_DEBT.md`; the existing code keeps the legacy symbol behind a `#[deprecated]` until the cutover.
2. **Add `Type0`**. Kimball's `Type 0` (retain original — no updates ever) is absent in the existing enum; `17` documents the full `Type0`–`Type6` semantics; the v1 enum roster (`18 §3`) is `{Type1, Type2}`, and the remaining subtypes are forward-reference additions, non-breaking per I10.
3. **Lift `valid_from` / `valid_to` payloads**. Existing code carries window columns inside `Type2` / `Type5` / `Type6` variants. `17` keeps per-subtype payloads (§2.2), ratifying the **shape** of each variant's payload rather than a flat single-struct carrier — this is a refinement on the existing enum, not a structural change in Round 1. Q-TEMPORAL-002 records the "flat vs per-subtype" alternative.

None of these changes block adoption; all are MINOR per `30 §2`.

### 1.4 Explicit DEFERRED-for-planner-support status

Per `00 §4.1`'s `TemporalShape` row and `00 §4.1`'s `AsOf` `JoinType` row, this document **ratifies vocabulary and model surface** but **defers planner-side support** for the features listed below. §10 enumerates the complete DEFERRED roster; callers must not read this doc as a contract that the Round-1 planner implements the behaviors described. Callers **can** read this doc as a stable vocabulary and model-layer specification — every `#[non_exhaustive]` enum, every declared field, every advisory-warning code allocated here is authoritative for crate-public-surface docs (`32`, `33`) and for downstream per-DataKind strategy docs (`20`–`25`).

DEFERRED status does **not** mean "ignore the vocabulary until the planner catches up." `parse`, `validate`, and `compile` already operate on `TemporalShape` today (the existing `TemporalHistorization` is parsed end-to-end through compile); this doc strengthens the `validate` / `compile` Preconditions (§9) to match the ratified semantics. DEFERRED items are planner behaviors only.

### 1.5 Invariants `17` directly upholds

- **I1 — canonical layer.** `TemporalShape`'s identifying Dimension references are `SemanticsName`s, never physical column names. The `SemanticsName → physical-column` resolution happens in `15 §4` via `ColumnMapping`; `17`'s rules operate on the canonical-layer names.
- **I4 — determinism.** Shape-gated planner decisions are fully determined by `(Model, Request)`; no non-deterministic tie-breaking. When shape-propagation through composition is ambiguous (e.g. a `Unionset` of constituents with divergent shapes), the rule is "emit `COMP_E_17NN`, do not pick a shape."
- **I5 — resolution at compile time.** `temporal:` references to Dimensions are resolved during `compile`; the SemanticManifest carries a `TemporalShape` on the leaf `DataKind.temporal` field (`33 §8B`) with each identifying-Dimension `SemanticsName` resolved to its semantic `EntityId`. This is the only resolution applied — temporal is otherwise propagation, so the manifest type carries no `Resolved*` prefix.
- **I8 — SemanticManifest planner-complete.** Every shape-aware planner decision is table-driven from SemanticManifest state; no YAML-time logic survives into `plan`.
- **I10 — non-exhaustive extensibility.** `TemporalShape`, `ScdType`, and `JoinType::AsOf`-extended `JoinType` are all `#[non_exhaustive]`. Variants may grow (SCD Type 7, non-temporal shapes, bi-temporal shapes) without MAJOR bumps.
- **I12 — first-class diagnostics.** Every Precondition and advisory in §9 carries a stable `*_E_17NN` / `*_W_17NN` code.

---

## 2. `TemporalShape` — the classification vocabulary

### 2.1 The four top-level variants

`TemporalShape` classifies a `SimpleDataKind` by how its rows record time. Four variants, each with a characteristic identifying Dimension (the Semantics name that carries the time axis) and variant-specific auxiliary fields.

| Variant | Identifying Dim | Characteristic semantics | Grain posture (§4) |
|---|---|---|---|
| `Timeseries` | `occurred_at_dim` | Dense regularly-spaced observations at a known cadence. Row existence for every `(entity, time)` combination at the cadence. | Rolls up freely along its declared `Grain`. |
| `Events` | `occurred_at_dim` | Sparse discrete occurrences; rows exist only where something happened. No regularity assumption. | Rolls up via bucket-then-aggregate; declared `Grain` is the coarsest meaningful bucket. |
| `Snapshot` | `snapshotted_at_dim` | Periodic full-state capture. Each snapshot is a complete as-of-time image of the underlying entity population; rolling up means choosing which snapshot(s) to read. | Fixed source grain = snapshot cadence; cannot sub-aggregate below it. |
| `Scd` | per-subtype (see §2.2) | Slowly-changing Dimension table — one or more rows per entity across the entity's history, with subtype-specific preservation rules. | No intrinsic grain; as-of anchoring is the composition mechanism. |

The variants are mutually exclusive per `SimpleDataKind`: a single `SimpleDataKind` has exactly one `TemporalShape` (or none; `TemporalShape` is optional authoring-wise, with "not stated" meaning "no shape declared and no shape-aware planner behavior" — §3.3).

### 2.2 `ScdType` — the full Kimball taxonomy

`Scd` carries a nested `ScdType` discriminator. The full Kimball Type 0–Type 6 set is ratified at this level. Ratifying the full set at vocabulary time is deliberate: subtypes are a closed, well-known taxonomy (no Type 2.5 exists), and enumerating all seven at once prevents MINOR churn as callers encounter uncommon subtypes.

| Subtype | Canonical description | History-preserving? | Window payload? |
|---|---|---|---|
| `Type0` | Retain original. Rows, once written, never update. New entities add new rows; existing rows are immutable. | Yes (by refusal-to-update). | No. |
| `Type1` | Overwrite. Each entity has exactly one row; updates overwrite in place. No history preserved. | No. | No. |
| `Type2` | Full history via valid windows. Each change inserts a new row with `valid_from` / `valid_to` covering its validity period; prior rows' `valid_to` is updated to the change moment. | Yes. | `valid_from_dim`, `valid_to_dim`, optional `current_flag_dim`. |
| `Type3` | Limited history via prior-value column. Each row carries one extra column holding the previous value; change overwrites current, shifts previous into the prior column. | Partial (one generation). | `prior_value_dim`. |
| `Type4` | History table. Current state lives on the primary table; historical versions live in a separate history table referenced by the shape payload. | Yes (externalized). | `history_data_kind_ref`. |
| `Type5` | Type 4 + mini-dimension. Adds an outrigger mini-dimension to Type 4's history-table shape, typically for high-cardinality changing attributes. Window payload as in Type 2. | Yes. | `valid_from_dim`, `valid_to_dim`, `mini_dim_ref`. |
| `Type6` | Hybrid. Combines Type 1 (current-value overwrite) + Type 2 (full history) + optionally Type 3 (one prior value) in a single row set. The row carries both a history window and a current-value column. | Yes. | `valid_from_dim`, `valid_to_dim`, optional `current_flag_dim`, optional `current_value_dim`. |

**Rationale for ratifying Type 0 – Type 6 at vocabulary time.** `00 §4.1` named Type 1, Type 2, Type 6 only; `17` fills in the gaps. Encountering an un-ratified subtype in a Model would otherwise force either an `Unknown` catch-all (which collides with I10 non-exhaustive, since `#[non_exhaustive]` already handles MINOR additions) or rejection at parse. Ratifying the full set up front lets `validate` emit a single clean error when authors write an unknown discriminator (`unknown variant "type_8"`) rather than forcing a shape-conflict cascade later.

**Does Round 1 implement all seven?** No. Vocabulary is ratified across the board, but planner-side support is limited:

- `Type1`, `Type2` are the common shapes with a well-defined as-of semantics; they are the targeted shapes for Round-2 planner work.
- `Type0`, `Type3` have no as-of-join ambiguity but are recognized for advisory purposes (§7.3).
- `Type4`, `Type5`, `Type6` are ratified at the vocabulary level but their full planner treatment is DEFERRED to a later milestone (§10). `Type4`'s `history_data_kind_ref` is the most structurally involved — it pins a second `DataKind` as the history table — and requires coordination with `16`'s composition surface, tracked as `[TD-SCD-TYPE4-HISTORY-REF]` in §10.

### 2.3 Variant-specific payloads — canonical shape pointer

> **Canonical shape lives in [`18 §3`](../18_entities.md#3-temporalshape).** The v1 authoring-layer struct is `TemporalShape { kind: TemporalShapeKind, grain: Option<Grain> }`, per-variant payloads live in `TimeseriesBody` / `EventsBody` / `SnapshotBody` / `ScdBody`, and the v1 `ScdType` roster is trimmed to `{Type1, Type2}`. The full `Type0`–`Type6` Kimball taxonomy documented in §2.2 above is **post-v1 forward-reference** — it describes what each subtype *means* semantically, but only `Type1` / `Type2` are in the v1 enum roster. Readers who need the authoritative struct / enum shape for v1 authoring should go to `18 §3`.
>
> The Rust enum sketch that previously lived at this section level was removed on 2026-04-17 to eliminate the duplicate-struct-definition hazard the consolidation pass was chartered to close. Git history preserves the pre-consolidation sketch; callers who want to read it should `git log docs/design/foundations/17_temporal_shape.md`.

**Design rationale (still applicable to `18`'s shape).** These notes motivated the per-subtype struct-payload design and apply equally to the `18 §3` ratification:

- **Per-subtype payload vs flat-fields.** The alternative — `Scd { subtype: ScdType, valid_from_dim: SemanticsName, valid_to_dim: SemanticsName, current_flag_dim: Option<SemanticsName>, ... }` with the fields meaningless for `Type0` / `Type1` / `Type3` — was rejected. Per-subtype payload lets the type system refuse nonsense (you cannot accidentally author `valid_from_dim` on a `Type0` record) and makes future additions (post-v1 Type 7 dual-view fields) additive inside a single variant. Q-TEMPORAL-002 in `questions/closed/17_questions.md` ratified this position (closed).
- **Payload references are `SemanticsName`s.** Per I1, every time-axis Dimension reference in `TemporalShape` is a canonical Semantics name, not a physical column. Resolution to physical columns happens in `15 §4` via `SemanticMapping`.
- **Non-exhaustive at every level.** `TemporalShape`, `TemporalShapeKind`, and `ScdType` are each `#[non_exhaustive]` per I10. Adding `ScdType::Type3` (post-v1 promotion), extending `TemporalShapeKind` with bi-temporal variants, or growing `ScdBody` with a sentinel-aware `valid_to` marker is MINOR.

### 2.4 Per-variant identifying Dimension — the contract

Each `TemporalShape` variant names **exactly one** identifying Dimension at the top level (for `Timeseries` / `Events` / `Snapshot`) or per-subtype at the nested level (for `Scd`). The identifying Dimension must:

1. **Be declared on the `SimpleDataKind`'s interface** — either authored there or inherited from Tier-1 per `11 §5`. Failure → `COMP_E_1701 TemporalIdentifyingDimMissing`.
2. **Carry `DimensionType::Temporal`** per `13 §4.2`. A non-temporal Dimension cannot serve as a time axis. Failure → `COMP_E_1702 TemporalIdentifyingDimNotTemporal`.
3. **Carry a `DataType` that is one of `Date`, `Time`, `Timestamp`, or `Interval`** (the time-typed subset of `13 §2.1`). `Interval` is unusual as a time axis but permitted for elapsed-time-indexed data. Failure → `COMP_E_1703 TemporalIdentifyingDimUntypedForTime`.

For `Timeseries.grain`:

4. **Must be a member of the identifying Dimension's `grains:` list** per `13 §4.2`. Declaring `grain: Hour` on a `Timeseries` whose `occurred_at_dim.grains: [Day, Month]` is `COMP_E_1704 TimeseriesGrainNotAvailable`.

For `Scd::Type2` / `Type5` / `Type6` window payloads:

5. **`valid_from_dim.DataType` and `valid_to_dim.DataType` must unify** per `13 §5` (strict equality; `11 §3.1`'s global-identity rule makes this a trivial shape-unification check). Failure → `COMP_E_1705 ScdWindowTypeMismatch`.
6. **`current_flag_dim.DataType`** (when present) **must be `Boolean`**. Failure → `COMP_E_1706 ScdCurrentFlagNotBoolean`.

For `Scd::Type4` / `Type5`:

7. **`history_data_kind_ref` / `mini_dim_ref` must resolve to a declared top-level `DataKind`**. Failure → `COMP_E_1707 ScdRefUnknown`.
8. **`history_data_kind_ref`, when resolved, SHOULD itself declare `TemporalShape::Scd { subtype: ScdType::Type2 { .. } }`**. Divergence is `PLAN_W_1701 ScdHistoryShapeUnexpected` (advisory, not an error — authors may have legitimate reasons for non-standard history tables).

Per-variant identifying Dimension participation in `Additivity`, composition, and `Request.temporal` is spelled out in §§4, 6, 7, 8.

---

## 3. Declaration site

### 3.1 `TemporalShape` lives on `SimpleDataKind`

Per `11 §5` and `15 §1.1`, a `SimpleDataKind` is the leaf of the data-kind tree: it carries exactly one `Binding`, one declared `SemanticInterface`, and — by `17` — at most one `TemporalShape`. The shape declaration sits alongside the interface (not inside it): conceptually, `TemporalShape` is a Kind-scope property about the DataKind, not a member of its Semantics roster.

```yaml
# YAML surface (authoritative shape in `32`; shown here for reader orientation).
datasets:
  - name: orders
    dimensions:
      - { name: order_date,    type: { temporal: { grains: [day, month] } }, data_type: date }
      - { name: valid_from,    type: { temporal: { grains: [day] } },        data_type: timestamp(6) }
      - { name: valid_to,      type: { temporal: { grains: [day] } },        data_type: timestamp(6) }
      - { name: is_current,    data_type: boolean }
    temporal_shape:
      scd:
        type_2:
          valid_from: valid_from
          valid_to:   valid_to
          current_flag: is_current
    # binding, measures, keys omitted
```

A `SimpleDataKind` MAY omit `temporal_shape:` entirely. "No shape declared" means the planner performs no shape-aware reasoning on this DataKind — it is neither `Timeseries`, `Events`, `Snapshot`, nor `Scd`. Shape-dependent planner behaviors (as-of joins, snapshot selection) are simply not available for the DataKind. This is the Round-1 default for every pre-existing DataKind that hasn't opted into shape-aware planning.

### 3.2 `ComplexDataKind`s do NOT carry a top-level `TemporalShape`

Per `12 §2` and `16 §5`, a `ComplexDataKind` composes children; it has no binding of its own (`15 §1.2`). `17` inherits the same structural constraint for shape: a `Unionset` / `Grainset` / `Joinset` does not declare its own `TemporalShape`. Instead, the effective shape of a `ComposedSemanticInterface` is **derived from its constituents** per §8's composition rules.

Specifically:

- **Unionset.** Branches must agree on shape (§8.1). The composed surface inherits the common shape. Divergence is `COMP_E_1710 UnionsetShapeConflict`.
- **Grainset.** Branches must all be `Timeseries` or all `Snapshot` (or all shape-free); mixing shape kinds inside a Grainset is `COMP_E_1711 GrainsetShapeMixed`. The composed surface inherits the common kind; the grain axis is resolved per `22 §…`.
- **Joinset.** Per-branch shapes are retained; the Joinset surface does not collapse to a single shape. `JoinType::AsOf` traversals (§5) bind specific shape pairs along individual edges.
- **Implicit `ComposedSemanticInterface` (`16 §11`).** Same as Joinset — per-constituent shapes are retained, and composition-surface rollup / as-of anchoring obeys the per-shape rules in §8.

### 3.3 Implicit "no shape declared" behavior

A `SimpleDataKind` without `temporal_shape:` is treated as **shape-unclassified**. The planner:

- Does NOT reject queries over the DataKind. Every non-shape-dependent operation (filter, group-by, aggregation, `Grain` rollup along the declared temporal Dimension) works as specified in `11`–`16`.
- Does NOT synthesize a default shape. No heuristic guessing ("this DataKind has an `occurred_at` column, therefore `Events`"); implicit-shape inference is out of scope per §1.2 forward-refs.
- DOES emit `PLAN_W_1702 ShapeAwareRequestOnUnclassifiedDataKind` when a `Request.temporal` block (§6) is provided targeting an unclassified DataKind. Vocabulary-ratified; advisory-only per I12.

### 3.4 Cross-occurrence shape-unification

`TemporalShape` is authored on a `SimpleDataKind`, not on a Semantics. The shape-unification rules in `11 §5.1` apply to Semantics shape fields (`data_type`, `additivity`, ...), not to DataKind-level `temporal_shape:`. A Semantics name like `occurred_at` may appear on multiple DataKinds (by global identity per `11 §3.1`); each containing DataKind independently states its own `TemporalShape`.

Consequence: the same `occurred_at` Dimension can be the `Timeseries.occurred_at_dim` of DataKind `A` AND the `Events.occurred_at_dim` of DataKind `B` AND the `Scd::Type2.valid_from_dim` of DataKind `C` — each DataKind chooses its own shape, and the Semantics name is shared only at the identity level.

---

## 4. Interaction with `Grain`

### 4.1 The matrix

| Shape | Declared `Grain` | Rollup legality | Sub-aggregation legality |
|---|---|---|---|
| `Timeseries { occurred_at_dim, grain: G }` | `G` (the declared cadence) | Rollup to any `G' >= G` in the coarseness order (per `13 §3.2`) is legal; planner emits a coarsening `DateTrunc` projection per `14a §…`. | Sub-aggregation below `G` is **illegal** — the source data is only dense at `G`. `PLAN_E_1710 TimeseriesSubGrainRequested`. |
| `Events { occurred_at_dim }` | inferred from `occurred_at_dim.grains:` per `13 §4.2` | Rollup to any `grain ∈ occurred_at_dim.grains:` is legal via bucket-then-aggregate; planner emits `DateTrunc(occurred_at_dim, grain)` + aggregate. | Sub-aggregation at a finer grain than the coarsest `grain ∈ occurred_at_dim.grains:` is legal (events are sparse, finer bucketing is meaningful); bounded by `occurred_at_dim`'s declared grain set. |
| `Snapshot { snapshotted_at_dim, cadence: Some(C) }` | `C` (the declared cadence) | Rollup to `Grain >= C` is **semi-legal**: the planner picks the set of snapshots whose timestamps bucket into the target period and applies a user-chosen reduction (first / last / average — policy per `25 §…`). Default: last snapshot in each period. | Sub-aggregation below `C` is **illegal** — no source data exists between snapshots. `PLAN_E_1711 SnapshotSubCadenceRequested`. |
| `Snapshot { snapshotted_at_dim, cadence: None }` | free-form | "Latest snapshot at or before `Request.as_of`" semantics per §6.2. No free rollup; the planner selects one snapshot per Request. | Not applicable — single snapshot per Request. |
| `Scd { subtype: Type0 / Type1 }` | no intrinsic temporal rollup axis | Shape carries no time axis (or carries a trivial one). Request-level rollups on non-temporal Dimensions work per `11`–`16`; time-axis rollups are meaningless. | Not applicable. |
| `Scd { subtype: Type2 / Type4 / Type5 / Type6 }` | no intrinsic grain; `valid_from` / `valid_to` carry event-point times, not cadence-aligned buckets | Rollup semantics depend on as-of anchoring (§5.2 / §8.3); the SCD surface itself does not carry a rollable `Grain`. A Request over an SCD surface without anchor specification defaults to "current rows" per §6.3. | Not applicable at the SCD layer; applicable downstream after as-of anchoring resolves the surface to one row per entity. |
| `Scd { subtype: Type3 }` | no intrinsic grain; prior-value carries one-generation-back data | As `Type1` for rollup purposes — the current row's values are authoritative. | Not applicable. |

### 4.3 Interaction with `Grainset`

Per `16 §5.3`, a `Grainset` composes `Simple` children at distinct grains for the planner to pick the cheapest covering grain. When the `Grainset`'s children all declare `TemporalShape::Timeseries` (the intended use case):

- Each child's `Timeseries.grain` must be a distinct value; the `Grainset`'s `levels:` per `22 §…` pair each child with its grain.
- Rollup from child grain `G_child` to Request grain `G_req` is legal iff `G_req >= G_child` in the coarseness order per `13 §3.2`.
- Downgrade (picking a coarser-grain child to serve a finer-grain Request) is forbidden per `11 §4.2`'s no-silent-downgrading stance; `PLAN_E_1712 GrainsetDowngradeRequested`.

When the `Grainset`'s children are `Events` or `Snapshot` shapes, `22 §…` ratifies the per-shape `Grainset` strategy; `17` records only that shape-mixing inside one `Grainset` is `COMP_E_1711 GrainsetShapeMixed`.

---

## 5. Interaction with `JoinType` — ratifying `AsOf`

### 5.1 The `AsOf` variant — vocabulary ratification (forward-reference)

The v1 `JoinType` roster is ratified in [`18_entities.md §2.3`](../18_entities.md) as `{Inner, Left, Right, Full}`; `AsOf` is **post-v1 deferred**. `16 §4.4.2` parked the `AsOf` variant behind this doc's `TemporalShape` vocabulary; `17 §5` discharges that gate by specifying what the variant *would look like* when it lands in a post-v1 MINOR.

The `AsOfAnchor` enum below is the carrier type for the eventual `JoinType::AsOf(AsOfAnchor)` variant; it is **not** part of the v1 authoring surface, does not appear in `18 §2.3`, and is not emitted by the Round-1 planner. Its shape is pinned here so the post-v1 extension is a pure additive change:

```rust
/// Forward-reference (post-v1). Anchor specification that would be carried
/// by `JoinType::AsOf(AsOfAnchor)` when the variant lands. Per-shape.
///
/// Implementation DEFERRED per `17 §10`. Vocabulary ratified so that the
/// eventual addition to `18 §2.3`'s `JoinType` enum is purely additive.
#[non_exhaustive]
pub enum AsOfAnchor {
    /// The `to` side is an `Scd` kind with valid-window payload. Each matched
    /// `to`-row must satisfy `valid_from <= probe < valid_to` (half-open
    /// interval; open-ended `valid_to = NULL / sentinel` matches indefinitely).
    ScdWindow {
        /// Canonical Semantics name of the `from`-side Dimension carrying the
        /// probe timestamp (the query's "as of").
        probe_dim: SemanticsName,

        /// Echoed from the `to`-side Scd shape's `valid_from_dim`.
        to_valid_from_dim: SemanticsName,

        /// Echoed from the `to`-side Scd shape's `valid_to_dim`.
        to_valid_to_dim: SemanticsName,
    },

    /// The `to` side is a `Snapshot` kind. The planner picks, per
    /// `from`-side row, the `to`-side row whose `snapshotted_at` is the
    /// latest value `<= probe`.
    SnapshotLatestAtOrBefore {
        probe_dim: SemanticsName,
        to_snapshotted_at_dim: SemanticsName,
    },
}
```

**Vocabulary-ratified, implementation-DEFERRED.** Per §10, the planner's Round-1 implementation does not construct `AsOf` joins. `JoinType` is derived at compile from `Relationship.optional` (`18 §2.9`) and is no longer authored at the YAML surface (post-2026-05-12 rebase); the derivation table's roster stays `{Inner, Left, Right, Full}` in v1, with `AsOf` admitted as an auto-activation by the temporal-shape pair in a later MINOR (`24 §7.2`). The gate `16 §4.3` / `18 §2.9` parked on `TemporalShape` availability is now in place, but the door is not yet open.

### 5.2 Per-shape-pair legality matrix

The matrix below enumerates which shape pairs on the `from` / `to` sides of a `Relationship` admit `AsOf` traversal. Only pairs marked **Admits `AsOf`** are legal; the planner rejects other combinations with `PLAN_E_1720 AsOfShapePairIllegal`.

| `from` shape | `to` shape | Admits `AsOf`? | Anchor | Notes |
|---|---|---|---|---|
| `Events` | `Scd::Type2 / Type5 / Type6` | Yes | `AsOfAnchor::ScdWindow { probe_dim = from.occurred_at_dim, to_valid_from_dim, to_valid_to_dim }` | The canonical "what was the entity state at the time of this event" pattern. |
| `Events` | `Scd::Type0 / Type1 / Type3 / Type4` | No — use `Inner` / `Left` | — | These subtypes do not carry a valid-window; "as of" resolves to the current row (Type 0 / 1 / 3) or externalizes to a separate history table (Type 4). The planner treats them as bare current-state lookups. `Type4` as-of joins are DEFERRED per §10 with a structural gap; tracked as `[TD-SCD-TYPE4-ASOF]`. |
| `Events` | `Snapshot` | Yes | `AsOfAnchor::SnapshotLatestAtOrBefore { probe_dim = from.occurred_at_dim, to_snapshotted_at_dim }` | "What was the latest snapshot at the time of this event." Requires at least one snapshot `<=` the probe; unmatched events fall through per the `JoinType`'s outer-join layer (`Left` outer gives NULL-fill; `Inner` drops). |
| `Timeseries` | `Scd::Type2 / Type5 / Type6` | Yes | `AsOfAnchor::ScdWindow { probe_dim = from.occurred_at_dim, ... }` | Same as Events-↔-SCD; cadence-regular version. |
| `Timeseries` | `Snapshot` | Yes | `AsOfAnchor::SnapshotLatestAtOrBefore { ... }` | Useful for joining a daily metric stream to a weekly snapshot of reference data. |
| `Snapshot` | `Scd::Type2` | Yes | `AsOfAnchor::ScdWindow { probe_dim = from.snapshotted_at_dim, ... }` | "At each snapshot moment, what was the entity state." Less common; ratified for completeness. |
| `Scd::Type2` | `Events` / `Timeseries` / `Snapshot` | No | — | The SCD side is the target of "as of" queries, not the probe side. Reverse the relationship direction or use `Inner`. |
| `Snapshot ↔ Snapshot` | — | No | — | Two snapshots join on their natural keys at a common `snapshotted_at`; no as-of anchoring. Use `Inner`. |
| `Events ↔ Events` | — | No | — | Event-to-event join is natural-key based; no as-of anchoring. Use `Inner`. |
| `Timeseries ↔ Timeseries` at matching grain | — | No | — | Natural key + grain-aligned join; use `Inner`. |
| Either side shape-unclassified | — | No | — | `AsOf` requires both sides to be ratified shapes. `PLAN_E_1721 AsOfShapeUnclassified`. |

### 5.3 Legal cardinalities per shape pair

`16 §3.5`'s `Cardinality × Additivity` matrix is sharpened when `TemporalShape` is involved:

| Shape pair | Typical `Cardinality` | Notes |
|---|---|---|
| `Events ↔ Scd::Type2` (AsOf, forward) | `ManyToOne` | Each event matches one validity window on the SCD side. No fanout. |
| `Events ↔ Snapshot` (AsOf, forward) | `ManyToOne` | Each event matches one snapshot (the latest at-or-before). No fanout. |
| `Timeseries ↔ Scd::Type2` (AsOf, forward) | `ManyToOne` | Same as Events; each tick matches one SCD window. |
| `Snapshot ↔ Snapshot` (Inner, same cadence) | `OneToOne` (per snapshot instant) | The natural pairing when two snapshots share the same cadence. |
| `Events ↔ Events` (Inner, same entity) | depends on volume — typically `ManyToMany` and shape-warn-worthy | `PLAN_W_0502 ManyToManyFanoutAdvisory` was retired in `16 §14.4` (2026-04-29) per Q-COMP-005's intent-advisory deferral; the volume / cardinality concern remains an authoring consideration but is no longer planner-emitted in v1. |

Reverse-direction traversal of an `AsOf` `Relationship` is **forbidden**. The probe side and the anchor side are semantically asymmetric; "what events were recorded during this SCD window" is not the mirror of "what was the SCD state at this event time" — the former is a range query that returns `0..N` events per window, the latter is a point query that returns exactly one SCD row per event. The planner rejects reverse traversal with `PLAN_E_1722 AsOfReverseTraversalForbidden`. Authors needing the reverse semantics declare an explicit `Joinset` with the opposite anchor — the authoring-layer `Directionality` enum is retired (`16 §2.4`, 2026-05-12), so the reverse-direction restriction here is `AsOf`-specific and lives on the temporal shape, not on the Relationship struct.

### 5.4 `PlanNode::Join` carriage — DEFERRED

Per `35 §…` (pending), `PlanNode::Join` carries `JoinType` verbatim (`16 §4.3`). The Round-1 IR shape (`35 §…`) does not yet include `AsOfAnchor` materialization; extending `JoinNode` with the anchor payload is MINOR per I10 and lands alongside the planner implementation. The vocabulary ratified here binds `35`'s eventual shape: when `JoinType::AsOf(anchor)` appears on a `JoinNode`, the `anchor` field must be round-trippable into the plan IR per I8.

### 5.5 `Joinset`-level `AsOf` overrides — DEFERRED, post-implicit ordering ratified

`16 §13.3` ratifies per-traversal `JoinType` overrides in `Joinset` declarations. Round-1 overrides are limited to `Inner / Left / Right / Full`; `AsOf` override inside a Joinset traversal is DEFERRED to a post-implicit milestone. Q-TEMPORAL-003 ratified Option B (implicit-first, 2026-04-28): when planner-side `AsOf` lands, **milestone 1** ships matrix-driven implicit synthesis — Joinset traversals automatically receive `AsOf` per `24 §7.2.1` without YAML override admission; **milestone 2** (or a later additive MINOR) extends the `Joinset` YAML override surface to accept `AsOf`, opening the narrowing-or-forcing escape hatch documented in `24 §5.3 / §7.2.2`.

---

## 6. Interaction with `Request`

### 6.1 The `Request.temporal` block — vocabulary ratification

`00 §4.1`'s `Request` row ratifies an optional `temporal:` block carrying as-of timestamp and time-range overrides, marked DEFERRED and gated on `TemporalShape`. `17 §6` ratifies the block's **shape** and **design-level semantics**; `34 §…` ratifies the planner's consumption. The ratified shape:

```rust
/// Optional Request-scoped temporal overrides. Consumption by the planner is
/// DEFERRED per `17 §10`; vocabulary and design-level semantics are ratified.
#[non_exhaustive]
pub struct RequestTemporal {
    /// The as-of timestamp for the Request. Applied:
    ///   - as the probe timestamp for `AsOf` joins (§5)
    ///   - as the anchor for "latest snapshot at or before" on Snapshot kinds (§4.1)
    ///   - as the window-selection point for SCD kinds when the Request has
    ///     no explicit window filter
    ///
    /// When `None`, shape-dependent default-current behavior applies (§6.3).
    pub as_of: Option<Timestamp>,

    /// Optional time-range override, restricting the Request to a contiguous
    /// temporal window on the shape's identifying Dimension.
    ///   - For `Timeseries` / `Events`: filters rows to `[range.start, range.end)`
    ///     on `occurred_at_dim`.
    ///   - For `Snapshot`: picks snapshots whose `snapshotted_at` falls in
    ///     `[range.start, range.end)`.
    ///   - For `Scd`: selects rows whose validity window **intersects** the
    ///     range (at least partial overlap).
    pub range: Option<TimeRange>,
}

#[non_exhaustive]
pub struct TimeRange {
    /// Inclusive lower bound.
    pub start: Timestamp,

    /// Exclusive upper bound.
    pub end: Timestamp,
}
```

### 6.2 As-of timestamp semantics

When `as_of` is specified on a Request targeting (or traversing) a DataKind with a ratified `TemporalShape`, the semantics per shape:

| Target / traversed shape | `as_of` behavior |
|---|---|
| `Timeseries` | No filter effect on the target rows (`as_of` is not a filter here); but it binds the probe timestamp for any `AsOf` join hop involving this kind. |
| `Events` | Same as `Timeseries`. |
| `Snapshot { cadence: Some }` | Selects the single snapshot `T` s.t. `T <= as_of` and `T` is the latest in the declared cadence series at or before `as_of`. No snapshot `<= as_of` → `PLAN_E_1730 SnapshotAsOfNoCoveringSnapshot`. |
| `Snapshot { cadence: None }` | Same as `Some` case — latest snapshot at or before `as_of`. |
| `Scd::Type0 / Type1 / Type3` | No history to pick from; current row is returned regardless of `as_of`. `PLAN_W_1730 ScdAsOfIgnored` (advisory — author likely intended Type-2-style history behavior). |
| `Scd::Type2 / Type5 / Type6` | Selects rows where `valid_from <= as_of < valid_to` (half-open interval; open-ended `valid_to` matches). Exactly one row per entity in the well-formed case. |
| `Scd::Type4` | Picks from `history_data_kind_ref` per `Type2` semantics when `as_of` is historic; picks from the current table when `as_of` is "current" (§6.3). DEFERRED — the cross-DataKind hop is the complicated part. |

### 6.3 Default-current behavior

When `as_of` is `None` and the Request targets a shape-aware DataKind:

- **`Timeseries` / `Events`**: No default-current behavior — these shapes have no "current row" concept. The Request returns all in-range rows per `range` (or all rows if `range` is also `None`).
- **`Snapshot`**: Default is "latest snapshot" (the snapshot with the maximum `snapshotted_at`). Equivalent to `as_of = SessionContext.now` for most queries.
- **`Scd::Type0 / Type1 / Type3`**: Current row (trivially, since there is only one per entity).
- **`Scd::Type2 / Type5 / Type6`**: Default is "current rows" — rows where `current_flag_dim = TRUE` if declared, else rows where `valid_to_dim IS NULL` (the open-ended convention; engine-specific sentinel conventions resolved per `registry/temporal_shape_mapping.md`).
- **`Scd::Type4`**: Default is the current table (not the history table).

When the default-current lookup rule is ambiguous (e.g. `Type2` with neither `current_flag_dim` declared nor a consistent `valid_to` open-ended convention), the planner emits `PLAN_W_1731 ScdCurrentRowHeuristic` and picks the row with the maximum `valid_from` per entity. This heuristic is **not** a ratified semantics; authors should declare `current_flag_dim` or normalize their `valid_to` sentinel convention.

### 6.4 Time-range override semantics

When `range` is specified, the planner emits a filter appropriate to the target shape (§6.1's `range` subsection). Shape-interaction with `range`:

- **`range` without `as_of` on an SCD kind** returns every row whose validity window intersects `range` — potentially multiple rows per entity. The Request must either include a `filter` pinning to a single row per entity or `group_by` at the entity level and aggregate across windows. `PLAN_W_1732 ScdRangeMultipleRows` on requests that would return >1 row per entity without intent declared.
- **`range` with `as_of` on an SCD kind** is a *conjunction*: as-of pins to one row per entity, and that row must intersect `range`. Typically degenerate — use one or the other.
- **`range` on `Snapshot`**: Returns the **sequence** of snapshots in-range. Rollup across that sequence uses the cadence-rollup policy from §4.1.

### 6.5 Shape-dependent planner delegation — DEFERRED

The algorithm that resolves `Request.temporal` against a composed surface with heterogeneous constituent shapes is DEFERRED. Illustrative cases:

- A `Request` spanning `Events A` ↔ `Snapshot B` with `as_of` specified: the events side takes `as_of` as a probe; the snapshots side takes it as a latest-at-or-before selector. Both directions work; composition is clean.
- A `Request` spanning `Timeseries A` at `grain: Hour` ↔ `Timeseries C` at `grain: Day` with `range` specified: the time-range is applied independently on both sides at their respective grains. Composition is clean.
- A `Request` with `from: None` implicit composition (`16 §11`) where the owning kinds include one `Scd::Type2` and one `Snapshot` both needing `as_of` interpretation: DEFERRED. Q-TEMPORAL-004.

The planner's shape-aware `Request.temporal` consumer lands in `34 §…` with per-shape resolution per the rules above and with the multi-shape-composition cases settled in the same pass.

---

## 7. Interaction with `Additivity`

### 7.1 Two-axis independence

Per `11 §7.2`, `Additivity` and `TemporalShape` are **independent** inputs to the planner. `Additivity` describes whether a Measure / Metric composes mechanically under rollup; `TemporalShape` describes how the underlying DataKind records time. Neither derives from the other, and neither overrides the other's authored value.

`17` preserves that independence. Specifically, `17` does **not**:

- Default `Additivity` based on `TemporalShape`. Explicit authoring at the Measure / Metric level per `11 §7.2` is always authoritative.
- Reject shape / additivity combinations as structural errors. Every combination is authoring-legal.

What `17` **does** is enumerate the combinations that are **semantically suspicious** — likely authoring mistakes — and emit advisory warnings (Severity `Warning`, never `Error`). Authors can ignore the advisory; a future opt-in `additivity_confirmed: true` field may suppress it (tracked per `11 §7.3`).

### 7.2 Interaction with `16 §3.5`'s `Cardinality × Additivity` matrix

`16 §3.5` ratified the Cardinality × Additivity matrix for composed surfaces. `17` sharpens that matrix when one or more constituents carries a ratified `TemporalShape`:

- **`Scd::Type2 / Type5 / Type6` on the probe side of an `AsOf` join.** Cardinality is `ManyToOne` (per §5.3); each probe row matches exactly one SCD row. `Additivity` behaves as for any `ManyToOne` forward walk — `Additive` safe, `SemiAdditive` safe, `NonAdditive` safe. The SCD fanout concern is the *reverse* direction (§5.3 forbids reverse traversal).
- **`Snapshot` on the anchor side of an `AsOf` join.** Same as SCD: `ManyToOne` forward; no fanout; no additional `Additivity` constraints.
- **`Timeseries` ↔ `Timeseries`** Cartesian risk when grains differ. Pre-`17`, the fanout concern is caught at the Grainset layer (§4.3). `17` does not add new rules here.

No new `PLAN_E_*` codes in the Cardinality × Additivity matrix originate from `17`; the sharpening is entirely in the *advisory* roster below.

### 7.3 Advisory-warning roster

The concrete advisories `17` ratifies. Each is `Severity::Warning` and carries a stable code in `PLAN_W_17NN`. All are advisory: planner proceeds, diagnostic surfaces, query runs.

| Code | Variant | Fires when | Rationale |
|---|---|---|---|
| `PLAN_W_1740` | `AdditiveOnSnapshot { measure, data_kind }` | A Measure / Metric with `Additivity::Additive` is queried over a `TemporalShape::Snapshot` DataKind with a group-by that includes `snapshotted_at_dim` (or a rollup of it). | Snapshots capture full state; naively summing stock-like quantities across snapshot-time double-counts. Author likely meant `SemiAdditive { unsafe_axes: [snapshotted_at_dim] }`. |
| `PLAN_W_1741` | `AdditiveOnScdHistoryPreserving { measure, data_kind, subtype }` | A Measure / Metric with `Additivity::Additive` is queried over a `TemporalShape::Scd` DataKind with subtype `Type2 / Type4 / Type5 / Type6` and a group-by that does not pin to `current_flag_dim = TRUE` or an equivalent. | History-preserving SCD kinds have multiple rows per entity; summing mechanically double-counts across validity windows. Author likely meant to anchor via `Request.temporal.as_of` or filter to current. |
| `PLAN_W_1742` | `SemiAdditiveUnsafeAxisAbsentFromShape { measure, data_kind, unsafe_axes, shape }` | A Measure / Metric with `Additivity::SemiAdditive { unsafe_axes: [X, ...] }` names an axis `X` that is NOT the shape's identifying Dimension (e.g. `unsafe_axes: [customer_id]` on a `Snapshot` shape where the natural unsafe axis is `snapshotted_at_dim`). | Non-temporal unsafe axes are legal (per `11 §7.4`) but the common SCD / Snapshot case targets the shape-time axis. Advisory prompts the author to verify. |
| `PLAN_W_1743` | `NonAdditiveOnTimeseriesGrainMismatch { metric, data_kind, request_grain, source_grain }` | A `NonAdditive` Metric is queried over a `Timeseries` DataKind at a request grain coarser than the source grain, and the underlying Measures do not resolve to the queried grain (would require materialization per `11 §6.3.1`). | The `NonAdditive` Metric must be recomputed at the queried grain; the planner emits the recompute plan but advises the author that the layer-crossing is expensive. |
| `PLAN_W_1744` | `SemiAdditiveOnEvents { measure, data_kind }` | A Measure / Metric with `Additivity::SemiAdditive` is queried over a `TemporalShape::Events` DataKind. | Events are naturally aggregable across time; `SemiAdditive` on an event-based flow-type Measure is unusual. The advisory does not block — some events are naturally stock-like snapshots encoded as events — but nudges the author to verify. |

**What about `Type0 / Type1 / Type3`?** These SCD subtypes carry one row per entity (or one + prior per entity for Type 3); mechanical additivity is safe without anchoring. No advisories originate from them.

### 7.4 Author-side suppression — DEFERRED

`11 §7.3` reserves a future `additivity_confirmed: true` flag for suppressing advisories. `17` does not ratify that flag's shape; it lands alongside the planner's advisory-emission path in `34 §…`. Until then, advisories always emit; consumers may filter them by code at the `Diagnostic` level.

---

## Testing Strategy

- Per-variant identifying Dimension validation (TS-C1 through TS-C3, TS-C4 Timeseries grain membership, TS-C5 SCD window type unification)
- Grain × Shape interaction matrix validation (PLAN_E_1710 TimeseriesSubGrainRequested, PLAN_E_1711 SnapshotSubCadenceRequested, PLAN_E_1712 GrainsetDowngradeRequested)
- AsOf shape-pair legality matrix (PLAN_E_1720 AsOfShapePairIllegal, PLAN_E_1721 AsOfShapeUnclassified, PLAN_E_1722 AsOfReverseTraversalForbidden — all DEFERRED implementation, test vocabulary only)
- Additivity advisory roster emission (PLAN_W_1740-1744 — all Severity::Warning, verify advisory fires without blocking planning)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
