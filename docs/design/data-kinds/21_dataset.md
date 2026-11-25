---
id: 21-dataset
status: ratified
version: 2.1
impl: drift
---


# 21. Dataset (`SimpleDataKind`)

> **Reconciliation (post-thirteenth-pass cascade, 2026-04-30).** The v1 authoring-layer canonical shape for `Dataset` is ratified across:
>
> - `[20_taxonomy.md §2](./20_taxonomy.md)` — sealed trait hierarchy (`DataKind`, `SimpleDataKind`, `PublicDataKind`, `NestedDataKind`); `Dataset` implements `SimpleDataKind + PublicDataKind`, `NestedDataset` implements `SimpleDataKind + NestedDataKind`.
> - `[../apis/32_semstrait_model.md §3.2](../apis/32_semstrait_model.md)` — `DatasetBody` per-variant body struct (`base: DataKindBase<LeafExtras>` + Dataset-specific fields).
> - `[../apis/32_semstrait_model.md §3.3](../apis/32_semstrait_model.md)` — `Dataset` and `NestedDataset` concrete forms.
> - `[../apis/32_semstrait_model.md §4](../apis/32_semstrait_model.md)` — `LeafExtras` block (`catalog` / `storage` / `semantic_mapping` / `temporal`); `Dataset` is the sole `SimpleDataKind` variant in v1 and authors the full `LeafExtras` field set.
> - `[../foundations/18_entities.md](../foundations/18_entities.md)` — canonical entity types consumed by `DatasetBody`: `SemanticInterface`, `TemporalShape`, `SemanticMapping`, `Keys`, `AiContext`, inline-vs-`ref` grammar for Dimensions / Measures / Metrics / filters.
> - `[26_nesting_matrix.md](./26_nesting_matrix.md)` — nesting rules (R1 / R2 / R3) and the "Nested-form structural-only" rule.
> - `[../apis/32b_catalogs_yaml.md](../apis/32b_catalogs_yaml.md)` — `CatalogRef` grammar consumed via `extras.catalog:`.
>
> `21` retains authority for:
>
> - The `Dataset` authoring surface — explicit (authored) vs. implicit (compile-derived) fields, with cross-references to the type-level shape in `32 §3` / `32 §4`.
> - Plan-time observable invariants of `Dataset` resolution — what an author or consumer can rely on from the emitted plan, *not* the algorithm body.
> - Variant-specific Precondition surfaces — `VALID_E_2100`–`2199`, `COMP_E_2100`–`2199`, `PLAN_E_2100`–`2199`.
>
> **Algorithm body relocated.** The `SimpleStrategy` L1–L5 plan emission rules (formerly `21 §4`) live in `[../apis/34_semstrait_planner.md §<SimpleStrategy>](../apis/34_semstrait_planner.md)` (forthcoming). This rebase parked the L1–L5 content in `[../_drafts/34_simple_strategy.md](../_drafts/34_simple_strategy.md)` pending the `34` drafting pass.
>
> **Per-variant `TemporalShape` cascade rules** (Unionset / Grainset symmetric children with cascade-down; Joinset cascade-boundary with authoritative composer shape) are forthcoming in `22` / `23` / `24`. `21` cross-references the framework without restating per-variant rules.

## 1. Purpose and Scope

### 1.1 What `21` ratifies

`21` is the per-variant chapter for `Dataset` (the sole `SimpleDataKind` concrete in v1). It owns three things: (a) the **authoring surface** — explicit vs. implicit fields the author / consumer sees on the type — with cross-references to the type-level shape in `32 §3` (forms + body) and `32 §4` (`LeafExtras`); (b) the **plan-time observable invariants** of `Dataset` resolution — what an author or consumer can rely on from the emitted plan, *not* the algorithm body; (c) **Dataset-specific Preconditions** in the per-variant code bands `VALID_E_2100`–`2199` (validate), `COMP_E_2100`–`2199` (compile), `PLAN_E_2100`–`2199` (plan).

### 1.2 What `21` does NOT ratify

The sealed trait hierarchy and shared `DataKind` invariants live in `20`. The Rust struct / YAML shape lives in `32 §3` / `32 §4`. The `Binding` mechanics live in `15`. The `SimpleStrategy` algorithm body (formerly `21 §4`) lives in `34 §<SimpleStrategy>` (forthcoming; parked in `_drafts/34_simple_strategy.md`). Per-variant `TemporalShape` cascade rules live in `22` / `23` / `24`. Composition under a `ComplexDataKind` lives in `16` and the variant chapters `22` / `23` / `24`. `PlanNode` shapes live in `35`. Engine rendering lives in `36`. Peer-vocabulary references (dbt MetricFlow, Cube.js, LookML) live in `00_overview.md` only.

### 1.3 Guardrails — how `21` upholds `00 §9` invariants


| Invariant                                  | Where `21` keeps it                                                                                                                                                                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I1** — no raw SQL in canonical layer     | The emitted plan tree carries `PhysicalExpr` trees and `PlanNode` variants only; no SQL-shaped strings. Adapter rendering (`36`) is the only SQL site.                                                                |
| **I2** — physical types via adapters only  | Plan emission projects columns typed by the resolved physical `Schema` (logical `DataType` per `15 §3.2` / `13 §2`). Declared-vs-physical reconciliation happens at the semantic boundary via compile-emitted `Cast`. |
| **I3** — no engine branching               | `Dataset`'s plan emission names canonical `PlanNode`s only — no engine, dialect, or engine-specific operator.                                                                                                         |
| **I4** — SemanticManifest determinism      | `Dataset` resolution is a pure function of `(ResolvedDataset, Request)`. Multi-source fan-out order follows lexical resolution of `extras.storage.paths:` per `15 §3.6`.                                              |
| **I5** — compile-time resolution           | Every input the planner needs (`ResolvedSemanticMapping`, `ResolvedExprTable`, per-source `Coverage`, per-source `Metadata` literals) is compile-built. Plan-time is O(1) lookups per Semantics.                      |
| **I6** — synchronous hot path              | No I/O at any stage of `Dataset` resolution.                                                                                                                                                                          |
| **I8** — planner-complete SemanticManifest | `Dataset` resolution touches only `ResolvedDataset`; no YAML, no catalog.                                                                                                                                             |
| **I10** — non-exhaustive public sum types  | `Dataset` and `NestedDataset` are `#[non_exhaustive]` per `32 §3.3`; sealed trait surface in `20 §2.2` excludes external implementers.                                                                                |
| **I12** — first-class Diagnostics          | Every error code in §§7–9 is stable and carries a `Diagnostic.location`.                                                                                                                                              |


## 2. Authoring-Time Surface

### 2.1 Explicit vs. implicit fields

The `Dataset` author touches the following surface. Type-level shape lives in `32 §3.3` (`Dataset` / `NestedDataset` concrete forms), `32 §3.2` (`DatasetBody`), and `32 §4` (`LeafExtras`); `21` enumerates obligations and cross-references foundations for substance.


| #   | Axis               | Explicit (authored)                                                                                                                                                                                                                                                         | Implicit (compile-derived)                                                                                                                                                                                                                                              | Cite                                                |
| --- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| A   | Identity           | `body.base.name` (Public form: globally unique across top-level `data_kinds:`; Nested form: scoped to parent's nested-kind scope per `11 §2.1`).                                                                                                                            | `id: EntityId` (model `id` carried onto the manifest `DataKind`; `33 §6.1`).                                                                                                                                                                                                               | `32 §3.1`, `32 §3.3`, `11 §3`                       |
| B   | Description        | `description: Option<String>` on `Dataset` (Public form only; absent on `NestedDataset`).                                                                                                                                                                                   | None.                                                                                                                                                                                                                                                                   | `32 §3.3`                                           |
| C   | Semantic interface | `semantic_interface: SemanticInterface` on `Dataset` (Public form only; absent on `NestedDataset` per `26 §3` "Nested-form structural-only" rule). Authors Dimensions, Measures, Metrics, Filters, Keys per `18 §2`.                                                        | Per-Semantics inferred-type fields, typed-leaf substitution table, Metric expansion, Computed-Dimension expression resolution — per `19 §3`.                                                                                                                               | `32 §3.3`, `18 §2`, `19 §3`                           |
| D   | AI context         | `ai_context: Option<AiContext>` on `Dataset` (Public form only).                                                                                                                                                                                                            | None.                                                                                                                                                                                                                                                                   | `32 §3.3`, `18 §6`                                  |
| E   | TemporalShape      | `extras.temporal: Option<TemporalShape>` (kind + grain) on `LeafExtras`.                                                                                                                                                                                                    | Cascade behavior depends on parent variant's rule: under Unionset / Grainset, silent leaves inherit ancestor's `kind` (must agree if both authored); under Joinset, leaf is autonomous. Per-variant rules ratified in `22`, `23`, `24`.                                 | `32 §4`, `17`, `22`, `23`, `24`                     |
| F   | Catalog / storage  | `extras.catalog: Option<CatalogRef>` and `extras.storage: StorageConfig` (paths-or-tables, mutually exclusive per `18 §3`).                                                                                                                                                 | `Binding` resolution: `ResolvedSemanticMapping`, `Vec<ResolvedPhysicalSource>`, per-source `Coverage`, per-source `Metadata` literals (per `15 §10`).                                                                                                                   | `32 §4`, `32b`, `18 §3`, `15`                       |
| G   | Semantic mapping   | `extras.semantic_mapping: SemanticMapping` (per-Semantics → physical column / literal / metadata recipe / computed expression per `18 §10`). Implicit name-match fallback per `15 §10.4`: Semantics absent from the explicit map fall back to a same-named physical column. | Per-Semantics resolved `SemanticMappingValue` (variants: `Column`, `Literal`, `Metadata(Recipe)`, `Computed(PhysicalExpr)`); compile-synthesized `Metadata` entries from Dimension-type-level recipes per `15 §10.4` step 4.0; boundary `Cast` insertion per `15 §9.1`. | `32 §4`, `18 §10`, `15 §7.2`, `15 §9.1`, `15 §10.4` |


**Public vs. Nested form distinction.** Rows B, C, D apply only to the Public form (`Dataset`). The Nested form (`NestedDataset`) carries only `body` (which contains `base.name` and `extras: LeafExtras`); fields B/C/D are structurally absent per `26 §3`. The `name` at nested scope is the structural anchor in the parent's nested-kind scope (`11 §2.1`); per Q-DS-001 (`closed/21_questions.md`), `name` doubles as the structural label without a separate `label:` field.

### 2.2 Nesting interaction

A `Dataset`:

- **MAY** appear at Root scope as a top-level public DataKind (per `32 §2.1`'s `datasets:` plural tag).
- **MAY** be nested inline as a `NestedDataset` under a `ComplexDataKind` per `26`'s nesting matrix (Unionset branches, Grainset levels, Joinset members). Nested form is structural-only (no `description` / `semantic_interface` / `ai_context`); the nested `name` is the structural anchor.
- **MUST NOT** contain any `DataKind` child — `Dataset` is a leaf in the nesting matrix per `26 §R1`.

### 2.3 Authoring example — full SemanticInterface surface

A worked YAML showing every authoring slot a `Dataset` exposes (focus is **completeness of the surface**, not plan emission — see §10 for the plan-shape worked example):

```yaml
datasets:
  - name: orders
    description: "Order-line fact dataset."
    dimensions:
      - name: ordered_at
        data_type: Timestamp
      - name: region
        data_type: String
      - name: product_category
        data_type: String
    measures:
      - name: gross_revenue
        agg: sum
        expr: amount_cents
        data_type: Long
      - name: order_count
        agg: count
        data_type: Long
    metrics:
      - name: avg_order_value
        expr: gross_revenue / order_count
        data_type: Decimal
    keys:
      - name: order_id
        kind: primary
      - name: customer_id
        kind: foreign
    filters:
      - name: paid_orders
        expr: status = 'paid'
    extras:
      catalog: polaris_prod
      storage:
        format: Parquet
        paths: ["s3://b/orders/*.parquet"]
      semantic_mapping:
        ordered_at:        order_date
        region:            region_code
        product_category:  prod_cat
        amount_cents:      amount_cents
        order_id:          id
        customer_id:       cust_id
        status:            order_status
      temporal:
        events:
          occurred_at: ordered_at
          grain: Day
```

Notes on this example:

- `dimensions:`, `measures:`, `metrics:`, `keys:`, `filters:` are **flat** at the Dataset level per `#[serde(flatten)]` on the SemanticInterface (per `32 §3.3`). The full canonical roster of these five components is enumerated in `18 §1.1`.
- `metrics:` are composed expressions over Measures (per `18 §1.1`). `avg_order_value = gross_revenue / order_count` is a derived ratio Metric.
- `keys:` declare entity-level identifiers (`kind: primary` / `kind: foreign`, per `18 §2.5`); they participate in `Relationship` resolution and are NOT projected at plan time unless a Request explicitly references them in `fields:`.
- `filters:` declare named saved filters (per `18 §2.4`); a Request may reference them by name (e.g. `filters: [paid_orders]`).
- The `extras.semantic_mapping:` block covers every Semantics that maps to a physical column. Implicit name-match per `15 §10.4` would cover Semantics whose physical name is identical (e.g. `amount_cents → amount_cents` could be omitted); shown explicitly here for clarity.
- `extras.temporal.events:` ratifies the historization shape — `occurred_at: ordered_at` names a Dimension on this Dataset; `grain: Day` is the source granularity.

## 3. Binding Consumer Contract

### 3.1 Single-`Binding` rule

A `Dataset` owns exactly one `Binding`. The structural rule lives in `15 §2.1` (one binding-block per leaf); `21` ratifies the **consumer contract**: every `SimpleStrategy` invocation operates over exactly one `ResolvedBinding`. Plan composition above a `Dataset` (Unionset branching, Grainset routing, Joinset joining) multiplies the number of `Dataset` plans, not the number of `Binding`s per `Dataset`.

The Binding-related authoring surface (`extras.storage:` paths-or-tables, `extras.semantic_mapping:`, `extras.catalog:`) is enumerated in §2.1 rows F / G with cross-refs to `32 §4`, `18 §3`, `18 §10`. `Binding` mechanics — its `SemanticMapping` variants, `PhysicalSource` expansion, per-format schema resolution, `Coverage` mechanics, compile-time resolution flow — are ratified in `15`.

### 3.2 Multi-source fan-out — implicit Unionset (mode `all`)

A single `Binding` may resolve to **multiple** `PhysicalSource`s — e.g. a glob in `extras.storage.paths:` expanding to N files. This case is structurally an **implicit Unionset (mode `all`)**: the planner mechanism is the same as a user-authored Unionset's children union, with per-source pseudo-leaves replacing user-authored DataKind children. The canonical mechanism (sub-aggregation + UNION ALL + conditional final-aggregation, with literals + per-source `Metadata` driving disjointness elision) is owned by `23 §<fan-out-mechanism>` (forthcoming); `21` cross-references it.

Two consumer rules `21` retains for the `Dataset`-specific case:

- **Uniform coverage required.** Every source in `Binding.sources` must serve the same `SemanticInterface` with `Native` `Coverage` for every Request-referenced Semantics. When any source has `NullFill` `Coverage` for a referenced Semantics, the `Dataset` rejects with `COMP_E_2106 SimpleMultiSourceIncompatibleNullFill` (a re-surface of `15 §6.1`'s `COMP_E_0310 UnusableNullFillInNonUnionContext`). To tolerate `NullFill`, the author wraps the `Dataset` in a Unionset (`23`).
- **Per-source `Metadata` materialization.** Per-source `Metadata` Dimensions appear as projected literals in each per-source plan branch (compile-resolved per `15 §10.5`, stored on `ResolvedPhysicalSource.metadata_values`). Never as physical reads. Materialization-vs-pruning rules — unconditional emission with downstream Project-pruning — are deferred to `34` (Q-DS-003).

### 3.3 Pointer to `15`

`15 §§2–10` is the authoritative reference for `Binding` shape, resolution flow, and `SemanticMapping` mechanics. Readers working on `Dataset` plan semantics must read `15` first; `21` assumes fluency.

## 4. Plan-Time Observable Behavior

A `Dataset`'s plan-time observable behavior is the realization of `SimpleStrategy::resolve` (algorithm body in `34 §<SimpleStrategy>`, parked in `_drafts/34_simple_strategy.md` pending `34` drafting). The observable invariants an author or consumer can rely on:

- A Request that names only Dimensions emits no aggregation node — `Scan` + `Project` (+ `Filter` where applicable).
- A Request that names Measures (or Metrics over Measures) emits exactly one aggregation boundary; re-aggregation across the boundary preserves the declared `additivity:` shape (D8).
- **Multi-source fan-out is structurally an implicit Unionset (mode `all`).** A glob or list in `extras.storage.paths:` resolves to N branches, each emitting per-branch resolution (`Scan` + `Project` + optional `Filter` and per-branch sub-aggregation when Measures are requested), combined via `PlanNode::Union`. A **final aggregation** after the union is emitted **only when** the planner cannot prove from literals + per-source `Metadata` that branches' output keys are disjoint; provable disjointness elides the final aggregation since per-branch partials are already correct. Per-source `Metadata` Dimensions appear as projected literals, never as physical reads. (Mechanism shared with Unionset — see `23 §<fan-out-mechanism>` and `34 §<implicit-union>`.)
- Filter pushdown is a planner guarantee; the layer at which a filter sits is `34`'s concern.
- No engine identity, no SQL, no I/O is observable from the plan (I3).

These invariants are stable across the V1 lifetime: any change to `SimpleStrategy`'s algorithm body that violates one of them is a `21` MAJOR per `30 §6.3` and triggers a coordination-cascade per `00 §6`.

## 5. TemporalShape Interaction

### 5.1 Authoring placement

A `Dataset` carries an optional `TemporalShape` declaration on `extras.temporal:` (`LeafExtras` field per `32 §4`). The v1 shape kinds are `Timeseries`, `Events`, `Snapshot`, and `Scd` (the v1 `ScdType` roster is `{Type1, Type2}`); each variant carries its own identifying-Dimension field (`occurred_at:` for `Timeseries` / `Events`, `snapshotted_at:` for `Snapshot`, `valid_from:` / `valid_to:` for `Scd`). Full per-variant grammar lives in `17`; `18 §3` carries the canonical-entity definitions.

Example excerpt:

```yaml
datasets:
  - name: web_events
    dimensions: [ ... ]
    measures:   [ ... ]
    extras:
      catalog:           ...
      storage:           ...
      semantic_mapping:  ...
      temporal:
        events:
          occurred_at: ordered_at
```

### 5.2 Cascade behavior — per-variant

When a `Dataset` is nested under a `ComplexDataKind`, the cascade behavior of `extras.temporal.kind` is **variant-determined** by the parent Complex:

- **Under Unionset** — children's shapes (kind + grain) must be equivalent. Silent leaves inherit ancestor's `kind` if authored; restated values must agree.
- **Under Grainset** — children's `kind` must be equal; grains must yield ≥ 2 unique values across children. Silent leaves inherit ancestor's `kind`; restated values must agree. `grain` is leaf-only (`LeafExtras` carries it; `ComplexExtras` does not).
- **Under Joinset** — children may have different shapes (kind and grain). Joinset's own `extras.temporal:` declaration, if present, is **authoritative for upward propagation**; children's shapes are advisory / planner-information only. Joinset is a **cascade boundary**.

The per-variant rules are ratified in their respective Complex chapters (`22 §<temporal-rule>`, `23 §<temporal-rule>`, `24 §<temporal-rule>`, all forthcoming). `21` describes the `Dataset` author's experience: at Root scope, a `Dataset` authors its own `extras.temporal:` freely; at Nested scope, the parent variant's rule applies.

### 5.3 Shape-less `Dataset`s

A `Dataset` with `extras.temporal: None` has no author-declared historization axis. Its plan emission runs with no shape-specific rules activated — no rollup gating, no advisory warnings, no as-of support. This is the common path for staging / flat-fact datasets without ratified temporal structure.

### 5.4 Scope boundary with `17`

`21` ratifies only:

- the `extras.temporal:` field carriage on `LeafExtras` (cross-ref `32 §4`),
- the per-variant cascade behavior at the *Dataset's* perspective (cross-ref to per-variant rules in `22` / `23` / `24`).

`21` does NOT ratify the `TemporalShape` enum variants (`17 §`*), the SCD subtype catalog (`17 §`*), the shape × grain rollup matrix (`17 §*`), the `AsOf` join variant (`17 §*` + `16`), or the advisory-warning predicates (`17 §*`).

## 6. Grain Interaction

### 6.1 Authoring placement

A `Dataset` carries an optional `grain` declaration **inside its `TemporalShape` variant** — e.g. `extras.temporal.timeseries.grain: Day`, `extras.temporal.snapshot.grain: Month` — per `18 §3.3`. There is no top-level `grain:` field on `LeafExtras`; `grain` is intrinsic to the shape variant. The `Grain` enum (`Second`, `Minute`, `Hour`, `Day`, `Week`, `Month`, `Year`) and its total coarseness order are ratified in `13 §5`.

Example:

```yaml
datasets:
  - name: daily_snapshots
    dimensions: [ ... ]
    measures:   [ ... ]
    extras:
      catalog:           ...
      storage:           ...
      semantic_mapping:  ...
      temporal:
        snapshot:
          snapshotted_at: as_of_date
          grain: Day
```

A `Dataset` without an `extras.temporal:` block has no author-declared `grain`.

### 6.2 Plan-time grain consultation

Two observable invariants involve `grain`:

- **Rollup coarsening.** When a Request rolls up a temporal Dimension, the target grain must be **at least as coarse** as the `Dataset`'s declared `grain`. A `grain: Day` `Dataset` rejects a `Hour`-target rollup with `PLAN_E_2102 RequestGrainFinerThanSource`. Rolling up to a coarser grain is legal; the planner emits a `DateTrunc` at the requested grain.
- **Shape-gated legality.** When `extras.temporal.kind` is also declared, the rollup is shape-gated per `17`'s matrix. The detailed matrix lives in `17`; `21` only identifies the hook.

### 6.3 Scope boundary with `13` and `22`

- `13 §5` ratifies the `Grain` enum and its coarseness order.
- `22` ratifies grain routing — the Grainset variant that dispatches a Request across children with different `grain` declarations. `21` does not ratify routing; a bare `Dataset` has at most one declared grain (per shape variant).
- `17` ratifies the `TemporalShape × Grain` legality matrix.

## 7. Validation Preconditions — `VALID_E_2100`–`2199`

`21` allocates its `VALID_E` sub-range for Dataset-specific structural checks. `validate` runs these against the parsed `SemanticModel`. Every check accumulates — `validate` collects all failures before returning.


| Code           | Variant                                                                                    | Trigger                                                                                                                                                                                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VALID_E_2101` | `SimpleMissingStorage { data_kind }`                                                       | A `Dataset`'s `extras.storage:` is absent or contains neither `paths:` nor `tables:`. (Authored `extras: {}` is structurally allowed at parse, caught here at validate.)                                                                                                                                        |
| `VALID_E_2102` | `SimpleStorageBothPathsAndTables { data_kind }`                                            | A `Dataset`'s `extras.storage:` declares BOTH `paths:` and `tables:`. The two are mutually exclusive per `18 §3` / `32 §4.1`.                                                                                                                                                                                   |
| `VALID_E_2103` | `SimpleNestedSelfReference { data_kind, semantics }`                                       | A `Dataset`'s `extras.semantic_mapping` carries a `Computed` entry whose expression references the Dataset's own semantic name as both LHS and a transitive dependency, forming a cycle. The underlying cycle-detection emits `EXPR_E_0203 CyclicReference` (`19 §3.5`); `21` reserves this code as the Dataset-structural framing. |
| `VALID_E_2104` | `SimpleInterfaceEmpty { data_kind }`                                                       | A `Dataset` (Public form) has zero Semantics across `dimensions:`, `measures:`, `metrics:`, `filters:`, `keys:`. A Dataset with no queryable surface is rejected. (Does not apply to `NestedDataset` — Nested form has no interface per `26 §3`.)                                                               |
| `VALID_E_2105` | reserved                                                                                   | Reserved. Post-thirteenth-pass: `grain` lives inside `extras.temporal.<variant>.grain` per `18 §3.3`, so structural shape-vs-grain conflicts cannot arise at the YAML level.                                                                                                                                    |
| `VALID_E_2106` | `SimpleSemanticMappingReferencesSelf { data_kind, semantics }`                             | A `SemanticMapping` `Column { name }` entry uses the Dataset's own semantic name as the physical column name. Physical-vs-semantic name overlaps within the same kind are user errors flagged here.                                                                                                             |
| `VALID_E_2107` | `SimpleTemporalShapeMissingIdentifier { data_kind, variant }`                              | An `extras.temporal.<variant>:` block omits its required identifying field (`events:` without `occurred_at:`, `snapshot:` without `snapshotted_at:`, `scd:` without `valid_from:` / `valid_to:`). Per-variant identifier roster lives in `18 §3.3`.                                                             |
| `VALID_E_2108` | `SimpleTemporalShapeIdentifierNotInInterface { data_kind, variant, identifier }`           | An `extras.temporal.<variant>` identifier names a Semantics not present in the Dataset's `semantic_interface`.                                                                                                                                                                                                  |
| `VALID_E_2109` | `SimpleTemporalShapeIdentifierWrongType { data_kind, variant, identifier, declared_type }` | An `extras.temporal.<variant>` identifier resolves to a Semantics whose declared `data_type:` is not a temporal type (`Date` / `Time` / `Timestamp`). Variant-specific roster in `17`.                                                                                                                          |


**Extensibility.** Codes `2110`–`2199` reserved; MINOR per `30 §6.3`.

**Re-surfaced errors from foundations docs.** Structural issues belonging to other docs (duplicate Dimension name — `11 §3`; mismatched `data_type:` — `11 §5.1`; ill-formed `SemanticMapping` — `15 §5.6`) are reported by the owning doc's code ranges; `21` does not re-codify. Diagnostics fill in the `data_kind` location for context.

## 8. Compile Preconditions — `COMP_E_2100`–`2199`

`21` allocates its `COMP_E` sub-range for Dataset-specific compile-time checks. `compile` runs these after `validate` has passed. Fail-fast — the first compile error aborts the Dataset's resolution.


| Code          | Variant                                                                                       | Trigger                                                                                                                                                                                                                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COMP_E_2101` | `SimpleBindingResolutionFailed { data_kind, cause }`                                          | A wrapper re-surface: `Binding`-level resolution (per `15 §10`) raised any of `COMP_E_02xx` / `COMP_E_03xx`. `21` attaches `data_kind` context but does not re-codify the inner cause.                                                                                                                                                                    |
| `COMP_E_2102` | `SimpleSourceGlobEmpty { data_kind, pattern }`                                                | The Dataset's `extras.storage.paths:` glob expanded to zero sources. Equivalent to `COMP_E_0301 NoSourcesMatched` per `15 §3.5.7`; re-surfaced under the Dataset's diagnostic context.                                                                                                                                                                    |
| `COMP_E_2103` | `SimpleSchemaInferenceRequiredForBinding { data_kind, source }`                               | A Dataset's CSV/JSON source has no `declared_schema:` and inference would violate I4 for the Dataset's declared `extras.temporal:` shape (e.g. `temporal.events:` demands column-level `Timestamp` typing; inference yielding `String` is not recoverable without explicit authoring). Advisory upgrade to error for Datasets with temporal declarations. |
| `COMP_E_2104` | `SimpleTemporalIdentifierTypeMismatch { data_kind, variant, identifier, declared, physical }` | The identifying column for an `extras.temporal.<variant>:` declaration resolves (via `SemanticMapping`) to a `PhysicalExpr` whose inferred type disagrees with the variant's required type. Distinct from `COMP_E_0315` (`15 §9.1`) — the temporal-variant context narrows the required type.                                                             |
| `COMP_E_2105` | `SimpleGrainNotSupportedBySource { data_kind, declared_grain, source }`                       | A Dataset declares `extras.temporal.<variant>.grain: Hour` but a source's declared partition transform (via `37`) reports coarser-than-Hour granularity. Reserved for the Dataset-level pre-routing check.                                                                                                                                                |
| `COMP_E_2106` | `SimpleMultiSourceIncompatibleNullFill { data_kind, source_index, semantics }`                | Wrapper re-surface of `15 §6.1`'s `COMP_E_0310 UnusableNullFillInNonUnionContext`, with the Dataset's name attached. Dataset rejects `NullFill`; the author wraps in a Unionset (`23`).                                                                                                                                                                   |
| `COMP_E_2107` | `SimpleComputedReferencesUnresolvableEntity { data_kind, semantics, unresolved }`             | Wrapper re-surface of `EXPR_E_0201 UnknownReference` (`19 §3.3` / `19 §8.1`). A `Computed` Dimension / Measure's `SemanticExpr` referenced a typed semantic leaf not in the Dataset's `semantic_interface` and with no `Relationship` path.                                                                                                                                     |


**Extensibility.** Codes `2108`–`2199` reserved; MINOR per `30 §6.3`.

**Wrapper vs pass-through.** Codes `2101`, `2102`, `2106`, `2107` re-surface foundation-level codes (`15` / `19`). Round-1 default is selective wrapping — wrap when Dataset-level context materially aids the operator, pass-through otherwise. Final ratification per Q-DS-002 (`open/21_questions.md`); pending `30 §6.2` / `34` drafting.

## 9. Plan-Stage Rules — `PLAN_E_2100`–`2199`

`21` allocates its `PLAN_E` sub-range for Dataset-specific checks that depend on Request shape. Most Request-level checks live in `34`; `21`'s range covers the narrow slice that specifically depends on Dataset shape.


| Code          | Severity | Variant                                                             | Trigger                                                                                                                                                                                                                                                                           |
| ------------- | -------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PLAN_E_2101` | Error    | `SimpleRequestReferencesUnknownSemantics { data_kind, semantics }`  | A Request's fields list or filter list references a Semantics name not present in the Dataset's `semantic_interface`. Hard error for Requests with explicit `from: <dataset>`; field-first resolution per `16 §11` routes via `34`.                                               |
| `PLAN_E_2102` | Error    | `RequestGrainFinerThanSource { data_kind, requested, declared }`    | A Request rolls a temporal Dimension finer than the Dataset's declared `extras.temporal.<variant>.grain:`. See §6.2.                                                                                                                                                              |
| `PLAN_W_2101` | Warning  | `LossyMultiSourceReaggregation { data_kind, measure }`              | Multi-source `Binding` + final-aggregation cannot be elided (no provable disjointness from literals + `Metadata` per §4) + Measure uses `COUNT_DISTINCT` or `AVG`. Advisory; the plan still executes but is lossy.                                                                |
| `PLAN_W_2102` | Warning  | `ShapeAdditivityMismatch { data_kind, measure, shape, additivity }` | A Measure's declared `additivity:` and the Dataset's `extras.temporal:` shape appear inconsistent per the advisory matrix in `17`. Advisory; plan proceeds.                                                                                                                       |
| `PLAN_E_2103` | Error    | `SimpleNullFillAtPlan { data_kind, source_index, semantics }`       | A multi-source `Binding` whose `Coverage` includes `NullFill` reached plan-time under a bare Dataset consumer. Should have been caught at compile (`COMP_E_2106`); defensive check fires this error if the SemanticManifest was hand-constructed or loaded from a stale artifact. |
| `PLAN_E_2104` | Error    | `SimpleEmptyPlanRequested { data_kind }`                            | A Request asks for zero fields. A Dataset plan with zero output columns is ill-formed; rejected.                                                                                                                                                                                  |
| `PLAN_E_2105` | Error    | `SimpleAggregationWithoutMeasure { data_kind, grouping }`           | A Request explicitly carries aggregation but names no Measure / Metric. (`SimpleStrategy` elides aggregation per §4 when no Measures are present; this fires if a future API carries explicit aggregation.)                                                                       |


**Extensibility.** Codes `2106`–`2199` reserved; MINOR per `30 §6.3`.

**Scope note.** Request-routing errors (which DataKind owns which Semantics, field-first vs explicit-`from` resolution, composition synthesis) are `34`'s concern. `21` only ratifies the narrow slice where Dataset shape is the root cause.

## 10. Worked Example

### 10.1 YAML

The example below is a **focused excerpt** — only the SemanticInterface elements needed to illustrate the multi-source / `Metadata` / disjointness mechanism (`dimensions:` + `measures:`) are shown. See [§2.3](#23-authoring-example--full-semanticinterface-surface) for a Dataset that exercises every authoring slot (`dimensions / measures / metrics / keys / filters`).

```yaml
datasets:
  - name: orders
    description: "Order-line fact dataset partitioned by year."
    dimensions:
      - name: ordered_at
        data_type: Timestamp
      - name: region
        data_type: String
      - name: year_dir
        data_type: String
        type:
          metadata:
            source:
              path:
                token: 0   # 0-indexed; raw segment per `15 §8.1.1`
    measures:
      - name: gross_revenue
        agg: sum
        expr: amount_cents
        data_type: Long
    extras:
      catalog: polaris_prod
      storage:
        format: Parquet
        paths:
          - "s3://b/orders/year=*/*.parquet"
      semantic_mapping:
        ordered_at:    ordered_at
        region:        region
        amount_cents:  amount_cents
        # `year_dir` not authored — recipe lives on the Dimension `type:`
        # block and is compile-synthesized into the SemanticMapping
        # per `15 §10.4` step 4.0.
      temporal:
        events:
          occurred_at: ordered_at
          grain: Day
```

### 10.2 Request

```
Request {
  from: orders,
  fields:  [ ordered_at (rollup Month), region, gross_revenue ],
  filters: [ region = 'EU' ],
}
```

### 10.3 Observable plan shape

The glob in `extras.storage.paths:` expands to two sources (`year=2024/*.parquet`, `year=2025/*.parquet`); fan-out is structurally an implicit Unionset (mode `all`) per §4. Each branch resolves independently and contributes a per-branch sub-aggregation; the union is followed by a final aggregation since the branches' GROUP BY keys (`DateTrunc(ordered_at, Month)`, `region`) are not provably disjoint from literals + `Metadata` (`year_dir` is not in the Request's `fields:`, so the disjointness predicate cannot use it).

```
Project                              (final shape — rename, ordering)
  Agg                                (final aggregation — provably required)
    Union                            (two branches; mode `all`)
      Agg                            (per-branch sub-aggregation, source 0)
        Filter: region = 'EU'
          Project                    (rename + per-branch metadata literal)
            Scan                     (year=2024/*.parquet)
      Agg                            (per-branch sub-aggregation, source 1)
        Filter: region = 'EU'
          Project                    (rename + per-branch metadata literal)
            Scan                     (year=2025/*.parquet)
```

### 10.4 Reading key

- The `extras.storage.paths:` glob expands to two `Scan`s combined by `Union`; this is the implicit Unionset (mode `all`) per §4. The canonical fan-out mechanism is owned by [`23 §4.7`](./23_unionset.md); `21 §3.2` cross-references it.
- `Filter` for `region = 'EU'` is pushed below the per-branch aggregation. Exact placement is `34`'s concern; `21` ratifies only that pushdown is a planner guarantee.
- Per-branch `Agg` produces partial aggregates; the final `Agg` above the `Union` is required because the GROUP BY keys aren't provably disjoint from literals + `Metadata`.
- `year_dir` is a `Metadata` Dimension extracted from the path token; per-source literals (`"year=2024"` for source 0, `"year=2025"` for source 1) are emitted at the per-branch `Project`. Since `year_dir` is not in the Request's `fields:`, it does not appear in the final projection.
- Algorithm body — exact `Project` columns, `Cast` placement, `DateTrunc` emission, partial-aggregate column staging — lives in `34 §<SimpleStrategy>` (currently parked in [`../_drafts/34_simple_strategy.md`](../_drafts/34_simple_strategy.md)).

### 10.5 Simpler case — single-source, Dimensions-only

For `Request { from: orders, fields: [region, ordered_at], filters: [], ... }` against a single-source `extras.storage.paths: ["s3://b/orders/2024.parquet"]`:

```
Project                              (rename only)
  Scan                               (single source)
```

No `Filter`, no `Union`, no `Agg` — the simplest observable shape per §4's "A Request that names only Dimensions emits no aggregation node".

## 11. Open Items

Round-1 drafting surfaced five open questions parked in `[../questions/open/21_questions.md](../questions/open/21_questions.md)`. Post-thirteenth-pass cascade (2026-04-30) review confirmed Round-1 defaults remain consistent with the post-rebase architecture.

- **Q-DS-001** — Structural label for nested `Dataset` under a Complex. **CLOSED at Option (A).** Migrated to `[../questions/closed/21_questions.md](../questions/closed/21_questions.md)`. At nested scope, `body.base.name` is the structural anchor; no separate `label:` field on `NestedDataset`.
- **Q-DS-002** — Wrapper code discipline for re-surfaced errors. Deferred to `30 §6.2` / `34` drafting; Round-1 default is selective wrapping (§8 codes `2101`, `2102`, `2106`, `2107`).
- **Q-DS-003** — Multi-source per-branch metadata emission. Deferred to `34 §<optimizer>` drafting; Round-1 default is unconditional emission with downstream Project-pruning.
- **Q-DS-004** — Temporal-shape identifier on Computed Dimensions. Deferred to `17` drafting; Round-1 default is permissive (any Dimension with a temporal `data_type:` qualifies).
- **Q-DS-005** — Re-aggregation-skip predicate over Computed Dimensions. Deferred to `34 §<implicit-union>` drafting; Round-1 default is V1-only-checks-`Metadata` (Computed extension is a future enhancement; correctness preserved via lossy advisory).

---

**End of document.** Ratified decisions are inline throughout §§1–9; open items live in `[../questions/open/21_questions.md](../questions/open/21_questions.md)`.

## Testing Strategy

- Validation preconditions `VALID_E_2100`–`2199` (§7) are proven by malformed `Dataset` fixtures asserting `VALID_E_2101 SimpleMissingStorage`, `VALID_E_2102 SimpleStorageBothPathsAndTables`, `VALID_E_2104 SimpleInterfaceEmpty`, and `VALID_E_2107 SimpleTemporalShapeMissingIdentifier`.
- Compile preconditions `COMP_E_2100`–`2199` (§8) are proven by resolution-failure fixtures asserting `COMP_E_2101 SimpleBindingResolutionFailed` (wrapper), `COMP_E_2102 SimpleSourceGlobEmpty`, and `COMP_E_2106 SimpleMultiSourceIncompatibleNullFill` (re-surface of `15 §6.1` `COMP_E_0310`).
- Multi-source fan-out (§3.2, §4) is proven by a glob expanding to N sources emitting an implicit Unionset (mode `all`) — per-branch sub-aggregation + `PlanNode::Union` + conditional final aggregation — with `NullFill` coverage under a bare `Dataset` rejected via `COMP_E_2106`.
- Plan-time observable invariants (§4) are proven by asserting a Dimensions-only Request emits `Scan`+`Project` with no `Agg`, a Measure Request emits exactly one aggregation boundary preserving declared `additivity:` (D8), and a rollup finer than the source grain raises `PLAN_E_2102 RequestGrainFinerThanSource` (§6.2).
- Plan-stage advisories/errors (§9) are proven by asserting `PLAN_W_2101 LossyMultiSourceReaggregation` and `PLAN_W_2102 ShapeAdditivityMismatch` fire as warnings, and a zero-field Request raises `PLAN_E_2104 SimpleEmptyPlanRequested`.

See the shared harness: [`../testing/50_behavioral_contract_gates.md`](../testing/50_behavioral_contract_gates.md).
