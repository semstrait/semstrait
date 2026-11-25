---
id: 11-semantics-element-catalog
status: ratified
version: 2.0
impl: implemented
---

# Semantics Element Catalog

Semantics element catalog — Dimension, Measure, Metric, Filter, Key role definitions and field catalogs. This leaf owns the role and field-catalog definitions; struct shapes are owned by `18 §4`–`§9`, [additivity](additivity.md) owns the Additivity enum and planner contract, and [constraint](constraint.md) owns the Constraint framework.

## 6. Semantics Element Catalog

Five element types. Scope and role definitions here; structural specs in downstream docs.

### 6.1 Dimension

A Dimension is a named axis for grouping / filtering. It has a `name`, a `data_type` (13), an optional `type` discriminator (temporal / metadata / plain) ratified in `13`, an optional `description`, an optional `expr`, and optional `constraint` blocks (§8).

Dimensions are the only Semantics type eligible for participation in Keys (§6.5).

#### 6.1.1 Field catalog

| Field | Required? | Kind | Role |
|---|---|---|---|
| `name` | yes | shape | global identifier (§3) |
| `data_type` | yes (in ≥1 occurrence) | shape | canonical `DataType` variant (13) |
| `type` | no | shape | discriminator: `plain` (default) / `temporal` / `metadata`. Determines which auxiliary fields are meaningful (13) |
| `description` | no | shape | prose; unifies across occurrences |
| `expr` | no | resolution-variant | per-DataKind expression computing this Dimension's value (14) |
| `constraint` | no | shape | declarative rules (§8) |
| `grains` | no (temporal only) | shape | list of `Grain` axes this Dimension serves as the time axis for (13, 17) |
| `metadata` | no (metadata only) | shape | structured `MetadataDimension { path?, partition? }`; v1 ratifies path-segment extraction (`path.token: N`), `partition.level:` deferred to v2 — full spec in 13 §4.7. |

**Interaction with `Grain`.** When `type: temporal`, the `grains:` list declares which rollup levels this Dimension's source data supports. A DataKind's declared `Grain` must be one of the listed `grains:` values on its temporal Dimension (ratified in `13` and `17`). Non-temporal Dimensions do not participate in `Grain`.

### 6.2 Measure

A Measure is a numeric quantity over a DataKind, aggregated according to its `agg`. It carries `additivity` (§7), a `data_type`, an `agg` function, an optional `expr` (horizontal transformation applied before aggregation), optional `constraint`. A Measure is always numeric (`13` constrains the allowed `data_type` set).

#### 6.2.1 Field catalog

| Field | Required? | Kind | Role |
|---|---|---|---|
| `name` | yes | shape | global identifier |
| `data_type` | yes (in ≥1 occurrence) | shape | canonical numeric `DataType` variant (13) |
| `agg` | yes (in ≥1 occurrence) | shape | aggregation function: `sum` / `avg` / `min` / `max` / `count` / `count_distinct` / … (full list in 13 / 14) |
| `additivity` | no (default `Additive`) | shape | aggregation-compatibility classification (§7) |
| `description` | no | shape | prose |
| `expr` | no | resolution-variant | horizontal transformation applied **before** aggregation (e.g. `revenue - discount` then `SUM(...)`); per-DataKind (14) |
| `constraint` | no | shape | declarative rules (§8) |

**Interaction between `agg` and `additivity`.** These are orthogonal fields with related consequences:

- `agg` specifies the aggregation function.
- `additivity` specifies whether `agg` composes mechanically under grain rollup.
- The planner uses both: `agg` drives the aggregate node; `additivity` drives whether that aggregate may be layered across grains or must be recomputed from underlying values at the queried grain.

Pre-ratified planner shape (details in `20–25`):

| `agg` | Default semantics under mechanical rollup |
|---|---|
| `sum` | `Additive` by definition; SemiAdditive when author restricts axes (§7.4) |
| `min` / `max` | Idempotent under refinement (MIN of MINs is the overall MIN); treated as `Additive` for rollup purposes |
| `count` / `count_distinct` | `count` is `Additive`; `count_distinct` is `NonAdditive` (recomputed from row identities at the queried grain) |
| `avg` | `NonAdditive` mechanically (weighted-AVG requires the weight); planner materializes `SUM(x)/COUNT(x)` at the queried grain |

These defaults are author-overridable via explicit `additivity:` as long as the override is consistent across occurrences (§5.1).

### 6.3 Metric

A Metric is a **derived** quantity computed from Measures or other Metrics via `expr`. It does not aggregate directly (no `agg:` field is allowed); its `expr` references Measure and Metric names and may combine them through any canonical function in `14`'s `FunctionRegistry`. A Metric carries `additivity` (§7), a `data_type`, an `expr` (required — a Metric without an `expr` is not a Metric), optional `constraint`.

**Additivity on Metrics.** Most Metrics that express a ratio (`cpc = cost / clicks`, `ctr = clicks / impressions`) are `NonAdditive` — mechanical rollup is incorrect; the correct behavior is to recompute at the queried grain from underlying Measures. Some Metrics that are linear combinations (`revenue_plus_shipping = revenue + shipping`) are `Additive`. Authors state Additivity on Metrics for the same reasons as on Measures (§7.2). The default `Additive` applies equally; for Metrics, this default is often wrong, and authors should review.

#### 6.3.1 Metric vs. Measure at plan time

The structural distinction at plan time (ratified in `20–25`):

- A **Measure** is aggregated: its resolution produces an aggregate plan node (`SUM(expr)`, `AVG(expr)`, etc.) in the plan tree's aggregate layer.
- A **Metric** is a projection: its resolution produces a projection node over the already-aggregated outputs of its referenced Measures/Metrics.

Concretely, for a Request naming `cpc` at day grain:

```
Aggregate    [grain = day]
  ├── cost     (Measure; SUM(spend_amount))
  └── clicks   (Measure; SUM(click_count))
Project
  └── cpc  = cost / clicks   (Metric; NonAdditive projection over aggregate outputs)
```

For an `Additive` Metric like `revenue_plus_shipping`, the same structure applies — the projection is safe to compute at any grain because linear combinations distribute over aggregation. For `NonAdditive` ratios, the projection MUST be computed at the queried grain; the planner cannot layer the Metric across grains (it would roll up a ratio of sums, which is not a sum of ratios).

The `expr` subset legal on Metrics is a sub-grammar of the authoring expression grammar ratified in `14`: Measure/Metric name references, `FunctionRegistry` calls, literals. No column references (Metrics operate on aggregate outputs, not raw rows). No `agg:` field. Detailed grammar in `14`; rollup policy under each Additivity value in `20–25`.

### 6.4 Filter

A Filter is a named boolean expression that restricts rows. It lives either at Model root (global Filter usable across DataKinds) or inside a Kind scope (DataKind-scoped Filter). A Filter's `expr` follows `14`'s boolean expression rules.

#### 6.4.1 Application semantics

Filters are **opt-in per Request**. A Filter named `F` is applied only when the Request explicitly names `F` in its `filters:` list.

Initial design deliberately avoids always-on Filters. Rationale: always-on Filters hide Request semantics at plan time — two identical-looking Requests can produce different results depending on which DataKind they resolve against. Opt-in keeps the Request's behavior locally legible.

**Future design — Filter-injection.** A future extension MAY let a Measure / Metric / DataKind declare Filter(s) as required (informally: "any Request using `premium_customer_revenue` must apply `is_premium_customer`"), with the planner auto-injecting them and recording the Constraint-injected provenance in `SemanticPlan` lineage per `35`. The current model has no carrier for this — see `[TD-REQUIRES-MECHANISM]` in `§8.5.2`. The field name is deliberately left open (`requires:` is reserved per the source comment in `semstrait-common::constraints` — `// This is NOT \`requires\``).

### 6.5 Key

A Key is a **structural declaration** on a top-level DataKind naming an ordered list of Dimension references that together uniquely identify a row at the DataKind's grain. Keys are not Semantics in the same sense as the preceding four — they do not have their own `data_type`, `expr`, or `description`. They are arrangements of Dimensions.

**Key placement rules:**

- Declared only on top-level DataKinds (not at Model root, not on nested kinds).
- Each top-level DataKind declares at most one primary Key and zero or more additional unique/foreign Keys.
- Keys do not appear in Tier-1 (there is no global Key registry).
- A Dimension listed in a Key must be in the declaring DataKind's interface (declared there or ref'd from Tier-1).

#### 6.5.1 Key kinds

Three kinds, selected by the `kind:` field (defaulting to `primary` when absent in the single-Key case):

| Kind | Multiplicity per DataKind | Purpose |
|---|---|---|
| `primary` | at most one | Identifies rows at the DataKind's grain. Drives row identity for Unionset dedup, Grainset rollup, and Joinset cardinality inference. |
| `unique` | zero or more | Additional uniqueness claims distinct from the primary Key (e.g. natural Key vs. surrogate Key). |
| `foreign` | zero or more | References another top-level DataKind's primary or unique Key. Must correspond to a declared `Relationship` (§9, ratified in `16`). |

**Cardinality implications.** A `foreign` Key pins one end of a `Relationship`'s Cardinality specification. `16` ratifies how a declared `Relationship` and the referenced DataKinds' Keys together determine the Cardinality (one-to-one / one-to-many / many-to-one / many-to-many).

**Uniqueness Preconditions.** `compile` does NOT verify that data is actually unique — that is a runtime property of the physical source. What `compile` DOES verify:

- `N-C7` — every member of a Key names a Dimension (not a Measure / Metric / Filter).
- `N-C8` — every member of a Key is in the declaring DataKind's interface.
- (new) `N-C9` — for a `foreign` Key, the referenced target DataKind + target Key pair exists, and a `Relationship` is declared between the two DataKinds.

These are compile-time structural checks; they ensure the Key declaration is well-formed. Runtime uniqueness (does SELECT DISTINCT over the Key columns match the full row count?) is out of scope.

## Testing Strategy

- Dimension field catalog integrity (§6.1.1 table: all fields parse, type-check, and unify correctly)
- Measure field catalog integrity (§6.2.1 table)
- Metric field catalog integrity (§6.3 + §6.3.1 plan-time distinction)
- Filter application semantics (opt-in per Request; no always-on)
- Key placement rules (top-level DataKinds only, no Model root, no nested kinds)
- Key kinds validation (§6.5.1: primary/unique/foreign multiplicity + Preconditions N-C7/N-C8/N-C9)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
