---
prereqs: [00, 10, 11, 13, 14, 14b]
authoritative-for:
  - the shared Semantics pools (`dimensions:`, `measures:`, `metrics:`) at the root level and their per-DataKind reference / override grammar
  - the `Relationship` struct and its companion `RelationshipId` newtype — unified shape used at both root `relationships:` and `JoinsetBody.relationships`
  - `JoinType`, `Cardinality`, `Directionality`, `JoinKeyExprPair`
  - the `TemporalShape` type hierarchy — `TemporalShape` struct, `TemporalShapeKind` enum, per-variant `*Body` structs, `ScdType` (v1 roster `{Type1, Type2}`); `Grain` is consumed via `TemporalShape.grain` but the `Grain` enum itself is owned by `13 §3.1`
  - the `Dimension` struct and `DimensionType` roster
  - the `Measure` struct, `AggregationType` roster, `AdditivityType` roster
  - the `Metric` struct
  - the filter taxonomy — `DataKindFilter` (DataKind-level, user-facing predicate) vs `AggregationFilter` (Measure / Metric-level, conditional aggregation)
  - the `AiContext` struct
  - the `Keys` struct (`primary`, `unique`, `foreign`) with bare-name entries
  - the `SemanticMapping` / `SemanticMappingValue` value shape at the model-authoring layer
  - the orphan-binding policy for root-pool Semantics
  - the `SR-E-*` structural-rule codes for entity-level validation
refined-by:
  - 15 (`foundations/15_mapping_and_binding.md` — the compile-time `Binding` process that consumes `SemanticMapping` values ratified in §10)
  - 16 (`foundations/16_composition.md` — how `Relationship` drives implicit composition, `ComposedSemanticInterface` construction, `Joinset` path synthesis)
  - 17 (`foundations/17_temporal_shape.md` — planner-level semantics of `TemporalShape` variants; shape × grain rollup matrix; `AsOf` forward-reference design)
  - 19 (`foundations/19_categories.md` — `MeasureCategory` / `MetricCategory` enums + body structs + implicit-constraint contract per category; SR-E-13 … SR-E-19 canonical home; YAML grammar for `category:` collapsed wrapper; expandability invariants — `SR-CAT-FWD`, `SR-CAT-CLOSED`, `SR-E-19`)
  - 20 (`data-kinds/20_taxonomy.md` — DataKind lifecycle hooks consuming these entity types)
  - 21 / 22 / 23 / 24 (`data-kinds/*.md` — per-variant YAML carriage of the entity types ratified here)
  - 25 (`data-kinds/25_applicability_matrix.md` — per-variant × entity-type cross-cuts)
  - 26 (`data-kinds/26_nesting_matrix.md` — SR-E-8 Grainset-child grain rule)
  - 30 (`apis/30_api_contracts.md` — error-code allocation for `SR-E-*`)
  - 32 (`apis/32_semstrait_model.md` — root YAML shape; hosts `relationships:` and the shared pools this doc ratifies; SR-* enforcement)
  - 32b (`apis/32b_catalogs_yaml.md` — catalog grammar)
  - 33 (`apis/33_semstrait_manifest.md` — Manifest-layer `Resolved*` counterparts of the types ratified here)
  - 34 (`apis/34_semstrait_planner.md` — planner consumption of resolved entity types)
  - 35 (`apis/35_semstrait_ir.md` — `PlanNode::Join` carriage of `JoinType`)
---

# 18. Canonical Entity Types

`18` is the consolidated specification for the ratified entity types that populate a `SemanticModel`: shared Semantics pools, relationships, temporal shapes, dimensions / measures / metrics, filters, AI context, keys, and the model-authoring `SemanticMapping` value shape. `32` fixes the root YAML shape and the `DataKind` hierarchy (an apis-layer concern); `18` fixes the entity shapes nested inside (a foundations-layer concern — these types cross-cut every DataKind variant, Manifest, Planner, and IR surface).

> **Reader's note (structural placement).** This doc originally landed as `apis/32c_entities.md` in the late-April 2026 entity-ratification pass. It was promoted to the foundations layer (`foundations/18_entities.md`) in the 2026-04-17 consolidation pass because the types it defines are structurally foundational — they cross-cut every `2x` data-kind variant, every `3x` api surface, and every planner/adapter consumer. Per the directionality rule in `00 §8`, canonical definitions belong in the lowest-numbered doc that owns them; the promotion places entity types in their correct layer. Section numbering is unchanged from `32c` — every `18 §N` was `32c §N` in the prior revision.

Every struct in this document is `#[non_exhaustive]` and every enum is `#[non_exhaustive]` per I10, unless a specific note overrides.

## Table of Contents

1. [Shared Semantics Pools & Reference Grammar](#1-shared-semantics-pools--reference-grammar)
2. [`Relationship`](#2-relationship)
3. [`TemporalShape`](#3-temporalshape)
4. [`Dimension`](#4-dimension)
5. [`Measure`](#5-measure)
6. [`Metric`](#6-metric)
7. [Filter Taxonomy](#7-filter-taxonomy)
8. [`AiContext`](#8-aicontext)
9. [`Keys`](#9-keys)
10. [`SemanticMapping` Value Shape](#10-semanticmapping-value-shape)
11. [Structural Rules (SR-E-*)](#11-structural-rules-sr-e-)

---

## 1. Shared Semantics Pools & Reference Grammar

### 1.1 Two authoring locations

A Semantic (`Dimension`, `Measure`, or `Metric`) may be authored at exactly one of two locations:

- **Root pool** — under the top-level `dimensions:` / `measures:` / `metrics:` arrays on the `SemanticModel`. Root-pool entries are globally unique per carrier (`dimensions["revenue"]` and `measures["revenue"]` can coexist; two `dimensions["revenue"]` cannot).
- **Inline on a DataKind's `SemanticInterface`** — authored under the Public form of a data kind (`Dataset` / `Grainset` / `Unionset` / `Joinset`). Inline Semantics are scoped to their data kind; cross-kind visibility is via the relationship graph.

A root-pool Semantic may be **referenced** at any number of DataKinds, optionally with a local override (§1.3). A root-pool Semantic with no references and no physical binding is an orphan (§1.4).

### 1.2 Reference site grammar

A DataKind's `SemanticInterface` lists Semantics in two forms — inline or referenced. YAML:

```yaml
datasets:
  - name: orders
    dimensions:
      # Inline declaration (DataKind-local).
      - name: order_id
        data_type: string
        type: categorical

      # Reference to a root-pool Dimension.
      - ref: country

      # Reference with a local expression override.
      - ref: revenue_bucket
        expr: case_when(
                revenue > 1000, 'high',
                revenue > 100,  'medium',
                                'low'
              )
    measures:
      # Reference to a root-pool Measure with a local expression override.
      - ref: total_revenue
        expr: amount_cents * 0.01
```

The YAML shapes deserialize into two variants of each carrier:

```rust
#[non_exhaustive]
pub enum DimensionEntry {
    Inline(Dimension),
    Ref(DimensionRef),
}

#[non_exhaustive]
pub struct DimensionRef {
    pub name: SemanticsName,
    /// Local override of the root-pool expression (see §1.3).
    pub expr: Option<crate::expr_block::ExprSource>,
}
```

`MeasureEntry` / `MeasureRef` and `MetricEntry` / `MetricRef` are analogous. The `Ref` variant carries only the fields that are legally overridable at the reference site (§1.3); every other attribute is read from the root-pool declaration at `compile` time.

### 1.3 Override scope at a reference site

Only a narrow set of fields may be overridden locally at a `Ref` site. Every other attribute is **immutable from the root-pool declaration**:

| Carrier | Overridable at `ref` | Immutable from root pool |
|---|---|---|
| `Dimension` | `expr` | `data_type`, `type`, `description`, `ai_context`, `constraints` (future) |
| `Measure` | `expr`, `filters` | `data_type`, `agg`, `additivity`, `description`, `ai_context`, `constraints` (future) |
| `Metric` | `expr`, `filters` | `data_type`, `agg`, `additivity`, `description`, `ai_context`, `constraints` (future) |

Attempting to author an immutable field at a `Ref` site is `validate.semantics-ref-immutable-override` (SR-E-1).

**Deferred-body pattern.** A root-pool Semantic MAY declare `data_type:`, `agg:`, `additivity:`, etc. without an `expr:` — the expression is expected to be provided per-DataKind at every `Ref` site. A root-pool Semantic with no `expr:` and a `Ref` site with no `expr:` is `validate.semantics-ref-missing-expr` (SR-E-2). Deferred-body root Metrics are how cross-DataKind metrics are authored — the root entry fixes the name / data type / aggregation shape; each DataKind's `Ref` provides the local `expr:` that binds to that DataKind's surface.

### 1.4 Orphan policy

Every Semantic — whether authored in the root pool or inline — MUST be physically bound at least once. A physical binding is any path from the Semantic to a `Dataset`'s `extras.semantic_mapping` that resolves without error:

- **Inline on a Dataset.** The Semantic is automatically bound (the containing leaf's `semantic_mapping` or its `auto` default covers it).
- **Inline on a ComplexDataKind.** Not legal — Public complex kinds flatten their Semantics only through their composed children, but direct authoring on a composer is allowed only for composer-unique Semantics (those surfaced by `Joinset`'s `ComposedSemanticInterface`). The path to physical still terminates in a `Dataset` leaf.
- **Root-pool.** The Semantic must be `Ref`-ed from at least one DataKind whose downstream tree eventually reaches a `Dataset` with a compatible `semantic_mapping`.

A Semantic with no downstream `Dataset` binding is `validate.semantics-orphan { carrier, name }` (SR-E-3). The diagnostic fires at `validate` so tooling can report orphans before `compile` tries to build physical plans.

### 1.5 Uniqueness rules

Per-carrier uniqueness (§2.1 of `32`):
- `dimensions[..]` unique by name in the root pool.
- `measures[..]` unique by name in the root pool.
- `metrics[..]` unique by name in the root pool.
- Inline Semantics on a DataKind unique by name within that DataKind's interface.
- A `Ref` site does not redeclare the name — the name is read from the root-pool entry.

A DataKind MAY shadow a root-pool name by inlining a same-named Semantic (e.g. a locally overridden `Measure`), but this is warned as `validate.semantics-shadow-root-pool` — authors who want shadowing should use `ref` + override instead.

---

## 2. `Relationship`

### 2.1 Unified struct — root and Joinset sites

One `Relationship` struct serves both authoring sites: root-level `relationships:` on `SemanticModel`, and `relationships:` on a `JoinsetBody`.

```rust
#[non_exhaustive]
pub struct Relationship {
    pub name: String,
    pub from: DataKindName,
    pub to:   DataKindName,

    #[serde(rename = "join_type")]
    pub join_type: JoinType,

    pub keys: Vec<JoinKeyExprPair>,

    /// Optional residual predicate evaluated against the joined rowset.
    /// `None` means "equi-join only per `keys`".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<crate::expr_block::ExprSource>,

    pub cardinality: Cardinality,

    #[serde(default)]
    pub directionality: Directionality,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
```

Companion identity newtype — stable `u32` handle used by Manifest indices and compile-time graph walks:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationshipId(pub u32);
```

`RelationshipId` is allocated at `compile` in declaration order over the root-level `relationships:` list. It is the key type for the `Manifest.relationship_index`, for `RelationshipGraph` traversal in `14b`, and for `RelationshipPath` in `16 §6`. `PartialOrd` / `Ord` are derived so downstream code (`14b`'s BFS neighbor iteration, `Manifest` indices keyed by `(DataKindId, RelationshipId)`) can rely on natural `u32` ordering without unwrapping the newtype. Its one-copy-only home is this doc; `14b`, `16`, and `33` all reference it from here.

### 2.2 YAML shape

```yaml
relationships:
  - name: orders_to_customers
    from: orders
    to:   customers
    join_type: left                      # YAML tag: `join_type:`
    keys:                                # list of equi-pairs
      - from: customer_id                # SemanticExpr on the `from` side
        to:   id                         # SemanticExpr on the `to` side
    # Optional residual predicate — evaluated against the joined rowset.
    filter: "from.order_ts <= to.customer_ts"
    cardinality: many_to_one              # REQUIRED
    directionality: bidirectional         # default; author can set `forward`
    description: "Customer ownership of orders."
```

### 2.3 `JoinType`

```rust
#[non_exhaustive]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}
```

Temporal / as-of joins (`AsOf`) are explicitly out of scope for v1. Implicit historical semantics (`as_of`, `valid_at`) belong to the planner's temporal-shape handling per `17`, not to the relationship definition.

### 2.4 `Cardinality` — required at every site

```rust
#[non_exhaustive]
pub enum Cardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}
```

`cardinality:` is required on every `Relationship`, at every authoring site. Authors MUST declare the cardinality they intend; the planner does not infer it. Missing cardinality is `parse.relationship-missing-cardinality` (SR-E-4).

### 2.5 `Directionality`

```rust
#[non_exhaustive]
pub enum Directionality {
    /// The relationship is usable as `from -> to` only.
    Forward,
    /// The relationship is usable in both directions. Default.
    Bidirectional,
}

impl Default for Directionality {
    fn default() -> Self { Directionality::Bidirectional }
}
```

`Forward` mode is an optimization / modeling hint — the composer can traverse only in the declared direction. `Bidirectional` admits traversal either way; `16 §11` uses the relationship graph to synthesize join paths in both directions.

### 2.6 `JoinKeyExprPair` — hybrid equi-key grammar

```rust
#[non_exhaustive]
pub struct JoinKeyExprPair {
    /// SemanticExpr on the `from` side. In the simplest case, a bare Semantic name.
    pub from: crate::expr_block::ExprSource,
    /// SemanticExpr on the `to` side. Symmetric.
    pub to: crate::expr_block::ExprSource,
}
```

Authors list one `JoinKeyExprPair` per equi-predicate. The planner emits `left.<from_expr> = right.<to_expr>` per pair and ANDs the residual `filter:` predicate (if any) on top.

**Why a hybrid `keys` + `filter` grammar.** The v1 expected traffic is simple equi-joins; `keys:` makes that common case readable. Non-equi residuals (e.g. `from.valid_from <= to.event_ts`) need a `filter:` escape. Splitting the two keeps equi-joins short and lets the planner still know which predicates are join-structural (for hash-join eligibility, partition pruning, etc.) vs post-join residual.

### 2.7 Authoring sites and scope

| Site | Semantics |
|---|---|
| Root-level `semantic_model.relationships:` | Visible to every DataKind in the model; feeds `16 §11`'s implicit composition graph; the planner synthesizes Joinsets per Request when the relationship graph permits. |
| `JoinsetBody.relationships:` | Scoped to the Joinset's members; pinning a pre-declared traversal. A Joinset-local `Relationship` MAY redeclare a root-level name, in which case the Joinset-local entry takes precedence within the Joinset. |

Root-level relationships with no corresponding DataKind (`from:` / `to:` resolves to no known data kind) are `validate.relationship-dangling-endpoint` (SR-E-5).

---

## 3. `TemporalShape`

### 3.1 Type hierarchy — `TemporalShape`, `TemporalShapeKind`, `*Body`

`TemporalShape` is a struct carrying a tagged variant kind plus its per-variant body:

```rust
#[non_exhaustive]
pub struct TemporalShape {
    /// Variant tag + body; flattened at the YAML surface so authors write
    /// `temporal: { <variant>: {...}, grain: ... }` (see §3.2).
    #[serde(flatten)]
    pub kind: TemporalShapeKind,

    /// Effective at a `Dataset` leaf (or on a grainset child that is a Dataset).
    /// Required on leaves when `temporal:` is authored; forbidden on
    /// `ComplexDataKind`s (a complex kind only cascades the SHAPE, not the grain).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain: Option<Grain>,
}

#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum TemporalShapeKind {
    Timeseries(TimeseriesBody),
    Events(EventsBody),
    Snapshot(SnapshotBody),
    Scd(ScdBody),
}

#[non_exhaustive]
pub struct TimeseriesBody {
    /// The Dimension name that marks the series time.
    pub occurred_at: SemanticsName,
}

#[non_exhaustive]
pub struct EventsBody {
    pub event_time: SemanticsName,
}

#[non_exhaustive]
pub struct SnapshotBody {
    pub snapshotted_at: SemanticsName,
}

#[non_exhaustive]
pub struct ScdBody {
    pub scd_type: ScdType,
    pub valid_from: SemanticsName,
    pub valid_to: SemanticsName,
}

#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ScdType {
    Type1,
    Type2,
}
```

`TemporalShape.grain` carries the Dimension-level minimum grain (the finest resolution of the stored rows) — consumed by the planner to choose between candidate Datasets and to validate rollup requests.

### 3.2 YAML shape — collapsed wrapper

YAML flattens the `TemporalShapeKind` variant into a single `<variant>:` block under `extras.temporal:`:

```yaml
# Events — a Dataset storing sparse discrete occurrences.
extras:
  temporal:
    events:
      event_time: order_ts
    grain: minute

# Timeseries — a Dataset storing regularly-sampled rows.
extras:
  temporal:
    timeseries:
      occurred_at: sample_ts
    grain: hour

# Snapshot — periodic full-state capture.
extras:
  temporal:
    snapshot:
      snapshotted_at: snapshot_date
    grain: day

# SCD Type-2 — history with valid-range pairs.
extras:
  temporal:
    scd:
      scd_type: type2
      valid_from: valid_from_ts
      valid_to:   valid_to_ts
    grain: day
```

Only one of `timeseries: / events: / snapshot: / scd:` may appear under `temporal:` (`parse.temporal-multiple-variants`). The `grain:` field sits at the same level as the variant block — it belongs to `TemporalShape`, not to the variant body.

### 3.3 Effective level: leaf required, complex forbidden

`temporal:` may appear in `extras` at any complex-DataKind level as a **shape default**, but the `grain:` field is forbidden at complex levels — only the shape cascades down, not the grain.

| Site | `temporal.<variant>:` | `temporal.grain:` |
|---|---|---|
| `Dataset` (leaf) | Authored directly OR inherited | Authored directly — **required** when `temporal:` is present (SR-E-6) |
| `ComplexDataKind` (any) | Authored as a default for descendant leaves | **Forbidden** (SR-E-7) |
| `Grainset` child | See §3.4 |

A `ComplexDataKind.extras.temporal.<variant>:` declaration cascades down to every descendant leaf that does not override. A descendant leaf that inherits a shape but does not declare its own `grain:` is `validate.temporal-leaf-missing-grain` (SR-E-6).

### 3.4 Grainset children — explicit grain required

A `Grainset`'s children form a rollup fan: each child declares a different grain on the same semantic axis. Grain inheritance would hide the axis declaration. Rule:

**Every `Grainset` child MUST author its own `extras.temporal.grain:`.** Shape can cascade from the Grainset parent; grain cannot. (SR-E-8: `validate.grainset-child-grain-required`.)

YAML:

```yaml
grainsets:
  - name: orders_by_grain
    extras:
      temporal:
        events:
          event_time: order_ts
        # No `grain:` here — forbidden on a complex kind (SR-E-7).
    datasets:
      - name: orders_minute
        extras:
          temporal:
            grain: minute         # required explicitly on the child
      - name: orders_hour
        extras:
          temporal:
            grain: hour
```

### 3.5 `Grain` — pointer only

The `Grain` enum is owned by [`13 §3.1`](./13_types_and_grain.md#31-enum); the v1 roster is `{Minute, Hour, Day, Week, Month, Quarter, Year}` with total coarseness order in declaration order (finest first). `18 §3` consumes `Grain` through the optional `TemporalShape.grain` field and the `TimeseriesBody.grain` / `SnapshotBody.cadence` payloads; it does not redefine the enum. Non-temporal grains (geographic, entity) are a deferred extensibility axis per `13`.

> **Note (2026-04-17 consolidation).** An earlier draft of this section inlined a `pub enum Grain` block that accidentally introduced a `Second` variant not present in `13 §3.1`. Per the precedence rule in `00 §4.4` + the directionality rule in `00 §8`, `13` is the canonical home; the divergent roster has been removed. Any future addition to `Grain` (e.g., `Second`) must land in `13 §3.1` first.

### 3.6 Forward reference to `17`

Planner-level semantics — the shape × grain rollup matrix, snapshot pin policies, SCD as-of anchoring — live in `17`. `18 §3` ratifies the shape of the type and its YAML carriage; `17` ratifies what each shape *means* for request execution.

---

## 4. `Dimension`

```rust
#[non_exhaustive]
pub struct Dimension {
    pub name: SemanticsName,

    /// Mandatory at declaration; immutable from the root-pool declaration.
    pub data_type: DataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    /// The Dimension category — drives planner and adapter behavior.
    #[serde(rename = "type")]
    pub dim_type: DimensionType,

    /// Optional derivation expression. `None` means the Dimension is
    /// bound directly from `semantic_mapping`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<crate::expr_block::ExprSource>,
}
```

### 4.1 `DimensionType` roster

`DimensionType` IS the **Dimension category axis** (per `19 §2`). Its category-axis semantics — what each variant implies for planner routing, adapter behavior, and validation — are ratified canonically in [`19 §2.2`](./19_categories.md#22-implicit-constraint-contract-per-dimension-category). This subsection (`18 §4.1`) remains the canonical home of the **enum roster + body struct shapes**; the implicit-constraint contract per variant lives in `19`.

```rust
#[non_exhaustive]
pub enum DimensionType {
    Temporal(TemporalDimensionBody),
    Categorical,
    Binary,
    Geo,
    Bucketed(BucketedDimensionBody),
    Metadata(MetadataDimensionBody),
}
```

All six variants stay in the v1 roster (no new variants in v1; a proposed `Identifier` variant is deferred as `Q-CAT-001` in [`questions/open/19_questions.md`](../questions/open/19_questions.md)). Per-variant body shapes are:

```rust
#[non_exhaustive]
pub struct TemporalDimensionBody {
    /// The set of grains at which this Dimension can be rolled up.
    /// Authors list only the grains the backing source actually supports.
    pub grains: Vec<Grain>,
}

#[non_exhaustive]
pub struct BucketedDimensionBody {
    /// Bucket boundaries. Each bucket spans [lower_inclusive, upper_exclusive).
    pub buckets: Vec<BucketSpec>,
}

#[non_exhaustive]
pub struct BucketSpec {
    pub name: String,
    pub lower: Option<BucketBound>,
    pub upper: Option<BucketBound>,
}

#[non_exhaustive]
pub enum BucketBound {
    Int(i64),
    Float(f64),
    Decimal(String),           // preserved as string for lossless round-trip
    Date(String),              // ISO-8601 date
    Timestamp(String),         // ISO-8601 timestamp
}

#[non_exhaustive]
pub struct MetadataDimensionBody {
    /// Where to extract the value from each PhysicalSource — path token,
    /// partition column, or S3 object metadata field. Details in `15 §8`.
    pub source: MetadataSource,
}
```

`TemporalDimensionBody.grains` is the Dimension-level rollup axis; it can be empty (the Dimension is declared as a Timestamp but not rollable), in which case only the source grain on `extras.temporal.grain:` applies. Sub-shape polish (`BucketBound` roster extension, `MetadataSource` full grammar) is tracked as a v1-minor item.

### 4.2 YAML

```yaml
dimensions:
  - name: order_ts
    data_type: timestamp
    type:
      temporal:
        grains: [minute, hour, day, week, month]

  - name: country
    data_type: string
    type: categorical

  - name: is_active
    data_type: boolean
    type: binary

  - name: order_region
    data_type: geo
    type: geo

  - name: order_value_bucket
    data_type: string
    type:
      bucketed:
        buckets:
          - { name: low,    upper: { int: 100 } }
          - { name: mid,    lower: { int: 100 }, upper: { int: 1000 } }
          - { name: high,   lower: { int: 1000 } }

  - name: source_partition
    data_type: string
    type:
      metadata:
        source:
          partition: year
```

---

## 5. `Measure`

```rust
#[non_exhaustive]
pub struct Measure {
    pub name: SemanticsName,

    pub data_type: DataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    /// The Measure category — drives implicit `agg` / `additivity` derivations
    /// and planner / adapter behavior. Canonical semantics in `19 §3`.
    /// REQUIRED post-`19`; pre-`19` manifests parse with category inferred
    /// from `agg:` + `additivity:` per the `[TD-CATEGORIES-MIGRATE]` window.
    #[serde(default = "MeasureCategory::default_legacy_inferred")]
    pub category: MeasureCategory,

    /// Selects the aggregation family. **Derived from `category:`** per
    /// `19 §3.3`; authors MAY still spell it explicitly (must agree, else
    /// SR-E-13). Required on legacy / `category: custom` Measures; otherwise
    /// optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agg: Option<AggregationType>,

    /// Optional horizontal-only transform applied before aggregation.
    /// E.g. `expr: amount_cents * 0.01` + `agg: sum` gives `sum(amount_cents * 0.01)`.
    /// `None` means the aggregation is applied directly to the Semantics named
    /// by the Measure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<crate::expr_block::ExprSource>,

    /// Additivity classification per `AdditivityType`. **Derived from
    /// `category:`** per `19 §3.3`; authors MAY still spell it explicitly
    /// (must agree, else SR-E-13). `category: snapshot` synthesizes
    /// `AdditivityType::Semi(SemiAdditivity { axes, strategy })` from its
    /// body — explicit `additivity: semi` authoring is discouraged in
    /// favor of `category: snapshot`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additivity: Option<AdditivityType>,

    /// Measure-level conditional-aggregation filters (see §7).
    /// Each filter wraps the Measure in a `CASE WHEN ... THEN expr ELSE NULL END`
    /// at compile time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<AggregationFilter>,

    /// Explicit refinement on top of category-implied implicit constraints.
    /// Carrier-roster + sub-block grammar live in `11 §8`. `None` ⇒ no
    /// explicit refinement; the category alone governs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<Constraints>,
}
```

`MeasureCategory` and its body structs are ratified canonically in [`19 §3`](./19_categories.md#3-measure-categories--measurecategory). `Constraints` (the shared post-rewrite type — see `[TD-CONSTRAINT-RENAME]`) is ratified in [`11 §8`](./11_names_and_scopes.md#8-constraint).

### 5.1 `AggregationType` roster

```rust
#[non_exhaustive]
pub enum AggregationType {
    Sum,
    Avg,
    Count,
    CountDistinct,
    Min,
    Max,
    Median,
    StdDev,
    Variance,
}
```

`Count` and `CountDistinct` are separate variants. Internally, function-level modifiers (distinct over specific columns, percentile variants) may be richer, but the model-level `agg:` surface carries only this roster.

### 5.2 `AdditivityType` roster

```rust
#[non_exhaustive]
pub enum AdditivityType {
    Full,
    Semi(SemiAdditivity),
    Non,
}

#[non_exhaustive]
pub struct SemiAdditivity {
    /// The Dimension axes along which this Measure is semi-additive.
    pub axes: Vec<SemanticsName>,
    /// Rollup strategy for the non-additive axes.
    pub strategy: SemiAdditivityStrategy,
}

#[non_exhaustive]
pub enum SemiAdditivityStrategy {
    Latest,
    Earliest,
    Average,
    First,
    Last,
}
```

### 5.3 YAML

```yaml
measures:
  # Additive Measure (preferred post-`19` authoring path) — agg + additivity derived.
  - name: gross_revenue
    data_type: decimal(18, 2)
    category: additive
    expr: amount_cents * 0.01

  # Snapshot Measure — replaces explicit `additivity: semi` authoring.
  # Inventory is additive across warehouses, snapshot-latest over time.
  - name: inventory_on_hand
    data_type: long
    category:
      snapshot:
        non_additive_axes: [snapshotted_at]
        strategy: latest
    agg: sum                            # optional; must agree with category

  # Average — non-additive; planner re-aggregates from SUM/COUNT at queried grain.
  - name: margin_rate
    data_type: double
    category: average

  # Custom escape hatch — author states agg + additivity manually.
  - name: weighted_revenue_proxy
    data_type: decimal(18, 2)
    category: custom
    agg: sum
    additivity: full
    expr: amount_cents * weight * 0.01
```

### 5.4 Requirements

- `category:` is REQUIRED post-`19` (legacy authoring without `category:` parses with category inferred from `agg:` + `additivity:` during the `[TD-CATEGORIES-MIGRATE]` window; see `19 §3`).
- `agg:` is **derived from `category:`** per `19 §3.3`. Authors MAY still spell it explicitly (must agree, else SR-E-13). Required on `category: custom` Measures.
- `expr:` is OPTIONAL; when absent, the aggregation applies to the Semantic whose name the Measure declares (via `semantic_mapping` resolution at binding time).
- `additivity:` is **derived from `category:`** per `19 §3.3`. Authors MAY still spell it explicitly (must agree, else SR-E-13). Required on `category: custom`.
- `data_type:` is mandatory at declaration (SR-E-10).
- `constraints:` is OPTIONAL — explicit refinement on top of category-derived implicit constraints (per `11 §8` rewrite). MAY only narrow what the category permits (SR-E-15).

---

## 6. `Metric`

```rust
#[non_exhaustive]
pub struct Metric {
    pub name: SemanticsName,

    pub data_type: DataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    /// The Metric category — drives expr-shape locks, additivity defaults,
    /// and planner derivation routing. Canonical semantics in `19 §4`.
    /// REQUIRED post-`19`; pre-`19` manifests parse with category inferred
    /// from `expr:` shape per the `[TD-CATEGORIES-MIGRATE]` window.
    #[serde(default = "MetricCategory::default_legacy_inferred")]
    pub category: MetricCategory,

    /// Optional post-aggregation wrapper. Default derives from
    /// `category:` (e.g. Simple inherits the wrapped Measure's agg).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agg: Option<AggregationType>,

    /// The derivation expression over Measures / Dimensions. Authoring
    /// rules depend on `category:` (`19 §4.2`):
    /// - `Simple` ⇒ `expr:` resolves to a single Measure name (or omitted;
    ///   `body.measure:` is the source of truth).
    /// - `Ratio` ⇒ `expr:` shape is `numerator / denominator`; both sides
    ///   MUST be Measure or Simple-Metric names.
    /// - `Derived` ⇒ `expr:` references ≥ 2 Measures / Metrics.
    /// May be deferred-body on a root-pool Metric and supplied at every
    /// Ref site (see §1.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<crate::expr_block::ExprSource>,

    /// Additivity classification. **Derived from `category:`** per
    /// `19 §4.2` (Ratio ⇒ Non implicit; Simple ⇒ inherited from wrapped
    /// Measure; Derived ⇒ author-stated, default Non).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additivity: Option<AdditivityType>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub filters: Vec<AggregationFilter>,

    /// Explicit refinement on top of category-implied implicit constraints.
    /// Carrier-roster + sub-block grammar live in `11 §8`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<Constraints>,
}
```

`MetricCategory` and its body structs are ratified canonically in [`19 §4`](./19_categories.md#4-metric-categories--metriccategory). The v1 roster is `{Simple, Ratio, Derived}`; `Cumulative` and `Conversion` are commented-out reserved variants per `00 §10`.

### 6.1 Requirements

- `category:` is REQUIRED post-`19` (legacy authoring parses with category inferred from `expr:` shape during the `[TD-CATEGORIES-MIGRATE]` window).
- `expr:` is REQUIRED whenever the category requires it (`Derived` always; `Simple` if `body.measure:` is omitted; `Ratio` may rely solely on `body.numerator:` + `body.denominator:`). May be deferred-body on a root-pool Metric and supplied at every `Ref` site (§1.3).
- `agg:` is OPTIONAL — derived from `category:` (e.g. `Simple` inherits the wrapped Measure's agg). When the author spells it explicitly it must agree (SR-E-13).
- `additivity:` is **derived from `category:`** per `19 §4.2`; explicit authoring must agree.
- `data_type:` is mandatory at declaration (SR-E-10).
- `constraints:` is OPTIONAL — explicit refinement on top of category-derived implicit constraints (per `11 §8` rewrite). MAY only narrow what the category permits (SR-E-15). Metric `expr:` shape MUST match the declared category (SR-E-17).

### 6.2 Cross-DataKind references

A root-pool Metric MAY reference Measures / Dimensions from different DataKinds. Resolution rules:

1. **Same-level references.** All names in the Metric's `expr:` are defined at the root pool (as `Measure` / `Dimension` entries). The planner synthesizes the join path via the relationship graph at Request time.
2. **Deferred-body.** The root Metric declares `data_type:`, `agg:` (optional), `additivity:`, etc. but omits `expr:`. Each DataKind that wants to expose the Metric writes `- ref: metric_name` with its local `expr:` (bound to that DataKind's interface).

A root Metric whose `expr:` names Semantics that span multiple DataKinds without a resolvable relationship path is `compile.metric-path-unresolvable { metric, endpoints }` — same diagnostic as a bare Request whose field set spans without a path.

### 6.3 YAML

```yaml
metrics:
  # Inline root-pool Metric — same-level refs against root-pool Measures.
  - name: revenue_per_order
    data_type: double
    expr: total_revenue / order_count
    additivity: non

  # Deferred-body root-pool Metric — expression is supplied at Ref sites.
  - name: conversion_rate
    data_type: double
    agg: avg
    # No expr here — required at every Ref site.

datasets:
  - name: orders
    metrics:
      - ref: conversion_rate
        expr: sum(case_when(is_converted, 1, 0)) / count(*)

  - name: leads
    metrics:
      - ref: conversion_rate
        expr: sum(case_when(lead_converted, 1, 0)) / count(*)
```

---

## 7. Filter Taxonomy

There are **two distinct filter types** in the model, sharing no common supertype. They serve different functional roles and may not cross-reference each other.

### 7.1 `DataKindFilter` — DataKind-level, user-facing predicate

A DataKind-level filter narrows the rowset a DataKind exposes. Authored under a DataKind's `SemanticInterface.filters:` and resolved against the DataKind's Dimension / Measure surface.

```rust
#[non_exhaustive]
pub struct DataKindFilter {
    pub name: FilterName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,
    pub expr: crate::expr_block::ExprSource,
}
```

YAML:

```yaml
datasets:
  - name: orders
    filters:
      - name: high_value_only
        expr: amount_cents >= 10000
        description: "Only orders ≥ $100."
```

A Request may name a DataKindFilter to activate it at query time:

```yaml
request:
  from: orders
  filters: [high_value_only]
```

Functional role: **gate the DataKind's rowset before any aggregation**. Equivalent to a WHERE clause at the semantic level.

### 7.2 `AggregationFilter` — Measure / Metric-level, conditional aggregation

A Measure / Metric-level filter specifies a conditional inside the aggregation. Authored under a Measure's or Metric's `filters:` list.

```rust
#[non_exhaustive]
pub struct AggregationFilter {
    pub name: FilterName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub expr: crate::expr_block::ExprSource,
}
```

YAML:

```yaml
measures:
  - name: revenue_usd_only
    data_type: decimal(18, 2)
    agg: sum
    expr: amount_cents * 0.01
    filters:
      - name: usd_only
        expr: currency == 'USD'
```

Compile lowering: the aggregation is wrapped in `CASE WHEN <filter.expr> THEN <measure.expr> ELSE NULL END`; the aggregation is then applied to that wrapped expression. Multiple filters on a single Measure AND together.

Functional role: **conditionally contribute rows to the aggregation**. Equivalent to `SUM(CASE WHEN ... THEN amount ELSE NULL END)` at the adapter level.

### 7.3 No cross-referencing

The two filter types are disjoint: a `DataKindFilter` cannot be named from a Measure / Metric's `filters:` list (and vice versa). The namespace boundary is the authoring site:

- A Request's `filters: [name]` resolves only against `DataKindFilter` names.
- A Measure's / Metric's `filters:` entries are local declarations (inline) — they have no global name table and are not addressable by Request filters.

Attempting a cross-reference is `validate.filter-wrong-kind { name, expected, actual }` (SR-E-11).

---

## 8. `AiContext`

`AiContext` is the LLM / agent-facing hint surface. It attaches to root-level SemanticModel, top-level data kinds, and individual Semantics — never to structural scaffolding (Nested forms, Extras blocks, Relationships themselves).

```rust
#[non_exhaustive]
pub struct AiContext {
    /// Synonyms the LLM may use to refer to the annotated entity.
    /// Each key is a logical alias; its value is the canonical form the
    /// model exposes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub synonyms: BTreeMap<String, Vec<String>>,

    /// A plain-language description the LLM may surface to the user.
    /// Duplicates the carrier's `description:` with more narrative freedom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Example queries or phrasings the LLM may emit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,

    /// Unit of measurement for numeric Semantics (e.g. "usd", "percent",
    /// "events_per_minute").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}
```

YAML:

```yaml
ai_context:
  synonyms:
    order: [purchase, txn]
    revenue: [sales, income]
  description: "Orders placed by end customers on the retail site."
  examples:
    - "total revenue for Q1 2025"
    - "orders by region last week"
  unit: usd
```

The four fields are the closed v1 roster. The enum / struct is `#[non_exhaustive]`, so adding a field (e.g. `tags`, `capability_hints`) is MINOR.

---

## 9. `Keys`

```rust
#[non_exhaustive]
pub struct Keys {
    /// Primary key — the unique row identifier. At most one declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<KeyDecl>,

    /// Additional unique keys beyond the primary.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unique: Vec<KeyDecl>,

    /// Foreign keys — references to other DataKinds' primary / unique keys.
    /// Relationship graph uses these to infer join paths.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub foreign: Vec<ForeignKeyDecl>,
}

#[non_exhaustive]
pub struct KeyDecl {
    /// Bare Semantic names — no physical column references.
    /// Resolution through `semantic_mapping` at binding time.
    pub columns: Vec<SemanticsName>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[non_exhaustive]
pub struct ForeignKeyDecl {
    pub columns: Vec<SemanticsName>,
    /// The target DataKind whose primary / unique key is referenced.
    pub references: DataKindName,
    /// The target DataKind's key columns — bare Semantic names.
    pub target_columns: Vec<SemanticsName>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
```

YAML:

```yaml
datasets:
  - name: orders
    keys:
      primary: { columns: [order_id] }
      unique:
        - { columns: [external_id], name: ext_id_uk }
      foreign:
        - columns: [customer_id]
          references: customers
          target_columns: [id]
```

### 9.1 Why bare Semantic names

Keys refer to Semantics, not physical columns. Binding through `semantic_mapping` resolves each Semantic to its physical column (or expression) at `compile`. Authors never touch physical shape at this layer.

### 9.2 Keys are metadata, not constraints

Keys are consumed for:

- **Relationship graph** — `16 §11`'s implicit composition consults foreign keys when author-declared `relationships:` are absent.
- **Manifest statistics** — `compile` emits pre-computed statistics for the planner.
- **Future SemanticInterface exposure** — external consumers (LSP, API) may surface keys as part of the entity description.

Keys are NOT enforced at query time (no duplicate checking, no referential-integrity validation). That's database territory, not a semantic-model concern.

---

## 10. `SemanticMapping` Value Shape

A `semantic_mapping:` entry carries one of three variants, resolved at `compile` during the Binding process (`15`):

```rust
#[non_exhaustive]
pub enum SemanticMappingValue {
    /// Bare physical column name — the Semantic is 1:1 to `Column(name)`.
    Column(String),

    /// A literal broadcast over every row — useful for per-source constants.
    Literal(LiteralValue),

    /// A `PhysicalExpr` tree — anything from a simple cast to a multi-column
    /// compute. Lives at binding time because the expression references
    /// physical column names, not Semantic names.
    Expr(crate::expr_block::PhysicalExpr),
}
```

### 10.1 YAML

```yaml
extras:
  semantic_mapping:
    # Variant 1 — bare column.
    revenue: net_revenue_cents

    # Variant 2 — literal broadcast.
    currency: { literal: "USD" }

    # Variant 3 — physical expression.
    hour_bucket:
      expr:
        trunc:
          column: event_ts
          unit: hour
```

Single-string values dispatch to `Column`; mapping values with `literal:` / `expr:` keys dispatch to `Literal` / `Expr`.

### 10.2 `LiteralValue`

```rust
#[non_exhaustive]
pub enum LiteralValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Decimal(String),
    String(String),
    Date(String),
    Timestamp(String),
}
```

Each literal carries its `DataType`-equivalent kind tag; the compile-time binding validates that the literal's kind is assignable to the Semantic's declared `data_type:`.

### 10.3 `auto` vs explicit

An absent `semantic_mapping:` block on a `Dataset`'s `extras` is equivalent to `semantic_mapping: auto`. Explicit entries narrow that default: every Semantic named in the explicit map receives the declared value; every other Semantic on the `Dataset`'s interface is resolved per `auto` (name-identical physical column).

---

## 11. Structural Rules (SR-E-*)

Entity-level invariants. Each rule has a stable kebab-case diagnostic code per `30 §6`. Numbered independently from `32`'s root-level `SR-*` rules so additions do not perturb the other series.

| ID | Rule | Diagnostic |
|---|---|---|
| **SR-E-1** | Reference-site override MAY NOT author immutable fields (`data_type`, `type`, `agg`, `additivity`, `description`, `ai_context`). See §1.3. | `validate.semantics-ref-immutable-override` |
| **SR-E-2** | A `Ref` site missing `expr:` AND root-pool entry missing `expr:` is ill-formed. | `validate.semantics-ref-missing-expr` |
| **SR-E-3** | Every Semantic (root-pool or inline) MUST physically bind at least once. | `validate.semantics-orphan` |
| **SR-E-4** | `Relationship.cardinality:` is required at every authoring site. | `parse.relationship-missing-cardinality` |
| **SR-E-5** | `Relationship.from:` / `.to:` MUST resolve to a declared DataKind. | `validate.relationship-dangling-endpoint` |
| **SR-E-6** | A leaf `Dataset` with `extras.temporal:` authored MUST declare `temporal.grain:`. | `validate.temporal-leaf-missing-grain` |
| **SR-E-7** | A `ComplexDataKind` MUST NOT author `extras.temporal.grain:` (only shape cascades). | `validate.temporal-grain-on-complex` |
| **SR-E-8** | Every `Grainset` child MUST author its own `extras.temporal.grain:` explicitly (no inheritance). | `validate.grainset-child-grain-required` |
| **SR-E-9** | A `Measure` MUST declare `agg:` at its declaration site. | `parse.measure-missing-agg` |
| **SR-E-10** | A `Dimension` / `Measure` / `Metric` MUST declare `data_type:` at its declaration site. | `parse.semantics-missing-data-type` |
| **SR-E-11** | Filter names are not cross-referenceable between `DataKindFilter` and `AggregationFilter`. | `validate.filter-wrong-kind` |
| **SR-E-12** | `data_type:` is immutable across all levels — root-pool and Ref sites must agree; local overrides are forbidden. | `validate.semantics-data-type-mismatch` |
| **SR-E-13** | `Measure.category:` and `Metric.category:` derived `agg:` / `additivity:` MUST agree with author-stated `agg:` / `additivity:` (when present). Canonical home: `19 §6`. | `validate.measure-category-mismatch` / `validate.metric-category-mismatch` |
| **SR-E-14** | A category whose body is non-empty MUST author every required body field. Canonical home: `19 §6`. | `validate.category-body-incomplete` |
| **SR-E-15** | A `constraints:` block MAY NOT widen what the category locks (e.g. `aggregation.allowed:` listing an aggregation outside the category-implied set). Canonical home: `19 §6` (Measure / Metric); subsumes Dimension `data_type:` mismatch. | `validate.constraints-incompatible-with-category` (Measure / Metric) / `validate.dimension-category-data-type-mismatch` (Dimension) |
| **SR-E-16** | Downstream re-aggregation of a Measure / Metric MUST satisfy the carrier's `aggregation:` constraint (implicit ∩ explicit). Canonical home: `19 §6`. | `plan.downstream-aggregation-violation` |
| **SR-E-17** | A `Metric.category:` shape MUST match the Metric's `expr:` / body shape. Canonical home: `19 §6`. | `validate.metric-category-expr-shape-mismatch` |
| **SR-E-18** | A `Filter` entity MUST NOT carry a top-level `constraints:` field (Q-R4.3d in `11 §8`). Canonical home: `19 §6`. | `validate.constraints-on-filter-entity` |
| **SR-E-19** | An unrecognized `category:` value (not in the spec-owned roster) is rejected; v1 does not lenient-downgrade unknown categories — see `[TD-CAT-LENIENT]`. Canonical home: `19 §6`. | `validate.unknown-category` |

SR-E-* numbering is append-only; adding a rule is MINOR per `30 §2`. SR-E-13 … SR-E-19 are ratified canonically in [`19 §6`](./19_categories.md#6-structural-rules-sr-e-13--sr-e-19); this section lists them for the entity-level diagnostic catalog.

---

## 12. Cross-References

- `32 §1` — root YAML shape and where the shared pools / relationships / data-kinds / semantic-mapping blocks live.
- `32 §3` — `DataKind` hierarchy; `DatasetBody` / `GrainsetBody` / `UnionsetBody` / `JoinsetBody`. `JoinsetBody.relationships: Vec<Relationship>` uses the unified struct from §2.
- `32 §4` — `Extras` block; `semantic_mapping:` lives inside, value shape per §10.
- `32 §6` — root `SR-*` rules; complementary to `SR-E-*` here.
- `32b` — catalog grammar; `CatalogRef` referenced from `extras.catalog:`.
- `26` — nesting matrix; complements the ≥ 2 children structural rule.
- `11` — name / scope rules for Semantics.
- `13` — `DataType`, `Grain`, `DimensionType` vocabulary this doc embeds.
- `14` — `SemanticExpr` / `PhysicalExpr` grammar for every `expr:` field here.
- `19` — Dimension / Measure / Metric **category** axis; `MeasureCategory` + `MetricCategory` enums and per-variant body structs; SR-E-13 … SR-E-19 canonical home.
- `14b` — compile-time resolution of `SemanticExpr` at `Ref` sites + override merging.
- `15` — `Binding` process that consumes `semantic_mapping` + the entities ratified here.
- `16` — `Relationship` graph consumption; `Joinset` path synthesis.
- `17` — planner-level `TemporalShape` semantics and rollup matrices.
- `21` / `22` / `23` / `24` — per-variant YAML carriage of the entity types.
- `30 §6` — diagnostic code conventions (kebab-case primary).

---

*Cross-references use `NN §M.K` for internal sections and full relative paths for other docs.*
