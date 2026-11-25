---
id: 34-planner-request
status: ratified
version: 2.1
impl: in-progress
---

# Request and SessionContext

This leaf owns Request, SessionContext, ResolvedQueryRequest, Filter/FilterOperator/FilterValue, SortDir, DataKindRef, TemporalRequest (reserved), RequestDimensionRef (rollup-aware carrier, reserved).

## 3. The `Request` type

### 3.1 Shape

```rust
/// A caller-authored query intent — the planner's input at the
/// stage-4 boundary (`10 §3.4`). Per `00 §4.1`'s Request row. Carries
/// no references to the SemanticManifest; the API layer constructs it from
/// the user-facing query form.
#[non_exhaustive]
pub struct Request {
    pub dimensions: Vec<SemanticsName>,
    pub measures:   Vec<SemanticsName>,
    pub metrics:    Vec<SemanticsName>,
    pub filters:    Vec<Filter>,
    pub order:      Vec<(Name, SortDir)>,
    pub limit:      Option<u64>,
    pub offset:     Option<u64>,
    pub from:       Option<DataKindRef>,
    pub temporal:   Option<TemporalRequest>,
    pub session:    SessionContext,
}
```

Each field is detailed in §3.2–§3.7. `#[non_exhaustive]` per I10: additions (e.g. a reserved `explain:` block for plan-tree introspection) are MINOR per `30 §2.2`.

### 3.2 `dimensions`, `measures`, `metrics`

Each field is a `Vec<SemanticsName>`. Order is user-visible: it pins the left-to-right column order in `SemanticPlan.output_names` (`35 §3.2`). Duplicates within any single list are rejected at step 1 with `PLAN_E_0510 DuplicateRequestedName`; empty lists on all three simultaneously are rejected with `PLAN_E_0511 EmptyRequest`.

The split into three vectors mirrors `00 §4.1`'s terminology. At the planner layer the distinction is structural, not semantic: every entry becomes a leaf lookup against the SemanticManifest's name index. The split matters downstream — a Measure wraps inside a `PlanNode::Agg`; a Dimension lands in `group_by`; a Metric routes through its `CompiledMetric` entry per `19 §3.2.4`.

**Type admissibility.** If a name's resolved Semantics type contradicts the field it was placed in (e.g. a Dimension name in `measures`), the planner raises `PLAN_E_0512 RequestFieldTypeMismatch` at step 1.

**Rollup-aware Dimension carrier (forward-extensible).** The bare `Vec<SemanticsName>` shape suffices when every Dimension is requested in its native form. When a Dimension needs a per-axis transform (e.g. temporal rollup to `month` via `DATE_TRUNC(month, axis)` at `GROUP BY`), the carrier `RequestDimensionRef { name, variation }` per §3.10 is the forward-extensible replacement. Tracked under `[TD-REQUEST-DIM-VARIATION]`; consumed by `[19 §6.2](../../foundations/19_expression_flow.md)`.

### 3.3 `filters`

`Vec<Filter>` — a flat list of row-level predicates conjoined by logical AND. Each `Filter` names a Semantics by `SemanticsName`, an operator, and a value vector (§3.5). Filter decomposition and placement across scan / rename / expression / aggregation layers is the planner's job (§7.6, `21 §4.6`); the caller sees only the conjunctive list.

Filters on a Measure or two-stage Metric with `agg:` land above `PlanNode::Agg` (HAVING-equivalent). Filters on a Dimension sink below aggregation. Filters on a declared Filter Semantics (`11 §6`) resolve through the Semantics' body. `filters: Vec::new()` is legal; default filters carried on the `ResolvedDataKind` (e.g. Filter-injection under reserved carrier `11 §8.5.2`'s `requires:`) are applied separately.

### 3.4 `order`, `limit`, `offset`

- `order: Vec<(Name, SortDir)>` — ORDER BY list. Names MUST appear in `dimensions` ∪ `measures` ∪ `metrics`; violation raises `PLAN_E_0513 OrderByUnknownName`.
- `limit: Option<u64>` — row cap on the post-sort output. `Some(0)` emits an empty-result `PlanNode::Fetch`.
- `offset: Option<u64>` — row offset applied before `limit`.

Placement: `order` → `PlanNode::Sort`; `offset` / `limit` → `PlanNode::Fetch`. Empty `order` with non-empty `limit` is legal; result order is non-deterministic unless `order` is specified.

### 3.5 `Filter`

```rust
/// A single user-supplied predicate. Conjoined with siblings via AND.
#[non_exhaustive]
pub struct Filter {
    /// The Semantics name being filtered. MUST resolve on the target
    /// `ResolvedDataKind`'s interface (Simple `SemanticInterface` or
    /// Complex `ComposedSemanticInterface`).
    pub field: SemanticsName,

    /// The operator applied between `field` and `values`.
    pub operator: FilterOperator,

    /// The operand(s) the operator consumes. Each operator has a
    /// per-variant arity rule enforced at step 1 (§7.2).
    pub values: Vec<FilterValue>,
}

#[non_exhaustive]
pub enum FilterOperator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    In,
    NotIn,
    Between,
    IsNull,
    IsNotNull,
}

/// Typed filter operand. The planner lowers every variant to an
/// `Expr::Literal` leaf with the appropriate `Literal` variant
/// during step 5 (§7.5).
#[non_exhaustive]
pub enum FilterValue {
    String(String),
    Number(f64),
    Integer(i64),
    Decimal(String),   // string form; parsed against the target's DataType
    Bool(bool),
    Timestamp(Timestamp),
    Date(Date),
    Null,
}
```

**Arity rules** (enforced at step 1):


| Operator                                       | Arity            | Error                             |
| ---------------------------------------------- | ---------------- | --------------------------------- |
| `Eq` / `NotEq` / `Lt` / `LtEq` / `Gt` / `GtEq` | 1                | `PLAN_E_0520 FilterArityMismatch` |
| `In` / `NotIn`                                 | ≥ 1              | same                              |
| `Between`                                      | 2 (lower, upper) | same                              |
| `IsNull` / `IsNotNull`                         | 0                | same                              |


**Type admissibility.** A `FilterValue` whose native Rust type cannot be cast to `field`'s resolved `DataType` raises `PLAN_E_0521 FilterValueTypeMismatch`. Conversions follow `13 §4`'s compatibility relation; no engine-specific coercion is performed (I2).

### 3.6 `SortDir`

```rust
#[non_exhaustive]
pub enum SortDir {
    Asc,
    Desc,
    AscNullsFirst,
    AscNullsLast,
    DescNullsFirst,
    DescNullsLast,
}
```

The null-aware variants are explicit vocabulary per `35 §5.7`. `Asc` and `Desc` are null-placement-agnostic; the planner forwards the agnostic variant to the `PlanNode::Sort` node and the adapter's null-placement default applies.

### 3.7 Delta with legacy code

The legacy `crates/semstrait-planner/src/request.rs::ResolvedQueryRequest` folds several concerns into one struct:

- `entity_name: String` — the legacy name for `Request.from` (scalar, not optional). `34` ratifies `from: Option<DataKindRef>`, with `None` triggering field-first resolution.
- `dimensions: Vec<String>` — split in this spec into `dimensions` / `measures` / `metrics` to match `00 §4.1`.
- `grain: Option<Grain>` — DEFERRED to `17 §6`'s `Request.temporal` block; current `ResolvedQueryRequest.grain` is a legacy convenience that the API layer derived from `dimensions`.
- `session_variables: HashMap<String, String>` — refined into `SessionContext` (§4).

The rename is tracked in `docs/TECH_DEBT.md` as `[TD-REQUEST-SHAPE]`. The legacy name `ResolvedQueryRequest` survives in the crate-internal `resolution::` pipeline as a stage-internal alias for the post-lookup form (§5) until the refactor lands.

### 3.8 `DataKindRef`

```rust
/// Reference to a top-level `ResolvedDataKind` by its canonical name.
/// Newtype over `DataKindName` (from `semstrait-common` / `11 §4`).
pub struct DataKindRef(pub DataKindName);

impl DataKindRef {
    pub fn new(name: impl Into<DataKindName>) -> Self;
    pub fn as_name(&self) -> &DataKindName;
}
```

`DataKindRef` is intentionally non-`#[non_exhaustive]` — it is a newtype over a stable inner type per `30`'s "newtype-over-stable" exception (`31 §3.2`).

### 3.9 `TemporalRequest` — reserved shape

```rust
/// Reserved per `17 §6.1`. Planner consumption is DEFERRED.
#[non_exhaustive]
pub struct TemporalRequest {
    pub as_of: Option<Timestamp>,
    pub range: Option<TimeRange>,
}

#[non_exhaustive]
pub struct TimeRange {
    pub start: Timestamp,
    pub end: Timestamp,
}
```

Exposed now, consumed later. Ratifying the shape in `34` lets the API layer (`semstrait-api`) populate the field in a forward-compatible way; per `17 §10`, the Round-1 planner emits `PLAN_W_1702` when a populated `TemporalRequest` targets a `TemporalShape::None` DataKind, and `PLAN_W_1700 ShapeAwareRequestDeferred` on any non-trivial consumption attempt.

### 3.10 `RequestDimensionRef` — rollup-aware Dimension carrier (reserved shape)

Forward-extensible replacement for the bare `Request.dimensions: Vec<SemanticsName>` shape. Used when a Dimension needs an axis-rollup transform at `GROUP BY` (e.g. temporal rollup to `month`). Tracked under `[TD-REQUEST-DIM-VARIATION]`; consumed by `[19 §6.2](../../foundations/19_expression_flow.md)`.

```rust
#[non_exhaustive]
pub struct RequestDimensionRef {
    pub name:      SemanticsName,
    pub variation: DimensionVariation,
}

#[non_exhaustive]
pub enum DimensionVariation {
    /// Native model projection — no transform wrap.
    None,
    /// Temporal rollup — wraps the axis in `DATE_TRUNC(grain, axis)` at GROUP BY.
    /// Legal only when the Dimension's `data_type` is temporal.
    Temporal { grain: Grain },
}
```

**Type admissibility.** `Temporal { grain }` against a non-temporal Dimension → `PlanErrorKind::DimensionVariationTypeMismatch`. Non-temporal Dimensions admit only `None` in v1 (string-cased / numeric-bucketed variations are forward-extensible). Multi-axis rollup composes naturally — one `RequestDimensionRef` per requested Dimension, each with its own `variation`.

**Distinct from `[18 §1.2](../../foundations/18_entities.md)`'s `DimensionRef`** (a model-side Semantics-reference-with-overrides carrier — different concern). Naming kept distinct: `RequestDimensionRef` is the request-layer rollup carrier; `DimensionRef` is the model-layer reference shape.

**Computed Dimensions** participate in `GROUP BY` as their materialised column; variation does not apply to computed Dimensions in v1.

**Embedder-level surface.** A CLI or other front-end may accept tokens such as `name.grain` and desugar them to `RequestDimensionRef { name, variation: Temporal { grain } }` before constructing the planner input — embedder convention, not a crate contract.

## 4. `SessionContext`

### 4.1 Shape

```rust
/// Per-invocation runtime parameters the planner consumes. Never
/// `Option` on `Request`; every Request carries a `SessionContext`.
#[non_exhaustive]
pub struct SessionContext {
    pub now:             Timestamp,
    pub timezone:        Timezone,
    pub feature_toggles: FeatureToggles,
    pub correlation_id:  Option<CorrelationId>,
}
```

The planner reads `SessionContext` at exactly two sites: `Expr::Literal` folding for Request filters referencing session-resolved literals (e.g. `as_of = session.now`), and expressions whose resolved `PhysicalExpr` carries a `FunctionCall` whose `FunctionSpec` is session-dependent (`14a §3.5` Q3, currently empty). `#[non_exhaustive]` per I10: additional parameters (e.g. a `tenant_id` for per-tenant rewrites) are MINOR per `30 §2.2`.

### 4.2 Field-by-field

- `**now: Timestamp**` — the single fact every session-aware expression resolves against. The planner never calls `SystemTime::now()` itself (I11); a caller replaying a historical query sets `now` to that instant and every strategy observes it uniformly.
- `**timezone: Timezone**` — `#[non_exhaustive]` newtype over an IANA identifier; `Timezone::utc()` is `Default`. Feeds into `PhysicalExpr::DateTrunc` nodes declaring `zone: ZoneScope::Session` per `13 §3`; v1's `Grain` variants are session-zone-oblivious so the field is carried but not consumed.
- `**feature_toggles: FeatureToggles**` — a `BTreeMap<String, FeatureToggleValue>` shim, `#[non_exhaustive]`. Per I4, the planner's output is a pure function of `(manifest, request, feature_toggles)`. The planner reads toggles only at ratified sites (§11.4, `OptimizerPass::apply`).
- `**correlation_id: Option<CorrelationId>**` — opaque caller-supplied identifier embedded into every emitted `Diagnostic.location.source` so downstream log-aggregation can correlate a request across stage boundaries.

### 4.3 Constructors

```rust
impl SessionContext {
    /// Build a `SessionContext` with `now = SystemTime::now()`,
    /// `timezone = Timezone::utc()`, empty toggles, no correlation ID.
    /// Intended for tests and single-shot CLI tools.
    pub fn new() -> Self;

    /// Build a `SessionContext` pinned to a specific instant in UTC.
    pub fn at(now: Timestamp) -> Self;

    pub fn with_timezone(self, tz: Timezone) -> Self;
    pub fn with_feature_toggle(self, key: &str, value: FeatureToggleValue) -> Self;
    pub fn with_correlation_id(self, id: CorrelationId) -> Self;
}
```

The only constructor that reads the host clock is `new()`. Every other code path must call `at(...)` with an explicit timestamp, preserving I11's "no implicit I/O / host-env dependency on the hot path" rule.

### 4.4 Delta with legacy code

Legacy `crates/semstrait-planner/src/request.rs::SessionVariables` is a `HashMap<String, String>`. `34` ratifies a structured shape with four named fields; free-form session bag moves to `feature_toggles`. Migration tracked as `[TD-SESSION-CONTEXT]` in `docs/TECH_DEBT.md`; the legacy `SessionVariables` alias survives for one MINOR cycle.

## 5. `ResolvedQueryRequest`

### 5.1 Shape

The SemanticManifest-contextualized Request: every `SemanticsName` looked up, every target DataKind resolved (either via explicit `from` or via field-first resolution per §10), every `Filter.field` bound to its `ResolvedSemanticRef`. Produced by step 1 (§7.2) and consumed by step 4 (§7.5). NEVER exposed on the `plan` entry signature — a caller who needs the resolved form destructures the returned `SemanticPlan` instead. The type is `pub` so `Strategy` impls can consume it directly; construction is `pub(crate)` (only the step-1 builder produces well-formed instances).

```rust
#[non_exhaustive]
pub struct ResolvedQueryRequest {
    pub target:     ResolvedTarget,
    pub dimensions: Vec<(SemanticsName, ResolvedSemanticRef)>,
    pub measures:   Vec<(SemanticsName, ResolvedSemanticRef)>,
    pub metrics:    Vec<(SemanticsName, ResolvedSemanticRef)>,
    pub filters:    Vec<ResolvedFilter>,
    pub order:      Vec<(Name, SortDir)>,
    pub limit:      Option<u64>,
    pub offset:     Option<u64>,
    pub session:    SessionContext,
    pub temporal:   Option<TemporalRequest>,
}

#[non_exhaustive]
pub enum ResolvedTarget {
    Explicit(DataKindRef),                            // Request.from = Some(d)
    Implicit(DataKindRef),                            // `16 §11.3` single-kind fast path
    Composition(DataKindRef),                         // `16 §11.4` — graph-index lookup
                                                      //   hit by constituent-set; origin is on
                                                      //   the resolved kind itself
}

#[non_exhaustive]
pub struct ResolvedSemanticRef {
    pub owner:      DataKindRef,
    pub element:    SemanticElement,
    pub binding_id: EntityId,                         // key into manifest `bindings`
}

#[non_exhaustive]
pub enum SemanticElement { Dimension, Measure, Metric, Filter, Key }
```

For `ResolvedTarget::Composition(name)`, the planner consumes the resolved kind through the runtime graph composition index (`graph.composition_index(name)`); traversed paths and constituent sets are graph-build products derived from manifest primitives (`16 §10` / `§11`). Strategies treat `Origin::Explicit` and `Origin::Implicit { id }` uniformly. For Complex-owned fields, `binding_id` points at the constituent Binding the field resolves through per `16 §7`'s `FieldOwnership`.

### 5.2 Invariants

A well-formed `ResolvedQueryRequest` satisfies: every `owner` resolves through graph/manifest lookup (`graph.name_index` + `manifest.data_kinds`); every `binding_id` is present in `manifest.bindings` (I8 / `19 §3.2.3`); `target` is consistent with all `owner` fields; `dimensions` / `measures` / `metrics` entries have `element` matching the field they appear in; `filters` preserve caller order. Violation is a planner-internal bug; step 1 constructs only well-formed instances and consumers MAY `debug_assert!` in test builds.

## Testing Strategy

- Request field roster round-trip (dimensions/measures/metrics/filters/order/limit/offset/from/temporal/session)
- FilterOperator arity-rule enforcement (Eq/NotEq/Lt/LtEq/Gt/GtEq → 1; In/NotIn → ≥1; Between → 2; IsNull/IsNotNull → 0)
- FilterValue type admissibility (String/Number/Integer/Decimal/Bool/Timestamp/Date/Null → per-target DataType)
- SortDir null-aware variants (Asc/Desc/AscNullsFirst/AscNullsLast/DescNullsFirst/DescNullsLast)
- SessionContext constructors (new/at/with_timezone/with_feature_toggle/with_correlation_id)
- ResolvedQueryRequest invariants (owner/binding_id resolution, target consistency, element field match)
- DataKindRef newtype over DataKindName
- TemporalRequest reserved shape (as_of/range fields)
- RequestDimensionRef rollup-aware carrier (Temporal variant type admissibility)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
