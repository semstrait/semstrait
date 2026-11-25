---
id: 11-constraint
status: ratified
version: 2.0
impl: implemented
---

# Constraint

Constraint framework — carriers, kind sub-blocks, v1 realized carriers (Measure/Metric), reserved carriers, validation lifecycle, error model. This leaf owns the Constraint concept and evaluation framework; [semantics element catalog](semantics-element-catalog.md) owns the Measure/Metric/Dimension/Filter/Key role definitions.

<!-- cross-cut: 11-semantics-element-catalog -->

## 8. Constraint

A **Constraint** is a declarative rule that narrows how a Semantics element participates in computation. The concept is **element-agnostic** — any carrier type (Measure, Metric, Dimension, Filter, Key, DataKind) MAY carry Constraints. The v1 implementation realizes the carrier mechanism for **Measure and Metric**; the other carriers are architecturally reserved extension points (§8.5). The DSL shape — a nested `constraints:` block composed of reusable **kind sub-blocks** — is itself carrier-agnostic (§8.3); different carriers select different kind sub-blocks appropriate to what that element does.

**Not Constraints** (boundaries the `§8` framework does not cover):

- SQL-style relational integrity (`NOT NULL`, `UNIQUE`, `FOREIGN KEY`, `PRIMARY KEY`) — Preconditions in `§12.2` (`N-C3` … `N-C9`).
- System-level compiler invariants ("SUM only on Measure `expr:`, not on a Dimension / Key column") — validate/compile-stage structural rules in `14` / `15` / `§12.2`.
- Engine-side operand-type admissibility — deferred to adapters per the `14 §5.4` pass-through posture.
- Arbitrary boolean-predicate escape-hatches — every admissible Constraint kind is a closed, structured sub-block. The `14 §6.3` inline DSL is not reused inside `constraints:`.
- Authorable severity — every Constraint violation is a hard error (§8.7).

### 8.1 The Constraint concept

A Constraint binds to one **carrier element** (the Semantics it restricts). It is composed of one or more **kind sub-blocks**, each drawn from a small reusable toolkit (§8.3). Each kind sub-block expresses a single admissibility rule using a familiar vocabulary (three-way set membership, two-way whitelist/blacklist, bounded range, identifier list).

Constraints are evaluated at a stage appropriate to the carrier and kind (§8.6). Violations are hard errors that abort the host stage immediately.

### 8.2 Carriers

Every Semantics element type is a candidate carrier. The carrier table below records the current realization status:

| Carrier | Role of Constraints | v1 state |
|---|---|---|
| **Measure** | Narrow which Dimensions may / must appear in the Request's query scope; narrow which aggregation functions are admissible | **Realized** — §8.4.1 |
| **Metric** | Narrow which Dimensions may / must appear in the Request's query scope; narrow which aggregation functions are admissible (applies to two-stage Metrics per `CompiledMetric.agg`) | **Realized (partial)** — §8.4.2 |
| **Dimension** | *Implicit today.* Grain admissibility is expressed by `13 §3` `grains:`; Key-membership is expressed by `§6.5`'s Key declarations. Future explicit `constraints:` block could carry null-policy, allowed-rollup paths, value-set policies, etc. | **Reserved** — §8.5.1 |
| **Filter** | Filter-reachability from carrier DataKinds; auto-application policy (the `requires:` mechanism the code comment reserves); admissibility of targeted Dimension names | **Reserved** — §8.5.2 |
| **Key** | Enforcement posture for Key uniqueness at the stated grain; admissibility of Dimensions as Key members | **Reserved** — §8.5.3 |
| **DataKind** | Row-count / grain-shape bounds; cross-Semantics invariants that bind at the DataKind scope | **Reserved** — §8.5.4 |

Each realized or reserved carrier has its own per-carrier `constraints:` field. The field is independently optional per carrier. The **set of kind sub-blocks admissible inside a carrier's `constraints:` block is closed per carrier** (§8.4 / §8.5); unknown kind names are a parse-stage rejection.

### 8.3 The kind sub-block toolkit

A reusable vocabulary that kind sub-blocks compose from. v1 realizes the first two; the remaining entries are reserved for future kind sub-blocks.

| Pattern | Shape | Example | Used by |
|---|---|---|---|
| **Three-way set policy** | `{ one_of: [...], none_of: [...], all: [...] }` — all three fields independently optional; AND-combined | `dimensions: { one_of: [date], none_of: [pii], all: [region] }` | `dimensions` kind (§8.4.1) |
| **Two-way list policy** | `{ allowed: [...], prohibited: [...] }` — both optional; both may co-exist | `aggregations: { allowed: [SUM, MIN], prohibited: [COUNT_DISTINCT] }` | `aggregations` kind (§8.4.1) |
| **Bounded range** | `{ min: N, max: M }` — both optional; `min ≤ max` when both present | (future) `row_count: { min: 1 }` | *reserved* |
| **Identifier list** | `[name_a, name_b]` — flat list of Semantics/Filter names | (future) `requires: [is_active]` | *reserved* |
| **Structured scalar bound** | typed value with bound (e.g. `{ max: 0.05 }` for a ratio) | (future) `null_fraction: { max: 0.05 }` | *reserved* |

A kind sub-block MAY compose multiple patterns. New kinds SHOULD use these patterns before inventing new shape grammars.

### 8.4 v1 realized carriers

#### 8.4.1 Measure `constraints:`

Two kind sub-blocks admissible:

**`dimensions:`** — three-way set policy (§8.3) over Dimension names. Restricts the Request's *query scope* = `request.dimensions` ∪ `{filter.field | filter ∈ request.filters, filter.field names a Dimension on the declaring DataKind's interface}`. Filter fields that are not known Dimensions do not contribute to scope.

| Field | Semantics |
|---|---|
| `one_of: [A, B, C]` | Query scope MUST include ≥ 1 of the listed Dimensions |
| `none_of: [X, Y]` | Query scope MUST NOT include ANY listed Dimension |
| `all: [P, Q]` | Query scope MUST include EVERY listed Dimension |

**`aggregations:`** — two-way list policy (§8.3) over UPPERCASE aggregation-name tokens matching the `Aggregation` enum names (currently `SUM`, `AVG`, `COUNT`, `COUNT_DISTINCT`, `MIN`, `MAX`; see `crates/semstrait-planner/src/validator.rs::agg_constraint_name`).

| Field | Semantics |
|---|---|
| `allowed: [...]` | Measure's effective aggregation MUST be one of the listed names |
| `prohibited: [...]` | Measure's effective aggregation MUST NOT be one of the listed names |

Authorable form:

```yaml
measure:
  name: order_amount
  agg: SUM
  data_type: Decimal(18, 4)
  constraints:
    dimensions:
      one_of: [date, order_month]
      none_of: [pii_customer_ssn]
      all: [region]
    aggregations:
      allowed: [SUM, MIN, MAX]
      prohibited: [COUNT_DISTINCT]
```

Unknown keys inside any Measure-`constraints:` sub-block are `ValidateError::ShapeMalformed` via serde's default unknown-field rejection. Both sub-blocks are independently optional; a `constraints:` block carrying only `dimensions:` (or only `aggregations:`) is legal.

#### 8.4.2 Metric `constraints:`

The same two kind sub-blocks as §8.4.1 are admissible. Authoring matches §8.4.1's grammar verbatim. Evaluation differs per kind:

- **`dimensions:`** fires identically to §8.4.1.
- **`aggregations:`** applies to Metrics that carry an `agg:` field (two-stage Metrics per `semstrait-manifest::CompiledMetric.agg: Option<Aggregation>`). For Metrics without `agg:` (pure derived expressions), the current validator silently skips the `aggregations:` check; this is the forward-compatibility behavior — a pure Metric has no direct Request-level aggregation surface.

> **[TD-AGG-ON-METRIC]** — open design question: should authoring `aggregations:` on a Metric that has no `agg:` field be a validate-stage error (hard reject), or remain silently ignored (current)? The silent-skip is convenient for forward-compatibility but hides likely author mistakes. Deferred for future tightening.

#### 8.4.3 Model-layer type name

The v1 implementation's struct is named `MeasureConstraints` in `semstrait-model::types::measure` and is attached to both Measure and Metric carriers. The name is a **legacy artifact** — the spec treats Constraints as per-carrier, so the conceptual naming is `{Measure,Metric}Constraints` (distinct types per carrier) even though the code currently reuses one struct.

> **[TD-CONSTRAINT-RENAME]** — rename `MeasureConstraints` in the model crate to a name that reflects its cross-carrier reuse (e.g. `ElementConstraints`, or per-carrier `MeasureConstraints` + `MetricConstraints`). Schedule with the broader SemanticManifest-schema revision pass; not a v1 blocker.

### 8.5 Reserved carriers (future design)

Carriers listed here are architecturally supported but have **no concrete kind sub-blocks realized in v1**. Each section names candidate kinds for scoping discussions; none are authoring-legal until specified.

#### 8.5.1 Dimension `constraints:`

Today, Dimension-level restrictions are expressed through dedicated shape fields: grain admissibility via `13 §3` `grains:`, Key participation via `§6.5`, metadata-source hints via `13 §4.7`. A future explicit `Dimension.constraints:` block could carry kinds like:

- `rollup:` — allowed-grain-rollup paths (extends `13 §3`'s grain lists with composition rules).
- `null_policy:` — participation when the Dimension is NULL for a row (three-way: require non-null / allow / forbid).
- `value_set:` — enum-style admissibility (deferred from the Round-1 value-predicates discussion).

None are authoring-legal in v1.

#### 8.5.2 Filter `constraints:`

Candidate kinds — not authoring-legal in v1:

- `reachability:` — DataKinds the Filter is admissible from (restricts which Requests may name this Filter).
- `requires:` — the Filter-injection mechanism the source comment in `semstrait-common::constraints` reserves (`// This is NOT \`requires\``). When authored on a Measure / Metric / DataKind (carrier TBD), declares that a named Filter must be in effect; planner auto-injects at step 0 or at a dedicated plan sub-step. Tracked as `[TD-REQUIRES-MECHANISM]`.

The future placement of `requires:` — whether on Filter itself or on Measure/Metric/DataKind — is part of the `[TD-REQUIRES-MECHANISM]` design work.

#### 8.5.3 Key `constraints:`

Keys today are purely structural metadata per `§6.5`. Candidate kinds — not authoring-legal in v1:

- `uniqueness:` — enforcement posture (advisory-only / plan-time-assert / runtime-check) for the Key's uniqueness at the stated grain. Today semstrait *assumes* Keys are unique; an explicit posture would let authors downgrade the assumption for nominally-key-like structures.
- `member_policy:` — which Dimension types / roles are admissible as Key members (e.g. forbid computed Dimensions, require physical-column binding).

#### 8.5.4 DataKind `constraints:`

Candidate kinds — not authoring-legal in v1:

- `row_count:` — bounded-range policy on the DataKind's row count at plan time. Requires a `CatalogProvider` statistics API not yet specified. Tracked as `[TD-CARDINALITY-CONSTRAINT]`.
- `null_fraction:` — per-column null-fraction bound evaluated at plan time; same `CatalogProvider` dependency.

The framework retains Cardinality constraints as DataKind-level kinds, deferred to the stats extension.

### 8.6 Validation lifecycle

**Per-carrier staging.** There is no single stage at which all Constraints run. Each carrier + kind pairing selects its natural stage:

| Carrier | Kind | Stage | Rationale |
|---|---|---|---|
| Measure | `dimensions` | **step 0** (pre-resolution) | Needs full Request scope (group-by + filters); runs before `10 §3.4` sub-steps |
| Measure | `aggregations` | step 0 | Same entry point; needs the Request's Measure-usage context |
| Metric | `dimensions` | step 0 | Same |
| Metric | `aggregations` (two-stage) | step 0 | Fires only when Metric has `agg:`; otherwise silent-skip per `[TD-AGG-ON-METRIC]` |
| *(Reserved)* Dimension `rollup:` | — | future `plan` sub-step | Needs resolved Request scope |
| *(Reserved)* Filter `reachability:` | — | future `compile` | Can be resolved statically against the SemanticManifest |
| *(Reserved)* Filter `requires:` | — | future `plan` sub-step (injection) | Needs Request context to know which Filters are already named |
| *(Reserved)* DataKind `row_count:` / `null_fraction:` | — | future `plan` sub-step (post-catalog) | Needs `CatalogProvider` stats |

**v1 evaluation entry point.** For realized carriers (Measure, Metric), `ConstraintValidator::check()` runs as the planner's first action — **step 0, pre-resolution** — BEFORE dataset routing, `from:`-resolution, Relationship traversal, or PlanNode construction. Per-Measure / per-Metric algorithm:

1. Resolve `request.entity_name` to a `CompiledDataKind` via the `SemanticManifest`. (If `entity_name` is empty — ad-hoc query — constraint validation is skipped entirely.)
2. Build the *query scope* set: `request.dimensions` ∪ filter-field Dimensions (§8.4.1).
3. For each name in `request.measures`:
   - If the name resolves to a Measure: run dimensions-check AND aggregations-check (passing the Measure's effective `agg` name).
   - Else if the name resolves to a Metric: run dimensions-check; aggregations-check fires only if `CompiledMetric.agg` is set (per `[TD-AGG-ON-METRIC]`).
   - Else: no action — a later plan stage raises the unresolved-reference error.
4. First violation short-circuits with `PlannerError::ConstraintViolation` — fail-fast.

### 8.7 Error model, accumulation, and severity

**Error carrier (v1).** Single typed variant on the planner:

```rust
pub enum PlannerError {
    ConstraintViolation { entity: String, message: String },
    // …
}
```

`entity` is the Semantics name the failed constraint attached to; `message` is a free-form rule-identifying string (e.g. `"one_of constraint violated: query must include at least one of [date, order_month]"`). No per-rule enum fan-out.

> **[TD-CONSTRAINT-ERROR-FANOUT]** — typed fan-out (e.g. `ConstraintError::{DimOneOf, DimNoneOf, DimAll, AggAllowed, AggProhibited, …}` with structured payloads for `Diagnostic` rendering per `10 §5`) is deferred. Decide carrier shape together with the broader `Diagnostic` model work and any v2 constraint kinds from reserved carriers (§8.5).

**Accumulation.** Fail-fast per carrier evaluation. First violation returns; subsequent constraints are not evaluated for that Request.

**Severity.** Hard error only. No authorable severity. If a future carrier requires lint-style warnings (e.g. Dimension key-participation advisories), they will live in a separate `lint:` block distinct from `constraints:`, not as a severity toggle.

### 8.8 DSL — shape field semantics and authoring

**Shape-field.** `constraints:` is a **shape field** per `§5.1` on every realized and reserved carrier. Two occurrences of the same Semantics across Binding vs. Semantics layers MUST declare deep-equal `constraints:` blocks or error `ValidateError::ShapeFieldConflict`.

**Closed kind-vocabulary per carrier.** The set of kind sub-blocks admissible inside a given carrier's `constraints:` block is closed. Unknown kind names fail parse with `ValidateError::ShapeMalformed` (via serde). This closure is per-carrier — a future Dimension kind (`rollup:`) would not activate on Measure.

**Declarative-only.** No inline predicate grammar inside `constraints:`. The inline expression form architecturally reserved by the authoring expression surface (`14 §6`; the inline DSL is deferred, compound strings parse-rejected per doc 45) is not reused here in v1. If a future kind requires a predicate body, a separate design pass decides whether to activate the inline grammar and which wrapper (SemanticExpr / PhysicalExpr) applies.

## Testing Strategy

- Constraint carrier table integrity (§8.2: Measure/Metric/Dimension/Filter/Key/DataKind; v1 realized Measure+Metric, others reserved)
- Kind sub-block toolkit validation (§8.3: three-way set policy, two-way list policy, bounded range, identifier list, structured scalar bound)
- `dimensions` / `aggregations` constraint enforcement on Measure/Metric (§8.4.1/§8.4.2)
- Validation lifecycle table (§8.6: per-carrier staging; step 0 for Measure/Metric `dimensions`+`aggregations`)
- `ConstraintViolation` error (§8.7: entity + message; fail-fast)
- Fail-fast accumulation (first violation returns; subsequent constraints not evaluated)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
