---
id: 11-additivity
status: ratified
version: 2.0
impl: implemented
---

# Additivity

Additivity enum, authoring rules, defaults, SemiAdditive axis encoding, and planner contract. This leaf owns the Additivity concept and planner contract; [semantics element catalog](semantics-element-catalog.md) owns the Measure/Metric role definitions.

<!-- cross-cut: none -->

## 7. Additivity

`Additivity` is a classification on Measures and Metrics describing which aggregation operations preserve correctness across which axes. It is authored inline on the Measure or Metric and defaults to `Additive` when omitted. It does NOT derive from the owning DataKind's `TemporalShape`.

Additivity is shape-locked (§5.1) per Semantics name. If a Metric `cpc` is authored `NonAdditive` in any occurrence, every other occurrence of `cpc` must either omit `additivity:` (accepting `NonAdditive`) or state `NonAdditive` explicitly. Stating a different Additivity at a different occurrence is a compile error (`CompileError::SemanticShapeConflict`).

### 7.1 Enum

```rust
enum Additivity {
    /// Safe to aggregate by any subset of the DataKind's Dimensions.
    Additive,

    /// Safe across some Dimensions but unsafe across others (typically
    /// unsafe across time). Additional metadata specifies which axes.
    SemiAdditive,

    /// Not mechanically aggregable; must be recomputed at the queried
    /// grain from underlying Measures. Planner cannot roll up.
    NonAdditive,
}
```

### 7.2 Authoring and default

Additivity is a shape field (§5.1) authored inline on any Measure or Metric occurrence. All occurrences of a given Semantics name must agree on Additivity. Default when every occurrence omits `additivity:`: **`Additive`**.

**Rationale for default-`Additive` (vs. deriving from `TemporalShape`):**

- Additivity describes aggregation semantics (how a numeric quantity composes under rollup). TemporalShape describes DataKind historization (how the underlying data records time). These are independent concerns; correlation between them is heuristic, not definitional.
- A Measure over a Snapshot DataKind can legitimately be `Additive` when the modeler's query methodology treats it correctly (for instance, always querying at one snapshot time). Auto-derivation would silently override the modeler's intent.
- Automatic derivation creates surprising defaults when the same Measure is used across DataKinds with different TemporalShapes — the Measure's Additivity would swing depending on which DataKind the author drafted first, violating global-identity ergonomics.
- Explicit authoring surfaces aggregation intent at the point of declaration. Modelers who want SemiAdditive or NonAdditive write it; the common case (Additive) is the default.

**Note on Metrics.** The default `Additive` is often wrong for Metrics, because the majority of authored Metrics are ratios (`cpc = cost / clicks`) which are `NonAdditive`. The default is kept identical to the Measure default for consistency (uniform rule across the Semantics namespace), but authors should treat the Metric default as a deliberate prompt to verify — not a recommendation. A future lint (tracked as TD, not ratified here) MAY warn when a Metric's `expr:` contains a non-linear operator (`/`, `*` with non-constant operands, etc.) and Additivity is left at the default.

### 7.3 Relationship to `TemporalShape`

`TemporalShape` (17) drives planner behavior independently of Additivity. At query time the planner uses `TemporalShape` to pick candidate sources, choose rollup levels, and resolve as-of semantics (17, 20–25). Additivity is a separate input the planner consults to decide whether mechanical rollup is legal for a Measure or Metric at a requested grain.

The planner MAY emit a Warning Diagnostic when Additivity and TemporalShape appear inconsistent — for example, `Additive` on a Measure whose containing DataKind has `TemporalShape: SCD` with a history-preserving subtype, where mechanical additivity would double-count within valid windows. These warnings are advisory; they do not block compile or plan, and they are overrideable (a future `additivity_confirmed: true` flag, TBD, may suppress them). Policy details ratified in `20–25`.

### 7.4 SemiAdditive axis encoding

Initial authoring shape (ratified here; full planner policy deferred to `20–25`):

```yaml
- name: account_balance
  data_type: decimal
  agg: sum
  additivity: semi_additive
  semi_additive:
    unsafe_axes: [date]       # Dimension names
```

Rules:

- Present only when `additivity: semi_additive`. Invalid on `additive` / `non_additive` — `ValidateError::InvalidSemiAdditiveBlock`.
- `unsafe_axes:` is a list of Dimension names from the Semantics's owning DataKind's interface. Unknown names are `CompileError::UnknownReference`. Names outside the DataKind's interface are `CompileError::SemanticNotInInterface`.
- All axes NOT in `unsafe_axes:` are implicitly safe. Empty list is illegal — `semi_additive` with zero unsafe axes is semantically `additive`; use that.
- Shape-locked per §5.1: the `semi_additive:` block is a shape field and must unify across all occurrences.

**Design choice — single `unsafe_axes:` list, not paired safe/unsafe.** One list is the complement of the other; carrying both invites disagreement (which list wins if a Dimension is in neither or in both?). The negative framing ("axes where mechanical rollup is wrong") matches how authors reason about SemiAdditive — the safe axes are "everywhere else" by default.

**Alternative considered — typed reference to `TemporalShape`'s time axis.** Rejected for initial design: SemiAdditive along non-temporal axes (e.g. balance "unsafe to sum across accounts of the same customer") is a real use case. A freeform Dimension-name list handles both temporal and non-temporal cases uniformly.

**Deferred.** Per-strategy semantics for rollup across unsafe axes (last-value / first-value / period-close) is ratified in `20–25` together with `TemporalShape`-driven defaults (17).

### 7.5 Planner consequences (contract)

For a Request at grain `G` on DataKind `K` naming Semantics `S`:

| `Additivity` | Planner behavior |
|---|---|
| `Additive` | Emit the mechanical aggregate (`agg(expr)`) at `G`. No special handling. |
| `SemiAdditive` | If every Request group-by Dimension is in the safe set (complement of `unsafe_axes:`), treat as `Additive`. If any group-by Dimension is in `unsafe_axes:`, apply the unsafe-axis policy ratified in `20–25` (typically last-value for Snapshot, period-close for SCD). |
| `NonAdditive` | Refuse mechanical rollup. For a Metric: materialize referenced Measures at `G` and compute the Metric in the projection layer. For a Measure: `PlanError::NonAdditiveRollupRequired` if no underlying column is available to recompute at `G`; else emit the recompute plan (treating the Measure as a Metric-like projection over finer-grain aggregates). |

Full per-variant planner rules, including the SCD / Snapshot / Timeseries / Events interaction with SemiAdditive, are ratified in `20–25`. This section fixes the contract — the planner MUST consult `Additivity` at every grain-changing plan step, MUST refuse mechanical rollup on `NonAdditive`, and MUST apply the unsafe-axis policy on `SemiAdditive`.

> **v1 scope (STATUS item V).** The `Additivity` the planner consults in v1 is **function-derived only** (the aggregate op via `AdditivitySource`, `14a §3.6.2`). Model-level `additivity:` (incl. `SemiAdditive` axes) is authored and validated but **not yet consulted at plan time** — so the `SemiAdditive` row above is the reserved contract for when model-level additivity lands (`19 §6.5`). A `sum(snapshot_balance)` therefore plans as `Additive` in v1 regardless of an authored `additivity: non_additive`; advisories flag suspect Additivity × TemporalShape combinations in the interim.

## Testing Strategy

- Additivity enum validation (parse `Additive` / `SemiAdditive` / `NonAdditive`)
- Default-Additive application (when omitted, §7.2)
- SemiAdditive axis encoding (`unsafe_axes:` list of Dimension names; shape-locked; §7.4 rules)
- Planner contract per variant (§7.5 table: Additive/SemiAdditive/NonAdditive behavior)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
