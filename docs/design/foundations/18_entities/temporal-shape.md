---
id: 18-temporal-shape
status: ratified
version: 2.1
impl: implemented
---

# Temporal Shape

This leaf owns the `TemporalShape` type hierarchy — struct, `TemporalShapeKind` enum, per-variant body structs, and `ScdType`. See `13` for the `Grain` roster and `17` for planner-level temporal semantics.

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
```

```rust
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

The `Grain` enum is owned by [`13 §3.1`](../13_types_and_grain.md#31-enum); the v1 roster is `{Minute, Hour, Day, Week, Month, Quarter, Year}` with total coarseness order in declaration order (finest first). `18 §3` consumes `Grain` only through the optional `TemporalShape.grain` field; the per-variant `*Body` structs (§3.1) carry the identifying-Dimension name, not a grain. It does not redefine the enum. Non-temporal grains (geographic, entity) are a deferred extensibility axis per `13`.

> **Note (2026-04-17 consolidation).** An earlier draft of this section inlined a `pub enum Grain` block that accidentally introduced a `Second` variant not present in `13 §3.1`. Per the precedence rule in `00 §4.4` + the directionality rule in `00 §8`, `13` is the canonical home; the divergent roster has been removed. Any future addition to `Grain` (e.g., `Second`) must land in `13 §3.1` first.

### 3.6 Forward reference to `17`

Planner-level semantics — the shape × grain rollup matrix, snapshot pin policies, SCD as-of anchoring — live in `17`. `18 §3` ratifies the shape of the type and its YAML carriage; `17` ratifies what each shape *means* for request execution.

## Testing Strategy

- SR-E-6 (leaf-missing-grain), SR-E-7 (grain-on-complex forbidden), SR-E-8 (Grainset-child-grain-required); variant-only vs variant+grain; YAML flattening; cascade vs explicit grain rules.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
