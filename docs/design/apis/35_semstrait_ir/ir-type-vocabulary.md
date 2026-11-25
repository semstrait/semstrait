---
id: 35-ir-type-vocabulary
status: ratified
version: 2.1
impl: implemented
---

# IR Type Vocabulary

The canonical logical-type primitives split by authorability (item Y / V2). The **authored** primitives — `DataType` (14 authorable variants + the compile-transient `Unknown` sentinel) and `Grain` (7 temporal levels) — are **owned by `semstrait-model`** (`32 §4.3`) and imported here; their variant rosters are ratified by `13`. The **non-authored** primitives — `TypeClass`, `Schema`, `SchemaColumn` — describe resolved physical shape that never appears in authored YAML, and are owned **here**. All five are referenced across the post-compile pipeline (expression leaves, function signatures, plan-tree primitives, adapter renderers).

## `DataType` — Imported from `semstrait-model`

```rust
use semstrait_model::DataType;
```

`DataType` is **owned by `semstrait-model`** (item Y / V2; body + `parse_scalar` + serde discipline at `32 §4.3`, roster ratified by `13 §2.1`). `semstrait-ir` imports it and references it from every resolved consumer — expression leaves and `Cast { target }`, function signatures (`ParamType::Exact`), plan-tree primitives (`ResolvedColumn`), and adapter renderers. The compile-transient `Unknown` sentinel (`19 §3.6`–`§3.7`) is part of the model-owned enum but never persists into a sealed `PhysicalExpr`; the write-side refusal that enforces this lives with the type at `32 §4.3`.

## `Grain` — Imported from `semstrait-model`

```rust
use semstrait_model::Grain;
```

`Grain` and its `coarseness() -> u8` accessor are **owned by `semstrait-model`** (item Y / V2; body at `32 §4.3`, roster + total order ratified by `13 §3`). Imported here for plan-level granularity reasoning (Grainset level selection).

## `TypeClass` — Type-Class Grouping

```rust
/// Bounded type classes used by `FnSignature` parameter expression per
/// `14a §3.3`. Pure grouping vocabulary; no method surface. Per `13 §4`.
#[non_exhaustive]
pub enum TypeClass {
    Numeric,      // Byte | Short | Integer | Long | Float | Double | Decimal
    Integral,     // Byte | Short | Integer | Long
    FloatingPt,   // Float | Double
    Textual,      // String
    Temporal,     // Date | Time | Timestamp
    Comparable,   // everything except Binary in the canonical set
    Any,          // all canonical variants
}
```

`TypeClass` is exposed but **not** wired into the v1 `ParamType` activation — `14a §3.3` Q6 ratified overload-set polymorphism, not type-class generics, for v1. `TypeClass` exists as vocabulary for future registry evolution (`TD-REGISTRY-TYPECLASS`, `14a §10.1`) and for documentation / advisory diagnostics. The reserved `ParamType::TypeClass(TypeClass)` variant lives in [ir-function-registry.md](ir-function-registry.md).

## `Schema` and `SchemaColumn` — Physical-Source Schema

```rust
/// The compile-time snapshot of physical columns exposed by a source.
/// Per `15 §3.2`. Referenced by:
/// - `15 §3.1 PhysicalSource::{File, Stream, Table, ObjectStore}` for the
///   resolved schema attached to every `PhysicalSource` variant.
/// - NodeMeta.output_schema for the per-`PlanNode` output
///   schema (via plan-level `Schema`, which has the same shape).
pub struct Schema {
    pub columns: Vec<SchemaColumn>,
}

pub struct SchemaColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}
```

`Schema.columns` is ordered in the source's native order (Parquet footer, Iceberg field order, CSV header order, …) per `15 §3.2`. Compile preserves the ordering for determinism (I4); the planner does not semantically depend on it. `nullable` is read from source metadata when available; it is not inferred. Both types are **field-stable** (no `#[non_exhaustive]`) per shared-vocabulary policy.

## Nullability

Nullability is **NOT** exposed as a separate enum. Per `13 §2` and `14a §3.4` Q7, `DataType` is nullable-by-default; per-column nullability lives as the `SchemaColumn.nullable: bool` field. A standalone `Nullability` type was considered and rejected at `14a §3.4`.

## Serialization Feature Flag

```rust
#[cfg(feature = "serde")]
impl serde::Serialize for Schema { /* ... */ }
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Schema { /* ... */ }
// ... similarly for SchemaColumn and TypeClass (all derived — no sentinel).
```

Feature-gated per §15.1, covering the **ir-owned** primitives (`TypeClass`, `Schema`, `SchemaColumn`) only. Off by default; `semstrait-manifest` / `semstrait-planner` enable it transitively (they round-trip values carried through these primitives). The imported `DataType` / `Grain` carry their own serde — including `DataType`'s hand-written `Unknown`-refusal discipline — in their owning crate (`32 §4.3`); `semstrait-ir` does not re-impl serde for imported types (orphan rule). `Schema` / `SchemaColumn` embed `DataType` as a field, so their derived impls compose with the model-owned `DataType` serde transparently.

## Testing Strategy

This leaf owns the non-authored type vocabulary (`TypeClass`, `Schema`, `SchemaColumn`). Tests:

- **`TypeClass` grouping correctness:** Unit tests verify each `TypeClass` variant enumerates the correct `DataType` subset per `13 §4` (e.g., `Numeric` includes `Integer` / `Float` / `Decimal` but not `String`).
- **`Schema` / `SchemaColumn` construction:** Integration tests verify schema is populated from source metadata (Parquet footer, catalog fetch) and preserved in `NodeMeta.output_schema` on every `PlanNode`.
- **Serde roundtrip under `serde` feature:** Feature-gated tests verify `Schema` / `SchemaColumn` / `TypeClass` serialize to deterministic JSON and deserialize to structurally-equal values. The embedded `DataType` serde composes transparently; `Unknown` refusal (model-side) is never triggered on ir-owned primitives.
- **Nullability representation:** Unit tests verify `nullable` bool field is read from source metadata when available; no separate `Nullability` enum exists.

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
