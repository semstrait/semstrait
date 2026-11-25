---
id: 32-model-extras-mapping
status: ratified
version: 2.1
impl: implemented
---

# Extras Blocks and Semantic Mapping

This leaf defines the two `Extras` flavors (LeafExtras / ComplexExtras) and the semantic mapping grammar that binds logical names to physical columns.

## 1. The `Extras` Blocks — `LeafExtras` and `ComplexExtras`

Two `Extras` types, one per structural axis. The split is the type-level expression of R-6 (`storage` / `catalog` / `semantic_mapping` are leaf-only — never authored on a Complex variant):

```rust
#[non_exhaustive]
pub struct LeafExtras {
    pub catalog:          Option<CatalogRef>,
    pub storage:          Option<StorageConfig>,
    pub semantic_mapping: Option<SemanticMapping>,
    pub temporal:         Option<TemporalShape>,
}

#[non_exhaustive]
pub struct ComplexExtras {
    pub temporal:         Option<TemporalShape>,
}
```

Both apply `#[serde(deny_unknown_fields)]`. A YAML author who places `catalog:` / `storage:` / `semantic_mapping:` under a Complex variant's `extras:` block hits `ParseErrorKind::UnknownField` at parse time — the field has no slot to deserialize into.

`StorageConfig` carries the physical-source list (`paths:` for files / folders / globs; `tables:` for catalog FQNs / table-name globs) and an optional declared partition layout (`partition_def:`):

```rust
#[non_exhaustive]
pub struct StorageConfig {
    /// Storage format for path-based sources (`Parquet`, `Csv`, `Json`,
    /// `Orc`, `Avro`). Required when `paths` is non-empty; ignored when
    /// only `tables` is authored (catalog metadata supplies the format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<StorageFormat>,

    /// File / folder / glob URIs. Each entry resolves at compile to one
    /// or more `PhysicalSource::File` per `15 §3.5`: a concrete path or
    /// folder URI produces one `PhysicalSource`; a wildcard path produces
    /// one `PhysicalSource` per resolved variation. Each `PhysicalSource`
    /// is an engine-level LogicalRelation — the engine handles file
    /// consolidation, schema merge, and Hive-partition discovery
    /// internally per `35 §5.2.1`. Empty by default.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<String>,

    /// Catalog FQNs or table-name globs. Each entry resolves at compile
    /// to one or more `PhysicalSource::Table` per `15 §3.5`: a concrete
    /// FQN produces one `PhysicalSource`; a table-name glob produces one
    /// `PhysicalSource` per FQN enumerated by `CatalogProvider::list_tables`.
    /// Empty by default.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<String>,

    /// Catalog-less partition declaration for file sources (Range / List).
    /// First-class v1 author surface; **runtime-dormant in v1** — parsed,
    /// schema-validated, and carried through compile for v2+ partition-
    /// aware planning. v1 adapters defer partition pruning to engine-side
    /// discovery from filter predicates per `35 §5.2.1`. See `Q-MAP-002`
    /// for the layered partition-info plumbing decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition_def: Option<PartitionDef>,
}
```

`PartitionDef` describes the on-disk partitioning layout for a physical source. At least one of `paths:` / `tables:` must be non-empty; both being empty is a structural error caught at validate (per the leaf-source rule in `15 §2.1`).

YAML tag names mirror field names 1:1. `TemporalShape` uses the collapsed variant-wrapper form ratified in `18 §3.2`. Three example shapes:

```yaml
# Path-based source (recommended form: table-root prefix; engine handles
# Hive-partition discovery and file consolidation internally).
extras:
  catalog: polaris_prod
  storage:
    format: parquet
    paths:
      - "s3://bucket/orders/"
    partition_def:
      type: range
      column: order_date
  semantic_mapping: auto
  temporal:
    timeseries:
      occurred_at: order_date
    grain: day
```

```yaml
# Table-based source (catalog-resolved; format omitted — supplied by catalog).
extras:
  catalog: polaris_prod
  storage:
    tables:
      - "warehouse.sales.orders"
```

```yaml
# Mixed source list (Union ALL across one path-based and one table-based
# `PhysicalSource`; per `21 §3.2 / §4.5`).
extras:
  catalog: polaris_prod
  storage:
    format: parquet
    paths:
      - "s3://bucket/legacy_orders/"
    tables:
      - "warehouse.sales.orders_current"
```

**Authoring guidance — table-root preferred for file sources.** When a path is a Hive-partitioned table or a single-table folder containing many files, the recommended form is the **table-root prefix only** (e.g. `"s3://bucket/orders/"`), not a Hive-partition glob (e.g. `"s3://bucket/orders/year=*/month=*/*.parquet"`). The latter is wrong usage of the wildcarding surface — it forces compile to enumerate per file or per partition, when the engine can do the same far more efficiently from the table-root alone (per `35 §5.2.1`'s 4-consumer alignment). Compile resolves whatever the author writes literally; it does not detect or reject this pattern. See `15 §3.5` for the resolution rule.

### 1.1 Per-effective-level validity

Each extras field is constrained by **two independent type-level rules**: which extras flavor (Leaf / Complex) carries the field, and which structural-axis trait owns it. Cascade-from-ancestor remains in v1 only for `temporal.<variant>:` (the shape kind); no other extras field cascades.

<!-- cross-cut: none -->

| Field                                    | Carrier                                               | Effective at                              | Cascadable from ancestor?                                 |
| ---------------------------------------- | ----------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| `catalog`                                | `LeafExtras`                                          | Leaf (`Dataset` / `NestedDataset`)        | **No** — type-level (not in `ComplexExtras`)              |
| `storage` (incl. nested `partition_def`) | `LeafExtras`                                          | Leaf                                      | **No** — type-level                                       |
| `semantic_mapping`                       | `LeafExtras`                                          | Leaf (default `auto` when absent)         | **No** — type-level                                       |
| `temporal.<variant>:` (shape kind)       | both                                                  | Leaf; inherited from any ancestor complex | **Yes** (more-specific-overrides-default merge)           |
| `temporal.grain:`                        | `LeafExtras` only — **forbidden** on Complex (SR-E-7) | Leaf `Dataset` only                       | **No** (SR-E-8: Grainset children must author explicitly) |

The cascade rule applies to the variant-tag layer of `temporal:` only. An author may declare `temporal: { timeseries: {...} }` on a root grainset and have that shape kind cascade to every leaf descendant; the `grain:` value never cascades.

Authoring `temporal:` at a scope with no eligible descendant is a structural warning (dead config, not fatal). Entity-level invariants (SR-E-6 through SR-E-8) live at `18 §11`.

### 1.2 Variant-intrinsic fields that are NOT in either `Extras` flavor

- `UnionMode` is a direct field on `UnionsetBody`. Always required; default `All`. Roster `{All, Unique}` — see `23 §4.1` for the full enum and `16 §5` for composition semantics.
- `relationships` on `JoinsetBody` is the variant's intrinsic structural field — never overridable through extras. Uses the unified `Relationship` shape per `18 §2`.
- `SemanticInterface` (dimensions / measures / metrics / keys / filters) is authored directly on every Public form, never in extras. Entry grammar (inline vs `ref`, override scope) lives at `18 §1`.

## 2. Semantic Mapping and Binding

### 2.1 `semantic_mapping` — authoring grammar

`semantic_mapping` lives in `extras`. Two authoring forms:

**Implicit default (`auto`).** Absent value or `semantic_mapping: auto` means the compiler treats every Semantic name as a 1:1 match to a physical column by the same name. Referential integrity is the author's responsibility.

```yaml
# Both are equivalent — `auto` is the default when omitted.
extras: {}
extras:
  semantic_mapping: auto
```

**Explicit map.** The author provides a `{ semantic_name: <SemanticMappingValue> }` mapping. Each entry's value is one of **three author-facing variants** (`Column` / `Literal` / `Expr`) — the full `SemanticMappingValue` enum is **4-variant** at the type level (`Column` / `Literal` / `Expr` / `Metadata`), with the 4th `Metadata(MetadataDimensionRecipe)` variant **compile-synthesized only** from the Dimension's own `type: { metadata: ... }` block (per `13 §4.7` / `18 §10.4` / `15 §5.5`). The 4th variant has no `semantic_mapping:` YAML form and is therefore not authored here. The full enum (shape + roster + `LiteralValue` grammar + `MetadataDimensionRecipe`) is ratified in `18 §10` and consumed by the `Binding` process in `15 §5`:

```yaml
extras:
  semantic_mapping:
    # Variant 1 — Column (bare string).
    revenue: net_revenue_cents
    country: region_lookup

    # Variant 2 — Literal broadcast.
    currency: { literal: "USD" }

    # Variant 3 — authored physical expression.
    hour_bucket:
      expr:
        trunc:
          column: event_ts
          unit: hour
```

Single-string values dispatch to `Column`; mapping values with `literal:` / `expr:` keys dispatch to `Literal` / `Expr`. The authored `expr:` block parses to an `UnresolvedPhysicalExpr` (the model authoring form `Expr<UnresolvedPhysicalLeaf>`; grammar at `14 §3`), which `Binding` (§2.3) resolves to `ir::PhysicalExpr` at compile. The `SemanticMappingValue` enum shape is owned by `18 §10`.

### 2.2 Leaf-only effective scope

`semantic_mapping` is effective only at leaves (`Dataset` / `NestedDataset`). Complex data kinds may carry `semantic_mapping` in their `extras` as a default for descendant leaves, but never as the complex kind's own mapping (complex kinds have no direct physical surface).

### 2.3 The `Binding` process

`Binding` is the internal compile-time process that consumes `semantic_mapping` plus the manifest-resolved `PhysicalSource` (schemas, partition metadata, catalog lookups) to produce executable physical expressions. It is a process, not a model-layer type.

- **Input:** `semantic_mapping` from the model; `PhysicalSource` from manifest-layer catalog resolution.
- **Output:** a resolved mapping of each Semantic name to an `ir::PhysicalExpr` tree, folded into the manifest's per-leaf resolved interface.
- **Stage:** runs at `compile` (per `10 §4`); never at parse or plan time.

See `15` for the full binding algorithm; `33` for where the binding output lives.

### 2.4 Aggregate synthesis from `(agg:, expr:)`

Measures and Metrics carry separate `agg:` and `expr:` fields in YAML. At **parse time**, `semstrait-model` wraps these into a single `Aggregate`-rooted `UnresolvedSemanticExpr`:

```text
agg: sum
expr:                 # block form (friendly-tag tree); the compound string
  multiply:           # "revenue * quantity" is the reserved inline DSL (deferred,
    - revenue         # parse-rejected per A-leaf / doc 45), so authors write the
    - quantity        # block form here
  ─→  UnresolvedSemanticExpr::Aggregate {
         op:       AggregationOp::Sum,
         args:     [parse_block(&expr_value)],  // field adapter; yields the multiply tree
         distinct: false,  // default; `distinct: true` is an explicit author field
         filter:   None,   // populated from `filter:` if present
       }
```

By the time `19 §3`'s `resolve` runs, the `UnresolvedSemanticExpr` already carries the `Aggregate` root if a Measure or Metric authored it. The resolution algorithm does not synthesize `Aggregate` — it only resolves the inner `args` tree.

This is a parse-time structural assembly, not a compile-time transformation. The dual-field `(agg:, expr:)` surface is author ergonomics; the canonical representation is always `Expr<L>::Aggregate`.

## Testing Strategy

- **SR-5** (leaf-only extras enforcement): authoring `catalog` / `storage` / `semantic_mapping` under a Complex variant's `extras:` block triggers `ParseErrorKind::UnknownField` at parse time (type-level + `deny_unknown_fields`) — verified by negative-parse test suite.
- **SR-E-7/SR-E-8** (temporal.grain placement): authoring `temporal.grain:` on a Complex variant triggers `ValidateError::TemporalGrainOnComplex`; a Grainset child missing an explicit grain triggers `ValidateError::GrainsetChildMissingGrain` — verified by validate-stage tests.
- **Semantic mapping variants**: parse correctly dispatches bare-string values to `Column`, `{ literal: ... }` to `Literal`, and `{ expr: ... }` to `Expr` — verified by round-trip tests.
- **Aggregate synthesis**: Measure/Metric `(agg:, expr:)` pairs round-trip through a single `Aggregate`-rooted `UnresolvedSemanticExpr` — verified by parse-stage tests that inspect the materialized expr tree.
- **StorageConfig validation**: at least one of `paths` / `tables` is non-empty (leaf-source rule) — verified by validate-stage test with empty `storage:` block (expect error).

Shared harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
