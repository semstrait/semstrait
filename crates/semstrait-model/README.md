# semstrait-model

Author surface for `semstrait` semantic models — owns the in-memory
`SemanticModel` tree, the parse/validate stages, the fluent
loader, and the per-struct fluent builders.

This file is current-state reference only. Decisions live in
[`docs/adr/`](../../docs/adr/); terminology lives in
[`CONTEXT.md`](../../CONTEXT.md). This crate's Rust types are the authoring
contract, and the JSON Schema for the YAML projection is generated from
these Rust types by [`src/schema.rs`](src/schema.rs) — there is no
committed reference schema.

---

## Surface

| Stage          | Entry point                              |
|----------------|-------------------------------------------|
| Parse YAML     | `parse` / `parse_with_source`             |
| Validate model | `validate`                                 |
| Parse catalogs | `parse_catalogs`                           |
| Fluent load    | `SemanticModel::loader()`                  |
| Fluent build   | `SemanticModel::builder()` / per-struct `::builder()` |

All stages are synchronous, and `parse`, `parse_catalogs`, and `validate` accumulate
rather than failing fast (ADR 0005). All are free of `stdout`/`stderr`
side effects — `tracing` is the only observability
channel, so no library code prints (ADR 0006). None of them branches on
which engine or provider is in use: that decision lives at the adapter
layer and nowhere below it. Diagnostics are typed and returned explicitly,
but the exact `Result` shapes differ per stage: `parse` returns a builder and
carries non-fatal parse warnings to `SemanticModelBuilder::build`, while
`validate` returns `Ok(warnings)` / `Err(errors + warnings)`.

---

## Construct from YAML

```rust
use semstrait_model::SemanticModel;

let (model, diagnostics) = SemanticModel::loader()
    .from_yaml_file("model.yaml")
    .from_catalogs_yaml_file("catalogs.yaml")
    .build()?;
```

Tests and tooling can swap the filesystem strategy:

```rust
use semstrait_model::{InMemoryFs, SemanticModel};

let mut fs = InMemoryFs::new();
fs.insert("model.yaml", REFERENCE_YAML);

let (model, _diagnostics) = SemanticModel::loader()
    .with_fs(fs)
    .from_yaml_file("model.yaml")
    .build()?;
```

The loader composes `parse` → `parse_catalogs` → `validate` and
folds every diagnostic into a single `ModelBuildErrorKind` accumulator.

---

## Construct from code

The builder API has two surfaces — both produce identical `SemanticModel`
values:

- **Ergonomic facade** — recommended for hand-authored models.
  Shorter call chains, cross-struct flatteners (`Dataset::builder("orders").catalog("polaris").path("s3://…")`),
  per-item inserters, additivity shortcuts. `agg:` is set with `.agg(agg)`, where
  `Agg::of_name("sum")` gives a catalog aggregate by its name or an alias, or `None`.
- **Primary structural surface** — every builder
  method name equals the Rust field name (`.dim_type(...)`,
  `.semantic_mapping(...)`). Used by round-trip helpers, schema generators,
  and code-gen tools; remains canonical.

### Facade form (recommended)

The `?`s below assume an enclosing function returning
`Result<_, semstrait_common::diagnostic::Diagnostics<ModelBuildErrorKind>>` — the error type both
`SemanticModel::builder()` and `SemanticModel::loader()` already batch into.
`Relationship::build` is the one fail-fast exception, returning a *single*
`semstrait_common::diagnostic::Diagnostic`, so it needs a one-element lift.

```rust
use semstrait_common::Granularity;
use semstrait_model::*;
use semstrait_sql::datatype::DataType;

let order_ts = Dimension::builder("order_ts")
    .data_type(DataType::Timestamp { precision: 6 })
    .temporal([Granularity::Minute, Granularity::Hour, Granularity::Day])
    .build();

let sum = Agg::of_name("sum").expect("a catalog aggregate");
let revenue = Measure::builder("revenue")
    .data_type(DataType::Decimal { precision: 18, scale: 2 })
    .agg(sum)
    .full()
    .build();

let orders = Dataset::builder("orders")
    .catalog("polaris_prod")
    .format(StorageFormat::Parquet)
    .path("s3://bucket/orders/")
    .semantic_mapping(
        SemanticMapping::builder().column("revenue", "amount_cents").build(),
    )
    .temporal(TemporalShape::events("order_ts", Some(Granularity::Minute)))
    .description("Order-line fact dataset.")
    .dimension(DimensionSite::r#ref("order_ts").with_key())
    .measure(MeasureSite::r#ref("revenue"))
    .build();

// `orders_to_customers` below names `customers` as an endpoint, so the model
// has to define it. The guard is `SemanticModel::builder().build()`, which
// re-runs `validate` and raises `RelationshipDanglingEndpoint` — not
// this infallible `Dataset` builder, and not `Relationship::build()`, which
// cannot see the model around it.
let customers = Dataset::builder("customers")
    .catalog("polaris_prod")
    .format(StorageFormat::Parquet)
    .path("s3://bucket/customers/")
    .build();

let orders_to_customers = Relationship::builder()
    .name("orders_to_customers")
    .from("orders")
    .to("customers")
    .on_key("customer_id")
    .many_to_one()
    .build()
    .map_err(|d| vec![d])?;

let (model, _diagnostics) = SemanticModel::builder()
    .name("analytics-v1")
    .dataset(orders)
    .dataset(customers)
    .dimension(order_ts)
    .measure(revenue)
    .relationship(orders_to_customers)
    .build()?;
```

#### Conflict semantics

- **Field-level last-write-wins.** When the same field is set more than
  once, the last call wins.
- **Sub-struct read-modify-write.** Cross-struct flatteners
  (`.catalog(...)`, `.dimension(...)`, etc.) read the current parent
  sub-struct, mutate the targeted sub-field, write back. So
  `.extras(my_extras).catalog("polaris")` preserves every other field of
  `my_extras` while setting `catalog`. Symmetrically for
  `.semantic_interface(my_iface).dimension(entry)`.

### Primary structural surface

Round-trip helpers and code-gen tools target the primary surface. Method
names equal Rust field names; helpers (`DimensionType::temporal(granularities)`,
`TemporalShape::events(...)`) are 1:1 with variant bodies.

```rust
use semstrait_common::Granularity;
use semstrait_sql::expr::Literal;

let order_ts = Dimension::builder("order_ts")
    .data_type(DataType::Timestamp { precision: 6 })
    .dim_type(DimensionType::temporal([Granularity::Minute, Granularity::Hour, Granularity::Day]))
    .build();

let extras = LeafExtras::builder()
    .semantic_mapping(
        SemanticMapping::builder()
            .column("revenue", "amount_cents")
            .with_semantic(
                "currency",
                SemanticMappingValue::Literal(Literal::String("USD".into())),
            )
            .build(),
    )
    .temporal(TemporalShape::events("order_ts", Some(Granularity::Minute)))
    .build();

let interface = SemanticInterface::builder()
    .dimensions(vec![DimensionSite::r#ref("order_ts").with_key()])
    .measures(vec![MeasureSite::r#ref("revenue")])
    .build();

let orders = Dataset::builder("orders")
    .extras(extras)
    .description("Order-line fact dataset.")
    .semantic_interface(interface)
    .build();
```

Every primary-surface builder method name equals the Rust field name, and
a variant-body constructor (such as `DimensionType::temporal`) takes
exactly that body's fields; the facade may rename and aggregate, but it
delegates to the primary surface rather than bypassing it.

---

## Type map

`semstrait_model` re-exports the public surface from the root
([`lib.rs`](src/lib.rs)). Types group along three axes:

### Root model

| Type | Notes |
|------|-------|
| `SemanticModel` | Root struct. Holds four data-kind `EntityBTreeMap`s and three shared-pool `EntityBTreeMap`s. No `catalogs:` field — that's a sibling file. |
| `AiContext` | LLM-facing hint surface. Optional on every declared entity — root model, Public DataKinds, Semantics, `DataKindFilter`, Relationships. Absent from `Nested*` forms, `Extras` and `ref:` sites. |

### Data kinds

The four data-kind variants come in two **forms** — Public (top-level)
and Nested (inside a complex parent). Public forms carry the full
`SemanticInterface`; nested forms are structurally-only.

| Public form | Nested form | Body |
|-------------|-------------|------|
| `Dataset`  | `NestedDataset`  | `DatasetBody`  |
| `Grainset` | `NestedGrainset` | `GrainsetBody` |
| `Unionset` | `NestedUnionset` | `UnionsetBody` |
| `Joinset`  | `NestedJoinset`  | `JoinsetBody`  |

Sealed traits (`DataKind`, `SimpleDataKind`, `ComplexDataKind`,
`PublicDataKind`, `NestedDataKind`) classify each concrete type
along structural and behavioral axes. View enums
(`AnyDataKindRef`, `PublicDataKindRef`, `NestedDataKindRef`,
`SimpleDataKindRef`, `ComplexDataKindRef`) provide unified
borrowing surfaces.

### Extras

Per-DataKind physical configuration. Two flavors:

| Type | Used by | Fields |
|------|---------|--------|
| `LeafExtras`    | `Dataset` / `NestedDataset` | `catalog`, `storage`, `semantic_mapping`, `temporal` |
| `ComplexExtras` | every other variant         | `temporal` only — a leaf-only field under a Complex variant's `extras:` is refused as `ParseErrorKind::UnknownField` |

Storage / mapping helpers: `StorageConfig`, `StorageFormat`,
`PartitionDef`, `CatalogRef`, `SemanticMapping`,
`SemanticMappingValue`. The `lit:` arm holds a tagged `Literal` or an
`UntypedLiteral`, both `semstrait-sql`'s, read by the one `lit:` grammar of an
expression's literal (ADR 0085, rule 8; ADR 0089, rule 8). The `{expr: …}` arm of
`SemanticMappingValue::Expr` carries an `UnresolvedPhysicalExpr`, `semstrait-sql`'s
`RowExpr<PhysicalLeaf>`: every node of the tree, a column, a literal, an
operation or a call, and an untyped literal as its leaf; a semantic tag is
refused there as `ParseError::TagNotAllowedAtSite`. The friendly authoring
projection runs in serde, not on a wrapper type.

### Semantic interface

`SemanticInterface` composes the per-DataKind authoring surface:

| Type | Variants |
|------|----------|
| `Dimension` / `DimensionSite` / `DimensionRef` | `DimensionType` = Temporal, Categorical, Binary, Geo, Bucketed, Metadata |
| `Measure` / `MeasureSite` / `MeasureRef` | `Agg`: a catalog aggregate by its name or an alias, or `count_distinct` (ADR 0089, rule 20); `AdditivityType` (Full, Semi via `SemiAdditivity`, Non) |
| `Metric` / `MetricSite` / `MetricRef` | Required `expr:` |
| `Keyness` | `is_key:` plus `keyset:` / `keysets:`, on a dimension site only (ADR 0039) |
| `DataKindFilter` | DataKind-scoped named filter; Measure/Metric carry one `filter:` predicate |

### Temporal shape

`TemporalShape` (struct) + `TemporalShapeKind` (enum):

| Variant | Body | When to use |
|---------|------|-------------|
| `Timeseries` | `TimeseriesBody` | periodic series, semi-additive |
| `Events`     | `EventsBody`     | independent occurrences |
| `Snapshot`   | `SnapshotBody`   | point-in-time facts |
| `Scd`        | `ScdBody` (+ `ScdType` Type1 / Type2) | slowly-changing dimensions |

### Relationships

`Relationship` (cross-public or per-`Joinset` internal).
Shape: `Cardinality`, `Integrity`, `Optional`, `CrossFilter`,
`JoinKeySpec`. The join basis is a name list, `auto`, or `cross` (ADR
0038); the `optional → join_type` derivation lives in `semstrait-manifest`,
since `JoinType` is a `semstrait-ir` type this crate no longer depends on.

---

## Structural-rule taxonomy

`parse` and `validate` together implement the model's structural rules.
Each rule is a variant of `ParseErrorKind` or `ValidateErrorKind`:

### Root-shape

| Rule | Stage    | Variant                                                  |
|------|----------|-----------------------------------------------------------|
| A field or block the declared shape does not allow — the root envelope, a Nested form's Public-only fields, same-variant self-nesting, or a leaf-only extras field under a Complex variant | parse    | `UnknownField` (type-enforced via struct shape + `deny_unknown_fields`) |
| A data-kind name, or a shared Dimension/Measure/Metric name, repeated       | validate | `DuplicateDataKindName`, `DuplicateSharedSemanticsName` (uniform `.build()` dedup) |
| An identifier that fails the naming grammar       | parse    | `InvalidIdentifier`                                      |
| A `${VAR}` substitution naming an unset environment variable       | parse    | `UnsetEnvVar`                                            |
| A Joinset or a `unique` Unionset with fewer than two children, or a Grainset or an `all` Unionset with none      | validate | `ComplexDataKindInsufficientChildren`                    |

### Entity-level

Not tabulated here. Each rule is a variant of `ParseErrorKind` or
`ValidateErrorKind`, stated by the stage that raises it and exercised by
that stage's tests (ADR 0044). The validate stage's checks run from the
`validation` module (`src/validation/mod.rs`).

The entity-level rules also fire from the builder:
`Relationship::builder()` runs the symmetric-completeness and
many-to-many cross-filter checks at `.build()` time — plus the two
declared-basis rules the wire form rejects at parse, since the builder can
spell an empty or repeating `on:` and a model that re-serialises must stay
parseable (ADR 0038). It restates the missing-`cardinality:` rule there
too: `cardinality` is `Option<Cardinality>`, so bon's typestate no longer
demands the setter and `.build()` is the only guard the code path has. And
`SemanticModel::builder().build()` re-runs the full `validate` pass
so YAML-loaded and code-built models share the same diagnostics.

On the same footing, `Keyness::keysets_named` and
`DimensionSite::with_keysets` are **fallible**: an empty name list and a
blank name are refused from Rust as parse refuses them from YAML, with
validate-stage kinds (ADR 0039). `Keyness::key()` and `with_key()` stay
infallible — neither names a keyset.

`SemanticsShadowRootPool` is the sole `Severity::Warning` variant —
emitted when a Public DataKind's inline declaration shadows a shared
root-pool entry of the same name.

---

## Catalogs

`catalogs.yaml` is a sibling file, never embedded.

| Type | Notes |
|------|-------|
| `CatalogsConfig`    | Top-level. `catalogs: BTreeMap<String, CatalogEntry>`. |
| `CatalogEntry`      | `type`, `name`, `url`, optional `realm` / `default_namespace`, `auth`. |
| `CatalogAuthMethod` | Internally-tagged enum (`oauth2` / `bearer` / `aws_secrets`). |
| `SecretKeyMapping`  | Secrets-Manager JSON-key override. |

`${VAR}` substitution runs ahead of YAML decoding for both model and
catalogs files.

---

## Reference materials

- [`schemas/reference.yaml`](schemas/reference.yaml) — every public
  authoring concept exemplified.
- [`schemas/catalogs_reference.yaml`](schemas/catalogs_reference.yaml)
  — every `CatalogAuthMethod` variant exemplified.
- [`src/schema.rs`](src/schema.rs) — generates the JSON Schema (draft
  2020-12) for authoring tools from the Rust types, for both the
  `semantic_model:` document and the sibling catalogs file.
- [`tests/schema_roundtrip.rs`](tests/schema_roundtrip.rs) — asserts
  reference YAML validates against the generated schema, parses, and
  validates clean.

---

## Diagnostic primitives

All diagnostic types live in `semstrait-common`:

```rust
use semstrait_common::diagnostic::{Diagnostic, Diagnostics, Severity, SourceId};
use semstrait_model::ParseErrorKind;

let result: Result<_, Diagnostics<ParseErrorKind>> = semstrait_model::parse(yaml);
```

Stage-specific kinds: `ParseErrorKind`, `ValidateErrorKind`,
`CatalogsParseErrorKind`, `ModelBuildErrorKind` (loader/builder
fused kind).

---

## Dependencies

- `semstrait-common` (with the `io` feature) — diagnostic primitives
  (`Diagnostic`, `Severity`, `SourceId`, …), the narrow `ValidateError`,
  the stage-agnostic `Tree` / `Visitor` / `Rewriter` family, imported from
  `semstrait-common` directly (ADR 0090, rule 6), the `EntityId` lane with its UUIDv5 mint, and the
  `Persistable` boundary [`repository`](src/repository.rs) implements.
- `semstrait-sql` — the canonical data types: `DataType`, `WrittenDataType` and
  `DataType::resolve_written`, which reads a type by its name or an alias in
  any case, as a string or an object, through `DocumentDialect::Canonical`'s
  declarations. Parse checks a semantics' `data_type:`, a `cast` target and a
  tagged literal's type with it, and compile resolves a semantics'
  `UnresolvedDataType` with it (ADR 0086, rule 7). Behind the `schemars`
  feature, `DataType::resolution_schema` gives `UnresolvedDataType`'s schema,
  and `any_case` the pattern of a function name in any case. Also the
  expression tree, with its literal, `NamedReference`, its two
  unresolved leaves (`SemanticLeaf`, `PhysicalLeaf`) and its calls; the
  operation declarations and the function registry, by whose `name` and
  `aliases` parse finds each operation's and each function's tag; and the call
  constructors, with which parse builds each call (ADR 0089, rules 4 and 5;
  ADR 0092, rule 1).
- `bytes` — `Bytes`, the payload that `Persistable::to_bytes` gives.
- `serde`, `serde_yaml` — YAML decode.
- `serde_json` — non-optional: the `Fingerprintable` preimage is JSON over the
  id-nulled model (ADR 0041), even though YAML is this crate's own wire codec.
- `regex-syntax` — the parser of DataFusion's pattern language, by which parse
  counts the groups of a literal `regexp_extract` pattern (ADR 0087, rule 12).
- `uuid` — the authoring-id mint. Pinned workspace-wide to `features = ["v5"]`,
  so a time-based generator is a compile error rather than a review catch
  (ADR 0051).
- `bon` — typestate builders.
- `indexmap` — author-order preservation in YAML intermediates.
- `thiserror` / `tracing` — internal plumbing.
- `schemars` — optional, off by default: JSON-Schema codegen for the authoring
  model ([`src/schema.rs`](src/schema.rs)).

**`semstrait-ir` is deliberately *not* a dependency.** This crate owns the
authoring vocabulary — the aliases of the two unresolved trees,
`UnresolvedSemanticExpr` and `UnresolvedPhysicalExpr`, with the builders
`col`, `field`, `dim`, `measure`, `metric`, `lit` and `physical_col`
([`expr`](src/expr/)), each of which, but `lit`, gives `None` for an empty
name — and `semstrait-ir` names none of it (ADR 0094, rule 9). It holds no expression type of its own (ADR 0089, rule 6): the
tree, its leaves and its calls are `semstrait-sql`'s, and every crate imports
them from `semstrait-sql` (ADR 0085, rule 3). A column is a `NamedReference`, a
name path (ADR 0085, rule 9) of one part, because the author writes no source
alias (rule 12). An empty name names nothing, so parse refuses one at `col:`,
`field:`, `dim:`, `measure:` and `metric:` with an error that names the field,
and a semantic mapping's bare column `""` with the advice that the empty string
is written `{lit: ""}`. Parse reads the tree's own tags (`lit`, `col`, `case`,
`cast`, …) before any function, finds each operation's and each function's tag
by its declaration's `name` or `aliases`, and reads a call's arguments as a
list, one argument, or a mapping that names parameters by their names, in any
order. It builds a mapping's arguments in the declaration's order, where a
skipped optional parameter that a later named parameter follows takes its
declared default (ADR 0089, rule 4). It reads a bare word at an enum or a
constant parameter as a string literal (ADR 0089, rule 7), and builds each call
with `Call::new`, which checks it against the declaration (ADR 0089, rule 4; ADR
0092, rule 1); a call whose arguments do not fit is
`ParseErrorKind::CallRefused`, and so is a mapping with a key that names no
parameter, or that leaves out a parameter that the call writes or names one
that it does not write, a `regexp_extract` group below 0, or above the
number of groups of a literal pattern (ADR 0087, rule 12), and a call that holds
a call that it cannot hold, as a window call that holds a window call, at any
depth, as in DataFusion; an undeclared tag is
`ParseErrorKind::UnknownTag`. The writer writes each call under its
declaration's `name`. Reading "IR" as the substrate
the authoring model is built on inverts the actual edge. The consequence shows
up above: `JoinType` is absent here, so the `optional → join_type` derivation
lives in `semstrait-manifest`, the only crate permitted to see both layers.

Authoring expression fields are `UnresolvedSemanticExpr` or
`UnresolvedPhysicalExpr` — no wrapper type, because the friendly projection
runs in serde — and row-level by type: a
`RowExpr` holds no aggregate call, so parse refuses an aggregate
function's tag in an expression body (`ParseErrorKind::AggregateInRowExpr`).
Each expression field attaches its tree with one attribute,
`#[serde(with = "Unresolved::<SemanticExpr>")]` or
`#[serde(with = "Unresolved::<PhysicalExpr>")]`, on a bare or an `Option`
field. `SemanticExpr` and `PhysicalExpr` name the expression, and each one
fixes its leaf (`SemanticLeaf`, `PhysicalLeaf`); a field whose type does not
match fails to compile. The adapter type `Unresolved<E>` in
[`model::serde`](src/model/serde/) gives the field's parse, its writer and its
schema, which refers the field to `UnresolvedSemanticExpr` or
`UnresolvedPhysicalExpr`. A semantic mapping's `expr:` arm reads and
describes its tree through `Unresolved::<PhysicalExpr>`. A semantics'
`data_type:` holds an `UnresolvedDataType`: it keeps the type as written,
writes the canonical name, and takes a `DataType` through `From`, so the
builders take a `DataType`; compile resolves it through `semstrait-sql` and
refuses one that is not valid. A top-level bare string resolves to a
default leaf at compile; a compound string is the reserved inline-DSL form,
deferred in v1 and parse-rejected (`ParseError::InlineExprUnsupported`).
