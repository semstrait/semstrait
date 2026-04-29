---
prereqs: [00, 10, 11, 12]
authoritative-for:
  - canonical `DataType` variant set (14 scalar types, width-differentiated numerics, engine-neutral naming) and YAML grammar
  - shape-unification rules for `data_type:` across Semantics occurrences
  - the authoritative engine type-mapping catalog lives in `registry/types_mapping.md` (pointer from §2.3)
  - `Grain` enum (7 temporal variants) and total coarseness order used by `12 §4.2`
  - `DimensionType` *planner-level and authoring-level semantics* — what each variant means and where it fits (enum roster + body struct shapes ratified in `18 §4.1`)
  - `Metadata` Dimension v1 authoring forms (`path.token:N` 0-indexed, `partition.level:N` 1-indexed) and their runtime-extraction forward-reference to `15 §8`
  - Key vs. Dimension separation of concerns (why Keys are NOT a `DimensionType`)
  - structural Preconditions run by `validate` and `compile` that concern types and grains
refined-by:
  - 14 (expressions — `ExprSource`, function signatures typed against `DataType`; computed Dimensions with derived `data_type`)
  - 15 (mapping and binding — physical-type reconciliation at the Binding layer; per-engine catalog type mapping)
  - 16 (composition — `Cardinality`, `JoinType`, cross-kind type unification)
  - 17 (temporal shape — `TemporalShape` gating of Grainset level eligibility and as-of semantics)
  - 19 (categories — `DimensionType` IS the Dimension category axis; per-variant implicit-constraint contract lives in `19 §2.2`; `13 §4` semantics consumed by `19`)
  - 20–25 (strategies — per-DataKind-variant grain walking, type-preservation invariants at each stage)
  - 34 / 36 / 37 (adapters, catalog — authoritative engine-specific type mapping implementations)
---

# 13. Types and Grain

> **Struct ownership (2026-04-17 consolidation).** The `DimensionType` enum roster and its payload structs (`TemporalDimensionBody`, `BucketedDimensionBody`, `BucketSpec`, `BucketBound`, `MetadataDimensionBody`) are ratified in [`18 §4.1`](./18_entities.md#4-1-dimensiontype-roster). This doc owns the *planner-level and authoring-level semantics* of each variant — what the role means, v1 vs. post-v1 shape boundary, and cross-references to `15 §8` (metadata runtime), `12 §4.4` (Grainset temporal-grain eligibility), and `17` (`TemporalShape` gating).
>
> **Status:** ratified. Canonical scalar `DataType` set, `Grain` total order, and the `DimensionType` semantics layer (per-variant roles) are all content-complete. Complex types (arrays, structs, maps, JSON/VARIANT) are explicitly out of scope for v1 (§2.5).

## 1. Purpose and Scope

`13` ratifies the vocabulary of data shape at the Semantics layer: what canonical types and grains a `SemanticModel` may author, how they map to the engines semstrait targets, and how they compose across occurrences of a Semantics name.

**What `13` ratifies:**

- The **canonical `DataType`** set (§2) — 14 scalar variants (with width-differentiated integers and floats) forming the engine-agnostic type lattice. Naming deliberately avoids any one engine's dialect: `Byte`/`Short`/`Integer`/`Long` for integers, `Float`/`Double` for IEEE-754 floats — widely-recognized primitive names drawn from SQL and mainstream programming-language conventions.
- The **YAML grammar** for `data_type:` authoring (§2.2) — accepts canonical names plus engine-specific aliases (`bigint`, `int64`, `long` all parse to `Long`).
- The **engine type-mapping catalog** — delegated to `docs/design/registry/types_mapping.md`. §2.3 carries a short summary and pointer; the registry is the authoritative catalog for DataFusion / Spark / DuckDB mappings, cast rules, and per-engine quirks. DataFusion is the primary implementation target; Spark and DuckDB calibrate the canonical set against broader analytical-engine conventions.
- The **shape-unification rules** for `data_type:` across Semantics occurrences (§2.4).
- `Grain` (§3) — 7 temporal variants with a total coarseness order used by `12 §4.2`.
- The **`DimensionType` discriminator** (§4) — *planner-level and authoring-level semantics* of each of the six variants (enum roster and body struct shapes ratified in `18 §4.1`).
- The **Metadata Dimension v1 authoring forms** (§4.7) — `path.token:N` and `partition.level:N`; runtime extraction is owned by `15 §8`.
- The **Key vs. Dimension** separation (§5) — explicit rationale against collapsing Keys into `DimensionType`.
- **Structural Preconditions** (§6) covering type and grain checks.

**What `13` does NOT ratify** (forward-refs):

- Expression typing rules, function signatures, and computed-Dimension type derivation — `14`.
- Physical-type reconciliation, per-column type mismatch policy, catalog-provided types — `15`.
- `Cardinality` axis on Relationships — `16`.
- `TemporalShape` classification, SCD subtypes, Grainset eligibility gating — `17`.
- Per-engine type-mapping implementation details — `34` / `36` / `37`.

**Key invariants from `00` / `11` / `12` that `13` directly upholds:**

- **I2** (`00 §8`) — physical-type mapping is adapter-owned. `13`'s canonical types never carry engine-specific type info.
- **I10** — `DataType`, `Grain`, and `DimensionType` enums are `#[non_exhaustive]`. Adding a variant is non-breaking.
- `11 §5.1` shape-unification — `13` defines `data_type:` equality (strict) and the `decimal` precision/scale treatment.
- `12 §4.2` coarsest-first enforcement — `13` provides the total order on `Grain`.

## 2. Canonical DataType Set

Fourteen scalar variants, named with **engine-neutral, widely-recognized primitive names** drawn from SQL and mainstream programming-language conventions. The naming deliberately avoids Arrow's dialect (`Int32`, `Float64`), DuckDB's pure-SQL dialect (`TINYINT`, `BIGINT`), and Spark's internal `*Type` suffixes, positioning the canonical set as a **neutral middle** that every adapter can map without the naming itself favoring one engine.

Engine-agnostic by construction; every adapter maps canonical variants to its engine's native types at `adapt` time (`10 §3.6`, `34` / `36` / `37`). The authoritative mapping catalog lives in `docs/design/registry/types_mapping.md`.

### 2.1 Variants

```rust
/// Canonical logical data types for the semantic model.
///
/// Names follow a neutral convention: primitive names common to Java / Scala /
/// Spark / Kotlin / C# (`Byte`, `Short`, `Integer`, `Long`, `Float`, `Double`)
/// and SQL keywords that cross every warehouse dialect (`Decimal`, `String`,
/// `Date`, `Time`, `Timestamp`, `Interval`, `Binary`, `Boolean`). The canonical
/// enum name never is engine-specific; adapters translate to native spellings.
///
/// Non-exhaustive per I10.
#[non_exhaustive]
pub enum DataType {
    /// Boolean (true/false).
    Boolean,

    // ------- integers (width-differentiated; signed only in v1) -------

    /// 8-bit signed integer (-128 ..= 127). SQL TINYINT / Spark ByteType.
    Byte,

    /// 16-bit signed integer. SQL SMALLINT / Spark ShortType.
    Short,

    /// 32-bit signed integer. SQL INTEGER / Spark IntegerType.
    Integer,

    /// 64-bit signed integer. SQL BIGINT / Spark LongType.
    Long,

    // ------- floats (width-differentiated IEEE-754) -------

    /// 32-bit IEEE-754 floating point. SQL REAL / Spark FloatType.
    Float,

    /// 64-bit IEEE-754 floating point. SQL DOUBLE PRECISION / Spark DoubleType.
    Double,

    // ------- fixed-precision numeric -------

    /// Fixed-precision decimal.
    /// Default when precision/scale omitted: (38, 9) — defensible max across
    /// DataFusion (Decimal128/256), Spark (DecimalType), and DuckDB (DECIMAL).
    Decimal { precision: u8, scale: i8 },

    // ------- strings / bytes -------

    /// Unicode text (arbitrary length at the semantic layer).
    String,

    /// Raw byte sequence.
    Binary,

    // ------- temporal -------

    /// Calendar date (no time component, no timezone).
    Date,

    /// Time-of-day without a date (precision in seconds-fractions: 0..=9).
    Time { precision: u8 },

    /// Timestamp without timezone. Precision is seconds-fractions (0..=9).
    /// Tz-aware timestamps are deliberately out of scope for v1; adapters
    /// reconcile tz at the Binding layer (I2).
    Timestamp { precision: u8 },

    /// Duration (year-month + day-time components).
    /// Engine representation varies; canonical semantic is "elapsed time".
    Interval,
}
```

**Design notes:**

- **Naming rationale.** Names are the neutral middle: `Integer` is both the SQL-standard keyword AND a primitive in Java/Scala; `Long` / `Short` / `Byte` are universal modern-language names for signed 64/16/8-bit integers and also match Spark's type names 1:1. `Float` / `Double` are IEEE-754 names understood across every ecosystem. None of these names are owned by a single engine — every target supports them either natively or as trivial aliases.
- **Width-differentiated integers.** DataFusion (Arrow's `Int8`/`Int16`/`Int32`/`Int64`), Spark (`ByteType`/`ShortType`/`IntegerType`/`LongType`), and DuckDB (`TINYINT`/`SMALLINT`/`INTEGER`/`BIGINT`) all distinguish the four widths natively. Modeling them as a single `Integer` would force adapters to guess at bind time whether a downstream operation overflows (`Integer × Integer` in 32-bit arithmetic can overflow; in `Long` it never does). The semantic layer surfaces the width explicitly.
- **Signed only in v1.** Unsigned integers exist in DataFusion and DuckDB (Arrow `UInt8..UInt64`, DuckDB `UTINYINT..UBIGINT`) but not in Spark. Tracked as **TD-TYPE-UNSIGNED-INT**; deferred to keep the canonical set engine-portable.
- **Width-differentiated floats.** `Float` (32-bit) and `Double` (64-bit) match Spark's `FloatType`/`DoubleType` 1:1, DataFusion's Arrow variants, and DuckDB's `REAL`/`DOUBLE`. Silently promoting 32-bit to 64-bit would obscure memory footprint and precision guarantees.
- **`Decimal { precision, scale }`** — precision `1..=38`, scale `0..=precision`. Out-of-range values fail at parse (`ParseError::InvalidDecimalParameters`). Default when written as bare `decimal`: `(38, 9)` (Round 1 ratified "max values as defaults").
- **`Time { precision }`** — `precision` is seconds-fractions `0..=9`. `0` = whole seconds; `3` = ms; `6` = µs; `9` = ns. Spark has no native Time type — the Spark adapter emulates via `String` encoding (see `registry/types_mapping.md`).
- **`Timestamp { precision }`** — tz-naive by convention. When a physical column is tz-aware, adapters emit a UTC-normalizing conversion and log a warning Diagnostic.
- **`Interval`** — canonical elapsed-time type. Engine representations (DataFusion's `IntervalYearMonth`/`IntervalDayTime`/`IntervalMonthDayNano`; Spark's `CalendarIntervalType`; DuckDB's `INTERVAL`) are reconciled per `34`.
- **No UUID / JSON / array / struct / map / unsigned-int.** Complex and unsigned types are explicitly deferred (§2.5).

### 2.2 YAML grammar

`data_type:` is a scalar string. The parser accepts the canonical name and a set of convenience aliases (for authoring ergonomics and migration from engine-native types).

| Canonical YAML form | Accepted aliases (case-insensitive) | Parses to |
|---|---|---|
| `boolean` | `bool` | `DataType::Boolean` |
| `byte` | `tinyint`, `int8`, `i8` | `DataType::Byte` |
| `short` | `smallint`, `int16`, `i16` | `DataType::Short` |
| `integer` | `int`, `int32`, `i32` | `DataType::Integer` |
| `long` | `bigint`, `int64`, `i64` | `DataType::Long` |
| `float` | `real`, `float32`, `f32` | `DataType::Float` |
| `double` | `float64`, `f64`, `number` (legacy warn) | `DataType::Double` |
| `decimal(p,s)` | `decimal` (→ `(38, 9)`), `numeric(p,s)`, `number(p,s)` | `DataType::Decimal { precision, scale }` |
| `string` | `text`, `varchar`, `char`, `utf8`, `large_utf8` | `DataType::String` |
| `binary` | `blob`, `bytes`, `varbinary` | `DataType::Binary` |
| `date` | `date32`, `date64` | `DataType::Date` |
| `time(p)` | `time` (→ `(6)`) | `DataType::Time { precision }` |
| `timestamp(p)` | `timestamp` (→ `(0)`), `timestamp_ms` (→ `(3)`), `timestamp_us` (→ `(6)`), `timestamp_ns` (→ `(9)`), `datetime` (→ `(6)`) | `DataType::Timestamp { precision }` |
| `interval` | (none) | `DataType::Interval` |

**Alias handling.** Aliases are parse-time syntactic sugar; the `SemanticModel` carries only canonical `DataType`s. A round-tripped `Manifest` emits canonical forms (`long`, not `bigint`; `integer`, not `int32`). Lints MAY warn on non-canonical aliases; they MUST NOT rewrite author YAML.

**Ambiguity-prone aliases.** Bare `int` → `Integer` (32-bit, SQL-standard width). Authors who want a 64-bit integer must write `long` or `bigint` or `int64` explicitly; no silent widening. `number` is accepted as a legacy alias for `Double` but emits a warning Diagnostic recommending migration to `double`. `float` alone — unlike in SQL where `FLOAT(p)` can mean anything — always canonicalizes to **32-bit** `Float`; use `double` for 64-bit.

**Invalid forms:** unknown strings (`i128`, `uint32`, `uuid`, `json`, `array<int>`, `struct<...>`) fail at parse with `ParseError::UnknownDataType { text, location }` plus a hint listing the canonical set.

### 2.3 Engine type-mapping — pointer

The **authoritative** engine type-mapping catalog lives in `docs/design/registry/types_mapping.md`. It ratifies the DataFusion / Spark / DuckDB mapping for every canonical variant, documents cast semantics, flags per-engine gaps (e.g. Spark's lack of native `Time`), and will extend as adapters add engines.

At-a-glance summary (abbreviated — see registry for full detail, cast rules, and gap notes):

| Canonical | DataFusion | Spark | DuckDB |
|---|---|---|---|
| `Boolean` | `Boolean` | `BooleanType` | `BOOLEAN` |
| `Byte` | `Int8` | `ByteType` | `TINYINT` |
| `Short` | `Int16` | `ShortType` | `SMALLINT` |
| `Integer` | `Int32` | `IntegerType` | `INTEGER` |
| `Long` | `Int64` | `LongType` | `BIGINT` |
| `Float` | `Float32` | `FloatType` | `REAL` |
| `Double` | `Float64` | `DoubleType` | `DOUBLE` |
| `Decimal(p,s)` | `Decimal128/256(p,s)` | `DecimalType(p,s)` | `DECIMAL(p,s)` |
| `String` | `Utf8` / `LargeUtf8` | `StringType` | `VARCHAR` |
| `Binary` | `Binary` / `LargeBinary` | `BinaryType` | `BLOB` |
| `Date` | `Date32` | `DateType` | `DATE` |
| `Time(p)` | `Time32` / `Time64` | *emulated as `String`* | `TIME` |
| `Timestamp(p)` | `Timestamp(unit, None)` | `TimestampNTZType` | `TIMESTAMP` |
| `Interval` | `IntervalYearMonth`/`DayTime`/`MonthDayNano` | `CalendarIntervalType` | `INTERVAL` |

Key policies (registry explains each in detail):

- **Adapter-owned reconciliation.** Safe widening casts are emitted silently; narrowing (e.g. physical `Long` vs. Semantics `Integer`) is a compile-time error, not a silent truncation.
- **Cross-width arithmetic.** `Integer + Long` → `Long` canonically (promotion lattice in `14`); adapters translate to engine-native promotion rules.
- **Tz on `Timestamp`.** Tz-naive canonical. Physical tz-aware columns are normalized to UTC at the adapter with a warning Diagnostic.
- **Spark `Time` gap.** Spark has no native Time type; the Spark adapter emulates via `String` with canonical `HH:MM:SS[.fff...]` encoding. Tracked as **TD-ADAPTER-SPARK-TIME**.
- **`Interval` variance.** DataFusion splits into three Arrow interval types, Spark has one, DuckDB has one; adapters pick the right engine representation from the interval's components at emit time.

### 2.4 Shape unification under `11 §5.1`

`data_type:` is a shape field. Across all occurrences of a Semantics name, `data_type:` must unify by strict equality. The rules:

- **Scalar variants without payload** (`Boolean`, `Byte`, `Short`, `Integer`, `Long`, `Float`, `Double`, `String`, `Binary`, `Date`, `Interval`) unify trivially when names match.
- **Width-differentiated numerics do NOT auto-widen.** Two occurrences stating `integer` and `long` are a `CompileError::SemanticShapeConflict` — the author explicitly chose different widths. If cross-width reconciliation is the intent, the narrower occurrence should be rewritten (or a distinct Semantics name introduced).
- **`Decimal { precision, scale }`** requires matching precision AND scale across occurrences. Two occurrences stating `decimal(10, 2)` and `decimal(12, 2)` conflict.
- **`Time { precision }` and `Timestamp { precision }`** require matching precision across occurrences.
- **Default-triggered forms** — an occurrence stating `data_type: decimal` (default-expanded to `(38, 9)`) and another stating `data_type: decimal(38, 9)` unify. An occurrence of bare `decimal` against `decimal(18, 2)` does NOT unify — the author explicitly chose a different precision.

**Implicit casting across occurrences is banned.** Aliases (`bigint` / `long` / `int64`) all normalize to `DataType::Long` and compare structurally post-normalization.

### 2.5 Complex and unsigned types — deferred

Arrays, structs, maps, JSON/VARIANT, UUID, ENUM, and unsigned integers are out of scope for v1. Authors who need complex types must represent them as `String` or `Binary` at the semantic layer and handle structure in `expr:` at plan time (with loss of semantic-level typing guarantees).

Tracked as TECH_DEBT:

- **TD-TYPE-UNSIGNED-INT** — `UInt8` / `UInt16` / `UInt32` / `UInt64` variants. DataFusion and DuckDB support natively; Spark does not. Will likely require adapter-level widening on Spark (e.g. `UInt32` → `LongType`).
- **TD-TYPE-ARRAY** — `Array<T>` variant. Needed for engines with first-class array support (all three targets).
- **TD-TYPE-STRUCT** — `Struct<{field: T, …}>` variant. Needed for nested JSON and row-types.
- **TD-TYPE-MAP** — `Map<K, V>` variant.
- **TD-TYPE-JSON** — `Json` variant. Covers DuckDB `JSON`, Spark `StringType`-with-schema, DataFusion JSON extensions.
- **TD-TYPE-UUID** — `Uuid` variant with canonical string/binary bridging.
- **TD-ADAPTER-SPARK-TIME** — Spark's lack of a native Time type requires adapter-side emulation.

Adding any variant is I10-non-breaking.

## 3. Grain

The canonical time-axis lattice.

### 3.1 Enum

```rust
/// Temporal Grain levels. Non-exhaustive per I10 — non-temporal axes
/// (geographic, entity) are a future extension.
#[non_exhaustive]
pub enum Grain {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Quarter,
    Year,
}
```

### 3.2 Total coarseness order

A single linear chain:

```
Minute (0) < Hour (1) < Day (2) < Week (3) < Month (4) < Quarter (5) < Year (6)
```

The rank is exposed as `Grain::coarseness() -> u8`. The total order is what `12 §4.2` uses to enforce `coarsest-first` `levels:` declaration in a Grainset.

### 3.3 Week and non-divisibility

`Week` sits between `Day` and `Month` in the coarseness order. Week does NOT divide Month evenly, and this is an intentional design choice:

- The order represents **selection rank**, not a divisibility relation. The planner uses the order to answer "is level A coarser than or equal to level B?"; it does not use the order to arithmetically convert between levels.
- Grain arithmetic (truncating a `Timestamp` to `Week`, then to `Month`) is handled at the expression layer (`14`'s `DateTrunc` canonical function) and produces engine-specific SQL (typically `DATE_TRUNC('week', ...)` → `DATE_TRUNC('month', ...)`). Results at adjacent grains are correct independently; they are not expected to compose arithmetically across `Week` / `Month` boundaries.
- **Authoring rule:** a Grainset that rolls up from `Day` through `Week` and then to `Month` is authoring-legal but the planner MUST treat each level as an independent materialization. `20–25`'s Grainset strategy ratifies this rule in full; `13` provides the order and the note.

A Grainset that mixes `Week` and `Month` in its `levels:` should represent genuinely independent weekly and monthly aggregations (e.g. a physical table for each). Attempting to derive monthly numbers from weekly numbers at plan time is out of scope.

### 3.4 Non-temporal grain axes — reserved

`00 §4.1` anticipates future non-temporal grain axes (geographic rollup: city → region → country; entity rollup: SKU → product → category). These are explicitly deferred:

- Tracked as **TD-GRAIN-NON-TEMPORAL** — add non-temporal `Grain` variants behind a feature flag, or introduce a parallel `GeoGrain` / `EntityGrain` enum. Design open: single extended enum vs. parallel enums.
- When ratified, the coarseness comparison becomes partial (temporal grains only comparable to temporal; geo only comparable to geo). Grainset `levels:` would need to declare a grain axis alongside the value, or allow mixed axes with stricter compatibility rules.

Adding non-temporal grains is I10-non-breaking.

## 4. DimensionType Discriminator

The `DimensionType` enum roster and per-variant body structs (`TemporalDimensionBody`, `BucketedDimensionBody`, `BucketSpec`, `BucketBound`, `MetadataDimensionBody`) are ratified in [`18 §4.1`](./18_entities.md#41-dimensiontype-roster). Six variants — `Temporal(TemporalDimensionBody)`, `Categorical`, `Binary`, `Geo`, `Bucketed(BucketedDimensionBody)`, `Metadata(MetadataDimensionBody)` — with the payload-bearing variants (`Temporal` / `Bucketed` / `Metadata`) carrying the fields the planner uses; `Categorical` / `Binary` / `Geo` are payload-free in v1 per the sub-shape-polish posture (see `18 §4.1`'s "Sub-shape polish" note for post-v1 extensions like `CategoricalBody::enum_values` / `GeoBody::{lat,lon}`).

> **`DimensionType` IS the Dimension category axis.** Per [`19 §2`](./19_categories.md#2-dimension-categories--dimensiontype), the implicit-constraint contract per variant (planner / adapter behavior, validation locks like `data_type:`-vs-variant) is ratified in `19`. This section (`13 §4`) owns the **planner-level and authoring-level semantics** of each variant; the **category-axis cross-cuts** (e.g. each variant's effect on planner routing, applicability per DataKind variant in `25`) live in `19 §2.2`. The two layers are non-overlapping: shape lives in `18 §4.1`; semantics live here; category-axis contract lives in `19 §2`.

This section owns the **planner-level and authoring-level semantics** of each variant — what the role means, where each type fits in the resolution pipeline, what the v1 v. post-v1 boundary is.

### 4.1 Default and `type:` authoring

Default when `type:` is omitted: `Categorical`. Authoring YAML is shown in `18 §4.2`.

The `type:` block is a single-key map — exactly one discriminator name with its payload (or bare name for unit variants). Zero or multiple keys = `ParseError::DimensionTypeMalformed`.

### 4.2 Temporal — planner semantics

- `TemporalDimensionBody.grains` (see `18 §4.1`) declares which `Grain` levels the source data supports. When a DataKind is a Grainset (`12 §4.4`), `levels[].grain` must be a subset of this list.
- When empty, the Dimension is temporal only in data type (a `date` column not intended for rollup). Temporal-with-empty-`grains` is legal but disables grain-based rollup for this Dimension. Planner treats it as "temporal for filter/grouping, not for rollup."
- Interaction with `TemporalShape` (17): a DataKind's `TemporalShape` may further constrain which grains are **planner-eligible** (e.g. `Snapshot` fixes a source grain). That gating lives in `17`; `13` specifies only the per-Dimension authoring shape.

### 4.3 Categorical (default) — planner semantics

- Unit variant in v1. The planner treats the Dimension as an open-ended categorical grouping/filter axis.
- Post-v1: a `CategoricalBody { enum_values: Option<Vec<String>> }` extension is tracked as a sub-shape polish (`18 §4.1`). When present, `enum_values` would give the planner a cardinality hint (Joinset cost estimation, forward to `20–25`) and let the validator check Filter values against the enum at `validate` time. Shape-unification of `enum_values` would follow `11 §5.1` (shape-locked across occurrences).

### 4.4 Binary — planner semantics

- Unit variant in v1. `Binary` at the Dimension layer is the **axis role** (two-valued grouping axis), not to be confused with `DataType::Binary` (raw bytes). The Dimension's `data_type:` is usually `Boolean` or `String`; `type: binary` annotates the intent.
- Useful when a two-valued string column (`active_flag: 'Y' | 'N'`) should be treated as a boolean axis at the semantic layer — the adapter translates to `CASE WHEN active_flag = 'Y' THEN TRUE ELSE FALSE END` in the projection.
- Post-v1: a `BinaryBody { binary_type: BinaryType }` extension (`{Boolean, Bit, String}`) would pin the underlying encoding and is tracked as a sub-shape polish.

### 4.5 Geo — planner semantics

- Unit variant in v1. Declares a geo-typed Dimension; the `data_type:` on the Dimension itself is typically `String` or `Double`.
- Geo Dimensions do NOT have a non-temporal `Grain` rollup mechanism in v1. Geographic grain (city → region → country) belongs to the deferred TD-GRAIN-NON-TEMPORAL.
- Post-v1: a `GeoBody { lat: String, lon: String }` extension carrying physical column references for two-column geo points is tracked as a sub-shape polish; consuming geo functions (ratified in `14`) would read those fields.

### 4.6 Bucketed — planner semantics

- `BucketedDimensionBody.buckets` (see `18 §4.1`) is a `Vec<BucketSpec>` with `BucketSpec { name, lower: Option<BucketBound>, upper: Option<BucketBound> }` and `BucketBound::{Int, Float, Decimal, Date, Timestamp}` covers the v1 bound types.
- Semantics: `lower` is inclusive (`>=`), `upper` is exclusive (`<`); `None` on either side means open-ended. The planner emits `CASE WHEN <col> [>= lower] [AND <col> < upper] THEN '<name>' ... END` in the projection. The physical column targeted is resolved via `SemanticMapping` (15) rather than being authored on the Dimension body in v1.
- `buckets:` must be non-empty and non-overlapping (checked at `validate`: `ValidateError::BucketsOverlap`). Gaps between buckets are permitted (values in gaps become `NULL`).
- `data_type:` on the Dimension is typically `String` (the bucket label).

### 4.7 Metadata — planner semantics

- `MetadataDimensionBody { source: MetadataSource }` (see `18 §4.1`; `MetadataSource` full grammar is a sub-shape-polish item). The authoring-time forms the v1 surface must cover are **path-segment extraction** and **Hive-style partition-value extraction**; the runtime extraction mechanic is owned by `15 §8`.
- **Path-segment extraction (`path.token: N`)** — tokenizes the source path on `/` and returns the 0-indexed segment. Example: path `s3://bucket/month=01/data.parquet` with `token: 2` returns `"month=01"` (raw, no key=value parsing). `15 §8.1` owns the runtime rules and error codes.
- **Partition-value extraction (`partition.level: N`)** — extracts the value from a Hive-style `key=value` partition at 1-indexed level `N`. Example: partition `year=2024/month=01` with `level: 1` returns `"2024"`. `15 §8.2` owns the runtime rules and error codes.
- Exactly one of `path:` / `partition:` must be present in the authored `MetadataSource`. Both present or both absent = `ValidateError::MetadataDimensionMalformed`.
- `11 §6.1.1`'s field catalog entry for `metadata:` references `18 §4.1`'s `MetadataDimensionBody` and this section for the per-variant semantics.
- `data_type:` is typically `String` (extracted values are text); authors may declare `Integer` / `Long` / `Date` with implicit casting at query time per `15 §9`.

## 5. Keys and Dimensions — Separation of Concerns

In Round 1 framing the question "should Keys be a `DimensionType` variant?" was raised. The answer is **no**, for reasons that are worth recording.

### 5.1 Why Keys are NOT a `DimensionType`

- **Different roles.** A Dimension is a *grouping/filtering axis* — something a Request names in `group_by:` or `filter:`. A Key is a *row-identity assertion at the DataKind's grain* — a structural claim about uniqueness, used by the planner for Joinset cardinality inference, Unionset dedup policy, and Grainset rollup target selection (`20–25`). These purposes rarely overlap; collapsing them conflates two orthogonal concerns.

- **Composite nature.** Keys are frequently composite: a `primary` Key on `(customer_id, order_date)` identifies rows by a tuple. Making Key a `DimensionType` would force either (a) treating each Key as a single-Dimension specialization — which loses composition — or (b) introducing a "meta-Dimension" that wraps multiple Dimensions, which is strictly worse than the current clear separation in `11 §6.5` ("Keys are arrangements of Dimensions").

- **Structural vs. semantic surface.** Keys do not appear in expressions; no `expr:` references a Key by name. The Request never names a Key directly. Keys are consumed by the planner internally. A `DimensionType` variant for Key would introduce a name that authors cannot use anywhere except the Key declaration itself — a design smell.

- **SQL correspondence.** In SQL, primary keys and foreign keys are **table-level constraints**, not column types. The current design (Keys as top-level arrangements of Dimensions, per `11 §6.5`) mirrors this conventional structure. Collapsing into `DimensionType` would create a semstrait-specific pattern that engineers versed in SQL would have to un-learn.

### 5.2 The common-resolution path already exists

The framing question's motivation — "common resolution path for Keys and Dimensions" — is already satisfied by `11 §6.5` and `11 §12.2 N-C7`/`N-C8`:

- A Key's `members:` list is a list of Dimension **names**. Name resolution for Key members goes through the exact same lookup algorithm (`11 §11.1`) as Dimension references in expressions.
- `N-C7` verifies every Key member names a Dimension (not a Measure / Metric / Filter).
- `N-C8` verifies every Key member is in the declaring DataKind's interface.

In plain terms: Keys reference Dimensions by name. The resolution path for that name is shared. No DimensionType variant needed.

### 5.3 Keys with specialized semantics — where lint-style flags live

Key-participation information is **not** a `DimensionType` variant (§5.1) and **not** a v1-realized Constraint. `11 §8` frames Constraints as per-carrier + per-kind with reserved carriers (Dimension, Filter, Key, DataKind) and v1-realized carriers (Measure, Metric) only. Dimension-carrier `constraints:` (`11 §8.5.1`) is reserved; `11 §8.5.3` reserves Key-carrier `member_policy:` as a candidate Key-level kind. Lint-style warnings ("this Dimension is a Key member of DataKind X") are future-design extensions to those reserved carriers.

Meanwhile: current tooling SHOULD derive Key participation by reverse-looking-up the DataKind's Key declarations (`11 §6.5`: "dimension_X participates in which Keys?"). No author-facing field is required in v1.

## 6. Structural Preconditions

Type- and grain-related checks. These complement `11 §12` (name-based) and `12 §7` (nesting-shape) Preconditions.

### 6.1 Run by `validate` (structural, accumulate)

| ID | Rule | What fails |
|---|---|---|
| TG-V1 | `data_type:` parses to a canonical variant (§2.1) | unknown alias, invalid decimal/timestamp precision, malformed complex-type attempt |
| TG-V2 | `DimensionType` discriminator single-key (§4.1) | `type:` block with zero keys or multiple keys |
| TG-V3 | Temporal Dimension has valid `grains:` (§4.2) | `grains:` list contains a non-`Grain` value |
| TG-V4 | Metadata Dimension has exactly one extractor (§4.7) | both `path:` and `partition:` present; neither present |
| TG-V5 | Bucketed Dimension buckets non-overlapping (§4.6) | two buckets cover an overlapping range |
| TG-V6 | Grainset `levels[].grain` values are valid `Grain` variants | non-`Grain` value |

### 6.2 Run by `compile` (registry-dependent, fail fast)

| ID | Rule | What fails |
|---|---|---|
| TG-C1 | `data_type:` unifies across occurrences (§2.4) | shape conflict per `11 §5.1` on `data_type` field |
| TG-C2 | Grainset level grain subset of temporal Dimension grains (§4.2) | a `levels[].grain` is not listed in the Grainset's temporal Dimension's `grains:` |
| TG-C3 | `DimensionType` unifies across occurrences | two occurrences declare different `type:` discriminators for the same Semantics name |
| TG-C4 | `TemporalDimension.grains` unifies across occurrences | grain lists disagree (subset not sufficient; exact equality required, per `11 §5.1`) |

### 6.3 Mapping to typed error variants

| Precondition | Stage | Typed variant (ratified in `31` / `32` / `33`) |
|---|---|---|
| TG-V1 | parse | `ParseError::UnknownDataType { text, location }` or `ParseError::InvalidDecimalParameters { precision, scale, location }` |
| TG-V2 | validate | `ValidateError::DimensionTypeMalformed { dimension, reason }` |
| TG-V3 | validate | `ValidateError::InvalidGrainValue { dimension, value }` |
| TG-V4 | validate | `ValidateError::MetadataDimensionMalformed { dimension, reason }` |
| TG-V5 | validate | `ValidateError::BucketsOverlap { dimension, bucket_a, bucket_b }` |
| TG-V6 | validate | `ValidateError::InvalidGrainValue { context: "grainset-level", value }` |
| TG-C1 | compile | `CompileError::SemanticShapeConflict { name, field: "data_type", occurrences }` (shared with `11` N-C1) |
| TG-C2 | compile | `CompileError::GrainAxisMismatch { dataset, level_grain, available_grains }` (shared with `12` NC-C1) |
| TG-C3 | compile | `CompileError::SemanticShapeConflict { name, field: "type-discriminator", occurrences }` |
| TG-C4 | compile | `CompileError::SemanticShapeConflict { name, field: "temporal-grains", occurrences }` |

## 7. Interaction with Other Docs

- **11** — `13` supplies `DataType` variants for `data_type:` shape unification (`11 §5.1`) and `DimensionType` payloads for the Dimension element (`11 §6.1.1`). `11 §6.1.1`'s `metadata:` field cites `13 §4.7` as the authoritative shape for the `MetadataDimension { path?, partition? }` structure.
- **12** — `13 §3.2`'s total Grain order is the comparator used by `12 §4.2` (coarsest-first enforcement). `12 §4.4`'s grain-axis check is the shared `CompileError::GrainAxisMismatch`.
- **14** — defines `ExprSource` / `Expr`, function signatures typed against `DataType` variants, and computed-Dimension `data_type` derivation rules (the `data_type: Option<DataType>` case in existing code). `13` provides the variant set; `14` specifies the typing rules.
- **15** — defines physical-column-to-semantic-type reconciliation at the Binding layer. Precision-narrower checks, tz-aware-to-tz-naive conversions, and catalog-provided type overrides all live in `15`.
- **16** — `Cardinality` inference leans on Keys (§5). Joinset `on.left` / `on.right` column compatibility follows semstrait's general rule (`14 §5.6`): **no cross-operand type validation at the semstrait layer** — the engine enforces join-key compatibility at execution time per its native semantics. `13` contributes only the `DataType` variant set; it does not drive a canonical operand-compatibility check.
- **17** — `TemporalShape` gates which Grainset levels are planner-eligible. `13` allows all grains declared in the temporal Dimension's `grains:` list; `17` may forbid some at plan time (e.g. `Snapshot` has a fixed source grain).
- **20–25** — per-DataKind-variant strategies consume `13`'s types and grains. Grainset level selection uses `Grain::coarseness()`; Joinset cardinality inference uses Keys (§5); expression typing in Filter/Metric evaluation uses `DataType` unification.
- **`registry/types_mapping.md`** — authoritative catalog of canonical `DataType` ↔ DataFusion / Spark / DuckDB native types, cast semantics, and per-engine gaps. `13 §2.3` is a short pointer; the registry carries the full matrices and TECH_DEBT index.
- **34 / 36 / 37** — adapter crates consume `registry/types_mapping.md` as their type-handling contract. Adapter authors MUST support every canonical `DataType` variant (per policy); unsupported variants at `adapt` time are `AdaptError::UnsupportedType { variant, engine }`.
