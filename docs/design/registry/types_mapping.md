---
id: types-mapping
status: ratified
version: 1.0
impl: implemented
---

# Types Mapping Catalog

> **Scope.** Authoritative mapping of every canonical `DataType` variant (ratified in `foundations/13_types_and_grain.md`) to native types in DataFusion, Spark, and DuckDB. This document is a **Living catalog**: entries may gain detail, annotations, or additional engine columns over time. It does NOT define new canonical variants — those live in `13`.

> **Status (2026-04-17):** Scaffold ratified against `13 §2.3`. Per-engine cast-semantics matrices and edge-case annotations are drafted but awaiting verification during adapter implementation (`34` / `36` / `37`). Entries marked 🟡 are plausible based on engine documentation but have not been empirically verified.

---

## 1. Canonical Type Mapping Matrix

Each row pins a canonical `DataType` variant to its native representation in each target engine. Multiple engine-native alternatives are separated by `/` in order of preference (the first form is what the adapter emits by default).

| Canonical | DataFusion (Arrow) | Spark | DuckDB | Notes |
|---|---|---|---|---|
| `Boolean` | `Boolean` | `BooleanType` / `BOOLEAN` | `BOOLEAN` | Direct map on every engine. |
| `Byte` | `Int8` | `ByteType` / `TINYINT` | `TINYINT` | 8-bit signed, range -128..127. |
| `Short` | `Int16` | `ShortType` / `SMALLINT` | `SMALLINT` | 16-bit signed. |
| `Integer` | `Int32` | `IntegerType` / `INT` / `INTEGER` | `INTEGER` / `INT4` | 32-bit signed. SQL-standard `INTEGER` is 32-bit across all three. |
| `Long` | `Int64` | `LongType` / `BIGINT` | `BIGINT` / `INT8` | 64-bit signed. |
| `Float` | `Float32` | `FloatType` / `FLOAT` | `REAL` / `FLOAT4` | 32-bit IEEE-754. DuckDB's `FLOAT` aliases `REAL`. |
| `Double` | `Float64` | `DoubleType` / `DOUBLE` | `DOUBLE` / `FLOAT8` | 64-bit IEEE-754. |
| `Decimal(p,s)` | `Decimal128(p,s)` (p≤38) / `Decimal256(p,s)` (p≤76) | `DecimalType(p,s)` (p≤38) | `DECIMAL(p,s)` / `NUMERIC(p,s)` (p≤38) | DataFusion is the only target supporting precision > 38 (via Decimal256). Canonical `Decimal` caps `precision` at 38 in v1; Decimal256 use is tracked as **TD-TYPE-DECIMAL256**. |
| `String` | `Utf8` / `LargeUtf8` | `StringType` / `STRING` / `VARCHAR` | `VARCHAR` / `TEXT` / `STRING` | No length constraint at the semantic layer. Adapters choose `Utf8` vs. `LargeUtf8` (DataFusion) based on catalog hints; elsewhere the native type is unparameterized. |
| `Binary` | `Binary` / `LargeBinary` | `BinaryType` / `BINARY` | `BLOB` / `BYTEA` / `VARBINARY` | Same adapter-choice policy as `String`. |
| `Date` | `Date32` (days since epoch) | `DateType` / `DATE` | `DATE` | DataFusion also supports `Date64` (ms); adapters prefer `Date32` to minimize footprint. |
| `Time(p)` | `Time32(Second\|Millisecond)` (p≤3) / `Time64(Microsecond\|Nanosecond)` (p≤9) | *(no native)* — emulated as `StringType` | `TIME` (microsecond precision only) | See §3.2 for Spark emulation strategy and DuckDB precision limit. |
| `Timestamp(p)` | `Timestamp(Second\|Millisecond\|Microsecond\|Nanosecond, None)` | `TimestampNTZType` (Spark 3.4+) | `TIMESTAMP` (microsecond precision) | All three targets are tz-naive in this mapping. See §3.3 for tz-aware physical columns. |
| `Interval` | `IntervalYearMonth` / `IntervalDayTime` / `IntervalMonthDayNano` | `CalendarIntervalType` / `INTERVAL` | `INTERVAL` | DataFusion splits into three Arrow variants; the adapter chooses based on interval composition at emit time. See §3.4. |

### 1.1 Legend

- **DataFusion** column shows **Arrow logical types** as seen in `arrow::datatypes::DataType`. When the SQL frontend spells it differently (e.g. `BIGINT`), that's an alias.
- **Spark** column shows the Scala/Java type name followed by the SQL keyword. Either is accepted by Spark SQL.
- **DuckDB** column shows the primary SQL keyword followed by recognized aliases.

---

## 2. Cast Semantics

How adapters reconcile a declared Semantics `data_type` with a physical column's type at `adapt` time. All rules apply uniformly across the three engines unless a row notes otherwise.

### 2.1 Integer widening (safe)

| Physical → Declared | Action | Emitted SQL (illustrative, DuckDB) |
|---|---|---|
| `Byte` → `Short` / `Integer` / `Long` | Silent widening cast | `CAST(col AS BIGINT)` |
| `Short` → `Integer` / `Long` | Silent widening cast | `CAST(col AS BIGINT)` |
| `Integer` → `Long` | Silent widening cast | `CAST(col AS BIGINT)` |

Widening is always safe (no value loss). Adapters emit the cast; planners do not need to be aware.

### 2.2 Integer narrowing (rejected)

Any physical-wider → declared-narrower (e.g. physical `Long` vs. declared `Integer`) is a compile-time error:

```
CompileError::PhysicalTypeNarrower {
    binding: "...",
    physical_type: "Long",
    declared_type: "Integer",
    hint: "Either widen the Semantics data_type, or introduce an explicit computed
           column expression that truncates deliberately.",
}
```

The adapter REFUSES silent narrowing — authors must make the truncation explicit via an `expr:` cast.

### 2.3 Float / Decimal widening

| Physical → Declared | Action | Notes |
|---|---|---|
| `Float` → `Double` | Silent widening cast | Safe. |
| `Decimal(p1,s1)` → `Decimal(p2,s2)` with `p2≥p1` AND `s2≥s1` AND `p2-s2≥p1-s1` | Silent widening cast | Both integer-part and fractional-part capacity must be ≥ physical. |
| `Integer` → `Decimal(p,s)` with `p-s ≥ 10` (for 32-bit) | Silent widening cast | Integer width must fit in integer part of decimal. |
| `Long` → `Decimal(p,s)` with `p-s ≥ 19` (for 64-bit) | Silent widening cast | |

### 2.4 Float → Integer (rejected)

Physical `Float`/`Double` → declared `Integer`/`Long` is a compile-time error (value semantics differ: floats may have fractional parts that silently truncate). Authors must cast explicitly in `expr:`.

### 2.5 Timestamp precision

| Physical `Timestamp(p1)` → Declared `Timestamp(p2)` | Action |
|---|---|
| `p2 ≥ p1` | Silent widening (extend with zero fractional units) |
| `p2 < p1` | Compile-time error `PhysicalTypeNarrower` — truncation must be explicit |

Same policy applies to `Time(p)`.

---

## 3. Per-Engine Notes

### 3.1 DataFusion (Arrow)

- **Primary target.** The mapping is 1:1 with Arrow's logical types; DataFusion is the first engine the semstrait reference adapter supports.
- **`Utf8` vs. `LargeUtf8`.** The adapter prefers `Utf8` (32-bit offsets, 2GiB-per-array limit) unless the Binding's catalog reports columns exceeding that threshold. `LargeUtf8` is chosen transparently. Same policy for `Binary` / `LargeBinary`.
- **`Date32` vs. `Date64`.** Always `Date32` (days-since-epoch, 32-bit). `Date64` (ms-since-epoch, 64-bit) is not emitted — it's a legacy Arrow variant.
- **`Decimal256`.** Supported in Arrow for precision > 38, but canonical `Decimal` caps at precision 38 in v1. When the Arrow source has a `Decimal256` column and the Semantics declares `Decimal(≤38, ≤38)`, the adapter emits an explicit precision-check cast. Tracked as **TD-TYPE-DECIMAL256** to lift the cap.
- **`Timestamp(unit, tz)`.** When a physical column has a non-`None` timezone, the adapter converts to UTC and emits a warning Diagnostic (`adapt.physical-timestamp-tz-normalized`). The canonical `Timestamp` remains tz-naive.

### 3.2 Spark

- **No native `Time` type.** `DataType::Time { precision }` on a Spark-backed Binding is emulated:
  - Physical storage: `StringType` with canonical encoding `HH:MM:SS[.fff...]` (ISO-8601 time-of-day, with `precision` fractional digits).
  - Ordering ops (`<`, `>`, `ORDER BY`): work correctly because the canonical encoding is lexicographically monotonic.
  - Arithmetic ops (`Time + Interval`): rewritten by the Spark adapter using date-arithmetic primitives (e.g. adding an interval is decomposed to `from_unixtime(unix_timestamp(CAST(... AS TIMESTAMP)) + seconds)` with appropriate parsing).
  - Tracked as **TD-ADAPTER-SPARK-TIME**. When a richer Spark type becomes available (TimeType proposal periodically surfaces in Spark issues), the emulation is replaced.
- **`TimestampType` vs. `TimestampNTZType`.** Spark's default `TimestampType` is tz-aware (stored UTC, displayed in session tz). Canonical `Timestamp` maps to `TimestampNTZType` (Spark 3.4+). Adapters targeting Spark < 3.4 MUST reject `Timestamp` in `adapt` with a supported-feature error.
- **`CalendarIntervalType` limitations.** Spark's interval type does not participate in all arithmetic operations (notably, `Date + CalendarInterval` requires `INTERVAL YEAR TO MONTH` or `INTERVAL DAY TO SECOND` SQL-type spellings, not the Scala primitive). The adapter emits SQL-typed interval literals in planner output. 🟡 verification pending.
- **Decimal precision.** Spark `DecimalType` caps at precision 38. This matches canonical `Decimal`'s v1 cap.

### 3.3 DuckDB

- **SQL-idiom reference.** When canonical naming and SQL keywords disagree (rare), DuckDB's spelling is the tiebreaker for canonical naming decisions.
- **`TIMESTAMP` precision.** DuckDB's `TIMESTAMP` is microsecond precision only (no configurable `p`). The adapter clamps precision to 6 on emit; declared `Timestamp(9)` → physical `TIMESTAMP` is a warning Diagnostic plus silent truncation-on-read. Tracked as **TD-ADAPTER-DUCKDB-TIMESTAMP-NS**.
- **`TIME` precision.** Same story: microsecond precision only. Declared `Time(9)` → physical `TIME` is a warning + truncation.
- **`TIMESTAMPTZ`.** DuckDB has a tz-aware `TIMESTAMPTZ`. Canonical is tz-naive; physical tz-aware columns are normalized to UTC by the adapter with a warning Diagnostic.
- **`DECIMAL` range.** Precision 1..=38 mirrors canonical.

### 3.4 Interval variance

The three engines differ significantly in how they model intervals:

| Engine | Native | Year-Month | Day-Time | Nanosecond |
|---|---|---|---|---|
| DataFusion | `IntervalYearMonth` (i32 months) | ✅ | ✗ | ✗ |
| DataFusion | `IntervalDayTime` (i32 days + i32 ms) | ✗ | ✅ (ms) | ✗ |
| DataFusion | `IntervalMonthDayNano` (i32 months + i32 days + i64 ns) | ✅ | ✅ | ✅ |
| Spark | `CalendarIntervalType` (months + days + microseconds) | ✅ | ✅ | ✗ |
| DuckDB | `INTERVAL` (months + days + microseconds) | ✅ | ✅ | ✗ |

**Adapter strategy.** The canonical `Interval` carries the union of all three components (year-month, day-time, sub-microsecond). At emit time:

- DataFusion: choose the narrowest Arrow variant that losslessly represents the components. If sub-microsecond is present, use `IntervalMonthDayNano`.
- Spark / DuckDB: if sub-microsecond components are present, adapter emits a warning Diagnostic and truncates to microsecond precision. Tracked as **TD-ADAPTER-SPARK-INTERVAL-NS** and **TD-ADAPTER-DUCKDB-INTERVAL-NS**.

**Month/day arithmetic ambiguity.** Adding an interval with a month component to a specific date near a month boundary produces different results across engines (e.g. Jan 31 + 1 month → Feb 28/29 vs. Mar 3). This is a **planner-level concern** ratified in `foundations/14_expressions.md`; the canonical `Expr` tree carries explicit enough interval semantics for adapters to emit consistent engine SQL.

---

## 4. TECH_DEBT Index

Every engine-shortfall in this catalog maps to a tracked TD entry. Resolving a TD means either extending the canonical set, adding an adapter feature, or accepting the limitation formally.

| TD ID | Description | Affects | Current posture |
|---|---|---|---|
| TD-TYPE-UNSIGNED-INT | Add `UByte` / `UShort` / `UInteger` / `ULong` canonical variants | DataFusion ✅, DuckDB ✅, Spark ✗ (requires adapter widening) | Deferred to keep canonical portable |
| TD-TYPE-DECIMAL256 | Lift `Decimal` precision cap from 38 to 76 | DataFusion ✅ (Decimal256), Spark ✗, DuckDB ✗ | Deferred; requires adapter feature gates |
| TD-TYPE-ARRAY | Add `Array<T>` canonical variant | All three engines support natively | Blocked on `14` (expression typing over arrays) |
| TD-TYPE-STRUCT | Add `Struct<{field: T, …}>` canonical variant | All three engines support natively | Blocked on `14` |
| TD-TYPE-MAP | Add `Map<K, V>` canonical variant | All three engines support natively | Blocked on `14` |
| TD-TYPE-JSON | Add `Json` canonical variant | DuckDB `JSON`, Spark Variant proposals, DataFusion JSON extensions | Deferred |
| TD-TYPE-UUID | Add `Uuid` canonical variant with string/binary bridging | DataFusion ✅ (Arrow extension), DuckDB ✅, Spark ✗ (emulate as String) | Deferred |
| TD-ADAPTER-SPARK-TIME | Spark lacks native Time; emulate via String | Spark only | Accepted emulation; documented in §3.2 |
| TD-ADAPTER-DUCKDB-TIMESTAMP-NS | DuckDB timestamps capped at microsecond precision | DuckDB only | Truncation-with-warning accepted |
| TD-ADAPTER-DUCKDB-TIME-NS | DuckDB Time capped at microsecond precision | DuckDB only | Truncation-with-warning accepted |
| TD-ADAPTER-SPARK-INTERVAL-NS | Spark intervals capped at microsecond precision | Spark only | Truncation-with-warning accepted |
| TD-ADAPTER-DUCKDB-INTERVAL-NS | DuckDB intervals capped at microsecond precision | DuckDB only | Truncation-with-warning accepted |

---

## 5. Interaction with Other Documents

- **`foundations/13_types_and_grain.md`** — this registry is the authoritative implementation of `13 §2.3`'s pointer. `13` defines the canonical set; this catalog maps it.
- **`foundations/15_mapping_and_binding.md`** (forward reference) — specifies when physical-column-to-semantic-type reconciliation runs (at `compile` time, consulting catalog metadata). The cast matrices in §2 of this document define *what* reconciliation does; `15` defines *when* and *where*.
- **`foundations/14_expressions.md`** (forward reference) — promotion lattice for cross-width arithmetic (`Integer + Long → Long`) is defined there. This document documents the per-engine expression of that promoted type.
- **`apis/36_semstrait_adapter.md`** (forward reference) — the `EngineAdapter` trait's `adapt` method consumes the mappings in §1 and the cast rules in §2. Per-engine adapter crates reference specific sections of this registry as their type-handling contract.

## Testing Strategy

Type-mapping behavioral gates:

- **Type-mapping round-trip**: For each canonical `DataType`, emit engine-native type string via §1 table, cast back to canonical form, verify identity. Covers all 14 canonical types × 3 engines = 42 round-trips.
- **Widening-cast matrix**: Integer widening (Byte→Short→Integer→Long), Float→Double; Decimal precision/scale widening; Timestamp precision widening. Per-engine validation that silent casts preserve value.
- **Narrowing-rejection**: Long→Integer, Decimal narrowing, Timestamp precision truncation MUST fail at compile time with `PhysicalTypeNarrower` error. Per-engine validation that adapters refuse silent lossy casts.
- **Decimal precision/scale validation**: §2.3 formula enforcement (p2≥p1 AND s2≥s1 AND p2-s2≥p1-s1).
- **Per-engine shortfall smoke tests**:
  - Spark Time→String emulation (§3.2): round-trip Time value through String encoding, verify ordering ops work.
  - DuckDB timestamp microsecond cap (§3.3): declared Timestamp(9) → physical TIMESTAMP, verify truncation warning + correct microsecond-level round-trip.
  - Decimal256 deferral (§3.1): attempt to declare Decimal(p>38), verify adapter emits precision-check cast.
- **TD entries ratified**: Each TD entry in §4 maps to a tracked issue in the adapter crate; validate the posture (deferred / accepted / blocked) matches the catalog.

See [../testing/50_behavioral_contract_gates.md](../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
