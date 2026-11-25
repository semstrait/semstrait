---
number: 86
title: A data type takes its SQL name, and has a canonical declaration and a dialect declaration where the dialect differs
status: accepted
date: 2026-10-08
source:
- CONTEXT.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0022-engine-independent-canonical-vocabulary.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md
- docs/adr/0092-an-operation-node-has-a-declaration-and-checks-its-operands-by-type-family-and-a-type-system-derives-the-result-types.md
- '`DataType` in `crates/semstrait-sql/src/datatype/mod.rs`; `TypeFamily` in `crates/semstrait-sql/src/datatype/family.rs`; `ParameterType` in `crates/semstrait-sql/src/functions/mod.rs`; `native_to_canonical` in `crates/semstrait-manifest/src/compile/catalog.rs`'
- '`crates/semstrait-sql/src/datatype/declaration.rs` and `crates/semstrait-sql/src/datatype/type_system.rs`'
- the research on DataFusion 55's function declarations and on how query engines structure an expression
- '`datafusion-substrait@55.0.0`: `src/logical_plan/consumer/types.rs:78-312`, `consumer/utils.rs:291-356`, `:551-564`, `consumer/expr/cast.rs:26-50`; `datafusion-common@55.0.0`: `src/dfschema.rs:112-120`, `:668-828`, `src/config.rs:858`; `arrow-cast@59.3.0:src/cast/mod.rs:115-347`; `datafusion-sql@55.0.0`: `src/planner.rs:680-696`, `:709-915`, `:832`, `src/unparser/expr.rs:1859-1908`'
- '`datafusion-physical-expr@55.0.0:src/expressions/cast.rs:37-44`; `arrow-cast@59.3.0:src/cast/decimal.rs:351-410`, `:770-819` (a float to a decimal rounds), `:883-895` (a decimal to an integer is cut), `src/cast/mod.rs:2550-2572` (`num_cast`: a float to an integer is cut); PostgreSQL 18 documentation, *Value Storage* (`typeconv-query.html`)'
- '`substrait@0.63.0`: `substrait/proto/substrait/type.proto:17-63`, `:143-158`, `:191-203`, and `substrait/extensions/*.yaml`; Substrait v0.63.0 `site/docs/types/type_classes.md`'
- 'Calcite 1.42.0: `core/src/main/java/org/apache/calcite/sql/type/SqlTypeName.java:55-144`, `RelDataTypeSystemImpl.java:99-213` and `site/_docs/reference.md:1226-1253`; the PostgreSQL documentation of its numeric, character and binary types'
- the human's rulings of 2026-10-08, 2026-10-12 and 2026-10-13
- 'a run of 19 Substrait plans on DataFusion 55 (2026-10-13), and the bare timestamp''s precision; the Spark SQL, PostgreSQL and Snowflake documentation of their numeric types; `datafusion-sql@55.0.0:src/planner.rs:717`, `:743-744`'
- the human's rulings (2026-10-17)
- 'the research (2026-10-19): the Iceberg spec at tag `apache-iceberg-1.12.0` (`format/spec.md:263-291`, `:1925`); Trino 484 `plugin/trino-iceberg/src/main/java/io/trino/plugin/iceberg/TypeConverter.java`; Iceberg 1.10.2 `spark/v3.5/spark/src/main/java/org/apache/iceberg/spark/TypeToSparkType.java`; runs of Substrait plans on DataFusion 55.2 over Parquet files written as Iceberg writes them, and `datafusion-common@55.2.0:src/dfschema.rs:822-826`'
- sharpened in place (an interval's whole seconds fit in 32 bits)
- sharpened in place
- sharpened in place (a model may write a type as an object that names each value)
links:
- target: 22
  kind: amends
- target: 68
  kind: amends
- target: 61
  kind: amends
---

# A data type takes its SQL name, and has a canonical declaration and a dialect declaration where the dialect differs

## Context and Problem Statement

`DataType` moves to
`semstrait-sql`'s `datatype` package (ADR 0085, rule 3). A function has one canonical
declaration and, under it, a declaration for each dialect where the engine differs
(ADR 0091, rule 3). A data type follows the same
pattern. Five questions follow: the canonical roster, which gives `Literal` its
variants (ADR 0085, rule 4); the types a dialect declaration names; how a data type
names its type family and the types it widens to, which an operation checks (ADR
0092, rule 2); how a function's parameter types relate to the type families; and which
casts are legal.

Today a data type has no declaration. `DataType` has 14 variants and the compile-only
`Unknown`. Its parts sit in four places:

| Part | Where it is today |
|---|---|
| its spellings | `DataType::parse_scalar`, which accepts about 45 spellings, among them `int64`, `i64`, `bigint`, `utf8` and `timestamp_ms`; `Display` writes the canonical name |
| its precision and scale | `precision: u8` and `scale: i8`, with no range checked. A bare `decimal` is `(38, 9)`, a bare `time` is precision 6, and a bare `timestamp` is precision 0 |
| its classes | `TypeClass::contains` in `semstrait-ir`, and a second family set in `ParamType` that disagrees with it |
| its widening order | `is_widening` in manifest compile |

`Interval` has no stated meaning. `regexp_match` is canonical in
DataFusion's shape, and that function returns an array of strings, which none of the 14
types can name.

Five of today's names are not SQL's. ISO SQL names the 16-bit integer type
`SMALLINT`, the 64-bit integer type `BIGINT` and the 32-bit floating-point type
`REAL`. Calcite names the 8-bit integer type `TINYINT`, and ISO SQL and Calcite name
a list `ARRAY`. DataFusion 55's SQL reads
these names, and it refuses `list<int>` (`planner.rs:709-915`). A bare `float` is 32
bits in DataFusion, Spark, DuckDB and MySQL, and 64 bits in PostgreSQL, Snowflake and
Calcite. Substrait also declares scalar types that the roster lacks: `varchar<L>`,
`fixedchar<L>`, `fixedbinary<L>`, `uuid`, `interval_year`, `interval_day<P>` and a
timestamp with a time zone (`type.proto:17-63`, `type_classes.md`).

Semstrait gives DataFusion a Substrait plan. The plan holds Substrait's own types, and
DataFusion's consumer turns each one into an Arrow type when it reads the plan
(`consumer/types.rs:78-312`). The consumer differs from the canonical types in these
places. It takes a time or a timestamp only at precision 0, 3, 6 or 9
(`consumer/utils.rs:551-564`). It drops the length of a `varchar<L>` and a
`fixedchar<L>`, it refuses a `uuid`, and it reads an `interval_day` as
`Interval(DayTime)`, which holds milliseconds (`consumer/types.rs:242-245`, `:310-312`).
And it reads a time at precision 6 or 9 correctly only at Substrait variation 1
(`consumer/types.rs:140-149`).

A catalog reports a column's type in its own vocabulary. An Iceberg catalog writes
Iceberg's type names (`spec.md:263-285`): `int`, `long` and `float`; `time`,
`timestamp` and `timestamptz`, bare, whose precision the Iceberg spec fixes at
microseconds; `timestamp_ns` and `timestamptz_ns`, at nanoseconds, from Iceberg's
format version 3 (`spec.md:1925`); and `fixed[16]`, whose length is in square
brackets. A `timestamptz` value is "a point in time: values are stored as UTC and do
not retain a source time zone" (`spec.md:290`).

## Considered Options

- **A canonical declaration and a declaration for each dialect where the dialect
  differs, each a private const of `DataType`** (chosen).
- **A struct for each variant, which carries its declarations through a trait**, as
  `DataType::Timestamp(Timestamp)` with `impl Declared for Timestamp { const
  DECLARATIONS: .. }`. In this form, as in the chosen one, only the match in
  `declarations()` ties a variant to its declarations. The trait cannot be a trait
  object, no semstrait code holds a type statically, and every pattern and constructor
  of `DataType` changes.
- **A declaration for each `Shape` (a type without its precision), in a static beside
  `DataType`.** The human: "what I'm not sure about - shape"; `Shape` "was unclear", as
  it repeats `DataType` without its precision. A static beside the enum "is not
  canonected anyhow with pub enum DataType".
- **`DataType` with methods only, and no declaration.** A data type would then have
  another form than a function. The human: "not everything as methods - but What
  needed for the scope and DataTypes specifically".
- **No mark for a type that a dialect does not have**, because Substrait has every
  canonical type. `Dialect` is `#[non_exhaustive]`, and a later dialect with no
  `interval` would then change what `on` returns, and every reader with it.
- **No dialect name for a type.** The human: "we still might need per dialect spelling,
  for example for adapter work".
- **A type's Substrait form only as the engine rules' format detail**, as "a type's
  encoding" (ADR 0068, *The seam*). The human: "we add substrait as dialect for typing
  - which is ok".
- **A DataFusion declaration for each new type where its consumer differs**: `None` for
  a `varchar(n)`, a `char(n)` and a `uuid`, and an `interval_day_second` only at
  precision 0 to 3. The human: "it feels like we don't need types for datafusion,
  because plan accepts substrait types, and datafusion converts from them. so canonical
  only".
- **DataFusion's declarations for the older types only**: `bytea` as its name for
  `binary`, and a time and a timestamp only at precision 0, 3, 6 or 9. The human: "what
  is valid for datafusion - is typing system from substrait".
- **An Arrow type as each type's DataFusion equivalent.** The plan holds Substrait
  types, and DataFusion's consumer chooses the Arrow type. The human: "this info is not
  supplied into plan, right ? so just canonical shape is mostly enough in such
  scenario".
- **Today's names as the canonical names**, `byte`, `short`, `long`, `float` and
  `list<T>`. They are not SQL's, and a bare `float` is 32 bits in Spark, Iceberg and
  DataFusion and 64 bits in PostgreSQL and Snowflake. The four scalar names are
  aliases (rule 7).
- **`varchar` for `string` and `varbinary` for `binary`**, Calcite's names. Substrait's
  `varchar<L>` is "a unicode string of at most L characters", and Calcite's `VARCHAR`
  and `VARBINARY` have a maximum length. `string` and `binary` have none.
- **No new type now**, because DataFusion drops a length and refuses a `uuid`. The
  human named the types that enter: "varchar<L>, char<L>, binary<L>, uuid,
  internal_year, interval_day and etc. so yes".
- **A timestamp with a time zone, and unsigned integers.** ADR 0022 keeps timestamps
  zone-free, and ADR 0072 says that a source's timestamp holds a UTC instant. Substrait
  and ISO SQL have no unsigned integer type. Iceberg's `timestamptz` holds a UTC
  instant, and ADR 0072 reads every timestamp that a source gives as a UTC instant, so
  a type with a zone would change no plan.
- **Substrait's names for the new intervals, `interval_year` and `interval_day(p)`.**
  Substrait's `interval_year` holds a number of months, which its name does not say.
- **`fixedchar(n)` and `fixedbinary(n)` as model names.** The human: "maybe it can be
  fixed_* like name internally, but model name is simpler".
- **An array type that only the DataFusion dialect declares**, as a separate
  `EType::List`. ADR 0091 refused "A function with no canonical declaration", and a data
  type, too, is canonical first.
- **A roster with no array.** `regexp_match` could then not be canonical in
  DataFusion's shape.
- **An array only as a call's argument or result.** A semantics could then not declare
  an array, and a column could not have one (rule 4).
- **`Unknown` leaves `DataType`.** The human: "we can use to , which can be used to spot
  errors. So it makes sense to preserve it to lightweight port of existing logic in
  manifest or parse."
- **Every spelling that `parse_scalar` accepts today as an alias.** `int8`, a bare
  `number`, a bare `varchar` and a bare `char` name another type (rule 7).
- **Aliases only in `native_to_canonical`**, for a catalog. A model would then write
  only the canonical names, and the corpus would change 86 lines.
- **For a catalog's native type, the canonical names and aliases first, then Iceberg's
  dialect for typing, which the Iceberg REST provider names** (chosen).
- **Iceberg's spellings in the Iceberg catalog provider**, as the connectors of Trino
  (`TypeConverter.java`), Spark (Iceberg's `TypeToSparkType.java`) and DuckDB hold
  theirs, in code. `semstrait-catalog` would then depend on `semstrait-sql`, and its
  rule that a native type is "never a canonical enum" would be reversed.
- **Iceberg's spellings in a static table beside `DataType`**, from each spelling to its
  type. A type's names would then sit apart from its declarations.
- **Iceberg's spellings as aliases only.** An alias changes only the name (rule 1), so
  `time`, `timestamp`, `timestamptz` and the `_ns` pair would map to no type.
- **A catalog's bare spelling at the canonical default**, so that Iceberg's `time` is
  `time(9)`. The Iceberg spec fixes microseconds, and DataFusion 55 refuses a read of a
  `precision_time(9)` column over the file's `Time64(Microsecond)`.
- **Iceberg's format version 1 types only**, without `timestamp_ns` and
  `timestamptz_ns`. They need no type and no rule beyond the `_ns` mark.
- **Iceberg's `fixed[L]` as a `binary`**, with no length, as Trino, Spark and DuckDB
  read it. DataFusion 55 refuses a `binary` read over the file's `FixedSizeBinary(L)`
  column.
- **Iceberg's `uuid` as `binary(16)`**, as its Parquet form is. The type would lose its
  meaning, and how DataFusion reads a `uuid` is its rules' (rule 6).
- **Iceberg's `list` as `array<T>` now.** Its JSON form needs a reader, and an element of
  a `struct` or a `map` type would still map to no type.
- **A read of Iceberg's `timestamptz` as Substrait's `precision_timestamp_tz(6)`.**
  DataFusion 55 gives the same rows either way, and the plan would hold a type that no
  canonical type has.
- **`TypeClass` beside the type families.** The human: "It feels like something I have
  described as family".
- **Two interval types in place of `interval`**, year-month and day-time, as ANSI SQL
  and Spark have. One `interval` holds every interval with no loss, and DataFusion
  holds it as `Interval(MonthDayNano)`.
- **One interval family**, in which `interval_year_month` and `interval_day_second`
  widen to `interval`. A comparison of `1 month` with `30 days` then passes the check,
  and the engine decides what it means.
- **A bare `timestamp` at precision 0 (today) or 9.** Precision 0 holds no fraction of
  a second. At 9, a 64-bit count of nanoseconds holds only the years 1677 to 2262
  (rule 2). Either value gives a bare type another precision than a bare literal tag
  (ADR 0085, rule 5).
- **Any cast is legal**, and DataFusion refuses an illegal one when it builds the plan
  that runs (today). The author then sees the refusal from the engine, not from
  semstrait.
- **DataFusion's own cast table** as the canonical one. It takes in casts whose meaning
  is DataFusion's own, such as a date to its count of days since 1970.
- For the canonical CAST's rounding:
  - **DataFusion's, for now** (chosen), as for a function:
    "Where a function cannot be unified for now, DataFusion's shape is the canonical
    shape".
  - **Round on every cast to an exact type**, as PostgreSQL does (`2.7` gives `3`), and
    the engine adapter adds a rounding on DataFusion before a cast to an integer.
  - **No canonical rounding**: each engine's CAST gives its own result, and `2.7` cast
    to an `integer` gives `2` on DataFusion and `3` on PostgreSQL.

## Decision Outcome

In short:

- A data type takes SQL's name for it, where SQL has a name with the same meaning.
- A data type has a canonical declaration, and a dialect declaration for each dialect
  that differs in the type's name or in the values it takes, or, for a dialect whose
  spellings compile reads, where its bare spelling takes a value that the dialect
  fixes. Each is a private const
  of `DataType`. No `Shape` exists. A dialect marks a type that it does not have.
- The roster is 21 types. `Unknown` stays compile-only.
- Substrait and Iceberg are dialects for typing: Substrait's declarations give
  Substrait's name for a type, and Iceberg's give Iceberg's name and the values it
  takes. DataFusion declares no data type.
- A model writes a type's canonical name or one of its aliases, as a string or as an
  object that names each value. A catalog's native type is read through the same
  names, then through the dialect that its provider names, and the manifest stores
  the canonical name.
- A data type names its type family and the types it widens to. A function's parameter
  may name a type family, as an operation's operand does, and `TypeClass` retires.
- A cast is legal where the cast table says so, and the canonical CAST rounds as
  DataFusion's does, for now.

### A data type has a canonical declaration and a declaration per dialect

1. `semstrait-sql`'s `datatype` package holds `DataType` and its declarations. They
   have a function's form (ADR 0091, rule 3): a canonical declaration, then one
   declaration of the same struct for each dialect where the dialect differs.

   ```rust
   pub enum DataType {
       Boolean, TinyInt, SmallInt, Integer, BigInt, Real, Double,
       Decimal { precision: u8, scale: i8 },
       String, VarChar { length: u32 }, FixedChar { length: u32 },
       Binary, FixedBinary { length: u32 },
       Date, Time { precision: u8 }, Timestamp { precision: u8 },
       Interval, IntervalYearMonth, IntervalDaySecond { precision: u8 },
       Uuid,
       Array(Box<DataType>),
       Unknown,
   }

   pub struct TypeDeclarations {
       pub canonical: TypeDeclaration,
       pub dialects: &'static [(Dialect, Option<TypeDeclaration>)], // None: the dialect has no such type
   }

   pub struct TypeDeclaration {
       pub name: &'static str,       // on a dialect: the dialect's own name
       pub aliases: &'static [&'static str], // rule 7; none on a dialect
       pub precision: Option<Limit>, // rule 2
       pub scale: Option<Limit>,     // rule 2
       pub length: Option<Limit>,    // rule 2
       pub digits: Option<u8>,       // rule 2: an integer's fixed precision, which no spelling writes
   }

   pub struct Limit { pub values: Values, pub default: Option<u32> } // default: a bare spelling's value; None: a spelling must give it
   pub enum Values { Range(u32, u32), OneOf(&'static [u32]) }

   impl DataType {
       const TIMESTAMP: TypeDeclarations = TypeDeclarations {
           canonical: TypeDeclaration {
               name: "timestamp",
               aliases: &["datetime"],
               precision: Some(Limit { values: OneOf(&[0, 3, 6, 9]), default: Some(6) }),
               scale: None,
               length: None,
               digits: None,
           },
           dialects: &[
               (Dialect::Substrait, Some(TypeDeclaration {
                   name: "precision_timestamp",
                   aliases: &[],
                   precision: Some(Limit { values: OneOf(&[0, 3, 6, 9]), default: Some(6) }),
                   scale: None,
                   length: None,
                   digits: None,
               })),
               (Dialect::Iceberg, Some(TypeDeclaration {
                   name: "timestamp",
                   aliases: &[],
                   precision: Some(Limit { values: OneOf(&[6, 9]), default: Some(6) }),
                   scale: None,
                   length: None,
                   digits: None,
               })),
           ],
       };
       // … one private const for each type

       pub fn declarations(&self) -> Option<&'static TypeDeclarations> {
           Some(match self {
               Self::Timestamp { .. } => &Self::TIMESTAMP,
               // … one arm for each type
               Self::Unknown => return None,
           })
       }

       pub const fn decimal() -> Self { Self::Decimal { precision: 38, scale: 9 } }
       pub const fn time() -> Self { Self::Time { precision: 9 } }
       pub const fn timestamp() -> Self { Self::Timestamp { precision: 6 } }
       pub const fn interval_day_second() -> Self { Self::IntervalDaySecond { precision: 9 } }
       pub fn is_valid(&self) -> bool;                      // rule 2
       pub fn is_valid_on(&self, dialect: Dialect) -> bool; // rule 6
       pub fn family(&self) -> Option<TypeFamily>;          // rule 9
       pub fn widens_to(&self, to: &DataType) -> bool;      // rule 9
       pub fn casts_to(&self, to: &DataType) -> bool;       // rule 10
   }

   impl TypeDeclarations {
       /// The dialect's declaration, the canonical one where the dialect has none,
       /// or `None` where the dialect has no such type.
       pub fn on(&'static self, dialect: Dialect) -> Option<&'static TypeDeclaration>;
   }
   ```

   - Each declaration is a private const of `DataType`. Code reaches a declaration only
     from a `DataType` value, through `declarations()`, so no declaration stands apart
     from its type. The human: "I think we just need one", and "so Type Declaration not
     only data holder, but part of DataTYpe itself".
   - `declarations()` has one arm for each variant, so a variant with no declaration
     does not compile.
   - `on(dialect)` gives that dialect's declaration, the canonical declaration where
     the dialect has none, or `None` where the dialect has no such type. Substrait has
     every canonical type (rule 6), so none of its entries is `None`. The mark exists
     so that a dialect that lacks a type does not change what `on` returns. Iceberg is
     the first dialect that marks one: it has no `tinyint`, for example (rule 6).
   - `aliases` are the other spellings of the type, which a model writes and
     `native_to_canonical` reads for a catalog (rule 7). An alias takes what its
     declaration takes, with no other change, as a function's alias does (ADR 0091,
     rule 4): `numeric` takes `decimal`'s precision and scale, and `varbinary`, an alias
     of the `binary` with no length, takes no length. A dialect declaration has none:
     its name is the dialect's own.
   - A constructor gives the type of the bare spelling, at its declaration's `default`.
     Where a signature needs a type at any precision, it names the type by its
     constructor (rule 8). A type whose length has no `default` has no bare spelling
     and no constructor.
   - A type's family, its widening order and its casts are methods of `DataType`, not
     parts of a declaration (rules 9 and 10).

2. A type's precision, scale and length take these values:

   | Type | Values | A bare spelling's value |
   |---|---|---|
   | `decimal` precision | 1 to 38 | 38 |
   | `decimal` scale | 0 to the precision | 9 |
   | `time` precision | 0, 3, 6 or 9 | 9 |
   | `timestamp` precision | 0, 3, 6 or 9 | 6 |
   | `interval_day_second` precision | 0 to 9 | 9 |
   | `varchar`, `char` and `binary` length | 1 to 2,147,483,647, Substrait's range | none: the spelling gives a length |

   `is_valid()` checks each value against the canonical declaration, and a scale
   against its precision. A `time` and a `timestamp` take only the four units of an
   Arrow timestamp: seconds, milliseconds, microseconds and nanoseconds. DataFusion's
   consumer takes only these (`consumer/utils.rs:551-564`).

   A bare `time` and a bare `interval_day_second` get the largest precision, 9, by the
   human's words: "we can assume default constructor for DatType with max values for
   each logical type". A bare `timestamp` gets 6, which holds about 292,000 years
   either side of 1970. At 9, a 64-bit count of nanoseconds holds only the years 1677
   to 2262: DataFusion 55 fails a query that casts a `9999-12-31` in microseconds to
   it ("converted value exceeds the representable i64 range"). A literal with no
   precision gets its precision by ADR 0085 rule 5.

   An integer type has a fixed precision, the decimal digits that hold each of its
   values: `tinyint` 3, `smallint` 5, `integer` 10 and `bigint` 19, as Calcite gives
   them (`RelDataTypeSystemImpl.java:131-138`). Substrait's `i8` to `i64` hold none.
   No spelling writes it, as ISO SQL writes no `integer(10)`, so it is the
   declaration's `digits`, not a `precision` that `is_valid()` checks, and `bigint(19)`
   stays refused. A rule that takes an integer as a decimal reads it there (ADR 0092,
   rule 3). The human: "we can define based off substrait or calcite logic".

3. These types hold these values:

   | Type | It holds | As Substrait's |
   |---|---|---|
   | `interval` | a number of months, a number of days and a number of nanoseconds, whose whole seconds, counted toward zero, fit in 32 bits, the width of Substrait's `IntervalDayToSecond.seconds` | `interval_compound` |
   | `interval_year_month` | a number of months | `interval_year` |
   | `interval_day_second(p)` | a number of days and a number of seconds, with p digits after the point, whose whole seconds, counted toward zero, fit in 32 bits, the width of Substrait's `IntervalDayToSecond.seconds` | `interval_day` |
   | `varchar(n)` | a string of at most n characters | `varchar` |
   | `char(n)` | a string of n characters, padded with spaces | `fixedchar` |
   | `binary(n)` | n bytes | `fixedbinary` |
   | `uuid` | a 128-bit identifier | `uuid` |

   A `string` and a `binary` take no length, as Substrait's do. The human: "we can
   bring more sql types into vocabulary - to be closer in types for what declares
   substrait and what datafusion consumer converts to arrow types ( smallint and etc.
   )".

4. An `array` holds values of one canonical data type, its element. An array is a type
   like any other: a semantics may declare it, and a column may have it. A model writes
   it as `array<string>` or as `{array: {element: string}}` (rule 7). An array literal
   is `Literal::Array { element: DataType, values: Vec<Literal> }` (ADR 0085, rule 4).

5. `Unknown` stays a compile-only variant, with no declaration and no family.
   `declarations()` gives `None` for it, `family()` gives `None` for it and for an array
   of it, and `is_valid()` gives `false` for both. Serde refuses it in both directions,
   so the manifest and the plan never hold it (ADR 0085, rule 7).

### A dialect declaration gives the dialect's name for a type and the values it takes

6. A dialect has a declaration for a type where the dialect differs from the canonical
   declaration, in the type's name or in the values it takes, or, for a dialect whose
   spellings compile reads, as Iceberg's, where its bare spelling takes a value that the
   dialect fixes. Its `name`
   is the
   dialect's own name for the type, as on a function's dialect declaration (ADR 0091,
   rule 5). `is_valid_on(dialect)` is `is_valid()`, narrowed by that dialect's
   declaration, and `false` where the dialect has no such type. The human: "we still
   might need per dialect spelling, for example for adapter work".

   Substrait is a dialect for typing: a vocabulary that several engines implement,
   each with its own extensions (ADR 0091, rule 6). The human: "substrait is dialect as
   well, which can be implemented by different engines with extensions". Its
   declarations give Substrait's own name for a type, as the type classes and the
   extension files of `substrait@0.63.0` write it. Substrait takes every canonical
   value: `decimal<P,S>` with P up to 38, a length from 1 to 2,147,483,647, and a
   precision from 0 to 9 (`type.proto:147-157`, `:191-203`). So each of its
   declarations gives a name only:

   | Type | Substrait's `name` |
   |---|---|
   | `tinyint`, `smallint`, `integer`, `bigint` | `i8`, `i16`, `i32`, `i64` |
   | `real`, `double` | `fp32`, `fp64` |
   | `char(n)`, `binary(n)` | `fixedchar`, `fixedbinary` |
   | `time(p)`, `timestamp(p)` | `precision_time`, `precision_timestamp` |
   | `interval` | `interval_compound` |
   | `interval_year_month`, `interval_day_second(p)` | `interval_year`, `interval_day` |
   | `array<T>` | `list` |

   Every other type (`boolean`, `decimal(p,s)`, `string`, `varchar(n)`, `binary`,
   `date` and `uuid`) has no Substrait declaration: Substrait names it as the canonical
   declaration does. The human: "we add substrait as dialect for typing - which is ok,
   same could be done for functions if will be resonable. adapter can use that to
   declare re-write or adapter specific behavior. at the end adapter still will be used
   to serialize this rust types into proto".

   - The format adapter writes each type's proto from its `DataType` variant. A rule
     and a rewrite read a type's Substrait name from its declaration. An `interval` is
     written as `interval_compound` at precision 9, because it holds nanoseconds
     (rule 3).
   - A Substrait type variation stays the DataFusion rules' own format detail (ADR 0068,
     *The seam*, as amended below). The Arrow spelling that `arrow_cast` takes is not a
     format detail of a type, because no `DataType` holds a zone (ADR 0022, as amended
     below). The DataFusion rules write it inside the whole call that ADR 0072's hook
     returns (ADR 0087, rule 22).

   Iceberg is a dialect for typing too: the vocabulary of the Iceberg table format, in
   which an Iceberg catalog reports a column's type (`spec.md:263-285`). Its
   declarations give Iceberg's name for a type and the values it takes, and compile
   reads an Iceberg catalog's native type through them where the canonical names and
   aliases give it no type (rule 7).

   | Type | Iceberg's `name` | Iceberg's values |
   |---|---|---|
   | `integer`, `bigint`, `real` | `int`, `long`, `float` | the canonical ones |
   | `time(p)` | `time` | precision 6, which its bare spelling gives |
   | `timestamp(p)` | `timestamp` | precision 6, which its bare spelling gives, or 9 |
   | `binary(n)` | `fixed` | the canonical lengths |
   | `array<T>` | `list` | the canonical ones |
   | `tinyint`, `smallint`, `varchar(n)`, `char(n)`, `interval`, `interval_year_month`, `interval_day_second(p)` | none: Iceberg has no such type | — |

   Every other type (`boolean`, `double`, `decimal(p,s)`, `string`, `binary`, `date`
   and `uuid`) has no Iceberg declaration: Iceberg names it as the canonical declaration
   does. Iceberg holds a time in microseconds, and a timestamp in microseconds or, from
   its format version 3, in nanoseconds (`spec.md:263-285`, `:1925`). So Iceberg's
   `time` and `timestamp` give their own bare value, 6. The canonical bare `time` is
   `time(9)`, and DataFusion 55 refuses a read of a `precision_time(9)` column over the
   file's `Time64(Microsecond)`.

   - Iceberg's `int`, `long` and `float` are also aliases (rule 7), so
     `native_to_canonical` reads them as aliases first. Their Iceberg declarations give
     `on(Iceberg)` Iceberg's names.
   - Iceberg's declarations hold no function and no operation, as Substrait's hold none,
     so a function's `on(Iceberg)` gives its canonical declaration.

   DataFusion declares no data type: its dialect declares functions and operations
   only. The plan holds Substrait's types, and DataFusion's consumer turns each one
   into its own, so DataFusion's typing is Substrait's. The human: "it feels like we
   don't need types for datafusion, because plan accepts substrait types, and
   datafusion converts from them. so canonical only", and "what is valid for
   datafusion - is typing system from substrait". So `on(DataFusion)` gives the
   canonical declaration for every type, and `is_valid_on(DataFusion)` is
   `is_valid()`. Where DataFusion 55 refuses a value or drops a part of it, no
   declaration records it, and the DataFusion rules answer it. Each row is a Substrait
   plan run on DataFusion 55:

   | Value | Canonical | DataFusion 55 |
   |---|---|---|
   | `CAST('abcdef' AS varchar(3))` | `'abc'` | `'abcdef'`: its consumer drops the length (`consumer/types.rs:310`, `consumer/expr/cast.rs:26-50`) |
   | `CAST('ab' AS char(5))` | `'ab   '` | `'ab'`: its consumer drops the length (`consumer/types.rs:311`) |
   | a `uuid` | the value | its consumer refuses the plan (`consumer/types.rs:312`); over a column that a file holds as `FixedSizeBinary(16)`, as Iceberg's Parquet form holds a `uuid`, a `string` read is refused too, and a `fixedbinary` read of length 16 runs (`datafusion-common@55.2.0:src/dfschema.rs:826`) |
   | `CAST(created_at AS string)`, where `created_at` is a `timestamp(6)` column whose file type carries a zone: an Iceberg `timestamptz`, or a Parquet timestamp with `isAdjustedToUTC` | the text that a zone-free column gives | `2026-01-31T23:30:00Z`, where a zone-free column gives `2026-01-31T23:30:00`: the read keeps the file's zone (`dfschema.rs:822-825`) |
   | a `binary` column that DataFusion reads from Parquet | the bytes | the read is refused: DataFusion reads the column as a binary view (`datafusion-common@55.2.0:src/config.rs:1246`), which a `binary` read does not match |
   | `CAST(s AS binary(16))`, and a `binary(16)` cast to `string` | the bytes; the text | each fails when it runs: arrow-cast has no cast between `Utf8` and `FixedSizeBinary`, and two casts through `binary` run (`arrow-cast@59.3.0:src/cast/mod.rs:257`, `:263`) |
   | an `interval_day_second(6)` | six digits after the point | `Interval(DayTime)`, which holds milliseconds: 2 days 1.500250 seconds reads as `2 days 1.500 secs` (`consumer/types.rs:242-245`) |

   - The DataFusion rules write a `uuid` as a `string`. The human: "it can be adapter
     rule to cast to string".
   - A dialect's name is no spelling that a model writes, except where rule 7 lists the
     same spelling as an alias, as it lists `i64`. A model's bare `time` is `time(9)`;
     Iceberg's bare value holds only for a catalog's native type, so a semantics
     declared `time` over an Iceberg `time` column gets ADR 0089 rule 17's cast to
     `time(9)`, which keeps every value.
   - A function's dialect declaration names canonical types, and a DataFusion
     declaration gives the canonical result type (ADR 0091, rule 5).
   - A dialect's own type families and widening order wait for later (ADR 0092,
     rule 2).

### A model writes the canonical name or an alias

7. A canonical name is SQL's name for the type, in lower case: ISO SQL's, or Calcite's
   where ISO SQL gives none or gives several words (`tinyint`, `double`,
   `interval_year_month`). Three types keep today's name in place of a SQL name.
   `string` and `binary` keep theirs, because SQL's `VARCHAR` and `VARBINARY` and
   Substrait's `varchar<L>` take a length, and `string` and `binary` take none.
   `interval` keeps its name, because each of Calcite's interval types holds either
   year-month or day-time fields (`SqlTypeName.java:85-108`), and an `interval` holds
   both. The human: "yep , but without complex types for now - but in sql names
   wheere it is possible".

   Where a model names a data type, as a semantics' `data_type:` and a cast's target
   do, it writes the type's canonical name or one of its aliases, as a string or as an
   object that names each value. The object's one key is the name or an alias. Under
   that key, the object writes every value of the type by the name that the
   declaration gives it, `precision`, `scale` or `length`, or an array's element as
   `element`: `{decimal: {precision: 18, scale: 4}}` is `decimal(18,4)`. A type that
   takes no value is written only as a string. The canonical names
   are `boolean`, `tinyint`, `smallint`, `integer`, `bigint`, `real`, `double`,
   `decimal(p,s)`, `string`, `varchar(n)`, `char(n)`, `binary`, `binary(n)`, `date`,
   `time(p)`, `timestamp(p)`, `interval`, `interval_year_month`,
   `interval_day_second(p)`, `uuid` and `array<T>`, and the bare `decimal`, `time`,
   `timestamp` and `interval_day_second` of rule 2. A bare `varchar` and a bare `char`
   are refused: a length is required (rule 2), and `string` is the type with no
   length. Parse reads a model's type name in any case, as today. The manifest stores
   the canonical name, and `Display` writes it.

   `native_to_canonical` reads a catalog's native type in two steps. First it reads
   the same names and aliases as a model does, with the values that the spelling
   writes in parentheses, so Iceberg's `int`, `long`, `float` and `decimal(9, 2)` are
   `integer`, `bigint`, `real` and `decimal(9,2)`. The human: "I think we can use
   aliases for this". Then, for a spelling that the first step gives no type, it
   reads the dialect that the catalog's provider names: the dialect's names, the bare
   value of its own declarations, and its spelling. The Iceberg REST provider names
   Iceberg (rule 6), whose spelling writes `fixed`'s length in square brackets
   (`fixed[16]`, each value trimmed) and two marks after `timestamp`: `_ns`, which is
   precision 9, and `tz`, which states that the value is an instant (Iceberg's zone
   flag). A timestamp that a source supplies holds a UTC instant (ADR 0072), so `tz`
   sets no value of the canonical type. Like the brackets, a mark is a part of
   Iceberg's spelling, not another name of a declaration. So Iceberg's `time` is
   `time(6)`, its `timestamp` and
   `timestamptz` are `timestamp(6)`, and its `timestamp_ns` and `timestamptz_ns` are
   `timestamp(9)`. An Iceberg `timestamp` holds a wall-clock value (`spec.md:289`), and
   it is read as an instant too, as every zone-free timestamp that a source gives is
   (ADR 0072). A catalog's spelling never takes a canonical default, so a bare
   `decimal` still maps to no type, and a catalog's casing still counts (ADR 0061). A
   spelling whose values its type cannot hold, such as `decimal(40, 2)` or `fixed[0]`,
   maps to no type (rule 2). A provider that names no dialect is read by the first
   step only.

   | Iceberg's native type | Canonical type |
   |---|---|
   | `boolean`, `int`, `long`, `float`, `double`, `date`, `string`, `uuid`, `binary` | `boolean`, `integer`, `bigint`, `real`, `double`, `date`, `string`, `uuid`, `binary` |
   | `decimal(P, S)` | `decimal(p,s)` |
   | `time` | `time(6)` |
   | `timestamp`, `timestamptz` | `timestamp(6)` |
   | `timestamp_ns`, `timestamptz_ns` | `timestamp(9)` |
   | `fixed[L]` | `binary(L)` |
   | `list`, `struct`, `map`, `variant`, `geometry`, `geography`, `unknown` | no type |

   A column that a semantics references, whose native type maps to no type, fails the
   compile with `CompileError::UnmappableNativeType` (ADR 0046); a column that no
   semantics references raises no error. So semstrait declares an alias, a dialect
   declaration or a type for each native type that it supports, and Iceberg's types
   come first. The human: "if catalog cannot map format types to what
   semstrait-sql declares - it is an error in compile ( or catalog side ) - so basically
   we need to store aliases or declration of type for what we are going to support -
   iceberg for now is priority". Iceberg's `list` maps to no type: compile has no
   reader of its JSON form. No canonical type holds a `struct`, a `map`, a `variant`, a
   `geometry` or a `geography`, and Substrait 0.63 has no type for the last three. An
   `unknown` column holds no value (`spec.md:267`).

   | Canonical name | Aliases |
   |---|---|
   | `boolean` | `bool` |
   | `tinyint` | `byte`, `i8` |
   | `smallint` | `short`, `int16`, `i16` |
   | `integer` | `int`, `int32`, `i32` |
   | `bigint` | `long`, `int64`, `i64` |
   | `real` | `float`, `float32`, `f32` |
   | `double` | `float64`, `f64` |
   | `decimal` | `numeric` |
   | `string` | `text`, `utf8`, `large_utf8` |
   | `binary` | `blob`, `bytes`, `varbinary` |
   | `date` | `date32`, `date64` |
   | `timestamp` | `datetime` |

   - Four of the spellings that `parse_scalar` accepts today are no aliases, because
     each names another type: `int8`, which is 64 bits in DataFusion's SQL
     (`planner.rs:717`) and in PostgreSQL; a bare `number`, which is a decimal in
     Snowflake and Oracle; and a bare `varchar` and a bare `char`, which need a length
     (rule 2).
   - Four more are no aliases, because an alias changes only the name (rule 1):
     `number(p,s)`, whose bare `number` is refused, and `timestamp_ms`, `timestamp_us`
     and `timestamp_ns`, which each fix a precision. No model in the corpus writes them.
   - A literal's tag takes only the canonical name (ADR 0089, rule 8). As a tag, `int`
     would read today's `{lit: {int: 0}}`, which gives a `bigint`, as an `integer`.
   - The 32-bit type is `real`, not `float`. Calcite's `REAL` is 4 bytes and its
     `FLOAT` is "A synonym for DOUBLE", and `real` is 32 bits in PostgreSQL, DuckDB,
     Spark, Trino and DataFusion. In PostgreSQL, "`float` with no precision specified
     is taken to mean `double precision`", and Snowflake "treats all three as 64-bit".
     `float` is an alias, because Spark, Iceberg and DataFusion's SQL read it as 32
     bits (`planner.rs:743`).
   - The 64-bit integer is `bigint`, ISO SQL's name, which DataFusion's SQL reads
     (`planner.rs:717`). `long` is Spark's and Iceberg's name, and an alias.
     DataFusion's SQL and PostgreSQL have no `long`.
   - Each `DataType` variant and each `Literal` variant takes its type's canonical name
     (`BigInt`, `Literal::BigInt(i64)`), except that a `char(n)` and a `binary(n)` are
     `FixedChar` and `FixedBinary`. The human: "maybe it can be fixed_* like name
     internally, but model name is simpler".

### A data type names its family, and a parameter may name a family

8. `TypeClass` retires, and the type family (rule 9) takes its place. A function's
   parameter names an exact type, a type family, one type at any precision, any type,
   or a type variable `T`:

   | Today | Now |
   |---|---|
   | `ParamType::Concrete(t)` | an exact type `t` |
   | `NumericFamily` | the number family |
   | `StringFamily` | the string family |
   | `DecimalFamily` (`round`, `ceil`, `floor`, `median`) | the number family (ADR 0087, rules 13 and 17) |
   | an overload for `Date` and one for `Timestamp { precision: 6 }` (`year`, `date_part`) | the date and timestamp family |
   | `Any` | any type |

   `Integral`, `FloatingPt`, `Comparable`, `Textual` and `Temporal` retire. A comparison
   takes two operands of one family (ADR 0092, rule 5).

9. A data type names its family and the types it widens to by two methods,
   `family()` and `widens_to()`, not by parts of its declaration. An operation checks
   its operands by these families (ADR 0092, rule 2), and a function's parameter may
   name one (rule 8). The human: "It feels like something I have described as family
   which can be used to declare whether operations can work with its operands ( like
   Long > Int works fine , because they are one family )".

   | Type family | Types | Widens to |
   |---|---|---|
   | number | `tinyint`, `smallint`, `integer`, `bigint`, `real`, `double`, `decimal` | `smallint` to `integer` or `bigint`; `integer` to `bigint`, `double` or `decimal`; `bigint` to `double` or `decimal`; `real` to `double`; a decimal to a decimal of more precision at the same scale |
   | string | `string`, `varchar(n)`, `char(n)` | `char(n)` and `varchar(n)` widen to `varchar(m)` where m ≥ n, and each widens to `string` |
   | boolean | `boolean` | — |
   | binary | `binary`, `binary(n)` | `binary(n)` widens to `binary` |
   | date and timestamp | `date`, `timestamp` | — |
   | time | `time` | — |
   | interval | `interval` | — |
   | year-month interval | `interval_year_month` | — |
   | day-time interval | `interval_day_second(p)` | to a higher precision |
   | uuid | `uuid` | — |
   | an array family for each element family | `array<T>` | — |

   - The widening order is the one compile has today in code (`is_widening`), and it
     moves into `widens_to()`. `tinyint` has no place in it.
   - A date and a timestamp are one family, and a time is apart. The human: "date and
     timestamp can be compared in operators … time is separate".
   - An array's family comes from its element: `array<bigint>` and `array<integer>`
     are in one family, and `array<bigint>` and `array<string>` are not. They share one
     declaration, and no array widens to another.
   - A year-month interval and a day-time interval are two families, as in Calcite
     (`SqlTypeName.java:85-108`), so `1 month < 30 days` fails the family check (ADR
     0092, rule 5), as it fails in ISO SQL.
   - The human, on the form: "not everything as methods - but What needed for the
     scope and DataTypes specifically".

### A cast is legal by its own table

10. `casts_to()` holds which casts are legal. A cast is a conversion from one type to
    another. It is not an operation's check of its operands. For now the table is
    written by type family, except where a cell names types: a timestamp to a time, and
    an array by its element. The human: "Family is operations, it is not abligation to
    cast … but I agree with family. we can enchance later on."

    | From \ To | number | string | boolean | binary | date, timestamp | time | interval | year-month interval | day-time interval | uuid | array |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | number | yes | yes | — | — | — | — | — | — | — | — | — |
    | string | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | — |
    | boolean | — | yes | yes | — | — | — | — | — | — | — | — |
    | binary | — | yes | — | yes | — | — | — | — | — | — | — |
    | date, timestamp | — | yes | — | — | yes | timestamp to time | — | — | — | — | — |
    | time | — | yes | — | — | — | yes | — | — | — | — | — |
    | interval | — | yes | — | — | — | — | yes | — | — | — | — |
    | year-month interval | — | yes | — | — | — | — | yes | yes | — | — | — |
    | day-time interval | — | yes | — | — | — | — | yes | — | yes | — | — |
    | uuid | — | yes | — | — | — | — | — | — | — | yes | — |
    | `array<T>` | — | — | — | — | — | — | — | — | — | — | `array<U>` where `T` casts to `U` |

    - Semstrait refuses a cast that the table does not allow. The table also holds for
      the cast to a semantics' declared type where the tree's type is known (ADR 0089,
      rule 17). Compile does not check a cast from `Unknown`, and the engine refuses a
      value that does not cast (ADR 0089, rule 18).
    - A legal cast can still fail on one value, such as `'abc'` to a number or
      `decimal(18,2)` to `decimal(5,2)`. It is then an error, or NULL with
      `on_failure: Null` (ADR 0092, rule 5), which DataFusion reads as `TRY_CAST`
      (`consumer/expr/cast.rs:26-50`).
    - A cast to a `varchar(n)` or a `char(n)` cuts the value to n characters, as ISO
      SQL's cast does, and a cast to a `char(n)` pads it with spaces. A cast to a
      `binary(n)` fails on a value of another length.
    - Digits after the point round or are cut by the canonical CAST's rule, which is
      DataFusion's for now. To a decimal with fewer digits after
      the point, and from a `real` or a `double` to a decimal, the value rounds half
      away from zero. To an integer, from a `real`, a `double` or a decimal, the value
      is cut toward zero (`arrow-cast@59.3.0:src/cast/decimal.rs:351-410`, `:770-819`,
      `:883-895`, `src/cast/mod.rs:2550-2572`). PostgreSQL's cast rounds `2.7` to `3`.
      A later dialect whose engine rounds otherwise needs its own rule then.
    - An `interval` does not cast to a year-month or a day-time interval, because each
      cast drops a part of a value: `1 month 3 days` loses its 3 days as an
      `interval_year_month`, and as an `interval_day_second` it has no single number of
      days, because a month has 28 to 31 days. DataFusion 55 fails both casts on every
      value ("Unsupported CAST from Interval(MonthDayNano) to Interval(YearMonth)"). A
      year-month or a day-time interval casts to `interval` with no loss.
    - Every cast that the table allows among `boolean`, the number types, `string`,
      `binary`, `date`, `timestamp`, `time` and `interval`, DataFusion 55 allows
      (`arrow-cast@59.3.0:src/cast/mod.rs:115`).

### Ruled elsewhere, not here

- A literal of each type, its variant and its parse default are ADR 0085's (rules 4
  and 5). Where `is_valid()`, `is_valid_on()` and `casts_to()` are checked, the YAML
  tag of a literal, and the literal grammar are ADR 0089's (rules 8 to 10, and 14).
- A catalog's native type spellings, as a catalog reports them, stay
  `semstrait-catalog`'s vocabulary (ADR 0046, *The schema is canonical; the native
  spelling is catalog-owned provenance*), and rule 7 reads them.
- How the DataFusion rules write a value where DataFusion's consumer differs (rule 6's
  last table), such as a `uuid` or a cast to a `varchar(3)`, is not ruled here.

### Amends

- **0061:** in *An unmappable native spelling stops the compile no longer*, "no
  type-mapping roster has been ruled", "This is not a new alias rule and does not reopen
  `native_to_canonical`'s refusal to make roster choices" and "no
  `DataType::parse_scalar` authoring aliases" become: `native_to_canonical` reads each
  type's canonical name and its aliases, then the dialect that the catalog's provider
  names (ADR 0086, rule 7), so Iceberg's `int`, `long` and `float` are `integer`,
  `bigint` and `real`. No
  casing tolerance, and no bare `decimal` acquiring an invented precision, stand: a
  catalog's bare spelling maps only where its dialect's declaration fixes the value, as
  Iceberg's `time` is `time(6)`. "`CompileError::UnmappableNativeType`
  is not deleted; it is demoted" no longer holds, nor does the rejection in
  *Considered Options* of "Keeping `UnmappableNativeType` fatal and extending the
  mapping roster instead": a referenced column whose native type maps to no type
  fails the compile (ADR 0086, rule 7). So
  "`native_to_canonical` returning `None` keeps the column", "A catalog-derived
  source can carry a `None` type", the entry for 0046's "demoted to a
  Warning-severity diagnostic, emitted only on the catalog-observed arm", and, in
  *Consequences*, "An Iceberg table carrying a spelling this workspace has not
  placed now compiles" no longer hold, and *`validate()` gains exactly one
  invariant* has no subject in a manifest that compile wrote. The rest of 0061
  stands.
- **0022:** in its *Decision Outcome*, in "The roster is deliberately narrower than the
  primary engine", the data type roster gains `array`, whose element is a canonical
  data type (ADR 0086, rule 4), and `varchar(n)`, `char(n)`, `binary(n)`, `uuid`,
  `interval_year_month` and `interval_day_second(p)` (ADR 0086, rule 3). The roster is
  no longer narrower than the primary engine for every type: DataFusion 55's consumer
  refuses a `uuid` (ADR 0086, rule 6). `Timestamp` and `Time` stay tz-naive, and
  `Decimal` precision stays capped at 38. The rest of 0022 stands.
- **0068:** in *The seam*, "a type's encoding", a format detail that a rule chooses,
  becomes: a type's Substrait type variation. A type's Substrait name comes from its
  Substrait declaration in `semstrait-sql` (ADR 0086, rule 6), and the format adapter
  writes each type's proto from its `DataType` variant. The rest of 0068 stands.

### Consequences

- Good, because a data type's declarations have a function's form: a canonical
  declaration, a dialect's declaration where the dialect differs, and `on(dialect)` to
  read them.
- Good, because code reaches a declaration only from its type, and a variant with no
  declaration does not compile.
- Good, because a rule and a rewrite read a type's Substrait name from its declaration.
- Good, because the manifest stores SQL's names. DataFusion's SQL reads `tinyint`,
  `smallint`, `bigint`, `real` and `array<T>` (`planner.rs:709-915`), and `real` means
  32 bits on most engines.
- Bad, because `on()` returns `Option`, so every reader handles a `None`, which only
  Iceberg gives today (rule 6). The gain: a dialect that lacks a type does not change
  what `on` returns.
- Good, because one vocabulary, the type family, serves both a function's parameter and
  an operation's operand. The gap closes, and a temporal function takes a timestamp of any
  precision.
- Good, because semstrait refuses an illegal cast, not the engine.
- Good, because the canonical CAST's rounding is declared, so a cast to a declared type
  gives one result.
- Bad, because Substrait's names are copied from Substrait's type classes and extension
  files, and nothing checks the copy. The gain: a rule and a rewrite read a type's
  Substrait name from one place.
- Bad, because DataFusion declares no data type, so its rules must know each place
  where its consumer differs (rule 6's last table), and no declaration records it. Until
  the rules write each one, those rows give DataFusion's result or a refused plan. A
  function that DataFusion does not have is marked `None` (ADR 0087, rule 5); a type
  that its consumer refuses is not. The gain: no type has a DataFusion declaration,
  because the plan holds Substrait's types.
- Bad, because a bare `binary` and `binary(16)` are two types, of variable and of fixed
  length, while a bare `decimal` is the one type `decimal(38,9)`. The gain: the name
  that a model writes is simple.
- Good, because a model may write `long`, `float`, `int64` and the other aliases of
  rule 7, a catalog's `long` and `float` are read as `bigint` and `real`, and the
  manifest stores one name for each type.
- Bad, because a model that writes `int8`, `number`, `number(p,s)`, a bare `varchar`,
  a bare `char`, `timestamp_ms`, `timestamp_us` or `timestamp_ns` is refused, where
  today it gets a type. The corpus changes one line:
  a fixture writes a bare `number`,
  which becomes `double`. The gain: an alias changes only the name, and the spellings
  that name another type in DataFusion's SQL, or that need a length, are no aliases.
- Good, because a bare `timestamp` holds every date that a column holds, `9999-12-31`
  included, and every precision of a time or a timestamp is one that DataFusion's
  consumer takes.
- Neutral: a bare `timestamp` changes from precision 0 to 6, and a bare `time` from 6
  to 9.
- Bad, because DataFusion refuses a table column that its caller registered as
  `LargeUtf8` when the plan reads it as Substrait `string` (`consumer/utils.rs:291-356`).
  Neither the plan nor the rules can see a table's encoding, so the caller registers the
  table. DataFusion reads a parquet string as `Utf8View`, which it accepts, and a
  parquet binary as `BinaryView`, which it refuses as `binary` (rule 6's last table).
- Bad, because the table refuses casts that DataFusion allows: a number to or from a
  boolean, a date, a time or a timestamp, an integer to binary, and any type to or from
  an array other than an array.
- Bad, because an `interval` with no months, such as `2 days 1.5 seconds`, does not cast
  to an `interval_day_second` either. The gain: no cast drops a part of an interval.
- Bad, because the canonical CAST's rounding is DataFusion's, so a `double` tree that
  holds `2.7`, declared `integer`, gives `2`, where PostgreSQL gives `3`. The gain is
  that the one engine needs no rewrite.
- Good, because a referenced Iceberg column of each of Iceberg's primitive types of
  format version 1, and of version 3's `timestamp_ns` and `timestamptz_ns`, gets a
  canonical type, at the precision and the length that the Iceberg spec fixes and at
  the values that rule 2 holds. Spark writes its default timestamp type to Iceberg as
  `timestamptz` (`SparkTypeToType.java:147-148`), and DataFusion 55 reads that column
  as `precision_timestamp(6)`. DataFusion 55 refuses a `uuid` and a `binary` read until
  its rules write them (rule 6's last table).
- Bad, because Iceberg's `timestamp`, a wall-clock value, and its `timestamptz`, an
  instant, read as one type, so a model cannot tell them apart. An Iceberg `timestamp`
  that holds the Stockholm wall-clock value `2026-03-31 23:30` is read as 23:30 UTC,
  which is 01:30 on 1 April in Stockholm, so under the request timezone
  `Europe/Stockholm` it falls in April, where its own clock says March (ADR 0072). The
  gain: the roster has no type with a zone, and the values that semstrait computes from
  an instant are the same, because ADR 0072 reads every zone-free timestamp that a
  source gives as a UTC instant; rule 6's last table holds DataFusion's differences.
- Bad, because a referenced Iceberg `list`, `struct`, `map`, `variant`, `geometry` or
  `geography` column fails the compile, and so does every referenced column of a Unity
  catalog, whose spellings are upper case, until semstrait reads Unity's spellings. The
  gain: Iceberg's types first; for now, the roster gains no record, map, variant or
  geometry type, and compile no reader of a `list`'s JSON form or of Unity's spellings.
- Bad, because Iceberg's spelling is its own code beside the canonical one: the `_ns`
  and `tz` marks and the square brackets. The gain: Iceberg's names and values sit in
  its types' declarations, beside Substrait's.
- Neutral: `Dialect` gains `Iceberg`, which declares data types only; no code matches on
  `Dialect`'s variants. `DocumentDialect` reads `iceberg` as a document's dialect, and a
  manifest refuses it, as it refuses `substrait`. A catalog provider names the dialect
  of its native types, and the Iceberg REST provider names `Iceberg`.
- Neutral: an array adds a `Literal` variant, array families and a spelling.
- Neutral: `TypeDeclaration` gains `aliases`, a `length` and an integer's `digits`, and a
  `Limit` holds a `u32` and an optional default. The renames reach the catalog's
  `array_element` and `array.rs`, and the literal tags, such as `{lit: {bigint: 0}}`.
- `CONTEXT.md` changes:
  - **Canonical data type**: the roster holds the array, so it is no longer only scalar
    types, and it takes SQL's names. A dialect may name a type in its own words and take
    only some of its values, and its _Avoid_ gains reading a dialect declaration as a
    second type, `byte`, `short`, `long` and `float` as a type's name, which are
    aliases, and `list`. It also says that a model may write a type by an alias.
  - **Canonical declaration** and **Dialect declaration** also cover a data type: its
    name and the values its precision, scale and length take. A data type's dialect
    declaration is written where the dialect's type differs in name or in the values it
    takes, and a dialect's entry says when it has no such type.
  - **Dialect**: a format's vocabulary is a dialect too, and Substrait's is one that
    several engines implement with their own extensions. DataFusion's declares
    functions and operation nodes, and Substrait's declares data types only. Its
    _Avoid_ drops "for a format such as Substrait" and gains reading Substrait's
    dialect as a Substrait dialect file, which states what one engine supports.
  - **Engine adapter**: its _Avoid_ says that a dialect is an engine's or a format's
    declared vocabulary. **Format artifact**: a dialect may declare a format's names
    for a type, and only a format artifact writes them.
  - **Type family**: it gains the uuid, year-month interval and day-time interval
    families and an array family for each element family, and a function's parameter
    may name a family. Its _Avoid_ drops the `Comparable` reason and gains reading a
    family as the cast rule.
  - **Dialect**: Iceberg's is a dialect too, which declares data types only, and compile
    reads a catalog's native type through it. **Dialect declaration**: for a dialect
    whose spellings compile reads, a data type's is also written where its bare
    spelling takes a value that the dialect fixes.
  - **Native type**, a new entry: a catalog provider's own spelling of a column's type,
    which compile's catalog stage reads where it enters. **Canonical layer** keeps its
    clause and places that stage on the layer's edge, and **Physical source** says so
    too. **Canonical data type**: a format adapter writes it in its format's own names.

## More Information

Decided on a canonical data type's declaration and its dialect
declarations, and on the SQL types and names the canonical data type
roster gains, for the names, the new types and the Substrait dialect. The canonical
CAST's rounding was decided on where expression typing and validation
run once semstrait-sql exists. The mark for a type
that a dialect does not have was ruled, and so were a bare timestamp's precision,
a time's and a timestamp's precisions, the casing, the aliases and an integer's fixed
precision. The human's rulings (2026-10-17) make an
unmapped catalog type a compile error. Iceberg's dialect, and a catalog's native type
read through the dialect that its provider names, were decided on the
semstrait-sql data type for each Iceberg type that a catalog
reports (2026-10-19), with the
research and the DataFusion runs.
