# semstrait-sql

The typed SQL vocabulary of semstrait: the canonical data types, their
declarations and type systems, the function declarations and the registry that
finds them, and the expression tree with its literal, the name path of a
reference, the operations' declarations, the fold and the tree's JSON form.

Decisions: ADR 0085 (the tree, the literal, the name path), ADR 0086 (the data
types), ADR 0091 rules 1 to 6 (the function declarations, a dialect
declaration, `FunctionRef`, the calls and their check, and `Dialect`), ADR 0087
rules 6 to 9 (where DataFusion has a declaration, the catalog's layout, the
registry, the narrowing of its handle and its seal; ADR 0020 as ADR 0087 amends
it), ADR 0087 rules 11 to 21 (the string, pattern, math, temporal, conditional,
array and aggregate functions, the NULL choice, the window functions and the
names that retire) and rules 5 and 23 (DataFusion's declarations, and its `None`
entries), ADR 0072 (`date_trunc`), ADR 0088 rules 1 to 3, 8 and 9 (the tree's
JSON form), ADR 0089 rules 3 and 5 (a `string` parameter takes the string
family, and reading checks a call as parse does), ADR 0089 rules 3, 11, 12, 14 and 18
(typing, with ADR 0085 rules 6 and 7, ADR 0086 rule 8 and ADR 0091 rule 5) and
rule 21 (the fold), and ADR 0092 rules 1 to 8 (the operations' declarations and
their typing by type family, the type systems, an untyped literal's context,
and `coalesce` and `nullif`). Terms: `CONTEXT.md`
(**Canonical data type**, **Canonical declaration**, **Dialect declaration**,
**Dialect**, **Type family**, **Type system**, **Operation node**, **Sugar**,
**Function catalog**, **Function type**, **Call clause**, **Function option**,
**Fold**).

It depends on `semstrait-common` (with its `serde` feature, for `Dialect`'s name),
for `Dialect`, the `Tree` trait,
`ValidateError`, `Diagnose` and `Severity`, on bon, which derives the calls'
builders, on serde, which writes and reads the tree's JSON form, and on thiserror
for its errors. It reads no YAML
(ADR 0089, rule 6).
`semstrait-model`, `semstrait-ir` and `semstrait-manifest` each import its types
from it directly, never through another crate (ADR 0090, rule 6).

---

## Module map

| Module | What it owns |
|--------|--------------|
| `datatype` | `DataType`, each type's `TypeDeclarations` (`TypeDeclaration`, `Limit`, `Values`; `declaration.rs`), `TypeFamily` with the widening order and the casts, and `TypeOrFamily`, by which a refusal names a node with no type (`family.rs`, where `value_casts_to` sits, beside `casts_to()`), the name function with `resolve` and `roster()`, `DataType`'s serde (`name.rs`), `WrittenDataType` with `resolve_written` (`written.rs`), behind the `schemars` feature `DataType`'s JSON schema, `resolution_schema` and `any_case` (`schema.rs`), and the type systems (`TypeSystem`, `CanonicalTypeSystem`, `SubstraitTypeSystem`, `Decimal`, `promote`), with the typing rules that apply their derivations (`type_system.rs`). |
| `literal` | Not public: `expr` re-exports it. `Literal`, with `Literal::parse` and `Literal::held_value` with its `HeldValue` (`parse.rs`), its two checks (`check.rs`), the value that the canonical CAST gives it (`cast.rs`), its serde form (`form.rs`) and `IntoLiteral`, and `LiteralError` (`error.rs`). |
| `functions` | `Function`, `AggregateFunction`, `Declaration` and its parts (`ReturnRule`, `Nullability`, `Volatility`, the options and the clauses), `ParameterType` with `takes_value`, `Signature` with the typing rules that read a declaration (`signature.rs`), the function types `Scalar`, `Aggregate` and `Window`, the operations (`operation/`): `OperationKind`, `Operation`, `OperationDeclaration`, `OperationOption` and the operator kinds `BinaryOpKind`, `UnaryOpKind` and `LikeKind`; the handle (`handle.rs`): `FunctionRef`, with its bound `Aggregation`, `AggregateAllowed` and `NoAggregate`; the function catalog (`catalog/`), the registry (`registry()`, `seal` and `Registry`), and `SealFault` (`error.rs`). `expr` re-exports the operations and the handle. |
| `expr` | The tree (`node.rs`): `Expr` with `CastFailure`, `RowExpr`, the bound of its leaf (`Leaf`, `NoLeaf`) and the narrowing into a `RowExpr`; its traversal (`traversal.rs`): `Tree` and `Expr::checked`; a node rebuilt over another leaf (`rebuild.rs`), `Expr::rebuilt` and `map_nodes`; the Rust operators that build an operation node (`operation/`), with `ExprOperationExt`; the calls `Call`, `ScalarCall`, `AggregateCall` and `WindowCall`, with their builders `ScalarCallBuilder`, `AggregateCallBuilder` and `WindowCallBuilder`, `NullTreatment` and `OptionSetting`, and their check (`CallCheck`, `CallClause`); an order key (`sort.rs`): `SortKey`, `SortDir` and `NullOrdering`; what `FunctionRef` reads of a call's arguments (`function.rs`); a literal's typing (`literal.rs`); `NamedReference`; the leaves `SemanticLeaf` and `PhysicalLeaf` and their `UntypedLiteral`, with `DimensionAccessor`, `MeasureAccessor` and `MetricAccessor`; typing's walk (`type_tree`, `InputTypes`, `Typed`), `check_cast` and `is_condition`; the window specification (`window.rs`): `Over` with `Over::check`, and its window frame, `WindowFrame`, `WindowFrameKind` and `WindowBound`; the fold, `Expr::fold` and `Expr::fold_with`; the tree's JSON form (`serde/json/`), `JsonForm` and `JsonReader`; and the errors (`error.rs`): `CallError`, `OverError`, `JsonFormError`, `RebuildError`, `AggregateInRowExpr` and `TypeError`. It re-exports `literal`'s types, `functions`' handle with its bound, and the operations. |
| crate root | `DocumentDialect`: the dialect a document is written in, `Canonical` or one `Dialect`; serde and `Display` write `canonical` or the dialect's name, `datafusion`. |

Outside their tests, the modules import only the modules above them in this
table, and the crate root. Each type has one import path:
`semstrait_sql::datatype::DataType`, `semstrait_sql::functions::Function`,
`semstrait_sql::expr::Literal`, `semstrait_sql::DocumentDialect`.

---

## datatype

`DataType` has the 21 canonical types of ADR 0086 rule 1, each under its SQL
name (`TinyInt`, `BigInt`, `Real`, `Array`, …; a `char(n)` and a `binary(n)` are
`FixedChar` and `FixedBinary`), and the compile-only `Unknown`.

**Declarations.** Each type's declarations are a private const of `DataType`,
reached only through `declarations()`. That method has one arm for each variant
and no wildcard (clippy's `wildcard_enum_match_arm` is denied on it), so a
variant with no declaration does not build. It gives `None` for `Unknown`.
A `TypeDeclarations` holds the canonical `TypeDeclaration` and one entry for
each dialect that differs; `on(dialect)` gives the dialect's declaration, the
canonical one where the dialect has none, or `None` where the dialect has no
such type.

A `TypeDeclaration` holds the type's `name`, its `aliases`, the `Limit` of its
`precision`, `scale` and `length` (a set of `Values` and a bare spelling's
`default`), and an integer's fixed `digits` (tinyint 3, smallint 5, integer 10,
bigint 19), which no spelling writes.

| Type | Values | A bare spelling's value |
|---|---|---|
| `decimal` precision | 1 to 38 | 38 |
| `decimal` scale | 0 to the precision | 9 |
| `time` precision | 0, 3, 6 or 9 | 9 |
| `timestamp` precision | 0, 3, 6 or 9 | 6 |
| `interval_day_second` precision | 0 to 9 | 9 |
| `varchar`, `char` and `binary(n)` length | 1 to 2,147,483,647 | none |

The constructors `decimal()`, `time()`, `timestamp()` and
`interval_day_second()` give a bare spelling's type. `is_valid()` checks each
value against the canonical declaration, and a scale against its precision;
`is_valid_on(dialect)` narrows it by the dialect's declaration. `Unknown` and an
array of it are not valid.

**Dialects.** Substrait is a dialect for typing. Its declarations give
Substrait's name only, because Substrait takes every canonical value: `i8` to
`i64`, `fp32`, `fp64`, `fixedchar`, `fixedbinary`, `precision_time`,
`precision_timestamp`, `interval_compound`, `interval_year`, `interval_day` and
`list`. Every other type has no Substrait declaration. DataFusion declares no
data type, because the plan it runs holds Substrait's types, so
`on(Dialect::DataFusion)` gives the canonical declaration for every type.

Iceberg is a dialect for typing too (ADR 0086, rule 6): `int`, `long` and
`float` are aliases of `integer`, `bigint` and `real`, so `from_name` already
reads them; `time` and `timestamp` take Iceberg's own bare values (6, or 6 and
9); `fixed` is `binary(n)`'s name. `tinyint`, `smallint`, `varchar(n)`,
`char(n)` and the three intervals have no Iceberg declaration. Iceberg's own
spelling — `fixed[L]`'s brackets, and the marks `_ns` and `tz` after
`timestamp` — is not another name of a declaration, so it is read through
`from_native_name` (below) rather than through `on`.

**Names.** `DataType::from_name` reads a canonical name or an alias, as written,
with its values: `bigint`, `long`, `decimal(18,2)`, `numeric`, `varchar(10)`,
`array<string>`. A bare spelling takes its declaration's defaults; a bare
`varchar` and a bare `char` read as no type, because a length has no default.
`int8`, `number`, `number(p,s)` and `timestamp_ms`, `_us` and `_ns` are no
aliases. It checks no limit: `timestamp(4)` reads as a type that is not valid,
and the caller checks `is_valid()`. It is case-sensitive, so a catalog's casing
counts: it is the first step of `from_native_name`. `Display` writes the
canonical name with the type's values (`timestamp(6)`, `array<string>`).

**Resolution.** `DataType::resolve(spelling, dialect)` reads a name or an alias
in any case through a `DocumentDialect`'s declarations, and keeps only a type
that is valid in that dialect; a model passes `DocumentDialect::Canonical`.
`DataType::resolve_written(written, dialect)` reads a `WrittenDataType`: a
spelling, which goes through `resolve`, or the object form. The object has one
type key, a name or an alias in any case, mapped to every value of its
declaration under the declaration's lowercase keys
(`{decimal: {precision: 18, scale: 4}}`); `array` takes only `element`, which is
any `WrittenDataType` (`{array: {element: string}}`). An object keeps only a
valid type of at most 128 array layers, and a number alone names no type. Both
forms give the same `DataType`. A model's parse checks each type it reads
through `resolve_written`, and compile resolves a semantics' declared type
through it.

`DataType::from_native_name(spelling, dialect)` reads a catalog's native type
(ADR 0086, rule 7): `from_name` first, with no canonical default, then — where
that gives no type and `dialect` is `Some(Dialect::Iceberg)` — Iceberg's own
spelling, kept only where it is valid. Every other dialect, `None` among them,
is read by the first step alone.

**Serde.** A `DataType` is its canonical name. Serialize refuses a type that is
not valid, `Unknown` and an array of it among them. Deserialize reads
`from_name`'s grammar with the canonical name only, as written: it refuses an
alias (`long`), another case (`BIGINT`) and a type that is not valid, so no
manifest and no plan holds an alias or `Unknown`. Behind the `schemars`
feature, its JSON schema has one branch for each type, titled with its spelling
(`bigint`, `decimal(p,s)`), whose pattern takes the canonical name only.
`DataType::resolution_schema(dialect, element)` builds those string branches for
the name or an alias in any case, plus an object branch for each type that
takes values and a `{array: {element: T}}` branch whose element is the
`element` schema: the forms that `DataType::resolve_written` reads. `any_case`
gives the pattern of a name in any ASCII case; this schema and a model's
expression schema both build their names with it.

**Families, widening and casts.** `family()` gives the type's `TypeFamily`, by
ADR 0086 rule 9: number, string, boolean, binary, date and timestamp, time,
interval, year-month interval, day-time interval, uuid, and an array family for
each element family. `Unknown` and an array of it have none. `widens_to()` is
rule 9's widening order: `smallint` to `integer` or `bigint`; `integer` to
`bigint`, `double` or a decimal; `bigint` to `double` or a decimal; `real` to
`double`; a decimal to more precision at the same scale; `char(n)` and
`varchar(n)` to `varchar(m)` where m ≥ n, and to `string`; `binary(n)` to
`binary`; a day-time interval to a higher precision. No type widens to itself,
`tinyint` widens to nothing, and no array widens to another. `casts_to()` is
rule 10's cast table: by family, except that a timestamp casts to a time and an
array casts by its element.

**Type systems.** A return rule names a derivation, and a type system holds it
(ADR 0092, rule 3). `TypeSystem` is a trait whose default methods are the
canonical rules; `CanonicalTypeSystem` takes every default, and
`SubstraitTypeSystem` overrides only `avg_of`. A decimal rule takes and gives a
`Decimal`, a `DataType::Decimal`'s parts; `Decimal::of` gives a decimal's parts,
or an integer's `digits` at scale 0 (tinyint (3,0), smallint (5,0), integer
(10,0), bigint (19,0)), and `None` for any other type.

| Rule | Canonical result |
|---|---|
| `decimal_union` (`Promote`) | scale max(s1, s2), precision max(p1 − s1, p2 − s2) + scale |
| `decimal_sum` (`+`, `-`) | scale max(s1, s2), precision scale + max(p1 − s1, p2 − s2) + 1 |
| `decimal_product` (`*`) | scale s1 + s2, precision p1 + p2 + 1 |
| `decimal_quotient` (`/`, `SafeDivide`) | scale max(6, s1 + p2 + 1), precision p1 − s1 + p2 + scale |
| `decimal_mod` (`%`) | scale max(s1, s2), precision min(p1 − s1, p2 − s2) + scale |
| `sum_of` | an integer gives bigint, a real or a double double, a decimal(p,s) decimal(38,s) |
| `avg_of` | an integer, a real or a double gives double, a decimal(p,s) decimal(38, max(s, 6)); Substrait's keeps an integer's, a real's or a double's type, and gives decimal(38,s) |
| `decimal_scale_zero` (`ceil`, `floor`) | decimal(min(p − s + 1, 38), 0) |
| `round_of` (`round`) | decimal(min(p + 1, 38), s) |

Above precision 38, each of the five decimal operation rules gives precision 38
and lowers the scale by the excess, to no less than 6, or the scale's own value
where that is less, as Substrait's do. `sum_of` and `avg_of` give any type they
do not name itself, as Calcite's do. Each rule reads the greatest precision, 38,
from the decimal's declaration, and an integer's digits from its own, so no
limit has a second home in the type system. `promote(system, types)` gives
`Promote`'s type of a list of types, `Unknown` where neither of two widens to
the other; the Substrait producer casts the operands of a call to it.

## functions

**Declarations.** A `Function<F>` holds its canonical `Declaration<F>`, and one
entry for each dialect where that dialect differs: a declaration of the same
struct, or `None` where the dialect has no such function. `on(dialect)` gives
the dialect's declaration, the canonical one where the dialect has none, or
`None`. It holds too a `releases` entry for each dialect's release or range of
releases that differs from its dialect entry: a dialect, a `RangeInclusive<Release>`
and a declaration of the same struct, or `None`. `on_release(dialect, release)`
gives the first `releases` entry whose range holds `release`, else `on(dialect)`.
An `AggregateFunction` holds its `Function<Aggregate>` and, once for
every dialect, the `Additivity` of its plain form and of its DISTINCT form.

A `Declaration` holds the function's `name` (on a dialect, the engine's own),
its `aliases` (none on a dialect), its `Signature`, its `ReturnRule`, its
`Nullability`, its `Volatility`, and in `specific` what only its function
type has: a scalar function's `Category` and options, and an aggregate or a
window function's options and `ClauseRule` for each clause. `FunctionType` is
sealed: only `Scalar`, `Aggregate` and `Window` implement it. A declaration and
each of its parts are `#[non_exhaustive]`, as their fields may grow (ADR 0007),
so only this crate builds one, and the tests that build one are in the crate.

A `Signature` holds the parameters, a variadic tail and the constraints across
the arguments: an enum value that an argument's type refuses (`EnumRefused`),
every argument of one type family (`OneFamily`), as `nullif`, `greatest` and
`least` take theirs, or one of some rows of families, a family for each argument
(`Families`). A `ParameterDecl` takes a `ParameterType` (an exact type, a type
family, one type at any precision, any type, or the type variable `T`), an
`ArgumentForm` (a value, a constant, an enum with its values, or a dialect's
enum, which names each canonical value in the dialect's words) and a
`Presence` (required, optional, or only when an earlier argument says so). An
optional parameter that a later parameter follows is `Presence::Defaulted`: it
declares its default as a `ParameterDefault`, an integer or a string, and
`ParameterDecl::default()` gives it as a `Literal`, which
`FunctionRef::arguments_by_name` writes where a call's mapping names a later
parameter and not this one (ADR 0091, rule 4; ADR 0089, rule 4). A parameter that
is present only when an earlier argument says so, `Presence::When`, may declare a
default too, which a call that leaves it out takes there: `date_trunc`'s
`week_start` `'monday'`, which parse writes, and its `timezone` `'UTC'`, which
typing writes. A
type in a declaration is a `&'static` reference, so a declaration holds
nothing to drop, and a `static` declaration can be built from another with
`..`. An exact type, of a parameter or of a `Fixed` result, is an `ExactType`:
a type, or the array of a type (`array<string>`), since no `static` can build
a `DataType::Array`, which boxes its element.

An `OptionDecl` names its option, the values the declaration takes and its
canonical value. An option's name is a variant of `ScalarOption`,
`AggregateOption` or `WindowOption` (or of `expr`'s `OperationOption`), which
gives its name and every value it can take, as Substrait spells them. Today
`concat`'s `null_handling` is the one function option, and no aggregate or
window function has one.

**The catalog.** `functions/catalog/` has one file for each scalar category
(`string`, `pattern`, `math`, `temporal`, `conditional`, `array`), one for
the aggregate functions and one for the window functions (ADR 0087, rule 7).
A function enters by its declarations in its file. A parameter that names a
type family takes every type of the family, as a `string` parameter takes a
`varchar(n)` (ADR 0086, rule 8; ADR 0089, rule 3); a `string` result is exactly
`string`. `catalog/mod.rs` holds the types and the builders that more than one
file of the catalog needs. The builder of a required parameter, `value`, and
the parameter types of the string and number families, `STRING` and `NUMBER`,
are in `functions`, as the operations use them too.

- `string.rs` declares rule 11's 30 string functions; an integer parameter
  takes the exact `integer`, and `concat` and `concat_ws` take two strings or
  more (rule 11). DataFusion has a declaration of `position`, as
  `strpos` with the value first, and of `concat`, whose `null_handling` takes
  only `IGNORE_NULLS` (rule 23).
- `pattern.rs` declares rule 12's six functions, with DataFusion's
  `regexp_replace` and its `None` for `regexp_extract` (rules 5 and 23).
  `regexp_instr`'s `start`, `occurrence` and `flags` declare their defaults, 1,
  1 and `''`, and its last parameter, `group`, declares none (rule 12).
- `math.rs` declares rule 13's 28 math functions. DataFusion's `sign` is
  `signum`, and its `width_bucket` is `None`: its rules write it (rules 5 and
  23).
- `temporal.rs` declares rule 14's nine functions; an enum parameter, and a
  constant one that the roster writes `string`, take the string family:

  | Function | Shape | NULL |
  |---|---|---|
  | `extract` | `(component: enum, value: date or timestamp) → bigint`; ten components, `day_of_week` ISO's (Monday = 1); `hour`, `minute` and `second` refused on a date | Mirror |
  | `date_trunc` | ADR 0072's `(granularity: enum, value: T date or timestamp, timezone: constant string, for a timestamp only, 'UTC' by default, week_start: monday or sunday, for the week grain only, 'monday' by default) → T`; `minute` and `hour` refused on a date | Mirror |
  | `date_diff` | `(part: enum, start: T date or timestamp, end: T) → bigint`; the count of the part's boundaries, a part being one of ADR 0072's grains, `second`, `millisecond`, `microsecond` or `nanosecond`, as Calcite's TIMESTAMPDIFF takes them | Mirror |
  | `current_date`, `current_time`, `current_timestamp` | `() → date`, `time(9)`, `timestamp(9)` with no zone; Stable | never NULL |
  | `make_date` | `(year: integer, month: integer, day: integer) → date` | Mirror |
  | `to_date`, `to_timestamp` | `(value: string) → date`, `timestamp(9)`; ISO 8601 text only | Mirror |

  DataFusion declares `extract` as `date_part`, whose part is an enum of its
  part names, one for each component (`isodow` for `day_of_week`, `doy` for
  `day_of_year`), written as a string, and `date_trunc` with its granularity a
  constant string and no timezone and no week start; its entry for `date_diff`
  is `None` (rules 5 and 23). Each gives the canonical result type. `year` …
  `second`, `date_part`, `date_add` and `date_sub` are not declared (rule 20).
  DataFusion 53's `isodow` counts Monday as 0, not 1, so `extract` has a
  `releases` entry for DataFusion 53 to 53 whose part table drops
  `day_of_week` (ADR 0091, rule 6; rule 23).
- `conditional.rs` declares rule 15's four functions: `coalesce`, `nullif`,
  `greatest` and `least`. `coalesce` takes two arguments or more, and
  `greatest` and `least` one or more, as ISO's do. DataFusion's `greatest` and
  `least` are NULL only when every argument is NULL (rule 23).
- `array.rs` declares rule 16's `array_element`.
- `aggregate.rs` declares rule 17's 28 aggregate functions, each with its
  clauses and the additivity of its plain form and of its DISTINCT form. FILTER
  is `Allowed` on every one; ORDER BY in the call is `NoEffect` but on
  `string_agg` (`Allowed`) and `percentile_cont`, whose order is its `Required`
  WITHIN GROUP. DataFusion declares four of them (rule 23): `stddev_samp` and
  `stddev_pop` refuse DISTINCT, `percentile_cont` takes its value as its first
  argument and its sort as ORDER BY in the call (`Allowed`), not as WITHIN
  GROUP, and `approx_count_distinct` is
  `approx_distinct`. `stddev` and `variance` are not declared (rule 20).
- `window.rs` declares rule 19's eleven window functions, six ranking and five
  analytical, with their shapes and clauses; only the analytical ones have a
  DataFusion declaration (rule 23). The `offset` of `lag` and `lead` declares
  its default, 1 (rule 19).

**The registry.** `registry()` seals the catalog once, at its first access, and
panics there on a fault (ADR 0020). `seal` seals any `FunctionCatalog` and gives
every `SealFault` of its declarations, so a test in the crate can feed a
hand-built one.
`Registry::get` takes a name as text, folds its ASCII case, and finds a
function by its canonical name or an alias, each a lower-case identifier, as
Calcite's operator table and DataFusion's registry take a plain name (ADR 0087,
rule 8); `all()` gives every function in the catalog's order;
`find(dialect, name)` finds a function by its
name in a document's dialect: in `canonical`, a canonical name and no alias; in
a dialect, the dialect declaration's name, or the canonical name where the
dialect has none, and nothing where the dialect's entry is `None`.

**The seal** refuses, each as one `SealFault`: two functions that share a name
or an alias, or a name in one dialect (a dialect declaration's name, or the
canonical name where the function has no entry there); a name or an alias that
is an operation's name or alias (`mod`, `between`); and, on the canonical and
on each dialect declaration, a category that is not its file's, an option value
outside its option, a canonical value that is not among the values, a required
parameter after one that a call may leave out, an optional parameter that a
later parameter follows and that declares no default, a default that is not a
literal of its parameter's type (ADR 0091, rule 4), a condition that names an
argument that is not an earlier one, an argument index that names no parameter,
a result of the type variable that no parameter binds, and an aggregate's
nullability on a scalar function. The tree's YAML tags (`col`, `lit`, `case`,
`cast`, …) are `semstrait-model`'s grammar, which this crate does not read, so
the seal does not check them: `semstrait-model`'s parse reads its own tags
before it looks a function up, and its test checks that no function has a
tag's name (ADR 0087, rule 9; ADR 0089, rule 6). On the dialect entries it refuses two for one dialect, aliases
on a dialect declaration, and a dialect declaration equal to the canonical one
in every part but the aliases (a dialect declaration has none). On the
`releases` entries it refuses two of one dialect whose ranges overlap, and one
whose range holds the dialect's reference release (ADR 0091, rule 6). Each
`SealFault` is an error and a diagnostic kind (`Diagnose`, ADR 0004) with one
message, and `registry()`'s panic gives each fault's message.

**Errors.** Each error of the crate is a typed kind (ADR 0004) that derives
`Display` and `std::error::Error` with thiserror, in `expr/error.rs`,
`functions/error.rs` or `literal/error.rs`; `RebuildError::Call` gives its
`CallError` as its source, and `CallError::Over` its `OverError`.

## expr

**The tree.** `Expr<L: Leaf = NoLeaf, A: Aggregation = AggregateAllowed>` holds
the `Literal` and `Column(NamedReference)` nodes, `Leaf(L)`, a stage's own leaf,
the operation nodes `BinaryOp`, `UnaryOp`, `Cast`, `Case`, `InList`,
`Between`, `Like` and `IsNull` (ADR 0085, rule 1), and `Function(Call)`, the one
call node (ADR 0091, rule 1). `RowExpr<L>` is `Expr<L, NoAggregate>`. It takes
no serde derive: its JSON form is written and read by hand (*The JSON form*,
below). `literal()` gives the literal that a node is: a `Literal` node, or a
leaf's untyped literal.

`Expr` implements `semstrait-common`'s `Tree`: `children()` gives a node's
operands in the order it holds them (a CASE gives each WHEN and its THEN, then its
ELSE; an IN its value, then its list; a call its arguments, then an aggregate
call's FILTER and order keys, or a window call's partition and order keys), and
`with_new_children()` puts new ones in their place, refusing another number of
children, an IN with no item and a CASE with no WHEN. A call is checked against
its declaration again there, as the check that built it says, and refused with
`CallRefused` where its new operands do not fit it or hold a call that it cannot
hold. `Expr::checked()` is that check of one node: the JSON form's reader runs
it on each node it builds, and
parse on each CASE and IN it builds, since a call's constructor has already
checked each call it builds. `Expr::rebuilt()` gives a node over another leaf,
with new operands in `children()` order and each call built again through its
constructor, so a stage that changes a tree's leaf node by node holds no copy
of the tree's structure: compile when it lowers an authored body, and the
manifest when it stores compile's tree as an entry's whole tree over its link
leaf and reads that tree back as the plan's. Rust's `+ - * / %` and `&` and
`|` (AND and OR) build a `BinaryOp`, and its unary `-` and `!` a `UnaryOp`;
`ExprOperationExt`'s `eq`, `neq`, `lt`, `lt_eq`, `gt`, `gt_eq` and
`is_not_distinct_from` build a comparison, and `is_null` an `IsNull`.

**Narrowing.** An `Expr<L, AggregateAllowed>` converts into a `RowExpr<L>` by
`TryFrom` only where it holds no aggregate call, and the registry's
`FunctionRef<AggregateAllowed>` into a row-level tree's `FunctionRef<NoAggregate>`
only where it is no aggregate function's; each refuses with `AggregateInRowExpr`
(ADR 0085, rule 3; ADR 0087, rule 8).

**The bounds.** `Leaf` is the bound of a stage's own leaf; each stage implements
it for its leaf. Its `untyped()` gives the untyped literal that the leaf holds,
and `None` by default. `NoLeaf` is the leaf of a tree that holds none, and it has
no value. `Aggregation` says whether a tree may hold an aggregate. It is sealed:
only `AggregateAllowed` and `NoAggregate` implement it, and `NoAggregate` has no
value.

**The handle.** `FunctionRef<A>` is a handle to a function, as the registry
gives it: a reference to a scalar, an aggregate or a window function's
`static`, which compares and hashes by the function's canonical name. Its
aggregate variant holds an `A`, so a tree that holds no aggregate holds no
handle to one. `canonical()` gives the name and the signature of the function's
canonical declaration.

**The calls.** A `Call<L, A>` is a call of its function's type: `Scalar` holds a
`ScalarCall`, `Aggregate` an `AggregateCall` and the tree's `A`, and `Window` a
`WindowCall` (ADR 0091, rule 1). Each holds its function's `&'static` handle, its
arguments, its `OptionSetting`s (an option's name and value; none where the call
takes the canonical values), and only its type's clauses: an aggregate call its
DISTINCT, its FILTER, its order keys (ORDER BY in the call, or WITHIN GROUP where
the declaration takes one) and its `NullTreatment`; a window call its
`NullTreatment` and its `Over`, the partition, the order keys and the window
frame. An
order key is a `SortKey`, an expression with its `SortDir`. `Call` gives
`name()`, the canonical name, `args()`, `argument(parameter)`, the argument at a
named parameter of the canonical declaration, and `options()` for every call, and
each typed call gives its `function()` and its clauses.

A typed call's fields are private, so a call is built only by `Call::new`, which
builds the call of a `FunctionRef`'s type with no clause and no option, or by
`ScalarCall::builder`, `AggregateCall::builder` and `WindowCall::builder`, which
also take the options and the clauses, each at its default where the call sets
none. Each checks the call against its function's canonical declaration once it
is complete, at `build()` (rule 2), and refuses with a typed `CallError`:

- the arguments, as its `CallCheck` says. `Full` checks their number, a literal
  at a constant or an enum parameter (a leaf that holds an untyped literal is
  one), and an enum value that the parameter lists. `Call::new` takes it, and
  every crate that builds a call builds it with `Full`, reading a manifest
  included: an enum or a constant argument is an inline literal of an entry's
  tree, so reading checks a call as parse does (ADR 0089, rule 5; ADR 0088,
  rule 9).
- each clause against its `ClauseRule`: a clause that the declaration marks
  `Refused` or `NoEffect` is refused where the call writes it, and one that it
  marks `Required` where the call does not.
- a window call's `Over`, against SQL's rules for a window
  frame (`Over::check`, `CallError::Over` with its `OverError`): a start of
  UNBOUNDED FOLLOWING, an end of UNBOUNDED PRECEDING, an end whose kind comes
  before the start's (`CURRENT ROW AND 1 PRECEDING`), a RANGE window frame with
  an offset and not exactly one ORDER BY key, and a GROUPS window frame with no
  ORDER BY are refused, as Calcite's `SqlWindow` and DataFusion's `WindowFrame`
  refuse them. The check compares the kinds of the bounds, not their offsets, so
  `5 FOLLOWING AND 2 FOLLOWING` is built, as PostgreSQL allows.
- the calls that its operands hold, at any depth, as DataFusion's
  `check_aggregate_and_window_nesting` does: an aggregate call's arguments,
  FILTER and order keys hold no aggregate call (`AggregateInAggregate`:
  `sum(1 + sum(x))`) and no window call (`WindowInAggregate`), and a window
  call's arguments, partition and order keys hold no window call
  (`WindowInWindow`). A window call can hold an aggregate call:
  `first_value(sum(x))` is built.

Each constructor drops a setting at its option's canonical value, so a call holds
no option where the canonical values apply.

A call keeps its `CallCheck`, so `with_new_children()` checks it again by the
same check. The check is a mode of the constructor, and no part of the call's
equality. A `CallError`'s message names a clause as SQL writes it (`DISTINCT`,
`WITHIN GROUP`) and an enum value as its text.

A call of fewer arguments than its function takes is refused, and the refusal
says how many the function takes: "`greatest` takes one argument or more",
"`coalesce` takes two arguments or more" (`CallError::TooFewArguments`). Each
argument binds to its parameter in order. A required parameter takes one; so
does a parameter that an earlier enum argument conditions, where that argument
says so. An optional parameter, and one that a type conditions, which only
typing knows, takes one where the call has an argument left beyond those that
the later parameters take; the variadic tail takes the rest. Typing reads each
argument's parameter by the same binding, and `FunctionRef::parameters()` gives
the canonical declaration's, by which parse reads each argument of a list in
its parameter's form; `FunctionRef::arguments_by_position` gives a list's
arguments with the declared default of each parameter that an earlier enum
argument says the call writes and that the list leaves out. A call's mapping
names parameters by their names in any order: `FunctionRef::parameter(name)`
gives the one that a key names, in whose form parse reads its argument, and
`FunctionRef::arguments_by_name` binds its arguments in declaration order,
writes a skipped parameter's declared default where a later named one follows,
and the default of one that an earlier argument's value says the call writes,
and refuses a name that no parameter has (`CallError::UnknownParameter`), a
parameter that the call writes, no argument names and that declares no default
(`MissingParameter`), and one that it does not write and an
argument names (`UnwrittenParameter`) (ADR 0089, rule 4). A call holds each
option once, so each constructor refuses an option that the call sets twice
(`CallError::RepeatedOption`), and no constructor checks an option's name or
value. `Call::check_options` refuses an option set twice by the same check, then
checks each one that the call sets against its canonical declaration's options,
and refuses an option name that it does not have (`CallError::UnknownOption`) or
a value that the option does not take (`CallError::OptionValue`) (ADR 0088, rule
3); in a manifest, `validate()` runs it (rule 9).

**The leaves.** An `UntypedLiteral` is a literal written with no tag, which holds
its parse default until compile types it (ADR 0085, rule 2). `SemanticLeaf` is a
semantic expression's leaf: `Untyped`, an untyped literal, and `Field`,
`Dimension`, `Measure` and `Metric` over a `NamedReference`, each typed reference
with its optional accessor (rule 10). `PhysicalLeaf` is a semantic mapping's leaf:
`Untyped`, an untyped literal; every other part of a mapping's tree is a node of
the tree. Each gives its untyped literal through `Leaf::untyped()`.

**The name path.** `NamedReference` is a name path of one part or more, which
names a column or a semantics (the four semantic leaves), as Calcite's
`SqlIdentifier` names a column, a table or a function. Its last part, `name()`,
is the name (rule 9). `new` gives `None` for no part, which has no name, and for
an empty part, which names nothing; reading refuses either. `unqualified(name)`
is the path of one part, which no source alias qualifies, and
`qualified(alias, name)` the path of two, a column under its source alias; each
gives `None` for an empty part, as `new` does, so no constructor builds a path
that holds one, and neither cuts a name at a dot (rule 12). Compile and
`validate()` refuse a stored column that is not two parts; a table's three names
are the scan's (ADR 0067), and no reference holds them. Its serde form is the
array of its parts, `["ps0", "amount"]`.

**The literal.** `Literal` has one variant per canonical data type, with
`Null(DataType)` always typed, and `data_type()` reads the type from the variant
(rule 4). `IntoLiteral` gives a Rust value the literal of its type: a `bool`, an
`i32` an `integer`, an `i64` a `bigint`, an `f64` a `double`, and a string a
`string`. `Literal::parse(text, to)` reads the literal of type `to` from its SQL
text (`true`, digits, ISO text for a date, a time, a timestamp or an interval),
and refuses a text that is no value of `to`, a binary or an array, and a literal
that does not fit `to`. Two public checks serve parse and typing, each refusing
with a typed `LiteralError`:

- `check_fit()`: whether the value fits its type (rule 5). It refuses a type
  that is not valid; a decimal value of more than 38 significant digits, and one
  with a digit that its precision or scale would cut; a time, a timestamp or a
  day-second interval with more digits after the second than its precision; a
  timestamp outside the range that its precision holds in 64 bits (`9999-12-31`
  at precision 9); a timestamp with a zone; an interval whose months, days or
  whole seconds, counted toward zero, pass 32 bits, or its units 64 at its
  precision; a value longer than its length; and a date, a time, a timestamp,
  an interval (an ISO 8601 duration) or a uuid that is not text of its type.
  These texts are read by hand.
- `read_as(to)`: the literal of type `to` that the value reads as with no loss
  (rule 6). A null reads as a null of any type, a string as text of the type, and
  a number as any number type through its shortest exact text, as a `Double`
  that takes a decimal type is read: `Double(1.5)` as `decimal(18,2)` is
  `Decimal { "1.5", 18, 2 }`, and `Double(1.125)` is refused. A string and a
  number's text are read by `Literal::parse`. A number has no reading as a
  string. The literal it gives passes `check_fit()`.

The fold reads a third, crate-private (`literal/cast.rs`): the value that the
canonical CAST gives a literal (ADR 0086, rule 10), `read_as`'s for a NULL and a
string, and Arrow's cut, rounding and float conversion for a number.

`held_value()` gives the numbers that a date, a time, a timestamp, a decimal, an
interval or a uuid holds (ADR 0086, rule 3), read from its text, as a
`HeldValue`: a date's days since 1970-01-01, a time's or a timestamp's units of
its precision, a decimal's digits at its scale as one `i128`, an interval's
months, days and units of its precision, and a uuid's 128 bits. A literal that
holds its value itself gives `None`, and one that `check_fit()` refuses its
error.

A literal's serde form is one key, its type's canonical name, whose value holds
its value with its length, precision or scale: `{"bigint":5}`,
`{"decimal":{"value":"1.5","precision":18,"scale":2}}`, `{"null":"bigint"}`,
`{"array":{"element":"string","values":[{"string":"a"}]}}` (ADR 0086, rule 7;
ADR 0089, rules 8 and 9). A real's and a double's value is its shortest text, a
string. Both ways are written by hand: each refuses what `check_fit()` refuses,
so what serde writes always reads back, and a literal read from JSON always
fits its type; and a `binary` and a `binary(n)` share the tag `binary`, so
reading tells them apart by the value.

**The kinds.** `BinaryOpKind`, `UnaryOpKind`, `LikeKind`, `CastFailure`,
`WindowFrameKind`, `WindowBound`, `SortDir`, `NullOrdering` and `NullTreatment`
write their names in snake case (`safe_divide`, `i_like`,
`{"asc":{"nulls":"first"}}`, `ignore_nulls`), and a `WindowFrame` its `kind`,
`start` and `end`. A `WindowFrame`, a `WindowBound` and a `SortDir` refuse a key
that they do not have, as the JSON form's `over` and order key do.

**The JSON form.** `JsonForm::new(&expr, dialect)` writes a tree of
`Expr<L>` in the JSON form of a document in `dialect`, and
`JsonReader::new(dialect)`, a `DeserializeSeed` because serde's `Deserialize`
takes no dialect, reads it (ADR 0088, rules 1 to 3, 8, 9 and 13). Each node
but a leaf is `{"type": <tag>, "content": <body>}`: `literal` holds the
literal's form, `column` the name path, `is_null` its operand, and each other
operation node an object of its fields, `binary_op`'s `op`, `left` and `right`.
A CASE's body is the YAML's own shape: under `whens`, a `{"when":…,"then":…}`
object for each WHEN, and its ELSE under `else`, which is not written where the
CASE has none. A CASE nests in any `when`, `then` or `else`. A leaf is written
in its own form, with no tag of the tree: a manifest's link is its expression
id, in UUID text. A reader reads an object whose `type` is a node's tag as that
node, whatever the order of its keys, and any other value as the leaf, so no
leaf's form holds a node's tag.

The reader reads the form with serde alone, from any self-describing
deserializer, and reads no value as its JSON text first. It reads each value as
it comes, and holds entries back only where it cannot yet know what reads them:
the entries of an object before a `type` that names a node's tag, such as a
`content` that comes first, and an object with no such `type`, which is the
leaf's form. It holds them in their order, two equal keys too, and reads them
back with the error of the document's deserializer, so serde_json places a
refusal in the document. Each object refuses a key that it writes twice.

A call's tag is its function type, and its body holds only that type's fields,
none at its default (`false`, `null`, an empty list, no options):

| Tag | Body fields |
|---|---|
| `scalar` | `name`, `args`, `options` |
| `aggregate` | `name`, `args`, `options`, `distinct`, `filter`, `order_by` (ORDER BY in the call, or WITHIN GROUP), `nulls` |
| `window` | `name`, `args`, `options`, `nulls`, `over` (`partition_by`, `order_by`, `window_frame`) |

`sum(amount)` is
`{"type":"aggregate","content":{"name":"sum","args":[{"type":"column","content":["ps0","amount"]}]}}`.
`name` is the declaration's name in the document's dialect, never an alias, and
writing refuses a call whose function the dialect does not have
(`JsonFormError::NoNameInDialect`). `options` is an object of each option's
name and its value's name, `{"null_handling":"IGNORE_NULLS"}`, and reading
refuses one that names an option twice (`RepeatedOption`). Reading finds
each call by its name in the document's dialect with `Registry::find`, refuses
an unknown name (`UnknownFunction`) and a tag that is not the function's type
(`NotItsFunctionType`), and builds the call with its constructor's full check
(`CallCheck::Full`), as parse does: it refuses a wrong number of arguments, an
argument that is not a literal at an enum or a constant parameter, an enum
value that the parameter does not list, a clause that the declaration marks
`Refused` or `NoEffect`, no clause where it marks one `Required`, a window
frame that SQL refuses, and a call that holds a call that it cannot hold. A
literal, a name path and a data type take their own serde forms, so `Unknown` is
refused both ways.

**The operations.** An operation is each kind of operation node that has a
declaration: each `BinaryOpKind`, `UnaryOpKind` and `LikeKind`, and `Between`,
`InList` and `IsNull`, which `OperationKind` names (ADR 0092, rule 1). `Case` and
`Cast` have none: code holds their rules. Each kind's `operation()` gives its
`Operation`, the canonical `OperationDeclaration` and DataFusion's entries, and
`on(dialect)` gives the dialect's declaration, the canonical one where the
dialect has none, or `None` where it has no such operation. A declaration holds
a function declaration's `name` (the generic name; on a dialect, the engine's
own), `aliases` (other names of the operation), `Signature` (each operand's type
family), `ReturnRule`, `Nullability` and options, but no clauses and no
volatility. An operation node has no option slot, so each option takes its
canonical value. One macro declares the three kinds' enums, each with every kind
it has, so `OperationKind::from_name`, which finds an operation by its canonical
name or an alias, as written, reads no list of its own.

| Operation | Operands | Result | NULL | Options |
|---|---|---|---|---|
| `add` | two numbers; or a date, a timestamp or a time with an interval of any interval family, in either order | `Sum` | `Mirror` | `overflow`, canonical `ERROR` |
| `subtract` | the same, with the date, the timestamp or the time first | `Sum` | `Mirror` | `overflow`, `ERROR` |
| `multiply` | two numbers | `Product` | `Mirror` | `overflow`, `ERROR` |
| `divide`, `mod` | two numbers | `Quotient`, `Mod` | `Mirror` | `on_division_by_zero`, `ERROR` |
| `safe_divide` | two numbers | `Quotient` | by `on_division_by_zero`: `NULL` gives `Always`, `ERROR` `Mirror` | `on_division_by_zero`, `NULL` |
| `negate` | a number | `SameAs(0)` | `Mirror` | `overflow`, `ERROR` |
| `eq`, `not_eq`, `lt`, `lt_eq`, `gt`, `gt_eq` | two operands of one family | boolean | `Mirror` | |
| `is_not_distinct_from` | two operands of one family | boolean | `Never`: true where both operands are NULL, false where one is | |
| `and`, `or` | two booleans | boolean | `Kleene` | |
| `not` | a boolean | boolean | `Mirror` | |
| `like`, `not_like`, `ilike`, `not_ilike` | two strings | boolean | `Mirror` | |
| `between` | three operands of one family | boolean | `Kleene` | |
| `in` | a value and items of one family | boolean | `Kleene` | |
| `is_null` | any | boolean | `Never` | |

DataFusion's declarations are rule 8's: the name `ilike` for `ILike` and
`NotILike`; `on_division_by_zero` with the one value `ERROR` on `SafeDivide`; and
`overflow` with the one value `SILENT` on `+`, `-`, `*` and negate. No
DataFusion entry is `None`, and no operation has a Substrait declaration.

**Typing.** `type_tree` is the one typing entry (ADR 0089, rule 11). It takes
the tree; an `InputTypes`, which the caller implements, for each column's and
each leaf's `Typed` (a type, whether it can be NULL, and its family, which
`Typed::new` gives from the type); the document's dialect; a type system; and
the context's type. It gives the tree with each untyped literal written as the
`Expr::Literal` of its type and each default that an argument's type says a call
takes written in the call (ADR 0091, rule 4), and the tree's `Typed`, whose family is the one
that a result with no type keeps; or a typed `TypeError`, which names a node
with no type by that family (`TypeOrFamily`: "a value of the number family with
no type"), never as `unknown`. The walk is `expr/typing.rs`'s, and each rule
sits beside the type it rules on, as Calcite and DataFusion keep the walk apart
from the rules: a signature's, a result's and a NULL rule's in
`functions/signature.rs`, a parameter's in `functions/mod.rs`, a derivation's in
`datatype/type_system.rs`, a cast's in `datatype/family.rs`, and a literal's in
`expr/literal.rs`. A rule takes types, each with the family that it keeps, and
no node; the walk gives each node's, and names a rule's refusal as a
`TypeError`.

- **An operation and a call** are typed by their declaration in the document's
  dialect: the canonical one for `canonical`, and `on(dialect)`'s for a dialect,
  which refuses with `NotInDialect` where the dialect has no such operation or
  function (rule 12). An operation's operands take its parameters in order; a
  call's arguments take theirs by the constructors' own binding, and where they
  do not fit the dialect declaration's signature, typing refuses the call with
  `Call`. Errors name the canonical name.
- **Each argument is checked** by its parameter's type, and refused with
  `Argument`: an operation's by type family only (ADR 0092, rule 2); a
  function's by its exact type, its family, one type at any precision, or the
  family of its `T` (ADR 0086, rule 8). The arguments at `T` take one exact type
  (`TypeVariable`), so `coalesce(qty, region)` is refused. Then the constraints
  across the arguments: one family for all (`OneFamily`) or one of the declared
  rows (`Families`), refused with `Arguments`; and an enum value that another
  argument's type refuses (`EnumRefused`: `extract`'s `hour` on a `date`, ADR
  0087, rule 14). A parameter that an earlier argument's type conditions, as
  `date_trunc`'s timezone for a timestamp, is written where that type holds and
  only there: it is refused where it is written beside another type
  (`ParameterRefused`); where it is left out beside that type, typing writes the
  default that it declares, `date_trunc`'s `'UTC'`, and refuses one that declares
  none (`ParameterRequired`; ADR 0091, rule 4). An argument with no type is checked by the family it
  keeps, here and at `T`: `upper(score * price)` is refused, as `upper(score)` is
  (ADR 0089, rule 3). A column that compile did not observe has no family, so it
  is not checked (rule 15).
- **A call's clauses** are typed after its arguments: an aggregate call's FILTER
  as a boolean, and its WITHIN GROUP's keys at the type that the declaration
  names, each refused with `Clause`; a FILTER with no type is checked by the
  family it keeps, so `sum(qty) FILTER (WHERE score * price)` is refused; its
  ORDER BY and a window call's partition and order keys take no context. A
  window call's RANGE window frame with an offset refuses an order key of the
  string or the boolean family, which has no distance between two values
  (`RangeOrderKey`; Calcite's `orderByDataTypeProhibitsRange`), by
  `WindowFrame::check_order_key`.
- **The result type** is the return rule's: `Fixed`; `Var`, the type of `T`;
  `SameAs`, its argument's, where an argument of type `Unknown` makes either
  `Unknown`, as a CASE's branch does: `coalesce(x, 1)` is `Unknown` where
  compile did not observe `x`; `Promote`, the bigger argument's by the widening
  order, or with a decimal the type system's decimal union. With a decimal
  operand, `+` and `-`, `*`, `/` and `%` take the type system's decimal sum,
  product, quotient and remainder, and an integer operand enters as its digits
  at scale 0; otherwise the bigger operand by the widening order; a date, a
  timestamp or a time with an interval keeps its own type. `SumOf`, `AvgOf`,
  `DecimalScaleZero` and `RoundOf` are the type system's, from the first
  argument; the last two give a type other than a decimal itself. Where neither
  operand widens to the other, as a double and a decimal, the result has no
  type, and typing gives `Unknown` (rule 3). A result with no type keeps the
  family of the arguments that its return rule reads, and the node above checks
  it by that family: the family of the argument whose type it is (`SameAs`'s,
  the first for a derivation of one argument, and the date, the timestamp or
  the time beside an interval); the one family of those at `T` for `Var`; and
  else the one family of every argument.
- **NULL** follows the declaration's rule for the values that its options take,
  a call's own or else the canonical ones: NULL if an argument can be (`Mirror`,
  and `Kleene` alike), if one at the named parameters can be (`MirrorArgs`), or
  only if every one can be (`AllArgs`); always (`Always`, and `EmptyGroupNull`,
  since typing does not know whether a group is empty); never (`Never`, IS
  NULL); and by an option's value (`ByOption`: `SafeDivide`, whose DataFusion
  declaration takes only `ERROR`, and `concat`) (rule 7; ADR 0091, rule 8).
- **An untyped literal** takes the type that binds the argument it fills
  (rule 4): at a parameter of an exact type, that type; at `T`'s, the type of the
  first other argument at `T` that has one; at any other, where the declaration
  binds its arguments to one type (a result it derives from them, or arguments of
  one family), the type of the first other argument that has one; under
  `Families`, only where a row names that argument's family for both, as `+`'s
  two numbers, so a date gives the operand beside it no type. Where no argument
  that binds it has a type, the node's own context, if its result is theirs;
  otherwise none, and it keeps its parse default, as beside an argument of type
  `Unknown` or at a parameter that names only a family: `sum(1)` is
  `BigInt(1)`. It reads as that type with no loss through `Literal::read_as`, or
  typing refuses it (ADR 0085, rule 6). A tagged literal keeps its type.
- **CASE** takes every WHEN as a boolean, with the boolean as its context, and
  refuses another type with `CaseWhen`, and a WHEN with no type by the family it
  keeps: `CASE WHEN score * price THEN …` is refused. `is_condition` is that
  check, and a FILTER's, which takes `BOOLEAN`; compile calls it on each tree that
  it types as a condition. CASE takes every THEN and the
  ELSE at one exact type, which is its result, and refuses two with
  `CaseBranches`; a branch with no type is checked by the family it keeps, and
  one with no family is not checked; either makes the result `Unknown`, which
  keeps the branches' one family. It can be NULL by a branch, or where it has no
  ELSE. **CAST** refuses an input type that `casts_to()` refuses, and an input
  with no type by the family it keeps: it casts where some type of the family
  casts to the target, so `CAST(score * price AS date)` is refused, and the cast
  to a string is not; an input with no family is not checked (ADR 0089, rules 14
  and 18). Its result is its target, NULL by its input or where its failure
  gives NULL. `check_cast` is that one check, which compile also calls, with the family
  that typing gives, on the cast of a tree to its semantics' declared type.

**The fold.** `Expr::fold` is the partial evaluation of a tree over its
literals (ADR 0089, rule 21). It folds each operand, then the node. Its scope is
a closed list: the parts whose value the planner's pruning of a Unionset branch
reads, or whose value lets compile store a metadata-determined signature as a
literal. It puts in each such part's place the value that its literals decide;
every other part stays as it is, for the engine, and the parts inside it fold.

- AND and OR are `Kleene` (ADR 0092, rule 7), so `x = 'a' AND FALSE` is FALSE
  and `TRUE OR NULL` is TRUE. NOT and IS NULL take a literal operand.
- A comparison gives NULL where an operand is NULL, so `x = NULL` is NULL.
  Otherwise it is evaluated over two booleans, two integers of any width, two
  `string` or `varchar(n)` values in any mix, which compare by code point, or
  two dates. BETWEEN is `x >= low AND x <= high`, so `5 BETWEEN NULL AND 3` is
  FALSE, and IN is `x = a OR x = b …`.
- A CASE gives the THEN of its first WHEN that is TRUE, past each WHEN that is
  FALSE or NULL; where none is left, its ELSE, or, where it has none, NULL of the
  type of a THEN that is a literal. It stays where a WHEN before the chosen one
  is no literal.
- Integer arithmetic (`+`, `-`, `*`, `/`, `%`, `SafeDivide` and negate) is
  evaluated only where each operand is a literal of an integer type, because
  their types give its type, in the bigger operand's type, by the widening order
  (ADR 0092, rule 3), where no overflow occurs. A division truncates toward
  zero. A NULL operand gives NULL, so `1 + NULL` is NULL of `integer`, but
  `x + NULL` stays, and so does a `SafeDivide` with a NULL operand. A `/`, a
  `%` or a `SafeDivide` by zero stays.
- A LIKE gives NULL where an operand is NULL. A CAST of a literal gives the
  value that the canonical CAST gives (ADR 0086, rule 10): NULL of its target
  type for NULL; a string as the literal of the target type that it reads as
  (ADR 0085, rule 6), so `CAST('2026-05-19' AS date)` is the date; a number as a
  number type, as Arrow casts it, cut toward zero to an integer, rounded half
  away from zero to a decimal, so `CAST(1.7 AS integer)` is 1; and a literal of
  its target type itself.
- Every other part stays: a comparison of other kinds, such as two decimals,
  two `char(n)` values or an integer and a decimal; a LIKE of two strings that
  are not NULL; arithmetic on a `real`, a `double`, a decimal or a date; a CAST
  of a literal that the canonical CAST gives no value for, such as
  `CAST('2026-13-45' AS date)`, of a number to a type that is no number, such
  as `CAST(5 AS string)`, or of a literal that is no NULL, string or number to
  another type; and every function call. Every operation is
  `Immutable` (ADR 0092, rule 1), so the fold evaluates nothing that is not.

`Expr::fold_with` is the same fold, where a function `literal_of` gives the
literal that a leaf stands for, such as a link to an entry whose tree is a
literal (ADR 0089, rule 23). Each part above reads such a leaf as that literal,
so `dataset_name = 'google'` is FALSE where `dataset_name` links an entry whose
tree is `'google_ads'`. The fold puts in a node's place only a value that it
evaluates, or the branch that a CASE chooses, so a leaf under a part that it
leaves stays that leaf: `upper(dataset_name)` and `dataset_name LIKE 'g%'`
stay. `Expr::fold` reads no leaf as a literal.

Every public enum of `expr` but `NoLeaf` and `NoAggregate`, which have no
value, is `#[non_exhaustive]`, and so are `WindowFrame` and
`OperationDeclaration`, whose fields may grow; another crate builds a window
frame with `WindowFrame::new` (ADR 0007).

---

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `schemars` | | `JsonSchema` for `DataType`, the stored form, and `DataType::resolution_schema`, for the model's JSON schema. |

## Dependencies

- `semstrait-common` -- `Dialect`; `Tree` for `Expr`; `ValidateError` for `Expr`; and `Diagnose` and `Severity` for `SealFault`
- `bon` -- the calls' builders, `ScalarCallBuilder`, `AggregateCallBuilder` and `WindowCallBuilder`, derived from each call's `new`
- `serde` -- the serde of `DataType`, `Literal`, `NamedReference`, the operator kinds, the window types and `NullTreatment`, and the tree's JSON form
- `thiserror` -- `Display` and `std::error::Error` for each error of the crate
- `schemars` (optional) -- see the features table
