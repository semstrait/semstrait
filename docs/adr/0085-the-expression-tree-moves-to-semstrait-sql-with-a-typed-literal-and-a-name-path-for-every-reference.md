---
number: 85
title: The expression tree moves to semstrait-sql, with a typed literal and a name path for every reference
status: accepted
date: 2026-10-08
source:
- CONTEXT.md
- docs/adr/0001-invert-model-ir-dependency-edge.md
- docs/adr/0002-single-leaf-parameterized-expression-tree.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md
- docs/adr/0078-a-source-alias-is-its-source-roles-letter-and-its-index-in-the-manifest.md
- docs/adr/0082-a-physical-read-types-an-unobserved-column-by-the-semantics-mapped-to-it-and-compile-writes-that-type-into-the-schema.md
- '`Expr`, `RowExpr` and `Leaf` in `crates/semstrait-sql/src/expr/node.rs`; the leaf sets in `crates/semstrait-sql/src/expr/leaf.rs` and `crates/semstrait-ir/src/expr.rs`; `Literal` in `crates/semstrait-sql/src/literal/mod.rs` and `NamedReference` in `crates/semstrait-sql/src/expr/reference.rs`; `SemanticMappingValue` in `crates/semstrait-model/src/model/types/entities/mapping.rs`; `Literal` lowering in `crates/semstrait-manifest/src/compile/expressions/mod.rs`'
- '`datafusion-expr@55.0.0:src/expr_schema.rs:142`, `datafusion-common@55.0.0:src/column.rs:30-37`, `datafusion-sql@55.0.0:src/expr/value.rs:378-435`; `spark@v4.2.0` `analysis/unresolved.scala:295`, `expressions/namedExpressions.scala:285-291`, `expressions/literals.scala:418`'
- '`calcite@calcite-1.42.0` (`c01f6b5`): `core/src/main/java/org/apache/calcite/sql/SqlUnknownLiteral.java:28-36`, `sql/validate/SqlValidatorImpl.java:7120-7137`'
- the human's rulings of 2026-10-08, 2026-10-13 and 2026-10-15
- the human's rulings of 2026-10-17
links:
- target: 1
  kind: amends
- target: 2
  kind: amends
- target: 57
  kind: amends
- target: 76
  kind: amends
- target: 46
  kind: amends
- target: 65
  kind: amends
- target: 78
  kind: amends
- target: 94
  kind: amendedby
---

# The expression tree moves to semstrait-sql, with a typed literal and a name path for every reference

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0094](0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md): rule 3's row ("`Grain`; the builders …; the aliases … | stay in `semstrait-model`"): `Grain` moves to `semstrait-common` as `Granularity` (rule 3); the builders and the aliases stay.

## Context and Problem Statement

A map makes a new crate,
`semstrait-sql`, which depends only on `semstrait-common`. `semstrait-model`,
`semstrait-ir` and `semstrait-manifest` each depend on it directly, and expressions
are centralized in `semstrait-sql` (ADR 0090, rule 6). ADR 0001 gives these types to `semstrait-model`:
"`semstrait-model` owns the shared authoring vocabulary — the `Expr<L>` family,
leaf sets, support enums, `Literal`/`ColumnRef`, `DataType`/`Grain`, and the
`CanonicalFn` name — and `semstrait-ir` depends *upward* on it" (*Decision
Outcome*). So the move needs a decision on what the tree holds once it leaves
`semstrait-model`.

Today there is one tree, `Expr<L, A>` (ADR 0002). Each stage gives it a leaf set
`L`, and each leaf set repeats the column and the literal:

| Leaf set | Crate | Variants |
|---|---|---|
| `UnresolvedSemanticLeaf` | model | `Literal`, `Column`, `Field`, `Dimension`, `Measure`, `Metric` |
| `UnresolvedPhysicalLeaf` | model | `Column`, `Literal` |
| `PhysicalLeaf` | ir | `Column`, `Literal`, `Parameter` |

Three more facts bear on the move:

- **`semstrait-model` has two literal types.** A semantic mapping's `lit:` arm holds
  `LiteralValue`, and an expression holds `Literal`. Only manifest compile reads
  `LiteralValue` (`lower_literal`), and it refuses a `null`, a `decimal` and a
  `timestamp` value. The two parsers spell their tags differently (`int` and
  `integer`), and the rule "integer → `Long`, float → `Double`" is written twice.
- **The column is one string.** `ColumnRef` is `pub struct ColumnRef(pub
  String)`. In a join condition compile writes `<alias>.<physical column>` into
  it, and a reader "cuts it at the **first** dot" (ADR 0057, *Decision
  Outcome*; ADR 0075 keeps the physical column until compile writes semantic
  leaves). In a Dataset's tree the string is the physical column name, with no
  qualifier, and "it follows from the expression's **attachment**" whether a
  leaf is qualified.
- **A column's type is not always known in the manifest.** A catalog observes
  it, or, on a model-fed schema, compile writes the type that the semantics
  mapped to the column give (ADR 0082, *Compile writes the read type into the
  schema*); a column that only an expression reads has none.

The engines draw these lines as follows:

| | Column leaf | Where the column's type comes from | Literal |
|---|---|---|---|
| DataFusion 55 | `Column { relation: Option<TableReference>, name, spans }`, with no type | the input schema, at each use: `Expr::Column(c) => Ok(schema.data_type(c)?.clone())` | `ScalarValue`, which carries its type |
| Spark 4.2 | `UnresolvedAttribute(nameParts: Seq[String])`; after analysis, `AttributeReference(name, dataType, nullable, metadata)(exprId, qualifier)` | copied from the child plan's output | `Literal(value: Any, dataType: DataType)` |

Both keep the qualifier apart from the name in the tree. Only SQL text joins
them.

The calls and the operation nodes move with the tree. ADR 0091 decides a call
and a function's declarations, and ADR 0092 an operation node's declaration.

## Considered Options

- **The tree with `Literal` and `Column` as nodes, a literal typed by its
  variant, and a name path for every reference** (chosen).
- **Move `Expr<L, A>` unchanged.** Each leaf set keeps repeating `Column` and
  `Literal`. `semstrait-sql` then has no column and no literal node, and every
  tree it builds borrows a leaf set from another crate.
- **A tree that has no `L`**. A semantic
  leaf in a physical tree becomes a runtime check. Today the type system does
  not let anyone build it (ADR 0002).
- **A placement: `semstrait-sql` above `semstrait-model`, with a
  function slot `F`**. The slot exists because `semstrait-model`
  could not name a resolved function. The chosen placement puts the crate below it, which removes
  that reason.
- **A typed column, `Column { name, data_type }`**, as Spark's
  `AttributeReference` has it. A column that only an expression
  reads has no type in the manifest (ADR 0082). A plan would hold each type twice, in the read's
  schema and in each leaf, and the two must agree.
- **A literal as a value plus an optional type**, `Literal { value: Decimal("1.5"),
  data_type: None }`. The value says "decimal" while the type says nothing, so
  the two parts contradict each other. The human: literals "must be based on
  Canonical DataType annotation".
- **Keep `LiteralValue` for a mapping's `lit:` arm.** One authored value then
  has two carriers and two width rules.
- **Keep the column as one string cut at the first dot** (ADR 0057's form). A
  physical column named `a.b` needs ADR 0078's "An alias holds no dot, so ADR
  0057's cut at the first dot still returns the whole alias" (*An alias is
  `ss`, its source role's letter and its index*), and a reader must still cut
  the string to route. ADR 0050 refused "Extending `ColumnRef` with a source or
  side qualifier" because it "forces a `semstrait-model` *and* `semstrait-ir`
  change" and "puts join-scope knowledge into a leaf that ADR 0046 deliberately
  resolves against one source's schema" (*Considered Options*). Neither ground
  holds now: the type leaves both crates, and each leaf names the one source
  whose schema it resolves against.

For an untagged literal:

- **An untagged literal as a leaf of the two unresolved trees, which compile
  resolves**, as Calcite's parse tree holds a literal whose type is not yet known
  (chosen).
- **A mark on the literal node, `Expr::Literal { value, tagged }`**, to tell an
  untagged literal from a tagged one. Every stage then carries a mark that only
  compile reads.
- **No mark: a tag that names the parse default binds nothing**, as in Spark and
  DataFusion, which type every literal at parse. Then `amount > {lit: {string:
  "5"}}` gives `Decimal { "5", 18, 2 }` and `qty: {lit: {double: 2.0}}` gives
  `Integer(2)`, against rule 6 and ADR 0092 rule 4.

For an untagged whole number past `BigInt`:

- **Refuse it, with a message that advises a tag that declares its type**
  (chosen).
- **An untagged `decimal` with precision 38 and scale 0.** The writer writes it
  as its digits with no tag, `{"lit":99999999999999999999}`, and serde_json
  reads that number back as a float, `Double(1e20)`, so the literal does not
  keep its type. The human: "I think we must preserve type".

## Decision Outcome

In short:

- `semstrait-sql` holds the tree. `Literal` and `Column` are nodes of the tree,
  and `L` holds only a stage's own leaf. An untagged literal is a leaf of the two
  unresolved trees until compile types it.
- A literal is one variant per canonical data type. A tagged literal keeps its
  tag's type. An untagged literal gets a faithful default at parse, and compile
  assigns it the type of its context where the value reads as that type with no
  loss, and refuses it otherwise. A literal keeps its type when the semantic
  model is written as YAML or JSON and read again.
- Every reference is a name path, `NamedReference`. In the manifest, the first
  part of a column is the alias of the physical source or the semantic source
  that holds it.

### The tree

1. `semstrait-sql`'s `expr` package holds the tree:

   ```rust
   pub enum Expr<L: Leaf = NoLeaf, A: Aggregation = AggregateAllowed> {
       Literal(Literal),
       Column(NamedReference),
       Leaf(L),
       BinaryOp { op: BinaryOpKind, left: Box<Self>, right: Box<Self> },
       // UnaryOp, Cast, Case, InList, Between, Like and IsNull, as today's tree has them
       Function(Call<L, A>), // a call of a scalar, an aggregate or a window function (ADR 0091, rule 1)
   }
   pub type RowExpr<L> = Expr<L, NoAggregate>;
   ```

   - The 8 operation nodes keep their form ("Operators are
     operators"), and each but `Case` and `Cast` has a declaration (ADR 0092,
     rule 1).
   - `Function` replaces the `FunctionCall`, `Aggregate` and `Window` nodes (ADR
     0091, rule 1), and the `Coalesce` and `NullIf` nodes, which become calls of
     the functions `coalesce` and `nullif` (ADR 0092, rule 6).
   - `Literal` and `Column` are nodes, not leaves of each leaf set. The human: "Literal is typed so type system wise it looks more natural
     PhysicalLeaf:Literal(Literal::Float) = Expr::Literal". An untagged literal is
     the one literal form that is a leaf (rule 2).
   - `Leaf` is today's `ExprLeaf`, renamed. `Aggregation` is a trait that only
     `AggregateAllowed` and `NoAggregate` implement.

2. `L` holds only a stage's own leaf:

   | Tree | `L` | Alias |
   |---|---|---|
   | every authored position that holds a semantic expression today: a semantics' `expr:`, a Measure's or a Metric's `filter:`, a named Filter, a relationship's `filter:` | `SemanticLeaf` | `UnresolvedSemanticExpr = RowExpr<SemanticLeaf>`, which `semstrait-model` holds (ADR 0089, rule 6) |
   | a semantic mapping's `expr:` arm | `PhysicalLeaf`, whose one variant, `Untyped(UntypedLiteral)`, holds an untagged literal | `UnresolvedPhysicalExpr = RowExpr<PhysicalLeaf>`, which `semstrait-model` holds |
   | the plan | `Parameter`, which `semstrait-ir` holds (ADR 0090, rule 1) | `PhysicalExpr = Expr<Parameter>`, which `semstrait-ir` holds |
   | a tree that a manifest entry stores | `Link`, the manifest's own: a link to another entry (ADR 0088, rules 4 and 5) | `Expr<Link>` |

   Each tree but a stored expression admits the same leaves as today. A stored
   expression's link stands only where the authored tree references another
   semantics, and names the entry of that semantics' signature on the same
   semantic source. A stored expression admits no `Parameter` (ADR 0088, rules 4
   and 5). So ADR 0002's forbidden combinations stay unrepresentable.

   An untagged literal is a leaf of the two unresolved trees: an
   `UntypedLiteral(Literal)`, which holds its parse default (rule 5), under
   `SemanticLeaf::Untyped` in a semantic expression (rule 10) and under
   `PhysicalLeaf::Untyped` in a mapping's tree. A tagged literal is an
   `Expr::Literal`. So parse gives `amount > {lit: "5"}` and
   `amount > {lit: {string: "5"}}` two trees, which compile types apart: the first
   is `Decimal { "5", 18, 2 }` beside a `decimal(18,2)` column, and the second is
   refused (ADR 0092, rule 4). This is Calcite's structure: its parse tree holds a
   literal "whose type is not yet known" as a node of its own
   (`SqlUnknownLiteral`), its validator resolves it (`resolveLiteral`), and its
   resolved literal, `RexLiteral`, holds only a value and a type. Compile writes
   each untagged literal as an `Expr::Literal` (rule 6), and neither the
   manifest's `Link` nor the plan's `Parameter` holds one (rule 7).

3. The types move as follows:

   | Type | Goes to |
   |---|---|
   | `Expr`, `RowExpr`, `Leaf`, `NoLeaf`, `CastFailure`, `AggregateInRowExpr` and its `TryFrom` | `semstrait-sql` `expr` |
   | `WindowFrame`, `WindowFrameKind`, `WindowBound`, `SortDir`, `NullOrdering` | `expr` |
   | `NamedReference`, `SemanticLeaf`, `PhysicalLeaf`, `UntypedLiteral`, `DimensionAccessor`, `MeasureAccessor`, `MetricAccessor` | `expr` |
   | the `std::ops` impls, `ExprOperationExt` | `expr` |
   | `Call`, `ScalarCall`, `AggregateCall`, `WindowCall`, `Over`, `SortKey`, `NullTreatment`, `OptionSetting` (ADR 0091, rule 1) | `expr` |
   | `Literal`, `IntoLiteral` | `semstrait-sql` `literal` |
   | `Function`, `Declaration`, `FunctionType`, `Scalar`, `Aggregate`, `Window`, `AggregateFunction`, `Category`, `Signature`, `ReturnRule`, `Nullability`, `Volatility`, `OptionDecl`, `ScalarOption`, `AggregateOption`, `WindowOption`, `AggregateClauses`, `WindowClauses`, `ClauseRule`, `Additivity` (ADR 0091, rules 3 and 4) | `semstrait-sql` `functions` |
   | `FunctionRef` (ADR 0091, rule 1), `Aggregation`, `AggregateAllowed`, `NoAggregate` | `functions` |
   | `BinaryOpKind`, `UnaryOpKind`, `LikeKind`, and the operation declarations (ADR 0092, rule 1) | `functions` |
   | `DataType` and `TypeFamily` (ADR 0086), and `TypeSystem` with the canonical type system and Substrait's (ADR 0092, rule 3) | `semstrait-sql` `datatype` |
   | `Dialect` | `semstrait-common` (ADR 0091, rule 6) |
   | `Grain`; the builders `col`, `field`, `dim`, `measure`, `metric`, `lit`, `physical_col`; the aliases `UnresolvedSemanticExpr` and `UnresolvedPhysicalExpr` | stay in `semstrait-model` (the builders "for now"; the aliases by ADR 0089, rule 6) |
   | `Parameter`, `ParameterKey` | stay in `semstrait-ir` (ADR 0090, rule 1) |

   `expr` re-exports `Literal`, `IntoLiteral`, `FunctionRef`, `Aggregation`,
   `AggregateAllowed`, `NoAggregate`, the three kinds and the operation
   declarations, so every public path of these types and of those in `expr`
   is `semstrait_sql::expr::*`.

   These retire: `ColumnRef`, `LiteralValue`, `IntegerWidth`, `FloatWidth`,
   `UnresolvedPhysicalLeaf` (`semstrait-sql`'s `PhysicalLeaf` replaces it),
   `UnresolvedSemanticLeaf` (`SemanticLeaf` replaces it), `semstrait-ir`'s
   `PhysicalLeaf`, whose `Parameter` becomes `L` itself,
   `AggregationOp`, `AggregateKind` and `WindowFn`, whose functions become
   declarations like every other (ADR 0091, rule 3), and `CanonicalFn`, because
   the registry finds a function by its name as text (ADR 0087, rule 8).

### A literal is one variant per canonical data type

4. A literal's variant is its canonical data type (ADR 0086):

   ```rust
   pub enum Literal {
       Boolean(bool), TinyInt(i8), SmallInt(i16), Integer(i32), BigInt(i64), Real(f32), Double(f64),
       Decimal { value: String, precision: u8, scale: i8 },
       String(String), VarChar { value: String, length: u32 }, FixedChar { value: String, length: u32 },
       Binary(Vec<u8>), FixedBinary { value: Vec<u8>, length: u32 },
       Date(String), Time { value: String, precision: u8 }, Timestamp { value: String, precision: u8 },
       Interval(String), IntervalYearMonth(String), IntervalDaySecond { value: String, precision: u8 },
       Uuid(String),
       Array { element: DataType, values: Vec<Literal> },
       Null(DataType),
   }
   ```

   - `Literal::data_type()` reads the type from the variant. A null is always
     typed.
   - A variant holds the value and, where its type has them, the length, the
     precision or the scale, as a `decimal` holds its precision and scale.
   - Each variant takes its type's canonical name, as each `DataType` variant
     does (ADR 0086, rule 7): `char(n)` and `binary(n)` are `FixedChar` and
     `FixedBinary`.
   - An array literal holds its element type and its values (ADR 0086, rule 4).

5. **Parse** types each literal.

   - An untagged literal gets the faithful default for what the YAML holds. The
     model's tree is the unresolved stage, so this type is provisional. Parse holds
     it as the untagged leaf of its tree (rule 2).
   - A tagged literal gets the canonical type that its tag names: `{integer: 5}`
     is `Integer(5)`. A tag is a canonical type name, and an expression's `lit:`
     and a semantic mapping's `lit:` read one grammar (ADR 0089, rule 8). This
     type is the author's.
   - A precision, a scale or a length that a tag leaves out gets the default
     below.

   | Authored | Type |
   |---|---|
   | a YAML string | `String` |
   | a YAML integer | `BigInt`; a whole number past `BigInt` is refused, and the message advises a tag that declares its type, such as `{decimal: "99999999999999999999"}` |
   | a YAML float | `Double`; a float at or past 2^128, or at or below -2^127, is refused, and the message names `{double: …}`: the YAML reader reads a whole number past `u128` or `i128`, which has 39 digits or more, as the same float, so parse cannot tell the two apart |
   | a bare `null` | `Null(String)` |
   | a `decimal` with no precision or scale | precision 38, and the scale of its own digits after the point: `1.5` is `decimal(38,1)` |
   | a `decimal` with a precision and no scale | the scale of its own digits: `{decimal: {value: "1.5", precision: 10}}` is `decimal(10,1)` |
   | a tag's body, as a YAML number or as a string | read by the tag's type: `{decimal: 99999999999999999999}` is `decimal(38,0)`, as `{decimal: "99999999999999999999"}` is |
   | a `decimal` value with more than 38 significant digits | refused: it exceeds the maximum precision 38 |
   | a `decimal` value that does not fit the precision its tag gives, such as `{decimal: {value: "123.45", precision: 4}}` | refused: it would cut a digit |
   | a `time` with no precision | precision 9 |
   | a `timestamp` with no precision | precision 6, or 9 where its value writes more than 6 digits after the second |
   | a `timestamp` value that its precision cannot hold, such as `9999-12-31` at precision 9 | refused |
   | `{varchar: {value: "abc", length: 10}}` | `varchar(10)` |
   | `{varchar: "abc"}` | `varchar(3)`: the length of its own value, as a `decimal` takes the scale of its own digits |
   | `{char: "ab"}` | `char(2)` |
   | `{binary: {value: …, length: 16}}` | `binary(16)` |
   | `{uuid: "550e8400-e29b-41d4-a716-446655440000"}` | `uuid` |
   | `{interval_day_second: {value: …, precision: 6}}` | `interval_day_second(6)`; with no precision, 9 |
   | a value that is longer than its length | refused |

   No default cuts a digit. The human: "trancation for literal is not allowed".
   A scale of 38 would not hold `1.5`, and the bare `decimal` type's `(38, 9)`
   would cut `0.1234567891`. A YAML float holds about 17 exact digits, so
   compile reads it through its shortest exact text. An author who needs every
   digit writes the value as a string, bare (`"0.123456789012345678"`) or as the
   `decimal` tag's body (`{decimal: "0.123456789012345678"}`). A `binary(n)`
   value is written as a `binary` value is, and an `interval_year_month` or an
   `interval_day_second` value as an `interval` value is.

   A literal keeps the type that the author declares when the semantic model
   is written and read again, as YAML or as JSON (ADR 0089, rule 8). The
   human: "I think we must preserve type", and "if it is untagged - it can be
   just default ne - bigint for both ser/de , and and then if user overflows
   default untagged - error and advice to declare type for literal". So an
   untagged whole number is a `BigInt` when parse reads it and when the writer
   writes it, and a whole number that `BigInt` does not hold takes a tag that
   declares its type.

6. **Compile** checks each literal against its context. The human: "in model we
   can preserve what we have lit(value) and we derrive in parsing type, by
   checking what we recieved from yaml ( considering that eveyrthign in yaml is a
   string mostly ). So can assign faithfull default -
   string, double, int in unresolved form. So compile will check and assign more
   precise type on compilation stage check".

   - An untagged literal takes the context's type where its value reads as that
     type with no loss, and compile refuses a value that does not. Its type was
     provisional, so this is not a coercion, and ADR 0021 stands.
   - A tagged literal keeps its type. Where it differs from its semantics'
     declared type, compile wraps the cast to the declared type (ADR 0089,
     rules 17 and 19). A tagged literal's narrowing cast stays refused, whatever its
     value (ADR 0089, rule 18): a cast inside one type family to a type that the
     literal's type does not widen to (ADR 0086, rule 9). So `{lit: {double:
     1.5}}` and `{lit: {double: 2.0}}` for an `integer` Dimension are refused,
     and a `string` literal for a `date` Dimension is cast.
   - A Dimension's literal takes the Dimension's declared type as its context. A
     Measure's literal is its aggregate's operand, and its context is the
     operand's typing. Inside an operation node or a call, ADR 0092 rule 4 gives
     the context.

   Each row is a Dimension:

   | Authored | Context | Model | Manifest |
   |---|---|---|---|
   | `rate: {lit: 1.5}` | `rate` declares `decimal(18,2)` | untagged `Double(1.5)` | `Decimal { "1.5", 18, 2 }` |
   | `day: {lit: "2026-05-19"}` | `day` declares `date` | untagged `String("2026-05-19")` | `Date("2026-05-19")` |
   | `day: {lit: {string: "2026-05-19"}}` | `day` declares `date` | tagged `String("2026-05-19")` | `String("2026-05-19")`, which compile casts to `date` |
   | `code: {lit: null}` | `code` declares `bigint` | untagged `Null(String)` | `Null(BigInt)` |
   | `qty: {lit: 1.5}` | `qty` declares `integer` | untagged `Double(1.5)` | refused: it would truncate |
   | `at: {lit: "2026-05-19T10:00:00+02:00"}` | `at` declares `timestamp(6)` | untagged `String(…)` | refused: a canonical `Timestamp` holds no zone |

7. The manifest and the plan hold only typed literals, typed nulls included. A
   null's type is a canonical data type, never the compile-only `Unknown` (ADR
   0086, rule 5). A literal with no context keeps its parse default. No untagged literal reaches
   either: compile writes each one as an `Expr::Literal` of its type, and the
   manifest's `Link` and the plan's `Parameter` hold none (rule 2).

8. A semantic mapping's `lit:` arm holds a tagged `Literal` or an untagged one,
   an `UntypedLiteral` (rule 2). `LiteralValue` retires, so
   a semantic mapping's `null`, `decimal` or `timestamp` literal compiles.

### Every reference is a name path

9. `NamedReference` is a name path: `pub struct NamedReference(Vec<String>)`.
   Its last part is the name. The parts before it name the physical source or
   the semantic source that holds it. The human: "something like used in spark
   which could store array of identifier as string, which points this reference
   to concrete entity in model or manifest".

10. The semantic leaf is built on it:

    ```rust
    pub enum SemanticLeaf {
        Untyped(UntypedLiteral), // an untagged literal, which holds its parse default (rule 2)
        Field(NamedReference),
        Dimension(NamedReference, Option<DimensionAccessor>),
        Measure(NamedReference, Option<MeasureAccessor>),
        Metric(NamedReference, Option<MetricAccessor>),
    }
    ```

    The accessors are the sugar (`CONTEXT.md` **Sugar**: "lowering an accessor
    … into its canonical windowed shape").

11. A physical source gets a source alias. Compile writes it on the physical
    source as `ps`, then the physical source's index over all physical sources,
    in canonical `EntityId` order: `ps0`, `ps1`. It is ADR 0078's rule with no
    letter, because a physical source has no source role. The `ps` prefix
    cannot equal a semantic source's `ss`. `validate()` derives each `ps` alias
    again and refuses one that differs, as ADR 0078 (*`validate()` derives each
    alias again*) does for a semantic source, with a variant that names the
    physical source. It also runs rule 12's check on every column: two parts,
    and the alias that the column's place names. The human: "column can store
    alias to source ( physical / semantic ). So it is uniform for semantic and
    physical leaf".

12. In the manifest, the first part of every column is a source alias:

    | Stage | A Dataset's tree | A join condition |
    |---|---|---|
    | model | `Column(["amount"])`: the author writes no alias | no column |
    | manifest | `Column(["ps0", "amount"])` | `Column(["ssd4", "customer_id"])` |

    In a composition's tree, the alias is the one the merged ADRs write today,
    in the form ADR 0078 gives it: the constituent's in a join condition or a
    relationship filter (ADR 0057), the operand's in a Joinset's projection
    (ADR 0065), and the semantic source's own in a Unionset's projection (ADR
    0065) and above a Grainset's drill-across (ADR 0084). Nothing is cut at a
    dot. Compile checks that the alias in a Dataset's tree is the alias of that
    Dataset's physical source, and that the alias in a composition's tree is
    the one those ADRs name for its place. ADR 0049 gives each Dataset one
    physical source, so one alias fits each Dataset's tree.

    A column in the manifest holds two parts, the alias and the name. That is
    the form `<qualifier>.<definition name>` that ADR 0075 gives a name in
    planning (*Every output name is the one its parent reads*). A table's own
    identity is the three names that ADR 0067 gives a table read (*The scan*:
    "the catalog, the namespace …, the name"), and the `ps` alias stands for it
    in a column. What each planning step writes into a reference is not ruled here.

13. A column carries no type. Its type is in the schema it reads: the physical
    source's schema, the physical read's (ADR 0082), or the output schema of
    the plan node below it (`CONTEXT.md` **Output field**: "One named, typed
    element of a plan node's output schema").

### Ruled elsewhere, not here

- How the planner binds a Dataset's leaf `["ps0", "amount"]` to its physical
  read is not ruled here. This
  ADR does not amend ADRs 0071 and 0075 on it.

### Amends

- **0001:** in its *Decision Outcome*, the first sentence becomes:
  `semstrait-sql` owns the shared expression vocabulary, the types that 0085's
  rule 3 moves to it, and `semstrait-model`, `semstrait-ir` and
  `semstrait-manifest` each depend on it directly. `semstrait-model` keeps what
  rule 3 keeps in it, and `semstrait-ir` still depends on `semstrait-model`, for
  `SemanticsName` only (ADR 0090, rule 3). The rest of 0001 stands.
- **0002:** in its *Decision Outcome*, the first sentence becomes: a single
  `Expr<L, A>` declares every structural variant once, with `Literal` and
  `Column(NamedReference)` as nodes of the tree, and `L` holds only a stage's
  own leaf, as 0085's rules 1 and 2 give them. The forbidden combinations stay
  unrepresentable. The rest of 0002 stands.
- **0046:** in its *Decision Outcome*, "Every type it persists is owned by
  `semstrait-ir`, `semstrait-model`, or `semstrait-catalog`" also names
  `semstrait-sql`, which owns the tree, `Literal`, `NamedReference` and
  `DataType`; in *The schema is canonical; the native spelling is catalog-owned
  provenance*, the canonical `DataType` is `semstrait-sql`'s. The rest of 0046
  stands.
- **0057:** its decision "The attachment says whether a leaf is qualified" is
  replaced by ADR 0085, rules 11 and 12, and the sections that only apply it
  read as those rules give: its title's "the attachment says whether it does";
  in *Considered Options*, the chosen option's "with the expression's attachment
  deciding whether a leaf is qualified"; in *Decision Outcome*, the sentence
  "Whether a leaf is qualified at all is not read from the string"; in *The
  attachment says whether a leaf is qualified, because a dot cannot*, the
  sentence that "a dot means qualified" would be a guess, and the sentences from
  "In a composition-attached expression every column leaf carries a qualifier"
  to the end of that paragraph; and the **Physical expression** clause of its
  *Consequences*, which is the one ADR 0085 writes. In its decision "A join
  condition's column leaf carries its constituent's alias", in its *Decision
  Outcome*, *The qualifier travels inside the string, and the type does not
  change*, *The attachment says whether a leaf is qualified, because a dot
  cannot* and *The check ADR 0050 owed is now definable*, a column reference is
  the name path that 0085's rules 9, 11 and 12 give. Nothing is cut at a dot:
  the check reads the alias from the first part. The rest of 0057 stands.
- **0065:** in *A composed source's signature names the constituent's projected
  column*, the qualified column is the two-part `NamedReference`, not the
  string `alias.column`. The rest of 0065 stands.
- **0076:** in *A window's order key carries its direction*, `SortDir` and
  `NullOrdering` move to `semstrait-sql`, not to `semstrait-model`, because
  `Expr` is now `semstrait-sql`'s. The rest of 0076 stands.
- **0078:** in *An alias is `ss`, its source role's letter and its index*, "An
  alias holds no dot, so ADR 0057's cut at the first dot still returns the
  whole alias" has no subject, because nothing in the manifest is cut; an alias
  still holds no dot. The rest of 0078 stands.

### Consequences

- Good, because `semstrait-sql` has a complete tree of its own: a column, a
  literal and the operators, with no leaf set from another crate.
- Good, because a literal's type is a canonical data type, `DataType`, and it has one
  carrier. A mapping's `null`, `decimal` and `timestamp` values compile.
- Good, because every column in the manifest names its source, and no reader
  cuts a string to find the qualifier.
- Neutral: every leaf set, every `Literal` site and every file that names
  `ColumnRef` (33 source files in model, ir and manifest) changes.
- Neutral: a column in a Dataset's tree holds the source alias before the
  physical column name, and a reader of the name reads the last part.
- Neutral: adding a physical source changes the index of each physical source
  after it in `EntityId` order, and so the `ps` alias in each Dataset tree that
  reads one. An alias holds inside one manifest only, as ADR 0078 has it for a
  semantic source.
- Bad, because an untagged literal is a leaf again in the two unresolved trees,
  beside the `Literal` node, so a reader of those trees matches two literal forms.
  The gain is that `Literal`, `Expr` and the manifest carry no mark, and no
  manifest or plan can hold an untagged literal.
- Bad, because `{lit: 1.0e39}` is refused, and its author writes `{lit:
  {double: 1.0e39}}`: the YAML reader gives the same float for a whole number
  past `u128` or `i128` (rule 5). The gain is that no digit is cut silently
  from a whole number that large.
- `CONTEXT.md` changes:
  - **Canonical layer**: "in which no SQL may appear" becomes "in which no SQL
    text may appear: SQL text is a format artifact, and only a format adapter
    writes it", and its _Avoid_ gains "reading the SQL clause as a ban on SQL's
    expression structure: it bans SQL text".
  - **Source alias** names a semantic source or a physical source, and holds
    `ps` and the index for a physical source.
  - **Physical source** names its source alias.
  - **Physical expression**: every reference is qualified by a source alias,
    and the attachment decides which kind of source the qualifier names.
  - **Column qualifier** is the first part of a qualified reference, and names a
    physical source in a Dataset's tree.
  - **Column**: a qualified reference names an output field in a composition's
    tree, and its qualifier names a semantic source or a physical source.
- ADRs 0031, 0037, 0044, 0046, 0049, 0053, 0055, 0059, 0061, 0065, 0071, 0073,
  0074, 0075, 0077, 0078, 0081 and 0082 write a column as `Column(x)`,
  `<alias>.<name>` or `ColumnRef`, or name a type that this ADR retires (a leaf
  set, `ColumnRef`, `LiteralValue`). Each stands for the `NamedReference`,
  `Literal` and leaf type of this ADR, and where one reads a column's name, it
  reads the last part. A `PhysicalLeaf` in them is `semstrait-ir`'s, the plan's
  leaf set, not the mapping's leaf of rule 2. Those ADRs are not amended for it.
- This closes ADR 0050's open question that "`ColumnRef` is arguably redundant
  vocabulary": the type retires.
- The expression vocabulary grows only by an ADR (`CONTEXT.md` **Canonical
  vocabulary**). This is that ADR for `Column` and `Literal` as nodes.

## More Information

Decided on the question of the expression tree in semstrait-sql — what moves from the
model and what stays. The literal's
variants follow the roster decided on the question of the SQL types and names the canonical data
type roster gains. The
human ruled a timestamp literal's precision, the manifest's leaf, the untagged
literal's leaf and the crates' dependencies. The human's rulings of 2026-10-15
keep a
literal's type through YAML and JSON, and refuse an untagged whole number past
`BigInt` and an untagged float at or past 2^128, or at or below -2^127. The
human's later rulings name the
types of the untagged literal and `ExprOperationExt`, and retire `CanonicalFn`.
