---
number: 92
title: An operation node has a declaration and checks its operands by type family, and a type system derives the result types
status: accepted
date: 2026-10-08
source:
- CONTEXT.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md
- docs/adr/0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md
- '`binary_op_type`, `reconcile_boundary`, `unify` and `is_widening` in `crates/semstrait-manifest/src/compile/expressions/lower.rs`; the `null_if` alias of `nullif` in `crates/semstrait-sql/src/functions/catalog/scalar/conditional.rs`; the registry in `crates/semstrait-sql/src/functions/registry.rs`; `DataType` in `crates/semstrait-sql/src/datatype/mod.rs`'
- '`calcite@calcite-1.42.0`: `core/src/main/java/org/apache/calcite/sql/type/OperandTypes.java:1316-1317`, `rel/type/RelDataTypeFactoryImpl.java:310-326`; `substrait@v0.85.0`: `extensions/functions_datetime.yaml:238-308`, `:381-468`; `datafusion-expr-common@55.0.0:src/type_coercion/binary.rs:382-394`'
- '`datafusion-expr-common@55.0.0:src/type_coercion/binary.rs:953-971`, `:1190-1199`, `:2067-2103`; `datafusion-functions@55.0.0:src/core/nullif.rs:27-98`; `datafusion-expr@55.0.0:src/expr.rs:2159-2161`'
- '`calcite@calcite-1.42.0`: `core/src/main/java/org/apache/calcite/sql/type/ReturnTypes.java:936-1081`, `rel/type/RelDataTypeSystem.java:47-169`, `:219-488`, `rel/type/RelDataTypeSystemImpl.java:131-138`; `substrait@0.63.0`: `substrait/extensions/functions_arithmetic_decimal.yaml:6-213`, `functions_arithmetic.yaml:1047-1238`, `:1498-1632`, `functions_rounding_decimal.yaml:6-83`, `proto/substrait/algebra.proto:1245-1247`; `arrow-arith@59.3.0:src/numeric.rs:841-974`; `datafusion-substrait@55.0.0:src/logical_plan/consumer/expr/cast.rs:26-50`'
- the human's rulings of 2026-10-08 and 2026-10-15
links:
- target: 21
  kind: amends
- target: 68
  kind: amends
---

# An operation node has a declaration and checks its operands by type family, and a type system derives the result types

## Context and Problem Statement

ADR 0085 moves the expression tree to `semstrait-sql`, and its operation nodes
keep their form ("Operators are operators"). They need a
decision on how each is typed and spelled per dialect:

- **Code types an operation node.** `binary_op_type` gives an arithmetic
  operator its left operand's type, and gives a comparison `Boolean` with no
  check of its operands. `Case` and `Coalesce` require equal types (`unify`).
  ADR 0021: "`BinaryOp` returns `SameAs(left)`; cross-operand compatibility is
  unvalidated; NULL propagation is delegated to the engine" (*Decision
  Outcome*).
- **Compile needs the result type at a semantics' boundary.**
  `reconcile_boundary` casts an expression to its semantics' declared type. Where
  the semantics declares no type, the expression's own type is the semantics'
  type.
- **DataFusion 55 differs from Substrait on some operators**. It ignores
  Substrait's options, so an `ILike` that the producer writes as `like` with
  `case_sensitivity` gets a case-sensitive match, and a `SafeDivide` written as
  `divide` with `on_division_by_zero: NULL` fails on `÷ 0`. Its integer
  arithmetic wraps on overflow.

Where an engine keeps operators as nodes, code types them: DataFusion has
`BinaryTypeCoercer`. Calcite and Substrait type an operator like a function.
A design that declares `+`, `=` and `AND` as functions (`ADD`, `EQ`, `AND`)
binds their operands to one type variable `T` by exact equality, so
`add(Integer, Double)` is refused. Operators staying nodes rules those
declarations out.

## Considered Options

- **A declaration for each operation, of a function declaration's parts, with
  the node kept as a node; operands checked by type family** (chosen).
- **Code types each operation, as today and as DataFusion does.** An engine's
  difference then has no declaration, and its rules hold both the difference and
  the rewrite.
- **Operators as functions in the catalog**, with a `syntax`
  field. Operators staying nodes rules it out.
- **Every operand binds one exact type `T`**, as that function design does. The human:
  "don't do deep typechecking … checking family of types and tags to make basic
  validation is enough for example long > decimal is ok, because engine can do
  conversion of one of the operand to uniform type and compare this". An exact
  `T` refuses `bigint > decimal` and `qty + units`.
- **The left operand's type, as today.** `qty + units` and `units + qty` get
  different types.
- **Widening by a lattice whose casts compile inserts**, the option ADR 0021
  refused. The engine already converts the operands of one family, so
  semstrait only names the result.
- For a result type that depends on the arguments' types: **a return rule that
  names a derivation, which a type system holds, as in Calcite; the canonical
  type system holds the result types that semstrait wants, which are
  Substrait's except for an average, and Substrait's type system overrides the
  canonical one for the average** (chosen, rule 3).
  - **The bigger precision and the bigger scale**, `Decimal(max,max)`. The
    manifest then calls `price * price` decimal(18,2) while the plan holds
    decimal(37,4), and compile writes no cast, because the inferred type equals
    the declared one.
  - **Calcite's default formulas, at precision 38**: `*` gives p1 + p2, where
    Substrait and DataFusion give p1 + p2 + 1.
  - **Each formula from its own source**: Substrait's for arithmetic,
    DataFusion's for `sum` and `avg`, ISO's for `ceil` and `floor`. The human:
    "there should canonical one ( based of what we want to achieve ) and dialect
    implements overwrite to this".
  - **Substrait's rule for an average too**: Substrait's `avg` keeps its
    argument's type, so the average of the integers 1 and 2 is an integer.
  - **A DataFusion type system with Arrow's rules**, which the engine adapter
    compares with the canonical one. The plan holds Substrait's types, and
    DataFusion turns them into its own: "what we want - is high level
    representation - not how internally datafusion operates" (the human).
  - **The type system holding the types' limits and defaults too**, as
    Calcite's `getMaxPrecision` and `getDefaultPrecision` are part of its type
    system (`RelDataTypeSystem.java:47-92`). A type's limits would then have two
    homes, its declaration (ADR 0086) and the type system.
- **`nullif` with a boolean condition and a value.** No engine names that
  function `nullif`: it is `CASE WHEN c THEN NULL ELSE v END`. semstrait parses
  `null_if: {left, right}`, and DataFusion's `nullif` "Returns _null_ if
  _expression1_ equals _expression2_; otherwise it returns _expression1_".
- **Substrait's NULL rule for BETWEEN**, NULL where any operand is NULL. SQL
  defines BETWEEN as two comparisons, and DataFusion follows SQL.
- **A `Concat` operator node for `||`.** The human: "right now we do not focus on
  operators || + * , we have multiply tags for this, so this is more for SQL text
  which is not in scope for now".
- **A volatility on each operation.** Every operation is immutable, and
  DataFusion asks a volatility only of a function call
  (`expr.rs:2159-2161`). The human: "i think it is redudant in this scope".
- For a date, a timestamp or a time with an interval: **any interval family, with
  the temporal operand's type as the result**, as Calcite and DataFusion have it
  (chosen). **A timestamp for a date with an interval**, as PostgreSQL gives, and
  as Substrait's `add` gives, though its `subtract` keeps the date. **Only the
  pairs that Substrait declares**: Substrait has no `time ± interval` and no
  `interval_compound` overload.
- **No mark for an operation that a dialect does not have**, because DataFusion
  has every operation. `Dialect` is `#[non_exhaustive]`, and a later engine with
  no `ILIKE` would then change what `on` returns, and every reader with it.

## Decision Outcome

In short:

- An operation node stays a node, and each operation has a declaration of a
  function declaration's parts. `Case` and `Cast` keep rules in code. A dialect
  has a declaration where its engine differs, and marks an operation that it
  does not have.
- An operation checks its operands by type family only: the engine converts
  them, and the declared widening order names the result (ADR 0086, rule 9).
- A result type that a return rule derives from the arguments' types comes from
  a type system, for an operation and a function alike, as in Calcite. The
  canonical type system holds the result types that semstrait wants, and
  Substrait's overrides it where Substrait's rules differ. A manifest is typed
  by the canonical one, and a Substrait plan by Substrait's.
- Each canonical declaration of an operation says when its result is NULL, by
  SQL's rule. `coalesce` and `nullif` are functions.

### An operation has a declaration

1. An **operation** is each kind of operation node that has a declaration:
   each kind of `BinaryOpKind`, `UnaryOpKind` and `LikeKind`, and `Between`,
   `InList` and `IsNull`. `Case` and `Cast` are operation nodes that code types
   (rule 5). Each operation has a declaration of a function declaration's
   parts (ADR 0091, rule 4), less what only a call has:

   ```rust
   pub struct Operation {
       pub canonical: OperationDeclaration,
       pub dialects: &'static [(Dialect, Option<OperationDeclaration>)], // None: the dialect has no such operation
   }

   pub struct OperationDeclaration {
       pub name: &'static str,                       // on a dialect: the engine's own name
       pub aliases: &'static [&'static str],         // other YAML spellings; none on a dialect
       pub signature: Signature,                     // each operand's type family
       pub returns: ReturnRule,
       pub nullability: Nullability,
       pub options: &'static [OptionDecl<OperationOption>],
   }

   impl Operation {
       /// The dialect's declaration, the canonical one where the dialect has none,
       /// or `None` where the dialect has no such operation.
       pub fn on(&'static self, dialect: Dialect) -> Option<&'static OperationDeclaration>;
   }

   impl BinaryOpKind { pub fn operation(self) -> &'static Operation { .. } }
   ```

   - The node stays a node: the tree holds `BinaryOp { op: Add, .. }`, never a
     call. The human: "op remains operator, not a functionally, but can
     additionaly gain instruction around such things".
   - The declarations sit in `functions`, beside the function declarations
     whose `Signature`, `ReturnRule`, `Nullability` and `OptionDecl` they
     reuse. The node stays in `expr`.
   - An operation has no call clauses and no volatility. Every operation is
     immutable, and an engine asks a volatility only of a call (DataFusion,
     `expr.rs:2159-2161`).
   - `Case` and `Cast` have no declaration, and code holds their rules (rule 5).
     A list of branches and a target type are not a list of operands, and every
     engine in the research keeps them apart.
   - An operation has a dialect declaration where the engine differs, as a
     function has (ADR 0091, rule 5), and also where only the values of an
     option differ, such as DataFusion's `overflow` (rule 8). It has at most one
     for each dialect.
   - A dialect's entry is `None` where its engine has no such operation.
     DataFusion has every operation (rule 9), so none of its entries is `None`.
     The mark exists now so that a later dialect that lacks an operation does
     not change what `on` returns.
   - An operation's `name` is its generic name, which is today's YAML tag
     (`multiply`, `eq`, `negate`). Its `aliases` are the other YAML spellings
     that parse accepts, as a function's are (ADR 0091, rule 4). Parse finds the
     element for a YAML tag by the `name` and the `aliases` of the
     declarations, so `sugar.rs`'s tag tables go away. The human: "they most
     probably need alternative spelling ( alias ), so parse in model can use
     pattern match to resolve from model to expression elements on parse".

### An operation checks its operands by type family

2. An operation checks each operand only by its **type family**. The engine
   converts the operands of one family to one type, and semstrait does not
   compute that conversion. The human: "we do not really execute operators - we
   declare them and we based on declaration in the model". Each canonical data
   type names its family and the types it widens to (ADR 0086, rule 9).

   - An operation's signature names a type family for each operand, and a
     function's parameter may name one too (ADR 0086, rule 8).
   - A tagged literal, `{lit: {string: "5"}}`, is checked by the family of its
     tag, as a column of that type is. An untagged literal, `{lit: "5"}`, takes
     its context's type by rule 4. The human: "tagged, means lit: "5", or lit:
     string: "5". as it is right now".
   - A dialect may declare its own families and order later. The human: "We can
     hold this families declaration with dialect as well, right now assuming that
     for all engines".
   - DataFusion 55 compares a date with a timestamp by casting both to a
     timestamp, and refuses a time with a timestamp (`binary.rs:2067-2103`).

### A type system derives a result type

3. An arithmetic operation's result is the bigger operand's type, by the
   widening order (`Promote`). The human: "bigger type promoted across for this
   case, but again we need way to declare this for operations". Where an
   operand is a decimal, the result comes from the **type system** of the
   `datatype` package, as in Calcite:

   | Operation | Return rule | With a decimal operand |
   |---|---|---|
   | `+`, `-` | `Sum` | the type system's decimal sum |
   | `*` | `Product` | its decimal product |
   | `/`, `SafeDivide` | `Quotient` | its decimal quotient |
   | `%` | `Mod` | its decimal remainder |

   The type system also holds each function's result type that its argument's
   type decides, as Calcite's holds the result type of `SUM`, `AVG` and `COVAR`
   beside decimal arithmetic (`RelDataTypeSystem.java:147-169`). The human: "this
   seems like limiting type system elemnts to decimal behavior, but if i'm not
   mistaken - such things in substrait and calcite is broader".

   | Return rule | Functions | Canonical result |
   |---|---|---|
   | `SumOf` | `sum` | an integer gives bigint, a real or a double gives double, and a decimal(p,s) decimal(38,s) |
   | `AvgOf` | `avg`, `median` | an integer, a real or a double gives double, and a decimal(p,s) decimal(38, max(s, 6)) |
   | `DecimalScaleZero` | `ceil`, `floor` | a decimal(p,s) gives decimal(min(p − s + 1, 38), 0); any other type gives itself |
   | `RoundOf` | `round` | a decimal(p,s) gives decimal(min(p + 1, 38), s); any other type gives itself |

   ```rust
   pub struct Decimal { pub precision: u8, pub scale: i8 } // a `DataType::Decimal`'s parts (ADR 0086, rule 1)

   pub trait TypeSystem {
       fn decimal_union(&self, a: Decimal, b: Decimal) -> Decimal { /* Promote */ }
       fn decimal_sum(&self, a: Decimal, b: Decimal) -> Decimal { /* Substrait's add and subtract */ }
       fn decimal_product(&self, a: Decimal, b: Decimal) -> Decimal { /* Substrait's multiply */ }
       fn decimal_quotient(&self, a: Decimal, b: Decimal) -> Decimal { /* Substrait's divide */ }
       fn decimal_mod(&self, a: Decimal, b: Decimal) -> Decimal { /* Substrait's modulus */ }
       fn sum_of(&self, arg: &DataType) -> DataType { /* Substrait's sum */ }
       fn avg_of(&self, arg: &DataType) -> DataType { /* double, or decimal(38, max(s, 6)) */ }
       fn decimal_scale_zero(&self, arg: Decimal) -> Decimal { /* Substrait's ceil and floor */ }
       fn round_of(&self, arg: Decimal) -> Decimal { /* Substrait's round */ }
   }

   pub struct CanonicalTypeSystem;
   impl TypeSystem for CanonicalTypeSystem {}

   pub struct SubstraitTypeSystem;
   impl TypeSystem for SubstraitTypeSystem { /* avg_of: an integer's, a real's or a double's own type, or decimal(38, s) */ }
   ```

   - A return rule holds no formula. It names the derivation, and the type
     system holds it. In Calcite, `*` holds `PRODUCT_NULLABLE`, which calls the
     type system's `deriveDecimalMultiplyType` (`ReturnTypes.java:936-958`).
     With no decimal operand, each of `Sum`, `Product`, `Quotient` and `Mod` is
     `Promote`. The human: "We could check how calcite solves that ( because it
     works for many engine ) and action semetrically".
   - A type's limits stay on its declaration (ADR 0086, rule 2), and each rule
     reads them there, so a decimal's greatest precision, 38, has one home.
     Calcite holds them in its type system (`getMaxPrecision`,
     `getDefaultPrecision`; `RelDataTypeSystem.java:47-92`), and its decimal
     rules read them from it.
   - The canonical type system holds the result types that semstrait wants,
     and a dialect's type system overrides it. The human: "there should
     canonical one ( based of what we want to achieve ) and dialect implements
     overwrite to this, it feels more natural". Each canonical rule that
     Substrait declares is Substrait's, the return program of the function
     that the producer writes (`functions_arithmetic_decimal.yaml:6-213`,
     `functions_arithmetic.yaml:1047-1238`,
     `functions_rounding_decimal.yaml:6-83`), except `AvgOf`. Substrait's
     `avg` keeps an integer's, a real's or a double's type and a decimal's
     scale, so the average of the integers 1 and 2 is an integer, and the
     average of a decimal(18,2), at decimal(38,2), cannot hold 1.015. Its
     `median` keeps an integer's, a real's or a double's type, and it has no
     decimal `median` (`functions_arithmetic.yaml:1498-1632`). The canonical
     `AvgOf` keeps the fraction: 6 is the least scale of Substrait's own decimal
     quotient (`functions_arithmetic_decimal.yaml:78`).
   - Each decimal operation's rule is Substrait's. Above precision 38, it
     lowers the scale, to no less than 6:

     | Rule | Scale | Precision, before the cap at 38 |
     |---|---|---|
     | `decimal_sum` | max(s1, s2) | scale + max(p1 − s1, p2 − s2) + 1 |
     | `decimal_product` | s1 + s2 | p1 + p2 + 1 |
     | `decimal_quotient` | max(6, s1 + p2 + 1) | p1 − s1 + p2 + scale |
     | `decimal_mod` | max(s1, s2) | min(p1 − s1, p2 − s2) + scale |
     | `decimal_union` | max(s1, s2) | max(p1 − s1, p2 − s2) + scale |

   - `decimal_union` gives `Promote`'s result for two decimals: the bigger
     scale and the most digits before the point. It is Calcite's
     `leastRestrictive` for two decimals and DataFusion's wider decimal
     (`binary.rs:1190-1199`).
   - An integer operand enters the decimal rule as a decimal of its type's
     digits (ADR 0086, rule 2), with scale 0: tinyint (3,0), smallint (5,0),
     integer (10,0), bigint (19,0).
   - The decimal union and an integer operand's entry are Calcite's, because a
     Substrait function takes no two operands of different types. The
     producer casts an argument to a Substrait overload's type (ADR 0087,
     rule 24).
   - Substrait's type system holds Substrait's rule where it differs from the
     canonical one. Of the rules that Substrait declares, only `AvgOf`
     differs: Substrait's gives an integer's, a real's or a double's own type,
     and decimal(38,s) for a decimal(p,s). The plan is Substrait's, and a
     call's type in it "Must be set to the return type of the function,
     exactly as derived using the declaration in the extension"
     (`algebra.proto:1245-1247`). Substrait is a dialect for typing (ADR 0086,
     rule 6): the human, "we add substrait as dialect for typing - which is ok".
   - Compile types a manifest with the canonical type system, and the Substrait
     producer types the Substrait plan with Substrait's (ADR 0089, rule 12).
     Where a result type differs, the producer casts the argument to a
     Substrait overload's type, so that the plan is correct (ADR 0087,
     rule 24): it writes `avg(qty)` as `avg(CAST(qty AS double))`, and
     `avg(price)` as `avg(CAST(price AS decimal(38,6)))`.
   - semstrait holds no type system for DataFusion. DataFusion's consumer
     reads a call's type in the plan only on a cast
     (`consumer/expr/cast.rs:26-50`), and turns each Substrait type into its
     own Arrow type. How DataFusion holds a value is the engine's side. The
     human: "Datafusion in plan uses substrait's declaration which then maps
     types into its own internal representation ( arrow ), so what we want -
     is high level representation - not how internally datafusion operates -
     because it is going to be on engine side - we need to correct logical
     plan". Calcite types a query with the one type system of the engine that
     runs it; semstrait types a manifest with the canonical one, and a
     Substrait plan with Substrait's.
   - Where neither operand widens to the other and neither is a decimal with a
     decimal or an integer, as with a double and a decimal, the result has no
     type. The semantics' declared type then gives it by a cast (ADR 0089,
     rule 18), and with no declared type compile refuses it, as
     `reconcile_boundary` refuses an unknown root today.
   - A date, a timestamp or a time with an interval of any interval family
     (`interval`, `interval_year_month` or `interval_day_second(p)`; ADR 0086,
     rule 9) gives the date's, the timestamp's or the time's type. Calcite's `+`
     and `-` take every datetime with every interval family and keep the
     datetime's type (`OperandTypes.java:1316-1317`,
     `RelDataTypeFactoryImpl.java:310-326`), and DataFusion 55 does the same
     (`binary.rs:382-394`).
   - A function takes `Promote` where its declaration names it: `greatest` and
     `least` (ADR 0087, rule 15). With two decimals, `Promote` gives the type
     system's decimal union.

   | Column | `qty` | `units` | `price` | `rate` | `score` | `day` | `region` |
   |---|---|---|---|---|---|---|---|
   | Declared type | integer | bigint | decimal(18,2) | decimal(10,4) | double | date | string |

   | Expression | Result |
   |---|---|
   | `qty + units`, `units + qty` | bigint |
   | `qty + price` | decimal(19,2) |
   | `units * price` | decimal(38,2) |
   | `price + rate` | decimal(21,4) |
   | `price * rate` | decimal(29,6) |
   | `price * price` | decimal(37,4) |
   | `price / rate` | decimal(38,12) |
   | `units + price` | decimal(22,2) |
   | `greatest(price, rate)` | decimal(20,4), the decimal union |
   | `sum(price)` | decimal(38,2) |
   | `avg(qty)`; `avg(price)` | double; decimal(38,6). Substrait's `avg` gives integer and decimal(38,2), so the plan casts the argument |
   | `ceil(price)`; `round(price, 0)` | decimal(17,0); decimal(19,2) |
   | `score * price` | no type: the declared type casts it, or compile refuses it |
   | `units > price` | boolean |
   | `day + interval '1 day'` | date |
   | `day - day` | refused: semstrait has no duration type, and `date_diff` gives it |
   | `region = 5` | refused: string and number are two families |

4. Inside an operation node or a call, the context of an untagged literal is
   what binds the parameter it fills. That is the type the other operands give,
   or the parameter's exact type. Where no operand has a type, the node's own
   context passes down. A parameter that names only a family gives no type, so
   the literal keeps its parse default. ADR 0085 rule 6 then applies: the
   literal takes the context's type where its value reads as that type with no
   loss, and compile refuses it otherwise.

   | Authored | Literal after compile |
   |---|---|
   | `amount > 1.5`, with `amount` decimal(18,2) | `Decimal { "1.5", 18, 2 }` |
   | `qty + 1`; `coalesce(qty, 0)` | `Integer(1)`; `Integer(0)` |
   | `amount > 1.125` | refused: it would cut a digit. The author writes `{decimal: "1.125"}` |
   | `amount > {lit: "5"}` | `Decimal { "5", 18, 2 }` |
   | `amount > {lit: {string: "5"}}` | refused: the tag's family is string (rule 2) |
   | Dimension `total` declares integer, `expr: 1 + 2` | `Integer(1) + Integer(2)`, which compile then folds to `Integer(3)` (ADR 0089, rule 23) |
   | Measure `sum(1)` | `BigInt(1)`: `sum` takes any number |
   | Measure `sum(amount * 1.5)` | `Decimal { "1.5", 18, 2 }` |

### The canonical declarations

5. The canonical declaration of each operation, and the rules of `Case` and
   `Cast`:

   | Operation | Operands | Result | NULL and options |
   |---|---|---|---|
   | `+`, `-` | two numbers; or a date, a timestamp or a time with an interval of any interval family | `Sum` (rule 3) | NULL if an operand is NULL. Option `overflow`, canonical `ERROR` |
   | `*`, `/`, `%` | two numbers | `Product`, `Quotient`, `Mod` (rule 3) | NULL if an operand is NULL. `*`: `overflow`, `ERROR`. `/` and `%`: `on_division_by_zero`, `ERROR` |
   | `SafeDivide` | two numbers | `Quotient` (rule 3) | NULL if an operand is NULL, or for `÷ 0` (`on_division_by_zero`, `NULL`) |
   | negate | a number | its type | NULL if its operand is NULL. `overflow`, `ERROR` |
   | `=`, `<>`, `<`, `<=`, `>`, `>=` | two operands of any one family | boolean | NULL if an operand is NULL |
   | `IsNotDistinctFrom` (ADR 0074) | two operands of any one family | boolean | never NULL: true where both operands are NULL, false where one is |
   | AND, OR | two booleans | boolean | `Kleene` |
   | NOT | a boolean | boolean | NULL if its operand is NULL |
   | LIKE, NOT LIKE, ILIKE, NOT ILIKE | two strings | boolean | NULL if an operand is NULL |
   | BETWEEN, NOT BETWEEN | three operands of one family | boolean | `Kleene`, as `x >= low AND x <= high` |
   | IN, NOT IN | a value and items of one family | boolean | `Kleene`, as `x = a OR x = b …` |
   | IS NULL | any | boolean | never NULL |
   | CASE (code) | every WHEN a boolean; every THEN and the ELSE of one exact type | that type | the chosen branch's value; NULL where no WHEN matches and there is no ELSE |
   | CAST (code) | an input whose type casts to the target type by `casts_to()` (ADR 0086, rule 10), and the target type | the target type | NULL for a NULL input. A failed cast is an error, or NULL with `on_failure: Null`. ADR 0086 rule 10 gives how a cast converts a value |

   For CASE, the human: "for case must be boolen and return types must be
   equal".

6. `coalesce` and `nullif` are functions. The human: "coalesce* is not an
   operator there, it is function". The catalog declares them (ADR 0087,
   rule 15):

   - `coalesce`: every argument has one exact type, and the result has that
     type. The result is NULL only if every argument is NULL. The human: "for
     caolesce - as for function, types must be even".
   - `nullif(a, b)`: SQL's NULLIF, which gives NULL where `a` equals `b`, and
     `a` otherwise. The two arguments are in one family, because `nullif`
     compares them. The result has `a`'s type, and it can always be NULL. Its
     alias is `null_if`, today's YAML tag. The human: "we can use aliases for
     giving variation of naming in different places - tag in yaml model works
     for that".
   - `coalesce` and `nullif` are no tags of the tree, so the catalog can
     declare them. A `semstrait-model` test checks that no function's name or
     alias equals a tag of the tree, and `RESERVED_AST_TAGS` retires (ADR 0087,
     rules 9 and 10).

7. Each canonical declaration of an operation says when its result is NULL, by
   SQL's rule (rule 5). ADR 0021's "NULL propagation is delegated to the
   engine" no longer holds for an operation node. `Nullability` gains `Kleene`:
   the result is NULL only where the operands that are not NULL do not decide
   it. FALSE AND NULL is FALSE, TRUE OR NULL is TRUE, and `5 BETWEEN NULL AND 3`
   is FALSE.

### DataFusion and the Substrait producer

8. DataFusion has a dialect declaration for these operations:

   | Operation | DataFusion 55 | Its dialect declaration |
   |---|---|---|
   | `ILike`, `NotILike` | it ignores Substrait's `case_sensitivity` option, and its own name is `ilike` | the name `ilike` |
   | `SafeDivide` | it ignores Substrait's `on_division_by_zero` option, so `÷ 0` is an error | `on_division_by_zero` has the one value `ERROR` |
   | `+`, `-`, `*`, negate | integer overflow wraps silently | `overflow` has the one value `SILENT` |

   - An operation node has no option slot, so the canonical value always
     applies.
   - The DataFusion rules write the canonical `SafeDivide` as
     `divide(a, nullif(b, 0))`.
   - Whether the engine's rules can write the canonical overflow on DataFusion is
     not ruled here.

9. The Substrait producer writes an operation under Substrait's official name
   (ADR 0068), and DataFusion reads each name that it writes:

   - `ILike` as `like` with `case_sensitivity: CASE_INSENSITIVE`, and
     `SafeDivide` as `divide` with `on_division_by_zero: NULL`.
   - NOT LIKE, NOT ILIKE and NOT BETWEEN as `not(…)`, because Substrait v0.85
     has no `not_like` and no `not_between`.
   - BETWEEN as `and(gte(…), lte(…))`, because Substrait's `between` gives NULL
     where any operand is NULL, and rule 5 gives `Kleene`.
   - Substrait has no `add` or `subtract` overload for a time with an interval,
     or for any datetime with an `interval_compound`
     (`functions_datetime.yaml:238-308`, `:381-468`). How the producer writes
     them is not ruled here.
   - The tree has no `||` node. Operator aliases such as `||` are SQL text.
     The human: "this is more for SQL text which is not in scope for now … in
     future we define aliases for such".

### Amends

- **0021:** in its *Decision Outcome*, "NULL propagation is delegated to the
  engine" no longer holds for an operation node: its canonical declaration says
  when its result is NULL (ADR 0092, rule 7), and where an engine differs, its
  dialect declaration says so. "`BinaryOp` returns `SameAs(left)`" and
  "cross-operand compatibility is unvalidated" become: an operation checks each
  operand by its type family (ADR 0092, rule 2), and the operation's declared
  return rule names the result. "Signatures require exact `DataType` equality"
  becomes, for an operation: an operation checks its operands by type family.
  "Semstrait publishes no promotion or coercion lattice" becomes: `Promote`
  reads the widening order of ADR 0086 rule 9 to name a result type, and with a
  decimal operand the type system names it (ADR 0092, rule 3). Compile inserts
  no cast between an operation's operands. "Unification is strict equality over
  width-differentiated types" stands for `Case` and `coalesce`. The rest of 0021
  stands.
- **0068:** in *The seam*, the clause from "Every dedicated expression variant
  Substrait models as a plain function call" to "`coalesce` included" becomes:
  every operation node that Substrait models as a plain function call —
  `BinaryOp`, `UnaryOp`, `Like`, `Between` and `IsNull` — lowers as a
  `ScalarFunction` under the name the rules' function hook returns, and a
  negated `Like` or `Between`, and a `Between`, lower as ADR 0092 rule 9 writes
  them. `coalesce` and `nullif` are function calls (ADR 0092, rule 6). The rest
  of 0068 stands.

### Consequences

- Good, because an operation's typing, NULL behaviour and options are declared,
  as a function's are, so an engine's difference is a declaration that the
  adapter reads.
- Good, because the type family check refuses `region = 5` and `qty + region`,
  and still takes `units > price` with no cast.
- Good, because each result type that the arguments' types decide has one
  home, the type system, and the canonical one is Substrait's, the language of
  the plan, wherever Substrait declares one, except for an average.
- Good, because parse finds each element by its declaration's `name` and
  `aliases`, and the tag tables in code go away.
- Bad, because semstrait holds a second set of declarations beside the
  functions, for nodes that are not calls. The gain is that an operation stays
  a node, and its typing, NULL behaviour and options are data that the adapter
  reads.
- Bad, because `Operation::on` returns `Option`, so every reader handles a
  `None` that DataFusion never gives. The gain is that a later dialect with no
  such operation changes neither what `on` returns nor any reader.
- Bad, because DataFusion's own result type can differ from the plan's:
  `price / rate` is decimal(38,12) in the plan, and DataFusion returns
  decimal(26,6), with six fewer digits after the point
  (`arrow-arith@59.3.0:src/numeric.rs:841-974`). The gain is a correct plan,
  whose types are Substrait's; how DataFusion holds a value is the engine's
  side.
- Bad, because the Substrait plan casts the argument of `avg` over an
  integer, a `real` or a decimal with fewer than 6 digits after the point, and of
  `median` over an integer or a `real`. The gain is
  that an average keeps its fraction, and the plan stays correct.
- Bad, because `day - day`, which compiled with no check, is refused, and its
  author writes `date_diff`, which DataFusion runs only by a rewrite (ADR
  0087). The gain is no wrong type: compile typed it as `date`, its left
  operand's type, where DataFusion returns a count of days.
- Bad, because ADR 0021's "no promotion or coercion lattice" bends: semstrait
  names each arithmetic result type, by the widening order or the type system,
  and the Substrait producer casts where Substrait's type differs.
  The gain is a result type for arithmetic, which the family check and the
  cast to a semantics' declared type need.
- Neutral: `Coalesce` and `NullIf` leave the tree, so every site that builds or
  matches them changes, and so do the registry's reserved tags.
- `CONTEXT.md` changes:
  - **Type system** is a new entry.
  - **Operation node** and **Type family** are new entries.
  - **Dialect** names functions, operation nodes and data types.
  - **Canonical declaration** and **Dialect declaration** also cover an
    operation, and a dialect's entry says when its engine has no such
    operation.
- ADRs 0057, 0065, 0066, 0068, 0071, 0073, 0074, 0075, 0076, 0077, 0079 and
  0081 name the `Coalesce` or `NullIf` node (`Coalesce(<from>.<name>,
  <to>.<name>)`, `Coalesce(Sum(<partial>), 0)`, `NullIf(Column(year), …)`;
  ADR 0066's "`Expr::Coalesce` and `Expr::Case` exist for their uses", and ADR
  0068's "the `Coalesce` a `Full` join's composed Dimension signature carries").
  Each stands for a call of the function `coalesce` or `nullif` (rule 6). Those
  ADRs are not amended for it.
- The expression vocabulary grows only by an ADR (`CONTEXT.md` **Canonical
  vocabulary**). This is that ADR for the removal of `Coalesce` and `NullIf`.

## More Information

Decided on the question of how each operator node is typed and spelled per
dialect. It
ruled the mark for an
operation that a dialect does not have and the reference design for the decimal
rules, and a further ruling
covered the type system's scope, its canonical rules and the decimal union. A later ruling
is that a date, a
timestamp or a time takes an interval of any interval family. Substrait's
type system rests on Substrait as a dialect for typing, decided on the question of the
SQL types and names the canonical data type roster
gains. The human's later rulings, which a record keeps, move the check
of the tree's tags to `semstrait-model` and keep the alias `null_if`.
