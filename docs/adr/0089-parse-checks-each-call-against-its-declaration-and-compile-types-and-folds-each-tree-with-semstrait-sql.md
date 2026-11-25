---
number: 89
title: Parse checks each call against its declaration, and compile types and folds each tree with semstrait-sql
status: accepted
date: 2026-10-12
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0082-a-physical-read-types-an-unobserved-column-by-the-semantics-mapped-to-it-and-compile-writes-that-type-into-the-schema.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md
- docs/adr/0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md
- docs/adr/0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md
- docs/adr/0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md
- docs/adr/0092-an-operation-node-has-a-declaration-and-checks-its-operands-by-type-family-and-a-type-system-derives-the-result-types.md
- '`parse_expr` and `parse_tagged_mapping` in `crates/semstrait-model/src/model/serde/de.rs`; `validate` in `crates/semstrait-model/src/validation/mod.rs`'
- '`reconcile_boundary`, `fold_binary`, `function_return_type`, `match_signature`, `binary_op_type` and `bare_column_type` in `crates/semstrait-manifest/src/compile/expressions/lower.rs`; `scalar_entry` and `lower_literal` in `crates/semstrait-manifest/src/compile/expressions/mod.rs`; the `NarrowingCast` tests in `crates/semstrait-manifest/tests/semantic_signatures.rs`; `test_data/acme_eu_ad_platform_v3.yaml`'
- '`Expr::fold` in `crates/semstrait-sql/src/expr/fold.rs`'
- '`crates/semstrait-sql/src/expr/typing.rs`'
- '`calcite@calcite-1.42.0` (`c01f6b5`): `core/src/main/java/org/apache/calcite/sql/SqlUnresolvedFunction.java:30-36`, `sql/validate/SqlValidatorImpl.java:7443-7446`, `sql/SqlOperator.java:516-590`, `sql2rel/SqlToRelConverter.java:622-631`, `rex/RexBuilder.java:319-367`, `sql/type/SqlTypeUtil.java:997-1028`, `sql/type/ReturnTypes.java`, `sql/type/OperandTypes.java`, `sql/SqlCallBinding.java` (`permutedOperands`)'
- '`spark@v4.2.0`: `sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/logical/FunctionBuilderBase.scala` (`InputParameter`, `NamedParametersSupport.defaultRearrange`)'
- '`datafusion-sql@55.0.0:src/expr/function.rs:329`, `:910-922`; `datafusion-expr@55.0.0:src/expr_schema.rs:583-602`, `src/type_coercion/functions.rs:110-155`; `datafusion-physical-expr@55.0.0:src/expressions/cast.rs:37-44`; `arrow-cast@59.3.0:src/cast/decimal.rs:351-410`, `:770-819` (a float to a decimal rounds), `:883-895` (a decimal to an integer is cut), `src/cast/mod.rs:2550-2572` (`num_cast`: a float to an integer is cut)'
- '`datafusion-optimizer@55.2.0`: `src/simplify_expressions/expr_simplifier.rs:545-570` (the constant evaluator folds a CAST of a literal)'
- Spark `docs/sql-ref-ansi-compliance.md` at `v4.0.0` (*Store assignment*); PostgreSQL 18 documentation, *Value Storage* (`typeconv-query.html`)
- the human's rulings of 2026-10-12, of 2026-10-13 on the design, and of 2026-10-17, and, for the fold's closed list (2026-10-14), and, for the form of a CASE and the writer of a literal, the human's rulings of 2026-10-15
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the cast of a row argument in a Unionset branch)
links:
- target: 21
  kind: amends
- target: 73
  kind: amends
- target: 82
  kind: amends
- target: 3
  kind: amends
- target: 81
  kind: amends
- target: 94
  kind: amendedby
---

# Parse checks each call against its declaration, and compile types and folds each tree with semstrait-sql

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0094](0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md): rule 20's "and `count_distinct` as today" and "Each of today's values stores the same bytes as today.": compile stores `count_distinct` as `"aggregation": "count", "distinct": true` (rule 5), and every other value stores the same bytes as today. "`count_distinct` stays `semstrait-model`'s sugar for `count` with DISTINCT." stands.

## Context and Problem Statement

ADRs 0085 to 0088, 0091 and 0092 declare each function, operation and data type in
`semstrait-sql`. They leave open which stage of the resolution pipeline (parse,
validate, compile, plan, adapt) runs each check, and where the code is. The design
gives `semstrait-sql` "typing, behavior, serialization, parser, algebra if
needed". The human: "algebra if needed ( to validate structural, typing behavior )",
and "others cargo re-use their typing declaration to do their work".

Today:

- **Parse** (`semstrait-model`) reads YAML through three tag tables in `sugar.rs`.
  It refuses an unknown tag, and a function that the YAML function table does not
  have (`UnknownTag`). It reads a bare name as a leaf, and it refuses an expression
  written as text (`InlineExprUnsupported`).
- **Validate** (`semstrait-model`) runs five passes over the semantic model and
  checks no expression.
- **Compile** (`semstrait-manifest`) types each tree with eleven functions in
  `lower.rs`. It refuses a table name that the registry does not have
  (`UnknownFunction`), and it gives every column the type `Unknown`, which passes every
  check: "a bare column's type is `Unknown` by design". It folds a binary operation
  over two literals (`fold_binary`).
- **At a semantics' declared `data_type`**, `reconcile_boundary` casts the tree to the
  declared type on every mismatch, and it warns (`NarrowingCast`) where the cast does
  not widen. ADR 0021 says: "a narrowing physical→declared cast is a compile error
  demanding an explicit author cast" (*Decision Outcome*).
- **Two `lit:` grammars.** An expression's `lit:` takes the tags `integer` and
  `boolean`, and a semantic mapping's takes `int` and `bool`.
- **`agg:`** takes seven values (`AggregationType`), and the function catalog has 28
  aggregates (ADR 0087).

The engines draw these lines as follows:

- **Calcite** parses a function's name as an unresolved function
  (`SqlUnresolvedFunction`). Its validator resolves the name, checks the operand count
  and types, and derives the result in one stage (`SqlOperator.validateOperands`). The
  validator finishes before relational conversion starts
  (`SqlToRelConverter.java:628`, then `:631`). `RexBuilder` derives a call's type with
  the same `inferReturnType`.
- **DataFusion** resolves a name when it plans the SQL
  (`function.rs:329`). It checks types in schema derivation (`fields_with_udf`), and
  its analyzer calls the same check to insert casts.

Both have one return-type method per function, and every stage calls it.

## Considered Options

For the stage of each check:

- **Parse checks what the declaration alone decides, and compile checks what needs
  types** (chosen).
- **Parse resolves names only, and compile checks the count and the types together**,
  as Calcite's validator does. Calcite's parser has no declarations. Semstrait's parse
  has them, because the tree holds a handle to each function (ADR 0091, rule 1).
- **Validate runs the checks that need only the declaration**, as a pass of
  its own.

For typing:

- **One typing module in `semstrait-sql`, which takes the document's dialect**
  (chosen).
- **Typing stays in compile**, and `semstrait-sql` holds only the declarations. The
  planner and the adapter then type again with code of their own.
- **Typing reads the canonical declarations only**, and the adapter compares the
  declarations by itself.

For the parser:

- **`semstrait-model` walks the YAML, and builds each element from its declaration in
  `semstrait-sql`** (chosen).
- **`semstrait-sql` reads the YAML**, with `semstrait-model`'s reader for its leaves. The human:
  "model parses tags and then calls semstrait-sql method to get concerete expr element
  - op, function - basically forming unresolved expression".
- **No bare name in an expression**: `divide: [{measure: cost}, {measure: clicks}]`.
  The corpus would change 22 lines.

For a CASE's YAML form:

- **Today's form, `case: {whens: [{when:, then:}, …], else: …}`** (chosen).
- **A list with a last `{else:}` item**, `case: [{when:, then:}, …, {else:}]`. The
  list reads as if it held any number of ELSE items, and only a rule over positions
  keeps it to one. The human: "under this case form it looks like you can have
  unlimited else elements, whereas else can be only one".

For a column that compile did not observe:

- **A direct mapping casts the column to its semantics' declared type, except under an
  aggregate whose parameter is `any` and whose result type is fixed, and every other read of the column takes ADR 0082's read type, else
  `Unknown`** (chosen).
- **`Unknown`, with no cast**, as today. An operation's type family check then never
  applies to it.
- **ADR 0082's read type for every read, with no cast in a semantics' tree.** That type
  is the widest type of all the semantics on the column, not each semantics' own.
- **`Unknown` for every read that is not a direct mapping.** Where a `string`
  Dimension maps `region_code`, `region_code = 5` in a Filter compiles, and it fails
  later, in the plan or the engine.

For a tree whose type the declared type narrows:

- **Cast it, where `casts_to()` allows the cast, and fail at run time on a value
  outside the declared type's range**, as Spark's ANSI policy and ISO's store
  assignment do (chosen).
- **Refuse it** (ADR 0021; Spark's `STRICT` policy, which "doesn't allow any possible
  precision loss"). Every Measure of `acme_eu_ad_platform_v3.yaml` is `agg: sum`
  over `decimal(18,2)`, declared `decimal(18,2)`, and `sum` gives `decimal(38,2)`, so
  each is refused. An author cannot write a cast around an aggregate, because no
  expression holds one.
- **Cast it and warn**, as today.

For `agg:`:

- **`agg:` names a catalog aggregate, and its declaration decides the call** (chosen).
- **`agg:` takes the catalog aggregates whose signature has one parameter, a value.**
  The human: "let's keep it simple form for now. I think agg might be extended. I just
  don't like rule which might be ammended later on". `count` takes its value as an
  optional argument, and a new form for `agg:` would change the rule.
- **Today's seven values.** `AggregationType` then stays a second list beside the
  catalog, which ADR 0087 rule 10 makes "the one list".

For the "algebra":

- **One fold of literals in `semstrait-sql`, which compile and the planner call**
  (chosen). The human: "I mainly meant partial evaluation for literals - so planner can
  prune some elements - what is current planner do for unionsets as an example".
- **Tree operations, or an expression simplifier.** No decision here needs them.
- **Compile stores each tree unfolded.** ADR 0073's pruning of a branch reads "the
  branch's own signature" and replaces a leaf only where that signature "is a
  `Literal`", so a predicate on a metadata-determined signature that is not folded
  never prunes a branch.

## Decision Outcome

Chosen options: parse checks each call against its declaration, compile types each tree
with `semstrait-sql`'s typing, and one fold serves compile and the planner. With them,
each check runs in the first stage that has what it needs, and every crate calls one
implementation, as in Calcite and DataFusion.

In short:

- Parse refuses what the declaration alone decides. Validate checks no
  expression. Compile refuses what needs types.
- `semstrait-model` walks the YAML and builds each call from its declaration, with
  `semstrait-sql`'s call constructors. An enum argument is its value, written bare or as
  a string literal, and a bare value at a constant parameter is a literal of the
  parameter's type. A CASE keeps today's form, `case: {whens: [{when:, then:}, …],
  else: …}`, with one ELSE at most, and it nests.
- An expression's `lit:` and a semantic mapping's `lit:` read one grammar, whose tags
  are canonical type names. The writer writes each literal so that it reads back with
  its type.
- Typing is one module in `semstrait-sql`. It takes the document's dialect, `canonical`
  in compile.
- Compile types a column by its observed type. It casts a column that it did not
  observe to the declared type of the semantics mapped to it, except under an
  aggregate whose parameter is `any` and whose result type is fixed, such as `count`.
  Every other read of that column takes ADR 0082's read type, else `Unknown`, and
  compile writes the read type into the model-fed schema. Over an observed column whose type differs, a `sum`,
  an `avg`, a `min` or a `max` casts the column too.
- A parameter whose presence an earlier argument decides takes its declared default
  where the call leaves it out: the semantic model's `date_trunc` takes `'UTC'` and Monday, and
  its timezone is `'UTC'` only.
- Compile casts a tree to its semantics' declared type where `casts_to()` allows the
  cast, and a value outside the declared type's range fails at run time. Digits after
  the point round or are cut as the canonical CAST does (ADR 0086, rule 10). A literal
  follows ADR 0085 rule 6.
- `agg:` names a catalog aggregate, in its simplest form.
- The **fold** evaluates a closed list of parts over their literals: the parts whose
  value the planner's pruning of a Unionset branch reads, or whose value lets compile
  store a metadata-determined signature as a literal. Every other part stays as it is,
  for the engine. Compile folds each tree that it stores, and the planner folds a
  predicate to prune a Unionset branch.

### Each stage checks what it can decide

1. **Parse** refuses what the declaration alone decides:

   - a tag, an operation or a function that is not declared;
   - a call whose arguments do not fit its signature: their number, a literal at an
     enum or a constant parameter, and an enum value that the declaration has;
   - a call's mapping that names parameters (rule 4), where it leaves out a required
     parameter, or one with no declared default that an earlier argument's value says
     the call writes, or where a key names no parameter, or one that an earlier
     argument's value says the call does not write;
   - a literal that ADR 0085 rule 5 refuses, and an array value that its element type
     does not hold (rule 9);
   - a `regexp_extract` group below 0, or above the number of groups of a literal pattern
     (ADR 0087, rule 12);
   - a data type that is neither a canonical name nor an alias (ADR 0086, rule 7), or
     for which `is_valid()` is false.

2. **Validate** keeps its five passes over the semantic model and checks no expression. Each
   expression check needs either the declaration only, which parse has, or the types of
   the leaves and columns, which only compile has (ADR 0003: compile fetches the
   catalog).

3. **Compile** refuses what needs types: an operand outside its declaration's type
   family (ADR 0092, rule 2), an argument that is not its parameter's exact type, type
   family or bound `T` (ADR 0086, rule 8), a call that leaves out a parameter with no
   declared default that an earlier argument's type says the call writes, or writes one
   that the type says the call does not write (ADR 0091, rule 4), a semantic model's
   `date_trunc` whose timezone is not `'UTC'` (rule 4), an untagged literal that its context's type
   does not hold (ADR 0085, rule 6; ADR 0092, rule 4), a cast that `casts_to()` refuses
   (ADR 0086, rule 10), and a tree that its semantics' declared type cannot take (rule
   18).

   | Authored | Today | Now |
   |---|---|---|
   | `upper: [name, 2]` | compile, `FunctionArityMismatch` | parse refuses it: `upper` takes one argument |
   | `left: [name, 3]` | compile, `UnknownFunction`: the registry has no `left` | it parses: `left` is in the function catalog |
   | `upper: amount`, where the catalog gives `amount` as `decimal(18,2)` | it compiles: the column is `Unknown` | compile refuses it: the parameter is in the string family |

### Parse builds each element from its declaration

4. `semstrait-model`'s parse walks the YAML and reads every tag. For an operation or a function, it
   finds the declaration by the tag, which is the declaration's `name` or one of its
   `aliases` (ADR 0091, rule 4; ADR 0092, rule 1). It reads each argument in the form
   that its parameter's signature gives, and it builds the call with `Call::new`:

   ```rust
   // semstrait-model: the walk
   let function = registry().get(&tag)?;   // by name or alias (ADR 0087, rule 8)
   let args = read_args(&function, body)?; // each argument in the form of its canonical declaration's parameter
   let call = Call::new(function, args)?;  // semstrait-sql's check (ADR 0091, rule 2)
   ```

   The human: "model parses tags and then calls semstrait-sql method to get concerete
   expr element - op, function - basically forming unresolved expression", and "I was
   thinking that it goes over declration, to get shape and signature and then makes call
   out of this".

   A call's arguments are a list, one argument, or a mapping that names parameters of the
   canonical declaration by their names, in any order: `null_if: {right: b, left: a}` is
   `nullif(a, b)`. A mapping of two entries or more names parameters. A one-entry mapping
   names a parameter only where its key is that parameter's name and neither a tag of the
   tree nor a function's name: `upper: {value: campaign}` is `upper(campaign)`. Any other
   one-entry mapping is one argument, so `upper: {col: campaign}` is `upper(campaign)`
   too, and the one argument of `null_if: {left: a}` is a call of `left`, as today.

   A mapping that names parameters names every required one. Parse refuses it where it
   leaves a required parameter out, or where a key names no parameter; the YAML reader
   refuses a key written twice. Parse builds the call's arguments in the declaration's
   order, so the tree stays positional. A skipped optional parameter that a later named
   parameter follows takes the default that its declaration declares (ADR 0091, rule 4).
   An optional parameter that no later named parameter follows stays absent, so each
   call that parses today keeps its arguments. The human: "only if there mandatory
   defined and parameters used yaml tag to identify where values is getting pass to",
   and "optional assumed to have defaults".

   A parameter that is present only when an earlier argument says so keeps that
   presence in a mapping (ADR 0091, rule 4), and each named argument stays at the
   parameter that its key names. Where the earlier argument's value decides, parse
   writes the parameter's declared default (ADR 0091, rule 4) where the call writes the
   parameter and the call leaves it out, in a mapping or in a list, and refuses a key
   that names it where the call does not write it, as it refuses a key that names no
   parameter: `date_trunc` writes `week_start` only where its `granularity` is `week` (ADR
   0072), and its default is `'monday'` (ADR 0087, rule 14). A parameter with no
   declared default that the call writes and leaves out is refused, as a required one
   is. Where the earlier argument's type decides, as for `date_trunc`'s `timezone`,
   which only a timestamp writes, compile checks the presence and writes the default
   `'UTC'` (rule 3). In the table, `order_date` is a date and `ts` is a timestamp.

   The semantic model's `date_trunc` takes the timezone `'UTC'` only, by its default or written,
   and compile refuses another (rule 3). A timestamp that it truncates in UTC is an
   instant, as every value that a source gives is, so a request predicate, the
   request's own bucket (ADR 0072) and a join read the Dimension as one. A timestamp
   truncated on another clock is a clock value, which those reads would take as an
   instant. `week_start` has no clock, so it takes either value that ADR 0072 gives, Monday or
   Sunday. The planner's own
   `date_trunc` writes the request's timezone (ADR 0063, ADR 0072). The human:
   "I think we will decide some timezone switches later on"; and: "date_trunc can
   be written in this forms - but defaults are assigned - UTC/Monday, to reduce burden
   of changes".

   | A mapping | The arguments that parse builds |
   |---|---|
   | `regexp_instr: {value: {col: c}, pattern: {lit: 'x'}, group: 1}` | `(c, 'x', 1, 1, '', 1)`: `start`, `occurrence` and `flags` take their defaults (ADR 0087, rule 12) |
   | `regexp_instr: {pattern: {lit: 'x'}, value: {col: c}}` | `(c, 'x')` |
   | `regexp_instr: {pattern: {lit: 'x'}, group: 1}` | refused: the required `value` is missing |
   | `regexp_instr: {value: {col: c}, pattern: {lit: 'x'}, grop: 1}` | refused: no parameter is named `grop` |
   | `lag: {value: {col: x}, default: 0}` | `(x, 1, 0)`: `offset` takes its default (ADR 0087, rule 19); `Call::new` then refuses the call, because it builds no clause and ORDER BY is `Required` on `lag` (ADR 0091, rule 2) |
   | `round: {value: {col: x}}` | `(x)`, as today |
   | `date_trunc: {granularity: day, value: {col: order_date}, week_start: {lit: monday}}` | refused: where `granularity` is `day`, the call does not write `week_start` |
   | `date_trunc: {granularity: week, value: {col: order_date}, week_start: {lit: monday}}` | `(week, order_date, 'monday')`, as the list form `date_trunc: [week, {col: order_date}, {lit: monday}]` gives: `'monday'` is at `week_start`, and a date writes no `timezone` |
   | `date_trunc: {granularity: week, value: {col: ts}, timezone: {lit: UTC}}` | `(week, ts, 'UTC', 'monday')`: where `granularity` is `week`, the call writes `week_start`, and it takes its default |
   | `date_trunc: [week, {col: order_date}]` | `(week, order_date, 'monday')`: the list form takes the default too |
   | `date_trunc: [month, {col: ts}]` | `(month, ts)`; compile writes the default `'UTC'`, because a timestamp writes `timezone` (rule 3) |
   | `date_trunc: [month, {col: ts}, {lit: Europe/Stockholm}]` | `(month, ts, 'Europe/Stockholm')`; compile refuses it: the semantic model's `date_trunc` takes `'UTC'` only |

   This is Spark's shape, where a parameter declares its default (`InputParameter`) and
   `NamedParametersSupport.defaultRearrange` puts the arguments in the declaration's
   order, with that default for a skipped parameter; Calcite also takes named arguments in
   any order and fills a skipped optional parameter, with NULL, which does not fit
   `regexp_instr` on DataFusion, where `regexp_instr(x, 'p', NULL, …)` gives NULL.

   A semantic model's dialect is `canonical`. The human: "Model by default uses
   canonical dialect, in future we will add dialect declration ( which is used to lookup
   expression elements and propogated into manifest )". The lookup needs no I/O, because
   the function catalog is `semstrait-sql`'s own; compile still resolves the names of
   semantics (ADR 0003). An untyped boundary is not the design,
   because parse resolves each name.

5. Every crate that builds a call builds it with `semstrait-sql`'s constructors, which
   check it against its declaration (ADR 0091, rule 2): `semstrait-model`'s parse, compile, the
   planner, and reading a manifest. In a manifest, a link stands only where the authored
   tree references another semantics (ADR 0088, rule 5), and an enum or a constant
   parameter takes a literal (rule 1), so reading a manifest checks each call as parse
   does. Parse builds a call that writes no clause with `Call::new`, and
   it drops a clause that the declaration marks `NoEffect` before it builds the call.

6. `semstrait-sql` reads no YAML. `semstrait-model` holds no expression type of its own: it
   holds the aliases `UnresolvedSemanticExpr = RowExpr<SemanticLeaf>` and
   `UnresolvedPhysicalExpr = RowExpr<PhysicalLeaf>`, and the builders that ADR 0085 rule 3
   leaves it "for now". The human: "so model imports semstrait-sql for that purpose. no
   expression implementation in model, only type alias for Unresolved forms and
   Leaf/Aggregate boundaroes". `SemanticLeaf`, `PhysicalLeaf`, `UntypedLiteral`,
   `NoLeaf`, and the `Leaf` and `Aggregation` bounds stay `semstrait-sql`'s (ADR 0085,
   rule 3).

7. A bare name in an expression is a leaf, as today: `divide: [cost, clicks]` reads two
   leaves. No expression is written as text: `expr: "cost / clicks"` stays refused. The
   human: "Right now we support expression declration using yaml tags - no bare strings
   for now in expr".

   An enum parameter takes one of the known constant values that its declaration lists,
   such as `extract`'s component, `date_trunc`'s granularity and `date_diff`'s part (ADR
   0087, rule 14). Its argument is the value, written bare or as a string literal: at
   an enum parameter, parse reads a bare word as that string literal, by the
   parameter's signature (rule 4). So `extract: [year, {col: order_date}]` and
   `extract: [{lit: year}, {col: order_date}]` give one call, and `Call::new` refuses
   `extract: [yeer, {col: order_date}]` (ADR 0091, rule 2), because `yeer` is not one of
   the values. The human: "we can serve several enums , which will store known constant
   values for different functions - so just literal string , and just value can be
   accepted".

   A constant parameter takes only a literal (ADR 0091, rule 2), so a bare value there, a
   word or a number, is a literal of the parameter's type. Parse reads a bare word at a
   constant parameter as an untagged string literal, by the parameter's signature (rule
   4), and compile gives the untagged literal the parameter's type (ADR 0085, rule 6;
   ADR 0092, rule 4). So `regexp_instr: [{col: campaign}, {lit: "^uk_"}, 1, 1, i]`
   gives the flags `'i'`. A named argument is at the parameter that its key names (rule
   4), so `regexp_instr: {value: {col: campaign}, pattern: {lit: "^uk_"}, flags: i}`
   gives the flags `'i'` too. At a value parameter a bare name stays a leaf, because a value
   can be a column: `regexp_like: [{col: campaign}, {lit: "^UK_"}]` writes its pattern
   with `lit:`. The human: "we know from declration what to expected possitionly … form
   that we can avoid concrete typing or use of tags for some cases".

   A CASE keeps today's form: `case:` holds `whens:`, a list with one `{when:, then:}`
   item for each WHEN, and an optional `else:`. The list holds at least one item. A
   second `else:` is a duplicate key, which the YAML reader refuses ("duplicate entry",
   serde_yaml 0.9 `mapping.rs`), so a CASE holds one ELSE at most. A CASE nests in any
   `when:`, `then:` or `else:`: `acme_eu_ad_platform_v3.yaml` nests one three
   deep, in the Dimension `market` of the Unionset `paid_media_campaign_performance`.
   The human: "is this allows nesting case statement one to another, just to make sure
   that it is preserved", and "maybe it is better to just restore original form".

   ```yaml
   case:
     whens:
       - when: {eq: [country, {lit: "Germany"}]}
         then: {lit: "DE"}
       - when: {eq: [country, {lit: "Spain"}]}
         then:
           case:
             whens:
               - when: {eq: [region, {lit: "Canarias"}]}
                 then: {lit: "IC"}
             else: {lit: "ES"}
     else: {lit: "OTHER"}
   ```

   Parse (`parse_case`), `semstrait-model`'s writer and the corpus keep this form. The
   tree's JSON form has the same shape, with the same keys, `whens`, `when`, `then` and
   `else` (ADR 0088, rule 1).

### One literal grammar

8. An expression's `lit:` and a semantic mapping's `lit:` arm read one grammar, and
   `LiteralValue` retires (ADR 0085, rule 8). A tag is a canonical type name (ADR 0086,
   rule 7):

   | Expression `lit:` today | Mapping `lit:` today | Now, in both |
   |---|---|---|
   | `boolean` | `bool` | `boolean` |
   | `integer` (it gives a `bigint`) | `int` (it gives a `bigint`) | `tinyint`, `smallint`, `integer` and `bigint`, each its own type (ADR 0085, rule 5: `{integer: 5}` is `Integer(5)`) |
   | `float` (it gives a `double`) | `float` | `real` and `double` |
   | `decimal`, `string`, `date`, `timestamp` | the same | the same |
   | `time`, `interval`, `binary` | none | the same |
   | none | none | `varchar`, `char`, `uuid`, `interval_year_month` and `interval_day_second` (ADR 0085, rule 5) |
   | `null`, with no type | a bare `null`, or the string `"null"` | a bare `null` is `Null(String)` (ADR 0085, rule 5); `{null: bigint}` is `Null(BigInt)`; the string `"null"` is a string |

   In both, a tagged literal is a `Literal`, and an untagged literal is an
   `UntypedLiteral`: under `SemanticLeaf::Untyped` in a semantic expression, under
   `PhysicalLeaf::Untyped` in a mapping's `expr:` tree, and alone in a mapping's `lit:`
   arm (ADR 0085, rules 2 and 8).

   `semstrait-model`'s writer writes each literal so that parse reads it back, from
   YAML or from JSON, as the same literal, its type included (ADR 0085, rule 5). An
   untagged literal always has a parse default's type, and the writer writes it with
   no tag: `{lit: 5}` is `BigInt(5)` both ways. A tagged literal is written with its
   tag, and a decimal's digits as a string: `{lit: {decimal: {value:
   "99999999999999999999", precision: 38, scale: 0}}}`. So the writer writes no bare
   number past `BigInt`: parse refuses one, and serde_json reads one past `u64` as a
   float. The writer refuses an untagged literal that parse refuses, such as a
   `Double` at or past 2^128, or at or below -2^127.

   The corpus changes 15 lines: `actions: {lit: {int: 0}}` in
   `acme_eu_ad_platform_v3.yaml` becomes `{lit: 0}` or `{lit: {bigint: 0}}`.
   `actions` declares `decimal(18,2)` with `agg: sum`, and both forms give
   `sum(BigInt(0))`, as today.

9. An array literal is `{lit: {array: {element: string, values: ["a", "b"]}}}`, which gives
   `Literal::Array { element: String, values: [String("a"), String("b")] }` (ADR 0086,
   rule 4). Parse reads each value by the element's grammar,
   gives it the element type, and refuses a value that does not read as that type
   with no loss, as ADR 0085 rule 6 says of a context's type. A
   `Literal::Array` holds only typed values, so no untagged literal is inside one (ADR
   0085, rule 4; ADR 0086, rule 4). An untagged YAML list, `{lit: ["a", "b"]}`, is
   refused for now, because ADR 0085 rule 5 gives no faithful default for an array.

10. A tag is the name of a canonical type, so a type or a name that the roster gains is
    a tag. The roster's names are ADR 0086's (rule 7), and a literal of a type with a
    length or a precision takes it as ADR 0085 rule 5 says.

### Typing is one module in `semstrait-sql`

11. `semstrait-sql` holds typing, the walk
    over a tree in its `expr` package, and each rule beside the type that it rules on.
    A type system's derivations sit in `datatype`, a cast's check beside `casts_to()`,
    the check of a call's arguments beside `Signature` in `functions`, and a literal's
    type in `expr`. Calcite and DataFusion keep the walk apart from the rules in the
    same way: Calcite's `SqlValidatorImpl` walks a tree, and `ReturnTypes`,
    `OperandTypes` and `SqlTypeUtil` hold the rules; DataFusion's `expr_schema.rs`
    walks one, and `type_coercion` holds them. Typing stays one implementation, in one
    crate, that compile and the planner share.

    - **Inputs:** the tree; the type of each leaf and each column, from a trait that
      the caller implements; the document's dialect: `canonical`, or one `Dialect`; and
      the context's type.
    - **Output:** the tree, with each untagged literal typed (ADR 0085, rule 6; ADR
      0092, rule 4), its type and its nullability; or a typed `TypeError` (ADR 0004).

12. Typing reads each call's and each operation's declaration through `on(dialect)`, or
    the canonical declaration where the document's dialect is `canonical`, and each
    derivation that a return rule names from a type system (ADR 0092, rule 3). Compile
    types with `canonical` and the canonical type system. The planner types each tree
    that it builds with the manifest's `dialect`. The Substrait producer types the
    Substrait plan with Substrait's type system, finds where a result type differs from
    the canonical one, and casts the argument to a Substrait overload's type, so that the
    plan is correct (ADR 0087, rule 24). A DataFusion declaration gives the canonical
    result type (ADR 0091, rule 5). The human: "we declare and use dialect declaration in
    compilation to validate. Then adapter , when it check, might decide if it needs to
    wrap-up something or not".

    | Call | Typed with the canonical type system | Typed with Substrait's | The Substrait producer |
    |---|---|---|---|
    | `avg(qty)`, `qty` an integer | `double` | `integer` | writes `avg(CAST(qty AS double))` |
    | `upper(name)` | `string` | `string` | nothing |

    In Calcite, each operator has one `inferReturnType`, and its `RelDataTypeSystem` is
    the policy for one engine that the return rules read. In `semstrait-sql`, each
    declaration has one `returns`, read through `on(dialect)`, and a type system holds
    each derivation that it names.

13. Compile's typing functions in `lower.rs` (`function_return_type`, `aggregate_type`,
    `match_signature`, `param_admits`, `promote`, `sum_promote`, `avg_promote`,
    `binary_op_type`, `unify`, `is_widening` and `literal_type`) retire into it.
    `reconcile_boundary` stays compile's, because a semantics' declared type is a model
    concept. Compile reports a `TypeError` through one `CompileError` variant, so the
    variant set stays the one list of compile's errors (ADR 0044).

14. The data type checks run here:

    - `is_valid()` at parse, on a type's spelling, a cast's target and a literal's
      precision or length;
    - `is_valid_on(dialect)` in the engine adapter, for its own engine, which today
      gives the same answer as `is_valid()`, because DataFusion declares no data type and
      Substrait takes every canonical value (ADR 0086, rule 6). Compile checks
      `is_valid()` only, because a semantic model's dialect is `canonical`. How the
      adapter writes a value that its engine does not take is not ruled here (ADR 0086);
    - `casts_to()` in typing, on an authored CAST and on the cast to a semantics'
      declared type.

### Compile types a column by what it observed, else by its read type

15. A column's type is its physical source's schema type where compile observed one:
    the catalog gave it. Otherwise rule 16 gives its type, or the column is `Unknown`,
    which passes each check, as today. On a catalog-derived schema, ADR 0061 stands:
    a schema's `None` "means one thing and only one thing: this compile did not
    observe it", and a referenced column whose native type maps to no type fails the
    compile (ADR 0086, rule 7). On a model-fed schema, compile writes
    rule 16's read type into the column's `data_type` (ADR 0082, *Compile writes the
    read type into the schema*).

    | Expression | Column | Today | Now |
    |---|---|---|---|
    | `region = 5` | the catalog gives `region` as `string` | it compiles | refused: string and number are two families |
    | `amount > 1.5` | the catalog gives `amount` as `decimal(18,2)` | `Double(1.5)` | `Decimal { "1.5", 18, 2 }` |
    | `region = 5`, in a Filter | not observed; the Dimension `region` (`string`) maps it | it compiles | refused: its read type is `string` (rule 16) |
    | `channel = 5`, in a Filter | not observed; no semantics maps it | it compiles | it compiles: `Unknown` passes, and the planner refuses the column (`PlanError::UnresolvedType`, ADR 0082) |

16. A column that compile did not observe, and that a semantics maps directly (ADR 0082:
    its semantic mapping "is that column"), is cast to that semantics' declared type in
    that semantics' tree. The human: "for columns from catalog, if provided, otherwise it
    takes unknown type which must lead to wrapping column into cast into type which
    declared for linked semantic interface elements".

    An aggregate whose parameter is `any` and whose result type is fixed casts nothing
    on its column: `count` and `approx_count_distinct` (ADR 0087, rule 17). A
    semantics' declared type is the type of its whole expression, so where the
    aggregate's result differs from it, rule 17 casts the result. The human: "when semantic declares type - it is type expected after whole
    expresion is executed, so if we see count function, and it differs from what count
    function return - we wrapp result into cast expression which bring semantic type
    declaration". A cast of the column would make the call fail where the read does not:
    where the Dimension `user_ref` (`string`) also maps `user_id`, the read is `string`
    (ADR 0082), so `approx_count_distinct(CAST(user_id AS bigint))` fails at run time on
    `'u-17'`, and `approx_count_distinct(user_id)` counts it. Where only such aggregates
    map the column, the read itself takes their declared type, as ADR 0082 has it for a
    count. `min`, `max` and
    `any_value` take a bound `T`, so their column is cast.

    A column that the catalog observed gets the same cast in a `sum`, an `avg`, a `min`
    or a `max` whose declared type differs from the column's type. Over an observed `string` column `x`, `max(x)` declared `integer` is
    `max(Cast(x, integer))`, so the rows `'9'` and `'10'` give 10, as they do where no
    catalog observed `x`, and not the string maximum `'9'`. Rule 17 then casts the
    result where the aggregate's own type differs. So a Measure gives one answer whether
    or not a catalog observed its column.

    Every other read of such a column, in an expression mapping, a Filter or a Measure's
    `filter:`, takes ADR 0082's read type: the widest declared type of the semantics
    mapped to the column, which the physical read reads it as. Typing takes that type,
    and nothing casts the column for it. Compile computes the read type once and writes
    it into the model-fed schema, and the planner reads it there (ADR 0082, *Compile
    writes the read type into the schema*): from the trees alone, the planner cannot tell this rule's cast from an
    authored cast over the column, which stores the same tree. A column that no
    semantics maps has no read type, and it stays `Unknown`.

    | Semantics (declared type) | Mapping | Stored tree today | Now |
    |---|---|---|---|
    | `country` (`string`) | `country: adwords-country` | `Column([ps0, adwords-country])` | `Cast(Column([ps0, adwords-country]), string)` |
    | `cost` (`decimal(18,2)`), `agg: sum` | `cost: adwords-cost` | `sum(Column(…))` | `sum(Cast(Column(…), decimal(18,2)))`, then rule 17's cast |
    | `order_count` (`integer`), `agg: count` | `order_count: order_id` | `count(Column(order_id))` | `Cast(count(Column(order_id)), integer)`: the column gets no cast, and the result gets rule 17's cast |
    | `users` (`bigint`), `agg: approx_count_distinct` | `users: user_id` | refused at parse: `agg:` has no `approx_count_distinct` | `approx_count_distinct(Column(user_id))`: the column gets no cast, and the result is already `bigint` |
    | `name_length` (`integer`) | `{length: {col: name}}` | `length(Column(name))` | the same, with no cast: `name` has its read type where a semantics maps it, else `Unknown` |
    | `order_date` (`date`) and `order_ts` (`timestamp`), both on `created_at` | direct | `Column(created_at)` in each | `Cast(…, date)` and `Cast(…, timestamp)`, each in its own tree |

    On a model-fed schema, compile writes ADR 0082's read type into the column's
    `data_type`, and the physical read takes it. The cast in each tree then converts the read's type to that
    semantics' own type, as ADR 0082 says of `order_date`: "the read is `Timestamp`, and
    `order_date` casts from it".

### The cast to a semantics' declared type

17. Where a tree's type differs from its semantics' declared `data_type`, compile wraps
    the tree in a cast to the declared type, wherever `casts_to()` allows the cast (ADR
    0086, rule 10), in one type family too. A value outside the declared type's range
    fails at run time, because the cast's failure is an error (ADR 0092, rule 5: "A
    failed cast is an error, or NULL with `on_failure: Null`"). This is
    Spark's default store assignment: under its ANSI policy, numeric to numeric is
    allowed, and "an overflow error will be thrown if the value is out of the target
    data type's range". PostgreSQL's sizing cast rounds the extra digits after the point,
    and it fails on overflow. DataFusion's `CAST` fails on overflow.

    Digits after the point round or are cut by the canonical CAST's rule (ADR 0086, rule
    10), which the table shows.

    A Metric whose `agg:` aggregates a formula that reads a Measure or a Metric gets no
    such cast: its value is the planner's second aggregate step over the formula's
    values, so compile stores the formula alone, and the planner casts that step's
    value (ADR 0074, *A Metric*).

    | Semantics | Declared | Its type | Now |
    |---|---|---|---|
    | `cost: sum` over `decimal(18,2)` | `decimal(18,2)` | `decimal(38,2)` (ADR 0087, rule 17) | cast; a sum past 10¹⁶ fails at run time |
    | `quantity: sum` over `integer` | `integer` | `bigint` | cast; a sum past 2³¹ fails at run time |
    | `avg_price: avg` over `decimal(18,2)` | `decimal(18,2)` | `decimal(38,6)` | cast; four digits after the point round: `2.345000` gives `2.35` |
    | `units: sum` over a `double` column | `integer` | `bigint`: rule 16 casts the column first | cast; each row's fraction is cut before the sum: `2.7` gives `2` (PostgreSQL gives `3`) |
    | `day: {lit: {string: "2026-05-19"}}` | `date` | `string` | cast: `casts_to()` allows string to date (ADR 0085, rule 6) |

18. A literal follows ADR 0085 rule 6. An untagged literal takes its context's type
    where its value reads as that type with no loss, and compile refuses it otherwise. A
    tagged literal keeps its type, and its narrowing cast (ADR 0085, rule 6) stays refused
    whatever its value: `{lit: {double: 1.5}}` and `{lit: {double: 2.0}}` for an `integer` Dimension
    are refused. A tree whose type is `Unknown` gets the cast with no check: compile
    cannot check it, and the engine refuses a value that does not cast. The human: "I
    dont think it changes much, we cannot check this - runtime will refuse that if so". A
    tree with no type, such as `score * price` (ADR 0092, rule 3), gets the cast. Where
    the semantics declares no type, compile refuses such a tree.

19. Compile, not the planner, wraps these casts, but for the formula push-down's
    casts and the cast above a Metric's second aggregate step (ADR 0074, *A
    Metric*). In a Unionset branch's row projection, the planner also writes the cast
    of a row argument that no cast reaches, to the branches' least-restrictive type.
    Where the row argument of a Measure's `count`, count distinct or
    `approx_count_distinct`, or of its `any_value` over an observed column (rule 16),
    has another column type on another branch, each branch that maps the Measure casts
    the argument to that type, and a branch that NULL-fills the Measure writes a
    `Literal(NULL)` of that type (ADR 0074, *A Unionset*, **Rows** and **NULL fill**).
    So `integer` rows 1 and 2 on one branch and a `bigint` row 3 on the other give a
    `bigint` argument, and `count` gives 3. Nothing raises the `NarrowingCast` error
    variant any more, so it retires.

### `agg:` names a catalog aggregate

20. A Measure's `agg:` names a catalog aggregate, by its name or an alias. Parse checks
    the aggregate's call over one value with `semstrait-sql`'s check of the aggregate's
    declaration, with DISTINCT for `count_distinct` and no other clause.
    `count_distinct` stays `semstrait-model`'s sugar for `count` with DISTINCT. Compile
    builds the call over each source's value with the aggregate call's builder (ADR
    0091, rule 2). The human: "agg function, in simplest possible form accepted, framing
    is wrong that we have so search - with no clause or one parameter". The declaration,
    not `agg:`, decides which calls fit. `AggregationType` retires. The manifest stores
    the aggregate's canonical name, `"aggregation": "stddev_samp"`, and `count_distinct`
    as today; `agg: every` is stored as `bool_and`. Each of today's values stores the
    same bytes as today.

    | `agg:` | The call | Result |
    |---|---|---|
    | `sum` | `sum(value)` | it parses |
    | `count` | `count(value)`: `count` takes its value as an optional argument | it parses |
    | `every` | `bool_and(value)` | it parses, and it is stored as `bool_and` |
    | `corr` | `corr(value)`: `corr` takes `(y, x)` | `Call::new` refuses it: one argument of two |
    | `string_agg` | `string_agg(value)`: it takes a separator | `Call::new` refuses it: one argument of two |
    | `percentile_cont` | its WITHIN GROUP is `Required`, and it takes a fraction | `Call::new` refuses it |

    `agg: std_dev` and `agg: variance` stay refused (ADR 0087, rule 20). A Metric's
    optional `agg:` parses the same way. A row-level Metric's `agg:` builds its call
    as a Measure's does (ADR 0074, *A Metric*). Over a formula that reads a Measure or
    a Metric, it builds no call: it is the planner's second aggregate step.

### One fold

21. The **fold** is `semstrait-sql`'s partial evaluation of an expression, in the `expr`
    package. It folds the operands of a part first, then the part. Its scope is a closed
    list: the parts whose value the planner's pruning of a Unionset branch reads (rule
    22), or whose value lets compile store a metadata-determined signature as a literal
    (rule 23). A part joins the list only when a rule of the planner or of compile
    reads its value. The human: "i dont think we need to go byond what is used in
    planner - for example brances skipping in unionset, sticking with MVP objectives".

    The fold evaluates these parts, and no other part:

    | Part | Its value |
    |---|---|
    | AND, OR | by the `Kleene` NULL rule (ADR 0092, rule 7): where `x` is not a literal, `x = 'a' AND FALSE` is FALSE, and `x = 'a' OR TRUE` is TRUE |
    | NOT | over TRUE, FALSE or NULL: FALSE, TRUE or NULL |
    | `=`, `<>`, `<`, `<=`, `>`, `>=` | NULL where an operand is NULL, so `x = NULL` is NULL; else its truth, where the two operands are literals of one of these kinds: booleans, integers of any width, `string` or `varchar(n)` values in any mix, or dates |
    | BETWEEN, NOT BETWEEN | as `value >= low AND value <= high`, by the comparisons above and the `Kleene` rule; NOT BETWEEN is its NOT |
    | IN, NOT IN | as `value = a OR value = b …`, by the comparisons above and the `Kleene` rule; NOT IN is its NOT |
    | IS NULL | over a literal: TRUE for a NULL, and FALSE for any other literal |
    | CASE | the THEN of the first WHEN that is TRUE, past each WHEN that is FALSE or NULL; where no WHEN is left, the ELSE, or where there is none, NULL of the type of a THEN that is a literal |
    | `+`, `-`, `*`, `/`, `%`, `SafeDivide` | over two literals of integer types, in the bigger operand's type (ADR 0092, rule 3): NULL where an operand is NULL, but a `SafeDivide` with a NULL operand stays; else the value, where no overflow occurs and the divisor is not 0. `/` cuts toward 0: `-7 / 2` is `-3` |
    | negate | over a literal of an integer type, in its type: NULL for a NULL; else the value, where no overflow occurs |
    | LIKE, NOT LIKE, ILIKE, NOT ILIKE | NULL where an operand is NULL |
    | CAST | over a literal: NULL of the target type, for a NULL input; else the value that the canonical CAST gives (ADR 0086, rule 10), where it gives one: a number takes rule 10's rounding or cut, so `Cast(Double(1.7), integer)` is `Integer(1)`, and a string takes the value that it reads as, as a literal of the target type (ADR 0085, rule 6), so `Cast(String("2026-05-19"), date)` is `Date("2026-05-19")` |

    Rule 22 reads the truth value that these parts give to a predicate, and rule 23
    reads the literal that they give to a stored tree. Every other part stays as it is,
    for the engine, and the parts inside it fold:

    - a comparison of two literals of any other kind, such as two decimals, two `real`
      or `double` values, two `char(n)` values, two times or two timestamps, and a
      comparison of two literals of two kinds, such as an integer and a decimal;
    - a LIKE, NOT LIKE, ILIKE or NOT ILIKE of two literals that are not NULL:
      `'Paid Search' LIKE 'Paid%'` stays;
    - arithmetic with an operand that is not a literal of an integer type, such as a
      `real`, a `double`, a decimal or a date, also where the other operand is NULL:
      `x + NULL` stays;
    - integer arithmetic that overflows its type, a `/` or a `%` by 0, and a
      `SafeDivide` with a NULL operand or a divisor of 0;
    - a CAST of a literal that the canonical CAST gives no value for, such as
      `Cast(String("2026-13-45"), date)`: the engine raises its error;
    - a function call, of a scalar, an aggregate or a window function, `Immutable` or
      not. Every operation is immutable (ADR 0092, rule 1), so the fold evaluates
      nothing that is not `Immutable`;
    - a part whose literals do not decide it, such as `x AND TRUE`; a CASE where a WHEN
      before the first TRUE is not a literal; and a CASE where each WHEN is FALSE or
      NULL, with no ELSE and no THEN that is a literal.

22. The planner's pruning of a Unionset branch (ADR 0073, *A Unionset prunes a branch
    its predicate excludes*) runs this fold. The pruning reads a branch's signature
    through its links (ADR 0088, rule 15), so a link, or a chain of links, to an entry
    whose tree is a literal is that literal. A predicate whose fold gives no literal
    keeps the branch.

    The request predicate `measurement_channel = 'Paid Social'`, on a Unionset with three
    branches:

    | Branch | `measurement_channel` on the branch | The fold | Read |
    |---|---|---|---|
    | `adwords` | `{lit: "Paid Search"}` | `'Paid Search' = 'Paid Social'` gives FALSE | no |
    | `facebook` | `{lit: "Paid Social"}` | TRUE | yes |
    | `ga` | the column `channel` | not a literal: no result | yes |

    With `upper(measurement_channel) = 'PAID SEARCH'`, `adwords` is read, because the
    fold evaluates no function. A Dimension `ds2: {expr: dataset_name}` stores
    `Link(dataset_name)` (ADR 0088, rule 5). On a branch where the metadata Dimension
    `dataset_name` is `"google_ads"`, the pruning reads `ds2` as `'google_ads'`, so the
    request predicate `ds2 = 'google'` folds to FALSE, and the branch is not read.

    Where `adwords` maps the Dimension `day`, declared `date`, as `{lit: {string:
    "2026-05-19"}}`, rule 17 casts the literal, and the fold gives the cast's value, so
    compile stores `Date("2026-05-19")` (rule 23). The request predicate `day =
    '2026-05-18'` then folds to FALSE, and `adwords` is not read. DataFusion's constant
    evaluator folds a CAST of a literal in the same way (`datafusion-optimizer` 55.2,
    `expr_simplifier.rs:545-570`).

23. Compile folds each tree that it stores with the same fold, and `fold_binary`
    retires into it. So a metadata-determined signature whose parts are on rule 21's
    list is stored as a literal, as `CONTEXT.md` **Sugar** has it: "folding
    metadata-determined subtrees".

    | Dimension | Its tree where the metadata Dimension `dataset_name` is `"google_ads"` | Stored |
    |---|---|---|
    | `is_google: {eq: [dataset_name, {lit: "google"}]}` | `'google_ads' = 'google'` | `FALSE`: a literal, so a predicate on `is_google` can exclude the branch |
    | `total: {add: [1, 2]}`, declared `integer` | `Integer(1) + Integer(2)` (ADR 0092, rule 4) | `Integer(3)`: typed, then folded |

    In the tree that compile folds, a link to an entry whose tree is a literal (ADR 0088,
    rule 5) takes that literal's value, and so does a link to an entry whose tree is a
    link that takes one. So a chain of links to a literal takes its value, and each chain
    ends, because compile refuses a cycle of references (ADR 0088, rule 5). A link to a
    Metric whose `agg:` aggregates a formula that reads a Measure or a Metric, or a chain
    through one, takes no value: that Metric's value is its second aggregate step (ADR
    0074, *A Metric*). Where the fold of a Metric's formula would leave no link to a
    Measure or a Metric, compile stores the formula as written: the fold changes no
    Metric's kind (ADR 0074, *A Metric*), and no rule reads that folded value (rule 21).
    So `m`, with `agg: count` over `CASE WHEN FALSE THEN order_count ELSE 1000 END`,
    keeps that formula and gives 1 in total, and `m + m` stays `Link(m) + Link(m)` and
    gives 2, not 2000. The same formula with no `agg:` stays a Metric over the
    aggregates: a total over no rows gives 1000, and it is not planned as a Dimension.
    A part over
    such a link that the fold evaluates becomes its value, and a link under a part that
    the fold leaves as it is stays a link. So `is_google`, whose tree links the entry of
    `dataset_name`, is stored as `FALSE`, and so is `is_google2: {eq: [ds2, {lit:
    "google"}]}`, whose tree links the entry of `ds2: {expr: dataset_name}`.

### Ruled elsewhere, not here

- How the engine adapter writes a value that its engine does not take is not ruled here.

### Amends

- **0003:** in its *Decision Outcome*, "Name resolution, catalog fetch, glob expansion,
  relationship resolution and expression lowering all happen inside one async
  `compile`" holds for the names of semantics; parse resolves a function's name against
  the function catalog, which needs no I/O (ADR 0091, rule 1; ADR 0089, rule 4). The
  rest of 0003 stands.
- **0021:** in its *Decision Outcome*, "a narrowing physical→declared cast is a
  compile error demanding an explicit author cast" becomes the cast to a
  semantics' declared type that ADR 0089, rules 17 and 18, give. The rest of
  0021 stands.
- **0073:** in *A Unionset prunes a branch its predicate excludes*, "an authored
  literal body is a `Literal` too" holds also for a tagged literal whose type
  differs from its semantics' declared type: compile casts it to the declared
  type, and the fold stores the cast's value as a literal (ADR 0089, rules 17, 21
  and 23); where the canonical CAST gives that literal no value (ADR 0086, rule
  10), the fold leaves the cast, so its leaf is not replaced, and a predicate that only that leaf
  decides keeps the branch; "The fold is a shared helper. It is a pure function
  over expressions" becomes the fold of ADR 0089, rules 21 to 23. The rest of
  0073 stands.
- **0081:** in *A partition key is a column* and in *`partition_def` names the
  Dimension*, "its signature is a bare `Column`" holds for the mapping: the
  Dimension maps the column directly, and because a partition column is never
  observed, its stored tree is the cast of ADR 0089, rule 16, over the column;
  "The signature of `order_year` on `sales` is `Column(year)`" and "the
  signature is still `Column(year)`" read with that cast. In *The manifest stores
  no link*, "a signature that is a bare `Column`" reads as the signature of a
  Dimension that maps the column directly. The rest of 0081 stands.
- **0082:** in *Which semantics types the column*, for a column that compile does
  not observe, "Its signature's tree is then the column" reads with the cast of
  ADR 0089, rule 16; "A `count` or `count_distinct` takes part" also covers an
  `approx_count_distinct` (ADR 0089, rule 16). In *Compile writes the read type into the schema*,
  every other read of such a column, in an expression mapping, a Filter or a
  Measure's `filter:`, takes the read type too (ADR 0089, rule 16). The rest of
  0082 stands.

### Consequences

- Good, because typing has one implementation, which compile, the planner and the
  Substrait producer call, as in Calcite and DataFusion.
- Good, because parse refuses a call that does not fit its declaration with no
  catalog, and a function in the catalog parses wherever it is written.
- Good, because an operation's type family check applies to an observed column and
  to a column that a semantics maps: `region = 5` is refused.
- Good, because an enum argument is written bare, as in SQL, or as a string literal.
- Good, because a `sum` Measure declared at its column's type compiles, and a value
  outside the declared type's range fails at run time instead of giving a wrong value.
- Good, because a semantic mapping's literal and an expression's literal have one
  grammar.
- Good, because the planner and compile fold with one fold, and a metadata-determined
  signature that rule 21's list decides is stored as a literal, a literal under the
  cast to its declared type included.
- Good, because the planner reads each unobserved column's read type from the
  schema, so an authored cast over the column cannot change the type that the plan
  reads it as.
- Good, because `agg:` names any catalog aggregate, and the aggregate's declaration
  checks the call.
- Neutral: `lower.rs`'s typing functions, `fold_binary`, the tag tables in `sugar.rs`,
  `LiteralValue` and `AggregationType` retire, and `semstrait-model` imports
  `semstrait-sql`.
- Neutral: the corpus changes 15 `{int: …}` literal tags.
- Neutral: for a column that compile does not observe, each tree of a semantics that
  maps it directly stores a cast over it, and each changed tree stores other bytes.
- Neutral: over a column that the catalog observed, a `sum`, an `avg`, a `min` or a
  `max` whose declared type differs from the column's type stores a cast over the
  column.
- Neutral: where the declared type of a `count` or an `approx_count_distinct` is not
  `bigint`, its tree stores a cast over the aggregate.
- Neutral: `NarrowingCast` retires, and ADR 0044 counts it among compile's warnings
  (*Twenty hard rejections exist and five carry a number*); ADR 0044 is not amended for
  it.
- Neutral: compile computes ADR 0082's read type once and writes it into the
  model-fed schema, which the planner reads.
- Bad, because a column that compile did not observe, and that no semantics maps,
  stays `Unknown`, so `channel = 5` in a Filter compiles there, and the planner
  refuses the column (`PlanError::UnresolvedType`). The gain is that compile types a
  column only by what it observed or by the type that the plan reads it as (ADR 0061,
  ADR 0082).
- Bad, because digits after the point are lost at a declared type with no message:
  `avg` over `decimal(18,2)` gives `decimal(38,6)`, and a declared `decimal(18,2)`
  rounds it. The gain is that the declared type means what the author wrote.
- Bad, because `Call::new` refuses an aggregate whose declaration needs more than the
  Measure's value, such as `corr`, `string_agg` and `percentile_cont`: the simplest
  form gives one argument. The gain is that `agg:` keeps one form.
- Bad, because a branch that only a part outside the fold's list would exclude is read,
  and the request predicate that the plan applies in it keeps none of its rows. With
  the request predicate `measurement_channel LIKE 'Paid So%'` on rule 22's Unionset,
  the fold leaves `'Paid Search' LIKE 'Paid So%'` as it is, so `adwords` is read, and
  the predicate applied in it keeps none of its rows; the request's result does not
  change. The gain is that the fold holds only what a rule reads, as the MVP needs,
  and it gives no value that no declaration states, such as whether `\` escapes in a
  LIKE pattern.
- Bad, because over the `double` rows 1.35, 1.35, 1.35 and 1.35, a `sum` declared
  `integer` gives 4: each row is cut before the sum (rule 16), where a cast of the sum
  alone gives 5. The gain is that a Measure gives one answer whether or not a catalog
  observed its column.
- Bad, because an author cannot write a timezone other than `'UTC'` in the semantic model's
  `date_trunc` for now. The gain is that every stored timestamp is an instant, which a
  request predicate, a bucket and a join read alike.
- `CONTEXT.md` gains the entry **Fold**.

## More Information

The question was where expression typing and validation run once semstrait-sql
exists. `agg:` is an enum
argument, a read type for every other read of an unobserved column, the canonical CAST's
rounding and a cast from `Unknown`, and the clause check and the type
systems that rule 12 reads are set. The untagged literal's leaf is set, and `approx_count_distinct` casts nothing on
its column, as `count` does. A later ruling, on 2026-10-14, set the
fold's closed list. The human's rulings of 2026-10-15
keep today's
form of a CASE, and have the writer keep each literal's type. The human's rulings
set where
typing's rules sit, the mapping form of a call's arguments and the default that it
gives a skipped optional parameter, a bare value at a constant parameter, and a chain
of links in the fold and in the pruning. The human's rulings on the reconciliation of the
decision with the ADRs and the implemented
semstrait-sql (2026-10-17), ruled the
read type that compile writes into a model-fed schema, the compile error on an
unmapped catalog type, the cast over an observed column, the fold of a literal's CAST, and
the defaults of `date_trunc`, with its `'UTC'` timezone in the semantic model. The
decisions (2026-10-17 and
2026-10-18), give the formula push-down its casts, and have compile build a
row-level Metric's `agg:` call, and its cast, as a Measure's: its formula over
Dimensions is one expression, as a Measure's `expr:` is. They give the planner
the cast above a Metric's second aggregate step. The decisions give the planner the
cast of a row argument in a Unionset branch's row projection (2026-10-21).
