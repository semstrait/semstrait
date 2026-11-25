---
number: 91
title: A function has a canonical declaration and a dialect declaration where the engine differs, and a call is a call of its function's type
status: accepted
date: 2026-10-08
source:
- CONTEXT.md
- docs/adr/0002-single-leaf-parameterized-expression-tree.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0022-engine-independent-canonical-vocabulary.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md
- docs/adr/0083-a-metric-declares-no-additivity.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- '`Function`, `Declaration`, `Aggregate` and `Window` in `crates/semstrait-sql/src/functions/mod.rs`; `FunctionRef` in `crates/semstrait-sql/src/functions/handle.rs`; the `Call` node in `crates/semstrait-sql/src/expr/call/mod.rs`'
- a prototype of canonical and engine function declarations (Prototype A)
- the research on DataFusion 55 function declarations and on the expression structure of query engines
- '`calcite@calcite-1.42.0` (`c01f6b5`): `core/src/main/java/org/apache/calcite/rex/RexCall.java:40-68`, `rex/RexOver.java:35-45`, `rel/core/AggregateCall.java:61-76`, `:138-141`, `tools/RelBuilder.java:4524-4527`, `:4615-4619`, `sql/fun/SqlStdOperatorTable.java:209-216`, `:1488-1492`, `:1734`, `util/Optionality.java:23-39`, `sql/SqlWindow.java:345-395`, `:678-701`; `datafusion-expr@55.0.0:src/expr.rs:949-954`, `:1114-1130`, `:1246-1268`, `src/window_frame.rs:127-137`, `:263-268`; `datafusion-sql@55.2.0:src/expr/function.rs:724-770`; `substrait@0.63.0`: `substrait/text/simple_extensions_schema.yaml:199-203`, `substrait/text/dialect_schema.yaml:192-234`'
- the human's rulings of 2026-10-08 and 2026-10-09, of 2026-10-15, and of 2026-10-16 in the grilling
- the human's rulings of 2026-10-17
- 'the decision (a dialect is read at a release); `calcite@calcite-1.42.0`: `core/src/main/java/org/apache/calcite/sql/SqlDialect.java:1538`, `sql/dialect/MysqlSqlDialect.java:157-160`, `sql/dialect/HiveSqlDialect.java:65-67`'
links:
- target: 21
  kind: amends
- target: 22
  kind: amends
- target: 30
  kind: amends
- target: 68
  kind: amends
- target: 72
  kind: amends
---

# A function has a canonical declaration and a dialect declaration where the engine differs, and a call is a call of its function's type

## Context and Problem Statement

ADR 0085 moves the expression tree to `semstrait-sql`. The functions move with
it, and they need a decision of their own:

- **A call holds a name.** The tree has three call nodes: `FunctionCall { name:
  CanonicalFn, args }`, `Aggregate { op: AggregateKind, .. }`, whose
  `AggregateKind` is one of five built-ins (`AggregationOp`) or an extension,
  and `Window { function: WindowFn, .. }`. Compile looks each name up in the
  registry.
- **A function's declaration holds no behaviour.** `FunctionSpec` holds a name,
  a function type, its signatures, a return rule and an additivity. It declares
  no NULL behaviour and no volatility: ADR 0021 says "NULL propagation is
  delegated to the engine" (*Decision Outcome*).
- **Engine identity has one home.** ADR 0068: "engine identity's only home is
  an `impl SubstraitRules`" (*Amends*, its ADR 0018 item). The research lists where
  DataFusion 55's own function differs from the canonical one, in name,
  arguments, result type or NULL behaviour. It has `signum` where the canonical
  function is `sign`, and its `concat` skips NULL arguments.

Prototype A declares a canonical function as `Decl<K: Kind>`, whose fields per
function type are associated types (`K::Props`, `K::Opt`), and a DataFusion
function as an `EngineFn` in the adapter, beside rewrite templates. The decision
rules out the templates and puts a dialect declaration in
`semstrait-sql`: it "declares the function as the engine has it: its typing,
its behaviour and the options that change it".

The engines draw the call two ways. Calcite has one typed call node, `RexCall
{ op, operands, type }`, for operators and functions alike: "The smarts should
be in the operator, rather than the call" (`RexCall.java:51-52`). A windowed
call adds its window to it (`RexOver`), and a grouped aggregate holds its
clauses as fields (`AggregateCall`). DataFusion has three call nodes, one per
function type. No engine in the research holds an engine's variant inside a
function's declaration: Substrait writes it in a dialect file per engine.

## Considered Options

- **One `Function` node whose call is typed by its function type, and a
  function that holds its canonical declaration and a declaration per dialect
  under it, all of one struct** (chosen).
- **Each dialect's declarations in a list of their own, each pointing at the
  canonical declaration it serves**, as Substrait's dialect file does. The
  human: "canonical - is prioritized form, and dialect level - under
  declration". A later feature lets a model declare its dialect, and its calls
  are then checked against that dialect's declarations.
- **A closed enum of functions**, Prototype B's `ScalarCall<E>`, with one
  `Lowerings` value per engine that holds one field per function. The human, choosing between this and the dialect declarations under each
  canonical one: "Under this consideration b is better, or enum form - which
  basically says - one of". The "one of" is the list of dialect declarations
  under the canonical one (rule 3), and `FunctionRef` over the three function
  types (rule 1).
- **Prototype A's `EngineFn` in the adapter**, with aliases and the option
  values it implements. The human: "I think aliases not needed, because name
  per dialect is exactly the name for engine. EngineFn shouldn't be in place ,
  because function declarations are already what we declare within dialect.
  You atr trying to implement substrait logic which is not needed here - we
  need to declare function, their form, and behavioral options, and their
  representation, having the same struct per dialect".
- **Prototype A's `Decl<K: Kind>`**, whose fields per function type are
  associated types. The human: "generics must be correct with name …
  not something like S:: Leaf".
- **Three call nodes that hold handles**, three variants of the tree for a
  call, as DataFusion's `Expr` has. The chosen design keeps one `Function` node
  and types the call inside it (rule 1). The human: "such wrappers are redudant
  because function declration is already declares call".
- **One call struct for every function type**, whose function type is only its
  handle's variant (`FunctionRef`), with one set of clauses for all three. Its
  Rust type then holds DISTINCT on `upper` and OVER on `sum`, and a node that
  takes only an aggregate, such as the plan's aggregation node, cannot say so
  by its type. The human: "fucntion call doesn't consists identity of function
  type - scalar, aggregate".
- **A call body per clause profile**, such as an ordered-set aggregate's
  WITHIN GROUP body. `count` and `max` share a profile, but DISTINCT is
  `Allowed` on one and `NoEffect` on the other, so the declaration still
  decides, and a new function with a new profile needs a new body type. No
  reference design has it: DataFusion, Substrait and Calcite each hold one body
  per function type, and the function's declaration says which clauses its
  calls take.
- **The additivity on each declaration, the same on every dialect.** Each
  dialect declaration then holds a copy, and no rule says which copy a reader
  takes. Substrait writes `decomposable` once, on the function, and its dialect
  file has no such field. **A part of `Function<F>` that depends on `F`**
  (`F::Traits`) holds it once, but it is an associated type, which rule 3
  refuses.
- **Each clause as a wrapper node** (`Expr::Filter`, `Expr::Over`), as in
  Calcite's parse tree. The order and validity of clauses become a matter of
  tree shape, and Calcite removes these wrappers before it types a call.
- **One concept for clauses and options.** A clause is what a call writes in
  SQL, and an option is a behaviour that SQL cannot write, so each word keeps
  one meaning.
- **A function with no canonical declaration**, for a function only DataFusion
  has. The human: "function is canonical if it is declraed with canonical
  shape".
- **The function's additivity in `semstrait-model`**, beside a Measure's
  declared additivity. The human: "We can preserve additivity, for fucntion
  itself".
- **No option values on a call until an authoring form sets one.** The human:
  "b is correct, even if right now it is not used, it's make it more
  complete".
- **No mark for a function that a dialect does not have, so only the engine's
  rules know.** `on(DataFusion)` then gives the canonical declaration of
  `regexp_extract`, a name that DataFusion refuses.
- For a dialect's result type: **DataFusion's own result type, where one of
  rule 4's return rules can say it**, as `signum`'s `double`, **or only where
  the type family differs.** DataFusion's result is how it holds a value
  inside, which is the engine's side, and a correct plan holds Substrait's
  types. The human: "everything that don't bring value to that goal, is not the
  objective of implementaiton". **Each one, with new return rules where rule
  4's cannot say it**, such as a decimal whose scale is a constant argument
  (`round`) and an unsigned integer, a type that semstrait does not have.

## Decision Outcome

In short:

- Every function call in a tree is one node, `Function(Call)`, and a node that
  takes only an aggregate holds an `AggregateCall` itself, as Substrait's
  `AggregateRel.Measure` does. A call is a scalar, an aggregate or a window
  call, by its function's type. It holds its function, its arguments, its
  option values and only the clauses that its function type takes.
- `semstrait-sql` builds every call. When it builds a call, it checks the call
  against its function's declaration, and a window call's `Over` against SQL's
  rules for a window frame.
- A function has one canonical declaration. Under it, each dialect where the
  engine's function differs has a declaration of the same struct, and a dialect
  marks a function that it does not have. An aggregate function holds its
  additivity once, beside its declarations.
- `Dialect` is `DataFusion`, `Substrait` or `Iceberg`. Engine identity has two homes: a
  dialect's declarations and its `impl SubstraitRules`.

### Every function call is one node

1. A call is a call of its function's type, and it holds its function and what
   it writes:

   ```rust
   pub enum Call<L: Leaf, A: Aggregation> {
       Scalar(ScalarCall<L, A>),
       Aggregate(AggregateCall<L, A>, A),
       Window(WindowCall<L, A>),
   }

   pub struct ScalarCall<L: Leaf, A: Aggregation> {
       function: &'static Function<Scalar>,
       args: Vec<Expr<L, A>>,
       options: Vec<OptionSetting>,           // empty where the canonical values apply
   }

   pub struct AggregateCall<L: Leaf, A: Aggregation> {
       function: &'static AggregateFunction,  // rule 3
       args: Vec<Expr<L, A>>,
       options: Vec<OptionSetting>,
       distinct: bool,
       filter: Option<Box<Expr<L, A>>>,
       order_by: Vec<SortKey<L, A>>,          // ORDER BY in the call, or WITHIN GROUP: the declaration says which
       nulls: Option<NullTreatment>,          // IGNORE NULLS or RESPECT NULLS
   }

   pub struct WindowCall<L: Leaf, A: Aggregation> {
       function: &'static Function<Window>,
       args: Vec<Expr<L, A>>,
       options: Vec<OptionSetting>,
       nulls: Option<NullTreatment>,
       over: Over<L, A>,                      // PARTITION BY, ORDER BY and the frame
   }

   pub enum FunctionRef<A: Aggregation> {     // a handle to a function, as the registry gives it (ADR 0087, rule 8)
       Scalar(&'static Function<Scalar>),
       Aggregate(&'static AggregateFunction, A),
       Window(&'static Function<Window>),
   }
   ```

   | Expression | Before | Now |
   |---|---|---|
   | `upper(name)` | `FunctionCall { name: "upper", args }` | `Function(Scalar(ScalarCall { function: &UPPER, args, .. }))` |
   | `sum(amount) FILTER (WHERE paid)` | `Aggregate { op: Builtin(Sum), args, distinct: false, filter: Some(paid), capability }` | `Function(Aggregate(AggregateCall { function: &SUM, args, filter: Some(paid), .. }, capability))` |
   | `lag(cost) OVER (PARTITION BY channel ORDER BY day)` | `Window { function: Lag, args, partition_by, order_by, frame: None }` | `Function(Window(WindowCall { function: &LAG, args, over: .., .. }))` |

   - A call's type is its function's type, as in DataFusion
     (`ScalarFunction`, `AggregateFunction`, `WindowFunction`), Substrait (the
     three messages of the same names) and Calcite (`RexCall`, `AggregateCall`,
     `RexOver`). Each holds only the clauses that its function type takes: a
     scalar call holds none, an aggregate call holds no OVER, and a window call
     holds no DISTINCT and no FILTER. So a node that takes only an aggregate
     holds an `AggregateCall`, as the plan's aggregation node does (ADR 0090,
     rule 2), and an aggregate over a window is not a call.
   - The handle is a reference to the function's `static`. It compares and
     hashes by name. So a tree cannot hold an unknown function, and compile,
     the planner and the adapter read the declaration with no lookup.
   - `A` sits on `Call::Aggregate` and `FunctionRef::Aggregate`. `NoAggregate`
     has no values, so a `RowExpr` still cannot hold an aggregate. The
     authoring form still writes no window function, as ADR 0002's
     "unauthorable `Window`" has it, and compile still refuses an authored
     window (ADR 0066, ruling 7).
   - `Call` gives `name()`, `args()` and `options()` for every call, so only a
     reader of a clause matches the call's type.
   - `Over` holds the partition, the order keys and the frame. An order key is
     a `SortKey`, which carries ADR 0076's `SortDir` and `NullOrdering`.

2. A call's fields are private, and `semstrait-sql` builds every call.
   `Call::new` builds the call of its function's type with no clause, and each
   typed call has a constructor that also takes the call's clauses. Each
   constructor checks the call once the call is complete: against its
   function's declaration, as Calcite's `AggregateCall` and `RelBuilder` do
   (`AggregateCall.java:138-141`, `RelBuilder.java:4524-4527`, `:4615-4619`),
   and a window call's `Over` against SQL's rules for a window frame. It
   checks:

   - the arguments against the signature (rule 4): their number, a literal at
     an enum or a constant parameter, and an enum value that the declaration
     has;
   - each clause against the declaration's `ClauseRule` (rule 4):

   | `ClauseRule` | A call that writes the clause | A call that does not write it |
   |---|---|---|
   | `Allowed` | kept: `count(DISTINCT x)` | built |
   | `Required` | kept: `percentile_cont(0.5) WITHIN GROUP (ORDER BY price)` | refused: `percentile_cont(0.5)` with no WITHIN GROUP, as `Call::new` gives it |
   | `NoEffect` | refused. Parse drops the clause first (ADR 0089, rule 5), so `max(DISTINCT x)` parses as `max(x)` | built |
   | `Refused` | refused: `corr(DISTINCT y, x)`, `row_number() IGNORE NULLS` | built |

   - a window call's `Over`, which holds PARTITION BY, ORDER BY and the window
     frame, against SQL's rules for a window frame. These are refused: a start
     of UNBOUNDED FOLLOWING; an end of UNBOUNDED PRECEDING; an end whose kind
     comes before the start's kind, in the order PRECEDING, then CURRENT ROW,
     then FOLLOWING (`CURRENT ROW AND 1 PRECEDING`,
     `1 FOLLOWING AND 1 PRECEDING`, `1 FOLLOWING AND CURRENT ROW`); a RANGE
     window frame with an offset bound and not exactly one ORDER BY key; and a
     GROUPS window frame with no ORDER BY. Calcite's `SqlWindow` refuses the
     two bounds, an end whose kind comes before the start's kind, and a RANGE
     window frame with an offset bound and two or more ORDER BY keys, or with
     no ORDER BY where the input is not sorted already
     (`SqlWindow.java:345-395`, `:678-701`). DataFusion's `WindowFrame` refuses
     the two bounds, the RANGE window frame and the GROUPS window frame
     (`datafusion-expr@55.0.0:src/window_frame.rs:127-137`, `:263-268`). The
     check compares the kinds of the two bounds, not their offsets, and neither
     engine compares two offsets, so `5 FOLLOWING AND 2 FOLLOWING` is built.

   So no call holds a clause that its declaration refuses or marks `NoEffect`,
   and every crate that builds a call gets the same check: parse, compile, the
   planner and reading a manifest (ADR 0089, rule 5). DataFusion checks a
   call's clauses only when it plans SQL text
   (`datafusion-sql@55.2.0:src/expr/function.rs:724-770`), and Substrait
   leaves the check to each consumer.

### A function has one canonical declaration, and a declaration per dialect where the engine differs

3. A function is its canonical declaration, then one entry for each dialect
   where that dialect differs. Every declaration is one struct:

   ```rust
   pub struct Function<F: FunctionType> {
       pub canonical: Declaration<F>,
       pub dialects: &'static [(Dialect, Option<Declaration<F>>)], // None: the dialect has no such function
       pub releases: &'static [(Dialect, RangeInclusive<Release>, Option<Declaration<F>>)], // only where those releases differ
   }

   impl<F: FunctionType> Function<F> {
       /// The dialect's declaration, the canonical one where the dialect has none,
       /// or `None` where the dialect has no such function.
       pub fn on(&'static self, dialect: Dialect) -> Option<&'static Declaration<F>>;
       /// The `releases` entry that holds `release`, else `on(dialect)`.
       pub fn on_release(&'static self, dialect: Dialect, release: Release) -> Option<&'static Declaration<F>>;
   }

   pub struct Declaration<F: FunctionType> {
       pub name: &'static str,       // on a dialect: the engine's own name
       pub aliases: &'static [&'static str], // other YAML spellings (rule 4); none on a dialect
       pub signature: Signature,     // the form: the arguments
       pub returns: ReturnRule,
       pub nullability: Nullability,
       pub volatility: Volatility,
       pub specific: F,              // what only this function type has
   }

   pub struct Scalar    { pub category: Category, pub options: &'static [OptionDecl<ScalarOption>] }
   pub struct Aggregate { pub options: &'static [OptionDecl<AggregateOption>], pub clauses: AggregateClauses }
   pub struct Window    { pub options: &'static [OptionDecl<WindowOption>],    pub clauses: WindowClauses }

   pub struct AggregateFunction {
       pub declarations: Function<Aggregate>,
       pub additivity: Additivity,          // the plain form's, on every dialect
       pub distinct_additivity: Additivity, // the DISTINCT form's, on every dialect
   }
   ```

   - `Declaration` holds what every function has, as Calcite's `SqlOperator`
     does for `SqlFunction` and `SqlAggFunction`. `F` adds what only its
     function type has.
   - Only an aggregate function has an additivity. `AggregateFunction` holds
     its declarations and, once, the additivity of its plain form and of its
     DISTINCT form, so no dialect declaration holds a copy. Substrait does the
     same: `decomposable` is a property of the function
     (`simple_extensions_schema.yaml:199-203`), and its dialect file has no
     such field (`dialect_schema.yaml:192-234`). The human: "only Aggregate
     function spell additivity, others do not need it, polymoprhic shape is ok,
     it can depend on function type - scalar, aggregate, window. which rather
     cleaner design then something bad". `FunctionRef::Aggregate` and
     `AggregateCall` hold an `AggregateFunction` (rule 1). Substrait's proto
     has a message of the same name, which only the producer writes.
   - `FunctionType` is a sealed trait that only `Scalar`, `Aggregate` and
     `Window` implement. It has no associated types.
   - A dialect's entry is `None` where the dialect has no such function, and
     the engine's rules then write the call from other functions. ADR 0087 rule
     5 names DataFusion's.
   - A `dialects` entry holds for each release of its dialect that no
     `releases` entry holds, and declares what the dialect's reference
     release has (rule 6). A `releases` entry holds for the releases that it
     names, where they differ from the dialect's reference release, and
     `on_release` reads it first. ADR 0087 rule 23 names DataFusion 53's
     `extract`.
   - The canonical declaration is required: "function is canonical if it is
     declraed with canonical shape" (the human). What is generic for semstrait
     can differ from what is generic for an engine.

   ```rust
   pub static SIGN: Function<Scalar> = Function {
       canonical: Declaration { name: "sign", returns: Fixed(Integer), .. },
       dialects: &[(Dialect::DataFusion, Some(Declaration { name: "signum", .. }))],
       releases: &[],
   };
   pub static UPPER: Function<Scalar> = Function { canonical: Declaration { name: "upper", .. }, dialects: &[], releases: &[] };
   pub static REGEXP_EXTRACT: Function<Scalar> = Function {
       canonical: Declaration { name: "regexp_extract", .. },
       dialects: &[(Dialect::DataFusion, None)],
       releases: &[],
   };
   ```

4. A declaration's parts are Prototype A's, less its `syntax` field and its
   `Arithmetic`, `Comparison` and `Logical` categories,
   and less its `Aggregate` and `Window` categories, which the function type
   already says:

   | Part | What it says | Example |
   |---|---|---|
   | `name` | the generic name; on a dialect, the engine's own | `upper`; `signum` |
   | `aliases` | the other YAML spellings that parse accepts for the same element, with no other change. A dialect declaration has none: its name is the engine's own | `nullif`: `null_if` |
   | `category` (scalar) | String, Math, Temporal, Pattern, Conditional or Array | String |
   | `signature` | for each parameter: its type (an exact type, a type family, one type at any precision, any type, or a type variable `T`; ADR 0086, rule 8), its form (value, constant or enum) and its presence (required, optional, or only when an earlier argument says so); for an optional parameter that a later parameter follows, its default, and for a parameter that is present only when an earlier argument says so, a default if it has one; a variadic tail; constraints across arguments | `(value: String)` |
   | `returns` | the rule for the result type: Fixed, Var, SameAs, Promote, or a derivation that the type system holds (ADR 0092, rule 3): Sum, Product, Quotient, Mod, SumOf, AvgOf, DecimalScaleZero or RoundOf | Fixed(String) |
   | `nullability` | when the result can be NULL: Mirror, MirrorArgs, AllArgs, Always, Never, ByOption, EmptyGroupNull or Kleene (`Kleene`: ADR 0092, rule 7) | Mirror: NULL if an argument is NULL |
   | `volatility` | Immutable, Stable or Volatile | Immutable |
   | `options` | for each option: its name, its values and its canonical value | `concat`'s `null_handling` |
   | `clauses` (aggregate) | a `ClauseRule` for DISTINCT, FILTER, ORDER BY in the call, WITHIN GROUP (with its sort key's type), and IGNORE NULLS or RESPECT NULLS. An aggregate call holds no OVER (rule 1) | `count`: DISTINCT and FILTER `Allowed`, ORDER BY `NoEffect` |
   | `clauses` (window) | a `ClauseRule` for IGNORE NULLS or RESPECT NULLS, PARTITION BY and ORDER BY, and the frame | `row_number`: frame `NoEffect` |
   | `additivity`, `distinct_additivity` (aggregate function) | the additivity of the plain form and of the DISTINCT form: Additive, SemiAdditive or NonAdditive, with no axes. `AggregateFunction` holds them once, beside its declarations (rule 3) | `count`: Additive; NonAdditive |

   - A dialect's enum names each canonical value in the dialect's words,
     except a value that a release's entry has no word for, which the
     engine's rules then write (ADR 0087, rule 23).
   - An optional parameter that a later parameter follows declares its default,
     a literal of the parameter's type: `regexp_instr`'s `start` declares 1 (ADR
     0087, rule 12). Parse writes the default where a call's mapping names a
     later parameter and not this one, so the call's arguments stay in the
     declaration's order (ADR 0089, rule 4). The seal refuses such a parameter
     with no default, and a default that is not a literal of the parameter's
     type (ADR 0087, rule 9). A last optional parameter needs none: a call that
     does not write it holds no argument for it.
   - A parameter that is present only when an earlier argument says so may
     declare a default, a literal of the parameter's type. Where the earlier
     argument says that the call writes the parameter, and the call leaves it out,
     the call takes the default (ADR 0089, rule 4): `date_trunc` declares
     `timezone` `'UTC'` and `week_start` `'monday'` (ADR 0087, rule 14).
   - `ClauseRule` is `Refused`, `Allowed` (it changes the result), `NoEffect`
     (DISTINCT on `max`) or `Required` (WITHIN GROUP on `percentile_cont`).
     These are Calcite's four `Optionality` states (`Optionality.java:23-39`).
   - A frame has no effect on `row_number`, `rank`, `dense_rank`,
     `percent_rank`, `cume_dist`, `ntile`, `lag` and `lead`. It is `Allowed` on
     `first_value`, `last_value` and `nth_value` (ADR 0087, rule 19), and a call
     that writes none gets SQL's default: with an ORDER BY, `RANGE BETWEEN
     UNBOUNDED PRECEDING AND CURRENT ROW`, which ends at the current row's
     last peer; with none, the whole partition.
   - A clause is SQL syntax that the call writes. An option is a behaviour
     that SQL cannot write, such as `concat`'s `null_handling`. Its canonical
     value is semstrait's own rule, since Substrait gives an option no default.
     IGNORE NULLS and RESPECT NULLS are clauses, not Prototype A's
     `null_treatment` option.

5. A dialect declaration declares the function as the engine has it. It exists only where the engine's function differs in its name,
   its arguments, its NULL behaviour, the clauses that a call may write, or its
   result type, and a function has at most one `dialects` entry
   per dialect.

   - A name differs where the engine does not find the canonical name as its
     function's name or alias.
   - It holds no rewrite: the engine adapter reads the difference and builds
     the rewrite from the declarations, so where the name differs, the rules
     write the dialect declaration's name.
   - Its own options say how the engine behaves: an option's values are the
     ones the engine's function gives, and its canonical value is what the
     engine does when the call sets none. DataFusion's `concat` declares
     `null_handling` with the one value `IGNORE_NULLS`.
   - A DataFusion declaration gives the canonical result type. DataFusion's own
     result type is no part of its declaration: the plan holds Substrait's
     types, a call's type in it is the one that Substrait's type system gives
     (ADR 0092, rule 3), and DataFusion's consumer turns each one into its own
     Arrow type. The human: "Datafusion in plan uses substrait's declaration
     which then maps types into its own internal representation ( arrow ), so
     what we want - is high level representation - not how internally
     datafusion operates - because it is going to be on engine side - we need to
     correct logical plan".

6. `Dialect` is an enum in `semstrait-common` with the variants `DataFusion`,
   `Substrait` and `Iceberg`, and `#[non_exhaustive]` (ADR 0007). It is the identity of a
   dialect's name and carries no adapter identity. It is
   a new type: ADR 0068 retires `semstrait-ir::artifact`'s `Dialect` trait
   (*Artifacts*; ADR 0090, rule 7).

   - `Substrait` is a dialect for typing: a vocabulary that several engines
     implement, each with its own extensions. Today it declares data types only
     (ADR 0086, rule 6), so a function and an operation have no Substrait
     declaration, and the producer writes "Substrait's official" spellings for
     them (ADR 0068, *Defaults are faithful*; ADR 0087, rule 24). The human: "we
     don't need function declaraiton for substrait - because substrait declare
     function more broader then datafusion does. so no value for now in this".
   - `Dialect` has no value for the canonical vocabulary: a manifest names
     `canonical` in its own `dialect` field (ADR 0088, rule 8).
   - A dialect that has releases is read at one of them. `Release` is a
     release's major, and `DialectRelease` is a dialect and its release, both
     in `semstrait-common`. A caller writes it `name=release`, as
     `datafusion=53`. A bare name is the dialect's reference release, which
     its `dialects` entries declare: DataFusion 55. DataFusion's oldest
     release is 53. Its parse refuses a release below the oldest, a release
     on a dialect that has none (`Substrait`, `Iceberg`), and a release that
     is not a major (`53.*`, `53.1`). A release that no `releases` entry
     holds reads the dialect's reference release's declarations, a release
     above the reference included. Calcite does the same: a dialect
     reads the database's major version (`SqlDialect.java:1538`) and branches on
     it (`MysqlSqlDialect.java:157-160`, `HiveSqlDialect.java:65-67`), and its
     function vocabulary, `SqlLibrary`, has no version.

7. Engine identity has two homes: a dialect's declarations in `semstrait-sql`,
   and its `impl SubstraitRules`, which holds every rewrite. Both read the
   dialect's release (rule 6). No canonical
   crate branches on engine identity, and the plan holds only canonical names
   (ADR 0072).

8. Each canonical declaration says when its result is NULL, for the call's
   option values. Where an engine differs, its dialect declaration says so,
   and the engine's rules write the canonical behaviour. An aggregate
   function's additivity, and that of its DISTINCT form, is held once on its
   `AggregateFunction` (rule 3), so every dialect reads the same one. A
   Measure's authored additivity narrows it, and for a DISTINCT call it
   narrows the DISTINCT form's (ADR 0030, as ADR 0083 amends it: a Metric
   declares no additivity).

### Amends

- **0021:** in its *Decision Outcome*, "NULL propagation is delegated to the
  engine" becomes, for a function, what ADR 0091, rule 8, says of when its
  result is NULL, and "Signatures require exact `DataType` equality" becomes: a
  parameter takes the type its declaration names (ADR 0091, rule 4; ADR 0086,
  rule 8). The rest of 0021 stands.
- **0022:** in its *Decision Outcome*, "Per-engine mappings live in a separate
  "Living" `registry/` tree that canonical specs may reference but never depend
  on" becomes: an engine's own declarations live in `semstrait-sql`, as dialect
  declarations under each canonical declaration. No canonical declaration
  depends on them. The rest of 0022 stands.
- **0030:** in its *Decision Outcome*, "the *restriction* of
  `FunctionSpec.additivity` by the model declaration" becomes: the restriction
  of the aggregate function's declared additivity by the model declaration: the
  declared additivity of its DISTINCT form for a DISTINCT call. The rest of 0030
  stands.
- **0068:** in its own *Amends* item on 0018, "engine identity's only home is an
  `impl SubstraitRules`" becomes: engine identity has two homes (ADR 0091, rule
  7): its dialect's declarations, which say how the engine has each function and
  each operation that differs, and its `impl SubstraitRules`. The rest of 0068
  stands.
- **0072:** in its own *Amends* item on 0068, "the plan holds only canonical
  names: engine identity stays in the rules" becomes: the plan holds only
  canonical names: engine identity stays in the rules and in its dialect's
  declarations. The rest of 0072 stands.

### Consequences

- Good, because a call's Rust type is its function's type, so DISTINCT on
  `upper` and OVER on `sum` cannot be built, and a node that takes only an
  aggregate holds an `AggregateCall`.
- Good, because every crate builds a call through `semstrait-sql`, which checks
  it against the declaration, so no call holds a clause that its declaration
  refuses.
- Good, because a function's NULL behaviour, volatility and call clauses are
  declared, and an engine's difference is a declaration the adapter reads, not
  a rule it holds alone.
- Good, because a dialect declaration is a full declaration of the same struct,
  so a model that later declares its dialect can be checked against it with no
  second form.
- Good, because an aggregate function's additivity is written once, so no
  dialect declaration can hold another.
- Good, because `on(DataFusion)` says when DataFusion has no such function, with
  no wrong answer.
- Bad, because a call is one of three types, and a reader of a clause matches
  the call's type. The gain is that the Rust types refuse a clause of another
  function type.
- Bad, because a dialect declaration repeats the parts that do not differ. The
  gain is a full declaration of one struct, which a model that later declares
  its dialect is checked against.
- Bad, because no engine in the research holds an engine's variant inside a
  function's declaration. The gain is that a semantic model that later declares
  its dialect is checked against that dialect's declarations.
- Bad, because a release that no suite runs, such as DataFusion 54 or 57,
  reads declarations that no run checked. The gain is that a new release of
  an engine plans with no change to semstrait.
- Neutral: every `Function` holds a `releases` field, which is empty except
  where a release differs (rule 3).
- Neutral: `Function::on()` returns `Option`, so every reader handles `None`,
  which DataFusion gives for `regexp_extract`, `width_bucket` and `date_diff`
  (ADR 0087, rule 5).
- Neutral: every `FunctionCall`, `Aggregate` and `Window` site, and every site
  that names `AggregationOp`, `AggregateKind` or `WindowFn`, changes.
- `CONTEXT.md` changes:
  - New entries: **Dialect**, **Canonical declaration**, **Dialect
    declaration**, **Function type**, **Call clause** and **Function option**.
  - **Dialect declaration** says when the dialect has no such function, and it
    is written for a difference in the clauses that a call may write. A
    DataFusion declaration gives the canonical result type, because the plan
    holds Substrait's types.
  - **Engine adapter**: its _Avoid_ "dialect" names these rules, since a
    dialect is the engine's declared vocabulary, which the rules read.
- ADR 0068 names `CanonicalFn` and `AggregateKind` in *The seam*, and ADRs
  0037, 0059, 0065, 0066, 0071, 0074, 0076, 0077 and 0083 name the
  `FunctionCall`, `Aggregate` or `Window` node (`Expr::Window`, "`Aggregate`
  node"). Each stands for rule 1's `Function` node with its function: an
  `Aggregate` node is a call of an aggregate function, and `Expr::Window` is a
  call of a window function, whose order keys are its `Over`'s. Those ADRs are
  not amended for it.
- The expression vocabulary grows only by an ADR (`CONTEXT.md` **Canonical
  vocabulary**). This is that ADR for `Function` as a node.

## More Information

Decided on the question of a function's canonical declaration and its dialect
declarations. The mark for a
function that a dialect does not have and a dialect declaration's clauses were
decided on the question of the semstrait-sql catalog package, its registry and the
functions it declares. A
review ruled the
aggregate function's additivity, and a further review a call per function
type, its check against the declaration, and a DataFusion declaration's result
type. `Dialect`'s `Substrait` variant was decided on the question of the SQL types and
names the canonical data type roster gains,
and a later review ruled
that Substrait declares no function for now. The human's rulings on a further review,
which a record keeps, add a window
call's `Over` to the check of a call, and give a default to an optional parameter
that a later parameter follows. The human's rulings of 2026-10-17 give a default
to a parameter that is present only when an earlier argument says so. A dialect's
release, and a function's `releases` entries, were decided in a later decision.
