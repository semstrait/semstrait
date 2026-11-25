---
number: 87
title: The function catalog declares the common SQL functions that DataFusion can run
status: accepted
date: 2026-10-09
source:
- CONTEXT.md
- docs/adr/0008-unsealed-extension-traits.md
- docs/adr/0020-sealed-process-global-registries.md
- docs/adr/0021-portability-floor-no-implicit-coercion.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md
- docs/adr/0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md
- docs/adr/0092-an-operation-node-has-a-declaration-and-checks-its-operands-by-type-family-and-a-type-system-derives-the-result-types.md
- '`crates/semstrait-sql/src/functions/catalog/` and `crates/semstrait-sql/src/functions/registry.rs`'
- the research on DataFusion 55's function declarations
- '`datafusion-functions@55.0.0`, `datafusion-functions-aggregate@55.0.0`, `datafusion-functions-window@55.0.0`, `datafusion-functions-nested@55.0.0` and `datafusion-substrait@55.0.0`: `src/logical_plan/consumer/expr/scalar_function.rs:28-147`, `consumer/expr/aggregate_function.rs`, `consumer/expr/window_function.rs:114-116`; `substrait@0.63.0`: `substrait/extensions/functions_*.yaml`'
- ISO/IEC 9075-2 feature list in the PostgreSQL 18 documentation, `features-sql-standard.html` (T054, T055, T056, T611, T612, T614, T615, T617, T622, T623, T624, T626); https://modern-sql.com/caniuse/greatest-least
- '`calcite@calcite-1.42.0`: `core/src/main/java/org/apache/calcite/sql/util/SqlOperatorTables.java` (`IndexedSqlOperatorTable`); `datafusion-expr@55.0.0:src/registry.rs:55`'
- the rulings of 2026-10-09, and the human's rulings of 2026-10-15, and the human's rulings of 2026-10-16 in the grilling, and the human's rulings of 2026-10-16 and 2026-10-17
- the human's rulings of 2026-10-17
- 'the decision (a dialect is read at a release); `datafusion-functions@53.1.0:src/datetime/date_part.rs:231-232`, and `src/datetime/date_part.rs:242-244` at `datafusion-functions` 54.0.0'
links:
- target: 8
  kind: amends
- target: 20
  kind: amends
- target: 21
  kind: amends
- target: 66
  kind: amends
- target: 72
  kind: amends
---

# The function catalog declares the common SQL functions that DataFusion can run

## Context and Problem Statement

The functions move to
`semstrait-sql`. ADR 0091 gives a function its form: a canonical declaration, then a declaration
of the same struct for each dialect where the engine differs (`Function<F> { canonical, dialects,
releases }`, rule 3). Every call is one `Function(Call)` node, which holds a handle to its
function's `static` (rule 1). Four things are open: which functions `semstrait-sql` declares,
which of them have a DataFusion declaration, how a function is added, and which functions take
`Promote` (ADR 0092, rule 3).

Today the functions sit in two lists that disagree:

- **The registry** (the former `semstrait-ir` function registry) declares 47 functions. YAML cannot write 15 of
  them: `date_part`, `year` … `second`, and eight aggregates.
- **The YAML function table** (`FUNCTION_TABLE`) names 46 functions. The registry lacks 14 of them,
  `regexp_match`, `regexp_extract`, `concat_ws`, `left`, `date_trunc` and `extract` among them, so
  compile refuses each as unknown.
- **The corpus** writes 6 function tags: `upper` (12), `regexp_match` (9), `regexp_extract` (9),
  `concat` (4), `lower` (3) and `length` (1). Its `regexp_match` is a Boolean match
  (`test_data/acme_eu_ad_platform_v3.yaml:261`).

ADR 0021 admits a function "only if DataFusion ∩ DuckDB ∩ Spark support it natively or by trivial
remap", and calls everything else "adapter-extended" (*Decision Outcome*). ADR 0020 seals the
registry "by folding in every linked `RegistryExtension`" (*Decision Outcome*), and that fold does
not exist. The brief asks for "every standard scalar function, all `regexp_*`,
math and logical ones among them, and every standard aggregate and window function", and leaves
out most functions on a list, a map or a struct.

DataFusion 55, the one dialect that declares functions:

- finds a function by its name or an alias, with an exact match, and never reads a Substrait
  function's options or `output_type`;
- has no `regexp_extract`, `date_diff` or `width_bucket`;
- names some functions in its own words (`signum`, `date_part`, `strpos`, `approx_distinct`);
- skips NULLs in `concat`, `greatest` and `least`.

## Considered Options

For the rule that puts a function in the catalog:

- **A function that a model can write today, that ISO SQL defines, or that DataFusion, DuckDB and
  Spark all have with one meaning, where DataFusion can run it** (chosen).
- **ISO SQL only.** `replace`, `median`, `left` and `split_part` fall out, though a model writes
  most of them today.
- **ADR 0021's floor only.** `regexp_extract` falls out, though the corpus uses it 9 times, and
  `covar_samp` stays only because every engine happens to have it.

For a function that DataFusion does not have:

- **Declare it, and mark DataFusion's entry `None`, so that the DataFusion rules write it** (chosen;
  ADR 0091, rule 3).
- **Leave out each function that DataFusion does not have.** The corpus's `regexp_extract` must
  then be written as an array function over `regexp_match`.

For the package and the registry:

- **A `scalar/` module with one file per category, and a sealed name index** (chosen).
- **One `scalar.rs`**, about 900 lines.
- **A map that a macro or a build script builds before compile.** A fault then fails the build,
  but this option needs a build step, and ADR 0020 changes.

For the NULL behaviour where ISO and the engines differ:

- **ISO's** (chosen): `concat` and `greatest` give NULL when an argument is NULL.
- **The engines'**: DataFusion, DuckDB and Postgres skip a NULL in `concat`, and every engine
  skips a NULL in `greatest`. Postgres calls its `greatest` "a deviation from the SQL standard".

For the functions inside a DataFusion rewrite:

- **Declare a function where a rewrite writes it in the canonical vocabulary** (chosen).
- **Declare `to_local_time` and `arrow_cast` too.** Their types (a timestamp with a zone, an Arrow
  type name) do not exist in semstrait, so they cannot have a canonical declaration.

For a `regexp_extract` group that takes no part in the match:

- **NULL, as when nothing matches** (chosen).
- **`''`, as Spark and DuckDB give.** "Nothing captured" then has two values, NULL and `''`.

For `regexp_instr`'s arguments:

- **DataFusion's optional `start`, `occurrence`, `flags` and `group`, each with DataFusion's
  default, so a call that writes none of them behaves as today's two-argument call** (chosen).
  The human: "(b) with defaults to preserve (a) behavior".
- **No flags argument.** DataFusion's declaration then puts its `flags` before the `group`.

## Decision Outcome

In short:

- A function is in the function catalog when the registry or the YAML function table has it today,
  ISO SQL defines it, or DataFusion, DuckDB and Spark all have it with one meaning; no other
  catalog form gives the same result; and DataFusion runs it, natively or by a rewrite.
- The catalog declares 78 scalar, 28 aggregate and 11 window functions.
- DataFusion marks the three functions that it does not have with `None`.
- The catalog is `functions/catalog/`, with one file per scalar category, and a sealed registry
  finds a function by its name or an alias.
- A function enters by a PR under the rule, with no ADR, and there are no extensions.
- The canonical behaviour is ISO's where ISO has one: `concat` and `greatest` give NULL for a NULL
  argument.
- DataFusion has a declaration for 17 functions, and none for three.

### What puts a function in the catalog

1. A function is in the **function catalog** when one of these holds, and no other catalog form
   gives the same result:

   - the registry or the YAML function table has it today;
   - ISO SQL defines it: SQL:2003, and the later editions' regular-expression functions (2008),
     `lag`, `lead` and `nth_value` (2011), trigonometric and logarithm functions (2016, features
     T622-T624), and `greatest`, `least`, `lpad`, `rpad` and `any_value` (2023, T054, T055, T626);
   - DataFusion, DuckDB and Spark all have it with one meaning.

   It enters only where DataFusion has it, or where the DataFusion rules can write it from functions
   that DataFusion has, with no loss. A function on an array enters only where it reads a catalog
   function's array, as `array_element` reads `regexp_match`'s, or where a DataFusion rewrite needs
   it and the function that the rewrite writes is used. The human: "what we bypassing - is majority
   of list for complex types fucntion ( array/map/struct orinterd ) and adding them up on need", and
   "array like ( case with adapter ) only if we observe a need for some function which can be used
   for adapter re-write rules and we have sufficied use cases". On today's functions: "currently
   dead code can serve some reference, so we don't loose functionality, but designing properly what
   is dead right now". On the roster: "all regexp_* standard scalar functions
   and all scalars , aggregation, window, math, logical must be in place".

2. Its canonical shape is ISO's where ISO defines one, else the shape that the engines share, else
   DataFusion's shape with a code comment that a later dialect may unify it.
   The regular-expression functions take the arguments and the result types that the engines
   share, not ISO's `LIKE_REGEX` and `SUBSTRING_REGEX`, which are XQuery forms that no engine has.
   Rule 12 gives their behaviour where the engines differ.

   | Function | ISO | DataFusion 55 | DuckDB | Spark | Today | In the catalog |
   |---|---|---|---|---|---|---|
   | `upper` | 2003 | yes | yes | yes | registry and YAML | yes |
   | `replace` | no | yes | yes | yes | registry and YAML | yes |
   | `covar_samp` | 2003 | yes | yes | yes | — | yes |
   | `regexp_extract` | `SUBSTRING_REGEX` | no | yes | yes | YAML; 9 corpus uses | yes: the rules write it from `regexp_like`, `regexp_replace` and `regexp_instr` |
   | `width_bucket` | 2003 | no | yes | yes | — | yes: the rules write it with CASE and arithmetic |
   | `percentile_disc` | 2003 | no | yes | yes | — | waits: the rules would need `array_agg`, an array function with no use |
   | `gcd` | no | yes | yes | no | — | no |
   | `mod` | 2003 | the operator `%` | yes | yes | the operation `mod` | no: the operation `mod` gives the same result |

3. A function enters by a PR that adds its declarations under rule 1. The PR names the part of
   rule 1 that admits it, for a function on an array the function whose array it reads or the
   rewrite that needs it, and the DataFusion
   rewrite where DataFusion has no such function. It needs no ADR. A new node variant, or a function
   that rule 1 does not admit, still needs one.

4. No extensions: `RegistryExtension` retires, as `AggregateKind::Extension` does (ADR 0085, rule
   3). A function that semstrait needs goes into the catalog. The human: "I don't like to have
   extensions , we don't need them … if something needs to be added - it goes into catalog or
   registry declarations. at least of rnow.".

### DataFusion's entries

5. DataFusion's entry is `None` for `regexp_extract`, `width_bucket` and `date_diff`, the three
   catalog functions that it does not have (ADR 0091, rule 3). The DataFusion rules write each one
   from DataFusion's own functions (rule 23).

6. A function has a DataFusion declaration only where DataFusion's function differs from the
   canonical one, by ADR 0091 rule 5, and that declaration gives the canonical result type.

   - DataFusion finds `length`, `substring`, `trim`, `power`, `stddev_samp` and `var_samp` by an
     alias, and `array_element` by its own name, so none of them has a DataFusion declaration.
   - DataFusion's `signum` and `date_part` give the result types of the canonical `sign` and
     `extract`.

### The package and the registry

7. `semstrait-sql`'s `functions` package holds the catalog, one file per function type, and one
   file per category for the scalar functions:

   ```text
   functions/
   ├── mod.rs              // Function, Declaration, Scalar, Aggregate, Window, … (ADR 0091, rules 3 and 4)
   ├── handle.rs           // FunctionRef, Aggregation, AggregateAllowed, NoAggregate (ADR 0091, rule 1)
   ├── registry.rs
   ├── operation/          // the operations and their declarations (ADR 0092, rule 1)
   └── catalog/
       ├── mod.rs          // the lists of the scalar, aggregate and window functions
       ├── scalar/
       │   ├── string.rs
       │   ├── pattern.rs
       │   ├── math.rs
       │   ├── temporal.rs
       │   ├── conditional.rs
       │   └── array.rs
       ├── aggregate.rs
       └── window.rs
   ```

8. The registry finds a function by its name or an alias:

   ```rust
   pub fn registry() -> &'static Registry; // sealed once, at first access (ADR 0020)

   impl Registry {
       pub fn get(&self, name: &str) -> Option<FunctionRef<AggregateAllowed>>;
       pub fn all(&self) -> impl Iterator<Item = FunctionRef<AggregateAllowed>>;
   }
   ```

   - `get` gives ADR 0091 rule 1's `FunctionRef`. A row-level tree narrows it as
     `AggregateInRowExpr`'s `TryFrom` does, so no second handle type exists.
   - `get` takes the name as text and folds its ASCII case, as Calcite's operator table looks a
     name up in an upper-case index (`IndexedSqlOperatorTable.lookUpOperators`), and DataFusion's
     registry takes the plain name (`udf(&self, name: &str)`). A name has no type of its own:
     every name and alias in the catalog is a lower-case identifier, so a check of a name's
     spelling would refuse no name that `get` finds. A canonical declaration is semstrait's
     default, and a function holds it beside its dialect declarations in one struct (ADR 0091,
     rule 3), as a data type does (ADR 0086, rule 1). The human: canonical "is declration, say
     default behavior we declare for semstrait, which can be overriden or spelled differently in
     other dialect", and "structure wise - it is suppose to use the same entry point as others".
   - Beside `get`, the registry finds a function by its name in `canonical` or in one dialect. In
     `canonical`, a name matches only a canonical declaration's `name`: an alias is another YAML
     spelling (ADR 0091, rule 4), so it matches nothing. In a dialect, a name matches that
     dialect's declaration's `name`, or the canonical `name` where the dialect has no declaration of
     its own, and it matches nothing where the dialect's entry is `None` (ADR 0091, rule 3).
   - Parse finds a function with `get`, by its name or an alias (ADR 0089, rule 4). Reading a
     manifest finds it by its name in the manifest's dialect (ADR 0088, rule 9).
   - Compile, the planner and the adapter read a call's handle with no lookup (ADR 0091, rule 1).

9. The seal refuses these faults, each a typed `SealFault` (ADR 0004), with a panic at first access
   (ADR 0020).

   | Fault | Example |
   |---|---|
   | Two functions share a name or an alias. A name or an alias equals an operation's name or alias (ADR 0092, rule 1) | a function `mod` beside the operation `mod` |
   | A declaration is malformed: its category is not its file's; an option value is outside its option; a canonical value is not among the values; an optional parameter comes before a required one; an optional parameter that a later parameter follows declares no default, or a declared default is not a literal of its parameter's type (ADR 0091, rule 4); a condition names a later argument; an argument index is out of range; a type variable `T` is not bound; a scalar has an aggregate's nullability | `round` with `Category::String` |
   | A dialect declaration is malformed: two `dialects` entries for one dialect; aliases on it (ADR 0091, rule 4); equal to the canonical declaration in every part; two `releases` entries of one dialect whose releases overlap; a `releases` entry whose releases hold the dialect's reference release (ADR 0091, rule 6) | a DataFusion declaration of `upper` that repeats the canonical one; a DataFusion `releases` entry for 53 to 55 |

   The seal checks each declaration's structure. A prototype's sweep over sample calls checks
   rewrites, which are the DataFusion rules'.

   The tags of the tree (`lit`, `col`, `case`, `cast`, …) are `semstrait-model`'s YAML grammar,
   which `semstrait-sql` does not read (ADR 0089, rule 6), so the seal does not check them.
   `semstrait-model`'s parse reads its own tags before it looks a function up, and a
   `semstrait-model` test checks that no function's name or alias equals a tag of the tree. The
   human: "model uses sql, and check against when parsing and applying typing". Calcite has no
   such list: CASE and CAST are operators, and its parser owns its keywords.

10. The catalog and the registry replace the former `semstrait-ir` function registry: `FunctionRegistry`,
    `function_registry()`, `FunctionSpec` and its types, the former built-in declarations, the former extension module and
    `RESERVED_AST_TAGS`, whose tags the `semstrait-model` test checks (rule 9). ADR 0092 rule 1
    removes the YAML tag tables, so the catalog is the one list.

### The roster

Each row is one canonical declaration: its parameters, then its result. `x…` means x, one time or
more. **Mirror** is NULL when an argument is NULL. An alias is an ISO name of the same function,
or a YAML spelling of today; an engine's own name goes into its dialect declaration (ADR 0091,
rule 4).

11. **String** (`string.rs`):

    | Function | Shape | NULL | Aliases |
    |---|---|---|---|
    | `upper`, `lower`, `reverse`, `initcap` | `(string) → string` | Mirror | |
    | `length` | `(string) → integer`, in characters | Mirror | `char_length`, `character_length` |
    | `octet_length`, `bit_length`, `ascii` | `(string) → integer` | Mirror | |
    | `chr` | `(integer) → string` | Mirror | |
    | `substring` | `(value: string, start: integer, length: integer, optional) → string`; 1-based; a negative length is an error | Mirror | |
    | `overlay` | `(value: string, placing: string, start: integer, length: integer, optional) → string` | Mirror | |
    | `position` | `(search: string, value: string) → integer`, ISO's order; 0 when not found | Mirror | |
    | `trim`, `ltrim`, `rtrim` | `(value: string, characters: string, optional) → string`; the default is the space | Mirror | `trim`: `btrim` |
    | `concat` | `(string, string…) → string`; option `null_handling`, canonical `ACCEPT_NULLS` (rule 18) | NULL when an argument is NULL | |
    | `concat_ws` | `(separator: string, string…) → string` | NULL only when the separator is NULL; a NULL value is skipped | |
    | `replace`, `translate` | `(value: string, from: string, to: string) → string` | Mirror | |
    | `lpad`, `rpad` | `(value: string, length: integer, fill: string, optional) → string`; the default fill is the space | Mirror | |
    | `repeat`, `left`, `right` | `(value: string, count: integer) → string` | Mirror | |
    | `split_part` | `(value: string, delimiter: string, part: integer) → string`; 1-based; a negative part counts from the end | Mirror | |
    | `starts_with`, `ends_with`, `contains` | `(value: string, search: string) → boolean` | Mirror | |
    | `levenshtein` | `(string, string) → integer` | Mirror | |
    | `md5` | `(string) → string`, as hex text | Mirror | |

12. **Pattern** (`pattern.rs`). No function takes a case option, and only `regexp_instr` takes a
    flag argument, whose default is no flag: a pattern's
    inline `(?i)` works on DataFusion, DuckDB and Spark.

    | Function | Shape | NULL |
    |---|---|---|
    | `regexp_like` | `(value: string, pattern: string) → boolean`; a match anywhere in the value | Mirror |
    | `regexp_match` | DataFusion's shape: `(value: string, pattern: string) → array<string>`, the capture groups; a code comment says that a later dialect may unify it | NULL when nothing matches |
    | `regexp_extract` | `(value: string, pattern: string, group: constant integer) → string`; group 0 is the whole match | NULL when nothing matches, or when the group takes no part in the match |
    | `regexp_replace` | `(value: string, pattern: string, replacement: string) → string`; every match; the replacement names a group as `\1` | Mirror |
    | `regexp_count` | `(value: string, pattern: string) → bigint` | Mirror |
    | `regexp_instr` | `(value: string, pattern: string, start: integer, optional, occurrence: integer, optional, flags: constant string, optional, group: constant integer, optional) → bigint`; 1-based; the position of the match, or of its group; 0 when nothing matches, or when the group takes no part | Mirror |

    - The corpus's 9 Boolean `regexp_match` tags become `regexp_like`.
    - `regexp_extract` gives NULL when nothing matches, and also when its group takes no part in
      the match. DuckDB and Spark give `''` for both
      (`regexpExpressions.scala:820-839`), a difference of their dialects, later. Parse refuses a
      group below 0, or above the number of groups of a literal pattern, because the check needs no
      type (ADR 0089, rule 1). Where the pattern is not a literal, a group above its number of
      groups gives NULL. Substrait's `regexp_match_substring` leaves it undefined
      (`functions_string.yaml:137-200`).
    - The pattern language is the one that DataFusion, DuckDB and Spark share. The `(?x)` flag is
      outside it: DuckDB refuses it, and DataFusion's `regexp_extract` rewrite (rule 23) cannot
      wrap a pattern that ends in a `(?x)` comment.
    - `regexp_instr` takes DataFusion's arguments, in DataFusion's order (`regexpinstr.rs:43-68`):
      `start`, `occurrence`, `flags`, the pattern's inline flags, and `group`. Its declared
      defaults are `start` 1, `occurrence` 1 and `flags` `''` (no flag), because a later parameter
      follows each (ADR 0091, rule 4). A call that writes no `flags` behaves as one with no flag
      argument. `group` is last and declares no default: a call that writes none gives the
      position of the whole match, as group 0 does. So DataFusion needs no declaration of its own
      for it.
    - `regexp_replace` replaces every match. The engines differ, and with the first match only, no
      call could replace every match, because the declaration has no flag argument.

13. **Math** (`math.rs`). A parameter that names the number family takes any number; DataFusion
    converts it.

    | Function | Shape | NULL | Aliases |
    |---|---|---|---|
    | `abs` | `(T: number) → T`; overflow is an error | Mirror | |
    | `ceil`, `floor` | `(number) → DecimalScaleZero`: a `decimal(p,s)` gives `decimal(min(p−s+1,38), 0)`, any other number its own type (ADR 0092, rule 3) | Mirror | `ceil`: `ceiling` |
    | `round` | `(value: number, scale: constant integer, optional) → RoundOf`: a `decimal(p,s)` gives `decimal(min(p+1,38), s)`, any other number its own type (ADR 0092, rule 3); a tie goes away from zero | Mirror | |
    | `sqrt`, `exp`, `ln`, `log10`, `log2`, `cbrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `degrees`, `radians` | `(number) → double`; `sqrt` of a negative number is an error | Mirror | |
    | `power` | `(number, number) → double` | Mirror | |
    | `log` | `(base: number, value: number) → double`, ISO's order; one argument is refused | Mirror | |
    | `atan2` | `(y: number, x: number) → double` | Mirror | |
    | `sign` | `(number) → integer` | Mirror | |
    | `width_bucket` | `(value: number, low: number, high: number, count: integer) → integer` | Mirror | |
    | `pi` | `() → double` | never NULL | |
    | `random` | `() → double`, Volatile | never NULL | |

14. **Temporal** (`temporal.rs`):

    | Function | Shape | NULL |
    |---|---|---|
    | `extract` | `(component: enum, value: date or timestamp) → bigint`; the ten components, `day_of_week` ISO's (Monday = 1); `hour`, `minute` and `second` refused on a date | Mirror |
    | `date_trunc` | ADR 0072's `(granularity: enum, value: T date or timestamp, timezone, week_start) → T`; `timezone` defaults to `'UTC'`, and `week_start` to `'monday'` (ADR 0091, rule 4) | Mirror |
    | `date_diff` | `(part: enum, start: T date or timestamp, end: T) → bigint`; the count of the part's boundaries between start and end | Mirror |
    | `current_date` | `() → date`, Stable | never NULL |
    | `current_time` | `() → time(9)`, Stable | never NULL |
    | `current_timestamp` | `() → timestamp(9)`, Stable, with no zone (ADR 0022) | never NULL |
    | `make_date` | `(year: integer, month: integer, day: integer) → date` | Mirror |
    | `to_date` | `(string) → date`; ISO 8601 text only, other text is an error | Mirror |
    | `to_timestamp` | `(string) → timestamp(9)`; ISO 8601 text only | Mirror |

15. **Conditional** (`conditional.rs`): `coalesce` and `nullif` as ADR 0092 rule 6 gives them
    (`nullif` with the alias `null_if`), and `greatest` and `least`. `coalesce` is
    `(value, value…)`. `greatest` and `least` are `(value…) → Promote`, every argument in one type
    family, NULL when an argument is NULL (ISO; rule 18). `greatest` and `least` are the only
    functions that take `Promote`; `coalesce` binds one exact type.

    `coalesce` takes two arguments or more, as ISO's `COALESCE` does. A call of one argument is
    that argument, which an engine writes as a rewrite; the canonical declaration does not take it.
    The human: "others are re-writes, when coalesce with 1 argument starts to be just this argument
    return as value". `greatest` and `least` take one argument or more, as ISO's do (T054).

16. **Array** (`array.rs`), a scalar category (rule 21): `array_element(array: array<string>, index: integer) →
    string`, 1-based, NULL when the index is out of range. It has the one type that its use needs:
    it reads the `array<string>` that `regexp_match` returns.

17. **Aggregate** (`aggregate.rs`). The clauses are the `ClauseRule`s that are not `Refused`
    (ADR 0091, rule 4): FILTER is `Allowed` on every aggregate. ORDER BY in the call is `NoEffect`,
    except on `string_agg`, where it is `Allowed`, and on `percentile_cont`, whose order is its
    WITHIN GROUP.

    | Function | Shape | NULL | Clauses | Additivity, DISTINCT form | Aliases |
    |---|---|---|---|---|---|
    | `count` | `(value: any, optional) → bigint` | never NULL; 0 on no rows | DISTINCT | Additive, NonAdditive | |
    | `sum` | `(number) → SumOf`: an integer gives `bigint`, a `real` or a `double` gives `double`, a `decimal(p,s)` gives `decimal(38,s)` (ADR 0092, rule 3) | NULL on no rows | DISTINCT | Additive, NonAdditive | |
    | `avg` | `(number) → AvgOf`: an integer, a `real` or a `double` gives `double`, a `decimal(p,s)` gives `decimal(38, max(s,6))` (ADR 0092, rule 3) | NULL on no rows | DISTINCT | NonAdditive | |
    | `min`, `max` | `(T: any) → T` | NULL on no rows | DISTINCT `NoEffect` | Additive | |
    | `any_value` | `(T: any) → T`; skips NULLs | NULL on no rows | DISTINCT `NoEffect` | Additive | |
    | `bool_and`, `bool_or` | `(boolean) → boolean` | NULL on no rows | DISTINCT `NoEffect` | Additive | `bool_and`: `every`; `bool_or`: `any`, `some` |
    | `stddev_samp`, `stddev_pop`, `var_samp`, `var_pop` | `(number) → double` | NULL on no rows; the sample forms give NULL on one row | DISTINCT | NonAdditive | |
    | `covar_pop`, `regr_avgx`, `regr_avgy`, `regr_sxx`, `regr_syy`, `regr_sxy` | `(y: number, x: number) → double` | NULL on no rows | | NonAdditive | |
    | `covar_samp`, `corr`, `regr_slope`, `regr_intercept`, `regr_r2` | `(y: number, x: number) → double` | NULL on no rows and on one row; all but `covar_samp` also where every `x` is equal, and `corr` also where every `y` is equal | | NonAdditive | |
    | `regr_count` | `(y: number, x: number) → bigint` | never NULL | | Additive | |
    | `percentile_cont` | `(fraction: constant double)` WITHIN GROUP `(ORDER BY value: number) → double` | NULL on no rows | WITHIN GROUP `Required` | NonAdditive | |
    | `median` | `(number) → AvgOf` | NULL on no rows | DISTINCT | NonAdditive | |
    | `string_agg` | `(value: string, separator: constant string) → string`; skips NULLs | NULL on no rows | DISTINCT, ORDER BY | NonAdditive | |
    | `approx_count_distinct` | `(any) → bigint` | never NULL; 0 on no rows | | NonAdditive | |

18. **The NULL choices.** Where ISO and the engines differ, the canonical behaviour is ISO's:

    | Row | `measurement_channel` | `traffic_source` | `concat(measurement_channel, ' - ', traffic_source)` |
    |---|---|---|---|
    | 1 | `Paid Search` | `Google` | `Paid Search - Google` |
    | 2 | `Paid Search` | NULL | NULL; DataFusion's `concat` gives `Paid Search - ` |

    - `concat` gives NULL for a NULL argument, as ISO's `||` does. Its option `null_handling` has
      the canonical value `ACCEPT_NULLS`. An author who wants to skip a NULL writes `concat_ws('',
      …)`, which skips a NULL value on every engine.
    - `greatest` and `least` give NULL for a NULL argument (ISO SQL:2023, feature T054).

19. **Window** (`window.rs`): the eleven ISO window functions. Authoring still writes none (ADR
    0002; ADR 0066, ruling 7), and the planner writes `row_number` and `rank` (ADR 0076).

    | Function | Shape | NULL | Clauses |
    |---|---|---|---|
    | `row_number` | `() → bigint` | never NULL | ORDER BY `Allowed`; frame `NoEffect` |
    | `rank`, `dense_rank` | `() → bigint` | never NULL | ORDER BY `Required`; frame `NoEffect` |
    | `percent_rank`, `cume_dist` | `() → double` | never NULL | ORDER BY `Required`; frame `NoEffect` |
    | `ntile` | `(buckets: constant integer) → bigint` | never NULL | ORDER BY `Required`; frame `NoEffect` |
    | `lag`, `lead` | `(value: T, offset: constant integer, optional, default: T, optional) → T` | can be NULL | ORDER BY `Required`; IGNORE NULLS; frame `NoEffect` |
    | `first_value`, `last_value` | `(value: T) → T` | can be NULL | ORDER BY `Allowed`; IGNORE NULLS; frame `Allowed` |
    | `nth_value` | `(value: T, n: constant integer) → T` | can be NULL | ORDER BY `Allowed`; IGNORE NULLS; frame `Allowed` |

    The `offset` of `lag` and `lead` has the declared default 1, because the parameter `default`
    follows it (ADR 0091, rule 4). The parameter `default` is last and declares no default.

    An aggregate over a window stays out: an aggregate call holds no OVER (ADR 0091, rule 1).

20. **The names that retire**, because another catalog form gives the same result:

    | Today | Instead |
    |---|---|
    | `year`, `month`, `day`, `hour`, `minute`, `second` (registry only) | `extract` |
    | `date_part` (registry only) | `extract`; `date_part` is DataFusion's own name for it (rule 23) |
    | `date_add`, `date_sub` (YAML table; no corpus use) | the operations `+` and `-` with an interval (ADR 0092, rule 5) |
    | `stddev`, `variance` (registry) | the ISO names `stddev_samp` and `var_samp` |

    - `agg:` names a catalog aggregate by its name or an alias, and `count_distinct` stays the
      model's sugar for `count` with DISTINCT (ADR 0089, rule 20).
    - `agg: std_dev` and `agg: variance` stay refused, with no alias: which form they mean, sample or
      population, stays out of scope for now.
    - Not in the catalog: `factorial` (Spark gives NULL above 20!, DataFusion an error), `trunc`
      (Spark has none), `gcd` and `lcm` (Spark has none), `grouping` (no plan writes grouping sets),
      `array_agg` (an array result that no rewrite needs), `first_value`, `last_value` and
      `nth_value` as aggregates (ISO has them as window functions, and one name is one function),
      `bit_and`, `bit_or` and `bit_xor` (ADR 0086 rule 8 has no integer family), and
      `approx_percentile_cont` and `approx_median` (each engine has its own name and result).

21. Array is a category of the scalar functions, beside String, Math, Temporal, Pattern and
    Conditional (ADR 0091, rule 4), for the functions on an array that rule 1 admits.

22. A function is declared where a DataFusion rewrite writes it in the canonical vocabulary, which
    needs canonical types for its arguments and its result: `regexp_like`, `regexp_replace` and
    `regexp_instr`, which the `regexp_extract` rewrite writes (rule 23).
    `to_local_time` and `arrow_cast` work on types that semstrait does not have, so they stay inside
    ADR 0072's hook, which writes the whole call in Substrait.

### DataFusion's declarations

23. DataFusion has a declaration for these functions. Each canonical declaration keeps its own
    shape:

    | Function | DataFusion 55 | DataFusion's declaration |
    |---|---|---|
    | `position` | `strpos(value, search)`; its alias `position` takes the same order | name `strpos`; arguments `(value, search)` |
    | `concat` | skips a NULL | `null_handling` with `IGNORE_NULLS` only (ADR 0091, rule 5) |
    | `regexp_replace` | the first match, unless a fourth argument holds `g` | arguments `(value, pattern, replacement, flags: constant string, optional)` |
    | `sign` | `signum`, `→ double` | name `signum` (ADR 0091, rule 3) |
    | `extract` | `date_part(part: constant string, value) → integer`; refuses an enum argument; its own part names (`isodow`, `doy`) | name `date_part`; `part` an enum of DataFusion's part names, one for each component, written as a string |
    | `date_trunc` | `(granularity: constant string, value)`; no timezone and no week start | `(granularity: constant string, value: T date or timestamp) → T`; ADR 0072's hook writes the rest |
    | `greatest`, `least` | skips a NULL | nullability `AllArgs`: NULL only if every argument is NULL |
    | `stddev_samp`, `stddev_pop` | refuses DISTINCT (`stddev.rs:113-115`, `:217-219`) | DISTINCT `Refused` |
    | `percentile_cont` | `(value, fraction)`; its consumer passes a sort only as ORDER BY | `(value: number, fraction: constant double)`; WITHIN GROUP `Refused` |
    | `approx_count_distinct` | `approx_distinct`, with no alias | name `approx_distinct` |
    | `lag`, `lead`, `first_value`, `last_value`, `nth_value` | its consumer builds every window call with `null_treatment: None` (`window_function.rs:114-116`) | IGNORE NULLS `Refused` |

    DataFusion has no `regexp_extract`, `width_bucket` or `date_diff`, so its entry for each is
    `None`, and its rules write each one from DataFusion's own functions. For `regexp_extract(v, p,
    n)`, they write this, which gives rule 12's result for every group `n` on DataFusion 55:

    ```sql
    CASE WHEN regexp_like(v, p) AND regexp_instr(v, p, 1, 1, '', n) > 0      -- the second test only where n > 0
         THEN regexp_replace(v, '^(?s:.*?)(' || p || ')(?s:.*)$', '\<n + 1>') -- the match's own group n is group n + 1
    END
    ```

    - The lazy prefix `(?s:.*?)` keeps the first match, and `(?s:…)` lets the prefix and the suffix
      cross a newline without changing what `.` means inside `p`.
    - The group around `p` keeps an alternation of `p` and its inline flags inside it, and it makes
      the whole match group 1.
    - The pattern matches the whole value, so `regexp_replace` replaces once, and DataFusion's
      first-match default gives the canonical result.
    - `regexp_instr` gives 0 for a group that takes no part in the match and for a group above the
      pattern's number of groups, so both give NULL.
    - `array_element(regexp_match(v, p), n)` is not this rewrite: `regexp_match` leaves out a group
      that takes no part, so `regexp_extract('bc', '(a)?(b)(c)', 2)` would give `'c'`, not `'b'`, and
      it holds no group 0 when the pattern has a group (`arrow-string@59.3.0:src/regexp.rs:254-262`).

    The table is DataFusion 55's, its reference release (ADR 0091, rule 6). DataFusion 53's
    `isodow` counts Monday as 0 (`datafusion-functions@53.1.0:src/datetime/date_part.rs:232`),
    and from 54.0.0 it counts Monday as 1 (`date_part.rs:242-244` at 54.0.0). So `extract` has a
    `releases` entry for DataFusion 53 to 53: the declaration above with no `day_of_week` part.
    DataFusion's rules write `day_of_week` from `dow` through ADR 0072's whole-call hook, which
    counts Sunday as 0 on both releases:
    `((date_part('dow', v) + 6) % 7) + 1`.

### The Substrait form

24. A call is one `Function(Call)` node in a tree of the plan, and an aggregation node's entry is an
    `AggregateCall` (ADR 0091, rule 1; ADR 0090, rule 2). The Substrait producer
    writes a scalar call as a `ScalarFunction`, an aggregate call as an `AggregateFunction` in
    `AggregateRel.measures`, and a window call as a `WindowFunction` in a Project.

    - The producer holds its own table: each canonical function's Substrait URN and name, or its form
      where Substrait has none (as ADR 0072 gives `date_trunc` one). A function with no row goes under
      semstrait's own URN with its canonical name.
    - The producer writes every option of a Substrait function at the canonical behaviour, because
      Substrait gives an option no default (ADR 0091, rule 4). It casts an argument to a Substrait
      overload's type, where Substrait's type system gives another result type (ADR 0092, rule 3).
    - The DataFusion rules write each call under `on_release(DataFusion, release)`'s name, which
      DataFusion finds.
    - The decision on the semstrait-sql catalog package, its registry and the functions
      it declares lists each function's
      Substrait name, as the producer's input.

### Ruled elsewhere, not here

- Building the Substrait producer's table and the DataFusion rules, with each rewrite, is left to later work.

### Amends

- **0008:** in its *Decision Outcome*, `RegistryExtension` leaves the list of traits open to
  out-of-workspace implementors: a function enters by its declaration in the function catalog (ADR
  0087, rule 3). The rest of 0008 stands.
- **0020:** in its *Decision Outcome*, "`function_registry()` … sealed at first access by folding in
  every linked `RegistryExtension`" becomes: `semstrait-sql`'s `registry()` is sealed at first
  access over the function catalog's declarations only, and its faults are ADR 0087's seal faults
  (rule 9). The rest of 0020 stands.
- **0021:** its decision "intersection function catalog", in its title, is replaced by ADR 0087,
  rules 1 and 4, and the sections that only apply it read as those rules give: in *Context and
  Problem Statement*, "omits ubiquitous SQL functions"; in *Considered Options*, the chosen option's
  "Intersection floor"; in *Decision Outcome*, the first sentence and "Everything else (…) is
  adapter-extended"; and, as ADR 0072 amends it, "the intersection floor stands for every other
  function and argument". ADR 0072's calendar-spec argument is one case of ADR 0087's rule 1: the
  DataFusion rules write it in DataFusion's own functions. The rest of 0021 stands.
- **0066:** in its ruling 2, "grows only by ADR, as before", said of the expression vocabulary,
  becomes: its node variants grow only by ADR, as before, and a function enters by its declaration
  under ADR 0087's rule 1. The rest of 0066 stands.
- **0072:** in *Admission under ADR 0021*, "The floor stands for every other function and argument",
  and in its *Amends* item on ADR 0021, "The intersection floor stands for every other function and
  argument", become: every other function is in the function catalog by ADR 0087's rule 1, which
  admits a function that the DataFusion rules write in DataFusion's own functions. The rest of 0072
  stands.

### Consequences

- Good, because one list, the function catalog, takes the place of the registry and the YAML
  function table, and every function that a model writes is declared.
- Good, because every declared function runs on DataFusion, so the adapter never refuses one (ADR
  0068).
- Good, because a canonical behaviour follows ISO where ISO has one, and the engine's difference is
  a declaration that the adapter reads.
- Good, because a function enters by a PR under one rule, with no ADR and no extension mechanism.
- Good, because `agg: std_dev` and `agg: variance` stay refused with a stated reason, and the
  registry no longer holds two equal specs for each.
- Good, because `regexp_extract` gives the same result on DataFusion as its canonical declaration:
  the group's text, or NULL where nothing is captured.
- Bad, because `concat` and `greatest` give NULL for a NULL argument, where DataFusion skips it
  today: row 2 of rule 18 changes. The gain is ISO's behaviour, which the DataFusion rules write on
  DataFusion.
- Bad, because DataFusion's `regexp_extract` rewrite runs up to three regular-expression passes on
  each row. The gain is an exact result.
- Neutral: the catalog grows from 47 declarations to 117, and every one needs its declaration and
  its tests.
- Neutral: the corpus's 9 Boolean `regexp_match` tags must become `regexp_like`, and `regexp_match`
  now names DataFusion's array function.
- Neutral: DataFusion has no `regexp_extract`, `width_bucket` or `date_diff`, so its rules need
  three rewrites.
- Neutral: `sum` and `avg` on a decimal now give `SumOf` and `AvgOf`, where compile keeps
  `decimal(p,s)` today, and `to_timestamp` and `current_timestamp` give `timestamp(9)`, where the
  registry gives `timestamp(6)`. The cast to a semantics' declared type gives it back (ADR 0089,
  rule 17).
- Neutral: `year` … `second`, `date_part`, `date_add`, `date_sub`, `stddev` and `variance` retire as
  names.
- `CONTEXT.md` changes:
  - **Function catalog** is a new entry.
  - **Canonical function** is a function in the function catalog. The admission clause leaves the
    entry: the rule is ADR 0087's.

## More Information

Decided by a proxy session at the human's
request, and accepted. A review
ruled the `regexp_extract` rewrite and its group that takes no part, an aggregate's OVER,
`regexp_instr`'s arguments and a DataFusion declaration's result type. The human's rulings on
that review retire `CanonicalFn`, move
the check of the tree's tags to `semstrait-model`, and declare the defaults of `regexp_instr`'s
`start`, `occurrence` and `flags` and of the `offset` of `lag` and `lead`. The human's rulings of
2026-10-17 declare the
defaults of `date_trunc`'s `timezone` and `week_start`. DataFusion 53's `extract` and the
seal's faults of a `releases` entry were decided.

The tables per category in `crates/semstrait-sql/src/functions/catalog/` and the seal faults in `crates/semstrait-sql/src/functions/registry.rs` are the base of rules 7 and 9.
