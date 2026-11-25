# semstrait-adapter-datafusion

The DataFusion engine adapter (`CONTEXT.md` **Engine adapter**; ADR 0068, *The
DataFusion rules*): `DataFusionRules`, the `SubstraitRules` that a
`SubstraitProducer` is built with for one DataFusion release's Substrait
consumer (the reference release, 55, by default; the oldest it supports is
53). Each rule rewrites where that release's DataFusion differs; every other
hook keeps `semstrait-adapter`'s identity or faithful default (ADR 0068, *The
seam*). Its library links no engine: DataFusion is a dev-dependency of its
tests only (ADR 0068, *Decision Outcome*).

| Rule | Hook | What it writes | ADR |
|---|---|---|---|
| Post-join filter | `rewrite_node` | a `JoinNode`'s `post_join_filter`, which the consumer refuses, as a `FilterNode` above the join; the join keeps its anchor, hint and trace, and the filter takes its emit form | ADR 0068, *The DataFusion rules*; ADR 0075 |
| Union branches | `rewrite_node` | each later branch of a `MultisetNode` that is no `ProjectNode` and has no emit form, under an identity `ProjectNode`, so that DataFusion names it as the union | ADR 0068, *The DataFusion rules*, **Union branches** |
| Tables | `reader` | a table with no catalog as `[namespace, name]`, in the session's default catalog | ADR 0068, *The DataFusion rules* |
| Files | `reader` | a files read of *n* paths as *n* one-file reads under one `UNION_ALL`; a folder read as its one `uri_folder` item | ADR 0068, *The DataFusion rules*; the folder read, ADR 0073's amendment of ADR 0068 |
| `concat` | `rewrite_expression` | a `concat` with no null handling as a `CASE` that is NULL where an argument is, over the `concat` that skips a NULL | ADR 0087, rules 11 and 23 |
| `regexp_extract` | `rewrite_expression` | rule 23's `CASE` over `regexp_like`, `regexp_instr` and `regexp_replace`, as DataFusion has no `regexp_extract` | ADR 0087, rule 23 |
| `regexp_like`, `regexp_instr` | `whole_call` | DataFusion's functions, on the canonical arguments | ADR 0087, rule 24 |
| `regexp_replace` | `whole_call` | DataFusion's function, with the flags `g` | ADR 0087, rule 23 |
| `date_trunc` | `whole_call` | DataFusion's `date_trunc`, on a timestamp moved to the call's timezone (its declared default where the call names none, and NULL on a NULL one) and back to its clock value, with a Sunday week start one day forward and back | ADR 0072, *Where each layer writes it*; ADR 0087, rule 14 |
| `extract` | `whole_call` | DataFusion's `date_part`, the part's name as a string, and its `Int32` cast to `bigint`; on DataFusion 53, `day_of_week` has no part name, so it is written as `add(modulus(add(date_part('dow', v), 6), 7), 1)` | ADR 0087, rules 14 and 23 |

A whole call is the engine's own functions around the arguments that the
producer has lowered, which the producer writes as returned (ADR 0072's
amendment of ADR 0068). Its functions are written under the URN
`extension:semstrait:datafusion`, of which the consumer reads the
name only.

| Module | Holds |
|---|---|
| crate root | `DataFusionRules`, its hooks, and the post-join filter, union-branch, table and files rules |
| `rewrite` | the expression rewrites, each in the canonical vocabulary and of the offered expression's type: `concat` and `regexp_extract` |
| `whole_call` | the whole calls, in DataFusion's own functions around the arguments that the producer has lowered |

## Tests

The test targets:

- `oracle`: each case runs through `Planner::plan`, or `by <fn>` in its place for a
  hand-built plan or a planner with a replaced Strategy. Its model compiles clean
  against its declaration; a rows case's plan validates, produces twice to
  byte-equal protobuf under `DataFusionRules`, passes the wire checks, and runs on
  DataFusion 55 to its rows. A `cases!` line in `oracle.rs` lists every case.
- `vocab`: each case's plan is hand-built, against the one model
  `vocab/models/reads/`, and runs the same way; each lives in its own test
  function, not a `cases!` line.
- `joins`, `windows`: each case's plan is hand-built and run by `gate`, against the
  target's own `data.yaml` with no model; `windows` holds the `rank` and
  `row_number` marker plans (ADR 0076).
- Each corpus model in the repository's `test_data/`, `<model>.yaml`, compiles
  clean against its own declaration, `<model>.data.yaml`.

### The fixture tree

```
tests/fixtures/
  oracle/
    models/
      commerce/  marketing/  finance/  events/     model.yaml  data.yaml
      <domain>_<difference>/                       model.yaml  data.yaml
    <subject>/
      <problem>.yaml
      <local model>/                                model.yaml  [data.yaml]
  vocab/
    models/reads/                                   model.yaml  data.yaml
    <subject>/<problem>.yaml
  joins/
    data.yaml
    <problem>.yaml
  windows/
    data.yaml
    <problem>.yaml
```

`oracle/models/` holds the four domain models and, beside them, the variants. A
variant `<domain>_<difference>/` holds the folders whose declaration cannot
share their domain model, and its name says how it differs: a table with
other columns or column types, a semantic of the same name with another
definition or kind, an inline semantic that would shadow a root one, a source
of another shape, or a folder that must hold no object. Every other case
reads a domain model directly, with `input_data:` restoring its own rows
where the domain's rows differ. A refusal model, or one fitting no domain
(`harness`, `resolution`'s candidates, `read`'s `partitioned`), sits beside its own
problem file instead; `joins` and `windows` hold no model, only the `data.yaml`
their hand-built plans read.

### The subjects

| Folder | Tests |
|---|---|
| `request` | The request's own mechanics over one Dataset: selection, filters, `where`, order, limit and offset. |
| `resolution` | Field-first resolution picks a request's scope root among public candidates by compositions, granularity fit, exposed semantics and name order. |
| `read` | A files or Uris read of one, two or no object, a partition folder's key=value columns, and a column's read type from its catalog, its declared semantics, or the widest of several. |
| `temporal` | A time axis buckets by granularity and week start, and a model-authored `date_trunc` or `extract` Dimension keeps its own clock. |
| `function` | A canonical function's behaviour on a NULL operand, a cast, or a pattern match. |
| `additivity` | A semi-additive Measure's strategy keeps the latest or earliest rows by axes and `by`, a Metric reads it at its marker, and a non-additive Measure re-aggregates already-aggregated rows at a coarser grain. |
| `formula` | A Metric's formula computes its CASE parts or runs over final aggregates, and pushes a sum down past a join, a Unionset's branches, or a Grainset operand. |
| `joinset` | A Joinset's operand delivers rows or partial aggregates below the join, places a request predicate or named filter above or below it, coalesces a name both operands map, reads a fan-out marker once per preserved row identity, or leaves out an operand the request does not read. |
| `unionset` | A Unionset branch delivers rows or partial aggregates combined once above the union, places its own or a branch's filter, leaves out an excluded branch or fills a name it does not map, becomes implicit where a Dataset's storage reduces to several physical sources, and a unique union's distinct compares its output, keyset included. |
| `grainset` | A Grainset routes a request to the child that serves it, drilling across several when none does alone. |
| `composition` | How a composed Unionset or Grainset constituent delivers rows or partial aggregates, folds a literal through its constituents, or reaches a marker or grain through them to re-aggregate, and how a built join's own field, a relationship filter, or a declared keyset extends a unique union's distinct. |
| `refusal` | A parse-stage refusal from a Measure with no `agg:`, and a compile-stage refusal from a Dataset's mapping naming a column its table lacks. |
| `harness` | The harness model's table and files reads, hand-built and planned, and a plan-stage refusal. |

`vocab` reuses `function`, `read`, `temporal` and `unionset` at its own, narrower
scope. `joins` and `windows` hold no subject folders: each problem file sits at the
target root.

### A problem file

```yaml
# <what this file tests, one line>
<case_key>:
  # <what this case checks, one line; may cite an ADR>
  model: models/commerce          # optional; a folder under the target root
  request:                        # from, select, where, filters, order_by, limit, offset
    from: orders
    select: [{dimension: date, granularity: month}, cost, revenue]
  context: {timezone: Europe/Stockholm, week_start: sunday}   # optional
  input_data:                     # optional; key: `namespace.table` or an object's URI
    warehouse.orders: [[A, 2026-03-10, 10], [A, 2026-03-20, 5]]
  types: [date, bigint, bigint]   # optional
  expect:
    rows: [[date, cost, revenue], [2026-03-01, 15, 150]]   # or error: {stage, variant, fields}
  plan: |                         # every rows case
    …
  message: |-                     # every error case
    …
  sql: >-                         # optional, never run
    SELECT …
```

`deny_unknown_fields` holds at every level. A case id is the problem file's path
under the target root, without `.yaml`, then `/`, then the case key; the runner
splits it at the last `/`. `model:` names a folder holding `model.yaml` and
`data.yaml` under the target root; it is absent for a hand-built case with no
model, whose data is the target root's own `data.yaml`. `input_data:` replaces the
rows of the tables or objects it names, for that case only, before the declaration
builds its store. `plan` holds every rows case's production, explained; `message`
holds every error case's refusal; both compare byte for byte. A problem file
carries one header line, which says what the file tests, a case one
comment line that may cite an ADR, and `model.yaml` one header line;
`data.yaml` carries none.

### Names

A folder, file and case name describes its behaviour, in snake_case.
A test name is the case id with each `/` replaced by `_`.

| Abbreviation | Meaning |
|---|---|
| `agg` | a Measure's aggregation, the model key `agg:` |
| `asc`, `desc` | sort directions |
| `avg` | the average aggregation |
| `dt` | a partition key named `dt` |
| `eq` | the equality function |
| `equi` | an equality join condition, as in *non-equi* |
| `expr` | the model key `expr:` |
| `utc` | the UTC time zone, a case's default |
| `eu` | a region value in the data |
| `e2e` | the corpus model `test_data/e2e_full_coverage.yaml` |

`concat`, `regexp_extract`, `date_trunc`, `bigint` and `timestamptz` keep the
spelling of the SQL function or type they name. Grainset, Joinset, Unionset and
keyset are glossary words (`CONTEXT.md`), not abbreviations.

### The harness

`tests/common/` holds:

| Module | Holds |
|---|---|
| `mod.rs` | the builders of a hand-built plan; `Rows`; and the wire checks (ADR 0068, *The node-common block on the wire, and trace*) |
| `case` | a problem file's cases, read by id: `request`, `context`, `input_data`, the optional output `types`, `expect`, and the optional `sql`, `plan` and `message`, built through the planner's public constructors |
| `declaration` | a model's data declaration: each table and files object that it reads, with its columns and rows, the catalog and the store that it builds, `apply_input_data`, and the check of the types that a manifest gives its columns |
| `data` | the data that a declaration gives, written at test time: Parquet files in one in-memory object store under `s3://semstrait-e2e/`, and tables as `MemTable`s |
| `consumer` | `FilesConsumer`: DataFusion's stock consumer, but a files read is read from the session's object store (ADR 0068, *The DataFusion rules*, **The consumer that runs a files read is the caller's**) |
| `refusal` | an error case's refusal: its stage, its variant and the fields that the case names |
| `runner` | the runners: `run`; `hand_built`, `run` of a case whose plan is hand-built, which also produces it under the faithful rules and gives its production; `gate`, which runs a hand-built plan from `validate` through the producer, under the faithful rules and under `DataFusionRules`, each twice to byte-equal protobuf, through the wire checks, on DataFusion 55 to literal rows, with its `plan:`, in the session of its target's `data.yaml`; `validated`, a rows case's plan, validated, and its production under `DataFusionRules`; `planned`, `compiled`, `corpus`, `production`, a rows case's production for a check on the protobuf; and the listing check |
| `golden` | `Golden`, a case's rendered `plan` or `message` and its comparison with the committed field; and the crate's one generator, which splices the rendered field into its problem file |

### Running and adding a case

`runner::run` takes a case through each stage it expects; an earlier stage runs
only as setup. An error case compares one refusal by its variant and fields, and
gives its `message`: a parse- or compile-stage case compares its model's refusal.
Otherwise the model compiles clean, two plannings of the request are equal (ADR
0024; ADR 0063), and a plan-stage error case compares the plan's refusal. A rows
case's plan validates, produces twice to byte-equal protobuf under
`DataFusionRules`, passes the wire checks, and runs to the `select` names, the
case's `types` where given, and its rows; it gives its `plan`, the explain
text of its production. `hand_built` and `gate` wrap these stages for a
hand-built, rows-only case. A listing check, in every target, fails a case
key that no test lists and a listed case id with no case.

To add a case:

1. Write it in its problem file, or a new one under its subject with a header
   line: the case key, a comment, `model:` where needed, its `request`, a
   `context` where not UTC and Monday (ADR 0063; ADR 0072), and `expect`, either
   `rows` or `error`.
2. Add its line to `cases!` (`oracle`) or write its test function (`vocab`,
   `joins`, `windows`).
3. Write its `plan` or `message` with the generator:
   `SEMSTRAIT_GOLDENS=1 cargo test -p semstrait-adapter-datafusion --test <target>
   generate_goldens`.
   It writes only when asked by its name with its switch set, and then
   splices the field into the problem file and leaves every other line as
   it is; every other run compares.

## Dependencies

It depends on `semstrait-adapter` (`SubstraitRules`, `Extensions`,
`InvalidRewrite`, `FunctionName`), `semstrait-common` (`Dialect`, which picks
a function's DataFusion declaration), `semstrait-ir` (the plan nodes,
`PhysicalExpr`, `Parameter` and `Schema`), `semstrait-sql` (the calls, the
function registry, its options and argument forms) and `substrait`, and
imports each type from the crate that owns it (ADR 0090, rule 6).

Dev-only: `datafusion` and `datafusion-substrait`, the engine and its consumer;
`tokio`, the async tests' runtime; `async-trait`, for the catalog, the store
and the consumer; `futures-util` and `bytes`, for the store's listing stream
and the bytes of `FileSystem`; `prost`, for the byte-equal check; `semstrait-catalog`, for `FileFormat` on a files read
(ADR 0067); `semstrait-common` with `serde`, for a data declaration's dialect;
and `semstrait-model`, `semstrait-manifest`, `semstrait-planner`, `serde` and
`serde_yaml`, for the runners: the oracle and `hand_built` compile a model
against its data declaration and plan a case, and `gate` reads its target's
data declaration.
