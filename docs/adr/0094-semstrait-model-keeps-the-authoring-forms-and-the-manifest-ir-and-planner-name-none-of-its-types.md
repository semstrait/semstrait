---
number: 94
title: semstrait-model keeps the authoring forms, and the manifest, IR and planner name none of its types
status: proposed
date: 2026-10-30
source:
- CONTEXT.md
- docs/adr/0001-invert-model-ir-dependency-edge.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md
- docs/adr/0090-semstrait-ir-keeps-the-plan-and-its-parameter-and-the-manifest-owns-the-expression-id-and-the-expression-layer.md
- '`crates/semstrait-ir/src/plan/trace.rs:1`; `crates/semstrait-ir/src/plan/multiset.rs:8-11`; `crates/semstrait-ir/src/primitives.rs:43-58`'
- '`crates/semstrait-planner/src/request.rs:2,6-14`; `src/read_model.rs:10`; `src/formula.rs:9,564-582`; `src/binding.rs:4`; `src/strategy.rs:12`; `src/build.rs:490-510`; `src/decision.rs:1-45,108-130`; `src/grain.rs:1-69`; `src/build/stack.rs:205-218`; `src/semi_additive.rs:6`; `src/error.rs:7`; `tests/build.rs:1525-1538,2665`'
- '`crates/semstrait-manifest/src/manifest/semantic_source.rs:1-34,85-96`; `src/manifest/semantic_definition.rs:1-100`; `crates/semstrait-ir/src/temporal.rs`; `src/manifest/expression.rs:100-150`; `src/compile/expressions/lower.rs:78-106`; `src/compile/expressions/bodies.rs:295-312`; `tests/fixtures/happy_manifest.snapshot.json:150-164,716-738`'
- '`crates/semstrait-api/src/wire.rs:5-6`; `src/orchestrator.rs:13`; `src/catalogs.rs:10-11`'
- '`crates/semstrait-model/src/expr/mod.rs:1-3`; `crates/semstrait-common/src/granularity.rs`; `crates/semstrait-common/src/names.rs`; `src/model/types/entities/temporal.rs:1-27`; `src/model/types/entities/dimension.rs:98-105,250-261`; `src/model/types/entities/relationship.rs:253-258,291-311`; `src/model/types/entities/measure.rs:144-148,235-238`'
- '`crates/semstrait-sql/src/functions/tests/catalog_temporal.rs:100-110,244-252`'
- '`crates/*/Cargo.toml` `[dependencies]`'
links:
- target: 1
  kind: amends
- target: 74
  kind: amends
- target: 85
  kind: amends
- target: 90
  kind: amends
- target: 46
  kind: amends
- target: 89
  kind: amends
---

# semstrait-model keeps the authoring forms, and the manifest, IR and planner name none of its types

## Context and Problem Statement

`semstrait-model` is the YAML authoring crate. ADR 0001's *Decision Outcome* also gave it the shared
vocabulary every stage uses: "the `Expr<L>` family, leaf sets, support enums,
`Literal`/`ColumnRef`, `DataType`/`Grain`, and the `CanonicalFn` name", with
`semstrait-ir` depending upward on it. ADR 0085's rule 3 moved the expression
tree out, and kept "`Grain`; the builders …; the aliases
`UnresolvedSemanticExpr` and `UnresolvedPhysicalExpr`" in `semstrait-model`.
ADR 0090's rule 3 narrowed `semstrait-ir`'s dependency to `SemanticsName`
only, "in each plan node's trace (`SemAnnotation`)"; its rule 6 ruled that "a
crate that uses `Grain` imports it from `semstrait-model`."

Today, non-test code still imports plain names and enums with no authoring
shape from `semstrait-model`, into every later stage:

- `semstrait-ir` imports `SemanticsName` for its plan node's trace
  (`crates/semstrait-ir/src/plan/trace.rs:1`).
- `semstrait-planner` imports `Agg`, `AdditivityType`, `DataKindName`,
  `DimensionType`, `FilterName`, `Grain`, `Integrity`, `SemanticsName`,
  `SemiAdditivityStrategy`, `UnionMode` and `UnresolvedSemanticExpr`
  (`request.rs:2`, `read_model.rs:10`, `formula.rs:9`, `binding.rs:4`,
  `strategy.rs:12`, `semi_additive.rs:6`, `error.rs:7`). The public
  `Request` holds `DataKindName`, `UnresolvedSemanticExpr`, and `FilterName`
  directly on its fields (`request.rs:6-14`).
- `semstrait-manifest`'s stored types hold model types directly: `SemanticSource`
  holds `DataKindName`, and its `RoleDefinition`'s Joinset variant holds
  `Integrity` and `CrossFilter`, and its Unionset variant holds `UnionMode`
  (`manifest/semantic_source.rs:3,16-34,85-96`); `SemanticRoleDefinition` holds
  `DimensionType`, `StoredAggregation(Agg)` and `AdditivityType`
  (`manifest/semantic_definition.rs:4,25-45,100`); `TemporalShape` holds
  `Grain` and `ScdType` (`manifest/temporal.rs:2,9`). Because the manifest's
  stored types hold model types, the planner must import them to read the
  manifest.
- `semstrait-api` imports `parse`, `validate`, `SemanticModel` and
  `CatalogsConfig` (`orchestrator.rs:13`, `catalogs.rs:10-11`) and the request
  wire form (`wire.rs:5-6`).

Some internal types already exist beside `semstrait-model`'s. IR's `MultisetOp {
UnionAll, UnionDistinct }` (`crates/semstrait-ir/src/plan/multiset.rs:8-11`)
mirrors `UnionMode { All, Unique }`; the planner maps `Unique` to `
UnionDistinct` (`crates/semstrait-planner/src/build/stack.rs:212`). IR owns `
Cardinality` and `JoinType` (`crates/semstrait-ir/src/primitives.rs:43,53`),
which the manifest already imports. `UnresolvedSemanticExpr` is only an
alias, `RowExpr<SemanticLeaf>` (`crates/semstrait-model/src/expr/mod.rs:3`).

`Grain` is the roster `Minute, Hour, Day, Week, Month, Quarter, Year`
(now `crates/semstrait-common/src/granularity.rs`). CONTEXT.md's
**Grain** is "the dimensions at which one row stands for one combination of
values, together with the granularity of each of them that has one", and its
_Avoid_ warns against "reading grain as the time axis alone". **Granularity**
is "how coarse the values of one dimension are, drawn from a totally ordered
roster"; today only a temporal dimension has one. So the `Grain` enum is a
granularity. The planner's own grain types are `RequestGrain` and `
SourceGrain { mask, xs, dimensions, axis, granularity, held }`
(`crates/semstrait-planner/src/grain.rs:20,62`).

The YAML spells granularity as `grain:` on a temporal shape
(`crates/semstrait-model/src/model/types/entities/temporal.rs:26`) and
`grains:` on a temporal Dimension's body
(`.../entities/dimension.rs:260`). Both forms deny unknown fields
(`temporal.rs:13`; `dimension.rs:250`, on `TemporalDimensionBody`). The
manifest stores `temporal.grain` (`crates/semstrait-ir/src/temporal.rs`);
`date_trunc`'s parameter is `grain` (`crates/semstrait-sql/src/functions/tests/catalog_temporal.rs:103,247`).
The request already says `granularity`.

Aggregation is stored twice. A Measure's definition stores `"aggregation":
"count_distinct"` (`crates/semstrait-manifest/tests/fixtures/happy_manifest.snapshot.json:157`)
while its tree stores the call (`:719-734`). `count_distinct` is not a
`semstrait-sql` function; ADR 0089's rule 20: "`count_distinct` stays
`semstrait-model`'s sugar for `count` with DISTINCT." For a Metric whose `agg:`
aggregates a formula that reads a Measure (example `regions_with_revenue`,
`agg: count`, `crates/semstrait-planner/tests/build.rs:1530-1538`), compile
stores only the formula (`crates/semstrait-manifest/src/compile/expressions/lower.rs:85-104`,
called at `compile/expressions/bodies.rs:301-307`), and the planner rebuilds
the call from the definition's aggregation, reading its `distinct`
(`crates/semstrait-planner/src/build.rs:500-503`). Test:
`an_agg_over_a_formula_aggregates_its_values_at_the_formula_grain_in_its_own_subtree`
(`crates/semstrait-planner/tests/build.rs:2665`). `StoredAggregation(Agg)`
holds `semstrait-model`'s `Agg`, that is the function and `distinct`,
serialized as a name.

Nothing in the adapter or the API reads the `aggregation` key.

`StoredExpr(Expr<Link>)` is the stored tree type; its leaf is a link, and
load refuses another leaf (`crates/semstrait-manifest/src/manifest/expression.rs:116-145`,
the rule the `Link` deserializer names `RULE_LEAF_IS_A_LINK`).

The planner mixes names and aliases for one semantic source:
`Decision.stand_in`, `DecisionValue::{StandIn, NotRead, NotAnOperand}` hold
`DataKindName`; `DecisionValue::UnknownOperand` holds a `SourceAlias`
(`crates/semstrait-planner/src/decision.rs:11,29-38,117-126`).

## Considered Options

- Keep the vocabulary in `semstrait-model` (today): the planner then links
  the YAML authoring crate (`serde_yaml`, `bon`, `regex-syntax`, the parser)
  only to name names and enums.
- Re-export model types through the manifest: ADR 0090's rule 6 refuses
  re-exports of foreign types.
- `Granularity` in `semstrait-ir`: `semstrait-model` cannot import IR, so
  `semstrait-model` would keep one roster, IR a second, and compile would
  convert between two rosters to keep equal.
- The manifest defines its own `Integrity`, `CrossFilter`, `AdditivityType`,
  `SemiAdditivityStrategy`, `ScdType`, `TemporalShape`, `TemporalType`,
  `JoinsetJoinType` and `SemanticRole`: the manifest is state, and a
  primitive it stores and the planner plans with is defined in
  `semstrait-ir`, not duplicated in the manifest.
- One name type, `SemanticsName`, for every named entity, semantic sources
  included: semantic sources are a separate namespace from semantics, which
  `DataKindName` keeps apart.
- Keep `FilterName` for a filter: a second type for one namespace; binding
  already converts it back with `SemanticsName::new(filter.as_str())`
  (`binding.rs:106-113`).
- Keep the names in `SemiAdditivity`'s `axes` and `by`: it would stay the
  only stored cross-reference by name, unlike a temporal shape's
  `occurred_at`, which already holds a definition id.
- `UnionMode` in `semstrait-sql`: a union is a relational operator, not an
  expression; IR's `MultisetOp` already holds it (Substrait `SetRel.SetOp`,
  Calcite `Union(all)`).
- Store the aggregation only as the stored tree's call, with no `aggregation`
  key: a Metric of two aggregate steps stores its formula alone, so the
  planner would need a new reader for its second step, and `planned_as`
  (`crates/semstrait-planner/src/formula.rs:564-582`) a new test for an
  aggregating entry.
- Keep `StoredAggregation` over `semstrait-sql`'s function and `distinct`,
  with today's bytes: the manifest keeps `count_distinct`,
  `semstrait-model`'s sugar, and a wrapper that adds nothing to the
  function and the flag.
- Rename `SemanticsName` to `SemanticName`: the glossary's **Semantics** is a
  deliberate countable singular.
- The planner names a semantic source by its `SourceAlias`: the alias is an
  internal anchor; the name is descriptive and the id is unique.
- Accept the old YAML keys as an alias: a second spelling to keep for ever.

## Decision Outcome

### What names no model type

1. **Scope.** The manifest's stored form, `semstrait-ir`, and
   `semstrait-planner` — with its public `Request`, `PlanError` and
   `Advisory` — name no `semstrait-model` type. Model types stay in
   `semstrait-model`, in the manifest's `compile` (it reads `semstrait-model`;
   ADR 0001 keeps cross-layer folds there), in `semstrait-api` (parse,
   validate, catalogs, the wire adapter for `where`), and in the `semstrait`
   facade's re-exports (ADR 0009).

### Names

2. **The name types move to `semstrait-common`, shared.** `SemanticsName`,
   `DataKindName` and `str_newtype!`
   (`crates/semstrait-common/src/names.rs`) have no
   authoring shape and mean the same at authoring and at planning.
   `semstrait-model` and every internal crate import them from
   `semstrait-common`. They cannot go to the manifest or IR, because
   `semstrait-model` cannot depend on a crate that depends on it.
   `SemanticsName` keeps its name: CONTEXT.md's **Semantics** is a
   countable singular — "one name is always one logical semantics".
   `DataKindName` stays its own type: a semantic source's name is a
   separate namespace from semantics, which `semstrait-model` checks apart
   (`DuplicateDataKindName`, `DuplicateSharedSemanticsName`,
   `crates/semstrait-model/src/error/validate.rs:20-28`) with no rule that
   compares the two, while CONTEXT.md's **Semantics** is a dimension,
   measure, metric or filter, never a data kind's own name. `FilterName`
   retires: a filter is named with `SemanticsName` everywhere —
   `semstrait-model`, the request, the API — because CONTEXT.md's
   **Semantics** includes a filter and the manifest already stores a
   filter as a semantic definition with role Filter
   (`manifest/semantic_definition.rs:507-512`); `SemanticsName`'s
   description names a Filter too.

### Granularity

3. **Granularity.** The enum `Grain`
   (in `semstrait-model`) becomes
   `Granularity` and moves to `semstrait-common`: one roster for a temporal
   Dimension's granularities, a temporal shape's granularity, the request's
   granularity, and the planner's coarseness checks. A field, a variable or a
   function that holds or returns one value of it is named `granularity`. A
   name for the glossary's grain keeps `grain`: the planner's `RequestGrain`
   and `SourceGrain` (`crates/semstrait-planner/src/grain.rs:20,62`) stay the
   planner's own types.

### Types the manifest stores and the planner plans with

4. **The placement rule.** A type has an authoring shape when it has a YAML
   spelling, a default, or a body; `semstrait-model` keeps it.
   `semstrait-ir` defines each type whose values the manifest stores and
   the planner plans with, even where it equals `semstrait-model`'s type
   today, because it is the planner's own machinery in resolved form; the
   manifest is state, so it imports that type and defines only its own
   state structures. `semstrait-common` defines what the model shares with
   them — the names (rule 2) and `Granularity` (rule 3) — so this rule does
   not put `Granularity` in `semstrait-ir` against rule 3:

   | Model type (stays) | Where the type is defined now |
   |---|---|
   | `UnionMode { All, Unique }`, default `All` | IR's `MultisetOp { UnionAll, UnionDistinct }` (unchanged) |
   | `Integrity { Enforced, Assumed, None }`, `CrossFilter { None, Left, Right, Both }` | IR's own `Integrity` — no default, because a default is an authoring shape and compile always writes the value — and `CrossFilter` |
   | `AdditivityType { Full, Semi(SemiAdditivity), Non }`, `SemiAdditivityStrategy { Latest, Earliest }` | IR's own types, with `SemiAdditivity`. CONTEXT.md's **Additivity**: "two inputs that stay apart: the aggregate function's additivity … and a Measure's declared additivity narrows it" |
   | `ScdType { Type1, Type2 }` | IR's own `ScdType`, with `TemporalShape` and `TemporalType` |
   | `DimensionType { Temporal(body), Categorical, Binary, Geo, Bucketed(body), Metadata(body) }` | the manifest's own `DimensionType`, with the same content, though the planner reads it — a trade-off (Consequences) |
   | `AiContext` | the manifest's own `AiContext`, descriptive data the planner does not plan with |
   | `Agg { function, distinct }`, spelled by name (`count_distinct`) | `semstrait-sql`'s `AggregateFunction` and a `distinct` flag (rule 5) |

   `JoinsetJoinType { Join(JoinType), Cross }` and `SemanticRole {
   Dimension, Measure, Metric, Filter }` are `semstrait-manifest` types
   today (`manifest/semantic_source.rs:158`,
   `manifest/semantic_definition.rs:507`), not `semstrait-model` types;
   they move to `semstrait-ir` too, beside the manifest's other imported
   primitives. The manifest's types keep today's names, and their stored
   JSON values do not change, except rule 5's aggregation and rule 6's
   semi-additive references.

### Aggregation

5. **The aggregation is stored as `semstrait-sql`'s function and its
   `distinct`.** A semantic definition stores its aggregation in two
   fields: `aggregation`, `semstrait-sql`'s `&'static AggregateFunction`,
   stored by its canonical name, and `distinct`, a flag that is left out
   when false. Compile lowers `count_distinct`, `semstrait-model`'s sugar,
   to `count` with `distinct`, the form that the call in the entry's
   stored tree already has. The manifest holds no `StoredAggregation` and
   no `Agg`. Each reader reads the same fact as today: a Metric whose
   `agg:` aggregates a formula that reads a Measure keeps today's form,
   compile stores the formula alone, and the planner builds the second
   step's call from `aggregation` and `distinct`
   (`crates/semstrait-planner/src/build.rs:500-503`). `semstrait-model`
   keeps `Agg` as the authoring form of `agg:` (`of_name`, the
   `count_distinct` spelling).

   | | Today | After |
   |---|---|---|
   | a `count_distinct` Measure's definition | `"aggregation": "count_distinct"` | `"aggregation": "count", "distinct": true` |
   | its tree's call | `{"name": "count", …, "distinct": true}` | the same |
   | a `sum` Measure's definition | `"aggregation": "sum"` | the same |

### Semi-additive references

6. **A semi-additive Measure's stored references are definition ids.**
   `SemiAdditivity`'s `axes` and `by` hold the semantic definitions' ids,
   the way a temporal shape's `occurred_at` already does. Compile resolves
   the authored names to ids and keeps its name-based checks and texts; the
   planner finds each definition by id, as it already does for a temporal
   shape's time axis (`semantic_definitions.get(id)`,
   `crates/semstrait-planner/src/read_model.rs:402-405`). Plans, traces and
   error texts do not change. An authored `by` that names no semantic
   definition is a compile error, whether or not a source serves the
   Measure, as an authored axis that names no definition already is.

   | | Today | After |
   |---|---|---|
   | a semi-additive Measure's `axes` | `["account"]` | `["<account's definition id>"]` |
   | its `by` | `"date"` | `"<date's definition id>"` |

### The request and naming a semantic source

7. **The request's predicate.** `Request.predicate` is
   `semstrait_sql::expr::RowExpr<SemanticLeaf>`. `semstrait-model` keeps its
   alias `UnresolvedSemanticExpr` for its own use (ADR 0089, rule 6).

8. **A semantic source is named by its name or its id.** Inside the planner
   and in what it returns, a semantic source is named by its name
   (descriptive) or its id (unique), never its `SourceAlias`, which stays an
   internal anchor. Today's name sites keep the name.
   `DecisionValue::UnknownOperand(SourceAlias)` keeps the alias: it is built
   only when the manifest holds no source for that alias (`decision.rs:29-38`),
   so no name or id exists.

### IR depends on no model crate

9. **IR depends on no model crate.** `SemAnnotation`'s names are
   `semstrait-common`'s; a data kind's name is a `DataKindName`, not a
   `SemanticsName` (today `SemAnnotation::DataKind(SemanticsName)`,
   `trace.rs:8`, and the planner converts at `build.rs:1618-1629`). ADR 0095
   gives the trace's form.

### Granularity in the YAML and the manifest

10. **Granularity in the YAML and the manifest.** `grain:` becomes
    `granularity:`, `grains:` becomes `granularities:`, the manifest key
    `grain` becomes `granularity`, a temporal Dimension's stored `grains`
    becomes `granularities`, and `date_trunc`'s parameter `grain` becomes
    `granularity`. A stored key that holds a Grainset's grains keeps
    `grains`. The old keys get no alias. Both YAML forms deny unknown fields,
    so a model that still writes `grain:` fails to parse with an "unknown
    field" error and never loses the value silently.

### Crate edges

11. **Crate edges after the change.** Verified against each `crates/*/Cargo.toml`
    `[dependencies]` table:

    | Crate | Depends on | For |
    |---|---|---|
    | `semstrait-common` | none | gains the names, `str_newtype!`, `Granularity` |
    | `semstrait-sql` | `semstrait-common` | `Tree`, `Visitor`, `Rewriter`, `ValidateError`, `Dialect` |
    | `semstrait-model` | `semstrait-common`, `semstrait-sql` | the expression, function and data types; the diagnostics, `EntityId` and `io` |
    | `semstrait-ir` | `semstrait-common`, `semstrait-sql`, `semstrait-catalog` (`FileFormat`, ADR 0067) | the plan's tree and calls; the scan's file format. No `semstrait-model` |
    | `semstrait-manifest` | `semstrait-common`, `semstrait-sql`, `semstrait-model`, `semstrait-ir`, `semstrait-catalog` | compile reads `semstrait-model`; the types `semstrait-ir` owns, among them `Integrity`, `CrossFilter`, `AdditivityType`, `SemiAdditivityStrategy`, `ScdType`, `TemporalShape`, `TemporalType`, `JoinsetJoinType` and `SemanticRole` (rule 4); the catalog compile reads |
    | `semstrait-planner` | `semstrait-common`, `semstrait-sql`, `semstrait-ir`, `semstrait-manifest` | the plan's tree; the manifest it reads. No `semstrait-model` in `[dependencies]` (tests that build a model keep it as a dev-dependency) |
    | `semstrait-api` | `semstrait-common`, `semstrait-model`, `semstrait-ir`, `semstrait-manifest`, `semstrait-catalog`, `semstrait-planner`, `semstrait-adapter`, `semstrait-adapter-datafusion` (optional) | parse, validate, catalogs, the `where` adapter; `SemanticRole` (`rest.rs:12`), behind the `rest` feature; the DataFusion dialect's rules, behind the `datafusion` feature |

    Today `semstrait-ir` and `semstrait-planner` each also name
    `semstrait-model` in `[dependencies]`; rules 1 and 9 drop it from both.

### Amends

- **0001:** *Decision Outcome* as ADR 0085 amended it ("`semstrait-ir` still
  depends on `semstrait-model`, for `SemanticsName` only (ADR 0090, rule 3)"):
  `semstrait-ir` depends on no model crate (rule 9), and `Grain` is
  `semstrait-common`'s `Granularity` (rule 3).
- **0046:** *Decision Outcome*: "the manifest is a storage standard, not a
  vocabulary owner. Every type it persists is owned by `semstrait-ir`,
  `semstrait-model`, or `semstrait-catalog`", as ADR 0090 last amended it
  ("also names `semstrait-manifest`, for its own storage types"); and
  "`Grain` is `semstrait-model`'s, and a crate that uses either imports it
  from there" (ADR 0090's words, near 0046:37): the names and `Granularity`
  are `semstrait-common`'s (rules 2 and 3); `semstrait-ir` owns `Integrity`,
  `CrossFilter`, `AdditivityType` (with `SemiAdditivity`),
  `SemiAdditivityStrategy`, `ScdType`, `TemporalShape`, `TemporalType`,
  `JoinsetJoinType` and `SemanticRole`, which the manifest imports; the
  manifest owns its own `DimensionType` and `AiContext` (rule 4); and stores
  the aggregation as `semstrait-sql`'s function and a `distinct` flag (rule
  5).
- **0074:** *Every request is planned against its grain first*, "The code
  keeps its `Grain` roster for the granularity until the model, the manifest
  and the IR get the sharper words.": the roster is `semstrait-common`'s
  `Granularity` (rule 3). The next sentence, "Where existing ADRs say "grain"
  for the temporal roster, they mean the granularity.", stands. The same
  section's "Its author may declare it on that composition, as
  `extras.temporal.grain:`." (near 0074:359-360) and *Consequences*' Neutral
  listing of `crates/` changes, "a composition that is a constituent may
  declare `extras.temporal.grain:`" (near 0074:1658): rule 10 spells the key
  `granularity`.
- **0085:** rule 3's row ("`Grain`; the builders …; the aliases … | stay in
  `semstrait-model`"): `Grain` moves to `semstrait-common` as `Granularity`
  (rule 3); the builders and the aliases stay.
- **0089:** rule 20's "and `count_distinct` as today" and "Each of today's
  values stores the same bytes as today.": compile stores `count_distinct`
  as `"aggregation": "count", "distinct": true` (rule 5), and every other
  value stores the same bytes as today. "`count_distinct` stays
  `semstrait-model`'s sugar for `count` with DISTINCT." stands.
- **0090:** rule 3 ("keeps its dependency on `semstrait-model` for
  `SemanticsName` only"): rule 9. Rule 6 ("A crate that uses `Grain` imports
  it from `semstrait-model`", and the dependency table's rows for
  `semstrait-ir` and `semstrait-manifest`): rules 3 and 11. *In short*'s "it
  takes only `SemanticsName` from `semstrait-model`" (near 0090:177-178):
  rule 9.

### Consequences

- Good: `semstrait-planner` and `semstrait-ir` link no YAML authoring code;
  no stored manifest field has a `semstrait-model` type; one word,
  granularity, runs from YAML to plan.
- Neutral: the moves and re-imports; the facade's re-exports point to the
  new homes; `Integrity`,
  `CrossFilter`, `AdditivityType`, `SemiAdditivityStrategy`, `ScdType`,
  `TemporalShape`, `TemporalType`, `JoinsetJoinType` and `SemanticRole` move
  to `semstrait-ir`; each authored enum exists twice — in `semstrait-model`,
  and in `semstrait-ir` or the manifest — compile maps one to the other, and
  the move itself changes no stored bytes (rule 6 changes the semi-additive
  references, and rule 5 the aggregation); fixtures and snapshots that hold
  `"grain"`, `"grains"`, `grain:`, `"aggregation": "count_distinct"`, or a
  semi-additive Measure's `axes` or `by` change.
- Bad, a trade-off: `DimensionType` (with `BucketSpec`, `MetadataSource`)
  stays the manifest's own type, although the planner reads it; the gain is
  a smaller change.
- Bad, a trade-off: every model that writes `grain:`, `grains:` or
  `date_trunc: {grain: …}` fails to parse until its author renames the key:
  the two YAML keys with "unknown field", and the `date_trunc` call with the
  call check's "no parameter is named `grain`"; gain: one word across YAML,
  manifest, request and code.

## More Information

Decided in a design session on 2026-10-29/30. ADR 0095 records the trace form that rule 9 points to.
