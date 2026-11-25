---
number: 82
title: A physical read types an unobserved column by the semantics mapped to it, and compile writes that type into the schema
status: accepted
date: 2026-10-05
source:
- CONTEXT.md
- docs/adr/0017-external-access-seams.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md
- docs/adr/0081-a-partition-key-is-a-column-and-partition-def-names-the-dimension-that-maps-it.md
- the first projection's casts, ruling 4
- two semantics on one column, ruling 11
- compile's widening order, `is_widening` in `crates/semstrait-manifest/src/compile/expressions/lower.rs`
- DataFusion 55.0.0, `datafusion-physical-expr-adapter` `src/schema_rewriter.rs` (the cast of each file's column), and `arrow-cast` 59.3.0 `src/cast/string.rs` (a text value that does not cast)
- decided, and on 2026-10-17 for the type that compile writes into a model-fed schema
links:
- target: 66
  kind: amends
- target: 61
  kind: amends
- target: 81
  kind: amends
- target: 89
  kind: amendedby
---

# A physical read types an unobserved column by the semantics mapped to it, and compile writes that type into the schema

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0089](0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md): in *Which semantics types the column*, for a column that compile does not observe, "Its signature's tree is then the column" reads with the cast of ADR 0089, rule 16; "A `count` or `count_distinct` takes part" also covers an `approx_count_distinct` (ADR 0089, rule 16). In *Compile writes the read type into the schema*, every other read of such a column, in an expression mapping, a Filter or a Measure's `filter:`, takes the read type too (ADR 0089, rule 16). The rest of 0082 stands.

## Context and Problem Statement

The planner writes a type for each column that a scan reads. ADR 0066 ruling 5:
"The planner's physical leaf is ADR 0061's one conversion: catalog-observed type
and nullability; else the definition's `data_type` where the Dimension's tree
*is* a bare `Column` (result and leaf are one value); else the planner's own
`PlanError::UnresolvedType` (a planned planner variant, not an IR one) naming
relation and column."

Compile observes a column's type only through a catalog. ADR 0061's *Context
and Problem Statement* says "There is no observer to consult for a file source",
so compile writes each column of a path source with `data_type: None` and
`nullable: None`. A column that only a Measure reads therefore has no type, and
the planner refuses every request for that Measure.

Example: Dataset `orders_daily` in `test_data/comprehensive_ecommerce.yaml`
reads `s3://warehouse/orders/daily/` in Parquet. The Dimension `region` maps
`region_code`, so ruling 5 types `region_code` as `String`. The Measure
`revenue` (`sum`, `data_type: float64`) maps `amount`, and the Measure
`order_count` (`count`, `data_type: int64`) maps `order_id`. No Dimension maps
`amount` or `order_id`, so a request for `revenue` or `order_count` fails with
`UnresolvedType`. Every path source in that model and in
`e2e_full_coverage.yaml` has this shape.

The scan cannot leave the type out. Substrait's `ReadRel.base_schema` has a type
for each column, and ruling 5 keeps plan schemas total. ADR 0068's files
consumer, in *The DataFusion rules*, "reads each `LocalFiles` item by its full
URI and its format, then applies the base schema". ADR 0073 adds the read of a
folder with partitions, and ADR 0081 has that consumer cast each key to the
read's type.

What this ADR builds on is already ruled, and it does not restate it. The
projection that turns physical columns into the semantic source's interface
names and casts each semantics: ADR 0071's *Decision Outcome* places a predicate
"directly above the projection that names and casts it", and the human described
that projection (ruling 4) as "the projection adds the semantic names,
first-level expressions, casts and the alias (`cast(expr as <type>)
semantic_definition.name`)". Each semantics carries its own cast there: the
human, "first projection for each seamntic will carry its own type",
and ruling 11, "Two Dimensions that map one column with different
declared types are two semantics". A Measure's cast to its declared type is
kept: ADR 0071's *A tree gives the computation, and the role gives the
placement*, "dropping its `Cast` changes the output type away from the declared
`data_type`". This decision says when that projection writes a cast,
and that a count's input is not cast.

## Considered Options

- **The widest declared type of the semantics mapped to the column** (chosen).
- **A Dimension first, then the first semantics in name order.** Refused. The
  human: "all semantics , no matter it is dimension or measure/metric", and the
  read type is to lose nothing, "( to prevent loss - for example double in
  priority out of long )". Name order can pick the narrower type: with
  `order_date` (`date`) before `order_ts` (`timestamp`) on one column, `order_ts`
  would always show midnight.
- **Only the semantics that the request demands.** Refused: the read type would
  change with the request, so one semantics could give two values. The human
  chose every semantics on the Dataset.
- **Compile reads the Parquet footer of each listed object**, ADR 0061's "named
  successor direction" (its *Considered Options*). Not in the MVP. The human:
  "we do not consider right now any footer reads and schema sampling". It would
  also change ADR 0017, whose *Decision Outcome* says that `FileSystem`'s `read`
  "MUST NOT parse Parquet footers, ORC stripes or CSV headers, so schema comes
  only from a catalog or the author".
- **The semantic model writes the column types**, as a new list on `storage`
  (for example `columns: { amount: double }`). Refused: the declared `data_type`
  of each semantics already gives the type. The human: "what are declared
  semantics, starts to be a type for column".
- **The consumer reads the files' own types, and the scan's type is a
  placeholder.** Refused: besides ADRs 0066 and 0061, it changes ADR 0068's
  consumer, which applies the base schema.
- **Compile writes the declared type into the source schema**
  (`SourceColumn.data_type`). Chosen for a model-fed schema and a
  semantics that maps the column directly (*Compile writes the read type into
  the schema*). ADR 0061 struck the typing of a leaf from a type that describes a
  definition's result; a direct mapping reads the column at a declared type
  (*Compile writes the read type into the schema*), and the typing of a leaf
  beneath an expression stays struck.
- **A column that only a count is mapped to is refused**, because a count's
  declared type is the type of the count, not of the column. Refused: every
  count over a path source in the two models above would fail, `order_count`
  among them.

## Decision Outcome

On a model-fed schema (a path source, or a table compiled with no catalog: ADR
0046), no catalog observes a column's type. There the column takes its **read
type**: the type that *Which semantics types the column* gives from the semantics
mapped to it. Compile writes the read type into the schema, as the column's
`data_type` (*Compile writes the read type into the schema*), the physical read
reads the column as that type, and ruling 5's one conversion reads it from the
schema. On a catalog-derived schema, a referenced column whose native type maps
to no type fails the compile (ADR 0086, rule 7), so every
column there has the type that the catalog observed.

### Which semantics types the column

A semantics is mapped to a column when its semantic mapping on the Dataset whose
physical read it is, is that column: `region: region_code`, `revenue: amount`,
or `auto` with a column of that name. Its signature's tree is then the column,
`Column(region_code)`, or a Measure's aggregate over it, `Sum(Column(amount))`.
A mapping that is an expression over the column (`length(name)`, `price * qty`)
does not map the semantics to it. A named Filter has no `data_type` and gives no
type.

- **Every semantics on the Dataset that is mapped to the column takes part**, a
  Dimension, a Measure or a Metric, whatever the request demands, and so does
  every semantics mapped to it on each other Dataset that reads the same
  physical source (ADR 0070), because the schema holds one type for the column.
- **A `count` or `count_distinct` takes part only when every semantics that
  gives a type to the column is a count**, because its declared type is the type
  of its result.
- **The widest declared type wins**: the one that each other declared type is
  equal to or widens to. A type widens to each type that a chain of these steps
  reaches: `Short` → `Integer` → `Long` → `Double`; `Float` → `Double`; `Integer`
  → `Decimal`; `Long` → `Decimal`; a `Decimal` → a wider precision at the same
  scale; `Date` → `Timestamp`; a `Timestamp` or a `Time` → a higher precision.
  The steps are compile's `is_widening` pairs and two temporal steps. They apply
  to this rule only: compile's `is_widening` does not change.
- **Where no declared type is equal to or wider than all the others** (`Long`
  and `String`, `Date` and `Long`, or two `Decimal`s with different scales), the
  first semantics in name order gives the type.
- **Nullability that compile did not observe** stays `true`, as ruling 5
  decides.
- **`PlanError::UnresolvedType` remains** for a column that no semantics with a
  `data_type` is mapped to, such as a column that only an expression mapping or a
  Measure's `filter:` reads.

For `orders_daily`, `amount` is `Double` from `revenue`, and `order_id` is `Long`
from `order_count`. If the Dimensions `order_date` (`date`) and `order_ts`
(`timestamp`) both map `created_at`, the read is `Timestamp`, and `order_date`
casts from it. If the Dimension `order_ref` (`string`) and `order_count` both map
`order_id`, the read is `String`.

### Compile writes the read type into the schema

The read type is not an observation, and the schema's arm still says that no
catalog observed the column (ADR 0046). But each semantics that maps the column
directly reads it at a declared type: a Dimension's result is the column (ADR
0066, ruling 5), and a Measure other than a count reads the column cast to its
own declared type (ADR 0089, rule 16). So on a model-fed schema compile writes
the read type as the column's `data_type`, and `nullable` stays `None`. A column that only an
expression reads gets no type, as ADR 0061 has it: `qty` in `sum(price * qty)`
stays `None`, and the planner refuses it (`PlanError::UnresolvedType`). The
planner reads the schema's type at its one conversion, and it does not work the
type out again from the trees, where an authored cast over the column and
compile's own cast store the same tree (ADR 0089, rule 16). The human:
"best effort we can do - is compile write type into schema - assumption is that
data-types we can potentially derive - is the ones declared in
semantic_definition. Later on we could bring file schema reader ( like parquet
header ) to find this information, but for now - schema column gets type from
attached semantic_definition". Compile reads no
file content, so it compares no file with another file and no file with a
declared type. The consumer reads each file as the physical read's types, and a
value that does not cast fails in the engine. ADR 0077's
`CompileError::AutoMappingWithoutSchema` still refuses `auto` on a constituent
over a source with no observed schema.

### Amends

- **0066:** on a model-fed schema, ruling 5's branch
  "else the definition's `data_type` where the Dimension's tree *is* a bare
  `Column` (result and leaf are one value)" reads: the column's type in the
  schema, which compile writes from the semantics mapped to the column as 0082's
  *Which semantics types the column* gives, whatever their roles;
  `PlanError::UnresolvedType` remains for a column that no semantics with a
  `data_type` is mapped to. In *Amends*, the entry for ADR 0061 reads with 0082's
  entry for 0061. The rest of 0066 stands.
- **0061:** in *Decision Outcome*, "`None` means one thing and only one thing:
  **this compile did not observe it.** Compile never fills the gap with a value
  derived from somewhere else" holds for a catalog-derived schema; on a
  model-fed schema, compile writes the type that 0082's *Which semantics types the
  column* gives into a column that a semantics maps directly, and `None` there
  means that no semantics with a `data_type` maps the column directly. In *The
  reader contract, stated normatively because no consumer exists to derive it
  from*, clause 1's "the conversion is the single place a `None` is resolved"
  holds for a column that keeps `None`, and clause 2, which bars substituting
  "the referencing definition's declared type at the point of use", holds: the
  type that compile writes is in the schema, and no reader substitutes one. In
  *Amends*, its entry for 0046's "The typing clause is struck: a referencing
  definition's declared type is a claim about the *definition's result*, not
  about any leaf beneath it" holds for a column that only an expression reads.
  The refusal in *Considered Options* to broadcast a declared type onto the
  persisted columns stands for a column that only an expression reads (0082's
  *Compile writes the read type into the schema*). The rest of 0061 stands.
- **0081:** in *A partition key is a column*, the ruling 5 that it quotes reads
  as 0082 amends it: a partition column takes the type that 0082's *Which
  semantics types the column* gives, and `PlanError::UnresolvedType` remains for
  a column that no semantics with a `data_type` is mapped to. The rest of 0081
  stands.

### Consequences

- Good, because a Measure mapped to a column of a path source plans, so the
  files gates of the e2e oracle run.
- Good, because compile reads no file, and the planner reads the read type from
  the schema, with no second computation that an authored cast could mislead.
- Neutral: a column has one read type for every Dataset that reads its physical
  source. Where `orders` maps `amount` to `revenue` (`double`) and `orders_eu`
  maps it to `revenue_eu` (`decimal(18,2)`), the two types do not widen to each
  other, so name order gives `double`, and `revenue_eu` casts from it.
- Bad, because the read type enters the physical source's `SchemaId`, which
  `mint_schema_id` hashes, so a change to a semantics' declared type changes the
  source's id (ADR 0061: a type that no provider reported "takes part in the
  artifact's identity"). The gain is that the schema holds the read's type, in
  one place.
- Bad, because a count over a text column fails in the engine where only counts
  are mapped to the column. DataFusion reads `order_id` as `Long`, and `'ORD-1'`
  fails ("Cannot cast string … to value of Int64 type"). A Dimension mapped to
  `order_id` as `string` avoids it.
- Bad, because each row is cast before any aggregate, in the physical read and
  again in the projection that names the semantics. Where a declared type is
  narrower than the values, the value changes and nothing fails: a `sum`
  declared `int64` over `0.6` and `0.6` reads each value as `0`.
- Bad, because the steps count `Long` → `Double` as widening, as compile does,
  although a `Long` above 2^53 loses its last digits as a `Double`.
- Bad, because where name order picks the type, another semantics can lose
  values: a `float` read type loses the digits of an `int64` value above 2^24.
- Bad, because on a table compiled with no catalog, DataFusion's table consumer
  refuses a physical read's type that differs from the registered table's type,
  before the plan runs.
- Bad, because a column that no semantics with a `data_type` is mapped to is
  still refused.

## More Information

Decided on 2026-10-05 and 2026-10-06, and on
2026-10-17 for the read type that compile writes into a model-fed schema. The human, on the
rule: "first projection type cast and assign semantic name for everything after
this can be correctly referenced. so it is not really "data type for column" it
is casting original type which can be unknown for mvp ( for example we don't
read footer, or semantic_mapping=auto ) it is just we don't care it much for
now - when there is schema provider ( catalog ) we can reconcile , when there is
no such base - we are wrapping up everything into cast, to avoid potential
problems with types for each layer of expressions ( from one semantic source to
another )". On the physical read's type: "all semantics , no matter it is
dimension or measure/metric ( take it as non hard rule, i might miss
something ) - is separate field, and each field can be having its own type -
column which is sourced - can be read once with type coming from catalog, when source are
files w/o catalog - we can use lowest in type possible ( to prevent loss - for
example double in priority out of long ). and others semantics when their
starts to be field - can make a cast and store their own type." On a count's
result: "type infered from fucntion result, but
can be wrapped up into type which defined for semantic definition to be aligned
with what model declares and manfiest consiequently." On the costs: "right now
ours priority - iceberg format which gives us schema _ polaris catalog, so
implementation for files might be more naive".

The engine side: DataFusion 55 casts each file's column to the physical read's
type as it reads the file (`schema_rewriter.rs`), and a text value that does not
cast fails the query (`arrow-cast` `src/cast/string.rs`: "Cannot cast string
'{v}' to value of {} type").
