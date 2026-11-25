---
number: 81
title: A partition key is a column, and partition_def names the Dimension that maps it
status: accepted
date: 2026-10-05
source:
- CONTEXT.md
- docs/adr/0031-metadata-as-compile-stratum.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0070-a-path-source-is-one-base-path-read-in-one-format.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- '`MetadataSource` and `PartitionDef` in `crates/semstrait-model`, and the test `a_metadata_dimension_is_an_ordinary_pooled_literal` in `crates/semstrait-manifest/tests/expression_pool.rs`'
- the research on how engines read a partition column (DataFusion 55.0.0, Spark v3.5.3, Hive 3.1.3 and 4.0.0, Iceberg 1.7.0, Substrait v0.103.1)
- a probe of DataFusion 55's Substrait producer on a partitioned table
links:
- target: 70
  kind: amends
- target: 31
  kind: amends
- target: 73
  kind: amends
- target: 82
  kind: amendedby
- target: 89
  kind: amendedby
---

# A partition key is a column, and partition_def names the Dimension that maps it

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0082](0082-a-physical-read-types-an-unobserved-column-by-the-semantics-mapped-to-it-and-compile-writes-that-type-into-the-schema.md): in *A partition key is a column*, the ruling 5 that it quotes reads as 0082 amends it: a partition column takes the type that 0082's *Which semantics types the column* gives, and `PlanError::UnresolvedType` remains for a column that no semantics with a `data_type` is mapped to. The rest of 0081 stands.
> - [ADR 0089](0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md): in *A partition key is a column* and in *`partition_def` names the Dimension*, "its signature is a bare `Column`" holds for the mapping: the Dimension maps the column directly, and because a partition column is never observed, its stored tree is the cast of ADR 0089, rule 16, over the column; "The signature of `order_year` on `sales` is `Column(year)`" and "the signature is still `Column(year)`" read with that cast. In *The manifest stores no link*, "a signature that is a bare `Column`" reads as the signature of a Dimension that maps the column directly. The rest of 0081 stands.

## Context and Problem Statement

ADR 0070, in *A partition key feeds a metadata Dimension*, admits the form
`metadata: { partition: { column } }` and leaves its carrier open: "How a
Dataset's signature marks a partition-fed Dimension is a carrier still to build,
under 0031's rule that metadata is never a `PhysicalExpr` variant." ADR 0073,
in *The physical read below a Dataset*, then makes a physical source with
discovered partitions one read of its base path: "The engine discovers the
partitions under the base path and supplies the key columns", and "The planner
writes no branch per partition, no literal and no union." Its consequence for
the manifest is: "The planner needs a partition-fed Dimension to be readable as
a column, named by its key."

Under ADR 0037 a metadata value is one literal. The test
`a_metadata_dimension_is_an_ordinary_pooled_literal` asserts the signature of a
path-token Dimension as `Expr::Literal(Literal::String("year=2024".into()))`
(ADR 0085, rule 7). One literal cannot hold a value that changes from
partition to partition inside one read.

The engines read a partition key as a column. The research
on partition columns compares them:

| | DataFusion 55 | Spark 3.5.3 | Hive 3.1 / 4.0 | Iceberg |
|---|---|---|---|---|
| Declared | on the table, `[(name, type)]`, by folder position | found from the paths | `PARTITIONED BY (name type)`, in folder order | a partition spec over a table column |
| In a query | an ordinary column, after the file columns | the same | the same | the table column itself |
| Key name | exact case | case ignored by default | case ignored | by field id |
| `__HIVE_DEFAULT_PARTITION__` | text; the field is never NULL | NULL | NULL for a type that is not string (read from the code) | — |
| A file column with the key's name | the query fails with a duplicate field name; with an explicit file schema, the folder value | the folder value | DDL error | no clash |

No field of Substrait's `ReadRel`, `NamedTable`, `LocalFiles` or `FileOrFiles`
marks a partition column. In a probe of DataFusion 55's Substrait producer,
with the partition columns declared Int32, DataFusion's own producer writes a
partitioned table as `Read[t => order_id:i64?, amount:i64?, year:i32,
month:i32]`. The partition columns are ordinary fields of the read.

So the question is which form a partition value takes in the semantic model and
in the semantic signature.

## Considered Options

- **A partition key is a column that a Dimension maps; `partition_def` names
  that Dimension, and compile infers the link where it discovers the key**
  (chosen).
- **ADR 0070's metadata form, with a `Column` signature.** Refused. The human
  said that a partition key is "technically remain just a column which part of
  semantic_mapping". A Dimension of `type: metadata` also has no other category,
  so a key such as `dt=2026-01-01` could not feed a temporal Dimension or a time
  axis.
- **ADR 0070's metadata form, with the key read as String and the signature
  `Cast(Column(year), <declared type>)`.** Refused with the metadata form.
- **`partition_def` as a list of typed columns**, `{ column, data_type }`, as
  Hive's `PARTITIONED BY`. Refused. The human: "instead of partition_def {
  column } - i think it is correct specify dimension linkage". The column name
  and its type are then stated once, in the Dimension's mapping and definition.
- **`partition_def` as a plain list of Dimension names, with `range` and `list`
  retired.** Refused because it drops ADR 0070's checks of `list` values.
- **A flag on the Dimension's site** (`is_partition: true`, as `is_key:` marks a
  key dimension). Refused because it states a fact about storage on the semantic
  interface.
- **The link stored in the manifest.** Refused because no rule reads it in the
  first versions, and ADR 0070 refused "Copying the model's partition definition
  into the manifest".
- **A compile error for a mapping that names a discovered key that no entry
  links.** Refused. The human: "compile might infer this if discover - this is
  basically internal metadata which marks dimension as partition key, and
  nothing more for now".
- **The column spelled as the folder spells the key** (`Column(Year)` for
  `Year=` folders and the mapping `year`). Refused. The human said that "Year"
  and "year" "means exactly the same". ADR 0070 already matches a key without
  regard to ASCII case, so the plan names the column as the mapping writes it.
- **`NullIf(Column(year), '__HIVE_DEFAULT_PARTITION__')` in the signature.**
  Refused because it puts a convention of the Hive layout into the signature.
  The consumer reads that layout (ADR 0073).

## Decision Outcome

### A partition key is a column

On a semantic source that reads by `Prefix` a physical source with partitions
(ADR 0070), each discovered key is a **partition column**: the engine gives its
values from the folder names, not from the files (ADR 0073). A Dimension reads
a partition column through an ordinary column mapping, and its signature is a
bare `Column`:

```yaml
sales:
  storage:
    format: parquet
    paths: [s3://lake/orders/]        # folders year=2026/region=eu/
    partition_def:
      - { type: range, dimension: order_year }
      - { type: list,  dimension: order_region, values: [eu, us] }
  semantic_mapping:
    order_year: year
    order_region: region
```

The signature of `order_year` on `sales` is `Column(year)`.

- **The Dimension keeps its category.** Its `type:` is categorical, temporal or
  any other category, so a key such as `dt=2026-01-01` can feed a temporal
  Dimension that is the time axis.
- **The column is named as the mapping writes it.** The mapping and the key
  match without regard to ASCII case, as ADR 0070's *The listing and its
  partitions* rules that "Keys match without regard to ASCII case". So
  `order_year: year` reads `Year=` folders, and the signature is still
  `Column(year)`. The consumer gives the key under the name of the read's
  column.
- **The column joins the physical source's column set**, because an expression
  names it (ADR 0061).
- **The planner reads it as any other column.** ADR 0066's ruling 5 types it. A
  path source observes no column type (ADR 0061: "There is no observer to
  consult for a file source"), so ruling 5 gives "the definition's `data_type`
  where the Dimension's tree *is* a bare `Column` (result and leaf are one
  value); else the planner's own `PlanError::UnresolvedType`". A predicate on
  the column is placed by ADR 0071, and the engine skips partitions by it (ADR
  0073).
- **The signature is not a `Literal`.** So the Dimension never makes ADR 0073's
  Unionset exclude a branch, and it never makes branches disjoint (ADR 0074).
- **The folder value wins.** On this semantic source, every mapping that names
  the key reads the folder value, a Measure's mapping included. A file column
  with the key's name is not read on this semantic source. A `Pattern` or `Uris`
  semantic source over the same physical source discovers no partitions (ADR
  0070), so it reads the file column.

### `partition_def` names the Dimension

The semantic model's `partition_def` is a list of entries
`{ type: range | list, dimension, values? }`. `dimension` replaces `column`, and
`values` is for `list` only. Each entry is a claim that compile checks:

- The entry names a Dimension of this Dataset, and no other entry names it.
- On this Dataset, that Dimension maps a bare column, so its signature is a
  bare `Column`. Any other signature, such as a literal, an expression or a
  path-token value, is a compile error.
- That column is a discovered key on each of the Dataset's sources. ADR 0070's
  table in *The model's partitions are checked, never copied* applies, with "The
  model's partition column", "The model's column" and "an undiscovered column"
  read as the column that the entry's Dimension maps. Its rows for `list`
  values apply too.
- An entry is an error on a table source, on a semantic source that does not
  read by `Prefix`, and on a semantic source whose physical source has no
  partitions. The error is raised for each such source. It names the Dimension,
  the key, the storage entry and the keys that entry has.

The order of the entries is free. Keys match by name, not by folder depth, so
two sources may nest the keys in different orders (case C below).

For `sales` with `order_year: year` and the entry
`{ type: range, dimension: order_year }`:

| Case | `paths:` and the folders found | Result |
|---|---|---|
| A | `orders/`, with `region=eu/` and `region=us/` | compile error: `orders/` has keys `[region]` |
| B | `[eu/orders/, us/orders/]`; `eu` has `year=` folders, `us` has `region=` folders | compile error for `us/orders/` only |
| C | `[eu/orders/, us/orders/]`; `eu` has `year=/month=`, `us` has `month=/year=` | allowed: the key is found by name, not by depth |
| D | `orders/`, with `year=2026/month=01/` | allowed: the value is `2026` |
| E | as D, and the files also hold a column `year` | allowed: `order_year` reads the folder value, and the file's `year` is not read |

### Compile infers the link

On a semantic source that reads by `Prefix` a physical source with partitions,
a Dimension that maps a bare column equal to a discovered key, without regard to
ASCII case, is **the key's Dimension**, with an entry or without one. The entry,
or this inference, is the **link** between the key and its Dimension. With no
entry and no discovered key, a column mapping reads a file column, and so does
every column mapping on a semantic source that reads by `Pattern` or `Uris`.

Each discovered value of a key must cast to the declared type of each Dimension
linked to that key, whether an entry names it or compile infers it. A value that
does not cast is a compile error: ADR 0070's "A failed cast is a compile error"
stands.

### The manifest stores no link

The link is compile's to check, and compile stores nothing for it. ADR 0070's
`partition_def`, with each partition's `key_values`, stays the only record of
partitions. A later planner that plans partition by partition can find the link
from what is stored: on a semantic source that reads by `Prefix`, a signature
that is a bare `Column` whose name is a discovered key of its physical source,
without regard to ASCII case.

### The consumer reads `__HIVE_DEFAULT_PARTITION__` as NULL

ADR 0070 rules that this value is NULL. The plan writes nothing for it. The
consumer reads the Hive layout of a folder read (ADR 0073):

- it gives each key under the name of the read's column, with the
  percent-decoded value of the folder (ADR 0070, *The listing and its
  partitions*) cast to the read's type;
- it gives `__HIVE_DEFAULT_PARTITION__` as NULL;
- it reads the files with a schema that leaves out the keys, so that the folder
  value wins.

Spark does this. DataFusion 55 keeps the text `__HIVE_DEFAULT_PARTITION__`, and
its partition fields are never NULL.

### Amends

- **0070:** in its *Decision Outcome* lead, "a partition key feeds a metadata
  Dimension" reads as 0081's *A partition key is a column*; in *A partition key
  feeds a metadata Dimension*, the form `metadata: { partition: { column } }`
  and the bullet "A file column with the key's name does not clash with it" are
  replaced by 0081's *A partition key is a column*, which answers "How a
  Dataset's signature marks a partition-fed Dimension"; "constant **per source
  or per partition**" no longer holds, since a metadata value is constant per
  source; "The form is an error on a source it does not apply to" applies to a
  `partition_def` entry (0081's *`partition_def` names the Dimension*); "only
  through a metadata Dimension the author declares" reads "only through a
  mapping the author writes that names the key"; the case table is 0081's; the
  bullet "The value is the partition's value" stands; in *The model's partitions
  are checked, never copied*, the model's `partition_def` and its column read as
  0081's *`partition_def` names the Dimension* gives; in *Consequences*,
  "`MetadataSource::Partition` gains its compile arm" no longer holds; in
  *Amends*, the entry for ADR 0031 is replaced by 0081's. The rest of 0070
  stands.
- **0031:** the `partition` form is not a metadata form, and a metadata value is
  constant per source; ADR 0070's clause "the `partition` form is admitted" no
  longer holds. `path_token` and the rest of 0031 stand.
- **0073:** in *The physical read below a Dataset*, the consumer that "reads the
  files in the `key=value` folders under `uri_folder`" also does what 0081's
  *The consumer reads `__HIVE_DEFAULT_PARTITION__` as NULL* lists; in
  *Consequences*, "The planner needs a partition-fed Dimension to be readable as
  a column, named by its key" is met as 0081's *A partition key is a column*
  rules. The rest of 0073 stands.

### Consequences

- Good, because a partition key reads like any other column, as it does in
  every engine compared, and it can feed a temporal Dimension.
- Good, because the planner needs no rule of its own for a partition column.
- Bad, because on a semantic source that reads by `Prefix`, a file column with a
  key's name cannot be read.
- Bad, because a folder read relies more on the consumer. DataFusion 55's
  listing reads no file when its partition columns are not a leading prefix of
  the folder keys in folder order, and it matches a key's case exactly. So a
  DataFusion consumer finds every key in the folders, matches each one to the
  read's column without regard to ASCII case, gives
  `__HIVE_DEFAULT_PARTITION__` as NULL, and reads the files with a schema that
  leaves out the keys.
- Bad, because compile checks the casts of the values it discovers, and the
  engine lists the folders again (ADR 0073). On DataFusion 55, a value that the
  engine finds after compile and that does not cast fails the read.
- `MetadataSource::Partition` leaves the semantic model, so a semantic model
  that writes `type: { metadata: { source: { partition: … } } }` (ADR 0070's
  `metadata: { partition: { column } }`) fails at parse. The `metadata`
  Dimension type keeps the `path` form only.
- The semantic model's `partition_def` changes from one entry
  `{ type, column, values? }` to a list of entries `{ type, dimension,
  values? }`.

## More Information

The research compared DataFusion 55.0.0, Spark v3.5.3, Hive 3.1.3 and 4.0.0,
Iceberg 1.7.0 and Substrait v0.103.1, with a quote and a source for each fact.
