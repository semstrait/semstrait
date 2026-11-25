---
number: 70
title: A path source is one base path read in one format
status: accepted
date: 2026-09-23
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0031-metadata-as-compile-stratum.md
- docs/adr/0036-bit-position-assignment.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0051-entity-ids-are-derived-from-the-natural-key.md
- docs/adr/0058-a-reduction-carries-the-authors-keysets-to-the-public-face.md
- docs/adr/0061-a-source-schema-may-be-partial.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0067-the-scan-carries-its-read-identity.md
- docs/adr/0068-adapters-are-layered-format-first.md
- discovery rules checked against Spark v3.5.3 (`shouldFilterOutPathName` in https://github.com/apache/spark/blob/v3.5.3/core/src/main/scala/org/apache/spark/util/HadoopFSUtils.scala, and https://github.com/apache/spark/blob/v3.5.3/sql/core/src/main/scala/org/apache/spark/sql/execution/datasources/PartitioningUtils.scala) and DataFusion 55.0.0 (https://github.com/apache/datafusion/blob/55.0.0/datafusion/catalog-listing/src/helpers.rs)
- replaces the unit of read proposed
- sharpened in place (no object is selected by two of a Dataset's sources)
links:
- target: 31
  kind: amends
- target: 45
  kind: amends
- target: 46
  kind: amends
- target: 49
  kind: amends
- target: 54
  kind: amends
- target: 58
  kind: amends
- target: 61
  kind: amends
- target: 67
  kind: amends
- target: 73
  kind: amendedby
- target: 33
  kind: amends
- target: 78
  kind: amendedby
- target: 81
  kind: amendedby
---

# A path source is one base path read in one format

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): "compile's listing is the only discovery" holds for reads without partitions. For discovered partitions, the engine discovers them and supplies the key column, and the planner writes no literal per partition.
> - [ADR 0078](0078-a-source-alias-is-its-source-roles-letter-and-its-index-in-the-manifest.md): in *A reduction leaf is minted from its Dataset and its physical source*, "**Engine aliases stay unique.**" holds by 0078's *An alias is `ss`, its source role's letter and its index*, not because "`mint_alias` (`semantic_source.rs`) adds an id suffix to an alias whose name collides". The rest of 0070 stands.
> - [ADR 0081](0081-a-partition-key-is-a-column-and-partition-def-names-the-dimension-that-maps-it.md): in its *Decision Outcome* lead, "a partition key feeds a metadata Dimension" reads as 0081's *A partition key is a column*; in *A partition key feeds a metadata Dimension*, the form `metadata: { partition: { column } }` and the bullet "A file column with the key's name does not clash with it" are replaced by 0081's *A partition key is a column*, which answers "How a Dataset's signature marks a partition-fed Dimension"; "constant **per source or per partition**" no longer holds, since a metadata value is constant per source; "The form is an error on a source it does not apply to" applies to a `partition_def` entry (0081's *`partition_def` names the Dimension*); "only through a metadata Dimension the author declares" reads "only through a mapping the author writes that names the key"; the case table is 0081's; the bullet "The value is the partition's value" stands; in *The model's partitions are checked, never copied*, the model's `partition_def` and its column read as 0081's *`partition_def` names the Dimension* gives; in *Consequences*, "`MetadataSource::Partition` gains its compile arm" no longer holds; in *Amends*, the entry for ADR 0031 is replaced by 0081's. The rest of 0070 stands.

## Context and Problem Statement

**This reverses an earlier ruling.** On 2026-09-17 the deciding human ruled
that "`PhysicalSource` keeps its shape, compile's expansion rules stand". The
same human reversed it for path sources while reviewing it, on the grounds
that the expansion rules were never precise. A planner
decision that needs a new manifest carrier is a handoff, so the carrier is
still to build.

## Considered Options

- One physical source per base path and format, read through a selector on
  each semantic source, with the rulings in Decision Outcome (chosen).
- **One physical source per matched file**, as `compile/catalog.rs`'s
  `expand_pattern` reads the earlier base-schema design today. Refused: no engine reads a
  file table that way, a per-file source cannot carry the partition values its
  files omit, and a partition layout matches nothing.
- **The first proposed unit:** identity minted from the pattern alone, contents
  `Flat | Partitioned`, and the name as the handle an engine registers.
  Refused:
  - registering under the name contradicts ADR 0068;
  - no glossary term supports `contents`;
  - an id from the pattern alone merges two formats on one path.
- **The selector in the id**, `mint(base_path, selector, format)`, ruled on
  2026-09-28 and reversed on 2026-09-29. Refused: the selector says what one
  semantic source reads, not what is stored, and in the id it splits one folder
  into one physical source for each way of reading it.
- **The Dataset's id in a physical source's id or name.** Refused: it reverses
  ADR 0049's deliberate `1:N` edge, and the Datasets that read a source are
  already known from that edge. Only the reduction leaf, which is semantic,
  carries its Dataset.
- **Full discovery**, listing the whole base path for every semantic source.
  Refused: a `Uris` source over `s3://lake/events.parquet` would list the whole
  bucket.
- **A sequential merge**, each semantic source updating the physical source in
  turn. Refused: the result would depend on the order compile meets them.
- **The extension decides the format.** Refused: the format is authored, and an
  extension is a naming habit, not a contract.
- **A name grown from parent segments until unique**, and a one-URI source
  named from its file. Refused: such a name changes when an unrelated source is
  added, and a label need not be unique.
- **A leaf's name as its natural key**, made unique by ADR 0058's varying
  segment. Refused: segment joining is ambiguous (`x_y/` and `x/y/`), no
  segment carries a scheme, and a name-seeded id then merges two leaves.
- **An ordered list of named objects.** Refused: the order changes nothing that
  is read, and two orders would give one selection two forms.
- **Optional location fields** (`file_path`, `file_paths`, `base_path`,
  `base_path_glob`), the first sketch. Refused for a sum type, so that
  a combination that means nothing cannot be written.
- **A nested partition tree**, key → values → listing. Refused: a Hive folder
  path is one `key=value` sequence, and one flat entry per partition matches it
  directly.
- **DataFusion's layout rule**, which passes over an object outside the layout
  without an error. Refused: the listing is what compile saw, so an object that
  is present and not read must be an error.
- **Exact key match**, as DataFusion does. Refused: the model's column and a
  folder written by a case-insensitive tool may differ in case, and the
  two-spellings error already catches the one case that exact matching guarded
  against.
- **Unicode case folding.** Refused: its special cases (`ß`, the Turkish `İ`
  and `ı`) differ between implementations.
- **`__HIVE_DEFAULT_PARTITION__` as plain text**, as DataFusion reads it.
  Refused: data written by Spark or Hive would then fail its cast.
- **A relative path kept as written.** Refused: it breaks "every listing item
  is a full URI", and compile and the engine would each resolve it under their
  own root, so they could read different objects.
- **A relative path resolved under compile's working directory**, as
  DataFusion does. Refused: the id would then depend on the machine that
  compiles.
- **A relative path resolved under a root the manifest stores.** Deferred: no model or compile field holds such a root today, and no
  model file writes a relative path.
- **A name made unique by an id suffix.** Refused: the id already identifies
  the source, and a suffix gives nobody a readable name.
- **`schema_version` beside `schema_id`.** Refused: two fields for one fact,
  against ADR 0046's own rule.
- **Copying the model's partition definition into the manifest.** Refused:
  discovery is what is read, and the model's claim is checked against it.
- **Two same-base-path sources of one Dataset kept apart by a selector index
  in the leaf name.** Refused: it names the double read and does not prevent
  it.
- **Table globs in the MVP.** Deferred.
- **Formats other than Parquet files and Iceberg tables in the MVP.** Deferred.
- **Default metadata Dimensions in this ADR.** Deferred, for the reasons
  below.

## Decision Outcome

A physical source over files is its **base path** (the deepest literal folder
it reads) read in one format. Each semantic source over it reads some of its
objects through a **selector**. Compile folds every semantic source over one
physical source into one listing. Under a bare folder, a Hive `key=value`
layout is recorded as flat partitions, and a partition key feeds a metadata
Dimension.

The ids come from what is stored, and the names are labels. No physical
source's id or name carries a Dataset. Only a reduction leaf, which is
semantic, carries its Dataset.

This replaces ADR 0046's "one `PhysicalSource` per expanded path" for path
sources, and the pattern-keyed unit proposed first. A table source keeps its
id.

### The shape

```
PhysicalSource {
  id: EntityId,                        // path: mint(base_path, format)
  name: String,                        // a label: the base path's last segment
  storage: StorageProperties {         // ADR 0046's
    target: Path {
      base_path: Uri,                  // normal form, ends in one `/`
      listings: [Uri],                 // every object a semantic source selects, sorted by URI
      partition_def: Option<{
        partitions: [{                 // sorted by partition folder
          key_values: [KeyValue],      // ordered by folder depth
          listing: [Uri],              // sorted by URI
        }],
      }>,
    } | Table { … },                   // unchanged; id stays mint(source_type, locator)
    format: Option<FileFormat>,        // a path source: `parquet` only in the MVP
  },
  schema_provenance: CatalogDerived | ModelFed,  // unchanged (ADR 0046, 0061)
  schema_id: SchemaId,                 // unchanged (ADR 0046)
}

RoleDefinition::Dataset {              // the reference ADR 0049 puts on a leaf
  source_id: EntityId,                 // unchanged
  selector: Prefix | Pattern(Glob) | Uris([Uri]) | Table,  // Uris: a set, sorted
}
```

The field names are ruled. The implementation names the types, with two limits.
The type of `partition_def` is not called `PartitionDef`, because that is the
model's authoring type. No field is called `contents`, because no glossary term
supports that name.

Every list in the shape has a fixed order, so the manifest's bytes do not change
when a file system returns a folder in a different order.

### Every URI is in normal form

Compile writes every URI in one normal form before it mints an id or lists a
folder. The manifest stores only that form: the base path, the selector's glob
and URIs, and the listings.

- **`s3a://` and `s3n://` become `s3://`**, and `file://` stays. They name one
  store, so they must give one id. Any other scheme is a compile error in the
  MVP. The engine registers one driver for each normal scheme.
- **An absolute local path gets `file://`. A relative path is a compile error
  in the MVP**, and the error names the storage entry and the path. Compile
  refuses it before it lists anything. The reason: compile would list a
  relative path under the root of the file system it is given, and the
  engine would read each object by its URI under its own root (ADR 0068). No
  root is stored, so the two could read different objects. An absolute path
  is the same text on every machine, so an id still does not depend on the
  machine that compiles.
- **The bucket (the URI's authority) stays the first segment**, a folder URI
  ends in exactly one `/`, and case is kept. S3 keys are case-sensitive.

A path source's id cannot equal a table source's id. A normal base path ends in
`/`, and a table's `source_type` does not, so the serialized tuples
(`identity.rs`'s `seed_bytes`) differ in their first part.

### A path source is one base path read in one format

A wildcard in a **folder** position multiplies sources: there is one physical
source for each folder the wildcard matches, and listing goes no deeper (the earlier
base-schema design). Everything else in an authored path is the selector: it says
which of the base path's objects a semantic source reads.

The examples in this ADR leave out the scheme and the bucket: `orders/` stands
for a full URI such as `s3://lake/orders/`.

| Authored `paths:` | Physical sources | Base path | Selector |
|---|---|---|---|
| `orders/` | one | `orders/` | `Prefix` |
| `orders/*.parquet` | one | `orders/` | `Pattern(orders/*.parquet)` |
| `orders/a.parquet` | one | `orders/` | `Uris([orders/a.parquet])` |
| `lake/*/orders/*.parquet` | one for each matched folder | `lake/eu/orders/`, `lake/us/orders/` | `Pattern(lake/eu/orders/*.parquet)`, … |
| `[orders/a.parquet, orders/b.parquet]` | one | `orders/` | `Uris([a, b])` |
| `[eu/a.parquet, us/b.parquet]` | two | `eu/`, `us/` | one `Uris` each |

The first three rows are **one** physical source. Datasets X over
`s3://lake/orders/`, Y over `s3://lake/orders/*.parquet` and Z over
`s3://lake/orders/a.parquet`, all in Parquet, read
`mint(s3://lake/orders/, parquet)` through three selectors.

- **The selector sits on the semantic source**, in the physical-source
  reference ADR 0049 gives a leaf. It is on the Dataset when the Dataset reads
  one physical source, and on each reduction leaf when it reads several. A
  semantic source reads the objects of the physical source's listing that its
  selector selects.
- **`Table` pairs only with a table source, and the other three only with a
  path source.** Compile builds no other pair.
- **A `Pattern` holds the glob rewritten for its base path.** A consumer that
  reads globs can then pass it to the engine as it is (ADR 0067's
  `FilesScan.pattern`). The glob is kept so the engine can read it; the listing
  is still what compile saw.
- **Explicit objects under one base path, in one `paths:` list, are one `Uris`
  selection.** Each glob is its own selection.
- **A `Uris` selector is a set.** Compile sorts it by URI, so `[a, b]` and
  `[b, a]` are one selection. A URI written twice in one `paths:` list is a
  compile error; nothing drops the repeat silently.
- **Only a `Prefix` selection discovers partitions.** A `Pattern` matches
  objects directly under its base path, and a `Uris` names its objects. A
  folder wildcard over Hive folders, such as `orders/*/*.parquet`, therefore
  gives one physical source per partition folder, with no partitions on any of
  them. To read `orders/` as one partitioned source, write `orders/`.

### Identity is the base path and the format

A path source's `id = mint(base_path, format)`, over the normal base path.

- **The selector is not in the id.** It says what one semantic source reads,
  not what is stored. In the id, it would split one folder into one physical
  source for each way of reading it.
- **The format is in the id, and it is authored.** One folder read as Parquet
  and read as CSV would be two different reads, so two sources, and this is not
  an error. A file's extension decides nothing, as in Spark, where the reader
  names the format and a `.dat` file under a Parquet path is read as Parquet.
- **The format's read options** (ADR 0067's `format_options`) join the id when
  the manifest first records them.

**A table source's id stays `mint(source_type, locator)`.** The catalog
provider renders the locator. A provider with a realm (Polaris) puts it in:
`catalog.realm.namespace.table`. A provider without one leaves it out.

ADR 0049's `1:N` reverse edge stands, and it is now the usual case: every
semantic source over one base path in one format shares one physical source,
whatever its selector.

### Compile folds the semantic sources of one physical source

Compile groups the semantic sources by physical-source id and builds each
physical source once, from all of them. This replaces "first write wins"
(`compile/catalog.rs`), which was correct only while the id covered everything
that makes two reads differ.

- **The listing is every object that any of them selects.**
- **A model-fed source's column set is every column that any of them
  references**, as ADR 0046 (amended by 0061) already derives it.
- **The source has partitions only if one of them reads by `Prefix`.**
- **Layout checks and partition discovery run on the `Prefix` selection
  alone.** A `Uris` source that names `_meta.parquet`, which discovery skips,
  therefore cannot break a `Prefix` source over the same folder. One semantic
  source never breaks another.

For X, Y and Z above, the listing is every object under `s3://lake/orders/`
after the skips, because X's `Prefix` selects them all. Y and Z read the part of
it that their selectors select.

### The name is a label

**A path source's name is its base path's last segment, always**:
`s3://lake/eu/orders/` is `orders`, and `s3://lake/` is `lake`, because the
bucket is the first segment. No segment is added and nothing is joined. A base
path with no segment (`file:///`) takes its scheme and authority as written,
`file://`. A table source's name is the table's name.

**A name may repeat.** `lake/eu/orders/` and `lake/us/orders/` are both
`orders`. So a diagnostic shows the base path and the format beside the name.

The name is **not** the natural key. This departs on purpose from the **Entity
id** entry, where a name is what every reference resolves. A physical source is
never authored by name and never referenced by name: its id comes from what is
stored (above). No engine registers a table under it: ADR 0068 still refuses
"registering a listing table under the physical source's name". It exists for
diagnostics, trace and explain text. This resolves ADR 0058's open question on the
`name` field.

### The listing and its partitions

**Compile's listing is the only discovery.** Under ADR 0068 the engine reads
each object by its URI and never lists a folder, so no engine's own discovery
runs on a semstrait read. Compile keeps what Spark and DataFusion agree on.
Where they differ, the rules below say which one it follows.

- **Every listing item is a full URI in normal form.** `listings` holds every
  object that a semantic source over the physical source selects.
- **Discovery skips four kinds of name under a `Prefix` or a `Pattern`
  selection:**
  - a name that starts with `.`;
  - a name that starts with `_` and holds no `=` (so `_region=x/` is still a
    partition);
  - a name that ends with `._COPYING_`, an object still being copied;
  - an object of 0 bytes, which no format reads.

  The first three are Spark's name rule, and the last is DataFusion's. A
  skipped object or folder (`_SUCCESS`, `.part-0.crc`, `_temporary/`) is not
  selected, and it does not count toward the layout. Spark keeps `_metadata` and
  `_common_metadata` for Parquet's summary, but no semstrait read uses them, so
  they are skipped too. A `Uris` selection names its objects, so nothing in it
  is skipped.
- **Two layouts are admitted under a `Prefix` selection:** objects directly
  under the base path (no partitions), or Hive `key=value` folders
  (partitions). An empty folder is no partition. Any other layout is a compile
  error. An object that is present but not read is never silent, even where an
  engine would pass over it:

  | Layout under the base path | Spark | DataFusion | Here |
  |---|---|---|---|
  | a folder without `=` beside partition folders | error | not read | compile error |
  | objects beside partition folders | not read | not read | compile error |
  | partitions whose key sequences differ | error | not read | compile error |
  | a folder without `=` below the last key | read | not read | compile error |

  A later decision may admit more layouts.
- **A `key=value` folder is split at the first `=`, and both sides are
  percent-decoded**, so `city=S%C3%A3o%20Paulo` holds `São Paulo`. Both engines
  split there and decode the value. Spark also decodes the key. An empty key or
  an empty value (`=2026/`, `year=/`) is a compile error, as in Spark. The value
  `__HIVE_DEFAULT_PARTITION__` is NULL: Spark and Hive write it for a NULL
  value, and Spark reads it back as NULL.
- **Keys match without regard to ASCII case**, as in Spark. DataFusion matches
  exactly. Only `A`–`Z` and `a`–`z` fold, and every other character must match
  exactly, so no locale or Unicode case table enters the rule. Within one
  source, two spellings of one key (`Year=` in one partition, `year=` in
  another) are a compile error. So is one key twice in one folder path
  (`year=2026/YEAR=01/`). `key_values` records the spelling found. Two sources
  of one Dataset may spell a key differently.
- **Partitions are flat, one entry per partition,** sorted by partition folder
  as found. `key_values` are in folder order, and every partition has the same
  key sequence. A value is its decoded text, or NULL. The partition folder is
  not stored. It is the parent folder of every object in the partition's
  listing, because the layout rules put each object directly in its
  partition folder. It keeps the text as found, so it is never rebuilt from
  `key_values`: `S%C3%A3o%20Paulo` and `São%20Paulo` decode to one value.
- **Under a Hive layout, each object of the `Prefix` selection is in exactly
  one partition, and the partition listings together equal that selection.**
  Compile checks both. Under direct objects, `partition_def` is `None` and the
  rule does not apply. An object that only a `Uris` selection names, such as a
  skipped `_meta.parquet`, is in `listings` and in no partition.

### The model's partitions are checked, never copied

The model's `partition_def` is an authoring claim. Compile checks it against
what discovery found, and the manifest records what was discovered. The
model's column matches a discovered key without regard to ASCII case, and a
`List` value matches a discovered value's decoded text exactly:

| Case | Result |
|---|---|
| The model's partition column is not a discovered key | compile error |
| A discovered value is not in the model's `List` values | compile error |
| A NULL partition under a `List` declaration | compile error, unless `values` holds `__HIVE_DEFAULT_PARTITION__` |
| The model declares a partition, and no `key=value` folders exist | compile error |
| The model declares no partition | the discovered layout stands |
| A declared `List` value has no folder | not an error |

A `Range` declaration names no values, so only the column rows apply to it: an
undiscovered column, and a declaration with no `key=value` folders. The value
rows are for `List` only.

### A partition key feeds a metadata Dimension

A partition key is a metadata Dimension's source, as in Spark, where a partition
column comes from the layout and not from the files.

- **The author form is the one the model already reserves:**
  `metadata: { partition: { column } }` (`MetadataSource::Partition`). Compile
  refuses it today with `MetadataExtractionUnsupported`, and this ADR admits it.
  The Dimension names its key through `column`, so a Dimension `order_year` can
  read the key `year`. `column` matches the key without regard to ASCII case,
  so `column: year` reads `Year=` folders.
- **The value is the partition's value**: the decoded text after the first
  `=`, cast to the Dimension's declared type. A NULL partition gives NULL. A
  failed cast is a compile error, as ADR 0031 rules for every metadata value.
  `path_token` still returns the whole raw segment. ADR 0031's rejection of
  "splitting on `=` inside the mechanic" stands for `path_token`; the partition
  form is a different mechanic.
- **The value is stored only in the partition's `key_values`.** The planner
  writes it as the literal on each partition branch (ADR 0067). A metadata value
  is therefore constant **per source or per partition**, where ADR 0031 said
  per source. How a Dataset's signature marks a partition-fed Dimension is
  a carrier still to build, under 0031's rule that metadata is never a
  `PhysicalExpr` variant.
- **The form is an error on a source it does not apply to**, by ADR 0031's
  existing rule: a table source (its partitions come from transforms such as
  `day(ts)`, not from folders), a semantic source that does not read by
  `Prefix` or whose physical source has no partitions, or one whose partitions
  lack the key. The error is raised for each of the Dataset's sources that
  lacks the key. It names the Dimension, the key, the storage entry and the
  keys that entry has. Nothing fills NULL for that source: ADR 0049 rules out
  null fill.
- **A file column with the key's name does not clash with it.** A metadata
  Dimension reads no column, so ADR 0061's model-fed derivation never adds a
  column for it. The author picks: a column Dimension reads the file column, and
  a metadata Dimension reads the folder.

For `sales` with `order_year: { metadata: { partition: { column: year } } }`:

| Case | `paths:` and the folders found | Result |
|---|---|---|
| A | `orders/`, with `region=eu/` and `region=us/` | compile error: `orders/` has keys `[region]` |
| B | `[eu/orders/, us/orders/]`; `eu` has `year=` folders, `us` has `region=` folders | compile error for `us/orders/` only |
| C | `[eu/orders/, us/orders/]`; `eu` has `year=/month=`, `us` has `month=/year=` | allowed: the key is found by name, not by depth |
| D | `orders/`, with `year=2026/month=01/` | allowed: the value is `2026` |
| E | as D, and the files also hold a column `year` | allowed: `order_year` reads the folder, never the column |

**A partition value reaches a query only through a metadata Dimension the
author declares.** Default metadata Dimensions (one per partition key, one for
the source's name) are not ruled here. They would add semantics that no author
wrote, and they would move bit positions (ADR 0036) and signatures (ADR 0065).
They are for a later decision.

### A table source is read whole

- **A table's selector is `Table`**: the semantic source reads the whole table.
- **In the MVP a `tables:` entry names one table.** A glob in it is a compile
  error that names the entry. Today the glob reaches the catalog as a table
  name and fails with `CatalogSourceMissing`, which tells the author that a
  table is missing when they wrote a pattern. Table globs are deferred.

### Parquet files and Iceberg tables only, in the MVP

A path source's format must be `parquet`, and a table source is an Iceberg
table. Any other format is a compile error that names the storage entry and the
format, so it never reaches an engine. Other formats are deferred. The
format stays in the id, so admitting one later changes no id.

### One Dataset reads each object once

**The sources of one Dataset have distinct base paths, and no object is selected
by two of them.**
One Dataset has one format, so distinct base paths are distinct physical
sources. Explicit objects under one base path are one `Uris` selection (above).
Any other two entries on one base path are a compile error, and so are two
entries that select one object. The error names the Dataset and both storage
entries. ADR 0049's implicit Unionset is `mode: All`, so an object read by two
of a Dataset's sources would be counted twice, silently. Examples are
`orders/*.parquet` beside `orders/2026-*.parquet`, `orders/` beside
`orders/a.parquet`, and `orders/` beside `orders/year=2026/`.

Two Datasets may read the same objects. That is the fold above, not a double
count.

### A reduction leaf is minted from its Dataset and its physical source

A reduction leaf exists only when one Dataset reads several physical sources:
several tables, several base paths, or a folder wildcard that matches several
folders.

- **`id = mint(Dataset id, physical-source id)`.** It is unique by construction:
  a Dataset reads each physical source once (above), and a Dataset id is
  unique. Nothing about the leaf's name enters it.
- **Its name is `<Dataset name>_<physical-source name>`, a label that may
  repeat.** `sales` over `lake/*/orders/*.parquet` has two leaves named
  `sales_orders`. A diagnostic shows the base path and the format beside the
  name, as for a physical source.
- **Engine aliases stay unique.** `mint_alias` (`semantic_source.rs`) adds an
  id suffix to an alias whose name collides.
- **`paths: [2024/*.parquet]` is one source and no reduction**, so its Dataset
  keeps the authored name. ADR 0058 gave that example as two leaves.

A name-derived leaf id would merge two leaves in cases a Dataset can hold. Under
segment joining, `s3://lake/x_y/` and `s3://lake/x/y/` both gave `x_y`. And
`file:///lake/orders/` beside `s3://lake/orders/` differ only in the scheme,
which no segment carries. Their physical-source ids differ, so their leaf ids
do too.

This amends ADR 0045's natural key ("a name is unique, and the name is the id
seed") and ADR 0058's refusal of "two entities with one name", **for private
leaves only**. Nobody writes a leaf's name and nobody references it, so its id
need not come from it. ADR 0058's varying segment, and its segment joining,
retire.

### Amends

Each is recorded on that ADR's Status line. No body is edited, so the bound
on in-place edits is not engaged.

- **0045:** a private reduction leaf's id is minted from its Dataset and its
  physical source, not from its name, and its name is a label that may repeat;
  the natural key stands for every other entry.
- **0046:** for path sources, "one `PhysicalSource` per expanded path, not per
  glob root" is replaced by one source per base path and format. A
  folder-position wildcard still multiplies sources. Expansion stays at compile,
  and the narrowing of compile determinism stands. Compile folds every semantic source over one
  physical source into its listing and, for a model-fed source, its column set,
  where it kept the first write.
- **0049:** a path source's id is minted from `(base_path, format)`, not
  `(source_type, locator)`. A table source's id is unchanged. The single
  physical-source reference carries the selector
  (`Prefix | Pattern | Uris | Table`). The sources of one Dataset have distinct
  base paths, and no object is selected by two of them. The `1:N` reverse edge
  stands.
- **0054:** in the `N > 1` consequence, a reduction leaf's id is
  `mint(Dataset id, physical-source id)`, not seeded by its name, and a glob
  multiplies physical sources only where a folder wildcard matches several
  folders.
- **0058:** a leaf's id is `mint(Dataset id, physical-source id)`, and its name
  `<Dataset name>_<physical-source name>` is a label that may repeat. The
  varying segment, its segment joining and the refusal of two leaves with one
  name retire. Its open question "whether a glob is one physical source or several" is
  resolved as one source per base path and format. Its open question on the `name` field
  is resolved as a label.
- **0061:** its restatement "one `PhysicalSource` per expanded path" reads as
  0070's unit: one source per base path and format.
- **0031:** the `partition` form is admitted, the form that the model reserves:
  its value is the percent-decoded text after the first `=` of a discovered
  `key=value` folder, and a metadata value is constant per source or per
  partition; `path_token` still returns the raw segment.
- **0067:** its unit of read is 0070's, not the first proposal's: `FilesScan.paths` is the
  semantic source's selection from the listing; `FilesScan.pattern` is the
  semantic source's `Pattern` glob, and `None` for a `Prefix` selection, a
  `Uris` selection and a partition branch; the partition branches are 0070's
  `partition_def.partitions`; each branch's `base_path` is its partition's
  folder as found (the parent folder of the objects in its listing, never
  rebuilt from `key_values`); and each branch's literals are the partition's
  values, cast to the metadata Dimension's type.
- **0033:** every semantic source over one base path in one format shares one
  physical source, whatever its selector; a `Pattern` holds the glob rewritten
  for its base path, so a consumer that reads globs can pass it to the engine as
  it is (ADR 0067's `FilesScan.pattern`); a folder-position wildcard still
  multiplies sources, one physical source for each folder the wildcard matches.

**ADR 0068:** unchanged. The refusal of registration under the name stands.

### Consequences

**For the manifest effort (a handoff), not done here.**

- `StorageTarget::File { location }` becomes the path target above, and
  `RoleDefinition::Dataset` gains `selector`.
- Compile writes every URI in normal form, and refuses any other scheme and
  any relative path.
- `expand_pattern`'s per-file matching becomes one source per base path.
- `mint_source_id`'s preimage becomes `(base_path, format)` for a path source.
  A table source keeps `(source_type, locator)`, with the realm in the locator
  where the provider has one.
- The first-write dedupe becomes the fold, and its comment is updated.
- Layout checks and partition discovery run on the `Prefix` selection.
- A reduction leaf's id becomes `mint(Dataset id, physical-source id)`, and the
  varying segment retires.
- The Parquet and Iceberg format checks and the table-glob error are added.
- `MetadataSource::Partition` gains its compile arm.
- Discovery gains its skip, split, decode and key-matching rules.
- The `Uris` repeat check, the distinct-base-path check and the listing checks
  are added.
- `MANIFEST_VERSION` advances.

The model's `PartitionDef` stays authoring surface.

## More Information

Decided 2026-09-23; refined 2026-09-28;
refined again 2026-09-29 (the id drops the selector, the
selector moves to the semantic source, compile folds the semantic sources of
one physical source, and a reduction leaf's id is minted from its Dataset and
its physical source; from Copilot's review, a relative path is a compile error
in the MVP, and a base path with no segment is named by its scheme and
authority).
