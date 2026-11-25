---
number: 67
title: The scan carries its read identity
status: accepted
date: 2026-09-23
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0018-canonical-plan-ir-contract.md
- docs/adr/0019-substrait-outbound-only.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- Substrait `proto/substrait/algebra.proto` at tag v0.103.1 (`ReadRel`)
- 'evidence: DataFusion resolves a table by `NamedTable.names` alone; the round-trip prototype had to treat the opaque handle as a table name to run at all'
links:
- target: 70
  kind: amendedby
- target: 73
  kind: amendedby
- target: 18
  kind: supersedes
---

# The scan carries its read identity

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): its unit of read is 0070's, not the amendment's: `FilesScan.paths` is the semantic source's selection from the listing; `FilesScan.pattern` is the semantic source's `Pattern` glob, and `None` for a `Prefix` selection, a `Uris` selection and a partition branch; the partition branches are 0070's `partition_def.partitions`; each branch's `base_path` is its partition's folder as found (the parent folder of the objects in its listing, never rebuilt from `key_values`); and each branch's literals are the partition's values, cast to the metadata Dimension's type.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): a read with discovered partitions is one `FilesScan` over its base path with no pattern and no paths; the engine discovers and skips partitions and supplies the key columns; the Partitions section's branches are not planned, and `paths` is the truth only for a read without partitions; a read without partitions that selects no object is the empty read, not a `FilesScan`.

## Context and Problem Statement

Under ADR 0018 a `ScanNode` held only an opaque `SourceRef`, resolved
adapter-side against the manifest. Three layers touch a read; this ADR rules
the middle one.

## Considered Options

- The scan carries its read information (chosen).
- **A `SourceBinding` trait handed to the producer**, implemented for
  `SemanticManifest` inside the adapter crate. Refused: the adapter becomes a
  function of two inputs by another name.
- **A pointer the adapter looks up** — a manifest id, or `SourceRef(String)`
  with a dotted-FQN encoding. Refused: the adapter reads the plan only, and a
  string the adapter must split is a contract hidden in a convention.
- **A read vocabulary with no Substrait image** — `ReadSource::{Table, Path,
  Iceberg, Virtual}` with `FileItem`, `Listing` and a format roster of the
  plan's own. Refused: under the mirror, a reader is a Substrait read type.
- **A semstrait `IcebergTable { name }`.** Refused: no Substrait release has a
  named Iceberg read, so every adapter would rewrite it to `NamedTable`.
- **`uri_folder { base_path, listing }`** as one item. Refused: a Substrait
  item is one string of one kind; the base path and the listing are two fields
  beside it.
- **The work-split fields on `FilesScan`.** Refused: they split work, not
  data, and number nothing without a catalog that indexes partitions.
- **`schema` as the physical source's complete column set.** Refused: a
  model-fed source's column types are optional, so the full set breaks ADR 0066
  ruling 5; the columns the scan reads always have a type.
- **A snapshot pin, or a metadata-file address, on the scan.** Refused: the
  plan needs the table's identity, not its version.

## Decision Outcome

A `ScanNode` names what it reads, completely, so that a format adapter reads
the plan and nothing else (ADR 0068). The planner copies the read information
from the manifest's physical source into the scan; the scan never points back
at the manifest.

### Scope

- **Compile owns the unit of read** — ADR 0046, as amended. A path
  source is built around its base path and holds the pattern as authored, a
  name minted from it, its format, and its contents: the objects listed under
  the base path, flat or per discovered partition. A table source holds its
  catalog identity and the format the catalog reported. Compile lists the
  storage (ADR 0003); the planner lists nothing and reparses nothing.
- **The scan carries the unit's read information** — this ADR.
- **The rules make it consumable** — ADR 0068: how each reader is written on
  the wire, and what a given consumer can read.

### The scan

`ScanNode` mirrors `ReadRel` (ADR 0066 ruling 1):

```
ScanNode {
  common,                          // ADR 0066 ruling 3
  schema,                          // base_schema
  filter: Option<PhysicalExpr>,    // filter
  reader,                          // read_type
}
```

- **`schema`** holds the columns the scan reads, each named and typed (ADR
  0066 ruling 5), a column only the filter uses among them. `common.output_schema`
  holds what the scan outputs, and `common.emit_form` selects it from `schema`.
  The read set and the output stay two lists. Scenario: the filter is
  `orders.status = 'paid'` and the request outputs `id` and `amount`; `schema`
  is `[id, amount, status]`, `output_schema` is `[id, amount]`.
- **`filter`** is `ReadRel.filter`: a predicate the scan must apply, over
  `schema`'s columns. Several pushed-down predicates are one conjunction.
- **`reader`** is one of three:

  | Reader | Substrait read type | Fields |
  |---|---|---|
  | `VirtualTable` | `VirtualTable` | `expressions`: the inline rows, each a list of expressions |
  | `FilesScan` | `LocalFiles` | `format`, `format_options`, `pattern`, `base_path`, `paths` |
  | `TableScan::NamedTable` | `NamedTable` | `names`, `format` |

  - **`VirtualTable`** holds the rows `ValuesNode` held (ADR 0066 ruling 4).
  - **`FilesScan`** holds one path source's read information:
    - `paths` — the listing compile resolved, one mirrored `uri_file` item per
      object. It is the truth: what compile saw.
    - `pattern` (semstrait-only) — the pattern as authored, one per expanded
      prefix as the amendment keeps it; `None` for a partition branch.
    - `base_path` (semstrait-only) — the unit's base path.
    - `format` — the source's format, optional because the manifest's is. In
      Substrait each item carries its own format; the producer writes this one
      on each.
    - `format_options` — the text format's read options: field delimiter,
      maximum line size, quote, header lines to skip, escape, and the value
      read as null (Substrait's `DelimiterSeparatedTextReadOptions`; its
      Parquet, Arrow and ORC options are empty). Admitted by the human's
      ruling; `None` today, since the manifest records no read options.
  - **`TableScan::NamedTable`** holds the catalog identity as `names`:
    exactly three entries, in this order — the catalog, the namespace (its
    parts joined by `.`, one entry that nothing splits), the name. A table
    with no catalog has `""` as its first entry: catalog `lake`, namespace
    `sales.prod`, table `orders` is `["lake", "sales.prod", "orders"]`, and
    the same table with no catalog is `["", "sales.prod", "orders"]`. Beside
    it, semstrait-only, the reader holds the `format` the catalog reported
    (`Iceberg`, `Delta`, …) so an adapter can tell. `TableScan` is the home of
    the table read types; `NamedTable` is the only one admitted.

The copy is total and needs no derivation: every field is one the manifest
holds. `SourceRef` retires with its promise that "adapters look it up against
the manifest", and `ResolvedColumn` with it: the columns and their types are
`schema`.

**The glob and the listing both ride.** The pattern and the base path are the
portable shortcut: an engine that reads globs writes one `uri_path_glob` of the
pattern, one that reads folders writes one `uri_folder` of the base path, and
the others read the listing. The planner never chooses; the faithful default
is the listing (ADR 0068). A rule that writes the glob or the folder lets the
engine list the storage again at run time, and what it lists may differ from
what compile saw; that choice is the rule's, cited in it.

### Partitions

A partitioned unit is planned as branches, one per partition that the amendment's
`Partitioned` contents record — the partition index is that list of key values
and files, which compile discovers from the `key=value` folder layout with no
catalog. For each partition the planner writes:

- a `ScanNode` whose `FilesScan` holds that partition's files, with
  `base_path` = the unit's base path plus the partition's `key=value`
  segments, and no pattern;
- above it, a `ProjectNode` whose expressions add the partition's key values
  as literals, typed by ADR 0066 ruling 5 — the files do not hold the
  partition columns.

Scenario: the unit `orders/` has the partitions `date=2026-09-01` and
`date=2026-09-02`; each branch scans one folder, `orders/date=2026-09-01/` and
`orders/date=2026-09-02/`, and adds `date` as a literal. The branches go under
one `MultisetNode(UnionAll)`; a single partition needs no union. Each branch's
`semantic_meta` names its partition.

Substrait's `FileOrFiles.partition_index`, `start` and `length` are not
admitted: they number and slice a work split of one file set among workers,
not a data partition, and semstrait plans no distributed compute (ADR 0066
ruling 7).

### An Iceberg table is read as a table

Substrait's `IcebergTable` holds only a metadata-file read,
`MetadataFileRead { metadata_uri, snapshot }`. The manifest records no metadata
file and no snapshot — ADR 0046's deferral stands, since a metadata file is a
version. A table the catalog reported as Iceberg is therefore
`TableScan::NamedTable` with its reported format, for every consumer. An
`IcebergTable` reader enters (ADR 0066 ruling 2) when the manifest carries a
metadata-file URI.

### Not admitted

`ReadRel.best_effort_filter`; the projection mask (the emit form selects);
`ExtensionTable`; `IcebergTable` until the above; the work-split fields; and
`ReadRel.advanced_extension`, so the scan has no `extensions` field (ADR 0066
ruling 7).

### Amends

- **0018:** its `ScanNode` clause, "holds only an opaque `SourceRef` resolved
  adapter-side against the manifest", becomes: the scan carries its reader, the
  physical source's read information, copied by the planner from the manifest's
  physical source, and `SourceRef` retires. Its "no partition metadata" and its
  rejection of expanded physical sources and partition columns on `ScanNode` are
  narrowed: a scan carries its listing and names its partition in trace, and
  partition columns are planned as literals above the scan.

**ADR 0046**'s snapshot deferral is confirmed; its unit of read is the amendment's,
copied here unchanged.

## More Information

Decided 2026-09-23 with a rewrite of ADR 0066.
