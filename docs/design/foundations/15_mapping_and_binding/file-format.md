---
id: 15-file-format
status: ratified
version: 2.1
impl: implemented
---

# File Format

The FileFormat enumeration (Parquet, Csv, Json, Orc, Avro) declares the physical encoding of file-based data sources. This document defines the five variants, CsvOptions/JsonOptions, per-format schema-resolution strategy, and format inference from path. See [physical-source](physical-source.md) for PhysicalSource structure. <!-- cross-cut: none -->

## 4. `FileFormat`

### 4.1 Enumeration

```rust
#[non_exhaustive]
pub enum FileFormat {
    Parquet,
    Csv(CsvOptions),
    Json(JsonOptions),
    Orc,
    Avro,
}
```

The v1 set covers the mainstream columnar + text formats. Adding a variant (e.g. `Delta`, `Iceberg` as a file-level format, `Arrow`) is MINOR per `30 §2`.

**Why `Parquet` and `Orc` have no `*Options` companion in v1.** They are self-describing: the schema is in the footer, and there are no per-read behavioral knobs that semstrait needs to expose on the Binding surface. Future read-option knobs (projection pushdown strategy, bloom-filter handling, compression selection) are adapter concerns (`36`) and do not propagate into `FileFormat`. If one ever needs to — for a per-Binding override — the variant can grow `Parquet(ParquetOptions)` as a MINOR.

**Why `Avro` has no `*Options` companion in v1.** Avro schemas are carried with the file (object-container format); there's nothing to declare at the Binding surface. If a schema registry integration lands later, the variant grows `Avro(AvroOptions)` as a MINOR.

### 4.2 `CsvOptions`

```rust
pub struct CsvOptions {
    pub has_header: bool,
    pub declared_schema: Option<Schema>,
    pub delimiter: u8,
    pub quote: u8,
    pub null_sentinel: Option<String>,
}
```

- `has_header` — when `true`, the first row is treated as column names (and the schema is read from it); when `false`, columns are named `_col0`, `_col1`, ... (positional). See §4.4 for the interaction with `declared_schema`.
- `declared_schema: Option<Schema>` — the author can supply the full schema inline in the Model. When present, it takes precedence over inferred or header-derived schemas (§4.4).
- `delimiter: u8` — default `b','`. Other common values: `b'\t'` for TSV, `b';'` for European CSV.
- `quote: u8` — default `b'"'`. Set to `b'\''` to enable single-quote strings.
- `null_sentinel: Option<String>` — a string value treated as NULL when reading. Default: empty string is NULL. `None` disables sentinel handling; an empty string is a legit empty-string value.

These are the minimum knobs observed in the peer set (`dbt` via `external_sources`, `duckdb`'s `read_csv_auto`, DataFusion's `CsvReadOptions`) that **cannot be ignored** without breaking the author's ability to describe the file. Every other CSV knob (date format, decimal separator, encoding) is a v2+ addition gated on concrete need.

### 4.3 `JsonOptions`

```rust
pub struct JsonOptions {
    pub shape: JsonShape,
    pub declared_schema: Option<Schema>,
    pub sample_rows: usize,
}

#[non_exhaustive]
pub enum JsonShape {
    Ndjson,
    JsonArray,
}
```

- `shape` — `Ndjson` is the streaming form (one JSON object per line, the common lake format); `JsonArray` is a single top-level array (the common REST-response form).
- `declared_schema` — same semantics as `CsvOptions`.
- `sample_rows` — how many records to scan when inferring a schema in the absence of `declared_schema`. Default: `100` (matches DataFusion and metricflow defaults).

### 4.4 Per-format schema-resolution strategy

The compile-time strategy per format:


| `FileFormat` | Primary source                   | Fallback                                                                                                                                                                     | Override | Result                                                                                                                                                                                                                                              |
| ------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Parquet`    | File footer metadata             | —                                                                                                                                                                            | —        | Schema from footer; physical Arrow type → canonical `DataType` via a fixed mapping table owned by `semstrait-catalog`'s filesystem reader.                                                                                                          |
| `Csv(opts)`  | `opts.declared_schema` if `Some` | if `opts.has_header`, read header row and treat every column as `String` unless a declared type is overridden per column; else positional `_colN` with every column `String` | n/a      | Every CSV column is `String` unless `declared_schema` overrides. Downstream `Cast`s (§9.1) do the conversion.                                                                                                                                       |
| `Json(opts)` | `opts.declared_schema` if `Some` | sample `opts.sample_rows` records, infer per-field scalar types by widest-observed promotion, treat mixed types as `String`, record nullability as "any null observed"       | n/a      | Inferred schemas carry a `W` diagnostic — `COMP_W_0301 SchemaInferenceUsed` — advising authors to declare the schema explicitly for reproducibility (I4 is best-effort when inference is used; a different sample may infer different nullability). |
| `Orc`        | File footer metadata             | —                                                                                                                                                                            | —        | Same shape as Parquet.                                                                                                                                                                                                                              |
| `Avro`       | Object-container schema          | —                                                                                                                                                                            | —        | Direct schema from the container; logical type conversion done in `semstrait-catalog` per the same canonical mapping table as Parquet.                                                                                                              |


Inferred schemas (CSV without declared schema; JSON without declared schema) DEGRADE the I4 determinism guarantee because the "inferred" result depends on the actual bytes of the first N records. The design admission is explicit: **Binding output is deterministic w.r.t. a given catalog snapshot + filesystem snapshot**; if the bytes at the source change between runs, the schema can change. This is captured as `COMP_W_0301 SchemaInferenceUsed` and advised against for production Models.

**Proposed (Round 1):** JSON inference does not recurse into nested objects — only top-level scalar fields are typed; nested-object fields fall through as `String`. Array typing is not supported (arrays become `String`). Complex types (arrays, structs) are out of scope per `00 §10`. Authors needing nested JSON model the unnest explicitly in upstream jobs. See `questions/closed/15_questions.md` Q-MAP-004 (closed).

### 4.5 Format inference from path

When a Model's binding spec is a file glob without an explicit `format:`, the format is inferred from the file-extension suffix of each resolved absolute path:


| Extension (lower-cased)      | Inferred `FileFormat`                                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `.parquet`, `.pq`            | `Parquet`                                                                                                                                                                                                                |
| `.csv`, `.tsv`               | `Csv(CsvOptions::default())` (`tsv` forces `delimiter = b'\t'`)                                                                                                                                                          |
| `.json`, `.jsonl`, `.ndjson` | `Json(JsonOptions { shape: Ndjson, ... default })`; `.json` alone uses `shape: JsonArray` only if the top-level is a single `[`, otherwise falls through to `Ndjson` — this tie-breaker is the heuristic ratified for v1 |
| `.orc`                       | `Orc`                                                                                                                                                                                                                    |
| `.avro`                      | `Avro`                                                                                                                                                                                                                   |
| anything else                | `CompileError::UnrecognizedFileFormat { path }` (COMP_E_0303)                                                                                                                                                            |


Mixed-format globs are not supported: every path resolved from one glob must infer to the same format, or `CompileError::MixedFormatsInGlob { pattern }` fires (COMP_E_0304). The decision is ratified against the peer-group norm — metricflow and Cube both require format homogeneity per source.

## Testing Strategy

- Per-format schema-resolution table: §4.4 five formats (Parquet auto, Csv/Json projected_schema or infer, Orc/Avro auto)
- Format inference: §4.5 extension-based inference when FileFormat::Auto
- UnrecognizedFileFormat: COMP_E_0303 for unknown extensions
- MixedFormatsInGlob: COMP_E_0304 when glob expands to multiple formats
- SchemaInferenceUsed: COMP_W_0301 warning when schema inferred at compile
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
