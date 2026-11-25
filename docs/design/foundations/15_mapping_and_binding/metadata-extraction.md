---
id: 15-metadata-extraction
status: ratified
version: 2.1
impl: implemented
---

# Metadata Extraction

MetadataDimension extraction semantics enable compile-time derivation of dimension values from file-path tokens. This document defines the v1 path-token mechanic (`path_token` compile function, token indexing, Cast to declared `data_type`, per-source LiteralValue resolution), Coverage::Metadata, and the v2 partition extraction deferral. See [expression-strata](expression-strata.md) for the three-stratum model and [coverage](coverage.md) for Coverage structure.

## 8. `MetadataDimension` Semantics

> **§8.0 v1-scope banner — path-only.** The author-side body in `13 §4.7` carries both `path` and `partition` shapes for forward-compatibility:
>
> ```rust
> // 13 §4.7 (recap, author surface)
> pub struct MetadataDimension {
>     pub path: Option<PathExtraction>,
>     pub partition: Option<PartitionExtraction>,
> }
> ```
>
> v1's runtime / compile-mechanic surface is **path-only**. The compile-stage recipe synthesis (§5.5 / §10.4 step 4.0) only handles `MetadataExtraction::Path` (`18 §10.4`); a Dimension with `partition: Some(_)` is rejected at compile with `COMP_E_0322 MetadataPartitionDeferredV2` until v2 ratifies the partition arm. The §8.2 / §8.3 / §8.4 sections below are retained as **v2 design parking** — they describe the eventual partition-extraction mechanic, but no compile path implements them in v1.

This section is the **layer-3 compile-time mechanic** (banner, §1) for path-token extraction. Recap of the relationship across layers and docs:

- `13 §4.7` / `18 §4` — the author surface (`MetadataDimensionBody`, `MetadataSource`).
- `18 §10.4` — the compile-output struct shapes (`MetadataDimensionRecipe`, `MetadataExtraction`).
- `15 §5.5` — the compile-stage synthesis flow that converts the author body into a `SemanticMappingValue::Metadata(...)` entry.
- `15 §8` (this section) — the actual layer-3 mechanic that runs at compile to produce per-source `LiteralValue`s for `ResolvedPhysicalSource.metadata_values` (§7.6).

The mechanic is **not a `PhysicalExpr` variant** (it has no SQL-equivalent; it operates on metadata not row data) and **not a `14a` registry function** (it has no per-row evaluation; it runs once at compile per `(source, semantics)` pair). It lives in `15`'s implementation as a free function `path_token(path: &str, token: u32) -> Result<String, CompileError>` invoked during the Coverage-derivation pass (§10.5).

### 8.1 `path_token` — the v1 path extraction mechanic

Applicable to `PhysicalSource::File` variants only. Tokenization rule:

1. Take the source's resolved path (e.g. `"s3://bucket/year=2024/month=01/day=15/data.parquet"`).
2. Strip any scheme prefix (`"s3://"`, `"gs://"`, `"file://"`).
3. Split the remainder on `/` into non-empty segments. Leading `/` produces no empty-string token; consecutive `/` collapse to a single delimiter.
4. The 0-indexed segment array is the input to `path_token`.

For the example path, segments are:

```
0: "bucket"
1: "year=2024"
2: "month=01"
3: "day=15"
4: "data.parquet"
```

`path_token(path, 1)` → `"year=2024"` (the full segment, not the value after `=` — see §8.1.1).

**Local-path case.** `"/mnt/data/year=2024/month=01/file.parquet"` → `0: "mnt"`, `1: "data"`, `2: "year=2024"`, etc.

**Windows-style paths are NOT supported in v1** (explicit non-goal; the Model surface is lake-native).

#### 8.1.1 Token extraction result type — raw segment

`path_token` returns the raw segment as a `String` — the whole token, NOT the `=`-suffix value. For `"year=2024"`, the result is `"year=2024"`.

**Rationale.** Splitting on `=` would be an implicit second-parse step that fails silently on paths like `"year-2024"` or `"year"` (no `=` at all). Returning the whole segment keeps the layer-3 contract narrow; authors who need the value-only form compose it via a `SemanticMappingValue::Expr` Dimension that calls `substring_after(@source_segment, '=')` from the `14a` function catalog, where `@source_segment` is itself a metadata-bound Dimension carrying the raw segment. (Constant folding at the `UnresolvedSemanticExpr → PhysicalExpr` lowering boundary collapses this composition into a per-source `Literal` in practice; see §1's three-stratum note.)

#### 8.1.2 `Cast` to declared `data_type:` — fail-fast at compile

The recipe (`MetadataDimensionRecipe.data_type`) carries the Dimension's declared type. After `path_token` returns the raw `String`, the compile stage applies a `Cast` to the declared type:

- `data_type: String` → identity (no cast emission).
- `data_type: Integer` / `Long` / `Decimal` → parse the string as the numeric type. Parse failure → `COMP_E_0321 MetadataCastFailed`.
- `data_type: Date` / `Timestamp` → RFC 3339 parse. Parse failure → `COMP_E_0321`.
- `data_type: Bool` → `"true"` / `"false"` (case-insensitive); else `COMP_E_0321`.

Cast failures are **fail-fast at compile**, not deferred to plan or run time. The metadata literal is known at compile and so is its target type; failure to cast indicates the author wrote an incompatible declared type for the Dimension or the source path does not match the expected shape — both are Model bugs that the author must fix.

#### 8.1.3 Error conditions


| Code                                       | Trigger                                                                                                                                                                                                                                                         | Detail                                                                                                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `COMP_E_0311 MetadataTokenOutOfRange`      | `token >= segments.len()` for a given source's path.                                                                                                                                                                                                            | `{ binding_id, source_index, token_index, path }`. Fail-fast at compile (§10.5).                                                                 |
| `COMP_E_0312 MetadataTokenOnNonFileSource` | A metadata-bound Semantic with `Path { token }` extraction is in a Binding whose source at `source_index` is `Table` or `Snapshot`.                                                                                                                             | `{ binding_id, source_index, semantics }`. Path-token extraction has no defined value for non-file sources in v1 — fail-fast at compile (§10.5). |
| `COMP_E_0320 MetadataTokenEmpty`           | The segment at the requested index is empty (e.g. trailing `/` with no following content; should not occur after the §8.1 step-3 normalization, but the check is defensive against author misuse like `token: N` where `N` aligns with a degenerate component). | `{ binding_id, source_index, token_index, path }`. Fail-fast at compile.                                                                         |
| `COMP_E_0321 MetadataCastFailed`           | The raw `String` segment cannot be cast to the recipe's declared `data_type:`.                                                                                                                                                                                  | `{ binding_id, source_index, semantics, raw_segment, declared_type }`. Fail-fast at compile (§8.1.2).                                            |


All four checks run per-source during §10.5 (Coverage derivation); failures triggered immediately at compile, never at plan or run time.

### 8.2 `partition.level: N` extraction — DEFERRED TO v2

> **v2-deferred per §8.0.** The compile pass rejects `MetadataDimensionBody.source.partition: Some(_)` with `COMP_E_0322`. The text below is **design parking** — the eventual partition-extraction mechanic, retained for forward-compat reference. It does not run in v1.

Applicable to all `PhysicalSource` variants that declare `partitions: Vec<PartitionColumn>`.

Rule: 1-indexed. `level: 1` is the first partition column (`PartitionColumn.position == 1`); `level: N` is the *N*-th partition column.

For `PhysicalSource::File` with Hive-style path partitioning (`year=2024/month=01/day=15/data.parquet`), the compile-time source-resolution algorithm in §3.5.6 step 4 builds the `PartitionColumn` list from the path components. The partitions are ordered outer-to-inner in the path; `position: 1` = `year`, `position: 2` = `month`, etc.

For `PhysicalSource::Table` and `Snapshot` backed by Iceberg (or equivalent), the partition column list is the table's declared partition spec (the `default-spec-id` at the catalog level). Partition-transform identities (`identity`, `year`, `month`, `bucket[N]`) are carried on the catalog side (`37`) and are not surfaced on the `PartitionColumn` struct itself in `15`'s v1.

#### 8.2.1 Partition extraction result type — v2 design parking

The extracted value is the partition column's value for a given row, typed as the declared `PartitionColumn.data_type`:

- Hive-style path: typically `String` unless a `data_type` override is declared.
- Iceberg / Unity table: the declared partition-column `DataType` (which may be `Integer`, `String`, `Date`, etc.).

The exact result-type contract for Hive-style partitions (raw segment vs value-after-`=`, declared override grammar, type-inference fallback) is a v2 ratification item — see `questions/deferred/15_questions.md` Q-MAP-009 (deferred).

#### 8.2.2 Error conditions — v2 design parking

- `CompileError::MetadataPartitionUnavailable { binding_id, source_index, semantics, level }` — the source has no partition spec or fewer partition columns than `level`. Proposed code: `COMP_E_0313` (deferred until v2 ratifies the partition arm).
- `CompileError::InconsistentPartitioning { binding_id, sources: Vec<usize>, semantics }` — multi-source Binding where sources disagree on partition structure for referenced levels. Proposed code: `COMP_E_0314` (deferred until v2).

### 8.3 Cross-variant exclusion — v2 design parking

When the partition arm lands in v2, the YAML parser (`32`) will enforce `path.is_some() XOR partition.is_some()` at the `MetadataDimensionBody` author surface. v1 only has the `path` arm in scope, so the `partition: Some(_)` arm is rejected at compile (`COMP_E_0322`) regardless of the `path` arm's state.

### 8.4 `Metadata` Coverage uniformity — v1

Per §6.2, a metadata-bound Semantic on every applicable source emits `Coverage::Metadata` and writes its resolved `LiteralValue` into `ResolvedPhysicalSource.metadata_values[semantics]`. **There is no `NullFill` interaction in v1**: applicability is a fail-fast Model-correctness check (`COMP_E_0312` for `Path { token }` on a non-File source), not a per-source soft-fallback. A Binding that mixes file-glob and table sources with a path-token-bound Dimension fails at compile, period — it is not legal even inside a Unionset consumer.

The lenient "fill with NULL on non-applicable sources" pattern (the previous `8.4` behavior) was rejected because: (a) metadata extraction is a Model-correctness contract, not a per-row data-quality concern; (b) silently NULLing a metadata Semantic disrupts the constant-folding feed (§5.5) that is a core motivation for the recipe; and (c) when authors need cross-source heterogeneity in this dimension, the right tool is a separate Binding per source class composed by a Unionset, not implicit per-source NULL fallback.

## Testing Strategy

- path_token layer-3 mechanic: §8.1 0-indexed token extraction, scheme-strip, slash-split
- MetadataTokenOutOfRange: COMP_E_0311 when token index exceeds path depth
- MetadataTokenOnNonFileSource: COMP_E_0312 when path_token on Table/Snapshot source
- MetadataTokenEmpty: COMP_E_0320 when extracted token is empty string
- MetadataCastFailed: COMP_E_0321 when token value cannot cast to declared data_type
- MetadataPartitionDeferredV2 guard: COMP_E_0322 when partition extraction attempted (v2 deferred)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
