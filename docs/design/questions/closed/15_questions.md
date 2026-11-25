---
doc: design/questions/closed/15_questions
status: Closed
purpose: Resolved questions originally raised against `foundations/15_mapping_and_binding.md`
---

# Closed Questions — `foundations/15_mapping_and_binding.md`

> Historical record of ratified mapping-and-binding decisions. Each entry preserves Round-1 framing alongside the closure banner. Live items are in `../open/15_questions.md`; deferred items are in `../deferred/15_questions.md`.

---

## Q-MAP-002 — Partition-info plumbing across the pipeline — CLOSED (2026-04-28)

**Status: CLOSED.** Round-1 default ratified per the four-consumer engine-contract research and the user-ratified `PhysicalSource = engine-level LogicalRelation` model.

**Question (closed scope).** What partition-related information must flow through the pipeline (Model → SemanticManifest → IR → Adapter → engine), and where does it live at each layer? Originally framed as "where should the Iceberg partition-transform record live?"; the closing analysis covers the broader question:

- (a) `15 §3.4 PartitionColumn` content — what structural fields it carries.
- (b) `15 §3.5.4 partition_def` — runtime status in v1.
- (c) `35 ScanNode` — whether any partition fields belong on the IR.
- (d) `PhysicalSource` granularity — single-LogicalRelation per author entry vs per-file expansion.
- (e) Iceberg partition-transforms placement (the original question).

**Refs.**

- `15 §3.4` — `PartitionColumn { name, position, data_type, nullable }`.
- `15 §3.5` — `paths:` / `tables:` resolution; `PhysicalSource` granularity (one engine-level LogicalRelation per author entry, with wildcard expansion at compile producing one `PhysicalSource` per resolved variation).
- `15 §3.5.4` — `partition_def` carriage (manifest-side, runtime-dormant in v1).
- `32 §4 StorageConfig` — author surface (`paths:` / `tables:` / `partition_def:`).
- `35 §5.2` — `ScanNode` shape (no partition fields).
- `35 §5.2.1` — partition-info-never-on-`ScanNode` clause grounded in 4-consumer alignment.
- Substrait `ReadRel`, DataFusion `TableScan`, Spark `LogicalRelation`, SQL emit (DuckDB / Spark SQL / Trino) — primary consumer engines. Research conducted 2026-04-28 confirms 4/4 engines exclude partition info from logical scan rels.
- `17 TemporalShape` — grain-aware planning that consumes partition info indirectly through catalog metadata.

**Ratified verdict — by layer.**


| Layer                                         | Carries partition info?          | Form                                                                                                                                                                                                        |
| --------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model (`32 §4 StorageConfig`)                 | **Yes** — author-facing          | `paths:` / `tables:` (source list); `partition_def:` (catalog-less `Range` / `List` declaration).                                                                                                           |
| SemanticManifest (`15 §3 PhysicalSource`)     | **Yes** — manifest-side metadata | `PartitionColumn[]` per source (name / position / type / nullability extracted from Hive-style path segments at compile, or from catalog partition spec); `partition_def` carried verbatim from the model.  |
| Catalog response (`37 TableMetadataResponse`) | **Yes** — catalog-side artifact  | Iceberg-style partition transforms (`identity`, `year`, `bucket[N]`, …); attached to `CatalogRef`-bearing sources. **Not** lifted onto `15 PartitionColumn`.                                                |
| IR (`35 ScanNode`)                            | **No** — never                   | `ScanNode` has no partition fields. Adapter reads partition metadata from the SemanticManifest via `ScanNode.source` per `35 §5.2.1`.                                                                       |
| Engine                                        | **Yes** — runtime                | All four primary consumers (Substrait, DataFusion, Spark, SQL emit) handle partition pruning engine-side from filter predicates against partition columns. Logical plans never carry partition annotations. |


**Architectural posture — logical planning is a mandate.** Logical-only planning (no partition annotations on the scan rel) is the consumer-ecosystem norm, not a semstrait choice. The 2026-04-28 research confirmed across all four primary consumers (Substrait `ReadRel`, DataFusion `TableScan`, Spark `LogicalRelation`, SQL emit). semstrait stays aligned: `35 ScanNode` is partition-free. Manifest-side partition metadata exists for adapters that wish to consult it, but v1 adapters defer to engine-side pruning (which all four engines support out of the box).

`**partition_def` v1 status.** First-class v1 author surface; **runtime-dormant in v1**. Parsed, schema-validated, and carried through compile for forward-compatibility. v2+ consumers (per-partition extraction per `Q-MAP-009`; partition-aware grain inference per `17`; planner pruning hints) activate against the same declaration without re-authoring the model.

`**PhysicalSource` granularity (closure side-effect).** Each YAML entry under `paths:` or `tables:` resolves at compile to one or more `PhysicalSource`s per the simplified rule in `15 §3.5`:

- Concrete path / FQN → 1 `PhysicalSource`.
- Wildcard path / table-name glob → N `PhysicalSource`s, one per resolved variation.
- Each `PhysicalSource` is an **engine-level LogicalRelation** — one Substrait `ReadRel`, one DataFusion `TableScan`, one Spark `LogicalRelation`, one SQL `FROM` reference. The engine handles file consolidation, schema merge, and Hive-partition discovery internally; `35 ScanNode` and the SemanticManifest do not enumerate per-file detail.
- Multi-`PhysicalSource` Bindings (multiple author entries across `paths:` / `tables:`, or one wildcard entry that fans out at compile) compose under `Union ALL` with per-source pre-aggregation as a planner optimization (`21 §3.2 / §4.5`).

**Iceberg partition-transforms placement (the original question).** Transforms remain catalog-side (`37 TableMetadataResponse`) and are NOT lifted onto `15 §3.4 PartitionColumn`. `22 Grainset` / `17 TemporalShape` consumers that need transform-aware grain inference cross-reference the catalog response via the `CatalogRef` on the `PhysicalSource`. Promoting transforms to `15` would couple the engine-agnostic Binding layer to Iceberg-specific vocabulary; the indirection through `37` keeps the layering clean and lets non-Iceberg catalogs (Hive metastore, Glue, Unity, Polaris) implement transforms or not without affecting the `15` contract.

**Spec edits landed by this closure.**

- `15 §3.5` — replaced four-line resolution sketch with simplified rule (`paths:` / `tables:` separation; concrete path = 1 PS; wildcard path = N PSs; engine-level LogicalRelation framing).
- `15 §3.5.1`–`§3.5.6` — restructured into `paths:` / `tables:` / asymmetry / `partition_def` / authoring guidance / algorithm subsections.
- `15 §3.5.4` — new subsection ratifying `partition_def` runtime-dormant status.
- `35 §5.2` — refreshed `ScanNode` doc-comment with engine-level LogicalRelation framing.
- `35 §5.2.1` — new subsection with the 4-consumer alignment table and partition-info-never-on-`ScanNode` rule.
- `32 §4` — added `tables: Vec<String>` to `StorageConfig` (doc-gap fix; test_data already uses it), made `format: Option<StorageFormat>`, refreshed YAML examples to drop the Hive-partition-glob anti-pattern.

**Open follow-ups.**

- `Q-MAP-009` (Hive-style partition value typing) — deferred to v2; tracks the runtime activation path for `partition_def` and per-partition value extraction.
- `17 TemporalShape` ratification — may revisit if as-of / snapshot logic demands per-partition grain access without round-tripping `37`. If so, the transform record could be promoted to a `15 §3.4` field; not needed in v1.

---

## Q-MAP-004 — JSON nested-field inference — CLOSED (2026-04-28)

**Status: CLOSED.** v1 ratifies "no recursion" — nested-object inference is out of scope per `00 §10` (complex types out of scope) and `15 §4.4` (no recursion). Authors flatten / unnest at ingest. Round-1 framing retained for historical reference; the question reactivates only if `00 §10` is amended to admit complex `DataType` variants.

**Question.** For JSON sources without a declared schema, `15 §4.4` infers scalar fields from a sample but does NOT recurse into nested objects (they fall through as `String`). Is that the right v1 stance?

**Refs.**

- `15 §4.4` — Round-1 default: no recursion.
- `13 §3` — complex types out of scope per `00 §10`.
- `15 §4.3` — `JsonShape { Ndjson, JsonArray }`.

**Arguments for no-recursion (Round-1 default).**

- Matches `00 §10` non-goal (complex types out of scope).
- Nested-object inference is a rabbit hole (`depth N?`, recursive structures, polymorphic unions).
- Authors wanting nested access can unnest upstream (the ingestion-layer answer).

**Arguments for recursion.**

- Lake JSON is often nested (event streams, API captures); flattening at ingest is extra work.
- Peer engines (DuckDB, DataFusion) infer nested types readily.

**Current position.** No recursion; nested-object fields become `String`. Authors unnest upstream.

**Next step.** Revisit when / if `DataType` grows complex-type variants (`Array`, `Struct`); currently blocked by `00 §10`.

---

## Q-MAP-005 — Is `Derived` distinct from `Native`? — CLOSED (2026-04-28)

**Status: CLOSED.** `15 §6.3` ratifies `Derived` as a distinct variant from `Native`; the distinction surfaces in pushdown reasoning and composition-level `FieldOwnership::Derived(expr)` (`16 §7.3`). The 4-variant `CoverageVariant { Native, NullFill, Derived, Metadata }` is now load-bearing across `15 §13 R22`, `16 §8`, and the per-DataKind eligibility predicates (`22 §3.3`, `24 §8.4`). Round-1 framing retained for historical reference.

**Question.** `15 §6.3` ratifies `Coverage::Derived` as distinct from `Native` — distinguishing "physical column is directly present" from "upstream columns for a Computed expression are present." Is the distinction pulling its weight, or could `Derived` collapse into `Native`?

**Refs.**

- `15 §6.1` — variant definition.
- `15 §6.3` — current ratification.
- `34 §5` (pending) — pushdown reasoning consumes the distinction.

**Arguments for keeping `Derived` distinct (Round-1 default).**

- Pushdown reasoning in `34 §5` benefits: `Native` reads are cheap (direct column scan); `Derived` reads push a computation into the scan (more expensive / subject to predicate-pushdown differences).
- `16`'s composition-level Coverage may need the distinction to surface provenance accurately on a `ComposedSemanticInterface`.

**Arguments for collapsing.**

- Simpler: the planner sees "source covers this Semantics natively" — the engine is responsible for efficient retrieval, not the canonical layer.
- `34` does not yet exist; the pushdown-reasoning use case is speculative.

**Current position.** Distinct. `Derived` remains its own variant.

**Next step.** Revisit at `34` drafting. If `34`'s pushdown logic does not materially consume the distinction, collapse `Derived` → `Native` on `Computed`-valued Semantics and simplify §6.

---

## Q-MAP-007 — `path.token` index convention — CLOSED (2026-04-27)

**Status: CLOSED.** Option A ratified — `path.token: N` is **0-indexed, scheme-stripped**. Segments are slash-delimited; leading `/` produces no empty token; consecutive `/` collapse. Example: `"s3://bucket/year=2024/month=01/data.parquet"` → tokens `[bucket, year=2024, month=01, data.parquet]`; `token: 1` = `"year=2024"`.

**Refs.**

- `13 §4.7` — `PathExtraction { token: u32 }` (v1 in scope per §4.7 v1-scope banner).
- `15 §8.1` — layer-3 mechanic, ratified.
- `15 §13 R28` — entry confirmed `✓ (Q-MAP-007 closed)`.
- `18 §10.4` — `MetadataExtraction::Path { token: u32 }` (v1).

**Resolution rationale.** Matches `Path::components` (Rust) / `pathlib.Path.parts` (Python). The `partition.level` 1-indexing asymmetry is itself v2-deferred (Q-MAP-009), so the consistency-with-partition-levels argument no longer applies in v1.

---

## Q-MAP-008 — Path-token extraction: raw segment vs value-after-`=` — CLOSED (2026-04-27)

**Status: CLOSED.** v1 ratifies **raw segment only**. `path_token` returns the whole slash-delimited segment (`"year=2024"`) as a `String`, never the `=`-suffix value. Authors who need value-after-`=` compose it via a separate `Expr`-mapped Dimension calling `substring_after(@source_segment, '=')` from the `14a` function catalog, where `@source_segment` is itself a metadata-bound Dimension. No companion `path.token_value: N` variant is ratified in v1.

**Refs.**

- `15 §8.1.1` — token extraction result type, ratified raw-only.
- `15 §13 R29` — entry confirmed `✓ (Q-MAP-008 closed)`.
- `14a` — function catalog hosts `substring_after(str, delim)` for the composition.
- `15 §1` — three-stratum note explains how constant-folding at the `SemanticExpr → PhysicalExpr` lowering boundary collapses the composition into a per-source `Literal` in practice.

**Resolution rationale.** Path structures vary (`year=2024`, `y2024`, `2024`); a `=`-split-baked-in extraction would fail on the latter two. Returning the whole segment keeps the layer-3 contract narrow and makes the value-extraction parse behavior explicit and auditable in author code.

**v2 consideration.** If early-usage feedback shows authors routinely wrap the raw extraction with `substring_after`, a `path.token_value: N` variant can be ratified additively as MINOR per `30 §6.3` (the `MetadataExtraction` enum is `#[non_exhaustive]`).

---

## Q-MAP-006 — `ResolvedColumnMapping.computed` storage: duplicate or alias? — CLOSED (2026-05-28)

**Status: CLOSED.** Settled by manifest ratification clauses C11 + C12 (see `_research/manifest/RATIFICATION_LOG.md`, 2026-05-28).

**Resolution (superseded by STATUS item V, 2026-05-29).** The original resolution persisted split typed pools (`semantic` + `physical`). Under the layered-expressions rework the manifest is **physical-only**: `ManifestExpressions { physical: BTreeMap<PhysicalExprId, ManifestExpression{ expr, layer }> }` (`33 §7.2`) — the `semantic` pool and `SemanticExprId` are dropped, `ExprLayer` is added per entry, and C12.5 (cross-pool link) is moot. The underlying question (duplicate vs alias across pools) dissolves with a single pool. Original resolution retained below for history: SemanticManifest persisted expressions as split typed pools — `semantic: BTreeMap<SemanticExprId, SemanticExpr>` + `physical: BTreeMap<PhysicalExprId, PhysicalExpr>` (C11/C12.2/C12.3/C12.4/C12.5).

`SemanticBinding` no longer owns `PhysicalExpr` trees: per the C2.4 / C11 cascade, mappings reference the typed pool via `SemanticMappingValue::Expr(PhysicalExprId)`. The Round-1 "duplicate vs alias" axis is dissolved — bindings hold ID references, not owned trees, so neither duplication nor `Arc`-aliasing applies. Single source of truth lives in the typed pool.

**Question (closed scope).** `19 §3.4`'s `ResolvedExprTable` is a global `(SemanticsName, BindingId) → PhysicalExpr` map. `15 §7.5`'s per-Binding `computed: HashMap<SemanticsName, PhysicalExpr>` serves the same data for per-Binding lookup. Does the SemanticManifest store the `PhysicalExpr` twice (duplicated), or do the per-Binding values alias into the global table?

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` C11, C12 — ratifications.
- `33 §4.6` (Phase 3) — `ManifestExpressions` shape.
- `19 §3.4` — original `ResolvedExprTable` framing (now superseded by pool-keyed lookup).
- `15 §7.5` — original per-Binding denormalization (now ID-referenced).

**Resolution rationale.** A manifest-level enum tagging Semantic / Physical was rejected (C12 P2 alternative) because it would regress on IR's type-level invariant per 35:698 / 35:702 — every binding-side lookup would re-introduce a runtime match. Split pools preserve the static-type discipline at every reference site.

---

## Q-MAP-001 — `BindingId` uniqueness (per- vs cross-manifest) — CLOSED / MOOTED (2026-05-29)

**Status: CLOSED — mooted by the id-first rework (STATUS item U.2).** The original question asked whether `BindingId(pub u32)` should be per-manifest (compile-counter) or carry a cross-manifest identity (e.g. content-hash). The eliminate-handles decision dissolves it: there is no `BindingId` newtype. Bindings are identified everywhere by a deterministic content-derived `EntityId` (UUIDv5 over `(data_kind id, source id, mapping)`, `33 §9.1`). That id is globally unique and **cross-run / cross-edit stable** for unchanged binding content, so cross-manifest per-binding comparison — the only real driver for the "cross-manifest" option — works by construction, without a counter or a separate hash scheme.

**Refs.**

- `15 §2.2` — binding identity = deterministic `EntityId`.
- `19 §3.2` — `ResolvedExprKey { semantics_name, binding_id: EntityId }`.
- `33 §7.1` / `§9.1` — manifest binding shape + UUIDv5 generation.
- `14b §OQ-7` — consumer-side cross-link (also mooted).
