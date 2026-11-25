---
doc: design/questions/deferred/33_questions
status: Deferred
purpose: Manifest questions parked for post-v1 ratification or follow-up research
---

# Deferred Questions — `apis/33_semstrait_manifest.md`

> Items deferred to v2 (or later) ratification, plus Phase 2 manifest-research candidates that weren't critical for the 2026-05-28 manifest ratification cascade. Live items are in [`../open/33_questions.md`](../open/33_questions.md); closed items in [`../closed/33_questions.md`](../closed/33_questions.md).

---

## Q-MAN-D01 — Thread B: glob-expansion semantics for `paths:` — DEFERRED

**Status: DEFERRED.** Tabled at the 2026-05-28 manifest ratification cascade as a model-layer concern that does not block manifest shape ratifications. The manifest carries `PhysicalSourceType::File { uri }` post-resolution; the question of how `paths: ["s3://bucket/year=*/data.parquet"]` expands into `Vec<PhysicalSource>` (eager listing at compile vs lazy at runtime, supported wildcard grammars, ordering of resolved paths) belongs to `32` / `15` resolution and the `37` catalog drift contract — not to the manifest's persistence shape.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — Thread B (model-layer concern, deferred from manifest cascade).
- `15 §3.5` — `PhysicalSource` granularity (one engine-level LogicalRelation per author entry; wildcard fanout at compile).
- `32 §4 StorageConfig` — `paths:` author surface.
- `37 §...` (pending) — drift-check posture for resolved paths.

**Open axes (preserved for v2 ratification).**

- **Eager vs lazy expansion.** Eager (compile lists each file) gives stable manifest content but couples manifest stability to filesystem state at compile. Lazy (compile keeps the glob string verbatim; engine expands at runtime) decouples but loses content-addressable stability.
- **Wildcard grammar.** POSIX glob (`*`, `?`, `[abc]`) vs Hive-style (`year=*`) vs full regex. Today's code uses POSIX glob; spec is silent.
- **Ordering of resolved paths.** Lexicographic vs filesystem-listing order vs author-declared order.

**Next step.** Address during `32` / `15` re-ratification of `paths:` resolution semantics; output feeds back into `33`'s `PhysicalSourceType::File` payload shape.

---

## Q-MAN-D02 — G5 workflow-friction guard: relationship-orphan policy — DEFERRED

**Status: DEFERRED.** The G5 validation gate (manifest-build-time check that no `Relationship` references a missing constituent DataKind) is **ratified** as part of C13's validation roster (RATIFICATION_LOG.md, 2026-05-28). The deferred question is whether G5 should be **strict** (reject) or **lenient** (warn-and-prune) in v1, given the workflow-friction concern raised during ratification: an author iterating on a Model who deletes a DataKind would see manifest compile fail until they also delete every referencing `Relationship`. v1 ratifies strict (G5 → error, `MANIFEST_E_...-RELATIONSHIP-ORPHAN`); the lenient alternative is parked.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C13 G5 (orphan-relationship gate; strict in v1).
- `33 §10.2` — manifest validation gate roster.
- `16 §13` — `Relationship` placement and reference discipline.

**Lenient-mode option (deferred).** Treat orphan `Relationship` as a warning (`MANIFEST_W_...-RELATIONSHIP-PRUNED`) and silently drop it from the manifest. Argument for: smoother in-loop iteration; argument against: silent drop violates I4 determinism — two compiles of the same Model produce different manifests if the constituent set differs.

**Next step.** Revisit if early-usage telemetry shows G5 firing on iteration sequences where the author intent is "delete the kind, the relationships are obviously gone too." A warn-and-prune mode could land as a `compile` flag (e.g. `compile_lenient`) without affecting the strict default.

---

## Q-MAN-D03 — G6 cycle-detection on relationship graph — DEFERRED

**Status: DEFERRED.** The G6 gate question — should manifest-build-time validation reject `Relationship` configurations that produce a cycle in the relationship graph? — was raised during the 2026-05-28 manifest ratification cascade and deferred. C13 validation gates G1–G5 are ratified for v1; G6 (cycle detection) is parked because (a) the planner runtime graph (`34 §1.4A`, daggy-backed) already enforces DAG-ness at request time, so a manifest-side cycle would surface as a planner error rather than corrupted output; (b) some legitimate Models use `Relationship`s that look cyclic at the relationship-graph level but are non-cyclic when traversed under `cross_filter` directionality (per `18 §2`).

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C13 validation gates (G1–G5 ratified; G6 deferred).
- `34 §1.4A` — planner runtime graph lifecycle (daggy enforces DAG at request time).
- `16 §13` — relationship-placement discipline.
- `18 §2` — `cross_filter` directionality.
- See sibling deferral: [`16_questions.md`](16_questions.md) — composition-side cycle-detection (Phase 2 Target E).

**Open axes (preserved for v2 ratification).**

- **Definition of cycle.** Undirected edge cycle (any back-edge) vs directed-under-`cross_filter` cycle. The latter aligns with planner traversal semantics; the former is stricter.
- **Error vs warning.** Strict reject vs warn-and-let-planner-error.
- **Detection scope.** Among explicit `Relationship`s only, or extended through implicit-composition synthesis.

**Next step.** Phase 2 research target (Target E in the manifest-research deferred candidates) covers cycle-detection algorithms. Address during a Round-4 framework cleanup pass after planner-runtime-graph implementation surfaces real-world cycle hazards.

---

## Q-MAN-D04 — Duplicate-authoring diagnostic for primitive collections — DEFERRED

**Status: DEFERRED.** C18.5 (the duplicate-authoring diagnostic that fires when the same entity `EntityId` — formerly `DataKindName` / `RelationshipId` / `BindingId`, now unified per STATUS item U.2 — appears twice in the manifest's primitive collections) was ratified as a build-time validation per the 2026-05-28 cascade. The deferred question is the **diagnostic shape** — specifically, should the error carry both occurrences' source spans (requires keeping span info on the collection-build path), or just the first-occurrence span plus the duplicate name?

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C18.5 (duplicate-authoring diagnostic).
- `33 §10.2` — validation gate roster.
- `30 §5.3` — `ContextLine` surfaces for diagnostic enrichment.

**Open axes.**

- **Span carriage.** Both spans (richer; requires manifest-build path to retain author-side spans through the primitive-collection build) vs first-occurrence-only (simpler; loses the "where the duplicate was authored" signal).
- **Diagnostic-channel placement.** `CompileError::DuplicateName` (fail-fast at first occurrence) vs an accumulator (gather all duplicates in one pass, then fail).

**Current position.** First-occurrence span only; fail-fast on first duplicate. Promotion to "both spans" is MINOR (no breaking changes to the public surface).

**Next step.** Revisit at `30` ratification of `ContextLine` discipline if the patterns that emerge from real Models suggest authors routinely need both spans to locate the conflict.

---

## Q-MAN-D05 — `PhysicalSourceType::File` payload refinement — DEFERRED

**Status: DEFERRED.** C2 ratifies `PhysicalSourceType::{Table { ... }, File { uri: String }}` for v1 (RATIFICATION_LOG.md, 2026-05-28). The deferred question is whether the `File` variant payload should grow additional fields — `format: Option<StorageFormat>` (parquet / csv / json / iceberg-table-as-files), `compression: Option<Compression>`, `partition_spec: Option<PartitionSpec>` — to carry more discovery information through the manifest without re-resolving on each load.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C2 (`PhysicalSourceType` enum with `File { uri }`).
- `15 §3.5` — `PhysicalSource` granularity.
- `32 §4 StorageConfig` — author surface (`paths:` / `format:`).
- `37 §...` (pending) — catalog discovery contract.

**v1 posture.** `File { uri: String }` carries the resolved URI only; format / compression / partition spec is resolved by adapters at runtime via filename inspection or upstream catalog lookup.

**Open axes (preserved for v2).**

- **Format inference.** Compile-time eager (parse `.parquet` suffix into `StorageFormat::Parquet`) vs runtime lazy (adapter inspects).
- **Compression detection.** Same axis (`.parquet.gz` → `Compression::Gzip` at compile vs adapter detection).
- **Partition-spec persistence.** If `15 §3.5.4`'s `partition_def` activates (per Q-MAP-009 v2 trigger), should the resolved per-source partition layout join `PhysicalSourceType::File`?

**Next step.** Revisit if v2 adapters surface need for compile-time format / compression info (e.g., to select adapter dispatch path before runtime).

---

## Q-MAN-D06 — C7 request-shaped pruning consideration — DEFERRED

**Status: DEFERRED.** C7 ratifies per-Joinset hop coverage with cumulative semantics (`coverage[i] = ⋃_{j ≤ i} contributors[j]`; RATIFICATION_LOG.md, 2026-05-28). The deferred question is whether the manifest should carry a **request-shaped pruning hint** alongside the cumulative coverage — pre-computed bitmaps for common Request prefixes (e.g., the Joinset's anchor + first hop, anchor + first two hops) so the planner can short-circuit hop-walking on partial-coverage Requests.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C7 (per-Joinset hop cumulative coverage).
- `34 §...` — planner pruning logic that consumes Joinset hop coverage.
- `24 §...` — Joinset traversal semantics.

**v1 posture.** Manifest carries cumulative coverage per hop; planner re-derives request-shaped masks on each `plan` call.

**Open axes.**

- **Pre-computation cost.** N hops × M common request shapes × bitmap-size memory; viable for small N · M but blows up on dense models.
- **Request-shape vocabulary.** What counts as a "common shape" — anchor-only? anchor + each owned dim? — is workload-specific.
- **Cache eviction.** If pre-computed pruning hints are persisted, manifest content-addressable stability requires deterministic shape vocabulary.

**Next step.** Phase 2 research target (Target A — Bitmap registries — covers this surface). Reactivates when planner profiling shows hop-walking as a hot path.

---

## Q-MAN-D07 — Phase 2 Target A: Bitmap-registry implementation — DEFERRED

**Status: DEFERRED — Phase 2 research candidate.** Targets C4 (SemanticBitmap registry shape) and C5–C8 (per-DataKind / per-hop / per-grain bitmap layering). The 2026-05-28 cascade ratified the **interface** (epoch-stable bit positions, `SemanticBitmap` newtype, `SemanticId` newtype as bit ordinal); the **implementation strategy** (representation: `roaring::RoaringBitmap` vs `bitvec::BitVec` vs a custom u128 chunked array; serialization: bit-stream vs run-length-encoded; epoch-rotation discipline) is parked for Phase 2 research.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C4–C8 (bitmap registry; interface ratified, implementation deferred).
- `33 §...` — manifest persistence shape post-cascade.
- `34 §...` — planner consumption surface.

**Open axes (Phase 2 scope).**

- **Representation crate.** `roaring` (compressed; good for sparse) vs `bitvec` (dense; constant-time bit ops) vs hand-rolled.
- **Epoch rotation.** Manifest content-hash includes the bitmap registry; how do we keep `SemanticId` ordinals stable across compiles of the same Model? Deterministic name-sort at registry-build is one approach.
- **Cross-manifest comparison.** If two manifests share most of their semantic-id space, can we share a registry? (Out of scope for v1 strict scope.)

**Next step.** Phase 2 research dossier; output feeds back into `33 §...` registry persistence and `34 §...` planner consumption.

---

## Q-MAN-D08 — Phase 2 Target B: Dual expression-form persistence — DEFERRED

**Status: DEFERRED — Phase 2 research candidate. Re-scoped by STATUS item V (2026-05-29).** v1 now persists a **physical-only** pool (no `semantic` pool, no `SemanticExprId`); the applicability `ExprLayer` replaced the diagnostic-only semantic form. The Phase-2 question is therefore reframed: if semantic-layer explainability or shared-aggregate provenance is wanted, v2 would **re-introduce** a persisted semantic form (or a `PhysicalExprId → source-semantic` provenance side-map) — this is the same post-v1 home as the reserved graph `Expression`-node / `DependsOn` edge (`35 §2A.1`). Not needed for v1 correctness.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C11 / C12 (split typed pools, no cross-pool linkage at expression level for v1).
- `closed/15_questions.md` — Q-MAP-006 closure (v1 stance: no cross-pool linkage).
- `33 §4.6` — `ManifestExpressions` shape.

**Open axes.**

- **Provenance edges.** A side-map `BTreeMap<PhysicalExprId, SemanticExprId>` (or `Vec<SemanticExprId>` if multiple semantic forms can share a physical lowering) would let downstream tooling explain a physical plan in terms of authored semantic expressions.
- **Storage cost.** N physical expressions; one pointer each. Cheap.
- **Round-trip integrity.** If lowering is non-injective (multiple distinct semantic exprs collapse to identical physical), the side-map needs `Vec` carriage.

**Next step.** Reactivate when an explainability or provenance use-case surfaces (e.g., a query-explainer UI that wants to map physical plan nodes back to semantic-layer authored forms).

---

## Q-MAN-D09 — Phase 2 Target C: Iceberg / Delta-Lake schema-evolution — DEFERRED

**Status: DEFERRED — Phase 2 research candidate.** Targets C3 (PhysicalSource version_ref), C14 (compile-time catalog metadata fetch), and C15 (schema fingerprint persistence). The interfaces are ratified for v1 (manifest carries `version_ref: Option<String>` per C3; schema fingerprint is SHA-256 over canonical-form schema per C15). The Phase 2 question is how the schema-fingerprint contract integrates with **time-travel-aware** catalogs (Iceberg snapshot id, Delta-Lake version), specifically:

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — C3 / C14 / C15 (PhysicalSource versioning + schema fingerprint persistence).
- `37 §...` (pending) — catalog drift-check contract.
- `33 §13.2` — determinism contract.

**Open axes.**

- **Version-pin discipline.** Should `version_ref` be required for time-travel-capable catalogs (Iceberg / Delta-Lake) but optional otherwise? Or always-optional?
- **Drift-check semantics.** When `version_ref` is `Some(v)` and the live catalog has moved past `v`, is that a drift error, a warning, or transparent (manifest is pinned to `v`, drift is irrelevant)?
- **Schema-evolution narrative.** Iceberg supports column add / drop / rename / type-promote with backward compatibility; schema-fingerprint comparison alone can't distinguish "evolved compatibly" from "broke." A richer drift-check with structural diff is Phase 2.

**Next step.** Phase 2 research dossier on Iceberg / Delta-Lake schema-evolution semantics; output feeds back into `37` drift-check contract and `33` `version_ref` discipline.

---

## Q-MAN-D10 — Phase 2 Target E: Cycle-detection algorithms (manifest-side) — DEFERRED

**Status: DEFERRED — Phase 2 research candidate.** Targets C13 G6 (manifest-build-time cycle detection on the relationship graph). G6 is itself deferred (see Q-MAN-D03 above); this entry tracks the **algorithmic** sub-question if G6 reactivates.

**Refs.**

- See sibling: Q-MAN-D03 — G6 deferral (governance question).
- `_research/manifest/RATIFICATION_LOG.md` — C13 (validation gates).
- `16 §13` — relationship placement.

**Open axes (Phase 2 scope).**

- **Algorithm choice.** Tarjan SCC vs DFS-with-back-edge-tracking vs incremental cycle detection (if `Relationship`s are added incrementally during manifest build).
- **Cycle representation in diagnostics.** A cycle-error should report the cycle nodes in traversal order; algorithm choice affects ordering quality.
- **Performance.** O(V + E) is achievable; for dense relationship graphs, the constant factor matters at compile-time.

**Next step.** Reactivate if and when Q-MAN-D03 (G6 governance) reactivates.

---

## Q-MAN-D11 — `NestedDataKindVariant` consolidation shape — DEFERRED

**Status: DEFERRED.** The manifest verification pass (2026-05-28, post-ratification cleanup) removed duplicate single-use tags (`DataKindRole`, standalone `PathOrigin`) but intentionally kept an explicit `NestedDataKindVariant` subset in `33 §6.3`. The open question is whether nested and top-level variant carriers should share one enum shape with a legality guard (`Joinset` forbidden in nested context) or remain split.

**Refs.**

- `33 §6.2` / `§6.3` — top-level vs nested variant rosters.
- `26` nesting rules R2/R3 — why nested Joinset is forbidden.

**Why deferred.**

- The current split makes nesting constraints explicit and keeps load-time checks simple.
- A shared enum shape would reduce duplicated variant text but introduces context-dependent validity rules and larger cascade surface across planner/model docs.

**Current default.** Keep explicit `NestedDataKindVariant` in v1 docs; revisit if a third nested-only form appears or if code-level duplication becomes measurable.

**Next step.** Revisit during post-v1 shape simplification after `24` and `34` rebases settle.

---

## Q-MAN-D12 — `EntityId` UUID-format relaxation for compile-generated ids — RATIFIED → moved to `closed/33`

**Status: RATIFIED (see [`../closed/33_questions.md`](../closed/33_questions.md) Q-MAN-D12).** Resolution: compile-generated ids are **UUIDv5** (name-based) with per-entity-type namespace + content seed; authored ids stay UUIDv7; CX1 validates canonical-UUID-text + uniqueness, not the version nibble. Landed in `33 §9.1` / `§4.3.1`, `32 §2`, `18` scope note.
