# Spec-Driven Development — Status

Living handoff file for active design work.

Read order for spec sessions:

1. `[00_overview.md](00_overview.md)`
2. `[STATUS.md](STATUS.md)`
3. `[INDEX.md](INDEX.md)` for task routing
4. `[DOCS_MAINTENANCE.md](DOCS_MAINTENANCE.md)` for editing discipline

Historical long-form narrative is archived in:

- `[_archive/STATUS_HISTORY.md](_archive/STATUS_HISTORY.md)`

---

## 1) Current phase

**Phase:** Reconciliation / consolidation (post-ratification cleanup)

**What is stable now**

- DataKind taxonomy and trait axes are ratified.
- Typed diagnostic-kind discipline and `tracing` observability are ratified across `30`-`39`.
- Per-Q-ID question-directory split (`open` / `closed` / `deferred`) is in place.
- Variant chapter rebases — `Dataset` (`21`), `Unionset` (`23`), and `Grainset` (`22`) — complete in slim form (algorithm bodies extracted to `_drafts/34_*_strategy.md` sidecars; cascade-aligned Coverage-inference, TemporalShape-equivalence, and plan-time observable behavior contracts).
- `UnionMode { All, Unique }` v1 roster re-confirmed (C6, 2026-05-03).
- `CompositionKind` shrunk to `{Joinset, Grainset}` V1 (Unionset variant retired post-thirteenth-pass cascade rebase 2026-05-03; Unionset uses bare `SemanticInterface` per `23 §3.2`); `ChildCoverageOverride { provides }` and YAML `coverage:` block retired; `ComposedSemanticInterface` broadened to cover both Joinset per-hop and **Grainset cross-grain LEFT OUTER JOIN composition** on shared `Key`s per `18 §2.5` (per G-2 ratification 2026-05-03).
- Grainset cross-grain JOIN composition mechanism ratified (G-2): driver = most-covering grain-eligible routing unit (declaration-order tie-break per G-2b); attached units in declaration order (G-2c); LEFT OUTER (G-2a); hard compile error `COMP_E_2204 GrainsetCrossGrainKeysAbsent` on missing shared Keys (G-2d). Internal `RollupPolicy { ShapeDefault, PinOnly, PreferFinest }` per G-4 (planner knob, NOT authored in V1 YAML).
- **Expression-flow design Rounds 1–5 fully closed; promoted to `foundations/19_expression_flow.md`** (2026-05-12). Phase A pipeline (parse → resolve → PhysicalExpr); Option B + traits type design (`Expr` / `Foldable` / `Sugarful` / `LowersTo<T>`); per-entity-typed `Accessor` (Q-T-2); typed `Parameter` (R-3); substep order **eliminate sugar → fold → translate** (Q-006); v1 fold language with ANSI-strict `Like` (Q-005); per-Binding materialisation (Q-008); `Aggregate` admitted in both forms with planner lift at Phase B (R-5); §4.3 Category I worked example. Round 4 Phase B placement axis-by-axis: filter placement source-of-definition split (§9.1, Q-009a/b/c/d); group_by = Dims+Keys with structured `DimensionRef { name, variation }` (§9.2, Q-014); computed dim inline pre-agg (§9.3); Metric refs-other-semantics-only with scalar-only `expr:` and `agg:` aggregation (§9.4, Q-018a/b, Q-019); unified `Additivity` enum function-level (`14a §3.1`) + model-level (`18 §5.2`) two-source SoC (§9.5, Q-015a/b); typed `Diagnostics<PlanErrorKind>` channel with unified `PLAN_W_2101 LossyReaggregation { data_kind, .. }` (§9.6, Q-021/-022/-023). Round 5: `MetricAccessor` v1 surface mirrors `MeasureAccessor` 1:1 (`Previous`, `Next`, `Lag(u32)`, `Lead(u32)`, `Delta`, `PercentChange`); same variant names per Q-T-2 type disambiguation; sugar-on-sugar resolved by fixpoint Family B elimination in `resolve` substep 1; §4.2 worked example reworded to align (Q-003a/b/c). Rust-encoding convention: numeric `PLAN_W_*` / `COMP_E_*` / `EXPR_E_*` codes are spec-cross-reference indices commented adjacent to typed-enum variants — NOT runtime data fields.
- **Builder ergonomic facade landed (`32 §9.7.8`)** (2026-05-13) — additive sugar layer on top of the primary structural builders. Per-entity shortcuts (`Dimension::builder().temporal(grains)`, `Measure::builder().sum().full()`); cross-struct flatteners with read-modify-write semantics (`Dataset::builder().catalog("polaris").path("s3://...")`); `SemanticInterface` per-item inserters; `Unionset.union_all()`/`.union_unique()`; symmetry plurals on `Nested-*` and `SemanticModel`. `state_mod(vis = "pub")` mandate on facade-supporting builders. Primary structural surface unchanged. 166 tests passing in `semstrait-model` (was 146); clippy clean. Branch `feature/model-builder-fluent-facade`.
- **Raw-filter unified-pipeline implementation pass landed** (2026-05-19, `feature/raw_filtering`) — implementation-only follow-up to the engine-unification ratification (`Q-RAW-FILTER`, closed in `questions/closed/34_questions.md`). `semstrait-api::RequestParser::to_resolved` now lowers `RawFilter` triples (string operator + JSON value) into `QueryFilter` and appends to `ResolvedQueryRequest.filters`, including the cross-reference rejection `ParseError::RawFilterNamesNamedFilter`. `semstrait-planner::query_filter_to_semantic_expr` rebuilds the predicate as a `SemanticExpr` over `EntityRef` field references, then walks `ExprResolver::resolve_expr` (identity at root level) — the same path `DataKindFilter` / `AggregationFilter` bodies use at scan level. No canonical type changes in `34 §3` / `§3.5` / `§5`; `RawQueryRequest` two-field API surface unchanged; `ResolvedQueryRequest.filters` element shape (`QueryFilter`) unchanged.

**What remains active**

- Adapter/catalog framing reconciliation (item C).
- Residual cross-doc vocabulary cleanup where retired error-code language still appears.
- v1 backlog trimming in open question sidecars.
- Variant chapter rebases — `Joinset` (`24`) pending (the last remaining; `33`/`34` come after). The 2026-05-12 relationship-block rebase (item K) addresses the Relationship-side authoring shape; the full Joinset rebase (algorithm body extracts, etc.) is still pending and tracked separately.
- **Relationship-block rebase (item K, 2026-05-12).** Authoring shape moved to semantic-first (`cardinality` + `integrity` + `optional` + `cross_filter`); `directionality:` and the `JoinTypeOverrides` / `HopPosition` per-hop override surface retired. `JoinType` is derived at compile from `optional` per `18 §2.9`. Validation rules SR-E-13 / SR-E-14 added. Joinset-local divergent semantics via scope-local `Relationship` shadow (`18 §2.10`, `16 §13.3`). Cascaded into `18`, `16`, `24`, `32`, `33`.
- Algorithm-body sidecars (`_drafts/34_simple_strategy.md`, `_drafts/34_unionset_strategy.md`, `_drafts/34_grainset_strategy.md`) pending lift into `34_semstrait_planner.md §<XStrategy>` when the planner doc opens its Strategy chapter.
- Deeper structural cleanup of `16 §9.3` / `§10.5` / `§13` (inert post-Unionset-retirement) parked behind new `Q-COMP-006` for a Round-4 framework cleanup pass.
- Stale `CompositionKind` / `ComposedSemanticInterface` references in `33` pending cleanup at that chapter's rebase.
- **`14_expressions.md §2` rebase** — closed under item J 2026-05-18 (twice-refined; per-kind typed `SemanticLeaf` shape ratified). Cross-doc cascade closed under items N + O.
- **Persisting open clauses** — none. Both post-promotion follow-ups (Q-EXPR-19-001, Q-EXPR-19-002) closed 2026-05-12; `questions/open/19_questions.md` retired (empty). All `19`-tagged ratifications live in `questions/closed/19_questions.md`.
- **Function catalog (`14a §3.6`) extension** — `Additivity` enum ratified at `14a §3.6` per item O. Closes the additivity-on-`FunctionSpec` ratification surface. Two-source SoC (function-level vs model-level `18 §5.2 AdditivityType`) ratified; composition rule lives at `19 §6.5`; deeper field-payload ratification per advisory rests in `[TD-19-ADDITIVITY-COMPOSITION]`.
- **Model-level `AdditivityType` rename** — existing `18 §5.2` `AdditivityType` aligns with new unified `Additivity` enum at refactor time. Variants map 1:1 (`Full` → `Additive`, `Semi(SemiAdditivity)` → `SemiAdditive { axes }`, `Non` → `NonAdditive`). Flagged in `41_deprecations.md` for landing during planner-doc rebase.
- **Advisory field payload `[TD-19-ADVISORY-FIELDS]`** — exact context fields on `PLAN_W_2101 LossyReaggregation { data_kind, .. }` beyond `data_kind` deferred to single-pass ratification at `34_semstrait_planner.md` Strategy chapter rebase.
- **Per-DataKind advisory specialisation `[TD-19-ADVISORY-SPECIALISATION]`** — flag for future split if a `LossyReaggregation` root cause structurally diverges per DataKind (currently unified under `PLAN_W_2101`).
- **`30 §6` typed-diagnostics framing codification** — Rust-encoding convention (numeric `*_W_*` / `*_E_*` codes as adjacent comments on typed-enum variants for grep-ability, NOT runtime fields) is project-wide; lift into `30 §6` next session.

---

## 2) Active reconciliation items


| Item | Summary                                                                                     | Status                                         | Primary docs                                           |
| ---- | ------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| A    | YAML surface and type hierarchy alignment                                                   | Ratified                                       | `32`, `32b`, `26`, pointers in `20`-`25`               |
| B    | `Binding` -> `SemanticMapping` framing and metadata synthesis                               | Ratified at authoring level                    | `15`, `18`, `32`, `33`                                 |
| C    | Adapter/catalog architecture framing (single crate + feature-gated modules vs alternatives) | **Open**                                       | `30`, `36`, `37`, `39`, `42`                           |
| D    | `Dataset` naming consistency                                                                | Ratified                                       | `20`-`25`, `32`, `33`                                  |
| E    | Constraints model depth                                                                     | Deferred                                       | `11`, `10`, `13`, `18`, `32`                           |
| F    | Nesting shape rules (`R1`/`R2`/`R3`)                                                        | Ratified                                       | `26`, `32`, `22`-`24`                                  |
| G    | I/O transport and `semstrait-core::io` posture                                              | Ratified                                       | `31b`, `31`, `32`, `33`                                |
| H    | Canonical entity type set                                                                   | Ratified                                       | `18` (+ cascades)                                      |
| I    | Typed diagnostics + `tracing` observability                                                 | Ratified; cleanup still pending in older prose | `30`-`39`, selected `10`/`13`/`14`*/`15`/registry docs |
| J    | `14 §2` type-shape rebase                                                                    | **Ratified** 2026-05-18 — see item N for cascade | `14_expressions.md` (rebased twice: parameterized `Expr<L>` + leaf sets 2026-05-14; per-kind typed `SemanticLeaf` + `EntityRef`/`Access` retirement 2026-05-18) |
| K    | Relationship-block shape rebase (semantic-first; drop `directionality` + per-hop overrides; add `integrity` / `optional` / `cross_filter`; derive `JoinType` from `optional`; Joinset-local divergence via scope-local Relationship shadow) | Ratified 2026-05-12                            | `18 §2`, `16 §2`/`§4`/`§13`, `24 §2`/`§5`/`§7`/`§10`, `32 §1`, `33 §8` |
| L    | `semstrait-model` spec implementation — diagnostic primitives + `ExprSource` lift in `semstrait-core`; spec-aligned types, `parse` + `validate`, `SemanticModelLoader<F: SourceFs>`, `bon`-derived builders, Vec-backed root storage with uniform SR-3 / SR-E-3 dedup at `.build()`, reference YAML + JSON Schemas, README. Downstream crates (`semstrait-manifest`, `semstrait-api`, `semstrait-planner`, `semstrait`) tagged with migration TODOs; cascade tracked under `40_refactor_plan.md`. | **Complete** 2026-05-12 (`feature/spec-driven-dev`) | `31 §6`, `32`, `32b`, `crates/semstrait-core/`, `crates/semstrait-model/` |
| M    | Builder ergonomic facade — additive sugar layer (`32 §9.7.1` re-framed two-surface principle + new `32 §9.7.8`) on top of the primary `bon`-derived structural builders; full-coverage per-entity, per-container, and DataKind facade methods; cross-struct flatteners with read-modify-write semantics; `state_mod(vis = "pub")` mandate; primary-surface symmetry plurals on `Nested-*` and `SemanticModel`. | **Complete** 2026-05-13 (`feature/model-builder-fluent-facade`) | `32 §9.7.1` / `§9.7.6.1` / `§9.7.7` / `§9.7.8`, `crates/semstrait-model/src/{builder/data_kind.rs, builder/model.rs, entities/*}`, `crates/semstrait-model/README.md`, `crates/semstrait-model/tests/builder_facade.rs` |
| N    | Expression compile-pipeline cascade — `14b` (resolution algorithm) and `19` (two-phase flow / sugar contract / Phase B placement) merged into a single canonical compile-pipeline document at `19_expression_flow.md`; `14b` retired as forwarding stub. Vocabulary rebased to the typed-leaf shape from `14` (`Field` / `Dimension` / `Measure` / `Metric` / `Key` per-kind leaves; sugar accessors as `Option<XxxAccessor>` fields; `EntityRef` / `Access` / wrapping `Accessor` enum retired). Auto-mapping synthesis pre-step added at `19 §3.11`; `ColumnInSemanticExprUnderManualMapping` and `SemanticKindMismatch` `CompileError` variants added at `19 §8.1` (variant naming aligned post-item-Q rename). `35` (`semstrait-ir`) absorbs ownership of `Expr<L>` + leaf sets + per-kind accessor enums + `Parameter` + the `expr_fn` / `std::ops` / `ExprFunctionExt` DSL + `CanonicalFn` / `FunctionRegistry`; `31` (`semstrait-core`) shrunk to primitives + trait scaffolding (`Tree` / `ExprLeaf` / `Visitor` / `Rewriter`) + support enums + diagnostics + constraints + `io`. Cross-ref cleanup across `15` / `33` / `34` and 10+ adjacent docs rerouting `14b §...` → `19 §...`. `INDEX.md` and `STATUS.md` updated. | **Complete** 2026-05-18 | `14b_expression_resolution.md` (retired stub), `19_expression_flow.md` (merged + rebased), `14_expressions.md` (twice-refined), `35_semstrait_ir.md` (absorbed ownership), `31_semstrait_core.md` (shrunk), `15_mapping_and_binding.md` / `33_semstrait_manifest.md` / `34_semstrait_planner.md` (cross-refs updated), `INDEX.md` (updated) |
| O    | **Expression-spec consolidation pass** — implementation-readiness trim of the `14` / `14a` / `19` / `33` / `34` cluster. `14 §8` collapsed to a 5-line pointer to `19 §3`; `14 §12` "Forward Rebases Required" deleted (work already landed under item N); residual `14b` cross-refs across all foundations / apis / data-kinds / questions / `INDEX.md` retargeted to `19`; foundations stub `14b_expression_resolution.md` deleted; `questions/{open,closed}/14b_questions.md` retired with forwarding outcomes table. `14a` trimmed (ratification-cruft prose stripped; §5 BinaryOp lattice condensed; §4 catalog candidate-list prose tightened); **function-level `Additivity` ratified at `14a §3.6`** as `Additive | SemiAdditive { axes } | NonAdditive` (closes `[TD-REGISTRY-DETERMINISM]` ratification surface for function-level additivity). `19` substantially trimmed: BFS / Tarjan / structural-recurse / auto-mapping algorithm code → prose-with-rationale; Q-rationale prose removed; §3.6.2 type-inference table compressed to deltas vs `14a §3.4`; §3.7 reconciliation flattened from 5 sub-sub-sections to one; §8 error roster trimmed (kept §8.1 / `19 §8.2` pointer-to-`14a`; removed speculative `LiteralOverflow` / `LiteralPrecisionLoss` / `UnrepresentablePhysicalType` / `ShapeInferenceConflict` / `AdditivityMismatch`). **Ownership moves**: `Provenance` body → `[33 §6.3.1](apis/33_semstrait_manifest.md)`; `RequestDimensionRef` / `DimensionVariation` body → `[34 §3.10](apis/34_semstrait_planner.md)` (with `19 §6.2` consumer pointer); function-level `Additivity` definition → `14a §3.6` (with model-level pointer in `18 §5.2` flagged for `41_deprecations.md` rename). Line-count delta: `14` 794 → 743; `14a` 365 → 244; `19` 1384 → 849 (~535 lines / 38% reduction); `19 §8` 4 sub-tables → 4 focused tables. | **Complete** 2026-05-18 | `14_expressions.md`, `14a_function_catalog.md`, `19_expression_flow.md`, `18_entities.md`, `33_semstrait_manifest.md`, `34_semstrait_planner.md`, plus cross-ref cleanups in `15` / `16` / `21` / `22` / `23` / `24` / `25` / `36` / `39b` / `_drafts/34_simple_strategy.md` / `questions/{open,closed}/*` / `INDEX.md` |
| P    | **Expression architecture cleanup (first cascade)** — post-consolidation ref-fix pass. (1) Non-coercion / pass-through posture rule codified at `14 §5.4` (was unreachable at old `14 §5.6` after §5 restructure); ~15 dangling `14 §5.6` cross-refs retargeted across `10` / `11` / `13` / `14a` / `19` / `23` / registry / drafts. (2) Stale `14a §3.1` ref in `19 §10` retargeted to `14a §3.6`. (3) Aggregate synthesis from `(agg:, expr:)` documented at `32 §5.4`. (4) Option A (spec-as-written) confirmed per `_drafts/expr-architecture-research.md` — no `ExprBlock` parallel AST, no accessor-enum duplication; `Expr<L>` in `semstrait-ir`, `semstrait-model` depends on it. **Superseded-in-spirit by item Q** (2026-05-19): the first cascade left the trait family + structural-variant support enums + identifier carriers + narrow `*ErrorKind` enums in `semstrait-core`; item Q closes that gap and lands the full Option A direction. Item P remains valid for its specific in-scope ref-fix work. | **Complete** 2026-05-18 — superseded-in-spirit by Q 2026-05-19 | `14 §5.4` (new), `32 §5.4` (new), `10` / `11` / `13` / `14a` / `19` / `23` / `registry/functions_mapping.md` / `_drafts/34_unionset_strategy.md` (ref retargets) |
| Q    | **Expression architecture — second cascade (full Option A landing).** Closes the gap left by item P: every expression-tree-tied type moved out of `semstrait-core` and into `semstrait-ir`. **Moved to `semstrait-ir`:** the traversal trait family (`Tree`, `Visitor<N>`, `Rewriter<N>`, `ExprLeaf`); the structural-variant support enums (`BinaryOpKind`, `UnaryOpKind`, `AggregationOp`, `LikeKind`, `CastFailure`, `WindowFn`, `WindowFrame`, `WindowFrameKind`, `WindowBound`); `Literal` typed-literal carrier; the identifier carriers `ColumnRef` / `SemanticsName`; the narrow ir-emitted error kinds (renamed `ValidateErrorKind` → `ValidateError`, `CompileErrorKind` → `CompileError` for the moved types per scoped error-naming cleanup tied to this move). **`semstrait-core` post-cascade surface:** logical-type vocabulary (`DataType`, `Grain`, `TypeClass`, `Schema`, `SchemaColumn`); constraint-DSL toolkit; diagnostic primitives; `io` transport. **`ExprBlock` deleted as a named type** — `ExprSource::Block(Expr<L>)` carries `Expr<L>` directly via serde from `semstrait-ir`'s derives; `semstrait-model` owns the parser-side `Deserialize` impl for `ExprSource<L>`. **Downstream D.ii embeds renamed:** `model::ValidateError` embeds `Ir(ir::ValidateError)`; `manifest::CompileError` embeds `Ir(ir::CompileError)`; `CompileWarningKind` renamed to `CompileWarning`. Cascade landed across `14` / `14a` / `19` / `30` / `31` / `32` / `33` / `35` / `37` / `38` / `39` / `39b` / `INDEX.md`. **Transient naming asymmetry:** broader `*ErrorKind` enums (`ParseErrorKind`, `IrErrorKind`, `RegistryErrorKind`, `ModelBuildErrorKind`, `RepositoryErrorKind`, `AdaptErrorKind`, …) still carry the `Kind` suffix per `19 §9` `std::io::ErrorKind` convention; their global rename remains a separate post-v1 sweep. `_drafts/expr-architecture-research.md` deleted; `questions/open/31_questions.md` Q4 (`ExprBlock` exposure) moved to closed as moot. | **Complete** 2026-05-19 | `31_semstrait_core.md` (shrunk to non-expression vocabulary), `35_semstrait_ir.md` (absorbs full expression vocabulary + narrow errors), `14_expressions.md` §9 (crate placement rewritten; `ExprBlock` deleted from §6.1; trait/Rewriter signatures rebased to `ValidateError`), `32_semstrait_model.md` (`ExprSource::Block(Expr<L>)`; model-level `ValidateError` embeds `Ir(...)`; `semstrait-ir` declared as 2nd dep), `33_semstrait_manifest.md` (manifest-level `CompileError` embeds `Ir(...)`; declared 4-crate dep set), `19_expression_flow.md` (rename pass), `30_api_contracts.md` (stability table rewritten), `37_semstrait_catalog.md` / `38_semstrait_api.md` / `39_semstrait_facade.md` / `39b_facade_fluent_api.md` (rename + ownership reassignments), `INDEX.md` (concept-map rows added for trait family / support enums / identifier carriers / ExprSource / narrow errors), `questions/closed/31_questions.md` (Q4 closed-as-moot), `questions/closed/19_questions.md` (Q-EXPR-19-015b ref-fix `14a §3.1` → `§3.6`), `_drafts/expr-architecture-research.md` (deleted) |


---

## 3) Deferred topics

### 3.1 Constraints design

Status: deferred to dedicated session.

Working context:

- `[questions/deferred/11_questions.md](questions/deferred/11_questions.md)`

Resume points:

- `aggregation` sub-block semantics
- key naming choices (`aggregation` vs `aggregations`, `all` vs `all_of`)
- `constraints.filter` scope choice vs entity-level fields

---

## 4) Questions state snapshot

Question sidecars are stateful by directory:

- Active v1 backlog: `[questions/open/](questions/open/)`
- Ratified history: `[questions/closed/](questions/closed/)`
- Parked/post-v1: `[questions/deferred/](questions/deferred/)`

Current footprint after balanced pruning:


| Directory   | Files | Lines |
| ----------- | ----- | ----- |
| `open/`     | 23    | ~2580 |
| `closed/`   | 19    | ~1430 |
| `deferred/` | 18    | 797   |

(Approximate after this round; precise counts will refresh on the next sweep.)


Recent pruning moves:

- registry sidecars (`functions`, `join-types`, `temporal-shape`) moved to deferred;
- facade ergonomics sidecar moved to deferred;
- adapter/catalog operational-depth sidecars split into focused open + deferred remainder;
- stale numeric-code-era entries in `17/20/23/30/31/35` moved to closed;
- `21` (Q-DS-001) and `23` (Q-UNI-003 / -005 / -007 / -008 / -010 / -011) closed as part of variant-chapter rebases (2026-05-03).
- `22` (Q-GRN-001 / -002 / -005) closed as part of `Grainset` rebase (2026-05-03); `Q-COMP-006` opened to track post-rebase `16 §9.3` / `§10.5` cleanup.
- `Q-COMP-007` (directionality granularity) and `Q-COMP-017` (`join_type` YAML default) closed 2026-05-12 with the relationship-block rebase (item K) — authoring-layer `Directionality` retired; `join_type` no longer authored. `Q-COMP-016` (m:m policy) updated with current resolution status (directional `cross_filter` rejected on m:m; otherwise open). `24` ratifies the `JoinTypeOverrides` retirement; forward notes added to `Q-24-03` (cardinality override) and `Q-24-07` (per-hop filter override).

Focused v1 questions in `open/` should remain:

- architecture-impacting (`30`/`36`/`37` framing, strategy openness coupling),
- compile/plan correctness-critical (`15`, `32`, `33`, selected `22`/`23`),
- explicitly queued cross-stage primitive decisions (`38` Q-API-012 class).

Non-blocking ergonomics and deep adapter empirics should be deferred unless they directly block v1 implementation planning.

---

## 5) Last checkpoint (concise)

**Checkpoint type:** Raw-filter unified-pipeline implementation pass on `feature/raw_filtering` (2026-05-19). Spec was already consistent per `Q-RAW-FILTER` (closed in `questions/closed/34_questions.md`); the pass closed two pre-existing implementation gaps without any canonical type changes:

- `semstrait-api::RequestParser` lifted the `RawFiltersNotImplemented` stub; lowers `RawFilter { field, operator: String, value: serde_json::Value }` into the planner's existing `QueryFilter`; appends to `ResolvedQueryRequest.filters` alongside any other entries. Cross-reference invariant enforced at parse: a raw filter whose `field` names a `DataKindFilter` declared on the kind is rejected with the new `ParseError::RawFilterNamesNamedFilter { entity, name }` — that's what `RawQueryRequest.filters: Vec<String>` is for.
- `semstrait-planner::inject_user_filters` was rewritten around `query_filter_to_semantic_expr`, which now builds the predicate as a `SemanticExpr` over `EntityRef` field references (NOT `Expr::column`). The predicate then walks `ExprResolver::resolve_expr` — the same call `DataKindFilter` / `AggregationFilter` bodies use at scan level (see `crates/semstrait-planner/src/data_kind/plan_layers.rs` and `crates/semstrait-planner/src/decomposer.rs`). At the user-filter injection site we use an identity resolver (empty `PhysicalResolver` over the post-rename schema), so the observable output predicate stays `Column`-shaped — but the lowering surface is now shared U-1 at `SemanticExpr`.
- `In` / `NotIn` v1 lowering: OR-chain / AND-chain of equalities over the same `EntityRef`. Future canonical multi-arity tracked under `[TD-RAW-IN-LOWERING]`.

Tests added: `query_filter_to_semantic_expr` equivalence with author-written `SemanticExpr` (`Eq` + `In`); cross-reference rejection round-trip through `engine.with_model(yaml).explain(raw)`; happy-path round-trip showing a raw filter materialises as a `WHERE` predicate in the emitted SQL.

Docs updated: `crates/semstrait-api/README.md` drops the `stub - not yet wired` note and adds a _Filter surfaces_ section pinning the two-field invariant; `crates/semstrait-planner/README.md` adds a _Unified Filter Pipeline_ section pointing at `34 §3.3` / `§3.5` and `19 §7.1`.

**Prior checkpoint:** `semstrait-model` spec implementation — consolidation pass (item L closure) on `feature/spec-driven-dev` (2026-05-12). Builds on the W1-W5 baseline, archived in `[_archive/STATUS_HISTORY.md](_archive/STATUS_HISTORY.md)`.

Six post-W5 phases landed in this pass:

- **P1a — `ExprBlock` archive.** Typed expression AST moved to `expr_ast.rs` `#[doc(hidden)]`; `ExprSource::Declarative` carries `serde_yaml::Value` (opaque pass-through pending `19 §3` AST landing). `RelationshipId`, `JoinType::from_optional`, `PhysicalExpr` deleted as unused.
- **P1b — error-kind audit.** 12 stub variants pruned across `ParseErrorKind` / `ValidateErrorKind` / `CatalogsParseErrorKind`; 5 rules implemented (SR-8 `InvalidIdentifier`, SR-E-4 `RelationshipMissingCardinality`, SR-E-9 `MeasureMissingAgg`, SR-E-10 `SemanticsMissingDataType`, `SemanticsShadowRootPool` warning). SR-6 retired — per-variant chapters (`21 §7` / `22` / `23` / `24`) own required-extras enforcement via their own `VALID_E_2[1-4]xx` bands; the Grainset-leaf-temporal case is covered end-to-end by SR-E-8. SR-E-11 variant renamed `FilterWrongKind` → `WrongFilterError`; rule itself still pending ratification.
- **P2 — vocabulary cascade.** Semantic-side `column` → `field` rename across `KeyDecl`, `ForeignKeyDecl`, `JoinKeyExprPair`; `SemanticMappingBuilder::with_semantic(name, value)` added; `LiteralValue::Deserialize` widened to bare YAML scalars. `PartitionDef::column` kept (physical-side reference).
- **P3 — module splits + DRY.** `Diagnostic::map_kind` lifted from `semstrait-model` to `semstrait-core`; `validate.rs` and `data_kind/mod.rs` split into folder modules; `walk_complex` visitor extracted.
- **P4ab + P4c — bon migration + construction unification (D-7 / D-8 / D-10).** All builders (`Dataset`, `Grainset`, `Unionset`, `Joinset` ± Nested, `SemanticModel`) migrated to `bon`-derived typestate. Root storage is `Vec<(Location, T)>` per collection. SR-3 / SR-E-3 dedup runs uniformly at `.build()` time — single-file, code-built, and cross-source collisions surface the same diagnostic. `parse(yaml)` returns `SemanticModelBuilder` (caller chains `.build()`); `loader::merge_models` deleted; `YamlRoot::lower_into(source, builder)` is the cross-source append entry. `Duplicate*` variants moved from `ParseErrorKind` to `ValidateErrorKind` with `Vec<Location>` payloads.
- **P5 — polish.** `LiteralValue::Serialize` hand-rolled to emit single-key map form (parity with `Deserialize`'s tagged path); loader's `catalogs_loaded` placeholder dropped; `Diagnose::cause` delegation added on `ModelBuildErrorKind`.

Final gates: `cargo clippy -p semstrait-core -p semstrait-model --all-targets -- -D warnings` clean; `cargo test -p semstrait-model` 146 pass (was 124 at W5); `semstrait-manifest` baseline 220 (no regression).

Spec deltas reflected in this pass: `32 §2.1` / `§6` / `§9.1` / `§9.2` / `§9.5` and `18 §7.3` / `§11` (Duplicate* migration parse → validate; `parse` returns builder; SR-1 / SR-2 / SR-4 enforcement collapsed into `ParseErrorKind::UnknownField` via type-level absence + `deny_unknown_fields`; SR-E-* roster expanded under `ValidateErrorKind`; SR-6 retired with row marker; SR-E-11 kebab label `validate.filter-wrong-kind` → `validate.wrong-filter-error`).

For pass-by-pass chronology and prior long-form diffs, use:

- `[_archive/STATUS_HISTORY.md](_archive/STATUS_HISTORY.md)`

---

## 6) Next-session starting point

1. Read `[00_overview.md](00_overview.md)`, `[STATUS.md](STATUS.md)`, `[INDEX.md](INDEX.md)`, then `[foundations/19_expression_flow.md](foundations/19_expression_flow.md)` (single source of truth for the expression compile pipeline). For expression-vocabulary placement after the second cascade (item Q), `[apis/35_semstrait_ir.md](apis/35_semstrait_ir.md)` is now the **complete crate-of-record** for the trait family, support enums, identifier carriers, leaves, `Expr<L>`, accessors, `Parameter`, `FunctionRegistry`, and the narrow `ValidateError` / `CompileError`. `[apis/31_semstrait_core.md](apis/31_semstrait_core.md)` is the **non-expression** shared-vocabulary home only (logical types, diagnostics, constraints, `io`).
2. **`30 §6` typed-diagnostics framing pass.** Codify the Rust-encoding convention (numeric codes as adjacent comments on enum variants, NOT runtime fields) project-wide. Inventory affected docs (`30`, `34`, `36`, `37`, advisory-emitting Strategy chapters).
3. **Joinset (`24`) variant rebase.** Last DataKind chapter still unmoved; algorithm body extracts to `_drafts/34_joinset_strategy.md` sidecar mirroring `21`/`22`/`23`.
4. **Model-level `AdditivityType` → `Additivity` rename** (`[18 §5.2](foundations/18_entities.md)` cascade per `[41_deprecations.md](implementation/41_deprecations.md)`). Item O ratified the function-level enum in `14a §3.6`; the model-level enum carries the same variant set with extended `SemiAdditive { axes: Vec<SemanticsName>, strategy }` fields. The rename is a mechanical edit; cascade affects `18`, `32`, and downstream consumers.
5. Parallel-tracked: item C (adapter/catalog framing) across `30`/`36`/`37`/`39` — now joined by `[TD-30-ADAPTER-CAPABILITY]` (two-path adapter dispatch per Q-EXPR-19-002 closure); stale `CompositionKind` / `ComposedSemanticInterface` cleanup in `33`; `Q-COMP-006` deeper `16 §9.3` / `§10.5` cleanup; `[TD-19-ADDITIVITY-COMPOSITION]` composition-rule ratification depth beyond the current `19 §6.5` table; `[TD-19-ADVISORY-FIELDS]` payload schema at `34` rebase; `[TD-REQUEST-DIM-VARIATION]` final ratification of `RequestDimensionRef` shape per `34 §3.10`.

---

## 7) Session update rule

At end of each spec session:

1. Update this file with only state changes (phase, active items, deferred, snapshot, checkpoint).
2. Keep checkpoint concise; place long narratives in archive or commit history.
3. Propose updates for human approval before committing.