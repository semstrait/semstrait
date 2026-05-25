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

**What remains active**

- Adapter/catalog framing reconciliation (item C).
- Residual cross-doc vocabulary cleanup where retired error-code language still appears.
- v1 backlog trimming in open question sidecars.
- Variant chapter rebases — `Joinset` (`24`) pending (the last remaining; `33`/`34` come after). The 2026-05-12 relationship-block rebase (item K) addresses the Relationship-side authoring shape; the full Joinset rebase (algorithm body extracts, etc.) is still pending and tracked separately.
- **Relationship-block rebase (item K, 2026-05-12).** Authoring shape moved to semantic-first (`cardinality` + `integrity` + `optional` + `cross_filter`); `directionality:` and the `JoinTypeOverrides` / `HopPosition` per-hop override surface retired. `JoinType` is derived at compile from `optional` per `18 §2.9`. Validation rules SR-E-13 / SR-E-14 added. Joinset-local divergent semantics via scope-local `Relationship` shadow (`18 §2.10`, `16 §13.3`). Cascaded into `18`, `16`, `24`, `32`, `33`.
- Algorithm-body sidecars (`_drafts/34_simple_strategy.md`, `_drafts/34_unionset_strategy.md`, `_drafts/34_grainset_strategy.md`) pending lift into `34_semstrait_planner.md §<XStrategy>` when the planner doc opens its Strategy chapter.
- Deeper structural cleanup of `16 §9.3` / `§10.5` / `§13` (inert post-Unionset-retirement) parked behind new `Q-COMP-006` for a Round-4 framework cleanup pass.
- Stale `CompositionKind` / `ComposedSemanticInterface` references in `33` pending cleanup at that chapter's rebase.
- **`14_expressions.md §2` rebase** — `14 §2` defines `SemanticExpr` / `PhysicalExpr` as newtype wrappers around a shared `Expr` enum; `19 §3` ratified two distinct enums linked by a shared `Expr` trait. `19` declares the scoped extension per `00 §8`; `14 §2` needs a rebase pass to align (authoritative-for line and code samples). Added as reconciliation item J.
- **Persisting open clauses** — none. Both post-promotion follow-ups (Q-EXPR-19-001, Q-EXPR-19-002) closed 2026-05-12; `questions/open/19_questions.md` retired (empty). All `19`-tagged ratifications live in `questions/closed/19_questions.md`.
- **Function catalog (`14a §3.1`) extension** — `Additivity` field added to `FunctionSpec` per `19 §9.5`. Closes `[TD-REGISTRY-DETERMINISM]`. Two-source SoC (function-level vs model-level `18 §5.2 AdditivityType`) ratified 2026-05-11; composition rule pending under `[TD-19-ADDITIVITY-COMPOSITION]`.
- **Model-level `AdditivityType` rename** — existing `18 §5.2` `AdditivityType` aligns with new unified `Additivity` enum at refactor time. Variants map 1:1 (`Full` → `Additive`, `Semi(SemiAdditivity)` → `SemiAdditive { axes }`, `Non` → `NonAdditive`). Flagged in `41_deprecations.md` for landing during planner-doc rebase.
- **Advisory field payload `[TD-19-ADVISORY-FIELDS]`** — exact context fields on `PLAN_W_2101 LossyReaggregation { data_kind, .. }` beyond `data_kind` deferred to single-pass ratification at `34_semstrait_planner.md` Strategy chapter rebase.
- **Per-DataKind advisory specialisation `[TD-19-ADVISORY-SPECIALISATION]`** — flag for future split if a `LossyReaggregation` root cause structurally diverges per DataKind (currently unified under `PLAN_W_2101`).
- **`30 §6` typed-diagnostics framing codification** — Rust-encoding convention (numeric `*_W_*` / `*_E_*` codes as adjacent comments on typed-enum variants for grep-ability, NOT runtime fields) is project-wide; lift into `30 §6` next session.
- **Inline request-time filters (`11 §6.4.2`, 2026-05-22).** Scoped extension under §6.4 Filter: request-scope anonymous `{field, operator, value}` predicate. Normalized at request resolution into a canonical boolean `SemanticExpr`; rides the same Phase B placement engine as a named DataKindFilter (`19 §7.1` `filters.<f>.expr` row). Validation owned by the API/request-resolution layer (unknown field / bad operator / value type mismatch). Cross-refs added in `19 §7.1` (placement) and `INDEX.md` (concept map). Spec hook for the code-side wiring tracked separately.

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
| J    | `14 §2` type-shape rebase to align with `19 §3` (enum + shared trait, not newtype wrapper)  | **Open** (scoped extension declared in `19`)   | `14_expressions.md §2`                                 |
| K    | Relationship-block shape rebase (semantic-first; drop `directionality` + per-hop overrides; add `integrity` / `optional` / `cross_filter`; derive `JoinType` from `optional`; Joinset-local divergence via scope-local Relationship shadow) | Ratified 2026-05-12                            | `18 §2`, `16 §2`/`§4`/`§13`, `24 §2`/`§5`/`§7`/`§10`, `32 §1`, `33 §8` |


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

**Checkpoint type:** Expression-flow design closed and promoted (`foundations/19_expression_flow.md` landed, 2026-05-12).

Delivered outcomes:

- **`foundations/19_expression_flow.md`** ratified in clean spec form (10 sections, ~640 lines). Owns the two-phase pipeline, type architecture (enum + shared `Expr` trait — refines `14 §2`), sugar contract (Family A fold / Family B `Access`), `Accessor` enums + v1 entries, `Parameter` placeholder mechanism, per-site `expr:` shape gates, Phase B placement (filter / group_by / computed-Dim / Metric / `Additivity` / advisory), typed `Diagnostics<PlanErrorKind>` channel with unified `PLAN_W_2101`, Rust encoding convention. Code samples flagged as illustrative-not-frozen per implementation discipline.
- **`questions/closed/19_questions.md`** populated with 32 closed clauses across Rounds 1–5 (concise tabular form; no rationale narrative — that lives in git history).
- **Post-promotion follow-up closures (2026-05-12, same-day).** Q-EXPR-19-001 reframed and closed: keys-as-special-dimension-type — added `KeyAccessor` enum mirroring `DimensionAccessor` 1:1 + `Accessor::Key(_)` variant + symmetric §5.2 worked snapshot. `[TD-19-KEY-SUGAR]` retired (architecturally moot as originally framed; replaced by spec content). Q-EXPR-19-002 closed: two-path adapter architecture (capability-driven dispatch); Path A reserved for canonical-consumer engines (no v1 deliverable), Path B is v1 concrete; substantive design content lives in `30` / `36` under `[TD-30-ADAPTER-CAPABILITY]`. `questions/open/19_questions.md` retired (now empty).
- **`INDEX.md`** updated: foundation `19` row added; topic-routing row added; concept-map rows added for `SemanticExpr` / `PhysicalExpr`, `Accessor` / `Parameter`, `DimensionRef`, `Additivity`; question-snapshot counts bumped.
- **`00_overview.md §4.1`** updated: concept rows added for `SemanticExpr` / `PhysicalExpr`, `Accessor` / `Parameter`, `DimensionRef`, `Additivity`; `Expr` row updated to multi-home (`14` + `19`).
- **`_drafts/19_expression_flow.md`** retired per single source-of-truth discipline.
- **Reconciliation item J opened**: `14 §2` rebase needed to align newtype-wrapper shape with `19 §3` enum+trait shape (scoped extension declared in `19` per `00 §8`).
- Tech-debt markers carried forward: `[TD-19-PHYSICAL-FOLD]`, `[TD-19-ADDITIVITY-COMPOSITION]`, `[TD-19-ADVISORY-FIELDS]`, `[TD-19-ADVISORY-SPECIALISATION]`, `[TD-19-MIXED-FILTER-OR]`, `[TD-30-ADAPTER-CAPABILITY]`.
- `[TD-19-KEY-SUGAR]` retired (superseded by `KeyAccessor` spec content in `19 §3.3` / §5.2).

Prior checkpoint (variant chapter rebases — Dataset / Unionset / Grainset) archived in `_archive/STATUS_HISTORY.md` per single-checkpoint convention.

For pass-by-pass chronology and prior long-form diffs, use:

- `[_archive/STATUS_HISTORY.md](_archive/STATUS_HISTORY.md)`

---

## 6) Next-session starting point

1. Read `[00_overview.md](00_overview.md)`, `[STATUS.md](STATUS.md)`, `[INDEX.md](INDEX.md)`, then `[foundations/19_expression_flow.md](foundations/19_expression_flow.md)`.
2. **`14 §2` rebase** (reconciliation item J). Align `14 §2`'s `SemanticExpr` / `PhysicalExpr` newtype-wrapper shape with the `19 §3` enum + shared trait form. Update `14`'s authoritative-for line; refresh code samples; cross-ref `19` for the canonical form. Should be a small targeted edit.
3. **`30 §6` typed-diagnostics framing pass.** Codify the Rust-encoding convention (numeric codes as adjacent comments on enum variants, NOT runtime fields) project-wide. Inventory affected docs (`30`, `34`, `36`, `37`, advisory-emitting Strategy chapters).
4. **Joinset (`24`) variant rebase.** Last DataKind chapter still unmoved; algorithm body extracts to `_drafts/34_joinset_strategy.md` sidecar mirroring `21`/`22`/`23`.
5. Parallel-tracked: item C (adapter/catalog framing) across `30`/`36`/`37`/`39` — now joined by `[TD-30-ADAPTER-CAPABILITY]` (two-path adapter dispatch per Q-EXPR-19-002 closure); stale `CompositionKind` / `ComposedSemanticInterface` cleanup in `33`; `Q-COMP-006` deeper `16 §9.3` / `§10.5` cleanup; `[TD-19-ADDITIVITY-COMPOSITION]` composition-rule ratification; `[TD-19-ADVISORY-FIELDS]` payload schema at `34` rebase.

---

## 7) Session update rule

At end of each spec session:

1. Update this file with only state changes (phase, active items, deferred, snapshot, checkpoint).
2. Keep checkpoint concise; place long narratives in archive or commit history.
3. Propose updates for human approval before committing.