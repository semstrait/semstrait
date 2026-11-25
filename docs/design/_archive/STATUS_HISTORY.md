# STATUS History Archive

This file is the archival companion to `[../STATUS.md](../STATUS.md)`.

Purpose:

- preserve high-level session chronology without inflating active handoff context;
- keep `STATUS.md` short, scannable, and decision-operational.

For full historical detail, use git history:

- `git log --oneline -- docs/design/STATUS.md`
- `git show <commit>:docs/design/STATUS.md`

---

## Milestone timeline (condensed)

- **2026-04-17**: consolidation pass; canonical entities promoted to `foundations/18_entities.md`.
- **2026-04-27**: questions directory restructure and model/entity ratification cascades.
- **2026-04-29**: typed diagnostic-kind and `tracing` observability policy ratified across `30`-`39`; per-Q-ID split completed.
- **2026-04-30**: DataKind taxonomy thirteenth pass; major trim in `20_taxonomy.md`.
- **2026-05-03**: variant-chapter rebases — `Dataset`/`Unionset`/`Grainset` slim form; `UnionMode { All, Unique }` reconfirmed; `CompositionKind` shrunk to `{Joinset, Grainset}`; Grainset cross-grain LEFT OUTER JOIN composition ratified (G-2).
- **2026-05-11**: function-catalog `Additivity` field added per `19 §9.5`; two-source SoC ratified (function-level vs model-level).
- **2026-05-12**: expression-flow design closed and promoted to `foundations/19_expression_flow.md` (10 sections, ~640 lines); 32 closed clauses across Rounds 1–5; `KeyAccessor` added to mirror `DimensionAccessor`; reconciliation item J opened (`14 §2` rebase needed). Same day: relationship-block shape rebase (item K) ratified — semantic-first authoring, drop `directionality`/per-hop overrides, derive `JoinType` from `optional`.
- **2026-05-12**: `semstrait-model` spec implementation (item L, W1-W5 baseline) — diagnostic primitives + `ExprSource` lift in `semstrait-core`; spec-aligned types, `parse` + `validate`, `SemanticModelLoader<F: SourceFs>`, per-struct `bon` builders, reference YAML + JSON Schemas, README rewrite. Phase 2 audit passes (`cargo clippy -- -D warnings` clean; 124 tests pass on `semstrait-core` + `semstrait-model`). Downstream crates tagged with banner `TODO(refactor)` comments pointing at `40_refactor_plan.md`.
- **2026-05-12**: `semstrait-model` post-W5 consolidation (item L closure) — six cleanup phases: P1a (`ExprBlock` archive to `#[doc(hidden)] expr_ast.rs`; `RelationshipId` / `JoinType::from_optional` / `PhysicalExpr` deleted); P1b (12 error-variant prune + 5 SR-* / SR-E-* rules implemented; SR-6 / SR-E-11 spec gaps recorded); P2 (`column` → `field` rename across Keys / FK / JoinKey; `SemanticMappingBuilder::with_semantic`; bare-scalar `LiteralValue` parse); P3 (`Diagnostic::map_kind` lift to core; `validate.rs` and `data_kind/mod.rs` split into folder modules); P4ab + P4c (bon migration of all builders; Vec-backed root storage with uniform SR-3 / SR-E-3 dedup at `.build()` (D-10); `parse` returns `SemanticModelBuilder`; `loader::merge_models` deleted; `Duplicate*` variants moved from `ParseErrorKind` to `ValidateErrorKind`); P5 (`LiteralValue::Serialize` emits single-key map form; `Diagnose::cause` delegation on `ModelBuildErrorKind`). 146 tests pass (was 124). Spec drift reflected in `32 §2.1` / `§6` / `§9.1` / `§9.2` / `§9.5`.
- **2026-06-15**: item Y — Vocabulary Evolution / public-internal expression split ("Arch III"), **inverts item Q**. Spec-doc cascade across `00`/`13`/`14`/`14a`/`19`/`15`/`31`/`32`/`35`/`INDEX.md`: authoring vocabulary (`Expr<L>` families, two leaf sets, accessors, support enums + `Literal`, carriers, `CanonicalFn`-name, `DataType`/`Grain`, `ExprLeaf`) relocates to `semstrait-model`; `Tree`/`Visitor`/`Rewriter` + `ValidateError` to `semstrait-common`; `semstrait-ir` keeps only the resolved concrete `PhysicalExpr` (+ `Window`/`Parameter`) + registry; `model→ir` Cargo edge inverted to `ir→model`. V1–V10 in `_research/vocabulary/RATIFICATION_LOG.md`. Spin-outs: `Q-MODEL-009` (OQ-C, `SemanticsName` collapse) open; `OQ-D` (`expr_fn` home) deferred; V9 (named per-node schema) deferred. Crate-code execution is the next phase (spec intentionally leads code until ports land).
- **Current consolidation pass**: index-primary navigation restoration, `00_overview` compression, `STATUS` thin-handoff conversion, balanced v1 open-question pruning.

---

## Archival policy

- Keep this file append-only for milestone summaries.
- Keep deep, line-by-line chronology in commit history rather than active status docs.
- `STATUS.md` should never contain diff dumps or multi-page replay narrative.