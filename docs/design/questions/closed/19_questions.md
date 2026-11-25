---
doc: design/questions/closed/19_questions
status: Closed
purpose: Resolved questions originally raised against `foundations/19_expression_flow.md`
---

# 19_expression_flow — Closed Questions

Historical ratifications for `foundations/19_expression_flow.md`. Each entry records what was decided; rationale lives in commit history.

---

## Round 1 — Type architecture and pipeline shape

| ID | Topic | Decision |
|---|---|---|
| Q-EXPR-A | `Parameter` carve-out for Request-dependent values | Branch (ii) — compile-emitted typed `Parameter` placeholders in `PhysicalExpr`; bound at Phase B from `Request`. Closed parameter set; not author-extensible. |
| R-1 | n-ary `Expr` variant shape | Struct-shaped variants for n-ary cases (`Case.whens: Vec<(Self, Self)>`, `FunctionCall.args: Vec<Self>`, etc.). |
| R-2 | Public `resolve` surface | Single public method `SemanticExpr::resolve(self, ctx) -> Result<PhysicalExpr, CompileError>`. Substeps internal; not exposed. |
| R-3 | `Parameter` typing | Typed `ParameterKey` enum; mandatory `data_type: DataType` at compile. |
| R-4 | `Expr` trait shape | `Expr` is a trait (children iteration, transform, inferred type); modelled on TreeNode. |
| R-5 | `Aggregate` admission | `Aggregate` admitted in both `SemanticExpr` and `PhysicalExpr`; operand type differs by phase. Planner lifts to `PlanNode::Aggregate` at Phase B. |
| M-1 | Per-site shape gate | Implicit gate via lowered category; no per-sugar allow-list. |
| M-2 | Sugar admission rule | Sugar is `SemanticExpr`-only; admitted by site if its lowered category satisfies §4's required result. |
| Q-T-2 | `Accessor` enum shape | Per-entity-typed: `Accessor::Measure(MeasureAccessor)` / `::Dimension(DimensionAccessor)` / `::Metric(MetricAccessor)`. |
| B + traits | Two-enum + shared trait | Separate `SemanticExpr` / `PhysicalExpr` enums linked by shared `Expr` trait. Refines `14 §2`'s newtype-wrapper shape. |
| Naming | `Kind`-suffix policy | Drop `Kind` suffix on new coinages (`Accessor`, `Parameter`); retain on existing ratified names (`DataKind`, `*ErrorKind`). |
| Avg posture | `Avg` as canonical aggregate | `Avg` is a canonical `AggregationOp` per `14a`, not sugar. No internal rewrite to `Sum / Count`. Lossy combinations surface as advisories per §7.6, not refusals. |

## Round 2 — Resolution semantics

| ID | Topic | Decision |
|---|---|---|
| Q-EXPR-19-005 | Fold-language scope | Maximal logical / binary: comparison, logical, null check, arithmetic, composite, structural `Case`, literal `Cast`, ANSI-strict `Like`. `FunctionCall` / regex / UDF excluded for v1. |
| Q-EXPR-19-006 | Substep order | Eliminate `Access` → fold + partial-eval → translate. Folding precedes translation so metadata-driven branch elimination collapses subtrees before `PhysicalExpr` construction. |
| Q-EXPR-19-007 | Metadata literal injection | Lives in fold substep (substep 2); resolved by `Q-EXPR-19-006` reorder. |
| Q-EXPR-19-008 | Per-`Binding` materialisation | Each `Binding`'s `PhysicalExpr` is independently folded against its own metadata literals; multi-source Datasets produce per-`Binding` distinct results. |
| Foldable trait scope | `Foldable` impl reach | `SemanticExpr: Foldable` load-bearing; `PhysicalExpr: Foldable` no-op default for v1. Tagged `[TD-19-PHYSICAL-FOLD]`. |

## Round 3 — Worked example

| ID | Topic | Decision |
|---|---|---|
| §5.3 worked example | Category I metadata fold | End-to-end walk of two-source Dataset with `year_dir` metadata literal; per-`Binding` divergence demonstrated. |

## Round 4 — Phase B placement

| ID | Topic | Decision |
|---|---|---|
| Q-EXPR-19-009a | Filter canonical form | `Aggregate.filter` slot canonical; emits `FILTER (WHERE)` on supporting engines; adapter rewrites to `CASE WHEN` elsewhere. |
| Q-EXPR-19-009b | Metric `filter:` scope | Dim / Key refs push into every constituent's `Aggregate.filter`; constituent Measure / Metric refs route to HAVING-like wrapper; non-constituent Measure / Metric refs are compile errors `EXPR_E_xxxx MetricFilterReferencesNonConstituent`. |
| Q-EXPR-19-009c | DataKind-level mixed-reference filter | AND-decomposable mixed-scope splits transparently to WHERE-part + HAVING-part; non-AND mixed-scope (`OR` / `NOT` across scopes) is compile error `COMP_E_xxxx MixedScopeFilterUndecomposable`. Tagged `[TD-19-MIXED-FILTER-OR]` for v2 HAVING fallback. |
| Q-EXPR-19-009d | Per-element `filter:` admission | `dimensions.<d>.filter:` and `keys.<k>.<member>.filter:` both structurally rejected. Admitted only on Measures, Metrics, and DataKind-level `filters:` block. |
| Q-EXPR-19-014 | Temporal rollup mechanism | Structured `DimensionRef { name, variation }` on `Request`; `DimensionVariation::{ None, Temporal { grain } }` for v1; CLI `name.grain` sugar; native model grain default; legacy `temporal_rollup` retired. |
| Q-EXPR-19-015a | `Additivity` shape | 3-class: `Additive` / `SemiAdditive { axes: Vec<DimensionAxis> }` / `NonAdditive`. |
| Q-EXPR-19-015b | `SemiAdditive::axes` declarability | Function-level `axes` hardcoded in `14a §3.6` per built-in aggregate; UDF authoring deferred post-v1. Model-level `axes` declarable on Measure / Metric per `18 §5.2`. |
| Q-EXPR-19-018a | Dim refs in Metric `expr` | Admitted per `18 §5.2`; evaluates as per-group value (post-aggregate context). Plan-time validation rejects requests missing the Dim with `PLAN_E_xxxx MetricRequiresDimensionInRequest`. |
| Q-EXPR-19-018b | Metric → Metric chain depth | Unbounded with DAG semantics; cycles rejected at compile (`EXPR_E_xxxx MetricCycle { path }`); no depth bound. |
| Q-EXPR-19-019 | Constituent column naming | Canonical Measure / Metric name (no anonymous-aggregate names — `expr:` syntax never carries inline aggregates). Dedup key is Measure / Metric name. Adapter output carries author-visible names. |
| Q-EXPR-19-021 | Advisory channel migration | Retire `tracing::warn!` for semantic warnings; structured `Diagnostics<PlanErrorKind>` emitting `PLAN_W_*` codes per `30 §6`. `tracing::*` retained for system-level observability only. |
| Q-EXPR-19-022 | `Avg` cross-grain JOIN advisory placement | Rule in `19 §7.5` / `§7.6` (function-tag-derived from `Additivity::NonAdditive` × cross-grain plan shape); emission in `34` Strategy code; `22 §4.3` / `24 §<cross-grain>` cross-reference `19`. Not Grainset-specific. |
| Q-EXPR-19-023 | Cross-DataKind advisory unification | Single canonical `PLAN_W_2101 LossyReaggregation { data_kind, .. }`. Retires draft `PLAN_W_2101 LossyMultiSourceReaggregation` and ratified Unionset `PLAN_W_2302`. Per-DataKind specialisation flagged `[TD-19-ADVISORY-SPECIALISATION]`. Field payload beyond `data_kind` flagged `[TD-19-ADVISORY-FIELDS]` for `34` Strategy rebase. |
| Rust encoding convention | Numeric codes in Rust API | Spec cross-reference indices, NOT runtime data fields. Adjacent comments on typed-enum variants. Codified project-wide at `30 §6`. |

## Round 5 — `MetricAccessor` v1

| ID | Topic | Decision |
|---|---|---|
| Q-EXPR-19-003a | `MetricAccessor` v1 surface | Mirrors `MeasureAccessor` 1:1: `Previous`, `Next`, `Lag(u32)`, `Lead(u32)`, `Delta`, `PercentChange`. |
| Q-EXPR-19-003b | `Delta` / `PercentChange` retention | Retained on both `MeasureAccessor` and `MetricAccessor`. Sugar-on-sugar resolved by fixpoint Family B elimination in `resolve` substep 1. |
| Q-EXPR-19-003c | Accessor variant naming | Same variant names across `MeasureAccessor` / `MetricAccessor` (no `Metric*` prefix). Type-system disambiguation per Q-T-2 handles unambiguity. |

## Post-promotion follow-ups (2026-05-12)

| ID | Topic | Decision |
|---|---|---|
| Q-EXPR-19-001 | Sugar admission for Key entity refs | Key is a special Dimension type for sugar purposes; admit Family B accessor on `EntityRef::Key(_)` symmetric with Dimension. New `KeyAccessor` enum mirrors `DimensionAccessor` 1:1 (`First`, `Last`, `Lag(u32)`, `Lead(u32)`); new `Accessor::Key(KeyAccessor)` variant. Available inside aggregate-context sites (Measure / Metric / Filter `expr:`) wherever `DimensionAccessor` is available; `keys.<k>.<member>.expr` continues to reject Window output (symmetric with `dimensions.<d>.expr`). `[TD-19-KEY-SUGAR]` retired — superseded by spec content in §3.3 / §5.2. |
| Q-EXPR-19-002 | Adapter-injectable rewrites | Two-path adapter architecture: adapter declares supported artifact / plan output type via capability surface; pipeline dispatches by capability. Path A (canonical-plan-to-canonical, e.g. Substrait-consumer) — adapter injected at `plan` stage to rewrite functions / plan signatures; produces engine-compatible canonical artifact. Path B (canonical-plan-to-engine artifact, v1 sole concrete path) — adapter transforms canonical plan post-`plan`. Architectural design lives in `30` / `36` chapters; `19` carries cross-refs only. Substantive trait surface + dispatch logic flagged `[TD-30-ADAPTER-CAPABILITY]`. |

## Retired

| ID | Reason |
|---|---|
| Q-EXPR-19-004 | Superseded by `Q-EXPR-19-022` (consolidated into Round 4 advisory placement clause). |
