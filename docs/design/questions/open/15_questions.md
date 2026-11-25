---
doc: design/questions/open/15_questions
status: Living
purpose: Open questions surfaced while drafting `foundations/15_mapping_and_binding.md`
depends-on:
  - foundations/15_mapping_and_binding.md
  - foundations/13_types_and_grain.md
  - foundations/14_expressions.md
  - foundations/19_expression_flow.md
  - apis/30_api_contracts.md
  - apis/33_semstrait_manifest.md
  - apis/37_semstrait_catalog.md
---

# Open Questions — `foundations/15_mapping_and_binding.md`

> **Reconciliation (Phase-3 / 2026-04-27 consolidation).** When this doc says `ColumnMapping` / `ColumnMappingValue` / `ColumnMappingValue::Computed`, read **`SemanticMapping` / `SemanticMappingValue`** with the v1 4-variant roster `{Column(String), Literal(LiteralValue), Expr(PhysicalExpr), Metadata(MetadataDimensionRecipe)}` per [`../../foundations/18_entities.md §10`](../../foundations/18_entities.md). The earlier `[TD-MAP-METADATA-FOLD]` tech-debt marker is **resolved (2026-04-27)** — `Metadata` is restored as a distinct 4th variant; v1 metadata extraction scope is **path-only**; partition extraction is deferred to v2. The `Computed` variant was renamed to `Expr`.

> Round-1 framing for the four live items below; closed items moved to [`../closed/15_questions.md`](../closed/15_questions.md); deferred items in [`../deferred/15_questions.md`](../deferred/15_questions.md). Each entry restates the question, lists its ratified references, and records the Round-1 default `15` currently uses. Entries migrate out of this file as later docs (`16`, `33`, `37`, per-DataKind `21`–`25`) confirm or amend `15`'s defaults.

---

## Q-MAP-001 — `BindingId` uniqueness — MOOTED → see `closed/15`

> **MOOTED by the id-first rework (STATUS item U.2).** `BindingId` is eliminated: bindings are identified by a deterministic content-derived `EntityId` (UUIDv5, `33 §9.1`), which is globally unique and cross-run/cross-edit stable — answering the original per-vs-cross-manifest scope question by construction (cross-manifest comparison by binding `EntityId` is meaningful for unchanged content). Full resolution recorded in [`../closed/15_questions.md`](../closed/15_questions.md).

---

## Q-MAP-003 — Compile-synthesis of derived-Measure `SemanticMapping` entries

**Question.** When a `Measure(Count, DerivesFrom(Key))` is declared per `11 §8.4`, must the author write a `SemanticMapping` entry for it, or should `compile` synthesize `SemanticMappingValue::Expr { expr: Count(Column(key_col)) }` from the Constraint automatically?
**Why.** A derived Measure is logically "already defined" by the Constraint, but implicit synthesis complicates error messages and the "Model equals flattened SemanticManifest" mental model.
**Current position.** Compile-synthesis (`15 §5.6` / `§10` step 4): the authored `SemanticMapping` may omit derived-Measure entries; `compile` fills them in before the completeness check. `32` (ratified) keeps derivation at compile, not parse, consistent with this default.
**Trigger.** Promote `15 §5.6` from "Proposed (Round 1)" to ratified, or reopen if a parse-time materialization stance emerges that would move the synthesis to a `SemanticModel`-level operation.

---

## Q-MAP-006 — `ResolvedColumnMapping.computed` storage: duplicate or alias? — CLOSED (2026-05-28)

> **Moved to [`../closed/15_questions.md`](../closed/15_questions.md), then superseded by STATUS item V (2026-05-29).** Originally settled by C11 + C12 (split typed pools). Item V makes the manifest **physical-only** — `ManifestExpressions { physical: BTreeMap<PhysicalExprId, ManifestExpression{ expr, layer }> }` (`33 §7.2`); the `semantic` pool and `SemanticExprId` are dropped. The duplicate-vs-alias question dissolves with a single pool. See `closed/15` for the full note.

---

## Q-MAP-010 — Nullability mismatch: warning or error? — v2 conversation

**Question.** `15 §9.4` treats a non-nullable declared Semantics bound to a nullable physical column as a **warning** (`COMP_W_0306 NullableSourceForNonNullableSemantics`). Should it be an **error**?
**Why.** Type-system rigor argues for error; but source-reported nullability is often conservative (Parquet `optional` for an always-populated column) and the runtime engine enforces on actual nulls.
**Current position.** Warning (`15 §9.4`, Round-1 default). Promotion to error is explicitly a v2 conversation.
**Trigger.** Reopen at the v2 ratification pass if early-usage feedback shows authors consistently hitting runtime nullability errors after ignoring the warning; promotion would carry a migration note.
