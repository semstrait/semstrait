---
doc: design/questions/open/32_questions
status: Living
purpose: Round-1 open items raised against `apis/32_semstrait_model.md`
---

# 32 — Open Questions

> Two questions remain open: Q-MODEL-005 (expression-surface parse-site audit table) and Q-MODEL-008 (`functions:` YAML block scope). Closed items moved to [`../closed/32_questions.md`](../closed/32_questions.md); deferred items in [`../deferred/32_questions.md`](../deferred/32_questions.md). Q-MODEL-009 (`SemanticsName` collapse) was closed as resolved-by-V10 in the item-Y ratification session — see [`../closed/32_questions.md`](../closed/32_questions.md).

---

## Q-MODEL-005 — Expression-surface parse-site table completeness

> **Vocabulary refined by doc 45 (2026-06-22).** The `ExprSource` wrapper is removed; each "parse site" below is now a **bare `Expr<L>` field** with a field-level friendly serde adapter, and parse dispatch is `parse_block<L>` (not `parse_semantic` / `parse_physical` on a wrapper). The *set* of sites enumerated here is unchanged — read "`ExprSource` site" as "authoring `Expr<L>` field." The audit-table question itself remains open.

**Question.** The model surface contains many sites that parse an `ExprSource` (Dimension `expr:`, Measure `expr:`, Metric `expr:`, Filter `expr:`, etc.). Is there a canonical enumeration of every site that parses an `ExprSource`, so that every site can be covered by an integration test?

**Expanded scope (post-2026-04-17 canonical-entity closure).** The `18`-entity ratifications added several new parse sites that must be included in any audit table:

- **`18 §2.2` (Relationship.keys).** `JoinKeyExprPair { from: ExprSource, to: ExprSource }` — both sides are `UnresolvedSemanticExpr` parse sites; authored at root `relationships:` AND `JoinsetBody.relationships`.
- **`18 §2.2` (Relationship.filter).** Optional residual predicate — `UnresolvedSemanticExpr` parse site.
- **`18 §1.3` (reference-site `expr:` override).** Every `- ref: <name>` entry on a DataKind's `SemanticInterface` may carry a local `expr:` override (Dimension / Measure / Metric) — `UnresolvedSemanticExpr` parse site.
- **`18 §7.1` (DataKindFilter.expr).** `UnresolvedSemanticExpr` parse site — unchanged in shape from the pre-ratification Filter but now ratified canonically.
- **`18 §7.2` (AggregationFilter.expr).** `UnresolvedSemanticExpr` parse site — evaluated inside the aggregation wrapper at compile lowering.
- **`18 §10.1` (SemanticMapping `Expr` variant).** `UnresolvedPhysicalExpr` parse site — distinct from the `Column` / `Literal` variants that dispatch without parsing an expression.

**Refs.**

- `14 §4.2` — `ExprSource` dispatch lives in `semstrait-model` parse sites.
- `14 §2.2` / `§2.3` — `SemanticExpr` vs `PhysicalExpr` admissibility rules.
- `18 §4` / `§6` / `§7` — Dimension / Measure / Metric `expr:` sites (all `UnresolvedSemanticExpr`).
- `18 §2.2` — `Relationship.keys[].from` / `.to` and `Relationship.filter` (all `UnresolvedSemanticExpr`).
- `18 §1.3` — `ref`-site `expr:` override (`UnresolvedSemanticExpr`, inherits carrier rules).
- `18 §7` — Filter taxonomy `expr:` sites (`UnresolvedSemanticExpr`).
- `18 §10.1` — `SemanticMappingValue::Expr(UnresolvedPhysicalExpr)` (`UnresolvedPhysicalExpr`).

**Why.** Makes the wrapper-typing contract auditable and an integration-test target: every `ExprSource` site MUST dispatch to exactly one of `parse_semantic` / `parse_physical`, else new sites silently default to one wrapper.

**Default.** Not tabulated; rule is implicit across `32 §5.1` and every `18 §N` `expr:` field. An integration test in `crates/semstrait-model/tests/` enumerates the sites programmatically. Tracked as `[TD-EXPR-PARSE-SITE-AUDIT]`.

**Trigger.** Re-open to add an exhaustive parse-site appendix to `18` when the next parse-site extension lands (most sites are owned by `18`).

---

## Q-MODEL-008 — `functions:` YAML block scope

**Question.** `32 §4.5` allows authors to declare adapter-contributed function extensions at the model scope via a `functions:` top-level block. `31 §5.8` already provides `RegistryExtension` impls at the adapter-crate level. Are both needed? If yes, how do per-model declarations interact with global `RegistryExtension` impls?

**Refs.**

- `14a §7.1` — adapter-contributed registry entries via `RegistryExtension`.
- `31 §5.8` — `RegistryExtension` trait surface; folded at `function_registry()` init.
- `32 §4.5` — `functions:` YAML block with `FunctionExtension` entries.
- `14a §7.2` — collision handling: `AdapterFunctionShadowsCore`, `AdapterFunctionCollision`.

**Why.** Per-model `functions:` is an escape hatch for pre-built distributions, but doubles collision rules at `compile` and conflicts with process-global function identity (`14a §2.2`); the v1 use case is speculative.

**Default.** Both supported in v1; collision handling at `compile` per `31 §5.8`. Tracked as `[TD-MODEL-FUNCTIONS-BLOCK]`.

**Trigger.** Re-open when a real per-model use-case surfaces; if none by mid-v1, deprecate and remove the `functions:` block in a v2 MINOR (only `RegistryExtension` impls remain).

---

*Cross-references in this document are by section (e.g. `32 §10.3`, `31 §8.3`). No code-path references are used, per `00 §8`.*
