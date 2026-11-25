---
id: 15-expression-strata
status: ratified
version: 2.1
impl: implemented
---

# Expression Strata

The three-stratum expression model (14/19/15) partitions expression semantics across three layers: UnresolvedSemanticExpr (layer 1, logical), PhysicalExpr (layer 2, lowered SQL-equivalent), and compile-time mechanics (layer 3 — `path_token` is the v1 inhabitant). This document is a navigational anchor defining the stratum boundary. See [metadata-extraction](metadata-extraction.md) for `path_token` runtime mechanics.

> **Three-stratum expression model.** `15` is the authoritative anchor for the layered model that spans the expression docs:
>
> 1. `**model::UnresolvedSemanticExpr` (layer 1, concept owned by `14`, homed in `semstrait-model`; resolved by `19 §3`)** — the public authoring-surface form (`model::Expr<UnresolvedSemanticLeaf>`); a logical expression tree that operates on Semantic names (`@semantics_ref`); supports logical constructs (`CASE WHEN`, etc.) and is the **input** to **constant folding / partial evaluation** when one or more leaves resolve to compile-time-known values (e.g. metadata literals, declared `Literal`-mapped Semantics). Shorthand `UnresolvedSemanticExpr` below.
> 2. `**ir::PhysicalExpr` (layer 2, concept owned by `14`, homed in `semstrait-ir`)** — the resolved internal-IR form: a lowered SQL-equivalent expression tree over canonical types and functions (`14a`); expression nodes only — no compile-time mechanics, no logical-only constructs that did not survive lowering. `semstrait-ir` defines **no** semantic-expression alias; the unresolved form is `model::UnresolvedSemanticExpr` (above).
> 3. **Compile-time mechanics (layer 3, owned by `15`)** — non-expression logic that runs purely during `compile` to produce values for the SemanticManifest. v1 host: `path_token` (§8). Layer-3 outputs feed back into layer-1 / layer-2 as resolved literals (`SemanticMappingValue::Metadata` recipe → eagerly-resolved per-source `LiteralValue` on `ResolvedPhysicalSource.metadata_values`); they are **never** registered as `14a` functions and **never** appear as `PhysicalExpr` variants.
>
> The fold from layer 1 to layer 2 is where partial evaluation lands: a `model::UnresolvedSemanticExpr` whose subtree depends only on layer-3 literals (e.g. `CASE WHEN @dataset_name = 'asd' THEN @col1 ELSE @col2 END` where `@dataset_name` is a metadata-bound Semantic) is rewritten at the lowering boundary into a flat `Column` projection per source, so plan-time scans never need to evaluate the conditional. The fold itself — the `compile` entry point that folds `model::UnresolvedSemanticExpr` → `ir::PhysicalExpr` — is homed in `semstrait-manifest` per `[14 §9.4](../14_expressions.md)`; the fold language is owned by `[19 §4.1](../19_expression_flow.md)`; `15` ratifies the layer-3 inputs that fuel it.

## Testing Strategy

- Navigational leaf — covered by doc-space-check structural gates
- Expression-stratum boundary tests live in 14 (layer 1 → 2 lowering) and 19 (substitution)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
