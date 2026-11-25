---
id: registry-functions-mapping-intro
status: ratified
version: 1.0
impl: drift
---

# Functions Mapping — Intro & Taxonomy

This leaf introduces the function-mapping catalog structure, defines the four rewrite tiers (Name-only / Name-remap / Structural / Unsupported), and specifies the two-layer adapter rewrite boundary (PlanBuilder vs Dialect).

## 1. Rewrite Tier Taxonomy

Every canonical function is classified, per engine, into one of four rewrite tiers. The tier governs where and how the adapter transforms the canonical `Expr` tree on its way to engine-native output.

| Tier | Description | Example |
|---|---|---|
| **Name-only** | Engine uses the same function name, same signature, same semantics. No rewrite — adapter emits the canonical name verbatim. | `upper` → `upper` (all three engines). |
| **Name-remap** | Engine uses a different function name, but same signature and same semantics. Adapter performs a string-level name substitution, no tree reshaping. | `position` → `strpos` (DataFusion, DuckDB). |
| **Structural** | Engine requires an expression-tree transformation — different arity, reordered args, substitution for a `BinaryOp`, wrap with another call, etc. | `date_add(d, i)` → `d + i` (DataFusion interval arithmetic). |
| **Unsupported** | Engine has no equivalent and no emulation path the adapter is willing to carry. Hard error at `adapt` time per `14a §6` via `AdaptErrorKind::UnsupportedFunction { name, engine }`. | (none in canonical set v1 — intersection-only population eliminates canonical-level unsupported by construction; appears only for adapter-extended entries consumed cross-engine.) |

The canonical catalog's Q10 intersection posture (per `14a §4.1`) guarantees that every canonical entry resolves to one of {Name-only, Name-remap, Structural} on each of the three first-class engines — `Unsupported` never arises at the canonical layer for DataFusion / Spark / DuckDB. Adapter-extended entries (`summary-gaps.md §12`) may legitimately resolve to `Unsupported` on engines that do not carry the adapter's registration.

## 2. The Two Rewrite Layers

Canonical `Expr` trees are transformed into engine-native output at two distinct adapter layers. Per `14a §6` / `apis/36_semstrait_adapter/rewrite-pipelines.md`, each function is handled at **exactly one** layer — the choice is driven by the node's IR shape.

| Layer | Runs | Input → Output | Handles |
|---|---|---|---|
| **PlanBuilder layer** (`EngineAdapter::rewrite_expr`) | During plan construction, before SQL emission | `Expr` tree → `Expr` tree | `FunctionCall` nodes (looked up via the per-adapter `FunctionRewriter` rewrite table), plus any dedicated-variant rewrites that convert to `FunctionCall` (e.g. DataFusion `RegexpExtract` → `array_element(regexp_match(...), idx+1)`). |
| **Dialect layer** (`SqlDialect` trait) | During SQL emission | `Expr` tree → SQL text | Dedicated `Expr` variants with dialect methods (`Like` / `ILike` / `RegexpMatch` / `RegexpExtract` / `DateTrunc` / `Cast`) rendered directly to engine-native SQL without first being reshaped into a `FunctionCall`. |

**The layering rule.** PlanBuilder-layer rewrites produce `Expr` — they never know the dialect. Dialect-layer rewrites produce SQL — they never reshape the `Expr` tree (only render). A PlanBuilder rewrite MAY convert a dedicated variant into a `FunctionCall`; after that conversion the node is indistinguishable from any other `FunctionCall` and is rendered by the generic scalar-call path. This happens for DataFusion's `RegexpExtract` only; all other dedicated variants render at the Dialect layer on all three engines.

## Testing Strategy

Rewrite-tier behavioral gates:

- **Tier definitions validated**: Each tier (Name-only / Name-remap / Structural / Unsupported) has a clear behavioral contract. The check validates that every canonical function + engine pair resolves to exactly one tier.
- **PlanBuilder/Dialect layering contract**: PlanBuilder rewrites produce `Expr` (never know the dialect), Dialect rewrites produce SQL (never reshape `Expr`). The check validates no function is handled at both layers.
- **No overlap**: Each function is rewritten at exactly one layer (PlanBuilder OR Dialect, never both). Cross-reference `summary-gaps.md §13` per-adapter rewrite summary to validate completeness.

See [../../testing/50_behavioral_contract_gates.md](../../testing/50_behavioral_contract_gates.md) for the shared behavioral harness.
