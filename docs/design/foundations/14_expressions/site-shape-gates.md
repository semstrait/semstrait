---
id: 14-site-shape-gates
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: §7 site catalog; parse/construct-time gate checks; Column-under-auto-mapping synthesis per §8
---

# Per-Site Shape Gates

This leaf defines the per-site shape gates that govern which expression shapes (scalar, Boolean, aggregate-admitting) are legal at which authoring sites, and the `Column`-under-auto-mapping admissibility rule. Shape gates are enforced at parse and construction time; the `Column`-under-manual-mapping rejection occurs at compile. See [yaml-authoring.md](yaml-authoring.md) for the parse-site catalog and [semantic-leaves.md](semantic-leaves.md) for the `Column` leaf structure.

## 7. Per-Site Shape Gates

Different authoring sites require different expression *shapes* — scalar / Boolean / aggregate-admitting. The gate is a property of the site, not of the expression type.

The site catalog (per-element shape and parse-site dispatch):

| Site | Parses to | Shape required | Aggregate-admitting? |
|---|---|---|---|
| `measures.<m>.expr` | `SemanticExpr` | scalar | no — aggregation carried by `agg:` per `[18 §5.2](../18_entities.md)` |
| `measures.<m>.filters[].expr` | `SemanticExpr` | Boolean | no — scalar predicate inlined into the aggregate's `filter` |
| `metrics.<m>.expr` | `SemanticExpr` | scalar | no — `agg:` (optional) at top-level; `expr:` is a scalar formula over already-aggregated values |
| `metrics.<m>.filters[].expr` | `SemanticExpr` | Boolean | no (compile-split per metric semantics) |
| `dimensions.<d>.expr` (computed) | `SemanticExpr` | scalar | no |
| `filters.<f>.expr` (DataKind-level) | `SemanticExpr` | Boolean | yes — HAVING-style predicates may reference aggregated values |
| `extras.semantic_mapping.<x>.expr` | `PhysicalExpr` | scalar | no |

**Structural shape gates** enforced at parse / construction time:

- Author-written `Aggregate { ... }` syntax inside `expr:` is **rejected at all sites except `filters.<f>.expr`**. Aggregation is carried by the structurally separate `agg:` tag on Measures and Metrics per `[18 §5.2](../18_entities.md)`.
- A typed leaf carrying `accessor: Some(_)` whose lowered shape contains `Aggregate` or `Window` is gated against sites whose required result is scalar/Boolean and not aggregate-admitting. The check is on the *lowered* shape, not on syntactic surface — sugars carry their own admissibility metadata.
- `filters[].expr` is admitted only on `measures.<m>` and `metrics.<m>`. No `keys` member-level filter slot. No `dimensions.<d>.filter` — DataKind-level filtering uses the `filters:` block.

The full mechanics of how Phase-B placement consumes these shape gates live in `[19](../19_expression_flow.md)`.

## 8. Compile-Time Resolution (pointer)

`SemanticExpr` → `PhysicalExpr` lowering is owned by `semstrait-manifest::compile`. The full algorithm — substep order, per-leaf-kind substitution rules, cross-DataKind path resolution, cycle detection, type inference, Semantics-boundary reconciliation, per-`(Semantics, Binding)` keying of `ResolvedExprTable`, auto-mapping synthesis pre-step, and the `Column`-under-manual-mapping + `SemanticKindMismatch` error rules — lives in `[19 §3](../19_expression_flow.md)`.

**Type-level postcondition (upheld here by `PhysicalLeaf`'s variant set).** `PhysicalExpr` carries no `Field` / `Dimension` / `Measure` / `Metric` / `Key` leaves and no typed leaves carrying `accessor: Some(_)`. Compile rewrites every such leaf; `PhysicalLeaf`'s structural shape makes the postcondition unforgeable (per §3.7).

## Testing Strategy

- **Site catalog coverage:** §7 site catalog is exhaustively covered — every site parses to the expected leaf family (`SemanticExpr` / `PhysicalExpr`), every site enforces the expected shape (scalar / Boolean / aggregate-admitting).
- **Aggregate syntax rejection:** `Aggregate` syntax inside `expr:` is rejected at all sites except `filters.<f>.expr`. Parse-time error for non-aggregate-admitting sites.
- **Sugar accessor shape gates:** Typed leaves with `accessor: Some(_)` whose lowered shape contains `Aggregate` / `Window` are rejected at scalar-only sites. Lowered-shape check, not syntactic surface.
- **Column-under-auto-mapping synthesis:** Compile synthesizes `SemanticMapping` entries for `Column` leaves under `semantic_mapping: auto`. Pre-step before resolution proper.
- **Column-under-manual-mapping rejection:** Compile rejects `UnresolvedSemanticLeaf::Column` under manual mapping with `ColumnInManualMapping` error.
- **Compile postcondition:** Every `PhysicalExpr` leaving compile carries no semantic references (`Field` / `Dimension` / `Measure` / `Metric` / `Key`), no `accessor: Some(_)`.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
