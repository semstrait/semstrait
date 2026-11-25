---
id: 19-expression-gates
status: ratified
version: 2.1
impl: implemented
---

# Expression Gates

This leaf defines per-site expression shape gates — scalar / Boolean / aggregate admission rules for Measure, Metric, Dimension, and filter contexts. These gates enforce structural shape requirements at parse and construction time.

## 5. Per-Site `expr:` Shape Gates

`14 §7` defines which sites carry `UnresolvedSemanticExpr` versus `UnresolvedPhysicalExpr`. This chapter ratifies the **shape gate** — what each site requires of `resolve`'s output:

| `expr:` site                       | Required result | Aggregate-function-call syntax in `expr:` |
|------------------------------------|-----------------|--------------------------------------------|
| `measures.<m>.expr`                | scalar          | no — aggregation is carried by the separate `agg:` tag (`18 §5.2`) |
| `measures.<m>.filters[].expr`      | Boolean         | no — scalar predicate; conditional aggregation per §6.1 |
| `metrics.<m>.expr`                 | scalar          | no — `agg:` (optional) at top-level; `expr:` is a scalar formula over already-aggregated values |
| `metrics.<m>.filters[].expr`       | Boolean         | no (compile-split per §6.1) |
| `dimensions.<d>.expr` (computed)   | scalar          | no |
| `filters.<f>.expr`                 | Boolean         | yes — HAVING-style predicates may reference aggregated values |
| `keys` members                     | n/a in v1       | no per-member `expr:` authoring slot is ratified (`18 §9`) |
| `relationships.<r>.keys[].{from,to}` | scalar        | no — equi-join operand; lowers per §3.12 (bare name → semantic id; computed → synthesised `Field`) |
| `relationships.<r>.filter`         | Boolean         | no — residual join predicate; lowers to a pooled `PhysicalExprId` (§3.12) |
| `extras.semantic_mapping.<x>.expr` | scalar          | no (parses to `PhysicalExpr`) |

**Structural shape gates** enforced at parse / construction time:

- Author-written `Aggregate { ... }` syntax inside `expr:` is **rejected at all sites except `filters.<f>.expr`**. Aggregation is carried by the structurally separate `agg:` tag on Measures and Metrics per `[18 §5.2](../18_entities.md)`. A Measure `agg: sum, expr: amount` resolves to `Aggregate { op: Sum, args: [Column("amount")], ... }` at Phase A, where the `Aggregate` node is *synthesised* by `agg:` and `expr:` together, not by author-written aggregate-function call syntax.
- A typed leaf carrying `accessor: Some(_)` whose lowered shape contains `Aggregate` or `Window` is gated against sites whose required result is scalar/Boolean and not aggregate-admitting. The check is on the *lowered* shape, not on syntactic surface — sugars carry their own admissibility metadata.
- `filters[].expr` is admitted only on `measures.<m>` and `metrics.<m>`. No `keys` member-level filter slot. No `dimensions.<d>.filter` — DataKind-level filtering uses the `filters:` block per §6.1.

---

## Testing Strategy

- Per-site shape-gate enforcement: verify Measure/Metric `expr:` admits aggregate-rooted expressions; verify Dimension `expr:` rejects aggregate-rooted expressions; verify filter contexts admit Boolean-typed scalar expressions.
- Aggregate-function-call admission: verify `Aggregate` nodes with valid `CanonicalAggFn` are accepted; verify unrecognized aggregate functions produce `UnsupportedAggregation` error.
- Structural rejection at parse/construction: verify malformed expressions (e.g. aggregate in Dimension `expr:`) are rejected before reaching Phase A resolution.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
