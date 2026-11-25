---
id: 32-model-expr-vocabulary
status: ratified
version: 2.1
impl: implemented
---

# Model Expression Vocabulary (Stub)

The expression vocabulary — all structural variants, leaf sets, support enums, identifier carriers, and the `Literal` carrier — is **ratified in [`../../foundations/14_expressions.md`](../../foundations/14_expressions.md)**. `semstrait-model` is the crate-level home and re-exports it for consumption by `semstrait-ir` (item Y / V1).

## Vocabulary

The single shared `model::Expr<L>` family owns:

- All structural variants **including `Window`** (A2 — carried by every instantiation, author-rejected on the author path).
- Leaf sets: `UnresolvedSemanticLeaf`, `UnresolvedPhysicalLeaf { Column, Literal }` (no `Parameter` — the sole leaf-set differentiator stays ir-only).
- Type aliases: `UnresolvedSemanticExpr`, `UnresolvedPhysicalExpr`.
- Support enums: `BinaryOpKind`, `UnaryOpKind`, `AggregationOp`, `AggregateKind`, `LikeKind`, `CastFailure`.
- Window support enums (A2): `WindowFn`, `WindowFrame`, `WindowFrameKind`, `WindowBound`.
- Carriers: `Literal`, `ColumnRef`, `SemanticsName`.
- Function identity: `CanonicalFn` name.

`ir::PhysicalExpr` is an alias to `Expr<PhysicalLeaf>` — the resolved physical form; `semstrait-ir` imports this crate's vocabulary and adds the `PhysicalLeaf` set and `Parameter` variant. See `14 §3` for the full variant roster, `14 §6.1` for the authoring field shape (bare `Expr<L>` at every expression-bearing site, no `ExprSource<L>` wrapper — removed per doc 45), and `14 §6.2` for the parse-site dispatch via the field-level adapter (`parse_block<L>`).

## Testing Strategy

Expression vocabulary testing is covered by the shared expression test suite described in [`../../foundations/14_expressions.md`](../../foundations/14_expressions.md). This stub leaf is a crate-home marker; the behavioral contract lives in doc 14.
