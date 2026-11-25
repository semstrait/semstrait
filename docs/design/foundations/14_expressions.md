---
prereqs: [00]
status: superseded
---

# 14_expressions (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`14_expressions/`](14_expressions/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [14-layered-model](14_expressions/layered-model.md) — The layered expression model — one structural tree parameterized by leaf type
- [14-tree-trait](14_expressions/tree-trait.md) — The Tree traversal trait — universal visitor/rewriter contract
- [14-expr-structural-variants](14_expressions/expr-structural-variants.md) — The Expr<L> structural-variant taxonomy and recursion model
- [14-physical-leaves](14_expressions/physical-leaves.md) — The two physical leaf sets — author-facing and resolved-IR, split by Parameter admissibility
- [14-semantic-leaves](14_expressions/semantic-leaves.md) — The UnresolvedSemanticLeaf per-kind typed set and sugar accessor carriers
- [14-type-aliases](14_expressions/type-aliases.md) — The authoring and resolved type aliases — model::UnresolvedSemanticExpr, model::UnresolvedPhysicalExpr, ir::PhysicalExpr
- [14-accessor-sugar](14_expressions/accessor-sugar.md) — Per-entity accessor sugar — the four accessor enums and Window-lowering contract
- [14-parameter](14_expressions/parameter.md) — The Parameter compile-emitted placeholder, ParameterKey closed set, and non-coercion posture
- [14-yaml-authoring](14_expressions/yaml-authoring.md) — The YAML authoring surface — bare Expr<L> fields, block form, reserved tags, bare-identifier resolution
- [14-site-shape-gates](14_expressions/site-shape-gates.md) — Per-site shape gates — scalar/Boolean/aggregate-admitting expression requirements by authoring site
- [14-crate-placement](14_expressions/crate-placement.md) — Expression-layer crate placement — semstrait-common, semstrait-model, semstrait-ir, semstrait-manifest ownership
