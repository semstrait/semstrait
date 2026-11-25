---
id: 19-error-model
status: ratified
version: 2.1
impl: implemented
---

# Error Model

This leaf defines the compile-stage CompileError variants for expression resolution (EXPR_E_02xx sub-range), including resolution-stage errors, function-resolution errors, boundary-reconciliation warnings, and fail-fast policy. See [phase-a-resolution.md](phase-a-resolution.md) for the resolution algorithm that emits these errors. <!-- cross-cut: none -->

## 8. Error Model

All expression compile-pipeline errors surface as `Diagnostic<CompileError>` per `[10 §5](../10_resolution_pipeline.md)` and `[30 §5](../../apis/30_api_contracts.md)`. Numeric codes are spec cross-reference indices; runtime identification is by variant identity.

### 8.1 Resolution-stage errors

| Variant | Code | When |
|---|---|---|
| `UnknownReference { name, scope }` | `EXPR_E_0201` | Identifier at a typed-semantic-leaf or `Field` site does not resolve in any visible scope per `[11 §11.1](../11_names_and_scopes.md)`. |
| `NoRelationshipPath { from, to }` | `EXPR_E_0202` | Cross-DataKind BFS exhausted without reaching the target (§3.4.3). |
| `CyclicReference { cycle }` | `EXPR_E_0203` | Reference-graph Tarjan SCC detected a cycle (size > 1 or self-loop) (§3.5). |
| `UnresolvedColumn { name, binding }` | `EXPR_E_0204` | `Column(name)` references a name not in the binding's `PhysicalSource` schema (§3.3.2.b). |
| `AmbiguousRelationshipPath { from, to, paths }` | `EXPR_E_0205` | BFS found two or more shortest-length paths (§3.4.3). |
| `TypeInferenceFailure { node, reason }` | `EXPR_E_0206` | Per-node rule could not derive a concrete type (untyped `Null` at boundary, mismatched `Case` / `Coalesce` branches, non-`Boolean` logical operand). |
| `ColumnInSemanticExprUnderManualMapping { binding, location }` | `EXPR_E_0207` | `UnresolvedSemanticLeaf::Column` authored under manual `semantic_mapping` (§3.11). |
| `SemanticKindMismatch { authored_kind, registered_kind, name, location }` | `EXPR_E_0208` | Typed leaf's variant tag disagrees with the registry's kind for `name` (§3.11). |

### 8.2 Function-resolution errors

Delegated to `[14a §8](../14a_function_catalog.md)`: `UnknownFunction`, `FunctionArityMismatch`, `NoMatchingSignature`, `ReservedTagCollision`.

### 8.3 Boundary-reconciliation warning

`CompileError::NarrowingCast { inferred, declared }` (`EXPR_W_0201`) — emitted by §3.7 when boundary reconciliation wraps the root in a narrowing `Cast`. Compile succeeds.

semstrait carries a single `CompileError` enum: warnings are `Severity::Warning` variants accumulated in the `Diagnostics<CompileError>` stream — there is no separate `CompileWarning` type. The `EXPR_W_*` codes live at the render layer, not on the variant.

### 8.4 Fail-fast policy

Phase A is fail-fast per `[00 §9](../../00_overview.md)` **I12**: the first error terminates `compile`. Warnings (`EXPR_W_*`) accumulate alongside errors in the `Diagnostic` stream per `[10 §5](../10_resolution_pipeline.md)` without aborting.

---

## Testing Strategy

- EXPR_E_0201-0208 error code coverage: verify each variant (`UnknownReference`, `NoRelationshipPath`, `CyclicReference`, `UnresolvedColumn`, `AmbiguousRelationshipPath`, `TypeInferenceFailure`, `ColumnInSemanticExprUnderManualMapping`, `SemanticKindMismatch`) is emitted for corresponding malformed inputs.
- Fail-fast policy enforcement: verify the first resolution error terminates the compile stage; verify no partial `ResolvedExprTable` is persisted on error.
- Boundary-reconciliation warning emission: verify `NarrowingCast` warning is emitted when declared `data_type` narrows inferred type; verify it includes source location and type details.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
