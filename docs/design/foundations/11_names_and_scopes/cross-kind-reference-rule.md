---
id: 11-cross-kind-reference-rule
status: ratified
version: 2.0
impl: implemented
---

# Cross-Kind Reference Rule

Cross-kind reference rule — Relationship-required invariant, compile pre-validation, plan walk resolution. This leaf owns the cross-kind reference invariant; [lookup algorithm](lookup-algorithm.md) owns the general name resolution mechanics.

## 9. Cross-Kind Reference Rule

When an expression, Filter, or Constraint in top-level DataKind X references a Semantics whose binding lives only in another top-level DataKind Y, a `Relationship` from X to Y (possibly via intermediate DataKinds) must exist in the Model.

### 9.1 The rule

For every reference `R` in the Kind scope of X naming Semantics `S`:

1. If `S` is bound in X's own Binding subtree, the reference is resolved locally — no Relationship needed.
2. If `S` is NOT bound in X, `compile` searches the Relationship graph for a path from X to a top-level DataKind in which `S` is bound.
3. If exactly one such path exists (or Model's Relationship rules select a unique preferred path — `16`), the reference is considered valid at compile time, and the path is recorded for `plan`.
4. If no path exists, or the path is ambiguous by `16`'s rules, the reference fails compile with `CompileError::UnresolvedCrossKindReference`.

### 9.2 Compile pre-validation, plan resolution

- **Compile** pre-validates path existence as a Precondition (§12). This is an I5 invariant: no unresolved references leak past compile.
- **Plan** walks the recorded Relationship path at plan time to produce `PlanNode`s that materialize the cross-kind resolution. The actual graph-walking algorithm is ratified in `16`.

### 9.3 Within-kind references

Inside a top-level DataKind's subtree (its Kind scope plus all nested scopes and Binding scopes underneath), every Semantics named in an expression resolves through the DataKind's own interface and Binding structure. No Relationship is needed — the references never cross a top-level DataKind boundary.

## Testing Strategy

- Relationship path existence validation (N-C5: cross-kind reference requires path)
- Path uniqueness (N-C6: ambiguous paths rejected by `16`'s disambiguation)
- Compile pre-validation (§9.2: path-existence checked at compile, recorded for plan)
- Plan walk resolution (§9.2: planner walks recorded Relationship path)
- Within-kind fast path (§9.3: same-DataKind references resolve locally, no Relationship needed)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
