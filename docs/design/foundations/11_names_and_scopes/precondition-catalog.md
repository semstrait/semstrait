---
id: 11-precondition-catalog
status: ratified
version: 2.0
impl: implemented
---

# Precondition Catalog

Name-related Precondition catalog — N-V1 through N-C9, mapping to typed error variants. This leaf owns the Precondition catalog and error-variant mapping; validation mechanics are owned by their implementing crates.

<!-- cross-cut: 32-model-parse-validate -->

## 12. Precondition Catalog

Name-related Preconditions and their stage of execution.

### 12.1 Run by `validate` (structural, accumulate)

| ID | Rule | What fails |
|---|---|---|
| N-V1 | `name` is syntactically well-formed | empty, whitespace-only, or invalid per identifier grammar (§13) |
| N-V2 | Per-level uniqueness (§3.3) | duplicate name within a single Semantics list |
| N-V3 | Structural-label uniqueness within parent (§10.1) | two nested children share a label |
| N-V4 | No Semantics at nested-kind scope (§10) | `dimensions:` / `measures:` / `metrics:` / `filters:` / `keys:` present in a nested-kind block |

### 12.2 Run by `compile` (catalog / registry-dependent, fail-fast)

| ID | Rule | What fails |
|---|---|---|
| N-C1 | Shape unification across occurrences (§5.1) | any shape field disagrees across occurrences of a name |
| N-C2 | Element-type consistency | a name appears as Dimension in one occurrence and Measure in another |
| N-C3 | Kind-scope interface membership (§11.1 step 2) | an `expr` or `constraint` references a name not in the enclosing Kind's interface |
| N-C4 | Binding-required rule | a Semantics is registered but has no physical binding anywhere (no `SemanticMapping` entry and no `expr`) |
| N-C5 | Cross-kind reference: path exists (§9.1) | reference crosses to Semantics bound only in another DataKind with no Relationship path |
| N-C6 | Cross-kind reference: path unique (§9.1) | multiple Relationship paths exist and `16`'s disambiguation cannot pick |
| N-C7 | Key member is a Dimension (§6.5) | a Key names a non-Dimension Semantics |
| N-C8 | Key member is in this DataKind's interface | a Key names a Semantics not in the declaring DataKind's interface |
| N-C9 | Foreign Key target exists with Relationship (§6.5.1) | a `foreign` Key references a non-existent target DataKind / target Key, or no `Relationship` connects the two |

### 12.3 Mapping to typed error variants

Each Precondition ID maps to a specific `ValidateError` or `CompileError` variant produced by the owning crate. The typed enums are ratified in `32` (`semstrait-model`) and `33` (`semstrait-manifest`). Table below is the authoritative mapping; downstream crates must not introduce alternative variants for these checks.

| Precondition | Stage | Typed variant |
|---|---|---|
| N-V1 | validate | `ValidateError::InvalidIdentifier { name, location }` |
| N-V2 | validate | `ValidateError::DuplicateSemanticName { list, name, first_at, second_at }` |
| N-V3 | validate | `ValidateError::DuplicateStructuralLabel { parent, name, first_at, second_at }` |
| N-V4 | validate | `ValidateError::SemanticsAtNestedScope { parent, block, location }` |
| N-C1 | compile | `CompileError::SemanticShapeConflict { name, field, occurrences }` |
| N-C2 | compile | `CompileError::SemanticElementTypeConflict { name, types_at }` |
| N-C3 | compile | `CompileError::SemanticNotInInterface { name, dataset, referenced_at }` |
| N-C4 | compile | `CompileError::SemanticWithoutBinding { name, first_occurrence }` |
| N-C5 | compile | `CompileError::UnresolvedCrossKindReference { name, from_dataset, bound_in }` |
| N-C6 | compile | `CompileError::AmbiguousCrossKindPath { name, from_dataset, candidate_paths }` |
| N-C7 | compile | `CompileError::KeyMemberNotDimension { dataset, key, member, element_type }` |
| N-C8 | compile | `CompileError::KeyMemberNotInInterface { dataset, key, member }` |
| N-C9 | compile | `CompileError::ForeignKeyTargetInvalid { dataset, key, reason }` where `reason ∈ { TargetDatasetUnknown, TargetKeyUnknown, NoRelationship }` |

Each variant produces a `Diagnostic` via the `StageError → Diagnostic` conversion at pipeline boundaries (`10 §5`). The `Diagnostic::code` string follows `10 §5.1`'s kebab-case convention (e.g. `compile.semantic-shape-conflict`).

## Testing Strategy

- Precondition N-V1/N-V2/N-V3/N-V4 validation (validate stage; §12.1)
- Precondition N-C1 through N-C9 validation (compile stage; §12.2)
- Precondition→error-variant cross-cut table enforcement (§12.3: ValidateError variants in 32-model-parse-validate; CompileError variants in 33)
- Stage assignment integrity (validate = structural, accumulate; compile = catalog-dependent, fail-fast)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
