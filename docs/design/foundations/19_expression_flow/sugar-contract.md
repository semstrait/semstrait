---
id: 19-sugar-contract
status: ratified
version: 2.1
impl: implemented
---

# Sugar Contract

This leaf defines the sugar elimination contract — Family A constant folding / partial evaluation for metadata-static expressions, and Family B per-kind accessor lowering to Window subtrees. Sugar elimination runs as a sub-step of Phase A resolution. See [phase-a-resolution.md](phase-a-resolution.md) for the overall resolve algorithm.

## 4. Sugar Contract

All sugars are `UnresolvedSemanticExpr`-only and desugar at compile time. The pipeline distinguishes two sugar families. (Parse-time *tag* normalization — `if:`→`case:`, `count_distinct: x`→`count: { args: [x], distinct: true }`, `not:` verbatim — is a separate, model-parser concern that adds no surface variants per `14 §6.4` / `STATUS.md` item Y / V8; it is upstream of the two compile-time families below.)

### 4.1 Family A — Constant folding / partial evaluation

Family A has **no AST variant**. Author writes plain `Case` / `BinaryOp` / `FunctionCall` / `Like`; the fold substep (§3.1 substep 3) collapses subtrees whose value is fully determined by Manifest-static state (per-`Binding` metadata literals, source-tagged Dimensions). Partial folds leave a residual that translation passes through to `PhysicalExpr`.

**Fold language (v1).** Reduction applies to nodes whose operands fold to a foldable value (`Literal` or metadata-substituted typed leaf):

| Class      | Members                       | Folding rule                                                                                |
|------------|-------------------------------|---------------------------------------------------------------------------------------------|
| Comparison | `=` `!=` `<` `>` `<=` `>=`    | both sides foldable → `Literal(bool)`                                                       |
| Logical    | `AND` `OR` `NOT`              | short-circuit; partial residual otherwise                                                   |
| Null check | `IsNull` `IsNotNull`          | metadata-substituted leaf non-null → `Literal(false)` / `Literal(true)`                     |
| Arithmetic | `+` `-` `*` `/` `%`           | numeric foldable → `Literal(N)`                                                             |
| Composite  | `IN` `NOT IN` `BETWEEN`       | desugar to comparison + logical, then fold                                                  |
| Structural | `Case`                        | short-circuit on first true `when`; drop false-`when` branches                              |
| Cast       | `Cast(Literal, T)`            | literal cast applied if successful; failures resolve per `on_failure`; column cast deferred |
| Pattern    | `Like(_, Literal(_))`         | ANSI-strict canonical (`%` zero-or-more, `_` one char); `LikeKind` is the operator discriminator `{ Like, NotLike, ILike, NotILike }` per `35 §3.4`; case-sensitive for `Like` / `NotLike` |

**`Like` canonicalisation.** Bracket classes / POSIX classes / `ILike` / `RLike` / regex extensions are **not** in v1 fold scope. Adapters emitting to engines with looser defaults (e.g. MySQL collation-driven case-folding) compensate during `PhysicalExpr` → engine-AST translation.

**Out of v1 fold scope.** `FunctionCall` (no purity flag in `14a §3.1` yet — `[TD-REGISTRY-DETERMINISM]`); regex operators; user-defined functions.

**Per-`Binding` materialisation.** Each `Binding`'s `PhysicalExpr` is independently folded against its own metadata literals; multi-source Datasets produce per-`Binding` distinct results.

### 4.2 Family B — Per-kind typed-leaf accessor sugar

A typed leaf with `accessor: Some(_)` lowers at compile to a canonical `Window`-rooted subtree per `[14 §4.2](../14_expressions.md)`; the lowering shape and elimination semantics are defined there. Phase A's substep 1 runs the lowering to fixpoint over the whole `UnresolvedSemanticExpr` so every accessor-bearing leaf is eliminated before substitution begins (§3.1 / §3.3.2.e). Sugar-on-sugar cases (`Delta` lowers into `op - op.Previous`, where `op.Previous` is itself a sugared leaf) converge naturally — no nesting depth limit on the author surface.

Kind agreement between leaf and accessor is type-enforced at construction per `[14 §4.1](../14_expressions.md)`; the algorithm here never has to validate it.

### 4.3 Worked example — metadata fold

Given a Dataset with two source bindings (per `[15 §10](../15_mapping_and_binding.md)` / `[21 §10](../../data-kinds/21_dataset.md)`) and one filter:

```yaml
filters:
  - name: data_quality_window
    expr: |
      case
        when year_dir = 'year=2024' then ordered_at >= '2024-03-15'
        else true
      end
```

At `resolve`, substep 1 is identity (no typed leaves with `accessor: Some(_)`). Substep 2 substitutes `year_dir` and `ordered_at` per binding. Substep 3 folds the resulting `Case` per binding's metadata:

| Binding | `PhysicalExpr`                                                     |
|---------|--------------------------------------------------------------------|
| **B₁**  | `BinaryOp(>=, Column("ordered_at"), Literal::Date("2024-03-15"))`  |
| **B₂**  | `Literal(true)`                                                    |

Phase-B observable: Strategy places B₁'s filter; B₂'s `Literal(true)` is a no-op and its `Filter` is elided. This is the canonical per-binding divergence pattern.

---

## Testing Strategy

- Family A constant-fold round-trip: construct metadata-static expressions (`If` over literals, `BinaryOp` over literals); verify fixpoint convergence; verify fold correctness against engine evaluation.
- Family B accessor-lowering round-trip: construct per-kind accessor sugar (`Key.lag(1)`, `Measure.last_value()`, `Dimension.first_value()`); verify lowering to Window subtrees; verify lowered shape matches planner's Window placement contract.
- Fixpoint convergence: verify the fold terminates within expected iteration bounds; verify idempotence (fold(fold(e)) = fold(e)).
- Per-Binding materialization correctness: verify accessor lowering respects per-Binding keying; verify Window partition / order clauses reference correct binding-local columns.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
