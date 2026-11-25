---
id: 24-joinset-interface
status: ratified
version: 2.1
impl: drift
---

# Joinset — Composed Semantic Interface

<!-- cross-cut: none -->

## 7. Interaction with `TemporalShape`

`17_temporal_shape.md` is a parallel draft. `24` fixes the integration points so `17`'s ratification lands as mechanical uptake.

### 7.1 What `17` will ratify (forward-ref)

Per `16 §4.4.2` and `00 §4.1`'s `TemporalShape` row:

- `TemporalShape` is a per-`DataKind` classification: `Timeseries`, `Events`, `Snapshot`, `Scd`, …
- `JoinType::AsOf` is the temporal-proximity join variant that, for each row on the anchor side, picks the most-recent row on the target side satisfying an as-of predicate (typically a `valid_from` / `valid_to` range).
- `AsOf` activation is gated on the pair of endpoint TemporalShapes. The activation matrix — which endpoint-shape pairs permit `AsOf` — lives in `17 §5`.

### 7.2 `Joinset`'s contract re `AsOf` (forward-ref, v1 descoped)

`JoinType::AsOf` is descoped for v1; the rules below are forward-ref scaffolding for the post-v1 MINOR that lands `AsOf` after `17` ratifies. The integration points are:

1. **Implicit activation.** If a Joinset hop walks between two DataKinds whose `TemporalShape` pair **mandates** as-of joining (per `17 §5`'s matrix; canonical case: `Events ↔ Snapshot` and `Events ↔ Scd`), the hop's effective `JoinType` is `AsOf` regardless of the canonical derivation from `Relationship.optional`. `JoinsetStrategy` activates silently and emits `PLAN_W_2404 JoinsetAsOfActivation` as an informational advisory so the author sees the activation.
2. **Missing temporal key.** `JoinType::AsOf` requires the `Relationship.keys` to include a time-typed `KeyPair` (`16 §4.4.2`). If absent, `COMP_E_2414 JoinsetAsOfMissingTemporalKey`. This is a compile-stage check; `17 §5` ratifies the temporal-key schema.

Author opt-out and shape-mismatch handling for `AsOf` are folded into the scope-local-Relationship model: a Joinset that needs to *prevent* `AsOf` activation declares a scope-local Relationship that diverges in shape (e.g. drops the temporal `KeyPair`), and the temporal pair no longer "mandates" `AsOf`.

### 7.3 Activation matrix skeleton (to be filled by `17 §5`)

`17 §5` will fill in the precise matrix; `24`'s integration point is the two buckets:

- **Mandated `AsOf`** (bullet 7.2.1): the derived `JoinType` is overridden to `AsOf` at plan emission. Advisory emitted.
- **Permitted `AsOf` but not mandated**: the canonical derivation from `optional` wins; no special handling.

When `17` ratifies, this section is updated to cite `17 §5.X` tables directly.

## 8. Interaction with `ComposedSemanticInterface`

A `Joinset`'s output surface is a `ComposedSemanticInterface` with `composition_kind == CompositionKind::Joinset`. This section specializes `16 §§5–8` for the Joinset case; canonical definitions remain in `16`.

### 8.1 `constituents` and ordering

- `constituents` contains the `members` of the Joinset, with the **anchor canonicalized to position 0** followed by the order hops reach the other members. In binary v1: `constituents = [anchor, hops[0].to]`.
- This ordering is stable per I4 and is used by `FieldProvenance` and `CompositionCoverage` to refer to members positionally (though the maps are keyed by `DataKindRef`, not by index).
- The anchor-first ordering is contract: planner code that relies on "constituent 0 is the FROM-clause member" is correct for Joinsets.

### 8.2 `UnifiedSemantics` merge rules

Per `16 §6`, `UnifiedSemantics` merges constituent `SemanticInterface`s into a single queryable surface. For Joinsets:


| Case                                                                       | Shared-compatible?    | `UnifiedSemantics` treatment                                                                                                                                                                                                             |
| -------------------------------------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A dimension appears on exactly one member.                                 | Trivially.            | Bare name on the unified surface; `FieldOwnership::Native(owning_member)`.                                                                                                                                                               |
| A dimension appears on multiple members, shape-compatible per `16 §6.2.3`. | Yes.                  | Shared; bare name; `FieldOwnership::Shared(members)`. Typical for the **join key column** (`customer_id` when joining Orders ↔ Customers), which is the same `SemanticsName` on both sides by construction (§16 `KeyPair.left / right`). |
| A dimension appears on multiple members, shape-**in**compatible.           | No.                   | Namespaced (`orders.customer_id`, `returns.customer_id`); bare name is ambiguous.                                                                                                                                                        |
| A measure appears on multiple members with incompatible aggregation.       | No (per `16 §6.2.4`). | Namespaced; planner emits `PLAN_E_0505 AmbiguousCompositionReference` if the Request uses the bare form.                                                                                                                                 |
| The Joinset declares its own derived dimension / measure / metric.         | n/a                   | `FieldOwnership::Derived(physical_expr)` on the unified surface. Contributes to `UnifiedSemantics` alongside per-member contributions.                                                                                                   |


The Joinset's own declared keys (§2.6) appear in `UnifiedSemantics.keys`; absent explicit declaration, the unified surface has no composed-level keys and the anchor's keys remain per-constituent.

### 8.3 `FieldProvenance`

Per `16 §7`, `FieldProvenance` records per-field ownership on the composed surface:

- `**Native(DataKindRef)`** — the field exists on exactly one member. Typical for dimensions / measures specific to a single side.
- `**Shared(Vec<DataKindRef>)**` — the field exists with compatible shape on multiple members; typical for the join-key column.
- `**NullFill(Vec<DataKindRef>)**` — per `16 §7.3.3`, this variant is produced ONLY for `CompositionKind::Unionset`. For `CompositionKind::Joinset`, missing-side fields under an outer join (`Left`, `Right`, `Full`) are carried by the JoinType's NULL-fill in SQL emission, NOT by `FieldOwnership::NullFill`. A Joinset's `FieldProvenance` therefore never contains `NullFill` entries.
- `**Derived(PhysicalExpr)**` — Joinset-level computed fields per §2.6.

### 8.4 `CompositionCoverage`

Per `16 §8.4`, `CompositionCoverage.entries` is the per-`(constituent, unified-name)` fold of binding-level `Coverage`. For Joinsets, the fold is:

```text
JOINSET_COVERAGE_FOLD(joinset) -> CompositionCoverage:
  entries ← {}
  for member in joinset.constituents:
    for unified_name in joinset.interface.interface.all_names():
      entries[(member, unified_name)] ←
        fold_binding_coverage(member, unified_name)
          // returns Native, Derived, NullFill, or Metadata per `16 §8.4`
  return CompositionCoverage { entries }
```

Consumers (planner advisories, SQL adapters) read this map to know, for each `(member, name)` pair, whether the member natively provides the column, computes it, or NULL-fills it.

### 8.5 `traversed_paths`

`16 §5.2`'s `ComposedSemanticInterface.traversed_paths: Vec<RelationshipPath>` is populated for Joinsets with a single-element `Vec<RelationshipPath>` containing the single hop's `[EntityId]` (v1 binary). For N-ary Joinsets, multiple paths may be present (one per leg of the spanning tree). The field is diagnostic — it lets `34` / `35` print the actual traversed edges in plan explanations without re-resolving.

### 8.6 Identity and equality (specialization of `16 §5.4`)

Two `ResolvedJoinset`s are equal iff:

- Their `name` match.
- Their `anchor` match.
- Their `constituents` sequences match (order-significant).
- Their `hops` sequences match (order-significant; each hop's `(relationship, direction, to)` triple).
- Their scope-local `relationships` rosters match (per-name canonical-form equality of every shadow entry; root-level resolution at each hop yields identical Relationships).
- Their `ComposedSemanticInterface.interface` match (`UnifiedSemantics` roster).

Per I4 the resolution pipeline is deterministic; two compiles of the same Model produce identical `ResolvedJoinset`s.

## Testing Strategy

- Constituents anchor-first ordering; UnifiedSemantics shape-compatible merge (join key = Shared per 16 §6.2.3); FieldProvenance Native/Shared/Derived (no NullFill per 16 §7.3.3); CompositionCoverage fold per 16 §8.4.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
