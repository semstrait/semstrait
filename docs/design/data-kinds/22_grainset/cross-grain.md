---
id: 22-grainset-cross-grain
status: ratified
version: 2.1
impl: drift
---

# Grainset — Cross-Grain Composition

### 3.4 Cross-grain composition — `ComposedSemanticInterface` + LEFT OUTER JOIN tree

When no single effective routing unit (per §4.1) covers all of a Request's Semantics at the eligible grain, the planner builds a **cross-grain LEFT OUTER JOIN tree**, mediated by a `ComposedSemanticInterface` (per `16 §5`) constructed at compile. Per G-2 (2026-05-03 ratification).

**Build-time (compile)**:

1. Build per-routing-unit Coverage projection of the Grainset's surface (per §3.2; effective routing units after same-grain pre-merge per §3.3).
2. For each pair of effective routing units, identify **shared Keys** — Keys (per `18 §2.5`) declared on both units' interfaces. The Keys must agree on `data_type` (under `13 §7`'s widening rules).
3. Construct the `ComposedSemanticInterface`:
   - `composition_kind = CompositionKind::Grainset` (per `16 §5.3`).
   - `constituents: Vec<DataKindRef>` — the effective routing units (one per distinct grain).
   - `interface: UnifiedSemantics` — the Grainset's own `SemanticInterface` lifted into the composition shape per `16 §6`.
   - `provenance: FieldProvenance` — per-Semantics ownership across units per `16 §7`.
   - `coverage: CompositionCoverage` — per-(unit, Semantics) Coverage entries per `16 §8`.
   - `traversed_paths: Vec<RelationshipPath>` — empty (Grainset cross-grain composition uses Keys, not Relationships; cross-grain JOIN-tree details live in `ResolvedGrainset` per `33`, not on `ComposedSemanticInterface`).
4. Pre-build the cross-grain JOIN-key index: for each pair of units that share Keys, record the equi-join condition.

**Plan-time (per Request)** — see §4.3.

**Hard failure modes (compile)**:

- **No shared Keys between two units the planner needs to join** — `COMP_E_2204 GrainsetCrossGrainKeysAbsent { driver, attached, missing_via }` (§8). Per G-2d ratification: hard compile error rather than runtime fallback. Author fixes by adding the missing Key declaration to one or both units' interfaces.
- **Key `data_type` disagreement between units** — `COMP_E_2205 GrainsetCrossGrainKeyTypeMismatch` (§8).

## Testing Strategy

- ComposedSemanticInterface build at compile (composition_kind Grainset); shared Keys per 18 §2.5; COMP_E_2204 GrainsetCrossGrainKeysAbsent; COMP_E_2205 GrainsetCrossGrainKeyTypeMismatch.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
