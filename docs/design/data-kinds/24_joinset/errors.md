---
id: 24-joinset-errors
status: ratified
version: 2.1
impl: drift
---

# Joinset — Error Rosters

## 9. Validation Preconditions

`VALID_E_2400`–`2499` are reserved for validate-stage Joinset errors. Validate runs structural checks that do not require Semantics type resolution (per `10 §3.3`).

### 9.1 The roster


| Code           | Variant                                                       | Condition                                                                                                                                                                                                           |
| -------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VALID_E_2400` | `JoinsetArityV1Violation { joinset, member_count }`           | `members.len() != 2` in v1. Also surfaced by `12 §7 NV-V5`; `24` provides the canonical-layer twin.                                                                                                                 |
| `VALID_E_2401` | `JoinsetAnchorMissing { joinset }`                            | `anchor` is absent (canonical-layer only; YAML surface forces the field present per `32`, but programmatic construction paths need the guard).                                                                      |
| `VALID_E_2402` | `JoinsetAnchorNotMember { joinset, anchor, members }`         | `anchor` is not in `members`.                                                                                                                                                                                       |
| `VALID_E_2403` | `JoinsetExplicitPathEmpty { joinset }`                        | `path == Some(ExplicitPath { hops: vec![] })`. An empty explicit path is not an implicit-fallback (§4.3).                                                                                                           |
| `VALID_E_2404` | `JoinsetMembersEmpty { joinset }`                             | `members` is empty (no anchor, no other member). Degenerate; covered-by-2401 in practice but kept as a distinct code for clarity when authoring tools surface the error.                                            |
| `VALID_E_2405` | `JoinsetExplicitPathCyclic { joinset, revisited_member }`     | Explicit path visits the same member twice. Binary v1 cannot trigger; N-ary-forward-looking.                                                                                                                        |
| `VALID_E_2406` | `JoinsetDuplicateMember { joinset, duplicated }`              | `members` contains the same `DataKindRef` twice.                                                                                                                                                                    |


### 9.2 Severity and stage

All `VALID_E_24xx` are `Severity::Error`; they fail validate. Per `30 §7`, validate-stage errors block the pipeline from proceeding to compile.

### 9.3 Interaction with `11` and `12`

Several `24` validate errors share territory with `11` / `12`:

- `VALID_E_2400 JoinsetArityV1Violation` ≡ `12 §7 NV-V5` nesting-shape check on the YAML side.
- `VALID_E_2402 JoinsetAnchorNotMember` is a canonical-layer-only check (YAML authoring does not have an anchor-is-member distinction; the anchor IS a member by YAML structure).
**Retired (2026-05-12).** `VALID_E_2407 JoinsetOverridesForNonexistentHop` is retired — the override surface (`overrides.per_hop` / `HopPosition`) no longer exists on `JoinsetDataKind` (`§2.2`). Code reserved for forward-compat.

Double-emission (a single programming-level error triggering both a `12` and a `24` diagnostic) is avoided at the dispatch level in `34`; when both would fire, the more-specific `24` variant is preferred.

## 10. Compile Preconditions

`COMP_E_2400`–`2499` are reserved for compile-stage Joinset errors. Compile runs type-sensitive and graph-walking checks (per `10 §3.4`).

### 10.1 The roster


| Code          | Variant                                                                                        | Condition                                                                                                                                                                                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COMP_E_2401` | `JoinsetImplicitNoPath { joinset, anchor, target }`                                            | Implicit-path BFS found no Relationship path from `anchor` to the other member (§4.1.2 step 5). Author must declare the Relationship or an explicit path.                                                                                                         |
| `COMP_E_2402` | `JoinsetImplicitAmbiguousPaths { joinset, anchor, target, candidates: Vec<RelationshipPath> }` | Implicit-path BFS found multiple equal-length shortest paths (§4.1.2 step 6). Author must pin via explicit path.                                                                                                                                                  |
| `COMP_E_2403` | `JoinsetAnchorUnreachable { joinset, unreachable_member }`                                     | Resolved path does not connect the anchor to one of the members. v1 binary-degenerate (the single hop either connects the two or fails at `COMP_E_2401`); N-ary-forward-looking.                                                                                  |
| `COMP_E_2404` | `JoinsetExplicitPathUnknownRelationship { joinset, position, relationship }`                   | An explicit `JoinHop.relationship` does not resolve to a SemanticManifest `EntityId`.                                                                                                                                                                       |
| `COMP_E_2405` | `JoinsetExplicitPathAnchorMismatch { joinset, expected_anchor, actual_source }`                | Hop 0's walked source (given `direction`) is not the Joinset's declared anchor.                                                                                                                                                                                   |
| `COMP_E_2406` | `JoinsetExplicitPathEndpointMismatch { joinset, position, declared_to, computed_to }`          | `hop.to` does not match the Relationship's walked target endpoint (given `direction`).                                                                                                                                                                            |
| `COMP_E_2407` | `JoinsetExplicitPathDiscontinuity { joinset, position }`                                       | Hop `i+1`'s walked source != hop `i`'s `to`. v1 binary-degenerate; N-ary-forward-looking.                                                                                                                                                                         |
| `COMP_E_2408` | `JoinsetArityV1Invariant { joinset, hops_count }`                                              | Resolved hops count `!= 1`. Defense-in-depth guard against validate failing to catch an N-ary input.                                                                                                                                                              |
| `COMP_E_2410` | `JoinsetExplicitPathUncoveredMembers { joinset, uncovered: Vec<DataKindRef> }`                 | The explicit path's endpoints do not cover every `member`. Binary v1 cannot trigger; N-ary-forward-looking.                                                                                                                                                       |
| `COMP_E_2414` | `JoinsetAsOfMissingTemporalKey { joinset, position, relationship }`                            | A hop's effective `JoinType == AsOf` but the underlying `Relationship.keys` contains no time-typed `KeyPair` (§7.2 bullet 2). Forward-ref only — v1 does not emit `AsOf`.                                                                                         |
| `COMP_E_2415` | `JoinsetMemberNotTopLevel { joinset, member }`                                                 | A `members` entry refers to a `DataKindRef` that is not a top-level DataKind in the Model. (Nested-`Simple` binding-only leaves are not queryable on their own, per `12 §6.2`.)                                                                                   |


### 10.2 Severity and stage

All `COMP_E_24xx` are `Severity::Error`; they fail compile. Per `30 §7`, compile-stage errors block `plan`.

### 10.3 Ordering

Path-resolution errors (`COMP_E_2401`–`COMP_E_2410`) fire before any temporal-shape-gated errors (`COMP_E_2414`): the hops must resolve before their effective `JoinType`s can be evaluated. `COMP_E_2415` fires before path resolution (member-resolution is a prerequisite).

**Retired (2026-05-12).** `COMP_E_2409 JoinsetReverseForwardOnlyRelationship` is retired — every Relationship is bidirectional per `16 §2.4`; reverse traversal is always legal. `COMP_E_2411 JoinsetIllegalJoinTypeOverride`, `COMP_E_2412 JoinsetAsOfDowngradeForbidden`, and `COMP_E_2413 JoinsetAsOfShapeMismatch` are retired with the per-hop override surface removal (`§2.2`); divergent join semantics surface through structural rules on the scope-local `Relationship` itself (SR-E-4 / SR-E-13 / SR-E-14 in `18 §11`). All four codes are reserved for forward-compat.

## 11. Plan-Stage Rules

`PLAN_E_2400`–`2499` and `PLAN_W_2400`–`2499` are reserved for plan-stage Joinset usage. Plan-stage errors surface when a `Request` interacts with an already-resolved Joinset in an illegal way.

### 11.1 `PlannerError` additions


| Code          | Variant                                                                  | Condition                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PLAN_E_2400` | `JoinsetRequestOutOfSurface { joinset, name }`                           | A Request with `from: Some(joinset)` names a `Semantics` not on the Joinset's `ComposedSemanticInterface`. Specialization of `16 §14.3 PLAN_E_0506 RequestOutOfSurface`; the specialized code carries the Joinset's name for a more actionable diagnostic. The planner MAY emit `PLAN_E_0506` OR `PLAN_E_2400`; `24`'s convention is to prefer `PLAN_E_2400` when the surface is a `CompositionKind::Joinset`. |
| `PLAN_E_2401` | `JoinsetAmbiguousReferenceOnSurface { joinset, name, candidates }`       | A bare `SemanticsName` on the Joinset's composed surface is ambiguous per `16 §6.2.3`'s namespacing rule. Specialization of `16 §14.3 PLAN_E_0505`. Same dispatch convention as `PLAN_E_2400`.                                                                                                                                                                                                                 |
| `PLAN_E_2402` | `JoinsetNonAdditiveRollupRequired { joinset, measure, hop_cardinality }` | A non-additive measure requires a fanout-safe rewrite over the Joinset's fanout shape, but the Request's shape (or `17`'s `TemporalShape × Additivity` matrix) forbids the rewrite. Joinset-specialized variant of `16 §14.3 PLAN_E_0506 CompositionAggregationConflict`.                                                                                                                                      |


### 11.2 `PlannerError` advisories


| Code          | Variant                                                                      | Condition                                                                                                                                                                                                                                                                                                                                                |
| ------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PLAN_W_2400` | `JoinsetFanoutAdvisory { joinset, position, cardinality }`                   | A hop introduces fanout; `JoinsetStrategy` inserted a fanout-safe rewrite. Joinset-specialized intent advisory; the cross-DataKind parent (`PLAN_W_0501 FanoutAdvisory`) was retired in `16 §14.4` (2026-04-29) per Q-COMP-005's intent-advisory deferral. Retention vs retirement of the Joinset-specialized variant is `[TD-JOINSET-FANOUT-ADVISORY]`. |
| `PLAN_W_2402` | `JoinsetManyToManyHopAdvisory { joinset, position, relationship }`           | A hop's underlying `Relationship.cardinality == ManyToMany`; consider junction-table modeling (`16 §3.3.4`). Joinset-specialized intent advisory; the cross-DataKind parent (`PLAN_W_0502 ManyToManyFanoutAdvisory`) was retired in `16 §14.4` (2026-04-29). Retention covered by `[TD-JOINSET-FANOUT-ADVISORY]`.                                        |
| `PLAN_W_2403` | `JoinsetMultiFanoutAdvisory { joinset, profile }`                            | The Joinset has more than one fan-out edge (v1 binary: a single `ManyToMany` hop; N-ary: cross-hop accumulation). `JoinsetStrategy` proceeds but the author should consider restructuring.                                                                                                                                                               |
| `PLAN_W_2404` | `JoinsetAsOfActivation { joinset, position, declared, activated }`           | `17 §5`'s matrix mandated `JoinType::AsOf` on a hop whose declared `JoinType` was non-`AsOf`. Informational so the author sees the activation.                                                                                                                                                                                                           |


### 11.3 Severity

All `PLAN_E_24xx` are `Severity::Error`; they fail plan. All `PLAN_W_24xx` are `Severity::Warning`; plan proceeds with the diagnostic emitted.

**Retired (2026-05-12).** `PLAN_W_2401 JoinsetJoinTypeOverrideAdvisory` is retired with the override-surface removal (`§2.2` / `§5.3.2`). Divergent join semantics are now declared structurally via a scope-local `Relationship` and do not warrant a per-resolution advisory. Code reserved for forward-compat.

### 11.4 Dispatch convention

`16 §14`'s composition errors (`PLAN_E_05xx`) are the generic, composition-kind-agnostic entries. `24`'s `PLAN_E_24xx` / `PLAN_W_24xx` are the Joinset-specialized entries. The planner's convention:

- If the error is clearly Joinset-specific (e.g., `PLAN_W_2403 JoinsetMultiFanoutAdvisory`), emit the `24` code.
- If the error is a generic composition concern that happens to apply to a Joinset (e.g., a fanout advisory on an implicit composition that coincidentally covers the Joinset's members — see `16 §13.5`), emit the `16` code.
- Dispatch happens in `34`; `24` fixes the taxonomy without hard-coding dispatch rules.

## 14. Cross-References

- `00 §4.1` — `Joinset` row (one of the four `CompositionKind` variants); `00 §4.2` — canonical vocabulary; `00 §9` — invariants I1, I4, I5, I7, I8, I10, I12.
- `10 §3.3, §3.4` — `validate` and `compile` stage contracts; `24` respects both without reopening.
- `11 §2, §3, §6.5, §9, §10.3` — scope chain, global identity, `Constraint::Key`, cross-kind references, structural labels used by `12 §5.1`'s YAML path shape.
- `12 §2, §5` — nesting matrix (Joinset ⊄ Joinset, member kinds); binary-v1 arity; YAML-level path shape (`path.on.left` / `path.on.right`).
- `13 §5` — `DataType` compatibility, consumed by `KeyPair` type-agreement checks `16 §12.2`.
- `14 §N` — expression grammar used by Joinset-level derived dimensions / filters; `19 §3.4.5` — `PathSignature` subsumption interaction (§11's `PLAN_E_0509` remains the canonical code for Request-path / expression-path mismatches, even on Joinset surfaces).
- `15 §4, §6` — `Binding`, `ColumnMapping`, `Coverage`; consumed when `JoinsetStrategy` resolves per-hop `KeyPair`s to physical columns.
- `16 §§2–14` — `Relationship`, `Cardinality`, `JoinType`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, implicit composition algorithm, composition error codes. `24` cites; `24` does NOT redefine any of `16`'s canonical types.
- `17 §5` (pending) — `TemporalShape` activation matrix for `JoinType::AsOf`.
- `20_taxonomy.md` (parallel draft) — data-kinds taxonomy.
- `23_unionset.md` (parallel draft) — `CompositionKind::Unionset` peer.
- `22_grainset.md` (parallel draft) — `CompositionKind::Grainset` peer.
- `21_dataset.md` — `Simple` / Dataset leaf.
- `25` (pending) — cross-kind strategy catalog (Joinset × Grainset, Joinset × Unionset interactions).
- `30 §4, §5, §6` — `#[non_exhaustive]` policy, `Diagnostic` shape, error-code range governance; `24` allocates `VALID_E_2400`–`2499`, `COMP_E_2400`–`2499`, `PLAN_E_2400`–`2499`, `PLAN_W_2400`–`2499` under `30 §6.2`'s per-DataKind-doc scheme.
- `32` (pending) — YAML surface for the `joinsets:` top-level block, `path:` sub-block, and Joinset-scoped `relationships:` sub-block (scope-local Relationship shadow shape).
- `33` (pending) — SemanticManifest surface: `ResolvedJoinset`, `ResolvedJoinHop`.
- `34` (pending) — planner surface: `JoinsetStrategy` impl, dispatch rules between `16` generic codes and `24` specialized codes.
- `35` (pending) — `PlanNode::Join` with `from_relationship` and `from_joinset` tagging fields.
- `questions/closed/24_questions.md` — Round-1 ratified Q-24-02..Q-24-08; `questions/deferred/24_questions.md` — Q-24-01 / Q-24-09 / Q-24-10 (deferred for post-v1).
- `questions/open/16_questions.md` — composition-level deferrals that touch Joinset (`[TD-COMPOSITION-JOINSET-REUSE]`, `[TD-COMPOSITION-SELFJOIN]`, `[TD-COMPOSITION-ASOF]`).

## Testing Strategy

- VALID_E_2400; COMP_E_2401/2402; PLAN_E_2400 JoinsetRequestOutOfSurface; PLAN_W_2402; full roster round-trip per 30 §6.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
