---
id: 24-joinset-path-resolution
status: ratified
version: 2.1
impl: drift
---

# Joinset — Join-Path Resolution

## 4. Join-Path Specification

### 4.1 Implicit path

#### 4.1.1 Authoring shape

The author declares `members` (including the `anchor`) and omits `path`. Compile derives a `ResolvedJoinset.hops` list by running an **anchor-rooted** variant of `16 §11.4`'s `RELATIONSHIP_BFS`.

#### 4.1.2 Algorithm (binary v1)

```text
JOINSET_IMPLICIT_PATH(anchor, members, manifest) -> Vec<ResolvedJoinHop> | CompileError:
  1. other_members ← members \ { anchor }
  2. if |other_members| != 1:
        return COMP_E_2408 JoinsetArityV1Invariant  // defense in depth; validate
                                                      // should have caught this already
  3. target ← unique(other_members)
  4. paths ← RELATIONSHIP_BFS_ANCHORED(anchor, target, manifest)
        where:
          - source = anchor
          - target = target
          - depth limit = unbounded (Joinset is the escape hatch; 16 §9.1's
            MAX_IMPLICIT_COMPOSITION_DEPTH does NOT apply here)
          - neighbor order = relationship name order (I4, same as 19 §3.4.3)
          - direction constraints = none; every Relationship is bidirectional
            per 16 §2.4
  5. if paths is empty:
        return COMP_E_2401 JoinsetImplicitNoPath { anchor, target }
  6. if |paths| > 1 AND shortest-path length is not unique:
        return COMP_E_2402 JoinsetImplicitAmbiguousPaths { anchor, target, candidates }
                                                      // author must pin via explicit path
  7. selected ← the unique shortest path
  8. return Vec<ResolvedJoinHop> constructed from `selected` with each hop
           carrying its EntityId, walked direction, and landed-on DataKindRef.
```

#### 4.1.3 Extension of `16 §11.4`

`16 §11`'s field-first algorithm visits **multiple** target kinds (one for each requested Semantics's owning kind). `Joinset`'s implicit-path variant visits exactly the other member(s) explicitly listed in `members`, with the anchor as the fixed source. The extension is:

1. **Anchor pinning.** Source is the `anchor`; BFS walks outward from it. In `16 §11.4` BFS walks from the full set of `OwningKinds` inward (Steiner-tree-style); here BFS walks from a single known source.
2. **No depth limit.** `16 §9.1`'s `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` does NOT apply. Declaring a `Joinset` is exactly what an author does when their path exceeds the implicit depth cap; imposing the same cap on the escape hatch would be pointless.
3. **Ambiguity handling deferred to compile.** Implicit composition surfaces ambiguity at plan time (`PLAN_E_0500`). Joinset's implicit path surfaces ambiguity at **compile** time (`COMP_E_2402`), because the Joinset is a named surface whose path must be deterministic before plan-time queries land.

Algorithmic I4-determinism is preserved: same SemanticManifest + same anchor + same target → same resolved hops.

#### 4.1.4 When to use implicit path

Authors prefer implicit when:

- The SemanticManifest has a single shortest Relationship path between anchor and target. Typical for star-schema shapes where the fact → dimension hop is unique.
- The Relationship path is expected to stabilize; new Relationships that create alternative paths would be caught at the `compile` that introduces them (`COMP_E_2402`), giving the author a chance to pin.

Authors prefer explicit (§4.2) when:

- Multiple equal-length paths exist and the author has a specific one in mind.
- The author needs divergent join semantics on a specific hop: declare the divergence on a scope-local Relationship (`§5.3.2`) whose `name` matches the hop's traversed Relationship, and pin the explicit path so the scope-local shadow lands on the intended hop.
- The `Joinset` will grow in the future (e.g. when `TD-NESTING-NARY-JOIN` lifts), and the author wants the traversal shape pinned regardless of later Relationship additions.

### 4.2 Explicit path

#### 4.2.1 Authoring shape

The author declares `path: Some(ExplicitPath { hops: Vec<JoinHop> })`. Each `JoinHop` cites a `EntityId` and a `HopDirection`. For v1 binary Joinsets, exactly one hop.

#### 4.2.2 Validation pipeline

Explicit path validation runs across validate and compile stages:


| Stage    | Check                                                                                                                                   | Error                                                                                                                                                                                           |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| validate | `hops` non-empty                                                                                                                        | `VALID_E_2403 JoinsetExplicitPathEmpty`                                                                                                                                                         |
| validate | (v1) `hops.len() == 1`                                                                                                                  | `VALID_E_2400 JoinsetArityV1Violation`                                                                                                                                                          |
| compile  | each `hop.relationship` resolves to a SemanticManifest `EntityId`                                                                 | `COMP_E_2404 JoinsetExplicitPathUnknownRelationship { position, relationship }`                                                                                                                 |
| compile  | `hop_0`'s walked source == `anchor` (given `direction`, the "source" endpoint matches the anchor DataKind)                              | `COMP_E_2405 JoinsetExplicitPathAnchorMismatch { expected_anchor, actual_source }`                                                                                                              |
| compile  | `hop_i.to` matches the Relationship's walked target endpoint (given `direction`)                                                        | `COMP_E_2406 JoinsetExplicitPathEndpointMismatch { position, declared_to, computed_to }`                                                                                                        |
| compile  | `hop_{i+1}`'s source (given direction) == `hop_i.to` (chain continuity) — no-op for v1 binary, but kept for N-ary forward-compatibility | `COMP_E_2407 JoinsetExplicitPathDiscontinuity { position }`                                                                                                                                     |
| compile  | (binary v1) exactly the `members` set is covered by the explicit path's endpoints                                                       | `COMP_E_2410 JoinsetExplicitPathUncoveredMembers { uncovered }`                                                                                                                                 |


#### 4.2.3 Self-joins and cyclic paths

- `Relationship.from == Relationship.to` (self-joins) is forbidden per `16 §12.4 COMP_E_0403`; a Joinset CANNOT reference a self-referential Relationship because none exist in v1. Tracked as `[TD-COMPOSITION-SELFJOIN]` in `16 §12.4`; `24` inherits.
- Cyclic explicit paths (visiting the same DataKind twice) are rejected at `VALID_E_2405 JoinsetExplicitPathCyclic`. Binary v1 cannot produce a cycle (one hop), so the rule is N-ary-forward-looking.

#### 4.2.4 Scope-local Relationship resolution

When a Joinset declares a scope-local `Relationship` (per `§5.3.2`) and an explicit path, compile resolves each `hop.relationship` `EntityId` against the Joinset's scope first, then falls back to root-level. A scope-local Relationship with the same `name` as a root-level one shadows it within this Joinset (`18 §2.10`); the path's `EntityId` is rebound to the scope-local entry. With an implicit path, scope resolution runs after path enumeration: BFS walks the root-level Relationship graph to enumerate paths, then each resolved hop's Relationship is rebound to its scope-local shadow (if any) before `EFFECTIVE_JOIN_TYPE` runs.

### 4.3 Mode-selection precedence


| `path` value                           | Resolved mode                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| `None`                                 | Implicit (§4.1).                                                                            |
| `Some(ExplicitPath { hops: vec![] })`  | Rejected at validate (`VALID_E_2403`). Empty explicit paths are never an implicit-fallback. |
| `Some(ExplicitPath { hops: [_, ..] })` | Explicit (§4.2).                                                                            |


Round 1 forbids hybrid modes ("use these specific hops plus let the planner fill in the rest"). Hybrid modes are tracked as `[TD-JOINSET-HYBRID-PATH]` in `questions/closed/24_questions.md` (Q-24-02 — closed; TD marker carries the post-v1 reactivation).

## Testing Strategy

- COMP_E_2401 JoinsetImplicitNoPath; COMP_E_2402 JoinsetImplicitAmbiguousPaths; COMP_E_2404–2407 explicit-path validation; scope-shadow resolution round-trip.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
