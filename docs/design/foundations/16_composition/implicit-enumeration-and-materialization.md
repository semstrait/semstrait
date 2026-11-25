---
id: 16-implicit-enumeration-and-materialization
status: ratified
version: 2.1
impl: implemented
---

# Explicit vs Implicit Composition + Materialization Policy

This leaf ratifies the boundary between `Origin::Explicit` (author-declared) and `Origin::Implicit` (compile-enumerated) compositions, the **eager-materialization policy** (all compositions materialized at `SemanticGraph` build time; manifest carries primitives only), the depth and count bounds on implicit enumeration, and the **canonical-form clash rejection rule** (`COMP_E_0414`).

**Ratified (Q2 revised 2026-04-29; Q3 revised 2026-04-29; rebased to lightweight-manifest posture 2026-05-28 — see `_research/manifest/RATIFICATION_LOG.md` clauses C7.4 / C8.2 / C9.2 / C10.2 / CCK).** All compositions — `Origin::Explicit` and `Origin::Implicit`, `Joinset` and `Unionset` and `Grainset` — are materialized as `ResolvedComplexDataKind` shapes inside the **`SemanticGraph` build** (in-memory, post-load). The `SemanticManifest` itself carries only **primitives** — `semantics: SemanticBitmap` (C4), per-DataKind `coverage` masks (CCK / C5–C8), the root-scope `relationships:` map, and Joinset-local shadow Relationships inline on the Joinset variant (C7.6) — and does NOT carry a top-level `compositions:` field (C9.2 / `33 §4.1` / I12.15). `SemanticGraph` runs the enumeration algorithms below over those primitives at build time, before the planner consumes the graph. Plan-time field-first resolution (see [field-first-resolution](field-first-resolution.md)) is then a pure lookup over `SemanticGraph`'s build-time composition index. Implicit enumeration is bounded by depth (`MAX_IMPLICIT_COMPOSITION_DEPTH = 4`) and a hard count cap (`MAX_IMPLICIT_ENUMERATION_COUNT = 2000`); both bounds are **code-level constants** in `semstrait-ir` (or the graph crate when materialized), not manifest-persisted values (C10.2 / I12.16).

## 9. Explicit vs Implicit Composition

Composition has a single structural model (`composition_kind` × `Origin`, see [composed-surface-type](composed-surface-type.md)) and two provenance flavours:

- **`Origin::Explicit`** — author-declared `joinsets:` / `unionsets:` / `grainsets:` blocks in the `SemanticModel`. The author owns the name and any per-leg overrides / filters / keys.
- **`Origin::Implicit`** — compile-enumerated from declared `Relationship`s (Joinset) or coverage overlap (Unionset). Compile produces the same `ResolvedJoinset` / `ResolvedUnionset` shape; the synthetic `DataKindName` and `ImplicitId` are assigned per [composed-surface-type](composed-surface-type.md) §5.7. No author overrides — defaults from the underlying `Relationship` declarations.

Both flavours land in `SemanticGraph`'s build-time composition index (§10) and are addressable through the same `name_index`. Explicit compositions also persist on the SemanticManifest as top-level `DataKind` entries (`33 §6` — variant carries primitives only); implicit compositions are graph-only (no manifest persistence per C9.2 / `33 §6.8`). Plan-time field-first resolution (see [field-first-resolution](field-first-resolution.md)) is a pure lookup over both forms via the graph's index.

### 9.1 Boundary rules

1. **Implicit Joinsets only over declared `Relationship`s.** Compile never enumerates a join over an anonymous predicate. If a Request requires a join that no `Relationship` declares — and no implicit Unionset covers the field set — the planner emits `PlannerError::NoCompositionPath` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0501`).
2. **Path ambiguity errors at plan time.** When multiple implicit Joinsets cover the same Request constituent set with equal shortest-path cost (e.g. `customer → billing_address → city` vs `customer → shipping_address → city`), compile enumerates **both** as distinct implicit Joinsets (different `ImplicitId`s); the planner detects the ambiguity at lookup time and emits `PLAN_E_0500 AmbiguousImplicitComposition` with both candidate `DataKindName`s in the diagnostic. Authors disambiguate by declaring a differentiated explicit `Joinset` (per-leg overrides, filters, or `keys` make the canonical form distinct from any enumerable implicit form, side-stepping the §10.6 clash).
3. **Coverage ambiguity → implicit `Unionset`.** When N independent top-level kinds (no `Relationship` between them) cover the same Semantics, compile enumerates an implicit `Unionset` over those kinds (§10.5). The planner addresses it through the standard field-first lookup; no error.
4. **Depth-limited to `MAX_IMPLICIT_COMPOSITION_DEPTH` hops.** Q-COMP-001 closed 2026-04-28 ratifies `4` hops for v1. Compile does not enumerate implicit Joinsets requiring a longer path. Requests that would need a longer path fall through to `PlannerError::CompositionDepthExceeded` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0502`). Authors wanting deeper compositions declare an explicit `Joinset`. The cap is a **compile-time / graph-build code-level constant** in `semstrait-ir` (or the graph crate when materialized) and is **not** persisted on manifest (per C10.2 / C10.4; see also `33 §6.8`, I12.16). It applies only to implicit composition over DataKinds (`CompositionKind::Relationship` / `Origin::Implicit`); it does **not** apply to Joinset implicit-path BFS (per `24 §4.1.3` — Joinset is the escape hatch) nor to the Relationship graph itself, which remains open to support future optimization passes (cycle detection, fan-out detection, query rewrite).
5. **Hard cap `MAX_IMPLICIT_ENUMERATION_COUNT = 2000`.** Compile enumerates at most 2000 implicit Joinsets + Unionsets per Model (§10.4). Cap-exceeded → `CompileError::ImplicitEnumerationExploded` (see [composition-validation-and-errors](composition-validation-and-errors.md), `COMP_E_0409`). Authors with pathological models tighten the implicit graph by declaring explicit Joinsets for common subsets or removing redundant Relationships.
6. **Bidirectional traversal.** Every `Relationship` is walked in both directions during enumeration (per [relationship-edge-model](relationship-edge-model.md) §2.4). There is no authoring-layer mechanism to restrict traversal direction; authors wanting one-way semantics declare an explicit Joinset.
7. **Transparent unfolding through composed surfaces.** Compile's implicit-Joinset enumeration walks the unfolded graph — a `Unionset` or `Joinset` constituent is treated as the union of its own constituents during enumeration. The previous Round-1 prohibition on chaining (`PLAN_E_0504 CompositionChainingForbidden`) is **retired** (2026-04-29). The implicit-explicit clash check (§10.6) prevents the unified compositions from accidentally degenerating into duplicates.
8. **Implicit-explicit clash → `COMP_E_0414`.** An author-declared explicit `Joinset` whose canonical form (sorted `(EntityId, direction)` tuples) matches an enumerable implicit Joinset is **rejected at compile** with `CompileError::ExplicitImplicitCompositionClash` (see [composition-validation-and-errors](composition-validation-and-errors.md), `COMP_E_0414`, §10.6 — **authoritative statement**). Authors differentiate the explicit form via scope-local `Relationship` shadow (see [joinset-model](joinset-model.md) §13.3), Joinset-level `filter:`, declared `keys`, or a non-shortest path; otherwise the planner uses the equivalent implicit Joinset (the explicit declaration is redundant).
9. **`Request.from = Some(DataKindRef)` skips field-first.** The named kind — Simple, explicit Complex, or implicit Complex (addressed by synthetic name) — is looked up directly. No re-resolution.

### 9.2 Implicit-Joinset enumeration sketch

At compile, after `RelationshipGraph` construction:

1. **Seed** with every pair `(A, B)` such that `A` and `B` are top-level kinds connected by at least one `Relationship`.
2. **Expand** each seed by walking outward up to `MAX_IMPLICIT_COMPOSITION_DEPTH` hops; both directions of every `Relationship` are walked (per [relationship-edge-model](relationship-edge-model.md) §2.4). Every reachable subset of size 2..(`depth + 1`) becomes a candidate.
3. **Canonicalize** each candidate by sorting its `(EntityId, direction)` tuples; duplicates collapse.
4. **Hash** the canonical form into `ImplicitId` (see [composed-surface-type](composed-surface-type.md) §5.7).
5. **Materialize** as `ResolvedJoinset { origin: Implicit { id }, … }` per `33`.
6. **Cap-check** at each addition: cumulative count > `MAX_IMPLICIT_ENUMERATION_COUNT` → `COMP_E_0409`.

The full algorithm is in §10.4. The pre-clash check (§10.6) runs before materialization to reject explicit Joinsets with matching canonical forms.

### 9.3 Implicit-Unionset enumeration sketch

At compile, after per-kind `Coverage` derivation:

1. **Build the inverse-coverage map**: for every `SemanticsName`, the set of top-level kinds that cover it natively or via `Derived` / `Metadata`.
2. **Identify coverage groups**: every set of 2+ kinds that all cover the same `SemanticsName` and are NOT connected by any `Relationship` form a candidate implicit Unionset.
3. **Canonicalize** by sorting `Vec<DataKindRef>`; duplicates collapse.
4. **Hash** the canonical form into `ImplicitId` (see [composed-surface-type](composed-surface-type.md) §5.7).
5. **Materialize** as `ResolvedUnionset { origin: Implicit { id }, … }`.
6. **Cap-check** as in §9.2 step 6.

Full algorithm in §10.5.

## 10. Materialization Policy

All compositions — `Origin::Explicit` and `Origin::Implicit`, `Joinset` and `Unionset` and `Grainset` — are materialized as `ResolvedComplexDataKind` shapes inside the **`SemanticGraph` build** (in-memory, post-load). The `SemanticManifest` itself carries only **primitives** and does NOT carry a top-level `compositions:` field (C9.2 / `33 §4.1` / I12.15). `SemanticGraph` runs the enumeration algorithms below over those primitives at build time, before the planner consumes the graph. Plan-time field-first resolution (see [field-first-resolution](field-first-resolution.md)) is then a pure lookup over `SemanticGraph`'s build-time composition index. Implicit enumeration is bounded by depth (`MAX_IMPLICIT_COMPOSITION_DEPTH = 4`) and a hard count cap (`MAX_IMPLICIT_ENUMERATION_COUNT = 2000`); both bounds are **code-level constants** in `semstrait-ir` (or the graph crate when materialized), not manifest-persisted values (C10.2 / I12.16).

### 10.1 Explicit — materialized at `SemanticGraph` build

At `compile`, the manifest persists each declared `ComplexDataKind` (`Unionset` / `Grainset` / `Joinset`) as a top-level `DataKind` entry keyed by `data_kind_id`, with the variant-specific primitives carried on `DataKindVariant` (per `33 §6.5–§6.7`): `Unionset` carries `branches` + per-branch `branch_coverage`; `Grainset` carries `levels` + per-level `level_coverage` + `routing_unit`; `Joinset` carries `anchor` + `members` + `hops` + `path_origin` + `scope_local_relationships`. The top-level `coverage` mask is the universal-union view (CCK.1).

At `SemanticGraph` build (in-memory, post-load), for each such manifest entry the graph synthesizes a `ResolvedComplexDataKind` with `origin: Origin::Explicit` and computes `ComposedSemanticInterface` carrying:

- A `composition_kind` matching the declared complex-kind variant.
- The author-declared `constituents` list (or declared member kinds for `Joinset`).
- A fully-computed `UnifiedSemantics` (namespace-aware merge, see [composed-surface-contents](composed-surface-contents.md) §6).
- A `FieldProvenance` (see [composed-surface-contents](composed-surface-contents.md) §7).
- A `CompositionCoverage` (see [composed-surface-contents](composed-surface-contents.md) §8).
- `traversed_paths` (for `Joinset`).
- `keys` per [composed-surface-contents](composed-surface-contents.md) §6.5 (declared or derived).

The synthesis is reproducible from the manifest primitives alone (C7.4 + C8.2 cascade): `ComposedSemanticInterface` is **not** persisted on manifest. The planner, when presented with `Request.from = Some(<complex-kind-name>)`, retrieves the `ResolvedComplexDataKind` from `SemanticGraph` by name and plans against the pre-built interface. No merge logic re-runs at plan time.

### 10.2 Implicit — also materialized at `SemanticGraph` build

At `SemanticGraph` build (in-memory, post-load), after explicit compositions are admitted into the graph (§10.1):

- Run the implicit-Joinset enumeration (§10.4) and implicit-Unionset enumeration (§10.5). Each enumerated composition is a `ResolvedComplexDataKind` with `origin: Origin::Implicit { id }`, living inside `SemanticGraph` — NOT in the manifest (per C9.2: manifest carries no `compositions:` field; cf. `33 §4.1` / §6.8 / I12.15).
- Run the implicit-explicit clash check (§10.6) before graph construction closes; clashes fail compile with `COMP_E_0414`. (The clash check is hoisted to compile because it depends only on primitives — the explicit canonical-form set is computable from manifest `data_kinds:` entries plus `relationships:` without running the full enumeration.)
- Index every implicit composition under its synthetic `DataKindName` (`__implicit_{joinset|unionset}_{first-8-hex-of-id}`, see [composed-surface-type](composed-surface-type.md) §5.7) and under its `ImplicitId`. Both indices are addressable on `SemanticGraph`'s build-time composition index (not on the manifest).
- The planner addresses implicit compositions via the same lookup path as explicit ones — both flow through `SemanticGraph`'s build-time index.

### 10.4 Implicit-`Joinset` enumeration

> **Phase 3 amendment (2026-05-28; cascade from C9.2 / C10.2).** The earlier framing positioned implicit-composition enumeration as **compile-time** materialization persisted on manifest. Per the ratified lightweight posture (manifest ratification log, C9.2 / C10.2) this is realigned: implicit-composition enumeration runs at **`SemanticGraph` build time** (in-memory, ephemeral), NOT at manifest compile (which would persist the enumerated set). The manifest carries only **primitives** and `SemanticGraph` runs the BFS / canonicalization / cap check below over those primitives at build, before the planner consumes the graph. There is **no** `compositions:` field on manifest (per `33 §4.1` and I12.15). The composition index, the synthetic-name / `ImplicitId` map, and the constituent-set lookup table referenced by [field-first-resolution](field-first-resolution.md) all live in `SemanticGraph`, not in the SemanticManifest. The `MAX_IMPLICIT_COMPOSITION_DEPTH` and `MAX_IMPLICIT_ENUMERATION_COUNT` constants below are **code-level invariants** in `semstrait-ir` (or the graph crate when materialized), not manifest-persisted values (per C10.2). The algorithm body is otherwise unchanged; the cascade rebases *where* it runs and *what surface* it writes to.

Algorithm at **`SemanticGraph` build** (in-memory, post-load), run after the graph reconstructs the `RelationshipGraph` adjacency from manifest `relationships:` (cf. `19 §3.4.2`'s shared infrastructure) and after explicit compositions have been admitted into the graph (§10.1):

```text
enumerate_implicit_joinsets(graph, explicit_canonical_forms) -> Vec<ResolvedJoinset>
    candidates: Set<CanonicalForm> = {}
    for each pair (A, B) of top-level kinds connected by ≥ 1 Relationship in graph:
        bfs_from(A, B, max_depth = MAX_IMPLICIT_COMPOSITION_DEPTH)
            for each path discovered (single edge or multi-hop):
                canonicalize(path) -> CanonicalForm
                candidates.insert(CanonicalForm)
    for each |T| in 3..(MAX_IMPLICIT_COMPOSITION_DEPTH + 1):
        steiner_enumerate(graph, |T|, max_depth = MAX_IMPLICIT_COMPOSITION_DEPTH)
            for each cover tree discovered:
                canonicalize(tree) -> CanonicalForm
                candidates.insert(CanonicalForm)
    if |candidates| > MAX_IMPLICIT_ENUMERATION_COUNT:
        emit COMP_E_0409 ImplicitEnumerationExploded { count, cap }
    return candidates
        .filter(|cf| cf NOT in explicit_canonical_forms)  // §10.6 clash check happens here
        .map(|cf| materialize(cf))
```

**Bounds:**

```rust
pub const MAX_IMPLICIT_COMPOSITION_DEPTH:   usize = 4;
pub const MAX_IMPLICIT_ENUMERATION_COUNT:   usize = 2000;
```

- `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` — Q-COMP-001 closed 2026-04-28. Covers Kimball star / snowflake / galaxy-via-bridge plus a 4-hop margin for cross-fact-via-shared-dim cases.
- `MAX_IMPLICIT_ENUMERATION_COUNT = 2000` — Q-COMP-005 closed 2026-04-29 (revised; was previously the strict-mode advisory question, now repurposed for the enumeration cap). Starting v1 value; revisitable post-v1 with telemetry. Cap-exceeded is a compile error, not a silent truncation — silent truncation would produce a SemanticManifest where some implicit compositions are enumerable and others are not, depending on enumeration order, a determinism violation.

**Canonicalization (Joinset):** sort the path / tree's `(EntityId, Direction)` tuples by `(EntityId, direction)` where `Direction::Forward < Direction::Reverse`. The sorted vector is the byte-encoded canonical form input to BLAKE3-256.

**Determinism:** neighbor iteration in `bfs_from` and `steiner_enumerate` is in **relationship-name order** (matching `19 §3.4.3`); this affects only enumeration/diagnostic ordering since the canonical form re-sorts by `(EntityId, Direction)` independently. The canonical-form sort is total; `BTreeMap` insertion order is canonical-form-sorted. The full enumeration is reproducible byte-for-byte given the same `RelationshipGraph`.

### 10.5 Implicit-`Unionset` enumeration

> **Phase 3 amendment (2026-05-28; cascade from C9.2 / C10.2 — same cascade as §10.4).** Implicit-Unionset enumeration runs at **`SemanticGraph` build time** (in-memory, post-load), not at manifest compile. Manifest carries only primitives and `SemanticGraph` derives the inverse-coverage map and runs the algorithm below over those primitives at planner setup. The pseudocode body is preserved.

Algorithm at `SemanticGraph` build, run after per-kind `Coverage` derivation (`15 §6`) and after the implicit-Joinset enumeration (§10.4):

```text
enumerate_implicit_unionsets(coverage_index, relationship_graph, explicit_canonical_forms)
    -> Vec<ResolvedUnionset>
    inverse_coverage: BTreeMap<SemanticsName, Set<DataKindRef>> = {}
    for each (kind, name) in coverage_index where coverage(kind, name) ∈ {Native, Derived, Metadata}:
        inverse_coverage[name].insert(kind)
    candidates: Set<CanonicalForm> = {}
    for each (semantics, kinds) in inverse_coverage where |kinds| ≥ 2:
        for each subset s ⊆ kinds where |s| ≥ 2:
            if no Relationship exists in relationship_graph between any pair in s:
                canonicalize(s) -> CanonicalForm
                candidates.insert(CanonicalForm)
    if |candidates| + |implicit_joinsets| > MAX_IMPLICIT_ENUMERATION_COUNT:
        emit COMP_E_0409 ImplicitEnumerationExploded { count, cap }
    return candidates
        .filter(|cf| cf NOT in explicit_canonical_forms)  // §10.6 clash check
        .map(|cf| materialize(cf))
```

**Why "no Relationship between any pair in `s`."** If a `Relationship` exists, the unified Joinset model handles it via implicit-Joinset enumeration (§10.4). The Unionset path is reserved for genuine coverage overlap — where the kinds are alternative sources of the same semantics, not joinable subsets.

**Canonicalization (Unionset):** sort `Vec<DataKindRef>` by `DataKindName` lex order. The sorted vector is the byte-encoded canonical form input to BLAKE3-256.

**Cap is shared with §10.4.** The enumeration cap counts implicit Joinsets + Unionsets together; the budget is per Model.

### 10.6 Implicit-explicit reconciliation — REJECT clashes (AUTHORITATIVE)

> **AUTHORITATIVE STATEMENT for `COMP_E_0414` (F1 fix).** This subsection is the single authoritative ratification of the explicit-implicit clash rule. Other leaves (e.g. [joinset-model](joinset-model.md) §13.5) reference this subsection rather than re-stating the rule.

A canonical form `cf` appearing in both the **explicit composition set** (declared `joinsets:` / `unionsets:` materialized in §10.1) and the **implicit candidate set** (§10.4 / §10.5) is a **clash**. Compile rejects with:

```text
CompileError::ExplicitImplicitCompositionClash {
    explicit_name:    DataKindName,
    canonical_kind:   CompositionKind,
    candidate_differentiators: Vec<&'static str>,
}
```

The Diagnostic message:

> Explicit `<kind>` `<name>` has the same canonical form as an enumerable implicit composition. Either remove the explicit declaration (the planner will use the equivalent implicit composition automatically) or differentiate by:
> - declaring per-leg `JoinType` overrides
> - adding `filter:` predicates
> - declaring `keys` that diverge from the anchor's primary key
> Implicit compositions enumerated against this canonical form are indexed under synthetic name `<__implicit_…>` (`ImplicitId`: `<hex>`).

**Why reject (not collapse, not coexist).** The user ratified "reject" as the v1 behavior:

- **Collapse** (Round-1 default proposal A) silently substitutes the explicit name for the implicit one. Risk: an author looking at the SemanticManifest sees only the explicit form, with no signal that the planner would have produced the same shape implicitly. The explicit declaration looks load-bearing when in fact it isn't.
- **Coexist** (proposal B) keeps both with distinct names. Risk: the field-first resolver finds two candidates with identical canonical form and must pick — exactly the ambiguity the rejection avoids.
- **Reject** (chosen) makes the redundancy a compile error. Authors who *want* an explicit form must add at least one differentiator; those without differentiators learn they can drop the declaration and rely on enumeration. This educates author intent and keeps the SemanticManifest's composition index minimal.

**`candidate_differentiators` payload.** Up to three suggested differentiations the author can add to make the canonical form distinct (e.g. `["join_type override on relationship_id=42", "filter: orders.status == 'open'", "keys: [orders.id, customers.id]"]`). The compiler computes these from the `Relationship` set's available overrides and surfaces them in the diagnostic per the `ContextLine`-style suggestion pattern (`30 §5.3`).

**Edge case — explicit Joinset's canonical form requires a hop count exceeding `MAX_IMPLICIT_COMPOSITION_DEPTH`.** No clash fires because the implicit enumeration would never produce that form. The explicit declaration stands on its own; it's the only path to the composition.

**Edge case — explicit Joinset declares non-shortest path.** No clash; non-shortest paths are not enumerated as implicit (§10.4 enumerates shortest only per `Q-COMP-002`). The explicit declaration is meaningful (it pins a specific non-default path).

## Testing Strategy

- **Invariants:** I5 (compile-time resolution — enumeration runs once at `SemanticGraph` build, not at plan), I8 (manifest carries primitives only; graph reconstructs compositions from `semantics` + `coverage` + `relationships:`), I12 (fail-fast: `COMP_E_0409` aborts compile, `COMP_E_0414` aborts compile).
- **Boundary rules:** Tests that implicit-Joinset enumeration respects `MAX_IMPLICIT_COMPOSITION_DEPTH` (4 hops), that `MAX_IMPLICIT_ENUMERATION_COUNT` (2000) cap fires `COMP_E_0409`, that bidirectional traversal produces forward and reverse variants, that transparent unfolding through composed surfaces works, that `COMP_E_0414` clash rejection fires when an explicit Joinset matches an enumerable implicit one.
- **Canonical-form stability:** Tests that `ImplicitId` hashes deterministically, that sorted `(EntityId, Direction)` tuples produce identical canonical forms across recompiles (when relationship `EntityId`s are stable).
- **Error codes:** `COMP_E_0409` (enumeration cap exceeded), `COMP_E_0414` (explicit-implicit clash, §10.6 — authoritative), `PLAN_E_0500` (path ambiguity), `PLAN_E_0501` (no composition path), `PLAN_E_0502` (depth exceeded) — see [composition-validation-and-errors](composition-validation-and-errors.md).
- **Behavioral gates:** Tests that implicit-Joinset enumeration at `SemanticGraph` build produces all pairwise shortest paths ≤4 hops, that implicit-Unionset enumeration produces coverage-overlap sets with no connecting Relationship, that explicit compositions materialize with `Origin::Explicit` and implicit with `Origin::Implicit { id }`, that field-first resolution (see [field-first-resolution](field-first-resolution.md)) finds both explicit and implicit via the same lookup path.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
