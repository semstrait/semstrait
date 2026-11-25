---
doc: design/questions/closed/16_questions
status: Closed
purpose: Resolved questions originally raised against `foundations/16_composition.md`
---

# Closed Questions — `foundations/16_composition.md`

> Historical record of ratified composition decisions. Live items are in [`../open/16_questions.md`](../open/16_questions.md). Round-2 closures (2026-04-29) introduced the unified Joinset model; older Round-1 framing is preserved on each entry for historical reference.

---

## Q-COMP-001 — `MAX_IMPLICIT_COMPOSITION_DEPTH` value — CLOSED (2026-04-28)

**Status: CLOSED.** Round-1 default `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` ratified for v1. Covers Kimball star / snowflake / galaxy-via-bridge patterns (typical depth ≤ 3) plus a 4-hop margin for cross-fact-via-shared-dim cases; brute-force Steiner enumeration (Q-COMP-003) stays sub-millisecond at this depth on typical 10–100 edge graphs; deeper analytic walks (hierarchical, social-graph) are channeled to explicit `Joinset` per `16 §13`. Review trigger preserved: `34` drafting + early-usage telemetry; raising to `6` is MINOR if `PLAN_E_0502 CompositionDepthExceeded` fires on legitimate models. Round-1 framing retained for historical reference.

**Question.** `16 §9.1` bullet 3 ratifies a depth limit on implicit composition. Round-1 sets `MAX_IMPLICIT_COMPOSITION_DEPTH = 4`. Is `4` the right value for v1?

**Refs.**

- `16 §9.1` — boundary ratification.
- `16 §14.3` — `PLAN_E_0502 CompositionDepthExceeded`.
- `34` (pending) — planner entry point; exposes the limit as a constant.

**Proposed (Round 1):** `4` hops. Covers common star / snowflake patterns (fact → dim → outrigger → …) without endorsing arbitrarily deep anonymous walks. Authors wanting deeper compositions are nudged toward explicit `Joinset`.

**Arguments for `4`.**

- Kimball-style analytic workloads rarely exceed 3-hop dim outrigger chains.
- Larger values admit implicit compositions authors likely did not mean; smaller values reject legitimate patterns.
- Matches the industry-folklore "keep dim hierarchies shallow" guidance.

**Arguments for a larger value (say, `8`).**

- Some graph-oriented models (e.g. organizational hierarchies, social-network-style aggregations) genuinely need deeper walks.
- Larger limit defers the "switch to explicit `Joinset`" moment.

**Arguments for a smaller value (say, `2`).**

- Encourages explicit modeling earlier.
- Safer default for Round-1 when the implicit-composition algorithm has no real-world usage.

**Current position in `16`.** `4` hops. The constant lives in `semstrait-planner` and is not author-configurable in v1.

**Next step.** Revisit at `34` drafting with real-Model sample data; if early usage consistently runs into `PLAN_E_0502` on sane models, raise to `6` or `8`. If early usage indicates authors are accidentally constructing deep compositions, lower to `3`.

---

## Q-COMP-002 — Ambiguous-path tie-breaking: could there be a deterministic heuristic? — CLOSED (2026-04-28; clarification 2026-04-29)

**Status: CLOSED.** Error-on-tie ratified for v1. Path ambiguity (e.g. `orders → billing_address → city` vs `orders → shipping_address → city`) is a **semantic** ambiguity producing distinct aggregates; a heuristic (lexically-smallest `RelationshipId`, fewest `ManyToMany` edges) silently picks one and authors with the alternate intent get a wrong answer with no signal — a correctness regression, not an ergonomics improvement. Explicit `Joinset` is the well-designed escape hatch. Diagnostic-shape sharpening (suggesting candidate paths in the error message) is tracked separately under `Q-COMP-014`. A heuristic-as-opt-in CLI flag remains a candidate post-v1 extension if usage data shows authors hitting this on safe models, but never as the default. Round-1 framing retained for historical reference.

**Clarification (2026-04-29, unified Joinset model).** Path ambiguity differs from **coverage ambiguity**:

- **Path ambiguity** — multiple compositions of equal canonical cost cover the same Request constituent set (e.g. billing-vs-shipping address paths). v1 ratification: error at plan time (`PLAN_E_0500`) per `16 §11.4`. The implicit-composition enumeration in `16 §10.4` produces both candidates as distinct implicit Joinsets (different `ImplicitId`s); ambiguity surfaces only when a Request's selected fields could be served by either. Authors disambiguate by declaring an explicit `Joinset` with at least one differentiator (per `§10.6`'s clash-rejection escape pattern).
- **Coverage ambiguity** — N independent top-level kinds (no `Relationship` between them) cover the same Semantics. v1 ratification: synthesize an implicit `Unionset` per `16 §10.5`. No error; the planner builds the union automatically per the unified model.

Both cases were previously thought of as "ambiguity errors"; the 2026-04-29 unified Joinset model makes coverage ambiguity a non-error (Unionset synthesis), reserving the error path for true semantic-distinct-paths cases.

**Question.** `16 §11.4` / `§9.1` bullet 2 ratifies "ties → error, no heuristic choice." Alternatives exist — e.g. prefer the path whose `RelationshipId`s are lexically smallest, or prefer the path with fewer `ManyToMany` edges. Should v1 adopt any such heuristic?

**Refs.**

- `16 §11.4` — BFS determinism via neighbor order.
- `16 §9.4` bullet 2 — rationale for error-on-tie.
- `16 §14.3` — `PLAN_E_0500 AmbiguousImplicitComposition`.

**Proposed (Round 1):** Error on ties. Authors must either declare an explicit `Joinset` (pinning the path) or remove the ambiguity by deleting / narrowing one of the candidate `Relationship`s.

**Arguments for error.**

- I4 determinism: the author can always predict the answer because "tie = error" is a clear rule.
- Heuristics inject judgment authors cannot trace without reading the planner internals.
- Explicit `Joinset` is the escape hatch and is designed for exactly this case.

**Arguments for a heuristic.**

- Rejects legitimate queries on Models with natural cross-join-key structures (e.g. a fact with two FKs to the same dim table via different semantic roles).
- Authors may prefer "some answer" over a compile error for exploratory queries.

**Current position in `16`.** Error on ties. Authors disambiguate by declaring a `Joinset`.

**Next step.** Revisit post-v1 if user studies show the error fires on legitimate work more than it rejects malformed queries. A candidate heuristic (lexically-smallest `RelationshipId`) could be added as an opt-in planner flag.

---

## Q-COMP-003 — Implicit-composition Steiner-tree solver sophistication — CLOSED (2026-04-28)

**Status: CLOSED.** Brute-force enumeration ratified for v1. Three reasons: (1) exact results matter — 2-approximation algorithms trade exact answers for speed, the wrong trade-off when the planner's cover-tree choice determines query semantics; (2) v1 scale (10s–100s of `Relationship`s, `|T|` 2–4, depth ≤ 4) is well within brute-force budget (sub-millisecond on typical Models); (3) determinism is structurally cheap with sorted enumeration order. Future-proofing tracked as `[TD-COMPOSITION-STEINER-SOLVER]`: if profiling surfaces implicit composition as a hot path on pathological Models, swap to a polynomial-time exact solver (e.g., dynamic-programming Steiner for small `|T|`) — preserves exactness and the surface contract. Approximation enters only as a last resort with author opt-in. Round-1 framing retained for historical reference.

**Question.** `16 §11.4`'s "multi-target BFS" is a simplified Steiner-tree approximation: find a subgraph connecting all owning kinds with minimum total hop count. Should v1 use an optimal Steiner-tree solver (NP-hard in general) or is the BFS approximation sufficient?

**Refs.**

- `16 §11.4` — multi-target BFS.
- `16 §11.5` — synthesis consumer.

**Proposed (Round 1):** BFS approximation. Graph size is small enough (10s–100s of `Relationship`s) that exhaustive enumeration of candidate cover trees up to `MAX_IMPLICIT_COMPOSITION_DEPTH` is feasible. For typical Models the approximation coincides with the optimum.

**Arguments for the approximation.**

- Round-1 scale (10s of DataKinds, 10s–100s of Relationships) is well within brute-force enumeration.
- Exact Steiner tree is NP-hard; investing in a sophisticated solver is premature.
- Determinism is easy to maintain with enumeration.

**Arguments for a sophisticated solver.**

- Pathological Models (thousands of kinds, dense Relationship graphs) would benefit.
- If the planner's implicit-composition time becomes a hot path, better algorithms matter.

**Current position in `16`.** Brute-force BFS enumeration within the depth bound.

**Next step.** Revisit when / if profiling shows implicit composition as a hot path. Well-established 2-approximation algorithms for Steiner tree exist and can be dropped in without changing the surface contract.

---

## Q-COMP-004 — Should implicit composition produce a `Joinset`-style surface or a bespoke `Relationship`-kind surface? — CLOSED (2026-04-29) — OVERRIDE

**Status: CLOSED with override.** Round-1's "keep distinct" answer is **superseded** by the unified Joinset model (2026-04-29). Implicit compositions are now `CompositionKind::Joinset` (or `Unionset`) with `Origin::Implicit { id: ImplicitId }`. The Round-1 fourth variant `CompositionKind::Relationship` is **retired**.

**Rationale for override.**

- Round-1 distinct-variant model treated implicit compositions as ephemeral request-scoped surfaces; the unified model materializes them at compile time, giving every composition the same `ResolvedComplexDataKind` shape.
- The same plan-time strategy code now serves explicit and implicit Joinsets identically — they share `traversed_paths`, `anchor`, `keys`, and lowering. The discriminator that mattered is `composition_kind` (Joinset vs Unionset vs Grainset); `Origin` carries the provenance distinction without forking strategy logic.
- The implicit-explicit clash check (`16 §10.6`, `COMP_E_0414`) makes the Joinset namespace single-source-of-truth: an explicit Joinset whose canonical form matches an enumerable implicit one is rejected, preventing duplicate composition entries.

**Refs.**

- `16 §5.3` — `CompositionKind` retains `{Joinset, Unionset, Grainset}`.
- `16 §5.6` — `Origin` axis.
- `16 §5.7` — `ImplicitId` canonical form.
- `16 §10` — eager-materialization policy.
- `16 §13.5` — implicit-explicit reconciliation.

**Round-1 framing retained for historical reference.** Original argument for distinctness (clean mental model: named/persisted → `Joinset`, anonymous/request-local → `Relationship`) was sound under the on-demand synthesis model. The 2026-04-29 shift to compile-time eager materialization removed the "request-local" justification; the unified model is now structurally and ergonomically simpler.

---

## Q-COMP-005 — Scope of `PLAN_W_0501 FanoutAdvisory`: error-by-default under `strict` mode? — CLOSED (2026-04-29) — OVERRIDE

**Status: CLOSED with override.** The advisory is **retired in v1** rather than promoted to a strict-mode error. Per `16 §14.4` (revised 2026-04-29), `PLAN_W_0501 FanoutAdvisory` and `PLAN_W_0502 ManyToManyFanoutAdvisory` are removed because fanout is the natural consequence of the author's declared `Cardinality` and `Additivity` — v1 trusts those declarations rather than second-guessing intent. Authors who want fanout-detection in their workflow can author a separate audit query.

**Strict mode considered and deferred.** Re-introducing the advisory under a future `strict` mode is tracked as `[TD-COMPOSITION-FANOUT-ADVISORY]` in `16 §16`. The slot codes `0501` / `0502` are reserved (no MINOR re-allocation) so a future v2 with telemetry can re-introduce them under `strict` mode if warranted.

**Note on slot repurposing.** The 2026-04-29 ratification *also* uses Q-COMP-005's slot to record the implicit-enumeration cap decision: `MAX_IMPLICIT_ENUMERATION_COUNT = 2000` for v1 (`16 §10.4`). The cap and the advisory drop are part of the same unified-Joinset-model batch.

**Round-1 framing retained for historical reference.** Original arguments for strict-mode promotion (analytic-engineering teams want fanout fails-compile) remain valid for a future `strict` mode. The 2026-04-29 ratification explicitly defers them.

**Refs.**

- `16 §3.3.2` — fanout-safe rewrite description (kept; unaffected by advisory drop).
- `16 §14.4` — advisory roster (revised: only `PLAN_W_0503` and `COMP_W_0401` remain in v1).
- `16 §10.4` — `MAX_IMPLICIT_ENUMERATION_COUNT` cap.
- `16 §16` — Round-2 closure index (Q-COMP-005, `[TD-COMPOSITION-FANOUT-ADVISORY]` deferred).

---

## Q-COMP-006 — Cross-composition-kind chaining (the §9.1 bullet 5 prohibition) — CLOSED (2026-04-29) — OVERRIDE

**Status: CLOSED with override.** The prohibition is **retired**. Per `16 §9.1` bullet 7 (revised 2026-04-29), implicit-composition enumeration walks **transparently** through composed surfaces — a `Unionset` or `Joinset` constituent is treated as the union of its constituents during canonical-form construction. The `PLAN_E_0504 CompositionChainingForbidden` error is retired (the code is reserved for forward-compat).

**Rationale for override.**

- Round-1 prohibition was based on the Round-1 plan-time synthesis model: walking through a freshly-synthesized composed surface created correctness hazards (which side of a composed `Shared` field "owns" onward-composition?).
- The 2026-04-29 unified Joinset model resolves the hazard: implicit-composition enumeration runs at compile over the **unfolded graph**, not against already-materialized composed surfaces. The unfolded graph is a flat `RelationshipGraph` with composed kinds expanded to their constituents.
- The implicit-explicit clash check (`§10.6`) prevents accidental duplication: if an author has explicit `Joinset` X over kinds {A, B} and implicit enumeration produces a Joinset over {A, B} by walking through some intermediate composed kind, the canonical-form match triggers `COMP_E_0414`.
- "Query a Unionset and pull in a related dimension via a Relationship" — a common pattern previously forced into explicit Joinset declaration — is now naturally supported.

**Refs.**

- `16 §9.1` bullet 7 — transparent-unfolding rule (replacing the prohibition).
- `16 §9.4` "Why transparent unfolding through composed surfaces" — rationale.
- `16 §10.4` — implicit-Joinset enumeration walks the unfolded graph.
- `16 §13.5` — explicit `Relationship`s into composed kinds permitted; implicit enumeration handles them.

**Round-1 framing retained for historical reference.** The Round-1 prohibition was sound under plan-time synthesis; it's superseded by compile-time enumeration over the unfolded graph plus the clash check.

---

## Q-COMP-011 — `traversed_paths` on implicit compositions: single path vs path per leg — CLOSED (2026-04-29)

**Status: CLOSED.** `Vec<RelationshipPath>` per leg ratified. Tree covers are real for compositions over 3+ owning kinds (Steiner tree per `16 §10.4`); the per-leg shape generalizes to single-leg paths as a degenerate case (single-element `Vec`). The unified Joinset model preserves this shape — the `traversed_paths` field on `ComposedSemanticInterface` is the same shape for `Origin::Explicit` (author-declared multi-leg paths, deferred to `[TD-JOINSET-NARY]`) and `Origin::Implicit` (compile-enumerated multi-leg covers).

**Refs.**

- `16 §5.2` — `traversed_paths: Vec<RelationshipPath>` carriage.
- `16 §10.4` — implicit-Joinset Steiner-tree enumeration.

**Round-1 framing retained for historical reference.**

---

## Q-COMP-012 — `Joinset` reuse optimization (`[TD-COMPOSITION-JOINSET-REUSE]`) — CLOSED (2026-04-29) — OVERRIDE

**Status: CLOSED with override.** The reuse-vs-synthesis question is **dissolved** by the unified Joinset model (2026-04-29). Under the new model, explicit Joinsets and implicit Joinsets share the same `ResolvedJoinset` shape and are addressed via the same field-first lookup. The `[TD-COMPOSITION-JOINSET-REUSE]` tech-debt marker is **retired** — it was a pointer to a question that no longer exists.

**Rationale for override.**

- Round-1 model had implicit compositions synthesized per Request and explicit Joinsets pre-materialized; "reuse" would mean substituting the explicit form when the implicit walk happened to match. The asymmetry created the open question.
- The 2026-04-29 unified model materializes both at compile. The Round-1 distinction "reuse vs synthesize" doesn't translate — there's no synthesis to swap against.
- The implicit-explicit clash rule (`16 §10.6`, `COMP_E_0414`) handles the canonical-form-collision case **at compile** by rejecting it. Authors who declare an explicit `Joinset` that exactly matches an enumerable implicit canonical form are required to add a differentiator (per-leg `JoinType` override, `filter:`, declared `keys`, or non-shortest path); otherwise compile fails. This forces the canonical compositions to be unique by construction — no reuse logic needed at plan time.
- "The `Joinset` is the canonical composition for these kinds; why synthesize another?" becomes literally true: there's only ever **one** canonical-form-matching composition in the SemanticManifest, addressed under either the author's name (explicit) or the synthetic `__implicit_…` name (implicit).

**Refs.**

- `16 §10.6` — clash-rejection rule.
- `16 §13.5` — reconciliation policy (rewritten to replace "no reuse").
- `16 §16` — `[TD-COMPOSITION-JOINSET-REUSE]` removed from deferred-to-v2 list.

**Round-1 framing retained for historical reference.** Original arguments for and against reuse are valid only under the synthesis model; the unified model makes them moot.

---

## Q-COMP-013 — Should explicit `Relationship`s between composed kinds (e.g. `Joinset → Simple`) be permitted? — CLOSED (2026-04-29)

**Status: CLOSED.** Permitted. A `Relationship` declares joinability between any two top-level kinds, including composed ones (`Simple`, `Unionset`, `Grainset`, `Joinset`). Per `16 §2.1`, the `KeyPair.left` / `.right` references a namespaced `SemanticsName` within the composed surface (e.g. `"paid_media.campaign_id"`).

**Updated under unified Joinset model (2026-04-29).** The Round-1 prohibition on chaining different `CompositionKind`s in implicit walks (§9.1 bullet 5) is **retired** (per `Q-COMP-006` closure). Implicit-composition enumeration in `16 §10.4` walks transparently through composed surfaces — a `Relationship` between a `Joinset` and a `Simple` is a first-class edge **and** participates in implicit enumeration when the constituents fit within the depth + cap bounds.

**Refs.**

- `16 §2.1` — placement (composed kinds permitted on either side).
- `16 §9.1` bullet 7 — transparent unfolding through composed surfaces.
- `16 §13.5` — explicit `Relationship`s referencing composed kinds participate in implicit enumeration.
- `12 §2` — nesting matrix.
- `32` (YAML surface) — namespaced `SemanticsName` authoring shape (`"composed_kind_name.semantics_name"`) for `KeyPair.left` / `.right`.

**Open follow-up (deferred):** `32` YAML surface ergonomics for namespaced `SemanticsName` references. Tracked as authoring-surface concern, not a structural one.

**Round-1 framing retained for historical reference.**

---

## Q-COMP-007 — `Directionality` granularity: per-`Relationship` vs per-direction — CLOSED (2026-05-12)

**Status: CLOSED — moot.** The authoring-layer `Directionality` enum is retired with the relationship-block rebase (item K, STATUS §2). Every `Relationship` is bidirectional by construction (`16 §2.4`). Authors who need to restrict traversal direction declare an explicit `Joinset` for the desired direction; there is no per-Relationship directionality field on the canonical struct.

**Rationale.** A per-Relationship directionality flag was an awkward fit alongside the new semantic-first fields (`integrity`, `optional`, `cross_filter`): the `cross_filter` enum already expresses directional filter-flow semantics, and the legitimate use-cases for forbidding reverse traversal (preventing auto-synthesized backwards walks for event-log fanout, slow-changing-dim lookups, etc.) are better served by explicit Joinset declarations — the author already needed to think about the direction of the resulting analytic, and that intent belongs at the Joinset site.

**Refs.**

- `16 §2.4` — traversal always bidirectional (rewritten 2026-05-12).
- `18 §2` — Relationship struct no longer carries `directionality`.
- STATUS reconciliation item K (2026-05-12).

**Round-1 / Round-2 framing retained for historical reference.** The original arguments for per-direction flags vs enum are valid only under the per-Relationship-directionality model; the unified bidirectional model makes them moot.

---

## Q-COMP-008 — `Directionality::Forward` reverse-traversal error surface — CLOSED (2026-05-12) — induced by Q-COMP-007

**Status: CLOSED — moot.** Induced retirement following Q-COMP-007: the `Directionality::Forward` variant no longer exists, so there is no reverse-traversal error to time. `PLAN_E_0503 CrossCompositionForbidden` is retired (`16 §14.3`; code reserved for forward-compat). `COMP_E_2409 JoinsetReverseForwardOnlyRelationship` is retired (`24 §10.3`; code reserved).

**Refs.**

- Q-COMP-007 — parent retirement.
- `16 §14.3` — `PLAN_E_0503` retirement note.
- `24 §10.3` — `COMP_E_2409` retirement note.

---

## Q-COMP-017 — Should `Relationship.join_type` default differ from `JoinType::Inner`? — CLOSED (2026-05-12)

**Status: CLOSED — moot.** The question of a YAML default for `join_type:` no longer applies: `JoinType` is **not authored** under the relationship-block rebase (item K). It is derived at compile from `Relationship.optional` per `18 §2.9`'s one-to-one derivation table (`None → Inner`, `Left → Left`, `Right → Right`, `Both → Full`). The YAML surface (`32`) carries `optional:` (with a defaults matrix per `18 §2.7`); `join_type:` is removed from the authoring surface entirely.

**Effective resolution.** For `cardinality: many_to_one + integrity: assumed` (the canonical fact-→-dim case), the matrix defaults to `optional: none` which derives `Inner` — matching the original Round-1 preference for `Inner` as the implicit default while keeping the authoring surface semantic-first.

**Refs.**

- `18 §2.7` — defaults matrix.
- `18 §2.9` — `optional → JoinType` derivation table.
- `16 §4` — `JoinType` reframed as derived, not authored.
- STATUS reconciliation item K (2026-05-12).

**Round-1 framing retained for historical reference.**

---

## Q-COMP-018 — `ComposedSemanticInterface.keys` on implicit compositions: empty vs derived? — CLOSED (2026-04-29) — OVERRIDE

**Status: CLOSED with override.** Round-1's "empty for implicit" answer is **superseded** by the unified Joinset model (2026-04-29). Per `16 §6.5` (revised), composed-surface keys are **declare-or-derived** for every Joinset, regardless of `Origin`:

- **`Origin::Explicit` Joinsets.** Author MAY declare `keys` on the `joinsets:` block. If declared, those keys win. Otherwise, derive from the anchor constituent's `Key::Primary`.
- **`Origin::Implicit` Joinsets.** Always derive from the anchor constituent (the first `DataKindRef` in the canonical `constituents` order, which corresponds to the canonical-form starting node). Same rule as the no-keys-declared explicit case.

**Rationale for override.** The Round-1 "empty for implicit" answer was rooted in the request-scoped synthesis model — implicit compositions were ephemeral, so the field had no permanent author surface. The 2026-04-29 unified model materializes implicit compositions in the SemanticManifest, making the field meaningful: planner strategies (deduplication, GROUP BY pins, certain optimizer rewrites) rely on a populated `keys` field. Empty keys would force the planner into a fallback path; deriving keys eliminates the fallback. Authors don't see the derived keys directly (they're addressed via the synthetic `__implicit_…` name) but planner internals benefit.

**Refs.**

- `16 §6.5` — declare-or-derive rule (revised 2026-04-29).
- `16 §10.4` — implicit-Joinset enumeration assigns canonical `constituents` order; first element is the anchor.
- `16 §16` — Q18 added to ratified-decisions index.

**Round-1 framing retained for historical reference.**

---

## Q-COMP-010 — `CompositionCoverage` — serialize per-constituent or collapsed? — CLOSED (2026-05-28)

**Status: CLOSED.** Settled by manifest ratification clause CCK.3 (Coverage Kernel; see `_research/manifest/RATIFICATION_LOG.md`, 2026-05-28).

**Resolution.** Per-constituent local masks live on the variant struct (Dataset / Grainset / Joinset / Unionset) — not at the top-level DataKind layer. The `CompositionCoverage` map is **tuple-keyed** by `(ConstituentRef, SemanticsName)`, preserving symmetry with `15 §6` binding-level Coverage. Round-1 default (per-constituent serialization) is upheld — collapsed serialization was rejected because it would erase the per-constituent provenance that planner strategies (composition-aware pushdown, per-constituent NullFill discipline) need at lookup time.

**Refs.**

- `_research/manifest/RATIFICATION_LOG.md` — CCK clause group, in particular CCK.3 (per-constituent local masks; tuple-keyed map shape).
- `16 §8.2` — `CompositionCoverage` keying.
- `15 §6` — binding-level `Coverage` parallel structure.

**Resolution rationale.** Per-constituent serialization keeps the composition-side coverage shape parallel to the binding-side coverage shape (`SemanticBinding::coverage`, keyed per source). The collapsed alternative (a flat `BTreeMap<SemanticsName, CoverageVariant>`) would have required the planner to re-derive provenance via constituent-level lookups on every consumer call — unwarranted overhead given the small tuple-key footprint.

**Question (closed scope).** `16 §8.2` ratifies `CompositionCoverage` carrying `(ConstituentRef, SemanticsName) → CoverageVariant`. Does the SemanticManifest serialize this per-constituent (preserving the tuple-key structure) or as a single collapsed `BTreeMap<SemanticsName, CoverageVariant>` with collision-merge semantics?

**Round-1 framing retained for historical reference.**
