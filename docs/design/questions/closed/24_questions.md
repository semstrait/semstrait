---
doc: design/questions/closed/24_questions
status: Closed
purpose: Resolved questions originally raised against `data-kinds/24_joinset.md`
---

# Closed Questions — `data-kinds/24_joinset.md`

> Historical record of ratified Joinset decisions. Deferred items in [`../deferred/24_questions.md`](../deferred/24_questions.md). The author-facing `data-kinds/24_joinset.md` doc is the canonical home for all v1 Joinset semantics.

---

## Q-24-02 — Hybrid path mode — CLOSED (2026-04-28)

**Status: CLOSED.** v1 ratifies "no hybrid path" — `24 §4.3` makes `path` strictly `None` (implicit) or fully declared (explicit). `TD-JOINSET-HYBRID-PATH` carries the post-v1 reactivation. Round-1 framing retained for historical reference.

**Question.** `24 §4.3` ratifies that `path` is either fully implicit (`None`) or fully explicit (`Some(ExplicitPath)`). Could there be a hybrid mode where the author declares some hops and lets the planner fill the remainder via BFS?

**Refs.**

- `24 §4.3` — mode-selection precedence.
- `24 §4.1` — implicit-path algorithm.
- `24 §4.2` — explicit-path algorithm.

**Current position.** Prohibited. Authors who need partial control either declare the full explicit path or declare multiple binary Joinsets, chaining via composition kinds that `12` permits.

---

## Q-24-03 — Override reach: `Cardinality` overrides? — CLOSED (2026-04-28)

**Status: CLOSED.** v1 ratifies "no `Cardinality` override". The declared `Relationship.cardinality` is authoritative for all consumers; sophisticated authors restructure the Model (junction-table DataKind per `16 §3.3.4`) rather than override. Round-1 framing retained for historical reference.

**Forward note (2026-05-12, item K).** The entire override surface (`JoinTypeOverrides` / `HopPosition`) is retired with the relationship-block rebase (see `Q-24-XX` below). The original "no `Cardinality` override" answer is unchanged in effect — a Joinset that asserts different cardinality declares a scope-local `Relationship` (`18 §2.10`), which is an independent declaration, not an "override" of the root-level claim. The original closure rationale (declared `cardinality` is authoritative; junction-table modeling for tighter constraints) carries through to the scope-local model.

**Question.** `24 §5.3`'s `JoinsetStrategy` override mechanism currently covers `JoinType` only. Should `overrides` also permit pinning a hop's effective `Cardinality` — e.g. declaring "treat this `ManyToMany` hop as `OneToMany` for this Joinset because the author knows the Joinset's slice satisfies the tighter constraint"?

**Refs.**

- `24 §5.3` — `JoinType` override matrix.
- `24 §5.4` — `Cardinality` propagation and fanout profile.
- `16 §3.5` — `Cardinality × Additivity` matrix (drives fanout rewrites).

**Current position.** No `Cardinality` override.

---

## Q-24-04 — `AsOf` activation matrix — CLOSED (2026-04-28)

**Status: CLOSED.** `17 §5.2` ratifies the per-shape-pair legality matrix; `24 §7.3` cross-references it. `TD-COMPOSITION-ASOF` continues to track the planner-side `AsOf` implementation deferral, but the activation-matrix question itself is settled. Round-1 framing retained for historical reference.

**Question.** `24 §7` fixes the integration points for `TemporalShape × JoinType::AsOf` but defers the exact activation matrix to `17 §5`. What are the precise `TemporalShape` pairs that mandate / permit / forbid `AsOf`?

**Refs.**

- `24 §7.1–§7.3` — Joinset's contract re `AsOf`; error codes `COMP_E_2412`–`COMP_E_2414`.
- `16 §4.4.2` — `AsOf` deferral; `TD-COMPOSITION-ASOF`.
- `17 §5` — activation matrix.
- `00 §4.1` — `TemporalShape` vocabulary row.

**Sketch of the ratified matrix.**

| anchor-side `TemporalShape` | target-side `TemporalShape` | `AsOf` activation |
|---|---|---|
| `Events` | `Snapshot` | Mandated (canonical as-of case: events-as-of-snapshot). |
| `Events` | `Scd` | Mandated (canonical as-of case: events-as-of-SCD). |
| `Timeseries` | `Snapshot` | Permitted, not mandated. |
| `Snapshot` | `Snapshot` | Forbidden (both already time-point-indexed; as-of is ill-defined). |
| `Timeseries` | `Timeseries` | Forbidden (same-grain timeseries joins on time are equality, not as-of). |
| `Events` | `Events` | Forbidden (no stable reference frame for "most-recent"). |

**Current position.** Integration points fixed in `24 §7.3`; matrix ratified in `17 §5.2`.

---

## Q-24-05 — Joinset reuse by implicit composition — CLOSED (Round 2, 2026-04-29)

**Status: CLOSED with override.** Round-1's "no reuse" answer is **superseded** by the unified Joinset model (Round 2, 2026-04-29). Implicit Joinsets are now first-class `Origin::Implicit { id: ImplicitId }` Joinsets enumerated at compile per `16 §10.4`; explicit Joinsets are `Origin::Explicit`. The two share `CompositionKind::Joinset` and a uniform `ResolvedJoinset` shape — there is no separate "implicit synthesis" surface to reuse. The reuse-vs-no-reuse question is dissolved by the architectural shift.

**Round-2 closure mechanics.** When an author's explicit Joinset's canonical form (`16 §5.7`) hashes to an `ImplicitId` already enumerated implicitly, the **clash check** (`16 §10.6`) rejects the explicit declaration with `COMP_E_0414 ExplicitImplicitCompositionClash`. Authors disambiguate by adding a per-leg `JoinType` override, a `filter:` clause, declared `keys`, or a non-shortest path (per the differentiator menu in `16 §13.5`). Otherwise, when the canonical forms differ, both coexist in the SemanticManifest as distinct compositions and field-first lookup over `composition_index.by_constituent_set` (`33 §7.2`) returns a unique winner per `16 §11.4`.

**`[TD-COMPOSITION-JOINSET-REUSE]` retired** alongside this closure — the mechanism it tracked (post-v1 reuse logic) is replaced by compile-time eager materialization with clash rejection.

**Round-1 framing — historical.**

> **Question.** `16 §13.5` (Round 1) records that an explicit `Joinset` does NOT shadow implicit composition: a `Request` with `from: None` over the same constituents produces a **distinct** `ComposedSemanticInterface` with `CompositionKind::Relationship`, not the Joinset's `CompositionKind::Joinset`. Should the planner learn to recognize the coincidence and substitute the pre-built Joinset surface?
>
> **Round-1 position.** No reuse. The two surfaces are semantically distinct objects; conflation risks behavior drift when the Joinset's overrides differ from the implicit composition's defaults.

**Refs.**

- `16 §10.4` — implicit-Joinset enumeration (Round 2).
- `16 §10.6` — implicit-explicit reconciliation via clash rejection.
- `16 §13.5` — explicit-implicit reconciliation under the unified model.
- `16 §5.6` — `Origin` axis on compositions.
- `33 §7.2` — `composition_index.by_constituent_set` for plan-time lookup.

---

## Q-24-06 — Self-referential Joinsets — CLOSED (2026-04-28)

**Status: CLOSED.** Forbidden in v1, transitively from `16 §12.4` (Relationship self-references forbidden) and the validate-layer rejection `VALID_E_2406 JoinsetDuplicateMember` (`24 §9.1`). `TD-COMPOSITION-SELFJOIN` continues to track the post-v1 lift. Round-1 framing retained for historical reference.

**Question.** Can a Joinset's anchor and target be the same DataKind (e.g. `employees` joined to itself along a `manager_id → id` relationship)?

**Refs.**

- `16 §12.4` — Relationship self-references forbidden in v1; `TD-COMPOSITION-SELFJOIN`.
- `24 §4.2.3` — restatement: since no self-referential Relationships exist in v1, self-referential Joinsets are structurally unreachable.
- `24 §9.1 VALID_E_2406 JoinsetDuplicateMember` — rejects `members = [X, X]` at the validate layer.

**Current position.** Forbidden. Tracked jointly with `TD-COMPOSITION-SELFJOIN`.

---

## Q-24-07 — Per-hop filter pushdown annotations — CLOSED (2026-04-28)

**Status: CLOSED.** v1 ratifies "no per-hop filters". `JoinHop` carries only `relationship`, `direction`, `to` (`24 §4.2`). Filters are declared at the Joinset level (`§2.6`) and the planner pushes them where safe. Authors needing per-hop scoping push the filter into the member's own interface or declare a narrower member DataKind. Round-1 framing retained for historical reference.

**Forward note (2026-05-12, item K).** With the override-surface retirement, the scope-local `Relationship` mechanism (`18 §2.10`, `16 §13.3`) becomes the natural home for per-hop concerns that diverge from the root-level declaration — including residual `filter:` predicates that only apply within this Joinset. The original "no per-hop filters as `JoinHop` field" answer carries through; the scope-local Relationship's `filter:` field provides the equivalent expressive power at the Relationship layer, which is structurally cleaner.

**Question.** Should `ExplicitPath.hops[i]` permit a per-hop filter expression — e.g. "only join with `addresses` where `country = 'US'`" — declared at the Joinset level?

**Refs.**

- `24 §4.2` — `JoinHop` struct (no filter field in Round 1).
- `24 §2.6` — Joinset-level Filter declarations (applied post-join, not per-hop).
- `14` — expression grammar.
- `34` — planner pushdown; selection-pushdown already an optimizer concern.

**Current position.** Joinset-level filters only.

---

## Q-24-08 — Structural `NullFill` for outer-join Joinsets — CLOSED (2026-04-28)

**Status: CLOSED.** v1 ratifies "no structural NullFill for Joinset". `FieldOwnership::NullFill` is Unionset-only per `16 §7.3.3`; outer-join NULL-fill is the `JoinType` at the plan tree's responsibility (`24 §5.5` step 3). Round-1 framing retained for historical reference.

**Question.** `16 §7.3.3` ratifies that `FieldOwnership::NullFill` is produced ONLY for `CompositionKind::Unionset`. For a `Joinset` with a `Left` / `Right` / `Full` outer join, missing-side columns are NULL-filled by the JoinType's semantics rather than recorded structurally on `FieldProvenance`. Should Joinset-side outer-join NULL-fill be recorded structurally?

**Refs.**

- `16 §7.3.3` — `NullFill` is Unionset-only in Round 1.
- `24 §8.3` — Joinset's `FieldProvenance` consequences; Joinset `FieldProvenance` has no `NullFill` entries.
- `24 §5.5` step 3 — SQL-side NULL-fill handled by JoinType at emission time.

**Current position.** Follow `16 §7.3.3`. Join-side NULL-fill is a JoinType concern, not a FieldProvenance concern.

---

## Q-24-11 — `JoinTypeOverrides` / `HopPosition` per-hop override surface — CLOSED (2026-05-12)

**Status: CLOSED — retired.** The per-hop override surface on `JoinsetDataKind` is retired with the relationship-block rebase (item K, STATUS §2). `overrides: JoinTypeOverrides` and the companion `HopPosition(u16)` newtype are removed from `JoinsetDataKind` (`24 §2.2`). Divergent join semantics inside a Joinset are now declared via **scope-local `Relationship`** entries in the Joinset's own `relationships:` block (`18 §2.10`, `16 §13.3`, `24 §5.3.2`).

**Decision rationale.**

- **Semantic-first alignment.** The new authoring shape (`cardinality` + `integrity` + `optional` + `cross_filter`) is the contract; `JoinType` is derived. A per-hop override that mutated only the operational `JoinType` cut across the new semantic layer awkwardly — authors would have ended up re-asserting `optional` indirectly through `JoinType` mid-Joinset.
- **Single mechanism.** A scope-local `Relationship` is a full Relationship: it carries `cardinality`, `integrity`, `optional`, `cross_filter`, `keys`, and `filter` together. Authors who need to vary join semantics inside a Joinset already need to think about the full shape, not just the mechanical join kind.
- **Surface symmetry.** `JoinsetBody.relationships` parallels `SemanticModel.relationships` exactly (per `18 §2.10`); there is no parallel override surface to learn, no per-position keying, and no override-legality matrix.

**Retired surface.**

- `JoinTypeOverrides` struct (`24 §2.2` legacy).
- `HopPosition(u16)` newtype (`24 §2.2` legacy).
- `JoinsetDataKind.overrides: JoinTypeOverrides` field (`24 §2.2` legacy).
- `EFFECTIVE_JOIN_TYPE(hop, overrides, manifest)` pseudocode signature replaced by `EFFECTIVE_JOIN_TYPE(hop, manifest)` (`24 §5.3.1`).

**Retired error codes (numbers reserved for forward-compat).**

- `VALID_E_2407 JoinsetOverridesForNonexistentHop` (`24 §9`).
- `COMP_E_2409 JoinsetReverseForwardOnlyRelationship` (induced by `Q-COMP-007` retirement; `24 §10`).
- `COMP_E_2411 JoinsetIllegalJoinTypeOverride` (`24 §10`).
- `COMP_E_2412 JoinsetAsOfDowngradeForbidden` (`24 §10`).
- `COMP_E_2413 JoinsetAsOfShapeMismatch` (`24 §10`).
- `PLAN_W_2401 JoinsetJoinTypeOverrideAdvisory` (`24 §11`).

**Refs.**

- `24 §2.2` — `JoinsetDataKind` post-retirement shape (carries `relationships: Vec<Relationship>` instead of `overrides`).
- `24 §5.3` — collapsed to `§5.3.1` (default selection — derivation from `optional`) and `§5.3.2` (scope-local Relationship as the v1 mechanism for divergent semantics).
- `16 §13.3` — composition-side ratification of the scope-local shadow mechanism.
- `18 §2.10` — Relationship struct's site-and-scope rule.
- STATUS reconciliation item K (2026-05-12).

**Forward-note touch points.** `Q-24-03` (Cardinality override) and `Q-24-07` (per-hop filter pushdown) updated with cross-references to this closure — the original "no per-hop X" answers carry through to the scope-local model, which provides the equivalent expressive power at the Relationship layer.
