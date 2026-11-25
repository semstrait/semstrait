---
doc: design/questions/deferred/24_questions
status: Deferred
purpose: Joinset questions parked for post-v1 ratification
---

# Deferred Questions — `data-kinds/24_joinset.md`

> Items deferred to v2 (or later) ratification. Closed items in [`../closed/24_questions.md`](../closed/24_questions.md). All v1 Joinset semantics live in `data-kinds/24_joinset.md` directly; this file holds questions parked for future versions.

---

## Q-24-01 — N-ary Joinset lift (`TD-NESTING-NARY-JOIN` / `TD-JOINSET-NARY`)

**Question.** `12 §5.2` ratifies binary-only Joinsets in v1; `24 §2.5` restates the constraint. The canonical struct in `24 §2.2` is N-ary-ready (`members: Vec<DataKindRef>`, `ExplicitPath { hops: Vec<JoinHop> }`). When does N-ary Joinset graduate from tech-debt to a MINOR release, and what is the exact authoring / compile contract that lifts?

**Refs.**

- `12 §5.2` — v1 binary arity, `TD-NESTING-NARY-JOIN`.
- `24 §2.2` — canonical struct sketch (N-ary-ready shape).
- `24 §4.2.2`, `24 §5.4` — explicit-path and fanout rules that are binary-v1-degenerate but N-ary-forward-looking.
- `24 §10.1` — `COMP_E_2403`, `COMP_E_2407`, `COMP_E_2410` are v1-unreachable; N-ary-forward-looking.
- `16 §13.3` — per-traversal `JoinType` overrides in explicit `Joinset` (already contemplates N-hop surface).

**Proposed (Round 1):** Deferred. The v1 Joinset is binary; N-ary is the recognized next step with the canonical struct already lifted to support it.

**Arguments for early lift (v1.1 MINOR).**

- The struct shape is already N-ary-ready; lifting requires only loosening `12 §5.3`'s arity check and extending the implicit-path BFS to handle N target members (Steiner-tree-style, as `16 §11.4` already sketches).
- Star-schema authoring is awkward when a single logical surface (`orders with customers, products, dates`) requires three separate binary Joinsets. N-ary removes the awkwardness.
- `19 §3.4.5`'s `PathSignature: BTreeSet<RelationshipPath>` already assumes multi-path traversal at the expression layer.

**Arguments for further deferral.**

- Multi-hop `Cardinality` accumulation (`24 §5.4`'s `profile.compose(walked)`) is non-trivial — `25` / `17` collaborate on the exact composition matrix, and neither is ratified.
- Multi-hop implicit-path BFS needs a Steiner-tree-esque solver; `16 §11.4`'s BFS is an approximation that has never been exercised on >2 target kinds.
- Override positioning becomes subtle: do overrides key on hop position, on endpoint-pair, or on the underlying `RelationshipId`? Round-1 chose `HopPosition`, which is unambiguous for binary and order-brittle for N-ary.

**Current position in `24`.** Binary-only in v1 per `12 §5.2`. Struct is N-ary-ready; MINOR lift requires only arity-check relaxation plus path-algorithm extension.

**Next step.** Revisit at `25` ratification; `25` specifies cross-kind composition rules including Joinset × Grainset, which is the first place multi-hop Cardinality accumulation becomes concrete.

---

## Post-v1 shape-hint clusters (folded in 2026-04-17)

> The following two clusters were previously parked in a standalone sidecar `joinset_shape_semantics.md`. That file was folded into this one on 2026-04-17 as part of the documentation-consolidation pass (H6). Neither cluster is blocking for v1; both describe *authoring-shape hints* the v1 Joinset body deliberately omits.

### Q-24-09 — `JoinAssociativity` hint (deferred) — `TD-JOINSET-ASSOCIATIVITY-PARK`

**Background.** Earlier drafts of the Joinset shape carried an `associativity:` field on the body — values such as `left`, `right`, `star`, `snowflake` — as a hint to the planner about how to order joins when the declared `relationships:` graph had multiple valid topological walks.

**v1 decision (ratified 2026-04-17).** Dropped. `JoinAssociativity` does NOT exist in the v1 Joinset body. `32 §14` records this as `TD-JOINSET-ASSOCIATIVITY-PARK`.

**v1 behavior.** The Joinset walks its `relationships:` in **name order** (`32 §7` / `18 §2` — relationships are `EntityId`-keyed and projected name-ordered post id-first rework, STATUS item U; the earlier "YAML author order" wording is superseded). When there is a choice, the planner picks the canonical left-deep traversal starting from the first member (by name) in the first `Relationship` entry. No cross-cutting hint shapes the walk. For the explicit-binary-join case that is v1 Joinset's only ratified arity, shape choices are degenerate anyway — there is exactly one join to emit.

**Deferred sub-questions.**

| Sub-ID | Question |
|---|---|
| Q-24-09.a | When v1's binary-only restriction lifts (`TD-NESTING-NARY-JOIN` / Q-24-01), does `associativity:` re-enter as a field, or does an ordering convention on `relationships:` suffice? |
| Q-24-09.b | If re-introduced, what values are legal? Candidates: `left` (left-deep), `right` (right-deep), `bushy` (planner chooses), `hinted` (planner uses author order). |
| Q-24-09.c | Is associativity advisory (planner may override for cost) or binding (planner must respect)? |
| Q-24-09.d | Interaction with `AsOf` joins (`17 §*`, Q-24-04 deferred): does AsOf carry its own shape-hint axis, or is it orthogonal? |

**Reference implementations.**

- **Cube.js** — no shape hint; join graph discovered at query time from declared relationships.
- **dbt_metricflow** — join-path resolution hints via `join_path:` on Metric definitions; implicit associativity.
- **LookML** — explicit `join` block per pair; implicit left-deep walk from the base `view:`.

v1 aligns with the Cube.js / LookML pattern (implicit from author-order). Re-introducing an explicit hint is a future decision when N-ary Joinsets (Q-24-01) land.

**Current position in `24`.** No `associativity:` field in the v1 Joinset body; `TD-JOINSET-ASSOCIATIVITY-PARK` carries the deferral. Revisit bundled with Q-24-01 (`TD-NESTING-NARY-JOIN`).

**Next step.** Bundle the decision with the N-ary lift: a shape-dedicated session, once N-ary support is on deck, re-opens both this cluster and Q-24-10 jointly.

---

### Q-24-10 — Star / Snowflake / 3NF shape-tag vocabulary (deferred) — `TD-JOINSET-SHAPE-VOCAB`

**Background.** During Q&A on the joinset YAML, a candidate field was considered that would let authors mark a Joinset as a specific canonical analytical shape:

```yaml
joinsets:
  - name: fact_sales_star
    shape: star              # or: snowflake, 3nf, data_vault, galaxy
    relationships: [ ... ]
```

The intent was to let the planner apply shape-specific optimizations (fact-table detection, conformed-dimension lookup, etc.) without deriving the shape from the `relationships:` graph.

**v1 decision.** Deferred. Not in the v1 body.

**v1 behavior.** Shape is derived at plan time from the `relationships:` graph + per-member `Cardinality`, when a shape-optimizing pass needs it. The planner does not require a declared shape and emits the same `PlanNode::Join` tree regardless.

**Deferred sub-questions.**

| Sub-ID | Question |
|---|---|
| Q-24-10.a | Is the `shape:` tag advisory (for diagnostics / tooling) or operationally meaningful (drives a planner pass)? |
| Q-24-10.b | What shape vocabulary is canonical? Candidates: `star`, `snowflake`, `galaxy`, `3nf`, `data_vault`, `one_big_table`, `junk_dimension_hub`. Open-ended strings vs a closed enum. |
| Q-24-10.c | If authored, is it validated against the declared relationships (e.g. `star` requires exactly one `many_to_one`-spine from one member to all others)? |
| Q-24-10.d | Does shape interact with `constraints:` at the joinset level (e.g. a star schema implicitly allows certain dimension rollups)? |
| Q-24-10.e | Does shape interact with the applicability matrix (`25`) — e.g. specific shapes enable specific aggregation patterns that flat graphs do not? |

**Shape vocabulary — reference implementations.**

- **Star schema** — one fact, many conforming dimensions, single-depth joins from fact.
- **Snowflake** — like star, with dimensions normalized (multi-level dimension hierarchies).
- **Galaxy / fact constellation** — multiple fact tables sharing conforming dimensions.
- **3NF / normalized** — arbitrary graph over normalized entities.
- **Data Vault** — hubs / satellites / links canonical shape.
- **One Big Table (OBT)** — denormalized, single wide member with no joins.

For v1, the Joinset represents explicit joins over ≥ 2 members with a declared graph. Shape classification can be layered on post-v1 without breaking the v1 body.

**Current position in `24`.** No `shape:` field; `TD-JOINSET-SHAPE-VOCAB` carries the deferral. Coordination with Q-24-09.

**Next step.** Re-open jointly with Q-24-09 when the N-ary Joinset arity lifts (Q-24-01).
