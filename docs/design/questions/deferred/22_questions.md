---
doc: design/questions/deferred/22_questions
status: Deferred
purpose: Grainset questions parked for post-v1 ratification
---

# Deferred Questions — `data-kinds/22_grainset.md`

> Items deferred to v2 (or later) ratification. Live items are in [`../open/22_questions.md`](../open/22_questions.md); closed items in [`../closed/22_questions.md`](../closed/22_questions.md).

---

## Q-GRN-003 — Cost function pluggability hook site: planner trait or adapter hook?

**Question.** `22 §4.4` ratifies the v1 cost function as source-count. A future stats-backed cost function is `[TD-GRAINSET-COST-STATS]`. When it lands, should the hook site be:

- **A** — a method on `34`'s `Planner` trait (`fn grainset_cost(&self, child: &ResolvedGrainsetChild, request: &Request) -> Cost`);
- **B** — a method on the `37` catalog-adapter trait (each adapter reports stats; the planner consumes them uniformly);
- **C** — a separate `CostEstimator` trait injected into the `plan` call site (third-axis of extensibility)?

**Refs.**

- `22 §4.4` — Round-1: source-count proxy.
- `34` (pending) — planner entry-point; owns the SemanticManifest-to-Plan strategy dispatch.
- `37` — catalog adapter; owns source metadata (file sizes, partition counts, row-count estimates).
- `30 §6.2` — the `22xx` code range is fixed regardless of cost-function placement.

**Arguments for A (on `Planner`).**

- Cost is a planner concern — it composes with the rest of the plan strategy (join-ordering, push-down) and is not uniquely Grainset's.
- Keeps adapter surface narrow; adapters supply raw stats, planner derives cost.

**Arguments for B (on adapter).**

- The numbers live in the catalog; asking each adapter to report cost directly avoids a round-trip through stat-fetching + planner-internal computation.
- Simpler for adapter authors who already know their own numeric characteristics.

**Arguments for C (separate `CostEstimator`).**

- Decouples cost from both planner and adapter; lets users inject a custom estimator without reimplementing either.
- Matches Calcite / DataFusion patterns.

**Current position in `22`.** Deferred. The hook site is a `34`-drafting decision; `22` does not commit.

**Next step.** Confirm at `34` drafting time; `22 §4.4` will cross-reference whichever trait lands.
