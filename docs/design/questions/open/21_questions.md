---
doc: design/questions/open/21_questions
status: Living
purpose: Parked unresolved questions surfaced while drafting `data-kinds/21_dataset.md`
depends-on:
  - data-kinds/21_dataset.md
  - data-kinds/20_taxonomy.md
  - foundations/11_names_and_scopes.md
  - foundations/12_nesting_policy.md
  - foundations/13_types_and_grain.md
  - foundations/14_expressions.md
  - foundations/19_expression_flow.md
  - foundations/15_mapping_and_binding.md
  - foundations/17_temporal_shape.md
  - apis/30_api_contracts.md
  - apis/34_semstrait_planner.md
---

# Open Questions — `data-kinds/21_dataset.md`

> Items surfaced during Round-1 drafting of the `SimpleDataKind` (Dataset) spec. Each entry restates the question, summarizes options, records `21`'s Round-1 working default, and flags the subsequent doc that must resolve before `21` is final. Post-thirteenth-pass re-evaluation (2026-04-30) confirmed Round-1 defaults remain consistent with the sealed trait hierarchy / `LeafExtras` architecture; Q-DS-001 has been ratified at Option A and migrated to [`../closed/21_questions.md`](../closed/21_questions.md). Q-DS-002..005 remain deferred to their named subsequent docs.

---

## Q-DS-002 — Wrapper code discipline for re-surfaced errors

**Question.** Should `21` wrap foundations-origin compile errors (`14`/`14a`/`19`/`15`) under `COMP_E_21xx` codes, pass them through unchanged, or wrap selectively?
**Why.** Trades Simple-level grepability against code-catalog size and origin-range preservation (`21 §8`, `15 §11.3`, `19 §8.1`, `30 §6.2`).
**Default.** Selective: `21 §8` wraps four cases (`2101`/`2102`/`2106`/`2107`) where context materially aids; others pass through with origin codes.
**Reopen.** At `30 §6.2` / `34` drafting — promote to universal wrapping if log aggregation prefers one-kind-one-code-range; shrink §8 to just `2101` if pure pass-through is preferred. Not blocking `21`.

---

## Q-DS-003 — Multi-source per-branch metadata emission at L2

**Question.** On a glob-fanned Simple, should `SimpleStrategy` emit every metadata-Dimension literal in each Union branch's L2 unconditionally, or prune those no downstream layer reads?
**Why.** Trades layer-locality (uniform L2 + general optimizer pruning) against smaller pre-optimizer plans (`_drafts/34_simple_strategy.md §3`, `21 §10.4`, `34 §5`).
**Default.** Unconditional at `SimpleStrategy`; `34`'s optimizer elides unused entries (§10.6 reading-key notes this).
**Reopen.** When `34 §5` ratifies the optimizer's Project-pruning rule — keep unconditional if pruning is a general pass, else prune at `SimpleStrategy`. Not blocking `21`.

---

## Q-DS-004 — Temporal-shape identifier on computed Dimensions

**Question.** Can a `temporal_shape:` identifier (`event_time` etc.) name a Computed Dimension (`ColumnMappingValue::Computed`), or only one mapping to a physical column?
**Why.** Permissiveness (cast-text-to-timestamp idiom) vs. as-of-join index-friendliness, a planner-performance not correctness concern (`21 §5`, `§7` `VALID_E_2108`/`2109`, `17`).
**Default.** Any Dimension with a temporal `data_type:` qualifies, regardless of mapping.
**Reopen.** At `17` drafting — emit a `TemporalIdentifierIsComputed` advisory if the as-of planner models Computed identifiers as a hard case. Affects only `17`'s checks, not `SimpleStrategy`. Not blocking `21`.

---

## Q-DS-005 — Re-aggregation-skip predicate over Computed Dimensions

**Question.** Should the re-aggregation-skip predicate's source-distinguishing test extend beyond metadata Dimensions (per-source literals) to Computed Dimensions whose expressions are demonstrably source-distinguishing?
**Why.** v1 metadata-only is simple and provably correct; extending handles the "wrap a metadata Dimension in a `Case`" idiom but needs an expression analyzer (`_drafts/34_simple_strategy.md §5.1`, `19 §3.4`, `15 §8`).
**Default.** v1 predicate inspects only metadata Dimensions; `LossyMultiSourceReaggregation` (`PLAN_W_2101`) fires more often than needed but correctness holds.
**Reopen.** At `34 §<implicit-union>` drafting or on explicit author request — extend via expression inspection or per-source evaluation. Not blocking `21`.

---

## Index

| # | Title | Owned-by | Blocking? | Next step |
|---|---|---|---|---|
| Q-DS-002 | Wrapper code discipline for re-surfaced errors | `21 §8` | No | Confirm at `30 §6.2` / `34` drafting |
| Q-DS-003 | Multi-source metadata emission at L2 | `_drafts/34_simple_strategy.md §3` | No | Confirm at `34 §5` |
| Q-DS-004 | Temporal-shape identifier on Computed Dimension | `21 §5`, `§7` | No | Confirm at `17` |
| Q-DS-005 | Re-aggregation skip over Computed Dimensions | `_drafts/34_simple_strategy.md §5.1` | No | Confirm at `34 §<implicit-union>` |

None of the remaining `21` open items block `21`'s ratification; every default is internally consistent and every follow-up is scoped to a later doc's drafting. Q-DS-001 was migrated to [`../closed/21_questions.md`](../closed/21_questions.md) on 2026-04-30 (post-thirteenth-pass cascade).
