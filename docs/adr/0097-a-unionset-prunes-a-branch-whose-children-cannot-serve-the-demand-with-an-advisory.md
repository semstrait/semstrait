---
number: 97
title: A Unionset prunes a branch whose children cannot serve the demand, with an advisory
status: accepted
date: 2026-11-03
source:
- CONTEXT.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md
- docs/adr/0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md
links:
- target: 73
  kind: amends
- target: 84
  kind: amends
- target: 74
  kind: amends
---

# A Unionset prunes a branch whose children cannot serve the demand, with an advisory

## Context and Problem Statement

ADR 0084 rule 1 fails a Grainset with `GrainNotServable` at its frame path where no child serves the whole demand and its parent takes rows or partial aggregates from it, or rule 4 leaves it no drill-across. Under a Unionset, that failure ends the whole request, although another branch can answer it. ADR 0073 prunes a Unionset branch only where a predicate placed on its rows folds to `false` or to NULL.

A production shape is a Unionset with one branch per upstream platform. One platform's branch is a Grainset of several reports at one granularity. Another platform's branch maps only some of the names. A request then fails if no child of the Grainset serves all its names, although the other branch can answer it.

When does a Unionset prune such a branch, and what does the caller see?

## Considered Options

- Prune a branch on any failure. In the nested sibling semantic model of the research folder (not the semantic model of the example below) it gives (2, 50, 700, 9000) and loses campaign 1 (run file `runs-round3.md`, section 5, in the research folder).
- Keep today's refusal.

## Decision Outcome

Chosen: a Unionset prunes a branch whose children cannot serve the demand by what they map, with an advisory.

### The rule

- A Grainset branch whose children cannot serve the demand by what they map is pruned. Two conditions both hold: no one child maps all the demanded names, and no join of children on a shared guaranteed keyset (ADR 0084, rule 4) would serve them.
- The test reads only the children's coverage masks and guaranteed keysets. It does not read the delivery.
- Where every branch is pruned and the test of this rule prunes at least one of them, the request is refused. This is not the empty read of ADR 0073. An empty read means "no rows". Here the answer is "cannot answer". Where the fold of ADR 0073 alone excludes every branch, the empty read of ADR 0073 stands.

### The advisory

- The advisory `BranchNotServable` names the branch and the names that it lacks.
- A branch that the fold of ADR 0073 prunes raises no advisory. That branch loses no rows.

### Partial aggregates

A branch that passes the test but cannot drill across, because its parent takes partial aggregates from it, stays refused with `GrainNotServable`. That case is not ruled here.

### Example

`facebook` is a Grainset of four reports at Day. `google` maps `date`, `campaign` and `clicks`. The Semantic model, as its author writes it:

```yaml
semantic_model:
  name: unionset_branch_not_servable
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
    - {name: campaign, data_type: string, type: categorical}
    - {name: gender, data_type: string, type: categorical}
    - {name: age, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
  unionsets:
    - name: ads
      mode: all
      dimensions: [{ref: date}, {ref: campaign}, {ref: gender}, {ref: age}]
      measures: [{ref: clicks}]
      grainsets:
        - name: facebook
          datasets:
            - name: totals
              extras:
                storage: {tables: [warehouse.facebook_totals]}
                semantic_mapping: {date: date, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: by_campaign
              extras:
                storage: {tables: [warehouse.facebook_by_campaign]}
                semantic_mapping: {date: date, campaign: campaign, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: by_gender
              extras:
                storage: {tables: [warehouse.facebook_by_gender]}
                semantic_mapping: {date: date, gender: gender, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: by_age
              extras:
                storage: {tables: [warehouse.facebook_by_age]}
                semantic_mapping: {date: date, age: age, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
      datasets:
        - name: google
          extras:
            storage: {tables: [warehouse.google_by_campaign]}
            semantic_mapping: {date: date, campaign: campaign, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
```

The rows of `warehouse.google_by_campaign`:

| date | campaign | clicks |
|---|---|---|
| 2024-01-01 | spring | 5 |
| 2024-01-01 | search | 6 |

The request `gender, age, clicks`:

| plan | output |
|---|---|
| Today (`main`) | the Semantic model is refused at compile with `GrainsetGrainRepeated` |
| With ADR 0096's changes only | the request fails with `GrainNotServable` |
| This ADR | (NULL, NULL, 11) plus the advisory `BranchNotServable` that names `facebook`, `gender` and `age` |

No `facebook` child maps both `gender` and `age`, so `facebook` is pruned and `google` answers with its 5 + 6 clicks. The oracle model `unionset_branch_not_servable` holds this model plus a `spend` Measure, and its case `branch_not_servable_is_pruned` asserts (NULL, NULL, 11). The `facebook` rows there hold the Grainset's claim. The answer does not change, because `facebook` is pruned.

### Glossary

*Pruned* adds to its list "a Unionset branch whose children cannot serve the demand", and its sentence on what the frame may read gains the children's coverage masks and guaranteed keysets.

### Amends

- **0073:** in *A Unionset prunes a branch its predicate excludes*, a Grainset branch is also pruned where its children cannot serve the demand by what they map (0097's *The rule*); "No advisory. A pruned branch loses no rows" holds for a branch that the fold prunes, and a branch pruned for this reason raises `BranchNotServable` (0097's *The advisory*); the empty read of "Every branch excluded" is not the answer where every branch is pruned and the test of 0097's *The rule* prunes at least one of them: the request is refused, and where the fold alone excludes every branch the empty read stands (0097's *The rule*); in *One decision shape, checked per body*, the Unionset row's cell "Operand not read", "a branch the predicate excludes", also holds a Grainset branch whose children cannot serve the demand by what they map (0097's *The rule*); the rest of 0073 stands.
- **0074:** in *A Unionset*, the Grainset-branch clause "where the union takes rows or partial aggregates from the branch, as it does from every branch that is not disjoint, and no single child maps every name that the branch serves, the plan fails with `GrainNotServable` at the branch's frame path", and in *A Grainset* "A Joinset that builds its join takes rows or partial aggregates from its operands, and so does a Unionset from a branch that is not disjoint. … where no single child serves the names in its input, the plan fails with `GrainNotServable` at the Grainset's frame path", do not apply to a Grainset branch whose children cannot serve the demand by what they map. That branch is pruned (0097's *The rule*). A branch that passes the test stays refused (0097's *Partial aggregates*). The rest of 0074 stands.
- **0084:** in *Which children answer*, rule 1, "Where its parent takes rows or partial aggregates from it, or rule 4 leaves it no drill-across, and no child serves the whole demand, the plan fails with `GrainNotServable` at its frame path" does not apply to a Grainset branch of a Unionset whose children cannot serve by what they map: that branch is pruned (0097's *The rule*); a branch that passes the test but fails only because its parent takes partial aggregates from it stays refused with `GrainNotServable` (0097's *Partial aggregates*); the rest of 0084 stands.

### Consequences

- Good: a Unionset answers from the branches that can serve, and one weak branch does not fail the request.
- Good: the caller sees which branch was pruned and which names it lacks.
- Good: "cannot answer" is a refusal, never read as "no rows".
- Neutral: the code changes with the decisions above: the prune in the Unionset Strategy, the advisory `BranchNotServable` and the refusal where every branch is pruned and the test prunes at least one of them.
- Neutral: the glossary entry *Pruned* changes with this decision.
- Bad, a trade-off: the rows of a pruned branch are absent from the answer, and the caller learns it from the advisory; gain: the request does not fail while another branch can answer it.

## More Information

Decided on Change 2.
