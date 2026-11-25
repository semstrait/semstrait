---
number: 64
title: Field-first resolution ranks public candidates by compositions, then granularity fit, then exposed semantics; a tie goes to name order with an advisory
status: accepted
date: 2026-09-10
source:
- CONTEXT.md
- docs/adr/0008-unsealed-extension-traits.md
- docs/adr/0011-thin-manifest-in-memory-graph.md
- docs/adr/0015-refuse-ambiguity-name-order-selection.md
- docs/adr/0018-canonical-plan-ir-contract.md
- docs/adr/0028-relationship-edge-model.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0059-the-coverage-axis-is-three-way-and-read-against-a-composition.md
- docs/adr/0062-a-request-states-intent-never-placement.md
- docs/adr/0063-the-planner-takes-a-request-context-not-a-session.md
- inherits the routing filter recorded
- sharpened in place (granularity fit)
- sharpened in place (the plan result, the trace of the choice)
links:
- target: 15
  kind: amends
- target: 28
  kind: amends
- target: 53
  kind: amends
- target: 54
  kind: amends
- target: 73
  kind: amendedby
- target: 74
  kind: amendedby
- target: 84
  kind: amendedby
- target: 5
  kind: amends
- target: 96
  kind: amendedby
---

# Field-first resolution ranks public candidates by compositions, then granularity fit, then exposed semantics; a tie goes to name order with an advisory

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): every Strategy call takes one input shape, built from the `Resolution` for the root and from the parent's decision for an operand; the driver checks the per-body table of stand-ins; "on a tie the scope root comes first" does not apply to a Grainset.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): a stand-in sub-composition must satisfy the rule for leaving out an operand: every demanded name read from what remains, no read row dropped, no relationship filter; among its grains that carry a granularity, a Grainset picks the one that serves the whole demand and has the coarsest granularity equal to or finer than the requested one; with no requested granularity on the time axis, it picks the grain with the coarsest granularity that serves the whole demand if the request does not select the time axis, and such a grain with the finest granularity, whose values are the least bucketed, if the request selects the time axis without a granularity; "requested grain" is read as the requested granularity.
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Decision Outcome*, "a composite's mask is the union of its operands' (ADR 0047), so the root serves the demanded set exactly when some unit in its scope does" holds for a Grainset only where one child serves the demand: a Grainset also serves a demand that it drills across two or more children for, and the mask of a Grainset is the union of its children's and the bit of each Metric that it computes (0084's rule 8); in *Decision Outcome*, "the child that its drill-across joins from (ADR 0084, rule 6)" is the first child of rule 6's left-deep joins; the rest of 0064 stands (0084's *Which children answer*).
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *Decision Outcome*, under **Granularity fit**, the premise "A public Grainset holds no granularity of its own" reads: a Grainset that declares no granularity carries the granularity that its children agree on, and none where they differ (0096's *Shared granularity*), and it still ranks at the granularity of what its routing answers with; "The Grainset's Strategy picks its child by granularity fit alone; it never routes to a child with no granularity (ADR 0074)" and "a Grainset's children are ordered by granularity fit alone (ADR 0074, *A Grainset*, as ADR 0084 amends it)" read: a Grainset picks its child by the child rank, and a child with no time axis is among the children that serve a demand that reads no time axis (0096's *The pick* and *No time axis*); the resolver's rank for public candidates does not change; the rest of 0064 stands.

## Context and Problem Statement

How is the scope root found, with and without `from`, which unit in its scope
answers, and what does a tie do?

## Considered Options

- Field-first resolution ranks public candidates by compositions, then
  granularity fit, then exposed semantics; a tie goes to name order with an
  advisory (chosen). Granularity fit ranks between the two terms of structural
  complexity.
- **An implicit Grainset over the whole model**: a root frame above the
  scope root runs the Grainset's Strategy over every public source, with no
  drill-across. Refused: that frame has no semantic source, that Strategy has
  no `RoleDefinition` variant (ADR 0073: "exactly one exists per variant"),
  every frame path gains a level, and a request with `from` plans through a
  root frame that decides nothing. Of the Grainset's routing, the mask test and
  the grain test already run at the root (ADR 0074, *Every request is planned
  against its grain first*). Only the order in which a Grainset picks its child
  did not, so the resolver reads it as granularity fit, one term of its order,
  and the Grainset's Strategy reads the same order.
- **No granularity term**. Refused: two Datasets at Day and Month both
  count 0 compositions, so the finer one wins on fewer exposed semantics or on
  name order, which is the defect that *Amends* says this order removes. The
  order needs no Grainset: the mask test already keeps only sources that serve
  the demanded set, and one name is one semantics. Only joining children needs
  what a Grainset declares, that they hold the same subject, and the resolver
  never joins.
- **Granularity fit after exposed semantics, or before compositions**.
  After it, granularity fit only breaks a tie, and a monthly Dataset that
  exposes one more semantics than a daily one loses at Month, which is the
  defect that *Amends* names. Before compositions, a
  Joinset at Month ranks above a Dataset at Day that serves the request alone,
  which reverses this ADR's order.
- **Reading each candidate on the requested temporal Dimension** rather than on
  its own time axis. Refused: `select order_date, revenue` would go to a
  Dataset with no time axis, whose values count as the finest, and not to the
  Dataset whose time axis is `order_date`.
- **A Grainset at the granularity of its best child by the grain test alone**.
  Refused: it ignores which names each child maps, so a Grainset can
  rank at a child that does not answer for it: at Month where only its Day
  child serves the demand, or as a candidate whose routing has no answer.
- **The resolver calls the Grainset Strategy's `decide`** on an input it builds
  itself, to learn a public Grainset's answer. Refused: a Strategy
  decides for one frame and its demand (`CONTEXT.md` **Strategy**), and no
  frame exists before the resolver has fixed the root.
- **A unit picker**: the resolver returns the one unit a Strategy runs on,
  having chosen the sub-composition or grain itself. Refused: "which unit in
  scope wins" is kind-dependent (the Unionset case), so a kind-agnostic resolver
  would have to re-encode each kind's semantics.
- **Refusing every tie** (the round-one recommendation). Refused by the human: a model that can answer should
  answer, and the caller holds the disposing intent (`from`).
- **Name order alone** (ADR 0015 as written). Refused: it knowingly picks a
  fanned-out join over a plain scan, or a finer unit over the direct grain,
  alphabetically.
- **Fields before compositions** — MetricFlow's precedence (`_sort_by_suitability`
  by exposed spec count, then `min` by join count; `dataflow_plan_builder.py`,
  read from source). Refused: its rationale is *aggregation* cost, and a join
  carries a correctness risk — fan-out — that a wide scan does not, so structure
  ranks first here. Cube takes the first pre-aggregation in declaration order;
  Looker takes the smallest covering aggregate. **None of the systems surveyed
  returns a tie signal to the caller** — MetricFlow logs at debug level and Cube
  says nothing — which is the posture this ADR refuses. (Precedent read directly
  from source.)
- **Constituent-set inclusion as the order** (the round-one recommendation). Subsumed: fewer compositions is
  the same test on the operand closure, and the field count breaks ties
  inclusion leaves incomparable.
- **A weighted or statistical cost**. Refused on ADR 0015's still-valid ground
  and because the canonical layer reads no engine or catalog statistics.
- **Recording the pick on `SemanticPlan` itself**, or in a side channel.
  Refused: the plan is canonical IR (ADR 0018); tracing is invisible to a caller.
  MetricFlow's `SourceNodeRecipe` (`source_node`, locally served specs, ordered
  join recipes) is the nearest precedent for the `Resolution` carrier, and it
  commits to one source with no alternatives list — the alternatives here exist
  because the Strategy, not the resolver, owns the bypass.
- **An unsealed public `Complexity` trait** on ADR 0008's roster. Refused as
  below; additive later if a caller ever needs to supply one.

## Decision Outcome

The planner answers a request through exactly one **scope root** — a public
semantic source — and the search for it runs over the **public** sources only.
`from` names the root directly: the named source must itself serve the whole
**demanded set**, or the request fails with `UnknownName` naming what is missing
and where it was looked for; it is never widened to another source. This is
ADR 0062's *"the named source and the private units it originated"* stated as
one test rather than two: a composite's mask is the union of its operands'
(ADR 0047), so the root serves the demanded set exactly when some unit in its
scope does, and a root that fails the test has no private unit that would pass
it.

Without `from`, **field-first resolution** ranks every public **candidate** — a
source whose mask covers the demanded set and whose own grain or kind-specific
routing admits the request — by `(compositions, granularity
fit, exposed semantics)` ascending, and takes the least. *Compositions* and
*exposed semantics* are the two terms of **structural complexity**.
*Compositions* counts the composed — non-Dataset — semantic sources among the
candidate and everything reachable from it through operand edges: a Dataset
scores 0, a Joinset over two Datasets 1, ADR 0054's left-deep chain over *n*
constituents *n* − 1, a Unionset or Grainset 1 plus whatever its branches or
grains score. *Exposed semantics* is the population count of the mask. Two
candidates equal on all three are **path ambiguity**: the planner takes the
first in name order, plans it, and returns an **advisory** naming the chosen
source and the tied ones beside the plan. Nothing is refused.

**Granularity fit** is the order in which a Grainset picks its child (ADR 0074,
*A Grainset*, as ADR 0084 amends it), read on each candidate's own time axis as
ADR 0074's grain test reads it. A candidate falls in one of three groups,
ranked in this order:

1. the request asks a granularity on its time axis;
2. the request selects its time axis without a granularity;
3. the request does not select its time axis, or the candidate has none.

Inside groups 1 and 3 the coarsest granularity ranks first. Inside group 2 the
finest ranks first, because its values are the least bucketed. A candidate with no
granularity holds its time axis at its own values (ADR 0074, *Every request is
planned against its grain first*), so it ranks as finer than every
granularity. A public Grainset holds no granularity of its own: it ranks at the
granularity of what its routing answers with — the child that its pick chooses
for the demand, or the child that its drill-across joins from (ADR 0084, rule
6). Where its routing has no answer, it fails the candidate test (ADR 0074, *A
Grainset*, as ADR 0084 amends it). The Grainset's routing — ADR 0074's pick of
a child, and ADR 0084's choice of the children and of the child that the
drill-across joins from — is one rule over the Grainset and the demand, with
two callers: the resolver calls it and keeps only the granularity of its
answer, and the Grainset's Strategy calls it again in its frame and decides
from it. So the resolver encodes no Grainset rule a second time and runs no
Strategy outside a frame. The Grainset's Strategy picks its child by granularity fit alone;
it never routes to a child with no granularity (ADR 0074). Granularity fit
ranks before exposed semantics, so between units with equal compositions a unit
at the requested granularity beats a finer one whatever their field counts. Of
a public `spend_daily` at Day and a public `spend_monthly` at
Month, both `Timeseries` on `date`, `select date at Month, campaign, clicks`
and `select campaign, clicks` read `spend_monthly`, and `select date, campaign,
clicks` reads `spend_daily`.

Once the root is fixed, the **resolution scope** is the root plus the private
units it originated (ADR 0054: reachable only through their public origin).
Field-first infers `from` and nothing more, so the two paths converge on one
rule. Which unit inside the scope actually answers is the root's **Strategy**'s
decision — a **bypass** — and the Strategy contributes only *which units are
alternatives*: a Joinset's sub-compositions and a Grainset's grains are, a
Unionset's branches never are, since every branch contributes rows to every
answer and no branch can stand in for the whole — the smallest covering branch
is a wrong answer, not a cheaper one. Among a Joinset's sub-compositions the
same order applies, granularity fit included, and a Grainset's children are
ordered by granularity fit alone (ADR 0074, *A Grainset*, as ADR 0084 amends
it); on a tie the **scope root comes
first**, and name order with the
same advisory decides only among the remaining tied alternatives, so a bypass is
always into a unit of strictly lower complexity than the root. The **resolver**
is therefore kind-agnostic: it binds names to bits, fixes the root, orders the
in-scope candidates, and hands the Strategy a `Resolution { scope_root,
demanded, bindings, requested_granularity, alternatives }`. Signatures,
per-constituent coverage (ADR 0059's derivation) and join conditions are read by
the Strategy from the unit it picks, never pre-resolved for every candidate.

### Amends

- **0005:** in *Decision Outcome*, "`Err((Diagnostic<K>, Diagnostics<K>))` for
  fail-fast ones" holds for `plan` in two kinds: the warnings take their own
  kind, `Advisory`, beside the error's kind, `PlanError`, on both arms (0064's
  *ADR 0018 is untouched*); the rest of 0005 stands.
- **0015:** superseded in two clauses, both scoped to plan-time candidate
  selection, in 0064 rather than by editing its frontmatter (the ADR 0037
  precedent). Its selection clause ruled that where a choice must be made it is
  made by *"coverage plus name order, with no statistics and no cost model, so a
  rollup-requiring finer unit can beat a direct-grain unit alphabetically"*, on
  the ground that a cost function added later would silently change results for
  existing models. No planner and no golden plan exist yet, so the ground does
  not bind the selection clause, and the example it accepts is a defect that
  0064's order removes (0064's *Decision Outcome*). Name order survives as the
  **last** tie-break, no longer the first, and no longer silent. Structural
  complexity is request-free and statistics-free, and granularity fit reads the
  request but no statistics, so 0015's refusal of a *statistical* cost model
  stands untouched, and the ground still holds against a statistical cost; what
  changes is that structure and granularity count before names do. Its refusal
  clause made *"equal-cost composition ties"* a hard plan error; under 0064 a
  tie on every term among candidates is answered and advised, never refused, so
  that clause too is superseded **for plan-time ties**. 0015's compile-time
  refusals — an explicit composition whose form equals an enumerable implicit
  one, and multiple shortest relationship paths — are not touched by 0064;
  ADR 0054 already re-ruled the first (a duplicate is a redundancy, not a fault)
  and removed the second's subject (there is no plan-time path search).
- **0054:** the refusal posture is narrowed, not abandoned. ADR 0054 and the
  former **Path ambiguity** entry made a tie something *"reported for the author
  to resolve, never silently tie-broken"*. It is still never silent — the
  advisory is the report, and it goes to the caller, who resolves it now with
  `from` — but the request is answered rather than rejected. It was found that
  ambiguity is a property of the entry point, not the artifact, so compile
  cannot report it and a refusal would fail requests the model can in fact
  answer. ADR 0054's own case — two sub-compositions over the same constituents
  with different relationship sets — falls under the same rule inside a scope:
  name-least is planned, the advisory tells the author a scope-local shadow is
  wanted.
- **0028:** there is no plan-time path search: a request spanning two public
  sources with no authored composition fails; ADR 0028 forbids inferring the
  edge and ADR 0054 already enumerated every legal one, so there is nothing left
  to discover at plan time.
- **0053:** field-first resolution ranks every public candidate by
  `(compositions, granularity fit, exposed semantics)` ascending, and takes the
  least. Two candidates equal on all three are path ambiguity: the planner takes
  the first in name order, plans it, and returns an advisory naming the chosen
  source and the tied ones beside the plan. Nothing is refused.

**The routing filter recorded is adopted as stated**: candidate selection
is a mask test over the flat `semantic_sources` collection filtered by `public`,
and once a root is entered, by the `originators` chain. ADR 0047's *"a mask test
over a flat list"* is completed by this ADR rather than edited.

**ADR 0018 is untouched.** A plan produced by field-first is structurally
indistinguishable from the same plan produced by `from`. Which source was chosen,
with the scope root's rank (its compositions and its granularity fit), rides only
in TRACE-class `SemAnnotation`s on the plan nodes (`semstrait-ir::plan::meta`),
which the canonical layer *records* and never *reads* (`CONTEXT.md` **Canonical
layer**). The resolver computes both terms for the
scope root however it was found, under `from` too, so the plan with `from` and
the field-first plan carry the same trace. Trace holds no other reason for the
choice: no other candidate, and no count of exposed semantics, which would change
a plan's trace when the source gains a semantics that the request does not read.
Where name order breaks a tie, the advisory names the chosen source and the tied
ones.

The advisory is not on the plan at all. `plan()` returns `Result<(SemanticPlan,
Diagnostics<Advisory>), (Diagnostic<PlanError>, Diagnostics<Advisory>)>`: ADR
0005's fail-fast form, with the warnings on both arms in their own kind,
`Advisory`, beside the error's kind, `PlanError`. This is the first `Advisory`
variant with a decided producer, so the carrier is fixed here, and the fan-out
advisories ADR 0033 implies land in it later. The advisory system, which
collects and analyzes advisories, is a later effort. This ADR fixes only the
carrier and the advisories that the planner raises.

### Three things the resolver deliberately does not do

- **It never composes.** A source covering part of the demanded set is
  diagnostic material only: under field-first, `NoCandidate { nearest }` names
  the **public** sources that came closest — ordered by fewest missing
  semantics, then structural complexity, then name, so the list is deterministic
  — each with `demanded \ mask` rendered as names through the bit → name
  direction of the **bitmap index**. Private units never appear in it: they are
  reachable only through a root, and `NoCandidate` is raised only when no root
  was found (under `from`, the failure is `UnknownName` against the named root).
  A request spanning two public sources with no authored composition fails;
  ADR 0028 forbids inferring the edge and ADR 0054 already enumerated every
  legal one, so there is nothing left to discover at plan time.
- **It never derives a join basis.** The unit's `on_expr_id` is written at
  compile (`on: auto` resolves in `compile/relationships.rs`; its failure is
  compile's stage gate). The "plan-time remainder of the auto-join
  ladder" inherited here has no subject and is struck.
- **It never reads statistics or engine identity.** The complexity key is
  computable from the manifest alone; extension is by further *model-declared*
  metadata (a completeness tag was named), through a crate-internal
  `Complexity` trait producing an `Ord` key for each of its two terms, with `Structural` as the default
  implementation. The trait is **not** added to ADR 0008's unsealed roster: the
  extension foreseen is our code reading our manifest, and unsealing is a
  one-way promise that buys nothing today. Granularity fit is the one term
  that reads the request, and it reads no statistics either. So `Complexity`
  keeps the two request-free terms, and the resolver's key orders by the
  first, then by granularity fit, then by the second.

### The seam a cached graph replaces

The rulings above *are* the algorithm; what a later `SemanticGraph` (ADR 0011,
out of scope for MVP) replaces is the lookups. The resolver consumes a read-model
trait — `bit_of(name)`, `name_of(bit)`, `public_sources()`,
`originated_by(root)`, `operand_closure(source)`, `signature(source, bit)` —
implemented over a direct manifest walk today and over a cached graph later. A
swappable `Resolver` was refused because a second implementation could rank
differently and nothing would notice.

### Consequences

- `CONTEXT.md` gains **Resolver**, **Demanded set**, **Candidate**, **Scope root**,
  **Resolution scope**, **Bypass**, **Structural complexity**; **Field-first
  resolution**, **Path ambiguity** and **Advisory** are rewritten (an advisory is
  the warn-and-proceed posture, of which lossiness was the first instance, not
  the definition); **Bitmap index** gains the reverse direction found
  wanting. The later sharpening adds **Granularity fit**, and **Structural complexity** and
  **Path ambiguity** read it.
- The resolver's typed errors: `UnknownName { name, looked_in }` for every
  named thing not found (`from` not a public source; a semantics not in the
  registry; a semantics not served by `from`), `NoCandidate { nearest }`,
  `GrainNotServable { path, source, requested, source_grain, nearest }`. As
  every error does (ADR 0073), it names a path: its frame's, or the named scope
  root's where the resolver raises it under `from`. `requested` is absent where
  the request asks no granularity, and `source_grain` where the source holds
  none, as a public Grainset does. `nearest` lists each child of the Grainset
  and the names that it lacks. Ambiguity is an advisory, not an
  error. Role mismatch and mixed-layer predicates stay with
  request binding (ADR 0062).
- **Handoffs.** To the Strategy seam: the per-kind *alternatives* rule
  and a shared per-constituent-coverage helper. To aggregation placement:
  whether bypassing a fanning constituent is answer-preserving is
  *placement*, not selection. To expression lowering: `signature(source,
  bit)` is read after the unit is chosen. To the open Grainset questions: structural
  complexity replaces name order for driver / attached selection. To plan
  determinism: whether TRACE annotations participate in golden-plan equality.
  Nothing is owed to the manifest effort; no new carrier is needed.
- `crates/` is not changed by this decision. The old planner's
  The former entity resolver's single-source scoring is the idea this ADR
  formalises; its multi-entity greedy set-cover and the former ad hoc join are the
  composition this ADR forbids.

## More Information

Decided 2026-09-10.

The ranking in *Decision Outcome* was sharpened on 2026-10-06 on a later decision. It ranked by
`(compositions, exposed semantics)`, so of two Datasets at Day and Month the
finer one could win on its field count or on name order, which is the defect
that *Amends* says this order removes. It now ranks by `(compositions,
granularity fit, exposed semantics)`, and the title names that order; it was
"Field-first resolution ranks public candidates by structural complexity; a
tie goes to name order with an advisory". The file keeps its name, so the links
to it hold. It was sharpened in place rather than in
a later ADR, on the human's explicit call: "enchancement edit in place for adr
which suppose to cover and produced by this map", because "there was not
encoding for that before". A prototype compared
the shapes.

