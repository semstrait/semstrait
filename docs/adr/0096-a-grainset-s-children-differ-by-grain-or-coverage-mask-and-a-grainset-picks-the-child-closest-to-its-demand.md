---
number: 96
title: A Grainset's children differ by grain or coverage mask, and a Grainset picks the child closest to its demand
status: accepted
date: 2026-11-03
source:
- CONTEXT.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0056-composed-temporal-shape-is-derived-at-compile.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0080-only-a-frame-that-gives-final-aggregates-buckets-and-every-filter-reads-the-value-before-bucketing.md
- docs/adr/0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md
links:
- target: 30
  kind: amends
- target: 56
  kind: amends
- target: 74
  kind: amends
- target: 84
  kind: amends
- target: 80
  kind: amends
- target: 64
  kind: amends
- target: 65
  kind: amends
- target: 73
  kind: amends
---

# A Grainset's children differ by grain or coverage mask, and a Grainset picks the child closest to its demand

## Context and Problem Statement

`main` refuses two Grainset children at one granularity (`GrainsetGrainRepeated`). It also refuses a child that has no granularity. ADR 0056 requires "a distinct grain per constituent" and one shape kind across a Grainset's children, and it records the one-kind rule as provisional. ADR 0030 requires strict equivalence of shape across children. ADR 0074 picks, among a Grainset's grains that carry a granularity, the one that has the coarsest granularity equal to or finer than the requested one. Before this ADR, ADR 0084 rule 6 said nothing about the order of the joins at a tie on the granularity fit, and rule 1 said nothing about an activated named filter.

A production shape needs both refusals gone. That shape is a Unionset with one branch per upstream platform, each branch a Grainset: the reports of a Grainset share Day and differ by breakdown, and a total report has no date. An activated named filter on a Grainset also fails with `NoSignature` where one child answers alone, because the Grainset's frame reads its own signature for the filter, and a Grainset owns a signature only for a Metric that it computes, never for a named Filter (ADR 0084, rule 8).

Which children may a Grainset hold, which child does it pick, and what does the child that answers alone take?

## Considered Options

- Grain defined to include the mapped names. A request's grain maps no Measures, so such a Grain never equals a demand's Grain.
- `Left` joins from one child, with a plain `Eq` key. A group that only an attached child holds is not in the answer, and a NULL group keeps the values of one child only, so renaming a child changes that group's row. ADR 0084 *Considered Options* rejects this form.
- `Full` joins, with each grouping Dimension coalesced over the children and a NULL-safe key (chosen, ADR 0084 rule 6).
- The rank with fewest Dimensions first everywhere, with no exception. A run with it passed 861 of 862 tests in four crates, with no golden change. The one failure is the `semstrait-planner` test `a_grainset_reads_the_coarsest_child_that_serves_the_granularity_and_else_the_finest`. It moved a request that selects `event_date` without a granularity from Day rows to Month rows (run file `runs-round5.md`, section "Check 2", in the research folder).

## Decision Outcome

Chosen: children that differ by grain or by coverage mask, children with no time axis, a pick by the child rank with one exception, `Full` joins in the order of the child rank with the granularity fit first, so that the first child is at the requested granularity, or the closest to it, and an activated named filter that goes to the child that answers alone.

### Shared granularity

- A Grainset's children may share a granularity.
- A Grainset that declares no granularity carries the granularity that every child that carries one agrees on, as ADR 0056 derives one for any composition, and none where they differ; only a Grainset that is a constituent may declare one (ADR 0074).
- `semstrait-manifest` compile refuses two children that have the same grain and the same coverage mask, with `GrainsetGrainRepeated`.
- Grain stays as the glossary defines it: the Dimensions and the granularity, with no Measures.
- ADR 0056 corrected ADR 0030's "≥2 distinct grains" to distinctness between every two children. This decision narrows what counts as two equal children: the same grain and the same coverage mask.
- The minimum number of children lives in no ADR. It is in `semstrait-model` validation and in the glossary (*Complex DataKind*).

### No time axis

- A Grainset may have no time axis.
- Some children may have a time axis and others none.
- A child with no time axis serves only a demand that reads no time axis. The demand reads a time axis where it selects it, where a request predicate or an activated named filter reads it, or where it buckets it.
- In the granularity fit, a child with no time axis ranks as coarser than Year.
- `GrainsetChildMissingGranularity` refuses only a child that maps the time axis and declares no granularity.
- `ComposedConstituentMissingGranularity` refuses a composed constituent that publishes a temporal shape and carries no granularity, neither declared nor derived.
- `ComposedConstituentMissingTemporalShape` refuses a composed child of a Grainset where the child publishes no temporal shape and carries a time axis: a constituent below it publishes a temporal shape. An example is a Joinset that states no shape, over `Events(order_date)` at Day and `Events(ship_date)` at Day: the two shapes tie at one rank, so it publishes none (ADR 0056: equal rank with different bodies yields an absent shape). The glossary *Grainset* clause "and is refused if it has none" becomes the rules of `ComposedConstituentMissingGranularity` and `ComposedConstituentMissingTemporalShape`.
- ADR 0056 recorded the one-shape-kind rule as provisional: "The restriction is recorded as **provisional** — what the model needs now, not a claim that no Grainset could ever mix kinds — so a later relaxation is an extension rather than a reversal." A child with no time axis beside children with one is that relaxation.

### The pick

- A Grainset sends each demanded name to the child whose grain is closest to the demand, that is, the child that the child rank ranks first for that name alone (ADR 0084, rule 3).
- Where every name goes to one child, that child answers alone.
- The child rank is: fewest Dimensions, then granularity fit, then name order.
- One exception applies where the demand selects the time axis without a granularity. There the granularity fit comes first (finest first), then fewest Dimensions, then name order. The exception keeps today's pick for that request.
- The resolver's rank for public candidates does not change (glossary *Structural complexity*).
- Where a child's grain equals the demand's grain, the child rank picks that child.

### Name order

- The planner's read model already sorts every semantic source's operands by name before any Strategy reads them. This is a fact of today, not a change.
- The manifest keeps declaration order.
- "Name order" therefore means the child's name.

### Drill-across at one granularity

- The left-deep joins follow the child rank with the granularity fit first, then fewest Dimensions, then name order. This is the form that the pick already uses where the demand selects the time axis without a granularity.
- So the first child of the joins is the chosen child at the requested granularity, or, where no chosen child carries it, the chosen child closest to it.
- The joins are `Full`, as ADR 0084 rule 6 says.
- On the prototype, the tie falls to the order of the semantic definitions' ids. Each id is a hash, so the order of the joins follows no name that the author sees.
- The order of the joins raises no advisory. Every chosen child is read, and under the reading of ADR 0084 rule 6 the rows do not depend on that order, including the row of a NULL group.
- ADR 0084 rule 6 puts every group that any chosen child holds in the answer.

### Ambiguity

- A pick that name order decides yields the advisory `AmbiguousCandidates`. The advisory names the chosen child and the tied children. The planner raises the advisory at every frame.
- The tie check for a Grainset's children reads the Grainset's own rank, not the resolver's rank.
- The glossary entry *Path ambiguity* gives the reason: "It is never tie-broken silently". The entry's edit: a tie in a Grainset's child rank is reported the same way, inside the resolution scope.

### A named filter on a Grainset

- A Grainset owns a signature only for a Metric that it computes (ADR 0084, rule 8). Named Filters get no Grainset signature. Its frame passes an activated named filter unchanged to the child that answers alone. That child reads its own signature for the filter, as each chosen child of a drill-across already does.
- The child that answers alone is the child that every demanded name goes to, or, where the fallback of ADR 0084 rule 1 applies, a child that serves the whole demand. The demand holds every activated named filter (glossary *Demanded set*), so that child takes the filter.
- The pick runs among the children that map the filter's names (ADR 0084, rules 1 and 3; glossary *Demanded set*).
- The same holds where the child that answers is a Joinset or a Unionset, and where the Grainset is a Unionset branch.

### Glossary

- *Grainset*: "and any other Grainset holds no granularity of its own" becomes: a root Grainset, which declares none, carries the granularity that its children agree on, and none where they differ; the rules above replace "two at least and each at its own grain" and "No two children share a granularity, and all of them share one temporal shape". The _Avoid_ item that reads "two or more distinct grains" as permitting two children at one granularity goes.
- *Unionset*: "or is a child of a Grainset" becomes "or, as a child of a Grainset, carries a time axis".
- *Granularity fit*: "A Grainset picks its child by this order" becomes the child rank above. The entry also says that a child with no time axis ranks coarser than Year.
- *Child rank*: "to pick one" becomes: to send each demanded name to the child ranked first for it alone (*The pick*).
- *Path ambiguity*: as under *Ambiguity* above.

### Amends

- **0030:** in *Decision Outcome*, the strict equivalence across a Grainset's children, which ADR 0056 kept, does not bar a child with no time axis: it may stand beside children with one (0096's *No time axis*); "Grainsets additionally require ≥2 distinct grains", which ADR 0056 corrected to distinctness between every two children, reads: no two children have the same grain and the same coverage mask (0096's *Shared granularity*); the rest of 0030 stands.
- **0056:** in *A Unionset agrees; a Grainset aligns on kind and differs on grain*, "A Grainset requires one shape kind and a distinct grain per constituent" gains two exceptions: a Grainset's children may share a granularity, and compile refuses two children only where they have the same grain and the same coverage mask (0096's *Shared granularity*); a child with no time axis may stand beside children with one, which is the relaxation that the paragraph's "provisional" restriction allows (0096's *No time axis*); in *The grain is derived where it is unanimous, …*, "a constituent with no grain is a constituent nothing can route to" no longer holds for a child with no time axis, and "A Grainset never satisfies it, by its own distinctness rule", there, and, in *Absence has one meaning per field*, "which for a Grainset is always" no longer hold: a Grainset that declares no granularity carries the granularity that its children agree on, and none where they differ (0096's *Shared granularity*); in *Decision Outcome*, "a **Grainset** requires one shape kind and a distinct grain per constituent" is read as the same sentence in *A Unionset agrees; a Grainset aligns on kind and differs on grain* is; "Distinctness is stronger than ADR 0030's "≥2 distinct grains"" reads: two children are equal only where they have the same grain and the same coverage mask (0096's *Shared granularity*); in *Three rules, three variants, two sites each*, rule B holds over the children that have a time axis, and rule C, "pairwise grain distinctness over the constituents", reads: no two children have the same grain and the same coverage mask (0096's *No time axis* and *Shared granularity*); the rest of 0056 stands.
- **0064:** in *Decision Outcome*, under **Granularity fit**, the premise "A public Grainset holds no granularity of its own" reads: a Grainset that declares no granularity carries the granularity that its children agree on, and none where they differ (0096's *Shared granularity*), and it still ranks at the granularity of what its routing answers with; "The Grainset's Strategy picks its child by granularity fit alone; it never routes to a child with no granularity (ADR 0074)" and "a Grainset's children are ordered by granularity fit alone (ADR 0074, *A Grainset*, as ADR 0084 amends it)" read: a Grainset picks its child by the child rank, and a child with no time axis is among the children that serve a demand that reads no time axis (0096's *The pick* and *No time axis*); the resolver's rank for public candidates does not change; the rest of 0064 stands.
- **0065:** in *A Grainset routes and does not read*, the premise "it holds no grain of its own" reads: a Grainset that declares no granularity carries the granularity that its children agree on, and none where they differ (0096's *Shared granularity*), and it still reads no rows; "It groups children holding one subject at different grains so a request can be routed to the grain that serves it" reads: it groups children holding one subject that differ by grain or coverage mask, so that a request can be routed to the child that serves it (0096's *Shared granularity*); the rest of 0065 stands.
- **0073:** in *One input for every Strategy call*, "A Grainset that is an operand of another composition picks its child at the granularity that it declares" reads: it picks its child at the granularity that it carries, declared or derived (0096's *Shared granularity*); the rest of 0073 stands.
- **0074:** in *A Grainset*, "Among its grains that carry a granularity, it picks the one that serves the whole demand and has the coarsest granularity equal to or finer than the requested one", and the two cases for no requested granularity that follow it, read: a Grainset sends each demanded name to the child that the child rank ranks first for it alone, and answers from one child where every name goes to it, or, where the chosen children share no guaranteed keyset, or its parent takes rows or partial aggregates from it, from the child that the child rank ranks first among those that serve the whole demand; a child with no time axis is among the children where the demand reads no time axis (0096's *The pick* and *No time axis*); in *Every request is planned against its grain first*, "A Grainset grain with no granularity is one that nothing can route to (ADR 0056), so Grainset routing reads only grains that carry one" and "A composed child always does too, because compile refuses one without it" do not hold for a child with no time axis, "A Dataset child always does, because the model refuses one without it (`GrainsetChildMissingGranularity`)" holds for a Dataset child that maps the time axis, and "a composition that is a constituent, carries no granularity, and publishes a temporal shape or is a child of a Grainset" is refused with `CompileError::ComposedConstituentMissingGranularity` where it publishes a temporal shape, and with `CompileError::ComposedConstituentMissingTemporalShape` where it is a child of a Grainset, carries a time axis and publishes no temporal shape; in the same section, under **A composition that is a constituent**, "A Grainset that is a constituent derives none, because its children differ by granularity, so its author declares one" reads: a Grainset that is a constituent derives the granularity that its children agree on, and its author declares one where they differ (0096's *Shared granularity*); wherever it says that a Grainset that is a constituent carries, picks at or fixes "the granularity that it declares" or "the granularity that the Grainset declares", in *Every request is planned against its grain first*, *A Grainset* and *Consequences*, it reads the granularity that it carries, declared or derived, and "declares a granularity that every child serves", in *Consequences*, reads: carries a granularity that every child serves; in *Every request is planned against its grain first*, "it never holds its time axis at it: no two of its children share a granularity, so one of them at least is not at it" no longer holds where its children share that granularity (0096's *Shared granularity*); in its entry for 0065, ""it holds no grain of its own" holds for a Grainset that is no constituent of another composition, and one that is a constituent holds the granularity that it declares" reads as 0096's entry for 0065 gives; in its entry for 0056, "the two refusals of a constituent's granularity" reads: those two and `ComposedConstituentMissingTemporalShape`; in *Every request is planned against its grain first*, "A composed source comes to the planner with no granularity only where it is a root composition, or a constituent that publishes no temporal shape, under a Unionset or a Joinset" also holds under a Grainset, for a child with no time axis (0096's *No time axis*); the same clauses in its `### Amends` entries for 0056 and 0064 and in *Consequences* read the same way; the rest of 0074 stands.
- **0084:** in *How they are joined*, rule 6 orders the left-deep joins by the child rank with the granularity fit first, then fewest Dimensions, then name order, so that the first child is at the requested granularity, or, where no chosen child carries it, the chosen child closest to it; and the joins are `Full` (0096's *Drill-across at one granularity*); in rule 3, the child that each name goes to is the child that the child rank ranks first for it alone (0096's *The pick*); in *Which children answer*, rule 2, "a Grainset that is a constituent of another composition fixes the granularity that it declares" and "runs rule 3's pick among the children that pass the grain test at the granularity that it declares" read the granularity that it carries, declared or derived (0096's *Shared granularity*); in *Which children answer*, rule 1, after "as ADR 0074 picks it." the child that answers alone takes the whole demand, each predicate and each activated named Filter included, and it reads its own signature for the filter (rule 8) (0096's *A named filter on a Grainset*); the rest of 0084 stands.
- **0080:** in *A named filter is placed as a request predicate*, an activated named filter on a Grainset also goes into the child that answers alone, which reads its own signature for it (0096's *A named filter on a Grainset*); the rest of 0080 stands.

### Consequences

- Good: a model with reports that share Day and differ by breakdown, and a model with a report that has no date, compile and plan.
- Good: the order of the joins of a drill-across follows the child rank with the granularity fit first, which the author can tell from the children's Dimensions and names, and not a hash.
- Good: an activated named filter on a Grainset is answered by the child that answers alone, with no `NoSignature`.
- Neutral: the code changes with the decisions above: compile's distinctness check, the two refusals for a missing granularity, the refusal `ComposedConstituentMissingTemporalShape` for a missing temporal shape, the child rank in the planner's pick, the tie check and `AmbiguousCandidates` for a Grainset's children, and the frame of a Grainset that passes a named filter unchanged.
- Neutral: the glossary entries *Grainset*, *Unionset*, *Granularity fit*, *Child rank* and *Path ambiguity* change with this decision.
- Bad, a trade-off: the child rank has an exception where the demand selects the time axis without a granularity, so the planner holds two orders; gain: that request keeps today's pick, Day rows, where the single order moved it to Month rows.
- Neutral: the joins are `Full`, so a group that only one child holds is in the answer, and a NULL group gives one row with the values of each child (ADR 0084 rule 6).

## More Information

Decided in Change 1, and for *A named filter on a Grainset*.
