---
number: 33
title: 'Composite plan shape: implicit Unionsets, grain routing, anchored Joinsets'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-planner/src/dataset_strategy.rs
- crates/semstrait-planner/src/unionset_strategy.rs
- crates/semstrait-planner/src/grainset_strategy.rs
- crates/semstrait-planner/src/joinset_strategy.rs
- crates/semstrait-planner/src/routing.rs
links:
- target: 45
  kind: amendedby
- target: 48
  kind: amendedby
- target: 54
  kind: amendedby
- target: 74
  kind: amendedby
- target: 73
  kind: amendedby
- target: 70
  kind: amendedby
- target: 84
  kind: amendedby
---

# Composite plan shape: implicit Unionsets, grain routing, anchored Joinsets

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0045](0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md): an implicit unit is an ordinary `semantic_sources` entry, its id seeded on its originator-qualified name rather than content; `RoutingUnitRef` and `DataKindOrigin` retire.
> - [ADR 0048](0048-composite-keysets-are-authored-inside-a-boundary.md): a Grainset emits no implicit Unionset; its cross-grain Joinsets read the grains' guaranteed keysets, and a Grainset whose grains carry no keyset is legal and "needs no diagnostic".
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): its decision "anchored Joinsets" is replaced by the lead of 0054's *Decision Outcome*, the bracketing rule of its *Identity, shape and name*, the `MAX_COMPOSITION_DEPTH` paragraph of its *The bound is on units emitted, not on constituents* and, in its *Consequences*, "Anchor retires as a term", and the sections that only apply it read as those give: its title's "anchored Joinsets"; in *Context and Problem Statement*, "moving Joinset ambiguity or depth checks back to plan time" and "two nearly identical BFS traversals"; in *Considered Options*, the chosen option's "anchored Joinsets" and the option "The anchorless symmetric depth-capped BFS"; and in *Decision Outcome*, the bullet "A Joinset declares exactly one anchor". In its decision "grain routing", compile emits one implicit Joinset per ordered pair of grains, finer to coarser only. The rest of 0033 stands.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): every semantic source over one base path in one format shares one physical source, whatever its selector; a `Pattern` holds the glob rewritten for its base path, so a consumer that reads globs can pass it to the engine as it is (ADR 0067's `FilesScan.pattern`); a folder-position wildcard still multiplies sources, one physical source for each folder the wildcard matches.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): in *Decision Outcome*, a Grainset is answered by a grain only, with a pass-through combine; its stand-ins are its grains (ADR 0064), and ADR 0054's cross-grain Joinsets are not among them in MVP, which loses no answer: interface-first is the only authoring direction (ADR 0053), and under it every grain of one Grainset maps the same interface, so a cross-grain unit never serves a demand that no grain serves; the rule for cross-grain units stays open for Grainsets until a narrower grain can be authored.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): a Unionset pre-aggregates per branch only in mode `all` and only when every aggregate the request needs (each demanded Measure, and each aggregate node of a demanded Metric) is additive or semi-additive; otherwise the branches give rows and one aggregate sits above the union, the form 0033 rejected; the disjointness elision is a selected Dimension that folds to a distinct non-NULL `Literal` on every branch, through a composed branch's operands as ADR 0073 folds them to prune a branch, a Joinset on the way giving a literal for a name that both operands map only where both give the same one, and a Unionset or a Grainset only where every branch or child gives the same one, still distinct after the request's bucketing, in mode `all` only, where each Metric of two aggregate steps groups its step 1 by the demand's grouping alone.
> - [ADR 0084](0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md): in *Decision Outcome*, "when no unit covers the Request, builds a LEFT OUTER JOIN tree equi-joined on Keys declared on both interfaces. Missing or disagreeing Keys are hard compile errors", as ADRs 0048 and 0073 changed it, reads: where the demanded names go to two or more children, the Grainset drills across them with `Full` joins on the demand's grouping Dimensions, compared NULL-safe, as 0084's rules 4 to 6 give, and children that carry no shared guaranteed keyset are no compile error: the Grainset then drills across no children (0084's rule 4); the rest of 0033 stands (0084's *Which children answer* and *How they are joined*).

## Context and Problem Statement

Everything this decision fixes is ratified observable plan shape plus manifest structure that adapters, tests and correctness reasoning depend on. Changing coverage strictness either breaks existing models or turns NULLs into errors, and moving Joinset ambiguity or depth checks back to plan time changes the planner-completeness contract.

It is deliberately surprising: a glob becomes an N-branch union tree with double aggregation rather than one multi-path Scan, unnamed nodes nobody authored appear as top-level DataKinds, one synthesis mechanism carries two opposite coverage disciplines, cross-grain joins ignore the relationship graph that exists for exactly that purpose, and two nearly identical BFS traversals have different depth bounds and different failure stages.

## Considered Options

- Implicit Unionsets, grain routing and anchored Joinsets (chosen).
- A single multi-path `Scan` delegating fan-out to the engine.
- One uniform strict coverage discipline across both synthesis seams.
- Inlining implicit Unionsets as `NestedDataKind` children.
- Unioning raw rows and aggregating once above the Union, leaving pre-aggregation and elision to the optimizer.
- Relationship-path-based cross-grain joins with runtime fallback to the best-covering unit (G-2d).
- Rejecting partial coverage outright.
- The anchorless symmetric depth-capped BFS with plan-time `AmbiguousImplicitComposition`.

## Decision Outcome

- Multi-source Bindings and same-grain Grainset children compile into *implicit Unionsets*, stored as top-level manifest DataKinds (`origin = Implicit`, content-derived id, referenced via `RoutingUnitRef::Synthesized`). Their coverage discipline is non-strict for the Grainset seam and strict for the Dataset fan-out.
- A Union always pre-aggregates per branch, and elides the aggregation above the Union only on a provable disjointness predicate.
- A Grainset normalizes children into per-grain routing units. It either splices the winning unit's own subplan plus a rollup wrapper or, when no unit covers the Request, builds a LEFT OUTER JOIN tree equi-joined on Keys declared on both interfaces. Missing or disagreeing Keys are hard compile errors, and Relationships are deliberately not used.
- A Joinset declares exactly one anchor, canonicalized to `constituents[0]`/`Join.left` (reverse hops flip `KeyPair` orientation). Paths are resolved at compile by anchor-rooted BFS, exempt from the depth-4 cap.

## More Information

### Open questions

Grainset cross-grain composition warns about fanout while composition ambiguity elsewhere is a hard error (see `refuse-ambiguity-name-order-selection`). Whether these are one posture or two is undecided; the two may be inconsistent.
