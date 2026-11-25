---
number: 54
title: Compile enumerates every legal sub-composition of an authored Joinset; `integrity` decides legality and the planner decides use
status: accepted
date: 2026-08-26
source:
- CONTEXT.md
- docs/adr/0011-thin-manifest-in-memory-graph.md
- docs/adr/0014-eager-bounded-composition-enumeration.md
- docs/adr/0015-refuse-ambiguity-name-order-selection.md
- docs/adr/0016-optional-catalog-model-wins.md
- docs/adr/0026-inline-only-nesting-tree.md
- docs/adr/0029-joinset-scoped-relationship-shadow.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0051-entity-ids-are-derived-from-the-natural-key.md
links:
- target: 11
  kind: amends
- target: 15
  kind: amends
- target: 28
  kind: amends
- target: 29
  kind: amends
- target: 33
  kind: amends
- target: 38
  kind: amends
- target: 44
  kind: amends
- target: 45
  kind: amends
- target: 48
  kind: amends
- target: 50
  kind: amends
- target: 52
  kind: amends
- target: 64
  kind: amendedby
- target: 70
  kind: amendedby
- target: 74
  kind: amendedby
- target: 14
  kind: supersedes
- target: 34
  kind: supersedes
- target: 49
  kind: amends
---

# Compile enumerates every legal sub-composition of an authored Joinset; `integrity` decides legality and the planner decides use

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0064](0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md): the refusal posture is narrowed, not abandoned. ADR 0054 and the former **Path ambiguity** entry made a tie something *"reported for the author to resolve, never silently tie-broken"*. It is still never silent — the advisory is the report, and it goes to the caller, who resolves it now with `from` — but the request is answered rather than rejected. It was found that ambiguity is a property of the entry point, not the artifact, so compile cannot report it and a refusal would fail requests the model can in fact answer. ADR 0054's own case — two sub-compositions over the same constituents with different relationship sets — falls under the same rule inside a scope: name-least is planned, the advisory tells the author a scope-local shadow is wanted.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): in the `N > 1` consequence, a reduction leaf's id is `mint(Dataset id, physical-source id)`, not seeded by its name, and a glob multiplies physical sources only where a folder wildcard matches several folders.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): answers the planner handoff: the rule for leaving out an operand reads the join type, and the deduplication takes the place of cardinality.

## Context and Problem Statement

Which sub-compositions should compile emit for an authored Joinset, and who chooses among them? An earlier framing of this decision tried to make `integrity` a *cost* input — enumerate less where a join is lossy — and it collapsed, because a cost rule cannot be stated without knowing the request. The same fields answer a question that can: *may this constituent stand apart from this relationship at all?*

## Considered Options

- Enumerate every legal sub-composition; `integrity` decides legality (chosen).
- **`integrity` as a cost input — enumerate fewer units where a join loses rows.** Refused because a cost rule cannot be stated without the request, and because it conflates the two operations the ruling separates; the planner has every field it needs to make that judgement per request.
- **Keeping ADR 0038's one-directional `integrity` and adding a clause.** Detaching the `to` side would additionally require `join_type ∈ {Right, Full}`. It preserves the ratified definition and the standard meaning of referential integrity, and it is the stronger reading on the data. Refused because it makes legality read a field assigned to the planner, and because it splits one field's meaning across two claims.
- **Enumerating every bracketing**, and **enumerating spanning trees only**. The first multiplies the artifact for a consumer-side rewrite; the second silently drops a mandatory relationship inside a cycle, which is a wrong answer rather than a smaller artifact.
- **Deduplicating units that cover the same constituents**, or tie-breaking among them at compile. Refused because equal-mask units *are* path ambiguity, which `CONTEXT.md` requires be reported rather than resolved silently.
- **Bounding the constituent count per Joinset.** Simpler for an author to understand. Refused because a star and a complete graph over eight constituents differ by three and a half orders of magnitude, so one number cannot serve both.
- **Bounding implicit composition depth**, ADR 0014's own mechanism. Refused on the ground that a compiler may legitimately compose deeper than a modeller assumed; the bound belongs on what a person writes.
- **Naming a unit `<originator>.<relationship>`.** ADR 0045's existing spelling, and the obvious reading of "namespace plus name". Refused because two units in one originator scope can share a causing relationship, so the name — and therefore the id — collides.
- **Compile-synthesised sub-Unionsets over a subset of a Unionset's branches**. Refused because the whole gain is a branch not read, which the earlier base-schema design assigned to the planner, and because such a unit reaches neither of ADR 0048's two keyset routes — derivation belongs to a Joinset alone and a boundary needs an author — so nothing could compose onto it. This is the third application of that reasoning, after ADR 0048 withdrew grain groups and ADR 0049 applied it to the synthesised Unionset.

## Decision Outcome

For one authored Joinset, compile emits one semantic source for **every legal sub-composition** of its specification — nothing withheld on cost grounds — and the planner decides per request whether to use, skip or bypass any of them. The two operations that read the same fields are separated deliberately, and the separation is what makes the rest tractable. **Legality** is compile's: `integrity` decides whether a constituent may be detached from a relationship at all. **Use** is the planner's: it chooses among the legal units, and the `role_definition` carries `join_type`, `cardinality`, `integrity` and `cross_filter` precisely so it can.

This reverses the central bet of ADR 0011 for authored Joinsets and re-homes ADR 0014 wholesale, so both are stated rather than absorbed. The reversal is narrow, and that is its load-bearing part: **sub-compositions within one authored Joinset's own specification**, never the model-wide relationship adjacency index, which is the structure ADR 0011 actually names and which stays forbidden.

### The enumeration rule

Let a Joinset's specification give constituents `C` and relationships `R` — its Joinset-local declarations plus the root-level ones, with a scope-local shadow replacing the root-level declaration it names, per ADR 0029.

1. A relationship is **bypassable** when its `integrity` is `Enforced` or `Assumed`, and **mandatory** when it is `None`.
2. A **candidate member set** is a subset `S ⊆ C` that is connected in `R` and **closed under mandatory relationships**: for every mandatory relationship of `R` with both endpoints in `C`, if either endpoint is in `S` then both are.
3. A **unit** is a pair `(S, E)` where `E` is a set of relationships of `R` over `S` that connects `S` and **contains every mandatory relationship whose endpoints both lie in `S`**. Compile emits one semantic source per unit.
4. Distinct `E` over one `S` are **distinct units and are never deduplicated.** Two compositions covering the same constituents carry the same coverage mask, which is exactly `CONTEXT.md`'s **Path ambiguity** — *"reported for the author to resolve, never silently tie-broken"*. Deduplicating them, or tie-breaking at compile, destroys the only evidence an author could be shown.

Closure alone is not enough, so step 3 is stated over relationships rather than over a spanning tree. Closure makes both endpoints of a mandatory relationship *present*; it does not make the relationship *applied*. In a cycle of mandatory relationships no tree can hold them all, so a tree-shaped unit would silently drop a filter the modeller intended. **A unit is therefore a connected sub-graph, not a tree**, and the relationships outside its traversal spine become additional conjuncts in its join condition — expressible because ADR 0050 makes `on_expr_id` one expression rather than a list.

### Identity, shape and name

**A unit is identified by `(S, E)`, and its bracketing is canonical rather than a third axis.** Each unit is realised as a left-deep chain of binary joins, per ADR 0050's rule that a cumulative unit joins its **two immediate operands** and never inherits theirs. The chain follows one deterministic traversal: start at the name-least constituent of `S`, then repeatedly join the name-least constituent reachable through a relationship of `E` not yet used. Every intermediate operand of that chain is itself a unit, so the chain costs no entries beyond the units already enumerated.

Emitting every bracketing was refused. It multiplies the artifact by up to Catalan(*k*−1) — fourteenfold at five constituents — to buy a rewrite ADR 0050 already assigned to the consumer, since *"squashing stays a consumer optimisation rather than a second persisted shape"* and a re-bracket is a squash read from the other end. The cost is disclosed: **re-bracketing is not semantics-preserving across outer joins**, so the persisted bracketing is one valid composition rather than the only one, and a consumer that re-brackets must respect preserved sides. That is the same cost ADR 0050 already accepted for squashing.

**A unit is named after the one relationship that caused it, qualified by its left operand's name, applied recursively.** Under this rule every unit — straight or cumulative — is caused by exactly one relationship: the one adding the new constituent. For `r1(A,B)` and `r2(B,C)` the names are `<originator>.r1`, `<originator>.r1.r2` and `<originator>.r2`. Qualifying by the **left operand** rather than by the originator is load-bearing: `((A,B),C)` and `(B,C)` are both caused by `r2` inside the same originator, so an originator-qualified name would give both `<originator>.r2` and, since the name seeds the id under ADR 0045, merge two distinct resolution paths into one entry — the failure 0045 names for itself. Chaining the qualification is the idiom ADR 0051 already ratified, which mints an id-less entity from *"the name qualified by the path from its nearest named root … `<parent>.<name>` nested"*.

**A derived unit name that collides with an authored data-kind name is a hard error.** ADR 0048 records that ADR 0045 *"flagged [this hazard] for sources and left it without a rule"*; this is the rule, on 0048's own precedent for derived keyset names.

### The authored Joinset is the unit that applies every relationship

**The authored Joinset's own entry is the unit whose relationship set is the whole of `R`.** It is the only maximal unit that applies every relationship the author wrote, it is unique, so no tie-break is needed, and it makes the authored name unambiguous where several sub-graphs span `C`. It keeps the authored `public` value.

**Every other unit is private, and reachable only through its public origin.** ADR 0050 already rules that *"public and private joinsets have identical structure and logic, at every depth"*, so `public` cannot narrow enumeration — only exposure. `CONTEXT.md`'s **Public semantic source** is *"a semantic source a request may name directly in its `from`"*, and every other one *"exists only as a resolution path element"*, which is what an enumerated unit is. So the earlier base-schema design's *"all semantics compositions is queryable for joinset if it is public defintion"* reads as **reachable by mask routing**, not nameable: routing into a private unit outside its originator is not merely unnecessary but illegal, and compile-derived names never enter the request surface. That makes the `originators` edge an access path as well as provenance, which is a rule ADR 0045 does not state.

### `integrity` claims the composition is consistent, so an attested relation filters neither side

`integrity` tags a relationship, the relationship is directed from `from` to `to`, and the tag claims **that composition is consistent**. An attested relationship filters neither side, so any constituent it touches may stand apart from it.

**This widens ADR 0038's definition rather than citing it, and the widening is the point.** ADR 0038 defines integrity as *"every `from` row finds a `to` row"* — a totality claim about the `from` side alone, which is what referential integrity means in every database. Under that reading an inner join still drops unmatched `to` rows, so no attestation could ever make the `to` side detachable, and the rule above would be unstatable without reading `join_type` — a field this decision assigns to the planner.

The price is named: **a modeller who writes `integrity: enforced` meaning ordinary referential integrity gets more units than they expect.** `CONTEXT.md` gains an **Integrity** entry, the term's first definition anywhere — it occurs once in the glossary today, only as a field name inside **Relationship**, and its three variants carry no documentation in the model crate.

**`Enforced` is trusted as authored when no catalog is supplied**, and a catalog that contradicts it yields a warning rather than an error. That is ADR 0016's posture applied unchanged — the catalog is `Option<&dyn CatalogProvider>` and *"with no catalog the model is the source of physical truth"*, with the model winning on conflict. One consequence protects a merged ruling: **the enumeration is a function of the model alone**, so ADR 0041's `model_hash`-only recompile gate is untouched and a change of catalog state can never change the entity set.

### The bound is on units emitted, not on constituents

`MAX_IMPLICIT_ENUMERATION_COUNT = 2000` is re-homed from `SemanticGraph` build to the enumerating compile stage, **per authored Joinset**, with a breach a **hard error and never truncation**. One `CompileError` variant carries the Joinset and the count reached; per ADR 0044 the rule is stated in the enumerating stage's own module documentation and joins no roster.

Bounding constituents was refused because topology dominates. Counting spanning trees alone — a lower bound, since step 3 admits sub-graphs — eight constituents give 127 units as a star and 441 204 as a complete graph, so any constituent cap either forbids ordinary models or admits fatal ones. The value 2000 is kept rather than re-derived, so re-homing changes the owning stage and not the number. **Six pairwise-related constituents in one Joinset is now a hard compile error.** A star of five conformed dimensions is 31 units and costs nothing, so the bound bites density rather than size.

ADR 0014's no-truncation rule transfers **strengthened**. Its ground was refusing to *"hide incomplete enumeration from authors"*, and a persisted artifact has none of the escape hatches a per-request graph build had.

**`MAX_COMPOSITION_DEPTH = 4` bounds authored nesting, and it is model validate's.** The subject is inverted from the retired position, which bounded implicit depth and *exempted* authored Joinsets outright as the escape hatch: what needs bounding is what a person can construct, because compile may legitimately compose deeper than a modeller assumed. The value reuses ADR 0014's ratified number so nothing new is invented. The stage is model validate rather than compile, since nesting depth is a property of the authored document — where ADR 0045 already put nested-name uniqueness as an obligation on the duplicate-name check.

**Neither bound persists.** A limit is a property of the compiler, not of the artifact, which is the retired constant's own non-persisted position.

### Amends

Each supersession is scoped in this body rather than by editing the older ADR's frontmatter, per the precedent ADR 0037 set over ADR 0031.

- **0011:** the enumeration clause, for authored Joinsets only: sub-compositions of one authored Joinset's specification are persisted. The **model-wide relationship adjacency index stays forbidden**, and so does every other structure 0011's rejected alternative names — `nodes:`/`edges:`/`compositions:`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, a pre-built `RelationshipGraph` and a serialized graph cache. **This is the second trade against 0011, not the first**: ADR 0049 already rejected *"synthesise the Unionset in the graph rather than in the manifest, which is ADR 0011's posture"* as a named alternative and recorded that its own reduction is *"more entities for the same model, in an artifact whose stated posture is thinness (ADR 0011)"*, without registering a supersession. 0011's open question about planner-completeness is resolved in the direction it anticipated: for these compositions, planner-completeness is met by the artifact rather than by graph-build synthesis.
- **0014:** Replaced rather than moved: compile emits every legal sub-composition of one authored Joinset's specification, never the model-wide relationship adjacency index, which stays forbidden. `MAX_IMPLICIT_ENUMERATION_COUNT = 2000` is re-homed from `SemanticGraph` build to the enumerating compile stage, per authored Joinset, and the no-truncation rule transfers strengthened (0054's *The bound is on units emitted, not on constituents*). The **depth** bound on implicit enumeration retires, with its subject inverted onto authored nesting: `MAX_COMPOSITION_DEPTH = 4` bounds authored nesting, and it is model validate's. Both build outcomes retire: `CompositionDepthExceeded` with the bound it reported, and `NoCompositionPath` because under eager enumeration a missing path is a plan-time absence of a candidate rather than a compile fault. 0014's posture is otherwise upheld — enumeration stays eager and exhaustive, and plan time stays a lookup. Nothing moves in code: both constants and both outcomes are grep-absent from `crates/`.
- **0015:** an authored Joinset duplicating an enumerable implicit one is not rejected; the two carry different names and ids, so nothing collides.
- **0028:** a catalog that contradicts an authored `Enforced` integrity yields a warning rather than an error; with no catalog, it is trusted as authored.
- **0029:** `COMP_E_0414`'s canonical form lost its subject; an authored Joinset duplicating an enumerable implicit one is *not* rejected, because the two carry different names, so they are different entries with different ids and nothing collides.
- **0033:** its decision "anchored Joinsets" is replaced by the lead of 0054's *Decision Outcome*, the bracketing rule of its *Identity, shape and name*, the `MAX_COMPOSITION_DEPTH` paragraph of its *The bound is on units emitted, not on constituents* and, in its *Consequences*, "Anchor retires as a term", and the sections that only apply it read as those give: its title's "anchored Joinsets"; in *Context and Problem Statement*, "moving Joinset ambiguity or depth checks back to plan time" and "two nearly identical BFS traversals"; in *Considered Options*, the chosen option's "anchored Joinsets" and the option "The anchorless symmetric depth-capped BFS"; and in *Decision Outcome*, the bullet "A Joinset declares exactly one anchor". In its decision "grain routing", compile emits one implicit Joinset per ordered pair of grains, finer to coarser only. The rest of 0033 stands.
- **0034:** There is no authored anchor: "Anchor retires as a term".
- **0038:** widens the `integrity` definition from "every `from` row finds a `to` row" to a claim that the composition is consistent, so an attested relationship filters neither side, as set out in 0054's *`integrity` claims the composition is consistent, so an attested relation filters neither side*. 0038's authoring grammar, its four join-key rules, its cardinality conditionality and its *"absence is never evidence"* principle are untouched — the last of these does not reach `Integrity::None`, which is a deliberate statement rather than an absence, since an author who writes nothing gets `Assumed`.
- **0044:** `MAX_COMPOSITION_DEPTH = 4` bounds authored nesting and is model validate's rule, not the enumerating stage's.
- **0045:** two clauses. The `role_definition` arm spellings (`Joinset{from, to}`, `Unionset{branches}`, `Grainset{grains}`) are **shorthand rather than exhaustive**, which is what ADR 0049 assumed when it asserted a value for a `mode` no ADR had placed: `mode` rides the **Unionset** arm by ADR 0043's placement rule, while `join_type` / `cardinality` / `integrity` / `cross_filter` ride the **Joinset** arm, and no field is shared between arms. And the implicit-source naming rule gains the recursive left-operand qualification of 0054's *Identity, shape and name*: an implicit unit's name is qualified by its left operand, recursively, not by the originator alone.
- **0048:** implicit-Joinset enumeration withholds nothing on cost grounds and its legality reads `integrity` alone, so the `(cardinality, integrity, preserved side)` triple is not what prunes it.
- **0049:** the synthesised multi-source Unionset exists at `N > 1`. A Unionset stacks branches, so with one branch there is nothing to stack and the composition means nothing: a Dataset over one physical source stays a leaf and keeps the authored name.
- **0050:** a unit is named after the one relationship that caused it, qualified by its left operand's name, applied recursively. Under this rule every unit — straight or cumulative — is caused by exactly one relationship: the one adding the new constituent.
- **0052:** `integrity` alone decides which sub-compositions are legal, so the `(cardinality, integrity, preserved side)` triple is not what prunes implicit-Joinset enumeration.

**That a Joinset's mask is the union of its two operands' masks is confirmed structural rather than checked**, which ADR 0047 already recorded.

### Consequences

- **A Grainset emits one implicit Joinset per ordered pair of grains, finer to coarser only.** The rollup-direction constraint is the bound; it makes the grain graph acyclic, so three grains yield three units rather than six. Chains through an intermediate grain are not emitted: they carry the same mask as the direct pair while being a strictly worse plan, so they would manufacture the equal-mask ambiguity this decision deliberately preserves for the real case. The basis resolves from the **grains'** guaranteed keysets, never the Grainset's own, per ADR 0048.
- **The synthesised multi-source Unionset exists at `N > 1`.** ADR 0049 routed the wrapper's existence here. A Unionset stacks branches, so with one branch there is nothing to stack and the composition means nothing: a Dataset over one physical source stays a leaf and keeps the authored name. The cost is that the threshold is visible in the artifact — when a second physical source appears the authored name moves from the leaf to the new Unionset, so that leaf takes a derived name and, since the name seeds the id (ADR 0051), is re-identified along with every `semantic_overrides` row, keyset id and bit position keyed to it, while ADR 0041 gates recompile on `model_hash` and the model did not change. That cost is accepted rather than removed, because wrapping at `N == 1` would not buy invariance: it would fix which entity holds the authored name, but ADR 0046 makes the source roster a function of filesystem contents, so the leaf set varies with a glob's match count either way.
- **`integrity: none` makes the artifact smaller, and the default makes it largest.** Mandatory relationships bind constituents together, so they remove candidate member sets. `Assumed` is the model's `#[default]`, so an author who writes no `integrity:` gets the fullest enumeration — the opposite of the intuition that a stronger claim costs more.
- **Three `CONTEXT.md` entries described a traversal the model cannot express, and `Integrity` is added.** There is no authored anchor: `anchor` occurs nowhere in `semstrait-model`, and `JoinsetBody` is exactly `{base, relationships, datasets, grainsets, unionsets}`. So **Anchor retires as a term** — it is not rewritten but removed, with the refusal recorded in **Hop**'s `_Avoid_` list so a reader who reaches for it lands somewhere — **Hop** is rewritten from a step-within-a-path into a composition in its own right, **Joinset** loses *"walking declared relationships outward from an anchor"*, and **Explicit composition** loses *"under a content-derived identity"*, which ADR 0045 superseded when it made identity name-seeded. This discharges the flag ADR 0050 left: it recorded **Anchor** and **Hop** as *"flagged, not edited … that wording belongs to the enumeration decision."*
- **An authored Joinset duplicating an enumerable implicit one is *not* rejected; this follows from the rules above rather than being weighed here.** ADR 0050 routed the question in — it is what survived of `COMP_E_0414` once the canonical form lost its subject. The answer falls out of identity: an authored Joinset over `{A,B}` and the implicit unit over `{A,B}` inside a larger Joinset carry **different names** (the authored one, versus `<originator>.<relationship>`), so they are different entries with different ids and nothing collides. One is public and one is private, so nothing is ambiguous at routing either. The two differentiators that arrived with the question — a declared basis and a Joinset-level `filter:` — need no special treatment, because both are already inside the unit's own pooled join condition. `CONTEXT.md`'s **Explicit composition** clause that an explicit declaration *"must differ from [an implicit one] structurally to be worth declaring"* stays **advisory**, which is what it says: it tells an author when a declaration earns its keep, and it is not a rejection rule. A duplicate is therefore a redundancy an author may write, not a fault — the only outcome consistent with public and private joinsets having identical structure and logic.
- **Handoff to the planner effort: `integrity`'s direction still matters at plan time.** This decision makes an attested relationship's *legality* symmetric, which is a statement about which units exist. It does not make an outer join's row-set behaviour symmetric, so a planner choosing between two legal units must still read `join_type` and `cardinality` to know which one answers the request faithfully.
- **Handoff to the implementation effort.** The composite guard rejected every complex form at 14 sites at the time of the decision and was the only thing standing between this ruling and a manifest that holds one; ADR 0048 already ruled the single-leaf composite guard legacy. `JoinsetHop` (17 references), `NestedDataKindVariant` (25), `RoutingUnitRef` (29) and `path_origin` (4) are retired by ADR 0045 and still present. Nothing in `crates/` is changed by this decision.

## More Information

The `N > 1` consequence was corrected on 2026-08-26 on a decision: it read `N ≥ 1` and argued for always wrapping, and the bound contradicted its own subject, since a **multi-source** Unionset has no instance at one source. It was repaired in place rather than in a later ADR, on the human's explicit call, so the rule is read once and in one place.
