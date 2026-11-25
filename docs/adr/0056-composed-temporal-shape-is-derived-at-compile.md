---
number: 56
title: A Complex data kind's temporal shape is derived at compile, and a horizontal composition takes the highest shape rank
status: accepted
date: 2026-08-26
source:
- CONTEXT.md
- docs/adr/0011-thin-manifest-in-memory-graph.md
- docs/adr/0026-inline-only-nesting-tree.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0052-compile-gates-the-cardinality-it-materialises.md
- docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- decided for a shape that a Joinset states
links:
- target: 11
  kind: amends
- target: 30
  kind: amends
- target: 74
  kind: amendedby
- target: 49
  kind: amends
- target: 96
  kind: amendedby
---

# A Complex data kind's temporal shape is derived at compile, and a horizontal composition takes the highest shape rank

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): where existing ADRs say "grain" for the temporal roster, they mean the granularity; in *Three rules, three variants, two sites each*, rule A's "full shape unanimity (kind body *and* grain)" reads "full shape unanimity (kind body)": the granularity is no part of a Unionset's agreement, as *A Unionset agrees; a Grainset aligns on kind and differs on grain* says, and "A Joinset raises nothing", there, and "a **Joinset** requires nothing", in *Decision Outcome*, hold only for a Joinset that is no constituent of another composition, since the two refusals of a constituent's granularity join the three rules, each with its own variant and two sites; "forbidding only `temporal.grain:` on it", in *A Unionset agrees; a Grainset aligns on kind and differs on grain*, and "re-scoped to authoring", in *The grain is derived where it is unanimous, …*, no longer hold for a composition that is a constituent of another composition: its author may declare its granularity, which it carries in place of a derived one, and the derivation of the granularity, not of the temporal shape, stops at a constituent that declares one, so "absent otherwise", there, in *Decision Outcome* and in its `### Amends` entry for 0049, and "`grain` is empty wherever the constituents do not agree on one", in *Absence has one meaning per field*, hold only for a composition that declares none (0074's *Every request is planned against its grain first*); in *Consequences*, `GrainsetChildMissingGranularity` is neither re-cut nor retired, and stays for a Dataset child of a Grainset: compile refuses, with `CompileError::ComposedConstituentMissingGranularity`, a composition that is a constituent, carries no granularity, and publishes a temporal shape or is a child of a Grainset (0074's *Every request is planned against its grain first*).
> - [ADR 0096](0096-a-grainset-s-children-differ-by-grain-or-coverage-mask-and-a-grainset-picks-the-child-closest-to-its-demand.md): in *A Unionset agrees; a Grainset aligns on kind and differs on grain*, "A Grainset requires one shape kind and a distinct grain per constituent" gains two exceptions: a Grainset's children may share a granularity, and compile refuses two children only where they have the same grain and the same coverage mask (0096's *Shared granularity*); a child with no time axis may stand beside children with one, which is the relaxation that the paragraph's "provisional" restriction allows (0096's *No time axis*); in *The grain is derived where it is unanimous, …*, "a constituent with no grain is a constituent nothing can route to" no longer holds for a child with no time axis, and "A Grainset never satisfies it, by its own distinctness rule", there, and, in *Absence has one meaning per field*, "which for a Grainset is always" no longer hold: a Grainset that declares no granularity carries the granularity that its children agree on, and none where they differ (0096's *Shared granularity*); in *Decision Outcome*, "a **Grainset** requires one shape kind and a distinct grain per constituent" is read as the same sentence in *A Unionset agrees; a Grainset aligns on kind and differs on grain* is; "Distinctness is stronger than ADR 0030's "≥2 distinct grains"" reads: two children are equal only where they have the same grain and the same coverage mask (0096's *Shared granularity*); in *Three rules, three variants, two sites each*, rule B holds over the children that have a time axis, and rule C, "pairwise grain distinctness over the constituents", reads: no two children have the same grain and the same coverage mask (0096's *No time axis* and *Shared granularity*); the rest of 0056 stands.

## Context and Problem Statement

The earlier base-schema design asked for exactly this, a shape carried in the semantic source body and derived for Complex data kinds — *"we need additionally to derive TemporalShape for complex ones if they are not defined"* — while a manifest invariant and `manifest/data_kind.rs`'s module doc both put the derivation at `SemanticGraph` build. The question is where the derivation runs and what rule each form follows.

The forms do not share one predicate, because only one of them stacks. `CONTEXT.md` already draws the line — composition is *"vertically by stacking **equivalent** shapes, horizontally by joining **complementary** ones"* — so equivalence is the Unionset's word and was never the Joinset's. Read literally over a Joinset's operands, ADR 0030's *"strict equivalence required across children as hard errors"* makes an ordinary star join a hard compile error, and forbids the `Events ⋈ Scd` composition ADR 0030 itself contemplates when it rules that an **unavailable** `AsOf` traversal hard-fails — a plan-stage capability check, not a shape conflict.

## Considered Options

- Derive at compile and persist flat on the semantic source, one rule per form, a Joinset taking the highest shape rank (chosen).
- **Derive at `SemanticGraph` build**, that invariant's own position and the one the manifest's module doc still asserts. It keeps the artifact thin and needs no supersession of ADR 0011. There is no derivation there to preserve — `crates/semstrait-ir/src/semantic_graph.rs` has zero temporal handling — so it would defer an unbuilt computation to an unbuilt consumer while leaving `temporal` in ADR 0043's defect class.
- **Strict equivalence across a Joinset's operands.** ADR 0030's literal wording, needing no supersession. It makes the ordinary star join a hard compile error, and forbids the `Events ⋈ Scd` composition ADR 0030 itself contemplates.
- **A hard error when two operands tie at the top rank with different bodies.** Sharper diagnostics, and no silently absent shape. It forbids joining two fact tables on different clocks, which is an ordinary model rather than an author error.
- **A four-by-four derivation matrix over the forms instead of a total order.** It reads more directly off the four shapes and can express a non-monotone cell. It guarantees neither associativity nor commutativity, so the derived shape would become a function of bracketing and of authored direction — the property ADR 0054 established is not semantically load-bearing.
- **Ranking `Timeseries` above `Snapshot`.** The two tie on row multiplication, so this cell is a judgement either way; rejected because only the `Snapshot` label warns a consumer off a cross-period fold.
- **A form-keyed polymorphic temporal body, one arm per data-kind form.** Symmetrical with `role_definition` and the shape a reader might expect. Once the grain sits on each constituent, no arm has a field of its own to hold, and the source would carry two form discriminants that must agree — an agreement check existing forever, against ADR 0054's rule that each `role_definition` arm carries only its own form's fields.
- **A discriminant recording why a shape or grain is absent.** Better diagnostics for a forgotten `temporal:`. No consumer branches on the reason, so it would persist a fact nothing reads.
- **Comparing the variant tag alone.** The cheapest predicate, and the one "strict equivalence" might be read as. Two branches both `events` on different axes would pass it, leaving the composed surface with no axis and compile selecting one.
- **Re-homing the coarsest-first ordering check.** It is enforced today and the rule is real. The ordering stops being persisted, so there is nothing left to violate.

## Decision Outcome

Compile derives the temporal shape of every Complex data kind and persists it **flat on the semantic source**, in the same slot a leaf uses. The rule is per form: a **Unionset** requires its constituents to agree, a **Grainset** requires one shape kind and a distinct grain per constituent, and a **Joinset** requires nothing of its operands where it states no shape — it takes the highest **shape rank** among the operands that carry a shape, under the total order `Scd < Timeseries < Snapshot < Events`. A Joinset that states a shape carries it where the stated shape equals a shape at that rank, or where no operand carries a shape; compile refuses any other stated shape. The **grain** is derived and carried where the constituents agree on one, and absent otherwise.

Absence carries **one meaning per field** and records no reason: an empty `temporal` means the source publishes no time axis, and an empty `grain` means it guarantees no single grain.

### The derivation runs at compile, and the result is persisted

Graph build is not a live alternative, and this is a fact rather than a preference: `crates/semstrait-ir/src/semantic_graph.rs` has **zero** occurrences of `temporal`, so no composed-shape derivation exists there or anywhere else in the tree. Choosing graph build preserves nothing, displaces nothing, and leaves a value unreachable to a planner ADR 0043 says *"needs no further lookup or I/O"*. This is the fourth time this effort has moved work out of graph build so planning becomes lookup, after ADR 0048's keyset materialisation, ADR 0050's pooled join condition and ADR 0054's enumeration.

The field is flat on the source rather than inside `role_definition`, matching the earlier base-schema design's own sample, and it stays flat because after the grain moves onto the constituents no form has temporal content of its own to hold. `TemporalShape` remains the four-variant kind plus a grain; that is the whole of its polymorphism.

**Persisting a derived value is against ADR 0011's stated ground** — it refused structures *"that are fully derivable"* — so ADR 0011 is superseded in one scoped clause, on the terms ADR 0054 already took for enumeration. What buys the supersession is not convenience: ADR 0043 rules that *"an unwritten carrier is a defect, a reserved optional is a contract, and only a pinned meaning for `None` tells them apart"*, and `compile` writes a hardcoded `None` into `temporal` today. The field currently sits in ADR 0043's **defect** class. Pinning the meaning of absence moves it into the contract class, and that cannot be done from a stage that never runs.

### A Joinset takes the highest shape rank, and rank is row multiplication

An operand that carries no shape **abstains**. ADR 0030 already says absence means *"permanently shape-unclassified"*, and an unclassified operand is not a shape to disagree with — which is what keeps the star join legal.

Among the operands that carry one, the result is the **maximum by shape rank**, ordered by how much a shape multiplies rows for one entity: validity ranges only on change (`Scd`), one row per period (`Timeseries`, `Snapshot`), and individual occurrences without bound per period (`Events`). So `Events ⋈ Scd` is `Events` — the `Scd` side supplies context, not rows — while `Snapshot ⋈ Scd` is `Snapshot` and `Scd ⋈ Scd` is `Scd`.

**`Snapshot` outranks `Timeseries`, and the two genuinely tie on multiplication**, so the tie-break is stated rather than derived: a `Snapshot` row is a *state* while a `Timeseries` row is a measurement, so of the two only `Snapshot`'s label warns a consumer off a cross-period fold. The weakness is disclosed: additivity rides each Semantics and ADR 0030 makes rollup legality depend on shape **and** additivity, so a snapshot measure keeps its own guard whichever way this cell falls.

**The rule is a total order rather than a derivation matrix, and that is the load-bearing structural choice.** ADR 0050 makes a unit binary and ADR 0054 makes bracketing canonical left-deep, so the rule is applied *pairwise up a tree*. A maximum over a total order is associative and commutative for free, so the derived shape is invariant to bracketing and to which side the relationship was authored from. A free four-by-four table guarantees neither, and where it disagreed the shape would become a function of bracketing — which ADR 0054 established is *not* semantically load-bearing.

**Equal rank with different bodies yields an absent shape, not an error, where the Joinset states no shape.** Two fact tables on different clocks — `Events{order_date}` joined to `Events{ship_date}` — is an ordinary model, so a hard error would forbid it. The composed rows record time in two ways, so the operands give the surface no single classification to publish, and absence is already the pinned word for that. ADR 0053's fuse makes "the same axis" decidable structurally rather than by name comparison: two constituents naming one axis share one pooled Semantics and therefore one `EntityId`.

**A Joinset may state its own shape, and the stated shape must agree with the operands.** The stated shape is legal where it equals one of the shapes at the highest shape rank among the operands that carry a shape. It is also legal where no operand carries a shape. The Joinset then carries the stated shape. Any other stated shape is refused with `ComposedTemporalDisagreement`, the variant of rule A in *Three rules, three variants, two sites each*, which lists the operands' shapes and the stated one.

**In a tie, the stated shape agrees with an operand at the top rank, so compile still selects nothing.** `Events{order_date}` joined to `Events{ship_date}` may state `Events{order_date}`. It may not state `Events{date}`, which neither operand carries. Where no operand carries a shape, agreement is vacuous: the stated shape is the only source, as for a Unionset under interface-first.

None of this is ADR 0030's rejected *"majority/first-branch shape selection in composition"*: the rank is over the shape kind and independent of branch order, and the equal-rank case refuses to select at all. A shape that the author states in a tie is the author's selection, and compile only checks it.

### A Unionset agrees; a Grainset aligns on kind and differs on grain

The Unionset rule is already ratified in the ubiquitous language: *"Its children agree on temporal shape and grain, where agreeing includes both being absent, so the requirement says nothing about time on children that classify none."*

**The comparison is over the whole resolved `TemporalType`** — the variant tag, `scd_type`, and every identifying-dimension `EntityId` — and never over the grain. The variant tag alone is not enough: two branches both authoring `events`, one on `order_ts` and one on `created_ts`, agree on the tag while leaving the composed surface with no axis to publish, so compile would have to select one — the first-branch selection ADR 0030 rejects. A union of `Scd{Type1}` and `Scd{Type2}` fails for the same reason at a different field, `Type1` keeping no history for an as-of anchor to resolve into. The live roster is `{Type1, Type2}`; the earlier design's seven subtypes are retired text and no composition rule operates on them.

**An explicit Unionset may state its own shape, and it joins the unanimity set rather than overriding it.** ADR 0030's rejected *"allowing Complex kinds to declare or override shape"* is read as forbidding the **override** only, which reconciles it with two things already true: `ComplexExtras { temporal: Option<TemporalShape> }` is live and validated, with validation forbidding only `temporal.grain:` on it; and ADR 0026 cascades `extras.temporal.<shape>` to descendants, which ADR 0053 then scoped to interface-first. That scoping makes a parent declaration load-bearing rather than redundant: under interface-first the constituents author no shape at all, so the parent's is the only source, and the glossary's *"agreeing includes both being absent"* makes absent constituents agree vacuously. Under constituent-first the cascade is forbidden, each constituent states its own, and the parent's declaration is checked against them.

**A Grainset requires one shape kind and a distinct grain per constituent.** The one-kind requirement is **ADR 0030's own strict-equivalence rule, which survives here**: this ADR supersedes that clause for Joinsets alone, so a Grainset still requires equivalence across its children. Nothing enforces it — there is no kind-unanimity check in the model or in the manifest — so it is stage-reachable and ungated in ADR 0042's classification. The restriction is recorded as **provisional** — what the model needs now, not a claim that no Grainset could ever mix kinds — so a later relaxation is an extension rather than a reversal.

**One tempting justification for it is void, and is named so nobody reaches for it.** ADR 0039 argued that *"a Grainset's per-level keysets are byte-identical member sets, because the temporal shape kind is unanimous across levels"*, which reads as the temporal axis doing keyset work. **ADR 0048 supersedes that clause outright** — *"ADR 0039's three derivation clauses are all superseded … the Grainset per-level keyset … by loss of subject under ADR 0045's binary carrier and this ADR's rules"* — and 0048's replacement, a boundary that is the product over all grains with each element an intersection, never reads a temporal shape at all. So a Grainset's keysets follow from the keyset rules and from this data kind's own specification; the shared shape neither causes nor guarantees their agreement, and the one-kind rule stands on ADR 0030 alone.

Distinctness is **stronger than ADR 0030's "≥2 distinct grains"**, and the stronger rule is what the code already enforces: `check_grainset_levels` rejects equal coarseness between adjacent levels outright. ADR 0030's wording is the loose paraphrase and is corrected in this body.

**Coarsest-first ordering retires as a check rather than being re-homed.** `check_grainset_levels` also enforces strictly decreasing coarseness, and ADR 0045 retires `GrainsetLevel` and `RoutingUnitRef` outright, so the ordering stops being persisted. It is recovered from `Grain::coarseness()` over each constituent's carried grain, and ADR 0054 already sorts for itself — *"one implicit Joinset per ordered grain pair, finer to coarser"*. An order that is never persisted cannot be violated, which is the subtraction ADR 0050 made when structural check 8 turned out to compare a bijection against itself.

### The grain is derived where it is unanimous, so the grain ban binds the author only

ADR 0049 handed this decision an obligation rather than a question: `TemporalShape.grain` is *"forbidden on `ComplexDataKind`"*, so when a multi-source leaf is reduced and its public face becomes a synthesised Unionset, the grain the author wrote on the Dataset has nowhere to go — *"the reduction must not drop it silently, which it would today."*

Two facts make that fatal rather than untidy. ADR 0045 retires `GrainsetLevel`, so a constituent's **own** `temporal.grain` becomes the only grain carrier that exists. And a Grainset routes a request to the grain that serves it, so a constituent with no grain is a constituent nothing can route to.

The rule needs no exception for the reduction: **the grain is derived and carried where the constituents agree on one, and absent otherwise.** An implicit Unionset always satisfies it, because every branch is the same authored Dataset over a different physical source, so the reduction *carries over* what the author wrote instead of deriving anything new. An explicit Unionset satisfies it exactly when the agreement above holds. A Grainset never satisfies it, by its own distinctness rule.

**The ban on `temporal.grain` on a Complex kind is therefore re-scoped to authoring.** It binds what a human may write and says nothing about what compile may derive — the authored-versus-derived split ADR 0048 drew for keysets and ADR 0052 drew for cardinality. The rule does not change; its reach does.

This also closes a live hole wider than ADR 0049's instance. `GrainsetChildMissingGranularity` is raised from a walk over `NestedDataset` only, while `GRAINSET_ALLOWED_CHILDREN` is `[Dataset, Unionset, Joinset]` — so a Grainset child that is a Unionset can never satisfy it and nothing looks for the failure.

### Absence has one meaning per field

Absence has real producers on both fields. `temporal` is empty for a leaf whose author wrote none, and for a Joinset that states no shape and whose operands all abstained or tied at rank with different bodies. `grain` is empty wherever the constituents do not agree on one, which for a Grainset is always.

One meaning rather than a discriminant, because every consumer asks the field the same question — may I roll up, may I anchor as-of — and all of these causes answer it identically. A reason field would carry a fact nobody branches on, and ADR 0040's derived-values-unmarked counter-precedent and ADR 0050's `Auto`-leaves-no-trace both point the same way.

**The cost is stated:** an author who forgot `temporal:` on a leaf and a Joinset that legitimately has no time axis produce the same artifact. The diagnostic for the forgotten case therefore has to live at authoring time, or not exist; the manifest cannot supply it.

### Three rules, three variants, two sites each

| rule | predicate |
| --- | --- |
| A | **Unionset** — full shape unanimity (kind body *and* grain) over the constituents together with any shape the Unionset itself states |
| B | **Grainset** — kind-body unanimity over the constituents together with any shape the Grainset itself states |
| C | **Grainset** — pairwise grain distinctness over the constituents |

A Joinset raises nothing where it states no shape: the rank rule always yields either a shape or absence. A shape that a Joinset states is checked as *A Joinset takes the highest shape rank, and rank is row multiplication* says, with rule A's variant and at the same two sites.

For a Unionset and a Grainset, an authored parent shape needs no rule of its own — it is one more participant in the same unanimity set, which follows from reading ADR 0030's prohibition as override-only rather than being a convenience.

**Each rule is one dedicated `CompileError` variant carrying its discriminating values** — the disagreeing constituent ids with their shapes for A and B, the duplicated grain with both constituent ids for C — on the payload pattern the `RelationshipJoinKey*` variants set and ADR 0052 followed. Not a reused `reason: String`: ADR 0044 ordered exactly this split, and ADR 0052 applied it at a neighbouring call site.

**Each rule is owned by the stage that assembles `semantic_sources` and restated in `validate()` — one rule, one variant, two sites.** ADR 0042 already makes `validate()` raise `CompileError`, so no second enum and no second identity is involved. For a shape that a Joinset states, the site in `validate()` is the same derivation run over the stored artifact, where the stored shape plays the stated one. The restatement is owed on ADR 0052's own test: `temporal` is `pub`, a decoded artifact can desync what no producer could, and the values are load-bearing for Grainset routing and for rollup legality. They are **not** load-bearing for keysets, although ADR 0039 once said they were: ADR 0048 superseded that clause, and its boundary arithmetic reads grains rather than shapes. `check_grainset_levels` is not merely a precedent for two of the three — it is that restatement, already written in the load validator against a type ADR 0045 removes.

This ADR fixes a **third ordering obligation** beside the two already recorded: the derivation runs bottom-up over the operand edges, after every constituent's own shape exists, and after ADR 0054's enumeration, since an implicit unit needs a shape too.

### Amends

- **0030:** the strict-equivalence clause is superseded for Joinsets only. It stands unchanged for Unionset and Grainset. ADR 0030's operative content otherwise survives whole: no inference on a leaf, from column names or by default; composed shape derived from children; disagreement a hard error; the `MixedShapeAdvisoryChildren` advisory retired. Two ADR 0030 clauses are corrected in 0056's *A Unionset agrees; a Grainset aligns on kind and differs on grain* rather than superseded, on the precedent ADR 0052 set for ADR 0038's deferred-cardinality paragraph. *"TemporalShape is declarable only on leaf `SimpleDataKind`s"* was already contradicted before 0056 — by ADR 0026's cascade, by ADR 0053's scoping of it, and by live `ComplexExtras.temporal` — and is corrected so that a Complex kind may declare a shape that agrees (only an override is forbidden). *"Grainsets additionally require ≥2 distinct grains"* is corrected to pairwise distinctness, which is what `check_grainset_levels` enforces.
- **0011:** composed temporal shape is derived at compile and persisted; the derived-structures clause is superseded for this one field, on the terms ADR 0054 took for enumeration; the model-wide relationship adjacency index and every other structure ADR 0011 lists stay forbidden.
- **0049:** the grain is derived and carried where the constituents agree on one, and absent otherwise. An implicit Unionset always satisfies it, because every branch is the same authored Dataset over a different physical source, so the reduction *carries over* what the author wrote instead of deriving anything new. The ban on `temporal.grain` on a Complex kind is therefore re-scoped to authoring.

### Consequences

- **`CONTEXT.md` gains `Shape rank`, and two existing entries are corrected in the same change** so the glossary never states a rule this artifact has not ruled. **Cascade**'s *"derived upward from unanimous children"* is false for a Joinset, and both **Cascade** and **Temporal shape** use "composer" — a word the **Complex DataKind** entry's own *avoid* list forbids. **Unionset** gains the clause that it may state the agreed shape itself, and **Grainset** gains distinctness per constituent and its one shape kind.
- **What this obliges in `crates/`** — recorded for the implementation effort, not done here:
  - `compile` passes a hardcoded `None` for `temporal` and nothing populates it — the live silent drop the primitives inventory reported first. This ruling closes it; the wiring is not ruled here.
  - The manifest's module doc for its data kinds asserted *"complex kinds derive their effective shape at graph build"* as live design. That sentence is what this ADR reverses, and it would read as authority until it is cut.
  - `check_grainset_levels` is **deleted rather than edited**: its subject `GrainsetLevel` retires with ADR 0045. Its distinctness rule re-homes onto the constituents; its coarsest-first rule does not; its `len >= 2` becomes the Grainset's own constituent count.
  - `GrainsetChildMissingGranularity` is raised from a `NestedDataset` walk and cannot see a Unionset or Joinset child. Under the derived grain it needs re-cutting or retiring, not extending.
  - There is **no graph-build derivation to delete**: `crates/semstrait-ir/src/semantic_graph.rs` has zero occurrences of `temporal`.
  - Kind-unanimity across a Grainset's levels is **ungated today**: it is ADR 0030's requirement and no check exists in the model or the manifest. Do not restore ADR 0039's keyset justification for it — ADR 0048 superseded that clause, so the rule stands on ADR 0030 alone.
  - The manifest's `TemporalShape` already carries `grain: Option<Grain>` and resolves each axis to an `EntityId`, so no type change is owed for the shape itself. What changes is which semantic sources carry one.

## More Information

### Open questions

- **Which of the named compile stages the stage that assembles `semantic_sources` is stays unruled here.** ADR 0048 and ADR 0050 say *"the stage that assembles `semantic_sources`"*, ADR 0054 names an *enumerating* stage, and no ruling has yet said whether these are one stage or three. That roster is owed to the compile control-flow question and is not resolved by minting a fourth name. This is the fourth ruling to name a stage without fixing which stage it is.
