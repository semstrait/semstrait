---
number: 52
title: 'Compile gates the cardinality it materialises: one invariant survives resolution, and ADR 0038''s accepted gap closes'
status: accepted
date: 2026-08-25
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0004-typed-diagnostic-kinds-no-error-codes.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
links:
- target: 38
  kind: amends
- target: 54
  kind: amendedby
- target: 66
  kind: amendedby
---

# Compile gates the cardinality it materialises: one invariant survives resolution, and ADR 0038's accepted gap closes

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): `integrity` alone decides which sub-compositions are legal, so the `(cardinality, integrity, preserved side)` triple is not what prunes implicit-Joinset enumeration.
> - [ADR 0066](0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md): its statement that ADR 0038's absent-cardinality row stands untouched is narrowed. The row still resolves cardinality to `ManyToMany`, and ADR 0052's cardinality gates stand; only its derived join type changes from `Inner` to cross.

## Context and Problem Statement

ADR 0038 accepted one cardinality gap and deferred the question of where cardinality's post-resolution *invariants* live.

This ADR exists to withdraw that ratified acceptance and to put a forward pointer on a live deferral. It deliberately does **not** state where the rules live, because three merged ADRs already do — ADR 0042 makes validity constitutive of a stage, ADR 0044 puts each stage's validity definition in that stage's own module documentation, and ADR 0050 ratifies the relationship-resolution stage together with its `CompileError::RelationshipResolution`. Restating that here would be the second-place-to-look failure the precedence ruling removed 46,284 lines to eliminate. What is recorded is only what does not follow: which invariants survive resolution, that the gap closes, and the two diagnostics.

## Considered Options

- Compile gates the cardinality it materialises, with two dedicated variants (chosen).
- **A validate-stage close, keying on the defaulted cardinality.** The strongest alternative, and genuinely available now in a way it was not when ADR 0038 wrote that such a rule "would be a guess about a compile-stage fact" — the matrix row is ratified text in 0038 itself, so reading it is applying a published rule rather than guessing. Refused because it leaves compile fabricating the bad tuple for a model that skipped validation, which ADR 0003 permits by construction, and because it would put the manifest stage's defaulting rule inside `semstrait-model`, where it can drift from the matrix that owns it.
- **Reopening the absent-cardinality matrix row** so a directional `cross_filter` forbids omitting cardinality at all. Fixes the pair at source, which is ordinarily preferable. Refused because the incoherence is not in the authoring: ADR 0038 makes `cross` plus `filter:` the sanctioned non-equi join, and on a theta condition directional propagation genuinely is coherent. The contradiction appears only when the row promotes cardinality to `ManyToMany`, so it is a fact about the resolved artifact, and re-deriving a ratified authoring grammar to fix it would be backwards.
- **Permanent acceptance, restated with a fresh reason.** The honest option had the deferral's ground survived. It did not: all three facts it rested on are void, and the blast radius has since reached keysets.
- **A provenance flag distinguishing a matrix-supplied companion value from an authored one**, which would give the symmetric-completeness rule a post-resolution form. Refused on ADR 0040's standing counter-precedent that this codebase stores derived values without marking them, and on ADR 0050's `Auto` leaving no trace.
- **One variant carrying both faults in a `reason` string.** Cheapest, and refused by ADR 0044's own recommendation to split exactly this overload at exactly this call site.

## Decision Outcome

The stage that mints a relationship's resolved `(cardinality, optional, cross_filter)` tuple is the stage that gates it. Two rules land there, and **ADR 0038's accepted gap closes as a consequence rather than by a separate decision**: a relationship with `cardinality` absent and `cross_filter: left` authored no longer reaches a consumer. The deferral is discharged with no remainder.

### The deferral's plural is a singular

Of the three candidates the deferral named, exactly **one** is a post-resolution invariant at all, and establishing that is most of this decision.

**"The resolved cardinality is consistent with the basis kind" is not expressible post-resolution.** ADR 0038 keeps all four cardinalities legal on `on: cross`, so `ManyToMany` beside a `cross` tautology and `ManyToMany` beside an equi-condition are both legal artifacts. What the rule keys on is the *absence* of an authored value, and absence is exactly what the matrix consumes. The artifact does witness the basis kind — ADR 0050 pins `cross` as `Eq(Literal 1, Literal 1)` and distinguishes it from a degenerate equi-term at the leaf kind — but the basis kind is not what this rule needs. It is checkable only at the moment of resolution, where the authored `Option` and the authored `on` are both still in hand, so it lands as a *precondition* on the resolving stage, not as an invariant on its output.

**"The symmetric-cardinality completeness rule against the resolved value" is vacuous.** `Optional::None` and `CrossFilter::None` are enum variants, not absence, so post-resolution "the value is present" is identically true. The rule is about authoring *completeness*, unobservable once the matrix has materialised a value. The only non-vacuous reading — that a symmetric cardinality's companions must have come from the author rather than the matrix — requires the artifact to witness provenance, which ADR 0040 refuses in terms: *"the derived-and-stored `join_type` is the standing counter-precedent: this codebase already stores derived values without marking them."* ADR 0050's `Auto` leaving no trace is the same refusal one field over. And ADR 0038 had already carved the exemption explicitly: a bare `on: cross` is complete, while an *explicitly* symmetric cardinality still owes its companions. So this candidate dissolves against ratified text, not against a preference.

**"The many-to-many cross-filter restriction re-tested after the matrix resolves" is the one that survives**, and it is the only non-vacuous constraint on the persisted tuple. Every other resolved pair is reachable from some legal authoring, because an authored `optional` or `cross_filter` overrides its matrix default and no rule ties `integrity` to `cardinality` — `integrity` carries no validation rule at all, and ADR 0038 makes it orthogonal on the principle that absence is never evidence.

### The gap closes because its ground expired, not because the trade-off was re-weighed

ADR 0038 accepted the gap on one stated ground: *"Inventing a validate-stage rule for it would be a guess about a compile-stage fact."* That ground was true and is now void, together with the three facts the deferral handed forward.

**The compile-stage home is no longer unreachable.** The deferral's first fact — relationship resolution is dormant, every relationship function in the seal pass dead-code-allowed — described the code, and ADR 0050 resolves those sites *by deletion rather than by wiring*, ratifying a relationship-resolution stage with its own module documentation and its own error variant. The dormant-lane cost the deferral asked a successor to weigh is not a cost this decision pays.

**The resolved value now persists.** ADR 0050 puts `join_type`, `cardinality`, `integrity` and `cross_filter` on the produced joinset's `role_definition`. When the deferral was written, the resolved tuple had no durable carrier at all.

**The resolved value now has a compile-stage consumer.** ADR 0048 gates composite-keyset derivation on `(cardinality, integrity, preserved side)`, the same triple that prunes implicit-Joinset enumeration. A wrong resolved cardinality therefore yields a wrong *keyset*, not only a wrong plan — the blast radius grew after the deferral was written.

The retired structural-rules leaf, carried as evidence rather than obligation, records why the surviving rule exists: *"A many-to-many relationship has no natural 'one' side; directional filter propagation is ambiguous."* The pair the matrix can mint is not merely unconventional; it is the pair the rule was written to forbid.

**Where the deferral's label lives, since it is now a retired citation.** The deferral was tracked in a *Technical Debt Registry*, deleted with the design tree on 2026-08-12 — and the project's rules forbid recreating it by name, "no debt registry". So the label survives only in ADR 0038's Cardinality section; it never reached `crates/` as a marker token. **"Discharged" therefore means what this ADR says and nothing else — there is no registry row to close, and none may be created.**

**The retired entry corroborates this ruling, reached without it.** Retrieved as evidence, not obligation: its heading is "cardinality's post-resolution invariants have no compile-stage home", its Location list names both sites ruled on here, and its two Problem bullets are exactly ADR 0038's two facts — which makes *no remainder* checkable rather than asserted. Its Remediation names this decision's option as the preferred one, *"re-check the basis at `resolve_defaults` and re-run the validate-stage basis rule against the resolved pair — the honest reading, and the one that closes both bullets with one rule"*, and it **forbids the alternative rejected above** in terms: *"Do not close this by widening the validate-stage basis rule at validate: it would have to predict the matrix, which is the duplication ADR 0038 declines elsewhere for the same reason."* Two departures are stated. The entry advised settling this *"when relationship lowering lands, in the same session, because that is when the lane becomes reachable"* — ADR 0050 removed that unreachability, and this decision is ruled on paper by charter, so the timing premise expired with the rest. And where the entry says *one rule*, this decision lands **two predicates and two variants**: the entry means one change, and the split is what ADR 0044 requires.

**The counter-precedent is named.** Twice the record cites wrong cardinality as an accepted, unguarded tolerance — ADR 0038 on derived-basis drift and ADR 0040 on derived-roster drift, both resting on *"a wrong `cardinality` already yields inflated sums with no warning."* That does not bind here, and the distinction is the whole of it: both tolerate a wrong value **the author owns** going undetected, whereas this decision concerns a value **compile fabricates** and a pair a ratified rule forbids. Undetected drift and manufactured contradiction are different faults, and only the first is tolerated.

### Two rules, and only one of them can reach the artifact

**The invariant** — the resolved `(cardinality, cross_filter)` pair is never `ManyToMany` with a directional `cross_filter` — is owned by the relationship-resolution stage **and** restated in `validate()`, for reasons that do not overlap. At compile, because `compile()` takes a plain `SemanticModel` and ADR 0003 refuses a `ValidatedModel` typestate, listing "no `ValidatedModel` typestate proving validation ran" among the design's deliberate oddities — so the stage cannot lean on validation having run. In `validate()`, because `role_definition`'s fields are public and field assignment on a decoded artifact can desync what no producer could, which is ADR 0042's own stated reason for its producer-unreachable class. This is not redundancy across vocabularies: ADR 0042 already makes `validate()` raise `CompileError`, so it is one rule, one variant, two sites.

**The precondition** — an absent authored `cardinality` is legal only under a `cross` basis — is **compile-only, and necessarily so**. Its input is authored absence, and absence does not reach the artifact, so `validate()` cannot host it even in principle. It is the same rule the model's validation walk already applies to the authored document; ADR 0004 makes diagnostic identity per-stage, so this is a restatement at the stage that reaches the fabricated value, not a borrowing.

The line that makes these two rules compile's rather than a general licence to re-validate the model: **compile owns a rule when the alternative is fabricating a value the author never wrote and no consumer can distinguish from an authored one.** That is ADR 0050's own argument for why a failed `Auto` may never emit the `cross` tautology, applied one field over. A rule whose violation makes compile *fail* needs no restatement here; a rule whose violation makes compile *invent* does.

**One ordering obligation follows.** ADR 0048's keyset derivation reads the resolved tuple, so the stage that gates it must run before the stage that consumes it. Which stages those are is not fixed here: ADR 0050 names "the relationship-resolution stage" and ADR 0048 "the stage that assembles `semantic_sources`", and no ruling has yet fixed what compile's stages are. That roster is owed elsewhere; this ADR fixes only the dependency between the gate and its consumer, which is the part a stage roster could not later contradict without breaking a keyset.

### Two dedicated variants, not a reused reason string

ADR 0044 already ordered this in its recommendations: *"Split `TypeInferenceFailure`'s overload into dedicated variants for the faults its own documentation flags as awaiting ratification."* It named the defect **diagnostic overloading** — a reused fail-fast shape standing in "rather than a dedicated variant". `compile/seal.rs`'s `relationship_resolution_error` is one of the flagged sites, so folding two invariant violations into `RelationshipResolution`'s `reason` string would re-create inside the new variant the overload 0044 ordered split. `CompileError::RelationshipResolution` stays scoped to what ADR 0050 ratified it for: a derivation that found nothing.

Both variants take the name their validate-stage counterpart already carries, because the same name in two enums is already two identities under ADR 0004 — the precedent ADR 0038 set when the missing-cardinality rule moved stages, "restated rather than borrowed".

- `CompileError::RelationshipMissingCardinality { relationship }` — the precondition. Identical predicate to the validate rule over identical inputs, so an identical name is the honest one.
- `CompileError::RelationshipManyToManyCrossFilterDirectional { relationship, cardinality, cross_filter }` — the invariant, **carrying the resolved pair**. The payload is what stops a reader assuming the same inputs as the validate rule, and it follows the four `RelationshipJoinKey*` variants' own pattern of carrying the discriminating value: id-mismatch carries both ids, unknown-on-side must name whether `from` or `to`. The resolved-ness rides the diagnostic, not only the module doc.

Per ADR 0044 both are stated in their raising stage's module documentation and join no roster.

### Amends

- **0038:** its *Open questions* paragraph on cardinality — the open question — is **discharged**. Three clauses of it are now wrong rather than merely superseded, and are corrected in 0052's *Amends* rather than edited in 0038, on the precedent ADR 0040 set when it corrected ADR 0036's closing sentence in its own body: *"A compile-stage home is unreachable today"* is void, since ADR 0050 ratifies the stage; *"one gap is accepted rather than patched"* is **withdrawn**, and the gap closes; *"The real question is where cardinality's post-resolution invariants live"* is answered, and the plural is a singular. **0038's operative content stands untouched**: the `on:` grammar, cardinality's conditionality on the basis, the matrix extension and its absent-cardinality row, the move of the missing-cardinality rule to validate, and the keyset evidence rule. In particular the matrix row keeps its `ManyToMany` and keeps allowing an authored value to override a default. 0052 catches a bad resolved pair; it does not re-derive the row that can produce one.

`CONTEXT.md`'s **Cardinality** entry is narrowed in the same change: "never inferred or verified" overreaches once the resolved pair is checked, so "never inferred" stands and the verification clause is scoped to *never verified against data*. The glossary already drew the distinction this decision turns on — "do not read the omitted case as an assertion — it is the absence of one" — and gains no new term.

### Consequences

The invariant is gated at two sites and the precondition at one, so a bad resolved pair is unrepresentable in a manifest that left compile through the `Ok` arm — which is ADR 0042's enforcement posture, not a new one. The cost is that compile now restates two rules the model stage already applies, and the boundary against restating *every* such rule is the fabrication line stated above rather than a list.

`crates/` convergence is out of scope for this decision and is handed to the implementation effort:

- **`resolve_defaults` cannot check the precondition as it stands** — its signature is `(cardinality, integrity, authored_optional, authored_cross_filter)` and it never receives `on`. The basis must reach it, or the check must sit at its caller. This is the one handoff without which the ruling is unimplementable.
- The `None` arm's `authored_cross_filter.unwrap_or(CrossFilter::None)` is what preserves the authored directional value beside the minted `ManyToMany`; it is correct under the matrix row and is the site the invariant guards.
- The deferral's open-question marker on that arm is discharged; where one remains, it should cite this ADR when the arm is next touched.
- Two `CompileError` variants are owed, and `relationship_resolution_error`'s FLAG asking for ratification is partly answered: ADR 0050 ratified the derivation-failure variant, and these two are the remainder of the split ADR 0044 ordered.
- The gate must run before ADR 0048's keyset derivation reads the tuple.
- Both rules need a test each, per ADR 0044.
