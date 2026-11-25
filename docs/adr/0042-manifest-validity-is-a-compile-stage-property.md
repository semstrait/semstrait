---
number: 42
title: '`SemanticManifest` validity is a compile-stage property; `semstrait-manifest` has no load boundary'
status: accepted
date: 2026-08-12
source:
- crates/semstrait-manifest/src/validate.rs
- crates/semstrait-manifest/src/compile/mod.rs
- crates/semstrait-manifest/src/manifest/keyset.rs
- crates/semstrait-manifest/src/repository.rs
- crates/semstrait-common/src/io/repository.rs
links:
- target: 12
  kind: amends
- target: 60
  kind: amendedby
- target: 73
  kind: amendedby
---

# `SemanticManifest` validity is a compile-stage property; `semstrait-manifest` has no load boundary

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0060](0060-a-scope-qualified-name-holds-one-entity.md): the identical-membership defect that 0042's *Decision Outcome* cites no longer exists, and it was removed by ADR 0048's carrier change rather than by choosing a stage for it: ADR 0040's load check 8 retires by loss of subject, and the same-membership rule keeps one home, the model's.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): the pipeline is parse, validate, compile, plan, adapt, not `parse → compile → resolve → plan`.

## Context and Problem Statement

**The boundary ADR 0012 named no longer exists, and would be wrong here even if it did.** That ADR ratified "`Repository::load` runs one fail-fast pass that re-checks every cross-reference and every derived value", with "the load guarantee is contractual for all consumers". Two independent facts retire the mechanism. First it became unimplementable: `Repository` was generified out of this crate into `semstrait_common::io`, parameterized over `Persistable`, so `Repository::load` sits in a crate that cannot see `SemanticManifest::validate` — and the workspace's acyclic crate dependency graph forbids the `common → semstrait-manifest` edge that would let it. Second, and independent of the move, the instruction places an I/O-bound obligation inside a crate whose remit is compile and shape.

This decision is recorded because it moves a ratified obligation out of the crate that owns the checker, retires a public error type, and reads backwards on first contact: the crate holding every integrity check deliberately never runs them on the way in, and the word "load" is struck from its vocabulary.

## Considered Options

- Validity owned by compile's stages, with `validate()` exposing their gates, raising `CompileError`, accumulating by default with a fail-fast flag and suppressing nothing; enforcement is `Result` (chosen).
- **Overriding `Persistable::integrity_check` on `SemanticManifest`.** The cheapest option, with `save`/`load` symmetry for free since the seam calls it on both. Refused because its error channel is a `Cow<'static, str>` folded into `IntegrityCheckFailed`, which flattens the variant identity ADR 0004 makes the published semver contract, and because it re-purposes a hook whose only established use is a single-field self-invariant — the model's root-id presence.
- **A manifest-owned validating repository or loader, with the bare seam withheld from consumers.** Honours ADR 0012's wording without touching `common`, and would make "contractual for all consumers" true by construction. Refused because it puts an I/O-bound load path inside a crate whose remit is compile and shape.
- **A `ValidatedManifest` typestate making an unvalidated manifest unrepresentable downstream.** The only candidate with a compile-time guarantee; refused as consumer policing from the wrong crate, the refusal ADR 0003 records for `ValidatedModel`.
- **A whole-artifact re-validation pass at the end of `compile()`.** Re-proves what the stage gates established, and pays a walk over every reference to catch only a semstrait bug — never a malformed input, which fails earlier with a located `CompileError`.
- **Keeping the fail-fast pass only.** Yields one located root cause per run with no cascade, but the consumer is judging a corrupt artifact whole rather than fixing it field by field, so it needs the entire report.
- **Cascade suppression through a check-dependency graph.** Minimal reports, at the cost of new structure nothing has today, protecting an edit-and-retry loop that corruption diagnostics do not have.
- **Leaving the 0012-versus-0017 collision unruled and validating at both the seam and the consumer.** Keeps both ADRs literally satisfied and is the reading the deleted contract tree invited. Refused because the seam half is unimplementable across the crate dependency boundary, so "both" collapses to "one" in practice while reading as a stronger guarantee than it is.

## Decision Outcome

`semstrait-manifest` owns `compile()` and a pure, I/O-free validator over the manifest's shape and entities. It owns no load path, no validating repository, and no hook on the persistence seam. Validity is established *inside* compile, by each stage's own gates: a stage's output is valid for that stage by definition, because the gate is part of the stage rather than a check bolted after it. `validate()` is that same conjunction of gates exposed as a standalone entry point, for a manifest that arrives with no stage history: one deserialized from storage. It raises `CompileError`, not a load-flavoured kind, because it is not a second stage; it accumulates by default with a fail-fast flag; it suppresses no cascades; and enforcement is `Result` — a `SemanticManifest` in an `Ok` arm is valid by construction, and anything else is an `Err` that breaks the pipeline the higher-level API composes from `parse → compile → resolve → plan`. This supersedes ADR 0012's *mechanism* while upholding its *guarantee*, and upholds ADR 0017's hookless seam.

The guarantee survives the mechanism's retirement and is relocated above the crate, which is what actually has a dependent: ADR 0040 justifies re-minting the keyset id at check time on the ground that "ADR 0012's load guarantee re-derives every derived value", and that argument is untouched.

**ADR 0017 is upheld, and had already drawn the line this ruling needs.** It describes the seam as "a fail-fast, id-keyed, whole-object seam with no diagnostics channel, no listing and no validation hook", and lists "one unified load path returning `(T, Diagnostics)` with a repository-injected `validate` hook" among its rejected alternatives. `Persistable::integrity_check` — a DTO-owned self-check the seam calls on both `save` and `load` — shipped on 2026-08-03, one day before ADR 0017 was written, so "no validation hook" was authored with it already in the tree and cannot mean "no self-check". The line it draws is *injected* versus *DTO-owned*, and the seam states the consequence at the site: "Shallow id-vs-key agreement, the one integrity check the seam runs… Deep validation is the consumer's." So the collision between 0012 and 0017 — the only kind of live contradiction left — resolves in 0017's favour, and 0017 needs no supersession.

**Validity is constitutive of a stage, not verified after it.** This is the load-bearing distinction and the one most easily stated backwards. `compile()` does validate its output — continuously, at every stage — and it does not perform a separate whole-artifact pass at the end, because such a pass would re-prove what the stages already established. The operative consequence is a definition rather than an exhortation: **`validate()`'s check set is the conjunction of every stage's validity definition, plus the rules no single stage can assert.** It therefore cannot be legitimately stricter than the producer. Where `validate()` holds a rule that a stage could have asserted and did not, that stage's definition of validity is incomplete, and the gap is a defect rather than an accepted asymmetry. One instance is verified: ADR 0040's per-DataKind identical-membership rule is grouped exactly as `mint_keysets` is scoped, so the rule is within that stage's reach, yet two sites naming the same member set under different names mint two ids and compile emits an artifact its own checker rejects. Which stage closes it is not decided here — it belongs with the duplicate-interface-site question, which stays open.

**The producer-unreachable checks are legitimate, and the code already identifies them.** They are not stage gates and never could be: `Keyset::new` mints the id rather than accepting it, so "the constructor path cannot produce an id-inconsistent keyset — which is exactly what load check 6 catches", and it takes `data_kind` so "the constructor cannot build the orphan that check rejects". Both quotations are the constructor's own doc comment. They exist because `name` and `members` are `pub`, so field assignment on a decoded artifact can desync what no constructor could. Global id uniqueness across collections is the same shape by a different route — a whole-artifact property that no single stage is positioned to assert. Whether any *remaining* check is redundant with a stage gate is settled per check, not by a blanket rule.

**`validate()` raises `CompileError`, and `LoadError` retires.** `validate()` is the compile stage's gates applied to an input, not a stage of its own, so ADR 0004's per-stage diagnostic identity yields one kind rather than two. `LoadError`'s two variants — `DanglingReference` and `StructuralViolation` — fold into `CompileError`, and `LoadError` is deleted. Three things follow. The duplicate-implementation problem dissolves: today `mint_keysets` raises `CompileError::KeysetMemberNotPooled` / `KeysetMemberNotADimension` while the keyset checks raise `LoadError` variants over the same two facts, which is one rule with two vocabularies kept in step by hand. `CompileError` already implements `Diagnose`, already carries warning-severity variants, and already flows through `split_by_severity`, so the accumulating envelope needs nothing built — whereas `LoadError` has no `Diagnose` impl at all and would have needed one. And the word "load" leaves the crate: `load.rs`, its "Load-time integrity validator" module doc, and `LoadError` all name an action this crate does not perform, while what remains of that vocabulary already has a home in `RepositoryErrorKind::{Io, EncodingMismatch, IntegrityCheckFailed}`.

**Accumulate by default, fail fast by flag, suppress nothing.** The shape is ADR 0005's accumulating pair, `Ok(Diagnostics<CompileError>)` / `Err(Diagnostics<CompileError>)`, superseding ADR 0012's "one fail-fast pass". The flag is a short-circuit, not a second return class: fail-fast returns the same type carrying one element, so the stage class never flips and ADR 0005's warning that "flipping a stage's class is an API break for every caller" does not fire. No check may assume another ran, and none is skipped because a precondition failed. The reason is what these faults *are* — corruption of a compiled artifact, not author error — so there is no edit-and-retry loop for cascade suppression to protect, and completeness is worth more than minimality to a consumer deciding accept-or-reject. The code had already made this call at exactly this site, re-establishing its own precondition and noting that "reporting beats skipping, since an `if let` would let the refinement silently *pass*". Suppression would additionally require a check-dependency graph that nothing today has. The payoff is that the traversal becomes order-**independent**: the fixed order survives only so the report is stably ordered, which is a far weaker property to keep true than a chain of checks leaning on each other.

**Enforcement is `Result`, not a type.** Compile's `Ok` arm *is* the proof of validity, because every stage gated its own output; a manifest that cannot be made valid never leaves compile in that arm. Symmetrically, `validate()` over a deserialized manifest returns the error arm and breaks the same pipeline. No `ValidatedManifest` typestate is introduced: it is the only option offering a compile-time guarantee, and it is refused because it would have this crate dictate every consumer's signature in order to police code it does not own — the same refusal ADR 0003 records for the sibling type, listing "no `ValidatedModel` typestate proving validation ran" among the design's deliberate oddities.

### Amends

- **0012:** supersedes the load mechanism and upholds its guarantee, which is relocated above the crate: `semstrait-manifest` has no load path, and `validate()` applies compile's gates to a decoded manifest, raising `CompileError` and accumulating by default with a fail-fast flag; a caller can reach past the composing layer to the generic `Repository<SemanticManifest>`, deserialize, and consume the result unvalidated — nothing prevents it.

### Consequences

**Two accepted costs.** A caller can reach past the composing layer to the generic `Repository<SemanticManifest>`, deserialize, and consume the result unvalidated; nothing prevents it. That is the identical residual ADR 0038 accepted for the model — "a consumer that decodes an authoring model without running validation now accepts a document the full pipeline refuses" — and it is the posture, not an oversight. Second, this decision does **not** make the checker reachable. `SemanticManifest::validate` has zero non-test call sites, no crate in the workspace references `SemanticManifest` at all, and the would-be composing layers (`semstrait-api` and the `semstrait` facade) are both `exclude`d from the workspace with the facade still on the legacy `CompiledManifest`. The call site this ruling names lives above the crate, so it lands when that layer is rebuilt and not before. ADR 0040 disclosed the same gap for its nine checks; a reader should not mistake "validated on the way in" for "validated on the way in today".
