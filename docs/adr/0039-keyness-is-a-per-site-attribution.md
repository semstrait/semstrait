---
number: 39
title: Keyness is a per-site attribution on a dimension site, not a Semantics
status: accepted
date: 2026-08-09
source:
- crates/semstrait-model/src/model/types/entities/keyness.rs
- crates/semstrait-model/src/model/types/entities/dimension.rs
- crates/semstrait-manifest/src/compile/keysets.rs
links:
- target: 48
  kind: amendedby
- target: 53
  kind: amendedby
- target: 27
  kind: amends
---

# Keyness is a per-site attribution on a dimension site, not a Semantics

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0048](0048-composite-keysets-are-authored-inside-a-boundary.md): the three derivation clauses are all superseded: the Unionset intersection by rule, the Grainset per-level keyset and the Joinset anchor propagation by loss of subject under ADR 0045's binary carrier and ADR 0048's rules. An authored composite keyset is checked against a boundary. The posture *a keyset advises; it never constrains* is narrowed, not kept whole, and *"A relationship resolves its join basis from its own `on:` specification and never from keyness"* is re-scoped, as ADR 0048's *A keyset constrains the semantic side, and never the data* gives.
> - [ADR 0053](0053-authoring-direction-is-exclusive-per-root-data-kind.md): a constituent-first inline body carries a full interface, and `NestedDataKindKeynessIgnored` retires, because keyness on a child stops being an ignored key. Interface-first keeps today's shape, per-site `is_key:` on the root's interface; constituent-first derives the keysets, and a constituent-first root may not author `is_key:` anywhere, since that would be the mix.

## Context and Problem Statement

Keyness was declared in a second place, a per-DataKind `keys:` block apart from the `dimensions:` list, and `Key` was a Semantics of its own. The question is where keyness lives instead, what a keyset promises, and which stage owns each rule.

## Considered Options

- Per-site keyness: `is_key:` on the dimension site, with `keyset:` / `keysets:` naming membership under cover arity (chosen).
- **A DataKind-level keyset block** — `keysets: {platform: [a, b]}` on the interface. Better legibility in one place, and it is what the retired `keys:` block already was. It loses site locality, which is the entire point: the author would again declare a dimension in one place and repeat its name in another, and adding a dimension would again silently leave the block stale.
- **Keyness on the pooled `Dimension`.** One field, no type split, and it reads as the obvious home. It makes keyness model-global, so a claim made by one DataKind would apply to every other DataKind referencing that dimension — the shape that *would* breach ADR 0027's unification rule, and the reason the split exists.
- **A standalone ADR for the local-only-field boundary rule.** The rule is one paragraph and its only current instance is this decision; a separate document would have to be read alongside this one to mean anything, and would invite drift between the rule and its motivating case.
- **Leaking the two-body split into YAML as nesting** — `keyness: {is_key: true}` on the site. Mirrors the internal type structure honestly, but costs the author a level of indentation for a distinction that exists only inside the crate. The wire form stays flat: `is_key:` beside `ref:` in the same mapping.
- **Partition arity — one keyset per site.** Simpler to reason about and a smaller surface. It is a strict expressiveness regression against something already ratified and implemented: the retired `unique:` list already permitted named, multi-column, freely overlapping key sets, and named overlapping key sets exist in production in the OLAP-cube lineage. It would also force an author with a tenancy column to pick one keyset and state a falsehood about the others.
- **Unnamed keysets, with the names living on the relationship.** Removes the naming ceremony entirely. It puts identity advice on the edge rather than on the entity that has it, and leaves a DataKind unable to state what identifies its rows unless some relationship happens to join on them.
- **Collapsing two identical-membership keysets silently.** Cheaper than an error and never blocks an author. It discards a name the author chose deliberately, and the two names may mean different things to a consumer even where their members coincide today.
- **A DataKind-level unknown-key gate instead of the warning.** Rejecting an unknown DataKind key by name, as each dimension site's body already does, would make a stale `keys:` block a hard error. It cannot be spelled with `deny_unknown_fields` while the wire mirror flattens its body, so it means hand-rolling four more decoders; and it would promote *every* tolerated DataKind-level key to an error at once, a surface change well outside a keyness slice. Left as an available follow-up: the warning names the key either way.
- **Reading `keysets: []` as `default` membership.** It is what the accessor would do unguarded, and it never blocks an author. It materialises membership from an empty basis — exactly the accident ADR 0038 refuses for `on: []` — and the author who meant `default` already has a shorter spelling.

## Decision Outcome

A dimension **site** — the entry under a DataKind's `dimensions:` list — marks itself `is_key: true` and optionally names the keyset(s) it belongs to, `keyset: platform` for one or `keysets: [platform, natural]` for several. The per-DataKind `keys:` block retires, and so does `Key` as a Semantics with its own registry role, expression leaf and accessor. Keyness becomes an attribute of the place where the dimension is already declared.

**A keyset advises; it never constrains.** Uniqueness over its members is assumed, never asserted, enforced, or verified against data. A relationship resolves its join basis from its own `on:` specification (ADR 0038) and never from keyness. Consumers may let a keyset inform identification-dependent decisions but may not rely on it. That posture is what makes the feature safe to ship with no verification story.

**Arity is cover, not partition.** A site may belong to any number of keysets, and overlap is the normal case: a tenancy column legitimately sits in every keyset a DataKind has, because neither `platform = {campaign_id}` nor `natural = {campaign_name}` would otherwise state a uniqueness claim the author knows to be false. A single-member keyset is legal, and so is a DataKind with no `is_key:` anywhere — absence means "no declaration", not "this DataKind has no keys".

**The legality table is complete and closed.** `is_key: false` with no keyset is not a key; `is_key: true` with no keyset is a member of `default`; `is_key: true` with exactly one of the two spellings is a member of what it names. Both spellings at once is a hard error, and so is either spelling without `is_key: true` — a membership that is not activated would silently do nothing, which is the state an author is most likely to write by accident.

**A spelling that names no keyset is a hard error, not an implicit `default`.** `keysets: []` and a blank name (`keyset: ""`, `keysets: [""]`) are refused, with the diagnostic naming bare `is_key: true` as the spelling for `default` membership. `{"default"}` is materialised for a site that spells *nothing*, never for one that spells an empty basis: reconciling `keysets: []` into `default` would invent a membership the author did not write. This is the same posture ADR 0038 takes on `on: []`, whose diagnostic names `cross` — an empty basis must not arrive by accident. `keysets: []` is refused ahead of the activation rule, so it is refused under `is_key: false` too; the empty list is the defect either way.

**Both refusals now hold on the code path too, closing the gap this decision left open.** `Keyness::keysets_named([])` used to yield `{"default"}` silently, so Rust minted exactly the membership the wire path refuses to invent — a caller obligation stated in a doc comment rather than enforced. The constructor is now fallible and refuses both an empty list and a blank name, and the site-level `with_keysets` propagates. `Keyness::key()` and `with_key()` stay infallible: neither names a keyset, so neither can fail, and marking a site a `default` member stays one chainable expression.

**The rules split by stage, deliberately.** The rejections are pure syntax over a single mapping, so they belong at **parse**, where the author gets a located diagnostic and typed `ParseErrorKind` variants. The singular-into-plural merge and the `{"default"}` materialisation are **reconciliation**. The consequence must not be missed: between parse and compile the field pair has **three** states, so canonical membership is only correct through the accessor, and any consumer reading `keysets` directly silently misses singular-spelled and implicit-`default` membership. The accessor *is* the reconciliation; the manifest keyset roster that consumes it is a separate decision. The constructor's two refusals are **validate**-stage `ValidateErrorKind` variants restated rather than borrowed from their parse twins, because nothing parsed at a constructor and diagnostic identity is per-stage (ADR 0004) — the same shape the relationship builder's declared-basis rules take (ADR 0038).

**`default` is reserved, and explicit and implicit spellings of it are indistinguishable.** `keyset: default` and `is_key: true` with no keyset reconcile to one membership, so there is one meaning rather than two near-synonyms. A DataKind may declare only non-`default` keysets, and a **partial `default`** — some sites naming keysets, others bare — is legal, so nothing forces an author to name every keyset the moment they name one. A partial `default` is the sharpest smell in the roster, but smells belong to the advisory system, which is out of scope.

**Three fogged validation rules dissolve rather than needing gates.** An empty keyset cannot exist, because a keyset exists only because some site names it. A keyset naming a dimension the DataKind does not expose cannot exist, because membership is by-site. Unrepresentability is the enforcement, not a check that could be forgotten. **Identical-membership keysets in one DataKind survive as the one thing to enforce**, and they are a hard error rather than a silent collapse: collapsing would discard a name the author deliberately chose, and the posture is to refuse rather than to choose on the author's behalf. It is reachable through the accessor, so it is a model-side `ValidateErrorKind`. *The ruling behind this said only that "there should not be a state with two identical keysets in one scope"; error-rather-than-collapse and the model-side home are readings of it.*

**Member order is site-list position, projected per keyset — and never an input to identity.** Because the interface's dimension collection is id-keyed with a name-ordered wire projection, that order is name-sorted, which is precisely what content-derived identity seeding requires. A site shared by two keysets therefore holds the **same relative position in both**; independent per-keyset ordering is not expressible, and that is an accepted non-goal. Order is preserved and observable but not load-bearing today; its first named consumer is the out-of-scope auto-join election rule.

**A forward hazard, disclosed rather than designed around.** The site types derive `PartialEq` over `keyness`, which the round-trip tests need — without it a plural-only-serialisation bug would compare equal and hide. The cross-DataKind occurrence-conflict merge is deferred and unbuilt; when it is built it must compare **pooled** shape and exclude per-site keyness, or one DataKind marking a shared dimension `is_key: true` would read as an occurrence conflict — which is precisely the model-global keyness this ADR forbids. Recorded at the deferral site in `compile/index.rs` as well as here.

**Two serialisation facts, stated so they are not discovered.** Membership is a `BTreeSet`, so `keysets: [a, a]` silently collapses to `{a}`. Serialisation emits the **plural form only**, omitted when membership is empty or exactly `{"default"}`, so `keyset: natural` re-serialises as `keysets: [natural]` and round-trip is **semantic, not byte-exact**. A corpus whose round-trip is asserted byte-exact must therefore author the plural.

### What the unification rule governs

ADR 0027 requires a Semantics' fields to unify by **strict equality** at every occurrence of its name, with only `expr` — plus `filters` for Measure and Metric — varying at a reference site. Read carelessly, `is_key:` sitting beside `ref:` looks like a breach of exactly that rule.

It is not, and the general form is stated once so the next case is decided rather than re-argued: **a field that exists only on the local body, with no pooled counterpart, is not an override.** Unification is a relation between two spellings of the same field — the pooled declaration's and the site's. Where the pooled body has no such field, the site's value has nothing to disagree *with*, so no equality is violated. `agg:` on a `MeasureRef` would be a genuine breach, because `agg` exists on the pooled `Measure` and a site value would contradict it. `is_key:` on a `DimensionRef` is not, because the pooled `Dimension` has no `is_key`.

**The rule's teeth are the type split, not the argument.** `Dimension` (pooled) and `DimensionDecl` / `DimensionRef` (local) are distinct types, and the keyness fields exist on the latter two only. Parse routes them to the local body and never accepts them into the pooled body or into the shared `Reference` — so a pooled `is_key:` fails by name as an unknown key, and keyness is unrepresentable on Measure and Metric sites for free. The boundary is enforced by construction. Putting keyness on the pooled body is the shape that *would* breach the rule: it would make keyness model-global, so one DataKind's key claim would leak into every other DataKind referencing that dimension.

The reading generalises: a local-only field on any Semantics carrier is admissible under ADR 0027 provided the pooled body cannot spell it, and the test for that is a type split rather than a review convention.

### Complex data kinds

All four Public forms flatten the same semantic-interface type, so a Unionset, Grainset or Joinset authors keysets exactly as a Dataset does. Nested forms carry a body only and have no interface, so keyness is unrepresentable there and the one-interface-per-Public-DataKind rule is enforced by the types rather than by a check.

**The drop is the enforcement; a warning is so it is not silent.** The DataKind wire mirrors flatten their body, so serde cannot carry `deny_unknown_fields` on them and a DataKind-level key it does not know is discarded. Two such keys used to mean something — the retired `keys:` block, and keyness on a `Nested*` form — so each raises a **warning-severity** parse diagnostic naming the ignored key. A keyset an author meant to activate may not silently do nothing, and `parse` fails on any diagnostic it *returns*, so the warning rides the builder to the one warnings channel there is, `build()`'s.

**Nothing checks an authored keyset on a complex kind, and that is correct.** A keyset asserts nothing a check could contradict:

- A **Unionset**'s branches may or may not be disjoint. Disjointness is a fact about *rows*, and verifying claims against data is barred, so a keyset on a composed union is advice about the composed surface and cannot be falsified from the model.
- A **Grainset**'s per-level keysets are byte-identical member sets, because the temporal shape kind is unanimous across levels. Two levels agreeing are therefore *the same advice*, not a collision to report.
- A **Joinset**'s keyset describes the **composed surface**, while its members and hops carry their own local implicit keysets. Those are different levels of the same structure, both true at once; a check comparing them would compare claims about different row sets.

The only hard gate on a complex kind is the structural one the manifest will apply: every member of a keyset must also belong to the referencing DataKind's interface dimensions.

**Derivation rules, recorded in prose and built nowhere.** They are handed to the compile-stage effort, because composite lowering does not exist — the single-leaf guard rejects every complex form — so there is nothing to build against:

- A **Unionset**'s composed keysets are the **intersection** of its branches'. A keyset that only some branches assert cannot identify a row of the union.
- A **Grainset** gets one implicit keyset per level, computed from that level's coverage.
- A **Joinset** propagates **all** of the anchor's keysets, with **no primary/alternate election** — cover arity means nothing has to be elected.

### Amends

- **0027:** a field that exists only on the local body, with no pooled counterpart, is not an override: a local-only field on any Semantics carrier is admissible provided the pooled body cannot spell it, and the test for that is a type split rather than a review convention; `agg:` on a `MeasureRef` would be a genuine breach, because `agg` exists on the pooled `Measure` and a site value would contradict it, while `is_key:` on a `DimensionRef` is not, because the pooled `Dimension` has no `is_key`.
