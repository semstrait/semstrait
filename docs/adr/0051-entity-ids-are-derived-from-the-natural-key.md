---
number: 51
title: 'An entity''s id is derived from its name: the technical key is a function of the natural key'
status: accepted
date: 2026-08-25
source:
- docs/adr/0010-single-entityid-lane-content-derived-ids.md
- docs/adr/0029-joinset-scoped-relationship-shadow.md
- docs/adr/0036-bit-position-assignment.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- CONTEXT.md
links:
- target: 10
  kind: amends
- target: 60
  kind: amendedby
- target: 41
  kind: relatesto
---

# An entity's id is derived from its name: the technical key is a function of the natural key

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0060](0060-a-scope-qualified-name-holds-one-entity.md): its precondition list is incomplete. The lanes it enumerates omit a data kind's interface member lists. ADR 0051 did notice that site-level entities exist — its rejected-alternatives section observes that *"every site-level entity (keyness sites, filters) needs its own authored id for the same reason"* — so what it missed was the rule, not the entities. Its backstop claim does not hold for that lane. *"`DuplicateEntityId` becomes the backstop for any scope a rule misses"* is true only where the mint records into `id_locations`. The interface-member mint does not, so for that lane there was no rule **and** no backstop. Everything else about the backstop stands, including its argument that detection is an adequate replacement for impossibility.

## Context and Problem Statement

Today an entity that omits `id:` is minted `Uuid::now_v7()`. Its name and its id are independent, and the technical key carries entropy that no author wrote and no reader can reproduce.

**The root already derives its id from its name, and the entities are the exception.** `derive_root_id` is `UUIDv5(ROOT_NAMESPACE, name)` — deterministic, name-seeded, pinned to a frozen namespace by a test that recomputes it independently. The manifest side follows the same rule: every compile-synthesised id is `Uuid::new_v5(namespace, seed)` through one primitive. The authoring stage's mint is the only lane in the workspace that reaches for randomness, and it sits between two lanes that do not.

### The tolerance is real, and its stated justification is false

This does not overturn an oversight. ADR 0010 mandated content-derived UUIDv5 "for everything unauthored" and rejected "time-ordered UUIDv7 or per-compile counters for synthesized entities" — and then explicitly tolerated v7 in the authoring lane, naming the cost as "a bare `String` mixes v7 and v5 UUIDs in one field with the version unchecked". `semstrait-common/src/identity.rs:12-13` records that tolerance as the contract. So the v7 lane is ratified text, and it is superseded here rather than corrected.

What is *not* ratified is the reason the code gives for it. a code comment in the former model-stage id mint discharged the determinism invariant (identical inputs give an identical artifact) at the mint: "Time- and RNG-based, so values vary across runs; **determinism holds because output is name-ordered, never id-ordered**". ADR 0036 rule 1 fixes `bit_position` as the 0-based index **in canonical `EntityId` order** over the `BTreeMap<EntityId, _>` registry, and lists "**ordering by author name** (the pre-existing behaviour of the compile index)" among its rejected alternatives. A later decision overturned the comment's premise, so nothing discharges the determinism invariant for the bitmap lane.

### Two ratified properties do not hold, and each is silent

**The content-addressing gate.** ADR 0041 pins `model_hash` as SHA-256 over `serde_json::to_vec` of the model with only the root `id` nulled, and rests its argument on preimage totality — "every field participates". Every entity `id` is therefore in the preimage, so the same document parsed twice produces a different `model_hash`, a different `model_id` (`UUIDv5(model_hash)`), and a different key under `key_for`'s `base/{id}.{extension}`. ADR 0041's gate — "an unchanged model lands on the key it already occupies" — holds only for a model that authors an id on every entity, and the default profile does not. The failure is an `exists()` miss and a recompile, every time, with no diagnostic.

**The bitmap layout.** Semantics in the `SemanticBitmap` registry carry their model-stage ids — "model-authored entities reuse their model `id` (UUIDv7)", as the former manifest identity module put it. Under ADR 0036 rule 1 their bit positions are the index of those ids in sort order, so two parses of one document assign **different bit positions to the same semantics**, and every coverage mask moves with them. This is the worse of the two: `bit_position` is a persisted `u32` and a mask is persisted bytes, so the artifact differs where the determinism invariant says it must not, rather than merely landing at a second key.

**The two decisions lean on each other.** ADR 0036 declines cross-recompile bit stability on the ground that it is "unnecessary given that an unchanged model does not recompile at all (the `model_hash` gate)" — the gate reported broken above. 0036 rests on 0041, 0041 rests on deterministic ids, and the ids are random. One change discharges both; nothing narrower discharges either.

**The test suite already compensates.** The golden-manifest fixture hand-authors an id per entity because "otherwise `model_hash` / `model_id` vary across builds and no golden is stable" (`semstrait-manifest/tests/manifest_snapshot.rs:66-70`), and repeats it for sites at `:95`. A reproducible byte-target is reachable today only by refusing the default profile — the clearest evidence that the default profile does not produce one.

## Considered Options

- An id-less entity's id is UUIDv5 over its scope-qualified name (chosen).
- **Keep UUIDv7 and require `StrictRequireProvided` for determinism.** Today's answer, and the cheapest. Refused because it makes reproducibility opt-in for a property two ratified decisions assume unconditionally, and because the tree already shows it failing: the golden fixture hand-authors an id per entity to work around the pipeline's default. It also scales badly in the direction the project is going — every site-level entity (keyness sites, filters) needs its own authored id for the same reason, which `manifest_snapshot.rs:95` already discovered.
- **Content-seeded UUIDv5 over the entity's canonical body.** The other reading of "content-derived", and superficially closer to ADR 0010's wording. Refused because identity must not be a function of mutable content: editing a `description` would re-key the entity, so every prose edit orphans whatever the id addressed — the failure the rename cost is accepted for, incurred continuously instead of at a rename. It also inverts the natural/technical key relation this decision rests on, deriving the durable handle from the fields most likely to change.
- **Null every entity id in `model_hash`'s preimage instead of deriving the ids.** Fixes the fingerprint without touching the mint, in a two-line change. Refused on three counts: it leaves the bitmap lane broken, since bit positions read the ids and not the hash; it makes two genuinely different models fingerprint alike whenever they differ only in authored ids, which is the *dangerous* direction ADR 0041 names ("a recompile skipped and a stale manifest served"); and authored ids are authored content, so excluding them breaks the totality argument that is 0041's whole defence.
- **A fixed clock and seeded RNG injected for tests.** Makes the suite reproducible and would let the golden fixture drop its hand-authored ids. Refused because determinism would be a property of the harness rather than of the pipeline, so the artifact a caller compiles in production stays irreproducible while every test asserts that it is not.
- **Per-class id newtypes, or a `stable_ids` sidecar mapping name to id.** The sidecar buys rename-stable identity *and* determinism together. ADR 0010 already rejected both on the single-lane argument, and nothing here reopens that; the sidecar also reintroduces a second place where identity is recorded, which is the state deriving the id from the name exists to remove.
- **Deriving from a bare name, unqualified by scope.** One less rule to state. Refused because it collides exactly where ADR 0029 requires two independent declarations to coexist, so it would force 0029 open to buy nothing.

## Decision Outcome

An entity that omits `id:` is minted `UUIDv5(namespace_for_its_class, scope_qualified_name)` rather than `Uuid::now_v7()`, so parsing one document twice yields one model, one fingerprint, and one set of bytes. The **name is the natural key** — what a definition states, what a reference resolves, and what every other uniqueness rule in the model already guards — and the `id` is the **technical key** layered on top for storage and cross-artifact addressing. Deriving the second from the first is what makes the pair coherent. The minting primitive and its per-class namespaces move to `semstrait-common`, which both stages already depend on, so one mechanism serves the model and the manifest instead of two.

### The seed is the scope-qualified name, and the scope is what makes shadows work

The seed is the entity's name qualified by the path from its nearest named root: a bare `name` for a root-pool semantics, a root data kind, or a root-level relationship — each of which already has a uniqueness rule or acquires one — and `<parent>.<name>` for anything authored inside another entity. This mirrors ADR 0045's seeding of a produced unit as `<originator>.<relationship>`, so the authoring and compile stages qualify names in one idiom rather than two.

Scope-qualification is what keeps ADR 0029 legal. A Joinset-local relationship deliberately reuses a root-level name and is "an independent data-shape claim, not an override", so the two must share a name and differ in identity. Under a bare-name seed they would collide; under `<joinset>.<name>` versus `<name>` they are distinct **by construction**, and 0029 needs no reopening. The same qualification answers the Joinset-local minting gap as a side effect: the seed exists wherever the entity is authored, so an id can be derived at every site.

Authored ids are untouched. An author who writes `id:` still wins, the id shape rule is unchanged, and a v5 id passes it — the version nibble is deliberately unvalidated, already pinned by `accepts_any_version_nibble`.

### A rename re-keys the entity, and that cost is already paid one level up

This is the decision's real cost and the strongest case for the status quo: under v7 an entity's identity survives a rename, and under a name-seeded derivation a rename mints a new id and orphans anything keyed on the old one. The cost is accepted because the level above already behaves this way and says so. ADR 0041 establishes that `SemanticModel.name` "is always in the preimage" and that `model_hash_excludes_root_id_but_not_content` "asserts in terms that renaming moves the hash" — a rename already moves the fingerprint, the compilation identity, and the storage key — and it names the frozen claim to the contrary as debt. So a rename is already a re-key event for the artifact. Making the entity id follow removes an inconsistency rather than introducing a cost, and an author who needs identity to outlive a rename has the honest tool for it, below.

### Collision becomes possible instead of impossible, and that is the trade

Randomness bought collision-freedom by construction. A name-seeded derivation gives it up: two entities sharing a scope-qualified name derive one id. That is why the relationship-name uniqueness rule is a **precondition of this decision** rather than a diagnostic improvement — it makes the seed injective for the one lane that lacks it, alongside `DuplicateDataKindName` and `DuplicateSharedSemanticsName`, which already do it for theirs. `DuplicateEntityId` becomes the backstop for any scope a rule misses, and it is adequate precisely because it is a *detection* where v7 offered *impossibility*: a collision now surfaces as a typed diagnostic naming the id and its occurrence count, a better failure than the silent non-reproducibility it replaces.

### The profile survives, with the only purpose it can honestly claim

`IdentityProfile` is not retired; what changes is what each arm means. `ConvenienceGenerateMissing` stops meaning "accept irreproducibility for convenience" and starts meaning "derive the id from the name" — the default becomes deterministic rather than merely quiet. `StrictRequireProvided` keeps `ValidateErrorKind::MissingEntityId` and gains its first defensible reason: an authored id is now the *only* way to make identity outlive a rename, so a caller who needs rename-stable storage keys authors them, and one who does not gets reproducibility for free. Today strict is the only route to determinism; this decision moves that burden off the author.

### Amends

Each supersession is stated and scoped here rather than by editing the older ADR, per the precedent ADR 0037 set over 0031 and ADR 0050 followed.

- **0010:** the authored-lane tolerance, and only that: the clause admitting v7 into the `EntityId` lane — "a bare `String` mixes v7 and v5 UUIDs in one field" — retires; an entity that omits `id:` mints `UUIDv5(namespace_for_its_class, scope_qualified_name)`, so the lane now holds v5 throughout. Everything else in 0010 stands and is extended rather than weakened: one untyped `EntityId` lane, no per-kind newtypes, no `stable_ids` sidecar, `EntityId`-keyed `BTreeMap`s, deterministic content-derived ids for everything unauthored. Its rejected alternative "time-ordered UUIDv7 … for synthesized entities" now covers the authoring mint too, which is where it always pointed.

**`semstrait-common/src/identity.rs:12-13`'s stated contract**, which documents the tolerance as policy. It is code, not a decision record, so it is corrected rather than superseded; it is named here because it is the artifact a reader will find first.

**ADR 0036's determinism discharge is not superseded; its premise is repaired.** 0036 rule 1 is untouched and its rejection of name ordering stands. Id order becomes reproducible, so rule 1 and the determinism invariant stop being in tension. 0036's dismissal of cross-recompile bit stability also keeps its conclusion — masks still may not cross manifests — while its supporting appeal to the `model_hash` gate becomes true for the first time.

**Nothing in ADR 0041 moves.** The preimage stays serde-total, the root `id` stays nulled, and the digest stays `[u8; 32]`. This decision makes 0041's gate hold as written; it renegotiates none of its terms, and in particular it does **not** reopen the rejected option of excluding entity ids from the preimage.

### Consequences

- **The cutover orphans every stored artifact and cache**, as ADR 0010 states of any change to the derivation. It is a manifest-version event and rides the single unshipped version move ADR 0050 already notes; sequencing the two into one release is the whole of the coordination owed.
- **Two preconditions are handed forward as separate work**, both tracked: every relationship must have a derivable id at both authoring sites, and relationship names must be unique within their scope so the seed stays injective. Neither is optional — the seed is not well-defined without the first and not injective without the second.
- **`CONTEXT.md`'s Model fingerprint entry is corrected in the same change**, not flagged. Its claim that the fingerprint "covers authored content and not the document's own identity, so renaming a document leaves it unmoved" is false in both halves today: ADR 0041:20 records that a rename does move it and calls the old rationale frozen-text debt, and the preimage covers minted ids that no author wrote. The second half becomes true once the ids are derived, which is why the entry is worth fixing rather than deleting.
- **Two code comments are corrected at their source**: `semstrait-model/src/model/identity.rs:32-33`'s name-ordering discharge of the determinism invariant, which ADR 0036 overturned, and `semstrait-common/src/identity.rs:12-13`'s v7 contract. Both currently read as live policy.
- **The golden fixture can drop its hand-authored ids** once the mint is derived, and the assertion worth adding in their place is the direct one: parse the same document twice, assert one `model_hash` and one bit-position assignment.
- **A per-class namespace roster is owed in `semstrait-common`.** The manifest's `entity_namespace(kind)` mechanism — `Uuid::new_v5(&ROOT_NAMESPACE, kind.label())` — is the model to copy, and the classes the authoring stage needs are not the same set as the compile stage's. Fixing the roster is implementation, but the namespaces are frozen once published, on the same terms as `ROOT_NAMESPACE`, so each wants the independently-computed pinning test that constant already has.
