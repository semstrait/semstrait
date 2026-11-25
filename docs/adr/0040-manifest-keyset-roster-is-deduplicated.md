---
number: 40
title: The manifest keyset roster is deduplicated, and `Key` is not a Semantics
status: accepted
date: 2026-08-10
source:
- crates/semstrait-manifest/src/manifest/keyset.rs
- crates/semstrait-manifest/src/compile/keysets.rs
- crates/semstrait-manifest/src/validate.rs
links:
- target: 36
  kind: amends
- target: 41
  kind: amendedby
- target: 43
  kind: amendedby
- target: 45
  kind: amendedby
- target: 47
  kind: amendedby
- target: 48
  kind: amendedby
- target: 60
  kind: amendedby
- target: 65
  kind: amendedby
---

# The manifest keyset roster is deduplicated, and `Key` is not a Semantics

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0041](0041-recompile-is-the-callers-decision.md): `manifest_epoch` is renamed `manifest_version`, and the rename rides the existing move and costs no additional version.
> - [ADR 0043](0043-authored-primitives-get-typed-carriers.md): the reference edge inverts to the data kind's own keyset list, and `SemanticInterface` stops being an entity.
> - [ADR 0045](0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md): both handed-forward items are discharged: a branch or hop is a semantic source, so no per-unit lane is owed, and the roster entry carries no origin field.
> - [ADR 0047](0047-coverage-mask-is-a-serveability-statement.md): the member-mask additive door closes, because a keyset entity has no member list for a mask to copy.
> - [ADR 0048](0048-composite-keysets-are-authored-inside-a-boundary.md): superseded on carrier grounds, scoped to the carrier. A keyset is `{id, name}`, minted over `(semantic_source_id, keyset name)`; its `members`, its `data_kinds` and its `(name, sorted members)` dedup all go, because there are no members on the entity to seed with. The seed the human ruled is precisely the rejected alternative *"Seeding the id per DataKind"*, whose rejected cost — no cross-source sharing, one roster entry per source per name — is now accepted deliberately, since a source-scoped seed makes cross-data-kind dedup impossible by construction rather than merely unimplemented. An authored composite keyset is boundary-checked. Everything else stands: keyness advises, `KeysetName` is its own newtype, there is no DataKind bit lane.
> - [ADR 0060](0060-a-scope-qualified-name-holds-one-entity.md): load check 8 retires by loss of subject; the identical-membership rule's one home is the model's.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): `manifest_version` moves 2 → 3, and no backward compatibility is owed.

## Context and Problem Statement

Identification advice was discarded at compile: the retired `keys:` block reached nothing (ADR 0039). An auto-join derivation, a fan-out advisory or a cartesian warning all need identity present in the artifact.

## Considered Options

- A deduplicated top-level roster with content-derived ids and id-set references (chosen).
- **A per-interface carrier** — the keysets living on `SemanticInterface`. Rejected as **unsound**, on a fact rather than a preference: the interface id is content-addressed on its sorted member-id set, so one interface is shared by every DataKind with identical membership, and two DataKinds with the same dimensions but different keysets would mint one interface with one set of keysets silently overwriting the other. It would also double-count in `member_count()`.
- **Inline on the DataKind** — a `keysets` vector on each `DataKind`. This was the session's own recommendation and it lost on one fact: **inline makes dedup impossible**. Two DataKinds asserting one shape would carry two copies, so the artifact could not record that they *are* one shape, and nothing could enumerate "every keyset in the model" without walking every DataKind.
- **Seeding the id per DataKind** — including the owning DataKind id in the identity seed. It forfeits sharing by construction: identical content in two DataKinds would mint two ids, so the roster would hold two entries for one shape and the merge path would be unreachable. The whole benefit of content-derived identity here is that the dedup costs nothing.
- **Mask-valued members** — a `SemanticBitmask` instead of an id set. A mask **cannot seed identity**: `bit_position` is a dense index into *one* manifest and explicitly not a persistable identifier (ADR 0036), so an id minted over a mask would not be stable across recompiles. The ids are needed for the seed regardless, which leaves a mask as pure redundancy; it stays available as an additive door for containment queries.
- **Skipping an inline-only keyed site.** Cheapest, and it never blocks an author. It silently discards authored keyness — the exact defect the retired `keys:` block had, where a carefully authored block reached nothing — so it is refused by name instead.
- **Warning severity for that refusal.** Tempting, because the site's *other* content still compiles and the roster could ship without it. Rejected: the manifest would then be missing a claim the author wrote, with only a warning to say so, and every downstream consumer would read absence as "no declaration" (ADR 0038: absence is never evidence). An author who wants the keyness pools the dimension; an author who does not deletes the flag. Both are one edit.
- **Keeping `SemanticRole::Key` dormant with a doc comment.** Symmetric with the dormant lanes ADR 0038 does disclose, and it would leave a landing site if `Key` ever returned. Rejected because this work indicted dead lanes specifically, and a role nothing has ever minted is the purest instance: disclosing it would preserve the thing the disclosure exists to warn about.

## Decision Outcome

The compiled manifest gains `keysets: BTreeMap<EntityId, Keyset>`, where a `Keyset` carries `id`, a `name: KeysetName`, `members: BTreeSet<EntityId>` of pooled Semantics ids, and `data_kinds: BTreeSet<EntityId>` of every DataKind asserting it. Uniqueness is **`(name, members)`**: two DataKinds declaring `default` over the same dimension are **one entry** carrying both DataKind ids, while the same name over *different* members is **two entries**, so the dedup never merges two genuinely different claims. Identification advice becomes addressable: "this side's keysets" is now a filter over the roster's DataKind references.

**Two numbers name this collection, and both are right.** It is the **sixth** top-level `EntityId`-keyed collection *field* on `SemanticManifest` (after `interfaces`, `data_kinds`, `bindings`, `sources`, `relationships`), and the **seventh** member of load's `id == map key` and global-uniqueness walks, which also visit the Semantics registry's own `entries` map — a collection that is not a top-level field. Prose about the manifest's shape says sixth; prose about a load walk says seventh. This sentence is the only place the two are reconciled, and every other mention points here.

**Deduplication makes the keyset a *shape* and moves the uniqueness claim onto the reference edge.** The entity is a named group of dimensions; the assumed-uniqueness *claim* is asserted per `(keyset, DataKind)` pair and is carried by membership in `data_kinds`, not by any field on the entity. This amends the earlier "a *DataKind's* key dimensions" wording, and only the locus of the claim moves — uniqueness stays assumed and guarantees nothing, arity stays cover, and no join basis reads keyness (ADR 0039, ADR 0038). `KeysetName` is its own newtype rather than a `SemanticsName` or a bare `String`, so passing a Semantics name where a keyset name belongs is a type error rather than a silent wrong lookup.

**The accepted cost, named rather than discovered.** Any future fact that is per-`(DataKind, keyset)` has nowhere to live on the entity. The concrete candidate is a primary-versus-alternate distinction: recording it would require the DataKind reference to grow from a bare id into a struct, which is a persisted-shape change and therefore an epoch move. Accepted, because it is speculative, nothing here needs it, and the growth is contained — one field on one reference — rather than a reshape of the carrier.

**Identity is content-derived, with no owning DataKind in the seed.** A new `EntityNamespace::Keyset` (label `keyset`) mints through the *existing* `mint_entity_id` primitive over `(name, sorted member ids)`, so nothing new is invented in the identity lane (ADR 0010). Three consequences are load-bearing. First, **the dedup is not a compile pass**: identical content mints an identical id and lands on the same map entry, so nothing has to detect duplicates. Second, **compile must MERGE `data_kinds` on collision, never overwrite** — a plain insert silently drops the second DataKind's assertion, which is exactly the class of silent loss this whole change exists to retire, so the merge path is tested explicitly. Third, **members hold pooled Semantics ids, never site ids**: a site id is absent from the registry so it would dangle, and because site ids are minted per site, seeding on them would make every keyset unique and the dedup would never fire. Because the seed sorts its members, reordering an authored `dimensions:` list cannot churn a keyset id — and so cannot renumber a neighbour's `bit_position` (ADR 0036).

**A keyset member must be a pooled Dimension, which promotes a convention to a requirement.** The Semantics registry is built from the model's three root pools only (`compile/index.rs` never walks DataKind interfaces), so a dimension site that is **inline-only** — its name absent from the root `dimensions:` pool — has no registry entry and therefore no pooled id to be a member. Such a site marked `is_key: true` raises an **error-severity** `CompileError::KeysetMemberNotPooled` naming the DataKind, the keyset and the site. It is not skipped: silently dropping authored keyness is the failure class ADR 0039 exists to retire, and a keyset an author meant to activate may not silently do nothing.

**A wrong-role member is a different fault and carries its own variant.** A keyed site whose name resolves to a pooled Semantics of *another* role raises `CompileError::KeysetMemberNotADimension`, which names **the role it resolved to**. Collapsing the two would make the diagnostic false: a Measure *is* pooled and *does* have a registry entry, so "no registry entry to reference" misdescribes it, and the one fact the author needs — which role the name actually reached — would go unnamed. This is the join-key precedent applied verbatim (ADR 0038's `RelationshipJoinKeyNotADimension` carries `role` for exactly this reason). Both variants are reachable through `parse` + `compile()`: the no-entry cause needs only an inline-only keyed site, and the wrong-role cause a dimension site — `ref:` or inline — naming a pooled Measure, which nothing upstream rejects (`guard_no_name_collision` compares root-pool names to each other, and the model's inline-shadow warning is same-carrier), so both are tested at the public compile seam. The corpus is unaffected — every `is_key` site in `crates/semstrait-model/schemas/reference.yaml` is a `ref:` site over a pooled dimension (path corrected from crate-relative `schemas/reference.yaml`; decision unchanged) — so what this buys is that "a keyset's basis is a shared pooled Semantics" is now enforced rather than merely conventional, which is the same promotion ADR 0038 records for a relationship's join basis.

**Both references are sets of ids, and no mask is persisted.** The DataKind reference *must* be ids: there is no DataKind bit lane, because every persisted mask is a coverage view over the *Semantics* registry (ADR 0036), and opening a second lane is not this work. The member reference is ids too, although a mask was genuinely available — the ids are needed for the identity seed regardless, a mask cannot *be* an identity seed, and the containment mask is derivable at graph build from the members. A member mask is therefore left deliberately as an **additive door**, pending the coverage-mask-semantics decision. Set-valued members mean **member order is unrepresentable and equality is set-equality structurally**, obtained by construction rather than by a rule someone must remember: a deliberate departure from the interface role vectors' style, and available because the model side is already name-sorted (ADR 0039).

### What the interface no longer carries

**`SemanticInterface.keys` is deleted.** Its two alternatives are closed by facts rather than by preference. A *keysets pointer* on the interface is dead, because the reference runs keyset → DataKinds and the interface holds no pointer back. A *projection of the key-marked dimension ids* is **unsound**: the interface id seeds on the sorted, deduped member-id set, so **one interface is shared by every DataKind with identical membership** — two DataKinds with the same dimensions but different keysets would mint one interface and one set of keysets would silently overwrite the other. That is the partitioned-dataset case, reachable rather than exotic. The projection would also double-count in `member_count()`. So the slot goes: the constructor drops from seven arguments to six, `member_count()` sums **four** vectors, and load's dangling walk drops the key chain. This answers the manifest-carrier effort's open question — "role tag, or must the declaration persist?" — as **neither**: the tag goes, and the declaration persists as the roster.

**`Key` retires as a Semantics, manifest side.** The `SemanticRole::Key` variant is deleted and load's join-key-side role class narrows from Dimension-or-Field-or-Key to **Dimension-or-Field**. Nothing spellable is lost: no `Key` registry entry has ever been minted, so `key: foo` could parse and never resolve, and the key accessor was variant-for-variant identical to the dimension accessor (ADR 0039 retired the model-side half). Keeping the role dormant behind a doc comment was rejected — it would be a *third* dead lane of exactly the class this work was chartered to remove. The `ResolvedJoinKey` doc comment that named a `Key` semantic is corrected to Dimension / Field in the same breath.

**Correction to ADR 0036.** Its closing sentence — "which keeps it correct as `Key`, `Field`, and named filter entries join the roster" — is wrong about `Key`: `Key` will **never** join the `bit_position` roster, because it is no longer a registry role. ADR 0036's **operative rules 1–5 stand untouched**, and its lane-parameterized statement of the assignment rule remains correct for `Field` and for any per-class lane a later effort adds; only the enumeration of future arrivals loses one member.

### The load checks

**Nine checks** cover the roster, numbered here once and followed by the `keyset_checkN_*` tests in `load.rs`:

1. the map key equals the value's `id`;
2. `members` is non-empty;
3. `data_kinds` is non-empty;
4. every member resolves in the registry **and** has role `Dimension`;
5. every DataKind id resolves;
6. the `id` is **re-minted from `(name, sorted members)` and compared**;
7. every member also belongs to each referencing DataKind's interface `dimensions`;
8. no two keysets referencing the **same** DataKind have identical members;
9. every keyset `id` joins the global id-uniqueness walk.

They do not all live in one place, which is what an earlier draft's count got wrong: **six run per entry** inside `check_keysets` (2–7), **one needs the whole roster** and so runs after that walk (8), and **two ride the collection-agnostic walks** that every `EntityId`-keyed collection joins — 1 in `check_id_equals_key`, 9 in `check_global_id_uniqueness`. Check 9 is not extra scope: load's rule is that all ids across the entity collections plus `model_id` are globally distinct, so a collection *abstaining* would be the thing needing justification. Three of these deserve their reason on the record.

The **orphan check is not an empty case, it is corruption**: a keyset exists only because some DataKind's site named it, so a keyset referenced by no DataKind is unmintable by construction and its presence means the artifact was hand-edited. The **re-mint is not optional**: ADR 0012's load guarantee re-derives every derived value, and the keyset id is derived, so it is compared like every other one — the constructor mints it rather than accepting it, which closes the *constructor path* to an id-inconsistent keyset. It is not a type invariant: `name` and `members` are `pub`, so field assignment can still desync id from content — which is precisely the case check 6 exists for, and precisely what its test does. The **membership refinement** follows directly from the keyset being minted by scanning that DataKind's own sites: a member outside the interface can only mean a hand-edit, so the check costs nothing to state and closes the one way an unreachable member could be smuggled in. The identical-membership rule is grouped **per DataKind**: across DataKinds, identical membership under different names is legal, because two DataKinds may genuinely share a shape.

**The identically-shaped refinement for relationship join keys stays deferred** (ADR 0038 records it as a `TODO` at site), so the asymmetry is deliberate rather than an oversight: the keyset refinement is enforced here because the keyset is minted *from* the interface it is checked against, whereas the join-key refinement waits on relationship lowering that does not yet run.

**A dormant lane, disclosed.** `SemanticManifest::validate()` still has **zero non-test call sites**, so these nine are contract tests on a checker nothing calls yet. That is not new with this decision — the whole load-time validation surface is in that position — but a reader should not mistake "checked at load" for "checked on the way in today".

### Drift

**Drift is disclosed and unguarded, deliberately.** No existing mechanism fires when a derived basis or a derived roster changes: recompiling a changed model produces a changed roster with no signal that the previous one said something different. The tolerance is inherited from the accepted-risk ruling on wrong keysets, on the precedent that a wrong `cardinality` already yields inflated sums with no warning. This **records the tolerance and claims no guard.** Three guards were considered and rejected. A **fingerprint field** on the roster is redundant against byte-identical encoding — deterministic compile already makes the whole manifest a fingerprint of its inputs. A **compile diagnostic on change** is unbuildable: `compile()` holds no previous manifest, and giving it one moves the compile/persistence boundary and breaks byte-identical determinism as stated (the same reason ADR 0036 rejected append-only bit numbering). A **derived-versus-authored provenance flag** is a persisted-shape change, and the derived-and-stored `join_type` is the standing counter-precedent: this codebase already stores derived values without marking them, and re-derives them at load instead.

### The epoch

**`MANIFEST_EPOCH` moves 1 → 2, and this is the single bump.** The move is forced, not chosen: the epoch is the only migration signal there is, and both adding a top-level collection and deleting the interface's key vector invalidate every persisted manifest and cache. The coordination point is discharged — the carrier-inventory decision is still open and has not bumped the epoch, so there is no double-move. The bump also covers churn it did not cause: ADR 0037's slice re-cut the authoring bodies, which reordered the model's wire keys and so moved `model_hash` and the `model_id` minted from it with the model's meaning unchanged. `manifest_epoch` fences manifest-shape changes rather than model-hash changes, so that churn would otherwise have shipped behind an unchanged epoch; it ships behind this one instead.

### Complex data kinds

The derivation rules for composed surfaces — a Unionset's composed keysets are the **intersection** of its branches', a Grainset gets one implicit keyset per level, a Joinset propagates **all** of the anchor's with no primary/alternate election — are recorded in ADR 0039 §"Complex data kinds" and are **not restated here**. Composite lowering does not exist (the single-leaf guard rejects every complex form), so this work mints only top-level keysets and there is nothing to build against. Likewise **nothing checks an authored keyset on a complex DataKind, and that is correct**: a keyset asserts nothing contradictable, and the only hard gate is the structural membership refinement above.

Two things are handed forward rather than swallowed. The **per-unit keyset carrier** and implicit-origin marking are settled in semantics but open in schema: implicit keysets are created per branch and per join hop, stored and referenced in this common roster rather than inlined per unit, and may carry an implicit origin plus its originator. Adding those lanes now would land a dormant lane with no reader. And the frozen clause that "every composed surface carries `keys` populated post-compile" must **repoint off the deleted interface slot onto the roster** when composite lowering lands.

### Amends

- **0036:** corrects the closing sentence: `Key` will never join the `bit_position` roster, because it is no longer a registry role; operative rules 1–5 stand untouched.
