---
number: 60
title: A scope-qualified name holds one entity, and the rule is checked where the id is minted
status: accepted
date: 2026-08-27
source:
- CONTEXT.md
- docs/adr/0004-typed-diagnostic-kinds-no-error-codes.md
- docs/adr/0005-warning-preserving-stage-error-policy.md
- docs/adr/0027-closed-explicit-authoring-surfaces.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0051-entity-ids-are-derived-from-the-natural-key.md
- docs/adr/0052-compile-gates-the-cardinality-it-materialises.md
- docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md
- docs/adr/0055-manifest-provenance-is-caller-supplied-and-reverse-indexes-are-not-persisted.md
- docs/adr/0056-composed-temporal-shape-is-derived-at-compile.md
- docs/adr/0058-a-reduction-carries-the-authors-keysets-to-the-public-face.md
links:
- target: 40
  kind: amends
- target: 42
  kind: amends
- target: 51
  kind: amends
- target: 65
  kind: amendedby
---

# A scope-qualified name holds one entity, and the rule is checked where the id is minted

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): the override row gives way to a signature carrying `expression_id`, `filter_expr_id` and `keysets`; 0065's D1 checks shape lock at compile — a role disagreement keeps raising `SemanticKindMismatch`, and a disagreement on `data_type`, a Dimension's category, a Measure's or Metric's aggregation family, or additivity raises `SemanticShapeMismatch`.

## Context and Problem Statement

ADR 0051 made an id-less entity `UUIDv5(namespace_for_its_class, scope_qualified_name)`, and named the consequence itself: *"Collision becomes possible instead of impossible, and that is the trade… two entities sharing a scope-qualified name derive one id."* It then listed the rules that keep the seed injective — `DuplicateDataKindName`, `DuplicateSharedSemanticsName`, and relationship-name uniqueness — and called them *preconditions of this decision*. A data kind's own interface member lists are absent from that list, and nothing else guards them.

**The fault is a silent overwrite, not a silent coexistence.** `resolve_members` builds a data kind's interface collections. It computes `member_path` as `{data kind}.{collection}.{name}`, mints through `resolve_entity_id`, then does `out.insert(id, member)` and **discards the returned value**. Under ADR 0051 two sites for one name in one collection derive **one** id, so the second site replaces the first, and takes that site's `expr` override and its keyness with it. Nothing observes the replacement.

**This inverts the premise the question was raised on.** The framing inherited was that two same-named sites *coexist* with distinct ids, because the mint was `Uuid::now_v7()`. That is still true of the code and no longer true of the decision. The state to reject is therefore not an ambiguity in the model; it is the **loss of an authored site**, and it happens before any later stage can see that anything is missing.

**The backstop ADR 0051 relies on does not reach this lane.** That ADR names `DuplicateEntityId` as *"the backstop for any scope a rule misses"*, and it is adequate where it applies: `collect_duplicate_entity_ids` walks the builder's `id_locations` provenance index, so a collision is witnessed even though the map insert masked it — which `dedup.rs`'s own module documentation states in terms. But `id_locations` is written by exactly eight inserters, the four data-kind pools and the four root pools, and `resolve_members` is not one of them. Interface members reach neither a name rule nor the backstop.

## Considered Options

- Enforce path uniqueness at the mint, in parse, with a dedicated accumulating `ValidateErrorKind` (chosen).
- **A name index maintained at validate**, the retired precondition catalog's answer. Refused because it is unimplementable rather than unattractive: under a name-seeded mint the second site is destroyed at parse and its location is never recorded, so validation has nothing to inspect. This is the one alternative eliminated by evidence rather than by weighing.
- **A compile-stage check**, the earlier design's answer and the original assertion. Refused because compile receives a model that is missing a site and consistent, and cannot tell it from a model authored that way.
- **A name-keyed interface collection**, making the state unrepresentable. The most robust answer, and what the fork in the question assumed. Refused because ADR 0051 already made the collection name-derived, so the structure exists and produces a silent overwrite; and because it is a `semstrait-model` shape change bought to replace an accumulator that the same function already has.
- **Reporting the fault as `DuplicateEntityId`.** Free — `DuplicateEntityId` already exists and needs only that the interface mint record into `id_locations`. Refused because it names the consequence rather than the fault, and an author reading *"two ids are equal"* about ids they never wrote learns nothing about the two lines they did write. Recording into `id_locations` remains worthwhile as defence in depth, but it is not the diagnostic.
- **One error kind for every level, parameterised by the level**, as the retired `DuplicateSemanticName { list, … }` did. It gives an author one vocabulary. Refused because it means rewriting a live, tested rule for uniformity in a diagnostic, and because the two payloads genuinely differ.
- **Recording the keyset assembly in the artifact** — contributing sites, the `default` provenance, the spelling used. It would make a duplicate site witnessable in the roster and answer ADR 0039's handoff with a carrier. Refused because ADR 0039 already ruled the two `default` spellings one meaning, because the attributions already carry the data kind, and because it persists a derived view against ADR 0055.

## Decision Outcome

A name qualified by its scope holds **exactly one** entity. For an interface member that scope-qualified name — its **path** — is `{data kind}.{collection}.{name}`; at a root pool it is the bare name. The rule is enforced wherever an id is minted from a path: the mint records the path and its source location, and the fault is reported with the other name-uniqueness rules, accumulating rather than stopping at the first.

This is not a new rule. It is the **precondition ADR 0051 already depends on**, completed for the one lane that lacks it.

### The stage question was undecidable when it was asked, and is now decided by evidence

Two retired design documents disagreed about the owning stage: one said *"Duplicates within a list are a compile error"*, while the other mapped duplicate names within a list to **validate** and declared itself *"the authoritative mapping"*. Both documents are retired, so neither governs. What settles it is that both were written before a name-seeded id existed.

**A validate-stage name index is refuted rather than rejected.** By the time validation runs, the second site is gone from the collection and its location was never recorded. There is no evidence left to test. The catalog's answer is unavailable, not merely outvoted.

**A compile-stage check is worse than unavailable — it is aimed at the wrong artifact.** Compile receives a model that is internally consistent and simply missing a site. It cannot distinguish that model from one the author wrote that way.

So the rule belongs to the stage that mints, which is parse, and the reporting follows the channel already at that call site. `resolve_members` already threads a `missing` accumulator for the strict identity profile, which lands in the builder's `missing_id` and is reported once every document has been read. A second accumulator for colliding paths uses the same shape, which is why the rule can be complete for a lane that accumulates across documents as well as for one that does not.

### Identity was already stated in the ubiquitous language, and nothing had cited it

`CONTEXT.md`'s **Site** entry opens: *"**The one place** a data kind's semantic interface exposes a named semantics — either an inline declaration or a reference to the shared pool."* Two sites for one name in one data kind contradict the definition of a site, so the collision identity needed choosing no more than it needed inventing. The glossary also puts *"duplicate"* on the **Occurrence** entry's *Avoid* list, which is why this ADR states the rule as path uniqueness rather than as duplicate-site rejection.

**No `semstrait-model` shape change is required, and the check-versus-structural-gate fork is void.** The fork assumed the alternatives were a diagnostic or a name-keyed collection that makes the state unrepresentable. ADR 0051 already made it structural: the collection *is* keyed by something derived from the name. What that structure produces is a silent last-wins write, so what is added is an **observable**, not a shape. ADR 0043's *"the `semstrait-model` answer is no"* stands.

Three authored shapes collide and are one fault: two inline declarations, two `ref` sites, and one inline declaration beside a `ref`. A reference site's exposed name is its target's, and both site kinds carry their own minted id.

### Every name-seeded lane, and what guards it

| Lane | Path | Name rule | In `id_locations` |
| --- | --- | --- | --- |
| Data kinds — the four pools | bare name | `DuplicateDataKindName` | yes |
| Root-pool semantics | bare name | `DuplicateSharedSemanticsName` | yes |
| Relationships | bare name | open | yes |
| Interface members — `dimensions` / `measures` / `metrics` / `filters` | `{data kind}.{collection}.{name}` | **this decision** | no |

The overwrite is real on all four lanes. On the first three it is **witnessed**: the entity is lost to the map insert, both locations survive in `id_locations`, and validation fails before compile runs, so a damaged model never reaches a consumer. On the fourth there is no witness, which is the whole of the difference and the reason only one lane needs a rule added.

**Filters are in scope for the same reason dimensions are.** A filter is an interface member with a minted id and a path, and ADR 0043 makes a named filter a first-class semantic definition with a Filter role. Because `member_path` carries the collection, a dimension and a filter sharing a name in one data kind have different paths and do not collide — the earlier design's unified-namespace rule over element types is a separate question and is untouched here.

### `DuplicateSharedSemanticsName` needs no reconciliation: the difference is scope

The live `ValidateErrorKind::DuplicateSharedSemanticsName { carrier, name, occurrences }` and the retired `ValidateError::DuplicateSemanticName { list, name, first_at, second_at }` are **not two spellings of one rule**, and the retired catalog's prohibition on *"alternative variants for these checks"* is not engaged. `collect_pool` is fed root-pool iterators only, so the live variant covers the root pools; the retired rule covered the root pools **and** the per-data-kind lists. The per-data-kind half was never built. Nothing is renamed, nothing is superseded, and nothing routes to the earlier set: one lane gains a rule it never had.

### The diagnostic is its own kind

A dedicated `ValidateErrorKind`, raised where the other name rules raise theirs, accumulating per ADR 0005's pair and ADR 0042's default. It is **not** `DuplicateEntityId`, and it is **not** `DuplicateSharedSemanticsName`.

- `DuplicateEntityId` asserts that two ids are equal. Under a name-seeded mint that is true here, but it is the *consequence*. Reporting it names the wrong defect, the same complaint this question already recorded against a duplicate inline site failing as `KeysetMemberNotPooled` — an error raised for an unrelated reason.
- `DuplicateSharedSemanticsName` names a carrier. An interface fault must name a data kind **and** a collection, so the payloads differ. Two kinds over disjoint scopes is not one rule spelled twice; per ADR 0004 two identities are two kinds.

The accepted cost: an author meets two error kinds for what reads to them as one kind of mistake. The two mistakes sit at different levels and their payloads differ, and the alternative — one kind parameterised by the level — would mean rewriting a live, tested rule to gain uniformity in a diagnostic rather than in a decision.

### The manifest keyset roster owes nothing, and its check retires with its fields

ADR 0039 handed this over by name: *"The accessor **is** the reconciliation; the manifest keyset roster that consumes it is a separate decision."* The artifact carries **no** record of how membership was assembled — not which sites contributed, not whether `default` was authored or supplied, not which spelling a site used.

- ADR 0039 already rules that an authored `keyset: default` and an implied one *"reconcile to one membership, so there is one meaning rather than two near-synonyms"*. There is nothing to tell apart, so nothing to record.
- Contributing sites are recoverable, because every attribution names its data kind. ADR 0055 fixed the general rule: the manifest keeps the direction compile produces, and a derived view is built when it is wanted.

**The same-membership rule keeps one home, and it is the model's.** It exists twice today. In the model it works: `check_identical_keyset_membership` reads the keyset accessor, groups by member list per public data kind, and raises `DataKindKeysetsIdenticalMembership`, with a test. In the manifest, `check_identical_keyset_membership` groups on the keyset entity's own `members` and `data_kinds` — **both fields ADR 0048 deleted**, since a keyset entity is now `{id, name}` with membership attributed from the semantics side. So ADR 0040's load check 8 **retires by loss of subject**, joining ADR 0050's tautological structural check 8 and ADR 0056's `check_grainset_levels` as a check deleted rather than re-homed.

This discharges the routing that raised the question. That routing sent check 8 forward as *"the first confirmed stage-reachable-but-ungated defect"* — an artifact compile could emit that its own checker rejects. That defect no longer exists, and it was removed by ADR 0048's carrier change rather than by choosing a stage for it.

**No compile-stage restatement is owed**, on ADR 0052's own test: a rule earns two sites when its value is load-bearing. Two keyset names over one member set is not. A derived join basis reading guaranteed keysets would see two names for one group of dimensions and tie-break by keyset name, so it picks one of two identical claims and either pick is correct.

**ADR 0048's keyset id seed is untouched and already delivers per-data-kind uniqueness.** The id is minted over `(semantic_source_id, keyset name)`, so one authored name becomes one keyset entity **per source** — ADR 0048's deliberately accepted cost, restated by ADR 0053 and ADR 0058. A keyset named `default` on two data kinds is therefore already two entities with two ids, referenced by id and never by name. A later decision examined replacing the source id with a name path and kept the id input on purpose, because a source id is itself name-derived under ADR 0045 while an authored source `id:` lets a keyset inherit that source's rename stability. Nothing here reopens it.

### Two questions lose their subject

**Disagreeing keyness across two same-named sites has no instance.** Keyness sits on dimension sites, and two sites for one name in one collection of one data kind are now a fault at the mint. So a bare site can no longer fail to dissent from an `is_key: true` twin, and `keyset: natural` beside `keyset: surrogate` on one name is unreachable. ADR 0039's per-site attribution is untouched: keyness still never unifies **across** data kinds, which is a different scope.

**The comparison of a name's definitions across data kinds is implementation-only.** ADR 0043 already rules the comparison, in the ubiquitous language rather than as a new fact: *"A semantics' identity-bearing fields are identical at every occurrence of its name, and only how its value is obtained may vary per data kind. So a field that says what a value **means** rides the definition, and a field that says how it is **obtained** rides the override."* `description` and `ai_context` ride the definition, `data_type` with them; the override roster is `expr_id` alone. So the comparison set is the definition's own fields, keyness excluded, and the shape-lock conflict has both a rule and a stage. Nothing is owed here but the observation that no site enforces it.

### Amends

Two corrections to ADR 0051 are carried here rather than in place, on the precedent ADR 0052 set over ADR 0038 and ADR 0056 followed over ADR 0030. In-place repair was considered and is **unavailable**: ADR-authority closes a merged ADR to in-place edits, its exception is for a clause that is *unreadable as written*, and an earlier case supplied the negative instance that a clause reading perfectly well while being wrong on the merits is not a writing defect. ADR 0051's sentences read perfectly well.

- **0051:** its precondition list is incomplete. The lanes it enumerates omit a data kind's interface member lists. ADR 0051 did notice that site-level entities exist — its rejected-alternatives section observes that *"every site-level entity (keyness sites, filters) needs its own authored id for the same reason"* — so what it missed was the rule, not the entities. Its backstop claim does not hold for that lane. *"`DuplicateEntityId` becomes the backstop for any scope a rule misses"* is true only where the mint records into `id_locations`. The interface-member mint does not, so for that lane there was no rule **and** no backstop. Everything else about the backstop stands, including its argument that detection is an adequate replacement for impossibility.
- **0040:** load check 8 retires by loss of subject; the identical-membership rule's one home is the model's.
- **0042:** the identical-membership defect that 0042's *Decision Outcome* cites no longer exists, and it was removed by ADR 0048's carrier change rather than by choosing a stage for it: ADR 0040's load check 8 retires by loss of subject, and the same-membership rule keeps one home, the model's.

ADR 0051's operative decision is unaffected and is in fact what forces this rule: an id derived from a name requires the name to be unique in its scope.

### Consequences

- **The rejection is a `semstrait-model` obligation, so it leaves `semstrait-manifest` entirely.** The claim that *"the rule belongs at compile-stage reconciliation"* is refuted. Its duplicate-site half becomes model-crate work, and its roster half loses its subject with check 8, so it retains no work in this crate. Since it has no owning map, the follow-on effort must adopt it deliberately as model-crate work or it will read as covered and not be.
- **`CONTEXT.md` needs no new term.** **Site**'s *"the one place"* already carries the rule, and this decision only gives it an enforcement site.
- **A model that today parses will fail to parse once ADR 0051's mint lands**, if it holds two sites for one path. That is the intended effect, and it is the same class of cutover ADR 0051 already accepted for orphaned ids.
- **The artifact is byte-identical whether the model held one site or two.** This was recorded as *"unwitnessable"* and stays true; it stops mattering, because the model that would produce the ambiguity is now rejected at the mint.
- **Handoff notes for the implementation effort.** `generate_entity_id` is still `Uuid::now_v7()`, so this rule has no reachable fault until ADR 0051's mint lands — the two are one change, and building this check against the v7 mint would produce a check that can never fire. `SemanticInterface::keysets`' own comment reasons from *"Site ids are randomly minted"*, a premise ADR 0051 makes false, though its name-ordered conclusion survives for a different reason. `build_semantic_definitions` reads the three root pools, where the ruled input is every public data kind's semantic interface together with the root pools, which is why an inline-only site never reaches the registry and a duplicate inline key site fails as `KeysetMemberNotPooled`. `mint_keysets` still constructs `Keyset::new(name, members, data_kind)` and writes `data_kinds`, both retired by ADR 0048. The manifest's `check_identical_keyset_membership` and its two tests are deleted, not re-homed.
