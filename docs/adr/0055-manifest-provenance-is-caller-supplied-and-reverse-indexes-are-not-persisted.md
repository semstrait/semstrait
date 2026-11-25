---
number: 55
title: Manifest provenance is caller-supplied, and reverse indexes are never persisted
status: accepted
date: 2026-08-26
source:
- CONTEXT.md
- docs/adr/0036-bit-position-assignment.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0052-compile-gates-the-cardinality-it-materialises.md
links:
- target: 37
  kind: amends
- target: 41
  kind: amends
- target: 65
  kind: amendedby
- target: 88
  kind: amendedby
---

# Manifest provenance is caller-supplied, and reverse indexes are never persisted

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): the `depends_on` direction clause is **void**, its subject removed. So is the premise that "no closure is needed at all" for the orphan sweep, which rested on compile interning only roots: the sweep now closes over child ids, because a stored node is a position inside a tree and not a root. The reverse-index rule — no persisted name → bit and no persisted expression → consumers — stands, and ADR 0065's refusal of a `source_id` on the signature applies it.
> - [ADR 0088](0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md): in *`model_checksum` is not a spelling this codebase has*, "The top-level block therefore ends as `manifest_version`, `created_at`, `updated_at`, `model_hash`, `model_id`" becomes: the top-level block also holds `dialect`. In *`depends_on` keeps the child direction, on expression ids*, the premise that "no closure is needed at all" for the orphan sweep, which ADR 0065 made void, holds: each entry is a root that one signature or one Joinset source names, so the sweep is one direct pass over those roots (ADR 0088, rule 14); and the sentence from "A **structural reference inside the tree** is indeed absent" to "breaks `classify`'s totality" becomes: a link is a structural reference inside the tree, and the layer stays total (ADR 0088, rules 5 and 6). The rest of 0055 stands.

## Context and Problem Statement

Three clauses of the earlier base-schema design and one deferral from each of ADR 0043, ADR 0044 and ADR 0045 are settled here. The question is what writes the earlier base-schema design's `updated_at`, given ADR 0041's "no second timestamp field is minted" and "manifest state is never modified at all"; which way `depends_on` points and on which ids; and whether the earlier base-schema design's `semantic_bitmap_index` and its `model_checksum` spelling are kept.

## Considered Options

- Caller-supplied stamps; only the direction compile produces persists (chosen).
- **Remove `updated_at`.** The first recommendation of the session that produced this ADR, and defensible on 0041's own terms: no producer in the ratified surface could write a value distinguishable from `created_at`, and `SemanticManifestMetadata.annotations` already admits a mutation stamp as free-form provenance with no shape change. Refused once the sync path was named as the writer, because the field then has a real one and adding it after release costs a migration.
- **`Option<SystemTime>`, with `None` meaning "never updated".** It separates *never synced* from *synced in the same clock tick as the compile*, which the mirrored form cannot. Refused on ADR 0050's own test, which distinguishes an **unconditional** fact from a **genuinely absent** one: 0050 mandates a tautological `on_expr_id` because "every join has a condition", and refuses a fabricated `lit(true)` residual because a residual "is genuinely absent rather than unconditional". Every artifact has a last-write instant, so this is the first case, and mirroring states an unconditional fact rather than fabricating an absent one. It also passes ADR 0052's test for fabricated values — a mirrored stamp is non-load-bearing **and** distinguishable, by equality with its sibling.
- **Compile takes the prior manifest** and preserves `created_at` from it. The obvious way to get the same result, and what ADR 0041 forbids in terms. Refused; the injected instant obtains the same outcome with no comparison and no persisted read.
- **A sync that mutates only the drifted fields in place**, rather than a full compile pass at the same key. Rejected because compile is the only thing that computes serveability masks, and ADR 0047 computes a composite's mask structurally from its operands', so a partial mutation would have to re-derive the composite closure outside compile or leave the artifact violating 0047.
- **Persist the parent index instead**, either alone or beside the child edges. Alone, it inverts the rule the bit index follows, forces compile to write a fact it does not compute, and misnames the field. Beside, it stores one relation twice and mints the agreement check the "no persisted reverse index" rule exists to avoid.
- **Keep ADR 0037's semantics-id codomain** on the relocated field. Cheaper against the existing ADR, and refused because a semantics may reach several expressions across occurrences, so the edge would lose which one was copied.
- **A temporary check asserting `updated_at == created_at`**, on ADR 0045's headroom precedent. Refused: the value is non-load-bearing, so the check would guard nothing and `validate()` would reject decoded artifacts for a consequence-free fault.
- **Mint a glossary term for the sync.** Refused because the workflow is out of scope with no owner, so the entry would define a process with no subject; the writer is named descriptively instead.

## Decision Outcome

The manifest's two time stamps are **provenance the caller supplies**, never facts the artifact derives about itself: `created_at` is injected, `updated_at` is injected-and-restamped, and neither is ever read to make a decision. And where a fact and its inverse are both useful, the manifest persists **the direction compile produces** and leaves the reverse to be built at load: the name-keyed bit index is not persisted, and the expression provenance edge points at what an expression was **built from**, never at what consumed it.

The second rule is the one that generalises, and it decides two questions that look unrelated. A reverse index is a *search* structure: it has no authority, it cannot disagree with its source because it is derived from it every time, and persisting one buys a lookup at the price of an integrity check that must exist forever. So the artifact holds `semantic_definitions[].bit_position` and the expression's `depends_on`, and a consumer that wants *name → bit* or *expression → consumers* builds it on load.

### `updated_at` is reserved, and its writer is compile stamping an injected instant

**The field is kept, against ADR 0041's refusal, because a writer now exists to name.** ADR 0041 states that "no second timestamp field is minted", and that clause is superseded on carrier grounds: `updated_at` joins the identity block as a wall-clock epoch mark beside `created_at`, non-optional, **equal to `created_at` until a sync writes it**. It is reserved headroom in the sense ADR 0043 established — a field that persists before the path that moves it exists — and it is admitted now rather than later because the persisted shape is what this map is fixing, and a shape change after release costs a migration where this one rides ADR 0040's single unshipped `manifest_version` move.

**The writer is compile, and it does not touch ADR 0041's purity clause.** A catalog **sync** — the earlier base-schema design's own word for it, in its `schema_id` being "effective from last compilation, requires sync on query" — re-runs compile against a fresh catalog with an unchanged model. Because the model is unchanged, `model_hash` is unchanged, so the artifact lands on the content-addressed key it already occupies, and everything model-derived recomputes to identical values under manifest determinism. What differs is catalog-derived state and the stamps. Compile receives the prior `created_at` as an **injected provenance input** and stamps `updated_at` with the instant of this pass.

That is not the prior-manifest argument ADR 0041 forbids, and the distinction is exact rather than convenient. 0041 refuses that compile "gains no prior-manifest argument, no `compile_if_changed` sibling, and no comparison against anything persisted" — it forbids a **gate** and a **comparison** inside compile. Injecting a provenance instant is neither, and it is already the ratified direction for the first stamp: `created_at` is caller-injected today, with the constant it currently uses flagged in its own comment as awaiting "a real build-stamp source (injected, not read here)". The input tuple gains a provenance field of a class manifest determinism already excludes; nothing in compile learns that a previous artifact exists.

**Compile only compiles.** The decision *to* sync is not compile's and is not this crate's: something above holds the model, compares its fingerprint against the stored manifest's, and drives the pass; the `Repository` decides which segment of a stored artifact to overwrite. ADR 0041 already reserved that mechanism, describing its fingerprint trait as "implementable at *component* granularity, so a later partial-parse or partial-update path can rehash only what changed rather than the whole document". This ADR fixes the field and its writer's obligations, and places the workflow outside.

**No check asserts the reservation, and that is a deliberate departure.** ADR 0043 and ADR 0045 both assert reserved headroom with a temporary check — 0045 "holds the list to length ≤ 1, to be deleted when reuse lands rather than relaxed after the drift is discovered". No such check is written for `updated_at`, at either the compile stage or in `validate()`. The discipline exists to stop drift in a value a consumer reads to **decide**, and nothing decides on a stamp: ADR 0041 classifies the metadata block as non-load-bearing, the fingerprint never reads it, and byte-comparison of two artifacts must already exclude provenance. A `validate()` restatement would also reject a decoded artifact for a fault with no consequence, where ADR 0052's two-site rule turned precisely on its pair being load-bearing. Recorded so the absence reads as a decision rather than an omission.

### ADR 0041's "manifest state is never modified at all" is corrected, not repealed

**The sentence is not true as written, and it stopped being true inside its own ADR.** 0041 argues that an unchanged model lands on the key it already occupies, so "R8's last-writer-wins can therefore only ever fire for a byte-identical fingerprint, which means **manifest state is never modified at all**". A byte-identical *fingerprint* is not a byte-identical *artifact*: the same ADR made `created_at` a real wall clock, and `save` is a blind map put — the seam's own words are "R8: save is a map put, later write always wins". Two compiles of one model therefore write different bytes to one key. Admitting the sync path above makes the same point about catalog-derived state.

The clause is corrected in place of being repealed, on ADR 0052's precedent for correcting ADR 0038 in an ADR body: **no load-bearing state is ever modified.** A sync may refresh catalog-derived state and provenance at the key the artifact already occupies, because the key is derived from the model fingerprint alone. Requirement 6a is still satisfied structurally and for 0041's stated reason — a changed model cannot overwrite an unchanged model's artifact — and what the correction removes is the licence a reader would otherwise take from "never modified", that an artifact is byte-stable across compiles. It is not, which the committed-snapshot tests already have to accommodate.

### `depends_on` keeps the child direction, on expression ids

**The two sources disagreed about the codomain as well as the direction, and only one of the two relations they refer to is real.** ADR 0037 inlines nested semantics into one expression tree and records "the semantic edges that inlining flattens away … as `depends_on: BTreeSet<EntityId>` on `ResolvedExprMeta`" — the parent's entry listing the semantics it was built from. The earlier base-schema design puts `depends_on` on the pooled expression entry and fills it with a "Parent EntityId reference to expression" — the child's entry listing the expressions that consumed it. ADR 0043 relocated the carrier onto the expression entry when `ResolvedExprMeta` retired and explicitly declined the direction, handing it here.

One reading has to be refused before the rest can be settled: that no such edge exists because ADR 0037 rejected `PhysicalLeaf::ExprRef`, so no expression references another. That confuses two relations. A **structural reference inside the tree** is indeed absent, and by design, since a handle-leaf hides the aggregate nodes beneath it and breaks `classify`'s totality. A **record that one pooled expression was copied into another** is not absent: a metric's expression contains its measures' subtrees, and each measure keeps its own pool entry, so the relation holds one member per inlining. Both sources refer to that second relation, and they are exact inverses of each other.

**The manifest persists the child direction — the metric's pooled expression names the pooled expressions copied into it — and keeps the name.** Three grounds, in order of weight.

The first is the rule this ADR opens with. A parent index answers "who consumed me", which is a reverse search index, and this artifact does not persist reverse indexes; a consumer that wants one inverts the edge at load in one pass. Persisting the parent direction while deriving the child direction would invert the rule the bit index follows in the section below, for two relations with the same shape.

The second is that the child direction is what compile has in hand. Compile walks a semantics' body downward and knows each child as it inlines it, so the edge is written where it is produced; the parent direction would need an inversion pass over the whole pool to write a fact compile never computes.

The third is precision, and it is why the **codomain is expression ids** rather than ADR 0037's semantics ids. Under ADR 0043 a semantics may reach several expressions across its occurrences, so a semantics id cannot say *which* expression was copied, while an expression id can; and the field now lives on the expression entry, so a codomain matching its carrier costs no join through the definitions. So the earlier base-schema design wins on codomain and carrier, and loses on direction.

**The name is load-bearing, so the direction had to be pinned rather than left to the reader.** `depends_on` states what its owner needs. Under the child direction the name is honest; under the parent direction it would be read backwards by everyone who greps it, and — the concrete cost ADR 0044 named — a sweep would "walk the wrong way under the wrong reading".

**Two deferrals close with no new check.** ADR 0045 deferred expression-`depends_on` acyclicity here on the ground that "an acyclicity rule cannot be stated before the edge has a direction". The direction is now stated and the check is still **not owed**: child edges over inlining are acyclic by construction, and `semstrait-model` already rejects the only way to author a cycle (`ValidateErrorKind::CyclicSemanticsReference`, with its own `cycle_check.rs`), so a manifest-side check would restate another stage's rule — ADR 0042's producer-unreachable class, on ADR 0045's own precedent for declining a nesting-legality check. ADR 0044 deferred the orphan sweep's "transitive closure over `expressions.depends_on`" for the same missing direction, and **no closure is needed at all**: compile interns only the roots of semantics' expressions plus ADR 0050's join conditions, never a bare subtree, so every id in a `depends_on` set is some semantics' own expression and is therefore already reachable directly. The sweep stays a single direct pass over the referring collections.

**What the edge is for, so it is not mistaken for a planning input.** The resolution path runs definition → expression → physical sources and never traverses this edge, so it is not read to plan. It is expression **provenance**: what a compiled expression was built from, which the manifest cannot otherwise answer, because ADR 0037 refused to "rely on content-dedup alone to recover the dependency graph" — subtree sharing reveals an edge only if the pool content-addresses subtrees, "which the contract does not promise". Its consumers are explanation and the post-v1 optimisation `semstrait-ir` already reserves vocabulary for, whose own module doc is worth reading beside this ruling: a dependency DAG that duplicated the expression tree would be "a DRY violation and a sync hazard". This edge does not duplicate the tree; it records what the tree destroyed.

One of ADR 0037's two stated consumers is gone, and the field survives on the other. "Recompile-invalidation" has no subject: ADR 0041 makes recompile a whole-model decision gated on the model fingerprint, with nothing incremental to invalidate.

### The reverse bit index is not a member of the identity block

The earlier base-schema design's top level carries `semantic_bitmap_index`, a list of `(name, bit)` tuples. **It is not persisted anywhere**, which confirms ADR 0047 rather than re-ruling it: `semantic_definitions[].bit_position` is the sole persisted authority, and the name-keyed index is built at load for fast search. This ADR records only that the field has an owner and that the answer is "no carrier", so it is not left unowned between two decisions that each touched it.

**A persisted `bit_position` and a runtime bitmap index are different things and are never conflated.** `bit_position` is assigned by compile in canonical `EntityId` order and "frozen once — inside the bitmap's own constructor, after the entry roster is complete and before any mask is built" (ADR 0036); it is persisted and it does not move within an artifact. The bitmap index is a lookup structure a consumer builds from those positions when it wants search, and it is never in the artifact. `CONTEXT.md` gains **Bitmap index** for exactly this distinction, because the confusion can be made and was made while deciding this.

What a sync *can* move is which bits are **set** — a source's serveability mask under ADR 0047, since a dropped column can make a semantics unserveable. Because a sync is a full compile pass against a fresh catalog, composite masks are recomputed structurally in the same pass, so 0047's structural-union rule needs no separate obligation.

### `model_checksum` is not a spelling this codebase has

The earlier base-schema design's identity block spells the model digest `model_checksum`. `CONTEXT.md` forbids that word by name — **Model fingerprint** lists `_Avoid_: "model version", "checksum", "model id"` — so the schema's spelling moves and the manifest field stays `model_hash`. No glossary clause forbids "hash" for this value; the prohibition on "manifest hash" belongs to **Manifest version**, a different field. The glossary gains nothing: it names the concept (*model fingerprint*, and *compilation identity* for `model_id`) while the carrier spells the fields, which is the split it already maintains.

**The digest itself is unchanged: SHA-256 over the model with the root `id` nulled.** Recorded because "MD5 over content" was offered as shorthand while deciding this, and ADR 0041 refuses MD5 by name — "MD5 and a UUID-shaped digest are both refused, on collision resistance. The field stays `[u8; 32]`" — on the ground that the failure this digest guards is a *missed* recompile, so collision resistance is the property being bought.

The top-level block therefore ends as `manifest_version`, `created_at`, `updated_at`, `model_hash`, `model_id`.

### Amends

- **0041:** "No second timestamp field is minted" is superseded on carrier grounds: `updated_at` is minted beside `created_at`, reserved, and given a writer; the rest of that clause stands, including that the freed word "epoch" is not reused. "Manifest state is never modified at all" is corrected to *no load-bearing state is ever modified* (0055's *ADR 0041's "manifest state is never modified at all" is corrected, not repealed*); its operative content is untouched: the content-addressed key, the fingerprint's preimage, the refusal of a per-document pointer, and the placement of the gate outside the crate. Compile's input tuple gains a provenance instant, so its output is no longer a function of `(model, catalog, filesystem)` alone; manifest determinism is unaffected, since 0041 already narrowed it to exclude non-load-bearing provenance as a class, not as one field.
- **0037:** the `depends_on` codomain is superseded from semantics ids to expression ids; its **direction** is confirmed, not superseded, and its inlining decision is untouched; its "recompile-invalidation" consumer is recorded as void under ADR 0041.

**The earlier base-schema design's direction** — superseded; its carrier and codomain are ratified. **The `semantic_bitmap_index` and `model_checksum` clauses** — ruled against, the first confirming ADR 0047.

**Nothing in ADR 0036, 0043, 0044, 0045, 0047 or 0052 is superseded.** Two deferrals (0044's closure, 0045's acyclicity) and one handoff (0043's direction) are discharged.

### Consequences

- **Compile's input tuple gains a provenance instant**, so its output is no longer a function of `(model, catalog, filesystem)` alone. Manifest determinism is unaffected: 0041 already narrowed it to exclude non-load-bearing provenance as a class, not as one field, so no further narrowing is owed and no second amendment is needed for a second stamp.
- **The manifest cannot answer "which consumers use this expression" without a load-time pass**, and cannot answer expression provenance at all for any subtree compile did not intern as some semantics' root. Both are accepted; neither is a planning input.
- **A stored artifact is not byte-stable across compiles of one model.** Anything comparing two artifacts must exclude the metadata block, which is what 0041 already required of the snapshot tests and what the correction above makes explicit.
- **The sync workflow is unowned by design.** Which API compares the fingerprints, which `Repository` method overwrites which segment, and what a segment is, all sit with the recompile gate and caching, already out of scope. They are named rather than assumed covered.
- **Handoff, not performed here.** `updated_at` does not exist in the crate; `model_checksum` never did; `depends_on` does not exist on any manifest type, and `ResolvedExprMeta` — which ADR 0043 retired — still carries three fields and no edge; `created_at` is still the hardcoded `FIXED_CREATED_AT_SECS`, so 0041's real-stamp ruling and this ADR's injected-prior-stamp both remain unbuilt. `manifest_epoch` is still the code's spelling of `manifest_version`. Code convergence is a separate effort's.
