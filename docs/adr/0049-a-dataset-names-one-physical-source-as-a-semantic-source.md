---
number: 49
title: A Dataset names N physical sources as authored and exactly one as a semantic source
status: accepted
date: 2026-08-20
source:
- CONTEXT.md
- docs/adr/0011-thin-manifest-in-memory-graph.md
- docs/adr/0016-optional-catalog-model-wins.md
- docs/adr/0030-no-inference-of-additivity-or-temporal-shape.md
- docs/adr/0032-homogeneous-binding-unit.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
links:
- target: 11
  kind: amends
- target: 32
  kind: amends
- target: 43
  kind: amends
- target: 47
  kind: amends
- target: 58
  kind: amendedby
- target: 65
  kind: amendedby
- target: 70
  kind: amendedby
- target: 54
  kind: amendedby
- target: 56
  kind: amendedby
---

# A Dataset names N physical sources as authored and exactly one as a semantic source

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): the synthesised multi-source Unionset exists at `N > 1`. A Unionset stacks branches, so with one branch there is nothing to stack and the composition means nothing: a Dataset over one physical source stays a leaf and keeps the authored name.
> - [ADR 0056](0056-composed-temporal-shape-is-derived-at-compile.md): the grain is derived and carried where the constituents agree on one, and absent otherwise. An implicit Unionset always satisfies it, because every branch is the same authored Dataset over a different physical source, so the reduction *carries over* what the author wrote instead of deriving anything new. The rule that forbids a grain on a `ComplexDataKind` is therefore re-scoped to authoring.
> - [ADR 0058](0058-a-reduction-carries-the-authors-keysets-to-the-public-face.md): the premise that the synthesised Unionset has no actor, which is the clause that routed this question to 0058, is superseded: "the actor was never absent — the reduction divides one authored data kind into several manifest entities, and the author's claims travel with the name." Its conclusion — that the route was unruled — was correct; its reason was not. Nothing else in 0049 moves: the leaf's single physical-source reference, the `1:N` reverse edge, the successor invariant of exactly one override row per `(semantic, data_kind)`, `mode: All`, no null fill, the two refusals of zero, and the files-and-tables restriction all stand.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): the successor invariant *"exactly one override row per `(semantic, data_kind)`"* is restated as **one signature per `(semantic source, definition)`**, and made total: the optional case it allowed is gone.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): a path source's id is minted from `(base_path, format)`, not `(source_type, locator)`. A table source's id is unchanged. The single physical-source reference carries the selector (`Prefix | Pattern | Uris | Table`). The sources of one Dataset have distinct base paths, and no object is selected by two of them. The `1:N` reverse edge stands.

## Context and Problem Statement

The base schema specifies three of the four roles and stops: `Joinset{from, to}`, `Unionset{branches}`, `GrainSet{grains}`, then `Dataset[SemanticSource] { .. }`. The earlier design says so in as many words — "Dataset - holds type base information for this data kind … We need to check whats meaningful can be stored there." The question is what the leaf's role carries.

## Considered Options

- One physical-source reference per leaf, composed by a synthesised Unionset (chosen).
- **Keep the leaf plural in the manifest**, as `bindings: Vec<EntityId>` does today and as ADR 0043 and ADR 0047 both presume. It is the smaller artifact and it changes nothing. It keeps two vertical-composition mechanisms in the system — a fold inside the leaf and a Unionset above it — which is the condition this ruling exists to remove; and it leaves per-source variation dependent on an override collection whose non-unique pair invites the ambiguity 0043's disjointness rule exists to forbid.
- **Synthesise the Unionset in the graph rather than in the manifest**, which is ADR 0011's posture and what the former `plan_layers.rs` planner did (per-source scan, `UNION ALL`, re-aggregate). It keeps the artifact thin and needs no new entities. It leaves the leaf plural, so it is the previous alternative wearing a different hat, and it puts the fold in a place no keyset, mask or coverage rule can address.
- **Allow mixed `paths` + `tables` and rule cross-branch type reconciliation now.** It preserves an authoring surface that exists today and serves the real archive-plus-live-table case. It requires a type-reconciliation rule over two schema-provenance arms before anything downstream can be trusted, which is a decision with its own weight and no forcing function here.
- **Content-dedup authored semantic sources**, so that two Datasets with equivalent semantics over one source collapse to one entry. ADR 0045 names this and refuses it — a content seed "would dedup structurally identical units … and wrong for sources, where two identical private branches under different parents are two different resolution paths" — and it would leave one of the author's own names with no entry to resolve in a request's `from`.
- **Propagate the leaf's keysets onto the synthesised Unionset** as part of this ruling. It is the obvious answer and it may well be the right one. It is a third keyset mechanism beside derivation and boundary, and ADR 0048 opens by recording that conflating those two "is what made every earlier framing of this question wrong" — so it is not this ADR's to assume.
- **Let a source-less or unmatched Dataset carry no physical source.** Cheapest, and it makes "exactly one" mean "at most one" — an absence with no meaning that every consumer must then handle.

## Decision Outcome

Authoring is unchanged: `storage:` keeps its `paths` and `tables` vectors, so one authored Dataset still names many physical targets. **In the manifest the leaf's `role_definition` carries a single physical-source reference, and the expansion becomes composition** — one leaf semantic source per physical source, composed by a compile-synthesised Unionset that carries the authored name. The reverse edge is deliberately `1:N`: one physical source serves many Datasets, because its id is minted from `(source_type, locator)` and nothing else.

The reason is uniformity of mechanism rather than economy of entities. A leaf that reaches many relations is a second composition machine beside the ratified one, and every consumer that walks a leaf has to decide whether it is looking at one relation or a fold over several. After the reduction there is exactly one vertical-composition shape in the artifact, and the multi-source case is a **narrowing** of it — not an escape hatch beside it. The entity count is the price, paid knowingly.

This is counter-intuitive in two directions. It is more entities for the same model, in an artifact whose stated posture is thinness (ADR 0011); and it re-lands, as a semantic source, a shape that ADR 0043 retired under another name — `SemanticBinding` was `{data_kind_id, source_id, mapping}`, one per `(data_kind, source)` pair. What differs is that the shape is no longer a private carrier with bespoke rules: it is an ordinary `semantic_sources` entry, and ADR 0045's flat, name-keyed, id-addressed collection is what makes that affordable.

### The base schema left the leaf's role elided, and this fills it

**The answer is one field, and the emptiness is the result rather than an omission.** Everything a candidate would name is already homed elsewhere: `temporal` is a top-level field on the semantic source; location, format, partitions and table identity are `StorageProperties` on the physical source (ADR 0046); `semantic_mapping` resolves into `semantic_overrides` rows and pooled expressions (ADR 0037, ADR 0043). What is left for the leaf to say is which relation it reads, and it says it once.

### Per-source variation stops being the override collection's problem

**ADR 0043's invariant does not go vacuous — it inverts, and the inversion is this ruling's load-bearing product.** 0043 made the pair `(semantic, data_kind)` *deliberately* non-unique in `semantic_overrides`, one row per distinct realization, and required the `physical_sources` lists of the rows sharing a pair to be "disjoint and covering over that data kind's sources". With one source per data kind, two rows for one pair both list it, disjointness fails, and the rule collapses into **exactly one override row per `(semantic, data_kind)`**. That is the successor invariant, and it is stated here because 0043's own text says the opposite: a reader who mints N rows per pair on a reduced leaf is following a clause whose premise is gone.

**0043 is superseded on this clause only, and its mechanism is unnecessary rather than wrong.** Per-source variation was real and needed a carrier; the reduction moves the variation into the source's own entry, so the compensating machinery on the override collection has nothing left to compensate for. Both edges 0043 established stand untouched: semantic → physical is `override.expr_id`, expression → source is `expression.physical_sources`. The second stays plural because the expression pool is content-deduped — one interned `SUM(amount)` is consumed by every reduced leaf that shares it — and because ADR 0046's model-fed fork derives a source's columns from "the `ColumnRef` leaves of every expression whose `physical_sources` includes this source". Its reading narrows from *which source does this row govern* to *which sources consume this expression*, and 0043's own refusal to put a `physical_source_id` back on the override is thereby left standing rather than reversed.

**ADR 0047's leaf-mask clause is superseded in the same narrow way.** "A `Dataset` source's mask is the union of what its own bindings can produce" has no subject once a leaf has one source: a reduced leaf's mask is what its single source can serve, and the union is 0047's own Unionset rule, unchanged.

### The implicit Unionset is a narrowing of the authored one

A leaf carries exactly one `semantic_mapping`, so **every branch of a synthesised Unionset exposes the same semantics**. Three properties follow, and together they are what earns the reuse:

- **No null fill, ever.** `UnionsetBranch` exists so graph build can derive the null-fill complement `top_level_coverage \ branch_coverage`; under one shared mapping that complement is empty by construction. The synthesised form needs no per-branch interface and no per-branch coverage.
- **`mode` is `All`.** Deduplicating rows across the expansion of one storage entry would silently delete data the author never asked to fold, and `All` is the model's own default.
- **The produced unit cannot be wider than the leaf that produced it.** Its semantics are the leaf's, so there is no authoring surface hiding inside the synthesis and nothing for a compile step to invent.

**What the synthesis is not licensed to do is settled elsewhere.** ADR 0045 explicitly does not license enumeration — "which implicit sources exist, and what bounds the set, is a separate decision against ADR 0011". This ADR rules the leaf's cardinality and the composition's shape; the mechanics of the synthesis, its identity and naming rule under ADR 0045's originator-qualified scheme, and whether the wrapper exists at `N == 1` belong to that decision.

### The keyset route is named, not closed

A reduced leaf's public face is a Unionset, and ADR 0048 gives a composite two routes to a keyset: derivation, which only a Joinset gets, and a boundary, which only rejects what an author declared. A synthesised Unionset has no author, so one route is closed by form and the other has no actor. ADR 0048 already followed exactly that reasoning once, when it withdrew the Grainset's grain groups — "an implicitly-formed Unionset derives no keyset, so it would have no identity for anything downstream to join to … reached where no author can be asked" — and it names the cost: "a composite with an empty keyset roster is a silently wrong artifact".

**The gap is recorded rather than closed, and it is recorded because it is load-bearing.** `Auto` basis resolution and a Grainset's implicit cross-grain Joinsets read guaranteed keysets only, so a keyset-less public source cannot resolve an `on: auto` relationship — and multi-sourceness is a function of glob expansion at compile (ADR 0046), which makes a second matching file a change in join behaviour with no change to the model. The route is not ruled here. Naming an unbuilt ruled property in the ADR that creates it is ADR 0039's precedent ("recorded in prose and built nowhere") and ADR 0047's.

### Zero is refused twice, and one refusal derives a source instead

"Exactly one" is a claim about a sealed artifact, so both ways of reaching zero are closed, each by the stage that can see it (ADR 0044).

**A Dataset with no `storage:` derives its source from its own name.** The table identity is `(the catalog entry's `name`, its `default_namespace`, the Dataset's name)`. `LeafExtras.storage` is `Option<StorageConfig>` and nothing rejects its absence today — `dataset_source_ids` maps such a Dataset to an empty vector — so this is new behaviour, not a restatement. It is the honest default: a leaf whose only physical fact is its own name is the commonest authoring shape a catalog-backed model has, and ADR 0016's model-wins posture already makes the catalog the physical source of truth when one is supplied.

**Where that derivation cannot complete, it is a hard error at model validate.** No `catalog:` on the leaf and no model-level default, or a catalog entry with no `default_namespace`, leaves the namespace blank and the identity unformed. `catalogs.yaml` and the leaf's `catalog:` are both model-side documents, so every input to the check is present before compile; only the provider round-trip is compile's.

**A pattern that matches nothing is a hard error at compile**, naming the Dataset and the unmatched pattern. ADR 0046 puts glob expansion at compile, so this is the only stage that can see it, and it is the case that ADR's "the source roster is a function of filesystem contents" makes routine. Without the error, "a manifest Dataset names exactly one physical source" would be false in an artifact that compiled *successfully*, which is worse than any authoring mistake.

### Files and tables do not mix in one Dataset

`paths` and `tables` may not both be non-empty on one Dataset. This is a **version restriction with a stated reason, not a principle**: after the reduction each resulting leaf carries its own schema-provenance arm (ADR 0046), so a mixed block yields branches whose types were partly observed from a catalog and partly assumed from semantics declarations with unknown nullability, and the plan- and compute-side consequences of stacking those are unexplored. Rejection is model validate's: both vectors non-empty is parse-visible.

**ADR 0032's remedy is what the restriction points at.** 0032 required heterogeneity to "be expressed as separate Bindings composed by a Unionset"; with the binding retired that reads as *separate Datasets composed by an authored Unionset*, which the author declares deliberately instead of discovering it inside one leaf.

**One of 0032's guarantees survives structurally and one does not.** "One physical column name per Semantics across all its sources" is now a property of the single shared `semantic_mapping` rather than a rule to check, and one `FileFormat` per unit follows from `format` being one field plus this restriction. Cross-source *type* disagreement — 0032's `COMP_E_0317` — still has a subject inside `tables` alone, where two catalog tables may type the same column differently; it becomes a cross-branch reconciliation question on the synthesised Unionset, and its successor rule is **not settled here**.

### Amends

- **0011:** a Dataset over several physical sources persists a compile-synthesised Unionset in the manifest; synthesising it in the graph, "ADR 0011's posture", is rejected.
- **0032:** with the binding retired, the remedy reads as *separate Datasets composed by an authored Unionset*; "one physical column name per Semantics across all its sources" is now a property of the single shared `semantic_mapping` rather than a rule to check, and one `FileFormat` per unit follows from `format` being one field plus the restriction that `paths` and `tables` may not both be non-empty on one Dataset; cross-source *type* disagreement (`COMP_E_0317`) still has a subject inside `tables` alone, where two catalog tables may type the same column differently, and becomes a cross-branch reconciliation question on the synthesised Unionset, whose successor rule is not settled.
- **0043:** supersedes the disjoint-and-covering invariant with exactly one override row per `(semantic, data_kind)`; `physical_sources` now says which sources consume an expression.
- **0047:** the leaf-mask clause is superseded. "A `Dataset` source's mask is the union of what its own bindings can produce" has no subject once a leaf has one source: a reduced leaf's mask is what its single source can serve, and the union is 0047's own Unionset rule, unchanged.

### Consequences

- **`CONTEXT.md` gains a cardinality clause on its Dataset entry** — one authored storage entry, one physical source per semantic source — and its **Implicit Unionset** entry becomes a statement about the reduction rather than about a "binding". Both land with this ADR rather than after it, so the glossary never states a rule this artifact has not ruled.
- **Two accepted ADRs are superseded in one clause each, scoped in this body rather than by frontmatter**, per the precedent ADR 0037 set on ADR 0031 and ADR 0045 set on ADR 0025: ADR 0043's disjoint-and-covering invariant, and ADR 0047's Dataset-mask rule.
- **Three validity rules are named with their owning stages** (ADR 0044): `paths` and `tables` both non-empty, and an unformed name-derived table identity, at model validate; a pattern matching no physical target, at compile. None belongs to a roster.
- **One question is left open.** `TemporalShape.grain` is "forbidden on `ComplexDataKind`", so if a reduced leaf's public face is a Unionset, whether the composed shape ADR 0030 derives may carry the leaf's authored grain — and what a Grainset then routes on — is not settled here. The reduction must not drop it silently, which it would today.
- **What this obliges in `crates/` — recorded for the implementation effort, not done here.** `DataKindVariant::Dataset { bindings: Vec<EntityId> }` becomes a single reference; `SemanticBinding` and the `bindings` collection go with ADR 0043; `dataset_source_ids`' one-Dataset-to-many-ids map becomes one-to-one and its "the driver's single-source happy path consumes only the first" flag retires with the synthesis; `push_leaf_decls` gains the name-derived fork; and the empty-vector case it produces today becomes unreachable.

## More Information

### Open questions

- The synthesised Unionset's keyset route.
- Cross-branch type reconciliation within `tables`, which is `COMP_E_0317`'s successor and has no owner yet.
