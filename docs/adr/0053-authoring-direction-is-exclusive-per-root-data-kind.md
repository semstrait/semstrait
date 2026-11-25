---
number: 53
title: A root data kind is authored interface-first or constituent-first, and never both
status: accepted
date: 2026-08-19
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0015-refuse-ambiguity-name-order-selection.md
- docs/adr/0025-sealed-datakind-hierarchy-with-manifest-mirror.md
- docs/adr/0026-inline-only-nesting-tree.md
- docs/adr/0027-closed-explicit-authoring-surfaces.md
- docs/adr/0034-composed-surface-shape.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- decided in the authoring-direction grilling session, 2026-08-19
links:
- target: 25
  kind: amends
- target: 26
  kind: amends
- target: 39
  kind: amends
- target: 45
  kind: amends
- target: 64
  kind: amendedby
- target: 34
  kind: supersedes
- target: 77
  kind: amendedby
---

# A root data kind is authored interface-first or constituent-first, and never both

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0064](0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md): field-first resolution ranks every public candidate by `(compositions, granularity fit, exposed semantics)` ascending, and takes the least. Two candidates equal on all three are path ambiguity: the planner takes the first in name order, plans it, and returns an advisory naming the chosen source and the tied ones beside the plan. Nothing is refused.
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *`semantic_mapping` stays Dataset-only; only its key-space moves*, "`auto` keeps its meaning — 1:1 to a same-named physical column — in both" holds for a root Dataset only; on a constituent, `auto` and an explicit mapping map a name as 0077 rules (0077's *A constituent serves only the names it maps*).

## Context and Problem Statement

Today interface-first is the only authoring direction: no child can carry a semantic interface. This ruling adds a second direction and is deliberately counterintuitive in four ways. ADR 0026 refused shared children and this admits them, in one direction, without weakening the nesting matrix. ADR 0034's qualified `constituent.name` field is deleted rather than generalized, so an author who wants two same-named facts on one composed surface is obliged to name them twice. The two directions are equally supported yet **cannot be mixed**, a stricter rule than either direction alone would need. And the whole ruling is invisible in the artifact — a reader diffing two manifests cannot tell which document shape produced them.

### Why now, and why the cost is lowest today

**There is no composite lowering to rewrite.** `guard_single_leaf_dataset` (`crates/semstrait-manifest/src/compile/mod.rs:294`) rejects every Grainset, Unionset and Joinset, and a *second* Dataset. The entire composite path is authored-but-uncompiled, so the direction ruling costs no lowering rewrite today, and its price only rises.

**Today, reaching for constituent-first silently produces a wrong model.** `retired.rs` raises a warning for exactly two spellings — the retired `keys:` block and the three keyness keys on a `Nested*` form (`KEYNESS_KEYS`, `retired.rs:24`, 60–72). A `dimensions:` / `measures:` / `metrics:` / `filters:` block on a nested child is warned about by nothing, and the serde shape drops it: `NestedDataset` → `DatasetBody` → `DataKindBase` is an unbroken `flatten` chain with `deny_unknown_fields` only on `extras:`, below where the block would sit. So an author's instinctive constituent-first document compiles to an interface they did not write. *(This reads from the types; no test pins it, and pinning it is part of the implementation effort.)*

**Two ratified derivation rules have no authorable input under interface-first.** ADR 0039 rules that "a Unionset's composed keysets are the **intersection** of its branches'" and, in the same document, that "nested forms carry a body only and have no interface, so keyness is unrepresentable there". ADR 0040 inherits the gap for its per-branch implicit keysets. Constituent-first is what gives those rules something to read.

**The interface-first cost is visible in the live corpus.** A branch's mapping expression arm is `Expr<UnresolvedPhysicalLeaf>` — physical columns only — so a branch cannot define a semantics in terms of other semantics. That forces `test_data/acme_eu_ad_platform_v3.yaml`'s `market` dimension to be a ~110-line `case` over `dataset_name` **at the root**, hand-emulating five per-branch definitions that five constituents would each state in three lines.

## Considered Options

- Two authoring directions, picked per root data kind and never mixed (chosen).
- **Inverting outright** — constituent-first as the only direction, superseding ADR 0026's authoring half. This session's own first recommendation. Refused because interface-first is the right shape when a composed vocabulary is genuinely authored once and mapped into many sources, and because it would invalidate every existing document for no gain the two-direction rule does not already give.
- **Reaffirming interface-first** and recording constituent-first as rejected. It largely restates ADR 0026 and would leave ADR 0039's intersection rule permanently uninputtable.
- **Allowing the two directions to mix within one root** — the root declares part of the interface and constituents the rest. The most flexible option and the one an author would reach for. It gives one fact two homes, which is the drift ADR 0044 and ADR 0047 both refuse, and it makes "what does this root expose" unanswerable without compiling.
- **A declared direction keyword** (`direction: constituent_first`) checked against the body. Honest and explicit, and it makes the diagnostic easier. Rejected as a second source of truth for a fact the body already states — and it would have to be kept in agreement with the body by a check that could disagree.
- **Keeping `constituent.name` as an expression address** while deleting it as a surface field. It was this session's own reconciliation until ADR 0027's "no aliases or dotted names" was read exactly. The structured election needs no dotted form, so nothing is lost.
- **Keeping ADR 0034's qualified pair.** It preserves two genuinely different facts under one name — and the model already forbids that reading, so what it really preserves is the ability to leave a modeling error undiagnosed.
- **A fuse without the pooling requirement.** Cheapest, and it never blocks an author. It makes the fuse's well-typedness depend on ADR 0039's deferred cross-DataKind merge, so until that lands two inline declarations could fuse across disagreeing data types.
- **Publicity carried by the reference** — the composer declaring which constituents stay queryable. Reads naturally and matches how an author thinks about exposure. Unrepresentable under sharing: one entry, one `public` bit, two parents that may disagree.
- **Default `public: true` for every top-level unit.** The simpler rule, and what placement means today. It multiplies field-first candidates so that ADR 0015 refuses the tie, making most requests against a constituent-first model fail.
- **Preferring the composer over its constituents in field-first resolution.** The obvious fix for that, and precisely the tie-break ADR 0015 exists to refuse.
- **Persisting the authoring direction** on the semantic source. Symmetric with the `DataKindOrigin` that ADR 0045 deleted, and refused on the same ground.

## Decision Outcome

A model may be authored in **two directions**, and the author picks one **per root data kind**. In the **interface-first** direction the root declares the semantic interface and each child carries structure plus the physical facts that map into it. In the **constituent-first** direction, each constituent declares its own semantic interface, is referenced by name or written as a full inline body, and the root's interface is *derived* as the total union of its constituents'. The direction **is** where the interface lives, so it needs no keyword and is read off the document; a root with an interface *and* constituents with interfaces is the mix, and the mix is refused. A reference **shares**: one `semantic_sources` entry with several originators. Two constituents exposing one semantics name produce **one** fused field, never a qualified pair, and the fused name must be pooled. Publicity stops following placement and becomes a property of the unit, defaulted from whether anything references it. Nothing about the direction survives compile: the manifest carries the resolved shape and cannot tell which direction produced it.

### The direction is where the interface lives, and that is how it is detected

**No keyword, and no second source of truth.** Interface on the root and not on the children is interface-first; interfaces on the constituents and none on the root is constituent-first; **both is the mix**, refused with a typed `ValidateErrorKind::MixedAuthoringDirection` naming the root and the two competing interface sites. A `ref:` child forces constituent-first by construction, since a referenced unit carries its own interface whether the parent wants it or not. A root Dataset has no constituents, so its direction is vacuous and no check fires.

**Model validate owns the check and already has the reach.** ADR 0044 puts a validity definition with the stage that can see the fault, and this one needs the whole root subtree plus resolved `ref:` children. Model validate already resolves intra-model data-kind names by name — `check_root_relationships_endpoints` (`validation/relationship.rs:86`) builds a name set over the four top-level collections and reports `RelationshipDanglingEndpoint` — so this applies that precedent rather than adding reach. ADR 0003's "compile owns resolution" concerns catalog and physical resolution, not intra-model name lookup.

**A harmless ambiguity that must still be documented.** A root with an interface whose constituents are all interface-less is indistinguishable from a constituent-first root with nothing to derive from. Both compute the same surface, so the diagnostic is written knowing the case exists rather than trying to separate it.

### A reference shares, and ADR 0026's refusal falls in one direction only

**One entry, several originators.** A unit referenced by two composers is one `semantic_sources` entry, and ADR 0045's `originators` becomes genuinely plural: its "deliberately temporary check" holding the list to length ≤ 1 is **deleted rather than relaxed**, which is the outcome 0045 reserved it for — "the same data kind being referenced and reused as a shared semantic source by more than one parent". Nothing new is built in the manifest; the flattened DAG whose sharing was "structurally permitted but not yet reachable" becomes reachable.

**ADR 0026's nesting matrix survives references untouched, which is why the DAG objection dissolves.** 0026 refused shared children because "scopes become a DAG requiring normalization semantics that do not exist". The matrix itself is not a scope rule: it is enforced by *which child fields exist* — `GrainsetBody` carries `datasets` / `unionsets` / `joinsets` and no `grainsets`, `UnionsetBody` no `unionsets`, `JoinsetBody` no `joinsets` — so same-variant self-nesting stays unrepresentable when those fields hold references instead of inline bodies. What sharing actually costs is the temporal cascade, paid explicitly below. Mixed-variant cycles are already ADR 0045's acyclicity check over the operand edges.

**So the supersession is scoped, not wholesale.** ADR 0026's `DataKindRef` retirement and its "`Nested*` forms carry structural fields only" clause are reversed **for constituent-first**; every other clause — inline-only within interface-first, no self-nesting, the matrix as parse-time unrepresentability, `deny_unknown_fields` as the mechanism — stands as the interface-first direction's rules. Scoped in this body rather than by flipping 0026's frontmatter, per the precedent ADR 0045 set over 0025 and ADR 0037 over 0031.

### The constituent-first root interface is the total union, and the root adds only what no constituent can state

**Total union, no narrowing.** Every semantics any constituent exposes reaches the composer's surface. A narrowing list would be a second statement of the vocabulary and would drift from the derivation; curation is a request-side concern. This makes ADR 0034's `ComposedSemanticInterface` and the authored `SemanticInterface` stop being a dual for one root — one direction produces the first, the other the second.

**The root may author a closed, disjoint set: composition-level metrics and composition-level filters**, under a key that cannot read as an interface declaration. The case that earns it is real, and neither direction otherwise covers it: a Joinset over `orders` and `ad_spend` with `cost_per_order = spend / order_count`, which no single constituent can define. Disjointness is the whole safeguard — the root may **add**, never **restate**, so no fact ever has two spellings. Defining the mix semantically instead ("the root may add but not restate") was available and refused: it dissolves the exclusivity rule and is one review lapse from re-admitting the drift.

### Publicity is a property of the unit, and its default is derived from reference

**Placement stops carrying it.** Under constituent-first every referenced constituent sits at the root, so "top-level means public" would make every constituent a query target — and field-first resolution would then find the leaf *and* the composer as candidates for the same semantics, with ADR 0047 making plan-time selection "a mask test over a flat list" in which both masks have the bit set. ADR 0015 refuses to break that tie and explicitly rejects "explicit-shadows-implicit precedence", so the naive default would make most field-first requests against a constituent-first model **refuse**.

**So the default is derived: referenced by a composer ⇒ private, unreferenced ⇒ public**, computed at model validate, with an authored `public:` overriding in either direction. This reproduces today's behaviour exactly — a unit that is only a constituent is not a query target — and it keeps publicity on the *unit*, which is what ADR 0045's single `public` bit per entry can hold. Letting the *reference* carry it was refused on that same fact: under sharing, two parents disagreeing about one entry has no representable outcome.

### Two constituents exposing one semantics fuse, and the fused name must be pooled

**One field, never a qualified pair.** ADR 0034's "colliding names on a Joinset surface exist only as `constituent.name`" is superseded, and `constituent.name` is deleted outright — not a field on any composed surface, and not an address in any expression or request. The project's own language already says so: `CONTEXT.md`'s **Occurrence** calls all occurrences of one name "symmetric contributors to the same logical semantics, not competing definitions", its **Composed surface** already makes fusion primary and qualification the fallback, and ADR 0027 gives one name one meaning model-wide. 0034's rejected-alternatives list rejects "uniform name unification" by name, so this is a supersession rather than a reading — of that clause only; the rest of 0034, including the `Origin` axis and NULL-fill belonging to Unionset alone, is untouched.

**The fused name must be pooled, and this makes the fuse well-typed by construction.** Shape lock guarantees that two occurrences agree on every field but `expr` — but only through the pool, because ADR 0039 records the cross-DataKind occurrence-conflict merge as "deferred and unbuilt". So a semantics exposed by more than one constituent must appear in the root `dimensions:` / `measures:` / `metrics:` pool, and two constituents declaring one name **inline** is a hard error naming both sites. ADR 0040 set this precedent verbatim with `KeysetMemberNotPooled`: an inline-only name has no shared identity, and silently discarding what the author wrote is the failure class that rule exists to retire. The check is model validate's, on ADR 0044 grounds and with the reach shown above.

**The override is structured and never dotted.** Where the default fuse elects wrongly, the author elects a constituent on the composer's own site. A dotted address would breach ADR 0027's "unqualified references only, no aliases or dotted names" — the very rule the fuse is justified by — so the spelling is a structured key, not `orders.country`.

**The fuse arithmetic is handed forward, deliberately.** Which value a fused field takes per variant — origin-selected for Unionset and Grainset, `coalesce` in traversal order for Joinset with the argument list narrowed by the derived join type, and metrics re-evaluated over fused inputs rather than coalesced — is a separate decision, as is the override's key name. This ADR rules that the fuse happens, that it yields one field, and that pooling is its precondition. Nothing here depends on the arithmetic.

### Temporal shape cascades in one direction and is forbidden in the other

ADR 0026 cascades `extras.temporal.<shape>` to descendants and forbids `grain:` at complex levels. **The cascade survives in interface-first and is forbidden in constituent-first**: a shared constituent under two parents would otherwise have two effective shapes, exactly the normalization problem 0026 named. A unit that may be shared states its own shape; a unit that cannot be shared loses nothing by inheriting.

The corpus pays nothing: all five nested datasets in `acme_eu_ad_platform_v3.yaml` already author their own `temporal.events` **and** `grain: day`. Another decision keeps ownership of how a composer's shape is *derived*; this rules only whether a parent may *supply* one.

### Keysets are authored in one direction and derived in the other

**Interface-first keeps today's shape** — per-site `is_key:` on the root's interface (ADR 0039). **Constituent-first derives them**, and a constituent-first root may not author `is_key:` anywhere, since that would be the mix. This is what finally gives ADR 0039's intersection rule an authorable input, and it leaves the arithmetic to a later decision.

One interaction is worth recording so it is not rediscovered: the revised base schema mints a keyset id over `(semantic_source_id, keyset name)`, so a derived composed keyset seeds on the **composer's** id and is therefore distinguishable from the constituent keysets it was derived from.

### `semantic_mapping` stays Dataset-only; only its key-space moves

**This clause is a confirmation, not a change.** `semantic_mapping` already lives on `LeafExtras`, and `ComplexExtras` has no slot for it (`base.rs:151`, "per R-6, `catalog` / `storage` / `semantic_mapping` are leaf-only"). A complex data kind never declares a mapping in either direction.

**What moves is what the keys name.** In interface-first a nested Dataset's mapping keys name the *parent's* interface semantics; in constituent-first a Dataset's mapping keys name its *own*. `auto` keeps its meaning — 1:1 to a same-named physical column — in both.

### The `Nested*` type set collapses into one form plus two fields

Once publicity is a field and a constituent-first inline body carries a full interface, `NestedDataset` and `Dataset` are the same type. The `Nested*` set exists only to make an interface unrepresentable on a child, which is the interface-first rule and not a universal one. So there is **one form per variant**, carrying `semantic_interface: Option<SemanticInterface>` and `public`, where `None` is an interface-first child and `Some` a constituent-first one — the direction marker *is* the type, the mix is one comparison rather than a walk, and the silent-discard hole closes because an interface on a child stops being an unknown key.

**This supersedes ADR 0025's behavioural axis (`Public` / `Nested`).** Its structural axis (`Simple` / `Complex`), its sealing, and its read-only-accessors-only rule are untouched.

### Only a top-level unit is addressable

A `ref:` child, a root-level relationship endpoint, and a request's `from` may each name a **top-level** unit only. An inline body has a name — for identity, since the name seeds its `EntityId` under ADR 0045, and for diagnostics — but nothing may reference it. This keeps the reference target space identical to the namespace ADR 0045 already reasoned about, and it means ADR 0029's Joinset-scoped shadow needs no rule here: a shadow is scoped to the Joinset that declares it and never touches a member's own resolution, so a shared member is unaffected.

**ADR 0045's unmet obligation becomes a precondition of constituent-first, restated rather than re-decided.** `collect_duplicate_data_kinds` indexes exactly the four top-level collections, so a nested unit may today duplicate another unit's name with no diagnostic — and because the name seeds the id, two same-named inline bodies collapse into one entry, which is 0045's own "silent wrong answer rather than a diagnostic". Widening the duplicate-name check to nested units is owed before inline bodies are safe under either direction.

### The direction does not survive compile, and no epoch moves

Compile needs the direction to compute the interface; the artifact does not keep it. The manifest structure is **unchanged and serves both directions**, because both produce the same resolved shape. ADR 0045 deleted `DataKindOrigin` on exactly this ground — "implicitness is not a planner fact", nothing in the resolution path branches on how an entry came to exist — and ADR 0047 refused a persisted provenance carrier on the same one. So **no `MANIFEST_EPOCH` move is owed by this ruling** — stated because the model-side type changes above will make a reader ask.

The consequence, accepted: a manifest cannot be diffed back to the document shape that produced it, and a diagnostic about the direction is only available while compile still holds the model — the same timing argument ADR 0045 used to retire `DataKindOrigin`.

### Amends

- **0025:** the behavioural axis (`Public` / `Nested`) is superseded by one form per variant carrying `semantic_interface: Option<SemanticInterface>` and `public`; the structural axis, sealing and read-only accessors stand.
- **0026:** the `DataKindRef` retirement and the "`Nested*` forms carry structural fields only" clause are reversed for constituent-first; every other clause — inline-only within interface-first, no self-nesting, the matrix as parse-time unrepresentability, `deny_unknown_fields` as the mechanism — stands as the interface-first direction's rules; the nesting matrix survives references untouched, so same-variant self-nesting stays unrepresentable when those fields hold references instead of inline bodies; the `extras.temporal.<shape>` cascade survives in interface-first and is forbidden in constituent-first.
- **0034:** On the `constituent.name` clause only: `constituent.name` is deleted outright, and two constituents that expose one name give one fused field, never a qualified pair, and the fused name must be pooled.
- **0039:** a constituent-first inline body carries a full interface, and `NestedDataKindKeynessIgnored` retires, because keyness on a child stops being an ignored key. Interface-first keeps today's shape, per-site `is_key:` on the root's interface; constituent-first derives the keysets, and a constituent-first root may not author `is_key:` anywhere, since that would be the mix.
- **0045:** a unit referenced by two complex data kinds is one entry with several `originators`; the temporary length ≤ 1 check is deleted rather than relaxed, so sharing becomes reachable.

### Consequences

**What changes in `crates/`, recorded for the implementation effort and not done here.** `NestedDataset` / `NestedGrainset` / `NestedUnionset` / `NestedJoinset` and the `NestedDataKindRef` / `AnyDataKindRef` view arms collapse into one form per variant; `SemanticInterface` becomes `Option` on that form; a `public` field arrives on the authoring surface; the child-list fields gain a reference arm; `retired.rs`'s `NestedDataKindKeynessIgnored` warning retires, because keyness on a child stops being an ignored key; `ValidateErrorKind` gains the mixed-direction and unpooled-fusion variants; `collect_duplicate_data_kinds` widens to nested units (ADR 0045's obligation).

**Four decisions are handed forward with owners.** The fuse arithmetic and the override key — its own session. Composed keyset derivation. Implicit Joinset enumeration. Composer temporal shape derivation. The expression-edge direction that an intra-unit semantics-level expression depends on.

**`CONTEXT.md` gains one term and amends three.** **Authoring direction** is new; **Nested data kind**, **Semantic interface** and **Composed surface** are amended — the last to drop the constituent-qualified fallback this ADR deletes.

**The corpus is unaffected today** and gains a second legal shape. Every existing document is interface-first by construction, since no child can currently carry an interface.

## More Information

**Renumbered 0048 → 0053.** This ADR merged as `0048-…`; a second ADR then merged as a second `0048-…` without the renumber it was notified for, leaving two files on one number. This one moved because nothing cited it — and `docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md` is what every "ADR 0048" reference in the repo already means. The renumbering edited no clause; only the number changed, so the bound on in-place ADR edits was not engaged.
