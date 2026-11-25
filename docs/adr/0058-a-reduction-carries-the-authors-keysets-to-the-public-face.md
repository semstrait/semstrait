---
number: 58
title: A reduction carries the author's keysets to its public face, and only a multi-source Dataset gets an implicit Unionset
status: accepted
date: 2026-08-26
source:
- CONTEXT.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0039-keyness-is-a-per-site-attribution.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0048-composite-keysets-are-authored-inside-a-boundary.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0051-entity-ids-are-derived-from-the-natural-key.md
- docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0056-composed-temporal-shape-is-derived-at-compile.md
links:
- target: 45
  kind: amends
- target: 49
  kind: amends
- target: 65
  kind: amendedby
- target: 70
  kind: amendedby
---

# A reduction carries the author's keysets to its public face, and only a multi-source Dataset gets an implicit Unionset

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): ADR 0049's successor invariant *"exactly one override row per `(semantic, data_kind)`"* is restated as one signature per `(semantic source, definition)`, and made total; the carry-over invariant reads against signatures for the same reason, and its rule is unchanged.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): a leaf's id is `mint(Dataset id, physical-source id)`, and its name `<Dataset name>_<physical-source name>` is a label that may repeat. The varying segment, its segment joining and the refusal of two leaves with one name retire. Its open question "whether a glob is one physical source or several" is resolved as one source per base path and format. Its open question on the `name` field is resolved as a label.

## Context and Problem Statement

When an authored Dataset names several physical sources, compile reduces it to one leaf semantic source per source and composes them with an implicit Unionset. Whether the keysets the author declared reach that Unionset was left open on purpose: ADR 0049 created the gap and named it rather than closing it — *"A synthesised Unionset has no author, so one route is closed by form and the other has no actor."*

## Considered Options

- Carry the author's keysets over to the Unionset and every leaf, and wrap only a multi-source Dataset (chosen).
- **Refuse a keyset route, and let the reduction's Unionset hold an empty roster.** What ADR 0048 and ADR 0049 read as the forced outcome, needing no supersession. Refused because it silently removes `on: auto` and Grainset composition from every multi-source Dataset, and because ADR 0049 already recorded the cost as load-bearing rather than acceptable. It also makes a model's join behaviour a function of how many files a glob matched, the hazard 0049 named.
- **Admit carry-over as a third mechanism beside derivation and boundary.** The obvious framing, and the one ADR 0049 flagged as the specific risk, since ADR 0048 opens by recording that conflating its two mechanisms *"is what made every earlier framing of this question wrong"*. Refused because it is not needed: the actor exists and the boundary contains the claim, so naming a third route would assert a distinction the arithmetic does not support.
- **Write the keyness rows on the Unionset only, and leave the leaves bare.** Fewer entities — one keyset per name instead of one per source — and no claim about a single file. Refused on ADR 0048's uniformity clause: a flat collection in which only some sources answer *what identifies a row* puts the operand walk back into the consumer, which is what flattening exists to remove.
- **Carry the `N > 1` fix in this ADR instead of repairing ADR 0054.** The route this repo normally takes for a merged ADR, and a bound points at it. Refused on the deciding human's call: a wrong bound left standing in the clause a reader reaches first, with the fix a document away, is the second-place-to-look failure this repo spent 46,284 deleted lines removing. A defect in the writing is repaired where it is written; a decision is revisited in a new ADR.
- **Raise a `CompileError` when the Unionset's roster is empty while its leaves' are not.** Sharper and consistent with ADR 0056's two-site pattern. Refused because the fault is compile's own, so the diagnostic would name a defect no author can repair; the invariant belongs to `validate()` alone.
- **A hard error, or a warning, when a Dataset declares no keyset at all.** It would make the loss of `on: auto` visible at compile instead of at plan time. Refused on the earlier base-schema design's default-and-optional keyset clause, on its *"Nevertheless, keyset is optional"*, and on an earlier ruling, which refused this exact warning for the Grainset analogue.
- **Salting a colliding leaf id and retrying.** It keeps a short, readable discriminator and never fails. Refused because it breaks ADR 0045's natural key, leaving two entities with one name and diagnostics that cannot tell them apart.
- **The whole locator as the discriminator.** Unique by construction with no reasoning about wildcard positions. Refused because it puts a full path inside an entity name, and names are what diagnostics show.

## Decision Outcome

**The keysets the author declared reach the reduction's Unionset by carry-over**, and they reach every leaf as well. Carry-over is not a third keyset mechanism: it is ADR 0048's **boundary** route with its actor identified. The author of the Dataset is the author of the Unionset, because the Unionset is what the authored name resolves to after the reduction. ADR 0049's premise is what fails: the actor was never absent — the reduction divides one authored data kind into several manifest entities, and the author's claims travel with the name.

**The reduction only happens when it has something to compose.** An implicit Unionset is created for a **multi-source** Dataset alone. A Dataset over one physical source stays a leaf, keeps the authored name, and holds its keysets directly, so nothing carries over and no wrapper exists.

### The actor is the Dataset's author, and ADR 0053 says where the claim was written

**Keyness is authored on the root's interface, not on a child.** ADR 0053 rules that a root data kind is authored interface-first or constituent-first and never both, and that *"interface-first keeps today's shape — per-site `is_key:` on the root's interface (ADR 0039)"*, while a constituent-first root *"may not author `is_key:` anywhere, since that would be the mix"*. Interface-first is the only direction the corpus has. So a multi-source Dataset that is a root carries its own interface, and the `is_key:` sites on it are the author's statement about **that name**.

This is why the question's own framing — *"does the leaf's authored keyset propagate to the synthesised Unionset"* — asks the question upside down. A leaf of a reduction authors nothing; it is compile's product. The claim was made once, on the root, for the whole storage entry. The Unionset is the entity that name resolves to, so the claim lands there **natively** rather than being propagated up from a branch.

**The direction matters because only one of them would be a strengthening.** An author writing `paths: [a.parquet, b.parquet]` with a keyset `[campaign_id]` claims that no two rows of the stack share a `campaign_id`. Carrying that to the Unionset restates the claim on the entity the claim was about. Carrying it down to each leaf is *weaker* than what the author said, since a subset of rows unique on a group is still unique on it. Nothing anywhere is strengthened, which distinguishes carry-over from the fabricated identity claim ADR 0048 refused when it superseded ADR 0039's branch-intersection rule.

**ADR 0056 already decided the symmetric case, so this is a precedent's second use rather than a new mechanism.** For the temporal grain it ruled that *"the reduction carries over what the author wrote instead of deriving anything new"* and that *"the rule needs no exception for the reduction"*, then re-scoped that rule to bind the author alone. The same authored-versus-derived split runs through ADR 0048 for keysets and ADR 0052 for cardinality. Keysets are the third instance and the last one the reduction was missing.

### Homogeneity collapses the boundary, so the carried keyset is admissible by construction

ADR 0048's Unionset boundary is *"the cartesian product over **all** branches, choosing one keyset per branch"*, each element the union of the chosen groups. Under ADR 0049 a leaf carries exactly one `semantic_mapping`, so *"every branch of a synthesised Unionset exposes the same semantics"* — the branches hold identical keyset rosters. Choosing keyset `K` from every branch therefore yields the element `K ∪ K ∪ … = K`.

**So every carried keyset is itself a boundary element.** Compliance is not argued, waived or excepted; the ceiling contains the claim by arithmetic. That is what makes this the boundary route rather than a fourth thing beside it, and it is the only distinguishing fact ADR 0049 said was available — *"that homogeneity is the only distinguishing fact available: if it does not license propagation, nothing does."*

**The grain-groups withdrawal is distinguishable, and the distinguishing fact is authorship of the name.** ADR 0048 withdrew grain groups because *"an implicitly-formed Unionset derives no keyset, so it would have no identity for anything downstream to join to — the same limit this ADR records for authored Unionsets, reached where no author can be asked."* A grain group had no authored name and no authored claim: it was a compile-invented partition of a Grainset's children, so there was nothing to carry and no one to ask. ADR 0054 applied the same reasoning a third time to sub-Unionsets over a subset of a Unionset's branches, which are invented the same way. The reduction is unlike all three: its Unionset **is** the author's name, and the author already answered.

**What identifies a row of the union is the author's problem, on ADR 0048's own posture.** Two branches may carry the same values for the carried group — two monthly files both holding `campaign_id = 7` — and the composed stack is then not unique on it. ADR 0048 already settled how much that matters: a boundary *"states what the model cannot contradict and a keyset is unverified advice whose author may know their data is total where the model does not say so"*, and it accepted the same exposure for a preserved side on an outer join. Compile has no row to inspect and no better informant than the person who wrote the storage entry. The cost is disclosed rather than designed away.

### Only a multi-source Dataset gets a Unionset

**The threshold belongs to ADR 0054 and is not restated here.** Its consequence clause rules that the synthesised Unionset exists at `N > 1`, and that clause was **repaired in place**: it read `N ≥ 1` and argued for always wrapping, which contradicted its own subject, since a *multi-source* Unionset has no instance at one source. The repair is recorded in ADR 0054's own text rather than carried here, on the deciding human's call, so the rule is read once and in one place. ADR 0049 routed the wrapper's existence there, so no third document is involved.

**What this decision owes is the keyset consequence of that threshold, and it is nil.** Under carry-over the author's keysets sit on whatever entity holds the authored name — the leaf at one physical source, the Unionset at several. A relationship naming that Dataset therefore resolves against the same keysets at every match count. So the threshold moves bytes rather than meaning, and the hazard ADR 0049 recorded — *"a second matching file [is] a change in join behaviour with no change to the model"* — is closed by carry-over rather than by where the wrapper starts.

### The rows land on the leaves and on the Unionset

Keysets are materialized, not derived on read (ADR 0048), as `semantic_overrides` rows keyed by `data_kind_id`. Carry-over writes them **on every leaf and on the Unionset above them**.

**Uniformity is ADR 0048's own instruction, not a convenience.** Its rules *"apply uniformly to implicit sources … a flat list where only some sources answer 'what identifies a row' forces the planner back into a walk"*. A reduction leaf is an implicit source, so a leaf that stated nothing would be exactly that hole. On the leaves the claim is sound because it is weaker than the authored one.

**The carried name is unchanged, and no collision is reachable.** ADR 0048's originator-qualified naming applies to a *derived* keyset; nothing here is derived, so the keyset keeps the name the author wrote. Its hard error for *"a derived keyset name colliding with an authored one on the same composite"* therefore has no subject on a reduction Unionset, which holds carried names only.

**One authored keyset name becomes one keyset entity per source, and that is ADR 0048's accepted cost rather than a new one.** The id is minted over `(semantic_source_id, keyset name)`, so a Dataset over four physical sources yields five entities for one name — four leaves and the Unionset. ADR 0048 chose that seed knowingly, recording that 0040's rejected *"seeding the id per DataKind"* and its cost — *"one roster entry per source per name"* — are *"now accepted deliberately, since a source-scoped seed makes cross-data-kind dedup impossible by construction rather than merely unimplemented"*.

**An absent declaration writes no row at all**, not an empty list: ADR 0039's `keysets: []` refusal stands, and ADR 0048's rule that an override's `keysets` *"replaces the default and never unions with it"* is untouched.

### Absence is legitimate, and the one wrong artifact left belongs to compile

**Nothing obliges an author to declare a keyset, and the earlier base-schema design says so twice.** Its first clause makes a Simple data kind's keysets *"default and assumed definition, but Optional, for all body types"*. Its second is the clause that reads like an obligation and is not: *"Unionset - requires explicit definition for his keysets on the unionset level … otherwise there is no reliable way to automatically derive keyset from the underlying "data kinds", unlike joinsets. **Nevertheless, keyset is optional.**"* ADR 0048 says that clause *"stands"*, and what stands is a statement about the only available **route**, not a requirement to take it.

**So a Dataset that declares no keyset produces a Unionset with an empty roster, and that is correct.** An earlier ruling settled the matching case and refused a diagnostic in terms: *"A Grainset whose grains carry no keyset simply offers no rollup composition, each grain stays individually routable, and no diagnostic is owed — this session's proposed warning was refused on that ground."* ADR 0048 carried the same outcome forward as *"a legitimate model [that] needs no diagnostic"*. The consequence is that some compositions are unavailable, never that a model is rejected: `Auto` basis resolution and a Grainset's cross-grain Joinsets read guaranteed keysets only, so with none present there is no candidate to enumerate.

**ADR 0048's *"a composite with an empty keyset roster is a silently wrong artifact"* is read narrowly, and the reading gives it a subject here.** The clause sits inside that ADR's handoff paragraph, whose subject is the legacy single-leaf guard. It condemns a roster that **a rule should have filled and did not** — a composite whose operands offer keysets and whose own roster comes out empty. It does not condemn a roster that is empty because the author declared nothing, which no rule was ever going to fill.

**Carry-over is such a rule, so it acquires one invariant: a reduction's Unionset carries every keyset its leaves carry.** Absence then has two causes and they are not alike. The author declared nothing, so the leaves are bare too — legitimate. Or the leaves hold keysets and the Unionset does not — carry-over failed.

**The invariant is `validate()`-only, and deliberately not a compile-stage gate.** A carry-over failure is a fault in compile, so no author can act on the diagnostic and an author-facing `CompileError` is the wrong instrument. An earlier reframing already names this class — *producer-unreachable, validate-only, legitimate* — and ADR 0042 makes `validate()` raise `CompileError` anyway, so no second enum is involved. ADR 0056's own test for a restatement is met on the value side: keysets are `pub`, a decoded artifact can desync what no producer could, and the values are load-bearing for basis derivation and Grainset composition. What differs from ADR 0056 is the site count: its rules are author-reachable and so run at two sites; this one is not, so it runs at one.

**No boundary check runs for this form, and the reason is recorded so nobody adds one.** Compliance is structural, as the arithmetic above shows, so ADR 0048's boundary diagnostic could never fire on a reduction Unionset. ADR 0048's check stays exactly where it has a subject: an **authored** composite whose declaration the operands contradict. A check that always passes is close to the fault ADR 0043 names when it says *"an unwritten carrier is a defect, a reserved optional is a contract, and only a pinned meaning for `None` tells them apart"* — it reads as protection and offers none.

### A leaf's name is the Dataset's name and the varying segment of the locator

Carry-over writes rows against each leaf, so each leaf needs an id, and under ADR 0051 an id needs a name. **A reduction leaf is named `<Dataset name>.<varying segment>`**, on ADR 0051's `<parent>.<name>` idiom.

**The varying segment is the part of the locator that distinguishes one of a Dataset's physical sources from another.** Where a glob expands to several sources it is the segment at the wildcard position, which ADR 0046 fixes as the only position expansion happens at — *"one `PhysicalSource` per expanded path, not per glob root … with no deeper recursion into the tree"*. So `paths: [2024/*.parquet]` gives `sales.jan.parquet` and `sales.feb.parquet`, while `paths: [data/*/part.parquet]` gives `sales.a` and `sales.b`. Where a storage entry does not expand, it is that entry's own last directory, and for `tables:` it is the table name.

**Uniqueness is by construction rather than by check, and that is the property being bought.** A Dataset's physical sources have distinct locators — ADR 0049 mints a physical source id from `(source_type, locator)` and nothing else — and the varying segment is what makes them distinct, so two leaves of one Dataset cannot collide. This matters because a name collision is an id collision under ADR 0051, which would merge two leaves into one entry: a wrong artifact, not a rejected compile. **Minting a colliding id and salting for a retry was considered and refused**, because it leaves two entities sharing one name, which contradicts ADR 0045's natural key — *"within the `data_kind` namespace, a name is unique, and the name is the id seed"* — and makes a diagnostic that names the leaf ambiguous. Making the name unique is strictly better than making the id unique.

**ADR 0054's collision rule reaches these names unchanged**: a derived name colliding with an authored data-kind name is a hard error.

### The wrapper is implicit and public, which ADR 0045's letter forbids

ADR 0045 states that *"an authored nested unit is private, and so is every implicit unit."* A reduction's Unionset is compile-synthesised and must be public, or the authored name resolves to nothing a request may put in its `from`. **The letter is corrected in one clause: privacy tracks a derived name, not implicitness.**

The intent survives whole, and `CONTEXT.md` already reads the corrected way — **Public semantic source** is *"a semantic source a request may name directly in its `from`"*, with every other one private. What ADR 0045 was protecting is stated in its own next sentence: that a request never resolves to *"an entry the author never wrote"*. A reduction Unionset is the entry for a name the author **did** write, so it is the one implicit unit for which publicity is not a promotion. Every leaf beneath it is private, because every leaf has a derived name.

### Amends

Each supersession is scoped in this body rather than by editing the older ADR, per the precedent ADR 0037 set over ADR 0031 and ADR 0050, ADR 0051 and ADR 0056 followed.

- **0049:** the premise that the synthesised Unionset has no actor, which is the clause that routed this question to 0058, is superseded: "the actor was never absent — the reduction divides one authored data kind into several manifest entities, and the author's claims travel with the name." Its conclusion — that the route was unruled — was correct; its reason was not. Nothing else in 0049 moves: the leaf's single physical-source reference, the `1:N` reverse edge, the successor invariant of exactly one override row per `(semantic, data_kind)`, `mode: All`, no null fill, the two refusals of zero, and the files-and-tables restriction all stand.
- **0045:** the clause that every implicit unit is private is narrowed to units with derived names: privacy tracks a derived name, not implicitness, so a reduction's compile-synthesised Unionset, which holds the authored name, is public.

**One ADR 0054 clause is repaired in place rather than superseded or corrected here**, a departure from this repo's usual route recorded as such. Its consequence clause on the synthesised Unionset read `N ≥ 1` and now reads `N > 1`; the bound contradicted the clause's own *multi-source* subject, so it was a defect in the writing rather than a decision to revisit, and the deciding human ruled that it be fixed where it lives so the rule is read once. **This engages the bound that a merged ADR is closed to in-place edits, and the exception is deliberate and narrow**: one bound, one withdrawn supporting argument, a correction note left in the clause, and no other clause of ADR 0054 touched. **Nothing else in ADR 0054 moves** — the enumeration rule over an authored Joinset's specification, `(S, E)` identity, left-deep canonical bracketing, the recursive left-operand naming rule, the count bound with its no-truncation posture, and the grain-pair rule for a Grainset all stand as written.

### Consequences

- **`CONTEXT.md` gains `Carry-over` and its `Implicit Unionset` entry gains a keyset clause.** The term is added because the glossary has no word for the mechanism, ADR 0056 used the phrase in prose without defining it, and it must be distinguishable from **Keyset boundary** and from derivation. Both land with this ADR so the glossary never states a rule this artifact has not ruled.
- **Two costs are accepted rather than designed away**, and both are recorded above: the re-identification of a leaf when a second physical source appears, against a `model_hash` gate that will not notice; and one keyset entity per source per name.
- **One invariant is added and no `CompileError` variant is.** A reduction's Unionset carries every keyset its leaves carry, at `validate()` only.
- The **`on: auto` failure diagnostic**: what this decision adds is the classification the human ruled while deciding it: **that diagnostic is compile's, as a stage gate, not the planner's.**
- **What this obliges in `crates/`** — recorded for the implementation effort, not done here:
  - `mint_keysets()` sits inside `seal_dataset` and scans one data kind's own dimension sites. ADR 0048 already ruled it *"replaced rather than extended"*; carry-over is one of the rules that replaces it, and it needs the reduction's leaf set, which `seal_dataset` does not have.
  - `dataset_source_ids` maps one Dataset to many ids today. Under ADR 0049 it becomes one-to-one, and under this ruling the multi-source case needs the leaf roster plus the varying segment per leaf, so the map's replacement carries a name and not only an id.
  - `Keyset` in `manifest/keyset.rs` still holds `members` and mints over `(name, sorted members)`. ADR 0048 superseded both; carry-over depends on the source-scoped seed, so this is on its path rather than beside it.
  - A compile guard at the time of the decision rejected every complex form, and ADR 0054 records it as *"the only thing standing between this ruling and a manifest that holds one"*. The same guard blocks the reduction's Unionset, so it blocks carry-over.
  - Nothing in `crates/` is changed by this decision.

## More Information

### Open questions

Four questions are named and left to their owners.

- The **`on: auto` failure diagnostic** — what happens when a declared relationship's basis cannot be derived because no keysets exist — stays open under auto-join, where a closed prep already records that it *"should absorb the outstanding `compile/seal.rs:229-241` FLAG asking the lead to ratify a relationship-resolution `CompileError` variant"*. ADR 0038 put the ladder's failure diagnostic out of scope in terms. (Its classification is ruled under Consequences.)
- **How a Grainset's grain comes to hold a guaranteed keyset under interface-first**, given that ADR 0053 puts per-site `is_key:` on the root's interface alone while ADR 0048 and ADR 0054 both have the cross-grain Joinsets read the **grains'** keysets. Recorded as open; it is not sharp enough to settle.
- **Whether a glob is one physical source or several.** The earlier base-schema design and ADR 0046 say one per expanded path at the wildcard position; the human's reading is that a glob's files are read by compute as one logical source, and a test of the physical source locator asserted a locator of `/data/*.parquet`, a glob kept whole. This ADR does not rule it, and the discriminator rule above is written to hold either way. Recorded as open.
- **The `name` field on `physical_sources`.** The earlier base-schema design's sample specifies *"Physical Source Name - computed from path or table name"*; ADR 0046 ruled the physical source's shape and did not carry it, and `PhysicalSource` holds `source_type`, `locator`, `version_ref`, `projected_schema`, `schema_fingerprint` and `provider_metadata` with no `name`. This ADR states the leaf-name derivation directly so that nothing here depends on an unbuilt field. Recorded as open.
