---
number: 45
title: Semantic sources are one flat, name-keyed, acyclic collection, and nesting stops being a type
status: accepted
date: 2026-08-14
source:
- CONTEXT.md
- docs/adr/0010-single-entityid-lane-content-derived-ids.md
- docs/adr/0025-sealed-datakind-hierarchy-with-manifest-mirror.md
- docs/adr/0026-inline-only-nesting-tree.md
- docs/adr/0029-joinset-scoped-relationship-shadow.md
- docs/adr/0040-manifest-keyset-roster-is-deduplicated.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
links:
- target: 10
  kind: amends
- target: 25
  kind: amends
- target: 33
  kind: amends
- target: 40
  kind: amends
- target: 53
  kind: amendedby
- target: 54
  kind: amendedby
- target: 58
  kind: amendedby
- target: 70
  kind: amendedby
- target: 34
  kind: supersedes
---

# Semantic sources are one flat, name-keyed, acyclic collection, and nesting stops being a type

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0053](0053-authoring-direction-is-exclusive-per-root-data-kind.md): a unit referenced by two complex data kinds is one entry with several `originators`; the temporary length ≤ 1 check is deleted rather than relaxed, so sharing becomes reachable.
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): two clauses. The `role_definition` arm spellings (`Joinset{from, to}`, `Unionset{branches}`, `Grainset{grains}`) are **shorthand rather than exhaustive**, which is what ADR 0049 assumed when it asserted a value for a `mode` no ADR had placed: `mode` rides the **Unionset** arm by ADR 0043's placement rule, while `join_type` / `cardinality` / `integrity` / `cross_filter` ride the **Joinset** arm, and no field is shared between arms. And the implicit-source naming rule gains the recursive left-operand qualification of 0054's *Identity, shape and name*: an implicit unit's name is qualified by its left operand, recursively, not by the originator alone.
> - [ADR 0058](0058-a-reduction-carries-the-authors-keysets-to-the-public-face.md): the clause that every implicit unit is private is narrowed to units with derived names: privacy tracks a derived name, not implicitness, so a reduction's compile-synthesised Unionset, which holds the authored name, is public.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): a private reduction leaf's id is minted from its Dataset and its physical source, not from its name, and its name is a label that may repeat; the natural key stands for every other entry.

## Context and Problem Statement

The manifest mirrored the model's nesting in its own composition types, and the mirror never carried ADR 0026's guarantee. It was wrong in both directions. `NestedDataKindVariant` is recursive through `UnionsetBranch.kind: NestedDataKind`, so `unionset ⊃ unionset` type-checks — under-restrictive against 0026 — while `Joinset`'s absence made a *legal* shape unrepresentable, which is the gap that chartered this decision. It approximated 0026 incidentally; it did not implement it.

The shape chosen here is deliberately the one ADR 0026 refused one layer up. That refusal stands and is not weakened (see below); what changes is that the *manifest* stops pretending to mirror it.

## Considered Options

- One flat `semantic_sources` collection, composition by reference on a polymorphic `role_definition`, and ids seeded on the name (chosen).
- **Extending `NestedDataKindVariant` with a `Joinset` arm.** The symmetric fix, and the first option considered. It deepens an already-recursive type, leaves `unionset ⊃ unionset` type-checking, and preserves a mirror that never enforced the matrix it appeared to.
- **A by-id arm on `UnionsetBranch.kind` alone.** The second option considered, and the right idea applied to one field. Once every unit needs an id-addressable home, a per-field escape hatch is strictly worse than doing it uniformly — and `RoutingUnitRef` had already grown exactly such an arm (`Synthesized`) for Grainset levels, so the mechanism was being invented twice.
- **A flattened tree with the originator in the id seed.** This session's own first recommendation. It forbids sharing by construction and would have made `originators` singular by fiat. Rejected once sharing was ruled a structural reservation rather than a prohibition: seeding on the name gets the same singularity today without closing the door.
- **A content-derived seed for a semantic source's id.** Symmetric with `mint_interface_id` and `mint_keyset_id`. It would dedup structurally identical units — desirable for a keyset roster, whose whole job is to say "these two are one shape", and wrong for sources, where two identical private branches under different parents are two different resolution paths. It also re-ids a unit on any body edit.
- **BLAKE3 for implicit compositions**, as ADR 0010 ratified. Refused because the digest is not canonical UUID text and the `EntityId` shape check is load-bearing for bit-position ordering.
- **Keeping `DataKindOrigin` for diagnostics.** Rejected on timing: compile holds the model when it diagnoses.
- **Nameless implicit sources.** Cheapest, but keying implicit entries by id alone makes the natural-key rule conditional on provenance, and an entry with no name cannot be named in a diagnostic, which is the only way an author would learn that an enumeration went wrong.
- **An unqualified derived name** from the relationship or the joined member names. Collides via ADR 0029's deliberate shadow reuse and via a literal authored `a_b`, and because the name seeds the id, the collision merges entries silently.
- **A `DuplicateRelationshipName` rule to make the unqualified form safe.** It would have to admit ADR 0029's shadow as a legal exception — the exact case that generates the collision.
- **A nesting-legality or self-nesting check in the manifest.** Producer-unreachable, and a restatement of a parse-stage rule.

## Decision Outcome

The manifest's composition types collapse into a single `semantic_sources` collection in which **every** unit — an authored top-level data kind, an authored nested child, and a compile-synthesised implicit unit alike — is an ordinary entry referenced by `EntityId`. `NestedDataKind`, `NestedDataKindVariant`, `UnionsetBranch.kind` and `RoutingUnitRef` retire with it: composition is expressed by reference on a polymorphic `role_definition` (`Joinset{from, to}`, `Unionset{branches}`, `Grainset{grains}`), not by a recursive variant enum. The resulting shape is a **flattened DAG** — acyclic, addressed by id, with sharing structurally permitted but not yet reachable — and what keeps it navigable is not the type system but a natural key: within the `data_kind` namespace, **a name is unique, and the name is the id seed**.

### Identity: the name is the key, and the key is the seed

**An authored id is kept where one exists; everything else mints UUIDv5 over the name through the `data_kind` namespace.** `EntityNamespace::DataKind` (label `data_kind`) already exists in `identity.rs` and has **zero uses and no minter** — a dormant lane this ruling wakes rather than invents, which is ADR 0040's precedent verbatim ("nothing new is invented in the identity lane"). Minting routes through the existing `mint_entity_id` primitive.

**ADR 0010's BLAKE3 clause is superseded, on carrier grounds rather than on preference.** ADR 0010 carries two derivations: UUIDv5 over a per-type namespace for sources, bindings, interfaces, implicit unionsets and `model_id`, and "BLAKE3 over the canonical form for implicit compositions". The second contradicts 0010's own lane rule once an implicit composition is an ordinary `semantic_sources` entry: `EntityId` is canonical UUID text with its shape checked at minting sites, the manifest check that each id is canonical UUID text validates exactly that form, and ADR 0036 (via the bit-position ruling) made that check load-bearing — it is what makes the ordering key total. A BLAKE3 digest is not canonical UUID text. Nothing moves in code: `MAX_IMPLICIT_COMPOSITION_DEPTH`, `MAX_IMPLICIT_ENUMERATION_COUNT` and `ImplicitId` are all grep-absent from `crates/`, so the BLAKE3 lane was ratified and never built.

**Seeding on the name and not on content is what makes the name a key rather than a label.** A content-derived seed would re-id a unit whenever its body is edited, orphaning every reference to it from the same artifact — and it would buy nothing, because ADR 0041 already pins `model_hash` over the whole model with serde totality, so content change is detected at the artifact level and addresses a different object. The accepted cost: renaming a nested unit re-ids it. That is the correct direction, since the name *is* its identity.

### Implicit sources are named after their origin of compilation, qualified by the originator

**An implicit source's name comes from what caused it.** Where a declared model relationship produced it, the name is taken from that relationship; where it is an implicit relation over partial results inside a Grainset, the name joins the participating data kinds' names with underscores. Both are then **qualified by the originator's name** — `<originator>.<relationship>`, `<originator>.<a>_<b>`.

**Without the qualification the derivation is not unique, and under the seeding rule a name collision is an id collision.** Two independent facts break the bare form. Relationship names are not unique: there is no `DuplicateRelationshipName` in `ValidateErrorKind` — the roster checks duplicate data-kind names, duplicate shared-Semantics names and duplicate entity ids, and nothing else — and ADR 0029's Joinset-scoped shadow *intends* one name to exist both at model level and inside a Joinset, so a rule forbidding the reuse would have to carve out the very case that produces the collision. And an underscore-join can collide outright with an authored data kind literally named `orders_customers`. Because the name seeds the id, either collision silently merges two distinct resolution paths into one entry — the failure is a wrong plan, not a rejected compile. Qualifying by the originator makes uniqueness fall out of a name that is *already* unique.

### Nested-name uniqueness is owed by model validate, and is not met today

The natural key holds for top-level data kinds and **not** for nested ones. `collect_duplicate_data_kinds` (`model/builder/dedup.rs`) indexes exactly four collections — `datasets`, `grainsets`, `unionsets`, `joinsets` — all top-level; nested units are absent from the index, so a nested unit may today duplicate another nested unit's name or a top-level unit's name with no diagnostic. Compile's own `guard_no_name_collision` does not close this: it is a *cross*-namespace guard (dimension / measure / metric against dataset), and among the data kinds it walks only `datasets`.

**The rule belongs to model validate, not to compile.** Under ADR 0044 a validity definition lives with the stage that owns it, and the duplicate-name index is that stage's rule; compile restating it would be the duplicated-vocabulary defect 0044 exists to prevent. So this is recorded as a **load-bearing obligation on the duplicate-name index** — widen the index to nested units — and not as a compile check. Until it lands, compile's minting will collapse two same-named nested units into one entry, which is a silent wrong answer rather than a diagnostic.

### Two edges: composition downward, provenance upward

**`role_definition`'s operands and `originators` are distinct edges, not one edge written twice.** For an authored nested unit they are exact inverses — the parent lists the child as an operand, the child names the parent. For a compile-implicit unit they diverge: an implicit binary joinset's operands are the two surfaces being joined, while the thing whose specification caused it to exist is the authored Joinset. That divergence earns the second field; the inverse agreement in the authored case becomes a consistency check, not a reason to denormalize one away.

**`originators` is the nearest authored ancestor, and there is exactly one today.** Empty means "authored top-level, no originator" — never "unknown". The list stays plural as **reserved headroom** for the case that makes it real, which is the same data kind being referenced and reused as a shared semantic source by more than one parent; that is not permitted now. Following ADR 0043's headroom discipline, the reservation is *asserted* rather than assumed: a deliberately temporary check holds the list to length ≤ 1, to be deleted when reuse lands rather than relaxed after the drift is discovered. Under the naming rule the singularity is also structural — a child's name is derived from its originator's, so a second originator has no spelling.

### `public` is the only flag, and `DataKindOrigin` retires

**`public` means this semantic source is a direct query interface — one a request can name in its `from`.** An authored nested unit is private, and so is every implicit unit; the base schema's remark that a public Joinset's compositions are queryable describes the *authored* unit's reach, resolved through its private entries, not a promotion of those entries into query targets. Were it otherwise, an authored name would resolve to an entry the author never wrote.

**`DataKindOrigin` is deleted rather than re-homed, because implicitness is not a planner fact.** The manifest exists so that planning "needs no further lookup or I/O"; nothing in the resolution path branches on whether a human wrote an entry. The diagnostic argument for keeping it fails on timing, not on principle: a compile diagnostic is raised while compile still holds the model, so it never needs to read provenance back off the artifact. What survives of the distinction is what is used — `public`, plus `originators` for the shape of the composition.

**This supersedes ADR 0025's manifest half and leaves its authoring half untouched.** 0025 recorded that "the manifest deliberately carries a second representation: a flat closed `DataKindVariant` enum plus `DataKindOrigin`, accepted as an intentional dual for serialization and id-keyed storage." Both halves of that dual go — the enum to the polymorphic `role_definition`, the origin tag to nothing — while the sealed two-axis `DataKind` trait hierarchy in the model is unaffected, since nothing here changes the authoring surface. The supersession is stated and scoped in this body rather than by flipping 0025's frontmatter, per the precedent set when ADR 0037 superseded ADR 0031 on carrier shape only. It lands closer to 0025's own rejected alternative, "carrying one representation end-to-end instead of accepting the model/manifest dual" — the dual dissolves not because it was wrong to accept but because the manifest stopped mirroring a taxonomy it never enforced.

### ADR 0026 stands, and the nesting matrix is enforced exactly once

**0026's mechanism is live, structural, and in the model.** `GrainsetBody` carries `datasets` / `unionsets` / `joinsets` and no `grainsets` field; `UnionsetBody` no `unionsets`; `JoinsetBody` no `joinsets`. Same-variant nesting is unrepresentable because there is no field to hold it, and `deny_unknown_fields` on `DataKindBase` turns the missing key into a parse error — precisely 0026's stated design. `allowed_child_variants()` is a read-only accessor and nothing enforces with it, per ADR 0025's "read-only accessors only". The cross-variant matrix, including `joinset` under both `grainset` and `unionset`, is therefore readable from live code and does not rest on any separate document.

**So the matrix is enforced once, at parse, on the authored document, and the manifest owes no nesting-legality check.** This is ADR 0042 applied to shape: validity is constitutive of the stage that can produce the fault, and the stage that can produce illegal nesting is parse. A self-nesting or matrix check in the manifest would be producer-unreachable and would restate a rule another stage already owns.

### Acyclicity is checked, over the operand edges

**Flattening removes the type system from the termination argument, so the rule becomes a check.** A unit may depend on another; none may depend on itself, directly or transitively. The check walks the **operand** edges, because that is the graph a consumer traverses and so the one in which a cycle fails to terminate; `originators` acyclicity follows from the naming rule, since a child's name is derived from its parent's and a cycle has no spelling.

**One `CompileError` variant, raised by the stage that assembles `semantic_sources`, documented in that stage's own module docs.** No roster, no registry, no `G`-number — ADR 0044 settled that the variant set plus the raising stage *is* the enumeration. The model side already has `CyclicSemanticsReference` as a naming precedent for the diagnostic.

**Expression `depends_on` cycles are out of scope here.** The base schema describes that edge as pointing at the *parent* while ADR 0037 records flattened child edges, and an acyclicity rule cannot be stated before the edge has a direction. That direction is the manifest-lifecycle-and-expression-edge decision's, not this one's.

### The per-unit keyset carrier dissolves, and gains no origin marking

ADR 0040 handed forward two items rather than swallowing them: the **per-unit keyset carrier** — implicit keysets "created per branch and per join hop, stored and referenced in this common roster rather than inlined per unit" — and **implicit-origin marking** on the roster entry, both "settled in semantics but open in schema", with the note that adding the lanes then "would land a dormant lane with no reader".

**Both are discharged by the flat collection rather than by adding lanes.** A branch and a hop are each now a first-class semantic source with an `EntityId`, so ADR 0040's existing `(keyset, data kind)` reference *already* attributes an implicit keyset to a branch or a hop. No per-unit lane is owed, which is the outcome 0040 preferred; the ruling holds whichever direction the composed-keyset decision gives that edge, and this ADR does not settle the direction.

**No origin field on the roster entry, on the same ground that retires `DataKindOrigin`.** Implicitness is not a planner fact, and 0040's own counter-precedent already says so — "this codebase already stores derived values without marking them", with the derived-and-stored `join_type` as the standing example. Traceability is preserved structurally: the keyset's reference names the source, and the source names its originator.

### Amends

- **0010:** the BLAKE3 clause is superseded, because a BLAKE3 digest is not canonical UUID text; a semantic source without an authored id, implicit units included, mints UUIDv5 over its name, not its content.
- **0025:** the manifest half is superseded — `DataKindVariant` gives way to the polymorphic `role_definition` and `DataKindOrigin` goes; the model's trait hierarchy is unaffected.
- **0033:** an implicit unit is an ordinary `semantic_sources` entry, its id seeded on its originator-qualified name rather than content; `RoutingUnitRef` and `DataKindOrigin` retire.
- **0034:** An authored unit and a compile-synthesised one are the same kind of entry; `DataKindOrigin` "is deleted rather than re-homed, because implicitness is not a planner fact", and what survives is `public` plus `originators`.
- **0040:** both handed-forward items are discharged: a branch or hop is a semantic source, so no per-unit lane is owed, and the roster entry carries no origin field.

### Consequences

- **What retires in `crates/` — recorded for the implementation effort, not done here.** `NestedDataKind`, `NestedDataKindVariant`, `UnionsetBranch.kind`, `RoutingUnitRef` (all four still present, at 25 / 18 / 29 references), `DataKind.origin`, `DataKindOrigin` itself and its reuse as the Joinset `path_origin` tag, and the one `DataKindOrigin::Explicit` construction in `compile/mod.rs`. `EntityNamespace::DataKind` gains its first use and a minter.
- **Two bounds are deliberately absent.** The authored-nesting depth bound and the derived-enumeration ceiling — ADR 0014's `MAX_IMPLICIT_COMPOSITION_DEPTH = 4`, `MAX_IMPLICIT_ENUMERATION_COUNT = 2000`, its no-truncation rule and its two build outcomes — belong to the enumeration decision, which bounds the input and the output of the same enumeration and should rule them together. This ADR rules only that a cycle is rejected, which is a different claim from a depth being capped.
- **Three checks are named for the manifest's check roster**, each with its owning stage rather than a roster entry: operand-graph acyclicity; `originators` length ≤ 1, temporary by design; and for an authored nested unit, `originators` agreeing with the parent's operand list.
- **This ADR does not license enumeration.** Which implicit sources exist, and what bounds the set, is a separate decision against ADR 0011 — as is the reversal of 0011's prohibition on persisting enumerated compositions. Nothing here depends on that answer: the carrier, the identity, the two edges, the one flag and the acyclicity rule hold for whatever set of implicit sources is ruled to exist.
