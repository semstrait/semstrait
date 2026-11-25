---
number: 61
title: 'A source schema may be partial: an unobserved column type is absent, never invented'
status: accepted
date: 2026-09-02
source:
- CONTEXT.md
- docs/adr/0004-typed-diagnostic-kinds-no-error-codes.md
- docs/adr/0005-warning-preserving-stage-error-policy.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
links:
- target: 46
  kind: amends
- target: 66
  kind: amendedby
- target: 70
  kind: amendedby
- target: 82
  kind: amendedby
- target: 86
  kind: amendedby
---

# A source schema may be partial: an unobserved column type is absent, never invented

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0066](0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md): reader-contract clause 2 admits the bare-`Column` Dimension exception; clause 3's "decide at the conversion" is met by 0066's ruling 5.
> - [ADR 0070](0070-a-path-source-is-one-base-path-read-in-one-format.md): its restatement "one `PhysicalSource` per expanded path" reads as 0070's unit: one source per base path and format.
> - [ADR 0082](0082-a-physical-read-types-an-unobserved-column-by-the-semantics-mapped-to-it-and-compile-writes-that-type-into-the-schema.md): in *Decision Outcome*, "`None` means one thing and only one thing: **this compile did not observe it.** Compile never fills the gap with a value derived from somewhere else" holds for a catalog-derived schema; on a model-fed schema, compile writes the type that 0082's *Which semantics types the column* gives into a column that a semantics maps directly, and `None` there means that no semantics with a `data_type` maps the column directly. In *The reader contract, stated normatively because no consumer exists to derive it from*, clause 1's "the conversion is the single place a `None` is resolved" holds for a column that keeps `None`, and clause 2, which bars substituting "the referencing definition's declared type at the point of use", holds: the type that compile writes is in the schema, and no reader substitutes one. In *Amends*, its entry for 0046's "The typing clause is struck: a referencing definition's declared type is a claim about the *definition's result*, not about any leaf beneath it" holds for a column that only an expression reads. The refusal in *Considered Options* to broadcast a declared type onto the persisted columns stands for a column that only an expression reads (0082's *Compile writes the read type into the schema*). The rest of 0061 stands.
> - [ADR 0086](0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md): in *An unmappable native spelling stops the compile no longer*, "no type-mapping roster has been ruled", "This is not a new alias rule and does not reopen `native_to_canonical`'s refusal to make roster choices" and "no `DataType::parse_scalar` authoring aliases" become: `native_to_canonical` reads each type's canonical name and its aliases, then the dialect that the catalog's provider names (ADR 0086, rule 7), so Iceberg's `int`, `long` and `float` are `integer`, `bigint` and `real`. No casing tolerance, and no bare `decimal` acquiring an invented precision, stand: a catalog's bare spelling maps only where its dialect's declaration fixes the value, as Iceberg's `time` is `time(6)`. "`CompileError::UnmappableNativeType` is not deleted; it is demoted" no longer holds, nor does the rejection in *Considered Options* of "Keeping `UnmappableNativeType` fatal and extending the mapping roster instead": a referenced column whose native type maps to no type fails the compile (ADR 0086, rule 7). So "`native_to_canonical` returning `None` keeps the column", "A catalog-derived source can carry a `None` type", the entry for 0046's "demoted to a Warning-severity diagnostic, emitted only on the catalog-observed arm", and, in *Consequences*, "An Iceberg table carrying a spelling this workspace has not placed now compiles" no longer hold, and *`validate()` gains exactly one invariant* has no subject in a manifest that compile wrote. The rest of 0061 stands.

## Context and Problem Statement

The defect: a declared type is not an observation, and one arm dropped columns
entirely.

**Broadcasting a definition's declared type down its expression tree fabricates
facts.** `sum(price * qty): decimal(18,2)` declares the type of a sum. The
status-quo `assumed_provenance` stamps `decimal(18,2)` onto **every** `ColumnRef`
leaf of that expression, so `qty` is persisted as `decimal(18,2)` — a type no
author wrote and no provider reported. Because `mint_schema_id` hashes canonical
types, the fabrication is then content-addressed into `SchemaId`, the value
runtime synchronization compares. An invented type does not merely sit in the
artifact; it takes part in the artifact's identity.

**And the type requirement silently shrank the column set on both arms.**
`column_demand` built a `declared` map by `filter_map` over
`definition.data_type` and skipped any override row whose definition had none.
Only a named `Filter` has `data_type: None` — dimensions, measures and metrics
always declare one — so a column read **only** by a filter never entered the
demand map at all. Two consequences, both confirmed by reading the code:

- **The column vanished from the schema on the catalog arm too**, where the
  provider had reported its type correctly. `observed_provenance` filters the
  provider's list *by the demand map*, so a fault in a type lookup deleted an
  observation that was already in hand.
- **`CompileError::ColumnNotInSourceSchema` became unreachable for filter-only
  columns.** The drop shrank both sides of the `columns.len() < wanted.len()`
  comparison, so the gate that fires when a referenced column is absent from the
  provider's schema could not fire for the one class of reference whose type is
  genuinely unavailable.

**There is no observer to consult for a file source, and that is structural,
not a wiring gap.** Every `CatalogProvider` method takes a `&TableRef`; a path
has nothing to ask a catalog about, which is why 0046 verified that a file source
is always model-fed. So a model-fed column's canonical type is *unobserved*, in
the strong sense that this stage has no instrument pointed at it. The only honest
persisted value is absence.

## Considered Options

- An optional canonical type and nullability, absent when unobserved (chosen).
- **Broadcast the referencing definition's declared `data_type` onto every
  `ColumnRef` leaf.** The status quo, and the defect: it persists `qty:
  decimal(18,2)` for `sum(price * qty): decimal(18,2)`, then content-addresses
  that fabrication into `SchemaId`. A declared type describes an expression's
  result; nothing licenses pushing it down the tree. Rejected on correctness, not
  on taste.
- **Refuse the compile when a referenced column has no observable type.** Loud,
  which is usually this codebase's preference, and unusable here: **the author
  has no key to write.** A named `Filter` has no `data_type` field in the
  authoring surface, and a file source has no catalog to ask, so the refusal names
  a fault with no remedy. A diagnostic an author cannot act on is a broken build,
  not a check.
- **Give a `Filter` definition a `Boolean` `data_type`.** It reads like a tidy fix
  and is inference on the filter's behalf — precisely the class ADR-era prose
  refused when it recorded that nullability "is not inferred from data" and that
  a native spelling is "refused rather than approximated". Worse, it *feeds* the
  broadcast bug rather than fixing it: the filter's leaves would then be stamped
  `Boolean`, so `region = 'EU'` would persist `region` as a boolean column. It
  converts a missing type into a wrong one.
- **Change `semstrait_ir::SchemaColumn.data_type` to `Option<DataType>`.**
  Rejected on three grounds stated below: a plan node's output schema must be
  total or emission is impossible; `ir::Schema` is documented field-stable shared
  vocabulary; and `plan/validate.rs:321` compares `data_type` for set-union
  agreement, where two `None`s comparing equal weakens the check invisibly. Any
  edit under `crates/semstrait-ir/` is out of scope for this ruling.
- **Keep the type mandatory and admit a `DataType::Unknown` sentinel instead.**
  Rejected because `Unknown`'s `Serialize` errors by design, as a guard against a
  compile-transient sentinel reaching disk. Persisting it would turn this
  decision into a serde failure at seal, and defeating that guard to make an
  absence representable is the guard working correctly.
- **A names-only column list on the model-fed arm.** Rejected: the arm's shape
  would carry the claim "a type is impossible here", which the successor
  direction falsifies, and it splits `SchemaProvenance::schema()` into two shapes
  for the sole accessor 0046 made mandatory.
- **Read parquet footers and CSV headers at compile time, so a file source is
  observed too.** **Not rejected on the merits — it is the named successor
  direction.** It is the only way an absence on the model-fed arm ever becomes
  an observation, and it is genuinely attractive. It is out of scope here because it
  is a different decision with its own costs: it adds compile-time I/O
  multiplied by 0046's unbounded per-path glob fan-out, on the stage that is
  already the only I/O step; it makes the artifact a function of file *contents*
  rather than only file existence, deepening the determinism narrowing 0046 recorded; and
  provenance would need either a third arm ("observed from the file itself") or
  a collapse of 0046's two-arm split, since neither "catalog was consulted" nor
  "nothing was consulted" would then be true.
  Whoever takes it inherits this ADR's carrier unchanged, which is the point of
  keeping the uniform struct on both arms.
- **Emitting a note on the model-fed arm so absence is never silent.** Rejected:
  absence is the expected state on the arm with no observer, and six tests assert
  a clean warning channel for that exact compile.
- **Keeping `UnmappableNativeType` fatal and extending the mapping roster
  instead.** The roster should be extended, and one identity mapping is added
  here. It is not a substitute: no roster is complete against providers this
  workspace does not control, and the fatal variant makes a missing roster entry
  look like an authoring error while blocking a compile that reads the column
  only through a type nothing needs yet.

## Decision Outcome

A persisted source schema column carries an **optional** canonical type and an
**optional** nullability. `None` means one thing and only one thing: **this
compile did not observe it.** Compile never fills the gap with a value derived
from somewhere else, because every candidate for that value is a claim about the
world that nothing in this compile checked.

### Amends

This **amends ADR 0046's schema-provenance ruling; it does not supersede 0046.**
Everything else 0046 fixed stands unchanged: one `PhysicalSource` per expanded
path, the flat shape with specialization beneath it in `semstrait-catalog`, the
two-arm `SchemaProvenance` discriminant, the native spelling kept beside the
canonical list as non-load-bearing provenance, `schema_id` replacing
`schema_fingerprint`, and the referenced-columns-only scope on both arms.

- **0046:** a persisted column's canonical type and nullability are optional. In
  its model-fed sections, two sentences change, and one rejected alternative
  with them. "The derivation is: the `ColumnRef` leaves of every expression
  whose `physical_sources` includes this source, **typed from the referencing
  semantic definition's mandatory `data_type`**." The leaf set survives verbatim
  as the derivation of the column *set*. The typing clause is struck: a
  referencing definition's declared type is a claim about the *definition's
  result*, not about any leaf beneath it. "**`nullable` is `true` on the
  model-fed arm, meaning 'unknown, treated as nullable'.**" That reading is
  struck. It was forced by `ir::SchemaColumn.nullable` being a mandatory `bool`.
  With the manifest carrying its own column type, unknown is spelled `None` and
  no longer has to be smuggled into a value a consumer will read as an
  observation. 0046's rejected alternative "**`nullable: false` on the model-fed
  arm** — claims non-nullability on no evidence" is preserved and generalised:
  `Some(true)` on that arm claims *nullability* on no evidence, in the same
  class of fault, and is now equally refused. The carrier is the manifest's own
  `SourceSchema`, not `ir::Schema`, and it is not a breach of the no-vocabulary
  rule; the retired native-spelling
  `SourceColumn { name, source_type: String, nullable }` is not what returns
  (0061's *The manifest gets its own carrier, and `semstrait-ir` does not
  move*). `CompileError::UnmappableNativeType` is not deleted but demoted to a
  Warning-severity diagnostic, emitted only on the catalog-observed arm (0061's
  *An unmappable native spelling stops the compile no longer*).

### The manifest gets its own carrier, and `semstrait-ir` does not move

`semstrait-manifest` declares `SourceSchema { columns: Vec<SourceColumn> }` and
`SourceColumn { name: String, data_type: Option<DataType>, nullable: Option<bool> }`.
`SchemaProvenance` carries `SourceSchema` on the canonical side of both arms.
**`semstrait_ir::Schema` and `semstrait_ir::SchemaColumn` are not touched.**

**The two types are not one type with a nullable field; they answer different
questions.** A `PlanNode`'s output schema is a statement the emitter *executes
against*: an adapter aligning Substrait `RelRoot.names` by ordinal, or writing a
cast, cannot proceed from "type unknown". It must be **total** or emission is
impossible. A source schema is a **claim about the world** — what a relation out
there contains — and a claim may be partial without being useless, because the
column set is still complete (below) and the planner still knows which columns
exist.

**Widening `ir::SchemaColumn` would also weaken a live check silently.**
`crates/semstrait-ir/src/plan/validate.rs:321` compares column types pairwise for
set-union agreement:

```rust
.all(|(ca, cb): (&SchemaColumn, &SchemaColumn)| ca.data_type == cb.data_type)
```

Under `Option<DataType>`, two unknown sides compare **equal** and the branches
pass a compatibility check neither one satisfied. That is the failure mode this
codebase is least able to detect: a wrong query produced by an omission, not a
rejected compile. `ir::Schema` is also declared field-stable — 0046 left a
`ColumnName` newtype divergence standing rather than disturb it, and even
*reading* one field is enough of a contract to have been recorded. So the
partial carrier belongs to the crate whose artifact is partial. The manifest
owning `SourceSchema` is not a breach of 0046's no-vocabulary rule:
optionality-of-observation is a **provenance** property of a persisted source
entry, which is the manifest's own subject, and the type inside the option is
still `semstrait-ir`'s `DataType`.

The retired `semstrait_manifest::SourceColumn { name, source_type: String, nullable }`
that 0046 deleted is **not** what returns. That type put a *native* spelling where
a canonical one belonged. This one holds the canonical vocabulary and adds only
the presence bit.

### The arm records which observer was consulted, not what the observation contained

**`SchemaProvenance` keeps both arms, and their meaning is unchanged: a catalog
was supplied or it was not.** Provenance is therefore *not* derivable by counting
`None`s, and no consumer, comment or doc may say otherwise. A catalog-derived
source can carry a `None` type — an untranslated native spelling, per the section
below — and a model-fed source carries nothing but `None`s today, but the
correlation is a fact about current mappings and current providers, not the
definition. 0046 refused to infer the fork from a field's absence; inferring it
from a *count* of absences is the same error with arithmetic in front of it.

**The model-fed arm keeps the uniform `SourceSchema` struct, not a names-only
list.** A names-only arm would read as a stronger claim than it is — "this arm
can never carry a type" — and it would make the successor direction a carrier
change rather than a value change. One shape on both arms also keeps
`SchemaProvenance::schema()` a single accessor, which 0046 made the *only*
sanctioned way to read the column list.

### An unmappable native spelling stops the compile no longer

**`native_to_canonical` returning `None` keeps the column.** The column is
persisted with `data_type: None`, `nullable` from the provider — which *is* an
observation, so it is `Some` — and the provider's own spelling preserved verbatim
in the catalog arm's native schema, which already stores exactly that.

**`CompileError::UnmappableNativeType` is not deleted; it is demoted.** It
becomes a Warning-severity diagnostic under ADR 0005's warning-preserving policy,
carrying `column`, `native_type`, `physical_source_id` and `expr_id`, and it is
emitted **only** on the catalog-observed arm. It can no longer be fatal for the
reason 0046 already gave for scoping translation to referenced columns, followed
one step further: the translator is deliberately strict, no type-mapping roster
has been ruled ("The type mapping table is not ruled here"), and a real Iceberg
table therefore contains spellings this workspace has not yet placed. Refusing
the compile makes an unratified roster entry indistinguishable from an authoring
error, and the author cannot fix it from the model. The warning keeps it loud,
addressed to the party who can act — whoever extends the mapping.

**One identity mapping is added with this ruling: Iceberg's 32-bit integer
spelling `int` maps to `DataType::Integer`.** This is not a new alias rule and
does not reopen `native_to_canonical`'s refusal to make roster choices: `integer`
and `int` are the same type under two names, and the function already accepts
`integer`. The stricture that stands is the one its own doc states — no casing
tolerance, no `DataType::parse_scalar` authoring aliases, no bare `decimal`
acquiring an invented precision.

### The column set stays complete and authoritative

**This invariant does not change, and it is what makes a partial schema safe.**
Every column any expression reads from a source appears in that source's schema.
Only its type and nullability may be absent. So:

- `column_demand` drops the `declared` map, the `filter_map` over
  `definition.data_type` and the per-row skip, and `ColumnDemand` drops its
  `data_type` field, retaining `expr_id` alone. The type becomes **structurally
  unreachable** from the demand map rather than merely unread, because the next
  author to add a field there must not find one waiting to be broadcast again.
- **`ColumnNotInSourceSchema` now fires for a filter-only column** naming a
  column the provider never reported. That is the gate 0046 assigned to the
  catalog arm alone, restored to the one class of reference it was blind to.
- A named `Filter` definition keeps `data_type: None`, exactly as authored.
  Filters are not given a `Boolean` type (see Considered Options).

### Presence is hashed before the value

**`mint_schema_id` writes a discriminant byte before each optional field —
`0x00` for absent, `0x01` followed by the bytes for present.** Absence is not the
empty string and not a skipped field. The rule is the one the existing
length-prefixing serves: no encoding of one value may be a valid encoding of
another, or two different schemas mint one id and drift goes unsignalled. Since
`SchemaId` is now minted from a partial schema, "absent" is a value the encoding
has to spell unambiguously.

**Every `SchemaId` moves, for essentially every source, and the golden fixture is
regenerated.** That is a consequence of the ruling, not a cost to be mitigated:
the old ids were hashes over types that were partly fabricated.

### The reader contract, stated normatively because no consumer exists to derive it from

Nothing downstream reads a source schema's type today, so this contract cannot be
inferred from behaviour and is fixed here instead of discovered later:

1. **A reader that needs a total schema must convert, and the conversion is the
   single place a `None` is resolved.** Building a `semstrait_ir::Schema` from a
   `SourceSchema` is that place. Every resolution decision — refuse, ask a
   provider, ask the caller, infer from the expression — belongs to that
   conversion and is visible in one function.
2. **A reader may not silently default an absent type.** Substituting `String`,
   `Unknown`, or the referencing definition's declared type at the point of use
   re-creates this ADR's defect downstream of the artifact, where no `validate()`
   can see it.
3. **A reader may not treat `nullable: None` as `true`.** Absence and "nullable"
   are different statements; a consumer that needs a boolean must decide, at the
   conversion, and record that it decided.
4. **A reader must not read absence as provenance.** See the two-arm section: ask
   the arm.

### `validate()` gains exactly one invariant

**A `CatalogDerived` column whose `data_type` is `None` must have a native
spelling beside it in the stored native schema.** That is the one thing a
hand-edited or mutated artifact can desync here: the catalog arm's absence is
only legible because the provider's own word for the type sits next to it, so an
absent canonical type with no native spelling is a column that claims to have
been observed and shows no observation.

It is written as a **call into the producing stage**, following the convention
this crate already uses for a cross-stage invariant — `validate.rs` invokes
`crate::compile::seal::derive_join_type` and keeps only its error rather than
restating the derivation. A reimplementation would be a second copy of the rule,
free to drift from the producer while still passing. This sits in ADR 0042's
defense-in-depth class: the producer cannot emit the fault, and `validate()`
polices post-compile mutation.

**No all-`None` invariant is added on the model-fed arm.** Today that arm's types
are uniformly absent, but the successor direction makes some of them present
without changing the arm, and an invariant asserting today's uniformity would
have to be deleted rather than relaxed the moment that lands.

### No diagnostic on the model-fed arm

**A model-fed source emits nothing for its now-uniformly-absent types** — not a
warning, not a note. This is the arm where absence is *normal and expected*:
there is no observer to consult, so there is nothing to report, and a note would
be pure noise on the most common compile shape. It would also break six
downstream integration tests that assert `warnings.is_empty()` while compiling
with `catalog: None`, for a reason unrelated to anything they are about. The
`UnmappableNativeType` warning is confined to the catalog-observed path, where an
absence really does mark something a human can fix.

### Consequences

- **Every `SchemaId` changes** and the golden snapshot fixture regenerates. Any
  artifact minted before this ruling compares unequal to a recompile of the same
  model, which reads as drift for one pass. Acceptable once, and unavoidable: the
  previous ids hashed invented types.
- **An Iceberg table carrying a spelling this workspace has not placed now
  compiles**, with a warning, where it previously failed. That is a real
  behaviour widening on the catalog arm, and it is as much the practical reason
  for the ruling as the fabrication is.
- **The manifest's public surface moves.** `manifest/mod.rs` stops re-exporting
  `semstrait_ir::SchemaColumn`; `SourceColumn` and `SourceSchema` take its place
  as the manifest's source-schema vocabulary. `semstrait_ir::Schema` was judged on
  its other users, as this clause said it would be, and it has none: every
  `Schema` user in the workspace — planner, adapter, and the manifest's own test
  harness — names it through `semstrait-ir` directly, and no field of the
  manifest contract carries one any more. So it drops out of the re-export too,
  and the conversion clause 1 of the reader contract fixes is written by whoever
  needs it, against `semstrait-ir`. Keeping the name re-exported to *advertise*
  that conversion would have documented a seam this crate does not provide.
- **A partial schema is now representable**, so `validate()` acquires the one
  check that a partial-but-consistent artifact can fail: a catalog-derived
  absence with no native spelling beside it. Exactly one new invariant, per
  ADR 0044's discipline of placing a check where its fault is reachable.
- **`ColumnDemand` carries only `expr_id`**, so no future author finds a declared
  type sitting in the demand map inviting the broadcast back.
- **Doc comments that assert the falsified shape are rewritten, not left
  standing.** This repo's comments are load-bearing, and several currently state
  the amended sentences as live design: the "typed from the referencing
  definition's declared `data_type`" prose in `compile/catalog.rs`'s
  `complete_sources` and in `SchemaProvenance::ModelFed`; the "refused rather
  than approximated" / "metadata leaves this stage canonical or the compile
  fails here" clauses in `catalog.rs`'s module doc and `native_to_canonical`'s
  doc; and the `nullable: true` == "unknown" convention note wherever it
  appears. A stale comment asserting a retired shape is the failure mode
  an earlier slice was written to clear.
- **`CONTEXT.md` needs no new term, and amends one entry.** No term is minted:
  what changes is the *optionality* of a field on a carrier, and **Canonical data
  type** and **Column** are unaffected. But **Physical source**'s definition body
  stated the amended sentences outright — "typed from those semantics' own
  declarations and of unknown nullability", and "Its schema is canonical in both
  cases" — so the entry is rewritten to say that the column *set* is complete on
  both routes while a column's type and nullability are optional, and its _Avoid_
  list gains the retired derivation, the defaulting readings, and inferring the
  route from a count of absences. The glossary never claimed the *term* was about
  a total field; it did claim the derivation, which is the thing struck.
