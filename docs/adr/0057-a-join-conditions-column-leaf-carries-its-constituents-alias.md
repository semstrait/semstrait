---
number: 57
title: A join condition's column leaf carries its constituent's alias, and the attachment says whether it does
status: accepted
date: 2026-08-26
source:
- CONTEXT.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0038-relationship-join-basis-is-a-name-list.md
- docs/adr/0041-recompile-is-the-callers-decision.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0050-a-relationship-is-compile-input-not-a-manifest-entity.md
- docs/adr/0051-entity-ids-are-derived-from-the-natural-key.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
- docs/adr/0055-manifest-provenance-is-caller-supplied-and-reverse-indexes-are-not-persisted.md
links:
- target: 65
  kind: amendedby
- target: 66
  kind: amendedby
- target: 74
  kind: amendedby
- target: 75
  kind: amendedby
- target: 78
  kind: amendedby
- target: 85
  kind: amendedby
---

# A join condition's column leaf carries its constituent's alias, and the attachment says whether it does

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): its successor invariant, ADR 0049's *"exactly one override row per `(semantic, data_kind)`"*, is restated as **one signature per `(semantic source, definition)`**, and made total: the optional case it allowed is gone, so "nothing is minted" is superseded; the constituent qualifier governs join-condition leaves only, and a signature projection is qualified by the immediate operand.
> - [ADR 0066](0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md): ADR 0050's cross tautology is superseded; a cross joinset plans as a `CrossNode`, and the two-literal identity is only an interim cross marker.
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): every join type reads `Coalesce` for a name both operands map, and `Right` reads it `to` then `from`, so the fixed `from`-then-`to` order no longer holds for `Right`.
> - [ADR 0075](0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md): in its *Decision Outcome*, a column leaf of a Joinset's join condition or of a relationship filter names the constituent's semantics, not "`<alias>.<physical column>`"; 0057's reading of ADR 0050's referents clause is withdrawn, where ADR 0057's *Amends* says the clause "does not say the projection **renames**"; in *The check ADR 0050 owed is now definable*, the name it checks after the dot is a semantics name once compile writes semantic leaves, and the check changes with that compile change; in *The attachment says whether a leaf is qualified, because a dot cannot*, a composition-attached expression carries the output fields that its constituents give, not the columns their mappings produced; its consequence "The cut before quoting is an adapter obligation with no adapter behind it" is replaced by 0075's **A name is one identifier**; in *Column order follows the semantics name*, below the root two output fields of one semantics are ordered by their whole names; in *The qualifier names the constituent, not the immediate operand*, "A constituent is concrete and a composite is not" reads: a constituent that a condition leaf names is a Dataset, a Unionset or a Grainset, or a child of a cross-grain Joinset, a Joinset included, and a composite, the Joinset that a cumulative unit has as an immediate operand, is never a constituent; a leaf names a Unionset or a Grainset constituent, and a child of a cross-grain Joinset, by its own alias and a semantics name, never by a branch, a child or a constituent inside it; the constituent qualifier stands (0075's *A join condition reads each constituent's semantics*, *Until compile writes semantic leaves*, *Every output name is the one its parent reads* and *Names are unique, and the planner's own names are reserved*).
> - [ADR 0078](0078-a-source-alias-is-its-source-roles-letter-and-its-index-in-the-manifest.md): in its *Decision Outcome*, the alias is not "derived from that source's name"; in *The alias is minted from the name, never from a counter*, the derivation from the name, the collision suffix and the paragraph that begins "**A counter is refused.**" are replaced by 0078's *An alias is `ss`, its source role's letter and its index*, and "The alias is not a duplicate of the name, and the distinction is why it earns a field" and "The reverse direction is not persisted" stand; in *The check ADR 0050 owed is now definable*, aliases are distinct by construction, and `validate()` derives each alias again (0078's *`validate()` derives each alias again*); the test of resolution without regard to case stands. The rest of 0057 stands.
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): its decision "The attachment says whether a leaf is qualified" is replaced by ADR 0085, rules 11 and 12, and the sections that only apply it read as those rules give: its title's "the attachment says whether it does"; in *Considered Options*, the chosen option's "with the expression's attachment deciding whether a leaf is qualified"; in *Decision Outcome*, the sentence "Whether a leaf is qualified at all is not read from the string"; in *The attachment says whether a leaf is qualified, because a dot cannot*, the sentence that "a dot means qualified" would be a guess, and the sentences from "In a composition-attached expression every column leaf carries a qualifier" to the end of that paragraph; and the **Physical expression** clause of its *Consequences*, which is the one ADR 0085 writes. In its decision "A join condition's column leaf carries its constituent's alias", in its *Decision Outcome*, *The qualifier travels inside the string, and the type does not change*, *The attachment says whether a leaf is qualified, because a dot cannot* and *The check ADR 0050 owed is now definable*, a column reference is the name path that 0085's rules 9, 11 and 12 give. Nothing is cut at a dot: the check reads the alias from the first part. The rest of 0057 stands.

## Context and Problem Statement

ADR 0050 ruled that a relationship persists as a pooled join condition on the produced joinset's `role_definition`, that "the expression's referents are the operands' projected output names", and that the naming rule making those referents unambiguous was "owed to the join-path-realisation research rather than defined here". This ADR is that rule.

The reference corpus produces the fault. `test_data/comprehensive_ecommerce.yaml:830-838` declares the relationship `orders_to_customers` from data kind `orders_daily` to data kind `customers`, with `customer_id` as its basis. Neither data kind remaps `customer_id` in its `column_mapping` (`:709-724`), so on both sides the physical column is literally `customer_id`. Compile lowers that basis to one expression, and unqualified it is `Eq(Column("customer_id"), Column("customer_id"))` — a predicate comparing a column with itself.

The fault is **not** that two semantics fused: ADR 0038's rule that "one name is one Semantics on both sides" is what makes a basis expressible at all. The fault is that **a physical expression is single-sourced by construction** — ADR 0046 resolves `Column("name")` against exactly one source's schema — while a join condition spans two. Nothing in the leaf says which of the two it reads.

## Considered Options

- An alias qualifier inside the `ColumnRef` string, cut at the first dot, with the expression's attachment deciding whether a leaf is qualified (chosen).
- **A qualifier field on the column leaf**, so a leaf is a name plus `qualifier: Option<String>`. This is the shape the retired SQL renderer had, quoting the two parts separately, and the only option where no consumer has to cut a string. Refused because the string form is what query engines themselves parse, and because it changes a `semstrait-model` type to carry something no author ever writes — the change ADR 0050 refused, so taking it would reverse a named rejection to buy a convenience.
- **A flat prefixed column name**, `orders_daily_customer_id`, unique with no separator to cut. One legal identifier, no type change. Refused because no physical source holds a column of that name, so the planner must add a rename above every scan *and* ADR 0046's schema derivation must strip the prefix — the cut moves rather than disappears, and lands in a rule about schemas rather than in the adapter where quoting already lives.
- **A positional convention**: in every `Eq` under the top-level conjunction, the left operand reads `from` and the right reads `to`. The cheapest option — no alias, no field, no type change, no cut — and how `JoinNode.on` already carries sidedness. Refused because it cannot carry `filter_expr_id`: ADR 0038 makes `cross` plus `filter` the sanctioned non-equi join, and `order_date BETWEEN start AND end` mixes both sides' leaves at arbitrary depth, where no position rule reaches.
- **Renaming each constituent's columns in a projection**, so the leaves stay bare and uniqueness is carried by the output names. Refused because the artifact has no per-source column list to rename into: a column exists only in a physical source's schema and in an expression leaf, and semantics are an internal construct the planner consumes. Inventing one would also multiply a composite's width with each constituent, while ADR 0047 gives it one bit per semantics, so the mask would stop corresponding to the columns.
- **Side tags, `from` and `to`, in place of a per-source alias.** Bounded and disjoint by construction, and it writes ADR 0038's own reading of a basis literally. Refused because the qualifier must name the **constituent** the causing relationship names, not the operand position: under ADR 0054 that constituent sits inside the left operand at depth, so a side tag names the wrong thing.
- **The semantics' `EntityId` as the column name.** Unique and stable. Refused for the reason `CONTEXT.md` already gives — one name is one logical semantics, so both sides carry the same id and the tautology returns unqualified — and because it spends identifier budget on an unreadable trace.

## Decision Outcome

A column leaf inside a join condition is written `<alias>.<physical column>` in the single string `ColumnRef` already holds, and a reader cuts it at the **first** dot. The alias is a new minted field on every semantic source, derived from that source's name. Whether a leaf is qualified at all is not read from the string: it follows from the expression's **attachment**, because an expression reached from a `role_definition` is a composition's and an expression reached from a physical source's schema derivation is that source's.

This completes ADR 0050 rather than superseding it. Nothing in 0050 is narrowed, and its rejected alternative on this exact subject survives — see below.

### The qualifier travels inside the string, and the type does not change

`ColumnRef` is `pub struct ColumnRef(pub String)` and `#[serde(transparent)]`, so its wire form is already a bare scalar. Putting the qualifier inside that string costs **no persisted bytes, no type change and no version consequence**, and it is the form engines themselves parse: `orders_daily.customer_id` is what a reader of the emitted SQL will see, and the artifact spells it the same way.

**ADR 0050's rejected alternative stands as written; this is not a quiet reversal of it.** 0050 refused "extending `ColumnRef` with a source or side qualifier" on two grounds. The first — that it "forces a `semstrait-model` *and* `semstrait-ir` change, flipping ADR 0043's 'the `semstrait-model` answer is no'" — never engages, because no type gains a field. The second — that it "puts join-scope knowledge into a leaf that ADR 0046 deliberately resolves against one source's schema" — is **satisfied rather than overridden**: a qualifier names exactly one source, so every leaf still resolves against one schema, and the source is stated instead of inferred from the leaf's context. Single-source resolution is strengthened by being made explicit.

### The attachment says whether a leaf is qualified, because a dot cannot

**A physical column name may contain a dot.** `SchemaColumn` is `{name: String, data_type, nullable}` and `ColumnRef` is a bare `String`; neither validates its content, and `Name` validation is non-emptiness and nothing else. So "a dot means qualified" would be a guess, and it would misread a source whose own column is named `a.b`.

The signal is already in the artifact. An expression is reached one of two disjoint ways: from a semantic source's `role_definition`, which is a **composition-attached** expression, or as a realisation whose `physical_sources` names the sources that consume it, which is **physical-source-attached**. In a composition-attached expression every column leaf carries a qualifier. In a physical-source-attached expression none does, and the string is the physical column name verbatim. Nothing is detected and nothing is inferred from the characters.

**Qualification does not mint a second kind of column, and the layering is why.** Physical form enters the language in exactly one place — a semantic mapping, where a semantics is tied to a column — and it is read below the projection that turns a source into its semantic interface. Above that projection a consumer names semantics only. A composition-attached expression therefore carries columns its constituents' mappings **already produced**, qualified by the constituent that produced each one, and introduces no physical form of its own. That is the fact this ADR uses to rule `physical_sources` empty, reached from the terminology side rather than the schema side, and it is why `CONTEXT.md` defines a column as the physical representation of a semantics rather than leaving the word to be read two ways.

**The cut is at the first dot, and that is why the alias may not contain one.** Cutting at the last dot would be SQL's own rule for a multi-part name, and it breaks here: a column legitimately named `a.b` would surrender `a` to the qualifier. Cutting at the first dot is total in the other direction: it requires only that the alias hold no dot, which minting controls, and it leaves an arbitrary column name untouched.

### The alias is minted from the name, never from a counter

Compile derives a source's alias from that source's own name: it replaces every character an engine may reject, holds no dot, cuts the result to a fixed length, and appends a short suffix derived from the source's `EntityId` where sanitising or cutting would collide with another alias. Because ADR 0051 seeds an id from the name, the whole chain stays name-derived and compile stays deterministic.

**A counter is refused.** The excluded SQL emitter aliases every node from a monotonic counter reset per `emit()`, which is determinism *within one emission* — it was recorded as "weaker than cross-recompile determinism". An alias that renumbers when an unrelated source appears moves the artifact's bytes for no semantic reason, the ground ADR 0050 already used to refuse roster order for its conjunction.

**The alias is not a duplicate of the name, and the distinction is why it earns a field.** ADR 0045 makes a semantic source's name unique and ADR 0051 seeds its id from that name, so a second unique handle would be the derived value ADR 0055 tells us not to persist. But a name is not an engine identifier: under ADR 0054 an enumerated unit is named `<originator>.r1.r2`, `Name` admits unicode and punctuation, and engines cut identifiers — Postgres at 63 bytes — so two distinct names can become one after the cut, which is exactly the ambiguity this rule exists to remove. The alias does a job the name cannot, and no job the name already does.

**The reverse direction is not persisted.** Alias → semantic source is a search structure built at load, per ADR 0055's rule that "where a fact and its inverse are both useful, the manifest persists the direction compile produces and leaves the reverse to be built at load".

### The qualifier names the constituent, not the immediate operand

Take a cumulative unit over `((A, B), C)`. ADR 0054 rules that exactly one relationship causes it — the one adding the new constituent, `r2(B, C)` — while ADR 0050 makes its `role_definition` binary, so its immediate operands are `(A, B)` and `C`. The relationship relates `B` and `C`, and `B` sits inside the left operand.

**The qualifier names `B`.** Qualifying by the immediate operand `(A, B)` reintroduces the fault one level up: if `A` and `B` both serve `customer_id`, a reference into the composite has two answers, and ADR 0047 gives the composite one bit for that semantics rather than two. A constituent is concrete and a composite is not.

This does not contradict ADR 0050's rule that "a cumulative unit carries its own condition over its two immediate operands, and never inherits or composes theirs". That sentence forbids a unit from inheriting or composing its children's **conditions**; referencing a constituent's column is neither, and the join itself remains over the two immediate operands exactly as 0050 states. The qualifier is a lookup key into the composition, which is why it can be short.

### A join condition names no physical source

`physical_sources` on a join condition — and on its optional filter — is **empty**, because the condition introduces no column anywhere. Every column leaf of a condition or a filter is some semantics' realisation on one side; that semantics already carries an override on that constituent whose expression lists that constituent's source, so the column is already in that source's derived schema before the condition exists. ADR 0046's derivation is a filter over expressions "whose `physical_sources` includes this source", so an empty list contributes nothing and needs no special case at any consumer.

That is also the honest answer under ADR 0049's reading of the edge, which narrowed it from *which source does this row govern* to *which sources consume this expression*: no single physical source consumes a two-sided condition. `CONTEXT.md`'s **Binding** entry already states the general form — "the expression names every physical source that consumes it" — and names joinset conditions as expressions of their own.

The rejected shape is listing the two constituents' sources: ADR 0046 would then pull the other side's column into each source's schema as a column that is not there.

### Column order follows the semantics name

A projection's columns are ordered by **semantics name**. The forcing fact is that `UnionNode` validation compares inputs by ordinal — same arity and same `DataType` at each position — and checks no names, so an undefined order can align two different semantics at one position. Name order also makes a Unionset's branch alignment structural instead of checked, because two branches over the same semantics produce the same sequence.

Roster order and `bit_position` order are refused on the ground ADR 0050 already stated: both are content-derived-id order, while ADR 0015's ratified mechanism is name order, so either would reorder on an unrelated id change.

### A Joinset coalesces a semantics two constituents serve

Where two or more constituents of a **Joinset** serve one semantics, compile mints a `semantic_override` attached to the composition whose root expression is `coalesce(from, to)`. `Coalesce` is a first-class `Expr` variant, so this needs no new vocabulary. Argument order is fixed as `from` then `to` for determinism; correctness does not depend on the order, because after an outer join at most one side is non-null, and where both are non-null the model has already claimed they agree — `CONTEXT.md`'s **Field provenance** carries that claim as "several in agreement".

**Where one constituent serves the semantics, nothing is minted.** Absence means resolve through that constituent, which keeps an override optional exactly as a self-contained `semantic_definition` allows. This makes ADR 0049's successor invariant — "exactly one override row per `(semantic, data_kind)`" — hold for a composite as it does for a leaf.

**This applies to a Joinset only.** A Unionset stacks rows and a Grainset routes between grains, so neither places two values of one semantics side by side in one row, and there is nothing for a coalesce to fold.

### A self-relationship is rejected at model validate, and it costs nothing spellable

Nothing rejects `from == to` today: the relationship rules are the endpoint, key and condition checks plus per-scope name uniqueness, and none of them looks. Such a relationship produces a unit whose two operands are one semantic source, so both sides of its condition carry one alias and no qualifier can tell them apart. Model validate rejects it, beside the endpoint-resolution rule (ADR 0044: the rule belongs to its stage and joins no roster).

**The limitation is ADR 0038's and not this rule's**, which matters for whoever wants self-joins later. Under a name-list basis `on: [s]` means `from.s = to.s`, a tautology whenever `from` and `to` are one data kind; `auto` derives from shared semantics and is the same tautology; `cross` is a self cartesian product. 0038 removed the only useful spelling itself, recording the cost as losing "the ability to relate two *distinct* Semantics without first collapsing them into one pooled dimension". So reopening self-joins means reopening the **authoring grammar**, not this rejection.

**The cost is disclosed:** `cross` with a `filter` over one data kind is genuinely spellable today and is what the rejection removes. It is also the one case where qualification could not have helped, since both sides would carry one alias.

### The check ADR 0050 owed is now definable

ADR 0050 handed forward "every name the condition references resolves in the operands' projections" and could not define it. It is owned by the stage that assembles `semantic_sources`, stated in that stage's own module documentation per ADR 0044: **every column leaf of a composition-attached expression cuts into a known alias plus a column name that resolves in that alias's source schema.** It is ADR 0046's column-name-uniqueness check reached from the other side — 0046 ruled that class the manifest's on the ground that "under name-based column resolution a duplicate makes resolution ambiguous, so this is stage-reachable and currently ungated".

Under `cross` the check is **vacuous by construction**, because ADR 0050 pins the condition to an identity over two literals and there is no column leaf to resolve.

A second predicate rides with it: disjointness and resolution must be tested **case-insensitively** even though `Name` is case-sensitive by ratified design, because engines fold identifiers, so a case-only difference passes every IR-level check and collides in emitted SQL.

### Amends

**Nothing.** This is worth stating positively, because the deciding session first proposed superseding ADR 0050's referents clause and was wrong to. "Each operand is an inline view with its own projection, and the condition references *those* names, unique by construction" does not say the projection **renames**: `customer_id` is a name in `orders_daily`'s projection, and `orders_daily.customer_id` versus `customers.customer_id` are unique by construction because the qualifier is part of the reference. 0050 asked for the mechanism and this supplies it.

One reading of 0050 is narrowed without any clause changing. Its "therefore" — each operand is an inline view — is now one permitted realisation rather than the only one, because an alias-qualified reference resolves under stacked joins as well. That **strengthens** 0050's own claim that "squashing stays a consumer optimisation rather than a second persisted shape": the artifact commits to no plan shape at all. A test of that claim found the consumer that would squash does not exist, is excluded from the workspace, and has an empty pass registry.

### Consequences

- **`crates/` convergence is out of scope on the deciding map** and is handed to the implementation effort. `ManifestExpression` is `{expr, layer}` today, so `physical_sources` does not exist yet and this decision is one of its inputs rather than a change to it. `semantic_sources` gains `alias`. Nothing in `crates/` was edited for this ruling.
- **The cut before quoting is an adapter obligation with no adapter behind it.** An emitter must cut at the first dot and quote each part separately, or it emits one quoted identifier naming a column no source holds. There is no live adapter to place this on — a check found no Substrait emission code in the tree at all and `SubstraitSerializer` has zero occurrences in `semstrait-ir/src/` — so it is recorded here to be inherited deliberately.
- **How a plan node carries the condition is another stage's concern.** The split of a pooled conjunction into `JoinNode.on`, the routing of `cross`'s `lit(1) = lit(1)` where `JoinType` has no cross variant and an empty `on` is rejected, and what Substrait's own node structure requires, are all outside this decision: the manifest supplies enough information for the planner to decide, and the plan's shape is decided later. This is the same line already drawn for `SemanticGraph` construction, the advisory system and the recompile gate.
- **`CONTEXT.md` gains `Column`, `Source alias` and `Column qualifier`**, and its **Physical expression** entry gains the clause that the attachment decides whether column references are qualified. `Column` is the load-bearing one and it was **missing**: the word appeared in nine entries and was defined in none, while three entries — **Semantics**, the dimension entry, and **Join basis** — already forbade using it *for* a semantics. Defining it as the physical representation of a semantics, introduced only by a semantic mapping and read only below the projection, is what stops a reader taking this ADR's "column leaf" as a second kind of column. All three land with this ADR, so the glossary never states a rule the artifact has not ruled.
- **One rule is named at model validate**: a relationship whose `from` and `to` name one data kind is rejected. It joins no roster (ADR 0044).
- **One rule is named at the stage that assembles `semantic_sources`**: the leaf-resolution check above, tested case-insensitively, vacuous under `cross`.

## More Information

### Open questions

- **Whether an authored semantic expression consisting only of columns becomes part of an implicitly created mapping expression** — pushed a layer down — is deliberately not ruled here. It belongs to compile's control flow, where, at the time of this decision, whether explicit and auto mapping are one flow or two was unruled and the body-lowering stage had no definition. This ADR states only the terminological boundary: a column is read below the projection, and above it a consumer names semantics.
