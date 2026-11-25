---
number: 59
title: The coverage axis is three-way, and coverage is read against a composition
status: accepted
date: 2026-08-27
source:
- CONTEXT.md
- docs/adr/0011-thin-manifest-in-memory-graph.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0043-authored-primitives-get-typed-carriers.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0047-coverage-mask-is-a-serveability-statement.md
- docs/adr/0049-a-dataset-names-one-physical-source-as-a-semantic-source.md
- docs/adr/0054-compile-enumerates-every-legal-sub-composition.md
links:
- target: 47
  kind: amends
- target: 74
  kind: amendedby
- target: 37
  kind: relatesto
- target: 77
  kind: amendedby
---

# The coverage axis is three-way, and coverage is read against a composition

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0074](0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md): the driver writes `NullFill` as a typed NULL for a Unionset branch that does not map a name, once the manifest lets a branch serve only the names it maps.
> - [ADR 0077](0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md): in *Coverage is read against a composition, and null fill is what forces it*, a semantics is null-filled at a constituent only on an explicit Unionset; a Joinset reads a name that one operand does not map from the other operand (0077's *A constituent serves only the names it maps*).

## Context and Problem Statement

ADR 0047 derives the coverage axis as: *"for a `(semantics, source)` pair, no reachable `expr_id` is `NullFill`; a `Column` leaf is `Native`; a `MetadataRef` leaf is `Metadata`; anything composite is `Derived`."* That derivation does not decide every case, which was found by executing it rather than by reading it: a prototype implements both the clause as written and the restatement below, and asserts the second total over every `(leaf, semantics)` pair in its fixtures.

## Considered Options

- Drop `Metadata`, make clause 3 a catch-all, and read coverage against a composition, still with no carrier (chosen).
- **Keep `Metadata` and mark a metadata literal in the expression pool.** The only way to make ADR 0047's test work as written. Refused because it adds a field for no reader, which is the defect ADR 0043 names — *"an unwritten carrier is a defect, a reserved optional is a contract"* — and because no consumer needs the distinction: under ADR 0037 the value is present either way, which is the whole of what the earlier design demanded.
- **Persist the axis** as an enum row or one mask lane per variant. ADR 0047's rejection stands on its own reasoning, and the prototype adds evidence rather than a new argument: candidate selection reads masks only.
- **Delete `Metadata` and say nothing about where a metadata value lands.** The minimal edit, and the literal reading of the question as it was put. Refused because it leaves the derivation non-total for a bare `Literal` leaf, a live case independent of metadata, so the axis would still fail to decide a pair a consumer can ask about.
- **Enumerate the leaf kinds in clause 3** — `Literal` and `Parameter` and composite. Explicit, and it reads better. Refused because `Expr` and `PhysicalLeaf` are both `#[non_exhaustive]`: the enumeration would need an ADR every time either type grows a variant, and the catch-all is already right for all of them.
- **Add a fourth `Constant` variant for a bare literal**, keeping `Native` to mean "a column of this source exists". It preserves the intuition that a constant is not derived. Refused because it recreates the original problem one variant along — nothing distinguishes an authored constant from a metadata-supplied one — so the new variant would be exactly as unrepresentable as the one being removed.
- **Leave the axis four-way and treat the gap as a documentation defect** to be repaired in ADR 0047 in place. Refused on ADR-authority's bound: ADR 0047 is merged, the clause is a *decision* about what is derivable rather than one wrong symbol, and its exception is explicitly *"not a general licence"*, its test being whether the clause is unreadable as written. ADR 0047's clause reads perfectly well and is wrong on the merits, which is the case that gets a new ADR.

## Decision Outcome

`CONTEXT.md`'s **Coverage** axis loses `Metadata` and becomes `{Native, Derived, NullFill}`. Its third clause is restated so that the derivation is **total**: anything that is not a bare column reference is `Derived`. And coverage is read **against a composition**, never off a contributor alone — null fill is the absence of a realization, so it means "null-filled" only where the composition above advertises the semantics, and means nothing at all elsewhere.

ADR 0047 refused a carrier for this axis and that refusal stands. What moves is the derivation, not the storage.

### `Metadata` has no distinguishable subject, and ADR 0037 said so first

The third test of ADR 0047's derivation has no subject. **ADR 0037 retired `SemanticMappingValue::MetadataRef` and resolved metadata "at compile to an ordinary pooled `Literal`"**, and `PhysicalLeaf` is `{Column, Literal, Parameter}` — `crates/semstrait-sql/src/expr/leaf.rs`, with `compile/seal.rs`'s module documentation stating it in terms: *"`PhysicalLeaf` has no `MetadataRef` variant (only `Column`/`Literal`/`Parameter`), so a `ResolvedExprEntry` can never decompose into one."*

So a metadata-supplied value and an authored constant differ in the model and are **the same bytes in the artifact**. This is not an unimplemented variant but an unrepresentable one, and no test over the persisted shape could separate the two.

**ADR 0037 predicted this in the paragraph that handed the question over**, which is why no supersession of 0037 is owed: *"Once metadata is an ordinary `Literal`, that rule's exclusion of `Metadata` likely has to fall, since the earlier design insists a metadata-covered semantics *has* a value. Both belong to the coverage-mask decision."* ADR 0047 dropped the earlier rule's *exclusion* of `Metadata` from the mask, correctly, and kept the *variant* — inheriting a test whose subject 0037 had already removed.

**The glossary was already half-right, which settles which of its two entries is the drift.** `CONTEXT.md`'s **Semantic mapping** states the four authoring forms — *"a column, a literal, an expression, or extracted source metadata"* — and then says in terms that *"those four are authoring vocabulary: in a manifest each semantics resolves to one pooled physical-expression handle, and the four collapse into expression leaves (ADR 0037)."* **Coverage** kept the same four as *manifest-side outcomes*, which un-does the collapse one entry later. This is ADR 0043's pattern exactly — there, *"the four-variant `SemanticRole` roster was the drift; the glossary was right"* — with the direction reversed: here one glossary entry is right and its neighbour drifted.

### The hole is wider than metadata, so this is a restatement rather than a deletion

Deleting the variant alone would leave the derivation non-total, because a second case falls through the same three tests and has nothing to do with metadata.

`PhysicalExpr` is `Expr<PhysicalLeaf>`; `Expr` is `Leaf(L)` plus a set of composite node variants (`BinaryOp`, `UnaryOp`, `FunctionCall`, `Cast`, `Case`, `InList`, `Between`, …). So **a bare `Literal` or `Parameter` leaf is neither a `Column` leaf nor composite** — an authored constant matches none of ADR 0047's three tests and is simply undecided. `ADR 0037` records that `Column` is *"the common path"*, not the only one.

The axis therefore reads:

1. **No realization for the pair** — `NullFill`.
2. **The realization is a bare `Column` leaf** — `Native`.
3. **Anything else** — `Derived`.

Clause 3 is deliberately a catch-all rather than an enumeration: `Expr` and `PhysicalLeaf` are both `#[non_exhaustive]`, so an enumerated clause would need amending every time either grows, while a catch-all is already correct for any variant either could gain. A metadata-supplied value lands in `Derived`, which is where ADR 0037 put it the moment it became a pooled literal.

**Nothing is lost that the artifact ever held.** `Metadata` was never recoverable after ADR 0037; this ADR stops the glossary from promising a distinction the manifest cannot make.

### Coverage is read against a composition, and null fill is what forces it

Under ADR 0047 null fill is the **absence** of a realization. Read off a contributor alone, that rule answers `NullFill` for every semantics in the registry that the contributor does not realize — which is most of them. The prototype asserts the consequence directly: for a leaf carrying no `country` and no `revenue`, the derivation returns `NullFill` for both, although only `country` is null-filled inside the composition above it and `revenue` belongs to an unrelated subtree.

The discriminator is the **enclosing composite's mask**. A semantics is null-filled *at a constituent* when the constituent has no realization for it and the composite advertises it — which under ADR 0047's union arithmetic is exactly what a set bit contributed by a sibling means. So a constituent's coverage is undefined until the composition is named, and `CONTEXT.md`'s existing instruction to *"always say whether it is per physical source or per constituent"* is sharpened rather than replaced: naming the constituent is not enough, the composition has to be named too.

**One structural consequence is stated so that two merged rules stop reading as contradictory.** A leaf's mask is built from the realizations it has, so **a leaf's mask can never carry a null-filled bit**; null fill is only ever a composite's bit. That is what makes ADR 0043's invariant — *"the `physical_sources` lists of the override rows sharing that pair must be disjoint and covering over that data kind's sources … covering stops a source silently lacking a realization for a semantics its bitmask advertises"* — compatible with ADR 0047's explicitly forbidden check, *"every semantics in a source's mask has an expression for that source"* being *"false by design"*. The two are about different entities: 0043 binds a leaf against its own physical sources, and 0047 forbids asking a composite the same question. Neither ADR says so, and read together without this sentence they appear to contradict.

### Still no carrier

ADR 0047's refusal of a `CoverageVariant` enum, a per-`(source, semantics)` provenance row, and a mask lane per variant is untouched. The axis stays derived, and its re-open condition is unchanged: *"Reconsider only for a consumer that needs provenance hot."* The prototype found no such consumer in candidate selection, which reads the mask and never the axis.

The cost is named: **a constant is now `Derived`**, which an author may not expect of a value that involves no computation. Each alternative spelling is worse (see Considered Options); the axis answers *how a value is obtained*, and "not from a column of this source" is the honest answer for a literal.

### Amends

Scoped in this body rather than by editing the older ADR's frontmatter, per the precedent ADR 0037 set over ADR 0031 and ADR 0050, 0051, 0056 and 0058 followed.

- **0047:** the derivation clause only. The four-way list becomes three-way and the third test becomes a catch-all. Everything else in 0047 stands unchanged and is not reopened: a set bit is a serveability statement, null fill contributes, a Unionset's mask is the union rather than the intersection, there is exactly one bit lane, the reverse bit index is not persisted, and a keyset carries no member mask.

**ADR 0037's handover is discharged**, not superseded. It named this question and assigned it here.

**Nothing in ADR 0043 is superseded.** Its disjoint-and-covering invariant is confirmed and given the scope that makes it consistent with ADR 0047.

### Consequences

- **`CONTEXT.md`'s **Coverage** entry is rewritten** — three outcomes instead of four, coverage read against a composition, and *"supplied from metadata"* added to its `_Avoid_` list so the retired outcome lands somewhere. That edit ships with this ADR so the glossary never states a rule this artifact has not ruled.
- **No code changes.** `CoverageVariant` is grep-absent from `crates/`, so nothing retires; this rules out *adding* it, exactly as ADR 0047 did.
- **The roster of checks gains none and loses none.** ADR 0047's *forbidden* entry keeps its subject and gains a reason worth recording beside it: the check would ask a leaf a question about its parent, since null fill is a composite's bit.
- **Handoff, not performed here.** Per-constituent coverage over a composed surface is `CONTEXT.md`'s **Field provenance**, which ADR 0011 forbids persisting and ADR 0043 confirms untouched. This ADR establishes that it is *derivable* — by walking the operand edges with the composite's mask in hand — and rules nothing about which consumer builds it, or when. That belongs to the planner effort, with the routing filter and the path-ambiguity report also handed there.

## More Information

### Open questions

- **The coverage mask has no glossary entry, and naming it is left unruled deliberately**, since it was raised as a finding rather than decided. The phrase occurs once, undefined, inside **Bit position**, and it reads as "a mask of **Coverage**" — the axis this ADR narrows and which ADR 0047 ruled the mask does *not* carry. ADR 0047 judged that no new term was needed partly because **Coverage** kept its four-way axis; that ground has moved.
