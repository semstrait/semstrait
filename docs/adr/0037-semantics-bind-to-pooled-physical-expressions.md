---
number: 37
title: The manifest binds each semantics to a pooled physical expression, not a mapping-value variant
status: accepted
date: 2026-08-05
source:
- crates/semstrait-manifest/src/manifest/expression_layer.rs
- crates/semstrait-manifest/src/manifest/expression.rs
- crates/semstrait-manifest/src/compile/expressions/mod.rs
links:
- target: 13
  kind: amends
- target: 31
  kind: amends
- target: 43
  kind: amendedby
- target: 55
  kind: amendedby
- target: 65
  kind: amendedby
- target: 71
  kind: amendedby
- target: 47
  kind: amendedby
- target: 59
  kind: relatesto
- target: 88
  kind: amendedby
---

# The manifest binds each semantics to a pooled physical expression, not a mapping-value variant

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0043](0043-authored-primitives-get-typed-carriers.md): `SemanticBinding` retires and per-source variation lives on the expression; `depends_on` relocates from `ResolvedExprMeta` onto the expression entry.
> - [ADR 0047](0047-coverage-mask-is-a-serveability-statement.md): the coverage axis is **derived**, and every input it needs is already persisted by the expression edge; the earlier exclusion of `Metadata` from the coverage bits was correct for a mask that meant source-side realizability, and it is wrong for a mask that means serveability.
> - [ADR 0055](0055-manifest-provenance-is-caller-supplied-and-reverse-indexes-are-not-persisted.md): the `depends_on` codomain is superseded from semantics ids to expression ids; its **direction** is confirmed, not superseded, and its inlining decision is untouched; its "recompile-invalidation" consumer is recorded as void under ADR 0041.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): superseded on the `depends_on` half only. Its inlining ruling and its complete-tree requirement are the reason this shape works, and they and its `ExprLayer`-on-the-entry decision stand; the recorded semantic edges go, because a tree that inlines its dependencies and is owned by one source has no second lineage to keep, and the expressions are persisted as a node store with "no interning and no dedup anywhere".
> - [ADR 0071](0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md): `ExprLayer` stays on every stored node, and the planner does not read it; a request predicate's placement is read from the role on the semantic definition of each of its leaves, never from the expression layer of any tree.
> - [ADR 0088](0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md): in its *Decision Outcome*, "Nested semantic dependencies are **inlined** into one tree rather than referenced" becomes: a reference to another signature on the same semantic source is a link to that signature's entry where ADR 0088 rule 5 writes one, and a Metric over Measures is `PostAggregate` (ADR 0088, rule 6). The reason that the rest of that sentence gives, "an opaque handle-leaf would hide its aggregates", and the reason that *Considered Options* gives against a DAG do not hold for a link: the linked entry's `layer` shows the aggregates, so the layer is still one function of what the manifest stores (ADR 0088, rule 6). In *Considered Options*, the chosen option's "inlined dependencies" are those links. The rest of 0037 stands.

## Context and Problem Statement

The decisive reason for this decision is that `ExprLayer` is persisted on the pool entry and nowhere else. the design makes layer a total, tree-derived function (`classify(&col("a")) == Scalar`; see `crates/semstrait-manifest/src/manifest/expression_layer.rs`), and states the planner *"reads the persisted layer to place each expression without re-walking provenance"*. Yet the seal deliberately declines to intern column- and literal-rooted mappings, so under the `SemanticMappingValue` enum the majority of semantics (`Column` is *"the common path … the majority of entries in any real Model"*) reach the planner with no layer at all. The enum was therefore not merely redundant with the pool; it withheld a ratified field from the common case.

This is recorded because it removes a public `#[non_exhaustive]` type from the manifest contract and changes the persisted wire form, and because it knowingly overrides ratified leaves rather than reconciling with them. It is also counter-intuitive on several counts: a manifest whose mapping table contains no values, only handles; four ratified carriers for one concept collapsing to none of them; a "recipe" enum variant that no parser could ever produce being deleted rather than implemented; and dependency edges being stored *beside* a tree that has already absorbed them, because the absorbing is what makes the layer honest.

## Considered Options

- Pooled expression handles, with inlined dependencies (chosen).
- **Keep the four-variant enum and add the typed per-source lane** ADR 0031 requires (`PhysicalSource.metadata_values: BTreeMap<String, Literal>`), deriving coverage at load. Faithful to the earlier design, but it preserves the `ExprLayer` gap for every column- and literal-realized semantics, still leaves `MetadataRef(String)` a key that nothing produces (`compile/seal.rs`: *"`MetadataRef` never originates here"*), and needs a fresh check to stop the key dangling.
- **Collapse metadata to `Literal` but keep a three-variant enum** (`Column` / `Literal` / `Expr`). Fixes the metadata contradiction and nothing else: the layer defect is identical, since the seal still declines to intern the two non-`Expr` arms.
- **Preserve semantic nesting as a DAG** via a new `PhysicalLeaf::ExprRef(PhysicalExprId)` instead of inlining. Rejected because it breaks `classify`'s totality: aggregates below a handle-leaf are invisible, so layer would have to be re-derived from provenance — precisely what persisting the layer avoids.
- **Rely on content-dedup alone to recover the dependency graph**, with no `depends_on`. Rejected because subtree sharing only reveals an edge if the pool content-addresses subtrees rather than interned roots, which the contract does not promise.
- **Retain the earlier `ResolvedColumnMapping`** four-flat-`HashMap`s envelope (`ResolvedBinding` / `ResolvedPhysicalSource`) for planner-speed O(1) lookup. Already displaced by ADR 0013; the pool handle is one lookup either way.

## Decision Outcome

`SemanticBinding.mapping` becomes `BTreeMap<EntityId, PhysicalExprId>` and the manifest-layer `SemanticMappingValue` enum is retired. Every covered semantics — a bare column and a broadcast literal no less than a computed expression — interns into the physical expression pool and is referenced by handle, so the `Column` / `Literal` / `Expr` roster survives only as `PhysicalLeaf`, which already spells exactly those three plus `Parameter`.

Nested semantic dependencies are **inlined** into one tree rather than referenced, because `classify` decides `PostAggregate` by finding an `Aggregate` node anywhere beneath a scalar root; an opaque handle-leaf would hide its aggregates and misclassify a Metric over Measures as `Scalar`. The semantic edges that inlining flattens away are recorded explicitly instead, as `depends_on: BTreeSet<EntityId>` on `ResolvedExprMeta`, so lineage and recompile-invalidation keep a first-class DAG without the tree ever lying about its layer.

### Amends

- **0013:** `SemanticBinding.mapping` becomes `BTreeMap<EntityId, PhysicalExprId>`; every covered semantics, a bare column or literal included, interns into the pool and is referenced by handle.
- **0031:** its decision "a `#[non_exhaustive]` public mapping variant plus a companion per-source storage lane in the persisted manifest" is replaced, on carrier shape only: metadata remains a compile-time third stratum, never a registry function and never a `PhysicalExpr` *variant*, resolved eagerly and fail-fast — but it lands as an ordinary pooled `Literal` leaf. The sections that only apply that decision read so: in *Context and Problem Statement*, that phrase and "an enum variant no parser can produce"; and in *Decision Outcome*, "a distinct `SemanticMappingValue::Metadata(recipe)` variant" and "onto `ResolvedPhysicalSource.metadata_values`". So the mandated variant and the lane are both retired, along with the earlier `manifest.metadata_index` (the planner now finds the constant already in `mapping`). The rest of 0031 stands.

**Supersedes** the earlier four-variant `SemanticMappingValue { Column, Literal, Expr, MetadataRef(String) }` and the manifest-layer half of the earlier roster.

`ResolvedPhysicalSource` is retired vocabulary outright; the live type is `PhysicalSource`, and the naming rule reserves the `Resolved*` prefix for compile-complete forms, which `PhysicalSource` already is.

## More Information

### Open questions

Not settled here: whether the `Native` / `NullFill` / `Derived` / `Metadata` coverage axis is persisted or derived at load. It has no manifest carrier today, so the earlier rule ("only `Native` and `Derived` contribute bits") currently has no representable subject. Once metadata is an ordinary `Literal`, that rule's exclusion of `Metadata` likely has to fall, since the earlier design insists a metadata-covered semantics *has* a value. Both belong to the coverage-mask decision.
