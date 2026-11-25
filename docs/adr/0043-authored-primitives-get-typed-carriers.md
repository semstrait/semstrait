---
number: 43
title: Every authored primitive gets a typed carrier, and shape lock decides where it attaches
status: accepted
date: 2026-08-14
source:
- CONTEXT.md
links:
- target: 11
  kind: amends
- target: 13
  kind: amends
- target: 32
  kind: amends
- target: 37
  kind: amends
- target: 40
  kind: amends
- target: 41
  kind: amends
- target: 48
  kind: amendedby
- target: 49
  kind: amendedby
- target: 50
  kind: amendedby
- target: 65
  kind: amendedby
---

# Every authored primitive gets a typed carrier, and shape lock decides where it attaches

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0048](0048-composite-keysets-are-authored-inside-a-boundary.md): a keyset entity is `{id, name}`, minted over `(semantic_source_id, keyset name)`, so the roster is no longer deduplicated across data kinds.
> - [ADR 0049](0049-a-dataset-names-one-physical-source-as-a-semantic-source.md): supersedes the disjoint-and-covering invariant with exactly one override row per `(semantic, data_kind)`; `physical_sources` now says which sources consume an expression.
> - [ADR 0050](0050-a-relationship-is-compile-input-not-a-manifest-entity.md): supersedes the `Relationship.name` and `JoinKeySpec`-discriminant clauses on carrier grounds, as set out in 0050's *The relationship is a process input, and that is why it has no entity* and *`cross` is a symmetric tautology, and an empty basis is unreachable*: a relationship is not a manifest entity, its `ai_context` rides the straight unit it produces, `cross` is explicit rather than an empty key vector, and a failed derivation fails compile.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): the override row becomes a signature, one per `(semantic source, definition)`: `{ id, definition_id, expression_id, filter_expr_id, keysets }`, with no `additivity` or `grains`; a `filter:` lowers to its own row-level tree, and `depends_on` is removed; `manifest_version` moves from 2 to 3, and no backward compatibility is owed.

## Context and Problem Statement

Requirement 6b: "all the basic primitives, behavioral or configurational structs coming from model must be included into the resulting manifest after compilation even if they are not consumed by any subsystem yet". Meeting it raises three questions: may documentary fields be excluded, do authored values ride a generic authored-passthrough block or typed fields, and does each typed field attach to the definition or to the override?

## Considered Options

- Typed fields placed by shape lock; flat optional override fields; per-source variation carried by the expression (chosen).
- **A generic authored-passthrough block, or a per-entity annotation map.** Invisible to the manifest's checks and unaddressable by name — a map that must be keyed by convention to reach per-entity docs is the drift magnet.
- **A second polymorphic body beside the definition's `role_definition`.** Worse, and unnecessary because the definition's role already tells a reader which fields are meaningful.
- **A `physical_source_id` on the override.** It reinstates the binding under a new name and duplicates an edge the expression already carries.

## Decision Outcome

Requirement 6b is satisfied with **typed fields only**. There is no generic authored-passthrough block, no per-entity annotation map, and no "documentary fields are excluded" class. Where a value attaches is not decided by guessing which subsystem might read it, but by the rule the ubiquitous language already states: **shape lock**. A semantics' identity-bearing fields are identical at every occurrence of its name, and *only how its value is obtained may vary per data kind*. So a field that says what a value **means** rides the definition, and a field that says how it is **obtained** rides the override.

**The passthrough block is refused, and the argument that would have justified it is void.** The case for excluding documentation rested on recompile churn: persisting `ai_context` would mean an `ai_context` edit alone triggers a recompile. It would not, because the coupling runs the other way. ADR 0041 pins `model_hash` as SHA-256 over `serde_json::to_vec` of the **model** with the root `id` nulled, and its whole argument is **totality** — every authored field participates, and a newly added one is covered the day it lands. Prose therefore moves the fingerprint whether or not the manifest carries it. Excluding documentation buys nothing on the axis it was proposed to protect; it only shrinks the artifact. ADR 0041 states the cost it accepted in the opposite direction — "documentary fields stay out of the manifest, so editing prose churns the fingerprint while producing a materially identical artifact" — and that observation is now the reason to carry them: the churn is paid regardless, so the artifact may as well hold the value.

**A value absent from the manifest is unreachable, not merely untyped.** `CONTEXT.md` defines the semantic manifest as the artifact in which every name, source and expression is "resolved and denormalized, so planning needs no further lookup or I/O", and the only handle back to the model is `SemanticManifestMetadata.source_model_uri: Option<String>`, ratified "never load-bearing: tooling provenance only". There is no reliable route from an artifact to the document that produced it — ADR 0041 makes that explicit, since the fingerprint's preimage nulls the root `id` precisely so the store addresses meaning rather than documents. A dropped authored value is therefore lost, not deferred, and the "we can always read the model" fallback does not exist.

**"Pure documentation" was a mis-classification, and `AiContext` is the case that shows it.** `AiContext` is a **closed v1 roster** — `{synonyms, description, examples, unit}` — so a typed carrier costs one re-exported struct, not N string slots, and the manifest crate already re-exports model enums wholesale (`Grain`, `Integrity`, `UnionMode`, `ScdType`). Two of its four fields are consumed rather than read: a **Request** is "expressed only in semantic names", and `synonyms` is what maps a caller's words onto those names; `unit` is the only numeric-presentation fact the model records. For a semantic layer built to be consumed by an AI, this is not inert prose.

### Shape lock places every behavioral field

**`additivity`, the aggregation family, and the Dimension category ride the definition, because they say what the value means.** They are absent from the reference forms today, and that absence is ADR 0027 enforced **structurally** rather than by a check: `SemanticBody` is exactly `{id, expr}`, and its `expr` field's own doc reads "the one field ADR 0027 makes overridable at a reference site". `MeasureRef` is `{body: Reference, filters}`; `DimensionRef` is `{body: Reference, keyness}`. Neither carries `agg`, `additivity` or `dim_type`. So the whole `DimensionType` family — the six-variant category plus the temporal, bucketed and metadata bodies, including the metadata extraction instruction — and the whole additivity family land on the semantic definition's polymorphic role body. This is the placement ADR 0011's own "wide registry" note anticipated, and ADR 0030 makes it mandatory rather than merely tidy: additivity and temporal shape are explicitly non-inferable, so a planner cannot rebuild what the artifact drops.

**The override carries one field, and reserves two.** `expr_id` is the only element populated. `additivity` and `grains` are present as **optional, flat** fields — reserved headroom for a model that cannot yet author them at a site — and `None` means **"inherit the pooled declaration"**, never "unknown". Because nothing can author them today, that is asserted rather than assumed: while the model has no site-level `additivity` or `grains`, every override's must be `None`, checked. This is the line between reserved headroom and the 55 `carried ⚠` rows the primitives inventory found, where a ratified carrier exists and nothing writes it — an unwritten carrier is a defect, a reserved optional is a contract, and only a pinned meaning for `None` tells them apart. The day the model gains the fields, the check is deleted rather than the drift discovered.

**Flat, not polymorphic, and the imprecision is the price.** A Dimension override carries a meaningless `additivity` slot and a Measure override a meaningless `grains` slot.

**The aggregation family gets no reserved slot.** Additivity has a monotonic reading in the glossary — "an authored declaration may narrow it but never relax it" — so headroom for it is meaningful even before authoring exists. A per-site aggregation family has no such reading: it would make one name two different measures, against **Semantics**'s "one name is always one logical semantics". `filters` likewise gets no slot, because it is **absorbed**: `MeasureRef.filters` folds into `CASE WHEN <filter> THEN <measure.expr> ELSE NULL END`, so a site with filters has a different expression, and that difference is already carried by `expr_id`.

**Per-site additivity, if it is ever wanted, is a separate decision with three named costs.** Fields added to the reference forms; a narrowing-monotonicity check to replace the structural enforcement ADR 0027 currently gets for free; and a `semstrait-model` spec change. It is not smuggled in behind the reserved slot — the slot exists so the persisted shape need not move, not to pre-authorize the authoring change.

### The filter arity question dissolves rather than resolving

**Admitting `Filter` was framed as a persisted-shape change, and it is not one, because `SemanticInterface` stops being an entity.** The framing weighed taking the manifest interface from four role vectors back to five — the constructor from six arguments to seven, `member_count` from four summands to five — as a **second** shape change in the same phase, needing an epoch argument after ADR 0040 spent the single bump. Under the base schema there are no role vectors: a data kind carries a `semantic_bitmask` and nothing else, and a filter is one `semantic_definition` whose role body is a Filter. That is what `CONTEXT.md` already says — Semantics is "one author-declared, globally named element of a data kind's queryable vocabulary — a dimension, measure, metric, **or filter**" — and what **Bit position** already assumes, since "coverage masks address semantics by bit position" and a filter is a semantics. The four-variant `SemanticRole` roster was the drift; the glossary was right.

**So no epoch argument is owed.** ADR 0040's rule stands unchanged: `2` is unshipped, and one bump absorbs every later persisted-shape change until release. Nothing here requests a third value.

**`bitmap.rs`'s "D7" rationale for excluding `Filter` is void, and was already known to be.** No such ratified D7 ever existed; the Filter constraint carrier was explicitly *reserved*; no constraint type exists in the crate to lower into; and gate G5's orphan-expression sweep rejects a pooled predicate that nothing references, so there was no legal way to persist a `DataKindFilter`. With the filter a first-class definition, its predicate is reachable from an override like any other semantics' and G5 is satisfied with no new mechanism.

### Per-source variation lives on the expression, and `SemanticBinding` retires

**The binding was the per-source carrier, and the expression replaces it.** `SemanticBinding` is `{id, data_kind_id, source_id, mapping, resolved}` — one binding per `(data_kind, source)` pair, singular `source_id` — so per-source variation is carried today by there being N bindings for N sources. That is the mechanism a names-only join basis rests on, since the reference corpus re-authors asymmetric join-key pairs onto shared pooled dimensions with per-source mappings. Under the base schema the semantic→physical edge is `override.expr_id` and the expression→source edge is `expression.physical_sources`, so the pair `(semantic, data_kind)` is deliberately **not unique** in the override collection: one row per distinct realization, and which row governs which source is resolved by intersecting that expression's `physical_sources` with the sources being read. Expression exploration is unavoidable for planning, so the expression is the real type for reaching a physical source.

**One invariant replaces the one that retires, and it is the load-bearing product of this ruling.** Today a check requires a binding's `mapping` and `resolved` keysets to coincide — "every covered semantic has exactly one entry in each". That check goes with `ResolvedExprMeta`. In its place: for every `(semantic, data_kind)` pair, the `physical_sources` lists of the override rows sharing that pair must be **disjoint and covering** over that data kind's sources. Disjoint stops two rows claiming the same column for one source; covering stops a source silently lacking a realization for a semantics its bitmask advertises. Without it, "not unique" degrades into "ambiguous", and the failure is a wrong query rather than a rejected compile.

**ADR 0037's `depends_on` relocates onto the expression.** It was placed on `ResolvedExprMeta` to hold the flattened child edges of an inlined nesting; with the binding gone it rides the expression entry, which is where the base schema puts it. The **direction** that field encodes is not settled here — the base schema describes it as a parent reference, which is the reverse of what its name implies and of what ADR 0037 recorded — and is handed to the expression-pool decision rather than assumed.

**`Binding` retires as a glossary term.** It named a single join-point entity; there is no such entity. Its second clause — "Complex data kinds own none and aggregate their children's" — becomes false independently, because a Joinset's `on` and `filter` conditions compile to expressions, so a complex data kind does own expressions.

### The remaining placements

**Documentation attaches wherever it is authored.** The semantic definition carries `description` and `ai_context`; the data-kind carrier gains both; the manifest root gains `description`, `ai_context` and `labels`. `Relationship.ai_context` is **carried**; it was the one row that could have gone the other way: the model type accepts the field while a retired leaf said `AiContext` never attaches to relationships, and with the earlier design deleted no ADR arbitrates. Refusing to persist a field the parser accepts is the worst of the three available states — it is a silent drop with no diagnostic — so it is carried, and ruling the *model* field a mistake is left as the only alternative, at the cost of a `semstrait-model` change this ADR declines to spend.

**The keyset keeps its name.** The base schema's sketch reduces a keyset to `{id, semantics}`. ADR 0040 mints keyset ids over `(name, sorted members)` under a dedicated namespace and gives `KeysetName` its own newtype so that passing a `SemanticsName` where a keyset name belongs is a type error. Dropping the name removes half the dedup seed, collapsing two differently-named keysets over one member set into a single entry, and leaves the explicit unionset-level keyset declaration nothing to reference. Nothing ADR 0040 settled is reopened: the roster stays deduplicated and content-addressed, the reference edge inverts to the data-kind's own keyset list, and the accepted cost 0040 named — that a per-`(data kind, keyset)` fact has nowhere to live — is inherited unchanged.

**`Relationship.name` is owed a carrier**, and one of its two ratified justifications has dissolved. Name-ordered BFS neighbour iteration was a plan-time need; with join paths enumerated at compile there is no plan-time BFS. Scope-local shadow precedence remains — a Joinset-local relationship may redeclare a root-level name and take precedence within that scope — and that resolution is by name. The persisted relationship shape is not ruled here.

**`JoinKeySpec`'s discriminant is owed a carrier, on correctness rather than on 6b.** `Cross` reaches the manifest only as an empty key vector, which is the shape authoring made unspellable, and a resolved `Auto` is byte-identical to a `Declared` basis after the provenance flag was rejected. So "empty" means both "deliberate cartesian product" and "derivation found nothing", and a consumer cannot tell the dangerous state from the failed one. That is a carrier obligation independent of requirement 6b; the shape it takes is the persisted-relationship decision's.

**No derived structure is admitted.** Every placement above is an authored value or an id-bearing primitive. ADR 0011's prohibitions — nodes, edges, `compositions:`, JOIN trees, `ComposedSemanticInterface`, `FieldProvenance`, adjacency indices — are untouched by this ADR. The base schema's enumeration of implicit composition units is a separate decision against 0011 and is not licensed here.

### Amends

- **0011:** `SemanticBinding` retires and `SemanticInterface` stops being an entity.
- **0013:** `SemanticBinding` retires, and `ResolvedExprMeta` with it; per-source variation lives on the expression.
- **0032:** `SemanticBinding` retires and per-source variation lives on the expression; "`Binding` retires as a glossary term", since there is no single join-point entity.
- **0037:** `SemanticBinding` retires and per-source variation lives on the expression; `depends_on` relocates from `ResolvedExprMeta` onto the expression entry.
- **0040:** the reference edge inverts to the data kind's own keyset list, and `SemanticInterface` stops being an entity.
- **0041:** documentation is carried in the manifest, so "documentary fields stay out of the manifest" no longer holds; prose moves the fingerprint whether or not the manifest carries it, so the churn is paid regardless and the artifact may as well hold the value.

### Consequences

**Six rows needed no ruling because a carrier already exists** and only the writer is missing: the data-kind `id` behind the name-as-id stopgap, the leaf `temporal` shape, and the four authored `semantic_interface` projections compile over-approximates from the global registry. Five more dissolved with ADR 0037, which resolves metadata to an ordinary pooled `Literal` and so owes the recipe no separate carrier.

**Twelve rows are `retired` rather than carried** — the verdict the primitives delta pass added as a fourth class. The whole `Keys` / `KeyDecl` / `ForeignKeyDecl` group, the model interface's `keys` field, and both `JoinKeyExprPair` sides are discharged by deleting the authored primitive, not by giving it a home. Counting them as gaps overstates the debt: keyness is carried more strongly than any of them could have been.

**Three groups get their home here and their shape elsewhere.** The twelve storage and catalog rows land on the physical source's storage and schema fields, with the concrete split belonging to the source creation-forks decision. The three nested-Joinset rows are representable the moment a nested unit is an ordinary non-public source with an originator, which is the flatter of the two shapes weighed. The `JoinKeySpec` states are owed a carrier here and shaped there.

**The `semstrait-model` answer is no.** Nothing in this ruling requires an authoring change, provided the override roster stays as ruled. The two places a model change would be needed are both declined: per-site additivity authoring, and retiring `Relationship.ai_context`.
