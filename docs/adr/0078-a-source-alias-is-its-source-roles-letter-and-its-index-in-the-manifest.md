---
number: 78
title: A source alias is its source role's letter and its index in the manifest
status: accepted
date: 2026-10-04
source:
- CONTEXT.md
- docs/adr/0036-bit-position-assignment.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0057-a-join-conditions-column-leaf-carries-its-constituents-alias.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0070-a-path-source-is-one-base-path-read-in-one-format.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0073-a-strategy-only-decides-and-the-driver-builds.md
- docs/adr/0075-a-join-condition-reads-each-constituents-semantics-and-every-output-name-is-the-one-its-parent-reads.md
- '`SourceAliases::derive` in `crates/semstrait-manifest/src/manifest/source_alias.rs` and `letter` in `crates/semstrait-manifest/src/manifest/semantic_source.rs`'
- '`crates/semstrait-manifest/tests/joinset_enumeration.rs`'
links:
- target: 57
  kind: amends
- target: 75
  kind: amends
- target: 70
  kind: amends
- target: 85
  kind: amendedby
---

# A source alias is its source role's letter and its index in the manifest

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0085](0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md): in *An alias is `ss`, its source role's letter and its index*, "An alias holds no dot, so ADR 0057's cut at the first dot still returns the whole alias" has no subject, because nothing in the manifest is cut; an alias still holds no dot. The rest of 0078 stands.

## Context and Problem Statement

A **source alias** names a semantic source in a qualified reference, such as a
join condition leaf `<alias>.<name>`. The Strategy routes a name by the alias in
a composed signature's leaf, `Column(<operand alias>.<name>)` (ADR 0073, *One
decision shape, checked per body*). ADR 0075, in *Names are unique, and the
planner's own names are reserved*, makes unique aliases "a precondition of these
rules, not a result of them".

ADR 0057, in *The alias is minted from the name, never from a counter*, derives
the alias from the semantic source's name: compile cleans the name to
engine-safe text, cuts it to a fixed length, and "appends a short suffix derived
from the source's `EntityId` where sanitising or cutting would collide with
another alias". On `main`, `plan_aliases` adds the first six characters of each
id to a base that two semantic sources share, and does not check the result
again. So a name can equal another semantic source's suffixed alias.

A probe compiles one Joinset `sales` with
the constituents `eu-orders`, `eu_orders` and `eu_orders_fb2f09`. Compile gives
no warning, and `validate()` accepts the manifest:

```
sales.eu-orders         alias  sales_eu_orders_fb2f09
sales.eu_orders         alias  sales_eu_orders_2d7224
sales.eu_orders_fb2f09  alias  sales_eu_orders_fb2f09
sales.r1                on     sales_eu_orders_fb2f09.customer_id = sales_eu_orders_fb2f09.customer_id
```

Two semantic sources share one alias, and the join condition of `r1` compares a
column with itself.

## Considered Options

- **`ss`, the source role's letter, and one index for the manifest, in
  `EntityId` order** (chosen).
- **A name-derived alias with the id suffix, and a compile refusal when an alias
  repeats.** Refused because a name-derived alias can repeat, so compile needs
  the cleanup, the cut, the suffix and a refusal. The human: "that way we will
  never overlap".
- **A name-derived alias whose suffix grows until no alias repeats.** Refused on
  the same ground.
- **One index for each source role** (`ssd0, ssd1, ssj0`). Refused because the
  human asked for a "unique number for manifest". With one count, the number
  alone is unique.
- **An index in the order in which compile walks the semantic sources.** Refused
  because the order of a walk is not a property of the manifest, so reordering
  the semantic model could change aliases. In `EntityId` order, as ADR 0036's
  *Decision Outcome* says of `bit_position`, "Positions are a function of the
  registry's key set".
- **A `validate()` check for repeated aliases only**, compared without regard to
  case. Refused because ADR 0042 makes `validate()`'s check set every stage's
  rule, and a check for repeats passes an alias with a dot, an empty alias and an
  alias with the wrong letter.

## Decision Outcome

### An alias is `ss`, its source role's letter and its index

Compile writes each semantic source's alias as `ss`, then one letter for the
semantic source's source role, then the semantic source's index:

| source role | letter |
|---|---|
| Dataset | `d` |
| Unionset | `u` |
| Grainset | `g` |
| Joinset | `j` |

The index counts from 0 over all of `semantic_sources`, in canonical `EntityId`
order, as ADR 0036 numbers `bit_position`. A source role that is added later
gets one more letter.

The letter follows the source role, also for a semantic source that compile
makes. A Dataset `orders` whose storage entry expands to three physical sources
becomes an implicit Unionset `orders` over three reduction leaves, so `orders`
gets `u` and each reduction leaf gets `d`. A Joinset
sub-composition such as `sales.r1` gets `j`.

This form is chosen because it gives three properties:

- **The alias is short.** It is `ss`, one letter and the digits of one index,
  at every composition depth. A qualified name such as `ssd4.customer_id` is the
  alias, a dot and a semantics' name, so the alias adds few characters to an
  output field's name. A semantic source's name grows with composition depth,
  and an engine can cut or refuse an identifier that is longer than its limit,
  an alias and an output field's name alike.
- **No two aliases are equal.** No two semantic sources have one index. An alias
  holds no dot, so ADR 0057's cut at the first dot still returns the whole alias.
- **The alias shows the source role.** In a plan, or in SQL emitted from a plan,
  the letter shows the source role of the semantic source that supplies a
  qualified name: `ssj5.region` comes from a Joinset, and `ssd0.region` from a
  Dataset. The alias does not show which semantic source it is.

The Joinset gets these aliases:

| `EntityId` (first 8) | semantic source | alias |
|---|---|---|
| `1d26bbb0` | `sales.eu_orders_fb2f09` | `ssd0` |
| `2d72245f` | `sales.eu_orders` | `ssd1` |
| `a1415b51` | `sales.r1` | `ssj2` |
| `f855b64c` | `sales.r2` | `ssj3` |
| `fb2f09ca` | `sales.eu-orders` | `ssd4` |
| `fef513d0` | `sales` | `ssj5` |

The join condition of `r1` is then `ssd4.customer_id = ssd0.customer_id`.

### `validate()` derives each alias again

An alias is a `pub` field, so a manifest decoded from bytes can hold any alias.
ADR 0042, in its *Decision Outcome*, says that "`validate()`'s check set is the
conjunction of every stage's validity definition". So `validate()` derives each
semantic source's alias again by the rule above, and refuses an alias that
differs with `CompileError::SourceAliasMismatch { semantic_source, expected,
found }`. `semantic_source` is the semantic source's `EntityId`. The message is
"semantic source `{semantic_source}` carries the alias `{found}`; its source role
and index give `{expected}`".

A repeated alias, an alias with a dot, an alias that differs only in case and an
empty alias all fail this check. Compile does not raise this variant.

### Amends

- **0057:** in its *Decision Outcome*, the alias is not "derived from that
  source's name"; in *The alias is minted from the name, never from a counter*,
  the derivation from the name, the collision suffix and the paragraph that
  begins "**A counter is refused.**" are replaced by 0078's *An alias is `ss`,
  its source role's letter and its index*, and "The alias is not a duplicate of
  the name, and the distinction is why it earns a field" and "The reverse
  direction is not persisted" stand; in *The check ADR 0050 owed is now
  definable*, aliases are distinct by construction, and `validate()` derives each
  alias again (0078's *`validate()` derives each alias again*); the test of
  resolution without regard to case stands. The rest of 0057 stands.
- **0070:** in *A reduction leaf is minted from its Dataset and its physical
  source*, "**Engine aliases stay unique.**" holds by 0078's *An alias is `ss`,
  its source role's letter and its index*, not because "`mint_alias`
  (`semantic_source.rs`) adds an id suffix to an alias whose name collides". The
  rest of 0070 stands.
- **0075:** in *Names are unique, and the planner's own names are reserved*,
  source aliases are unique by construction (0078's *An alias is `ss`, its
  source role's letter and its index*), so "ADR 0057 mints an alias with no dot
  and intends aliases to be disjoint" reads 0078's rule, and the sentence that
  begins "Compile's minting does not check its suffixed aliases again" and "How
  uniqueness is guaranteed is not ruled here" no longer hold; "Unique aliases are a
  precondition of these rules, not a result of them" stands; in *Considered
  Options*, the option "An earlier working design's `<name>$filter`, or a counter" stays refused on
  both its grounds, and "which ADR 0057 refused for aliases" no longer holds.
  The rest of 0075 stands.

### Consequences

- **An alias holds inside one manifest only.** Adding or removing a semantic
  source, by an edit of the semantic model or by a change in the physical
  sources that a Dataset's storage entry expands to, changes the index of each
  semantic source after it in `EntityId` order. A Dataset whose storage entry
  goes from one physical source to several changes its letter from `d` to `u`. ADR 0036 scopes `bit_position` to one artifact too.
- **An alias reaches the format in one place.** ADR 0075, in *Every output name
  is the one its parent reads*, says that "the empty read of a Unionset (ADR
  0073) gives the union's output names" as the read's base schema, so such a
  read carries names like `ssu3.region`. Elsewhere an alias appears only in
  manifest expressions and in plan output schemas, where `ssd4.customer_id` is
  harder to read than a name-derived alias.
- **Example aliases in ADRs 0057, 0065, 0066, 0071 and 0075** (`orders_daily`,
  `orders`, `sales_orders`, `sales_r1`) are derived from names. Each stands for
  that semantic source's alias, and those ADRs are not amended for it.

## More Information

The probe's Joinset and its aliases are pinned by a test in
`crates/semstrait-manifest/tests/joinset_enumeration.rs`.
