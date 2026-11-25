---
id: 16-joinset-model
status: ratified
version: 2.1
impl: implemented
---

# Joinset Model

This leaf elaborates the `Joinset` `DataKind` under the unified composition model (2026-04-29). The unified model gives every `Joinset` an `Origin` axis (see [composed-surface-type](composed-surface-type.md) §5.6): `Origin::Explicit` (author-declared `joinsets:` block) or `Origin::Implicit` (compile-enumerated from declared `Relationship`s). Both forms produce the same `ResolvedJoinset` shape (per `33 §4.5`) and are addressable via the same field-first lookup (see [field-first-resolution](field-first-resolution.md)).

**`Joinset` authoritative home.** The detailed `Joinset` authoring surface, planning strategies, and serialization shape are owned by [`../../data-kinds/24_joinset.md`](../../data-kinds/24_joinset.md). This leaf refines `24` by ratifying the composition-level semantics: the `Origin` axis distinction, the scope-local `Relationship` shadow mechanism for per-Joinset join semantics, the canonical-form differentiators that distinguish explicit from implicit, and the role of `traversed_paths` in the `ComposedSemanticInterface`.

> **F3 fix:** This leaf is **seam-less** per the coordinator's correction. `Joinset` as a `DataKind` is defined and owned by `24_joinset.md`. This leaf refines `24` by elaborating the composition-level semantics (explicit vs implicit, scope-local shadow, differentiators). The pre-existing §13 mis-cite "detailed in 23" is corrected to "24" throughout.

## 13. `Joinset` — Explicit and Implicit

`Joinset` (per `12 §5`, detailed in [`24_joinset.md`](../../data-kinds/24_joinset.md)) is the `Relationship`-mediated horizontal composition kind. The unified model (2026-04-29) gives every `Joinset` an `Origin` axis (see [composed-surface-type](composed-surface-type.md) §5.6):

- **`Origin::Explicit`** — author-declared `joinsets:` block. The author owns the name, anchor, traversal order, and any per-leg `JoinType` overrides / filters / `keys`.
- **`Origin::Implicit`** — compile-enumerated from declared `Relationship`s up to the depth bound (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4). Synthetic name + `ImplicitId` per [composed-surface-type](composed-surface-type.md) §5.7. Defaults from the underlying `Relationship` declarations.

Both forms produce the same `ResolvedJoinset` shape (per `33 §4.5`) and are addressable via the same field-first lookup (see [field-first-resolution](field-first-resolution.md) §11).

### 13.1 Role in the composition hierarchy

- **Where `Relationship` is the edge**, `Joinset` is the **named or enumerated walk**. Explicit Joinsets carry an author-declared name; implicit ones carry a synthetic name derived from the canonical-form hash.
- **Both forms are addressable across plan calls.** Explicit Joinsets persist on the SemanticManifest as `DataKindVariant::Joinset` entries (per `33 §6.7`) carrying `anchor` / `members` / `hops` / `path_origin` / `scope_local_relationships` primitives; implicit Joinsets are NOT persisted (cf. C9.2 / `33 §6.8`). Both surface uniformly to the planner as `ResolvedJoinset` shapes inside `SemanticGraph`'s build-time composition index, reconstructed on planner setup. Plan cost is a lookup, not synthesis.
- **Differentiation on canonical form (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6).** An explicit Joinset whose canonical form matches an enumerable implicit Joinset is **rejected at compile** (`COMP_E_0414`). Authors who declare an explicit `Joinset` are committing to at least one differentiator — scope-local `Relationship` shadow, Joinset-level `filter:`, declared `keys`, or a non-shortest path — that makes the canonical form distinct (per §13.5).
- **Depth bound applies only to implicit.** Explicit Joinsets can declare arbitrarily deep traversals (no `MAX_IMPLICIT_COMPOSITION_DEPTH` cap on author-declared paths) and may pick non-shortest paths among alternatives.

### 13.2 Anchored root child

A `Joinset` declares one **root** constituent — the "driving" `DataKind` that all joins hang off. The root's primary key (per `11 §8.3`) becomes the `Joinset`'s composed-surface primary key (per [composed-surface-contents](composed-surface-contents.md) §6.5). Walks proceed from the root outward.

In v1 (per `12 §5.3`'s binary-`Joinset` ratification), a `Joinset` has exactly one non-root constituent — the joined side. The root + one-join structure mirrors the `fact + dim` star-schema pattern but `semstrait` does not privilege one constituent structurally. N-ary `Joinset`s are deferred as `[TD-JOINSET-NARY]` (v2).

### 13.3 Traversed relationships — scope-local declaration

A `Joinset` references one or more `Relationship`s as its traversal path. There is **no per-leg override mechanism**. A `Joinset` that needs join semantics different from a root-level `Relationship` **declares a scope-local `Relationship`** in its own `relationships:` block — the Relationship struct shape, defaults matrix, and validation rules in `18 §2` are identical at the Joinset site. The scope-shadow rule in `18 §2.10` resolves visibility: a Joinset-local name shadows the root-level entry within that Joinset's scope only.

Concretely, an author wanting NULL-padded enrichment for a relationship that the root-level declaration left as `optional: none` (Inner) writes the divergent variant into the Joinset's local `relationships:`:

```yaml
joinsets:
  - name: orders_with_optional_customer
    relationships:
      - name: orders_to_customers           # shadows root-level entry
        from: orders
        to:   customers
        keys:
          - { from: customer_id, to: id }
        cardinality: many_to_one
        integrity: assumed
        optional: left                       # diverges: preserve facts
        cross_filter: left
```

This is the **only** way to vary join semantics inside a Joinset. There is no override map and no per-edge override surface.

**Authoring constraints under scope shadow.** The scope-local Relationship is a full Relationship — `cardinality`, `optional` (where required), and `cross_filter` (where required) MUST be declared per SR-E-4 / SR-E-13 / SR-E-14. Authors cannot "partially override" individual fields of a root-level Relationship.

**Cardinality** is not overridable in any sense: a scope-local Relationship may declare a different cardinality, but doing so asserts a different data-shape claim, not an "override" of the root-level claim. The two Relationships are independent declarations that happen to share a name; the root-level one applies in all other Joinsets and in implicit composition.

### 13.4 Pinned cardinality / fanout

A `Joinset`'s `ComposedSemanticInterface.cardinality_profile` (carried on `ResolvedComplexDataKind`, per `33` and [`24_joinset.md`](../../data-kinds/24_joinset.md)) records the effective cardinality of the composed surface: a fold of per-edge `Cardinality` along the traversal path. For a binary `Joinset` with `OneToMany` on its one edge, the effective cardinality is `OneToMany` (the `from` side is the "anchor", the `to` side is the "fanout"). The planner reuses the pinned profile at plan time for fanout-safe rewrite decisions.

### 13.5 Explicit-implicit reconciliation — clash rejection

Under the unified Joinset model, an explicit `Joinset` whose canonical form (sorted `(EntityId, Direction)` tuples per [composed-surface-type](composed-surface-type.md) §5.7) matches an enumerable implicit `Joinset` is **rejected at compile** with `CompileError::ExplicitImplicitCompositionClash` (see [composition-validation-and-errors](composition-validation-and-errors.md), `COMP_E_0414`). Per [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6 — **the authoritative statement of this rule** — the author-facing message lists candidate differentiators and recommends either dropping the explicit declaration (the planner will use the equivalent implicit form) or adding at least one differentiator.

**Differentiator menu** (any one suffices to make the canonical form distinct):

- **Scope-local `Relationship` shadow.** The explicit Joinset declares a `Relationship` in its `relationships:` block that shadows a root-level one with at least one field diverging — typically `optional`, `cross_filter`, or the residual `filter:`. The canonical form's per-hop `optional` / derived `JoinType` byte differs (per §13.3 example). This is the v1 mechanism for "different join semantics inside this Joinset only".
- **`filter:` predicate on the Joinset body.** The explicit Joinset declares a Joinset-level `filter:` constraint per `12 §5`. The canonical form includes the filter's serialized `PhysicalExpr` byte hash.
- **Declared `keys`.** The explicit Joinset overrides [composed-surface-contents](composed-surface-contents.md) §6.5's derived keys with author-declared ones. The canonical form includes the declared `keys` byte hash.
- **Non-shortest path.** The explicit Joinset traverses a path with hop count strictly greater than the implicit Joinset's shortest-path enumeration would produce for the same constituent set. Implicit enumeration generates only shortest paths, so a longer-path explicit Joinset has no implicit twin.

This replaces Round-1's "no reuse" rule (formerly tracked as `[TD-COMPOSITION-JOINSET-REUSE]`, retired 2026-04-29 — `Q-COMP-012` closed).

**Explicit `Relationship`s that reference a `Joinset`.** [Relationship-edge-model](relationship-edge-model.md) §2.1 permits `Relationship`s whose `from` or `to` is a `Joinset`. The `KeyPair.left` / `.right` references a namespaced `SemanticsName` within the composed surface (e.g. `"order_details.customer_id"`). Such `Relationship`s are declarable and the implicit-Joinset enumeration (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4) walks transparently through the composed kind by treating it as the union of its constituents during canonical-form construction — the prior Round-1 prohibition on chaining is retired (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §9.1 bullet 7). See `Q-COMP-013` (closed 2026-04-29).

## Testing Strategy

- **Invariants:** I8 (manifest carries primitives; explicit Joinsets persist as `DataKindVariant::Joinset` per `33 §6.7`; implicit Joinsets are graph-only per C9.2), I10 (non-exhaustive enums).
- **Explicit vs Implicit:** Tests that `Origin::Explicit` and `Origin::Implicit` produce identical `ResolvedJoinset` shapes for the same constituent set + canonical form, that explicit Joinsets persist on manifest and implicit do not, that both are addressable via field-first lookup (see [field-first-resolution](field-first-resolution.md)).
- **Scope-local shadow:** Tests that a Joinset-local `relationships:` block shadows root-level entries within that Joinset's scope only, that the scope-local Relationship diverges in `optional` / `cross_filter` / `filter` to differentiate canonical form, that cardinality divergence is a data-shape assertion (not an "override").
- **Differentiators:** Tests that each of the four differentiator types (scope-local shadow, `filter:`, declared `keys`, non-shortest path) produces a canonical form distinct from the equivalent implicit Joinset, that `COMP_E_0414` fires when an explicit Joinset lacks all differentiators (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6).
- **Anchored root + cardinality profile:** Tests that the Joinset's anchor constituent's primary key becomes the composed surface's primary key, that the pinned `cardinality_profile` folds per-edge `Cardinality` along the traversal path, that the planner uses the pinned profile for fanout-safe rewrite decisions.
- **Error codes:** `COMP_E_0414` (explicit-implicit clash, see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6 for authoritative statement).
- **Behavioral gates:** Tests that Joinsets traverse `Relationship`s bidirectionally (both forward and reverse), that explicit Joinsets may declare arbitrarily deep traversals (no `MAX_IMPLICIT_COMPOSITION_DEPTH` cap), that transparent unfolding through composed surfaces works (Joinset-over-Joinset permitted).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
