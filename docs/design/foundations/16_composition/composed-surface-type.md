---
id: 16-composed-surface-type
status: ratified
version: 2.1
impl: implemented
---

# ComposedSemanticInterface Type

This leaf ratifies the `ComposedSemanticInterface` struct — the unified queryable surface the planner works against when a Request's selected `Semantics` span multiple `DataKind`s — along with its `CompositionKind` discriminator, the `Origin` axis (explicit vs implicit provenance), the `ImplicitId` canonical-form hash, and the shared `SemanticsView` trait.

> **Scope reconciliation (post-thirteenth-pass cascade rebase, 2026-05-03).** The `ComposedSemanticInterface` shape ratified below survives, but its scope is narrowed in V1 to **two `CompositionKind` variants**: `{Grainset, Joinset}`. The `Unionset` variant is retired from `CompositionKind` because Unionset uses its own bare `SemanticInterface` (per [`../../data-kinds/23_unionset.md §3.2`](../../data-kinds/23_unionset.md)); Unionset's row-wise composition is plan-emission-only and does not need the unified-surface / per-field-provenance machinery `ComposedSemanticInterface` provides. Grainset is added in V1 (per `22 §3.4`) for **cross-grain LEFT OUTER JOIN composition** when no single effective routing unit covers a Request — driver + attached units are equi-joined on shared Keys per `18 §2.5`, mediated by a `ComposedSemanticInterface`. Joinset's per-hop relationship-walk shape is unchanged. Same-grain implicit Unionsets that arise inside a Grainset (per `22 §3.3`) do NOT route through `ComposedSemanticInterface` — they use the bare Unionset SemanticInterface like any explicit Unionset. Cross-grain JOIN-tree details (driver/attached pairing, equi-join Keys per pair) live on `ResolvedGrainset` per `33`, not on `ComposedSemanticInterface` itself; the `traversed_paths: Vec<RelationshipPath>` field below remains Joinset-only.
>
> Body sections below preserve the pre-rebase wording for readability; readers should mentally apply the V1 scope: `CompositionKind` reads as `{Grainset, Joinset}`; references to `CompositionKind::Unionset` and "implicit-Unionset enumeration" are inert in V1 (Unionset enumeration / inference happens inside Grainset and Dataset per their own chapters, not via `ComposedSemanticInterface`). The deeper structural rewrite lands in a follow-up sweep tracked as Q-COMP-006 (post-rebase Unionset-retirement structural cleanup).

## 5. `ComposedSemanticInterface`

```rust
#[non_exhaustive]
pub struct ComposedSemanticInterface {
    pub composition_kind: CompositionKind,
    pub origin: Origin,                          // §5.6 — Explicit vs Implicit
    pub constituents: Vec<DataKindRef>,
    pub interface: UnifiedSemantics,             // see composed-surface-contents
    pub provenance: FieldProvenance,             // see composed-surface-contents
    pub coverage: CompositionCoverage,           // see composed-surface-contents
    pub traversed_paths: Vec<RelationshipPath>,  // §5.2 — Joinset only;
                                                 //        Grainset cross-grain
                                                 //        JOIN-tree lives on
                                                 //        ResolvedGrainset per 33
}
```

The **unified queryable surface** the planner works against when a Request's selected `Semantics` span multiple `DataKind`s. A `ComposedSemanticInterface` is the composition-level analogue of a `SemanticInterface` (`11 §6`), but its shape is materially different: where `SemanticInterface` enumerates the `Semantics` of a single kind, `ComposedSemanticInterface` enumerates the `Semantics` of N kinds reconciled under `UnifiedSemantics`, with per-field ownership (`FieldProvenance`) and per-constituent coverage (`CompositionCoverage`).

### 5.1 Structure

- `composition_kind: CompositionKind` — the kind discriminator (§5.3). V1 carries **two variants**: `Joinset` / `Grainset` (Unionset retired per the ratification note above).
- `origin: Origin` — the provenance axis (§5.6). `Origin::Explicit` for author-declared compositions; `Origin::Implicit { id: ImplicitId }` for compile-enumerated compositions. `Grainset` is always `Explicit`.
- `constituents: Vec<DataKindRef>` — the top-level `DataKind`s participating. Exactly the kinds that contribute at least one field or one edge to the composition. Order is significant — author-declared for `Origin::Explicit`; canonical (sorted by `DataKindName`) for `Origin::Implicit`. For Grainset cross-grain composition (per `22 §3.4`), `constituents` is the effective routing unit list (one per distinct grain), in the canonical order specified by `22 §3.1`.
- `interface: UnifiedSemantics` — the merged semantic surface (see [composed-surface-contents](composed-surface-contents.md) §6).
- `provenance: FieldProvenance` — per-unified-name ownership (see [composed-surface-contents](composed-surface-contents.md) §7).
- `coverage: CompositionCoverage` — per-constituent per-name coverage (see [composed-surface-contents](composed-surface-contents.md) §8).
- `traversed_paths: Vec<RelationshipPath>` — the `Relationship` traversal that produced the composition (§5.2). Empty for `Grainset` (cross-grain composition uses Keys per `18 §2.5`, not Relationships; the per-pair JOIN-key index lives on `ResolvedGrainset` per `33`); non-empty for `Joinset` (regardless of `Origin`).

### 5.2 `traversed_paths`

The `RelationshipPath` struct is owned by [`19 §3.4.5`](../19_expression_flow.md#345-pathsignature) — a `#[derive(Ord, PartialOrd, Eq, PartialEq)]` newtype over `Vec<EntityId>`. This leaf consumes that shape; it does not redefine it.

For `CompositionKind::Joinset` (regardless of `Origin`), this records the `EntityId` chain that produced the composition. Shape is `Vec<RelationshipPath>`, not a single `RelationshipPath`, because the implicit-Joinset enumeration may yield a **tree cover** over 3+ constituents (Steiner tree) — one `RelationshipPath` per "leg" of the tree. Explicit Joinsets with multi-leg traversals (deferred per `[TD-JOINSET-NARY]`) follow the same shape.

For `CompositionKind::Grainset`, this field is empty: Grainset cross-grain composition (per `22 §3.4`) uses entity-level `Key`s per `18 §2.5` for its equi-join structure, not declared `Relationship`s. The per-pair JOIN-key index, the chosen driver, and the attached unit list live on `ResolvedGrainset` per `33 §<ResolvedGrainset>` (forthcoming).

`traversed_paths` is consistent with `19 §3.4.5`'s `PathSignature`: for a cross-kind reference inside a composed Request, the per-expression `PathSignature` is a subset of the composition's `traversed_paths`.

### 5.3 `CompositionKind`

```rust
#[non_exhaustive]
pub enum CompositionKind {
    Joinset,    // horizontal — Relationship-mediated traversal
    Grainset,   // grain-sharded — cross-grain LEFT OUTER JOIN tree on shared Keys
                //                 (single-unit delegation does not synthesize
                //                  a ComposedSemanticInterface; cross-grain
                //                  JOIN composition does)
    // `Unionset` retired in V1 (post-thirteenth-pass cascade rebase, 2026-05-03);
    // see ratification note at top of this leaf. Unionset uses its own bare
    // `SemanticInterface` per `23 §3.2`. Same-grain implicit Unionsets
    // arising inside a Grainset (per `22 §3.3`) are a Grainset-internal
    // construct and do not surface a `CompositionKind` discriminator.
}
```

V1 has **two variants**, narrowed from three. The fourth Round-1 variant — `CompositionKind::Relationship` for "implicit Relationship-driven composition" — was retired (2026-04-29) when the unified Joinset model collapsed implicit and explicit composition into a single kind discriminator + an `Origin` axis (§5.6). The `CompositionKind::Unionset` variant was retired (2026-05-03) per the post-thirteenth-pass cascade ratification: Unionset's row-wise composition does not need `ComposedSemanticInterface`'s unified-surface machinery (a bare `SemanticInterface` plus per-child Coverage suffices, per `23 §3.2`). Implicit Relationship-mediated compositions remain `CompositionKind::Joinset` with `Origin::Implicit`.

**Per-variant origin matrix** (per §5.6):

<!-- cross-cut: none -->

| Variant | `Origin::Explicit` | `Origin::Implicit` |
|---|---|---|
| `Joinset` | author-declared `joinsets:` block (see [joinset-model](joinset-model.md), `12 §5`) | compile-enumerated from declared `Relationship`s (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md), [field-first-resolution](field-first-resolution.md)) |
| `Grainset` | author-declared `grainsets:` block (`12 §4`); cross-grain LEFT OUTER JOIN tree built at compile per `22 §3.4` from declared Keys (per `18 §2.5`) | n/a — Grainset is always explicit (no implicit grain inference in v1) |

Both variants are materialized as `ResolvedComplexDataKind` inside `SemanticGraph` at build time (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md); rebased 2026-05-28 — explicit compositions persist on manifest as `DataKind` entries with the `DataKindVariant` carrying primitives per `33 §6.5–§6.7`, and the `ResolvedComplexDataKind` is reconstructed from those primitives; implicit compositions are graph-only). The author addresses explicit compositions by their declared name (`from: <name>`); implicit compositions are addressed by their `ImplicitId` canonical hash (§5.7), surfaced under a synthetic name the planner assigns at graph build.

`#[non_exhaustive]` per I10. Future MINOR additions (e.g. `Snapshotset`, `Windowset`) are admissible without semver breakage.

**Grainset cross-grain composition shape.** A Grainset's `ComposedSemanticInterface` exists only when the planner needs to assemble a cross-grain JOIN tree (per `22 §3.4` / §4.3) — i.e. when no single effective routing unit covers a Request. The shape:

- `composition_kind = Grainset`.
- `origin = Explicit` (Grainset is always author-declared in V1).
- `constituents` = the effective routing units (one per distinct grain across the Grainset's children, after same-grain pre-merge per `22 §3.3`).
- `interface` = `UnifiedSemantics` lifted from the Grainset's own authored `SemanticInterface`.
- `provenance` = per-Semantics ownership across units (Native / Shared / NullFill per [composed-surface-contents](composed-surface-contents.md) §7).
- `coverage` = per-(unit, Semantics) Coverage entries.
- `traversed_paths` = empty (Grainset cross-grain composition uses Keys per `18 §2.5`, not Relationships; the per-pair JOIN-key index lives on `ResolvedGrainset` per `33 §<ResolvedGrainset>`, not on `ComposedSemanticInterface` itself).

Joinset's per-hop relationship-walk shape is unchanged from prior ratifications and remains the canonical use of `traversed_paths`.

### 5.4 Distinct type vs bare `SemanticInterface` — resolves open item (i)

**Ratified (Q1):** `ComposedSemanticInterface` is a **distinct Rust type**; it is **not** a variant, wrapper, or subtype of `SemanticInterface`. The two share a **common trait** (`SemanticsView`; §5.5) for the accessors both expose, but otherwise have independent structures.

**Alternatives considered and rejected.**

- **Variant of a `SemanticInterface` enum.** Rejected because the two shapes share only a subset of their fields. `SemanticInterface` has no `constituents` or `provenance`; `ComposedSemanticInterface` has no owning `DataKindRef` (it has `Vec<DataKindRef>` instead). Shoehorning both into one enum forces every consumer to match on every variant, even when only one is semantically possible.
- **Wrapper over `SemanticInterface`.** Rejected because `UnifiedSemantics` is not a `SemanticInterface` — it reconciles N of them. A wrapper would suggest `ComposedSemanticInterface.inner: SemanticInterface` plus decorations, but no single `SemanticInterface` captures the unified surface.
- **Subtype via trait inheritance (`ComposedSemanticInterface: SemanticInterface`).** Rejected because Rust does not have subtype polymorphism for structs. A trait-based encoding is exactly what `SemanticsView` is.

**Rationale for distinct type.**

- **Pattern-matching clarity.** Planner code that needs to distinguish composed vs non-composed surfaces pattern-matches on the type or the source of the surface; a single enum would hide this distinction behind a variant check.
- **Field shape independence.** Future revisions to either surface (e.g. adding `materialization_mode` to `ComposedSemanticInterface`) do not force a matching field on `SemanticInterface`.
- **Trait-based convergence where useful.** For the few operations that make sense on both (query "does this surface expose a dimension named X?"), `SemanticsView` provides a uniform accessor (§5.5).
- **Serialization.** `SemanticInterface` serializes as a single-kind manifest entry; `ComposedSemanticInterface` serializes (when persisted — only for explicit compositions; see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md)) as a `ResolvedComplexDataKind` with materially different on-disk shape.

### 5.5 `SemanticsView` trait

```rust
pub trait SemanticsView {
    fn dimension_names(&self) -> &[SemanticsName];
    fn measure_names(&self)   -> &[SemanticsName];
    fn metric_names(&self)    -> &[SemanticsName];
    fn key_names(&self)       -> &[SemanticsName];
    fn filter_names(&self)    -> &[SemanticsName];
    fn has_semantics(&self, name: &SemanticsName) -> bool;
    fn semantics_role(&self, name: &SemanticsName) -> Option<SemanticsRole>;
}
```

Both `SemanticInterface` and `ComposedSemanticInterface` implement `SemanticsView`. `SemanticsRole` (from `11 §6`) is the enum `Dimension` / `Measure` / `Metric` / `Filter` / `Key`. This trait is the planner's primary shape-query interface; strategy code that doesn't need to distinguish composed vs non-composed surfaces operates generically over `&dyn SemanticsView`.

Where strategies need to distinguish:

- `&SemanticInterface` directly exposes the owning `DataKindRef`.
- `&ComposedSemanticInterface` directly exposes `constituents`, `provenance`, `coverage`, `traversed_paths`.

### 5.6 `Origin` axis

```rust
#[non_exhaustive]
pub enum Origin {
    Explicit,
    Implicit { id: ImplicitId },
}
```

**Why an axis, not separate types.** The unified-Joinset model (2026-04-29) collapsed the previous `CompositionKind::Relationship` (implicit, planner-synthesized) into `CompositionKind::Joinset` with `Origin::Implicit`. Both forms share the same struct shape, the same `UnifiedSemantics`, the same `FieldProvenance`, and the same plan-time contract — they differ only in **where the composition's identity comes from**.

- **`Explicit`** — author-declared `joinsets:` / `unionsets:` / `grainsets:` block. The composition's name (`DataKindName`) is the author's text; the planner addresses it directly. Author MAY declare overrides (per-leg `JoinType`, `keys`, filters) that differentiate it from the equivalent implicit composition.
- **`Implicit { id: ImplicitId }`** — compile-enumerated from declared `Relationship`s (Joinset) or coverage overlap (Unionset). Compile assigns a synthetic `DataKindName` derived from the `ImplicitId` (§5.7) and indexes it under both that name and the canonical hash. No author overrides — the composition uses defaults from the underlying `Relationship` declarations.

`Grainset` is always `Origin::Explicit` in v1; there is no implicit grain inference. (Future v2 might introduce `Origin::Implicit` for catalog-discovered grain hierarchies, tracked as `[TD-GRAINSET-IMPLICIT]`.)

**Equivalence under `Origin`.** Two compositions with the same `composition_kind`, the same `constituents` set (as an unordered set), and the same canonical form (sorted `(EntityId, direction)` for Joinset; sorted `Vec<DataKindRef>` for Unionset) are **equivalent** by canonical form. The implicit-explicit clash check (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6) detects when an `Origin::Explicit` composition has the same canonical form as an enumerable `Origin::Implicit` and rejects it (`COMP_E_0414`).

**Plan-time semantics are origin-agnostic.** Strategies, the field-first resolver, and adapter rendering all operate on the struct shape; they do not branch on `origin`. The axis matters at compile (enumeration, clash detection) and in author diagnostics (error messages cite the `name` for explicit, the canonical `ImplicitId` for implicit).

`#[non_exhaustive]` per I10.

### 5.7 `ImplicitId` — canonical-form hash

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImplicitId(pub [u8; 32]);
```

A 32-byte content-stable hash of the composition's **canonical form** — the sorted, normalized, byte-stable encoding of the structural identity. Hash function is BLAKE3-256 (the same primitive `13 §5` uses for `SourceHash`); collision-resistance and speed both more than adequate at v1 scale. The exact byte-encoding is `pub(crate)` (compile internals); public surface is the 32-byte tag and round-trip equality.

**Canonical form per `composition_kind`:**

- **`Joinset`.** Sorted `Vec<(EntityId, Direction)>` — the set of `Relationship` traversals, each tagged with its direction (forward / reverse). `EntityId` is the relationship's durable id (`18 §2.1`), stable across recompiles. Sort key: `(EntityId, Direction::Forward < Direction::Reverse)` — lexicographic by `EntityId` text.
- **`Unionset`.** Sorted `Vec<DataKindRef>` — the set of constituent top-level kinds covered by the implicit Unionset, each represented by its `DataKindName`. Sort key: `DataKindName` lex order.
- **`Grainset`.** Not applicable — Grainset is always `Origin::Explicit` in v1.

**Stability properties.**

- **Within one SemanticManifest.** `ImplicitId` is fully stable — identical canonical forms always hash to identical bytes.
- **Across recompiles.** `ImplicitId` is **stable** across recompiles **when the underlying relationship `EntityId`s are stable** — i.e. relationship ids are authored (`IdentityProfile::StrictRequireProvided`) or persisted and round-tripped. Adding a new `Relationship` does not renumber existing ones, so previously-derived `ImplicitId`s are unchanged. (Strengthened guarantee under the id-first rework, STATUS item U.2 — previously `ImplicitId` was unstable because it rode the retired compile-counter handle.) **Caveat:** under the default `ConvenienceGenerateMissing` profile (`32 §9.0.1`), relationships authored without an `id` receive a fresh UUIDv7 per parse, so their `ImplicitId`s drift across fresh parses unless the generated ids are persisted. Synthesized entities' ids (deterministic UUIDv5) remain stable regardless of profile.
- **Across runs of the same SemanticManifest.** Stable, because `SemanticManifest` is byte-deterministic per `33 §4`'s determinism contract.

**Synthetic name derivation.** Compile assigns each implicit composition a `DataKindName` of the form `__implicit_{joinset|unionset}_{first-8-hex-chars-of-ImplicitId}`. The double-underscore prefix and `__implicit_` namespace are reserved per `11 §3.2` (no author may declare a `DataKindName` matching this pattern; `ValidateError::ReservedName` fires per `11 §14.x`). The synthetic name lets the planner address implicit compositions through the same `name_index` (held on `SemanticGraph`'s build-time composition index; not on manifest) it uses for explicit ones; collisions on the 8-hex prefix are resolved by extending the suffix to full 64 hex chars (extremely rare at v1 scale).

## Testing Strategy

- **Invariants:** I10 (non-exhaustive enums: `CompositionKind`, `Origin`), I8 (manifest carries primitives only; `ComposedSemanticInterface` is graph-only except for explicit's per-variant primitive carriage).
- **Structural validations:** `SemanticsView` trait coverage (both `SemanticInterface` and `ComposedSemanticInterface` implement it), distinct-type resolution (Q1 — not a variant/wrapper/subtype).
- **Canonical-form stability:** `ImplicitId` hash stability across recompiles (when relationship `EntityId`s are stable), synthetic name derivation collision-free under 2000-cap enumeration.
- **Error codes:** `COMP_E_0414` (explicit-implicit clash; see [composition-validation-and-errors](composition-validation-and-errors.md)).
- **Behavioral gates:** Tests that `Origin::Explicit` and `Origin::Implicit` produce identical `UnifiedSemantics` for the same constituent set + canonical form; that `ImplicitId` deterministically re-hashes to the same bytes; that synthetic names do not collide under normal enumeration loads.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
