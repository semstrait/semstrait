---
id: 24-joinset-canonical-shape
status: ratified
version: 2.1
impl: drift
---


# 24. Joinset

> **Reconciliation (Phase-3, 2026-04-17; relationship-block rebase 2026-05-12).** The v1 authoring-layer canonical shape for `Joinset` is ratified across:
>
> - `[../apis/32_semstrait_model.md §3](../../apis/32_semstrait_model.md)` — top-level YAML tag (`joinsets:`), `JoinsetBody` struct shape.
> - `[../foundations/18_entities.md §2](../../foundations/18_entities.md)` — canonical `Relationship` struct (unified across root-level `relationships:` and `JoinsetBody.relationships`). Key decisions:
>   - `JoinsetBody.relationships: BTreeMap<EntityId, Relationship>` — no separate `JoinRelationship` type. The unified struct is **semantic-first**, carrying `{ id, name, from, to, keys, filter?, cardinality, integrity, optional?, cross_filter?, ai_context?, description? }`.
>   - `JoinType` is **derived** at compile from `Relationship.optional` per `18 §2.9`, not authored. v1 roster: `{Inner, Left, Right, Full}`, `#[non_exhaustive]`. `AsOf` is descoped for v1 (post-v1 deferred per `17`).
>   - Join keys shape: `keys: [{from: <SemanticExpr>, to: <SemanticExpr>}, …]` equi-pair list + optional `filter: <SemanticExpr>` residual predicate.
>   - `cardinality:` is required at every Relationship authoring site (SR-E-4). `optional:` and `cross_filter:` are required on `OneToOne` / `ManyToMany` (SR-E-13); directional `cross_filter` is forbidden on `ManyToMany` (SR-E-14).
>   - The earlier `directionality:` field and `Directionality` enum are **retired (2026-05-12)** — every Relationship is bidirectional by construction; authors who need to forbid auto-synthesized reverse traversal declare an explicit Joinset.
>   - The earlier per-hop **`JoinTypeOverrides` / `HopPosition`** carriage on `JoinsetDataKind` is **retired (2026-05-12)**. A Joinset that needs different join semantics declares a **scope-local `Relationship`** in its own `relationships:` block; the scope-shadow rule in `18 §2.10` resolves visibility. See `16 §13.3` for the sole permitted mechanism.
> - `[26_nesting_matrix.md](../26_nesting_matrix.md)` — nesting rules. Notably **R3** (every `ComplexDataKind` requires ≥ 2 children).
> - `[../questions/deferred/24_questions.md](../../questions/deferred/24_questions.md)` — Q-24-09 (`JoinAssociativity`) + Q-24-10 (star / snowflake / 3NF shape-tag vocabulary), folded from the former `joinset_shape_semantics.md` sidecar on 2026-04-17.
>
> This document retains authority for:
>
> - The **anchor** contract (§3) — mandatory single root child, FROM-clause semantics, fan-out reference frame.
> - The **join-path** contract (§4) — explicit-path vs implicit-path authoring modes and their interaction with the Relationship graph.
> - `JoinsetStrategy` planner contract (§5) — lowering to `PlanNode::Join` sequences in anchor-outward order; per-hop `Cardinality` propagation; per-hop `JoinType` is derived from each traversed Relationship's `optional` field per `18 §2.9` (no override surface).
> - `VALID_E_24NN` / `COMP_E_24NN` / `PLAN_E_24NN` / `PLAN_W_24NN` error-code allocations.
>
> Rust-struct and YAML-surface body sections predate `18` (formerly `32c`); `JoinRelationship` / `JoinTypeOverrides` / `HopPosition` in legacy body text are pre-unification or pre-rebase vocabulary — read the `Relationship` shape from `18 §2`. `AsOf` sections are forward-reference only; v1 does not emit `AsOf` joins. `ColumnMapping` → `SemanticMapping` rename per `18 §10`.

## 1. Purpose and Scope

### 1.1 What `24` ratifies

`24` is the canonical-layer specification for the `**Joinset`** variant of `DataKind`. It fills in what `16 §13` deferred:

- **§2** — the `JoinsetDataKind` canonical struct shape: `{ anchor, members, path, relationships, interface, ... }`, its placement as a `ComplexDataKind` variant, and how it composes via `CompositionKind::Joinset`.
- **§3** — the **anchor** contract: exactly one root child, which member plays the FROM-clause role, which member drives the fan-out reference frame. Rationale for mandating the anchor (determinism, fanout predictability, and traversal ambiguity resolution).
- **§4** — the **join-path** contract: the two authoring modes (implicit path = name the members, let the planner compute; explicit path = pin a `EntityId`-indexed traversal), their selection rules, and their interaction with `16`'s Relationship graph.
- **§5** — the `**JoinsetStrategy`** planner contract: how the planner lowers a resolved `Joinset` into `PlanNode::Join` nodes in anchor-outward order, how per-hop `JoinType` is derived from each traversed Relationship's `optional` field (per `18 §2.9`; scope-local Relationship shadow per `18 §2.10` is the sole mechanism for divergent join semantics), how `Cardinality` propagates, and how post-join Project nodes reconcile the unified surface per `16 §6`.
- **§6** — `Joinset` as a **consumer** (not declarer) of `Relationship`s. `Joinset`'s `path` references Relationships by id or by citing the Relationship graph; it never introduces a new Relationship.
- **§7** — the `**TemporalShape`** interaction: as-of join gating — `JoinType::AsOf` is legal only when the hop's `to`-side carries a temporal shape that `17 §5` sanctions (`Events ↔ Snapshot`, `Events ↔ SCD`, etc.). `17` is parallel-drafted; `24` fixes the integration points so the hookup is mechanical once `17` lands.
- **§8** — the `**ComposedSemanticInterface`** the `Joinset` publishes: how `UnifiedSemantics` (`16 §6`), `FieldProvenance` (`16 §7`), and `CompositionCoverage` (`16 §8`) apply under `CompositionKind::Joinset`.
- **§§9–11** — the validate-, compile-, and plan-stage error rosters specific to `Joinset`, allocated against the `*_E_24NN` / `PLAN_W_24NN` ranges.

### 1.2 What `24` does NOT ratify (forward-refs)

- `**Relationship` shape, `Cardinality`, `JoinType`** — all three live canonically in `16 §§2–4`. `24` cites and consumes; it never redefines.
- **Implicit composition for `Request.from = None`** — that is `16 §11`'s field-first lookup over compile-time-enumerated compositions (`16 §10.4`). Both implicit and explicit Joinsets produce `CompositionKind::Joinset`; the distinguishing axis is `Origin` per `16 §5.6` (`Origin::Explicit` for author-declared, `Origin::Implicit { id }` for compile-enumerated). §1.3 below sharpens the boundary.
- `**TemporalShape` variants and the `Additivity × TemporalShape` matrix** — `17`. `24` only records the integration point for `JoinType::AsOf` activation.
- **N-ary `Joinset` authoring shape** — deferred. `12 §5.2` ratifies binary-arity for v1 under `TD-NESTING-NARY-JOIN`; `24 §2.5` restates the constraint and tracks the semantic shape the N-ary extension will take. The canonical struct in §2.2 is written for N-ary to keep the MINOR lift-to-N-ary purely structural (adding `path: Vec<JoinHop>` rather than swapping in a new struct).
- **YAML surface for `joinsets:`** — `32`. The `path.on.left` / `path.on.right` structural-label-dotted-column form declared in `12 §5.1` is the v1 YAML convention; `24` reasons about the canonical-layer shape (post-parse, `ResolvedJoinset`).
- `**ResolvedJoinset` / `PlanNode::Join` exact field rosters** — `33` / `35`. `24` fixes the carriage contract (which fields each layer needs) without freezing struct layout.

### 1.3 Sharpening the implicit-vs-`Joinset` boundary

Every Joinset could, in principle, be expressed by declaring the constituent DataKinds plus declaring the required Relationships, and then issuing a `Request.from = None` — `16 §11`'s field-first algorithm would synthesize a `ComposedSemanticInterface` over the same constituents. The two are **not** the same object, and Round-1 design preserves that distinction:


| Dimension                   | Implicit (`16 §11`)                                     | `Joinset` (`24`)                                                  |
| --------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `CompositionKind`           | `Relationship`                                          | `Joinset`                                                         |
| Named / addressable         | No (anonymous, Request-scoped)                          | Yes (a `DataKindRef`; valid `Request.from`)                       |
| Anchor                      | None (BFS is symmetric over the Semantics owners)       | Mandatory exactly-one root (§3)                                   |
| `JoinType` per hop          | Derived from each Relationship's `optional` field       | Same derivation; divergent semantics via scope-local Relationship shadow (`§5.3.2`) |
| SemanticManifest lifecycle  | Synthesized at `plan` (`16 §10.1`)                      | Materialized at `compile` (`16 §10.1`)                            |
| Depth bound                 | `MAX_IMPLICIT_COMPOSITION_DEPTH = 4` (`16 §9.1`)        | None (authoring a `Joinset` is the escape hatch)                  |
| Ambiguous paths             | `PLAN_E_0500 AmbiguousImplicitComposition` (`16 §14.3`) | Author pins via explicit path (§4.2); never an error at plan time |


A `Joinset` is what the author reaches for when they need (a) a named surface, (b) a pinned path, (c) divergent join semantics on a hop (via scope-local `Relationship` shadow per `§5.3.2`), or (d) a composition deeper than 4 hops. Everything else an author might otherwise reach for a `Joinset` for is served by implicit composition + declared Relationships.

### 1.4 Invariants `24` directly upholds

- **I1 — canonical layer.** A `Joinset`'s anchor, members, and path references are `DataKindRef` / `EntityId` handles; never SQL column names. Resolution to physical join predicates happens during `JoinsetStrategy` → `PlanNode::Join` lowering, which delegates to `15`-resolved `Binding.column_mapping` at each hop.
- **I4 — determinism.** For a fixed SemanticManifest and a fixed `Joinset`, the materialized `ComposedSemanticInterface` is bit-identical; the planner's `JoinsetStrategy` emits the same `PlanNode::Join` sequence on every invocation. Implicit-path resolution uses `16 §11.4`'s deterministic neighbor order (extended for anchor bias in §4.1.3).
- **I5 — compile-time resolution.** `Joinset.path` (whether implicit or explicit) is fully resolved to `Vec<EntityId>` at `compile`. The `plan` stage never re-walks the Relationship graph for a `Joinset`.
- **I7 — strict crate DAG.** `Joinset` resolution lives in `semstrait-manifest`; `JoinsetStrategy` lives in `semstrait-planner`; `PlanNode::Join` construction consumes the resolved `Joinset` without re-resolution.
- **I8 — SemanticManifest is planner-complete.** `ResolvedJoinset`'s **manifest contract** persists the anchor (`EntityId`), the binary-v1 `members: Vec<EntityId>`, `Vec<JoinsetHop>` (each hop carrying `from` / `to` (`EntityId`) / `relationship: EntityId` / `direction: HopDirection` / `hop_coverage: SemanticBitmask` with cumulative semantics per `33 §6.7`), `path_origin: PathOrigin { Explicit, Implicit }`, and `scope_local_relationships: Vec<Relationship>` for `18 §2.10` shadowing. The resolved `ComposedSemanticInterface` (`UnifiedSemantics` + `FieldProvenance` + `CompositionCoverage`) is **deferred to graph build** (see Phase 3 amendment below); planner-completeness is preserved because `SemanticGraph` synthesises the unified surface deterministically from manifest primitives (anchor + per-hop coverage + members' interfaces) at build time. `JoinType` is still derived at plan emission from the resolved Relationship's `optional` field per `18 §2.9`.

> **Phase 3 amendment (2026-05-28; cascade from C7.4).** Earlier drafts of `ResolvedJoinset`'s manifest contract included the resolved `ComposedSemanticInterface` (`UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, `traversed_paths`) as a persisted field. Per ratified clause C7.4 (manifest-pass `RATIFICATION_LOG.md`), these are dropped from the manifest contract and re-classified as **graph-build outputs**. The manifest now persists only the path primitives — anchor, members, `Vec<JoinsetHop>` with cumulative `hop_coverage`, `path_origin`, and `scope_local_relationships` — plus the universal top-level `coverage: SemanticBitmask` (which equals `hops.last().hop_coverage` per CCK.1 and `33 §6.7`). The `ComposedSemanticInterface` synthesis at graph build time uses (a) per-hop `hop_coverage` for first-cut path search and (b) the members' canonical `SemanticInterface`s for the unified-surface merge per `16 §6`. The cross-references for the manifest-resident shape are [`../apis/33_semstrait_manifest.md §6.7`](../../apis/33_semstrait_manifest.md) (`DataKindVariant::Joinset` + `JoinsetHop`) and `33 §10` (CX1 load-time integrity). The plan-time observable behaviour in `24 §5` is unchanged — `JoinsetStrategy::lower` still consumes a fully-resolved hops sequence; the difference is that the unified-interface materialisation is synthesised into the graph during build, not loaded from the manifest. `§2.4`'s pseudo-shape and `§8`'s `ComposedSemanticInterface`-on-Joinset narrative are recast accordingly: the same content exists, but at graph-build time, not at manifest-load time.
- **I10 — non-exhaustive extensibility.** `JoinsetDataKind`, `JoinHop`, `ExplicitPath`, `JoinsetStrategy`'s inputs / outputs are all `#[non_exhaustive]`. N-ary lift (`TD-NESTING-NARY-JOIN`) and `JoinType::AsOf` activation (pending `17`) are MINOR additions.
- **I12 — first-class diagnostics.** Every `Joinset` Precondition has a stable error code in the `*_E_24xx` / `PLAN_W_24xx` ranges (§§9–11).

## 2. The `Joinset` Variant

### 2.1 Placement in the DataKind taxonomy

`Joinset` is one of the four `CompositionKind` variants ratified in `16 §5.3`:

```rust
// (From `16 §5.3`, quoted for reference only.)
#[non_exhaustive]
pub enum CompositionKind {
    Relationship,   // implicit (16 §11)
    Unionset,       // explicit — `23_unionset.md`
    Grainset,       // explicit — `22_grainset.md`
    Joinset,        // explicit — this document
}
```

The author-declared top-level kind that lowers to `CompositionKind::Joinset` is the `DataKind::Joinset(JoinsetDataKind)` variant, sketched below.

### 2.2 The canonical struct

```rust
/// Author-declared `Joinset` in canonical-layer (post-parse, pre-bind)
/// form. `21`'s Unionset and `22`'s Grainset peers share the
/// `ComplexDataKind` shape without further coupling.
#[non_exhaustive]
pub struct JoinsetDataKind {
    /// Human-readable canonical name; globally unique per `11 §3`.
    pub name: DataKindName,

    /// The root child — the member that plays the FROM-clause
    /// role and anchors fanout analysis. See §3. MUST be an element
    /// of `members`. Exactly one.
    pub anchor: DataKindRef,

    /// The named subset of DataKinds composed by this Joinset.
    /// Contains `anchor` plus zero-or-more other members. v1 arity
    /// per `12 §5.2` is binary (`members.len() == 2`); §2.5 restates
    /// the constraint. The struct is intentionally N-ary-ready so
    /// `TD-NESTING-NARY-JOIN` lifts without a struct swap.
    pub members: Vec<DataKindRef>,

    /// The traversal specification. `None` → implicit path (planner
    /// computes via §4.1); `Some(ExplicitPath)` → author-pinned path
    /// (§4.2).
    pub path: Option<ExplicitPath>,

    /// Scope-local `Relationship` declarations. The struct shape,
    /// validation rules, and defaults matrix are identical to the
    /// root-level `relationships:` block per `18 §2`. A scope-local
    /// entry that shares its name with a root-level Relationship
    /// shadows the root-level one within this Joinset's scope only
    /// (`18 §2.10`). This is the sole mechanism for varying join
    /// semantics inside a Joinset — there is no per-hop override
    /// surface.
    #[serde(default)]
    pub relationships: BTreeMap<EntityId, Relationship>,

    /// The Joinset's own interface — the declared Dimensions /
    /// Measures / Metrics / Filters / Keys that live at the Joinset
    /// scope. Per `11 §2`, nested members contribute their semantics
    /// through the unified surface; the `Joinset` itself may declare
    /// composition-level semantics (e.g. a composite key, a derived
    /// dimension).
    pub interface: JoinsetInterface,
}
```

Every field is required at the canonical layer except `path`, `relationships`, and (post-compile) the derived `ComposedSemanticInterface`. The struct is `#[non_exhaustive]` per I10; MINOR additions (`fanout_strategy`, `parallelism_hint`, ...) carry explicit defaults.

> **Retired (2026-05-12).** Earlier drafts of this struct carried `overrides: JoinTypeOverrides` and a companion `HopPosition` newtype for per-hop join-type override carriage. Both are removed in v1 in favour of the scope-local `Relationship` shadow model (`16 §13.3`). See `[../questions/closed/24_questions.md](../../questions/closed/24_questions.md)` for the closure record.

### 2.3 What `ExplicitPath` carries

```rust
#[non_exhaustive]
pub struct ExplicitPath {
    /// The ordered sequence of hops from anchor outward. A single-
    /// element `Vec` is the binary-Joinset case. N-ary (`TD-NESTING-
    /// NARY-JOIN`) extends this to a spanning-tree walk; v1 keeps it
    /// linear.
    pub hops: Vec<JoinHop>,
}

#[non_exhaustive]
pub struct JoinHop {
    /// The Relationship traversed by this hop. MUST be a
    /// `EntityId` declared in the SemanticManifest's `relationships:`
    /// top-level block per `16 §2.1`.
    pub relationship: EntityId,

    /// The traversal direction. `Forward` walks
    /// `Relationship.from → Relationship.to`; `Reverse` walks the
    /// opposite. Every `Relationship` is bidirectional in v1
    /// (`16 §2.4`), so either direction is always permissible.
    pub direction: HopDirection,

    /// The DataKindRef that this hop lands on. Redundant with the
    /// underlying Relationship's endpoint under `direction`, but
    /// stored for validation symmetry (compile cross-checks the
    /// match; see §10.1 `COMP_E_2402`).
    pub to: DataKindRef,
}

#[non_exhaustive]
pub enum HopDirection {
    Forward,
    Reverse,
}
```

Round 1 uses `HopDirection` explicitly rather than inferring direction from the ordered `relationship.from` / `relationship.to` fields because the same `EntityId` can be walked either direction on a bidirectional edge, and explicit direction makes validation (§§10.1–10.2) straightforward.

### 2.4 Composition via `CompositionKind::Joinset`

Per `16 §10.1`, an explicit `ComplexDataKind` is materialized at `compile` as a `ResolvedComplexDataKind`. For `Joinset` under the manifest-resident shape (per C7.4 cascade), `compile` produces a manifest entry carrying anchor + members + per-hop coverage + scope-local relationships; the `ComposedSemanticInterface` is synthesised by `SemanticGraph` at build time, not persisted on the manifest:

```text
compile(JoinsetDataKind) → ResolvedJoinset {
    name, anchor, members, hops: Vec<ResolvedJoinHop>, relationships,
    // interface: ComposedSemanticInterface { ... }  — RETIRED at the manifest layer (C7.4).
    //   Synthesised by SemanticGraph at build time from anchor + per-hop coverage +
    //   members' SemanticInterfaces. See `33 §6.7` for the persisted shape and the
    //   Phase 3 amendment under §1.4 above for the cascade rationale.
}
```

> **Retired at the manifest layer (2026-05-28; cascade from C7.4).** The `interface: ComposedSemanticInterface { … }` member of the pseudo-shape above is removed from the manifest contract per ratified clause C7.4. The semantic content (composition kind, anchor-first constituents, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, `traversed_paths`) is unchanged; the materialisation site moves from compile to graph build. `33 §6.7` ratifies the manifest-resident `DataKindVariant::Joinset` and `JoinsetHop` struct roster ([`../apis/33_semstrait_manifest.md`](../../apis/33_semstrait_manifest.md)). `24 §8` (the `ComposedSemanticInterface`-on-Joinset narrative) remains correct as a description of the graph-build output; the only shift is the lifecycle stage at which the unified surface materialises.

`24` continues to fix the semantic content: the `ComposedSemanticInterface` synthesised at graph build for a Joinset MUST have `composition_kind == CompositionKind::Joinset` and MUST populate `constituents` in anchor-first order (see §3.2 for why).

### 2.5 v1 arity — binary only (restatement)

`12 §5.2` ratifies binary-only Joinsets in v1: exactly two members, `path.hops.len() == 1`. `24` does not relax this constraint and does not redefine it; the canonical struct in §2.2 is N-ary-ready to make `TD-NESTING-NARY-JOIN`'s MINOR lift a struct-preserving extension.

Enforcement:

- **Validate.** `members.len() == 2` or `VALID_E_2400 JoinsetArityV1Violation` per §9.1. (`12 §7` also emits its own nesting-shape variant `NV-V5`, ratified there; `24 §9.1` provides the canonical-layer twin so non-YAML construction paths are covered.)
- **Compile.** `hops.len() == 1` for an explicit path, or `implicit_path_result.hops.len() == 1` for an implicit path. N-ary inputs were rejected at validate; this is a defense-in-depth check against mis-compilation (`COMP_E_2408 JoinsetArityV1Invariant`).

For v1 the implicit-path algorithm (§4.1) always produces exactly one hop when it succeeds, or fails with `COMP_E_2401 JoinsetImplicitNoPath` (§10.1).

### 2.6 Nested Joinset interfaces

A `JoinsetDataKind` declares its own `JoinsetInterface` — Dimensions / Measures / Metrics / Filters / Keys at the Joinset scope (per `11 §2`). The canonical-layer rules are:

- **Keys.** MAY be declared on the Joinset (typically echoing the anchor's key tuple per `16 §6.5`). Declared keys participate in `UnifiedSemantics.keys` (§8.2).
- **Dimensions / Measures / Metrics.** MAY be declared at the Joinset scope as composition-level semantics. They are recorded in `UnifiedSemantics` with `FieldOwnership::Derived` (for computed) or `FieldOwnership::Native(anchor)` (for delegation to the anchor member), per `16 §7.3`.
- **Filters.** MAY be declared; they apply at the composed surface, post-join.

The majority of authored Joinsets declare no Joinset-level semantics and rely entirely on the member surfaces unified per §8.2.

## 3. Anchor Specification

### 3.2 Rules for the anchor


| Rule                                                                      | Enforcement                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The anchor MUST be non-empty.                                             | `VALID_E_2401 JoinsetAnchorMissing` — validate-stage structural check.                                                                                                                                                                                              |
| The anchor MUST be a member of `members`.                                 | `VALID_E_2402 JoinsetAnchorNotMember` — validate-stage.                                                                                                                                                                                                             |
| Exactly one anchor per Joinset.                                           | Enforced by `anchor: DataKindRef` being a scalar; multi-anchor declarations are rejected at YAML parse per `32` (the canonical layer only exposes the one-anchor shape).                                                                                            |
| The anchor MUST be reachable by every other member via the resolved path. | `COMP_E_2403 JoinsetAnchorUnreachable { unreachable_member }` — compile-stage, after path resolution. In v1 with binary arity this is degenerate (the other member is directly reached by the single hop); the rule is stated for forward-compatibility with N-ary. |
| The anchor MUST NOT itself be a `Joinset` at the YAML authoring level.    | Same-kind ban per `12 §2` matrix. `Joinset` is not a legal child of `Joinset` at any role, including anchor. Enforced by `12 §7`'s nesting-matrix validate pass; `24` does not re-enforce.                                                                          |


### 3.3 FROM-clause semantics

The FROM-clause role is made explicit in `JoinsetStrategy`'s plan-tree output. For a resolved Joinset with anchor `A` and members `[A, B]`:

```text
PlanNode::Project(<unified surface>)
  └── PlanNode::Join {
          join_type: <per §5.3>,
          left: PlanNode::Scan(A),     // anchor = left
          right: PlanNode::Scan(B),    // reached member = right
          keys: <from Relationship.keys, resolved via `15`>,
          from_relationship: Some(<hop_0.relationship>),
          from_joinset: Some(<Joinset.name>),
      }
```

The `left`-slot identification of the anchor is the plan-tree manifestation of §3.2's row-set-reference-frame rule. Even when the hop's underlying `Relationship.from / to` is reversed (i.e. `JoinHop.direction == Reverse`), the anchor remains the plan-tree left: `JoinsetStrategy` flips the `KeyPair` orientation as needed but preserves anchor-left structurally.

### 3.4 Anchor and fan-out

Fanout analysis per `16 §3.3` operates on **walked** cardinality. For a Joinset with hop `H` from the anchor to member `M`:

- If `Relationship.cardinality == ManyToOne` and `H.direction == Forward`: no fanout; anchor rows lookup-join single-row matches on `M`.
- If `Relationship.cardinality == OneToMany` and `H.direction == Forward`: **anchor rows fan out** — each anchor row matches zero-or-more `M` rows.
- If `Relationship.cardinality == ManyToOne` and `H.direction == Reverse`: fanout (effective cardinality mirrors to `OneToMany` per `16 §3.4`).
- If `Relationship.cardinality == OneToOne`: no fanout, direction-independent.
- If `Relationship.cardinality == ManyToMany`: fanout in both directions; `PLAN_W_2402 JoinsetManyToManyHopAdvisory` per §11.2.

Multi-hop fanout (N-ary Joinsets) accumulates per §5.4; v1's single-hop case is one of the cases above.

## Testing Strategy

- VALID_E_2400 JoinsetArityV1Violation; VALID_E_2401 JoinsetAnchorMissing; VALID_E_2402 JoinsetAnchorNotMember; struct-shape round-trip.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
