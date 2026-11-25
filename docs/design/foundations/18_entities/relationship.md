---
id: 18-relationship
status: ratified
version: 2.2
impl: implemented
---

# Relationship

This leaf owns the `Relationship` struct shape (unified for root and Joinset sites), `JoinType`, `Cardinality`, and `JoinKeyExprPair`. See [semantics-pools](semantics-pools.md) for root-level authoring patterns, and `16` for composition mechanics.

<!-- cross-cut: none -->

## 2. `Relationship`

### 2.1 Unified struct — root and Joinset sites

One `Relationship` struct serves both authoring sites: root-level `relationships:` on `SemanticModel`, and `relationships:` on a `JoinsetBody`. The struct is **semantic-first**: authors declare the relationship's shape (`cardinality`, `integrity`, `optional`, `cross_filter`) and the planner derives the operational `JoinType` per `§2.9`.

```rust
#[non_exhaustive]
pub struct Relationship {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in the model's `relationships` map
    /// (and a `JoinsetBody`'s `relationships` map), and the sole identity used
    /// for relationships across manifest and runtime (§2.1).
    pub id: Option<EntityId>,

    pub name: String,
    pub from: DataKindName,
    pub to:   DataKindName,

    pub keys: Vec<JoinKeyExprPair>,

    /// Optional residual predicate evaluated against the joined rowset.
    /// `None` means "equi-join only per `keys`".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<UnresolvedSemanticExpr>,

    pub cardinality: Cardinality,

    #[serde(default)]
    pub integrity: Integrity,

    /// Which side is preserved when matches are absent. Default
    /// derived from `(cardinality, integrity)` per §2.7. Required
    /// when `cardinality ∈ {OneToOne, ManyToMany}` per SR-E-13.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optional: Option<Optional>,

    /// Direction of filter propagation through the join. Default
    /// derived from `cardinality` per §2.7. Required when
    /// `cardinality ∈ {OneToOne, ManyToMany}` per SR-E-13.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_filter: Option<CrossFilter>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
```

**Identity is the `id` field.** A `Relationship` is identified everywhere — model, manifest, and runtime graph — by its own `EntityId`. The authoring field is `id: Option<EntityId>` (`None` = authored-absent, minted at parse per `32 §9.0.1`); past the parse boundary it is invariably `Some`, and the manifest/runtime carry the bare `EntityId`. There is no separate `RelationshipId` newtype. The manifest keys `relationships` by `EntityId` and every relationship reference is an `EntityId` (`33 §4.1`, `§6.7` — `JoinsetHop.relationship` is an `EntityId`); `16 §6`'s `RelationshipPath` is a `Vec<EntityId>`. The planner runtime graph (`34`) builds its own daggy edge handles (`SemanticEdgeId`, `35 §2A`) for hot-path adjacency, but those address the durable relationship by `EntityId` — they are graph-structure positions, not a parallel identity.

> **Ordering basis (id-first rework).** Relationships are stored model-side in an `EntityId`-keyed map projected in name order (`32 §2`, `§7`), not an author-ordered `Vec`. First-match-wins resolution (`16 §11`) and BFS neighbor iteration (`19 §3.4.3`) therefore proceed in **name order** rather than YAML author order — deterministic and author-meaningful.

> **Id-first cascade (STATUS item U.2, landed).** `RelationshipId` is eliminated tree-wide; relationships are identified by `EntityId` everywhere (`16` composition-id derivation + canonical `(EntityId, Direction)` forms, `19 §3.4.5` `PathSignature: Vec<EntityId>`, `33 §6.7` `JoinsetHop.relationship`, `34` planner). Material consequence (ratified): composition `ImplicitId` is now **stable across recompiles** when relationship `EntityId`s are stable (authored/persisted; synthesized ids always stable), strengthening `16 §5.7`.

**Authored vs derived.** `JoinType` is **not** an authoring-layer field. It is derived at compile from `optional` per `§2.9` and carried on the manifest-layer `ResolvedRelationship` (`33 §8A`) — the authoring `Relationship` here is **not** re-exported into the manifest verbatim; compile resolves it (endpoints to `EntityId`, defaults applied, keys lowered) into `ResolvedRelationship`.

**No parallel override surface.** Authors who need different join semantics inside a specific Joinset declare a **scope-local Relationship** in that Joinset's `relationships:` block with the divergent fields directly. The scope-shadow rule in `§2.10` resolves the visibility. There is no per-hop override map and no per-edge override mechanism.

### 2.2 YAML shape

```yaml
relationships:
  - name: orders_to_customers
    from: orders
    to:   customers
    keys:                                # list of equi-pairs
      - from: customer_id                # UnresolvedSemanticExpr on the `from` side
        to:   id                         # UnresolvedSemanticExpr on the `to` side
    # Optional residual predicate — evaluated against the joined rowset.
    filter: "from.order_ts <= to.customer_ts"
    cardinality: many_to_one              # REQUIRED
    integrity: assumed                    # default; alternatives: enforced | none
    optional: none                        # default per matrix; required on 1:1 / m:m
    cross_filter: left                    # default per matrix; required on 1:1 / m:m
    description: "Customer ownership of orders."
```

`left` / `right` in `optional` and `cross_filter` correspond to the `from` / `to` sides respectively (`from` ≡ left, `to` ≡ right). The convention is: **the value names the side on the receiving end of the action** — preservation for `optional`, filter reception for `cross_filter`.

### 2.3 `Cardinality` — required at every site

```rust
#[non_exhaustive]
pub enum Cardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}
```

`cardinality:` is required on every `Relationship`, at every authoring site. Authors MUST declare the cardinality they intend; the planner does not infer it. Missing cardinality is `parse.relationship-missing-cardinality` (SR-E-4).

`Cardinality` is **planning metadata, not runtime enforcement**. `semstrait` does not scan data to verify declared multiplicity. See `16 §3` for fanout consequences per variant.

### 2.4 `Optional` — preserved-side enum

```rust
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Optional {
    /// Preserve neither side; drop unmatched rows on both sides. Derives Inner.
    None,
    /// Preserve the `from` (left) side; null-pad `to` for unmatched. Derives Left.
    Left,
    /// Preserve the `to` (right) side; null-pad `from` for unmatched. Derives Right.
    Right,
    /// Preserve both sides; null-pad whichever side lacks a match. Derives Full.
    Both,
}
```

`optional:` declares which side's rows are preserved when the join condition lacks matches. The value **names the preserved side**. Default per `§2.7`. **Required** on `OneToOne` and `ManyToMany` (SR-E-13) — there is no defensible default for symmetric cardinalities.

### 2.5 `CrossFilter` — filter-flow enum

```rust
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum CrossFilter {
    /// No filter propagation between sides.
    None,
    /// The `from` (left) side receives filters from the `to` side.
    /// Filter flow: `to → from`.
    Left,
    /// The `to` (right) side receives filters from the `from` side.
    /// Filter flow: `from → to`.
    Right,
    /// Bidirectional propagation; both sides receive filters from the other.
    Both,
}
```

`cross_filter:` declares the direction of filter propagation through the join. The value **names the side that receives filters**. Default per `§2.7`. **Required** on `OneToOne` and `ManyToMany` (SR-E-13).

**`ManyToMany` constraint.** When `cardinality == ManyToMany`, `cross_filter ∈ {Left, Right}` is rejected per SR-E-14 (`validate.relationship-many-to-many-cross-filter-directional`). A many-to-many relationship has no natural "one" side, so directional filter propagation is ambiguous; authors MUST declare `Both` or `None`.

**Planner contract.** `cross_filter` is authored on this `Relationship` and surfaces to the planner via the manifest-layer `ResolvedRelationship` (`33 §8A`), where compile materialises its default (`§2.7`) so the manifest carries a concrete value, not an `Option`. The exact predicate-placement and pushdown rules driven by this field are owned by the planner doc; `18` ratifies the authoring shape and the validation rules only.

### 2.6 `Integrity` — author-asserted RI strength

```rust
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Integrity {
    /// Referential integrity is enforced upstream of `semstrait`
    /// (catalog FK, application invariant, ETL guarantee). The planner
    /// MAY trust the assertion for join-default derivation and any
    /// future RI-trust optimizations. Author-asserted only — semstrait
    /// performs no compile-time cross-check.
    Enforced,
    /// The author believes integrity holds but provides no guarantee.
    /// Same trust posture as `Enforced` at the planner layer in v1;
    /// reserved as a separate variant so future optimizations can
    /// differentiate (e.g. NULL-elision under `Enforced` only).
    Assumed,
    /// Orphans exist on at least one side. The planner assumes unmatched
    /// rows are possible and the default-derivation table biases toward
    /// preserving the from-side anchor.
    None,
}

impl Default for Integrity {
    fn default() -> Self { Integrity::Assumed }
}
```

`integrity:` is informational in v1. `semstrait` does not verify it at compile time against catalog metadata or declared `Key::Foreign` entries (`§9.1`).

### 2.7 Defaults & requirements matrix

The defaults below apply when `optional:` and/or `cross_filter:` are omitted. Required cells must be authored explicitly.

| `cardinality` | `integrity` | `optional` default | `cross_filter` default |
|---|---|---|---|
| `ManyToOne` | `Enforced` | `None` | `Left` |
| `ManyToOne` | `Assumed` | `None` | `Left` |
| `ManyToOne` | `None`     | `Left` | `Left` |
| `OneToMany` | `Enforced` | `None` | `Right` |
| `OneToMany` | `Assumed` | `Left` | `Right` |
| `OneToMany` | `None`     | `Left` | `Right` |
| `OneToOne`  | *any*     | **required** | **required** |
| `ManyToMany`| *any*     | **required** | **required** (Left/Right rejected) |

Reading conventions:
- `from` ≡ left, `to` ≡ right.
- For `ManyToOne` (typical fact → dim): `optional: None` ⇒ Inner (RI-trusted enrichment); `optional: Left` ⇒ Left from-anchored (preserve facts when dim missing).
- For `OneToMany + Enforced`: defaults to `Inner` because RI guarantees every to-side row has its from-side parent. Authors wanting all from-side rows (including those with no children) declare `optional: Left` explicitly.

Validation rules:
- `optional` and `cross_filter` required when `cardinality ∈ {OneToOne, ManyToMany}` — `validate.relationship-symmetric-cardinality-incomplete` (SR-E-13).
- `cross_filter ∈ {Left, Right}` rejected when `cardinality == ManyToMany` — `validate.relationship-many-to-many-cross-filter-directional` (SR-E-14).
- `integrity: Enforced` is **not** cross-checked at compile in v1.

### 2.8 `JoinKeyExprPair` — hybrid equi-key grammar

```rust
#[non_exhaustive]
pub struct JoinKeyExprPair {
    /// UnresolvedSemanticExpr on the `from` side. In the simplest case, a bare Semantic name.
    pub from: UnresolvedSemanticExpr,
    /// UnresolvedSemanticExpr on the `to` side. Symmetric.
    pub to: UnresolvedSemanticExpr,
}
```

Authors list one `JoinKeyExprPair` per equi-predicate. The planner emits `left.<from_expr> = right.<to_expr>` per pair and ANDs the residual `filter:` predicate (if any) on top.

`keys` carries equi-join predicates. `filter` carries residual non-equi predicates (for example `from.valid_from <= to.event_ts`).

### 2.9 `JoinType` — derived at compile, manifest-only

`JoinType` is **not authored**. It is derived at compile from the relationship's `optional` field and carried on `ResolvedRelationship` (`33 §8A`) for downstream consumption by `JoinsetStrategy` and `PlanNode::Join` emission.

```rust
#[non_exhaustive]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}
```

Derivation table:

| `optional` | Effective `JoinType` |
|---|---|
| `Optional::None`  | `Inner` |
| `Optional::Left`  | `Left` |
| `Optional::Right` | `Right` |
| `Optional::Both`  | `Full` |

Temporal / as-of joins (`AsOf`) remain out of scope for v1; see `17 §5` for the temporal-shape activation matrix.

### 2.10 Authoring sites and scope

| Site | Semantics |
|---|---|
| Root-level `semantic_model.relationships:` | Visible to every DataKind in the model; feeds `16 §11`'s implicit composition graph; the planner synthesizes Joinsets per Request when the relationship graph permits. |
| `JoinsetBody.relationships:` | Scope-local to that Joinset. The Relationship struct shape, validation rules, defaults matrix, and derivation table are identical to root-level. A Joinset-local `Relationship` MAY redeclare a root-level name, in which case the Joinset-local entry takes precedence **within the Joinset's scope only**. Scope shadow is the sole mechanism for divergent join semantics inside a Joinset — there is no separate override surface. |

Root-level relationships with no corresponding DataKind (`from:` / `to:` resolves to no known data kind) are `validate.relationship-dangling-endpoint` (SR-E-5).

## Testing Strategy

- SR-E-4 (missing cardinality), SR-E-5 (dangling endpoint), SR-E-13 (symmetric cardinality incomplete), SR-E-14 (many-to-many directional cross_filter rejection); defaults-matrix derivation; JoinType derivation table; scope-shadow resolution; self-contained variant tables (§2.7 + §2.9) are LOCAL (waiver).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
