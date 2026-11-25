---
id: 16-relationship-edge-model
status: ratified
version: 2.2
impl: implemented
---

# Relationship / Cardinality / JoinType

This leaf ratifies the `Relationship` edge-type that binds top-level `DataKind`s, the `Cardinality` enum that declares join multiplicity, the derived `JoinType`, and the bidirectional traversal semantics.

> **Struct shape**: the `Relationship` struct itself — including its fields (`name`, `from`, `to`, `keys`, `filter`, `cardinality`, `integrity`, `optional`, `cross_filter`, `ai_context`, `description`), its `id: EntityId` identity field, the `JoinKeyExprPair` hybrid equi-key grammar, and the `Cardinality` / `Integrity` / `Optional` / `CrossFilter` / (derived) `JoinType` enums — is defined in [`18_entities.md §2`](../18_entities.md#2-relationship). This doc ratifies the *composition semantics* on top (placement, scope, traversal, fanout analysis). Where the body prose below uses `KeyPair`, read `JoinKeyExprPair` per `18 §2.8`.

## 2. The `Relationship`

A `Relationship` is a **pairwise, named connector** between two top-level `DataKind`s declaring a joinable edge: a pair (or tuple) of `SemanticsName`s on each side, the relationship's shape (`cardinality`, `integrity`, `optional`, `cross_filter`), and an optional residual `filter`. The operational `JoinType` is **derived** from `optional` at compile per `18 §2.9`, not authored. `Relationship` is the semstrait type-system analogue of a foreign-key association, lifted to the semantic layer so keys are `Semantics`, not SQL columns.

### 2.1 Placement and scope

Root-level `relationships:` in the `SemanticModel` is the primary declaration site and the **only** one that feeds `§11`'s implicit composition graph: the `Root` scope owns that `Relationship` list. A `Joinset` MAY additionally declare scope-local `Relationship`s in its own `relationships:` block, bounded to that Joinset's scope. The authoring sites and their scope rules are owned by [`18 §2.10`](../18_entities.md#2-relationship); the composition semantics of the Joinset-scoped shadow are ratified in [joinset-model](joinset-model.md) §13.3. Root-level placement matters for three reasons:

1. **Symmetric participation.** A `Relationship` between `A` and `B` is a property of the pair, not a field on either side. Embedding `Relationship`s inside `A` would (falsely) imply `A` owns the edge.
2. **Scope visibility.** Per `11 §2.1`, root-level `Relationship`s are visible at `Root` scope; `11 §9`'s cross-kind reference rule grants any `Kind` scope the ability to dereference a `SemanticsName` owned by another kind **iff** a `Relationship` path exists that is visible in the resolving scope — the root-level list, plus any Joinset-local shadows when resolving inside that Joinset.
3. **Authoring ergonomics.** Star-schema-style Models declare one root-level `relationships:` block with N entries; snowflake patterns extend this linearly. Authors do not hunt through `DataKind` bodies to find the model-wide edges.

A `Relationship` between a top-level `DataKind` and a `Nested-kind` (child of a `Unionset` / `Grainset` / `Joinset`) is rejected at `validate`: nested kinds do not have `Root` scope identity. Authors who need the equivalent lift the nested kind to top-level first.

**Permitted constituents.** Both `from` and `to` must resolve to a top-level `DataKind` — `Simple`, `Unionset`, `Grainset`, or `Joinset`. A `Relationship` between two composed kinds is permitted (see [`questions/closed/16_questions.md`](../../questions/closed/16_questions.md) Q-COMP-013 — closed); its `KeyPair.left` or `.right` may reference a namespaced name within the composed surface (e.g. `"order_details.customer_id"`).

### 2.2 Structure

Fields:

- `id: EntityId` — the relationship's durable identity (`18 §2.1`): the authored model `id` (UUIDv7) or a parse-generated one. Globally unique and **stable across recompiles** (unlike the retired compile-counter handles). It is the sole relationship identity used across model, manifest, and runtime.
- `from: DataKindRef`, `to: DataKindRef` — named references to top-level `DataKind`s. `DataKindRef` is defined in `11 §4` as a newtype over `DataKindName`.
- `keys: Vec<KeyPair>` — non-empty; each entry declares one join-column pair. Composite keys (N > 1) are positional: `keys[0]` on the left side matches `keys[0]` on the right side, etc. (§2.3; open Q-COMP-009).
- `cardinality: Cardinality` — one of four variants (§3).
- `integrity: Integrity` — author-asserted referential-integrity strength (`Enforced` / `Assumed` / `None`; default `Assumed`). See `18 §2.6`.
- `optional: Option<Optional>` — preserved-side enum (`Left` / `Right` / `Both` / `None`). Default derived from `(cardinality, integrity)` per `18 §2.7`; required on `OneToOne` / `ManyToMany` (SR-E-13). See `18 §2.4`.
- `cross_filter: Option<CrossFilter>` — filter-flow enum (`Left` / `Right` / `Both` / `None`). Default derived per `18 §2.7`; required on `OneToOne` / `ManyToMany`; directional values forbidden on `ManyToMany` (SR-E-14). See `18 §2.5`.
- `filter: Option<UnresolvedSemanticExpr>` — optional residual predicate AND-ed on top of the equi-join keys; carried as `18 §2.8`'s `model::UnresolvedSemanticExpr` authoring form (a bare `Expr<UnresolvedSemanticLeaf>` field with field-level friendly serde — the `ExprSource` wrapper is removed per doc 45; lives in `semstrait-model`).
- **Derived** `join_type: JoinType` — produced at compile from `optional` per the `18 §2.9` table; carried on the manifest-layer `ResolvedRelationship` (`33 §8A`), into which compile resolves the authoring `Relationship` (endpoints to `EntityId`, defaults applied, keys lowered) — it is **not** a verbatim re-export. Not an authoring field.

**Conventional orientation.** `from` is the "owning" or "driving" side and `to` is the "referenced" side, mirroring foreign-key narrative. For a `ManyToOne` relationship (e.g. `orders → customers`), `from = orders` and `to = customers`. `from` ≡ left, `to` ≡ right for the purposes of `optional` / `cross_filter` enum values. Semantics do not depend on the orientation — the planner walks the edge in either direction by default — but the convention aids readability and drives `Cardinality`'s per-variant naming (`ManyToOne` reads naturally as `from → to`).

**Stable identity.** `EntityId` is the primary key every downstream layer uses: `19 §3.4.5`'s `PathSignature` carries `Vec<EntityId>`, `PlanNode::Join` (per `35`) carries a `EntityId` as metadata, and diagnostics reference `relationship_id` for precise blame. Names (`from.to` style) are not used as identity — two `Relationship`s between the same `DataKindRef` pair differ by `id` even if they declare the same `from` / `to` / `keys` (but will be rejected by well-formedness checks).

### 2.3 `KeyPair`

```rust
#[non_exhaustive]
pub struct KeyPair {
    pub left: SemanticsName,  // on Relationship.from — a Key or Dimension
    pub right: SemanticsName, // on Relationship.to — a Key or Dimension
}
```

**Name, not column (I1).** Both sides are `SemanticsName`s — the canonical layer's identity for a semantic element. Physical resolution (what column does `SemanticsName::new("customer_id")` resolve to on the `orders` Binding?) is `15 §5`'s `ColumnMapping` job. `16` never looks at columns.

**Shape.** Both sides must resolve at `compile` to either:

- A `Key` declared in the referenced `DataKind`'s interface (`Key::Primary`, `Key::Foreign`, or `Key::Unique` per `11 §8.3`), or
- A `Dimension` declared in the interface.

`Measure`, `Metric`, and `Filter` names are rejected — those are aggregates or guards, not join columns. `CompileError::RelationshipKeyNotJoinable` (see [composition-validation-and-errors](composition-validation-and-errors.md)) fires if a `KeyPair` references one.

**Composite keys — positional pairing.** A `Relationship` with a composite join condition declares multiple `KeyPair` entries:

```rust
Relationship {
    from: "order_lines", to: "product_variants",
    keys: vec![
        KeyPair { left: "product_id",    right: "product_id" },
        KeyPair { left: "variant_code",  right: "variant"    },
    ],
    // ...
}
```

Ordering is significant only for `PathSignature` canonicalization (`19 §3.4.5`); the join condition itself is commutative (`A AND B == B AND A`). `32`'s YAML surface emits `keys:` as a list of `{left, right}` pairs. Positional vs single-entry `Vec<SemanticsName>` is tracked as Q-COMP-009; Round 1 ratifies positional pairs.

**Type agreement.** Both sides' inferred `DataType` (per `19 §3.6`'s type inference over the `Binding`'s `SemanticMapping` and any `Declarative` expression) must be compatible under `13 §4`'s type-compatibility relation. Numeric sides widen per `13 §5`'s widening rules; string sides require exact match; temporal sides require equal precision and nullability compatibility. Mismatch triggers `CompileError::RelationshipKeyTypeMismatch` (see [composition-validation-and-errors](composition-validation-and-errors.md)). The check runs at `compile`, not `validate`, because inferring the physical type requires the `19 §3` resolution pass.

**Nullability.** `semstrait` does not propagate NOT NULL as a relationship precondition. If a `KeyPair.left` resolves to a nullable semantics, `NULL = NULL` returns `UNKNOWN` per standard SQL and the row is dropped on `Inner` / non-matched rows are `NULL`-padded on `Left` / `Right`. Authors who need a NOT-NULL contract add a `Constraint::NotNull` in the referenced `DataKind`'s interface (per `11 §8.4`); violations surface at plan time when the planner emits the join.

### 2.4 Traversal — always bidirectional

`Relationship`s are always walkable in both directions for the purposes of implicit composition and path-finding. The earlier `Directionality` enum (`Bidirectional` / `Forward`) is retired; v1 has no authoring-layer mechanism to forbid reverse traversal. Authors who need to prevent the planner from auto-synthesizing reverse walks (e.g. forbidding `users → page_views` aggregation fanout when only `page_views → users` enrichment is intended) lift that intent into an **explicit Joinset declaration** for the desired direction.

#### 2.4.1 Symmetric traversal mechanics

Forward and reverse walks share the same `EntityId` — they are the same edge, walked in two directions. The compile-time enumeration normalizes direction at walk time: given a `current_node` and an unvisited neighbour `target_node`, the step is flagged `reverse: true` when `current_node == Relationship.to && target_node == Relationship.from`, and `reverse: false` otherwise. The `PathSignature` (`19 §3.4.5`) records the `EntityId` alone; the direction is reconstructed at plan time by matching `current_node` against the stored `from` / `to`.

**`Cardinality` under reversal.** A `Relationship { cardinality: ManyToOne, from: A, to: B }` walked in reverse (`B → A`) is read as `OneToMany` for fanout-analysis purposes. The stored enum variant does not change; the planner inverts the interpretation on reverse walks. Analogous for `OneToMany` ↔ `ManyToOne`. `OneToOne` and `ManyToMany` are inversion-symmetric — no mental flip needed.

**Derived `JoinType` under reversal.** With `JoinType` derived from `Relationship.optional` per `18 §2.9`, reverse walks substitute the mirror form at plan emission: derived `Left` walked in reverse behaves as `Right` (the reverse side is preserved with NULL fill on the now-from side); derived `Right` reversed becomes `Left`. `Inner` and `Full` are inversion-symmetric. The canonical `optional` value on `Relationship` does not change; the substitution is per-walk at emission.

## 3. `Cardinality`

Enum defined in [`18 §2.4`](../18_entities.md). v1 variants: `OneToOne`, `OneToMany`, `ManyToOne`, `ManyToMany`.

Declared on every `Relationship`. `Cardinality` is **planning metadata**, not a runtime enforcement — `semstrait` does not scan data to verify that the declared multiplicity holds. Authors assert it; the planner trusts it and shapes the plan accordingly. A Model that declares a `OneToOne` where the physical data exhibits `ManyToMany` will produce arithmetically incorrect aggregations (per-row duplication) without a runtime error. This is an explicit trade-off: verification would require a scan, which violates the compile-time-resolution posture (I5).

### 3.1 Enum — `#[non_exhaustive]`

Four variants cover the ratified case matrix. `#[non_exhaustive]` (I10) reserves room for future variants (`ZeroOrOne` for SCDs, `OneOrMore` for strict-existence joins) without semver breakage.

### 3.2 `Cardinality` vs key-level uniqueness

`Cardinality` declares the multiplicity of the **join**, not the uniqueness of any particular side's keys. A `ManyToOne` `Relationship` does not imply `from.key` is non-unique and `to.key` is unique — those uniqueness claims are `11 §8.3`'s `Key::Primary` / `Key::Unique` territory. Inconsistencies between the two (`ManyToOne` from a side with `Key::Primary` on the left key) emit `PLAN_W_0503` `RelationshipCardinalityKeyMismatch` advisory (see [composition-validation-and-errors](composition-validation-and-errors.md)).

### 3.3 Per-variant semantics

<!-- cross-cut: none -->

#### 3.3.1 `OneToOne`

Each row in `from` matches at most one row in `to`, and vice versa.

- **Fanout:** none. Measure aggregations over the composed surface behave as-if on a single relation.
- **Join-type compatibility:** any of `Inner` / `Left` / `Right` / `Full`. `Inner` drops rows with no match on either side; `Left` / `Right` preserve one side's rows and NULL-pad; `Full` preserves both.
- **Planning implication:** the planner may schedule aggregation either before or after the join without correctness risk. Cost-based choice drives ordering.

#### 3.3.2 `OneToMany`

One row in `from` matches zero-or-more rows in `to`.

- **Fanout:** potential. A `SUM(from.measure)` computed after joining would sum a `from` row once per matched `to` row, overstating the aggregate by the `to`-side fanout factor.
- **Fanout-safe rewrite.** When the planner composes a surface that requires a measure on the `from` side **and** a dimension from `to`-side (or a transitive `to`-side constituent), it rewrites the plan to aggregate `from` to its grain before the join. This is a `17 §*` × `Additivity` interaction: measures declared `Additive` under the join's grain are safely rewritten; `SemiAdditive` requires the grain axis match `11 §7`'s declared axes; `NonAdditive` measures emit the straightforward join (potentially yielding duplicated contributions the author is responsible for handling in their Request — fanout is the natural consequence of declaring a `OneToMany` relationship and asking aggregation across it).
- **Join-type compatibility:** `Inner` with `OneToMany` is fine. `Left` from the `One` side with `OneToMany` produces one row per `to`-side match; zero-match `from` rows are preserved with NULL `to`-side fields. `Right` from the `One` side is equivalent to `Left` from the `Many` side; `Full` preserves unmatched rows on both.
- **Walked in reverse:** behaves as `ManyToOne` (see §3.3.3); traversal is always bidirectional per §2.4.

#### 3.3.3 `ManyToOne`

Many rows in `from` match exactly one row in `to` (typical fact → dim).

- **Fanout:** none on measures declared on `from`. Measures declared on `to` are subject to re-distribution: a `SUM(to.population)` grouped by a `from` attribute sums `to.population` once per `from` row matching that attribute. The author owns this trade-off when declaring the relationship.
- **Join-type compatibility:** any. `Inner` drops unmatched `from` rows; `Left` preserves unmatched `from` rows with NULL `to` fields; `Right` / `Full` are less common but permitted.
- **Planning implication:** the planner may schedule aggregation after the join without correctness risk for measures on `from`; measures on `to` require pre-join aggregation on `to` if the strategy supports it, otherwise the straightforward join.

#### 3.3.4 `ManyToMany`

Multiple rows on each side match multiple rows on the other.

- **Fanout:** bilateral. Aggregations are at risk on both sides.
- **Canonical modeling.** `ManyToMany` without an intermediate junction `DataKind` is usually an anti-pattern in analytics. Authors are nudged toward declaring two `ManyToOne` `Relationship`s through a junction `DataKind`. v1 permits direct `ManyToMany` declaration (Q-COMP-016); the structural-mismatch advisory `PLAN_W_0503` (see [composition-validation-and-errors](composition-validation-and-errors.md)) still fires when the cardinality contradicts declared uniqueness on the key sides.
- **Join-type compatibility:** any. Deduplication — when the planner needs it to preserve cardinality of a primary-key side — is `DISTINCT`-based (plan-layer decision; `20`).

## 4. `JoinType` — derived at compile

Enum defined in [`18 §2.9`](../18_entities.md#29-jointype--derived-at-compile-manifest-only). v1 variants: `Inner`, `Left`, `Right`, `Full`. `Semi` / `Anti` are deferred (see §4.3 below). `AsOf` is deferred and gated on `17 TemporalShape` (see §4.3 and `17 §5`).

The join-kind is **not authored**. It is derived at compile from `Relationship.optional` per the `18 §2.9` derivation table, recorded on the manifest-layer `ResolvedRelationship` (`33 §8A`) — into which the authoring `Relationship` (`18 §2`) is resolved at compile, not re-exported verbatim — and lowers from there to `PlanNode::Join`'s `join_type` field in `35`.

### 4.1 Enum — ratified variants

Four variants in v1: `Inner`, `Left`, `Right`, `Full`. `#[non_exhaustive]` per I10. All four translate 1:1 to standard SQL `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`.

### 4.2 Per-variant semantics — and the `optional:` source

<!-- cross-cut: none -->

The derivation `optional → JoinType` is one-to-one per `18 §2.9`:

| Authored `optional` | Derived `JoinType` |
|---|---|
| `Optional::None`  | `Inner` |
| `Optional::Left`  | `Left` |
| `Optional::Right` | `Right` |
| `Optional::Both`  | `Full` |

The mechanical semantics of each derived variant:

- **`Inner`** — produces rows with a match on **both** sides of the `KeyPair`. Non-matching rows on either side are dropped. Combined with `Cardinality::OneToOne` and non-NULL join columns, `Inner` is the canonical "joined table" semantics. Author intent: "no preservation" (`optional: none`).
- **`Left`** — preserves all rows from `Relationship.from`; NULL-pads the `to` side for unmatched `from` rows. Combined with `ManyToOne`, this is the canonical "enrich facts with dim, keep unmatched facts" pattern. Author intent: "preserve left/from" (`optional: left`).
- **`Right`** — preserves all rows from `Relationship.to`; NULL-pads the `from` side for unmatched `to` rows. Less common in analytic workloads but symmetric. Author intent: "preserve right/to" (`optional: right`).
- **`Full`** — preserves rows from both sides; NULL-pads whichever side lacks a match. Useful for union-like semantics that the shape of a `Unionset` would not express cleanly. Author intent: "preserve both" (`optional: both`).

**Under traversal reversal.** Derived `Left` walked in reverse behaves as `Right` at plan emission (see §2.4.1). The canonical `Relationship.optional` value does not change; the emission substitutes the mirror form.

### 4.3 Deferred variants

**`Semi` / `Anti`.** Deferred from v1. Both are correctness-preserving optimizations of `Inner` (with and without matching rows respectively) that the adapter layer may emit as an engine-specific rewrite of an `Inner` + dedup-and-project pattern. Authors explicitly wanting `Semi` / `Anti` semantics in v1 write a `Filter` constraint referencing an `EXISTS`-style semantics; a future v2 revision may introduce `JoinType::Semi` / `JoinType::Anti` as canonical variants. Tracked as `[TD-COMPOSITION-SEMI-ANTI]`.

**`AsOf`.** Deferred; gated on `17 TemporalShape`'s `SCD` and `Snapshot` variants. An `AsOf` join's semantics require matching the closest-preceding or closest-following timestamp in the `to` side to a reference timestamp on the `from` side — which presupposes a ratified temporal axis on both constituents. `17` is the gating document; `16` does not reserve `JoinType::AsOf` in v1. Tracked as `[TD-COMPOSITION-ASOF]`.

### 4.4 `PlanNode::Join` carriage

Per `35`'s `PlanNode::Join` variant:

```rust
pub enum PlanNode {
    // ...
    Join {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_type: JoinType,           // ratified in 16 §4
        predicate: PhysicalExpr,       // resolved from Relationship.keys via 15
        cardinality: Cardinality,      // ratified in 16 §3
        relationship_id: Option<EntityId>, // set for traversal-derived joins
        // ...
    },
    // ...
}
```

The `relationship_id` field is `Some(EntityId)` when the `Join` node was emitted from a `Relationship` traversal (implicit composition or `Joinset`) and `None` when emitted from a direct predicate (e.g. a `Unionset`'s internal `UNION ALL` has no `Join` node, so this question does not arise; a hand-authored `filter:` with an `EXISTS` construct would, when reducible to a `Semi` rewrite by the adapter, lose `relationship_id`). Adapters consume the trio (`join_type`, `cardinality`, `relationship_id`) to emit engine-correct SQL / IR.

## Testing Strategy

- **Invariants:** I1 (name-not-column), I5 (compile-time resolution), I10 (non-exhaustive enums).
- **Structural validations:** `Relationship` duplicate detection (`ValidateError::DuplicateRelationship`), key type agreement (`CompileError::RelationshipKeyTypeMismatch`), self-reference rejection (`CompileError::RelationshipSelfReference`), bidirectional traversal symmetry.
- **Error codes:** `COMP_E_0401`, `COMP_E_0402`, `COMP_E_0403`, `PLAN_W_0503` (see [composition-validation-and-errors](composition-validation-and-errors.md)).
- **Behavioral gates:** Tests that `Cardinality` × `JoinType` derivation matches `18 §2.9`, that reverse-walk inversion flips `Cardinality` and `JoinType` correctly, that `KeyPair` type agreement enforces `13 §4` compatibility.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
