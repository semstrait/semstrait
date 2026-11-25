---
id: 24-joinset-strategy
status: ratified
version: 2.1
impl: drift
---

# Joinset — JoinsetStrategy

## 5. `JoinsetStrategy`

### 5.1 Path resolution

`JoinsetStrategy` is the planner-side contract that lowers a `ResolvedJoinset` to `PlanNode::Join` nodes. Path resolution runs at `compile` (per §4); `JoinsetStrategy` consumes the already-resolved `hops: Vec<ResolvedJoinHop>` without re-walking the Relationship graph.

Expression placement within each member's subplan and at the post-join reconciliation follows the member expression's `ExprLayer` (`19 §6.0`): `Scalar` Dimensions/Keys feed pre-join projection and `GROUP BY`; `Aggregate` Measures land in the member's `Agg`; `PostAggregate` Metric residuals land in the post-join `Project`. Non-additive measures under join fanout are gated by function-derived `Additivity` (`14a §3.6.2`) — see §11.1's `JoinsetNonAdditiveRollupRequired`.

```rust
#[non_exhaustive]
pub struct JoinsetStrategy<'m> {
    pub joinset: &'m ResolvedJoinset,
    pub manifest: &'m SemanticManifest,
}

impl<'m> JoinsetStrategy<'m> {
    /// Lower the Joinset into a `PlanNode` tree. Caller passes the
    /// Request's projection list; strategy returns the post-join
    /// Project node. `34` ratifies the exact signature.
    pub fn lower(self, projection: &Projection, advisories: &mut DiagnosticSink)
        -> Result<PlanNode, PlannerError>;
}
```

`34` ratifies the exact signature and the `DiagnosticSink` shape. `24` fixes the contract: resolution has already happened; `lower` emits nodes and advisories.

### 5.2 `PlanNode::Join` emission

For each `ResolvedJoinHop` in anchor-outward order:

1. **Left operand.** Hop 0's `left` is the anchor's `PlanNode::Scan`. Hop `i > 0`'s `left` is the join-tree built up through hop `i - 1`. (For v1 binary, only hop 0 exists.)
2. **Right operand.** Hop `i`'s `right` is the `PlanNode::Scan` of `hops[i].to`.
3. **Keys.** From `Relationship.keys` (`16 §2.3`), resolved to physical columns via `15`-resolved `Binding.column_mapping`. Direction-aware swap: if `hop.direction == Reverse`, `KeyPair.left` and `KeyPair.right` are swapped so the plan-tree's left-side key matches the anchor-side of the hop.
4. `**JoinType`.** Per §5.3's derivation from the resolved Relationship's `optional` field.
5. `**from_relationship`.** `Some(hop.relationship)`, carrying the originating Relationship for debug / audit.
6. `**from_joinset`.** `Some(self.joinset.name)` — a new field on `JoinNode` per `35`'s parallel draft, carrying the resolved Joinset's `DataKindName` (which is the synthetic `__implicit_joinset_<8-hex>` for `Origin::Implicit { id }` Joinsets per `16 §5.7`, or the author-declared name for `Origin::Explicit` Joinsets). The two are uniform downstream — strategy emission does not distinguish on `Origin`. (`24` records the requirement; `35` ratifies the struct.)

ASCII plan tree for a binary Joinset `OrdersWithCustomers` anchor-left:

```text
Project <unified surface>
  └── Join {
         join_type: <effective, §5.3>,
         from_relationship: Some(Rel(orders_to_customers)),
         from_joinset: Some(DataKindName("orders_with_customers")),
         keys: [KeyPair { left: "customer_id" (on orders), right: "customer_id" (on customers) }],
         left: Scan(orders) {
             <push-down projection / filters per 19 §3 and 34>
         },
         right: Scan(customers) {
             <same>
         }
     }
```

### 5.3 `JoinType` selection — derived from the traversed Relationship

#### 5.3.1 Default selection

Hop `i`'s `JoinType` is **derived** from the underlying `Relationship` at `hops[i].relationship`: the planner reads `Relationship.optional` and applies the `18 §2.9` derivation table (`None → Inner`, `Left → Left`, `Right → Right`, `Both → Full`). Direction-agnostic at the canonical layer: `Reverse` walks read the same canonical `optional` value; the planner substitutes the mirror form (`Left ↔ Right`) at plan emission (per `16 §2.4.1`).

Pseudocode:

```text
EFFECTIVE_JOIN_TYPE(hop, manifest) -> JoinType:
  rel ← manifest.relationship_for_hop(hop)        // applies scope-local shadow per 18 §2.10
  derived ← derive_join_type(rel.optional)        // 18 §2.9 table
  if hop.direction == Reverse:
    return mirror(derived)                        // Left ↔ Right; Inner/Full unchanged
  else:
    return derived
```

#### 5.3.2 Varying join semantics inside a Joinset

There is **no per-hop override surface** in v1. A Joinset that needs different join semantics for a given hop declares a **scope-local `Relationship`** in its own `relationships:` block that shadows the root-level one (`18 §2.10`, `16 §13.3`). The scope-local Relationship is a full Relationship: it must declare `cardinality:`, and `optional:` / `cross_filter:` per SR-E-13 / SR-E-14. The derivation table then runs against the scope-local `optional` value during `EFFECTIVE_JOIN_TYPE` resolution.

Example (semi-formal):

```yaml
# Root-level:
relationships:
  - name: orders_to_customers
    optional: none                    # derives Inner
    # ...

# Joinset wanting NULL-padded enrichment for facts:
joinsets:
  - name: orders_with_optional_customer
    relationships:
      - name: orders_to_customers     # scope-local shadow
        optional: left                # derives Left for this Joinset only
        # ...
```

The root-level Relationship's derived `JoinType` is unchanged for all other consumers (implicit composition, other Joinsets). Only this Joinset sees the shadowed entry.

> **Retired (2026-05-12).** Earlier drafts of this section carried §5.3.2 *Override application*, §5.3.3 *Override-legality matrix*, and §5.3.4 *Override interaction with fanout*, with a permitted-override matrix gated on the underlying Relationship's declared `JoinType`. The override surface is removed in favour of the scope-local-shadow rule above. The associated `COMP_E_2411 JoinsetIllegalJoinTypeOverride` and `PLAN_W_2401 JoinsetJoinTypeOverrideAdvisory` are retired (see `[../questions/closed/24_questions.md](../../questions/closed/24_questions.md)`); their numeric codes are reserved for forward-compat. Fanout interactions (e.g. anchor preservation under `OneToMany`) carry through verbatim — the scope-local Relationship simply declares the divergent `optional`, and `§5.4`'s fanout accounting reads the same derived `JoinType` it always did.


### 5.4 `Cardinality` propagation

For a single-hop (v1 binary) Joinset, the composed surface's effective cardinality is whatever the single hop produces:


| Walked cardinality            | Composed surface cardinality             | Fanout              |
| ----------------------------- | ---------------------------------------- | ------------------- |
| `OneToOne`                    | anchor-cardinality preserved             | none                |
| `OneToMany` (anchor → target) | anchor rows fan out by 0..N              | yes                 |
| `ManyToOne` (anchor → target) | anchor-cardinality preserved             | none                |
| `ManyToMany`                  | anchor rows fan out; target rows fan out | yes (bidirectional) |


For N-ary Joinsets (`TD-NESTING-NARY-JOIN`), per-hop walked cardinalities accumulate:

```text
CUMULATIVE_FANOUT(hops, anchor) -> FanoutProfile:
  profile ← FanoutProfile::new_unit()
  running_kind ← anchor
  for hop in hops:
    walked ← effective_cardinality(hop.relationship, hop.direction)
    profile ← profile.compose(walked)
    running_kind ← hop.to
  return profile
```

`profile.compose(walked)` is the fanout-accumulation function (defined in `25` / `17` — parallel drafts). Binary v1 needs only the single-hop reading; the composition rule is fixed forward-compatibly.

#### 5.4.1 Multi-fanout advisory (N-ary forward-looking; v1-degenerate)

If more than one hop in the Joinset introduces fanout (accumulated `profile` has `> 1` fan-out edges), `JoinsetStrategy` emits `PLAN_W_2403 JoinsetMultiFanoutAdvisory`. In v1 binary this can only fire when a single `ManyToMany` hop is present (counted as two fan-out edges under `profile.compose`'s accounting). N-ary lift extends the advisory to cross-hop multi-fanout without a surface change.

### 5.5 Post-join projection

After the last `PlanNode::Join`, `JoinsetStrategy` emits a `PlanNode::Project` that:

1. Renames per-member columns to their `UnifiedName` (`16 §6.2`): bare name for single-contributor fields, namespaced (`orders.total`) for shape-incompatible collisions.
2. Applies `FieldProvenance` (`16 §7`) per column: `Native` fields select from the single contributor's projected columns; `Shared` fields (uncommon in a Joinset but possible on a join-key column, per `16 §6.3`'s `customer_id` example) are emitted once; `Derived` Joinset-level fields are computed post-join.
3. Applies `CompositionCoverage` (`16 §8`): fields with coverage variant `NullFill` on any member emit `NULL AS {name}` on that member's projection branch (this is primarily a Unionset concern; for Joinset the NULL-fill is carried by the `JoinType`'s outer-join NULL behavior rather than by `FieldOwnership::NullFill` per `16 §7.3.3`).
4. Applies any Joinset-level Filter declarations (`§2.6`) as a `PlanNode::Filter` atop the Project.

The final plan-tree shape for a binary Joinset with anchor `A`, member `B`:

```text
[Filter(joinset.filters)?]
  └── Project(unified surface)
        └── Join { anchor-left, per-§5.1–§5.3 }
              ├── Scan(A) with per-member pushdown
              └── Scan(B) with per-member pushdown
```

Exact pushdown semantics (predicate pushdown, projection pruning) are `19 §3` / `25` / `34`'s concern.

## 6. Interaction with `Relationship`

`Joinset` **consumes** top-level `Relationship`s declared in the Model's `relationships:` block (per `16 §2.1`, YAML surface per `32`) and MAY declare **scope-local `Relationship`s** in its own `relationships:` block that shadow root-level entries within the Joinset's scope only (per `18 §2.10`). `Joinset` does NOT:

- Modify root-level Relationships. A root-level Relationship's `cardinality`, `integrity`, `optional`, `cross_filter`, `keys` remain authoritative for implicit composition (`16 §11`) and for every other Joinset that does not shadow it. A scope-local shadow affects only the Joinset that declares it; the root-level `ResolvedRelationship` entry on the SemanticManifest (`33 §8A`) is unchanged, while the shadow is persisted inline on `DataKindVariant::Joinset.scope_local_relationships` (`33 §6.7`).
- Participate in root-level Relationship-graph validation. `16 §12`'s well-formedness (duplicate detection, key-type agreement, self-reference, etc.) runs on the root-level `relationships:` block independently of any `Joinset` that may consume the resulting edges. Scope-local Relationships run the same structural rules (`18 §11`, SR-E-4 / SR-E-13 / SR-E-14) within the Joinset's scope.

The reverse direction holds: `Relationship`s are unaware of which `Joinset`s consume them. A Relationship with no consumers (no implicit-composition walks, no `Joinset.path` references) is a perfectly valid SemanticManifest entry; authors routinely declare Relationships defensively for future use.

### 6.1 `Joinset`-side references to Relationships


| Reference                                           | Purpose                            | Validation stage                                |
| --------------------------------------------------- | ---------------------------------- | ----------------------------------------------- |
| `ExplicitPath.hops[i].relationship: EntityId` | Author-pinned path citation.       | compile (`COMP_E_2404`–`COMP_E_2410`).          |
| Implicit-path resolution (`§4.1.2 step 4`)          | BFS over the Relationship graph.   | compile (`COMP_E_2401`–`COMP_E_2402`).          |
| `JoinsetStrategy`'s `effective_join_type` (§5.3)    | Per-hop `JoinType` derivation from resolved Relationship's `optional`. | compile + plan; scope-local Relationship shadowing resolved at compile. |
| `JoinsetStrategy`'s key-pair emission (§5.2 step 3) | Plan-node `KeyPair` carriage.      | plan (lookup only; no validation).              |


## Testing Strategy

- Anchor-left plan emission; per-hop JoinType derivation per 18 §2.9; scope-local shadow resolution; PLAN_W_2402 ManyToManyHopAdvisory; PLAN_W_2403 MultiFanoutAdvisory.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
