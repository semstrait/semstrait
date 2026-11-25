---
id: 23-unionset-shape
status: ratified
version: 2.1
impl: drift
---

# Unionset — TemporalShape & Grain

## 5. TemporalShape Interaction

### 5.1 Authoring placement and cascade-up

A Unionset's effective `TemporalShape` is **inherited from its children** under the V1 strict equivalence rule (§5.2). Authoring at the Unionset level is permitted via `extras.temporal:` (`ComplexExtras`) and must agree with children's; silent authoring at the Unionset level reads the (unanimous) children's shape as the effective value.

Per `26 §3.1` and the `32 §4` cascade table, only `temporal.<variant>:` (the shape kind) cascades; `grain` is leaf-only and lives on each leaf descendant's `LeafExtras` (`32 §4`).

### 5.2 V1 strict equivalence rule

In V1, all children of a Unionset MUST have **equivalent** TemporalShape:

- **Equal `kind`** — `Timeseries`, `Events`, `Snapshot`, or `Scd` — must match across all children.
- **Equal `ScdType`** when `kind = Scd` — `Scd(Type1) + Scd(Type2)` is rejected; subtype must match.
- **Equal `grain`** when present on leaf children — `grain: Day` and `grain: Hour` cannot coexist.

Mismatch is a hard compile error: `COMP_E_2301 UnionsetChildShapeMismatch { unionset, children: Vec<(name, TemporalShape)> }` (§8). Single consolidated code (per C1 ratification, 2026-05-03) covering all equivalence violations (kind, subtype, grain).

**Future direction.** Smart alignment of non-equivalent shapes (e.g. `Scd + Snapshot` projecting `Snapshot` as a degenerate `Scd(Type1)`) is post-v1 and tracked as `[TD-UNIONSET-SHAPE-ALIGN]` in `open/23_questions.md`.

### 5.3 Multi-as-of `Snapshot` advisories — survive equivalence rule

Two `Snapshot` children with equivalent shape (same `kind`, same `grain`) but different as-of timestamps are equivalent under §5.2 — yet semantically heterogeneous (rows mix two distinct snapshot worlds). Two advisories survive:

- `COMP_W_2302 UnionsetSnapshotMultipleAsOf` — the composed surface exposes the snapshot's identifying Dimension (`snapshotted_at` per `18 §3.3`); rows are distinguishable but the Union mixes worlds.
- `COMP_W_2303 UnionsetSnapshotMultipleAsOfNoDiscriminator` — the composed surface does NOT expose `snapshotted_at`; rows are indistinguishable.

Both are warnings. (Per C2 ratification, 2026-05-03.)

### 5.4 Scope boundary with `17`

`23` ratifies only:

- the `extras.temporal:` field carriage on `ComplexExtras` (cross-ref `32 §4`),
- the V1 strict equivalence rule across children (§5.2),
- the cascade-up direction (children → Unionset).

`23` does NOT ratify the `TemporalShape` enum variants, the SCD subtype catalog, the shape × grain rollup matrix, the `AsOf` join variant, or the advisory-warning predicates beyond the multi-as-of cases above. All of these live in `17`.

## 6. Grain Interaction

### 6.1 Cascade — equal grain across children

`grain` lives on each leaf descendant's `LeafExtras` (per `32 §4`); Complex variants do not author `grain` directly. Under §5.2's strict equivalence rule, the union of every leaf descendant under a Unionset must yield a single common `grain`. Mismatch is `COMP_E_2301` (the same consolidated code; grain is part of TemporalShape per `17`).

### 6.2 Plan-time consultation

Two observable invariants involve `grain`:

- **Rollup coarsening.** When a Request rolls up a temporal Dimension on the Unionset's surface, the target grain must be at least as coarse as the children's common `grain`. A `Day`-grain Unionset rejects an `Hour`-target rollup — same code reused from `21 §6.2` (`PLAN_E_2102 RequestGrainFinerThanSource`); the diagnostic locator points at the Unionset rather than a single Dataset.
- **Shape-gated legality.** When `temporal.kind` is also declared, the rollup is shape-gated per `17`'s matrix.

### 6.3 Scope boundary with `13` and `17`

- `13 §5` ratifies the `Grain` enum and its coarseness order.
- `17` ratifies the `TemporalShape × Grain` legality matrix.
- `22` ratifies grain routing (the Grainset-specific dispatch across children at different grains; out of scope for `23`'s equivalent-grain rule).

## Testing Strategy

- V1 strict equivalence (equal kind/ScdType/grain); COMP_E_2301 UnionsetChildShapeMismatch; COMP_W_2302/2303 multi-as-of Snapshot advisories; cascade-up.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
