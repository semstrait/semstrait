---
id: 22-grainset-shape
status: ratified
version: 2.1
impl: drift
---

# Grainset — TemporalShape Interaction

## 5. TemporalShape Interaction

### 5.1 Authoring placement and cascade-up of `kind` only

A Grainset's effective `TemporalShape.kind` is **inherited from its children** under the V1 strict equivalence rule (§5.2). Authoring at the Grainset level is permitted via `extras.temporal:` (`ComplexExtras`) and must agree with children's; silent authoring at the Grainset level reads the (unanimous) children's kind as the effective value.

Per SR-E-7 (`18 §3.3`) and the `32 §4` cascade table, only `temporal.<variant>:` (the shape kind) may appear on `ComplexExtras`. **`grain` is type-level forbidden** at the Grainset level — it lives on each child's `LeafExtras` only (per `32 §4`'s cascade table) and per SR-E-8 (`18 §3.4`) MUST be authored explicitly on every Grainset child. Ambient grain inheritance from the Grainset to children is structurally impossible (the `ComplexExtras` shape has no `grain` field).

### 5.2 V1 strict equivalence rule + ≥ 2 unique grains

In V1, all children of a Grainset MUST have:

1. **Equal `kind`** — `Timeseries`, `Events`, `Snapshot`, or `Scd` — must match across all children. Mismatch is `COMP_E_2201 GrainsetChildShapeKindMismatch` (§8). Single consolidated code per the parallel C1 ratification used for Unionset.
2. **Equal `ScdType`** when `kind = Scd` — `Scd(Type1) + Scd(Type2)` is rejected; subtype must match.
3. **Children's grain values must include ≥ 2 unique values** — stronger than R3's `≥ 2 children`. A Grainset with `[Day, Day]` (2 children, 1 unique grain) is degenerate (it's just an implicit Unionset at Day with no distinct routing). Validated via `VALID_E_2202 GrainsetInsufficientUniqueGrains` (§7).

**Future direction.** Smart alignment of non-equivalent shapes (e.g. `Events + Snapshot` mixing) is post-V1 and tracked as `[TD-GRAINSET-SHAPE-MIX]`. The pre-cascade `MixedShapeAdvisoryChildren` warning is retired.

### 5.3 Scope boundary with `17`

`22` ratifies only:

- the `extras.temporal:` field carriage on `ComplexExtras` (cross-ref `32 §4`),
- the V1 strict equivalence rule across children (§5.2),
- the cascade-up direction (children → Grainset, `kind` only).

`22` does NOT ratify the `TemporalShape` enum variants, the SCD subtype catalog, the shape × grain rollup matrix, the per-shape pin / anchor mechanics, or the `RollupPolicy` enum's default-derivation logic. All of these live in `17`.

## Testing Strategy

- V1 strict equivalence (equal kind, equal ScdType if Scd, ≥2 unique grains); COMP_E_2201 GrainsetChildShapeKindMismatch; cascade-up of kind only.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
