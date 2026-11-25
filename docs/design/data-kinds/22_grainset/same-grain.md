---
id: 22-grainset-same-grain
status: ratified
version: 2.1
impl: drift
---

# Grainset — Same-Grain Pre-Merge

### 3.3 Same-grain pre-merge — implicit Unionset (`mode: All`, non-strict)

When two or more children share the same `extras.temporal.grain:` value, they are folded at compile into an **implicit Unionset** (`mode: All`, **non-strict** Coverage discipline — i.e. NullFilling per `23 §3.2` rather than the Dataset-multi-source strict discipline of `21 §3.2`). The implicit Unionset becomes one **effective routing unit** at that grain (§4.1). Per G-1 (2026-05-03 ratification).

**Identification.** Same-grain implicit Unionsets get content-derived hash-ids per `33 §<implicit-unionset-id>` (forthcoming) — same mechanism as `21 §3.2`'s multi-source-Dataset implicit Unionsets. They carry no `name:` (authors don't see them).

**Discipline contrast with Dataset multi-source.** Dataset's multi-source implicit Unionset is **strict** (`21 §3.2` `COMP_E_2106` on missing coverage); Grainset's same-grain implicit Unionset is **non-strict** (NullFill at the seam per `23 §4.3`). Reason: same-grain Grainset children are typically authored as variants of the same data slab with intentional roster differences (e.g. two daily snapshots from different business units, each carrying its own subset of Measures), whereas Dataset multi-source represents the same data sliced by physical layout (paths / partitions) where roster heterogeneity signals an authoring mistake.

## Testing Strategy

- Same-grain pre-merge produces implicit Unionset (mode All, non-strict per 23 §4.3); effective routing unit = one per distinct grain; content-derived hash-id per 33.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
