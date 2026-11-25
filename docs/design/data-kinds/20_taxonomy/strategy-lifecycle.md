---
id: 20-strategy-lifecycle
status: ratified
version: 2.1
impl: implemented
---

# DataKind Strategy & Lifecycle — Dispatch, Per-Stage Skeleton, Shared Error Codes

This leaf ratifies the `Strategy` trait surface and its plan-time dispatch contract, the shared per-stage lifecycle responsibilities every variant participates in, and the shared DataKind error-code roster (`*_E_2000`–`*_E_2099`).

## 5. Construction / Resolution Strategy — the Strategy Taxonomy

### 5.1 Strategy-per-variant principle

| Variant | Strategy | Authoritative doc |
|---|---|---|
| `Simple` | `SimpleStrategy` | `21 §*` |
| `Grainset` | `GrainsetStrategy` | `22 §*` |
| `Unionset` | `UnionsetStrategy` | `23 §*` |
| `Joinset` | `JoinsetStrategy` | `24 §*` |

> **Invariant D9 — variant-to-strategy binding is total and exclusive.** Every ratified `DataKind` variant maps to exactly one `Strategy` type. No `UnionsetOrGrainsetStrategy`, no `CompositeStrategy`.

The current codebase names the trait `DataKindPlanner` and the registry `DataKindPlannerRegistry`; the rename to `Strategy` / `StrategyRegistry` is tracked in `docs/TECH_DEBT.md` per I9.

### 5.2 Trait surface — `planner::Strategy`

`Strategy` lives in `semstrait-planner` (per `34`'s public surface); `20` ratifies its **shape**, not its concrete field names (which are `34`'s scope).

```rust
pub trait Strategy: Send + Sync {
    /// Resolve the given Request slice against this DataKind using the
    /// SemanticManifest. Returns the `PlanNode` subtree rooted at this DataKind.
    fn resolve(
        &self,
        manifest: &SemanticManifest,
        request: &RequestSlice,
        ctx: &mut PlannerCtx,
    ) -> Result<PlanNode, PlanError>;

    /// The strategy's human-readable name, for diagnostics.
    fn name(&self) -> &'static str;
}
```

Strategies are stateless once constructed; per-invocation state lives in `PlannerCtx`. No `async` (I6); no catalog calls, no filesystem calls, no expression re-compilation (I5, I11). A `UnionsetStrategy` (analogously `GrainsetStrategy`, `JoinsetStrategy`) resolves its own composed surface, then **recursively dispatches** into constituents via `ctx.strategy_registry.dispatch(child)`.

### 5.3 Dispatch mechanism at plan time

```rust
pub fn dispatch_strategy<'r>(
    kind: &ResolvedDataKind,
    registry: &'r StrategyRegistry,
) -> &'r dyn Strategy {
    match kind {
        ResolvedDataKind::Simple(_) => registry.simple(),
        ResolvedDataKind::Complex(ResolvedComplexDataKind::Unionset(_)) => {
            registry.unionset()
        }
        ResolvedDataKind::Complex(ResolvedComplexDataKind::Grainset(_)) => {
            registry.grainset()
        }
        ResolvedDataKind::Complex(ResolvedComplexDataKind::Joinset(_)) => {
            registry.joinset()
        }
        // Future `Complex` variants (I10) require a new arm here and a new
        // method on `StrategyRegistry`. The `#[non_exhaustive]` on
        // `ResolvedDataKind` / `ResolvedComplexDataKind` forces a match
        // arm to be added in a MINOR bump (`30 §2`).
    }
}
```

**Dispatch guarantees.**

- **O(1) per dispatch.** Each arm is a direct reference-returning branch. I6 hot-path safe.
- **One match site.** Every other planner site that needs a strategy consumes `&dyn Strategy` from this function.
- **`#[non_exhaustive]` discipline.** Adding a future `Complex` variant forces this match arm to be extended; no silent compile-time hole.
- **Strategy registry ownership.** The `StrategyRegistry` holds exactly one strategy per variant. Constructed at planner init, shared across requests (all strategies are `Send + Sync`).

`PLAN_E_2050 StrategyDispatchFailed` covers the "no strategy for this variant" case, which — given the `#[non_exhaustive]` compile-time check — should never fire for ratified variants. Reserved defensively for third-party extensions.

---

## 6. Lifecycle — Per-Stage Responsibilities (Shared Skeleton)

### 6.1 `parse` (stage 1)

Variant discriminator recognition only. Per-variant YAML surface in `32` + per-variant docs. Per-variant `ParseError`s in `PARSE_E_02xx` per `30 §6.2`. No `20`-scope codes.

### 6.2 `validate` (stage 2)

**Shared responsibility (`20`):** Structural checks that apply to every variant:

- `26 §1`'s nesting matrix (enforcement is `26`'s; type-level projection at `32 §3.2`).
- §2.3's Invariant D1 — no `binding:` key on any `ComplexDataKind`.
- §4.1's Invariant D2 — top-level DataKind name global uniqueness.
- §4.4's Invariant D5 — interface-type match to variant (manifest layer).

Diagnostics accumulate per `10 §5`'s fail-accumulate policy.

**Per-variant responsibility (`21`–`24`):** Per-variant structural Preconditions — Unionset's "≥2 children" (`12 §3.2`), Grainset's coarsest-first ordering (`12 §4.2`), Joinset's binary-v1 arity (`12 §5.3`), and each variant's block-shape rules.

**`20`-scope error codes:** `VALID_E_2000`–`VALID_E_2029` — see §8.2.

### 6.3 `compile` (stage 3)

**Shared responsibility (`20`):**

- Dispatch per-variant resolution in dependency order (children before parents per `10 §4.3`).
- For `Simple`: delegate to `15 §10`'s Binding resolution flow.
- For `Complex`: produce a `ResolvedComplexDataKind` carrying (a) resolved constituent references, (b) synthesized `ComposedSemanticInterface` per `16 §5` / `§6` / `§7` / `§8`, (c) composition-level `FieldProvenance` / `CompositionCoverage` records.
- Fail fast on first error per `10 §4.4`.
- Register the resolved DataKind in the SemanticManifest's `DataKindIndex` (`33 §*`).

**Per-variant responsibility (`21`–`24`):** `Simple` (`21`) — single-Binding resolution; `Unionset` (`23`) — branch-by-branch resolution + `CompositionCoverage` fold; `Grainset` (`22`) — per-level resolution + temporal-shape interaction; `Joinset` (`24`) — member Binding resolution + path validation against `Relationship` graph (`16 §2`) + materialized-surface synthesis.

**`20`-scope error codes:** `COMP_E_2000`–`COMP_E_2029` — see §8.2. Notably `COMP_E_2005 MissingCoverage` (D4) and `COMP_E_2010 StrategyBindingUnresolved`.

`compile`'s depth-first traversal is deterministic per I4 — siblings processed in name order (child collections are `EntityId`-keyed but projected name-ordered per `32 §7`, post id-first rework); the `DataKindIndex` is populated in the same order.

### 6.4 `plan` (stage 4)

**Shared responsibility (`20`):**

1. Field-first resolution (`16 §11`) locates the top-level DataKind owning each requested field.
2. Strategy dispatch (§5.3) returns `&dyn Strategy`.
3. `Strategy::resolve(manifest, request_slice, ctx)` returns the `PlanNode` subtree.
4. Result splicing into the overall `SemanticPlan`.

All four steps synchronous (I6); no I/O (I11).

**Per-variant responsibility (`21`–`24`):** The per-variant algorithm inside `Strategy::resolve`.

**`20`-scope error codes:** `PLAN_E_2040`–`PLAN_E_2069` — see §8.2.

### 6.5 `optimize` / `adapt` (stages 5–6)

No `20`-scope responsibility. Both stages operate at the `PlanNode` level (`10 §5.4` / `§5.5`) and are variant-agnostic.

---

## 7. Applicability Cross-Cuts

Forward-pointer to [`25_applicability_matrix.md`](../25_applicability_matrix.md) — the full cross-cut of "which foundation rule applies to which DataKind variant." Each row in `25` is a foundation rule (`10`–`17`); each column is a DataKind variant; cells are `always` / `conditional` / `n/a`.

---

## 8. Error-Code Roster

### 8.1 Allocation summary

| Range | Scope | Doc |
|---|---|---|
| `*_E_2000`–`*_E_2099` | shared across all DataKind variants | `20` (this doc) |
| `*_E_2100`–`*_E_2199` | `SimpleDataKind` / Dataset | `21` |
| `*_E_2200`–`*_E_2299` | `Grainset` | `22` |
| `*_E_2300`–`*_E_2399` | `Unionset` | `23` |
| `*_E_2400`–`*_E_2499` | `Joinset` | `24` |
| `*_E_2500`–`*_E_2599` | applicability matrix / cross-variant planner | `25` |
| `*_E_2600`–`*_E_2699` | reserved for future `Complex` variants per I10 | — |

> **Cross-doc fix CDF-30-01.** `30 §6.2`'s current `VALID_E` / `COMP_E` / `PLAN_E` subsystem caps stop below `2000`. Each must be extended to include a reserved `2000`–`2999` band labeled "data-kinds taxonomy block (`20`–`25`)", with `2000`–`2599` allocated per the table above. MINOR per `30 §2`. Tracked here pending the `30 §6.2` table update.

The subsystem prefix (`*`) depends on the stage the diagnostic surfaces at; see §8.2.

### 8.2 `20`-scope error codes (the `*_E_2000`–`*_E_2099` block)

Every code follows `{SUBSYSTEM}_{SEVERITY}_{NUMBER}` per `30 §6.1`. Variants are `#[non_exhaustive]` per I10.

#### `VALID_E_20xx` — shared structural validation (validate stage)

| Code | Variant | Meaning |
|---|---|---|
| `VALID_E_2000` | `DataKindNameCollision` | Two top-level DataKinds share the same name. Violates D2 (§4.1). |
| `VALID_E_2001` | `BindingOnComplex` | A `ComplexDataKind` YAML block carries a `binding:` key. Violates D1 (§2.3.1). |
| `VALID_E_2002` | `InterfaceTypeMismatch` | At the SemanticManifest layer, a `ResolvedComplexDataKind` produced a bare `SemanticInterface` or vice versa. Violates D5 (§4.4). |
| `VALID_E_2003` | _(reserved — formerly `NestingCapabilityInconsistent`; retired with the runtime capability struct)_ | Reserved against future use. |
| `VALID_E_2004` | `StructuralLabelCollision` | Two sibling nested DataKinds share a structural label under the same parent (`11 §10.3`). |
| `VALID_E_2005` | `DataKindNameReserved` | A DataKind's name shadows a reserved identifier. Reserved against `11 §4`'s future expansion. |

#### `COMP_E_20xx` — shared compile-time errors

| Code | Variant | Meaning |
|---|---|---|
| `COMP_E_2000` | `DataKindCompileFailed` | Generic wrapper when a variant's compile path returns an error without a more-specific subsystem code. |
| `COMP_E_2001` | `DataKindTraversalOrderInvalid` | Compile's depth-first traversal hit a DataKind whose children were not yet resolved. Should never fire for ratified content. |
| `COMP_E_2005` | `MissingCoverage` | A `ResolvedDataKind` attempted to enter the SemanticManifest without a coverage surface. Violates D4 (§4.3). |
| `COMP_E_2010` | `StrategyBindingUnresolved` | A variant's strategy was invoked against a `ResolvedDataKind` whose required resolution data is missing (e.g. a Simple with `ResolvedBinding::None` or a Complex with empty constituent list). |
| `COMP_E_2015` | `InterfaceSynthesisFailed` | For a `ComplexDataKind`, the synthesized `ComposedSemanticInterface` could not be produced. `16 §14` owns specific COMP_E_04xx codes; `2015` is the variant-boundary wrapper. |
| `COMP_E_2020` | `DataKindIndexConflict` | Populating `DataKindIndex` found a conflict — two `ResolvedDataKind`s resolved to the same index key. Should be caught earlier by `VALID_E_2000`. |

#### `PLAN_E_20xx` — shared plan-time errors

| Code | Variant | Meaning |
|---|---|---|
| `PLAN_E_2040` | `DataKindNotInSemanticManifest` | Field-first resolution or explicit `Request.from` named a DataKind not in `DataKindIndex`. |
| `PLAN_E_2050` | `StrategyDispatchFailed` | The strategy registry produced no `Strategy` for a variant. Reserved defensively (the `#[non_exhaustive]` match in §5.3 makes this fire only for third-party `Complex` variants without registered strategies). |
| `PLAN_E_2051` | `StrategyMissingForVariant` | A ratified variant has no registered `Strategy`. Planner init bug. |
| `PLAN_E_2052` | `SemanticManifestIndexInconsistent` | During dispatch, `DataKindIndex` returned a `ResolvedDataKind` whose variant tag disagrees with its inner content. Should not fire for ratified content. |

**Reserved `20`-scope codes:** `VALID_E_2010`–`2029`, `COMP_E_2030`–`2049`, `PLAN_E_2060`–`2099`.

### 8.3 Per-variant sub-range allocations (`21`–`25`)

Each of `21`–`25` owns a 100-code sub-range. Within each, authors follow `30 §6.4`'s discipline.

| Sub-range | Recommended sub-structure per variant |
|---|---|
| `2100`–`2129` (`21` Simple) | `VALID_E_21xx` block-shape; `COMP_E_21xx` per-Simple Binding checks; `PLAN_E_21xx` per-Simple Strategy failures. |
| `2200`–`2229` (`22` Grainset) | `VALID_E_22xx` level ordering, grain axis match; `COMP_E_22xx` per-level coverage folding; `PLAN_E_22xx` level-selection failure. |
| `2300`–`2329` (`23` Unionset) | `VALID_E_23xx` branch well-formedness; `COMP_E_23xx` branch coverage folding, NULL-fill typing; `PLAN_E_23xx` branch-selection failure. |
| `2400`–`2429` (`24` Joinset) | `VALID_E_24xx` binary-v1, path well-formedness, relationship cross-check; `COMP_E_24xx` anchor resolution, materialized surface synthesis; `PLAN_E_24xx` join assembly failure. |
| `2500`–`2529` (`25` applicability matrix) | `PLAN_E_25xx` cross-variant planner failures surfaced only at the matrix level. |

Each variant's reserved 100-code sub-range has a further-reserved tail (70 codes) against future growth.

---

Open questions (`Q-KIND-001`..`004`) are tracked in [`questions/open/20_questions.md`](../../questions/open/20_questions.md).

**End of `20_taxonomy.md`.**

## Testing Strategy

- Invariant D9 (variant-to-strategy binding is total and exclusive, §5.1) is proven by asserting every ratified variant maps to exactly one `Strategy`, and that the §5.3 dispatch `match` is `#[non_exhaustive]`-guarded so a new `Complex` variant forces a new arm (no silent compile-time hole).
- Strategy dispatch guarantees (§5.3) are proven by asserting O(1) single-match-site dispatch and that missing-strategy cases raise `PLAN_E_2050 StrategyDispatchFailed` / `PLAN_E_2051 StrategyMissingForVariant`.
- Shared validate-stage codes (§8.2) are proven by fixtures asserting `VALID_E_2000 DataKindNameCollision` (D2), `VALID_E_2001 BindingOnComplex` (D1), `VALID_E_2002 InterfaceTypeMismatch` (D5), and `VALID_E_2004 StructuralLabelCollision`.
- Shared compile-stage codes (§8.2) are proven by asserting `COMP_E_2005 MissingCoverage` (D4), `COMP_E_2010 StrategyBindingUnresolved`, `COMP_E_2015 InterfaceSynthesisFailed`, and `COMP_E_2020 DataKindIndexConflict`.
- Lifecycle skeleton (§6) upholds I4/I6: `compile` depth-first traversal is deterministic (name-ordered `DataKindIndex` population, §6.3) and `plan` dispatch is synchronous with no I/O (§6.4); every code matches `30 §6.1` format with ranges reserved per `30 §6.2`.

See the shared harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
