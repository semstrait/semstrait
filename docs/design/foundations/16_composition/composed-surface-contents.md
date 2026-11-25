---
id: 16-composed-surface-contents
status: ratified
version: 2.1
impl: implemented
---

# UnifiedSemantics / FieldProvenance / CompositionCoverage

This leaf ratifies the namespace-aware merge logic (`UnifiedSemantics`), per-field ownership tracking (`FieldProvenance` + `FieldOwnership`), and composition-level coverage (`CompositionCoverage` extending `15 §6`). These three fields live inside `ComposedSemanticInterface` (see [composed-surface-type](composed-surface-type.md)) and describe **what the unified surface contains** and **where each field comes from**.

## 6. `UnifiedSemantics`

```rust
#[non_exhaustive]
pub struct UnifiedSemantics {
    pub dimensions: BTreeMap<UnifiedName, Dimension>,
    pub measures:   BTreeMap<UnifiedName, Measure>,
    pub metrics:    BTreeMap<UnifiedName, Metric>,
    pub filters:    BTreeMap<UnifiedName, Filter>,
    pub keys:       BTreeMap<UnifiedName, Key>,
}

pub struct UnifiedName(pub String); // bare `SemanticsName` or `constituent.name`
```

The namespace-aware merge of the constituents' `SemanticInterface`s. `UnifiedName` carries either a bare name (no collision) or a qualified name (`"constituent.name"`). All accessors on `SemanticsView` return `SemanticsName`s which may be qualified strings — consumers must treat them as opaque identifiers.

### 6.1 Input — constituents' `SemanticInterface`s

For each `DataKindRef` in `ComposedSemanticInterface.constituents`, the merge consults the constituent's `SemanticInterface` (from the SemanticManifest's per-kind index, I8). For `Unionset` and `Grainset` (any `Origin`), constituents are the top-level `ComplexDataKind` children (flattened per `12 §8`). For `Joinset` (any `Origin`), constituents are the author-declared member kinds (`Origin::Explicit`) or the kinds visited by the canonical-form enumeration (`Origin::Implicit`) plus any path-intermediate kinds the traversal touches (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md)).

### 6.2 Name-collision resolution — namespace-aware

When two or more constituents declare a `Semantics` of the same `SemanticsName` and the same role, the merge rule depends on `composition_kind`:

**`Unionset` / `Grainset`.** Collisions **unify**: both constituents promote the shared name into the composed surface under its bare `UnifiedName`. The `FieldOwnership` is `Shared(vec![A, B])` (§7.3.2). Per `24` / `22` this is the `UNION ALL` / grain-sharding default.

**`Joinset`.** Collisions **qualify** (regardless of `Origin`): each contributing constituent promotes its name into the composed surface under a qualified `UnifiedName` of the form `constituent.name`. The bare `name` does NOT exist on the composed surface — any Request referencing the bare form triggers `PlannerError::AmbiguousCompositionReference` (see [composition-validation-and-errors](composition-validation-and-errors.md), `PLAN_E_0505`) with the candidate qualifications in the Diagnostic.

**Example.** `orders` and `returns` each declare a `total: Measure`. Under a `Joinset` composing both — explicit or implicit — the composed surface exposes `orders.total` and `returns.total` but not a bare `total`.

**Role disagreement.** Two constituents declaring the same name with **different roles** (e.g. one `Dimension`, one `Measure`) is always an error, regardless of `composition_kind`: `CompileError::CompositionRoleConflict` (see [composition-validation-and-errors](composition-validation-and-errors.md)). This check runs at `compile` for explicit compositions; at plan for implicit ones.

### 6.3 Common-dimension promotion

When two or more constituents declare a `Dimension` of the same `SemanticsName`, the same `DataType`, and semantically compatible definition (same `expr` if any, same `type:` facet per `11 §6.1`), the merge promotes it to a single composed-surface dimension under the bare `UnifiedName` with `FieldOwnership::Shared`. This is the "all datasets have a `date` dimension" case.

**Compatibility requires:**

- Same role (`Dimension`).
- Same `data_type` (exact match; no widening).
- Same `type:` facet (`temporal.grains`, `categorical.values`, etc. match structurally per `11 §6.1`).
- Same `expr:` (structural AST equality) or both lacking `expr`.
- Same `description` (if any; mismatches emit `COMP_W_0401` advisory but do not block).

**Incompatible dimensions** with the same name follow §6.2's `composition_kind` rule — `Unionset` / `Grainset` unify under type widening (per `13`), `Joinset` qualifies.

**Rationale.** Promotion produces the ergonomic, "author probably meant this" surface. Qualification is available as a fallback when the promotion check fails.

### 6.4 `Measure` / `Metric` surface

**Aggregation-semantics conflict — rejected at compile.** Two constituents that declare a `Measure` of the same `SemanticsName` with **different aggregation functions** (e.g. `revenue` declared `agg: sum` on one constituent, `agg: avg` on another) **cannot** be unified: `CompileError::CompositionAggregationConflict` (see [composition-validation-and-errors](composition-validation-and-errors.md)) fires during composition materialization, regardless of `Origin` (the implicit-composition enumerator at compile sees the same constituent shapes as the explicit path). Authors resolve by renaming one side or narrowing the implicit Joinset enumeration via filters declared on an explicit `Joinset` (which differentiates it from the equivalent implicit form).

**Measures on one constituent, queried over the composed surface.** Permitted. The Measure surfaces under its bare name (no collision) or qualified name (collision). Planning consults `Cardinality` to decide whether to pre-aggregate (fanout-safe rewrite per [relationship-edge-model](relationship-edge-model.md) §3.3.2) or emit the straightforward join when the `Additivity × TemporalShape` interaction (per `17`) is unsafe — fanout falls to the author's declared shape.

**Metrics.** Aggregation-function conflicts are rejected by the same rules as Measures. Metrics that reference other Measures / Metrics via `expr:` are resolved per `19 §3.6.4`'s compile-time reference-DAG traversal; if the references span constituents, the implicit composition's `PathSignature` is recorded on the ResolvedExprEntry for the Metric and the planner consumes it at plan time.

**Filters.** Filters compose like Dimensions — promotion under compatibility, qualification otherwise.

### 6.5 Composed-surface keys — declare-or-derive

Every composed surface carries `keys` populated post-compile. The population rule depends on `composition_kind` and `origin`:

**`Unionset`** — composed-surface keys are the intersection of the constituents' `Key::Primary` / `Key::Unique` declarations sharing the same `SemanticsName` and type (per `11 §8.3`). Keys that hold on all constituents under union semantics are preserved; keys that hold on some but not all are dropped. Same rule for `Origin::Explicit` and `Origin::Implicit`.

**`Grainset`** — similar to `Unionset`; keys that hold at all declared grains survive. Always `Origin::Explicit` in v1.

**`Joinset` (`Origin::Explicit`)** — the author MAY declare `keys` on the `joinsets:` block. If declared, those keys win. Otherwise, the composed-surface primary key is **derived** from the anchor constituent's `Key::Primary` (see [joinset-model](joinset-model.md) §13.2). `Key::Foreign` declarations become internal join conditions (not user-surface keys).

**`Joinset` (`Origin::Implicit`)** — keys are **derived** from the anchor constituent (the first `DataKindRef` in the canonical `constituents` order, which corresponds to the canonical-form starting node). Same rule as the no-keys-declared explicit case; the implicit form has no author-declaration override.

**Why declare-or-derive (vs Round-1 "implicit = empty").** The unified Joinset model treats explicit and implicit as the same shape with different origin. Empty keys on implicit Joinsets would force the planner into a fallback path for "key-required" plan-time decisions (deduplication, GROUP BY pins, certain optimizer rewrites). Deriving keys eliminates the fallback; the derivation is the same logic the explicit-no-keys-declared path runs. (Q-COMP-018 closed 2026-04-29.)

## 7. `FieldProvenance`

```rust
#[non_exhaustive]
pub struct FieldProvenance {
    pub ownership: BTreeMap<UnifiedName, FieldOwnership>,
}

#[non_exhaustive]
pub enum FieldOwnership {
    Native(DataKindRef),              // this constituent natively provides the field
    Shared(Vec<DataKindRef>),         // multiple constituents agree on the field
    NullFill(Vec<DataKindRef>),       // listed constituents lack; NULL-fill on missing side
    Derived(PhysicalExpr),            // field exists only post-composition (rare)
}
```

Per-field ownership on a composed surface. Analogous to `15 §6`'s `Coverage` but at the composition level and with different semantics (ownership, not coverage-variant classification).

### 7.1 Purpose

The planner uses `FieldProvenance` to decide, for each requested field:

1. **Which constituent(s) contribute the data.** `Native` → one constituent. `Shared` → multiple, unifiable. `NullFill` → one provider, others contribute NULL. `Derived` → no constituent; synthesized post-join.
2. **Which Scan nodes need the field.** `Native(X)` → include in X's Scan column list; `Shared(V)` → include in all of V's Scans.
3. **When NULL-handling matters.** `NullFill` triggers `COALESCE` projection from the provider side.

### 7.2 `FieldOwnership` roster

The four variants cover the cases observed across the four `CompositionKind`s. `#[non_exhaustive]` per I10; additional variants may be introduced without semver breakage.

### 7.3 Per-variant semantics

<!-- cross-cut: none -->

#### 7.3.1 `Native(DataKindRef)`

The named constituent is the **sole** native provider. Other constituents may not declare this field, or declare it with an incompatible shape rejected by §6.3.

**Example (Joinset).** `orders.revenue` — only `orders` carries `revenue`; `customers` does not. `FieldOwnership::Native("orders")`.

**Planner lowering.** The field's `PhysicalExpr` (from the constituent's `ResolvedExprTable`, per `19 §3.2`) is retrieved; the Scan node for the constituent includes the necessary physical columns.

#### 7.3.2 `Shared(Vec<DataKindRef>)`

Two or more constituents agree on the field (same role, same type, same `expr:` — per §6.3's compatibility check). The ownership vec lists all contributing constituents.

**Example (Unionset).** `date` declared on every constituent with the same `temporal.grains: [day, week, month]` facet. `FieldOwnership::Shared(vec!["adwords", "facebook", "klaviyo"])`.

**Planner lowering.** Per `composition_kind`:

- **`Unionset`** — each constituent's Scan contributes its own `PhysicalExpr` for the field; the `UNION ALL` stacks them.
- **`Grainset`** — each participating grain's Scan contributes; the grain-router picks the cheapest.
- **`Joinset`** — the planner elects one constituent as the "canonical source" for the field (typically the shortest-path one from the Joinset's anchor) and projects through it; other constituents' contributions are used for coverage analysis but not projected. Same rule for `Origin::Explicit` and `Origin::Implicit`.

#### 7.3.3 `NullFill(Vec<DataKindRef>)`

Listed constituents **lack** the field; at least one constituent provides it natively. This is the `Unionset` / `Grainset` partial-coverage case (per `11 §*` / `15 §6.2`).

**Example (Unionset).** `unionset: paid_media` with three constituents `adwords`, `facebook`, `klaviyo`. `country` is declared on the union interface; `adwords` and `facebook` map it; `klaviyo` does not. `FieldOwnership::NullFill(vec!["klaviyo"])`. The native providers (`adwords`, `facebook`) are tracked separately in the `CompositionCoverage` entry for `country`.

**Planner lowering.** The constituent(s) lacking the field emit `NULL` (literal) in their Scan projection for that column. The `UNION ALL` stacks a NULL-contributing branch alongside the mapped branches.

#### 7.3.4 `Derived(PhysicalExpr)`

The field exists **only on the composed surface** — no constituent provides it, but a composition-level expression derives it from other composed fields. Rare; typical cases are composition-level synthesized keys, join-indicator columns, or composed metrics that reference cross-constituent measures.

**Example.** A `Joinset` that wants a `source` dimension encoding which constituent contributed each row of a `Unionset` constituent. The `source` field does not exist on any constituent; a `Derived` expression — e.g. `LIT('orders')` for rows from the `orders` branch — populates it at composition time.

**Planner lowering.** The `PhysicalExpr` is attached to the composed surface's post-join projection. No constituent Scan columns needed.

Whether `Derived` warrants a distinct variant (vs. folding into `Native` with a synthetic `DataKindRef`) is tracked as Q-COMP-015; Round 1 ratifies `Derived` distinct.

## 8. `CompositionCoverage`

```rust
#[non_exhaustive]
pub struct CompositionCoverage {
    pub entries: BTreeMap<(DataKindRef, UnifiedName), CoverageVariant>,
}
```

Reuses `15 §6`'s `CoverageVariant` enum (`Native` / `NullFill` / `Derived` / `Metadata`) but keyed by `(DataKindRef, UnifiedName)` rather than `(SourceIndex, SemanticsName)`.

### 8.1 Relation to `15 §6`

`15` owns Binding-level coverage: for each `(source_index, semantics)` pair within a single `Binding`, does the source natively provide the Semantics, NULL-fill it, or derive it? `16` extends the concept to the **composition level**: for each `(constituent_kind, unified_name)` pair on a composed surface, does the constituent natively provide the unified field, NULL-fill it, or derive it?

**`§6.4`'s scope boundary holds:** `15 §6.4` explicitly hands off composition-level coverage to `16`. `16` **consumes** `15`'s Binding-level coverage as input — the composition-level coverage for a constituent is a **fold** of the constituent's per-source coverage (§8.3).

### 8.2 Shape — keyed by `(DataKindRef, UnifiedName)`

The key tuple:

- `DataKindRef` — one constituent of the composition. Every constituent has an entry for every `UnifiedName` on the composed surface (missing = not-covered, not zero-entries — absence denotes the constituent does not participate in this field at all).
- `UnifiedName` — the composed-surface name (bare or qualified per §6.2).

Value: `CoverageVariant::{Native, NullFill, Derived, Metadata}`.

- **`Native`** — the constituent provides the field natively. Derived from `FieldOwnership::Native(this_kind)` or `FieldOwnership::Shared([..., this_kind, ...])`.
- **`NullFill`** — the constituent does not provide the field; NULL is projected. Derived from `FieldOwnership::NullFill([..., this_kind, ...])` or from the constituent being listed in a `Shared` ownership where its per-Binding `Coverage` records `NullFill`.
- **`Derived`** — rare; the constituent synthesizes the field from other fields via an `expr:` unique to that constituent.
- **`Metadata`** — the constituent provides the field as a metadata literal (path-token, etc.) eagerly resolved at compile per `15 §5.5` / `15 §8`. The composed surface inherits this provenance from the constituent's per-Binding `Coverage::Metadata`. Plan-time consumers treat a `Metadata` cell as a per-source constant; partial evaluation at the `UnresolvedSemanticExpr → ir::PhysicalExpr` lowering boundary may collapse composed expressions whose only non-`Native` inputs are `Metadata` cells (`15 §1`'s three-stratum note).

Per-constituent vs collapsed shape is tracked as Q-COMP-010; Round 1 ratifies per-constituent keyed-by-tuple.

### 8.3 Derivation — fold of per-Binding coverages

For each constituent and each `UnifiedName` on the composed surface, the composition-level coverage folds the constituent's per-Binding coverage entries (from `15 §6`):

```text
compose(kind_ref, unified_name) =
    let semantics = un_qualify(unified_name)               // strip `kind.` prefix if present
    let binding   = manifest.binding_for(kind_ref)         // I8: compile-time index
    match binding.coverage.entries.get((source_0, semantics)):
        Some(Native)   => Native
        Some(NullFill) => NullFill
        Some(Derived)  => Derived
        Some(Metadata) => Metadata
        None           => NullFill    // not mapped by any source
```

The `Metadata` case folds identically to `Native` / `Derived` (the constituent provides the field; the difference is the read path). `15 §8.4` ratifies that metadata-bound Semantics are fail-fast at compile if any source in the constituent's Binding cannot resolve them — so `compose` never sees a "metadata applicable on some sources, NullFill on others" mix within a single constituent.

For multi-source constituents (a `Unionset` / `Grainset` constituent within a `Joinset`), the fold across sources follows the constituent's own composition rules (per `22` / `24`) first, then the result is re-folded into the outer composition. Authors of nested compositions (e.g. `Joinset` over `Unionset`) can trace each field's provenance back to the originating source via this cascade.

**Use by planner.** `CompositionCoverage` drives per-Scan column inclusion and NULL-projection decisions. A Request selecting only `Native`-covered fields can prune `NullFill`-only constituents from the plan entirely under `composition_kind == Joinset` (regardless of `Origin`); under `Unionset` / `Grainset` all constituents participate regardless.

## Testing Strategy

- **Invariants:** I8 (manifest carries primitives; graph reconstructs compositions from `semantics` + `coverage`), I10 (non-exhaustive enums: `FieldOwnership`).
- **Namespace-aware merge:** Tests that `Unionset` / `Grainset` unify shared names as `Shared`, that `Joinset` qualifies collisions, that role-disagreement fires `COMP_E_0404`, that aggregation-function conflict fires `COMP_E_0405`, that common-dimension promotion succeeds under §6.3 compatibility and fails otherwise, that `COMP_W_0401` advisory fires on `description` mismatch during promotion.
- **FieldProvenance:** Tests that `Native` / `Shared` / `NullFill` / `Derived` are correctly derived from constituent interface shapes, that planner lowering uses `FieldProvenance` to drive Scan column inclusion and NULL projection, that `Joinset` elects the canonical source for `Shared` fields.
- **CompositionCoverage:** Tests that composition-level coverage folds from `15 §6` per-Binding coverage, that `Metadata` cells fold correctly, that multi-source constituents cascade fold through nested compositions, that planner uses `CompositionCoverage` to prune `NullFill`-only constituents under `Joinset`.
- **Error codes:** `COMP_E_0404` (role conflict), `COMP_E_0405` (aggregation conflict), `COMP_W_0401` (shared dimension description mismatch), `PLAN_E_0505` (ambiguous composition reference) — see [composition-validation-and-errors](composition-validation-and-errors.md).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
