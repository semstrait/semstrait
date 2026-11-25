---
id: 16-composition-validation-and-errors
status: ratified
version: 2.1
impl: implemented
---

# Composition Validation and Errors

This leaf ratifies the `Relationship` graph well-formedness preconditions (checked at `validate` and `compile`) and the composition error model: new `CompileError`, `ValidateError`, and `PlannerError` variants with stable codes in the `COMP_E_04xx` / `PLAN_E_05xx` / `PLAN_W_05xx` ranges.

## 12. `Relationship` Graph Well-formedness

Checked at `validate` (declarations) and `compile` (type-dependent validation). All violations abort the containing stage per I12 (fail-fast).

### 12.1 Duplicate `Relationship`s — `ValidateError::DuplicateRelationship`

Two `Relationship`s that share the same unordered `{from, to}` pair **and** the same `Vec<KeyPair>` (up to permutation) are duplicates. Under bidirectional semantics, `A → B` and `B → A` are the same edge; the author must choose one direction to declare.

**Detection.** At `validate`, for each `Relationship`, canonicalize the `{from, to}` into a sorted pair and the `KeyPair` list into a sorted vec; collect into a set; a collision emits `ValidateError::DuplicateRelationship { between: (DataKindRef, DataKindRef), relationship_ids: (EntityId, EntityId) }`.

**Intentional multi-edges.** Authors who want two joins between the same pair (e.g. a "primary customer" edge and a "shipping customer" edge) declare two `Relationship`s with **different** `keys` (different `SemanticsName`s). The unordered `{from, to, keys}` triple must be unique.

### 12.2 `KeyPair` type agreement — `CompileError::RelationshipKeyTypeMismatch`

At `compile` (after `19 §3.6`'s type inference), for each `KeyPair { left, right }` in each `Relationship`:

1. Resolve `left`'s inferred `DataType` by consulting `19 §3.2`'s `ResolvedExprTable` for the `Relationship.from` binding (or each binding of a multi-source constituent).
2. Resolve `right`'s similarly.
3. Apply `13 §4`'s type-compatibility relation.
4. If incompatible, emit:

```text
CompileError::RelationshipKeyTypeMismatch {
    relationship_id,
    key_pair_index,
    left_type: DataType,
    right_type: DataType,
}
```

**Widening.** Numeric sides widen per `13 §5` (narrower side promotes to wider side). No error if widening succeeds. String sides require exact match — `String` vs `StringLarge` (if `13` distinguishes them) is an error. Temporal sides require equal precision.

**Multi-source constituents.** If `Relationship.from` is a `Unionset` / `Grainset`, the key's type must agree across all source bindings (compatible per `13 §4`). Mismatch within one constituent's sources is a `15 §6`-domain error (`CoverageInconsistency`); mismatch across constituents of the same side of a `Relationship` is the same `CompileError::RelationshipKeyTypeMismatch` with `left_type` = the disagreeing-source type.

### 12.3 No self-references — `CompileError::RelationshipSelfReference`

A `Relationship` with `from == to` is rejected in v1:

```text
CompileError::RelationshipSelfReference { relationship_id, kind: DataKindRef }
```

Self-joins (e.g. `employees → managers` on the same `employees` kind) are a legitimate modeling need, but v1 does not support them — the compile-time implicit-composition enumeration (see [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4) and the plan-time field-first lookup (see [field-first-resolution](field-first-resolution.md) §11) both assume simple-graph edges without self-loops. Deferred as `[TD-COMPOSITION-SELFJOIN]`. Authors needing self-joins in v1 declare two distinct `DataKind`s (typically with the same underlying `Binding`) and a `Relationship` between them.

### 12.4 `Cardinality` / `JoinType` consistency

Soft check — emits structural advisory `PLAN_W_0503 RelationshipCardinalityKeyMismatch` (§14.4) when declared `Cardinality` contradicts `Key::Primary` / `Key::Unique` declarations on the key sides. v1 has no intent-level fanout advisories — fanout is the natural consequence of the relationship's declared cardinality and the author owns it.

### 12.5 Graph connectivity — observational only

`validate` does NOT reject disconnected `RelationshipGraph`s. A Model can legitimately have several disconnected subgraphs (multiple business domains under one SemanticManifest). Disconnection only matters at plan time, when a Request's selected names span disconnected owners (`PlannerError::NoCompositionPath`, §14.3).

## 14. Error Model

New variants extend `CompileError`, `ValidateError`, and `PlannerError` (defined in `10 §5`). Stable codes allocated per `30 §6`:

- **`COMP_E_0400-0499`** — compile-stage composition errors.
- **`PLAN_E_0500-0599`** — plan-stage composition errors.
- **`PLAN_W_0500-0599`** — plan-stage composition advisories (warnings).

All variants are `#[non_exhaustive]` per I10 on the parent enum.

### 14.1 `CompileError` additions

<!-- cross-cut: none -->

| Variant | Code | When |
|---|---|---|
| `RelationshipKeyTypeMismatch { relationship_id, key_pair_index, left_type, right_type }` | `COMP_E_0401` | §12.2 — `KeyPair` sides have incompatible `DataType`s after `19 §3.6` inference. Defined in `../18_entities.md#2-relationship`, elaborated here. |
| `RelationshipSelfReference { relationship_id, kind }` | `COMP_E_0402` | §12.3 — `Relationship.from == Relationship.to` (self-join, deferred). |
| `RelationshipKeyNotJoinable { relationship_id, key_pair_index, side, role }` | `COMP_E_0403` | [Relationship-edge-model](relationship-edge-model.md) §2.3 — `KeyPair` references a `Measure` / `Metric` / `Filter` rather than `Key` / `Dimension`. |
| `CompositionRoleConflict { composition_name, name, roles }` | `COMP_E_0404` | [Composed-surface-contents](composed-surface-contents.md) §6.2 — constituents declare same name with different roles (Dimension vs Measure). |
| `CompositionAggregationConflict { composition_name, name, aggregations }` | `COMP_E_0405` | [Composed-surface-contents](composed-surface-contents.md) §6.4 — constituents declare same Measure name with different `agg:`. |
| `JoinsetUnknownRelationship { joinset_name, relationship_name }` | `COMP_E_0407` | `Joinset` references a `Relationship` that does not exist. |
| `JoinsetUnreachableConstituent { joinset_name, constituent }` | `COMP_E_0408` | `Joinset` declares a constituent not connected to the root via declared `Relationship`s. |
| `ImplicitEnumerationExploded { count, cap }` | `COMP_E_0409` | [Implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4 / §10.5 — implicit Joinset + Unionset enumeration exceeded `MAX_IMPLICIT_ENUMERATION_COUNT`. Author tightens the implicit graph by declaring explicit Joinsets for common subsets or restructuring Relationships. |
| `ExplicitImplicitCompositionClash { explicit_name, canonical_kind, candidate_differentiators }` | `COMP_E_0414` | [Implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.6 (authoritative statement) / [joinset-model](joinset-model.md) §13.5 — explicit `Joinset` / `Unionset` canonical form matches an enumerable implicit composition. Author either drops the explicit declaration or adds a differentiator (scope-local `Relationship` shadow, Joinset-level `filter:`, declared `keys`, or non-shortest path). |

`COMP_E_0406 JoinsetCardinalityOverride` is **retired (2026-05-12)**: the per-edge override surface (`JoinTypeOverrides`) is removed; a Joinset that needs different join semantics declares a scope-local `Relationship` per [joinset-model](joinset-model.md) §13.3, which is a full independent declaration, not an override. Code reserved for forward-compat (no MINOR re-allocation).

### 14.2 `ValidateError` additions

<!-- cross-cut: none -->

| Variant | Code | When |
|---|---|---|
| `DuplicateRelationship { between, relationship_ids }` | `COMP_E_0410` | §12.1 — two `Relationship`s declare the same unordered `{from, to, keys}`. |
| `RelationshipDataKindUnknown { relationship_id, side, kind }` | `COMP_E_0411` | `Relationship.from` / `.to` references an undeclared `DataKindRef`. |
| `RelationshipOnNestedKind { relationship_id, side, kind }` | `COMP_E_0412` | [Relationship-edge-model](relationship-edge-model.md) §2.1 — `Relationship.from` / `.to` references a nested (non-Root-scope) kind. |
| `RelationshipEmptyKeys { relationship_id }` | `COMP_E_0413` | [Relationship-edge-model](relationship-edge-model.md) §2.2 — `keys: Vec<KeyPair>` is empty. |

`ValidateError` uses the `COMP_E_04xx` range because these are composition-specific checks even though they run at `validate`. The `30 §6` allocation permits validate-stage codes to share range with compile-stage codes when they're scoped to the same conceptual area.

### 14.3 `PlannerError` additions

<!-- cross-cut: none -->

| Variant | Code | When |
|---|---|---|
| `AmbiguousImplicitComposition { constituent_set, candidates }` | `PLAN_E_0500` | [Field-first-resolution](field-first-resolution.md) §11.4 / [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §9.1 bullet 2 — two or more compositions of equal canonical cost cover the same Request constituent set (path ambiguity, e.g. billing-vs-shipping). `candidates: Vec<DataKindName>` cites every conflicting composition (explicit or implicit, with `ImplicitId` for the latter). |
| `NoCompositionPath { from, to }` | `PLAN_E_0501` | [Field-first-resolution](field-first-resolution.md) §11.4 — no composition (explicit or implicit) covers the Request constituent set. |
| `CompositionDepthExceeded { from_kinds, max_depth }` | `PLAN_E_0502` | [Field-first-resolution](field-first-resolution.md) §11.4 / [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §9.1 bullet 4 — required walk exceeds `MAX_IMPLICIT_COMPOSITION_DEPTH`; no implicit Joinset enumerated for the constituent set. Author declares an explicit Joinset to escape. |
| `AmbiguousCompositionReference { name, candidates }` | `PLAN_E_0505` | [Composed-surface-contents](composed-surface-contents.md) §6.2 — Request uses bare name on a composed surface with multiple qualifications. `candidates: Vec<UnifiedName>` carries the valid qualified forms. |
| `CompositionAggregationConflict { name, aggregations }` | `PLAN_E_0506` | [Composed-surface-contents](composed-surface-contents.md) §6.4 — left in place for backward-compat; in v1 this fires only when a Request's selected fields cross composition boundaries in a way the eager materialization could not anticipate (rare). Most aggregation conflicts surface at compile per `COMP_E_0405`. |
| `SemanticsNotOnSurface { name, surface }` | `PLAN_E_0507` | [Field-first-resolution](field-first-resolution.md) §11.6 — Request's `from:` is set but selected name is not on the resolved surface. |
| `UnknownSemantics { name }` | `PLAN_E_0508` | [Field-first-resolution](field-first-resolution.md) §11.2 — `Request.select` references a `SemanticsName` not in the SemanticManifest. |

`PLAN_E_0504 CompositionChainingForbidden` is **retired (2026-04-29)**: the unified Joinset model walks transparently through composed surfaces ([implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §9.1 bullet 7); the chaining prohibition no longer applies. The code is reserved for forward-compat (no MINOR re-allocation).

`PLAN_E_0503 CrossCompositionForbidden` is **retired (2026-05-12)**: the authoring-layer `Directionality` enum is removed; every `Relationship` is bidirectional by construction. Code reserved for forward-compat (no MINOR re-allocation).

The `candidates` field on `PLAN_E_0505` carries `Vec<UnifiedName>` of the valid qualified forms; diagnostic rendering includes one `ContextLine` per candidate (per `30 §5.3`) with "use this form" suggestions (per open `Q-COMP-014`, ratified yes).

### 14.4 Advisory (warning) additions

<!-- cross-cut: none -->

| Variant | Code | When |
|---|---|---|
| `RelationshipCardinalityKeyMismatch { relationship_id, declared_cardinality, inferred_uniqueness }` | `PLAN_W_0503` | [Relationship-edge-model](relationship-edge-model.md) §3.2 — `Cardinality` declared inconsistent with `Key::Primary` / `Key::Unique` on the key sides. **Structural** advisory — the model itself is internally inconsistent. Kept in v1. |
| `CompositionSharedDimensionDescription { composition_name, name, descriptions }` | `COMP_W_0401` | [Composed-surface-contents](composed-surface-contents.md) §6.3 — `Shared` promotion succeeded but constituents' `description` fields differ. **Structural** advisory — kept in v1. |

**Retired advisories (2026-04-29).** Two intent-level fanout advisories are removed in v1:

- `PLAN_W_0501 FanoutAdvisory` — flagged `OneToMany` / `ManyToOne` walks with `SemiAdditive` / `NonAdditive` measures on the fanout side. Removed because fanout is the natural consequence of the author's declared `Cardinality` and `Additivity`; v1 trusts those declarations rather than second-guessing intent. Authors who need fanout-detection in their workflow can author a separate audit query.
- `PLAN_W_0502 ManyToManyFanoutAdvisory` — fired on every `ManyToMany` walk. Removed for the same reason; junction-table modeling is a documentation recommendation, not an enforcement surface in v1.

Both codes (`0501`, `0502`) are reserved (no MINOR re-allocation) so a future v2 with telemetry can re-introduce them under `strict` mode if warranted.

Advisories that remain do NOT abort the pipeline; they are collected as `Diagnostic` entries alongside the produced plan / manifest. Consumers (CLI, IDE) render them.

### 14.5 Code range summary

```text
COMP_E_04xx     validate + compile composition errors (§12, §13, see implicit-enumeration-and-materialization)
                in v1: 0401–0409, 0410–0413, 0414
                reserved (forward-compat): 0415–0499
PLAN_E_05xx     plan-time composition errors (see field-first-resolution)
                in v1: 0500–0503, 0505–0508
                retired:  0504 (CompositionChainingForbidden)
                reserved (forward-compat): 0509–0599
PLAN_W_05xx     plan-time composition advisories (see relationship-edge-model §3.2, §12.4)
                in v1: 0503
                retired in v1 (reserved for future): 0501, 0502
                reserved (forward-compat): 0504–0599
COMP_W_04xx     compile-time composition advisories (see composed-surface-contents §6.3)
                in v1: 0401
                reserved (forward-compat): 0402–0499
```

Per `30 §5.1`, there is no central error-code allocation table; identification is by typed-kind variant. `16` allocates the composition-specific code ranges with headroom for future additions.

## 15. Interaction with Other Documents

`16`'s ratifications feed and consume several neighbours:

### 15.1 `19` — path signatures

`19 §3.4.2`'s `RelationshipGraph` is the shared infrastructure both documents consume at compile: `19 §3.4` for expression cross-kind resolution, [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10.4 / §10.5 for implicit-composition enumeration. `19 §3.4.5`'s `PathSignature` (`Vec<EntityId>`) is subset-consistent with [composed-surface-type](composed-surface-type.md)'s `traversed_paths` on a composed surface — for every `PathSignature` entry inside an expression on a composed Request, the path is covered by the composition's `traversed_paths`.

Plan-time, `19` consumers and [field-first-resolution](field-first-resolution.md) both read pre-built indices from `SemanticGraph` (built once at planner setup from manifest primitives per C9.2 / C9.6) — no graph traversal at plan.

[Relationship-edge-model](relationship-edge-model.md) ratifies what a `Relationship` **is**; `19 §3.4.5`'s `PathSignature` `Vec<EntityId>` is meaningful against that ratification. Changes to `Relationship`'s shape in `16` propagate to `19`'s path semantics.

### 15.2 `15` — coverage extension

`15 §6.4` explicitly hands off composition-level coverage to `16`; [composed-surface-contents](composed-surface-contents.md) §8.3 consumes `15 §6`'s per-Binding `Coverage` as the fold input. The two documents form a cascade: `15` → Binding-level coverage; `16` → composition-level coverage. Authors of nested compositions (e.g. `Joinset` over `Unionset`) trace field provenance through both.

`15 §2.2`'s binding `EntityId` appears in `16`'s planner algorithm as the lookup key into per-kind bindings during implicit-composition synthesis.

### 15.3 `17` — temporal-shape-gated `AsOf` joins

[Relationship-edge-model](relationship-edge-model.md) §4.3 defers `JoinType::AsOf` to `17`. `17`'s ratified `TemporalShape` variants (`Timeseries`, `Events`, `Snapshot`, `SCD`) gate the introduction: `AsOf` requires both constituents to declare compatible temporal axes. Future `16` revisions adding `AsOf` will forward-reference `17`'s shape-compatibility rules. Until then, `JoinType::{Inner, Left, Right, Full}` exhaust the supported joins.

`17`'s `Additivity × TemporalShape` interaction also drives [relationship-edge-model](relationship-edge-model.md) §3.3.2's fanout-safe-rewrite triggering: the `Measure`'s `Additivity` per `11 §7` plus the `TemporalShape` per `17` determines whether pre-join aggregation preserves correctness.

### 15.4 `20-25` — per-`DataKind` strategies

- `20` (`data-kinds/20_taxonomy.md` — DataKind taxonomy / shared invariants) consumes `Cardinality` for fanout decisions and emits `PLAN_W_05xx` advisories from per-variant strategies.
- `21` (`data-kinds/21_dataset.md` — Simple / Dataset) is the leaf participant in any composition; [composed-surface-contents](composed-surface-contents.md)'s `FieldProvenance::Native` for a single-kind composition targets a `21`-owned `SemanticInterface`.
- `22` (`data-kinds/22_grainset.md` — Grainset) consumes [composed-surface-contents](composed-surface-contents.md)'s `CompositionCoverage` to select effective routing units; for partial-coverage Requests across grains, `22 §3.4` builds a `ComposedSemanticInterface` per [composed-surface-type](composed-surface-type.md) and emits a LEFT OUTER JOIN tree on shared `Key`s per `18 §2.5`. Same-grain pre-merge inside a Grainset (per `22 §3.3`) produces an implicit Unionset shaped per `23 §3.2`; that Unionset's `SemanticInterface` is bare (NOT a `ComposedSemanticInterface`).
- `23` (`data-kinds/23_unionset.md` — Unionset) details vertical composition. Post-thirteenth-pass cascade rebase (2026-05-03): Unionset uses its own bare `SemanticInterface` and per-child Coverage, NOT a `ComposedSemanticInterface`; the `CompositionKind::Unionset` variant in [composed-surface-type](composed-surface-type.md) §5.3 was retired accordingly. Plan emission detail in `23 §4` (parked in `_drafts/34_unionset_strategy.md`).
- `24` (`data-kinds/24_joinset.md` — Joinset) details the full Joinset authoring and planning surface; [joinset-model](joinset-model.md) refines the type-level contract; [composed-surface-type](composed-surface-type.md)'s `ComposedSemanticInterface` carries the per-hop relationship-walk shape via `traversed_paths`.
- `25` (`data-kinds/25_applicability_matrix.md` — Applicability Matrix) cross-tabulates which `DataKind` consumes which composition rules from `16`; per-variant cells should be re-checked when `16` ratifies anything new at the top.

### 15.5 `32` / `33` / `34` / `35` — surface / artefact / planner / IR

- `32` (YAML surface) ratifies the author-facing syntax for `relationships:` and `joinsets:` (including `cardinality:`, `integrity:`, `optional:`, `cross_filter:`) and the defaults matrix from `18 §2.7`.
- `33` (SemanticManifest) persists **primitives only** under the lightweight-manifest posture (cascade 2026-05-28; clauses C7, C7.4, C8.2, C9.2, C10.2): `Relationship` at root scope (`SemanticManifest.relationships`) and inline as Joinset shadow (`Joinset.scope_local_relationships` per `33 §6.7`); `Joinset` as `DataKindVariant::Joinset` carrying `anchor` + `members` + `hops` + per-hop `hop_coverage` + `path_origin` (per C7); per-DataKind `coverage: SemanticBitmask` and the global `semantics: SemanticBitmap` registry. `33` does NOT persist `ResolvedComplexDataKind`, [composed-surface-type](composed-surface-type.md)'s `ComposedSemanticInterface`, or a pre-built `RelationshipGraph` adjacency index — those are reconstructed by `SemanticGraph` at build time from the primitives (C7.4 + C8.2 + C9.6).
- `34` (Planner) consumes [field-first-resolution](field-first-resolution.md) as its entry point when `Request.from = None`. `34` ratifies the `plan` call-graph and the `SemanticGraph` build step that runs [implicit-enumeration-and-materialization](implicit-enumeration-and-materialization.md) §10's enumeration.
- `35` (IR) carries `JoinType`, `Cardinality`, `EntityId` on `PlanNode::Join` per [relationship-edge-model](relationship-edge-model.md) §4.4.

## Testing Strategy

- **Invariants:** I12 (fail-fast — all composition errors abort the containing stage), I10 (non-exhaustive error enums).
- **Well-formedness checks:** Tests that duplicate `Relationship`s fire `COMP_E_0410`, that `KeyPair` type mismatch fires `COMP_E_0401`, that self-reference fires `COMP_E_0402`, that empty `keys` fires `COMP_E_0413`, that nested-kind endpoints fire `COMP_E_0412`, that undeclared `DataKindRef` endpoints fire `COMP_E_0411`.
- **Error code stability:** Tests that all `COMP_E_04xx` / `PLAN_E_05xx` / `PLAN_W_05xx` codes match the ratified allocation table (§14.5), that retired codes (`COMP_E_0406`, `PLAN_E_0503`, `PLAN_E_0504`, `PLAN_W_0501`, `PLAN_W_0502`) are reserved and not re-allocated.
- **Diagnostic payloads:** Tests that `COMP_E_0414` includes `candidate_differentiators`, that `PLAN_E_0500` includes all ambiguous `candidates`, that `PLAN_E_0505` includes all valid `UnifiedName` qualifications, that `PLAN_W_0503` cites the conflicting `Cardinality` and `Key` uniqueness.
- **Advisory vs error:** Tests that `PLAN_W_0503` and `COMP_W_0401` do NOT abort the pipeline (collected as `Diagnostic` entries), that all `COMP_E_*` / `PLAN_E_*` DO abort.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
