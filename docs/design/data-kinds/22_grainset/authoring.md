---
id: 22-grainset-authoring
status: ratified
version: 2.1
impl: drift
---


# 22. Grainset (`ComplexDataKind`)

> **Reconciliation (post-thirteenth-pass cascade rebase, 2026-05-03).** The v1 authoring-layer canonical shape for `Grainset` is ratified across:
>
> - [`20_taxonomy.md §2`](../20_taxonomy.md) — sealed trait hierarchy (`DataKind`, `ComplexDataKind`, `PublicDataKind`, `NestedDataKind`); `Grainset` implements `ComplexDataKind + PublicDataKind`, `NestedGrainset` implements `ComplexDataKind + NestedDataKind`.
> - [`../apis/32_semstrait_model.md §3.2`](../../apis/32_semstrait_model.md) — `GrainsetBody { base: DataKindBase<ComplexExtras>, datasets, unionsets, joinsets }`. Children are inlined `Nested*` structs (no `DataKindRef` mechanism for cross-Complex child references in v1).
> - [`../apis/32_semstrait_model.md §3.3`](../../apis/32_semstrait_model.md) — `Grainset` and `NestedGrainset` concrete forms.
> - [`../apis/32_semstrait_model.md §4`](../../apis/32_semstrait_model.md) — `ComplexExtras { temporal: Option<TemporalShape> }`; Grainset authors only `temporal:` in `extras` (no `catalog` / `storage` / `semantic_mapping` per R-6 / SR-5; no `grain` per SR-E-7).
> - [`../foundations/18_entities.md`](../../foundations/18_entities.md) — canonical entity types consumed by `GrainsetBody`'s SemanticInterface (Dimensions / Measures / Metrics / **Keys** / Filters); SR-E-7 (no `grain` on Complex), SR-E-8 (Grainset child MUST author `extras.temporal.grain:` explicitly), Keys per `18 §2.5` (the cross-grain JOIN surface).
> - [`26_nesting_matrix.md`](../26_nesting_matrix.md) — nesting matrix; R1 (leaves don't nest), R2 (no same-variant self-nesting; type-level field absence in `GrainsetBody`), R3 (`ComplexDataKind` requires ≥ 2 children).
> - [`../foundations/16_composition.md §5`](../../foundations/16_composition.md) — `ComposedSemanticInterface` (broadened in this rebase to cover both Joinset per-hop and Grainset cross-grain composition).
>
> `22` retains authority for:
>
> - The `Grainset` authoring surface — explicit (authored) vs. implicit (compile-derived) fields, with cross-references to the type-level shape in `32 §3` / `32 §4`.
> - The V1 strict TemporalShape kind equivalence rule across children (mixed shapes is a hard compile error).
> - The "≥ 2 unique grains" requirement (stronger than R3's ≥ 2 children).
> - The cascade-up rule for `kind` only — grain stays per-child (per SR-E-7 / SR-E-8).
> - The same-grain pre-merge mechanism (implicit Unionset per `23 §4`).
> - The cross-grain LEFT OUTER JOIN composition mechanism (driver = most-covering effective routing unit; attached units equi-joined on shared Keys per `18 §2.5`).
> - Per-child Coverage inference (no authored override in v1).
> - Plan-time observable invariants of Grainset resolution.
> - Variant-specific Precondition surfaces — `VALID_E_2200`–`2299`, `COMP_E_2200`–`2299`, `PLAN_E_2200`–`2299`.
>
> **Algorithm body relocated.** The `GrainsetStrategy` algorithm body (effective-routing-unit assembly, single-unit delegation, cross-grain JOIN-tree construction, rollup wrapper emission) lives in [`../apis/34_semstrait_planner.md §<GrainsetStrategy>`](../../apis/34_semstrait_planner.md) (forthcoming). This rebase parked the body in [`../_drafts/34_grainset_strategy.md`](../../_drafts/34_grainset_strategy.md) pending the `34` drafting pass.
>
> **Concepts retired or repositioned in this rebase.**
>
> - `RollupPolicy` (`ShapeDefault` / `PinOnly` / `PreferFinest`) — implemented internally as a planner knob, NOT exposed in V1 authoring grammar (per G-4). Default behavior follows shape rules in `17`; future fine-tuning hooks may surface in `34`.
> - The pre-cascade `MixedShapeAdvisoryChildren` warning (`PLAN_W_2202` in pre-rebase `22`) — retired. Mixed shapes are now a hard compile error per V1 strict equivalence (§5.2).
> - `DataKindRef`-based child references (`GrainsetChild.constituent: DataKindRef`) — retired. Children are inlined `Nested*` structs per `32 §3.2`.
> - YAML `binding:` / `column_mapping:` wrappers (pre-thirteenth-pass spelling) — dissolved into `extras.storage:` / `extras.semantic_mapping:` per `32 §4`.
> - `CompositionKind::Grainset` discriminator — survives in `16 §5.3`'s shrunken `CompositionKind { Grainset, Joinset }` enum, but with broadened semantics (Grainset's `ComposedSemanticInterface` carries the cross-grain JOIN-tree shape, distinct from Joinset's relationship-walk shape).

## 1. Purpose and Scope

### 1.1 What `22` ratifies

`22` is the per-variant chapter for `Grainset` (one of three `ComplexDataKind` variants in v1, alongside `Unionset` (`23`) and `Joinset` (`24`)). It owns five things: (a) the **authoring surface** — explicit vs. implicit fields the author / consumer sees on the type — with cross-references to the type-level shape in `32 §3` (forms + body) and `32 §4` (`ComplexExtras`); (b) the **temporal-shape rules** specific to Grainset — kind equivalence across children, ≥ 2 unique grains, cascade-up of kind only; (c) the **same-grain pre-merge mechanism** — implicit Unionset assembly at compile per `23 §4`; (d) the **cross-grain composition mechanism** — when no single effective routing unit covers a Request, the planner emits a LEFT OUTER JOIN tree on shared Keys per `18 §2.5`, mediated by a `ComposedSemanticInterface` per `16 §5`; (e) **Grainset-specific Preconditions** in the per-variant code bands `VALID_E_2200`–`2299`, `COMP_E_2200`–`2299`, `PLAN_E_2200`–`2299`.

### 1.2 What `22` does NOT ratify

The sealed trait hierarchy and shared `DataKind` invariants live in `20`. The Rust struct / YAML shape lives in `32 §3` / `32 §4`. Child-DataKind structural rules (R1 / R2 / R3, addressing scheme) live in `26`. SR-E-7 / SR-E-8 (grain-on-complex forbidden; grainset-child-grain-required) live in `18 §3` / `18 §11`. Per-child Binding mechanics live in `15`. The `ComposedSemanticInterface` Rust shape, `UnifiedSemantics` merge, `FieldProvenance` roster, and `CompositionCoverage` fold all live in `16 §5`–`§8`. The same-grain pre-merge implicit Unionset's algorithm body lives in `23 §4` + `_drafts/34_unionset_strategy.md`. The `GrainsetStrategy` algorithm body lives in `34 §<GrainsetStrategy>` (forthcoming; parked in `_drafts/34_grainset_strategy.md`). `TemporalShape` enum variants, SCD subtype catalog, shape × grain rollup matrix, `RollupPolicy` semantics, per-shape pin / anchor mechanics live in `17`. `PlanNode` shapes live in `35`. Engine rendering lives in `36`. Peer-vocabulary references (Cube.js `preAggregations` with `granularity`, dbt MetricFlow `time_spine`, OLAP aggregate-awareness / Kimball) live in `00_overview.md` only.

### 1.3 Guardrails — how `22` upholds `00 §9` invariants

| Invariant | Where `22` keeps it |
|---|---|
| **I1** — no raw SQL in canonical layer | The emitted plan tree carries `PlanNode::{Project, Filter, Agg, Union, Join}` with `PhysicalExpr` payloads; no SQL-shaped strings. Adapter rendering (`36`) is the only SQL site. |
| **I2** — physical types via adapters only | Cross-grain JOIN type-reconciliation uses canonical `DataType` per `13 §2`; no engine types. |
| **I3** — no engine branching | Zero engine-identity checks in `22`. Every decision reads either resolved manifest indices or per-unit Coverage (compile-derived). |
| **I4** — SemanticManifest determinism | Effective-routing-unit ordering follows **name order** across the `datasets:` / `unionsets:` / `joinsets:` collections (id-keyed, name-ordered projection per `32 §7`, post id-first rework). Same-grain pre-merge produces implicit Unionsets in name order per `23 §3.1`. Driver selection is most-covering with **name-order** tie-break; attached ordering follows name order (G-2c). |
| **I5** — compile-time resolution | Per-unit Coverage is folded once at compile (§3.2); cross-grain JOIN-tree shape is built at compile from Keys + Coverage; per-Request routing is O(\|units\| × \|Semantics\|) lookups. |
| **I6** — synchronous hot path | No I/O at any stage of Grainset resolution. |
| **I8** — planner-complete SemanticManifest | `ResolvedGrainset`'s **manifest contract** persists `GrainsetLevel { grain, routing_unit, level_coverage }` per level (`levels.len() >= 2`, distinct grains, coarsest-first per `33 §6.6`). Cross-grain JOIN-tree shape, per-pair JOIN-key index, and the `ComposedSemanticInterface` are **deferred to graph build** (see Phase 3 amendment below); planner-completeness is preserved because `SemanticGraph` synthesises them deterministically from manifest primitives (per-level coverage + Keys carried on each level's interface) at build time. |

> **Phase 3 amendment (2026-05-28; cascade from C8.2).** Earlier drafts of `ResolvedGrainset`'s manifest contract included the cross-grain JOIN-tree shape, the per-pair JOIN-key index, and the resolved `ComposedSemanticInterface` as persisted fields. Per ratified clause C8.2 (manifest-pass `RATIFICATION_LOG.md`), these are dropped from the manifest contract and re-classified as **graph-build outputs**. The manifest now persists only the per-level coverage primitives — `GrainsetLevel { grain, routing_unit: RoutingUnitRef, level_coverage: SemanticBitmask }` — plus the universal top-level `coverage: SemanticBitmask` per CCK.1. `RoutingUnitRef` is `Inline(NestedDataKind)` for single same-grain children and `Synthesized(EntityId)` for ≥2 same-grain children (referencing a top-level implicit Unionset per C9.4). Cross-grain JOIN-tree synthesis at graph build time uses (a) per-level `level_coverage` for first-cut path search and (b) Keys on each level's interface (per `18 §2.5`) for the equi-join skeleton. The cross-references for the manifest-resident shape are [`../apis/33_semstrait_manifest.md §6.6`](../../apis/33_semstrait_manifest.md) (`DataKindVariant::Grainset`) and `33 §10` (CX1 load-time integrity). The plan-time observable behaviour in `22 §4` is unchanged — `GrainsetStrategy::resolve` still receives a fully-resolved per-level routing-unit list; the difference is that the JOIN-tree skeleton is synthesised into the graph during build, not loaded from the manifest.
| **I10** — non-exhaustive public sum types | `Grainset` and `NestedGrainset` are `#[non_exhaustive]` per `32 §3.3`. The internal `RollupPolicy` (per G-4) is `#[non_exhaustive]` even though not authored. |
| **I12** — first-class Diagnostics | Every error code in §§7–9 is stable and carries a `Diagnostic.location`. |

## 2. Authoring-Time Surface

### 2.1 Explicit vs. implicit fields

The `Grainset` author touches the following surface. Type-level shape lives in `32 §3.3` (`Grainset` / `NestedGrainset` concrete forms), `32 §3.2` (`GrainsetBody`), and `32 §4` (`ComplexExtras`); `22` enumerates obligations and cross-references foundations for substance.

| # | Axis | Explicit (authored) | Implicit (compile-derived) | Cite |
|---|---|---|---|---|
| A | Identity | `body.base.name` (Public form: globally unique across top-level `data_kinds:` per SR-3; Nested form: scoped to parent's nested-kind scope per `26 §4`). | `id: EntityId` (model `id` for explicit kinds; deterministic content-derived id for same-grain implicit Unionsets created during pre-merge, per `33 §9.1`). | `32 §3.1`, `32 §3.3`, `26 §4`, `23 §3.3` |
| B | Description | `description: Option<String>` on `Grainset` (Public form only; absent on `NestedGrainset` per `26 §3` "Nested-form structural-only" rule). | None. | `32 §3.3` |
| C | Semantic interface | `semantic_interface: SemanticInterface` on `Grainset` (Public form only; absent on `NestedGrainset`). Authors Dimensions, Measures, Metrics, **Keys** (the cross-grain JOIN surface — see §3.4), Filters per `18 §1`–`§2`. Components flatten in YAML per `#[serde(flatten)]`. | Per-Semantics inferred-type fields; typed-leaf substitution for `- ref: <name>` entries; Metric expansion; Computed-Dimension expression resolution per `19 §3`. | `32 §3.3`, `18 §1`–`§2`, `19 §3` |
| D | AI context | `ai_context: Option<AiContext>` on `Grainset` (Public form only). | None. | `32 §3.3`, `18 §6` |
| E | Children | `body.{datasets, grainsets, joinsets}: BTreeMap<EntityId, Nested*>` — children are **inlined** `NestedDataset` / `NestedUnionset` / `NestedJoinset` structs per `26`'s nesting matrix (Grainset admits all three Complex variants except itself per R2). Each child MUST author `extras.temporal.grain:` explicitly per SR-E-8. | Per-child Coverage inference (§3.2); same-grain pre-merge into implicit Unionsets (§3.3); cross-grain JOIN-tree shape via `ComposedSemanticInterface` (§3.4). | `32 §3.2`, `26 §1`, `26 §2`, `18 §3.4` (SR-E-8) |
| F | TemporalShape | `extras.temporal: Option<TemporalShape>` on `ComplexExtras` — carries only `kind` (incl. SCD subtype). Per the V1 strict equivalence rule (§5.2), the Grainset's effective `kind` is the cascade-up of children's kinds (which must all be equivalent). `grain` is type-level forbidden on `ComplexExtras` per SR-E-7. | Cascade-up from children's `extras.temporal.kind`. The Grainset's grain axis is the **set** of children's distinct grain values (≥ 2 per §5.2). | `32 §4`, `17`, `18 §3.3` (SR-E-7), `26 §3.1` |

**Public vs. Nested form distinction.** Rows B, C, D apply only to the Public form (`Grainset`). The Nested form (`NestedGrainset`) carries only `body` (which contains `base.name`, `extras: ComplexExtras`, and the nested child arrays); fields B/C/D are structurally absent per `26 §3`. The `name` at nested scope is the structural anchor in the parent's nested-kind scope (`26 §4` addressing).

**Internal-only fields (NOT authored).** `RollupPolicy { ShapeDefault, PinOnly, PreferFinest }` is implemented internally as a planner knob (per G-4 ratification, 2026-05-03). YAML carries no `rollup_policy:` / `rollup_override:` field. The default policy follows shape rules in `17`; future fine-tuning hooks may surface in `34`.

### 2.2 Nesting interaction

A `Grainset`:

- **MAY** appear at Root scope as a top-level public DataKind (per `32 §2.1`'s `grainsets:` plural tag).
- **MAY** be nested inline as a `NestedGrainset` under a `Unionset` or `Joinset` (per `26 §1`'s matrix). Nested form is structural-only (no `description` / `semantic_interface` / `ai_context`); the nested `name` is the structural anchor.
- **MUST NOT** be nested under another `Grainset` (R2; type-level — no `grainsets:` field on `GrainsetBody` per `26 §2.2`).
- **MUST** carry **≥ 2 children** across `datasets:` / `unionsets:` / `joinsets:` combined (R3 per `26 §2.3`; structural-rule diagnostic owned by `26` / `32`).
- **MUST** carry **≥ 2 unique `extras.temporal.grain:` values** across the child set (V1 strict; §5.2). Cardinality check is at validate stage; same-grain duplicates are legal individually (they pre-merge per §3.3) but the distinct-grain count must be ≥ 2.

### 2.3 Authoring example — full SemanticInterface surface

A worked YAML showing every authoring slot a `Grainset` exposes (focus is **completeness of the surface**, not plan emission — see §10 for the plan-shape worked example):

```yaml
grainsets:
  - name: paid_media_rollups
    description: "Paid-media spend across grains; cross-grain composition supported."
    dimensions:
      - name: report_date
        data_type: Timestamp
      - name: campaign_id
        data_type: String
    measures:
      - name: cost
        agg: sum
        expr: cost_cents
        data_type: Long
      - name: clicks
        agg: sum
        expr: click_count
        data_type: Long
    metrics:
      - name: cpc
        expr: cost / clicks
        data_type: Decimal
    keys:
      - name: campaign_id
        kind: foreign
    filters:
      - name: paid_only
        expr: cost > 0
    extras:
      temporal:
        snapshot:
          snapshotted_at: report_date
        # NOTE: no `grain:` here — SR-E-7 forbids grain at any Complex level;
        #       grain lives on each child's LeafExtras only (SR-E-8).
```

The parent interface above declares the shared surface; the `datasets:` block below lists the two children that supply it at different grains:

```yaml
    datasets:
      - name: paid_media_daily
        extras:
          catalog: marketing_warehouse
          storage: { format: Parquet, paths: ["s3://b/daily/*.parquet"] }
          semantic_mapping:
            report_date: snapshot_date
            campaign_id: campaign_pk
            cost_cents: daily_cost
            click_count: daily_clicks
          temporal:
            snapshot:
              snapshotted_at: snapshot_date
              grain: Day
      - name: paid_media_monthly
        extras:
          catalog: marketing_warehouse
          storage: { format: Parquet, paths: ["s3://b/monthly/*.parquet"] }
          semantic_mapping:
            report_date: month_start
            campaign_id: campaign_pk
            cost_cents: monthly_cost
            # note: no `click_count` mapping — child does not provide `clicks`
          temporal:
            snapshot:
              snapshotted_at: month_start
              grain: Month
```

Notes on this example:

- `dimensions:` / `measures:` / `metrics:` / `keys:` / `filters:` are **flat** at the Grainset level per `#[serde(flatten)]` on the SemanticInterface (per `32 §3.3`).
- `keys:` declares `campaign_id` as a foreign Key — this is the cross-grain JOIN surface. When a Request needs `clicks` (only on the daily child) AND comes at grain `Month` (only directly served by the monthly child), the planner builds a LEFT OUTER JOIN tree using `campaign_id` as the equi-join key. See §3.4 + §10.
- `extras.temporal:` on the Grainset carries only `kind` (here `snapshot:`); `grain:` is type-level forbidden on `ComplexExtras` (SR-E-7).
- Each child authors its own `extras.temporal.grain:` explicitly (SR-E-8) — `Day` for daily, `Month` for monthly. Two distinct grains; satisfies the "≥ 2 unique grains" rule (§5.2).
- All children share `kind: snapshot` (V1 strict equivalence per §5.2). A child with `kind: events` would be a hard compile error.
- `paid_media_monthly` does NOT map `click_count` — its Coverage of `clicks` is `NullFill`. Routing for Requests touching `clicks` skips the monthly child unless cross-grain JOIN composition kicks in (§3.4).

## 3. Per-Child Contract

### 3.1 Children list shape and ordering

The `GrainsetBody` carries three child maps — `datasets: BTreeMap<EntityId, NestedDataset>`, `unionsets: BTreeMap<EntityId, NestedUnionset>`, `joinsets: BTreeMap<EntityId, NestedJoinset>` (per `32 §3.2`). The **canonical child sequence** for plan emission is:

1. all `datasets:` entries in name order, then
2. all `unionsets:` entries in name order, then
3. all `joinsets:` entries in name order.

This order is stable (per `32 §7` ordering table — collections are id-keyed but projected name-ordered; intra-variant basis changed from author order to name order with the id-first rework) and is the order:

- Same-grain children are merged (name order within the same grain determines their order inside the implicit Unionset's `inputs` per `23 §3.1`).
- Driver selection's tie-break runs (most-covering first; name-order tie-break per G-2b).
- Attached units are added in cross-grain JOIN-tree construction (name order per G-2c — deterministic).

R2 (no same-variant self-nesting) is type-level absence per `32 §3.2` — a `grainsets:` field does not exist on `GrainsetBody`. R3 (≥ 2 children) is the validate-stage rule per `26 §2.3`. Nested-child name uniqueness within a single Grainset's combined child set is the Grainset author's responsibility; collisions surface via the `26 §4` addressing scheme as ambiguous addresses.

### 3.2 Per-child Coverage — v1 inference-only

V1 carries **no authored per-child Coverage override** (parallels `23 §3.2`'s ratification). Per-child Coverage is inferred at compile from each child's interface plus the child's Binding-level Coverage (Simple children) or the child's resolved interface (Complex children).

**Per-Semantics inference rule.** For each surface Semantics `s` on the Grainset's own `SemanticInterface`, and for each child `i`:

- If child `i`'s interface declares a Semantics with the same name `s` (or a `- ref: s` to a root-pool entry shared by the Grainset) AND the child can produce a value (per `15 §10` for Simple, per the child's own resolution for Complex), then child `i` **provides** `s`.
- Otherwise, child `i` **does not provide** `s` — its Coverage entry is `NullFill`.

**Coverage-completeness check.** Unlike Unionset, a Grainset surface Semantics that NO child provides is NOT auto-rejected: it could still be unreachable at runtime if no Request touches it. But because such a Semantics is unanswerable, it is a structural mistake — `COMP_E_2202 GrainsetCoverageIncomplete` fires at compile (§8). Authors fix by adding the Semantics to ≥ 1 child or removing it from the Grainset's surface.

## Testing Strategy

- Explicit vs implicit field round-trip; SR-E-7 (no grain on ComplexExtras) / SR-E-8 (child grain required); VALID_E_2202 GrainsetInsufficientUniqueGrains; ≥2 unique grains rule.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
