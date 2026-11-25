---
id: 23-unionset-authoring
status: ratified
version: 2.1
impl: drift
---


# 23. Unionset (`ComplexDataKind`)

> **Reconciliation (post-thirteenth-pass cascade, 2026-05-03).** The v1 authoring-layer canonical shape for `Unionset` is ratified across:
>
> - `[20_taxonomy.md §2](../20_taxonomy.md)` — sealed trait hierarchy (`DataKind`, `ComplexDataKind`, `PublicDataKind`, `NestedDataKind`); `Unionset` implements `ComplexDataKind + PublicDataKind`, `NestedUnionset` implements `ComplexDataKind + NestedDataKind`.
> - `[../apis/32_semstrait_model.md §3.2](../../apis/32_semstrait_model.md)` — `UnionsetBody { base: DataKindBase<ComplexExtras>, datasets, grainsets, joinsets, mode: UnionMode }`. Children are inlined `Nested`* structs (no `DataKindRef` mechanism for cross-Complex child references in v1).
> - `[../apis/32_semstrait_model.md §3.3](../../apis/32_semstrait_model.md)` — `Unionset` and `NestedUnionset` concrete forms.
> - `[../apis/32_semstrait_model.md §4](../../apis/32_semstrait_model.md)` — `ComplexExtras { temporal: Option<TemporalShape> }`; Unionset authors only `temporal:` in `extras` (no `catalog` / `storage` / `semantic_mapping`, per R-6 / SR-5).
> - `[../foundations/18_entities.md](../../foundations/18_entities.md)` — canonical entity types consumed by `UnionsetBody`'s SemanticInterface: Dimensions / Measures / Metrics / Keys / Filters; inline-vs-`ref` grammar for entity-level reuse.
> - `[26_nesting_matrix.md](../26_nesting_matrix.md)` — nesting matrix; R1 (leaves don't nest), R2 (no same-variant self-nesting; type-level field absence in `UnionsetBody`), R3 (`ComplexDataKind` requires ≥ 2 children).
>
> `23` retains authority for:
>
> - The `Unionset` authoring surface — explicit (authored) vs. implicit (compile-derived) fields, with cross-references to the type-level shape in `32 §3` / `32 §4`.
> - The variant-local `UnionMode { All, Unique }` enum (per `32 §3.2`'s deferral note).
> - Per-child Coverage inference (no authored override in v1).
> - The canonical fan-out mechanism shared with multi-source `Dataset` (per `21 §3.2`).
> - V1 strict TemporalShape equivalence rule (kind incl. SCD subtype + grain across all children).
> - Plan-time observable invariants of Unionset resolution.
> - Variant-specific Precondition surfaces — `VALID_E_2300`–`2399`, `COMP_E_2300`–`2399`, `PLAN_E_2300`–`2399`.
>
> **Algorithm body relocated.** The `UnionsetStrategy` algorithm body (per-branch wrapping, NULL-fill projection emission, column-type reconciliation, branch pruning, conditional final-aggregation predicate) lives in `[../apis/34_semstrait_planner.md §<UnionsetStrategy>](../../apis/34_semstrait_planner.md)` (forthcoming). This rebase parked the body in `[../_drafts/34_unionset_strategy.md](../../_drafts/34_unionset_strategy.md)` pending the `34` drafting pass.
>
> **Concepts retired in this rebase.** `CompositionKind`, `ComposedSemanticInterface` (as a Unionset-level concept; `ComposedSemanticInterface` survives as a Joinset-only per-hop join-path-search artifact per `24`), `ChildCoverageOverride { provides }`, the YAML `coverage:` block on child entries, `DataKindRef`-based child references — all were pre-cascade artifacts not authored by the user. They have no slot in the v1 authoring surface or in `23`'s contract.

## 1. Purpose and Scope

### 1.1 What `23` ratifies

`23` is the per-variant chapter for `Unionset` (one of three `ComplexDataKind` variants in v1, alongside `Grainset` (`22`) and `Joinset` (`24`)). It owns four things: (a) the **authoring surface** — explicit vs. implicit fields the author / consumer sees on the type — with cross-references to the type-level shape in `32 §3` (forms + body) and `32 §4` (`ComplexExtras`); (b) the variant-local `**UnionMode { All, Unique }`** enum (per `32 §3.2`'s deferral); (c) the **plan-time observable invariants** of Unionset resolution — what an author or consumer can rely on from the emitted plan, *not* the algorithm body; (d) **Unionset-specific Preconditions** in the per-variant code bands `VALID_E_2300`–`2399` (validate), `COMP_E_2300`–`2399` (compile), `PLAN_E_2300`–`2399` (plan).

### 1.2 What `23` does NOT ratify

The sealed trait hierarchy and shared `DataKind` invariants live in `20`. The Rust struct / YAML shape lives in `32 §3` / `32 §4`. Child-DataKind structural rules (R1 / R2 / R3, addressing scheme) live in `26`. Per-child Binding mechanics live in `15`. The `UnionsetStrategy` algorithm body lives in `34 §<UnionsetStrategy>` (forthcoming; parked in `_drafts/34_unionset_strategy.md`). `TemporalShape` enum variants, SCD subtype catalog, shape × grain rollup matrix live in `17`. `PlanNode` shapes live in `35`. Engine rendering lives in `36`. Peer-vocabulary references (Cube.js `unionAll`, dbt MetricFlow, LookML, DataFusion `LogicalPlan::Union`, Substrait `SetRel`) live in `00_overview.md` only.

### 1.3 Guardrails — how `23` upholds `00 §9` invariants


| Invariant                                  | Where `23` keeps it                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **I1** — no raw SQL in canonical layer     | The emitted plan tree carries `PlanNode::Union` (with the `UnionMode` flag projected), per-branch `PlanNode::Project` (carrying `PhysicalExpr::Cast(Null, T)` for NullFilled columns), and `PlanNode::Agg` for terminal re-aggregation; no SQL-shaped strings. Adapter rendering (`36`) is the only SQL site.                                  |
| **I2** — physical types via adapters only  | Column-type reconciliation (§4.3) uses canonical `DataType` per `13 §2`; LUB widening uses `13 §7`'s cast matrix. No engine types.                                                                                                                                                                                                             |
| **I3** — no engine branching               | Zero engine-identity checks in `23`. Every decision reads either resolved manifest indices or per-child Coverage (compile-derived).                                                                                                                                                                                                            |
| **I4** — SemanticManifest determinism      | Child ordering follows **name order** across the `datasets:` / `grainsets:` / `joinsets:` collections (id-keyed, name-ordered projection per `32 §7`, post id-first rework). NULL-fill projections walk the Unionset's own `SemanticInterface` in name order. The implicit-Unionset (multi-source `Dataset`) per-source order follows `15 §3.6`'s lexical resolution. |
| **I5** — compile-time resolution           | Per-child Coverage is folded once at compile (§3.2); column-type LUB is computed once at compile (§4.3); per-branch projection rosters are pre-built. Plan-time is O(1) lookups per Semantics.                                                                                                                                                 |
| **I6** — synchronous hot path              | No I/O at any stage of Unionset resolution.                                                                                                                                                                                                                                                                                                    |
| **I8** — planner-complete SemanticManifest | `Unionset` resolution touches only the resolved manifest; no YAML, no catalog. Implicit-Unionset hash-ids (per `33 §<implicit-unionset-id>`, forthcoming) are compile-assigned and stable.                                                                                                                                                     |
| **I10** — non-exhaustive public sum types  | `UnionMode` is `#[non_exhaustive]` per `32 §3.2`; `Unionset` and `NestedUnionset` are `#[non_exhaustive]` per `32 §3.3`. Adding `UnionMode::Intersect` or `UnionMode::ByName` is MINOR per `30 §6.3`.                                                                                                                                          |
| **I12** — first-class Diagnostics          | Every error code in §§7–9 is stable and carries a `Diagnostic.location`.                                                                                                                                                                                                                                                                       |


## 2. Authoring-Time Surface

### 2.1 Explicit vs. implicit fields

The `Unionset` author touches the following surface. Type-level shape lives in `32 §3.3` (`Unionset` / `NestedUnionset` concrete forms), `32 §3.2` (`UnionsetBody`), and `32 §4` (`ComplexExtras`); `23` enumerates obligations and cross-references foundations for substance.


| #   | Axis               | Explicit (authored)                                                                                                                                                                                                                                                                                                                                                                                        | Implicit (compile-derived)                                                                                                                                                                                                                                                                                          | Cite                                     |
| --- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| A   | Identity           | `body.base.name` (Public form: globally unique across top-level `data_kinds:` per SR-3; Nested form: scoped to parent's nested-kind scope per `26 §4`).                                                                                                                                                                                                                                                    | `id: EntityId` for explicit Unionsets (the model `id` per `33 §6.1`); `id` for **implicit** Unionsets (a deterministic content-derived id over the parent `Dataset`'s structural identity + per-source list, per `33 §9.1`). Implicit Unionsets carry no `name` and live as **top-level** entries in `data_kinds: BTreeMap<EntityId, DataKind>` with `origin = DataKindOrigin::Implicit` (per C9.4 + C9.5); they are **not** carried inside any `NestedDataKind`. | `32 §3.1`, `32 §3.3`, `26 §4`, `21 §3.2` |
| B   | Description        | `description: Option<String>` on `Unionset` (Public form only; absent on `NestedUnionset` per `26 §3` "Nested-form structural-only" rule).                                                                                                                                                                                                                                                                 | None.                                                                                                                                                                                                                                                                                                               | `32 §3.3`                                |
| C   | Semantic interface | `semantic_interface: SemanticInterface` on `Unionset` (Public form only; absent on `NestedUnionset` per `26 §3`). Authors Dimensions, Measures, Metrics, Filters, Keys per `18 §1`–`§2`. Components flatten in YAML per `#[serde(flatten)]`.                                                                                                                                                               | Per-Semantics inferred-type fields, typed-leaf substitution for `- ref: <name>` entries, Metric expansion, Computed-Dimension expression resolution per `19 §3`.                                                                                                                                                       | `32 §3.3`, `18 §1`–`§2`, `19 §3`           |
| D   | AI context         | `ai_context: Option<AiContext>` on `Unionset` (Public form only).                                                                                                                                                                                                                                                                                                                                          | None.                                                                                                                                                                                                                                                                                                               | `32 §3.3`, `18 §6`                       |
| E   | Mode               | `body.mode: UnionMode` (default `All`; v1 roster `{All, Unique}` per `§4.1`).                                                                                                                                                                                                                                                                                                                              | None.                                                                                                                                                                                                                                                                                                               | `32 §3.2`, `§4.1`                        |
| F   | Children           | `body.{datasets, grainsets, joinsets}: BTreeMap<EntityId, Nested*>` — children are **inlined** `NestedDataset` / `NestedGrainset` / `NestedJoinset` structs per `26`'s nesting matrix (Unionset admits all three Complex variants except itself per R2). Cross-Complex `DataKindRef` references are not part of v1; entity-level `- ref:` for shared root-pool Dimensions / Measures / Metrics is unaffected (`18 §1.1`). | Per-child Coverage inference (§3.2); per-child resolved interface (Simple → `15 §10`; Complex → child's own resolution).                                                                                                                                                                                            | `32 §3.2`, `26 §1`, `26 §2`, `18 §1.1`   |
| G   | TemporalShape      | `extras.temporal: Option<TemporalShape>` on `ComplexExtras`. Per the V1 strict equivalence rule (§5.2), the Unionset's effective `temporal.kind` is the cascade-up of children's shapes (which must all be equivalent); explicit authoring at the Unionset level is permitted and must agree with children.                                                                                                | Cascade-up from children: when authoring is silent at the Unionset level, the effective shape is read from the (unanimous) children. `grain` is leaf-only (`LeafExtras` carries it; `ComplexExtras` does not — per `32 §4` table).                                                                                  | `32 §4`, `17`, `18 §3.3`, `26 §3.1`      |


**Public vs. Nested form distinction.** Rows B, C, D apply only to the Public form (`Unionset`). The Nested form (`NestedUnionset`) carries only `body` (which contains `base.name`, `extras: ComplexExtras`, `mode: UnionMode`, and the nested child arrays); fields B/C/D are structurally absent per `26 §3`. The `name` at nested scope is the structural anchor in the parent's nested-kind scope (`26 §4` addressing).

> **Phase 3 confirmation (2026-05-28; cascade from C6.5 + C9.4 + C9.5).** Row A's "implicit Unionsets carry a content-derived hash `data_kind_id`" claim is reaffirmed for the manifest-resident shape: implicit Unionsets (multi-source `Dataset` auto-synthesis per `21 §3.2`) live as **top-level entries** in `data_kinds: BTreeMap<EntityId, DataKind>` per [`../apis/33_semstrait_manifest.md §4.1`](../../apis/33_semstrait_manifest.md), tagged with `origin: DataKindOrigin::Implicit` per C9.5, and structurally identical to explicit `DataKindVariant::Unionset` (per `33 §6.5`). They are **not** inlined as `NestedDataKind` (C6.5). When a Grainset level needs to reference an implicit Unionset (for the ≥2 same-grain children case per C8.1), it does so via `RoutingUnitRef::Synthesized(EntityId)` rather than `RoutingUnitRef::Inline(NestedDataKind)`. C9.4 confirms the persistence is unchanged from C6's closing note — the manifest top-level placement is canonical, not under elaboration.

### 2.2 Nesting interaction

A `Unionset`:

- **MAY** appear at Root scope as a top-level public DataKind (per `32 §2.1`'s `unionsets:` plural tag).
- **MAY** be nested inline as a `NestedUnionset` under a `Grainset` or `Joinset` (per `26 §1`'s matrix). Nested form is structural-only (no `description` / `semantic_interface` / `ai_context`); the nested `name` is the structural anchor.
- **MUST NOT** be nested under another `Unionset` (R2; type-level — no `unionsets:` field on `UnionsetBody` per `26 §2.2`).
- **MUST** carry **≥ 2 children** across `datasets:` / `grainsets:` / `joinsets:` combined (R3 per `26 §2.3`; structural-rule diagnostic owned by `26` / `32`).

### 2.3 Authoring example — full SemanticInterface surface

A worked YAML showing every authoring slot a `Unionset` exposes (focus is **completeness of the surface**, not plan emission — see §10 for the plan-shape worked example):

```yaml
unionsets:
  - name: paid_media
    description: "Cross-platform paid media activity, normalized."
    mode: all
    dimensions:
      - name: ordered_at
        data_type: Timestamp
      - name: source_platform
        data_type: String
      - name: campaign_id
        data_type: String
    measures:
      - name: cost
        agg: sum
        expr: cost_cents
        data_type: Long
      - name: impressions
        agg: sum
        expr: impressions_count
        data_type: Long
    metrics:
      - name: cpm
        expr: cost / impressions * 1000
        data_type: Decimal
    keys:
      - name: campaign_id
        kind: foreign
    filters:
      - name: paid_only
        expr: cost > 0
    extras:
      temporal:
        events:
          occurred_at: ordered_at
```

The parent interface above declares the shared surface; the `datasets:` block below lists the two platform children that supply it:

```yaml
    datasets:
      - name: adwords_daily
        extras:
          catalog: marketing_warehouse
          storage:
            format: Parquet
            paths: ["s3://b/adwords/*.parquet"]
          semantic_mapping:
            ordered_at: order_date
            source_platform: { literal: "adwords" }
            campaign_id: campaign_pk
            cost_cents: total_cost
            impressions_count: total_impressions
          temporal:
            events:
              occurred_at: order_date
              grain: Day
      - name: facebook_daily
        extras:
          catalog: marketing_warehouse
          storage:
            format: Parquet
            paths: ["s3://b/facebook/*.parquet"]
          semantic_mapping:
            ordered_at: ts
            source_platform: { literal: "facebook" }
            campaign_id: cmpg_id
            cost_cents: spend_cents
            impressions_count: imps
          temporal:
            events:
              occurred_at: ts
              grain: Day
```

Notes on this example:

- `dimensions:` / `measures:` / `metrics:` / `keys:` / `filters:` are **flat** at the Unionset level per `#[serde(flatten)]` on the SemanticInterface (per `32 §3.3`).
- `mode: all` is the default and may be omitted; shown explicitly here for clarity.
- Each nested `datasets:` entry inlines a `NestedDataset` per `32 §3.2`. Each authors its own `extras.semantic_mapping:` to project its physical columns onto the Unionset's surface Semantics.
- `source_platform` is mapped to a per-source **literal** in each child — this is the per-source distinguishing metadata mechanism that drives disjointness elision at the final-aggregation layer (§4.4). Per-source `Metadata` recipes (path-token extraction per `15 §8`) are an alternative carrier.
- Children's `temporal:` blocks must be equivalent per the V1 strict equivalence rule (§5.2) — same `events:` shape, same `grain: Day`. The Unionset's `extras.temporal.events:` cascades up to match.

## 3. Per-Child Contract

### 3.1 Children list shape and ordering

The `UnionsetBody` carries three child maps — `datasets: BTreeMap<EntityId, NestedDataset>`, `grainsets: BTreeMap<EntityId, NestedGrainset>`, `joinsets: BTreeMap<EntityId, NestedJoinset>` (per `32 §3.2`). The **canonical child sequence** for plan emission is:

1. all `datasets:` entries in name order, then
2. all `grainsets:` entries in name order, then
3. all `joinsets:` entries in name order.

This order is stable (per `32 §7` ordering table — collections are id-keyed but projected name-ordered) and is the order branches appear in the emitted `PlanNode::Union`'s `inputs` (§4.2). The intra-variant ordering basis changed from YAML author order to name order with the id-first model rework (`32 §7`); it remains deterministic.

R2 (no same-variant self-nesting) is type-level absence per `32 §3.2` — a `unionsets:` field does not exist on `UnionsetBody`. R3 (≥ 2 children) is the validate-stage rule per `26 §2.3` (diagnostic `validate.complex-data-kind-insufficient-children`); `23` does not re-codify. Nested-child name uniqueness within a single Unionset's combined child set is the Unionset author's responsibility; collisions surface via the `26 §4` addressing scheme as ambiguous addresses.

### 3.2 Per-child Coverage — v1 inference-only

V1 carries **no authored per-child Coverage override**. The `coverage:` block on child entries (and the `ChildCoverageOverride { provides }` shape) from pre-cascade `23` are retired. Per-child Coverage is inferred at compile from each child's own interface plus the child's Binding-level Coverage (for Simple children) or its resolved composed interface (for Complex children).

**Per-Semantics inference rule.** For each composed-surface Semantics `s` declared on the Unionset's own `SemanticInterface`, and for each child `i`:

- If child `i`'s interface declares a Semantics with the same name `s` (or a `- ref: s` to a root-pool entry shared by the Unionset) AND the child can produce a value (per `15 §10` for Simple, per the child's own resolution for Complex), then child `i` **provides** `s`.
- Otherwise, child `i` **NullFills** `s` — `s` is absent from the child's projection and a typed-NULL constant is emitted at the seam (§4.3).

This is the **default non-strict** behavior: any Semantics provided by ≥ 1 child is queryable on the Unionset; non-providers emit NULL at the seam.

**Coverage-completeness check.** A composed-surface Semantics that NO child provides yields `COMP_E_2302 UnionsetCoverageIncomplete` (§8) — the column would be always-NULL across every branch, indicating an authoring mistake.

### 3.3 Implicit-vs-explicit Coverage discipline

A second branch of the Unionset contract exists for **implicit Unionsets** — created by the planner when a `Dataset` has multiple `PhysicalSource`s (currently the only origin per `21 §3.2`). The implicit case differs from the explicit case in one respect:

- **Explicit Unionset** — non-strict per-child Coverage as above. Per-child branches NullFill missing Semantics.
- **Implicit Unionset** — **strict** per-source SemanticMapping coverage. Every per-source pseudo-leaf must serve every Semantics in the parent `Dataset`'s `extras.semantic_mapping:`; absence is `COMP_E_2106 SimpleMultiSourceIncompatibleNullFill` per `21 §3.2`.

The strict-vs-non-strict distinction is an internal planner discriminator (the `strict` flag); it is **never authored** in YAML. Authors who need NullFill across heterogeneous physical sources wrap their `Dataset`s in an explicit Unionset per `21 §3.2`'s remediation.

## Testing Strategy

- Explicit vs implicit round-trip; UnionMode {All, Unique} enum round-trip; per-child Coverage inference (non-strict); implicit Unionset (multi-source Dataset per 21 §3.2).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
