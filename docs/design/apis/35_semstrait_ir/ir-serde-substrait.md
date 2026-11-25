---
id: 35-ir-serde-substrait
status: ratified
version: 2.1
impl: implemented
---

# IR Serde / Substrait Mapping

The wire-form serialization contracts for `SemanticPlan`, `ir::PhysicalExpr`, and the declarative mapping from `PlanNode` variants to Substrait `Rel` kinds. Conversion code lives in `36`; `35` ratifies the mapping so `36`'s emitter and the crate's outbound round-trip tests agree on correspondence.

## Serde

Every public IR type derives `Serialize` / `Deserialize` under the crate-level `serde` feature flag. `SemanticPlan` is the intended portable form: a serialized plan can be round-tripped across processes sharing the same SemanticManifest. Wire-form stability rules:

- **Enums carrying at least one data-bearing variant** serialize **adjacently tagged** — `#[serde(tag = "type", content = "content")]` — so each value is `{"type": "<snake_case_variant>", "content": <payload>}`; a unit variant emits `{"type": "<snake_case_variant>"}` (no `content`). Adding a variant preserves round-trip of existing variants. **Internal tagging** (`#[serde(tag = "type")]`, flat) is deliberately *not* used: several variants wrap a sequence or primitive payload (`Expr::Coalesce(Vec<_>)`, `Expr::IsNull(Box<_>)`, `Literal::String(String)`) into which serde cannot fold a flat tag field. Adjacent tagging is the only form that yields a literal named `type` discriminator across *all* variant shapes.
- **Fieldless enums** (pure unit-variant rosters — `BinaryOpKind`, `UnaryOpKind`, `AggregationOp`, `LikeKind`, `CastFailure`, `WindowFn`, `ParameterKey`) serialize as the **bare snake_case variant string** (`"safe_divide"`, `"i_like"`, `"request_temporal_axis"`); a `type`/`content` wrapper buys nothing where the variant name alone is unambiguous. The identifier carriers `ColumnRef` / `SemanticsName` and `CanonicalFn` are `#[serde(transparent)]` / bare-string.
- **Every `#[non_exhaustive]` struct** serializes with absent-field-tolerant deserialization. Adding a field preserves round-trip of existing values (new field defaults to its `Default::default` or `None`).
- **Serde splits by surface, not by enum (item Y / V7 as refined by A2).** Under A2 there is one shared `Expr<L>` (model-owned) and `ir::PhysicalExpr` is its alias, so the split is between two *projections* of that one enum, not between two enums. (1) The **machine transport** form — the shared `Expr<L>` itself derives `Serialize` / `Deserialize` adjacently tagged, **in `semstrait-model`** where the enum lives (orphan rule); this is the internal cross-process form between same-version peers (never author-facing YAML), so the `type` / `content` discriminator is correct, and `ir::PhysicalExpr` inherits it by aliasing. (2) The **authoring friendly** form is a **field-level projection** — `#[serde(with=…)]` + `#[schemars(schema_with=…)]` adapters on each authoring expression field, **hand-rolled in `semstrait-model`** (V7) to emit friendly tags. The field adapter intercepts the bare `Expr<L>` field's serialization, so `binary_op:` / `type:` / `content:` never reach authored YAML or its schema (V7 invariant, A2-preserved). The model-owned support enums and carriers compose into the shared `Expr<L>`'s wire shape transparently. `PlanNode`'s on-wire shape is specified with its own serde deliverable (not yet derived in code; adjacently tagged `type` / `content` convention is recommended for cross-enum consistency).
- **Outbound-only Substrait emission (Q4.C, 2026-05-21).** The `Serialize` half of every IR type is fully populated for the canonical wire form. The `Deserialize` half is for **IR-internal round-trip** only — process-to-process plan transport between same-version `semstrait` peers, or test fixtures. It is NOT a contract for inbound conversion from arbitrary `substrait::proto::Plan` values produced outside `semstrait`. Cross-engine inbound deserialization is out of scope for v1.

### Expression Serialization Derive Contract (per C18)

Per the manifest ratification clause C18 (`_research/manifest/RATIFICATION_LOG.md`), the canonical expression types have a sharper serialization contract than the general "every public IR type derives serde" rule above. C18 makes four specific picks that this crate carries:

1. **Derivation lives in the owning crate, not in `semstrait-manifest`.** Under A2 the *resolved* transport form `ir::PhysicalExpr` **is** the alias `Expr<PhysicalLeaf>`, so the structural-tree derive lives with the shared `Expr<L>` enum **in `semstrait-model`** (orphan rule); `semstrait-ir` derives `serde::Serialize` / `serde::Deserialize` only on its **own** leaf payloads — `PhysicalLeaf`, `Parameter` / `ParameterKey`. The Window enums and the other model-owned support enums / carriers carry their serde in `semstrait-model`; the friendly authoring projection (field-level serde + schemars adapters, `14 §6`) is hand-rolled there too (V7). Manifest (`33`) consumes whichever derive applies by storing `BTreeMap<PhysicalExprId, ManifestExpression>` (physical-only per `33 §7.2`); manifest-side `Serialize` / `Deserialize` impls on any of these types are forbidden.
2. **Encoding format is JSON via serde.** Matches the manifest-level §1 readability posture (`33`). No bespoke binary encoding for expression payloads.
3. **Deterministic encoding required.** The same `ir::PhysicalExpr` value (the persisted, physical-only pool form per `33 §7.2`) MUST produce the same byte sequence on every encode. This is a hard requirement — not a quality-of-implementation goal — because two downstream contracts depend on it:
   - `manifest_epoch` / `model_hash` stability across repeat compiles of the same `(model, catalog snapshot)` input (`00 §9` I4); a non-deterministic encoding would surface phantom epoch bumps even when nothing changed.
   - Compile-time content dedup of expressions (C18.5): the manifest allocates `ExprId` by hashing the canonical-encoded form, so encoding determinism is the precondition for "same expression value -> same id." Without it, structurally-identical authored expressions land in distinct pool entries.

   Concretely: `BTreeMap` field iteration ordering (already deterministic), explicit field ordering in derived struct serialization, and stable enum-tag spelling are sufficient. Floating-point literals in `Literal::Float` use a fixed canonical-string representation (no platform-dependent rounding).
4. **No per-`Expr` format version.** `manifest_epoch` is the only versioning channel for expression-payload format evolution (C18.4). Adding a `format_version: u32` field to `ir::PhysicalExpr` or its `PhysicalLeaf` is forbidden in v1; if a future format break is needed, the migration signal is a `manifest_epoch` bump, not a per-expression discriminator.

These four picks are the v1 contract. Open extensions (e.g., binary encoding for hot-path round-trip, content-addressed wire form) land additively under `30 §2.2` MINOR rules.

## Substrait Mapping Table

The adapter crate (`36`) owns the **outbound** conversion from `SemanticPlan` to `substrait::proto::Plan`. `35` ratifies the **mapping** so `36`'s emitter and the crate's outbound round-trip tests agree on which `substrait::proto::Rel` corresponds to which `PlanNode`.

Per Q4.C (2026-05-21): the table is **unidirectional (outbound only)**. Inbound conversion (`substrait::proto::Plan` → `SemanticPlan`) is not a v1 contract; arbitrary external Substrait plans are not accepted as a `SemanticPlan` source.

| `PlanNode` variant | Substrait `Rel` kind           | Notes                                                                                                                                            |
|--------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `Scan`             | `ReadRel`                      | `source` resolves to `ReadRel.read_type` via the adapter's SemanticManifest lookup. `filters_pushdown` → `ReadRel.filter` (one conjunction).             |
| `Filter`           | `FilterRel`                    | `predicate` → `FilterRel.condition`.                                                                                                             |
| `Project`          | `ProjectRel`                   | `projections` → `ProjectRel.expressions` (order-preserving). Output names carried in `RelRoot.names` at the plan root.                           |
| `Agg`              | `AggregateRel`                 | `group_by` → one `Grouping` with the referenced columns as `grouping_expressions`. `aggregates` → `AggregateRel.measures`.                       |
| `Join`             | `JoinRel`                      | `join_type` → `JoinRel.type`. `on` → `JoinRel.expression` (equijoin with conjunction of `KeyPair` equalities). `cardinality` → `AdvancedExtension.enhancement` with URN `urn:semstrait:join-cardinality:v1`. |
| `Union`            | `SetRel` with `op = UNION` / `UNION_DISTINCT` | `distinct = false` → `SET_OP_UNION_ALL`; `distinct = true` → `SET_OP_UNION_DISTINCT`.                                               |
| `Sort`             | `SortRel`                      | `order[*].0` → `SortField.expr` (resolved to the referenced column). `order[*].1` → `SortField.direction`.                                       |
| `Fetch`            | `FetchRel`                     | `limit` / `offset` → `FetchRel.count` / `FetchRel.offset` (`-1` when `None`).                                                                    |

The adapter is free to emit Substrait proto plans with extra hints (capacity, parallelism) in `AdvancedExtension.enhancement` slots; those hints are adapter-owned and not round-tripped through `35`.

## Annotation Substrait Carrier

1. **Carrier.** Every `SemAnnotation` on a `PlanNode` emits into `RelCommon.advanced_extension.optimization[]` of the corresponding `substrait::proto::Rel`, namespaced by URN `urn:semstrait:annotations:v1`; the `enhancement` slot is reserved for adapter execution hints and MUST NOT carry annotations.
2. **Roundtrip.** Binary `Plan.encode_to_vec` is fully roundtrip-safe; `36`'s emitter collects every annotation URN into the plan-root `Plan.expected_type_urls`. JSON emission is best-effort (proto3 JSON loses `Any` payload fidelity) and MAY elide annotations.
3. **Drop-safe.** Per Substrait `optimization[]` semantics, any consumer MAY discard entries it does not recognize; consumers MUST behave correctly with annotations stripped.

## Load-Time Integrity Contract (per CX1)

Per the manifest ratification clause CX1 (`_research/manifest/RATIFICATION_LOG.md`), every cross-reference threading IR-typed identifiers through the manifest wire form is integrity-checked at load. `35` owns the structural shape of the IR-typed reference vocabulary that participates in this check (`PhysicalExprId` per [ir-resolved-expr.md](ir-resolved-expr.md) §5.3.1); the load-time validation pass itself is performed by the consumer (`33`'s deserializer). The IR-side contract is:

1. **Single integrity pass on load.** Manifest deserialization performs one integrity walk over every IR-typed cross-reference. In v1 wire form, the required external site is `SemanticBinding.mapping[EntityId] = SemanticMappingValue::Expr(PhysicalExprId)` and each referenced `PhysicalExprId` MUST resolve in `expressions.physical` (whose entries are `ManifestExpression { expr, layer }`). (The mapping key is the semantic's `EntityId`; the expression-pool id is its own content-dedup handle per `33 §7.2`.)
2. **Failure shape.** A reference whose target is absent from its pool surfaces as `LoadError::DanglingReference{ from, to_kind, target_id }`. The `from` field carries the referencing site (binding id, hop, mapping key); `to_kind` discriminates the target pool (the physical expression pool vs other manifest collections); `target_id` is the unresolved newtype value. Per `30 §5.4`, identification is by variant identity.
3. **Hardens compile-time gates.** This pass extends C13's compile-time gates G1 / G2 / G4 from "compile-time only" to "compile-time + load-time defense in depth." A hand-edited or repository-corrupted manifest cannot slip a dangling IR-typed reference past load.
4. **No JSON `$ref` indirection.** This contract was ratified in lieu of OpenAPI-style `$ref` JSON pointers. The wire format remains the id-keyed map plus newtype-wrapped id reference (less verbose than `$ref`, no inline-vs-pointer ambiguity, no serde-side `$ref` resolver complexity). Integrity is preserved by the load-time pass, not by the wire form's structural shape.

The strongly-typed newtypes from [ir-resolved-expr.md](ir-resolved-expr.md) §5.3.1 are what make this pass type-directed rather than name-directed: each cross-reference site declares which pool it targets via the newtype, and the integrity walk dispatches to the correct pool by type. Renaming a newtype is MAJOR per `30 §2.1` because every load-time integrity site references it by name.

## Testing Strategy

This leaf owns serde posture for `SemanticPlan` / `PhysicalExpr`, Substrait mapping, annotation carrier, and load-time integrity. Tests:

- **Serde roundtrip:** Feature-gated tests verify `SemanticPlan` serializes to deterministic JSON and deserializes to a structurally-equal plan. Adjacent `type` / `content` tagging for data-bearing enums; bare snake_case string for fieldless enums.
- **Expression serialization determinism (C18.3):** Unit tests verify the same `ir::PhysicalExpr` value produces identical byte sequences on every encode (no platform-dependent floating-point rounding, stable field ordering, `BTreeMap` iteration).
- **No per-`Expr` format version (C18.4):** Code review enforces no `format_version` field appears on `ir::PhysicalExpr` or `PhysicalLeaf`; `manifest_epoch` is the only versioning channel.
- **Substrait mapping correctness:** Integration tests verify each `PlanNode` variant maps to the expected `substrait::proto::Rel` kind per the table above; `36`'s emitter and `35`'s round-trip tests agree on correspondence.
- **Annotation Substrait carrier:** Unit tests verify every `SemAnnotation` on a `PlanNode` emits into `RelCommon.advanced_extension.optimization[]` with URN `urn:semstrait:annotations:v1`; binary `Plan.encode_to_vec` is fully roundtrip-safe.
- **Load-time integrity (CX1):** Integration tests verify manifest deserialization walks every `PhysicalExprId` reference and raises `LoadError::DanglingReference` when a reference's target is absent from `expressions.physical`. A hand-edited manifest with a dangling id cannot load.
- **Outbound-only contract (Q4.C):** Code review enforces `Deserialize` impls are for IR-internal round-trip only; no inbound conversion from arbitrary external Substrait plans is contractually supported in v1.

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
