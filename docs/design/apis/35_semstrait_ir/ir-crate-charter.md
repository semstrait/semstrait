---
id: 35-ir-crate-charter
status: ratified
version: 2.1
impl: implemented
---

# IR Crate Charter

`semstrait-ir` is the **resolved-IR crate**. It carries every type the post-compile pipeline operates on: the resolved expression type (`ir::PhysicalExpr`), the function **registry** (`FunctionRegistry`, resolving the model-owned `CanonicalFn` name to a `FunctionSpec`), canonical semantic-graph vocabulary, the in-memory plan tree (`SemanticPlan`), and adapter-consumable output types.

## 1. Purpose and Scope

### 1.1 What `semstrait-ir` OWNS

| Surface | Variant catalog ratified by |
|---|---|
| The **resolved** expression type `ir::PhysicalExpr` — the alias `Expr<ir::PhysicalLeaf>` — its owned leaf set `PhysicalLeaf` (admits `Parameter`), and `Parameter` + `ParameterKey` | `14 §3` / `§4` |
| Non-authored type vocabulary — `TypeClass` (function-signature matching), `Schema`, `SchemaColumn` (plan-side output schema, compile/catalog-derived) | `13` (item Y / V2) |
| `FunctionRegistry` / `FunctionSpec` / `FnSignature` / `ParamType` / `ReturnTypeRule` / `FunctionCategory` / `RegistryExtension` / `function_registry()` — the registry resolving the model-owned `CanonicalFn` name | `14a §2` (item Y / V4) |
| Canonical semantic-graph types — `SemanticGraph`, `SemanticNode`, `SemanticEdge`, `SemanticGraphFragment`, `SegmentKey`, `SemanticInterfaceBitmask` + read traits | `33` / `34` split contract (shape in ir, lifecycle in planner) |
| Narrow ir-emitted error kind — `CompileError` (function resolution + `ReturnTypeRule::Custom`) | `14a §3.5` |
| `SemanticPlan` root + `PlanNode` sum (8 variants) + well-formedness invariants | `00 §4.1` |
| Plan-level primitives — `SourceRef`, `ResolvedColumn`, `Name`, `KeyPair`, `SortDir`, `NullOrdering`, `AggregateExpr`, `NodeMeta` | `15` / `16 §5` |
| Adapter-consumable output family — `EngineArtifact`, `EnginePlan`, `SqlArtifact`, `DialectId`, `Dialect`, `Capability` | `00 §4.1` (shape only; emission in `36`) |
| Visitor / traversal API over `PlanNode` trees | unified with `Tree` |
| Serde derivations for `SemanticPlan`, `ir::PhysicalExpr`, and every public IR type | — |
| `IrErrorKind` typed-kind enum + `Diagnose` impl | — |
| Substrait-mapping table (declarative `PlanNode` ↔ `substrait::proto::Rel`); conversion code lives in `36` | — |

**Imported, not owned (item Y / V1–V10).** `35` builds the resolved IR atop vocabulary homed in lower-rank crates:

| Imported surface | From | Clause |
|---|---|---|
| Universal-traversal trait family — `Tree`, `Visitor<N>`, `Rewriter<N>` + the tree-machinery `ValidateError` | `semstrait-common` | V5 |
| `ExprLeaf` trait | `semstrait-model` | V5 |
| Shared `Expr<L>` structural enum — carries every variant incl. `Window` — + author-facing leaf sets + accessor enums | `semstrait-model` | V3 / V6 |
| Support enums — `BinaryOpKind`, `UnaryOpKind`, `AggregateKind`, `AggregationOp`, `LikeKind`, `CastFailure` + `Literal` / Window enums + carriers | `semstrait-model` | V10 |
| `CanonicalFn` name identity (the registry that resolves it is ir-owned, above) | `semstrait-model` | V4 |
| Authored type vocabulary `DataType`, `Grain` | `semstrait-model` | V2 |
| Authoring-surface DSL — `expr_fn`, `std::ops`, `ExprFunctionExt` | `semstrait-model` (OQ-D closed — option (a); constructs the model authoring types) | OQ-D |

### 1.2 What `semstrait-ir` does NOT own

- **Planning strategy and per-DataKind plan assembly** — lives in `semstrait-planner` per `34`.
- **Optimization passes** — rule-based rewrites over `SemanticPlan` live in `34`. `35`'s `walk` / `transform` helpers are the substrate those rewrites run on.
- **Adapter emission** — translating a `SemanticPlan` into an `EngineArtifact` is `36`'s contract.
- **SemanticManifest shape** — `SemanticPlan` refers to the SemanticManifest through the opaque `SourceRef` handle. SemanticManifest types live in `33`; `35` never embeds them inline.
- **Runtime graph lifecycle policy** — TTL/LRU cache ownership, segment admission/eviction, memory-budget enforcement, and drift-policy execution live in planner (`34`). `35` owns graph *shape and read contracts* only.
- **The authoring expression vocabulary itself** — under item Y the authoring `Expr<L>` family, both author-facing leaf sets, support enums, `Literal`, carriers, `CanonicalFn` name, and authored `DataType` / `Grain` are **owned by `semstrait-model`** and merely *imported* here. `35` owns the resolved `ir::PhysicalExpr`.
- **Planner-time `UnresolvedSemanticExpr` → `ir::PhysicalExpr` realization policy** — the runtime algorithm lives in planner runtime (`34`). `35` owns only the resolved expression type and graph types.
- **Phase B placement and `Parameter` binding** — the `Strategy`-driven plan-tree construction lives in `34`. `35` owns the `PlanNode` and `PhysicalExpr` types; the planning algorithm itself is `34`'s contract.
- **Canonical-function semantics and per-engine mapping** — `14a` and `registry/functions_mapping.md` are authoritative. `35` owns the registry's *shape*, not its *contents*.
- **Engine / dialect identity outside the artifact family** — `DialectId` MUST appear only on `SqlArtifact.dialect`, `Dialect::ID`, and `Capability` membership. It MUST NOT appear on `SemanticPlan`, any `PlanNode` variant, any `Expr<L>` variant, any leaf-set variant, `NodeMeta`, or any registry-side type. Per S7 and Q4.A (2026-05-21).

### 1.3 Design posture — pure, sync, canonical

`semstrait-ir` is deliberately **pure** (no I/O, no async, no engine identity):

- **Zero I/O surface** — Concrete I11 guarantee (per `30 §9`).
- **Zero async** — Every method on every public type is synchronous; `SemanticPlan` is built, walked, rewritten, and serialized on the caller's thread. I6 guarantee.
- **Zero engine identity** — No `datafusion::*`, no `arrow::*`, no `spark::*`, no `duckdb::*` types are visible on any public surface. `DialectId` is an opaque newtype; `substrait::proto::Plan` is the one exception and appears only inside `EnginePlan::Substrait(_)` as the adapter-consumable payload.
- **Two upstream dependencies (item Y)** — `semstrait-ir` depends on `semstrait-common` (diagnostic primitives, `Tree` / `Visitor` / `Rewriter` family + tree `ValidateError`, constraint-DSL toolkit) and on `semstrait-model` (the authoring expression vocabulary + authored type vocabulary it imports). The `model → ir` edge of earlier revisions is inverted to `ir → model` (V1). Every crate above ir (`manifest`, `planner`, `adapter`) depends on it directly or transitively.

### 1.4 Engine-IR concept inspiration

`PlanNode` borrows its **catalog of operators** and **tree composition** from engine IRs — DataFusion's `LogicalPlan`, Calcite's `RelNode`, Substrait's `Rel`. Per `00 §3`, the inspiration is **structural only**:

- **Borrowed**: the set of operators (`Scan`, `Filter`, `Project`, `Agg`, `Join`, `Union`, `Sort`, `Fetch`), the box-per-child tree shape, the invariant that every non-leaf carries typed inputs.
- **Rejected**: cost / statistics fields on any `PlanNode`; physical / distribution properties; dialect or adapter branching on node variants; vendor-specific rel kinds.

### 1.5 Eight ratified statements

**S1 — IR provides the resolved canonical shapes.** `35` carries the resolved `ir::PhysicalExpr`, the plan tree, the registry, and the non-authored type vocabulary; it imports the authoring vocabulary + authored `DataType` / `Grain` from `semstrait-model` (item Y).

**S2 — IR provides typed `PlanNode` implementations, Substrait-shaped.** `PlanNode { Scan, Filter, Project, Agg, Join, Union, Sort, Fetch }` is a closed sum mirroring Substrait's `Rel` family with name-based (`Name`) column references.

**S3 — IR operates on canonical forms only.** Engine identity, dialect identity, SQL text, raw paths, and vendor types are forbidden on every public surface except the artifact family; this makes `00 §9` `I1` / `I2` / `I3` an IR-internal contract.

**S4 — Substrait is the main inspiration AND a co-equal output target — not the canonical wire form.** Canonical wire form is the in-process Rust types; Serde and Substrait are derived surfaces.

**S5 — Plan nodes carry descriptive annotations for traceability.** Annotations are non-computational metadata, classified TRACE/PLAN, with the Substrait carrier at §15.3 (see [ir-serde-substrait.md](ir-serde-substrait.md)).

**S6 — IR holds the surface listed in §1.1; `semstrait-common` holds diagnostic primitives + constraint-DSL + io transport per `31 §1.1`.**

**S7 — IR holds zero engine information; adapters do.** `DialectId` is the only engine-identity vocabulary in `35`, and it appears only on `SqlArtifact.dialect` and `Dialect::ID`.

**S8 — IR is shape and traversal; computation lives in producers.** `35` provides `Tree::transform` and the typed shapes only; lowering and constant-folding per `19` are producer-side rewrites.

### 1.6 Cross-cutting rules

Rules that apply across all of `35`'s content. Violations are spec defects.

**R1 — Canonical-only.** Every public surface of `35` carries canonical vocabulary only. Engine identity, dialect identity, SQL text, raw paths, and vendor types are forbidden on every public surface except the artifact family and `Dialect::ID`. Per S3.

**R2 — Naming readability.** Every public name (modules, types, traits, variants, fields, methods) MUST be guessable in isolation by a fresh reader. "Kind" is a modifier suffix on categorizing enums (`DataKind`, `LikeKind`, `IrErrorKind`); never a standalone noun. References are named after the referenced thing (`DataKindRef`, `SourceRef`), never as bare `KindRef`. Renaming a public name is MAJOR per `30 §2.1`.

**R3 — Producer / consumer roles.** The planner (`34`) is the producer of `SemanticPlan` values. Adapters (`36` and family) are the consumers. `35` defines the shape; `34` writes it; `36` reads it.

**R4 — Closed catalog, registry-mediated extension.** `PlanNode` variants, `Expr<L>` variants, leaf-set variants are `#[non_exhaustive]` for additive growth, but every variant lives in this spec. Engine-specific or strategy-specific extensions land as `FunctionRegistry` entries per `14a §7`, never as new `PlanNode` or `Expr<L>` variants. Optimizer passes rewrite over the closed catalog.

## 2. Module Layout

Top-level `pub mod` structure. One module per cohesive concept.

```
semstrait-ir
├── types                // TypeClass, Schema, SchemaColumn (13 — non-authored only;
│                        //   DataType / Grain are imported from semstrait-model, item Y / V2)
├── expr                 // The RESOLVED expression alias + owned leaf set + Parameter
│   ├── physical         //   ir::PhysicalExpr = Expr<PhysicalLeaf> (alias — the shared
│   │                    //     tree incl. Window comes from semstrait-model, re-aliased here)
│   ├── leaves           //   PhysicalLeaf { Column, Literal, Parameter } (14 §3.4)
│   └── parameter        //   Parameter struct, ParameterKey closed enum (14 §5)
├── functions            // FunctionRegistry, FunctionSpec, FnSignature, ParamType,
│                        //   ReturnTypeRule, FunctionCategory, RegistryExtension trait,
│                        //   function_registry() accessor (14a §2 / §3 / §8). Resolves the
│                        //   model-owned CanonicalFn name (imported).
├── semantic_graph       // SemanticGraph, SemanticNode, SemanticEdge, GraphExprRef,
│                        //   SemanticGraphFragment, SegmentKey,
│                        //   SemanticInterfaceBitmask, read-only graph contracts
├── plan                 // SemanticPlan, PlanNode, per-variant structs, NodeMeta
│   ├── node             //   PlanNode enum + variant-struct shapes
│   └── traversal        //   PlanVisitor, walk_pre / walk_post / transform
├── primitives           // SourceRef, ResolvedColumn, Name, KeyPair, SortDir,
│                        //   NullOrdering, AggregateExpr
├── artifact             // EngineArtifact, EnginePlan, SqlArtifact, DialectId, Dialect
├── error                // CompileError (raised by ReturnTypeRule::Custom callbacks);
│                        //   IrErrorKind (plan-shape diagnostics).
│                        //   ValidateError is NOT here — it co-moved to semstrait-common (item Y / V5).
└── substrait_map        // Substrait mapping TABLE only (no conversion code)

// Imported (item Y), not declared here:
//   Tree / Visitor<N> / Rewriter<N> + ValidateError ........ semstrait-common
//   Expr<L>, UnresolvedSemanticLeaf, UnresolvedPhysicalLeaf, ExprLeaf,
//     accessor enums, BinaryOpKind/…/Literal, ColumnRef,
//     SemanticsName, CanonicalFn-name, DataType, Grain ...... semstrait-model
```

**Split rationale:**

| Module | Reason for separate top-level home |
|---|---|
| `types` | Non-authored type vocabulary (`TypeClass` / `Schema` / `SchemaColumn`); plan-side, compile/catalog-derived. The authored `DataType` / `Grain` are imported from `semstrait-model`. |
| `expr` | The resolved `ir::PhysicalExpr` (+ Window + Parameter); referenced by `plan` but not vice-versa. |
| `expr::{physical, leaves, parameter}` | Per-type-family split for `cargo public-api` audit clarity; `parameter` the compile-emitted placeholder. |
| `functions` | The registry resolving the model-owned `CanonicalFn` name; adjacent to `expr` because every `FunctionCall` consumer resolves `name` against it. |
| `semantic_graph` | Canonical graph vocabulary shared by manifest/planner boundaries; keeps graph-shape evolution independent from planner cache policy. |
| `plan` vs `primitives` | `PlanNode` references every primitive but not vice-versa; primitives can be consumed without the full `PlanNode` surface. |
| `plan::node` vs `plan::traversal` | Traversal-API method count scales with variant count; isolating it limits I10 blast radius. |
| `artifact` | Output shape, decoupled from input shape; consumed by the engine layer above `semstrait-adapter`. |
| `error` | Co-locates `CompileError`, `IrErrorKind`. The tree-machinery `ValidateError` is **not** here — it co-moved to `semstrait-common` with the `Tree` family (item Y / V5). |
| `substrait_map` | Table reference only; conversion code lives in `36`. |

**Re-exports.** The crate root re-exports a curated surface. Non-root re-exports of internal helpers are forbidden.

## 2A. Canonical `semantic_graph` Surface

`semstrait-ir::semantic_graph` defines the canonical graph type system used between manifest seeds (`33`) and planner runtime lifecycle (`34`).

```rust
#[non_exhaustive]
pub struct SemanticGraph {
    pub nodes: BTreeMap<SemanticNodeId, SemanticNode>,
    pub edges: BTreeMap<SemanticEdgeId, SemanticEdge>,
    pub outgoing_edges: BTreeMap<SemanticNodeId, Vec<SemanticEdgeId>>,
}

#[non_exhaustive]
pub struct SemanticNode {
    pub payload: SemanticNodePayload,
    pub interface_id: Option<EntityId>,    // -> manifest `interfaces`
}

#[non_exhaustive]
pub enum SemanticNodePayload {
    DataKind {
        data_kind_id: EntityId,            // -> manifest `data_kinds`
    },
    Semantic {
        semantic_id: EntityId,             // -> manifest `semantics` registry
    },
    Expression {                           // reserved (not emitted by v1)
        expr_ref: GraphExprRef,
    },
    Source {
        source_id: EntityId,               // -> manifest `sources`
    },
}

#[non_exhaustive]
pub enum GraphExprRef {
    Physical(PhysicalExprId),
}
```

Edges connect nodes via explicit adjacency tracking. `SemanticEdge` carries optional predicates for conditional relationships.

```rust
#[non_exhaustive]
pub struct SemanticEdge {
    pub from: SemanticNodeId,
    pub to: SemanticNodeId,
    pub edge_type: SemanticEdgeType,
    pub predicate_expr: Option<GraphExprRef>,
}

#[non_exhaustive]
pub enum SemanticEdgeType {
    Relationship,
    Composition,
    Binding,
    DependsOn,
}
```

Graph fragments encapsulate cache-admissible subgraphs with their segment identity and source dependencies.

```rust
#[non_exhaustive]
pub struct SemanticGraphFragment {
    pub key: SegmentKey,
    pub graph: SemanticGraph,
    pub touched_sources: Vec<EntityId>,    // -> manifest `sources`
}

#[non_exhaustive]
pub struct SegmentKey {
    pub manifest_epoch: u64,
    pub datakind_ids: BTreeSet<EntityId>,
    pub requested_semantics: BTreeSet<EntityId>,
    pub constraints_fingerprint: [u8; 32],
}

#[non_exhaustive]
pub struct SemanticInterfaceBitmask {
    pub words: Vec<u64>,
}
```

`SemanticInterfaceBitmask` is a local coverage view only. The global semantic registry stays in manifest (`33`) as `SemanticBitmap`; graph structures never duplicate registry ownership.

**Graph-local handle types.** `SemanticNodeId` / `SemanticEdgeId` are `u32` newtypes — dense, build-local positional indices:

```rust
pub struct SemanticNodeId(u32);   // from_raw / as_u32
pub struct SemanticEdgeId(u32);   // from_raw / as_u32
```

They are deliberately *not* the workspace's `u128` identity width (`NodeId` / `ExprId`): those carry durable or content-addressed identity, whereas these are throwaway traversal indices that never leave memory. `u32` (4-billion-node ceiling) is sufficient and compact in every map key and adjacency entry.

**Identity vs. graph handles.** Every reference from a node payload back to a manifest entity (`data_kind_id`, `semantic_id`, `source_id`, `interface_id`, `SegmentKey` sets, `touched_sources`) is an `EntityId` into the corresponding `EntityId`-keyed manifest collection (`33 §4.1`). This is the "compact handles in memory, `EntityId` identity on disk" boundary.

**`DataKind` payload carries identity only.** `SemanticNodePayload::DataKind` holds `data_kind_id` and **no** `name`. The authoring `DataKindName` lives in `semstrait-model`, *above* `semstrait-ir` in the crate DAG (I7); embedding it in an IR type would be an illegal upward dependency. The name is recoverable from the manifest `data_kinds` collection keyed by `data_kind_id`.

**No serialization.** None of the `semantic_graph` types derive serde. They are reconstructed in memory at graph build and never persisted — the manifest stores identity primitives only ("identity on disk, structure in memory", `33`).

**Read contracts:**

```rust
pub trait SemanticGraphRead {
    fn node(&self, id: SemanticNodeId) -> Option<&SemanticNode>;
    fn edge(&self, id: SemanticEdgeId) -> Option<&SemanticEdge>;
    fn outgoing(&self, id: SemanticNodeId) -> &[SemanticEdgeId];
}

pub trait SemanticGraphFragmentRead {
    fn key(&self) -> &SegmentKey;
    fn graph(&self) -> &SemanticGraph;
    fn touched_sources(&self) -> &[EntityId];   // -> manifest `sources`
}

pub trait SemanticInterfaceBitmaskOps {
    fn is_subset_of(&self, other: &Self) -> bool;
    fn intersects(&self, other: &Self) -> bool;
}
```

**Graph invariants:**

- `SemanticGraphFragment` is a DAG. Introducing a cycle is an error at planner graph-build time.
- Planner runtime DAG backend target is `daggy` (per `34 §1.4A`); IR remains backend-agnostic and exposes no backend-specific graph types.
- Node/edge collections are stable-id keyed maps; adjacency is explicit via `outgoing_edges`.
- `outgoing_edges[node]` is sorted by `SemanticEdgeId` for deterministic traversal.
- `SemanticNodePayload::{DataKind, Semantic}` carry identity only; full semantic metadata remains owned by manifest (`33` `SemanticBitmap`).
- Graph-held expression references are pool-typed via `GraphExprRef` (physical-only); runtime pool dispatch by untyped `ExprId` is forbidden.
- Every `GraphExprRef` must resolve to an existing pool entry before the fragment is admitted. The referenced `ManifestExpression` carries the `ExprLayer` (`33 §7.2`); the graph does **not** duplicate the layer.

### Expression layering and the `DependsOn` reservation

The graph does **not** persist or synthesize an expression-dependency DAG (Metric → Measure → Expression) in v1. The vocabulary exists — `SemanticNodePayload::Expression` and `SemanticEdgeType::DependsOn` — but is **reserved**, not built:

- **Why unnecessary for correctness:** A Metric's lowered `PhysicalExpr` already contains its constituent `Aggregate` nodes (named-Measure references are substituted inline at Phase A, `19 §3.3`). Phase B's aggregate-lift (`19 §7`) extracts those `Aggregate` nodes into pre-aggregatable slots; the `ExprLayer` tells the planner the residual lands above the final aggregation. Together with per-aggregate `Additivity` this drives correct pre-/re-aggregation across complex DataKinds **without** a separate dependency structure.
- **What it's reserved for (post-v1):** Cross-Metric shared-aggregate elimination (compute `sum(x)` once when several Metrics use it) and semantic-layer explainability (showing a physical plan in terms of authored Measures). Both are optimizations/UX, not correctness.
- **v1 graph build** synthesizes only `Relationship` / `Composition` / `Binding` edges from manifest primitives; `DependsOn` and `Expression` nodes are not emitted.

**Boundary rule:** IR owns shape and read semantics only. Runtime mutability, cache policy, cycle-check strategy, and drift-policy execution are planner contracts (`34`), never IR contracts.

## Dependency Posture

```toml
[dependencies]
semstrait-common = { path = "../semstrait-common" }
semstrait-model  = { path = "../semstrait-model" }   # item Y / V1 — imports authoring vocab
thiserror      = "^"
prost          = "^"
substrait      = "^"   # substrait::proto::Plan for EnginePlan::Substrait
uuid           = { version = "^", features = ["v4"] }

[dependencies.serde]
version  = "^"
optional = true
features = ["derive"]

[features]
default = []
serde   = ["dep:serde", "semstrait-common/serde", "semstrait-model/serde"]
```

**No runtime-only dependencies.** No `tokio`, `async-trait`, `futures`, `reqwest`, `hyper`, `sqlx`.
**No engine dependencies.** `substrait` (a proto codegen crate) is permitted because `EnginePlan::Substrait(Box<substrait::proto::Plan>)` is structural, not engine-identity. No `datafusion`, no `arrow`, no `duckdb`, no `spark-*`.
**Exactly two in-workspace dependencies — `semstrait-common` and `semstrait-model` (item Y / V1).** `semstrait-ir` imports the authoring expression vocabulary + authored `DataType` / `Grain` from `semstrait-model` and the traversal-trait family + tree `ValidateError` from `semstrait-common`; the `model → ir` edge of earlier revisions is inverted to `ir → model`. CI-enforced manifest audit per `30 §9` bounds the edge set to these two.

## Testing Strategy

This leaf covers the crate charter and module structure. Testing is covered by:

- **Structural gates:** `doc-space-check` enforces frontmatter completeness and module-naming adherence.
- **Dependency audit:** CI check enforced by `30 §9` ensures only `semstrait-common` and `semstrait-model` appear in `[dependencies]`.
- **Graph invariant coverage:** See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for the DAG acyclicity check, `SemanticNodeId` positional-index range verification, and `GraphExprRef` pool-resolution validation.
- **Pure/sync guarantees (I6/I11):** `forbid_async_fn!` macro audit + integration-test `cargo public-api` scan for any `async fn` or I/O-surface leakage.
