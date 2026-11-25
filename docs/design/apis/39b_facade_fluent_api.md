---
id: 39b-facade-fluent-api
status: ratified
version: 2.1
impl: planned
---


# 39b. Facade Fluent API — v1 Base Abstractions and Control Flow

> **Status:** v1 reservation. **Pins base abstractions, their crate locations, and the pipeline-stage contract.** Concrete builder method signatures, ergonomic shorthand methods, and per-stage internal mechanics are deferred to the per-crate API docs (`32` / `33` / `34` / `36`). The formal facade contract lives in [`39_semstrait_facade.md`](39_semstrait_facade.md); this doc reserves the API ergonomics 39 must serve.

---

## 1. Purpose

For v1, reserve:

- The **base abstractions** that participate in the user-facing fluent pipeline.
- Their **crate locations** (where each type lives).
- The **per-crate responsibility split** — what each crate owns and what it does not own.
- The **control-flow stage vocabulary** (`build`, `compile`, `query`, `adapt`, `execute`) and the owning crate of each transition.
- The **stage-transition mechanism** — extension traits in the owning crate, reachable through `semstrait::prelude::*`.
- The **prelude composition by category**.
- The **resolution of the model↔expr coupling** at the import level (one prelude line covers both).

Out of scope for this v1 reservation:

- Concrete builder method signatures, parameter naming, default values.
- Whether sub-builder methods return `Self` or `Result<Self, _>`.
- Internal substep ordering inside `compile` / `plan` / `adapt`.
- Engine-specific execute() shapes — those belong to per-engine adapter crates.

---

## 2. Base abstraction inventory

The minimum set of types a fluent v1 author touches end-to-end, with their authoritative home.

| Abstraction | Kind | Owning crate | Doc owner |
|---|---|---|---|
| `SemanticModel` | struct | `semstrait-model` | `[32](32_semstrait_model.md)` |
| `Dataset` / `Grainset` / `Unionset` / `Joinset` (+ Nested variants) | struct | `semstrait-model` | `[32](32_semstrait_model.md)` |
| `Dimension` / `Measure` / `Metric` / `Filter` / `Relationship` / `Keys` | struct | `semstrait-model` | `[18](../foundations/18_entities.md)` (shape) + `[32](32_semstrait_model.md)` (placement) |
| `SemanticInterface` (per-DataKind exposure of entities) | struct | `semstrait-model` | `[32](32_semstrait_model.md)` |
| `SemanticMapping` (semantics-to-physical) | struct | `semstrait-model` | `[15](../foundations/15_mapping_and_binding.md)` (shape) + `[18 §10](../foundations/18_entities.md)` |
| YAML authoring expression surface (field-level friendly serde/schemars adapters on bare `Expr<L>` fields — block form + bare-leaf shorthand; no separate `ExprBlock` AST; the `ExprSource` wrapper is removed per doc 45 / `[14 §6.1](../foundations/14_expressions.md)`) | field-level adapters | `semstrait-model` | `[14 §6](../foundations/14_expressions.md)` (contract) + `[32](32_semstrait_model.md)` (impl) |
| `UnresolvedSemanticExpr` / `UnresolvedPhysicalExpr` (authoring expression forms) | type alias over `Expr<L>` | `semstrait-model` | `[14 §3.6](../foundations/14_expressions.md)` (item Y / V6) |
| `PhysicalExpr` (resolved canonical expression — no authoring `SemanticExpr`, V6 deletes it) | enum | `semstrait-ir` | `[35 §3.3](35_semstrait_ir.md)` |
| `Expr<L>` / `UnresolvedSemanticLeaf` / `UnresolvedPhysicalLeaf` (authoring structural tree + authoring leaf sets) | enum / enum / enum | `semstrait-model` | `[14](../foundations/14_expressions.md)` (item Y / V3) |
| `PhysicalLeaf` (resolved-IR leaf set; admits `Parameter`) | enum | `semstrait-ir` | `[35 §3.4](35_semstrait_ir.md)` |
| `Tree` / `Visitor` / `Rewriter` (traversal trait family) | trait / trait / trait | `semstrait-common` | `[31 §3](31_semstrait_common.md)` (item Y / V5) + `[14 §3.1](../foundations/14_expressions.md)` |
| `ExprLeaf` (leaf-payload trait) | trait | `semstrait-model` | `[14 §3.2](../foundations/14_expressions.md)` (item Y / V5) |
| `BinaryOpKind` / `UnaryOpKind` / `AggregationOp` / `AggregateKind` / `LikeKind` / `CastFailure` / `Literal` (authoring structural-variant support enums + typed-literal carrier) | enum | `semstrait-model` | `[14 §3.3](../foundations/14_expressions.md)` (item Y / V10) |
| `WindowFn` / `WindowFrame` / `WindowFrameKind` / `WindowBound` (window support enums; carried by the shared `Expr<L>`'s `Window` variant) | enum | `semstrait-model` | `[14 §3.3](../foundations/14_expressions.md)` (item Y / V10 as amended by A2; `semstrait-ir` imports them) |
| `ColumnRef` / `SemanticsName` (identifier carriers) | newtype / newtype | `semstrait-model` | `[14 §3.4 / §3.5](../foundations/14_expressions.md)` (item Y / V10; `semstrait-ir` imports both) |
| `Accessor` family (`MeasureAccessor` / `DimensionAccessor` / `MetricAccessor` / `KeyAccessor`) | enum | `semstrait-model` | `[14 §4](../foundations/14_expressions.md)` (item Y / V6; authoring) |
| `Parameter` (compile-emitted; the one resolved-only leaf) | struct | `semstrait-ir` | `[14 §5](../foundations/14_expressions.md)` + `[35 §6.2](35_semstrait_ir.md)` |
| `Window` (shared-tree variant; author-rejected, compile-emitted via §4.2 sugar) | enum variant | `semstrait-model` | `[14 §3.3](../foundations/14_expressions.md)` (item Y / A2) |
| `CanonicalFn` name (authored function token) | newtype | `semstrait-model` | `[14a](../foundations/14a_function_catalog.md)` (item Y / V4) |
| `FunctionRegistry` / `FunctionSpec` (resolution machinery) | struct / struct | `semstrait-ir` | `[14a](../foundations/14a_function_catalog.md)` + `[35 §8](35_semstrait_ir.md)` |
| `ValidateError` (tree-machinery error) | enum | `semstrait-common` | `[31 §3.1](31_semstrait_common.md)` (item Y / V5) |
| `CompileError` (resolved-stage error — `ReturnTypeRule::Custom` etc.) | enum | `semstrait-ir` | `[35 §16.2](35_semstrait_ir.md)` |
| `SemanticManifest` (compile output) | struct | `semstrait-manifest` | `[33](33_semstrait_manifest.md)` |
| `ResolvedExprTable` (per-(Semantics, Binding) PhysicalExpr) | struct | `semstrait-manifest` | `[19 §3.2](../foundations/19_expression_flow.md)` |
| `Request` / `RequestDimensionRef` (query-time input) | struct / struct | `semstrait-planner` | `[34](34_semstrait_planner.md)` |
| `SemanticPlan` / `PlanNode` (canonical plan tree) | struct / enum | `semstrait-ir` | `[35](35_semstrait_ir.md)` |
| `EngineArtifact` (adapter output) | enum | `semstrait-ir` | `[35](35_semstrait_ir.md)` |
| `EngineAdapter` (trait) | trait | `semstrait-adapter` | `[36](36_semstrait_adapter.md)` |
| `CatalogProvider` / `FileSystem` (traits) | trait / trait | `semstrait-catalog` | `[37](37_semstrait_catalog.md)` |
| `DataType` / `Grain` (authored type vocab) | enum / enum | `semstrait-model` | `[13](../foundations/13_types_and_grain.md)` (shape) + `[32](32_semstrait_model.md)` (placement, item Y / V2) |
| `Schema` | struct | `semstrait-common` | `[13](../foundations/13_types_and_grain.md)` (shape) + `[31](31_semstrait_common.md)` (placement) |
| `Diagnostic<K>` / `Diagnostics<K>` / `Severity` / `Diagnose` | struct / alias / enum / trait | `semstrait-common` | `[30 §5](30_api_contracts.md)` + `[31](31_semstrait_common.md)` |

This is the v1 reservation roster. Any new top-level abstraction is a clause-level MINOR addition per `[30 §2.2](30_api_contracts.md)`.

---

## 3. Per-crate responsibilities

A type lives in the crate that owns its **definition and v1 evolution**. A crate may **consume** types from upstream crates without owning them. The table is intentionally explicit about both halves.

| Crate | Owns | Does NOT own |
|---|---|---|
| `semstrait-common` | Non-expression shared vocabulary **plus the leaf-agnostic traversal traits** (item Y / V5): logical-type primitives moved to model under V2 are **not** here — common keeps `TypeClass`, `Schema`, `SchemaColumn`; the **traversal trait family** `Tree` / `Visitor` / `Rewriter` (V5, leaf-set-agnostic, impl'd by both `model::Expr<L>` and `ir::PlanNode`); the **tree-machinery `ValidateError`** (V5); constraint-DSL shapes (`MeasureConstraints`, `DimensionConstraints`, `AggregationConstraints`); diagnostic primitives (`Diagnostic<K>`, `Diagnostics<K>`, `Severity`, `Location`, `Span`, `SourceId`, `Diagnose`); byte-blob `io` transport | The authoring vocabulary — no `Expr<L>`, no `Unresolved{Semantic,Physical}Expr` / `PhysicalExpr`, no leaf sets, no `ExprLeaf` (model, V5), no support enums (`BinaryOpKind`, …), no `Literal`, no identifier carriers (`ColumnRef`, `SemanticsName`), no `CanonicalFn` / `FunctionRegistry`, no `DataType` / `Grain` (model, V2), no `CompileError` (ir); any model / manifest / plan type |
| `semstrait-ir` | The **resolved-machinery home** (item Y / V1–V6, A2): the resolved canonical expression `PhysicalExpr` — the `Expr<ir::PhysicalLeaf>` **alias** (A2; not a concrete re-declared enum), **no authoring `SemanticExpr`** (V6 deletes it) — over `PhysicalLeaf` (`{ Column, Literal, Parameter }`); the `Parameter` leaf (compile-emitted; the one resolved-only leaf); `FunctionRegistry` + `FunctionSpec` + function-level `Additivity`; `CompileError` (`ReturnTypeRule::Custom`); `PlanNode` + `SemanticPlan` + `SemanticGraph` family + `EngineArtifact` containers. Imports the authoring vocabulary (incl. the shared `Expr<L>`, `ExprLeaf`, and the Window support enums — A2) from `semstrait-model` via the thin `ir → model` edge (V1) and lowers it | Parsing (no YAML, no inline DSL parser); the **authoring** expression vocabulary (`Expr<L>` incl. the `Window` variant, `Unresolved*`, leaf sets, support enums incl. Window enums, `Literal`, carriers, `CanonicalFn`-name, `DataType`/`Grain` — all model under V2–V6, V10, A2); the traversal traits + `ValidateError` (common, V5); planning algorithms; engine emission |
| `semstrait-model` | **The authoring surface — owner of the unresolved expression vocabulary** (item Y / V2–V6, V10): the **YAML surface** (bare `Expr<L>` authoring fields with field-level friendly serde/schemars adapters — no separate `ExprBlock` parallel AST per `[14 §6.1](../foundations/14_expressions.md)`; the `ExprSource` wrapper is removed per doc 45; parse-site dispatch via `parse_block<L>`); the shared `Expr<L>` (incl. the author-rejected `Window` variant — A2) + `Unresolved{Semantic,Physical}Expr` aliases + the two authoring leaf sets (`UnresolvedSemanticLeaf` / `UnresolvedPhysicalLeaf`) + the `Accessor` family + `ExprLeaf` trait (V5); structural-variant support enums (`BinaryOpKind`, `UnaryOpKind`, `AggregationOp`, `AggregateKind`, `LikeKind`, `CastFailure`) + the Window support enums (`WindowFn`, `WindowFrame`, `WindowFrameKind`, `WindowBound` — A2) + `Literal` carrier (V10); identifier carriers (`ColumnRef`, `SemanticsName`) (V10); the `CanonicalFn` **name** (V4); the authored type vocabulary `DataType` / `Grain` (V2); **hand-rolled serde + schemars** as field-level adapters on each authoring expression field (V7 as refined by A2 and again by doc 45 — friendly projection; the former `ExprSource<L>` wrapper is removed, projection moved to per-field adapters, invariant unchanged; the shared `Expr<L>`'s own derived `type`/`content` serde is the machine transport form `ir::PhysicalExpr` aliases); **entity types** (`SemanticModel`, `Dataset`, `Grainset`, `Unionset`, `Joinset`, `Dimension`, `Measure`, `Metric`, `Relationship`, `Filter`, `Keys`, `SemanticInterface`, `SemanticMapping`); the validate-stage SR-* roster (a model `ValidateError` kind that embeds the `common::ValidateError` tree error per D.ii) | The **resolved** expression `PhysicalExpr` + `PhysicalLeaf` (ir); compile-time resolution / lowering (no name lookup, no cross-DataKind path BFS); the `FunctionRegistry` machinery + function-level `Additivity` (ir); the traversal-trait *definitions* (common, V5 — model only impls `Tree` for its `Expr<L>`) |
| `semstrait-manifest` | Compile-time resolution / lowering (`model::Unresolved*` → `ir::PhysicalExpr`); `ResolvedExprTable`; `SemanticManifest` (sealed artifact); the manifest-level `CompileError` (wider resolution-stage roster; embeds `Ir(ir::CompileError)` via D.ii); `Repository` trait + bundled impls; the `Compile` extension trait (build → compile transition) | Authoring (no YAML parsing); planning (no `Request` handling); definition of the authoring `Expr<L>` (model) / `FunctionRegistry` / `PlanNode` (those are `semstrait-ir`'s) |
| `semstrait-planner` | Planning (`Request` × `Manifest` → `SemanticPlan`); per-DataKind Strategy; `Aggregate` lift; `Parameter` binding from `Request`; the `Query` extension trait + `RequestBuilder` (compile → query transition) | Adaptation (no engine-specific rewrites); compile-time resolution |
| `semstrait-adapter` | Adapter trait (`EngineAdapter`); per-engine rendering of `PhysicalExpr` and `PlanNode` to engine artifacts; the `Adapt` extension trait (query → adapt transition) | Plan construction; expression resolution |
| `semstrait-catalog` | `CatalogProvider` + `FileSystem` trait surfaces; bundled providers (local, in-memory); optional providers behind features (Iceberg, Unity, etc.) | Compile-stage logic; planning |
| `semstrait-api` | Orchestration helpers (`SemStrait`, `SemStraitBuilder`); transport entry points (REST / gRPC / CLI behind feature flags) | Any sub-stage logic — composes only |
| `semstrait` (facade) | The crate root re-exports (`semstrait::core`, `::model`, `::ir`, `::manifest`, `::planner`, `::adapter`, `::catalog`, `::api`); `semstrait::prelude::*` (the curated bundle); `semstrait::builder()` re-export (= `SemanticModel::builder()`); optional `semstrait::run` YAML one-shot per `[39 §4](39_semstrait_facade.md)` | Any new types; any logic beyond `run` |

**The model↔ir split deserves an explicit clause** (item Y / V1–V6). `semstrait-model` **owns** the authoring expression vocabulary — `Expr<L>`, `UnresolvedSemanticExpr` / `UnresolvedPhysicalExpr`, the two authoring leaf sets, support enums, and the hand-rolled YAML serde — and parses YAML into those values. It does not own how an authored expression is *resolved*, *registry-dispatched*, or *lowered to the resolved `ir::PhysicalExpr`*. Those concerns live in `semstrait-ir` (registry, return-type rules, the resolved tree) and `semstrait-manifest` (the resolution fold). The model crate's contract with the expression layer is:

- It **owns and constructs** the authoring forms (`UnresolvedSemanticExpr` / `UnresolvedPhysicalExpr`) — parser output and DSL output alike.
- It applies **structural validation** to those values (per-site shape gates, reserved-identifier checks), surfacing the `semstrait-common`-owned `ValidateError` (V5).
- It does **not** resolve them. Lowering `model::Unresolved*` → `ir::PhysicalExpr` is the manifest compile fold (`[19 §3](../foundations/19_expression_flow.md)`); `semstrait-ir` consumes the authoring types via its thin `ir → model` edge (V1) and never re-defines them.

---

## 4. Control-flow stages

The pipeline at the API level. Each stage has a verb, an input, an output, an owning crate, and a transition mechanism.

```mermaid
flowchart LR
  AUTH["author<br/>(YAML or fluent)"] --> M["SemanticModel<br/>(semstrait-model)"]
  M --> MAN["SemanticManifest<br/>(semstrait-manifest)"]
  MAN --> P["SemanticPlan<br/>(semstrait-ir)"]
  P --> ART["EngineArtifact<br/>(semstrait-ir)"]
  ART --> RES["Engine result<br/>(engine adapter)"]

  AUTH -.->|"parse(yaml) → SemanticModel"| M
  AUTH -.->|"builder().…build() → SemanticModel"| M
  M -.->|".compile(&catalog).await"| MAN
  MAN -.->|".query().…build()"| P
  P -.->|".adapt(&engine)"| ART
  ART -.->|".execute().await"| RES
```

Per-stage contract:

| Stage | Input | Output | Owning crate | Transition mechanism |
|---|---|---|---|---|
| **build** | author input (programmatic or YAML) | `SemanticModel` | `semstrait-model` | `SemanticModel::builder().…build()` (programmatic) or `parse(&yaml)` (YAML) |
| **compile** | `SemanticModel`, `&dyn CatalogProvider` | `SemanticManifest` | `semstrait-manifest` | `Compile` extension trait on `SemanticModel` |
| **query** | `SemanticManifest`, `Request` (or `RequestBuilder`) | `SemanticPlan` | `semstrait-planner` | `Query` extension trait on `SemanticManifest` |
| **adapt** | `SemanticPlan`, `&dyn EngineAdapter` | `EngineArtifact` | `semstrait-adapter` | `Adapt` extension trait on `SemanticPlan` |
| **execute** | `EngineArtifact`, engine-specific runtime context | engine-specific result | per-engine adapter crate | inherent method on `EngineArtifact` or on the engine handle (engine-specific) |

**Verb vocabulary is reserved for v1**: `build` / `compile` / `query` / `adapt` / `execute`. These are the canonical names exposed through the prelude. Internal helper methods may shadow these names with different specificity (e.g., `parse(&yaml)` is a build-stage helper), but no stage transition takes a different name.

---

## 5. Stage-transition mechanism

Each cross-crate stage transition is implemented as a **single named extension trait** in the owning crate, implemented on the input-stage type. The pattern is uniform:

```rust
// Pattern (one of three concrete instances below):
pub trait <Verb> {
    type Output;
    type Error;
    fn <verb>(self, …context…) -> Result<(Self::Output, Diagnostics<Self::Error>), …>;
}
impl <Verb> for <UpstreamStageType> {
    type Output  = <DownstreamStageType>;
    type Error   = <PerStageErrorKind>;
    fn <verb>(…) -> … { … }
}
```

Concrete v1 traits:

| Trait | Owning crate | Implemented on | Output type |
|---|---|---|---|
| `Compile` | `semstrait-manifest` | `SemanticModel` | `SemanticManifest` |
| `Query` | `semstrait-planner` | `SemanticManifest` | `SemanticPlan` (via `RequestBuilder` terminal) |
| `Adapt` | `semstrait-adapter` | `SemanticPlan` | `EngineArtifact` |

**Why extension traits**: each owning crate is downstream of the type it adds methods to. The orphan rule says we can either define the trait in the owning crate (and impl it on the upstream type — legal because we own the trait) or define inherent methods on the upstream type (illegal across crates). Extension traits are the only working pattern.

**Discoverability**: each trait is re-exported through `semstrait::prelude::*`. A user with the prelude in scope sees all three transition methods as if they were inherent on the chain. The user never types `use semstrait::manifest::Compile;`.

**Single trait per transition, not many**: we reserve exactly one trait per cross-crate stage boundary. Trait proliferation (`CompileExt` + `CompileWithRepo` + `CompileWithMetadata` + …) is rejected. If a transition gains options, they go on the input type via a setter (`model.with_options(…).compile(&catalog)`) or on a builder, not on a new trait.

---

## 6. Prelude composition (by category)

The facade prelude (`semstrait::prelude::*`) brings the following **categories** into scope. The exact item list per category is owned by `[39 §3.2](39_semstrait_facade.md)`; this doc reserves the categories themselves.

| Category | Sourced from | Reserved for the prelude? |
|---|---|---|
| **Entity types** (`SemanticModel`, `Dataset`, `Grainset`, `Unionset`, `Joinset`, `Dimension`, `Measure`, `Metric`, `Relationship`, `Filter`, `Keys`, `SemanticInterface`) | `semstrait-model` | **Yes** — required for any author to construct a model |
| **Entity builders** (the `::builder()` constructors as inherent methods on the entity types) | `semstrait-model` | **Yes** — implicitly reachable when the types are in scope |
| **Authoring expression types** (`UnresolvedSemanticExpr`, `UnresolvedPhysicalExpr`, `Accessor` family) | `semstrait-model` (item Y / V3 / V6) | **Yes** — required as the value type of `Measure::expr(…)`, etc. The resolved `ir::PhysicalExpr` and compile-emitted `Parameter` are not author-facing |
| **Expression DSL** (`col`, `lit`, `field` / `dim` / `measure` / `metric` / `key`, `when`, `case`, aggregate helpers like `sum` / `count` / `avg` / `min` / `max`) | `semstrait-model` — the DSL constructs model authoring types and lands wholly beside them (OQ-D closed, option (a); see `[../questions/closed/35_questions.md](../questions/closed/35_questions.md)`) | **Yes** — required for inline expression authoring |
| **Expression operator overloads** (`impl Add for UnresolvedSemanticExpr`, etc.) | with the authoring type (`semstrait-model`); in scope automatically when the type is in scope | **Implicit** — no explicit prelude entry needed |
| **Stage-transition traits** (`Compile`, `Query`, `Adapt`) | per owning crate | **Yes** — without them the fluent chain breaks |
| **Query-input types** (`Request`, `RequestBuilder`, `RequestDimensionRef`) | `semstrait-planner` | **Yes** — required to author queries |
| **Engine-adapter trait** (`EngineAdapter`) | `semstrait-adapter` | **Yes** — required to bind a plan to an engine |
| **Catalog-provider trait** (`CatalogProvider`) | `semstrait-catalog` | **Yes** — required as `.compile(&catalog)`'s argument type |
| **Plan-tree types** (`SemanticPlan`, `PlanNode`, `EngineArtifact`) | `semstrait-ir` | **Yes** — required to inspect/manipulate plans between stages |
| **Diagnostic primitives** (`Diagnostic`, `Diagnostics`, `Severity`, `Diagnose`) | `semstrait-common` | **Yes** — required to consume the canonical `(Output, Diagnostics<K>)` tuple |
| **Logical type primitives** (`DataType`, `Grain`) | `semstrait-model` (item Y / V2) | **Yes** — required as field types on entities |
| **Top-level convenience** (`semstrait::builder()` free function) | facade | **Yes** (function not re-exported but reachable at crate root) |

**Categories explicitly NOT in the prelude** (reachable through `semstrait::<module>::*`):

- Internal-only types (`ResolvedExprTable`, `Binding`, plan-strategy types).
- Per-stage `*ErrorKind` enums except the unified `SemStraitErrorKind` (consumed via `Diagnostic::kind`).
- Function-spec types (`FunctionSpec`, `FnSignature`, `ParamType`) — the registry is process-global; these are rarely referenced by name at the call site.
- Per-engine adapter handles beyond the default (default engine ships in the facade default features; other engines reachable through their adapter crates).

---

## 7. Resolution of the model↔expr coupling

**The observation.** A `Measure` carries an authored expression (`UnresolvedSemanticExpr`). Constructing a `Measure` value programmatically requires that expression value. Both the entity type and the expression form live in `semstrait-model`, so authoring is self-contained within one crate.

**Conceptual coupling**: unavoidable. The domain has this shape — but under item Y both halves co-locate in `semstrait-model`.

**Crate-DAG coupling**: handled, and **inverted** under item Y / V1. The authoring expression vocabulary now lives **in `semstrait-model` itself**, so authoring a `Measure` needs no cross-crate dependency at all. The dependency edge runs the other way: `semstrait-ir` depends on `semstrait-model` (a thin `ir → model` edge, V1) to consume the authoring types it lowers. `semstrait-model` depends only on `semstrait-common` (`[14 §9.3](../foundations/14_expressions.md)`).

**Import coupling at the call site**: **eliminated**. One `use semstrait::prelude::*;` brings both entity types and the expression DSL into scope (both from `semstrait-model`). The author types neither `semstrait_model::…` nor `semstrait_ir::…` ever.

**The split that makes this work cleanly** (item Y / V1–V6):

| Concern | Lives in |
|---|---|
| The shape of the **authoring** expression forms (`Expr<L>`, `Unresolved{Semantic,Physical}Expr`) | `semstrait-model` |
| The shape of entity types (`Measure`, `Dimension`, `Metric`) | `semstrait-model` |
| Construction of authoring expressions via parsing **and** DSL constructors | `semstrait-model` |
| The traversal trait family (`Tree` / `Visitor` / `Rewriter`) | `semstrait-common` (V5) |
| The **resolved** expression (`PhysicalExpr`), registry, return-type rules | `semstrait-ir` |
| Resolution / lowering (`model::Unresolved*` → `ir::PhysicalExpr`) | `semstrait-manifest` |

`semstrait-model` is the **owner** of the authoring expression surface — entity types and the expressions they carry sit in one crate, with no dependency on resolution machinery. `semstrait-ir` is the downstream consumer that lowers those authored values into the resolved `PhysicalExpr`. The model crate's contract is "produce well-formed authored expressions that pass structural rules"; the resolution contract belongs to `semstrait-ir` and `semstrait-manifest`.

This realises the item-Y inversion ratified in `[14 §9.3](../foundations/14_expressions.md)` (V1); the 39b reservation makes the *model-owns-authoring* posture explicit on the facade side.

---

## 8. Composition point vs implementation point

The user-facing fluent surface lives at the facade for one reason: **the facade is the only crate that depends on every sub-crate**. It is the workspace's terminal node in the DAG. Composition happens at the top; implementation happens in sub-crates.

- **Composition point** (`semstrait` facade): the prelude, the `semstrait::builder()` re-export, the optional `run` convenience. Adds zero new types per `[39 §1.5](39_semstrait_facade.md)`.
- **Implementation point** (sub-crates): every type definition, every method body, every algorithm. Each sub-crate has exactly one job per §3.

Equivalent statement: the facade is **discoverability infrastructure**. It reorganizes upstream names into ergonomic groupings (the prelude) and provides one-shot conveniences for the common cases. It does not invent vocabulary.

---

## 9. Open shape decisions reserved for clause-level ratification

These are the decisions that affect *base abstractions* and *control-flow shape*, not concrete ergonomics. Ergonomic decisions (builder method signatures, optional method shortcuts, namespace newtypes for type-family methods like `.str()` / `.dt()`) live in the per-crate API docs and can be added MINOR-additively.

| ID | Decision | Status |
|---|---|---|
| **D1** | `Compile` / `Query` / `Adapt` extension-trait pattern (§5) — accepted as the v1 standard for cross-crate stage transitions | Open |
| **D2** | `query()` as the verb for compile → plan transition (vs `plan()` / `select()`) | Open — recommended `query()` |
| **D3** | `Query` extension trait returns `RequestBuilder` (Polars-flavored) vs takes a `Request` argument (DataFusion-flavored) — affects the v1 fluent UX shape | Open — recommended `RequestBuilder` |
| **D4** | Prelude includes the three transition traits, the expression DSL (`expr_fn::*`), and the operator-overloaded types | Open — recommended yes (all three categories) |
| **D5** | `semstrait::builder()` re-exports `SemanticModel::builder()` at the facade crate root | Open — recommended yes |
| **D6** | `execute()` ownership — method on `EngineArtifact` (engine-bound) vs method on `SemanticPlan` (plan-bound, dispatches via adapter) | Open — recommended on `EngineArtifact` |

Decisions explicitly deferred to post-v1 (not in this reservation):

- Inline DSL parser shape (still possible to ship `Declarative` form alone in v1; reserved behind a feature flag if added later).
- Namespace newtypes for type-family methods (`col("x").str()…` Polars-style).
- SQL-frontend boundary parser (separate `semstrait-sql` crate, post-v1).
- Streaming / incremental compile.

---

## 10. Required follow-ups in [39](39_semstrait_facade.md)

This reservation implies the following updates to `39` once D1–D6 ratify (each is its own clause):

1. **`39 §3.2`** — prelude inventory adds:
   - `Compile`, `Query`, `Adapt` (the three transition traits, from §5).
   - `RequestBuilder` (returned by `Query::query`).
   - `expr_fn::*` (free constructors).
   - `UnresolvedSemanticExpr`, `UnresolvedPhysicalExpr`, `Accessor` family (authoring expression types from `semstrait-model`; item Y / V3 / V6).
   - `EngineAdapter`, `CatalogProvider` (the two trait inputs to stage transitions).
2. **`39 §4`** — adds `semstrait::builder()` free-function re-export of `SemanticModel::builder()`. Preserves the existing `run(yaml, …)` convenience (different mental model, YAML-first).
3. **`39 §1.5`** — the zero-new-logic principle holds throughout. The fluent surface emerges entirely from re-exports + upstream extension traits + the orphan-rule-co-located DSL.

---

## 11. Cross-references

- [`14_expressions.md`](../foundations/14_expressions.md) — `Expr<L>`, authoring `Unresolved{Semantic,Physical}Expr` (model) / resolved `PhysicalExpr` (ir), expression DSL, per-crate placement (item Y).
- [`14a_function_catalog.md`](../foundations/14a_function_catalog.md) — `CanonicalFn`, `FunctionRegistry`, `FunctionSpec`.
- [`32_semstrait_model.md`](32_semstrait_model.md) — entity types and parse-time structural rules (the model crate's contract).
- [`33_semstrait_manifest.md`](33_semstrait_manifest.md) — `compile`, `SemanticManifest`; will host the `Compile` trait.
- [`34_semstrait_planner.md`](34_semstrait_planner.md) — `plan`, `Request`; will host the `Query` trait and `RequestBuilder`.
- [`35_semstrait_ir.md`](35_semstrait_ir.md) — canonical-IR layer; owns `SemanticPlan`, `PlanNode`, `EngineArtifact` plus (per `[14](../foundations/14_expressions.md)`) the expression layer.
- [`36_semstrait_adapter.md`](36_semstrait_adapter.md) — `EngineAdapter`; will host the `Adapt` trait.
- [`38_semstrait_api.md`](38_semstrait_api.md) — `SemStrait` orchestrator at the API layer.
- [`39_semstrait_facade.md`](39_semstrait_facade.md) — formal facade contract.

## Testing Strategy

- Base abstraction inventory completeness (§2 table: 31 abstractions across 9 crates).
- Per-crate responsibility split (§3 table: Owns / Does NOT own — explicit bidirectional ownership statements for each crate).
- Control-flow stages (§4 table: build/compile/query/adapt/execute — verb, input, output, owning crate, transition mechanism).
- Stage-transition mechanism (§5: Compile/Query/Adapt extension traits exist and are re-exported through prelude; pattern uniformity).
- Prelude composition by category (§6 table: 13 categories, each marked Reserved or Implicit; entity types, expression DSL, transition traits, diagnostic primitives).
- Model↔expr coupling resolution (§7: one `use semstrait::prelude::*;` brings both entity types and expression DSL into scope; no `semstrait_model::…` or `semstrait_ir::…` imports needed).

See the shared behavioral harness: [`../testing/50_behavioral_contract_gates.md`](../testing/50_behavioral_contract_gates.md)
