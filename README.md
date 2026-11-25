# semstrait

> **Note**: This README describes the **current code state**. For the **target architecture**, read [`docs/adr/`](docs/adr/) — architecture decisions are the only ratified record, and terminology is fixed in [`CONTEXT.md`](CONTEXT.md). Sections below will diverge from the target until the phased migration lands.

The ideology of semstrait is to bring typed `DataKind`s, mapped to engine-agnostic query-plan generation via `Substrait`.

semstrait implements a unified `Internal Representation (IR)` grounded in ANSI standards and the most common SQL syntax and function shapes. This unlocks interoperability and strategy-driven expression rewrites that use the semantics of `adapters`.

> **Two views below.** [Current State](#current-state) is what the workspace builds *today*; [Target State](#target-state) is the architecture we are migrating toward. They differ on purpose. The target is recorded in `docs/adr/`.

---

## Current State

The library is in an active development phase, focused on interoperability and proving the core assumption: that `DataKind`s — `Dataset`, `Grainset`, `Joinset`, `Unionset` — give both ease of use and an optimal semantic query planner.

At this phase semstrait is driven by a **single** YAML document that configures downstream components to plan and adapt query requests. The current focus is type safety, transparent structural rules, and ease of setup and testing.

Behavioral contracts are carried by tests. Architecture decisions live in `docs/adr/` and terminology in `CONTEXT.md`; together they are the whole ratified record.

### What builds today

The Cargo workspace is **eleven crates**, and every crate in `crates/` is a member — the model + manifest layer, the typed SQL vocabulary whose data types they use, the planner's public contract, the Substrait format adapter, the DataFusion engine adapter, the orchestrator with its CLI and REST transports, and the `semstrait` facade. The crates publish together at one version: each internal entry of the root `[workspace.dependencies]` pins it exactly (ADR 0009).

| Crate | Role today |
|---|---|
| `semstrait-common` | Shared primitives with no semantic opinion: `EntityId`, `Location` / `SourceId` / `Span`, typed `Diagnostic`s, the stage-agnostic traversal traits (`Tree` / `Visitor` / `Rewriter`), the `Dialect` name, `Granularity`, and an optional `io` module (`FileSystem` transport). |
| `semstrait-sql` | The typed SQL vocabulary (ADRs 0085–0092): the canonical `DataType` with its canonical and Substrait declarations, type families, widening, casts and type systems, the function declarations with their sealed registry, and the expression tree (`Expr`, with `Literal` and `Column(NamedReference)` as nodes, its leaves and the literal checks). Depends only on `-common`; `-model`, `-ir` and `-manifest` import its types directly (ADR 0090, rule 6). |
| `semstrait-model` | Authoring surface. Typed structs + builders mirroring the YAML shape, the aliases of the two unresolved trees over `-sql`'s tree (`UnresolvedSemanticExpr` / `UnresolvedPhysicalExpr`) with their builders, lossless serde round-trip (JSON + YAML). Parse builds each call through its declaration in `-sql`'s registry, and the model defines no expression, call or function type (ADR 0089, rules 4 and 6). |
| `semstrait-ir` | The plan: plan nodes (`PlanNode`, `ScanNode`, …), `SemanticPlan`, `Schema` / `SchemaColumn`, the plan's leaf `Parameter` and its tree `PhysicalExpr` (`-sql`'s `Expr<Parameter>`), an aggregation node's entries as `-sql`'s `AggregateCall`, and the `SemanticGraph` value-type read traits. Defines no expression, function or data type, depends on no model crate, and takes its names from `-common` (ADR 0094, rule 9) and `FileFormat` from `-catalog`, and re-exports no type it does not own: each crate imports `-sql`'s and `-common`'s types from them directly (ADR 0090, rule 6). No crate reads the `SemanticGraph` types, which stay as they are (ADR 0090, rule 9): the planner's lookups walk the manifest, and a later semantic graph, built at setup, replaces them (ADR 0064, *The seam a cached graph replaces*). |
| `semstrait-manifest` | Compile + artifact. `compile()` lowers the model (+ optional catalog) into a `SemanticManifest` (interned identity, physical expression pool, coverage bitmaps); `-common`'s `Repository` / `FileSystemRepository` persist it via deterministic JSON. Re-exports no type it does not own (ADR 0090, rule 6). |
| `semstrait-catalog` | `CatalogProvider` trait (async, object-safe) for metadata-source integrations, a `NoopCatalogProvider`, and, behind features, the Iceberg REST (`iceberg`) and Databricks Unity (`unity`) providers and AWS Secrets Manager credentials (`aws`). Depends only on `-common`. |
| `semstrait-planner` | The planner's public contract: `Request`, `RequestContext`, `PlanError`, `Advisory` and `Planner::plan(&manifest, &request, &context)` (ADRs 0062–0064, 0073). It plans a request over a Dataset, a Joinset, a Unionset or a Grainset, at every depth of composition (ADR 0054; ADR 0073; ADR 0074, *A Joinset*, *A Unionset*, *A Grainset* and **A composition that is a constituent**; ADR 0079; ADR 0084); `crates/semstrait-planner/README.md` gives each form with its ADR. The request selects Dimensions, each at a requested granularity or not, Measures, and Metrics that are row-level, run over the final aggregates, are in the formula push-down's pattern or take two aggregate steps (ADR 0083, rules 1 and 4; ADR 0074, **Two aggregate steps**); it may filter with `where` and activated named filters, and order and page its answer. A semi-additive Measure reads its rows through its `rank` marker (ADR 0076). Links no engine and no adapter (ADR 0068). |
| `semstrait-adapter` | The Substrait format adapter (ADR 0068): `SubstraitProducer`, which lowers a `SemanticPlan` that `validate` accepted to `substrait::proto::Plan` (`substrait` 0.63, spec v0.85.0); the unsealed `SubstraitRules`, whose every hook defaults to the identity or the faithful answer; and `explain`, which prints the plan's explain copy with each node's trace as `+ Opt:` lines (`substrait-explain` 0.9). It holds no per-engine rules and links no engine, not even as a dev-dependency. |
| `semstrait-adapter-datafusion` | The DataFusion engine adapter (ADR 0068): `DataFusionRules`, the `SubstraitRules` for DataFusion 55's Substrait consumer, which write a table with no catalog as `[namespace, name]`, a files read of *n* paths as *n* one-file reads under one `UNION_ALL`, and a join's `post_join_filter`, which the consumer refuses, as a `FilterRel` above the `JoinRel`; write each later branch of a union that the consumer reads as no projection under an identity `ProjectRel`, so that DataFusion names it as the union; rewrite `regexp_extract` (ADR 0087, rule 23) and `concat` (ADR 0087, rules 11 and 23) in the canonical vocabulary; and write `regexp_like`, `regexp_instr`, `regexp_replace`, `date_trunc` (ADR 0072) and `extract` in DataFusion's own functions; `crates/semstrait-adapter-datafusion/README.md` gives each rule with its ADR. Its library links no engine. DataFusion is a dev-dependency of its tests only: hand-built plans, and the oracle, which compiles each fixture model against its data declaration and runs each case through `Planner::plan`, or through the function that its `cases!` line names, and compiles each corpus model in `test_data/` against its own. |
| `semstrait-api` | The orchestrator: `Semstrait`, which `SemstraitBuilder` builds with one optional `CatalogProvider` and one `FileSystem`, and which holds an in-memory manifest store. Its verbs compile a model (`compile_from_yaml`, `compile_from_model`), keep a manifest in the store (`exists`, `save_manifest`, `load_manifest`), and `validate`, `explain` and `plan` a `Request` in a `Dialect`, the last to `substrait::proto::Plan`. `load` gives a validated manifest from an `Input` (model YAML, a model path, a manifest or a manifest path) through the store; `RequestPart` reads the Request wire form, a YAML or JSON document; `with_catalogs` connects one `catalogs.yaml` entry. Behind features: `DataFusionRules` for `Dialect::DataFusion` (`datafusion`), Iceberg REST catalogs (`iceberg`), `s3://` paths and Secrets Manager auth (`aws`), the axum `router` (`rest`) and the `semstrait` binary (`cli`); `crates/semstrait-api/README.md` gives each. Behind `rest` it names `semstrait-ir`'s `SemanticRole`; it links no engine. |
| `semstrait` | The facade (ADR 0009): it re-exports the types a caller names from `-api`, `-planner`, `-model`, `-manifest`, `-catalog` and `-common`, and has one free function, `run`, which loads an `Input` and plans a `Request` to a Substrait plan. It owns no type; each of its features forwards to one of `-api`'s. |

### Dependency edges (today)

```
common             → (nothing internal)
sql                → common
model              → common, sql
ir                 → common, sql, catalog
catalog            → common
manifest           → common, sql, ir, model, catalog   (compile() lives here)
planner            → common, sql, ir, manifest         (plan() lives here)
adapter            → common, sql, ir, catalog          (produce() and explain() live here)
adapter-datafusion → common, sql, ir, adapter          (DataFusionRules lives here)
api                → common, model, manifest, catalog, planner, adapter; optional: adapter-datafusion, ir  (Semstrait and load() live here)
semstrait          → common, model, manifest, catalog, planner, api      (the facade: its re-exports and run())
```

`semstrait-ir` depends on no model crate and takes its names from `-common`
(ADR 0094, rule 9); it depends on `catalog` for `FileFormat` only, on a scan's
reader (ADR 0067). Each crate depends directly on every crate whose types it
uses, imports each type from the crate that owns it, and re-exports none that it
does not own (ADR 0090, rule 6); the `semstrait` facade re-exports by design
(ADR 0009). `api`'s edges to `adapter-datafusion` and `ir` are optional, behind its
`datafusion` and `rest` features; the facade has no `ir` edge. Otherwise, inside
`api`, the semantic plan passes from the planner to the adapter by inference.

### Pipeline (today)

```
YAML model
  → model     : parse (lossless surface AST)
  → manifest  : compile() — resolve names, lower SemanticExpr → PhysicalExpr,
                write artifact (interned ids + physical expr pool + coverage bitmaps)
  → planner   : plan() — a SemanticPlan for a request over a Dataset, a Joinset,
                a Unionset or a Grainset
  → adapter   : produce() — a validated SemanticPlan → substrait::proto::Plan;
                explain() — its explain copy as text, with the trace
```

---

## Target State

The architecture we are migrating toward, as recorded in `docs/adr/`; this is only the high-level shape.

### Core principles

- semstrait defines the shape and semantics of entities — the vocabulary and its meaning.
- semstrait does **not** execute engine queries or expressions.
- Two main public contracts and crates:
  - `semstrait-model` — the authoring contract & schema.
  - `semstrait-ir` — the adaptability contract: the plan nodes, which hold `semstrait-sql`'s expression tree, calls and data types.
- `semstrait-model` is a standalone, self-contained modeling surface — applying structural rules over the input shape and expressions in a lightweight, serializable form.
- The library uses the sans-io pattern to define I/O interactions between modules.
- All data shapes are expressed through the `DataKind` abstraction, subtyped into two categories: Simple (leaf) and Complex (behavioral & strategy-oriented).
- Strict & non-strict modes — flexibility or control.

### Crate map

| Crate | Visibility | Responsibility |
|---|---|---|
| `semstrait-common` | shared primitives | Types with no semantic opinion: identifiers, source spans, diagnostics plumbing, the stage-agnostic traversal traits (`Tree` / `Visitor` / `Rewriter`), and `Granularity`. Nothing domain-shaped. |
| `semstrait-model` | **public contract** | Authoring surface (ADR 0001). Structs/traits mirroring the YAML shape (with YAML-side sugar). The aliases of the two unresolved trees over `semstrait-sql`'s tree and their builders. Lossless round-trip for the authoring model. |
| `semstrait-ir` | **public contract** | The plan: plan nodes, `SemanticPlan`, `Schema`, and the plan's leaf `Parameter`, over `semstrait-sql`'s tree, calls and data types. Re-exports no type it does not own (ADR 0090, rule 6). |
| `semstrait-manifest` | **public contract** (narrow) | Compiled artifact format: versioned envelope, interned identifier tables, sections, pre-computed join lattice, accessible interfaces and reachability maps. |
| `semstrait-planner` | internal pipeline | Plans each request against a `semstrait-manifest` and emits `ir::SemanticPlan`. Its lookups walk the manifest, and a later semantic graph, built from the manifest at setup, replaces them (ADR 0064, *The seam a cached graph replaces*). |
| `semstrait-adapter` | consumers | The format adapter: `SubstraitProducer::produce(&SemanticPlan)` gives a `substrait::proto::Plan`, under an engine adapter's `SubstraitRules` (ADR 0068, *Artifacts*). |
| `semstrait-adapter-<engine>` (per engine) | consumers | An engine adapter: one engine's `SubstraitRules`, in its own crate, which depends on `semstrait-adapter` (ADR 0068, *Decision Outcome*). |
| `semstrait-catalog/*` (per catalog) | producers | Implements metadata-catalog integrations with the manifest/compiler. Producer of real metadata to reconcile the model. |

Future crate (planned — not pre-built):

- `semstrait-compiler` — ties model & manifest together: resolves the model, lowers it, and produces the durable artifact (`SemanticManifest`). *Today this lowering lives inside `semstrait-manifest::compile()`.*

### Dependency rules

Allowed edges ("depends on"):

```
compiler         → model, ir, manifest, sql, common
planner          → manifest, ir, sql, common
adapter          → ir, catalog, sql, common
adapter-<engine> → adapter, ir, sql, common
manifest         → ir, model, catalog, sql, common
catalog          → common
ir               → catalog, sql, common
model            → sql, common
sql              → common
common           → (nothing internal)
```

`ir` does not depend on `model`: it takes its names from `common` (ADR 0094,
rule 9). Nothing reaches `model` through `ir`; the crates that read the YAML
forms (`compiler`, `manifest`) name it directly.

### Visual overview

Read every arrow as "depends on", pointing downward. `model` sits *beside* `ir`:

```
┌────────────────────┐      ┌───────────────────┐      ┌────────────────────────┐
│ semstrait-compiler │      │ semstrait-planner │      │   semstrait-adapter    │
│   lowering pass    │      │ graph + planning  │      │   + adapter-<engine>   │
└──┬─────────────┬───┘      └─────────┬───────┬─┘      └────────┬───────────────┘
   │             │                    │       │                 │
   │             ▼                    ▼       │                 │
   │       ┌──────────────────────────────┐   │                 │
   │       │      semstrait-manifest      │   │                 │
   │       │       artifact format        │   │                 │
   │       └────────────────┬─────────────┘   │                 │
   ▼                        ▼                 ▼                 ▼
  ┌──────────────────────────────────────────────────────────────┐ ┌────────────────┐
  │  semstrait-ir                                                │ │ semstrait-model│
  │  Adaptation Contract · SemanticPlan · plan nodes             │ │ YAML · aliases │
  └──────────────────────────────┬───────────────────────────────┘ └───────┬────────┘
                                 │                                         │
                                 └────────────────────┬────────────────────┘
                                                      │
                                                      ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  semstrait-common                                                               │
  │  spans · identifiers · Tree / Visitor / Rewriter · Granularity · io?            │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

The edge list above is the authority; the diagram shows the layering rather than
every edge. `compiler` and `manifest` also name `model` directly, left undrawn
for clarity. `sql` is left undrawn too: every crate but `catalog` names it
directly, and it names only `common` (ADR 0090, rule 6).

### Expressions (the dual-tree split)

| Entity | Crate | Notes |
|---|---|---|
| Surface `Expr` (authoring) | `sql`, over `model`'s aliases | `semstrait-sql`'s `Expr` over its `SemanticLeaf` or `PhysicalLeaf`, which `model` holds as `UnresolvedSemanticExpr` and `UnresolvedPhysicalExpr` (ADR 0089, rule 6). Name-based, **unresolved** identifiers. Span-carrying, serde-friendly. Round-trips losslessly to the YAML the author wrote, sugar intact (YAML tags map to surface AST nodes, e.g. `Expr::BinaryOp` and its `BinaryOpKind` stay internal to the mapping, not exposed in YAML). No evaluation, no rewrites. |
| Canonical `Expr` (resolved) | `sql`, over `ir`'s leaf | `semstrait-sql`'s `Expr` over the plan's leaf `Parameter`, which `ir` holds as its alias `PhysicalExpr`; `ir` defines no expression of its own (ADR 0090, rule 1). A column is a name path, and a call holds its function's declaration from the registry (ADR 0085, rule 9; ADR 0091, rule 1). Desugared, normalized, free to grow planner annotations (types, additivity) without touching the public authoring API. |
| Lowering surface → canonical | `compiler` | Name resolution, desugaring, validation. Natural home for diagnostics with good spans. *(Today: `manifest::compile()`.)* |
| Tree-rewrite traversal | `common` | Generic bottom-up rewrite framework (`Tree` / `Visitor` / `Rewriter`, with `Tree::transform`). Stage-agnostic, so ADR 0001 sinks it to `semstrait-common` at the foot of the dependency graph; each crate imports it from there (ADR 0090, rule 6). Concrete rewrite/folding passes are future planner-side work. |

### Control flow (pipeline)

```
YAML model
  → model      : parse (lossless surface AST)
  → compile    : resolve names, desugar, validate; lower surface SemanticExpr → PhysicalExpr
  → manifest   : write compiled artifact (envelope + interned ids + join lattice + reachability bitmaps)
  → planner    : per query — lookups that walk the manifest, which a semantic graph
                 built at setup replaces later (ADR 0064) → SemanticPlan
  → ir         : sourcing SemanticPlan (canonical, closed, explicit casts)
  → adapters   : produce(&SemanticPlan) under an engine adapter's rules
                 → Substrait plan, SQL later (ADR 0068)
  → engine     : execution outside
```

### Invariants & litmus tests

Use these when reviewing changes; each maps to a forbidden edge or a contract property.

1. **Adapter independence.** A third party can build a working engine adapter (e.g. DuckDB, Trino) with `semstrait-adapter`, `semstrait-ir`, `semstrait-sql` and `semstrait-common` as the only semstrait crates it *declares* (the edge `adapter-<engine>` above; ADR 0068) — never declaring `model`, `compiler`, `planner`, or `manifest`. The invariant is about that declared dependency, not the build graph: under ADR 0001 `model` still compiles as `ir`'s own dependency. An adapter imports the tree, the calls and the data types from `semstrait-sql`, which owns them, because `ir` re-exports none of them (ADR 0090, rule 6).
2. **Plan closure.** An adapter unit test can construct a `SemanticPlan` from a golden fixture and produce SQL without the planner linked. If a plan needs external lookups, it is not closed.
3. **No implicit semantics.** Every coercion in a plan is an explicit cast node. If two engines could disagree about a plan's meaning, the plan is underspecified.
4. **Round-trip fidelity.** `model` parse → serialize reproduces the authored YAML (modulo formatting), sugar intact. Desugaring happens only in `compiler`.
5. **Format/structure decoupling.** Refactoring an `ir` enum does not change the manifest's on-disk format. If it does, the artifact is serializing IR structs directly — fix the envelope.
6. **Graph privacy.** Nothing outside `planner` names `SemanticGraph`. Wanting to expose it to adapters = a planning decision trying to escape into adaptation.
7. **Epoch correctness.** No code path mutates a graph across manifest fingerprints; a new fingerprint always starts a fresh graph epoch.
8. **`common` stays thin.** Every type in `common` passes: "means exactly the same thing to a YAML author and a SQL adapter." No repositories, no IO backends, no domain shapes.

---

## Documentation

Each concern has one home. Start here:

- `CONTEXT.md` — the glossary: authority for terminology.
- `docs/adr/` — architecture decisions, dependency-ordered.
- each crate's own `README.md` — module map, key types, control flow.
