---
id: 36-adapter-crate-charter
status: ratified
version: 2.1
impl: in-progress
---

# semstrait-adapter Crate Charter

This leaf owns the semstrait-adapter crate charter including its module layout, purpose/scope/layering contract, and the EngineAdapter trait with its signature, invariants, AdapterId constants, and adapt/emit verbs.

> **Status:** ratified. `36` nails down the public surface of `semstrait-adapter` — the crate that turns a `SemanticPlan` (`35`) into an `EngineArtifact` (`35 §7`). No new IR vocabulary is introduced; `36` refines `35`'s adapter-artifact shapes with the *emission* contract, the *dialect* operational trait, the *adapter roster*, the *function / type rewrite* pipelines, and the `AdaptErrorKind` surface (typed-kind discipline per `30 §5` / `31 §3`). All engine-identity branching in the workspace is confined behind `EngineAdapter` per I3.

## 1. Purpose, Scope, and Layering

`semstrait-adapter` is the **engine-boundary crate**. Every piece of engine-specific knowledge in the workspace — dialect quirks, SQL idioms, Substrait wire format, per-engine function rewrites, per-engine type spellings, capability declarations — lives behind the `EngineAdapter` trait here or in a per-engine crate that implements it. No canonical-layer crate (`semstrait-common`, `semstrait-ir`, `semstrait-model`, `semstrait-manifest`, `semstrait-planner`) branches on engine identity (I3); every such branch resolves to an `EngineAdapter` dispatch.

### 1.1 What `semstrait-adapter` OWNS

- The `EngineAdapter` trait (§3) — the single abstraction every downstream consumer dispatches through.
- The `DialectEmit` operational trait (§4) — per-dialect SQL rendering: identifier quoting, function rewriting, `DataType` spelling, cast semantics, null-ordering syntax, AsOf-join emission.
- The v1 built-in adapter roster (§5): `AnsiSqlAdapter`, `DataFusionSqlAdapter`, `DuckDbSqlAdapter`, `SparkSqlAdapter`, `SubstraitAdapter`. Per-engine adapters live in dedicated crates (`semstrait-adapter-datafusion`, …) per `30 §10.2`; `semstrait-adapter` carries the trait + shared emission scaffolding + the two baseline impls that depend on no engine crate (`AnsiSqlAdapter`, `SubstraitAdapter`).
- `AdapterCapabilities` (§6) — the declarative capability-flag surface.
- The function-rewrite pipeline (§7) — per-adapter rewrite tier tables (Name-only / Name-remap / Structural / Unsupported) consuming `registry/functions_mapping.md` at the adapter layer.
- The type-rewrite pipeline (§8) — canonical `DataType` → engine-native textual / Substrait form consuming `registry/types_mapping.md`.
- The `adapt` / `emit` pipeline (§9) — the per-variant `PlanNode` walk that produces an `EngineArtifact`.
- The `AdaptErrorKind` typed-error enum (§10) — identified by variant identity per `30 §5`; legacy `ADAPT_E_*` / `ADAPT_W_*` codes are retired. Warning-severity variants flow as `Diagnostic<AdaptErrorKind>` with `Severity::Warning` per `31 §3`.
- The `AdapterRegistry` (§11) — `OnceLock`-backed process-global dispatch table keyed by `AdapterId` / `DialectId`.
- SQL-emission safety scaffolding (§14) — quoting helpers, literal-escape helpers, the injection-safety audit.

### 1.2 What `semstrait-adapter` does NOT own

- **Planning strategy, optimization, plan-tree construction.** `semstrait-planner` (`34`) owns plan production and optimization. `semstrait-adapter` is pure consumer.
- **The `SemanticPlan` tree shape, `PlanNode` variants, `EngineArtifact` / `EnginePlan` / `SqlArtifact` / `DialectId` structural types.** Ratified in `semstrait-ir` (`35`). `36` consumes and emits those types but does not define them.
- **The canonical function catalog.** Ratified in `14a`. Adapters consult the sealed `FunctionRegistry` at `function_registry()` (`31 §9.1`) and the per-engine mapping at `registry/functions_mapping.md`; they do not own the canonical set.
- **SemanticManifest construction, name resolution, or catalog I/O.** `semstrait-manifest` (`33`) owns the SemanticManifest; `semstrait-catalog` (`37`) owns I/O. The adapter receives the `SemanticManifest` as a read-only borrow alongside the `SemanticPlan` (§3.1).
- **Per-engine runtime integration.** Actually executing a `SqlArtifact` or `EnginePlan` against a live engine is **out of scope** — that work lives in executor shims one layer above `semstrait-adapter`. `36` produces the artifact; someone else executes it.
- **Authoring-layer YAML, expression parsing.** The authoring expression surface (bare `Expr<L>` fields + field-level parse/serde, `14 §6`) / parse dispatch live in `semstrait-model` (`32`).

### 1.3 The three axes `36` confines

Three independent axes of engine-variation meet in this crate. `36` is where each is declared and confined:

1. **Emission mode** — SQL text (`EngineArtifact::Sql(SqlArtifact)`) vs structured IR (`EngineArtifact::Plan(EnginePlan)`). Selected per-adapter; `EngineAdapter::dialect()` returns `None` for non-SQL adapters (§3.1).
2. **Dialect** — for SQL-emitting adapters, the per-engine SQL variation axis (identifier quoting, keyword casing, function spelling, type name, cast semantics). Carried with every `SqlArtifact` via `DialectId` (`35 §7.4`). Ratified here as `DialectEmit` (§4).
3. **Engine identity** — the concrete target engine (DataFusion, DuckDB, Spark, Substrait consumer, …). Identified by `AdapterId` (§3.3). Each per-engine adapter picks its own `Dialect` (or `None` for Substrait) + its own `AdapterCapabilities` + its own `RegistryExtension` impl (`14a §7`).

These axes are *independent* in principle — two adapters MAY share a `DialectId` (`AnsiSqlAdapter` and a hypothetical `TrinoSqlAdapter` could both claim `DialectId::ANSI` as a baseline) — but in v1 every `DialectId` has exactly one built-in adapter that emits it.

### 1.4 Design posture — sync, no I/O, deterministic

`semstrait-adapter` is deliberately **pure**:

- **Zero I/O surface.** Concrete I11 guarantee. No `std::fs`, no `std::net`, no `reqwest`. Every method on `EngineAdapter` is an in-memory transformation.
- **Zero async.** Every `EngineAdapter` method is `fn`, not `async fn`. I6 guarantee. `adapt` runs on the caller's thread and returns on the caller's thread; no awaits, no futures, no scheduler integration.
- **Deterministic given `(SemanticPlan, SemanticManifest, adapter config)`.** Two `adapt` calls with the same inputs produce byte-identical `EngineArtifact`s. Enables content-addressable caching of emitted SQL / Substrait per `00 §9` I4.
- **No hidden global state inside `adapt`.** The `AdapterRegistry` (§11) is used for *dispatch* (picking which adapter runs), not as a mutable workspace during `adapt`. The `FunctionRegistry` (`31 §5.2`) is consulted as read-only. No `thread_local!` side-state.

## 2. Module Layout

Top-level `pub mod` structure for `semstrait-adapter` itself. Per-engine crates (`semstrait-adapter-datafusion`, `-duckdb`, `-spark`, `-substrait`) mirror this layout inside their own crate root; `36` is authoritative for the shared `semstrait-adapter` core.

```
semstrait-adapter
├── adapter              // EngineAdapter trait, AdapterId, debug_sql free fn
├── dialect              // DialectEmit trait, shared default-impl helpers
│   ├── ansi             //   AnsiDialect + AnsiSqlAdapter (baseline)
│   └── quoting          //   identifier-quoting + literal-escape primitives
├── capabilities         // AdapterCapabilities, Capability-helper methods
├── rewrite              // per-node rewrite pipeline; PlanBuilder-layer rewrites
│   ├── function         //   canonical function → engine form (Name-only /
│   │                    //   Name-remap / Structural / Unsupported)
│   └── type             //   canonical DataType → engine type name / Substrait
├── emit                 // SQL emitter: PlanNode → SqlArtifact; orchestrator
├── substrait            // SubstraitAdapter + proto-emission scaffolding
├── registry             // AdapterRegistry + adapter_registry() accessor
└── error                // AdaptErrorKind, UnsupportedFeatureKind
```

**Split rationale:**

- `adapter` vs `dialect` — the trait surface (what adapters DO) is decoupled from dialect rendering (how SQL-emitting adapters spell things). A Substrait-only adapter (§5.5) implements `EngineAdapter` but never touches `dialect`.
- `dialect::ansi` carries `AnsiSqlAdapter` alongside its dialect — one of the two baseline adapters every workspace build includes.
- `dialect::quoting` — identifier + literal escape primitives are the top of the SQL-injection-safety audit (§14). Isolating them lets the audit suite target one module.
- `capabilities` — standalone because every adapter declares its `AdapterCapabilities`; the shared struct lives here and is consumed by planner (`34`), API (`38`), and each adapter's `capabilities()`.
- `rewrite::function` vs `rewrite::type` — two independent pipelines; both consume registry data (`functions_mapping`, `types_mapping`) but operate on different node positions (`FunctionCall` nodes vs `Cast` targets and `PhysicalExpr.inferred_type`). Isolating limits the blast radius when a new rewrite tier lands.
- `emit` — the SQL orchestrator; traverses `SemanticPlan` and delegates per-variant rendering to `DialectEmit` methods.
- `substrait` — isolated because it's the only module that pulls in `prost` / `substrait::proto` dependencies; gated by `substrait-emit` feature on per-engine downstreams.
- `registry` — dispatch table; no emission code lives here.
- `error` — mirrors `31 §2`'s / `35 §2`'s `error` split; one stable home for `AdaptErrorKind`.

**Re-exports.** The crate root re-exports the curated surface listed in §15. Third-party adapter crates re-export their own `<Engine>Adapter` and `<Engine>Dialect` from their own crate root; `semstrait-adapter` re-exports only the shared core and the baseline `AnsiSqlAdapter` + `SubstraitAdapter`.

## 3. `EngineAdapter` Trait

### 3.1 Signature

```rust
/// The abstraction every engine-specific adapter implements. Consumed by
/// `semstrait-api` (`38`) for pipeline dispatch and by `semstrait-planner`
/// (`34`) for capability-gated plan rules. Per `00 §4.1`.
///
/// Not sealed — third-party adapter crates outside the workspace MUST be
/// able to `impl EngineAdapter` without a private escape hatch
/// (`30 §8.2`).
pub trait EngineAdapter: Send + Sync {
    /// The adapter's stable identity. Used for registry-keyed dispatch
    /// (§11) and for diagnostic reporting. See §3.3 for the roster.
    fn id(&self) -> AdapterId;

    /// The adapter's target dialect.
    ///
    /// - `Some(&dyn DialectEmit)` — SQL-emitting adapter; returned
    ///   reference is used by the generic SQL orchestrator (`emit-orchestrators.md §9.2`).
    /// - `None` — non-SQL adapter (e.g. `SubstraitAdapter`). The
    ///   default `emit` impl (§3.2) rejects with
    ///   `AdaptErrorKind::EmissionNotSupported`.
    fn dialect(&self) -> Option<&dyn DialectEmit>;

    /// Declarative capability advertisement. Consumed by
    /// `semstrait-planner` (`34`) for capability-gated plan rules, by
    /// `semstrait-api` (`38`) for pre-`adapt` feasibility checks, and by
    /// cross-engine test harnesses. Not consumed inside `adapt` itself;
    /// the authoritative feasibility check is the emission path
    /// (`AdaptErrorKind::Unsupported*`). Per `Q-ADAPT-002`.
    fn capabilities(&self) -> &AdapterCapabilities;
```

The trait's `adapt` method is the primary stage entry point:

```rust
    /// Produce an engine-ready artifact from a canonical plan.
    ///
    /// Stage-entry-point shape per `30 §7`: fail-fast tuple
    /// (`(fatal, warnings)` on `Err`; `(artifact, warnings)` on `Ok`).
    /// Adapters that produce no warnings return an empty
    /// `Diagnostics<AdaptErrorKind>` on the success arm — the typed-
    /// kind discipline obviates the prior `adapt_with_diagnostics`
    /// extension.
    ///
    /// Invariants:
    /// - Synchronous (I6).
    /// - No I/O (I11). The `SemanticManifest` is consumed read-only; no
    ///   catalog-provider traffic, no filesystem reads.
    /// - Deterministic given `(plan, manifest, self)`: two invocations
    ///   with identical inputs produce byte-identical outputs and
    ///   byte-identical warning sequences.
    /// - Takes `plan` and `manifest` by borrow so the caller retains
    ///   ownership; neither is mutated.
    /// - `plan.diagnostics` is NOT consulted by default — the adapter
    ///   emits against `plan.root` on the assumption that the planner
    ///   raised every needed warning already.
    fn adapt(
        &self,
        plan: &SemanticPlan,
        manifest: &SemanticManifest,
    ) -> Result<
        (EngineArtifact, Diagnostics<AdaptErrorKind>),
        (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>),
    >;
```

The trait continues with the `emit` method:

```rust
    /// The SQL-specialized form of `adapt`. Produces `SqlArtifact`
    /// directly; the default impl rejects for non-SQL adapters.
    ///
    /// Motivation: every SQL-emitting adapter implements `adapt` by
    /// calling `emit` and wrapping in `EngineArtifact::Sql`; exposing
    /// `emit` separately avoids the `EngineArtifact::Sql(..)` unwrap at
    /// every caller site that already knows it wants SQL.
    fn emit(
        &self,
        plan: &SemanticPlan,
        manifest: &SemanticManifest,
    ) -> Result<
        (SqlArtifact, Diagnostics<AdaptErrorKind>),
        (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>),
    > {
        let _ = (plan, manifest);
        Err((
            Diagnostic::error(AdaptErrorKind::EmissionNotSupported {
                adapter: self.id(),
            }),
            Diagnostics::empty(),
        ))
    }
}
```

### 3.2 Trait-level invariants

Every `impl EngineAdapter` MUST uphold:

- **I-ADAPT-1 — sync.** No `.await`, no `block_on`. `adapt` / `emit` run on the caller's thread. I6 guarantee per `30 §9`.
- **I-ADAPT-2 — no I/O.** No filesystem access, no network access, no catalog-provider calls. I11 guarantee. The `SemanticManifest` the adapter consumes already contains every piece of metadata needed; drift checks live outside `adapt` per `00 §9` I11.
- **I-ADAPT-3 — deterministic.** Two calls with identical `(plan, manifest)` must produce byte-identical `EngineArtifact`s. UUID generation, timestamp capture, randomized ordering — all forbidden inside `adapt`.
- **I-ADAPT-4 — quoting mandatory.** Every identifier embedded in `SqlArtifact.text` passes through the adapter's `DialectEmit::quote_identifier` (§4.3). Every literal value passes through `DialectEmit::quote_literal` or equivalent. String concatenation of unquoted identifiers is a soundness bug (§14).
- **I-ADAPT-5 — error-first fallback.** If a `PlanNode` variant, a `FunctionCall` name, or a `DataType` is not representable in the adapter's target, `AdaptError::Unsupported*` fires at `adapt` time. Silent truncation, stub emission, runtime-deferred panics — all banned. Matches `14a §6.3`'s hard-error policy for `UnsupportedFunction`.
- **I-ADAPT-6 — structural-mapping conformance.** SQL-emitting adapters render each `PlanNode` variant per the structural conventions of §9; the generic orchestrator assumes conformance and composes clauses accordingly. Substrait adapters conform to the mapping table at `35 §10.2`.

### 3.3 `AdapterId`

```rust
/// Stable identity newtype, paralleling `DialectId`'s posture at
/// `35 §7.4`. Construction is crate-private; consumers compare against
/// `pub const` identities.
///
/// Newtype-over-stable exception per `30 §4.3`: no `#[non_exhaustive]`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AdapterId(&'static str);

impl AdapterId {
    pub const fn as_str(self) -> &'static str { self.0 }

    pub const ANSI_SQL:        AdapterId = AdapterId("ansi-sql");
    pub const DATAFUSION_SQL:  AdapterId = AdapterId("datafusion-sql");
    pub const DUCKDB_SQL:      AdapterId = AdapterId("duckdb-sql");
    pub const SPARK_SQL:       AdapterId = AdapterId("spark-sql");
    pub const SUBSTRAIT:       AdapterId = AdapterId("substrait");
}
```

Adding a new `pub const AdapterId` — e.g. `AdapterId::TRINO_SQL`, `AdapterId::CLICKHOUSE_SQL` — is MINOR per `30 §11.1`. Third-party adapter crates declare their own constants on their own type; they MAY NOT add `pub const`s to `AdapterId` directly (construction is crate-private). The crate-private construction mirrors `CanonicalFn` / `DialectId` (`31 §5.1` / `35 §7.4`).

### 3.4 Diagnostics flow (no separate extension)

`adapt` and `emit` carry warnings inline through their fail-fast tuple
return shape (§3.1). The historical `adapt_with_diagnostics(&mut Vec<Diagnostic>)`
extension is **retired** by the workspace-wide diagnostic-shape decision in
`30 §7`: every adapter receives a slot for warnings on the success arm of
`Result` and a `(fatal, warnings_so_far)` tuple on the `Err` arm. Adapters
that produce no warnings return an empty `Diagnostics<AdaptErrorKind>`;
adapters that do (e.g. DuckDB precision clamping, Spark structural
rewrites) accumulate them in the same vector and propagate.

### 3.5 `debug_sql` free function

```rust
/// Render a `SemanticPlan` as ANSI-baseline SQL for debugging. Not an
/// `EngineAdapter` method — adapter-independent rendering per
/// `Q-ADAPT-004`. Consumed by test harnesses, logging tools, and the
/// `semstrait-api` plan-inspection surface.
///
/// Bare-kind shape per `31 §3.1` construction-site convention; callers
/// (debug paths, inspection UIs) wrap into `Diagnostic<AdaptErrorKind>` if
/// they need a stage-boundary location.
pub fn debug_sql(plan: &SemanticPlan, manifest: &SemanticManifest) -> Result<String, AdaptErrorKind>;
```

Routes through `AnsiSqlAdapter::emit` internally. Non-SQL adapters (`SubstraitAdapter`) have no native "debug_sql" — callers use this free function for any plan, regardless of target adapter. Keeping it outside the trait avoids forcing every adapter to provide a SQL renderer (Substrait adapters genuinely lack one). Current code exposes `debug_sql` as a trait method with a default impl; the migration to a free function is tracked as `[TD-ADAPTER-DEBUG-SQL-FREE-FN]`.

## Testing Strategy

- EngineAdapter trait openness (third-party impls compile against the public surface, no sealed escape hatch per §3.1)
- AdapterId roster round-trip (all 5 v1 constants stable, as_str() bijection per §3.3)
- adapt/emit signature shapes (fail-fast tuple per §3.1, warnings slot on both arms)
- Trait invariants I-ADAPT-1 through I-ADAPT-6 (sync/no-I/O/deterministic/quoting-mandatory/error-first/structural-conformance per §3.2)
- debug_sql produces ANSI baseline (per §3.5, byte-identical snapshot)
- Module layout coverage (§2 tree structure mirrors actual crate file layout)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
