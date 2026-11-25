---
id: 35-ir-adapter-artifacts
status: ratified
version: 2.1
impl: implemented
---

# IR Adapter Artifacts

The output types produced by `semstrait-adapter::adapt()` (`36`) from a `SemanticPlan`. `35` ratifies the structural shape; `36` owns the emission semantics.

## `EngineArtifact`

```rust
/// Engine-ready plan artifact. Produced by `EngineAdapter::adapt()`
/// in `36`.
///
/// Two variants covering the two adapter emission modes:
/// - `Sql` for engines that consume SQL strings (DuckDB, Spark, PostgreSQL).
/// - `Plan` for engines that consume Substrait plans natively (DataFusion,
///   some Spark paths, future Substrait-consuming engines).
#[non_exhaustive]
pub enum EngineArtifact {
    Sql  (SqlArtifact),
    Plan (EnginePlan),
}
```

## `EnginePlan`

```rust
/// Structured-IR engine output. Variants today: `Substrait(...)`.
/// Non-exhaustive per I10 — future engine IRs (e.g. direct
/// DataFusion-LogicalPlan emission, future vendor IRs) can be added
/// as MINOR variants.
#[non_exhaustive]
pub enum EnginePlan {
    /// A Substrait plan. Boxed to keep `EnginePlan`'s size moderate
    /// across platforms.
    Substrait(Box<substrait::proto::Plan>),
}

impl EnginePlan {
    /// Serialize the Substrait plan to protobuf bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, IrErrorKind>;
    /// Serialize the Substrait plan to pretty JSON.
    pub fn to_json(&self) -> Result<String, IrErrorKind>;
}
```

## `SqlArtifact`

```rust
/// Text-based engine output. Per `00 §4.1`.
#[non_exhaustive]
pub struct SqlArtifact {
    /// The emitted SQL text. UTF-8.
    pub text: String,
    /// The dialect this text targets. Consumers use this to route the
    /// text to the correct engine.
    pub dialect: DialectId,
}
```

## `DialectId` + `Dialect`

`DialectId` is **artifact-side identity only** (S7, R1; see [ir-crate-charter.md](ir-crate-charter.md) §1.5). It identifies the dialect of an emitted `SqlArtifact` and the dialect a `Dialect` impl serves; it does NOT travel on `SemanticPlan`, `PlanNode`, `Expr<L>`, `NodeMeta`, or any registry-side type. The pattern parallels dbt's `manifest.adapter_type`: a single artifact-level provenance tag that lets a downstream consumer route the artifact, without coupling the upstream model graph to the target adapter.

Per Q4.A (2026-05-21): the planner (`34`) produces a `SemanticPlan` whose entire payload is dialect-free; the adapter (`36`) is the producer of `EngineArtifact` and the only crate that stamps `DialectId`.

```rust
/// Stable identifier for a SQL dialect. Per `00 §4.1` and `36`.
///
/// Implemented as a newtype over a `&'static str` with `pub const`
/// identities per built-in adapter; adapters outside the workspace
/// register new dialects via the `Dialect` trait.
///
/// Newtype-over-stable exception per `30 §4.3`: no `#[non_exhaustive]`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct DialectId(&'static str);

impl DialectId {
    pub const fn name(self) -> &'static str { self.0 }

    // Canonical built-in dialect identities — one `pub const` per
    // in-workspace adapter per `00 §4.1`.
    pub const ANSI:       DialectId = DialectId("ansi");
    pub const DATAFUSION: DialectId = DialectId("datafusion");
    pub const DUCKDB:     DialectId = DialectId("duckdb");
    pub const SPARK:      DialectId = DialectId("spark");
}
```

### `Dialect` Trait

```rust
/// Capability / identity trait implemented by every SQL-emitting
/// adapter. Per `00 §4.1` and `36 §3`.
///
/// Not sealed — third-party adapter crates outside the workspace
/// (e.g. `semstrait-adapter-clickhouse`) MUST be able to impl
/// `Dialect` for their own `DialectId`.
pub trait Dialect {
    /// The dialect's stable identity.
    const ID: DialectId;

    /// Adapter-declared capability flags consumed by the planner's
    /// capability check (`36 §5`). Readers SHOULD NOT pattern-match
    /// exhaustively; the set is additive.
    fn capabilities(&self) -> &'static [Capability];
}
```

`Capability` is the dialect / adapter capability vocabulary. Type definition lives in `35`; per-adapter rosters and variant-addition drivers live in `36` (Q-IR-010, 2026-05-21).

## `Capability`

```rust
/// Cross-boundary capability vocabulary. Type definition lives here;
/// per-adapter rosters and variant-addition drivers live in `36`.
///
/// Per Q-IR-010 (2026-05-21) and Q-ADAPT-002 (2026-05-21):
/// - This enum's body is the canonical roster (closed catalog rule R4).
/// - Adding a variant is a `35`-side MINOR edit driven by a concrete
///   adapter feature ratified in `36 §4` / `§6`.
/// - **Scope rule:** `Capability` enumerates ONLY features that cannot
///   be synthesized at an adapter's PlanBuilder layer — features whose
///   absence in the consuming engine cannot be papered over without
///   changing semantics. Adapter-internal rewrite strategies (CTE
///   expansion, GROUPING SETS expansion, DISTINCT-aggregate emulation)
///   are NOT capabilities; they are private adapter strategy.
/// - **Load-bearing consumer:** the Substrait-handoff boundary, where
///   semstrait emits a Substrait plan and a foreign engine consumes it
///   without semstrait-side rewrite. SQL-emitting adapters own their
///   full rewrite pipeline; their capability advertisement is ergonomic
///   (planner pre-flight, api pre-`adapt` UX), not contractual.
/// - Each adapter advertises its supported subset via its
///   `AdapterCapabilities { capabilities: &'static [Capability] }`
///   (`36 §6.1`).
#[non_exhaustive]
pub enum Capability {
    RegexpMatch,
    RegexpExtract,
    IntervalLiteral,
    AsOfJoin,
    StructAccess,
    // Variants added under `36`-driven rationale; current full roster
    // mirrored in `36 §6`.
}
```

## Testing Strategy

This leaf owns the adapter artifact family (`EngineArtifact`, `EnginePlan`, `SqlArtifact`, `DialectId`, `Dialect`, `Capability`). Tests:

- **`DialectId` artifact-side only:** Integration tests verify `DialectId` appears only on `SqlArtifact.dialect` and `Dialect::ID`; it does NOT appear on `SemanticPlan`, any `PlanNode` variant, `Expr<L>`, `NodeMeta`, or registry-side types (S7 guarantee, see [ir-crate-charter.md](ir-crate-charter.md) §1.5).
- **`EnginePlan::to_bytes` / `to_json` roundtrip:** Unit tests verify Substrait `Plan` serializes to deterministic protobuf bytes and pretty JSON; deserialization (when implemented) recovers the original plan.
- **`SqlArtifact` routing:** Integration tests verify `SqlArtifact.dialect` correctly identifies the emitted SQL's target engine (ANSI / DATAFUSION / DUCKDB / SPARK); downstream consumers can route the text by reading `.dialect`.
- **`Dialect` trait extensibility:** Integration tests verify third-party adapter crates can impl `Dialect` with a new `DialectId` const; the trait is not sealed.
- **`Capability` scope rule:** Code review enforces only features that cannot be synthesized adapter-side (e.g., native regexp support) appear as `Capability` variants; adapter-internal rewrite strategies (CTE expansion, DISTINCT emulation) do NOT land as capabilities.
- **Artifact type-level distinction:** Unit tests verify `EngineArtifact::Sql` carries `SqlArtifact` (text + dialect) and `EngineArtifact::Plan` carries `EnginePlan` (structured IR); the two are disjoint (no mixed artifact).

See [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md) for shared harness.
