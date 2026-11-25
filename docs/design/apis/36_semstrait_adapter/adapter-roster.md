---
id: 36-adapter-roster
status: ratified
version: 2.1
impl: in-progress
---

# v1 Built-in Adapter Roster

This leaf owns the v1 built-in adapter roster including AnsiSqlAdapter, DataFusionSqlAdapter, DuckDbSqlAdapter, SparkSqlAdapter, and SubstraitAdapter along with the AdapterCapabilities surface.

## 5. Adapter Implementations (v1 roster)

Each subsection ratifies the per-adapter `AdapterId`, dialect choice, artifact output, and PlanBuilder-layer rewrite scope. Per `30 §10.2`, per-engine adapter crates (DataFusion / DuckDB / Spark / Substrait) live in their own crates with their own stability tiers; `semstrait-adapter` carries only the baseline `AnsiSqlAdapter` and the dependency-free `SubstraitAdapter` scaffold. The roster is `#[non_exhaustive]` in practice — new adapter crates land independently per `14a §7.1` / `30 §10.2`.

### 5.1 `AnsiSqlAdapter`

```rust
/// Baseline SQL adapter. Emits ANSI-standard SQL with double-quoted
/// identifiers, `FETCH FIRST n ROWS ONLY` for limits, `IS NOT DISTINCT
/// FROM` for null-safe equality. No PlanBuilder-layer rewrites —
/// emits the canonical `Expr` tree verbatim where ANSI SQL supports it.
pub struct AnsiSqlAdapter;

impl EngineAdapter for AnsiSqlAdapter {
    fn id(&self) -> AdapterId { AdapterId::ANSI_SQL }
    fn dialect(&self) -> Option<&dyn DialectEmit> { Some(&AnsiDialect) }
    fn capabilities(&self) -> &AdapterCapabilities { &ANSI_CAPABILITIES }
    fn adapt(&self, plan: &SemanticPlan, manifest: &SemanticManifest)
        -> Result<
            (EngineArtifact, Diagnostics<AdaptErrorKind>),
            (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>),
        >
    {
        let (sql, warnings) = self.emit(plan, manifest)?;
        Ok((EngineArtifact::Sql(sql), warnings))
    }
    fn emit(&self, plan: &SemanticPlan, manifest: &SemanticManifest)
        -> Result<
            (SqlArtifact, Diagnostics<AdaptErrorKind>),
            (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>),
        >
    {
        crate::emit::emit_sql(plan, manifest, &AnsiDialect, AdapterId::ANSI_SQL)
    }
}
```

- Purpose — the reference / fallback adapter. Used by `debug_sql` (§3.5), by diagnostics that surface rendered SQL for troubleshooting, and as a starting point for authors bringing up a new dialect. Matches current code's `AnsiDialect` + ANSI emitter path.
- Capabilities — conservative. Advertises none of the irreducible features (`AsOfJoin = false`, `RegexpMatch = false`, `RegexpExtract = false`, `IntervalLiteral = false`, `StructAccess = false`). Authors targeting the ANSI fallback who require those features get a `34`/`38`-side pre-flight diagnostic. See §6.
- Rewrite posture — zero PlanBuilder-layer rewrites; every canonical `FunctionCall` emits its canonical name verbatim. `DialectEmit::rewrite_function` always returns `Ok(RewrittenCall::NameOnly)`. Adapters that need rewrites use the per-engine impls below.

### 5.2 `DataFusionSqlAdapter`

Lives in `semstrait-adapter-datafusion`. Produces SQL targetable at `datafusion-sql` or — under feature `substrait-emit` — routes through `SubstraitAdapter` (the DataFusion engine natively consumes Substrait; the SQL path is for human-facing inspection or session-API consumers that prefer SQL).

- `id() = AdapterId::DATAFUSION_SQL`.
- `dialect() = Some(&DataFusionDialect)` (dialect lives in `semstrait-adapter-datafusion`).
- PlanBuilder-layer rewrites per `registry/functions_mapping.md §13.1`: `position` → `strpos` with arg-reorder; `sign` → `Cast(signum(x), Integer)`; `variance` → `var_samp`; `approx_count_distinct` → `approx_distinct`; `date_add` / `date_sub` → `BinaryOp` forms; `date_diff` (3-arg) → `date_part`-extracted form; `year`/`month`/`day`/`hour`/`minute`/`second` → `date_part('part', x)`; `RegexpExtract` → `array_element(regexp_match(...), group + 1)`.
- Capabilities — `AsOfJoin = false`, `RegexpMatch = true`, `RegexpExtract = true`, `IntervalLiteral = true`, `StructAccess = false` (pending canonical `DataType::Struct` which is out of v1).

### 5.3 `DuckDbSqlAdapter`

Lives in `semstrait-adapter-duckdb`.

- `id() = AdapterId::DUCKDB_SQL`.
- `dialect() = Some(&DuckDbDialect)`.
- PlanBuilder-layer rewrites per `registry/functions_mapping.md §13.2`: `position` → `strpos` with arg-reorder; `concat` → `||`-chain (NULL-propagation alignment); `lpad` / `rpad` 2-arg → 3-arg space-injection; `to_date` / `to_timestamp` (1-arg) → `Cast`. `log(base, x)` — if DuckDB version floor does not carry the 2-arg form, raises `AdaptError::UnsupportedFeature { feature: UnsupportedFeatureKind::Function, .. }`. Dialect-layer rendering of `percentile_cont` uses native `WITHIN GROUP (ORDER BY col)` syntax per `§13.2`.
- Capabilities — `AsOfJoin = true` (DuckDB 0.9+ native `ASOF JOIN` syntax), `RegexpMatch = true`, `RegexpExtract = true`, `IntervalLiteral = true`, `StructAccess = false` (pending canonical `DataType::Struct`).
- Dialect notes — `DuckDbDialect.type_name` maps `Timestamp(p)` to precision-specific keywords (`TIMESTAMP_S` / `TIMESTAMP_MS` / `TIMESTAMP` / `TIMESTAMP_NS`) per `registry/types_mapping.md §3.3`.

### 5.4 `SparkSqlAdapter`

Lives in `semstrait-adapter-spark`.

- `id() = AdapterId::SPARK_SQL`.
- `dialect() = Some(&SparkDialect)`.
- PlanBuilder-layer rewrites per `registry/functions_mapping.md §13.3`: `length(arr)` → `size(arr)`; `lpad` / `rpad` 2-arg → 3-arg space-injection; `ceil(Float)` / `floor(Float)` → `Cast(... as Double)`; `date_add` / `date_sub` → `BinaryOp` forms; `date_diff` (3-arg, day part) → `datediff(end, start)` with arg-reorder; `date_diff` (3-arg, non-day part) → per-part extraction via `date_part(part, end) - date_part(part, start)`; `SafeDivide` (Spark 3.3+) → `try_divide` (optional optimization). Dialect-layer rendering of `percentile_cont` uses native `WITHIN GROUP (ORDER BY col)` syntax (Spark 3.1+).
- Capabilities — `AsOfJoin = false` (Spark lacks native as-of), `RegexpMatch = true`, `RegexpExtract = true`, `IntervalLiteral = true` (Spark 3.4+ for `TimestampNTZType` per `registry/types_mapping.md §3.2`), `StructAccess = false` (pending canonical `DataType::Struct`).
- `SparkDialect::emit_ilike` uses the `LOWER(...) LIKE LOWER(...)` form per §4.7 default; Spark has no native `ILIKE`.

### 5.5 `SubstraitAdapter`

```rust
/// Non-SQL adapter producing `EngineArtifact::Plan(EnginePlan::Substrait)`.
/// Consumed by Substrait-native engines (DataFusion via
/// `datafusion-substrait`; future Substrait-direct consumers).
pub struct SubstraitAdapter;

impl EngineAdapter for SubstraitAdapter {
    fn id(&self) -> AdapterId { AdapterId::SUBSTRAIT }
    fn dialect(&self) -> Option<&dyn DialectEmit> { None }   // non-SQL
    fn capabilities(&self) -> &AdapterCapabilities { &SUBSTRAIT_CAPABILITIES }
    fn adapt(&self, plan: &SemanticPlan, manifest: &SemanticManifest)
        -> Result<
            (EngineArtifact, Diagnostics<AdaptErrorKind>),
            (Diagnostic<AdaptErrorKind>, Diagnostics<AdaptErrorKind>),
        >
    {
        let (substrait_plan, warnings) =
            crate::substrait::emit_substrait(plan, manifest)?;
        Ok((
            EngineArtifact::Plan(EnginePlan::Substrait(Box::new(substrait_plan))),
            warnings,
        ))
    }
    // `emit` inherits the default impl — returns
    // AdaptErrorKind::EmissionNotSupported { adapter: SUBSTRAIT }.
}
```

- Emission follows the mapping table at `35 §10.2` — each `PlanNode` maps to exactly one `substrait::proto::Rel` kind.
- Function anchors — every `FunctionCall.name` / every `Aggregation` variant maps directly to a Substrait-standard function URN. V1 uses `CanonicalFn::as_str()` as the anchor; per-function URN overrides are tracked as `[TD-ADAPTER-SUBSTRAIT-ANCHOR]` (see `Q-ADAPT-008`).
- `SemAnnotation` round-trip — per `35 §10.2`, annotations travel through `AdvancedExtension.optimization` with URN `urn:semstrait:annotations:v1`. Deserializers unknown to the consuming engine skip unknown annotations gracefully per `35 §12.2`.
- Capabilities — `AsOfJoin = false` in v1 (Substrait has no standard as-of; extension path tracked as `[TD-ADAPTER-SUBSTRAIT-ASOF]`), `RegexpMatch = true`, `RegexpExtract = true`, `IntervalLiteral = true`, `StructAccess = true` (Substrait supports struct-field access natively; advertisement gated by canonical `DataType::Struct` which is out of v1, but the consumer-side support is real). **Substrait is the load-bearing capability-contract path** per §6.2 — capabilities advertised here are what the Substrait plan REQUIRES the consuming engine to support.

## 6. `AdapterCapabilities`

### 6.1 Shape

```rust
/// Declarative capability advertisement. Returned by
/// `EngineAdapter::capabilities()`. Every adapter holds its
/// capabilities in a `static` instance (no per-invocation mutation).
#[non_exhaustive]
pub struct AdapterCapabilities {
    pub capabilities: &'static [Capability],
    // Ratified flags — exposed as methods, not public fields, so the
    // struct can grow on its `#[non_exhaustive]` surface without
    // breaking pattern-match consumers (`30 §4.2`).
}

impl AdapterCapabilities {
    /// True when `capabilities` contains `cap`. Linear in
    /// `capabilities.len()`, but the slices are bounded (≤ 32 entries).
    pub fn supports(&self, cap: Capability) -> bool;

    // --- Ratified predicate shortcuts: one per variant in the
    // non-exhaustive `Capability` roster defined in `35 §12.6`. ---
    pub fn supports_as_of(&self)            -> bool;
    pub fn supports_regexp_match(&self)     -> bool;
    pub fn supports_regexp_extract(&self)   -> bool;
    pub fn supports_interval_literal(&self) -> bool;
    pub fn supports_struct_access(&self)    -> bool;
    // Additional predicates MAY be added in MINOR per `30 §2.1`.
}
```

The struct surface is `#[non_exhaustive]` per `30 §4.2`. Adding a new `Capability` variant is MINOR per I10 / `30 §11.1`.

### 6.2 Consumers — SQL adapters vs Substrait handoff

Per `Q-ADAPT-002` (closed 2026-05-21), `AdapterCapabilities` plays two architecturally distinct roles depending on the emission target. The asymmetry is load-bearing.

**SQL-emitting adapters** (`AnsiSqlAdapter`, `DataFusionSqlAdapter`, `DuckDbSqlAdapter`, `SparkSqlAdapter`) — capabilities are **ergonomic hints**, not contracts. semstrait owns the full PlanBuilder-layer rewrite pipeline and emits engine-native SQL itself; adapter-internal rewrite strategies (CTE expansion, GROUPING SETS expansion, DISTINCT-aggregate emulation, function-name remaps) are private to each adapter and NOT advertised through `Capability`. Consumers:

- **`semstrait-planner` (`34`)** — pre-flight UX. A plan rule that emits `Capability::AsOfJoin` SHOULD be disabled when the target adapter advertises `supports_as_of() == false`, so authors get a clear `34`-side diagnostic instead of an opaque `adapt`-time failure.
- **`semstrait-api` (`38`)** — pre-`adapt` feasibility check. The caller consults the adapter's capabilities before dispatching. Not mandatory — `adapt` is the authoritative feasibility check, raising `AdaptError::Unsupported*` when an irreducible gap surfaces downstream.

**`SubstraitAdapter`** — capabilities are the **handoff contract**. semstrait emits a Substrait plan to be consumed by a foreign engine across a process / engine boundary; semstrait CANNOT rewrite on the consumer's behalf. The capability set declares what the consuming engine MUST support. The same `AdapterCapabilities` advertisement is consulted by `34` / `38` exactly as for SQL adapters, but the role flips from ergonomic to contractual at the handoff line.

**Inside `adapt` itself** — capabilities are NOT consulted. Each adapter's `adapt()` either renders successfully or raises `AdaptError::Unsupported*`. Q-ADAPT-002 closes with the rule: capability checks live at planner / api pre-flight; `adapt`-time failures are the fallback.

### 6.3 Roster growth policy

Adding a new `Capability` variant is MINOR. The type definition lives in `35 §12.6` (closed catalog rule R4); `36` drives variant additions through concrete adapter-feature need and owns the per-adapter `AdapterCapabilities` roster (Q-IR-010, 2026-05-21).

**Scope test for a new variant.** Before adding a variant, answer: *"Is this feature irreducible across the Substrait-handoff boundary?"* — i.e. is its absence in the consuming engine impossible to paper over with a semstrait-side PlanBuilder rewrite without changing semantics? If yes, add it. If the feature is universally synthesizable via SQL-adapter rewrite (CTE → subquery, GROUPING SETS → UNION ALL, DISTINCT-aggregate → universal), it is adapter-internal strategy and does NOT belong in `Capability`.

Each new variant:

1. Earns a paragraph in §4 / §5 describing its emission pathway and (if Substrait-bound) the consumer-engine requirement.
2. Earns a predicate shortcut on `AdapterCapabilities` if it's a commonly-consulted flag.
3. Documents the adapters that advertise it in the per-adapter subsections of §5.

## Testing Strategy

- AnsiSqlAdapter baseline (no rewrites, ANSI-standard SQL per §5.1)
- DataFusionSqlAdapter rewrites (position→strpos, RegexpExtract→array_element, date_part extraction per §5.2)
- DuckDbSqlAdapter rewrites (position→strpos, concat→||, AsOf native syntax per §5.3)
- SparkSqlAdapter rewrites (length→size, date_diff arg-reorder, try_divide per §5.4)
- SubstraitAdapter proto emission (per 35 §10.2 mapping per §5.5)
- AdapterCapabilities predicate shortcuts (supports_as_of/regexp_match/regexp_extract/interval_literal/struct_access per §6.1)
- Capability roster per-adapter (AsOfJoin=true DuckDB only per §5.1–5.5)
- SQL vs Substrait capability role (ergonomic vs contractual per §6.2)
- Capability growth policy (irreducible-test per §6.3)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
