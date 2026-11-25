---
id: 36-dialect-emit
status: ratified
version: 2.1
impl: in-progress
---

# DialectEmit Operational Trait

This leaf owns the DialectEmit operational trait which defines per-dialect SQL rendering including identifier quoting, literal escaping, DataType spelling, function rewriting, cast semantics, null-ordering clauses, AsOf join emission, and regex/date_trunc methods.

## 4. `DialectEmit` — Operational Dialect Trait

### 4.1 Relationship to `Dialect` in `35`

`35 §7.5` ratifies the **structural** `Dialect` trait: associated `const ID: DialectId` and `fn capabilities(&self) -> &'static [Capability]`. That trait is consumed by planner-side capability gates — it answers "is this dialect in the set?" and "what can it do?".

`36` adds the **operational** half: `DialectEmit`, a supertrait extension that carries the per-dialect SQL rendering methods. The split per `Q-ADAPT-003`:

```rust
/// Operational dialect surface. Extends the structural `Dialect` trait
/// (`35 §7.5`) with rendering methods every SQL-emitting adapter relies
/// on. `DialectEmit: Dialect` (supertrait chain).
///
/// Not sealed — third-party dialects (e.g. a hypothetical
/// `ClickHouseDialect`) MUST be able to impl `DialectEmit` without a
/// private escape hatch.
pub trait DialectEmit: Dialect {
    // ... §4.2–§4.7 methods
}
```

Substrait-only adapters (§5.5) impl `Dialect` (to advertise capabilities) but do NOT impl `DialectEmit` (they never render SQL). `EngineAdapter::dialect()` returns `None` for them.

### 4.2 Identifier quoting

```rust
/// Quote `ident` per dialect rules. Every identifier emitted into
/// `SqlArtifact.text` passes through this method — column names, table
/// names, CTE labels, aliases. Never concatenated raw. I14 guarantee.
fn quote_identifier(&self, ident: &str) -> String;
```

Every impl MUST:

- Enclose `ident` in the dialect's identifier-quote delimiters (ANSI / DataFusion / DuckDB: `"…"`; Spark: backtick-fallback per legacy SQL idiom; MySQL-family: backtick).
- Escape the delimiter character occurring inside `ident` per dialect rules (ANSI: double the quote — `abc"def` → `"abc""def"`).
- NOT truncate, case-fold, or reject `ident` based on length / reserved-word collisions — any identifier that survived `Name::new` (`35 §6.4`) is quoting-safe at this layer. Length limits and reserved-word handling are engine-side concerns.

### 4.3 Literal escaping

```rust
/// Render a `Literal` as dialect-appropriate SQL text.
/// Per-variant rendering per `registry/types_mapping.md` conventions.
fn quote_literal(&self, lit: &Literal) -> String;

/// Escape an embedded string-literal value (for `Expr::Literal {
/// value: String(_) }`). Default impl doubles internal single-quotes
/// per ANSI SQL; adapters MAY override for engines with non-standard
/// rules (none at v1).
fn escape_string_literal(&self, s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}
```

### 4.4 `DataType` spelling

```rust
/// Render a canonical `DataType` as the dialect's native SQL type
/// name. Consumed by `Expr::Cast` rendering and by projection-alias
/// type annotations. Per `registry/types_mapping.md §1`.
///
/// Error if the canonical `DataType` variant (e.g. a future
/// `DataType::Json`) has no dialect-native form:
/// `AdaptErrorKind::UnsupportedType { ty, adapter, .. }`.
/// Bare-kind shape per `31 §3.1` construction-site convention.
fn type_name(&self, dt: &DataType) -> Result<String, AdaptErrorKind>;
```

### 4.5 Function rewriting

```rust
/// Rewrite a canonical `FunctionCall` per the adapter's per-function
/// tier table (`registry/functions_mapping.md`; tier classification in `rewrite-pipelines.md §7`).
///
/// Returns:
/// - `Ok(RewrittenCall::NameOnly)` — emit the canonical name verbatim.
/// - `Ok(RewrittenCall::Remapped { name })` — emit `name` with the same
///   args.
/// - `Ok(RewrittenCall::Structural(expr))` — caller re-renders the
///   returned `Expr` in place of the original `FunctionCall`.
/// - `Err(AdaptErrorKind::UnsupportedFeature { .. })` — hard fail per
///   `14a §6.3`.
///
/// Called during §9's emit walk at every `FunctionCall` encounter.
/// Bare-kind shape per `31 §3.1` construction-site convention.
fn rewrite_function(
    &self,
    call: &FunctionCall,
) -> Result<RewrittenCall, AdaptErrorKind>;

#[non_exhaustive]
pub enum RewrittenCall {
    NameOnly,
    Remapped   { name: &'static str },
    Structural(Box<PhysicalExpr>),
}
```

`rewrite_function` is where the PlanBuilder-layer rewrites of `registry/functions_mapping.md §13` land. Structural rewrites (e.g. DataFusion's `date_diff` → `CAST(d2 - d1 AS BIGINT)`) produce a `PhysicalExpr` the generic emitter renders as if authored — maintains the single-pass emission posture.

### 4.6 Cast semantics

```rust
/// Render an `Expr::Cast { expr, target }` node. Default impl emits
/// `CAST(<expr> AS <type_name(target)>)`; adapters with a native
/// short-form (DuckDB's `expr::TYPE`) MAY override for terser output,
/// but the default is always the canonical SQL form.
///
/// Per `registry/types_mapping.md §2` — widening is silent, narrowing
/// is author-explicit. The adapter NEVER inserts a cast the planner
/// did not author; boundary reconciliation Casts originate at compile
/// per `14 §6.4`.
fn emit_cast(&self, expr: &str, target: &DataType) -> Result<String, AdaptErrorKind> {
    Ok(format!("CAST({} AS {})", expr, self.type_name(target)?))
}
```

### 4.7 Null-ordering, AsOf join, DateTrunc, regex — ratified methods

```rust
/// SQL rendering of `NullOrdering` as an ORDER BY suffix. `First` →
/// `" NULLS FIRST"`, `Last` → `" NULLS LAST"`, `Unspecified` → `""`
/// (engine default applies). Per `35 §6.6`.
fn null_ordering_clause(&self, nulls: NullOrdering) -> &'static str;

/// `Expr::DateTrunc { expr, grain }` emission. All three first-class
/// engines render as `date_trunc('grain', expr)` — keep the method for
/// dialect override anyway (casing, quoting of the grain token).
fn emit_date_trunc(&self, expr: &str, grain: Grain) -> String;

/// `Expr::Like / ILike / RegexpMatch / RegexpExtract` emission, one
/// method per dedicated variant. Default impls fall back to the
/// ANSI forms; per-engine overrides live on
/// `DataFusionDialect` / `DuckDbDialect` / `SparkDialect` per
/// `registry/functions_mapping.md §13`.
fn emit_like(&self, expr: &str, pattern: &str, negated: bool) -> String;
fn emit_ilike(&self, expr: &str, pattern: &str, negated: bool) -> String;
fn emit_regexp_match(&self, expr: &str, pattern: &str, full_match: bool) -> String;
fn emit_regexp_extract(&self, expr: &str, pattern: &str, group: usize) -> String;

/// Emission of an `AsOf` join variant when the adapter advertises
/// `Capability::AsOfJoin`. Called only when `capabilities()` includes
/// `AsOfJoin`; the generic emitter raises
/// `AdaptErrorKind::UnsupportedFeature { UnsupportedFeatureKind::JoinType, .. }`
/// when the adapter does NOT advertise it but a plan carries an
/// `AsOf` join. Per `16 §5.2` / `17 §5`. Bare-kind shape per
/// `31 §3.1` construction-site convention.
fn emit_asof_join(
    &self,
    left: &str,
    right: &str,
    anchor: &AsOfAnchor,
    on: &[KeyPair],
) -> Result<String, AdaptErrorKind> {
    let _ = (left, right, anchor, on);
    Err(AdaptErrorKind::UnsupportedFeature {
        feature: UnsupportedFeatureKind::JoinType,
        name: "as-of".into(),
        adapter: self.id_hint(),
    })
}

/// Helper: the adapter's `AdapterId` as available to the dialect.
fn id_hint(&self) -> AdapterId;
```

### 4.8 Dialect-surface stability

`DialectEmit` is `#[non_exhaustive]` in effect — adding a new method with a default body is MINOR per `30 §2.1`. Removing or changing a method signature is MAJOR. Every method carries a default impl where a canonical fallback makes sense (`emit_asof_join`, `escape_string_literal`, `emit_cast`, `emit_regexp_*`); methods with engine-specific emission (`quote_identifier`, `type_name`, `date_trunc`, `null_ordering_clause`) are required.

## Testing Strategy

- DialectEmit trait openness (third-party impls compile per §4.1)
- quote_identifier round-trip (delimiter escaping per §4.2, ANSI/DataFusion/DuckDB/Spark variants)
- escape_string_literal ANSI double-quote (default impl per §4.3)
- type_name per-engine coverage (ANSI/DataFusion/DuckDB/Spark type spelling per §4.4)
- rewrite_function dispatch (NameOnly/Remapped/Structural/Unsupported per §4.5)
- emit_cast default ANSI form (CAST(expr AS type) per §4.6)
- null_ordering_clause (First/Last/Unspecified per §4.7)
- emit_asof_join capability gate (only DuckDB advertises AsOfJoin per §4.7)
- Per-engine regex variants (emit_regexp_match/extract per §4.7)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
