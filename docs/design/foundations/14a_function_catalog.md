---
id: 14a-function-catalog
status: ratified
version: 2.1
impl: implemented
---


# 14a. Function Catalog

> Ratifies the machinery `14` defers to at every `FunctionCall` site and at `14 §6.4`'s Declarative-block tag dispatch: the `FunctionRegistry` API shape and lifecycle, the function-spec model, the canonical scalar / non-closed-aggregate catalog, function-level `Additivity`, and the adapter extension API.
>
> Crate placement (item Y / V4): the function **name identity** `CanonicalFn` lives in `semstrait-model` (public authoring vocabulary); the resolved `FunctionRegistry`, the spec types, and `Additivity` / `DimensionAxis` stay in `semstrait-ir`. See `[14 §9.2/§9.3](14_expressions.md)` for the placement concept-of-record. Name existence, signature match, and return-type inference are compile-stage validation in `semstrait-manifest` against the ir registry.

## 1. Purpose and Scope

`14a` ratifies how semstrait models functions. The forward references from `14` (`§3.3` `FunctionCall`, `§6.4` Declarative-block tag dispatch) resolve here.

**Ratifies:** `FunctionRegistry` API + lifecycle (§2); function-spec model + signature polymorphism + return-type rules + variadic posture (§3); canonical scalar / non-closed-aggregate catalog policy (§4); `BinaryOp` promotion posture — no canonical lattice in v1, per-engine deviation lives in `registry/functions_mapping.md` (§5); portability model (§6); adapter extension API (§7); function-resolution errors (§8).

**Does NOT ratify:** the compile-pipeline substitution/resolution algorithm — `[19 §3](19_expression_flow.md)`; per-engine native-name mapping, rewrite tiers, emulation — `registry/functions_mapping.md`; adapter rendering of resolved `FunctionCall` — `[36](../apis/36_semstrait_adapter.md)`.

**Invariants upheld:** **I1** — registry operates on canonical `Expr::FunctionCall { name, args }`; no SQL text. **I2** — signatures typed in canonical `DataType` / `TypeClass`. **I10** — every public sum carries `#[non_exhaustive]`. Per `14 §6.4`, registered names form the non-reserved Declarative-block tag vocabulary. Per `14 §5.4`, no implicit coercion at call sites — exact-type signature match or `NoMatchingSignature`.

## 2. The `FunctionRegistry` — API and Lifecycle

The `FunctionRegistry` is the compile-time catalog `compile` consults to resolve `Expr::FunctionCall.name` to a `FunctionSpec` (§3) — the **compile-time source of truth** for which canonical names exist. It is a `semstrait-ir` type (item Y / V4); the unresolved `CanonicalFn` name identity it resolves against is `semstrait-model`-owned. Surface: populate at startup, seal, look up.

The Declarative-block parser does **not** consult the registry. Per `14 §6.4`, a non-reserved tag is normalized into a `FunctionCall` node whose `name` is a `CanonicalFn` built **purely syntactically** — grammar + case-normalization, with membership tested against a model-local function table (the parser's roster), never the registry. Name *existence* against the registry is the compile-stage check (§3, in `semstrait-manifest`); the static table is that registry's roster.

### 2.1 Lifecycle

Sealed `&'static FunctionRegistry` — one shared instance per process, initialized eagerly via `pub fn function_registry() -> &'static FunctionRegistry` (`OnceLock`-backed). Immutable post-seal; no per-Model configurability in v1. Eliminates lifetime noise from every compile-stage signature. Multi-config (test-harness swappable adapter sets) tracked under `[TD-REGISTRY-MULTI-CONFIG]`.

### 2.2 Storage and entries

Two entry families share one flat `HashMap<&'static str, FunctionSpec>`: **core** entries (canonical catalog, §4) shipped with the runtime; **adapter-extended** entries (engine-specific functions like Spark `collect_set`, DuckDB `list_extract`) contributed by adapter crates per §7. No tier / overlay; collision handling per §7.2.

### 2.3 Lookup

```rust
impl FunctionRegistry {
    pub fn lookup(&self, name: &str) -> Option<&FunctionSpec>;
}
```

Strict case-sensitive exact match on the canonical name. Canonical names are lowercase ASCII per `14 §6.5`'s identifier grammar. No aliases in v1 (`[TD-REGISTRY-ALIASES]`). Namespace shared with `14 §6.4.1`'s reserved AST tags — registry entries colliding with reserved tags surface as `ReservedTagCollision` at startup (§7.2). Callers: `compile` at name-existence + type-inference time (§3); adapters at `adapt` time (which also consult `registry/functions_mapping.md` for the engine-native name). The Declarative-block parser is **not** a caller — it builds `CanonicalFn` names syntactically against a model-local table (above), and existence-against-registry is deferred to `compile`.

## 3. Function Spec

### 3.1 `FunctionSpec` shape

```rust
#[non_exhaustive]
pub struct FunctionSpec {
    pub canonical_name: &'static str,
    pub category:       FunctionCategory,
    pub signatures:     NonEmpty<FnSignature>,
    pub additivity:     Option<Additivity>,  // §3.6; None for scalar / non-aggregate
}
```

No `description` field — author-facing prose lives in this catalog document (§4), not on the runtime spec; the canonical name is sufficient routing for adapters and the registry. No `null_handling` (NULL-in/out delegated to engine per `14 §5.4`). No `deterministic` (`[TD-REGISTRY-DETERMINISM]`). No `aliases` / `since_version` / `stability` (versioning lives in `apis/30`). Per-engine portability carriage lives entirely in `registry/functions_mapping.md` (§6).

### 3.2 `FunctionCategory`

```rust
#[non_exhaustive]
pub enum FunctionCategory {
    Scalar,
    Aggregate,
    // Window — compile-emitted only per `14 §3.3` notes; not author-registered in v1.
}
```

Discriminates scalar vs non-closed-aggregate. Consumed by `14 §7`'s shape-gate enforcement (`Aggregate`-category calls are gated against scalar-only sites) and by the planner's additivity / rollup reasoning (`19 §6.5`). No scalar sub-categorization in v1 (`[TD-REGISTRY-SUBCATEGORY]`); catalog §4 groups by domain in headings only.

### 3.3 Signature polymorphism — overload set

```rust
#[non_exhaustive]
pub struct FnSignature {
    pub args:        &'static [ParamType],
    pub variadic:    Option<ParamType>,
    pub return_type: ReturnTypeRule,
}
```

`signatures: NonEmpty<FnSignature>`; lookup attempts each signature in declaration order; first exact match (per-arg `DataType` equality) wins. No TypeClass generics, no implicit coercion (`14 §5.4`). On no match: `NoMatchingSignature { name, arg_types, tried_signatures }` (the `tried_signatures` list mirrors the full overload set so the author can compare directly). TypeClass-parameterised generics deferred under `[TD-REGISTRY-TYPECLASS]`.

**Variadic posture.** `variadic: Some(T)` accepts zero-or-more trailing args of type `T` after the fixed prefix. Mid-signature variadic (e.g. `printf`-style) is `[TD-REGISTRY-MID-VARIADIC]`; not needed in v1. Optional args (`substring(expr, start, [length])`) are expressed as multiple overloads differing in arity.

### 3.4 `ReturnTypeRule`

```rust
#[non_exhaustive]
pub enum ReturnTypeRule {
    /// Concrete canonical type.
    Fixed(DataType),
    /// Mirror arg N's type (0-indexed).
    SameAs(usize),
    /// Common-supertype promotion of the listed arg indices. `14a` publishes
    /// no canonical promotion lattice (§5); per-engine observed rules live in
    /// `registry/functions_mapping.md`'s `BinaryOp` section.
    Promoted(&'static [usize]),
    /// Decimal scale-zero collapse: `Decimal(p, s) → Decimal(p, 0)`. For
    /// integer-rounding kernels (`ceil`, `floor`) on Decimal inputs.
    DecimalScaleZero,
    /// Escape hatch for width-sensitive / literal-driven rules (e.g. `cast(x, T) -> T`).
    Custom(fn(&[DataType]) -> Result<DataType, CompileError>),
}
```

Nullability is pass-through per `13`'s NULL model; no `NullableOf` variant. Per-engine nullability tightening lives in `registry/types_mapping.md`.

`ReturnTypeRule` does not implement `PartialEq` — `Custom` carries a function pointer, and pointer equality is implementation-defined. Inspection sites (signature lookup, registry assertions) compare by the surrounding `FnSignature.args` instead.

`DecimalScaleZero` is restricted to overloads whose first argument is `ParamType::DecimalFamily` (§3.5). Float / Double overloads of the same canonical name use `SameAs(0)`.

### 3.5 `ParamType`

```rust
#[non_exhaustive]
pub enum ParamType {
    Concrete(DataType),
    AnyOf(Vec<DataType>),
    NumericFamily,
    DecimalFamily,
    StringFamily,
    TemporalFamily,
    Any,
}
```

Family variants ground in the `TypeClass` vocabulary ratified at `[35 §4.3](../apis/35_semstrait_ir.md)` over `13 §2.1`'s `DataType` set. Family use is restricted to declared overloads in v1; full TypeClass generics still deferred under `[TD-REGISTRY-TYPECLASS]`. `DecimalFamily` is the additive variant for decimal-precision-sensitive overloads (e.g. `ceil` / `floor` / `round` / `median` on Decimal); it pairs with `ReturnTypeRule::DecimalScaleZero` (§3.4) for scale-zero collapse.

### 3.6 `Additivity` (function-level)

Canonical mathematical additivity of an aggregate function. Carried on `FunctionSpec.additivity`; `None` for scalar functions.

```rust
#[non_exhaustive]
pub enum Additivity {
    Additive,                                  // SUM, COUNT, MIN, MAX
    SemiAdditive { axes: Vec<DimensionAxis> }, // FIRST, LAST (temporal axis)
    NonAdditive,                               // AVG, COUNT_DISTINCT, MEDIAN, PERCENTILE
}

#[non_exhaustive]
pub enum DimensionAxis {
    Temporal,
}
```

**Two-source SoC** (per `[19 §6.5](19_expression_flow.md)`'s composition table): function-level here vs model-level `AdditivityType` on a Measure / Metric per `[18 §5.2](18_entities.md)`. The **effective** additivity (intersection of axes, narrower wins) drives lossy-reaggregation advisories. **v1 reads function-level only** (model-level deferred per §3.6.2); the composition table is the reserved extension.

Not author-extensible in v1 — hardcoded per built-in aggregate; UDF additivity is `[TD-REGISTRY-DETERMINISM]`'s sibling deferred item.

#### 3.6.1 Common behavior and branching

Pre-aggregation/re-aggregation safety is the same question regardless of where additivity comes from, so the **branching logic lives once** on `Additivity` itself:

```rust
impl Additivity {
    /// True when partial aggregates over disjoint partitions may be combined
    /// (per-branch / per-level pre-aggregation then a final re-aggregation).
    pub fn pre_aggregatable(&self) -> bool {
        matches!(self, Additivity::Additive)
        // SemiAdditive is pre-aggregatable only when its `axes` are preserved
        // by the partitioning — evaluated by the planner against the request;
        // see `19 §6.5`. NonAdditive is never blind-pre-aggregatable.
    }

    /// The operator used to combine partial aggregates at the re-aggregation
    /// step, when `pre_aggregatable`. `Sum -> Sum`, `Count -> Sum`,
    /// `Min -> Min`, `Max -> Max`; `None` when re-aggregation is unsafe.
    pub fn reaggregation(&self, partial: AggregationOp) -> Option<AggregationOp> { /* … */ }
}
```

#### 3.6.2 Source abstraction (v1 function-only; extensible)

The *source* of an expression's additivity is abstracted so the planner is written once and additional sources slot in without touching its branching:

```rust
/// Yields the effective `Additivity` for an aggregation. Implemented in v1 by
/// the aggregate op itself (function-derived). Reserved for future composite
/// sources (function × model-level × temporal-shape) without changing callers.
pub trait AdditivitySource {
    fn additivity(&self) -> Additivity;
}
```

- **v1 — function-only.** The only source is the aggregate op: `impl AdditivitySource for AggregateKind` returns the closed-five mapping (`Sum`/`Count`/`Min`/`Max` → `Additive`, `Avg` → `NonAdditive`) or `FunctionSpec.additivity` for `Extension` aggregates. Model-level `AdditivityType` (`18 §5.2`, incl. `SemiAdditive { axes }`) is **deferred**; the `SemiAdditive` variant and the `19 §6.5` composition table are the reserved extension point — the model source is simply absent in v1, so effective additivity equals the function-level value.
- **Zero-cost.** Planner aggregation code is generic over `A: AdditivitySource` (static dispatch); promoting a `CompositeAdditivity { function, model }` source later is additive, no call-site churn (DRY/SOLID).

## 4. Canonical Function Catalog

### 4.1 Presentation + population policy

Presentation: one row per `FunctionSpec`; overloads collapsed into a `signature(s)` cell (e.g. `(Int, Int) -> Int | (Long, Long) -> Long`). Columns: `canonical_name`, `arity` (or `variadic` marker), `signature(s)`, `return-type rule`, `notes`. Per-engine coverage strip is intentionally absent — `14a` is engine-agnostic; per-engine detail lives in `registry/functions_mapping.md`.

Population policy: **adapter-capability intersection** across DataFusion ∩ Spark ∩ DuckDB. A function enters the canonical catalog only when all three first-class targets support it natively or via trivial name-remap per `registry/functions_mapping.md`'s rewrite tiers. Subset-supported functions are **adapter-extended** entries per §7. This makes every canonical entry portable-by-construction; the intersection grows as adapter coverage matures.

§4.2–§4.6 entries are candidate lists pending intersection verification; the final catalog MAY be a proper subset.

### 4.2 Scalar — String

Ratified Round-2 (2026-05-21). Reserved predicates (`Like` / `ILike` / `RegexpMatch` / `RegexpExtract`) are NOT registry entries — see `[14 §3.3](14_expressions.md)`.

| Canonical | Signatures | Notes |
|---|---|---|
| `upper` | `(String) -> String` | Unicode-aware case mapping. |
| `lower` | `(String) -> String` | Unicode-aware case mapping. |
| `length` | `(String) -> Integer` | Character count. The `(Array<T>) -> Integer` element-count overload is gated on `TD-TYPE-ARRAY` (no Array `DataType` variant in v1 per `13 §2.5`). |
| `substring` | `(String, Integer) -> String`, `(String, Integer, Integer) -> String` | 1-indexed. Positive `pos` only in v1; negative `pos` is non-portable (each engine differs). |
| `trim` | `(String) -> String`, `(String, String) -> String` | Default = strip ASCII space `0x20`. 2-arg form: set semantics — strips any character in the second arg from both ends. |
| `ltrim` | `(String) -> String`, `(String, String) -> String` | Same defaults / set semantics as `trim`. |
| `rtrim` | `(String) -> String`, `(String, String) -> String` | Same defaults / set semantics as `trim`. |
| `concat` | variadic `(String, String...) -> String` | NULL-propagating (any NULL arg → NULL result), per SQL standard. |
| `replace` | `(String, String, String) -> String` | Replaces all occurrences. Literal substring (not regex). |
| `lpad` | `(String, Integer) -> String`, `(String, Integer, String) -> String` | 2-arg form defaults pad to ASCII space. Truncates if input already exceeds target length. |
| `rpad` | `(String, Integer) -> String`, `(String, Integer, String) -> String` | Same as `lpad` mirrored. |
| `reverse` | `(String) -> String` | Reverses by code points. The `(Array<T>) -> Array<T>` element-order overload is gated on `TD-TYPE-ARRAY` (no Array `DataType` variant in v1 per `13 §2.5`). |

Per-engine native names, name-remaps, and structural rewrites: `registry/functions_mapping.md §7`.

### 4.3 Scalar — Math

Ratified Round-2 (2026-05-21). `mod(x, y)` is NOT in the catalog — `BinaryOpKind::Mod` (`%`) is the canonical form.

| Canonical | Signatures | Notes |
|---|---|---|
| `abs` | `(Numeric) -> same` | Type-preserving across all numeric types. Signed-integer minimum (e.g. `abs(INT_MIN)`) is engine-visible overflow. |
| `round` | `(Float) -> Float`, `(Float, Integer) -> Float`, `(Double) -> Double`, `(Double, Integer) -> Double`, `(Decimal(p,s)) -> Decimal(p,s)`, `(Decimal(p,s), Integer) -> Decimal(p,s)` | Half-away-from-zero. Integer arg form: rounds to N decimal places (negative = left of decimal). Integer inputs are rejected at compile — authors cast explicitly. |
| `ceil` | `(Float) -> Float`, `(Double) -> Double`, `(Decimal(p,s)) -> Decimal(p,0)` | Single-arg only in v1. 2-arg with `scale` deferred. Decimal overload uses `ReturnTypeRule::DecimalScaleZero` (§3.4); Float / Double use `SameAs(0)`. |
| `floor` | `(Float) -> Float`, `(Double) -> Double`, `(Decimal(p,s)) -> Decimal(p,0)` | Same as `ceil` mirrored. |
| `sqrt` | `(Float) -> Float` | Negative input is engine-visible (DuckDB errors, DF/Spark return NaN). |
| `power` | `(Float, Float) -> Float` | Out-of-domain behavior (`power(0, neg)`, `power(neg, 0.5)`) is engine-visible. |
| `exp` | `(Float) -> Float` | |
| `ln` | `(Float) -> Float` | Natural logarithm. Non-positive input is engine-visible. |
| `log` | `(Float, Float) -> Float` | 2-arg only: `log(base, value)`. 1-arg `log(x)` is NOT canonical — engines disagree (DF/DuckDB = log10, Spark = ln); authors use `ln(x)` or `log10(x)` explicitly. |
| `log10` | `(Float) -> Float` | Base-10 logarithm. Non-positive input is engine-visible. |
| `sign` | `(Numeric) -> Integer` | Returns `-1`, `0`, `1`. Return type widened to Integer for portability. |

Per-engine native names, name-remaps, and structural rewrites: `registry/functions_mapping.md §8`.

### 4.4 Scalar — Temporal

Ratified Round-2 (2026-05-21). `EXTRACT(part FROM source)` is parser sugar that lowers to canonical `date_part('part', source)` — `extract` is NOT a registry entry.

| Canonical | Signatures | Notes |
|---|---|---|
| `date_part` | `(String, Date) -> Long`, `(String, Timestamp) -> Long` | First arg is a part literal (`'year'`, `'month'`, `'day'`, `'hour'`, `'minute'`, `'second'`, `'millisecond'`, …). Canonical name; `extract` is parser sugar that lowers to this. |
| `year` | `(Date) -> Long`, `(Timestamp) -> Long` | Convenience for `date_part('year', x)`. |
| `month` | `(Date) -> Long`, `(Timestamp) -> Long` | Convenience for `date_part('month', x)`. |
| `day` | `(Date) -> Long`, `(Timestamp) -> Long` | Convenience for `date_part('day', x)`. |
| `hour` | `(Timestamp) -> Long` | Convenience for `date_part('hour', x)`. |
| `minute` | `(Timestamp) -> Long` | Convenience for `date_part('minute', x)`. |
| `second` | `(Timestamp) -> Long` | Integer seconds. Sub-second extraction goes through `date_part('millisecond', …)` or similar. |
| `date_add` | `(Date, Interval) -> Date`, `(Timestamp, Interval) -> Timestamp` | Author-facing surface is FunctionCall. Some engines render as `d + i` (BinaryOp) — that is a per-engine rewrite, not a demotion. Spark's integer-days `date_add(d, n)` form is adapter-extended only. |
| `date_sub` | `(Date, Interval) -> Date`, `(Timestamp, Interval) -> Timestamp` | Mirror of `date_add`. |
| `date_diff` | `(String, Date, Date) -> Long`, `(String, Timestamp, Timestamp) -> Long` | 3-arg form: `date_diff(part, start, end)`. Returns difference in `part` units (signed: positive when `end > start`). 2-arg integer-days form is adapter-extended. |
| `to_date` | `(String) -> Date` | ISO-8601 input only (`'YYYY-MM-DD'`). Format-string overload `(String, String)` is adapter-extended — engine format-string dialects (Chrono strftime / Java DateTimeFormatter / DuckDB strptime) are mutually incompatible. |
| `to_timestamp` | `(String) -> Timestamp` | ISO-8601 input only. Format-string overload adapter-extended (same rationale as `to_date`). |
| `current_date` | `() -> Date` | Per-query determinism (sourced from session). |
| `current_timestamp` | `() -> Timestamp` | Per-query determinism. |

Per-engine native names, name-remaps, and structural rewrites: `registry/functions_mapping.md §9`.

### 4.5 Scalar — Logical / Conditional helpers

Ratified Round-2 (2026-05-21). Reserved AST variants (`Case`, `Coalesce`, `NullIf`, `IsNull`) are NOT registry entries.

| Canonical | Signatures | Notes |
|---|---|---|
| `greatest` | variadic `(T, T...) -> T` | Returns greatest non-NULL value. NULL-skip semantics: returns NULL only when every argument is NULL. Args must share a comparable common type. |
| `least` | variadic `(T, T...) -> T` | Mirror of `greatest`. NULL-skip semantics. |

`if(cond, then, else)`, `ifnull(a, b)`, `nvl(a, b)` are NOT registered — `Expr::Case` and `Expr::Coalesce` (dedicated variants per `14 §3.3`) cover the same use-cases. Authors write the dedicated forms directly.

Per-engine native names, name-remaps, and structural rewrites: `registry/functions_mapping.md §10`.

### 4.6 Aggregate — non-closed

Ratified Round-2 (2026-05-21). Engine-specific forms (DuckDB's `approx_top_k`, Spark's `percentile_approx`) are adapter-extended per §7. Entries carry `FunctionCategory::Aggregate` and `Additivity::NonAdditive` per §3.6.

| Canonical | Signatures | Notes |
|---|---|---|
| `stddev` | `(Numeric) -> Double` | Sample standard deviation. Bare name is sample across all three engines. `stddev_samp` is parser sugar. |
| `stddev_pop` | `(Numeric) -> Double` | Population standard deviation. Separate canonical entry. |
| `variance` | `(Numeric) -> Double` | Sample variance. Bare name is sample across all three engines. `var_samp` is parser sugar. |
| `var_pop` | `(Numeric) -> Double` | Population variance. Separate canonical entry. |
| `median` | `(Numeric) -> Double`, `(Decimal(p,s)) -> Decimal(p,s)` | Exact median. Spark floor at 3.4+. |
| `string_agg` | `(String, String) -> String` | Concatenates non-NULL values with the second-arg separator. ORDER BY / DISTINCT clause modifiers are adapter-extended. Spark floor at 3.3+. |
| `percentile_cont` | `(Float, Numeric) -> Double` | First arg is percentile fraction `[0.0, 1.0]`; second arg is the value column. Author-facing surface is FunctionCall — engines render the SQL-standard `WITHIN GROUP (ORDER BY col)` form at the dialect layer (same shape as `count(DISTINCT)`'s rendering). Spark floor at 3.1+. |
| `approx_count_distinct` | `(Any) -> Long` | HyperLogLog-class approximation. Implementation backends differ across engines (HyperLogLog vs HyperLogLog++) — engine-delegated per `§6.2`. |

`percentile_disc` is NOT canonical — DataFusion lacks it entirely; intersection violation per `§4.1`. Adapter-extended on DuckDB + Spark only.

Per-engine native names, name-remaps, and structural rewrites: `registry/functions_mapping.md §3.2`.

### 4.7 The closed five aggregates

`Sum` / `Avg` / `Count` / `Min` / `Max` are carried by `Expr::Aggregate { op: AggregationOp, … }` (the closed `AggregationOp` enum in `[31 §5.2](../apis/31_semstrait_common.md)`) — **not** registry entries. Their return-type rules follow SQL:2016 promotion; effective additivity is `Additive` for Sum / Count / Min / Max, `NonAdditive` for Avg.

## 5. BinaryOp Promotion — No Canonical Lattice

`BinaryOpKind` (per `[14 §3.3](14_expressions.md)`) is a dedicated `Expr` variant, not a registry entry. `14a` **does not** publish a canonical promotion lattice because:

1. `14 §5.4`'s pass-through posture delegates result-type derivation to the engine.
2. SQL dialects disagree on details (`Decimal` rounding/scale, `Integer / Integer` precision, mixed `Integer + Float`).
3. Per-engine observed rules live in `registry/functions_mapping.md`'s `BinaryOp` section; authors and adapter implementers consult it there.

Semstrait emits no implicit widening at `BinaryOp` sites and gates no compile-time error on lattice mismatch. `Expr::Cast` appears only via (a) author-written casts and (b) Semantics-boundary reconciliation per `[19 §3.7](19_expression_flow.md)`. Future canonical-with-deviations promotion (if per-engine divergence proves narrow enough) tracked under `[TD-REGISTRY-BINOP-LATTICE]`.

## 6. Portability Model

`FunctionSpec` carries **no portability flag**. The §4.1 intersection policy makes every canonical entry portable-by-construction; for adapter-extended entries (§7), portability is implicit in the registration (the adapter that registers the function IS its supported engine).

Per-engine detail (native-name remap, arity quirks, cast-semantics corner cases, emulation strategies) lives in `registry/functions_mapping.md`. Adapters read both; authors working against the canonical catalog read only `14a`.

**Missing-function policy.** Hard error at `adapt` time when the target engine does not natively support a registered function and no emulation strategy applies — `AdaptErrorKind::UnsupportedFunction { name, engine }`. Mirrors `13` / `registry/types_mapping.md`'s `UnsupportedType` precedent; aligns with `00 §9` **I6** / **I12** (no runtime-deferred stubs).

## 7. Adapter Extension API

Engine-specific functions (Spark `collect_set`, DuckDB `list_extract`, DataFusion `array_to_string`, …) layer onto the core registry via compile-time trait-impl registration:

```rust
pub trait RegistryExtension {
    const ADAPTER_ID: &'static str;
    const FUNCTIONS: &'static [FunctionSpec];
}
```

Adapter crates impl `RegistryExtension` on a zero-size marker; `function_registry()` enumerates linked extensions at startup and folds them into the §2.2 flat map. Extension wiring detail (feature gate vs build.rs aggregation): `[TD-REGISTRY-EXTENSION-WIRING]`.

**Collision policy.** Hard reject at registry initialization — both adapter-vs-core and adapter-vs-adapter. Three error variants per §8: `ReservedTagCollision` (name shadows a `14 §6.4.1` reserved AST tag), `AdapterFunctionShadowsCore` (name already in §4), `AdapterFunctionCollision` (two adapters registered same name). Because the registry is `&'static`, these surface as panics at `function_registry()` initialization — adapter misconfiguration is a build-time problem. No tier precedence, no signature-compatible shadowing. Engine-specific semantics overrides for canonical entries are expressed via `registry/functions_mapping.md`'s rewrite tiers, not re-registration.

## 8. Error Model

Function-resolution errors feed `[19 §8.2](19_expression_flow.md)`'s error roster.

<!-- cross-cut: none -->

| Variant | When |
|---|---|
| `CompileError::UnknownFunction { name }` | `FunctionCall.name` not in the sealed registry. |
| `CompileError::FunctionArityMismatch { name, expected, got }` | Call-site arity outside the spec's declared range. |
| `CompileError::NoMatchingSignature { name, arg_types, tried_signatures }` | No overload matches; per `14 §5.4` no implicit coercion — author must `CAST` explicitly. |
| `CompileError::ReservedTagCollision { tag, source }` | Adapter registration shadows a `14 §6.4.1` reserved AST tag. |
| `CompileError::AdapterFunctionShadowsCore { name, adapter }` | Adapter registered a name already in §4. Surfaces as panic at `function_registry()` initialization. |
| `CompileError::AdapterFunctionCollision { name, adapters }` | Two adapters registered the same name. Same surfacing. |
| `AdaptErrorKind::UnsupportedFunction { name, engine }` | Adapter cannot render a registered function for its target engine. |

## 9. Cross-References

- `[14_expressions.md](14_expressions.md)` — `Expr::FunctionCall { name, args }` consumer; `§6.4` Declarative-block tag dispatch; `§9.2/§9.3` crate-placement concept-of-record (item Y / V4).
- `[18_entities.md](18_entities.md)` — model-level `AdditivityType` carried per-Measure / per-Metric (composed with function-level `Additivity` from §3.6 per `19 §6.5`).
- `[19_expression_flow.md](19_expression_flow.md)` — compile-pipeline consumer; effective-additivity composition (§6.5).
- `[../apis/32_semstrait_model.md](../apis/32_semstrait_model.md)` — crate home of the `CanonicalFn` **name identity** (public authoring vocabulary) and its model-local function table (item Y / V4).
- `[../apis/35_semstrait_ir.md](../apis/35_semstrait_ir.md)` — crate home of the resolved `FunctionRegistry`, `FunctionSpec`, the signature/return-type machinery, and `Additivity` / `DimensionAxis`.
- `[../apis/36_semstrait_adapter.md](../apis/36_semstrait_adapter.md)` — adapter rendering of resolved `FunctionCall`s.
- `registry/functions_mapping.md` — per-engine catalog of native names, rewrite tiers, emulation strategies, portability gaps.

## Testing Strategy

- `FunctionRegistry` lifecycle: sealed `&'static` singleton via `function_registry()`, `OnceLock`-backed initialization, immutable post-seal.
- Lookup: strict case-sensitive exact match on canonical name, returns `Option<&FunctionSpec>`, no aliases.
- `FunctionSpec` shape: `canonical_name`, `category`, `NonEmpty<FnSignature>`, `Option<Additivity>` (None for scalar/non-aggregate).
- Signature polymorphism: first-exact-match-by-arg-types declaration-order resolution, no TypeClass generics, no implicit coercion, `NoMatchingSignature` when no overload matches with full `tried_signatures` list.
- `ReturnTypeRule` variants: `Fixed`, `SameAs`, `Promoted`, `DecimalScaleZero` (Decimal overloads only), `Custom` (non-PartialEq, function-pointer carriage).
- Canonical catalog population: adapter-capability intersection (DataFusion ∩ Spark ∩ DuckDB), portable-by-construction, §4.2–§4.6 coverage.
- Closed-five aggregates: `Sum`/`Avg`/`Count`/`Min`/`Max` carried by `Expr::Aggregate { op: AggregationOp }`, NOT registry entries.
- Function-level `Additivity`: `Additive` (SUM/COUNT/MIN/MAX), `SemiAdditive { axes }` (temporal axis), `NonAdditive` (AVG/COUNT_DISTINCT/MEDIAN); `pre_aggregatable()` and `reaggregation()` branching logic on `Additivity` itself.
- `AdditivitySource` trait: v1 function-only (aggregate op → `Additivity`), model-level `AdditivityType` deferred, zero-cost generic planner code.
- Adapter `RegistryExtension`: compile-time trait-impl registration, flat-map fold at startup, collision policy (hard reject, panics on `ReservedTagCollision`/`AdapterFunctionShadowsCore`/`AdapterFunctionCollision`).
- Error variants: `UnknownFunction`, `FunctionArityMismatch`, `NoMatchingSignature`, `ReservedTagCollision`, `AdapterFunctionShadowsCore`, `AdapterFunctionCollision` (initialization panics), `UnsupportedFunction` (adapt-time).

See the shared behavioral harness: [`../testing/50_behavioral_contract_gates.md`](../testing/50_behavioral_contract_gates.md)
