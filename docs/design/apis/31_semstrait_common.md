---
id: 31-semstrait-common
status: ratified
version: 2.4
impl: implemented
---


# 31. semstrait-common

> **Status:** ratified. `semstrait-common` is the workspace's **infrastructure crate**: it owns the expression traversal trait family (`Tree` / `Visitor` / `Rewriter` + `ValidateError`, item Y / V5), the diagnostic envelope, byte-blob `io` transport, and shape-only constraint DSL. Stage-agnostic; carries the traversal traits but **none** of the expression vocabulary or plan tree. Every other workspace crate depends on it directly or transitively.

## 1. Purpose and Scope

`semstrait-common` is the substrate crate at the root of the workspace DAG (I7). It owns stage-agnostic infrastructure: the expression traversal trait family (`Tree` / `Visitor` / `Rewriter` + `ValidateError`), the diagnostic envelope, the byte-blob `io` transport, and the shape-only constraint DSL.

Surface roster: see §1.1.

### 1.1 What `semstrait-common` OWNS

- The expression **traversal trait family** (`§3`): `Tree`, `Visitor<N>`, `Rewriter<N>`, plus the narrow `ValidateError` kind they return. Impl'd by both `model::Expr<L>` and `ir::PlanNode`; the DAG leaf is their natural home (orphan-safe — each `impl Tree` lives with its local type). Placement per item Y / V5 — rationale at `[14 §9.1](../foundations/14_expressions.md)`.
- The cross-cutting **diagnostic primitives** (`§4`): `Diagnostic<K>`, `Diagnostics<K>`, `Severity`, `Location`, `Span`, `SourceId`, and the `Diagnose` trait. Per `[30 §5](30_api_contracts.md)`. Every consumer crate composes these around its own per-stage typed-kind enum.
- The byte-blob **`io` transport** (`§5`): the `FileSystem` trait, `Path`, `FileEntry`, `IoErrorKind`, the `Persistable` / `Repository<T>` persistence seam, and the `backends::{local_fs, s3_fs}` family. Full sub-spec at `[31b](31b_semstrait_common_io.md)`.
- The shape-only **constraint-DSL toolkit** (`§6`): `MeasureConstraints`, `DimensionConstraints`, `AggregationConstraints` per `[11 §8.3](../foundations/11_names_and_scopes.md)` / `§8.4`. Shared between `semstrait-model` (Measure / Metric authoring carriers) and `semstrait-planner` (constraint evaluation).
- The canonical **entity-identity alias** (`§2.1`): `EntityId` — canonical UUID text shared by model authoring (`32 §2`), the manifest single-id lane (`33 §4.3.1`), and the IR semantic graph (`35 §2A`). Stage-agnostic per §1.3 (consumed by three stages; carries no per-stage vocabulary).

### 1.2 What `semstrait-common` does NOT own

| Surface | Owning crate | Doc |
|---|---|---|
| Authored type vocabulary (`DataType`, `Grain`) | `semstrait-model` | `[32](32_semstrait_model.md)` (item Y / V2) |
| Non-authored type vocabulary (`TypeClass`, `Schema`, `SchemaColumn`) | `semstrait-ir::types` | `[35](35_semstrait_ir.md)` |
| Expression-tree types (`Expr<L>`, leaves, op enums, `Literal`, `ColumnRef`, `SemanticsName`, `CanonicalFn`-name, `ExprLeaf`) | `semstrait-model` | `[14 §9.2](../foundations/14_expressions.md)`, `[32](32_semstrait_model.md)` |
| Resolved expression types (`Parameter`, `Window` support, `FunctionRegistry`), and the wider `CompileError` / `IrErrorKind` (they carry `CanonicalFn` / `DataType` / `Name`) | `semstrait-ir` | `[35](35_semstrait_ir.md)` |
| Plan-tree types (`SemanticPlan`, `PlanNode`, `NodeMeta`, `SourceRef`, `ResolvedColumn`) | `semstrait-ir` | `[35](35_semstrait_ir.md)` |
| YAML authoring surface (bare `Expr<L>` fields with field-level friendly serde/schemars adapters, reserved-tag dispatch — doc 45) | `semstrait-model` | `[14 §9.3](../foundations/14_expressions.md)`, `[32](32_semstrait_model.md)` |
| `SemanticManifest`, planner + plan tree, engine identity / dialect, catalog / filesystem, name resolution, domain load / dump wrappers | per-crate | `[registry.json](../registry.json)` |

### 1.3 Design posture — workspace-wide infrastructure

Placement rule: a type belongs in `semstrait-common` iff (a) consumed by two or more stages and (b) carries no per-stage vocabulary (no expression tree, no plan tree, no engine identity, no canonical type roster).

- Stage-agnostic: only types meeting both clauses above.
- Workspace-DAG root: zero `semstrait-*` workspace dependencies.
- No engine-identity dependencies: `datafusion`, `arrow`, `duckdb`, `substrait` are rejected.
- I/O posture: `io` feature default ON pulls `tokio` / `bytes` / `object_store` / `async-trait` / `futures-util`; `--no-default-features` keeps only `thiserror`. Cloud SDKs sit behind additional opt-in flags. Full spec: `[31b](31b_semstrait_common_io.md)`.

### 1.4 Boundary rule for expression-bodied future constraints

Shape-only constraint blocks live in `semstrait-common::constraints` (§6); any constraint carrying `Expr<L>` or other expression-tree vocabulary lives in `semstrait-ir`.

Triggers: a new constraint field references `Expr<L>`, `UnresolvedSemanticLeaf`, or any expression-bearing type.

Escalation: introduce the new constraint in `semstrait-ir`, never in `semstrait-common`; cross-reference `[35 §1.1](35_semstrait_ir.md)`.

## 2. Module Layout

Top-level `pub mod` structure. One module per cohesive concept; no cross-module cycles.

```
semstrait-common
├── identity            // EntityId (canonical UUID-text identity alias)
├── tree                // Tree, Visitor<N>, Rewriter<N>, ValidateError (item Y / V5)
├── diagnostic           // Diagnostic<K>, Diagnostics<K>, Severity, Location, Span,
│                        //   SourceId, Diagnose trait
├── constraints          // MeasureConstraints, DimensionConstraints, AggregationConstraints
└── io                   // FileSystem, Path, FileEntry, IoErrorKind, Persistable,
                         //   Repository<T>, FileSystemRepository<T>
                         //   + backends::{local_fs, s3_fs, object_store_fs}
                         //   (feature "io", default ON; s3 under "io-aws")
                         //   Full spec: 31b
```

**Five modules.** Each module is independently importable; `lib.rs` re-exports a curated convenience surface (§7). The `identity` module is a single type alias (`§2.1`) — it gets its own module so the workspace-shared id lane sits at the DAG root without dragging the diagnostic or io surfaces along.

**Split rationale:**

- `tree` is its own module so the traversal trait family (`§3.1`) ships independently of the diagnostic envelope. Its only implementers (`model::Expr<L>`, `ir::PlanNode`) live above; the leaf is the orphan-safe home (item Y / V5). The co-moved `ValidateError` (the `with_new_children` return type) ships here too.
- `diagnostic` is its own module so the generic envelope (`Diagnostic<K>`, `Diagnose`) ships with no per-stage kind enum coupling. The wider per-stage kind enums (`ParseErrorKind`, `CompileError`, `IrErrorKind`, `PlanErrorKind`, `AdaptErrorKind`, …) live in their owning crates; only `Diagnose` ties them to the envelope. (The narrow `ValidateError` is the exception — it co-moves to the `tree` module with `Tree`, per §3.1.)
- `constraints` is separated from `diagnostic` because the two evolve at different cadences — `diagnostic` changes when stability rules around envelope rendering change; `constraints` changes when authoring-DSL shapes evolve. Separating lets downstream consumers import one without the other's recompile cost.
- `io` — full split rationale and back-end roster live in `[31b §2](31b_semstrait_common_io.md)`.

### 2.1 Identity alias (`EntityId`)

```rust
/// Canonical entity identity: UUID text (lowercase, hyphenated).
///
/// One identity lane across every stage. Authored ids are UUIDv7
/// (`32 §2` / `§9.0.1`); compile-synthesised ids are deterministic
/// UUIDv5 (name-based, `33 §9.1`) so identical `(model, catalog)` inputs
/// yield byte-identical manifests (I4). The version nibble (7 vs 5)
/// records provenance but is informational — consumers validate
/// canonical-UUID-text shape, not the specific version.
pub type EntityId = String;
```

**The validated contract is shape, not derivation.** Every consumer — `32` SR-11 at parse, `33 §10` CX1 at manifest load — checks exactly these properties:

| Property        | Value                                                       |
| --------------- | ----------------------------------------------------------- |
| Length          | exactly 36 bytes (= 36 chars; the charset is ASCII)          |
| Layout          | `8-4-4-4-12`, hyphens at indices 8, 13, 18, 23              |
| Charset         | lowercase hex `[0-9a-f]` in the 32 non-hyphen positions     |
| Parses as UUID  | yes — canonical text equals the parsed value's re-rendering  |
| Not nil         | `00000000-0000-0000-0000-000000000000` is rejected          |
| Version nibble  | **unconstrained** — provenance only, never validated        |

Rejecting the nil UUID is what the shape rule adds over "parses as a UUID": nil satisfies every layout property but is the canonical *absent* value, and `Option<EntityId>` already owns absence at the authoring layer (`32 §1.4`) — admitting a second spelling of "no id" would give one state two encodings.

**Why the version nibble is excluded.** Two derivation strategies already coexist (v7 authored, v5 derived), and both are implementation choices that have changed once and may change again: a future requirement for content-addressed ids on *every* entity — not just the manifest's synthesised ones — would swap the derivation without altering a single thing a consumer relies on. Consumers depend on ids being fixed-width, filename-safe, and comparable; none depends on how the bytes were produced. Pinning the nibble in a validation rule couples every consumer to the current derivation and forces a ratified-rule amendment each time it moves — which has already happened once.

Two properties make the shape rule the load-bearing one:

- **Ids are path segments.** `31b §4`'s `key_for` interpolates the id straight into a storage key (`"{base}/{id}.{extension}"`), so fixed length and a hyphen-and-lowercase-hex charset are what keep keys collision-free, traversal-free, and portable across local FS, S3, GCS, and Azure. A version nibble contributes nothing here. The id is case-folded on the way in (`47 §4a` **R1**), so a charset claim of *lowercase* hex is what the key actually carries, whatever case the caller passed.
- **Nothing reads the version.** No ordering, routing, or dispatch anywhere in the workspace branches on it — ids are never the ordering axis (`32 §2`: name-ordered projection, and `33` likewise), so the v7 timestamp is not decodable-and-used, merely present.

**Why `semstrait-common`.** `EntityId` is consumed by three stages that sit at different depths of the workspace DAG (I7): model authoring (`32 §2`), manifest persistence (`33 §4.3.1`), and the IR semantic graph (`35 §2A`, which references entity ids in its node/edge payloads). `semstrait-ir` sits **below** `semstrait-model`, so the alias cannot live in the model crate without an illegal upward dependency from `ir`. The DAG root is the only home reachable by all three. It satisfies the §1.3 placement rule: multi-stage, and carrying no per-stage vocabulary (a UUID-text alias has no expression tree, plan tree, engine identity, or type roster).

It is a transparent `String` alias — not a newtype — so it adds zero method surface and re-exports trivially. The shape contract above, plus global uniqueness, is enforced at the construction / validation sites in the owning stages (`32 §9.0`, `33 §10` CX1), not by `semstrait-common`. Because the alias is transparent, the shape is a *checked-at-the-boundary* invariant rather than a type-enforced one: any stage that mints an id owes the check, and no stage may add a stricter one — a stage-local version or prefix rule would reject ids another stage legally produced.

## 3. Trait Family

`semstrait-common` exposes two trait groups: the expression **traversal trait family** `Tree` / `Visitor<N>` / `Rewriter<N>` (§3.1), and the diagnostic `Diagnose` trait (§4.4). Every per-stage kind enum implements `Diagnose`.

### 3.1 Traversal trait family (`Tree`, `Visitor<N>`, `Rewriter<N>`) + `ValidateError`

The traversal trait family relocated to `semstrait-common` per item Y / V5; the trait shapes are ratified at `[14 §3.1](../foundations/14_expressions.md)` and the placement rationale at `[14 §9.1](../foundations/14_expressions.md)`. This doc carries the crate contract, not the trait bodies.

- `Tree`, `Visitor<N>`, `Rewriter<N>` are the node-shape-agnostic traversal contract. Their only implementers are `model::Expr<L>` (authoring) and `ir::PlanNode` (resolved plan tree). The DAG leaf is the natural home reachable by both — orphan-safe, since each `impl Tree` lives with its local type in the owning crate.
- `ValidateError` co-moves with `Tree`: it is `Tree::with_new_children`'s return type, so it cannot stay above the trait. It is primitive-only (`usize` / `String` / `&'static str` — no `DataType` / `CanonicalFn` / `Name` payload) and already implements `Diagnose` / `Severity` (§4), so it sits cleanly in the leaf.
- **Boundary.** `ValidateError` is the *only* expression-side error kind common owns. The wider `CompileError` / `IrErrorKind` stay in `semstrait-ir` (they carry `CanonicalFn` / `DataType` / `Name`). `ExprLeaf` does NOT come here — it names `DataType`, so it lives in `semstrait-model` per `[14 §9.2](../foundations/14_expressions.md)`. The `RegistryExtension` adapter-contribution hook lives in `semstrait-ir::functions` per `[35 §8.2](35_semstrait_ir.md)`.

## 4. Public Types — Diagnostic Primitives

`semstrait-common` provides the **diagnostic envelope** every consumer crate composes around its own per-stage typed-kind enum, plus the `Diagnose` trait those kinds implement. Authoritative sub-shape per `[30 §5](30_api_contracts.md)`.

### 4.1 `Severity`

```rust
/// Per `30 §5.2`. Two variants only — `Info` retired into the `tracing`
/// channel (`30 §6`).
#[non_exhaustive]
pub enum Severity {
    Error,
    Warning,
}
```

Severity carries message intent only; control flow (accumulating vs fail-fast) lives in the function signature, not the diagnostic. Fail-fast stages return `Result<T, Diagnostic<K>>`; accumulating stages return `Diagnostics<K>` plus an outcome. Both modes use the same envelope shape.

### 4.2 `Location`, `Span`, `SourceId`

```rust
/// Source-level location. Per `30 §5.3`. `#[non_exhaustive]`, so producing
/// crates build it via `Location::new` rather than a struct literal.
#[non_exhaustive]
pub struct Location {
    pub source: SourceId,
    pub span: Span,
}

impl Location {
    pub const fn new(source: SourceId, span: Span) -> Self;
}

/// Half-open byte-offset span into the source buffer.
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// The zero-width span `0..0` — a known source document with no
    /// resolved byte range yet (e.g. a name-level validation that
    /// predates span tracking). Symmetric with `SourceId::unknown`.
    pub const fn unknown() -> Self;
}

/// Opaque identifier for the originating source document. Producing crates
/// mint a named id with `SourceId::named` (`semstrait-model` for YAML files,
/// inline-string for tests, …); `SourceId::unknown` is the no-origin sentinel.
/// Not `Copy`: a named id owns a heap string. `semstrait-common` exposes only
/// the shape consumers need.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct SourceId(/* crate-private */);

impl SourceId {
    pub const fn unknown() -> Self;
    /// Mint a named id from a source label. Accepts `&str` or `String`
    /// (both `Into<Box<str>>`); the id is immutable after construction.
    pub fn named(label: impl Into<Box<str>>) -> Self;
    pub fn as_str(&self) -> &str;
}
```

`Span` retains its `[30 §5.3](30_api_contracts.md)` byte-range shape, plus `Span::unknown()` for the known-source / no-byte-range case. `SourceId` is opaque — it is NOT `#[non_exhaustive]` because its variant set is private; the crate's public surface is exactly `SourceId::unknown()` / `SourceId::named(..)` / `SourceId::as_str()` plus the `Eq` / `Hash` / `Display` traits consumers need. It is **not** `Copy` — a named id owns a `Box<str>` (chosen over `String`: a source id is immutable after construction, so `String`'s spare-capacity word is dead weight, and the non-null `Box` pointer lets the inner enum reuse the niche for the `Unknown` sentinel, keeping the shared `Diagnostic<K>` envelope compact).

### 4.3 `Diagnostic<K>` — generic envelope

```rust
/// Generic envelope wrapping a per-stage typed kind. Per `30 §5.1`.
/// Each consumer crate plugs in its own `K: Diagnose`.
#[non_exhaustive]
pub struct Diagnostic<K: Diagnose> {
    pub kind: K,
    pub severity: Severity,
    pub location: Option<Location>,
    pub notes: Vec<String>,
}

/// Type alias for an accumulating-stage diagnostic vector.
pub type Diagnostics<K> = Vec<Diagnostic<K>>;
```

Construction goes through a minimal primitive pair on `Diagnostic<K>`:

```rust
impl<K: Diagnose> Diagnostic<K> {
    /// Seeds severity from `kind.severity_default()`, location = None, notes = vec![].
    pub fn new(kind: K) -> Self;

    pub fn with_severity(self, severity: Severity) -> Self;   // override default
    pub fn with_location(self, location: Location) -> Self;   // attach span
    pub fn with_note(self, note: impl Into<String>) -> Self;  // append note

    /// Re-wrap this diagnostic's `kind` into a different kind enum `K2`,
    /// preserving `severity` / `location` / `notes` unchanged. Used when a
    /// downstream stage lifts a narrower per-pass kind into its own fused
    /// aggregate kind (e.g. model's `ModelBuildErrorKind::{Parse, Validate}`).
    pub fn map_kind<K2: Diagnose, F: FnOnce(K) -> K2>(self, f: F) -> Diagnostic<K2>;
}

/// Partition a diagnostic batch into `(errors, warnings)` by severity, each
/// preserving the input's relative order. An accumulating stage's `Result`
/// error arm fires when the returned `errors` vec is non-empty.
pub fn split_by_severity<K: Diagnose>(diags: Diagnostics<K>)
    -> (Diagnostics<K>, Diagnostics<K>);
```

`semstrait-common` exposes no further builder. Per-stage consumer crates wrap the construction primitive in their own private helper (typically `fn diag(kind: K) -> Diagnostic<K>` for fail-fast sites, or a small builder for accumulating sites) so the call site stays idiomatic to each stage. The chainable `with_*` mutators are the *only* mechanism for attaching `location` / `notes` / overriding severity from outside this crate; the struct stays `#[non_exhaustive]` so future fields remain additive (I10).

Two helpers serve the multi-stage accumulating model (`32`): `Diagnostic::map_kind` lifts a per-pass kind into a fused aggregate kind at a stage boundary (preserving the envelope metadata), and the free `split_by_severity` partitions an accumulated `Diagnostics<K>` batch so a stage can fail on errors while still surfacing warnings.

### 4.4 `Diagnose` trait

```rust
/// The trait every per-stage kind implements. Per `30 §5.4`.
pub trait Diagnose {
    /// Human-readable rendering. Powers the `Display` impl on
    /// `Diagnostic<K>`. Must not include line breaks (callers add
    /// their own framing).
    fn message(&self) -> String;

    /// Default severity for this variant. Construction sites may
    /// override; most callers accept the default.
    fn severity_default(&self) -> Severity;

    /// Foreign-error chain for `std::error::Error` interop. Default
    /// `None`. Variants wrapping foreign errors override to return
    /// `Some(&inner)`.
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
```

`Diagnose` is **open** — third-party kind enums (e.g. a downstream plugin's typed error) implement it and slot into the `Diagnostic<K>` envelope.

### 4.5 Blanket impls on `Diagnostic<K>`

```rust
impl<K: Diagnose> std::fmt::Display for Diagnostic<K> { /* delegates to K::message() */ }
impl<K: Diagnose + std::fmt::Debug> std::error::Error for Diagnostic<K> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.kind.cause()
    }
}
```

Both blanket impls live in `semstrait-common::diagnostic` because of the orphan rule — no other crate could provide them. `Diagnose::cause()` is the source of truth for the `std::error::Error` chain.

### 4.6 `Diagnostics<K>` ergonomics

`Diagnostics<K>` is a transparent `Vec<Diagnostic<K>>` alias, so all `Vec` methods apply directly. Per `[30 §5.6](30_api_contracts.md)`, fused helpers in `semstrait-api` use a sum-typed kind (`SemStraitErrorKind`) and lift per-stage results via `From<ParseErrorKind>`, `From<ValidateError>`, `From<CompileError>`, etc. — `From` impls live on the fused-kind enum, not on `Diagnostic<K>`.

## 5. Public Types — `io` Transport

Full spec: `[31b](31b_semstrait_common_io.md)`. This section is a re-export sketch only; consult `31b` for trait shapes, back-end semantics, and `IoErrorKind` variants.

```
pub use self::io::{
    FileSystem,                         // 37-filesystem §5.2 — object-safe transport trait
    FileSystemErrorKind,                // 37 §8.2 — #[non_exhaustive] transport error enum
    IoErrorKind,                        // 31b §7 — #[non_exhaustive] error enum
};

pub mod io {
    pub struct Path;                    // URI-shaped newtype
    pub struct FileEntry;
    pub struct InMemoryFileSystem;      // process-local double
    pub trait Persistable;              // 31b-io-repository §2
    pub trait Repository<T>;            // 31b-io-repository §3
    pub struct FileSystemRepository<T>; // 31b-io-repository §4
}

pub mod io::backends::local_fs { pub struct LocalFileSystem; }
#[cfg(feature = "io-aws")]
pub mod io::backends::s3_fs    { pub struct S3FileSystem; }
```

Internally every back-end thin-wraps `object_store` (Apache Arrow project); `object_store` types never appear on a public signature. No `FileSystem` impl carries an escape hatch (`37-filesystem §6`).

**The `Source` / `Sink` transport is gone.** `Source`, `Sink`, `FromIoBytes`, `IntoIoBytes`, `io::Location` and `backends::{memory, local, s3}` were deleted (`47 §4a` **R6**, closing `[TD-STRIP-LANE-A-FROM-31]`): 1302 lines with zero consumers outside their own module once the `FileSystem` backends landed. Per DOC-I6 the sections that specified them are removed, not retained — `31b_semstrait_common_io.md` is the live sub-spec for what remains. Note `diagnostic::Location` (`§4.2`) is a **different type** and stays.

## 6. Public Types — Constraint Family

Per `[11 §8.3](../foundations/11_names_and_scopes.md)`–`§8.4`. Shape-only DSL block attached to `Measure` and `Metric` carriers. All three sub-blocks are shape-only: scalar fields, scalar lists, no expression vocabulary. Per the §1.4 boundary rule, expression-bodied future constraints belong in `semstrait-ir`, not here.

The current type name `MeasureConstraints` is a legacy artifact for the `Measure` + `Metric` carriers (`[11 §8.4.3](../foundations/11_names_and_scopes.md)`, `[TD-CONSTRAINT-RENAME]`); the three sub-blocks below retain their current names to avoid a breaking rename before the SemanticManifest-schema revision pass.

### 6.1 `MeasureConstraints`

```rust
/// Constraint DSL block attached to Measure and Metric carriers.
/// Per `11 §8.4.1` / `§8.4.2`. Shape-field (`11 §8.8`).
#[non_exhaustive]
pub struct MeasureConstraints {
    pub dimensions: Option<DimensionConstraints>,
    pub aggregations: Option<AggregationConstraints>,
}

impl MeasureConstraints {
    pub fn none() -> Self;
    pub fn is_empty(&self) -> bool;
}
```

### 6.2 `DimensionConstraints`

```rust
/// Three-way set-membership policy over Dimension names. Per `11 §8.3` /
/// `§8.4.1`. All three fields optional; AND-combined.
#[non_exhaustive]
pub struct DimensionConstraints {
    pub one_of: Vec<String>,
    pub none_of: Vec<String>,
    pub all: Vec<String>,
}

impl DimensionConstraints {
    pub fn none() -> Self;
    pub fn is_empty(&self) -> bool;
}
```

### 6.3 `AggregationConstraints`

```rust
/// Two-way whitelist/blacklist policy over UPPERCASE aggregation-name
/// tokens (matching the `AggregationOp` enum names plus `COUNT_DISTINCT`
/// for the `distinct: true` encoding). Per `11 §8.4.1`.
#[non_exhaustive]
pub struct AggregationConstraints {
    pub allowed: Vec<String>,
    pub prohibited: Vec<String>,
}

impl AggregationConstraints {
    pub fn none() -> Self;
    pub fn is_empty(&self) -> bool;
}
```

The `allowed` / `prohibited` fields take UPPERCASE tokens (`COUNT_DISTINCT` included; this is not an `AggregationOp` enum variant but the `Expr<L>::Aggregate { op: Count, distinct: true, ... }` encoding). Token matching is planner-owned per `[11 §8.6](../foundations/11_names_and_scopes.md)`; `semstrait-common` exposes the shape, not the matching logic.

## 7. Public API Surface Sketch

One rustdoc-style line per exported item, grouped by module. Doubles as the "test-the-contract" target — an integration test enumerates this list against `cargo public-api` output.

### 7.1 `tree`

```
pub trait  Tree                                          // children + with_new_children -> Result<Self, ValidateError>; apply/transform defaults
pub trait  Visitor<N>                                    // type Output; f_down + f_up -> ControlFlow
pub trait  Rewriter<N>                                   // f_down + f_up -> Result<N, ValidateError>
pub enum   ValidateError                                 // primitive-only kind; impls Diagnose. Shape per [14 §3.1].
```

### 7.2 `diagnostic`

```
pub struct Diagnostic<K: Diagnose>                        // { kind, severity, location, notes }; new/with_*/map_kind
pub type   Diagnostics<K> = Vec<Diagnostic<K>>
pub fn     split_by_severity<K>(Diagnostics<K>) -> (errors, warnings)
pub enum   Severity                                       // Error | Warning
pub struct Location                                       // { source: SourceId, span: Span }; Location::new
pub struct Span                                           // { start, end }; Span::unknown()
pub struct SourceId                                       // opaque; unknown() + named(..) + as_str(); not Copy
pub trait  Diagnose                                       // fn message + severity_default + cause
// Blanket: Display / std::error::Error on Diagnostic<K> via Diagnose.
```

### 7.3 `constraints`

```
pub struct MeasureConstraints                             // { dimensions, aggregations }
pub struct DimensionConstraints                           // { one_of, none_of, all }
pub struct AggregationConstraints                         // { allowed, prohibited }
// Each of the three exposes fn none() and fn is_empty(&self) -> bool.
```

### 7.4 `io` (feature `io`, default ON)

Full surface in `[31b](31b_semstrait_common_io.md)`. Re-export sketch in §5.

### 7.5 Crate-root re-exports (stable convenience surface)

```rust
// lib.rs
pub use crate::identity::EntityId;
pub use crate::tree::{Tree, Visitor, Rewriter, ValidateError};
pub use crate::diagnostic::{
    Diagnostic, Diagnostics, Severity, Location, Span, SourceId, Diagnose,
    split_by_severity,
};
pub use crate::constraints::{
    MeasureConstraints, DimensionConstraints, AggregationConstraints,
};
// io re-exports per §5 (feature-gated).
```

Non-root re-exports of internal helpers are forbidden — consumers either import `semstrait_common::Diagnostic` or `semstrait_common::diagnostic::Diagnostic`, never both via two paths.

## 8. Feature Flags

| Feature | Default | Gates | Reason |
|---|---|---|---|
| `serde` | OFF | `Serialize` / `Deserialize` on every public type in this crate: `Diagnostic<K>` (where `K: Serialize`), `Severity`, `Location`, `Span`, `SourceId`, `ValidateError`, the constraint family | keeps the crate's dependency footprint minimal for consumers that only need types; `semstrait-ir` / `semstrait-model` / `semstrait-manifest` enable it transitively |
| `schemars` | OFF | JSON schema derivations on the same types | consumers needing JSON-Schema emission pay a second compile cost |
| `io` | **ON** | The `io` module — `FileSystem` / `Path` / `FileEntry` / `IoErrorKind` / `Persistable` / `Repository<T>` / `FileSystemRepository<T>` + `backends::{local_fs, object_store_fs}` + `InMemoryFileSystem`; pulls `tokio`, `bytes`, `object_store` (`fs` feature), `async-trait`, `futures-util` | ergonomic common case for every downstream crate that wants transport; disable with `default-features = false` for pure-type consumers (see `[31b §9.1](31b_semstrait_common_io.md)`) |
| `io-aws` | OFF | `backends::s3_fs::S3FileSystem`; enables `object_store/aws` | cloud SDK footprint stays opt-in |

Future `io-http`, `io-gcs`, `io-azure` land additively behind the same gating pattern.

## 9. Dependency Posture

### 9.1 External dependencies

```toml
[dependencies]
thiserror = "^"

[dependencies.serde]
version = "^"
optional = true
features = ["derive"]

[dependencies.schemars]
version = "^"
optional = true

[dependencies.tokio]
version = "^"
optional = true
features = ["rt", "fs", "io-util", "macros"]

[dependencies.bytes]
version = "^"
optional = true

[dependencies.object_store]
version = "^"
optional = true
default-features = false

[features]
default  = ["io"]
serde    = ["dep:serde"]
schemars = ["dep:schemars", "serde"]
io       = ["dep:tokio", "dep:bytes", "dep:object_store", "dep:async-trait", "dep:futures-util", "object_store/fs"]
io-aws   = ["io", "object_store/aws"]
```

**Runtime dependency posture.** Under default features, `semstrait-common` pulls `tokio`, `bytes`, `object_store` (`fs` feature; no cloud SDKs unless `io-aws` is enabled), `async-trait`, and `futures-util`. Under `--no-default-features`, only `thiserror` remains. `dashmap` was dropped with **R6** — its only consumers were the deleted `backends::memory` registry and `backends::s3` client cache.

**`object_store` as internal detail.** Consumers never see `object_store::ObjectStore`, `object_store::Path`, or any of its error types on a public signature. No escape hatch survives **R6** — the one that existed was on the deleted `S3SourceBuilder`. See `[31b §1.4](31b_semstrait_common_io.md)`.

**No engine-identity dependencies.** No `datafusion`, no `arrow`, no `spark-*`, no `duckdb`, no `substrait`. These live in `semstrait-adapter` and its per-engine modules.

### 9.2 Internal (workspace) dependencies

**Zero.** `semstrait-common` is the root of the workspace DAG per I7. Every other crate depends on `semstrait-common` directly or transitively; `semstrait-common` depends on no workspace crate. Attempting to add a `semstrait-*` dependency to `Cargo.toml` is a compile error in CI.

## 10. Invariants Upheld by the Crate

Concrete crate-level guarantees mapping to `[00 §9](../00_overview.md)` invariants. The canonical-type invariants anchor across the item Y split: I2 (logical types only) on the authored `DataType` / `Grain` now homed in `semstrait-model` (V2); I1 / I5 / I10 in `semstrait-ir` per `[35](35_semstrait_ir.md)`, where the resolved expression / plan tree and the non-authored `TypeClass` / `Schema` / `SchemaColumn` live.

| Invariant | `semstrait-common` guarantee |
|---|---|
| **I7** — strict DAG | `Cargo.toml` contains zero `semstrait-*` entries in `[dependencies]`. A CI check greps the manifest and fails on any workspace-internal entry. |
| **I11** — no downward I/O surprises | Transport primitives (`io::Source`, `io::Sink`, `io::Location`, `io::backends::{memory, local, s3}`) live here under the `io` feature flag (ratified in `[31b](31b_semstrait_common_io.md)`). Domain-specific load / dump (`load_model`, `load_manifest`) do not — they live in the crate that owns the typed artifact. `reqwest`, `hyper`, raw `std::net` sockets remain rejected; cloud SDKs (`aws-sdk-s3`) sit behind opt-in `io-aws`. The dependency audit (§9.1) is enforced in CI via `cargo deny`. |
| **I12** — first-class diagnostics | `Diagnostic<K>` and `Diagnose` are the workspace's diagnostic primitives. `Diagnostic<K>` carries `kind: K, severity, location, notes`; the kind decides per-variant rendering and severity defaults via `Diagnose`. No central error-code allocation; stable identification is variant identity. The parallel observability channel (`tracing`) is described in `[30 §6](30_api_contracts.md)`; library code never writes to stdout / stderr. `IoErrorKind` per `[31b §6](31b_semstrait_common_io.md)` and the traversal-side `ValidateError` (§3.1) are each their own kind enum implementing `Diagnose`. |

## Testing Strategy

- EntityId alias identity and transparent String-based contracts (§2.1 canonical UUID-text shape).
- Diagnostic envelope construction via `Diagnostic::new`, chainable `with_*` mutators, and `map_kind` re-wrapping (§4.3 construction primitives).
- Diagnose blanket impls for Display and std::error::Error on `Diagnostic<K>` (§4.5 trait forwarding).
- Tree/Visitor/Rewriter trait conformance: children + with_new_children signatures and ValidateError return types (§3.1 traversal family).
- ValidateError as primitive-only kind (§3.1 boundary rule — no DataType / CanonicalFn payload).
- Constraint-DSL shapes: MeasureConstraints, DimensionConstraints, AggregationConstraints with their none/is_empty helpers (§6 toolkit surface).
- io transport: the `FileSystem` byte primitive and its backend roster, plus the `Persistable`/`Repository<T>` persistence seam (`31b` sub-spec; the `Source`/`Sink` lane this bullet once named was deleted by `47 §4a` **R6** — see `§10`).
- Feature-flag composition: io default-ON gates tokio/bytes/object_store; io-aws opt-in gates S3; serde/schemars gates derive macros (§8 feature roster).
- Zero workspace dependencies: cargo metadata validation that [dependencies] contains no semstrait-* entries (§9.2 I7 guarantee).
- No downward I/O surprises: dependency audit via cargo deny rejecting datafusion/arrow/duckdb/substrait/reqwest/hyper (§9.1 + §10 I11).
- First-class diagnostics: Diagnostic<K> carrying kind/severity/location/notes; Diagnose trait specifying message/severity_default/cause (§10 I12).

See the shared behavioral harness: [`../testing/50_behavioral_contract_gates.md`](../testing/50_behavioral_contract_gates.md)
