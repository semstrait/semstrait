---
id: 30-api-versioning-surface-extensibility
status: ratified
version: 2.1
impl: implemented
---

# API Versioning and Surface Extensibility

This leaf owns the cross-crate semver posture, public vs internal surface policy, and non-exhaustive-by-default policy that bind every crate's API contract.

> **Status:** ratified. The cross-cutting policies in §§2–12 bind every `3x` per-crate doc; the stability table (§13) fixes v1 maturity markers; the non-exhaustive type roster (§4.2), the generic `Diagnostic<K>` envelope (§5), the workspace observability policy (§6), and the per-stage return-shape table (§7) are authoritative. Open reconciliation items are parked in `questions/open/30_questions.md`.
>
> **Ratification delta from earlier drafts.** §6 ("Stable Error-Code Format") is **retired**: the `{SUBSYSTEM}_{SEVERITY}_{NUMBER}` allocation table, reserved-prefix list, and per-subsystem ranges are all gone. Diagnostic identification is now by **per-stage typed-kind variant identity**. The §6 slot now hosts the **observability policy** — a tracing-side companion to the diagnostics-side of §5. `Severity` is reduced from `{Info, Warning, Error}` to `{Error, Warning}`; the `Info` channel moved to `tracing::info!`. Every public stage entry-point that returns errors does so via the typed shapes of §7 — no `Diagnostic` without a `K` parameter, no `Vec<Diagnostic>` of mixed kinds.

## 1. Purpose and Scope

`30` opens the `3x` per-crate API-contract series by fixing the policies every crate's public surface must satisfy. It is the cross-cutting contract each of `31`–`39` refines for its own types, traits, and functions.

**What `30` ratifies:**

- The **semver posture** for the workspace (§2) — what MAJOR / MINOR / PATCH mean when every `semstrait-*` crate ships on the same release cadence.
- The **public-vs-internal** discipline (§3) — minimally public by default, every `pub` type documented, `pub(crate)` for cross-module.
- The **`#[non_exhaustive]`-by-default** policy (§4) and the concrete roster of types this binds in v1.
- The **generic `Diagnostic<K>` envelope** (§5) — generic struct, severity enum, location types, the `Diagnose` trait — authoritative here; other docs refine but do not redefine. The per-stage typed-kind discipline (no central error-code allocation; variant identity is the stable contract).
- The **observability policy** (§6) — `tracing` adoption, no stdout/stderr from library code, canonical field-name vocabulary, Diagnostic ↔ tracing orthogonality.
- The **per-stage error-emission policy** at API boundaries (§7) — mirrors `10 §5` at the crate-surface level.
- The **public-trait rules** (§8) — sealed-pattern vs open, per-trait invariant documentation obligations.
- The **per-crate async posture** (§9) — the crate-granularity refinement of I6 / I11.
- The **feature-flag policy** (§10) — adapter and catalog crates as separate crates, not features.
- **Breaking-change governance** (§11) and the **deprecation lifecycle** (§12).
- The v1 **stability table** (§13) — per-crate maturity marker.

**What `30` does NOT ratify** (forward-refs):

- Per-crate type rosters, exact function signatures, and per-stage kind-variant fanouts — `31`–`39`.
- Migration notes for any MAJOR transition — `42`.
- Deprecation tombstones for retired symbols — `41`.
- Per-engine adapter specifics, dialect quirks — `36`.
- Per-catalog-provider specifics, schema-drift mechanics — `37`.

**Key invariants from `00` that `30` directly upholds:**

- **I7** — the crate DAG is strict and acyclic. `30` codifies this at the public-surface level: no crate may depend upward, and every `pub use` re-export is documented per `31`–`39`.
- **I10** — extensibility. `30 §4` is where the non-exhaustive policy stops being an invariant footnote and becomes a concrete roster.
- **I11** — gated I/O. `30 §9`'s per-crate async posture is the crate-level refinement of I11's "only compile-time and two out-of-band entries" rule.
- **I12** — first-class diagnostics. `30 §5` defines the generic `Diagnostic<K>` envelope and the per-stage typed-kind discipline; `30 §6` defines the parallel tracing channel and the no-stdout invariant. Together they constitute the workspace's diagnostic-and-observability contract.

## 2. Semver Posture

Every `semstrait-*` crate in this workspace participates in a **coordinated release**: the workspace ships a single version number, and every published crate bumps to that version in lock-step. No crate publishes independently of its peers.

This matters because I7's strict DAG means inter-crate API changes can only flow in one direction (upward through the layering), and a coordinated release eliminates the combinatorial surface of version-compatibility across crates.

### 2.1 What MAJOR / MINOR / PATCH mean

| Bump | Triggers |
|---|---|
| **MAJOR** | Any non-additive change to a `pub` type, function, trait, or trait method. Removing or renaming a variant of a per-stage kind enum, adding a required field to a non-`#[non_exhaustive]` struct, changing a function signature (other than relaxing bounds), changing `Diagnostic<K>` shape, changing the per-stage classification (accumulating ↔ fail-fast), changing the canonical `tracing` field-name vocabulary in a non-additive way. Every MAJOR requires a corresponding migration-notes entry. |
| **MINOR** | Additive-only changes. Adding a variant to a `#[non_exhaustive]` sum type, adding a field to a `#[non_exhaustive]` struct, adding a new `FunctionSpec` to the registry, adding a new `pub fn` or `pub struct`, adding a default-impl method to a public trait (with a default), widening a method's accepted input type (e.g. `&str` → `impl AsRef<str>`), introducing a new `#[deprecated]` symbol. Warnings are additive. |
| **PATCH** | Bug fixes that preserve observable behavior. Doc-comment corrections, internal algorithm improvements that produce identical `SemanticManifest`s / `SemanticPlan`s / `EngineArtifact`s, dependency bumps that do not change public types. |

### 2.2 What additivity means for specific changes

Concrete cases that arise repeatedly in the `3x` docs:

- **Adding a new variant to a `#[non_exhaustive]` enum** is MINOR. This is the whole point of the I10 roster.
- **Adding a new field to a `#[non_exhaustive]` struct** is MINOR. Consumers must use construction forms that tolerate new fields (`..Default::default()` or dedicated builders).
- **Widening an error-enum with a new variant** is MINOR, provided the enum is `#[non_exhaustive]` (which per §4.2 all `pub` error enums are).
- **Renaming or removing a per-stage kind variant** is MAJOR. Adding `#[deprecated]` to a variant is MINOR; the variant continues to be produced for at least one MINOR cycle before removal (§12).
- **Adding a new `Diagnostic` severity** is MINOR (`Severity` is `#[non_exhaustive]` per §4.2). Consumers pattern-matching must already have a wildcard arm.
- **Adding a new `EngineArtifact` variant** is MINOR. Adapters that consumed a specific variant via match need a wildcard arm per I10.
- **Widening a public trait's method set** is MAJOR unless the new method carries a default body. Default-bodied methods are MINOR.
- **Narrowing a trait bound on a generic parameter** is MINOR. Widening it is MAJOR.

### 2.3 Pre-1.0

All crates in this workspace are pre-1.0 until the design docs clear the `31`–`39` map and a synchronized v1.0 release is cut. Pre-1.0 semver rules apply: MINOR bumps may carry breaking changes, and every MINOR merits the same migration-note discipline MAJOR does. The v1.0 cut is the moment the stability tiers of §13 lock.

## 3. Public vs Internal Surface

### 3.1 Default stance: minimally public

Every crate starts every type, function, and trait as `pub(crate)`. A symbol is promoted to `pub` only when:

1. An external consumer has a documented need for it (another workspace crate or an end user going through `semstrait-facade`).
2. The symbol's invariants are captured in its doc comment.
3. The symbol is included in the crate's `3x` contract doc.

Symbols that satisfy (1) and (2) but are exposed only for macro-expansion or generic signature inference carry `#[doc(hidden)]` and are excluded from the `3x` surface discussion.

### 3.2 Documentation obligation

Every `pub` type, function, trait, trait method, and associated item MUST carry a doc comment. The doc comment for a public type states:

- What the type represents (the vocabulary anchor from `00 §4` where applicable).
- Its invariants that a consumer must uphold (or that the constructor upholds).
- Its `#[non_exhaustive]` status if applicable (§4).
- A pointer to the authoritative `3x` doc when the type's contract is ratified there.

Traits additionally document their sealed status, their consumer crate, their method invariants, and any blanket `impl`s they own (§8).

`cargo doc` warning-level is maintained at `missing_docs` for every `semstrait-*` crate. A missing doc comment fails CI.

### 3.3 `pub(crate)` vs `pub(super)` vs `pub`

- `pub` — consumed by another crate or by `semstrait-facade` end users. Listed in the crate's `3x` contract.
- `pub(crate)` — cross-module within the owning crate. Not on the `3x` surface.
- `pub(super)` — visibility within a module tree. Internal convenience only; never appears on a `3x` surface.

### 3.4 Re-export policy

`semstrait-facade` re-exports the minimum set users need to invoke the `parse → … → adapt` pipeline from a single entry. `semstrait-api` re-exports the mid-level surface (callers who want to pick their adapter / catalog but use a bundled pipeline). Other crates do not re-export each other's public symbols by default; a deliberate re-export is a `3x` decision, not a `30` policy.

Every re-export carries the same `#[non_exhaustive]` annotation as the underlying type — re-exports do not relax extensibility.

## 4. Non-Exhaustive-By-Default Policy

I10 binds: every public sum type that models a classification with future extensions is `#[non_exhaustive]`. `30` extends this to the matching discipline on public structs: every public struct whose field set may grow is `#[non_exhaustive]`.

### 4.1 Sum types — MUST be non-exhaustive

The v1 roster, pulled from `00 §9 I10` and extended by the error-enum families ratified here and in `10 §5`:

- **Canonical domain enums.** `DataType`, `Additivity`, `Cardinality`, `JoinType`, `DialectId`, `EngineArtifact`, `EnginePlan`, the shared `Expr<L>` variant set (the authoring carrier — `14 §6`; the former `ExprSource` wrapper is removed per doc 45), `TemporalShape` and its `Scd` subtype enum, the composition-kind tag of `ComposedSemanticInterface`, `DimensionType`, `Grain`, `LiteralValue`, `BinaryOpKind`, `AggregationOp`, `AggregateKind`, `FunctionCategory`, `ParamType`, `ReturnTypeRule`, `Portability`.
- **Diagnostic-surface enums.** `Severity` (§5.2). `SourceId` is opaque (no `#[non_exhaustive]` annotation needed — its variant set is private to its producing crate; see §5.3).
- **Per-stage kind enums.** Every per-stage typed-kind enum: `ParseErrorKind`, `ValidateError`, `CompileError`, `PlanErrorKind`, `OptimizeErrorKind`, `AdaptErrorKind`, `CatalogProviderErrorKind`, `FileSystemErrorKind`, `RepositoryErrorKind`, `SemStraitErrorKind`. New variant addition is the whole point — kind enums grow as new conditions surface.

**Special case: `CanonicalFn`.** `CanonicalFn` is a newtype `struct CanonicalFn(&'static str)` with `pub const` identities (`CanonicalFn::UPPER`, `CanonicalFn::LOWER`, …) per `00 §4.1` and `14a §2`. It is **inherently extensible** — a new adapter-contributed constant does not change the type's shape. No `#[non_exhaustive]` is needed because there is no `enum` to annotate; extensibility is structural. Matching semantics use the `pub const` identities directly.

**Special case: `DataKind`.** Per item-Y, `DataKind` is a **sealed trait** in `semstrait-model`, not an `enum`. `#[non_exhaustive]` does not apply the same way — extensibility is governed by the sealed-trait pattern (§8.1), and the impl set is controlled by the defining workspace.

### 4.2 Structs — MUST be non-exhaustive when MAY-grow

Public structs whose field set may grow in MINOR are annotated `#[non_exhaustive]`:

- `Diagnostic<K>` (§5.1) — `notes`, future cross-reference fields may grow.
- `Location` (§5.3) — location metadata may gain fields (e.g., file-relative line/column).
- `FunctionSpec` (ratified in `14a §3.1` — already `#[non_exhaustive]`).
- `SemanticManifest` and its `Resolved*` family (`ResolvedDataKind`, `ResolvedSource`, `ResolvedColumnMapping`, `ResolvedExprTable`). The `33` doc fixes the exact roster; every public leaf is `#[non_exhaustive]`.
- `SemanticPlan` and `PlanNode` sub-structs — indices and metadata may grow (`35`).
- `SemanticInterface`, `ComposedSemanticInterface` — fields grow as composition semantics sharpen (`16`).
- `Request`, `SessionContext` — session state evolves (`34`).
- `SqlArtifact`, `EngineAdapter` method return types (`36`).

**Retired.** `ContextLine` (was a public struct in earlier drafts) is gone — see §5.3.

### 4.3 Internal-only enums MAY be exhaustive

Crate-private (`pub(crate)` or narrower) enums used on hot paths — e.g. a planner-internal strategy discriminator, a substitution-state flag — are free to be exhaustive for efficient match exhaustiveness checks. Exhaustiveness is a compile-time gift the matcher should not give up when nobody outside the crate sees the type.

A symbol promoted from `pub(crate)` to `pub` gains `#[non_exhaustive]` at the same PR; treating that as a minor refactor is a bug.

### 4.4 Match discipline for non-exhaustive types

Consumers of `#[non_exhaustive]` types must always include a wildcard arm. `30` carries one cross-cutting consumer rule: **a wildcard arm that returns a `Diagnostic<K>` whose kind variant carries enough information to debug the path is always acceptable; a wildcard arm that `panic!`s or `unreachable!()`s in a library is a bug.** End-user applications are free to panic; `semstrait-*` library code never does.

## 14. Cross-References

- `00 §4.1` — `Diagnostic` row (authoritative-doc pointer → `30`).
- `00 §4.2` — verb catalog; `30 §7` ratifies the API-boundary return shapes.
- `00 §9` — I7 (strict DAG), I10 (non-exhaustive), I11 (gated I/O), I12 (first-class diagnostics; cross-doc reconciliation queued under foundations — `STATUS.md §2`).
- `10 §3` — per-stage contracts consumed by `30 §7`.
- `10 §5` — internal-error model; `30 §5` is the public-boundary refinement (the prior code-allocation framework retired in this doc).
- `11 §8` — Constraint framework; `ConstraintViolation` is now a `PlanErrorKind` variant rather than a numeric code.
- `14 §7` — expression-error catalog (cross-doc reconciliation queued under foundations — these become variants on `ParseErrorKind` / `ValidateError` / `CompileError`).
- `14a §3.1`, `§7`, `§8` — `FunctionSpec` `#[non_exhaustive]`, `RegistryExtension`, function-resolution kinds (queued for foundations alignment).
- `13 §6` — type-related Precondition IDs (`TG-*`) map to `ValidateError` / `CompileError` variants.
- `18 §11` — SR-E rules become `ValidateError` variants (queued for foundations alignment).
- `31`–`39` — per-crate refinements of every policy in this doc.
- Deprecation register — deprecation lifecycle tracking (§12.2 of `30-breaking-change-governance`).
- Migration notes — MAJOR migration entries (§11.2 of `30-breaking-change-governance`).
- `questions/open/30_questions.md` — parked reconciliation items.

## Testing Strategy

- cargo-semver-checks lint confirms no unintentional breaks before publish (§2.1 MAJOR triggers)
- Non-exhaustive roster completeness validated against codebase types (§4.2 struct/enum rosters)
- Match discipline contract upheld (§4.4 wildcard arms return Diagnostic<K>, never panic in library)
- Stability table census matches current crate list and maturity markers (§13)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
