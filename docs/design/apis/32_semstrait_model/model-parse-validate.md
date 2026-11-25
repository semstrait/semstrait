---
id: 32-model-parse-validate
status: ratified
version: 2.4
impl: implemented
---

# Parse and Validate Stage APIs

The model crate hosts two accumulating stages per `30 §7.1`: `parse` (YAML → `SemanticModelBuilder`) and `validate` (structural-precondition pass over a parsed `SemanticModel`). Each owns its own kind enum (`ParseErrorKind`, `ValidateError`), each implements `Diagnose` per `30 §5.4`, and each uses the accumulating return shape per `30 §7.2`. The fluent loader (model-loader leaf) composes the two behind one async-or-sync entry point.

## 1. Parse/build materialization pipeline (internal; shape-preserving)

`semstrait-model` separates authoring decode from canonical model materialization. `SemanticModel` is the canonical type at every stage that has an `id`; the only intermediate is an off-schema `pub(crate)` decode mirror that exists solely to hold entities while their ids are still `None` (it carries no `JsonSchema` and is never on the public surface):

1. `${VAR}` substitution on raw YAML text (see model-crate-boundaries leaf §8).
2. A single `serde` decode pass. The schema + public serde root is `SemanticModel` itself; the decode target is a `pub(crate)` `Vec`-field mirror under the `semantic_model:` envelope (SR-1), holding each named entity with its `id: Option<EntityId>` exactly as authored (`None` when omitted). This mirror is the one permitted intermediate — internal, never on the schema. There is no public `YamlRoot`.
3. Missing-id resolution per `IdentityProfile` (§1.1), run in `parse` over the decoded `Vec`s **and the model root** (`32 §1.4a`) **before** indexing — because only `parse` holds the profile (serde cannot mint or resolve, nor emit `MissingEntityId`): convenience mints a fresh UUIDv7 into every entity `None` and **derives** the root `id` (UUIDv5 over `name`, never time-based); strict records a `MissingEntityId` diagnostic for either. After this pass every `id` is `Some`, root included.
4. Index the now-`Some` entities into the canonical `SemanticModel` id-keyed maps via each collection's `EntityBTreeMap::try_from(Vec<T>)` — the single index site, which unwraps the `id`, keys by it, and rejects duplicate ids (SR-12). No separate accumulation store, no handle table.
5. `.build()` runs validate-stage checks, including the per-layer duplicate scans below.

There is no separate builder accumulation layout, `EntityHandle`, or `SemanticModelStorage`. The `EntityBTreeMap`-wrapped id-keyed maps **are** the storage: because the map key is the entity's own (post-mint) `id` (unique by construction — authored ids are checked, generated ids are fresh UUIDv7), every authored entity is retained without insert-time collision, and duplicate-by-name is detected by scanning each map at `.build()`:

- **Global layer** — data-kind name uniqueness (`SR-3`) across the four top-level data-kind maps; `id` global uniqueness (`SR-12`) across all named entities. The model-root `id` (`32 §1.4a`) is not a collection member and is excluded from the SR-12 scan.
- **Per-carrier layer** — shared-pool name uniqueness within `dimensions` / `measures` / `metrics`.
- **Parent-scoped layer** — nested-child name uniqueness within each complex body's child maps (`26 §5`).

```rust
pub type EntityId = String; // canonical UUID text (36 bytes, lowercase, hyphenated)
```

Rules:

- `EntityId` is a boundary string. Canonical form is 36-byte lowercase hyphenated UUID text, non-nil, version nibble unvalidated (`SR-11`; contract at `31 §2.1`).
- The authoring field is `id: Option<EntityId>`: `None` = authored-absent (no `""` sentinel). Mint (step 3) resolves every `None` to `Some` before indexing, so the map key — the unwrapped inner `EntityId` — is always present.
- IDs may be authored, minted (entities, v7), or derived (root, v5); all three MUST satisfy the one SR-11 shape rule, which none of them distinguishes.
- Dedup of *names* is by explicit scan (`SR-3` / `SR-E-3`); the map key is the `id`, so name collisions are reported, not silently overwritten.
- Public/nested structural envelopes and nesting rules (`26`) remain unchanged.
- Identity is a single `EntityId` lane across model and manifest (`33`); the model does not allocate any separate compile-id newtypes (the former `DataKindId` / `RelationshipId` / `SemanticsId` / `BindingId` are retired). This section does not move that boundary.

### 1.1 Identity profiles

`semstrait-model` defines two profiles for missing named-entity IDs:

```rust
#[non_exhaustive]
pub enum IdentityProfile {
    /// Missing `id` is generated as UUIDv7.
    ConvenienceGenerateMissing,
    /// Every named entity must provide `id`; missing is an error.
    StrictRequireProvided,
}
```

Default profile is `ConvenienceGenerateMissing` for `parse(&str)` and `SemanticModel::loader()`.

- **Strict deterministic profile** (`StrictRequireProvided`) requires authored `id` for every named entity **and for the model root** (`32 §1.4a`) — it does not accept root derivation.
- **Convenience profile** (`ConvenienceGenerateMissing`) generates UUIDv7 for missing named-entity `id`, and **derives** a missing root `id` from `name` as UUIDv5 (`32 §1.4a`) — the root is deterministic under both profiles.
- Cross-run identity stability from source alone requires `StrictRequireProvided` or persisted id reuse from manifest (`33 §4.3.1`). This caveat covers **entity** ids only: the model-root `id` is cross-run stable under both profiles by construction (`32 §1.4a`).

## 2. `parse` signature

```rust
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};

/// YAML `&str` → `SemanticModelBuilder`. Pure and synchronous. Accumulating
/// stage per `30 §7.1`: every independent parse error is collected; warnings
/// found during the pass surface on the success arm.
///
/// Returns a builder rather than a finalised model so single-source,
/// code-built, and cross-source accumulations can share one materialise
/// path. Callers chain `.build()` (see model-builder-api leaf) to obtain the canonical
/// `SemanticModel` together with any validate-stage diagnostics.
pub fn parse(
    input: &str,
) -> Result<SemanticModelBuilder, Diagnostics<ParseErrorKind>>;

pub fn parse_with_profile(
    input: &str,
    profile: IdentityProfile,
) -> Result<SemanticModelBuilder, Diagnostics<ParseErrorKind>>;
```

`parse` is shorthand for `parse_with_profile(input, IdentityProfile::ConvenienceGenerateMissing)`.

Per D-10, dup-name diagnostics (SR-3 / SR-E-3) are emitted at `.build()` so single-source, multi-source, and code-built paths share one output contract. Implementations may track names incrementally during lowering into the id-keyed maps (§1), but they must not surface final `Duplicate*` diagnostics before `.build()`.

## 3. `ParseErrorKind` roster

```rust
use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_common::Location;

#[non_exhaustive]
pub enum ParseErrorKind {
    // — YAML surface —
    YamlSyntax                    { message: String },
    UnsetEnvVar                   { var: String },
    UnknownField                  { field: String, parent: String },

    // — Structural rules (SR-*) —
    InvalidIdentifier             { raw: String, reason: String },           // SR-8
    InvalidEntityId               { raw: String, reason: String },           // SR-11
    InvalidDecimalParameters      { precision: u8, scale: i8 },              // 13 TG-V1

    // — Authoring expression surface (`14 §6.3`) —
    //   A top-level COMPOUND expression string (whitespace/operator chars) is the
    //   reserved inline-DSL form, deferred in v1; rejected at parse. A BARE string
    //   is NOT an error — it tokenizes to a default leaf (`14 §6.1`/`§6.5`). The
    //   variant was formerly `InlineDslNotImplemented` (renamed per doc 45, G-NAME).
    InlineExprUnsupported         { expr: String },

    // — Entity-level invariants (SR-E-*) — fired at parse —
    RelationshipMissingCardinality { relationship: String },                 // SR-E-4
    MeasureMissingAgg             { carrier: String, name: String },         // SR-E-9
    SemanticsMissingDataType      { carrier: String, name: String },         // SR-E-10
}

impl Diagnose for ParseErrorKind {
    fn message(&self) -> String { /* per-variant human text */ }
    fn severity_default(&self) -> Severity { Severity::Error }
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}
```

SR-1 / SR-2 / SR-4 (root-key, nested-data-kind, self-nesting) collapse into `UnknownField` — each is enforced by the absence of the offending field at the type level plus serde's `deny_unknown_fields`, so the surfaced kind is uniform.

`Duplicate*` variants migrated to `ValidateError` (§4) per D-10 so cross-source accumulations share the single-source enforcement path.

**Per-variant location.** The primary source span of an error lives on the wrapping `Diagnostic<ParseErrorKind>` (`30 §5.1`'s `location: Option<Location>` field), not on the variant. Variants carry payload data only.

**No `code()` accessor.** Identification of a parse error is by **variant identity** (`matches!(diag.kind, ParseErrorKind::InvalidIdentifier { .. })`), not by a string code. Renaming a variant is a MAJOR change per `30 §2.1`; adding a variant inside `#[non_exhaustive]` is MINOR per `30 §2.2`.

**Accumulation.** `parse` collects every recoverable error into the returned `Diagnostics<ParseErrorKind>` vector before deciding success vs failure. Catastrophic syntactic errors (the YAML parser cannot continue past byte N) short-circuit with a single `YamlSyntax` entry plus any warnings emitted before that point.

### 3.1 What `parse` does NOT do

- **No name resolution.** `SemanticsName` references inside authoring expressions (`Expr<L>` leaves) stay as opaque identifiers; resolution happens at `compile` (`11 §7`).
- **No type inference.** A derived Dimension / Measure / Metric with no `data_type:` stays `None` at parse; `compile` infers per `13 §6`.
- **No composition materialization.** `ComposedSemanticInterface` is a compile-time output.
- **No catalog I/O.** Catalog references are parsed as `CatalogRef` values and left unresolved; `catalogs.yaml` is a separate file loaded by the caller.
- **No multi-file loading.** `parse` takes one `&str`.

## 4. `validate` signature

```rust
/// Structural-precondition pass over a parsed `SemanticModel`. Pure and
/// synchronous. Accumulating stage per `30 §7.1`'s `validate` special
/// case: produces no value on success — the success arm carries only the
/// warnings vector (no tuple wrapper).
pub fn validate(
    model: &SemanticModel,
) -> Result<Diagnostics<ValidateError>, Diagnostics<ValidateError>>;
```

`validate` runs the SR-* rules whose enforcement column reads "Enforced at `validate`" (SR-10, SR-12) plus the entity-level `SR-E-`* invariants from `18 §11`. It does not transform the model; it is a pure precondition checker. Call it as a separate step before `compile` (`33`) when per-stage error routing matters; the fluent loader (model-loader leaf) chains `parse` and `validate` for callers that don't need that granularity.

## 5. `ValidateError` roster

```rust
#[non_exhaustive]
pub enum ValidateError {
    // — Composition shape (SR-10) —
    ComplexDataKindInsufficientChildren { parent: String, child_count: usize },

    // — Empty model —
    EmptyModel,

    // — Cross-source / single-file dup detection (SR-3, D-10) —
    DuplicateDataKindName            { name: String, occurrences: Vec<Location> },
    DuplicateSharedSemanticsName     { carrier: String, name: String, occurrences: Vec<Location> },
    DuplicateEntityId                { id: String, occurrences: Vec<Location> },   // SR-12
    MissingEntityId                  { path: String },                              // strict profile only

    // — Entity-level invariants (SR-E-*) per `18 §11` —
    OrphanSharedSemantics            { carrier: String, name: String },
    TemporalGrainOnComplex           { data_kind: String },
    GrainsetChildMissingGrain        { grainset: String, child: String },
    /* … additional SR-E-* variants per `18 §11` … */

    // — Shadowing warning (`18 §1.5`) —
    SemanticsShadowRootPool          { carrier: String, name: String },

    // — Tree-construction-boundary failure (D.ii kind-nesting per `30 §7.4`).
    //   Surfaces `Tree::with_new_children` / `Rewriter::f_*` failures from the
    //   tree machinery in `semstrait-common` during the `Expr<L>` field-adapter
    //   serde round-trip (`14 §6.1`). `Expr<L>` is model-owned (item Y / V6);
    //   the tree-machinery `ValidateError` is common-owned (item Y / V5), so
    //   this wraps `common::ValidateError` directly — the prior
    //   `ir::ValidateError` hop is gone (V1).
    Tree(common::ValidateError),
}

impl Diagnose for ValidateError {
    fn message(&self) -> String { /* per-variant human text */ }
    fn severity_default(&self) -> Severity { Severity::Error }
    fn cause(&self) -> Option<&(dyn std::error::Error + 'static)> { None }
}

impl From<common::ValidateError> for ValidateError {
    fn from(e: common::ValidateError) -> Self { Self::Tree(e) }
}
```

The `Duplicate*` variants migrated from `ParseErrorKind` per D-10: dedup runs uniformly at `.build()` as a per-layer name scan over the id-keyed maps (§1), so single-source, code-built, and cross-source accumulations all surface the same diagnostic with every occurrence's `Location`.

`DuplicateEntityId` is emitted by the same `.build()` pass over resolved named-entity IDs. `MissingEntityId` is emitted only under `IdentityProfile::StrictRequireProvided`.

The `SemanticsShadowRootPool` variant carries `Severity::Warning` (the only warning-class variant in v1) and rides through on the `Ok` arm.

The `Tree(common::ValidateError)` variant covers the narrow class of construction-boundary failures raised by the tree-machinery contract now homed in `semstrait-common` (item Y / V5; `14 §9.1`): when an expression field's adapter deserializes a YAML tree, the serde shape *is* the model-owned `Expr<L>` (item Y / V6) so any `Tree::with_new_children` violation surfaces as a `common::ValidateError` value that the model crate wraps via D.ii nesting (`30 §7.4`). `model → common` is the only hop — the prior `ir::ValidateError` intermediary is gone (V1).

The full `SR-E-*` variant roster is enumerated alongside the entity-level invariants in `18 §11` (the canonical home of SR-E-*); variants here mirror those rules 1-to-1 and land MINOR per `30 §2.2` as new SR-E-* numbers are appended.

## Testing Strategy

- **ParseErrorKind roster**: every variant is exercised by negative-parse fixtures — each malformed YAML input (unset env var, unknown field, invalid identifier, invalid entity id, invalid decimal params, inline DSL attempt, missing cardinality/agg/data_type) triggers its corresponding variant.
- **ValidateError roster**: every variant is exercised by validate-stage fixtures — each structural violation (insufficient children, empty model, duplicate names/ids, orphan semantics, temporal grain misplacement, missing grain on Grainset child, shadowing) triggers its corresponding variant.
- **Identity profiles**: tests verify that `ConvenienceGenerateMissing` mints UUIDv7 for missing entity ids and derives the root id, and that `StrictRequireProvided` emits `MissingEntityId` diagnostics for both. Profile tests assert the *derivation* (which version each site produces); SR-11 tests assert the *validation* (which shapes are accepted) — the two are deliberately separate, so swapping a derivation touches only the former.
- **Accumulation contract**: parse collects multiple independent errors into the `Diagnostics` vector before failing; warnings ride through on the success arm.
- **D-10 dup-name contract**: duplicate-by-name diagnostics (`SR-3`, `SR-E-3`) fire at `.build()` uniformly across single-source, multi-source, and code-built paths — verified by cross-source and builder-based duplicate-name tests.

Shared harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
