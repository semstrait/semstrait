# semstrait-common

Shared primitives for the semstrait toolchain, with no semantic opinion.

Owns the diagnostic envelope, the one entity-identity mint, the stage-agnostic
tree-traversal traits, the shape-only constraint DSL, the `Dialect` name, and —
behind the default-on `io` feature — the byte-blob transport and the
persistence seam.

This crate sits at the **bottom of the crate dependency graph**: it names no
other `semstrait-*` crate, and every other workspace crate —
`semstrait-sql`, `semstrait-model`, `semstrait-ir`, `semstrait-manifest`,
`semstrait-catalog`, `semstrait-planner`, `semstrait-adapter`,
`semstrait-adapter-datafusion`, `semstrait-api` and the `semstrait` facade —
names it. The graph is strict and acyclic, enforced by Cargo's
own dependency edges rather than by a document.

It is deliberately **expression-free and plan-free**. No expression tree, no
plan node, no canonical data type, no canonical function, and no manifest type
appears here — the authoring vocabulary lives in `semstrait-model`, the
canonical expression and data-type vocabulary (`Expr`, `DataType`) in
`semstrait-sql`, the plan vocabulary in `semstrait-ir`, and the compiled
artifact in `semstrait-manifest`. What lands here is the stage-agnostic
machinery both sides need, sunk to the root so neither has to depend on the
other (ADR 0001): the `Tree` / `Visitor` / `Rewriter` family and the
primitive-only `ValidateError` are here for exactly that reason. `Granularity`
is here for a related reason: one roster for a temporal Dimension's
granularities, a temporal shape's granularity, the request's granularity and
the planner's coarseness checks, sunk to the root so the authoring, the
compiled and the planning side all read the same roster.

---

## Module map

| Module | Feature | What it owns |
|--------|---------|--------------|
| `diagnostic` | *(always)* | The `Diagnostic<K>` envelope, the `Diagnose` trait every per-stage kind enum implements, `Severity`, and the source-position triple `Location` / `SourceId` / `Span`. |
| `dialect` | *(always)* | `Dialect`, the name of a dialect's vocabulary, and `Release` / `DialectRelease`, a dialect read at a release — see the section below. |
| `identity` | *(always)* | The `EntityId` lane, the UUIDv5 mint (`derive_uuid_v5`, `mint_entity_id`, `mint_named_entity_id`), the per-class namespace roster (`EntityNamespace`, `entity_namespace`), and the two frozen root namespace constants. |
| `tree` | *(always)* | `Tree` / `Visitor` / `Rewriter` — the traversal contract every semstrait tree implements — plus `ValidateError`, the construction-boundary failure they raise. |
| `constraints` | *(always)* | The shape-only constraint DSL: `MeasureConstraints`, `DimensionConstraints`, `AggregationConstraints`. |
| `glob` | *(always)* | `GlobPattern`, a newtype over a pattern string. No matching logic, and no consumer today — see the section below. |
| `io` | `io` *(default)* | The `FileSystem` byte transport with its back-ends, and the `Repository` / `Persistable` / `Fingerprintable` persistence seam. |
| `names` | `serde` | The `str_newtype!` macro and the two name newtypes it builds: `SemanticsName`, `DataKindName`. Each derives `JsonSchema` under the `schemars` feature. |
| `granularity` | `serde` | `Granularity`, the roster Minute, Hour, Day, Week, Month, Quarter, Year, with `coarseness`, `serves` and `Display`. |

---

## diagnostic

`Diagnostic<K>` is the one envelope; `K` is a per-stage typed-kind enum that
implements `Diagnose` (`message()`, `severity_default()`, an optional `cause()`).
Diagnostic identity is *variant identity* — there is no error-code registry, and
`String` / `anyhow` / `Box<dyn Error>` are banned on public surfaces (ADR 0004).
This crate exposes the envelope shape only: construction lives at the consumer,
which typically wraps it in its own `fn diag(kind: K) -> Diagnostic<K>` helper.

Control flow through the envelope:

1. A stage builds a bare kind at the failure site (`ValidateError`,
   `FileSystemErrorKind`, a crate's own roster).
2. `Diagnostic::new(kind)` seeds `severity` from `Diagnose::severity_default`;
   the chainable `with_severity` / `with_location` / `with_note` attach the rest.
3. `map_kind` re-wraps a narrow kind into a wider one when a stage lifts an
   upstream diagnostic into its own roster, preserving severity, location and
   notes.
4. `split_by_severity` partitions a `Diagnostics<K>` batch into
   `(errors, warnings)`, each preserving input order. A stage's error arm fires
   when `errors` is non-empty, and warnings survive both arms (ADR 0005) — the
   partition never sorts or de-duplicates.

`Location` is `SourceId` + `Span`. `Span` is a half-open byte range;
`Span::unknown()` is the zero-width `0..0` for a diagnostic with a known
document but no resolved range. `SourceId::named` mints a labelled id (a file
path, an inline-string tag) and `SourceId::unknown()` is the no-origin sentinel.
`SourceId`'s inner variant set is private, so — unlike most public types here —
it is not `#[non_exhaustive]`: its whole surface is `named` / `unknown` /
`as_str` plus the derived `Eq` / `Hash` / `Display`.

## dialect

`Dialect` names a dialect: one engine's or one format's own vocabulary of
functions, operation nodes and data types, as `semstrait-sql` declares it. Its
variants are `DataFusion`, `Substrait` and `Iceberg`, and it is
`#[non_exhaustive]`. It carries no adapter identity and has no value for the
canonical vocabulary (ADR 0091, rule 6). Under the `serde` feature, serde
writes its name in lower case, `datafusion`, as a manifest's `dialect` names it
(ADR 0088, rule 8). `semstrait-sql`, `semstrait-catalog`, `semstrait-manifest`,
`semstrait-adapter-datafusion` and `semstrait-api` read it; it lives here by
ADR 0091, rule 6.

A dialect is read at a release (ADR 0091, rule 7). `Release` is a
newtype over a release's major, `pub struct Release(pub u32)`; its two
associated consts, `Release::DATAFUSION_REFERENCE` (`Release(55)`) and
`Release::DATAFUSION_OLDEST` (`Release(53)`), are DataFusion's. `Dialect::reference_release`
and `Dialect::oldest_release` give those consts for `DataFusion` and `None` for
`Substrait` and `Iceberg`; both are `const fn`. `DialectRelease { dialect,
release }` pairs a dialect with an optional release — `From<Dialect>` fills
`release` from `reference_release()` — and its `FromStr` / `Display` read and
write `name` or `name=release` (`datafusion`, `datafusion=53`), the same
lower-case names serde writes. `FromStr` refuses an unknown dialect name, a
release that is not a major, a release on a dialect that has none, and a
release below the oldest, each its own `DialectReleaseError` variant.

## identity

An **entity id** is the technical key of a named entity: one untyped lane of
canonical UUID text, durable across compiles, and the handle every stored
artifact addresses. The entity's **name** is its natural key, and the id is
derived from it — an entity that omits `id:` is minted
`UUIDv5(namespace_for_its_class, scope_qualified_name)` (ADR 0051). One lane, no
per-class newtypes (ADR 0010).

The **lane** and the **mint** are homed here for two different reasons, and only
the first involves `semstrait-ir`. `EntityId` is reachable from all three stages
that address an entity — including `semstrait-ir`'s semantic graph, which is
keyed by it — and `semstrait-ir` depends on no model crate (ADR 0094, rule 9),
so the model crate could not host the type without adding that edge. The mint has no such
constraint (`semstrait-ir` never calls it); it lives here because **both stages
that do call it already depend on this crate**, so one mechanism serves them
instead of two (ADR 0051). `semstrait-model` mints authored entities and
`semstrait-manifest` mints compile-synthesised ones; both call the same
functions, and neither reimplements the derivation.

- `mint_named_entity_id(kind, scope_qualified_name)` — the ADR 0051 rule spelled
  once. The caller owns building the scope-qualified path, because only the
  caller knows the scope.
- `mint_entity_id(namespace, seed)` — for a seed that is not a plain name (a
  fingerprint, or a canonical multi-field encoding the caller owns).
- `derive_uuid_v5(namespace, seed)` — the raw UUIDv5 derivation both of the
  above render.
- `entity_namespace(kind)` — the per-class namespace, `UUIDv5` of
  `ENTITY_ROOT_NAMESPACE` over the class label.

Two frozen constants seed everything: `ENTITY_ROOT_NAMESPACE` (an invented
literal every per-class namespace descends from) and `MODEL_ROOT_NAMESPACE` (the
model root's own, a sibling root rather than a child). Both are **frozen for all
time** — changing either re-mints every id in scope and orphans whatever
addressed it. Each namespace is pinned by a test whose expected value was
computed by a separate UUIDv5 implementation, so a rewritten constant fails
rather than silently re-keying the world. A `EntityNamespace` variant added
without a roster entry fails the same pinning test.

## tree

Every semstrait tree — `Expr<L, A>` in `semstrait-sql`, `PlanNode` in
`semstrait-ir` — implements `Tree`, so one generic algorithm walks both shapes:

- `Tree` — the structural contract. Implementors supply `children()` (borrows,
  deterministic order) and `with_new_children()` (rebuild, or reject with
  `ValidateError`). The default-provided `apply` composes those into a pre-order
  `Visitor` walk; `pre_order` into an iterator over the nodes in pre-order, with no
  recursion; `transform` into a bottom-up rewrite.
- `Visitor` — read-only walk with `f_down` / `f_up` hooks, returning
  `ControlFlow` so a walk can short-circuit and yield an output.
- `Rewriter` — the owned-node counterpart, for a caller that needs to inspect a
  node both before and after its children are rewritten.
- `ValidateError` — the construction-boundary failure the above raise:
  arity mismatch (`ChildCountMismatch`), an empty `InList` / `Case`
  (`EmptyInList`, `EmptyCase`), an empty name (`EmptyName`), and a call whose
  new operands no longer fit its function's declaration, or hold a call that it
  cannot hold (`CallRefused`, carrying the refusal's message), and a plan node
  that its constructor refuses over its new inputs (`NodeRefused`, likewise).

`ValidateError` is primitive-only — its variants carry `usize` / `String` /
`&'static str` and nothing from a downstream vocabulary — which is why it lives
here rather than in `semstrait-ir`, and each crate that raises or nests it imports
it from here (ADR 0090, rule 6). Downstream stages nest it inside their own kind:
`semstrait_model::ParseError` carries `InvalidExpression(ValidateError)`.

## constraints

Shape-only policy carried on a measure: `MeasureConstraints` holds an optional
`DimensionConstraints` (`one_of` / `none_of` / `all`, AND-combined) and an
optional `AggregationConstraints` (`allowed` / `prohibited` tokens). All three
are plain name and token lists — a constraint with an expression body would need
the canonical expression vocabulary and therefore belongs in `semstrait-ir`,
never here. No crate reads them today.

## glob

`GlobPattern` is a newtype over a pattern string with `new` / `as_str` /
`Display` / `From`. It carries **no matching logic**, and no code in the
workspace consumes it: the one live glob matcher in the tree is
`semstrait-manifest`'s own, which does not use this type. Whether the newtype is
deleted or given a real home is an open decision, so treat this
section as a description of what exists, not of an intended integration.

## io *(feature `io`, on by default)*

Two layers, and the split is the point: `FileSystem` moves bytes, `Repository`
stores values. `common` names no DTO of its own, so a state crate implements
`Persistable` for its own type and gets `FileSystemRepository<T>` for free — the
dependency graph enforces that direction, since this crate cannot name `model`
or `manifest`.

**Transport.** `FileSystem` is object-safe (`Send + Sync + Debug`,
`#[async_trait]`, so `Arc<dyn FileSystem>` is shareable) and strictly
format-agnostic: `list` / `read` / `write` / `exists` over `Path` and `Bytes`,
no streaming. `list` is a literal `starts_with` over full keys, not
segment-aware, and results are not sorted by the trait — a caller needing
determinism sorts client-side. Methods return a bare `FileSystemErrorKind`;
consumers wrap it into their own kind via `From` at the call site. `Path` is
URI-shaped, with `Path::parse` gating the scheme vocabulary (`file`, `s3`,
`s3a`; a one-letter scheme is a Windows drive letter, not a scheme) and
`Path::new` skipping validation for already-shaped input.

Back-ends: `LocalFileSystem` (always, under `io`), `S3FileSystem` (behind
`io-aws`), and `InMemoryFileSystem` for tests and deterministic replay. The two
`object_store`-backed ones wrap that crate's stores into compliance with the
`FileSystem` contract, absorbing the disagreement over `list` semantics rather
than leaking it.

**Persistence.** `Persistable` is the DTO's codec tag — `entity_id`,
`extension`, `to_bytes`, `from_bytes`, plus an `integrity_check` hook that
refuses a value whose own invariants are broken. It performs no I/O. `to_bytes`
gives a `bytes::Bytes`, so a crate that implements it depends on `bytes` directly.
Its fallible methods return a bare `Cow<'static, str>` reason rather than a kind
enum, which the seam folds into `EncodingMismatch` / `IntegrityCheckFailed`: the DTO's own
failure text is a flattened cause under ADR 0004, not a typed surface of its own,
so the implementor owes no error roster.
`Repository<T>` is the object-safe port (`save` / `load` / `exists`) and
`FileSystemRepository<T>` the sole v1 adapter, keying `base/{id}.{extension}`
over any `FileSystem`. `RepositoryErrorKind` folds every back-end failure
through `IoErrorKind`, so a consumer depends on one I/O vocabulary rather than
each back-end's.

Save and load, end to end: `save` runs `integrity_check` first — before the
infallible `entity_id()` is called on the value — then encodes via `to_bytes`
and writes at `{base}/{id}.{extension}`; `load` reads that key, decodes via
`from_bytes`, re-runs `integrity_check`, and checks that the decoded value's own
id agrees with the requested one. A mismatch is `IntegrityCheckFailed`, not a
silent wrong answer. Keys fold ASCII case and refuse an id carrying a path
separator, so the id→key mapping stays injective; deeper validation is the
consumer's.

**`Fingerprintable`** sits beside `Persistable`, deliberately not on it. It
names a *preimage* and never a format, and its provided `fingerprint()` is
SHA-256 over that preimage — the **model fingerprint**, the content digest that
answers "did this value change". It is separate for two independent reasons:
`Persistable::integrity_check` refuses the id-less value a fingerprint preimage
projects, and `to_bytes` is a storage codec whose spelling a digest must not
inherit. Comparison is *not* on the trait: whether two fingerprints differ, and
what to do about it, is the caller's decision (ADR 0041). Today
`semstrait-manifest`'s `SemanticManifest` and `semstrait-model`'s `SemanticModel`
both implement `Persistable`; `SemanticModel` is the one `Fingerprintable`.

---

## Key Types

| Type | Description |
|------|-------------|
| `Diagnostic<K>` / `Diagnostics<K>` | The diagnostic envelope and its transparent `Vec` alias. |
| `Diagnose` | Open trait a per-stage kind enum implements to slot into the envelope. |
| `Severity` | `Error` or `Warning`. Two variants only. |
| `Location` / `SourceId` / `Span` | Source position: document id plus half-open byte range. |
| `split_by_severity` | Partition a batch into `(errors, warnings)`, order-preserving. |
| `Dialect` | The name of a dialect's vocabulary: `DataFusion`, `Substrait` or `Iceberg`. |
| `Release` | A release's major, `pub struct Release(pub u32)`; `DATAFUSION_REFERENCE` / `DATAFUSION_OLDEST` consts. |
| `DialectRelease` | A dialect paired with an optional release; parsed from `name` or `name=release`. |
| `DialectReleaseError` | `DialectRelease`'s `FromStr` failure: unknown dialect, non-major release, no releases, below oldest. |
| `EntityId` | The identity lane: canonical UUID text, a transparent `String` alias. |
| `EntityNamespace` / `entity_namespace` | The entity class an id is minted under, and its namespace UUID. |
| `derive_uuid_v5` / `mint_entity_id` / `mint_named_entity_id` | The UUIDv5 mint, its `EntityId` rendering, and the name-seeded rule. |
| `ENTITY_ROOT_NAMESPACE` / `MODEL_ROOT_NAMESPACE` | The two frozen root namespaces. |
| `Tree` / `Visitor` / `Rewriter` | Traversal contract, read-only walk, owned rewrite. |
| `ValidateError` | Construction-boundary failure raised by `with_new_children` and rewrite callbacks. |
| `MeasureConstraints` / `DimensionConstraints` / `AggregationConstraints` | Shape-only constraint DSL. |
| `GlobPattern` | Pattern-string newtype. No matching logic; no consumer. |
| `FileSystem` / `Path` / `FileEntry` | Byte transport trait, URI-shaped path, one listing entry. |
| `LocalFileSystem` / `S3FileSystem` / `InMemoryFileSystem` | The back-end roster (`io`, `io-aws`, and tests respectively). |
| `Persistable` / `Fingerprintable` | The DTO's storage codec tag, and its codec-free digest sibling. |
| `Repository` / `FileSystemRepository` | The persistence port and its sole v1 adapter. |
| `FileSystemErrorKind` / `IoErrorKind` / `RepositoryErrorKind` | The three typed-kind error enums of the `io` surface. |

Public sum types and public structs whose fields may grow are
`#[non_exhaustive]` (ADR 0007), so consumers construct through the named
constructors (`Diagnostic::new`, `Location::new`, `FileEntry::new`) and write a
wildcard arm when matching. Variant and field additions are MINOR *because* of
that. `SourceId` deliberately carries no such attribute and says so: its inner
variant set is private, so it is already closed by construction.

---

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `io` | ✅ | The `io` module: `FileSystem`, its back-ends, and the `Repository` seam. Pulls `tokio`, `bytes`, `object_store`, `async-trait`, `futures-util`, `sha2`. |
| `io-aws` | | Adds `S3FileSystem` (implies `io`, enables `object_store/aws`). |
| `serde` | | `Serialize` / `Deserialize` on the constraint types, on `Dialect`, on the `names` module (`str_newtype!` and the two name types it builds: `SemanticsName`, `DataKindName`), and on `Granularity`. |
| `schemars` | | Pulls the optional `schemars` dependency and implies `serde`. Derives `JsonSchema` on `SemanticsName`, `DataKindName` and `Granularity`. |

## Dependencies

No `semstrait-*` crate — that is what "bottom of the graph" means. Externally:

- `thiserror` -- derived `Display` on `ValidateError` and `DialectReleaseError`
- `uuid` -- the UUIDv5 mint
- `serde` (optional) -- constraint-type (de)serialization
- `schemars` (optional) -- see the features table
- `tokio` / `bytes` / `async-trait` / `futures-util` (optional, behind `io`) -- async transport plumbing
- `object_store` (optional, behind `io`) -- the local and S3 back-ends
- `sha2` (optional, behind `io`) -- the `Fingerprintable` digest, which rides the same feature as `Persistable` beside it
