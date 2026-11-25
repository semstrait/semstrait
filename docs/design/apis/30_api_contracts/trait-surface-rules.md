---
id: 30-trait-surface-rules
status: ratified
version: 2.1
impl: implemented
---

# Trait Surface Rules

This leaf owns the public trait surface rules — sealed vs open, cross-cutting trait roster, trait-method return shape, and async trait methods.

## 8. Public Trait Surface Rules

Every public trait in the `semstrait-*` workspace documents:

1. **Its consumer crate** — the crate that calls the trait's methods.
2. **Its implementation crate(s)** — the crate(s) that provide `impl`s.
3. **Its method invariants** — what callers guarantee on input, what implementers guarantee on output.
4. **Its sealed/open status** — whether third-party impls are permitted.
5. **Its blanket impls** — which generic `impl` blocks ship with the trait.

### 8.1 Sealed vs open

A **sealed** public trait restricts implementations to the defining workspace. Sealing uses a private super-trait pattern:

```rust
mod sealed {
    pub trait Sealed {}
}

pub trait FooExt: sealed::Sealed {
    fn foo(&self) -> Result<Bar, Diagnostic>;
}

impl sealed::Sealed for LocalType {}
impl FooExt for LocalType { ... }
```

Sealed traits are used where semstrait must control the impl set for correctness (e.g. where invariants cross the trait boundary and a buggy external impl could violate I4 / I5 / I8).

An **open** public trait is implementable by any crate. Used where semstrait benefits from third-party extension and can tolerate misbehaving impls as caller errors.

### 8.2 Cross-cutting trait roster

| Trait | Consumer | Impl(s) | Sealed? | `3x` doc |
|---|---|---|---|---|
| `CatalogProvider` | `semstrait-manifest` (compile), `semstrait-api` (drift check) | `semstrait-catalog-*` crates | open — third-party catalog adapters are a supported extension | `37` |
| `FileSystem` | `semstrait-manifest` (compile, glob expansion), `semstrait-catalog-*` (source reads) | local-fs impl in `semstrait-catalog`; object-store impls in per-provider crates | open | `37` |
| `Repository` | callers at the `semstrait-api` / `semstrait-facade` layer | in-memory, filesystem-backed (bundled); third-party may add | open | `33` |
| `EngineAdapter` | `semstrait-planner` (injection hooks), `semstrait-api` (terminal `adapt`) | `semstrait-adapter-*` crates | open — new engine support is a primary extension axis | `36` |
| `RegistryExtension` | `function_registry()` initializer in `semstrait-ir::functions` | `semstrait-adapter-*` crates | open (see `questions/deferred/30_questions.md` Q-API-009) | `36` (via `14a §7`) |
| `Diagnose` | `Display` / `Error` blanket impls on `Diagnostic<K>`; the `?` operator | each per-stage kind enum | open — enables third-party kinds to participate in the diagnostic surface | `31` |

### 8.3 Trait-method return shape

Every fallible public trait method returns one of the per-stage shapes from §7.2 — `Result<(T, Diagnostics<K>), (Diagnostic<K>, Diagnostics<K>)>` (fail-fast) or `Result<(T, Diagnostics<K>), Diagnostics<K>>` (accumulating) — with `K` being the trait-owning crate's per-trait kind enum.

Concretely:

| Trait | Method | Return shape |
|---|---|---|
| `CatalogProvider` | `resolve(...)` | fail-fast with `CatalogProviderErrorKind` |
| `FileSystem` | `read(...)`, `list(...)`, `glob(...)` | fail-fast with `FileSystemErrorKind` |
| `Repository` | `load(...)`, `save(...)` | fail-fast with `RepositoryErrorKind` |
| `EngineAdapter` | `adapt(...)` | fail-fast with `AdaptErrorKind` |

Untyped or string-payloaded `Result<T, String>` / `Result<T, Box<dyn Error>>` returns are banned per §5.5. Typed enums ARE the API; they are not internal carriers converted at a boundary.

### 8.4 Async trait methods

When a trait method is async (only `CatalogProvider`, `FileSystem`, `Repository` in v1 — see §9), the trait uses `async fn` directly (Rust 1.75+ async-fn-in-trait). `dyn` dispatch over the trait is provided via an adjacent object-safe facade when needed; the trait itself is not object-safe. Per-crate docs (`33`, `37`) specify the exact facade shape.

## Testing Strategy

- Sealed trait pattern via private super-trait (§8.1 sealed::Sealed trait)
- Cross-cutting trait roster completeness validated against codebase (§8.2 table)
- Trait-method return shape matches §7.2 patterns (fail-fast or accumulating)
- Async fn in trait usage for CatalogProvider/FileSystem/Repository (§8.4)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
