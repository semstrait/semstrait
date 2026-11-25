---
id: 30-feature-flag-policy
status: ratified
version: 2.1
impl: implemented
---

# Feature Flag Policy

This leaf owns the feature-flag policy — default features minimum viable, adapter crates are separate crates not feature flags, catalog provider crates are separate, optional serialization features, and banned feature patterns.

## 10. Feature-Flag Policy

### 10.1 Default features: minimum viable

Every `semstrait-*` crate ships with default features equal to the minimum set needed to use that crate's primary function. No crate gates a core type or trait behind an opt-in feature. A consumer depending on `semstrait-common` gets every public type in `semstrait-common` with no `--features` hunt.

### 10.2 Adapter crates are SEPARATE crates, not feature flags

The adapter surface is the primary extension axis of the workspace. Each engine adapter lives in its own crate:

- `semstrait-adapter-datafusion`
- `semstrait-adapter-duckdb`
- `semstrait-adapter-spark`
- `semstrait-adapter-substrait`
- (future) per additional target engine

Consumers add the adapter crate to their `Cargo.toml` dependency list rather than flipping a feature on a monolithic `semstrait-adapter` crate. Rationale:

- Dependency closures are surgical: a DuckDB-only consumer never compiles Spark's (heavy) transitive deps.
- The `RegistryExtension` layering of `14a §7` is one registration path per crate; feature flags would scatter `cfg(feature = "…")` across the adapter trait's implementations.
- Stability tiers (§13) can be set per-adapter-crate independently — a `Provisional` adapter does not drag a `Stable in v1` workspace label down.

### 10.3 Catalog provider crates are also separate

Same reasoning as §10.2 for `CatalogProvider` impls:

- `semstrait-catalog` (trait + minimal local-fs impl)
- `semstrait-catalog-iceberg`
- `semstrait-catalog-unity`
- (future) per additional metadata source

### 10.4 Optional serialization features

Serde support on public types is opt-in via a `serde` feature per crate. Default-off for `semstrait-common`, `semstrait-model`, `semstrait-manifest`, `semstrait-ir`. Enabled downstream (e.g. `semstrait-facade` turns it on by default for end-user convenience).

Serde support is documented per-crate in `31`–`39`. A `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` attribute on a public type is itself part of the public surface — adding it is MINOR (no existing caller code breaks); removing it is MAJOR.

### 10.5 Banned feature patterns

- **No `nightly`-only features.** Every crate compiles on stable Rust. `no_std` is not targeted.
- **No `default = ["every-adapter"]`.** A consumer that never mentions adapters gets no adapter crate in their build.
- **No mutually exclusive features within a crate.** Features compose additively; a consumer enabling `features = ["a", "b"]` gets the union.

## Testing Strategy

- Default features = minimum viable (no core type gated behind opt-in)
- Adapter crates exist as separate workspace crates (§10.2 separate-crate policy)
- Catalog provider crates exist as separate workspace crates (§10.3)
- Serde feature opt-in (default-off for core crates, §10.4)
- Banned feature patterns not present (§10.5 no nightly-only, no default-all, no mutual exclusion)

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
