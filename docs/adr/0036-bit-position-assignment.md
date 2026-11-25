---
number: 36
title: '`bit_position` is a dense index into one manifest, not an identifier'
status: accepted
date: 2026-08-05
source:
- crates/semstrait-manifest/src/manifest/bitmap.rs
links:
- target: 12
  kind: amends
- target: 40
  kind: amendedby
- target: 65
  kind: amendedby
---

# `bit_position` is a dense index into one manifest, not an identifier

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0040](0040-manifest-keyset-roster-is-deduplicated.md): corrects the closing sentence: `Key` will never join the `bit_position` roster, because it is no longer a registry role; operative rules 1–5 stand untouched.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): the `Field` role and its minter are removed (the roster is Dimension | Measure | Metric | Filter), so `Field` will not join it either; `bit_position` stays under 0036 unchanged, and 0036's bit-position assignment gains a carrier that proves what it asserts.

## Context and Problem Statement

`bit_position` is counter-intuitive in a way that invites silent corruption: it is a persisted `u32` on a persisted struct, it looks exactly like a stable key, and the natural mistake — caching a coverage mask across recompiles — produces wrong answers rather than an error.

## Considered Options

- A dense index, frozen once (chosen).
- **Append-only numbering that keeps a semantic's bit across recompiles, leaving holes on removal.** The only shape that delivers cross-recompile stability, and it requires the previous manifest as a `compile()` input — moving the compile/persistence boundary and breaking the "byte-identical for identical inputs" guarantee as stated. Rejected in favour of scoping the guarantee to one artifact and forbidding cross-manifest mask reuse.
- **Ordering by author name** (the pre-existing behaviour at `compile/index.rs:194`). Names are not unique across the registry: a synthesized join-key `Field` id is seeded on `(owning_data_kind_id, name, role)` (`identity.rs:219`), so two DataKinds may each contribute a `Field` of the same name, leaving a name sort with no deterministic tiebreak. It coincided with `EntityId` order only while ids were transiently the name itself.
- **Ordering by declaration order.** Not recoverable: the model's `dimensions` / `measures` / `metrics` pools are id-keyed `EntityBTreeMap`s (`semstrait-model/src/model/mod.rs:73-79`), so authoring order is gone before `compile()` runs.
- **Assigning positions at step 3 of the earlier compile design**, where the compile index assigned them at the time of this decision. Synthesized `Field`s, `Key`s, and named filters are minted after step 3, and under density a late arrival would renumber everything, so no correct assignment exists at that point.
- **Widening `bit_position` to `u64` to make overflow unrepresentable.** Reads convert a position to a word index with `as usize` (`manifest/bitmap.rs:156`, `:171`), which truncates *silently* on 32-bit targets — trading one explicit narrowing at a single write site for silent wraparound at several read sites. Capacity is bounded by mask memory (`ceil((n+1)/64)` words to use position `n`), not by integer range, so the extra width is unallocatable. It would also edit a ratified persisted field and lose exact JSON round-tripping above 2^53.
- **Clamping past the width** (`u32::try_from(ordinal).unwrap_or(u32::MAX)`, the pre-existing behaviour). Collapses every excess entry onto one position, violating both uniqueness and density, and surfaces at load as a misleading "must be unique" error rather than the real cause.
- **Panicking at the size guard** instead of returning an error. Rejected to keep the compile path uniformly `Result`-shaped: a library crate hands an embedder a typed error rather than aborting its process.

## Decision Outcome

`bit_position` is assigned as the entry's 0-based index in canonical `EntityId` order over the `SemanticBitmap` registry, frozen once — inside the bitmap's own constructor, after the entry roster is complete and before any mask is built — so the positions of an `n`-entry registry are exactly `{0..n-1}`. That density discharges the "monotonic, no gaps, starting from zero" requirement and strengthens the ratified manifest contract, which requires only uniqueness within a `manifest_epoch`. It also lands the `bit_position` contiguity check the earlier status notes listed as a deferred derived-view check.

The guarantee is scoped to **one manifest artifact**. Positions are a function of the registry's key set, so adding or removing any semantic renumbers its neighbours. A `bit_position` is therefore explicitly **not a persistable identifier**: no consumer may persist a bit or apply a mask from one manifest to another, and `EntityId` is the only durable handle.

The cross-recompile stability that would make caching safe is deliberately not bought. It is unnecessary given that an unchanged model does not recompile at all (the `model_hash` gate), and it would cost either byte-determinism or density.

Assignment is a rule over whatever the registry holds, so it is stated lane-parameterized: if per-class registries are later added, the identical rule applies independently within each lane and every mask declares the lane it addresses, which keeps it correct as `Key`, `Field`, and named filter entries join the roster.

### Operative rules

1. **Ordering key** — canonical `EntityId` order, as ratified. Realized by enumerating the `BTreeMap<EntityId, _>` rather than sorting a list, so uniqueness and density hold by construction instead of by test. Ids are canonical lowercase UUID text, so the ordering is total and deterministic.
2. **Freeze point** — the roster is complete before positions exist. `compile()` must therefore accumulate coverage as `EntityId` sets and convert them to masks *after* the freeze, rather than reading positions off definitions mid-walk.
3. **Density** — for `n` entries the position set is exactly `{0..n-1}`. Removal renumbers; holes are unrepresentable.
4. **Load check** — The load check asserts the occupied position set *equals* `0..n`, not merely that its cardinality matches `n`. The former rejects a gapped or hand-edited manifest that the latter admits.
5. **Bound** — the ratified `u32` width is kept, and the registry size is guarded once at the freeze, yielding `CompileError::BitmaskValueOverflow`. Per-entry conversions after that guard are infallible.

### Amends

- **0012:** `bit_position` is a dense index scoped to one manifest artifact, not a persistable identifier, and no consumer may persist a bit or apply a mask from one manifest to another.
