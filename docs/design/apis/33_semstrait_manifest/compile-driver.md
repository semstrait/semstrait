---
id: 33-compile-driver
status: ratified
version: 2.2
impl: implemented
---

# Compile Driver

This leaf owns the compile entry point, pass orchestration, validation gates G1–G5, desugar policy, and deterministic entity identity generation.

## 9. Compile and Persistence Boundary (C13 + C14)

### 9.1 `compile`

```rust
pub async fn compile(
    model: SemanticModel,
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
) -> Result<
    (SemanticManifest, Diagnostics<CompileError>),
    (Diagnostic<CompileError>, Diagnostics<CompileError>),
>;
```

Notes:

- `catalog: Option<&dyn CatalogProvider>` reflects C1.1's catalog-optional posture.
- The two-channel return shape (`Diagnostics` carries warnings and informational items alongside the success or failure case) is unchanged from the previous spec.
- **Relationship resolution.** Compile turns each authoring `Relationship` (spec 18 §2) into a `ResolvedRelationship` (§8A): resolve `from` / `to` to `EntityId`; apply the `optional` / `cross_filter` defaults matrix (spec 18 §2.7); derive `join_type` (spec 18 §2.9); lower each join-key side through Phase A and represent it as a semantic-id pair (decomposing computed keys into synthesised implicit `Field`s); lower the residual `filter` to a pooled `PhysicalExprId`.
- **Temporal propagation.** Compile carries each leaf `Dataset`'s authored `TemporalShape` (spec 18 §3) onto its manifest `DataKind.temporal` (§8B), resolving the identifying-Dimension `SemanticsName`(s) to `EntityId`. Shape is leaf-effective; complex kinds carry `None`.
- **Identity propagation and generation.** `compile` carries each model entity's `id` (`32 §2`/`§2.3`) straight onto the corresponding manifest entity (it is both the value's `id` field and its map key). For entities with no model-authored identity — `PhysicalSource`, `SemanticBinding`, `SemanticInterface`, implicit Unionsets (`origin = Implicit`), synthesised `Field` semantics, and the top-level `model_id` — `compile` generates a **deterministic, content-derived** `EntityId` using **UUIDv5** (RFC 4122 name-based, SHA-1 over namespace + content). Deterministic derivation (not time-based UUIDv7) is mandatory so identical `(model, catalog)` inputs yield byte-identical manifests (I4).
  - **Per-entity-type namespace + content seed** (so distinct entity classes never collide and the same content reproduces the same id):
    - `PhysicalSource` → namespace `source`, seed = `(source_type, locator, version_ref)`.
    - `SemanticBinding` → namespace `binding`, seed = `(data_kind id, source id, canonical mapping encoding)`.
    - `SemanticInterface` → namespace `interface`, seed = the sorted member-`EntityId` set.
    - Implicit `Unionset` → namespace `data_kind`, seed = the parent `Dataset`'s structural identity + per-source list.
    - Synthesised `Field` semantic → namespace `semantic`, seed = `(owning data_kind id, name, role)`. This includes the implicit `Field` synthesised for a **computed relationship join key** (§8A.2), whose owning DataKind is the key's side (`from` / `to`) and whose binding mapping holds the lowered `PhysicalExprId`.
    - `model_id` → namespace `model`, seed = `model_hash`.
  - **Format.** `EntityId` is canonical UUID text per `31 §2.1`: authored ids are UUIDv7 (`32`); generated ids are UUIDv5. The version nibble (`7` vs `5`) records provenance but is informational — CX1 (§10) validates canonical-UUID-text shape and global uniqueness, not the specific version. The namespaces and seeds above are therefore a *derivation* contract, not a *validation* one: they fix which id compile produces, while what any consumer may reject is only the shape.

### 9.2 Validation gates (C13)

The compile pass runs five gates. All gates emit **errors** unless explicitly noted otherwise.

| Gate | Scope | Notes |
|---|---|---|
| **G1** | Cycle detection across `UnresolvedSemanticExpr`. | Error. Scoped to expression DAGs; cycle detection across the Relationship graph is a separate gate (G6 candidate, deferred). |
| **G2** | Semantic resolution. | Error. Every `UnresolvedSemanticLeaf`'s referenced semantic (by `SemanticsName`) must resolve to a registry entry (`EntityId`). |
| **G3** | Type validation. | Error. Leaves match registry type; calls match function signature; operators match operand types. |
| **G4** | `PhysicalExpr` binding-side checks. | Error. Referenced `PhysicalExprId` exists; `ColumnRef` resolves in the bound `PhysicalSource.projected_schema`; inferred type matches the bound semantic. |
| **G5** | Orphan / dead-code detection. | **Error** (escalated from warning per ratification). Unreachable expressions risk propagating into plan as undefined behavior; strict-default starting place. Workflow-friction watch is recorded under deferred threads. |

### 9.3 Desugar policy (D2)

Sugar is rewritten **before persistence** (C13 D2). The persisted `PhysicalExpr` is post-desugar physical form; the `UnresolvedSemanticExpr` (post-sugar) exists only transiently at compile and is not persisted. Benefits `manifest_epoch` / `model_hash` stability across cosmetic reformulations.

### 9.4 Graph-build invalidation inputs

- `manifest_epoch` is the coarse invalidation boundary for graph fragments and planner caches.
- `catalog_fingerprint` and per-source `schema_fingerprint` provide drift-sensitive invalidation inputs.
- Expression-pool ids are content-stable inside an epoch; expression edits produce new ids and naturally invalidate stale references.
- Cycle state is not persisted. Compile gate G1 enforces `UnresolvedSemanticExpr` acyclicity; relationship-graph cycle policy remains a separate planner/runtime concern (deferred G6).

### 9.5 `Repository`

`Repository` remains the persistence boundary and is the only out-of-band manifest I/O surface. Loading is contractually paired with §10's load-time integrity validation.

### 9.6 Validation ownership by phase

| Phase | Owner | What is validated |
|---|---|---|
| Model parse | `32` (`semstrait-model`) | Authoring syntax/shape admissibility and parse-time primitive checks (for example malformed literals / invalid decimal parameters), plus named-entity `id` format/profile checks. |
| Compile | `33::compile` | G1 expression-cycle detection, G2 semantic-id resolution, G3 type checks, G4 binding/column/expr checks, G5 orphan/dead-code checks. |
| Manifest load | `33::Repository::load` | CX1 cross-reference integrity and structural constraints (including bitmask canonical form and out-of-range bit rejection). |
| Graph build | `34` runtime over `35` graph types | Rebuild runtime indices, resolve every `GraphExprRef` to the correct expression pool, reject graph-fragment cycles before planning. |
| Planner build/validate | `34` + `35::SemanticPlan::validate` | Plan-tree invariants (types, schema alignment, boolean predicates, join/union constraints). |

Cycle-prevention scope in v1: expression cycles are compile-time (G1); runtime graph-fragment cycles are graph-build-time. Relationship-graph global cycle gate (G6) remains deferred.

## Testing Strategy

- G1–G5 gate enforcement: each gate detects its error class and emits the correct diagnostic
- D2 desugar round-trip: sugar rewrites produce canonical physical form pre-persistence
- UUIDv5 determinism: identical (model, catalog) pairs yield byte-identical manifests
- Per-entity-type namespace isolation: no id collisions across entity classes
- model_id generation: derived from model_hash via UUIDv5 with namespace 'model'

See the shared behavioral harness: [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md)
