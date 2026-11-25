---
id: 15-catalog-absent-fallback
status: ratified
version: 2.1
impl: implemented
---

# Catalog-Absent Fallback

When no catalog is present, the model becomes the source of physical truth for source-shape fields on the Binding: `locator`, `source_type`, `projected_schema`, and `version_ref`. This document defines the catalog-optional override matrix per C1.1/C14.4/C14.5/C15.5. See [binding-process](binding-process.md) for the compile-time resolution flow.

### 2.5 Catalog-absent fallback — model-as-truth source-shape fields

> **Posture cascade.** Per the ratified catalog-optional rule (C1.1) and catalog-absent compile path (C14.4): when `catalog: Option<&dyn CatalogProvider>` is `None` at the `compile` entry point, the model is the source of truth for physical shape. The compile pass reads the per-binding fields ratified here directly into the manifest-layer `PhysicalSource` (`33 §PhysicalSource`) without a catalog round-trip. Today these inputs live implicitly in catalog responses; this subsection makes them explicit on the `Binding` so authoring is well-defined when no catalog is wired.

**Per-binding optional fields.** The Binding shape in §2.1 grows four optional fields, structurally additive (each field is `Option`, default `None`, `#[non_exhaustive]` invariants preserved):

```rust
#[non_exhaustive]
pub struct Binding {
    pub binding_id: EntityId,
    pub sources: Vec<PhysicalSource>,
    pub semantic_mapping: SemanticMapping,
    pub coverage: Option<Coverage>,

    // Catalog-absent fallback inputs — model-as-truth source-shape fields.
    // Each field is `None` when the author defers to catalog; `Some(_)` when
    // the author authors the shape directly (per the override matrix below).
    pub locator: Option<String>,
    pub source_type: Option<PhysicalSourceType>,
    pub projected_schema: Option<Vec<SourceColumn>>,
    pub version_ref: Option<PhysicalSourceVersionRef>,
}
```

Field semantics:

- **`locator: Option<String>`** — provider-interpreted physical address. v1 carriage matches the `33 §PhysicalSource.locator` shape — table-id (`"iceberg.sales.transactions"`) for `Table`-typed sources; absolute file path or glob root (`"s3://bucket/year=*/orders/"`) for `File`-typed sources. Semstrait does no normalization beyond what `15 §3.5` already specifies for resolved-source ordering.
- **`source_type: Option<PhysicalSourceType>`** — the v1 closed roster `{ Table, File }` per `33 §PhysicalSource` C3.3. Distinguishes catalog-table reads from filesystem reads at the manifest layer; the foundations-layer `PhysicalSource::{File, Table, Snapshot}` enum (§3.1) is the resolved counterpart.
- **`projected_schema: Option<Vec<SourceColumn>>`** — the column shape (`name`, `source_type: String`, `nullable: bool`) per `33 §SourceColumn` C3.2. `source_type` here is the native-string type the author declares (e.g. `"int4"`, `"VARCHAR"`); canonical `DataType` mapping is deferred to the planner / engine registry per the C3.2 cascade. **When `projected_schema` is `None` and catalog is absent, `schema_fingerprint` on the resulting `PhysicalSource` is `None`** per C15.5 — empty / unknown schemas do not synthesize an all-zeroes hash.
- **`version_ref: Option<PhysicalSourceVersionRef>`** — optional drift signal carrying `IcebergSnapshotId(i64)` or `MonotonicVersion(u64)` per `33 §PhysicalSourceVersionRef` C3.4. Author opts in when the source has a versionable identity worth pinning; otherwise `None` (no drift signal participates in `manifest_epoch` for this source).

**Compile-time semantics.** The override matrix in the catalog-absent fallback path:

| Catalog presence | Model fields | Behavior | Diagnostic |
| --- | --- | --- | --- |
| `Some(_)` | all `None` | catalog wins — eager fetch per C14.1 produces the `PhysicalSource` shape | — |
| `Some(_)` | any `Some(_)` | model wins — author override per C14.5; catalog value not consulted for the overridden field | `COMP_W_CATALOG_OVERRIDE { source_id, catalog_value, model_value }` (warning) |
| `None` | all `Some(_)` necessary | compile reads model directly into `PhysicalSource` (no round-trip per C14.4); `version_ref` honored if model provides it, else `None` | — |
| `None` | required field absent | hard fail-fast at compile — defense-in-depth for the catalog-optional posture | `COMP_E_BINDING_SOURCE_SHAPE_MISSING { binding_id }` (error) |

The "required field absent" row treats `locator` and `source_type` as the minimum pair needed to construct any manifest-layer `PhysicalSource`; `projected_schema` may be `None` (drives `schema_fingerprint = None` per C15.5); `version_ref` is always optional. The diagnostic surfaces at the binding-resolution pass (§10 step 2) when the catalog branch is not taken.

**Cross-references.**

- Forward to `33 §PhysicalSource` (C3 final shape), `33 §PhysicalSourceVersionRef` (C3.4), `33 §SourceColumn` (C3.2), `33 §PhysicalSourceType` (C3.3) — the runtime shapes consumed downstream.
- Forward to `33 §C14.1` (eager-all catalog fetch), `33 §C14.4` (catalog-absent path), `33 §C14.5` (model-wins override semantics), `33 §C15.5` (fingerprint-none-when-empty rule).
- Backward to `15 §3.1`'s `PhysicalSource` sum type — that is the **post-compile resolved** form; the optional fields ratified here are the **pre-compile authoring inputs** that feed it when the catalog is absent.
- The Model YAML carrier for these fields (an analogous extension to `extras.storage` in `32`) is a separate spec touch, owned by `32`. `15` ratifies the Binding-shape contract; `32` ratifies the YAML grammar that fills it.

## Testing Strategy

- Override matrix validation: §2.5 catalog-present vs absent × model fields Some/None → catalog-wins / model-wins / compile-reads-model / required-field-absent
- C14.4 catalog-absent path: compilation succeeds when model provides locator/source_type
- C14.5 model-wins override: when catalog present and model also specifies source-shape fields, model values override catalog
- COMP_W_CATALOG_OVERRIDE warning: emitted when model override occurs
- COMP_E_BINDING_SOURCE_SHAPE_MISSING error: when required fields (locator, source_type) absent in catalog-absent mode
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
