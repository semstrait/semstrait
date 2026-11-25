---
id: 18-keys
status: ratified
version: 2.1
impl: implemented
---

# Keys

This leaf owns the `Keys` struct — primary, unique, and foreign key declarations with bare Semantic names. Keys are metadata for relationship graph consumption, not runtime constraints.

## 9. `Keys`

```rust
#[non_exhaustive]
pub struct Keys {
    /// Primary key — the unique row identifier. At most one declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary: Option<KeyDecl>,

    /// Additional unique keys beyond the primary, keyed by EntityId.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unique: BTreeMap<EntityId, KeyDecl>,

    /// Foreign keys — references to other DataKinds' primary / unique keys,
    /// keyed by EntityId. Relationship graph uses these to infer join paths.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub foreign: BTreeMap<EntityId, ForeignKeyDecl>,
}

#[non_exhaustive]
pub struct KeyDecl {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in `Keys.unique`.
    pub id: Option<EntityId>,
    /// Bare Semantic names — no physical column references.
    /// Resolution through `semantic_mapping` at binding time.
    pub fields: Vec<SemanticsName>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[non_exhaustive]
pub struct ForeignKeyDecl {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in `Keys.foreign`.
    pub id: Option<EntityId>,
    pub fields: Vec<SemanticsName>,
    /// The target DataKind whose primary / unique key is referenced.
    pub references: DataKindName,
    /// The target DataKind's key Semantic names.
    pub target_fields: Vec<SemanticsName>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
```

Ordering of `unique` / `foreign` for serialization follows the optional `name` when present, falling back to `id` order when absent (these key decls carry no required name). The `primary` key remains a single optional declaration; its `id` is generated like any other.

YAML:

```yaml
datasets:
  - name: orders
    keys:
      primary: { fields: [order_id] }
      unique:
        - { fields: [external_id], name: ext_id_uk }
      foreign:
        - fields: [customer_id]
          references: customers
          target_fields: [id]
```

### 9.2 Keys are metadata, not constraints

Keys are consumed for:

- **Relationship graph** — `16 §11`'s implicit composition consults foreign keys when author-declared `relationships:` are absent.
- **SemanticManifest statistics** — `compile` emits pre-computed statistics for the planner.
- **Future SemanticInterface exposure** — external consumers (LSP, API) may surface keys as part of the entity description.

Keys are NOT enforced at query time (no duplicate checking, no referential-integrity validation). That's database territory, not a semantic-model concern.

## Testing Strategy

- KeyDecl/ForeignKeyDecl struct round-trip; EntityId minting on optional id; primary (single Option) vs unique/foreign (BTreeMap keyed by EntityId); bare Semantic names; metadata-only (not runtime enforcement).
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
