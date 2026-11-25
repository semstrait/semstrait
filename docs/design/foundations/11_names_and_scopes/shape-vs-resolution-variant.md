---
id: 11-shape-vs-resolution-variant
status: ratified
version: 2.0
impl: implemented
---

# Shape vs. Resolution Variant

Shape-vs-resolution-variant boundary — which fields lock across occurrences, which may differ per DataKind. This leaf owns the field classification boundary; [semantics element catalog](semantics-element-catalog.md) owns the per-element field catalogs.

<!-- cross-cut: 11-semantics-element-catalog -->

## 5. Shape vs. Resolution Variant

Every field of a Semantics occurrence is classified as either **shape** (must unify across all occurrences of a given name) or **resolution variant** (may differ per DataKind).

### 5.1 Shape fields (locked across occurrences)

These fields must be identical (or absent) across all occurrences of a given Semantics name. Conflict is a compile error with a Diagnostic pointing to every disagreeing occurrence.

| Field | Applies to | Role |
|---|---|---|
| `name` | all Semantics | the global identifier |
| element type (Dimension / Measure / Metric / Filter) | all Semantics | fixed by which list the name first appears in; switching lists across occurrences is an error |
| `data_type` | all Semantics carrying a value | canonical `DataType` (13) |
| `description` | all Semantics | prose; must unify |
| `additivity` | Measure, Metric | classification (§7); explicit per-occurrence override locked — must agree |
| `agg` | Measure | aggregation function (13 / 14); locked |
| `grains` (temporal) | Dimension of temporal `Type` | temporal grain axis (13 / 17); locked |
| `metadata` hints | Dimension | metadata source (`01 metadata path`, etc.); locked |

**Casting of `data_type`.** Initial design: strict equality. Implicit-cast detection across occurrences (treating `decimal` and `numeric` as equivalent etc.) is out of scope; it would weaken the global-identity guarantee without clear benefit. Every occurrence must spell `data_type` identically.

### 5.2 Resolution-variant fields (may differ per occurrence)

The only resolution-variant field is `expr`. Each top-level DataKind occurrence may carry its own `expr:` for the Semantics, producing a DataKind-local resolution variant. Nested DataKinds do NOT carry Semantics occurrences at all (§10), so the variant granularity is exactly one-per-top-level-DataKind.

A Semantics may have:

- **zero** `expr:` occurrences (resolution comes from `SemanticMapping` in the Simple leaf Bindings, `15`),
- **one** `expr:` at Model root (Tier-1 default, used by every top-level DataKind that doesn't supply its own),
- **one** `expr:` per top-level DataKind (DataKind-local variant overriding any Tier-1 default),
- **or a mix** of the above.

### 5.4 Defaults for omitted shape fields

Default values applied when a shape field is omitted on every occurrence of a Semantics:

| Field | Applies to | Default when all occurrences omit |
|---|---|---|
| `additivity` | Measure, Metric | `Additive` |
| `description` | all | empty string |
| `expr` | all (it's a resolution-variant field) | none (resolution falls through to direct `SemanticMapping` — 15) |

Shape fields with NO default — must be stated in at least one occurrence, else `CompileError::SemanticShapeIncomplete`:

- `name` (required by construction)
- element type (determined by which list the name appears in — `dimensions:` / `measures:` / `metrics:` / `filters:`)
- `data_type`
- `agg` (for Measure)

**Note on `additivity`.** The default is `Additive`, not derived from the owning DataKind's `TemporalShape`. See §7.2 for rationale — Additivity and TemporalShape are independent concerns. A Warning Diagnostic MAY be surfaced when the two appear likely to disagree (e.g. default-`Additive` on a Measure whose containing DataKind has `TemporalShape: SCD` with a history-preserving subtype), but the author's explicit or default `additivity:` value is always authoritative.

## Testing Strategy

- Shape field unification (N-C1: conflict across occurrences of same name is rejected)
- Resolution-variant field divergence (`expr:` may differ per DataKind, §5.2)
- Default application (§5.4: omitted `additivity` defaults to `Additive`, omitted `description` to empty string)
- Shape-fields-per-type cross-cut table enforcement (§5.1 table matches per-element catalog field shapes)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
