---
id: 11-yaml-declaration-forms
status: ratified
version: 2.0
impl: implemented
---

# YAML Declaration Forms

YAML declaration forms for Semantics occurrences — single-form rule, scalar shorthand, field-driven role. This leaf owns the authoring surface; [semantics identity](semantics-identity.md) owns the identity rules, [shape vs. resolution variant](shape-vs-resolution-variant.md) owns the field classification.

## 4. YAML Declaration Forms for Semantics Occurrences

Exactly one form, with an optional scalar shorthand for pure reference.

### 4.1 Primary form — `- name: X` + optional fields

```yaml
dimensions:
  - name: date                  # with or without additional fields
    data_type: date             # optional; if present, contributes to shape (§5)
    description: "calendar date" # optional; if present, contributes to shape
    expr: event_date            # optional; if present, is this scope's resolution variant
```

The role of an occurrence is determined **purely by which fields are present**:

| Fields present | Role |
|---|---|
| `name:` only | Pure reference (contributes nothing beyond "this Semantics is used here") |
| `name:` + shape fields (`data_type`, `description`, type discriminator, `additivity`, `agg`, …) | Shape-contributing declaration (must agree with all other occurrences per §5) |
| `name:` + `expr:` only | Resolution variant for the enclosing DataKind |
| `name:` + shape fields + `expr:` | Declaration + resolution variant simultaneously |

`compile` unifies the set of occurrences keyed by name and produces one Semantics registry entry per name.

### 4.2 Scalar shorthand — `- X`

```yaml
dimensions:
  - date                        # equivalent to `- name: date` with no other fields
```

Pure syntactic sugar. Semantically identical to the primary form with only a `name:` field. Lints/formatters may normalize either direction.

### 4.3 What is NOT a form

- **No `ref:` keyword.** Legacy's `- ref: X` is replaced by pure-reference via the primary form (or scalar shorthand). Tooling SHOULD surface a helpful error if `ref:` is encountered during parse.
- **No `override:` keyword.** Legacy's implicit or explicit override is replaced by per-DataKind `expr:` specialization (§5) under global identity.
- **No `alias:` or dotted-name form.** Names are global; aliasing or dot-qualification is out of scope (`00 §4.2` banned terms).

### 4.4 YAML anchors and aliases

YAML anchors (`&foo`) and aliases (`*foo`) operate at the YAML parse layer. The parser sees the expanded document tree; anchors/aliases are transparent at the Semantics level.

- `- *some_measure_anchor` expands to an inline copy of whatever the anchor referenced. That expanded content is then interpreted per §4's field-driven rules — no special treatment.
- Anchors/aliases are a YAML-layer DRY mechanism for authors, not a Semantics-level reference mechanism. They do NOT create resolution variants, participate in identity, or affect the Semantics registry. Two occurrences produced from the same anchor are two equivalent occurrences, unified by `compile` the same way any other two occurrences are.
- Tools that serialize or pretty-print a `SemanticManifest` MUST NOT assume that anchors/aliases from the original YAML survive the round trip. The `SemanticManifest` is normalized; the original YAML shape is not reconstructible from it (this matches `00 §4.1` `SemanticModel` as post-parse representation — anchors are resolved during YAML parse, before `parse` returns).

## Testing Strategy

- Single-form rule parsing (primary form `- name: X` + optional fields)
- Scalar shorthand conversion (`- X` ≡ `- name: X` with no other fields)
- Field-driven role determination (§4.1 table: pure reference / shape-contributing / resolution variant / both)
- YAML anchors/aliases transparency (expanded content is interpreted via §4 rules, no special treatment)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
