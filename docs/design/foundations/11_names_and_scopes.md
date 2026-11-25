---
prereqs: [00]
status: superseded
---

# 11_names_and_scopes (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`11_names_and_scopes/`](11_names_and_scopes/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [11-scope-chain](11_names_and_scopes/scope-chain.md) — The four-level scope tree (Root / Kind / Nested-kind / Binding), tree-shape invariants, and nested-kind structural labels
- [11-semantics-identity](11_names_and_scopes/semantics-identity.md) — Global identity, unified namespace, per-level uniqueness, free introduction, and identifier grammar for Semantics names
- [11-yaml-declaration-forms](11_names_and_scopes/yaml-declaration-forms.md) — YAML declaration forms for Semantics occurrences — single-form rule, scalar shorthand, field-driven role
- [11-shape-vs-resolution-variant](11_names_and_scopes/shape-vs-resolution-variant.md) — Shape-vs-resolution-variant boundary — which fields lock across occurrences, which may differ per DataKind
- [11-semantics-element-catalog](11_names_and_scopes/semantics-element-catalog.md) — Semantics element catalog — Dimension, Measure, Metric, Filter, Key role definitions and field catalogs
- [11-additivity](11_names_and_scopes/additivity.md) — Additivity enum, authoring rules, defaults, SemiAdditive axis encoding, and planner contract
- [11-constraint](11_names_and_scopes/constraint.md) — Constraint framework — carriers, kind sub-blocks, v1 realized carriers (Measure/Metric), reserved carriers, validation lifecycle, error model
- [11-cross-kind-reference-rule](11_names_and_scopes/cross-kind-reference-rule.md) — Cross-kind reference rule — Relationship-required invariant, compile pre-validation, plan walk resolution
- [11-lookup-algorithm](11_names_and_scopes/lookup-algorithm.md) — Lookup algorithm — how compile resolves a named reference against the scope tree and global registry, producing indices for plan
- [11-precondition-catalog](11_names_and_scopes/precondition-catalog.md) — Name-related Precondition catalog — N-V1 through N-C9, mapping to typed error variants
