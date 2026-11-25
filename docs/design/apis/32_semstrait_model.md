---
prereqs: [00]
status: superseded
---

# 32_semstrait_model (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`32_semstrait_model/`](32_semstrait_model/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [32-model-root-shape](32_semstrait_model/model-root-shape.md) — Root YAML shape and SemanticModel in-memory type for the semantic_model envelope
- [32-model-datakind-hierarchy](32_semstrait_model/model-datakind-hierarchy.md) — DataKind type hierarchy — DataKindBase common struct, per-variant bodies, sealed traits, trait matrix, view enums, and storage layout
- [32-model-extras-mapping](32_semstrait_model/model-extras-mapping.md) — Extras blocks (LeafExtras and ComplexExtras) and semantic mapping grammar with Binding process
- [32-model-expr-vocabulary](32_semstrait_model/model-expr-vocabulary.md) — Crate-level home of the authoring expression vocabulary (stub — ratified in doc 14)
- [32-model-parse-validate](32_semstrait_model/model-parse-validate.md) — Parse and validate stage APIs — parse signature, ParseErrorKind roster, validate signature, ValidateError roster, parse/build pipeline, identity profiles
- [32-model-loader](32_semstrait_model/model-loader.md) — Fluent loader API — SemanticModel::loader() ergonomic wrapper over parse + validate
- [32-model-builder-api](32_semstrait_model/model-builder-api.md) — Builder API — per-struct fluent builders with structural fidelity rules, variant-body 1:1 constructors, ergonomic facade
- [32-model-crate-boundaries](32_semstrait_model/model-crate-boundaries.md) — Structural rules (SR-*), deterministic ordering (I4), ${VAR} substitution, crate boundaries, feature flags, pointers to child docs, hand-rolled authoring serde
