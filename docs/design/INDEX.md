# semstrait Design Index

Status: **Living**

This file is the primary navigator for `docs/design/`.

Use this file when:
- you know the concept and need its canonical home quickly;
- you need a first-read path for a specific task;
- you want to confirm question state and where a Q-ID currently lives.

Use `00_overview.md` when you need the governing contract (vocabulary, invariants, directionality rules).

---

## 1) First-Read Paths

### Any spec/design session
1. [`00_overview.md`](00_overview.md)
2. [`STATUS.md`](STATUS.md)
3. Relevant section below by topic

### Topic routing

| If you are working on... | Start here | Then read |
|---|---|---|
| Pipeline semantics and stage boundaries | [`foundations/10_resolution_pipeline.md`](foundations/10_resolution_pipeline.md) | [`apis/30_api_contracts.md`](apis/30_api_contracts.md) |
| Names, scopes, Semantics elements | [`foundations/11_names_and_scopes.md`](foundations/11_names_and_scopes.md) | [`foundations/16_composition.md`](foundations/16_composition.md) |
| Types, expressions, function semantics | [`foundations/13_types_and_grain.md`](foundations/13_types_and_grain.md) | [`foundations/14_expressions.md`](foundations/14_expressions.md), [`foundations/14a_function_catalog.md`](foundations/14a_function_catalog.md), [`foundations/14b_expression_resolution.md`](foundations/14b_expression_resolution.md), [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) |
| Expression pipeline, sugar, placement | [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) | [`foundations/14_expressions.md`](foundations/14_expressions.md), [`foundations/14a_function_catalog.md`](foundations/14a_function_catalog.md), [`apis/34_semstrait_planner.md`](apis/34_semstrait_planner.md) |
| Mapping/binding and metadata synthesis | [`foundations/15_mapping_and_binding.md`](foundations/15_mapping_and_binding.md) | [`apis/32_semstrait_model.md`](apis/32_semstrait_model.md), [`apis/33_semstrait_manifest.md`](apis/33_semstrait_manifest.md) |
| Temporal semantics | [`foundations/17_temporal_shape.md`](foundations/17_temporal_shape.md) | [`data-kinds/22_grainset.md`](data-kinds/22_grainset.md), [`data-kinds/23_unionset.md`](data-kinds/23_unionset.md) |
| DataKind taxonomy and variant behavior | [`data-kinds/20_taxonomy.md`](data-kinds/20_taxonomy.md) | [`data-kinds/21_dataset.md`](data-kinds/21_dataset.md), [`data-kinds/22_grainset.md`](data-kinds/22_grainset.md), [`data-kinds/23_unionset.md`](data-kinds/23_unionset.md), [`data-kinds/24_joinset.md`](data-kinds/24_joinset.md), [`data-kinds/25_applicability_matrix.md`](data-kinds/25_applicability_matrix.md), [`data-kinds/26_nesting_matrix.md`](data-kinds/26_nesting_matrix.md) |
| Crate-level API contract | [`apis/30_api_contracts.md`](apis/30_api_contracts.md) | target crate doc in `31`-`39` |
| Engine/provider mapping details | [`registry/README.md`](registry/README.md) | concrete mapping catalog(s) |
| Migration/refactor planning | [`implementation/40_refactor_plan.md`](implementation/40_refactor_plan.md) | [`implementation/41_deprecations.md`](implementation/41_deprecations.md), [`implementation/42_migration_notes.md`](implementation/42_migration_notes.md) |

---

## 2) Canonical Document Map

### Foundations (1x)
- `10` [`foundations/10_resolution_pipeline.md`](foundations/10_resolution_pipeline.md) — stage contracts and pipeline flow.
- `11` [`foundations/11_names_and_scopes.md`](foundations/11_names_and_scopes.md) — naming, scope, Semantics definitions.
- `12` [`foundations/12_nesting_policy.md`](foundations/12_nesting_policy.md) — nesting constraints and structural matrix.
- `13` [`foundations/13_types_and_grain.md`](foundations/13_types_and_grain.md) — canonical logical types and Grain.
- `14` [`foundations/14_expressions.md`](foundations/14_expressions.md) — canonical expression model.
- `14a` [`foundations/14a_function_catalog.md`](foundations/14a_function_catalog.md) — canonical function identity/registry.
- `14b` [`foundations/14b_expression_resolution.md`](foundations/14b_expression_resolution.md) — expression resolution semantics.
- `15` [`foundations/15_mapping_and_binding.md`](foundations/15_mapping_and_binding.md) — mapping/binding algorithm.
- `16` [`foundations/16_composition.md`](foundations/16_composition.md) — relationships and composed interfaces.
- `17` [`foundations/17_temporal_shape.md`](foundations/17_temporal_shape.md) — temporal shape model.
- `18` [`foundations/18_entities.md`](foundations/18_entities.md) — canonical entity type definitions.
- `19` [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) — two-phase expression pipeline, sugar contract, Phase B placement.

### DataKinds (2x)
- `20` [`data-kinds/20_taxonomy.md`](data-kinds/20_taxonomy.md) — taxonomy and shared invariants.
- `21` [`data-kinds/21_dataset.md`](data-kinds/21_dataset.md) — Dataset behavior.
- `22` [`data-kinds/22_grainset.md`](data-kinds/22_grainset.md) — Grainset behavior.
- `23` [`data-kinds/23_unionset.md`](data-kinds/23_unionset.md) — Unionset behavior.
- `24` [`data-kinds/24_joinset.md`](data-kinds/24_joinset.md) — Joinset behavior.
- `25` [`data-kinds/25_applicability_matrix.md`](data-kinds/25_applicability_matrix.md) — cross-variant applicability matrix.
- `26` [`data-kinds/26_nesting_matrix.md`](data-kinds/26_nesting_matrix.md) — allowed nesting combinations.

### API contracts (3x)
- `30` [`apis/30_api_contracts.md`](apis/30_api_contracts.md) — cross-crate API and stability policy.
- `31` [`apis/31_semstrait_core.md`](apis/31_semstrait_core.md) — `semstrait-core`.
- `31b` [`apis/31b_semstrait_core_io.md`](apis/31b_semstrait_core_io.md) — `semstrait-core::io`.
- `32` [`apis/32_semstrait_model.md`](apis/32_semstrait_model.md) — `semstrait-model`.
- `32b` [`apis/32b_catalogs_yaml.md`](apis/32b_catalogs_yaml.md) — catalogs YAML side-surface.
- `33` [`apis/33_semstrait_manifest.md`](apis/33_semstrait_manifest.md) — `semstrait-manifest`.
- `34` [`apis/34_semstrait_planner.md`](apis/34_semstrait_planner.md) — `semstrait-planner`.
- `35` [`apis/35_semstrait_ir.md`](apis/35_semstrait_ir.md) — `semstrait-ir`.
- `36` [`apis/36_semstrait_adapter.md`](apis/36_semstrait_adapter.md) — `semstrait-adapter`.
- `37` [`apis/37_semstrait_catalog.md`](apis/37_semstrait_catalog.md) — `semstrait-catalog`.
- `38` [`apis/38_semstrait_api.md`](apis/38_semstrait_api.md) — `semstrait-api`.
- `39` [`apis/39_semstrait_facade.md`](apis/39_semstrait_facade.md) — top-level facade crate.

### Implementation stubs (4x)
- `40` [`implementation/40_refactor_plan.md`](implementation/40_refactor_plan.md)
- `41` [`implementation/41_deprecations.md`](implementation/41_deprecations.md)
- `42` [`implementation/42_migration_notes.md`](implementation/42_migration_notes.md)

### Registry catalogs (living)
- [`registry/README.md`](registry/README.md) — registry policy and catalog index.
- [`registry/types_mapping.md`](registry/types_mapping.md)
- [`registry/functions_mapping.md`](registry/functions_mapping.md)
- [`registry/temporal_shape_mapping.md`](registry/temporal_shape_mapping.md)
- [`registry/join_types_mapping.md`](registry/join_types_mapping.md)

---

## 3) High-Value Concept Map (single-home pointers)

| Concept | Canonical home |
|---|---|
| Pipeline (`parse -> validate -> compile -> plan -> optimize -> adapt`) | [`foundations/10_resolution_pipeline.md`](foundations/10_resolution_pipeline.md) |
| Semantics element types and naming constraints | [`foundations/11_names_and_scopes.md`](foundations/11_names_and_scopes.md) |
| `DataType`, `Grain` | [`foundations/13_types_and_grain.md`](foundations/13_types_and_grain.md) |
| `Expr`, `SemanticExpr`, `PhysicalExpr` | [`foundations/14_expressions.md`](foundations/14_expressions.md), [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) (type-architectural form) |
| `CanonicalFn`, `FunctionRegistry` | [`foundations/14a_function_catalog.md`](foundations/14a_function_catalog.md) |
| Expression substitution/resolution | [`foundations/14b_expression_resolution.md`](foundations/14b_expression_resolution.md) |
| Two-phase expression flow, `resolve`, sugar (Family A/B) | [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) |
| `Accessor`, `MeasureAccessor`, `DimensionAccessor`, `MetricAccessor`, `KeyAccessor`, `Parameter` | [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) |
| Inline request-time filter (anonymous `{field, operator, value}` predicate) | [`foundations/11_names_and_scopes.md`](foundations/11_names_and_scopes.md) §6.4.2 |
| `DimensionRef`, `DimensionVariation` (Request shape) | [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md) |
| `Additivity`, `DimensionAxis` (function-tag axis) | [`foundations/19_expression_flow.md`](foundations/19_expression_flow.md), [`foundations/14a_function_catalog.md`](foundations/14a_function_catalog.md), [`foundations/18_entities.md`](foundations/18_entities.md) |
| `SemanticMapping` and binding flow | [`foundations/15_mapping_and_binding.md`](foundations/15_mapping_and_binding.md) |
| `Relationship`, composed interface semantics | [`foundations/16_composition.md`](foundations/16_composition.md) |
| `TemporalShape` and shape semantics | [`foundations/17_temporal_shape.md`](foundations/17_temporal_shape.md) |
| Canonical entities (`Cardinality`, `Integrity`, `Optional`, `CrossFilter`, derived `JoinType`, `DimensionType`, `AggregationType`, etc.) | [`foundations/18_entities.md`](foundations/18_entities.md) |
| `DataKind` taxonomy and trait axes | [`data-kinds/20_taxonomy.md`](data-kinds/20_taxonomy.md) |
| Typed diagnostics contract and observability policy | [`apis/30_api_contracts.md`](apis/30_api_contracts.md) |
| Unified API error sum (`SemStraitErrorKind`) | [`apis/38_semstrait_api.md`](apis/38_semstrait_api.md) |

---

## 4) Question-State Dashboard

Question sidecars are stateful by directory. Directory is authoritative for state.

- Open (active v1 backlog): [`questions/open/`](questions/open/)
- Closed (historical ratifications): [`questions/closed/`](questions/closed/)
- Deferred (post-v1 / parked): [`questions/deferred/`](questions/deferred/)

Current snapshot:

| Directory | Files | Lines |
|---|---:|---:|
| `open/` | 23 | ~2630 |
| `closed/` | 20 | ~1450 |
| `deferred/` | 18 | 797 |

For registry-specific questions, use the aggregate navigator:
- [`questions/open/registry_questions.md`](questions/open/registry_questions.md)

---

## 5) Sync Contract (must-follow)

When a canonical concept owner changes, update in the same commit:
1. The authoritative source doc.
2. This file (`INDEX.md`).
3. `STATUS.md` if phase/reconciliation/question state changed.

When question status changes:
1. Move/land the Q entry body into the correct state directory.
2. Leave only a short forwarding stub in the previous location when needed for discoverability.
3. Reflect state changes in `STATUS.md`.

For full authoring discipline and AI editing rules, see:
- [`DOCS_MAINTENANCE.md`](DOCS_MAINTENANCE.md)

