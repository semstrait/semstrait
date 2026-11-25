---
prereqs: [00]
status: superseded
---

# 15_mapping_and_binding (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`15_mapping_and_binding/`](15_mapping_and_binding/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [15-binding-process](15_mapping_and_binding/binding-process.md) — The compile-time Binding process: EntityId derivation, source resolution (glob/catalog), schema reconciliation, SemanticMapping completeness, Coverage derivation, and ResolvedBinding materialization
- [15-catalog-absent-fallback](15_mapping_and_binding/catalog-absent-fallback.md) — Catalog-absent fallback — model-as-truth source-shape fields on Binding (locator, source_type, projected_schema, version_ref) authored directly on the model when no catalog is present; the catalog-optional override matrix per C1.1/C14.4/C14.5/C15.5
- [15-physical-source](15_mapping_and_binding/physical-source.md) — PhysicalSource sum type (File/Table/Snapshot variants), Schema/PartitionColumn/CatalogRef shapes, source resolution (glob expansion, catalog fetch, deterministic ordering)
- [15-file-format](15_mapping_and_binding/file-format.md) — FileFormat enumeration (Parquet/Csv/Json/Orc/Avro), CsvOptions/JsonOptions, per-format schema-resolution strategy, format inference from path
- [15-semantic-mapping](15_mapping_and_binding/semantic-mapping.md) — SemanticMapping structure (BTreeMap keyed on SemanticsName), per-variant compile semantics (Column/Literal/Expr/Metadata), completeness rule (every Semantics covered exactly once)
- [15-coverage](15_mapping_and_binding/coverage.md) — Coverage at Binding level (Native/NullFill/Derived/Metadata), per-source × per-Semantics truth table, computation at compile, scope boundary with 16's composition-level Coverage
- [15-resolved-column-mapping](15_mapping_and_binding/resolved-column-mapping.md) — SemanticManifest-layer ResolvedColumnMapping (four flat HashMaps per value category), ResolvedBinding/ResolvedPhysicalSource envelope, planner-speed contract, relation to 19's ResolvedExprTable
- [15-expression-strata](15_mapping_and_binding/expression-strata.md) — Three-stratum expression model: UnresolvedSemanticExpr (layer 1, logical), PhysicalExpr (layer 2, lowered SQL-equivalent), compile-time mechanics (layer 3, path_token v1 inhabitant)
- [15-metadata-extraction](15_mapping_and_binding/metadata-extraction.md) — MetadataDimension extraction semantics: v1 path-token mechanics (path_token compile function, token indexing, Cast to declared data_type, per-source LiteralValue resolution), v2 partition extraction deferred, Coverage::Metadata
- [15-schema-reconciliation](15_mapping_and_binding/schema-reconciliation.md) — Schema reconciliation at compile: widening/narrowing/incompatible casts (per 14 §6.4), Computed-entry type inference reconciliation, cross-source type agreement (strict), nullability reconciliation (warning)
