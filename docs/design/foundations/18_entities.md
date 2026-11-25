---
prereqs: [00]
status: superseded
---

# 18_entities (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`18_entities/`](18_entities/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [18-semantics-pools](18_entities/semantics-pools.md) — Shared Semantics pools (dimensions/measures/metrics) at the root level and their per-DataKind reference / override grammar, plus orphan-binding policy
- [18-relationship](18_entities/relationship.md) — Relationship struct shape, identity, cardinality, integrity, optional, cross_filter, JoinType derivation, and JoinKeyExprPair
- [18-temporal-shape](18_entities/temporal-shape.md) — TemporalShape type hierarchy — TemporalShape struct, TemporalShapeKind enum, per-variant body structs, ScdType roster, grain placement rules
- [18-dimension](18_entities/dimension.md) — Dimension struct shape and DimensionType roster (Temporal/Categorical/Binary/Geo/Bucketed/Metadata)
- [18-measure](18_entities/measure.md) — Measure struct shape, AggregationType roster, and AdditivityType classification
- [18-metric](18_entities/metric.md) — Metric struct shape and cross-DataKind reference resolution
- [18-filter-taxonomy](18_entities/filter-taxonomy.md) — Two distinct filter types — DataKindFilter (DataKind-level predicate) vs AggregationFilter (Measure/Metric-level conditional aggregation)
- [18-ai-context](18_entities/ai-context.md) — AiContext struct — LLM/agent-facing hint surface (synonyms, description, examples, unit)
- [18-keys](18_entities/keys.md) — Keys struct — primary, unique, and foreign key declarations with bare Semantic names
- [18-semantic-mapping-value](18_entities/semantic-mapping-value.md) — SemanticMappingValue enum — four variants (Column/Literal/Expr/Metadata) for binding Semantics to physical sources
- [18-structural-rules](18_entities/structural-rules.md) — Entity-level structural rules (SR-E-*) — numbered validation invariants SR-E-1 through SR-E-14
