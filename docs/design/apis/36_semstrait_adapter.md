---
prereqs: [00]
status: superseded
---

# 36_semstrait_adapter (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`36_semstrait_adapter/`](36_semstrait_adapter/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [36-adapter-crate-charter](36_semstrait_adapter/adapter-crate-charter.md) — semstrait-adapter crate charter, module layout, purpose/scope/layering, and EngineAdapter trait (signature, invariants, AdapterId, adapt/emit verbs)
- [36-dialect-emit](36_semstrait_adapter/dialect-emit.md) — DialectEmit operational trait — per-dialect SQL rendering (identifier quoting, literal escaping, DataType spelling, function rewriting, cast semantics, null-ordering/AsOf/date_trunc/regex methods)
- [36-adapter-roster](36_semstrait_adapter/adapter-roster.md) — v1 built-in adapter roster (AnsiSqlAdapter/DataFusionSqlAdapter/DuckDbSqlAdapter/SparkSqlAdapter/SubstraitAdapter) and AdapterCapabilities surface
- [36-rewrite-pipelines](36_semstrait_adapter/rewrite-pipelines.md) — Function-rewrite pipeline (inputs, tiers, pipeline flow, aggregates, registry-extension interplay) and type-rewrite pipeline (inputs, spelling, cast semantics, type-rewrite exceptions)
- [36-emit-orchestrators](36_semstrait_adapter/emit-orchestrators.md) — adapt/emit pipeline orchestrators (SQL-emission/Substrait-emission), per-variant rendering contracts, error propagation, AdaptErrorKind taxonomy, AdapterRegistry, crate boundaries, sandboxing/safety, and public API surface
