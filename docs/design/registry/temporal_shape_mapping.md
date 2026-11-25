---
prereqs: [00]
status: superseded
---

# temporal_shape_mapping (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`temporal_shape_mapping/`](temporal_shape_mapping/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [registry-temporal-shape-mapping-intro-simple](temporal_shape_mapping/intro-simple-shapes.md) — Temporal-shape mapping intro, scope, engine roster, plus Timeseries and Events (simple shapes) — universal per-engine emission via date_trunc + GROUP BY
- [registry-temporal-shape-mapping-snapshot](temporal_shape_mapping/snapshot.md) — Snapshot shape emission — WHERE-max canonical fallback, native table-level time-travel (Spark Delta/Iceberg VERSION AS OF, Snowflake AT/BEFORE, BigQuery FOR SYSTEM_TIME AS OF), QUALIFY-based latest-at-or-before, cadence-rollup reducer policies
- [registry-temporal-shape-mapping-scd](temporal_shape_mapping/scd.md) — Scd (Slowly-Changing Dimension) Type0–Type6 emission — valid-window half-open predicate (Type2/5/6), per-subtype mapping (Type0 append-only, Type1 MERGE, Type2 valid_from/valid_to, Type3 prior-value column, Type4/5/6 DEFERRED), sentinel-vs-NULL conventions, default-current lookup
- [registry-temporal-shape-mapping-asof](temporal_shape_mapping/asof-idioms-gaps.md) — AsOf anchor emission (ScdWindow / SnapshotLatestAtOrBefore), rewrite-tier table (First-class / Structural / Unsupported per anchor×engine), cross-engine idioms (windowed-ASOF QUALIFY/subquery, half-open predicate, default-current), per-engine gap catalog, TD index, versioning, interaction
