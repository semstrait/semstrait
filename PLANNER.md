# Planner Module

Transforms a `QueryRequest` into a logical `PlanNode` tree that can be emitted
as Substrait or SQL.

## Module Map

```
planner/
├── plan.rs      Router — classifies query, dispatches to builders
├── table.rs     Single-dataset plans (Scan → Join → Filter → Aggregate → Project → Sort)
├── cross.rs     Cross-datasetGroup metrics (UNION branches → re-aggregate)
├── union.rs     Conformed / qualified / partitioned / virtual-only UNIONs
├── join.rs      Multi-table JOIN within one datasetGroup
├── expr.rs      Semantic model expressions → plan expressions
├── util.rs      Shared helpers (column builders, dimension parsing, virtual values)
└── error.rs     PlanError
```

## Decision Tree

`plan_semantic_query` (in `plan.rs`) is the single entry point. It inspects the
query and routes to exactly one planning path:

```
plan_semantic_query
│
├─ cross-datasetGroup metric (1)?
│  └─► cross::plan_cross_dataset_group_query
│
├─ cross-datasetGroup metrics (>1)?
│  └─► cross::plan_multi_cross_dataset_group_query
│
├─ qualified groups > 1?  (e.g. "adwords.dates.year" + "facebookads.dates.year")
│  └─► union::plan_multi_tablegroup_query          UNION with NULL projection
│
├─ qualified group == 1?  (e.g. "adwords.dates.year")
│  └─► union::plan_single_tablegroup_query          constrain to that group
│
└─ unqualified (normal path)
   │
   ├─ select_datasets OK
   │  ├─ partitioned (multiple datasets with partition)?
   │  │  └─► union::plan_partitioned_union           UNION ALL per partition
   │  ├─ conformed dimensions + multiple groups?
   │  │  └─► union::plan_conformed_query             UNION across groups
   │  └─ single dataset
   │     └─► resolve → table::plan_query             standard path
   │
   └─ select_datasets FAIL
      ├─ conformed + multiple groups?
      │  └─► union::plan_conformed_query
      └─ try multi-table JOIN
         ├─ only 1 table needed after all?
         │  └─► resolve → table::plan_query
         └─ multiple tables
            └─► join::plan_same_tablegroup_join      FULL OUTER JOIN
```

## Plan Shapes

Each path produces a different plan tree. The leaves are always `Scan` nodes;
the roots are usually `Sort` or `Project`.

**Standard (single table)**
```
Sort
└── Project (dimensions + metrics)
    └── Aggregate (GROUP BY dimensions, agg measures)
        └── Join* (LEFT JOIN dimension tables)
            └── Scan (fact table)
```

**Cross-datasetGroup metric**
```
Sort
└── Aggregate (re-aggregate: SUM metric by dimensions)
    └── Union
        ├── Project (dims + metric, NULLs for other groups)
        │   └── Aggregate → Join → Scan   [group A]
        └── Project
            └── Aggregate → Join → Scan   [group B]
```

**Conformed dimension UNION**
```
Union
├── Sort → Project → Aggregate → Join → Scan   [group A]
└── Sort → Project → Aggregate → Join → Scan   [group B]
```

**Partitioned UNION ALL**
```
Union
├── Sort → Project → Aggregate → Join → Scan   [partition 1]
└── Sort → Project → Aggregate → Join → Scan   [partition 2]
```

**Multi-table JOIN (same group)**
```
Sort
└── Project (COALESCE dims, pick measures from owning table)
    └── Join (FULL OUTER on dimensions)
        ├── Project → Aggregate → Scan   [table 1]
        └── Project → Aggregate → Scan   [table 2]
```

**Virtual-only (no table scan)**
```
VirtualTable (literal rows, one per group/partition)
```

## Key Concepts

| Term | Meaning |
|------|---------|
| **DatasetGroup** | A logical group of related datasets sharing dimensions and measures |
| **Conformed dimension** | A dimension defined at the model level, queryable across all groups |
| **Qualified dimension** | A 3-part path like `adwords.dates.year` scoping a dimension to one group |
| **Cross-datasetGroup metric** | A metric referencing measures from different groups (e.g. `adwords.cost + facebook.spend`) |
| **Partitioned dataset** | A dataset split into physical partitions, each served by a separate Scan |
| **Degenerate dimension** | A dimension whose attributes live directly on the fact table (no JOIN) |
| **Virtual dimension** | `_dataset.*` metadata attributes projected as literals, not columns |

## Module Responsibilities

**`plan.rs`** — Query classification and dispatch. No plan-building logic of its
own; purely a router.

**`table.rs`** — The workhorse. `plan_query` handles the resolver-based path
(single resolved dataset). `build_tablegroup_branch` is the unified builder used
by `cross.rs` when it needs a per-group aggregate sub-plan.

**`cross.rs`** — Builds UNION plans for metrics that span multiple datasetGroups.
Each branch aggregates its own group, projects to a common schema with NULLs for
missing columns, then a final re-aggregation (SUM) combines everything.

**`union.rs`** — Handles four UNION-flavored scenarios: conformed dimensions,
multi-group qualified dimensions, partitioned datasets, and virtual-only queries.

**`join.rs`** — When measures are spread across multiple tables in the same group,
builds sub-queries per table and joins them with FULL OUTER JOIN + COALESCE on
dimension columns.

**`expr.rs`** — Pure conversion from semantic model expression trees (`MeasureExpr`,
`MetricExpr`, `ConditionExpr`) to plan `Expr` nodes. Also handles filter and
JSON literal conversion.

**`util.rs`** — Reusable helpers: `needs_join_for_dimension`, `build_column`,
`ParsedDimensionAttr`, virtual attribute value resolution, column collection for
scan schemas.
