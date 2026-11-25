# semstrait-planner

The planner's public contract: `Planner::plan(&self, &SemanticManifest, &Request,
&RequestContext)`, which returns `Result<(SemanticPlan, Diagnostics<Advisory>),
(Diagnostic<PlanError>, Diagnostics<Advisory>)>`, ADR 0005's fail-fast form with
the advisories in their own kind beside the plan (ADR 0064, *ADR 0018 is
untouched*). It is the call that semstrait-api's `Semstrait` makes in `validate`,
`explain` and `plan`.

The forms that it plans, each with the ADRs that rule it:

| Form | ADR |
|---|---|
| A Dataset: a read of one catalog table, or of one path source's files | ADR 0070; ADR 0073, *The physical read below a Dataset* |
| A Joinset of Datasets, Unionsets or Grainsets, or a left-deep chain of Joinsets: the join or the cross of its operands | ADR 0054; ADR 0066, ruling 4; ADR 0074, *A Joinset* |
| A Joinset that reads one operand alone, or a sub-composition that stands in for it at the scope root | ADR 0064, *Decision Outcome*; ADR 0074, **Leaving out an operand** |
| A Unionset (`all` or `unique`) of Datasets, Joinsets or Grainsets: the stack of the branches that it reads, with typed NULLs for a name that a branch does not map | ADR 0073, *A Unionset prunes a branch its predicate excludes*; ADR 0074, *A Unionset*; ADR 0077 |
| A Unionset branch gives rows, partial aggregates that the union aggregates again, or final aggregates | ADR 0074, *A Unionset*; ADR 0079; ADR 0089, rule 21 |
| A Grainset, through one routing rule: one child answers alone, or a drill-across joins the chosen children's final aggregates; a Metric that no child maps whole, in a Compute frame that reads one frame for each formula group and computes the Metric over their joined values | ADR 0074, *A Grainset*; ADR 0084, rules 1 to 8 |
| A composition as a constituent of another, at every depth | ADR 0074, **A composition that is a constituent** |
| Each operand's delivery: rows, partial or final aggregates, which its parent's demand fixes | ADR 0080, *The parent's demand fixes an operand's delivery* |
| Dimensions, at a requested granularity or not, bucketed on the request context's timezone and week start | ADR 0063; ADR 0072; ADR 0080, *Only a frame that gives final aggregates buckets* |
| Measures, with each `filter:`, and a Unionset's own filter for a Measure | ADR 0071, *Aggregation-level conditions*; ADR 0080, *Every filter reads the value before bucketing*; ADR 0084, rule 9 |
| A semi-additive Measure, through its `rank` marker | ADR 0076; ADR 0083, rule 3 |
| An aggregate whose rows a join repeats, through a `row_number` marker | ADR 0074, **Fan-out**; ADR 0076; ADR 0077 |
| A row-level Metric, as a Dimension or as a Measure | ADR 0074, *A Metric*; ADR 0083, rules 1 and 4 |
| A Metric over the final aggregates, in the formula push-down's pattern, by its parts in two aggregate steps, or with an `agg:` over its formula's values; one that a Grainset exposes, inside one child | ADR 0074, *A Metric*; ADR 0084, rule 8 |
| A link to a Metric whose `agg:` aggregates a formula, as a reserved field of that Metric's subtree | ADR 0088, rule 15 |
| Request predicates, each placed by its role, and activated named filters; a Grainset owns a signature only for a Metric that no child maps whole and that its children can compute together, so its frame passes an activated named filter unchanged to the child that answers alone, which reads its own signature for it | ADR 0071; ADR 0080, *A named filter is placed as a request predicate*; ADR 0084, rule 1; ADR 0096, *A named filter on a Grainset* |
| Order, then offset and limit, at the scope root | ADR 0062; ADR 0073, *The build order* |

`plan` runs, in order:

1. **bind** each name that the request reads to its bit, and refuse a malformed
   request in one fixed order (ADR 0005; ADR 0062);
2. **resolve** the scope root, under `from` or by field-first resolution, and a
   Joinset root's alternatives (ADR 0064; ADR 0074, *Every request is planned
   against its grain first*; ADR 0084, rule 1);
3. **decide** each frame that a decision reads: each delivery and each request
   predicate's place, checked per body (ADR 0073, *Two passes*, *One decision
   shape, checked per body*; ADR 0080);
4. **build** each frame bottom-up in the stage order, each node through its
   public constructor and each name as its parent reads it, and number the
   anchors once (ADR 0073, *The build order*, *Details fixed with the seam*;
   ADR 0075);
5. **validate** the plan that it built: one that `SemanticPlan::validate` refuses
   is `PlanRefused`, with `validate`'s `IrErrorKind` and its note, which names the
   node, so every plan that it returns is one that the format adapter takes (ADR
   0068, *The seam*).

Each refusal is a `PlanError` variant, and each advisory an `Advisory` variant
beside the plan (`error.rs`, `advisory.rs`).

A caller replaces the Dataset Strategy through `Planner::with_dataset_strategy`, the
Joinset Strategy through `Planner::with_joinset_strategy`, the Unionset Strategy
through `Planner::with_unionset_strategy`, and the Grainset Strategy through
`Planner::with_grainset_strategy`.

Terms: `CONTEXT.md` (**Request**, **Selection**,
**Request predicate**, **Requested granularity**, **Request context**, **Scope
root**, **Field-first resolution**, **Candidate**, **Structural complexity**,
**Grain**, **Time axis**, **Guaranteed keyset**, **Granularity fit**,
**Re-aggregation**, **Additivity**, **Unionset**, **Grainset**, **Operand**, **Frame**,
**Demand**, **Delivery**, **Strategy**, **Combine**, **Bypass**, **Pruned**, **Physical
read**, **Marker**, **Advisory**).

| Module | Holds |
|---|---|
| `request` | `Request`, `Selection`, `DimensionModifier`, `OrderKey`, `Direction` |
| `context` | `RequestContext`, `Timezone` (checked against `chrono-tz`'s compiled-in IANA table), `UnknownTimezone`, `Weekday` |
| `planner` | `Planner`, one Strategy for each source role |
| `binding` | `BoundName`, and request binding: each name to its bit, and the request-shape refusals |
| `predicate` | `Predicate`, `Placement`, the typing of the request predicate, where it or each copy of it goes at a join, and a named filter's copies on a composition read back as one request predicate |
| `resolver` | the scope root, under `from` or by field-first resolution, its alternatives, and structural complexity's two terms |
| `grain` | the grain test of a request against a source, granularity fit, a request's grain at the granularity that a drill-across fixes, the re-aggregation verdict, and an aggregate's effective additivity |
| `read_model` | `ReadModel`, sealed, `TimeAxis`, `PhysicalSource`, `Relationship`, and the one implementation over a walk of the manifest |
| `strategy` | `Strategy`, unsealed, its one input `StrategyInput`, with the `FormulaNode`s that a stack above reads and the `part_key` that a parts Metric's step 1 groups by, which each operand computes on its own rows, `DatasetArm`, `UnionsetArm` with the fold that prunes a branch (whether a predicate on the Unionset's rows excludes it, which the Unionset Strategy and the dispatch site's check read), `GrainsetArm`, and the dispatch site |
| `dataset_strategy` | the Dataset Strategy and the physical read that it creates |
| `joinset_strategy` | the Joinset Strategy, which answers through its first stand-in that may answer, else routes each name to the operands that serve it and reads one alone where it may prune the other |
| `unionset_strategy` | the Unionset Strategy |
| `grainset_strategy` | the Grainset Strategy, which answers through the child that its routing gives, else drills across the chosen children |
| `grainset_metrics` | the Metrics that a Grainset computes: whether it owns a name's signature, the names that a signature's tree reads and links, and a demand's formula groups (the base group, a parts Metric's group at its part key, and an own-`agg:` Metric's group at its formula grain), with each group's condition and the Metrics computed after the groups' joins |
| `routing` | a Grainset's routing, the one rule that the resolver and the Grainset Strategy call: the rank-first child of each name, which answers alone where one child is first for every name, else the chosen children of a drill-across in the Child rank's order, and the granularity that the drill-across fixes |
| `frame` | the frame tree, with the Compute frame, which holds one frame for each formula group, and the decide pass that fixes each operand's demand |
| `error` | `PlanError`, `LookedIn`, `UnbuildableState`: typed, `#[non_exhaustive]` kinds (ADR 0004; ADR 0007) |
| `advisory` | `Advisory`, and `advise`, which gives each advisory once |
| `frame_path` | `FramePath` |
| `decision` | `Decision`, `Operand`, `PhysicalRead`, `DecisionValue`, `Combine`, and the per-body table that checks a decision |
| `build` | the build pass: each frame in ADR 0073's stage order, its buckets, its markers, its names and its trace, and the re-aggregation of partial aggregates; `build::stack`, the stack combine, each branch's row projection with its NULL fill and the cast of a row argument to the branches' least-restrictive type (ADR 0089, rule 19), and its partial aggregate step, or a disjoint branch's final one, or the own aggregate step of a branch that combines operands, or a drill-across; `build::drill_across`, the drill-across combine and the projections of its names, its NULL fill and the formula nodes that its children give; `build::computed`, the Compute frame: each formula group's subtree at its `<grainset>#group<n>` path with that group's formula step, the `Full` joins of the groups, each Metric computed after the joins, and the request predicates above them; `build::volatile`, `Advisory::VolatileInTwoReads` and `Advisory::VolatileCopied` over a frame's stages |
| `semi_additive` | `RankMarker`: where a semi-additive Measure's strategy applies, its default `by`, the Dataset in which it reads its names on a Joinset, and its `rank` window |
| `fan_out` | `RowNumberMarker`, and `FanOutWalk`, which gives a frame's markers: the walk of a value leaf down to the unit that reads it, the operand that a join repeats, the row identity, the `row_number` window, `FanOutMarkerNotPlaceable` (rules 1 to 3) and `Advisory::FanOut` |
| `formula` | where a Metric's formula is applied over a frame's aggregate step: the role each definition is planned as, the grouping that a formula reads as fields, the push-down, the formula grain and a Metric's two aggregate steps |
| `parts` | the part key of a Metric's formula parts, and the formula on each part |
| `tree` | the reads and rewrites of a tree that the planner shares: its column leaves, its aggregate calls, and the tree with its column leaves replaced |
| `steps` | `step_scan`, `step_empty`, `step_join`, `step_cross`, `step_union`, `step_project`, `step_filter`, `step_agg`, `step_distinct`, `step_sort` and `step_fetch`, each one plan node, and the names and catalog calls that the planner writes |

It depends on `semstrait-common` (`Diagnostic`, `Diagnose`, `EntityId`, `Tree`,
`ValidateError`, `Granularity`, the name types), `semstrait-sql` (the predicate's tree, `DataType`, `Decimal`, `TypeFamily`,
`TypeError`, `promote` and `CanonicalTypeSystem`, the aggregate call, the function catalog's `date_trunc` and `sum` calls,
the `rank` and `row_number` window calls, a function's `Volatility` and an aggregate's `Additivity`),
`semstrait-ir` (`SemanticPlan`, `PhysicalExpr`, the plan nodes, `MultisetOp`, their
`IrErrorKind`, `EmitForm`, `SemAnnotation`, `SourceRole`, `SemanticRole`, `GranularityGroup`
and `Name`, a join's `JoinType`, `Cardinality`, `JoinsetJoinType` and `Integrity`, the
physical read's `Reader`, `TableScan`, `FilesScan`, `VirtualTable` and `Schema`, a source's
`TemporalType`, `Delivery`, a Measure's `AdditivityType` and `SemiAdditivityStrategy`),
`semstrait-manifest` (`SemanticManifest`,
the values the read model hands out, and
`join_filter_onto_aggregates`), `chrono-tz` and
`thiserror`, and imports each type from the crate that owns it (ADR 0090, rule 6).
It links no engine and no adapter, its tests included (ADR 0068, *The seam*); its
tests compile a small `semstrait-model` model as setup, over a catalog double (`semstrait-catalog`,
`async-trait`), run async on `tokio` since `compile` is (ADR 0003), assert each
planned request's `SemanticPlan`, and compare each
refusal's message with a golden that only `tests/common/golden.rs`'s generator
writes: `SEMSTRAIT_GOLDENS=1 cargo test -p semstrait-planner generate_goldens`.
