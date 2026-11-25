---
number: 68
title: Adapters are layered format-first
status: accepted
date: 2026-09-23
source:
- CONTEXT.md
- docs/adr/0003-compile-owns-resolution-and-io.md
- docs/adr/0006-tracing-only-observability.md
- docs/adr/0008-unsealed-extension-traits.md
- docs/adr/0009-per-engine-crates-lockstep-releases.md
- docs/adr/0018-canonical-plan-ir-contract.md
- docs/adr/0019-substrait-outbound-only.md
- docs/adr/0022-engine-independent-canonical-vocabulary.md
- docs/adr/0023-adapter-emission-architecture.md
- docs/adr/0024-pure-request-and-stateless-orchestrator.md
- docs/adr/0044-no-gate-roster-validity-definitions-belong-to-stages.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0067-the-scan-carries-its-read-identity.md
- Substrait `proto/substrait/algebra.proto` at tags v0.85.0 and v0.103.1
- crates `substrait` 0.63.0, `datafusion-substrait` 55.1 (consumer), `substrait-explain` 0.9
- 'evidence: gap matrix, version triple; DataFusion consumer; round-trip prototype; aggregation phase'
- sharpened in place (the engine adapter's own crate)
- sharpened in place (the union-branch rule of the DataFusion rules)
- sharpened in place (the DataFusion rules are built for one release, and DataFusion 53's consumer)
links:
- target: 3
  kind: amends
- target: 8
  kind: amends
- target: 9
  kind: amends
- target: 22
  kind: amends
- target: 23
  kind: amends
- target: 44
  kind: amends
- target: 72
  kind: amendedby
- target: 73
  kind: amendedby
- target: 18
  kind: supersedes
- target: 86
  kind: amendedby
- target: 91
  kind: amendedby
- target: 92
  kind: amendedby
---

# Adapters are layered format-first

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0072](0072-a-requested-grain-buckets-with-one-canonical-date-trunc.md): a rewrite stays in the canonical vocabulary, and no rule rewrites a node the replacement creates. The format-detail hook for a canonical function widens: besides the Substrait name and URI, it may return the whole call, written in the engine's own functions around the arguments the producer has already lowered. The producer writes it as returned and offers it to no rule. So the walk still ends by construction, and the plan holds only canonical names: engine identity stays in the rules.
> - [ADR 0073](0073-a-strategy-only-decides-and-the-driver-builds.md): its "as it already is for `OptimizerPass`" no longer applies. The faithful default and the DataFusion Files rule write a `FilesScan` with no paths as one `uri_folder` item, the base path. That default is faithful only for a consumer that reads the `key=value` folders under it and supplies the key columns, since no Substrait field carries them.
> - [ADR 0086](0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md): in *The seam*, "a type's encoding", a format detail that a rule chooses, becomes: a type's Substrait type variation. A type's Substrait name comes from its Substrait declaration in `semstrait-sql` (ADR 0086, rule 6), and the format adapter writes each type's proto from its `DataType` variant. The rest of 0068 stands.
> - [ADR 0091](0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md): in its own *Amends* item on 0018, "engine identity's only home is an `impl SubstraitRules`" becomes: engine identity has two homes (ADR 0091, rule 7): its dialect's declarations, which say how the engine has each function and each operation that differs, and its `impl SubstraitRules`. The rest of 0068 stands.
> - [ADR 0092](0092-an-operation-node-has-a-declaration-and-checks-its-operands-by-type-family-and-a-type-system-derives-the-result-types.md): in *The seam*, the clause from "Every dedicated expression variant Substrait models as a plain function call" to "`coalesce` included" becomes: every operation node that Substrait models as a plain function call — `BinaryOp`, `UnaryOp`, `Like`, `Between` and `IsNull` — lowers as a `ScalarFunction` under the name the rules' function hook returns, and a negated `Like` or `Between`, and a `Between`, lower as ADR 0092 rule 9 writes them. `coalesce` and `nullif` are function calls (ADR 0092, rule 6). The rest of 0068 stands.

## Context and Problem Statement

Engines share formats, and ADRs 0003 and 0018 left open whether an
`EngineAdapter` is injected into `plan()`.

## Considered Options

- A format adapter constructed with one engine's rewrite rules (chosen).
- **A format adapter that takes the manifest**, or a `SourceBinding` derived
  from it. Refused for the reason ADR 0067 records: the adapter becomes a
  function of two inputs.
- **One crate per format and per engine** (`semstrait-substrait`,
  `semstrait-adapter-datafusion`). Chosen for the engines: each engine adapter
  is its own crate, and DataFusion's is `semstrait-adapter-datafusion`. Refused
  for the format: the crate graph grows for a single format, so the format
  adapter stays in `semstrait-adapter`.
- **The DataFusion rules in `semstrait-adapter::engines::datafusion`**, and the
  consumer and the engine tests in a test-only package. Refused: a rule and the
  rows that prove it would change in two packages. The engine tests need a
  crate in which `datafusion` is a dev-dependency, and the consumer stays out of
  `semstrait-adapter`; the engine adapter's own crate is that crate, so one
  crate holds the rules, the consumer and every engine test.
- **Inheritance-style hooks** — the walk as a provided method on the rules
  trait an engine may override. Refused: a hook that can reorder the traversal
  can emulate silently, which no rule may.
- **Text-first production** — emitting `substrait-explain` text and parsing it
  to protobuf, as a prototype did. Refused: the text grammar becomes a
  de-facto wire contract and parsing a production failure mode.
- **A `ConsumerProfile` parameter** beside the rules. Refused: the injected
  `impl SubstraitRules` *is* the profile.
- **Engine constraints baked into the portable defaults.** Refused: it hides an
  engine inside the format; a faithful default plus engine deviations in the
  engine's impl is the layering.
- **A refusal tier at adapt**, after ADR 0022 — a typed `Unsupported*` for a
  form an engine cannot consume, with a warn-and-degrade tier beside it.
  Refused by the human: a plan the planner produced is not the adapter's to
  reject, a shortfall is a rewrite, and a degradation is either a
  semantics-preserving rewrite or a wrong answer. What no engine can take yet,
  `validate` refuses before adapt.
- **Linking `datafusion` in `semstrait-adapter`, behind a feature or
  otherwise.** Refused by ADR 0009's reason: feature-gated engines let type
  identity diverge across consumers; an engine binding is a crate.
- **Retiring the `filter` clause on `AggregateCall`** (ADR 0090, rule 2), or
  having the planner rewrite it to `CASE WHEN`. Refused: the clause is
  Substrait's `Measure.filter`, and a `CASE WHEN` form is compile's to hold in
  the manifest or the rules' to write for a target that lacks the filter.
- **Writing `Plan.version` as the crate's number** (`0.63.0`). Refused: the
  field names the spec release the plan follows, v0.85.0 for that crate.
- **Trace as an enhancement in `RelCommon.advanced_extension`** of the executed
  plan. Refused: an enhancement is the extension kind a consumer that does not
  know it must refuse, and trace is not executed; it rides as an optimization
  in the explain copy only.
- **Trace through `RelCommon.hint`.** Refused on the facts: `substrait-explain`
  does not print `Hint`.
- **Merging trace into the rendered text afterwards**, line by line. Refused:
  it parses `substrait-explain`'s output, a format semstrait does not own.
- **Registering a listing table under the physical source's name** and writing
  a files scan as `NamedTable([name])`. Refused: a model's files are read as
  files; the listing is compile's, and a registered listing table lists the
  storage again.
- **`NamedTable([minted name])` per file**, the adapter returning a name → URI
  list for the caller to register. Refused: the registration list is a second
  output beside the artifact, and the harness's consumer reads the URI itself.
- **The stock consumer's base-name registration.** Refused: two sources that
  share a file name collide.

## Decision Outcome

The adapt stage has two layers because engines share formats. A **format
adapter** lowers a `SemanticPlan` to one wire form — today the Substrait
**producer**, later a SQL emitter. Its inputs are the plan and the engine rules
it was constructed with, and nothing else: no manifest, no catalog, no engine
state (ADR 0067 puts into the plan the one thing the old adapter had to look
up). An **engine adapter** is the set of per-engine **rewrite rules** a format
adapter is parameterised by; it does not walk the plan, and in MVP it does not
execute it either. The format adapter lives in `semstrait-adapter`, rewritten
in place: the dead crate's `sql/` and `engines/` trees and its `traits.rs` are
deleted (they target `LogicalPlan`, `PlanArtifact` and a `SubstraitSerializer`
that do not exist), and the crate re-enters the workspace with
`SubstraitProducer`, `SubstraitRules` and `explain`. Each engine adapter is its
own crate, which depends on `semstrait-adapter`: DataFusion's is
`semstrait-adapter-datafusion`.

### The seam

`SubstraitProducer` owns the traversal: positional schema state down the tree
(a semstrait expression names a column; the producer turns the name into a
position against the input's `common.output_schema`), extension-declaration
minting (deterministic per plan), type mapping of the total output schemas,
`Plan.version`, and `RelRoot.names` from `SemanticPlan.output_names`. It is
constructed **with** an engine's rules — `SubstraitProducer::new(&dyn
SubstraitRules)`, dependency injection: the engine adapter reuses the format
adapter and injects itself into artifact production.

**Rewrite rules** are one contract with one trait per format — the unsealed
**`SubstraitRules`** today, a SQL counterpart when SQL emission returns — and
one implementation per engine that consumes the format. A rule does one of two
things. It **rewrites** a plan node or expression — in the canonical
vocabulary, before the producer lowers that subtree; the producer recurses into
the replacement, so a rewritten subtree receives ordinary lowering. Or it
chooses a per-variant **format detail**: the extension function name and URI
for a function's declaration (ADR 0091, rule 3), a type's encoding, how a
reader is written (ADR 0067). A rewrite that needs no format knowledge (an
aggregate filter folded into a `CASE`, below) and one that does (a post-join
filter lifted into a `FilterRel`) run in the same walk, under the same
contract.

**A rewrite is applied once per node.** The producer offers a node to the rules
once, lowers the replacement's root as returned, and recurses into the
replacement's children, each offered once in turn; a replacement's root is
never re-offered. A rule that needs a second pass writes the target shape in
one step: no rule rewrites a node the replacement creates, and the rest of the
replacement is the offered node's own inputs. So the walk terminates by
construction.

**Defaults are faithful.** Because each node mirrors a Substrait relation (ADR
0066 ruling 1), every node's default is its relation, 1:1, under the mapping
table of names: `ScanNode` → `ReadRel`, with `NamedTable { names }`,
`VirtualTable { expressions }`, or `LocalFiles` holding one `uri_file` item per
path of the listing, each with `file_format` set where Substrait has a slot for
the source's format (`Parquet`, `Orc`; `Csv` as `text` with the scan's
`format_options`) and unset otherwise; `JoinNode` → `JoinRel`, `Full` as
`OUTER`, `on` as `expression`, `post_join_filter` as itself; `CrossNode` →
`CrossRel`; `MultisetNode` → `SetRel` (`UNION_ALL`, `UNION_DISTINCT`);
`AggregateNode` → `AggregateRel`, each aggregate as a `Measure`, its
`filter` as `Measure.filter`, one grouping; `SortNode` → `SortRel`;
`FetchNode` → `FetchRel`, the bounds as integer literals in `offset_expr` /
`count_expr`; `NodeCommon` → `RelCommon`, below. Function spellings are
Substrait's official ones with their real extension URIs (`equal`, not `eq`), with no type variations. Every dedicated expression variant Substrait
models as a plain function call — `BinaryOp`, `UnaryOp`, `Like`, `Between`,
`IsNull`, `Coalesce`, `NullIf` (Substrait has no structural counterpart
for any of them) — lowers as a `ScalarFunction` under the name the rules'
function hook returns, `coalesce` included; the `Coalesce` a `Full` join's
composed Dimension signature carries (ADR 0065) needs nothing beyond that hook.
A rewrite hook's default is the **identity**. A method **added later** defaults
the same way — identity, or the faithful mapping — never to an error: ADR
0008's `MethodNotSupported` default is amended for rewrite rules below, because
an honest identity is not a lie.

**The adapter adapts; it does not refuse.** A plan the planner produced and
`validate` accepted is valid by construction — this seam adapts the planner's
output to a format and an engine; it is not a deserialisation boundary — and a
rules implementation's contract is to rewrite it into the shape its engine
consumes: for every form the planner emits, a semantics-preserving rewrite or
the faithful default. There is no `Unsupported*` error at adapt, no capability
gate, and no degraded emission behind a warning. The one typed error is
**`InvalidRewrite`**: a rules implementation returned a replacement that breaks
the rewrite contract (a subtree with a different schema, an expression of
another type) — ADR 0008's bound on an unverifiable third-party rewrite, a
defect in the rules, never a judgement on the plan. The obligation is scoped to
what the planner emits: a node admitted later (ADR 0066 ruling 2) brings its
rewrite obligation with it, and until then `validate` refuses it, as it refuses
`TopNNode`. The producer's output is the format artifact,
`substrait::proto::Plan`.

ADR 0023's reason for this shape holds a format at a time: hooks that cannot
reorder the walk keep every rewrite visible in one place and
semantics-preserving by contract — a hook that owned the traversal could
emulate silently. An engine's injection at **adapt** also answers the question
ADRs 0003 and 0018 left open — an `EngineAdapter` injected into `plan()`: no
engine reaches `plan`, ever.

### The release gap is the producer's

The plan mirrors Substrait v0.103.1 (ADR 0066); the producer writes the
release its linked `substrait` crate carries. Crate 0.63.0 — the one
`datafusion-substrait` 55.1 and `substrait-explain` 0.9 link — carries
**spec v0.85.0**, and `Plan.version` is written by
`substrait::version::version_with_producer("semstrait")`: the crate's spec
version, never the crate's own number. A field or relation the target release
lacks is dropped where its absence changes no result, and rewritten where it
would:

- `anchor` → `rel_anchor`, which v0.85 lacks (added in v0.89): dropped;
- `FetchNode` → `offset_expr` / `count_expr`, which both releases have;
- `VirtualTable.expressions`, which both releases have;
- `TopNNode`, which v0.85 lacks (added in v0.88): never reaches the producer,
  since `validate` refuses it (ADR 0066 ruling 4).

Moving the linked crate is a dependency bump the producer's version table
absorbs; the plan does not move.

DataFusion 53's consumer links crate 0.62.2, which carries **spec v0.77.0**.
From v0.77 to v0.85 the spec drops the extension URIs, which v0.77 already
deprecates, and adds lambdas, the `func` type and type aliases. The producer
writes none of them, and the consumer reads a function by its name only
(`datafusion-substrait@53.1.0:src/extensions.rs:81-113`), so DataFusion 53
reads the plan that the producer writes. Its Rust types are crate 0.62.2's, so
a caller hands it the plan as protobuf bytes.

### The node-common block on the wire, and trace

`NodeCommon` (ADR 0066 ruling 3) writes to `RelCommon`: `emit_form` →
`emit_kind`; `hint` → `hint`; `anchor` → `rel_anchor` where the release has
it. `output_schema` feeds the producer's positions and types and is not
written. `semantic_meta` is **never in the executed artifact**: no consumer
executes trace, and the format artifact carries no semantic vocabulary.

**Trace reaches only the explain copy.** `explain` produces the plan a second
time, an explain-only copy in which each relation carries its node's
`semantic_meta` as an **optimization** on the relation's own
`advanced_extension` — the extension kind a consumer may ignore — and renders
that copy with `substrait-explain`'s `format_with_registry`, given semstrait's
extension registry, which prints it as `+ Opt:` addenda under the relation. One
call produces the print format with the trace beside each
relation: which dataset is behind a `NamedTable`, which strategy boundary a
subtree starts. It is ADR 0006's tracing made visible, and it touches no
executed plan.

`substrait-explain` is a normal dependency: `explain` is product surface, not a
test tool. The producer builds `substrait::proto::Plan` directly; the text
grammar is never a production path. Constraint recorded from the prototype: when linked
beside `datafusion-substrait`, `substrait-explain` must enable its `serde`
feature or the `Any` carriers disagree at compile time.

**`datafusion` is never a dependency of `semstrait-adapter`**, a dev-dependency
included (ADR 0009's rule, kept). `datafusion` and `datafusion-substrait` 55 are
dev-dependencies of `semstrait-adapter-datafusion` only, so its library links
no engine, and every test that runs a plan on DataFusion is in that crate's
tests: consuming and executing a produced plan against an engine is the e2e
oracle's, and the oracle runs there. When the API wants in-process execution,
the execution binding ships as **its own crate**, which links the engine, per
ADR 0009, never as a feature; it is out of scope. The engine adapter's MVP job
is rules, and its crate exists so the oracle instantiates a named thing rather
than the default by omission.

### The DataFusion rules

`DataFusionRules`, in `semstrait-adapter-datafusion`, is an
`impl SubstraitRules` built for one DataFusion release (ADR 0091, rule 6).
It rewrites where that release's Substrait consumer differs, each rewrite
cited to the consumer's source. A rule that holds on every release from
the oldest is global, and is cited at the release that `Cargo.lock` pins;
a rule for some releases names them and cites each release's source. Each
rule below is global, and the suite runs it on DataFusion 55 and 53:

- **Post-join filter.** `JoinNode { post_join_filter: Some(f) }` →
  `FilterNode(f) ← JoinNode { post_join_filter: None }`: the same rows, since
  the filter applies to the joined output; the consumer refuses the field
  (`join_rel.rs:32-33`). The inner join has no filter, so the rule does not
  match it again.
- **Tables.** The consumer reads three names as catalog, schema and table
  (`read_rel.rs:89-104`). A table with no catalog — `""` first (ADR 0067) — is
  written as the two names `[namespace, name]`, which the consumer resolves in
  the session's default catalog.
- **Files.** The consumer accepts a `LocalFiles` of exactly one `uri_file` item
  (`read_rel.rs:207-247`). A `FilesScan` of *n* paths is written as *n*
  `ReadRel`s, each a `LocalFiles` of one `uri_file` item with its
  `file_format`, each carrying the scan's `schema`, `filter` and emit form,
  under one `SetRel(UNION_ALL)`; one path needs no union. A glob or folder read
  (ADR 0067) is written only for a consumer that reads one, which this one
  does not.
- **Union branches.** Cited at 55.2.0, the version that `Cargo.lock` pins: the
  consumer builds a union branch by branch (`set_rel.rs:52-70`), and the union
  takes its field names from the first branch (`datafusion-expr`
  `logical_plan/plan.rs:3339-3364`). DataFusion's
  analyzer gives a later branch the union's names only where that branch is a
  projection (`datafusion-optimizer` `analyzer/type_coercion.rs:289-316`), and
  the consumer reads a relation with an emit form as a projection
  (`rel/mod.rs:82-131`). Where the optimizer leaves one branch, it reads that
  branch by the union's names (`datafusion-optimizer`
  `propagate_empty_relation.rs:210-223`), so a later branch with other names
  fails. So each later branch of a `MultisetNode` that is no `ProjectNode` and
  has no emit form goes under an identity `ProjectNode`, which has no
  expressions: the same rows, and the analyzer gives the projection the
  union's names. A files read's `SetRel` needs no rule: the `reader` hook
  writes it as it lowers a scan, so no rule is offered it, and each of its
  reads carries the scan's one schema.
- **No rule needed.** A table with a catalog is read by its three names;
  `CrossRel` is consumed; `AggregateRel.Measure.filter` is consumed
  (`aggregate_rel.rs:77`); the emit form is applied after every relation
  (`rel/mod.rs:79`); `hint` and `advanced_extension` are ignored.

**The consumer that runs a files read is the caller's.** The stock consumer
resolves a `uri_file` by the file's base name against the tables registered in
its session, and fails with "No table named" when none is (`read_rel.rs:60-62`):
it does not read a file it is told to read. The DataFusion e2e harness, in
`semstrait-adapter-datafusion`'s tests, therefore carries its own consumer,
which overrides `SubstraitConsumer::consume_read` (`substrait_consumer.rs:218`):
it reads each `LocalFiles` item by its full URI and its format, then applies the
base schema, the filter and the emit form as the stock path does. A model's
files then need no registration. A model's tables are `TableScan::NamedTable`,
and whoever executes registers them — the caller that names a table in the
model provides it. The consumer is not in `semstrait-adapter`, which links no
engine.

### The aggregate filter is a mirror, and `CASE WHEN` is a rewrite

An `AggregateNode` aggregate mirrors Substrait's
`AggregateRel.Measure { measure, filter }`: an aggregate function and an
optional filter (ADR 0066 ruling 8). The manifest holds a filtered aggregation
in one of two shapes, and the plan accepts both:

- **Lowered.** The filter is folded into the aggregate's argument, and the
  call's `filter` clause is `None` (ADR 0090, rule 2). The adapter maps the
  aggregate as any other.
- **Carried.** The filter is in the call's `filter` clause (ADR 0090, rule 2).
  The faithful default maps it to `Measure.filter`, 1:1. An engine whose
  target lacks a per-aggregate filter — a SQL dialect without `FILTER (WHERE
  …)` — has its rules rewrite it per aggregate kind: the common
  `Aggregate(filter, arg)` → `Aggregate(CASE WHEN filter THEN arg ELSE NULL END)`
  is the model example of an expression rewrite that needs no format
  knowledge, valid where an injected `NULL` is equivalent to excluding the row
  and not elsewhere — conditional on the target and the kind, never universal,
  and never the planner's.

Which shape the manifest holds is compile's, and a lowering is valid on the
same terms as the rewrite. Today it is **ADR 0065's**, and carried: a
Measure's `filter:` is a second, row-level root on its own signature
(`filter_expr_id`), and a Metric that inlines a filtered Measure has that
predicate spliced onto the inlined `Aggregate.filter` node at substitution.
Either origin — the signature's second root the planner places, or the splice
on the tree — reaches the plan as the call's `filter` clause (ADR 0090, rule
2). This ADR rules the adapter only.

### Artifacts

`semstrait-ir::artifact` — `EngineArtifact`, `EnginePlan`, `SqlArtifact`,
`DialectId`, `Dialect`, `Capability` — is deleted and `semstrait-ir` drops its
`substrait` dependency: the canonical crate links no wire-format crate for an
envelope nothing fills. The producer's output type is `substrait::proto::Plan`;
a cross-format sum type returns when a second format exists. The glossary's
*Engine artifact* is replaced by **Format artifact**: no semantic vocabulary
reaches it, and a format's reuse of a role name (`AggregateRel.Measure` is an
aggregate computation) names compute, never the role.

### Amends

- **0008:** in its roster, `EngineAdapter` and `DialectEmit` are replaced by the
  per-format rewrite-rules traits, `SubstraitRules` today and a SQL counterpart
  when SQL emission returns. The unsealing promise transfers whole. Its
  defaults-may-not-lie rule is met differently for these traits: a rewrite
  hook's default is the identity and a mapping hook's default is the faithful
  format answer, and a MINOR-added method defaults the same way rather than to a
  `MethodNotSupported` error — an honest identity is not a lie, and a refusing
  default would be the refusal that 0068's *The seam* removes. `InvalidRewrite`
  is these traits' bound on an unverifiable third-party rewrite, as it already
  is for `OptimizerPass`.
- **0009:** its rule is about engine *linkage*, and that rule stands: no engine
  dependency and no `cfg(feature)` engine gating in `semstrait-adapter`, and an
  engine adapter that links its engine is its own crate. Amended in one clause:
  in *Decision Outcome*, "Each engine adapter … is its own independently-tiered
  crate" reads: an engine adapter is its own crate that depends on
  `semstrait-adapter`. Its library links no engine, and its engine is a
  dev-dependency of its tests only: the tests that run a plan on the engine need
  a crate that takes the engine as a dev-dependency, and the consumer stays out
  of `semstrait-adapter` (0068's *Decision Outcome*).
- **0018:** its artifact clause named `SqlArtifact.dialect`, `Dialect::ID` and
  `Capability` as the only homes of engine identity; `semstrait-ir::artifact`
  (`EngineArtifact`, `EnginePlan`, `SqlArtifact`, `DialectId`, `Dialect`,
  `Capability`) is deleted (0068's *Artifacts*), and engine identity's only home
  is an `impl SubstraitRules`, which lives in the engine adapter's own crate and
  never in `semstrait-adapter`. The rule that no canonical crate branches on
  engine identity is unchanged. Its open question, "injecting an `EngineAdapter`
  into `plan()`" versus "no engine influence before `adapt()`", is resolved for
  the second: the engine enters as rules at adapt and never at plan, and no
  engine reaches `plan`.
- **0022:** its decision "adapters rewrite, warn, or refuse" is replaced by the
  paragraph "The adapter adapts; it does not refuse" of 0068's *The seam*, and
  by 0068's *Artifacts*, and the sections that only apply it read as those give:
  its title's "adapters rewrite, warn, or refuse"; in *Context and Problem
  Statement*, "capability variants become public contract" and the bullets
  "adapter rewrite strategies" and "`adapt` re-derives feasibility"; in
  *Considered Options*, the chosen option's "with a rewrite-warn-refuse ladder"
  and the options "Best-effort degraded emission or emulation" and "A
  descriptive per-dialect capability matrix"; and in *Decision Outcome*, the
  sentence "An engine shortfall is answered in strict order" and the
  `Capability` bullet: the `Capability` *type* is retired with no reader (the
  planner never reads engine capability). What stands, and is
  0068's mechanism: the canonical vocabulary is fixed independently of engine
  capability, an engine never widens it, and there is no silent emulation — a
  rewrite is a visible rule with a cited reason. The rest of 0022 stands.
- **0023:** its shape (a central orchestrator owns the walk; per-variant hooks
  contribute rendering only; adapt is pure, sync, byte-deterministic and touches
  no catalog and no running engine — an engine's *rules* are an input to adapt,
  an engine's runtime never is) is generalised from SQL to every format and is
  the seam in 0068's *The seam*, with one widening: a hook may rewrite a subtree
  before lowering, not only render it; `SubstraitProducer` owns the Substrait
  walk. Its SQL-specific clauses — the `emit_sql`/`emit_substrait` pair, the
  `Dialect`/`DialectEmit` split across two crates, dialect-blind `Expr→Expr`
  rewrites, mandatory quoting — describe the SQL format adapter, which is out of
  scope with SQL emission, and are dormant, not repealed. `semstrait-ir`'s
  `Dialect`, `DialectId` and `Capability` types are deleted, and `EngineAdapter`
  and `DialectEmit` are replaced by the per-format rewrite-rules traits,
  `SubstraitRules` today and a SQL counterpart when SQL emission returns.
- **0044:** its aside "the only `AsOf` in the workspace is
  `Capability::AsOfJoin`" loses its subject: `Capability::AsOfJoin` is deleted
  with its type, and no `AsOf` capability or join type remains (Substrait's
  `JoinType` has none either); its ruling — no AsOf authoring form; it returns
  with temporal-join authoring — stands.
- **0003:** its open question, "injecting an `EngineAdapter` into `plan()`"
  versus "no engine influence before `adapt()`", is resolved for the second: the
  engine enters as rewrite rules at adapt and never at plan.

**ADR 0006** — consistent, and given a print surface: trace reaches
`substrait-explain` through the explain copy; nothing executes it.

**ADR 0019 stands as written** — Substrait remains derived and outbound-only,
and conversion code lives in `semstrait-adapter`.

**Consistent with ADR 0024** — the format adapter inherits the planner's purity:
a function of `(plan, rules)`, no clock, no I/O. Not an amendment.

## More Information

Decided 2026-09-23, with a rewrite of ADR 0066, and on 2026-10-20 for the engine adapter's own crate.
