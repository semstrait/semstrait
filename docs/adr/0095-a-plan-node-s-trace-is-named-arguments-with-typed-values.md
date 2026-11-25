---
number: 95
title: A plan node's trace is named arguments with typed values
status: proposed
date: 2026-10-30
source:
- CONTEXT.md
- docs/adr/0045-semantic-sources-are-a-flat-name-keyed-acyclic-collection.md
- docs/adr/0053-authoring-direction-is-exclusive-per-root-data-kind.md
- docs/adr/0064-field-first-resolution-ranks-public-candidates-by-structural-complexity.md
- docs/adr/0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md
- '`semantic_meta` in `crates/semstrait-ir/src/plan/common.rs:18`; `SemAnnotation` in `crates/semstrait-ir/src/plan/trace.rs:7-25`'
- '`trace`, `Trace` and `subject` in `crates/semstrait-adapter/src/explain.rs:41-62,88-138`; `advanced_extension` in `crates/semstrait-adapter/src/producer.rs:289-292`'
- '`crates/semstrait-planner/src/build.rs:76-83,193,220-229,246-249,1618-1639`; `crates/semstrait-planner/src/build/stack.rs:36-40`; `crates/semstrait-planner/src/build/drill_across.rs:59-60`'
- '`crates/semstrait-adapter-datafusion/tests/fixtures/oracle/joinset/pruned.yaml:3-44,46-96`; `crates/semstrait-adapter-datafusion/tests/fixtures/oracle/composition/time_axis.yaml:730`; `crates/semstrait-adapter-datafusion/tests/fixtures/oracle/composition/fold.yaml:355`; `crates/semstrait-adapter-datafusion/tests/fixtures/oracle/read/partitions.yaml:27-30`; `crates/semstrait-adapter/tests/explain.rs:20-40`'
- '`substrait-explain-0.9.0`: `src/extensions/args.rs:192-199,396-422`, `src/textify/extensions.rs:20-27`, `tests/adv_extension_roundtrip.rs:224-225`'
- '`crates/semstrait-manifest/src/compile/assemble.rs:161-170,364-386,699`; `crates/semstrait-manifest/src/manifest/semantic_source.rs:16-34`'
---

# A plan node's trace is named arguments with typed values

## Context and Problem Statement

Each plan node holds `semantic_meta: Vec<SemAnnotation>`
(`crates/semstrait-ir/src/plan/common.rs:18`). `SemAnnotation`
(`crates/semstrait-ir/src/plan/trace.rs:7-25`) has the variants `DataKind`,
`Dataset`, `PhysicalSource`, `StrategyBoundary { source, position }`,
`Cardinality`, `BranchDelivery`, `Pruned`, `AnsweredBy`, `ScopeRoot`,
`Compositions`, and `GranularityFit { group, granularity }`.

The adapter renders each one as a positional string (`explain.rs:88-138`,
function `subject`), carried as `Trace { subjects: repeated string }`
(`explain.rs:41-62`), type URL `/semstrait.Trace`, in the advanced extension's
`optimization`. `advanced_extension` builds it only in explain production,
never in an executed plan (`producer.rs:289-292`). The REST `/explain`
endpoint and the CLI return that text.

Producers, in `semstrait-planner/src`: `build.rs:76-83` writes `ScopeRoot`,
`Compositions`, `GranularityFit`; `:193` writes `AnsweredBy`; `:220-229`
writes `Dataset` and `PhysicalSource`; `:246-249` writes `Cardinality`;
`:1618-1629`'s `trace` helper always pairs `DataKind(source)` with
`StrategyBoundary { Entry }`, converting a `DataKindName` to a
`SemanticsName`; `:1632-1639` writes `Pruned`; `build/stack.rs:36-40` writes
`BranchDelivery`; `build/drill_across.rs:59-60` writes
`Cardinality::OneToOne`.

Real output, `crates/semstrait-adapter-datafusion/tests/fixtures/oracle/joinset/pruned.yaml`,
case `left_join_prunes_other_side` (request `from: orders_customers, select:
[region, revenue]`, model `commerce_own_types`), lines 39-44:

```
Root[region, revenue]
  Aggregate[$0 => $0, sum($1, overflow⇒[ERROR]):i64?]
    + Opt:Trace['data kind orders_customers', 'strategy boundary orders_customers entry', 'pruned orders_customers.customers', 'scope root orders_customers', 'compositions 1', 'granularity fit not selected']
    Project[$0, $1]
      Read[warehouse.orders => region_code:string?, amount:i64?]
        + Opt:Trace['data kind orders_customers.orders', 'dataset orders_customers.orders', 'physical source orders', 'strategy boundary orders_customers.orders entry']
```

Case `bypassed_middle_operand` (request `from: sales, select: [region_name,
revenue]`), lines 83-92 (the `sales.regions` branch repeats the same shape
and is omitted):

```
Root[region_name, revenue]
  Aggregate[$0 => $0, sum($1, overflow⇒[ERROR]):i64?]
    + Opt:Trace['data kind sales', 'strategy boundary sales entry', 'answered by sales.r2', 'scope root sales', 'compositions 2', 'granularity fit not selected']
    Project[$2, $1]
      Join[&Inner, equal($0, $3):boolean? => $0, $1, $2, $3]
        + Opt:Trace['data kind sales.r2', 'strategy boundary sales.r2 entry', 'cardinality many-to-one']
        Aggregate[$0 => $0, sum($1, overflow⇒[ERROR]):i64?]
          Project[$1, $0]
            Read[warehouse.orders => amount:i64?, region_code:string?]
              + Opt:Trace['data kind sales.orders', 'dataset sales.orders', 'physical source orders', 'strategy boundary sales.orders entry']
```

The adapter test
`the_explain_text_has_one_opt_line_for_a_traced_scan_and_one_for_a_traced_aggregate`
(`crates/semstrait-adapter/tests/explain.rs:20-40`) pins the same positional
form for a simpler plan.

`substrait-explain` 0.9.0 already renders named arguments with typed
values: `ExtensionArgs { positional, named: IndexMap<String,
ExtensionValue>, output_columns }` (`src/extensions/args.rs:192-199`), with
`ExtensionValue::{String, Integer, Float, Boolean, Null, Error, Expr,
Enum(String), Tuple(TupleValue)}` (`:396-422`). An enum renders as `&CORE`
in text (`:414`); `&HASH` is the tuple example (`:417`); its own fixture
already writes `+ Opt:PlanHint[hint='use_index']`
(`tests/adv_extension_roundtrip.rs:224`).

CONTEXT.md has no **Trace** entry. ADR 0064, *ADR 0018 is untouched* (near
line 263): "Which source was chosen, with the scope root's rank (its
compositions and its granularity fit), rides only in TRACE-class
`SemAnnotation`s on the plan nodes (`semstrait-ir::plan::meta`), which the
canonical layer *records* and never *reads*". That stays true; the module is
`plan::trace` today.

Glossary words for the keys. **Semantic source** avoids bare "source",
which is Physical source's word; it is "authored, nested and
compile-synthesized alike". **Scope root** avoids "entry point", "anchor",
"target", "base". **Explicit composition** avoids bare "implicit", which
separately labels synthesized data kinds, derived join paths, and
enumerated compositions. **Implicit Unionset** "holds the authored name".
**Reduction leaf** "is private, since nobody wrote its name". **Source
role** "What a semantic source is and does in planning, in structure and in
behaviour: it reads one physical source (Dataset), joins two operands
(Joinset), stacks branches (Unionset), or picks one child or joins its
children (Grainset)".

Several kinds of semantic source are private, all built in
`crates/semstrait-manifest/src/compile/assemble.rs`: a nested authored data
kind is private (`Placement::nested`, `:161-170`: `public: false,
originators: [parent]`), and so is a Joinset's hop, which compile names
after its relationship (`:364-386`, the name built at `:699`, for example
`sales.r2`). `sales.r2` (compile-made) and `sales.orders` (authored, nested)
both end up with `sales` as their one recorded originator; `public` and
`originators` cannot tell which one compile made
(`manifest/semantic_source.rs:16-34`).

The named form must hold three shapes the fixtures already show, under
`crates/semstrait-adapter-datafusion/tests/fixtures/`. A node can plan two
semantic sources on one line today (`oracle/composition/time_axis.yaml:730`,
4 such lines): `+ Opt:Trace['data kind gb2', 'data kind gb2.gbal3',
'strategy boundary gb2 entry', 'strategy boundary gb2.gbal3 entry', 'pruned
gb2.accounts', 'answered by gb2.gbal3.u_bal', 'scope root gb2', 'compositions
3', 'granularity fit requested year']`. One semantic source can carry two
pruned operands (`oracle/composition/fold.yaml:355`, 8 lines with two or
more decisions): `+ Opt:Trace['data kind u0.j.u3', 'strategy boundary
u0.j.u3 entry', 'branch delivery partial aggregates', 'pruned u0.j.u3.e7',
'pruned u0.j.u3.e8']`. The rank can sit on a node that plans no semantic
source (`oracle/read/partitions.yaml:27-30`, 239 of 276 `scope root` lines):
`+ Opt:Trace['scope root sales', 'compositions 0', 'granularity fit not
selected']`.

## Considered Options

- Keep positional strings and write `key=value` inside each string: still
  text; no typed value reaches the Substrait extension.
- Read a `source_implicit` key as private (`public=false`): it marks the
  authored nested `sales.orders` the same as the compile-made `sales.r2`;
  `public` already says what a request may name, not provenance.
- Read a `source_implicit` key as "field-first resolution inferred the
  scope root": a property of the request, not of the semantic source.
- Bare `source=` and "entry point": reserved for Physical source, and
  avoided, by the glossary.
- Keep `data kind`, `strategy boundary … entry` and `dataset`: each
  repeats another key.
- A `synthesized` bool on each semantic source: it re-homes the origin
  fact ADR 0045 retired — "`DataKindOrigin` is deleted rather than
  re-homed, because implicitness is not a planner fact."
- One `decision` key holding a tuple of `(kind, name)` pairs: one key per
  kind reads cleaner.
- `scope_root=true` on the source's line: the rank then leaves the node
  that holds it, or splits from its source.
- `originator='<name>'` as a string: a second form when the list has
  several.
- `semantic_source_type`: the glossary's word is **Source role**.

## Decision Outcome

Chosen option: named arguments with typed values on the `semstrait.Trace`
extension, because a typed value reaches the extension directly, and a key
name replaces the three lines that only repeated other keys.

### 1. Named arguments with typed values

Each `+ Opt:Trace[...]` line uses named arguments with typed values:
`'text'`, `&Enum`, `true`/`false`, integers, and tuples. The Substrait
extension `semstrait.Trace` carries typed fields, not a list of strings. IR's
`SemAnnotation` holds typed values with `semstrait-common` names (ADR 0094,
rule 9); only the adapter writes text. It stays in explain production only.

### 2. One line per semantic source

Each node carries one `+ Opt:Trace[...]` line for each semantic source that
the node plans; each line names one semantic source. Substrait's
`optimization` is a repeated field
(`crates/semstrait-adapter/src/explain.rs:46-49` builds it as a `vec!`), and
`substrait-explain` prints one `+ Opt:` line for each entry (its test
`tests/adv_extension_roundtrip.rs:224-225` shows two on one node). A named
argument is an `IndexMap` key, so a key appears once per line.

### 3. The rank line

The scope root's rank stays on the node that holds it today, on a line of
its own, and `scope_root` holds the scope root's name: `+
Opt:Trace[scope_root='sales', compositions=0, granularity_fit=&NotSelected]`
(plus `granularity=&…` when present).

### 4. Keys

- `semantic_source='<name>'`
- `source_role=&Dataset | &Joinset | &Unionset | &Grainset`
- `originators=('<name>', …)`, when the source records one or more
- `physical_source='<name>'`
- `scope_root='<name>'`
- `compositions=<n>`
- `granularity_fit=&Requested | &Selected | &NotSelected`
- `granularity=&Minute … &Year`, when present
- `cardinality=&OneToOne | &OneToMany | &ManyToOne | &ManyToMany`
- `delivery=&Rows | &PartialAggregates | &FinalAggregates`

### 5. Decisions

A stand-in is `answered_by='<name>'`; pruned
operands are `pruned=('<name>', …)`, a tuple even for one name
(`substrait-explain` prints a one-element tuple as `('x',)`,
`src/textify/extensions.rs:20-27`). Each goes on the line of the semantic
source that made the choice. A line that makes neither choice has neither
key.

### 6. Mapping from today

| Today | After |
|---|---|
| `data kind X` + `strategy boundary X entry` | `semantic_source='X', source_role=…` |
| `dataset X` | dropped; `source_role=&Dataset` says it |
| `physical source Y` | `physical_source='Y'` |
| `scope root X` | `scope_root='X'` on the rank line |
| `compositions N` | `compositions=N` |
| `granularity fit G [g]` | `granularity_fit=&G` (+ `granularity=&g`) |
| `cardinality c` | `cardinality=&C` |
| `branch delivery d` | `delivery=&D` |
| `answered by X` | `answered_by='X'` |
| `pruned X` | `pruned=('X', …)` |

Today's `bypassed_middle_operand` output, beside the new one:

```
# today
Aggregate[…]
  + Opt:Trace['data kind sales', 'strategy boundary sales entry', 'answered by sales.r2', 'scope root sales', 'compositions 2', 'granularity fit not selected']
  … Join[&Inner, …]
      + Opt:Trace['data kind sales.r2', 'strategy boundary sales.r2 entry', 'cardinality many-to-one']
      … Read[warehouse.orders …]
          + Opt:Trace['data kind sales.orders', 'dataset sales.orders', 'physical source orders', 'strategy boundary sales.orders entry']

# after
Aggregate[…]
  + Opt:Trace[scope_root='sales', compositions=2, granularity_fit=&NotSelected]
  + Opt:Trace[semantic_source='sales', source_role=&Joinset, answered_by='sales.r2']
  … Join[&Inner, …]
      + Opt:Trace[semantic_source='sales.r2', source_role=&Joinset, originators=('sales',), cardinality=&ManyToOne]
      … Read[warehouse.orders …]
          + Opt:Trace[semantic_source='sales.orders', source_role=&Dataset, originators=('sales',), physical_source='orders']
```

### 7. `originators`

A semantic source whose manifest entry records one or more originators
carries `originators=('<name>', …)`, a tuple even for one name, as the
manifest's `originators` is a list. ADR 0053, *A reference shares, and ADR
0026's refusal falls in one direction only* — "One entry, several
originators": "A unit referenced by two composers is one `semantic_sources`
entry, and ADR 0045's `originators` becomes genuinely plural" — the list
holds one name or several, so a tuple holds either. The name is how a
semantic source is named (ADR 0094, rule 8). CONTEXT.md **Originator**: "The
nearest authored data kind whose specification produced a given semantic
source, recorded on that source. It states provenance, which is a different
edge from the operand references that state composition." A semantic source
with no originator, such as an authored root (`sales`), has no `originators`
key.

### 8. Glossary

CONTEXT.md gains **Trace**: "The planning facts and decisions that a plan
node holds for explain. The canonical layer records it and never reads it.
_Avoid_: 'annotation', 'metadata', 'hint'."

### Amends

None. ADR 0064's sentence on TRACE-class `SemAnnotation`s stays true: the
rank still rides only in the trace, and the canonical layer still records it
and never reads it.

### Consequences

- Good: a reader or a tool reads each trace value by key and by type.
- Good: one line per semantic source says which one it is, of which source
  role, and, where the manifest recorded one or more, its originators.
- Neutral: every pinned plan text with `+ Opt:Trace[` changes. A search
  (`rg -l "Opt:Trace" crates`) finds 37 files. The files, all under
  `crates/semstrait-adapter-datafusion/tests/fixtures/` except the first:
  `crates/semstrait-adapter/tests/explain.rs`;
  `joins/{cross,drill_across,join_type,predicate,re_aggregated_count,union,unique_union}.yaml`;
  `oracle/additivity/{non_additive,semi_additive}.yaml`;
  `oracle/composition/{delivery,fold,grainset_constituent,time_axis,unique_union}.yaml`;
  `oracle/formula/{metric,push_down}.yaml`, `oracle/function/behavior.yaml`,
  `oracle/grainset/routing.yaml`, `oracle/harness/mechanics.yaml`;
  `oracle/joinset/{delivery,fan_out,predicate,projection,pruned}.yaml`;
  `oracle/read/{files,partitions,types}.yaml`, `oracle/request/elements.yaml`,
  `oracle/resolution/scope_root.yaml`, `oracle/temporal/shape.yaml`;
  `oracle/unionset/{delivery,filter,implicit,pruned,unique}.yaml`; and
  `vocab/unionset/implicit.yaml`.
- Bad, a trade-off: `sales.r2` (compile-made) and `sales.orders` (authored,
  nested) both read `originators=('sales',)`, so the trace does not say
  which one compile made; gain: no new stored fact, no ADR amended, and no
  new term for provenance.

## More Information

Decided in a design session on 2026-10-29/30. It depends on ADR 0094: the names `SemAnnotation`
holds move to `semstrait-common` (rule 9).
