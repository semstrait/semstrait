---
number: 90
title: semstrait-ir keeps the plan and its Parameter, and the manifest owns the expression id and the expression layer
status: accepted
date: 2026-10-13
source:
- CONTEXT.md
- docs/adr/0001-invert-model-ir-dependency-edge.md
- docs/adr/0046-physical-source-is-flat-with-explicit-schema-provenance.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0066-the-plan-vocabulary-mirrors-substrait-v0-103-1-relations.md
- docs/adr/0068-adapters-are-layered-format-first.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0076-a-window-marker-gives-each-aggregate-its-rows-and-latest-and-earliest-order-by-one-dimension-inside-their-axes.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0086-a-data-type-takes-its-sql-name-and-has-a-canonical-declaration-and-a-dialect-declaration-where-the-dialect-differs.md
- docs/adr/0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md
- docs/adr/0088-the-tree-has-one-json-form-and-a-manifest-entry-holds-a-whole-tree-that-links-each-other-entry-it-references.md
- docs/adr/0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md
- docs/adr/0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md
- '`PhysicalExprId` in `crates/semstrait-manifest/src/manifest/expression.rs`; `ExprLayer` and `ExprLayer::of_node` in `crates/semstrait-manifest/src/manifest/expression_layer.rs`; `Parameter` and `ParameterKey` in `crates/semstrait-ir/src/expr.rs`; `PhysicalLeaf` in `crates/semstrait-sql/src/expr/leaf.rs`; `PhysicalExpr` in `crates/semstrait-ir/src/expr.rs`; `SemAnnotation` in `crates/semstrait-ir/src/plan/trace.rs`; `CompileError` in `crates/semstrait-manifest/src/error.rs`; the re-exports in `crates/semstrait-ir/src/lib.rs`'
- '`CompileError::Ir` in `crates/semstrait-manifest/src/error.rs`; `ManifestExpression` in `crates/semstrait-manifest/src/manifest/expression.rs`; `an_accessor_bearing_leaf_is_refused` in `crates/semstrait-manifest/src/compile/expressions/lower.rs`; `crates/semstrait-manifest/tests/fixtures/happy_manifest.snapshot.json`'
- the human's rulings, of 2026-10-13
links:
- target: 46
  kind: amends
- target: 66
  kind: amends
- target: 76
  kind: amends
- target: 94
  kind: amendedby
---

# semstrait-ir keeps the plan and its Parameter, and the manifest owns the expression id and the expression layer

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0094](0094-semstrait-model-keeps-the-authoring-forms-and-the-manifest-ir-and-planner-name-none-of-its-types.md): rule 3 ("keeps its dependency on `semstrait-model` for `SemanticsName` only"): rule 9. Rule 6 ("A crate that uses `Grain` imports it from `semstrait-model`", and the dependency table's rows for `semstrait-ir` and `semstrait-manifest`): rules 3 and 11. *In short*'s "it takes only `SemanticsName` from `semstrait-model`" (near 0090:177-178): rule 9.

## Context and Problem Statement

ADRs 0085 to 0089, 0091 and 0092 move the expression tree, the function
declarations and the data types into `semstrait-sql`. Today `semstrait-ir`
holds these expression and function types:

| Type | File in `crates/semstrait-ir/src` | What it is |
|---|---|---|
| `PhysicalLeaf`, `PhysicalExpr = Expr<PhysicalLeaf>` | `expr/leaves.rs` | the plan's leaf set and tree |
| `Parameter`, `ParameterKey` | `expr/parameter.rs` | the placeholder for a value that only a request decides |
| `ExprId`, `PhysicalExprId(pub ExprId)` | `expr/id.rs` | the id of a manifest entry |
| `ExprLayer`, `classify`, `ExprLayer::of_node` | `expr/layer.rs` | the expression layer |
| `AggregateExpr` | `primitives.rs` | one aggregate of the plan's aggregation node |
| `functions/`, `TypeClass` | the former `functions/` and `types.rs` | the function registry and the type classes |
| `CompileError` | `error.rs` | the registry's function-resolution errors |
| 26 types of `semstrait-model`; `Tree`, `Visitor` and `Rewriter` of `semstrait-common` | `lib.rs`, the former `types.rs`, `expr_kinds.rs` and `tree.rs` | re-exports |

Four questions are open: where `Parameter` lives, what `semstrait-ir`
re-exports, where `PhysicalExprId` and `ExprLayer` live, and what a stored tree
holds for the part of a lowered accessor (`CONTEXT.md` **Sugar**) that only a
request decides.

In the workspace, only `semstrait-manifest` uses `semstrait-ir`. The manifest
makes each expression id, and it stores the id and the expression layer on each
entry. This is a stored aggregate and its child, from
`crates/semstrait-manifest/tests/fixtures/happy_manifest.snapshot.json`. Each
long id is shortened here to `#1` and `#2`:

```json
{"id":#1,"node":{"type":"aggregate","content":{"op":{"type":"builtin","content":"sum"},"args":[#2],"distinct":false,"filter":null}},"layer":"aggregate","physical_sources":["bf575591-…"]}
{"id":#2,"node":{"type":"leaf","content":{"type":"column","content":"revenue"}},"layer":"scalar","physical_sources":["bf575591-…"]}
```

The manifest computes the layer node by node with `ExprLayer::of_node`, and its
validator calls `classify`. ADR 0071 says: "`ExprLayer` stays on every stored
node. The planner does not read it." No other part of `semstrait-ir` uses the
layer. `semstrait-ir`'s own semantic graph names the id, in
`GraphExprRef::Physical(PhysicalExprId)`, and nothing in the workspace builds a
semantic graph.

No code makes a `Parameter` today. Compile refuses every accessor: its test
`an_accessor_bearing_leaf_is_refused` says "An accessor is refused, not
lowered". A stored expression holds no `Parameter` (ADR 0088, rule 4).

The plan's aggregation node holds `AggregateExpr { aggregation: AggregateKind,
args, distinct, filter, inferred_type }`. `AggregateKind` retires (ADR 0085,
rule 3). `semstrait-sql`'s `AggregateCall` holds an aggregate function, its
arguments, its options and its clauses, among them `distinct` and `filter`
(ADR 0091, rule 1).

After the move, `semstrait-ir` uses one type of `semstrait-model`:
`SemanticsName`, in each plan node's trace (`SemAnnotation`, in the
**Node-common block**).

The human, on this decision's question: "It might be the case that IR doesn't
need expression identity anymore, or import generics for plan." On
this point: "all others cargo -
removes their own typing for expressions and datatype - and use semstrait-sql
based types, they still can build their own wrapper types , to keep their
functionality, but core - from semstrait-sql."

What does `semstrait-ir` keep of expressions and functions, and where do the
expression id and the expression layer go?

## Decision Drivers

- Each crate takes its expression and data types from `semstrait-sql`. It
  builds a wrapper type only to keep its own functionality (the human).
- A consumer's mechanics live in the consumer's crate: the manifest's id and
  layer in the manifest, and the plan's leaf beside the plan.
- Each type has one import path: the path of the crate that owns it.
- The changes stay small. `semstrait-ir` keeps its dependency on
  `semstrait-model`, and `semstrait-model`, `semstrait-ir` and `semstrait-manifest`
  each depend on `semstrait-sql` directly.
- The semantic graph and the sugar implementation are not part of this work.

## Considered Options

- **`semstrait-ir` keeps the plan's own types, and the manifest owns the
  expression id and the expression layer** (chosen).
- For the expression id: **in `semstrait-common`, beside `EntityId`**, or **in
  `semstrait-ir`, as today**. Refused: the id is the manifest's. The human: "it
  id is manifest side". **Two types, `ExprId` and `PhysicalExprId`**. Refused:
  the wrapper kept the ids of two pools apart, and the second pool,
  `SemanticExprId`, is retired.
- For the expression layer: **in `semstrait-sql`'s `expr`**, because the layer
  follows from a tree's structure alone. Refused: only the manifest writes and
  reads it, and `semstrait-sql` holds no manifest mechanics. The human: "PhysicalExprId - manifest concern and Manifest layering logic - is in
  manifest, not in semstrait-sql".
- For `Parameter`: **in `semstrait-sql`'s `expr`, beside `SemanticLeaf` and
  `NoLeaf`**. Refused: only the plan's tree holds a `Parameter`. The manifest
  keeps its own leaf, `Link`, in its own crate (ADR 0088), so the plan does the
  same.
- For the plan's aggregate: **`AggregateExpr` as a wrapper of a `Call` and its
  `inferred_type`**. Refused: the node's output schema already holds the type.
  The human: "do we even need this in IR ? I think ir acn import sql cargo to
  define what plan nodes store, so we don't need to implement new typing".
  **Any tree as an entry, `(Name, PhysicalExpr)`, with `validate()` checking
  its root**. Refused: a call is the narrower type, because it holds exactly one
  function. **A `Call` of any function type, with `validate()` refusing one
  that is not an aggregate**. Refused: the type then holds a scalar or a
  window call, and only a run of `validate()` refuses it, where the compiler
  can. The human: "fucntion call doesn't consists identity of function type -
  scalar, aggregate".
- For the re-exports: **re-export the types that the plan's own types hold**
  (`DataType`, `SortDir`, `NullOrdering`, `Expr`), or **keep today's re-exports,
  re-pointed at `semstrait-sql`**. Refused: each gives a type a second path. The
  human: "it is not that manifest must import expr/datatype dependencies over
  ir, both import from sql cargo".
- For the dependency on `semstrait-model`: **`semstrait-ir` drops it, and
  `SemanticsName` moves down**. Refused: it moves a
  type out of `semstrait-model` and changes the dependencies of `semstrait-ir`. The human: "yes,
  to minimize changes".
- For the semantic graph: **move `semantic_graph.rs` to `semstrait-manifest`
  with no change**, or **delete `GraphExprRef`**. Refused: the human: "graph out
  of scope. implementation stays as it is".

## Decision Outcome

Chosen option: `semstrait-ir` keeps the plan's own types, and the manifest owns
the expression id and the expression layer, because each crate then holds only
its own mechanics, the plan stores `semstrait-sql`'s tree and calls with no
typing of its own, each type has one import path, and
`semstrait-model`, `semstrait-ir` and `semstrait-manifest` each depend on
`semstrait-sql` directly.

In short:

- `Parameter` and the plan's tree, `PhysicalExpr = Expr<Parameter>`, stay in
  `semstrait-ir`. `AggregateExpr` retires, and the plan's aggregation node holds
  `semstrait-sql`'s `AggregateCall`.
- One expression id type, `PhysicalExprId`, and the expression layer move to
  `semstrait-manifest`.
- `semstrait-ir` re-exports no type that it does not own, and it takes only
  `SemanticsName` from `semstrait-model`.
- `semstrait-ir`'s `CompileError` retires. The manifest's `CompileError::Ir`,
  which only carries it, retires too.
- The semantic graph and the sugar implementation are outside this decision.

### `semstrait-ir` keeps the plan's own types

1. `Parameter`, `ParameterKey` and the alias `PhysicalExpr = Expr<Parameter>`
   stay in `semstrait-ir`, beside the plan nodes, which hold their expressions
   as `PhysicalExpr` (ADR 0066, ruling 4). `semstrait-ir`
   imports `Expr` from `semstrait-sql`. ADR 0085's rule 2 gives the plan this
   leaf, and only the plan's tree holds it.

2. `AggregateExpr` retires. The aggregation node holds `semstrait-sql`'s
   aggregate call. The human: "may be then better bring different types of
   call Function( Scalar/Aggregate/Window )":

   ```rust
   // semstrait-ir; the node's other fields are ADR 0066's
   pub aggregates: Vec<(Name, AggregateCall<Parameter, AggregateAllowed>)>,
   ```

   For `sum(amount) FILTER (WHERE paid)` as `total`, grouped by `channel`:

   | | `aggregates` | `output_schema` |
   |---|---|---|
   | Before | `("total", AggregateExpr { aggregation: Builtin(Sum), args: [amount], distinct: false, filter: Some(paid), inferred_type: long })` | `channel: string`, `total: long` |
   | Now | `("total", AggregateCall { function: &SUM, args: [amount], options: [], filter: Some(paid), .. })` | `channel: string`, `total: bigint` |

   - An aggregate's type is the `data_type` of its column on the node's
     `output_schema`. ADR 0066 already makes that `data_type` mandatory
     (ruling 5).
   - The aggregate call's `distinct` and `filter` clauses hold what
     `AggregateExpr`'s `distinct` and `filter` hold today. The `filter` clause
     is the mirror of `AggregateRel.Measure { measure, filter }` (ADR 0066,
     ruling 8).
   - An entry's Rust type is an aggregate call, so an entry cannot hold a call
     of a scalar or a window function, such as `("x", upper(name))`, or an
     aggregate with OVER, which `AggregateRel.Measure` cannot hold (ADR 0091,
     rule 1).

3. `semstrait-ir` keeps its dependency on `semstrait-model` for `SemanticsName`
   only, in each plan node's trace (`SemAnnotation`). It takes no expression type from `semstrait-model`, and it depends on
   `semstrait-sql` directly (rule 6). The human: "I think IR based of semstrait-sql, most likely
   we don't need expr anything from model".

### The manifest owns the expression id and the expression layer

4. The expression id is one type in `semstrait-manifest`:

   ```rust
   // semstrait-manifest
   pub struct PhysicalExprId(Uuid);         // a v5 UUID, written as its text
   ```

   `ExprId` merges into it. Compile mints each id from its entry's owner and
   root marker, and the JSON writes it as UUID text, as it writes an
   `EntityId`: `"id":"253641c7-819a-5b60-a3dc-c276bfceb36d"` (ADR 0088, rule
   13).

5. `ExprLayer` and `ExprLayer::of_node` move to `semstrait-manifest`, where
   `ManifestExpressions::layer_of` is the one layer rule of an entry. Each
   entry holds its `layer`, as ADR 0088 rule 6 has it, and the planner does not
   read it (ADR 0071).

### Each crate imports a type from the crate that owns it

6. `semstrait-ir` re-exports no type that it does not own. `semstrait-ir` and
   `semstrait-manifest` each import the expression and data types from
   `semstrait-sql`, and neither crate imports them through the other. A crate
   that uses `Grain` imports it from `semstrait-model`. the former `expr_kinds.rs` and
   `tree.rs`, which hold only
   re-exports, are deleted. `types.rs` keeps `Schema` and `SchemaColumn`, whose
   `DataType` is `semstrait-sql`'s. The human: "manifest imports type system
   from sql, which is correct path. Ir holds some dependencies from sql to put
   into plan sql related typing - expressions, types. manifest do the same. so
   it is not that manifest must import expr/datatype dependencies over ir, both
   import from sql cargo". A `semstrait-ir` that
   re-exports `registry()` is not the design.

   Each crate depends directly on each crate whose types it uses, and a dependency
   on a crate other than `semstrait-common` and `semstrait-sql` is for a type that
   crate owns. The human: "each of them can import sql cargo directly for what they
   need, and cross import needed only if there is a need to link type or reuse type
   declared based on sql but in different cargo".

   | Crate | Depends on | For |
   |---|---|---|
   | `semstrait-sql` | `semstrait-common` | `Tree`, `Visitor`, `Rewriter` and `ValidateError` for `Expr`, and `Dialect` (ADR 0091, rule 6) |
   | `semstrait-model` | `semstrait-common`, `semstrait-sql` | the expression, function and data types; the diagnostics, `EntityId` and `io` |
   | `semstrait-ir` | `semstrait-common`, `semstrait-sql`, `semstrait-model` | `Tree` for `PlanNode`; `Expr`, `AggregateCall` and `DataType`; `SemanticsName` (rule 3) |
   | `semstrait-manifest` | `semstrait-common`, `semstrait-sql`, `semstrait-model`, `semstrait-ir`, `semstrait-catalog` | the expression, function and data types; the semantic model that it compiles; the types that `semstrait-ir` owns, such as `Schema`, `Cardinality` and `JoinType`; the catalog that compile reads |

### What retires from `semstrait-ir`

7. These retire:

   | Type | Rule |
   |---|---|
   | `PhysicalLeaf` | ADR 0085, rule 3 |
   | the former `functions/`: `FunctionRegistry`, `function_registry()`, `FunctionSpec`, `ParamType`, `FnSignature`, `ReturnTypeRule`, `FunctionCategory`, `DimensionAxis`, the former `builtins/` and `extension.rs` | ADR 0087, rule 10. `Additivity` moves to `semstrait-sql` (ADR 0085, rule 3) |
   | `TypeClass` | ADR 0086, rule 8 |
   | the former `artifact.rs`: `EngineArtifact`, `EnginePlan`, `SqlArtifact`, `DialectId`, `Dialect`, `Capability` | ADR 0068, *Artifacts* |
   | `AggregateKind`, `AggregationOp`, `WindowFn` | ADR 0085, rule 3 |
   | `CompileError`, and the manifest's `CompileError::Ir`, which only carries it | this ADR |

   `semstrait-ir`'s `CompileError` holds the registry's function-resolution
   errors: `UnknownFunction`, `FunctionArityMismatch`, `NoMatchingSignature`,
   `CustomRuleRejected` and `SpecInconsistent`. Each of these errors now has
   another home. Compile reports a typing error as a `TypeError` "through one
   `CompileError` variant" (ADR 0089, rule 13), and the registry's seal refuses
   its faults, "each a typed `SealFault`" (ADR 0087, rule 9).

8. `IrErrorKind`, `Schema`, `SchemaColumn` and the plan nodes stay.

### Outside this decision

9. The semantic graph is outside this decision, and the implementation keeps
   `semantic_graph.rs` as it is. Its `GraphExprRef::Physical(PhysicalExprId)`
   names a type that `semstrait-ir` cannot see once the id is the manifest's.
   Where the file cannot build, the implementation comments out the part that
   cannot build, and may mark the rest dead code. The human: "graph out of
   scope. implementation stays as it is. (c). can be marked as dead_code or
   commented".

10. The sugar implementation is outside this decision. That includes what a stored
    tree holds for an accessor, whose partition and order only a request
    decides. Compile still refuses every accessor, and `ParameterKey` keeps its
    two keys. The human: "not in scope of this implementation.
    sugar implementation later on feature".

### Amends

- **0046:** In its *Decision Outcome*, "Every type it persists is owned by
  `semstrait-ir`, `semstrait-model`, or `semstrait-catalog`" also names
  `semstrait-manifest`, for its own storage types: the expression id and the
  expression layer (ADR 0090, rules 4 and 5), and a stored expression and its
  link (ADR 0088, rule 4). In *The schema is canonical; the native spelling is
  catalog-owned provenance*, the sentence on the
  "`pub use semstrait_model::{DataType, Grain}` re-import" of `ir::types`
  becomes: `semstrait-ir` re-exports neither type; `DataType` is
  `semstrait-sql`'s and `Grain` is `semstrait-model`'s, and a crate that uses
  either imports it from there (ADR 0090, rule 6). The rest of 0046 stands.
- **0066:** in *Rulings*, ruling 5's "`SchemaColumn.data_type` / `nullable` and
  `AggregateExpr.inferred_type` are mandatory, on every `output_schema` and on
  `ScanNode.schema`" becomes: `SchemaColumn.data_type` / `nullable` are
  mandatory, on every `output_schema` and on `ScanNode.schema`, and an
  aggregate's type is its column on the `AggregateNode`'s `output_schema` (ADR
  0090, rule 2). Ruling 8's "`AggregateExpr` keeps its `filter` slot as the
  mirror of `AggregateRel.Measure { measure, filter }`" becomes: an
  `AggregateNode` entry's call keeps its `filter` clause as the mirror of
  `AggregateRel.Measure { measure, filter }` (ADR 0090, rule 2). The rest of 0066
  stands.
- **0076:** in *A window's order key carries its direction*, "`semstrait-ir`
  re-exports them" becomes: `semstrait-ir` imports `SortDir` and `NullOrdering`
  from `semstrait-sql` (ADR 0090, rule 6). The rest of 0076 stands.

### Consequences

- Good, because `semstrait-ir` holds no expression tree, typing or function of
  its own. Its only expression types are the plan's leaf, `Parameter` with its
  `ParameterKey`, and the alias `PhysicalExpr` over `semstrait-sql`'s tree.
- Good, because the manifest's id and layer live in the one crate that writes
  and reads them.
- Good, because each type has one import path, the path of the crate that owns
  it.
- Good, because the plan writes an aggregate's type once, on its output schema.
- Good, because an entry's Rust type is an aggregate call, so the compiler
  refuses `("x", upper(name))` and an aggregate with OVER, and no plan check
  has to.
- Bad, because `GraphExprRef::Physical(PhysicalExprId)` names a type that moves
  to the manifest. That part of the semantic graph is commented out, and a
  `SemanticEdge` then has no type for its join condition. The gain is that this
  map makes no change to the graph, which is not a priority now. The human:
  "First bad is trade off, which is ok, right now semantic graph is not a
  priority to implement".
- Neutral: the manifest's imports of `DataType` through `semstrait_ir` change
  to `semstrait_sql`, as each crate imports from `semstrait-sql`.
- Neutral: the sites that build or read `AggregateExpr` change to the aggregate
  call. Nothing else in the workspace uses `AggregateExpr`.
- Neutral: `CompileError::Ir` retires, and ADR 0044 counts it among compile's
  variants (*Twenty hard rejections exist and five carry a number*); ADR 0044 is
  not amended for it.
- Outside the two rulings that this ADR amends, ADRs 0066, 0068 and 0071 name
  `AggregateExpr` and its fields: ADR 0066's "`AggregateExpr.phase`" and its
  item "`inferred_type` and the filter splice"; ADR 0068's refused option
  "Retiring the `filter` slot on `AggregateExpr`", its "`AggregateExpr.filter` is
  `None`", "The filter is in `AggregateExpr.filter`" and "reaches the plan as
  `AggregateExpr.filter`"; and ADR 0071's "that Measure's
  `AggregateExpr.filter`". Each `AggregateExpr` stands for an
  `AggregateNode` entry's aggregate call, its `filter` for the call's `filter` clause, and
  its `inferred_type` for the entry's column on the node's `output_schema`. Those
  ADRs are not amended for it.

## More Information

The question was what semstrait-ir keeps of expressions and
functions. The aggregation node holds an `AggregateCall`, and the crates' dependencies are as ruled in the design review.
