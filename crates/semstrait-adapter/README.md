# semstrait-adapter

The Substrait format adapter (`CONTEXT.md` **Format adapter**; ADR 0068).
`SubstraitProducer::new(&dyn SubstraitRules)` lowers a `SemanticPlan` that
`validate` accepted to `substrait::proto::Plan`: `produce(&self, &SemanticPlan)`
returns `Result<(Plan, Diagnostics<InvalidRewrite>), (Diagnostic<InvalidRewrite>,
Diagnostics<InvalidRewrite>)>`, ADR 0005's fail-fast form, with no warning on
either arm. `explain(&SemanticPlan, &dyn SubstraitRules)` returns the same form
over the text of the explain copy, in which each relation carries one
`+ Opt:Trace` line per semantic source its node plans, plus a rank line when
the node holds one (ADR 0095). The public `trace_registry()` registers the
`semstrait.Trace` extension that `format_with_registry` reads. Its inputs
are the plan and the rules, and nothing else.

It lowers the plan's vocabulary: each node's faithful default (ADR
0068, *Defaults are faithful*, as ADR 0073 amends it), each call by the
producer's own table, each argument cast to the type of Substrait's overload
(ADR 0087, rule 24; ADR 0092, rules 3 and 9), each type's proto (ADR 0086,
rule 6) and each typed literal (ADR 0085, rule 7), a typed NULL included. It
lowers a `JoinNode` as a `JoinRel`, `Full` as `OUTER` and `post_join_filter` as
itself, a `CrossNode` as a `CrossRel`, and a `MultisetNode` as a `SetRel` with
`UNION_ALL` or `UNION_DISTINCT`, each with its field positions from its inputs'
output schemas (ADR 0068, *The seam*). A window call in a `ProjectNode`'s
expressions lowers as a `WindowFunction` in its `ProjectRel` (ADR 0087, rule
24). A `TopNNode` has no lowering yet.

| Module | Holds |
|---|---|
| `producer` | `SubstraitProducer`: the walk, each node's relation, the expressions and the calls |
| `rules` | `SubstraitRules`, unsealed: every hook an identity or the faithful answer |
| `functions` | the table of each function's and operation's Substrait form, `FunctionName`, and the overload types |
| `extensions` | `Extensions`: the extension declarations, anchored in the order of first use |
| `literal` | each typed literal as a Substrait literal |
| `types` | each type's proto, and a schema's `NamedStruct` |
| `explain` | `explain`, and the trace's message |
| `error` | `InvalidRewrite`, `VocabularyPart` |

It depends on `semstrait-common` (`Diagnostic`, `Diagnose`), `semstrait-sql`
(the expression tree, the declarations, `DataType` and the type systems),
`semstrait-ir` (the plan), `semstrait-catalog` (`FileFormat`), `substrait`,
`substrait-explain`, `prost` and `thiserror`, and imports each type from the
crate that owns it (ADR 0090, rule 6). It links no engine, its tests included,
and holds no engine's rules (ADR 0068).
