---
number: 88
title: The tree has one JSON form, and a manifest entry holds a whole tree that links each other entry it references
status: accepted
date: 2026-10-12
source:
- CONTEXT.md
- docs/adr/0002-single-leaf-parameterized-expression-tree.md
- docs/adr/0037-semantics-bind-to-pooled-physical-expressions.md
- docs/adr/0042-manifest-validity-is-a-compile-stage-property.md
- docs/adr/0055-manifest-provenance-is-caller-supplied-and-reverse-indexes-are-not-persisted.md
- docs/adr/0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md
- docs/adr/0071-a-predicate-is-applied-where-its-semantics-carry-their-meaning.md
- docs/adr/0074-an-aggregate-sits-where-grain-and-additivity-allow-and-moves-up-otherwise.md
- docs/adr/0077-metric-additivity-applies-after-the-formula-a-filter-on-a-composition-reads-semantics-and-a-constituent-serves-only-the-names-it-maps.md
- docs/adr/0083-a-metric-declares-no-additivity.md
- docs/adr/0084-a-grainset-takes-each-name-from-its-first-ranked-child-and-full-joins-the-children-on-a-guaranteed-keyset.md
- docs/adr/0085-the-expression-tree-moves-to-semstrait-sql-with-a-typed-literal-and-a-name-path-for-every-reference.md
- docs/adr/0087-the-function-catalog-declares-the-common-sql-functions-that-datafusion-can-run.md
- docs/adr/0089-parse-checks-each-call-against-its-declaration-and-compile-types-and-folds-each-tree-with-semstrait-sql.md
- docs/adr/0091-a-function-has-a-canonical-declaration-and-a-dialect-declaration-where-the-engine-differs-and-a-call-is-a-call-of-its-function-s-type.md
- '`ExpressionNode` on `main`, and `StoredNode` and `ManifestExpression`, in `crates/semstrait-manifest/src/manifest/expression.rs`; `mint_expr_node_id` and `EXPR_NAMESPACE` in `crates/semstrait-manifest/src/identity.rs`; `mint_entity_id` in `crates/semstrait-common/src/identity.rs`; `substitute_semantic_leaf` in `crates/semstrait-manifest/src/compile/expressions/lower.rs`; `JsonReader` in `crates/semstrait-sql/src/expr/serde/json/de.rs`; the `children()` walks in `crates/semstrait-manifest/src/validate.rs`; `crates/semstrait-manifest/tests/fixtures/happy_manifest.snapshot.json`'
- '`datafusion-expr@55.0.0:src/expr.rs:866-873` (`Case`); `datafusion-proto-models@55.0.0:proto/datafusion.proto:725-734` (`CaseNode`, `WhenThen`); `substrait@0.63.0:substrait/proto/substrait/algebra.proto:1389-1399` (`IfThen`, `IfClause`); `calcite@calcite-1.42.0`: `core/src/main/java/org/apache/calcite/rel/externalize/RelJson.java` (a `RexCall` as `op` and `operands`), `rex/RexBuilder.java` (a CASE''s operands `[when, then, …, else]`)'
- the research on layered expression trees and layered expression storage
- the human's rulings of 2026-10-12, 2026-10-14, 2026-10-15 and 2026-10-17, including those of 2026-10-15 on the form of a CASE and on a review
- the decisions of 2026-10-17 and 2026-10-18, and the research on MetricFlow, Cube, Looker, Malloy and Calcite
- the decisions of 2026-10-18, and the research on link reading in Calcite and Substrait
links:
- target: 65
  kind: amends
- target: 55
  kind: amends
- target: 2
  kind: amends
- target: 37
  kind: amends
- target: 71
  kind: amends
---

# The tree has one JSON form, and a manifest entry holds a whole tree that links each other entry it references

## Context and Problem Statement

The manifest stores each expression tree as a **node store** (ADR 0065): one
entry per operator or leaf, each child addressed by its expression id, every
tree disjoint, and ids minted from position. On `main`, each entry's node is
`ExpressionNode` (`crates/semstrait-manifest/src/manifest/expression.rs`),
which repeats the tree's 14 variants with a `PhysicalExprId` in each child
position. The tree is written in its derived adjacently-tagged form (ADR 0002:
"machine transport only"), in which a leaf is wrapped as
`{"type":"leaf","content":…}` and each call node has one generic body that it
writes in full: `"distinct":false,"filter":null`.

ADR 0085 moves the tree to `semstrait-sql`, and gives that
crate "typing, behavior, serialization, parser, algebra if needed". With
`ExpressionNode`, the manifest keeps a second copy of the tree's variants and
of its serialization. The human's goal: "minimize changes in compile logic -
but use primitives and typing from semstrait-sql, so manifest do not need to
re-implement the same primitives and their ser/de on it's side".

A node store spreads one expression over many entries. On
`feature/semstrait-sql`, each entry's node is a `StoredNode`, one node of
`semstrait-sql`'s tree whose every child is a link, and the golden manifest
(`crates/semstrait-manifest/tests/fixtures/happy_manifest.snapshot.json`)
stores the trees of its 14 signatures in 38 entries. The Measure `revenue`
(`agg: sum`, declared `decimal(18,2)`, on the Dataset `orders`) is 4 entries.
Each long id is shortened here to `#1` … `#4`:

```json
{"id":#1,"node":{"type":"cast","content":{"input":#2,"target":"decimal(18,2)","on_failure":"error"}},"layer":"post_aggregate","physical_sources":["bf575591-…"]}
{"id":#2,"node":{"type":"aggregate","content":{"name":"sum","args":[#3]}},"layer":"aggregate","physical_sources":["bf575591-…"]}
{"id":#3,"node":{"type":"cast","content":{"input":#4,"target":"decimal(18,2)","on_failure":"error"}},"layer":"scalar","physical_sources":["bf575591-…"]}
{"id":#4,"node":{"type":"column","content":["ps0","revenue"]},"layer":"scalar","physical_sources":["bf575591-…"]}
```

A tree that references another semantics holds a copy of that semantics' tree
(ADR 0037, ADR 0065). In a run, a Dimension
`zone`, `CASE WHEN region IN ('eu', 'uk') THEN 'emea' ELSE 'other' END` over the
Dimension `region`, is 8 entries, and two of them are a copy of `region`'s cast
and column.

Each id is a v5 UUID, minted through `EXPR_NAMESPACE` from the owner, the root
marker and the path from the root (`mint_expr_node_id`), and written as a bare
JSON number of up to 39 digits. serde_json reads a number whose type it is not
told as a `u64`, an `i64` or an `f64`, so to keep each digit of a link above
`u64::MAX`, `semstrait-sql`'s reader reads each value as its JSON text first,
with serde_json's `raw_value` feature.

A CASE is written `{"whens":[[<when>, <then>], …],"else_":…}`: each WHEN and
its THEN are a list of two items with no key, and `else_` is the name of a Rust
field. The semantic model's YAML writes `case: {whens: [{when:, then:}], else:}`.

Today an `ExpressionNode` leaf may be "a `Parameter` placeholder". A
`Parameter` is "A typed placeholder left in a canonical expression for a value
knowable only once a request exists" (`CONTEXT.md`), and compile writes none
today (ADR 0071).

Two other points wait on the stored form: how reading a manifest finds a call
in the registry (ADR 0087, rule 8), and `CONTEXT.md` **Manifest version**,
which is moved "when the artifact's shape changes".

How is the tree written as JSON, what does a manifest entry hold, how does it
reference another entry, and how is it read back?

## Decision Drivers

- `semstrait-sql` owns the tree and its serialization. The manifest
  re-implements neither.
- `semstrait-sql`'s `Expr` does not change for one consumer. `L` keeps its
  meaning, the leaves that a tree accepts, and `A` keeps its meaning, whether a
  tree may hold an aggregate.
- An entry follows one expression of the semantic model, not one operator of
  the tree. The human: "inline descisions in expr is driven rather by
  signature then by decomposing each and every expression", and "if there is a
  signature - expr references another expr, otherwise - it is inlined".
- The expression id follows the entity id's scheme. The human: "I bielieve it
  was previously defined hash, not u128 ( something like uuid5, cannot recall -
  it is everywhere used over entity id, so we can follow the same )".
- A name says "expr", not "node". The human: "I like "expr" instead of "node"
  everywhere to be honest".
- The manifest stores no value that only a request decides.
- Compile's assembly of additivity does not change.
- The tree's JSON form is typed by each node's kind. The human: "I think serde
  form might more typed and driven by polymorphic body types - like right now
  differenciated functions operators, less generic for each".
- A call's body holds only the fields that its function type takes. The human:
  "I think we need to check against functions , understand what is each
  function type takes and drive by this, empty fields can be skipped ( default
  empty ). each type of function ( window, default aggregate, scalar ) can have
  different set of fields availible".
- A CASE has one shape in YAML and in JSON, holds one ELSE at most, and nests.
  The human: "else can be only one", "is this allows nesting case statement
  one to another, just to make sure that it is preserved", and "maybe it is
  better to just restore original form".

## Considered Options

- **An entry per expression of the semantic model, which holds the whole tree
  and a link to each other signature's expression on the same semantic source
  that it references, with `semstrait-sql`'s `Expr` over a link leaf that the
  manifest owns** (chosen). A filter on a Joinset holds ADR 0077's copy for
  each operand, and no link.
- **One entry per node, each child a link**, ADR 0065's node store. Refused:
  one expression is spread over many entries, 38 for the golden's 14 trees, and
  a reference to another semantics is still a copy of its tree.
- **One entry per expression of the semantic model, with each referenced tree
  copied into it**, as ADR 0037 and ADR 0065 inline a reference. Refused: each
  tree that references a semantics holds its own copy of that semantics' tree,
  so `zone`'s entry holds `region`'s cast and column again. The human:
  "reference is only used when expression attached to signature, filter, or
  some entity expressed in model ( meaningully ) otherwise it is inlined".
- **A child parameter on `semstrait-sql`'s `Expr`**: a child is a `Box<Expr>`
  in a nested tree and an id in the manifest. Refused: every `Expr` in every
  crate carries a third generic, and `semstrait-sql`'s tree must know that a
  child can be held another way, which is the manifest's concern.
- **A manifest mirror enum**: `ExpressionNode` stays, with id children, and
  imports every other type from `semstrait-sql`. Refused: the manifest still
  re-implements the list of variants and its serialization. The human:
  "ExpressionNode feels redudant in current form".
- For the expression id: **a number in the order in which compile emits each
  entry**, or **a hash of 64 bits**. Refused: an order number moves each later
  id where one entry is added, and neither is the entity id's scheme, which
  the human named: "it is everywhere used over entity id, so we can follow the
  same".
- For the JSON form: **today's derived form**, with each leaf under a `leaf`
  tag and each call body written in full, or **one call body for every function
  type**. Refused: a call then writes `"distinct":false,"filter":null` where it
  holds no clause, or `distinct`, `filter` and `over` that its function type
  cannot take.
- For a CASE's JSON: **a list with a last `{"else":…}` item**,
  `[{"when":…,"then":…}, …, {"else":…}]`, the JSON of the YAML list that ADR
  0089 refuses (*Considered Options*); **Calcite's operands**, `[<when>,
  <then>, …, <else>]`, as its `RelJson` writes a CASE call; or **today's JSON
  pairs and `else_`**. Refused: the list reads as if it held any number of ELSE
  items, and only a rule over positions keeps it to one. In Calcite's operands,
  the place of a value decides whether it is a WHEN, a THEN or the ELSE. A pair
  has no keys, and `else_` is a Rust field's name.
- For the stored leaf: **`Parameter` beside the link**. Refused: a
  `Parameter`'s value is knowable only once a request exists, so the manifest
  cannot hold it. The human: "might be decided only runtime during query
  planning, and cannot be stored or computed in manifest".
- For a link's JSON: **inside the tree's leaf tag**,
  `{"type":"leaf","content":{"type":"node","content":"…"}}`, or **under a tag
  of its own**, `{"type":"ref","content":"…"}`. Refused: each link becomes an
  object where its id alone says what it is. The human: "bare id seems more
  clean".
- For the dialect field: **a `Canonical` value in `Dialect`**. Refused:
  `CONTEXT.md` **Dialect** is "One engine's or one format's own vocabulary of
  functions, operation nodes and data types", and a function could then hold a
  dialect declaration for `Canonical`, which means nothing.

## Decision Outcome

Chosen option: one JSON form typed by each node's kind, and an entry per
expression of the semantic model that holds the whole tree, as
`semstrait-sql`'s `Expr` over a link leaf that the manifest owns, because with
it a call's body holds only the fields that its function type takes,
`semstrait-sql`'s `Expr` does not change, the manifest re-implements no part of
the tree, an entry holds one expression of the semantic model and copies no
other signature's tree, the expression id is written as an entity id is, and
the manifest holds no value that only a request decides.

In short:

- Each node but a leaf is `{"type": <tag>, "content": <body>}`. A call's tag is
  its function type, and its body holds only that type's fields. A CASE's body
  has the YAML's own shape: its WHENs, each with its THEN, under `whens`, and
  its ELSE.
- A leaf is written in its own form, and a link as its expression id.
- An entry holds the whole tree of one expression of the semantic model: a
  signature's expression or filter, or a Joinset's condition or residual. In it, a
  reference to another signature on the same semantic source is a link to that
  signature's entry, and every other part is inlined. A filter on a Joinset
  keeps ADR 0077's copy for each operand, which holds no link.
- A stored expression is `semstrait-sql`'s `Expr` over the manifest's leaf, a
  link, and it holds no `Parameter`. `ExpressionNode` retires, and the entry's
  field is `expr`.
- An entry's `layer` counts each link as the layer of the entry that it links.
- The expression id is a v5 UUID from the owner and the root marker, written
  as UUID text, as an `EntityId` is.
- `validate()` checks that each link names an entry of the same semantic
  source, that no link makes a cycle, and that no entry is an orphan. Reading
  follows each link, and joins a linked Measure's or Metric's filter as ADR
  0065 has it.
- The manifest names its dialect, `canonical` for now, and reading a manifest
  finds each call by its name in that dialect. `validate()` checks the rest.
- Compile's assembly of additivity does not change.
- `MANIFEST_VERSION` does not move, and the manifest records no catalog
  version.

### The tree has one JSON form, typed by each node's kind

1. `semstrait-sql` writes and reads the tree's JSON form, because it owns the
   tree's serialization. Each node but a leaf is
   `{"type": <tag>, "content": <body>}`. `Literal`, `Column` and each
   operation node keep their own tag, and their body holds their own fields:
   `{"type":"column","content":["ps0","amount"]}`,
   `{"type":"binary_op","content":{"op":"subtract","left":…,"right":…}}`.

   - A leaf is written in its own form, with no tag of the tree. A manifest's
     link is its expression id, in UUID text (rule 13):
     `"left":"15184182-69e5-55a9-9292-b09b5731f128"`. The human: "bare id seems
     more clean".
   - A reader reads an object whose `type` is a node's tag as that node, and
     any other value as the leaf. So no leaf's form holds a node's tag, and a
     broken node's error names the field, as today.
   - A CASE's body holds its WHENs, each with its THEN, under `whens`, and its
     ELSE under `else`, which is not written where the CASE has none:
     `{"type":"case","content":{"whens":[{"when":…,"then":…}],"else":…}}`.
     This is the YAML's own shape, `case: {whens: [{when:, then:}, …], else:
     …}`, with the same keys (ADR 0089, rule 7). Each key holds one form of
     value, as in Substrait's `IfThen` (`ifs: [{if, then}]`, `else`) and
     DataFusion's `CaseNode` (`when_then_expr: [{when_expr, then_expr}]`,
     `else_expr`), so a JSON Schema types each key, and `else` holds one tree,
     not a list. A CASE nests in any `when`, `then` or `else`. Reading refuses a
     CASE with no WHEN, as today.

2. A function call's tag is its function type, `scalar`, `aggregate` or
   `window`, and its body holds only the fields that its function type takes:

   | Tag | Body fields |
   |---|---|
   | `scalar` | `name`, `args`, `options` |
   | `aggregate` | `name`, `args`, `options`, `distinct`, `filter`, `order_by`, `nulls` |
   | `window` | `name`, `args`, `options`, `nulls`, `over` |

   `sum(amount)` is
   `{"type":"aggregate","content":{"name":"sum","args":[{"type":"column","content":["ps0","amount"]}]}}`.

   - `name` is the name of the function's declaration in the dialect that the
     document names (rule 8).
   - A field at its default value is not written: `false`, `null`, an empty
     list, no options.
   - A call writes a clause only where its declaration marks that clause
     `Allowed` or `Required` (ADR 0091, rule 4). Reading a manifest refuses a
     tag that is not the declaration's function type, and a clause that the
     declaration marks `Refused` or `NoEffect` (rule 9). In the catalog,
     `order_by` is written only
     for `string_agg` and for `percentile_cont`'s WITHIN GROUP (ADR 0087, rule
     17), and `nulls` only for a window function that takes IGNORE NULLS (ADR
     0087, rule 19).
   - On those two aggregates, the human: "it is special function, and in such
     scenario order by more of input argument, rather then expression
     extension or option". `AggregateCall` still holds that order in its
     `order_by` clause (ADR 0091, rule 1).

3. A call's options are written under their names, each with its value's
   name: `"options":{"null_handling":"IGNORE_NULLS"}`. A call that takes the
   canonical values writes no `options` (ADR 0091, rule 1). In the manifest,
   `validate()` refuses an option name, or a value, that the declaration does
   not have (rule 9).

### A stored expression is `semstrait-sql`'s `Expr` over the manifest's link

4. An entry holds one stored expression, a tree of `semstrait-sql`:

   ```rust
   // semstrait-manifest
   pub struct ManifestExpression {
       pub id: PhysicalExprId,
       pub expr: StoredExpr,
       pub layer: ExprLayer,
       pub physical_sources: BTreeSet<EntityId>,
   }

   /// One whole tree, with a link where it references another entry (rule 5).
   pub struct StoredExpr(Expr<Link>);       // semstrait-sql's Expr, with A = AggregateAllowed

   /// A reference to another entry: the expression of a signature on the same semantic source.
   pub struct Link(pub PhysicalExprId);
   ```

   - `ExpressionNode` retires. The manifest declares no variant of the tree.
     The human: "It can base ExpressionNode of semstrait-sql Expr, where it
     just constructing expr and links that with id ( same way as it is ). So it
     just a wrapper on top of what Expr declares in semstrait-sql".
   - Each name says "expr", not "node": the entry's field `expr`, the type
     `StoredExpr` and its `links()`, and the mint `mint_expr_id` (rule 13).
   - `Expr` does not change. A stored expression's `L` is `Link`: the only leaf
     that it accepts is a link to another entry. ADR 0085's rule 2 names it.
   - `PhysicalExprId` and `ExprLayer` are the manifest's own types (ADR 0090,
     rules 4 and 5).
   - A stored expression holds no `Parameter`. Only a request decides a
     `Parameter`'s value, for example a window's partition keys, so the
     manifest cannot hold it. With each link followed (rule 15), a tree is the
     plan's `PhysicalExpr = Expr<Parameter>` (ADR 0085, rule 2), with no
     `Parameter` in it. A link that reading keeps (rule 15) is a leaf of the
     plan's tree that names an output field of the node below (ADR 0075).

5. An entry holds the whole tree of one expression of the semantic model. The
   human: "so basically reference is only used when expression attached to
   signature, filter, or some entity expressed in model ( meaningully )
   otherwise it is inlined". These are the entries, and no other entry exists:

   | Entry | Its owner | Its root marker |
   |---|---|---|
   | a signature's `expression_id`: a Dimension's, a Measure's or a Metric's tree, or a named Filter's body | the signature | `signature_expression` |
   | a signature's `filter_expr_id`: a Measure's or a Metric's `filter:` | the signature | `signature_filter` |
   | a Joinset source's condition | the Joinset source | `joinset_on` |
   | a Joinset source's residual `filter:` | the Joinset source | `joinset_filter` |

   In an entry, operators, function calls, CASE, casts, literals and columns
   are inlined. A link stands where the authored tree references another
   semantics, at the place where ADR 0065 has compile substitute "*that same
   source's* tree for the referenced name": compile writes a link to that
   signature's expression entry there, but for a filter on a Joinset, as the
   bullets below say.

   | The reference | In | Becomes |
   |---|---|---|
   | a Dimension | a Dimension's, a Measure's or a Metric's `expr:` | a link to its expression entry |
   | a Measure or a Metric | a Metric's `expr:` | a link to its expression entry |
   | a Dimension | a Measure's or a Metric's `filter:` | a link to its expression entry |
   | a Dimension or a named Filter | a named Filter's body | a link to its expression entry |

   - A row-level Metric with no `agg:` holds its `filter:` in its own tree, as
     `CASE WHEN <filter> THEN <formula> END`, and has no filter entry (ADR 0074,
     *A Metric*).
   - No `expr:` and no `filter:` of a Dimension, a Measure or a Metric links a
     named Filter: compile refuses one that reads a named Filter (ADR 0071,
     *Placement is read from the role*).
   - A link names an entry of the same semantic source, never another
     source's: on a composition, a link names the composition's own signature.
   - A composed signature's column reads its operand's output (ADR 0065), and
     a Joinset's condition and residual read the operands' output fields (ADR
     0075). Neither references a signature of its own source, so each holds
     columns and no link.
   - A Measure that compile lowers on a Joinset (ADR 0077, *A constituent
     serves only the names it maps*) holds, for each name that its body reads,
     the tree of the Joinset's own signature for that name: its operand's
     output field, or ADR 0065's `Coalesce` where both operands map it. So its
     expression entry holds columns and no link, as rule 14 requires, and ADR
     0077's walk follows the qualifier of each value leaf. A Metric that compile
     lowers on a Joinset links the Joinset's own signatures (rule 14).
   - A filter that compile lowers on a Joinset, a composition's own filter for
     a Measure or a Metric or a named Filter's body, holds ADR 0077's copy for
     each operand (*A filter on a composition reads semantics, never a
     `Coalesce`*). Each copy reads its operand's columns, so the filter holds
     no link. A link to the Joinset's own Dimension would read its `Coalesce`
     (ADR 0074, *A Joinset*), which under `Or` and `Not` selects other rows
     than the filter held on each operand.
   - A Grainset holds a signature only for a Metric that it computes (ADR 0084,
     rule 8). Its tree links only entries of the Grainset itself and reads each
     other name as a column of the Grainset's own alias.
   - Compile types each link with the type of the entry that it links, as it
     types a substituted tree today. A link to a Metric whose `agg:` aggregates
     a formula that reads a Measure or a Metric is typed with the type of that
     Metric's value: its `data_type`, or the result type of its `agg:` over the
     formula's type where it declares none (ADR 0074, *A Metric*). Its fold reads a link to an entry whose
     tree is a literal, or whose tree is a link that reads as one, as that
     literal (ADR 0089, rule 23). A link to a Metric whose `agg:` aggregates a
     formula that reads a Measure or a Metric reads as no literal, and neither
     does a chain of links through one, whatever that Metric's tree is: its
     value is its second aggregate step, not its formula (rule 15).
   - Compile stores an entry after each entry that it links, in the order in
     which it lowers bodies, and it refuses a cycle of references on a source
     (ADR 0065).

6. `layer` and `physical_sources` stay on each entry. An entry's `layer` is the
   layer of its tree, in which each link counts as the layer of the entry that
   it links (`ExprLayer::of_node`, from the leaves up). In the tree of a
   Grainset's signature, a column of the Grainset's own alias reads the final
   value of the name that it names (ADR 0084, rule 8), and that value has the
   layer of its leaf: a Dimension is scalar; a Measure, or a Metric with its
   own `agg:`, is an aggregate; a Metric with no `agg:` stores as its child, so
   it is scalar when the child is scalar (a row-level Metric) and an aggregate
   otherwise. Every other column counts as scalar, as before. A tree over an
   aggregate is `post_aggregate`. A row-level Metric is never a signature of a
   Grainset. So a Metric whose tree
   subtracts two links to Measures is `post_aggregate`, as it is with a copy of
   each Measure's tree in it. The reason that ADR 0037 and ADR 0065 gave to
   refuse a handle leaf in their *Considered Options*, "aggregates below a
   handle-leaf are invisible" (ADR 0037) and "aggregates beneath a handle are
   invisible" (ADR 0065), does not hold for a link: the linked entry's `layer`
   shows them, so the layer is still one function of what the manifest stores.
   `physical_sources` names the physical sources that consume the owner's tree,
   as today.

7. Three examples in this ADR's form, written by hand from rules 1 to 6 and 13.
   They are the design's, not the output of a run. Each id of an entry that the
   golden holds is the one that rule 13's mint gives for its signature, and `…`
   shortens a long value or stands for the id of an entry that the golden does
   not hold.

   The Measure `revenue`, 4 entries in the golden today, is 1 entry:

   ```json
   {"id":"253641c7-819a-5b60-a3dc-c276bfceb36d","expr":{"type":"cast","content":{"input":{"type":"aggregate","content":{"name":"sum","args":[{"type":"cast","content":{"input":{"type":"column","content":["ps0","revenue"]},"target":"decimal(18,2)","on_failure":"error"}}]}},"target":"decimal(18,2)","on_failure":"error"}},"layer":"post_aggregate","physical_sources":["bf575591-…"]}
   ```

   A Dimension `zone` on `orders`, over the golden's Dimension `region`, is 1
   entry that links `region`'s entry, where it was 8 entries with a copy of
   `region`'s tree:

   ```yaml
   - name: zone
     data_type: string
     type: categorical
     expr:
       case:
         whens:
           - when: {in: {value: region, list: [{lit: "eu"}, {lit: "uk"}]}}
             then: {lit: "emea"}
         else: {lit: "other"}
   ```

   ```json
   {"id":"cb8b6c6b-b311-5484-9a64-5966edb7851d","expr":{"type":"cast","content":{"input":{"type":"column","content":["ps0","region"]},"target":"string","on_failure":"error"}},"layer":"scalar","physical_sources":["bf575591-…"]}
   {"id":"…","expr":{"type":"case","content":{"whens":[{"when":{"type":"in_list","content":{"value":"cb8b6c6b-b311-5484-9a64-5966edb7851d","list":[{"type":"literal","content":{"string":"eu"}},{"type":"literal","content":{"string":"uk"}}],"negated":false}},"then":{"type":"literal","content":{"string":"emea"}}}],"else":{"type":"literal","content":{"string":"other"}}}},"layer":"scalar","physical_sources":["bf575591-…"]}
   ```

   A Metric `items_net: {subtract: [item_count, short_tally]}` on `orders`,
   declared `integer`, over the golden's Measures `item_count` (`integer`) and
   `short_tally` (`smallint`), is 1 entry with two links. Each linked entry is
   `post_aggregate`, a cast over a `sum`, so the Metric's entry is too:

   ```json
   {"id":"…","expr":{"type":"binary_op","content":{"op":"subtract","left":"15184182-69e5-55a9-9292-b09b5731f128","right":"09712050-49b8-534f-98f1-34d64cf76e3c"}},"layer":"post_aggregate","physical_sources":["bf575591-…"]}
   ```

### The manifest names its dialect

8. The manifest has a root field `dialect`. It holds `canonical` or one
   dialect's name: `"dialect": "canonical"` today. `Dialect` holds no
   `canonical` value, and its variants are `DataFusion`, `Substrait` and `Iceberg`
   (ADR 0091, rule 6). For now, compile writes `canonical`, and reading a manifest
   refuses any other value. The human: "manifest registers dialect name -
   canonical for now, and search through name + dialect".

   Each call writes the name of its function's declaration in that dialect. For
   `canonical`, that is the canonical declaration's `name`, never an alias.

### Reading a manifest finds each call by its name in the manifest's dialect

9. Reading a manifest finds each call with the registry's lookup by name in the
   manifest's dialect (ADR 0087, rule 8). A manifest holds one dialect, and
   every call must match it. The human: "Only one dialect can be written in
   manifest", and "if it is not - error". Reading a manifest refuses what it
   needs to build a tree:

   - a `dialect` value other than `canonical`;
   - a call whose name the lookup does not find in the manifest's dialect,
     because without it no handle to the function's `static` exists (ADR 0091,
     rule 1);
   - a call whose tag is not the function type of the declaration it finds
     (rule 2);
   - a call whose arguments do not fit its signature (their number, a literal
     at an enum or a constant parameter, and an enum value that the
     declaration lists), a clause that the declaration marks `Refused` or
     `NoEffect`, a call with no clause that the declaration marks `Required`,
     and a window call whose `Over` ADR 0091 rule 2 refuses,
     because reading builds each call with `semstrait-sql`'s
     constructors and their check (ADR 0091, rule 2). A link stands only where
     the authored tree references another semantics (rule 5), and an enum or a
     constant parameter takes a literal (ADR 0089, rule 1), so reading checks a
     call as parse does (ADR 0089, rule 5);
   - a leaf that is not a link: a value that is not a node and not UUID text
     (rule 13).

   The manifest's validator, `validate()`, refuses the rest. ADR 0042 gives the
   crate that validator and no load path, and that stands:

   - an option name, or an option value, that the declaration does not have
     (rule 3);
   - a link, an entry or a `layer` that rule 14 refuses.

### Compile's assembly of additivity does not change

10. Compile's assembly of additivity does not change. A Metric's aggregate node
    reads its Measure's additivity through its link (ADR 0083, rule 3), and
    nothing is written on the node. The human: "if it is function level addivity - it can be internal
    on spec, if it is signature level - then signature defines addivity ( as it
    is belong semantic , on its level )", "let's leave additivity in same
    place", and "additivity of function declares possibility to have
    pre-aggregations - addivity from semstrait-model - more or less the same but
    more broader, with strategy selection and handling".

    - A function's additivity stays with its declarations (ADR 0091, rule 3:
      `AggregateFunction { declarations, additivity, distinct_additivity }`). It
      says whether a partial aggregate of the function may be combined again.
    - A Measure's additivity is broader, with its `axes`, `by` and semi-additive
      strategy. It stays on the Measure's definition, as today.

### What does not change

11. `MANIFEST_VERSION` stays `3`. The human: "manifest version can be
    untouched".
12. The manifest records no catalog version. The human: "no changes in catalog
    versions needed for now".

### The expression id is a v5 UUID, written as UUID text

13. The expression id, `PhysicalExprId`, is a v5 UUID, minted through
    `EXPR_NAMESPACE` from the entry's owner and its root marker (rule 5), with
    no path, because an entry holds one whole tree. One model compiled twice
    mints the same ids, as it mints the same entity ids (ADR 0051). Serde
    writes the id as UUID text, the 36-character lower-case form with hyphens
    of every `EntityId` (`mint_entity_id`):
    `"expression_id":"253641c7-819a-5b60-a3dc-c276bfceb36d"`. A link is the
    same text, and the store's map keys are too. The human: "it is everywhere
    used over entity id, so we can follow the same".

    - The id stays a type of its own, not an `EntityId` (ADR 0090, rule 4): an
      entry is one of its owner's trees, not a named entity, and
      `EXPR_NAMESPACE` is no entity's namespace.
    - One owner holds one tree under each root marker, so each entry's id is
      unique. Compile still refuses an id that it mints twice.
    - A link is a JSON string, so no value of the tree's JSON form is a number
      above `u64::MAX`. `semstrait-sql`'s reader reads the form with serde
      alone, whatever the order of an object's keys, and it reads no value as
      its JSON text first. `semstrait-sql` drops serde_json's `raw_value`
      feature.

### `validate()` checks each link

14. `validate()` refuses, in a decoded artifact or in one whose fields were
    changed after compile (ADR 0042):

    - a link that names no entry;
    - a link in a Joinset source's condition or residual, in the expression
      entry of a composition's Dimension or Measure signature, in the filter
      entry of a Joinset's signature, or in the expression
      entry of a Joinset's named Filter signature, which hold none (rule 5),
      and a link in a signature's entry that names an entry other than the
      `expression_id` of a signature that the same semantic source lists. A
      Joinset Metric's expression entry keeps its links to the Joinset's own
      signatures;
    - a cycle: an entry that its own links reach again;
    - an entry that no signature and no Joinset source names, as its
      expression, its filter, its condition or its residual (an orphan), and
      an entry that two of them name;
    - an entry whose `layer` is not the layer that rule 6 gives, where a column
      of a Grainset's own alias has the layer of the leaf that it names and
      every other column counts as scalar (ADR 0084, rule 8);
    - a signature's filter whose entry is its expression's entry, or whose
      `layer`, with each link counted, is not `scalar` (ADR 0065).

    Compile builds none of these, so their subject is such an artifact. An
    entry that many entries link is still its own owner's: a link reads the
    entry, and owns nothing.

### Reading follows each link

15. Reading an entry's tree through its links
    (`ManifestExpressions::reassemble`) puts in each link's place the tree of
    the entry that it links, read the same way, but for the links that the last
    two items below keep. Where the linked signature is a
    Measure's or a Metric's and has a filter, reading joins that filter with
    `And`, on the right, onto the aggregate calls of the linked tree, as ADR
    0065 rules for a Metric that references them (*A Metric inlining a filtered
    Measure carries that Measure's predicate on the inlined aggregate*): a
    Measure's filter onto its one aggregate call, and a Metric's filter onto
    each aggregate call that its tree reaches, after any filter already there.
    So the tree that reading gives is the tree that a copy of each referenced
    tree gave.

    - The reader finds the linked signature among the signatures that the same
      source lists (rule 14), by id alone. The manifest stores no reverse
      index (ADR 0055).
    - A Metric `net = revenue - refunds`, where `revenue` has `filter: {eq:
      [status, {lit: "paid"}]}`, stores `Link(revenue) - Link(refunds)`, and
      reads as `sum(…) FILTER (WHERE status = 'paid') - sum(…)`.
    - Where the reader is a projection above an aggregate step (ADR 0074, *A
      Metric*), reading keeps each link to a Dimension, or to a row-level
      Metric with no `agg:`, that the step groups by, and puts no tree in its
      place: the projection reads the step's output field for it. Below the
      step, in the row-level projection, reading puts the linked tree in such a
      link's place, as the first sentence of this rule says.
    - Reading keeps a link to a Metric whose `agg:` aggregates a formula that
      reads a Measure or a Metric, and puts no tree in its place. Such a
      Metric's tree holds no call of its `agg:` (ADR 0089, rule 20), and it
      keeps a link to a Measure or a Metric (ADR 0089, rule 23). That
      Metric's value is the second aggregate step over its formula's values,
      which the planner builds (ADR 0074, *A Metric*), not its formula: a
      Metric `revenue_per_region = revenue / regions_with_revenue` reads the
      count of the regions, not the formula that it counts. A filter that the
      reader carries narrows the step-1 aggregates of the linked Metric's
      subtree, after that Metric's own, and removes none of its groups. The
      reader reads that Metric's value as the output field of its step 2, after
      its cast, which the join of the two subtrees passes up (ADR 0074, *A
      Metric*).

### Amends

- **0002:** in its *Decision Outcome*, "The derived adjacently-tagged serde
  form is machine transport only" becomes: the tree's JSON form, which
  `semstrait-sql` writes and reads, in which a call's tag is its function type
  and a leaf is written in its own form (ADR 0088, rules 1 to 3), is machine
  transport only. The rest of 0002 stands.
- **0037:** in its *Decision Outcome*, "Nested semantic dependencies are
  **inlined** into one tree rather than referenced" becomes: a reference to
  another signature on the same semantic source is a link to that signature's
  entry where ADR 0088 rule 5 writes one, and a Metric over Measures is
  `PostAggregate` (ADR 0088, rule 6). The reason that the rest of that sentence
  gives, "an opaque handle-leaf would hide its aggregates", and the reason that
  *Considered Options* gives against a DAG do not hold for a link: the linked
  entry's `layer` shows the aggregates, so the layer is still one function of
  what the manifest stores (ADR 0088, rule 6). In *Considered Options*, the
  chosen option's "inlined dependencies" are those links. The rest of 0037
  stands.
- **0065:** its decision "the expression store is a forest of positional nodes",
  in its title, is replaced by ADR 0088, rules 4 to 6 and 13 to 15, where no id
  is minted from content, and the sections that only apply it read as those
  rules and ADR 0088's entry for 0055 give: in *Context and Problem Statement*,
  "what shape does the persisted expression store take" and "the shape of every
  expression entry"; in *Considered Options*, the chosen option's "a forest of
  positional nodes" and the options "**Keeping ADR 0013's flat-tree,
  content-addressed pool**" and "**A DAG pool with shared subtrees**"; in
  *Decision Outcome*, the paragraph on the **node store**; the whole of *The
  pool is a node-store forest with positional ids*; in *A composed Measure's
  filter is the conjunction of the two views (D2)*, "so it stays disjoint from
  the projection it filters (positional ids under `(owner, marker, path)`)"; in
  *A Metric inlining a filtered Measure carries that Measure's predicate on the
  inlined aggregate*, the "disjoint" of "the disjoint, row-level
  second root"; in *One `filter:` predicate, and no `AggregationFilter`*, "the
  two trees stay disjoint"; in *What `validate()` gains, and the version move*,
  "the store being a forest — every child id resolves, no node has two parents,
  no node is an orphan"; in its own *Amends*, the ADR 0013 entry's "a node store
  of positional entries" and "computed from the subtree", the ADR 0037 entry's
  "the expressions are persisted as", the ADR 0055 entry's sentence on the
  premise that "no closure is needed at all", and the ADR 0044 entry's "nothing
  is interned" and "the orphan sweep closes over child ids"; and in
  *Consequences*, "**The artifact grows.**", where two sources over one model
  duplicate every entry, "two `Column` entries", which are two entries, each
  with its own `Column`, and "a positional id". In its decision "A semantics has
  one global definition and one signature per semantic source", in *Decision
  Outcome*, "exactly one complete, fully physical expression tree, an optional
  row-level filter tree" becomes: exactly one expression entry, whose tree is
  complete and fully physical once each link in it is followed, and an optional
  row-level filter entry (ADR 0088, rules 5 and 15). In *Considered Options* and
  in *A definition is the contract; a signature is the implementation* (**Why
  not a deferred choice**), the reason given against a handle leaf, that it
  hides the aggregates beneath it, does not hold for a link: the linked entry's
  `layer` shows them, and compile binds each link to an entry of its own source,
  so a link defers no resolution to a request (ADR 0088, rules 5 and 6). In
  *What a tree contains, and who builds it*, "a **metric**'s tree is its body
  with each referenced measure or metric substituted" becomes ADR 0088 rule 5's
  reading: the body with a link to *that same source's* entry for each reference
  where that rule writes one; and "it is what makes inlining (ADR 0037) cheaper
  than it looks: the walk happens once per compile, not once per request"
  becomes: compile walks each reference once, to a link, and a request follows
  each link by id (ADR 0088, rule 15). In *A Metric inlining a filtered Measure
  carries that Measure's predicate on the inlined aggregate*, the
  predicate is joined where the link is read, not spliced onto a copy (ADR 0088,
  rule 15). In *What `validate()` gains, and the version move*, "a filter tree
  that is row-level and disjoint from its expression tree" reads as ADR 0088
  rule 14 gives. In its own *Amends*, in the ADR 0037 entry, "Its inlining
  ruling and its complete-tree requirement are the reason this shape works"
  becomes: its complete-tree requirement holds once each link is followed (ADR
  0088, rule 15), and its inlining ruling gives way to links, as ADR 0088's
  entry for 0037 gives; and "no second lineage to keep" holds for a tree that
  links its dependencies, because each link is that edge, inside the tree. The
  rest of 0065 stands.
- **0055:** in *`model_checksum` is not a spelling this codebase has*, "The
  top-level block therefore ends as `manifest_version`, `created_at`,
  `updated_at`, `model_hash`, `model_id`" becomes: the top-level block also
  holds `dialect`. In *`depends_on` keeps the child direction, on expression
  ids*, the premise that "no closure is needed at all" for the orphan sweep,
  which ADR 0065 made void, holds: each entry is a root that one signature or
  one Joinset source names, so the sweep is one direct pass over those roots
  (ADR 0088, rule 14); and the sentence from "A **structural reference inside
  the tree** is indeed absent" to "breaks `classify`'s totality" becomes: a
  link is a structural reference inside the tree, and the layer stays total
  (ADR 0088, rules 5 and 6). The rest of 0055 stands.
- **0071:** in *Placement is read from the role*, "`ExprLayer` stays on every
  stored node" becomes: `ExprLayer` stays on every entry (ADR 0088, rule 6). In
  *A tree gives the computation, and the role gives the placement*,
  "Every signature tree is materialised once per request with
  `ManifestExpressions::reassemble`" holds, and `reassemble` reads each link as
  ADR 0088 rule 15 says.
  In *Aggregation-level conditions*, "A condition already on an inlined
  `Aggregate` node" becomes: the filter of a Measure or a Metric that a Metric
  links is joined where the link is read (ADR 0088, rule 15), never dropped and
  never applied twice; and "Compile applies the same rule when another Metric
  inlines it" becomes: reading applies the same rule where another Metric links
  it (ADR 0088, rule 15). The rest of 0071 stands.
### Consequences

- Good, because a call's body holds only the fields that its function type
  takes, so `upper(name)` writes no `distinct`, `filter` or `over`.
- Good, because the manifest holds no second copy of the tree. `semstrait-sql`
  writes and reads every node.
- Good, because an entry is one expression of the semantic model: the
  golden's 14 trees take 14 entries, not 38, and no entry copies another
  signature's tree.
- Good, because the expression id is written as an entity id is, and the
  reader of the JSON form needs no `raw_value`.
- Good, because a CASE's JSON names each of its parts.
- Bad, because a reader tells a leaf from a node by its JSON's shape, so a new
  leaf's form must not hold a node's tag. The gain is that a manifest's link is
  its bare expression id.
- Bad, because a reader that follows a link must join the linked signature's
  filter (rule 15): a reader that takes `revenue`'s entry alone, in `net`'s
  tree, reads `revenue` with no filter. The gain is that a Measure's tree and
  its filter are each stored once, in the Measure's own entries.
- Bad, because inside one frame a semantics whose tree calls a Volatile
  function is computed once for each reader that reaches it through a copy of
  the linked tree (rule 15: "the tree that a copy of each referenced tree
  gave"), so `CONTEXT.md` **Volatility**'s "Every reader of the semantics on
  that row reads that value" does not hold there: with Dimensions `a =
  random()` and `b = a * 2`, `select a, b` gives `b ≠ 2a` on every row. The
  planner returns `Advisory::VolatileCopied` beside the plan (ADR 0074,
  *Advisories and refusals*). The gain is that a Measure link stays a copy and
  the row-level projection stays one projection, where `random()` is the one
  Volatile function of the catalog (ADR 0087).
- Bad, because `MANIFEST_VERSION` stays `3` while the shape changes, against
  `CONTEXT.md` **Manifest version**. A reader of the old shape then fails on a
  new manifest with a parse error, not with a version refusal.
- Bad, because the manifest records no catalog version. A `concat` call that
  stores no option means `null_handling` = `ACCEPT_NULLS` (ADR 0087). If the
  catalog changes that canonical value, the same bytes mean something else, and
  nothing detects it.
- Neutral: the tree's transport form (ADR 0002) changes, for example
  `{"type": "leaf", "content": {"type": "column", …}}` becomes
  `{"type": "column", …}`, `{"type":"function_call", …}` becomes
  `{"type":"scalar", …}`, and a CASE's `[<when>, <then>]` pairs and `else_`
  become `{"when":…,"then":…}` items and `else`, the YAML's own shape.
- Neutral: every site that builds or matches `ExpressionNode` (44 uses in 8
  files) changes to `StoredExpr`, and every site that reads an entry's `node`
  reads `expr`. `CallCheck::StoredNode` retires, because reading checks a call
  as parse does (rule 9).
- Neutral: every expression id in the golden changes, each is written as UUID
  text, and the golden holds 14 entries.
- Neutral: the registry gains one lookup (ADR 0087, rule 8).
- Neutral: ADRs 0074, 0077 and 0083, ADR 0071 in *A tree gives the
  computation, and the role gives the placement*, and ADR 0068 in *The
  aggregate filter is a mirror, and `CASE WHEN` is a rewrite* call each
  aggregate call of a Metric's tree an inlined node or copy. Each reads as an
  aggregate call of the Metric's tree read through its links (rule 15), so
  0071's "a Metric's inlined aggregates have no definition in hand at all" and
  0074's "An inlined node carries no link to the Measure it came from" hold
  for it: in the tree that reading gives, the linked tree stands in each
  link's place. The planner reads each node's Measure's additivity as it follows
  the link, before the linked tree stands in its place (ADR 0083, rule 3). ADR 0068's splice of a filtered Measure's predicate "at
  substitution" reads as the join of that predicate where reading follows the
  link (rule 15), and the predicate still reaches the plan as
  `AggregateExpr.filter`; 0068 rules the adapter only. Those clauses are not
  amended for it.
- Neutral: `CONTEXT.md` **Semantic manifest**, **Physical expression**,
  **Semantic signature** and **Expression layer** state the entry, its links
  and the layer of a link.

## More Information

Decided on what the manifest persists for an expression on semstrait-sql's tree.
The reviews ruled a link's JSON
and a stored expression's leaf, that reading refuses
a `Refused` or `NoEffect` clause, and an entry per
expression of the semantic model with its links, "expr" for "node" and the id's
UUID text. The human's rulings of 2026-10-15 keep the
YAML form of a CASE, whose shape its JSON has. The human's rulings on a review
let the fold
follow a chain of links to a literal entry, and let reading refuse a window call
whose `Over` ADR 0091 rule 2 refuses. The human's rulings (2026-10-17) ruled that a
Metric's aggregate node reads its Measure's additivity through its link. The
Metric formula decision (2026-10-17) records
that no Dimension, Measure or Metric reads a named Filter, and its decisions of
2026-10-18 keep the link to a Metric whose `agg:` aggregates a formula that
reads a Measure or a Metric. The link-reading decision (2026-10-18) records
that a projection above an aggregate step keeps each link to a Dimension, or
to a row-level Metric with no `agg:`, that the step groups by, that a link that reading keeps is a leaf that names an
output field, and the Bad of a Volatile tree copied inside one frame. Its
research read Calcite and Substrait.
