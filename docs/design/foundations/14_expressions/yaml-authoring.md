---
id: 14-yaml-authoring
status: ratified
version: second refinement landed 2026-05-18; cascade rebases of 19/31/35 complete
impl: hand-rolled field-level authoring serde per item Y / V7 + A2; reserved tag catalog §6.4.1; parse-site dispatch §6.2
---

# YAML Authoring Surface

This leaf defines the YAML authoring surface for expressions — bare `Expr<L>` fields with hand-rolled field-level adapters, parse-site dispatch that produces `UnresolvedSemanticExpr` or `UnresolvedPhysicalExpr` per site, the reserved-tag catalog, bare-identifier resolution rules, and the reserved (deferred) inline DSL. The authoring surface uses friendly tags only; internal discriminants (`binary_op`, `type`, `content`) never appear in authored YAML or schemas. See [expr-structural-variants.md](expr-structural-variants.md) for structural variants and [site-shape-gates.md](site-shape-gates.md) for per-site requirements.

## 6. YAML Authoring Surface

### 6.1 The authoring carrier is `Expr<L>`

Every expression-bearing authoring field carries a **bare** `Expr<L>` directly — at the parse-site's expected type: `UnresolvedSemanticExpr` (= `Expr<UnresolvedSemanticLeaf>`) at semantic sites, or `UnresolvedPhysicalExpr` (= `Expr<UnresolvedPhysicalLeaf>`) at physical-mapping sites. There is **no separate `ExprSource<L>` wrapper enum**: the prior `ExprSource<L> = Inline(String) | Block(Expr<L>)` carrier is removed (doc 45, 2026-06-22). The author writes the **block form** (the friendly-tag tree of §6.4) or its **bare-leaf shorthand** (§6.5) — a top-level bare identifier such as `expr: revenue`, which tokenizes to the default leaf for the site (`Field` at a semantic site, `Column` at a physical-mapping site) and resolves at compile. A string-form **inline DSL** is reserved, not live (§6.3).

The author-facing serde and `schemars` are **hand-rolled at field level** (item Y / V7 as refined by A2, mechanism refined by doc 45): each expression field carries a `#[serde(with = "…")]` adapter whose `Deserialize` delegates to the friendly-tag block parser (`parse_block`) and whose `Serialize` delegates to the reverse (`expr_to_value`), plus a `#[schemars(schema_with = "…")]` adapter that projects the friendly schema. Both live in `semstrait-model`. These field adapters are what previously sat on the `ExprSource<L>` wrapper; deleting the wrapper moved the *attachment point* from the enum to the fields, leaving the emitted bytes unchanged. Letting a bare `Expr<L>` reach the author path **through its own derives** — i.e. deriving `#[serde(tag = "...", content = "...")]` or `#[derive(JsonSchema)]` *as the authoring projection* — remains **forbidden**, because it would emit the internal `type` / `content` / `binary_op` discriminants and Rust variant names into the public YAML and its schema. (The shared `Expr<L>` *does* carry a derived `#[serde(tag,content)]` impl, but that is the **machine transport** form `ir::PhysicalExpr` aliases for cross-process plan transport — never the author path; A2. The author path always goes through the field adapter, never the bare-`Expr<L>` derive.) The reserved-tag catalog of §6.4 is implemented as friendly-tag discrimination in the field adapter's deserialize path (structural variants) plus a function-table look-aside for non-reserved tags. **Invariant (V7, unchanged): `binary_op:` / `unary_op:` / `type:` / `content:` MUST NOT appear in authored YAML or its generated schema** — only friendly tags (`add:`, `eq:`, `case:`, `sum:`, …). Doc 45 refines V7's *mechanism* (wrapper → field adapter), not its invariant.

### 6.2 Parse-site dispatch

Parsing produces a typed result per site. Each expression field deserializes directly to its `Expr<L>` via the field adapter (§6.1): the adapter's `Deserialize` is the runtime parse, dispatched by the field's declared leaf family. The friendly-tag block parser `parse_block<L>` is the shared core; the leaf-family difference is which `LeafResolver` it tokenizes bare identifiers against:

```rust
// Illustrative core — one generic friendly-tag parser, leaf-family-parameterized.
// At a semantic site `L = UnresolvedSemanticLeaf` (bare ident → Field(name));
// at a physical-mapping site `L = UnresolvedPhysicalLeaf` (bare ident → Column(name),
// and semantic tags `field`/`dim`/`measure`/`metric`/`key` are rejected).
pub fn parse_block<L: LeafResolver>(value: &Value) -> Result<Expr<L>, ParseErrorKind>;
```

The site catalog — which authoring locations parse at which leaf family — is in §7. The owning crate for parse-site dispatch is `semstrait-model` (§9.3).

### 6.3 Inline DSL grammar (deferred; compounds parse-rejected)

The string-form **inline DSL** is **reserved, not live in v1** (doc 45, ruling A-leaf). The authoring surface in v1 is the block form (§6.4) plus the bare-leaf shorthand (§6.5). The two string lanes are handled at parse as follows:

- A **bare** top-level string — a single identifier token, e.g. `expr: revenue` — is **not** inline DSL: it tokenizes to the default leaf for the site (`Field` semantic / `Column` physical, §6.5) and resolves at compile, exactly as a nested bare identifier already does. This is live.
- A **compound** top-level string — one carrying whitespace or an operator/punctuation character (`+ - * / % = < > ( ) , & |` …), e.g. `expr: "revenue / order_count"` — is the inline-DSL shape, and it raises `ParseErrorKind::InlineExprUnsupported` **at parse**. The real inline parser is deferred; this compound-guard is the placeholder it later replaces. Authors rewrite a compound to block form (`{divide: [revenue, order_count]}`).

The guard keys on whitespace/operator characters (the inline-DSL signal), **not** the SR-8 identifier grammar — so legitimate dash-names and other tokenizer-accepted bare leaves still flow through to compile-time validation unchanged.

The eventual inline grammar (when the parser lands) is the SQL-shaped form sketched below; it is recorded here only as the deferred target, not as a v1 contract:

- **Identifiers**: `[A-Za-z_][A-Za-z0-9_]*`, resolved per parse site (§6.5).
- **Literals**: integer / float / single- or double-quoted string / boolean / `null`.
- **Operators**: `+`, `-`, `*`, `/`, `%`, comparison (`=`, `<>`, `<`, `<=`, `>`, `>=`), logical (`AND`, `OR`, `NOT`), unary negation, `IS NULL`, `IS NOT NULL`.
- **Function call form**: `name(arg1, arg2, ...)`; `name` resolves against `FunctionRegistry` at compile.
- **Parentheses** for grouping.

Forms the inline DSL would deliberately NOT accept (use Declarative block — these are the same omissions the block form covers today):

- `CAST(x AS Type)` — use `{cast: {...}}`
- `CASE WHEN ... THEN ... ELSE ... END` — use `{case: {...}}`
- `BETWEEN`, `IN`, `LIKE`, regex operators, `DATE_TRUNC` — use explicit tags
- Aggregations — use `{aggregate: {...}}` or carry via Measure `agg:` per `[18 §5.2](../18_entities.md)`
- `NULLIF` / `COALESCE` — use explicit tags

### 6.4 Declarative block tags

A Declarative block is a single-key map whose key is a **tag**:

1. **Reserved AST tags** map 1:1 to `Expr<L>` structural variants. The full catalog is in §6.4.1.
2. **Function-registry tags** — any non-reserved tag is looked up in `FunctionRegistry` (`14a`); on hit, the block parses as a shortcut for `FunctionCall { name: <tag>, args }` using the registry's declared arity / arg-shape. On miss, `ParseErrorKind::UnknownTag`.

This dispatch model means **every registered scalar function is authorable as a top-level tag** without bloating the parser or the AST.

#### 6.4.1 Reserved tag catalog

Each reserved tag maps 1:1 to an `Expr<L>` structural variant from §3.3 or to a leaf from §3.4 / §3.5. The leaf-tag names align exactly with the Rust DSL constructor names so YAML and Rust use the same vocabulary.

**Leaf tags** (authoring-surface constructors):

| Tag | Maps to | Site legality | Carries accessor? |
|---|---|---|---|
| `col` | `PhysicalLeaf::Column` (or `UnresolvedSemanticLeaf::Column` under `auto`) | both | no |
| `literal` | `PhysicalLeaf::Literal` / `UnresolvedSemanticLeaf::Literal` | both | no |
| `field` | `UnresolvedSemanticLeaf::Field` | semantic only | no |
| `dim` | `UnresolvedSemanticLeaf::Dimension` | semantic only | yes — optional `DimensionAccessor` |
| `measure` | `UnresolvedSemanticLeaf::Measure` | semantic only | yes — optional `MeasureAccessor` |
| `metric` | `UnresolvedSemanticLeaf::Metric` | semantic only | yes — optional `MetricAccessor` |
| `key` | `UnresolvedSemanticLeaf::Key` | semantic only | yes — optional `KeyAccessor` |

Short forms (no accessor):

```yaml
expr: { dim: region }
expr: { measure: revenue }
expr: { field: conversion_rate }
expr: { col: amount_cents }       # legal in SemanticExpr only under auto mapping (§8)
```

Long forms (with accessor):

```yaml
expr: { measure: { name: revenue, accessor: previous } }
expr: { dim: { name: order_date, accessor: { lag: 2 } } }
```

The Rust DSL mirrors these exactly:

```rust
expr: dim("region"),
expr: measure("revenue"),
expr: measure("revenue").previous(),
expr: dim("order_date").lag(2),
expr: col("amount_cents"),                      // under auto mapping only
```

**Structural tags** (per §3.3 variants):

| Tag | Structural variant |
|---|---|
| `binary_op` | `BinaryOp` |
| `unary_op` | `UnaryOp` |
| `function_call` | `FunctionCall` (explicit; the verbose form) |
| `cast` | `Cast` |
| `case` | `Case` |
| `in` | `InList` |
| `between` | `Between` |
| `like` | `Like` |
| `is_null` | `IsNull` |
| `coalesce` | `Coalesce` |
| `null_if` | `NullIf` |
| `aggregate` | `Aggregate` |

`Window` is intentionally **not** authorable — it is compile-emitted only (§3.3 note).

Reserved-tag collisions with function-registry registrations are rejected at registry seal time per `[14a](../14a_function_catalog.md)`.

**Site legality**. `dim` / `measure` / `metric` / `key` / `field` are rejected at physical-mapping sites (`semantic_mapping.<x>.expr:`) — a `PhysicalExpr` cannot reference semantics. `col` is legal in either site, but `UnresolvedSemanticLeaf::Column` is conditionally legal at compile per §8.

### 6.5 Bare-identifier resolution per site

Within Declarative form's bare-scalar slots — and at the top level (the bare-leaf shorthand of §6.1, e.g. `expr: revenue`; the reserved inline DSL of §6.3 would share the same bare-identifier rule when it lands):

- At a **semantic site** parse — bare identifier → `UnresolvedSemanticLeaf::Field(name)`. Equivalent to writing `{ field: name }` explicitly. Kind is resolved at compile by registry lookup.
- At a **physical-mapping site** parse — bare identifier → `PhysicalLeaf::Column(name)`. Equivalent to writing `{ col: name }` explicitly.

There is no sigil; the parse site supplies the context. To force a specific accessor when the bare-identifier default isn't what you want — e.g. when a column and a semantic share a name under `semantic_mapping: auto` (see §8) — use the explicit typed constructor:

- `{ col: revenue }` — unambiguously the physical column.
- `{ field: revenue }` — unambiguously the semantic at that name (kind resolved at compile).
- `{ measure: revenue }` — unambiguously the Measure named `revenue`; compile rejects with `KindMismatch` if `revenue` resolves to a Dimension / Metric / Key instead.

To force literal interpretation when an identifier collides with a reserved keyword or a literal-shaped name (e.g. `true` as a string column name), authors use the explicit `{ literal: ... }` tagged form.

## Testing Strategy

- **Parse coverage:** Reserved-tag catalog of §6.4.1 is exhaustively covered — every leaf tag and structural tag parses to the matching variant. Function-registry tags parse to `FunctionCall` with the correct arity/arg-shape.
- **Bare-identifier resolution:** Semantic sites parse bare identifiers to `Field(name)`. Physical-mapping sites parse bare identifiers to `Column(name)`.
- **Parse-site dispatch:** Semantic sites produce `UnresolvedSemanticExpr`. Physical-mapping sites produce `UnresolvedPhysicalExpr`. Semantic tags (`field`, `dim`, `measure`, `metric`, `key`) are rejected at physical-mapping sites.
- **Round-trip identity:** YAML → parse → serialize → YAML identity per doc 45 authoring carrier (modulo bare-leaf shorthand expansion to explicit tagged form).
- **Invariant enforcement:** `binary_op:` / `unary_op:` / `type:` / `content:` never appear in authored YAML or generated schemas (V7 invariant per §6.1).
- **Inline DSL guard:** Compound strings (whitespace or operator/punctuation characters) raise `InlineExprUnsupported` at parse. Bare-leaf strings (single identifier token) tokenize to `Field` / `Column` and resolve at compile.
- **Reserved-tag collision rejection:** FunctionRegistry seal rejects any function name matching a reserved tag.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
