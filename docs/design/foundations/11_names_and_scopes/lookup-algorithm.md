---
id: 11-lookup-algorithm
status: ratified
version: 2.0
impl: implemented
---

# Lookup Algorithm

Lookup algorithm — how compile resolves a named reference against the scope tree and global registry, producing indices for plan. This leaf owns the lookup mechanics; [semantics identity](semantics-identity.md) owns the identity rules, [cross kind reference rule](cross-kind-reference-rule.md) owns the Relationship-required invariant.

## 11. Lookup Algorithm

`compile`'s name-resolution pass produces the following indices in the `SemanticManifest`:

- A **global Semantics registry** keyed by name, carrying the unified shape plus the per-DataKind resolution-variant map.
- A **per-DataKind Semantics table** listing which Semantics are exposed by each top-level DataKind's interface.
- A **Relationship graph** of top-level DataKinds (edges = Relationships).
- A **per-Semantics binding table** listing, for each Semantics, the set of top-level DataKinds in whose Binding subtree it is materially bound (directly via column or via local `expr`).

### 11.1 Lookup steps

Given a reference to Semantics `S` from a scope `Sc` in top-level DataKind `K`:

1. **Name existence** — check the global Semantics registry. If `S` is not registered, `CompileError::UnknownReference`.
2. **Kind-scope membership** — check K's per-DataKind Semantics table. If `S` is not in K's interface, `CompileError::SemanticNotInInterface`.
3. **Binding resolution** — check the per-Semantics binding table:
   a. If `S` is bound in K's own Binding subtree, record a local-binding resolution.
   b. If not, search the Relationship graph for a path from K to a DataKind in which `S` is bound. Record the path.
   c. If no path exists, `CompileError::UnresolvedCrossKindReference`.
   d. If multiple paths exist and `16`'s disambiguation rules do not select a unique one, `CompileError::AmbiguousCrossKindPath`.
4. **Variant selection** — choose the DataKind-local `expr:` for `S` if K declares one; else the Tier-1 default if present; else bind directly to the physical column via `SemanticMapping` (15).

### 11.2 Expression-level references

Within an `expr` (from any Semantics in K's Kind scope), referenced names are Semantics names (or function names, ratified in `14`). Expression-level name resolution is a specialization of the algorithm above, scoped to K's interface. Functions are resolved against the `FunctionRegistry` (14).

### 11.3 Deterministic ordering

All SemanticManifest indices and all Diagnostic streams emitted by `11`'s lookup algorithm MUST be deterministic across runs of the same input. The ratified order:

| Index | Ordering rule |
|---|---|
| Global Semantics registry | lexicographic by Semantics `name` |
| Per-DataKind Semantics table | primary key: DataKind `name` (lex); secondary: Semantics `name` (lex) |
| Per-Semantics binding table | primary key: Semantics `name` (lex); secondary: binding DataKind `name` (lex) |
| Relationship graph | lexicographic by `(from_dataset, to_dataset)` pair |
| Diagnostic stream | primary: source-document location when present (by `(file, line, col)`); secondary: structural walk order of the Model root's declarations (Model-root Tier-1 blocks first, then top-level DataKinds in declaration order, then Relationships); tertiary: within-list index |

Rationale: deterministic output is required for stable test fixtures, stable SemanticManifest artifacts under snapshot-style CI, and I5-compatible caching. `BTreeMap`-backed indices inside the SemanticManifest make the Semantics / DataKind / path orderings structural rather than policy. Diagnostic ordering is policy because locations may be absent (context-free errors, per `10 §5.1`).

Structural walk order for Diagnostic streams is the tie-breaker when two Diagnostics share a `None` location — they sort by the lexical position of the declaration they arose from. Within-run determinism is guaranteed; cross-run stability across non-trivial Model edits is not (adding a Tier-1 declaration shifts structural positions).

### 11.4 Interaction with `ComposedSemanticInterface`

`ComposedSemanticInterface` (CSI; ratified in `16`) is the plan-time projection of one or more DataKinds' interfaces unified via `Relationship`s. Its interaction with `11`'s lookup algorithm:

- **Compile** does NOT materialize a CSI per Request (Requests do not exist at compile time). It DOES materialize the `Relationship` graph, the per-DataKind Semantics table, and the per-Semantics binding table — the three indices from which a CSI can be derived for any given Request.

- **Plan** constructs a CSI on the fly during Request processing. Given a Request's `from:` DataKind `K` and its referenced Semantics set `{S_1, ..., S_n}`, `plan` walks §11.1's algorithm per Semantics:
  - Semantics bound locally to `K` contribute directly.
  - Semantics requiring a Relationship walk contribute via a CSI that spans `K` + the reached DataKind(s).
  - The CSI's coverage is the union of the contributing interfaces; its shape matches `16`'s CSI specification.

- **Compile-time invariant**: every Semantics reference that COULD appear in a valid Request at plan time is pre-validated at compile (N-C3 + N-C5 + N-C6). The CSI that plan constructs at query time only involves Relationship walks that compile has already validated as path-existent and unambiguous.

In plain terms: `11`'s algorithm decides at compile **whether** a cross-kind resolution is structurally legal; CSI (16) is the data structure plan uses at query time to **realize** that resolution for a specific Request.

Forward-ref: CSI shape, field-coverage unification, and NULL-handling policy for partially-covered fields are all ratified in `16`. `11` is silent on those internals.

## Testing Strategy

- Lookup steps validation (§11.1: name existence → Kind-scope membership → Binding resolution → variant selection)
- Global Semantics registry (keyed by name; unified shape + per-DataKind resolution-variant map)
- Per-DataKind table (which Semantics exposed by each top-level DataKind's interface)
- Relationship graph (edges = Relationships between top-level DataKinds)
- Binding resolution (local vs. cross-kind path walk; §11.1 step 3)
- Variant selection (DataKind-local `expr:` vs. Tier-1 default vs. direct `SemanticMapping`; §11.1 step 4)
- Deterministic ordering (§11.3 table: all indices and Diagnostic streams lexicographic or structural-walk)
- `ComposedSemanticInterface` interaction (§11.4: compile validates, plan constructs CSI per Request)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
