---
id: 11-semantics-identity
status: ratified
version: 2.0
impl: implemented
---

# Semantics Identity

Global identity, unified namespace, per-level uniqueness, free introduction, and identifier grammar for Semantics names. This leaf owns the identity rules; [scope chain](scope-chain.md) owns the scope levels, [yaml declaration forms](yaml-declaration-forms.md) owns the authoring surface, and [lookup algorithm](lookup-algorithm.md) owns compile-time resolution.

## 3. Semantics Identity

### 3.1 Global identity rule

A Semantics **name** is a globally-unique identifier for exactly one logical Semantics in the Model. If the name `revenue` appears in multiple places — at Model root, in top-level DataKind X, in top-level DataKind Y — these are the same logical Semantics with (potentially) per-DataKind resolution variants. There is no mechanism for "two different things both called `revenue`."

This rule is the foundation of unqualified reference (`00 §4.1`): because every name identifies exactly one thing, no dot-qualified syntax is needed.

**Consequence for Tier-1 declarations.** A Tier-1 Model-root Semantics declaration is *one occurrence* of that Semantics; it does not own the Semantics exclusively. Any top-level DataKind that names the same Semantics is contributing another occurrence. All occurrences must agree on shape (§5).

### 3.2 Unified namespace

Within the single global namespace, no two Semantics may share a name **regardless of element type**. You cannot have a Dimension named `revenue` and a Measure named `revenue`; the name `revenue` identifies exactly one Semantics whose type (Dimension / Measure / Metric / Filter) is fixed by its declaration.

This is a tighter rule than legacy (which used per-type sub-namespaces). The motivation is simplicity of unqualified reference: a name gives the consumer and planner the full identity of the Semantics, including its element type.

### 3.3 Per-level uniqueness within a list

Within any single Semantics list — the top-level `dimensions:` / `measures:` / `metrics:` at Model root, OR a given DataKind's `dimensions:` / `measures:` / `metrics:` / `filters:` — a name may appear at most once. Duplicates within a list are a compile error.

This is weaker than "globally unique" (which covers the cross-level case) and exists to catch authoring mistakes. The same name MAY appear across multiple lists (e.g. once at Model root and once in DataKind X) — those are distinct occurrences contributing to the same logical Semantics.

**Example — legal (same name across lists):**

```yaml
# Model root
dimensions:
  - name: date
    data_type: date

datasets:
  - name: shopify
    dimensions:
      - name: date              # reference; contributes nothing new
    # ...
```

**Example — illegal (same name within one list):**

```yaml
datasets:
  - name: shopify
    dimensions:
      - name: date
      - name: date              # ← ERROR: duplicate in same list
        expr: created_at
```

### 3.4 Free introduction

A Semantics may first appear at any tier — Model root (Tier-1 shape declaration) or inside a top-level DataKind's Kind scope (first declaration there). YAML occurrence order does not determine "first": `compile` treats all occurrences as symmetric contributors to a single Semantics registry entry.

Authoring consequence: modelers write Semantics declarations wherever it reads best. There is no requirement to pre-declare at Model root before use.

## 13. Identifier Grammar

A valid identifier — for any Semantics name, DataKind name, Relationship name, or structural label — matches:

```
identifier  ::= [A-Za-z_] [A-Za-z0-9_]*
```

Rules:

- **Character set.** ASCII only. The first character is a letter or underscore; subsequent characters add digits. No hyphens, dots, spaces, or other punctuation.
- **Case sensitivity.** Identifiers are case-sensitive. `Revenue` and `revenue` are distinct (per §3's global identity, they would be two different Semantics). Authoring convention is lowercase snake_case; tooling MAY lint non-conforming casing but MUST NOT rewrite it.
- **Length.** No explicit upper bound. Bounded in practice by YAML key size (which in turn is bounded by parser memory).
- **Reserved.** None for initial design. The double-underscore prefix (`__`) is informally reserved for future semstrait-internal synthesized names (e.g. `__anon_joinset_3`); authors SHOULD avoid it, but `validate` does not currently reject it.
- **Non-exhaustive per I10.** Extension to broader Unicode identifiers (per UAX #31) is a forward-compatible relaxation — today's identifiers remain valid under any future widening.

Validation point: `N-V1` in `validate` (§12.1). Regex-level check only; no semantic inspection of the name.

**Note on collision with YAML special characters.** The regex deliberately excludes characters that force YAML quoting (`:`, `-`, `#`, …). This keeps `- revenue` scalar-shorthand (§4.2) unambiguous — any scalar that needs quoting cannot be a valid identifier, so the parser can fail fast rather than producing a surprising Semantics name.

## Testing Strategy

- Global identity (same name across lists is one logical Semantics, §3.1)
- Per-level uniqueness (N-V2: duplicate within one list is rejected)
- Identifier grammar validation (N-V1: ASCII `[A-Za-z_][A-Za-z0-9_]*`, reject empty/whitespace/punctuation)
- Collision with YAML special characters (grammar excludes `:`, `-`, `#` → unambiguous scalar-shorthand)
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
