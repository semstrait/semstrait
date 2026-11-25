---
id: 50-behavioral-contract-gates
status: proposed
version: 1.0
impl: none
---

# 50. Behavioral Contract Gates

> **Status: proposed (not ratified).** This document specifies a behavioral
> quality-gate suite. It is offered for ratification per the decide-first loop
> (`AGENTS.md §2`): establish intended behavior → ratify → implement. The
> registry / `STATUS.md` sync (per the removed `semstrait-doc` MAINTAINER
> procedure — this tree is now frozen, `AGENTS.md §2`) and the harness
> crate-code are **deliberately not yet applied** — they are the
> post-ratification steps gated in §12.

## 1. Purpose and Posture

`50` ratifies a suite of **behavioral, contract-like test gates** that prove the
quality of a feature (or a set of features) against what `semstrait` *aims to
be at its core*, not against what the current code happens to do.

The distinction is the whole point:

- A **regression test** asserts that today's code keeps doing what it does
  today. It is written *after* the behavior and *passes by construction*.
- A **contract gate** asserts the **ratified target behavior** (`docs/design/`).
  It may **fail against the current code on purpose** — current `crates/` are a
  mid-migration baseline (`00 §1`), so a gate that fails is telling you the gap
  between the baseline and the target. A gate that passes is a clause of the
  ideology that the code has actually earned.

These gates are therefore **executable specification**. They are the
end-to-end / integration tier (the wide-blast-radius behavioral tier), distinct
from the per-crate unit tests in `crates/*/src` (`TESTING.md`). Where a unit
test proves a function, a gate proves an **invariant of the pipeline**.

### 1.1 What a gate is allowed to assert

A gate asserts **only** behavior owned by a ratified `docs/design/` clause, and
**cites that clause** (the `Then` step or scenario carries the `NN §M` anchor).
A gate MUST NOT encode an undocumented implementation detail; if a gate has no
clause to cite, the missing clause is a spec defect to file first (`00 §4.4`).

This keeps the suite anti-drift: when a clause changes, the gates that cite it
are the change's blast radius.

### 1.2 Relationship to the existing root `tests/`

The current root `tests/` suite (`e2e_pipeline_test.rs`, …) targets the
**excluded** legacy crates (`semstrait-api` / `-planner` / `-adapter`) and the
pre-rework `CompiledManifest` / `entities` vocabulary (`Cargo.toml:9`). It is a
current-state MVP harness, not a target-state contract, and is **not** what `50`
governs. `50`'s gates bind to the **canonical** pipeline verbs and the
**ratified** vocabulary. The legacy suite is retired-in-place reference; its
migration is tracked separately (`docs/TECH_DEBT.md`).

## 2. Why BDD / cucumber-rs

The chosen mechanism is [`cucumber-rs`](https://cucumber-rs.github.io/cucumber/main/)
with Gherkin (`Given`/`When`/`Then`) feature files.

Rationale:

- **The contract is readable by humans, not just `rustc`.** A gate like *"a
  `unionset` may not nest a `unionset`"* (`26 §2.2`) reads identically to its
  spec clause. A reviewer ratifying the spec can ratify the gate.
- **Scenario Outlines are the natural shape of semstrait's matrices.** The
  nesting matrix (`26 §1`), the applicability matrix (`25`), the aggregation-type
  roster, the temporal-shape roster — each is a table of `(input → expected
  outcome)` rows. Gherkin `Scenario Outline` + `Examples` is exactly this.
- **The feature files are the ratifiable artifact.** They are language-agnostic;
  the Rust step glue is implementation. Per the spec-driven loop, we ratify the
  `.feature` files, then implement the glue.
- **Tags carry the gate lifecycle.** `@now` vs `@target` (see §5) lets one suite
  hold both currently-earned clauses and aspirational clauses without two
  codebases.

## 3. The Four Pillars

The suite is organized into four contract pillars, each mapping to one ratified
concern and one feature file.

| Pillar | Feature file | Asserts | Authoritative docs |
|---|---|---|---|
| **P1 — Nesting policy** | `nesting_policy.feature` | structural legality of the DataKind tree | `12`, `26` |
| **P2 — Schema shape** | `schema_shape.feature` | the shape/invariants of the compiled `SemanticManifest` | `33` |
| **P3 — Resolved / unresolved** | `resolved_unresolved.feature` | the authoring↔resolved expression boundary and the compile fold | `14`, `19`, `35 §3`/`§5` |
| **P4 — Pipeline end-result** | `pipeline_end_result.feature` | the terminal artifact of each stage (`SemanticManifest`, then `SemanticPlan`, then `EngineArtifact`) | `10`, `35 §9`/`§10`/`§13`, `36` |

A fifth cross-cutting concern — **determinism (I4)** — is tagged `@determinism`
and threaded through P2 and P4 rather than given its own file (§10).

## 4. Harness Architecture

### 4.1 Home and shape

The harness is a single test-only crate, **`crates/semstrait-contract-gates`**
(workspace member; `publish = false`), with:

```
crates/semstrait-contract-gates/
  Cargo.toml                # dev-deps: cucumber, tokio; deps: the buildable crates
  tests/
    gates.rs                # the cucumber runner (one #[tokio::main] entry)
    features/
      nesting_policy.feature
      schema_shape.feature
      resolved_unresolved.feature
      pipeline_end_result.feature
    steps/                  # step-definition modules (Rust glue)
      world.rs
      given.rs
      when.rs
      then.rs
    fixtures/               # YAML models referenced by name from Gherkin
```

> The feature files in this specification ship at `tests/features/` at the repo
> root for ratification review; on implementation they move under the harness
> crate. The crate-code creation is a §12 gate.

The crate depends **only on the buildable workspace subset**
(`semstrait-common`, `-model`, `-ir`, `-manifest`, `-catalog`). Pillar gates
that need `-planner` / `-adapter` (P4's plan/adapt stages) are tagged `@target`
and skipped until those crates re-enter the workspace (§12). This keeps the
suite green-or-known on `make test` today.

### 4.2 The `World`

cucumber's `World` is the per-scenario state machine. It holds the inputs and
the *outcomes* of each pipeline stage so a `Then` can inspect any of them:

```rust
#[derive(Debug, Default, cucumber::World)]
struct GateWorld {
    // Inputs
    model_yaml: Option<String>,

    // Stage outcomes (each an `Option<Result<_, _>>` — a step may assert
    // either the Ok shape or the typed error kind).
    parsed:   Option<Result<SemanticModel, Vec<ParseError>>>,
    validated: Option<Result<(), Vec<ValidateErrorKind>>>,
    compiled:  Option<Result<SemanticManifest, CompileError>>,
    // P4 @target — populated only when planner/adapter are in-workspace:
    // planned:  Option<Result<SemanticPlan, PlanError>>,
    // artifact: Option<Result<EngineArtifact, AdaptError>>,
}
```

The `World` binds to the **real canonical entry points** — no shims, no
re-implementation:

| Stage | Entry point (today) | Source |
|---|---|---|
| `parse` | `semstrait_model::parse(&str)` | `model/src/lib.rs` |
| `validate` | `semstrait_model::validate(&SemanticModel)` | `model/src/lib.rs` |
| `compile` | `semstrait_manifest::compile(...)` (async) | `manifest/src/lib.rs` |
| `plan` | `semstrait_planner::…` (`@target`) | excluded crate |
| `adapt` | `semstrait_adapter::…` (`@target`) | excluded crate |

Binding to the public surface means a gate breaks if the public contract drifts
— which is the desired signal.

### 4.3 Step library

Steps are reusable across all four feature files. The canonical roster:

**Given** (set up the input model)
- `Given the reference model` — loads `crates/semstrait-model/schemas/reference.yaml`.
- `Given a model fixture {string}` — loads `fixtures/{name}.yaml`.
- `Given the model:` (docstring) — an inline YAML model in the scenario body.

**When** (drive the pipeline; each records the outcome, never panics)
- `When I parse it`
- `When I validate it` (parse-then-validate)
- `When I compile it` (parse-then-validate-then-compile, with an in-memory
  catalog/filesystem from the fixture; §4.4)
- `When I compile it twice` (`@determinism`)

**Then** (assert the outcome against a cited clause)
- `Then parsing fails with {error_kind}` / `Then it parses successfully`
- `Then validation rejects it with {error_kind}`
- `Then compilation fails with {error_kind}`
- `Then the manifest <assertion>` (P2/P4 — structural assertions on the
  `SemanticManifest`)
- `Then the resolved expression <assertion>` (P3)
- `Then the two manifests are byte-identical` (`@determinism`)

`{error_kind}` matches the typed `*ErrorKind` variant name (e.g.
`parse.illegal-self-nesting`, `validate.complex-data-kind-insufficient-children`)
— the kebab-case diagnostic code (`12 §7.4`, `26 §5`), so the Gherkin reads as
the spec's diagnostic code, and the step maps it to the Rust variant.

### 4.4 Catalog / filesystem doubles

`compile` requires `impl CatalogProvider` + `impl FileSystem` (`10 §3.3`). The
harness supplies **deterministic in-memory doubles** seeded from the fixture
(or a sidecar `*.catalog.yaml`), so compile is hermetic and I4-checkable. A
fixture that exercises catalog-driven schema resolution declares its catalog
columns inline; a fixture that is catalog-free (model-as-truth, `15 §2.5`) uses
the empty double.

## 5. Gate Lifecycle and Tags

Every scenario carries exactly one **lifecycle tag** and any number of
**pillar/topic tags**.

### 5.1 Lifecycle tags

| Tag | Meaning | CI behavior |
|---|---|---|
| `@now` | The clause is expected to hold against the **current buildable code**. | Hard gate. A failure breaks `make test`. |
| `@target` | The clause is ratified target state but **not yet earned** by the code (needs item-Y crate-code, planner port, or an unimplemented compile rule). | Run but **allowed to fail** (xfail). A `@target` gate that *passes* triggers a "promote me to `@now`" notice (§5.3). |
| `@determinism` | Cross-cutting I4 byte-identity gate (`33 §9.1`). | Runs under `@now` or `@target` as appropriate. |

`@target` is what makes this a *quality gate toward the ideology* rather than a
snapshot: the aspirational clauses are committed, executable, and visible — they
just don't yet block. As the code converges, gates migrate `@target → @now`,
and the suite's `@now` fraction is a literal progress meter toward the target.

### 5.2 Topic tags

`@nesting`, `@schema`, `@resolution`, `@pipeline`, plus fine-grained anchors
mirroring spec rule IDs: `@R1 @R2 @R3` (nesting rules), `@I1 @I4 @I5`
(invariants), `@CX1` (load integrity), `@V7` (friendly-tag serde). A reviewer
can run "all I4 gates" with `cargo test … -- --tags @I4`.

### 5.3 Promotion discipline

The runner is configured so a **passing `@target` scenario is reported, not
silently green**. When a `@target` gate starts passing, the change author
promotes it to `@now` in the same PR (the code has earned the clause). This is
the ratcheting mechanism: the target floor only ever rises.

### 5.4 What a gate failure means

- `@now` failure → a **regression or an unmet contract on supposedly-done code**
  → block.
- `@target` failure → **expected**; the gap is not yet closed → informational.
- A gate that cannot cite a clause → **not a valid gate** → remove or file the
  missing spec clause.

## 6. Pillar 1 — Nesting Policy (`@nesting`)

Source of truth: `12` (policy) + `26` (matrix). These gates prove the structural
skeleton every `SemanticModel` must obey *before* any semantic resolution.

### 6.1 Contract clauses gated

- **Matrix legality (`26 §1`).** Parent×child admissibility. A `Scenario
  Outline` over the matrix cells. Self-nesting diagonal cells are R2 rejections.
- **R1 — leaves never nest (`26 §2.1`).** A `dataset` with any child container
  → `parse.unknown-field`.
- **R2 — no same-variant self-nesting (`26 §2.2`).** `unionsets:` under a
  unionset (etc.) → `parse.illegal-self-nesting`.
- **R3 — composers need ≥ 2 children (`26 §2.3`).** A 0- or 1-child composer →
  `validate.complex-data-kind-insufficient-children`. Counting excludes
  `relationships:` (`26 §2.3`).
- **Nested-form structural-only (`26 §3`).** A nested DataKind carrying
  `dimensions:`/`measures:`/…/`description:`/`ai_context:` →
  `parse.nested-data-kind-carries-interface` (or `parse.unknown-field`).
- **Depth bound (`12 §2.3`).** The longest legal chain is a permutation of the
  three composer variants over a leaf; same-kind nesting at any depth is
  rejected.
- **Grainset structural rules (`12 §4`, `26 §6`).** ≥ 2 levels; coarsest-first
  ordering (`validate.grainset-levels-not-sorted-coarsest-first`); every leaf
  child authors its own `extras.temporal.grain:` (`validate.grainset-child-grain-required`).
- **Joinset binary-v1 (`12 §5.3`).** Exactly 2 members → `JoinsetMustBeBinaryV1`.
- **Stage placement.** R1/R2 fail at `parse` (type-level / `deny_unknown_fields`);
  R3 + ordering + interface-at-nested-scope fail at `validate`; grain-axis and
  relationship-endpoint resolution fail at `compile` (`12 §7`).

### 6.2 Example scenario

```gherkin
@nesting @R2 @now
Scenario: A unionset may not nest a unionset
  Given the model:
    """
    semantic_model:
      name: m
      dimensions: [{ name: d, data_type: string, type: categorical }]
      unionsets:
        - name: outer
          dimensions: [{ ref: d }]
          unionsets:            # R2 violation
            - name: inner
    """
  When I parse it
  Then parsing fails with parse.illegal-self-nesting
```

## 7. Pillar 2 — Schema Shape (`@schema`)

Source of truth: `33`. These gates prove the **shape and invariants of the
compiled `SemanticManifest`** — the boundary artifact between compile-time and
query-time (`10 §7`).

### 7.1 Contract clauses gated

- **Single identity lane (`33 §2`/`§4`).** Every entity registry is
  `BTreeMap<EntityId, _>`; every entity carries `id == its map key`; every
  cross-reference is an `EntityId`. No `stable_ids` side-map; no per-kind id
  newtypes. (`@CX1` re-checks `id == key`.)
- **`model_id` (`33 §4.3.1`).** Present, deterministically minted, globally
  unique alongside all entity ids.
- **The lone integer (`33 §5`).** `bit_position: u32` is the only integer in the
  bitmap subsystem; `EntityId` never participates in bit math; the
  `bit_position → EntityId` reverse index is **not** persisted.
- **Physical-only expression pool (`33 §7.2`).** `ManifestExpressions.physical:
  BTreeMap<PhysicalExprId, ManifestExpression { expr, layer }>`. No semantic
  pool, no `SemanticExprId`, no `UnresolvedSemanticExpr` persisted, no sugar, no
  additivity stored.
- **Bindings (`33 §7.1`).** `SemanticBinding` attaches to leaf `Dataset`s only;
  `mapping` and `resolved` keysets coincide exactly; `Expr`-realized semantics
  reference the pool by `PhysicalExprId`.
- **Resolved relationships (`33 §8A`).** `ResolvedRelationship` with materialized
  `optional`/`cross_filter`, derived `join_type`, semantic-id join keys
  (`role ∈ {Dimension, Field, Key}`); no authoring `Relationship` persisted.
- **Temporal (`33 §8B`).** `DataKind.temporal: Option<TemporalShape>` is `Some`
  only on leaf `Dataset`s; generic `grain`; no `Resolved*` prefix.
- **CX1 load integrity (`33 §10`).** A well-formed manifest passes
  `SemanticManifest::validate()`; a tampered one fails with `LoadError::{
  DanglingReference, StructuralViolation }`. Sub-gates: dangling refs, bitmask
  contiguity (no trailing zero words; every set bit has a `bit_position`),
  `layer == classify(expr)`, `join_type == derive(optional)`.
- **MUST-NOT persistence boundary (`33 §7`).** No runtime `SemanticGraph`
  (`nodes`/`edges`/`compositions`), no planner cache, no authoring expression form
  (the friendly field-level projection / `Expr<L>` author leaves — `ExprSource` is
  removed per doc 45; the claim is strengthened, not changed), no engine
  / raw-SQL types in the manifest.

### 7.2 Example scenario

```gherkin
@schema @now
Scenario: Compiled manifest keys every entity by its own id
  Given the reference model
  When I compile it
  Then the manifest every entity registry is keyed by EntityId   # 33 §4.1
  And the manifest every entity id equals its map key            # 33 §4.3.1 / CX1
  And the manifest contains no stable_ids side-map               # 33 §2

@schema @CX1 @now
Scenario: A well-formed manifest passes load-time integrity
  Given the reference model
  When I compile it
  Then the manifest passes SemanticManifest::validate            # 33 §10
```

## 8. Pillar 3 — Resolved vs Unresolved (`@resolution`)

Source of truth: `14` (families) + `19` (the fold) + `35 §3`/`§5` (resolved
form). These gates prove the **authoring↔resolved boundary** — the heart of
item Y.

### 8.1 Contract clauses gated

- **Two expression families (`14 §3.3`, `35 §3.3`).** The authoring
  `model::Expr<L>` has **no `Window`, no `Parameter`**; the resolved
  `ir::PhysicalExpr` adds `Window`, and `ir::PhysicalLeaf` adds `Parameter`.
- **Two leaf sets (`14 §3.4`/`§3.5`).** `UnresolvedPhysicalLeaf { Column,
  Literal }`; `UnresolvedSemanticLeaf { …, Field, Dimension, Measure, Metric,
  Key }` with per-kind accessors; resolved `PhysicalLeaf { Column, Literal,
  Parameter }`.
- **No `ir::SemanticExpr` (`35 §5.3`, V6).** The resolved crate defines only the
  physical form.
- **Friendly-tag serde / V7 (`14 §6.1`).** Authored YAML and the generated JSON
  schema use friendly tags only; `binary_op:` / `unary_op:` / `type:` /
  `content:` MUST NOT appear in authored YAML or schema. (`@V7`; gated against
  `reference.yaml` + the schemars output.)
- **The fold (`19 §3`).** After `compile`, **no authoring expression form survives**
  (`@I1`; the `ExprSource` wrapper is removed per doc 45 — the gate's "no ExprSource"
  anchor is strengthened, the *type* itself being gone),
  no `Field`/`Dimension`/`Measure`/… semantic leaf survives — only
  `PhysicalExpr` (modulo `Parameter`). Sugar accessors are lowered to
  `Window` + `Parameter`. Function calls are resolved against the registry at
  compile (`UnknownFunction` on miss).
- **Name resolution at compile / I5 (`19 §2`, `35 §13.1`).** Unresolved
  `SemanticsName` is substituted to `EntityId` at compile; the planner does
  lookup only. `PhysicalLeaf` carries no semantic names.
- **`ExprLayer` (`19 §3.2.6`, `35 §5.5`).** Every pooled expression has
  `layer == classify(expr)` where `classify` is a pure function of the resolved
  tree (`Scalar` / `Aggregate` / `PostAggregate`).
- **Round-trip normal form / V8 (`14 §6`, vocab V8).** `parse ∘ dump ∘ parse ==
  parse`; `if:` normalizes to `case:`, `count_distinct:` to
  `count{distinct:true}`.

### 8.2 Example scenarios

```gherkin
@resolution @V7 @now
Scenario: Authored expressions never leak internal serde discriminants
  Given the reference model
  When I parse it
  And I serialize it back to YAML
  Then the YAML contains no "binary_op:" key       # 14 §6.1 (V7)
  And the YAML contains no "type:" expression tag
  And the YAML contains no "content:" expression tag

@resolution @I1 @target
Scenario: Compile lowers all authoring expressions to the physical form
  Given the reference model
  When I compile it
  Then no ExprSource survives in the manifest                 # 14 §10 (I1)
  And no semantic leaf (Field/Dimension/Measure/Metric/Key) survives  # 19 §3.3.2
  And every pooled expression has layer == classify(expr)     # 19 §3.2.6 / CX1
```

(The first is `@now` — `parse`/serialize are buildable today. The second is
`@target` — it asserts the post-item-Y physical-only pool, which the
mid-migration manifest does not yet fully produce.)

## 9. Pillar 4 — Pipeline End-Result (`@pipeline`)

Source of truth: `10` (stage contracts) + `35 §9`/`§10`/`§13` (`SemanticPlan`) +
`36` (`EngineArtifact`). These gates prove the **terminal artifact of each
stage**.

Because the query-time stages live in excluded crates today, P4 is layered:

### 9.1 Stage-by-stage (the buildable frontier is `compile`)

- **`parse → SemanticModel` (`@now`).** Well-formed YAML yields a typed model;
  malformed YAML accumulates `ParseError`s (accumulate policy, `10 §3.1`).
- **`validate → Result<(), Vec<ValidateError>>` (`@now`).** Pure predicate; all
  violations accumulate (`10 §3.2`).
- **`compile → SemanticManifest` (`@now`/`@target`).** The furthest
  currently-buildable end state. `compile` is `fail-fast` (`10 §3.3`); the I/O
  permission matrix holds (no `Repository`, only `CatalogProvider`/`FileSystem`).
  The manifest is planner-complete (I8) — but the deep shape gates live in P2.
- **`plan → SemanticPlan` (`@target`).** When the planner re-enters the
  workspace: `SemanticPlan { root: PlanNode, output_names, diagnostics }`
  (`35 §9.1`); `PlanNode` is the **closed eight-variant sum** `{ Scan, Filter,
  Project, Agg, Join, Union, Sort, Fetch }` (`35 §10.1`);
  `output_names.len() == root.meta().output_schema.len()` (`35 §9.2`); plan
  invariants `35 §13` (no aggregate embedded in a filter/projection predicate;
  every stored `PhysicalExpr` type-resolved; join/union nullability widening;
  unique agg output names; boolean filter predicate).
- **`adapt → EngineArtifact` (`@target`).** `EngineArtifact = Sql(SqlArtifact) |
  Plan(EnginePlan)` (`35 §12.1`, `36 §3.1`); no `Parameter` survives into adapt
  (`14 §5.3`); SQL appears only here (I1).

### 9.2 Behavioral end-result gates (engine-shape, `@target`)

These encode "what the pipeline should ultimately produce" at the behavioral
level — the kind of assertions the legacy suite made, but re-anchored to the
canonical plan:

- A grainset query selects the coarsest level satisfying the request grain
  (`22`).
- A unionset query produces a `Union` node; partial coverage NULL-fills missing
  semantics (`23`, `15`).
- A joinset query produces `Join` node(s) with the derived `join_type` (`24`,
  `16`).
- A computed dimension lowers to its function tree (not a bare column ref)
  (`19`).
- A constraint violation is a `plan`-stage `ConstraintViolation`, raised before
  artifact emission (`11 §8`, `34`).

## 10. Determinism (`@determinism` / `@I4`)

I4 (`00 §9`, `33 §9.1`) — identical `(model, catalog snapshot)` inputs yield a
**byte-identical** `SemanticManifest`. This is the gate that protects the whole
compile boundary from order-dependent drift.

- **Compile-twice byte-identity.** `Given the reference model / When I compile
  it twice / Then the two manifests are byte-identical` (`@now` once `compile`
  serialization is in place; `@target` until then).
- **Order-insensitivity.** The same model with its top-level pools authored in a
  different order compiles to the same manifest bytes (name-ordered output,
  `33 §7.3` / C18).
- **Round-trip stability.** `parse ∘ dump ∘ parse == parse` over `reference.yaml`
  (`@now`; this is the V8 normal-form property and is buildable today).

## 11. Fixtures

Fixtures are small, single-purpose YAML models, one concern each, plus the
canonical broad model.

- **`reference.yaml`** (`crates/semstrait-model/schemas/reference.yaml`) — the
  canonical "every concept" model; the backbone of P2/P3/P4 happy-path and
  determinism gates. Reused, not copied.
- **Per-rule rejection fixtures** — minimal models that violate exactly one rule
  (R1/R2/R3, coarsest-first, binary-v1, nested-interface, …). Most are small
  enough to inline as Gherkin docstrings rather than files.
- **Pillar fixtures** — one grainset, one unionset, one joinset, one
  computed-dimension model for P4 behavioral gates (analogous to, but distinct
  from, the legacy `comprehensive_ecommerce.yaml`).

Fixtures MUST parse/validate per their stated intent: a "valid" fixture is part
of the contract that it stays valid; a "rejection" fixture is part of the
contract that it stays rejected with the cited code.

## 12. Open Decisions (GATES — not assumptions)

Per repo workstyle (`AGENTS.md §2`; doc 44 §6), these are directional choices the
spec cannot settle by fiat. Each pauses implementation until ruled on.

- **G-1 — Spec home.** This doc sits at `docs/design/testing/50_*`. Confirm the
  `testing/` group + the number `50`, OR relocate (e.g. under `implementation/`).
  On confirmation, wire its registry manifest row + `rollup` (new leaf record)
  and `STATUS.md` per the MAINTAINER procedure — **held for human approval**
  (that procedure was removed when this tree was frozen; see `AGENTS.md §2`).
- **G-2 — Harness crate.** Add `crates/semstrait-contract-gates` to the
  workspace `members`, OR host the harness inside an existing crate's
  `tests/`, OR keep it out-of-workspace until the planner returns. Recommend the
  dedicated member (clean dep boundary, `publish = false`).
- **G-3 — `@target` CI policy.** Confirm `@target` runs as **xfail**
  (allowed-to-fail, passing-is-flagged) vs. **skipped**. Recommend xfail — it
  keeps the aspirational floor executable and the progress meter live.
- **G-4 — Scope of P4 now.** With `-planner`/`-adapter` excluded, P4's
  plan/adapt gates are all `@target`. Confirm we ship them now as `@target`
  (executable spec, dormant) vs. defer the file until the crates return.
  Recommend ship-now-as-`@target`.
- **G-5 — Determinism oracle (mirrors doc 44 §6 I4 gate).** The byte-identity
  gate needs a serialized manifest. Use (a) the live `serde` JSON of
  `SemanticManifest`, or (b) a committed golden snapshot fixture. Recommend (a)
  for the equality gate + (b) a single golden for drift detection.
- **G-6 — `dump`/serialize surface for V7/V8 gates.** P3's friendly-tag and
  round-trip gates need the model's authoring serializer. Confirm the public
  surface (`semstrait_model` serialize path + the `schemars` feature) is the
  binding point. It exists today (`model/src/lib.rs`, `schema` mod behind
  `schemars`).

## 13. Cross-References

- `00 §9` — invariants (I1/I4/I5/I8/I10/I12) the gates cite.
- `10` — per-stage contract (inputs/outputs/error policy/IO matrix) the harness
  binds to.
- `12`, `26` — P1 source.
- `33` — P2 source; `33 §10` CX1 integrity surface.
- `14`, `19`, `35 §3`/`§5` — P3 source.
- `35 §9`/`§10`/`§13`, `36` — P4 source.
- `_research/vocabulary/RATIFICATION_LOG.md` (V1–V10) — item-Y direction that P3
  encodes; `implementation/44_*` — the crate-code effort that migrates P3/P4
  gates `@target → @now`.
- `.agents/skills/semstrait-dev/TESTING.md` — the unit-tier this suite sits
  above; `make test` is the umbrella gate.

## Testing Strategy

This document itself IS the behavioral test harness specification. The testing strategy is defined by the following gate families and cross-cuts:

- **Nesting policy gates (§6)** — prove the structural legality of the DataKind tree against the ratified nesting matrix and structural rules.
- **Schema shape gates (§7)** — prove the compiled `SemanticManifest` shape and invariants, including single identity lane, physical-only expression pool, and CX1 load integrity.
- **Resolved/unresolved boundary gates (§8)** — prove the authoring↔resolved expression boundary, the compile fold, and the two-family expression architecture.
- **Pipeline end-result gates (§9)** — prove the terminal artifact of each pipeline stage (parse → validate → compile → plan → adapt) against the ratified stage contracts.
- **Determinism cross-cut (§10)** — the I4 byte-identity gate threaded through P2 and P4, proving that identical inputs yield byte-identical manifests.
- **Gate lifecycle (§5)** — the `@now`/`@target`/`@determinism` tags that gate CI behavior and track progress toward the ideology.
- **Cucumber-rs harness (§4)** — the `semstrait-contract-gates` crate that binds real canonical entry points to Gherkin feature files, with deterministic in-memory doubles for catalog and filesystem.

This document itself is the shared behavioral harness that the per-leaf ## Testing Strategy sections across all spaces point to.
