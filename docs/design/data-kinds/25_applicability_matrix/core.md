---
id: 25-applicability-matrix-core
status: ratified
version: 2.2
impl: none
---


# 25. Applicability Matrix

> **Reconciliation (Phase-3 / 2026-04-17 consolidation).** The per-variant authoring shape is ratified across [`../apis/32_semstrait_model.md §3`](../../apis/32_semstrait_model.md), [`../foundations/18_entities.md`](../../foundations/18_entities.md), and [`26_nesting_matrix.md`](../26_nesting_matrix.md) (R1 / R2 / R3 structural rules). The cross-variant matrix below remains authoritative as the index of which foundation rules apply to which variant; where body text cites `ColumnMapping` or pre-`18` struct names, apply the renames per `18 §10` (`SemanticMapping`) and `18 §2` (unified `Relationship`). `JoinType::AsOf` cells are forward-reference only (v1 roster excludes AsOf).

---

## 1. Purpose and Scope

### 1.1 What `25` is

A reader who wants to know whether **rule X** applies to **variant Y**, and *how*, should find the answer in exactly one place. `25` is that place. Concretely, `25` ratifies:

- **§2** — the master matrix. One row per foundation-doc section or clause from `10`–`17`; four columns (`Simple/Dataset`, `Grainset`, `Unionset`, `Joinset`). Each cell is a single short phrase plus a cross-ref, with one of the discipline tags `always` / `conditional` / `via Simple children` / `n/a`.
- **§3** — per-variant planner-strategy summary. Three to five bullets per variant plus a pointer to the authoritative strategy section in `21`–`24`.
- **§4** — per-variant rollup-legality summary. What "roll up" means for each variant, which variants do it natively, which require an anchor or a pin policy, which emit advisories.
- **§5** — per-variant Coverage & NULL-fill summary. Where each variant's `Coverage` surface lives (`Binding`-level vs `CompositionCoverage`-level) and how NULL-fill materializes.
- **§6** — per-variant error-code band index. A single table showing which `*_E_2xxx` sub-range each data-kind doc owns, plus the cross-variant band `25` owns.

The scope of `25` ends there. Any cell that needs a decision (variant-specific rule tightening, a new advisory, a new error variant) is referred back to the owning doc.

### 1.2 What `25` is NOT

- **Not** a ratification site for any rule in `10`–`17` or `20`–`24`. Every cell cites an authoritative section; `25` contains no independent normative text.
- **Not** a place to introduce new Semantics, new `DataKind` variants, or new strategies. Those additions are MINOR changes that land in the owning doc (`11` / `20`–`24` / `34`) and then propagate to `25`'s cells in a follow-up doc edit.
- **Not** a place to override per-variant rules. A reader who finds `25`'s cell disagreeing with the authoritative section's text should treat the authoritative section as correct and the `25` cell as needing a doc edit (via `§1.3`).
- **Not** a complete planner contract. The strategy summaries in `§3` are pointers; `34` ratifies the planner's public API.
- **Not** a replacement for `20 §3`'s at-a-glance matrix. `20 §3` compares the four variants against each other on six dimensions; `25` compares the four variants against the **foundation rules**. The two are complementary.

### 1.3 Cross-doc fixes flagged while drafting

Items surfaced during Round-1 drafting of `25` where two authoritative docs appear inconsistent. Each is tagged `[CROSS-DOC-FIX-NEEDED]` in the cell (`§2`) where it bites, and recorded here for the owning doc's next revision. Per the hard constraint on `25`, these are **not resolved** by this document — they are flagged and parked.

| ID | Location | Observed inconsistency | Authoritative owner |
|---|---|---|---|
| `CDF-21-01` | `21 §2.1`'s `DataKind` enum | **RESOLVED.** The earlier flat 4-arm `DataKind` enum has been dropped from `21 §2.1`; the canonical shape is `20`'s two-level `DataKind { Simple(_), Complex(ComplexDataKind) }`. No remaining divergence. | `21` (landed). |
| `CDF-21-02` | `20 §2` / `33 §6` | **RESOLVED.** The divergent inner names (`UnionsetDecl`, `*Spec`) are retired. `20 §2` and `33 §6` ratify a single closed `DataKindVariant { Dataset, Unionset, Grainset, Joinset }`; the bare per-variant names are canonical. No remaining divergence. | `20` (landed). |
| `CDF-23-01` | `23 §1.1` and `23 §4.4` | Both cite `13 §7`'s "cast matrix" / "widening rules". `13 §7` (per the current `13` outline) is "Interaction with Other Docs" — there is no cast-matrix content under that anchor. The widening rules `23` consumes are authored across `13 §2.4` (shape unification) and `14a` (promotion lattice / cast policy). Either `13 §7` needs to grow a "Cast Matrix" subsection, or `23`'s refs should be retargeted to `13 §2.4` + `14a`. | `13` or `23` (editorial). |
| `CDF-17-01` | `17 §7.3` advisory roster × `22 §5` / `23 §6` | `22`'s mixed-shape case is now the hard compile error `COMP_E_2201 GrainsetChildShapeKindMismatch` (the pre-rebase `MixedShapeAdvisoryChildren` advisory is retired); `23`'s `COMP_W_2302`–`W_2306` shape-mismatch advisories still overlap with `17 §7.3`'s advisory-warning roster without a single owning code. A cross-variant `TemporalShape` advisory shared across `Grainset` / `Unionset` (and, once `17 §5` lands, `Joinset`) is a natural `25 §2.9` cell — but the code-owning doc is unclear. | `17` owns the advisory taxonomy; the per-variant `PLAN_W_*` / `COMP_W_*` codes in `22` / `23` / `24` are the emission-site labels. `25` does not resolve; see Q1 in `questions/open/25_questions.md`. |

None of these block `25`'s Round-1 ratification: every affected cell lists **both** citations and tags them, so a reader following either side sees the same rule.

### 1.4 Reading the matrix

Every cell in `§2` carries at most three things:

1. A short discipline tag — one of `always`, `conditional`, `via Simple children`, or `n/a`.
2. An optional one-line qualifier (e.g. "requires anchor per `24 §3`").
3. A cross-ref — a section pointer in `N §M.K` format — to the authoritative doc.

Tag meanings (stable across `§2`):

- **`always`** — the rule applies unconditionally to this variant. No per-variant narrowing.
- **`conditional`** — the rule applies, but under a variant-specific gate (different `TemporalShape`, different child shape, different request shape). The qualifier names the gate.
- **`via Simple children`** — the rule applies to the variant **only** through its constituent `Simple`s' Bindings. A `Complex` variant does not carry the rule directly; its `Simple` children do.
- **`n/a`** — the rule structurally does not apply. `n/a` is never a gap; it is always the correct answer for the cell.

Cross-refs use `N §M.K` format. A blank authoritative doc — happens only for forward-referenced `17` sections whose numbering is still landing — is tagged `17 §*` per `21`–`24`'s existing convention.

### 1.5 Reading paths

Three reading paths cover the majority of intended consumption:

- **Path A — "does this foundation rule apply to my variant?"** Start at `§2`; pick the sub-table owning the foundation doc (`§2.2`–`§2.9`); read the row for the clause and the column for the variant. Cell contents are self-contained; follow the cross-ref for detail.
- **Path B — "how does the planner handle my variant?"** Start at `§3`; read the relevant sub-section's bullets; follow `§3.5`'s interaction table for cross-variant effects; follow the authoritative pointer for full strategy semantics.
- **Path C — "what error-code range do I draft in?"** Start at `§6.1`'s allocation summary; if cross-variant, see `§6.2`; for a decision aid, see `§6.4`.

Readers exploring `Coverage` / `NULL-fill` semantics should use `§5`; rollup semantics, `§4`. Neither is a substitute for the owning docs — both are indexes with pointers.

---

## 2. The Matrix — Foundation Rule × DataKind Variant

The master cross-cut. Split across per-doc sub-tables (`§2.1`–`§2.9`) for readability; the rows are identical in form to the preview in `20 §7`. A reader who wants the one-screen summary of "which variant composes how" should start from `20 §3`'s at-a-glance matrix; the sub-tables below zoom into each foundation-rule owner individually.

### 2.1 Legend

See `§1.4` for tag meanings. Discipline is identical across every sub-table: rows are foundation-rule clauses, columns are `Simple` / `Grainset` / `Unionset` / `Joinset`, cells carry at most (tag, qualifier, cross-ref).

Two meta-rules govern every cell and cover many of the trivially-uniform rows:

- **Meta-rule M1 (interface exposure).** Per `20 §4.4` Invariant D5: every `Simple` exposes a bare `SemanticInterface` (`16 §5.4`); every `Complex` variant exposes a `ComposedSemanticInterface` (`16 §5.1`). A cell tagged `via Simple children` on the Complex columns therefore implicitly mediates through each constituent's `SemanticInterface` as surfaced through `UnifiedSemantics` (`16 §6`).
- **Meta-rule M2 (strategy delegation).** Per `20 §5.1` Invariant D9: exactly one `Strategy` implementation per variant (`SimpleStrategy` / `GrainsetStrategy` / `UnionsetStrategy` / `JoinsetStrategy`). A cell tagged `always` on a `plan`-stage row means the corresponding strategy consumes the rule; a cell tagged `n/a` means the strategy never references the rule.

Cells that would otherwise read identical across the four columns are still written out — the reader benefit of "no column needs guessing" is worth the repetition. Repeated cells are sometimes abbreviated by a preceding `—` followed by "same as §M.K row N" when context is unambiguous.

### 2.2 Pipeline — `10`

Every variant participates in every pipeline stage. Per-variant specialization lives in stage-owned hooks ratified per `20 §2.2` (lifecycle hooks live outside the sealed trait hierarchy): `validate` rules at `10 §3`, `compile` work at `15 §10` (Simple) / `16 §10.1` (Complex), and `Strategy::resolve` (`20 §5.2`).

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `10 §3.1` `parse` | `always` — recognizes `datasets:` discriminator (`12 §3`–`§6`). | `always` — recognizes `grainsets:` discriminator (`12 §4`). | `always` — recognizes `unionsets:` discriminator (`12 §3`). | `always` — recognizes `joinsets:` discriminator (`12 §5`). |
| `10 §3.2` `validate` | `always` — structural + `VALID_E_2100`–`2199` (`21 §7`). | `always` — `VALID_E_2200`–`2299` (`22 §7`). | `always` — `VALID_E_2300`–`2399` (`23 §8`). | `always` — `VALID_E_2400`–`2499` (`24 §9`). |
| `10 §3.3` `compile` | `always` — one `Binding` resolution per `15 §10`; emits `ResolvedSimpleDataKind` (`21 §2.3`). | `always` — per-child `Binding` resolution + level-sorted indices + `ComposedSemanticInterface` build (`22 §8.1`). | `always` — per-child resolution + `CompositionCoverage` fold + `ResolvedUnionset` (`23 §9`). | `always` — path resolution + `JoinType`-override check + materialized `ComposedSemanticInterface` (`24 §10`). |
| `10 §3.4` `plan` | `always` — `SimpleStrategy` (`21 §4`). | `always` — `GrainsetStrategy` dispatches single child (`22 §4` / `§10`). | `always` — `UnionsetStrategy` emits `PlanNode::Union` over children (`23 §4`). | `always` — `JoinsetStrategy` lowers anchor-outward hops to `PlanNode::Join` (`24 §5`). |
| `10 §3.5` `optimize` | `always` — variant-agnostic; operates on `PlanNode` tree (`10 §3.5`). | `always` — same. | `always` — same. | `always` — same. |
| `10 §3.6` `adapt` | `always` — variant-agnostic; lowers `PlanNode` → `EngineArtifact` (`10 §3.6`). | `always` — same. | `always` — same. | `always` — same. |

Every `plan` / `optimize` / `adapt` cell is identical by construction — `20 §4.2`'s shared skeleton and I3 / I6 together forbid variant-specialization past the strategy-dispatch boundary.

Per-variant specialization within each stage is confined to the stage-owned hooks per `20 §2.2` (the sealed trait hierarchy carries only read-only accessors; lifecycle work is stage-owned):

| Stage | Shared skeleton | Per-variant hook |
|---|---|---|
| `parse` | `parse_yaml_tree` (`10 §3.1`). | Per-variant YAML surface (`32 §3`) + `serde` derive on each `*Body` struct (`32 §3.2`). |
| `validate` | `run_structural_checks` accumulates diagnostics (`10 §3.2`). | Per-variant structural preconditions dispatched at `10 §3` — rules in `21 §7` / `22 §7` / `23 §8` / `24 §9`. |
| `compile` | `resolve_bindings` + `resolve_interfaces` (`10 §3.3`). | Per-variant compile work — `15 §10` for Simple (emits `ResolvedSimpleDataKind`, `21 §2.3`); `16 §10.1` for Complex (emits `ResolvedGrainset` (`22 §2.2`) / `ResolvedUnionset` (`23 §2.3`) / `ResolvedJoinset` (`24 §2.4`)). |
| `plan` | `Strategy::resolve` dispatched by variant tag (`20 §5.3`). | Per-variant `Strategy` impl in `21 §4` / `22 §10` / `23 §4` / `24 §5`. |
| `optimize` | Variant-agnostic rule set (`10 §3.5`). | None. |
| `adapt` | Variant-agnostic emission (`10 §3.6`). | None. |

### 2.3 Names, Scopes, Semantics — `11`

`11` covers identifier rules, Semantics-element declarations (Dimensions / Measures / Metrics / Filters / Keys), the `Additivity` axis, and the `Constraint` framework. Every variant exposes an interface, so every variant interacts with every clause of `11`.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `11 §3` global identity | `always` — DataKind `name:` is globally unique at Root scope (`11 §3.1`). | `always` — same. | `always` — same. | `always` — same. |
| `11 §6.1` Dimension | `always` — on the bare `SemanticInterface` (`21 §2.2`). | `always` — on the `ComposedSemanticInterface` declared on the Grainset (`22 §2.1` / `§3.1`). | `always` — declared on the Unionset; children do not declare Semantics (`23 §2.1` / `12 §3.3`). | `always` — declared on the Joinset + inherited from constituents via `UnifiedSemantics` (`24 §2.6` / `§8.2`). |
| `11 §6.2` Measure | `always`. | `always` — same composition path. | `always` — Measure emission interacts with `UnionMode` re-aggregation (`23 §4.5`). | `always` — Measure composition interacts with `Cardinality` fanout advisories (`24 §5.4` / `§11.2`). |
| `11 §6.3` Metric | `always`. | `always`. | `always` — Metric decomposition flows through per-child Measures (`23 §4.5`). | `always`. |
| `11 §6.4` Filter | `always` — Filter placed between `SimpleStrategy` layers (`21 §4.1`). | `always` — filter applies post-dispatch to the chosen child's sub-plan (`22 §10.2`). | `always` — filter decomposes per-child per `23 §4.2` step 3 (partial-NullFill branch-elimination). | `always` — Joinset-level filter applies post-join `Project` (`24 §5.5` step 4). |
| `11 §6.5` Key | `always` — Simple Keys live on the bare interface (`21 §2.2`). | `always` — Grainset may declare composed keys; child keys compose per `16 §6.5` (`22 §2.3`). | `always` — same composed-key rule (`23 §11.2`). | `always` — Joinset default-inherits anchor's keys (`24 §2.6` / `§8.2`). |
| `11 §7` Additivity | `always` — declared on Measure/Metric; shape-locked per `11 §7.4`. | `conditional` — same shape-lock, but planner consistency check at compile (`22 §8`). | `conditional` — union-composed measures may require explicit `constraints:` per `11 §8` (`23 §4.5` lossy-reagg table). | `conditional` — inherits Additivity from the natively-providing member per `FieldProvenance` (`24 §8.3`). |
| `11 §8` Constraint | `always` — Measures and Metrics carry `Constraint`s regardless of variant (`11 §8.4`). | `always` — same. | `always` — same. | `always` — same. |

`§2.9` re-visits the `Additivity × TemporalShape` interaction as a separate applicability axis per `17 §7`.

The `11 §5` shape-vs-resolution split (a Semantics shape is locked across occurrences; resolution-variant fields may differ per `DataKind`) applies uniformly:

| `11 §5` aspect | `Simple` | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| Shape fields (`11 §5.1`) | Locked at the declaration site. | Same — a Grainset's composed-surface shape is locked across children per `11 §5.3`'s worked example. | Same — cross-child shape divergence is a compile error (`23 §9` `COMP_E_2302`). | Same — composed-surface shape locked; per-member differences surfaced via `FieldProvenance` (`16 §7`). |
| Resolution-variant fields (`11 §5.2`) | Per-Simple `Binding.column_mapping[].expr` may vary. | Per-child resolution varies — each Grainset level has its own `Binding` resolution. | Per-child resolution varies. | Per-member resolution varies; cross-member expressions use `19 §3.4.5`'s `PathSignature`. |

### 2.4 Nesting Policy — `12`

`12` ratifies the nesting matrix (`12 §2`) plus each `ComplexDataKind` variant's block shape (`12 §3`–`§5`) and the `Simple` leaf rule (`12 §6`). The matrix's structural rule per `20 §4.6`'s projection is reproduced below.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `12 §2` nesting matrix — as parent | `n/a` — Simple never nests children (`12 §6`). | `always` — may contain `Simple` / `Unionset` / `Joinset`; same-variant self-nest banned (`12 §2` row `Grainset per level`). | `always` — may contain `Simple` / `Grainset` / `Joinset`; same-variant self-nest banned. | `always` — may contain `Simple` / `Unionset` / `Grainset` as members; same-variant self-nest banned. |
| `12 §2` nesting matrix — as child | `always` — may nest under any `ComplexDataKind` (`12 §6.2`). | `always` — may be a Unionset branch or Joinset member; **forbidden** as a Grainset child per `26 §R2` (structural ban; `[TD-GRAINSET-NESTED]` retired). | `always` — may be a Grainset child or Joinset member; banned as a Unionset child (same-variant). | `always` — may be a Unionset branch or Grainset child; banned as a Joinset member (same-variant, `12 §2`). |
| `12 §3` Unionset block shape | `n/a`. | `n/a`. | `always` — `unionsets:` block w/ ≥ 2 children (`12 §3.2`, `23 §2.1`). | `n/a`. |
| `12 §4` Grainset block shape | `n/a`. | `always` — `grainsets:` block w/ ordered levels (`12 §4.2`, `22 §2.1`). | `n/a`. | `n/a`. |
| `12 §5` Joinset block shape | `n/a`. | `n/a`. | `n/a`. | `always` — `joinsets:` block; binary arity in v1 per `12 §5.2` / `24 §2.5`. |
| `12 §6` Simple nesting | `always` — terminal leaf; no children allowed (`12 §6.2`). | `n/a` — Grainset is not a leaf. | `n/a`. | `n/a`. |

The canonical nesting matrix from `12 §2` is reproduced here in projection form for reader convenience; authoritative content is in `12 §2`:

| Parent \ Child | `Simple` | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `Simple` | n/a — no children (`12 §6`). | n/a. | n/a. | n/a. |
| `Grainset` (per-level) | Legal (`12 §4.3`). | **Banned** — same-variant self-nest per `26 §R2` (structural; `[TD-GRAINSET-NESTED]` retired). | Legal. | Legal. |
| `Unionset` (per-branch) | Legal (`12 §3.2`). | Legal. | **Banned** — same-variant self-nest. | Legal. |
| `Joinset` (per-member, v1 binary) | Legal (`12 §5.3`). | Legal. | Legal. | **Banned** — same-variant self-nest. |

### 2.5 Types and Grain — `13`

All variants carry logical types. `Grain` participates in every variant but only `Grainset` is **grain-aware** at the DataKind level (per `20 §4.5` Invariant D6).

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `13 §2` `DataType` catalog | `always` — on every Semantics + every `Binding` column (`21 §2.2`). | `always` — on every composed-surface Semantics (`22 §2.1`). | `always` — on every composed-surface Semantics; `23 §4.4` reconciles across children. | `always` — on every composed-surface Semantics + every `Relationship.keys` (`16 §12.2`). |
| `13 §3` `Grain` enum + total coarseness order | `conditional` — optional `grain:` field on `SimpleDataKind` (`21 §2.2` / `§6`); shape-gated per `17`. | **`always`** — per-child `extras.temporal.grain:` authoring is mandatory per SR-E-8 (`18 §3.4`; `22 §6.1`); ≥ 2 unique grains per Grainset (`22 §5.2`); routing-unit eligibility uses `Grain::coarseness()` (`22 §4.1`). | `conditional` — children should share a common grain or be rollable to a common coarsest per `17`; advisory `COMP_W_2308` (`23 §7`). | `conditional` — per-member grain is each member's concern; Joinset inherits anchor grain as the join's output grain by default (`24 §5.4`). |
| `13 §4` `DimensionType` discriminator | `always` — authored on every Dimension (`13 §4`). | `always`. | `always`. | `always`. |
| `13 §5` Keys vs Dimensions | `always` — Keys are declared separately from Dimensions (`13 §5.1`). | `always`. | `always`. | `always`. |
| `13 §2.4` shape unification (authoritative cast rules) + `14a` cast policy | `always` — used at the Binding boundary (`15 §9`). | `always` — used during child-shape-conflict checks (`22 §8` `COMP_E_2204` / `COMP_E_2206`). | `always` — used during cross-child type reconciliation (`23 §4.4` / `§9.4`). `[CROSS-DOC-FIX-NEEDED]`: `23` cites `13 §7` as the cast-matrix owner; see `§1.3 CDF-23-01`. | `always` — used during `Relationship.keys` type agreement (`16 §12.2`). |

The `13 §7` row in the user's spec ("Joinset column type-compat: only Joinset") maps onto the actual authoritative sites: `13 §2.4` (shape unification) + `16 §12.2` (`KeyPair` type agreement). `Joinset` is where the `Relationship.keys` type agreement bites most often in practice, but the underlying rule is not Joinset-specific — it applies to any consumer of a `Relationship`, including implicit composition (`16 §11`). `25` records both the narrow (Joinset-only) and broad (any `Relationship` consumer) readings; the narrow reading lives in `24 §5.2` step 3, the broad one in `16 §12.2`.

### 2.6 Expressions — `14`, `14a`, `19`

All three expression docs apply to every variant. Authoring happens on `Simple`s directly; `Complex` variants inherit the resolved expressions via their constituents.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `14 §2` `SemanticExpr` / `PhysicalExpr` split | `always` — authored at `Binding.column_mapping[].expr` + Semantics-level `expr:` (`21 §4.4`). | `via Simple children` — authored per child; Grainset declares no Bindings (`20 §4.3`). | `via Simple children` — same. | `via Simple children` — same; Joinset-level derived fields use `FieldOwnership::Derived` at the composed surface (`24 §2.6`). |
| `14 §3` shared `Expr` AST | `always`. | `always` — consumed through child sub-plans. | `always`. | `always` — consumed in `KeyPair` predicate lowering (`24 §5.2` step 3) + Joinset-level derived fields. |
| `14 §6` authoring `Expr<L>` YAML grammar | `always`. | `always` — only in per-level declarations that live on `Simple` children. | `always` — only in per-branch declarations that live on `Simple` children. | `always` — only in per-member declarations + in Joinset-level derived fields. |
| `14 §5`–`§6` typing + computed-Dimension `data_type:` inference | `always` — consumed at Computed-Dimension inference per `14 §6.2`. | `via Simple children`. | `via Simple children` + cross-child widening at `UnifiedSemantics` build (`23 §4.4`). | `via Simple children` + `KeyPair` type agreement (`16 §12.2`). |
| `14a` `FunctionRegistry` / `CanonicalFn` | `always`. | `always`. | `always` — used at cross-child widening LUB (`23 §4.4`). | `always`. |
| `19 §3.2` `ResolvedExprTable` + cross-DataKind path pre-resolution | `always` — per-`(Semantics, Binding)` pre-resolved at compile (`19 §3.2`). | `via Simple children` + composed-surface lifts where a Grainset-level Semantics refers to a child's field. | `via Simple children` + composed-surface lifts where cross-child shape reconciliation wraps a `Cast` (`23 §4.4`). | `via Simple children` + Joinset-level `Derived` expressions pre-resolved + `PathSignature` (`19 §3.4.5`) for cross-member expressions. |

Round-1 decision recorded: no `Complex` variant authors `SemanticExpr` / `PhysicalExpr` directly on a `Binding`; every such expression lives on a `Simple` and is composed up. The only Complex-authored expressions are (a) Joinset-level `Derived` fields (`24 §2.6`) and (b) Joinset-level Filter clauses applied post-join (`24 §5.5` step 4).

`19 §3.4.5`'s `PathSignature` is a `BTreeSet<RelationshipPath>`; only `Joinset` and the implicit-composition consumer in `16 §11` populate it with `len() > 0`. Bare `Simple` and single-constituent Complex composition produce the empty signature.

| `14` axis | Authored where? | Pre-resolved where? |
|---|---|---|
| `PhysicalExpr::Column` (ColumnMapping value) | `Simple.binding.column_mapping[].value` (`15 §5`). | `compile`; result keyed by `ResolvedExprKey` (`19 §3.2.1`). |
| `PhysicalExpr::Computed` (ColumnMapping value) | `Simple.binding.column_mapping[].value` (`15 §5.4`). | `compile`; types inferred per `14 §6`. |
| `SemanticExpr` (Computed Dimension / Measure / Metric) | `Simple.semantics.*.expr` (`11 §6`). | `compile`; lifted to the Complex's `ComposedSemanticInterface` unchanged (`16 §5.5`). |
| Joinset-level `Derived` expression | `Joinset.semantics.*.expr` (`24 §2.6`). | `compile`; `PathSignature` populated if the expression references a non-anchor member. |
| Joinset-level `Filter` | `Joinset.filters[]` (`24 §2.6`). | `compile`; placement is post-join, pre-`Project` (`24 §5.5` step 4). |

### 2.7 Mapping and Binding — `15`

`15` is the `Binding` / `ColumnMapping` / `PhysicalSource` doc. Per `20 §2.3` Invariant D1, the `Binding` lives on `Simple` exactly; Complex variants have none and compose their children's.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `15 §2` `Binding` | **`always`** — exactly one per `SimpleDataKind` (`15 §2.1`, `20 §2.3`, `21 §3.1`). | `via Simple children` — no own Binding (`20 §2.3` Invariant D1). | `via Simple children` — same. | `via Simple children` — same. |
| `15 §3` `PhysicalSource` | `always` — ≥ 1 per Binding (`15 §3.5` glob expansion). | `via Simple children`. | `via Simple children`. | `via Simple children`. |
| `15 §5` `ColumnMapping` | `always` — Column / Literal / Computed / Metadata per `15 §5.1`. | `via Simple children`. | `via Simple children`. | `via Simple children`. |
| `15 §6` Coverage — Binding-level | `always` — one `Coverage` entry per `(Semantics, PhysicalSource)` (`15 §6.1`). | `via Simple children` — folded into per-effective-routing-unit Coverage at compile per `22 §3.2`; lifted to `CompositionCoverage` per `16 §8.4` only when cross-grain JOIN composition activates (`22 §3.4`). | `via children` — folded into bare per-child `Coverage` per `23 §3.2`; no `CompositionCoverage` post-rebase. | `via Simple children` — then lifted per `24 §8.4`. |

Coverage gets re-visited at the composition layer in `§2.8` row `16 §8`; the Binding-level cell above is the leaf-per-`PhysicalSource` rule.

`15`'s `Binding`-per-`Simple` rule is load-bearing enough to warrant an explicit table mapping the canonical SemanticManifest-layer counterpart (`15 §7`) onto each variant:

| `15` SemanticManifest counterpart | Owned by | Consumers |
|---|---|---|
| `ResolvedColumnMapping` (`15 §7`) | `ResolvedSimpleDataKind.binding` (`21 §2.3`). | Every variant's plan emission reaches it via its Simple children. |
| `ResolvedPhysicalSource` (`15 §7`) | Same. | Same. |
| `ResolvedCoverage` (`15 §6` resolution) | Same. | `CompositionCoverage` fold in `16 §8.4` / `22 §3.2` (cross-grain JOIN only) / `24 §8.4`. Unionset uses bare per-child `Coverage` per `23 §3.2` (no `CompositionCoverage` post-rebase). |
| `ResolvedSchema` (`15 §3.2`) | Same. | Type-agreement checks in `23 §4.4`, `24 §5.2` step 3. |

### 2.8 Composition — `16`

`16` is the home of `Relationship`s, `Cardinality`, `JoinType`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, and the explicit-vs-implicit composition boundary.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `16 §2` `Relationship` — declaring | `n/a` — DataKinds do not declare Relationships; the root-level `relationships:` block does (`16 §2.1`, `18 §2.10`). | `n/a`. | `n/a`. | `conditional` — a Joinset **consumes** root-level Relationships (`24 §6`) and MAY declare **scope-local** ones in its own `relationships:` block, bounded to its scope (`18 §2.10`, `24 §5.3.2`). |
| `16 §2` `Relationship` — consuming via implicit composition | `conditional` — when a Request spans multiple top-level DataKinds, any `Simple` may participate as a constituent (`16 §11.4`). | `conditional` — same; a Grainset exposes a composed surface that can participate as a constituent in implicit composition (`22 §1.2`). | `conditional` — same. | `always` — `Joinset.path` references Relationships by id (`24 §6`). |
| `16 §3` `Cardinality` | `conditional` — consumed only when the `Simple` participates in implicit composition. | `conditional` — same. | `conditional` — same; also consumed in `Additivity × Cardinality` post-Union re-aggregation checks (`23 §7.4`). | `always` — per-hop walked `Cardinality` drives fan-out advisories (`24 §3.4` / `§5.4`). |
| `16 §4` `JoinType` | `conditional` — used only in implicit composition. | `conditional` — same. | `conditional` — same. | `always` — per-hop `JoinType` derived from each traversed `Relationship.optional` (`18 §2.9`); divergent join semantics via scope-local Relationship shadow (`18 §2.10`, `24 §5.3.2`). |
| `16 §5` `ComposedSemanticInterface` | `n/a` — Simple exposes bare `SemanticInterface` (`20 §4.4` Invariant D5). | **`conditional`** — built ONLY for cross-grain JOIN composition (per `22 §3.4` / `§4.3`); single-unit-delegation case (`22 §4.2`) does not synthesize one. When built, `composition_kind == Grainset`. | `n/a` — post-thirteenth-pass cascade rebase (2026-05-03): Unionset uses bare `SemanticInterface` per `23 §3.2`; the `CompositionKind::Unionset` variant is retired in `16 §5.3`. | `always` — `composition_kind == Joinset` (`24 §2.4`). |
| `16 §6` `UnifiedSemantics` merge rules | `n/a`. | **`conditional`** — same predicate as `16 §5` row above (only when cross-grain JOIN composition activates per `22 §4.3`). | `n/a` — Unionset uses its own bare `SemanticInterface` (`23 §3.2`); no `UnifiedSemantics` involvement post-rebase. | `always` — merge with namespacing for shape-incompatible collisions (`24 §8.2`). |
| `16 §7` `FieldProvenance` — `Native` / `Shared` / `Derived` | `n/a` — no provenance on a bare interface. | **`conditional`** — only for cross-grain JOIN composition (`22 §3.4`); same predicate as `16 §5` row. | `n/a` — bare `SemanticInterface` has no `FieldProvenance`. | `always`. |
| `16 §7.3.3` `FieldOwnership::NullFill` | `n/a`. | `n/a` — Grainset cross-grain composition uses LEFT OUTER JOIN; outer-join NULL-fill is carried by `JoinType::LeftOuter` semantics at plan time, not by `FieldOwnership`. | `n/a` — bare interface; Unionset's per-child seam emits `Cast(Null, _)` projections per `23 §4.3`, not via `FieldOwnership`. | `n/a` — Joinset NULL-fill is carried by `JoinType` outer-join semantics (`24 §8.3`). |
| `16 §8` `CompositionCoverage` | `n/a` — `Coverage` lives at the Binding level for Simple. | **`conditional`** — built only for cross-grain JOIN composition (`22 §3.4` / `§3.2`); fold mirrors per-unit Coverage projection. Single-unit delegation reads per-unit Coverage directly (no `CompositionCoverage` synthesis). | `always` — per-child fold per `23 §3.2` / `§4`. | `always` — per-member fold per `24 §8.4`. |
| `16 §9` explicit vs implicit composition | `n/a` — Simple is a leaf, neither explicit nor implicit. | **explicit** — Grainset is an explicit Complex (`16 §9.2`). | **explicit** — Unionset is an explicit Complex. | **explicit** — Joinset is the canonical explicit-composition DataKind (`16 §13`). |
| `16 §10` materialization policy | `n/a`. | `always` — materialized at compile (`22 §8.1`). | `always` — materialized at compile (`23 §9`). | `always` — materialized at compile (`24 §2.4`). |
| `16 §11` field-first resolution (entry point when `Request.from: None`) | `always` — a Simple may be the sole owner of a requested field (`16 §11.3` single-kind fast path). | `always` — a Grainset's composed surface can own requested fields; implicit composition may layer above it (`22 §1.2`). | `always` — a Unionset's composed surface owns its declared fields; implicit composition may layer above. | `always` — a Joinset's composed surface owns its declared fields; author uses `from: Some(joinset)` to target the pre-built surface (`24 §1.3`). |
| `16 §12` Relationship-graph well-formedness | `conditional` — Simple participates as a Relationship endpoint. | `conditional` — same. | `conditional` — same. | `conditional` — Joinset consumes the resulting graph (`24 §6`). |

The per-variant interface-exposure invariant (`20 §4.4` D5) is reproduced by the `16 §5` row: `Simple` → `n/a`, every `Complex` → `always`.

The `CompositionKind` enum distribution across the matrix is summarized in `16 §5.3` (V1 carries 2 variants — Unionset retired post-thirteenth-pass cascade rebase 2026-05-03):

| `CompositionKind` variant | Emitted by | `25` row pointer |
|---|---|---|
| `CompositionKind::Joinset` (`Origin::Implicit`) | `16 §10.4` compile-time enumeration of Joinsets from the relationship graph. | `§2.8` row `16 §10`. |
| `CompositionKind::Joinset` (`Origin::Explicit`) | Author-declared `Joinset` (`24 §2.4`). | `§2.8` row `16 §5`, Joinset column. |
| `CompositionKind::Grainset` (`Origin::Explicit`) | `Grainset` cross-grain JOIN composition only (per `22 §3.4`); single-unit delegation does not produce a `CompositionKind`. Built at compile from declared `Key`s (per `18 §2.5`); per-pair JOIN-key index lives on `ResolvedGrainset` per `33`, not on `ComposedSemanticInterface`. | `§2.8` row `16 §5`, Grainset column. |

Neither `Simple` nor `Unionset` produces a `CompositionKind` in V1: Simple exposes a bare `SemanticInterface` per `20 §4.4` D5; Unionset uses its own bare `SemanticInterface` per `23 §3.2` (post-rebase). Same-grain implicit Unionsets that arise inside a Grainset (per `22 §3.3`) are Grainset-internal constructs and do not carry a `CompositionKind` discriminator.

### 2.9 Temporal Shape — `17`

Vocabulary ratified in `17`; planner-side support is DEFERRED in places (see `00 §4.1` `TemporalShape` row + `17 §1.4`). `17 §*` placeholders track sections whose numbering is still landing.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `17 §2` `TemporalShape` vocabulary | `always` — as an optional declaration axis on the DataKind. | `always` — same. | `always` — same. | `always` — same. |
| `17 §3.1` declaration site on `SimpleDataKind` | **`always`** — declared inline via `temporal_shape:` (`21 §5.1`). | `n/a` — Grainset itself does not carry a top-level shape; each child may (`17 §3.2`). | `n/a` — same. | `n/a` — same. |
| `17 §3.2` Complex does NOT carry its own `TemporalShape` | `n/a`. | `always` — shape derived from children per `17`. | `always` — same; mixed-shape branches emit advisories (`23 §6.1`). | `always` — shape derived from the two members; `AsOf` activation per `17 §5` consults both sides. |
| `17 §4` shape × `Grain` rollup matrix | `conditional` — legality gated per `17 §4.1`'s matrix; shape-less Simples default-legal (`21 §6.3`). | **`always`** — gates per-routing-unit rollup wrapper emission per `22 §4.4` (`PLAN_E_2203 GrainsetSnapshotRollupWithoutPin`, `PLAN_E_2204 GrainsetSCDRollupWithoutAsOf`). | `conditional` — gates post-Union rollup shape per `23 §7` (`PLAN_E_2302`). | `conditional` — per-member rollup is each member's concern. |
| `17 §5` `AsOf` `JoinType` | `n/a` — Simple has no `JoinType`. | `conditional` — when a Grainset child is itself an SCD, the grainset-level rollup requires as-of anchoring per `22 §5` `SCDRollupWithoutAsOf`. | `conditional` — cross-shape branches may require as-of reconciliation (`23 §6.3`, `[TD-UNIONSET-SHAPE-PLANNING]`). | **`always`** — `JoinType::AsOf` activation is gated by shape pairs + temporal-key presence (`24 §7`, `COMP_E_2414 JoinsetAsOfMissingTemporalKey`). Also fires for implicit Joinsets (`Origin::Implicit`) enumerated at compile per `16 §10.4` — see next row. |
| `17 §5` `AsOf` across implicit Joinsets | `n/a`. | `conditional` — `16 §10.4`-enumerated implicit Joinsets containing a Grainset constituent inherit the same as-of gating. | `conditional` — same. | `conditional` — uniform with `Origin::Explicit` Joinsets per the unified Joinset model. |
| `17 §6` `Request.temporal` block (`as_of:`, time-range overrides) | DEFERRED per `17 §6.5` — no variant consumes in v1. | DEFERRED. | DEFERRED. | DEFERRED. |
| `17 §7` `Additivity × TemporalShape` advisories | `conditional` — emits advisory when Simple's Measure `Additivity` and the Simple's `TemporalShape` appear inconsistent (`21 §5.2` / `PLAN_W_2102`). | `n/a` — V1 strict equivalence (`22 §5.2`) makes shape mismatch a hard compile error (`COMP_E_2201`); the pre-rebase mixed-shape advisory (`PLAN_W_2202`) was retired. Per-child Additivity / Measure advisories are emitted by each child's own Strategy. | `conditional` — emits advisories for cross-child shape mismatch (`23 §6.1` `COMP_W_2302`–`W_2306`). `[CROSS-DOC-FIX-NEEDED]`: `§1.3 CDF-17-01`. | `conditional` — emits advisory when `AsOf` is activated over a hop whose declared `JoinType` was non-`AsOf` (`24 §11.2` `PLAN_W_2404`). |
| `17 §8` shape-gated composition rules (`17 §8.1` Unionset branches; `§8.2` Grainset levels; `§8.3` Joinset hops) | `n/a`. | `always` — Grainset routing-unit selection / rollup is shape-gated per `17 §8.2` (consumed by `22 §4.4`). | `always` — Unionset branches are shape-gated per `17 §8.1` (consumed by `23 §6`). | `always` — Joinset hops are shape-gated per `17 §8.3` (consumed by `24 §7`). Implicit Joinsets (`Origin::Implicit`) inherit the same gating uniformly under the unified Joinset model (`16 §5.6`, `§10.4`). |

Rows for `17 §5` explicitly spell out the "all variants — wherever a join hop crosses shape pairs requiring as-of semantics" rule from the user's spec by splitting the `AsOf` applicability into (a) the explicit Joinset hop and (b) the implicit-Relationship hop. Both cite `17 §5`; both inherit the matrix.

A companion table lists `TemporalShape`-declaration capability across the four variants. "Carries" means the variant has a top-level declaration site; "derives" means the shape is inferred from constituents at `compile`:

| Variant | Top-level `temporal_shape:` field? | Derived at compile? | Authoritative ref |
|---|---|---|---|
| `Simple` / Dataset | Yes. Inline on `SimpleDataKind` (`21 §5.1`). | n/a — declared or implicit-absent. | `17 §3.1`. |
| `Grainset` | No (`17 §3.2`). | Yes — per-level; the Grainset surface may be heterogeneous, advisory via `22 §5`. | `17 §3.2`. |
| `Unionset` | No. | Yes — per-branch; heterogeneous across branches is advisory per `23 §6.1`. | `17 §3.2` + `17 §8.1`. |
| `Joinset` | No. | Yes — per-member; anchor-side vs target-side shape pair drives `AsOf` activation (`17 §5`). | `17 §3.2` + `17 §8.3`. |

### 2.10 Invariants — `00`

The cross-doc invariants `I1`–`I12` ratified in `00 §9` apply uniformly to every variant. The table below is for reader convenience; no variant claims exemption from any invariant.

| Invariant | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| I1 (no raw SQL in Model / SemanticManifest) | `always`. | `always`. | `always`. | `always`. |
| I2 (one canonical `Expr` AST) | `always` — per `14 §3`. | `always`. | `always`. | `always`. |
| I3 (variant-agnostic `PlanNode` IR) | `always`. | `always`. | `always`. | `always`. |
| I4 (SemanticManifest deterministic) | `always`. | `always` — strategy output deterministic per `22 §10`. | `always`. | `always` — explicit-path hop order is deterministic per `24 §4.2`. |
| I5 (all resolution at compile) | `always`. | `always`. | `always`. | `always`. |
| I6 (plan hot path synchronous) | `always`. | `always`. | `always`. | `always`. |
| I7 (single adapter surface `lower_plan`) | `always`. | `always`. | `always`. | `always`. |
| I8 (SemanticManifest is planner-complete) | `always`. | `always`. | `always`. | `always`. |
| I9 (Session-context-based overrides) | `always`. | `always`. | `always`. | `always`. |
| I10 (`#[non_exhaustive]` on public sum types) | `always` — e.g. `DataType`, `Grain`, `TemporalShape`, `ColumnMappingValue`. | `always`. | `always`. | `always` — `JoinType`, `Cardinality`. |
| I11 (separation of Model / SemanticManifest / Planner / IR) | `always`. | `always`. | `always`. | `always`. |
| I12 (first-class diagnostics) | `always` — `VALID_E_21xx` / `COMP_E_21xx` / `PLAN_E_21xx` / `PLAN_W_21xx`. | `always`. | `always`. | `always`. |

---

## Testing Strategy

- Matrix completeness (every foundation clause × variant cell has a pointer); seam-uniqueness (no cell overrides an authoritative section); cross-doc-fix flag detection.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
