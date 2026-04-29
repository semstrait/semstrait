---
prereqs: [18, 19, 20, 21, 22, 23, 24]
authoritative-for:
  - the cross-variant applicability matrix mapping foundation rules (`10`–`17`) onto every `DataKind` variant (`Simple` / `Grainset` / `Unionset` / `Joinset`)
  - the per-variant planner-strategy summary (pointers into `21 §4`, `22 §4`–`§5`, `23 §4`, `24 §5`); strategy semantics themselves are NOT ratified here
  - the per-variant rollup-legality summary (Grain × TemporalShape × strategy); detailed rules remain in `13 §3`, `17 §4`, `22 §4.3`, `23 §7`, `24 §5.4`
  - the per-variant Coverage surface summary: `Binding`-level `Coverage` for `Simple` vs `CompositionCoverage` / `FieldProvenance` for the three Complex variants
  - the per-variant error-code-band index (`*_E_2000–2099` shared; `*_E_2100–2499` per-variant; `*_E_2500–2599` reserved for `25`'s cross-variant diagnostics)
  - the scope boundary rule: `25` never re-specifies a rule owned by `10`–`17` or `20`–`24`; it only indexes them
  - the cross-doc-fix list surfaced while drafting (`§1.3`); items listed there are NOT resolved by `25` — they are flagged for the owning doc's next revision
refined-by:
  - 30 (`apis/30_api_contracts.md` — final placement of `*_E_2500–2599` in the cross-subsystem code-range table)
  - 33 (`apis/33_semstrait_manifest.md` — `ResolvedDataKind` / `ResolvedSimpleDataKind` / `ResolvedComplexDataKind` struct rosters)
  - 34 (`apis/34_semstrait_planner.md` — `Strategy` trait and per-variant strategy public surface)
  - 17 (`foundations/17_temporal_shape.md` — as-of activation matrix, shape-gated composition rules; `§2.9`'s cells tighten when `17` closes round-1 items)
---

# 25. Applicability Matrix

> **Reconciliation (Phase-3 / 2026-04-17 consolidation).** The per-variant authoring shape is ratified across [`../apis/32_semstrait_model.md §3`](../apis/32_semstrait_model.md), [`../foundations/18_entities.md`](../foundations/18_entities.md), and [`26_nesting_matrix.md`](./26_nesting_matrix.md) (R1 / R2 / R3 structural rules). The cross-variant matrix below remains authoritative as the index of which foundation rules apply to which variant; where body text cites `ColumnMapping` or pre-`18` struct names, apply the renames per `18 §10` (`SemanticMapping`) and `18 §2` (unified `Relationship`). `JoinType::AsOf` cells are forward-reference only (v1 roster excludes AsOf).

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [The Matrix — Foundation Rule × DataKind Variant](#2-the-matrix--foundation-rule--datakind-variant)
3. [Per-Variant Planner-Strategy Summary](#3-per-variant-planner-strategy-summary)
4. [Per-Variant Rollup-Legality Summary](#4-per-variant-rollup-legality-summary)
5. [Per-Variant Coverage & NULL-Fill Summary](#5-per-variant-coverage--null-fill-summary)
6. [Per-Variant Error-Code Bands](#6-per-variant-error-code-bands)
7. [Out-of-Scope for `25`](#7-out-of-scope-for-25)
8. [Round-1 Open Items](#8-round-1-open-items)

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
| `CDF-21-01` | `21 §2.1`'s `DataKind` enum | `21 §2.1` shows a **flat** 4-arm `DataKind { Simple, Unionset, Grainset, Joinset }` enum. `20 §2.1` and `23 §2.1` ratify a **two-level** shape `DataKind { Simple(_), Complex(ComplexDataKind) }` with `ComplexDataKind { Unionset, Grainset, Joinset }`. The two cannot both be right. Per `00 §6.3`'s read-order / directionality rule, `20`'s two-level shape is the earlier ratification and therefore authoritative; `21 §2.1` needs a doc edit. | `21`. |
| `CDF-21-02` | `21 §2.1` / `22 §2.1` / `23 §2.1` / `24 §2.2` | Inner variant struct names diverge: `21` uses `UnionsetDataKind` / `GrainsetDataKind` / `JoinsetDataKind`; `22` uses `GrainsetDataKind`; `23` uses `UnionsetDecl`; `24` uses `JoinsetDataKind`. `20 §2.1` uses `UnionsetSpec` / `GrainsetSpec` / `JoinsetSpec`. A single canonical name per variant must exist; `25 §2` and `§3` pick a non-normative default (`<Variant>DataKind`) consistent with `21` / `22` / `24`, but `23`'s `UnionsetDecl` and `20`'s `*Spec` both need reconciling. | `20` (authoritative for the two-level roster; each of `21`–`24` updates its own inner struct name). |
| `CDF-23-01` | `23 §1.1` and `23 §4.4` | Both cite `13 §7`'s "cast matrix" / "widening rules". `13 §7` (per the current `13` outline) is "Interaction with Other Docs" — there is no cast-matrix content under that anchor. The widening rules `23` consumes are authored across `13 §2.4` (shape unification) and `14a` (promotion lattice / cast policy). Either `13 §7` needs to grow a "Cast Matrix" subsection, or `23`'s refs should be retargeted to `13 §2.4` + `14a`. | `13` or `23` (editorial). |
| `CDF-17-01` | `17 §7.3` advisory roster × `22 §5` / `23 §6` | `22`'s `PLAN_W_2202 MixedShapeAdvisoryChildren` and `23`'s `COMP_W_2302`–`W_2306` shape-mismatch advisories overlap with `17 §7.3`'s advisory-warning roster without a single owning code. A cross-variant `TemporalShape` advisory shared across `Grainset` / `Unionset` (and, once `17 §5` lands, `Joinset`) is a natural `25 §2.9` cell — but the code-owning doc is unclear. | `17` owns the advisory taxonomy; the per-variant `PLAN_W_*` codes in `22` / `23` / `24` are the emission-site labels. `25` does not resolve; see Q1 in `questions/open/25_questions.md`. |

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

Every variant participates in every pipeline stage. Per-variant specialization lives inside `validate_structure` / `compile_into` / `Strategy::resolve` hooks ratified in `20 §2.2` / `§5`.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `10 §3.1` `parse` | `always` — recognizes `datasets:` discriminator (`12 §3`–`§6`). | `always` — recognizes `grainsets:` discriminator (`12 §4`). | `always` — recognizes `unionsets:` discriminator (`12 §3`). | `always` — recognizes `joinsets:` discriminator (`12 §5`). |
| `10 §3.2` `validate` | `always` — structural + `VALID_E_2100`–`2199` (`21 §7`). | `always` — `VALID_E_2200`–`2299` (`22 §7`). | `always` — `VALID_E_2300`–`2399` (`23 §8`). | `always` — `VALID_E_2400`–`2499` (`24 §9`). |
| `10 §3.3` `compile` | `always` — one `Binding` resolution per `15 §10`; emits `ResolvedSimpleDataKind` (`21 §2.3`). | `always` — per-child `Binding` resolution + level-sorted indices + `ComposedSemanticInterface` build (`22 §8.1`). | `always` — per-child resolution + `CompositionCoverage` fold + `ResolvedUnionset` (`23 §9`). | `always` — path resolution + `JoinType`-override check + materialized `ComposedSemanticInterface` (`24 §10`). |
| `10 §3.4` `plan` | `always` — `SimpleStrategy` (`21 §4`). | `always` — `GrainsetStrategy` dispatches single child (`22 §4` / `§10`). | `always` — `UnionsetStrategy` emits `PlanNode::Union` over children (`23 §4`). | `always` — `JoinsetStrategy` lowers anchor-outward hops to `PlanNode::Join` (`24 §5`). |
| `10 §3.5` `optimize` | `always` — variant-agnostic; operates on `PlanNode` tree (`10 §3.5`). | `always` — same. | `always` — same. | `always` — same. |
| `10 §3.6` `adapt` | `always` — variant-agnostic; lowers `PlanNode` → `EngineArtifact` (`10 §3.6`). | `always` — same. | `always` — same. | `always` — same. |

Every `plan` / `optimize` / `adapt` cell is identical by construction — `20 §4.2`'s shared skeleton and I3 / I6 together forbid variant-specialization past the strategy-dispatch boundary.

Per-variant specialization within each stage is confined to the hooks named in `20 §5.2`'s `DataKindOps` trait:

| Stage | Shared skeleton | Per-variant hook |
|---|---|---|
| `parse` | `parse_yaml_tree` (`10 §3.1`) | `DataKindOps::deserialize` (`20 §5.2`). |
| `validate` | `run_structural_checks` accumulates diagnostics (`10 §3.2`). | `DataKindOps::validate_structure` (`20 §5.2`) — per-variant structural preconditions in `21 §7` / `22 §7` / `23 §8` / `24 §9`. |
| `compile` | `resolve_bindings` + `resolve_interfaces` (`10 §3.3`). | `DataKindOps::compile_into` — emits `ResolvedSimpleDataKind` (`21 §2.3`) / `ResolvedGrainsetDataKind` (`22 §2.2`) / `ResolvedUnionset` (`23 §2.3`) / `ResolvedJoinset` (`24 §2.4`). |
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
| Resolution-variant fields (`11 §5.2`) | Per-Simple `Binding.column_mapping[].expr` may vary. | Per-child resolution varies — each Grainset level has its own `Binding` resolution. | Per-child resolution varies. | Per-member resolution varies; cross-member expressions use `14b §4.5`'s `PathSignature`. |

### 2.4 Nesting Policy — `12`

`12` ratifies the nesting matrix (`12 §2`) plus each `ComplexDataKind` variant's block shape (`12 §3`–`§5`) and the `Simple` leaf rule (`12 §6`). The matrix's structural rule per `20 §4.6`'s projection is reproduced below.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `12 §2` nesting matrix — as parent | `n/a` — Simple never nests children (`12 §6`). | `always` — may contain `Simple` / `Unionset` / `Joinset`; same-variant self-nest banned (`12 §2` row `Grainset per level`). | `always` — may contain `Simple` / `Grainset` / `Joinset`; same-variant self-nest banned. | `always` — may contain `Simple` / `Unionset` / `Grainset` as members; same-variant self-nest banned. |
| `12 §2` nesting matrix — as child | `always` — may nest under any `ComplexDataKind` (`12 §6.2`). | `always` — may be a Unionset branch or Joinset member; **deferred** as a Grainset child per `22 §3.4` `TD-GRAINSET-NESTED`. | `always` — may be a Grainset child or Joinset member; banned as a Unionset child (same-variant). | `always` — may be a Unionset branch or Grainset child; banned as a Joinset member (same-variant, `12 §2`). |
| `12 §3` Unionset block shape | `n/a`. | `n/a`. | `always` — `unionsets:` block w/ ≥ 2 children (`12 §3.2`, `23 §2.1`). | `n/a`. |
| `12 §4` Grainset block shape | `n/a`. | `always` — `grainsets:` block w/ ordered levels (`12 §4.2`, `22 §2.1`). | `n/a`. | `n/a`. |
| `12 §5` Joinset block shape | `n/a`. | `n/a`. | `n/a`. | `always` — `joinsets:` block; binary arity in v1 per `12 §5.2` / `24 §2.5`. |
| `12 §6` Simple nesting | `always` — terminal leaf; no children allowed (`12 §6.2`). | `n/a` — Grainset is not a leaf. | `n/a`. | `n/a`. |

The canonical nesting matrix from `12 §2` is reproduced here in projection form for reader convenience; authoritative content is in `12 §2`:

| Parent \ Child | `Simple` | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `Simple` | n/a — no children (`12 §6`). | n/a. | n/a. | n/a. |
| `Grainset` (per-level) | Legal (`12 §4.3`). | **Banned** — same-variant self-nest. Also [TD-GRAINSET-NESTED] (`22 §3.4`). | Legal. | Legal. |
| `Unionset` (per-branch) | Legal (`12 §3.2`). | Legal. | **Banned** — same-variant self-nest. | Legal. |
| `Joinset` (per-member, v1 binary) | Legal (`12 §5.3`). | Legal. | Legal. | **Banned** — same-variant self-nest. |

### 2.5 Types and Grain — `13`

All variants carry logical types. `Grain` participates in every variant but only `Grainset` is **grain-aware** at the DataKind level (per `20 §4.5` Invariant D6).

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `13 §2` `DataType` catalog | `always` — on every Semantics + every `Binding` column (`21 §2.2`). | `always` — on every composed-surface Semantics (`22 §2.1`). | `always` — on every composed-surface Semantics; `23 §4.4` reconciles across children. | `always` — on every composed-surface Semantics + every `Relationship.keys` (`16 §12.2`). |
| `13 §3` `Grain` enum + total coarseness order | `conditional` — optional `grain:` field on `SimpleDataKind` (`21 §2.2` / `§6`); shape-gated per `17`. | **`always`** — grain-axis binding is mandatory (`22 §2.1`'s `grain_axis`); child selection uses `Grain::coarseness()` (`22 §4.2`). | `conditional` — children should share a common grain or be rollable to a common coarsest per `17`; advisory `COMP_W_2308` (`23 §7`). | `conditional` — per-member grain is each member's concern; Joinset inherits anchor grain as the join's output grain by default (`24 §5.4`). |
| `13 §4` `DimensionType` discriminator | `always` — authored on every Dimension (`13 §4`). | `always`. | `always`. | `always`. |
| `13 §5` Keys vs Dimensions | `always` — Keys are declared separately from Dimensions (`13 §5.1`). | `always`. | `always`. | `always`. |
| `13 §2.4` shape unification (authoritative cast rules) + `14a` cast policy | `always` — used at the Binding boundary (`15 §9`). | `always` — used during child-shape-conflict checks (`22 §8` `COMP_E_2204` / `COMP_E_2206`). | `always` — used during cross-child type reconciliation (`23 §4.4` / `§9.4`). `[CROSS-DOC-FIX-NEEDED]`: `23` cites `13 §7` as the cast-matrix owner; see `§1.3 CDF-23-01`. | `always` — used during `Relationship.keys` type agreement (`16 §12.2`). |

The `13 §7` row in the user's spec ("Joinset column type-compat: only Joinset") maps onto the actual authoritative sites: `13 §2.4` (shape unification) + `16 §12.2` (`KeyPair` type agreement). `Joinset` is where the `Relationship.keys` type agreement bites most often in practice, but the underlying rule is not Joinset-specific — it applies to any consumer of a `Relationship`, including implicit composition (`16 §11`). `25` records both the narrow (Joinset-only) and broad (any `Relationship` consumer) readings; the narrow reading lives in `24 §5.2` step 3, the broad one in `16 §12.2`.

### 2.6 Expressions — `14`, `14a`, `14b`

All three expression docs apply to every variant. Authoring happens on `Simple`s directly; `Complex` variants inherit the resolved expressions via their constituents.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `14 §2` `SemanticExpr` / `PhysicalExpr` split | `always` — authored at `Binding.column_mapping[].expr` + Semantics-level `expr:` (`21 §4.4`). | `via Simple children` — authored per child; Grainset declares no Bindings (`20 §4.3`). | `via Simple children` — same. | `via Simple children` — same; Joinset-level derived fields use `FieldOwnership::Derived` at the composed surface (`24 §2.6`). |
| `14 §3` shared `Expr` AST | `always`. | `always` — consumed through child sub-plans. | `always`. | `always` — consumed in `KeyPair` predicate lowering (`24 §5.2` step 3) + Joinset-level derived fields. |
| `14 §4` `ExprSource` YAML grammar | `always`. | `always` — only in per-level declarations that live on `Simple` children. | `always` — only in per-branch declarations that live on `Simple` children. | `always` — only in per-member declarations + in Joinset-level derived fields. |
| `14 §5`–`§6` typing + computed-Dimension `data_type:` inference | `always` — consumed at Computed-Dimension inference per `14 §6.2`. | `via Simple children`. | `via Simple children` + cross-child widening at `UnifiedSemantics` build (`23 §4.4`). | `via Simple children` + `KeyPair` type agreement (`16 §12.2`). |
| `14a` `FunctionRegistry` / `CanonicalFn` | `always`. | `always`. | `always` — used at cross-child widening LUB (`23 §4.4`). | `always`. |
| `14b` `ResolvedExprTable` + cross-DataKind path pre-resolution | `always` — per-`(Semantics, Binding)` pre-resolved at compile (`14b §2`). | `via Simple children` + composed-surface lifts where a Grainset-level Semantics refers to a child's field. | `via Simple children` + composed-surface lifts where cross-child shape reconciliation wraps a `Cast` (`23 §4.4`). | `via Simple children` + Joinset-level `Derived` expressions pre-resolved + `PathSignature` (`14b §4.5`) for cross-member expressions. |

Round-1 decision recorded: no `Complex` variant authors `SemanticExpr` / `PhysicalExpr` directly on a `Binding`; every such expression lives on a `Simple` and is composed up. The only Complex-authored expressions are (a) Joinset-level `Derived` fields (`24 §2.6`) and (b) Joinset-level Filter clauses applied post-join (`24 §5.5` step 4).

`14b`'s `PathSignature` (`14b §4.5`) is a `BTreeSet<RelationshipPath>`; only `Joinset` and the implicit-composition consumer in `16 §11` populate it with `len() > 0`. Bare `Simple` and single-constituent Complex composition produce the empty signature.

| `14` axis | Authored where? | Pre-resolved where? |
|---|---|---|
| `PhysicalExpr::Column` (ColumnMapping value) | `Simple.binding.column_mapping[].value` (`15 §5`). | `compile`; result keyed by `ResolvedExprKey` (`14b §2.1`). |
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
| `15 §6` Coverage — Binding-level | `always` — one `Coverage` entry per `(Semantics, PhysicalSource)` (`15 §6.1`). | `via Simple children` — then lifted to `CompositionCoverage` per `16 §8.4` and `22 §6.1`. | `via Simple children` — then lifted per `23 §5`. | `via Simple children` — then lifted per `24 §8.4`. |

Coverage gets re-visited at the composition layer in `§2.8` row `16 §8`; the Binding-level cell above is the leaf-per-`PhysicalSource` rule.

`15`'s `Binding`-per-`Simple` rule is load-bearing enough to warrant an explicit table mapping the canonical Manifest-layer counterpart (`15 §7`) onto each variant:

| `15` Manifest counterpart | Owned by | Consumers |
|---|---|---|
| `ResolvedColumnMapping` (`15 §7`) | `ResolvedSimpleDataKind.binding` (`21 §2.3`). | Every variant's plan emission reaches it via its Simple children. |
| `ResolvedPhysicalSource` (`15 §7`) | Same. | Same. |
| `ResolvedCoverage` (`15 §6` resolution) | Same. | `CompositionCoverage` fold in `16 §8.4` / `22 §6.1` / `23 §5` / `24 §8.4`. |
| `ResolvedSchema` (`15 §3.2`) | Same. | Type-agreement checks in `23 §4.4`, `24 §5.2` step 3. |

### 2.8 Composition — `16`

`16` is the home of `Relationship`s, `Cardinality`, `JoinType`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, and the explicit-vs-implicit composition boundary.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `16 §2` `Relationship` — declaring | `n/a` — Relationships are top-level; DataKinds do not declare them (`16 §2.1`). | `n/a`. | `n/a`. | `n/a` — Joinset **consumes** Relationships; never declares (`24 §6`). |
| `16 §2` `Relationship` — consuming via implicit composition | `conditional` — when a Request spans multiple top-level DataKinds, any `Simple` may participate as a constituent (`16 §11.4`). | `conditional` — same; a Grainset exposes a composed surface that can participate as a constituent in implicit composition (`22 §1.2`). | `conditional` — same. | `always` — `Joinset.path` references Relationships by id (`24 §6`). |
| `16 §3` `Cardinality` | `conditional` — consumed only when the `Simple` participates in implicit composition. | `conditional` — same. | `conditional` — same; also consumed in `Additivity × Cardinality` post-Union re-aggregation checks (`23 §7.4`). | `always` — per-hop walked `Cardinality` drives fan-out advisories (`24 §3.4` / `§5.4`). |
| `16 §4` `JoinType` | `conditional` — used only in implicit composition. | `conditional` — same. | `conditional` — same. | `always` — per-hop `JoinType` + `24 §5.3`'s override-legality matrix. |
| `16 §5` `ComposedSemanticInterface` | `n/a` — Simple exposes bare `SemanticInterface` (`20 §4.4` Invariant D5). | `always` — `composition_kind == Grainset` (`22 §2.3`). | `always` — `composition_kind == Unionset` (`23 §2.3`). | `always` — `composition_kind == Joinset` (`24 §2.4`). |
| `16 §6` `UnifiedSemantics` merge rules | `n/a`. | `always` — trivial merge (children share the composed surface) (`22 §2.3`). | `always` — merge over heterogeneous children + cross-child name-collision policy (`23 §4.4` / `§5`). | `always` — merge with namespacing for shape-incompatible collisions (`24 §8.2`). |
| `16 §7` `FieldProvenance` — `Native` / `Shared` / `Derived` | `n/a` — no provenance on a bare interface. | `always`. | `always`. | `always`. |
| `16 §7.3.3` `FieldOwnership::NullFill` | `n/a`. | `n/a` — Grainset selection never emits `NullFill` in FieldProvenance (Coverage-layer `NullFill` is distinct; see `§2.7` row `15 §6`). | **`always`** — Unionset is the only `CompositionKind` that emits `FieldOwnership::NullFill` (`16 §7.3.3`, `23 §5.5`). | `n/a` — Joinset NULL-fill is carried by `JoinType` outer-join semantics, not by `FieldOwnership` (`24 §8.3`). |
| `16 §8` `CompositionCoverage` | `n/a` — `Coverage` lives at the Binding level for Simple. | `always` — per-child projection onto the composed surface (`22 §6.1`). | `always` — per-child fold per `23 §5`. | `always` — per-member fold per `24 §8.4`. |
| `16 §9` explicit vs implicit composition | `n/a` — Simple is a leaf, neither explicit nor implicit. | **explicit** — Grainset is an explicit Complex (`16 §9.2`). | **explicit** — Unionset is an explicit Complex. | **explicit** — Joinset is the canonical explicit-composition DataKind (`16 §13`). |
| `16 §10` materialization policy | `n/a`. | `always` — materialized at compile (`22 §8.1`). | `always` — materialized at compile (`23 §9`). | `always` — materialized at compile (`24 §2.4`). |
| `16 §11` field-first resolution (entry point when `Request.from: None`) | `always` — a Simple may be the sole owner of a requested field (`16 §11.3` single-kind fast path). | `always` — a Grainset's composed surface can own requested fields; implicit composition may layer above it (`22 §1.2`). | `always` — a Unionset's composed surface owns its declared fields; implicit composition may layer above. | `always` — a Joinset's composed surface owns its declared fields; author uses `from: Some(joinset)` to target the pre-built surface (`24 §1.3`). |
| `16 §12` Relationship-graph well-formedness | `conditional` — Simple participates as a Relationship endpoint. | `conditional` — same. | `conditional` — same. | `conditional` — Joinset consumes the resulting graph (`24 §6`). |

The per-variant interface-exposure invariant (`20 §4.4` D5) is reproduced by the `16 §5` row: `Simple` → `n/a`, every `Complex` → `always`.

The `CompositionKind` enum distribution across the matrix is summarized in `16 §5.3`:

| `CompositionKind` variant | Emitted by | `25` row pointer |
|---|---|---|
| `CompositionKind::Relationship` | `16 §11` implicit composition (any variant mix). | `§2.8` row `16 §11`. |
| `CompositionKind::Joinset` | `Joinset` explicit composition (`24 §2.4`). | `§2.8` row `16 §5`, Joinset column. |
| `CompositionKind::Unionset` | `Unionset` explicit composition (`23 §2.3`). | `§2.8` row `16 §5`, Unionset column. |
| `CompositionKind::Grainset` | `Grainset` explicit composition (`22 §2.3`). | `§2.8` row `16 §5`, Grainset column. |

A `Simple` never produces a `CompositionKind`; it exposes only a bare `SemanticInterface` per `20 §4.4` D5.

### 2.9 Temporal Shape — `17`

Vocabulary ratified in `17`; planner-side support is DEFERRED in places (see `00 §4.1` `TemporalShape` row + `17 §1.4`). `17 §*` placeholders track sections whose numbering is still landing.

| Clause | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `17 §2` `TemporalShape` vocabulary | `always` — as an optional declaration axis on the DataKind. | `always` — same. | `always` — same. | `always` — same. |
| `17 §3.1` declaration site on `SimpleDataKind` | **`always`** — declared inline via `temporal_shape:` (`21 §5.1`). | `n/a` — Grainset itself does not carry a top-level shape; each child may (`17 §3.2`). | `n/a` — same. | `n/a` — same. |
| `17 §3.2` Complex does NOT carry its own `TemporalShape` | `n/a`. | `always` — shape derived from children per `17`. | `always` — same; mixed-shape branches emit advisories (`23 §6.1`). | `always` — shape derived from the two members; `AsOf` activation per `17 §5` consults both sides. |
| `17 §4` shape × `Grain` rollup matrix | `conditional` — legality gated per `17 §4.1`'s matrix; shape-less Simples default-legal (`21 §6.3`). | **`always`** — gates per-child eligibility in `22 §4.3` `ROLLUP_LEGAL` (`SnapshotRollupWithoutPin`, `SCDRollupWithoutAsOf`). | `conditional` — gates post-Union rollup shape per `23 §7` (`PLAN_E_2302`). | `conditional` — per-member rollup is each member's concern. |
| `17 §5` `AsOf` `JoinType` | `n/a` — Simple has no `JoinType`. | `conditional` — when a Grainset child is itself an SCD, the grainset-level rollup requires as-of anchoring per `22 §5` `SCDRollupWithoutAsOf`. | `conditional` — cross-shape branches may require as-of reconciliation (`23 §6.3`, `[TD-UNIONSET-SHAPE-PLANNING]`). | **`always`** — `JoinType::AsOf` activation matrix gates per-hop overrides (`24 §7`, `COMP_E_2412`–`COMP_E_2414`). Also fires for implicit Relationship composition under `16 §11` — see next row. |
| `17 §5` `AsOf` across implicit Relationship composition | `n/a`. | `conditional` — `16 §11`-synthesized composition over a Grainset constituent inherits the same as-of gating. | `conditional` — same. | `conditional` — redundant with explicit Joinset `AsOf`. |
| `17 §6` `Request.temporal` block (`as_of:`, time-range overrides) | DEFERRED per `17 §6.5` — no variant consumes in v1. | DEFERRED. | DEFERRED. | DEFERRED. |
| `17 §7` `Additivity × TemporalShape` advisories | `conditional` — emits advisory when Simple's Measure `Additivity` and the Simple's `TemporalShape` appear inconsistent (`21 §5.2` / `PLAN_W_2102`). | `conditional` — emits advisory per child at strategy time (`22 §9.2` `PLAN_W_2202` mixed-shape). | `conditional` — emits advisories for cross-child shape mismatch (`23 §6.1` `COMP_W_2302`–`W_2306`). `[CROSS-DOC-FIX-NEEDED]`: `§1.3 CDF-17-01`. | `conditional` — emits advisory when `AsOf` is activated over a hop whose declared `JoinType` was non-`AsOf` (`24 §11.2` `PLAN_W_2404`). |
| `17 §8` shape-gated composition rules (`17 §8.1` Unionset branches; `§8.2` Grainset levels; `§8.3` Joinset hops) | `n/a`. | `always` — Grainset levels are shape-gated per `17 §8.2` (consumed by `22 §4.3`). | `always` — Unionset branches are shape-gated per `17 §8.1` (consumed by `23 §6`). | `always` — Joinset hops are shape-gated per `17 §8.3` (consumed by `24 §7`). Implicit Relationship composition inherits the same gating via `16 §11` + `17 §5`. |

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
| I1 (no raw SQL in Model / Manifest) | `always`. | `always`. | `always`. | `always`. |
| I2 (one canonical `Expr` AST) | `always` — per `14 §3`. | `always`. | `always`. | `always`. |
| I3 (variant-agnostic `PlanNode` IR) | `always`. | `always`. | `always`. | `always`. |
| I4 (Manifest deterministic) | `always`. | `always` — strategy output deterministic per `22 §10`. | `always`. | `always` — explicit-path hop order is deterministic per `24 §4.2`. |
| I5 (all resolution at compile) | `always`. | `always`. | `always`. | `always`. |
| I6 (plan hot path synchronous) | `always`. | `always`. | `always`. | `always`. |
| I7 (single adapter surface `lower_plan`) | `always`. | `always`. | `always`. | `always`. |
| I8 (Manifest is planner-complete) | `always`. | `always`. | `always`. | `always`. |
| I9 (Session-context-based overrides) | `always`. | `always`. | `always`. | `always`. |
| I10 (`#[non_exhaustive]` on public sum types) | `always` — e.g. `DataType`, `Grain`, `TemporalShape`, `ColumnMappingValue`. | `always`. | `always`. | `always` — `JoinType`, `Cardinality`. |
| I11 (separation of Model / Manifest / Planner / IR) | `always`. | `always`. | `always`. | `always`. |
| I12 (first-class diagnostics) | `always` — `VALID_E_21xx` / `COMP_E_21xx` / `PLAN_E_21xx` / `PLAN_W_21xx`. | `always`. | `always`. | `always`. |

### 2.11 Category cross-cuts — `19`

The category axis ratified in [`19`](../foundations/19_categories.md) (Dimension `DimensionType`, `MeasureCategory`, `MetricCategory`) is variant-agnostic at the schema level — every DataKind variant accepts every category. The cross-cuts that **do** vary per variant are operational:

| Category axis | `Simple` / Dataset | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| `DimensionType::Temporal(TemporalDimensionBody)` (`19 §2.2`) | `always` — Temporal Dimension on a Simple's interface; rollup grains live in `body.grains`. | **`always`** — Grainset's `grain_axis` MUST resolve to a Temporal Dimension whose `body.grains` covers the level set (`22 §4.2`). | `conditional` — Temporal Dimensions across children are reconciled per `23 §4.4` (cast-up to LUB). | `conditional` — Temporal Dimensions per member; cross-member temporal alignment falls under `17 §5` (`AsOf` joins, deferred). |
| `DimensionType::{Categorical, Binary, Geo}` | `always`. | `always`. | `always`. | `always`. |
| `DimensionType::Bucketed(BucketedDimensionBody)` | `always` — `CASE WHEN` projection emitted at compile (`19 §2.2`). | `always`. | `always`. | `always`. |
| `DimensionType::Metadata(MetadataDimensionBody)` | `always` — extracted at scan-bind (`15 §8`). | `via Simple children`. | `via Simple children`. | `via Simple children`. |
| `MeasureCategory::Additive` / `MinMax` / `Boolean` (`19 §3.3`) | `always`. | `always`. | `always`. | `always`. |
| `MeasureCategory::Average` (non-additive — re-aggregate from SUM/COUNT at queried grain) | `always`. | `conditional` — re-aggregation runs at the queried Grainset level; planner materializes SUM/COUNT once (`22 §10`). | `conditional` — Union-composed Average needs per-child SUM/COUNT decomposition (`23 §4.5` lossy-reagg). | `always` — runs post-join. |
| `MeasureCategory::Distinct(DistinctMeasureBody)` (non-additive; engine-specific approx) | `always`. | `conditional` — distinct count over a non-axis grain may force a coarser Grainset level (`22 §4.4`). | `conditional` — Union-composed distinct **does not** decompose; planner must materialize at the union output (`23 §4.5`). | `always` — runs post-join. |
| `MeasureCategory::Statistical(StatisticalMeasureBody)` (StdDev / Variance / Median / Percentile) | `always`. | `conditional` — same as Average (recompute at queried grain). | `conditional` — Union-composed stat measures decompose only when the engine supports the partial-aggregate form (`23 §4.5`). | `always` — runs post-join. |
| `MeasureCategory::Snapshot(SnapshotMeasureBody)` (semi-additive) | `always` — synthesizes `AdditivityType::Semi` per `19 §3.3`. | `conditional` — non-additive axis MUST be among the Grainset's `grain_axis` members or a sibling Dimension; otherwise `validate.snapshot-axis-not-on-host`. | `conditional` — Union-composed Snapshot requires per-child axis agreement (advisory). | `always` — runs post-join. |
| `MeasureCategory::Custom` | `always` — author states `agg:` + `additivity:` manually. | `always`. | `always`. | `always`. |
| `MetricCategory::Simple(SimpleMetricBody)` (`19 §4.2`) | `always`. | `always` — inherits the wrapped Measure's variant cross-cut. | `always`. | `always`. |
| `MetricCategory::Ratio(RatioMetricBody)` | `always` — materializes numerator + denominator separately, combines in a post-aggregate `Project`. | `always` — same. | `conditional` — Union-composed ratio requires per-child decomposition of numerator + denominator (advisory if one child is missing one side). | `always` — runs post-join. |
| `MetricCategory::Derived(DerivedMetricBody)` | `always`. | `always`. | `conditional` — Union-composed Derived may require lossy-reagg per `23 §4.5`. | `always`. |
| `MetricCategory::Cumulative` (post-v1; cross-cuts `00 §10` window-function deferral) | `n/a` (post-v1). | `n/a` (post-v1) — when ratified, requires the host DataKind to have a Temporal axis; planner-direction hint. | `n/a` (post-v1). | `n/a` (post-v1). |
| `MetricCategory::Conversion` (post-v1) | `n/a` (post-v1). | `n/a`. | `n/a`. | `n/a`. |

The `conditional` cells point at planner moves that the per-variant `2x` docs already detail; `25 §2.11` only records that the move exists. Concrete planner emission per category lives in [`19 §3.3`](../foundations/19_categories.md#33-implicit-constraint-contract-per-measure-category) / [`19 §4.2`](../foundations/19_categories.md#42-implicit-constraint-contract-per-metric-category).

---

## 3. Per-Variant Planner-Strategy Summary

One subsection per variant. Each summary is three-to-five bullets naming the strategy's key moves + a pointer into the authoritative section. Strategy semantics themselves are NOT ratified here; `§3` is a reader's convenience.

Across all four strategies, four discipline points hold uniformly (per `20 §5.2` `Strategy` trait + `10 §3.4` `plan` stage + `10 §8`):

- **Synchronous hot path** per I6. No `.await` in `Strategy::resolve`; no blocking I/O; no `PhysicalSource` probe.
- **Manifest-complete input** per I8. Every datum the strategy needs (interface, binding, coverage, provenance, relationships, temporal shape) is already in the `Manifest` by the time `plan` runs.
- **Deterministic output** per I4. Given identical `(Manifest, Request, SessionContext)`, `Strategy::resolve` produces an identical `PlanNode` tree.
- **Emits diagnostics, not panics** per I12. Unreachable / invariant-violation states emit a `PLAN_E_*` diagnostic; panics are reserved for implementer-bug assertions.

Every strategy's ratification doc repeats these points in its own voice; `§3` does not re-state them per-strategy.

### 3.1 `SimpleStrategy` — `Simple` / Dataset

- **Fast path.** `Simple` is the only variant that owns a `Binding`; its strategy emits a single `PlanNode::Scan` subtree (plus optional `Project` / `Agg` / `Project` layers per the 5-layer plan shape of `21 §4.1`).
- **5-layer canonical shape.** L1 `Scan` → L2 `Rename` → L3 `Expression` → L4 `Aggregate` → L5 `Project`. Every layer except L1 is optional per the skip rules in `21 §4.7`.
- **Multi-`PhysicalSource` fan-out.** One `Scan` per source; union at L1 when glob expansion yields ≥ 2 sources with uniform Coverage (`21 §3.2` / `§4.2` step 3).
- **No composition machinery.** `SimpleStrategy` never walks `Relationship` / `Cardinality` / `CompositionCoverage`; those belong to Complex strategies. Simple participates in composition as a **constituent**, not as a composer.
- **Filter placement.** Filter clauses (`11 §6.4`) are placed between L1 (`Scan`) and L2 (`Rename`) when they reference only pre-rename column names, or between L3 (`Expression`) and L4 (`Aggregate`) when they reference computed expressions (`21 §4.1` / `§4.4`).
- **Authoritative section:** `21 §4` (with the skip-rule table in `21 §4.7` and the worked example in `21 §10`).

### 3.2 `GrainsetStrategy` — `Grainset`

- **Single-child delegation.** Grainset is a **router**, not a composer of rows. The strategy runs `REQUEST_GRAIN_EXTRACT` → `ELIGIBILITY` → `COST` → `CHOOSE`; exactly one child wins (`22 §10.1`).
- **Coverage-driven eligibility.** A child is eligible iff its `grain ≤ request.grain` AND every requested Semantics has `Coverage ∈ {Native, Derived}` on that child (`22 §4.2`). The `semantics_to_covering_children` index makes this an O(1) probe per Semantics (`22 §2.2`).
- **Shape-gated rollup legality.** Per `17 §4`'s matrix: `Timeseries`/`Events` roll freely, `Snapshot` pins at source grain (requires pin policy), `SCD` requires as-of anchoring. Errors in `22 §9.1` `PLAN_E_2205` / `PLAN_E_2206`.
- **Deterministic tie-break.** Source-count cost proxy; ties break by declaration order. Advisory `PLAN_W_2201 TieBrokenByOrder`.
- **Plan shape is a splice, not a node.** The chosen child's sub-plan is spliced into the position where the Grainset was queried; there is no `PlanNode::Grainset` (`22 §10.5`).
- **Rollup policy shape.** `RollupPolicy::Auto` (default; pick any eligible child), `PinOnly` (forbid rollup for `Snapshot` children), `AsOfRequired` (require SCD anchoring). Policy consumed at strategy time per `22 §5`.
- **Authoritative section:** `22 §4` (selection algorithm) + `22 §5` (`TemporalShape` gating) + `22 §10` (plan shape).

### 3.3 `UnionsetStrategy` — `Unionset`

- **UNION ALL / UNION DISTINCT over children.** Emits `PlanNode::Union { distinct, inputs }` where `inputs[i]` is the `i`-th child's strategy output wrapped by a per-child `Project` seam (`23 §4.1`).
- **NULL-fill at the seam.** For every composed-surface field a child does NOT provide, the per-child `Project` emits `PhysicalExpr::Cast(Null, unified_type)` (`23 §4.3`); `unified_type` derives from the LUB across contributing children (`23 §4.4`).
- **Delegates per-child.** The strategy never reaches into a child's subplan construction — Simple children run `SimpleStrategy`, Grainset children run `GrainsetStrategy`, Joinset children run `JoinsetStrategy` (`23 §4.2`).
- **Coverage-driven branch pruning.** A child whose every Request-selected field is `NullFill` is pruned from the Union with advisory `PLAN_W_2301 UnionsetBranchPrunable`; exceptions for `COUNT(*)`-like queries (`23 §4.6`).
- **Terminal re-aggregation.** Post-Union `PlanNode::Aggregate` re-groups by requested Dimensions and re-aggregates Measures; skipped when no Measures requested, when metadata-Dimension values are source-distinguishing, or when branch-pruning collapsed the Union to a single branch (`23 §4.5`).
- **Authoritative section:** `23 §4` (strategy) + `23 §5` (Coverage semantics) + `23 §11` (worked example).

### 3.4 `JoinsetStrategy` — `Joinset`

- **Anchor-outward join emission.** The anchor's `Scan` is the plan-tree left of hop 0; each subsequent hop joins onto the running tree's right (`24 §5.2`). v1 is binary (`12 §5.2` / `24 §2.5`), so one hop.
- **Path resolution at compile.** Implicit path runs `RELATIONSHIP_BFS_ANCHORED`; explicit path validates `JoinHop` by `JoinHop`; both produce `Vec<ResolvedJoinHop>` before plan (`24 §4`).
- **`JoinType` with override-legality gate.** Default from `Relationship.join_type`; override allowed per `24 §5.3.3`'s matrix (`Inner` relaxable in any direction; `Left` / `Right` tightenable to `Inner` only; `Full` any tightening; `AsOf` immutable). Error `COMP_E_2411`.
- **Shape-gated `AsOf` activation.** Per-hop shape pairs consult `17 §5`'s activation matrix. `Events ↔ Snapshot` / `Events ↔ SCD` mandate `AsOf`; `Snapshot ↔ Snapshot` forbids it. Advisory `PLAN_W_2404` on silent activation (`24 §7`).
- **Post-join unified `Project`.** Consumes `UnifiedSemantics` (`16 §6`) for rename / namespacing and `FieldProvenance` (`16 §7`) for per-field ownership. Joinset-level Filter applies atop the Project (`24 §5.5`).
- **Authoritative section:** `24 §5` (strategy) + `24 §7` (`AsOf` integration) + `24 §8` (`ComposedSemanticInterface` shape) + `24 §12` (worked examples).

### 3.5 Cross-variant composition — worked pointer

A Joinset over a Grainset member is the canonical example of cross-variant composition (permitted by `12 §2` and `12 §5.1`). Intended plan emission (authoritative in the owning variants' plan-shape sections):

1. `RESOLVE_JOINSET_PATH` runs per `24 §4` — anchor and target members are resolved to their `ResolvedDataKind` entries.
2. For each Grainset member, `GrainsetStrategy` runs first to pick a winning child for the requested grain (`22 §4`).
3. The winning child's sub-plan is spliced in place of the Grainset (`22 §10.5`); the Joinset's `PlanNode::Join` then composes over the spliced subplans.
4. `AsOf` activation consults the spliced child's `TemporalShape` (not the Grainset's — Grainset has no top-level shape per `§2.9`).

`24 §12` and `22 §11` each carry worked versions of this composition from their side.

### 3.6 Strategy interaction table

A compact reading of how the four strategies interact along the axes that matter for cross-variant composition:

| Axis | `SimpleStrategy` | `GrainsetStrategy` | `UnionsetStrategy` | `JoinsetStrategy` |
|---|---|---|---|---|
| Delegates to child strategies? | No. | Yes — one (winner). | Yes — N (all branches). | No (delegates via inline Scan per member, no separate child strategy invocation in v1). |
| Emits a `PlanNode` of its own kind? | No — only shared nodes (`Scan` / `Project` / `Aggregate`). | No — splices chosen child's subplan (`22 §10.5`). | Yes — `PlanNode::Union` (`35 §*`). | Yes — `PlanNode::Join` (`35 §*`). |
| Requires per-member Coverage at plan? | No — single Binding (`15 §6`). | Yes — `CompositionCoverage` drives eligibility. | Yes — drives NULL-fill projection. | Yes — drives `FieldProvenance` at Project time. |
| Consumes `Relationship` at plan? | No. | No. | No. | Yes — per hop. |
| Consumes `TemporalShape` at plan? | Advisory only. | Yes — gates child eligibility. | Advisory (heterogeneity) + error (post-Union grain incompat). | Yes — gates `AsOf` activation. |
| Emits advisories (`PLAN_W_*`)? | Yes (`21 §9`). | Yes (`22 §9.2`). | Yes (`23 §10.2`). | Yes (`24 §11.2`). |

---

## 4. Per-Variant Rollup-Legality Summary

"Rollup" means transforming rows at one `Grain` into rows at a coarser `Grain` via bucket-then-aggregate (`DATE_TRUNC` + `GROUP BY` + additive aggregation). Rollup legality depends on `Grain` coarseness order (`13 §3.2`), `TemporalShape` (`17 §4`), and `Additivity` (`11 §7` / `17 §7`).

| Variant | Native rollup? | Rollup driver | Requires anchor? | Emits advisories? |
|---|---|---|---|---|
| `Simple` / Dataset | **Yes** — at `SimpleStrategy` L4 `Aggregate`. | Simple's declared `grain:` (`21 §6`) compared against Request's requested grain. | No. | `PLAN_W_2101 LossyMultiSourceReaggregation`; `PLAN_W_2102 ShapeAdditivityMismatch`. |
| `Grainset` | **Yes, at the strategy level** — `Grainset` **picks** a child whose grain rolls up to the Request's grain per `17 §4`'s matrix; the chosen child's own strategy then handles the rollup. | `GrainsetDataKind.grain_axis` + per-child `grain:` declarations. | **Yes** — implicit: the Grainset's `grain_axis` anchors the decision (`22 §2.1`). `Snapshot` children with `PinOnly` never roll up (`22 §5`). | `PLAN_W_2200 GrainsetRollupUnusedChild`; `PLAN_W_2203 RequestGrainAbsentUsingCoarsest`; `PLAN_W_2202 MixedShapeAdvisoryChildren`. |
| `Unionset` | **Indirectly** — the terminal `Aggregate` above `PlanNode::Union` performs the rollup. The Union itself has no grain; grain is determined by the common-coarsest of contributing branches per `23 §7`. | Children's grains reconciled at compile; Request grain compared against common-coarsest. | No in v1 (union mode is set-like, not anchor-biased). | `COMP_W_2308 UnionsetGrainDivergent`; `PLAN_E_2302 UnionsetGrainIncompatibleWithRequest`. |
| `Joinset` | **Inherits anchor grain** — the Joinset's output grain is the anchor's grain; target-side rollup is the target member's own concern (`24 §5.4`). | `Joinset.anchor`'s grain + per-member grain + `17 §5`'s `AsOf` activation. | **Yes** — the `Joinset.anchor` IS the rollup anchor (`24 §3.1` / `§3.4`). | `PLAN_W_2400 JoinsetFanoutAdvisory`; `PLAN_W_2402 ManyToManyHopAdvisory`; `PLAN_W_2403 MultiFanoutAdvisory`; `PLAN_W_2404 AsOfActivation`. |

Cross-refs: `17 §4` (shape × grain matrix), `22 §4.3` (`ROLLUP_LEGAL`), `23 §7` (Unionset post-Union rollup), `24 §7` (`AsOf`-gated Joinset hops).

### 4.1 Worked-pointer table

For a reader who prefers an example-first traversal, the authoritative rollup walk-throughs are:

| Variant | Worked example | Rollup path shape |
|---|---|---|
| `Simple` | `21 §10` multi-source Dataset rollup. | `Scan → Aggregate(group_by = Request.dimensions, agg = Measures)`. |
| `Grainset` | `22 §11` daily-vs-monthly rollup selection. | Delegates to the winning child; parent Grainset contributes no `PlanNode`. |
| `Unionset` | `23 §11` heterogeneous branches with NULL-fill. | `Union → Aggregate`. |
| `Joinset` | `24 §12` two-Simple anchor + target. | `Scan(anchor) → Join(target) → (optional Aggregate over anchor's grain)`. |

### 4.2 Shape-gated rollup legality — cross-reference

Per `17 §4.1`'s matrix (referenced by every variant that rolls up):

| Shape | Rollup discipline | Owning variant |
|---|---|---|
| `Timeseries` | Roll freely along the Grain axis (`13 §3.2`). | Consumed by `SimpleStrategy` L4 (`21 §4.5`) and `GrainsetStrategy` selection (`22 §4.3`). |
| `Events` | Roll to coarser buckets via `DATE_TRUNC`; additive only. | Same consumers. |
| `Snapshot` | `PinOnly` — no rollup across snapshots without a pin policy (`22 §5`). | `GrainsetStrategy` rejects (`PLAN_E_2205 SnapshotRollupWithoutPin`). |
| `SCD` | Rollup requires as-of anchoring (`17 §5`). | `GrainsetStrategy` rejects without as-of pin (`PLAN_E_2206 SCDRollupWithoutAsOf`). |

---

## 5. Per-Variant Coverage & NULL-Fill Summary

`Coverage` appears at two levels (per `15 §1.3` / `15 §6.4` / `16 §8.1`):

- **Binding-level** `Coverage` — the X-axis of source selection inside a `Simple` (`15 §6`). Records per-`(Semantics, PhysicalSource)` whether the source is `Native`, `Derived`, or `NullFill`.
- **Composition-level** `CompositionCoverage` — the per-`(ConstituentRef, UnifiedName)` fold over constituents' Binding-level `Coverage`s (`16 §8`). Records per-child / per-member coverage of the composed surface.

NULL-fill appears in two distinct axes:

- **`CoverageVariant::NullFill`** at the `Coverage` / `CompositionCoverage` layer (`15 §6.1` / `16 §8.4`): a source / constituent that does not serve a given Semantics.
- **`FieldOwnership::NullFill`** at the `FieldProvenance` layer (`16 §7.3.3`): a composition-level record of which constituents provide a field and which do not. **Per `16 §7.3.3`, only `Unionset` emits `FieldOwnership::NullFill`.** Joinset NULL-fill is carried by `JoinType` outer-join semantics at the plan tree, not by `FieldProvenance`.

| Variant | Coverage surface | NULL-fill in `FieldProvenance`? | NULL-fill in plan? |
|---|---|---|---|
| `Simple` / Dataset | **Binding-level** per-`(Semantics, PhysicalSource)` per `15 §6`. Heterogeneous `NullFill` across sources in a bare Simple is a compile error (`COMP_E_0310 UnusableNullFillInNonUnionContext`); author must wrap in a Unionset (`21 §3.2`). | `n/a` — bare `SemanticInterface` has no `FieldProvenance`. | `n/a` under bare Simple; `NullFill` is rejected before plan. |
| `Grainset` | **Per-child `CompositionCoverage`** — a projection of each child's Binding-level or composition-level Coverage onto the Grainset's composed surface (`22 §6.1`). `NullFill` Coverage is legal (a child may lack a Semantics the Request does not name). | No `FieldOwnership::NullFill` — Grainset chooses exactly one child; coverage drives the eligibility predicate (`22 §4.2`), not runtime NULL emission. | `n/a` — no cross-child row mixing; chosen child's rows flow directly. |
| `Unionset` | **Per-child `CompositionCoverage`**, optionally overridden by an author-declared `ChildCoverageOverride.provides` set (`23 §3.2` / `§5`). | **Yes** — `FieldOwnership::NullFill(providers)` records which children DO cover each field; non-providers inferred by set-difference (`23 §5.5`, `16 §7.3.3`). | **Yes** — per-child `Project` emits `Cast(Null, unified_type)` at the seam for every `NullFill` field (`23 §4.3`). The only variant that materializes structural NULL-fill in `PlanNode`s. |
| `Joinset` | **Per-member `CompositionCoverage`** — fold per `24 §8.4`; every member is either `Native` or `NullFill` on each composed-surface Semantics. Most Joinset coverage rows are `Native` on one side and `NullFill` on the other. | **No** — `16 §7.3.3` reserves `FieldOwnership::NullFill` for Unionset; Joinset-side outer-join NULL-fill is carried by `JoinType` semantics at plan time (`24 §8.3` / `§5.5` step 3). | NULL-fill is emitted by the `PlanNode::Join`'s outer-join semantics (`Left` / `Right` / `Full`), not by a typed `Cast(Null, _)` projection. |

Q-24-08 in `questions/open/24_questions.md` revisits whether Joinset should gain structural `NullFill` records for outer joins; Round-1 position is no.

Cross-refs: `15 §6` (Binding-level Coverage), `16 §7.3.3` (per-variant `FieldOwnership::NullFill` policy), `16 §8.4` (`CompositionCoverage` fold), `21 §3.2` (Simple + multi-source), `22 §6` (Grainset), `23 §5` (Unionset — the canonical NULL-fill case), `24 §8.3` / `§8.4` (Joinset).

### 5.1 Coverage-vs-Provenance axis contrast

The two terms both talk about "where a field comes from" but operate at different layers. `25` reads the two axes explicitly rather than assuming readers remember which is which:

| Question | Answered by `Coverage` / `CompositionCoverage` | Answered by `FieldProvenance` |
|---|---|---|
| Does this source / constituent serve this field? | Yes — `CoverageVariant` enumerates `Native` / `Derived` / `NullFill`. | No — provenance is about composition ownership, not source provision. |
| Who owns this field on the composed surface? | No. | Yes — `FieldOwnership::Native(src)` / `Shared(srcs)` / `Derived(expr)` / `NullFill(providers)`. |
| Does this branch contribute a row? | Indirectly — `NullFill` coverage + strategy semantics imply pruning (`23 §4.6`). | No. |
| Does this variant emit `NullFill` rows at plan? | Only Unionset (`23 §4.3`). | Only Unionset (`16 §7.3.3`). Joinset outer-join NULL-fill is `PlanNode::Join` semantics (`24 §8.3`). |

### 5.2 NULL-fill origin routing

Which mechanism produces a runtime NULL depends on the composition kind:

| Source of the runtime NULL | `Simple` | `Grainset` | `Unionset` | `Joinset` |
|---|---|---|---|---|
| Binding-level `CoverageVariant::NullFill` (`15 §6`). | Rejected at `compile` in bare Simple (`21 §3.2`); legal only inside a Unionset branch. | n/a — child Bindings are resolved per Simple's rules. | Legal; drives `FieldOwnership::NullFill` at composition. | n/a. |
| Unionset per-child `Project` with `Cast(Null, unified_type)` (`23 §4.3`). | n/a. | n/a. | Yes — primary emission site. | n/a. |
| Outer-join (`JoinType::Left` / `Right` / `Full`) emission in `PlanNode::Join` (`24 §5.5` step 3). | n/a. | n/a. | n/a (except inside a Joinset child of a Unionset). | Yes — primary emission site. |
| SQL engine's intrinsic NULL handling (expression over NULL, missing row). | Engine concern; not emitted structurally. | Same. | Same. | Same. |

### 5.3 Coverage lift — from `Binding` to `CompositionCoverage`

The lift operation that promotes Binding-level Coverage to CompositionCoverage is cross-referenced from each Complex doc but authored once in `16 §8.4`. The per-variant specialization:

| Variant | Lift input | Lift output | Authoritative ref |
|---|---|---|---|
| `Grainset` | Per-child `Coverage` (each child is either a Simple with `Binding`-level Coverage or a Complex with its own `CompositionCoverage`). | `CompositionCoverage` keyed by `(child-ref, unified-name)`. | `22 §6.1`. |
| `Unionset` | Per-branch `Coverage` + optional `ChildCoverageOverride.provides` (`23 §3.2`). | Same keying shape. | `23 §5`. |
| `Joinset` | Per-member `Coverage`. | Same keying shape; anchor side typically `Native`, target side typically `Native` or `NullFill` per outer-join. | `24 §8.4`. |

No variant re-implements the lift algorithm; all three reuse the `16 §8.4` fold.

---

## 6. Per-Variant Error-Code Bands

`20 §8.1` reserved the `*_E_2000`–`*_E_2599` block for the data-kinds tree, with sub-ranges for each doc. `25` owns `*_E_2500`–`*_E_2599` for genuinely cross-variant diagnostics.

### 6.1 Allocation summary

| Range | Scope | Owning doc | Emission stages |
|---|---|---|---|
| `*_E_2000`–`*_E_2099` | Shared across all DataKind variants | `20 §8.2` | `validate` (`VALID_E_2000`–`2029`), `compile` (`COMP_E_2000`–`2029`), `plan` (`PLAN_E_2040`–`2069`). |
| `*_E_2100`–`*_E_2199` | `Simple` / Dataset | `21 §§7`–`§9` | `VALID_E_2100`–`2199` (`21 §7`); `COMP_E_2100`–`2199` (`21 §8`); `PLAN_E_2100`–`2199` + `PLAN_W_2100`–`2199` (`21 §9`). |
| `*_E_2200`–`*_E_2299` | `Grainset` | `22 §§7`–`§9` | `VALID_E_2200`–`2299` (`22 §7`); `COMP_E_2200`–`2299` (`22 §8`); `PLAN_E_2200`–`2299` + `PLAN_W_2200`–`2299` (`22 §9`). |
| `*_E_2300`–`*_E_2399` | `Unionset` | `23 §§8`–`§10` | `VALID_E_2300`–`2399` (`23 §8`); `COMP_E_2300`–`2399` + `COMP_W_2300`–`2399` (`23 §9`); `PLAN_E_2300`–`2399` + `PLAN_W_2300`–`2399` (`23 §10`). |
| `*_E_2400`–`*_E_2499` | `Joinset` | `24 §§9`–`§11` | `VALID_E_2400`–`2499` (`24 §9`); `COMP_E_2400`–`2499` (`24 §10`); `PLAN_E_2400`–`2499` + `PLAN_W_2400`–`2499` (`24 §11`). |
| `*_E_2500`–`*_E_2599` | Cross-variant diagnostics | **`25` (this doc)** | See `§6.2`. |
| `*_E_2600`–`*_E_2699` | Reserved for future Complex variants per I10 | — | — |

The subsystem prefix (`VALID_E_*` / `COMP_E_*` / `PLAN_E_*`) continues to match the emission stage per `30 §6.1`; `20 §8.5`'s cross-doc-fix to `30 §6.2` (extending `VALID_E` / `COMP_E` / `PLAN_E` subsystem caps to include the `2000`–`2999` data-kinds block) applies to `25`'s `2500`–`2599` sub-range unchanged.

### 6.2 `*_E_2500`–`*_E_2599` reservations

`25`'s block is reserved for diagnostics that span **more than one variant** and cannot be naturally owned by a single per-variant doc. Round 1 reserves the following bands within the `2500`–`2599` window:

| Sub-range | Reserved for | Round-1 status |
|---|---|---|
| `VALID_E_2500`–`2519` | Cross-variant authoring errors (e.g. a Relationship endpoint that is a valid DataKind on its own but illegal in combination with a declared Grainset / Unionset / Joinset under `12 §2`). | Reserved; no codes allocated in Round 1. `12 §2` itself handles same-variant self-nesting. |
| `COMP_E_2500`–`2529` | Cross-variant compile-time errors (e.g. `AsOf` activation across an implicit Relationship composition where the two owning DataKinds are one Simple + one Grainset + one Unionset). | Reserved. |
| `PLAN_E_2530`–`2559` | Cross-variant plan-time errors (e.g. ambiguous field-first resolution across a heterogeneous variant set — partially overlapping with `16 §14.3 PLAN_E_0507 AmbiguousFieldFirstResolution`). | Reserved; no new codes until cross-variant scenarios surface beyond `16 §14.3`'s coverage. |
| `PLAN_W_2560`–`2589` | Cross-variant advisories (e.g. a `TemporalShape × Additivity` advisory that applies across all Measure-bearing variants; see `CDF-17-01` in `§1.3`). | Reserved; Round 1 emits per-variant advisories instead (`21 §9 PLAN_W_2102`, `22 §9.2 PLAN_W_2202`, `23 §10.2`, `24 §11.2 PLAN_W_2404`) — Q2 in `questions/open/25_questions.md` asks whether a unified code is preferable. |

No `*_E_2500`–`*_E_2599` codes are **allocated** in Round 1. The range is reserved against cross-variant diagnostics that surface in `34` / `17` / `30` ratification.

### 6.3 Severity distribution

- `E` (Error) — all reserved `VALID_E_25xx` / `COMP_E_25xx` / `PLAN_E_25xx` bands default to Error.
- `W` (Warning) — `PLAN_W_25xx` advisory band.
- `I` (Info) — reserved; no codes in Round 1.

Per I10 / `30 §6.2`, every severity enum and every allocated `*_E_25xx` variant carries `#[non_exhaustive]`.

### 6.4 Decision aid — picking a range when adding a new diagnostic

A drafter adding a new diagnostic that could plausibly live in multiple ranges should follow this decision discipline (consistent with `20 §8` / `30 §6`):

1. **Single-variant scope?** The diagnostic fires only for one `DataKind` variant. → Use the variant's band (`*_E_21xx` for Simple, `*_E_22xx` for Grainset, `*_E_23xx` for Unionset, `*_E_24xx` for Joinset). Example: `24 §11.2 PLAN_W_2404 AsOfActivation`.
2. **Shared across all four variants, owned by no single per-variant doc?** Use `*_E_2000–2099` (per `20 §8.2`). Example: `VALID_E_2002 InterfaceTypeMismatch`.
3. **Cross-variant, fires for ≥ 2 but < 4 variants, no natural per-variant home?** Use `*_E_2500–2599` (this doc's band). Example: hypothetical `COMP_E_2510 CrossVariantAsOfAmbiguity` for an `AsOf` that crosses a Joinset and an implicit Relationship composition simultaneously.
4. **Same advisory semantics emitted by multiple variants (e.g. shape × additivity)?** Round-1 default is per-variant emission per Q2 in `questions/open/25_questions.md`; retain the per-variant code unless Q2 resolves differently.

The subsystem prefix (`VALID_E_*` / `COMP_E_*` / `PLAN_E_*`) follows the emission stage (`30 §6.1`); per-variant numbering within the band is the drafter's discretion.

---

## 7. Out-of-Scope for `25`

`25` does NOT do any of the following; each is flagged so readers know where to look instead:

- **Override a per-variant rule.** If `25`'s cell appears to disagree with the authoritative section in `10`–`17` or `20`–`24`, the authoritative section wins and the cell needs a doc edit (via `§1.3`). A reader MUST NOT reason from `25`'s cell text alone.
- **Define new Semantics.** New Dimensions / Measures / Metrics / Filters / Keys / Constraints are ratified in `11`; `25` indexes which variant exposes which, but never introduces a new element kind.
- **Introduce new strategies.** New `Strategy` types land in `34` and (in the variant-specific doc) `21`–`24`. `25` summarizes what each strategy does (`§3`) but never ratifies.
- **Introduce new `CompositionKind` variants.** Per `16 §5.3` and `20 §2.1`'s `#[non_exhaustive]` posture, new variants land in `20` / `16` and propagate to `25`'s cells.
- **Adjust error-code allocations outside `25 §6.2`.** Other docs' `*_E_21xx` / `*_E_22xx` / etc. sub-ranges are owned by `21`–`24`; `25` only owns the `*_E_2500`–`*_E_2599` band.
- **Ratify the `PlanNode` roster.** That is `35`'s.
- **Specify `Manifest` persistence or the `ResolvedDataKind` struct roster.** Those are `33`'s.
- **Specify the `Request` shape or the `SessionContext` payload.** Those are `34`'s (plus `00 §4.1`'s vocabulary entries).
- **Resolve `[CROSS-DOC-FIX-NEEDED]` items.** Per the hard constraint, `25` flags contradictions but never resolves them. Each is parked in `§1.3` with the owning-doc pointer; resolution happens in the flagged doc's next revision, possibly with supporting items in `questions/open/25_questions.md`.
- **Define `#[non_exhaustive]` policy.** That is `30 §4`'s. `25` consumes I10 without amending it.
- **Define an `Authoritative-for` schema.** That is `00 §6`'s governance section. `25`'s front-matter follows the conventions already in use; any schema formalization (see Q4 in `questions/open/25_questions.md`) is not `25`'s territory.

### 7.1 What belongs in per-variant docs (not `25`)

A contrasting "where does this belong?" table to help drafters identify when a change belongs in `25` vs an individual variant doc:

| Change type | Belongs in | `25` touched? |
|---|---|---|
| New `TemporalShape` subtype (e.g. `ScdSubtype::Type4`) | `17 §2.2`. | Only the `§2.9` row `17 §2` qualifier. |
| New Semantics element (e.g. `Derivation`) | `11 §6.*`. | `§2.3` rows for the new element. |
| New `DataKind` variant (e.g. `Hierarchyset`) | `20 §2.1` + a new per-variant doc (`26_hierarchyset.md`). | New column across every sub-table. |
| New strategy for an existing variant | `21` / `22` / `23` / `24` §4. | The corresponding `§3` bullets + `§3.5` table. |
| New diagnostic for a single variant | The variant's `*_E_2xxx` band. | None. |
| New diagnostic spanning ≥ 2 variants | `25 §6.2` `*_E_2500–2599`. | Yes — allocate here. |
| New invariant (`I13+`) | `00 §9`. | `§2.10` invariants table. |

---

## 8. Round-1 Open Items

Round-1 drafting surfaced four questions where `25`'s scope boundary interacts with the owning docs in a way that cannot be closed from `10`–`17` or `20`–`24` alone. Each is parked in `questions/open/25_questions.md`:

| ID | Title | Section | Blocking? |
|---|---|---|---|
| Q1 | Matrix as snapshot vs living reference — maintenance discipline | `25 §2` | no |
| Q2 | `PLAN_W_25xx` unified cross-variant advisory band vs per-variant emission | `25 §6.2` | no |
| Q3 | `13 §7` cast-matrix ref chase — retarget vs grow a `13 §7` subsection | `25 §1.3 CDF-23-01`; `25 §2.5` | no |
| Q4 | Auto-generation of `§2`'s matrix from per-doc `authoritative-for:` front-matter | `25 §2` | no |

None block `25 §§1`–`§7`'s ratifications. Blocking status will be revisited as `17` / `30` / `34` land.

### 8.1 Tracking markers and deferrals

Round-1 deferrals recorded at specific cells of `§2` reuse the owning doc's `[TD-*]` tag rather than introducing new `25`-scoped tags. Tags actively referenced by `25`'s cells:

| Tag | Owned by | `25` cell(s) |
|---|---|---|
| `[TD-GRAINSET-NESTED]` | `22 §3.4` | `§2.4` `12 §2` row, `Grainset` as-child cell. |
| `[TD-UNIONSET-SHAPE-PLANNING]` | `23 §6.4` | `§2.9` `17 §5` row, `Unionset` cell. |
| `[TD-COMPOSITION-ASOF]` | `16 §4.4.2` | `§2.8` `16 §4` row; `§2.9` `17 §5` rows; `§3.4` bullet 3; `§3.5` interaction table. |
| `[TD-COMPOSITION-JOINSET-REUSE]` | `16 §13.5` | `§2.8` `16 §11` row, Joinset cell (implicit). |
| `[TD-NESTING-NARY-JOIN]` / `[TD-JOINSET-NARY]` | `12 §5.2` / `24 §13` | `§2.4` `12 §5` row, `Joinset` cell; `§3.4` bullet 1. |

`25` does NOT introduce new `[TD-*]` tags. Future cross-variant deferrals adopted in `§6.2`'s reserved `*_E_2500–2599` sub-ranges will route through the owning-rule doc's tag set per `00 §6.3`.

### 8.2 Ratified at Round 1 — summary

| Q | Section | Decision |
|---|---|---|
| Q-RAT-1 | `§1` | `25` is a cross-reference-only doc; no new normative rules. |
| Q-RAT-2 | `§2` | Matrix organized as `(foundation clause, DataKind variant)` cells with tag `always` / `conditional` / `via Simple children` / `n/a`; eight sub-tables split by foundation doc (`10`, `11`, `12`, `13`, `14`/`14a`/`14b`, `15`, `16`, `17`) plus an invariants index (`§2.10`). |
| Q-RAT-3 | `§3` | Per-variant planner-strategy summary is a 3–5 bullet roster per variant + cross-ref to the owning strategy section; strategy semantics not re-ratified. |
| Q-RAT-4 | `§4` | Per-variant rollup-legality summary: `Simple` rolls directly; `Grainset` rolls by selection; `Unionset` rolls via terminal aggregation; `Joinset` inherits anchor grain. |
| Q-RAT-5 | `§5` | NULL-fill clarification: `FieldOwnership::NullFill` belongs to Unionset alone (`16 §7.3.3`); Joinset NULL-fill is a `JoinType` outer-join concern, not a `FieldOwnership` concern. |
| Q-RAT-6 | `§6.1` | Error-code band allocation: `*_E_2000–2099` shared, `*_E_2100–2199` Simple, `*_E_2200–2299` Grainset, `*_E_2300–2399` Unionset, `*_E_2400–2499` Joinset, `*_E_2500–2599` cross-variant (`25`), `*_E_2600–2699` reserved. |
| Q-RAT-7 | `§6.2` | No `*_E_2500–2599` codes allocated in Round 1; reservations only. |
| Q-RAT-8 | `§7` | Scope boundary: overrides, new Semantics elements, new strategies, new `CompositionKind` variants, and `[CROSS-DOC-FIX-NEEDED]` resolutions all belong to owning docs, not `25`. |

---

## 9. Cross-References

- `00 §4.1, §4.2` — canonical vocabulary for `DataKind`, `SemanticInterface`, `ComposedSemanticInterface`, `Relationship`, `TemporalShape`, `Grain`, `CompositionKind`.
- `00 §6.3` — cross-reference directionality; read-order convention.
- `00 §9` — invariants `I1`–`I12`; `§2.10` is the per-variant applicability index.
- `10 §3.1`–`§3.6` — per-stage pipeline contract; `§2.2` indexes every row.
- `11 §3, §5, §6, §7, §8` — Semantics identity, shape vs resolution variants, element catalog, `Additivity`, `Constraint`; `§2.3` indexes.
- `12 §2, §3, §4, §5, §6` — nesting matrix + per-variant block shapes; `§2.4` indexes.
- `13 §2.4, §3, §4, §5` — `DataType`, `Grain`, `DimensionType`, Keys-vs-Dimensions separation; `§2.5` indexes. `[CROSS-DOC-FIX-NEEDED]` `CDF-23-01` flagged re `13 §7` cast-matrix reference chase (see `§1.3`).
- `14 §§2–6`, `14a`, `14b §§2, §3, §4.5` — expression model, function registry, `ResolvedExprTable`, `PathSignature`; `§2.6` indexes.
- `15 §§2, §3, §5, §6, §7` — `Binding`, `PhysicalSource`, `ColumnMapping`, Binding-level `Coverage`; `§2.7` indexes.
- `16 §§2–14` — `Relationship`, `Cardinality`, `JoinType`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, field-first resolution, Joinset-as-explicit-subset; `§2.8` indexes. `§2.8`'s `16 §7.3.3` row is the canonical anchor for the "Unionset-only `FieldOwnership::NullFill`" rule reproduced in `§5`.
- `17 §§2–8` — `TemporalShape`, `AsOf` JoinType, shape-gated composition; `§2.9` indexes. Forward-refs marked where `17 §*` numbering is still landing.
- `20 §§2–8` — `DataKind` taxonomy, `DataKindOps` trait, `Strategy` trait, strategy-dispatch discipline, error-code roster. `20 §3`'s at-a-glance table is the at-a-glance cross-reference; `25` is the exhaustive one.
- `21 §§2–9` — `Simple` / Dataset; consumed by `§2` Simple columns, `§3.1`, `§4`, `§5`.
- `22 §§2–10` — `Grainset`; consumed by `§2` Grainset columns, `§3.2`, `§4`, `§5`.
- `23 §§2–12` — `Unionset`; consumed by `§2` Unionset columns, `§3.3`, `§4`, `§5`.
- `24 §§2–13` — `Joinset`; consumed by `§2` Joinset columns, `§3.4`, `§4`, `§5`.
- `30 §2, §4, §6` — SemVer discipline, `#[non_exhaustive]` policy, error-code range governance; `§6.1`'s allocation summary cross-refs `20 §8.5`'s cross-doc-fix to `30 §6.2`.
- `33` (pending) — `Manifest` layer: `ResolvedDataKind` / `ResolvedSimpleDataKind` / `ResolvedComplexDataKind` struct rosters; `§2.7`'s Manifest counterpart table will tighten when `33` lands.
- `34` (pending) — planner surface: `Strategy` trait, `PlannerCtx`, `RequestSlice`, `StrategyRegistry`; `§3` bullets consume.
- `35` (pending) — `PlanNode` IR: `PlanNode::Union` / `PlanNode::Join` variants referenced by `§3.3` and `§3.4`.
- `questions/open/25_questions.md` — Round-1 deferred items Q1–Q4 (`§8`).
- `questions/open/17_questions.md` — temporal-shape deferrals that ripple into `§2.9`'s qualifier cells.
- `questions/open/23_questions.md` — Unionset shape-planning deferrals (`[TD-UNIONSET-SHAPE-PLANNING]`).
- `questions/open/24_questions.md` — Joinset N-ary / `AsOf` / reuse items consumed by `§3.4`'s bullets and `§3.5`'s table.

No legacy-doc cross-refs: `25` is new in Round 1 and has no pre-ratification predecessor.
