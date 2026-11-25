---
id: 25-error-bands
status: ratified
version: 2.1
impl: none
---

# Applicability — Error-Code Bands

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

The subsystem prefix (`VALID_E_*` / `COMP_E_*` / `PLAN_E_*`) continues to match the emission stage per `30 §6.1`; `20 §8.1`'s `CDF-30-01` cross-doc-fix to `30 §6.2` (extending `VALID_E` / `COMP_E` / `PLAN_E` subsystem caps to include the `2000`–`2999` data-kinds block) applies to `25`'s `2500`–`2599` sub-range unchanged.

### 6.2 `*_E_2500`–`*_E_2599` reservations

`25`'s block is reserved for diagnostics that span **more than one variant** and cannot be naturally owned by a single per-variant doc. Round 1 reserves the following bands within the `2500`–`2599` window:

| Sub-range | Reserved for | Round-1 status |
|---|---|---|
| `VALID_E_2500`–`2519` | Cross-variant authoring errors (e.g. a Relationship endpoint that is a valid DataKind on its own but illegal in combination with a declared Grainset / Unionset / Joinset under `12 §2`). | Reserved; no codes allocated in Round 1. `12 §2` itself handles same-variant self-nesting. |
| `COMP_E_2500`–`2529` | Cross-variant compile-time errors (e.g. `AsOf` activation across an implicit Relationship composition where the two owning DataKinds are one Simple + one Grainset + one Unionset). | Reserved. |
| `PLAN_E_2530`–`2559` | Cross-variant plan-time errors (e.g. ambiguous field-first resolution across a heterogeneous variant set — partially overlapping with `16 §14.3 PLAN_E_0507 AmbiguousFieldFirstResolution`). | Reserved; no new codes until cross-variant scenarios surface beyond `16 §14.3`'s coverage. |
| `PLAN_W_2560`–`2589` | Cross-variant advisories (e.g. a `TemporalShape × Additivity` advisory that applies across all Measure-bearing variants; see `CDF-17-01` in `§1.3`). | Reserved; Round 1 emits per-variant advisories instead (`21 §9 PLAN_W_2102`, `22 §9 PLAN_W_2201` / `PLAN_W_2202`, `23 §10.2`, `24 §11.2 PLAN_W_2404`) — Q2 in `questions/open/25_questions.md` asks whether a unified code is preferable. |

No `*_E_2500`–`*_E_2599` codes are **allocated** in Round 1. The range is reserved against cross-variant diagnostics that surface in `34` / `17` / `30` ratification.

### 6.3 Severity distribution

- `E` (Error) — all reserved `VALID_E_25xx` / `COMP_E_25xx` / `PLAN_E_25xx` bands default to Error.
- `W` (Warning) — `PLAN_W_25xx` advisory band.
- `I` (Info) — reserved; no codes in Round 1.

Per I10 / `30 §6.2`, every severity enum and every allocated `*_E_25xx` variant carries `#[non_exhaustive]`.

## 7. Out-of-Scope for `25`

`25` does NOT do any of the following; each is flagged so readers know where to look instead:

- **Override a per-variant rule.** If `25`'s cell appears to disagree with the authoritative section in `10`–`17` or `20`–`24`, the authoritative section wins and the cell needs a doc edit (via `§1.3`). A reader MUST NOT reason from `25`'s cell text alone.
- **Define new Semantics.** New Dimensions / Measures / Metrics / Filters / Keys / Constraints are ratified in `11`; `25` indexes which variant exposes which, but never introduces a new element kind.
- **Introduce new strategies.** New `Strategy` types land in `34` and (in the variant-specific doc) `21`–`24`. `25` summarizes what each strategy does (`§3`) but never ratifies.
- **Introduce new `CompositionKind` variants.** Per `16 §5.3` and `20 §2.1`'s `#[non_exhaustive]` posture, new variants land in `20` / `16` and propagate to `25`'s cells.
- **Adjust error-code allocations outside `25 §6.2`.** Other docs' `*_E_21xx` / `*_E_22xx` / etc. sub-ranges are owned by `21`–`24`; `25` only owns the `*_E_2500`–`*_E_2599` band.
- **Ratify the `PlanNode` roster.** That is `35`'s.
- **Specify `SemanticManifest` persistence or the `ResolvedDataKind` struct roster.** Those are `33`'s.
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

## 9. Cross-References

- `00 §4.1, §4.2` — canonical vocabulary for `DataKind`, `SemanticInterface`, `ComposedSemanticInterface`, `Relationship`, `TemporalShape`, `Grain`, `CompositionKind`.
- `00 §6.3` — cross-reference directionality; read-order convention.
- `00 §9` — invariants `I1`–`I12`; `§2.10` is the per-variant applicability index.
- `10 §3.1`–`§3.6` — per-stage pipeline contract; `§2.2` indexes every row.
- `11 §3, §5, §6, §7, §8` — Semantics identity, shape vs resolution variants, element catalog, `Additivity`, `Constraint`; `§2.3` indexes.
- `12 §2, §3, §4, §5, §6` — nesting matrix + per-variant block shapes; `§2.4` indexes.
- `13 §2.4, §3, §4, §5` — `DataType`, `Grain`, `DimensionType`, Keys-vs-Dimensions separation; `§2.5` indexes. `[CROSS-DOC-FIX-NEEDED]` `CDF-23-01` flagged re `13 §7` cast-matrix reference chase (see `§1.3`).
- `14 §§2–6`, `14a`, `19 §§3.2, §3.3, §3.4.5` — expression model, function registry, `ResolvedExprTable`, `PathSignature`; `§2.6` indexes.
- `15 §§2, §3, §5, §6, §7` — `Binding`, `PhysicalSource`, `ColumnMapping`, Binding-level `Coverage`; `§2.7` indexes.
- `16 §§2–14` — `Relationship`, `Cardinality`, `JoinType`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, `CompositionCoverage`, field-first resolution, Joinset-as-explicit-subset; `§2.8` indexes. `§2.8`'s `16 §7.3.3` row is the canonical anchor for the "Unionset-only `FieldOwnership::NullFill`" rule reproduced in `§5`.
- `17 §§2–8` — `TemporalShape`, `AsOf` JoinType, shape-gated composition; `§2.9` indexes. Forward-refs marked where `17 §*` numbering is still landing.
- `20 §§2–8` — `DataKind` taxonomy (sealed trait hierarchy on two orthogonal axes), `Strategy` trait, strategy-dispatch discipline, error-code roster. `20 §3`'s at-a-glance table is the cross-reference; `25` is the exhaustive one.
- `21 §§2–9` — `Simple` / Dataset; consumed by `§2` Simple columns, `§3.1`, `§4`, `§5`.
- `22 §§2–10` — `Grainset`; consumed by `§2` Grainset columns, `§3.2`, `§4`, `§5`.
- `23 §§2–12` — `Unionset`; consumed by `§2` Unionset columns, `§3.3`, `§4`, `§5`.
- `24 §§2–13` — `Joinset`; consumed by `§2` Joinset columns, `§3.4`, `§4`, `§5`.
- `30 §2, §4, §6` — SemVer discipline, `#[non_exhaustive]` policy, error-code range governance; `§6.1`'s allocation summary cross-refs `20 §8.1`'s `CDF-30-01` cross-doc-fix to `30 §6.2`.
- `33` (pending) — `SemanticManifest` layer: `ResolvedDataKind` / `ResolvedSimpleDataKind` / `ResolvedComplexDataKind` struct rosters; `§2.7`'s SemanticManifest counterpart table will tighten when `33` lands.
- `34` (pending) — planner surface: `Strategy` trait, `PlannerCtx`, `RequestSlice`, `StrategyRegistry`; `§3` bullets consume.
- `35` (pending) — `PlanNode` IR: `PlanNode::Union` / `PlanNode::Join` variants referenced by `§3.3` and `§3.4`.
- `questions/open/25_questions.md` — Round-1 deferred items Q1–Q4 (`§8`).
- `questions/open/17_questions.md` — temporal-shape deferrals that ripple into `§2.9`'s qualifier cells.
- `questions/open/23_questions.md` — Unionset shape-planning deferrals (`[TD-UNIONSET-SHAPE-PLANNING]`).
- `questions/deferred/24_questions.md` — Joinset N-ary deferral (Q-24-01) consumed by `§3.4`'s bullets; `questions/closed/24_questions.md` — `AsOf` activation matrix (Q-24-04), reuse policy (Q-24-05) consumed by `§3.5`'s table.

No legacy-doc cross-refs: `25` is new in Round 1 and has no pre-ratification predecessor.

## Testing Strategy

- Allocation-summary completeness (4 variants + cross-variant band); no overlapping code ranges; reserved sub-range documentation per 30 §6.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
