---
doc: design/questions/open/25_questions
status: Living
purpose: Parked unresolved questions surfaced while drafting `data-kinds/25_applicability_matrix.md`
depends-on:
  - data-kinds/25_applicability_matrix.md
  - data-kinds/20_taxonomy.md
  - data-kinds/21_dataset.md
  - data-kinds/22_grainset.md
  - data-kinds/23_unionset.md
  - data-kinds/24_joinset.md
  - foundations/13_types_and_grain.md
  - foundations/14_expressions.md
  - foundations/14a_function_catalog.md
  - foundations/19_expression_flow.md
  - foundations/16_composition.md
  - foundations/17_temporal_shape.md
  - apis/30_api_contracts.md
  - apis/34_semstrait_planner.md
---

# 25 — Open Questions

> Items surfaced during Round-1 drafting of the cross-cutting applicability matrix. Each entry restates the question, lists its ratified references, and records the Round-1 default `25` currently uses. Entries migrate out of this file as `17`, `30`, `33`, `34`, or a revision of the per-variant docs (`21`–`24`) make decisions that either confirm or amend `25`'s indexing. None of these open items block `25`'s Round-1 matrix (`§§1`–`§7`).

---

## Q1. Matrix as snapshot vs living reference — maintenance discipline

How does `25 §2`'s matrix stay consistent with its eight indexed docs (`10`–`17`, `20`–`24`) as they take MINORs?

**Why.** A new variant/element/subtype in any indexed doc adds rows/cells to `25 §2`.

**Default.** Hybrid — exhaustive re-sync at each MAJOR, lossy between MINORs.

**Reopen.** `00 §6` doc-governance grows a `25`-sync paragraph, or the first MAJOR re-sync.

---

## Q2. `PLAN_W_25xx` unified cross-variant advisory band vs per-variant emission

Should the cross-variant `Additivity × TemporalShape` advisory be one unified `25 §6.2` code or stay four per-variant codes (`PLAN_W_2102`/`2202`/`COMP_W_2302`–`2306`/`PLAN_W_2404`)?

**Why.** `17 §7.3`'s advisory roster spans every Measure-carrying variant.

**Default.** Per-variant emission; `PLAN_W_2560`–`2589` reserved (no unified/aggregate code).

**Reopen.** `17 §7.3` Round-2, or the revision of `30 §6.2` that touches data-kinds-block code allocations.

---

## Q3. `13 §7` cast-matrix reference chase — retarget vs grow a `13 §7` subsection

`23 §1.1`/`23 §4.4` cite a `13 §7` "cast matrix" that does not exist (the rules live in `13 §2.4` + `14a`) — retarget `23`'s refs or grow a `13 §7` Cast Matrix section?

**Why.** Cross-child type reconciliation in `Unionset` needs an authoritative widening-rule cite.

**Default.** Retarget `23`'s refs to `13 §2.4` + `14a` (marker `CDF-23-01`); no structural change to `13`.

**Reopen.** `23`'s Round-2 (retarget) or `13`'s Round-2 (grow `§7`).

---

## Q4. Auto-generation of `§2`'s matrix from per-doc `authoritative-for:` front-matter

Should `25 §2` auto-generate from a machine-readable `authoritative-for:` schema, be hand-written per-doc, or stay hand-written centrally?

**Why.** Every cell maps to some indexed doc's `authoritative-for:` entry; auto-gen would prevent drift.

**Default.** Hand-written centrally, maintained per Q1's cadence.

**Reopen.** `00 §6` doc-governance, or the first MAJOR where `25 §2` drift forces tooling.

---

## Summary

| ID | Title | Round-1 default | Tracking marker |
|---|---|---|---|
| Q1 | Matrix as snapshot vs living reference | Hybrid (C) — MAJOR-synced snapshot, lossy at MINOR | — |
| Q2 | `PLAN_W_25xx` unified cross-variant advisory band | Per-variant emission (A); `PLAN_W_2560`–`2589` reserved | — |
| Q3 | `13 §7` cast-matrix reference chase | Retarget `23` refs to `13 §2.4` + `14a` (A) | `CDF-23-01` |
| Q4 | Auto-generation of `§2` from `authoritative-for:` | Hand-written centrally (C) | — |

---

**End of `25_questions.md`.**
