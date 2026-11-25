---
doc: design/questions/closed/39_questions
status: Closed
purpose: Resolved questions originally raised against `apis/39_semstrait_facade.md`
---

# Closed Questions — `apis/39_semstrait_facade.md`

> Historical record of ratified facade decisions. Live items are in [`../open/39_questions.md`](../open/39_questions.md).

---

## Q-FAC-003 — `semstrait::run` error type  *[Closed — superseded by typed-kind transition]*

**Status.** Closed. The workspace-wide typed-kind transition (`30 §5` / `31 §3` / `38 §6`) resolves the prior ambiguity: `semstrait::run` returns the same fail-fast tuple shape as `38 §7.1`'s `compile_and_plan_and_adapt`, namely `Result<(EngineArtifact, Diagnostics<SemStraitErrorKind>), (Diagnostic<SemStraitErrorKind>, Diagnostics<SemStraitErrorKind>)>`. This composes `30 §7.2`'s tuple-return rule (warnings ride alongside the fatal in both arms) with the unified `SemStraitErrorKind` ratified at `38 §6`. Recorded for migration tracking.

**Original framing (preserved).** `39 §4.1` formerly signed `run` as `Result<EngineArtifact, SemStraitError>` with the rationale that one unified error type spanned all five stages. The retired `SemStraitError` enum and the `IntoDiagnostic` trait are gone (`30 §5` / `31 §3`); the unified shape now is `Diagnostic<SemStraitErrorKind>` plus `Diagnostics<SemStraitErrorKind>` in the fail-fast tuple. The "Diagnostic singular vs Vec<Diagnostic>" tension dissolved when `30 §7.2` ratified the `(fatal, warnings)` tuple as the canonical fail-fast surface.

**Resolution.** `39 §4.1` declares the tuple shape directly; the rationale is the same as `38 §7.1`'s.
