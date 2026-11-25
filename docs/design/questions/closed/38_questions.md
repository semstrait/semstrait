---
doc: design/questions/closed/38_questions
status: Closed
purpose: Resolved questions originally raised against `apis/38_semstrait_api.md`
---

# Closed Questions — `apis/38_semstrait_api.md`

> Historical record of ratified API decisions. Live items are in [`../open/38_questions.md`](../open/38_questions.md).

---

## Q-API-001 — Dedicated `API_E_*` subsystem prefix for structural configuration errors  *[Closed — superseded by typed-kind transition]*

**Status.** Closed. Subsumed by the workspace-wide retirement of stable string codes (`30 §6` typed-kind discipline). Configuration errors are now intrinsic typed variants `SemStraitErrorKind::{BuilderInvalid, NoRepositoryConfigured}` with no numeric prefix; identification is by variant identity. Recorded here for migration tracking only.

**Original framing (preserved).** `38 §6.2` formerly added two configuration-level `SemStraitError` variants (`BuilderInvalid`, `NoRepositoryConfigured`) that did not correspond to any stage and were assigned the placeholder `COMP_E_0101`. The question of whether to reserve a dedicated `API_E_*` prefix is moot under the typed-kind discipline.

**Resolution.** `38 §6.2` declares both variants directly on `SemStraitErrorKind`; their messages render via `Diagnose::message()`; their severity is `Severity::Error`. No code-table entry; no prefix allocation. The `[TD-API-CODE-TABLE-AMEND]` tech-debt item is retired in the same `30` amendment that retires the stable-code surface.

---

## Q-API-003 — Stage-ownership of escalated warnings under `WarningPolicy`

**Status.** Closed (preserved as Round-1 default). The audit treats this as closed because the doc commits to "preserve variant identity; escalation leaves the outer kind unchanged" — a ratified position with `38 §5.6` carrying the invariant that variant / severity / message remain unchanged by escalation.

**Question.** When `WarningPolicy::FailOnWarning` escalates a compile-stage warning to a fatal, `38 §5.3` re-emits it as the originating stage's fail-fast tuple — the originating-stage `Diagnostic<K>` lands in the `Err` tuple's fatal slot with its kind variant preserved (e.g. `CompileError::SchemaInferenceClamped`). Should the outer wrap be the stage's native kind (current default), or a dedicated `SemStraitErrorKind::ApiEscalated { origin: StageOrigin, kind: SemStraitErrorKind }` variant that clarifies the escalation occurred?

**Refs.**

- `38 §5.3` — current escalation: the kind variant of the underlying `*ErrorKind` is preserved; only the slot in the `Result` tuple changes (warnings → fatal).
- `38 §5.6` — invariant: kind variant / severity / message unchanged by escalation.
- `38 §6.6` — `StageOrigin` enum already distinguishes stage-of-origin via `SemStraitErrorKind::origin()`.

**Arguments for variant-identity preservation (current Round-1 default).**

- Caller code that pattern-matches on (e.g.) `SemStraitErrorKind::Compile(CompileError::SchemaInferenceClamped { .. })` sees no difference between "compile intrinsically errored" and "compile emitted a warning escalated by policy" — both are legitimate reasons for compile to halt from the caller's perspective. The inner `Diagnostic::severity()` is the sole bit that differs.
- Keeps the variant set small; adding `ApiEscalated` is a new discriminator to pattern-match on, and would force every consumer to consider both shapes.
- Aligns with `30 §7`'s per-stage fail-fast rule: the kind variant identifies the failure category regardless of how it became fatal.

**Arguments for `ApiEscalated`.**

- Explicit is better than implicit. Callers that want to distinguish "compile intrinsically errored" from "compile emitted a warning and the policy escalated" gain a structural discriminator.
- Error-reporting UX often wants to say "the policy fired" rather than "compile failed" — an explicit variant makes that phrasing easy.

**Current position in `38`.** Preserve variant identity; escalation leaves the outer kind unchanged. Callers distinguish via `Diagnostic::severity()` (the original was `Warning`, the escalation does not rewrite it).

**Next step.** Revisit if the facade crate (`39`) surfaces a "pretty error-report printer" that benefits from a dedicated escalation discriminator.
