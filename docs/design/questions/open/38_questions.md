---
doc: design/questions/open/38_questions
status: Living
purpose: Parked unresolved questions surfaced while drafting `apis/38_semstrait_api.md`
depends-on:
  - apis/38_semstrait_api.md
  - apis/30_api_contracts.md
  - apis/31_semstrait_common.md
  - apis/32_semstrait_model.md
  - apis/33_semstrait_manifest.md
  - apis/34_semstrait_planner.md
  - apis/36_semstrait_adapter.md
  - apis/37_semstrait_catalog.md
---

# Open Questions — `apis/38_semstrait_api.md`

> Ten questions remain open: Q-API-002, -004 through -012 (excluding -001 / -003 closed). Closed items moved to [`../closed/38_questions.md`](../closed/38_questions.md). Each entry restates the question, lists its ratified references, and records the Round-1 default `38` currently uses. None of these items block the headline ratifications in `38 §15`.

---

## Q-API-001 — Dedicated `API_E_*` subsystem prefix — CLOSED

> **Moved to [`../closed/38_questions.md`](../closed/38_questions.md).** Superseded by the typed-kind discipline.

---

## Q-API-002 — `PipelineOutcome` struct vs fail-fast tuple return

**Question.** Should `38 §7.1`'s fused helper return a dedicated `PipelineOutcome` struct (reserved `Provisional` at `§2.1`) instead of the fail-fast tuple `Result<(EngineArtifact, Diagnostics<K>), (Diagnostic<K>, Diagnostics<K>)>`?

**Why.** A struct admits additive fields (per-stage timings, ids) and named access; the tuple is idiomatic and avoids two-field API bloat (`38 §2.1`, `§7.1`; `30 §4.2`, `§7`).

**Current position in `38`.** Tuple return; `PipelineOutcome` reserved but not used.

**Next step.** Revisit when an instrumentation surface lands and callers want timings as data, not `tracing` events.

---

## Q-API-003 — Stage-ownership of escalated warnings under `WarningPolicy` — CLOSED

> **Moved to [`../closed/38_questions.md`](../closed/38_questions.md).** Round-1 default ratified — preserve variant identity through escalation.

---

## Q-API-004 — Diagnostic de-duplication policy across stages

**Question.** Should `semstrait-api` fold cross-stage duplicate diagnostics (same root cause reported at `validate` then `compile` over one `Span`) under a `(kind variant + Span)` predicate, rather than `38 §10.3`'s current no-dedup policy?

**Why.** Dedup cleans caller UX but needs a ratified cross-kind equivalence map and a keep-which-copy rule, neither of which exists; cross-stage duplication is also informative (`38 §10.3`; `30 §5`; `31 §3`).

**Current position in `38`.** No de-duplication in v1; callers apply their own predicate on the returned `Diagnostics<K>`.

**Next step.** If production UX shows demand, define the equivalence class + keep-which-copy rule in a `30` amendment, then implement here.

---

## Q-API-005 — Incremental-compile surface

**Question.** Should a Round-2 MINOR add `SemStrait::recompile_with_changes(&manifest, ModelDiff)` that re-compiles only affected bindings, rather than `38 §8.4`'s full re-compile on model mutation?

**Why.** Faster dev/multi-tenant loops, but it complicates the byte-determinism guarantee and the binding-dependency invalidation analysis is its own project (`33 §7.4`, `§13`; `38 §8.4`).

**Current position in `38`.** Not in v1; callers hold a manifest cache keyed on model hash.

**Next step.** Revisit after v1 if compile-stage latency dominates; prerequisite is a `ModelDiff` type owned by `32`.

---

## Q-API-006 — Per-`SemStrait`-handle `FunctionRegistry`

**Question.** Should `SemStraitBuilder::with_function_registry` accept a caller-built per-process `&'static FunctionRegistry`, or stay pinned to the process-global `semstrait_common::function_registry()` (`38 §4.5`, `§1.3`)?

**Why.** Per-handle registries aid test isolation and multi-tenancy, but `CanonicalFn` identity is process-global, so divergent rosters would mint IDs that don't compose (violating I3); whether isolation is feasible has no tracked question yet (`38 §4.5`).

**Current position in `38`.** The `function_registry` field accepts any `&'static FunctionRegistry`, but v1 ships only `function_registry()`; the field shape accommodates future isolation without API change.

**Next step.** If `31` adopts per-process registry isolation, `38` picks it up with no signature change.

---

## Q-API-007 — Sync wrappers for non-async callers

**Question.** Should `semstrait-api` ship `compile_from_yaml_blocking(&self, yaml)` wrapping `block_on(...)` for non-async callers (CLI, sync tests, WASM)?

**Why.** A `_blocking` idiom aids ergonomics but pulls a runtime dependency into the executor-agnostic crate and adds a feature-matrix flag; `39_semstrait_facade.md` is a more natural home (`30 §9`, `§10.5`; `38 §9.5`).

**Current position in `38`.** Not in v1; callers bridge via their own runtime's `block_on`.

**Next step.** Consider for `39_semstrait_facade.md`, where "one-shot-no-config" pairs with "one-shot-no-async".

---

## Q-API-008 — `SemStrait::builder` as `const fn`

**Question.** Should `SemStrait::builder` be refactored (lazy allocation at `build`) to become `const fn`, given `38 §11.4` blocks it on non-const `Vec::new()` / `Arc<dyn ...>` construction?

**Why.** It would enable `static` builder declarations and unify const discipline, but the gain is small and callers needing const already use `SemStraitBuilder::new()` (already `const fn`) (`38 §4.7`; `30 §8.1`).

**Current position in `38`.** `SemStrait::builder` is not `const`; `SemStraitBuilder::new` is.

**Next step.** Revisit only if callers show widespread use of `static` builder declarations.

---

## Q-API-009 — Batch `compile_and_plan_and_adapt(Vec<Request>)`

**Question.** Should `38 §7.1`'s fused helper accept a batch of requests against one manifest?

**Why.** Saves a trivial loop and opens parallelism, but batch-error semantics (fail-first / accumulate / partial) are each a separate API decision, and `§8.1`'s compile-once/plan-many already expresses this (`38 §7.1`, `§8.1`).

**Current position in `38`.** Not in v1; callers loop.

**Next step.** Revisit if observability shows many callers writing identical batch loops.

---

## Q-API-010 — `validate_manifest` aggregation policy

**Question.** Should `38 §3.4` surface per-source drift as `Vec<DriftReport>` instead of one max-severity-aggregated `DriftReport`?

**Why.** A vector eases tracing which source drifted, but callers typically act on the aggregate and per-source details already survive in `DriftReport.details: Vec<DriftKind>` (`37 §9.2`, `§9.4`; `38 §3.4`).

**Current position in `38`.** Max-severity aggregation; per-source details concatenated into the `DriftKind` vector.

**Next step.** Revisit if UX shows callers consistently wanting to trace which source contributed which drift.

---

## Q-API-011 — Direct `emit` on `SemStrait` for SQL-emission convenience

**Question.** Should `SemStrait::emit(&self, adapter, &plan, &manifest)` exist as a passthrough to `EngineAdapter::emit` (`36 §3`), mirroring `adapt`'s discipline?

**Why.** It gives ergonomic parity with `adapt`, but SQL is an adapter concern, `emit` is undefined for non-SQL (protobuf) adapters, and the wrapper is three lines (`36 §3.1`; `38 §1.3`, `§3.4`).

**Current position in `38`.** No direct `emit`; callers reach it through the adapter.

**Next step.** Revisit if `39_semstrait_facade.md` adds an SQL-emission convenience; `38` would follow.

---

## Q-API-012 — Wrapping primitive for lifting `Diagnostic<K1>` into `Diagnostic<K2>`

**Question.** What primitive lifts `Diagnostic<K1>` / `Diagnostics<K1>` into the unified `SemStraitErrorKind` envelope (given `§6.3`'s `From<K1> for K2` impls): a blanket `From` on `31`'s `Diagnostic` (A), an explicit `cast_kind::<K2>()` method (B), or per-element caller rewrap (C)?

**Why.** A composes with `?`/`.into()` automatically but touches every `Diagnostic<*>` with no opt-out; B is discoverable and coherence-safe but more verbose; the choice governs `31`'s diagnostic surface and the §7 fused-helper body (`31 §3`; `38 §6.3`, `§7.2`).

**Current position in `38`.** Forward-references whichever shape lands; §7.2 shows the lift site abstractly. Structural default is A; B is the conservative fallback if `31`'s blanket-impl posture is contested.

**Next step.** Pick A or B during the next `31` revision and update §7.2 + other lift-site prose to match.
