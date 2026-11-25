---
doc: design/questions/deferred/35_questions
status: Deferred (post-v1 IR surface refinements)
purpose: Deferred non-blocking questions moved from `open/35_questions.md`
depends-on:
  - apis/35_semstrait_ir.md
---

# Deferred Questions — `apis/35_semstrait_ir.md`

| ID | Topic | Last known default |
|---|---|---|
| Q-IR-003 | non-equi residual join field reservation | defer field addition |
| Q-IR-004 | aggregate filter field reservation | reserve field; usage deferred |
| Q-IR-005 | `Dialect` sealing posture | keep non-sealed |
| Q-IR-008 | visitor enter/exit trait expansion | keep single-method |
| Q-IR-009 | `EnginePlan::Substrait` boxing strategy | keep boxed variant |
| Q-IR-011 | `SourceRef` rendering/accessor posture | accessors, no `Display` |
| Q-IR-012 | dedicated `Distinct` node variant | keep aggregate lowering |
| Q-IR-013 | split `FetchNode` into limit/offset nodes | keep combined node |
| Q-IR-IMPL-01 | `IntoLiteral for f32` (surface `FloatWidth::Float` via DSL) | not implemented today; parser-side construction only |
| Q-IR-IMPL-02 | `RegistryExtension` re-export at `functions::` module | trait still public via `functions::RegistryExtension`; intent was crate-internal |
| Q-IR-IMPL-03 | `widen_for_join` allocation pressure on Inner joins | clones full `Schema` per side; borrow-comparing variant possible |
| Q-IR-IMPL-04 | `ReturnTypeRule::DecimalScaleZero` resolution for non-Decimal inputs | undefined for Float/Double overloads of ceil/floor |
| Q-IR-IMPL-05 | `closed_five_*` helpers ahead of consumer | `#[allow(dead_code)]`; lifts when Phase B Strategy consumes |
| Q-IR-IMPL-06 | `promote_numeric` single-call-site | one caller (`closed_five_return_type`); inline candidate |
| Q-IR-SPEC-01 | spec `35 §3.4` / `§11.7` carries `AggregationOp` on `Aggregate` variant + `input_expr` singular | code now uses `AggregateKind` + `args: Vec<PhysicalExpr>` |
| Q-IR-SPEC-02 | spec `35 §3.4` carries `Literal::{Integer(i64), Float(f64), Null}` tuple-form | code now uses struct-form variants with `IntegerWidth` / `FloatWidth` |
| Q-IR-SPEC-03 | spec `14a §3.4` / `§3.5` does not declare `DecimalScaleZero` / `DecimalFamily` | code adds both for ceil/floor/round/median Decimal overloads |
| Q-IR-SPEC-04 | spec `14a §3.1` line 65 declares `pub description: &'static str` on `FunctionSpec` | code removed the field |
| Q-IR-SPEC-05 | accessor enum dedup (`MetricAccessor` vs `MeasureAccessor`, `KeyAccessor` vs `DimensionAccessor`) | keep explicit mirrored enums for now |

Re-open when optimizer/adapter implementation needs any of these shape choices to ship behavior.

> **OQ-D** (`expr_fn` DSL crate home) was **closed** in the item-Y ratification session (2026-06-15) — ratified as option (a), wholly in `semstrait-model`. See [`../closed/35_questions.md`](../closed/35_questions.md).

---

## Q-IR-IMPL-01 — `IntoLiteral for f32`

**Source:** post-implementation review (2026-05-26), W1-N-2.

**Finding.** `crates/semstrait-ir/src/expr/expr_fn.rs:35-60` provides
`IntoLiteral` impls for `i32` (→ `IntegerWidth::Int`), `i64`
(→ `IntegerWidth::Long`), and `f64` (→ `FloatWidth::Double`). There is no
`IntoLiteral for f32`. `FloatWidth::Float` is reachable only via direct
`Literal::Float { width: FloatWidth::Float, value }` construction.

**Decision needed.** Add `IntoLiteral for f32` (surfaces `Float` width
through the DSL) or keep the variant parser-side only.

**Default.** Defer — Rust authors typically write `f64`. The variant remains
useful for IR-level round-trip from parsed DSL where author wrote a 32-bit
constant.

---

## Q-IR-IMPL-02 — `RegistryExtension` re-export at `functions::` module

**Source:** post-implementation review (2026-05-26), W2-N-1.

**Finding.** P-7 removed `RegistryExtension` from the crate-root re-export
(`lib.rs`), but `crates/semstrait-ir/src/functions/mod.rs:28` still has
`pub use extension::RegistryExtension;`, leaving the trait reachable as
`semstrait_ir::functions::RegistryExtension`.

**Decision needed.** `pub(crate) use` to fully hide until wired, or
remove the re-export and let consumers import via the explicit
`extension::RegistryExtension` path when the trait wires up.

**Default.** `pub(crate) use` — completes the P-7 intent without breaking
internal use.

---

## Q-IR-IMPL-03 — `widen_for_join` allocation pressure on Inner joins

**Source:** post-implementation review (2026-05-26), W2-N-2.

**Finding.** `crates/semstrait-ir/src/plan/validate.rs:194-207` —
`widen_for_join` allocates a fresh `Schema` (Vec<SchemaColumn>) for each
side per Join validation, then `.extend()`s into a third Vec. For Inner
joins this is two unnecessary clones (the helper short-circuits to
`schema.clone()` and we then equality-check the cloned Vec).

**Decision needed.** Replace with a borrow-comparing variant
(`expected_nullable_at(i, side, jt) -> bool`) checked positionally
against `j.meta.output_schema.columns[i].nullable`, or accept the clone
cost.

**Default.** Defer — current scale of plan trees doesn't warrant the
optimization; revisit if profiling flags it.

---

## Q-IR-IMPL-04 — `ReturnTypeRule::DecimalScaleZero` for non-Decimal inputs

**Source:** post-implementation review (2026-05-26), W3-N-1.

**Finding.** `crates/semstrait-ir/src/functions/builtins/math.rs:31-44` —
ceil/floor declare three signature overloads (Float, Double, Decimal).
`DecimalScaleZero`'s rustdoc only documents `Decimal(p, s) → Decimal(p, 0)`.
Resolution semantics for Float/Double inputs are implicit ("same as first
arg"). Spec `14a §4.3` line 194-195 only lists `(Float) -> Float,
(Decimal(p,s)) -> Decimal(p,0)` — the catalog adds a Double overload not
in the spec.

**Decision needed.** (a) Document `DecimalScaleZero` as "Decimal(p, s) →
Decimal(p, 0); other inputs unchanged"; (b) Restrict the rule to Decimal
overloads and use `SameAsFirstArg` for Float/Double; (c) Remove the Double
overload to match `14a §4.3`.

**Default.** (a) — easiest doc-only fix; preserves overload roster.

---

## Q-IR-IMPL-05 — `closed_five_*` helpers ahead of consumer

**Source:** post-implementation review (2026-05-26), XC-1.

**Finding.** `crates/semstrait-ir/src/functions/builtins/mod.rs:32-61` —
`closed_five_additivity` and `closed_five_return_type` are `pub(crate)`
with `#[allow(dead_code)]`. Zero production call sites; tests cover the
contract. Reserved for Phase B Strategy consumption.

**Decision needed.** Accept the dead-code allow as documented Phase-B
prep, or gate behind a feature until Phase B lands.

**Default.** Accept — Phase B is the next-landing phase; helpers retire
the allow immediately on first non-test caller.

---

## Q-IR-IMPL-06 — `promote_numeric` single-call-site

**Source:** post-implementation review (2026-05-26), XC-2.

**Finding.** `crates/semstrait-ir/src/functions/builtins/mod.rs:64-74` —
`promote_numeric` has exactly one call site (`closed_five_return_type`).

**Decision needed.** Inline into the Sum/Avg arm, or keep as a named
helper.

**Default.** Keep — name documents the promotion semantic; inlining adds
match noise to a 5-arm table that should stay scannable.

---

## Q-IR-SPEC-05 — Accessor enum dedup posture

**Source:** manifest/graph post-ratification verification pass (2026-05-28).

**Finding.** `35 §6.1` currently carries four accessor enums where two pairs are structurally mirrored:

- `MetricAccessor` mirrors `MeasureAccessor`
- `KeyAccessor` mirrors `DimensionAccessor`

The mirrored shape is intentional for explicit API readability, but it is a DRY trade-off.

**Decision needed.** Keep four explicit enums, or collapse mirrored pairs into aliases/shared enums.

**Default.** Keep explicit mirrored enums for now. They preserve kind-local readability and avoid a wider cascade into `14` docs and DSL examples. Revisit when expression-DSL simplification becomes an active stream.

---

## OQ-D — `expr_fn` authoring-DSL crate home — **CLOSED (option (a)), see [`../closed/35_questions.md`](../closed/35_questions.md)**

Moved to the closed sidecar on ratification (2026-06-15). Retained here as a forwarding stub for discoverability.

