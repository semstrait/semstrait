---
number: 44
title: 'There is no gate roster: a validity definition belongs to its stage, and the `CompileError` variant set is the only enumeration'
status: accepted
date: 2026-08-14
source:
- crates/semstrait-manifest/src/compile/driver.rs
- crates/semstrait-manifest/src/compile/definitions/graph.rs
- crates/semstrait-manifest/src/compile/definitions/mod.rs
- crates/semstrait-manifest/src/compile/seal.rs
- crates/semstrait-manifest/src/error.rs
- crates/semstrait-common/src/diagnostic.rs
- decided for the codes in crates/
links:
- target: 54
  kind: amendedby
- target: 65
  kind: amendedby
- target: 68
  kind: amendedby
---

# There is no gate roster: a validity definition belongs to its stage, and the `CompileError` variant set is the only enumeration

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): `MAX_COMPOSITION_DEPTH = 4` bounds authored nesting and is model validate's rule, not the enumerating stage's.
> - [ADR 0065](0065-a-semantics-has-one-global-definition-and-one-signature-per-semantic-source.md): a definition owns no expression and a signature replaces the override row; nothing is interned, `depends_on` is removed, and the orphan sweep closes over child ids.
> - [ADR 0068](0068-adapters-are-layered-format-first.md): its aside "the only `AsOf` in the workspace is `Capability::AsOfJoin`" loses its subject: `Capability::AsOfJoin` is deleted with its type, and no `AsOf` capability or join type remains (Substrait's `JoinType` has none either); its ruling — no AsOf authoring form; it returns with temporal-join authoring — stands.

## Context and Problem Statement

Requirement 5 asks for gates "formalized and frozen". Twenty hard rejections exist and five carry the numbers `G1`–`G5`, a roster that four earlier documents treated as ratified. The request framed G5 as a conflict between two sources to adjudicate, offered enforcing G4 by giving `CompileError::UnresolvedColumn` construction sites, asked which code namespace gates use, and inherited three obligations: `COMP_E_2414`, the reserved G6 and the composition-depth bound. This decision is recorded because it retires a roster that four earlier documents treated as ratified, deletes a public error variant, resolves a conflict the request framed as an adjudication into a mis-bundling, and reads backwards on first contact: the crate satisfies "gates formalized and frozen" precisely by refusing to keep a list of its gates.

## Considered Options

- No gate roster; each stage owns its validity definition (chosen).
- **Freeze the roster at exactly `G1`–`G5`.** The literal reading of requirement 5, and the cheapest. Refused because membership has no principle behind it: fifteen sibling hard rejections are excluded by nothing but the provenance of a numbering whose ratification traces to a file that never existed.
- **Keep `G1`–`G5` and mint `G6`…`G20` for the unnumbered rules.** Honest about closure, and it would make the roster total. Refused because it stands up a second registry beside the one that is already the semver contract, requires a renumbering decision every time a rule lands, and `G6` is additionally unavailable — four earlier documents reserve it for relationship-graph cycle detection.
- **Rule G5 for Relationships and let the expression sweep go.** The reading that takes the deferred question as the later and therefore governing source. Refused because the two rules are converse edge directions, so choosing between them discards a real rule for no reason.
- **Rule G5 for Relationships on the ground that the base schema makes relationships reachable by construction.** Considered and **refused as a false positive** during the session that produced this ADR: the relationship rule is a dangling-endpoint check, not an unreachability check, so reachability was never its subject and dangling `from` / `to` pointers survive the base schema intact.
- **A named gate list in one place, as a document or a Rust enum.** What "frozen" most naturally suggests. Refused as a second place to look with nothing keeping it in step with the stages — the precise failure mode measured when 50 ratified leaves were found describing types their own phase had deleted.
- **Add `code()` to `Diagnose` and publish `COMP_E_*` as live codes.** Would give gates a real namespace and make the retired codes meaningful. Refused as a cross-workspace diagnostics decision that creates a second identity for every fault beside the variant identity ADR 0004 publishes.
- **Enforce G4 by giving `UnresolvedColumn` construction sites.** The request's other option. Refused because the variant's payload names a retired entity, so enforcing it first requires re-minting it — at which point deletion and re-mint under the correct subject is the same work without preserving a dead reference.
- **Block the freeze until the authored-body path is defined.** Defensible, since two of five gates are starved without it. Refused because a stage's validity definition does not depend on its input being populated today; conflating the two would make every definition wait on the last piece of wiring.

## Decision Outcome

The five-gate roster `G1`–`G5` is retired as an identifier scheme. Its five rules survive, each re-homed onto the stage that owns it, and the roster is declared **derived and open by construction**: it is exactly the union of the stages' validity definitions, so it grows when a stage gains a rule and is never "extended by minting". Nothing is frozen in a list. What is frozen is a rule about *where a definition lives* — each stage states its own validity definition in its own module documentation, each rule has a test, and the only enumeration is the one that already exists and is already authoritative: the `CompileError` variant set, plus, per variant, the stage that raises it. This is ADR 0042 carried to its conclusion — if validity is constitutive of a stage, then a roster maintained beside the stages is a second place to look, and it can drift from them.

### The roster of five was never a roster

**Twenty hard rejections exist and five carry a number.** `CompileError` declares 23 variants: two are `Severity::Warning` (`NarrowingCast`, `CatalogOverride` — the catch-all arm at `error.rs` returns `Severity::Error` for everything else), one is the `Ir` passthrough, and the remaining twenty are hard compile-stage rejections. Exactly five of those twenty carry a `G`-number, and nothing structural distinguishes them from their siblings. `KeysetMemberNotPooled` is a hard rejection gated inside `seal_dataset`; `OrphanExpression` is a hard rejection gated in the driver. Both reject an invalid artifact at compile; only the second was numbered. `DatasetWithoutSource`, `MeasureNotInModel`, `UnsupportedAggregation`, `CatalogSourceMissing`, the four `RelationshipJoinKey*` diagnostics and `SemanticKindMismatch` are in the same position. Freezing a list of five would freeze an arbitrary fifteen-member exclusion.

**The numbering's ratification is unverifiable at its root.** Both retired deferrals that discuss the roster cite the same authority: a ratification log — "validation gates (G1–G5 ratified; G6 deferred)". That file never existed. The decision on precedence and the ratification log's fate established that `_research/` was a directory that never existed at all, with 33 references pointing into it. So "G1–G5 is the ratified roster" traces to a phantom, and the numbering has no verifiable ratification to preserve. This is not the reason for retiring it — the drift argument above is — but it removes the obligation to supersede anything.

**`G`-numbers are not published and never were.** `Diagnose` is `{message, severity_default, cause}` and `Diagnostic<K>` is `{kind, severity, location, notes}`. Neither carries a code. A `G`-number has never reached a consumer, so retiring the scheme breaks nothing.

### G5 bundled two converse edge directions rather than holding two definitions

**The two ratified sources were not in conflict; they were one name over two rules.** The earlier compile-driver table gives G5 as "Orphan / dead-code detection", with the rationale naming unreachable *expressions*. A deferred question of the manifest build gives G5 as the "manifest-build-time check that no `Relationship` references a missing constituent DataKind" — a **dangling-endpoint** check. Those are converse directions of the same edge: the first requires that every pooled expression is *pointed at*, the second that every pointer *resolves*. Read as rival definitions of one gate they appear irreconcilable, which is why the request demanded a losing source; read as two rules they are both correct and simply belong to different stages.

**So G5's subject is expressions, and the relationship rule is re-homed rather than defeated.** Unreachability of a pooled expression stays with the stage that owns the pool. Endpoint resolution moves onto the relationship-resolution stage as a reference-resolution rule of `G2`'s family. Neither source loses its rule; what is discarded is the numbering that forced them into one slot. **And the endpoint rule is today stage-reachable with no owning site at all** — ADR 0042's third class, a defect: at the time of this decision, `resolve_relationship`'s documentation stated that it did not resolve the names behind `from_id` / `to_id`, and every live `UnknownReference` construction site is in `lower.rs` for semantic names, so nothing resolves relationship endpoints anywhere.

**That deferred question's real axis closes strict.** Its open question was strict rejection versus lenient warn-and-prune, and its own argument against lenient survives the deletion of the tree that carried it: silently dropping an orphan relationship violates determinism, because two compiles of the same model then produce different manifests when the constituent set differs. that determinism rule is still ratified, so strict stands and the parked `compile_lenient` mode is refused rather than left parked.

**G5 belongs to the sealing stage, not to the driver.** Today the sweep runs after sealing, over the assembled artifact — literally the end-of-compile re-walk ADR 0042 states compile does not perform. Scoped instead to the stage that owns the expression pool, "every entry I interned is referenced" is that stage's own rule, asserted where the pool is complete. ADR 0042 is satisfied with no weakening of the check.

### G4 splits three ways, and its diagnostic is deleted

The retired G4 row carries three clauses, not the one the request inherited: "Referenced `PhysicalExprId` exists; `ColumnRef` resolves in the bound `PhysicalSource.projected_schema`; inferred type matches the bound semantic."

**Clause 1 survives as a stage-owned rule.** It is the converse of G5's sweep, and it is assertable by the stage that writes the reference, because that stage has just interned the entry. Under ADR 0043 its subject becomes `definition.expr_id`, `override.expr_id` and the joinset condition pointers resolving in `expressions` — which is, in substance, the universal-resolution check the `SemanticMappingValue` ruling already established when it collapsed the per-variant mapping dispatch into one check that every mapping value resolves in the expression pool.

**Clause 2 is real, and it is vacuous on one of two source forks.** The rule — every `ColumnRef` in a pooled expression resolves in the schema of every physical source that expression lists — becomes the catalog stage's validity definition. It is assertable only on the **catalog-derived** fork, where the catalog is the source of truth for schema. On the **model-fed** fork the schema is derived *from* the mapping, so a dangling column is unrepresentable and the rule is **vacuous by construction, not deferred**. The obligation to say which fork carries it belongs to the physical-source creation-forks decision.

**Clause 3 retires as redundant.** "Inferred type matches the bound semantic" is what boundary reconciliation already does as G3 — ADR 0042's second class, the same rule in one stage under one vocabulary.

**`CompileError::UnresolvedColumn` is deleted rather than given construction sites.** It has none today, and its payload names `binding: EntityId` — an entity ADR 0043 retires. So it is not merely unconstructible; its subject is gone. The successor, minted when the source-fork decision lands, carries `(column, expr_id, physical_source_id)`. A declared-but-unconstructible error is worse than either enforcing or deferring, and deleting it is the only option that does not preserve a reference to a retired entity.

### Two gates are producer-starved, which is a fourth class

**G1 and G2 are defined, gated, tested, and starved.** G1 is acyclicity over the authored-reference graph — Tarjan SCC in `graph::topological_order`, returning `Err(CyclicReference)`, and scoped by the retired table to expression DAGs only. G2 is resolution of every referenced `SemanticsName` to a registry entry, with three `UnknownReference` construction sites in `definitions/mod.rs` and a test. Neither is reachable in a real compile, because an authored `expr:` body is never parsed into a `SemanticExpr`, so the reference graph has no edges by construction.

**That is neither of ADR 0042's three classes, and naming it correctly locates the gap.** They are not producer-unreachable — they are not whole-artifact properties. They are not redundant. They are not stage-reachable-but-ungated, because they *are* gated. They are **gated but producer-starved**: the rule is defined and enforced, and the stage that would supply its input is itself undefined. The consequence is that the definition does not wait on the missing stage. A stage's validity definition does not depend on whether its input is currently populated, so requirement 5 is satisfiable now, and what remains open is a *missing stage definition upstream* — the control-flow question for authored bodies — not a gap in G1, G2, or the roster.

**The marker recording the starvation is not a deferral.** The empty body path is marked in code, at the time of the decision, by a deferral marker across 69 sites in five files, described at its declaration site as "deferred orchestration glue". It appeared in **zero** design documents. It is an undocumented implementation scope cut, so there is no ratified deferral licensing the starved state and no ruling on that marker to await; the definition it withholds is owed by the control-flow question, and this ADR neither inherits it as a boundary nor treats it as settled.

### G3 is overloaded, not partial

Two of G3's three clauses are live: boundary reconciliation raising `TypeInferenceFailure`, and "calls match function signature", which is `lower.rs`'s arity-plus-per-argument match against a signature overload. Its defect is not coverage but **diagnostic overloading** — `TypeInferenceFailure` is used, in the driver's own words, "rather than a dedicated variant" for faults it "reserves for explicit ratification", as a "reused fail-fast shape".

**That is the mirror image of G4's defect, and both are what the freeze mechanism is designed to surface.** G4 held a variant no stage raises; G3 holds one variant raising several distinct faults. A variant set paired with the stage that raises each entry makes both visible from either direction: an entry no stage raises is dead, and a stage rule with no entry cannot be reported. Splitting the overload is code convergence and is not performed here.

**One more unnumbered rule confirms the roster's arbitrariness.** `SemanticKindMismatch` is live, with construction sites in expression lowering and definition compilation (`compile/expressions/lower.rs` and `compile/definitions/mod.rs`), and it is a **role**-mismatch rule — neither name resolution (G2) nor type agreement (G3). It was never numbered, and under this ruling it needs no number.

### Frozen means the definition lives with the stage

Requirement 5 asks for gates "formalized and frozen". The operational meaning is:

- **Each stage states its own validity definition in its own module documentation.** No roster document is created. `docs/` holds `adr/` and `agents/` and nothing else, and a gate registry would be exactly the second-place-to-look failure that the precedence decision removed 46,284 lines to eliminate.
- **Each rule has a test.** This is the only mechanism that fails when a definition and its behaviour diverge.
- **The enumeration is the `CompileError` variant set plus the raising stage per variant.** It is already authoritative, since ADR 0004 makes variant identity the published semver contract, and it is already total.
- **No `G`-numbered enum, no gate registry, no per-gate identifier type.**

### The namespace question has no subject

There is no live code namespace to pin: neither `Diagnose` nor `Diagnostic<K>` carries a code field, so nothing is ever emitted or matched on. The published identity of a compile-stage rejection is the `CompileError` variant and its message. No doc in `crates/` names a `COMP_E_*` string or any other diagnostic code: the codes come from the retired design documents, and a variant's doc states its rule in plain words. `MANIFEST_E_*` is never introduced. Adding `code()` to `Diagnose` would create a second identity for the same fact across every stage in the workspace — a `semstrait-common` diagnostics decision, not a gate decision.

### Three inherited obligations close

**`COMP_E_2414 JoinsetAsOfMissingTemporalKey` is retired.** It required a `JoinType::AsOf` relationship to carry a time-typed `KeyPair`, and it has two dead subjects rather than one. `KeyPair` is deleted, and ADR 0038's names-only basis leaves no per-key site to carry a type. Independently, **there is no `AsOf` join type**: `JoinType` is `{Inner, Left, Right, Full}` and is *derived* from `optional` by `derive_join_type`, never authored. The only `AsOf` in the workspace is `Capability::AsOfJoin`, an adapter capability flag, and an engine capability does not create an authoring form. Restating the gate over the resolved dimension's `data_type` would mint a rule for an authoring form that does not exist — deliberately committing the error that G4's unconstructible variant committed by accident. It returns with temporal-join authoring, if that ever lands.

**The reserved G6, relationship-graph cycle detection, is ruled out of compile's validity definition — closed, not deferred.** It was parked on two grounds. The first, that the planner's request-time graph already enforces DAG-ness, is **struck**: under ADR 0042 "a downstream stage will catch it" is not a validity statement about *this* stage, and it defers to a crate that is `exclude`d from the workspace and still on the legacy compiled-manifest type. The second is sufficient and is the right kind of reason — some legitimate models declare relationships that look cyclic undirected but are acyclic when traversed under `cross_filter` directionality. A property that legitimate models exhibit is not a defect, so rejecting it would be a false positive.

**The composition-depth bound is handed over with its subject inverted.** `MAX_IMPLICIT_COMPOSITION_DEPTH` does not exist in `crates/` at all — it was retired text (`= 4`) that was never built, while ADR 0014 names it live alongside `MAX_IMPLICIT_ENUMERATION_COUNT = 2000` at graph-build time with a breach as a hard error. The human ruled that the bound should be a **construction boundary on authored nested data kinds** — `MAX_COMPOSITION_DEPTH` — and that compiler-derived implicit composition may legitimately run deeper than a modeller would assume. That **inverts** the retired position, which exempted authored Joinsets outright: "Declaring a `Joinset` is exactly what an author does when their path exceeds the implicit depth cap; imposing the same cap on the escape hatch would be pointless." The bound's subject and value belong to the implicit-Joinset enumeration decision, which owns what bounds enumeration and owes the supersession against ADR 0014; ruled here is only that if the breach survives as an error, it is a rule of the enumerating stage and gets no number.

### Consequences

Compile keeps every rule it has today except `UnresolvedColumn`, which is deleted. No check is weakened, and one previously unowned rule — relationship endpoint resolution — gains an owner. The cost is that "which rules does compile enforce?" is answered by reading the stages and the variant set rather than one table, which is a worse experience for a reader wanting a summary and a better one for a reader wanting the truth, since the table has no mechanism keeping it honest and the variant set does.

Handoff notes for the implementation effort, none of them performed here:

- Delete `CompileError::UnresolvedColumn`, whose `binding` payload names a retired entity.
- Give relationship endpoint resolution an owning site.
- Move the orphan-expression sweep from the driver into the sealing stage, so no pass runs over the assembled artifact.
- Split `TypeInferenceFailure`'s overload into dedicated variants for the faults its own documentation flags as awaiting ratification.
- State each stage's validity definition in that stage's module documentation, and pair each `CompileError` variant with its raising stage.
- The orphan sweep's transitive closure over `expressions.depends_on` cannot be written until the expression-edge direction is settled: the base schema stores the *parent*, ADR 0037 stores what an expression *depends on*. A sweep walks the wrong way under the wrong reading.
