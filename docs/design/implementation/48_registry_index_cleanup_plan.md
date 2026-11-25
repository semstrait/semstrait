---
title: Registry Index Cleanup (docs harness) — Implementation Plan
status: to do
effort: strip placeholder/hand-maintained rollup fields; declare status+impl enums in schema; resolve the "mapping" lexical collision; add traversal protocol pointer; generate rollups by script + verify in CI; optional per-space aliases
integration_branch: created at execution off feature/semstrait-mvp (design-only this round — no branch yet)
worktree: .claude/worktrees/docs+io-persistence-seam (authoring home)
spec: docs/design/registry.json (the artifact); .agents/scripts/doc-space-check.py (the gate; rollup check at lines 272–305); .agents/skills/semstrait-doc/MAINTAINER.md:29 (status lifecycle — current home after CONVENTIONS.md deletion)
sibling_baseline: doc 46 (doc-space restructure — P5 legacy reconcile still open); commit 7baad0f (governance centralized into .agents/skills/semstrait-doc/, CONVENTIONS.md deleted)
session_ruling: scoped 2026-07-24 from a 6-bullet directive. R1/R2 below are BLOCKING decisions for the user — real data violates the enum the plan is asked to declare. Design-only; approve-then-execute.
---

# 48. Registry Index Cleanup (docs harness)

`registry.json` is the top-level routing index. Goal: remove noise, formalize
semantics, resolve lexical collisions, and make rollups tool-generated rather than
hand-maintained.

## 1) Verified baseline (2026-07-24, post-pull)

Measured, not assumed — `jq` over every `_manifest.json` at this branch's HEAD.

**Registry state:** unchanged by the recent governance commit (`7baad0f`). All six
requested cleanups are open. `version_range: "unknown"` appears in **all 6** spaces;
`migrated` is hand-maintained in all 6.

**The gate already half-does task 5.** `doc-space-check.py:272–305` verifies rollup
`leaves`, `ratified`, and `draft` against each manifest — and *ignores* `proposed`,
`migrated`, and `version_range` (comment: *"only ratified/draft for simplicity"*).
That is exactly why the placeholders survived: nothing checks them.

**Actual counts vs registry rollup** (all `leaves`/`ratified`/`draft` currently
agree — the gate enforces those three):

| Space | leaves | ratified | proposed | other | registry `migrated` |
|---|---|---|---|---|---|
| foundations | 70 | 61 | 0 | **2 `partially ratified`** | 63 |
| data-kinds | 28 | 23 | 0 | 0 | 23 |
| apis | 61 | 51 | 1 | 0 | 52 |
| testing | 1 | 0 | 1 | 0 | 1 |
| registry | 20 | 17 | 0 | 0 | 17 |
| questions | 65 | 0 | 0 | 0 | 0 |

**Values actually in use** (the enum must be declared as a **superset of these**,
not a subset — the directive's `draft | ratified | ...` was illustrative, and no
previously-used status is dropped except `migrated`, which was migration-tracking
only):

| Axis | Values in use | Count |
|---|---|---|
| `status` | `ratified` | 152 |
| | `proposed` | 2 |
| | `partially ratified` | 2 |
| `impl` | `implemented` | 85 |
| | `drift` | 27 |
| | `in-progress` | 14 |
| | `planned` | 10 |
| | `none` | 7 |
| | `partially implemented` | 2 |
| | free-text prose (see below) | 12 |

**`status`/`impl` absent on 89 leaves = exactly the unmigrated set.** Verified: the
count of leaves with no `status` is **89**, identical to the gate's census
(`156 migrated, 89 unmigrated`). This is not a defect — it is the structural shape
of an unmigrated row, and the gate deliberately skips them
(`doc-space-check.py:323-324, 369`). The enum therefore constrains **migrated leaves
only**; unmigrated rows are exempt until migrated.

**`impl` carries free-text prose on 12 leaves** — e.g. *"semstrait-model owns
UnresolvedSemanticLeaf; per-kind typed; accessor Option<XxxAccessor> carriers"*,
*"item Y inversion (ir→model edge, not model→ir); common owns Tree + ValidateError;
…"*. The prose is load-bearing ownership detail, so it is **relocated, not
deleted** (T2).

**`superseded` is retired as a status** (ruling, this session): each new design is a
new branch + commit, so git carries prior state and a doc-level "superseded" marker
tracks nothing useful. MAINTAINER.md:37-48 already forbids creating or retaining
`status: superseded`. **Verified: 24 files still carry `status: superseded`** — so
this is a real cleanup (T8), not just an enum edit. (`retired`: 0 files.)

## 2) Rulings applied (this session)

- **R1 — `partially ratified` / `partially implemented`: KEEP as declared enum
  members.** The enum is declared at a higher level as a superset of real usage; no
  previously-used status is omitted. (Affects 2 + 2 leaves; no leaf edits needed.)
- **R2 — `superseded`: REMOVED.** New design = new branch/commit; git is the history.
  Requires the 24-file cleanup in T8.
- **R3 — free-text `impl` (12 leaves): relocate to `impl_notes`**, set `impl` to its
  enum value. Prose preserved, axis made machine-checkable.
- **`migrated`: the only genuinely dropped field** — it tracked the old→new docs
  structure migration. **Constraint:** drop it from the **registry rollup** (T1), but
  the **leaf-level flag must stay** until migration completes — gate conditions 1
  and 4 branch on it (`if not leaf.get('migrated'): continue`), so deleting it
  tree-wide would silently un-gate all 156 migrated leaves. T3 covers the sequencing.

## 3) Tasks

`V:` = verification.

| # | Task | Change | Done when | Deps |
|---|---|---|---|---|
| **T1** | drop dead fields | remove `version_range` (all 6); remove `migrated` from rollup (leaf flag stays — see T3) | no placeholder-valued field remains in registry | — |
| **T2** | declare enums in schema | add `leaf_status` + `leaf_impl` enums (superset of real usage) + `impl_notes`; state the two axes are independent | status semantics derivable from `schema` without reading manifests | — |
| **T3** | migration metrics → report | rollup field removed; leaf `migrated` flag retained until migration closes | census still available; gate conditions 1/4 still gate | T1 |
| **T4** | fix "mapping" collision | qualify `foundations.use_when` "mapping" → "model-to-source mapping" | every `use_when` term unique across spaces or resolved via `skip_when` | — |
| **T5** | traversal protocol | `how_to_use` = one-line pointer to `EXPLORER.md` (already owns it) | routable from the registry alone; no duplicated protocol text | — |
| **T6** | generate + verify rollups | script builds rollups from `_manifest.json`; gate checks **all** enum'd statuses | rollup reproducible by command; mismatch = CI fail | T1, T2 |
| **T7** | *optional* per-space `aliases` | synonym lists ("FK", "join keys", "relationship") | lexical coverage without bloating `use_when` | T4 |
| **T8** | purge `status: superseded` | remove the 24 files/sections still carrying it (R2) | `grep -rl '^status: *superseded' docs/design/` = 0; gate green | T2 |

### T1 — Remove dead fields

- **`version_range`**: `"unknown"` in all 6 spaces — a placeholder that has never
  carried a value. Drop it. (Alternative: actually compute `min–max` leaf `version`
  at generation time — the gate already collects a `versions` set at
  `doc-space-check.py:288` but never uses it. **Recommend drop**; add back only if a
  consumer needs it.)
- **`migrated`**: hand-maintained and a *migration-phase* metric, not routing
  information. Remove from rollup; see T3.

**V:** `jq '.spaces[].rollup | keys' docs/design/registry.json` shows no
`version_range` / `migrated`; `grep -c '"unknown"'` = 0.

### T2 — Declare status/impl semantics in `schema`

Declared as a **superset of real usage** (R1) with `superseded` omitted (R2):

```jsonc
"leaf_status": ["draft", "proposed", "partially ratified", "ratified", "retired"],
"leaf_impl":   ["none", "planned", "in-progress", "partially implemented",
                "implemented", "drift"],
"impl_notes":  "optional free text — ownership/scope detail; NOT a status value (R3)",
"axes": "status and impl are INDEPENDENT: status = spec maturity, impl = code state.
         A ratified leaf may be impl:none; an implemented leaf may be status:draft.",
"applies_to": "migrated leaves only — unmigrated rows carry neither field"
```

`retired` and `draft` are declared though currently unused (0 leaves each): the enum
is the *lifecycle*, not a census of today's tree. `superseded` is deliberately absent
(R2) — git carries prior state.

The two axes are **mutually independent, not nested** — that is the relationship the
directive asks to be made explicit, and it is load-bearing (a `ratified` + `drift`
leaf means "spec agreed, code diverged", the single most actionable combination).

Rollup counts are then computed **strictly** against `leaf_status`, and a **migrated**
leaf whose value is off-enum is a **hard gate failure**, not a silent `unknown`
bucket. Unmigrated rows (no `status`/`impl`) stay exempt, matching the gate's existing
skip.

**V:** gate fails on a deliberately off-enum fixture (e.g. `status: bogus` on a
migrated leaf); passes on the current tree with **no leaf status edits** (R1 legalizes
`partially ratified` / `partially implemented`); the 12 prose rows move to
`impl_notes` and their `impl` becomes an enum value.

### T3 — Migration metrics → separate artifact

`migrated` is a doc-46-restructure progress metric (156 migrated / 89 unmigrated
today, printed by the gate's census line). It is migration bookkeeping, not routing
information, so it leaves the **registry rollup** (T1).

**Constraint — the leaf-level flag must stay for now.** Gate conditions 1 and 4
branch on it (`if not leaf.get('migrated', False): continue` at
`doc-space-check.py:323-324, 369`): required-field and frontmatter-equality checks
apply *only* to migrated leaves. Deleting the flag tree-wide would make every leaf
look unmigrated and **silently un-gate all 156 migrated leaves**. Sequence:

1. now — remove from registry rollup (T1); keep leaf flag; keep census output;
2. when the last leaf migrates (doc 46 P5) — drop the flag and the branch together,
   so the gate then applies unconditionally.

**V:** census line still printed; registry carries no migration field; gate findings
count unchanged before/after T1 (proves rollup removal did not weaken gating).

### T4 — Resolve the "mapping" lexical collision

`foundations.use_when` contains bare **"mapping"**, which overlaps
`registry.use_when`'s "types/functions/temporal/join mapping per engine". A router
matching "mapping" cannot discriminate.

- `foundations`: "mapping" → **"model-to-source mapping"**
- `registry`: already qualified ("… per engine") — keep, verify no residual bare term.

**V:** a term-uniqueness check across all `use_when` strings reports no token
appearing in 2 spaces unqualified. Worth adding to the gate as a 9th condition
(cheap, prevents recurrence) — this is the enforcement the acceptance criterion
implies.

### T5 — Traversal protocol

**Verified: the protocol already exists in the harness.**
`.agents/skills/semstrait-doc/EXPLORER.md` is declared *"Navigate `docs/design/`
from registry metadata without scanning leaf Markdown"* (line 3) and its steps
already run *"Read `docs/design/registry.json`"* → *"selected space's
`_manifest.json`"* (lines 11–13).

**Ruling: pointer, not a copy.** Per the directive's own escape clause ("skip if
the protocol already lives in the harness — reference it instead"), `how_to_use`
becomes a one-line reference to `EXPLORER.md`, not a restatement. Duplicating it
would recreate exactly the two-homes drift that deleting `CONVENTIONS.md` resolved.

**V:** `how_to_use` names the owning skill file; `grep` finds no second copy of the
3-hop protocol text under `docs/design/`.

### T6 — Automate rollup generation + CI verification

- New script (e.g. `.agents/scripts/gen-registry-rollup.py`) recomputes every
  `rollup` from the space manifests and rewrites `registry.json` in place.
- Extend the gate's existing check (`doc-space-check.py:272–305`) from
  `['ratified','draft']` to **every** `leaf_status` member, and fail on off-enum or
  missing values.

**V:** `gen-registry-rollup.py && git diff --exit-code docs/design/registry.json`
→ exit 0 (idempotent, generated state == committed state). Mutating a manifest
status without regenerating ⇒ gate exit non-zero.

### T7 — Optional: per-space `aliases`

`"aliases": ["FK", "join keys", "relationship", …]` per space, for query lexical
coverage without bloating `use_when`. Additive; no gate change required beyond
allowing the key. Defer if T1–T6 land clean — value is speculative until a routing
miss is observed.

### T8 — Purge `status: superseded` (24 files)

Per R2, `superseded` is not a legal status: a new design is a new branch + commit, so
git carries prior state and a doc-level marker tracks nothing. The harness already
mandates this — `MAINTAINER.md:37` (*"Replacement is removal of stale design, not
preservation of a superseded…"*), `:46` (*"Do not create or retain a `status:
superseded` section, leaf, forwarding stub"*), `:48` (*"Existing migration-only
superseded stubs are removed when their inbound [references are repointed]"*) — but
nothing enforces it, and **24 files still carry it**.

Per-file disposition: repoint inbound references, then delete the stub/section.
Deletion order matters — a stub removed before its inbound refs are repointed
produces dangling links (gate condition 7).

**V:** `grep -rl '^status: *superseded' docs/design/` = 0; gate exit 0 (condition 7
proves no dangling links were introduced); enum check (T2) has no `superseded` member
to legalize.

## 4) Acceptance (from the directive, mapped)

| Criterion | Task | Verification |
|---|---|---|
| No placeholder-valued fields remain | T1 | no `version_range`; `grep '"unknown"'` = 0 |
| Status semantics derivable from `schema` alone | T2 | `leaf_status` + `leaf_impl` + `impl_notes` + axes note present |
| Every `use_when` term unique across spaces or resolved via `skip_when` | T4 | term-uniqueness check (candidate gate condition 9) |
| Rollup reproducible by command + verified in CI | T6 | generator is idempotent; gate covers all statuses |
| No `superseded` state retained (R2) | T8 | `grep` = 0; gate green |

## 5) Cross-references

- **Artifact:** `docs/design/registry.json`.
- **Gate:** `.agents/scripts/doc-space-check.py` (rollup check 272–305; conditions
  1–8; census 172–173).
- **Governance home:** `.agents/skills/semstrait-doc/` — `MAINTAINER.md:29` (status
  lifecycle), `EXPLORER.md` (navigation, candidate owner of the T5 protocol).
  `docs/design/CONVENTIONS.md` was deleted in `7baad0f`; do not cite it.
- **Sibling:** doc 46 (doc-space restructure) — its P5 legacy reconcile is the other
  open harness workstream; T3's migration artifact retires with it.
