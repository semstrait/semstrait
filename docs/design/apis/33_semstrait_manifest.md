---
prereqs: [00]
status: superseded
---

# 33_semstrait_manifest (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`33_semstrait_manifest/`](33_semstrait_manifest/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [33-compile-driver](33_semstrait_manifest/compile-driver.md) — The compile driver (compile() entry + §3.9 pass orchestration), validation gates G1–G5, desugar policy D2, and deterministic EntityId minting (UUIDv5 content-derived, per-entity-type namespace)
- [33-manifest-contract](33_semstrait_manifest/manifest-contract.md) — The SemanticManifest contract: top-level shape (§4 SemanticManifest + SemanticManifestMetadata + top-level fingerprints), SemanticBitmap registry (§5 C4: SemanticDefinition + SemanticBitmask + bit_position mapping), DataKind/DataKindVariant roster (§6 CCK + C5–C9: Dataset/Unionset/Grainset/Joinset + NestedDataKind Public/Nested split), single-identity-lane EntityId posture (no per-kind compile-id newtypes, no stable_ids side-map)
- [33-binding-expr-source](33_semstrait_manifest/binding-expr-source.md) — The manifest binding/expression/source shapes: SemanticBinding (C2: leaf-only linkage, mapping vs resolved dual views, per-(semantic, binding) context), ManifestExpressions (C11/C12/C18: physical-only pool + PhysicalExprId handle + ExprLayer extra-data, compile-time dedup, Serialize IR-owned), PhysicalSource (C1/C3/C14/C15: schema_fingerprint SHA-256, catalog fetch policy, locator + version_ref identity tuple), ResolvedRelationship/ResolvedJoinKey (§8A: semantic-id-pair join-key contract, residual filter PhysicalExprId, defaults-applied optional/cross_filter, derived join_type, root-scope vs Joinset-scope-local), TemporalShape/TemporalType (§8B: leaf-effective propagation, identifying-Dimension EntityId resolution, grain generic field)
