---
prereqs: [00]
status: superseded
---

# 37_semstrait_catalog (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`37_semstrait_catalog/`](37_semstrait_catalog/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [37-catalog-crate-surface](37_semstrait_catalog/catalog-crate-surface.md) — semstrait-catalog crate charter: purpose/scope/layering, public crate surface (roster + module layout + structural-type sketches), CatalogProviderErrorKind / FileSystemErrorKind taxonomy + cross-crate wrapping + code-range registration, caller responsibilities (construction/DI/thread-safety/error-propagation/async-runtime), stability posture, crate boundaries, and cross-references
- [37-catalog-provider](37_semstrait_catalog/catalog-provider.md) — CatalogProvider trait: purpose, trait definition (resolve_table entry), method contracts, invariants, non-goals; per-provider v1 implementations (Noop/IcebergRest/Unity/Filesystem); and schema-drift gated I/O (I11b check_schema_drift signature + deterministic contract + caller policy + I11b boundary)
- [37-filesystem](37_semstrait_catalog/filesystem.md) — FileSystem trait: purpose, trait definition, method contracts, invariants, non-goals; per-filesystem v1 implementations (Local/S3/Azure/Gcs); and glob expansion (shared utility, composition with CatalogProvider, determinism)
