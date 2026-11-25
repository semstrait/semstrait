---
prereqs: [00]
status: superseded
---

# 31b_semstrait_common_io (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`31b_semstrait_common_io/`](31b_semstrait_common_io/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [31b-io-error-kind](31b_semstrait_common_io/io-error-kind.md) — IoErrorKind typed-kind enum, variant taxonomy, Diagnose impl, stage-of-origin identification, and cross-crate composition
- [31b-io-repository](31b_semstrait_common_io/repository.md) — Persistable (the trait a DTO implements to be stored), generic Repository<T>, FileSystemRepository<T> over a FileSystem transport, RepositoryErrorKind taxonomy, and the backend-extension seam

The byte-transport half of this module — the former `Source` / `Sink` / `Location`
handle surface and its `memory` / `local` / `s3` back-ends — was consolidated into
the single `FileSystem` facade ([`37_semstrait_catalog/filesystem.md`](37_semstrait_catalog/filesystem.md));
load/save by identity is the [`31b-io-repository`](31b_semstrait_common_io/repository.md)
seam above. Git history carries the consolidated leaves.
