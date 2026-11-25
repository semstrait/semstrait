---
prereqs: [00]
status: superseded
---

# 30_api_contracts (superseded — split)

Status: **Superseded** — split into single-concern leaves under
[`30_api_contracts/`](30_api_contracts/) per the design doc-space restructure.
Navigate via [`registry.json`](../registry.json); do not read this file to decide.
Kept as a forwarding stub for inbound links.

## Leaves

- [30-api-versioning-surface-extensibility](30_api_contracts/api-versioning-surface-extensibility.md) — Cross-crate semver posture, public vs internal surface policy, and non-exhaustive-by-default policy
- [30-diagnostic-structure](30_api_contracts/diagnostic-structure.md) — Generic Diagnostic<K> envelope, Severity, Location, Span, SourceId, Diagnose trait, per-stage typed-kind discipline, cross-stage aggregation
- [30-observability-tracing](30_api_contracts/observability-tracing.md) — Workspace observability policy — tracing adoption, no stdout/stderr from library code, canonical field names, Diagnostic ↔ tracing orthogonality
- [30-error-emission-policy](30_api_contracts/error-emission-policy.md) — Per-stage error-emission policy at API boundaries — accumulating vs fail-fast, return shape per stage, warnings never silently dropped
- [30-trait-surface-rules](30_api_contracts/trait-surface-rules.md) — Public trait surface rules — sealed vs open, cross-cutting trait roster, trait-method return shape, async trait methods
- [30-async-posture](30_api_contracts/async-posture.md) — Per-crate sync/async posture — which crates are sync-only, which have compile-time async, runtime choice, and no .await inside the hot path
- [30-feature-flag-policy](30_api_contracts/feature-flag-policy.md) — Feature-flag policy — default features minimum viable, adapter crates are separate crates not feature flags, catalog provider crates are separate, optional serialization features, banned feature patterns
- [30-breaking-change-governance](30_api_contracts/breaking-change-governance.md) — Breaking-change governance and deprecation policy — additivity escape hatch, non-additive changes require MAJOR changelog + migration notes + deprecation window, deprecation lifecycle
