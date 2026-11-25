---
number: 11
title: 'Thin manifest: persist identity primitives, rebuild all structure in memory'
status: accepted
date: 2026-08-04
source:
- crates/semstrait-manifest/src/manifest
- crates/semstrait-manifest/src/compile/driver.rs
- crates/semstrait-ir/src/semantic_graph.rs
links:
- target: 43
  kind: amendedby
- target: 49
  kind: amendedby
- target: 50
  kind: amendedby
- target: 54
  kind: amendedby
- target: 56
  kind: amendedby
---

# Thin manifest: persist identity primitives, rebuild all structure in memory

> **Amended.** Later ADRs changed parts of this decision. Where they differ from the text below, the later ADR governs.
>
> - [ADR 0043](0043-authored-primitives-get-typed-carriers.md): `SemanticBinding` retires and `SemanticInterface` stops being an entity.
> - [ADR 0049](0049-a-dataset-names-one-physical-source-as-a-semantic-source.md): a Dataset over several physical sources persists a compile-synthesised Unionset in the manifest; synthesising it in the graph, "ADR 0011's posture", is rejected.
> - [ADR 0050](0050-a-relationship-is-compile-input-not-a-manifest-entity.md): `SemanticManifest.relationships`, `ResolvedRelationship` and `ResolvedJoinKey` retire; a relationship persists as the implicit joinset it produced, carrying a pooled join condition.
> - [ADR 0054](0054-compile-enumerates-every-legal-sub-composition.md): the enumeration clause, for authored Joinsets only: sub-compositions of one authored Joinset's specification are persisted. The **model-wide relationship adjacency index stays forbidden**, and so does every other structure 0011's rejected alternative names — `nodes:`/`edges:`/`compositions:`, `ComposedSemanticInterface`, `UnifiedSemantics`, `FieldProvenance`, a pre-built `RelationshipGraph` and a serialized graph cache. **This is the second trade against 0011, not the first**: ADR 0049 already rejected *"synthesise the Unionset in the graph rather than in the manifest, which is ADR 0011's posture"* as a named alternative and recorded that its own reduction is *"more entities for the same model, in an artifact whose stated posture is thinness (ADR 0011)"*, without registering a supersession. 0011's open question about planner-completeness is resolved in the direction it anticipated: for these compositions, planner-completeness is met by the artifact rather than by graph-build synthesis.
> - [ADR 0056](0056-composed-temporal-shape-is-derived-at-compile.md): composed temporal shape is derived at compile and persisted; the derived-structures clause is superseded for this one field, on the terms ADR 0054 took for enumeration; the model-wide relationship adjacency index and every other structure ADR 0011 lists stay forbidden.

## Context and Problem Statement

The manifest's shape is the serialized artifact schema and the definition of where planner-completeness is satisfied, so it is fixed deliberately: re-admitting derived structures would change every persisted manifest, its loader and all of its consumers. The shape is intentionally counter-intuitive — a "compiled", "planner-complete" artifact throws away exactly the resolved graph a reader expects it to contain and recomputes it on every load.

## Considered Options

- A thin manifest of identity primitives, with all structure rebuilt in memory (chosen).
- The earlier ratified heavy-manifest contracts, which persisted `nodes:`/`edges:`/`compositions:`, per-Grainset JOIN trees, `ResolvedGrainset`/`ResolvedJoinset` composed interfaces and a pre-built `RelationshipGraph`, together with a serialized graph cache. Rejected because they widen the persisted schema with values that are fully derivable, forcing every structural change to become a format migration.

## Decision Outcome

The `SemanticManifest` carries only identity primitives — registry, DataKinds, interfaces, bindings, sources, expression pool, resolved relationships, coverage masks.

Nodes, edges, `compositions:`, JOIN trees, `ComposedSemanticInterface`/`UnifiedSemantics`/`FieldProvenance`, the relationship adjacency index and every per-Request structure are reconstructed at `SemanticGraph` build. No `semantic_graph` type derives serde.

## More Information

### Open questions

Manifest thinness sits uneasily with the planner-completeness invariant ("planner-complete"), which reads as "every plan-time input is in the manifest". Either the invariant's wording needs amending, or graph-build synthesis must be declared formally part of planner-completeness.
