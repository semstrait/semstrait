# Pillar 2 — Schema Shape
#
# Contract gates for the shape and invariants of the compiled SemanticManifest —
# the boundary artifact between compile-time and query-time. Source of
# truth: ADR 0010 (docs/adr/0010-single-entityid-lane-content-derived-ids.md) and
# ADR 0011 (docs/adr/0011-thin-manifest-in-memory-graph.md). Each Then gates one
# clause of them.
#
# Most gates are @target: the mid-migration manifest (CompiledManifest) does not
# yet match the ratified single-id-lane / physical-only-pool shape. They are
# executable specification of the target and ratchet to @now as the code earns
# them.

@schema
Feature: Schema shape — the compiled SemanticManifest matches its contract

  # --- Single identity lane --------------------------------------

  @target
  Scenario: Every entity registry is keyed by EntityId
    Given the reference model
    When I compile it
    Then the manifest registries (data_kinds, interfaces, bindings, sources, relationships) are each keyed by EntityId

  @target
  Scenario: Every entity's id equals its map key
    Given the reference model
    When I compile it
    Then the manifest every entity id equals its containing map key

  @target
  Scenario: Cross-references are EntityIds, never per-kind id newtypes
    Given the reference model
    When I compile it
    Then the manifest contains no DataKindId, BindingId, SourceId, or RelationshipId newtype
    And the manifest contains no stable_ids side-map

  @target
  Scenario: A deterministically minted model_id exists and is globally unique
    Given the reference model
    When I compile it
    Then the manifest has a model_id that is a canonical UUID
    And the model_id is distinct from every entity id

  # --- The lone integer: bit_position --------------------------------

  @target
  Scenario: bit_position is the only integer in the bitmap subsystem
    Given the reference model
    When I compile it
    Then every SemanticDefinition carries a unique bit_position
    And the bit_position-to-EntityId reverse index is not persisted

  # --- Physical-only expression pool -------------------------------

  @target
  Scenario: The expression pool is physical-only and content-addressed
    Given the reference model
    When I compile it
    Then the manifest expression pool is keyed by PhysicalExprId
    And the manifest has no semantic expression pool and no SemanticExprId
    And no UnresolvedSemanticExpr is persisted in the manifest

  @target
  Scenario: Every pooled expression stores a layer equal to classify(expr)
    Given the reference model
    When I compile it
    Then every ManifestExpression has layer == classify(expr)

  @target
  Scenario: Additivity and sugar never reach the pool
    Given the reference model
    When I compile it
    Then the pooled expressions are post-desugar canonical form
    And no additivity is stored on the manifest

  # --- Bindings ----------------------------------------------------

  @target
  Scenario: Bindings attach to leaf Datasets and their mapping/resolved keysets coincide
    Given the reference model
    When I compile it
    Then every SemanticBinding attaches to a leaf Dataset data_kind
    And every SemanticBinding's mapping and resolved keysets are equal
    And every Expr-realized mapping references the pool by PhysicalExprId

  # --- Resolved relationships ---------------------------------------

  @target
  Scenario: Relationships are stored resolved, not in authoring form
    Given the reference model
    When I compile it
    Then the manifest stores ResolvedRelationship, never the authoring Relationship
    And every ResolvedRelationship has materialized optional and cross_filter
    And every ResolvedRelationship's join_type equals derive(optional)

  @target
  Scenario: Join keys reference semantics by EntityId with a legal role
    Given the reference model
    When I compile it
    Then every ResolvedJoinKey side references a semantic by EntityId
    And every ResolvedJoinKey side has role in {Dimension, Field, Key}

  # --- Temporal -----------------------------------------------------

  @target
  Scenario: temporal is Some only on leaf Datasets
    Given the reference model
    When I compile it
    Then every DataKind with temporal == Some is a leaf Dataset
    And every Unionset, Grainset, and Joinset has temporal == None

  # --- Load-time integrity --------------------------------------

  @target
  Scenario: A well-formed manifest passes load-time integrity
    Given the reference model
    When I compile it
    Then the manifest passes SemanticManifest::validate

  @target
  Scenario Outline: Load integrity rejects a tampered manifest
    Given the reference model compiled to a manifest
    When I tamper the manifest by "<tamper>"
    Then loading it fails with <load_error>

    Examples:
      | tamper                                      | load_error                     |
      | pointing a binding at a missing source id   | LoadError::DanglingReference   |
      | setting an entity id different from its key | LoadError::StructuralViolation |
      | adding a trailing all-zero bitmask word     | LoadError::StructuralViolation |
      | setting a ManifestExpression layer wrongly  | LoadError::StructuralViolation |
      | setting join_type inconsistent with optional| LoadError::StructuralViolation |

  # --- MUST-NOT persistence boundary -------------------------

  @target
  Scenario: The manifest persists identity on disk, not the runtime graph
    Given the reference model
    When I compile it
    Then the manifest serialization contains no nodes, edges, or compositions collection
    And the manifest serialization contains no planner cache state
    And the manifest serialization contains no ExprSource
    And the manifest serialization contains no raw SQL or engine-native types

  # --- Determinism — see also pipeline_end_result.feature ------

  @determinism @target
  Scenario: Identical inputs compile to byte-identical manifests
    Given the reference model
    When I compile it twice
    Then the two manifests are byte-identical

  @determinism @target
  Scenario: Reordering authored pools does not change the manifest bytes
    Given the reference model
    And the same model with its top-level pools authored in reverse order
    When I compile both
    Then the two manifests are byte-identical
