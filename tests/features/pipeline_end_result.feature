# Pillar 4 — Pipeline End-Result
#
# Contract gates for the terminal artifact of each pipeline stage. Source of
# truth: ADR 0003 (docs/adr/0003-compile-owns-resolution-and-io.md, stage
# contracts), ADR 0018 (docs/adr/0018-canonical-plan-ir-contract.md, SemanticPlan)
# and ADR 0023 (docs/adr/0023-adapter-emission-architecture.md, EngineArtifact).
# Each Then gates one clause of them.
#
# No test runs this file: ADR 0035's suite crate, `semstrait-contract-gates`,
# is not built. parse/validate/compile gates are @now; plan/adapt gates are
# @target.

@pipeline
Feature: Pipeline end-result — each stage yields its contracted artifact

  # --- Stage 1: parse -> SemanticModel -----------------------------

  @now
  Scenario: Parsing a well-formed model yields a typed SemanticModel
    Given the reference model
    When I parse it
    Then it parses successfully into a SemanticModel

  @now
  Scenario: Parse accumulates errors rather than failing fast
    Given a model with two independent malformed declarations
    When I parse it
    Then parsing reports more than one diagnostic

  # --- Stage 2: validate -> Result<(), Vec<ValidateError>> ---------

  @now
  Scenario: Validation is a pure accumulating predicate
    Given a model with two independent structural violations
    When I validate it
    Then validation reports more than one diagnostic

  # --- Stage 3: compile -> SemanticManifest ------------------------

  @now
  Scenario: Compiling the reference model yields a SemanticManifest
    Given the reference model
    When I compile it
    Then compilation succeeds and yields a SemanticManifest

  @now
  Scenario: Compile is fail-fast
    Given a model with a compile-stage resolution error followed by another
    When I compile it
    Then compilation fails with a single fatal diagnostic

  @now
  Scenario: Raw SQL in an expression is rejected, never carried into the manifest
    Given a measure whose expr is a raw SQL string
    When I compile it
    Then compilation fails rejecting the raw SQL

  @determinism @target
  Scenario: Compile is deterministic over identical inputs
    Given the reference model
    When I compile it twice
    Then the two manifests are byte-identical

  # --- Stage 4: plan -> SemanticPlan -------------------------

  @target
  Scenario: Planning yields a SemanticPlan rooted in a PlanNode
    Given the reference model compiled to a manifest
    When I plan a request for revenue by country
    Then the plan has a root PlanNode and output_names
    And output_names.len() equals the root output schema arity

  @target
  Scenario: The PlanNode sum is closed to the eight canonical variants
    Given any planned request
    Then every PlanNode is one of { Scan, Filter, Project, Agg, Join, Union, Sort, Fetch }

  @target
  Scenario Outline: Plan-level structural invariants hold
    Given a planned request
    Then <invariant>

    Examples:
      | invariant                                                              |
      | no Filter or Project predicate embeds an Aggregate node                |
      | every stored PhysicalExpr is fully type-resolved                       |
      | a Filter predicate has Boolean inferred type                           |
      | every AggregateNode aggregate has a unique output Name                 |
      | a Join output schema widens nullability on the outer side             |

  # --- Stage 5: adapt -> EngineArtifact ------------------

  @target
  Scenario: Adapt produces an EngineArtifact and SQL appears only here
    Given a planned request
    When I adapt it with the ANSI SQL adapter
    Then the artifact is EngineArtifact::Sql
    And no raw SQL existed in the plan before adapt

  @target
  Scenario: No Parameter survives into adapt
    Given a planned request that used a dimension accessor
    When I adapt it
    Then the artifact contains no unbound Parameter

  # --- Behavioral end-result by DataKind (the "what should it produce" tier) -
  # @target — these re-anchor the legacy suite's intent onto the canonical plan.

  @target
  Scenario: A grainset query selects the coarsest level satisfying the grain
    Given a grainset model with year, month, and day levels
    When I plan a request at month grain
    Then the plan scans the month (or coarser) level, not the day level

  @target
  Scenario: A unionset query unions its branches and NULL-fills partial coverage
    Given a unionset whose branches cover different semantics
    When I plan a request spanning all branches
    Then the plan contains a Union node
    And semantics absent from a branch are NULL-filled for that branch

  @target
  Scenario: A joinset query joins members with the derived join_type
    Given a binary joinset with an optional relationship
    When I plan a request spanning both members
    Then the plan contains a Join node with the join_type derived from optional

  @target
  Scenario: A constraint violation is raised at plan time, before artifact emission
    Given a measure constrained to require a temporal dimension
    When I plan a request for that measure without the temporal dimension
    Then planning fails with plan.constraint-violation
