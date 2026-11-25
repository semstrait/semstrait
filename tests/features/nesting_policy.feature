# Pillar 1 — Nesting Policy
#
# Contract gates for the structural skeleton every SemanticModel must obey,
# BEFORE any semantic resolution. Source of truth: ADR 0026
# (docs/adr/0026-inline-only-nesting-tree.md) and the validation in
# crates/semstrait-model/. Each Then gates one clause of it.
#
# Lifecycle tags: @now = expected to hold against the current buildable code
# (parse/validate run today); @target = ratified target not yet earned.

@nesting
Feature: Nesting policy — the DataKind tree is structurally legal

  # --- The nesting matrix --------------------------------------------
  # Parent x child admissibility. Legal cells parse; same-variant diagonal
  # cells are R2 rejections; leaf-as-parent cells are R1 rejections.

  @R2 @now
  Scenario Outline: Same-variant self-nesting is forbidden at any composer
    Given a complex data kind of variant "<parent>" containing a child of variant "<parent>"
    When I parse it
    Then parsing fails with parse.illegal-self-nesting

    Examples:
      | parent   |
      | unionset |
      | grainset |
      | joinset  |

  @R1 @now
  Scenario Outline: A leaf (dataset) never contains children
    Given a dataset authored with a "<child_container>" child container
    When I parse it
    Then parsing fails with parse.unknown-field

    Examples:
      | child_container |
      | datasets        |
      | unionsets       |
      | grainsets       |
      | joinsets        |

  @now
  Scenario Outline: The matrix admits the legal parent/child combinations
    Given a complex data kind of variant "<parent>" containing a child of variant "<child>"
    When I parse it
    Then it parses successfully

    Examples:
      | parent   | child    |
      | grainset | dataset  |
      | grainset | unionset |
      | grainset | joinset  |
      | unionset | dataset  |
      | unionset | grainset |
      | unionset | joinset  |
      | joinset  | dataset  |
      | joinset  | grainset |
      | joinset  | unionset |

  # --- R3 — composers require at least two children -----------------

  @R3 @now
  Scenario Outline: A complex data kind with fewer than two children is rejected
    Given a "<variant>" composer with <child_count> child(ren)
    When I validate it
    Then validation rejects it with validate.complex-data-kind-insufficient-children

    Examples:
      | variant  | child_count |
      | unionset | 0           |
      | unionset | 1           |
      | grainset | 0           |
      | grainset | 1           |
      | joinset  | 1           |

  @R3 @now
  Scenario: Joinset relationships do not count toward the child minimum
    Given the model:
      """
      semantic_model:
        name: m
        dimensions: [{ name: d, data_type: string, type: categorical }]
        joinsets:
          - name: trivial
            dimensions: [{ ref: d }]
            datasets:
              - name: orders
            relationships: []
      """
    When I validate it
    Then validation rejects it with validate.complex-data-kind-insufficient-children

  # --- Nested-form structural-only rule ------------------------------

  @now
  Scenario Outline: A nested data kind carries no SemanticInterface
    Given a nested dataset that authors a "<interface_field>" block
    When I parse it
    Then parsing fails with parse.nested-data-kind-carries-interface

    Examples:
      | interface_field |
      | dimensions      |
      | measures        |
      | metrics         |
      | keys            |
      | filters         |
      | description     |
      | ai_context      |

  # --- Grainset structural rules ------------------------------

  @now
  Scenario: A grainset needs at least two levels
    Given a grainset with a single level
    When I validate it
    Then validation rejects it with validate.grainset-must-have-multiple-levels

  @now
  Scenario: Grainset levels must be declared coarsest-first
    Given a grainset whose levels are ordered day then month
    When I validate it
    Then validation rejects it with validate.grainset-levels-not-sorted-coarsest-first

  @now
  Scenario: Every grainset leaf child must author its own temporal grain
    Given a grainset leaf child that omits extras.temporal.grain
    When I validate it
    Then validation rejects it with validate.grainset-child-grain-required

  @now
  Scenario: TemporalShape grain is forbidden on a complex data kind
    Given a unionset that authors extras.temporal.grain
    When I validate it
    Then validation rejects it with validate.temporal-grain-on-complex

  # --- Joinset binary-v1 arity -------------------------------------

  @now
  Scenario Outline: A v1 joinset must have exactly two members
    Given a joinset with <member_count> members
    When I validate it
    Then validation rejects it with validate.joinset-must-be-binary-v1

    Examples:
      | member_count |
      | 1            |
      | 3            |

  # --- Depth bound -------------------------------------------------

  @now
  Scenario: A legal three-deep composer chain over a leaf parses
    Given a joinset wrapping a grainset wrapping a unionset wrapping a dataset
    When I parse it
    Then it parses successfully

  @R2 @now
  Scenario: Same-kind nesting is rejected even when buried two levels deep
    Given a unionset containing a grainset that contains a grainset
    When I parse it
    Then parsing fails with parse.illegal-self-nesting

  # --- The reference model is structurally legal (regression anchor) ---------

  @now
  Scenario: The canonical reference model passes all structural gates
    Given the reference model
    When I validate it
    Then it validates successfully
