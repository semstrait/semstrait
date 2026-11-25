# Pillar 3 — Resolved vs Unresolved Data Structure
#
# Contract gates for the authoring <-> resolved expression boundary — the heart
# of the public/internal expression split. Source of truth:
# ADR 0002 (docs/adr/0002-single-leaf-parameterized-expression-tree.md, the
# expression families) and ADR 0089 (the compile fold). Each Then gates one
# clause of them.
#
# Two clocks: @now gates exercise parse/validate/serialize (buildable today);
# @target gates assert the post-item-Y physical-only fold the manifest does not
# yet fully produce.

@resolution
Feature: Resolved vs unresolved expression form

  # --- Friendly-tag authoring serde ---------------------------

  @now
  Scenario: Authored YAML never leaks internal serde discriminants
    Given the reference model
    When I parse it and serialize it back to YAML
    Then the YAML contains no "binary_op:" key
    And the YAML contains no "unary_op:" key
    And the YAML contains no "type:" expression-node tag
    And the YAML contains no "content:" expression-node tag

  @now
  Scenario: The generated JSON schema uses friendly tags only
    When I generate the semantic-model JSON schema
    Then the schema contains no "binary_op", "unary_op", "type", or "content" expression discriminant

  # --- Round-trip normal form -------------------------

  @now
  Scenario: Authoring round-trips to a fixpoint normal form
    Given the reference model
    When I parse, dump, and parse it again
    Then the two parsed models are equal

  @now
  Scenario Outline: Sugar normalizes to its canonical friendly tag
    Given an inline expression authored as "<sugar>"
    When I parse and dump it
    Then the dumped expression uses the "<canonical>" tag

    Examples:
      | sugar                | canonical             |
      | if                   | case                  |
      | count_distinct       | count (distinct:true) |
      | not                  | not                   |

  # --- Two expression families ----------------------------
  # These assert the type-level shape of the families. They are @target because
  # the families are mid-relocation; once landed they pin the
  # public-vs-internal contract.

  @target
  Scenario: The authoring expression family has no compile-only nodes
    Then model::Expr has no Window variant
    And model::UnresolvedPhysicalLeaf is exactly { Column, Literal }
    And model::UnresolvedSemanticLeaf carries typed leaves { Field, Dimension, Measure, Metric, Key }
    And neither authoring leaf set carries Parameter

  @target
  Scenario: The resolved expression family adds the compile-only nodes
    Then ir::PhysicalExpr has the Window variant
    And ir::PhysicalLeaf is exactly { Column, Literal, Parameter }
    And semstrait-ir defines no SemanticExpr type

  # --- The compile fold ----------------------------------------------

  @target
  Scenario: Compile lowers every authoring expression to the physical form
    Given the reference model
    When I compile it
    Then no ExprSource survives anywhere in the manifest
    And no semantic leaf (Field/Dimension/Measure/Metric/Key) survives in any pooled expression
    And every pooled expression is an ir::PhysicalExpr modulo Parameter

  @target
  Scenario: Sugar accessors lower to a Window + Parameter subtree
    Given a measure ref that uses the .previous() accessor
    When I compile it
    Then the resolved expression is a Window node
    And its partition_by/order_by carry Parameter leaves

  @target
  Scenario: A computed dimension lowers to its function tree, not a bare column
    Given a computed dimension authored as "upper(region)"
    When I compile it
    Then the resolved expression is a FunctionCall over a Column leaf
    And it is not stored as a bare Column reference

  @target
  Scenario: An unknown function is rejected at compile against the registry
    Given a computed dimension authored as "no_such_fn(x)"
    When I compile it
    Then compilation fails with compile.unknown-function

  # --- Name resolution at compile ---------------------

  @target
  Scenario: Semantic names are resolved to EntityIds at compile, not at plan time
    Given the reference model
    When I compile it
    Then no SemanticsName survives inside any resolved PhysicalExpr
    And every semantic reference is carried as an EntityId

  @target
  Scenario: An unresolved semantic reference fails at compile
    Given a metric whose expr references a missing measure
    When I compile it
    Then compilation fails with compile.unknown-reference

  # --- ExprLayer classification -------------------------

  @target
  Scenario Outline: classify assigns the layer purely from tree structure
    Given a resolved expression that is "<shape>"
    Then its ExprLayer is "<layer>"

    Examples:
      | shape                                  | layer         |
      | a scalar with no aggregate             | Scalar        |
      | an aggregate at the root               | Aggregate     |
      | a scalar wrapping an aggregate         | PostAggregate |
