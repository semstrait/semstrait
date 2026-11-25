mod common;

use common::agg;
use common::golden::{self, Golden};
use semstrait_common::Granularity;
use semstrait_model::{
    parse, Dataset, Dimension, DimensionSite, DimensionType, LeafExtras, Measure, MeasureSite,
    ModelBuildErrorKind, Optional, ParseErrorKind, Relationship, SemanticInterface,
    SemanticMapping, SemanticModel, TemporalShape, ValidateErrorKind,
};
use semstrait_sql::datatype::DataType;

#[test]
fn parse_emits_invalid_identifier_when_dataset_name_starts_with_digit() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: 1bogus
      extras:
        semantic_mapping: auto
"#;
    let err = parse(yaml).unwrap_err();
    assert!(err.iter().any(|d| matches!(
        &d.kind,
        ParseErrorKind::InvalidIdentifier { raw, .. } if raw == "1bogus"
    )));
}

#[test]
fn parse_emits_invalid_identifier_when_inline_dimension_name_has_dash() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      dimensions:
        - name: bad-dim
          data_type: string
          type: categorical
"#;
    let err = parse(yaml).unwrap_err();
    assert!(err.iter().any(|d| matches!(
        &d.kind,
        ParseErrorKind::InvalidIdentifier { raw, .. } if raw == "bad-dim"
    )));
}

#[test]
fn parse_does_not_emit_invalid_identifier_when_names_match_grammar() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      dimensions:
        - name: country_code
          data_type: string
          type: categorical
"#;
    let _builder = parse(yaml).expect("parse ok");
}

fn relationship_model_yaml(body: &str) -> String {
    format!(
        r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
    - name: customers
      extras:
        semantic_mapping: auto
  relationships:
    - name: orders_to_customers
      from: orders
      to: customers
{body}
"#
    )
}

fn build_errors(yaml: &str) -> Vec<ModelBuildErrorKind> {
    parse(yaml)
        .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
        .build()
        .err()
        .unwrap_or_else(|| panic!("build must fail for:\n{yaml}"))
        .into_iter()
        .map(|d| d.kind)
        .collect()
}

fn names_missing_cardinality(kinds: &[ModelBuildErrorKind], named: &str) -> bool {
    kinds.iter().any(|k| {
        matches!(
            k,
            ModelBuildErrorKind::Validate(ValidateErrorKind::RelationshipMissingCardinality {
                relationship
            }) if relationship == named
        )
    })
}

#[test]
fn validate_emits_relationship_missing_cardinality_when_omitted_under_a_declared_basis() {
    let kinds = build_errors(&relationship_model_yaml("      on: [customer_id]"));
    assert!(
        names_missing_cardinality(&kinds, "orders_to_customers"),
        "expected RelationshipMissingCardinality naming the relationship, got {kinds:?}"
    );
}

#[test]
fn validate_emits_relationship_missing_cardinality_when_omitted_under_a_derived_basis() {
    for basis in ["      on: auto", ""] {
        let kinds = build_errors(&relationship_model_yaml(basis));
        assert!(
            names_missing_cardinality(&kinds, "orders_to_customers"),
            "authored basis `{basis}` must still owe a cardinality, got {kinds:?}"
        );
    }
}

#[test]
fn validate_emits_relationship_missing_cardinality_for_a_joinset_local_relationship() {
    let yaml = r#"
semantic_model:
  name: ana
  joinsets:
    - name: orders_with_customers
      extras: {}
      datasets:
        - name: orders_l
          extras:
            semantic_mapping: auto
        - name: customers_r
          extras:
            semantic_mapping: auto
      relationships:
        - name: orders_l_to_customers_r
          from: orders_l
          to: customers_r
          on: [customer_id]
"#;
    let kinds = build_errors(yaml);
    assert!(
        names_missing_cardinality(&kinds, "orders_l_to_customers_r"),
        "expected RelationshipMissingCardinality naming the joinset-local relationship, \
         got {kinds:?}"
    );
}

#[test]
fn validate_does_not_emit_relationship_missing_cardinality_when_the_basis_is_cross() {
    let (model, warnings) = parse(&relationship_model_yaml("      on: cross"))
        .expect("parse ok")
        .build()
        .expect("a bare `on: cross` relationship validates clean");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    let dumped = serde_yaml::to_string(&model).expect("serialize");
    assert!(
        !dumped.contains("cardinality:"),
        "a bare `on: cross` relationship emitted a `cardinality:` key:\n{dumped}"
    );
}

#[test]
fn validate_does_not_emit_relationship_missing_cardinality_when_cardinality_present() {
    for basis in [
        "      on: [customer_id]",
        "      on: auto",
        "      on: cross",
        "",
    ] {
        let yaml = relationship_model_yaml(&format!("      cardinality: many_to_one\n{basis}"));
        let (model, _warnings) = parse(&yaml)
            .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
            .build()
            .unwrap_or_else(|d| panic!("authored basis `{basis}` must validate clean, got {d:?}"));
        let dumped = serde_yaml::to_string(&model).expect("serialize");
        assert!(
            dumped.contains("cardinality: many_to_one"),
            "authored basis `{basis}` dropped the stated cardinality:\n{dumped}"
        );
    }
}

#[test]
fn a_preserved_side_beside_a_cross_basis_is_refused() {
    preserved_side_refusal().assert_committed();
}

fn preserved_side_refusal() -> Golden {
    let diagnostics = parse(&relationship_model_yaml(
        "      on: cross\n      optional: left",
    ))
    .expect("parse ok")
    .build()
    .expect_err("a preserved side beside a cross basis is refused");
    let kinds: Vec<&ModelBuildErrorKind> = diagnostics.iter().map(|d| &d.kind).collect();
    assert_eq!(
        kinds,
        [&ModelBuildErrorKind::Validate(
            ValidateErrorKind::RelationshipCrossWithPreservedSide {
                relationship: "orders_to_customers".to_owned(),
                optional: Optional::Left,
            }
        )]
    );
    Golden::message(
        "a_preserved_side_beside_a_cross_basis_is_refused",
        &diagnostics,
    )
}

#[test]
fn optional_none_beside_a_cross_basis_and_a_symmetric_cardinality_builds_clean() {
    let yaml = relationship_model_yaml(
        "      on: cross\n      cardinality: many_to_many\n      optional: none\n      cross_filter: both",
    );
    let (_model, warnings) = parse(&yaml)
        .expect("parse ok")
        .build()
        .unwrap_or_else(|d| panic!("`optional: none` beside `on: cross` validates clean: {d:?}"));
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[test]
fn the_builder_refuses_a_preserved_side_beside_a_cross_basis() {
    for optional in [Optional::Left, Optional::Right, Optional::Both] {
        let refused = Relationship::builder()
            .name("orders_to_customers")
            .from("orders")
            .to("customers")
            .on_cross()
            .optional(optional)
            .build()
            .expect_err("a preserved side beside a cross basis is refused")
            .kind;
        assert_eq!(
            refused,
            ModelBuildErrorKind::Validate(ValidateErrorKind::RelationshipCrossWithPreservedSide {
                relationship: "orders_to_customers".to_owned(),
                optional,
            })
        );
    }
}

const WEB_JOINSET: &str = "\
- name: web
  datasets:
    - name: sessions
      extras:
        semantic_mapping: {day: day, amount: amount}
        temporal: {timeseries: {occurred_at: day}, granularity: day}
    - name: budget
      extras: {semantic_mapping: {tag: tag}}
  relationships:";

fn web_joinset(relationships: &[&str], indent: usize) -> String {
    let margin = " ".repeat(indent);
    let entries = relationships.iter().map(|r| format!("    - {r}"));
    WEB_JOINSET
        .lines()
        .map(str::to_owned)
        .chain(entries)
        .map(|line| format!("{margin}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn nested_joinset_model(body: &str) -> String {
    format!(
        r#"
semantic_model:
  name: m
  dimensions:
    - {{name: day, data_type: date, type: {{temporal: {{granularities: [day, month]}}}}}}
    - {{name: tag, data_type: string, type: categorical}}
  measures:
    - {{name: amount, data_type: bigint, agg: sum}}
{body}
"#
    )
}

fn nested_joinset_models(relationships: &[&str]) -> [(&'static str, String); 4] {
    let shallow = web_joinset(relationships, 8);
    let deep = web_joinset(relationships, 12);
    [
        (
            "Unionset branch",
            nested_joinset_model(&format!(
                r#"  unionsets:
    - name: orders
      mode: all
      dimensions: [{{ref: day}}, {{ref: tag}}]
      measures: [{{ref: amount}}]
      datasets:
        - name: app
          extras: {{semantic_mapping: {{day: day, amount: amount}}}}
      joinsets:
{shallow}"#
            )),
        ),
        (
            "Grainset child",
            nested_joinset_model(&format!(
                r#"  grainsets:
    - name: g
      dimensions: [{{ref: day}}, {{ref: tag}}]
      measures: [{{ref: amount}}]
      datasets:
        - name: monthly
          extras:
            semantic_mapping: {{day: day, amount: amount}}
            temporal: {{timeseries: {{occurred_at: day}}, granularity: month}}
      joinsets:
{shallow}"#
            )),
        ),
        (
            "Joinset > Unionset > Joinset",
            nested_joinset_model(&format!(
                r#"  joinsets:
    - name: root
      dimensions: [{{ref: day}}, {{ref: tag}}]
      measures: [{{ref: amount}}]
      datasets:
        - name: tags
          extras: {{semantic_mapping: {{tag: tag}}}}
      unionsets:
        - name: orders
          mode: all
          datasets:
            - name: app
              extras: {{semantic_mapping: {{day: day, tag: tag, amount: amount}}}}
          joinsets:
{deep}
      relationships:
        - {{name: orders_tags, from: orders, to: tags, on: [tag], cardinality: many_to_one}}"#
            )),
        ),
        (
            "Unionset > Grainset > Joinset",
            nested_joinset_model(&format!(
                r#"  unionsets:
    - name: orders
      mode: all
      dimensions: [{{ref: day}}, {{ref: tag}}]
      measures: [{{ref: amount}}]
      datasets:
        - name: app
          extras: {{semantic_mapping: {{day: day, amount: amount}}}}
      grainsets:
        - name: g
          datasets:
            - name: monthly
              extras:
                semantic_mapping: {{day: day, amount: amount}}
                temporal: {{timeseries: {{occurred_at: day}}, granularity: month}}
          joinsets:
{deep}"#
            )),
        ),
    ]
}

const CROSS_LEFT: &str =
    "{name: sessions_budget, from: sessions, to: budget, on: cross, optional: left}";

fn cross_left_refusal() -> ModelBuildErrorKind {
    ModelBuildErrorKind::Validate(ValidateErrorKind::RelationshipCrossWithPreservedSide {
        relationship: "sessions_budget".to_owned(),
        optional: Optional::Left,
    })
}

#[test]
fn a_preserved_side_beside_a_cross_basis_is_refused_at_every_depth() {
    for (depth, yaml) in nested_joinset_models(&[CROSS_LEFT]) {
        assert_eq!(
            build_errors(&yaml),
            [cross_left_refusal()],
            "{depth}: the whole refusal"
        );
    }
}

#[test]
fn a_preserved_side_beside_a_cross_basis_in_a_unionset_branch_is_refused() {
    nested_preserved_side_refusal().assert_committed();
}

fn nested_preserved_side_refusal() -> Golden {
    let [(_, yaml), ..] = nested_joinset_models(&[CROSS_LEFT]);
    let diagnostics = parse(&yaml)
        .expect("parse ok")
        .build()
        .expect_err("a preserved side beside a cross basis in a Unionset branch is refused");
    let kinds: Vec<&ModelBuildErrorKind> = diagnostics.iter().map(|d| &d.kind).collect();
    assert_eq!(kinds, [&cross_left_refusal()]);
    Golden::message(
        "a_preserved_side_beside_a_cross_basis_in_a_unionset_branch_is_refused",
        &diagnostics,
    )
}

#[test]
fn optional_none_beside_a_cross_basis_in_a_nested_joinset_builds_clean() {
    for relationship in [
        "{name: sessions_budget, from: sessions, to: budget, on: cross, optional: none}",
        "{name: sessions_budget, from: sessions, to: budget, on: cross}",
    ] {
        for (depth, yaml) in nested_joinset_models(&[relationship]) {
            let (_model, warnings) = parse(&yaml)
                .expect("parse ok")
                .build()
                .unwrap_or_else(|d| panic!("{depth}: `{relationship}` validates clean: {d:?}"));
            assert!(
                warnings.is_empty(),
                "{depth}: unexpected warnings: {warnings:?}"
            );
        }
    }
}

#[test]
fn each_per_list_relationship_rule_reaches_a_nested_joinset() {
    let symmetric_incomplete = |missing: &str| {
        ModelBuildErrorKind::Validate(
            ValidateErrorKind::RelationshipSymmetricCardinalityIncomplete {
                relationship: "sessions_budget".to_owned(),
                missing: missing.to_owned(),
            },
        )
    };
    let rules: [(&[&str], Vec<ModelBuildErrorKind>); 4] = [
        (
            &["{name: sessions_budget, from: sessions, to: budget}"],
            vec![ModelBuildErrorKind::Validate(
                ValidateErrorKind::RelationshipMissingCardinality {
                    relationship: "sessions_budget".to_owned(),
                },
            )],
        ),
        (
            &["{name: sessions_budget, from: sessions, to: budget, cardinality: one_to_one}"],
            vec![
                symmetric_incomplete("optional"),
                symmetric_incomplete("cross_filter"),
            ],
        ),
        (
            &[
                "{name: sessions_budget, from: sessions, to: budget, cardinality: many_to_many, \
                 optional: none, cross_filter: left}",
            ],
            vec![ModelBuildErrorKind::Validate(
                ValidateErrorKind::RelationshipManyToManyCrossFilterDirectional {
                    relationship: "sessions_budget".to_owned(),
                },
            )],
        ),
        (
            &[
                "{name: sessions_budget, from: sessions, to: budget, on: cross}",
                "{name: sessions_budget, from: sessions, to: budget, on: cross}",
            ],
            vec![ModelBuildErrorKind::Validate(
                ValidateErrorKind::DuplicateRelationshipName {
                    relationship: "sessions_budget".to_owned(),
                    scope: "web".to_owned(),
                    occurrences: 2,
                },
            )],
        ),
    ];
    for (relationships, refused) in rules {
        for (depth, yaml) in nested_joinset_models(relationships) {
            assert_eq!(
                build_errors(&yaml),
                refused,
                "{depth}: the whole refusal of {relationships:?}"
            );
        }
    }
}

#[test]
fn generate_goldens() {
    golden::generate(|| vec![preserved_side_refusal(), nested_preserved_side_refusal()]);
}

fn two_relationship_model_yaml(relationships: &str) -> String {
    format!(
        r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
    - name: customers
      extras:
        semantic_mapping: auto
  relationships:
{relationships}
"#
    )
}

fn names_duplicate_relationship(kinds: &[ModelBuildErrorKind], named: &str, scope: &str) -> bool {
    kinds.iter().any(|k| {
        matches!(
            k,
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateRelationshipName {
                relationship,
                scope: s,
                occurrences,
            }) if relationship == named && s == scope && *occurrences == 2
        )
    })
}

#[test]
fn validate_emits_duplicate_relationship_name_for_a_repeat_in_the_root_list() {
    let kinds = build_errors(&two_relationship_model_yaml(
        "    - name: orders_to_customers\n      from: orders\n      to: customers\n      \
         cardinality: many_to_one\n    - name: orders_to_customers\n      from: customers\n      \
         to: orders\n      cardinality: one_to_many",
    ));
    assert!(
        names_duplicate_relationship(&kinds, "orders_to_customers", "semantic_model"),
        "expected DuplicateRelationshipName naming the root scope, got {kinds:?}"
    );
}

#[test]
fn validate_emits_duplicate_relationship_name_for_a_repeat_in_one_joinset() {
    let yaml = r#"
semantic_model:
  name: ana
  joinsets:
    - name: orders_with_customers
      extras: {}
      datasets:
        - name: orders_l
          extras:
            semantic_mapping: auto
        - name: customers_r
          extras:
            semantic_mapping: auto
      relationships:
        - name: l_to_r
          from: orders_l
          to: customers_r
          on: [customer_id]
          cardinality: many_to_one
        - name: l_to_r
          from: customers_r
          to: orders_l
          on: [customer_id]
          cardinality: one_to_many
"#;
    let kinds = build_errors(yaml);
    assert!(
        names_duplicate_relationship(&kinds, "l_to_r", "orders_with_customers"),
        "expected DuplicateRelationshipName naming the joinset scope, got {kinds:?}"
    );
}

#[test]
fn validate_does_not_emit_duplicate_relationship_name_for_an_adr_0029_shadow() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
    - name: customers
      extras:
        semantic_mapping: auto
  relationships:
    - name: orders_to_customers
      from: orders
      to: customers
      cardinality: many_to_one
  joinsets:
    - name: orders_with_customers
      extras: {}
      datasets:
        - name: orders_l
          extras:
            semantic_mapping: auto
        - name: customers_r
          extras:
            semantic_mapping: auto
      relationships:
        - name: orders_to_customers
          from: orders_l
          to: customers_r
          on: [customer_id]
          cardinality: one_to_many
"#;
    let (_model, warnings) = parse(yaml)
        .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
        .build()
        .unwrap_or_else(|d| panic!("an ADR 0029 shadow must validate clean, got {d:?}"));
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[test]
fn validate_does_not_emit_duplicate_relationship_name_for_two_distinct_names() {
    let (_model, warnings) = parse(&two_relationship_model_yaml(
        "    - name: orders_to_customers\n      from: orders\n      to: customers\n      \
         cardinality: many_to_one\n    - name: customers_to_orders\n      from: customers\n      \
         to: orders\n      cardinality: one_to_many",
    ))
    .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
    .build()
    .unwrap_or_else(|d| panic!("two distinct names must validate clean, got {d:?}"));
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[test]
fn parse_emits_measure_missing_agg_when_agg_omitted() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      measures:
        - name: revenue
          data_type: integer
"#;
    let err = parse(yaml).unwrap_err();
    assert!(err
        .iter()
        .any(|d| matches!(d.kind, ParseErrorKind::MeasureMissingAgg { .. })));
}

#[test]
fn parse_does_not_emit_measure_missing_agg_when_agg_present() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      measures:
        - name: revenue
          data_type: integer
          agg: sum
"#;
    let _builder = parse(yaml).expect("parse ok");
}

#[test]
fn parse_emits_semantics_missing_data_type_when_dimension_lacks_data_type() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      dimensions:
        - name: country
          type: categorical
"#;
    let err = parse(yaml).unwrap_err();
    assert!(err
        .iter()
        .any(|d| matches!(d.kind, ParseErrorKind::SemanticsMissingDataType { .. })));
}

#[test]
fn parse_does_not_emit_semantics_missing_data_type_when_data_type_present() {
    let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      dimensions:
        - name: country
          data_type: string
          type: categorical
"#;
    let _builder = parse(yaml).expect("parse ok");
}

fn build_revenue_pool_measure() -> Measure {
    Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("sum"))
        .build()
}

fn build_order_ts_pool_dimension() -> Dimension {
    Dimension::builder("order_ts")
        .data_type(DataType::Timestamp { precision: 6 })
        .dim_type(DimensionType::temporal([Granularity::Day]))
        .build()
}

fn build_dataset(name: &str, interface: SemanticInterface) -> Dataset {
    let extras = LeafExtras::builder()
        .semantic_mapping(
            SemanticMapping::builder()
                .column("revenue", "amount_cents")
                .build(),
        )
        .temporal(TemporalShape::events("order_ts", Some(Granularity::Day)))
        .build();
    Dataset::builder(name)
        .extras(extras)
        .semantic_interface(interface)
        .build()
}

#[test]
fn validate_emits_semantics_shadow_root_pool_when_inline_collides_with_pool_name() {
    let revenue_inline = Measure::builder("revenue")
        .data_type(DataType::Decimal {
            precision: 18,
            scale: 2,
        })
        .agg(agg("max"))
        .build();

    let shadowing_iface = SemanticInterface::builder()
        .dimensions(vec![DimensionSite::r#ref("order_ts")])
        .measures(vec![MeasureSite::inline(revenue_inline)])
        .build();
    let referring_iface = SemanticInterface::builder()
        .dimensions(vec![DimensionSite::r#ref("order_ts")])
        .measures(vec![MeasureSite::r#ref("revenue")])
        .build();

    let (_m, diags) = SemanticModel::builder()
        .name("shadow-case")
        .dataset(build_dataset("shadowing", shadowing_iface))
        .dataset(build_dataset("referring", referring_iface))
        .dimension(build_order_ts_pool_dimension())
        .measure(build_revenue_pool_measure())
        .build()
        .expect("validate clean (warning-only)");

    assert!(diags.iter().any(|d| matches!(
        &d.kind,
        semstrait_model::ModelBuildErrorKind::Validate(
            ValidateErrorKind::SemanticsShadowRootPool { carrier, name }
        ) if carrier == "Measure" && name == "revenue"
    )));
}

#[test]
fn validate_does_not_emit_semantics_shadow_root_pool_when_ref_used_instead() {
    let iface = SemanticInterface::builder()
        .dimensions(vec![DimensionSite::r#ref("order_ts")])
        .measures(vec![MeasureSite::r#ref("revenue")])
        .build();

    let (_m, diags) = SemanticModel::builder()
        .name("no-shadow")
        .dataset(build_dataset("orders", iface))
        .dimension(build_order_ts_pool_dimension())
        .measure(build_revenue_pool_measure())
        .build()
        .expect("validate clean");

    assert!(!diags.iter().any(|d| matches!(
        d.kind,
        semstrait_model::ModelBuildErrorKind::Validate(
            ValidateErrorKind::SemanticsShadowRootPool { .. }
        )
    )));
}
