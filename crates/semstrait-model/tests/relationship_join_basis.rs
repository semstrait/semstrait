mod common;

use common::golden::{self, Golden};
use semstrait_common::diagnostic::Diagnose;
use semstrait_model::{
    parse, Cardinality, JoinKeySpec, ModelBuildErrorKind, ParseErrorKind, Relationship,
    SemanticModel, ValidateErrorKind,
};

fn model_yaml(on: &str) -> String {
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
      cardinality: many_to_one
{on}
"#
    )
}

fn parse_model(on: &str) -> SemanticModel {
    let (model, _warnings) = parse(&model_yaml(on))
        .expect("model parses")
        .build()
        .expect("model builds");
    model
}

fn relationship(model: &SemanticModel) -> &Relationship {
    model
        .relationships
        .values()
        .next()
        .expect("the one relationship")
}

fn parse_error(on: &str) -> Vec<ParseErrorKind> {
    parse(&model_yaml(on))
        .expect_err("the basis is rejected")
        .iter()
        .map(|d| d.kind.clone())
        .collect()
}

fn message_of(kinds: &[ParseErrorKind]) -> String {
    kinds
        .iter()
        .map(|k| k.message())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_join_basis_state_round_trips() {
    let cases: [(&str, JoinKeySpec); 4] = [
        ("", JoinKeySpec::Auto),
        ("      on: auto", JoinKeySpec::Auto),
        ("      on: cross", JoinKeySpec::Cross),
        (
            "      on: [region_id, customer_id]",
            JoinKeySpec::Declared(vec!["region_id".into(), "customer_id".into()]),
        ),
    ];
    for (on, expected) in cases {
        let model = parse_model(on);
        assert_eq!(
            relationship(&model).on,
            expected,
            "authored `{on}` did not parse to the expected basis"
        );
        let dumped = serde_yaml::to_string(&model).expect("serialize");
        assert_eq!(
            dumped.contains("on:"),
            expected != JoinKeySpec::Auto,
            "authored `{on}` emitted the wrong `on:` presence:\n{dumped}"
        );
        assert!(
            !dumped.contains("keys:"),
            "authored `{on}` emitted a `keys:` key:\n{dumped}"
        );
        let back: SemanticModel = serde_yaml::from_str(&dumped).expect("deserialize");
        assert_eq!(model, back, "authored `{on}` did not survive a round-trip");
        assert!(
            !dumped.contains('!'),
            "YAML dump leaked a `!`-tag:\n{dumped}"
        );
    }
}

#[test]
fn the_retired_keys_spelling_is_refused() {
    let kinds = parse_error("      keys: [customer_id]");
    assert!(
        kinds.iter().any(|k| matches!(
            k,
            ParseErrorKind::UnknownField { field, .. } if field == "keys"
        )),
        "expected UnknownField for `keys`, got {kinds:?}"
    );
}

#[test]
fn cross_is_accepted_with_and_without_a_filter() {
    for on in [
        "      on: cross",
        "      on: cross\n      filter:\n        gt:\n          - order_ts\n          - valid_from",
    ] {
        let model = parse_model(on);
        assert_eq!(relationship(&model).on, JoinKeySpec::Cross);
    }
    let with_filter = parse_model(
        "      on: cross\n      filter:\n        gt:\n          - order_ts\n          - valid_from",
    );
    assert!(
        relationship(&with_filter).filter.is_some(),
        "`cross` must permit `filter:`"
    );
}

#[test]
fn an_empty_list_is_rejected_and_the_diagnostic_names_cross() {
    let kinds = parse_error("      on: []");
    assert!(
        kinds.contains(&ParseErrorKind::RelationshipJoinKeysEmpty),
        "expected RelationshipJoinKeysEmpty, got {kinds:?}"
    );
    assert!(kinds.iter().any(|k| k.message().contains("cross")));
    let message = message_of(&kinds);
    assert!(
        message.contains("`on:") && !message.contains("keys:"),
        "the diagnostic must name `on:` and never `keys:`: {message}"
    );
}

#[test]
fn an_unknown_string_is_rejected_and_the_diagnostic_names_both_spellings() {
    unknown_spelling_refusal().assert_committed();
}

fn unknown_spelling_refusal() -> Golden {
    let diagnostics = parse(&model_yaml("      on: crossed")).expect_err("the basis is rejected");
    let kinds: Vec<ParseErrorKind> = diagnostics.iter().map(|d| d.kind.clone()).collect();
    assert!(
        kinds.iter().any(|k| matches!(
            k,
            ParseErrorKind::RelationshipJoinKeysUnknownSpelling { spelling } if spelling == "crossed"
        )),
        "expected RelationshipJoinKeysUnknownSpelling, got {kinds:?}"
    );
    let message = message_of(&kinds);
    assert!(message.contains("auto") && message.contains("cross"));
    assert!(
        message.contains("`on:") && !message.contains("keys:"),
        "the diagnostic must name `on:` and never `keys:`: {message}"
    );
    Golden::message(
        "an_unknown_string_is_rejected_and_the_diagnostic_names_both_spellings",
        &diagnostics,
    )
}

#[test]
fn a_repeated_name_is_rejected() {
    let kinds = parse_error("      on: [customer_id, customer_id]");
    assert!(
        kinds.iter().any(|k| matches!(
            k,
            ParseErrorKind::RelationshipJoinKeyDuplicated { key } if key == "customer_id"
        )),
        "expected RelationshipJoinKeyDuplicated, got {kinds:?}"
    );
    let message = message_of(&kinds);
    assert!(
        message.contains("`on:") && !message.contains("keys:"),
        "the diagnostic must name `on:` and never `keys:`: {message}"
    );
}

#[test]
fn a_non_string_entry_is_rejected() {
    let kinds = parse_error("      on:\n        - from: customer_id\n          to: id");
    assert!(!kinds.is_empty(), "the object form must not parse");
    let message = message_of(&kinds);
    assert!(
        message.contains("on:") && !message.contains("keys:"),
        "the diagnostic must name `on:` and never `keys:`: {message}"
    );
}

#[test]
fn a_legal_basis_raises_none_of_the_three() {
    let model = parse_model("      on: [customer_id]");
    assert_eq!(
        relationship(&model).on,
        JoinKeySpec::Declared(vec!["customer_id".into()])
    );
}

#[test]
fn the_builder_reaches_every_state() {
    let base = || {
        Relationship::builder()
            .name("orders_to_customers")
            .from("orders")
            .to("customers")
            .cardinality(Cardinality::ManyToOne)
    };
    assert_eq!(base().build().expect("builds").on, JoinKeySpec::Auto);
    assert_eq!(
        base().on_auto().build().expect("builds").on,
        JoinKeySpec::Auto
    );
    assert_eq!(
        base().on_cross().build().expect("builds").on,
        JoinKeySpec::Cross
    );
    assert_eq!(
        base()
            .on(["region_id", "customer_id"])
            .build()
            .expect("builds")
            .on,
        JoinKeySpec::Declared(vec!["region_id".into(), "customer_id".into()])
    );
    assert_eq!(
        base()
            .on_cross()
            .on_key("region_id")
            .on_key("customer_id")
            .build()
            .expect("builds")
            .on,
        JoinKeySpec::Declared(vec!["region_id".into(), "customer_id".into()])
    );
}

#[test]
fn the_builder_refuses_both_declared_basis_rejections() {
    let refused = |names: Vec<&str>| {
        Relationship::builder()
            .name("orders_to_customers")
            .from("orders")
            .to("customers")
            .cardinality(Cardinality::ManyToOne)
            .on(names)
            .build()
            .expect_err("the declared basis is not legal")
            .kind
    };
    assert_eq!(
        refused(vec![]),
        ModelBuildErrorKind::Validate(ValidateErrorKind::RelationshipJoinBasisEmpty {
            relationship: "orders_to_customers".to_string(),
        })
    );
    assert!(refused(vec![]).message().contains("cross"));
    assert!(refused(vec![]).message().contains("on_cross()"));
    assert_eq!(
        refused(vec!["customer_id", "customer_id"]),
        ModelBuildErrorKind::Validate(ValidateErrorKind::RelationshipJoinKeyDuplicated {
            relationship: "orders_to_customers".to_string(),
            key: "customer_id".to_string(),
        })
    );
}

#[test]
fn the_builder_refuses_an_omitted_cardinality_under_a_non_cross_basis() {
    let base = || {
        Relationship::builder()
            .name("orders_to_customers")
            .from("orders")
            .to("customers")
    };
    let expected =
        ModelBuildErrorKind::Validate(ValidateErrorKind::RelationshipMissingCardinality {
            relationship: "orders_to_customers".to_string(),
        });
    for refused in [
        base().on(["customer_id"]).build(),
        base().on_auto().build(),
        base().build(),
    ] {
        assert_eq!(
            refused
                .expect_err("an omitted cardinality is not legal here")
                .kind,
            expected
        );
    }
    assert_eq!(
        base()
            .on_cross()
            .build()
            .expect("a bare `on_cross()` relationship builds")
            .cardinality,
        None
    );
}

#[test]
fn generate_goldens() {
    golden::generate(|| vec![unknown_spelling_refusal()]);
}
