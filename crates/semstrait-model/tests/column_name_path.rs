mod common;

use common::{built, expression, refused};
use semstrait_model::{ParseErrorKind, SemanticMapping, SemanticMappingValue, SemanticModel};
use semstrait_sql::expr::{Expr, NamedReference};

fn sourced_model(expr: &str, mapping: &str) -> String {
    format!(
        r#"
semantic_model:
  name: columns
  dimensions:
    - name: plain
      data_type: string
      type: categorical
    - name: written
      data_type: string
      type: categorical
      expr: {expr}
  datasets:
    - name: orders
      extras:
        storage:
          tables:
            - warehouse.fact_rows
        semantic_mapping: {mapping}
      dimensions:
        - ref: plain
        - ref: written
"#
    )
}

fn refusal(yaml: &str) -> String {
    match &refused(yaml)[..] {
        [ParseErrorKind::YamlSyntax { message }] => message.clone(),
        other => panic!("one refusal of the YAML's value:\n{yaml}\n{other:?}"),
    }
}

#[test]
fn parse_refuses_an_empty_name_and_names_its_field() {
    for (expr, mapping, field) in [
        (r#"{col: ""}"#, "auto", "col"),
        (r#"{upper: {col: ""}}"#, "auto", "col"),
        (r#"{concat: [plain, {col: ""}]}"#, "auto", "col"),
        ("plain", r#"{plain: {expr: {col: ""}}}"#, "col"),
        (r#"{field: ""}"#, "auto", "field"),
        (r#"{dim: ""}"#, "auto", "dim"),
        (r#"{dim: {name: ""}}"#, "auto", "dim"),
        (r#"{measure: ""}"#, "auto", "measure"),
        (
            r#"{metric: {name: "", accessor: previous}}"#,
            "auto",
            "metric",
        ),
    ] {
        let model = sourced_model(expr, mapping);
        let refusal = refusal(&model);
        assert!(
            refusal.contains(&format!(
                "invalid value for field `{field}`: a name cannot be empty"
            )),
            "{model}\n{refusal}"
        );
    }
}

#[test]
fn parse_refuses_an_empty_mapped_column_with_the_advice_of_the_empty_string_literal() {
    let model = sourced_model("plain", r#"{plain: ""}"#);
    let refusal = refusal(&model);
    assert!(
        refusal.contains(
            r#"invalid value for field `semantic_mapping`: a column name cannot be empty; the empty string is written `{lit: ""}`"#
        ),
        "{model}\n{refusal}"
    );
}

#[test]
fn the_empty_string_literal_maps_where_the_empty_column_is_refused() {
    let model = built(&sourced_model("plain", r#"{plain: {lit: ""}}"#));
    let dataset = model.datasets.values().next().expect("one Dataset");
    let SemanticMapping::Explicit(entries) = &dataset.body.base.extras.semantic_mapping else {
        panic!("the Dataset maps `plain`");
    };
    assert!(
        matches!(
            entries.values().next(),
            Some(SemanticMappingValue::Untyped(_))
        ),
        "{entries:?}"
    );
}

fn model(expr: &str, mapped: &str) -> SemanticModel {
    built(&format!(
        r#"
semantic_model:
  name: columns
  dimensions:
    - name: written
      data_type: string
      type: categorical
      expr: {expr}
  datasets:
    - name: orders
      extras:
        semantic_mapping:
          written: {{expr: {mapped}}}
      dimensions:
        - ref: written
"#
    ))
}

fn mapped(model: &SemanticModel) -> &SemanticMappingValue {
    let dataset = model.datasets.values().next().expect("one Dataset");
    let SemanticMapping::Explicit(entries) = &dataset.body.base.extras.semantic_mapping else {
        panic!("the Dataset maps its Dimension");
    };
    entries.values().next().expect("one entry")
}

fn one_part(name: &str) -> NamedReference {
    NamedReference::new([name]).expect("one part")
}

#[test]
fn an_authored_column_parses_as_a_one_part_name_path() {
    for (expr, mapping, name) in [
        ("{col: amount}", "amount", "amount"),
        ("{col: a.b}", "{col: a.b}", "a.b"),
    ] {
        let model = model(expr, mapping);
        assert_eq!(
            expression(&model),
            Expr::Column(one_part(name)),
            "`{expr}` in a semantic expression"
        );
        assert_eq!(
            mapped(&model),
            &SemanticMappingValue::Expr(Expr::Column(one_part(name))),
            "`{mapping}` in a mapping's `expr:` arm"
        );
    }
}

#[test]
fn the_writer_writes_a_column_as_its_name_and_refuses_a_qualified_one() {
    let mut model = model("{col: a.b}", "{col: a.b}");
    let written = serde_yaml::to_string(&model).expect("the writer writes the model");
    assert!(written.contains("col: a.b"), "{written}");
    let from_yaml: SemanticModel = serde_yaml::from_str(&written).expect("the YAML reads back");
    assert_eq!(from_yaml, model);
    let json = serde_json::to_string(&model).expect("the writer writes JSON");
    let from_json: SemanticModel = serde_json::from_str(&json).expect("the JSON reads back");
    assert_eq!(from_json, model);

    let dimension = model.dimensions.values_mut().next().expect("one Dimension");
    dimension.body.body.expr = Some(Expr::Column(
        NamedReference::new(["ps0", "amount"]).expect("two parts"),
    ));
    for written in [
        serde_yaml::to_string(&model).map_err(|error| error.to_string()),
        serde_json::to_string(&model).map_err(|error| error.to_string()),
    ] {
        assert!(
            written
                .as_ref()
                .is_err_and(|message| message.contains("not representable")),
            "{written:?}"
        );
    }
}
