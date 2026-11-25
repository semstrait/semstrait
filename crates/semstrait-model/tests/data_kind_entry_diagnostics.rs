mod common;

use common::refused;
use semstrait_model::ParseErrorKind;

fn model(mapping: &str) -> String {
    format!(
        r#"
semantic_model:
  name: d
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        storage:
          tables:
            - warehouse.fact_rows
        semantic_mapping: {mapping}
      dimensions:
        - ref: region
    - name: [x]
      extras:
        semantic_mapping: {mapping}
"#
    )
}

fn assert_refused_at_the_second_dataset(yaml: &str) {
    let line = 1 + yaml
        .lines()
        .position(|text| text == "    - name: [x]")
        .expect("the second Dataset");
    let refusal = refused(yaml);
    assert!(
        matches!(&refusal[..], [ParseErrorKind::YamlSyntax { message }]
            if message.starts_with("semantic_model.datasets[1]: ")
                && message.contains(&format!("at line {line} column"))),
        "{refusal:?}"
    );
}

#[test]
fn a_yaml_error_in_a_second_dataset_names_its_index_and_its_line() {
    assert_refused_at_the_second_dataset(&model("auto"));
}

#[test]
fn a_yaml_error_after_an_integer_past_64_bits_names_its_index_and_its_line() {
    assert_refused_at_the_second_dataset(&model(
        "{region: {lit: {decimal: 99999999999999999999}}}",
    ));
}
