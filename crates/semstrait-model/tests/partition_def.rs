mod common;

use common::built;
use common::golden::{self, Golden};
use semstrait_model::{parse, ParseErrorKind, PartitionDef};

fn sales(order_year_type: &str, storage: &str) -> String {
    format!(
        r#"
semantic_model:
  name: sales
  dimensions:
    - name: order_year
      data_type: integer
      type: {order_year_type}
  datasets:
    - name: sales
      extras:
        storage:
          format: parquet
          paths: [s3://lake/orders/]
          {storage}
        semantic_mapping:
          order_year: year
      dimensions:
        - ref: order_year
"#
    )
}

#[test]
fn partition_def_is_a_list_of_entries_each_naming_a_dimension() {
    let model = built(&sales(
        "categorical",
        "partition_def:\n            - { type: range, dimension: order_year }\n            \
         - { type: list, dimension: order_region, values: [eu, us] }",
    ));
    let dataset = model.datasets.values().next().expect("one Dataset");
    let storage = (dataset.body.base.extras.storage.as_ref()).expect("a storage block");
    assert_eq!(
        storage.partition_def,
        [
            PartitionDef::Range {
                dimension: "order_year".to_owned(),
            },
            PartitionDef::List {
                dimension: "order_region".to_owned(),
                values: vec!["eu".to_owned(), "us".to_owned()],
            },
        ]
    );
}

fn the_retired_partition_form() -> Golden {
    let yaml = sales(
        "{ metadata: { source: { partition: { column: year } } } }",
        "",
    );
    let diagnostics = parse(&yaml).expect_err("parse refuses the retired form");
    assert!(
        matches!(&diagnostics[..], [diagnostic] if matches!(&diagnostic.kind,
            ParseErrorKind::YamlSyntax { message }
                if message.contains("MetadataSource: unknown variant `partition`"))),
        "{diagnostics:?}"
    );
    Golden::message(
        "the_retired_partition_form_is_refused_at_parse",
        &diagnostics,
    )
}

#[test]
fn the_retired_partition_form_is_refused_at_parse() {
    the_retired_partition_form().assert_committed();
}

#[test]
fn generate_goldens() {
    golden::generate(|| vec![the_retired_partition_form()]);
}
