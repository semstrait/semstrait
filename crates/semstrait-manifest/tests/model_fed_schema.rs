mod common;

use common::{compile_model_on, decimal_18_2, model_from_yaml, seeded_fs};
use semstrait_manifest::SchemaProvenance;
use semstrait_sql::datatype::DataType;

const STORAGE: &str = "{format: parquet, paths: ['s3://lake/orders/']}";

async fn schema_of(body: &str) -> Vec<(String, Option<DataType>)> {
    let model = model_from_yaml(&format!(
        "semantic_model:\n  name: sales\n{}",
        body.replace("STORAGE", STORAGE)
    ));
    let fs = seeded_fs(["s3://lake/orders/part-0.parquet"]).await;
    let manifest = compile_model_on(model, &fs).await;
    let [source] = manifest.sources.values().collect::<Vec<_>>()[..] else {
        panic!("one physical source, got {:?}", manifest.sources.keys());
    };
    let SchemaProvenance::ModelFed { schema } = &source.schema_provenance else {
        panic!("no catalog, so the schema is model-fed");
    };
    (schema.columns.iter())
        .map(|column| (column.name.clone(), column.data_type.clone()))
        .collect()
}

#[tokio::test]
async fn a_date_and_a_timestamp_on_one_column_read_it_as_the_timestamp() {
    let schema = schema_of(
        "  dimensions:
    - {name: order_date, data_type: date, type: categorical}
    - {name: order_ts, data_type: timestamp, type: categorical}
  datasets:
    - name: orders_by_day
      extras: {storage: STORAGE, semantic_mapping: {order_date: created_at}}
      dimensions: [{ref: order_date}]
    - name: orders_by_time
      extras: {storage: STORAGE, semantic_mapping: {order_ts: created_at}}
      dimensions: [{ref: order_ts}]
",
    )
    .await;
    assert_eq!(
        schema,
        [(
            "created_at".to_owned(),
            Some(DataType::Timestamp { precision: 6 })
        )]
    );
}

#[tokio::test]
async fn a_string_dimension_and_a_count_on_one_column_read_it_as_the_string() {
    let schema = schema_of(
        "  dimensions:
    - {name: order_ref, data_type: string, type: categorical}
  measures:
    - {name: order_count, data_type: bigint, agg: count}
  datasets:
    - name: orders
      extras: {storage: STORAGE, semantic_mapping: {order_ref: order_id, order_count: order_id}}
      dimensions: [{ref: order_ref}]
      measures: [{ref: order_count}]
",
    )
    .await;
    assert_eq!(schema, [("order_id".to_owned(), Some(DataType::String))]);
}

#[tokio::test]
async fn a_bigint_and_a_string_with_no_dimension_read_as_the_first_in_name_order() {
    let schema = schema_of(
        "  measures:
    - {name: tag, data_type: string, agg: max}
    - {name: quantity, data_type: bigint, agg: sum}
  datasets:
    - name: orders
      extras: {storage: STORAGE, semantic_mapping: {tag: qty, quantity: qty}}
      measures: [{ref: tag}, {ref: quantity}]
",
    )
    .await;
    assert_eq!(schema, [("qty".to_owned(), Some(DataType::BigInt))]);
}

#[tokio::test]
async fn a_model_fed_column_holds_the_type_of_the_measure_that_maps_it() {
    let schema = schema_of(
        "  measures:
    - {name: revenue, data_type: 'decimal(18,2)', agg: sum}
  metrics:
    - name: amount_label
      data_type: string
      expr: {cast: {input: revenue, target: string}}
  datasets:
    - name: orders
      extras: {storage: STORAGE, semantic_mapping: auto}
      measures: [{ref: revenue}]
      metrics: [{ref: amount_label}]
",
    )
    .await;
    assert_eq!(schema, [("revenue".to_owned(), Some(decimal_18_2()))]);
}

#[tokio::test]
async fn a_column_that_only_an_expression_or_a_filter_reads_keeps_no_type() {
    let schema = schema_of(
        "  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: sales, data_type: double, agg: sum, expr: {multiply: [{col: price}, {col: qty}]}}
  datasets:
    - name: orders
      extras: {storage: STORAGE, semantic_mapping: auto}
      dimensions: [{ref: region}]
      measures: [{ref: sales}]
      filters:
        - {name: discounted, expr: {gt: [{col: discount}, 0]}}
",
    )
    .await;
    assert_eq!(
        schema,
        [
            ("discount".to_owned(), None),
            ("price".to_owned(), None),
            ("qty".to_owned(), None),
            ("region".to_owned(), Some(DataType::String)),
        ]
    );
}
