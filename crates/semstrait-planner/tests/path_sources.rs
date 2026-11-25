mod common;

use common::golden::{self, Golden};
use common::{
    aggregate_step, at_granularity, below_aggregate, call, column, fields, filtered, harness,
    names, output_names, over_objects, planned, projection, refusal, scan_of, select, stored, text,
    with_semantics, Catalog, EVENTS_OBJECT,
};
use semstrait_catalog::FileFormat;
use semstrait_common::Granularity;
use semstrait_ir::{
    FilesScan, PhysicalExpr, PlanNode, Reader, ScanNode, SchemaColumn, SemanticPlan, VirtualTable,
};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit};
use semstrait_planner::{LookedIn, PlanError};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Call, Expr, ExprOperationExt, Literal};

const SEMANTICS: &str = r#"
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
    - {name: order_count, data_type: bigint, agg: count}
"#;

fn orders_daily(paths: &str) -> String {
    format!(
        r#"
    - name: orders_daily
      extras:
        storage:
          format: parquet
          paths: {paths}
        semantic_mapping:
          region: region
          revenue: amount
          order_count: order_id
      dimensions:
        - ref: region
      measures:
        - ref: revenue
        - ref: order_count
"#
    )
}

const ORDERS_DAILY_FOLDER: &str = "['s3://lake/orders_daily/']";

const ORDERS_DAILY_OBJECTS: [&str; 2] = [
    "s3://lake/orders_daily/part-1.parquet",
    "s3://lake/orders_daily/part-0.parquet",
];

fn parquet(pattern: Option<&str>, base_path: &str, paths: &[&str]) -> Reader {
    let paths = paths.iter().map(|path| (*path).to_owned()).collect();
    let pattern = pattern.map(str::to_owned);
    FilesScan::new(Some(FileFormat::Parquet), pattern, base_path, paths).into()
}

fn nullable((name, data_type): (&str, DataType)) -> SchemaColumn {
    SchemaColumn {
        name: name.to_owned(),
        data_type,
        nullable: true,
    }
}

fn row_level(tree: PhysicalExpr) -> PhysicalExpr {
    let Expr::Function(Call::Aggregate(call, _)) = &tree else {
        return tree;
    };
    let [operand] = call.args() else {
        panic!("one operand, got {:?}", call.args());
    };
    operand.clone()
}

fn as_stored(
    manifest: &SemanticManifest,
    plan: &SemanticPlan,
    dataset: &str,
    names: &[&str],
) -> (Vec<PhysicalExpr>, Vec<PhysicalExpr>) {
    let trees = (fields(below_aggregate(plan)).into_iter())
        .map(|(_, tree)| tree)
        .collect();
    let as_stored = (names.iter())
        .map(|name| row_level(stored(manifest, dataset, name)))
        .collect();
    (trees, as_stored)
}

fn read_of(
    manifest: &SemanticManifest,
    dataset: &str,
    semantics: &str,
) -> (Vec<SchemaColumn>, Vec<PhysicalExpr>, Vec<PhysicalExpr>) {
    let (plan, _) = planned(manifest, &select(&[semantics]));
    let (trees, as_stored) = as_stored(manifest, &plan, dataset, &[semantics]);
    (scan_of(&plan).schema.columns.clone(), trees, as_stored)
}

#[tokio::test]
async fn a_prefix_reads_the_files_scan_of_its_listing_with_the_read_types() {
    let datasets = orders_daily(ORDERS_DAILY_FOLDER);
    let manifest = over_objects(SEMANTICS, &datasets, &ORDERS_DAILY_OBJECTS).await;

    let (plan, _) = planned(&manifest, &select(&["region", "revenue"]));

    let scan = scan_of(&plan);
    assert_eq!(
        scan.reader,
        parquet(
            None,
            "s3://lake/orders_daily/",
            &[
                "s3://lake/orders_daily/part-0.parquet",
                "s3://lake/orders_daily/part-1.parquet",
            ],
        )
    );
    assert_eq!(
        scan.schema.columns,
        [("region", DataType::String), ("amount", DataType::Double)].map(nullable)
    );
    let (trees, as_stored) = as_stored(&manifest, &plan, "orders_daily", &["region", "revenue"]);
    assert_eq!(trees, as_stored);
}

#[tokio::test]
async fn a_column_that_only_a_count_maps_reads_as_the_count_type() {
    let datasets = orders_daily(ORDERS_DAILY_FOLDER);
    let manifest = over_objects(SEMANTICS, &datasets, &ORDERS_DAILY_OBJECTS).await;

    let (columns, trees, as_stored) = read_of(&manifest, "orders_daily", "order_count");

    assert_eq!(columns, [nullable(("order_id", DataType::BigInt))]);
    assert_eq!(trees, as_stored);
}

const ORDERS_FOLDER: &str = "{format: parquet, paths: ['s3://lake/orders/']}";
const ORDERS_OBJECT: &str = "s3://lake/orders/part-0.parquet";

#[tokio::test]
async fn two_types_on_one_column_read_it_as_the_widest_and_each_casts_from_it() {
    for (semantics, datasets, column, reads) in [
        (
            r#"
  dimensions:
    - {name: order_date, data_type: date, type: categorical}
    - {name: order_ts, data_type: timestamp, type: categorical}
"#,
            r#"
    - name: orders_by_day
      extras: {storage: STORAGE, semantic_mapping: {order_date: created_at}}
      dimensions: [{ref: order_date}]
    - name: orders_by_time
      extras: {storage: STORAGE, semantic_mapping: {order_ts: created_at}}
      dimensions: [{ref: order_ts}]
"#,
            ("created_at", DataType::Timestamp { precision: 6 }),
            [
                ("orders_by_day", "order_date"),
                ("orders_by_time", "order_ts"),
            ],
        ),
        (
            r#"
  dimensions:
    - {name: order_ref, data_type: string, type: categorical}
  measures:
    - {name: order_count, data_type: bigint, agg: count}
"#,
            r#"
    - name: orders
      extras: {storage: STORAGE, semantic_mapping: {order_ref: order_id, order_count: order_id}}
      dimensions: [{ref: order_ref}]
      measures: [{ref: order_count}]
"#,
            ("order_id", DataType::String),
            [("orders", "order_ref"), ("orders", "order_count")],
        ),
        (
            r#"
  measures:
    - {name: tag, data_type: string, agg: max}
    - {name: quantity, data_type: bigint, agg: sum}
"#,
            r#"
    - name: orders
      extras: {storage: STORAGE, semantic_mapping: {tag: qty, quantity: qty}}
      measures: [{ref: tag}, {ref: quantity}]
"#,
            ("qty", DataType::BigInt),
            [("orders", "tag"), ("orders", "quantity")],
        ),
    ] {
        let datasets = datasets.replace("STORAGE", ORDERS_FOLDER);
        let manifest = over_objects(semantics, &datasets, &[ORDERS_OBJECT]).await;

        for (dataset, name) in reads {
            let (columns, trees, as_stored) = read_of(&manifest, dataset, name);

            assert_eq!(columns, [nullable(column.clone())], "{name}");
            assert_eq!(trees, as_stored, "{name}");
        }
    }
}

#[tokio::test]
async fn a_max_over_a_string_column_holds_the_cast_on_a_table_and_on_files() {
    let semantics = r#"
  measures:
    - {name: top_code, data_type: integer, agg: max}
"#;
    let codes = |storage: &str| {
        format!(
            r#"
    - name: codes
      extras:
        storage: {storage}
        semantic_mapping: {{top_code: code}}
      measures: [{{ref: top_code}}]
"#
        )
    };
    let catalog = Catalog::default().table(
        "warehouse.codes",
        FileFormat::Iceberg,
        &[("code", "string")],
    );
    let table = with_semantics(semantics, &codes("{tables: [warehouse.codes]}"), &catalog).await;
    let files = codes("{format: parquet, paths: ['s3://lake/codes/']}");
    let files = over_objects(semantics, &files, &["s3://lake/codes/part-0.parquet"]).await;

    for (manifest, read) in [(table, DataType::String), (files, DataType::Integer)] {
        let (columns, trees, as_stored) = read_of(&manifest, "codes", "top_code");

        assert_eq!(columns, [nullable(("code", read.clone()))], "{read:?}");
        assert_eq!(trees, as_stored, "{read:?}");
        let cast = matches!(
            &trees[..],
            [Expr::Cast {
                target: DataType::Integer,
                ..
            }]
        );
        assert!(cast, "{trees:?}");
    }
}

#[tokio::test]
async fn a_prefix_over_an_empty_folder_reads_the_empty_read() {
    let datasets = orders_daily(ORDERS_DAILY_FOLDER);
    let empty = over_objects(SEMANTICS, &datasets, &[]).await;
    let listed = over_objects(SEMANTICS, &datasets, &ORDERS_DAILY_OBJECTS).await;

    for (names, columns) in [
        (
            ["order_count", "revenue"],
            [("order_id", DataType::BigInt), ("amount", DataType::Double)],
        ),
        (
            ["region", "revenue"],
            [("region", DataType::String), ("amount", DataType::Double)],
        ),
    ] {
        let request = select(&names);
        let (plan, _) = planned(&empty, &request);
        let (usual, _) = planned(&listed, &request);

        let scan = scan_of(&plan);
        assert_eq!(
            scan.reader,
            VirtualTable::new(Vec::new()).into(),
            "{names:?}"
        );
        assert_eq!(scan.schema.columns, columns.map(nullable), "{names:?}");
        assert_eq!(scan.schema, scan_of(&usual).schema, "{names:?}");
        let (step, usual_step) = (aggregate_step(&plan.root), aggregate_step(&usual.root));
        assert_eq!(step.groupings, usual_step.groupings, "{names:?}");
        assert_eq!(step.aggregates, usual_step.aggregates, "{names:?}");
        assert_eq!(
            fields(below_aggregate(&plan)),
            fields(below_aggregate(&usual)),
            "{names:?}"
        );
        assert_eq!(output_names(&plan), names, "{names:?}");
    }
}

async fn an_expression_mapped_measure_over_untyped_columns() -> Golden {
    let semantics = r#"
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum, expr: {multiply: [{col: price}, {col: qty}]}}
"#;
    let datasets = r#"
    - name: orders
      extras:
        storage: {format: parquet, paths: ['s3://lake/orders/']}
        semantic_mapping: auto
      dimensions: [{ref: region}]
      measures: [{ref: revenue}]
"#;
    let manifest = over_objects(semantics, datasets, &[ORDERS_OBJECT]).await;

    let (error, advisories) = refusal(&manifest, &select(&["region", "revenue"]));

    let PlanError::UnresolvedType {
        path,
        relation,
        column,
    } = &error.kind
    else {
        panic!("UnresolvedType, got {error:?}");
    };
    assert_eq!(path.to_string(), "orders");
    assert_eq!(relation, "s3://lake/orders/");
    assert_eq!(column, "price");
    assert!(advisories.is_empty());
    Golden::message("unresolved_type", &error)
}

#[tokio::test]
async fn a_column_that_only_an_expression_mapping_reads_is_unresolved_type() {
    an_expression_mapped_measure_over_untyped_columns()
        .await
        .assert_committed();
}

#[tokio::test]
async fn a_formula_push_down_over_untyped_columns_is_unresolved_type() {
    let semantics = r#"
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum, expr: {multiply: [{col: price}, {col: qty}]}}
  metrics:
    - name: revenue_eu
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: revenue
"#;
    let datasets = r#"
    - name: orders
      extras:
        storage: {format: parquet, paths: ['s3://lake/orders/']}
        semantic_mapping: auto
      dimensions: [{ref: region}]
      measures: [{ref: revenue}]
      metrics: [{ref: revenue_eu}]
"#;
    let manifest = over_objects(semantics, datasets, &[ORDERS_OBJECT]).await;

    let (error, _) = refusal(&manifest, &select(&["revenue_eu"]));

    let PlanError::UnresolvedType { column, .. } = &error.kind else {
        panic!("UnresolvedType, got {error:?}");
    };
    assert_eq!(column, "price");
}

const XYZ: &str = r#"
    - name: orders_all
      extras:
        storage: {format: parquet, paths: ['s3://lake/orders/']}
        semantic_mapping: {revenue: amount}
      measures: [{ref: revenue}]
    - name: orders_b
      extras:
        storage: {format: parquet, paths: ['s3://lake/orders/b*.parquet']}
        semantic_mapping: {b_revenue: amount}
      measures: [{ref: b_revenue}]
    - name: orders_meta
      extras:
        storage: {format: parquet, paths: ['s3://lake/orders/_meta.parquet']}
        semantic_mapping: {meta_revenue: amount}
      measures: [{ref: meta_revenue}]
"#;

const XYZ_SEMANTICS: &str = r#"
  measures:
    - {name: revenue, data_type: double, agg: sum}
    - {name: b_revenue, data_type: double, agg: sum}
    - {name: meta_revenue, data_type: double, agg: sum}
"#;

#[tokio::test]
async fn each_selector_over_one_physical_source_reads_its_part_of_the_listing() {
    let objects = [
        "s3://lake/orders/b.parquet",
        "s3://lake/orders/a.parquet",
        "s3://lake/orders/_meta.parquet",
    ];
    let manifest = over_objects(XYZ_SEMANTICS, XYZ, &objects).await;

    for (measure, pattern, paths) in [
        (
            "revenue",
            None,
            &["s3://lake/orders/a.parquet", "s3://lake/orders/b.parquet"][..],
        ),
        (
            "b_revenue",
            Some("s3://lake/orders/b*.parquet"),
            &["s3://lake/orders/b.parquet"],
        ),
        ("meta_revenue", None, &["s3://lake/orders/_meta.parquet"]),
    ] {
        let (plan, _) = planned(&manifest, &select(&[measure]));

        let expected = parquet(pattern, "s3://lake/orders/", paths);
        assert_eq!(scan_of(&plan).reader, expected, "{measure}");
    }
}

#[tokio::test]
async fn a_uris_dataset_reads_its_objects_sorted() {
    let objects = ["s3://lake/orders/a.parquet", "s3://lake/orders/b.parquet"];
    for (paths, read) in [
        (
            "['s3://lake/orders/b.parquet', 's3://lake/orders/a.parquet']",
            &objects[..],
        ),
        ("['s3://lake/orders/a.parquet']", &objects[..1]),
    ] {
        let manifest = over_objects(SEMANTICS, &orders_daily(paths), &objects).await;

        let (plan, _) = planned(&manifest, &select(&["region", "revenue"]));

        let expected = parquet(None, "s3://lake/orders/", read);
        assert_eq!(scan_of(&plan).reader, expected, "{paths}");
    }
}

#[tokio::test]
async fn the_harness_folder_case_plans_over_the_files_read_of_its_object() {
    let (plan, _) = planned(&harness().await, &select(&["channel", "clicks"]));

    let expected = parquet(None, "s3://semstrait-e2e/events/", &[EVENTS_OBJECT]);
    assert_eq!(scan_of(&plan).reader, expected);
    assert_eq!(aggregate_step(&plan.root).groupings, [column("channel")]);
    assert_eq!(output_names(&plan), ["channel", "clicks"]);
}

const SALES_SEMANTICS: &str = r#"
  measures:
    - {name: total_amount, data_type: double, agg: sum}
"#;

fn sales(dimension: &str, declared: &str, key: &str) -> String {
    format!(
        r#"
    - name: sales
      extras:
        storage: {ORDERS_FOLDER}
        semantic_mapping: {{{dimension}: {key}, total_amount: amount}}
      dimensions: [{{name: {dimension}, {declared}}}]
      measures: [{{ref: total_amount}}]
"#
    )
}

const INTEGER: &str = "data_type: integer, type: categorical";

const YEAR_MONTH_OBJECTS: [&str; 2] = [
    "s3://lake/orders/year=2026/month=01/part-0.parquet",
    "s3://lake/orders/year=2026/month=02/part-0.parquet",
];

fn folder_read(plan: &SemanticPlan) -> &ScanNode {
    let mut node = &plan.root;
    while let [input] = node.children()[..] {
        node = input;
    }
    let PlanNode::Scan(scan) = node else {
        panic!("one chain down to the scan, got {node:?}");
    };
    assert_eq!(scan.reader, parquet(None, "s3://lake/orders/", &[]));
    scan
}

#[tokio::test]
async fn a_prefix_over_partitions_is_one_folder_read_with_the_key_as_a_column() {
    let datasets = sales("order_year", INTEGER, "year");
    let manifest = over_objects(SALES_SEMANTICS, &datasets, &YEAR_MONTH_OBJECTS).await;

    let request = ["order_year", "total_amount"];
    let (plan, _) = planned(&manifest, &select(&request));

    assert_eq!(
        folder_read(&plan).schema.columns,
        [("year", DataType::Integer), ("amount", DataType::Double)].map(nullable)
    );
    let (trees, as_stored) = as_stored(&manifest, &plan, "sales", &request);
    assert_eq!(trees, as_stored);
}

#[tokio::test]
async fn a_predicate_on_a_key_is_a_filter_node_over_the_folder_read() {
    let datasets = sales("order_year", INTEGER, "year");
    let manifest = over_objects(SALES_SEMANTICS, &datasets, &YEAR_MONTH_OBJECTS).await;
    let in_2026 = dim("order_year").expect("a name").eq(lit(2026));
    let request = select(&["order_year", "total_amount"]).with_predicate(in_2026);

    let (plan, _) = planned(&manifest, &request);

    let (filter, row_level) = filtered(&plan);
    let year = Expr::Literal(Literal::Integer(2026));
    assert_eq!(filter.filter, column("order_year").eq(year));
    assert!(names(&row_level.common.output_schema).contains(&"order_year"));
    assert_eq!(folder_read(&plan).filter, None);
}

#[tokio::test]
async fn a_temporal_dimension_over_a_key_is_bucketed_over_the_folder_read() {
    let temporal = "data_type: date, type: {temporal: {granularities: [day]}}";
    let datasets = sales("event_date", temporal, "dt");
    let objects = ["s3://lake/orders/dt=2026-01-01/part-0.parquet"];
    let manifest = over_objects(SALES_SEMANTICS, &datasets, &objects).await;
    let request = at_granularity("event_date", Granularity::Month, &["total_amount"]);

    let (plan, _) = planned(&manifest, &request);

    let bucketing = below_aggregate(&plan);
    let [("event_date", bucket), (argument, passed)] = &fields(bucketing)[..] else {
        panic!("the bucket, then sum's argument, got {bucketing:?}");
    };
    let date_trunc = call("date_trunc", vec![text("month"), column("event_date")]);
    assert_eq!(*bucket, date_trunc);
    assert_eq!(*passed, column(argument));
    let [("event_date", tree), _] = &fields(projection(&bucketing.input))[..] else {
        panic!("`event_date`'s row-level field, then sum's argument");
    };
    assert_eq!(*tree, stored(&manifest, "sales", "event_date"));
    assert_eq!(
        folder_read(&plan).schema.columns,
        [("dt", DataType::Date), ("amount", DataType::Double)].map(nullable)
    );
    assert_eq!(aggregate_step(&plan.root).groupings, [column("event_date")]);
}

async fn a_key_that_maps_no_semantics() -> Golden {
    let datasets = sales("order_month", INTEGER, "month");
    let manifest = over_objects(SALES_SEMANTICS, &datasets, &YEAR_MONTH_OBJECTS).await;

    let (plan, _) = planned(&manifest, &select(&["order_month", "total_amount"]));
    let (error, advisories) = refusal(&manifest, &select(&["year", "total_amount"]));

    assert_eq!(names(&folder_read(&plan).schema), ["month", "amount"]);
    assert_eq!(
        error.kind,
        PlanError::UnknownName {
            name: "year".into(),
            looked_in: LookedIn::SemanticDefinitions,
        }
    );
    assert!(advisories.is_empty());
    Golden::message("unmapped_key", &error)
}

#[tokio::test]
async fn a_key_that_maps_no_semantics_is_read_by_no_column_and_is_unknown_name() {
    a_key_that_maps_no_semantics().await.assert_committed();
}

const N_YEARS_SEMANTICS: &str = r#"
  measures:
    - {name: n_years, data_type: bigint, agg: count_distinct}
"#;

#[tokio::test]
async fn a_folder_read_writes_nothing_per_partition() {
    let sales = sales("order_year", INTEGER, "year");
    let n_years = format!(
        r#"
    - name: sales
      extras:
        storage: {ORDERS_FOLDER}
        semantic_mapping: {{order_year: year, n_years: year}}
      dimensions: [{{name: order_year, {INTEGER}}}]
      measures: [{{ref: n_years}}]
"#
    );
    let in_2026 = "s3://lake/orders/year=2026/part-0.parquet";
    let defaulted = "s3://lake/orders/year=__HIVE_DEFAULT_PARTITION__/part-0.parquet";
    let read = [("year", DataType::Integer), ("amount", DataType::Double)];
    let mut plans = Vec::new();

    for (semantics, datasets, objects, request, columns) in [
        (
            SALES_SEMANTICS,
            &sales,
            &[in_2026, defaulted][..],
            &["order_year", "total_amount"][..],
            &read[..],
        ),
        (
            SALES_SEMANTICS,
            &sales,
            &[in_2026],
            &["order_year", "total_amount"],
            &read,
        ),
        (
            N_YEARS_SEMANTICS,
            &n_years,
            &["s3://lake/orders/year=2025/part-0.parquet", in_2026],
            &["n_years"],
            &read[..1],
        ),
    ] {
        let manifest = over_objects(semantics, datasets, objects).await;

        let (plan, _) = planned(&manifest, &select(request));

        let scan = folder_read(&plan);
        let columns: Vec<_> = columns.iter().cloned().map(nullable).collect();
        assert_eq!(scan.schema.columns, columns, "{objects:?}");
        assert_eq!(scan.filter, None, "{objects:?}");
        let (trees, as_stored) = as_stored(&manifest, &plan, "sales", request);
        assert_eq!(trees, as_stored, "{objects:?}");
        plans.push(plan);
    }
    assert_eq!(plans[0], plans[1], "the default partition writes nothing");
}

#[tokio::test]
async fn a_uris_dataset_over_a_partitioned_source_lists_its_object() {
    let meta = "s3://lake/orders/_meta.parquet";
    let datasets = format!(
        r#"{}
    - name: sales_file
      extras:
        storage: {{format: parquet, paths: ['{meta}']}}
        semantic_mapping: {{file_year: year}}
      dimensions: [{{name: file_year, {INTEGER}}}]
"#,
        sales("order_year", INTEGER, "year")
    );
    let objects = ["s3://lake/orders/year=2026/part-0.parquet", meta];
    let manifest = over_objects(SALES_SEMANTICS, &datasets, &objects).await;

    let (file, _) = planned(&manifest, &select(&["file_year"]));
    let (folder, _) = planned(&manifest, &select(&["order_year"]));

    assert_eq!(manifest.sources.len(), 1, "one physical source");
    let scan = scan_of(&file);
    assert_eq!(scan.reader, parquet(None, "s3://lake/orders/", &[meta]));
    assert_eq!(scan.schema.columns, [nullable(("year", DataType::Integer))]);
    let (trees, as_stored) = as_stored(&manifest, &file, "sales_file", &["file_year"]);
    assert_eq!(trees, as_stored);
    assert_eq!(names(&folder_read(&folder).schema), ["year"]);
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            an_expression_mapped_measure_over_untyped_columns().await,
            a_key_that_maps_no_semantics().await,
        ]
    })
    .await;
}
