mod common;

use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::DataKindName;
use semstrait_common::Diagnostics;
use semstrait_common::Granularity;
use semstrait_common::Tree;
use semstrait_ir::{
    field_name, Delivery, JoinNode, JoinType, MultisetNode, MultisetOp, PhysicalExpr, PlanNode,
    ProjectNode, Reader, SemAnnotation, SemanticPlan, SourceRole, VirtualTable,
};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit, measure, metric};
use semstrait_planner::{Advisory, Request};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{CastFailure, Expr, ExprOperationExt, Literal, NamedReference};

use common::golden::{self, Golden};
use common::{
    advisory_kinds, aggregate_step, aggregate_steps, alias, at_granularity, balance_marker, bigint,
    call, column, columns_of, compiled, computed, datasets, every_marker, fields, filter,
    first_aggregate, grain_not_servable, join, markers, measure_markers, names,
    not_servable_branches, output_names, over_objects, planned, projection, qualified, rank,
    scan_below, scan_of, seeded, select, stored, text, union_of, with_semantics, written, Catalog,
    LATEST,
};

const SALES: &str = r#"
semantic_model:
  name: sales
  dimensions:
    - name: region
      data_type: string
      type: categorical
  measures:
    - name: avg_amount
      data_type: double
      agg: avg
    - name: customers
      data_type: bigint
      agg: count_distinct
  unionsets:
    - name: sales
      mode: all
      dimensions:
        - ref: region
      measures:
        - ref: avg_amount
        - ref: customers
      datasets:
        - name: online_sales
          extras:
            semantic_mapping: {region: region, avg_amount: amount, customers: customer_id}
            storage:
              tables: [warehouse.online]
        - name: store_sales
          extras:
            semantic_mapping: {region: region, avg_amount: amount, customers: customer_id}
            storage:
              tables: [warehouse.stores]
"#;

async fn over_tables(yaml: &str, tables: &[&str], columns: &[(&str, &str)]) -> SemanticManifest {
    let tables: Vec<_> = tables.iter().map(|table| (*table, columns)).collect();
    over_catalog(yaml, &tables).await
}

async fn over_catalog(yaml: &str, tables: &[(&str, &[(&str, &str)])]) -> SemanticManifest {
    let catalog = (tables.iter()).fold(Catalog::default(), |catalog, (table, columns)| {
        catalog.table(table, FileFormat::Iceberg, columns)
    });
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

async fn sales() -> SemanticManifest {
    let columns = [
        ("region", "string"),
        ("amount", "double"),
        ("customer_id", "string"),
    ];
    over_tables(SALES, &["warehouse.online", "warehouse.stores"], &columns).await
}

#[tokio::test]
async fn a_unionset_reads_its_branches_rows_under_one_aggregate_above_the_union() {
    let manifest = sales().await;
    let request = select(&["region", "avg_amount", "customers"]).with_from("sales");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "sales");
    let (above, union) = union_of(&plan);
    assert_eq!(union.op, MultisetOp::UnionAll);
    assert_eq!(
        aggregate_steps(&plan.root),
        1,
        "one aggregate step in {plan:?}"
    );
    let sales = DataKindName::new("sales");
    assert_eq!(
        union.common.semantic_meta,
        [SemAnnotation::SemanticSource {
            semantic_source: sales,
            source_role: SourceRole::Unionset,
            originators: Vec::new(),
            physical_source: None,
            cardinality: None,
            delivery: Some(Delivery::Rows),
            answered_by: None,
            pruned: Vec::new(),
        }]
    );
    let held = Expr::Literal(Literal::Boolean(true));
    let [online, stores] = &union.inputs[..] else {
        panic!("two branches, got {union:?}");
    };
    for (branch, table) in [(online, "online"), (stores, "stores")] {
        assert_eq!(
            aggregate_steps(branch),
            0,
            "a branch gives rows: {branch:?}"
        );
        let column = columns_of(&manifest, table);
        let qualified = |name: &str| format!("{alias}.{name}");
        let expected = [
            (qualified("__plan_cond_avg_amount"), held.clone()),
            (qualified("avg_amount"), column("amount")),
            (qualified("__plan_cond_customers"), held.clone()),
            (qualified("customers"), column("customer_id")),
            (qualified("region"), column("region")),
        ];
        let written: Vec<_> = (fields(projection(branch)).into_iter())
            .map(|(name, tree)| (name.to_owned(), tree))
            .collect();
        assert_eq!(written, expected, "the row projection of `{table}`");
    }
    let schema = &union.common.output_schema;
    let fields = names(schema);
    for name in [
        "avg_amount",
        "__plan_cond_avg_amount",
        "customers",
        "region",
    ] {
        let leaf = NamedReference::qualified(alias, name).expect("a name");
        assert!(
            projection(above)
                .expressions
                .contains(&Expr::Column(leaf.clone())),
            "the projection above the union reads `{name}`"
        );
        let bound = schema.position(&leaf).map(|position| fields[position]);
        assert_eq!(bound, Some(format!("{alias}.{name}").as_str()));
    }
}

const PAID_MEDIA: &str = r#"
semantic_model:
  name: ad_platforms
  dimensions:
    - {name: country, data_type: string, type: categorical}
    - {name: market, data_type: string, type: categorical}
    - {name: traffic_source, data_type: string, type: categorical}
  measures:
    - {name: cost, data_type: double, agg: sum}
  unionsets:
    - name: paid_media
      mode: all
      dimensions: [{ref: country}, {ref: market}, {ref: traffic_source}]
      measures: [{ref: cost}]
      filters:
        - name: swedish
          expr: {eq: [country, {lit: SE}]}
        - name: not_swedish
          expr: {not: {eq: [country, {lit: SE}]}}
        - name: unplaced
          expr: {is_null: country}
      datasets:
        - name: adwords_campaign_data
          extras:
            storage: {tables: [polaris.adwords]}
            semantic_mapping:
              traffic_source: {lit: {string: Google}}
              country: country
              market: market
              cost: cost
        - name: bing_campaign_data
          extras:
            storage: {tables: [polaris.bing]}
            semantic_mapping:
              traffic_source: {lit: {string: Bing}}
              market: campaign
              cost: cost
"#;

async fn paid_media() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "polaris.adwords",
            FileFormat::Iceberg,
            &[
                ("country", "string"),
                ("market", "string"),
                ("cost", "double"),
            ],
        )
        .table(
            "polaris.bing",
            FileFormat::Iceberg,
            &[("campaign", "string"), ("cost", "double")],
        );
    compiled(PAID_MEDIA, &catalog, &InMemoryFileSystem::new()).await
}

fn row_projection(branch: &PlanNode) -> &ProjectNode {
    match branch {
        PlanNode::Filter(filter) => row_projection(&filter.input),
        branch => projection(branch),
    }
}

fn rows_below_partial(branch: &PlanNode) -> &PlanNode {
    &projection(&aggregate_step(branch).input).input
}

fn delivery_of(meta: &[SemAnnotation]) -> Option<Delivery> {
    meta.iter().find_map(|subject| match subject {
        SemAnnotation::SemanticSource { delivery, .. } => *delivery,
        _ => None,
    })
}

fn semantic_source_of(meta: &[SemAnnotation]) -> Option<&DataKindName> {
    meta.iter().find_map(|subject| match subject {
        SemAnnotation::SemanticSource {
            semantic_source, ..
        } => Some(semantic_source),
        _ => None,
    })
}

#[tokio::test]
async fn a_branch_writes_a_typed_null_for_a_name_that_it_does_not_map() {
    let manifest = paid_media().await;
    let request = select(&["country", "market", "cost"]).with_from("paid_media");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "paid_media");
    let (_, union) = union_of(&plan);
    let [adwords, bing] = &union.inputs[..] else {
        panic!("two branches, in name order, got {union:?}");
    };
    let (country, market) = (format!("{alias}.country"), format!("{alias}.market"));
    let bing = row_projection(rows_below_partial(bing));
    assert_eq!(
        written(bing, &country),
        Some(Expr::Literal(Literal::Null(DataType::String)))
    );
    assert_eq!(
        written(bing, &market),
        Some(columns_of(&manifest, "bing")("campaign"))
    );
    let adwords = row_projection(rows_below_partial(adwords));
    assert_eq!(
        written(adwords, &country),
        Some(columns_of(&manifest, "adwords")("country"))
    );
}

#[tokio::test]
async fn a_predicate_on_a_name_that_a_branch_null_fills_reads_its_typed_null() {
    let manifest = paid_media().await;
    let in_sweden =
        dim("country").expect("a name").eq(lit("SE")) | dim("market").expect("a name").eq(lit("x"));

    for selected in [&["country", "cost"][..], &["cost"]] {
        let request = select(selected)
            .with_from("paid_media")
            .with_predicate(in_sweden.clone());

        let (plan, _) = planned(&manifest, &request);

        let alias = &alias(&manifest, "paid_media");
        let (_, union) = union_of(&plan);
        let [_, bing] = &union.inputs[..] else {
            panic!("two branches, in name order, got {union:?}");
        };
        let PlanNode::Filter(filter) = rows_below_partial(bing) else {
            panic!("the predicate above the branch's row projection, got {bing:?}");
        };
        let expected =
            qualified(alias, "country").eq(text("SE")) | qualified(alias, "market").eq(text("x"));
        assert_eq!(filter.filter, expected);
        assert_eq!(
            written(projection(&filter.input), &format!("{alias}.country")),
            Some(Expr::Literal(Literal::Null(DataType::String))),
            "the Bing branch of `select {selected:?}`"
        );
    }
}

#[tokio::test]
async fn a_branch_that_null_fills_a_name_is_not_read_when_the_filter_cannot_be_true_on_it() {
    let manifest = paid_media().await;
    let country = || dim("country").expect("a name");
    let read = |predicate| {
        let request = select(&["cost"])
            .with_from("paid_media")
            .with_predicate(predicate);
        let (plan, advisories) = planned(&manifest, &request);
        assert_eq!(advisory_kinds(advisories), []);
        let read: Vec<String> = datasets(&plan).iter().map(ToString::to_string).collect();
        (read, pruned(&plan).len())
    };

    let (equals, equals_pruned) = read(country().eq(lit("SE")));
    assert_eq!(equals, ["paid_media.adwords_campaign_data"]);
    assert_eq!(equals_pruned, 1);
    let (negated, _) = read(!country().eq(lit("SE")));
    assert_eq!(negated, ["paid_media.adwords_campaign_data"]);
    let (null, null_pruned) = read(country().is_null());
    assert_eq!(
        null,
        [
            "paid_media.adwords_campaign_data",
            "paid_media.bing_campaign_data"
        ]
    );
    assert_eq!(null_pruned, 0);
    let (either, _) = read(country().eq(lit("SE")) | dim("market").expect("a name").eq(lit("x")));
    assert_eq!(either.len(), 2);
}

#[tokio::test]
async fn an_activated_named_filter_prunes_the_branch_that_null_fills_its_name() {
    let manifest = paid_media().await;
    let read = |filter: &str| {
        let request = select(&["cost"])
            .with_from("paid_media")
            .with_filters(vec![filter.into()]);
        let (plan, advisories) = planned(&manifest, &request);
        assert_eq!(advisory_kinds(advisories), []);
        let names = |names: Vec<&str>| names.into_iter().map(String::from).collect::<Vec<_>>();
        (names(datasets(&plan)), names(pruned(&plan)))
    };

    for filter in ["swedish", "not_swedish"] {
        let (read, pruned) = read(filter);
        assert_eq!(read, ["paid_media.adwords_campaign_data"], "{filter}");
        assert_eq!(pruned, ["paid_media.bing_campaign_data"], "{filter}");
    }
    let (read, pruned) = read("unplaced");
    assert_eq!(
        read,
        [
            "paid_media.adwords_campaign_data",
            "paid_media.bing_campaign_data"
        ]
    );
    assert_eq!(pruned, Vec::<String>::new());
}

const WEB_SALES: &str = r#"
semantic_model:
  name: web_sales
  dimensions:
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: web_sales
      mode: all
      dimensions: [{ref: channel}]
      measures: [{ref: revenue}]
      filters:
        - name: web_only
          expr:
            eq: [channel, {lit: web}]
      datasets:
        - name: east
          extras:
            storage: {tables: [warehouse.east]}
            semantic_mapping: {channel: channel, revenue: amount}
        - name: west
          extras:
            storage: {tables: [warehouse.west]}
            semantic_mapping: {channel: channel, revenue: amount}
"#;

#[tokio::test]
async fn a_named_filter_on_a_unionset_goes_into_every_branch() {
    let columns = [("channel", "string"), ("amount", "double")];
    let tables = ["warehouse.east", "warehouse.west"];
    let manifest = over_tables(WEB_SALES, &tables, &columns).await;
    let request = select(&["revenue"])
        .with_from("web_sales")
        .with_filters(vec!["web_only".into()]);

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "web_sales");
    let (above, union) = union_of(&plan);
    assert!(
        matches!(projection(above).input.as_ref(), PlanNode::Multiset(_)),
        "the Unionset's frame holds no filter: {above:?}"
    );
    assert!(!names(&projection(above).common.output_schema).contains(&"web_only"));
    for (branch, table) in union.inputs.iter().zip(["east", "west"]) {
        let PlanNode::Filter(filter) = projection(rows_below_partial(branch)).input.as_ref() else {
            panic!("the named filter above the branch's row projection, got {branch:?}");
        };
        assert_eq!(filter.filter, qualified(alias, "web_only"));
        let rows = projection(&filter.input);
        let web_only = stored(&manifest, &format!("web_sales.{table}"), "web_only");
        assert_eq!(written(rows, &format!("{alias}.web_only")), Some(web_only));
        assert_eq!(
            written(rows, &format!("{alias}.channel")),
            Some(columns_of(&manifest, table)("channel")),
        );
    }
}

const TWO_FOLDERS: &str = r#"
    - name: sales
      extras:
        storage: {format: parquet, paths: ['s3://lake/eu/orders/', 's3://lake/us/orders/']}
        semantic_mapping: {order_year: year, total_amount: amount}
      dimensions: [{name: order_year, data_type: integer, type: categorical}]
      measures: [{ref: total_amount}]
"#;

const SALES_SEMANTICS: &str = r#"
  measures:
    - {name: total_amount, data_type: double, agg: sum}
"#;

const BASE_PATHS: [&str; 2] = ["s3://lake/eu/orders/", "s3://lake/us/orders/"];

fn column_at(manifest: &SemanticManifest, base_path: &str, name: &str) -> PhysicalExpr {
    let source = (manifest.sources.values())
        .find(|source| source.locator() == base_path)
        .expect("the physical source");
    Expr::Column(NamedReference::qualified(source.alias.as_str(), name).expect("a name"))
}

#[tokio::test]
async fn a_dataset_over_two_folders_reads_both_with_the_predicate_in_each_branch() {
    let objects = [
        "s3://lake/eu/orders/year=2025/part-0.parquet",
        "s3://lake/eu/orders/year=2026/part-0.parquet",
        "s3://lake/us/orders/year=2026/part-0.parquet",
    ];
    let manifest = over_objects(SALES_SEMANTICS, TWO_FOLDERS, &objects).await;
    let request = select(&["order_year", "total_amount"]);
    let in_2026 = dim("order_year").expect("a name").eq(lit(2026));

    let (plan, _) = planned(&manifest, &request);
    let (filtered, _) = planned(&manifest, &request.with_predicate(in_2026));

    let alias = &alias(&manifest, "sales");
    for plan in [&plan, &filtered] {
        let (above, union) = union_of(plan);
        assert_eq!(union.op, MultisetOp::UnionAll);
        assert_eq!(
            aggregate_steps(&plan.root),
            3,
            "three aggregates in {plan:?}"
        );
        assert!(
            matches!(projection(above).input.as_ref(), PlanNode::Multiset(_)),
            "the Unionset's frame holds no filter: {above:?}"
        );
        let read: Vec<_> = (union.inputs.iter())
            .map(|branch| match &scan_below(branch).reader {
                Reader::FilesScan(files) => files.base_path.as_str(),
                other => panic!("a files read, got {other:?}"),
            })
            .collect();
        assert_eq!(read, BASE_PATHS);
    }
    let (_, union) = union_of(&filtered);
    let year = NamedReference::qualified(alias, "order_year").expect("a name");
    let in_2026 = Expr::Column(year).eq(Expr::Literal(Literal::Integer(2026)));
    for (branch, base_path) in union.inputs.iter().zip(BASE_PATHS) {
        let PlanNode::Filter(filter) = rows_below_partial(branch) else {
            panic!("the predicate above the branch's row projection, got {branch:?}");
        };
        assert_eq!(filter.filter, in_2026);
        let year = Expr::Cast {
            input: Box::new(column_at(&manifest, base_path, "year")),
            target: DataType::Integer,
            on_failure: CastFailure::Error,
        };
        assert_eq!(
            written(projection(&filter.input), &format!("{alias}.order_year")),
            Some(year),
        );
    }
}

const TWO_TABLES: &str = r#"
    - name: orders
      extras:
        storage: {tables: [warehouse.eu_orders, warehouse.us_orders]}
        semantic_mapping: {revenue: amount}
      measures: [{ref: revenue}]
"#;

#[tokio::test]
async fn a_dataset_over_two_tables_reads_its_reduction_leaves_under_one_union() {
    let columns = [("amount", "double")];
    let catalog = Catalog::default()
        .table("warehouse.eu_orders", FileFormat::Iceberg, &columns)
        .table("warehouse.us_orders", FileFormat::Iceberg, &columns);
    let revenue = "  measures:\n    - {name: revenue, data_type: double, agg: sum}\n";
    let manifest = with_semantics(revenue, TWO_TABLES, &catalog).await;

    let (plan, _) = planned(&manifest, &select(&["revenue"]));

    let orders = DataKindName::new("orders");
    assert!(
        (plan.root.common().semantic_meta).iter().any(
            |subject| matches!(subject, SemAnnotation::Rank { scope_root, .. } if *scope_root == orders)
        ),
        "the scope root is `orders`: {plan:?}"
    );
    let (_, union) = union_of(&plan);
    assert_eq!(union.op, MultisetOp::UnionAll);
    assert_eq!(
        semantic_source_of(&union.common.semantic_meta),
        Some(&orders)
    );
    assert_eq!(
        aggregate_steps(&plan.root),
        3,
        "three aggregates in {plan:?}"
    );
    let read: Vec<_> = (union.inputs.iter())
        .map(|branch| semantic_source_of(&scan_below(branch).common.semantic_meta).cloned())
        .collect();
    let leaf = |name: &str| Some(DataKindName::new(name));
    assert_eq!(read, [leaf("orders_eu_orders"), leaf("orders_us_orders")]);
}

const SHIPMENTS: &str = r#"
semantic_model:
  name: shipments
  dimensions:
    - {name: carrier, data_type: string, type: categorical}
  measures:
    - {name: parcels, data_type: integer, agg: count}
  unionsets:
    - name: shipments
      mode: all
      dimensions: [{ref: carrier}]
      measures: [{ref: parcels}]
      datasets:
        - name: rail
          extras:
            storage: {tables: [warehouse.rail]}
            semantic_mapping: {carrier: carrier, parcels: parcel_id}
        - name: road
          extras:
            storage: {tables: [warehouse.road]}
            semantic_mapping: {carrier: carrier, parcels: parcel_id}
"#;

const AIR: &str = r#"
        - name: air
          extras:
            storage: {tables: [warehouse.air]}
            semantic_mapping: {carrier: carrier}
"#;

async fn shipments(yaml: &str) -> SemanticManifest {
    let parcel_id = |data_type| [("carrier", "string"), ("parcel_id", data_type)];
    let catalog = Catalog::default()
        .table("warehouse.rail", FileFormat::Iceberg, &parcel_id("integer"))
        .table("warehouse.road", FileFormat::Iceberg, &parcel_id("bigint"))
        .table(
            "warehouse.air",
            FileFormat::Iceberg,
            &[("carrier", "string")],
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_count_argument_of_two_types_is_cast_to_the_least_restrictive_in_each_branch() {
    let manifest = shipments(SHIPMENTS).await;
    let request = select(&["parcels"]).with_from("shipments");

    let (plan, _) = planned(&manifest, &request);

    let parcels = format!("{}.parcels", alias(&manifest, "shipments"));
    let (_, union) = union_of(&plan);
    for (branch, table) in union.inputs.iter().zip(["rail", "road"]) {
        let bigint = Expr::Cast {
            input: Box::new(columns_of(&manifest, table)("parcel_id")),
            target: DataType::BigInt,
            on_failure: CastFailure::Error,
        };
        let rows = rows_below_partial(branch);
        assert_eq!(written(projection(rows), &parcels), Some(bigint), "{table}");
        let schema = &rows.common().output_schema;
        let field = (schema.columns.iter()).find(|field| field.name == parcels);
        assert_eq!(field.map(|field| &field.data_type), Some(&DataType::BigInt));
    }
}

const VISITS: &str = r#"
semantic_model:
  name: visits
  dimensions:
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
    - {name: visitors, data_type: bigint, agg: count_distinct}
  metrics:
    - name: revenue_per_visitor
      data_type: double
      expr:
        divide: [revenue, visitors]
  unionsets:
    - name: visits
      mode: all
      dimensions: [{ref: channel}]
      measures: [{ref: revenue}, {ref: visitors}]
      metrics:
        - ref: revenue_per_visitor
          filter:
            eq: [channel, {lit: web}]
      datasets:
        - name: shop_visits
          extras:
            storage: {tables: [warehouse.shop]}
            semantic_mapping: {channel: channel, revenue: amount, visitors: visitor_id}
        - name: web_visits
          extras:
            storage: {tables: [warehouse.web]}
            semantic_mapping: {channel: channel, revenue: amount, visitors: visitor_id}
"#;

#[tokio::test]
async fn a_composed_metric_on_a_unionset_runs_its_formula_over_the_aggregate_above_the_union() {
    let columns = [
        ("channel", "string"),
        ("amount", "double"),
        ("visitor_id", "string"),
    ];
    let tables = ["warehouse.shop", "warehouse.web"];
    let manifest = over_tables(VISITS, &tables, &columns).await;
    let request = select(&["revenue_per_visitor", "visitors"]).with_from("visits");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "visits");
    let union_field =
        |name: &str| Expr::Column(NamedReference::qualified(alias, name).expect("a name"));
    let (_, union) = union_of(&plan);
    assert_eq!(aggregate_steps(&plan.root), 1, "one aggregate in {plan:?}");
    for branch in &union.inputs {
        assert_eq!(
            aggregate_steps(branch),
            0,
            "a branch gives rows: {branch:?}"
        );
    }
    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let row_level = projection(&aggregate.input);
    assert!(matches!(row_level.input.as_ref(), PlanNode::Multiset(_)));
    let [(revenue_entry, revenue), (visitors_entry, visitors), (entry, _)] =
        aggregate.aggregates.as_slice()
    else {
        panic!("three entries, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "visitors");
    let web = union_field("channel").eq(text("web"));
    for (call, function, measure) in [(revenue, "sum", "revenue"), (visitors, "count", "visitors")]
    {
        assert_eq!(call.function().declarations.canonical.name, function);
        let read = |operand: &PhysicalExpr| match operand {
            Expr::Column(field) => written(row_level, field.name()),
            other => panic!("a field of stage 2, got {other:?}"),
        };
        assert_eq!(read(&call.args()[0]), Some(union_field(measure)));
        let condition = union_field(&format!("__plan_cond_{measure}"));
        assert_eq!(
            call.filter().map(read),
            Some(Some(web.clone() & condition)),
            "the filter clause of `{measure}`"
        );
    }
    let formula = column(revenue_entry.as_str()) / column(visitors_entry.as_str());
    assert_eq!(
        derivation.expressions,
        [formula, column("visitors")],
        "the formula over the step's entries"
    );
}

#[tokio::test]
async fn an_ungrouped_formula_on_a_unionset_takes_its_parts_over_partials_by_its_key_columns() {
    let web_bonus = "    - name: web_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {add: [revenue, {lit: 100}]}
          else: revenue
  unionsets:
";
    let site = "        - ref: revenue_per_visitor\n";
    let yaml = (VISITS.replacen("  unionsets:\n", web_bonus, 1)).replacen(
        site,
        &format!("{site}        - ref: web_bonus\n"),
        1,
    );
    let columns = [
        ("channel", "string"),
        ("amount", "double"),
        ("visitor_id", "string"),
    ];
    let manifest = over_tables(&yaml, &["warehouse.shop", "warehouse.web"], &columns).await;

    let (plan, _) = planned(&manifest, &select(&["web_bonus"]).with_from("visits"));

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::PartialAggregates);
    for branch in &union.inputs {
        let (grouped, entries) = branch_step(branch);
        assert_eq!(grouped, ["channel"]);
        assert_eq!(entries, ["__plan_agg1_web_bonus", "__plan_agg2_web_bonus"]);
    }
    let steps: Vec<_> = (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect();
    let [step_2, step_1, _, _] = steps[..] else {
        panic!("two aggregate steps above the union's two, got {steps:?}");
    };
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let [key] = step_1.groupings.as_slice() else {
        panic!("the part key alone, got {:?}", step_1.groupings);
    };
    let channel = qualified(&alias(&manifest, "visits"), "channel");
    assert_eq!(
        computed(projection(&step_1.input), key),
        Expr::Case {
            whens: vec![(channel.eq(text("web")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    assert_eq!(output_names(&plan), ["web_bonus"]);
}

const SPLIT_ORDERS: &str = r#"
semantic_model:
  name: split_orders
  dimensions:
    - {name: region, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  metrics:
    - name: revenue_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
  unionsets:
    - name: split_orders
      mode: all
      dimensions: [{ref: region}, {ref: channel}]
      measures: [{ref: revenue}]
      metrics: [{ref: revenue_adj}]
      datasets:
        - name: eu_orders
          extras:
            storage: {tables: [warehouse.eu_orders]}
            semantic_mapping: {region: region, channel: channel, revenue: amount}
        - name: us_orders
          extras:
            storage: {tables: [warehouse.us_orders]}
            semantic_mapping: {region: region, channel: channel, revenue: amount}
"#;

#[tokio::test]
async fn a_formula_push_down_over_a_unionset_reads_its_branches_rows_under_one_sum() {
    let columns = [
        ("region", "string"),
        ("channel", "string"),
        ("amount", "double"),
    ];
    let tables = ["warehouse.eu_orders", "warehouse.us_orders"];
    let manifest = over_tables(SPLIT_ORDERS, &tables, &columns).await;
    let request = select(&["channel", "revenue_adj"]).with_from("split_orders");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "split_orders");
    let union_field = |name: &str| qualified(alias, name);
    let (_, union) = union_of(&plan);
    assert_eq!(aggregate_steps(&plan.root), 1, "one aggregate in {plan:?}");
    for (branch, table) in union.inputs.iter().zip(["eu_orders", "us_orders"]) {
        assert_eq!(
            aggregate_steps(branch),
            0,
            "a branch gives rows: {branch:?}"
        );
        let column = columns_of(&manifest, table);
        let written: Vec<_> = (fields(projection(branch)).into_iter())
            .map(|(name, tree)| (name.to_owned(), tree))
            .collect();
        let expected = [
            (format!("{alias}.channel"), column("channel")),
            (format!("{alias}.region"), column("region")),
            (
                format!("{alias}.__plan_cond_revenue"),
                Expr::Literal(Literal::Boolean(true)),
            ),
            (format!("{alias}.revenue"), column("amount")),
        ];
        assert_eq!(written, expected, "the row projection of `{table}`");
    }
    let aggregate = first_aggregate(&plan.root);
    assert_eq!(aggregate.groupings, [column("channel")]);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "revenue_adj");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let row_level = projection(&aggregate.input);
    let term = Expr::Case {
        whens: vec![(
            union_field("region").eq(text("eu")),
            union_field("revenue") * Expr::Literal(Literal::Double(1.2)),
        )],
        else_: Some(Box::new(union_field("revenue"))),
    };
    assert_eq!(computed(row_level, &sum.args()[0]), term);
    assert_eq!(
        sum.filter().map(|condition| computed(row_level, condition)),
        Some(union_field("__plan_cond_revenue"))
    );
    assert!(matches!(row_level.input.as_ref(), PlanNode::Multiset(_)));
    assert_eq!(output_names(&plan), ["channel", "revenue_adj"]);
}

#[tokio::test]
async fn a_formula_whose_links_read_two_conditions_over_a_unionset_takes_its_parts() {
    let refunds = "    - {name: refunds, data_type: double, agg: sum}\n  metrics:
    - name: revenue_or_refunds
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: revenue
          else: refunds
";
    let yaml = (SPLIT_ORDERS.replacen("  metrics:\n", refunds, 1))
        .replacen(
            "measures: [{ref: revenue}]",
            "measures: [{ref: revenue}, {ref: refunds}]",
            1,
        )
        .replacen(
            "metrics: [{ref: revenue_adj}]",
            "metrics: [{ref: revenue_adj}, {ref: revenue_or_refunds}]",
            1,
        )
        .replace("revenue: amount}", "revenue: amount, refunds: refund}");
    let columns = [
        ("region", "string"),
        ("channel", "string"),
        ("amount", "double"),
        ("refund", "double"),
    ];
    let tables = ["warehouse.eu_orders", "warehouse.us_orders"];
    let manifest = over_tables(&yaml, &tables, &columns).await;
    let request = select(&["channel", "revenue_or_refunds"]).with_from("split_orders");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::PartialAggregates);
    for branch in &union.inputs {
        let (grouped, entries) = branch_step(branch);
        assert_eq!(grouped, ["channel", "region"]);
        let parts = [
            "__plan_agg1_revenue_or_refunds",
            "__plan_agg2_revenue_or_refunds",
        ];
        assert_eq!(entries, parts);
    }
    assert_eq!(steps_above_unions(&plan), 2, "the parts in {plan:?}");
}

#[tokio::test]
async fn a_null_fill_of_a_count_argument_takes_the_least_restrictive_type() {
    let manifest = shipments(&format!("{SHIPMENTS}{AIR}")).await;
    let request = select(&["parcels"]).with_from("shipments");

    let (plan, _) = planned(&manifest, &request);

    let parcels = format!("{}.parcels", alias(&manifest, "shipments"));
    let (_, union) = union_of(&plan);
    let [air, ..] = &union.inputs[..] else {
        panic!("three branches, in name order, got {union:?}");
    };
    assert_eq!(
        written(projection(rows_below_partial(air)), &parcels),
        Some(Expr::Literal(Literal::Null(DataType::BigInt)))
    );
}

const ORDERS: &str = r#"
semantic_model:
  name: orders
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: order_count, data_type: bigint, agg: count}
    - {name: revenue, data_type: 'decimal(18,2)', agg: sum}
  unionsets:
    - name: orders
      mode: all
      dimensions: [{ref: region}]
      measures: [{ref: order_count}, {ref: revenue}]
      datasets:
        - name: eu_orders
          extras:
            storage: {tables: [warehouse.eu_orders]}
            semantic_mapping: {region: region, order_count: order_id, revenue: amount}
        - name: us_orders
          extras:
            storage: {tables: [warehouse.us_orders]}
            semantic_mapping: {order_count: order_id, revenue: amount}
"#;

const ORDER_TABLES: [&str; 2] = ["warehouse.eu_orders", "warehouse.us_orders"];

const ORDER_COLUMNS: [(&str, &str); 3] = [
    ("region", "string"),
    ("order_id", "string"),
    ("amount", "decimal(18,2)"),
];

async fn orders() -> SemanticManifest {
    over_tables(ORDERS, &ORDER_TABLES, &ORDER_COLUMNS).await
}

#[tokio::test]
async fn a_count_over_a_unionset_sums_its_branches_partial_counts_coalesced_with_zero() {
    let manifest = orders().await;
    let request = select(&["order_count"]).with_from("orders");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "orders");
    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
    let partial = format!("{alias}.order_count");
    for branch in &union.inputs {
        let [(entry, count)] = aggregate_step(branch).aggregates.as_slice() else {
            panic!("one partial count, got {branch:?}");
        };
        assert_eq!(entry.as_str(), partial);
        assert_eq!(count.function().declarations.canonical.name, "count");
    }
    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        qualified(alias, "order_count")
    );
    assert_eq!(sum.filter(), None, "the branches applied each condition");
    let zero = Expr::Literal(Literal::BigInt(0));
    assert_eq!(
        derivation.expressions,
        [call("coalesce", vec![column(entry.as_str()), zero])]
    );
}

#[tokio::test]
async fn a_partial_sum_is_cast_to_its_measure_type_only_above_the_union() {
    let manifest = orders().await;
    let request = select(&["revenue"]).with_from("orders");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "orders");
    let (_, union) = union_of(&plan);
    let partial = format!("{alias}.__plan_agg1_revenue");
    for branch in &union.inputs {
        let [(entry, sum)] = aggregate_step(branch).aggregates.as_slice() else {
            panic!("one partial sum, got {branch:?}");
        };
        assert_eq!(entry.as_str(), partial);
        assert_eq!(sum.function().declarations.canonical.name, "sum");
    }
    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        qualified(alias, "__plan_agg1_revenue")
    );
    let cast = Expr::Cast {
        input: Box::new(column(entry.as_str())),
        target: DataType::Decimal {
            precision: 18,
            scale: 2,
        },
        on_failure: CastFailure::Error,
    };
    assert_eq!(derivation.expressions, [cast]);
}

#[tokio::test]
async fn a_grouped_count_over_a_unionset_coalesces_the_sum_of_its_partial_counts() {
    let manifest = orders().await;
    let request = select(&["region", "order_count"]).with_from("orders");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "orders");
    let (_, union) = union_of(&plan);
    let [eu, us] = &union.inputs[..] else {
        panic!("two branches, in name order, got {union:?}");
    };
    let region = format!("{alias}.region");
    assert_eq!(
        written(projection(rows_below_partial(us)), &region),
        Some(Expr::Literal(Literal::Null(DataType::String)))
    );
    for branch in [eu, us] {
        let partial = aggregate_step(branch);
        assert_eq!(partial.groupings, [qualified(alias, "region")]);
        assert_eq!(
            names(&partial.common.output_schema),
            [region.as_str(), &format!("{alias}.order_count")]
        );
    }
    let derivation = projection(&projection(&plan.root).input);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let zero = Expr::Literal(Literal::BigInt(0));
    assert_eq!(
        derivation.expressions,
        [
            call("coalesce", vec![column(entry.as_str()), zero]),
            column("region")
        ]
    );
}

const BALANCES: &str = r#"
semantic_model:
  name: balances
  dimensions:
    - {name: account, data_type: string, type: categorical}
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: balance
      data_type: bigint
      agg: sum
      additivity:
        semi: {axes: [], by: date, strategy: latest}
  unionsets:
    - name: balances
      mode: all
      dimensions: [{ref: account}, {ref: date}]
      measures: [{ref: balance}]
      datasets:
        - name: day_30
          extras:
            storage: {tables: [warehouse.day_30]}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
            semantic_mapping: {account: account, date: date, balance: balance}
        - name: day_31
          extras:
            storage: {tables: [warehouse.day_31]}
            semantic_mapping: {account: account, date: date, balance: balance}
"#;

async fn balances() -> SemanticManifest {
    let columns = [("account", "string"), ("date", "date"), ("balance", "long")];
    let tables = ["warehouse.day_30", "warehouse.day_31"];
    over_tables(BALANCES, &tables, &columns).await
}

#[tokio::test]
async fn a_semi_additive_partial_is_grouped_by_the_day_and_marked_above_the_union() {
    let manifest = balances().await;
    let request = at_granularity("date", Granularity::Month, &["balance"]).with_from("balances");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "balances");
    let (_, union) = union_of(&plan);
    for branch in &union.inputs {
        assert_eq!(every_marker(branch), [], "no marker below the union");
        let partial = aggregate_step(branch);
        assert_eq!(partial.groupings, [qualified(alias, "date")]);
        let [(entry, sum)] = partial.aggregates.as_slice() else {
            panic!("one partial sum, got {partial:?}");
        };
        assert_eq!(entry.as_str(), format!("{alias}.balance"));
        assert_eq!(sum.function().declarations.canonical.name, "sum");
        let condition = sum
            .filter()
            .map(|filter| computed(projection(&partial.input), filter));
        assert_eq!(
            condition,
            Some(qualified(alias, "__plan_cond_balance")),
            "its branch's condition alone"
        );
    }
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    balance_marker(aggregate, &["date"]);
    let bucketing = projection(&projection(&aggregate.input).input);
    assert_eq!(
        written(bucketing, "date"),
        Some(call("date_trunc", vec![text("month"), column("date")]))
    );
    let row_level = projection(&bucketing.input);
    assert_eq!(written(row_level, "date"), Some(qualified(alias, "date")));
    assert_eq!(
        computed(row_level, &sum.args()[0]),
        qualified(alias, "balance")
    );
    assert!(matches!(row_level.input.as_ref(), PlanNode::Multiset(_)));
}

#[tokio::test]
async fn a_semi_additive_partial_is_grouped_by_a_by_that_the_request_drops() {
    let manifest = balances().await;
    let request = select(&["account", "balance"]).with_from("balances");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "balances");
    let (_, union) = union_of(&plan);
    for branch in &union.inputs {
        assert_eq!(every_marker(branch), [], "no marker below the union");
        assert_eq!(
            aggregate_step(branch).groupings,
            [qualified(alias, "account"), qualified(alias, "date")]
        );
    }
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("account")]);
    let marking = projection(&aggregate.input);
    let [(_, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    assert_eq!(
        window,
        &rank(vec![column("account")], column("date"), LATEST)
    );
    assert_eq!(output_names(&plan), ["account", "balance"]);
}

const DAILY: &str = r#"
semantic_model:
  name: daily
  dimensions:
    - {name: account, data_type: string, type: categorical}
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: a
      data_type: double
      agg: sum
      additivity:
        semi: {axes: [], by: date, strategy: latest}
    - name: b
      data_type: double
      agg: sum
      additivity:
        semi: {axes: [], by: date, strategy: latest}
  metrics:
    - name: m_lin
      data_type: double
      expr:
        subtract: [a, b]
  unionsets:
    - name: daily
      mode: all
      dimensions: [{ref: account}, {ref: date}]
      measures: [{ref: a}, {ref: b}]
      metrics: [{ref: m_lin}]
      datasets:
        - name: eu
          extras:
            storage: {tables: [warehouse.daily_eu]}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
            semantic_mapping: {account: account, date: date, a: a, b: b}
        - name: rest
          extras:
            storage: {tables: [warehouse.daily_rest]}
            semantic_mapping: {account: account, date: date, a: a, b: b}
"#;

#[tokio::test]
async fn a_metrics_semi_additive_partials_are_marked_above_the_union_below_its_formula() {
    let columns = [
        ("account", "string"),
        ("date", "date"),
        ("a", "double"),
        ("b", "double"),
    ];
    let tables = ["warehouse.daily_eu", "warehouse.daily_rest"];
    let manifest = over_tables(DAILY, &tables, &columns).await;
    let request = at_granularity("date", Granularity::Month, &["m_lin"]).with_from("daily");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "daily");
    let partial = |n: usize| format!("__plan_agg{n}_m_lin");
    let (_, union) = union_of(&plan);
    for branch in &union.inputs {
        assert_eq!(every_marker(branch), [], "no marker below the union");
        let step = aggregate_step(branch);
        assert_eq!(step.groupings, [qualified(alias, "date")]);
        let row_level = projection(&step.input);
        let partials: Vec<_> = (step.aggregates.iter())
            .map(|(entry, sum)| {
                assert_eq!(sum.function().declarations.canonical.name, "sum");
                let condition = sum.filter().map(|filter| computed(row_level, filter));
                (entry.as_str().to_owned(), condition)
            })
            .collect();
        assert_eq!(
            partials,
            [("a", 1), ("b", 2)].map(|(measure, n)| (
                format!("{alias}.{}", partial(n)),
                Some(qualified(alias, &format!("__plan_cond_{measure}")))
            )),
            "each partial reads its branch's condition alone"
        );
    }
    let source = |measure: &str| qualified(alias, &partial(if measure == "a" { 1 } else { 2 }));
    let derivation = projection(&plan.root);
    let aggregate = measure_markers(source, derivation, ("m_lin", |a, b| a - b), &["date"]);
    assert_eq!(aggregate.groupings, [column("date")]);
}

const SHOPS: &str = r#"
semantic_model:
  name: shops
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: eu_revenue, data_type: double, agg: sum}
  unionsets:
    - name: shops
      mode: all
      dimensions: [{ref: region}]
      measures:
        - ref: eu_revenue
          filter:
            eq: [region, {lit: eu}]
      datasets:
        - name: north_shops
          extras:
            storage: {tables: [warehouse.north]}
            semantic_mapping: {region: region, eu_revenue: amount}
        - name: south_shops
          extras:
            storage: {tables: [warehouse.south]}
            semantic_mapping: {region: region, eu_revenue: amount}
"#;

#[tokio::test]
async fn a_branchs_partial_reads_the_unionsets_filter_and_its_own_signature_filter() {
    let columns = [("region", "string"), ("amount", "double")];
    let tables = ["warehouse.north", "warehouse.south"];
    let manifest = over_tables(SHOPS, &tables, &columns).await;
    let request = select(&["eu_revenue"]).with_from("shops");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "shops");
    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
    let eu = |region: PhysicalExpr| region.eq(text("eu"));
    let condition = format!("{alias}.__plan_cond_eu_revenue");
    for (branch, table) in union.inputs.iter().zip(["north", "south"]) {
        let partial = aggregate_step(branch);
        let [(_, sum)] = partial.aggregates.as_slice() else {
            panic!("one partial sum, got {partial:?}");
        };
        let own = eu(qualified(alias, "region"));
        assert_eq!(
            sum.filter()
                .map(|filter| computed(projection(&partial.input), filter)),
            Some(own & qualified(alias, "__plan_cond_eu_revenue")),
            "the filter clause of `{table}`'s partial"
        );
        let signature = eu(columns_of(&manifest, table)("region"));
        let rows = projection(rows_below_partial(branch));
        assert_eq!(written(rows, &condition), Some(signature), "{table}");
    }
    let [(_, sum)] = aggregate_step(&plan.root).aggregates.as_slice() else {
        panic!("one entry, got {:?}", plan.root);
    };
    assert_eq!(sum.filter(), None, "the branches applied each filter");
}

const PAID_SHOPS: &str = r#"
semantic_model:
  name: paid_shops
  dimensions:
    - {name: region, data_type: string, type: categorical}
    - {name: region_name, data_type: string, type: categorical}
  measures:
    - name: paid_revenue
      data_type: double
      agg: sum
      filter: {eq: [{col: status}, {lit: paid}]}
    - name: paid_avg
      data_type: double
      agg: avg
      filter: {eq: [{col: status}, {lit: paid}]}
  unionsets:
    - name: shops
      mode: all
      dimensions: [{ref: region}]
      measures: [{ref: paid_revenue}, {ref: paid_avg}]
      datasets:
        - name: north
          extras:
            storage: {tables: [warehouse.north]}
        - name: south
          extras:
            storage: {tables: [warehouse.south]}
    - name: shops_j
      mode: all
      dimensions: [{ref: region}, {ref: region_name}]
      measures: [{ref: paid_revenue}]
      datasets:
        - name: south
          extras:
            storage: {tables: [warehouse.south]}
      joinsets:
        - name: north_regions
          datasets:
            - name: north
              extras:
                storage: {tables: [warehouse.north]}
            - name: regions
              extras:
                storage: {tables: [warehouse.regions]}
          relationships:
            - name: north_to_regions
              from: north
              to: regions
              on: [region]
              cardinality: many_to_one
              optional: left
"#;

async fn paid_shops() -> SemanticManifest {
    let shop: &[(&str, &str)] = &[
        ("region", "string"),
        ("status", "string"),
        ("paid_revenue", "double"),
        ("paid_avg", "double"),
    ];
    let regions: &[(&str, &str)] = &[("region", "string"), ("region_name", "string")];
    let tables = [
        ("warehouse.north", shop),
        ("warehouse.south", shop),
        ("warehouse.regions", regions),
    ];
    over_catalog(PAID_SHOPS, &tables).await
}

fn paid(manifest: &SemanticManifest, table: &str) -> PhysicalExpr {
    columns_of(manifest, table)("status").eq(text("paid"))
}

#[tokio::test]
async fn a_branchs_partial_reads_its_own_signature_filter_where_the_unionset_has_none() {
    let manifest = paid_shops().await;
    let request = select(&["region", "paid_revenue"]).with_from("shops");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "shops");
    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
    let condition = format!("{alias}.__plan_cond_paid_revenue");
    for (branch, table) in union.inputs.iter().zip(["north", "south"]) {
        let partial = aggregate_step(branch);
        let [(_, sum)] = partial.aggregates.as_slice() else {
            panic!("one partial sum, got {partial:?}");
        };
        assert_eq!(
            sum.filter()
                .map(|filter| computed(projection(&partial.input), filter)),
            Some(qualified(alias, "__plan_cond_paid_revenue")),
            "the filter clause of `{table}`'s partial"
        );
        let rows = projection(rows_below_partial(branch));
        assert_eq!(
            written(rows, &condition),
            Some(paid(&manifest, table)),
            "{table}"
        );
    }
    let [(_, sum)] = aggregate_step(&plan.root).aggregates.as_slice() else {
        panic!("one entry, got {:?}", plan.root);
    };
    assert_eq!(sum.filter(), None, "the branches applied each filter");
}

#[tokio::test]
async fn over_rows_the_aggregate_above_the_union_reads_each_branchs_own_signature_filter() {
    let manifest = paid_shops().await;
    let request = select(&["region", "paid_revenue", "paid_avg"]).with_from("shops");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "shops");
    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::Rows)
    );
    let measures = ["paid_revenue", "paid_avg"];
    for (branch, table) in union.inputs.iter().zip(["north", "south"]) {
        for measure in measures {
            let condition = format!("{alias}.__plan_cond_{measure}");
            assert_eq!(
                written(projection(branch), &condition),
                Some(paid(&manifest, table)),
                "{table}: {measure}"
            );
        }
    }
    let aggregate = aggregate_step(&plan.root);
    let clauses: Vec<_> = (aggregate.aggregates.iter())
        .map(|(_, call)| {
            let function = call.function().declarations.canonical.name;
            let filter = call.filter();
            (
                function,
                filter.map(|filter| computed(projection(&aggregate.input), filter)),
            )
        })
        .collect();
    let condition = |measure: &str| Some(qualified(alias, &format!("__plan_cond_{measure}")));
    assert_eq!(
        clauses,
        [
            ("sum", condition("paid_revenue")),
            ("avg", condition("paid_avg"))
        ]
    );
}

#[tokio::test]
async fn a_composed_branchs_dataset_reads_its_own_signature_filter_below_the_join() {
    let manifest = paid_shops().await;
    let request = select(&["region_name", "paid_revenue"]).with_from("shops_j");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "shops_j");
    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
    let [composed, south] = &union.inputs[..] else {
        panic!("two branches, got {union:?}");
    };
    let mut steps = (composed.pre_order()).filter_map(|node| match node {
        PlanNode::Aggregate(aggregate) => Some(aggregate),
        _ => None,
    });
    let own = steps.next().expect("the branch's own aggregate step");
    let [(_, sum)] = own.aggregates.as_slice() else {
        panic!("one entry, got {:?}", own.aggregates);
    };
    assert_eq!(sum.filter(), None, "the branch's own step reads no filter");
    let below: Vec<_> = (steps.flat_map(|aggregate| {
        (aggregate.aggregates.iter()).filter_map(|(_, call)| {
            let filter = call.filter()?;
            Some(computed(projection(&aggregate.input), filter))
        })
    }))
    .collect();
    assert_eq!(below, [paid(&manifest, "north")]);
    let partial = aggregate_step(south);
    let [(_, sum)] = partial.aggregates.as_slice() else {
        panic!("one partial sum, got {partial:?}");
    };
    assert_eq!(
        sum.filter()
            .map(|filter| computed(projection(&partial.input), filter)),
        Some(qualified(alias, "__plan_cond_paid_revenue"))
    );
    let condition = format!("{alias}.__plan_cond_paid_revenue");
    let rows = projection(rows_below_partial(south));
    assert_eq!(written(rows, &condition), Some(paid(&manifest, "south")));
}

const AD_PLATFORMS: &str = r#"
semantic_model:
  name: ad_platforms
  dimensions:
    - name: day
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - {name: cost, data_type: double, agg: sum}
  unionsets:
    - name: ad_platforms
      mode: all
      dimensions: [{ref: day}]
      measures: [{ref: cost}]
      datasets:
        - name: adwords
          extras:
            storage: {tables: [polaris.adwords]}
            semantic_mapping: {day: {lit: {date: "2026-05-19"}}, cost: cost}
        - name: facebook
          extras:
            storage: {tables: [polaris.facebook]}
            semantic_mapping: {day: {lit: {date: "2026-05-18"}}, cost: cost}
        - name: ga
          extras:
            storage: {tables: [polaris.ga]}
            semantic_mapping: {day: day, cost: cost}
"#;

#[tokio::test]
async fn a_branch_whose_literal_folds_the_predicate_to_false_is_not_read() {
    let columns = [("day", "date"), ("cost", "double")];
    let tables = ["polaris.adwords", "polaris.facebook", "polaris.ga"];
    let manifest = over_tables(AD_PLATFORMS, &tables, &columns).await;
    let may_18 = dim("day")
        .expect("a name")
        .eq(lit(Literal::Date("2026-05-18".into())));
    let request = select(&["cost"])
        .with_from("ad_platforms")
        .with_predicate(may_18);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(
        datasets(&plan),
        ["ad_platforms.facebook", "ad_platforms.ga"]
    );
    let (_, union) = union_of(&plan);
    let pruned = DataKindName::new("ad_platforms.adwords");
    let trace = &union.common.semantic_meta;
    assert!(
        trace.iter().any(
            |subject| matches!(subject, SemAnnotation::SemanticSource { pruned: names, .. } if names.contains(&pruned))
        ),
        "{pruned:?} in {trace:?}"
    );
}

fn one_branch_left<'p>(plan: &'p SemanticPlan, read: &str, pruned: &str) -> &'p PlanNode {
    let unions = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Multiset(_)));
    assert_eq!(unions.count(), 0, "no union over one branch");
    assert_eq!(datasets(plan), [read]);
    let pruned = DataKindName::new(pruned);
    let mut traced = (plan.root.pre_order()).filter(|node| {
        (node.common().semantic_meta.iter()).any(
            |subject| matches!(subject, SemAnnotation::SemanticSource { pruned: names, .. } if names.contains(&pruned)),
        )
    });
    let (Some(top), None) = (traced.next(), traced.next()) else {
        panic!("one node names {pruned:?} in {plan:?}");
    };
    top
}

const ADS: &str = r#"
semantic_model:
  name: ads
  dimensions:
    - name: day
      data_type: date
      type: {temporal: {granularities: [day]}}
  measures:
    - {name: cost, data_type: double, agg: sum}
  unionsets:
    - name: ads
      mode: all
      dimensions: [{ref: day}]
      measures: [{ref: cost}]
      datasets:
        - name: good
          extras:
            storage: {tables: [warehouse.good]}
            semantic_mapping: {day: {lit: {string: "2026-05-19"}}, cost: cost}
        - name: bad
          extras:
            storage: {tables: [warehouse.bad]}
            semantic_mapping: {day: {lit: {string: "2026-13-45"}}, cost: cost}
"#;

#[tokio::test]
async fn a_literal_that_the_canonical_cast_gives_no_value_keeps_its_branch() {
    let tables = ["warehouse.good", "warehouse.bad"];
    let manifest = over_tables(ADS, &tables, &[("cost", "double")]).await;
    let may_18 = dim("day")
        .expect("a name")
        .eq(lit(Literal::Date("2026-05-18".into())));
    let request = select(&["cost"]).with_from("ads").with_predicate(may_18);

    let (plan, _) = planned(&manifest, &request);

    one_branch_left(&plan, "ads.bad", "ads.good");
    let day = format!("{}.day", alias(&manifest, "ads"));
    let written_day = (plan.root.pre_order()).find_map(|node| match node {
        PlanNode::Project(project) => written(project, &day),
        _ => None,
    });
    let cast = Expr::Cast {
        input: Box::new(text("2026-13-45")),
        target: DataType::Date,
        on_failure: CastFailure::Error,
    };
    assert_eq!(written_day, Some(cast));
}

#[tokio::test]
async fn one_branch_left_in_mode_all_is_used_as_it_is_with_the_unionsets_trace() {
    let manifest = paid_media().await;
    let bing_rows = dim("traffic_source").expect("a name").eq(lit("Bing"));
    let request = select(&["country", "market", "cost"])
        .with_from("paid_media")
        .with_predicate(bing_rows);

    let (plan, _) = planned(&manifest, &request);

    let read = "paid_media.bing_campaign_data";
    let top = one_branch_left(&plan, read, "paid_media.adwords_campaign_data");
    let aggregate = aggregate_step(top);
    let paid_media = DataKindName::new("paid_media");
    let expected = SemAnnotation::SemanticSource {
        semantic_source: paid_media,
        source_role: SourceRole::Unionset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: None,
        delivery: Some(Delivery::PartialAggregates),
        answered_by: None,
        pruned: vec![DataKindName::new("paid_media.adwords_campaign_data")],
    };
    let trace = &aggregate.common.semantic_meta;
    assert!(trace.contains(&expected), "{expected:?} in {trace:?}");
}

const ALL_SALES: &str = r#"
semantic_model:
  name: all_sales
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: order_count, data_type: bigint, agg: count}
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: all_sales
      mode: all
      dimensions: [{ref: region}]
      measures: [{ref: order_count}, {ref: revenue}]
      datasets:
        - name: store_sales
          extras:
            storage: {tables: [warehouse.store_sales]}
            semantic_mapping: {region: {lit: {string: emea}}, order_count: sale_id, revenue: amount}
        - name: web_sales
          extras:
            storage: {tables: [warehouse.web_sales]}
            semantic_mapping: {region: {lit: {string: amer}}, order_count: sale_id, revenue: amount}
"#;

fn every_branch_excluded(
    manifest: &SemanticManifest,
    name: &str,
) -> (SemanticPlan, SemanticPlan, String) {
    let request = select(&[name]).with_from("all_sales");
    let apac = dim("region").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(manifest, &request.clone().with_predicate(apac));
    let (usual, _) = planned(manifest, &request);

    let empty = scan_of(&plan);
    assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
    assert_eq!(datasets(&plan), [] as [&str; 0], "no branch is read");
    let (_, union) = union_of(&usual);
    assert_eq!(
        empty.common.output_schema, union.common.output_schema,
        "the empty read is typed as the union's output"
    );
    let pruned = ["all_sales.store_sales", "all_sales.web_sales"].map(DataKindName::new);
    let trace = &empty.common.semantic_meta;
    assert!(
        trace.iter().any(|subject| matches!(
            subject,
            SemAnnotation::SemanticSource { pruned: names, .. } if pruned.iter().all(|name| names.contains(name))
        )),
        "{pruned:?} in {trace:?}"
    );
    (plan, usual, alias(manifest, "all_sales"))
}

#[tokio::test]
async fn a_count_over_every_branch_excluded_reads_the_empty_read_coalesced_with_zero() {
    let manifest = all_sales(ALL_SALES).await;
    let (plan, usual, alias) = every_branch_excluded(&manifest, "order_count");

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [], "one row");
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        qualified(&alias, "order_count")
    );
    let zero = Expr::Literal(Literal::BigInt(0));
    assert_eq!(
        derivation.expressions,
        [call("coalesce", vec![column(entry.as_str()), zero])]
    );
    assert_eq!(derivation.expressions, projection(&usual.root).expressions);
}

#[tokio::test]
async fn a_sum_over_every_branch_excluded_reads_the_empty_read() {
    let manifest = all_sales(ALL_SALES).await;
    let (plan, usual, alias) = every_branch_excluded(&manifest, "revenue");

    let aggregate = aggregate_step(&plan.root);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        qualified(&alias, "revenue")
    );
    assert_eq!(aggregate.aggregates, aggregate_step(&usual.root).aggregates);
}

const UNIQUE_SALES: &str = r#"
semantic_model:
  name: all_sales
  dimensions:
    - {name: sale_id, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: all_sales
      mode: unique
      dimensions: [{ref: sale_id, is_key: true}]
      measures: [{ref: revenue}]
      datasets:
        - name: store_sales
          extras:
            storage: {tables: [warehouse.store_sales]}
            semantic_mapping: {sale_id: sale_id, revenue: amount}
          dimensions: [{ref: sale_id, is_key: true}]
          measures: [{ref: revenue}]
        - name: web_sales
          extras:
            storage: {tables: [warehouse.web_sales]}
            semantic_mapping: {sale_id: sale_id, revenue: amount}
          dimensions: [{ref: sale_id, is_key: true}]
          measures: [{ref: revenue}]
"#;

async fn all_sales(yaml: &str) -> SemanticManifest {
    let columns = [("sale_id", "string"), ("amount", "double")];
    let tables = ["warehouse.store_sales", "warehouse.web_sales"];
    over_tables(yaml, &tables, &columns).await
}

fn compared<'p>(plan: &'p SemanticPlan, alias: &str) -> (&'p MultisetNode, Vec<String>) {
    let (_, union) = union_of(plan);
    let fields = names(&union.common.output_schema);
    for branch in &union.inputs {
        assert_eq!(names(&branch.common().output_schema), fields, "{branch:?}");
    }
    let unqualified = (fields.iter())
        .map(|field| {
            field
                .strip_prefix(&format!("{alias}."))
                .unwrap_or(field)
                .to_owned()
        })
        .collect();
    (union, unqualified)
}

#[tokio::test]
async fn a_unique_union_compares_its_keyset_and_the_aggregate_above_does_not_group_by_it() {
    let manifest = all_sales(UNIQUE_SALES).await;
    let request = select(&["revenue"]).with_from("all_sales");

    let (plan, _) = planned(&manifest, &request);

    let (union, fields) = compared(&plan, &alias(&manifest, "all_sales"));
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(fields, ["__plan_cond_revenue", "revenue", "sale_id"]);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::Rows)
    );
    assert_eq!(aggregate_steps(&plan.root), 1, "one aggregate in {plan:?}");
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [], "no grouping by `sale_id`");
    assert_eq!(output_names(&plan), ["revenue"]);
}

#[tokio::test]
async fn a_unique_union_with_no_keyset_compares_only_the_names_of_its_demand() {
    let no_keyset = UNIQUE_SALES.replace(", is_key: true", "");
    let manifest = all_sales(&no_keyset).await;
    let request = select(&["revenue"]).with_from("all_sales");

    let (plan, _) = planned(&manifest, &request);

    let (union, fields) = compared(&plan, &alias(&manifest, "all_sales"));
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(fields, ["__plan_cond_revenue", "revenue"]);
}

const UNIQUE_SPEND: &str = r#"
semantic_model:
  name: spend_all
  dimensions:
    - {name: campaign_id, data_type: string, type: categorical}
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, week, month, quarter, year]
  measures:
    - {name: cost, data_type: bigint, agg: sum}
  unionsets:
    - name: spend_all
      mode: unique
      dimensions: [{ref: campaign_id, is_key: true}, {ref: date, is_key: true}]
      measures: [{ref: cost}]
      datasets:
        - name: web
          extras:
            storage: {tables: [warehouse.spend_web]}
            semantic_mapping: {campaign_id: campaign_id, date: date, cost: cost}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
          dimensions: [{ref: campaign_id, is_key: true}, {ref: date, is_key: true}]
          measures: [{ref: cost}]
        - name: app
          extras:
            storage: {tables: [warehouse.spend_app]}
            semantic_mapping: {campaign_id: campaign_id, date: date, cost: cost}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
          dimensions: [{ref: campaign_id, is_key: true}, {ref: date, is_key: true}]
          measures: [{ref: cost}]
"#;

#[tokio::test]
async fn a_unique_union_compares_the_days_below_the_bucket() {
    let manifest = spend_all(UNIQUE_SPEND).await;
    let request = at_granularity("date", Granularity::Month, &["cost"]).with_from("spend_all");

    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "spend_all");
    let (union, fields) = compared(&plan, alias);
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(fields, ["campaign_id", "__plan_cond_cost", "cost", "date"]);
    let date = format!("{alias}.date");
    for (branch, table) in union.inputs.iter().zip(["spend_app", "spend_web"]) {
        let day = columns_of(&manifest, table)("date");
        assert_eq!(written(projection(branch), &date), Some(day), "{table}");
    }
    assert_eq!(aggregate_steps(&plan.root), 1, "one aggregate in {plan:?}");
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    let bucketing = projection(&aggregate.input);
    assert_eq!(
        written(bucketing, "date"),
        Some(call("date_trunc", vec![text("month"), column("date")]))
    );
    let row_level = projection(&bucketing.input);
    assert_eq!(written(row_level, "date"), Some(qualified(alias, "date")));
    assert!(matches!(row_level.input.as_ref(), PlanNode::Multiset(_)));
}

async fn unique_all_sales() -> SemanticManifest {
    all_sales(&ALL_SALES.replace("mode: all", "mode: unique")).await
}

#[tokio::test]
async fn one_branch_left_in_mode_unique_is_read_under_a_distinct() {
    let manifest = unique_all_sales().await;
    let emea = dim("region").expect("a name").eq(lit("emea"));
    let request = select(&["revenue"])
        .with_from("all_sales")
        .with_predicate(emea);

    let (plan, _) = planned(&manifest, &request);

    let top = one_branch_left(&plan, "all_sales.store_sales", "all_sales.web_sales");
    let distinct = aggregate_step(top);
    assert_eq!(distinct.aggregates, [], "a distinct");
    let every_field: Vec<_> = (names(&distinct.input.common().output_schema).into_iter())
        .map(column)
        .collect();
    assert_eq!(distinct.groupings, every_field);
    assert_eq!(
        delivery_of(&distinct.common.semantic_meta),
        Some(Delivery::Rows)
    );
    assert_eq!(aggregate_steps(&plan.root), 2, "the sum above the distinct");
}

#[tokio::test]
async fn a_unique_union_with_every_branch_excluded_aggregates_the_empty_read_once() {
    let manifest = unique_all_sales().await;
    for (name, function) in [("order_count", "count"), ("revenue", "sum")] {
        let (plan, usual, alias) = every_branch_excluded(&manifest, name);

        assert_eq!(aggregate_steps(&plan.root), 1, "one aggregate in {plan:?}");
        let aggregate = aggregate_step(&plan.root);
        let [(_, aggregation)] = aggregate.aggregates.as_slice() else {
            panic!("one entry, got {:?}", aggregate.aggregates);
        };
        assert_eq!(aggregation.function().declarations.canonical.name, function);
        assert_eq!(
            computed(projection(&aggregate.input), &aggregation.args()[0]),
            qualified(&alias, name)
        );
        assert_eq!(aggregate.aggregates, aggregate_step(&usual.root).aggregates);
    }
}

const SPEND_ALL: &str = r#"
semantic_model:
  name: spend_all
  dimensions:
    - {name: campaign_id, data_type: string, type: categorical}
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, week, month, quarter, year]
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - {name: avg_cost, data_type: double, agg: avg}
  unionsets:
    - name: spend_all
      mode: all
      dimensions: [{ref: campaign_id}, {ref: date}, {ref: channel}]
      measures: [{ref: cost}, {ref: avg_cost}]
      datasets:
        - name: web
          extras:
            storage: {tables: [warehouse.spend_web]}
            semantic_mapping:
              {campaign_id: campaign_id, date: date, channel: {lit: web}, cost: cost, avg_cost: cost}
        - name: app
          extras:
            storage: {tables: [warehouse.spend_app]}
            semantic_mapping:
              {campaign_id: campaign_id, date: date, channel: {lit: app}, cost: cost, avg_cost: cost}
"#;

async fn spend_all(yaml: &str) -> SemanticManifest {
    let columns = [
        ("campaign_id", "string"),
        ("date", "date"),
        ("cost", "long"),
    ];
    let tables = ["warehouse.spend_web", "warehouse.spend_app"];
    over_tables(yaml, &tables, &columns).await
}

#[tokio::test]
async fn disjoint_branches_give_final_aggregates_with_nothing_above_the_union() {
    let manifest = spend_all(SPEND_ALL).await;
    let request = select(&["channel", "avg_cost"]).with_from("spend_all");

    let (plan, _) = planned(&manifest, &request);

    let PlanNode::Multiset(union) = &plan.root else {
        panic!("the union at the root, got {:?}", plan.root);
    };
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::FinalAggregates)
    );
    assert_eq!(union.op, MultisetOp::UnionAll);
    assert_eq!(output_names(&plan), ["channel", "avg_cost"]);
    assert_eq!(union.inputs.len(), 2);
    for branch in &union.inputs {
        assert_eq!(aggregate_steps(branch), 1, "one aggregate in {branch:?}");
        let aggregate = aggregate_step(branch);
        assert_eq!(aggregate.groupings, [column("channel")]);
        let [(_, avg)] = aggregate.aggregates.as_slice() else {
            panic!("one entry, got {:?}", aggregate.aggregates);
        };
        assert_eq!(avg.function().declarations.canonical.name, "avg");
    }
}

#[tokio::test]
async fn branches_that_a_bucketed_dimension_separates_give_rows_under_one_aggregate() {
    let yaml = (SPEND_ALL.replacen("date: date", r#"date: {lit: {date: "2026-09-01"}}"#, 1))
        .replacen("date: date", r#"date: {lit: {date: "2026-09-02"}}"#, 1);
    let manifest = spend_all(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["avg_cost"]).with_from("spend_all");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::Rows)
    );
    assert_eq!(aggregate_steps(&plan.root), 1, "one aggregate in {plan:?}");
    for branch in &union.inputs {
        assert_eq!(aggregate_steps(branch), 0, "rows from {branch:?}");
    }
}

#[tokio::test]
async fn branches_whose_literals_the_fold_does_not_compare_are_not_disjoint() {
    let yaml = (SPEND_ALL.replacen(
        "{name: channel, data_type: string,",
        "{name: channel, data_type: 'decimal(38,1)',",
        1,
    ))
    .replacen("{lit: web}", r#"{lit: {decimal: "1.5"}}"#, 1)
    .replacen("{lit: app}", r#"{lit: {decimal: "2.5"}}"#, 1);
    let manifest = spend_all(&yaml).await;
    let request = select(&["channel", "cost"]).with_from("spend_all");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
}

#[tokio::test]
async fn a_disjoint_branchs_final_aggregate_reads_the_unionsets_filter_and_its_own() {
    let yaml = SPEND_ALL.replacen(
        "{ref: avg_cost}]",
        "{ref: avg_cost, filter: {eq: [campaign_id, {lit: A}]}}]",
        1,
    );
    let manifest = spend_all(&yaml).await;
    let request = select(&["channel", "avg_cost"]).with_from("spend_all");

    let (plan, _) = planned(&manifest, &request);

    let PlanNode::Multiset(union) = &plan.root else {
        panic!("the union at the root, got {:?}", plan.root);
    };
    let alias = &alias(&manifest, "spend_all");
    let campaign_a = |campaign_id: PhysicalExpr| campaign_id.eq(text("A"));
    let condition = format!("{alias}.__plan_cond_avg_cost");
    for (branch, table) in union.inputs.iter().zip(["spend_app", "spend_web"]) {
        let aggregate = aggregate_step(branch);
        let [(_, avg)] = aggregate.aggregates.as_slice() else {
            panic!("one entry, got {:?}", aggregate.aggregates);
        };
        let row_level = projection(&aggregate.input);
        let own = campaign_a(qualified(alias, "campaign_id"));
        assert_eq!(
            avg.filter().map(|filter| computed(row_level, filter)),
            Some(own & qualified(alias, "__plan_cond_avg_cost")),
            "the filter clause of `{table}`'s avg"
        );
        let signature = campaign_a(columns_of(&manifest, table)("campaign_id"));
        let rows = projection(&row_level.input);
        assert_eq!(written(rows, &condition), Some(signature), "{table}");
    }
}

fn with_spend_ratio() -> String {
    let metric = "  metrics:\n    - name: spend_ratio\n      data_type: double\n      expr:\n        divide: [cost, avg_cost]\n  unionsets:";
    let measures = "measures: [{ref: cost}, {ref: avg_cost}]";
    (SPEND_ALL.replacen("  unionsets:", metric, 1)).replacen(
        measures,
        &format!("{measures}\n      metrics: [{{ref: spend_ratio}}]"),
        1,
    )
}

#[tokio::test]
async fn a_composed_metric_over_disjoint_branches_runs_its_formula_above_the_union() {
    let manifest = spend_all(&with_spend_ratio()).await;
    let over = metric("spend_ratio").expect("a name").gt(lit(1));
    let request = select(&["channel", "spend_ratio"])
        .with_from("spend_all")
        .with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let above = filter(&plan.root);
    let derivation = projection(&above.input);
    let PlanNode::Multiset(union) = derivation.input.as_ref() else {
        panic!("the union below the formula, got {:?}", derivation.input);
    };
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::FinalAggregates)
    );
    let formula = written(derivation, "spend_ratio").expect("the formula");
    assert_ne!(formula, column("spend_ratio"), "computed above the union");
    for branch in &union.inputs {
        assert_eq!(aggregate_steps(branch), 1, "one aggregate in {branch:?}");
        let aggregate = aggregate_step(branch);
        let functions: Vec<_> = (aggregate.aggregates.iter())
            .map(|(_, call)| call.function().declarations.canonical.name)
            .collect();
        assert_eq!(functions, ["sum", "avg"]);
    }
    assert_eq!(
        aggregate_steps(&plan.root),
        2,
        "no aggregate above the union"
    );
    assert_eq!(output_names(&plan), ["channel", "spend_ratio"]);
}

fn with_cost_metrics() -> String {
    let metrics = "  metrics:
    - name: cost_adj
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [campaign_id, {lit: c1}]}
              then: {add: [cost, cost]}
          else: cost
    - {name: max_cost, data_type: bigint, agg: max, expr: cost}
  unionsets:";
    let measures = "measures: [{ref: cost}, {ref: avg_cost}]";
    (SPEND_ALL.replacen("  unionsets:", metrics, 1)).replacen(
        measures,
        &format!("{measures}\n      metrics: [{{ref: cost_adj}}, {{ref: max_cost}}]"),
        1,
    )
}

#[tokio::test]
async fn a_step_1_finer_than_the_demand_reads_partials_of_disjoint_branches() {
    let manifest = spend_all(&with_cost_metrics()).await;

    for names in [
        &["channel", "cost_adj"][..],
        &["channel", "cost_adj", "max_cost"],
    ] {
        let request = select(names).with_from("spend_all");
        let (plan, _) = planned(&manifest, &request);

        let (_, union) = union_of(&plan);
        let delivery = branch_delivery(union);
        assert_eq!(delivery, Delivery::PartialAggregates, "{names:?}");
        for branch in &union.inputs {
            let (grouped, _) = branch_step(branch);
            assert_eq!(grouped, ["campaign_id", "channel"], "{names:?}");
        }
        assert_eq!(output_names(&plan), names);
    }
}

#[tokio::test]
async fn a_step_1_at_the_demand_reads_final_aggregates_as_fields_above_disjoint_branches() {
    let manifest = spend_all(&with_cost_metrics()).await;
    let request = select(&["channel", "max_cost"]).with_from("spend_all");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::FinalAggregates);
    for branch in &union.inputs {
        let (grouped, entries) = branch_step(branch);
        assert_eq!(grouped, ["channel"]);
        assert_eq!(entries, ["__plan_agg1_max_cost"]);
    }
    assert_eq!(unions(&plan).len(), 1, "one read of the union: {plan:?}");
    assert_eq!(joins(&plan), 0, "no join: {plan:?}");
    let max = first_aggregate(&plan.root);
    let [(_, call)] = max.aggregates.as_slice() else {
        panic!("step 2's one entry, got {:?}", max.aggregates);
    };
    assert_eq!(call.function().declarations.canonical.name, "max");
    assert_eq!(steps_above_unions(&plan), 1, "step 2 alone: {plan:?}");
    assert_eq!(output_names(&plan), ["channel", "max_cost"]);
}

#[tokio::test]
async fn a_predicate_on_a_measure_sits_in_each_disjoint_branch_above_its_aggregate() {
    let manifest = spend_all(SPEND_ALL).await;
    let over = measure("cost").expect("a name").gt(lit(10));
    let request = select(&["channel", "avg_cost"])
        .with_from("spend_all")
        .with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let top = match &plan.root {
        PlanNode::Project(project) => project.input.as_ref(),
        other => other,
    };
    let PlanNode::Multiset(union) = top else {
        panic!("the union at the root, got {:?}", plan.root);
    };
    for branch in &union.inputs {
        let on_cost = filter(branch);
        assert!(
            matches!(on_cost.input.as_ref(), PlanNode::Aggregate(_)),
            "{branch:?}"
        );
    }
    assert_eq!(
        aggregate_steps(&plan.root),
        2,
        "no aggregate above the union"
    );
}

#[tokio::test]
async fn a_predicate_on_a_cast_measure_sits_above_the_cast_in_each_disjoint_branch() {
    let yaml = (ORDERS.replacen("{region: region,", "{region: {lit: eu},", 1)).replacen(
        "{order_count:",
        "{region: {lit: us}, order_count:",
        1,
    );
    let manifest = over_tables(&yaml, &ORDER_TABLES, &ORDER_COLUMNS).await;
    let over = || measure("revenue").expect("a name").gt(lit(100));
    let request = select(&["region", "revenue"])
        .with_from("orders")
        .with_predicate(over());

    let (plan, _) = planned(&manifest, &request);

    let top = match &plan.root {
        PlanNode::Project(project) => project.input.as_ref(),
        other => other,
    };
    let PlanNode::Multiset(union) = top else {
        panic!("the union at the root, got {:?}", plan.root);
    };
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::FinalAggregates)
    );
    for branch in &union.inputs {
        let finished = projection(&filter(branch).input);
        let [(entry, sum)] = aggregate_step(&finished.input).aggregates.as_slice() else {
            panic!("one entry, got {branch:?}");
        };
        assert_eq!(sum.function().declarations.canonical.name, "sum");
        let cast = Expr::Cast {
            input: Box::new(column(entry.as_str())),
            target: DataType::Decimal {
                precision: 18,
                scale: 2,
            },
            on_failure: CastFailure::Error,
        };
        assert_eq!(written(finished, "revenue"), Some(cast), "{branch:?}");
    }
    assert_eq!(
        aggregate_steps(&plan.root),
        2,
        "no aggregate above the union"
    );
    assert_eq!(output_names(&plan), ["region", "revenue"]);

    let unselected = select(&["region", "order_count"])
        .with_from("orders")
        .with_predicate(over());
    let (plan, _) = planned(&manifest, &unselected);
    assert_eq!(output_names(&plan), ["region", "order_count"]);
}

const SALES_CUSTOMERS: &str = r#"
semantic_model:
  name: sales_customers
  dimensions:
    - {name: sale_id, data_type: string, type: categorical}
    - {name: customer_id, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
    - {name: d, data_type: bigint, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  joinsets:
    - name: sales_customers
      dimensions: [{ref: sale_id}, {ref: customer_id}, {ref: region}, {ref: channel}, {ref: d}]
      measures: [{ref: revenue}]
      unionsets:
        - name: all_sales
          mode: unique
          datasets:
            - name: web_sales
              extras:
                storage: {tables: [warehouse.web_sales]}
                semantic_mapping:
                  {sale_id: sale_id, customer_id: customer_id, channel: {lit: web}, d: d, revenue: amount}
            - name: store_sales
              extras:
                storage: {tables: [warehouse.store_sales]}
                semantic_mapping:
                  {sale_id: sale_id, customer_id: customer_id, channel: {lit: store}, d: d, revenue: amount}
      datasets:
        - name: customers
          extras:
            storage: {tables: [warehouse.customers]}
      relationships:
        - name: sales_to_customers
          from: all_sales
          to: customers
          on: [customer_id]
          cardinality: many_to_one
          optional: none
          integrity: none
"#;

async fn sales_customers(yaml: &str) -> SemanticManifest {
    let sales = &[
        ("sale_id", "string"),
        ("customer_id", "string"),
        ("d", "long"),
        ("amount", "double"),
    ][..];
    let tables = [
        ("warehouse.web_sales", sales),
        ("warehouse.store_sales", sales),
        (
            "warehouse.customers",
            &[("customer_id", "string"), ("region", "string")],
        ),
    ];
    over_catalog(yaml, &tables).await
}

#[tokio::test]
async fn a_unique_union_under_a_built_join_compares_the_field_of_its_on() {
    let manifest = sales_customers(SALES_CUSTOMERS).await;

    let (plan, _) = planned(&manifest, &select(&["revenue"]));

    assert_eq!(the_join(&plan).join_type, JoinType::Inner);
    let (union, fields) = compared(&plan, &alias(&manifest, "sales_customers.all_sales"));
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(fields, ["customer_id", "__plan_cond_revenue", "revenue"]);
}

fn the_join(plan: &SemanticPlan) -> &JoinNode {
    let mut joins = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Join(_)));
    let (Some(built), None) = (joins.next(), joins.next()) else {
        panic!("one join in {plan:?}");
    };
    join(built)
}

fn traced_with(node: &PlanNode, matches: impl Fn(&SemAnnotation) -> bool) -> &PlanNode {
    (node.pre_order())
        .find(|node| node.common().semantic_meta.iter().any(&matches))
        .unwrap_or_else(|| panic!("no node with the expected trace, in {node:?}"))
}

fn is_pruned(name: &DataKindName) -> impl Fn(&SemAnnotation) -> bool + '_ {
    move |subject| matches!(subject, SemAnnotation::SemanticSource { pruned, .. } if pruned.contains(name))
}

fn assumed() -> String {
    SALES_CUSTOMERS.replace("integrity: none", "integrity: assumed")
}

#[tokio::test]
async fn a_unique_union_whose_joinset_prunes_the_other_operand_compares_only_its_demand() {
    let manifest = sales_customers(&assumed()).await;

    let (plan, _) = planned(&manifest, &select(&["revenue"]));

    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Join(_))));
    let (union, fields) = compared(&plan, &alias(&manifest, "sales_customers.all_sales"));
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(fields, ["__plan_cond_revenue", "revenue"]);
}

#[tokio::test]
async fn a_unique_union_whose_joinset_prunes_the_other_operand_compares_its_keyset() {
    let keyed = (assumed()).replace(
        "[{ref: sale_id}, {ref: customer_id}, {ref: region},",
        "[{ref: sale_id, is_key: true, keysets: [sale]}, \
         {ref: customer_id, is_key: true, keysets: [customer]}, \
         {ref: region, is_key: true, keysets: [customer]},",
    );
    let manifest = sales_customers(&keyed).await;

    let (plan, _) = planned(&manifest, &select(&["revenue"]));

    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Join(_))));
    let (union, fields) = compared(&plan, &alias(&manifest, "sales_customers.all_sales"));
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(fields, ["__plan_cond_revenue", "revenue", "sale_id"]);
}

#[tokio::test]
async fn a_unique_union_under_a_join_compares_the_field_of_its_relationship_filter() {
    let filtered = SALES_CUSTOMERS.replace(
        "          integrity: none\n",
        "          integrity: none\n          filter: {gt: [d, {lit: 10}]}\n",
    );
    let manifest = sales_customers(&filtered).await;

    let (plan, _) = planned(&manifest, &select(&["revenue"]));

    let all_sales = &alias(&manifest, "sales_customers.all_sales");
    let over_ten = qualified(all_sales, "d").gt(Expr::Literal(Literal::BigInt(10)));
    assert_eq!(the_join(&plan).post_join_filter, Some(over_ten));
    let (union, fields) = compared(&plan, all_sales);
    assert_eq!(union.op, MultisetOp::UnionDistinct);
    assert_eq!(
        fields,
        ["customer_id", "d", "__plan_cond_revenue", "revenue"]
    );
}

#[tokio::test]
async fn a_predicate_prunes_a_branch_of_a_unique_union_below_a_join() {
    let manifest = sales_customers(SALES_CUSTOMERS).await;
    let web = dim("channel").expect("a name").eq(lit("web"));

    let (plan, _) = planned(&manifest, &select(&["revenue"]).with_predicate(web));

    assert_eq!(
        datasets(&plan),
        [
            "sales_customers.all_sales.web_sales",
            "sales_customers.customers"
        ]
    );
    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Multiset(_))));
    let pruned = DataKindName::new("sales_customers.all_sales.store_sales");
    traced_with(&the_join(&plan).left, is_pruned(&pruned));
}

const CAMPAIGNS_SPEND_ALL: &str = r#"
semantic_model:
  name: campaigns_spend_all
  dimensions:
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: campaign_name, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: budget, data_type: bigint, agg: sum}
    - {name: cost, data_type: bigint, agg: sum}
    - {name: avg_cost, data_type: double, agg: avg}
  joinsets:
    - name: campaigns_spend_all
      dimensions: [{ref: campaign_id}, {ref: campaign_name}, {ref: channel}]
      measures: [{ref: budget}, {ref: cost}, {ref: avg_cost}]
      datasets:
        - name: campaigns
          extras:
            storage: {tables: [warehouse.campaigns]}
      unionsets:
        - name: spend_all
          mode: all
          datasets:
            - name: web
              extras:
                storage: {tables: [warehouse.spend_web]}
                semantic_mapping:
                  {campaign_id: campaign_id, channel: {lit: web}, cost: cost, avg_cost: cost}
            - name: app
              extras:
                storage: {tables: [warehouse.spend_app]}
                semantic_mapping:
                  {campaign_id: campaign_id, channel: {lit: app}, cost: cost, avg_cost: cost}
      relationships:
        - name: campaigns_to_spend
          from: campaigns
          to: spend_all
          on: [campaign_id]
          cardinality: one_to_many
          optional: none
          integrity: assumed
"#;

async fn campaigns_spend_all(names: &[&str]) -> (SemanticPlan, String) {
    let spend = &[("campaign_id", "string"), ("cost", "long")][..];
    let tables = [
        ("warehouse.spend_web", spend),
        ("warehouse.spend_app", spend),
        (
            "warehouse.campaigns",
            &[
                ("campaign_id", "string"),
                ("campaign_name", "string"),
                ("budget", "long"),
            ],
        ),
    ];
    let manifest = over_catalog(CAMPAIGNS_SPEND_ALL, &tables).await;
    let (plan, _) = planned(&manifest, &select(names));
    (plan, alias(&manifest, "campaigns_spend_all.spend_all"))
}

fn first_union(node: &PlanNode) -> &MultisetNode {
    (node.pre_order())
        .find_map(|node| match node {
            PlanNode::Multiset(union) => Some(union),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a union in {node:?}"))
}

fn union_right_of_the_join(plan: &SemanticPlan) -> &MultisetNode {
    first_union(&the_join(plan).right)
}

#[tokio::test]
async fn a_unionset_under_a_join_gives_rows_where_the_joinset_demands_rows() {
    let (plan, _) = campaigns_spend_all(&["channel", "campaign_name", "avg_cost"]).await;

    let union = union_right_of_the_join(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::Rows)
    );
    assert_eq!(aggregate_steps(&the_join(&plan).right), 0);
    assert_eq!(
        output_names(&plan),
        ["channel", "campaign_name", "avg_cost"]
    );
}

#[tokio::test]
async fn a_unionset_under_a_join_gives_partial_aggregates_per_join_key() {
    let (plan, spend_all) = campaigns_spend_all(&["campaign_name", "budget", "cost"]).await;

    let union = union_right_of_the_join(&plan);
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
    let per_campaign = aggregate_step(&the_join(&plan).right);
    assert_eq!(
        per_campaign.groupings,
        [qualified(&spend_all, "campaign_id")]
    );
    assert_eq!(output_names(&plan), ["campaign_name", "budget", "cost"]);
}

const ALL_SPEND3: &str = r#"
semantic_model:
  name: all_spend3
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
  unionsets:
    - name: all_spend3
      mode: all
      dimensions: [{ref: date}]
      measures: [{ref: cost}, {ref: budget}]
      grainsets:
        - name: spend_g3
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: g3_daily
              extras:
                storage: {tables: [warehouse.g3_daily]}
                semantic_mapping: {date: day, cost: cost}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: g3_monthly
              extras:
                storage: {tables: [warehouse.g3_monthly]}
                semantic_mapping: {date: month_start, cost: cost, budget: budget}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
      datasets:
        - name: spend_c3
          extras:
            storage: {tables: [warehouse.spend_c3]}
            semantic_mapping: {date: day, cost: cost, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

async fn all_spend3() -> SemanticManifest {
    let daily: &[(&str, &str)] = &[("day", "date"), ("cost", "long"), ("budget", "long")];
    let tables = [
        (
            "warehouse.g3_daily",
            &[("day", "date"), ("cost", "long")][..],
        ),
        (
            "warehouse.g3_monthly",
            &[
                ("month_start", "date"),
                ("cost", "long"),
                ("budget", "long"),
            ],
        ),
        ("warehouse.spend_c3", daily),
    ];
    over_catalog(ALL_SPEND3, &tables).await
}

async fn budget_at_day() -> Vec<Golden> {
    let manifest = all_spend3().await;
    let request = at_granularity("date", Granularity::Day, &["budget"]);
    let names = ["all_spend3", "all_spend3.spend_g3"];
    let held = (Some(Granularity::Day), Some(Granularity::Month));
    let nearest: [(&str, &[&str]); 2] = [
        ("all_spend3.spend_g3.g3_monthly", &[]),
        ("all_spend3.spend_g3.g3_daily", &["budget"]),
    ];
    let refused = grain_not_servable(&manifest, &request, &names, held, &nearest);
    vec![Golden::message("budget_at_day", &refused)]
}

#[tokio::test]
async fn a_name_in_a_grainset_branchs_mask_that_no_child_serves_is_never_null_filled() {
    for golden in budget_at_day().await {
        golden.assert_committed();
    }
}

fn two_branches(plan: &SemanticPlan) -> [&PlanNode; 2] {
    let union = first_union(&plan.root);
    let [first, second] = &union.inputs[..] else {
        panic!("two branches, in name order, got {union:?}");
    };
    [first, second]
}

#[tokio::test]
async fn a_grainset_branch_that_one_child_serves_is_answered_by_that_child_alone() {
    let manifest = all_spend3().await;
    let request = at_granularity("date", Granularity::Month, &["budget"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(
        datasets(&plan),
        ["all_spend3.spend_c3", "all_spend3.spend_g3.g3_monthly"]
    );
    let combined = |node: &PlanNode| matches!(node, PlanNode::Join(_) | PlanNode::Cross(_));
    assert!(!plan.root.pre_order().any(combined), "no drill-across");
    let [_, spend_g3] = two_branches(&plan);
    let rows = rows_below_partial(spend_g3);
    let trace = &rows.common().semantic_meta;
    let semantic_source = DataKindName::new("all_spend3.spend_g3");
    let answered_by = DataKindName::new("all_spend3.spend_g3.g3_monthly");
    assert!(
        trace.iter().any(|subject| matches!(
            subject,
            SemAnnotation::SemanticSource { semantic_source: name, answered_by: Some(answer), .. }
                if *name == semantic_source && *answer == answered_by
        )),
        "{semantic_source:?} answered by {answered_by:?} in {trace:?}"
    );
    let alias = &alias(&manifest, "all_spend3");
    let monthly = columns_of(&manifest, "g3_monthly");
    assert_eq!(
        written(row_projection(rows), &format!("{alias}.budget")),
        Some(monthly("budget"))
    );
}

const SALES_STORES: &str = r#"
semantic_model:
  name: all_sales
  dimensions:
    - {name: channel, data_type: string, type: categorical}
    - {name: store_id, data_type: string, type: categorical}
    - {name: store_name, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: all_sales
      mode: all
      dimensions: [{ref: channel}, {ref: store_id}, {ref: store_name}]
      measures: [{ref: revenue}]
      datasets:
        - name: web_sales
          extras:
            storage: {tables: [warehouse.web_sales]}
            semantic_mapping: {channel: {lit: web}, revenue: amount}
      joinsets:
        - name: store_sales
          datasets:
            - name: store_orders
              extras:
                storage: {tables: [warehouse.store_orders]}
                semantic_mapping: {channel: {lit: store}, store_id: store_id, revenue: amount}
            - name: stores
              extras:
                storage: {tables: [warehouse.stores]}
                semantic_mapping: {store_id: store_id, store_name: store_name}
          relationships:
            - name: orders_to_stores
              from: store_orders
              to: stores
              on: [store_id]
              cardinality: many_to_one
              optional: none
              integrity: assumed
"#;

async fn sales_stores() -> SemanticManifest {
    let tables = [
        ("warehouse.web_sales", &[("amount", "double")][..]),
        (
            "warehouse.store_orders",
            &[("store_id", "string"), ("amount", "double")],
        ),
        (
            "warehouse.stores",
            &[("store_id", "string"), ("store_name", "string")],
        ),
    ];
    over_catalog(SALES_STORES, &tables).await
}

#[tokio::test]
async fn a_joinset_branch_is_a_frame_of_its_own_under_the_unions_alias() {
    let manifest = sales_stores().await;

    let (plan, _) = planned(&manifest, &select(&["channel", "revenue"]));

    let [store_sales, _] = two_branches(&plan);
    let pruned = DataKindName::new("all_sales.store_sales.stores");
    let joinset = DataKindName::new("all_sales.store_sales");
    let rows = traced_with(store_sales, |subject| {
        matches!(
            subject,
            SemAnnotation::SemanticSource { semantic_source, pruned: names, .. }
                if *semantic_source == joinset && names.contains(&pruned)
        )
    });
    let alias = &alias(&manifest, "all_sales");
    let written = |name: &str| written(row_projection(rows), &format!("{alias}.{name}"));
    assert_eq!(written("channel"), Some(text("store")));
    assert_eq!(
        written("revenue"),
        Some(columns_of(&manifest, "store_orders")("amount"))
    );
}

async fn device_sales_stores() -> SemanticManifest {
    let edits = [
        (
            "    - {name: store_name, data_type: string, type: categorical}\n",
            "    - {name: store_name, data_type: string, type: categorical}\n    - {name: device, data_type: string, type: categorical}\n",
        ),
        ("{ref: store_name}]", "{ref: store_name}, {ref: device}]"),
        ("{lit: web}, revenue", "{lit: web}, device: device, revenue"),
    ];
    let yaml = (edits.iter()).fold(SALES_STORES.to_owned(), |yaml, (from, to)| {
        yaml.replace(from, to)
    });
    let tables = [
        (
            "warehouse.web_sales",
            &[("device", "string"), ("amount", "double")][..],
        ),
        (
            "warehouse.store_orders",
            &[("store_id", "string"), ("amount", "double")],
        ),
        (
            "warehouse.stores",
            &[("store_id", "string"), ("store_name", "string")],
        ),
    ];
    over_catalog(&yaml, &tables).await
}

#[tokio::test]
async fn a_predicate_on_a_name_that_a_joinset_branch_null_fills_stays_on_its_rows() {
    let manifest = device_sales_stores().await;
    let name = |name| dim(name).expect("a name");
    let predicate = name("store_name").eq(lit("S1")) | name("device").is_null();
    let request = select(&["store_name", "revenue"]).with_predicate(predicate);

    let (plan, _) = planned(&manifest, &request);

    let [store_sales, _] = two_branches(&plan);
    let join = the_join(&plan);
    let filters = |node: &PlanNode| {
        (node.pre_order())
            .filter(|node| matches!(node, PlanNode::Filter(_)))
            .count()
    };
    assert_eq!(filters(&join.left) + filters(&join.right), 0);
    assert_eq!(filters(store_sales), 1);
}

const SPEND_CLICKS: &str = r#"
semantic_model:
  name: all_spend
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - {name: clicks, data_type: bigint, agg: sum}
  unionsets:
    - name: all_spend
      mode: all
      dimensions: [{ref: date}]
      measures: [{ref: cost}, {ref: clicks}]
      grainsets:
        - name: spend_g
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: g_daily
              extras:
                storage: {tables: [warehouse.g_daily]}
                semantic_mapping: {date: day, cost: cost}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: g_monthly
              extras:
                storage: {tables: [warehouse.g_monthly]}
                semantic_mapping: {date: month_start, cost: cost}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
      datasets:
        - name: spend_c
          extras:
            storage: {tables: [warehouse.spend_c]}
            semantic_mapping: {date: day, cost: cost, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

fn answered_rows(branch: &PlanNode) -> &PlanNode {
    (branch.pre_order())
        .find(|node| {
            (node.common().semantic_meta.iter()).any(|subject| {
                matches!(
                    subject,
                    SemAnnotation::SemanticSource {
                        answered_by: Some(_),
                        ..
                    }
                )
            })
        })
        .unwrap_or_else(|| panic!("the rows of the child that answers, in {branch:?}"))
}

#[tokio::test]
async fn the_child_that_answers_a_grainset_branch_writes_its_typed_null() {
    let daily = [("day", "date"), ("cost", "long")];
    let tables = [
        ("warehouse.g_daily", &daily[..]),
        (
            "warehouse.g_monthly",
            &[("month_start", "date"), ("cost", "long")],
        ),
        (
            "warehouse.spend_c",
            &[("day", "date"), ("cost", "long"), ("clicks", "long")],
        ),
    ];
    let manifest = over_catalog(SPEND_CLICKS, &tables).await;
    let request = at_granularity("date", Granularity::Month, &["cost", "clicks"]);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_g] = two_branches(&plan);
    let clicks = format!("{}.clicks", alias(&manifest, "all_spend"));
    assert_eq!(
        written(row_projection(answered_rows(spend_g)), &clicks),
        Some(Expr::Literal(Literal::Null(DataType::BigInt)))
    );
}

const SPEND_CHANNELS: &str = r#"
semantic_model:
  name: all_spend
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
  unionsets:
    - name: all_spend
      mode: all
      dimensions: [{ref: date}, {ref: campaign}, {ref: channel}]
      measures: [{ref: clicks}, {ref: budget}]
      grainsets:
        - name: spend_g
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: g_daily
              extras:
                storage: {tables: [warehouse.g_daily]}
                semantic_mapping: {campaign: campaign_id, date: day, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: g_monthly
              extras:
                storage: {tables: [warehouse.g_monthly]}
                semantic_mapping: {campaign: campaign_id, date: month_start, budget: budget}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
      datasets:
        - name: spend_c
          extras:
            storage: {tables: [warehouse.spend_c]}
            semantic_mapping:
              {campaign: campaign_id, date: day, channel: channel, clicks: clicks, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

const DAILY_CLICKS: &[(&str, &str)] = &[
    ("day", "date"),
    ("campaign_id", "string"),
    ("clicks", "long"),
];

const MONTHLY_BUDGET: &[(&str, &str)] = &[
    ("month_start", "date"),
    ("campaign_id", "string"),
    ("budget", "long"),
];

async fn spend_channels() -> SemanticManifest {
    spend_channels_over(SPEND_CHANNELS).await
}

async fn spend_channels_over(yaml: &str) -> SemanticManifest {
    let tables = [
        ("warehouse.g_daily", DAILY_CLICKS),
        ("warehouse.g_monthly", MONTHLY_BUDGET),
        (
            "warehouse.spend_c",
            &[
                ("day", "date"),
                ("campaign_id", "string"),
                ("channel", "string"),
                ("clicks", "long"),
                ("budget", "long"),
            ][..],
        ),
    ];
    over_catalog(yaml, &tables).await
}

fn every_branch_unserved() -> String {
    let (head, _) = SPEND_CHANNELS
        .split_once("      datasets:\n        - name: spend_c")
        .expect("the Dataset branch");
    let (_, grainset) = head
        .split_once("        - name: spend_g")
        .expect("the Grainset branch");
    let second = format!("        - name: spend_g{grainset}")
        .replace("spend_g", "spend_h")
        .replace("g_daily", "h_daily")
        .replace("g_monthly", "h_monthly")
        .replace(
            "campaign: campaign_id",
            "campaign: campaign_id, channel: channel",
        );
    format!("{head}{second}")
}

async fn spend_unserved() -> SemanticManifest {
    let channelled =
        |columns: &[(&'static str, &'static str)]| [columns, &[("channel", "string")]].concat();
    let (h_daily, h_monthly) = (channelled(DAILY_CLICKS), channelled(MONTHLY_BUDGET));
    let tables = [
        ("warehouse.g_daily", DAILY_CLICKS),
        ("warehouse.g_monthly", MONTHLY_BUDGET),
        ("warehouse.h_daily", h_daily.as_slice()),
        ("warehouse.h_monthly", h_monthly.as_slice()),
    ];
    over_catalog(&every_branch_unserved(), &tables).await
}

async fn clicks_and_budget() -> Vec<Golden> {
    let manifest = spend_unserved().await;
    let request = at_granularity("date", Granularity::Month, &["channel", "clicks", "budget"]);
    let names = ["all_spend", "all_spend.spend_g"];
    let held = (Some(Granularity::Month), Some(Granularity::Month));
    let nearest: [(&str, &[&str]); 2] = [
        ("all_spend.spend_g.g_daily", &["budget"]),
        ("all_spend.spend_g.g_monthly", &["clicks"]),
    ];
    let refused = grain_not_servable(&manifest, &request, &names, held, &nearest);
    vec![Golden::message("clicks_and_budget", &refused)]
}

#[tokio::test]
async fn a_grainset_branch_whose_names_no_one_child_serves_is_not_servable() {
    for golden in clicks_and_budget().await {
        golden.assert_committed();
    }
}

fn spend_literal_channels() -> String {
    let web = "channel: {lit: {string: web}}";
    SPEND_CHANNELS
        .replace(
            "dimensions: [{ref: date}, {ref: campaign}, {ref: channel}]",
            "dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}, {ref: channel}]",
        )
        .replace(
            "campaign: campaign_id, date: day, clicks: clicks}",
            &format!("campaign: campaign_id, date: day, {web}, clicks: clicks}}"),
        )
        .replace(
            "campaign: campaign_id, date: month_start, budget: budget}",
            &format!("campaign: campaign_id, date: month_start, {web}, budget: budget}}"),
        )
        .replace(
            "channel: channel, clicks: clicks, budget: budget}",
            &format!("{web}, clicks: clicks, budget: budget}}"),
        )
}

#[tokio::test]
async fn a_unionset_with_a_grainset_branch_reads_the_empty_read_where_the_fold_excludes_every_branch(
) {
    let manifest = spend_channels_over(&spend_literal_channels()).await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "clicks", "budget"],
    );
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let empty = scan_of(&plan);
    assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
    assert_eq!(datasets(&plan), [] as [&str; 0], "no branch is read");
    let names: Vec<_> = (empty.common.output_schema.columns.iter())
        .map(|column| (column.name.as_str(), column.data_type.clone()))
        .collect();
    let spend = alias(&manifest, "all_spend");
    let expected = [
        ("date", DataType::Date),
        ("campaign", DataType::String),
        ("clicks", DataType::BigInt),
        ("budget", DataType::BigInt),
    ]
    .map(|(name, data_type)| (format!("{spend}.{name}"), data_type));
    assert_eq!(
        names
            .iter()
            .map(|(name, ty)| (name.to_string(), ty.clone()))
            .collect::<Vec<_>>(),
        expected
    );
}

#[tokio::test]
async fn a_joinset_operand_unionset_with_a_grainset_branch_reads_the_empty_read_where_the_fold_excludes_every_branch(
) {
    let yaml = spend_literal_channels();
    let (head, inner) = yaml.split_once("  unionsets:\n").expect("the unionsets");
    let inner: String = inner
        .lines()
        .map(|line| format!("    {line}\n"))
        .collect::<String>()
        .replacen(
            "mode: all\n",
            "mode: all\n          extras:\n            temporal: {timeseries: {occurred_at: date}, granularity: month}\n",
            1,
        );
    let nested = format!(
        "{head}  joinsets:\n    - name: spend_campaigns\n      dimensions: [{{ref: date}}, {{ref: campaign}}, {{ref: channel}}]\n      measures: [{{ref: clicks}}, {{ref: budget}}]\n      unionsets:\n{inner}      datasets:\n        - name: campaigns\n          extras:\n            storage: {{tables: [warehouse.campaigns]}}\n            semantic_mapping: {{campaign: campaign_id}}\n            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}\n      relationships:\n        - name: spend_to_campaigns\n          from: all_spend\n          to: campaigns\n          on: [campaign]\n          cardinality: many_to_one\n          optional: none\n          integrity: none\n"
    );
    let tables = [
        ("warehouse.g_daily", DAILY_CLICKS),
        ("warehouse.g_monthly", MONTHLY_BUDGET),
        (
            "warehouse.spend_c",
            &[
                ("day", "date"),
                ("campaign_id", "string"),
                ("clicks", "long"),
                ("budget", "long"),
            ][..],
        ),
        ("warehouse.campaigns", &[("campaign_id", "string")][..]),
    ];
    let manifest = over_catalog(&nested, &tables).await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "clicks", "budget"],
    );
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let empty = |node: &PlanNode| matches!(node, PlanNode::Scan(scan) if scan.reader == VirtualTable::new(Vec::new()).into());
    assert!(
        plan.root.pre_order().any(empty),
        "the union reads the empty read"
    );
    assert_eq!(datasets(&plan), ["spend_campaigns.campaigns"]);
}

fn schema_columns(schema: &semstrait_ir::Schema) -> Vec<(String, DataType, bool)> {
    (schema.columns.iter())
        .map(|column| {
            (
                column.name.clone(),
                column.data_type.clone(),
                column.nullable,
            )
        })
        .collect()
}

fn under(alias: &str, columns: &[(&str, DataType, bool)]) -> Vec<(String, DataType, bool)> {
    (columns.iter())
        .map(|(name, data_type, nullable)| {
            (format!("{alias}.{name}"), data_type.clone(), *nullable)
        })
        .collect()
}

fn pruned_names(scan: &semstrait_ir::ScanNode) -> Vec<String> {
    let mut names: Vec<String> = (scan.common.semantic_meta.iter())
        .flat_map(|subject| match subject {
            SemAnnotation::SemanticSource { pruned, .. } => pruned.clone(),
            _ => Vec::new(),
        })
        .map(|name| name.as_str().to_owned())
        .collect();
    names.sort_unstable();
    names
}

fn empty_reads(plan: &SemanticPlan) -> Vec<&semstrait_ir::ScanNode> {
    (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Scan(scan) if scan.reader == VirtualTable::new(Vec::new()).into() => {
                Some(scan)
            }
            _ => None,
        })
        .collect()
}

fn every_spend_branch_a_grainset() -> String {
    let yaml = spend_literal_channels();
    let (head, _) = yaml
        .split_once("      datasets:\n        - name: spend_c")
        .expect("the Dataset branch");
    let web = "channel: {lit: {string: web}}";
    format!(
        "{head}        - name: spend_c\n          extras:\n            temporal: {{timeseries: {{occurred_at: date}}, granularity: month}}\n          datasets:\n            - name: c_daily\n              extras:\n                storage: {{tables: [warehouse.c_daily]}}\n                semantic_mapping: {{campaign: campaign_id, date: day, {web}, clicks: clicks}}\n                temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}\n            - name: c_monthly\n              extras:\n                storage: {{tables: [warehouse.c_monthly]}}\n                semantic_mapping: {{campaign: campaign_id, date: month_start, {web}, budget: budget}}\n                temporal: {{timeseries: {{occurred_at: date}}, granularity: month}}\n"
    )
}

fn grainset_children() -> [(&'static str, &'static [(&'static str, &'static str)]); 5] {
    [
        ("warehouse.g_daily", DAILY_CLICKS),
        ("warehouse.g_monthly", MONTHLY_BUDGET),
        (
            "warehouse.spend_c",
            &[
                ("day", "date"),
                ("campaign_id", "string"),
                ("clicks", "long"),
                ("budget", "long"),
            ][..],
        ),
        ("warehouse.c_daily", DAILY_CLICKS),
        ("warehouse.c_monthly", MONTHLY_BUDGET),
    ]
}

#[tokio::test]
async fn a_unionset_of_grainset_branches_only_reads_the_empty_read_typed_by_its_branches() {
    let yaml = SPEND_BY_CHANNEL
        .replace("    - {name: orders, data_type: bigint, agg: count}\n", "")
        .replace(", {ref: orders}", "");
    let (head, _) = yaml
        .split_once("      datasets:\n        - name: spend_store")
        .expect("the Dataset branch");
    let yaml = format!(
        "{head}        - name: spend_store\n          extras:\n            temporal: {{timeseries: {{occurred_at: date}}, granularity: month}}\n          datasets:\n            - name: store_daily\n              extras:\n                storage: {{tables: [warehouse.store_daily]}}\n                semantic_mapping: {{date: day, channel: {{lit: store}}, clicks: clicks}}\n                temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}\n            - name: store_monthly\n              extras:\n                storage: {{tables: [warehouse.store_monthly]}}\n                semantic_mapping: {{date: month_start, channel: {{lit: store}}, budget: budget}}\n                temporal: {{timeseries: {{occurred_at: date}}, granularity: month}}\n"
    );
    let manifest = spend_by_channel_over(
        &yaml,
        &[
            ("warehouse.store_daily", DAILY_CLICKS),
            ("warehouse.store_monthly", MONTHLY_BUDGET),
        ],
    )
    .await;
    let request = at_granularity("date", Granularity::Month, &["clicks", "budget"]);
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let empty = scan_of(&plan);
    assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
    assert_eq!(datasets(&plan), [] as [&str; 0], "no branch is read");
    let spend = alias(&manifest, "spend_by_channel");
    assert_eq!(
        schema_columns(&empty.common.output_schema),
        under(
            &spend,
            &[
                ("date", DataType::Date, true),
                ("__plan_agg1_clicks", DataType::BigInt, true),
                ("__plan_agg1_budget", DataType::BigInt, true),
            ]
        )
    );
    assert_eq!(
        delivery_of(&empty.common.semantic_meta),
        Some(Delivery::PartialAggregates)
    );
    assert_eq!(
        pruned_names(empty),
        ["spend_by_channel.spend_store", "spend_by_channel.spend_web"]
    );
    assert_eq!(output_names(&plan), ["date", "clicks", "budget"]);
}

#[tokio::test]
async fn a_joinset_branch_among_the_excluded_gives_the_empty_read_its_partial_aggregate_types() {
    let inner = ALL_ORDERS.replace("optional: right", "optional: none");
    for (data_type, entry) in [
        ("bigint", "customer_rows"),
        ("double", "__plan_agg1_customer_rows"),
    ] {
        let yaml = inner.replace(
            "{name: customer_rows, data_type: bigint, agg: count}",
            &format!("{{name: customer_rows, data_type: {data_type}, agg: count}}"),
        );
        let manifest = all_orders(&yaml).await;
        let apac = dim("channel").expect("a name").eq(lit("apac"));

        let (plan, _) = planned(&manifest, &select(&["customer_rows"]).with_predicate(apac));

        let empty = scan_of(&plan);
        assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
        assert_eq!(datasets(&plan), [] as [&str; 0], "{data_type}");
        let orders = alias(&manifest, "all_orders");
        assert_eq!(
            schema_columns(&empty.common.output_schema),
            under(&orders, &[(entry, DataType::BigInt, true)]),
            "{data_type}"
        );
        assert_eq!(
            pruned_names(empty),
            ["all_orders.jw", "all_orders.orders_store"],
            "{data_type}"
        );
    }
}

#[tokio::test]
async fn an_implicit_unionset_branch_among_the_excluded_gives_the_empty_read_its_row_types() {
    let yaml = U1.replace(
        "['s3://lake/a/orders/', 's3://lake/c/orders/']",
        "['s3://lake/a/orders/', 's3://lake/a/refunds/']",
    );
    let objects = [
        "s3://lake/a/orders/part-0.parquet",
        "s3://lake/a/refunds/part-0.parquet",
        "s3://lake/c/returns/part-0.parquet",
    ];
    let manifest = compiled(&yaml, &Catalog::default(), &seeded(&objects).await).await;

    let (plan, _) = planned(&manifest, &on_channel(&["avg_amount"], "b"));

    let empty = scan_of(&plan);
    assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
    assert_eq!(datasets(&plan), [] as [&str; 0], "no branch is read");
    assert_eq!(
        schema_columns(&empty.common.output_schema),
        under(
            &alias(&manifest, "u1"),
            &[
                ("__plan_cond_avg_amount", DataType::Boolean, false),
                ("avg_amount", DataType::Double, true),
            ]
        )
    );
    assert_eq!(
        delivery_of(&empty.common.semantic_meta),
        Some(Delivery::Rows)
    );
    assert_eq!(pruned_names(empty), ["u1.d9", "u1.u2"]);
}

#[tokio::test]
async fn a_joinset_operand_unionset_of_grainset_branches_only_reads_the_empty_read() {
    let yaml = every_spend_branch_a_grainset();
    let (head, inner) = yaml.split_once("  unionsets:\n").expect("the unionsets");
    let inner: String = inner
        .lines()
        .map(|line| format!("    {line}\n"))
        .collect::<String>()
        .replacen(
            "mode: all\n",
            "mode: all\n          extras:\n            temporal: {timeseries: {occurred_at: date}, granularity: month}\n",
            1,
        );
    let nested = format!(
        "{head}  joinsets:\n    - name: spend_campaigns\n      dimensions: [{{ref: date}}, {{ref: campaign}}, {{ref: channel}}]\n      measures: [{{ref: clicks}}, {{ref: budget}}]\n      unionsets:\n{inner}      datasets:\n        - name: campaigns\n          extras:\n            storage: {{tables: [warehouse.campaigns]}}\n            semantic_mapping: {{campaign: campaign_id}}\n            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}\n      relationships:\n        - name: spend_to_campaigns\n          from: all_spend\n          to: campaigns\n          on: [campaign]\n          cardinality: many_to_one\n          optional: none\n          integrity: none\n"
    );
    let mut tables = grainset_children().to_vec();
    tables.push(("warehouse.campaigns", &[("campaign_id", "string")][..]));
    let manifest = over_catalog(&nested, &tables).await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "clicks", "budget"],
    );
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let [empty] = empty_reads(&plan)[..] else {
        panic!("one empty read in {plan:?}");
    };
    assert_eq!(
        schema_columns(&empty.common.output_schema),
        under(
            &alias(&manifest, "spend_campaigns.all_spend"),
            &[
                ("date", DataType::Date, true),
                ("campaign", DataType::String, true),
                ("__plan_agg1_clicks", DataType::BigInt, true),
                ("__plan_agg1_budget", DataType::BigInt, true),
            ]
        )
    );
    assert_eq!(datasets(&plan), ["spend_campaigns.campaigns"]);
}

#[tokio::test]
async fn a_count_cast_to_double_over_every_branch_excluded_reads_the_empty_read_with_the_count_entry(
) {
    let yaml = ALL_SALES.replace(
        "{name: order_count, data_type: bigint, agg: count}",
        "{name: order_count, data_type: double, agg: count}",
    );
    let manifest = all_sales(&yaml).await;

    let (plan, _, alias) = every_branch_excluded(&manifest, "order_count");

    assert_eq!(
        schema_columns(&scan_of(&plan).common.output_schema),
        under(
            &alias,
            &[("__plan_agg1_order_count", DataType::BigInt, false)]
        )
    );
    assert_eq!(
        schema_columns(&plan.root.common().output_schema)
            .into_iter()
            .map(|(_, data_type, _)| data_type)
            .collect::<Vec<_>>(),
        [DataType::Double]
    );
}

const OUTER_SALES: &str = r#"
semantic_model:
  name: outer_sales
  dimensions:
    - {name: region, data_type: string, type: categorical}
    - {name: customer_id, data_type: string, type: categorical}
  measures:
    - {name: buyers, data_type: bigint, agg: count_distinct}
  unionsets:
    - name: outer
      mode: all
      dimensions: [{ref: region}, {ref: customer_id}]
      measures: [{ref: buyers}]
      datasets:
        - name: online
          extras:
            storage: {tables: [warehouse.online]}
            semantic_mapping: {region: region, customer_id: customer_id, buyers: customer_id}
      joinsets:
        - name: jw
          unionsets:
            - name: all_sales
              mode: all
              datasets:
                - name: store_sales
                  extras:
                    storage: {tables: [warehouse.store_sales]}
                    semantic_mapping: {region: {lit: {string: emea}}, customer_id: customer_id, buyers: customer_id}
                - name: web_sales
                  extras:
                    storage: {tables: [warehouse.web_sales]}
                    semantic_mapping: {region: {lit: {string: amer}}, customer_id: customer_id, buyers: customer_id}
          datasets:
            - name: customers
              extras:
                storage: {tables: [warehouse.customers]}
                semantic_mapping: {customer_id: customer_id}
          relationships:
            - {name: sales_to_customers, from: all_sales, to: customers, on: [customer_id], cardinality: many_to_one, optional: none, integrity: none}
"#;

const OUTER_COLUMNS: &[(&str, &str)] = &[
    ("region", "string"),
    ("customer_id", "string"),
    ("store_id", "string"),
    ("price", "long"),
    ("qty", "long"),
];

async fn outer_over(yaml: &str, extra: &[(&str, &[(&str, &str)])]) -> SemanticManifest {
    let mut tables = vec![
        ("warehouse.online", OUTER_COLUMNS),
        ("warehouse.store_sales", OUTER_COLUMNS),
        ("warehouse.web_sales", OUTER_COLUMNS),
        ("warehouse.customers", &[("customer_id", "string")][..]),
    ];
    tables.extend_from_slice(extra);
    over_catalog(yaml, &tables).await
}

fn typed_field(schema: &semstrait_ir::Schema, name: &str) -> (DataType, bool) {
    let column = (schema.columns.iter())
        .find(|column| column.name.ends_with(name) && !column.name.contains("__plan_cond_"))
        .unwrap_or_else(|| panic!("`{name}` in {schema:?}"));
    (column.data_type.clone(), column.nullable)
}

fn assert_read_beside_kept<'p>(
    plan: &'p SemanticPlan,
    read: &[&str],
    measure: &str,
    expected: (DataType, bool, bool),
    context: &dyn std::fmt::Debug,
) -> &'p semstrait_ir::ScanNode {
    let mut datasets = datasets(plan);
    datasets.sort_unstable();
    assert_eq!(datasets, read, "{context:?}");
    let [empty] = empty_reads(plan)[..] else {
        panic!("one empty read in {plan:?}");
    };
    let union = first_union(&plan.root);
    assert_eq!(union.inputs.len(), 2, "{context:?}");
    let (data_type, empty_nullable, kept_nullable) = expected;
    assert_eq!(
        typed_field(&empty.common.output_schema, measure),
        (data_type.clone(), empty_nullable),
        "{context:?}"
    );
    assert_eq!(
        typed_field(&union.inputs[0].common().output_schema, measure),
        (data_type, kept_nullable),
        "{context:?}"
    );
    empty
}

#[tokio::test]
async fn an_inner_unionset_with_every_branch_excluded_is_read_beside_a_kept_branch() {
    let counts = [
        (
            "buyers",
            OUTER_SALES.to_owned(),
            (DataType::String, true, true),
        ),
        (
            "buyers",
            OUTER_SALES.replace("agg: count_distinct", "agg: count"),
            (DataType::BigInt, false, true),
        ),
    ];
    for (measure, yaml, expected) in counts {
        let manifest = outer_over(&yaml, &[]).await;
        let apac = dim("region").expect("a name").eq(lit("apac"));

        for selected in [vec![measure], vec!["region", measure]] {
            let request = select(&selected).with_predicate(apac.clone());
            let (plan, _) = planned(&manifest, &request);

            assert_read_beside_kept(
                &plan,
                &["outer.jw.customers", "outer.online"],
                measure,
                expected.clone(),
                &selected,
            );
        }
    }
}

fn outer_sales_joined() -> String {
    let joined = |name: &str, sales: &str, region: &str, dimension: &str, lookup: &str| {
        format!(
            "                - name: {name}\n                  datasets:\n                    - name: {sales}\n                      extras:\n                        storage: {{tables: [warehouse.{sales}]}}\n                        semantic_mapping: {{region: {{lit: {{string: {region}}}}}, customer_id: customer_id, buyers: customer_id, {dimension}: {dimension}}}\n                    - name: {lookup}\n                      extras:\n                        storage: {{tables: [warehouse.{lookup}]}}\n                        semantic_mapping: {{{dimension}: {dimension}}}\n                  relationships:\n                    - {{name: {sales}_to_{lookup}, from: {sales}, to: {lookup}, on: [{dimension}], cardinality: many_to_one, optional: none, integrity: none}}\n"
        )
    };
    let (head, _) = OUTER_SALES
        .split_once("              datasets:\n                - name: store_sales")
        .expect("the sales datasets");
    let (_, tail) = OUTER_SALES
        .split_once("          datasets:\n            - name: customers")
        .expect("the customers dataset");
    format!(
        "{}              joinsets:\n{}{}          datasets:\n            - name: customers{tail}",
        head.replace(
            "dimensions: [{ref: region}, {ref: customer_id}]",
            "dimensions: [{ref: region}, {ref: customer_id}, {ref: store_id}]"
        )
        .replace(
            "    - {name: customer_id, data_type: string, type: categorical}\n",
            "    - {name: customer_id, data_type: string, type: categorical}\n    - {name: store_id, data_type: string, type: categorical}\n"
        ),
        joined("store_j", "store_sales", "emea", "store_id", "stores"),
        joined("web_j", "web_sales", "amer", "store_id", "sites"),
    )
}

#[tokio::test]
async fn an_inner_unionset_of_joinset_branches_with_every_branch_excluded_is_read_beside_a_kept_branch(
) {
    let manifest = outer_over(
        &outer_sales_joined(),
        &[
            ("warehouse.stores", &[("store_id", "string")][..]),
            ("warehouse.sites", &[("store_id", "string")][..]),
        ],
    )
    .await;
    let apac = dim("region").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &select(&["buyers"]).with_predicate(apac));

    let empty = assert_read_beside_kept(
        &plan,
        &["outer.jw.customers", "outer.online"],
        "buyers",
        (DataType::String, true, true),
        &"joinsets",
    );
    assert_eq!(
        pruned_names(empty),
        ["outer.jw.all_sales.store_j", "outer.jw.all_sales.web_j"]
    );
}

fn outer_sales_grained() -> String {
    let day = "temporal: {timeseries: {occurred_at: date}, granularity: day}";
    let month = "temporal: {timeseries: {occurred_at: date}, granularity: month}";
    let grained = |name: &str, region: &str| {
        let lit = format!("region: {{lit: {{string: {region}}}}}");
        format!(
            "                - name: {name}\n                  extras:\n                    {month}\n                  datasets:\n                    - name: {name}_daily\n                      extras:\n                        storage: {{tables: [warehouse.{name}_daily]}}\n                        semantic_mapping: {{date: day, {lit}, customer_id: customer_id, buyers: customer_id}}\n                        {day}\n                    - name: {name}_monthly\n                      extras:\n                        storage: {{tables: [warehouse.{name}_monthly]}}\n                        semantic_mapping: {{date: month_start, {lit}, budget: budget}}\n                        {month}\n"
        )
    };
    format!(
        r#"
semantic_model:
  name: outer_sales
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {{name: region, data_type: string, type: categorical}}
    - {{name: customer_id, data_type: string, type: categorical}}
  measures:
    - {{name: buyers, data_type: bigint, agg: count_distinct}}
    - {{name: budget, data_type: bigint, agg: sum}}
  unionsets:
    - name: outer
      mode: all
      dimensions: [{{ref: date}}, {{ref: region}}, {{ref: customer_id}}]
      measures: [{{ref: buyers}}, {{ref: budget}}]
      datasets:
        - name: online
          extras:
            storage: {{tables: [warehouse.online]}}
            semantic_mapping: {{date: day, region: region, customer_id: customer_id, buyers: customer_id, budget: budget}}
            {day}
      joinsets:
        - name: jw
          extras:
            {month}
          unionsets:
            - name: all_sales
              mode: all
              extras:
                {month}
              grainsets:
{}{}          datasets:
            - name: customers
              extras:
                storage: {{tables: [warehouse.customers]}}
                semantic_mapping: {{customer_id: customer_id}}
                {day}
          relationships:
            - {{name: sales_to_customers, from: all_sales, to: customers, on: [customer_id], cardinality: many_to_one, optional: none, integrity: none}}
"#,
        grained("store", "emea"),
        grained("web", "amer"),
    )
}

#[tokio::test]
async fn an_inner_unionset_of_grainset_branches_with_every_branch_excluded_is_read_beside_a_kept_branch(
) {
    let daily = &[
        ("day", "date"),
        ("region", "string"),
        ("customer_id", "string"),
    ][..];
    let monthly = &[("month_start", "date"), ("budget", "long")][..];
    let tables = [
        (
            "warehouse.online",
            &[
                ("day", "date"),
                ("region", "string"),
                ("customer_id", "string"),
                ("budget", "long"),
            ][..],
        ),
        ("warehouse.store_daily", daily),
        ("warehouse.store_monthly", monthly),
        ("warehouse.web_daily", daily),
        ("warehouse.web_monthly", monthly),
        ("warehouse.customers", &[("customer_id", "string")][..]),
    ];
    let manifest = over_catalog(&outer_sales_grained(), &tables).await;
    let apac = dim("region").expect("a name").eq(lit("apac"));

    for (request, measure) in [
        (
            at_granularity("date", Granularity::Day, &["buyers"]),
            "buyers",
        ),
        (
            at_granularity("date", Granularity::Month, &["buyers", "budget"]),
            "buyers",
        ),
    ] {
        let (plan, _) = planned(&manifest, &request.with_predicate(apac.clone()));

        assert_read_beside_kept(
            &plan,
            &["outer.jw.customers", "outer.online"],
            measure,
            (DataType::String, true, true),
            &"grainsets",
        );
    }
}

const SPEND_UNION_CHILD: &str = r#"
semantic_model:
  name: spend_union_child
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
  unionsets:
    - name: top
      mode: all
      dimensions: [{ref: date, is_key: true}, {ref: channel, is_key: true}]
      measures: [{ref: clicks}, {ref: budget}]
      grainsets:
        - name: spend
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: monthly
              extras:
                storage: {tables: [warehouse.monthly]}
                semantic_mapping: {date: month_start, channel: {lit: {string: store}}, budget: budget}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
          unionsets:
            - name: daily
              mode: all
              extras:
                temporal: {timeseries: {occurred_at: date}, granularity: day}
              datasets:
                - name: daily_web
                  extras:
                    storage: {tables: [warehouse.daily_web]}
                    semantic_mapping: {date: day, channel: {lit: {string: web}}, clicks: clicks}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
                - name: daily_app
                  extras:
                    storage: {tables: [warehouse.daily_app]}
                    semantic_mapping: {date: day, channel: {lit: {string: app}}, clicks: clicks}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

#[tokio::test]
async fn a_grainset_child_unionset_with_every_branch_excluded_reads_the_empty_read() {
    let tables = [
        ("warehouse.monthly", MONTHLY_BUDGET),
        ("warehouse.daily_web", DAILY_CLICKS),
        ("warehouse.daily_app", DAILY_CLICKS),
    ];
    let manifest = over_catalog(SPEND_UNION_CHILD, &tables).await;
    let request = at_granularity("date", Granularity::Month, &["clicks", "budget"]);
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let [empty] = empty_reads(&plan)[..] else {
        panic!("one empty read in {plan:?}");
    };
    assert_eq!(datasets(&plan), ["top.monthly"]);
    assert_eq!(
        pruned_names(empty),
        ["top.daily.daily_app", "top.daily.daily_web"]
    );
    assert_eq!(
        schema_columns(&empty.common.output_schema),
        under(
            &alias(&manifest, "top.daily"),
            &[
                ("date", DataType::Date, true),
                ("clicks", DataType::BigInt, true),
            ]
        )
    );
}

#[tokio::test]
async fn a_grainset_child_unionset_with_every_branch_excluded_is_read_beside_a_kept_branch() {
    let yaml = SPEND_UNION_CHILD
        .replace(
            "    - {name: budget, data_type: bigint, agg: sum}\n",
            "    - {name: budget, data_type: bigint, agg: sum}\n    - {name: users, data_type: bigint, agg: count_distinct}\n",
        )
        .replace(
            "measures: [{ref: clicks}, {ref: budget}]",
            "measures: [{ref: clicks}, {ref: budget}, {ref: users}]",
        )
        .replace("clicks: clicks}", "clicks: clicks, users: user_id}")
        + "      datasets:\n        - name: store\n          extras:\n            storage: {tables: [warehouse.store]}\n            semantic_mapping: {date: day, channel: channel, clicks: clicks, users: user_id}\n            temporal: {timeseries: {occurred_at: date}, granularity: day}\n";
    let clicks = &[("day", "date"), ("clicks", "long"), ("user_id", "string")][..];
    let tables = [
        ("warehouse.monthly", MONTHLY_BUDGET),
        ("warehouse.daily_web", clicks),
        ("warehouse.daily_app", clicks),
        (
            "warehouse.store",
            &[
                ("day", "date"),
                ("channel", "string"),
                ("clicks", "long"),
                ("user_id", "string"),
            ][..],
        ),
    ];
    let manifest = over_catalog(&yaml, &tables).await;
    let request = at_granularity("date", Granularity::Month, &["users"]);
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let mut read = datasets(&plan);
    read.sort_unstable();
    assert_eq!(read, ["top.store"]);
    let [empty] = empty_reads(&plan)[..] else {
        panic!("one empty read in {plan:?}");
    };
    let union = first_union(&plan.root);
    assert_eq!(union.inputs.len(), 2);
    assert_eq!(
        typed_field(&empty.common.output_schema, "users"),
        (DataType::String, true)
    );
    assert_eq!(
        typed_field(&union.inputs[0].common().output_schema, "users"),
        (DataType::String, true)
    );
}

const UNTYPED_SALES: &str = r#"
semantic_model:
  name: untyped_sales
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: untyped
      mode: all
      dimensions: [{ref: region}]
      measures: [{ref: revenue}]
      datasets:
        - name: emea_files
          extras:
            storage: {format: parquet, paths: ['s3://lake/emea/']}
            semantic_mapping: {region: {lit: {string: emea}}, revenue: amount}
        - name: amer_files
          extras:
            storage: {format: parquet, paths: ['s3://lake/amer/']}
            semantic_mapping: {region: {lit: {string: amer}}, revenue: amount}
"#;

#[tokio::test]
async fn an_empty_read_over_columns_with_no_catalog_type_takes_the_model_type() {
    let objects = [
        "s3://lake/emea/part-0.parquet",
        "s3://lake/amer/part-0.parquet",
    ];
    let manifest = compiled(UNTYPED_SALES, &Catalog::default(), &seeded(&objects).await).await;
    let apac = dim("region").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &select(&["revenue"]).with_predicate(apac));

    let empty = scan_of(&plan);
    assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
    assert_eq!(datasets(&plan), [] as [&str; 0]);
    assert_eq!(
        schema_columns(&empty.common.output_schema),
        under(
            &alias(&manifest, "untyped"),
            &[("revenue", DataType::Double, true)]
        )
    );
}

#[tokio::test]
async fn a_grainset_branch_keeps_a_name_that_it_null_fills_out_of_its_pick() {
    let manifest = spend_channels().await;
    let request = at_granularity("date", Granularity::Month, &["channel", "budget"]);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_g] = two_branches(&plan);
    let rows = answered_rows(spend_g);
    let answered = DataKindName::new("all_spend.spend_g.g_monthly");
    let trace = &rows.common().semantic_meta;
    assert!(
        trace.iter().any(
            |subject| matches!(subject, SemAnnotation::SemanticSource { answered_by: Some(name), .. } if *name == answered)
        ),
        "{answered:?} in {trace:?}"
    );
    let channel = format!("{}.channel", alias(&manifest, "all_spend"));
    assert_eq!(
        written(row_projection(rows), &channel),
        Some(Expr::Literal(Literal::Null(DataType::String)))
    );
}

#[tokio::test]
async fn a_predicate_leaf_that_a_grainset_branch_null_fills_stays_out_of_its_pick() {
    let manifest = spend_channels().await;
    let since = Literal::Date("2026-03-15".into());
    let web = dim("channel").expect("a name").eq(lit("web"))
        | dim("date")
            .expect("a name")
            .gt_eq(Expr::Literal(since.clone()));
    let request = at_granularity("date", Granularity::Month, &["budget"]).with_predicate(web);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_g] = two_branches(&plan);
    let alias = &alias(&manifest, "all_spend");
    let PlanNode::Filter(filter) = answered_rows(spend_g) else {
        panic!("the predicate above the row projection, got {spend_g:?}");
    };
    let expected = qualified(alias, "channel").eq(text("web"))
        | qualified(alias, "date").gt_eq(Expr::Literal(since));
    assert_eq!(filter.filter, expected);
    assert_eq!(
        written(projection(&filter.input), &format!("{alias}.channel")),
        Some(Expr::Literal(Literal::Null(DataType::String)))
    );
}

#[tokio::test]
async fn the_fold_reads_through_a_joinset_branchs_operands() {
    let manifest = sales_stores().await;
    let web = dim("channel").expect("a name").eq(lit("web"));

    let (plan, _) = planned(&manifest, &select(&["revenue"]).with_predicate(web));

    one_branch_left(&plan, "all_sales.web_sales", "all_sales.store_sales");
}

const U0: &str = r#"
semantic_model:
  name: u0
  dimensions:
    - {name: channel, data_type: string, type: categorical}
    - {name: region_id, data_type: string, type: categorical}
    - {name: region_name, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: u0
      mode: all
      dimensions: [{ref: channel}, {ref: region_id}, {ref: region_name}]
      measures: [{ref: revenue}]
      datasets:
        - name: d10
          extras:
            storage: {tables: [warehouse.d10]}
            semantic_mapping: {channel: {lit: b}, region_id: region_id, revenue: amount}
      joinsets:
        - name: j
          unionsets:
            - name: u3
              mode: all
              datasets:
                - name: e7
                  extras:
                    storage: {tables: [warehouse.e7]}
                    semantic_mapping: {channel: {lit: a}, region_id: region_id, revenue: amount}
                - name: e8
                  extras:
                    storage: {tables: [warehouse.e8]}
                    semantic_mapping: {channel: {lit: b}, region_id: region_id, revenue: amount}
          datasets:
            - name: regions
              extras:
                storage: {tables: [warehouse.regions]}
                semantic_mapping: {region_id: region_id, region_name: region_name}
          relationships:
            - name: u3_to_regions
              from: u3
              to: regions
              on: [region_id]
              cardinality: many_to_one
              optional: none
              integrity: assumed
"#;

#[tokio::test]
async fn a_unionset_below_a_joinset_branch_folds_a_name_only_where_every_branch_agrees() {
    let sales = &[("region_id", "string"), ("amount", "double")][..];
    let tables = [
        ("warehouse.d10", sales),
        ("warehouse.e7", sales),
        ("warehouse.e8", sales),
        (
            "warehouse.regions",
            &[("region_id", "string"), ("region_name", "string")],
        ),
    ];
    let manifest = over_catalog(U0, &tables).await;
    let a = dim("channel").expect("a name").eq(lit("a"));
    let request = select(&["region_name", "revenue"]).with_predicate(a);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(datasets(&plan), ["u0.j.u3.e7", "u0.j.regions"]);
    for name in ["u0.d10", "u0.j.u3.e8"] {
        let name = DataKindName::new(name);
        traced_with(&plan.root, is_pruned(&name));
    }
}

fn pruned(plan: &SemanticPlan) -> Vec<&str> {
    (plan.root.pre_order())
        .flat_map(|node| &node.common().semantic_meta)
        .flat_map(|subject| match subject {
            SemAnnotation::SemanticSource { pruned, .. } => {
                pruned.iter().map(DataKindName::as_str).collect()
            }
            _ => Vec::new(),
        })
        .collect()
}

#[tokio::test]
async fn a_composed_branch_whose_branches_exclude_the_predicate_by_other_literals_is_read() {
    let sales = &[("region_id", "string"), ("amount", "double")][..];
    let tables = [
        ("warehouse.d10", sales),
        ("warehouse.e7", sales),
        ("warehouse.e8", sales),
        (
            "warehouse.regions",
            &[("region_id", "string"), ("region_name", "string")],
        ),
    ];
    let manifest = over_catalog(U0, &tables).await;
    let c = dim("channel").expect("a name").eq(lit("c"));
    let request = select(&["region_name", "revenue"]).with_predicate(c);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(datasets(&plan), ["u0.j.regions"]);
    let name = DataKindName::new("u0.j");
    traced_with(
        &plan.root,
        |subject| matches!(subject, SemAnnotation::SemanticSource { semantic_source, .. } if *semantic_source == name),
    );
    assert_eq!(pruned(&plan), ["u0.d10", "u0.j.u3.e7", "u0.j.u3.e8"]);
}

const SPEND_BY_FEED: &str = r#"
semantic_model:
  name: spend_by_feed
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: feed, data_type: string, type: {metadata: {source: {path: {token: 1}}}}}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
  unionsets:
    - name: spend_by_feed
      mode: all
      dimensions: [{ref: date, is_key: true}, {ref: feed, is_key: true}]
      measures: [{ref: clicks}, {ref: budget}]
      grainsets:
        - name: spend_web
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: web_daily
              extras:
                storage: {format: parquet, paths: ['s3://lake/daily/web/']}
                semantic_mapping: {date: day, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: web_monthly
              extras:
                storage: {format: parquet, paths: ['s3://lake/monthly/web/']}
                semantic_mapping: {date: month_start, budget: budget}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
      datasets:
        - name: spend_store
          extras:
            storage: {format: parquet, paths: ['s3://lake/store/spend/']}
            semantic_mapping: {date: day, clicks: clicks, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

#[tokio::test]
async fn a_grainset_branch_is_read_and_prunes_only_a_child_that_no_name_goes_to() {
    let objects = [
        "s3://lake/daily/web/part-0.parquet",
        "s3://lake/monthly/web/part-0.parquet",
        "s3://lake/store/spend/part-0.parquet",
    ];
    let manifest = compiled(SPEND_BY_FEED, &Catalog::default(), &seeded(&objects).await).await;
    let daily = dim("feed").expect("a name").eq(lit("daily"));
    let request = select(&["budget"]).with_predicate(daily);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(datasets(&plan), ["spend_by_feed.spend_web.web_monthly"]);
    assert_eq!(pruned(&plan), ["spend_by_feed.spend_store"]);
}

#[tokio::test]
async fn a_predicate_on_a_name_that_a_branch_null_fills_stays_above_the_unionset_it_reads() {
    let yaml = (U0.replace(
        "    - {name: region_name, data_type: string, type: categorical}\n",
        "    - {name: region_name, data_type: string, type: categorical}\n    - {name: device, data_type: string, type: categorical}\n",
    ))
    .replace("{ref: region_name}]", "{ref: region_name}, {ref: device}]")
    .replace(
        "warehouse.d10]}\n            semantic_mapping: {channel: {lit: b}, ",
        "warehouse.d10]}\n            semantic_mapping: {channel: {lit: b}, device: device, ",
    );
    let sales = &[("region_id", "string"), ("amount", "double")][..];
    let tables = [
        (
            "warehouse.d10",
            &[
                ("region_id", "string"),
                ("device", "string"),
                ("amount", "double"),
            ][..],
        ),
        ("warehouse.e7", sales),
        ("warehouse.e8", sales),
        (
            "warehouse.regions",
            &[("region_id", "string"), ("region_name", "string")],
        ),
    ];
    let manifest = over_catalog(&yaml, &tables).await;
    let no_device = dim("device").expect("a name").is_null();
    let request = select(&["revenue"]).with_predicate(no_device);

    let (plan, _) = planned(&manifest, &request);

    let [_, j] = two_branches(&plan);
    let device = format!("{}.device", alias(&manifest, "u0.j.u3"));
    let null = Some(Expr::Literal(Literal::Null(DataType::String)));
    for branch in &first_union(j).inputs {
        let PlanNode::Filter(filter) = &*aggregate_step(branch).input else {
            panic!("the predicate below the branch's partial aggregate, got {branch:?}");
        };
        let rows = &projection(&filter.input).input;
        let fills = |node: &PlanNode| matches!(node, PlanNode::Project(project) if written(project, &device) == null);
        assert!(
            rows.pre_order().any(fills),
            "{device} NULL-filled in {rows:?}"
        );
        let filtered = |node: &PlanNode| matches!(node, PlanNode::Filter(_));
        assert!(!rows.pre_order().any(filtered), "no predicate in {rows:?}");
    }
}

const U0_SHOPS: &str = r#"
semantic_model:
  name: u0
  dimensions:
    - {name: channel, data_type: string, type: categorical}
    - {name: device, data_type: string, type: categorical}
    - {name: region_id, data_type: string, type: categorical}
    - {name: region_name, data_type: string, type: categorical}
    - {name: shop_id, data_type: string, type: categorical}
    - {name: shop_name, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  unionsets:
    - name: u0
      mode: all
      dimensions:
        [{ref: channel}, {ref: device}, {ref: region_id}, {ref: region_name}, {ref: shop_id}, {ref: shop_name}]
      measures: [{ref: revenue}]
      datasets:
        - name: d10
          extras:
            storage: {tables: [warehouse.d10]}
            semantic_mapping: {channel: {lit: b}, device: device, region_id: region_id, revenue: amount}
      joinsets:
        - name: j
          unionsets:
            - name: u3
              mode: all
              datasets:
                - name: e8
                  extras:
                    storage: {tables: [warehouse.e8]}
                    semantic_mapping: {channel: {lit: b}, region_id: region_id, shop_name: shop_name, revenue: amount}
              joinsets:
                - name: ej
                  datasets:
                    - name: e7
                      extras:
                        storage: {tables: [warehouse.e7]}
                        semantic_mapping: {channel: {lit: a}, region_id: region_id, shop_id: shop_id, revenue: amount}
                    - name: shops
                      extras:
                        storage: {tables: [warehouse.shops]}
                        semantic_mapping: {shop_id: shop_id, shop_name: shop_name}
                  relationships:
                    - {name: e7_to_shops, from: e7, to: shops, on: [shop_id], cardinality: many_to_one, optional: none, integrity: assumed}
          datasets:
            - name: regions
              extras:
                storage: {tables: [warehouse.regions]}
                semantic_mapping: {region_id: region_id, region_name: region_name}
          relationships:
            - {name: u3_to_regions, from: u3, to: regions, on: [region_id], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

#[tokio::test]
async fn a_composed_branch_applies_the_predicate_that_its_unionset_keeps_on_a_null_fill() {
    let tables = [
        (
            "warehouse.d10",
            &[
                ("region_id", "string"),
                ("device", "string"),
                ("amount", "double"),
            ][..],
        ),
        (
            "warehouse.e8",
            &[
                ("region_id", "string"),
                ("shop_name", "string"),
                ("amount", "double"),
            ],
        ),
        (
            "warehouse.e7",
            &[
                ("region_id", "string"),
                ("shop_id", "string"),
                ("amount", "double"),
            ],
        ),
        (
            "warehouse.shops",
            &[("shop_id", "string"), ("shop_name", "string")],
        ),
        (
            "warehouse.regions",
            &[("region_id", "string"), ("region_name", "string")],
        ),
    ];
    let manifest = over_catalog(U0_SHOPS, &tables).await;
    let name = |name| dim(name).expect("a name");
    let predicate = name("device").eq(lit("x")) | name("shop_name").eq(lit("S1"));
    let request = select(&["shop_name", "revenue"]).with_predicate(predicate);

    let (plan, _) = planned(&manifest, &request);

    let [_, j] = two_branches(&plan);
    let [_, ej] = &first_union(j).inputs[..] else {
        panic!("u3's two branches, in name order, got {j:?}");
    };
    let PlanNode::Filter(predicate) = &*aggregate_step(ej).input else {
        panic!("the predicate below ej's partial aggregate, got {ej:?}");
    };
    let device = format!("{}.device", alias(&manifest, "u0.j.u3"));
    assert_eq!(
        written(projection(&predicate.input), &device),
        Some(Expr::Literal(Literal::Null(DataType::String)))
    );
}

const U1: &str = r#"
semantic_model:
  name: u1
  dimensions:
    - {name: channel, data_type: string, type: {metadata: {source: {path: {token: 1}}}}}
  measures:
    - {name: avg_amount, data_type: double, agg: avg}
  unionsets:
    - name: u1
      mode: all
      dimensions: [{ref: channel}]
      measures: [{ref: avg_amount}]
      datasets:
        - name: d9
          extras:
            storage: {format: parquet, paths: ['s3://lake/c/returns/']}
            semantic_mapping: {avg_amount: amount}
        - name: u2
          extras:
            storage: {format: parquet, paths: ['s3://lake/a/orders/', 's3://lake/c/orders/']}
            semantic_mapping: {avg_amount: amount}
"#;

#[tokio::test]
async fn a_branch_whose_reduction_leaves_give_two_literals_is_not_disjoint() {
    let objects = [
        "s3://lake/a/orders/part-0.parquet",
        "s3://lake/c/orders/part-0.parquet",
        "s3://lake/c/returns/part-0.parquet",
    ];
    let manifest = compiled(U1, &Catalog::default(), &seeded(&objects).await).await;

    let (plan, _) = planned(&manifest, &select(&["channel", "avg_amount"]));

    let (_, union) = union_of(&plan);
    let expected = SemAnnotation::SemanticSource {
        semantic_source: DataKindName::new("u1"),
        source_role: SourceRole::Unionset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: None,
        delivery: Some(Delivery::Rows),
        answered_by: None,
        pruned: Vec::new(),
    };
    let trace = &union.common.semantic_meta;
    assert!(trace.contains(&expected), "{expected:?} in {trace:?}");
    assert_eq!(union.inputs.len(), 2, "two branches in {union:?}");
    assert_eq!(
        aggregate_steps(&plan.root),
        1,
        "one aggregate step in {plan:?}"
    );
}

const ORDERS_BY_CHANNEL: &str = r#"
semantic_model:
  name: orders_by_channel
  dimensions:
    - {name: channel, data_type: string, type: {metadata: {source: {path: {token: 1}}}}}
  measures:
    - {name: avg_amount, data_type: double, agg: avg}
  datasets:
    - name: orders
      dimensions: [{ref: channel}]
      measures: [{ref: avg_amount}]
      extras:
        storage: {format: parquet, paths: ['s3://lake/a/orders/', 's3://lake/c/orders/']}
        semantic_mapping: {avg_amount: amount}
"#;

async fn orders_by_channel() -> SemanticManifest {
    let objects = [
        "s3://lake/a/orders/part-0.parquet",
        "s3://lake/c/orders/part-0.parquet",
    ];
    compiled(
        ORDERS_BY_CHANNEL,
        &Catalog::default(),
        &seeded(&objects).await,
    )
    .await
}

fn on_channel(names: &[&str], value: &str) -> Request {
    select(names).with_predicate(dim("channel").expect("a name").eq(lit(value)))
}

#[tokio::test]
async fn a_root_implicit_unionset_prunes_and_is_disjoint_by_a_path_token() {
    let manifest = orders_by_channel().await;

    let (pruned, _) = planned(&manifest, &on_channel(&["avg_amount"], "a"));
    let (disjoint, _) = planned(&manifest, &select(&["channel", "avg_amount"]));

    let reads: Vec<_> = (pruned.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Scan(scan) => Some(&scan.reader),
            _ => None,
        })
        .collect();
    let [Reader::FilesScan(files)] = &reads[..] else {
        panic!("one files read, got {reads:?}");
    };
    assert_eq!(files.base_path, "s3://lake/a/orders/");
    let unions = (pruned.root.pre_order()).filter(|node| matches!(node, PlanNode::Multiset(_)));
    assert_eq!(unions.count(), 0, "no union over one leaf");
    let leaf = DataKindName::new("orders_orders");
    let traced = (pruned.root.pre_order())
        .filter(|node| node.common().semantic_meta.iter().any(is_pruned(&leaf)));
    assert_eq!(traced.count(), 1, "{leaf:?} in {pruned:?}");
    let PlanNode::Multiset(union) = &disjoint.root else {
        panic!("the union at the root, got {:?}", disjoint.root);
    };
    assert_eq!(
        delivery_of(&union.common.semantic_meta),
        Some(Delivery::FinalAggregates)
    );
    let steps: Vec<_> = union.inputs.iter().map(aggregate_steps).collect();
    assert_eq!(
        steps,
        [1, 1],
        "one aggregate step in each branch, none above"
    );
}

#[tokio::test]
async fn a_root_implicit_unionset_with_every_leaf_excluded_reads_the_empty_read() {
    let manifest = orders_by_channel().await;

    for (names, groupings) in [
        (&["avg_amount"][..], Vec::new()),
        (&["channel", "avg_amount"], vec![column("channel")]),
    ] {
        let (plan, _) = planned(&manifest, &on_channel(names, "b"));

        let empty = scan_of(&plan);
        assert_eq!(empty.reader, VirtualTable::new(Vec::new()).into());
        assert_eq!(datasets(&plan), [] as [&str; 0], "no leaf is read");
        let leaf = DataKindName::new("orders_orders");
        let trace = &empty.common.semantic_meta;
        let pruned: usize = trace
            .iter()
            .map(|subject| match subject {
                SemAnnotation::SemanticSource { pruned, .. } => {
                    pruned.iter().filter(|name| **name == leaf).count()
                }
                _ => 0,
            })
            .sum();
        assert_eq!(pruned, 2, "each leaf in {trace:?}");
        assert_eq!(aggregate_step(&plan.root).groupings, groupings, "{names:?}");
    }
}

const SHARED_LITERALS: &str = r#"
semantic_model:
  name: shared_literals
  dimensions:
    - {name: k, data_type: string, type: categorical}
    - {name: x, data_type: string, type: categorical}
    - {name: y, data_type: string, type: categorical}
    - {name: source, data_type: string, type: categorical}
    - {name: platform, data_type: string, type: categorical}
  measures:
    - {name: cost, data_type: double, agg: sum}
    - {name: revenue, data_type: double, agg: sum}
    - {name: clicks, data_type: bigint, agg: sum}
  unionsets:
    - name: u1
      mode: all
      dimensions: [{ref: k}, {ref: x}, {ref: y}]
      measures: [{ref: cost}, {ref: revenue}]
      datasets:
        - name: d1
          extras:
            storage: {tables: [warehouse.d1]}
            semantic_mapping: {k: k, x: x, y: y, cost: cost, revenue: revenue}
      joinsets:
        - name: j1
          datasets:
            - name: a1
              extras:
                storage: {tables: [warehouse.a1]}
                semantic_mapping: {k: k, x: {lit: a}, y: {lit: {null: string}}, cost: cost}
            - name: b1
              extras:
                storage: {tables: [warehouse.b1]}
                semantic_mapping: {k: k, x: {lit: {null: string}}, y: {lit: b}, revenue: revenue}
          relationships:
            - {name: a1_to_b1, from: a1, to: b1, on: [k], cardinality: many_to_one, optional: none, integrity: assumed}
    - name: u3
      mode: all
      dimensions: [{ref: k}, {ref: source}]
      measures: [{ref: cost}, {ref: revenue}]
      datasets:
        - name: d3
          extras:
            storage: {tables: [warehouse.d3]}
            semantic_mapping: {k: k, source: {lit: shop}, cost: cost, revenue: revenue}
      joinsets:
        - name: j3
          datasets:
            - name: a3
              extras:
                storage: {tables: [warehouse.a3]}
                semantic_mapping: {k: k, source: {lit: erp}, cost: cost}
            - name: b3
              extras:
                storage: {tables: [warehouse.b3]}
                semantic_mapping: {k: k, source: {lit: stripe}, revenue: revenue}
          relationships:
            - {name: a3_to_b3, from: a3, to: b3, on: [k], cardinality: many_to_one, optional: none, integrity: assumed}
    - name: u10
      mode: all
      dimensions: [{ref: k}, {ref: platform}]
      measures: [{ref: clicks}]
      joinsets:
        - name: jf
          datasets:
            - name: fs
              extras:
                storage: {tables: [warehouse.fs]}
                semantic_mapping: {k: k, platform: {lit: facebook}, clicks: clicks}
            - name: fc
              extras:
                storage: {tables: [warehouse.fc]}
                semantic_mapping: {k: k, platform: {lit: facebook}}
          relationships:
            - {name: fs_to_fc, from: fs, to: fc, on: [k], cardinality: many_to_one, optional: none, integrity: assumed}
        - name: jg
          datasets:
            - name: gs
              extras:
                storage: {tables: [warehouse.gs]}
                semantic_mapping: {k: k, platform: {lit: google}, clicks: clicks}
            - name: gc
              extras:
                storage: {tables: [warehouse.gc]}
                semantic_mapping: {k: k, platform: {lit: google}}
          relationships:
            - {name: gs_to_gc, from: gs, to: gc, on: [k], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

async fn shared_literals() -> SemanticManifest {
    let key = &[("k", "string")][..];
    let cost = &[("k", "string"), ("cost", "double")][..];
    let revenue = &[("k", "string"), ("revenue", "double")][..];
    let both = &[("k", "string"), ("cost", "double"), ("revenue", "double")][..];
    let clicks = &[("k", "string"), ("clicks", "long")][..];
    let tables = [
        ("warehouse.a1", cost),
        ("warehouse.b1", revenue),
        (
            "warehouse.d1",
            &[
                ("k", "string"),
                ("x", "string"),
                ("y", "string"),
                ("cost", "double"),
                ("revenue", "double"),
            ][..],
        ),
        ("warehouse.a3", cost),
        ("warehouse.b3", revenue),
        ("warehouse.d3", both),
        ("warehouse.fs", clicks),
        ("warehouse.fc", key),
        ("warehouse.gs", clicks),
        ("warehouse.gc", key),
    ];
    over_catalog(SHARED_LITERALS, &tables).await
}

#[tokio::test]
async fn a_name_that_two_operands_give_two_literals_folds_to_none() {
    let manifest = shared_literals().await;
    let null = |name: &str| dim(name).expect("a name").is_null();
    let request = select(&["cost", "revenue"])
        .with_from("u1")
        .with_predicate(null("x") | null("y"));

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(datasets(&plan), ["u1.d1", "u1.j1.a1", "u1.j1.b1"]);
}

#[tokio::test]
async fn a_joinset_branch_whose_operands_differ_on_a_literal_is_read() {
    let manifest = shared_literals().await;
    let stripe = dim("source").expect("a name").eq(lit("stripe"));
    let request = select(&["revenue"]).with_from("u3").with_predicate(stripe);

    let (plan, _) = planned(&manifest, &request);

    one_branch_left(&plan, "u3.j3.b3", "u3.d3");
}

#[tokio::test]
async fn a_joinset_branch_whose_operands_agree_on_a_literal_is_pruned_by_it() {
    let manifest = shared_literals().await;
    let google = dim("platform").expect("a name").eq(lit("google"));
    let request = select(&["clicks"]).with_from("u10").with_predicate(google);

    let (plan, _) = planned(&manifest, &request);

    one_branch_left(&plan, "u10.jg.gs", "u10.jf");
}

const ALL_ORDERS: &str = r#"
semantic_model:
  name: all_orders
  dimensions:
    - {name: channel, data_type: string, type: categorical}
    - {name: customer_id, data_type: string, type: categorical}
  measures:
    - {name: customer_rows, data_type: bigint, agg: count}
  unionsets:
    - name: all_orders
      mode: all
      dimensions: [{ref: channel}, {ref: customer_id}]
      measures: [{ref: customer_rows}]
      datasets:
        - name: orders_store
          extras:
            storage: {tables: [warehouse.orders_store]}
            semantic_mapping: {channel: {lit: store}, customer_id: customer_id, customer_rows: order_id}
      joinsets:
        - name: jw
          datasets:
            - name: orders_web
              extras:
                storage: {tables: [warehouse.orders_web]}
                semantic_mapping: {channel: {lit: web}, customer_id: customer_id}
            - name: customers
              extras:
                storage: {tables: [warehouse.customers]}
                semantic_mapping: {customer_id: customer_id, customer_rows: customer_id}
          relationships:
            - {name: web_to_customers, from: orders_web, to: customers, on: [customer_id], cardinality: many_to_one, optional: right, integrity: none}
"#;

async fn all_orders(yaml: &str) -> SemanticManifest {
    let orders = &[("order_id", "string"), ("customer_id", "string")][..];
    let tables = [
        ("warehouse.orders_store", orders),
        ("warehouse.orders_web", orders),
        ("warehouse.customers", &[("customer_id", "string")]),
    ];
    over_catalog(yaml, &tables).await
}

#[tokio::test]
async fn the_fold_stops_at_an_operand_that_its_join_null_fills() {
    let manifest = all_orders(ALL_ORDERS).await;
    let channel = || dim("channel").expect("a name");
    let jw = ["all_orders.jw.customers", "all_orders.jw.orders_web"];
    let with_store = [
        "all_orders.jw.customers",
        "all_orders.jw.orders_web",
        "all_orders.orders_store",
    ];

    for (predicate, read) in [
        (channel().is_null(), &jw[..]),
        (channel().eq(lit("store")), &with_store),
    ] {
        let request = select(&["customer_id", "customer_rows"]).with_predicate(predicate);
        let (plan, _) = planned(&manifest, &request);

        let mut datasets = datasets(&plan);
        datasets.sort_unstable();
        assert_eq!(datasets, read);
    }
}

#[tokio::test]
async fn a_joinset_branch_that_builds_its_join_gives_the_union_its_own_partial_aggregates() {
    let manifest = all_orders(ALL_ORDERS).await;

    let (plan, _) = planned(&manifest, &select(&["channel", "customer_rows"]));

    let join = the_join(&plan);
    assert_eq!(aggregate_steps(&join.left), 1);
    assert_eq!(aggregate_steps(&join.right), 1);
    let [jw, _] = two_branches(&plan);
    let [(entry, partial)] = aggregate_step(jw).aggregates.as_slice() else {
        panic!("one partial count, got {jw:?}");
    };
    let alias = &alias(&manifest, "all_orders");
    assert_eq!(entry.as_str(), format!("{alias}.customer_rows"));
    assert_eq!(partial.function().declarations.canonical.name, "sum");
    let aggregate = aggregate_step(&projection(&plan.root).input);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        qualified(alias, "customer_rows")
    );
}

#[tokio::test]
async fn a_joinset_branch_gives_a_metrics_formula_node_as_the_partial_of_its_measure() {
    let metric = "  metrics:\n    - name: doubled_rows\n      data_type: bigint\n      expr:\n        multiply: [customer_rows, {lit: 2}]\n  unionsets:";
    let measures = "measures: [{ref: customer_rows}]";
    let yaml = (ALL_ORDERS.replacen("  unionsets:", metric, 1)).replacen(
        measures,
        &format!("{measures}\n      metrics: [{{ref: doubled_rows}}]"),
        1,
    );
    let manifest = all_orders(&yaml).await;

    let (plan, _) = planned(&manifest, &select(&["channel", "doubled_rows"]));

    let alias = &alias(&manifest, "all_orders");
    let node = format!("{alias}.__plan_agg1_doubled_rows");
    let [jw, _] = two_branches(&plan);
    let given = projection(jw);
    assert_eq!(
        written(given, &node),
        Some(qualified(alias, "customer_rows"))
    );
    assert!(
        written(given, &format!("{alias}.customer_rows")).is_none(),
        "only the node reads `customer_rows`: {given:?}"
    );
    assert!(matches!(given.input.as_ref(), PlanNode::Aggregate(_)));
    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let partial = computed(projection(&aggregate.input), &sum.args()[0]);
    assert_eq!(partial, qualified(alias, "__plan_agg1_doubled_rows"));
    assert_eq!(output_names(&plan), ["channel", "doubled_rows"]);
}

fn all_orders_cast(metrics: &str, site: &str) -> String {
    let cast = ALL_ORDERS.replacen("data_type: bigint", "data_type: double", 1);
    let measures = "measures: [{ref: customer_rows}]";
    (cast.replacen("  unionsets:", metrics, 1)).replacen(measures, &format!("{measures}{site}"), 1)
}

const DOUBLED_ROWS_DOUBLE: &str = "  metrics:\n    - name: doubled_rows\n      data_type: double\n      expr:\n        multiply: [customer_rows, {lit: 2}]\n  unionsets:";
const WITH_DOUBLED_ROWS: &str = "\n      metrics: [{ref: doubled_rows}]";

#[tokio::test]
async fn a_joinset_branch_gives_a_cast_measures_partial_in_the_entry_that_the_union_reads() {
    let manifest = all_orders(&all_orders_cast("  unionsets:", "")).await;

    let (plan, _) = planned(&manifest, &select(&["channel", "customer_rows"]));

    let alias = &alias(&manifest, "all_orders");
    let [jw, _] = two_branches(&plan);
    let given = projection(jw);
    assert_eq!(
        written(given, &format!("{alias}.__plan_agg1_customer_rows")),
        Some(qualified(alias, "customer_rows"))
    );
    assert!(
        written(given, &format!("{alias}.customer_rows")).is_none(),
        "only the reserved entry reads the partial: {given:?}"
    );
    assert_one_partial_entry(given, &format!("{alias}.customer_rows"));
    let aggregate = aggregate_step(&projection(&plan.root).input);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        qualified(alias, "__plan_agg1_customer_rows")
    );
    assert_eq!(output_names(&plan), ["channel", "customer_rows"]);
}

fn assert_one_partial_entry(given: &ProjectNode, entry: &str) {
    let step = aggregate_step(&given.input);
    let [(held, _)] = step.aggregates.as_slice() else {
        panic!("one partial count, got {:?}", step.aggregates);
    };
    assert_eq!(held.as_str(), entry);
}

#[tokio::test]
async fn a_joinset_branch_gives_a_cast_measure_and_a_metrics_node_from_one_partial() {
    let yaml = all_orders_cast(DOUBLED_ROWS_DOUBLE, WITH_DOUBLED_ROWS);
    let manifest = all_orders(&yaml).await;

    let request = select(&["channel", "customer_rows", "doubled_rows"]);
    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "all_orders");
    let [jw, _] = two_branches(&plan);
    let given = projection(jw);
    for name in ["customer_rows", "doubled_rows"] {
        assert_eq!(
            written(given, &format!("{alias}.__plan_agg1_{name}")),
            Some(qualified(alias, "customer_rows"))
        );
    }
    assert!(
        written(given, &format!("{alias}.customer_rows")).is_none(),
        "no plain entry: {given:?}"
    );
    assert_one_partial_entry(given, &format!("{alias}.customer_rows"));
    assert_eq!(
        output_names(&plan),
        ["channel", "customer_rows", "doubled_rows"]
    );
}

#[tokio::test]
async fn a_cast_measure_that_a_predicate_reads_takes_the_unions_entry() {
    let yaml = all_orders_cast(DOUBLED_ROWS_DOUBLE, WITH_DOUBLED_ROWS);
    let manifest = all_orders(&yaml).await;
    let kept = measure("customer_rows").expect("a name").gt(lit(1));

    let request = select(&["channel", "doubled_rows"]).with_predicate(kept);
    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "all_orders");
    let [jw, _] = two_branches(&plan);
    let given = projection(jw);
    for name in ["customer_rows", "doubled_rows"] {
        assert_eq!(
            written(given, &format!("{alias}.__plan_agg1_{name}")),
            Some(qualified(alias, "customer_rows"))
        );
    }
    assert!(
        written(given, &format!("{alias}.customer_rows")).is_none(),
        "no plain entry: {given:?}"
    );
    assert_one_partial_entry(given, &format!("{alias}.customer_rows"));
    assert_eq!(output_names(&plan), ["channel", "doubled_rows"]);
}

const REGION_MODEL: &str = r#"
semantic_model:
  name: top
  dimensions:
    - {name: channel, data_type: string, type: categorical}
    - {name: customer_id, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: customer_rows, data_type: double, agg: count}
    - {name: cost, data_type: 'decimal(18,2)', agg: sum}
  joinsets:
    - name: top
      dimensions: [{ref: channel}, {ref: customer_id}, {ref: region}]
      measures: [{ref: customer_rows}, {ref: cost}]
      datasets:
        - name: customer_regions
          extras:
            storage: {tables: [warehouse.customer_regions]}
            semantic_mapping: {customer_id: customer_id, region: region}
      unionsets:
        - name: all_orders
          mode: all
          datasets:
            - name: orders_store
              extras:
                storage: {tables: [warehouse.orders_store]}
                semantic_mapping: {channel: {lit: store}, customer_id: customer_id, customer_rows: order_id, cost: cost}
          joinsets:
            - name: jw
              datasets:
                - name: orders_web
                  extras:
                    storage: {tables: [warehouse.orders_web]}
                    semantic_mapping: {channel: {lit: web}, customer_id: customer_id, cost: cost}
                - name: customers
                  extras:
                    storage: {tables: [warehouse.customers]}
                    semantic_mapping: {customer_id: customer_id, customer_rows: customer_id}
              relationships:
                - {name: web_to_customers, from: orders_web, to: customers, on: [customer_id], cardinality: many_to_one, optional: right, integrity: none}
      relationships:
        - {name: orders_to_regions, from: all_orders, to: customer_regions, on: [customer_id], cardinality: many_to_one, optional: left, integrity: none}
"#;

const REGION_OVER_DATASET: &str = r#"
semantic_model:
  name: oc
  dimensions:
    - {name: customer_id, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: customer_rows, data_type: double, agg: count}
    - {name: cost, data_type: 'decimal(18,2)', agg: sum}
  joinsets:
    - name: oc
      dimensions: [{ref: customer_id}, {ref: region}]
      measures: [{ref: customer_rows}, {ref: cost}]
      datasets:
        - name: orders
          extras:
            storage: {tables: [warehouse.orders_web]}
            semantic_mapping: {customer_id: customer_id, customer_rows: order_id, cost: cost}
        - name: customer_regions
          extras:
            storage: {tables: [warehouse.customer_regions]}
            semantic_mapping: {customer_id: customer_id, region: region}
      relationships:
        - {name: orders_to_regions, from: orders, to: customer_regions, on: [customer_id], cardinality: many_to_one, optional: left, integrity: none}
"#;

async fn over_regions(yaml: &str) -> SemanticManifest {
    let orders = &[
        ("order_id", "string"),
        ("customer_id", "string"),
        ("cost", "decimal(18,2)"),
    ][..];
    let tables = [
        ("warehouse.orders_store", orders),
        ("warehouse.orders_web", orders),
        ("warehouse.customers", &[("customer_id", "string")][..]),
        (
            "warehouse.customer_regions",
            &[("customer_id", "string"), ("region", "string")][..],
        ),
    ];
    over_catalog(yaml, &tables).await
}

#[tokio::test]
async fn a_joinset_over_a_unionset_reads_a_cast_measures_partial_under_its_own_name() {
    let manifest = over_regions(REGION_MODEL).await;

    let (plan, _) = planned(&manifest, &select(&["region", "cost"]));

    let alias = &alias(&manifest, "top.all_orders");
    let reserved = format!("{alias}.__plan_agg1_cost");
    let writes = (plan.root.pre_order()).any(|node| match node {
        PlanNode::Project(project) => written(project, &reserved).is_some(),
        _ => false,
    });
    assert!(writes, "a branch writes {reserved}: {plan:?}");
    let plain = format!("{alias}.cost");
    let outer = (plan.root.pre_order())
        .find(|node| matches!(node, PlanNode::Join(_)))
        .map(join)
        .expect("the outer join");
    let left = &outer.left;
    let reads = (left.pre_order()).any(|node| match node {
        PlanNode::Aggregate(step) => step
            .aggregates
            .iter()
            .any(|(entry, _)| entry.as_str() == plain),
        _ => false,
    });
    assert!(reads, "the join's left side gives {plain}: {left:?}");
    assert_eq!(output_names(&plan), ["region", "cost"]);
}

#[tokio::test]
async fn a_joinset_over_a_dataset_reads_the_sum_at_the_type_that_it_gives() {
    let manifest = over_regions(REGION_OVER_DATASET).await;

    let (plan, _) = planned(&manifest, &select(&["region", "cost"]));

    let orders = alias(&manifest, "oc.orders");
    let column = format!("{orders}.cost");
    let left = &the_join(&plan).left.common().output_schema;
    let field = (left.columns.iter()).find(|field| field.name == column);
    let wide = DataType::Decimal {
        precision: 38,
        scale: 2,
    };
    assert_eq!(field.map(|field| &field.data_type), Some(&wide));
    assert_eq!(output_names(&plan), ["region", "cost"]);
}

#[tokio::test]
async fn a_metric_filter_over_a_composed_branch_is_the_condition_of_its_measure() {
    let metric = "  metrics:\n    - name: doubled_rows\n      data_type: bigint\n      expr:\n        multiply: [customer_rows, {lit: 2}]\n  unionsets:";
    let measures = "measures: [{ref: customer_rows}]";
    let site = "metrics: [{ref: doubled_rows, filter: {eq: [customer_id, {lit: c1}]}}]";
    let yaml = (ALL_ORDERS.replacen("  unionsets:", metric, 1)).replacen(
        measures,
        &format!("{measures}\n      {site}"),
        1,
    );
    let manifest = all_orders(&yaml).await;

    let (plan, _) = planned(&manifest, &select(&["channel", "doubled_rows"]));

    let alias = &alias(&manifest, "all_orders");
    let [jw, _] = two_branches(&plan);
    let given = projection(jw);
    assert_eq!(
        written(given, &format!("{alias}.__plan_agg1_doubled_rows")),
        Some(qualified(alias, "customer_rows"))
    );
    let step = aggregate_step(&given.input);
    let [(_, count)] = step.aggregates.as_slice() else {
        panic!("one partial count, got {:?}", step.aggregates);
    };
    assert!(
        count.filter().is_some(),
        "under the Metric's filter: {count:?}"
    );
    let aggregate = aggregate_step(&projection(&plan.root).input);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(output_names(&plan), ["channel", "doubled_rows"]);
}

const SPEND_BY_CHANNEL: &str = r#"
semantic_model:
  name: spend_by_channel
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
    - {name: orders, data_type: bigint, agg: count}
  unionsets:
    - name: spend_by_channel
      mode: all
      dimensions: [{ref: date, is_key: true}, {ref: channel, is_key: true}]
      measures: [{ref: clicks}, {ref: budget}, {ref: orders}]
      grainsets:
        - name: spend_web
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: web_daily
              extras:
                storage: {tables: [warehouse.web_daily]}
                semantic_mapping: {date: day, channel: {lit: web}, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: web_monthly
              extras:
                storage: {tables: [warehouse.web_monthly]}
                semantic_mapping: {date: month_start, channel: {lit: web}, budget: budget}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
      datasets:
        - name: spend_store
          extras:
            storage: {tables: [warehouse.spend_store]}
            semantic_mapping:
              {date: day, channel: {lit: store}, clicks: clicks, budget: budget, orders: order_id}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

async fn spend_by_channel(yaml: &str) -> SemanticManifest {
    spend_by_channel_over(yaml, &[]).await
}

async fn spend_by_channel_over(yaml: &str, extra: &[(&str, &[(&str, &str)])]) -> SemanticManifest {
    let mut tables: Vec<(&str, &[(&str, &str)])> = vec![
        (
            "warehouse.web_daily",
            &[
                ("day", "date"),
                ("campaign_id", "string"),
                ("clicks", "long"),
            ][..],
        ),
        (
            "warehouse.web_monthly",
            &[
                ("month_start", "date"),
                ("campaign_id", "string"),
                ("clicks", "long"),
                ("budget", "long"),
            ],
        ),
        (
            "warehouse.spend_store",
            &[
                ("day", "date"),
                ("campaign_id", "string"),
                ("clicks", "long"),
                ("budget", "long"),
                ("order_id", "string"),
                ("region", "string"),
            ],
        ),
    ];
    tables.extend_from_slice(extra);
    over_catalog(yaml, &tables).await
}

fn drill_across(branch: &PlanNode) -> (&ProjectNode, &JoinNode) {
    (branch.pre_order())
        .find_map(|node| match node {
            PlanNode::Project(projection) => match &*projection.input {
                PlanNode::Join(join) => Some((projection, join)),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or_else(|| panic!("a projection above the drill-across in {branch:?}"))
}

#[tokio::test]
async fn a_disjoint_grainset_branch_drills_across_on_its_own() {
    let manifest = spend_by_channel(SPEND_BY_CHANNEL).await;
    let rest = ["channel", "orders", "clicks", "budget"];
    let request = at_granularity("date", Granularity::Month, &rest);

    let (plan, _) = planned(&manifest, &request);

    let branches = two_branches(&plan);
    let below = branches.iter().map(|branch| aggregate_steps(branch));
    assert_eq!(aggregate_steps(&plan.root), below.sum::<usize>());
    let [_, spend_web] = branches;
    let (drill_across, join) = drill_across(spend_web);
    assert_eq!(join.join_type, JoinType::Full);
    let orders = format!("{}.orders", alias(&manifest, "spend_by_channel.spend_web"));
    assert_eq!(
        written(drill_across, &orders),
        Some(Expr::Literal(Literal::Null(DataType::BigInt)))
    );
}

#[tokio::test]
async fn a_metric_over_a_drill_across_branch_reads_its_measures_as_the_formulas_nodes() {
    let yaml = with_click_rate(SPEND_BY_CHANNEL);
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "click_rate"]);

    let (plan, _) = planned(&manifest, &request);

    let branches = two_branches(&plan);
    let below = branches.iter().map(|branch| aggregate_steps(branch));
    assert_eq!(aggregate_steps(&plan.root), below.sum::<usize>());
    let [_, spend_web] = branches;
    let nodes = |entry: &str| {
        (spend_web.pre_order()).find_map(|node| match node {
            PlanNode::Project(projection) => {
                written(projection, entry).filter(|tree| *tree != column(entry))
            }
            _ => None,
        })
    };
    let node = |measure: &str| Some(column(&format!("__plan_node1_{measure}")));
    assert_eq!(nodes("__plan_agg1_click_rate"), node("clicks"));
    assert_eq!(nodes("__plan_agg2_click_rate"), node("budget"));
    assert_eq!(output_names(&plan), ["date", "channel", "click_rate"]);
}

#[tokio::test]
async fn a_formula_node_of_a_measure_that_a_drill_across_branch_null_fills_is_a_typed_null() {
    let metric = "  metrics:\n    - name: mix\n      data_type: double\n      expr:\n        divide: [{add: [clicks, orders]}, budget]\n  unionsets:";
    let measures = "measures: [{ref: clicks}, {ref: budget}, {ref: orders}]";
    let yaml = (SPEND_BY_CHANNEL.replacen("  unionsets:", metric, 1)).replacen(
        measures,
        &format!("{measures}\n      metrics: [{{ref: mix}}]"),
        1,
    );
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "mix"]);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_web] = two_branches(&plan);
    let node = (spend_web.pre_order()).find_map(|node| match node {
        PlanNode::Project(projection) => written(projection, "__plan_agg2_mix"),
        _ => None,
    });
    assert_eq!(node, Some(Expr::Literal(Literal::Null(DataType::BigInt))));
    assert_eq!(output_names(&plan), ["date", "channel", "mix"]);
}

#[tokio::test]
async fn a_drill_across_branch_gives_a_metrics_formula_node_below_its_measures_cast() {
    let yaml = with_click_rate(SPEND_BY_CHANNEL).replacen(
        "{name: clicks, data_type: bigint, agg: sum}",
        "{name: clicks, data_type: double, agg: count}",
        1,
    );
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "click_rate"]);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_web] = two_branches(&plan);
    let entry = "__plan_agg1_click_rate";
    let node = (spend_web.pre_order()).find_map(|node| match node {
        PlanNode::Project(projection) => {
            written(projection, entry).filter(|tree| *tree != column(entry))
        }
        _ => None,
    });
    assert_eq!(node, Some(column("__plan_node1_clicks")));
    let types = branch_types(&plan, entry);
    assert_eq!(types, [DataType::BigInt, DataType::BigInt]);
    assert_eq!(output_names(&plan), ["date", "channel", "click_rate"]);
}

fn with_campaign(yaml: &str) -> String {
    let campaign = "    - {name: campaign, data_type: string, type: categorical}\n";
    yaml.replacen("  measures:\n", &format!("{campaign}  measures:\n"), 1)
        .replacen(
            "{ref: channel, is_key: true}]",
            "{ref: channel, is_key: true}, {ref: campaign}]",
            1,
        )
}

fn campaign_filtered() -> String {
    [
        (
            "[{ref: clicks},",
            "[{ref: clicks, filter: {eq: [campaign, {lit: A}]}},",
        ),
        ("{date: day,", "{campaign: campaign_id, date: day,"),
    ]
    .iter()
    .fold(with_campaign(SPEND_BY_CHANNEL), |yaml, (from, to)| {
        yaml.replace(from, to)
    })
}

fn drilled_clicks(manifest: &SemanticManifest, plan: &SemanticPlan) -> Option<PhysicalExpr> {
    let [_, spend_web] = two_branches(plan);
    let clicks = format!("{}.clicks", alias(manifest, "spend_by_channel.spend_web"));
    written(drill_across(spend_web).0, &clicks)
}

#[tokio::test]
async fn a_unionsets_own_filter_is_a_condition_on_the_drill_across_child_that_serves_it() {
    let manifest = spend_by_channel(&campaign_filtered()).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "clicks", "budget"]);

    let (plan, _) = planned(&manifest, &request);

    let web_daily = &alias(&manifest, "spend_by_channel.spend_web.web_daily");
    assert_eq!(
        drilled_clicks(&manifest, &plan),
        Some(qualified(web_daily, "__plan_filter_clicks"))
    );
    let [_, spend_web] = two_branches(&plan);
    let reads_campaign = |tree: &PhysicalExpr| {
        (tree.pre_order()).any(|node| matches!(node, Expr::Column(c) if c.name() == "campaign"))
    };
    for node in spend_web.pre_order() {
        match node {
            PlanNode::Join(join) => assert!(!reads_campaign(&join.on), "{join:?}"),
            PlanNode::Aggregate(aggregate) => {
                assert!(
                    !aggregate.groupings.iter().any(reads_campaign),
                    "{aggregate:?}"
                )
            }
            _ => {}
        }
    }
    let daily = format!("{web_daily}.clicks");
    let aggregate = (spend_web.pre_order())
        .find_map(|node| match node {
            PlanNode::Aggregate(aggregate) => (aggregate.aggregates.iter())
                .any(|(entry, _)| entry.as_str() == daily)
                .then_some(aggregate),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the aggregate of {daily} in {spend_web:?}"));
    let [(_, clicks)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let Some(Expr::Column(condition)) = clicks.filter() else {
        panic!("a condition field on {clicks:?}");
    };
    let (condition, campaign_id) = (
        field_name(condition),
        columns_of(&manifest, "web_daily")("campaign_id"),
    );
    let reads = (aggregate.input.pre_order()).any(|node| match node {
        PlanNode::Project(project) => written(project, &condition)
            .is_some_and(|tree| tree.pre_order().any(|node| node == &campaign_id)),
        _ => false,
    });
    assert!(reads, "{condition} reads {campaign_id:?} in {aggregate:?}");
}

#[tokio::test]
async fn a_child_that_does_not_map_a_conditions_dimension_does_not_answer_alone() {
    let monthly = "{date: month_start, channel: {lit: web}, budget: budget}";
    let yaml = campaign_filtered().replace(
        monthly,
        "{date: month_start, channel: {lit: web}, clicks: clicks, budget: budget}",
    );
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "clicks", "budget"]);

    let (plan, _) = planned(&manifest, &request);

    let mut read = datasets(&plan);
    read.sort_unstable();
    assert_eq!(
        read,
        [
            "spend_by_channel.spend_store",
            "spend_by_channel.spend_web.web_daily",
            "spend_by_channel.spend_web.web_monthly"
        ]
    );
    let web_daily = &alias(&manifest, "spend_by_channel.spend_web.web_daily");
    assert_eq!(
        drilled_clicks(&manifest, &plan),
        Some(qualified(web_daily, "__plan_filter_clicks"))
    );
}

#[tokio::test]
async fn a_predicate_on_a_dimension_goes_into_each_child_of_a_drill_across_branch() {
    let manifest = spend_by_channel(SPEND_BY_CHANNEL).await;
    let day = Literal::Date("2026-03-15".into());
    let since = dim("date")
        .expect("a name")
        .gt_eq(Expr::Literal(day.clone()));
    let rest = ["channel", "clicks", "budget"];
    let request = at_granularity("date", Granularity::Month, &rest).with_predicate(since);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_web] = two_branches(&plan);
    let (_, join) = drill_across(spend_web);
    assert_eq!(join.join_type, JoinType::Full);
    let filters: Vec<_> = (spend_web.pre_order())
        .filter_map(|node| match node {
            PlanNode::Filter(filter) => Some(filter.filter.clone()),
            _ => None,
        })
        .collect();
    let child = |name: &str| {
        let alias = &alias(&manifest, &format!("spend_by_channel.spend_web.{name}"));
        qualified(alias, "date").gt_eq(Expr::Literal(day.clone()))
    };
    assert_eq!(filters, [child("web_monthly"), child("web_daily")]);
}

#[tokio::test]
async fn a_predicate_on_a_null_filled_dimension_reads_its_typed_null_in_each_drill_across_child() {
    let region = "    - {name: region, data_type: string, type: categorical}\n";
    let yaml = [
        ("  measures:\n", &format!("{region}  measures:\n")[..]),
        (
            "{ref: channel, is_key: true}]",
            "{ref: channel, is_key: true}, {ref: region}]",
        ),
        (
            "channel: {lit: store},",
            "channel: {lit: store}, region: region,",
        ),
    ]
    .iter()
    .fold(SPEND_BY_CHANNEL.to_owned(), |yaml, (from, to)| {
        yaml.replace(from, to)
    });
    let manifest = spend_by_channel(&yaml).await;
    let since = Literal::Date("2026-03-15".into());
    let eu = dim("region").expect("a name").eq(lit("eu"))
        | dim("date")
            .expect("a name")
            .gt_eq(Expr::Literal(since.clone()));
    let rest = ["channel", "clicks", "budget"];
    let request = at_granularity("date", Granularity::Month, &rest).with_predicate(eu);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_web] = two_branches(&plan);
    let (_, join) = drill_across(spend_web);
    assert_eq!(join.join_type, JoinType::Full);
    let filters: Vec<_> = (spend_web.pre_order())
        .filter_map(|node| match node {
            PlanNode::Filter(filter) => Some(filter.filter.clone()),
            _ => None,
        })
        .collect();
    let child = |name: &str| {
        let alias = &alias(&manifest, &format!("spend_by_channel.spend_web.{name}"));
        Expr::Literal(Literal::Null(DataType::String)).eq(text("eu"))
            | qualified(alias, "date").gt_eq(Expr::Literal(since.clone()))
    };
    assert_eq!(filters, [child("web_monthly"), child("web_daily")]);
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        let mut goldens = budget_at_day().await;
        goldens.extend(clicks_and_budget().await);
        goldens
    })
    .await;
}

fn branch_delivery(union: &MultisetNode) -> Delivery {
    (union.common.semantic_meta.iter())
        .find_map(|annotation| match annotation {
            SemAnnotation::SemanticSource { delivery, .. } => *delivery,
            _ => None,
        })
        .unwrap_or_else(|| panic!("a branch delivery in {:?}", union.common.semantic_meta))
}

fn branch_step(branch: &PlanNode) -> (Vec<String>, Vec<String>) {
    let step = first_aggregate(branch);
    let mut grouped: Vec<_> = (step.groupings.iter())
        .map(|grouping| match grouping {
            Expr::Column(column) => column.name().to_owned(),
            other => panic!("a field, got {other:?}"),
        })
        .collect();
    grouped.sort();
    let entries = (step.aggregates.iter())
        .map(|(name, _)| unqualified(name.as_str()))
        .collect();
    (grouped, entries)
}

fn unqualified(name: &str) -> String {
    name.rsplit('.').next().unwrap_or_default().to_owned()
}

fn unions(plan: &SemanticPlan) -> Vec<&MultisetNode> {
    (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Multiset(union) => Some(union),
            _ => None,
        })
        .collect()
}

fn steps_above_unions(plan: &SemanticPlan) -> usize {
    let below: usize = (unions(plan).into_iter())
        .flat_map(|union| union.inputs.iter())
        .map(aggregate_steps)
        .sum();
    aggregate_steps(&plan.root) - below
}

fn joins(plan: &SemanticPlan) -> usize {
    (plan.root.pre_order())
        .filter(|node| matches!(node, PlanNode::Join(_)))
        .count()
}

const STEPPED_ORDERS: &str = r#"
semantic_model:
  name: split_orders
  dimensions:
    - {name: region, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
    - {name: customer_id, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
    - {name: buyers, data_type: bigint, agg: count_distinct}
  metrics:
    - name: revenue_eu_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {add: [revenue, {lit: 100}]}
          else: revenue
    - name: revenue_web_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {add: [revenue, {lit: 100}]}
          else: revenue
    - name: revenue_web_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
    - {name: max_rev, data_type: double, agg: max, expr: revenue}
    - {name: max_buyers, data_type: bigint, agg: max, expr: buyers}
    - name: share_of_max
      data_type: double
      expr: {divide: [revenue, max_rev]}
  unionsets:
    - name: split_orders
      mode: all
      dimensions: [{ref: region}, {ref: channel}, {ref: customer_id}]
      measures: [{ref: revenue}, {ref: buyers}]
      metrics:
        - {ref: revenue_eu_bonus}
        - {ref: revenue_web_bonus}
        - {ref: revenue_web_adj}
        - {ref: max_rev}
        - {ref: max_buyers}
        - {ref: share_of_max}
      datasets:
        - name: eu_orders
          extras:
            storage: {tables: [warehouse.eu_orders]}
            semantic_mapping: {region: REGION_EU, channel: channel, customer_id: customer_id, revenue: amount, buyers: customer_id}
        - name: us_orders
          extras:
            storage: {tables: [warehouse.us_orders]}
            semantic_mapping: {region: REGION_US, channel: channel, customer_id: customer_id, revenue: amount, buyers: customer_id}
"#;

async fn stepped_orders(disjoint: bool) -> SemanticManifest {
    let (eu, us) = match disjoint {
        true => ("{lit: eu}", "{lit: us}"),
        false => ("region", "region"),
    };
    let yaml = (STEPPED_ORDERS.replace("REGION_EU", eu)).replace("REGION_US", us);
    let columns = [
        ("region", "string"),
        ("channel", "string"),
        ("customer_id", "string"),
        ("amount", "double"),
    ];
    over_tables(&yaml, &ORDER_TABLES, &columns).await
}

#[tokio::test]
async fn one_union_form_serves_the_frame_and_a_metrics_subtree() {
    let manifest = stepped_orders(false).await;
    let request = select(&["channel", "revenue", "revenue_eu_bonus"]).with_from("split_orders");

    let (plan, _) = planned(&manifest, &request);

    let unions = unions(&plan);
    assert_eq!(
        unions.len(),
        2,
        "the frame's read and the subtree's: {plan:?}"
    );
    for union in unions {
        assert_eq!(branch_delivery(union), Delivery::PartialAggregates);
        for branch in &union.inputs {
            let (grouped, entries) = branch_step(branch);
            assert_eq!(grouped, ["channel", "region"]);
            let step_1 = [
                "__plan_agg1_revenue_eu_bonus",
                "__plan_agg2_revenue_eu_bonus",
            ];
            assert_eq!(entries[0], "revenue");
            assert_eq!(entries[1..], step_1);
        }
    }
    assert_eq!(
        output_names(&plan),
        ["channel", "revenue", "revenue_eu_bonus"]
    );
}

#[tokio::test]
async fn a_count_distinct_beside_a_metrics_subtree_makes_both_reads_take_rows() {
    let manifest = stepped_orders(false).await;
    let request = select(&["channel", "revenue_eu_bonus", "buyers"]).with_from("split_orders");

    let (plan, _) = planned(&manifest, &request);

    for union in unions(&plan) {
        assert_eq!(branch_delivery(union), Delivery::Rows);
        for branch in &union.inputs {
            assert_eq!(aggregate_steps(branch), 0, "rows in {branch:?}");
        }
    }
}

#[tokio::test]
async fn a_step_1_at_the_demand_reads_the_final_aggregates_of_disjoint_branches() {
    let manifest = stepped_orders(true).await;
    for (names, entry) in [
        (&["region", "buyers", "max_rev"][..], "__plan_agg1_max_rev"),
        (&["region", "max_buyers"], "__plan_agg1_max_buyers"),
        (&["region", "share_of_max"], "__plan_agg1_max_rev"),
    ] {
        let request = select(names).with_from("split_orders");
        let (plan, _) = planned(&manifest, &request);

        let (_, union) = union_of(&plan);
        let delivery = branch_delivery(union);
        assert_eq!(delivery, Delivery::FinalAggregates, "{names:?}");
        for branch in &union.inputs {
            let (grouped, entries) = branch_step(branch);
            assert_eq!(grouped, ["region"], "{names:?}");
            assert!(
                entries.iter().any(|held| held == entry),
                "{names:?}: {entries:?}"
            );
        }
        assert_eq!(
            steps_above_unions(&plan),
            1,
            "{names:?}: step 2 in {plan:?}"
        );
        assert_eq!(output_names(&plan), names);
    }
}

#[tokio::test]
async fn a_step_1_finer_than_disjoint_branches_reads_their_partials_by_its_key_columns() {
    let manifest = stepped_orders(true).await;
    let request = select(&["region", "revenue_web_bonus"]).with_from("split_orders");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::PartialAggregates);
    for branch in &union.inputs {
        let (grouped, entries) = branch_step(branch);
        assert_eq!(grouped, ["channel", "region"]);
        let step_1 = [
            "__plan_agg1_revenue_web_bonus",
            "__plan_agg2_revenue_web_bonus",
        ];
        assert_eq!(entries, step_1);
    }
    assert_eq!(steps_above_unions(&plan), 2, "both steps above: {plan:?}");
}

#[tokio::test]
async fn a_formula_push_down_reads_the_rows_of_disjoint_branches() {
    let orders = stepped_orders(true).await;
    let metrics = "  metrics:
    - name: cost_c1_adj
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [campaign_id, {lit: c1}]}
              then: {multiply: [cost, {lit: 2}]}
          else: cost
  unionsets:";
    let measures = "measures: [{ref: cost}, {ref: avg_cost}]";
    let yaml = (SPEND_ALL.replacen("  unionsets:", metrics, 1)).replacen(
        measures,
        &format!("{measures}\n      metrics: [{{ref: cost_c1_adj}}]"),
        1,
    );
    let spend = spend_all(&yaml).await;
    for (manifest, from, names) in [
        (&orders, "split_orders", ["region", "revenue_web_adj"]),
        (&spend, "spend_all", ["channel", "cost_c1_adj"]),
    ] {
        let (plan, _) = planned(manifest, &select(&names).with_from(from));

        let (_, union) = union_of(&plan);
        assert_eq!(branch_delivery(union), Delivery::Rows, "{names:?}");
        assert_eq!(output_names(&plan), names);
    }
}

#[tokio::test]
async fn a_semi_additive_step_1_reads_partials_by_its_key_column_and_its_by() {
    let metric = "  metrics:
    - name: x_bonus
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [account, {lit: X}]}
              then: {add: [balance, {lit: 1}]}
          else: balance
  unionsets:";
    let site = "measures: [{ref: balance}]\n      datasets";
    let yaml = (BALANCES.replacen("  unionsets:", metric, 1)).replacen(
        site,
        "measures: [{ref: balance}]\n      metrics: [{ref: x_bonus}]\n      datasets",
        1,
    );
    let columns = [("account", "string"), ("date", "date"), ("balance", "long")];
    let tables = ["warehouse.day_30", "warehouse.day_31"];
    let manifest = over_tables(&yaml, &tables, &columns).await;

    let (plan, _) = planned(&manifest, &select(&["x_bonus"]).with_from("balances"));

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::PartialAggregates);
    for branch in &union.inputs {
        assert_eq!(every_marker(branch), [], "no marker below the union");
        let (grouped, _) = branch_step(branch);
        assert_eq!(grouped, ["account", "date"]);
    }
    assert!(
        !every_marker(&plan.root).is_empty(),
        "a marker above the union"
    );
}

#[tokio::test]
async fn a_semi_additive_step_1_is_marked_in_each_disjoint_branch() {
    let metric = "  metrics:
    - {name: max_bal, data_type: bigint, agg: max, expr: balance}
  unionsets:";
    let site = "measures: [{ref: balance}]\n      datasets";
    let yaml = (BALANCES.replacen("  unionsets:", metric, 1))
        .replacen(
            site,
            "measures: [{ref: balance}]\n      metrics: [{ref: max_bal}]\n      datasets",
            1,
        )
        .replacen("account: account", "account: {lit: X}", 1)
        .replacen("account: account", "account: {lit: Y}", 1);
    let columns = [("account", "string"), ("date", "date"), ("balance", "long")];
    let tables = ["warehouse.day_30", "warehouse.day_31"];
    let manifest = over_tables(&yaml, &tables, &columns).await;

    for (names, marked) in [
        (&["account", "max_bal"][..], 1),
        (&["account", "balance", "max_bal"], 2),
    ] {
        let (plan, _) = planned(&manifest, &select(names).with_from("balances"));

        let below: usize = (unions(&plan).into_iter())
            .flat_map(|union| {
                assert_eq!(
                    branch_delivery(union),
                    Delivery::FinalAggregates,
                    "{names:?}"
                );
                union.inputs.iter()
            })
            .map(|branch| {
                assert_eq!(every_marker(branch).len(), marked, "{names:?}: {branch:?}");
                marked
            })
            .sum();
        let markers = every_marker(&plan.root).len();
        assert_eq!(markers, below, "{names:?}: no marker above the union");
        assert_eq!(output_names(&plan), names);
    }
}

const COMPOSED_SALES: &str = r#"
semantic_model:
  name: all_sales
  dimensions:
    - {name: order_id, data_type: string, type: categorical}
    - {name: kind, data_type: string, type: categorical}
    - {name: src, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  metrics:
    - name: gift_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [kind, {lit: gift}]}
              then: {add: [revenue, {lit: 100}]}
          else: revenue
    - {name: max_rev, data_type: double, agg: max, expr: revenue}
  unionsets:
    - name: all_sales
      mode: all
      dimensions: [{ref: order_id}, {ref: kind}, {ref: src}]
      measures: [{ref: revenue}]
      metrics:
        - ref: gift_bonus
          filter: {eq: [order_id, {lit: o1}]}
        - {ref: max_rev}
      datasets:
        - name: store_sales
          extras:
            storage: {tables: [warehouse.store_sales]}
            semantic_mapping: {order_id: order_id, kind: kind, revenue: amount, src: SRC_STORE}
      joinsets:
        - name: order_items
          datasets:
            - name: order_lines
              extras:
                storage: {tables: [warehouse.order_lines]}
                semantic_mapping: {order_id: order_id, revenue: amount, src: SRC_ONLINE}
            - name: orders
              extras:
                storage: {tables: [warehouse.orders]}
                semantic_mapping: {order_id: order_id, kind: kind}
          relationships:
            - {name: lines, from: order_lines, to: orders, on: [order_id], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

async fn composed_sales(disjoint: bool) -> SemanticManifest {
    let (store, online) = match disjoint {
        true => ("{lit: store}", "{lit: online}"),
        false => ("src", "src"),
    };
    let yaml = (COMPOSED_SALES.replace("SRC_STORE", store)).replace("SRC_ONLINE", online);
    let sales = [
        ("order_id", "string"),
        ("kind", "string"),
        ("amount", "double"),
        ("src", "string"),
    ];
    let tables = [
        ("warehouse.store_sales", &sales[..]),
        ("warehouse.order_lines", &sales),
        (
            "warehouse.orders",
            &[("order_id", "string"), ("kind", "string")],
        ),
    ];
    over_catalog(&yaml, &tables).await
}

#[tokio::test]
async fn a_composed_branch_groups_its_partials_by_a_step_1_column() {
    let manifest = composed_sales(false).await;
    let request = select(&["src", "gift_bonus"]).with_from("all_sales");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::PartialAggregates);
    for branch in &union.inputs {
        let (grouped, _) = branch_step(branch);
        assert_eq!(grouped, ["kind", "src"], "{branch:?}");
    }
    assert_eq!(output_names(&plan), ["src", "gift_bonus"]);
}

#[tokio::test]
async fn a_composed_branch_gives_step_1_over_disjoint_branches() {
    let manifest = composed_sales(true).await;
    let request = select(&["src", "max_rev"]).with_from("all_sales");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::FinalAggregates);
    let fields: Vec<_> = (union.common.output_schema.columns.iter())
        .map(|column| unqualified(column.name.as_str()))
        .collect();
    let entry = "__plan_agg1_max_rev";
    assert!(fields.iter().any(|name| name == entry), "{fields:?}");
    assert_eq!(steps_above_unions(&plan), 1, "step 2 alone: {plan:?}");
}

#[tokio::test]
async fn a_drill_across_branch_gives_step_1_over_disjoint_branches() {
    let metric = "  metrics:
    - {name: max_clicks, data_type: bigint, agg: max, expr: clicks}
  unionsets:";
    let site = "measures: [{ref: clicks}, {ref: budget}, {ref: orders}]";
    let yaml = (SPEND_BY_CHANNEL.replacen("  unionsets:", metric, 1)).replacen(
        site,
        &format!("{site}\n      metrics: [{{ref: max_clicks}}]"),
        1,
    );
    let manifest = spend_by_channel(&yaml).await;
    let rest = ["channel", "clicks", "budget", "max_clicks"];
    let request = at_granularity("date", Granularity::Month, &rest);

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    assert_eq!(branch_delivery(union), Delivery::FinalAggregates);
    assert_eq!(
        output_names(&plan),
        ["date", "channel", "clicks", "budget", "max_clicks"]
    );
}

async fn all_orders_with(metrics: &str, site: &str) -> SemanticManifest {
    all_orders(&with_metrics(ALL_ORDERS, metrics, site)).await
}

fn with_metrics(yaml: &str, metrics: &str, site: &str) -> String {
    let measures = "measures: [{ref: customer_rows}]";
    let metrics = format!("  metrics:\n{metrics}  unionsets:");
    (yaml.replacen("  unionsets:", &metrics, 1)).replacen(
        measures,
        &format!("{measures}\n      metrics: [{site}]"),
        1,
    )
}

const DOUBLED_ROWS: &str =
    "    - {name: doubled_rows, data_type: bigint, expr: {multiply: [customer_rows, {lit: 2}]}}\n";

fn branch_aggregate(branch: &PlanNode) -> &semstrait_ir::AggregateNode {
    aggregate_step(&projection(branch).input)
}

#[tokio::test]
async fn a_composed_branch_gives_a_measure_plainly_and_under_a_metrics_filter() {
    let site = "{ref: doubled_rows, filter: {eq: [customer_id, {lit: c1}]}}";
    let manifest = all_orders_with(DOUBLED_ROWS, site).await;

    let request = select(&["channel", "customer_rows", "doubled_rows"]);
    let (plan, _) = planned(&manifest, &request);

    let alias = &alias(&manifest, "all_orders");
    let [jw, _] = two_branches(&plan);
    let step = branch_aggregate(jw);
    let [(plain, plain_call), (node, node_call)] = step.aggregates.as_slice() else {
        panic!("two partial counts, got {:?}", step.aggregates);
    };
    assert_eq!(plain.as_str(), format!("{alias}.customer_rows"));
    assert!(plain_call.filter().is_none(), "{plain_call:?}");
    assert!(
        node_call.filter().is_some(),
        "under the filter: {node_call:?}"
    );
    let (qualifier, name) = node.as_str().split_once('.').expect("a qualified entry");
    let entry = format!("{alias}.__plan_agg1_doubled_rows");
    assert_eq!(
        written(projection(jw), &entry),
        Some(qualified(qualifier, name))
    );
    assert_eq!(
        output_names(&plan),
        ["channel", "customer_rows", "doubled_rows"]
    );
}

async fn two_conditions(kind: &str) -> SemanticManifest {
    let metrics = [
        "    - {name: rows_c1, data_type: bigint, expr: customer_rows}\n",
        "    - {name: rows_c2, data_type: bigint, expr: customer_rows}\n",
        "    - {name: doubled_c1, data_type: bigint, expr: {multiply: [rows_c1, {lit: 2}]}}\n",
        "    - {name: pair, data_type: bigint, expr: {add: [rows_c1, rows_c2]}}\n",
        DOUBLED_ROWS,
    ]
    .concat();
    let site = [
        "{ref: rows_c1, filter: {eq: [customer_id, {lit: c1}]}}",
        "{ref: rows_c2, filter: {eq: [customer_id, {lit: c2}]}}",
        "{ref: doubled_c1}",
        "{ref: pair}",
        "{ref: doubled_rows}",
    ]
    .join(", ");
    all_orders(&typed(&with_metrics(ALL_ORDERS, &metrics, &site), kind)).await
}

fn typed(yaml: &str, kind: &str) -> String {
    yaml.replace("data_type: bigint", &format!("data_type: {kind}"))
}

#[tokio::test]
async fn a_composed_branch_gives_one_field_for_each_measure_and_condition() {
    for kind in ["bigint", "double"] {
        let manifest = two_conditions(kind).await;

        for (names, calls) in [
            (&["channel", "doubled_c1"][..], 1),
            (&["channel", "customer_rows", "doubled_c1"], 2),
            (&["channel", "pair"], 2),
            (&["channel", "rows_c1", "rows_c2"], 2),
            (&["channel", "customer_rows", "doubled_rows"], 1),
        ] {
            let (plan, _) = planned(&manifest, &select(names));

            let [jw, _] = two_branches(&plan);
            let step = branch_aggregate(jw);
            let entries = step.aggregates.len();
            assert_eq!(entries, calls, "{kind}, {names:?}: {:?}", step.aggregates);
            assert_eq!(output_names(&plan), names);
        }
    }
}

const SHARE_OF_MAX_ROWS: &str = concat!(
    "    - {name: max_rows, data_type: bigint, agg: max, expr: customer_rows}\n",
    "    - {name: share, data_type: double, expr: {divide: [customer_rows, max_rows]}}\n",
);

#[tokio::test]
async fn a_composed_branch_gives_the_step_1_of_a_linked_metric_of_two_aggregate_steps() {
    let site = "{ref: max_rows}, {ref: share}";
    let inner = ALL_ORDERS.replacen(", optional: right, integrity: none", "", 1);
    for (yaml, kind) in [ALL_ORDERS, &inner]
        .into_iter()
        .flat_map(|yaml| ["bigint", "double"].map(|kind| (yaml, kind)))
    {
        let yaml = typed(&with_metrics(yaml, SHARE_OF_MAX_ROWS, site), kind);
        let manifest = all_orders(&yaml).await;
        for names in [
            &["channel", "share"][..],
            &["channel", "customer_rows", "share"],
            &["channel", "max_rows", "share"],
        ] {
            let (plan, _) = planned(&manifest, &select(names));

            let [jw, _] = two_branches(&plan);
            let given: Vec<_> = (jw.common().output_schema.columns.iter())
                .map(|column| unqualified(column.name.as_str()))
                .collect();
            let step_1 = given.iter().any(|name| name.ends_with("max_rows"));
            assert!(
                given.contains(&"__plan_agg1_share".to_owned()) && step_1,
                "{kind}: {given:?}"
            );
            assert_eq!(output_names(&plan), names);
        }
    }
}

fn reads_column(node: &PlanNode, tree: &PhysicalExpr, column: &str) -> bool {
    let fields: Vec<(String, PhysicalExpr)> = (node.pre_order())
        .filter_map(|node| match node {
            PlanNode::Project(project) => Some(fields(project)),
            _ => None,
        })
        .flatten()
        .map(|(name, tree)| (name.to_owned(), tree))
        .collect();
    let (mut pending, mut seen) = (vec![tree.clone()], Vec::new());
    while let Some(tree) = pending.pop() {
        for leaf in tree.pre_order() {
            let Expr::Column(read) = leaf else { continue };
            let name = field_name(read);
            if read.name() == column {
                return true;
            }
            if !seen.contains(&name) {
                let trees = fields.iter().filter(|(field, _)| field == &name);
                pending.extend(trees.map(|(_, tree)| tree.clone()));
                seen.push(name);
            }
        }
    }
    false
}

fn filtered_on(node: &PlanNode, name: &str, column: &str) -> Option<bool> {
    match node {
        PlanNode::Aggregate(step) => {
            let (_, call) = (step.aggregates.iter()).find(|(entry, _)| entry.as_str() == name)?;
            let filter = call.filter();
            Some(filter.is_some_and(|filter| reads_column(&step.input, filter, column)))
        }
        PlanNode::Project(project) => {
            let tree = written(project, name)?;
            let reads: Vec<_> = (tree.pre_order())
                .filter_map(|leaf| match leaf {
                    Expr::Column(read) => Some(field_name(read)),
                    _ => None,
                })
                .collect();
            (reads.iter()).find_map(|read| filtered_on(&project.input, read, column))
        }
        other => filtered_on(other.children().first().copied()?, name, column),
    }
}

fn step_1_filtered(branch: &PlanNode, metric: &str, column: &str) -> Vec<bool> {
    let mut held: Vec<_> = (branch.common().output_schema.columns.iter())
        .filter(|field| unqualified(field.name.as_str()).ends_with(metric))
        .map(|field| {
            let name = field.name.as_str();
            let filtered = filtered_on(branch, name, column);
            filtered.unwrap_or_else(|| panic!("{name}'s entry in {branch:?}"))
        })
        .collect();
    held.sort_unstable();
    held
}

#[tokio::test]
async fn a_linked_metric_under_its_readers_filter_is_a_step_1_apart_in_each_branch() {
    let on =
        |column: &str, value: &str| format!(", filter: {{eq: [{column}, {{lit: {value}}}]}}}}");
    let (c1, web) = (on("customer_id", "c1"), on("channel", "web"));
    let columns = [
        ("region", "string"),
        ("channel", "string"),
        ("customer_id", "string"),
        ("amount", "double"),
    ];
    let mut cases = Vec::new();
    for (filter, column, names) in [
        (
            &c1,
            "customer_id",
            &[
                &["region", "share_of_max"][..],
                &["channel", "share_of_max"],
            ][..],
        ),
        (
            &c1,
            "customer_id",
            &[&["region", "max_rev", "share_of_max"][..]],
        ),
        (
            &web,
            "channel",
            &[
                &["region", "share_of_max"][..],
                &["region", "max_rev", "share_of_max"],
            ],
        ),
    ] {
        let reader = format!("{{ref: share_of_max{filter}");
        let stepped = STEPPED_ORDERS.replacen("{ref: share_of_max}", &reader, 1);
        for (eu, us) in [("region", "region"), ("{lit: eu}", "{lit: us}")] {
            let yaml = (stepped.replace("REGION_EU", eu)).replace("REGION_US", us);
            let manifest = over_tables(&yaml, &ORDER_TABLES, &columns).await;
            for names in names {
                cases.push((manifest.clone(), *names, "max_rev", "split_orders", column));
            }
        }
    }
    let site = format!("{{ref: max_rows}}, {{ref: share{c1}");
    let inner = ALL_ORDERS.replacen(", optional: right, integrity: none", "", 1);
    for (yaml, kind) in [ALL_ORDERS, &inner]
        .into_iter()
        .flat_map(|yaml| ["bigint", "double"].map(|kind| (yaml, kind)))
    {
        let yaml = typed(&with_metrics(yaml, SHARE_OF_MAX_ROWS, &site), kind);
        let manifest = all_orders(&yaml).await;
        for names in [&["channel", "share"][..], &["channel", "max_rows", "share"]] {
            cases.push((
                manifest.clone(),
                names,
                "max_rows",
                "all_orders",
                "customer_id",
            ));
        }
    }

    for (manifest, names, linked, from, column) in cases {
        let (plan, _) = planned(&manifest, &select(names).with_from(from));

        let named = names.contains(&linked);
        let (_, union) = union_of(&plan);
        for branch in &union.inputs {
            let expected = if named { &[false, true][..] } else { &[true] };
            let filtered = step_1_filtered(branch, linked, column);
            assert_eq!(filtered, expected, "{names:?}: {branch:?}");
        }
        assert_eq!(output_names(&plan), names);
    }
}

#[tokio::test]
async fn two_readers_under_one_filter_share_the_subtree_of_the_metric_that_they_link() {
    let on =
        |column: &str, value: &str| format!(", filter: {{eq: [{column}, {{lit: {value}}}]}}}}");
    let share2 = "    - {name: share2, data_type: double, expr: {divide: [revenue, max_rev]}}\n  \
                  unionsets:";
    let columns = [
        ("region", "string"),
        ("channel", "string"),
        ("customer_id", "string"),
        ("amount", "double"),
    ];
    let names = ["region", "share_of_max", "share2"];
    for (filter, subtrees, fields) in [
        (on("customer_id", "c1"), 1, &[true][..]),
        (on("channel", "web"), 2, &[false, true]),
    ] {
        let c1 = on("customer_id", "c1");
        let site = format!("{{ref: share_of_max{c1}\n        - {{ref: share2{filter}");
        let metrics = STEPPED_ORDERS.replacen("  unionsets:", share2, 1);
        let stepped = metrics.replacen("{ref: share_of_max}", &site, 1);
        for (eu, us) in [("region", "region"), ("{lit: eu}", "{lit: us}")] {
            let yaml = (stepped.replace("REGION_EU", eu)).replace("REGION_US", us);
            let manifest = over_tables(&yaml, &ORDER_TABLES, &columns).await;

            let (plan, _) = planned(&manifest, &select(&names).with_from("split_orders"));

            assert_eq!(joins(&plan), subtrees, "{filter}, {eu}");
            let (_, union) = union_of(&plan);
            for branch in &union.inputs {
                let filtered = step_1_filtered(branch, "max_rev", "customer_id");
                assert_eq!(filtered, fields, "{filter}, {eu}: {branch:?}");
            }
            assert_eq!(output_names(&plan), names);
        }
    }
}

fn branch_types(plan: &SemanticPlan, name: &str) -> Vec<DataType> {
    let (_, union) = union_of(plan);
    (union.inputs.iter())
        .map(|branch| {
            let schema = &branch.common().output_schema;
            let column = (schema.columns.iter())
                .find(|column| column.name.as_str() == name)
                .unwrap_or_else(|| panic!("{name} in {schema:?}"));
            column.data_type.clone()
        })
        .collect()
}

#[tokio::test]
async fn a_formula_node_whose_operand_casts_below_the_call_is_read_at_the_operands_level() {
    let visits = "    - {name: visits, data_type: double, agg: sum}\n  metrics:\n    - {name: vr, data_type: double, expr: {divide: [visits, budget]}}\n";
    let measures = "measures: [{ref: clicks}, {ref: budget}, {ref: orders}]";
    let daily = "{date: day, channel: {lit: web}, clicks: clicks}";
    let yaml = [
        ("  unionsets:\n", &format!("{visits}  unionsets:\n")[..]),
        (
            measures,
            &format!("{measures}, {{ref: visits}}]\n      metrics: [{{ref: vr}}]")
                .replacen("]]", "]", 1)
                .replacen("{ref: orders}]", "{ref: orders}", 1),
        ),
        (
            daily,
            "{date: day, channel: {lit: web}, clicks: clicks, visits: clicks}",
        ),
    ]
    .iter()
    .fold(SPEND_BY_CHANNEL.to_owned(), |yaml, (from, to)| {
        yaml.replacen(from, to, 1)
    });
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "vr"]);

    let (plan, _) = planned(&manifest, &request);

    let types = branch_types(&plan, "__plan_agg1_vr");
    assert_eq!(types, [DataType::Double, DataType::Double]);
    assert_eq!(output_names(&plan), ["date", "channel", "vr"]);
}

#[tokio::test]
async fn a_null_filled_formula_node_takes_the_type_of_its_call_below_the_cast() {
    let metric = "  metrics:\n    - name: mix\n      data_type: double\n      expr:\n        divide: [{add: [clicks, orders]}, budget]\n  unionsets:";
    let measures = "measures: [{ref: clicks}, {ref: budget}, {ref: orders}]";
    let yaml = (SPEND_BY_CHANNEL.replacen("  unionsets:", metric, 1))
        .replacen(
            measures,
            &format!("{measures}\n      metrics: [{{ref: mix}}]"),
            1,
        )
        .replacen(
            "{name: orders, data_type: bigint, agg: count}",
            "{name: orders, data_type: double, agg: count}",
            1,
        );
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "mix"]);

    let (plan, _) = planned(&manifest, &request);

    let [_, spend_web] = two_branches(&plan);
    let node = (spend_web.pre_order()).find_map(|node| match node {
        PlanNode::Project(projection) => written(projection, "__plan_agg2_mix"),
        _ => None,
    });
    assert_eq!(node, Some(Expr::Literal(Literal::Null(DataType::BigInt))));
    assert_eq!(output_names(&plan), ["date", "channel", "mix"]);
}

async fn reaggregated_count() -> SemanticManifest {
    let key = "    - {name: customer_key, data_type: string, type: categorical}\n";
    let yaml = [
        ("  measures:\n", &format!("{key}  measures:\n")[..]),
        (
            "{ref: customer_id}]",
            "{ref: customer_id}, {ref: customer_key}]",
        ),
        (
            "customer_id: customer_id, customer_rows: order_id}",
            "customer_id: customer_id, customer_key: customer_id, customer_rows: order_id}",
        ),
        (
            "{channel: {lit: web}, customer_id: customer_id}",
            "{channel: {lit: web}, customer_id: customer_id, customer_rows: order_id}",
        ),
        (
            "{customer_id: customer_id, customer_rows: customer_id}",
            "{customer_id: customer_id, customer_key: customer_id}",
        ),
        (", optional: right, integrity: none", ""),
    ]
    .iter()
    .fold(ALL_ORDERS.to_owned(), |yaml, (from, to)| {
        yaml.replacen(from, to, 1)
    });
    all_orders(&with_metrics(&yaml, DOUBLED_ROWS, "{ref: doubled_rows}")).await
}

#[tokio::test]
async fn a_formula_node_of_a_reaggregated_count_is_its_finished_tree() {
    let manifest = reaggregated_count().await;

    let request = select(&["channel", "customer_key", "doubled_rows"]);
    let (plan, _) = planned(&manifest, &request);

    assert_eq!(
        output_names(&plan),
        ["channel", "customer_key", "doubled_rows"]
    );
}

fn with_clicks_a(yaml: &str) -> String {
    let metrics = "  metrics:\n    - {name: clicks_a, data_type: bigint, expr: clicks}\n    - {name: diff_a, data_type: bigint, expr: {subtract: [clicks, clicks_a]}}\n  unionsets:\n";
    let measures = "measures: [{ref: clicks}, {ref: budget}, {ref: orders}]";
    let site = "metrics: [{ref: clicks_a, filter: {eq: [campaign, {lit: A}]}}, {ref: diff_a}]";
    yaml.replacen("  unionsets:\n", metrics, 1).replacen(
        measures,
        &format!("{measures}\n      {site}"),
        1,
    )
}

async fn clicks_of_a() -> SemanticManifest {
    let yaml = with_clicks_a(&with_campaign(SPEND_BY_CHANNEL))
        .replace("{date: day,", "{campaign: campaign_id, date: day,");
    spend_by_channel(&yaml).await
}

#[tokio::test]
async fn a_drill_across_child_gives_a_measure_plainly_and_under_a_metrics_filter() {
    let manifest = clicks_of_a().await;
    let rest = ["channel", "clicks", "clicks_a", "budget"];
    let request = at_granularity("date", Granularity::Month, &rest);

    let (plan, _) = planned(&manifest, &request);

    let web_daily = &alias(&manifest, "spend_by_channel.spend_web.web_daily");
    let [_, spend_web] = two_branches(&plan);
    let entries = (spend_web.pre_order())
        .find_map(|node| match node {
            PlanNode::Aggregate(aggregate) => (aggregate.aggregates.iter())
                .any(|(entry, _)| entry.as_str() == format!("{web_daily}.clicks"))
                .then_some(&aggregate.aggregates),
            _ => None,
        })
        .unwrap_or_else(|| panic!("web_daily's aggregate step in {spend_web:?}"));
    let filtered: Vec<_> = (entries.iter())
        .map(|(entry, call)| (entry.as_str(), call.filter().is_some()))
        .collect();
    let node = format!("{web_daily}.__plan_node1_clicks");
    assert_eq!(
        filtered,
        [(&format!("{web_daily}.clicks")[..], false), (&node, true)]
    );
    let (drill_across, _) = drill_across(spend_web);
    let given = format!(
        "{}.__plan_node1_clicks",
        alias(&manifest, "spend_by_channel.spend_web")
    );
    assert_eq!(
        written(drill_across, &given),
        Some(qualified(web_daily, "__plan_node1_clicks"))
    );
    assert_eq!(
        output_names(&plan),
        ["date", "channel", "clicks", "clicks_a", "budget"]
    );
}

#[tokio::test]
async fn a_formula_reads_its_measure_plainly_and_under_a_linked_metrics_filter() {
    let manifest = clicks_of_a().await;
    let request = at_granularity("date", Granularity::Month, &["channel", "clicks", "diff_a"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(output_names(&plan), ["date", "channel", "clicks", "diff_a"]);
}

async fn u1_counts(filtered: bool, kind: &str) -> SemanticManifest {
    let filter = match filtered {
        true => ", filter: {eq: [channel, {lit: c}]}",
        false => "",
    };
    let yaml = [
        (
            "    - {name: avg_amount, data_type: double, agg: avg}\n",
            "    - {name: order_count, data_type: bigint, agg: count}\n  metrics:\n    - {name: c_orders, data_type: bigint, expr: order_count}\n",
        ),
        (
            "measures: [{ref: avg_amount}]",
            &format!("measures: [{{ref: order_count{filter}}}]\n      metrics: [{{ref: c_orders, filter: {{eq: [channel, {{lit: c}}]}}}}]")[..],
        ),
        ("avg_amount: amount", "order_count: amount"),
    ]
    .iter()
    .fold(U1.to_owned(), |yaml, (from, to)| yaml.replace(from, to));
    let yaml = typed(&yaml, kind);
    let objects = [
        "s3://lake/a/orders/part-0.parquet",
        "s3://lake/c/orders/part-0.parquet",
        "s3://lake/c/returns/part-0.parquet",
    ];
    compiled(&yaml, &Catalog::default(), &seeded(&objects).await).await
}

#[tokio::test]
async fn an_implicit_unionset_branch_gives_the_formula_node_that_its_stack_reads() {
    let manifest = u1_counts(false, "bigint").await;
    let u1 = &alias(&manifest, "u1");

    for names in [
        &["channel", "order_count", "c_orders"][..],
        &["order_count", "c_orders"],
    ] {
        let (plan, _) = planned(&manifest, &select(names));

        let (_, union) = union_of(&plan);
        let node = Some(qualified(u1, "__plan_node1_order_count"));
        let entry = format!("{u1}.c_orders");
        let reads = (union.inputs.iter())
            .filter(|branch| matches!(branch, PlanNode::Project(p) if written(p, &entry) == node));
        assert_eq!(reads.count(), 1, "u2 gives the node: {union:?}");
        assert_eq!(output_names(&plan), names);
    }
}

#[tokio::test]
async fn an_implicit_unionset_branch_gives_a_cast_count_in_the_entry_that_the_union_reads() {
    let manifest = u1_counts(false, "double").await;
    let u1 = &alias(&manifest, "u1");
    let names = ["channel", "order_count", "c_orders"];

    let (plan, _) = planned(&manifest, &select(&names));

    let (_, union) = union_of(&plan);
    let entry = format!("{u1}.__plan_agg1_order_count");
    let reads = (union.inputs.iter())
        .filter(|branch| matches!(branch, PlanNode::Project(p) if written(p, &entry).is_some()));
    assert_eq!(reads.count(), 1, "u2 gives the entry: {union:?}");
    assert_eq!(output_names(&plan), names);
}

#[tokio::test]
async fn a_stack_in_a_stack_applies_a_condition_in_its_branches() {
    let manifest = u1_counts(true, "bigint").await;

    let (plan, _) = planned(&manifest, &select(&["order_count"]));

    assert_eq!(output_names(&plan), ["order_count"]);
}

fn ads_yaml(key: &str) -> String {
    let child = |name: &str, mapped: &[&str]| {
        let mapping: Vec<_> = mapped
            .iter()
            .map(|name| format!("{name}: {name}"))
            .collect();
        let mapping = mapping.join(", ");
        format!(
            r#"
            - name: {name}
              extras:
                storage: {{tables: [warehouse.{name}]}}
                semantic_mapping: {{{mapping}}}
                temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}"#
        )
    };
    let children = [
        child("totals", &["date", "clicks", "spend"]),
        child("by_campaign", &["date", "campaign", "clicks", "spend"]),
        child("by_gender", &["date", "gender", "clicks"]),
        child("by_age", &["date", "age", "clicks"]),
    ]
    .concat();
    format!(
        r#"
semantic_model:
  name: ads
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
    - {{name: network, data_type: string, type: categorical}}
    - {{name: campaign, data_type: string, type: categorical}}
    - {{name: gender, data_type: string, type: categorical}}
    - {{name: age, data_type: string, type: categorical}}
  measures:
    - {{name: clicks, data_type: bigint, agg: sum}}
    - {{name: spend, data_type: bigint, agg: sum}}
  unionsets:
    - name: ads
      mode: all
      dimensions: [{{ref: date{key}}}, {{ref: network}}, {{ref: campaign}}, {{ref: gender}}, {{ref: age}}]
      measures: [{{ref: clicks}}, {{ref: spend}}]
      grainsets:
        - name: facebook
          datasets:{children}
      datasets:
        - name: google
          extras:
            storage: {{tables: [warehouse.google]}}
            semantic_mapping: {{network: {{lit: google}}, date: date, campaign: campaign, clicks: clicks}}
            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}
"#
    )
}

async fn ads(key: &str) -> SemanticManifest {
    ads_over(&ads_yaml(key)).await
}

async fn ads_over(yaml: &str) -> SemanticManifest {
    let date = ("date", "date");
    let (network, campaign, gender, age) = (
        ("network", "string"),
        ("campaign", "string"),
        ("gender", "string"),
        ("age", "string"),
    );
    let (clicks, spend) = (("clicks", "long"), ("spend", "long"));
    let tables: [(&str, &[(&str, &str)]); 5] = [
        ("warehouse.totals", &[date, clicks, spend]),
        ("warehouse.by_campaign", &[date, campaign, clicks, spend]),
        ("warehouse.by_gender", &[date, gender, clicks]),
        ("warehouse.by_age", &[date, age, clicks]),
        ("warehouse.google", &[date, network, campaign, clicks]),
    ];
    over_catalog(yaml, &tables).await
}

const FACEBOOK_NEAREST: [(&str, &[&str]); 4] = [
    ("ads.facebook.by_age", &["gender"]),
    ("ads.facebook.by_gender", &["age"]),
    ("ads.facebook.totals", &["gender", "age"]),
    ("ads.facebook.by_campaign", &["gender", "age"]),
];

fn assert_pruned_with_an_advisory(
    (plan, advisories): (SemanticPlan, Diagnostics<Advisory>),
    read: &[&str],
    (path, branch): (&str, &str),
    lacks: &[&str],
) {
    assert_eq!(datasets(&plan), read);
    assert!(pruned(&plan).contains(&branch), "{:?}", pruned(&plan));
    let lacks = lacks.iter().map(ToString::to_string).collect();
    assert_eq!(
        not_servable_branches(advisories),
        [(path.to_string(), branch.to_string(), lacks)]
    );
}

fn assert_facebook_pruned_with_an_advisory(manifest: &SemanticManifest) {
    let request = select(&["gender", "age", "clicks"]);

    assert_pruned_with_an_advisory(
        planned(manifest, &request),
        &["ads.google"],
        ("ads", "ads.facebook"),
        &["gender", "age"],
    );
}

#[tokio::test]
async fn a_grainset_branch_whose_children_cannot_serve_the_demand_is_pruned_with_an_advisory() {
    assert_facebook_pruned_with_an_advisory(&ads("").await);
}

#[tokio::test]
async fn a_branch_that_the_fold_prunes_raises_no_advisory_even_where_it_cannot_serve() {
    let manifest = ads("").await;
    let google = dim("network").expect("a name").eq(lit("google"));
    let request = select(&["gender", "age", "clicks"]).with_predicate(google);

    let (plan, advisories) = planned(&manifest, &request);

    assert_eq!(datasets(&plan), ["ads.google"]);
    assert!(pruned(&plan).contains(&"ads.facebook"));
    assert_eq!(not_servable_branches(advisories), []);
}

#[tokio::test]
async fn a_grainset_branch_that_a_keyed_join_of_its_children_cannot_serve_is_pruned_with_an_advisory(
) {
    assert_facebook_pruned_with_an_advisory(&ads(", is_key: true, keysets: [natural]").await);
}

#[tokio::test]
async fn a_grainset_branch_whose_names_no_one_child_serves_is_pruned_where_another_branch_answers()
{
    let manifest = spend_channels().await;
    let request = at_granularity("date", Granularity::Month, &["channel", "clicks", "budget"]);

    assert_pruned_with_an_advisory(
        planned(&manifest, &request),
        &["all_spend.spend_c"],
        ("all_spend", "all_spend.spend_g"),
        &["clicks", "budget"],
    );
}

#[tokio::test]
async fn a_request_whose_branches_are_all_pruned_is_refused_where_the_serve_test_prunes_one() {
    let manifest = ads("").await;
    let unmapped = dim("network").expect("a name").is_null();
    let request = select(&["gender", "age", "clicks"]).with_predicate(unmapped);

    grain_not_servable(
        &manifest,
        &request,
        &["ads", "ads.facebook"],
        (None, Some(Granularity::Day)),
        &FACEBOOK_NEAREST,
    );
}

#[tokio::test]
async fn a_grainset_branch_that_a_keyed_drill_across_serves_stays_refused_under_partial_aggregates()
{
    let keyed = SPEND_CHANNELS.replace(
        "dimensions: [{ref: date}, {ref: campaign}, {ref: channel}]",
        "dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}, {ref: channel}]",
    );
    let manifest = spend_channels_over(&keyed).await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "clicks", "budget"],
    );

    grain_not_servable(
        &manifest,
        &request,
        &["all_spend", "all_spend.spend_g"],
        (Some(Granularity::Month), Some(Granularity::Month)),
        &[
            ("all_spend.spend_g.g_daily", &["budget"]),
            ("all_spend.spend_g.g_monthly", &["clicks"]),
        ],
    );
}

const CLICK_RATE: &str = "  metrics:\n    - name: click_rate\n      data_type: double\n      expr:\n        divide: [clicks, budget]\n  unionsets:";

fn with_click_rate(yaml: &str) -> String {
    let measures = "measures: [{ref: clicks}, {ref: budget}, {ref: orders}]";
    yaml.replacen("  unionsets:", CLICK_RATE, 1).replacen(
        measures,
        &format!("{measures}\n      metrics: [{{ref: click_rate}}]"),
        1,
    )
}

async fn click_rate_with_totals() -> SemanticManifest {
    let totals_child = "            - name: web_totals\n              extras:\n                storage: {tables: [warehouse.web_totals]}\n                semantic_mapping: {date: month_start, channel: {lit: web}, clicks: clicks, budget: budget}\n                temporal: {timeseries: {occurred_at: date}, granularity: month}\n";
    let yaml = [
        (
            "{date: day, channel: {lit: web},",
            "{date: day, campaign: campaign_id, channel: {lit: web},",
        ),
        (
            "{date: month_start, channel: {lit: web}, budget",
            "{date: month_start, campaign: campaign_id, channel: {lit: web}, budget",
        ),
        (
            "      datasets:\n        - name: spend_store",
            &format!("{totals_child}      datasets:\n        - name: spend_store"),
        ),
    ]
    .iter()
    .fold(
        with_campaign(&with_click_rate(SPEND_BY_CHANNEL)),
        |yaml, (from, to)| yaml.replace(from, to),
    );
    let totals = [
        ("month_start", "date"),
        ("clicks", "long"),
        ("budget", "long"),
    ];
    spend_by_channel_over(&yaml, &[("warehouse.web_totals", &totals)]).await
}

const WEB_DATASETS: [&str; 3] = [
    "spend_by_channel.spend_store",
    "spend_by_channel.spend_web.web_daily",
    "spend_by_channel.spend_web.web_monthly",
];

fn assert_web_drilled_across((plan, advisories): (SemanticPlan, Diagnostics<Advisory>)) {
    let mut read = datasets(&plan);
    read.sort_unstable();
    assert_eq!(read, WEB_DATASETS);
    assert_eq!(not_servable_branches(advisories), []);
}

#[tokio::test]
async fn a_unionset_metric_is_asked_of_a_branch_as_its_measures_when_the_prune_tests_it() {
    let manifest = click_rate_with_totals().await;
    let rest = ["channel", "campaign", "click_rate"];
    let request = at_granularity("date", Granularity::Month, &rest);

    assert_web_drilled_across(planned(&manifest, &request));
}

#[tokio::test]
async fn a_predicate_placed_after_aggregation_is_not_asked_of_a_branch_when_the_prune_tests_it() {
    let manifest = click_rate_with_totals().await;
    let rest = ["channel", "campaign", "clicks", "budget"];
    let request = at_granularity("date", Granularity::Month, &rest)
        .with_predicate(metric("click_rate").expect("a name").gt(lit(0)));

    assert_web_drilled_across(planned(&manifest, &request));
}

#[tokio::test]
async fn an_unkeyed_grainset_branch_is_pruned_for_the_measures_of_a_unionset_metric() {
    let yaml = with_click_rate(&SPEND_BY_CHANNEL.replace(", is_key: true", ""));
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "click_rate"]);

    assert_pruned_with_an_advisory(
        planned(&manifest, &request),
        &["spend_by_channel.spend_store"],
        ("spend_by_channel", "spend_by_channel.spend_web"),
        &["clicks", "budget"],
    );
}

fn clicks_a_where_web_daily_lacks_campaign(keyed: bool, mode: &str) -> String {
    let yaml = with_clicks_a(&with_campaign(SPEND_BY_CHANNEL))
        .replacen(
            "{date: month_start,",
            "{campaign: campaign_id, date: month_start,",
            1,
        )
        .replacen(
            "{date: day, channel: {lit: store},",
            "{date: day, campaign: campaign_id, channel: {lit: store},",
            1,
        )
        .replacen("mode: all", &format!("mode: {mode}"), 1);
    if keyed {
        yaml
    } else {
        yaml.replace(", is_key: true", "")
    }
}

#[tokio::test]
async fn a_condition_that_no_child_of_a_grainset_branch_serves_with_its_measure_prunes_the_branch()
{
    for (keyed, mode, rest) in [
        (false, "all", &["channel", "clicks_a"][..]),
        (true, "all", &["channel", "clicks_a"][..]),
        (false, "unique", &["clicks_a"][..]),
    ] {
        let yaml = clicks_a_where_web_daily_lacks_campaign(keyed, mode);
        let manifest = spend_by_channel(&yaml).await;
        let request = at_granularity("date", Granularity::Month, rest);

        assert_pruned_with_an_advisory(
            planned(&manifest, &request),
            &["spend_by_channel.spend_store"],
            ("spend_by_channel", "spend_by_channel.spend_web"),
            &["clicks", "campaign"],
        );
    }
}

#[tokio::test]
async fn a_measure_that_a_predicate_after_aggregation_reads_is_asked_of_a_branch_when_the_prune_tests_it(
) {
    for mode in ["all", "unique"] {
        let yaml = SPEND_BY_CHANNEL.replace(", is_key: true", "").replacen(
            "mode: all",
            &format!("mode: {mode}"),
            1,
        );
        let manifest = spend_by_channel(&yaml).await;
        let request = at_granularity("date", Granularity::Month, &["channel", "budget"])
            .with_predicate(measure("clicks").expect("a name").gt(lit(0)));

        assert_pruned_with_an_advisory(
            planned(&manifest, &request),
            &["spend_by_channel.spend_store"],
            ("spend_by_channel", "spend_by_channel.spend_web"),
            &["budget", "clicks"],
        );
    }
}

#[tokio::test]
async fn a_measure_that_only_a_predicate_reads_is_drilled_across_with_the_selected_measures() {
    let yaml = SPEND_BY_CHANNEL.replacen(
        "semantic_mapping: {date: day, channel: {lit: web}, clicks: clicks}",
        "semantic_mapping: {date: day, channel: {lit: web}, clicks: clicks, orders: clicks}",
        1,
    );
    let manifest = spend_by_channel(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "clicks", "budget"])
        .with_predicate(measure("orders").expect("a name").gt(lit(0)));

    assert_web_drilled_across(planned(&manifest, &request));
}

#[tokio::test]
async fn a_measure_that_only_a_predicate_reads_is_drilled_across_where_a_child_maps_a_filtered_measure(
) {
    let manifest = spend_by_channel(&campaign_filtered()).await;
    let request = at_granularity("date", Granularity::Month, &["channel", "budget"])
        .with_predicate(measure("clicks").expect("a name").gt(lit(0)));

    assert_web_drilled_across(planned(&manifest, &request));
}

#[tokio::test]
async fn a_joinset_lowered_measure_in_an_inner_unionset_with_every_branch_excluded_is_read_beside_a_kept_branch(
) {
    let yaml = outer_sales_joined()
        .replace(
            "    - {name: customer_id, data_type: string, type: categorical}\n",
            "    - {name: customer_id, data_type: string, type: categorical}\n    - {name: segment, data_type: string, type: categorical}\n",
        )
        .replace(
            "    - {name: buyers, data_type: bigint, agg: count_distinct}\n",
            "    - {name: buyers, data_type: bigint, agg: count_distinct}\n    - {name: vip_buyers, data_type: bigint, agg: count_distinct, expr: {case: {whens: [{when: {eq: [segment, {lit: vip}]}, then: customer_id}]}}}\n",
        )
        .replace(
            "measures: [{ref: buyers}]",
            "measures: [{ref: buyers}, {ref: vip_buyers}]",
        )
        .replace(
            "semantic_mapping: {region: region, customer_id: customer_id, buyers: customer_id}",
            "semantic_mapping: {region: region, customer_id: customer_id, segment: segment, buyers: customer_id}",
        )
        .replace(
            "{store_id: store_id}",
            "{store_id: store_id, segment: segment}",
        );
    let yaml = yaml.replace(
        "dimensions: [{ref: region}, {ref: customer_id}, {ref: store_id}]",
        "dimensions: [{ref: region}, {ref: customer_id}, {ref: store_id}, {ref: segment}]",
    );
    let online = &[
        ("region", "string"),
        ("customer_id", "string"),
        ("segment", "string"),
    ][..];
    let sales = &[
        ("region", "string"),
        ("customer_id", "string"),
        ("store_id", "string"),
    ][..];
    let lookup = &[("store_id", "string"), ("segment", "string")][..];
    let manifest = over_catalog(
        &yaml,
        &[
            ("warehouse.online", online),
            ("warehouse.store_sales", sales),
            ("warehouse.web_sales", sales),
            ("warehouse.customers", &[("customer_id", "string")][..]),
            ("warehouse.stores", lookup),
            ("warehouse.sites", lookup),
        ],
    )
    .await;
    let apac = dim("region").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &select(&["vip_buyers"]).with_predicate(apac));

    assert_read_beside_kept(
        &plan,
        &["outer.jw.customers", "outer.online"],
        "vip_buyers",
        (DataType::String, true, true),
        &"vip_buyers",
    );
}

async fn mixed_child_types(members: bool) -> SemanticManifest {
    let daily = &[
        ("day", "date"),
        ("region", "string"),
        ("customer_id", "string"),
    ][..];
    let monthly = &[
        ("month_start", "date"),
        ("budget", "long"),
        ("customer_key", "long"),
    ][..];
    let mut yaml = outer_sales_grained()
        .replace(
            "region: {lit: {string: emea}}, budget: budget}",
            "region: {lit: {string: emea}}, budget: budget, buyers: customer_key}",
        )
        .replace(
            "region: {lit: {string: amer}}, budget: budget}",
            "region: {lit: {string: amer}}, budget: budget, buyers: customer_key}",
        );
    if members {
        yaml = yaml
            .replace(
                "    - {name: budget, data_type: bigint, agg: sum}\n",
                "    - {name: budget, data_type: bigint, agg: sum}\n    - {name: members, data_type: bigint, agg: count_distinct}\n",
            )
            .replace(
                "measures: [{ref: buyers}, {ref: budget}]",
                "measures: [{ref: buyers}, {ref: budget}, {ref: members}]",
            )
            .replace(
                "buyers: customer_id, budget: budget}",
                "buyers: customer_id, members: customer_id, budget: budget}",
            )
            .replace(
                "semantic_mapping: {customer_id: customer_id}\n                temporal",
                "semantic_mapping: {customer_id: customer_id, members: customer_id}\n                temporal",
            );
    }
    assert!(yaml.contains("buyers: customer_key"));
    let tables = [
        (
            "warehouse.online",
            &[
                ("day", "date"),
                ("region", "string"),
                ("customer_id", "string"),
                ("budget", "long"),
            ][..],
        ),
        ("warehouse.store_daily", daily),
        ("warehouse.store_monthly", monthly),
        ("warehouse.web_daily", daily),
        ("warehouse.web_monthly", monthly),
        ("warehouse.customers", &[("customer_id", "string")][..]),
    ];
    over_catalog(&yaml, &tables).await
}

fn nulls_field(node: &PlanNode, name: &str) -> bool {
    (node.pre_order()).any(|node| match node {
        PlanNode::Project(project) => fields(project).into_iter().any(|(field, tree)| {
            field.ends_with(name) && tree == Expr::Literal(Literal::Null(DataType::String))
        }),
        _ => false,
    })
}

#[tokio::test]
async fn a_grainset_child_typed_differently_from_its_sibling_still_gives_the_empty_read_the_kept_type(
) {
    let manifest = mixed_child_types(false).await;
    let apac = dim("region").expect("a name").eq(lit("apac"));
    let request = at_granularity("date", Granularity::Month, &["buyers"]);

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    assert_read_beside_kept(
        &plan,
        &["outer.jw.customers", "outer.online"],
        "buyers",
        (DataType::String, true, true),
        &"mixed child types",
    );
}

#[tokio::test]
async fn a_branch_typed_only_by_an_empty_read_with_no_branch_is_filled_with_the_kept_type() {
    let manifest = mixed_child_types(false).await;
    let apac = dim("region").expect("a name").eq(lit("apac"));
    let request = at_granularity("date", Granularity::Day, &["buyers"]);

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let mut read = datasets(&plan);
    read.sort_unstable();
    assert_eq!(read, ["outer.jw.customers", "outer.online"]);
    let [empty] = empty_reads(&plan)[..] else {
        panic!("one empty read in {plan:?}");
    };
    assert_eq!(
        typed_field(&empty.common.output_schema, "buyers"),
        (DataType::BigInt, true)
    );
    let union = first_union(&plan.root);
    assert_eq!(union.inputs.len(), 2);
    for input in &union.inputs {
        assert_eq!(
            typed_field(&input.common().output_schema, "buyers").0,
            DataType::String
        );
    }
    let filled = (union.inputs.iter()).filter(|input| nulls_field(input, "buyers"));
    assert_eq!(filled.count(), 1, "{plan:?}");
}

#[tokio::test]
async fn an_operand_that_types_the_name_itself_is_not_filled_with_null() {
    let manifest = mixed_child_types(true).await;
    let apac = dim("region").expect("a name").eq(lit("apac"));
    let request = at_granularity("date", Granularity::Day, &["buyers", "members"]);

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    let union = first_union(&plan.root);
    assert_eq!(union.inputs.len(), 2);
    let filled: Vec<_> = (union.inputs.iter())
        .filter(|input| nulls_field(input, "buyers"))
        .collect();
    assert_eq!(filled.len(), 1, "{plan:?}");
    assert!(!nulls_field(filled[0], "members"), "{plan:?}");
    assert_eq!(
        typed_field(&filled[0].common().output_schema, "members").0,
        DataType::String
    );
}

const DRAWN_SPEND: &str = r#"
semantic_model:
  name: drawn
  dimensions:
    - {name: channel, data_type: string, type: categorical}
    - {name: draw, data_type: double, type: categorical, expr: {random: []}}
    - {name: draw_up, data_type: double, type: categorical, expr: {add: [draw, {lit: 1.0}]}}
  measures:
    - {name: spend, data_type: double, agg: sum}
  unionsets:
    - name: drawn_spend
      mode: all
      dimensions: [{ref: channel}, {ref: draw}, {ref: draw_up}]
      measures: [{ref: spend}]
      datasets:
        - name: spend_store
          extras:
            storage: {tables: [warehouse.spend_store]}
            semantic_mapping: {channel: {lit: {string: store}}, spend: amount}
        - name: spend_web
          extras:
            storage: {tables: [warehouse.spend_web]}
            semantic_mapping: {channel: {lit: {string: web}}, spend: amount}
"#;

#[tokio::test]
async fn an_excluded_branch_raises_no_advisory_and_a_kept_branch_keeps_its_own() {
    let manifest = over_catalog(
        DRAWN_SPEND,
        &[
            ("warehouse.spend_store", &[("amount", "double")][..]),
            ("warehouse.spend_web", &[("amount", "double")][..]),
        ],
    )
    .await;
    let request = select(&["draw", "draw_up", "spend"]);
    let on = |channel: &str| {
        request
            .clone()
            .with_predicate(dim("channel").expect("a name").eq(lit(channel)))
    };

    let (plan, none) = planned(&manifest, &on("apac"));
    assert_eq!(empty_reads(&plan).len(), 1, "{plan:?}");
    assert_eq!(datasets(&plan), [] as [&str; 0]);
    let (_, kept) = planned(&manifest, &on("web"));

    assert_eq!(advisory_kinds(none), []);
    let kept = advisory_kinds(kept);
    let [Advisory::VolatileCopied { path, semantics }] = &kept[..] else {
        panic!("one advisory, got {kept:?}");
    };
    assert_eq!(semantics.as_str(), "draw");
    assert!(
        format!("{path:?}").contains("drawn_spend.spend_web"),
        "{path:?}"
    );
}

const SPEND_UNION_GRAINS: &str = r#"
semantic_model:
  name: spend_union_grains
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: users, data_type: bigint, agg: count_distinct}
  unionsets:
    - name: top
      mode: all
      dimensions: [{ref: date, is_key: true}, {ref: channel, is_key: true}]
      measures: [{ref: clicks}, {ref: users}]
      grainsets:
        - name: spend
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: monthly
              extras:
                storage: {tables: [warehouse.monthly]}
                semantic_mapping: {date: month_start, channel: {lit: {string: store}}, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
          unionsets:
            - name: daily
              mode: all
              extras:
                temporal: {timeseries: {occurred_at: date}, granularity: month}
              grainsets:
                - name: web_g
                  extras:
                    temporal: {timeseries: {occurred_at: date}, granularity: month}
                  datasets:
                    - name: web_u
                      extras:
                        storage: {tables: [warehouse.web_u]}
                        semantic_mapping: {date: month_start, channel: {lit: {string: web}}, users: user_id}
                        temporal: {timeseries: {occurred_at: date}, granularity: month}
                    - name: web_c
                      extras:
                        storage: {tables: [warehouse.web_c]}
                        semantic_mapping: {date: month_start, channel: {lit: {string: web}}, clicks: clicks}
                        temporal: {timeseries: {occurred_at: date}, granularity: month}
                - name: app_g
                  extras:
                    temporal: {timeseries: {occurred_at: date}, granularity: month}
                  datasets:
                    - name: app_u
                      extras:
                        storage: {tables: [warehouse.app_u]}
                        semantic_mapping: {date: month_start, channel: {lit: {string: app}}, users: user_id}
                        temporal: {timeseries: {occurred_at: date}, granularity: month}
                    - name: app_c
                      extras:
                        storage: {tables: [warehouse.app_c]}
                        semantic_mapping: {date: month_start, channel: {lit: {string: app}}, clicks: clicks}
                        temporal: {timeseries: {occurred_at: date}, granularity: month}
      datasets:
        - name: store
          extras:
            storage: {tables: [warehouse.store]}
            semantic_mapping: {date: day, channel: channel, clicks: clicks, users: user_id}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

#[tokio::test]
async fn a_grainset_child_unionset_of_unservable_grainsets_with_every_branch_excluded_is_read_beside_a_kept_branch(
) {
    let monthly = &[
        ("month_start", "date"),
        ("clicks", "long"),
        ("user_id", "string"),
    ][..];
    let tables = [
        ("warehouse.monthly", monthly),
        ("warehouse.web_u", monthly),
        ("warehouse.web_c", monthly),
        ("warehouse.app_u", monthly),
        ("warehouse.app_c", monthly),
        (
            "warehouse.store",
            &[
                ("day", "date"),
                ("channel", "string"),
                ("clicks", "long"),
                ("user_id", "string"),
            ][..],
        ),
    ];
    let manifest = over_catalog(SPEND_UNION_GRAINS, &tables).await;
    let request = at_granularity("date", Granularity::Month, &["users", "clicks"]);
    let apac = dim("channel").expect("a name").eq(lit("apac"));

    let (plan, _) = planned(&manifest, &request.with_predicate(apac));

    assert_eq!(datasets(&plan), ["top.store"]);
    let [empty] = empty_reads(&plan)[..] else {
        panic!("one empty read in {plan:?}");
    };
    assert_eq!(
        pruned_names(empty),
        ["top.spend.daily.app_g", "top.spend.daily.web_g"]
    );
    let union = first_union(&plan.root);
    assert_eq!(union.inputs.len(), 2);
    for input in &union.inputs {
        assert_eq!(
            typed_field(&input.common().output_schema, "users").0,
            DataType::String
        );
    }
    let filled = (union.inputs.iter()).filter(|input| nulls_field(input, "users"));
    assert_eq!(filled.count(), 1, "{plan:?}");
}
