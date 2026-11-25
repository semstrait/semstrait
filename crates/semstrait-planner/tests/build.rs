mod common;

use common::golden::{self, Golden};
use common::{
    aggregate_filters, aggregate_step, aggregate_steps, at_granularity, below_aggregate, bigint,
    call, column, columns_of, computed, cross, fields, filter, filtered, harness, integer, join,
    names, orders_catalog, output_names, parts, planned, planned_in, projection, refusal, scan_of,
    select, skeleton, stored, tables_read, text, two_steps, with_datasets, with_semantics, Catalog,
    ORDERS, TWINS,
};
use semstrait_catalog::FileFormat;
use semstrait_common::DataKindName;
use semstrait_common::Granularity;
use semstrait_common::{Dialect, Tree};
use semstrait_ir::{
    AggregateNode, GranularityGroup, JoinType, Name, Parameter, PhysicalExpr, PlanNode,
    ProjectNode, Reader, SemAnnotation, SemanticPlan, SortNode, SourceRole, TableScan,
};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit, measure, metric};
use semstrait_model::UnresolvedSemanticExpr;
use semstrait_planner::{Direction, OrderKey, PlanError, RequestContext, Timezone, Weekday};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, CastFailure, Expr, ExprOperationExt, Literal,
    NullOrdering, SemanticLeaf, SortDir, SortKey, UntypedLiteral,
};

type Aggregate = AggregateCall<Parameter, AggregateAllowed>;

fn untyped(value: &str) -> UnresolvedSemanticExpr {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::String(
        value.into(),
    ))))
}

fn distinct_rows(plan: &SemanticPlan) -> Vec<(&str, PhysicalExpr)> {
    let aggregate = aggregate_step(&plan.root);
    assert!(
        aggregate.aggregates.is_empty(),
        "{:?}",
        aggregate.aggregates
    );
    let rows = fields(projection(&aggregate.input));
    let groupings: Vec<PhysicalExpr> = rows.iter().map(|(name, _)| column(name)).collect();
    assert_eq!(aggregate.groupings, groupings);
    rows
}

fn sort(node: &PlanNode) -> &SortNode {
    match node {
        PlanNode::Sort(sort) => sort,
        other => panic!("a sort, got {other:?}"),
    }
}

fn dataset<Part: AsRef<str>>(name: &str, dimensions: &[[Part; 3]]) -> String {
    let entries: String = (dimensions.iter())
        .map(|entry| {
            let [dimension, data_type, expr] = entry.each_ref().map(|part| part.as_ref());
            format!(
                r#"
        - name: {dimension}
          data_type: {data_type}
          type: categorical
          expr: {expr}"#
            )
        })
        .collect();
    format!(
        r#"
    - name: {name}
      extras:
        storage:
          tables: [warehouse.{name}]
      dimensions:{entries}
"#
    )
}

fn orders_table() -> Reader {
    let names = [String::new(), "warehouse".into(), "orders".into()];
    Reader::TableScan(TableScan::named_table(names, Some(FileFormat::Iceberg)))
}

#[tokio::test]
async fn a_dimension_and_a_measure_plan_as_a_scan_a_row_level_projection_and_one_aggregate_step() {
    let (plan, advisories) = planned(&skeleton().await, &select(&["region", "revenue"]));

    let aggregate = aggregate_step(&plan.root);
    let row_level = projection(&aggregate.input);
    let PlanNode::Scan(scan) = &*row_level.input else {
        panic!("the scan, got {:?}", row_level.input);
    };
    assert_eq!(names(&scan.schema), ["region", "amount"]);
    let read = scan.schema.columns.len();
    let emitted = row_level
        .common
        .emit_form
        .as_ref()
        .map(|emit| &emit.output_mapping);
    assert_eq!(
        emitted,
        Some(&vec![read, read + 1]),
        "no physical column passes"
    );
    let [region, argument] = names(&row_level.common.output_schema)[..] else {
        panic!("two fields, got {:?}", row_level.common.output_schema);
    };
    assert_eq!(region, "region");
    assert!(
        argument.starts_with("__plan_") && argument.ends_with("_revenue"),
        "{argument}"
    );
    assert_eq!(aggregate.groupings, [column("region")]);
    let [(revenue, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(revenue.as_str(), "revenue");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(sum.args(), [column(argument)]);
    assert_eq!(output_names(&plan), ["region", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_request_that_selects_only_measures_has_an_aggregate_step_with_no_grouping() {
    let (plan, _) = planned(&skeleton().await, &select(&["revenue"]));

    let aggregate = aggregate_step(&plan.root);
    assert!(aggregate.groupings.is_empty());
    let entries: Vec<&str> = (aggregate.aggregates.iter())
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(entries, ["revenue"]);
    assert_eq!(output_names(&plan), ["revenue"]);
}

#[tokio::test]
async fn a_request_that_selects_only_dimensions_has_an_aggregate_step_with_no_entry() {
    let (plan, _) = planned(&skeleton().await, &select(&["region"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("region")]);
    assert!(aggregate.aggregates.is_empty());
    let row_level = projection(&aggregate.input);
    assert_eq!(names(&row_level.common.output_schema), ["region"]);
    assert_eq!(output_names(&plan), ["region"]);
}

#[tokio::test]
async fn a_dimension_predicate_is_a_filter_node_above_the_row_level_projection() {
    let web = dim("channel").expect("a name").eq(lit("web"));
    let request = select(&["region", "revenue"]).with_predicate(web);

    let (plan, _) = planned(&skeleton().await, &request);

    let (filter, row_level) = filtered(&plan);
    assert_eq!(filter.filter, column("channel").eq(text("web")));
    let fields = names(&row_level.common.output_schema);
    assert_eq!(
        (&fields[..2], fields.len()),
        (&["channel", "region"][..], 3)
    );
    let scan = scan_of(&plan);
    assert_eq!(names(&scan.schema), ["region", "amount", "channel"]);
    assert_eq!(scan.filter, None);
    assert_eq!(aggregate_step(&plan.root).groupings, [column("region")]);
    assert_eq!(output_names(&plan), ["region", "revenue"]);
}

#[tokio::test]
async fn a_dimension_predicate_sits_below_the_distinct_rows_aggregate_step() {
    let web = dim("channel").expect("a name").eq(lit("web"));

    let (plan, _) = planned(&skeleton().await, &select(&["region"]).with_predicate(web));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("region")]);
    assert!(aggregate.aggregates.is_empty());
    let (filter, row_level) = filtered(&plan);
    assert_eq!(filter.filter, column("channel").eq(text("web")));
    assert_eq!(
        names(&row_level.common.output_schema),
        ["channel", "region"]
    );
    assert_eq!(output_names(&plan), ["region"]);
}

const PAIR_SEMANTICS: &str = r#"
  dimensions:
    - name: a
      data_type: string
      type: categorical
    - name: b
      data_type: string
      type: categorical
  measures:
    - name: n
      data_type: bigint
      agg: count
"#;

const PAIRS: &str = r#"
    - name: pairs
      extras:
        storage:
          tables: [warehouse.pairs]
        semantic_mapping:
          a: a
          b: b
          n: {lit: 1}
      dimensions:
        - ref: a
        - ref: b
      measures:
        - ref: n
"#;

#[tokio::test]
async fn a_dimension_predicate_keeps_its_comparison_in_the_filter_node() {
    let catalog = Catalog::default().table(
        "warehouse.pairs",
        FileFormat::Iceberg,
        &[("a", "string"), ("b", "string")],
    );
    let manifest = with_semantics(PAIR_SEMANTICS, PAIRS, &catalog).await;
    let leaf = |name| dim(name).expect("a name");
    let comparisons = [
        (
            leaf("a").is_not_distinct_from(leaf("b")),
            column("a").is_not_distinct_from(column("b")),
        ),
        (leaf("a").eq(leaf("b")), column("a").eq(column("b"))),
    ];

    for (predicate, written) in comparisons {
        let (plan, _) = planned(&manifest, &select(&["n"]).with_predicate(predicate));

        assert_eq!(filtered(&plan).0.filter, written);
    }
}

#[tokio::test]
async fn an_activated_named_filter_is_a_boolean_field_that_stage_3_applies() {
    let manifest = skeleton().await;
    let orders = columns_of(&manifest, "orders");
    let request = select(&["region", "channel", "revenue"]).with_filters(vec!["eu_web".into()]);

    let (plan, _) = planned(&manifest, &request);

    let (filter, row_level) = filtered(&plan);
    assert_eq!(filter.filter, column("eu_web"));
    let fields = names(&row_level.common.output_schema);
    assert_eq!(
        (&fields[..3], fields.len()),
        (&["channel", "eu_web", "region"][..], 4)
    );
    let eu_web = &row_level.common.output_schema.columns[1];
    assert_eq!(eu_web.data_type, DataType::Boolean);
    assert_eq!(
        row_level.expressions[1],
        orders("region").eq(text("eu")) & orders("channel").eq(text("web"))
    );
    assert_eq!(
        names(&scan_of(&plan).schema),
        ["region", "channel", "amount"]
    );
    assert_eq!(
        aggregate_step(&plan.root).groupings,
        [column("region"), column("channel")]
    );
    assert_eq!(output_names(&plan), ["region", "channel", "revenue"]);
}

#[tokio::test]
async fn activated_named_filters_are_conjoined_with_the_request_predicate() {
    let web = dim("channel").expect("a name").eq(lit("web"));
    let request = (select(&["revenue"]).with_predicate(web))
        .with_filters(vec!["is_eu".into(), "eu_web".into()]);

    let (plan, _) = planned(&skeleton().await, &request);

    let (filter, row_level) = filtered(&plan);
    assert_eq!(
        filter.filter,
        column("channel").eq(text("web")) & column("is_eu") & column("eu_web")
    );
    let fields = names(&row_level.common.output_schema);
    assert_eq!(
        (&fields[..3], fields.len()),
        (&["channel", "eu_web", "is_eu"][..], 4)
    );
    assert_eq!(output_names(&plan), ["revenue"]);
}

const CREATED_SEMANTICS: &str = r#"
  dimensions:
    - name: created_at
      data_type: timestamp
      type:
        temporal:
          granularities: [day]
  measures:
    - name: revenue
      data_type: bigint
      agg: sum
"#;

const CREATED_ORDERS: &str = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          created_at: created_at
          revenue: amount
      dimensions:
        - ref: created_at
      measures:
        - ref: revenue
"#;

async fn created_orders() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[("created_at", "timestamptz"), ("amount", "long")],
        )
        .with_dialect(Dialect::Iceberg);
    with_semantics(CREATED_SEMANTICS, CREATED_ORDERS, &catalog).await
}

#[tokio::test]
async fn a_predicate_on_an_iceberg_timestamptz_types_its_literal_as_the_field() {
    let manifest = created_orders().await;
    let since = (dim("created_at").expect("a name")).gt_eq(untyped("2026-02-01 00:00:00"));

    let (plan, _) = planned(&manifest, &select(&["revenue"]).with_predicate(since));

    let (filter, row_level) = filtered(&plan);
    let created_at = &row_level.common.output_schema.columns[0];
    assert_eq!(
        (created_at.name.as_str(), &created_at.data_type),
        ("created_at", &DataType::Timestamp { precision: 6 })
    );
    let typed = Expr::Literal(Literal::Timestamp {
        value: "2026-02-01 00:00:00".into(),
        precision: 6,
    });
    assert_eq!(filter.filter, column("created_at").gt_eq(typed));
    assert_eq!(output_names(&plan), ["revenue"]);
}

const EVENTS: &str = r#"
    - name: events
      extras:
        storage:
          tables: [warehouse.events]
        semantic_mapping:
          ts: ts
          d: d
          event_count: {lit: 1}
      dimensions:
        - name: ts
          data_type: timestamp
          type:
            temporal:
              granularities: [hour, month]
        - name: d
          data_type: date
          type:
            temporal:
              granularities: [week]
      measures:
        - name: event_count
          data_type: bigint
          agg: count
"#;

async fn events() -> SemanticManifest {
    let catalog = Catalog::default().table(
        "warehouse.events",
        FileFormat::Iceberg,
        &[("ts", "timestamp(6)"), ("d", "date")],
    );
    with_semantics("", EVENTS, &catalog).await
}

fn in_timezone(name: &str) -> RequestContext {
    RequestContext::default().with_timezone(Timezone::new(name).expect("an IANA timezone"))
}

#[tokio::test]
async fn a_timestamp_at_a_granularity_is_bucketed_on_the_request_timezone_before_the_aggregate_step(
) {
    let manifest = events().await;

    for (granularity, written, timezone) in [
        (Granularity::Month, "month", "Europe/Stockholm"),
        (Granularity::Hour, "hour", "America/New_York"),
    ] {
        let request = at_granularity("ts", granularity, &["event_count"]);

        let (plan, _) = planned_in(&manifest, &request, &in_timezone(timezone));

        let bucketing = below_aggregate(&plan);
        let args = vec![text(written), column("ts"), text(timezone)];
        assert_eq!(fields(bucketing), [("ts", call("date_trunc", args))]);
        let row_level = projection(&bucketing.input);
        assert_eq!(names(&row_level.common.output_schema), ["ts"]);
        assert_eq!(aggregate_step(&plan.root).groupings, [column("ts")]);
        assert_eq!(output_names(&plan), ["ts", "event_count"]);
    }
}

#[tokio::test]
async fn a_timestamp_bucket_writes_the_request_timezone_even_at_its_default() {
    let manifest = created_orders().await;
    let request = at_granularity("created_at", Granularity::Month, &["revenue"]);

    for (context, timezone) in [
        (RequestContext::default(), "UTC"),
        (in_timezone("Europe/Stockholm"), "Europe/Stockholm"),
    ] {
        let (plan, _) = planned_in(&manifest, &request, &context);

        let bucketing = below_aggregate(&plan);
        let row_level = projection(&bucketing.input);
        let ["created_at", argument] = names(&row_level.common.output_schema)[..] else {
            panic!("two fields, got {:?}", row_level.common.output_schema);
        };
        let args = vec![text("month"), column("created_at"), text(timezone)];
        assert_eq!(
            fields(bucketing),
            [
                ("created_at", call("date_trunc", args)),
                (argument, column(argument)),
            ]
        );
        let aggregate = aggregate_step(&plan.root);
        assert_eq!(aggregate.groupings, [column("created_at")]);
        let [(_, sum)] = aggregate.aggregates.as_slice() else {
            panic!("one entry, got {:?}", aggregate.aggregates);
        };
        assert_eq!(sum.args(), [column(argument)]);
        assert_eq!(output_names(&plan), ["created_at", "revenue"]);
    }
}

#[tokio::test]
async fn a_date_at_the_week_grain_is_bucketed_on_the_request_week_start_with_no_timezone() {
    let manifest = events().await;
    let request = at_granularity("d", Granularity::Week, &["event_count"]);
    let sunday = RequestContext::default().with_week_start(Weekday::Sunday);

    for (context, week_start) in [(sunday, "sunday"), (RequestContext::default(), "monday")] {
        let (plan, _) = planned_in(&manifest, &request, &context);

        let bucketing = below_aggregate(&plan);
        let args = vec![text("week"), column("d"), text(week_start)];
        assert_eq!(fields(bucketing), [("d", call("date_trunc", args))]);
        let [Expr::Function(date_trunc)] = &bucketing.expressions[..] else {
            panic!("one call, got {:?}", bucketing.expressions);
        };
        assert_eq!(date_trunc.argument("week_start"), Some(&text(week_start)));
        assert_eq!(date_trunc.argument("timezone"), None);
        let row_level = projection(&bucketing.input);
        assert_eq!(names(&row_level.common.output_schema), ["d"]);
        assert_eq!(aggregate_step(&plan.root).groupings, [column("d")]);
        assert_eq!(output_names(&plan), ["d", "event_count"]);
    }
}

const SPEND: &str = r#"
    - name: spend
      extras:
        storage:
          tables: [warehouse.spend]
        semantic_mapping:
          campaign_id: campaign_id
          date: date
          cost: cost
      dimensions:
        - name: campaign_id
          data_type: string
          type: categorical
        - name: date
          data_type: date
          type:
            temporal:
              granularities: [day]
      measures:
        - name: cost
          data_type: bigint
          agg: sum
"#;

#[tokio::test]
async fn a_predicate_on_a_bucketed_dimension_reads_its_value_below_the_bucket() {
    let catalog = Catalog::default().table(
        "warehouse.spend",
        FileFormat::Iceberg,
        &[
            ("campaign_id", "string"),
            ("date", "date"),
            ("cost", "long"),
        ],
    );
    let manifest = with_semantics("", SPEND, &catalog).await;
    let since = (dim("date").expect("a name")).gt_eq(untyped("2026-03-15"));
    let request = at_granularity("date", Granularity::Month, &["cost"]).with_predicate(since);

    let (plan, _) = planned(&manifest, &request);

    let bucketing = below_aggregate(&plan);
    let filter = filter(&bucketing.input);
    let day = Expr::Literal(Literal::Date("2026-03-15".into()));
    assert_eq!(filter.filter, column("date").gt_eq(day));
    let row_level = projection(&filter.input);
    let [argument, "date"] = names(&row_level.common.output_schema)[..] else {
        panic!("two fields, got {:?}", row_level.common.output_schema);
    };
    let date_trunc = call("date_trunc", vec![text("month"), column("date")]);
    assert_eq!(
        fields(bucketing),
        [(argument, column(argument)), ("date", date_trunc)]
    );
    assert_eq!(aggregate_step(&plan.root).groupings, [column("date")]);
    assert_eq!(output_names(&plan), ["date", "cost"]);
}

#[tokio::test]
async fn the_scope_root_projects_the_request_names_in_select_order() {
    let (plan, _) = planned(&skeleton().await, &select(&["revenue", "region"]));

    let root = projection(&plan.root);
    let aggregate = aggregate_step(&root.input);
    assert_eq!(
        names(&aggregate.common.output_schema),
        ["region", "revenue"]
    );
    assert_eq!(root.expressions, [column("revenue"), column("region")]);
    let passed = aggregate.common.output_schema.columns.len();
    let emitted = (root.common.emit_form.as_ref()).map(|emit| &emit.output_mapping);
    assert_eq!(emitted, Some(&vec![passed, passed + 1]));
    assert_eq!(names(&root.common.output_schema), ["revenue", "region"]);
    assert_eq!(output_names(&plan), ["revenue", "region"]);
}

#[tokio::test]
async fn the_scan_reads_the_named_table_with_an_empty_catalog_and_the_reported_format() {
    let (plan, _) = planned(&skeleton().await, &select(&["region", "revenue"]));

    assert_eq!(scan_of(&plan).reader, orders_table());
}

const COUNTS: &str = r#"
    - name: counts
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          share: order_id
      dimensions:
        - ref: region
      measures:
        - name: share
          data_type: double
          agg: count
"#;

#[tokio::test]
async fn a_cast_measure_keeps_its_cast_in_a_derivation_above_the_aggregate_step() {
    let manifest = with_datasets(&format!("{ORDERS}{COUNTS}"), &orders_catalog()).await;

    let (plan, _) = planned(&manifest, &select(&["region", "share"]));

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let [(entry, count)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert!(entry.as_str().starts_with("__plan_"), "{entry:?}");
    assert_eq!(count.function().declarations.canonical.name, "count");
    let cast = Expr::Cast {
        input: Box::new(column(entry.as_str())),
        target: DataType::Double,
        on_failure: CastFailure::Error,
    };
    assert_eq!(derivation.expressions, [column("region"), cast]);
    assert_eq!(output_names(&plan), ["region", "share"]);
}

#[tokio::test]
async fn a_metric_over_measures_is_a_stage_5_projection_over_one_aggregate_step() {
    let manifest = skeleton().await;
    let orders = columns_of(&manifest, "orders");

    let (grouped, _) = planned(&manifest, &select(&["region", "aov"]));
    let (one_row, _) = planned(&manifest, &select(&["aov"]));

    let root = projection(&grouped.root);
    assert_eq!(root.expressions, [column("region"), column("aov")]);
    let derivations = [
        (projection(&root.input), vec![column("region")]),
        (projection(&one_row.root), Vec::new()),
    ];
    for (derivation, groupings) in derivations {
        let aggregate = aggregate_step(&derivation.input);
        assert_eq!(aggregate.groupings, groupings);
        let [(sum_entry, sum), (count_entry, count)] = aggregate.aggregates.as_slice() else {
            panic!("two entries, got {:?}", aggregate.aggregates);
        };
        assert_eq!(sum.function().declarations.canonical.name, "sum");
        assert_eq!(count.function().declarations.canonical.name, "count");
        let row_level = projection(&aggregate.input);
        let arguments: Vec<_> = (sum.args().iter().chain(count.args()))
            .map(|argument| computed(row_level, argument))
            .collect();
        assert_eq!(arguments, [orders("amount"), orders("order_id")]);
        let formula = column(sum_entry.as_str()) / column(count_entry.as_str());
        assert_eq!(derivation.expressions[0], formula);
    }
    assert_eq!(output_names(&grouped), ["region", "aov"]);
    assert_eq!(output_names(&one_row), ["aov"]);
}

fn condition<'call>(call: &'call Aggregate, semantics: &str) -> &'call PhysicalExpr {
    let condition = call.filter().expect("a filter clause");
    let Expr::Column(field) = condition else {
        panic!("a field of stage 2, got {condition:?}");
    };
    assert!(
        field.name().starts_with("__plan_cond") && field.name().ends_with(&format!("_{semantics}")),
        "{field:?}"
    );
    condition
}

#[tokio::test]
async fn a_measure_filter_is_its_aggregate_calls_filter_clause_over_a_stage_2_field() {
    let manifest = skeleton().await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["region", "web_revenue", "revenue"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [(web_entry, web), (revenue_entry, revenue)] = aggregate.aggregates.as_slice() else {
        panic!("two entries, got {:?}", aggregate.aggregates);
    };
    assert_eq!(
        [web_entry.as_str(), revenue_entry.as_str()],
        ["web_revenue", "revenue"]
    );
    for entry in [web, revenue] {
        assert_eq!(entry.function().declarations.canonical.name, "sum");
    }
    assert_eq!(revenue.filter(), None);
    let row_level = projection(&aggregate.input);
    assert_eq!(
        computed(row_level, condition(web, "web_revenue")),
        orders("channel").eq(text("web"))
    );
    assert_eq!(output_names(&plan), ["region", "web_revenue", "revenue"]);
}

const FLAG_SEMANTICS: &str = r#"
  dimensions:
    - name: region
      data_type: string
      type: categorical
    - name: flag
      data_type: boolean
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
"#;

const FLAGGED_ORDERS: &str = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          flag: __plan_cond_amount
          revenue: amount
      dimensions:
        - ref: region
        - ref: flag
      measures:
        - ref: revenue
"#;

#[tokio::test]
async fn a_physical_column_named_like_a_condition_field_is_no_aggregation_condition() {
    let catalog = Catalog::default().table(
        "warehouse.orders",
        FileFormat::Iceberg,
        &[
            ("region", "string"),
            ("amount", "double"),
            ("__plan_cond_amount", "boolean"),
        ],
    );
    let manifest = with_semantics(FLAG_SEMANTICS, FLAGGED_ORDERS, &catalog).await;

    for names in [&["flag", "revenue"][..], &["region", "flag", "revenue"]] {
        let (plan, _) = planned(&manifest, &select(names));

        assert_eq!(
            aggregate_filters(&plan.root),
            [("revenue", None)],
            "{names:?}"
        );
        assert_eq!(output_names(&plan), names);
    }
}

fn difference_of_sums(derivation: &ProjectNode) -> ([&Aggregate; 2], &ProjectNode) {
    let aggregate = aggregate_step(&derivation.input);
    let [(left_entry, left), (right_entry, right)] = aggregate.aggregates.as_slice() else {
        panic!("two entries, got {:?}", aggregate.aggregates);
    };
    for entry in [left, right] {
        assert_eq!(entry.function().declarations.canonical.name, "sum");
    }
    let formula = column(left_entry.as_str()) - column(right_entry.as_str());
    assert_eq!(derivation.expressions[0], formula);
    ([left, right], projection(&aggregate.input))
}

#[tokio::test]
async fn a_metric_keeps_a_linked_measure_filter_on_its_inlined_aggregate_once() {
    let manifest = skeleton().await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["region", "non_web"]));

    let root = projection(&plan.root);
    assert_eq!(root.expressions, [column("region"), column("non_web")]);
    let ([revenue, web], row_level) = difference_of_sums(projection(&root.input));
    assert_eq!(revenue.filter(), None);
    assert_eq!(
        computed(row_level, condition(web, "non_web")),
        orders("channel").eq(text("web"))
    );
    assert_eq!(output_names(&plan), ["region", "non_web"]);
}

const EU_ORDERS: &str = r#"
    - name: eu_orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          channel: channel
          revenue: amount
          web_revenue: amount
      dimensions:
        - ref: region
        - ref: channel
      measures:
        - ref: revenue
        - ref: web_revenue
      metrics:
        - name: eu_non_web
          data_type: double
          filter:
            eq: [region, {lit: eu}]
          expr:
            subtract: [revenue, web_revenue]
"#;

#[tokio::test]
async fn a_metric_filter_is_joined_onto_every_aggregate_entry_after_a_linked_filter() {
    let manifest = with_datasets(&format!("{ORDERS}{EU_ORDERS}"), &orders_catalog()).await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["eu_non_web"]));

    let ([revenue, web], row_level) = difference_of_sums(projection(&plan.root));
    let eu = orders("region").eq(text("eu"));
    let conditions = [revenue, web].map(|call| computed(row_level, condition(call, "eu_non_web")));
    assert_eq!(
        conditions,
        [eu.clone(), orders("channel").eq(text("web")) & eu]
    );
    assert_eq!(output_names(&plan), ["eu_non_web"]);
}

#[tokio::test]
async fn a_measure_predicate_is_a_filter_node_directly_above_the_aggregate_step() {
    let over = measure("revenue").expect("a name").gt(lit(160));
    let request = select(&["region", "revenue"]).with_predicate(over);

    let (plan, _) = planned(&skeleton().await, &request);

    let filter = filter(&plan.root);
    assert_eq!(filter.filter, column("revenue").gt(integer(160)));
    let aggregate = aggregate_step(&filter.input);
    assert_eq!(aggregate.groupings, [column("region")]);
    assert_eq!(
        names(&aggregate.common.output_schema),
        ["region", "revenue"]
    );
    assert!(
        matches!(*aggregate.input, PlanNode::Project(_)),
        "{:?}",
        aggregate.input
    );
    assert_eq!(scan_of(&plan).filter, None);
    assert_eq!(output_names(&plan), ["region", "revenue"]);
}

#[tokio::test]
async fn a_metric_predicate_is_a_filter_node_above_the_projection_that_computes_it() {
    let over = metric("aov").expect("a name").gt(lit(80));
    let request = select(&["region", "aov"]).with_predicate(over);

    let (plan, _) = planned(&skeleton().await, &request);

    let root = projection(&plan.root);
    assert_eq!(root.expressions, [column("region"), column("aov")]);
    let filter = filter(&root.input);
    assert_eq!(filter.filter, column("aov").gt(integer(80)));
    let derivation = projection(&filter.input);
    assert_eq!(names(&derivation.common.output_schema), ["aov", "region"]);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [(sum, _), (count, _)] = aggregate.aggregates.as_slice() else {
        panic!("two entries, got {:?}", aggregate.aggregates);
    };
    let formula = column(sum.as_str()) / column(count.as_str());
    assert_eq!(derivation.expressions[0], formula);
    assert_eq!(output_names(&plan), ["region", "aov"]);
}

#[tokio::test]
async fn a_measure_filtered_but_not_selected_is_computed_and_dropped_from_the_output() {
    let over = measure("revenue").expect("a name").gt(lit(160));

    let (plan, _) = planned(&skeleton().await, &select(&["region"]).with_predicate(over));

    let root = projection(&plan.root);
    assert_eq!(root.expressions, [column("region")]);
    let filter = filter(&root.input);
    assert_eq!(filter.filter, column("revenue").gt(integer(160)));
    let aggregate = aggregate_step(&filter.input);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [(entry, revenue)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "revenue");
    assert_eq!(revenue.function().declarations.canonical.name, "sum");
    assert_eq!(output_names(&plan), ["region"]);
}

#[tokio::test]
async fn an_order_key_puts_its_nulls_high_in_either_direction() {
    let manifest = skeleton().await;
    let directions = [
        (
            Direction::Asc,
            SortDir::Asc {
                nulls: NullOrdering::Last,
            },
        ),
        (
            Direction::Desc,
            SortDir::Desc {
                nulls: NullOrdering::First,
            },
        ),
    ];

    for (direction, dir) in directions {
        let key = OrderKey::new("channel", direction);
        let request = select(&["channel", "revenue"]).with_order_by(vec![key]);

        let (plan, _) = planned(&manifest, &request);

        let sort = sort(&plan.root);
        let expr = column("channel");
        assert_eq!(sort.order_by, [SortKey { expr, dir }]);
        assert_eq!(aggregate_step(&sort.input).groupings, [column("channel")]);
        assert_eq!(output_names(&plan), ["channel", "revenue"]);
    }
}

#[tokio::test]
async fn offset_and_limit_are_a_fetch_node_above_the_sort_with_each_bound_clamped() {
    let manifest = skeleton().await;
    let key = OrderKey::new("revenue", Direction::Desc);
    let ordered = select(&["region", "revenue"]).with_order_by(vec![key]);
    let bounds = [
        (ordered.clone().with_offset(1).with_limit(1), (1, Some(1))),
        (ordered.clone().with_offset(1), (1, None)),
        (
            ordered.clone().with_offset(1).with_limit(u64::MAX),
            (1, Some(i64::MAX)),
        ),
        (ordered.with_limit(1), (0, Some(1))),
    ];

    for (request, bound) in bounds {
        let (plan, _) = planned(&manifest, &request);

        let PlanNode::Fetch(fetch) = &plan.root else {
            panic!("a fetch, got {:?}", plan.root);
        };
        assert_eq!((fetch.offset, fetch.limit), bound);
        let sort = sort(&fetch.input);
        let dir = SortDir::Desc {
            nulls: NullOrdering::First,
        };
        let expr = column("revenue");
        assert_eq!(sort.order_by, [SortKey { expr, dir }]);
        assert_eq!(aggregate_step(&sort.input).groupings, [column("region")]);
        assert!(sort.common.semantic_meta.is_empty());
        assert!(
            matches!(
                fetch.common.semantic_meta[..],
                [SemAnnotation::Rank {
                    compositions: 0,
                    granularity_fit: GranularityGroup::NotSelected,
                    granularity: None,
                    ..
                }]
            ),
            "{:?}",
            fetch.common.semantic_meta
        );
        assert_eq!(output_names(&plan), ["region", "revenue"]);
    }
}

const TYPED_ORDERS: &str = r#"
    - name: typed_orders
      extras:
        storage:
          tables: [warehouse.typed_orders]
        semantic_mapping:
          region: region
          customer_id: customer_id
      dimensions:
        - ref: region
        - ref: customer_id
"#;

#[tokio::test]
async fn the_row_level_projection_casts_only_where_the_catalog_type_differs() {
    let catalog = orders_catalog().table(
        "warehouse.typed_orders",
        FileFormat::Iceberg,
        &[("customer_id", "long"), ("region", "string")],
    );
    let manifest = with_datasets(&format!("{ORDERS}{TYPED_ORDERS}"), &catalog).await;
    let request = select(&["customer_id", "region"]);

    let (typed, _) = planned(&manifest, &request.clone().with_from("typed_orders"));
    let (agreeing, _) = planned(&manifest, &request.with_from("orders"));

    let typed_orders = columns_of(&manifest, "typed_orders");
    let cast = Expr::Cast {
        input: Box::new(typed_orders("customer_id")),
        target: DataType::String,
        on_failure: CastFailure::Error,
    };
    assert_eq!(
        below_aggregate(&typed).expressions,
        [cast, typed_orders("region")]
    );
    let orders = columns_of(&manifest, "orders");
    assert_eq!(
        below_aggregate(&agreeing).expressions,
        [orders("customer_id"), orders("region")]
    );
}

const LABEL_SEMANTICS: &str = r#"
  dimensions:
    - name: region
      data_type: string
      type: categorical
    - name: region_label
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: double
      agg: sum
"#;

const LABELS: &str = r#"
    - name: upper_orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          revenue: amount
      dimensions:
        - ref: region
        - ref: region_label
          expr: {upper: region}
      measures:
        - ref: revenue
    - name: lower_orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          revenue: amount
      dimensions:
        - ref: region
        - ref: region_label
          expr: {lower: region}
      measures:
        - ref: revenue
"#;

#[tokio::test]
async fn the_row_level_projection_computes_the_tree_of_the_dataset_that_from_names() {
    let manifest = with_semantics(LABEL_SEMANTICS, LABELS, &orders_catalog()).await;
    let orders = columns_of(&manifest, "orders");

    for (dataset, function) in [("upper_orders", "upper"), ("lower_orders", "lower")] {
        let request = select(&["region_label", "revenue"]).with_from(dataset);
        let (plan, _) = planned(&manifest, &request);

        assert_eq!(
            below_aggregate(&plan).expressions,
            [call(function, vec![orders("region")]), orders("amount")],
            "{dataset}"
        );
    }
}

const MARKET: &str = "{case: {whens: [\
    {when: {in: {value: {col: dataset_name}, list: [{lit: adwords}, {lit: facebookads}]}}, \
     then: {upper: {regexp_extract: [{col: campaign}, {lit: '^([A-Z]{2})_'}, {lit: 1}]}}}, \
    {when: {eq: [{col: dataset_name}, {lit: impact}]}, \
     then: {regexp_extract: [{col: campaign}, {lit: '^Acme ([A-Z]{2})'}, {lit: 1}]}}], \
    else: {lit: ''}}}";

#[tokio::test]
async fn a_regexp_extract_dimension_plans_as_its_stored_tree_under_distinct_rows() {
    let catalog = Catalog::default().table(
        "warehouse.campaigns",
        FileFormat::Iceberg,
        &[("dataset_name", "string"), ("campaign", "string")],
    );
    let campaigns = dataset("campaigns", &[["market", "string", MARKET]]);
    let manifest = with_semantics("", &campaigns, &catalog).await;

    let (plan, _) = planned(&manifest, &select(&["market"]));

    let market = stored(&manifest, "campaigns", "market");
    assert_eq!(distinct_rows(&plan), [("market", market)]);
}

#[tokio::test]
async fn a_concat_dimension_plans_as_its_tree_under_distinct_rows() {
    for [dimension, first, second] in [["country_type", "country", "txn_type"], ["c", "a", "b"]] {
        let catalog = Catalog::default().table(
            "warehouse.rows",
            FileFormat::Iceberg,
            &[(first, "string"), (second, "string")],
        );
        let concat = format!("{{concat: [{{col: {first}}}, {{lit: '-'}}, {{col: {second}}}]}}");
        let datasets = dataset("rows", &[[dimension, "string", concat.as_str()]]);
        let manifest = with_semantics("", &datasets, &catalog).await;

        let (plan, _) = planned(&manifest, &select(&[dimension]));

        let rows = columns_of(&manifest, "rows");
        let concat = call("concat", vec![rows(first), text("-"), rows(second)]);
        assert_eq!(distinct_rows(&plan), [(dimension, concat)]);
    }
}

const COMPONENTS: [&str; 10] = [
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "day_of_week",
    "day_of_year",
    "hour",
    "minute",
    "second",
];

fn temporal(
    prefix: &str,
    value: &str,
    data_type: &str,
    grains: &[&str],
    components: &[&str],
) -> Vec<[String; 3]> {
    let truncated = (grains.iter()).map(|grain| {
        let expr = format!("{{date_trunc: [{grain}, {{col: {value}}}]}}");
        [format!("{prefix}_{grain}"), data_type.to_owned(), expr]
    });
    let extracted = (components.iter()).map(|component| {
        let expr = format!("{{extract: [{component}, {{col: {value}}}]}}");
        [
            format!("{prefix}_{component}_part"),
            "bigint".to_owned(),
            expr,
        ]
    });
    truncated.chain(extracted).collect()
}

#[tokio::test]
async fn a_date_trunc_or_extract_dimension_plans_as_compile_wrote_it_in_every_context() {
    let catalog = Catalog::default()
        .table(
            "warehouse.sensor_readings",
            FileFormat::Iceberg,
            &[("event_timestamp", "timestamp(6)")],
        )
        .table(
            "warehouse.events",
            FileFormat::Iceberg,
            &[("created_at", "date")],
        );

    let readings = temporal(
        "reading",
        "event_timestamp",
        "timestamp",
        &["hour", "week"],
        &COMPONENTS,
    );
    let events = temporal(
        "event",
        "created_at",
        "date",
        &["month", "week"],
        &COMPONENTS[..7],
    );
    let datasets = dataset("sensor_readings", &readings) + &dataset("events", &events);
    let manifest = with_semantics("", &datasets, &catalog).await;
    let stockholm = Timezone::new("Europe/Stockholm").expect("an IANA timezone");
    let contexts = [
        RequestContext::default().with_timezone(stockholm),
        RequestContext::default().with_week_start(Weekday::Sunday),
    ];

    for (dataset, dimensions) in [("sensor_readings", &readings), ("events", &events)] {
        let mut names: Vec<&str> = (dimensions.iter())
            .map(|[name, ..]| name.as_str())
            .collect();
        names.sort_unstable();
        let request = select(&names);

        let (plan, _) = planned(&manifest, &request);

        let written: Vec<(&str, PhysicalExpr)> = (names.iter())
            .map(|name| (*name, stored(&manifest, dataset, name)))
            .collect();
        assert_eq!(distinct_rows(&plan), written, "{dataset}");
        for context in &contexts {
            assert_eq!(
                planned_in(&manifest, &request, context).0,
                plan,
                "{context:?}"
            );
        }
    }
    let (plan, _) = planned(&manifest, &select(&["reading_hour", "reading_week"]));
    let value = columns_of(&manifest, "sensor_readings")("event_timestamp");
    let date_trunc = |grain, week_start: Option<&str>| {
        let args = [text(grain), value.clone(), text("UTC")].into_iter();
        call("date_trunc", args.chain(week_start.map(text)).collect())
    };
    assert_eq!(
        distinct_rows(&plan),
        [
            ("reading_hour", date_trunc("hour", None)),
            ("reading_week", date_trunc("week", Some("monday"))),
        ]
    );
}

#[tokio::test]
async fn the_harness_table_case_plans_over_its_table_dataset() {
    let (plan, _) = planned(&harness().await, &select(&["region", "revenue"]));

    assert_eq!(scan_of(&plan).reader, orders_table());
    assert_eq!(aggregate_step(&plan.root).groupings, [column("region")]);
    assert_eq!(output_names(&plan), ["region", "revenue"]);
}

#[tokio::test]
async fn a_tie_plans_as_from_the_first_in_name_order_trace_included() {
    let manifest = with_datasets(&format!("{ORDERS}{TWINS}"), &orders_catalog()).await;
    let request = select(&["region", "revenue"]);

    let (field_first, _) = planned(&manifest, &request);
    let (from, _) = planned(&manifest, &request.clone().with_from("orders_a"));

    assert_eq!(field_first, from);
    let chosen = DataKindName::new("orders_a");
    assert_eq!(
        scan_of(&field_first).common.semantic_meta,
        [SemAnnotation::SemanticSource {
            semantic_source: chosen.clone(),
            source_role: SourceRole::Dataset,
            originators: Vec::new(),
            physical_source: Some(Name::new("orders").expect("a name")),
            cardinality: None,
            delivery: None,
            answered_by: None,
            pruned: Vec::new(),
        }]
    );
    assert_eq!(
        field_first.root.common().semantic_meta,
        [SemAnnotation::Rank {
            scope_root: chosen,
            compositions: 0,
            granularity_fit: GranularityGroup::NotSelected,
            granularity: None,
        }]
    );
}

const SHOP_SEMANTICS: &str = r#"
  dimensions:
    - name: region
      data_type: string
      type: categorical
    - name: channel
      data_type: string
      type: categorical
    - name: customer_id
      data_type: string
      type: categorical
    - name: ordered_at
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: revenue
      data_type: double
      agg: sum
      expr: {col: amount}
    - name: order_count
      data_type: bigint
      agg: count
      expr: {col: order_id}
    - name: buyers
      data_type: bigint
      agg: count_distinct
      expr: {col: customer_id}
    - name: revenue_eu_adj
      data_type: double
      agg: sum
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [{col: amount}, {lit: 1.2}]}
          else: {col: amount}
    - name: tv_revenue
      data_type: double
      agg: sum
      expr: {col: amount}
      filter:
        eq: [channel, {lit: tv}]
  metrics:
    - name: revenue_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
    - name: revenue_eu_only
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
    - name: revenue_eu_or_zero
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: {lit: 0}
    - name: revenue_eu_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {add: [revenue, {lit: 100}]}
          else: revenue
    - name: avg_order_value_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [{divide: [revenue, order_count]}, {lit: 1.2}]}
          else: {divide: [revenue, order_count]}
    - name: buyers_eu_weighted
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [buyers, {lit: 2}]}
          else: buyers
    - name: revenue_eu_web_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then:
                add:
                  - revenue
                  - case:
                      whens:
                        - when: {eq: [channel, {lit: web}]}
                          then: {lit: 100}
                      else: {lit: 0}
          else: revenue
    - name: buyers_weighted
      data_type: bigint
      expr:
        multiply:
          - case:
              whens:
                - when: {eq: [region, {lit: eu}]}
                  then: {lit: 2}
                - when: {eq: [channel, {lit: web}]}
                  then: {lit: 2}
              else: {lit: 1}
          - buyers
    - name: buyers_inner_doubled
      data_type: bigint
      expr:
        case:
          whens:
            - when:
                eq:
                  - case:
                      whens:
                        - when: {eq: [region, {lit: eu}]}
                          then: {lit: 1}
                        - when: {eq: [channel, {lit: web}]}
                          then: {lit: 1}
                      else: {lit: 0}
                  - {lit: 1}
              then: {multiply: [{lit: 2}, buyers]}
          else: buyers
    - name: buyers_later_condition
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: buyers
            - when:
                eq:
                  - case:
                      whens:
                        - when: {eq: [channel, {lit: web}]}
                          then: {lit: 1}
                        - when: {eq: [channel, {lit: app}]}
                          then: {lit: 1}
                      else: {lit: 0}
                  - {lit: 1}
              then: {multiply: [{lit: 2}, buyers]}
          else: buyers
    - name: buyers_drawn
      data_type: bigint
      expr:
        add:
          - multiply:
              - case:
                  whens:
                    - when: {lt: [{random: []}, {lit: 0.5}]}
                      then: {lit: 2}
                  else: {lit: 1}
              - buyers
          - case:
              whens:
                - when: {eq: [region, {lit: eu}]}
                  then: buyers
              else: {lit: 0}
    - name: buyers_weighted_drawn
      data_type: double
      expr:
        multiply:
          - multiply:
              - {random: []}
              - case:
                  whens:
                    - when: {eq: [region, {lit: eu}]}
                      then: {lit: 2}
                    - when: {eq: [channel, {lit: web}]}
                      then: {lit: 2}
                  else: {lit: 1}
          - buyers
    - name: buyers_eu_twice
      data_type: bigint
      expr:
        multiply:
          - case:
              whens:
                - when: {eq: [region, {lit: eu}]}
                  then: {lit: 2}
              else: {lit: 1}
          - buyers
    - name: buyers_eu_thrice
      data_type: bigint
      expr:
        multiply:
          - case:
              whens:
                - when: {eq: [region, {lit: eu}]}
                  then: {lit: 3}
              else: {lit: 1}
          - buyers
    - name: revenue_early_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {lt: [ordered_at, {lit: '2026-10-15'}]}
              then: {multiply: [revenue, {lit: 0.9}]}
          else: revenue
    - name: region_label
      data_type: string
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {lit: EU}
          else: {lit: other}
    - name: target
      data_type: bigint
      expr:
        case:
          whens:
            - when: {lit: false}
              then: order_count
          else: {lit: 1000}
    - name: eu_revenue
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region_label, {lit: EU}]}
              then: revenue
    - name: tier
      data_type: string
      expr:
        case:
          whens:
            - when: {and: [{eq: [region_label, {lit: EU}]}, {gt: [revenue, {lit: 100}]}]}
              then: {lit: high}
          else: {lit: low}
    - name: eu_order_count
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: order_count
    - name: c1_tv_revenue_adj
      data_type: double
      filter:
        eq: [customer_id, {lit: c1}]
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [tv_revenue, {lit: 1.2}]}
          else: tv_revenue
    - name: tv_or_revenue
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: tv_revenue
          else: revenue
    - name: web_revenue_eu_or_zero
      data_type: double
      filter:
        eq: [channel, {lit: web}]
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: {lit: 0}
    - name: web_eu_order_count
      data_type: bigint
      filter:
        eq: [channel, {lit: web}]
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: order_count
    - name: regions_with_revenue
      data_type: bigint
      agg: count
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
    - name: peak_region_revenue
      data_type: decimal(18,2)
      agg: max
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
    - name: avg_customer_revenue
      data_type: double
      agg: avg
      expr: revenue
    - name: revenue_early_adj_sum
      data_type: double
      agg: sum
      expr:
        case:
          whens:
            - when: {lt: [ordered_at, {lit: '2026-10-15'}]}
              then: {multiply: [revenue, {lit: 0.9}]}
          else: revenue
    - name: n_other
      data_type: bigint
      agg: count
      expr:
        case:
          whens:
            - when: {eq: [region_label, {lit: other}]}
              then: revenue
    - name: eu_web_regions
      data_type: bigint
      agg: count
      expr:
        case:
          whens:
            - when: {and: [{eq: [region, {lit: eu}]}, {eq: [channel, {lit: web}]}]}
              then: revenue
    - name: web_eu_regions
      data_type: bigint
      agg: count
      expr:
        case:
          whens:
            - when: {and: [{eq: [channel, {lit: web}]}, {eq: [region, {lit: eu}]}]}
              then: revenue
    - name: revenue_per_region
      data_type: double
      expr: {divide: [revenue, regions_with_revenue]}
    - name: eu_regions
      data_type: bigint
      filter:
        eq: [region, {lit: eu}]
      expr: regions_with_revenue
    - name: regions_gap
      data_type: bigint
      expr: {subtract: [regions_with_revenue, eu_regions]}
    - name: web_regions
      data_type: bigint
      agg: sum
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: regions_with_revenue
    - name: web_regions_in_eu
      data_type: bigint
      filter:
        eq: [region, {lit: eu}]
      expr: web_regions
    - name: regions_eu_bonus
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {add: [regions_with_revenue, {lit: 1}]}
          else: regions_with_revenue
    - name: big_orders
      data_type: bigint
      agg: count
      expr:
        case:
          whens:
            - when: {gt: [revenue, {lit: 100}]}
              then: {lit: big}
    - name: big_eu_orders
      data_type: bigint
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: big_orders
"#;

async fn shop(region: &str) -> SemanticManifest {
    let datasets = format!(
        r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
      dimensions:
        - ref: region
          expr: {{col: {region}}}
        - ref: channel
        - ref: customer_id
        - ref: ordered_at
      measures:
        - ref: revenue
        - ref: order_count
        - ref: buyers
        - ref: revenue_eu_adj
        - ref: tv_revenue
      metrics:
        - ref: revenue_adj
        - ref: revenue_eu_only
        - ref: revenue_eu_or_zero
        - ref: revenue_eu_bonus
        - ref: avg_order_value_adj
        - ref: buyers_eu_weighted
        - ref: revenue_eu_web_bonus
        - ref: buyers_weighted
        - ref: buyers_inner_doubled
        - ref: buyers_later_condition
        - ref: buyers_drawn
        - ref: buyers_weighted_drawn
        - ref: buyers_eu_twice
        - ref: buyers_eu_thrice
        - ref: revenue_early_adj
        - ref: region_label
        - ref: target
        - ref: eu_revenue
        - ref: tier
        - ref: eu_order_count
        - ref: c1_tv_revenue_adj
        - ref: tv_or_revenue
        - ref: web_revenue_eu_or_zero
        - ref: web_eu_order_count
        - ref: regions_with_revenue
        - ref: peak_region_revenue
        - ref: avg_customer_revenue
        - ref: revenue_early_adj_sum
        - ref: n_other
        - ref: eu_web_regions
        - ref: web_eu_regions
        - ref: revenue_per_region
        - ref: eu_regions
        - ref: regions_gap
        - ref: web_regions
        - ref: web_regions_in_eu
        - ref: regions_eu_bonus
        - ref: big_orders
        - ref: big_eu_orders
    - name: refunds
      extras:
        storage:
          tables: [warehouse.refunds]
      metrics:
        - name: refund_total
          data_type: double
          agg: sum
          expr: {{col: amount}}
"#
    );
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "string"),
                ("customer_id", "string"),
                (region, "string"),
                ("channel", "string"),
                ("amount", "double"),
                ("ordered_at", "date"),
            ],
        )
        .table(
            "warehouse.refunds",
            FileFormat::Iceberg,
            &[("amount", "double")],
        );
    with_semantics(SHOP_SEMANTICS, &datasets, &catalog).await
}

fn first_condition(formula: &PhysicalExpr) -> &PhysicalExpr {
    let Expr::Case { whens, .. } = formula else {
        panic!("a case, got {formula:?}");
    };
    &whens[0].0
}

#[tokio::test]
async fn a_grouped_formula_reads_the_grouping_field_of_the_aggregate_step() {
    let manifest = shop("region").await;

    for metric in ["revenue_adj", "revenue_eu_bonus"] {
        let (plan, _) = planned(&manifest, &select(&["region", metric]));

        let derivation = projection(&plan.root);
        let aggregate = aggregate_step(&derivation.input);
        assert_eq!(aggregate.groupings, [column("region")], "{metric}");
        let [region, formula] = derivation.expressions.as_slice() else {
            panic!("two fields, got {:?}", derivation.expressions);
        };
        assert_eq!(region, &column("region"), "{metric}");
        assert_eq!(
            first_condition(formula),
            &column("region").eq(text("eu")),
            "{metric}"
        );
        assert_eq!(output_names(&plan), ["region", metric]);
    }
}

fn reads(tree: &PhysicalExpr, name: &str) -> bool {
    (tree.pre_order()).any(|node| matches!(node, Expr::Column(column) if column.name() == name))
}

#[tokio::test]
async fn no_expression_above_the_aggregate_step_reads_a_column_of_the_scan() {
    let manifest = shop("region_code").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["region", "revenue_adj"]));

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let row_level = projection(&aggregate.input);
    assert_eq!(
        computed(row_level, &column("region")),
        orders("region_code")
    );
    for tree in &derivation.expressions {
        assert!(!reads(tree, "region_code"), "{tree:?}");
    }
    assert_eq!(output_names(&plan), ["region", "revenue_adj"]);
}

#[tokio::test]
async fn a_constant_formula_over_a_measure_is_one_projection_over_the_final_aggregates() {
    let (plan, _) = planned(&shop("region").await, &select(&["target"]));

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    assert!(aggregate.groupings.is_empty(), "{:?}", aggregate.groupings);
    let [(entry, count)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(count.function().declarations.canonical.name, "count");
    let formula = Expr::Case {
        whens: vec![(
            Expr::Literal(Literal::Boolean(false)),
            column(entry.as_str()),
        )],
        else_: Some(Box::new(Expr::Literal(Literal::BigInt(1000)))),
    };
    assert_eq!(derivation.expressions, [formula]);
    assert_eq!(output_names(&plan), ["target"]);
}

#[tokio::test]
async fn a_measure_reads_a_linked_dimension_per_row_as_a_copy_of_its_tree() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["region", "revenue_eu_adj"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "revenue_eu_adj");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let argument = computed(projection(&aggregate.input), &sum.args()[0]);
    assert_eq!(first_condition(&argument), &orders("region").eq(text("eu")));
    assert_eq!(output_names(&plan), ["region", "revenue_eu_adj"]);
}

#[tokio::test]
async fn a_row_level_metric_with_no_agg_is_a_grouping_of_the_aggregate_step() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["channel", "region_label"]));

    assert_eq!(
        distinct_rows(&plan),
        [
            ("channel", orders("channel")),
            ("region_label", stored(&manifest, "orders", "region_label"))
        ]
    );
    assert_eq!(output_names(&plan), ["channel", "region_label"]);
}

#[tokio::test]
async fn a_row_level_metric_with_agg_is_one_entry_of_the_aggregate_step() {
    let manifest = shop("region").await;
    let refunds = columns_of(&manifest, "refunds");

    let (plan, _) = planned(&manifest, &select(&["refund_total"]).with_from("refunds"));

    let aggregate = aggregate_step(&plan.root);
    assert!(aggregate.groupings.is_empty(), "{:?}", aggregate.groupings);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "refund_total");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        refunds("amount")
    );
    assert_eq!(output_names(&plan), ["refund_total"]);
}

const LABELLED_ORDERS: &str = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          region: region
          revenue: amount
      dimensions:
        - ref: region
        - ref: region_label
          expr:
            case:
              whens:
                - when: {eq: [region, {lit: eu}]}
                  then: {lit: EU}
              else: {lit: other}
      measures:
        - ref: revenue
      metrics:
        - name: eu_revenue
          data_type: double
          expr:
            case:
              whens:
                - when: {eq: [region_label, {lit: EU}]}
                  then: revenue
"#;

#[tokio::test]
async fn a_formula_reads_a_grouped_row_level_name_as_the_field_of_the_aggregate_step() {
    let dimension = with_semantics(LABEL_SEMANTICS, LABELLED_ORDERS, &orders_catalog()).await;

    for manifest in [dimension, shop("region").await] {
        let (plan, _) = planned(&manifest, &select(&["region_label", "eu_revenue"]));

        let derivation = projection(&projection(&plan.root).input);
        let aggregate = aggregate_step(&derivation.input);
        assert_eq!(aggregate.groupings, [column("region_label")]);
        let [formula, region_label] = derivation.expressions.as_slice() else {
            panic!("two fields, got {:?}", derivation.expressions);
        };
        assert_eq!(region_label, &column("region_label"));
        assert_eq!(
            first_condition(formula),
            &column("region_label").eq(text("EU"))
        );
        for tree in &derivation.expressions {
            assert!(!reads(tree, "region"), "{tree:?}");
        }
        assert_eq!(output_names(&plan), ["region_label", "eu_revenue"]);
    }
}

#[tokio::test]
async fn a_grouped_formula_reads_a_row_level_metric_as_one_dimension_through_its_kept_link() {
    let (plan, _) = planned(&shop("region").await, &select(&["region_label", "tier"]));

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("region_label")]);
    let [region_label, tier] = derivation.expressions.as_slice() else {
        panic!("two fields, got {:?}", derivation.expressions);
    };
    assert_eq!(region_label, &column("region_label"));
    assert_eq!(
        first_condition(tier).children()[0],
        &column("region_label").eq(text("EU"))
    );
    assert!(!reads(tier, "region"), "{tier:?}");
    assert_eq!(output_names(&plan), ["region_label", "tier"]);
}

#[tokio::test]
async fn a_predicate_on_a_row_level_metric_with_no_agg_applies_before_aggregation() {
    let manifest = shop("region").await;
    let request = select(&["region_label", "revenue"]);
    let eu = metric("region_label").expect("a name").eq(lit("EU"));

    let (plan, _) = planned(&manifest, &request);
    let (filtered_plan, _) = planned(&manifest, &request.with_predicate(eu));

    let label = stored(&manifest, "orders", "region_label");
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("region_label")]);
    assert_eq!(
        computed(projection(&aggregate.input), &column("region_label")),
        label
    );
    let (filter, row_level) = filtered(&filtered_plan);
    assert_eq!(filter.filter, column("region_label").eq(text("EU")));
    assert_eq!(computed(row_level, &column("region_label")), label);
    let aggregate = aggregate_step(&filtered_plan.root);
    assert_eq!(aggregate.groupings, [column("region_label")]);
    for plan in [&plan, &filtered_plan] {
        assert_eq!(output_names(plan), ["region_label", "revenue"]);
    }
}

fn per_row_term(aggregate: &AggregateNode, row_level: &ProjectNode, metric: &str) -> PhysicalExpr {
    let (_, sum) = (aggregate.aggregates.iter())
        .find(|(entry, _)| entry.as_str() == metric)
        .unwrap_or_else(|| panic!("the entry {metric}, got {:?}", aggregate.aggregates));
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let [argument @ Expr::Column(field)] = sum.args() else {
        panic!("one field, got {:?}", sum.args());
    };
    assert!(field.name().starts_with("__plan_arg"), "{field:?}");
    computed(row_level, argument)
}

#[tokio::test]
async fn a_formula_in_the_pattern_is_one_sum_over_a_reserved_per_row_term() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let amount = orders("amount");
    let results = [
        ("revenue_adj", Some(amount.clone())),
        ("revenue_eu_only", None),
        (
            "revenue_eu_or_zero",
            Some(Expr::Literal(Literal::Double(0.0))),
        ),
    ];

    for (metric, else_) in results {
        let (plan, _) = planned(&manifest, &select(&["channel", metric]));

        let aggregate = aggregate_step(&plan.root);
        assert_eq!(aggregate.groupings, [column("channel")], "{metric}");
        let term = Expr::Case {
            whens: vec![(
                orders("region").eq(text("eu")),
                amount.clone() * Expr::Literal(Literal::Double(1.2)),
            )],
            else_: else_.map(Box::new),
        };
        let row_level = projection(&aggregate.input);
        assert_eq!(per_row_term(aggregate, row_level, metric), term);
        assert_eq!(aggregate_steps(&plan.root), 1, "{metric}");
        assert_eq!(output_names(&plan), ["channel", metric]);
    }
}

#[tokio::test]
async fn only_a_formula_in_the_pattern_pushes_down() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (adjusted, _) = planned(&manifest, &select(&["channel", "revenue_adj"]));
    let (bonus, _) = planned(&manifest, &select(&["channel", "revenue_eu_bonus"]));
    let (grouped, _) = planned(&manifest, &select(&["region", "revenue_eu_bonus"]));

    let aggregate = aggregate_step(&adjusted.root);
    let term = per_row_term(aggregate, projection(&aggregate.input), "revenue_adj");
    assert!(matches!(term, Expr::Case { .. }), "{term:?}");
    assert_eq!(aggregate_steps(&bonus.root), 2);
    let aggregate = aggregate_step(&projection(&grouped.root).input);
    let arguments: Vec<_> = (aggregate.aggregates.iter())
        .map(|(_, sum)| computed(projection(&aggregate.input), &sum.args()[0]))
        .collect();
    assert_eq!(arguments, [orders("amount"), orders("amount")]);
}

#[tokio::test]
async fn a_formula_push_down_reads_a_temporal_dimension_at_its_value_before_bucketing() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let request = at_granularity("ordered_at", Granularity::Month, &["revenue_early_adj"]);

    let (plan, _) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("ordered_at")]);
    let bucketing = projection(&aggregate.input);
    let term = per_row_term(aggregate, projection(&bucketing.input), "revenue_early_adj");
    let early = Expr::Literal(Literal::Date("2026-10-15".into()));
    assert_eq!(first_condition(&term), &orders("ordered_at").lt(early));
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(computed(bucketing, &sum.args()[0]), sum.args()[0]);
    assert_eq!(
        computed(bucketing, &column("ordered_at")),
        call("date_trunc", vec![text("month"), column("ordered_at")])
    );
    assert_eq!(output_names(&plan), ["ordered_at", "revenue_early_adj"]);
}

#[tokio::test]
async fn the_formula_push_downs_sum_is_an_entry_beside_the_other_aggregates_of_the_step() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["channel", "revenue_adj", "buyers"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("channel")]);
    let entries: Vec<_> = (aggregate.aggregates.iter())
        .map(|(entry, call)| {
            let function = call.function().declarations.canonical.name;
            (entry.as_str(), function, call.distinct())
        })
        .collect();
    assert_eq!(
        entries,
        [("revenue_adj", "sum", false), ("buyers", "count", true)]
    );
    let term = per_row_term(aggregate, projection(&aggregate.input), "revenue_adj");
    assert!(matches!(term, Expr::Case { .. }), "{term:?}");
    assert_eq!(aggregate_steps(&plan.root), 1);
    assert_eq!(output_names(&plan), ["channel", "revenue_adj", "buyers"]);
}

#[tokio::test]
async fn a_count_link_is_one_or_zero_per_row_inside_its_term() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["channel", "eu_order_count"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("channel")]);
    let counted = Expr::Case {
        whens: vec![(
            !orders("order_id").is_null(),
            Expr::Literal(Literal::BigInt(1)),
        )],
        else_: Some(Box::new(Expr::Literal(Literal::BigInt(0)))),
    };
    let term = Expr::Case {
        whens: vec![(orders("region").eq(text("eu")), counted)],
        else_: None,
    };
    let row_level = projection(&aggregate.input);
    assert_eq!(per_row_term(aggregate, row_level, "eu_order_count"), term);
    assert_eq!(output_names(&plan), ["channel", "eu_order_count"]);
}

#[tokio::test]
async fn a_formula_push_downs_filter_clause_joins_the_metric_filter_onto_the_filter_its_links_share(
) {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["channel", "c1_tv_revenue_adj"]));

    let aggregate = aggregate_step(&plan.root);
    let row_level = projection(&aggregate.input);
    let amount = orders("amount");
    let term = Expr::Case {
        whens: vec![(
            orders("region").eq(text("eu")),
            amount.clone() * Expr::Literal(Literal::Double(1.2)),
        )],
        else_: Some(Box::new(amount)),
    };
    assert_eq!(
        per_row_term(aggregate, row_level, "c1_tv_revenue_adj"),
        term
    );
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(
        computed(row_level, condition(sum, "c1_tv_revenue_adj")),
        orders("channel").eq(text("tv")) & orders("customer_id").eq(text("c1"))
    );
    assert_eq!(aggregate_steps(&plan.root), 1);
}

#[tokio::test]
async fn differing_filters_and_a_zero_or_a_count_under_a_filter_take_the_parts() {
    let manifest = shop("region").await;

    for metric in [
        "tv_or_revenue",
        "web_revenue_eu_or_zero",
        "web_eu_order_count",
    ] {
        let (plan, _) = planned(&manifest, &select(&["channel", metric]));

        assert_eq!(aggregate_steps(&plan.root), 2, "{metric}");
    }
}

#[tokio::test]
async fn a_formula_push_down_casts_a_row_argument_only_where_the_cast_holds_every_value() {
    let semantics = r#"
  dimensions:
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: wide_quantity, data_type: 'decimal(10,0)', agg: sum, expr: {col: qty}}
    - {name: narrow_quantity, data_type: 'decimal(9,0)', agg: sum, expr: {col: qty}}
  metrics:
    - name: wide_eu
      data_type: decimal(10,0)
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: wide_quantity
    - name: narrow_eu
      data_type: decimal(9,0)
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: narrow_quantity
"#;
    let datasets = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
      dimensions: [{ref: region}]
      measures: [{ref: wide_quantity}, {ref: narrow_quantity}]
      metrics: [{ref: wide_eu}, {ref: narrow_eu}]
"#;
    let catalog = Catalog::default().table(
        "warehouse.orders",
        FileFormat::Iceberg,
        &[("region", "string"), ("qty", "integer")],
    );
    let manifest = with_semantics(semantics, datasets, &catalog).await;
    let orders = columns_of(&manifest, "orders");

    let (wide, _) = planned(&manifest, &select(&["wide_eu"]));
    let (narrow, _) = planned(&manifest, &select(&["narrow_eu"]));

    let decimal = DataType::Decimal {
        precision: 10,
        scale: 0,
    };
    let cast = |input: PhysicalExpr| Expr::Cast {
        input: Box::new(input),
        target: decimal.clone(),
        on_failure: CastFailure::Error,
    };
    let derivation = projection(&wide.root);
    let aggregate = aggregate_step(&derivation.input);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        Expr::Case {
            whens: vec![(orders("region").eq(text("eu")), cast(orders("qty")))],
            else_: None,
        }
    );
    assert_eq!(
        computed(derivation, &column("wide_eu")),
        cast(column(entry.as_str()))
    );
    assert_eq!(aggregate_steps(&narrow.root), 2);
}

#[tokio::test]
async fn a_formula_push_down_moves_only_an_operation_exact_in_its_type_into_the_sum() {
    let semantics = r#"
  dimensions:
    - {name: region, data_type: string, type: categorical}
    - {name: units, data_type: integer, type: categorical}
    - {name: rate, data_type: 'decimal(10,2)', type: categorical}
  measures:
    - {name: quantity, data_type: 'decimal(10,0)', agg: sum, expr: {col: qty}}
    - {name: total, data_type: 'decimal(28,2)', agg: sum, expr: {col: total}}
    - {name: lines, data_type: bigint, agg: count, expr: {col: qty}}
  metrics:
    - name: eu_quantity_by_units
      data_type: decimal(21,0)
      expr: {case: {whens: [{when: {eq: [region, {lit: eu}]}, then: {multiply: [quantity, units]}}]}}
    - name: eu_total_at_rate
      data_type: double
      expr: {case: {whens: [{when: {eq: [region, {lit: eu}]}, then: {multiply: [total, rate]}}]}}
    - name: eu_half_total
      data_type: double
      expr: {case: {whens: [{when: {eq: [region, {lit: eu}]}, then: {divide: [total, {lit: 2}]}}]}}
    - name: eu_double_lines
      data_type: bigint
      expr: {case: {whens: [{when: {eq: [region, {lit: eu}]}, then: {multiply: [lines, {lit: 2}]}}]}}
    - name: eu_negated_lines
      data_type: bigint
      expr: {case: {whens: [{when: {eq: [region, {lit: eu}]}, then: {negate: lines}}]}}
"#;
    let datasets = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
      dimensions: [{ref: region}, {ref: units}, {ref: rate}]
      measures: [{ref: quantity}, {ref: total}, {ref: lines}]
      metrics:
        - {ref: eu_quantity_by_units}
        - {ref: eu_total_at_rate}
        - {ref: eu_half_total}
        - {ref: eu_double_lines}
        - {ref: eu_negated_lines}
"#;
    let catalog = Catalog::default().table(
        "warehouse.orders",
        FileFormat::Iceberg,
        &[
            ("region", "string"),
            ("units", "integer"),
            ("rate", "decimal(10,2)"),
            ("qty", "integer"),
            ("total", "decimal(28,2)"),
        ],
    );
    let manifest = with_semantics(semantics, datasets, &catalog).await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["eu_quantity_by_units"]));

    let aggregate = aggregate_step(&projection(&plan.root).input);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let quantity = Expr::Cast {
        input: Box::new(orders("qty")),
        target: DataType::Decimal {
            precision: 10,
            scale: 0,
        },
        on_failure: CastFailure::Error,
    };
    assert_eq!(
        computed(projection(&aggregate.input), &sum.args()[0]),
        Expr::Case {
            whens: vec![(orders("region").eq(text("eu")), quantity * orders("units"))],
            else_: None,
        }
    );
    for metric in [
        "eu_total_at_rate",
        "eu_half_total",
        "eu_double_lines",
        "eu_negated_lines",
    ] {
        let (parts, _) = planned(&manifest, &select(&[metric]));
        assert_eq!(aggregate_steps(&parts.root), 2, "{metric}");
    }
}

#[tokio::test]
async fn a_parts_subtree_joins_the_frames_subtree_inner_on_is_not_distinct_from() {
    let manifest = shop("region").await;
    let request = select(&["channel", "revenue_eu_bonus", "buyers"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let join = join(&projection(&plan.root).input);
    assert_eq!(join.join_type, JoinType::Inner);
    let trace = &join.common.semantic_meta;
    let cardinality = |subject: &SemAnnotation| {
        matches!(
            subject,
            SemAnnotation::SemanticSource {
                cardinality: Some(_),
                ..
            }
        )
    };
    assert!(!trace.iter().any(cardinality), "{trace:?}");
    let frame = aggregate_step(&join.left);
    assert_eq!(frame.groupings, [column("channel")]);
    assert_eq!(names(&frame.common.output_schema), ["channel", "buyers"]);
    let (step_2, _, _) = parts(&join.right);
    assert_eq!(step_2.groupings, [column("channel")]);
    let [reserved, value] = names(&step_2.common.output_schema)[..] else {
        panic!(
            "a grouping and a value, got {:?}",
            step_2.common.output_schema
        );
    };
    assert!(reserved.starts_with("__plan_"), "{reserved}");
    assert_eq!(value, "revenue_eu_bonus");
    assert_eq!(
        join.on,
        column("channel").is_not_distinct_from(column(reserved))
    );
    assert_eq!(aggregate_steps(&plan.root), 3);
    assert_eq!(
        output_names(&plan),
        ["channel", "revenue_eu_bonus", "buyers"]
    );
}

#[tokio::test]
async fn with_no_grouping_a_parts_subtree_crosses_the_frames_subtree() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["revenue_eu_bonus", "buyers"]));

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let cross = cross(&projection(&plan.root).input);
    let frame = aggregate_step(&cross.left);
    assert!(frame.groupings.is_empty(), "{:?}", frame.groupings);
    assert_eq!(names(&frame.common.output_schema), ["buyers"]);
    let (step_2, _, _) = parts(&cross.right);
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    assert_eq!(names(&step_2.common.output_schema), ["revenue_eu_bonus"]);
    assert_eq!(output_names(&plan), ["revenue_eu_bonus", "buyers"]);
}

#[tokio::test]
async fn a_formula_outside_the_pattern_takes_its_parts_in_two_aggregate_steps() {
    let (plan, key, per_part, entries) = in_total("revenue_eu_bonus").await;

    assert_eq!(aggregate_steps(&plan.root), 2);
    let [("sum", false, eu), ("sum", false, other)] = &entries[..] else {
        panic!("a sum for each result, got {entries:?}");
    };
    let Expr::Case { whens, else_ } = &per_part else {
        panic!("a case, got {per_part:?}");
    };
    let conditions: Vec<_> = whens.iter().map(|(when, _)| when.clone()).collect();
    assert_eq!(conditions, [key.eq(bigint(1))]);
    assert!(whens[0].1.pre_order().any(|node| node == eu), "{whens:?}");
    assert_eq!(else_.as_deref(), Some(other));
    assert_eq!(output_names(&plan), ["revenue_eu_bonus"]);
}

fn entries(step: &AggregateNode) -> Vec<(&'static str, bool, PhysicalExpr)> {
    (step.aggregates.iter())
        .map(|(entry, call)| {
            let function = call.function().declarations.canonical.name;
            (function, call.distinct(), column(entry.as_str()))
        })
        .collect()
}

async fn in_total(
    metric: &str,
) -> (
    SemanticPlan,
    PhysicalExpr,
    PhysicalExpr,
    Vec<(&'static str, bool, PhysicalExpr)>,
) {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let (plan, _) = planned(&manifest, &select(&[metric]));
    assert_eq!(tables_read(&plan), ["orders"]);
    let (step_2, per_part, step_1) = parts(&plan.root);
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let [key] = step_1.groupings.as_slice() else {
        panic!("the part key alone, got {:?}", step_1.groupings);
    };
    assert_eq!(
        computed(projection(&step_1.input), key),
        Expr::Case {
            whens: vec![(orders("region").eq(text("eu")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    let (key, entries) = (key.clone(), entries(step_1));
    (plan, key, per_part, entries)
}

#[tokio::test]
async fn in_total_a_quotient_of_two_measures_takes_its_parts() {
    let (plan, key, per_part, entries) = in_total("avg_order_value_adj").await;

    let [("sum", false, revenue), ("count", false, orders), ("sum", false, other_revenue), ("count", false, other_orders)] =
        &entries[..]
    else {
        panic!("a sum and a count for each result, got {entries:?}");
    };
    let eu = (revenue.clone() / orders.clone()) * Expr::Literal(Literal::Double(1.2));
    let formula = Expr::Case {
        whens: vec![(key.eq(bigint(1)), eu)],
        else_: Some(Box::new(other_revenue.clone() / other_orders.clone())),
    };
    assert_eq!(per_part, formula);
    assert_eq!(output_names(&plan), ["avg_order_value_adj"]);
}

#[tokio::test]
async fn in_total_a_distinct_count_takes_its_parts() {
    let (plan, key, per_part, entries) = in_total("buyers_eu_weighted").await;

    let [("count", true, eu), ("count", true, other)] = &entries[..] else {
        panic!("a distinct count for each result, got {entries:?}");
    };
    let formula = Expr::Case {
        whens: vec![(key.eq(bigint(1)), eu.clone() * bigint(2))],
        else_: Some(Box::new(other.clone())),
    };
    assert_eq!(per_part, formula);
    let value = &plan.root.common().output_schema.columns[0];
    assert_eq!(
        (value.name.as_str(), &value.data_type),
        ("buyers_eu_weighted", &DataType::BigInt)
    );
}

#[tokio::test]
async fn an_inner_case_is_a_key_column_null_where_its_result_is_not_selected() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["revenue_eu_web_bonus"]));

    let (_, per_part, step_1) = parts(&plan.root);
    let [key, inner] = step_1.groupings.as_slice() else {
        panic!("two key columns, got {:?}", step_1.groupings);
    };
    let row_level = projection(&step_1.input);
    let selected = Expr::Case {
        whens: vec![(orders("region").eq(text("eu")), bigint(1))],
        else_: Some(Box::new(bigint(2))),
    };
    assert_eq!(computed(row_level, key), selected);
    assert_eq!(
        computed(row_level, inner),
        Expr::Case {
            whens: vec![(
                selected.eq(bigint(1)),
                Expr::Case {
                    whens: vec![(orders("channel").eq(text("web")), bigint(1))],
                    else_: Some(Box::new(bigint(2))),
                }
            )],
            else_: None,
        }
    );
    let Expr::Case { whens, .. } = &per_part else {
        panic!("a case, got {per_part:?}");
    };
    assert_eq!(whens[0].0, key.clone().eq(bigint(1)));
    let web = inner.clone().eq(bigint(1));
    assert!(whens[0].1.pre_order().any(|node| node == &web), "{whens:?}");
}

fn eu_else_web(orders: impl Fn(&str) -> PhysicalExpr, results: [PhysicalExpr; 3]) -> PhysicalExpr {
    let [eu, web, other] = results;
    Expr::Case {
        whens: vec![
            (orders("region").eq(text("eu")), eu),
            (orders("channel").eq(text("web")), web),
        ],
        else_: Some(Box::new(other)),
    }
}

#[tokio::test]
async fn a_row_level_case_selects_by_position_and_is_no_piece() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["buyers_weighted"]));

    assert_eq!(tables_read(&plan), ["orders"]);
    let (step_2, per_part, step_1) = parts(&plan.root);
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let [key] = step_1.groupings.as_slice() else {
        panic!("one key column, got {:?}", step_1.groupings);
    };
    assert_eq!(
        computed(projection(&step_1.input), key),
        eu_else_web(&orders, [bigint(1), bigint(2), bigint(3)])
    );
    let entries = entries(step_1);
    let [("count", true, buyers)] = &entries[..] else {
        panic!("one distinct count, got {entries:?}");
    };
    let weight = Expr::Case {
        whens: vec![
            (key.clone().eq(bigint(1)), bigint(2)),
            (key.clone().eq(bigint(2)), bigint(2)),
        ],
        else_: Some(Box::new(bigint(1))),
    };
    assert_eq!(per_part, weight * buyers.clone());
}

#[tokio::test]
async fn a_case_inside_a_selecting_condition_selects() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["buyers_inner_doubled"]));

    let (_, per_part, step_1) = parts(&plan.root);
    let [key, inner_key] = step_1.groupings.as_slice() else {
        panic!("two key columns, got {:?}", step_1.groupings);
    };
    let row_level = projection(&step_1.input);
    let inner = eu_else_web(&orders, [bigint(1), bigint(1), bigint(0)]);
    assert_eq!(
        computed(row_level, key),
        Expr::Case {
            whens: vec![(inner.eq(bigint(1)), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    assert_eq!(
        computed(row_level, inner_key),
        eu_else_web(&orders, [bigint(1), bigint(2), bigint(3)])
    );
    let Expr::Case { whens, .. } = &per_part else {
        panic!("a case, got {per_part:?}");
    };
    assert_eq!(whens[0].0, key.clone().eq(bigint(1)));
}

#[tokio::test]
async fn a_case_in_a_later_condition_selects_where_no_earlier_condition_is_true() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["buyers_later_condition"]));

    let (_, _, step_1) = parts(&plan.root);
    let [key, inner_key] = step_1.groupings.as_slice() else {
        panic!("two key columns, got {:?}", step_1.groupings);
    };
    let row_level = projection(&step_1.input);
    let eu = orders("region").eq(text("eu"));
    let web = orders("channel").eq(text("web"));
    let app = orders("channel").eq(text("app"));
    let inner = Expr::Case {
        whens: vec![(web.clone(), bigint(1)), (app.clone(), bigint(1))],
        else_: Some(Box::new(bigint(0))),
    };
    assert_eq!(
        computed(row_level, key),
        Expr::Case {
            whens: vec![(eu.clone(), bigint(1)), (inner.eq(bigint(1)), bigint(2))],
            else_: Some(Box::new(bigint(3))),
        }
    );
    let after_eu = Expr::Case {
        whens: vec![(eu, Expr::Literal(Literal::Boolean(false)))],
        else_: Some(Box::new(Expr::Literal(Literal::Boolean(true)))),
    };
    assert_eq!(
        computed(row_level, inner_key),
        Expr::Case {
            whens: vec![(
                after_eu,
                Expr::Case {
                    whens: vec![(web, bigint(1)), (app, bigint(2))],
                    else_: Some(Box::new(bigint(3))),
                }
            )],
            else_: None,
        }
    );
}

#[tokio::test]
async fn a_volatile_condition_selects_on_each_row() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let random = call("random", vec![]);

    let (plan, _) = planned(&manifest, &select(&["buyers_drawn"]));

    let (_, per_part, step_1) = parts(&plan.root);
    let [drawn, eu] = step_1.groupings.as_slice() else {
        panic!("two key columns, got {:?}", step_1.groupings);
    };
    let row_level = projection(&step_1.input);
    let drawn = computed(row_level, drawn);
    assert!(drawn.pre_order().any(|node| node == &random), "{drawn:?}");
    assert_eq!(
        computed(row_level, eu),
        Expr::Case {
            whens: vec![(orders("region").eq(text("eu")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    assert!(
        !per_part.pre_order().any(|node| node == &random),
        "{per_part:?}"
    );
}

#[tokio::test]
async fn a_volatile_call_beside_a_case_is_computed_on_each_part() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let random = call("random", vec![]);

    let (plan, _) = planned(&manifest, &select(&["buyers_weighted_drawn"]));

    let (_, per_part, step_1) = parts(&plan.root);
    let [key] = step_1.groupings.as_slice() else {
        panic!("one key column, got {:?}", step_1.groupings);
    };
    assert_eq!(
        computed(projection(&step_1.input), key),
        eu_else_web(&orders, [bigint(1), bigint(2), bigint(3)])
    );
    assert!(
        per_part.pre_order().any(|node| node == &random),
        "{per_part:?}"
    );
}

#[tokio::test]
async fn two_metrics_whose_cases_have_one_condition_share_one_subtree() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["buyers_eu_twice", "buyers_eu_thrice"]));

    assert_eq!(tables_read(&plan), ["orders"]);
}

#[tokio::test]
async fn an_agg_over_a_formula_aggregates_its_values_at_the_formula_grain_in_its_own_subtree() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let request = select(&["channel", "regions_with_revenue", "buyers"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let join = join(&projection(&plan.root).input);
    assert_eq!(join.join_type, JoinType::Inner);
    let frame = aggregate_step(&join.left);
    assert_eq!(names(&frame.common.output_schema), ["channel", "buyers"]);
    let (step_2, formula, step_1) = two_steps(&join.right, "count");
    let [channel, region] = step_1.groupings.as_slice() else {
        panic!("the formula grain, got {:?}", step_1.groupings);
    };
    assert_eq!(channel, &column("channel"));
    assert_eq!(
        computed(projection(&step_1.input), region),
        orders("region")
    );
    assert_eq!(first_condition(&formula), &region.clone().eq(text("eu")));
    assert_eq!(step_2.groupings, [column("channel")]);
    let [reserved, "regions_with_revenue"] = names(&step_2.common.output_schema)[..] else {
        panic!("a grouping and the value, got {:?}", step_2.common);
    };
    assert_eq!(
        join.on,
        column("channel").is_not_distinct_from(column(reserved))
    );
    assert_eq!(
        output_names(&plan),
        ["channel", "regions_with_revenue", "buyers"]
    );
}

#[tokio::test]
async fn a_predicate_on_a_metric_of_two_aggregate_steps_alone_sits_above_its_step_2() {
    let manifest = shop("region").await;
    let over = metric("regions_with_revenue").expect("a name").gt(lit(1));
    let request = select(&["channel", "regions_with_revenue", "buyers"]).with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let join = join(&projection(&plan.root).input);
    aggregate_step(&join.left);
    let filter = filter(&join.right);
    assert_eq!(filter.filter, column("regions_with_revenue").gt(integer(1)));
    two_steps(&filter.input, "count");
}

#[tokio::test]
async fn a_formula_that_reads_no_dimension_has_the_demand_as_its_formula_grain() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["avg_customer_revenue"]));

    assert_eq!(tables_read(&plan), ["orders"]);
    let step_2 = aggregate_step(&plan.root);
    let step_1 = aggregate_step(&step_2.input);
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    assert!(step_1.groupings.is_empty(), "{:?}", step_1.groupings);
    let ([(_, avg)], [(revenue, sum)]) = (&step_2.aggregates[..], &step_1.aggregates[..]) else {
        panic!("one entry each, got {step_2:?}");
    };
    assert_eq!(avg.function().declarations.canonical.name, "avg");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    assert_eq!(avg.args(), [column(revenue.as_str())]);
    assert_eq!(output_names(&plan), ["avg_customer_revenue"]);
}

#[tokio::test]
async fn with_no_frames_subtree_the_second_metric_subtree_joins_the_first() {
    let manifest = shop("region").await;
    let request = select(&["channel", "regions_with_revenue", "avg_customer_revenue"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let join = join(&projection(&plan.root).input);
    assert_eq!(join.join_type, JoinType::Inner);
    let (regions, _, regions_step_1) = two_steps(&join.left, "count");
    assert_eq!(regions_step_1.groupings.len(), 2, "{regions_step_1:?}");
    assert_eq!(
        names(&regions.common.output_schema),
        ["channel", "regions_with_revenue"]
    );
    let average = aggregate_step(&join.right);
    assert_eq!(
        average.aggregates[0]
            .1
            .function()
            .declarations
            .canonical
            .name,
        "avg"
    );
    assert_eq!(
        aggregate_step(&average.input).groupings,
        [column("channel")]
    );
    let [reserved, "avg_customer_revenue"] = names(&average.common.output_schema)[..] else {
        panic!("a grouping and the value, got {:?}", average.common);
    };
    assert_eq!(
        join.on,
        column("channel").is_not_distinct_from(column(reserved))
    );
    assert_eq!(
        output_names(&plan),
        ["channel", "regions_with_revenue", "avg_customer_revenue"]
    );
}

#[tokio::test]
async fn two_metrics_whose_step_1_has_one_grouping_and_one_filter_share_one_subtree() {
    let manifest = shop("region").await;
    let request = select(&["channel", "regions_with_revenue", "peak_region_revenue"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders"]);
    let cast = projection(&plan.root);
    let step_2 = aggregate_step(&cast.input);
    assert_eq!(step_2.groupings, [column("channel")]);
    let functions: Vec<_> = (step_2.aggregates.iter())
        .map(|(entry, call)| (entry.as_str(), call.function().declarations.canonical.name))
        .collect();
    assert_eq!(
        functions,
        [
            ("regions_with_revenue", "count"),
            ("peak_region_revenue", "max")
        ]
    );
    assert_eq!(
        aggregate_step(&projection(&step_2.input).input)
            .groupings
            .len(),
        2
    );
    let decimal = DataType::Decimal {
        precision: 18,
        scale: 2,
    };
    let peak = Expr::Cast {
        input: Box::new(column("peak_region_revenue")),
        target: decimal,
        on_failure: CastFailure::Error,
    };
    assert_eq!(
        cast.expressions,
        [column("channel"), column("regions_with_revenue"), peak]
    );
    assert_eq!(output_names(&plan), names(&cast.common.output_schema));
}

#[tokio::test]
async fn the_formula_grain_groups_a_temporal_dimension_at_a_granularity_twice() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");
    let request = at_granularity("ordered_at", Granularity::Month, &["revenue_early_adj_sum"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders"]);
    let (step_2, formula, step_1) = two_steps(&plan.root, "sum");
    assert_eq!(step_2.groupings, [column("ordered_at")]);
    let [bucketed, before @ Expr::Column(reserved)] = step_1.groupings.as_slice() else {
        panic!("ordered_at twice, got {:?}", step_1.groupings);
    };
    assert_eq!(bucketed, &column("ordered_at"));
    assert!(reserved.name().starts_with("__plan_"), "{reserved:?}");
    let bucketing = projection(&step_1.input);
    assert_eq!(
        computed(bucketing, bucketed),
        call("date_trunc", vec![text("month"), column("ordered_at")])
    );
    assert_eq!(
        computed(projection(&bucketing.input), before),
        orders("ordered_at")
    );
    let early = Expr::Literal(Literal::Date("2026-10-15".into()));
    assert_eq!(first_condition(&formula), &before.clone().lt(early));
    assert_eq!(output_names(&plan), ["ordered_at", "revenue_early_adj_sum"]);
}

#[tokio::test]
async fn the_formula_grain_holds_a_row_level_metric_as_one_dimension() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["channel", "n_other"]));

    assert_eq!(tables_read(&plan), ["orders"]);
    let (step_2, formula, step_1) = two_steps(&plan.root, "count");
    assert_eq!(step_2.groupings, [column("channel")]);
    let [channel, label] = step_1.groupings.as_slice() else {
        panic!("channel and region_label, got {:?}", step_1.groupings);
    };
    assert_eq!(channel, &column("channel"));
    assert_eq!(
        computed(projection(&step_1.input), label),
        stored(&manifest, "orders", "region_label")
    );
    assert_eq!(first_condition(&formula), &label.clone().eq(text("other")));
    let value = &plan.root.common().output_schema.columns[1];
    assert_eq!(
        (value.name.as_str(), &value.data_type),
        ("n_other", &DataType::BigInt)
    );
}

#[tokio::test]
async fn two_metrics_that_read_their_grain_in_two_orders_share_one_subtree() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["eu_web_regions", "web_eu_regions"]));

    assert_eq!(tables_read(&plan), ["orders"]);
    let step_2 = aggregate_step(&plan.root);
    assert_eq!(step_2.aggregates.len(), 2, "{step_2:?}");
}

#[tokio::test]
async fn two_metrics_whose_parts_have_one_part_key_share_one_subtree() {
    let manifest = shop("region").await;
    let request = select(&["revenue_eu_bonus", "avg_order_value_adj"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders"]);
    let step_2 = aggregate_step(&plan.root);
    assert_eq!(step_2.aggregates.len(), 2, "{step_2:?}");
    let formula = projection(&step_2.input);
    let step_1 = aggregate_step(&formula.input);
    let [key @ Expr::Column(reserved)] = step_1.groupings.as_slice() else {
        panic!("the part key alone, got {:?}", step_1.groupings);
    };
    assert!(reserved.name().starts_with("__plan_"), "{reserved:?}");
    for (_, call) in &step_2.aggregates {
        let per_part = computed(formula, &call.args()[0]);
        assert_eq!(first_condition(&per_part), &key.clone().eq(bigint(1)));
    }
    assert_eq!(
        output_names(&plan),
        ["revenue_eu_bonus", "avg_order_value_adj"]
    );
}

const DRAWS: &str = r#"
    - name: draws
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          revenue: amount
      dimensions:
        - name: d1
          data_type: double
          type: categorical
          expr: {random: []}
        - name: d2
          data_type: double
          type: categorical
          expr: {random: []}
      measures:
        - ref: revenue
      metrics:
        - name: d1_above_d2
          data_type: bigint
          agg: count
          expr:
            case:
              whens:
                - when: {gt: [d1, d2]}
                  then: revenue
        - name: d1_above_half
          data_type: bigint
          agg: count
          expr:
            case:
              whens:
                - when: {gt: [d1, {lit: 0.5}]}
                  then: revenue
        - name: d2_above_half
          data_type: bigint
          agg: count
          expr:
            case:
              whens:
                - when: {gt: [d2, {lit: 0.5}]}
                  then: revenue
        - name: revenue_by_draws
          data_type: double
          expr:
            multiply: [{multiply: [revenue, d1]}, d2]
        - name: revenue_by_d1
          data_type: double
          expr:
            multiply: [revenue, d1]
        - name: revenue_by_d2
          data_type: double
          expr:
            multiply: [revenue, d2]
        - name: revenue_by_d1_less_d2
          data_type: double
          expr:
            subtract: [{multiply: [revenue, d1]}, d2]
        - name: revenue_by_d2_less_d1
          data_type: double
          expr:
            subtract: [{multiply: [revenue, d2]}, d1]
"#;

async fn draws() -> SemanticManifest {
    with_datasets(&format!("{ORDERS}{DRAWS}"), &orders_catalog()).await
}

#[tokio::test]
async fn the_formula_grain_reads_two_dimensions_with_one_tree_by_their_own_fields() {
    let (plan, _) = planned(&draws().await, &select(&["d1_above_d2"]));

    let (_, formula, step_1) = two_steps(&plan.root, "count");
    let [d1, d2] = ["d1", "d2"].map(|name| column(&format!("__plan_unbucketed_{name}")));
    assert_eq!(step_1.groupings, [d1.clone(), d2.clone()]);
    assert_eq!(first_condition(&formula), &d1.gt(d2));
}

#[tokio::test]
async fn two_metrics_over_two_dimensions_with_one_tree_share_no_subtree() {
    let (plan, _) = planned(&draws().await, &select(&["d1_above_half", "d2_above_half"]));

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
}

#[tokio::test]
async fn two_pieces_with_one_tree_are_two_columns_of_the_part_key() {
    let (plan, _) = planned(&draws().await, &select(&["revenue_by_draws"]));

    let (_, per_part, step_1) = parts(&plan.root);
    let [d1, d2] = step_1.groupings.as_slice() else {
        panic!("a piece for each Dimension, got {:?}", step_1.groupings);
    };
    let [("sum", false, revenue)] = &entries(step_1)[..] else {
        panic!("revenue's sum, got {step_1:?}");
    };
    assert_eq!(per_part, (revenue.clone() * d1.clone()) * d2.clone());
}

#[tokio::test]
async fn two_metrics_whose_pieces_read_two_dimensions_with_one_tree_share_no_subtree() {
    let (plan, _) = planned(&draws().await, &select(&["revenue_by_d1", "revenue_by_d2"]));

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
}

#[tokio::test]
async fn two_metrics_whose_pieces_read_two_dimensions_in_two_orders_read_each_by_its_link() {
    let request = select(&["revenue_by_d1_less_d2", "revenue_by_d2_less_d1"]);

    let (plan, _) = planned(&draws().await, &request);

    assert_eq!(tables_read(&plan), ["orders"]);
    let step_2 = aggregate_step(&plan.root);
    let formula = projection(&step_2.input);
    let step_1 = aggregate_step(&formula.input);
    let [d1, d2] = step_1.groupings.as_slice() else {
        panic!("a piece for each Dimension, got {:?}", step_1.groupings);
    };
    let pieces: Vec<Vec<_>> = (step_2.aggregates.iter())
        .map(|(_, call)| {
            let per_part = computed(formula, &call.args()[0]);
            (per_part.pre_order())
                .filter(|node| step_1.groupings.contains(node))
                .cloned()
                .collect()
        })
        .collect();
    assert_eq!(pieces, [[d1.clone(), d2.clone()], [d2.clone(), d1.clone()]]);
}

fn over_the_join(plan: &SemanticPlan) -> &ProjectNode {
    projection(&projection(&plan.root).input)
}

fn linked_field(name: &str, metric: &str) -> bool {
    name.starts_with("__plan_") && name.ends_with(&format!("_{metric}"))
}

#[tokio::test]
async fn a_formula_reads_a_linked_metric_of_two_aggregate_steps_as_its_step_2_field() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["channel", "revenue_per_region"]));

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let derivation = over_the_join(&plan);
    let join = join(&derivation.input);
    assert_eq!(join.join_type, JoinType::Inner);
    let frame = aggregate_step(&join.left);
    let [(revenue, sum)] = &frame.aggregates[..] else {
        panic!("revenue's sum, got {frame:?}");
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let (step_2, _, step_1) = two_steps(&join.right, "count");
    assert_eq!(step_1.groupings.len(), 2, "{step_1:?}");
    let [reserved, linked] = names(&step_2.common.output_schema)[..] else {
        panic!("a grouping and the value, got {:?}", step_2.common);
    };
    assert!(linked_field(linked, "regions_with_revenue"), "{linked}");
    assert_eq!(
        join.on,
        column("channel").is_not_distinct_from(column(reserved))
    );
    let formula = computed(derivation, &column("revenue_per_region"));
    assert!(
        reads(&formula, revenue.as_str()) && reads(&formula, linked),
        "{formula:?}"
    );
    assert!(of_the_row(&formula), "{formula:?}");
    assert_eq!(output_names(&plan), ["channel", "revenue_per_region"]);
}

#[tokio::test]
async fn a_linked_metric_beside_its_own_selection_reads_one_value() {
    let manifest = shop("region").await;
    let request = select(&["channel", "regions_with_revenue", "revenue_per_region"]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let join = join(&over_the_join(&plan).input);
    let both = projection(&join.right);
    two_steps(&both.input, "count");
    let [_, "regions_with_revenue", linked] = names(&both.common.output_schema)[..] else {
        panic!(
            "a grouping, the value and its linked field, got {:?}",
            both.common
        );
    };
    assert!(linked_field(linked, "regions_with_revenue"), "{linked}");
    assert_eq!(
        computed(both, &column(linked)),
        column("regions_with_revenue")
    );
}

fn of_the_row(tree: &PhysicalExpr) -> bool {
    !(tree.pre_order()).any(|node| matches!(node, Expr::Function(Call::Aggregate(..))))
}

fn narrowed_to_eu(manifest: &SemanticManifest, node: &PlanNode) {
    let orders = columns_of(manifest, "orders");
    let (_, _, step_1) = two_steps(node, "count");
    let row_level = projection(&step_1.input);
    assert!(
        matches!(*row_level.input, PlanNode::Scan(_)),
        "{row_level:?}"
    );
    for (_, call) in &step_1.aggregates {
        let condition = condition(call, "regions_with_revenue");
        assert_eq!(
            computed(row_level, condition),
            orders("region").eq(text("eu"))
        );
    }
}

#[tokio::test]
async fn a_readers_filter_narrows_the_step_1_aggregates_of_its_linked_subtree() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["channel", "eu_regions"]));

    assert_eq!(tables_read(&plan), ["orders"]);
    let derivation = over_the_join(&plan);
    narrowed_to_eu(&manifest, &derivation.input);
    let (step_2, _, step_1) = two_steps(&derivation.input, "count");
    assert_eq!(step_1.groupings.len(), 2, "{step_1:?}");
    let ["channel", linked] = names(&step_2.common.output_schema)[..] else {
        panic!("channel and the value, got {:?}", step_2.common);
    };
    assert!(linked_field(linked, "regions_with_revenue"), "{linked}");
    assert_eq!(computed(derivation, &column("eu_regions")), column(linked));
    assert_eq!(output_names(&plan), ["channel", "eu_regions"]);
}

#[tokio::test]
async fn two_linked_subtrees_of_one_metric_write_two_reserved_names() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["channel", "regions_gap"]));

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let derivation = over_the_join(&plan);
    let join = join(&derivation.input);
    let (all, _, all_step_1) = two_steps(&join.left, "count");
    assert!(all_step_1
        .aggregates
        .iter()
        .all(|(_, call)| call.filter().is_none()));
    narrowed_to_eu(&manifest, &join.right);
    let ["channel", left] = names(&all.common.output_schema)[..] else {
        panic!("channel and the value, got {:?}", all.common);
    };
    let eu = aggregate_step(&join.right);
    let [reserved, right] = names(&eu.common.output_schema)[..] else {
        panic!("a grouping and the value, got {:?}", eu.common);
    };
    assert!(
        linked_field(left, "regions_with_revenue") && linked_field(right, "regions_with_revenue")
    );
    assert_ne!(left, right);
    assert_eq!(
        join.on,
        column("channel").is_not_distinct_from(column(reserved))
    );
    assert_eq!(
        computed(derivation, &column("regions_gap")),
        column(left) - column(right)
    );
}

fn joined_above_step_1(plan: &SemanticPlan, grouped_by: &PhysicalExpr) -> PhysicalExpr {
    assert_eq!(tables_read(plan), ["orders", "orders"]);
    let step_2 = aggregate_step(&plan.root);
    let [(_, call)] = step_2.aggregates.as_slice() else {
        panic!("one entry, got {:?}", step_2.aggregates);
    };
    assert_eq!(call.function().declarations.canonical.name, "sum");
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let formula = projection(&step_2.input);
    let join = join(&formula.input);
    assert_eq!(join.join_type, JoinType::Inner);
    let step_1 = aggregate_step(&join.left);
    assert_eq!(step_1.groupings, std::slice::from_ref(grouped_by));
    let (linked, _, linked_step_1) = two_steps(&join.right, "count");
    assert_eq!(linked_step_1.groupings.len(), 2, "{linked_step_1:?}");
    let [reserved, value] = names(&linked.common.output_schema)[..] else {
        panic!("a grouping and the value, got {:?}", linked.common);
    };
    assert!(linked_field(value, "regions_with_revenue"), "{value}");
    assert_eq!(
        join.on,
        grouped_by.clone().is_not_distinct_from(column(reserved))
    );
    let per_group = computed(formula, &call.args()[0]);
    assert!(
        reads(&per_group, value) && of_the_row(&per_group),
        "{per_group:?}"
    );
    per_group
}

#[tokio::test]
async fn a_linked_metric_of_an_agg_over_a_formula_joins_directly_above_its_step_1() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["web_regions"]));

    joined_above_step_1(&plan, &column("__plan_unbucketed_channel"));
    assert_eq!(output_names(&plan), ["web_regions"]);
}

#[tokio::test]
async fn a_readers_filter_narrows_each_linked_subtree_of_its_linked_metric() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["web_regions_in_eu"]));

    let step_2 = aggregate_step(&over_the_join(&plan).input);
    let join = join(&projection(&step_2.input).input);
    narrowed_to_eu(&manifest, &join.right);
    assert_eq!(output_names(&plan), ["web_regions_in_eu"]);
}

#[tokio::test]
async fn a_linked_metric_of_the_parts_joins_directly_above_its_step_1() {
    let manifest = shop("region").await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["regions_eu_bonus"]));

    let key = column("__plan_key1_regions_eu_bonus");
    let per_part = joined_above_step_1(&plan, &key);
    let step_1 = aggregate_step(&join(&projection(&aggregate_step(&plan.root).input).input).left);
    assert_eq!(
        computed(projection(&step_1.input), &key),
        Expr::Case {
            whens: vec![(orders("region").eq(text("eu")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    let Expr::Case { whens, .. } = &per_part else {
        panic!("a case, got {per_part:?}");
    };
    assert_eq!(whens[0].0, key.eq(bigint(1)));
    assert_eq!(output_names(&plan), ["regions_eu_bonus"]);
}

#[tokio::test]
async fn a_formula_that_links_a_metric_of_two_aggregate_steps_has_its_stored_type() {
    let manifest = shop("region").await;

    let (plan, _) = planned(&manifest, &select(&["big_eu_orders"]));

    assert_eq!(tables_read(&plan), ["orders", "orders"]);
    let step_2 = aggregate_step(&plan.root);
    let [(_, call)] = step_2.aggregates.as_slice() else {
        panic!("one entry, got {:?}", step_2.aggregates);
    };
    assert_eq!(call.function().declarations.canonical.name, "sum");
    assert_eq!(output_names(&plan), ["big_eu_orders"]);
}

const CUSTOMER_SEMANTICS: &str = r#"
  measures:
    - name: revenue
      data_type: double
      agg: sum
      expr: {col: amount}
  metrics:
    - name: avg_cust
      data_type: double
      agg: avg
      expr:
        case:
          whens:
            - when: {not: {is_null: {col: customer_id}}}
              then: revenue
"#;

const CUSTOMER_ORDERS: &str = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
      measures:
        - ref: revenue
      metrics:
        - ref: avg_cust
"#;

#[tokio::test]
async fn the_formula_grain_holds_a_column_under_auto_under_a_reserved_name() {
    let manifest = with_semantics(CUSTOMER_SEMANTICS, CUSTOMER_ORDERS, &orders_catalog()).await;
    let orders = columns_of(&manifest, "orders");
    let known = !orders("customer_id").is_null();
    assert_eq!(
        first_condition(&stored(&manifest, "orders", "avg_cust")),
        &known
    );

    let (plan, _) = planned(&manifest, &select(&["avg_cust"]));

    assert_eq!(tables_read(&plan), ["orders"]);
    let (step_2, formula, step_1) = two_steps(&plan.root, "avg");
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let [customer @ Expr::Column(reserved)] = step_1.groupings.as_slice() else {
        panic!("the column alone, got {:?}", step_1.groupings);
    };
    assert!(reserved.name().starts_with("__plan_"), "{reserved:?}");
    assert_eq!(
        computed(projection(&step_1.input), customer),
        orders("customer_id")
    );
    assert_eq!(first_condition(&formula), &!customer.clone().is_null());
    assert_eq!(output_names(&plan), ["avg_cust"]);
}

const RATES: &str = r#"
    - name: rates
      extras:
        storage:
          tables: [warehouse.orders]
        semantic_mapping:
          revenue: amount
      dimensions:
        - name: rate
          data_type: double
          type: categorical
          expr: {lit: 2.0}
        - name: doubled_rate
          data_type: double
          type: categorical
          expr:
            multiply: [rate, {lit: 2.0}]
      measures:
        - ref: revenue
      metrics:
        - name: revenue_at_rate
          data_type: double
          expr:
            multiply: [revenue, rate]
"#;

#[tokio::test]
async fn a_dimension_that_only_a_selected_dimension_reads_is_not_grouped() {
    let manifest = with_datasets(&format!("{ORDERS}{RATES}"), &orders_catalog()).await;
    let orders = columns_of(&manifest, "orders");

    let (plan, _) = planned(&manifest, &select(&["doubled_rate", "revenue_at_rate"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("doubled_rate")]);
    let row_level = projection(&aggregate.input);
    assert_eq!(
        per_row_term(aggregate, row_level, "revenue_at_rate"),
        orders("amount") * Expr::Literal(Literal::Double(2.0))
    );
    assert_eq!(output_names(&plan), ["doubled_rate", "revenue_at_rate"]);
}

async fn a_formula_that_is_not_numeric_over_the_parts() -> Golden {
    let (error, advisories) = refusal(&shop("region").await, &select(&["tier"]));

    let PlanError::UngroupedMetricFormula { metric, name, .. } = &error.kind else {
        panic!("UngroupedMetricFormula, got {error:?}");
    };
    assert_eq!((metric.as_str(), name.as_str()), ("tier", "region_label"));
    let path = error.kind.path().map(ToString::to_string);
    assert_eq!(path.as_deref(), Some("orders"));
    assert!(advisories.is_empty());
    Golden::message("ungrouped", &error)
}

#[tokio::test]
async fn a_formula_that_is_not_numeric_and_takes_the_parts_is_ungrouped_metric_formula() {
    a_formula_that_is_not_numeric_over_the_parts()
        .await
        .assert_committed();
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async { vec![a_formula_that_is_not_numeric_over_the_parts().await] }).await;
}
