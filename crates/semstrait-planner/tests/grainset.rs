mod common;

use common::golden::{self, Golden};
use common::{
    advisory_kinds, aggregate_filters, aggregate_steps, alias, answered_by_source, at_granularity,
    call, column, columns_of, compiled, computed, cross, fields, filter, first_aggregate,
    grain_not_servable, integer, join, lossy, names, output_names, planned, projection, qualified,
    read_table, refusal, select, signature_of, stored, table_below, tables_read, text, ties,
    txn_by_grain, without_signature, written, Catalog, Replaced, TRANSACTIONS, TXN_MONTHLY_AGG,
    TXN_WEEKLY_AGG,
};
use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::DataKindName;
use semstrait_common::Granularity;
use semstrait_common::{Diagnose, Diagnostic, Tree};
use semstrait_ir::PhysicalExpr;
use semstrait_ir::{
    AggregateNode, Cardinality, FilterNode, GranularityGroup, JoinType, PlanNode, SemAnnotation,
    SemanticPlan,
};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit, measure};
use semstrait_planner::{Combine, Decision, PlanError, Planner, Request, RequestContext};
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, Literal};

async fn three_children() -> SemanticManifest {
    txn_by_grain(&[TRANSACTIONS, TXN_MONTHLY_AGG, TXN_WEEKLY_AGG]).await
}

fn reads_one_child(plan: &SemanticPlan, child: &str, table: &str, names: &[&str]) {
    let scans = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Scan(_)));
    assert_eq!(scans.count(), 1);
    assert_eq!(read_table(plan), table);
    let combined = |node: &PlanNode| matches!(node, PlanNode::Join(_) | PlanNode::Cross(_));
    assert!(!plan.root.pre_order().any(combined));
    let answered = DataKindName::new(child);
    let grainset = DataKindName::new("txn_by_grain");
    let trace = &plan.root.common().semantic_meta;
    assert!(
        trace.iter().any(|subject| matches!(
            subject,
            SemAnnotation::SemanticSource { semantic_source, answered_by: Some(name), .. }
                if *semantic_source == grainset && *name == answered
        )),
        "{answered:?} in {trace:?}"
    );
    assert_eq!(output_names(plan), names);
}

fn drill_across(plan: &SemanticPlan) -> &PlanNode {
    let mut combined = (plan.root.pre_order())
        .filter(|node| matches!(node, PlanNode::Join(_) | PlanNode::Cross(_)));
    let (Some(node), None) = (combined.next(), combined.next()) else {
        panic!("one join or cross in {plan:?}");
    };
    let below: usize = node.children().into_iter().map(aggregate_steps).sum();
    assert_eq!(aggregate_steps(&plan.root), below);
    assert!((node.common().semantic_meta).iter().any(|subject| matches!(
        subject,
        SemAnnotation::SemanticSource {
            cardinality: Some(Cardinality::OneToOne),
            ..
        }
    )));
    node
}

fn own_aggregate<'p>(side: &'p PlanNode, child: &str) -> &'p AggregateNode {
    assert_eq!(table_below(side), child);
    let mut aggregates = (side.pre_order()).filter_map(|node| match node {
        PlanNode::Aggregate(aggregate) => Some(aggregate),
        _ => None,
    });
    let (Some(aggregate), None) = (aggregates.next(), aggregates.next()) else {
        panic!("one aggregate step in {side:?}");
    };
    aggregate
}

fn buckets_alone(manifest: &SemanticManifest, side: &PlanNode, child: &str, grain: &str) {
    let aggregate = own_aggregate(side, child);
    let child = alias(manifest, &format!("txn_by_grain.{child}"));
    let bucket = call(
        "date_trunc",
        vec![text(grain), qualified(&child, "event_date")],
    );
    let event_date = format!("{child}.event_date");
    let bucketing = fields(projection(&aggregate.input));
    assert!(
        bucketing.contains(&(event_date.as_str(), bucket)),
        "{bucketing:?}"
    );
}

fn drills_across_month_and_day(manifest: &SemanticManifest, plan: &SemanticPlan, grain: &str) {
    let join = join(drill_across(plan));
    let monthly = alias(manifest, "txn_by_grain.txn_monthly_agg");
    let daily = alias(manifest, "txn_by_grain.transactions");
    let on = |name| qualified(&monthly, name).is_not_distinct_from(qualified(&daily, name));
    assert_eq!(join.join_type, JoinType::Full);
    assert_eq!(join.on, on("event_date") & on("country"));
    assert_eq!(join.post_join_filter, None);
    buckets_alone(manifest, &join.left, "txn_monthly_agg", grain);
    buckets_alone(manifest, &join.right, "transactions", grain);
    let names = ["event_date", "country", "quantity", "revenue_target"];
    assert_eq!(output_names(plan), names);
}

#[tokio::test]
async fn each_name_goes_to_its_rank_first_child_and_a_child_first_for_every_name_answers_alone() {
    let manifest = three_children().await;
    let names = ["quantity", "revenue"];
    let at_month = at_granularity("event_date", Granularity::Month, &names);
    let at_week = at_granularity("event_date", Granularity::Week, &names);

    let (split, _) = planned(&manifest, &at_month);
    let (alone, _) = planned(&manifest, &at_week);

    let join = join(drill_across(&split));
    assert_eq!(table_below(&join.left), "txn_monthly_agg");
    assert_eq!(table_below(&join.right), "transactions");
    assert_eq!(output_names(&split), ["event_date", "quantity", "revenue"]);
    reads_one_child(
        &alone,
        "txn_by_grain.txn_weekly_agg",
        "txn_weekly_agg",
        &["event_date", "quantity", "revenue"],
    );
}

#[tokio::test]
async fn the_weekly_child_answers_at_week_and_never_at_month() {
    let manifest = three_children().await;

    for (granularity, child) in [
        (Granularity::Month, "transactions"),
        (Granularity::Week, "txn_weekly_agg"),
    ] {
        let request = at_granularity("event_date", granularity, &["quantity"]);

        let (plan, _) = planned(&manifest, &request);

        let answered = format!("txn_by_grain.{child}");
        reads_one_child(&plan, &answered, child, &["event_date", "quantity"]);
    }
}

#[tokio::test]
async fn a_grainset_reads_the_coarsest_child_that_serves_the_granularity_and_else_the_finest() {
    let manifest = txn_by_grain(&[TRANSACTIONS, TXN_MONTHLY_AGG]).await;

    for (request, child) in [
        (
            at_granularity("event_date", Granularity::Year, &["revenue"]),
            "txn_monthly_agg",
        ),
        (select(&["event_date", "revenue"]), "transactions"),
    ] {
        let (plan, _) = planned(&manifest, &request);

        let answered = format!("txn_by_grain.{child}");
        reads_one_child(&plan, &answered, child, &["event_date", "revenue"]);
    }
}

const ADS: &str = r#"
semantic_model:
  name: ads
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign, data_type: string, type: categorical}
    - {name: account, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
  grainsets:
    - name: ads
      dimensions: [{ref: date}, {ref: campaign}, {ref: account}]
      measures: [{ref: clicks}]
      filters:
        - name: recent
          expr:
            gt_eq: [date, {lit: {date: "2024-01-02"}}]
      datasets:
        - name: daily
          extras:
            storage: {tables: [warehouse.daily]}
            semantic_mapping: {date: date, campaign: campaign, account: account, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
        - name: by_campaign_total
          extras:
            storage: {tables: [warehouse.by_campaign_total]}
            semantic_mapping: {campaign: campaign, clicks: clicks}
        - name: undated
          extras:
            storage: {tables: [warehouse.undated]}
            semantic_mapping: {date: date, campaign: campaign, clicks: clicks}
"#;

async fn mixed_time_axis() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.daily",
            FileFormat::Iceberg,
            &[
                ("date", "date"),
                ("campaign", "string"),
                ("account", "string"),
                ("clicks", "long"),
            ],
        )
        .table(
            "warehouse.by_campaign_total",
            FileFormat::Iceberg,
            &[("campaign", "string"), ("clicks", "long")],
        )
        .table(
            "warehouse.undated",
            FileFormat::Iceberg,
            &[("date", "date"), ("campaign", "string"), ("clicks", "long")],
        );
    compiled(ADS, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_child_with_no_time_axis_answers_a_demand_that_reads_none() {
    let manifest = mixed_time_axis().await;

    let (plan, _) = planned(&manifest, &select(&["campaign", "clicks"]));

    answered_by_source(&plan, "ads.by_campaign_total");
}

#[tokio::test]
async fn a_child_with_no_time_axis_never_answers_a_demand_that_reads_the_time_axis() {
    let manifest = mixed_time_axis().await;
    let day = Literal::Date("2024-01-01".into());
    let since = dim("date").expect("a name").gt_eq(Expr::Literal(day));
    let rest = ["campaign", "clicks"];

    for request in [
        select(&["date", "campaign", "clicks"]),
        at_granularity("date", Granularity::Month, &rest),
        select(&rest).with_predicate(since),
        select(&rest).with_filters(vec!["recent".into()]),
    ] {
        let (plan, _) = planned(&manifest, &request);

        answered_by_source(&plan, "ads.daily");
    }
}

const TOTALS: &str = r#"
semantic_model:
  name: totals
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, year]
    - {name: campaign, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
  grainsets:
    - name: totals
      dimensions: [{ref: date}, {ref: campaign}, {ref: region}]
      measures: [{ref: clicks}]
      datasets:
        - name: a_yearly
          extras:
            storage: {tables: [warehouse.a_yearly]}
            semantic_mapping: {date: date, campaign: campaign, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: year}
        - name: b_untimed
          extras:
            storage: {tables: [warehouse.b_untimed]}
            semantic_mapping: {campaign: campaign, region: region, clicks: clicks}
"#;

#[tokio::test]
async fn a_child_with_no_time_axis_outranks_a_year_child_for_a_demand_that_reads_no_time_axis() {
    let catalog = Catalog::default()
        .table(
            "warehouse.a_yearly",
            FileFormat::Iceberg,
            &[("date", "date"), ("campaign", "string"), ("clicks", "long")],
        )
        .table(
            "warehouse.b_untimed",
            FileFormat::Iceberg,
            &[
                ("campaign", "string"),
                ("region", "string"),
                ("clicks", "long"),
            ],
        );
    let manifest = compiled(TOTALS, &catalog, &InMemoryFileSystem::new()).await;

    let (plan, advisories) = planned(&manifest, &select(&["campaign", "clicks"]));

    answered_by_source(&plan, "totals.b_untimed");
    assert!(advisory_kinds(advisories).is_empty());
}

#[tokio::test]
async fn a_drill_across_at_month_joins_the_daily_child_to_the_monthly_one() {
    let manifest = three_children().await;
    let rest = ["country", "quantity", "revenue_target"];
    let request = at_granularity("event_date", Granularity::Month, &rest);

    let (plan, _) = planned(&manifest, &request);

    drills_across_month_and_day(&manifest, &plan, "month");
}

#[tokio::test]
async fn a_drill_across_at_year_buckets_each_child_to_year() {
    let manifest = three_children().await;
    let rest = ["country", "quantity", "revenue_target"];
    let request = at_granularity("event_date", Granularity::Year, &rest);

    let (plan, _) = planned(&manifest, &request);

    drills_across_month_and_day(&manifest, &plan, "year");
}

#[tokio::test]
async fn a_selected_time_axis_with_no_granularity_fixes_month_first() {
    let manifest = three_children().await;
    let request = select(&["event_date", "country", "quantity", "revenue_target"]);

    let (plan, _) = planned(&manifest, &request);

    drills_across_month_and_day(&manifest, &plan, "month");
}

fn filters(plan: &SemanticPlan) -> Vec<&FilterNode> {
    (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Filter(filter) => Some(filter),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_predicate_on_a_later_childs_measure_sits_above_the_joins() {
    let manifest = three_children().await;
    let rest = ["country", "quantity", "revenue_target"];
    let over = measure("quantity").expect("a name").gt(lit(10));
    let request = at_granularity("event_date", Granularity::Month, &rest).with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let daily = alias(&manifest, "txn_by_grain.transactions");
    let [above] = filters(&plan)[..] else {
        panic!("one filter in {plan:?}");
    };
    assert_eq!(above.filter, qualified(&daily, "quantity").gt(integer(10)));
    assert_eq!(*above.input, *drill_across(&plan));
    drills_across_month_and_day(&manifest, &plan, "month");
}

#[tokio::test]
async fn a_predicate_on_the_first_childs_measure_sits_above_the_joins() {
    let manifest = three_children().await;
    let rest = ["country", "quantity", "revenue_target"];
    let over = measure("revenue_target").expect("a name").gt(lit(1000));
    let request = at_granularity("event_date", Granularity::Month, &rest).with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let monthly = alias(&manifest, "txn_by_grain.txn_monthly_agg");
    let [above] = filters(&plan)[..] else {
        panic!("one filter in {plan:?}");
    };
    assert_eq!(
        above.filter,
        qualified(&monthly, "revenue_target").gt(integer(1000))
    );
    assert_eq!(*above.input, *drill_across(&plan));
    drills_across_month_and_day(&manifest, &plan, "month");
}

#[tokio::test]
async fn a_measure_filtered_but_not_selected_is_given_by_the_child_that_serves_it() {
    let manifest = three_children().await;
    let over = measure("quantity").expect("a name").gt(lit(10));
    let rest = ["country", "revenue_target"];
    let request = at_granularity("event_date", Granularity::Month, &rest).with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let daily = alias(&manifest, "txn_by_grain.transactions");
    let [above] = filters(&plan)[..] else {
        panic!("one filter in {plan:?}");
    };
    assert_eq!(above.filter, qualified(&daily, "quantity").gt(integer(10)));
    let join = join(&above.input);
    assert_eq!(table_below(&join.right), "transactions");
    let quantity = format!("{daily}.quantity");
    let given = names(&join.right.common().output_schema);
    assert!(given.contains(&quantity.as_str()), "{given:?}");
    assert_eq!(
        output_names(&plan),
        ["event_date", "country", "revenue_target"]
    );
}

#[tokio::test]
async fn a_predicate_on_a_dimension_goes_into_each_chosen_child_before_bucketing() {
    let manifest = three_children().await;
    let rest = ["country", "quantity", "revenue_target"];
    let day = Literal::Date("2026-03-15".into());
    let since = dim("event_date")
        .expect("a name")
        .gt_eq(Expr::Literal(day.clone()));
    let request = at_granularity("event_date", Granularity::Month, &rest).with_predicate(since);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(filters(&plan).len(), 2);
    let join = join(drill_across(&plan));
    for (side, child) in [
        (&join.left, "txn_monthly_agg"),
        (&join.right, "transactions"),
    ] {
        let child_alias = alias(&manifest, &format!("txn_by_grain.{child}"));
        let bucketing = projection(&own_aggregate(side, child).input);
        let below = filter(&bucketing.input);
        let event_date = qualified(&child_alias, "event_date");
        assert_eq!(below.filter, event_date.gt_eq(Expr::Literal(day.clone())));
    }
    drills_across_month_and_day(&manifest, &plan, "month");
}

#[tokio::test]
async fn a_non_additive_measure_in_a_drill_across_is_reaggregated_in_the_child_that_maps_it() {
    let manifest = three_children().await;
    let rest = ["country", "avg_qty", "revenue_target"];
    let requested = at_granularity("event_date", Granularity::Month, &rest);
    let fixed = select(&["event_date", "country", "avg_qty", "revenue_target"]);

    for request in [requested, fixed] {
        let (plan, _) = planned(&manifest, &request);

        let join = join(drill_across(&plan));
        buckets_alone(&manifest, &join.right, "transactions", "month");
    }
}

const SALES: &str = r#"
semantic_model:
  name: sales
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: region, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: bigint, agg: sum}
    - {name: target, data_type: bigint, agg: sum}
  metrics:
    - name: web_revenue
      data_type: bigint
      expr: revenue
      filter: {eq: [channel, {lit: web}]}
  grainsets:
    - name: sales
      dimensions:
        - {ref: date, is_key: true}
        - {ref: region, is_key: true}
        - {ref: channel}
      measures: [{ref: revenue}, {ref: target}]
      metrics: [{ref: web_revenue}]
      datasets:
        - name: sales_monthly
          extras:
            storage: {tables: [warehouse.sales_monthly]}
            semantic_mapping: {date: month_start, region: region, revenue: amount, target: target}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
        - name: sales_daily
          extras:
            storage: {tables: [warehouse.sales_daily]}
            semantic_mapping: {date: day, region: region, channel: channel, revenue: amount}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

async fn sales() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.sales_monthly",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("region", "string"),
                ("amount", "long"),
                ("target", "long"),
            ],
        )
        .table(
            "warehouse.sales_daily",
            FileFormat::Iceberg,
            &[
                ("day", "date"),
                ("region", "string"),
                ("channel", "string"),
                ("amount", "long"),
            ],
        );
    compiled(SALES, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_grainsets_metric_is_computed_in_the_child_that_has_its_signature() {
    let manifest = sales().await;
    assert!(signature_of(&manifest, "sales.sales_monthly", "web_revenue").is_none());
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["region", "web_revenue", "target"],
    );

    let (plan, _) = planned(&manifest, &request);

    assert!(filters(&plan).is_empty(), "{plan:?}");
    let join = join(drill_across(&plan));
    let (monthly, daily) = (
        alias(&manifest, "sales.sales_monthly"),
        alias(&manifest, "sales.sales_daily"),
    );
    let web_revenue = |child: &str| format!("{child}.web_revenue");
    let left = names(&join.left.common().output_schema);
    assert_eq!(table_below(&join.left), "sales_monthly");
    assert!(!left.contains(&web_revenue(&monthly).as_str()), "{left:?}");
    let right = names(&join.right.common().output_schema);
    assert!(right.contains(&web_revenue(&daily).as_str()), "{right:?}");
    let aggregate = own_aggregate(&join.right, "sales_daily");
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let condition = sum.filter().expect("the Metric's filter");
    let row_level = projection(&projection(&aggregate.input).input);
    let channel = columns_of(&manifest, "sales_daily")("channel");
    assert_eq!(computed(row_level, condition), channel.eq(text("web")));
    assert_eq!(
        output_names(&plan),
        ["date", "region", "web_revenue", "target"]
    );
}

const WEB_SALES: &str = r#"
semantic_model:
  name: web_sales
  dimensions:
    - name: ordered_at
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: channel, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: bigint, agg: sum}
  grainsets:
    - name: sales
      dimensions:
        - {ref: ordered_at, is_key: true}
        - {ref: channel, KEY}
      measures: [{ref: revenue}]
      filters:
        - name: web
          expr:
            eq: [channel, {lit: web}]
      datasets:
        - name: d
          extras:
            storage: {tables: [warehouse.d]}
            semantic_mapping: {ordered_at: day, channel: channel, revenue: amount}
            temporal: {timeseries: {occurred_at: ordered_at}, granularity: day}
        - name: m
          extras:
            storage: {tables: [warehouse.m]}
            semantic_mapping: {ordered_at: month_start, MONTHLY revenue: amount}
            temporal: {timeseries: {occurred_at: ordered_at}, granularity: month}
"#;

async fn web_sales(monthly_maps_channel: bool) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.d",
            FileFormat::Iceberg,
            &[("day", "date"), ("channel", "string"), ("amount", "long")],
        )
        .table(
            "warehouse.m",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("channel", "string"),
                ("amount", "long"),
            ],
        );
    let (key, monthly) = match monthly_maps_channel {
        true => ("is_key: true", "channel: channel,"),
        false => ("is_key: false", ""),
    };
    let yaml = WEB_SALES.replace("KEY", key).replace("MONTHLY", monthly);
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_named_filter_goes_into_the_child_that_answers_alone() {
    let manifest = web_sales(true).await;
    let request = at_granularity("ordered_at", Granularity::Month, &["revenue"])
        .with_filters(vec!["web".into()]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["m"]);
    answered_by_source(&plan, "sales.m");
    assert_eq!(filters(&plan).len(), 1, "{plan:?}");
    let aggregate = own_aggregate(&plan.root, "m");
    let below = filter(&projection(&aggregate.input).input);
    assert_eq!(below.filter, column("web"));
    let web = stored(&manifest, "sales.m", "web");
    assert_eq!(web, columns_of(&manifest, "m")("channel").eq(text("web")));
    assert_eq!(written(projection(&below.input), "web"), Some(web));
}

#[tokio::test]
async fn a_named_filter_goes_to_the_child_that_maps_its_dimension() {
    let manifest = web_sales(false).await;
    assert!(signature_of(&manifest, "sales.m", "web").is_none());
    let request = at_granularity("ordered_at", Granularity::Month, &["revenue"])
        .with_filters(vec!["web".into()]);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["d"]);
    answered_by_source(&plan, "sales.d");
    assert_eq!(filters(&plan).len(), 1, "{plan:?}");
}

const BUDGETS: &str = r#"
semantic_model:
  name: budgets
  dimensions:
    - name: valid_from
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - name: valid_to
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - name: campaign
      data_type: string
      type: categorical
  measures:
    - name: spend
      data_type: bigint
      agg: sum
    - name: budget
      data_type: bigint
      agg: sum
  grainsets:
    - name: budgets
      dimensions:
        - ref: valid_from
          is_key: true
        - ref: valid_to
          is_key: true
        - ref: campaign
          is_key: true
      measures:
        - ref: spend
        - ref: budget
      datasets:
        - name: budgets_daily
          extras:
            storage:
              tables: [warehouse.budgets_daily]
            semantic_mapping:
              valid_from: valid_from
              valid_to: valid_to
              campaign: campaign_id
              spend: spend
            temporal:
              scd:
                scd_type: type2
                valid_from: valid_from
                valid_to: valid_to
              granularity: day
        - name: budgets_monthly
          extras:
            storage:
              tables: [warehouse.budgets_monthly]
            semantic_mapping:
              valid_from: valid_from
              valid_to: valid_to
              campaign: campaign_id
              budget: amount
            temporal:
              scd:
                scd_type: type2
                valid_from: valid_from
                valid_to: valid_to
              granularity: month
"#;

#[tokio::test]
async fn a_fixed_granularity_buckets_both_dimensions_of_an_scd_time_axis_in_each_child() {
    let columns = |measure| {
        [
            ("valid_from", "date"),
            ("valid_to", "date"),
            ("campaign_id", "string"),
            (measure, "long"),
        ]
    };
    let catalog = Catalog::default()
        .table(
            "warehouse.budgets_daily",
            FileFormat::Iceberg,
            &columns("spend"),
        )
        .table(
            "warehouse.budgets_monthly",
            FileFormat::Iceberg,
            &columns("amount"),
        );
    let manifest = compiled(BUDGETS, &catalog, &InMemoryFileSystem::new()).await;
    let request = select(&["valid_from", "valid_to", "campaign", "spend", "budget"]);

    let (plan, _) = planned(&manifest, &request);

    let join = join(drill_across(&plan));
    for (side, child) in [
        (&join.left, "budgets_monthly"),
        (&join.right, "budgets_daily"),
    ] {
        let child_alias = alias(&manifest, &format!("budgets.{child}"));
        let bucketing = fields(projection(&own_aggregate(side, child).input));
        for name in ["valid_from", "valid_to"] {
            let bucket = call(
                "date_trunc",
                vec![text("month"), qualified(&child_alias, name)],
            );
            let field = format!("{child_alias}.{name}");
            assert!(
                bucketing.contains(&(field.as_str(), bucket)),
                "{child}: {bucketing:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_drill_across_with_no_grouping_crosses_one_row_from_each_child() {
    let manifest = three_children().await;

    let (plan, _) = planned(&manifest, &select(&["quantity", "revenue_target"]));

    let cross = cross(drill_across(&plan));
    for (side, child) in [
        (&cross.left, "txn_monthly_agg"),
        (&cross.right, "txn_weekly_agg"),
    ] {
        assert!(own_aggregate(side, child).groupings.is_empty(), "{child}");
    }
    assert_eq!(output_names(&plan), ["quantity", "revenue_target"]);
}

#[tokio::test]
async fn a_public_grainset_ranks_at_the_granularity_of_what_answers() {
    let manifest = three_children().await;
    let all = ["event_date", "country", "quantity", "revenue_target"];

    for (request, group, granularity) in [
        (
            at_granularity("event_date", Granularity::Week, &["quantity"]),
            GranularityGroup::Requested,
            Granularity::Week,
        ),
        (select(&all), GranularityGroup::Selected, Granularity::Month),
    ] {
        let (plan, _) = planned(&manifest, &request);

        let granularity = Some(granularity);
        let trace = &plan.root.common().semantic_meta;
        assert!(
            trace.iter().any(|subject| matches!(
                subject,
                SemAnnotation::Rank { granularity_fit, granularity: found, .. }
                    if *granularity_fit == group && *found == granularity
            )),
            "{group:?}/{granularity:?} in {trace:?}"
        );
    }
}

const SPEND: &str = r#"
semantic_model:
  name: spend
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - name: campaign
      data_type: string
      type: categorical
  measures:
    - name: clicks
      data_type: bigint
      agg: sum
    - name: budget
      data_type: bigint
      agg: sum
  grainsets:
    - name: spend
      dimensions:
        - ref: date
        - ref: campaign
      measures:
        - ref: clicks
        - ref: budget
      datasets:
        - name: spend_daily
          extras:
            storage:
              tables: [warehouse.spend_daily]
            semantic_mapping:
              date: day
              campaign: campaign_id
              clicks: click_count
            temporal:
              timeseries:
                occurred_at: date
              granularity: day
        - name: spend_monthly
          extras:
            storage:
              tables: [warehouse.spend_monthly]
            semantic_mapping:
              date: month_start
              campaign: campaign_id
              budget: amount
            temporal:
              timeseries:
                occurred_at: date
              granularity: month
"#;

const DAILY_CLICKS: [(&str, &str); 3] = [
    ("day", "date"),
    ("campaign_id", "string"),
    ("click_count", "long"),
];

fn spend_catalog() -> Catalog {
    Catalog::default()
        .table("warehouse.spend_daily", FileFormat::Iceberg, &DAILY_CLICKS)
        .table(
            "warehouse.spend_monthly",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("campaign_id", "string"),
                ("amount", "long"),
            ],
        )
}

async fn spend() -> SemanticManifest {
    compiled(SPEND, &spend_catalog(), &InMemoryFileSystem::new()).await
}

const ALL_CLICKS: &str = r#"
    - name: all_clicks
      mode: all
      dimensions:
        - ref: date
        - ref: campaign
      measures:
        - ref: clicks
      datasets:
        - name: web_clicks
          extras:
            storage:
              tables: [warehouse.web_clicks]
            semantic_mapping: {date: day, campaign: campaign_id, clicks: click_count}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
        - name: app_clicks
          extras:
            storage:
              tables: [warehouse.app_clicks]
            semantic_mapping: {date: day, campaign: campaign_id, clicks: click_count}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

async fn spend_beside_all_clicks() -> SemanticManifest {
    let catalog = spend_catalog()
        .table("warehouse.web_clicks", FileFormat::Iceberg, &DAILY_CLICKS)
        .table("warehouse.app_clicks", FileFormat::Iceberg, &DAILY_CLICKS);
    let yaml = format!("{SPEND}  unionsets:\n{ALL_CLICKS}");
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_unionset_that_exposes_fewer_semantics_outranks_a_grainset_at_the_same_fit() {
    let manifest = spend_beside_all_clicks().await;
    let request = at_granularity("date", Granularity::Month, &["campaign", "clicks"]);

    let (field_first, advisories) = planned(&manifest, &request);
    let (from, _) = planned(&manifest, &request.with_from("spend"));

    for (plan, root) in [(&field_first, "all_clicks"), (&from, "spend")] {
        let trace = &plan.root.common().semantic_meta;
        let rank = SemAnnotation::Rank {
            scope_root: DataKindName::new(root),
            compositions: 1,
            granularity_fit: GranularityGroup::Requested,
            granularity: Some(Granularity::Day),
        };
        assert!(trace.contains(&rank), "{rank:?} in {trace:?}");
    }
    let tables = tables_read(&field_first);
    assert!(
        tables.iter().all(|table| !table.starts_with("spend_")),
        "{tables:?}"
    );
    let named = (field_first.root.pre_order())
        .flat_map(|node| &node.common().semantic_meta)
        .flat_map(|annotation| match annotation {
            SemAnnotation::Rank { scope_root, .. } => vec![scope_root.as_str()],
            SemAnnotation::SemanticSource {
                semantic_source,
                answered_by,
                pruned,
                ..
            } => {
                let mut names = vec![semantic_source.as_str()];
                names.extend(answered_by.as_ref().map(DataKindName::as_str));
                names.extend(pruned.iter().map(DataKindName::as_str));
                names
            }
            _ => Vec::new(),
        });
    for name in named {
        assert!(name != "spend" && !name.starts_with("spend."), "{name}");
    }
    assert!(advisories.is_empty());
}

fn no_candidate(
    manifest: &SemanticManifest,
    request: &Request,
    grainset: &str,
) -> Diagnostic<PlanError> {
    let (error, advisories) = refusal(manifest, request);
    assert_eq!(
        error.kind,
        PlanError::NoCandidate {
            nearest: vec![(grainset.into(), Vec::new())],
        }
    );
    assert!(advisories.is_empty());
    error
}

#[tokio::test]
async fn children_that_share_no_guaranteed_keyset_are_not_a_candidate() {
    let manifest = spend().await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "clicks", "budget"],
    );

    no_candidate(&manifest, &request, "spend");
    let from = request.with_from("spend");
    grain_not_servable(
        &manifest,
        &from,
        &["spend"],
        (Some(Granularity::Month), None),
        &[
            ("spend.spend_daily", &["budget"]),
            ("spend.spend_monthly", &["clicks"]),
        ],
    );
}

async fn target_at_day() -> Vec<Golden> {
    let manifest = three_children().await;
    let request = at_granularity(
        "event_date",
        Granularity::Day,
        &["quantity", "revenue_target"],
    );
    let field_first = no_candidate(&manifest, &request, "txn_by_grain");
    let from = request.with_from("txn_by_grain");
    let named = grain_not_servable(
        &manifest,
        &from,
        &["txn_by_grain"],
        (Some(Granularity::Day), None),
        &[
            ("txn_by_grain.txn_monthly_agg", &["quantity"]),
            ("txn_by_grain.txn_weekly_agg", &["revenue_target"]),
            ("txn_by_grain.transactions", &["revenue_target"]),
        ],
    );
    vec![
        Golden::message("target_at_day", &field_first),
        Golden::message("target_at_day_from", &named),
    ]
}

#[tokio::test]
async fn a_name_that_only_the_monthly_child_maps_serves_no_day() {
    for golden in target_at_day().await {
        golden.assert_committed();
    }
}

#[tokio::test]
async fn a_dimension_that_the_child_of_a_measure_does_not_map_is_no_candidate() {
    let manifest = three_children().await;
    let request = select(&["txn_type", "quantity", "revenue_target"]);

    no_candidate(&manifest, &request, "txn_by_grain");
}

const SALES_FLAT: &str = r#"
  datasets:
    - name: sales_flat
      extras:
        storage: {tables: [warehouse.sales_flat]}
        semantic_mapping: {date: month_start, region: region, channel: channel, revenue: amount, target: target}
        temporal: {timeseries: {occurred_at: date}, granularity: month}
      dimensions: [{ref: date}, {ref: region}, {ref: channel}]
      measures: [{ref: revenue}, {ref: target}]
      metrics: [{ref: web_revenue}]
"#;

async fn sales_beside_sales_flat() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.sales_monthly",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("region", "string"),
                ("amount", "long"),
                ("target", "long"),
            ],
        )
        .table(
            "warehouse.sales_daily",
            FileFormat::Iceberg,
            &[
                ("day", "date"),
                ("region", "string"),
                ("channel", "string"),
                ("amount", "long"),
            ],
        )
        .table(
            "warehouse.sales_flat",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("region", "string"),
                ("channel", "string"),
                ("amount", "long"),
                ("target", "long"),
            ],
        );
    let yaml = format!("{SALES}{SALES_FLAT}");
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

fn web_revenue_broken(mut manifest: SemanticManifest) -> SemanticManifest {
    without_signature(&mut manifest, "sales.sales_daily", "web_revenue");
    manifest
}

fn broken_web_revenue(manifest: &SemanticManifest, request: &Request) {
    let (error, advisories) = refusal(manifest, request);
    let PlanError::BrokenTree {
        path,
        semantic_source,
        semantics,
    } = &error.kind
    else {
        panic!("BrokenTree, got {error:?}");
    };
    assert_eq!(path.to_string(), "sales");
    assert_eq!(semantic_source.as_str(), "sales.sales_daily");
    assert_eq!(semantics.as_str(), "web_revenue");
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_route_over_a_childs_broken_signature_is_a_broken_tree() {
    let manifest = web_revenue_broken(sales().await);
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["region", "web_revenue", "target"],
    );

    broken_web_revenue(&manifest, &request);
    broken_web_revenue(&manifest, &request.with_from("sales"));
}

#[tokio::test]
async fn a_broken_grainset_beside_a_dataset_that_serves_the_request_leaves_the_dataset_answering() {
    let intact = sales_beside_sales_flat().await;
    let broken = web_revenue_broken(intact.clone());
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["region", "web_revenue", "target"],
    );

    let (plan, advisories) = planned(&broken, &request);

    assert_eq!(tables_read(&plan), ["sales_flat"]);
    let (expected, expected_advisories) = planned(&intact, &request);
    assert_eq!(plan, expected);
    assert_eq!(
        advisory_kinds(advisories),
        advisory_kinds(expected_advisories)
    );
}

#[tokio::test]
async fn a_broken_candidate_is_named_where_no_candidate_answers() {
    let manifest = web_revenue_broken(sales_beside_sales_flat().await);
    let request = at_granularity(
        "date",
        Granularity::Day,
        &["region", "web_revenue", "target"],
    );

    broken_web_revenue(&manifest, &request);
}

fn report_child(name: &str, mapped: &str, grain: &str) -> String {
    format!(
        r#"
        - name: {name}
          extras:
            storage:
              tables: [warehouse.{name}]
            semantic_mapping: {{date: day, campaign: campaign_id{mapped}}}
            temporal:
              timeseries:
                occurred_at: date
              granularity: {grain}
"#
    )
}

const REPORTS: &str = r#"
semantic_model:
  name: reports
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign, data_type: string, type: categorical}
    @SOURCE_DIMENSION@
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    @SPEND_AND_VIEWS@
  grainsets:
    - name: reports
      dimensions:
        - {ref: date, is_key: true}
        - {ref: campaign, is_key: true}
        @SOURCE_REF@
      measures: [@MEASURES@]
      datasets:
"#;

type Columns<'a> = &'a [(&'a str, &'a str)];

fn warehouse(tables: &[(&str, Columns)]) -> Catalog {
    let base = [("day", "date"), ("campaign_id", "string")];
    tables
        .iter()
        .fold(Catalog::default(), |catalog, (name, extra)| {
            let columns = [&base[..], extra].concat();
            catalog.table(&format!("warehouse.{name}"), FileFormat::Iceberg, &columns)
        })
}

async fn reports(leaning: bool, children: &[String]) -> SemanticManifest {
    let counted: Columns = &[("clicks", "long"), ("spend", "long"), ("views", "long")];
    let catalog = warehouse(&[
        ("a_child", counted),
        ("b_child", counted),
        ("lean", &[("clicks", "long")]),
        ("wide", &[("clicks", "long"), ("source", "string")]),
        ("monthly", &[("clicks", "long"), ("source", "string")]),
    ]);
    let [dimension, measures, source_ref, measure_refs] = if leaning {
        [
            "- {name: source, data_type: string, type: categorical}",
            "",
            "- {ref: source}",
            "{ref: clicks}",
        ]
    } else {
        [
            "",
            "- {name: spend, data_type: bigint, agg: sum}\n    - {name: views, data_type: bigint, agg: sum}",
            "",
            "{ref: clicks}, {ref: spend}, {ref: views}",
        ]
    };
    let model = REPORTS
        .replace("@SOURCE_DIMENSION@", dimension)
        .replace("@SPEND_AND_VIEWS@", measures)
        .replace("@SOURCE_REF@", source_ref)
        .replace("@MEASURES@", measure_refs);
    let yaml = format!("{model}{}", children.concat());
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_replaced_grainset_strategy_decides_which_child_a_tie_names() {
    let manifest = reports(false, &a_and_b("spend", "views")).await;
    let decision = Decision {
        stand_in: Some("reports.b_child".into()),
        operands: Vec::new(),
        combine: Combine::PassThrough,
        physical_read: None,
    };
    let planner = Planner::new().with_grainset_strategy(Box::new(Replaced(decision)));
    let request = select(&["campaign", "clicks"]);

    let planned = planner.plan(&manifest, &request, &RequestContext::default());

    let Ok((plan, advisories)) = planned else {
        panic!("{request:?} is refused: {planned:?}");
    };
    answered_by_source(&plan, "reports.b_child");
    assert_eq!(ties(advisories), []);
}

fn a_and_b(a_measure: &str, b_measure: &str) -> [String; 2] {
    [
        report_child(
            "a_child",
            &format!(", clicks: clicks, {a_measure}: {a_measure}"),
            "day",
        ),
        report_child(
            "b_child",
            &format!(", clicks: clicks, {b_measure}: {b_measure}"),
            "day",
        ),
    ]
}

#[tokio::test]
async fn children_at_one_grain_with_another_mask_tie_on_the_child_rank() {
    let manifest = reports(false, &a_and_b("spend", "views")).await;
    let expected = ("reports", "reports.a_child", vec!["reports.b_child"]);
    let expected = [(
        expected.0.into(),
        expected.1.into(),
        expected.2.into_iter().map(Into::into).collect(),
    )];

    let (alone, alone_advisories) = planned(&manifest, &select(&["campaign", "clicks"]));
    let (split, split_advisories) = planned(&manifest, &select(&["campaign", "clicks", "views"]));

    answered_by_source(&alone, "reports.a_child");
    assert_eq!(ties(alone_advisories), expected);
    let join = join(drill_across(&split));
    assert_eq!(table_below(&join.left), "a_child");
    assert_eq!(table_below(&join.right), "b_child");
    assert_eq!(ties(split_advisories), expected);
}

#[tokio::test]
async fn a_pick_that_the_child_rank_decides_raises_no_ambiguity() {
    let manifest = reports(false, &a_and_b("spend", "views")).await;

    for (measure, child) in [("spend", "a_child"), ("views", "b_child")] {
        let (plan, advisories) = planned(&manifest, &select(&["campaign", measure]));

        answered_by_source(&plan, &format!("reports.{child}"));
        assert!(advisory_kinds(advisories).is_empty(), "{measure}");
    }
}

fn leaning() -> [String; 3] {
    [
        report_child("lean", ", clicks: clicks", "day"),
        report_child("wide", ", clicks: clicks, source: source", "day"),
        report_child("monthly", ", clicks: clicks, source: source", "month"),
    ]
}

#[tokio::test]
async fn a_child_with_fewer_dimensions_answers_before_a_closer_granularity_fit() {
    let manifest = reports(true, &leaning()).await;
    let request = at_granularity("date", Granularity::Month, &["campaign", "clicks"]);

    let (plan, advisories) = planned(&manifest, &request);

    answered_by_source(&plan, "reports.lean");
    assert!(advisory_kinds(advisories).is_empty());
}

#[tokio::test]
async fn a_request_whose_grain_equals_a_childs_grain_reads_that_child() {
    let manifest = reports(true, &leaning()).await;

    for (rest, child) in [
        (&["campaign", "clicks"][..], "lean"),
        (&["campaign", "source", "clicks"][..], "wide"),
    ] {
        let request = at_granularity("date", Granularity::Day, rest);

        let (plan, advisories) = planned(&manifest, &request);

        answered_by_source(&plan, &format!("reports.{child}"));
        assert!(advisory_kinds(advisories).is_empty());
    }
}

#[tokio::test]
async fn the_first_child_of_the_left_deep_join_at_a_tie_is_the_child_first_in_name_order() {
    for (a_measure, b_measure) in [("spend", "views"), ("views", "spend")] {
        let manifest = reports(false, &a_and_b(a_measure, b_measure)).await;
        let rest = ["campaign", "spend", "views"];
        let request = at_granularity("date", Granularity::Day, &rest);

        let (plan, advisories) = planned(&manifest, &request);

        let join = join(drill_across(&plan));
        assert_eq!(join.join_type, JoinType::Full);
        assert_eq!(table_below(&join.left), "a_child", "{a_measure}");
        assert_eq!(table_below(&join.right), "b_child", "{a_measure}");
        assert!(advisory_kinds(advisories).is_empty());
    }
}

#[tokio::test]
async fn a_pick_that_the_granularity_fit_decides_raises_no_ambiguity() {
    let manifest = reports(true, &leaning()).await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "source", "clicks"],
    );

    let (plan, advisories) = planned(&manifest, &request);

    answered_by_source(&plan, "reports.monthly");
    assert!(advisory_kinds(advisories).is_empty());
}

struct CampaignChild<'a> {
    name: &'a str,
    names: &'a [&'a str],
    grain: &'a str,
}

fn child<'a>(name: &'a str, names: &'a [&'a str], grain: &'a str) -> CampaignChild<'a> {
    CampaignChild { name, names, grain }
}

async fn campaign_reports(children: &[CampaignChild<'_>]) -> SemanticManifest {
    let mapped = |names: &[&str]| -> String {
        names
            .iter()
            .map(|name| format!(", {name}: {name}"))
            .collect()
    };
    let mut names: Vec<&str> = children
        .iter()
        .flat_map(|child| child.names)
        .copied()
        .collect();
    names.sort_unstable();
    names.dedup();
    let (dimensions, measures): (Vec<&str>, Vec<&str>) =
        names.iter().partition(|name| **name == "source");
    let declared_dimension = if dimensions.is_empty() {
        String::new()
    } else {
        "\n    - {name: source, data_type: string, type: categorical}".to_owned()
    };
    let declared_measures: String = measures
        .iter()
        .map(|name| format!("\n    - {{name: {name}, data_type: bigint, agg: sum}}"))
        .collect();
    let source_ref = if dimensions.is_empty() {
        ""
    } else {
        ", {ref: source}"
    };
    let measure_refs = measures
        .iter()
        .map(|name| format!("{{ref: {name}}}"))
        .collect::<Vec<_>>()
        .join(", ");
    let datasets: String = children
        .iter()
        .map(|child| report_child(child.name, &mapped(child.names), child.grain))
        .collect();
    let yaml = format!(
        r#"
semantic_model:
  name: campaigns
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {{name: campaign, data_type: string, type: categorical}}{declared_dimension}
  measures:{declared_measures}
  grainsets:
    - name: campaigns
      dimensions: [{{ref: date, is_key: true}}, {{ref: campaign, is_key: true}}{source_ref}]
      measures: [{measure_refs}]
      datasets:
{datasets}"#
    );
    let columns: Vec<Vec<(&str, &str)>> = children
        .iter()
        .map(|child| {
            (child.names.iter())
                .map(|name| (*name, if *name == "source" { "string" } else { "long" }))
                .collect()
        })
        .collect();
    let tables: Vec<(&str, Columns)> = (children.iter())
        .zip(&columns)
        .map(|(child, columns)| (child.name, columns.as_slice()))
        .collect();
    let catalog = warehouse(&tables);
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn the_drill_across_joins_the_best_fit_child_first_even_where_the_pick_would_rank_another_first(
) {
    let manifest = campaign_reports(&[
        child("lean", &["spend"], "day"),
        child("wide", &["views", "source"], "month"),
    ])
    .await;
    let rest = ["campaign", "spend", "views"];
    let request = at_granularity("date", Granularity::Month, &rest);

    let (plan, advisories) = planned(&manifest, &request);

    let join = join(drill_across(&plan));
    assert_eq!(table_below(&join.left), "wide");
    assert_eq!(table_below(&join.right), "lean");
    assert!(advisory_kinds(advisories).is_empty());
}

#[tokio::test]
async fn a_later_child_compares_its_keys_with_the_coalesce_of_the_earlier_childrens_keys() {
    let manifest = campaign_reports(&[
        child("a_child", &["spend"], "day"),
        child("b_child", &["views"], "day"),
        child("c_child", &["clicks"], "day"),
    ])
    .await;
    let rest = ["campaign", "spend", "views", "clicks"];
    let request = at_granularity("date", Granularity::Day, &rest);

    let (plan, advisories) = planned(&manifest, &request);

    let joins: Vec<_> = (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .collect();
    let [later, first] = joins[..] else {
        panic!("two joins in {plan:?}");
    };
    let [a, b, c] = ["a_child", "b_child", "c_child"]
        .map(|child| alias(&manifest, &format!("campaigns.{child}")));
    let first_on = |name| qualified(&a, name).is_not_distinct_from(qualified(&b, name));
    let earlier = |name| call("coalesce", vec![qualified(&a, name), qualified(&b, name)]);
    let later_on = |name| earlier(name).is_not_distinct_from(qualified(&c, name));
    assert_eq!(first.join_type, JoinType::Full);
    assert_eq!(first.on, first_on("date") & first_on("campaign"));
    assert_eq!(later.join_type, JoinType::Full);
    assert_eq!(later.on, later_on("date") & later_on("campaign"));
    assert_eq!(join(&later.left), first);
    assert!(advisory_kinds(advisories).is_empty());
}

#[tokio::test]
async fn two_split_measures_whose_picks_each_tie_report_both_ties() {
    let manifest = campaign_reports(&[
        child("p1", &["spend", "clicks"], "day"),
        child("p2", &["spend", "impressions"], "day"),
        child("q1", &["views", "clicks"], "day"),
        child("q2", &["views", "impressions"], "day"),
    ])
    .await;
    let request = at_granularity("date", Granularity::Day, &["campaign", "spend", "views"]);

    let (plan, advisories) = planned(&manifest, &request);

    let join = join(drill_across(&plan));
    assert_eq!(table_below(&join.left), "p1");
    assert_eq!(table_below(&join.right), "q1");
    let tie = |chosen: &str, tied: &str| {
        (
            "campaigns".to_owned(),
            format!("campaigns.{chosen}"),
            vec![format!("campaigns.{tied}")],
        )
    };
    assert_eq!(ties(advisories), [tie("p1", "p2"), tie("q1", "q2")]);
}

const TIED_BRANCH: &str = r#"
semantic_model:
  name: tied_branch
  dimensions:
    - {name: campaign, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
    - {name: source, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
  unionsets:
    - name: tied_branch
      mode: all
      dimensions: [{ref: campaign}, {ref: region}, {ref: source}]
      measures: [{ref: clicks}]
      grainsets:
        - name: inner
          datasets:
            - name: a_child
              extras:
                storage: {tables: [warehouse.a_child]}
                semantic_mapping: {campaign: campaign, region: region, clicks: clicks}
            - name: b_child
              extras:
                storage: {tables: [warehouse.b_child]}
                semantic_mapping: {campaign: campaign, source: source, clicks: clicks}
      datasets:
        - name: outer
          extras:
            storage: {tables: [warehouse.outer]}
            semantic_mapping: {campaign: campaign, region: region, source: source, clicks: clicks}
"#;

#[tokio::test]
async fn a_grainset_branch_whose_children_tie_reports_the_tie_at_its_path_in_the_unionset() {
    let catalog = Catalog::default()
        .table(
            "warehouse.a_child",
            FileFormat::Iceberg,
            &[
                ("campaign", "string"),
                ("region", "string"),
                ("clicks", "long"),
            ],
        )
        .table(
            "warehouse.b_child",
            FileFormat::Iceberg,
            &[
                ("campaign", "string"),
                ("source", "string"),
                ("clicks", "long"),
            ],
        )
        .table(
            "warehouse.outer",
            FileFormat::Iceberg,
            &[
                ("campaign", "string"),
                ("region", "string"),
                ("source", "string"),
                ("clicks", "long"),
            ],
        );
    let manifest = compiled(TIED_BRANCH, &catalog, &InMemoryFileSystem::new()).await;

    let (_, advisories) = planned(&manifest, &select(&["campaign", "clicks"]));

    let expected = (
        "tied_branch / tied_branch.inner".to_owned(),
        "tied_branch.inner.a_child".to_owned(),
        vec!["tied_branch.inner.b_child".to_owned()],
    );
    assert_eq!(ties(advisories), [expected]);
}

#[tokio::test]
async fn the_advice_for_a_tie_of_grainset_children_says_to_make_them_differ_in_the_model() {
    let manifest = reports(false, &a_and_b("spend", "views")).await;

    let (_, advisories) = planned(&manifest, &select(&["campaign", "clicks"]));

    let messages: Vec<_> = (advisories.into_iter())
        .map(|advisory| advisory.kind.message())
        .collect();
    assert_eq!(
        messages,
        ["at `reports`: `reports.a_child` is chosen by name order over `reports.b_child`: make them differ in the model"]
    );
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(target_at_day()).await;
}

const SPEND_METRICS: &str = r#"
semantic_model:
  name: spend
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
    - {name: avg_clicks, data_type: double, agg: avg}
  metrics:
    - {name: web_budget, data_type: bigint, expr: budget, filter: {eq: [channel, {lit: web}]}}
    - {name: web_cpc2, data_type: double, expr: {divide: [{cast: {input: web_budget, target: double}}, clicks]}}
    - name: web_cpc
      data_type: double
      expr: {divide: [{cast: {input: budget, target: double}}, clicks]}
      filter: {eq: [channel, {lit: web}]}
    - {name: cpc_avg, data_type: double, agg: avg, expr: {divide: [{cast: {input: budget, target: double}}, clicks]}}
    - {name: budget_per_avg_click, data_type: double, expr: {divide: [{cast: {input: budget, target: double}}, avg_clicks]}}
    - name: web_only_budget_per_avg_click
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {divide: [{cast: {input: budget, target: double}}, avg_clicks]}
          else: {lit: 0.0}
    - name: web_only_cpc
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {divide: [{cast: {input: budget, target: double}}, clicks]}
          else: {lit: 0.0}
    - name: web_label
      data_type: string
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {cast: {input: budget, target: string}}
          else: {lit: none}
    - name: web_spend_label
      data_type: string
      expr:
        case:
          whens:
            - when: {eq: [channel, {lit: web}]}
              then: {cast: {input: {add: [budget, clicks]}, target: string}}
          else: {lit: none}
  grainsets:
    - name: spend
      dimensions:
        - {ref: date, is_key: true}
        - {ref: campaign, is_key: true}
        - {ref: channel, is_key: true}
      measures: [{ref: clicks}, {ref: budget}, {ref: avg_clicks}]
      metrics: [{ref: web_budget}, {ref: web_cpc2}, {ref: web_cpc}, {ref: cpc_avg}, {ref: web_only_cpc}, {ref: web_label}, {ref: web_spend_label}, {ref: budget_per_avg_click}, {ref: web_only_budget_per_avg_click}]
      datasets:
        - name: spend_daily
          extras:
            storage: {tables: [warehouse.spend_daily]}
            semantic_mapping: {date: day, campaign: campaign_id, channel: channel, clicks: click_count, avg_clicks: click_avg}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
        - name: spend_monthly
          extras:
            storage: {tables: [warehouse.spend_monthly]}
            semantic_mapping: {date: month_start, campaign: campaign_id, channel: channel, budget: amount}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
  joinsets:
    - name: j
      dimensions: [{ref: campaign, is_key: true}, {ref: date}, {ref: channel}]
      measures: [{ref: clicks}, {ref: budget}, {ref: avg_clicks}]
      metrics: [{ref: web_budget}, {ref: web_cpc2}, {ref: budget_per_avg_click}]
      datasets:
        - name: campaigns
          extras:
            storage: {tables: [warehouse.campaigns]}
            semantic_mapping: {campaign: campaign_id}
      grainsets:
        - name: spend
          dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}, {ref: channel, is_key: true}]
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: spend_daily
              extras:
                storage: {tables: [warehouse.spend_daily]}
                semantic_mapping: {date: day, campaign: campaign_id, channel: channel, clicks: click_count, avg_clicks: click_avg}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: spend_monthly
              extras:
                storage: {tables: [warehouse.spend_monthly]}
                semantic_mapping: {date: month_start, campaign: campaign_id, channel: channel, budget: amount}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
      relationships:
        - {name: spend_to_campaigns, from: spend, to: campaigns, on: [campaign], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

async fn spend_metrics() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.spend_daily",
            FileFormat::Iceberg,
            &[
                ("day", "date"),
                ("campaign_id", "string"),
                ("channel", "string"),
                ("click_count", "long"),
                ("click_avg", "double"),
            ],
        )
        .table(
            "warehouse.spend_monthly",
            FileFormat::Iceberg,
            &[
                ("month_start", "date"),
                ("campaign_id", "string"),
                ("channel", "string"),
                ("amount", "long"),
            ],
        )
        .table(
            "warehouse.campaigns",
            FileFormat::Iceberg,
            &[("campaign_id", "string")],
        );
    compiled(SPEND_METRICS, &catalog, &InMemoryFileSystem::new()).await
}

fn rows_filter(side: &PlanNode) -> PhysicalExpr {
    let aggregate = first_aggregate(side);
    let [(_, call)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let mut filter = call.filter().expect("a filter on the measure").clone();
    let mut below: &PlanNode = &aggregate.input;
    while let PlanNode::Project(project) = below {
        filter = computed(project, &filter);
        below = &project.input;
    }
    filter
}

fn reads_channel_equal_to_web(filter: &PhysicalExpr) {
    let Expr::BinaryOp {
        op: BinaryOpKind::Eq,
        left,
        right,
    } = filter
    else {
        panic!("an equality, got {filter:?}");
    };
    assert!(matches!(left.as_ref(), Expr::Column(column) if column.name() == "channel"));
    assert_eq!(**right, text("web"));
}

fn sides_by_table(plan: &SemanticPlan) -> (&PlanNode, &PlanNode) {
    let join = join(drill_across(plan));
    match table_below(&join.left) {
        "spend_monthly" => (&join.left, &join.right),
        _ => (&join.right, &join.left),
    }
}

#[tokio::test]
async fn a_filtered_metric_a_child_maps_whole_keeps_its_filter_in_that_childs_aggregate() {
    let manifest = spend_metrics().await;
    let request = at_granularity("date", Granularity::Month, &["web_cpc2"]);

    let (plan, advisories) = planned(&manifest, &request);

    let (monthly, daily) = sides_by_table(&plan);
    assert_eq!(table_below(daily), "spend_daily");
    reads_channel_equal_to_web(&rows_filter(monthly));
    assert_eq!(aggregate_filters(daily).len(), 1);
    assert!(aggregate_filters(daily)
        .iter()
        .all(|(_, filter)| filter.is_none()));
    assert_eq!(output_names(&plan), ["date", "web_cpc2"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_metrics_own_filter_sits_in_the_aggregate_of_each_child_that_serves_a_measure_it_reads() {
    let manifest = spend_metrics().await;
    let request = at_granularity("date", Granularity::Month, &["web_cpc"]);

    let (plan, advisories) = planned(&manifest, &request);

    let (monthly, daily) = sides_by_table(&plan);
    reads_channel_equal_to_web(&rows_filter(monthly));
    reads_channel_equal_to_web(&rows_filter(daily));
    assert_eq!(output_names(&plan), ["date", "web_cpc"]);
    assert!(advisories.is_empty());
}

fn divides(project: &semstrait_ir::ProjectNode) -> bool {
    let divide = |node: &Expr<_>| {
        matches!(
            node,
            Expr::BinaryOp {
                op: BinaryOpKind::Divide,
                ..
            }
        )
    };
    (project.expressions.iter()).any(|tree| tree.pre_order().any(divide))
}

#[tokio::test]
async fn a_computed_metrics_formula_is_one_projection_above_the_full_join() {
    let manifest = spend_metrics().await;
    let request = at_granularity("date", Granularity::Month, &["web_cpc2"]);

    let (plan, _) = planned(&manifest, &request);

    let formula = projection(&plan.root);
    assert!(divides(formula));
    let joined = projection(&formula.input);
    assert!(!divides(joined));
    let across = join(&joined.input);
    assert_eq!(across.join_type, JoinType::Full);
    let projections = plan.root.pre_order().filter_map(|node| match node {
        PlanNode::Project(project) => Some(project),
        _ => None,
    });
    assert_eq!(projections.filter(|project| divides(project)).count(), 1);
    assert_eq!(aggregate_steps(&plan.root), 2);
    drill_across(&plan);
}

#[tokio::test]
async fn a_computed_metrics_own_aggregation_is_one_step_above_the_formula_projection() {
    let manifest = spend_metrics().await;
    let request = at_granularity("date", Granularity::Month, &["cpc_avg"]);

    let (plan, _) = planned(&manifest, &request);

    let PlanNode::Aggregate(averaged) = projection(&plan.root).input.as_ref() else {
        panic!("the aggregation step under the output, got {:?}", plan.root);
    };
    let formula = projection(&averaged.input);
    assert!(divides(formula));
    let joined = projection(&formula.input);
    assert!(!divides(joined));
    assert_eq!(join(&joined.input).join_type, JoinType::Full);
    let projections = plan.root.pre_order().filter_map(|node| match node {
        PlanNode::Project(project) => Some(project),
        _ => None,
    });
    assert_eq!(projections.filter(|project| divides(project)).count(), 1);
    assert_eq!(aggregate_steps(&plan.root), 3);
    assert_eq!(output_names(&plan), ["date", "cpc_avg"]);
}

#[tokio::test]
async fn a_computed_metric_that_reads_a_dimension_outside_an_aggregate_takes_the_parts_in_each_child(
) {
    let manifest = spend_metrics().await;
    let numeric = at_granularity("date", Granularity::Month, &["web_only_cpc"]);
    let text = at_granularity("date", Granularity::Month, &["web_label"]);
    let computed_text = at_granularity("date", Granularity::Month, &["web_spend_label"]);
    let grouped = at_granularity("date", Granularity::Month, &["channel", "web_only_cpc"]);
    let labelled = at_granularity("date", Granularity::Month, &["channel", "web_label"]);

    let (plan, advisories) = planned(&manifest, &numeric);
    let (text_error, _) = refusal(&manifest, &text);
    let (computed_text_error, _) = refusal(&manifest, &computed_text);
    let (grouped_plan, _) = planned(&manifest, &grouped);
    let (labels, _) = planned(&manifest, &labelled);
    let (total, _) = planned(&manifest, &select(&["web_only_cpc"]));

    let PlanNode::Aggregate(summed) = projection(&plan.root).input.as_ref() else {
        panic!(
            "the sum over the parts under the output, got {:?}",
            plan.root
        );
    };
    let [(_, sum)] = summed.aggregates.as_slice() else {
        panic!("one sum over the parts, got {:?}", summed.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let joins: Vec<_> = (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .collect();
    let [across] = joins.as_slice() else {
        panic!("one join of the children, got {joins:?}");
    };
    assert_eq!(across.join_type, JoinType::Full);
    let null_safe = (across.on.pre_order())
        .filter(|node| {
            matches!(
                node,
                Expr::BinaryOp {
                    op: BinaryOpKind::IsNotDistinctFrom,
                    ..
                }
            )
        })
        .count();
    assert_eq!(null_safe, 2);
    let (monthly, daily) = if table_below(&across.left) == "spend_monthly" {
        (&across.left, &across.right)
    } else {
        (&across.right, &across.left)
    };
    assert_eq!(own_aggregate(monthly, "spend_monthly").groupings.len(), 2);
    assert_eq!(own_aggregate(daily, "spend_daily").groupings.len(), 2);
    assert_eq!(aggregate_steps(&plan.root), 3);
    assert_eq!(output_names(&plan), ["date", "web_only_cpc"]);
    assert!(advisories.is_empty());
    assert!(matches!(
        &text_error.kind,
        PlanError::UngroupedMetricFormula { metric, name, .. }
            if metric.as_str() == "web_label" && name.as_str() == "channel"
    ));
    assert!(matches!(
        &computed_text_error.kind,
        PlanError::UngroupedMetricFormula { path, metric, name }
            if path.to_string() == "spend"
                && metric.as_str() == "web_spend_label"
                && name.as_str() == "channel"
    ));
    assert_eq!(
        output_names(&grouped_plan),
        ["date", "channel", "web_only_cpc"]
    );
    assert_eq!(output_names(&labels), ["date", "channel", "web_label"]);
    assert_eq!(output_names(&total), ["web_only_cpc"]);
    assert_eq!(aggregate_steps(&total.root), 3);
}

#[tokio::test]
async fn a_joinset_passes_a_metric_over_a_grainset_operand_through_to_the_operands_compute_frame() {
    let manifest = spend_metrics().await;
    let request = select(&["campaign", "web_cpc2"]).with_from("j");

    let (plan, _) = planned(&manifest, &request);

    let formulas = plan.root.pre_order().filter_map(|node| match node {
        PlanNode::Project(project) => Some(project),
        _ => None,
    });
    assert_eq!(formulas.filter(|project| divides(project)).count(), 1);
    let joins: Vec<_> = (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Join(join) => Some(join.join_type),
            _ => None,
        })
        .collect();
    assert_eq!(joins, [JoinType::Full]);
    assert_eq!(output_names(&plan), ["campaign", "web_cpc2"]);
}

fn lossy_at_month(
    manifest: &SemanticManifest,
    from: &str,
    names: &[&str],
) -> Vec<(String, String)> {
    let request = at_granularity("date", Granularity::Month, names).with_from(from);
    let (_, advisories) = planned(manifest, &request);
    (lossy(&advisories).into_iter())
        .map(|(path, semantics)| (path, semantics.to_owned()))
        .collect()
}

#[tokio::test]
async fn a_computed_metric_over_the_drill_across_raises_the_lossy_advisory_of_the_non_additive_measure_it_reads(
) {
    let manifest = spend_metrics().await;

    let named = lossy_at_month(&manifest, "spend", &["avg_clicks"]);
    let computed = lossy_at_month(&manifest, "spend", &["budget_per_avg_click"]);
    let two_groups = ["budget_per_avg_click", "web_only_budget_per_avg_click"];
    let grouped = lossy_at_month(&manifest, "spend", &two_groups);

    let at_daily = vec![(
        "spend / spend.spend_daily".to_owned(),
        "avg_clicks".to_owned(),
    )];
    assert_eq!(named, at_daily);
    assert_eq!(computed, at_daily);
    assert_eq!(grouped, at_daily);
}

#[tokio::test]
async fn a_joinset_over_a_grainset_operand_raises_the_lossy_advisory_of_the_non_additive_measure_its_computed_metric_reads(
) {
    let manifest = spend_metrics().await;

    let named = lossy_at_month(&manifest, "j", &["avg_clicks"]);
    let computed = lossy_at_month(&manifest, "j", &["budget_per_avg_click"]);

    let at_root = vec![("j".to_owned(), "avg_clicks".to_owned())];
    assert_eq!(named, at_root);
    assert_eq!(computed, at_root);
}

const TIED_COST_CLICKS: &str = r#"
semantic_model:
  name: tied
  dimensions:
    - {name: campaign, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: impressions, data_type: bigint, agg: sum}
  metrics:
    - {name: cpc, data_type: double, expr: {divide: [{cast: {input: cost, target: double}}, clicks]}}
    - name: region_cpc_avg
      data_type: double
      agg: avg
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [{divide: [{cast: {input: cost, target: double}}, clicks]}, 1.2]}
          else: {divide: [{cast: {input: cost, target: double}}, clicks]}
  grainsets:
    - name: g
      dimensions: [{ref: campaign, is_key: true}, {ref: region, is_key: true}]
      measures: [{ref: cost}, {ref: clicks}, {ref: impressions}]
      metrics: [{ref: cpc}, {ref: region_cpc_avg}]
      datasets:
        - name: c_cost
          extras:
            storage: {tables: [warehouse.c_cost]}
            semantic_mapping: {campaign: campaign, region: region, cost: cost}
        - name: group0
          extras:
            storage: {tables: [warehouse.c_clicks_a]}
            semantic_mapping: {campaign: campaign, region: region, clicks: clicks}
        - name: x_clicks
          extras:
            storage: {tables: [warehouse.c_clicks_b]}
            semantic_mapping: {campaign: campaign, region: region, clicks: clicks, impressions: impressions}
"#;

async fn tied_cost_clicks() -> SemanticManifest {
    let s = |name| (name, "string");
    let l = |name| (name, "long");
    let catalog = Catalog::default()
        .table(
            "warehouse.c_cost",
            FileFormat::Iceberg,
            &[s("campaign"), s("region"), l("cost")],
        )
        .table(
            "warehouse.c_clicks_a",
            FileFormat::Iceberg,
            &[s("campaign"), s("region"), l("clicks")],
        )
        .table(
            "warehouse.c_clicks_b",
            FileFormat::Iceberg,
            &[s("campaign"), s("region"), l("clicks"), l("impressions")],
        );
    compiled(TIED_COST_CLICKS, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn an_advisory_raised_inside_a_formula_group_names_that_groups_own_path() {
    let manifest = tied_cost_clicks().await;

    let (_, advisories) = planned(&manifest, &select(&["campaign", "cpc", "region_cpc_avg"]));

    let at: Vec<String> = (advisory_kinds(advisories).iter())
        .map(|advisory| advisory.path().to_string())
        .collect();
    assert_eq!(at, ["g / g#group0", "g / g#group1"]);
}

#[tokio::test]
async fn a_formula_groups_path_segment_is_never_the_name_of_a_child() {
    let manifest = tied_cost_clicks().await;

    let (_, advisories) = planned(&manifest, &select(&["campaign", "cpc", "region_cpc_avg"]));

    let messages: Vec<_> = (advisories.into_iter())
        .map(|advisory| advisory.kind.message())
        .collect();
    assert_eq!(
        messages,
        [
            "at `g / g#group0`: `g.group0` is chosen by name order over `g.x_clicks`: make them differ in the model",
            "at `g / g#group1`: `g.group0` is chosen by name order over `g.x_clicks`: make them differ in the model",
        ]
    );
}

#[tokio::test]
async fn a_formula_group_that_one_child_answers_names_that_child_on_the_groups_line() {
    let manifest = spend_metrics().await;
    let request = at_granularity("date", Granularity::Month, &["clicks", "cpc_avg"]);

    let (plan, _) = planned(&manifest, &request);

    let grainset = DataKindName::new("spend");
    let answered: Vec<_> = (plan.root.pre_order())
        .flat_map(|node| &node.common().semantic_meta)
        .filter_map(|subject| match subject {
            SemAnnotation::SemanticSource {
                semantic_source,
                answered_by,
                ..
            } if *semantic_source == grainset => answered_by.as_ref(),
            _ => None,
        })
        .collect();
    assert_eq!(answered, [&DataKindName::new("spend.spend_daily")]);
    assert_eq!(output_names(&plan), ["date", "clicks", "cpc_avg"]);
}

const REGION_CPC: &str = r#"
semantic_model:
  name: region_cpc
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: avg_cost, data_type: double, agg: avg}
    - {name: budget, data_type: bigint, agg: sum}
  metrics:
    - name: region_cpc_avg
      data_type: double
      agg: avg
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [{divide: [{cast: {input: cost, target: double}}, clicks]}, 1.2]}
          else: {divide: [{cast: {input: cost, target: double}}, clicks]}
    - name: region_avg_cost_per_click_avg
      data_type: double
      agg: avg
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [{divide: [avg_cost, clicks]}, 1.2]}
          else: {divide: [avg_cost, clicks]}
    - {name: region_cpc_max, data_type: double, agg: max, expr: {multiply: [region_cpc_avg, 2.0]}}
    - name: early_cpc_avg
      data_type: double
      agg: avg
      expr:
        case:
          whens:
            - when: {lt: [date, {lit: {date: "2026-03-02"}}]}
              then: {divide: [{cast: {input: cost, target: double}}, clicks]}
          else: {lit: 0.0}
  datasets:
    - name: region_daily
      extras:
        storage: {tables: [warehouse.region_daily]}
        semantic_mapping: {date: date, campaign: campaign, region: region, cost: cost, clicks: clicks, avg_cost: avg_cost}
        temporal: {timeseries: {occurred_at: date}, granularity: day}
      dimensions:
        - {ref: date, is_key: true}
        - {ref: campaign, is_key: true}
        - {ref: region, is_key: true}
      measures: [{ref: cost}, {ref: clicks}, {ref: avg_cost}]
      metrics: [{ref: region_cpc_avg}, {ref: region_cpc_max}, {ref: region_avg_cost_per_click_avg}]
  grainsets:
    - name: region_split
      dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}, {ref: region, is_key: true}]
      measures: [{ref: cost}, {ref: clicks}, {ref: avg_cost}]
      metrics: [{ref: region_cpc_avg}, {ref: region_cpc_max}, {ref: region_avg_cost_per_click_avg}]
      datasets:
        - name: region_cost
          extras:
            storage: {tables: [warehouse.region_cost]}
            semantic_mapping: {date: date, campaign: campaign, region: region, cost: cost, avg_cost: avg_cost}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
        - name: region_clicks
          extras:
            storage: {tables: [warehouse.region_clicks]}
            semantic_mapping: {date: date, campaign: campaign, region: region, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
    - name: region_monthly
      dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}, {ref: region, is_key: true}]
      measures: [{ref: cost}, {ref: clicks}, {ref: avg_cost}]
      metrics: [{ref: early_cpc_avg}]
      datasets:
        - name: monthly_cost
          extras:
            storage: {tables: [warehouse.region_cost]}
            semantic_mapping: {date: date, campaign: campaign, region: region, cost: cost}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
        - name: monthly_clicks
          extras:
            storage: {tables: [warehouse.region_clicks]}
            semantic_mapping: {date: date, campaign: campaign, region: region, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
        - name: daily_avg_cost
          extras:
            storage: {tables: [warehouse.region_cost]}
            semantic_mapping: {date: date, campaign: campaign, region: region, avg_cost: avg_cost}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
    - name: region_budgeted
      dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}, {ref: region}]
      measures: [{ref: cost}, {ref: clicks}, {ref: avg_cost}, {ref: budget}]
      metrics: [{ref: region_avg_cost_per_click_avg}]
      datasets:
        - name: budgeted_cost
          extras:
            storage: {tables: [warehouse.region_cost]}
            semantic_mapping: {date: date, campaign: campaign, region: region, cost: cost, avg_cost: avg_cost}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
        - name: budgeted_clicks
          extras:
            storage: {tables: [warehouse.region_clicks]}
            semantic_mapping: {date: date, campaign: campaign, region: region, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
        - name: campaign_budget
          extras:
            storage: {tables: [warehouse.campaign_budget]}
            semantic_mapping: {date: date, campaign: campaign, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
"#;

async fn region_cpc() -> SemanticManifest {
    let columns = |last: &[&'static str]| {
        let mut columns = vec![
            ("date", "date"),
            ("campaign", "string"),
            ("region", "string"),
        ];
        columns.extend(last.iter().map(|name| match *name {
            "avg_cost" => (*name, "double"),
            _ => (*name, "long"),
        }));
        columns
    };
    let catalog = Catalog::default()
        .table(
            "warehouse.region_daily",
            FileFormat::Iceberg,
            &columns(&["cost", "clicks", "avg_cost"]),
        )
        .table(
            "warehouse.region_cost",
            FileFormat::Iceberg,
            &columns(&["cost", "avg_cost"]),
        )
        .table(
            "warehouse.region_clicks",
            FileFormat::Iceberg,
            &columns(&["clicks"]),
        )
        .table(
            "warehouse.campaign_budget",
            FileFormat::Iceberg,
            &[("date", "date"), ("campaign", "string"), ("budget", "long")],
        );
    compiled(REGION_CPC, &catalog, &InMemoryFileSystem::new()).await
}

fn reads_of(plan: &SemanticPlan, table: &str) -> usize {
    tables_read(plan)
        .into_iter()
        .filter(|read| *read == table)
        .count()
}

#[tokio::test]
async fn a_late_own_agg_metric_reads_each_child_no_more_times_than_one_dataset_reads_its_source() {
    let manifest = region_cpc().await;
    let names = ["campaign", "clicks", "region_cpc_max"];
    let request = |from: &str| at_granularity("date", Granularity::Month, &names).with_from(from);

    let (dataset, _) = planned(&manifest, &request("region_daily"));
    let (grainset, _) = planned(&manifest, &request("region_split"));

    let source = reads_of(&dataset, "region_daily");
    assert_eq!(source, 3);
    assert_eq!(reads_of(&grainset, "region_clicks"), 2);
    assert_eq!(reads_of(&grainset, "region_cost"), 1);
    assert_eq!(
        output_names(&grainset),
        ["date", "campaign", "clicks", "region_cpc_max"]
    );
}

#[tokio::test]
async fn a_formula_grain_group_that_holds_its_childs_keyset_raises_no_lossy_advisory() {
    let manifest = region_cpc().await;
    let metric = "region_avg_cost_per_click_avg";

    let keyed = select(&["date", "campaign", metric]).with_from("region_split");
    let (_, keyed_advisories) = planned(&manifest, &keyed);
    let coarse = lossy_at_month(&manifest, "region_split", &["campaign", metric]);

    assert_eq!(lossy(&keyed_advisories), []);
    assert_eq!(
        coarse,
        [(
            "region_split / region_split.region_cost".to_owned(),
            "avg_cost".to_owned()
        )]
    );
}

#[tokio::test]
async fn a_formula_grain_group_that_reads_the_time_axis_is_not_servable_finer_than_the_children_that_give_its_values(
) {
    let manifest = region_cpc().await;
    let at = |granularity, names: &[&str]| {
        at_granularity("date", granularity, names).with_from("region_monthly")
    };

    let plain = refusal(&manifest, &at(Granularity::Day, &["campaign", "cost"])).0;
    let computed = refusal(
        &manifest,
        &at(Granularity::Day, &["campaign", "early_cpc_avg"]),
    )
    .0;
    let (monthly, _) = planned(
        &manifest,
        &at(Granularity::Month, &["campaign", "early_cpc_avg"]),
    );

    for error in [plain, computed] {
        let PlanError::GrainNotServable {
            path,
            requested,
            source_granularity,
            ..
        } = &error.kind
        else {
            panic!("GrainNotServable, got {error:?}");
        };
        assert_eq!(path.to_string(), "region_monthly");
        assert_eq!(
            (*requested, *source_granularity),
            (Some(Granularity::Day), None)
        );
    }
    assert_eq!(
        output_names(&monthly),
        ["date", "campaign", "early_cpc_avg"]
    );
}

#[tokio::test]
async fn a_formula_grain_group_under_a_coarser_drill_across_raises_the_lossy_advisory_of_the_measure_it_rolls_up(
) {
    let manifest = region_cpc().await;
    let rest = ["campaign", "budget", "region_avg_cost_per_click_avg"];
    let names = [
        "date",
        "campaign",
        "budget",
        "region_avg_cost_per_click_avg",
    ];

    let (plan, advisories) = planned(&manifest, &select(&names).with_from("region_budgeted"));
    let at_month = at_granularity("date", Granularity::Month, &rest).with_from("region_budgeted");
    let (_, month_advisories) = planned(&manifest, &at_month);

    let rolled_up = [(
        "region_budgeted / region_budgeted.budgeted_cost".to_owned(),
        "avg_cost",
    )];
    assert_eq!(output_names(&plan), names);
    assert_eq!(lossy(&month_advisories), rolled_up);
    assert_eq!(lossy(&advisories), rolled_up);
}
