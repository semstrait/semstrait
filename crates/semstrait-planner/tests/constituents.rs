mod common;

use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::Granularity;
use semstrait_common::{Diagnostic, Tree};
use semstrait_ir::{JoinNode, JoinType, PhysicalExpr, PlanNode, SemanticPlan};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit};
use semstrait_planner::{PlanError, Request};
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal};

use common::golden::{self, Golden};
use common::{
    aggregate_filters, aggregate_step, aggregate_steps, alias, answered_by_source, at_granularity,
    bigint, call, column, columns_of, compiled, computed, entry, every_marker, fields,
    first_aggregate, grain_not_servable, join, lossy, markers, names, once, output_names, parts,
    planned, projection, qualified, refusal, row_number, select, table_below, tables_read, text,
    union_of, without_signature, written, Catalog,
};

const SPEND_GS: &str = r#"
      grainsets:
        - name: spend_gs
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: spend_daily
              extras:
                storage: {tables: [warehouse.spend_daily]}
                semantic_mapping: {date: day, campaign_id: campaign_id, cost: cost, amount: cost}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: spend_monthly
              extras:
                storage: {tables: [warehouse.spend_monthly]}
                semantic_mapping: {date: month_start, campaign_id: campaign_id, cost: cost, amount: cost}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
"#;

const ALL_SPEND: &str = r#"
semantic_model:
  name: all_spend
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: amount, data_type: bigint, type: categorical}
  measures:
    - {name: cost, data_type: bigint, agg: sum}
  unionsets:
    - name: all_spend
      mode: all
      dimensions: [{ref: date}, {ref: campaign_id}, {ref: amount}]
      measures: [{ref: cost}]
      datasets:
        - name: spend_store
          extras:
            storage: {tables: [warehouse.spend_store]}
            semantic_mapping: {date: day, campaign_id: campaign_id, cost: cost, amount: cost}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

const SPEND_BY_CAMPAIGN: &str = r#"
semantic_model:
  name: spend_by_campaign
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: campaign_name, data_type: string, type: categorical}
    - {name: segment, data_type: string, type: categorical}
    - {name: amount, data_type: bigint, type: categorical}
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - name: vip_cost
      data_type: bigint
      agg: sum
      expr:
        case:
          whens:
            - when: {eq: [segment, {lit: vip}]}
              then: amount
  joinsets:
    - name: spend_by_campaign
      dimensions:
        - {ref: date}
        - {ref: campaign_id}
        - {ref: campaign_name}
        - {ref: segment}
        - {ref: amount}
      measures: [{ref: cost}, {ref: vip_cost}]
      datasets:
        - name: campaigns
          extras:
            storage: {tables: [warehouse.campaigns]}
      relationships:
        - name: campaigns_to_spend
          from: campaigns
          to: spend_gs
          on: [campaign_id]
          cardinality: one_to_many
          optional: none
          integrity: assumed
"#;

async fn over(yaml: &str, tables: &[(&str, &str)]) -> SemanticManifest {
    let catalog = (tables.iter()).fold(Catalog::default(), |catalog, (name, columns)| {
        let columns: Vec<_> = (columns.split(", "))
            .map(|column| column.split_once(' ').expect("`<column> <native type>`"))
            .collect();
        catalog.table(&format!("warehouse.{name}"), FileFormat::Iceberg, &columns)
    });
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

async fn with_spend_gs(yaml: &str) -> SemanticManifest {
    let spend = "day date, campaign_id string, cost long";
    let tables = [
        ("spend_daily", spend),
        ("spend_store", spend),
        (
            "spend_monthly",
            "month_start date, campaign_id string, cost long",
        ),
        (
            "campaigns",
            "campaign_id string, campaign_name string, segment string",
        ),
    ];
    over(&format!("{yaml}{SPEND_GS}"), &tables).await
}

fn first_join(node: &PlanNode) -> &JoinNode {
    (node.pre_order())
        .find_map(|node| match node {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("a join")
}

fn drill_across<'p>(
    plan: &'p SemanticPlan,
    left: &str,
    right: &str,
    keys: &[&str],
) -> &'p JoinNode {
    let drill = first_join(&plan.root);
    let on = (keys.iter())
        .map(|key| qualified(left, key).is_not_distinct_from(qualified(right, key)))
        .reduce(|on, key| on & key);
    assert_eq!(drill.join_type, JoinType::Full);
    assert_eq!(Some(&drill.on), on.as_ref());
    let below = aggregate_steps(&drill.left) + aggregate_steps(&drill.right);
    assert_eq!(
        aggregate_steps(&plan.root),
        below,
        "none above the drill-across"
    );
    drill
}

async fn grainset_branch_at_day() -> Golden {
    let manifest = with_spend_gs(ALL_SPEND).await;
    let request = at_granularity("date", Granularity::Day, &["cost"]);
    let path = ["all_spend", "all_spend.spend_gs"];
    let refused = grain_not_servable(
        &manifest,
        &request,
        &path,
        (Some(Granularity::Day), Some(Granularity::Month)),
        &[
            ("all_spend.spend_gs.spend_daily", &[]),
            ("all_spend.spend_gs.spend_monthly", &[]),
        ],
    );
    Golden::message("grainset_branch_at_day", &refused)
}

#[tokio::test]
async fn a_constituent_grainset_serves_no_granularity_finer_than_it_declares() {
    grainset_branch_at_day().await.assert_committed();
}

async fn composition_at_day() -> Golden {
    let manifest = with_spend_gs(SPEND_BY_CAMPAIGN).await;
    let request = at_granularity("date", Granularity::Day, &["campaign_name", "cost"])
        .with_from("spend_by_campaign");
    let path = ["spend_by_campaign"];
    let refused = grain_not_servable(
        &manifest,
        &request,
        &path,
        (Some(Granularity::Day), Some(Granularity::Month)),
        &[],
    );
    Golden::message("composition_at_day", &refused)
}

#[tokio::test]
async fn a_composition_over_a_constituent_grainset_serves_nothing_finer_than_its_declaration() {
    composition_at_day().await.assert_committed();
}

#[tokio::test]
async fn a_constituent_grainset_picks_its_child_at_the_granularity_it_declares() {
    let manifest = with_spend_gs(SPEND_BY_CAMPAIGN).await;
    let request = at_granularity("date", Granularity::Month, &["campaign_name", "cost"])
        .with_from("spend_by_campaign");

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["campaigns", "spend_monthly"]);
    let join = first_join(&plan.root);
    let spend_gs = alias(&manifest, "spend_by_campaign.spend_gs");
    let campaigns = alias(&manifest, "spend_by_campaign.campaigns");
    assert_eq!(
        join.on,
        qualified(&campaigns, "campaign_id").eq(qualified(&spend_gs, "campaign_id"))
    );
    assert_eq!(table_below(&join.right), "spend_monthly");
    let written = names(&join.right.common().output_schema);
    assert!(written.contains(&format!("{spend_gs}.campaign_id").as_str()));
}

#[tokio::test]
async fn a_constituent_grainset_fixes_the_granularity_it_declares_where_the_request_asks_none() {
    let manifest = with_spend_gs(SPEND_BY_CAMPAIGN).await;
    let request = select(&["date", "campaign_name", "cost"]).with_from("spend_by_campaign");

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["campaigns", "spend_monthly"]);
}

#[tokio::test]
async fn a_value_walk_goes_through_the_child_that_a_constituent_grainset_picks() {
    let manifest = with_spend_gs(SPEND_BY_CAMPAIGN).await;
    let request =
        at_granularity("date", Granularity::Month, &["vip_cost"]).with_from("spend_by_campaign");

    let (plan, advisories) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["campaigns", "spend_monthly"]);
    assert_eq!(aggregate_steps(&plan.root), 1);
    let aggregate = aggregate_step(&plan.root);
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let [campaigns, spend_gs] = ["campaigns", "spend_gs"]
        .map(|name| alias(&manifest, &format!("spend_by_campaign.{name}")));
    let vip = Expr::Case {
        whens: vec![(
            qualified(&campaigns, "segment").eq(text("vip")),
            qualified(&spend_gs, "amount"),
        )],
        else_: None,
    };
    let bucketing = projection(&aggregate.input);
    let rows = projection(&bucketing.input);
    assert_eq!(computed(rows, &sum.args()[0]), vip);
    assert_eq!(sum.filter(), None);
    let monthly = projection(&join(&rows.input).right);
    let amount = written(monthly, &format!("{spend_gs}.amount"));
    assert_eq!(amount, Some(columns_of(&manifest, "spend_monthly")("cost")));
    assert!(every_marker(&plan.root).is_empty());
    assert_eq!(output_names(&plan), ["date", "vip_cost"]);
    assert!(advisories.is_empty());
}

const SPEND_G: &str = r#"
semantic_model:
  name: spend_g
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: campaign_id, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: target_clicks, data_type: bigint, agg: sum}
  grainsets:
    - name: spend_g
      dimensions: [{ref: date, is_key: true}, {ref: campaign_id, is_key: true}]
      measures: [{ref: clicks}, {ref: target_clicks}]
      datasets:
        - name: spend_year
          extras:
            storage: {tables: [warehouse.spend_year]}
            semantic_mapping: {date: year_start, campaign_id: campaign_id, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: year}
      joinsets:
        - name: spend_targets
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: spend
              extras:
                storage: {tables: [warehouse.spend]}
                semantic_mapping: {date: day, campaign_id: campaign_id, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: targets
              extras:
                storage: {tables: [warehouse.targets]}
                semantic_mapping: {date: month_start, campaign_id: campaign_id, target_clicks: target}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
          relationships:
            - {name: spend_to_targets, from: spend, to: targets, on: [campaign_id], cardinality: many_to_one, optional: left, integrity: assumed}
"#;

const SPEND_G_TABLES: &[(&str, &str)] = &[
    ("spend", "day date, campaign_id string, clicks long"),
    (
        "spend_year",
        "year_start date, campaign_id string, clicks long",
    ),
    (
        "targets",
        "month_start date, campaign_id string, target long",
    ),
];

#[tokio::test]
async fn a_composition_that_declares_month_over_day_rows_buckets_them_and_re_aggregates() {
    let manifest = over(SPEND_G, SPEND_G_TABLES).await;
    let request = at_granularity("date", Granularity::Month, &["campaign_id", "clicks"]);

    let (plan, advisories) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["spend"]);
    assert_eq!(aggregate_steps(&plan.root), 1);
    let aggregate = aggregate_step(&plan.root);
    let date_trunc = call("date_trunc", vec![text("month"), column("date")]);
    assert!(fields(projection(&aggregate.input)).contains(&("date", date_trunc)));
    assert_eq!(aggregate.groupings, [column("date"), column("campaign_id")]);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(
        (entry.as_str(), sum.function().declarations.canonical.name),
        ("clicks", "sum")
    );
    assert_eq!(output_names(&plan), ["date", "campaign_id", "clicks"]);
    assert!(advisories.is_empty());
}

const ORDERS_GS: &str = r#"
semantic_model:
  name: orders_gs
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: customer_id, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  grainsets:
    - name: orders_gs
      dimensions: [{ref: date}, {ref: customer_id}, {ref: region}]
      measures: [{ref: revenue}]
      datasets:
        - name: region_sales_monthly
          extras:
            storage: {tables: [warehouse.region_sales_monthly]}
            semantic_mapping: {date: month_start, region: region, revenue: revenue}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
      joinsets:
        - name: orders_daily
          datasets:
            - name: web_orders_dated
              extras:
                storage: {tables: [warehouse.web_orders_dated]}
                semantic_mapping: {date: order_date, customer_id: customer_id, revenue: amount}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: customers_s4
              extras:
                storage: {tables: [warehouse.customers_s4]}
                semantic_mapping: {customer_id: customer_id, region: region}
          relationships:
            - {name: orders_to_customers, from: web_orders_dated, to: customers_s4, on: [customer_id], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

const ORDERS_GS_TABLES: &[(&str, &str)] = &[
    (
        "region_sales_monthly",
        "month_start date, region string, revenue double",
    ),
    (
        "web_orders_dated",
        "order_date date, customer_id string, amount double",
    ),
    ("customers_s4", "customer_id string, region string"),
];

fn answered_by(plan: &SemanticPlan, child: &str) {
    answered_by_source(plan, &format!("orders_gs.{child}"));
}

#[tokio::test]
async fn a_grainset_routes_by_the_granularity_that_its_joinset_child_derives() {
    let manifest = over(ORDERS_GS, ORDERS_GS_TABLES).await;
    let names = ["date", "region", "revenue"];
    let at = |granularity| at_granularity("date", granularity, &names[1..]);

    let (daily, advisories) = planned(&manifest, &at(Granularity::Day));

    answered_by(&daily, "orders_daily");
    assert_eq!(tables_read(&daily), ["web_orders_dated", "customers_s4"]);
    let [orders, customers] = ["web_orders_dated", "customers_s4"]
        .map(|operand| alias(&manifest, &format!("orders_gs.orders_daily.{operand}")));
    let join = first_join(&daily.root);
    assert_eq!(
        join.on,
        qualified(&orders, "customer_id").eq(qualified(&customers, "customer_id"))
    );
    assert_eq!(output_names(&daily), names);
    assert!(advisories.is_empty());

    let (monthly, _) = planned(&manifest, &at(Granularity::Month));

    answered_by(&monthly, "region_sales_monthly");
    assert_eq!(tables_read(&monthly), ["region_sales_monthly"]);
    assert_eq!(output_names(&monthly), names);
}

#[tokio::test]
async fn a_grainset_child_s_operand_column_named_like_a_condition_field_is_no_aggregation_condition(
) {
    let flagged = (ORDERS_GS.replacen(
        "    - {name: region, data_type: string, type: categorical}\n",
        "    - {name: region, data_type: string, type: categorical}\n    \
         - {name: flag, data_type: boolean, type: categorical}\n",
        1,
    ))
    .replacen("{ref: region}]", "{ref: region}, {ref: flag}]", 1)
    .replacen(
        "revenue: amount}",
        "revenue: amount, flag: __plan_cond_amount}",
        1,
    );
    let web = "order_date date, customer_id string, amount double, __plan_cond_amount boolean";
    let tables: Vec<_> = (ORDERS_GS_TABLES.iter())
        .map(|&(name, columns)| {
            (
                name,
                if name == "web_orders_dated" {
                    web
                } else {
                    columns
                },
            )
        })
        .collect();
    let manifest = over(&flagged, &tables).await;

    for names in [&["region", "flag", "revenue"][..], &["flag", "revenue"]] {
        let (plan, _) = planned(&manifest, &at_granularity("date", Granularity::Day, names));

        answered_by(&plan, "orders_daily");
        let filters = aggregate_filters(&plan.root);
        assert!(!filters.is_empty(), "{names:?}");
        assert!(
            filters.iter().all(|(_, filter)| filter.is_none()),
            "{names:?}: {filters:?}"
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
    - {name: campaign, data_type: string, type: categorical}
  measures:
    - {name: cost, data_type: bigint, agg: sum}
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
  grainsets:
    - name: spend
      dimensions: [{ref: date, is_key: true}, {ref: campaign, is_key: true}]
      measures: [{ref: cost}, {ref: clicks}, {ref: budget}]
      datasets:
        - name: budget_monthly
          extras:
            storage: {tables: [warehouse.budget_monthly]}
            semantic_mapping: {date: month_start, campaign: campaign, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
      joinsets:
        - name: spend_daily
          datasets:
            - name: cost_daily
              extras:
                storage: {tables: [warehouse.cost_daily]}
                semantic_mapping: {date: day, campaign: campaign, cost: cost}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: clicks_daily
              extras:
                storage: {tables: [warehouse.clicks_daily]}
                semantic_mapping: {date: day, campaign: campaign, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
          relationships:
            - {name: cost_to_clicks, from: cost_daily, to: clicks_daily, on: [date, campaign], cardinality: one_to_one, optional: both, integrity: none, cross_filter: both}
"#;

const SPEND_TABLES: &[(&str, &str)] = &[
    ("cost_daily", "day date, campaign string, cost long"),
    ("clicks_daily", "day date, campaign string, clicks long"),
    (
        "budget_monthly",
        "month_start date, campaign string, budget long",
    ),
];

#[tokio::test]
async fn a_drill_across_over_a_joinset_child_compares_the_coalesce_that_it_projects() {
    let manifest = over(SPEND, SPEND_TABLES).await;
    let rest = ["campaign", "cost", "clicks", "budget"];
    let request = at_granularity("date", Granularity::Month, &rest);

    let (plan, advisories) = planned(&manifest, &request);

    let [budget, daily, cost, clicks] = [
        "budget_monthly",
        "spend_daily",
        "spend_daily.cost_daily",
        "spend_daily.clicks_daily",
    ]
    .map(|name| alias(&manifest, &format!("spend.{name}")));
    let drill = drill_across(&plan, &budget, &daily, &["date", "campaign"]);
    assert_eq!(table_below(&drill.left), "budget_monthly");
    let aggregate = first_aggregate(&drill.right);
    let bucketing = projection(&aggregate.input);
    let rows = projection(&bucketing.input);
    let full = join(&rows.input);
    assert_eq!(full.join_type, JoinType::Full);
    let key = |name| format!("{daily}.{name}");
    for name in ["date", "campaign"] {
        let coalesced = call(
            "coalesce",
            vec![qualified(&cost, name), qualified(&clicks, name)],
        );
        assert_eq!(written(rows, &key(name)), Some(coalesced), "{name}");
    }
    let month = call("date_trunc", vec![text("month"), qualified(&daily, "date")]);
    assert_eq!(written(bucketing, &key("date")), Some(month));
    assert_eq!(
        output_names(&plan),
        ["date", "campaign", "cost", "clicks", "budget"]
    );
    assert!(advisories.is_empty());
}

const G_CAMPAIGNS: &str = r#"
semantic_model:
  name: g_campaigns
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: country, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: avg_clicks, data_type: double, agg: avg}
    - {name: budget, data_type: double, agg: sum}
  joinsets:
    - name: g_campaigns
      dimensions:
        - {ref: date, is_key: true, keysets: [g]}
        - {ref: campaign_id, is_key: true, keysets: [campaign, g]}
        - {ref: country, is_key: true, keysets: [g]}
        - {ref: region}
      measures: [{ref: avg_clicks}, {ref: budget}]
      datasets:
        - name: campaigns
          extras:
            storage: {tables: [warehouse.campaigns]}
      grainsets:
        - name: g
          extras:
            temporal: {events: {event_time: date}, granularity: month}
          datasets:
            - name: g_day
              extras:
                storage: {tables: [warehouse.g_day]}
                semantic_mapping: {date: date, campaign_id: campaign_id, country: country, avg_clicks: clicks}
                temporal: {events: {event_time: date}, granularity: day}
            - name: g_month
              extras:
                storage: {tables: [warehouse.g_month]}
                semantic_mapping: {date: date, campaign_id: campaign_id, country: country, budget: budget}
                temporal: {events: {event_time: date}, granularity: month}
      relationships:
        - {name: g_to_campaigns, from: g, to: campaigns, on: [campaign_id], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

const G_CAMPAIGNS_TABLES: &[(&str, &str)] = &[
    (
        "g_day",
        "date date, campaign_id string, country string, clicks double",
    ),
    (
        "g_month",
        "date date, campaign_id string, country string, budget double",
    ),
    ("campaigns", "campaign_id string, region string"),
];

async fn grainset_operand_of_a_built_join() -> Golden {
    let manifest = over(G_CAMPAIGNS, G_CAMPAIGNS_TABLES).await;
    let request = select(&["region", "avg_clicks", "budget"]);
    let path = ["g_campaigns", "g_campaigns.g"];
    let refused = grain_not_servable(
        &manifest,
        &request,
        &path,
        (None, Some(Granularity::Month)),
        &[
            ("g_campaigns.g.g_day", &["budget"]),
            ("g_campaigns.g.g_month", &["avg_clicks"]),
        ],
    );
    Golden::message("grainset_operand_of_a_built_join", &refused)
}

#[tokio::test]
async fn a_grainset_operand_that_gives_rows_refuses_a_demand_that_no_single_child_serves() {
    grainset_operand_of_a_built_join().await.assert_committed();
}

#[tokio::test]
async fn a_grainset_operand_lists_a_child_that_lacks_only_a_name_a_predicate_reads() {
    let yaml = G_CAMPAIGNS
        .replacen(
            "    - {name: region, data_type: string, type: categorical}\n",
            "    - {name: region, data_type: string, type: categorical}\n    - {name: channel, \
             data_type: string, type: categorical}\n",
            1,
        )
        .replacen(
            "        - {ref: region}\n",
            "        - {ref: region}\n        - {ref: channel}\n",
            1,
        )
        .replacen(
            "country: country, avg_clicks: clicks}",
            "country: country, channel: channel, avg_clicks: clicks}",
            1,
        );
    let tables = [
        (
            "g_day",
            "date date, campaign_id string, country string, channel string, clicks double",
        ),
        G_CAMPAIGNS_TABLES[1],
        G_CAMPAIGNS_TABLES[2],
    ];
    let manifest = over(&yaml, &tables).await;
    let web = dim("channel").expect("a name").eq(lit("web"));
    let request = select(&["region", "budget"]);
    let (served, _) = planned(&manifest, &request);
    assert_eq!(tables_read(&served), ["g_month", "campaigns"]);

    grain_not_servable(
        &manifest,
        &request.with_predicate(web),
        &["g_campaigns", "g_campaigns.g"],
        (None, Some(Granularity::Month)),
        &[
            ("g_campaigns.g.g_month", &["channel"]),
            ("g_campaigns.g.g_day", &["budget"]),
        ],
    );
}

#[tokio::test]
async fn a_grainset_operand_lists_a_child_that_lacks_only_a_name_a_condition_reads() {
    let yaml = G_CAMPAIGNS
        .replacen(
            "    - {name: region, data_type: string, type: categorical}\n",
            "    - {name: region, data_type: string, type: categorical}\n    - {name: channel, \
             data_type: string, type: categorical}\n",
            1,
        )
        .replacen(
            "        - {ref: region}\n",
            "        - {ref: region}\n        - {ref: channel}\n",
            1,
        )
        .replacen(
            "{ref: budget}]",
            "{ref: budget, filter: {eq: [channel, {lit: web}]}}]",
            1,
        )
        .replacen(
            "country: country, avg_clicks: clicks}",
            "country: country, channel: channel, budget: budget}",
            1,
        )
        .replacen(
            "country: country, budget: budget}",
            "country: country, budget: budget, avg_clicks: clicks}",
            1,
        );
    let tables = [
        (
            "g_day",
            "date date, campaign_id string, country string, channel string, budget double",
        ),
        (
            "g_month",
            "date date, campaign_id string, country string, budget double, clicks double",
        ),
        G_CAMPAIGNS_TABLES[2],
    ];
    let manifest = over(&yaml, &tables).await;
    let request = select(&["region", "budget", "avg_clicks"]);

    grain_not_servable(
        &manifest,
        &request,
        &["g_campaigns", "g_campaigns.g"],
        (None, Some(Granularity::Month)),
        &[
            ("g_campaigns.g.g_day", &["avg_clicks"]),
            ("g_campaigns.g.g_month", &["budget", "channel"]),
        ],
    );
}

#[tokio::test]
async fn a_grainset_operand_read_for_one_measure_gives_its_child_s_own_rows() {
    let manifest = over(G_CAMPAIGNS, G_CAMPAIGNS_TABLES).await;

    let (plan, advisories) = planned(&manifest, &select(&["region", "avg_clicks"]));

    assert_eq!(tables_read(&plan), ["g_day", "campaigns"]);
    let aggregate = first_aggregate(&plan.root);
    let [(_, avg)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let rows = projection(&aggregate.input);
    let g = alias(&manifest, "g_campaigns.g");
    assert_eq!(computed(rows, &avg.args()[0]), qualified(&g, "avg_clicks"));
    let joined = join(&rows.input);
    assert_eq!(aggregate_steps(&joined.left), 0, "`g_day` gives its rows");
    let g_day = projection(&joined.left);
    let clicks = written(g_day, &format!("{g}.avg_clicks"));
    assert_eq!(clicks, Some(columns_of(&manifest, "g_day")("clicks")));
    assert_eq!(output_names(&plan), ["region", "avg_clicks"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_grainset_operand_that_its_joinset_keeps_alone_drills_across() {
    let manifest = over(G_CAMPAIGNS, G_CAMPAIGNS_TABLES).await;
    let names = ["campaign_id", "avg_clicks", "budget"];

    let (plan, advisories) = planned(&manifest, &select(&names));

    assert_eq!(tables_read(&plan), ["g_month", "g_day"]);
    let [g_month, g_day] =
        ["g_month", "g_day"].map(|child| alias(&manifest, &format!("g_campaigns.g.{child}")));
    let drill = drill_across(&plan, &g_month, &g_day, &["campaign_id"]);
    for child in [&drill.left, &drill.right] {
        assert_eq!(aggregate_steps(child), 1);
    }
    assert_eq!(output_names(&plan), names);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_grainset_operand_whose_child_has_a_broken_signature_is_a_broken_tree() {
    let yaml = G_CAMPAIGNS
        .replace(
            "    - {name: budget, data_type: double, agg: sum}\n",
            "    - {name: budget, data_type: double, agg: sum}\n  metrics:\n    - {name: budget_m, \
             data_type: double, expr: budget}\n",
        )
        .replace(
            "      measures: [{ref: avg_clicks}, {ref: budget}]\n",
            "      measures: [{ref: avg_clicks}, {ref: budget}]\n      metrics: [{ref: budget_m}]\n",
        );
    let mut manifest = over(&yaml, G_CAMPAIGNS_TABLES).await;
    let request = select(&["campaign_id", "avg_clicks", "budget_m"]);
    let (plan, _) = planned(&manifest, &request);
    assert_eq!(tables_read(&plan), ["g_month", "g_day"]);

    without_signature(&mut manifest, "g_campaigns.g.g_month", "budget_m");
    let (error, advisories) = refusal(&manifest, &request);

    let PlanError::BrokenTree {
        path,
        semantic_source,
        semantics,
    } = &error.kind
    else {
        panic!("BrokenTree, got {error:?}");
    };
    assert_eq!(path.to_string(), "g_campaigns / g_campaigns.g");
    assert_eq!(semantic_source.as_str(), "g_campaigns.g.g_month");
    assert_eq!(semantics.as_str(), "budget_m");
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_grainset_on_a_links_path_takes_the_parts() {
    let metric = r#"  metrics:
    - name: budget_eu
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [budget, {lit: 1.2}]}
          else: budget
  joinsets:
"#;
    let yaml = (G_CAMPAIGNS.replacen("  joinsets:\n", metric, 1)).replacen(
        "      measures: [{ref: avg_clicks}, {ref: budget}]\n",
        "      measures: [{ref: avg_clicks}, {ref: budget}]\n      metrics: [{ref: budget_eu}]\n",
        1,
    );
    let manifest = over(&yaml, G_CAMPAIGNS_TABLES).await;

    let (plan, advisories) = planned(&manifest, &select(&["budget_eu"]));

    assert_eq!(tables_read(&plan), ["g_month", "campaigns"]);
    let (step_2, per_part, step_1) = parts(&plan.root);
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let [key] = step_1.groupings.as_slice() else {
        panic!("the part key alone, got {:?}", step_1.groupings);
    };
    let [(eu, _), (other, _)] = step_1.aggregates.as_slice() else {
        panic!("a sum for each result, got {:?}", step_1.aggregates);
    };
    let Expr::Case { whens, else_ } = per_part else {
        panic!("the formula on each part, a case");
    };
    let [(selects, then)] = whens.as_slice() else {
        panic!("one result and the else, got {whens:?}");
    };
    assert_eq!(selects, &key.clone().eq(bigint(1)));
    let reads =
        |result: &PhysicalExpr, entry: &str| result.pre_order().any(|node| node == &column(entry));
    assert!(reads(then, eu.as_str()), "{then:?}");
    assert!(else_.is_some_and(|other_part| reads(&other_part, other.as_str())));
    let row_level = projection(&step_1.input);
    let campaigns = alias(&manifest, "g_campaigns.campaigns");
    assert_eq!(
        computed(row_level, key),
        Expr::Case {
            whens: vec![(qualified(&campaigns, "region").eq(text("eu")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    let g = alias(&manifest, "g_campaigns.g");
    let g_step = first_aggregate(&join(&row_level.input).left);
    assert_eq!(g_step.groupings, [qualified(&g, "campaign_id")]);
    assert_eq!(output_names(&plan), ["budget_eu"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_formula_push_down_that_reads_a_grainset_operand_alone_takes_the_parts() {
    let metric = r#"  metrics:
    - name: budget_de
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [country, {lit: de}]}
              then: {multiply: [budget, {lit: 1.2}]}
          else: budget
  joinsets:
"#;
    let yaml = (G_CAMPAIGNS.replacen("  joinsets:\n", metric, 1)).replacen(
        "      measures: [{ref: avg_clicks}, {ref: budget}]\n",
        "      measures: [{ref: avg_clicks}, {ref: budget}]\n      metrics: [{ref: budget_de}]\n",
        1,
    );
    let manifest = over(&yaml, G_CAMPAIGNS_TABLES).await;

    let (plan, advisories) = planned(&manifest, &select(&["region", "budget_de"]));

    assert_eq!(tables_read(&plan), ["g_month", "campaigns"]);
    let (step_2, _, step_1) = parts(&plan.root);
    let joined = join(&projection(&step_1.input).input);
    let g = alias(&manifest, "g_campaigns.g");
    let mut grouped = first_aggregate(&joined.left).groupings.clone();
    grouped.sort_by_key(|grouping| format!("{grouping:?}"));
    assert_eq!(
        grouped,
        [qualified(&g, "campaign_id"), qualified(&g, "country")],
        "`g` pre-aggregates step 1 by the join's column and the one that the key reads"
    );
    assert_eq!(step_2.groupings.len(), 1, "{:?}", step_2.groupings);
    let [_, key] = step_1.groupings.as_slice() else {
        panic!("`region` and the part key, got {:?}", step_1.groupings);
    };
    let row_level = projection(&step_1.input);
    assert_eq!(
        computed(row_level, key),
        Expr::Case {
            whens: vec![(qualified(&g, "country").eq(text("de")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    assert_eq!(output_names(&plan), ["region", "budget_de"]);
    assert!(advisories.is_empty());
}

const U_CAMPAIGNS: &str = r#"
semantic_model:
  name: u_campaigns
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: country, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: avg_clicks, data_type: double, agg: avg}
    - {name: budget, data_type: double, agg: sum}
  metrics:
    - name: budget_eu
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [budget, {lit: 1.2}]}
          else: budget
  joinsets:
    - name: u_campaigns
      dimensions:
        - {ref: date, is_key: true, keysets: [u]}
        - {ref: campaign_id, is_key: true, keysets: [campaign, u]}
        - {ref: country, is_key: true, keysets: [u]}
        - {ref: region}
      measures: [{ref: avg_clicks}, {ref: budget}]
      metrics: [{ref: budget_eu}]
      datasets:
        - name: campaigns
          extras:
            storage: {tables: [warehouse.campaigns]}
      unionsets:
        - name: u
          mode: all
          datasets:
            - name: g_extra
              extras:
                storage: {tables: [warehouse.g_extra]}
                semantic_mapping: {date: date, campaign_id: campaign_id, country: country, budget: budget}
                temporal: {events: {event_time: date}, granularity: month}
          grainsets:
            - name: g
              extras:
                temporal: {events: {event_time: date}, granularity: month}
              datasets:
                - name: g_day
                  extras:
                    storage: {tables: [warehouse.g_day]}
                    semantic_mapping: {date: date, campaign_id: campaign_id, country: country, avg_clicks: clicks}
                    temporal: {events: {event_time: date}, granularity: day}
                - name: g_month
                  extras:
                    storage: {tables: [warehouse.g_month]}
                    semantic_mapping: {date: date, campaign_id: campaign_id, country: country, budget: budget}
                    temporal: {events: {event_time: date}, granularity: month}
      relationships:
        - {name: u_to_campaigns, from: u, to: campaigns, on: [campaign_id], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

#[tokio::test]
async fn a_grainset_branch_of_a_unionset_operand_takes_the_parts() {
    let tables = [
        G_CAMPAIGNS_TABLES,
        &[(
            "g_extra",
            "date date, campaign_id string, country string, budget double",
        )],
    ]
    .concat();
    let manifest = over(U_CAMPAIGNS, &tables).await;

    let (plan, advisories) = planned(&manifest, &select(&["budget_eu"]));

    let (step_2, _, step_1) = parts(&plan.root);
    let u = alias(&manifest, "u_campaigns.u");
    let u_step = first_aggregate(&join(&projection(&step_1.input).input).left);
    assert_eq!(u_step.groupings, [qualified(&u, "campaign_id")]);
    assert!(step_2.groupings.is_empty(), "{:?}", step_2.groupings);
    let [key] = step_1.groupings.as_slice() else {
        panic!("the part key alone, got {:?}", step_1.groupings);
    };
    let campaigns = alias(&manifest, "u_campaigns.campaigns");
    assert_eq!(
        computed(projection(&step_1.input), key),
        Expr::Case {
            whens: vec![(qualified(&campaigns, "region").eq(text("eu")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    assert_eq!(output_names(&plan), ["budget_eu"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn the_pick_of_a_constituent_grainset_buckets_at_the_granularity_it_declares() {
    let manifest = over(G_CAMPAIGNS, G_CAMPAIGNS_TABLES).await;
    let names = ["date", "avg_clicks"];

    let (plan, advisories) = planned(&manifest, &select(&names));

    assert_eq!(tables_read(&plan), ["g_day"]);
    let month = call("date_trunc", vec![text("month"), column("date")]);
    let bucketing = fields(projection(&aggregate_step(&plan.root).input));
    assert!(bucketing.contains(&("date", month)), "{bucketing:?}");
    assert_eq!(output_names(&plan), names);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_constituent_grainset_drills_across_at_the_granularity_it_declares() {
    let quarter = (G_CAMPAIGNS.replace(
        "granularities: [day, month]",
        "granularities: [day, month, quarter]",
    ))
    .replace(
        "month}\n          datasets:",
        "quarter}\n          datasets:",
    );
    let manifest = over(&quarter, G_CAMPAIGNS_TABLES).await;
    let names = ["date", "campaign_id", "avg_clicks", "budget"];

    let (plan, advisories) = planned(&manifest, &select(&names));

    let [g_month, g_day] =
        ["g_month", "g_day"].map(|child| alias(&manifest, &format!("g_campaigns.g.{child}")));
    let drill = drill_across(&plan, &g_month, &g_day, &["date", "campaign_id"]);
    for (side, child) in [(&drill.left, &g_month), (&drill.right, &g_day)] {
        let bucket = call(
            "date_trunc",
            vec![text("quarter"), qualified(child, "date")],
        );
        let bucketing = projection(&first_aggregate(side).input);
        assert_eq!(written(bucketing, &format!("{child}.date")), Some(bucket));
    }
    assert_eq!(output_names(&plan), names);
    assert!(advisories.is_empty());
}

const CHANNEL_SPEND: &str = r#"
semantic_model:
  name: channel_spend
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: channel, data_type: string, type: categorical}
    - {name: channel_group, data_type: string, type: categorical}
  measures:
    - {name: total_budget, data_type: double, agg: sum}
    - {name: avg_budget, data_type: double, agg: avg}
    - {name: cost_m, data_type: double, agg: sum}
  joinsets:
    - name: channel_spend
      dimensions:
        - {ref: date, is_key: true, keysets: [g]}
        - {ref: campaign_id, is_key: true, keysets: [campaign, g]}
        - {ref: channel, is_key: true, keysets: [channel, g]}
        - {ref: channel_group}
      measures: [{ref: total_budget}, {ref: avg_budget}, {ref: cost_m}]
      datasets:
        - name: channels
          extras:
            storage: {tables: [warehouse.channels]}
      grainsets:
        - name: g
          extras:
            temporal: {events: {event_time: date}, granularity: month}
          datasets:
            - name: c_month
              extras:
                storage: {tables: [warehouse.monthly_spend]}
                semantic_mapping: {date: date, campaign_id: campaign_id, channel: channel, cost_m: cost}
                temporal: {events: {event_time: date}, granularity: month}
          joinsets:
            - name: c_day
              datasets:
                - name: daily_spend
                  extras:
                    storage: {tables: [warehouse.daily_spend]}
                    semantic_mapping: {date: date, campaign_id: campaign_id, channel: channel}
                    temporal: {events: {event_time: date}, granularity: day}
                - name: campaigns
                  extras:
                    storage: {tables: [warehouse.campaigns]}
                    semantic_mapping: {campaign_id: campaign_id, total_budget: budget, avg_budget: budget}
              relationships:
                - {name: spend_to_campaigns, from: daily_spend, to: campaigns, on: [campaign_id], cardinality: many_to_one, optional: none, integrity: assumed}
      relationships:
        - {name: g_to_channels, from: g, to: channels, on: [channel], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

const CHANNEL_SPEND_TABLES: &[(&str, &str)] = &[
    (
        "daily_spend",
        "date date, campaign_id string, channel string",
    ),
    (
        "monthly_spend",
        "date date, campaign_id string, channel string, cost double",
    ),
    ("campaigns", "campaign_id string, budget double"),
    ("channels", "channel string, channel_group string"),
];

#[tokio::test]
async fn a_grainset_operand_kept_alone_marks_the_rows_that_its_drill_across_child_repeats() {
    let manifest = over(CHANNEL_SPEND, CHANNEL_SPEND_TABLES).await;
    let [c_month, c_day, campaigns] = ["c_month", "c_day", "c_day.campaigns"]
        .map(|unit| alias(&manifest, &format!("channel_spend.g.{unit}")));
    let names = ["channel", "total_budget", "cost_m"];

    let (plan, advisories) = planned(&manifest, &select(&names));

    assert_eq!(
        tables_read(&plan),
        ["monthly_spend", "daily_spend", "campaigns"]
    );
    let drill = drill_across(&plan, &c_month, &c_day, &["channel"]);
    let step = first_aggregate(&drill.right);
    let ends = join(&projection(&step.input).input);
    for end in [&ends.left, &ends.right] {
        assert_eq!(aggregate_steps(end), 1, "partial aggregates");
    }
    assert!(every_marker(&plan.root).is_empty());
    assert_eq!(output_names(&plan), names);
    assert!(advisories.is_empty());

    let (marked, advisories) = planned(&manifest, &select(&["channel", "avg_budget", "cost_m"]));

    let drill = drill_across(&marked, &c_month, &c_day, &["channel"]);
    let [(marker, window)] = &every_marker(&drill.right)[..] else {
        panic!("one marker in `c_day`");
    };
    assert_eq!(every_marker(&marked.root).len(), 1);
    let step = first_aggregate(&drill.right);
    let avg_budget = (step.aggregates.iter())
        .find_map(|(name, call)| name.as_str().ends_with("avg_budget").then_some(call))
        .expect("`avg_budget`'s entry");
    let field = marker
        .strip_prefix(&format!("{c_day}."))
        .expect("`c_day`'s field");
    assert_eq!(
        avg_budget.filter(),
        Some(&qualified(&c_day, field).eq(bigint(1)))
    );
    let partition = vec![
        qualified(&c_day, "channel"),
        qualified(&campaigns, "campaign_id"),
        avg_budget.args()[0].clone(),
    ];
    assert_eq!(window, &row_number(partition));
    assert!(advisories.is_empty());
}

const G_ADS: &str = r#"
semantic_model:
  name: g_ads
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: country, data_type: string, type: categorical}
    - {name: ad_id, data_type: string, type: categorical}
    - {name: ad_type, data_type: string, type: categorical}
  measures:
    - {name: avg_budget, data_type: double, agg: avg}
    - {name: clicks, data_type: double, agg: sum}
  joinsets:
    - name: g_ads
      dimensions:
        - {ref: date, is_key: true, keysets: [spend]}
        - {ref: campaign_id, is_key: true, keysets: [spend]}
        - {ref: country}
        - {ref: ad_id, is_key: true, keysets: [ad]}
        - {ref: ad_type}
      measures: [{ref: avg_budget}, {ref: clicks}]
      datasets:
        - name: ads
          extras:
            storage: {tables: [warehouse.ads]}
      grainsets:
        - name: g
          extras:
            temporal: {events: {event_time: date}, granularity: month}
          datasets:
            - name: g_day
              extras:
                storage: {tables: [warehouse.g_day]}
                semantic_mapping: {date: date, campaign_id: campaign_id, country: country, clicks: clicks}
                temporal: {events: {event_time: date}, granularity: day}
            - name: g_month
              extras:
                storage: {tables: [warehouse.g_month]}
                semantic_mapping: {date: date, campaign_id: campaign_id, avg_budget: budget}
                temporal: {events: {event_time: date}, granularity: month}
      relationships:
        - {name: g_to_ads, from: g, to: ads, on: [campaign_id], cardinality: one_to_many, optional: none, integrity: assumed}
"#;

const G_ADS_TABLES: &[(&str, &str)] = &[
    (
        "g_day",
        "date date, campaign_id string, country string, clicks double",
    ),
    ("g_month", "date date, campaign_id string, budget double"),
    ("ads", "ad_id string, campaign_id string, ad_type string"),
];

#[tokio::test]
async fn a_fanned_grainset_operand_s_row_identity_is_the_keyset_of_its_answering_child() {
    let manifest = over(G_ADS, G_ADS_TABLES).await;

    let (plan, advisories) = planned(&manifest, &select(&["ad_type", "avg_budget"]));

    assert_eq!(tables_read(&plan), ["g_month", "ads"]);
    let aggregate = first_aggregate(&plan.root);
    let [(_, avg_budget)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    assert_eq!(avg_budget.filter(), Some(&once(marker)));
    let mut identity = ["date", "campaign_id"];
    identity.sort_by_key(|name| {
        let definition = manifest.semantic_definitions.by_name(name);
        definition.expect("the Dimension").bit_position
    });
    let g = alias(&manifest, "g_ads.g");
    let partition = (std::iter::once(column("ad_type")))
        .chain(identity.iter().map(|name| qualified(&g, name)))
        .chain([avg_budget.args()[0].clone()])
        .collect();
    assert_eq!(window, &row_number(partition));
    assert!(advisories.is_empty(), "no `FanOut`: {advisories:?}");
}

fn marker_not_placeable(
    manifest: &SemanticManifest,
    request: &Request,
    semantics: &str,
    path: &[&str],
) -> Diagnostic<PlanError> {
    let (error, advisories) = refusal(manifest, request);
    let PlanError::FanOutMarkerNotPlaceable {
        semantics: of,
        path: at,
    } = &error.kind
    else {
        panic!("FanOutMarkerNotPlaceable, got {error:?}");
    };
    assert_eq!(of.as_str(), semantics);
    assert_eq!(at.to_string(), path.join(" / "));
    assert!(advisories.is_empty());
    error
}

const ALL_SALES: &str = r#"
semantic_model:
  name: all_sales
  dimensions:
    - {name: order_id, data_type: string, type: categorical}
    - {name: kind, data_type: string, type: categorical}
  measures:
    - {name: avg_amount, data_type: double, agg: avg}
  unionsets:
    - name: all_sales
      mode: all
      dimensions: [{ref: order_id}, {ref: kind}]
      measures: [{ref: avg_amount}]
      datasets:
        - name: store_sales
          extras:
            storage: {tables: [warehouse.store_sales]}
            semantic_mapping: {kind: kind, avg_amount: amount}
      joinsets:
        - name: order_items
          datasets:
            - name: orders
              extras:
                storage: {tables: [warehouse.orders]}
                semantic_mapping: {order_id: order_id, avg_amount: amount}
            - name: order_lines
              extras:
                storage: {tables: [warehouse.order_lines]}
                semantic_mapping: {order_id: order_id, kind: kind}
          relationships:
            - {name: lines, from: orders, to: order_lines, on: [order_id], cardinality: one_to_many, optional: none, integrity: assumed}
"#;

const ALL_SALES_TABLES: &[(&str, &str)] = &[
    ("store_sales", "sale_id string, kind string, amount double"),
    ("orders", "order_id string, amount double"),
    (
        "order_lines",
        "line_id string, order_id string, kind string",
    ),
];

async fn all_sales_rule_2() -> Golden {
    let manifest = over(ALL_SALES, ALL_SALES_TABLES).await;
    let request = select(&["kind", "avg_amount"]);
    let path = ["all_sales", "all_sales.order_items"];
    let refused = marker_not_placeable(&manifest, &request, "avg_amount", &path);
    Golden::message("all_sales_rule_2", &refused)
}

#[tokio::test]
async fn rows_that_a_join_repeats_below_a_union_have_no_marker_above_it() {
    all_sales_rule_2().await.assert_committed();
}

#[tokio::test]
async fn a_result_with_a_leaf_that_no_join_repeats_reads_the_rows_of_a_branch_as_they_are() {
    let yaml = [
        (
            "  measures:\n",
            "    - {name: order_amount, data_type: double, type: categorical}\n    - {name: line_amount, data_type: double, type: categorical}\n  measures:\n    - {name: avg_total, data_type: double, agg: avg, expr: {add: [order_amount, line_amount]}}\n",
        ),
        (
            "dimensions: [{ref: order_id}, {ref: kind}]",
            "dimensions: [{ref: order_id}, {ref: kind}, {ref: order_amount}, {ref: line_amount}]",
        ),
        ("measures: [{ref: avg_amount}]", "measures: [{ref: avg_amount}, {ref: avg_total}]"),
        (
            "{kind: kind, avg_amount: amount}",
            "{kind: kind, avg_amount: amount, order_amount: amount, line_amount: amount}",
        ),
        ("{order_id: order_id, avg_amount: amount}", "{order_id: order_id, avg_amount: amount, order_amount: amount}"),
        ("{order_id: order_id, kind: kind}", "{order_id: order_id, kind: kind, line_amount: line_amount}"),
    ]
    .iter()
    .fold(ALL_SALES.to_owned(), |yaml, (from, to)| yaml.replacen(from, to, 1));
    let tables = [
        ALL_SALES_TABLES[0],
        ALL_SALES_TABLES[1],
        (
            "order_lines",
            "line_id string, order_id string, kind string, line_amount double",
        ),
    ];
    let manifest = over(&yaml, &tables).await;

    let (plan, _) = planned(&manifest, &select(&["kind", "avg_total"]));

    assert!(every_marker(&plan.root).is_empty(), "{plan:#?}");
    assert_eq!(output_names(&plan), ["kind", "avg_total"]);
}

const REGION_SALES: &str = r#"
semantic_model:
  name: region_sales
  dimensions:
    - {name: order_id, data_type: string, type: categorical}
    - {name: kind, data_type: string, type: categorical}
    - {name: store_id, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - {name: avg_amount, data_type: double, agg: avg}
  joinsets:
    - name: region_sales
      dimensions: [{ref: order_id}, {ref: kind}, {ref: store_id}, {ref: region}]
      measures: [{ref: avg_amount}]
      unionsets:
        - name: all_sales
          mode: all
          datasets:
            - name: store_sales
              extras:
                storage: {tables: [warehouse.store_sales]}
                semantic_mapping: {store_id: store_id, kind: kind, avg_amount: amount}
          joinsets:
            - name: order_items
              datasets:
                - name: orders
                  extras:
                    storage: {tables: [warehouse.orders]}
                    semantic_mapping: {order_id: order_id, store_id: store_id, avg_amount: amount}
                - name: order_lines
                  extras:
                    storage: {tables: [warehouse.order_lines]}
                    semantic_mapping: {order_id: order_id, kind: kind}
              relationships:
                - {name: lines, from: orders, to: order_lines, on: [order_id], cardinality: one_to_many, optional: none, integrity: assumed}
      datasets:
        - name: stores
          extras:
            storage: {tables: [warehouse.stores]}
            semantic_mapping: {store_id: store_id, region: region}
      relationships:
        - {name: sales_to_stores, from: all_sales, to: stores, on: [store_id], cardinality: many_to_one, optional: none, integrity: assumed}
"#;

#[tokio::test]
async fn rows_that_a_join_repeats_reach_the_step_above_through_a_union_and_are_refused() {
    let tables = [
        ("store_sales", "store_id string, kind string, amount double"),
        ("orders", "order_id string, store_id string, amount double"),
        (
            "order_lines",
            "line_id string, order_id string, kind string",
        ),
        ("stores", "store_id string, region string"),
    ];
    let manifest = over(REGION_SALES, &tables).await;
    let request = select(&["region", "kind", "avg_amount"]);
    let path = [
        "region_sales",
        "region_sales.all_sales",
        "region_sales.all_sales.order_items",
    ];

    marker_not_placeable(&manifest, &request, "avg_amount", &path);
}

#[tokio::test]
async fn a_result_with_a_leaf_that_no_join_repeats_reaches_the_step_above_a_union_as_it_is() {
    let yaml = [
        (
            "  measures:\n",
            "    - {name: order_amount, data_type: double, type: categorical}\n    - {name: line_amount, data_type: double, type: categorical}\n  measures:\n    - {name: avg_total, data_type: double, agg: avg, expr: {add: [order_amount, line_amount]}}\n",
        ),
        (
            "{ref: store_id}, {ref: region}]",
            "{ref: store_id}, {ref: region}, {ref: order_amount}, {ref: line_amount}]",
        ),
        ("measures: [{ref: avg_amount}]", "measures: [{ref: avg_amount}, {ref: avg_total}]"),
        (
            "kind: kind, avg_amount: amount}",
            "kind: kind, avg_amount: amount, order_amount: amount, line_amount: amount}",
        ),
        ("store_id: store_id, avg_amount: amount}", "store_id: store_id, avg_amount: amount, order_amount: amount}"),
        ("{order_id: order_id, kind: kind}", "{order_id: order_id, kind: kind, line_amount: line_amount}"),
    ]
    .iter()
    .fold(REGION_SALES.to_owned(), |yaml, (from, to)| yaml.replacen(from, to, 1));
    let tables = [
        ("store_sales", "store_id string, kind string, amount double"),
        ("orders", "order_id string, store_id string, amount double"),
        (
            "order_lines",
            "line_id string, order_id string, kind string, line_amount double",
        ),
        ("stores", "store_id string, region string"),
    ];
    let manifest = over(&yaml, &tables).await;

    let (plan, _) = planned(&manifest, &select(&["region", "avg_total"]));

    assert!(every_marker(&plan.root).is_empty(), "{plan:#?}");
    assert_eq!(output_names(&plan), ["region", "avg_total"]);
}

const UNION_CUSTOMERS: &str = r#"
semantic_model:
  name: union_customers
  dimensions:
    - {name: customer_id, data_type: bigint, type: categorical}
    - {name: region, data_type: string, type: categorical}
    - {name: segment, data_type: string, type: categorical}
  measures:
    - {name: revenue, data_type: double, agg: sum}
  metrics:
    - name: vip_weighted_revenue
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [segment, {lit: vip}]}
              then: {multiply: [revenue, {lit: 2.0}]}
          else: revenue
  joinsets:
    - name: union_customers
      dimensions: [{ref: customer_id}, {ref: region}, {ref: segment}]
      measures: [{ref: revenue}]
      metrics: [{ref: vip_weighted_revenue}]
      unionsets:
        - name: orders_u
          mode: all
          datasets:
            - name: orders_eu
              extras:
                storage: {tables: [warehouse.orders_eu]}
                semantic_mapping: {customer_id: cust_key, region: region_code, revenue: amount}
            - name: orders_us
              extras:
                storage: {tables: [warehouse.orders_us]}
                semantic_mapping: {customer_id: cust_key, region: region_code, revenue: amount}
      datasets:
        - name: customers
          extras:
            storage: {tables: [warehouse.customers]}
      relationships:
        - {name: orders_to_customers, from: orders_u, to: customers, on: [customer_id], cardinality: many_to_one, optional: left}
"#;

#[tokio::test]
async fn the_formula_push_downs_walk_enters_each_branch_of_a_unionset_operand() {
    let orders = "cust_key long, region_code string, amount double";
    let tables = [
        ("orders_eu", orders),
        ("orders_us", orders),
        ("customers", "customer_id long, segment string"),
    ];
    let manifest = over(UNION_CUSTOMERS, &tables).await;

    let (plan, advisories) = planned(&manifest, &select(&["region", "vip_weighted_revenue"]));

    assert_eq!(aggregate_steps(&plan.root), 1);
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [orders_u, customers] = ["orders_u", "customers"]
        .map(|operand| alias(&manifest, &format!("union_customers.{operand}")));
    let rows = projection(&aggregate.input);
    let revenue = qualified(&orders_u, "revenue");
    let term = Expr::Case {
        whens: vec![(
            qualified(&customers, "segment").eq(text("vip")),
            revenue.clone() * Expr::Literal(Literal::Double(2.0)),
        )],
        else_: Some(Box::new(revenue)),
    };
    let sum = entry(aggregate, "vip_weighted_revenue");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let [argument] = sum.args() else {
        panic!("one argument, got {:?}", sum.args());
    };
    assert_eq!(computed(rows, argument), term);
    let condition = sum.filter().expect("the condition that the branches give");
    assert_eq!(
        computed(rows, condition),
        qualified(&orders_u, "__plan_cond_revenue")
    );
    join(&rows.input);
    let (_, union) = union_of(&plan);
    assert_eq!(union.inputs.len(), 2);
    assert_eq!(tables_read(&plan), ["orders_eu", "orders_us", "customers"]);
    assert_eq!(output_names(&plan), ["region", "vip_weighted_revenue"]);
    assert!(advisories.is_empty());
}

async fn union_customers_adj(site: &str) -> SemanticManifest {
    let metric = "    - name: revenue_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
  joinsets:
";
    let yaml = (UNION_CUSTOMERS.replacen("  joinsets:\n", metric, 1))
        .replacen(
            "metrics: [{ref: vip_weighted_revenue}]",
            "metrics: [{ref: vip_weighted_revenue}, {ref: revenue_adj}]",
            1,
        )
        .replacen(
            "          mode: all\n",
            &format!("          mode: all\n{site}"),
            1,
        );
    let orders = "cust_key long, region_code string, amount double";
    let tables = [
        ("orders_eu", orders),
        ("orders_us", orders),
        ("customers", "customer_id long, segment string"),
    ];
    over(&yaml, &tables).await
}

#[tokio::test]
async fn a_unionset_operand_pre_aggregates_the_formula_push_downs_sum_that_reads_it_alone() {
    let manifest = union_customers_adj("").await;

    let (plan, _) = planned(&manifest, &select(&["segment", "revenue_adj"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("segment")]);
    let orders_u = alias(&manifest, "union_customers.orders_u");
    let rows = projection(&aggregate.input);
    let sum = entry(aggregate, "revenue_adj");
    assert_eq!(
        computed(rows, &sum.args()[0]),
        qualified(&orders_u, "revenue_adj")
    );
    let partials = aggregate_step(&join(&rows.input).left);
    assert_eq!(partials.groupings, [qualified(&orders_u, "customer_id")]);
    let (_, union) = union_of(&plan);
    for branch in &union.inputs {
        assert_eq!(
            aggregate_steps(branch),
            0,
            "a branch gives rows: {branch:?}"
        );
    }
    assert_eq!(output_names(&plan), ["segment", "revenue_adj"]);
}

#[tokio::test]
async fn a_unionset_operand_pre_aggregates_the_formula_push_down_under_its_branches_condition() {
    let site = "          measures: [{ref: revenue, filter: {eq: [region, {lit: eu}]}}]\n";
    let manifest = union_customers_adj(site).await;

    let (plan, _) = planned(&manifest, &select(&["segment", "revenue_adj"]));

    let orders_u = alias(&manifest, "union_customers.orders_u");
    let rows = projection(&aggregate_step(&plan.root).input);
    let partials = aggregate_step(&join(&rows.input).left);
    let sum = entry(partials, &format!("{orders_u}.revenue_adj"));
    let condition = sum.filter().expect("the condition that the branches give");
    assert_eq!(
        computed(projection(&partials.input), condition),
        qualified(&orders_u, "__plan_cond_revenue")
    );
    assert_eq!(output_names(&plan), ["segment", "revenue_adj"]);
}

const G_LOSSY: &str = r#"
semantic_model:
  name: g_lossy
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: campaign_id, data_type: string, type: categorical}
  measures:
    - {name: avg_cpc, data_type: double, agg: avg}
    - {name: clicks, data_type: bigint, agg: sum}
  grainsets:
    - name: g_lossy
      dimensions: [{ref: date, is_key: true}, {ref: campaign_id, is_key: true}]
      measures: [{ref: avg_cpc}, {ref: clicks}]
      datasets:
        - name: cpc_year
          extras:
            storage: {tables: [warehouse.cpc_year]}
            semantic_mapping: {date: year_start, campaign_id: campaign_id, avg_cpc: cpc, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: year}
      unionsets:
        - name: u_cpc
          mode: all
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: cpc_web
              extras:
                storage: {tables: [warehouse.cpc_web]}
                semantic_mapping: {date: day, campaign_id: campaign_id, avg_cpc: cpc, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: cpc_store
              extras:
                storage: {tables: [warehouse.cpc_store]}
                semantic_mapping: {date: day, campaign_id: campaign_id, avg_cpc: cpc, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;

const G_LOSSY_TABLES: &[(&str, &str)] = &[
    (
        "cpc_year",
        "year_start date, campaign_id string, cpc double, clicks long",
    ),
    (
        "cpc_web",
        "day date, campaign_id string, cpc double, clicks long",
    ),
    (
        "cpc_store",
        "day date, campaign_id string, cpc double, clicks long",
    ),
];

#[tokio::test]
async fn a_grainset_child_that_declares_month_over_day_values_re_aggregates_a_lossy_measure() {
    let manifest = over(G_LOSSY, G_LOSSY_TABLES).await;
    let request = at_granularity("date", Granularity::Month, &["campaign_id", "avg_cpc"]);

    let (plan, advisories) = planned(&manifest, &request);

    assert_eq!(tables_read(&plan), ["cpc_store", "cpc_web"]);
    assert_eq!(
        lossy(&advisories),
        [("g_lossy / g_lossy.u_cpc".to_owned(), "avg_cpc")]
    );
}

fn uc(branches: &str) -> String {
    format!(
        r#"
semantic_model:
  name: uc
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {{name: campaign_id, data_type: string, type: categorical}}
    - {{name: fiscal_week, data_type: string, type: categorical}}
  measures:
    - {{name: avg_cpc, data_type: double, agg: avg}}
  joinsets:
    - name: uc
      dimensions: [{{ref: date}}, {{ref: campaign_id}}, {{ref: fiscal_week}}]
      measures: [{{ref: avg_cpc}}]
      unionsets:
        - name: u2
          mode: all
          extras:
            temporal: {{timeseries: {{occurred_at: date}}, granularity: month}}
          datasets:
            - name: c_web
              extras:
                storage: {{tables: [warehouse.c_web]}}
                semantic_mapping: {{date: date, campaign_id: campaign_id, avg_cpc: cpc}}
                temporal: {{timeseries: {{occurred_at: date}}, granularity: {branches}}}
            - name: c_store
              extras:
                storage: {{tables: [warehouse.c_store]}}
                semantic_mapping: {{date: date, campaign_id: campaign_id, avg_cpc: cpc}}
                temporal: {{timeseries: {{occurred_at: date}}, granularity: {branches}}}
      datasets:
        - name: calendar
          extras:
            storage: {{tables: [warehouse.calendar]}}
            semantic_mapping: {{date: date, fiscal_week: fiscal_week}}
      relationships:
        - {{name: rel, from: u2, to: calendar, on: [date], cardinality: many_to_one, optional: left, integrity: assumed}}
"#
    )
}

const UC_TABLES: &[(&str, &str)] = &[
    ("c_web", "campaign_id string, date date, cpc double"),
    ("c_store", "campaign_id string, date date, cpc double"),
    ("calendar", "date date, fiscal_week string"),
];

#[tokio::test]
async fn a_joinset_holds_the_month_of_its_union_operand_only_where_each_branch_holds_it() {
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign_id", "fiscal_week", "avg_cpc"],
    );
    let lossy_at_uc = [("uc".to_owned(), "avg_cpc")];
    for (branches, advised) in [("day", &lossy_at_uc[..]), ("month", &[])] {
        let manifest = over(&uc(branches), UC_TABLES).await;

        let (_, advisories) = planned(&manifest, &request);

        assert_eq!(lossy(&advisories), advised, "{branches} branches");
    }
}

const W2: &str = r#"
semantic_model:
  name: w2
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: campaign_id, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: target, data_type: bigint, agg: sum}
  unionsets:
    - name: w2
      mode: all
      dimensions: [{ref: date}, {ref: campaign_id}]
      measures: [{ref: clicks}, {ref: target}]
      datasets:
        - name: store
          extras:
            storage: {tables: [warehouse.store]}
            semantic_mapping: {campaign_id: campaign_id, date: month_start, clicks: clicks}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
      joinsets:
        - name: j5
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: spend
              extras:
                storage: {tables: [warehouse.spend]}
                semantic_mapping: {campaign_id: campaign_id, date: day, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: targets
              extras:
                storage: {tables: [warehouse.targets]}
                semantic_mapping: {campaign_id: campaign_id, target: target}
          relationships:
            - {name: spend_to_targets, from: spend, to: targets, on: [campaign_id], cardinality: many_to_one, optional: left, integrity: assumed}
"#;

const W2_TABLES: &[(&str, &str)] = &[
    ("spend", "day date, campaign_id string, clicks long"),
    ("targets", "campaign_id string, target long"),
    ("store", "month_start date, campaign_id string, clicks long"),
];

#[tokio::test]
async fn a_joinset_constituent_that_declares_month_over_day_values_refuses_a_fanned_measure() {
    let manifest = over(W2, W2_TABLES).await;
    let request = at_granularity("date", Granularity::Month, &["target"]);

    marker_not_placeable(&manifest, &request, "target", &["w2", "w2.j5"]);
}

const W4: &str = r#"
semantic_model:
  name: w4
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: campaign_id, data_type: string, type: categorical}
  measures:
    - {name: impressions, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
    - {name: target, data_type: bigint, agg: sum}
  unionsets:
    - name: w4
      mode: all
      dimensions: [{ref: date}, {ref: campaign_id}]
      measures: [{ref: impressions}, {ref: budget}, {ref: target}]
      datasets:
        - name: store
          extras:
            storage: {tables: [warehouse.store]}
            semantic_mapping: {date: month_start, campaign_id: campaign_id, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
      joinsets:
        - name: jg
          grainsets:
            - name: g
              dimensions: [{ref: date, is_key: true}, {ref: campaign_id, is_key: true}]
              extras:
                temporal: {timeseries: {occurred_at: date}, granularity: month}
              datasets:
                - name: g_day
                  extras:
                    storage: {tables: [warehouse.g_day]}
                    semantic_mapping: {date: day, campaign_id: campaign_id, impressions: impressions}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
                - name: g_month
                  extras:
                    storage: {tables: [warehouse.g_month]}
                    semantic_mapping: {date: month_start, campaign_id: campaign_id, budget: budget}
                    temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: campaigns
              extras:
                storage: {tables: [warehouse.campaigns]}
                semantic_mapping: {campaign_id: campaign_id, target: target}
          relationships:
            - {name: g_to_campaigns, from: g, to: campaigns, on: [campaign_id], cardinality: many_to_one, optional: left, integrity: assumed}
"#;

const W4_TABLES: &[(&str, &str)] = &[
    ("g_day", "day date, campaign_id string, impressions long"),
    (
        "g_month",
        "month_start date, campaign_id string, budget long",
    ),
    ("campaigns", "campaign_id string, target long"),
    ("store", "month_start date, campaign_id string, budget long"),
];

#[tokio::test]
async fn a_grainset_constituent_holds_its_time_axis_at_the_granularity_of_the_child_it_picks() {
    let manifest = over(W4, W4_TABLES).await;
    for names in [&["target"][..], &["campaign_id", "target"]] {
        let request = at_granularity("date", Granularity::Month, names);

        let (plan, _) = planned(&manifest, &request);

        assert_eq!(tables_read(&plan), ["g_month", "campaigns", "store"]);
        assert_eq!(every_marker(&plan.root), []);
    }
}

const W6: &str = r#"
semantic_model:
  name: w6
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: campaign_id, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: views, data_type: bigint, agg: sum}
    - {name: budget, data_type: bigint, agg: sum}
    - {name: target, data_type: bigint, agg: sum}
  unionsets:
    - name: w6
      mode: all
      dimensions: [{ref: date}, {ref: campaign_id}]
      measures: [{ref: clicks}, {ref: views}, {ref: budget}, {ref: target}]
      datasets:
        - name: store
          extras:
            storage: {tables: [warehouse.store]}
            semantic_mapping: {date: month_start, campaign_id: campaign_id, budget: budget}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
      joinsets:
        - name: jp
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: month_spend
              extras:
                storage: {tables: [warehouse.month_spend]}
                semantic_mapping: {date: month_start, campaign_id: campaign_id, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
            - name: day_views
              extras:
                storage: {tables: [warehouse.day_views]}
                semantic_mapping: {date: day, campaign_id: campaign_id, views: views}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
            - name: targets
              extras:
                storage: {tables: [warehouse.targets]}
                semantic_mapping: {campaign_id: campaign_id, target: target}
          relationships:
            - {name: spend_to_views, from: month_spend, to: day_views, on: [campaign_id], cardinality: one_to_many, optional: left, integrity: assumed}
            - {name: spend_to_targets, from: month_spend, to: targets, on: [campaign_id], cardinality: many_to_one, optional: left, integrity: assumed}
"#;

const W6_TABLES: &[(&str, &str)] = &[
    (
        "month_spend",
        "month_start date, campaign_id string, clicks long",
    ),
    ("day_views", "day date, campaign_id string, views long"),
    ("targets", "campaign_id string, target long"),
    ("store", "month_start date, campaign_id string, budget long"),
];

#[tokio::test]
async fn a_joinset_that_passes_through_holds_its_time_axis_as_the_operand_it_reads() {
    let manifest = over(W6, W6_TABLES).await;
    for names in [&["target"][..], &["campaign_id", "target"]] {
        let request = at_granularity("date", Granularity::Month, names);

        let (plan, _) = planned(&manifest, &request);

        assert_eq!(tables_read(&plan), ["month_spend", "targets", "store"]);
        assert_eq!(every_marker(&plan.root), []);
    }
}

const W7: &str = r#"
semantic_model:
  name: w7
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: campaign_id, data_type: string, type: categorical}
    - {name: fiscal_month, data_type: string, type: categorical}
  measures:
    - {name: clicks, data_type: bigint, agg: sum}
    - {name: impressions, data_type: bigint, agg: sum}
    - {name: target, data_type: bigint, agg: sum}
  unionsets:
    - name: w7
      mode: all
      dimensions: [{ref: date}, {ref: campaign_id}, {ref: fiscal_month}]
      measures: [{ref: clicks}, {ref: impressions}, {ref: target}]
      datasets:
        - name: store
          extras:
            storage: {tables: [warehouse.store]}
            semantic_mapping: {date: month_start, campaign_id: campaign_id, impressions: impressions}
            temporal: {timeseries: {occurred_at: date}, granularity: month}
      joinsets:
        - name: jc
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: ad_clicks
              extras:
                storage: {tables: [warehouse.ad_clicks]}
                semantic_mapping: {date: month_start, campaign_id: campaign_id, clicks: clicks}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
            - name: ad_views
              extras:
                storage: {tables: [warehouse.ad_views]}
                semantic_mapping: {date: month_start, campaign_id: campaign_id, impressions: impressions}
                temporal: {timeseries: {occurred_at: date}, granularity: month}
            - name: calendar
              extras:
                storage: {tables: [warehouse.calendar]}
                semantic_mapping: {date: month_start, fiscal_month: fiscal_month}
            - name: targets
              extras:
                storage: {tables: [warehouse.targets]}
                semantic_mapping: {campaign_id: campaign_id, target: target}
          relationships:
            - {name: clicks_to_views, from: ad_clicks, to: ad_views, on: [date, campaign_id], cardinality: one_to_one, optional: none, integrity: assumed, cross_filter: both}
            - {name: views_to_calendar, from: ad_views, to: calendar, on: [date], cardinality: many_to_one, optional: left, integrity: assumed}
            - {name: clicks_to_targets, from: ad_clicks, to: targets, on: [campaign_id], cardinality: many_to_one, optional: left, integrity: assumed}
"#;

const W7_TABLES: &[(&str, &str)] = &[
    (
        "ad_clicks",
        "month_start date, campaign_id string, clicks long",
    ),
    (
        "ad_views",
        "month_start date, campaign_id string, impressions long",
    ),
    ("calendar", "month_start date, fiscal_month string"),
    ("targets", "campaign_id string, target long"),
    (
        "store",
        "month_start date, campaign_id string, impressions long",
    ),
];

#[tokio::test]
async fn a_joinset_holds_its_time_axis_through_the_equality_of_an_inner_join_below_it() {
    let manifest = over(W7, W7_TABLES).await;
    let names = ["fiscal_month", "impressions", "target"];
    let request = at_granularity("date", Granularity::Month, &names);

    let (plan, _) = planned(&manifest, &request);

    assert_eq!(
        tables_read(&plan),
        ["ad_clicks", "ad_views", "calendar", "targets", "store"]
    );
    assert_eq!(every_marker(&plan.root), []);
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            all_sales_rule_2().await,
            composition_at_day().await,
            grainset_branch_at_day().await,
            grainset_operand_of_a_built_join().await,
        ]
    })
    .await;
}
