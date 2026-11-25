mod common;

use common::golden::{self, Golden};
use common::{
    at_granularity, compiled, grain_not_servable, lossy, planned, read_table, refusal, select,
    with_semantics, Catalog,
};
use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::DataKindName;
use semstrait_common::Granularity;
use semstrait_ir::{GranularityGroup, SemAnnotation};
use semstrait_manifest::SemanticManifest;
use semstrait_planner::{DimensionModifier, PlanError, Request, Selection};

const SEMANTICS: &str = r#"
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
"#;

const SPEND_DAILY: &str = r#"
    - name: spend_daily
      extras:
        storage:
          tables: [warehouse.spend_daily]
        temporal:
          timeseries:
            occurred_at: date
          granularity: day
      dimensions:
        - ref: date
          is_key: true
          keysets: [natural]
        - ref: campaign
          is_key: true
          keysets: [natural]
      measures:
        - ref: clicks
"#;

const SPEND_MONTHLY: &str = r#"
    - name: spend_monthly
      extras:
        storage:
          tables: [warehouse.spend_monthly]
        temporal:
          timeseries:
            occurred_at: date
          granularity: month
      dimensions:
        - ref: date
          is_key: true
          keysets: [natural]
        - ref: campaign
          is_key: true
          keysets: [natural]
      measures:
        - ref: clicks
        - ref: budget
"#;

const BUDGET_MONTHLY: &str = r#"
    - name: budget_monthly
      extras:
        storage:
          tables: [warehouse.budget_monthly]
        temporal:
          timeseries:
            occurred_at: date
          granularity: month
      dimensions:
        - ref: date
        - ref: campaign
      measures:
        - ref: budget
"#;

const SPEND_MONTHLY_CAMPAIGNS: &str = r#"
  joinsets:
    - name: spend_monthly_campaigns
      dimensions:
        - ref: date
        - ref: campaign
      measures:
        - ref: clicks
        - ref: budget
      datasets:
        - name: spend_monthly
          extras:
            storage:
              tables: [warehouse.spend_monthly]
            temporal:
              timeseries:
                occurred_at: date
              granularity: month
        - name: campaigns
          extras:
            storage:
              tables: [warehouse.campaigns]
      relationships:
        - name: spend_monthly_to_campaigns
          from: spend_monthly
          to: campaigns
          on: [campaign]
          cardinality: many_to_one
"#;

async fn spend_of(sources: &[&str]) -> SemanticManifest {
    let daily = [("date", "date"), ("campaign", "string"), ("clicks", "long")];
    let catalog = Catalog::default()
        .table("warehouse.spend_daily", FileFormat::Iceberg, &daily)
        .table(
            "warehouse.spend_monthly",
            FileFormat::Iceberg,
            &[daily.as_slice(), &[("budget", "long")]].concat(),
        )
        .table(
            "warehouse.budget_monthly",
            FileFormat::Iceberg,
            &[("date", "date"), ("campaign", "string"), ("budget", "long")],
        )
        .table(
            "warehouse.campaigns",
            FileFormat::Iceberg,
            &[("campaign", "string")],
        );
    with_semantics(SEMANTICS, &sources.concat(), &catalog).await
}

async fn spend() -> SemanticManifest {
    spend_of(&[SPEND_DAILY, SPEND_MONTHLY]).await
}

const WEEKLY_SEMANTICS: &str = r#"
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, week, month]
  measures:
    - name: cost
      data_type: double
      agg: sum
"#;

const SPEND_WEEKLY: &str = r#"
    - name: spend_weekly
      extras:
        storage:
          tables: [warehouse.spend_weekly]
        temporal:
          timeseries:
            occurred_at: date
          granularity: week
      dimensions:
        - ref: date
      measures:
        - ref: cost
"#;

async fn week_at_month() -> Golden {
    let catalog = Catalog::default().table(
        "warehouse.spend_weekly",
        FileFormat::Iceberg,
        &[("date", "date"), ("cost", "double")],
    );
    let manifest = with_semantics(WEEKLY_SEMANTICS, SPEND_WEEKLY, &catalog).await;
    let request = at_granularity("date", Granularity::Month, &["cost"]).with_from("spend_weekly");
    let refused = grain_not_servable(
        &manifest,
        &request,
        &["spend_weekly"],
        (Some(Granularity::Month), Some(Granularity::Week)),
        &[],
    );
    Golden::message("week_at_month", &refused)
}

#[tokio::test]
async fn a_source_at_week_under_from_cannot_serve_month() {
    week_at_month().await.assert_committed();
}

async fn grain_from() -> Golden {
    let request = at_granularity("date", Granularity::Day, &["campaign", "clicks"])
        .with_from("spend_monthly");
    let refused = grain_not_servable(
        &spend().await,
        &request,
        &["spend_monthly"],
        (Some(Granularity::Day), Some(Granularity::Month)),
        &[],
    );
    Golden::message("grain_from", &refused)
}

#[tokio::test]
async fn a_monthly_source_under_from_cannot_serve_day() {
    grain_from().await.assert_committed();
}

const SCD_SEMANTICS: &str = r#"
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
  measures:
    - name: budget
      data_type: bigint
      agg: sum
"#;

const BUDGET_HISTORY: &str = r#"
    - name: budget_history
      extras:
        storage:
          tables: [warehouse.budget_history]
        temporal:
          scd:
            scd_type: type2
            valid_from: valid_from
            valid_to: valid_to
          granularity: month
      dimensions:
        - ref: valid_from
        - ref: valid_to
      measures:
        - ref: budget
"#;

async fn scd_axis_at_day() -> Golden {
    let catalog = Catalog::default().table(
        "warehouse.budget_history",
        FileFormat::Iceberg,
        &[
            ("valid_from", "date"),
            ("valid_to", "date"),
            ("budget", "long"),
        ],
    );
    let manifest = with_semantics(SCD_SEMANTICS, BUDGET_HISTORY, &catalog).await;
    let at = |name: &str, granularity| Selection::Dimension {
        name: name.into(),
        modifier: Some(DimensionModifier::Granularity(granularity)),
    };
    let request = Request::new(vec![
        at("valid_from", Granularity::Month),
        at("valid_to", Granularity::Day),
        Selection::Field("budget".into()),
    ])
    .with_from("budget_history");
    let refused = grain_not_servable(
        &manifest,
        &request,
        &["budget_history"],
        (Some(Granularity::Day), Some(Granularity::Month)),
        &[],
    );
    Golden::message("scd_axis_at_day", &refused)
}

#[tokio::test]
async fn a_source_under_from_cannot_serve_a_finer_granularity_on_either_scd_dimension() {
    scd_axis_at_day().await.assert_committed();
}

fn assert_reads(cases: &[(&str, Request, &str)], manifest: &SemanticManifest) {
    for (case, request, table) in cases {
        let (plan, advisories) = planned(manifest, request);

        assert_eq!(read_table(&plan), *table, "{case}");
        assert!(advisories.is_empty(), "{case}: {advisories:?}");
    }
}

#[tokio::test]
async fn granularity_fit_ranks_by_what_the_request_asks_on_the_time_axis() {
    let cases = [
        (
            "G1a",
            at_granularity("date", Granularity::Month, &["campaign", "clicks"]),
            "spend_monthly",
        ),
        (
            "G1c",
            select(&["date", "campaign", "clicks"]),
            "spend_daily",
        ),
        ("G1b", select(&["campaign", "clicks"]), "spend_monthly"),
    ];
    assert_reads(&cases, &spend().await);
}

#[tokio::test]
async fn a_source_that_cannot_serve_the_granularity_is_no_candidate() {
    let request = at_granularity("date", Granularity::Day, &["campaign", "clicks"]);
    assert_reads(&[("G1d", request, "spend_daily")], &spend().await);
}

#[tokio::test]
async fn a_pinned_root_carries_the_rank_that_field_first_gives_it() {
    let manifest = spend().await;
    let request = at_granularity("date", Granularity::Month, &["campaign", "clicks"]);

    let (field_first, _) = planned(&manifest, &request);
    let (from, _) = planned(&manifest, &request.clone().with_from("spend_monthly"));

    assert_eq!(field_first, from);
    assert_eq!(
        from.root.common().semantic_meta,
        [SemAnnotation::Rank {
            scope_root: DataKindName::new("spend_monthly"),
            compositions: 0,
            granularity_fit: GranularityGroup::Requested,
            granularity: Some(Granularity::Month),
        }]
    );
}

#[tokio::test]
async fn a_dataset_outranks_a_joinset_that_fits_the_granularity_better() {
    let manifest = spend_of(&[SPEND_DAILY, SPEND_MONTHLY_CAMPAIGNS]).await;
    let request = at_granularity("date", Granularity::Month, &["campaign", "clicks"]);

    assert_reads(&[("G2", request.clone(), "spend_daily")], &manifest);
    let (plan, _) = planned(&manifest, &request.with_from("spend_monthly_campaigns"));
    let joinset = DataKindName::new("spend_monthly_campaigns");
    assert!(
        (plan.root.common().semantic_meta)
            .iter()
            .any(|subject| matches!(subject, SemAnnotation::Rank { scope_root, .. } if *scope_root == joinset)),
        "{plan:?}"
    );
}

const ORDER_SEMANTICS: &str = r#"
  dimensions:
    - name: order_date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
    - name: ship_date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - name: revenue
      data_type: double
      agg: sum
"#;

const ORDERS_BY_TIME_AXIS: &str = r#"
    - name: orders
      extras:
        storage:
          tables: [warehouse.orders]
        temporal:
          timeseries:
            occurred_at: order_date
          granularity: day
      dimensions:
        - ref: order_date
      measures:
        - ref: revenue
    - name: shipments
      extras:
        storage:
          tables: [warehouse.shipments]
        temporal:
          timeseries:
            occurred_at: ship_date
          granularity: month
      dimensions:
        - ref: order_date
        - ref: ship_date
      measures:
        - ref: revenue
    - name: orders_raw
      extras:
        storage:
          tables: [warehouse.orders_raw]
      dimensions:
        - ref: order_date
      measures:
        - ref: revenue
"#;

#[tokio::test]
async fn each_candidate_is_ranked_on_its_own_time_axis() {
    let raw = [("order_date", "date"), ("revenue", "double")];
    let catalog = Catalog::default()
        .table("warehouse.orders", FileFormat::Iceberg, &raw)
        .table("warehouse.orders_raw", FileFormat::Iceberg, &raw)
        .table(
            "warehouse.shipments",
            FileFormat::Iceberg,
            &[raw.as_slice(), &[("ship_date", "date")]].concat(),
        );
    let manifest = with_semantics(ORDER_SEMANTICS, ORDERS_BY_TIME_AXIS, &catalog).await;
    let cases = [
        (
            "G3a",
            at_granularity("order_date", Granularity::Month, &["revenue"]),
            "orders",
        ),
        ("G3b", select(&["order_date", "revenue"]), "orders"),
        ("G3c", select(&["revenue"]), "shipments"),
    ];
    assert_reads(&cases, &manifest);
}

async fn no_candidate() -> Golden {
    let manifest = spend_of(&[SPEND_DAILY, BUDGET_MONTHLY]).await;
    let request = at_granularity(
        "date",
        Granularity::Month,
        &["campaign", "clicks", "budget"],
    );
    let (error, advisories) = refusal(&manifest, &request);

    assert_eq!(
        error.kind,
        PlanError::NoCandidate {
            nearest: vec![
                ("budget_monthly".into(), vec!["clicks".into()]),
                ("spend_daily".into(), vec!["budget".into()]),
            ],
        }
    );
    assert_eq!(error.kind.path(), None);
    assert!(advisories.is_empty());
    Golden::message("no_candidate", &error)
}

#[tokio::test]
async fn a_demand_split_across_two_sources_is_no_candidate_naming_the_nearest() {
    no_candidate().await.assert_committed();
}

const CPC_SEMANTICS: &str = r#"
  dimensions:
    - name: campaign_id
      data_type: string
      type: categorical
    - name: campaign_name
      data_type: string
      type: categorical
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - name: avg_cpc
      data_type: double
      agg: avg
    - name: clicks
      data_type: bigint
      agg: sum
    - name: reach
      data_type: bigint
      agg: sum
      additivity: non
  metrics:
    - name: cpc_x2
      data_type: double
      expr:
        multiply: [avg_cpc, {lit: 2}]
    - name: clicks_x2
      data_type: bigint
      expr:
        multiply: [clicks, {lit: 2}]
"#;

const CAMPAIGN_DAILY: &str = r#"
    - name: campaign_daily
      extras:
        storage:
          tables: [warehouse.campaign_daily]
        temporal:
          timeseries:
            occurred_at: date
          granularity: day
      dimensions:
        - ref: campaign_id
          is_key: true
          keysets: [natural]
        - ref: campaign_name
        - ref: date
          is_key: true
          keysets: [natural]
      measures:
        - ref: avg_cpc
        - ref: clicks
        - ref: reach
      metrics:
        - ref: clicks_x2
        - ref: cpc_x2
        - name: cpc_mean
          data_type: double
          agg: avg
          expr: {col: avg_cpc}
"#;

const CAMPAIGN_RAW: &str = r#"
    - name: campaign_raw
      extras:
        storage:
          tables: [warehouse.campaign_raw]
      dimensions:
        - ref: campaign_id
        - ref: date
      measures:
        - ref: avg_cpc
"#;

async fn campaigns() -> SemanticManifest {
    let raw = [
        ("campaign_id", "string"),
        ("date", "date"),
        ("avg_cpc", "double"),
    ];
    let catalog = Catalog::default()
        .table(
            "warehouse.campaign_daily",
            FileFormat::Iceberg,
            &[
                raw.as_slice(),
                &[
                    ("campaign_name", "string"),
                    ("clicks", "long"),
                    ("reach", "long"),
                ],
            ]
            .concat(),
        )
        .table("warehouse.campaign_raw", FileFormat::Iceberg, &raw);
    let sources = [CAMPAIGN_DAILY, CAMPAIGN_RAW].concat();
    with_semantics(CPC_SEMANTICS, &sources, &catalog).await
}

#[tokio::test]
async fn a_non_additive_measure_over_daily_rows_is_planned_at_month() {
    let request = at_granularity("date", Granularity::Month, &["avg_cpc"]);
    let (plan, _) = planned(&campaigns().await, &request);

    assert_eq!(read_table(&plan), "campaign_daily");
}

#[tokio::test]
async fn a_request_that_drops_a_keyset_dimension_is_planned_over_the_daily_rows() {
    let manifest = campaigns().await;
    for names in [
        &["campaign_id", "date", "avg_cpc", "clicks"][..],
        &["campaign_id", "avg_cpc", "clicks"],
        &["campaign_id", "reach"],
        &["campaign_id", "cpc_x2"],
        &["campaign_id", "clicks_x2"],
        &["campaign_id", "cpc_mean"],
    ] {
        let (plan, _) = planned(&manifest, &select(names));

        assert_eq!(read_table(&plan), "campaign_daily", "{names:?}");
    }
}

#[tokio::test]
async fn reaggregating_rows_that_are_not_aggregates_raises_nothing() {
    let request = select(&["campaign_id", "avg_cpc"]).with_from("campaign_raw");
    assert_reads(&[("no shape", request, "campaign_raw")], &campaigns().await);
}

async fn spend_calendar(relationship: &str, temporal: &str) -> SemanticManifest {
    let yaml = format!(
        r#"
semantic_model:
  name: sc
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
    - name: sc
      dimensions: [{{ref: date}}, {{ref: campaign_id}}, {{ref: fiscal_week}}]
      measures: [{{ref: avg_cpc}}]
      datasets:
        - name: spend
          extras:
            storage: {{tables: [warehouse.spend]}}
            semantic_mapping: {{date: date, campaign_id: campaign_id, avg_cpc: cpc}}
            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}
        - name: calendar
          extras:
            storage: {{tables: [warehouse.calendar]}}
            semantic_mapping: {{date: date, fiscal_week: fiscal_week}}
            {temporal}
      relationships:
        - {{name: rel, {relationship}, on: [date], integrity: assumed}}
"#
    );
    let catalog = Catalog::default()
        .table(
            "warehouse.spend",
            FileFormat::Iceberg,
            &[
                ("date", "date"),
                ("campaign_id", "string"),
                ("cpc", "double"),
            ],
        )
        .table(
            "warehouse.calendar",
            FileFormat::Iceberg,
            &[("date", "date"), ("fiscal_week", "string")],
        );
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_root_joinset_holds_its_time_axis_where_each_value_of_its_coalesce_is_held() {
    let spend_to_calendar = "from: spend, to: calendar, cardinality: many_to_one";
    let calendar_to_spend = "from: calendar, to: spend, cardinality: one_to_many";
    let at_day = "temporal: {timeseries: {occurred_at: date}, granularity: day}";
    let lossy_at_sc = [("sc".to_owned(), "avg_cpc")];
    let cases = [
        (spend_to_calendar, "left", "", &[][..]),
        (spend_to_calendar, "none", "", &[]),
        (calendar_to_spend, "right", "", &[]),
        (calendar_to_spend, "left", at_day, &[]),
        (spend_to_calendar, "both", "", &lossy_at_sc),
        (calendar_to_spend, "left", "", &lossy_at_sc),
    ];
    let request = at_granularity(
        "date",
        Granularity::Day,
        &["campaign_id", "fiscal_week", "avg_cpc"],
    );
    for (relationship, optional, temporal, advised) in cases {
        let relationship = format!("{relationship}, optional: {optional}");
        let manifest = spend_calendar(&relationship, temporal).await;

        let (_, advisories) = planned(&manifest, &request);

        assert_eq!(lossy(&advisories), advised, "{relationship} {temporal}");
    }
}

async fn raw_daily_calendar(a_raw_mapping: &str) -> SemanticManifest {
    let yaml = format!(
        r#"
semantic_model:
  name: abc
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {{name: account, data_type: string, type: categorical}}
    - {{name: fiscal_week, data_type: string, type: categorical}}
  measures:
    - {{name: avg_cpc, data_type: double, agg: avg}}
  joinsets:
    - name: abc
      dimensions: [{{ref: date}}, {{ref: account}}, {{ref: fiscal_week}}]
      measures: [{{ref: avg_cpc}}]
      datasets:
        - name: a_raw
          extras:
            storage: {{tables: [warehouse.a_raw]}}
            semantic_mapping: {{{a_raw_mapping}}}
        - name: b_daily
          extras:
            storage: {{tables: [warehouse.b_daily]}}
            semantic_mapping: {{date: date, account: account, avg_cpc: cpc}}
            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}
        - name: c_calendar
          extras:
            storage: {{tables: [warehouse.c_calendar]}}
            semantic_mapping: {{date: date, fiscal_week: fiscal_week}}
            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}
      relationships:
        - {{name: a_to_b, from: a_raw, to: b_daily, on: [account], cardinality: one_to_many, optional: left, integrity: assumed}}
        - {{name: b_to_c, from: b_daily, to: c_calendar, on: [date], cardinality: many_to_one, optional: none, integrity: assumed}}
"#
    );
    let catalog = Catalog::default()
        .table(
            "warehouse.a_raw",
            FileFormat::Iceberg,
            &[("date", "date"), ("account", "string")],
        )
        .table(
            "warehouse.b_daily",
            FileFormat::Iceberg,
            &[("date", "date"), ("account", "string"), ("cpc", "double")],
        )
        .table(
            "warehouse.c_calendar",
            FileFormat::Iceberg,
            &[("date", "date"), ("fiscal_week", "string")],
        );
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_join_condition_holds_an_operands_time_axis_only_where_it_compares_the_operands_value() {
    let cases = [
        (
            "date: date, account: account",
            &[("abc".to_owned(), "avg_cpc")][..],
        ),
        ("account: account", &[]),
    ];
    let request = at_granularity(
        "date",
        Granularity::Day,
        &["account", "fiscal_week", "avg_cpc"],
    );
    for (a_raw_mapping, advised) in cases {
        let manifest = raw_daily_calendar(a_raw_mapping).await;

        let (_, advisories) = planned(&manifest, &request);

        assert_eq!(lossy(&advisories), advised, "{a_raw_mapping}");
    }
}

async fn daily_pair_calendar(optional: &str, end: &str) -> SemanticManifest {
    let yaml = format!(
        r#"
semantic_model:
  name: abd
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {{name: account, data_type: string, type: categorical}}
    - {{name: fiscal_week, data_type: string, type: categorical}}
  measures:
    - {{name: avg_cpc, data_type: double, agg: avg}}
    - {{name: clicks, data_type: bigint, agg: sum}}
  joinsets:
    - name: abd
      dimensions: [{{ref: date}}, {{ref: account}}, {{ref: fiscal_week}}]
      measures: [{{ref: avg_cpc}}, {{ref: clicks}}]
      datasets:
        - name: a_daily
          extras:
            storage: {{tables: [warehouse.a_daily]}}
            semantic_mapping: {{date: date, account: account, avg_cpc: cpc}}
            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}
        - name: b_daily
          extras:
            storage: {{tables: [warehouse.b_daily]}}
            semantic_mapping: {{date: date, account: account, clicks: clicks}}
            temporal: {{timeseries: {{occurred_at: date}}, granularity: day}}
        - name: calendar
          extras:
            storage: {{tables: [warehouse.calendar]}}
            semantic_mapping: {{date: date, fiscal_week: fiscal_week}}
      relationships:
        - {{name: a_to_b, from: a_daily, to: b_daily, on: [date, account], cardinality: one_to_one, optional: {optional}, integrity: assumed, cross_filter: both}}
        - {{name: to_calendar, from: {end}, to: calendar, on: [date], cardinality: many_to_one, optional: left, integrity: assumed}}
"#
    );
    let catalog = Catalog::default()
        .table(
            "warehouse.a_daily",
            FileFormat::Iceberg,
            &[("date", "date"), ("account", "string"), ("cpc", "double")],
        )
        .table(
            "warehouse.b_daily",
            FileFormat::Iceberg,
            &[("date", "date"), ("account", "string"), ("clicks", "long")],
        )
        .table(
            "warehouse.calendar",
            FileFormat::Iceberg,
            &[("date", "date"), ("fiscal_week", "string")],
        );
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn a_join_condition_equates_an_operands_time_axis_through_the_equality_of_its_own_join() {
    let lossy_at_abd = [("abd".to_owned(), "avg_cpc")];
    let cases = [
        ("none", "b_daily", &[][..]),
        ("none", "a_daily", &[]),
        ("left", "a_daily", &[]),
        ("left", "b_daily", &lossy_at_abd),
        ("both", "a_daily", &lossy_at_abd),
    ];
    let request = at_granularity(
        "date",
        Granularity::Day,
        &["account", "fiscal_week", "avg_cpc"],
    );
    for (optional, end, advised) in cases {
        let manifest = daily_pair_calendar(optional, end).await;

        let (_, advisories) = planned(&manifest, &request);

        assert_eq!(lossy(&advisories), advised, "{optional} {end}");
    }
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            week_at_month().await,
            grain_from().await,
            no_candidate().await,
            scd_axis_at_day().await,
        ]
    })
    .await;
}

#[tokio::test]
async fn a_one_child_grainset_answers_as_a_dataset_with_the_lossy_advisory() {
    let yaml = r#"
semantic_model:
  name: one_child_grainset
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - name: reach
      data_type: bigint
      agg: sum
      additivity: non
  grainsets:
    - name: ads
      dimensions: [{ref: date}]
      measures: [{ref: reach}]
      datasets:
        - name: daily
          extras:
            storage: {tables: [warehouse.daily]}
            semantic_mapping: {date: date, reach: reach}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;
    let catalog = Catalog::default().table(
        "warehouse.daily",
        FileFormat::Iceberg,
        &[("date", "date"), ("reach", "bigint")],
    );
    let manifest = compiled(yaml, &catalog, &InMemoryFileSystem::new()).await;

    let (_, advisories) = planned(&manifest, &select(&["reach"]));

    assert_eq!(lossy(&advisories), vec![("ads".to_owned(), "reach")]);
}

#[tokio::test]
async fn a_one_branch_unionset_stating_month_over_a_day_dataset_rolls_a_non_additive_measure_up_with_the_lossy_advisory(
) {
    let yaml = r#"
semantic_model:
  name: one_child_month_over_day
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
  measures:
    - name: reach
      data_type: bigint
      agg: sum
      additivity: non
  grainsets:
    - name: one
      dimensions: [{ref: date}]
      measures: [{ref: reach}]
      datasets:
        - name: yearly
          extras:
            storage: {tables: [warehouse.yearly]}
            semantic_mapping: {date: date, reach: reach}
            temporal: {timeseries: {occurred_at: date}, granularity: year}
      unionsets:
        - name: inner
          mode: all
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          datasets:
            - name: daily
              extras:
                storage: {tables: [warehouse.daily]}
                semantic_mapping: {date: date, reach: reach}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;
    let catalog = Catalog::default()
        .table(
            "warehouse.yearly",
            FileFormat::Iceberg,
            &[("date", "date"), ("reach", "bigint")],
        )
        .table(
            "warehouse.daily",
            FileFormat::Iceberg,
            &[("date", "date"), ("reach", "bigint")],
        );
    let manifest = compiled(yaml, &catalog, &InMemoryFileSystem::new()).await;

    let (_, advisories) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["reach"]),
    );

    assert_eq!(
        lossy(&advisories),
        vec![("one / one.inner".to_owned(), "reach")]
    );
}

const SOURCES_YAML_HEAD: &str = r#"
semantic_model:
  name: sources_month
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - name: reach
      data_type: bigint
      agg: sum
      additivity: non
  unionsets:
    - name: sources
      mode: all
      dimensions: [{ref: date}]
      measures: [{ref: reach}]
      datasets:
        - name: direct
          extras:
            storage: {tables: [warehouse.direct]}
            semantic_mapping: {date: date, reach: reach}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
      grainsets:
        - name: tiktok
"#;

async fn sources_advisories(tiktok_body: &str) -> Vec<(String, String)> {
    let yaml = format!("{SOURCES_YAML_HEAD}{tiktok_body}");
    let columns: &[(&str, &str)] = &[("date", "date"), ("reach", "bigint")];
    let catalog = Catalog::default()
        .table("warehouse.direct", FileFormat::Iceberg, columns)
        .table("warehouse.t1", FileFormat::Iceberg, columns)
        .table("warehouse.t2", FileFormat::Iceberg, columns);
    let manifest = compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await;
    let (_, advisories) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["reach"]),
    );
    lossy(&advisories)
        .into_iter()
        .map(|(path, measure)| (path, measure.to_owned()))
        .collect()
}

#[tokio::test]
async fn the_kept_and_the_sugar_form_give_the_same_lossy_advisory_on_the_root() {
    let kept = sources_advisories(
        r#"
          unionsets:
            - name: tiktok_disjoint_1
              mode: all
              datasets:
                - name: t1
                  extras:
                    storage: {tables: [warehouse.t1]}
                    semantic_mapping: {date: date, reach: reach}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
                - name: t2
                  extras:
                    storage: {tables: [warehouse.t2]}
                    semantic_mapping: {date: date, reach: reach}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
"#,
    )
    .await;
    let sugar = sources_advisories(
        r#"
          datasets:
            - name: t1
              extras:
                storage: {tables: [warehouse.t1]}
                semantic_mapping: {date: date, reach: reach}
                temporal: {timeseries: {occurred_at: date}, granularity: day}
"#,
    )
    .await;

    let expected = vec![("sources".to_owned(), "reach".to_owned())];
    assert_eq!(kept, expected);
    assert_eq!(sugar, expected);
}

#[tokio::test]
async fn a_day_request_under_a_kept_grainset_that_states_month_is_not_servable() {
    let yaml = r#"
semantic_model:
  name: kept_grainset_month
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month]
  measures:
    - name: spend
      data_type: bigint
      agg: sum
  unionsets:
    - name: sources
      mode: all
      dimensions: [{ref: date}]
      measures: [{ref: spend}]
      datasets:
        - name: a
          extras:
            storage: {tables: [warehouse.a]}
            semantic_mapping: {date: date, spend: spend}
            temporal: {timeseries: {occurred_at: date}, granularity: day}
      grainsets:
        - name: tiktok
          extras:
            temporal: {timeseries: {occurred_at: date}, granularity: month}
          unionsets:
            - name: tiktok_disjoint_1
              mode: all
              datasets:
                - name: t1
                  extras:
                    storage: {tables: [warehouse.t1]}
                    semantic_mapping: {date: date, spend: spend}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
                - name: t2
                  extras:
                    storage: {tables: [warehouse.t2]}
                    semantic_mapping: {date: date, spend: spend}
                    temporal: {timeseries: {occurred_at: date}, granularity: day}
"#;
    let columns: &[(&str, &str)] = &[("date", "date"), ("spend", "bigint")];
    let catalog = Catalog::default()
        .table("warehouse.a", FileFormat::Iceberg, columns)
        .table("warehouse.t1", FileFormat::Iceberg, columns)
        .table("warehouse.t2", FileFormat::Iceberg, columns);
    let manifest = compiled(yaml, &catalog, &InMemoryFileSystem::new()).await;

    let request = at_granularity("date", Granularity::Day, &["spend"]);
    grain_not_servable(
        &manifest,
        &request,
        &["sources", "sources.tiktok"],
        (Some(Granularity::Day), Some(Granularity::Month)),
        &[("sources.tiktok.tiktok_disjoint_1", &[])],
    );
}
