mod common;

use common::{
    aggregate_step, at_granularity, balance_marker, call, column, columns_of, compiled,
    every_marker, fields, filter, join, latest_by_date, markers, measure_markers, names, once,
    output_names, planned, projection, rank, scan_of, select, summed, text, union_of,
    with_semantics, written, Catalog, Formula, LATEST, UNBUCKETED_DATE,
};
use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::Granularity;
use semstrait_ir::{AggregateNode, PhysicalExpr, PlanNode, ProjectNode, SemanticPlan};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{lit, measure, metric};
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal};

async fn balances(semi: &str) -> SemanticManifest {
    let dataset = format!(
        r#"
    - name: balances
      extras:
        storage:
          tables: [warehouse.balances]
        temporal:
          timeseries:
            occurred_at: date
          granularity: day
      dimensions:
        - name: account
          data_type: string
          type: categorical
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
            semi: {semi}
        - name: cost
          data_type: bigint
          agg: sum
"#
    );
    let catalog = Catalog::default().table(
        "warehouse.balances",
        FileFormat::Iceberg,
        &[
            ("account", "string"),
            ("date", "date"),
            ("balance", "long"),
            ("cost", "long"),
        ],
    );
    with_semantics("", &dataset, &catalog).await
}

#[tokio::test]
async fn a_latest_measure_reads_its_rank_marker_after_the_bucket_by_the_default_by() {
    let manifest = balances("{strategy: latest}").await;

    let (plan, _) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["balance"]),
    );

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "balance");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let marker = balance_marker(aggregate, &["date"]);
    let marking = projection(&aggregate.input);
    let bucketing = projection(&marking.input);
    let passed: Vec<_> = (fields(bucketing).into_iter())
        .map(|(name, _)| (name, column(name)))
        .collect();
    let unmarked: Vec<_> = (fields(marking).into_iter())
        .filter(|(name, _)| name != &marker)
        .collect();
    assert_eq!(unmarked, passed);
    assert_eq!(
        written(bucketing, "date"),
        Some(call("date_trunc", vec![text("month"), column("date")]))
    );
    assert_eq!(output_names(&plan), ["date", "balance"]);
}

#[tokio::test]
async fn a_request_that_keeps_by_adds_no_marker() {
    let manifest = balances("{strategy: latest}").await;

    let (plan, _) = planned(&manifest, &select(&["date", "balance"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "balance");
    assert_eq!(sum.filter(), None);
    assert_eq!(every_marker(&plan.root), []);
    assert_eq!(output_names(&plan), ["date", "balance"]);
}

#[tokio::test]
async fn a_neighbour_of_a_semi_additive_measure_reads_every_row_in_the_same_aggregate_step() {
    let manifest = balances("{by: date, strategy: latest}").await;

    let (plan, _) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["balance", "cost"]),
    );

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    let entries: Vec<_> = (aggregate.aggregates.iter())
        .map(|(entry, call)| (entry.as_str(), call.function().declarations.canonical.name))
        .collect();
    assert_eq!(entries, [("balance", "sum"), ("cost", "sum")]);
    balance_marker(aggregate, &["date"]);
    assert_eq!(aggregate.aggregates[1].1.filter(), None);
    assert_eq!(output_names(&plan), ["date", "balance", "cost"]);
}

#[tokio::test]
async fn a_predicate_on_a_semi_additive_measure_sits_directly_above_the_aggregate_step() {
    let manifest = balances("{by: date, strategy: latest}").await;
    let over = measure("balance").expect("a name").gt(lit(50));
    let request = at_granularity("date", Granularity::Month, &["balance"]).with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let filter = filter(&plan.root);
    assert_eq!(
        filter.filter,
        column("balance").gt(Expr::Literal(Literal::Integer(50)))
    );
    balance_marker(aggregate_step(&filter.input), &["date"]);
    assert_eq!(output_names(&plan), ["date", "balance"]);
}

#[tokio::test]
async fn an_unselected_axis_partitions_the_marker_and_is_read_below_it_then_dropped() {
    let manifest = balances("{axes: [account], strategy: latest}").await;
    let source = columns_of(&manifest, "balances");

    let (plan, _) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["balance"]),
    );

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    balance_marker(aggregate, &["date", "account"]);
    let bucketing = projection(&projection(&aggregate.input).input);
    assert_eq!(written(bucketing, "account"), Some(column("account")));
    let row_level = projection(&bucketing.input);
    assert_eq!(written(row_level, "account"), Some(source("account")));
    assert_eq!(
        names(&scan_of(&plan).schema),
        ["date", "balance", "account"]
    );
    assert_eq!(output_names(&plan), ["date", "balance"]);
}

async fn f4() -> SemanticManifest {
    let dataset = r#"
    - name: f4
      extras:
        storage:
          tables: [warehouse.f4]
      dimensions:
        - name: g
          data_type: string
          type: categorical
        - name: d
          data_type: date
          type:
            temporal:
              granularities: [day]
      measures:
        - name: v
          data_type: bigint
          agg: sum
          additivity:
            semi: {axes: [], by: d, strategy: latest}
"#;
    let catalog = Catalog::default().table(
        "warehouse.f4",
        FileFormat::Iceberg,
        &[("g", "string"), ("d", "date"), ("v", "long")],
    );
    with_semantics("", dataset, &catalog).await
}

#[tokio::test]
async fn a_marker_orders_by_an_unselected_by_with_nulls_last_and_no_window_frame() {
    let manifest = f4().await;
    let source = columns_of(&manifest, "f4");

    let (plan, _) = planned(&manifest, &select(&["g", "v"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("g")]);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    assert_eq!(window, &rank(vec![column("g")], column("d"), LATEST));
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(entry.as_str(), "v");
    assert_eq!(sum.filter(), Some(&once(marker)));
    let row_level = projection(&marking.input);
    assert!(
        matches!(*row_level.input, PlanNode::Scan(_)),
        "{row_level:?}"
    );
    assert_eq!(written(row_level, "d"), Some(source("d")));
    assert_eq!(names(&scan_of(&plan).schema), ["g", "v", "d"]);
    assert_eq!(output_names(&plan), ["g", "v"]);
}

#[tokio::test]
async fn a_marker_partitions_by_the_bucket_then_by_its_axes() {
    let manifest = balances("{axes: [account], by: date, strategy: latest}").await;

    let (plan, _) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["balance", "cost"]),
    );

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    balance_marker(aggregate, &["date", "account"]);
    assert_eq!(aggregate.aggregates[1].1.filter(), None);
    assert_eq!(output_names(&plan), ["date", "balance", "cost"]);
}

async fn daily(semi: &str) -> SemanticManifest {
    let dataset = format!(
        r#"
    - name: daily
      extras:
        storage:
          tables: [warehouse.daily]
        temporal:
          timeseries:
            occurred_at: date
          granularity: day
        semantic_mapping:
          account: account
          date: date
          a: a
          b: b
          cost: cost
          avg_cost: cost
          users: usr
      dimensions:
        - name: account
          data_type: string
          type: categorical
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
            semi: {semi}
        - name: b
          data_type: double
          agg: sum
          additivity:
            semi: {semi}
        - name: cost
          data_type: bigint
          agg: sum
        - name: avg_cost
          data_type: double
          agg: avg
        - name: users
          data_type: bigint
          agg: count_distinct
      metrics:
        - name: m
          data_type: double
          expr:
            divide: [a, b]
        - name: m_lin
          data_type: double
          expr:
            subtract: [a, b]
        - name: accounts_with_cost
          data_type: bigint
          agg: count
          expr:
            case:
              whens:
                - when: {{gt: [cost, {{lit: 0}}]}}
                  then: cost
        - name: a_per_account
          data_type: double
          expr:
            divide: [a, accounts_with_cost]
"#
    );
    let catalog = Catalog::default().table(
        "warehouse.daily",
        FileFormat::Iceberg,
        &[
            ("account", "string"),
            ("date", "date"),
            ("a", "double"),
            ("b", "double"),
            ("cost", "long"),
            ("usr", "string"),
        ],
    );
    with_semantics("", &dataset, &catalog).await
}

const IN_ACCOUNT: &str = "{axes: [account], by: date, strategy: latest}";

const RATIO: (&str, Formula) = ("m", |a, b| a / b);

const LINEAR: (&str, Formula) = ("m_lin", |a, b| a - b);

#[tokio::test]
async fn a_metrics_aggregate_nodes_read_their_measures_markers_in_one_projection() {
    let manifest = daily(IN_ACCOUNT).await;

    let (plan, _) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &["m"]),
    );

    let derivation = projection(&plan.root);
    assert_eq!(written(derivation, "date"), Some(column("date")));
    let source = columns_of(&manifest, "daily");
    let aggregate = measure_markers(source, derivation, RATIO, &["date", "account"]);
    assert_eq!(aggregate.groupings, [column("date")]);
    assert_eq!(aggregate.aggregates.len(), 2);
    assert_eq!(every_marker(&plan.root).len(), 2);
    assert_eq!(output_names(&plan), ["date", "m"]);
}

#[tokio::test]
async fn a_metrics_aggregate_nodes_read_their_measures_markers_with_their_axes() {
    let cases: [(&str, (&str, Formula), &[&str]); 3] = [
        (IN_ACCOUNT, RATIO, &["date", "account"]),
        (IN_ACCOUNT, LINEAR, &["date", "account"]),
        ("{axes: [], by: date, strategy: latest}", RATIO, &["date"]),
    ];

    for (semi, formula, partition) in cases {
        let manifest = daily(semi).await;
        let request = at_granularity("date", Granularity::Month, &[formula.0]);

        let (plan, _) = planned(&manifest, &request);

        let source = columns_of(&manifest, "daily");
        measure_markers(source, projection(&plan.root), formula, partition);
        let read: Vec<_> = ["date", "a", "b"]
            .into_iter()
            .chain(partition[1..].iter().copied())
            .collect();
        assert_eq!(names(&scan_of(&plan).schema), read, "{semi}");
        assert_eq!(output_names(&plan), ["date", formula.0]);
    }
}

#[tokio::test]
async fn only_a_metrics_semi_additive_nodes_read_a_marker_beside_its_neighbours() {
    let manifest = daily(IN_ACCOUNT).await;
    let selected = ["m", "cost", "avg_cost", "users"];

    let (plan, _) = planned(
        &manifest,
        &at_granularity("date", Granularity::Month, &selected),
    );

    let derivation = projection(&projection(&plan.root).input);
    let source = columns_of(&manifest, "daily");
    let aggregate = measure_markers(source, derivation, RATIO, &["date", "account"]);
    let neighbours: Vec<_> = (aggregate.aggregates.iter())
        .filter(|(entry, _)| !entry.as_str().starts_with("__plan_"))
        .map(|(entry, call)| (entry.as_str(), call.filter()))
        .collect();
    assert_eq!(
        neighbours,
        [("cost", None), ("avg_cost", None), ("users", None)]
    );
    assert_eq!(every_marker(&plan.root).len(), 2);
    assert_eq!(
        output_names(&plan),
        ["date", "m", "cost", "avg_cost", "users"]
    );
}

#[tokio::test]
async fn a_predicate_on_a_metric_over_semi_additive_measures_sits_above_its_formula() {
    let manifest = daily(IN_ACCOUNT).await;
    let over = metric("m_lin").expect("a name").gt(lit(-10));
    let request = at_granularity("date", Granularity::Month, &["m_lin"]).with_predicate(over);

    let (plan, _) = planned(&manifest, &request);

    let filter = filter(&plan.root);
    assert_eq!(
        filter.filter,
        column("m_lin").gt(Expr::Literal(Literal::Integer(-10)))
    );
    measure_markers(
        columns_of(&manifest, "daily"),
        projection(&filter.input),
        LINEAR,
        &["date", "account"],
    );
    assert_eq!(output_names(&plan), ["date", "m_lin"]);
}

#[tokio::test]
async fn a_measures_marker_reads_its_own_call_beside_a_kept_link() {
    let manifest = daily(IN_ACCOUNT).await;
    let request = at_granularity("date", Granularity::Month, &["a_per_account"]);

    let (plan, _) = planned(&manifest, &request);

    let join = join(&projection(&projection(&plan.root).input).input);
    let frame = aggregate_step(&join.left);
    let [(_, sum)] = frame.aggregates.as_slice() else {
        panic!("a's entry, got {:?}", frame.aggregates);
    };
    let marking = projection(&frame.input);
    let [marker] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    latest_by_date(marking, marker, &["date", "account"]);
    assert_eq!(sum.filter(), Some(&once(marker.0)));
    let linked = aggregate_step(&join.right);
    let [reserved, _] = names(&linked.common.output_schema)[..] else {
        panic!("a grouping and the value, got {:?}", linked.common);
    };
    assert_eq!(
        join.on,
        column("date").is_not_distinct_from(column(reserved))
    );
    assert_eq!(output_names(&plan), ["date", "a_per_account"]);
}

async fn ledger() -> SemanticManifest {
    let dataset = r#"
    - name: ledger
      extras:
        storage:
          tables: [warehouse.ledger]
        temporal:
          timeseries:
            occurred_at: date
          granularity: day
        semantic_mapping:
          account: account
          date: date
          loaded_at: loaded_at
          balance: balance
          fees: fees
          fees_closing: fees
      dimensions:
        - name: account
          data_type: string
          type: categorical
        - name: date
          data_type: date
          type:
            temporal:
              granularities: [day]
        - name: loaded_at
          data_type: timestamp
          type:
            temporal:
              granularities: [hour]
      measures:
        - name: balance
          data_type: bigint
          agg: sum
          additivity:
            semi: {axes: [account], by: loaded_at, strategy: latest}
        - name: fees
          data_type: bigint
          agg: sum
        - name: fees_closing
          data_type: bigint
          agg: sum
          additivity:
            semi: {axes: [account], by: date, strategy: latest}
      metrics:
        - name: net
          data_type: bigint
          expr:
            subtract: [balance, fees]
        - name: closing_net
          data_type: bigint
          expr:
            subtract: [balance, fees_closing]
"#;
    let catalog = Catalog::default().table(
        "warehouse.ledger",
        FileFormat::Iceberg,
        &[
            ("account", "string"),
            ("date", "date"),
            ("loaded_at", "timestamp(6)"),
            ("balance", "long"),
            ("fees", "long"),
        ],
    );
    with_semantics("", dataset, &catalog).await
}

fn row_level(plan: &SemanticPlan) -> &ProjectNode {
    let mut node = &plan.root;
    while let [input] = node.children()[..] {
        if matches!(input, PlanNode::Scan(_)) {
            return projection(node);
        }
        node = input;
    }
    panic!("a projection over the scan, got {:?}", plan.root);
}

fn sums<'a, const N: usize>(
    plan: &SemanticPlan,
    aggregate: &'a AggregateNode,
) -> [(&'a str, PhysicalExpr, Option<PhysicalExpr>); N] {
    let windows = markers(projection(&aggregate.input));
    let row_level = row_level(plan);
    let read: Vec<_> = (aggregate.aggregates.iter())
        .map(|(entry, sum)| (entry.as_str(), summed(row_level, &windows, sum)))
        .collect();
    let mut readers: Vec<_> = (read.iter())
        .filter_map(|(_, (_, marker))| marker.map(|(name, _)| *name))
        .collect();
    let mut names: Vec<_> = windows.iter().map(|(name, _)| *name).collect();
    readers.sort_unstable();
    names.sort_unstable();
    assert_eq!(readers, names);
    assert_eq!(every_marker(&plan.root).len(), windows.len());
    let read: Vec<_> = (read.into_iter())
        .map(|(entry, (tree, marker))| (entry, tree, marker.map(|(_, window)| window.clone())))
        .collect();
    <[_; N]>::try_from(read).unwrap_or_else(|read| panic!("{N} entries, got {read:?}"))
}

#[tokio::test]
async fn a_metric_reads_its_semi_additive_measures_strategy_and_has_no_marker_of_its_own() {
    let manifest = ledger().await;
    let source = columns_of(&manifest, "ledger");
    let request = at_granularity("date", Granularity::Month, &["balance", "fees", "net"]);

    let (plan, _) = planned(&manifest, &request);

    let derivation = projection(&projection(&plan.root).input);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("date")]);
    let partition = vec![column("date"), column("account")];
    let by_loaded_at = rank(partition, column("loaded_at"), LATEST);
    let [balance, fees, over_balance, over_fees] = sums(&plan, aggregate);
    assert_eq!(
        balance,
        ("balance", source("balance"), Some(by_loaded_at.clone()))
    );
    assert_eq!(fees, ("fees", source("fees"), None));
    assert_eq!(
        (over_balance.1, over_balance.2),
        (source("balance"), Some(by_loaded_at))
    );
    assert_eq!((over_fees.1, over_fees.2), (source("fees"), None));
    assert_eq!(
        written(derivation, "net"),
        Some(column(over_balance.0) - column(over_fees.0))
    );
    assert_eq!(
        written(row_level(&plan), "loaded_at"),
        Some(source("loaded_at"))
    );
    assert_eq!(output_names(&plan), ["date", "balance", "fees", "net"]);
}

#[tokio::test]
async fn each_aggregate_node_of_a_metric_reads_its_own_measures_marker() {
    let manifest = ledger().await;
    let source = columns_of(&manifest, "ledger");
    let request = at_granularity("date", Granularity::Month, &["closing_net"]);

    let (plan, _) = planned(&manifest, &request);

    let derivation = projection(&projection(&plan.root).input);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("date")]);
    let latest = |by| rank(vec![column("date"), column("account")], column(by), LATEST);
    let [(over_balance, balance, by_loaded_at), (over_fees, fees, by_date)] =
        sums(&plan, aggregate);
    assert_eq!(
        (balance, by_loaded_at),
        (source("balance"), Some(latest("loaded_at")))
    );
    assert_eq!(
        (fees, by_date),
        (source("fees"), Some(latest(UNBUCKETED_DATE)))
    );
    let bucketing = projection(&projection(&aggregate.input).input);
    assert_eq!(written(bucketing, UNBUCKETED_DATE), Some(column("date")));
    assert_eq!(
        written(bucketing, "date"),
        Some(call("date_trunc", vec![text("month"), column("date")]))
    );
    assert_eq!(
        written(derivation, "closing_net"),
        Some(column(over_balance) - column(over_fees))
    );
    assert_eq!(output_names(&plan), ["date", "closing_net"]);
}

#[tokio::test]
async fn a_marker_partitions_by_a_kept_date_and_its_axes_and_orders_by_a_dropped_by() {
    let manifest = ledger().await;
    let source = columns_of(&manifest, "ledger");

    let (plan, _) = planned(&manifest, &select(&["date", "net"]));

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("date")]);
    let by_loaded_at = rank(
        vec![column("date"), column("account")],
        column("loaded_at"),
        LATEST,
    );
    let [(over_balance, balance, marker), (over_fees, fees, none)] = sums(&plan, aggregate);
    assert_eq!((balance, marker), (source("balance"), Some(by_loaded_at)));
    assert_eq!((fees, none), (source("fees"), None));
    assert_eq!(
        written(derivation, "net"),
        Some(column(over_balance) - column(over_fees))
    );
    let marking = projection(&aggregate.input);
    assert_eq!(projection(&marking.input), row_level(&plan));
    assert_eq!(
        names(&scan_of(&plan).schema),
        ["date", "balance", "fees", "loaded_at", "account"]
    );
    assert_eq!(output_names(&plan), ["date", "net"]);
}

const NESTED_BALANCES: &str = r#"
semantic_model:
  name: nested_balances
  dimensions:
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day, month, year]
    - {name: account, data_type: string, type: categorical}
    - {name: region, data_type: string, type: categorical}
  measures:
    - name: balance
      data_type: bigint
      agg: sum
      additivity:
        semi: {axes: [], by: date, strategy: latest}
  joinsets:
    - name: j
      dimensions: [{ref: date}, {ref: account}, {ref: region}]
      measures: [{ref: balance}]
      unionsets:
        - name: u
          mode: all
          extras:
            temporal: {snapshot: {snapshotted_at: date}, granularity: month}
          datasets:
            - name: bal_eu
              extras:
                storage: {tables: [warehouse.bal_eu]}
                semantic_mapping: {account: account, date: date, balance: balance}
                temporal: {snapshot: {snapshotted_at: date}, granularity: day}
            - name: bal_us
              extras:
                storage: {tables: [warehouse.bal_us]}
                semantic_mapping: {account: account, date: date, balance: balance}
      datasets:
        - name: accounts
          extras:
            storage: {tables: [warehouse.accounts]}
            semantic_mapping: {account: account, region: region}
      relationships:
        - {name: u_to_accounts, from: u, to: accounts, on: [account], cardinality: many_to_one, optional: left, integrity: assumed}
"#;

#[tokio::test]
async fn a_union_that_declares_month_over_day_values_marks_a_semi_additive_measure_above_it() {
    let columns = [("account", "string"), ("date", "date"), ("balance", "long")];
    let catalog = Catalog::default()
        .table("warehouse.bal_eu", FileFormat::Iceberg, &columns)
        .table("warehouse.bal_us", FileFormat::Iceberg, &columns)
        .table(
            "warehouse.accounts",
            FileFormat::Iceberg,
            &[("account", "string"), ("region", "string")],
        );
    let manifest = compiled(NESTED_BALANCES, &catalog, &InMemoryFileSystem::new()).await;
    let request = at_granularity("date", Granularity::Month, &["balance"]).with_from("j");

    let (plan, _) = planned(&manifest, &request);

    let (_, union) = union_of(&plan);
    for branch in &union.inputs {
        assert_eq!(every_marker(branch), [], "no marker below the union");
        assert_eq!(
            aggregate_step(branch).groupings.len(),
            1,
            "a partial by the day"
        );
    }
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    balance_marker(aggregate, &["date"]);
    assert_eq!(output_names(&plan), ["date", "balance"]);
}
