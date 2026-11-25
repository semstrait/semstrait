mod common;

use common::golden::{self, Golden};
use common::{
    aggregate_filters, aggregate_step, aggregate_steps, alias, at_granularity, below_aggregate,
    bigint, call, column, columns_of, compiled, computed, cross, datasets, entry, every_marker,
    fields, filter, join, names, output_names, parts, planned, projection, qualified, read_table,
    refusal, scan_of, select, signature, signature_of, spend_bookings, spend_bookings_with, summed,
    text, Catalog, SPEND_BOOKINGS,
};
use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::DataKindName;
use semstrait_common::Diagnose;
use semstrait_common::Granularity;
use semstrait_common::Tree;
use semstrait_ir::{
    field_name, Cardinality, JoinType, PhysicalExpr, PlanNode, SemAnnotation, SemanticPlan,
    SourceRole,
};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit};
use semstrait_model::UnresolvedSemanticExpr;
use semstrait_planner::{PlanError, Request};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{CastFailure, Expr, ExprOperationExt, Literal};

const TIERED_ORDERS: &str = r#"
semantic_model:
  name: tiered_orders
  dimensions:
    - name: order_id
      data_type: bigint
      type: categorical
    - name: amount
      data_type: bigint
      type: categorical
    - name: tier
      data_type: string
      type: categorical
    - name: floor
      data_type: bigint
      type: categorical
  joinsets:
    - name: tiered_orders
      dimensions:
        - ref: order_id
        - ref: amount
        - ref: tier
        - ref: floor
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
        - name: tiers
          extras:
            storage:
              tables: [warehouse.tiers]
      relationships:
        - name: orders_reach_tiers
          from: orders
          to: tiers
          on: cross
          optional: none
          cross_filter: both
          filter:
            gt_eq: [amount, floor]
"#;

const CHANNELS: &str = r#"
semantic_model:
  name: channels
  dimensions:
    - name: campaign_id
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: bigint
      agg: sum
  joinsets:
    - name: channels
      dimensions:
        - ref: campaign_id
      measures:
        - ref: revenue
      datasets:
        - name: web
          extras:
            storage:
              tables: [warehouse.web]
        - name: store
          extras:
            storage:
              tables: [warehouse.store]
      relationships:
        - name: web_to_store
          from: web
          to: store
          on: [campaign_id]
          cardinality: one_to_one
          optional: both
          cross_filter: both
"#;

const SALES: &str = r#"
semantic_model:
  name: sales
  dimensions:
    - name: customer_id
      data_type: bigint
      type: categorical
    - name: region
      data_type: string
      type: categorical
    - name: segment
      data_type: string
      type: categorical
    - name: region_name
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: bigint
      agg: sum
  joinsets:
    - name: sales
      dimensions:
        - ref: customer_id
        - ref: region
        - ref: segment
        - ref: region_name
      measures:
        - ref: revenue
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              customer_id: cust_key
              region: region_code
              revenue: amount
        - name: customers
          extras:
            storage:
              tables: [warehouse.customers]
        - name: regions
          extras:
            storage:
              tables: [warehouse.regions]
            semantic_mapping:
              region: reg
              region_name: region_name
      relationships:
        - name: r1
          from: orders
          to: customers
          on: [customer_id]
          cardinality: many_to_one
          optional: left
        - name: r2
          from: orders
          to: regions
          on: [region]
          cardinality: many_to_one
"#;

const ORDERS_CUSTOMERS: &str = r#"
semantic_model:
  name: orders_customers
  dimensions:
    - name: customer_id
      data_type: bigint
      type: categorical
    - name: region
      data_type: string
      type: categorical
    - name: segment
      data_type: string
      type: categorical
  measures:
    - name: revenue
      data_type: bigint
      agg: sum
  joinsets:
    - name: orders_customers
      dimensions:
        - ref: customer_id
        - ref: region
        - ref: segment
      measures:
        - ref: revenue
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              customer_id: cust_key
              region: region_code
              revenue: amount
        - name: customers
          extras:
            storage:
              tables: [warehouse.customers]
      relationships:
        - name: orders_to_customers
          from: orders
          to: customers
          on: [customer_id]
          cardinality: many_to_one
          optional: left
"#;

const CAMPAIGNS_SPEND: &str = r#"
semantic_model:
  name: campaigns_spend
  dimensions:
    - name: campaign_id
      data_type: string
      type: categorical
    - name: campaign_name
      data_type: string
      type: categorical
  measures:
    - name: budget
      data_type: bigint
      agg: sum
    - name: cost
      data_type: bigint
      agg: sum
  joinsets:
    - name: campaigns_spend
      dimensions:
        - ref: campaign_id
        - ref: campaign_name
      measures:
        - ref: budget
        - ref: cost
      datasets:
        - name: campaigns
          extras:
            storage:
              tables: [warehouse.campaigns]
        - name: spend
          extras:
            storage:
              tables: [warehouse.spend]
      relationships:
        - name: campaigns_to_spend
          from: campaigns
          to: spend
          on: [campaign_id]
          cardinality: one_to_many
          optional: none
"#;

const TAGS_SPEND: &str = r#"
semantic_model:
  name: tags_spend
  dimensions:
    - name: campaign_id
      data_type: string
      type: categorical
    - name: tag
      data_type: string
      type: categorical
  measures:
    - name: cost
      data_type: bigint
      agg: sum
  joinsets:
    - name: tags_spend
      dimensions:
        - ref: campaign_id
        - ref: tag
      measures:
        - ref: cost
      datasets:
        - name: tags
          extras:
            storage:
              tables: [warehouse.tags]
        - name: spend2
          extras:
            storage:
              tables: [warehouse.spend2]
      relationships:
        - name: tags_to_spend2
          from: tags
          to: spend2
          on: [campaign_id]
          cardinality: many_to_many
          optional: none
          cross_filter: both
"#;

async fn tags_spend() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.tags",
            FileFormat::Iceberg,
            &[("campaign_id", "string"), ("tag", "string")],
        )
        .table(
            "warehouse.spend2",
            FileFormat::Iceberg,
            &[("campaign_id", "string"), ("cost", "long")],
        );
    compiled(TAGS_SPEND, &catalog, &InMemoryFileSystem::new()).await
}

async fn campaigns_spend() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.campaigns",
            FileFormat::Iceberg,
            &[
                ("campaign_id", "string"),
                ("campaign_name", "string"),
                ("budget", "long"),
            ],
        )
        .table(
            "warehouse.spend",
            FileFormat::Iceberg,
            &[("campaign_id", "string"), ("cost", "long")],
        );
    compiled(CAMPAIGNS_SPEND, &catalog, &InMemoryFileSystem::new()).await
}

async fn orders_customers(yaml: &str) -> SemanticManifest {
    orders_customers_with(yaml, "long").await
}

async fn orders_customers_with(yaml: &str, amount: &str) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("cust_key", "long"),
                ("region_code", "string"),
                ("amount", amount),
            ],
        )
        .table(
            "warehouse.customers",
            FileFormat::Iceberg,
            &[("customer_id", "long"), ("segment", "string")],
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

async fn sales(yaml: &str) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("cust_key", "long"),
                ("region_code", "string"),
                ("amount", "long"),
            ],
        )
        .table(
            "warehouse.customers",
            FileFormat::Iceberg,
            &[
                ("customer_id", "long"),
                ("segment", "string"),
                ("region", "string"),
            ],
        )
        .table(
            "warehouse.regions",
            FileFormat::Iceberg,
            &[("reg", "string"), ("region_name", "string")],
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

async fn channels(yaml: &str) -> SemanticManifest {
    let columns = [("campaign_id", "string"), ("revenue", "long")];
    let catalog = Catalog::default()
        .table("warehouse.web", FileFormat::Iceberg, &columns)
        .table("warehouse.store", FileFormat::Iceberg, &columns);
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

async fn tiered_orders() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[("order_id", "long"), ("amount", "long")],
        )
        .table(
            "warehouse.tiers",
            FileFormat::Iceberg,
            &[("tier", "string"), ("floor", "long")],
        );
    compiled(TIERED_ORDERS, &catalog, &InMemoryFileSystem::new()).await
}

fn combined(joinset: &str, cardinality: Cardinality) -> [SemAnnotation; 1] {
    [SemAnnotation::SemanticSource {
        semantic_source: DataKindName::new(joinset),
        source_role: SourceRole::Joinset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: Some(cardinality),
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    }]
}

#[tokio::test]
async fn a_join_combine_holds_the_relationship_filter_below_one_aggregate_step() {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let request = at_granularity("date", Granularity::Month, &["cost", "revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    let bucketing = below_aggregate(&plan);
    let join = join(&projection(&bucketing.input).input);
    let day = Expr::Literal(Literal::Date("2026-03-15".into()));
    let spend = alias(&manifest, "spend_bookings.spend");
    assert_eq!(join.join_type, JoinType::Inner);
    assert_eq!(
        join.post_join_filter,
        Some(qualified(&spend, "date").gt_eq(day))
    );
    assert_eq!(
        join.common.semantic_meta,
        combined("spend_bookings", Cardinality::OneToOne)
    );
    let operands = aggregate_steps(&join.left) + aggregate_steps(&join.right);
    assert_eq!(aggregate_steps(&plan.root) - operands, 1);
    assert_eq!(output_names(&plan), ["date", "cost", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn each_operand_of_a_join_delivers_rows_under_its_own_alias() {
    let non_additive = SPEND_BOOKINGS.replace(
        "      agg: sum\n",
        "      agg: sum\n      additivity: non\n",
    );
    let manifest = spend_bookings(&non_additive).await;
    let request = at_granularity("date", Granularity::Month, &["cost", "revenue"]);

    let (plan, _) = planned(&manifest, &request);

    let bucketing = below_aggregate(&plan);
    let join = join(&projection(&bucketing.input).input);
    for (operand, input, names) in [
        ("spend", &join.left, ["campaign_id", "cost", "date"]),
        ("bookings", &join.right, ["campaign_id", "date", "revenue"]),
    ] {
        let qualifier = alias(&manifest, &format!("spend_bookings.{operand}"));
        let read = columns_of(&manifest, operand);
        let rows = projection(input);
        let expected: Vec<(String, PhysicalExpr)> = (names.iter())
            .map(|name| (format!("{qualifier}.{name}"), read(name)))
            .collect();
        let written: Vec<(String, PhysicalExpr)> = (fields(rows).into_iter())
            .map(|(name, tree)| (name.to_owned(), tree))
            .collect();
        assert_eq!(written, expected, "{operand}");
        assert!(matches!(*rows.input, PlanNode::Scan(_)), "{operand}");
    }
}

#[tokio::test]
async fn each_operand_of_a_join_gives_partial_aggregates_by_the_join_condition() {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let request = at_granularity("date", Granularity::Month, &["cost", "revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let bucketing = projection(&aggregate.input);
    let date_trunc = call("date_trunc", vec![text("month"), column("date")]);
    assert!(fields(bucketing).contains(&("date", date_trunc)));
    let rows = projection(&bucketing.input);
    let join = join(&rows.input);
    let [spend, bookings] =
        ["spend", "bookings"].map(|operand| alias(&manifest, &format!("spend_bookings.{operand}")));
    let equal = |name| qualified(&spend, name).eq(qualified(&bookings, name));
    assert_eq!(join.on, equal("campaign_id") & equal("date"));
    for (operand, qualifier, input, measure) in [
        ("spend", &spend, &join.left, "cost"),
        ("bookings", &bookings, &join.right, "revenue"),
    ] {
        let partials = aggregate_step(input);
        let grouped = ["date", "campaign_id"].map(|name| qualified(qualifier, name));
        assert_eq!(partials.groupings, grouped, "{operand}");
        let [(entry, _)] = partials.aggregates.as_slice() else {
            panic!("one entry, got {:?}", partials.aggregates);
        };
        assert_eq!(entry.as_str(), format!("{qualifier}.{measure}"));
        let day = format!("{qualifier}.date");
        let read = columns_of(&manifest, operand)("date");
        let below = projection(&partials.input);
        assert!(fields(below).contains(&(day.as_str(), read)), "{operand}");
        let (_, sum) = (aggregate.aggregates.iter())
            .find(|(entry, _)| entry.as_str() == measure)
            .expect("the Joinset's entry");
        assert_eq!(sum.function().declarations.canonical.name, "sum");
        let [argument] = sum.args() else {
            panic!("one argument, got {:?}", sum.args());
        };
        assert_eq!(computed(rows, argument), qualified(qualifier, measure));
    }
    assert_eq!(output_names(&plan), ["date", "cost", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn partial_aggregates_per_join_key_remove_the_fan_out_with_no_marker() {
    let manifest = campaigns_spend().await;

    let (plan, advisories) = planned(&manifest, &select(&["campaign_name", "budget", "cost"]));

    let join = join(&below_aggregate(&plan).input);
    let [campaigns, spend] = ["campaigns", "spend"]
        .map(|operand| alias(&manifest, &format!("campaigns_spend.{operand}")));
    assert_eq!(
        aggregate_step(&join.right).groupings,
        [qualified(&spend, "campaign_id")]
    );
    assert_eq!(
        aggregate_step(&join.left).groupings,
        ["campaign_name", "campaign_id"].map(|name| qualified(&campaigns, name))
    );
    assert!(every_marker(&plan.root).is_empty());
    assert_eq!(output_names(&plan), ["campaign_name", "budget", "cost"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_many_to_many_operand_gives_partial_aggregates_per_join_key() {
    let manifest = tags_spend().await;

    let (plan, advisories) = planned(&manifest, &select(&["tag", "cost"]));

    let join = join(&below_aggregate(&plan).input);
    let [tags, spend2] =
        ["tags", "spend2"].map(|operand| alias(&manifest, &format!("tags_spend.{operand}")));
    let partials = aggregate_step(&join.right);
    assert_eq!(partials.groupings, [qualified(&spend2, "campaign_id")]);
    let [(entry, sum)] = partials.aggregates.as_slice() else {
        panic!("one partial sum, got {:?}", partials.aggregates);
    };
    assert_eq!(entry.as_str(), format!("{spend2}.cost"));
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let distinct = aggregate_step(&join.left);
    assert_eq!(
        distinct.groupings,
        ["tag", "campaign_id"].map(|name| qualified(&tags, name))
    );
    assert!(distinct.aggregates.is_empty());
    assert_eq!(output_names(&plan), ["tag", "cost"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_with_a_non_additive_aggregate_gives_rows_beside_partial_aggregates() {
    let measure = "    - name: avg_cost\n      data_type: double\n      agg: avg\n  joinsets:\n";
    let yaml = (SPEND_BOOKINGS.replacen("  joinsets:\n", measure, 1))
        .replacen(
            "        - ref: revenue\n",
            "        - ref: revenue\n        - ref: avg_cost\n",
            1,
        )
        .replacen(
            "              tables: [warehouse.spend]\n",
            "              tables: [warehouse.spend]\n            semantic_mapping:\n              campaign_id: campaign_id\n              date: date\n              cost: cost\n              avg_cost: cost\n",
            1,
        );
    let manifest = spend_bookings(&yaml).await;
    let request = at_granularity("date", Granularity::Month, &["avg_cost", "revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let bucketing = projection(&aggregate.input);
    let date_trunc = call("date_trunc", vec![text("month"), column("date")]);
    assert!(fields(bucketing).contains(&("date", date_trunc)));
    let rows = projection(&bucketing.input);
    let join = join(&rows.input);
    let [spend, bookings] =
        ["spend", "bookings"].map(|operand| alias(&manifest, &format!("spend_bookings.{operand}")));
    assert!(matches!(*projection(&join.left).input, PlanNode::Scan(_)));
    assert_eq!(
        aggregate_step(&join.right).groupings,
        ["date", "campaign_id"].map(|name| qualified(&bookings, name))
    );
    assert_eq!(aggregate_steps(&plan.root), 2);
    for (measure, function, qualifier) in
        [("avg_cost", "avg", &spend), ("revenue", "sum", &bookings)]
    {
        let (_, entry) = (aggregate.aggregates.iter())
            .find(|(entry, _)| entry.as_str() == measure)
            .expect("the Joinset's entry");
        assert_eq!(entry.function().declarations.canonical.name, function);
        let [argument] = entry.args() else {
            panic!("one argument, got {:?}", entry.args());
        };
        assert_eq!(computed(rows, argument), qualified(qualifier, measure));
    }
    assert_eq!(output_names(&plan), ["date", "avg_cost", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_count_over_a_join_sums_its_partial_counts_coalesced_with_zero() {
    let yaml = (ORDERS_CUSTOMERS.replacen("optional: left", "optional: none", 1))
        .replacen(
            "  joinsets:\n",
            "    - name: order_count\n      data_type: bigint\n      agg: count\n  joinsets:\n",
            1,
        )
        .replacen(
            "        - ref: revenue\n",
            "        - ref: revenue\n        - ref: order_count\n",
            1,
        )
        .replacen(
            "              revenue: amount\n",
            "              revenue: amount\n              order_count: amount\n",
            1,
        );
    let manifest = orders_customers(&yaml).await;
    let vip = dim("segment").expect("a name").eq(lit("vip"));
    let request = select(&["order_count"]).with_predicate(vip);

    let (plan, advisories) = planned(&manifest, &request);

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    let [(entry, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let orders = alias(&manifest, "orders_customers.orders");
    let rows = projection(&aggregate.input);
    assert_eq!(
        computed(rows, &sum.args()[0]),
        qualified(&orders, "order_count")
    );
    let zero = Expr::Literal(Literal::BigInt(0));
    assert_eq!(
        derivation.expressions,
        [call("coalesce", vec![column(entry.as_str()), zero])]
    );
    let join = join(&rows.input);
    assert_eq!(join.join_type, JoinType::Inner);
    let partials = aggregate_step(&join.left);
    let [(partial, count)] = partials.aggregates.as_slice() else {
        panic!("one partial count, got {:?}", partials.aggregates);
    };
    assert_eq!(partial.as_str(), format!("{orders}.order_count"));
    assert_eq!(count.function().declarations.canonical.name, "count");
    assert_eq!(output_names(&plan), ["order_count"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_gives_the_partial_of_a_cast_measure_under_the_name_the_joinset_reads() {
    let counted = SPEND_BOOKINGS.replacen(
        "      data_type: bigint\n      agg: sum\n",
        "      data_type: double\n      agg: count\n",
        1,
    );
    let manifest = spend_bookings(&counted).await;
    let request = at_granularity("date", Granularity::Month, &["cost", "revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    let derivation = projection(&projection(&plan.root).input);
    let aggregate = aggregate_step(&derivation.input);
    let rows = projection(&projection(&aggregate.input).input);
    let spend = alias(&manifest, "spend_bookings.spend");
    let partials = aggregate_step(&join(&rows.input).left);
    let [(partial, count)] = partials.aggregates.as_slice() else {
        panic!("one partial count, got {:?}", partials.aggregates);
    };
    assert_eq!(partial.as_str(), format!("{spend}.cost"));
    assert_eq!(count.function().declarations.canonical.name, "count");
    let (entry, sum) = (aggregate.aggregates.iter())
        .find(|(_, sum)| computed(rows, &sum.args()[0]) == qualified(&spend, "cost"))
        .expect("the Joinset's sum of the partial counts");
    assert_eq!(sum.function().declarations.canonical.name, "sum");
    let zero = Expr::Literal(Literal::BigInt(0));
    let cast = Expr::Cast {
        input: Box::new(call("coalesce", vec![column(entry.as_str()), zero])),
        target: DataType::Double,
        on_failure: CastFailure::Error,
    };
    assert!(
        derivation.expressions.contains(&cast),
        "got {:?}",
        derivation.expressions
    );
    assert_eq!(output_names(&plan), ["date", "cost", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_measure_lowered_on_the_joinset_reads_rows_from_each_operand() {
    let measure = "    - name: vip_orders\n      data_type: bigint\n      agg: count\n      \
                   expr:\n        case:\n          whens:\n            \
                   - when: {eq: [segment, {lit: vip}]}\n              then: region\n  joinsets:\n";
    let yaml = (ORDERS_CUSTOMERS.replacen("  joinsets:\n", measure, 1)).replacen(
        "        - ref: revenue\n",
        "        - ref: revenue\n        - ref: vip_orders\n",
        1,
    );
    let manifest = orders_customers(&yaml).await;
    assert!(signature_of(&manifest, "orders_customers.orders", "vip_orders").is_none());

    let (plan, advisories) = planned(&manifest, &select(&["vip_orders"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate_steps(&plan.root), 1);
    let [(_, count)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(count.function().declarations.canonical.name, "count");
    let rows = projection(&aggregate.input);
    let [orders, customers] = ["orders", "customers"]
        .map(|operand| alias(&manifest, &format!("orders_customers.{operand}")));
    let vip = Expr::Case {
        whens: vec![(
            qualified(&customers, "segment").eq(text("vip")),
            qualified(&orders, "region"),
        )],
        else_: None,
    };
    assert_eq!(computed(rows, &count.args()[0]), vip);
    let join = join(&rows.input);
    for (operand, input) in [("orders", &join.left), ("customers", &join.right)] {
        assert!(
            matches!(*projection(input).input, PlanNode::Scan(_)),
            "{operand}"
        );
    }
    assert_eq!(output_names(&plan), ["vip_orders"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_gives_a_measure_filter_as_a_condition_field_that_the_aggregate_reads() {
    let filtered = "      agg: sum\n      additivity: non\n      filter:\n        \
                    gt_eq: [date, {lit: {date: \"2026-01-01\"}}]\n";
    let manifest = spend_bookings(&SPEND_BOOKINGS.replacen("      agg: sum\n", filtered, 1)).await;

    let (plan, advisories) = planned(&manifest, &select(&["campaign_id", "cost"]));

    let aggregate = aggregate_step(&plan.root);
    let [(_, cost)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let Some(Expr::Column(condition)) = cost.filter() else {
        panic!("a field of stage 2, got {:?}", cost.filter());
    };
    let rows = projection(&aggregate.input);
    let [spend, bookings] =
        ["spend", "bookings"].map(|operand| alias(&manifest, &format!("spend_bookings.{operand}")));
    let day = || Expr::Literal(Literal::Date("2026-01-01".into()));
    let own = qualified(&spend, "date").gt_eq(day()) & qualified(&bookings, "date").gt_eq(day());
    let written = fields(rows);
    let (_, computed) = (written.iter())
        .find(|(name, _)| *name == condition.name())
        .expect("the condition's field");
    assert_eq!(computed, &(own & qualified(&spend, "__plan_cond_cost")));
    let read = columns_of(&manifest, "spend");
    let expected = [
        ("campaign_id", read("campaign_id")),
        ("__plan_cond_cost", read("date").gt_eq(day())),
        ("cost", read("cost")),
        ("date", read("date")),
    ]
    .map(|(name, tree)| (format!("{spend}.{name}"), tree));
    let given: Vec<(String, PhysicalExpr)> = (fields(projection(&join(&rows.input).left)))
        .into_iter()
        .map(|(name, tree)| (name.to_owned(), tree))
        .collect();
    assert_eq!(given, expected);
    assert_eq!(output_names(&plan), ["campaign_id", "cost"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_s_physical_column_named_like_a_condition_field_is_no_aggregation_condition() {
    let flagged = (SPEND_BOOKINGS.replacen(
        "  measures:\n",
        "    - {name: flag, data_type: boolean, type: categorical}\n  measures:\n",
        1,
    ))
    .replacen(
        "        - ref: date\n",
        "        - ref: date\n        - ref: flag\n",
        1,
    )
    .replacen(
        "              tables: [warehouse.bookings]\n",
        "              tables: [warehouse.bookings]\n            semantic_mapping:\n              campaign_id: campaign_id\n              date: date\n              revenue: revenue\n              flag: __plan_cond_revenue\n",
        1,
    );
    let manifest = spend_bookings_with(&flagged, &[("__plan_cond_revenue", "boolean")]).await;

    for names in [
        &["flag", "cost", "revenue"][..],
        &["campaign_id", "flag", "revenue"],
    ] {
        let (plan, _) = planned(&manifest, &select(names));

        let filters = aggregate_filters(&plan.root);
        assert!(!filters.is_empty(), "{names:?}");
        assert!(
            filters.iter().all(|(_, filter)| filter.is_none()),
            "{names:?}: {filters:?}"
        );
        assert_eq!(output_names(&plan), names);
    }
}

#[tokio::test]
async fn an_operand_gives_the_literal_argument_of_a_count_as_its_row_field() {
    let counted = (SPEND_BOOKINGS.replacen(
        "  joinsets:\n",
        "    - name: booking_count\n      data_type: bigint\n      agg: count\n      \
         additivity: non\n  joinsets:\n",
        1,
    ))
    .replacen(
        "        - ref: revenue\n",
        "        - ref: revenue\n        - ref: booking_count\n",
        1,
    )
    .replacen(
        "              tables: [warehouse.bookings]\n",
        "              tables: [warehouse.bookings]\n            semantic_mapping:\n              campaign_id: campaign_id\n              date: date\n              revenue: revenue\n              booking_count: {lit: 1}\n",
        1,
    );
    let manifest = spend_bookings(&counted).await;

    let (plan, advisories) = planned(&manifest, &select(&["campaign_id", "booking_count"]));

    let aggregate = aggregate_step(&plan.root);
    let [(_, count)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let [Expr::Column(argument)] = count.args() else {
        panic!("a field of stage 2, got {:?}", count.args());
    };
    let rows = projection(&aggregate.input);
    let bookings = alias(&manifest, "spend_bookings.bookings");
    let written = fields(rows);
    let (_, computed) = (written.iter())
        .find(|(name, _)| *name == argument.name())
        .expect("the argument's field");
    assert_eq!(computed, &qualified(&bookings, "booking_count"));
    let given = fields(projection(&join(&rows.input).right));
    let field = format!("{bookings}.booking_count");
    let (_, row) = (given.iter())
        .find(|(name, _)| *name == field)
        .expect("the row argument's field");
    assert!(matches!(row, Expr::Literal(_)), "got {row:?}");
    assert_eq!(output_names(&plan), ["campaign_id", "booking_count"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_cross_combine_holds_the_relationship_filter_in_a_filter_node_above_it() {
    let manifest = tiered_orders().await;

    let (plan, advisories) = planned(&manifest, &select(&["order_id", "tier"]));

    let above = filter(&below_aggregate(&plan).input);
    let crossed = cross(&above.input);
    let orders = alias(&manifest, "tiered_orders.orders");
    let tiers = alias(&manifest, "tiered_orders.tiers");
    assert_eq!(
        above.filter,
        qualified(&orders, "amount").gt_eq(qualified(&tiers, "floor"))
    );
    assert_eq!(
        crossed.common.semantic_meta,
        combined("tiered_orders", Cardinality::ManyToMany)
    );
    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Join(_))));
    assert_eq!(output_names(&plan), ["order_id", "tier"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_name_both_operands_map_reads_the_coalesce_from_then_to() {
    let manifest = channels(CHANNELS).await;

    let (plan, advisories) = planned(&manifest, &select(&["campaign_id", "revenue"]));

    let aggregate = aggregate_step(&plan.root);
    let rows = projection(&aggregate.input);
    assert_eq!(join(&rows.input).join_type, JoinType::Full);
    let (web, store) = (
        alias(&manifest, "channels.web"),
        alias(&manifest, "channels.store"),
    );
    let coalesce = |name| {
        call(
            "coalesce",
            vec![qualified(&web, name), qualified(&store, name)],
        )
    };
    let written = fields(rows);
    let [(dimension, campaign_id), (argument, revenue)] = written.as_slice() else {
        panic!("two fields, got {written:?}");
    };
    assert_eq!(
        (*dimension, campaign_id),
        ("campaign_id", &coalesce("campaign_id"))
    );
    assert_eq!(revenue, &coalesce("revenue"));
    let [(_, sum)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    assert_eq!(sum.args(), [column(argument)]);
    assert_eq!(output_names(&plan), ["campaign_id", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_measure_both_operands_map_reads_rows_from_both_where_one_gives_rows() {
    let measure = "      agg: sum\n    - name: avg_revenue\n      data_type: double\n      \
                   agg: avg\n";
    let yaml = (CHANNELS.replacen("      agg: sum\n", measure, 1))
        .replacen(
            "        - ref: revenue\n",
            "        - ref: revenue\n        - ref: avg_revenue\n",
            1,
        )
        .replacen(
            "              tables: [warehouse.web]\n",
            "              tables: [warehouse.web]\n            semantic_mapping:\n              campaign_id: campaign_id\n              revenue: revenue\n              avg_revenue: revenue\n",
            1,
        );
    let manifest = channels(&yaml).await;
    assert!(signature_of(&manifest, "channels.store", "avg_revenue").is_none());
    let request = select(&["campaign_id", "revenue", "avg_revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate_steps(&plan.root), 1);
    let rows = projection(&aggregate.input);
    let join = join(&rows.input);
    for (operand, input) in [("web", &join.left), ("store", &join.right)] {
        assert!(
            matches!(*projection(input).input, PlanNode::Scan(_)),
            "{operand}"
        );
    }
    let [web, store] =
        ["web", "store"].map(|operand| alias(&manifest, &format!("channels.{operand}")));
    let (_, sum) = (aggregate.aggregates.iter())
        .find(|(entry, _)| entry.as_str() == "revenue")
        .expect("the Joinset's entry");
    assert_eq!(
        computed(rows, &sum.args()[0]),
        call(
            "coalesce",
            vec![qualified(&web, "revenue"), qualified(&store, "revenue")]
        )
    );
    assert_eq!(
        output_names(&plan),
        ["campaign_id", "revenue", "avg_revenue"]
    );
    assert!(advisories.is_empty());
}

async fn channels_filtering(filter: &str) -> SemanticManifest {
    let filtered =
        format!("      agg: sum\n      filter:\n        not_eq: [{filter}, {{lit: house}}]\n");
    channels(&CHANNELS.replacen("      agg: sum\n", &filtered, 1)).await
}

async fn a_measure_both_operands_map_with_their_conditions_and_no_own_filter() -> Golden {
    let manifest = channels_filtering("{col: campaign_id}").await;
    let own_filter = |source| (signature(&manifest, source, "revenue").filter_expr_id).is_some();
    let sources = ["channels.web", "channels.store", "channels"];
    assert_eq!(sources.map(own_filter), [true, true, false]);

    let (error, advisories) = refusal(&manifest, &select(&["revenue"]));

    let PlanError::AmbiguousOperandCondition {
        semantics,
        semantic_source,
        ..
    } = &error.kind
    else {
        panic!("an ambiguous operand condition, got {error:?}");
    };
    assert_eq!(
        (semantics.as_str(), semantic_source.as_str()),
        ("revenue", "channels")
    );
    let path = error.kind.path().map(ToString::to_string);
    assert_eq!(path.as_deref(), Some("channels"));
    assert!(advisories.is_empty());
    Golden::message("ambiguous_operand_condition", &error)
}

#[tokio::test]
async fn a_measure_both_operands_map_with_their_conditions_and_no_own_filter_is_refused() {
    (a_measure_both_operands_map_with_their_conditions_and_no_own_filter().await)
        .assert_committed();
}

#[tokio::test]
async fn a_measure_both_operands_map_reads_the_own_filter_of_its_composition_alone() {
    let manifest = channels_filtering("campaign_id").await;
    assert!((signature(&manifest, "channels", "revenue").filter_expr_id).is_some());

    let (plan, advisories) = planned(&manifest, &select(&["revenue"]));

    let aggregate = aggregate_step(&plan.root);
    let [(_, revenue)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let condition = revenue.filter().expect("a condition");
    let [web, store] =
        ["web", "store"].map(|operand| alias(&manifest, &format!("channels.{operand}")));
    let copy = |qualifier| qualified(qualifier, "campaign_id").neq(text("house"));
    assert_eq!(
        computed(projection(&aggregate.input), condition),
        copy(&web) & copy(&store)
    );
    assert_eq!(output_names(&plan), ["revenue"]);
    assert!(advisories.is_empty());
}

const SHOP_METRICS: &str = r#"  metrics:
    - name: vip_weighted_revenue
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [segment, {lit: vip}]}
              then: {multiply: [revenue, {lit: 2.0}]}
          else: revenue
    - name: revenue_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [revenue, {lit: 1.2}]}
          else: revenue
  joinsets:
"#;

async fn shop_orders_customers() -> SemanticManifest {
    orders_customers_with(&shop_orders_customers_yaml(), "double").await
}

fn shop_orders_customers_yaml() -> String {
    let double = "      data_type: double\n      agg: sum\n";
    (ORDERS_CUSTOMERS.replacen("      data_type: bigint\n      agg: sum\n", double, 1))
        .replacen("  joinsets:\n", SHOP_METRICS, 1)
        .replacen(
            "        - ref: revenue\n      datasets:\n",
            "        - ref: revenue\n      metrics:\n        - ref: vip_weighted_revenue\n        \
             - ref: revenue_adj\n      datasets:\n",
            1,
        )
}

#[tokio::test]
async fn a_metric_on_a_joinset_pushes_down_into_one_sum_above_the_join() {
    let manifest = shop_orders_customers().await;

    let (plan, advisories) = planned(&manifest, &select(&["region", "vip_weighted_revenue"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate_steps(&plan.root), 1);
    assert_eq!(aggregate.groupings, [column("region")]);
    let [orders, customers] = ["orders", "customers"]
        .map(|operand| alias(&manifest, &format!("orders_customers.{operand}")));
    let rows = projection(&aggregate.input);
    let revenue = qualified(&orders, "revenue");
    let term = Expr::Case {
        whens: vec![(
            qualified(&customers, "segment").eq(text("vip")),
            revenue.clone() * Expr::Literal(Literal::Double(2.0)),
        )],
        else_: Some(Box::new(revenue)),
    };
    assert_eq!(
        summed(rows, &[], entry(aggregate, "vip_weighted_revenue")).0,
        term
    );
    let join = join(&rows.input);
    for (operand, input) in [("orders", &join.left), ("customers", &join.right)] {
        assert!(
            matches!(*projection(input).input, PlanNode::Scan(_)),
            "{operand}"
        );
    }
    assert_eq!(output_names(&plan), ["region", "vip_weighted_revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_grouped_metric_on_a_joinset_reads_the_field_of_the_step_above_the_join() {
    let manifest = shop_orders_customers().await;

    let (plan, advisories) = planned(&manifest, &select(&["segment", "vip_weighted_revenue"]));

    let derivation = projection(&plan.root);
    let aggregate = aggregate_step(&derivation.input);
    assert_eq!(aggregate.groupings, [column("segment")]);
    let [segment, Expr::Case { whens, .. }] = derivation.expressions.as_slice() else {
        panic!(
            "the field and the formula, got {:?}",
            derivation.expressions
        );
    };
    assert_eq!(segment, &column("segment"));
    assert_eq!(whens[0].0, column("segment").eq(text("vip")));
    join(&projection(&aggregate.input).input);
    assert_eq!(output_names(&plan), ["segment", "vip_weighted_revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_pre_aggregates_the_formula_push_downs_sum_that_reads_it_alone() {
    let manifest = shop_orders_customers().await;

    let (plan, advisories) = planned(&manifest, &select(&["segment", "revenue_adj"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("segment")]);
    let orders = alias(&manifest, "orders_customers.orders");
    let rows = projection(&aggregate.input);
    assert_eq!(
        summed(rows, &[], entry(aggregate, "revenue_adj")).0,
        qualified(&orders, "revenue_adj")
    );
    let partials = aggregate_step(&join(&rows.input).left);
    assert_eq!(partials.groupings, [qualified(&orders, "customer_id")]);
    let [(name, sum)] = partials.aggregates.as_slice() else {
        panic!("one partial sum, got {:?}", partials.aggregates);
    };
    assert_eq!(name.as_str(), format!("{orders}.revenue_adj"));
    let read = columns_of(&manifest, "orders");
    let term = Expr::Case {
        whens: vec![(
            read("region_code").eq(text("eu")),
            read("amount") * Expr::Literal(Literal::Double(1.2)),
        )],
        else_: Some(Box::new(read("amount"))),
    };
    assert_eq!(summed(projection(&partials.input), &[], sum).0, term);
    assert_eq!(output_names(&plan), ["segment", "revenue_adj"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_that_gives_rows_gives_the_fields_of_the_formula_push_down() {
    let measure = "    - name: avg_amount\n      data_type: double\n      agg: avg\n  metrics:\n";
    let mapping = "              revenue: amount\n";
    let yaml = (shop_orders_customers_yaml().replacen("  metrics:\n", measure, 1))
        .replacen(
            "        - ref: revenue\n",
            "        - ref: revenue\n        - ref: avg_amount\n",
            1,
        )
        .replacen(
            mapping,
            &format!("{mapping}              avg_amount: amount\n"),
            1,
        );
    let manifest = orders_customers_with(&yaml, "double").await;

    let (plan, advisories) = planned(
        &manifest,
        &select(&["segment", "avg_amount", "revenue_adj"]),
    );

    let aggregate = aggregate_step(&plan.root);
    let orders = alias(&manifest, "orders_customers.orders");
    let rows = projection(&aggregate.input);
    let term = Expr::Case {
        whens: vec![(
            qualified(&orders, "region").eq(text("eu")),
            qualified(&orders, "revenue") * Expr::Literal(Literal::Double(1.2)),
        )],
        else_: Some(Box::new(qualified(&orders, "revenue"))),
    };
    assert_eq!(summed(rows, &[], entry(aggregate, "revenue_adj")).0, term);
    assert!(matches!(
        *projection(&join(&rows.input).left).input,
        PlanNode::Scan(_)
    ));
    assert_eq!(
        output_names(&plan),
        ["segment", "avg_amount", "revenue_adj"]
    );
    assert!(advisories.is_empty());
}

const ORDER_SEGMENTS: &str = r#"
semantic_model:
  name: order_segments
  dimensions:
    - {name: order_id, data_type: bigint, type: categorical}
    - {name: customer_id, data_type: bigint, type: categorical}
    - {name: region, data_type: string, type: categorical}
    - {name: segment, data_type: string, type: categorical}
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
  joinsets:
    - name: sales
      dimensions: [{ref: order_id}, {ref: customer_id}, {ref: region}, {ref: segment}]
      measures: [{ref: revenue}]
      metrics: [{ref: revenue_adj}]
      datasets:
        - name: orders
          extras:
            storage: {tables: [warehouse.orders]}
            semantic_mapping: {order_id: order_id, customer_id: cust_key, revenue: amount}
        - name: customers
          extras:
            storage: {tables: [warehouse.customers]}
        - name: order_segments
          extras:
            storage: {tables: [warehouse.order_segments]}
      relationships:
        - {name: r1, from: orders, to: customers, on: [customer_id], cardinality: many_to_one, optional: left}
        - {name: r2, from: orders, to: order_segments, on: [order_id], cardinality: many_to_one}
"#;

#[tokio::test]
async fn a_joinset_operand_pre_aggregates_the_formula_push_downs_sum_over_its_own_join() {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "long"),
                ("cust_key", "long"),
                ("amount", "double"),
            ],
        )
        .table(
            "warehouse.customers",
            FileFormat::Iceberg,
            &[("customer_id", "long"), ("region", "string")],
        )
        .table(
            "warehouse.order_segments",
            FileFormat::Iceberg,
            &[("order_id", "long"), ("segment", "string")],
        );
    let manifest = compiled(ORDER_SEGMENTS, &catalog, &InMemoryFileSystem::new()).await;

    let (plan, advisories) = planned(&manifest, &select(&["segment", "revenue_adj"]));

    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("segment")]);
    let [orders, customers, r1] =
        ["orders", "customers", "r1"].map(|name| alias(&manifest, &format!("sales.{name}")));
    let rows = projection(&aggregate.input);
    assert_eq!(
        summed(rows, &[], entry(aggregate, "revenue_adj")).0,
        qualified(&r1, "revenue_adj")
    );
    let partials = aggregate_step(&join(&rows.input).left);
    assert_eq!(partials.groupings, [qualified(&orders, "order_id")]);
    let [(name, sum)] = partials.aggregates.as_slice() else {
        panic!("one partial sum, got {:?}", partials.aggregates);
    };
    assert_eq!(name.as_str(), format!("{r1}.revenue_adj"));
    let revenue = qualified(&orders, "revenue");
    let term = Expr::Case {
        whens: vec![(
            qualified(&customers, "region").eq(text("eu")),
            revenue.clone() * Expr::Literal(Literal::Double(1.2)),
        )],
        else_: Some(Box::new(revenue)),
    };
    let ends = projection(&partials.input);
    assert_eq!(summed(ends, &[], sum).0, term);
    assert_eq!(join(&ends.input).join_type, JoinType::Left);
    assert_eq!(output_names(&plan), ["segment", "revenue_adj"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_that_pre_aggregates_the_parts_groups_by_the_columns_of_the_key() {
    let metric = r#"  metrics:
    - name: revenue_eu_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {add: [revenue, {lit: 100.0}]}
          else: revenue
"#;
    let yaml = (shop_orders_customers_yaml().replacen("  metrics:\n", metric, 1)).replacen(
        "      metrics:\n",
        "      metrics:\n        - ref: revenue_eu_bonus\n",
        1,
    );
    let manifest = orders_customers_with(&yaml, "double").await;

    let (plan, advisories) = planned(&manifest, &select(&["segment", "revenue_eu_bonus"]));

    let (step_2, _, step_1) = parts(&plan.root);
    assert_eq!(step_2.groupings, [column("segment")]);
    let [segment, key] = step_1.groupings.as_slice() else {
        panic!("the grouping and the part key, got {:?}", step_1.groupings);
    };
    assert_eq!(segment, &column("segment"));
    let row_level = projection(&step_1.input);
    let orders = alias(&manifest, "orders_customers.orders");
    assert_eq!(
        computed(row_level, key),
        Expr::Case {
            whens: vec![(qualified(&orders, "region").eq(text("eu")), bigint(1))],
            else_: Some(Box::new(bigint(2))),
        }
    );
    let partials = aggregate_step(&join(&row_level.input).left);
    let read = columns_of(&manifest, "orders");
    assert_eq!(
        partials.groupings,
        [
            qualified(&orders, "region"),
            qualified(&orders, "customer_id")
        ]
    );
    let [(_, revenue)] = partials.aggregates.as_slice() else {
        panic!("one partial sum, got {:?}", partials.aggregates);
    };
    assert_eq!(
        summed(projection(&partials.input), &[], revenue).0,
        read("amount")
    );
    assert_eq!(output_names(&plan), ["segment", "revenue_eu_bonus"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_pre_aggregates_step_1_of_parts_whose_key_reads_the_other_operand() {
    let metric = r#"  metrics:
    - name: vip_bonus
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [segment, {lit: vip}]}
              then: {add: [revenue, {lit: 100.0}]}
          else: revenue
"#;
    let yaml = (shop_orders_customers_yaml().replacen("  metrics:\n", metric, 1)).replacen(
        "      metrics:\n",
        "      metrics:\n        - ref: vip_bonus\n",
        1,
    );
    let manifest = orders_customers_with(&yaml, "double").await;

    let (plan, _) = planned(&manifest, &select(&["region", "vip_bonus"]));

    let (step_2, _, step_1) = parts(&plan.root);
    assert_eq!(step_2.groupings, [column("region")]);
    let orders = alias(&manifest, "orders_customers.orders");
    let partials = aggregate_step(&join(&projection(&step_1.input).input).left);
    assert_eq!(
        partials.groupings,
        [
            qualified(&orders, "region"),
            qualified(&orders, "customer_id")
        ]
    );
    assert_eq!(output_names(&plan), ["region", "vip_bonus"]);
}

#[tokio::test]
async fn an_operand_pre_aggregates_no_formula_whose_measure_does_not_combine() {
    let measure = "    - name: avg_amount\n      data_type: double\n      agg: avg\n  metrics:
    - name: avg_adj
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [region, {lit: eu}]}
              then: {multiply: [avg_amount, {lit: 1.2}]}
          else: avg_amount
";
    let mapping = "              revenue: amount\n";
    let yaml = (shop_orders_customers_yaml().replacen("  metrics:\n", measure, 1))
        .replacen(
            "        - ref: revenue\n",
            "        - ref: revenue\n        - ref: avg_amount\n",
            1,
        )
        .replacen(
            "      metrics:\n",
            "      metrics:\n        - ref: avg_adj\n",
            1,
        )
        .replacen(
            mapping,
            &format!("{mapping}              avg_amount: amount\n"),
            1,
        );
    let manifest = orders_customers_with(&yaml, "double").await;

    let (plan, _) = planned(&manifest, &select(&["segment", "avg_adj"]));

    let (_, _, step_1) = parts(&plan.root);
    let rows = &join(&projection(&step_1.input).input).left;
    assert!(matches!(*projection(rows).input, PlanNode::Scan(_)));
    assert_eq!(output_names(&plan), ["segment", "avg_adj"]);
}

#[tokio::test]
async fn a_cumulative_join_reads_the_condition_field_that_its_operand_passes_up() {
    let manifest = sales(SALES).await;
    let smb = dim("segment").expect("a name").eq(lit("SMB"));
    let request = select(&["region_name", "segment", "revenue"]).with_predicate(smb);

    let (plan, advisories) = planned(&manifest, &request);

    let [orders, customers, regions, r1] = ["orders", "customers", "regions", "r1"]
        .map(|name| alias(&manifest, &format!("sales.{name}")));
    let outer = join(&below_aggregate(&plan).input);
    assert_eq!(outer.join_type, JoinType::Inner);
    assert_eq!(
        outer.on,
        qualified(&orders, "region").eq(qualified(&regions, "region"))
    );
    let partials = aggregate_step(&outer.left);
    assert_eq!(
        partials.groupings,
        [qualified(&r1, "segment"), qualified(&orders, "region")]
    );
    let [(entry, revenue)] = partials.aggregates.as_slice() else {
        panic!("one entry, got {:?}", partials.aggregates);
    };
    assert_eq!(entry.as_str(), format!("{r1}.revenue"));
    let [Expr::Column(argument)] = revenue.args() else {
        panic!("a field of stage 2, got {:?}", revenue.args());
    };
    let smb = filter(&partials.input);
    assert_eq!(smb.filter, qualified(&r1, "segment").eq(text("SMB")));
    let rows = projection(&smb.input);
    let passed_up = format!("{orders}.region");
    let (argument, segment) = (field_name(argument), format!("{r1}.segment"));
    assert_eq!(
        fields(rows),
        [
            (passed_up.as_str(), qualified(&orders, "region")),
            (argument.as_str(), qualified(&orders, "revenue")),
            (segment.as_str(), qualified(&customers, "segment")),
        ]
    );
    assert_eq!(join(&rows.input).join_type, JoinType::Left);
    let filters = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Filter(_)));
    assert_eq!(filters.count(), 1);
    assert_eq!(output_names(&plan), ["region_name", "segment", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_never_outputs_a_leaf_that_only_a_predicate_moved_into_it_reads() {
    let manifest = sales(SALES).await;
    let smb = dim("segment").expect("a name").eq(lit("SMB"));
    let request = select(&["region_name", "revenue"]).with_predicate(smb);

    let (plan, _) = planned(&manifest, &request);

    let [orders, r1] = ["orders", "r1"].map(|name| alias(&manifest, &format!("sales.{name}")));
    let outer = join(&below_aggregate(&plan).input);
    let partials = aggregate_step(&outer.left);
    assert_eq!(
        names(&partials.common.output_schema),
        [format!("{orders}.region"), format!("{r1}.revenue")]
    );
    let smb = filter(&partials.input);
    assert_eq!(smb.filter, qualified(&r1, "segment").eq(text("SMB")));
    assert_eq!(output_names(&plan), ["region_name", "revenue"]);
}

fn with_named_filter(yaml: &str, name: &str, expr: &str) -> String {
    let filters = format!(
        "      filters:\n        - name: {name}\n          expr:\n            {expr}      relationships:\n"
    );
    yaml.replace("      relationships:\n", &filters)
}

#[tokio::test]
async fn a_named_filter_over_a_null_filled_operand_is_placed_as_its_request_predicate() {
    let smb = "eq: [segment, {lit: SMB}]\n";
    let manifest = sales(&with_named_filter(SALES, "smb", smb)).await;
    let request = select(&["region_name", "revenue"]).with_filters(vec!["smb".into()]);

    let (plan, advisories) = planned(&manifest, &request);

    let [orders, r1] = ["orders", "r1"].map(|name| alias(&manifest, &format!("sales.{name}")));
    let outer = join(&below_aggregate(&plan).input);
    assert_eq!(outer.join_type, JoinType::Inner);
    let partials = aggregate_step(&outer.left);
    assert_eq!(
        names(&partials.common.output_schema),
        [format!("{orders}.region"), format!("{r1}.revenue")]
    );
    let smb = filter(&partials.input);
    assert_eq!(smb.filter, qualified(&r1, "segment").eq(text("SMB")));
    let left = join(&projection(&smb.input).input);
    assert_eq!(left.join_type, JoinType::Left);
    let outputs_smb = |node: &PlanNode| names(&node.common().output_schema).contains(&"smb");
    assert!(!plan.root.pre_order().any(outputs_smb));
    assert_eq!(output_names(&plan), ["region_name", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn each_operand_of_a_full_join_applies_its_own_copy_of_a_named_filter() {
    let not_house = "not_eq: [campaign_id, {lit: house}]\n";
    let manifest = channels(&with_named_filter(CHANNELS, "not_house", not_house)).await;
    let request = select(&["campaign_id", "revenue"]).with_filters(vec!["not_house".into()]);

    let (plan, advisories) = planned(&manifest, &request);

    let join = join(&projection(&aggregate_step(&plan.root).input).input);
    assert_eq!(join.join_type, JoinType::Full);
    for (operand, input) in [("web", &join.left), ("store", &join.right)] {
        let qualifier = alias(&manifest, &format!("channels.{operand}"));
        let copy = qualified(&qualifier, "campaign_id").neq(text("house"));
        let partials = aggregate_step(input);
        assert_eq!(filter(&partials.input).filter, copy, "{operand}");
    }
    let filters = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Filter(_)));
    assert_eq!(filters.count(), 2);
    assert_eq!(output_names(&plan), ["campaign_id", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_operand_with_no_stand_in_passes_up_once_a_field_that_two_conditions_read() {
    let filtered = (SALES.replace(
        "optional: left\n",
        "optional: left\n          integrity: none\n",
    ))
    .replace(
        "on: [region]\n",
        "on: [region]\n          filter:\n            not_eq: [region, {lit: LATAM}]\n",
    );
    let manifest = sales(&filtered).await;

    let (plan, _) = planned(&manifest, &select(&["region_name", "revenue"]));

    let [orders, r1] = ["orders", "r1"].map(|name| alias(&manifest, &format!("sales.{name}")));
    let outer = join(&below_aggregate(&plan).input);
    assert_eq!(
        outer.post_join_filter,
        Some(qualified(&orders, "region").neq(text("LATAM")))
    );
    assert_eq!(
        names(&outer.left.common().output_schema),
        [format!("{orders}.region"), format!("{r1}.revenue")]
    );
}

#[tokio::test]
async fn a_left_join_prunes_the_operand_that_no_demanded_name_reads() {
    let manifest = orders_customers(ORDERS_CUSTOMERS).await;

    let (plan, advisories) = planned(&manifest, &select(&["region", "revenue"]));

    let scans = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Scan(_)));
    assert_eq!(scans.count(), 1);
    assert_eq!(read_table(&plan), "orders");
    let mut read = names(&scan_of(&plan).schema);
    read.sort_unstable();
    assert_eq!(read, ["amount", "region_code"]);
    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Join(_))));
    let joinset = DataKindName::new("orders_customers");
    let pruned_name = DataKindName::new("orders_customers.customers");
    let trace = &plan.root.common().semantic_meta;
    assert!(
        trace.iter().any(|subject| matches!(
            subject,
            SemAnnotation::SemanticSource { semantic_source, pruned, .. }
                if *semantic_source == joinset && pruned.contains(&pruned_name)
        )),
        "{pruned_name:?} pruned by {joinset:?} in {trace:?}"
    );
    assert!(
        (trace.iter()).any(
            |subject| matches!(subject, SemAnnotation::Rank { scope_root, .. } if *scope_root == joinset)
        ),
        "scope root {joinset:?} in {trace:?}"
    );
    assert_eq!(output_names(&plan), ["region", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn an_inner_join_with_no_integrity_keeps_the_operand_that_no_demanded_name_reads() {
    let inner = ORDERS_CUSTOMERS.replace(
        "optional: left\n",
        "optional: none\n          integrity: none\n",
    );
    let manifest = orders_customers(&inner).await;

    let (plan, _) = planned(&manifest, &select(&["region", "revenue"]));

    let kept = join(&below_aggregate(&plan).input);
    assert_eq!(kept.join_type, JoinType::Inner);
    let pruned = |node: &PlanNode| {
        (node.common().semantic_meta.iter()).any(
            |subject| matches!(subject, SemAnnotation::SemanticSource { pruned, .. } if !pruned.is_empty()),
        )
    };
    assert!(!plan.root.pre_order().any(pruned));
}

#[tokio::test]
async fn a_sub_composition_that_serves_the_request_answers_for_the_scope_root() {
    let manifest = sales(SALES).await;
    let request = select(&["region_name", "revenue"]).with_from("sales");

    let (plan, advisories) = planned(&manifest, &request);

    let [orders, regions] =
        ["orders", "regions"].map(|name| alias(&manifest, &format!("sales.{name}")));
    let stand_in = join(&below_aggregate(&plan).input);
    assert_eq!(stand_in.join_type, JoinType::Inner);
    assert_eq!(
        stand_in.on,
        qualified(&orders, "region").eq(qualified(&regions, "region"))
    );
    assert_eq!(datasets(&plan), ["sales.orders", "sales.regions"]);
    let trace = &plan.root.common().semantic_meta;
    let answered = DataKindName::new("sales.r2");
    assert!(
        trace.iter().any(
            |subject| matches!(subject, SemAnnotation::SemanticSource { answered_by: Some(name), .. } if *name == answered)
        ),
        "{answered:?} in {trace:?}"
    );
    let scope_root = DataKindName::new("sales");
    assert!(
        trace.iter().any(
            |subject| matches!(subject, SemAnnotation::Rank { scope_root: root, .. } if *root == scope_root)
        ),
        "{scope_root:?} in {trace:?}"
    );
    assert_eq!(output_names(&plan), ["region_name", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_bypass_that_would_leave_out_an_operand_of_a_filtered_relationship_is_never_taken() {
    let filtered = SALES.replace(
        "optional: left\n",
        "optional: left\n          filter:\n            not_eq: [segment, {lit: test}]\n",
    );
    the_scope_root_answers_as_itself(&filtered).await;
}

#[tokio::test]
async fn a_sub_composition_that_skips_a_filtered_relationship_of_a_cycle_never_answers() {
    let r3 = concat!(
        "        - name: r3\n",
        "          from: customers\n",
        "          to: regions\n",
        "          on: [region]\n",
        "          cardinality: many_to_one\n",
        "          filter:\n",
        "            not_eq: [segment, {lit: test}]\n",
    );
    the_scope_root_answers_as_itself(&format!("{SALES}{r3}")).await;
}

async fn the_scope_root_answers_as_itself(yaml: &str) {
    let manifest = sales(yaml).await;
    assert!((manifest.semantic_sources.values()).any(|unit| unit.name.as_str() == "sales.r2"));
    let request = select(&["region_name", "revenue"]).with_from("sales");

    let (plan, _) = planned(&manifest, &request);

    let datasets = datasets(&plan);
    assert_eq!(
        datasets,
        ["sales.orders", "sales.customers", "sales.regions"]
    );
    let answered = |node: &PlanNode| {
        (node.common().semantic_meta.iter()).any(|subject| {
            matches!(
                subject,
                SemAnnotation::SemanticSource {
                    answered_by: Some(_),
                    ..
                }
            )
        })
    };
    assert!(!plan.root.pre_order().any(answered));
}

#[tokio::test]
async fn a_tie_among_sub_compositions_goes_to_name_order() {
    let manifest = sales(SALES).await;
    let request = select(&["region", "revenue"]).with_from("sales");

    let (plan, _) = planned(&manifest, &request);

    let answered = DataKindName::new("sales.r1");
    let trace = &plan.root.common().semantic_meta;
    assert!(
        trace.iter().any(
            |subject| matches!(subject, SemAnnotation::SemanticSource { answered_by: Some(name), .. } if *name == answered)
        ),
        "{answered:?} in {trace:?}"
    );
}

#[tokio::test]
async fn the_advice_for_a_tie_of_sub_compositions_says_to_make_them_differ_in_the_model() {
    let manifest = sales(SALES).await;
    let request = select(&["region", "revenue"]).with_from("sales");

    let (_, advisories) = planned(&manifest, &request);

    let messages: Vec<_> = (advisories.into_iter())
        .map(|advisory| advisory.kind.message())
        .collect();
    assert_eq!(
        messages,
        ["at `sales`: `sales.r1` is chosen by name order over `sales.r2`: make them differ in the model"]
    );
}

const SINCE: &str = "gt_eq: [date, {lit: {date: \"2026-03-15\"}}]\n";

fn unfiltered() -> String {
    SPEND_BOOKINGS.replace(&format!("          filter:\n            {SINCE}"), "")
}

async fn late_revenue() -> SemanticManifest {
    let measure = format!(
        "    - name: late_revenue\n      data_type: bigint\n      agg: sum\n      filter:\n        {SINCE}  joinsets:\n"
    );
    let yaml = (unfiltered().replacen("  joinsets:\n", &measure, 1))
        .replace(
            "        - ref: revenue\n",
            "        - ref: revenue\n        - ref: late_revenue\n",
        )
        .replace(
            "              tables: [warehouse.bookings]\n",
            "              tables: [warehouse.bookings]\n            semantic_mapping:\n              campaign_id: campaign_id\n              date: date\n              revenue: revenue\n              late_revenue: revenue\n",
        );
    let manifest = spend_bookings(&yaml).await;
    assert!(signature_of(&manifest, "spend_bookings.spend", "late_revenue").is_none());
    assert!((signature(&manifest, "spend_bookings", "late_revenue").filter_expr_id).is_some());
    manifest
}

fn ides() -> PhysicalExpr {
    Expr::Literal(Literal::Date("2026-03-15".into()))
}

#[tokio::test]
async fn a_measure_filter_applies_on_the_one_operand_that_the_plan_reads() {
    let manifest = late_revenue().await;
    let request = at_granularity("date", Granularity::Month, &["late_revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    assert_eq!(read_table(&plan), "bookings");
    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Join(_))));
    let pruned = DataKindName::new("spend_bookings.spend");
    assert!((plan.root.common().semantic_meta.iter()).any(
        |subject| matches!(subject, SemAnnotation::SemanticSource { pruned: names, .. } if names.contains(&pruned))
    ));
    let aggregate = aggregate_step(&plan.root);
    assert_eq!(aggregate.groupings, [column("date")]);
    let [(_, late_revenue)] = aggregate.aggregates.as_slice() else {
        panic!("one entry, got {:?}", aggregate.aggregates);
    };
    let condition = late_revenue.filter().expect("a condition");
    let bucketing = projection(&aggregate.input);
    let date_trunc = call("date_trunc", vec![text("month"), column("date")]);
    assert!(fields(bucketing).contains(&("date", date_trunc)));
    assert_eq!(
        computed(projection(&bucketing.input), condition),
        columns_of(&manifest, "bookings")("date").gt_eq(ides())
    );
    assert_eq!(output_names(&plan), ["date", "late_revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_measure_filter_reads_the_copy_of_its_operand_where_the_plan_reads_both() {
    let manifest = late_revenue().await;
    let request = at_granularity("date", Granularity::Month, &["cost", "late_revenue"]);

    let (plan, advisories) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let [(_, cost), (_, late_revenue)] = aggregate.aggregates.as_slice() else {
        panic!("two entries, got {:?}", aggregate.aggregates);
    };
    assert_eq!(cost.filter(), None);
    let condition = late_revenue.filter().expect("a condition");
    let rows = projection(&below_aggregate(&plan).input);
    let [spend, bookings] =
        ["spend", "bookings"].map(|operand| alias(&manifest, &format!("spend_bookings.{operand}")));
    let own = qualified(&spend, "date").gt_eq(ides()) & qualified(&bookings, "date").gt_eq(ides());
    assert_eq!(computed(rows, condition), own);
    let join = join(&rows.input);
    let partials = aggregate_step(&join.right);
    let entry = format!("{bookings}.late_revenue");
    let (_, partial) = (partials.aggregates.iter())
        .find(|(name, _)| name.as_str() == entry)
        .expect("bookings' entry");
    let copy = partial.filter().expect("bookings' copy");
    assert_eq!(
        computed(projection(&partials.input), copy),
        columns_of(&manifest, "bookings")("date").gt_eq(ides())
    );
    let field = format!("{bookings}.__plan_cond_late_revenue");
    assert!(!names(&partials.common.output_schema).contains(&field.as_str()));
    let spend_fields = names(&join.left.common().output_schema);
    assert!(!spend_fields.iter().any(|name| name.contains("__plan_cond")));
    assert_eq!(output_names(&plan), ["date", "cost", "late_revenue"]);
    assert!(advisories.is_empty());
}

async fn by_channel(optional: &str) -> SemanticManifest {
    spend_bookings_with(&by_channel_model(optional), &[("channel", "string")]).await
}

fn by_channel_model(optional: &str) -> String {
    let channel = "    - name: channel\n      data_type: string\n      type: categorical\n";
    (unfiltered().replace("optional: none", &format!("optional: {optional}")))
        .replacen("  measures:\n", &format!("{channel}  measures:\n"), 1)
        .replace(
            "        - ref: date\n",
            "        - ref: date\n        - ref: channel\n",
        )
}

#[tokio::test]
async fn a_request_at_month_over_two_daily_operands_joins_their_days() {
    let manifest = by_channel("none").await;
    let request = at_granularity("date", Granularity::Month, &["channel", "cost"]);

    let (plan, advisories) = planned(&manifest, &request);

    let bucketing = below_aggregate(&plan);
    let date_trunc = call("date_trunc", vec![text("month"), column("date")]);
    assert!(fields(bucketing).contains(&("date", date_trunc)));
    let join = join(&projection(&bucketing.input).input);
    for (operand, input, grouped) in [
        ("spend", &join.left, &["date", "campaign_id"][..]),
        ("bookings", &join.right, &["date", "channel", "campaign_id"]),
    ] {
        let qualifier = alias(&manifest, &format!("spend_bookings.{operand}"));
        let grouped: Vec<_> = (grouped.iter())
            .map(|name| qualified(&qualifier, name))
            .collect();
        let partials = aggregate_step(input);
        assert_eq!(partials.groupings, grouped, "{operand}");
        let day = format!("{qualifier}.date");
        let read = columns_of(&manifest, operand)("date");
        let rows = projection(&partials.input);
        assert!(fields(rows).contains(&(day.as_str(), read)), "{operand}");
    }
    assert_eq!(aggregate_steps(&plan.root), 3);
    assert_eq!(output_names(&plan), ["date", "channel", "cost"]);
    assert!(advisories.is_empty());
}

fn since_ides(predicate: impl FnOnce(UnresolvedSemanticExpr) -> UnresolvedSemanticExpr) -> Request {
    let since =
        (dim("date").expect("a name")).gt_eq(Expr::Literal(Literal::Date("2026-03-15".into())));
    at_granularity("date", Granularity::Month, &["cost", "revenue"])
        .with_predicate(predicate(since))
}

#[tokio::test]
async fn a_copy_that_reads_both_operands_is_a_filter_node_above_the_join() {
    let manifest = by_channel("none").await;
    let web = |since| since | dim("channel").expect("a name").eq(lit("web"));
    let request = since_ides(web);

    let (plan, advisories) = planned(&manifest, &request);

    let [spend, bookings] =
        ["spend", "bookings"].map(|operand| alias(&manifest, &format!("spend_bookings.{operand}")));
    let copy = |date: &str| {
        qualified(date, "date").gt_eq(ides()) | qualified(&bookings, "channel").eq(text("web"))
    };
    let above = filter(&projection(&below_aggregate(&plan).input).input);
    assert_eq!(above.filter, copy(&spend));
    let join = join(&above.input);
    assert_eq!(join.join_type, JoinType::Inner);
    assert_eq!(join.post_join_filter, None);
    let day = format!("{spend}.date");
    let read = columns_of(&manifest, "spend")("date");
    let rows = projection(&aggregate_step(&join.left).input);
    assert!(fields(rows).contains(&(day.as_str(), read)));
    let partials = aggregate_step(&join.right);
    assert_eq!(filter(&partials.input).filter, copy(&bookings));
    let filters = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Filter(_)));
    assert_eq!(filters.count(), 2);
    assert_eq!(output_names(&plan), ["date", "cost", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn each_copy_of_a_conjunctive_named_filter_is_placed_as_its_request_predicate() {
    let late_web =
        "and: [{gt_eq: [date, {lit: {date: \"2026-03-15\"}}]}, {eq: [channel, {lit: web}]}]\n";
    let yaml = with_named_filter(&by_channel_model("right"), "late_web", late_web);
    let manifest = spend_bookings_with(&yaml, &[("channel", "string")]).await;
    let named = at_granularity("date", Granularity::Month, &["cost", "revenue"])
        .with_filters(vec!["late_web".into()]);
    let web = |since| since & dim("channel").expect("a name").eq(lit("web"));
    let [spend, bookings] =
        ["spend", "bookings"].map(|operand| alias(&manifest, &format!("spend_bookings.{operand}")));
    let copy = |date: &str| {
        qualified(date, "date").gt_eq(ides()) & qualified(&bookings, "channel").eq(text("web"))
    };

    for request in [named, since_ides(web)] {
        let (plan, advisories) = planned(&manifest, &request);

        let above = filter(&projection(&below_aggregate(&plan).input).input);
        assert_eq!(above.filter, copy(&spend));
        let join = join(&above.input);
        assert_eq!(join.join_type, JoinType::Right);
        let partials = aggregate_step(&join.right);
        assert_eq!(filter(&partials.input).filter, copy(&bookings));
        assert_eq!(output_names(&plan), ["date", "cost", "revenue"]);
        assert!(advisories.is_empty());
    }
}

fn each_operand_applies_its_own_copy_below_the_join(
    manifest: &SemanticManifest,
    request: &Request,
    join_type: JoinType,
) -> SemanticPlan {
    let (plan, advisories) = planned(manifest, request);

    let join = join(&projection(&below_aggregate(&plan).input).input);
    assert_eq!(join.join_type, join_type);
    for (operand, input) in [("spend", &join.left), ("bookings", &join.right)] {
        let qualifier = alias(manifest, &format!("spend_bookings.{operand}"));
        let copy = qualified(&qualifier, "date").gt_eq(ides());
        let partials = aggregate_step(input);
        assert_eq!(filter(&partials.input).filter, copy, "{operand}");
    }
    let filters = (plan.root.pre_order()).filter(|node| matches!(node, PlanNode::Filter(_)));
    assert_eq!(filters.count(), 2);
    assert_eq!(output_names(&plan), ["date", "cost", "revenue"]);
    assert!(advisories.is_empty());
    plan
}

#[tokio::test]
async fn each_operand_applies_its_own_copy_below_an_inner_join() {
    let manifest = by_channel("none").await;
    each_operand_applies_its_own_copy_below_the_join(
        &manifest,
        &since_ides(|since| since),
        JoinType::Inner,
    );
}

#[tokio::test]
async fn each_operand_applies_its_own_copy_below_a_left_join() {
    let manifest = by_channel("left").await;
    each_operand_applies_its_own_copy_below_the_join(
        &manifest,
        &since_ides(|since| since),
        JoinType::Left,
    );
}

async fn with_late() -> SemanticManifest {
    spend_bookings(&with_named_filter(&unfiltered(), "late", SINCE)).await
}

fn late(rest: &[&str]) -> Request {
    at_granularity("date", Granularity::Month, rest).with_filters(vec!["late".into()])
}

#[tokio::test]
async fn each_operand_applies_its_own_copy_of_a_named_filter_below_the_join() {
    let manifest = with_late().await;

    let plan = each_operand_applies_its_own_copy_below_the_join(
        &manifest,
        &late(&["cost", "revenue"]),
        JoinType::Inner,
    );

    let outputs_late = |node: &PlanNode| names(&node.common().output_schema).contains(&"late");
    assert!(!plan.root.pre_order().any(outputs_late));
}

#[tokio::test]
async fn the_copies_of_a_named_filter_keep_no_operand_so_the_joinset_prunes_one() {
    let manifest = with_late().await;

    let (plan, advisories) = planned(&manifest, &late(&["cost"]));

    assert_eq!(read_table(&plan), "spend");
    assert!(!(plan.root.pre_order()).any(|node| matches!(node, PlanNode::Join(_))));
    let pruned = DataKindName::new("spend_bookings.bookings");
    assert!((plan.root.common().semantic_meta.iter()).any(
        |subject| matches!(subject, SemAnnotation::SemanticSource { pruned: names, .. } if names.contains(&pruned))
    ));
    let filters: Vec<_> = (plan.root.pre_order())
        .filter_map(|node| match node {
            PlanNode::Filter(filter) => Some(&filter.filter),
            _ => None,
        })
        .collect();
    assert_eq!(filters, [&column("date").gt_eq(ides())]);
    assert_eq!(output_names(&plan), ["date", "cost"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![a_measure_both_operands_map_with_their_conditions_and_no_own_filter().await]
    })
    .await;
}
