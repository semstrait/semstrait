mod common;

use common::golden::{self, Golden};
use common::{
    aggregate_step, alias, at_granularity, bigint, column, compiled, computed, entry, every_marker,
    join, markers, names, once, output_names, parts, planned, projection, qualified, rank, refusal,
    row_number, select, text, written, Catalog, LATEST,
};
use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::Granularity;
use semstrait_ir::{JoinType, PhysicalExpr, SemanticPlan};
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{dim, lit};
use semstrait_planner::PlanError;
use semstrait_sql::expr::{Call, Expr, ExprOperationExt};

const AD_ORDERS: &str = r#"
semantic_model:
  name: ad_orders
  dimensions:
    - name: order_id
      data_type: string
      type: categorical
    - name: line_id
      data_type: string
      type: categorical
    - name: ad_id
      data_type: string
      type: categorical
    - name: ad_type
      data_type: string
      type: categorical
  measures:
    - name: total
      data_type: bigint
      agg: sum
    - name: avg_amount
      data_type: double
      agg: avg
  joinsets:
    - name: ad_orders
      dimensions:
        - ref: order_id
        - ref: line_id
        - ref: ad_id
        - ref: ad_type
      measures:
        - ref: total
        - ref: avg_amount
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              order_id: order_id
              total: amount
              avg_amount: amount
          dimensions:
            - ref: order_id
              is_key: true
          measures:
            - ref: total
            - ref: avg_amount
        - name: order_lines
          extras:
            storage:
              tables: [warehouse.order_lines]
          dimensions:
            - ref: line_id
              is_key: true
            - ref: order_id
            - ref: ad_id
        - name: placements
          extras:
            storage:
              tables: [warehouse.placements]
          dimensions:
            - ref: ad_id
              is_key: true
            - ref: ad_type
      relationships:
        - name: lines
          from: orders
          to: order_lines
          on: [order_id]
          cardinality: one_to_many
        - name: placed
          from: order_lines
          to: placements
          on: [ad_id]
          cardinality: many_to_one
"#;

async fn ad_orders(yaml: &str) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[("order_id", "string"), ("amount", "long")],
        )
        .table(
            "warehouse.order_lines",
            FileFormat::Iceberg,
            &[
                ("line_id", "string"),
                ("order_id", "string"),
                ("ad_id", "string"),
            ],
        )
        .table(
            "warehouse.placements",
            FileFormat::Iceberg,
            &[("ad_id", "string"), ("ad_type", "string")],
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

async fn rule_1_refused() -> Golden {
    let manifest = ad_orders(AD_ORDERS).await;

    let (error, advisories) = refusal(&manifest, &select(&["ad_type", "total"]));

    let PlanError::FanOutMarkerNotPlaceable { semantics, path } = &error.kind else {
        panic!("rule 1's refusal, got {error:?}");
    };
    assert_eq!(semantics.as_str(), "total");
    assert_eq!(path.to_string(), "ad_orders / ad_orders.lines");
    assert!(advisories.is_empty());
    Golden::message("rule_1_refused", &error)
}

#[tokio::test]
async fn partial_aggregates_grouped_by_a_name_the_fanned_row_does_not_determine_are_refused() {
    rule_1_refused().await.assert_committed();
}

#[tokio::test]
async fn a_request_whose_grain_holds_ad_id_is_planned() {
    let manifest = ad_orders(AD_ORDERS).await;

    planned(&manifest, &select(&["ad_id", "total"]));
}

#[tokio::test]
async fn a_marker_above_the_outer_join_reads_the_row_identity_that_the_rows_pass_up() {
    let manifest = ad_orders(AD_ORDERS).await;

    let (plan, _) = planned(&manifest, &select(&["ad_type", "avg_amount"]));

    let aggregate = aggregate_step(&plan.root);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let avg_amount = entry(aggregate, "avg_amount");
    let [argument] = avg_amount.args() else {
        panic!("one argument, got {:?}", avg_amount.args());
    };
    let identity = qualified(&alias(&manifest, "ad_orders.orders"), "order_id");
    let partition = vec![column("ad_type"), identity.clone(), argument.clone()];
    assert_eq!(window, &row_number(partition));
    assert_eq!(avg_amount.filter(), Some(&once(marker)));
    let row_level = projection(&marking.input);
    join(&row_level.input);
    assert_eq!(computed(row_level, &identity), identity);
    assert_eq!(output_names(&plan), ["ad_type", "avg_amount"]);
}

#[tokio::test]
async fn an_aggregate_whose_two_joins_repeat_operands_that_serve_it_is_refused() {
    let yaml = AD_ORDERS.replacen(
        "cardinality: many_to_one\n",
        "cardinality: many_to_many\n          optional: none\n          cross_filter: both\n",
        1,
    );
    let manifest = ad_orders(&yaml).await;

    let (error, _) = refusal(&manifest, &select(&["ad_type", "avg_amount"]));

    let PlanError::FanOutMarkerNotPlaceable { semantics, path } = &error.kind else {
        panic!("rule 3's refusal, got {error:?}");
    };
    assert_eq!(semantics.as_str(), "avg_amount");
    assert_eq!(path.to_string(), "ad_orders");
}

const SHIPPING: &str = r#"
semantic_model:
  name: shipping
  dimensions:
    - name: order_id
      data_type: string
      type: categorical
    - name: region
      data_type: string
      type: categorical
    - name: line_id
      data_type: string
      type: categorical
  measures:
    - name: avg_shipping
      data_type: double
      agg: avg
    - name: max_shipping
      data_type: double
      agg: max
    - name: shipped_orders
      data_type: bigint
      agg: count_distinct
    - name: avg_price
      data_type: double
      agg: avg
  joinsets:
    - name: shipping
      dimensions:
        - ref: order_id
          is_key: true
          keysets: [natural]
        - ref: region
        - ref: line_id
      measures:
        - ref: avg_shipping
        - ref: max_shipping
        - ref: shipped_orders
        - ref: avg_price
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              order_id: order_id
              region: region
              avg_shipping: shipping_cost
              max_shipping: shipping_cost
              shipped_orders: order_id
          dimensions:
            - ref: order_id
              is_key: true
            - ref: region
          measures:
            - ref: avg_shipping
            - ref: max_shipping
            - ref: shipped_orders
        - name: order_lines
          extras:
            storage:
              tables: [warehouse.order_lines]
            semantic_mapping:
              line_id: line_id
              order_id: order_id
              avg_price: price
          dimensions:
            - ref: line_id
              is_key: true
            - ref: order_id
          measures:
            - ref: avg_price
      relationships:
        - name: lines
          from: orders
          to: order_lines
          on: [order_id]
          cardinality: one_to_many
"#;

async fn shipping() -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "string"),
                ("region", "string"),
                ("shipping_cost", "double"),
            ],
        )
        .table(
            "warehouse.order_lines",
            FileFormat::Iceberg,
            &[
                ("line_id", "string"),
                ("order_id", "string"),
                ("price", "double"),
            ],
        );
    compiled(SHIPPING, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn an_aggregate_over_a_repeated_operand_reads_one_row_per_row_identity() {
    let manifest = shipping().await;

    let request = select(&["region", "avg_shipping", "avg_price"]);
    let (plan, advisories) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    assert!(marker.starts_with("__plan_"), "{marker}");
    let avg_shipping = entry(aggregate, "avg_shipping");
    let [argument] = avg_shipping.args() else {
        panic!("one argument, got {:?}", avg_shipping.args());
    };
    let orders = alias(&manifest, "shipping.orders");
    let identity = qualified(&orders, "order_id");
    let partition = vec![column("region"), identity.clone(), argument.clone()];
    assert_eq!(window, &row_number(partition));
    assert_eq!(avg_shipping.filter(), Some(&once(marker)));
    assert_eq!(entry(aggregate, "avg_price").filter(), None);
    let row_level = projection(&marking.input);
    join(&row_level.input);
    assert_eq!(
        computed(row_level, argument),
        qualified(&orders, "avg_shipping")
    );
    assert_eq!(computed(row_level, &identity), identity);
    assert_eq!(output_names(&plan), ["region", "avg_shipping", "avg_price"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn max_and_count_distinct_over_a_repeated_operand_read_no_marker() {
    let manifest = shipping().await;

    let names = [
        "region",
        "avg_shipping",
        "avg_price",
        "max_shipping",
        "shipped_orders",
    ];
    let (plan, _) = planned(&manifest, &select(&names));

    let aggregate = aggregate_step(&plan.root);
    let [(marker, _)] = &every_marker(&plan.root)[..] else {
        panic!("one marker, got {plan:?}");
    };
    assert_eq!(
        entry(aggregate, "avg_shipping").filter(),
        Some(&once(marker))
    );
    for measure in ["max_shipping", "shipped_orders"] {
        assert_eq!(entry(aggregate, measure).filter(), None, "{measure}");
    }
}

const CUSTOMER_ORDERS: &str = r#"
semantic_model:
  name: customer_orders
  dimensions:
    - name: customer_id
      data_type: string
      type: categorical
    - name: segment
      data_type: string
      type: categorical
    - name: tier
      data_type: string
      type: categorical
    - name: order_id
      data_type: string
      type: categorical
  measures:
    - name: avg_credit
      data_type: double
      agg: avg
    - name: avg_amount
      data_type: double
      agg: avg
  joinsets:
    - name: customer_orders
      dimensions:
        - ref: customer_id
        - ref: segment
        - ref: tier
        - ref: order_id
      measures:
        - ref: avg_credit
        - ref: avg_amount
      datasets:
        - name: customers
          extras:
            storage:
              tables: [warehouse.customers]
            semantic_mapping:
              customer_id: customer_id
              segment: segment
              tier: tier
              avg_credit: credit
          dimensions:
            - ref: customer_id
              is_key: true
            - ref: segment
            - ref: tier
          measures:
            - ref: avg_credit
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              order_id: order_id
              customer_id: customer_id
              avg_amount: amount
          dimensions:
            - ref: order_id
              is_key: true
            - ref: customer_id
          measures:
            - ref: avg_amount
      relationships:
        - name: placed
          from: customers
          to: orders
          on: [customer_id]
          cardinality: one_to_many
          optional: left
"#;

async fn customer_orders(yaml: &str) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.customers",
            FileFormat::Iceberg,
            &[
                ("customer_id", "string"),
                ("segment", "string"),
                ("tier", "string"),
                ("credit", "double"),
            ],
        )
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "string"),
                ("customer_id", "string"),
                ("amount", "double"),
            ],
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

#[tokio::test]
async fn the_row_identity_is_the_keyset_else_every_dimension_of_the_fanned_operand() {
    let keyed = CUSTOMER_ORDERS.replacen(
        "        - ref: customer_id\n",
        "        - ref: customer_id\n          is_key: true\n          keysets: [natural]\n",
        1,
    );
    for (yaml, identity) in [
        (keyed.as_str(), &["customer_id"][..]),
        (CUSTOMER_ORDERS, &["customer_id", "segment", "tier"]),
    ] {
        let manifest = customer_orders(yaml).await;

        let request = select(&["segment", "avg_credit", "avg_amount"]);
        let (plan, _) = planned(&manifest, &request);

        let aggregate = aggregate_step(&plan.root);
        let marking = projection(&aggregate.input);
        let [(marker, window)] = &markers(marking)[..] else {
            panic!("one marker, got {marking:?}");
        };
        let avg_credit = entry(aggregate, "avg_credit");
        let [argument] = avg_credit.args() else {
            panic!("one argument, got {:?}", avg_credit.args());
        };
        let mut identity = identity.to_vec();
        identity.sort_by_key(|name| {
            let definition = manifest.semantic_definitions.by_name(name);
            definition.expect("the Dimension").bit_position
        });
        let customers = alias(&manifest, "customer_orders.customers");
        let partition = (std::iter::once(column("segment")))
            .chain(identity.iter().map(|name| qualified(&customers, name)))
            .chain([argument.clone()])
            .collect();
        assert_eq!(window, &row_number(partition), "identity: {identity:?}");
        assert_eq!(avg_credit.filter(), Some(&once(marker)));
        assert_eq!(entry(aggregate, "avg_amount").filter(), None);
        assert_eq!(
            join(&projection(&marking.input).input).join_type,
            JoinType::Left
        );
    }
}

const GIFTS: &str = r#"
semantic_model:
  name: gifts
  dimensions:
    - name: order_id
      data_type: string
      type: categorical
    - name: region
      data_type: string
      type: categorical
    - name: amount
      data_type: double
      type: categorical
    - name: line_id
      data_type: string
      type: categorical
    - name: kind
      data_type: string
      type: categorical
  measures:
    - name: gift_amount
      data_type: double
      agg: sum
      expr:
        case:
          whens:
            - when: {eq: [kind, {lit: gift}]}
              then: amount
    - name: line_count
      data_type: bigint
      agg: count
  joinsets:
    - name: orders_order_lines
      dimensions:
        - ref: order_id
          is_key: true
          keysets: [natural]
        - ref: region
        - ref: amount
        - ref: line_id
        - ref: kind
      measures:
        - ref: gift_amount
        - ref: line_count
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              order_id: order_id
              region: region
              amount: amount
          dimensions:
            - ref: order_id
              is_key: true
            - ref: region
            - ref: amount
        - name: order_lines
          extras:
            storage:
              tables: [warehouse.order_lines]
            semantic_mapping:
              line_id: line_id
              order_id: order_id
              kind: kind
              line_count: line_id
          dimensions:
            - ref: line_id
              is_key: true
            - ref: order_id
            - ref: kind
          measures:
            - ref: line_count
      relationships:
        - name: lines
          from: orders
          to: order_lines
          on: [order_id]
          cardinality: one_to_many
"#;

async fn gifts(yaml: &str) -> SemanticManifest {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "string"),
                ("region", "string"),
                ("amount", "double"),
            ],
        )
        .table(
            "warehouse.order_lines",
            FileFormat::Iceberg,
            &[
                ("line_id", "string"),
                ("order_id", "string"),
                ("kind", "string"),
            ],
        );
    compiled(yaml, &catalog, &InMemoryFileSystem::new()).await
}

fn partition(window: &PhysicalExpr) -> &[PhysicalExpr] {
    let Expr::Function(Call::Window(call)) = window else {
        panic!("a window call, got {window:?}");
    };
    &call.over().partition_by
}

#[tokio::test]
async fn a_result_whose_value_a_join_repeats_reads_one_row_per_row_identity() {
    let manifest = gifts(GIFTS).await;

    let request = select(&["region", "gift_amount", "line_count"]).with_from("orders_order_lines");
    let (plan, advisories) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let gift_amount = entry(aggregate, "gift_amount");
    let [argument] = gift_amount.args() else {
        panic!("one argument, got {:?}", gift_amount.args());
    };
    let [.., position] = partition(window) else {
        panic!("a partition, got {window:?}");
    };
    let [orders, order_lines] = ["orders", "order_lines"]
        .map(|operand| alias(&manifest, &format!("orders_order_lines.{operand}")));
    let identity = qualified(&orders, "order_id");
    let partition = vec![
        column("region"),
        identity,
        argument.clone(),
        position.clone(),
    ];
    assert_eq!(window, &row_number(partition));
    let joined_rows = position.clone().eq(bigint(2));
    assert_eq!(gift_amount.filter(), Some(&(once(marker) | joined_rows)));
    assert_eq!(entry(aggregate, "line_count").filter(), None);
    let row_level = projection(&marking.input);
    join(&row_level.input);
    let gift = qualified(&order_lines, "kind").eq(text("gift"));
    let selected = |then, else_| Expr::Case {
        whens: vec![(gift.clone(), then)],
        else_,
    };
    assert_eq!(
        computed(row_level, argument),
        selected(qualified(&orders, "amount"), None)
    );
    assert_eq!(
        computed(row_level, position),
        selected(bigint(1), Some(Box::new(bigint(2))))
    );
    assert_eq!(output_names(&plan), ["region", "gift_amount", "line_count"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_measure_filtered_over_the_other_operand_reads_one_row_per_row_identity() {
    let filtered = "      filter:\n        eq: [kind, {lit: gift}]\n";
    let yaml = (GIFTS.replacen(
        "      expr:\n        case:\n          whens:\n            \
         - when: {eq: [kind, {lit: gift}]}\n              then: amount\n",
        filtered,
        1,
    ))
    .replacen(
        "              amount: amount\n",
        "              amount: amount\n              gift_amount: amount\n",
        1,
    )
    .replacen(
        "            - ref: amount\n        - name: order_lines\n",
        "            - ref: amount\n          measures:\n            - ref: gift_amount\n        \
         - name: order_lines\n",
        1,
    );
    let manifest = gifts(&yaml).await;

    let request = select(&["region", "gift_amount", "line_count"]).with_from("orders_order_lines");
    let (plan, _) = planned(&manifest, &request);

    let aggregate = aggregate_step(&projection(&projection(&plan.root).input).input);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let gift_amount = entry(aggregate, "gift_amount");
    let [argument] = gift_amount.args() else {
        panic!("one argument, got {:?}", gift_amount.args());
    };
    let [.., condition] = partition(window) else {
        panic!("a partition, got {window:?}");
    };
    let [orders, order_lines] = ["orders", "order_lines"]
        .map(|operand| alias(&manifest, &format!("orders_order_lines.{operand}")));
    let identity = qualified(&orders, "order_id");
    let partition = vec![
        column("region"),
        identity,
        argument.clone(),
        condition.clone(),
    ];
    assert_eq!(window, &row_number(partition));
    assert_eq!(
        gift_amount.filter(),
        Some(&(condition.clone() & once(marker)))
    );
    let row_level = projection(&marking.input);
    let gift = qualified(&order_lines, "kind").eq(text("gift"));
    assert_eq!(computed(row_level, condition), gift);
    assert_eq!(
        computed(row_level, argument),
        qualified(&orders, "gift_amount")
    );
}

#[tokio::test]
async fn a_formula_whose_link_a_join_repeats_reads_each_repeated_row_once_in_each_part() {
    let metric = r#"  metrics:
    - name: gift_revenue
      data_type: double
      expr:
        case:
          whens:
            - when: {eq: [kind, {lit: gift}]}
              then: revenue
  joinsets:
"#;
    let yaml = (GIFTS.replacen(
        "  joinsets:\n",
        &format!("    - name: revenue\n      data_type: double\n      agg: sum\n{metric}"),
        1,
    ))
    .replacen(
        "        - ref: line_count\n      datasets:\n",
        "        - ref: line_count\n        - ref: revenue\n      metrics:\n        \
         - ref: gift_revenue\n      datasets:\n",
        1,
    )
    .replacen(
        "              amount: amount\n",
        "              amount: amount\n              revenue: amount\n",
        1,
    )
    .replacen(
        "            - ref: amount\n        - name: order_lines\n",
        "            - ref: amount\n          measures:\n            - ref: revenue\n        \
         - name: order_lines\n",
        1,
    );
    let manifest = gifts(&yaml).await;

    let request = select(&["region", "gift_revenue"]).with_from("orders_order_lines");
    let (plan, advisories) = planned(&manifest, &request);

    let (step_2, per_part, step_1) = parts(&plan.root);
    assert_eq!(step_2.groupings, [column("region")]);
    let [region, key] = step_1.groupings.as_slice() else {
        panic!("the grouping and the part key, got {:?}", step_1.groupings);
    };
    assert_eq!(region, &column("region"));
    let [(entry, revenue)] = step_1.aggregates.as_slice() else {
        panic!("one sum, got {:?}", step_1.aggregates);
    };
    assert_eq!(
        per_part,
        Expr::Case {
            whens: vec![(key.clone().eq(bigint(1)), column(entry.as_str()))],
            else_: None,
        }
    );
    let marking = projection(&step_1.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    assert_eq!(revenue.filter(), Some(&once(marker)));
    let [argument] = revenue.args() else {
        panic!("one argument, got {:?}", revenue.args());
    };
    let [orders, order_lines] = ["orders", "order_lines"]
        .map(|operand| alias(&manifest, &format!("orders_order_lines.{operand}")));
    let partition = vec![
        region.clone(),
        key.clone(),
        qualified(&orders, "order_id"),
        argument.clone(),
    ];
    assert_eq!(window, &row_number(partition));
    let row_level = projection(&marking.input);
    join(&row_level.input);
    assert_eq!(
        computed(row_level, key),
        Expr::Case {
            whens: vec![(qualified(&order_lines, "kind").eq(text("gift")), bigint(1))],
            else_: None,
        }
    );
    assert_eq!(computed(row_level, argument), qualified(&orders, "revenue"));
    assert_eq!(output_names(&plan), ["region", "gift_revenue"]);
    assert!(advisories.is_empty());
}

const ORDERS_CUSTOMERS: &str = r#"
semantic_model:
  name: orders_customers
  dimensions:
    - name: order_id
      data_type: string
      type: categorical
    - name: region
      data_type: string
      type: categorical
    - name: customer_id
      data_type: string
      type: categorical
    - name: segment
      data_type: string
      type: categorical
  measures:
    - name: vip_orders
      data_type: bigint
      agg: count
      expr:
        case:
          whens:
            - when: {eq: [segment, {lit: vip}]}
              then: order_id
  joinsets:
    - name: orders_customers
      dimensions:
        - ref: order_id
        - ref: region
        - ref: customer_id
        - ref: segment
      measures:
        - ref: vip_orders
      datasets:
        - name: orders_j
          extras:
            storage:
              tables: [warehouse.orders]
          dimensions:
            - ref: order_id
              is_key: true
            - ref: region
            - ref: customer_id
        - name: customers_j
          extras:
            storage:
              tables: [warehouse.customers]
          dimensions:
            - ref: customer_id
              is_key: true
            - ref: segment
      relationships:
        - name: orders_j_to_customers_j
          from: orders_j
          to: customers_j
          on: [customer_id]
          cardinality: many_to_one
"#;

#[tokio::test]
async fn a_result_whose_value_no_join_repeats_reads_the_joined_rows() {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "string"),
                ("region", "string"),
                ("customer_id", "string"),
            ],
        )
        .table(
            "warehouse.customers",
            FileFormat::Iceberg,
            &[("customer_id", "string"), ("segment", "string")],
        );
    let manifest = compiled(ORDERS_CUSTOMERS, &catalog, &InMemoryFileSystem::new()).await;

    let request = select(&["region", "vip_orders"]).with_from("orders_customers");
    let (plan, advisories) = planned(&manifest, &request);

    assert!(every_marker(&plan.root).is_empty(), "{plan:?}");
    let aggregate = aggregate_step(&plan.root);
    let vip_orders = entry(aggregate, "vip_orders");
    assert_eq!(vip_orders.filter(), None);
    let rows = projection(&aggregate.input);
    let [orders, customers] = ["orders_j", "customers_j"]
        .map(|operand| alias(&manifest, &format!("orders_customers.{operand}")));
    let vip = Expr::Case {
        whens: vec![(
            qualified(&customers, "segment").eq(text("vip")),
            qualified(&orders, "order_id"),
        )],
        else_: None,
    };
    assert_eq!(computed(rows, &vip_orders.args()[0]), vip);
    join(&rows.input);
    assert_eq!(output_names(&plan), ["region", "vip_orders"]);
    assert!(advisories.is_empty());
}

const TAGGED_SPEND: &str = r#"
semantic_model:
  name: tagged_spend
  dimensions:
    - name: campaign_id
      data_type: string
      type: categorical
    - name: tag
      data_type: string
      type: categorical
    - name: weight
      data_type: double
      type: categorical
    - name: cost
      data_type: double
      type: categorical
  measures:
    - name: weighted_cost
      data_type: double
      agg: sum
      expr:
        multiply: [weight, cost]
  joinsets:
    - name: tagged_spend
      dimensions:
        - ref: campaign_id
        - ref: tag
        - ref: weight
        - ref: cost
      measures:
        - ref: weighted_cost
      datasets:
        - name: tags
          extras:
            storage:
              tables: [warehouse.tags]
          dimensions:
            - ref: campaign_id
            - ref: tag
            - ref: weight
        - name: spend
          extras:
            storage:
              tables: [warehouse.spend]
          dimensions:
            - ref: campaign_id
            - ref: cost
      relationships:
        - name: tags_to_spend
          from: tags
          to: spend
          on: [campaign_id]
          cardinality: many_to_many
          optional: none
          cross_filter: both
"#;

async fn rule_3_refused() -> Golden {
    let catalog = Catalog::default()
        .table(
            "warehouse.tags",
            FileFormat::Iceberg,
            &[
                ("campaign_id", "string"),
                ("tag", "string"),
                ("weight", "double"),
            ],
        )
        .table(
            "warehouse.spend",
            FileFormat::Iceberg,
            &[("campaign_id", "string"), ("cost", "double")],
        );
    let manifest = compiled(TAGGED_SPEND, &catalog, &InMemoryFileSystem::new()).await;

    let request = select(&["tag", "weighted_cost"]).with_from("tagged_spend");
    let (error, advisories) = refusal(&manifest, &request);

    let PlanError::FanOutMarkerNotPlaceable { semantics, path } = &error.kind else {
        panic!("rule 3's refusal, got {error:?}");
    };
    assert_eq!(semantics.as_str(), "weighted_cost");
    assert_eq!(path.to_string(), "tagged_spend");
    assert!(advisories.is_empty());
    Golden::message("rule_3_refused", &error)
}

#[tokio::test]
async fn a_result_that_would_read_two_row_number_markers_is_refused() {
    rule_3_refused().await.assert_committed();
}

async fn holdings(semi: &str) -> SemanticManifest {
    let yaml = format!(
        r#"
semantic_model:
  name: holdings
  dimensions:
    - name: account
      data_type: string
      type: categorical
    - name: order_date
      data_type: date
      type:
        temporal:
          granularities: [day]
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
  joinsets:
    - name: holdings
      dimensions:
        - ref: account
        - ref: order_date
        - ref: date
      measures:
        - ref: balance
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              account: account
              order_date: order_date
              date: booked_on
            temporal:
              events:
                event_time: order_date
              granularity: day
          dimensions:
            - ref: account
            - ref: order_date
            - ref: date
        - name: balances
          extras:
            storage:
              tables: [warehouse.balances]
            semantic_mapping:
              account: account
              date: snap_date
              balance: balance
            temporal:
              snapshot:
                snapshotted_at: date
              granularity: day
          dimensions:
            - ref: account
            - ref: date
          measures:
            - ref: balance
      relationships:
        - name: held
          from: orders
          to: balances
          on: [account, date]
          cardinality: many_to_one
          optional: left
"#
    );
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("account", "string"),
                ("order_date", "date"),
                ("booked_on", "date"),
            ],
        )
        .table(
            "warehouse.balances",
            FileFormat::Iceberg,
            &[
                ("account", "string"),
                ("snap_date", "date"),
                ("balance", "long"),
            ],
        );
    compiled(&yaml, &catalog, &InMemoryFileSystem::new()).await
}

fn reads_balances_date(manifest: &SemanticManifest, plan: &SemanticPlan) {
    let aggregate = aggregate_step(&plan.root);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let balances = alias(manifest, "holdings.balances");
    let reserved = format!("__plan_unbucketed_{balances}_date");
    let by = column(&reserved);
    assert_eq!(window, &rank(vec![column("date")], by.clone(), LATEST));
    assert_eq!(entry(aggregate, "balance").filter(), Some(&once(marker)));
    let bucketing = projection(&marking.input);
    let row_level = projection(&bucketing.input);
    assert_eq!(written(bucketing, &reserved), Some(by));
    assert_eq!(
        written(row_level, &reserved),
        Some(qualified(&balances, "date"))
    );
    assert_eq!(join(&row_level.input).join_type, JoinType::Left);
    assert_eq!(output_names(plan), ["date", "balance"]);
}

#[tokio::test]
async fn a_rank_marker_on_a_joinset_reads_by_from_the_dataset_of_its_aggregate() {
    let manifest = holdings("{strategy: latest, by: date}").await;

    let request = at_granularity("date", Granularity::Month, &["balance"]).with_from("holdings");
    let (plan, advisories) = planned(&manifest, &request);

    reads_balances_date(&manifest, &plan);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_rank_marker_on_a_joinset_orders_by_the_time_axis_of_its_dataset() {
    let manifest = holdings("{strategy: latest}").await;

    let request = at_granularity("date", Granularity::Month, &["balance"]).with_from("holdings");
    let (plan, advisories) = planned(&manifest, &request);

    reads_balances_date(&manifest, &plan);
    assert!(advisories.is_empty());
}

const VALUED: &str = r#"
semantic_model:
  name: valued
  dimensions:
    - name: account
      data_type: string
      type: categorical
    - name: currency
      data_type: string
      type: categorical
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
    - name: units
      data_type: double
      type: categorical
    - name: price
      data_type: double
      type: categorical
  measures:
    - name: value
      data_type: double
      agg: sum
      expr:
        multiply: [units, price]
      additivity:
        semi: {strategy: latest, by: date, axes: [currency]}
  joinsets:
    - name: valued
      dimensions:
        - ref: account
        - ref: currency
        - ref: date
        - ref: units
        - ref: price
      measures:
        - ref: value
      datasets:
        - name: balances
          extras:
            storage:
              tables: [warehouse.balances]
            semantic_mapping:
              account: account
              date: snap_date
              units: units
            temporal:
              snapshot:
                snapshotted_at: date
              granularity: day
          dimensions:
            - ref: account
            - ref: date
            - ref: units
        - name: prices
          extras:
            storage:
              tables: [warehouse.prices]
          dimensions:
            - ref: account
            - ref: currency
            - ref: price
      relationships:
        - name: priced
          from: balances
          to: prices
          on: [account]
          cardinality: many_to_one
"#;

#[tokio::test]
async fn a_rank_marker_on_a_joinset_reads_each_axis_from_the_dataset_that_maps_it() {
    let catalog = Catalog::default()
        .table(
            "warehouse.balances",
            FileFormat::Iceberg,
            &[
                ("account", "string"),
                ("snap_date", "date"),
                ("units", "double"),
            ],
        )
        .table(
            "warehouse.prices",
            FileFormat::Iceberg,
            &[
                ("account", "string"),
                ("currency", "string"),
                ("price", "double"),
            ],
        );
    let manifest = compiled(VALUED, &catalog, &InMemoryFileSystem::new()).await;

    let request = at_granularity("date", Granularity::Month, &["value"]).with_from("valued");
    let (plan, _) = planned(&manifest, &request);

    let marking = projection(&aggregate_step(&plan.root).input);
    let [(_, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let currency = qualified(&alias(&manifest, "valued.prices"), "currency");
    assert_eq!(partition(window), [column("date"), currency]);
}

const DATED_LINES: &str = r#"
semantic_model:
  name: dated
  dimensions:
    - name: order_id
      data_type: string
      type: categorical
    - name: line_id
      data_type: string
      type: categorical
    - name: line_date
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: total
      data_type: bigint
      agg: sum
  joinsets:
    - name: dated
      dimensions:
        - ref: order_id
          is_key: true
          keysets: [natural]
        - ref: line_id
        - ref: line_date
      measures:
        - ref: total
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              order_id: order_id
              total: amount
          dimensions:
            - ref: order_id
              is_key: true
          measures:
            - ref: total
        - name: order_lines
          extras:
            storage:
              tables: [warehouse.order_lines]
          dimensions:
            - ref: line_id
              is_key: true
            - ref: order_id
            - ref: line_date
      relationships:
        - name: lines
          from: orders
          to: order_lines
          on: [order_id]
          cardinality: one_to_many
"#;

#[tokio::test]
async fn partial_groups_that_the_request_buckets_coarser_read_one_row_per_row_identity() {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[("order_id", "string"), ("amount", "long")],
        )
        .table(
            "warehouse.order_lines",
            FileFormat::Iceberg,
            &[
                ("line_id", "string"),
                ("order_id", "string"),
                ("line_date", "date"),
            ],
        );
    let manifest = compiled(DATED_LINES, &catalog, &InMemoryFileSystem::new()).await;

    let request = at_granularity("line_date", Granularity::Month, &["total"]).with_from("dated");
    let (plan, _) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let marking = projection(&aggregate.input);
    let [(marker, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let identity = qualified(&alias(&manifest, "dated.orders"), "order_id");
    assert_eq!(partition(window)[..2], [column("line_date"), identity]);
    assert_eq!(entry(aggregate, "total").filter(), Some(&once(marker)));
}

#[tokio::test]
async fn partial_groups_by_a_name_that_only_a_predicate_reads_are_refused() {
    let manifest = ad_orders(AD_ORDERS).await;

    let search = || dim("ad_type").expect("a name").eq(lit("search"));
    for (names, read, value) in [
        (["ad_type", "total"], "ad_id", "ad1"),
        (["ad_id", "total"], "line_id", "l1"),
    ] {
        let predicate = dim(read).expect("a name").eq(lit(value)) | search();
        let (error, _) = refusal(&manifest, &select(&names).with_predicate(predicate));

        let PlanError::FanOutMarkerNotPlaceable { semantics, path } = &error.kind else {
            panic!("rule 1's refusal where `{read}` is read, got {error:?}");
        };
        assert_eq!(semantics.as_str(), "total");
        assert_eq!(path.to_string(), "ad_orders / ad_orders.lines");
    }
}

const CURRENCY_HOLDINGS: &str = r#"
semantic_model:
  name: currency_holdings
  dimensions:
    - name: account
      data_type: string
      type: categorical
    - name: currency
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
        semi: {strategy: latest, by: date, axes: [currency]}
  joinsets:
    - name: holdings
      dimensions:
        - ref: account
        - ref: currency
        - ref: date
      measures:
        - ref: balance
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
          dimensions:
            - ref: account
            - ref: currency
            - ref: date
        - name: balances
          extras:
            storage:
              tables: [warehouse.balances]
            temporal:
              snapshot:
                snapshotted_at: date
              granularity: day
          dimensions:
            - ref: account
            - ref: currency
            - ref: date
          measures:
            - ref: balance
      relationships:
        - name: held
          from: orders
          to: balances
          on: [account, date]
          cardinality: many_to_one
          optional: left
"#;

#[tokio::test]
async fn a_rank_marker_partitions_by_its_datasets_axis_beside_a_coalesced_grouping() {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("account", "string"),
                ("currency", "string"),
                ("date", "date"),
            ],
        )
        .table(
            "warehouse.balances",
            FileFormat::Iceberg,
            &[
                ("account", "string"),
                ("currency", "string"),
                ("date", "date"),
                ("balance", "long"),
            ],
        );
    let manifest = compiled(CURRENCY_HOLDINGS, &catalog, &InMemoryFileSystem::new()).await;

    let request = select(&["currency", "balance"]).with_from("holdings");
    let (plan, _) = planned(&manifest, &request);

    let marking = projection(&aggregate_step(&plan.root).input);
    let [(_, window)] = &markers(marking)[..] else {
        panic!("one marker, got {marking:?}");
    };
    let currency = qualified(&alias(&manifest, "holdings.balances"), "currency");
    assert_eq!(partition(window), [column("currency"), currency]);
}

const LEDGER: &str = r#"
semantic_model:
  name: ledger
  dimensions:
    - name: order_id
      data_type: string
      type: categorical
    - name: account
      data_type: string
      type: categorical
    - name: date
      data_type: date
      type:
        temporal:
          granularities: [day]
  measures:
    - name: total
      data_type: bigint
      agg: sum
    - name: balance
      data_type: bigint
      agg: sum
      additivity:
        semi: {strategy: latest, by: date}
  joinsets:
    - name: ledger
      dimensions:
        - ref: order_id
        - ref: account
        - ref: date
      measures:
        - ref: total
        - ref: balance
      datasets:
        - name: orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping:
              order_id: order_id
              account: account
              total: amount
          dimensions:
            - ref: order_id
              is_key: true
            - ref: account
          measures:
            - ref: total
        - name: snapshots
          extras:
            storage:
              tables: [warehouse.snapshots]
            temporal:
              snapshot:
                snapshotted_at: date
              granularity: day
          dimensions:
            - ref: account
            - ref: date
          measures:
            - ref: balance
      relationships:
        - name: held
          from: orders
          to: snapshots
          on: [account]
          cardinality: one_to_many
"#;

#[tokio::test]
async fn a_row_identity_that_a_rank_read_makes_a_marker_need_is_passed_up() {
    let catalog = Catalog::default()
        .table(
            "warehouse.orders",
            FileFormat::Iceberg,
            &[
                ("order_id", "string"),
                ("account", "string"),
                ("amount", "long"),
            ],
        )
        .table(
            "warehouse.snapshots",
            FileFormat::Iceberg,
            &[("account", "string"), ("date", "date"), ("balance", "long")],
        );
    let manifest = compiled(LEDGER, &catalog, &InMemoryFileSystem::new()).await;

    let request = select(&["account", "total", "balance"]).with_from("ledger");
    let (plan, _) = planned(&manifest, &request);

    let aggregate = aggregate_step(&plan.root);
    let marking = projection(&aggregate.input);
    let total = entry(aggregate, "total").filter();
    let found = (markers(marking).into_iter()).find(|(marker, _)| total == Some(&once(marker)));
    let Some((_, window)) = found else {
        panic!("`total`'s marker, got {marking:?}");
    };
    let orders = alias(&manifest, "ledger.orders");
    assert!(partition(&window).contains(&qualified(&orders, "order_id")));
    let joined = join(&projection(&marking.input).input);
    let order_id = format!("{orders}.order_id");
    assert!(
        names(&joined.left.common().output_schema).contains(&order_id.as_str()),
        "{joined:?}"
    );
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async { vec![rule_1_refused().await, rule_3_refused().await] }).await;
}
