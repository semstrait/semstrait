use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{CompileError, KeptField, KeptLeaf, SemanticManifest};
use semstrait_model::expr::{col, dim, lit, measure, metric};
use semstrait_model::{Dimension, Measure, Metric, SemanticModel, UnresolvedSemanticExpr};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Aggregation, Expr, ExprOperationExt, Leaf, Literal, NamedReference};

mod common;
use common::golden::{self, Golden};
use common::*;

fn refunded() -> UnresolvedSemanticExpr {
    dim("status").expect("a name").eq(lit("refunded"))
}

#[tokio::test]
async fn a_row_level_metric_with_agg_is_lowered_as_a_measure() {
    let manifest = compile_model(orders_model_with_metrics(
        vec![dimension("status", DataType::String)],
        vec![Measure::builder("refunds")
            .data_type(DataType::Double)
            .agg(agg("sum"))
            .expr(col("amount").expect("a name"))
            .filter(refunded())
            .build()],
        vec![
            Metric::builder("refund_total")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .expr(col("amount").expect("a name"))
                .filter(refunded())
                .build(),
            Metric::builder("refund_total_twice")
                .data_type(DataType::Double)
                .expr(metric("refund_total").expect("a name") * lit(2.0))
                .build(),
        ],
    ))
    .await;
    let summed = |filter| {
        cast(
            aggregate("sum", false, orders_column("amount"), filter),
            DataType::Double,
        )
    };
    for semantic in ["refund_total", "refunds"] {
        let signature = signature(&manifest, "orders", semantic);
        assert_eq!(tree(&manifest, signature), summed(None), "{semantic}");
        assert_eq!(
            filter_tree(&manifest, signature),
            is("status", "refunded"),
            "{semantic}"
        );
    }
    assert_eq!(
        tree(
            &manifest,
            signature(&manifest, "orders", "refund_total_twice")
        ),
        summed(Some(is("status", "refunded"))) * double(2.0)
    );
}

#[tokio::test]
async fn a_row_level_metric_with_agg_over_a_constant_counts_rows() {
    let count = |name: &str, expr| {
        Metric::builder(name)
            .data_type(DataType::BigInt)
            .agg(agg("count"))
            .expr(expr)
            .build()
    };
    let manifest = compile_model(orders_model_with_metrics(
        vec![dimension("region", DataType::String)],
        Vec::new(),
        vec![
            count("n", lit(1)),
            Metric::builder("n_twice")
                .data_type(DataType::BigInt)
                .expr(metric("n").expect("a name") + metric("n").expect("a name"))
                .build(),
            Metric::builder("label")
                .data_type(DataType::String)
                .expr(dim("region").expect("a name"))
                .build(),
            count("labels", metric("label").expect("a name")),
        ],
    ))
    .await;
    let id = |semantic| signature(&manifest, "orders", semantic).expression_id;
    let counted = |argument| aggregate("count", false, argument, None);
    assert_eq!(
        stored_expr(&manifest, id("n")),
        &counted(Expr::Literal(Literal::Integer(1)))
    );
    assert_eq!(
        stored_expr(&manifest, id("n_twice")),
        &(link(id("n")) + link(id("n")))
    );
    assert_eq!(
        stored_expr(&manifest, id("labels")),
        &counted(link(id("label")))
    );
}

fn case<L: Leaf, A: Aggregation>(
    when: Expr<L, A>,
    then: Expr<L, A>,
    otherwise: Expr<L, A>,
) -> Expr<L, A> {
    Expr::Case {
        whens: vec![(when, then)],
        else_: Some(Box::new(otherwise)),
    }
}

fn when_then<L: Leaf, A: Aggregation>(when: Expr<L, A>, then: Expr<L, A>) -> Expr<L, A> {
    Expr::Case {
        whens: vec![(when, then)],
        else_: None,
    }
}

fn revenue_adj<L: Leaf, A: Aggregation>(region: Expr<L, A>, revenue: Expr<L, A>) -> Expr<L, A> {
    case(
        region.eq(string("eu")),
        revenue.clone() * double(2.0),
        revenue,
    )
}

fn revenue_and_order_count(metrics: Vec<Metric>) -> SemanticModel {
    orders_model_with_metrics(
        vec![dimension("region", DataType::String)],
        vec![
            common::measure("revenue", DataType::Double, "sum"),
            common::measure("order_count", DataType::BigInt, "count"),
        ],
        metrics,
    )
}

fn formula(
    name: &str,
    data_type: DataType,
    aggregate: Option<&str>,
    expr: UnresolvedSemanticExpr,
) -> Metric {
    Metric::builder(name)
        .data_type(data_type)
        .maybe_agg(aggregate.map(agg))
        .expr(expr)
        .build()
}

#[tokio::test]
async fn a_metric_whose_agg_aggregates_a_formula_stores_the_formula_alone() {
    let manifest = compile_model(revenue_and_order_count(vec![
        formula(
            "regions_with_revenue",
            DataType::Integer,
            Some("count"),
            revenue_adj(
                dim("region").expect("a name"),
                measure("revenue").expect("a name"),
            ),
        ),
        formula(
            "regions_copy",
            DataType::Integer,
            None,
            metric("regions_with_revenue").expect("a name"),
        ),
        formula(
            "regions_distinct_copy",
            DataType::Integer,
            Some("count_distinct"),
            metric("regions_with_revenue").expect("a name"),
        ),
    ]))
    .await;
    let id = |semantic| signature(&manifest, "orders", semantic).expression_id;
    assert_eq!(
        stored_expr(&manifest, id("regions_with_revenue")),
        &revenue_adj(link(id("region")), link(id("revenue")))
    );
    assert_eq!(
        stored_expr(&manifest, id("regions_copy")),
        &link(id("regions_with_revenue"))
    );
    let aggregation_of = |semantic: &str| {
        manifest
            .semantic_definitions
            .by_name(semantic)
            .expect("the Metric is registered")
            .role_definition
            .aggregation()
    };
    assert!(
        aggregation_of("regions_with_revenue").is_some_and(|(function, distinct)| function
            .declarations
            .canonical
            .name
            == "count"
            && !distinct),
        "a Metric's own `agg:` is pinned as the function and the distinct flag"
    );
    assert_eq!(
        aggregation_of("regions_copy"),
        None,
        "a Metric with no `agg:` stores no aggregation"
    );
    let stored_content = |semantic: &str| {
        let definition = manifest
            .semantic_definitions
            .by_name(semantic)
            .expect("the Metric is registered");
        serde_json::to_value(&definition.role_definition).expect("the role body serializes")
            ["content"]
            .clone()
    };
    assert_eq!(
        stored_content("regions_with_revenue"),
        serde_json::json!({"aggregation": "count"}),
        "a Metric's `agg: count` stores only the function"
    );
    assert_eq!(
        stored_content("regions_copy"),
        serde_json::json!({"aggregation": null}),
        "a Metric with no `agg:` still stores a null aggregation"
    );
    assert_eq!(
        stored_content("regions_distinct_copy"),
        serde_json::json!({"aggregation": "count", "distinct": true}),
        "a Metric's `agg: count_distinct` stores the function `count` and the distinct flag"
    );
}

#[tokio::test]
async fn the_fold_changes_no_metrics_kind() {
    let never = || {
        case(
            lit(false),
            measure("order_count").expect("a name"),
            lit(1000_i64),
        )
    };
    let read = |name| metric(name).expect("a name");
    let manifest = compile_model(revenue_and_order_count(vec![
        formula("m", DataType::BigInt, Some("count"), never()),
        formula("m_formula", DataType::BigInt, None, never()),
        formula("m_twice", DataType::BigInt, None, read("m") + read("m")),
        formula("m_alias", DataType::BigInt, None, read("m")),
        formula(
            "m_alias_twice",
            DataType::BigInt,
            None,
            read("m_alias") + read("m_alias"),
        ),
    ]))
    .await;
    let id = |semantic| signature(&manifest, "orders", semantic).expression_id;
    let written = case(
        Expr::Literal(Literal::Boolean(false)),
        link(id("order_count")),
        Expr::Literal(Literal::BigInt(1000)),
    );
    for semantic in ["m", "m_formula"] {
        assert_eq!(stored_expr(&manifest, id(semantic)), &written, "{semantic}");
    }
    assert_eq!(stored_expr(&manifest, id("m_alias")), &link(id("m")));
    for (semantic, linked) in [("m_twice", "m"), ("m_alias_twice", "m_alias")] {
        assert_eq!(
            stored_expr(&manifest, id(semantic)),
            &(link(id(linked)) + link(id(linked))),
            "{semantic}"
        );
    }
}

fn region_label<L: Leaf, A: Aggregation>(region: Expr<L, A>) -> Expr<L, A> {
    case(region.eq(string("eu")), string("EU"), string("other"))
}

fn web_label_model(dimensions: Vec<Dimension>) -> SemanticModel {
    let region_and_channel = ["region", "channel"].map(|name| dimension(name, DataType::String));
    orders_model_with_metrics(
        region_and_channel.into_iter().chain(dimensions).collect(),
        Vec::new(),
        vec![Metric::builder("web_label")
            .data_type(DataType::String)
            .expr(region_label(dim("region").expect("a name")))
            .filter(dim("channel").expect("a name").eq(lit("web")))
            .build()],
    )
}

#[tokio::test]
async fn a_row_level_metric_with_no_agg_stores_its_filter_in_a_case() {
    let manifest = compile_model(web_label_model(Vec::new())).await;
    let web_label = signature(&manifest, "orders", "web_label");
    let id = |semantic| signature(&manifest, "orders", semantic).expression_id;
    let [channel, region] = ["channel", "region"].map(|name| link(id(name)));
    assert_eq!(
        stored_expr(&manifest, web_label.expression_id),
        &when_then(channel.eq(string("web")), region_label(region))
    );
    assert_eq!(web_label.filter_expr_id, None);
    let [channel, region] =
        ["channel", "region"].map(|name| cast(orders_column(name), DataType::String));
    assert_eq!(
        tree(&manifest, web_label),
        when_then(channel.eq(string("web")), region_label(region))
    );
}

#[tokio::test]
async fn a_case_is_stored_after_each_entry_that_its_filter_links() {
    let manifest = compile_model(orders_model_with_metrics(
        vec![
            dimension("channel", DataType::String),
            bodied(
                "z_channel",
                DataType::String,
                dim("channel").expect("a name"),
            ),
        ],
        Vec::new(),
        vec![Metric::builder("a_value")
            .data_type(DataType::Integer)
            .expr(lit(1))
            .filter(dim("z_channel").expect("a name").eq(lit("web")))
            .build()],
    ))
    .await;
    let id = |semantic| signature(&manifest, "orders", semantic).expression_id;
    assert_eq!(
        stored_expr(&manifest, id("a_value")),
        &when_then(
            link(id("z_channel")).eq(string("web")),
            Expr::Literal(Literal::Integer(1))
        )
    );
}

#[tokio::test]
async fn a_row_level_body_that_reads_a_row_level_metric_is_refused() {
    row_level_body_refusal().await.assert_committed();
}

async fn row_level_body_refusal() -> Golden {
    let labelled = bodied(
        "labelled",
        DataType::String,
        metric("web_label").expect("a name"),
    );
    let refusal = compile_diagnostic(web_label_model(vec![labelled])).await;
    let CompileError::AggregateReferenceInRowLevelBody {
        semantics,
        referenced,
    } = &refusal.kind
    else {
        panic!("expected AggregateReferenceInRowLevelBody, got {refusal:?}");
    };
    assert_eq!(
        (semantics.as_str(), referenced.as_str()),
        ("labelled", "web_label")
    );
    Golden::message(
        "a_row_level_body_that_reads_a_row_level_metric_is_refused",
        refusal.to_string(),
    )
}

fn field(name: &str) -> NamedReference {
    NamedReference::unqualified(name).expect("a name")
}

fn read_keeping(manifest: &SemanticManifest, semantic: &str, fields: &[KeptField]) -> PhysicalExpr {
    let root = signature(manifest, "orders", semantic).expression_id;
    (manifest.expressions)
        .reassemble(&root, &source_signatures(manifest, root), fields)
        .expect("the entry's tree reads whole")
}

fn revenue() -> PhysicalExpr {
    aggregate(
        "sum",
        false,
        cast(orders_column("revenue"), DataType::Double),
        None,
    )
}

#[tokio::test]
async fn a_link_to_a_metric_of_two_aggregate_steps_reads_as_its_step_2_field() {
    let manifest = compile_model(revenue_and_order_count(vec![
        formula(
            "regions_with_revenue",
            DataType::Integer,
            Some("count"),
            revenue_adj(
                dim("region").expect("a name"),
                measure("revenue").expect("a name"),
            ),
        ),
        formula(
            "revenue_per_region",
            DataType::Double,
            None,
            measure("revenue").expect("a name") / metric("regions_with_revenue").expect("a name"),
        ),
    ]))
    .await;
    let regions = signature(&manifest, "orders", "regions_with_revenue").expression_id;
    let step_2 = field("__plan_value_regions_with_revenue");
    assert_eq!(
        read_keeping(
            &manifest,
            "revenue_per_region",
            &[KeptField {
                leaf: KeptLeaf::Link(regions),
                field: step_2.clone()
            }]
        ),
        revenue() / Expr::Column(step_2)
    );
}

#[tokio::test]
async fn above_a_step_a_link_that_it_groups_by_reads_as_its_field() {
    let [region, qty] = ["region", "qty"].map(|name| dim(name).expect("a name"));
    let [revenue_link, units_link, label] =
        [measure("revenue"), measure("units"), metric("region_label")]
            .map(|link| link.expect("a name"));
    let manifest = compile_model(orders_model_with_metrics(
        vec![
            dimension("region", DataType::String),
            dimension("qty", DataType::Integer),
        ],
        vec![
            common::measure("revenue", DataType::Double, "sum"),
            Measure::builder("units")
                .data_type(DataType::BigInt)
                .agg(agg("sum"))
                .expr(qty.clone())
                .build(),
            Measure::builder("eu_sales")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .filter(region.clone().eq(lit("eu")))
                .build(),
        ],
        vec![
            formula(
                "region_label",
                DataType::String,
                None,
                region_label(region.clone()),
            ),
            formula(
                "adjusted",
                DataType::Double,
                None,
                revenue_adj(region, revenue_link.clone()),
            ),
            formula(
                "eu_revenue",
                DataType::Double,
                None,
                when_then(label.eq(lit("EU")), revenue_link.clone()),
            ),
            formula(
                "unknown_customers",
                DataType::Double,
                None,
                when_then(col("customer_id").expect("a name").is_null(), revenue_link),
            ),
            formula("big_units", DataType::BigInt, None, big(qty, units_link)),
            formula(
                "eu_share",
                DataType::Double,
                None,
                measure("eu_sales").expect("a name") / metric("adjusted").expect("a name"),
            ),
        ],
    ))
    .await;
    let id = |semantic| signature(&manifest, "orders", semantic).expression_id;
    let region = || cast(orders_column("region"), DataType::String);
    let qty = || cast(orders_column("qty"), DataType::Integer);
    let units = || aggregate("sum", false, qty(), None);
    let eu_sales = || {
        let sales = cast(orders_column("eu_sales"), DataType::Double);
        aggregate("sum", false, sales, Some(is("region", "eu")))
    };
    let customer_id = NamedReference::qualified("ps0", "customer_id").expect("a column");
    let cases: [(_, _, _, &dyn Fn(PhysicalExpr) -> PhysicalExpr, _); 5] = [
        (
            "adjusted",
            KeptLeaf::Link(id("region")),
            "region",
            &|region| revenue_adj(region, revenue()),
            region(),
        ),
        (
            "eu_revenue",
            KeptLeaf::Link(id("region_label")),
            "region_label",
            &|label| when_then(label.eq(string("EU")), revenue()),
            region_label(region()),
        ),
        (
            "unknown_customers",
            KeptLeaf::Column(customer_id.clone()),
            "__plan_auto_customer_id",
            &|customer| when_then(customer.is_null(), revenue()),
            Expr::Column(customer_id),
        ),
        (
            "big_units",
            KeptLeaf::Link(id("qty")),
            "qty",
            &|qty| big(qty, units()),
            qty(),
        ),
        (
            "eu_share",
            KeptLeaf::Link(id("region")),
            "region",
            &|region| eu_sales() / revenue_adj(region, revenue()),
            region(),
        ),
    ];
    for (semantic, grouped, name, tree, below) in cases {
        assert_eq!(
            read_keeping(&manifest, semantic, &[]),
            tree(below),
            "{semantic} below the step"
        );
        assert_eq!(
            read_keeping(
                &manifest,
                semantic,
                &[KeptField {
                    leaf: grouped,
                    field: field(name)
                }]
            ),
            tree(Expr::Column(field(name))),
            "{semantic} above the step"
        );
    }
}

fn big<L: Leaf, A: Aggregation>(qty: Expr<L, A>, units: Expr<L, A>) -> Expr<L, A> {
    let integer = |value| Expr::Literal(Literal::Integer(value));
    case(qty.gt(integer(1)), units, Expr::Literal(Literal::BigInt(0)))
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async { vec![row_level_body_refusal().await] }).await;
}

fn double<L: Leaf, A: Aggregation>(value: f64) -> Expr<L, A> {
    Expr::Literal(Literal::Double(value))
}
