use semstrait_sql::datatype::DataType;

use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{CompileError, ExprLayer, PhysicalExprId, SemanticManifest};
use semstrait_model::expr::{col, dim, field, lit, measure, metric};
use semstrait_model::{
    DataKindFilter, Dataset, Dimension, DimensionSite, Measure, MeasureSite, Metric, MetricSite,
    NestedDataset, SemanticMapping, SemanticModel, Unionset, UnresolvedSemanticExpr,
};
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, Literal, UnaryOpKind};

mod common;
use common::golden::{self, Golden};
use common::*;

fn string_literal(value: &str) -> PhysicalExpr {
    Expr::Literal(Literal::String(value.to_owned()))
}

fn eq(left: PhysicalExpr, right: PhysicalExpr) -> PhysicalExpr {
    Expr::BinaryOp {
        op: BinaryOpKind::Eq,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn status_is_paid() -> UnresolvedSemanticExpr {
    ExprOperationExt::eq(dim("status").expect("a name"), lit("paid"))
}

fn named_filter(name: &str, expr: UnresolvedSemanticExpr) -> DataKindFilter {
    DataKindFilter::builder(name).expr(expr).build()
}

fn two_datasets_filtering_on_status() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_dimension(
        &mut model,
        Dimension::builder("is_test")
            .data_type(DataType::Boolean)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(status_is_paid())
            .build(),
    );
    for (dataset, amount, status, is_test) in [
        ("orders", "amount", "order_state", "test_flag"),
        ("invoices", "total", "invoice_state", "is_test"),
    ] {
        insert_dataset(
            &mut model,
            Dataset::builder(dataset)
                .table(format!("warehouse.{dataset}"))
                .dimension(DimensionSite::r#ref("status"))
                .dimension(DimensionSite::r#ref("is_test"))
                .measure(MeasureSite::r#ref("revenue"))
                .semantic_mapping(
                    SemanticMapping::builder()
                        .column("revenue", amount)
                        .column("status", status)
                        .column("is_test", is_test)
                        .build(),
                )
                .build(),
        );
    }
    model
}

#[tokio::test]
async fn a_pooled_filter_lowers_per_source_over_that_sources_columns() {
    let manifest = compile_model(two_datasets_filtering_on_status()).await;

    for (source, amount, status) in [
        ("orders", "amount", "order_state"),
        ("invoices", "total", "invoice_state"),
    ] {
        let revenue = signature(&manifest, source, "revenue");
        let alias = source_alias(&manifest, source);
        assert_eq!(
            filter_tree(&manifest, revenue),
            eq(
                cast(column(&alias, status), DataType::String),
                string_literal("paid")
            ),
            "`{source}` filters on its own `status` column"
        );
        assert_eq!(
            tree(&manifest, revenue),
            integer_sum(&alias, amount),
            "the expression tree is the aggregate, and its own `filter` slot stays empty"
        );
    }
}

#[tokio::test]
async fn a_ref_site_filter_replaces_the_pooled_one_for_that_source_only() {
    let mut model = two_datasets_filtering_on_status();
    let orders = model
        .datasets
        .values_mut()
        .find(|dataset| dataset.body.base.name == "orders")
        .expect("the fixture holds `orders`");
    let site = orders
        .semantic_interface
        .measures
        .values_mut()
        .next()
        .expect("`orders` exposes one measure");
    match site {
        semstrait_model::SemanticSite::Reference(reference) => {
            reference.filter = Some(!dim("is_test").expect("a name"));
        }
        other => panic!("the fixture exposes `revenue` at a ref site, got {other:?}"),
    }

    let manifest = compile_model(model).await;

    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "revenue")),
        Expr::UnaryOp {
            op: UnaryOpKind::Not,
            operand: Box::new(cast(
                column(&source_alias(&manifest, "orders"), "test_flag"),
                DataType::Boolean,
            )),
        },
        "the site's predicate is the one that lowers for `orders`"
    );
    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "invoices", "revenue")),
        eq(
            cast(
                column(&source_alias(&manifest, "invoices"), "invoice_state"),
                DataType::String
            ),
            string_literal("paid")
        ),
        "`invoices` still sees the pooled predicate"
    );
}

#[tokio::test]
async fn a_filter_entry_is_not_the_expression_entry_it_belongs_to() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("amount")
            .data_type(DataType::Integer)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(ExprOperationExt::gt(
                dim("amount").expect("a name"),
                lit(0i64),
            ))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("amount"))
            .measure(MeasureSite::r#ref("revenue"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("revenue", "amount")
                    .column("amount", "amount")
                    .build(),
            )
            .build(),
    );

    let manifest = compile_model(model).await;
    let revenue = signature(&manifest, "orders", "revenue");
    let filter = revenue
        .filter_expr_id
        .expect("`revenue` authors a predicate");

    assert_ne!(
        filter, revenue.expression_id,
        "the filter entry is not the expression's entry"
    );

    let sources_of = |root: &PhysicalExprId| {
        manifest
            .expressions
            .get(root)
            .expect("the root is stored")
            .physical_sources
            .clone()
    };
    assert_eq!(sources_of(&filter), sources_of(&revenue.expression_id));
    assert!(!sources_of(&filter).is_empty());
}

#[tokio::test]
async fn a_filter_referencing_a_measure_is_refused() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("net")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(ExprOperationExt::gt(
                measure("revenue").expect("a name"),
                lit(0i64),
            ))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .measure(MeasureSite::r#ref("net"))
            .semantic_mapping(
                SemanticMapping::builder()
                    .column("revenue", "amount")
                    .column("net", "net_amount")
                    .build(),
            )
            .build(),
    );

    match compile_error(model).await {
        CompileError::AggregateReferenceInRowLevelBody {
            semantics,
            referenced,
        } => {
            assert_eq!(semantics, "net");
            assert_eq!(referenced, "revenue");
        }
        other => panic!("expected AggregateReferenceInRowLevelBody, got {other:?}"),
    }
}

#[tokio::test]
async fn a_measures_aggregation_filter_still_requires_boolean() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(lit(1))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .build(),
    );

    match compile_error(model).await {
        CompileError::TypeInferenceFailure { node, reason } => {
            assert!(
                node.contains("revenue"),
                "node should name the filter, got `{node}`"
            );
            assert!(
                reason.contains("Boolean"),
                "reason should state the Boolean requirement, got `{reason}`"
            );
        }
        other => panic!("expected TypeInferenceFailure, got {other:?}"),
    }
}

#[tokio::test]
async fn an_untyped_null_filter_is_a_boolean_null() {
    let null = || untyped(Literal::Null(DataType::String));
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(null())
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .filter(DataKindFilter::builder("nothing").expr(null()).build())
            .build(),
    );

    let manifest = compile_model(model).await;
    let boolean_null = Expr::Literal(Literal::Null(DataType::Boolean));
    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "revenue")),
        boolean_null,
        "a Measure's `filter:`"
    );
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "nothing")),
        boolean_null,
        "a named Filter's body"
    );
}

#[tokio::test]
async fn a_filter_referencing_a_scalar_bodied_metric_is_refused() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("is_paid")
            .data_type(DataType::Boolean)
            .expr(status_is_paid())
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("net")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(metric("is_paid").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .measure(MeasureSite::r#ref("net"))
            .metric(MetricSite::r#ref("is_paid"))
            .build(),
    );

    match compile_error(model).await {
        CompileError::AggregateReferenceInRowLevelBody {
            semantics,
            referenced,
        } => {
            assert_eq!(semantics, "net");
            assert_eq!(referenced, "is_paid");
        }
        other => panic!("expected AggregateReferenceInRowLevelBody, got {other:?}"),
    }
}

#[tokio::test]
async fn a_named_filter_with_a_non_boolean_body_is_refused_at_compile() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .measure(MeasureSite::r#ref("revenue"))
            .filter(
                DataKindFilter::builder("not_a_predicate")
                    .expr(lit(1))
                    .build(),
            )
            .build(),
    );

    match compile_error(model).await {
        CompileError::TypeInferenceFailure { node, reason } => {
            assert!(
                node.contains("not_a_predicate"),
                "node should name the refused Filter, got `{node}`"
            );
            assert!(
                reason.contains("Boolean"),
                "reason should state the Boolean requirement, got `{reason}`"
            );
        }
        other => panic!("expected TypeInferenceFailure, got {other:?}"),
    }
}

#[tokio::test]
async fn a_named_filter_over_an_unobserved_column_is_refused_and_its_not_compiles() {
    let model = |body: UnresolvedSemanticExpr| {
        let mut model = SemanticModel::default();
        model.name = "sales".to_owned();
        insert_measure(
            &mut model,
            Measure::builder("revenue")
                .data_type(DataType::Integer)
                .agg(agg("sum"))
                .build(),
        );
        insert_dataset(
            &mut model,
            Dataset::builder("orders")
                .table("warehouse.orders")
                .measure(MeasureSite::r#ref("revenue"))
                .filter(DataKindFilter::builder("flagged").expr(body).build())
                .build(),
        );
        model
    };
    match compile_error(model(col("flag").expect("a name"))).await {
        CompileError::TypeInferenceFailure { node, .. } => assert_eq!(node, "flagged"),
        other => panic!("expected TypeInferenceFailure, got {other:?}"),
    }
    let manifest = compile_model(model(!col("flag").expect("a name"))).await;
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "flagged")),
        !orders_column("flag")
    );
}

#[tokio::test]
async fn a_named_filter_with_a_boolean_body_compiles_and_owns_a_signature() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .measure(MeasureSite::r#ref("revenue"))
            .filter(
                DataKindFilter::builder("paid_only")
                    .expr(status_is_paid())
                    .build(),
            )
            .build(),
    );

    let manifest = compile_model(model).await;

    let signature = signature(&manifest, "orders", "paid_only");
    assert_eq!(
        tree(&manifest, signature),
        eq(
            cast(column("ps0", "status"), DataType::String),
            string_literal("paid")
        ),
        "a named Filter's signature roots its own Boolean predicate"
    );
}

#[tokio::test]
async fn a_metric_carries_a_filter_and_a_dimension_or_named_filter_never_does() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("doubled")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
            .filter(status_is_paid())
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("doubled"))
            .filter(
                DataKindFilter::builder("paid_only")
                    .expr(status_is_paid())
                    .build(),
            )
            .build(),
    );

    let manifest = compile_model(model).await;

    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "doubled")),
        eq(
            cast(column("ps0", "status"), DataType::String),
            string_literal("paid")
        ),
        "a Metric's predicate lowers row-level over the dimension it reads"
    );
    for semantic in ["status", "revenue", "paid_only"] {
        assert_eq!(
            signature(&manifest, "orders", semantic).filter_expr_id,
            None,
            "`{semantic}` authors no `filter:`, so its signature carries no filter root"
        );
    }
}

#[tokio::test]
async fn a_metric_inlining_a_filtered_measure_carries_that_measures_predicate_on_the_inlined_aggregate(
) {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(status_is_paid())
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("doubled")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("doubled"))
            .build(),
    );

    let manifest = compile_model(model).await;

    let predicate = eq(
        cast(column("ps0", "status"), DataType::String),
        string_literal("paid"),
    );
    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "revenue")),
        predicate.clone(),
        "the Measure's own signature carries the predicate, standalone and disjoint"
    );

    let metric = signature(&manifest, "orders", "doubled");
    assert_eq!(
        metric.filter_expr_id, None,
        "the Metric's signature carries no predicate root of its own"
    );

    let filtered_fold = integer_sum_filtered("ps0", "revenue", Some(predicate));
    assert_eq!(
        tree(&manifest, metric),
        Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(filtered_fold.clone()),
            right: Box::new(filtered_fold),
        },
        "both folds read through the links carry revenue's own predicate on their own FILTER"
    );
}

#[tokio::test]
async fn a_metric_inlining_a_cast_wrapped_filtered_measure_carries_the_predicate_inside_the_cast() {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .expr(col("amount").expect("a name"))
            .filter(status_is_paid())
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("doubled")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name") + measure("revenue").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("doubled"))
            .build(),
    );

    let manifest = compile_model(model).await;

    let predicate = eq(
        cast(column("ps0", "status"), DataType::String),
        string_literal("paid"),
    );
    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "revenue")),
        predicate.clone(),
        "the Measure's own signature carries the predicate, standalone and disjoint"
    );
    assert_eq!(
        tree(&manifest, signature(&manifest, "orders", "revenue")),
        cast(
            aggregate("sum", false, column("ps0", "amount"), None),
            DataType::Integer
        ),
        "the Measure's own realisation keeps its boundary Cast with no aggregate filter"
    );

    let metric = signature(&manifest, "orders", "doubled");
    assert_eq!(
        metric.filter_expr_id, None,
        "the Metric's signature carries no predicate root of its own"
    );

    let filtered_cast_fold = cast(
        aggregate("sum", false, column("ps0", "amount"), Some(predicate)),
        DataType::Integer,
    );
    assert_eq!(
        tree(&manifest, metric),
        Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(filtered_cast_fold.clone()),
            right: Box::new(filtered_cast_fold),
        },
        "both folds read through the links keep their boundary Cast and carry revenue's \
         predicate inside it"
    );
}

#[tokio::test]
async fn a_metric_composing_differently_filtered_measures_keeps_each_predicate_at_its_own_aggregate(
) {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("paid")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(status_is_paid())
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("pending")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(ExprOperationExt::eq(
                dim("status").expect("a name"),
                lit("pending"),
            ))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("total")
            .data_type(DataType::Integer)
            .expr(measure("paid").expect("a name") + measure("pending").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .measure(MeasureSite::r#ref("paid"))
            .measure(MeasureSite::r#ref("pending"))
            .metric(MetricSite::r#ref("total"))
            .build(),
    );

    let manifest = compile_model(model).await;

    let paid_predicate = eq(
        cast(column("ps0", "status"), DataType::String),
        string_literal("paid"),
    );
    let pending_predicate = eq(
        cast(column("ps0", "status"), DataType::String),
        string_literal("pending"),
    );
    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "paid")),
        paid_predicate.clone(),
        "`paid`'s own signature carries only `paid`'s predicate"
    );
    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "pending")),
        pending_predicate.clone(),
        "`pending`'s own signature carries only `pending`'s predicate"
    );

    let total = signature(&manifest, "orders", "total");
    assert_eq!(
        total.filter_expr_id, None,
        "the Metric's signature carries no combined predicate of its own"
    );

    assert_eq!(
        tree(&manifest, total),
        Expr::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(integer_sum_filtered("ps0", "paid", Some(paid_predicate))),
            right: Box::new(integer_sum_filtered(
                "ps0",
                "pending",
                Some(pending_predicate)
            )),
        },
        "each fold keeps its own Measure's predicate, never the other's and never a combined one"
    );
}

#[tokio::test]
async fn a_metric_inlining_a_filtered_metric_conjoins_that_metrics_predicate_onto_every_inlined_aggregate(
) {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("status")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_dimension(
        &mut model,
        Dimension::builder("is_test")
            .data_type(DataType::Boolean)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .filter(status_is_paid())
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("a")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name"))
            .filter(dim("is_test").expect("a name"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("b")
            .data_type(DataType::Integer)
            .expr(metric("a").expect("a name"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("status"))
            .dimension(DimensionSite::r#ref("is_test"))
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("a"))
            .metric(MetricSite::r#ref("b"))
            .build(),
    );

    let manifest = compile_model(model).await;

    let revenue_predicate = eq(
        cast(column("ps0", "status"), DataType::String),
        string_literal("paid"),
    );
    let a_predicate = cast(column("ps0", "is_test"), DataType::Boolean);

    assert_eq!(
        filter_tree(&manifest, signature(&manifest, "orders", "revenue")),
        revenue_predicate.clone(),
        "`revenue`'s own signature carries only its own predicate"
    );

    let a = signature(&manifest, "orders", "a");
    assert_eq!(
        filter_tree(&manifest, a),
        a_predicate.clone(),
        "`a`'s own signature carries only its own predicate"
    );
    assert_eq!(
        tree(&manifest, a),
        integer_sum_filtered("ps0", "revenue", Some(revenue_predicate.clone())),
        "the tree read through `a`'s link to `revenue` carries `revenue`'s own predicate"
    );

    let b = signature(&manifest, "orders", "b");
    assert_eq!(
        b.filter_expr_id, None,
        "`b`'s signature carries no predicate root of its own"
    );
    assert_eq!(
        tree(&manifest, b),
        integer_sum_filtered(
            "ps0",
            "revenue",
            Some(Expr::BinaryOp {
                op: BinaryOpKind::And,
                left: Box::new(revenue_predicate),
                right: Box::new(a_predicate),
            })
        ),
        "`b`'s read aggregate carries `revenue`'s predicate conjoined with `a`'s own, \
         `revenue`'s on the left because reading joins it first"
    );
}

#[tokio::test]
async fn a_metric_composing_a_metric_whose_own_measure_is_unfiltered_carries_only_the_metrics_own_predicate(
) {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    insert_dimension(
        &mut model,
        Dimension::builder("is_test")
            .data_type(DataType::Boolean)
            .categorical()
            .build(),
    );
    insert_dimension(
        &mut model,
        Dimension::builder("region")
            .data_type(DataType::String)
            .categorical()
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::Integer)
            .agg(agg("sum"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("a")
            .data_type(DataType::Integer)
            .expr(measure("revenue").expect("a name"))
            .filter(dim("is_test").expect("a name"))
            .build(),
    );
    insert_metric(
        &mut model,
        Metric::builder("b")
            .data_type(DataType::Integer)
            .expr(metric("a").expect("a name"))
            .filter(ExprOperationExt::eq(
                dim("region").expect("a name"),
                lit("us"),
            ))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimension(DimensionSite::r#ref("is_test"))
            .dimension(DimensionSite::r#ref("region"))
            .measure(MeasureSite::r#ref("revenue"))
            .metric(MetricSite::r#ref("a"))
            .metric(MetricSite::r#ref("b"))
            .build(),
    );

    let manifest = compile_model(model).await;

    let a_predicate = cast(column("ps0", "is_test"), DataType::Boolean);
    let b_predicate = eq(
        cast(column("ps0", "region"), DataType::String),
        string_literal("us"),
    );

    let a = signature(&manifest, "orders", "a");
    assert_eq!(
        filter_tree(&manifest, a),
        a_predicate.clone(),
        "`a`'s own signature carries its own authored predicate"
    );
    assert_eq!(
        tree(&manifest, a),
        integer_sum("ps0", "revenue"),
        "`a`'s own tree stays unfiltered — `revenue` carries none and `a`'s own filter \
         never joins its own tree"
    );

    let b = signature(&manifest, "orders", "b");
    assert_eq!(
        filter_tree(&manifest, b),
        b_predicate,
        "`b`'s own signature carries only `b`'s own authored predicate"
    );
    assert_eq!(
        tree(&manifest, b),
        integer_sum_filtered("ps0", "revenue", Some(a_predicate)),
        "`b`'s read aggregate carries only `a`'s predicate, with nothing to conjoin"
    );
}

#[tokio::test]
async fn a_metric_over_a_filtered_measure_stores_links_and_reads_its_filter() {
    let net = Metric::builder("net")
        .data_type(DataType::Integer)
        .expr(measure("revenue").expect("a name") - measure("refunds").expect("a name"))
        .build();
    let manifest = compile_model(revenue_and_refunds(vec![net])).await;
    let [revenue, refunds] =
        ["revenue", "refunds"].map(|name| signature(&manifest, "orders", name).expression_id);
    let net = signature(&manifest, "orders", "net");
    assert_eq!(
        stored_expr(&manifest, net.expression_id),
        &(link(revenue) - link(refunds))
    );
    assert_eq!(
        tree(&manifest, net),
        integer_sum_filtered("ps0", "revenue", Some(is("status", "paid")))
            - integer_sum("ps0", "refunds")
    );
}

fn amount_in_eu() -> Measure {
    Measure::builder("amount")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .filter(ExprOperationExt::eq(
            dim("region").expect("a name"),
            lit("eu"),
        ))
        .build()
}

#[tokio::test]
async fn a_measures_filter_links_the_dimension_it_reads() {
    let manifest = compile_model(orders_model(
        vec![dimension("region", DataType::String)],
        vec![amount_in_eu()],
        |orders| orders,
    ))
    .await;
    let region = signature(&manifest, "orders", "region").expression_id;
    let filter = (signature(&manifest, "orders", "amount").filter_expr_id)
        .expect("`amount`'s filter is its own entry");
    assert_eq!(
        stored_expr(&manifest, filter),
        &ExprOperationExt::eq(link(region), string("eu"))
    );
    assert_eq!(
        manifest.expressions.physical[&filter].layer,
        ExprLayer::Scalar
    );
}

#[tokio::test]
async fn a_measures_filter_reading_a_named_filter_is_refused() {
    named_filter_reference_on_a_measure()
        .await
        .assert_committed();
}

async fn named_filter_refusal(
    case: &str,
    model: SemanticModel,
    semantics: &str,
    filter: &str,
) -> Golden {
    let diagnostic = compile_diagnostic(model).await;
    match &diagnostic.kind {
        CompileError::NamedFilterReference {
            semantics: got_semantics,
            filter: got_filter,
        } => {
            assert_eq!(got_semantics, semantics);
            assert_eq!(got_filter, filter);
        }
        other => panic!("expected NamedFilterReference, got {other:?}"),
    }
    Golden::message(case, diagnostic.to_string())
}

async fn named_filter_reference_on_a_measure() -> Golden {
    let revenue = Measure::builder("revenue")
        .data_type(DataType::Integer)
        .agg(agg("sum"))
        .filter(field("paid_only").expect("a name"))
        .build();
    let model = orders_model(
        vec![dimension("status", DataType::String)],
        vec![revenue],
        |orders| orders.filter(named_filter("paid_only", status_is_paid())),
    );
    named_filter_refusal(
        "a_measures_filter_reading_a_named_filter_is_refused",
        model,
        "revenue",
        "paid_only",
    )
    .await
}

#[tokio::test]
async fn a_metrics_filter_reading_a_named_filter_is_refused() {
    named_filter_reference_on_a_metric()
        .await
        .assert_committed();
}

async fn named_filter_reference_on_a_metric() -> Golden {
    let mut model = orders_model(
        vec![
            dimension("status", DataType::String),
            dimension("region", DataType::String),
        ],
        vec![],
        |orders| {
            orders
                .metric(MetricSite::r#ref("doubled"))
                .filter(named_filter("paid_only", status_is_paid()))
                .filter(named_filter(
                    "paid_eu",
                    field("paid_only").expect("a name")
                        & ExprOperationExt::eq(dim("region").expect("a name"), lit("eu")),
                ))
        },
    );
    insert_metric(
        &mut model,
        Metric::builder("doubled")
            .data_type(DataType::Integer)
            .expr(lit(2) + lit(2))
            .filter(field("paid_eu").expect("a name"))
            .build(),
    );
    named_filter_refusal(
        "a_metrics_filter_reading_a_named_filter_is_refused",
        model,
        "doubled",
        "paid_eu",
    )
    .await
}

#[tokio::test]
async fn a_dimensions_expr_reading_a_named_filter_is_refused() {
    named_filter_reference_on_a_dimensions_expr()
        .await
        .assert_committed();
}

async fn named_filter_reference_on_a_dimensions_expr() -> Golden {
    let mut model = orders_model(
        vec![dimension("status", DataType::String)],
        vec![],
        |orders| {
            orders
                .dimension(DimensionSite::r#ref("is_paid"))
                .filter(named_filter("paid_only", status_is_paid()))
        },
    );
    insert_dimension(
        &mut model,
        bodied(
            "is_paid",
            DataType::Boolean,
            field("paid_only").expect("a name"),
        ),
    );
    named_filter_refusal(
        "a_dimensions_expr_reading_a_named_filter_is_refused",
        model,
        "is_paid",
        "paid_only",
    )
    .await
}

#[tokio::test]
async fn a_named_filter_reading_a_named_filter_and_a_dimension_compiles() {
    let model = orders_model(
        vec![
            dimension("status", DataType::String),
            dimension("region", DataType::String),
        ],
        vec![],
        |orders| {
            orders
                .filter(named_filter("paid_only", status_is_paid()))
                .filter(named_filter(
                    "paid_eu",
                    field("paid_only").expect("a name")
                        & ExprOperationExt::eq(dim("region").expect("a name"), lit("eu")),
                ))
        },
    );
    let manifest = compile_model(model).await;
    let [paid_only, region] =
        ["paid_only", "region"].map(|name| signature(&manifest, "orders", name).expression_id);
    let paid_eu = signature(&manifest, "orders", "paid_eu").expression_id;
    assert_eq!(
        stored_expr(&manifest, paid_eu),
        &(link(paid_only) & ExprOperationExt::eq(link(region), string("eu")))
    );
}

#[tokio::test]
async fn a_measure_named_before_the_dimension_its_filter_reads_compiles() {
    compile_model(orders_model(
        vec![dimension("region", DataType::String)],
        vec![amount_in_eu()],
        |orders| orders,
    ))
    .await;
    let branch = |name: &str| {
        NestedDataset::builder(name)
            .table(format!("warehouse.{name}"))
            .semantic_mapping(identity_mapping(&["region", "amount"]))
            .build()
    };
    let stacked = Unionset::builder("stacked")
        .dataset(branch("east"))
        .dataset(branch("west"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("amount"))
        .union_all()
        .build();
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(dimension("region", DataType::String))
        .measure(amount_in_eu())
        .unionset(stacked)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compile_model(model).await;
    let region = signature(&manifest, "stacked", "region").expression_id;
    let filter = (signature(&manifest, "stacked", "amount").filter_expr_id)
        .expect("the Unionset's own filter is its own entry");
    assert_eq!(
        stored_expr(&manifest, filter),
        &ExprOperationExt::eq(link(region), string("eu"))
    );
}

#[tokio::test]
async fn compiling_twice_yields_the_same_filter_roots() {
    let first = compile_model(two_datasets_filtering_on_status()).await;
    let second = compile_model(two_datasets_filtering_on_status()).await;

    let roots = |manifest: &SemanticManifest| -> Vec<Option<PhysicalExprId>> {
        manifest
            .semantic_signatures
            .values()
            .map(|signature| signature.filter_expr_id)
            .collect()
    };
    assert_eq!(roots(&first), roots(&second));
    assert!(
        roots(&first).iter().filter(|root| root.is_some()).count() == 2,
        "both signatures for `revenue` carry a filter root: {:?}",
        roots(&first)
    );
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            named_filter_reference_on_a_measure().await,
            named_filter_reference_on_a_metric().await,
            named_filter_reference_on_a_dimensions_expr().await,
        ]
    })
    .await;
}
