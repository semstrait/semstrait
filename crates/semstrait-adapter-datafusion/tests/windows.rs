mod common;

use std::error::Error;

use common::golden::{self, Golden};
use common::runner::{self, gate};
use common::{aggregate, leaf, literal, name, plan, project, relations, scanned};
use semstrait_ir::{
    AggregateNode, IrErrorKind, Parameter, PhysicalExpr, PlanNode, ProjectNode, SemanticPlan,
    TableScan,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, Expr, ExprOperationExt, FunctionRef, NullOrdering, Over,
    SortDir, SortKey, WindowCall,
};
use semstrait_sql::functions::registry;
use substrait::proto::expression::RexType;
use substrait::proto::rel::RelType;
use substrait::proto::{plan_rel, Expression, Plan};

const LISTED: &[&str] = &[
    "marker/latest_by_month",
    "marker/latest_by_account",
    "condition/rank_ignores_condition",
    "nulls/nulls_last_tie",
    "fan_out/marker_per_order",
    "fan_out/no_marker_control",
];

#[test]
fn the_listing_holds_each_case_and_no_other() -> Result<(), Box<dyn Error>> {
    runner::listing(LISTED)?;
    Ok(())
}

fn window(
    function: &str,
    args: Vec<PhysicalExpr>,
    over: Over<Parameter, AggregateAllowed>,
) -> PhysicalExpr {
    let Some(FunctionRef::Window(function)) = registry().get(function) else {
        panic!("the catalog declares the window function `{function}`");
    };
    let call = WindowCall::builder(function, args).over(over).build();
    Expr::Function(Call::Window(call.expect("a window call")))
}

fn rows(table: &str, columns: &[(&str, DataType)]) -> Result<PlanNode, IrErrorKind> {
    let reader = TableScan::named_table([String::new(), "sales".into(), table.into()], None);
    scanned(reader, columns, Some("ssd0"))
}

fn marked(
    table: &str,
    columns: &[(&str, DataType)],
    marker: &str,
    call: PhysicalExpr,
) -> Result<PlanNode, IrErrorKind> {
    let marker = (name(&format!("ssd0.{marker}")), call);
    Ok(ProjectNode::new(rows(table, columns)?, vec![marker])?.into())
}

fn first(marker: &str) -> PhysicalExpr {
    leaf(&["ssd0", marker]).eq(literal(1_i64))
}

fn aggregated(
    input: PlanNode,
    grouping: &str,
    aggregates: Vec<(&str, AggregateCall<Parameter, AggregateAllowed>)>,
) -> Result<SemanticPlan, IrErrorKind> {
    let fields: Vec<_> = std::iter::once(grouping)
        .chain(aggregates.iter().map(|(field, _)| *field))
        .map(|field| (name(field), leaf(&["ssd0", field])))
        .collect();
    let aggregates = (aggregates.into_iter())
        .map(|(field, call)| (name(&format!("ssd0.{field}")), call))
        .collect();
    let groupings = vec![(name(&format!("ssd0.{grouping}")), leaf(&["ssd0", grouping]))];
    let aggregated = AggregateNode::new(input, groupings, aggregates)?;
    Ok(plan(project(aggregated, fields)?))
}

fn assert_planned_shape(case: &str, plan: &Plan, markers: usize) {
    use RelType::{Aggregate, Project, Read};
    let shape: Vec<_> = (relations(plan).into_iter())
        .flat_map(|relation| &relation.rel_type)
        .collect();
    let [Project(_), Aggregate(_), marked @ .., Project(_), Read(_)] = &shape[..] else {
        panic!("{case}: scan, row-level projection, marker projection, aggregate, root projection");
    };
    assert_eq!(marked.len(), markers, "{case}: the marker projections");
    for relation in marked {
        let Project(marker) = relation else {
            panic!("{case}: a marker projection");
        };
        assert!(
            matches!(
                marker.expressions[..],
                [Expression {
                    rex_type: Some(RexType::WindowFunction(_))
                }]
            ),
            "{case}: the marker projection holds one window call"
        );
    }
    let names = (plan.relations.iter()).flat_map(|relation| match &relation.rel_type {
        Some(plan_rel::RelType::Root(root)) => root.names.as_slice(),
        _ => &[],
    });
    for name in names {
        assert!(!name.contains("__plan_"), "{case}: the root name {name}");
    }
}

fn latest(partition: &[&str], by: &str) -> PhysicalExpr {
    let over = Over {
        partition_by: (partition.iter())
            .map(|column| leaf(&["ssd0", column]))
            .collect(),
        order_by: vec![SortKey {
            expr: leaf(&["ssd0", by]),
            dir: SortDir::Desc {
                nulls: NullOrdering::Last,
            },
        }],
        window_frame: None,
    };
    window("rank", Vec::new(), over)
}

fn daily(partition: &[&str]) -> Result<PlanNode, IrErrorKind> {
    let columns = [
        ("account", DataType::String),
        ("date", DataType::Date),
        ("month", DataType::String),
        ("balance", DataType::BigInt),
        ("cost", DataType::BigInt),
    ];
    let rank = latest(partition, "date");
    marked("daily", &columns, "__plan_rank_balance", rank)
}

fn latest_balance_and_cost(partition: &[&str]) -> Result<SemanticPlan, IrErrorKind> {
    let balance = vec![leaf(&["ssd0", "balance"])];
    let aggregates = vec![
        (
            "balance",
            aggregate("sum", balance, Some(first("__plan_rank_balance"))),
        ),
        (
            "cost",
            aggregate("sum", vec![leaf(&["ssd0", "cost"])], None),
        ),
    ];
    aggregated(daily(partition)?, "month", aggregates)
}

async fn latest_by_month() -> Result<Golden, Box<dyn Error>> {
    let (plan, golden) = gate(
        "marker/latest_by_month",
        &latest_balance_and_cost(&["month"])?,
    )
    .await?;

    assert_planned_shape("marker/latest_by_month", &plan, 1);
    Ok(golden)
}

#[tokio::test]
async fn the_latest_balance_and_the_total_cost_read_different_rows() -> Result<(), Box<dyn Error>> {
    latest_by_month().await?.assert_committed();
    Ok(())
}

async fn latest_by_account() -> Result<Golden, Box<dyn Error>> {
    let semantic = latest_balance_and_cost(&["month", "account"])?;

    let (plan, golden) = gate("marker/latest_by_account", &semantic).await?;

    assert_planned_shape("marker/latest_by_account", &plan, 1);
    Ok(golden)
}

#[tokio::test]
async fn the_axes_partition_the_rank() -> Result<(), Box<dyn Error>> {
    latest_by_account().await?.assert_committed();
    Ok(())
}

async fn rank_ignores_condition() -> Result<Golden, Box<dyn Error>> {
    let condition = leaf(&["ssd0", "account"]).eq(literal("B")) & first("__plan_rank_balance");
    let balance = aggregate("sum", vec![leaf(&["ssd0", "balance"])], Some(condition));
    let semantic = aggregated(daily(&["month"])?, "month", vec![("balance", balance)])?;

    let (plan, golden) = gate("condition/rank_ignores_condition", &semantic).await?;

    assert_planned_shape("condition/rank_ignores_condition", &plan, 1);
    Ok(golden)
}

#[tokio::test]
async fn the_rank_reads_every_row_of_its_partition_whatever_the_condition(
) -> Result<(), Box<dyn Error>> {
    rank_ignores_condition().await?.assert_committed();
    Ok(())
}

async fn nulls_last_tie() -> Result<Golden, Box<dyn Error>> {
    let columns = [
        ("g", DataType::String),
        ("d", DataType::Date),
        ("v", DataType::BigInt),
    ];
    let marker = "__plan_rank_v";
    let input = marked("f4", &columns, marker, latest(&["g"], "d"))?;
    let v = aggregate("sum", vec![leaf(&["ssd0", "v"])], Some(first(marker)));
    let semantic = aggregated(input, "g", vec![("v", v)])?;

    let (plan, golden) = gate("nulls/nulls_last_tie", &semantic).await?;

    assert_planned_shape("nulls/nulls_last_tie", &plan, 1);
    Ok(golden)
}

#[tokio::test]
async fn nulls_last_keeps_the_latest_non_null_by_or_every_row_of_a_null_by(
) -> Result<(), Box<dyn Error>> {
    nulls_last_tie().await?.assert_committed();
    Ok(())
}

async fn marker_per_order_and_no_marker_control() -> Result<(Golden, Golden), Box<dyn Error>> {
    let columns = [
        ("order_id", DataType::BigInt),
        ("region", DataType::String),
        ("shipping_cost", DataType::BigInt),
        ("line_id", DataType::BigInt),
        ("price", DataType::BigInt),
    ];
    let averages = |once| {
        vec![
            (
                "shipping_cost",
                aggregate("avg", vec![leaf(&["ssd0", "shipping_cost"])], once),
            ),
            (
                "price",
                aggregate("avg", vec![leaf(&["ssd0", "price"])], None),
            ),
        ]
    };
    let marker = "__plan_row_number_shipping_cost";
    let over = Over {
        partition_by: ["region", "order_id", "shipping_cost"]
            .map(|column| leaf(&["ssd0", column]))
            .to_vec(),
        ..Over::default()
    };
    let row_number = window("row_number", Vec::new(), over);
    let input = marked("order_rows", &columns, marker, row_number)?;
    let semantic = aggregated(input, "region", averages(Some(first(marker))))?;
    let control = aggregated(rows("order_rows", &columns)?, "region", averages(None))?;

    let (plan, golden) = gate("fan_out/marker_per_order", &semantic).await?;
    assert_planned_shape("fan_out/marker_per_order", &plan, 1);

    let (plan, control_golden) = gate("fan_out/no_marker_control", &control).await?;
    assert_planned_shape("fan_out/no_marker_control", &plan, 0);
    Ok((golden, control_golden))
}

#[tokio::test]
async fn the_row_number_counts_each_order_once_beside_its_lines() -> Result<(), Box<dyn Error>> {
    let (golden, control) = marker_per_order_and_no_marker_control().await?;
    golden.assert_committed();
    control.assert_committed();
    Ok(())
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        let (golden, control) = marker_per_order_and_no_marker_control()
            .await
            .expect("fan_out/marker_per_order and fan_out/no_marker_control");
        vec![
            latest_by_month().await.expect("marker/latest_by_month"),
            latest_by_account().await.expect("marker/latest_by_account"),
            rank_ignores_condition()
                .await
                .expect("condition/rank_ignores_condition"),
            nulls_last_tie().await.expect("nulls/nulls_last_tie"),
            golden,
            control,
        ]
    })
    .await;
}
