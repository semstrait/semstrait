mod support;

use std::error::Error;

use semstrait_adapter::{InvalidRewrite, SubstraitRules, VocabularyPart};
use semstrait_ir::{
    AggregateNode, EmitForm, IrErrorKind, Parameter, PhysicalExpr, PlanNode, ProjectNode, ScanNode,
    Schema, TableScan,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, Call, Expr, ExprOperationExt, FunctionRef, Over, WindowBound, WindowCall,
    WindowFrame, WindowFrameKind,
};
use semstrait_sql::functions::registry;
use substrait::proto::expression::window_function::bound::{
    CurrentRow, Kind, Preceding, Unbounded,
};
use substrait::proto::expression::window_function::{Bound, BoundsType};
use substrait::proto::expression::{RexType, WindowFunction};
use substrait::proto::sort_field::{SortDirection, SortKind};
use substrait::proto::{Expression, Plan, Rel, SortField};

use support::{
    aggregate, aggregate_rel, column, declared_function, descending, field, leaf, literal, name,
    plan, produce, project_rel, refusal, relations, root_input, schema, text, Faithful,
};

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

fn rows(table: &str, columns: &[(&str, DataType)]) -> Result<PlanNode, Box<dyn Error>> {
    let fields = (columns.iter()).map(|(column, data_type)| field(column, data_type.clone(), true));
    let reader = TableScan::named_table([String::new(), "sales".into(), table.into()], None);
    let leaves = (columns.iter())
        .map(|(column, _)| (name(&format!("ssd0.{column}")), leaf(&["ps0", column])))
        .collect();
    let rows = ProjectNode::new(ScanNode::new(reader, schema(fields))?, leaves)?;
    let emit = EmitForm::new(columns.len()..2 * columns.len());
    Ok(PlanNode::from(rows).with_emit_form(emit)?)
}

fn marker_plan(
    rows: PlanNode,
    (marker, call): (&str, PhysicalExpr),
    function: &str,
    grouping: &str,
    [first, second]: [&str; 2],
) -> Result<PlanNode, Box<dyn Error>> {
    let marked = ProjectNode::new(rows, vec![(name(&format!("ssd0.{marker}")), call)])?;
    let once = leaf(&["ssd0", marker]).eq(literal(1_i64));
    let aggregates = [(first, Some(once)), (second, None)].map(|(field, filter)| {
        let call = aggregate(function, vec![leaf(&["ssd0", field])], filter);
        (name(&format!("ssd0.{field}")), call)
    });
    let groupings = vec![(name(&format!("ssd0.{grouping}")), leaf(&["ssd0", grouping]))];
    let aggregated = AggregateNode::new(marked, groupings, Vec::from(aggregates))?;
    let fields = [grouping, first, second];
    let leaves = fields.map(|field| (name(field), leaf(&["ssd0", field])));
    let root = ProjectNode::new(aggregated, Vec::from(leaves))?;
    let emit = EmitForm::new(fields.len()..2 * fields.len());
    Ok(PlanNode::from(root).with_emit_form(emit)?)
}

fn daily() -> Result<PlanNode, Box<dyn Error>> {
    rows(
        "daily",
        &[
            ("account", DataType::String),
            ("date", DataType::Date),
            ("month", DataType::String),
            ("balance", DataType::BigInt),
            ("cost", DataType::BigInt),
        ],
    )
}

fn windows_r1() -> Result<PlanNode, Box<dyn Error>> {
    let rank = Over {
        partition_by: vec![leaf(&["ssd0", "month"])],
        order_by: vec![descending(leaf(&["ssd0", "date"]))],
        window_frame: None,
    };
    let marker = ("__plan_rank_balance", window("rank", Vec::new(), rank));
    marker_plan(daily()?, marker, "sum", "month", ["balance", "cost"])
}

fn windows_r5() -> Result<PlanNode, Box<dyn Error>> {
    let order_rows = rows(
        "order_rows",
        &[
            ("order_id", DataType::BigInt),
            ("region", DataType::String),
            ("shipping_cost", DataType::BigInt),
            ("line_id", DataType::BigInt),
            ("price", DataType::BigInt),
        ],
    )?;
    let row_number = Over {
        partition_by: ["region", "order_id", "shipping_cost"]
            .map(|column| leaf(&["ssd0", column]))
            .to_vec(),
        ..Over::default()
    };
    let row_number = window("row_number", Vec::new(), row_number);
    let marker = ("__plan_row_number_shipping_cost", row_number);
    marker_plan(
        order_rows,
        marker,
        "avg",
        "region",
        ["shipping_cost", "price"],
    )
}

fn marked(relation: &Rel) -> (&Rel, &WindowFunction) {
    let [_, aggregate, marked, ..] = relations(relation)[..] else {
        panic!("a root projection over an aggregate over a marker projection");
    };
    (aggregate, window_function(marked))
}

fn window_function(relation: &Rel) -> &WindowFunction {
    let [expression] = project_rel(relation).expressions.as_slice() else {
        panic!("one expression");
    };
    let Some(RexType::WindowFunction(window)) = &expression.rex_type else {
        panic!("a WindowFunction");
    };
    window
}

fn assert_marker(plan: &Plan, function: &str, partitions: &[Expression], sorts: &[SortField]) {
    let (aggregate, marker) = marked(root_input(plan));
    assert_eq!(
        declared_function(plan, marker.function_reference),
        Some(("extension:io.substrait:functions_arithmetic", function))
    );
    assert_eq!(marker.partitions, partitions);
    assert_eq!(marker.sorts, sorts);
    assert_eq!(
        (marker.bounds_type, marker.lower_bound, marker.upper_bound),
        (BoundsType::Unspecified.into(), None, None)
    );
    let [marked, unmarked] = aggregate_rel(aggregate).measures.as_slice() else {
        panic!("two measures");
    };
    let filter = (marked.filter.as_ref()).map(|filter| text(plan, filter));
    assert_eq!(filter.as_deref(), Some("equal($5, 1)"));
    assert_eq!(unmarked.filter, None);
}

#[test]
fn windows_p1_r1s_rank_is_a_window_function_and_its_condition_the_measure_filter(
) -> Result<(), Box<dyn Error>> {
    let date = SortField {
        expr: Some(column(1)),
        sort_kind: Some(SortKind::Direction(SortDirection::DescNullsLast.into())),
    };
    let plan = produce(windows_r1()?, &Faithful)?;
    assert_marker(&plan, "rank", &[column(2)], &[date]);
    Ok(())
}

#[test]
fn windows_p1_r5s_row_number_is_a_window_function_and_its_condition_the_measure_filter(
) -> Result<(), Box<dyn Error>> {
    let partitions = [column(1), column(0), column(2)];
    let plan = produce(windows_r5()?, &Faithful)?;
    assert_marker(&plan, "row_number", &partitions, &[]);
    Ok(())
}

#[test]
fn a_window_frame_or_sql_s_default_lowers_as_its_bounds_type_and_its_two_bounds(
) -> Result<(), Box<dyn Error>> {
    let bound = |kind| Some(Bound { kind: Some(kind) });
    let current_row = bound(Kind::CurrentRow(CurrentRow {}));
    let unbounded = bound(Kind::Unbounded(Unbounded {}));
    let by_date = || vec![descending(leaf(&["ssd0", "date"]))];
    let cases = [
        (
            "first_value",
            by_date(),
            Some(WindowFrame::new(
                WindowFrameKind::Rows,
                WindowBound::Preceding(1),
                WindowBound::CurrentRow,
            )),
            (
                BoundsType::Rows,
                bound(Kind::Preceding(Preceding { offset: 1 })),
                current_row,
            ),
        ),
        (
            "first_value",
            by_date(),
            Some(WindowFrame::new(
                WindowFrameKind::Range,
                WindowBound::UnboundedPreceding,
                WindowBound::Following(0),
            )),
            (BoundsType::Range, unbounded, current_row),
        ),
        (
            "last_value",
            by_date(),
            None,
            (BoundsType::Range, unbounded, current_row),
        ),
        (
            "first_value",
            Vec::new(),
            None,
            (BoundsType::Unspecified, None, None),
        ),
    ];
    for (function, order_by, window_frame, (bounds_type, lower_bound, upper_bound)) in cases {
        let over = Over {
            order_by,
            window_frame,
            ..Over::default()
        };
        let call = window(function, vec![leaf(&["ssd0", "balance"])], over);
        let plan = produce(
            ProjectNode::new(daily()?, vec![(name("value"), call)])?,
            &Faithful,
        )?;

        let got = window_function(root_input(&plan));
        assert_eq!(
            (got.bounds_type, got.lower_bound, got.upper_bound),
            (bounds_type.into(), lower_bound, upper_bound),
            "{function} {window_frame:?}"
        );
    }
    Ok(())
}

fn first_value_by_date(kind: WindowFrameKind) -> PhysicalExpr {
    let over = Over {
        order_by: vec![descending(leaf(&["ssd0", "date"]))],
        window_frame: Some(WindowFrame::new(
            kind,
            WindowBound::CurrentRow,
            WindowBound::CurrentRow,
        )),
        ..Over::default()
    };
    window("first_value", vec![leaf(&["ssd0", "balance"])], over)
}

fn first_value_projection(kind: WindowFrameKind) -> Result<ProjectNode, Box<dyn Error>> {
    let call = first_value_by_date(kind);
    Ok(ProjectNode::new(daily()?, vec![(name("value"), call)])?)
}

struct RowsAsGroups;

impl SubstraitRules for RowsAsGroups {
    fn rewrite_expression(&self, expr: &PhysicalExpr, _input: &Schema) -> Option<PhysicalExpr> {
        (*expr == first_value_by_date(WindowFrameKind::Rows))
            .then(|| first_value_by_date(WindowFrameKind::Groups))
    }
}

fn unknown_window_frame() -> Option<(InvalidRewrite, Vec<String>)> {
    let kind = InvalidRewrite::UnknownForm {
        part: VocabularyPart::WindowFrame,
    };
    assert_eq!(kind.to_string(), "the producer knows no window frame");
    Some((kind, vec!["node 1 (ProjectNode)".to_owned()]))
}

#[test]
fn a_groups_window_frame_that_reaches_the_producer_is_an_unknown_window_frame(
) -> Result<(), Box<dyn Error>> {
    let projection = first_value_projection(WindowFrameKind::Groups)?;
    let validated = plan(projection.clone())
        .validate()
        .map_err(|refusal| refusal.kind);
    assert_eq!(
        validated,
        Err(IrErrorKind::UnfitWindowFrame {
            function: "first_value"
        })
    );

    assert_eq!(refusal(projection, &Faithful), unknown_window_frame());
    Ok(())
}

#[test]
fn a_rule_that_writes_a_groups_window_frame_is_an_unknown_window_frame(
) -> Result<(), Box<dyn Error>> {
    let projection = first_value_projection(WindowFrameKind::Rows)?;
    assert_eq!(
        plan(projection.clone())
            .validate()
            .map_err(|refusal| refusal.kind),
        Ok(())
    );

    assert_eq!(refusal(projection, &RowsAsGroups), unknown_window_frame());
    Ok(())
}
