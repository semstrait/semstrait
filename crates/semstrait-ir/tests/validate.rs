mod support;

use semstrait_ir::{
    AggregateNode, Anchor, CrossNode, EmitForm, FetchMode, FetchNode, FilterNode, IrErrorKind,
    JoinNode, JoinType, Parameter, ParameterKey, PhysicalExpr, PlanNode, ProjectNode, Reader,
    ScanNode, SchemaColumn, SemanticPlan, SortNode, TableScan, TopNNode, VirtualTable,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, CallError, Expr, ExprOperationExt, FunctionRef, Literal,
    LiteralError, NamedReference, NullOrdering, NullTreatment, Over, SortDir, WindowBound,
    WindowCall, WindowFrame, WindowFrameKind,
};
use semstrait_sql::functions::registry;

use support::{
    ascending, at, bigint, customers, field, key, leaf, name, orders, orders_by_customers,
    orders_with_customers, plan, projected, projection, refusal, same_region, schema, string_agg,
    sum, text, unbound,
};

#[test]
fn a_plan_that_its_constructors_build_is_accepted() -> Result<(), IrErrorKind> {
    let row_projection = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![
            (name("ssd0.region"), leaf(&["ps0", "region"])),
            (name("ssd0.amount"), leaf(&["ps0", "amount"])),
        ],
    )?)
    .with_emit_form(EmitForm::new([3, 4]))?;
    let filter = FilterNode::new(row_projection, leaf(&["ssd0", "amount"]).gt(bigint(10)))?;
    let aggregate = AggregateNode::new(
        filter,
        vec![(name("ssd0.region"), leaf(&["ssd0", "region"]))],
        vec![(name("ssd0.total"), sum(leaf(&["ssd0", "amount"]), None))],
    )?;
    let root = PlanNode::from(ProjectNode::new(
        aggregate,
        vec![
            (name("region"), leaf(&["ssd0", "region"])),
            (name("revenue"), leaf(&["ssd0", "total"])),
        ],
    )?)
    .with_emit_form(EmitForm::new([2, 3]))?;
    let sorted = SortNode::new(root, vec![ascending(leaf(&["region"]))])?;

    assert_eq!(refusal(&plan(FetchNode::new(sorted, 0, Some(10))?)), None);
    Ok(())
}

#[test]
fn an_anchor_that_repeats_is_refused_at_the_node_that_repeats_it() -> Result<(), IrErrorKind> {
    let filter = FilterNode::new(orders()?, leaf(&["ps0", "amount"]).gt(bigint(0)))?;
    let mut plan = plan(filter);
    assert_eq!(refusal(&plan), None);
    if let PlanNode::Filter(filter) = &mut plan.root {
        filter.common.anchor = Anchor(2);
    }

    assert_eq!(
        refusal(&plan),
        Some((
            IrErrorKind::RepeatedAnchor { anchor: Anchor(2) },
            at(2, "ScanNode")
        ))
    );
    Ok(())
}

#[test]
fn each_construction_rule_is_checked_again_over_a_plan_built_in_parts_or_rewritten(
) -> Result<(), IrErrorKind> {
    let amount = || leaf(&["ps0", "amount"]);
    let positive = || amount().gt(bigint(0));

    let mut unbound = FilterNode::new(orders()?, positive())?;
    unbound.input = Box::new(
        ScanNode::new(
            VirtualTable::new(Vec::new()),
            schema([field("region", DataType::String, true)]),
        )?
        .into(),
    );
    let mut inexact = FilterNode::new(orders()?, positive())?;
    inexact.filter = leaf(&["ps0", "Amount"]).gt(bigint(0));

    let mut untyped = ProjectNode::new(orders()?, vec![(name("x"), amount() * bigint(2))])?;
    untyped.expressions = vec![amount() + text("one")];
    let no_type = ProjectNode::new(orders()?, vec![(name("x"), amount() + text("one"))]);

    let mut unknown = orders()?;
    unknown.schema.columns[1].data_type = DataType::Unknown;

    let scores = ScanNode::new(
        VirtualTable::new(Vec::new()),
        schema([
            field("score", DataType::Double, false),
            field(
                "price",
                DataType::Decimal {
                    precision: 18,
                    scale: 2,
                },
                false,
            ),
        ]),
    )?;
    let mut typed_unknown = ProjectNode::new(scores, vec![(name("x"), leaf(&["ps0", "score"]))])?;
    typed_unknown.expressions = vec![leaf(&["ps0", "score"]) * leaf(&["ps0", "price"])];

    let mut not_a_condition = FilterNode::new(orders()?, positive())?;
    not_a_condition.filter = amount() + bigint(1);

    let mut out_of_range = FilterNode::new(orders()?, positive())?;
    out_of_range.common.emit_form = Some(EmitForm::new([0, 3]));

    let mut repeated = ProjectNode::new(
        orders()?,
        vec![(name("total"), amount()), (name("paid"), amount())],
    )?;
    repeated.common.output_schema.columns[4].name = "TOTAL".into();

    let mut negative = FetchNode::new(orders()?, 0, None)?;
    negative.offset = -1;

    let mut unspecified = SortNode::new(orders()?, vec![ascending(amount())])?;
    unspecified.order_by[0].dir = SortDir::Desc {
        nulls: NullOrdering::Unspecified,
    };

    let mut unspecified_window = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![(name("r"), rank(NullOrdering::Last))],
    )?)
    .with_emit_form(EmitForm::new([3]))?;
    if let PlanNode::Project(projection) = &mut unspecified_window {
        projection.expressions = vec![rank(NullOrdering::Unspecified)];
    }

    let faults: [(PlanNode, IrErrorKind, &str); 11] = [
        (
            unbound.into(),
            IrErrorKind::UnboundLeaf {
                leaf: NamedReference::qualified("ps0", "amount").expect("a name path"),
                fields: vec!["region".into()],
            },
            "FilterNode",
        ),
        (
            inexact.into(),
            IrErrorKind::UnboundLeaf {
                leaf: NamedReference::qualified("ps0", "Amount").expect("a name path"),
                fields: vec!["region".into(), "amount".into(), "status".into()],
            },
            "FilterNode",
        ),
        (
            untyped.into(),
            no_type.expect_err("typing refuses a number plus a string"),
            "ProjectNode",
        ),
        (
            unknown.into(),
            IrErrorKind::InvalidType {
                field: "amount".into(),
                data_type: DataType::Unknown,
            },
            "ScanNode",
        ),
        (
            typed_unknown.into(),
            IrErrorKind::InvalidType {
                field: "x".into(),
                data_type: DataType::Unknown,
            },
            "ProjectNode",
        ),
        (
            not_a_condition.into(),
            IrErrorKind::NotACondition {
                got: DataType::BigInt,
            },
            "FilterNode",
        ),
        (
            out_of_range.into(),
            IrErrorKind::EmitFormIndexOutOfRange {
                index: 3,
                fields: 3,
            },
            "FilterNode",
        ),
        (
            repeated.into(),
            IrErrorKind::RepeatedName {
                name: "TOTAL".into(),
            },
            "ProjectNode",
        ),
        (
            negative.into(),
            IrErrorKind::NegativeBound { bound: -1 },
            "FetchNode",
        ),
        (
            unspecified.into(),
            IrErrorKind::UnspecifiedNullOrdering { key: 0 },
            "SortNode",
        ),
        (
            unspecified_window,
            IrErrorKind::UnspecifiedWindowNullOrdering {
                function: "rank",
                key: 0,
            },
            "ProjectNode",
        ),
    ];
    for (root, kind, node) in faults {
        assert_eq!(refusal(&plan(root)), Some((kind, at(1, node))));
    }
    Ok(())
}

#[test]
fn each_construction_rule_is_checked_again_over_a_join_built_in_parts_or_rewritten(
) -> Result<(), IrErrorKind> {
    let left_join =
        |post_join_filter| orders_with_customers(JoinType::Left, same_region(), post_join_filter);
    let no_name = || Some(leaf(&["ssd1", "name"]).is_null());
    let natural = [
        "ssd0.region",
        "ssd0.amount",
        "ssd0.status",
        "ssd2.region",
        "ssd2.name",
    ];

    let mut moved = left_join(None)?;
    moved.right = Box::new(projection("ssd2", customers()?)?);
    let mut filter_moved = left_join(no_name())?;
    filter_moved.right = Box::new(projection("ssd2", customers()?)?);
    filter_moved.on = leaf(&["ssd0", "region"]).eq(leaf(&["ssd2", "region"]));

    let mut inner_schema = left_join(None)?;
    inner_schema.common.output_schema =
        orders_with_customers(JoinType::Inner, same_region(), None)?
            .common
            .output_schema;
    let mut retyped = left_join(None)?;
    retyped.join_type = JoinType::Right;

    let faults = [
        (moved, unbound(&["ssd1", "region"], &natural)),
        (filter_moved, unbound(&["ssd1", "name"], &natural)),
        (inner_schema, IrErrorKind::OutputSchemaMismatch { index: 3 }),
        (retyped, IrErrorKind::OutputSchemaMismatch { index: 1 }),
    ];
    for (join, kind) in faults {
        assert_eq!(refusal(&plan(join)), Some((kind, at(1, "JoinNode"))));
    }

    let emitted = PlanNode::from(left_join(no_name())?).with_emit_form(EmitForm::new([0, 1]))?;
    assert_eq!(refusal(&plan(emitted)), None);
    Ok(())
}

#[test]
fn a_virtual_table_row_of_another_arity_or_other_types_than_its_schema_is_refused(
) -> Result<(), IrErrorKind> {
    let fields = || {
        schema([
            field("id", DataType::BigInt, false),
            field("region", DataType::String, true),
        ])
    };
    let rows = |rows| ScanNode::new(VirtualTable::new(rows), fields());
    let no_region = Expr::Literal(Literal::Null(DataType::String));
    let read = rows(vec![
        vec![bigint(1), text("eu")],
        vec![bigint(2), no_region],
    ])?;
    assert_eq!(refusal(&plan(read.clone())), None);

    let short = IrErrorKind::RowArity {
        row: 1,
        fields: 2,
        got: 1,
    };
    let mistyped = IrErrorKind::RowType {
        row: 0,
        field: "id".into(),
        expected: DataType::BigInt,
        got: DataType::String,
    };
    let no_id = Expr::Literal(Literal::Null(DataType::BigInt));
    let null = IrErrorKind::RowNull {
        row: 0,
        field: "id".into(),
    };
    for (values, kind) in [
        (vec![vec![bigint(1), text("eu")], vec![bigint(2)]], short),
        (vec![vec![text("1"), text("eu")]], mistyped),
        (vec![vec![no_id, text("eu")]], null),
    ] {
        assert_eq!(rows(values.clone()).err(), Some(kind.clone()));
        let mut rewritten = read.clone();
        rewritten.reader = Reader::VirtualTable(VirtualTable::new(values));
        assert_eq!(refusal(&plan(rewritten)), Some((kind, at(1, "ScanNode"))));
    }
    Ok(())
}

#[test]
fn an_output_schema_that_is_not_the_emit_form_over_the_natural_output_is_refused(
) -> Result<(), IrErrorKind> {
    let mismatch = |index| {
        Some((
            IrErrorKind::OutputSchemaMismatch { index },
            at(1, "ProjectNode"),
        ))
    };
    let doubled = || {
        PlanNode::from(ProjectNode::new(
            orders()?,
            vec![(name("doubled"), leaf(&["ps0", "amount"]) * bigint(2))],
        )?)
        .with_emit_form(EmitForm::new([0, 3]))
    };
    let rewritten = |rewrite: fn(&mut Vec<SchemaColumn>)| {
        let PlanNode::Project(mut projection) = doubled()? else {
            panic!("a projection");
        };
        rewrite(&mut projection.common.output_schema.columns);
        Ok::<_, IrErrorKind>(refusal(&plan(projection)))
    };

    assert_eq!(rewritten(|fields| fields[0].nullable = false)?, mismatch(0));
    assert_eq!(
        rewritten(|fields| fields[1].data_type = DataType::Integer)?,
        mismatch(1)
    );
    assert_eq!(
        rewritten(|fields| fields.push(field("extra", DataType::BigInt, true)))?,
        mismatch(2)
    );
    assert_eq!(rewritten(|fields| fields.truncate(1))?, mismatch(1));
    assert_eq!(refusal(&plan(doubled()?)), None);
    Ok(())
}

#[test]
fn a_computed_field_with_no_name_is_refused() -> Result<(), IrErrorKind> {
    let mut projection = ProjectNode::new(orders()?, vec![(name("x"), leaf(&["ps0", "amount"]))])?;
    projection.common.output_schema.columns[3].name = String::new();

    assert_eq!(
        refusal(&SemanticPlan::new(projection.into(), Vec::new())),
        Some((IrErrorKind::UnnamedField { index: 3 }, at(1, "ProjectNode")))
    );
    Ok(())
}

#[test]
fn an_aggregate_whose_name_or_type_is_not_its_fields_is_refused() -> Result<(), IrErrorKind> {
    let total = || {
        AggregateNode::new(
            orders()?,
            vec![(name("region"), leaf(&["ps0", "region"]))],
            vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
        )
    };
    let stored = field("total", DataType::BigInt, true);

    let mut renamed = total()?;
    renamed.aggregates[0].0 = name("revenue");
    let mut retyped = total()?;
    retyped.common.output_schema.columns[1].data_type = DataType::Double;

    assert_eq!(
        refusal(&plan(renamed)),
        Some((
            IrErrorKind::AggregateFieldMismatch {
                expected: field("revenue", DataType::BigInt, true),
                got: stored.clone(),
            },
            at(1, "AggregateNode")
        ))
    );
    assert_eq!(
        refusal(&plan(retyped)),
        Some((
            IrErrorKind::AggregateFieldMismatch {
                expected: stored,
                got: field("total", DataType::Double, true),
            },
            at(1, "AggregateNode")
        ))
    );
    Ok(())
}

#[test]
fn a_repeated_name_case_ignored_is_refused_in_every_output_schema_but_a_physical_read(
) -> Result<(), IrErrorKind> {
    let repeated = |name: &str, node| {
        Some((
            IrErrorKind::RepeatedName {
                name: name.to_owned(),
            },
            at(1, node),
        ))
    };
    let shadowing = ProjectNode::new(orders()?, vec![(name("REGION"), leaf(&["ps0", "region"]))])?;
    assert_eq!(refusal(&plan(shadowing)), repeated("REGION", "ProjectNode"));
    let mut twice = FilterNode::new(orders()?, leaf(&["ps0", "amount"]).gt(bigint(0)))?;
    twice.common.emit_form = Some(EmitForm::new([0, 0]));
    twice.common.output_schema = schema([
        field("region", DataType::String, true),
        field("region", DataType::String, true),
    ]);
    assert_eq!(refusal(&plan(twice)), repeated("region", "FilterNode"));

    let fields = || {
        schema([
            field("id", DataType::BigInt, false),
            field("ID", DataType::BigInt, false),
        ])
    };
    let table = TableScan::named_table([String::new(), "sales".into(), "ids".into()], None);
    let reads = [
        ScanNode::new(table, fields())?,
        ScanNode::new(VirtualTable::new(Vec::new()), fields())?,
    ];
    for read in reads {
        assert_eq!(refusal(&plan(read.clone())), None);
        let emitted = PlanNode::from(read.clone()).with_emit_form(EmitForm::new([0, 0]))?;
        assert_eq!(refusal(&plan(emitted)), None);
        let above = FilterNode::new(read, leaf(&["ps0", "id"]).gt(bigint(0)))?;
        assert_eq!(refusal(&plan(above)), repeated("ID", "FilterNode"));
    }

    let inputs = || -> Result<[PlanNode; 2], IrErrorKind> {
        Ok([
            projection("ssd0", customers()?)?,
            projection("SSD0", orders()?)?,
        ])
    };
    let [left, right] = inputs()?;
    let on = leaf(&["ssd0", "name"]).eq(leaf(&["SSD0", "status"]));
    let join = JoinNode::new(left, right, JoinType::Inner, on, None)?;
    assert_eq!(refusal(&plan(join)), repeated("SSD0.region", "JoinNode"));
    let [left, right] = inputs()?;
    let cross = CrossNode::new(left, right);
    assert_eq!(refusal(&plan(cross)), repeated("SSD0.region", "CrossNode"));
    Ok(())
}

#[test]
fn output_names_that_are_not_the_roots_field_names_are_refused() -> Result<(), IrErrorKind> {
    let root = || PlanNode::from(orders()?).with_emit_form(EmitForm::new([0, 1]));
    let names = |names: &[&str]| names.iter().map(|each| name(each)).collect::<Vec<_>>();
    let differ = |output_names: &[&str]| {
        Some((
            IrErrorKind::OutputNamesMismatch {
                output_names: output_names.iter().map(|each| each.to_string()).collect(),
                fields: vec!["region".into(), "amount".into()],
            },
            at(1, "ScanNode"),
        ))
    };

    for other in [
        &["region"][..],
        &["region", "amount", "status"],
        &["region", "Amount"],
    ] {
        let plan = SemanticPlan::new(root()?, names(other));
        assert_eq!(refusal(&plan), differ(other));
    }
    let plan = SemanticPlan::new(root()?, names(&["region", "amount"]));
    assert_eq!(refusal(&plan), None);
    Ok(())
}

#[test]
fn a_root_name_that_starts_with_a_reserved_prefix_is_refused_at_the_root() -> Result<(), IrErrorKind>
{
    for reserved in ["__semstrait_x", "__plan_x", "__agg_x"] {
        let root = PlanNode::from(ProjectNode::new(
            orders()?,
            vec![(name(reserved), leaf(&["ps0", "amount"]))],
        )?)
        .with_emit_form(EmitForm::new([3]))?;
        let plan = SemanticPlan::new(root, vec![name(reserved)]);

        assert_eq!(
            refusal(&plan),
            Some((
                IrErrorKind::ReservedRootName {
                    name: reserved.to_owned()
                },
                at(1, "ProjectNode")
            ))
        );
    }
    Ok(())
}

#[test]
fn a_top_n_is_refused_as_a_node_with_no_lowering_yet() -> Result<(), IrErrorKind> {
    let top = TopNNode::new(
        orders()?,
        vec![ascending(leaf(&["ps0", "amount"]))],
        0,
        3,
        FetchMode::WithTies,
    )?;
    let root = FetchNode::new(top, 0, None)?;

    assert_eq!(
        refusal(&plan(root)),
        Some((IrErrorKind::TopNNotLowered, at(2, "TopNNode")))
    );
    Ok(())
}

#[test]
fn a_join_or_a_cross_that_its_constructor_builds_passes_validate() -> Result<(), IrErrorKind> {
    let join = orders_with_customers(JoinType::Left, same_region(), None)?;
    assert_eq!(refusal(&plan(join)), None);

    let filter = FilterNode::new(orders_by_customers()?, same_region())?;
    assert_eq!(refusal(&plan(filter)), None);
    Ok(())
}

fn sum_of_amount() -> PhysicalExpr {
    Expr::Function(Call::Aggregate(
        sum(leaf(&["ps0", "amount"]), None),
        AggregateAllowed,
    ))
}

#[test]
fn an_aggregate_call_is_refused_anywhere_but_as_an_aggregate_entry() -> Result<(), IrErrorKind> {
    let misplaced = |root: PlanNode, kind| {
        assert_eq!(
            refusal(&plan(root)),
            Some((
                IrErrorKind::MisplacedAggregate { function: "sum" },
                at(1, kind)
            ))
        );
    };
    misplaced(
        PlanNode::from(ProjectNode::new(
            orders()?,
            vec![(name("x"), sum_of_amount() + bigint(1))],
        )?)
        .with_emit_form(EmitForm::new([3]))?,
        "ProjectNode",
    );
    misplaced(
        FilterNode::new(orders()?, sum_of_amount().gt(bigint(1)))?.into(),
        "FilterNode",
    );
    misplaced(
        AggregateNode::new(orders()?, vec![(name("g"), sum_of_amount())], Vec::new())?.into(),
        "AggregateNode",
    );
    misplaced(
        SortNode::new(orders()?, vec![ascending(sum_of_amount())])?.into(),
        "SortNode",
    );

    let entry = AggregateNode::new(
        orders()?,
        Vec::new(),
        vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
    )?;
    assert_eq!(refusal(&plan(entry)), None);

    let Some(FunctionRef::Aggregate(function, AggregateAllowed)) = registry().get("sum") else {
        panic!("the catalog declares the aggregate `sum`");
    };
    let nested = [
        AggregateCall::builder(function, vec![bigint(1) + sum_of_amount()]).build(),
        AggregateCall::builder(function, vec![leaf(&["ps0", "amount"])])
            .filter(sum_of_amount().gt(bigint(1)))
            .build(),
    ];
    for entry in nested {
        assert!(matches!(entry, Err(CallError::AggregateInAggregate { .. })));
    }
    Ok(())
}

#[test]
fn a_parameter_is_refused_in_every_expression() -> Result<(), IrErrorKind> {
    let parameter = || {
        Expr::Leaf(Parameter {
            key: ParameterKey::RequestDimensionsMinusTemporal,
            data_type: DataType::BigInt,
        })
    };
    let unbound = Some((
        IrErrorKind::UnboundParameter {
            key: ParameterKey::RequestDimensionsMinusTemporal,
        },
        at(1, "AggregateNode"),
    ));

    let in_entry = AggregateNode::new(
        orders()?,
        Vec::new(),
        vec![(name("total"), sum(parameter(), None))],
    )?;
    assert_eq!(refusal(&plan(in_entry)), unbound);
    let in_grouping = AggregateNode::new(
        orders()?,
        vec![(name("g"), parameter() + bigint(1))],
        Vec::new(),
    )?;
    assert_eq!(refusal(&plan(in_grouping)), unbound);
    Ok(())
}

#[test]
fn a_call_that_carries_a_null_treatment_is_refused() -> Result<(), IrErrorKind> {
    let Some(FunctionRef::Window(first_value)) = registry().get("first_value") else {
        panic!("the catalog declares the window function `first_value`");
    };
    for treatment in [NullTreatment::IgnoreNulls, NullTreatment::RespectNulls] {
        let call = WindowCall::builder(first_value, vec![leaf(&["ps0", "amount"])])
            .nulls(treatment)
            .build()
            .expect("first_value takes a null treatment");
        let window = Expr::Function(Call::Window(call));
        let projection = projected(orders()?, vec![(name("field"), window)])?;

        assert_eq!(
            refusal(&plan(projection)),
            Some((
                IrErrorKind::CallWithNullTreatment {
                    function: "first_value"
                },
                at(1, "ProjectNode")
            ))
        );
    }
    Ok(())
}

#[test]
fn a_literal_whose_value_does_not_fit_its_type_is_refused_at_its_node() -> Result<(), IrErrorKind> {
    let day_second = |value: &str| {
        Expr::Literal(Literal::IntervalDaySecond {
            value: value.to_owned(),
            precision: 0,
        })
    };
    let cases = [
        (day_second("PT1000000H"), Some(LiteralError::OutOfRange)),
        (day_second("PT596523H14M8S"), Some(LiteralError::OutOfRange)),
        (day_second("PT596523H14M7S"), None),
        (
            Expr::Literal(Literal::Date("2026-02-30".to_owned())),
            Some(LiteralError::Malformed),
        ),
    ];
    for (literal, refused) in cases {
        let expected =
            refused.map(|error| (IrErrorKind::UnfitLiteral { error }, at(1, "ProjectNode")));
        assert_eq!(
            refusal(&plan(projected(
                orders()?,
                vec![(name("field"), literal.clone())]
            )?)),
            expected,
            "{literal:?}"
        );
    }
    Ok(())
}

fn window(
    function: &str,
    args: Vec<PhysicalExpr>,
    over: Over<Parameter, AggregateAllowed>,
) -> PhysicalExpr {
    let Some(FunctionRef::Window(window)) = registry().get(function) else {
        panic!("the catalog declares the window function `{function}`");
    };
    let call = WindowCall::builder(window, args).over(over).build();
    Expr::Function(Call::Window(call.expect("a window call")))
}

fn rank(nulls: NullOrdering) -> PhysicalExpr {
    let over = Over {
        order_by: vec![key(leaf(&["ps0", "amount"]), SortDir::Desc { nulls })],
        ..Over::default()
    };
    window("rank", Vec::new(), over)
}

fn row_number() -> PhysicalExpr {
    window("row_number", Vec::new(), Over::default())
}

#[test]
fn a_window_call_in_a_filter_is_refused_in_its_place() -> Result<(), IrErrorKind> {
    let marker = FilterNode::new(orders()?, rank(NullOrdering::Last).eq(bigint(1)))?;

    assert_eq!(
        refusal(&plan(marker)),
        Some((
            IrErrorKind::WindowInFilter { function: "rank" },
            at(1, "FilterNode")
        ))
    );
    Ok(())
}

#[test]
fn a_window_call_outside_a_projection_is_refused_with_the_variant_of_its_place(
) -> Result<(), IrErrorKind> {
    let marker = || rank(NullOrdering::Last);
    let rows = ScanNode::new(
        VirtualTable::new(vec![vec![row_number()]]),
        schema([field("n", DataType::BigInt, false)]),
    )?;
    let places: [(PlanNode, IrErrorKind, Vec<String>); 4] = [
        (
            orders()?.with_filter(marker().eq(bigint(1)))?.into(),
            IrErrorKind::WindowInFilter { function: "rank" },
            at(1, "ScanNode"),
        ),
        (
            SortNode::new(orders()?, vec![ascending(marker())])?.into(),
            IrErrorKind::WindowInSortKey { function: "rank" },
            at(1, "SortNode"),
        ),
        (
            AggregateNode::new(orders()?, vec![(name("g"), marker())], Vec::new())?.into(),
            IrErrorKind::WindowInAggregate { function: "rank" },
            at(1, "AggregateNode"),
        ),
        (
            rows.into(),
            IrErrorKind::WindowInReader {
                function: "row_number",
            },
            at(1, "ScanNode"),
        ),
    ];
    for (root, kind, node) in places {
        assert_eq!(refusal(&plan(root)), Some((kind, node)));
    }

    let Some(FunctionRef::Aggregate(count, AggregateAllowed)) = registry().get("count") else {
        panic!("the catalog declares the aggregate `count`");
    };
    let entries = [
        AggregateCall::builder(count, vec![marker()]).build(),
        AggregateCall::builder(count, vec![leaf(&["ps0", "amount"])])
            .filter(marker().eq(bigint(1)))
            .build(),
    ];
    for entry in entries {
        assert!(matches!(entry, Err(CallError::WindowInAggregate { .. })));
    }
    Ok(())
}

#[test]
fn an_aggregate_call_a_parameter_or_a_window_call_in_a_join_condition_or_a_post_join_filter_is_refused(
) -> Result<(), IrErrorKind> {
    let amount = || leaf(&["ssd0", "amount"]);
    let key = ParameterKey::RequestDimensionsMinusTemporal;
    let operands = [
        (
            Expr::Function(Call::Aggregate(sum(amount(), None), AggregateAllowed)),
            IrErrorKind::MisplacedAggregate { function: "sum" },
        ),
        (
            Expr::Leaf(Parameter {
                key,
                data_type: DataType::BigInt,
            }),
            IrErrorKind::UnboundParameter { key },
        ),
        (
            row_number(),
            IrErrorKind::WindowInJoin {
                function: "row_number",
            },
        ),
    ];

    for (operand, kind) in operands {
        let condition = || amount().gt(operand.clone());
        let in_on = orders_with_customers(JoinType::Inner, same_region() & condition(), None)?;
        let in_post_join_filter =
            orders_with_customers(JoinType::Left, same_region(), Some(condition()))?;
        for join in [in_on, in_post_join_filter] {
            assert_eq!(
                refusal(&plan(join)),
                Some((kind.clone(), at(1, "JoinNode")))
            );
        }
    }
    Ok(())
}

#[test]
fn a_window_order_key_with_an_unspecified_null_order_is_refused_as_a_sort_key_is(
) -> Result<(), IrErrorKind> {
    assert_eq!(
        ProjectNode::new(
            orders()?,
            vec![(name("r"), rank(NullOrdering::Unspecified))]
        ),
        Err(IrErrorKind::UnspecifiedWindowNullOrdering {
            function: "rank",
            key: 0
        })
    );
    Ok(())
}

#[test]
fn an_aggregates_order_key_with_an_unspecified_null_order_is_refused_over_a_rewritten_plan(
) -> Result<(), IrErrorKind> {
    let regions = |nulls| {
        let dir = SortDir::Asc { nulls };
        string_agg(leaf(&["ps0", "region"]), key(leaf(&["ps0", "status"]), dir))
    };
    let mut aggregate = AggregateNode::new(
        orders()?,
        Vec::new(),
        vec![(name("regions"), regions(NullOrdering::Last))],
    )?;
    aggregate.aggregates[0].1 = regions(NullOrdering::Unspecified);

    assert_eq!(
        refusal(&plan(aggregate)),
        Some((
            IrErrorKind::UnspecifiedAggregateNullOrdering {
                function: "string_agg",
                key: 0
            },
            at(1, "AggregateNode")
        ))
    );
    Ok(())
}

#[test]
fn a_window_call_in_a_projection_types_bigint_never_null_and_is_accepted() -> Result<(), IrErrorKind>
{
    let marker = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![(name("__plan_rank_amount"), rank(NullOrdering::Last))],
    )?)
    .with_emit_form(EmitForm::new([0, 3]))?;
    assert_eq!(
        marker.common().output_schema,
        schema([
            field("region", DataType::String, true),
            field("__plan_rank_amount", DataType::BigInt, false),
        ])
    );
    let root =
        PlanNode::from(ProjectNode::new(marker, Vec::new())?).with_emit_form(EmitForm::new([0]))?;

    assert_eq!(refusal(&plan(root)), None);
    Ok(())
}

#[test]
fn a_window_frame_that_substrait_cannot_hold_is_refused_at_its_projection(
) -> Result<(), IrErrorKind> {
    let amount = || leaf(&["ps0", "amount"]);
    let largest = i64::MAX.unsigned_abs();
    let frames = [
        (WindowFrameKind::Groups, WindowBound::CurrentRow, true),
        (
            WindowFrameKind::Range,
            WindowBound::Following(largest + 1),
            true,
        ),
        (
            WindowFrameKind::Rows,
            WindowBound::Following(largest),
            false,
        ),
    ];
    for (kind, end, refused) in frames {
        let over = Over {
            order_by: vec![ascending(amount())],
            window_frame: Some(WindowFrame::new(kind, WindowBound::CurrentRow, end)),
            ..Over::default()
        };
        let first = window("first_value", vec![amount()], over);

        let function = "first_value";
        let expected = refused.then_some((
            IrErrorKind::UnfitWindowFrame { function },
            at(1, "ProjectNode"),
        ));
        let projection = projected(orders()?, vec![(name("field"), first)])?;
        assert_eq!(refusal(&plan(projection)), expected);
    }
    Ok(())
}

#[test]
fn a_parameter_in_a_window_calls_partition_is_refused_in_its_projection() -> Result<(), IrErrorKind>
{
    let key = ParameterKey::RequestDimensionsMinusTemporal;
    let partition = Expr::Leaf(Parameter {
        key,
        data_type: DataType::BigInt,
    });
    let over = Over {
        partition_by: vec![partition],
        ..Over::default()
    };
    let row_number = window("row_number", Vec::new(), over);

    assert_eq!(
        refusal(&plan(projected(
            orders()?,
            vec![(name("field"), row_number)]
        )?)),
        Some((IrErrorKind::UnboundParameter { key }, at(1, "ProjectNode")))
    );
    Ok(())
}
