mod support;

use semstrait_catalog::FileFormat;
use semstrait_common::DataKindName;
use semstrait_common::{Tree, ValidateError};
use semstrait_ir::{
    AggregateNode, Anchor, Cardinality, EmitForm, FetchMode, FetchNode, FilesScan, FilterNode,
    GranularityGroup, IrErrorKind, JoinNode, JoinType, MultisetNode, MultisetOp, PhysicalExpr,
    PlanNode, ProjectNode, Reader, ScanNode, Schema, SchemaColumn, SemAnnotation, SemanticPlan,
    SortNode, SourceRole, TableScan, TopNNode, VirtualTable,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, NamedReference, NullOrdering, SortDir};

use support::{
    ascending, at, bigint, customers, field, key, leaf, name, orders, orders_by_customers,
    orders_with_customers, plan, projected, projection, refusal, same_region, schema, string_agg,
    sum, text, unbound,
};

fn anchors(node: &PlanNode) -> Vec<Anchor> {
    std::iter::once(node.common().anchor)
        .chain(node.children().into_iter().flat_map(anchors))
        .collect()
}

#[test]
fn semantic_plan_new_numbers_a_five_node_plan_in_pre_order_the_root_one() -> Result<(), IrErrorKind>
{
    let row_projection = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![
            (name("region"), leaf(&["ps0", "region"])),
            (name("amount"), leaf(&["ps0", "amount"])),
        ],
    )?)
    .with_emit_form(EmitForm::new([3, 4]))?;
    let filter = FilterNode::new(row_projection, leaf(&["amount"]).gt(bigint(10)))?;
    let aggregate = AggregateNode::new(
        filter,
        vec![(name("region"), leaf(&["region"]))],
        vec![(name("total"), sum(leaf(&["amount"]), None))],
    )?;
    let root = PlanNode::from(ProjectNode::new(
        aggregate,
        vec![(name("revenue"), leaf(&["total"]))],
    )?)
    .with_emit_form(EmitForm::new([0, 2]))?;
    assert_eq!(
        anchors(&root),
        [Anchor::default(); 5],
        "a constructor writes no anchor"
    );

    let plan = SemanticPlan::new(root, vec![name("region"), name("revenue")]);

    assert_eq!(plan.root.common().anchor, Anchor(1));
    assert_eq!(anchors(&plan.root), (1..=5).map(Anchor).collect::<Vec<_>>());
    let scan = plan.root.children()[0].children()[0].children()[0].children()[0];
    assert!(matches!(scan, PlanNode::Scan(_)));
    assert_eq!(scan.common().anchor, Anchor(5));
    Ok(())
}

#[test]
fn a_nodes_diagnostic_names_the_node_by_its_anchor_and_its_type() -> Result<(), IrErrorKind> {
    let plan = plan(FilterNode::new(
        orders()?,
        leaf(&["amount"]).gt(bigint(10)),
    )?);
    let kind = IrErrorKind::EmitFormLeavesOutComputedField { index: 0 };

    let diagnostic = plan.root.children()[0].diagnostic(kind.clone());

    assert_eq!(diagnostic.kind, kind);
    assert_eq!(diagnostic.notes, ["node 2 (ScanNode)"]);
    Ok(())
}

#[test]
fn a_projection_whose_emit_form_leaves_out_the_input_fields_outputs_only_what_it_computes(
) -> Result<(), IrErrorKind> {
    let projection = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![(name("doubled"), leaf(&["ps0", "amount"]) * bigint(2))],
    )?)
    .with_emit_form(EmitForm::new([3]))?;

    assert_eq!(
        projection.common().output_schema,
        schema([field("doubled", DataType::BigInt, false)])
    );
    Ok(())
}

#[test]
fn a_projection_outputs_the_input_fields_then_its_expressions() -> Result<(), IrErrorKind> {
    let projection = ProjectNode::new(
        orders()?,
        vec![(name("paid"), leaf(&["ps0", "status"]).eq(text("paid")))],
    )?;

    assert_eq!(
        projection.common.output_schema,
        schema([
            field("region", DataType::String, true),
            field("amount", DataType::BigInt, false),
            field("status", DataType::String, true),
            field("paid", DataType::Boolean, true),
        ])
    );
    assert_eq!(projection.common.emit_form, None);
    Ok(())
}

#[test]
fn a_physical_leaf_binds_by_its_last_part_and_any_other_by_its_parts_case_ignored(
) -> Result<(), IrErrorKind> {
    let schema_of = |expr: PhysicalExpr| -> Result<Schema, IrErrorKind> {
        let projection = ProjectNode::new(orders()?, vec![(name("x"), expr)])?;
        Ok(PlanNode::from(projection)
            .with_emit_form(EmitForm::new([3]))?
            .common()
            .output_schema
            .clone())
    };

    assert_eq!(
        schema_of(leaf(&["ps0", "amount"]))?,
        schema([field("x", DataType::BigInt, false)])
    );
    assert_eq!(
        schema_of(leaf(&["Region"]))?,
        schema([field("x", DataType::String, true)])
    );
    assert!(matches!(
        schema_of(leaf(&["ps0", "Amount"])),
        Err(IrErrorKind::UnboundLeaf { .. })
    ));
    Ok(())
}

#[test]
fn a_qualified_leaf_binds_to_the_field_named_by_its_parts_joined_and_no_name_is_cut(
) -> Result<(), IrErrorKind> {
    let qualified = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![(name("ssd0.region"), leaf(&["ps0", "region"]))],
    )?)
    .with_emit_form(EmitForm::new([3]))?;

    let bound = FilterNode::new(qualified.clone(), leaf(&["SSD0", "Region"]).eq(text("eu")))?;
    assert_eq!(bound.common.output_schema, qualified.common().output_schema);
    assert_eq!(
        FilterNode::new(qualified, leaf(&["region"]).eq(text("eu"))),
        Err(IrErrorKind::UnboundLeaf {
            leaf: NamedReference::unqualified("region").expect("a name path"),
            fields: vec!["ssd0.region".to_owned()],
        })
    );
    Ok(())
}

#[test]
fn with_emit_form_derives_the_output_schema_again_on_any_node() -> Result<(), IrErrorKind> {
    let reordered = schema([
        field("status", DataType::String, true),
        field("region", DataType::String, true),
    ]);
    let nodes: [PlanNode; 6] = [
        orders()?.into(),
        FilterNode::new(orders()?, leaf(&["ps0", "amount"]).gt(bigint(0)))?.into(),
        ProjectNode::new(orders()?, Vec::new())?.into(),
        SortNode::new(orders()?, vec![ascending(leaf(&["ps0", "region"]))])?.into(),
        FetchNode::new(orders()?, 0, Some(10))?.into(),
        TopNNode::new(
            orders()?,
            vec![ascending(leaf(&["ps0", "amount"]))],
            0,
            3,
            FetchMode::RowsOnly,
        )?
        .into(),
    ];
    for node in nodes {
        let emitted = node.with_emit_form(EmitForm::new([2, 0]))?;
        assert_eq!(emitted.common().output_schema, reordered);
        assert_eq!(emitted.common().emit_form, Some(EmitForm::new([2, 0])));
        let again = emitted.with_emit_form(EmitForm::new([1]))?;
        assert_eq!(
            again.common().output_schema,
            schema([field("amount", DataType::BigInt, false)])
        );
    }

    let aggregate = PlanNode::from(AggregateNode::new(
        orders()?,
        vec![(name("area"), leaf(&["ps0", "region"]))],
        vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
    )?)
    .with_emit_form(EmitForm::new([1, 0]))?;
    assert_eq!(
        aggregate.common().output_schema,
        schema([
            field("total", DataType::BigInt, true),
            field("area", DataType::String, true),
        ])
    );
    let again = aggregate.with_emit_form(EmitForm::new([0, 1]))?;
    assert_eq!(
        again.common().output_schema,
        schema([
            field("area", DataType::String, true),
            field("total", DataType::BigInt, true),
        ])
    );
    Ok(())
}

#[test]
fn an_emit_form_selects_and_orders_the_read_set_of_a_scan() -> Result<(), IrErrorKind> {
    let read_set = schema([
        field("id", DataType::BigInt, false),
        field("amount", DataType::BigInt, false),
        field("status", DataType::String, true),
    ]);
    let scan = ScanNode::new(
        TableScan::named_table(["lake".into(), "sales.prod".into(), "orders".into()], None),
        read_set.clone(),
    )?
    .with_filter(leaf(&["ps0", "status"]).eq(text("paid")))?;
    let scan = PlanNode::from(scan).with_emit_form(EmitForm::new([0, 1]))?;

    let PlanNode::Scan(scan) = &scan else {
        panic!("a scan");
    };
    assert_eq!(scan.schema, read_set);
    assert_eq!(
        scan.common.output_schema,
        schema([
            field("id", DataType::BigInt, false),
            field("amount", DataType::BigInt, false),
        ])
    );
    Ok(())
}

#[test]
fn an_output_schema_is_total() -> Result<(), IrErrorKind> {
    let aggregate = AggregateNode::new(
        orders()?,
        vec![(name("region"), leaf(&["ps0", "region"]))],
        vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
    )?;

    for SchemaColumn {
        name,
        data_type,
        nullable: _,
    } in &aggregate.common.output_schema.columns
    {
        assert!(!name.is_empty() && !name.contains(' '));
        assert_ne!(*data_type, DataType::Unknown);
    }
    assert_eq!(
        aggregate.common.output_schema,
        schema([
            field("region", DataType::String, true),
            field("total", DataType::BigInt, true),
        ])
    );
    Ok(())
}

#[test]
fn a_node_holds_its_trace_in_a_fixed_key_order_and_a_plan_in_anchor_pre_order(
) -> Result<(), IrErrorKind> {
    let orders_name = DataKindName::new("orders");
    let customers_name = DataKindName::new("customers");
    let scan = PlanNode::from(orders()?).with_trace([SemAnnotation::SemanticSource {
        semantic_source: orders_name.clone(),
        source_role: SourceRole::Dataset,
        originators: Vec::new(),
        physical_source: Some(name("orders")),
        cardinality: None,
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    }]);
    let root = PlanNode::from(ProjectNode::new(scan, vec![])?).with_trace([
        SemAnnotation::SemanticSource {
            semantic_source: customers_name.clone(),
            source_role: SourceRole::Joinset,
            originators: Vec::new(),
            physical_source: None,
            cardinality: None,
            delivery: None,
            answered_by: Some(orders_name.clone()),
            pruned: Vec::new(),
        },
        SemAnnotation::Rank {
            scope_root: orders_name.clone(),
            compositions: 0,
            granularity_fit: GranularityGroup::NotSelected,
            granularity: None,
        },
    ]);

    let plan = SemanticPlan::new(root, vec![]);

    let trace: Vec<(Anchor, Vec<SemAnnotation>)> = [&plan.root, plan.root.children()[0]]
        .iter()
        .map(|node| (node.common().anchor, node.common().semantic_meta.clone()))
        .collect();
    assert_eq!(
        trace,
        [
            (
                Anchor(1),
                vec![
                    SemAnnotation::Rank {
                        scope_root: orders_name.clone(),
                        compositions: 0,
                        granularity_fit: GranularityGroup::NotSelected,
                        granularity: None,
                    },
                    SemAnnotation::SemanticSource {
                        semantic_source: customers_name,
                        source_role: SourceRole::Joinset,
                        originators: Vec::new(),
                        physical_source: None,
                        cardinality: None,
                        delivery: None,
                        answered_by: Some(orders_name.clone()),
                        pruned: Vec::new(),
                    },
                ]
            ),
            (
                Anchor(2),
                vec![SemAnnotation::SemanticSource {
                    semantic_source: orders_name,
                    source_role: SourceRole::Dataset,
                    originators: Vec::new(),
                    physical_source: Some(name("orders")),
                    cardinality: None,
                    delivery: None,
                    answered_by: None,
                    pruned: Vec::new(),
                }]
            ),
        ]
    );
    Ok(())
}

#[test]
fn a_virtual_table_reads_its_rows_and_with_none_is_the_empty_read_typed_by_its_schema(
) -> Result<(), IrErrorKind> {
    let fields = schema([
        field("id", DataType::BigInt, false),
        field("region", DataType::String, true),
    ]);
    let rows = VirtualTable::new(vec![
        vec![bigint(1), text("eu")],
        vec![bigint(2), text("us")],
    ]);
    let empty = VirtualTable::new(Vec::new());

    for reader in [rows, empty] {
        let scan = ScanNode::new(reader.clone(), fields.clone())?;
        assert_eq!(scan.reader, Reader::VirtualTable(reader));
        assert_eq!(scan.common.output_schema, fields);
    }
    Ok(())
}

#[test]
fn a_files_scan_reads_its_listing_or_the_folder_of_its_base_path() -> Result<(), IrErrorKind> {
    let listing = FilesScan::new(
        Some(FileFormat::Parquet),
        Some("orders/*.parquet".into()),
        "s3://lake/orders/",
        vec![
            "s3://lake/orders/a.parquet".into(),
            "s3://lake/orders/b.parquet".into(),
        ],
    );
    let amount = schema([field("amount", DataType::BigInt, false)]);
    let listed = ScanNode::new(listing, amount.clone())?;
    let Reader::FilesScan(files) = &listed.reader else {
        panic!("a files scan");
    };
    assert_eq!(files.format_options, None);
    assert_eq!(listed.common.output_schema, amount);

    let folder = FilesScan::new(
        Some(FileFormat::Parquet),
        None,
        "s3://lake/orders/",
        Vec::new(),
    );
    let partitioned = schema([
        field("year", DataType::Integer, true),
        field("amount", DataType::BigInt, false),
    ]);
    let read = ScanNode::new(folder, partitioned.clone())?;
    let Reader::FilesScan(files) = &read.reader else {
        panic!("a files scan");
    };
    assert!(files.paths.is_empty() && files.pattern.is_none());
    assert_eq!(files.base_path, "s3://lake/orders/");
    assert_eq!(read.common.output_schema, partitioned);
    Ok(())
}

#[test]
fn a_named_table_holds_three_names_the_first_empty_for_no_catalog() -> Result<(), IrErrorKind> {
    let scan = orders()?;

    let Reader::TableScan(TableScan::NamedTable { names, format, .. }) = &scan.reader else {
        panic!("a named table");
    };
    assert_eq!(names, &["", "sales", "orders"]);
    assert_eq!(*format, None);
    assert_eq!(
        scan.common.output_schema,
        schema([
            field("region", DataType::String, true),
            field("amount", DataType::BigInt, false),
            field("status", DataType::String, true),
        ])
    );
    Ok(())
}

#[test]
fn a_filter_outputs_its_input_fields() -> Result<(), IrErrorKind> {
    let input = orders()?;
    let fields = input.common.output_schema.clone();

    let filter = FilterNode::new(input, leaf(&["ps0", "status"]).eq(text("paid")))?;

    assert_eq!(filter.common.output_schema, fields);
    Ok(())
}

#[test]
fn an_aggregate_outputs_its_groupings_then_its_aggregates_each_with_its_filter_clause(
) -> Result<(), IrErrorKind> {
    let paid = leaf(&["ps0", "status"]).eq(text("paid"));
    let input = ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "orders".into()], None),
        schema([
            field("region", DataType::String, true),
            field("amount", DataType::Integer, false),
            field("status", DataType::String, true),
        ]),
    )?;

    let aggregate = AggregateNode::new(
        input,
        vec![(name("region"), leaf(&["ps0", "region"]))],
        vec![
            (name("total"), sum(leaf(&["ps0", "amount"]), None)),
            (
                name("paid"),
                sum(leaf(&["ps0", "amount"]), Some(paid.clone())),
            ),
        ],
    )?;

    assert_eq!(
        aggregate.common.output_schema,
        schema([
            field("region", DataType::String, true),
            field("total", DataType::BigInt, true),
            field("paid", DataType::BigInt, true),
        ])
    );
    assert_eq!(aggregate.groupings, [leaf(&["ps0", "region"])]);
    assert_eq!(aggregate.aggregates[1].1.filter(), Some(&paid));
    Ok(())
}

#[test]
fn distinct_rows_are_an_aggregate_with_no_aggregate() -> Result<(), IrErrorKind> {
    let distinct = AggregateNode::new(
        orders()?,
        vec![
            (name("region"), leaf(&["ps0", "region"])),
            (name("status"), leaf(&["ps0", "status"])),
        ],
        Vec::new(),
    )?;

    assert!(distinct.aggregates.is_empty());
    assert_eq!(
        distinct.common.output_schema,
        schema([
            field("region", DataType::String, true),
            field("status", DataType::String, true),
        ])
    );
    Ok(())
}

#[test]
fn a_sort_orders_by_keys_with_a_direction_and_a_null_order() -> Result<(), IrErrorKind> {
    let input = orders()?;
    let fields = input.common.output_schema.clone();
    let keys = vec![
        key(
            leaf(&["ps0", "region"]),
            SortDir::Asc {
                nulls: NullOrdering::Last,
            },
        ),
        key(
            leaf(&["ps0", "amount"]),
            SortDir::Desc {
                nulls: NullOrdering::First,
            },
        ),
    ];

    let sort = SortNode::new(input, keys.clone())?;

    assert_eq!(sort.order_by, keys);
    assert_eq!(sort.common.output_schema, fields);
    Ok(())
}

#[test]
fn a_fetch_of_offset_zero_and_no_limit_reads_every_row() -> Result<(), IrErrorKind> {
    let input = orders()?;
    let fields = input.common.output_schema.clone();

    let fetch = FetchNode::new(input, 0, None)?;

    assert_eq!((fetch.offset, fetch.limit), (0, None));
    assert_eq!(fetch.common.output_schema, fields);
    let paged = FetchNode::new(orders()?, 20, Some(10))?;
    assert_eq!((paged.offset, paged.limit), (20, Some(10)));
    Ok(())
}

#[test]
fn a_top_n_keeps_its_bounds_and_fetch_mode_and_outputs_its_input_fields() -> Result<(), IrErrorKind>
{
    let input = orders()?;
    let fields = input.common.output_schema.clone();

    let top = TopNNode::new(
        input,
        vec![ascending(leaf(&["ps0", "amount"]))],
        0,
        3,
        FetchMode::WithTies,
    )?;

    assert_eq!(
        (top.offset, top.limit, top.fetch_mode),
        (0, 3, FetchMode::WithTies)
    );
    assert_eq!(top.common.output_schema, fields);
    Ok(())
}

#[test]
fn a_left_join_makes_the_right_side_nullable_and_binds_on_against_left_then_right(
) -> Result<(), IrErrorKind> {
    let on = leaf(&["ssd0", "region"]).eq(leaf(&["SSD1", "Region"]));

    let join = orders_with_customers(JoinType::Left, on, None)?;

    assert_eq!(
        join.common.output_schema,
        schema([
            field("ssd0.region", DataType::String, true),
            field("ssd0.amount", DataType::BigInt, false),
            field("ssd0.status", DataType::String, true),
            field("ssd1.region", DataType::String, true),
            field("ssd1.name", DataType::String, true),
        ])
    );
    Ok(())
}

#[test]
fn a_joins_natural_output_is_left_then_right_and_each_outer_side_is_nullable(
) -> Result<(), IrErrorKind> {
    let customers_twice = |join_type| -> Result<Vec<(String, bool)>, IrErrorKind> {
        let join = JoinNode::new(
            projection("ssd0", customers()?)?,
            projection("ssd1", customers()?)?,
            join_type,
            same_region(),
            None,
        )?;
        Ok((join.common.output_schema.columns.into_iter())
            .map(|field| (field.name, field.nullable))
            .collect())
    };

    for (join_type, left, right) in [
        (JoinType::Inner, false, false),
        (JoinType::Left, false, true),
        (JoinType::Right, true, false),
        (JoinType::Full, true, true),
    ] {
        let expected = [
            ("ssd0.region", left),
            ("ssd0.name", left),
            ("ssd1.region", right),
            ("ssd1.name", right),
        ]
        .map(|(name, nullable)| (name.to_owned(), nullable));
        assert_eq!(customers_twice(join_type)?, expected, "{join_type:?}");
    }
    Ok(())
}

#[test]
fn a_post_join_filter_binds_against_the_natural_output_before_any_emit_form(
) -> Result<(), IrErrorKind> {
    let join = orders_with_customers(
        JoinType::Left,
        same_region(),
        Some(leaf(&["ssd1", "name"]).is_null()),
    )?;

    let emitted = PlanNode::from(join).with_emit_form(EmitForm::new([0, 1]))?;

    assert_eq!(
        emitted.common().output_schema,
        schema([
            field("ssd0.region", DataType::String, true),
            field("ssd0.amount", DataType::BigInt, false),
        ])
    );
    let inputs = emitted.children().into_iter().cloned().collect();
    assert_eq!(emitted.clone().with_new_children(inputs), Ok(emitted));
    Ok(())
}

#[test]
fn a_join_takes_an_emit_form_and_a_trace_as_every_node_does() -> Result<(), IrErrorKind> {
    let join = orders_with_customers(JoinType::Full, same_region(), None)?;
    let traced = SemAnnotation::SemanticSource {
        semantic_source: DataKindName::new("orders_with_customers"),
        source_role: SourceRole::Joinset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: Some(Cardinality::ManyToOne),
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    };

    let join = PlanNode::from(join)
        .with_emit_form(EmitForm::new([4, 0]))?
        .with_trace([traced.clone()]);

    assert_eq!(
        join.common().output_schema,
        schema([
            field("ssd1.name", DataType::String, true),
            field("ssd0.region", DataType::String, true),
        ])
    );
    assert_eq!(join.common().emit_form, Some(EmitForm::new([4, 0])));
    assert_eq!(join.common().semantic_meta, [traced]);
    Ok(())
}

#[test]
fn a_join_condition_or_a_post_join_filter_that_binds_to_no_field_or_is_no_condition_is_refused_as_validate_refuses_it(
) -> Result<(), IrErrorKind> {
    let natural = [
        "ssd0.region",
        "ssd0.amount",
        "ssd0.status",
        "ssd1.region",
        "ssd1.name",
    ];
    let faults = [
        (
            leaf(&["ssd0", "amount"]) + bigint(1),
            None,
            IrErrorKind::NotACondition {
                got: DataType::BigInt,
            },
        ),
        (
            same_region(),
            Some(leaf(&["ssd1", "name"])),
            IrErrorKind::NotACondition {
                got: DataType::String,
            },
        ),
        (
            leaf(&["ssd2", "region"]).eq(leaf(&["ssd1", "region"])),
            None,
            unbound(&["ssd2", "region"], &natural),
        ),
        (
            same_region(),
            Some(leaf(&["ps0", "name"]).is_null()),
            unbound(&["ps0", "name"], &natural),
        ),
    ];

    for (on, post_join_filter, kind) in faults {
        assert_eq!(
            orders_with_customers(JoinType::Inner, on.clone(), post_join_filter.clone()),
            Err(kind.clone())
        );
        let mut rewritten = orders_with_customers(JoinType::Inner, same_region(), None)?;
        (rewritten.on, rewritten.post_join_filter) = (on, post_join_filter);
        assert_eq!(refusal(&plan(rewritten)), Some((kind, at(1, "JoinNode"))));
    }
    Ok(())
}

#[test]
fn a_cross_outputs_left_then_right_and_changes_no_nullability() -> Result<(), IrErrorKind> {
    let cross = PlanNode::from(orders_by_customers()?);

    assert_eq!(
        cross.common().output_schema,
        schema([
            field("ssd0.region", DataType::String, true),
            field("ssd0.amount", DataType::BigInt, false),
            field("ssd0.status", DataType::String, true),
            field("ssd1.region", DataType::String, false),
            field("ssd1.name", DataType::String, false),
        ])
    );
    let emitted = cross.with_emit_form(EmitForm::new([4, 0]))?;
    assert_eq!(
        emitted.common().output_schema,
        schema([
            field("ssd1.name", DataType::String, false),
            field("ssd0.region", DataType::String, true),
        ])
    );
    let inputs = emitted.children().into_iter().cloned().collect();
    assert_eq!(emitted.clone().with_new_children(inputs), Ok(emitted));
    Ok(())
}

#[test]
fn a_filter_above_a_cross_binds_a_filter_that_reads_one_field_of_each_side(
) -> Result<(), IrErrorKind> {
    let fields = orders_by_customers()?.common.output_schema;

    let filter = FilterNode::new(orders_by_customers()?, same_region())?;

    assert_eq!(filter.common.output_schema, fields);
    assert_eq!(
        FilterNode::new(
            orders_by_customers()?,
            leaf(&["ps0", "region"]).eq(leaf(&["ssd1", "region"]))
        ),
        Err(IrErrorKind::UnboundLeaf {
            leaf: NamedReference::qualified("ps0", "region").expect("a name path"),
            fields: (fields.columns.into_iter())
                .map(|field| field.name)
                .collect(),
        })
    );
    Ok(())
}

#[test]
fn a_join_holds_its_cardinality_and_a_cross_its_own() -> Result<(), IrErrorKind> {
    let traced = |cardinality| SemAnnotation::SemanticSource {
        semantic_source: DataKindName::new("orders_with_customers"),
        source_role: SourceRole::Joinset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: Some(cardinality),
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    };
    let join = PlanNode::from(orders_with_customers(JoinType::Left, same_region(), None)?)
        .with_trace([traced(Cardinality::ManyToOne)]);
    let cross = |resolved| -> Result<PlanNode, IrErrorKind> {
        Ok(PlanNode::from(orders_by_customers()?).with_trace([traced(resolved)]))
    };
    let joins = [
        join,
        cross(Cardinality::OneToMany)?,
        cross(Cardinality::ManyToMany)?,
    ];

    let traces = joins.map(|node| node.common().semantic_meta.clone());
    assert_eq!(
        traces,
        [
            vec![traced(Cardinality::ManyToOne)],
            vec![traced(Cardinality::OneToMany)],
            vec![traced(Cardinality::ManyToMany)],
        ]
    );
    Ok(())
}

fn branch(read: ScanNode, fields: Vec<(&str, PhysicalExpr)>) -> Result<PlanNode, IrErrorKind> {
    let fields = (fields.into_iter())
        .map(|(field, expr)| (name(&format!("ssu2.{field}")), expr))
        .collect();
    projected(read, fields)
}

fn amounts() -> Result<PlanNode, IrErrorKind> {
    branch(
        orders()?,
        vec![
            ("region", leaf(&["ps0", "region"])),
            ("amount", leaf(&["ps0", "amount"])),
        ],
    )
}

fn assert_refused(branches: Vec<PlanNode>, kind: IrErrorKind) -> Result<(), IrErrorKind> {
    assert_eq!(
        MultisetNode::new(branches.clone(), MultisetOp::UnionAll),
        Err(kind.clone())
    );
    let mut rewritten = MultisetNode::new(vec![amounts()?, amounts()?], MultisetOp::UnionAll)?;
    rewritten.inputs = branches;
    assert_eq!(
        refusal(&plan(rewritten)),
        Some((kind, at(1, "MultisetNode")))
    );
    Ok(())
}

#[test]
fn a_branch_whose_field_has_another_type_than_the_first_branchs_is_refused_as_validate_refuses_it(
) -> Result<(), IrErrorKind> {
    let status_as_amount = branch(
        orders()?,
        vec![
            ("region", leaf(&["ps0", "region"])),
            ("amount", leaf(&["ps0", "status"])),
        ],
    )?;

    assert_refused(
        vec![amounts()?, status_as_amount],
        IrErrorKind::BranchFieldMismatch {
            branch: 1,
            index: 1,
        },
    )
}

#[test]
fn a_branch_with_another_name_at_a_position_or_another_number_of_fields_is_refused_as_validate_refuses_it(
) -> Result<(), IrErrorKind> {
    let region = || ("region", leaf(&["ps0", "region"]));
    let amount = |field| (field, leaf(&["ps0", "amount"]));
    let faults = [
        (vec![region(), amount("total")], 1),
        (vec![region()], 1),
        (
            vec![
                region(),
                amount("amount"),
                ("status", leaf(&["ps0", "status"])),
            ],
            2,
        ),
    ];

    for (fields, index) in faults {
        let third = branch(orders()?, fields)?;
        assert_refused(
            vec![amounts()?, amounts()?, third],
            IrErrorKind::BranchFieldMismatch { branch: 2, index },
        )?;
    }
    Ok(())
}

#[test]
fn a_union_of_fewer_than_two_branches_is_refused_as_validate_refuses_it() -> Result<(), IrErrorKind>
{
    for branches in [vec![amounts()?], Vec::new()] {
        let got = branches.len();
        assert_refused(branches, IrErrorKind::MultisetArity { got })?;
    }
    Ok(())
}

#[test]
fn a_union_outputs_the_first_branchs_names_and_types_each_nullable_where_any_branchs_is(
) -> Result<(), IrErrorKind> {
    let filled = branch(
        customers()?,
        vec![
            ("Region", leaf(&["ps0", "region"])),
            ("Amount", Expr::Literal(Literal::Null(DataType::BigInt))),
        ],
    )?;

    let union = MultisetNode::new(vec![amounts()?, filled], MultisetOp::UnionAll)?;

    assert_eq!(
        union.common.output_schema,
        schema([
            field("ssu2.region", DataType::String, true),
            field("ssu2.amount", DataType::BigInt, true),
        ])
    );
    let mut rewritten = union;
    rewritten.common.output_schema.columns[1].nullable = false;
    assert_eq!(
        refusal(&plan(rewritten)),
        Some((
            IrErrorKind::OutputSchemaMismatch { index: 1 },
            at(1, "MultisetNode")
        ))
    );
    Ok(())
}

#[test]
fn a_leaf_that_binds_to_no_field_is_refused() -> Result<(), IrErrorKind> {
    assert_eq!(
        FilterNode::new(orders()?, leaf(&["ps0", "price"]).gt(bigint(0))),
        Err(IrErrorKind::UnboundLeaf {
            leaf: NamedReference::qualified("ps0", "price").expect("a name path"),
            fields: vec!["region".into(), "amount".into(), "status".into()],
        })
    );
    Ok(())
}

#[test]
fn a_type_that_typing_cannot_derive_or_derives_as_unknown_is_refused() -> Result<(), IrErrorKind> {
    let refused = ProjectNode::new(
        orders()?,
        vec![(name("x"), leaf(&["ps0", "amount"]) + text("one"))],
    );
    assert!(matches!(refused, Err(IrErrorKind::TypeRefused { .. })));

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
    let no_type = || leaf(&["ps0", "score"]) * leaf(&["ps0", "price"]);
    let invalid = |field: &str, data_type| {
        Some(IrErrorKind::InvalidType {
            field: field.to_owned(),
            data_type,
        })
    };
    assert_eq!(
        ProjectNode::new(
            scores.clone(),
            vec![
                (name("score"), leaf(&["ps0", "score"])),
                (name("x"), no_type()),
            ],
        )
        .err(),
        invalid("x", DataType::Unknown)
    );
    assert_eq!(
        AggregateNode::new(scores.clone(), vec![(name("g"), no_type())], Vec::new()).err(),
        invalid("g", DataType::Unknown)
    );
    assert_eq!(
        SortNode::new(
            scores,
            vec![ascending(leaf(&["ps0", "score"])), ascending(no_type())]
        ),
        Err(IrErrorKind::InvalidKeyType {
            key: 1,
            data_type: DataType::Unknown
        })
    );

    let unknown_element = DataType::Array(Box::new(DataType::Unknown));
    let scale_above_precision = DataType::Decimal {
        precision: 2,
        scale: 5,
    };
    for data_type in [DataType::Unknown, unknown_element, scale_above_precision] {
        assert_eq!(
            ScanNode::new(
                VirtualTable::new(Vec::new()),
                schema([
                    field("id", DataType::BigInt, false),
                    field("x", data_type.clone(), true),
                ])
            )
            .err(),
            invalid("x", data_type)
        );
    }
    Ok(())
}

#[test]
fn a_filter_that_is_not_a_condition_is_refused() -> Result<(), IrErrorKind> {
    assert_eq!(
        FilterNode::new(orders()?, leaf(&["ps0", "amount"]) + bigint(1)),
        Err(IrErrorKind::NotACondition {
            got: DataType::BigInt
        })
    );
    assert_eq!(
        orders()?.with_filter(leaf(&["ps0", "region"])),
        Err(IrErrorKind::NotACondition {
            got: DataType::String
        })
    );
    let no_type = || Expr::Literal(Literal::Null(DataType::Unknown));
    assert_eq!(
        FilterNode::new(orders()?, no_type()),
        Err(IrErrorKind::NotACondition {
            got: DataType::Unknown
        })
    );
    for (filter, got) in [
        (no_type(), DataType::Unknown),
        (leaf(&["ps0", "amount"]), DataType::BigInt),
    ] {
        assert_eq!(
            AggregateNode::new(
                orders()?,
                Vec::new(),
                vec![(name("total"), sum(leaf(&["ps0", "amount"]), Some(filter)))],
            ),
            Err(IrErrorKind::NotACondition { got })
        );
    }
    Ok(())
}

#[test]
fn an_emit_form_index_out_of_range_is_refused() -> Result<(), IrErrorKind> {
    assert_eq!(
        PlanNode::from(orders()?).with_emit_form(EmitForm::new([0, 3])),
        Err(IrErrorKind::EmitFormIndexOutOfRange {
            index: 3,
            fields: 3
        })
    );
    Ok(())
}

#[test]
fn an_emit_form_that_leaves_out_a_computed_field_is_refused() -> Result<(), IrErrorKind> {
    let projection = ProjectNode::new(
        orders()?,
        vec![(name("doubled"), leaf(&["ps0", "amount"]) * bigint(2))],
    )?;

    assert_eq!(
        PlanNode::from(projection).with_emit_form(EmitForm::new([0])),
        Err(IrErrorKind::EmitFormLeavesOutComputedField { index: 3 })
    );
    Ok(())
}

#[test]
fn an_aggregates_emit_form_that_leaves_out_its_grouping_is_refused_and_a_reordered_one_passes(
) -> Result<(), IrErrorKind> {
    let aggregate = PlanNode::from(AggregateNode::new(
        orders()?,
        vec![(name("region"), leaf(&["ps0", "region"]))],
        vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
    )?);

    assert_eq!(
        aggregate.clone().with_emit_form(EmitForm::new([1])),
        Err(IrErrorKind::EmitFormLeavesOutComputedField { index: 0 })
    );
    let reordered = aggregate.clone().with_emit_form(EmitForm::new([1, 0]))?;
    assert_eq!(
        reordered.common().output_schema,
        schema([
            field("total", DataType::BigInt, true),
            field("region", DataType::String, true),
        ])
    );
    let above = PlanNode::from(ProjectNode::new(aggregate, Vec::new())?);
    assert_eq!(
        above
            .with_emit_form(EmitForm::new([1]))?
            .common()
            .output_schema,
        schema([field("total", DataType::BigInt, true)])
    );
    Ok(())
}

#[test]
fn a_repeated_name_case_ignored_is_refused_but_in_a_physical_read() -> Result<(), IrErrorKind> {
    let repeated = |name: &str| IrErrorKind::RepeatedName {
        name: name.to_owned(),
    };
    assert_eq!(
        ProjectNode::new(
            orders()?,
            vec![
                (name("total"), leaf(&["ps0", "amount"])),
                (name("Total"), leaf(&["ps0", "amount"])),
            ],
        ),
        Err(repeated("Total"))
    );
    assert_eq!(
        AggregateNode::new(
            orders()?,
            vec![(name("region"), leaf(&["ps0", "region"]))],
            vec![(name("REGION"), sum(leaf(&["ps0", "amount"]), None))],
        ),
        Err(repeated("REGION"))
    );
    let filter = FilterNode::new(orders()?, leaf(&["ps0", "amount"]).gt(bigint(0)))?;
    assert_eq!(
        PlanNode::from(filter).with_emit_form(EmitForm::new([0, 0])),
        Err(repeated("region"))
    );
    let read = PlanNode::from(orders()?).with_emit_form(EmitForm::new([0, 0]))?;
    assert_eq!(
        read.common().output_schema,
        schema([
            field("region", DataType::String, true),
            field("region", DataType::String, true),
        ])
    );
    Ok(())
}

#[test]
fn a_negative_fetch_bound_is_refused() -> Result<(), IrErrorKind> {
    let negative = |bound| Some(IrErrorKind::NegativeBound { bound });
    assert_eq!(FetchNode::new(orders()?, -1, None).err(), negative(-1));
    assert_eq!(FetchNode::new(orders()?, 0, Some(-5)).err(), negative(-5));
    assert_eq!(
        TopNNode::new(orders()?, Vec::new(), 0, -2, FetchMode::RowsOnly).err(),
        negative(-2)
    );
    Ok(())
}

#[test]
fn a_sort_key_with_an_unspecified_null_order_is_refused() -> Result<(), IrErrorKind> {
    let unspecified = key(
        leaf(&["ps0", "amount"]),
        SortDir::Desc {
            nulls: NullOrdering::Unspecified,
        },
    );
    let keys = vec![ascending(leaf(&["ps0", "region"])), unspecified];
    assert_eq!(
        SortNode::new(orders()?, keys.clone()),
        Err(IrErrorKind::UnspecifiedNullOrdering { key: 1 })
    );
    assert_eq!(
        TopNNode::new(orders()?, keys, 0, 1, FetchMode::RowsOnly),
        Err(IrErrorKind::UnspecifiedNullOrdering { key: 1 })
    );
    Ok(())
}

#[test]
fn an_aggregates_order_key_with_an_unspecified_null_order_is_refused_as_a_sort_key_is(
) -> Result<(), IrErrorKind> {
    let regions = |nulls| {
        let dir = SortDir::Asc { nulls };
        string_agg(leaf(&["ps0", "region"]), key(leaf(&["ps0", "status"]), dir))
    };
    assert_eq!(
        AggregateNode::new(
            orders()?,
            Vec::new(),
            vec![(name("regions"), regions(NullOrdering::Unspecified))]
        )
        .err(),
        Some(IrErrorKind::UnspecifiedAggregateNullOrdering {
            function: "string_agg",
            key: 0
        })
    );
    let ordered = AggregateNode::new(
        orders()?,
        Vec::new(),
        vec![(name("regions"), regions(NullOrdering::Last))],
    );
    assert!(ordered.is_ok());
    Ok(())
}

#[test]
fn replacing_a_nodes_input_rebuilds_it_and_derives_its_output_schema_again(
) -> Result<(), IrErrorKind> {
    let nullable_amounts = ScanNode::new(
        VirtualTable::new(Vec::new()),
        schema([
            field("region", DataType::String, false),
            field("amount", DataType::BigInt, true),
            field("status", DataType::String, true),
        ]),
    )?;
    let projection = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![(name("doubled"), leaf(&["ps0", "amount"]) * bigint(2))],
    )?)
    .with_emit_form(EmitForm::new([3, 0]))?
    .with_trace([SemAnnotation::Rank {
        scope_root: DataKindName::new("orders"),
        compositions: 0,
        granularity_fit: GranularityGroup::NotSelected,
        granularity: None,
    }]);
    let plan = SemanticPlan::new(projection, Vec::new());

    let rebuilt = (plan.root.clone())
        .with_new_children(vec![nullable_amounts.clone().into()])
        .expect("the projection binds over the new input");

    assert_eq!(
        rebuilt.common().output_schema,
        schema([
            field("doubled", DataType::BigInt, true),
            field("region", DataType::String, false),
        ])
    );
    assert_eq!(rebuilt.common().anchor, Anchor(1));
    assert_eq!(
        rebuilt.common().semantic_meta,
        [SemAnnotation::Rank {
            scope_root: DataKindName::new("orders"),
            compositions: 0,
            granularity_fit: GranularityGroup::NotSelected,
            granularity: None,
        }]
    );
    assert_eq!(
        plan.root.clone().transform(&mut |node| Ok(node)),
        Ok(plan.root.clone())
    );

    let aggregate = PlanNode::from(AggregateNode::new(
        orders()?,
        vec![(name("area"), leaf(&["ps0", "region"]))],
        vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
    )?)
    .with_emit_form(EmitForm::new([1, 0]))?;
    let rebuilt = aggregate
        .with_new_children(vec![nullable_amounts.into()])
        .expect("the aggregate binds over the new input");
    assert_eq!(
        rebuilt.common().output_schema,
        schema([
            field("total", DataType::BigInt, true),
            field("area", DataType::String, false),
        ])
    );

    let scan = PlanNode::from(orders()?);
    assert_eq!(scan.clone().with_new_children(Vec::new()), Ok(scan.clone()));
    assert_eq!(
        scan.clone().with_new_children(vec![scan]),
        Err(ValidateError::ChildCountMismatch {
            expected: 0,
            got: 1
        })
    );

    let no_amount = ScanNode::new(
        VirtualTable::new(Vec::new()),
        schema([field("region", DataType::String, true)]),
    )?;
    assert!(matches!(
        plan.root.clone().with_new_children(vec![no_amount.into()]),
        Err(ValidateError::NodeRefused { .. })
    ));
    assert_eq!(
        plan.root.with_new_children(Vec::new()),
        Err(ValidateError::ChildCountMismatch {
            expected: 1,
            got: 0
        })
    );
    Ok(())
}

#[test]
fn replacing_a_nodes_input_by_one_of_another_width_keeps_the_names_of_its_computed_fields(
) -> Result<(), IrErrorKind> {
    let read = |fields| ScanNode::new(VirtualTable::new(Vec::new()), schema(fields));
    let (region, amount) = (
        || field("region", DataType::String, true),
        || field("amount", DataType::BigInt, false),
    );
    let narrower = read(vec![region(), amount()])?;
    let wider = read(vec![
        region(),
        amount(),
        field("status", DataType::String, true),
        field("channel", DataType::String, true),
    ])?;
    let output = |node: &PlanNode, input: &ScanNode| {
        (node.clone().with_new_children(vec![input.clone().into()]))
            .map(|rebuilt| rebuilt.common().output_schema.clone())
    };

    let doubled = PlanNode::from(ProjectNode::new(
        orders()?,
        vec![(name("doubled"), leaf(&["ps0", "amount"]) * bigint(2))],
    )?);
    for input in [&narrower, &wider] {
        let mut fields = input.schema.columns.clone();
        fields.push(field("doubled", DataType::BigInt, false));
        assert_eq!(output(&doubled, input), Ok(schema(fields)));
    }

    let aggregate = PlanNode::from(AggregateNode::new(
        orders()?,
        vec![(name("area"), leaf(&["ps0", "region"]))],
        vec![(name("total"), sum(leaf(&["ps0", "amount"]), None))],
    )?)
    .with_emit_form(EmitForm::new([1, 0]))?;
    for input in [&narrower, &wider] {
        assert_eq!(
            output(&aggregate, input),
            Ok(schema([
                field("total", DataType::BigInt, true),
                field("area", DataType::String, true),
            ]))
        );
    }

    let emitted = doubled.with_emit_form(EmitForm::new([3, 0]))?;
    assert_eq!(
        output(&emitted, &narrower),
        Err(ValidateError::from(IrErrorKind::EmitFormIndexOutOfRange {
            index: 3,
            fields: 3
        }))
    );
    assert_eq!(
        output(&emitted, &wider),
        Err(ValidateError::from(
            IrErrorKind::EmitFormLeavesOutComputedField { index: 4 }
        ))
    );
    Ok(())
}
