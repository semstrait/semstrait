mod support;

use std::error::Error;

use prost::Message;
use semstrait_adapter::{Extensions, FunctionName, InvalidRewrite, SubstraitRules};
use semstrait_catalog::FileFormat;
use semstrait_ir::{
    EmitForm, FetchMode, FilesScan, JoinNode, JoinType, MultisetNode, MultisetOp, Parameter,
    PhysicalExpr, PlanNode, ProjectNode, ScanNode, Schema, TopNNode, VirtualTable,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{AggregateAllowed, ExprOperationExt, Literal, LiteralError, ScalarCall};
use substrait::proto::expression::literal::interval_day_to_second::PrecisionMode;
use substrait::proto::expression::literal::{
    Decimal, IntervalCompound, IntervalDayToSecond, IntervalYearToMonth, LiteralType,
    PrecisionTime, PrecisionTimestamp,
};
use substrait::proto::expression::{self, RexType, ScalarFunction};
use substrait::proto::extensions::simple_extension_declaration::MappingType;
use substrait::proto::function_argument::ArgType;
use substrait::proto::r#type::{self, Kind, Nullability};
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat as ItemFormat, OrcReadOptions, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{self, LocalFiles, ReadType};
use substrait::proto::rel_common::{Direct, Emit, EmitKind};
use substrait::proto::{
    join_rel, rel, set_rel, Expression, FunctionArgument, JoinRel, ReadRel, Rel, SetRel, Type,
};

use support::{
    aggregate_rel, column, declared_function, descending, field, hand_built,
    has_advanced_extension, leaf, literal, moments, name, one_expression_of_each_kind, orders,
    orders_files, orders_schema, paid_by_region, produce, project_rel, projected, read, read_type,
    refusal, relations, root, root_input, scalar, schema, text, Faithful, ORDERS_FOLDER,
    ORDERS_PATHS,
};

fn items(read_type: &ReadType) -> Vec<(Option<PathType>, Option<ItemFormat>)> {
    let ReadType::LocalFiles(LocalFiles { items, .. }) = read_type else {
        panic!("a LocalFiles read");
    };
    (items.iter())
        .map(|item| (item.path_type.clone(), item.file_format.clone()))
        .collect()
}

#[test]
fn a_files_scan_of_two_paths_is_one_local_files_with_two_uri_file_items(
) -> Result<(), Box<dyn Error>> {
    let plan = produce(orders_files(Some(FileFormat::Parquet))?, &Faithful)?;

    assert_eq!(
        items(read_type(&plan)),
        ORDERS_PATHS.map(|path| (
            Some(PathType::UriFile(path.into())),
            Some(ItemFormat::Parquet(ParquetReadOptions {}))
        ))
    );
    Ok(())
}

#[test]
fn each_item_carries_the_format_where_substrait_has_a_slot_for_it() -> Result<(), Box<dyn Error>> {
    let cases = [
        (
            Some(FileFormat::Parquet),
            Some(ItemFormat::Parquet(ParquetReadOptions {})),
        ),
        (
            Some(FileFormat::Orc),
            Some(ItemFormat::Orc(OrcReadOptions {})),
        ),
        (Some(FileFormat::Csv), None),
        (Some(FileFormat::Avro), None),
        (Some(FileFormat::Json), None),
        (Some(FileFormat::Delta), None),
        (Some(FileFormat::Iceberg), None),
        (None, None),
    ];

    for (format, expected) in cases {
        let plan = produce(orders_files(format)?, &Faithful)?;

        let formats: Vec<_> = (items(read_type(&plan)).into_iter())
            .map(|(_, format)| format)
            .collect();
        assert_eq!(formats, [expected.clone(), expected], "{format:?}");
    }
    Ok(())
}

#[test]
fn a_folder_read_is_one_uri_folder_item_of_its_base_path() -> Result<(), Box<dyn Error>> {
    let folder = FilesScan::new(Some(FileFormat::Parquet), None, ORDERS_FOLDER, Vec::new());

    let plan = produce(ScanNode::new(folder, orders_schema())?, &Faithful)?;

    assert_eq!(
        items(read_type(&plan)),
        [(
            Some(PathType::UriFolder(ORDERS_FOLDER.into())),
            Some(ItemFormat::Parquet(ParquetReadOptions {}))
        )]
    );
    Ok(())
}

#[test]
fn a_named_table_keeps_its_three_names_the_empty_catalog_included() -> Result<(), Box<dyn Error>> {
    let plan = produce(orders()?, &Faithful)?;

    let ReadType::NamedTable(table) = read_type(&plan) else {
        panic!("a NamedTable read");
    };
    assert_eq!(table.names, ["", "sales", "orders"]);
    Ok(())
}

#[test]
fn the_empty_read_is_a_virtual_table_with_no_rows() -> Result<(), Box<dyn Error>> {
    let empty = ScanNode::new(VirtualTable::new(Vec::new()), orders_schema())?;

    let plan = produce(empty, &Faithful)?;

    assert_eq!(
        read_type(&plan),
        &ReadType::VirtualTable(read_rel::VirtualTable::default())
    );
    Ok(())
}

#[test]
fn the_executed_plan_has_no_advanced_extension() -> Result<(), Box<dyn Error>> {
    let folder = FilesScan::new(None, None, ORDERS_FOLDER, Vec::new());
    let reads = [
        orders()?,
        orders_files(Some(FileFormat::Parquet))?,
        ScanNode::new(folder, orders_schema())?,
        ScanNode::new(VirtualTable::new(Vec::new()), orders_schema())?,
    ];

    for scan in reads {
        let plan = produce(scan, &Faithful)?;

        let read = read(root_input(&plan));
        assert_eq!(plan.advanced_extensions, None);
        assert_eq!(read.advanced_extension, None);
        assert_eq!(
            read.common
                .as_ref()
                .map(|common| &common.advanced_extension),
            Some(&None)
        );
    }
    Ok(())
}

#[test]
fn the_root_writes_the_output_names_and_the_plan_its_spec_release() -> Result<(), Box<dyn Error>> {
    let plan = produce(orders()?, &Faithful)?;

    assert_eq!(root(&plan).names, ["region", "amount", "status"]);
    let version = plan.version.as_ref().expect("a version");
    assert_eq!(
        (
            version.major_number,
            version.minor_number,
            version.patch_number
        ),
        (0, 85, 0)
    );
    assert_eq!(version.producer, "semstrait");
    assert!(plan.extensions.is_empty() && plan.extension_urns.is_empty());
    Ok(())
}

#[test]
fn a_scan_writes_its_read_set_as_the_base_schema_and_its_emit_form() -> Result<(), Box<dyn Error>> {
    let scan = PlanNode::from(orders()?).with_emit_form(EmitForm::new([1, 0]))?;

    let plan = produce(scan, &Faithful)?;

    let read = read(root_input(&plan));
    let base_schema = read.base_schema.as_ref().expect("a base schema");
    assert_eq!(base_schema.names, ["region", "amount", "status"]);
    let types = base_schema.r#struct.as_ref().map(|fields| &fields.types);
    let string = |nullability: Nullability| {
        kind(Kind::String(r#type::String {
            type_variation_reference: 0,
            nullability: nullability.into(),
        }))
    };
    let i64 = kind(Kind::I64(r#type::I64 {
        type_variation_reference: 0,
        nullability: Nullability::Required.into(),
    }));
    assert_eq!(
        types,
        Some(&vec![
            string(Nullability::Nullable),
            i64,
            string(Nullability::Nullable)
        ])
    );
    let emit_kind = read
        .common
        .as_ref()
        .and_then(|common| common.emit_kind.clone());
    assert_eq!(
        emit_kind,
        Some(EmitKind::Emit(Emit {
            output_mapping: vec![1, 0]
        }))
    );
    Ok(())
}

#[test]
fn a_node_with_no_emit_form_is_direct() -> Result<(), Box<dyn Error>> {
    let plan = produce(orders()?, &Faithful)?;

    let read = read(root_input(&plan));
    let emit_kind = read
        .common
        .as_ref()
        .and_then(|common| common.emit_kind.clone());
    assert_eq!(emit_kind, Some(EmitKind::Direct(Direct {})));
    Ok(())
}

fn kind(kind: Kind) -> Type {
    Type { kind: Some(kind) }
}

#[test]
fn each_type_is_written_from_its_data_type() -> Result<(), Box<dyn Error>> {
    let required = Nullability::Required.into();
    let cases = [
        (
            DataType::Interval,
            Kind::IntervalCompound(r#type::IntervalCompound {
                precision: 9,
                type_variation_reference: 0,
                nullability: required,
            }),
        ),
        (
            DataType::Timestamp { precision: 6 },
            Kind::PrecisionTimestamp(r#type::PrecisionTimestamp {
                precision: 6,
                type_variation_reference: 0,
                nullability: required,
            }),
        ),
        (
            DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            Kind::Decimal(r#type::Decimal {
                scale: 2,
                precision: 10,
                type_variation_reference: 0,
                nullability: required,
            }),
        ),
        (
            DataType::VarChar { length: 5 },
            Kind::Varchar(r#type::VarChar {
                length: 5,
                type_variation_reference: 0,
                nullability: required,
            }),
        ),
        (
            DataType::Array(Box::new(DataType::Date)),
            Kind::List(Box::new(r#type::List {
                r#type: Some(Box::new(kind(Kind::Date(r#type::Date {
                    type_variation_reference: 0,
                    nullability: Nullability::Nullable.into(),
                })))),
                type_variation_reference: 0,
                nullability: required,
            })),
        ),
    ];

    for (data_type, expected) in cases {
        let scan = ScanNode::new(
            VirtualTable::new(Vec::new()),
            schema([field("x", data_type.clone(), false)]),
        )?;

        let plan = produce(scan, &Faithful)?;

        let base_schema = read(root_input(&plan)).base_schema.as_ref();
        let types = base_schema.and_then(|fields| fields.r#struct.as_ref());
        assert_eq!(
            types.map(|fields| fields.types.as_slice()),
            Some([kind(expected)].as_slice()),
            "{data_type}"
        );
    }
    Ok(())
}

struct TwoNamesAndAVariation;

impl SubstraitRules for TwoNamesAndAVariation {
    fn reader(&self, _scan: &ScanNode, mut read: ReadRel) -> Rel {
        if let Some(ReadType::NamedTable(table)) = &mut read.read_type {
            table.names.retain(|name| !name.is_empty());
        }
        Rel {
            rel_type: Some(rel::RelType::Read(Box::new(read))),
        }
    }

    fn type_variation(&self, data_type: &DataType) -> u32 {
        u32::from(*data_type == DataType::BigInt)
    }
}

#[test]
fn the_rules_choose_the_reader_and_the_type_variation() -> Result<(), Box<dyn Error>> {
    let plan = produce(orders()?, &TwoNamesAndAVariation)?;

    let read = read(root_input(&plan));
    let Some(ReadType::NamedTable(table)) = &read.read_type else {
        panic!("a NamedTable read");
    };
    assert_eq!(table.names, ["sales", "orders"]);
    let base_schema = read
        .base_schema
        .as_ref()
        .and_then(|fields| fields.r#struct.as_ref());
    let variations: Vec<_> = (base_schema.map(|fields| fields.types.as_slice()))
        .unwrap_or_default()
        .iter()
        .map(|field| match &field.kind {
            Some(Kind::String(string)) => string.type_variation_reference,
            Some(Kind::I64(bigint)) => bigint.type_variation_reference,
            other => panic!("a string or a bigint, not {other:?}"),
        })
        .collect();
    assert_eq!(variations, [0, 1, 0]);
    Ok(())
}

#[test]
fn a_scan_filter_reads_each_column_by_its_position() -> Result<(), Box<dyn Error>> {
    let fields = schema([
        field("region", DataType::String, true),
        field("paid", DataType::Boolean, false),
    ]);
    let scan = ScanNode::new(orders()?.reader, fields)?.with_filter(leaf(&["paid"]))?;

    let plan = produce(scan, &Faithful)?;

    let filter = read(root_input(&plan)).filter.as_deref();
    assert_eq!(filter, Some(&column(1)));
    Ok(())
}

struct DropsAProjectField;

impl SubstraitRules for DropsAProjectField {
    fn rewrite_node(&self, node: &PlanNode) -> Option<PlanNode> {
        let PlanNode::Project(project) = node else {
            return None;
        };
        let input = project.input.as_ref().clone();
        ProjectNode::new(input, Vec::new()).ok().map(PlanNode::from)
    }
}

#[test]
fn a_rewrite_that_drops_a_project_field_is_an_invalid_rewrite() -> Result<(), Box<dyn Error>> {
    let project = ProjectNode::new(orders()?, vec![(name("paid_amount"), leaf(&["amount"]))])?;

    let refusal = refusal(project, &DropsAProjectField);

    assert_eq!(
        refusal,
        Some((
            InvalidRewrite::SchemaChanged,
            vec!["node 1 (ProjectNode)".to_owned()]
        ))
    );
    Ok(())
}

struct CopiesTheOfferedSchema;

impl SubstraitRules for CopiesTheOfferedSchema {
    fn rewrite_node(&self, node: &PlanNode) -> Option<PlanNode> {
        let PlanNode::Project(project) = node else {
            return None;
        };
        let input = project.input.as_ref().clone();
        let mut fewer = ProjectNode::new(input, Vec::new()).ok()?;
        fewer.common.output_schema = project.common.output_schema.clone();
        Some(fewer.into())
    }
}

#[test]
fn a_rewrite_that_stores_the_offered_schema_over_fewer_fields_is_an_invalid_rewrite(
) -> Result<(), Box<dyn Error>> {
    let project = ProjectNode::new(orders()?, vec![(name("paid_amount"), leaf(&["amount"]))])?;

    let refusal = refusal(project, &CopiesTheOfferedSchema);

    assert_eq!(
        refusal,
        Some((
            InvalidRewrite::SchemaChanged,
            vec!["node 1 (ProjectNode)".to_owned()]
        ))
    );
    Ok(())
}

#[test]
fn a_type_that_substrait_cannot_write_is_refused_naming_the_node() -> Result<(), Box<dyn Error>> {
    let mut scan = orders()?;
    scan.schema.columns[1].data_type = DataType::Unknown;

    let refusal = refusal(scan, &Faithful);

    assert_eq!(
        refusal,
        Some((
            InvalidRewrite::NoSubstraitType {
                data_type: DataType::Unknown
            },
            vec!["node 1 (ScanNode)".to_owned()]
        ))
    );
    Ok(())
}

#[test]
fn an_aggregate_filter_clause_is_its_measure_filter_and_no_relation_has_an_advanced_extension(
) -> Result<(), Box<dyn Error>> {
    let plan = produce(paid_by_region()?, &Faithful)?;

    let [measure] = aggregate_rel(root_input(&plan)).measures.as_slice() else {
        panic!("one measure");
    };
    assert_eq!(measure.filter, Some(column(3)));
    let sum = measure.measure.as_ref().expect("an aggregate function");
    assert_eq!(
        declared_function(&plan, sum.function_reference),
        Some(("extension:io.substrait:functions_arithmetic", "sum"))
    );
    assert!(!relations(root_input(&plan))
        .into_iter()
        .any(has_advanced_extension));
    assert_eq!(plan.advanced_extensions, None);
    Ok(())
}

struct AmountAsText;

impl SubstraitRules for AmountAsText {
    fn rewrite_expression(&self, expr: &PhysicalExpr, _input: &Schema) -> Option<PhysicalExpr> {
        (*expr == leaf(&["amount"])).then(|| literal("amount"))
    }
}

#[test]
fn a_rewrite_that_changes_an_expression_type_is_an_invalid_rewrite() -> Result<(), Box<dyn Error>> {
    let refusal = refusal(paid_by_region()?, &AmountAsText);

    assert_eq!(
        refusal,
        Some((
            InvalidRewrite::TypeChanged {
                offered: DataType::BigInt,
                replacement: DataType::String
            },
            vec!["node 1 (AggregateNode)".to_owned()]
        ))
    );
    Ok(())
}

#[test]
fn a_top_n_node_that_reaches_the_producer_is_an_invalid_rewrite_naming_it(
) -> Result<(), Box<dyn Error>> {
    let order_by = vec![descending(leaf(&["amount"]))];
    let top = TopNNode::new(orders()?, order_by, 0, 10, FetchMode::RowsOnly)?;

    let refusal = refusal(top, &Faithful);

    assert_eq!(
        refusal,
        Some((
            InvalidRewrite::NoLowering { node: "TopNNode" },
            vec!["node 1 (TopNNode)".to_owned()]
        ))
    );
    Ok(())
}

fn orders_as(alias: &str) -> Result<PlanNode, Box<dyn Error>> {
    let fields =
        ["amount", "region"].map(|field| (name(&format!("{alias}.{field}")), leaf(&[field])));
    let project = ProjectNode::new(orders()?, fields.into())?;
    Ok(PlanNode::from(project).with_emit_form(EmitForm::new([3, 4]))?)
}

fn orders_joined(
    join_type: JoinType,
    post_join_filter: Option<PhysicalExpr>,
) -> Result<JoinNode, Box<dyn Error>> {
    let on = leaf(&["ssd0", "region"]).eq(leaf(&["ssd1", "region"]));
    let (left, right) = (orders_as("ssd0")?, orders_as("ssd1")?);
    Ok(JoinNode::new(left, right, join_type, on, post_join_filter)?)
}

fn join_rel(relation: &Rel) -> &JoinRel {
    let Some(rel::RelType::Join(join)) = &relation.rel_type else {
        panic!("a JoinRel");
    };
    join
}

#[test]
fn a_left_join_writes_its_condition_and_its_post_join_filter_by_position(
) -> Result<(), Box<dyn Error>> {
    let filter = leaf(&["ssd1", "amount"]).gt(literal(100_i64));

    let plan = produce(orders_joined(JoinType::Left, Some(filter))?, &Faithful)?;

    let join = join_rel(root_input(&plan));
    assert_eq!(join.r#type(), join_rel::JoinType::Left);
    let on = join.expression.as_deref().map(|on| text(&plan, on));
    assert_eq!(on.as_deref(), Some("equal($1, $3)"));
    let filter = (join.post_join_filter.as_deref()).map(|filter| text(&plan, filter));
    assert_eq!(filter.as_deref(), Some("gt($2, 100)"));
    assert!(join.left.is_some() && join.right.is_some());
    Ok(())
}

#[test]
fn a_full_join_is_an_outer_join_rel() -> Result<(), Box<dyn Error>> {
    let plan = produce(orders_joined(JoinType::Full, None)?, &Faithful)?;

    let join = join_rel(root_input(&plan));
    assert_eq!(join.r#type(), join_rel::JoinType::Outer);
    assert_eq!(join.post_join_filter, None);
    Ok(())
}

fn orders_stacked(op: MultisetOp) -> Result<MultisetNode, Box<dyn Error>> {
    let fields = vec![
        (name("ssu2.amount"), leaf(&["amount"])),
        (
            name("ssu2.region"),
            literal(Literal::Null(DataType::String)),
        ),
    ];
    let filled = PlanNode::from(ProjectNode::new(orders()?, fields)?)
        .with_emit_form(EmitForm::new([3, 4]))?;
    Ok(MultisetNode::new(vec![orders_as("ssu2")?, filled], op)?)
}

fn multiset_rel(relation: &Rel) -> &SetRel {
    let Some(rel::RelType::Set(set)) = &relation.rel_type else {
        panic!("a SetRel");
    };
    set
}

#[test]
fn a_union_is_a_set_rel_whose_parent_reads_its_output_by_position() -> Result<(), Box<dyn Error>> {
    let ops = [
        (MultisetOp::UnionAll, set_rel::SetOp::UnionAll),
        (MultisetOp::UnionDistinct, set_rel::SetOp::UnionDistinct),
    ];
    for (op, set_op) in ops {
        let region = vec![(name("region"), leaf(&["ssu2", "region"]))];

        let plan = produce(ProjectNode::new(orders_stacked(op)?, region)?, &Faithful)?;

        let above = project_rel(root_input(&plan));
        assert_eq!(above.expressions, [column(1)]);
        let union = multiset_rel(above.input.as_deref().expect("the union"));
        assert_eq!(union.op(), set_op);
        let [_, filled] = union.inputs.as_slice() else {
            panic!("two inputs");
        };
        let null = LiteralType::Null(kind(Kind::String(r#type::String {
            type_variation_reference: 0,
            nullability: Nullability::Nullable.into(),
        })));
        let null = Expression {
            rex_type: Some(RexType::Literal(expression::Literal {
                nullable: true,
                type_variation_reference: 0,
                literal_type: Some(null),
            })),
        };
        assert_eq!(project_rel(filled).expressions, [column(1), null]);
    }
    Ok(())
}

#[test]
fn two_productions_of_each_hand_built_plan_are_byte_equal() -> Result<(), Box<dyn Error>> {
    for (name, plan) in hand_built()? {
        let first = produce(plan.clone(), &Faithful)?.encode_to_vec();
        let second = produce(plan, &Faithful)?.encode_to_vec();

        assert_eq!(first, second, "{name}");
    }
    Ok(())
}

#[test]
fn an_operation_of_each_expression_kind_takes_substrait_s_official_name(
) -> Result<(), Box<dyn Error>> {
    let plan = produce(one_expression_of_each_kind()?, &Faithful)?;

    let names: Vec<_> = (plan.extensions.iter())
        .map(|declaration| match &declaration.mapping_type {
            Some(MappingType::ExtensionFunction(function)) => function.name.as_str(),
            other => panic!("a function, not {other:?}"),
        })
        .collect();
    assert_eq!(
        names,
        ["gte", "not", "lte", "and", "like", "divide", "upper", "coalesce"]
    );
    Ok(())
}

#[test]
fn date_trunc_of_a_timestamp_buckets_its_utc_instant_on_the_timezone_s_clock(
) -> Result<(), Box<dyn Error>> {
    let month = scalar(
        "date_trunc",
        vec![literal("month"), leaf(&["at"]), literal("Europe/Stockholm")],
    );
    let project = ProjectNode::new(moments()?, vec![(name("month"), month)])?;

    let plan = produce(project, &Faithful)?;

    let [month] = projected(&plan) else {
        panic!("one expression");
    };
    assert_eq!(
        text(&plan, month),
        "local_timestamp(round_calendar(assume_timezone($0, 'UTC'), FLOOR, MONTH, YEAR, 1, \
         'Europe/Stockholm'), 'Europe/Stockholm')"
    );
    let urns: Vec<_> = (plan.extension_urns.iter())
        .map(|urn| urn.urn.as_str())
        .collect();
    assert_eq!(urns, ["extension:io.substrait:functions_datetime"]);
    Ok(())
}

#[test]
fn date_trunc_of_a_date_is_round_calendar_alone() -> Result<(), Box<dyn Error>> {
    let buckets = vec![
        (
            name("week"),
            scalar(
                "date_trunc",
                vec![literal("week"), leaf(&["day"]), literal("sunday")],
            ),
        ),
        (
            name("quarter"),
            scalar("date_trunc", vec![literal("quarter"), leaf(&["day"])]),
        ),
    ];
    let project = ProjectNode::new(moments()?, buckets)?;

    let plan = produce(project, &Faithful)?;

    let texts: Vec<_> = (projected(&plan).iter())
        .map(|expr| text(&plan, expr))
        .collect();
    assert_eq!(
        texts,
        [
            "round_calendar($1, FLOOR, WEEK, US_WEEK, 1, date 3)",
            "round_calendar($1, FLOOR, MONTH, YEAR, 3, date 0)",
        ]
    );
    Ok(())
}

#[test]
fn date_trunc_of_a_timestamp_without_a_timezone_buckets_on_utc_s_clock(
) -> Result<(), Box<dyn Error>> {
    let buckets = vec![
        (
            name("month"),
            scalar("date_trunc", vec![literal("month"), leaf(&["at"])]),
        ),
        (
            name("week"),
            scalar(
                "date_trunc",
                vec![literal("week"), leaf(&["at"]), literal("sunday")],
            ),
        ),
    ];
    let project = ProjectNode::new(moments()?, buckets)?;

    let plan = produce(project, &Faithful)?;

    let texts: Vec<_> = (projected(&plan).iter())
        .map(|expr| text(&plan, expr))
        .collect();
    assert_eq!(
        texts,
        [
            "local_timestamp(round_calendar(assume_timezone($0, 'UTC'), FLOOR, MONTH, YEAR, 1, \
             'UTC'), 'UTC')",
            "local_timestamp(round_calendar(assume_timezone($0, 'UTC'), FLOOR, WEEK, US_WEEK, 1, \
             'UTC'), 'UTC')",
        ]
    );
    Ok(())
}

struct ExtractAsDatePart;

impl SubstraitRules for ExtractAsDatePart {
    fn whole_call(
        &self,
        call: &ScalarCall<Parameter, AggregateAllowed>,
        arguments: &[FunctionArgument],
        output_type: &Type,
        extensions: &mut Extensions,
    ) -> Result<Option<Expression>, InvalidRewrite> {
        let [FunctionArgument {
            arg_type: Some(ArgType::Enum(component)),
        }, value] = arguments
        else {
            return Ok(None);
        };
        if call.function().canonical.name != "extract" {
            return Ok(None);
        }
        let date_part = FunctionName {
            urn: "extension:test:engine",
            name: "date_part",
        };
        let component = Expression {
            rex_type: Some(RexType::Literal(expression::Literal {
                literal_type: Some(LiteralType::String(component.clone())),
                ..expression::Literal::default()
            })),
        };
        Ok(Some(Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: extensions.declare(date_part)?,
                arguments: vec![
                    FunctionArgument {
                        arg_type: Some(ArgType::Value(component)),
                    },
                    value.clone(),
                ],
                output_type: Some(output_type.clone()),
                ..ScalarFunction::default()
            })),
        }))
    }
}

#[test]
fn the_rules_whole_call_is_written_as_returned() -> Result<(), Box<dyn Error>> {
    let weekday = scalar("extract", vec![literal("day_of_week"), leaf(&["at"])]);
    let project = ProjectNode::new(moments()?, vec![(name("weekday"), weekday)])?;

    let plan = produce(project, &ExtractAsDatePart)?;

    let [weekday] = projected(&plan) else {
        panic!("one expression");
    };
    assert_eq!(text(&plan, weekday), "date_part('day_of_week', $0)");
    assert_eq!(plan.extensions.len(), 1);
    Ok(())
}

#[test]
fn extract_writes_its_component_as_substrait_s_enum() -> Result<(), Box<dyn Error>> {
    let parts = vec![
        (
            name("weekday"),
            scalar("extract", vec![literal("day_of_week"), leaf(&["at"])]),
        ),
        (
            name("year"),
            scalar("extract", vec![literal("year"), leaf(&["day"])]),
        ),
    ];
    let project = ProjectNode::new(moments()?, parts)?;

    let plan = produce(project, &Faithful)?;

    let texts: Vec<_> = (projected(&plan).iter())
        .map(|expr| text(&plan, expr))
        .collect();
    assert_eq!(
        texts,
        ["extract(MONDAY_DAY_OF_WEEK, ONE, $0)", "extract(YEAR, $1)"]
    );
    let urns: Vec<_> = (plan.extension_urns.iter())
        .map(|urn| urn.urn.as_str())
        .collect();
    assert_eq!(urns, ["extension:io.substrait:functions_datetime"]);
    Ok(())
}

#[test]
fn each_literal_is_a_substrait_literal_of_the_numbers_its_type_holds() -> Result<(), Box<dyn Error>>
{
    let text = |value: &str| value.to_owned();
    let values = [
        Literal::Date(text("2000-01-01")),
        Literal::Time {
            value: text("01:00:00.5"),
            precision: 3,
        },
        Literal::Timestamp {
            value: text("2000-01-01T00:00:01.25"),
            precision: 3,
        },
        Literal::Decimal {
            value: text("-1.5"),
            precision: 18,
            scale: 2,
        },
        Literal::Interval(text("P1Y2M3DT4H5M6.5S")),
        Literal::IntervalYearMonth(text("-P1Y2M")),
        Literal::IntervalDaySecond {
            value: text("P1DT1.5S"),
            precision: 6,
        },
        Literal::Uuid(text("550e8400-e29b-41d4-a716-446655440000")),
    ];
    let expressions = (values.into_iter().enumerate())
        .map(|(position, value)| (name(&format!("value_{position}")), literal(value)))
        .collect();
    let project = ProjectNode::new(orders()?, expressions)?;

    let plan = produce(project, &Faithful)?;

    let literals: Vec<_> = (projected(&plan).iter())
        .map(|expr| match &expr.rex_type {
            Some(RexType::Literal(literal)) => literal.literal_type.clone(),
            _ => None,
        })
        .collect();
    let day_second = |days, seconds, subseconds, precision| IntervalDayToSecond {
        days,
        seconds,
        subseconds,
        precision_mode: Some(PrecisionMode::Precision(precision)),
    };
    let minus_150 = [[0x6a].as_slice(), &[0xff; 15]].concat();
    let uuid = vec![
        0x55, 0x0e, 0x84, 0x00, 0xe2, 0x9b, 0x41, 0xd4, 0xa7, 0x16, 0x44, 0x66, 0x55, 0x44, 0x00,
        0x00,
    ];
    let expected = [
        LiteralType::Date(10_957),
        LiteralType::PrecisionTime(PrecisionTime {
            precision: 3,
            value: 3_600_500,
        }),
        LiteralType::PrecisionTimestamp(PrecisionTimestamp {
            precision: 3,
            value: 946_684_801_250,
        }),
        LiteralType::Decimal(Decimal {
            value: minus_150,
            precision: 18,
            scale: 2,
        }),
        LiteralType::IntervalCompound(IntervalCompound {
            interval_year_to_month: Some(IntervalYearToMonth {
                years: 1,
                months: 2,
            }),
            interval_day_to_second: Some(day_second(3, 14_706, 500_000_000, 9)),
        }),
        LiteralType::IntervalYearToMonth(IntervalYearToMonth {
            years: -1,
            months: -2,
        }),
        LiteralType::IntervalDayToSecond(day_second(1, 1, 500_000, 6)),
        LiteralType::Uuid(uuid),
    ];
    assert_eq!(literals, expected.map(Some));
    Ok(())
}

#[test]
fn a_literal_whose_value_does_not_fit_its_type_is_refused_naming_the_node(
) -> Result<(), Box<dyn Error>> {
    let cases = [
        (
            Literal::Date("2026-02-30".to_owned()),
            LiteralError::Malformed,
        ),
        (
            Literal::VarChar {
                value: "north".to_owned(),
                length: 3,
            },
            LiteralError::TooLong,
        ),
        (
            Literal::Array {
                element: DataType::BigInt,
                values: vec![Literal::Integer(1)],
            },
            LiteralError::ElementType,
        ),
    ];
    for (value, error) in cases {
        let project = ProjectNode::new(orders()?, vec![(name("value"), literal(value))])?;

        let refusal = refusal(project, &Faithful);

        assert_eq!(
            refusal,
            Some((
                InvalidRewrite::UnfitLiteral { error },
                vec!["node 1 (ProjectNode)".to_owned()]
            ))
        );
    }
    Ok(())
}

#[test]
fn a_fixed_length_literal_is_padded_to_its_length() -> Result<(), Box<dyn Error>> {
    let values = vec![
        (
            name("code"),
            literal(Literal::FixedChar {
                value: "ab".to_owned(),
                length: 4,
            }),
        ),
        (
            name("bytes"),
            literal(Literal::FixedBinary {
                value: vec![1, 2],
                length: 4,
            }),
        ),
    ];
    let project = ProjectNode::new(orders()?, values)?;

    let plan = produce(project, &Faithful)?;

    let literals: Vec<_> = (projected(&plan).iter())
        .map(|expr| match &expr.rex_type {
            Some(RexType::Literal(literal)) => literal.literal_type.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(
        literals,
        [
            Some(LiteralType::FixedChar("ab  ".to_owned())),
            Some(LiteralType::FixedBinary(vec![1, 2, 0, 0])),
        ]
    );
    Ok(())
}
