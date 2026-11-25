#![allow(dead_code)]
#![allow(
    clippy::result_large_err,
    reason = "every stage returns its Diagnostic envelope unboxed"
)]

use std::error::Error;

use semstrait_adapter::{explain, InvalidRewrite, SubstraitProducer, SubstraitRules};
use semstrait_catalog::FileFormat;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_ir::{
    AggregateNode, EmitForm, FetchNode, FilesScan, FilterNode, GranularityGroup, IrErrorKind, Name,
    Parameter, PhysicalExpr, PlanNode, ProjectNode, ScanNode, Schema, SchemaColumn, SemAnnotation,
    SemanticPlan, SortNode, SourceRole, TableScan,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, BinaryOpKind, Call, CastFailure, Expr, ExprOperationExt,
    FunctionRef, IntoLiteral, LikeKind, NamedReference, NullOrdering, SortDir, SortKey,
    UnaryOpKind,
};
use semstrait_sql::functions::registry;
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::{
    field_reference, reference_segment, FieldReference, ReferenceSegment, RexType,
};
use substrait::proto::extensions::simple_extension_declaration::MappingType;
use substrait::proto::extensions::AdvancedExtension;
use substrait::proto::function_argument::ArgType;
use substrait::proto::r#type::Kind;
use substrait::proto::{
    plan_rel, read_rel, rel, AggregateRel, Expression, FunctionArgument, Plan, ProjectRel, ReadRel,
    Rel, RelCommon, RelRoot,
};

pub struct Faithful;

impl SubstraitRules for Faithful {}

pub fn name(name: &str) -> Name {
    Name::new(name).expect("a name")
}

pub fn field(name: &str, data_type: DataType, nullable: bool) -> SchemaColumn {
    SchemaColumn {
        name: name.to_owned(),
        data_type,
        nullable,
    }
}

pub fn schema(fields: impl IntoIterator<Item = SchemaColumn>) -> Schema {
    Schema {
        columns: fields.into_iter().collect(),
    }
}

pub fn leaf(parts: &[&str]) -> PhysicalExpr {
    Expr::Column(NamedReference::new(parts.iter().copied()).expect("a name path"))
}

pub fn column(field: i32) -> Expression {
    let segment =
        reference_segment::ReferenceType::StructField(Box::new(reference_segment::StructField {
            field,
            child: None,
        }));
    Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(field_reference::ReferenceType::DirectReference(
                ReferenceSegment {
                    reference_type: Some(segment),
                },
            )),
            root_type: Some(field_reference::RootType::RootReference(
                field_reference::RootReference {},
            )),
        }))),
    }
}

pub fn orders_schema() -> Schema {
    schema([
        field("region", DataType::String, true),
        field("amount", DataType::BigInt, false),
        field("status", DataType::String, true),
    ])
}

pub fn orders() -> Result<ScanNode, IrErrorKind> {
    ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "orders".into()], None),
        orders_schema(),
    )
}

pub const ORDERS_FOLDER: &str = "s3://lake/sales/orders/";

pub const ORDERS_PATHS: [&str; 2] = [
    "s3://lake/sales/orders/part-0.parquet",
    "s3://lake/sales/orders/part-1.parquet",
];

pub fn orders_files(format: Option<FileFormat>) -> Result<ScanNode, IrErrorKind> {
    ScanNode::new(
        FilesScan::new(
            format,
            None,
            ORDERS_FOLDER,
            ORDERS_PATHS.map(String::from).to_vec(),
        ),
        orders_schema(),
    )
}

pub fn literal(value: impl IntoLiteral) -> PhysicalExpr {
    Expr::Literal(value.into_literal())
}

pub fn aggregate(
    function: &str,
    args: Vec<PhysicalExpr>,
    filter: Option<PhysicalExpr>,
) -> AggregateCall<Parameter, AggregateAllowed> {
    let Some(FunctionRef::Aggregate(function, AggregateAllowed)) = registry().get(function) else {
        panic!("the catalog declares the aggregate `{function}`");
    };
    AggregateCall::builder(function, args)
        .maybe_filter(filter)
        .build()
        .expect("an aggregate call")
}

pub fn paid_by_region() -> Result<PlanNode, Box<dyn Error>> {
    let scan = PlanNode::from(orders()?).with_trace([SemAnnotation::SemanticSource {
        semantic_source: "orders".into(),
        source_role: SourceRole::Dataset,
        originators: Vec::new(),
        physical_source: Some(name("orders_table")),
        cardinality: None,
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    }]);
    let paid_flag = leaf(&["status"]).eq(literal("paid"));
    let flagged = ProjectNode::new(scan, vec![(name("__plan_filter_paid"), paid_flag)])?;
    let paid = aggregate(
        "sum",
        vec![leaf(&["amount"])],
        Some(leaf(&["__plan_filter_paid"])),
    );
    let regions = vec![(name("region"), leaf(&["region"]))];
    let aggregate = AggregateNode::new(flagged, regions, vec![(name("paid"), paid)])?;
    Ok(PlanNode::from(aggregate).with_trace([SemAnnotation::Rank {
        scope_root: "orders".into(),
        compositions: 0,
        granularity_fit: GranularityGroup::NotSelected,
        granularity: None,
    }]))
}

pub fn scalar(function: &str, args: Vec<PhysicalExpr>) -> PhysicalExpr {
    let function = registry().get(function).expect("a catalog function");
    Expr::Function(Call::new(function, args).expect("a scalar call"))
}

pub fn descending(expr: PhysicalExpr) -> SortKey<Parameter, AggregateAllowed> {
    SortKey {
        expr,
        dir: SortDir::Desc {
            nulls: NullOrdering::Last,
        },
    }
}

pub fn large_orders_page() -> Result<PlanNode, Box<dyn Error>> {
    let condition = Expr::BinaryOp {
        op: BinaryOpKind::And,
        left: Box::new(leaf(&["amount"]).gt(literal(100_i64))),
        right: Box::new(Expr::UnaryOp {
            op: UnaryOpKind::Not,
            operand: Box::new(leaf(&["status"]).is_null()),
        }),
    };
    let filtered = FilterNode::new(orders()?, condition)?;
    let ordered = SortNode::new(filtered, vec![descending(leaf(&["amount"]))])?;
    Ok(FetchNode::new(ordered, 5, Some(10))?.into())
}

pub fn distinct_regions_and_statuses() -> Result<PlanNode, Box<dyn Error>> {
    let groupings = vec![
        (name("region"), leaf(&["region"])),
        (name("status"), leaf(&["status"])),
    ];
    let distinct = AggregateNode::new(orders()?, groupings, Vec::new())?;
    Ok(PlanNode::from(distinct).with_emit_form(EmitForm::new([1, 0]))?)
}

pub fn one_expression_of_each_kind() -> Result<PlanNode, Box<dyn Error>> {
    let (amount, region, status) = (leaf(&["amount"]), leaf(&["region"]), leaf(&["status"]));
    let expressions = vec![
        (
            name("band"),
            Expr::Case {
                whens: vec![(amount.clone().gt_eq(literal(100_i64)), literal("high"))],
                else_: Some(Box::new(literal("low"))),
            },
        ),
        (
            name("amount_text"),
            Expr::Cast {
                input: Box::new(amount.clone()),
                target: DataType::String,
                on_failure: CastFailure::Error,
            },
        ),
        (
            name("elsewhere"),
            Expr::InList {
                value: Box::new(region.clone()),
                list: vec![literal("north"), literal("south")],
                negated: true,
            },
        ),
        (
            name("mid"),
            Expr::Between {
                value: Box::new(amount.clone()),
                low: Box::new(literal(10_i64)),
                high: Box::new(literal(20_i64)),
                negated: false,
            },
        ),
        (
            name("paid_like"),
            Expr::Like {
                value: Box::new(status.clone()),
                pattern: Box::new(literal("pa%")),
                kind: LikeKind::ILike,
            },
        ),
        (
            name("half"),
            Expr::BinaryOp {
                op: BinaryOpKind::SafeDivide,
                left: Box::new(amount),
                right: Box::new(literal(2_i64)),
            },
        ),
        (
            name("named"),
            scalar(
                "coalesce",
                vec![scalar("upper", vec![region]), literal("none")],
            ),
        ),
    ];
    Ok(ProjectNode::new(orders()?, expressions)?.into())
}

pub fn aggregates_over_the_files() -> Result<PlanNode, Box<dyn Error>> {
    let Some(FunctionRef::Aggregate(count, AggregateAllowed)) = registry().get("count") else {
        panic!("the catalog declares the aggregate `count`");
    };
    let regions = AggregateCall::builder(count, vec![leaf(&["region"])])
        .distinct(true)
        .build()?;
    let aggregates = vec![
        (name("regions"), regions),
        (
            name("largest"),
            aggregate("max", vec![leaf(&["amount"])], None),
        ),
        (
            name("spread"),
            aggregate("stddev_samp", vec![leaf(&["amount"])], None),
        ),
    ];
    let files = orders_files(Some(FileFormat::Parquet))?;
    Ok(AggregateNode::new(files, Vec::new(), aggregates)?.into())
}

pub fn hand_built() -> Result<Vec<(&'static str, PlanNode)>, Box<dyn Error>> {
    Ok(vec![
        ("paid_by_region", paid_by_region()?),
        ("large_orders_page", large_orders_page()?),
        (
            "distinct_regions_and_statuses",
            distinct_regions_and_statuses()?,
        ),
        (
            "one_expression_of_each_kind",
            one_expression_of_each_kind()?,
        ),
        ("aggregates_over_the_files", aggregates_over_the_files()?),
    ])
}

pub fn plan(root: impl Into<PlanNode>) -> SemanticPlan {
    let root = root.into();
    let names = (root.common().output_schema.columns.iter())
        .map(|field| name(&field.name))
        .collect();
    SemanticPlan::new(root, names)
}

#[allow(
    clippy::type_complexity,
    reason = "ADR 0005's fail-fast form, spelled out as a caller binds it"
)]
fn no_warnings<T>(
    adapted: Result<
        (T, Diagnostics<InvalidRewrite>),
        (Diagnostic<InvalidRewrite>, Diagnostics<InvalidRewrite>),
    >,
) -> Result<T, Diagnostic<InvalidRewrite>> {
    let (output, warnings) = match adapted {
        Ok((output, warnings)) => (Ok(output), warnings),
        Err((error, warnings)) => (Err(error), warnings),
    };
    assert!(warnings.is_empty(), "adapt raises no warning: {warnings:?}");
    output
}

pub fn produce(
    root: impl Into<PlanNode>,
    rules: &dyn SubstraitRules,
) -> Result<Plan, Diagnostic<InvalidRewrite>> {
    no_warnings(SubstraitProducer::new(rules).produce(&plan(root)))
}

pub fn explained(
    root: impl Into<PlanNode>,
    rules: &dyn SubstraitRules,
) -> Result<String, Diagnostic<InvalidRewrite>> {
    no_warnings(explain(&plan(root), rules))
}

pub fn refusal(
    root: impl Into<PlanNode>,
    rules: &dyn SubstraitRules,
) -> Option<(InvalidRewrite, Vec<String>)> {
    (produce(root, rules).err()).map(|refusal| (refusal.kind, refusal.notes))
}

pub fn root(plan: &Plan) -> &RelRoot {
    let [relation] = plan.relations.as_slice() else {
        panic!("one relation");
    };
    let Some(plan_rel::RelType::Root(root)) = &relation.rel_type else {
        panic!("a root relation");
    };
    root
}

pub fn root_input(plan: &Plan) -> &Rel {
    root(plan).input.as_ref().expect("the root's input")
}

pub fn read(relation: &Rel) -> &ReadRel {
    let Some(rel::RelType::Read(read)) = &relation.rel_type else {
        panic!("a ReadRel");
    };
    read
}

fn parts(relation: &Rel) -> (Option<&Rel>, Option<&RelCommon>, Option<&AdvancedExtension>) {
    use rel::RelType;
    match &relation.rel_type {
        Some(RelType::Read(read)) => (None, read.common.as_ref(), read.advanced_extension.as_ref()),
        Some(RelType::Filter(rel)) => (
            rel.input.as_deref(),
            rel.common.as_ref(),
            rel.advanced_extension.as_ref(),
        ),
        Some(RelType::Project(rel)) => (
            rel.input.as_deref(),
            rel.common.as_ref(),
            rel.advanced_extension.as_ref(),
        ),
        Some(RelType::Aggregate(rel)) => (
            rel.input.as_deref(),
            rel.common.as_ref(),
            rel.advanced_extension.as_ref(),
        ),
        Some(RelType::Sort(rel)) => (
            rel.input.as_deref(),
            rel.common.as_ref(),
            rel.advanced_extension.as_ref(),
        ),
        Some(RelType::Fetch(rel)) => (
            rel.input.as_deref(),
            rel.common.as_ref(),
            rel.advanced_extension.as_ref(),
        ),
        other => panic!("a relation of one Dataset, not {other:?}"),
    }
}

pub fn relations(relation: &Rel) -> Vec<&Rel> {
    let (input, ..) = parts(relation);
    std::iter::once(relation)
        .chain(input.into_iter().flat_map(relations))
        .collect()
}

pub fn has_advanced_extension(relation: &Rel) -> bool {
    let (_, common, own) = parts(relation);
    own.is_some() || common.is_some_and(|common| common.advanced_extension.is_some())
}

pub fn aggregate_rel(relation: &Rel) -> &AggregateRel {
    let Some(rel::RelType::Aggregate(aggregate)) = &relation.rel_type else {
        panic!("an AggregateRel");
    };
    aggregate
}

pub fn project_rel(relation: &Rel) -> &ProjectRel {
    let Some(rel::RelType::Project(project)) = &relation.rel_type else {
        panic!("a ProjectRel");
    };
    project
}

pub fn declared_function(plan: &Plan, anchor: u32) -> Option<(&str, &str)> {
    let function =
        plan.extensions
            .iter()
            .find_map(|declaration| match &declaration.mapping_type {
                Some(MappingType::ExtensionFunction(function))
                    if function.function_anchor == anchor =>
                {
                    Some(function)
                }
                _ => None,
            })?;
    let urn = (plan.extension_urns.iter())
        .find(|urn| urn.extension_urn_anchor == function.extension_urn_reference)?;
    Some((urn.urn.as_str(), function.name.as_str()))
}

pub fn read_type(plan: &Plan) -> &read_rel::ReadType {
    read(root_input(plan))
        .read_type
        .as_ref()
        .expect("a read type")
}

pub fn moments() -> Result<ScanNode, IrErrorKind> {
    ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "moments".into()], None),
        schema([
            field("at", DataType::timestamp(), false),
            field("day", DataType::Date, false),
            field("clock", DataType::time(), false),
        ]),
    )
}

pub fn lines() -> Result<ScanNode, IrErrorKind> {
    let decimal = |precision, scale| DataType::Decimal { precision, scale };
    ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "lines".into()], None),
        schema([
            field("qty", DataType::Integer, false),
            field("units", DataType::BigInt, false),
            field("price", decimal(18, 2), false),
            field("rate", decimal(10, 4), false),
            field("score", DataType::Double, false),
        ]),
    )
}

pub fn codes() -> Result<ScanNode, IrErrorKind> {
    ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "codes".into()], None),
        schema([
            field("status", DataType::String, true),
            field("code", DataType::VarChar { length: 8 }, true),
            field("flag", DataType::FixedChar { length: 2 }, true),
        ]),
    )
}

pub fn projected(plan: &Plan) -> &[Expression] {
    &project_rel(root_input(plan)).expressions
}

pub fn call_text(plan: &Plan, anchor: u32, arguments: &[FunctionArgument]) -> String {
    let name = declared_function(plan, anchor).map_or("undeclared", |(_, name)| name);
    let arguments: Vec<_> = (arguments.iter())
        .map(|argument| match &argument.arg_type {
            Some(ArgType::Value(value)) => text(plan, value),
            Some(ArgType::Enum(value)) => value.clone(),
            other => format!("{other:?}"),
        })
        .collect();
    format!("{name}({})", arguments.join(", "))
}

pub fn text(plan: &Plan, expr: &Expression) -> String {
    match &expr.rex_type {
        Some(RexType::ScalarFunction(function)) => {
            call_text(plan, function.function_reference, &function.arguments)
        }
        Some(RexType::Literal(literal)) => match &literal.literal_type {
            Some(LiteralType::String(value)) => format!("'{value}'"),
            Some(LiteralType::I32(value)) => value.to_string(),
            Some(LiteralType::I64(value)) => value.to_string(),
            Some(LiteralType::Date(days)) => format!("date {days}"),
            other => format!("{other:?}"),
        },
        Some(RexType::Cast(cast)) => {
            let value = cast
                .input
                .as_deref()
                .map_or_else(String::new, |input| text(plan, input));
            let to = match cast.r#type.as_ref().and_then(|to| to.kind.as_ref()) {
                Some(Kind::Fp64(_)) => "fp64".to_owned(),
                Some(Kind::I64(_)) => "i64".to_owned(),
                Some(Kind::String(_)) => "string".to_owned(),
                Some(Kind::Decimal(decimal)) => {
                    format!("decimal({}, {})", decimal.precision, decimal.scale)
                }
                other => format!("{other:?}"),
            };
            format!("cast({value} as {to})")
        }
        Some(RexType::Selection(selection)) => match &selection.reference_type {
            Some(field_reference::ReferenceType::DirectReference(ReferenceSegment {
                reference_type: Some(reference_segment::ReferenceType::StructField(field)),
            })) => format!("${}", field.field),
            other => format!("{other:?}"),
        },
        other => format!("{other:?}"),
    }
}
