#![allow(dead_code)]

use semstrait_ir::{
    CrossNode, EmitForm, IrErrorKind, JoinNode, JoinType, Name, Parameter, PhysicalExpr, PlanNode,
    ProjectNode, ScanNode, Schema, SchemaColumn, SemanticPlan, TableScan,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Expr, ExprOperationExt, FunctionRef, Literal, NamedReference,
    NullOrdering, SortDir, SortKey,
};
use semstrait_sql::functions::registry;

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

pub fn unbound(parts: &[&str], fields: &[&str]) -> IrErrorKind {
    IrErrorKind::UnboundLeaf {
        leaf: NamedReference::new(parts.iter().copied()).expect("a name path"),
        fields: fields.iter().map(|&field| field.to_owned()).collect(),
    }
}

pub fn bigint(value: i64) -> PhysicalExpr {
    Expr::Literal(Literal::BigInt(value))
}

pub fn text(value: &str) -> PhysicalExpr {
    Expr::Literal(Literal::String(value.to_owned()))
}

pub fn sum(
    argument: PhysicalExpr,
    filter: Option<PhysicalExpr>,
) -> AggregateCall<Parameter, AggregateAllowed> {
    let Some(FunctionRef::Aggregate(sum, AggregateAllowed)) = registry().get("sum") else {
        panic!("the catalog declares the aggregate `sum`");
    };
    AggregateCall::builder(sum, vec![argument])
        .maybe_filter(filter)
        .build()
        .expect("sum takes one argument and a FILTER")
}

pub fn string_agg(
    argument: PhysicalExpr,
    order: SortKey<Parameter, AggregateAllowed>,
) -> AggregateCall<Parameter, AggregateAllowed> {
    let Some(FunctionRef::Aggregate(string_agg, AggregateAllowed)) = registry().get("string_agg")
    else {
        panic!("the catalog declares the aggregate `string_agg`");
    };
    AggregateCall::builder(string_agg, vec![argument, text(",")])
        .order_by(vec![order])
        .build()
        .expect("string_agg takes two arguments and an ORDER BY")
}

pub fn orders() -> Result<ScanNode, IrErrorKind> {
    ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "orders".into()], None),
        schema([
            field("region", DataType::String, true),
            field("amount", DataType::BigInt, false),
            field("status", DataType::String, true),
        ]),
    )
}

pub fn customers() -> Result<ScanNode, IrErrorKind> {
    ScanNode::new(
        TableScan::named_table([String::new(), "sales".into(), "customers".into()], None),
        schema([
            field("region", DataType::String, false),
            field("name", DataType::String, false),
        ]),
    )
}

pub fn projection(alias: &str, read: ScanNode) -> Result<PlanNode, IrErrorKind> {
    let fields = (read.schema.columns.iter())
        .map(|column| {
            let output = name(&format!("{alias}.{}", column.name));
            (output, leaf(&["ps0", &column.name]))
        })
        .collect();
    projected(read, fields)
}

pub fn projected(
    read: ScanNode,
    fields: Vec<(Name, PhysicalExpr)>,
) -> Result<PlanNode, IrErrorKind> {
    let width = read.schema.columns.len();
    let computed = width..width + fields.len();
    PlanNode::from(ProjectNode::new(read, fields)?).with_emit_form(EmitForm::new(computed))
}

pub fn same_region() -> PhysicalExpr {
    leaf(&["ssd0", "region"]).eq(leaf(&["ssd1", "region"]))
}

pub fn orders_with_customers(
    join_type: JoinType,
    on: PhysicalExpr,
    post_join_filter: Option<PhysicalExpr>,
) -> Result<JoinNode, IrErrorKind> {
    JoinNode::new(
        projection("ssd0", orders()?)?,
        projection("ssd1", customers()?)?,
        join_type,
        on,
        post_join_filter,
    )
}

pub fn orders_by_customers() -> Result<CrossNode, IrErrorKind> {
    Ok(CrossNode::new(
        projection("ssd0", orders()?)?,
        projection("ssd1", customers()?)?,
    ))
}

pub fn plan(root: impl Into<PlanNode>) -> SemanticPlan {
    let root = root.into();
    let names = (root.common().output_schema.columns.iter())
        .map(|field| name(&field.name))
        .collect();
    SemanticPlan::new(root, names)
}

pub fn refusal(plan: &SemanticPlan) -> Option<(IrErrorKind, Vec<String>)> {
    plan.validate()
        .err()
        .map(|diagnostic| (diagnostic.kind, diagnostic.notes))
}

pub fn at(anchor: u128, kind: &str) -> Vec<String> {
    vec![format!("node {anchor} ({kind})")]
}

pub fn key(expr: PhysicalExpr, dir: SortDir) -> SortKey<Parameter, AggregateAllowed> {
    SortKey { expr, dir }
}

pub fn ascending(expr: PhysicalExpr) -> SortKey<Parameter, AggregateAllowed> {
    key(
        expr,
        SortDir::Asc {
            nulls: NullOrdering::Last,
        },
    )
}
