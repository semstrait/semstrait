use std::fmt;

use semstrait_common::SemanticsName;
use semstrait_common::ValidateError;
use semstrait_ir::{
    field_name, AggregateNode, CrossNode, EmitForm, FetchNode, FilterNode, IrErrorKind, JoinNode,
    JoinType, MultisetNode, MultisetOp, Name, Parameter, PhysicalExpr, PlanNode, ProjectNode,
    Reader, ScanNode, Schema, SemAnnotation, SortNode, VirtualTable,
};
use semstrait_sql::expr::{
    AggregateAllowed, AggregateCall, Call, Expr, FunctionRef, NamedReference, Over, SortKey,
    WindowCall,
};
use semstrait_sql::functions::{registry, AggregateFunction};

use crate::error::call_refused;

pub(crate) type Aggregate = AggregateCall<Parameter, AggregateAllowed>;

pub(crate) fn catalog_call(
    name: &str,
    args: Vec<PhysicalExpr>,
) -> Result<PhysicalExpr, ValidateError> {
    let refused = |reason| ValidateError::CallRefused { reason };
    let function = (registry().get(name))
        .ok_or_else(|| refused(format!("the function catalog holds no `{name}`")))?;
    let call = Call::new(function, args).map_err(call_refused)?;
    Ok(Expr::Function(call))
}

pub(crate) fn window(
    name: &str,
    over: Over<Parameter, AggregateAllowed>,
) -> Result<PhysicalExpr, ValidateError> {
    let refused = |reason| ValidateError::CallRefused { reason };
    let Some(FunctionRef::Window(function)) = registry().get(name) else {
        return Err(refused(format!(
            "the function catalog holds no window function `{name}`"
        )));
    };
    let call =
        (WindowCall::builder(function, Vec::new()).over(over).build()).map_err(call_refused)?;
    Ok(Expr::Function(Call::Window(call)))
}

pub(crate) fn formula_field(
    qualifier: Option<&str>,
    metric: &SemanticsName,
) -> Result<(Name, PhysicalExpr), ValidateError> {
    field(qualifier, reserved("formula", metric))
}

pub(crate) fn aggregated_over(
    aggregation: (&'static AggregateFunction, bool),
    value: PhysicalExpr,
) -> Result<Aggregate, ValidateError> {
    let (function, distinct) = aggregation;
    AggregateCall::builder(function, vec![value])
        .distinct(distinct)
        .build()
        .map_err(call_refused)
}

pub(crate) fn field(
    qualifier: Option<&str>,
    name: impl Into<String>,
) -> Result<(Name, PhysicalExpr), ValidateError> {
    leaf_field(qualified(qualifier, name)?)
}

pub(crate) fn qualified(
    qualifier: Option<&str>,
    name: impl Into<String>,
) -> Result<NamedReference, ValidateError> {
    let parts = (qualifier.map(str::to_owned).into_iter()).chain([name.into()]);
    NamedReference::new(parts).ok_or(ValidateError::EmptyName)
}

pub(crate) fn reserved(purpose: impl fmt::Display, name: impl fmt::Display) -> String {
    format!("{}{purpose}_{name}", Name::PLAN_PREFIX)
}

pub(crate) fn unbucketed(
    qualifier: Option<&str>,
    name: &SemanticsName,
) -> Result<NamedReference, ValidateError> {
    qualified(qualifier, reserved("unbucketed", name))
}

pub(crate) fn condition_field(
    qualifier: Option<&str>,
    measure: &SemanticsName,
) -> Result<(Name, PhysicalExpr), ValidateError> {
    field(qualifier, reserved("cond", measure))
}

pub(crate) fn stemmed_entry(
    qualifier: Option<&str>,
    stem: Option<&str>,
    semantics: &SemanticsName,
    n: usize,
    plain: bool,
) -> Result<(Name, PhysicalExpr), ValidateError> {
    match (plain, stem) {
        (true, None) => field(qualifier, semantics.as_str()),
        (true, Some(stem)) => field(qualifier, reserved(stem, semantics)),
        (false, stem) => {
            let purpose = format_args!("{}agg{n}", stem.unwrap_or_default());
            field(qualifier, reserved(purpose, semantics))
        }
    }
}

pub(crate) fn leaf_field(leaf: NamedReference) -> Result<(Name, PhysicalExpr), ValidateError> {
    Ok((Name::new(field_name(&leaf))?, Expr::Column(leaf)))
}

pub(crate) fn reference(name: &str) -> Result<NamedReference, ValidateError> {
    NamedReference::unqualified(name).ok_or(ValidateError::EmptyName)
}

pub(crate) fn step_scan(
    reader: Reader,
    schema: Schema,
    trace: Vec<SemAnnotation>,
) -> Result<PlanNode, IrErrorKind> {
    ScanNode::new(reader, schema).map(|scan| PlanNode::from(scan).with_trace(trace))
}

pub(crate) fn step_empty(
    schema: Schema,
    trace: Vec<SemAnnotation>,
) -> Result<PlanNode, IrErrorKind> {
    step_scan(VirtualTable::new(Vec::new()).into(), schema, trace)
}

pub(crate) fn step_union(inputs: Vec<PlanNode>, op: MultisetOp) -> Result<PlanNode, IrErrorKind> {
    MultisetNode::new(inputs, op).map(PlanNode::from)
}

pub(crate) fn step_project(
    input: PlanNode,
    fields: Vec<(Name, PhysicalExpr)>,
) -> Result<PlanNode, IrErrorKind> {
    let passed = input.common().output_schema.columns.len();
    let emit_form = EmitForm::new(passed..passed + fields.len());
    PlanNode::from(ProjectNode::new(input, fields)?).with_emit_form(emit_form)
}

pub(crate) fn step_filter(input: PlanNode, filter: PhysicalExpr) -> Result<PlanNode, IrErrorKind> {
    FilterNode::new(input, filter).map(PlanNode::from)
}

pub(crate) fn step_join(
    left: PlanNode,
    right: PlanNode,
    join_type: JoinType,
    on: PhysicalExpr,
    post_join_filter: Option<PhysicalExpr>,
) -> Result<PlanNode, IrErrorKind> {
    JoinNode::new(left, right, join_type, on, post_join_filter).map(PlanNode::from)
}

pub(crate) fn step_cross(left: PlanNode, right: PlanNode) -> PlanNode {
    CrossNode::new(left, right).into()
}

pub(crate) fn step_agg(
    input: PlanNode,
    groupings: Vec<(Name, PhysicalExpr)>,
    aggregates: Vec<(Name, Aggregate)>,
) -> Result<PlanNode, IrErrorKind> {
    AggregateNode::new(input, groupings, aggregates).map(PlanNode::from)
}

pub(crate) fn step_distinct(
    input: PlanNode,
    kept: Vec<(Name, PhysicalExpr)>,
) -> Result<PlanNode, IrErrorKind> {
    step_agg(input, kept, Vec::new())
}

pub(crate) fn step_sort(
    input: PlanNode,
    keys: Vec<SortKey<Parameter, AggregateAllowed>>,
) -> Result<PlanNode, IrErrorKind> {
    SortNode::new(input, keys).map(PlanNode::from)
}

pub(crate) fn step_fetch(
    input: PlanNode,
    offset: i64,
    limit: Option<i64>,
) -> Result<PlanNode, IrErrorKind> {
    FetchNode::new(input, offset, limit).map(PlanNode::from)
}
