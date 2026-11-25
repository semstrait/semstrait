use std::borrow::Cow;
use std::ops::BitAnd;

use semstrait_common::SemanticsName;
use semstrait_common::ValidateError;
use semstrait_ir::{Cardinality, JoinType, Name, PhysicalExpr, PlanNode};
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, NamedReference};

use super::{build, every_field, ordered, trace, OutputField};
use crate::frame::Decided;
use crate::read_model::ReadModel;
use crate::steps::{
    catalog_call, field, reserved, step_cross, step_filter, step_join, step_project,
};
use crate::strategy::{FormulaNode, StrategyInput};
use crate::{Advisory, BoundName, PlanError, RequestContext};

pub(super) struct Side<'a> {
    pub(super) input: Cow<'a, StrategyInput>,
    pub(super) qualifier: Cow<'a, str>,
}

impl<'a> Side<'a> {
    pub(super) fn of(input: &'a StrategyInput, qualifier: &'a str) -> Self {
        Self {
            input: Cow::Borrowed(input),
            qualifier: Cow::Borrowed(qualifier),
        }
    }
}

pub(super) fn joined(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    decided: &Decided,
    qualifier: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let refused = |error| input.refused(error);
    let node_refused = |error| input.node_refused(error);
    let (sides, nodes): (Vec<Side>, Vec<PlanNode>) = (decided.operands.iter())
        .map(|child| {
            let alias = child.input.alias.as_str();
            let node = build(read_model, child, Some(alias), context, advised)?;
            let side = Side::of(&child.input, alias);
            let node = under_conditions(&side, node)?;
            Ok((side, node))
        })
        .collect::<Result<_, PlanError>>()?;
    let mut given = Vec::new();
    for side in &sides {
        for node in &side.input.formula_nodes {
            let (_, read) = node
                .field(&side.input, Some(&side.qualifier))
                .map_err(refused)?;
            given.push((node, read));
        }
    }
    let mut node = left_deep_full(read_model, input, &sides, nodes)?;
    if let Some(filter) = &decided.above_join {
        node = step_filter(node, filter.clone()).map_err(node_refused)?;
    }
    let own = input.alias.as_str();
    let mut shown = shown_fields(read_model, input, &input.names, |binding| {
        held_read(&sides, input, binding, child_field)
    })?;
    for (key, _) in &input.part_key {
        let read = key_over_holders(&sides, key).map_err(refused)?;
        shown.push((
            SemanticsName::new(key.name()),
            read.ok_or_else(|| input.broken())?,
        ));
    }
    let mut fields = output_fields(input, &shown, Some(own))?;
    for (formula_node, read) in &given {
        let (field, _) = formula_node.field(input, Some(own)).map_err(refused)?;
        let measure = formula_node.measure.clone();
        fields.push(OutputField::new(measure, field, read.clone()));
    }
    node = step_project(node, ordered(Some(own), fields)).map_err(node_refused)?;
    if qualifier != Some(own) {
        let nodes: Vec<&FormulaNode> = given.into_iter().map(|(node, _)| node).collect();
        node = requalified(input, own, qualifier, node, &nodes)?;
    }
    Ok(node)
}

pub(super) fn shown_fields<'b>(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    bindings: impl IntoIterator<Item = &'b BoundName>,
    read: impl Fn(&BoundName) -> Result<PhysicalExpr, PlanError>,
) -> Result<Vec<(SemanticsName, PhysicalExpr)>, PlanError> {
    let mut shown = Vec::new();
    for binding in bindings {
        shown.push((binding.name.clone(), read(binding)?));
    }
    for name in &input.null_filled {
        let null = Literal::Null(input.null_fill_type(read_model, name)?);
        shown.push((name.clone(), Expr::Literal(null)));
    }
    Ok(shown)
}

pub(super) fn output_fields(
    input: &StrategyInput,
    shown: &[(SemanticsName, PhysicalExpr)],
    qualifier: Option<&str>,
) -> Result<Vec<OutputField>, PlanError> {
    shown
        .iter()
        .map(|(semantics, read)| {
            let (name, _) = field(qualifier, semantics.as_str()).map_err(|e| input.refused(e))?;
            Ok(OutputField::new(semantics.clone(), name, read.clone()))
        })
        .collect()
}

pub(super) fn requalified(
    input: &StrategyInput,
    own: &str,
    qualifier: Option<&str>,
    node: PlanNode,
    given: &[&FormulaNode],
) -> Result<PlanNode, PlanError> {
    let refused = |error| input.refused(error);
    let keys = (input.part_key.iter()).map(|(key, _)| SemanticsName::new(key.name()));
    let mut fields = (input.names.iter().map(|binding| binding.name.clone()))
        .chain(input.null_filled.iter().cloned())
        .chain(keys)
        .map(|name| {
            let (_, read) = field(Some(own), name.as_str())?;
            let (field, _) = field(qualifier, name.as_str())?;
            Ok(OutputField::new(name, field, read))
        })
        .collect::<Result<Vec<_>, ValidateError>>()
        .map_err(refused)?;
    for formula_node in given {
        let (_, read) = formula_node.field(input, Some(own)).map_err(refused)?;
        let (name, _) = formula_node.field(input, qualifier).map_err(refused)?;
        fields.push(OutputField::new(formula_node.measure.clone(), name, read));
    }
    step_project(node, ordered(qualifier, fields)).map_err(|error| input.node_refused(error))
}

pub(super) fn left_deep_full(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    sides: &[Side],
    nodes: Vec<PlanNode>,
) -> Result<PlanNode, PlanError> {
    let mut nodes = nodes.into_iter();
    let Some(mut node) = nodes.next() else {
        return Err(input.broken());
    };
    for (index, right) in (1..).zip(nodes) {
        node = full_join(
            read_model,
            input,
            &sides[..index],
            &sides[index],
            node,
            right,
        )?;
    }
    Ok(node)
}

pub(super) fn full_join(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    earlier: &[Side],
    side: &Side,
    left: PlanNode,
    right: PlanNode,
) -> Result<PlanNode, PlanError> {
    let refused = |error| input.refused(error);
    let mut on = Vec::new();
    for name in &side.input.names {
        let Some(key) = read_over_holders(earlier, name, child_field).map_err(refused)? else {
            continue;
        };
        let (_, read) = field(Some(&side.qualifier), name.name.as_str()).map_err(refused)?;
        on.push(key.is_not_distinct_from(read));
    }
    for (key, _) in &side.input.part_key {
        let Some(held) = key_over_holders(earlier, key).map_err(refused)? else {
            continue;
        };
        let (_, read) = field(Some(&side.qualifier), key.name()).map_err(refused)?;
        on.push(held.is_not_distinct_from(read));
    }
    joined_full(read_model, input, on, left, right)
}

pub(super) fn joined_full(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    on: Vec<PhysicalExpr>,
    left: PlanNode,
    right: PlanNode,
) -> Result<PlanNode, PlanError> {
    let node = match on.into_iter().reduce(BitAnd::bitand) {
        Some(on) => step_join(left, right, JoinType::Full, on, None)
            .map_err(|error| input.node_refused(error))?,
        None => step_cross(left, right),
    };
    let one_to_one = trace(
        read_model,
        input,
        None,
        Some(Cardinality::OneToOne),
        None,
        None,
        Vec::new(),
    )?;
    Ok(node.with_trace([one_to_one]))
}

type FieldOf = fn(&Side, &SemanticsName) -> Result<(Name, PhysicalExpr), ValidateError>;

fn read_over_holders(
    sides: &[Side],
    name: &BoundName,
    field_of: FieldOf,
) -> Result<Option<PhysicalExpr>, ValidateError> {
    let mut reads = (sides.iter())
        .filter(|side| side.input.names.contains(name))
        .map(|side| field_of(side, &name.name).map(|(_, read)| read))
        .collect::<Result<Vec<_>, _>>()?;
    match reads.len() {
        0 | 1 => Ok(reads.pop()),
        _ => catalog_call("coalesce", reads).map(Some),
    }
}

pub(super) fn key_over_holders(
    sides: &[Side],
    key: &NamedReference,
) -> Result<Option<PhysicalExpr>, ValidateError> {
    let mut reads = (sides.iter())
        .filter(|side| (side.input.part_key.iter()).any(|(held, _)| held == key))
        .map(|side| field(Some(&side.qualifier), key.name()).map(|(_, read)| read))
        .collect::<Result<Vec<_>, _>>()?;
    match reads.len() {
        0 | 1 => Ok(reads.pop()),
        _ => catalog_call("coalesce", reads).map(Some),
    }
}

pub(super) fn held_read(
    sides: &[Side],
    input: &StrategyInput,
    binding: &BoundName,
    field_of: FieldOf,
) -> Result<PhysicalExpr, PlanError> {
    let read = read_over_holders(sides, binding, field_of).map_err(|e| input.refused(e))?;
    read.ok_or_else(|| input.broken())
}

pub(super) fn plain_field(
    side: &Side,
    name: &SemanticsName,
) -> Result<(Name, PhysicalExpr), ValidateError> {
    field(Some(&side.qualifier), name.as_str())
}

fn child_field(side: &Side, name: &SemanticsName) -> Result<(Name, PhysicalExpr), ValidateError> {
    if side.input.conditions.contains_key(name) {
        return field(Some(&side.qualifier), reserved("filter", name));
    }
    field(Some(&side.qualifier), name.as_str())
}

fn under_conditions(side: &Side, node: PlanNode) -> Result<PlanNode, PlanError> {
    let child = &*side.input;
    if child.conditions.is_empty() {
        return Ok(node);
    }
    let refused = |error| child.refused(error);
    let mut fields = every_field(&node).map_err(refused)?;
    for measure in child.conditions.keys() {
        let (plain, _) = field(Some(&side.qualifier), measure.as_str()).map_err(refused)?;
        let (own, _) = child_field(side, measure).map_err(refused)?;
        for (name, _) in fields.iter_mut().filter(|(name, _)| *name == plain) {
            *name = own.clone();
        }
    }
    step_project(node, fields).map_err(|error| child.node_refused(error))
}
