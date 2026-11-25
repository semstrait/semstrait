use std::borrow::Cow;
use std::collections::BTreeMap;
use std::slice;

use semstrait_common::SemanticsName;
use semstrait_ir::{Name, PhysicalExpr, PlanNode, Schema};
use semstrait_sql::expr::{CastFailure, NamedReference};

use super::drill_across::{
    held_read, key_over_holders, left_deep_full, output_fields, plain_field, requalified,
    shown_fields, Side,
};
use super::{
    answering, bucket, build, every_field, ordered, traced, typed_values, Declared, OutputField,
};
use crate::formula::cast_where_differs;
use crate::frame::{Compute, Frame, GroupFrame};
use crate::grainset_metrics::{
    aggregation, groups_by, owns_metric, signature, GroupKind, GroupMetric, Parts, ReachKind,
};
use crate::predicate::{above_joins, leaves_beyond};
use crate::read_model::ReadModel;
use crate::steps::{aggregated_over, field, formula_field, step_agg, step_filter, step_project};
use crate::strategy::{FormulaNode, StrategyInput};
use crate::tree::replace_columns;
use crate::{Advisory, BoundName, PlanError, RequestContext};

pub(super) fn built(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    compute: &Compute,
    qualifier: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let own = input.alias.as_str();
    let mut nodes = Vec::new();
    let mut held: Vec<Side> = Vec::new();
    for (
        index,
        GroupFrame {
            group,
            frame: operand,
        },
    ) in compute.groups.iter().enumerate()
    {
        let stem = group_qualifier(own, index);
        let node = group_built(read_model, operand, &stem, context, advised)?;
        let (metrics, condition, parts) = match &group.kind {
            GroupKind::Base => {
                nodes.push(node);
                held.push(Side {
                    input: Cow::Borrowed(&operand.input),
                    qualifier: Cow::Owned(stem),
                });
                continue;
            }
            GroupKind::FormulaGrain(grain) => {
                (grain.metrics.as_slice(), grain.condition.as_ref(), None)
            }
            GroupKind::Parts(parts) => (
                slice::from_ref(&parts.metric),
                parts.condition.as_ref(),
                Some(parts),
            ),
        };
        let sides = [Side::of(&operand.input, &stem)];
        let formula = FormulaStep {
            over: Over {
                read_model,
                input: &operand.input,
                sides: &sides,
            },
            compute: input,
            metrics,
            condition,
            fine: group.fine(),
            parts,
            context,
            stem: &stem,
        };
        let (node, side) = formula.aggregated(node)?;
        nodes.push(node);
        held.push(Side {
            input: Cow::Owned(side),
            qualifier: Cow::Owned(stem),
        });
    }
    let mut node = left_deep_full(read_model, input, &held, nodes)?;
    for (offset, after) in compute.after_join.iter().enumerate() {
        let stem = group_qualifier(own, compute.groups.len() + offset);
        let formula = FormulaStep {
            over: Over {
                read_model,
                input,
                sides: &held,
            },
            compute: input,
            metrics: slice::from_ref(&after.metric),
            condition: after.condition.as_ref(),
            fine: &[],
            parts: None,
            context,
            stem: &stem,
        };
        let (computed, side) = formula.in_place(node)?;
        node = computed;
        held.push(Side {
            input: Cow::Owned(side),
            qualifier: Cow::Owned(stem),
        });
    }
    let schema = node.common().output_schema.clone();
    let reads = Reads {
        over: Over {
            read_model,
            input,
            sides: &held,
        },
        schema: &schema,
    };
    let shown = reads.shown()?;
    let given = reads.given()?;
    let fields = |qualifier: Option<&str>| -> Result<Vec<OutputField>, PlanError> {
        let mut fields = output_fields(input, &shown, qualifier)?;
        for (key, _) in &input.part_key {
            let read = key_over_holders(&held, key).map_err(|e| input.refused(e))?;
            let (name, _) = field(qualifier, key.name()).map_err(|e| input.refused(e))?;
            let semantics = SemanticsName::new(key.name());
            fields.push(OutputField::new(
                semantics,
                name,
                read.ok_or_else(|| input.broken())?,
            ));
        }
        for (formula_node, read) in &given {
            let (field, _) =
                (formula_node.field(input, qualifier)).map_err(|e| input.refused(e))?;
            let measure = formula_node.measure.clone();
            fields.push(OutputField::new(measure, field, read.clone()));
        }
        Ok(fields)
    };
    if input.predicates.is_empty() {
        let fields = fields(qualifier)?;
        return step_project(node, ordered(qualifier, fields)).map_err(|e| input.node_refused(e));
    }
    let fields = fields(Some(own))?;
    let node = step_project(node, ordered(Some(own), fields)).map_err(|e| input.node_refused(e))?;
    let node = filtered(input, own, node)?;
    let nodes: Vec<&FormulaNode> = input.formula_nodes.iter().collect();
    requalified(input, own, qualifier, node, &nodes)
}

fn group_built(
    read_model: &dyn ReadModel,
    frame: &Frame,
    stem: &str,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let (unit, passed) = answering(read_model, frame)?;
    let node = build(read_model, unit, Some(stem), context, advised)?;
    Ok(traced(node, passed))
}

fn filtered(input: &StrategyInput, own: &str, node: PlanNode) -> Result<PlanNode, PlanError> {
    match above_joins(input, &input.predicates, |_| Some(own))? {
        Some(filter) => step_filter(node, filter).map_err(|error| input.node_refused(error)),
        None => Ok(node),
    }
}

fn group_qualifier(own: &str, group: usize) -> String {
    match group {
        0 => own.to_owned(),
        _ => super::reserved(format_args!("group{group}"), own),
    }
}

#[derive(Clone, Copy)]
struct Over<'a> {
    read_model: &'a dyn ReadModel,
    input: &'a StrategyInput,
    sides: &'a [Side<'a>],
}

struct FormulaStep<'a> {
    over: Over<'a>,
    compute: &'a StrategyInput,
    metrics: &'a [GroupMetric],
    condition: Option<&'a PhysicalExpr>,
    fine: &'a [u32],
    parts: Option<&'a Parts>,
    context: &'a RequestContext,
    stem: &'a str,
}

impl FormulaStep<'_> {
    fn aggregated(&self, node: PlanNode) -> Result<(PlanNode, StrategyInput), PlanError> {
        let (compute, read_model) = (self.compute, self.over.read_model);
        let refused = |error| compute.refused(error);
        let node_refused = |error| compute.node_refused(error);
        let stem = Some(self.stem);
        let schema = node.common().output_schema.clone();
        let reads = Reads {
            over: self.over,
            schema: &schema,
        };
        let keys: Vec<BoundName> = (compute.names.iter())
            .filter(|name| groups_by(read_model, &compute.alias, name.bit))
            .cloned()
            .collect();
        let mut rows = Vec::new();
        for key in &keys {
            let (named, _) = field(stem, key.name.as_str()).map_err(refused)?;
            let read = reads.plain(key)?;
            rows.push((named, self.bucketed(key, read)?));
        }
        let mut groupings = keys
            .iter()
            .map(|key| field(stem, key.name.as_str()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(refused)?;
        for (key, _) in &compute.part_key {
            let read = key_over_holders(self.over.sides, key).map_err(refused)?;
            let (named, leaf) = field(stem, key.name()).map_err(refused)?;
            rows.push((named.clone(), read.ok_or_else(|| compute.broken())?));
            groupings.push((named, leaf));
        }
        let mut values = Vec::new();
        for metric in self.metrics {
            let bit = metric.bit;
            let binding = compute.bound_by_bit(read_model, bit)?;
            let (formula, _) = formula_field(stem, &binding.name).map_err(refused)?;
            let read = match self.parts {
                Some(parts) => reads.parts(parts)?,
                None => reads.metric(bit, self.condition)?,
            };
            rows.push((formula, read));
            values.push(binding);
        }
        let node = step_project(node, rows).map_err(node_refused)?;
        let nodes = self.nodes(&values);
        let mut aggregates = Vec::new();
        for (binding, metric) in values.iter().zip(self.metrics) {
            let (_, formula) = formula_field(stem, &binding.name).map_err(refused)?;
            let call = aggregated_over(metric.aggregation, formula).map_err(refused)?;
            let (name, _) = self.named(binding, &nodes)?;
            aggregates.push((name, call));
        }
        let node = step_agg(node, groupings, aggregates).map_err(node_refused)?;
        let node = self.typed(node, &values, &nodes)?;
        let (names, formula_nodes) = match self.condition {
            Some(_) => (keys, nodes),
            None => (keys.into_iter().chain(values).collect(), nodes),
        };
        let side = StrategyInput {
            names,
            conditions: BTreeMap::new(),
            formula_nodes,
            part_key: compute.part_key.clone(),
            ..self.over.input.clone()
        };
        Ok((node, side))
    }

    fn in_place(&self, node: PlanNode) -> Result<(PlanNode, StrategyInput), PlanError> {
        let (compute, read_model) = (self.compute, self.over.read_model);
        let refused = |error| compute.refused(error);
        let node_refused = |error| compute.node_refused(error);
        let stem = Some(self.stem);
        let schema = node.common().output_schema.clone();
        let reads = Reads {
            over: self.over,
            schema: &schema,
        };
        let kept = every_field(&node).map_err(refused)?;
        let mut rows = kept.clone();
        let mut values = Vec::new();
        for metric in self.metrics {
            let binding = compute.bound_by_bit(read_model, metric.bit)?;
            let (formula, _) = formula_field(stem, &binding.name).map_err(refused)?;
            rows.push((formula, reads.metric(metric.bit, self.condition)?));
            values.push(binding);
        }
        let node = step_project(node, rows).map_err(node_refused)?;
        let nodes = self.nodes(&values);
        let mut aggregates = Vec::new();
        for (binding, metric) in values.iter().zip(self.metrics) {
            let (_, formula) = formula_field(stem, &binding.name).map_err(refused)?;
            let call = aggregated_over(metric.aggregation, formula).map_err(refused)?;
            let (name, _) = self.named(binding, &nodes)?;
            aggregates.push((name, call));
        }
        let node = step_agg(node, kept, aggregates).map_err(node_refused)?;
        let node = self.typed(node, &values, &nodes)?;
        let names = match self.condition {
            Some(_) => Vec::new(),
            None => values,
        };
        let side = StrategyInput {
            names,
            conditions: BTreeMap::new(),
            formula_nodes: nodes,
            part_key: Vec::new(),
            ..self.over.input.clone()
        };
        Ok((node, side))
    }

    fn nodes(&self, values: &[BoundName]) -> Vec<FormulaNode> {
        let mut nodes = Vec::new();
        if let Some(condition) = self.condition {
            for value in values {
                FormulaNode::place(&mut nodes, value.name.clone(), Some(condition.clone()));
            }
        }
        nodes
    }

    fn named(
        &self,
        binding: &BoundName,
        nodes: &[FormulaNode],
    ) -> Result<(Name, PhysicalExpr), PlanError> {
        let stem = Some(self.stem);
        match (nodes.iter()).find(|node| node.measure == binding.name) {
            Some(node) => node.field(self.over.input, stem),
            None => field(stem, binding.name.as_str()),
        }
        .map_err(|error| self.compute.refused(error))
    }

    fn bucketed(&self, key: &BoundName, read: PhysicalExpr) -> Result<PhysicalExpr, PlanError> {
        if !self.fine.contains(&key.bit) {
            return Ok(read);
        }
        let data_type = (self.over.read_model.definition(key.bit)).and_then(|d| d.data_type);
        bucket(
            self.compute,
            self.context,
            &key.name,
            data_type.as_ref(),
            read,
        )
        .map_err(|error| self.compute.refused(error))
    }

    fn typed(
        &self,
        node: PlanNode,
        values: &[BoundName],
        nodes: &[FormulaNode],
    ) -> Result<PlanNode, PlanError> {
        typed_values(self.compute, node, |name| {
            let value = (values.iter()).find(|value| {
                self.named(value, nodes)
                    .is_ok_and(|(named, _)| &named == name)
            })?;
            let definition = self.over.read_model.definition(value.bit)?;
            let on_failure = self
                .parts
                .map_or(CastFailure::Error, |parts| parts.on_failure);
            Some(Declared {
                value_type: definition.data_type?,
                on_failure,
                aliases: Vec::new(),
            })
        })
    }
}

struct Reads<'a> {
    over: Over<'a>,
    schema: &'a Schema,
}

impl Reads<'_> {
    fn shown(&self) -> Result<Vec<(SemanticsName, PhysicalExpr)>, PlanError> {
        let input = self.over.input;
        let leaves = leaves_beyond(&input.names, &input.predicates);
        let bindings = input.names.iter().chain(leaves);
        shown_fields(self.over.read_model, input, bindings, |binding| {
            self.value(binding)
        })
    }

    fn given(&self) -> Result<Vec<(&FormulaNode, PhysicalExpr)>, PlanError> {
        let Over {
            read_model, input, ..
        } = self.over;
        let mut given = Vec::new();
        for formula_node in &input.formula_nodes {
            let binding = input.bound(read_model, formula_node.measure.as_str())?;
            let read = self.value_under(&binding, formula_node.condition.as_ref())?;
            given.push((formula_node, read));
        }
        Ok(given)
    }

    fn value(&self, binding: &BoundName) -> Result<PhysicalExpr, PlanError> {
        let condition = self.over.input.conditions.get(&binding.name);
        self.value_under(binding, condition)
    }

    fn value_under(
        &self,
        binding: &BoundName,
        condition: Option<&PhysicalExpr>,
    ) -> Result<PhysicalExpr, PlanError> {
        let Over {
            read_model, input, ..
        } = self.over;
        if !owns_metric(read_model, &input.alias, binding.bit) {
            return self.read(binding, condition);
        }
        let grouped = (self.over.sides.iter()).any(|side| side.input.names.contains(binding));
        match (condition, aggregation(read_model, binding.bit)) {
            (None, _) if grouped => self.plain(binding),
            (Some(condition), Some(_)) => self.read(binding, Some(condition)),
            (condition, _) => {
                let formula = self.metric(binding.bit, condition)?;
                self.typed(formula, binding.bit)
            }
        }
    }

    fn plain(&self, binding: &BoundName) -> Result<PhysicalExpr, PlanError> {
        held_read(self.over.sides, self.over.input, binding, plain_field)
    }

    fn read(
        &self,
        binding: &BoundName,
        condition: Option<&PhysicalExpr>,
    ) -> Result<PhysicalExpr, PlanError> {
        match condition {
            Some(condition) if !self.held(binding, condition) => {
                self.conditioned(binding, condition)
            }
            _ => self.plain(binding),
        }
    }

    fn metric(&self, bit: u32, outer: Option<&PhysicalExpr>) -> Result<PhysicalExpr, PlanError> {
        let Over {
            read_model, input, ..
        } = self.over;
        let binding = input.bound_by_bit(read_model, bit)?;
        let broken = || input.broken_tree(&binding);
        let signature = signature(read_model, &input.alias, bit).ok_or_else(broken)?;
        let mut by: Vec<(NamedReference, PhysicalExpr)> = Vec::new();
        for reach in signature
            .reaches(read_model, &input.alias, outer)
            .ok_or_else(broken)?
        {
            if by.iter().any(|(held, _)| held == &reach.column) {
                continue;
            }
            let condition = reach.read.condition.as_ref();
            let read = match reach.kind {
                ReachKind::Final => {
                    let binding = input.bound_by_bit(read_model, reach.read.bit)?;
                    self.read(&binding, condition)?
                }
                ReachKind::Formula => self.metric(reach.read.bit, condition)?,
                ReachKind::Leaf => self.leaf(reach.read.bit, condition)?,
            };
            by.push((reach.column, read));
        }
        replace_columns(signature.tree, &by).map_err(|error| input.refused(error))
    }

    fn parts(&self, parts: &Parts) -> Result<PhysicalExpr, PlanError> {
        let input = self.over.input;
        let refused = |error| input.refused(error);
        let mut by = Vec::new();
        for value in &parts.values {
            by.push((
                value.marker.clone(),
                self.leaf(value.read.bit, value.read.condition.as_ref())?,
            ));
        }
        for (column, _) in &parts.key.columns {
            let read = key_over_holders(self.over.sides, column).map_err(refused)?;
            by.push((column.clone(), read.ok_or_else(|| input.broken())?));
        }
        replace_columns(parts.key.per_part.clone(), &by).map_err(refused)
    }

    fn leaf(&self, bit: u32, condition: Option<&PhysicalExpr>) -> Result<PhysicalExpr, PlanError> {
        let binding = self.over.input.bound_by_bit(self.over.read_model, bit)?;
        let read = self.read(&binding, condition)?;
        self.typed(read, bit)
    }

    fn held(&self, binding: &BoundName, condition: &PhysicalExpr) -> bool {
        (self.over.sides.iter())
            .any(|side| side.input.conditions.get(&binding.name) == Some(condition))
    }

    fn conditioned(
        &self,
        binding: &BoundName,
        condition: &PhysicalExpr,
    ) -> Result<PhysicalExpr, PlanError> {
        let input = self.over.input;
        let held = |node: &&FormulaNode| {
            node.measure == binding.name && node.condition.as_ref() == Some(condition)
        };
        let holder = (self.over.sides.iter()).find_map(|side| {
            let node = side.input.formula_nodes.iter().find(held)?;
            Some((side, node))
        });
        let (side, node) = holder.ok_or_else(|| input.broken_tree(binding))?;
        let (_, read) = node
            .field(&side.input, Some(&side.qualifier))
            .map_err(|error| input.refused(error))?;
        Ok(read)
    }

    fn typed(&self, value: PhysicalExpr, bit: u32) -> Result<PhysicalExpr, PlanError> {
        let Over {
            read_model, input, ..
        } = self.over;
        let declared = (read_model.definition(bit)).and_then(|definition| definition.data_type);
        let Some(declared) = declared else {
            return Ok(value);
        };
        let actual = (self.schema.type_of(&value)).map_err(|error| input.node_refused(error))?;
        Ok(cast_where_differs(
            value,
            &actual.data_type,
            &declared,
            CastFailure::Error,
        ))
    }
}
