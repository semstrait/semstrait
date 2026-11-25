use std::collections::{BTreeMap, BTreeSet};
use std::mem;

use semstrait_common::SemanticsName;
use semstrait_common::Tree;
use semstrait_ir::SemanticRole;
use semstrait_ir::{Delivery, MultisetOp, Name, PhysicalExpr, PlanNode, Schema};
use semstrait_manifest::{RoleDefinition, SourceAlias};
use semstrait_sql::datatype::{promote, CanonicalTypeSystem, DataType};
use semstrait_sql::expr::{Call, CastFailure, Expr, Literal};

use super::{
    answering, by_semantics, combine, combined, computed, drill_across, every_field, holds,
    in_order, pruned, row_parts, split_after_aggregation, staged, trace, traced, volatile,
    Combined, FrameStages, OutputField,
};
use crate::advisory::advise;
use crate::dataset_strategy::typed;
use crate::formula::{finished, planned, Finished, Grouping, Linked, Naming};
use crate::frame::{Decided, Frame, FrameKind};
use crate::read_model::{is_unique, ReadModel};
use crate::steps::{
    catalog_call, condition_field, field, stemmed_entry, step_distinct, step_empty, step_filter,
    step_project, step_union, Aggregate,
};
use crate::strategy::{FormulaNode, StrategyInput, UnionsetArm};
use crate::{Advisory, BoundName, Combine, FramePath, PlanError, RequestContext};

pub(super) fn union(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    frame: &Decided,
    qualifier: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let alias = input.alias.as_str();
    let delivery = combined(frame);
    let empty = frame.empty_read.is_some();
    let pruned = (read_model.operands(&input.alias).iter())
        .filter(|branch| !(frame.operands.iter()).any(|read| &read.input.alias == *branch))
        .map(|branch| pruned(read_model, input, branch))
        .collect::<Result<Vec<_>, _>>()?;
    for (branch, lacks) in &frame.unservable {
        let (path, branch, lacks) = (input.path.clone(), branch.clone(), lacks.clone());
        advise(
            advised,
            Advisory::BranchNotServable {
                path,
                branch,
                lacks,
            },
        );
    }
    let mut own = StrategyInput {
        delivery,
        ..input.clone()
    };
    if delivery == Delivery::PartialAggregates {
        own.names.extend(own.key_leaves(read_model)?);
        own.part_key.clear();
    }
    let (written, on_measures) = match delivery {
        Delivery::FinalAggregates => (
            qualifier,
            split_after_aggregation(read_model, input, qualifier)?.0,
        ),
        _ => (Some(alias), None),
    };
    let readers = match &frame.read_again {
        Some(again) => (&again.readers[..], &again.served[..]),
        None => (&[][..], &[][..]),
    };
    let stacked = |branches: &[&Frame], reading: Reading| {
        let union = stack(
            read_model,
            input,
            branches,
            &own,
            written,
            on_measures.clone(),
            context,
            reading,
        )?;
        read_by_readers(read_model, &own, written, union, readers)
    };
    let built = if empty {
        let excluded: Vec<&Frame> = frame.excluded.iter().collect();
        Err(match stacked(&excluded, Reading::Excluded) {
            Ok(union) => union.common().output_schema.clone(),
            Err(_) => {
                let rows = no_branch(read_model, input, alias)?;
                let node = stage_over_rows(
                    read_model,
                    &own,
                    written,
                    context,
                    rows,
                    on_measures,
                    &mut Vec::new(),
                )?;
                node.common().output_schema.clone()
            }
        })
    } else {
        let operands: Vec<&Frame> = frame.operands.iter().collect();
        Ok(stacked(&operands, Reading::Kept(advised))?)
    };
    let trace = vec![trace(
        read_model,
        input,
        None,
        None,
        Some(delivery),
        None,
        pruned,
    )?];
    match built {
        Ok(union) => Ok(traced(union, trace)),
        Err(schema) => step_empty(schema, trace).map_err(|error| input.node_refused(error)),
    }
}

enum Reading<'a> {
    Kept(&'a mut Vec<Advisory>),
    Excluded,
}

#[allow(clippy::too_many_arguments)]
fn stack<'f>(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    branches: &[&'f Frame],
    own: &StrategyInput,
    written: Option<&str>,
    on_measures: Option<PhysicalExpr>,
    context: &RequestContext,
    mut reading: Reading,
) -> Result<PlanNode, PlanError> {
    let mut scratch = Vec::new();
    let (lenient, advised) = match &mut reading {
        Reading::Kept(advised) => (false, &mut **advised),
        Reading::Excluded => (true, &mut scratch),
    };
    let alias = input.alias.as_str();
    let delivery = own.delivery;
    let mut held: BTreeMap<SemanticsName, Vec<DataType>> = BTreeMap::new();
    let mut unvoted: BTreeMap<SemanticsName, Vec<DataType>> = BTreeMap::new();
    let mut built = Vec::new();
    let read_branch = |operand: &'f Frame, advised: &mut Vec<Advisory>| -> Result<_, PlanError> {
        let mut types = Vec::new();
        let rows = operand.gives_rows();
        let (operand, passed) = answering(read_model, operand)?;
        let branch_input = &operand.input;
        let finished = |node: PlanNode| -> Result<PlanNode, PlanError> {
            let node = traced(node, passed.clone());
            let node = match on_measures.clone() {
                Some(filter) => {
                    step_filter(node, filter).map_err(|error| own.node_refused(error))?
                }
                None => node,
            };
            formula_nodes(read_model, own, branch_input, written, node)
        };
        let decided = match &operand.kind {
            FrameKind::Compute(compute) => {
                let node =
                    computed::built(read_model, branch_input, compute, written, context, advised)?;
                return Ok(((finished(node)?, None), types));
            }
            FrameKind::Decided(decided) => decided,
        };
        if !rows {
            let node = match decided.decision.combine {
                Combine::Join | Combine::Cross | Combine::Stack => aggregated(
                    read_model,
                    own,
                    (branch_input, decided),
                    written,
                    context,
                    advised,
                )?,
                Combine::DrillAcross if delivery == Delivery::FinalAggregates => {
                    drill_across::joined(
                        read_model,
                        branch_input,
                        decided,
                        written,
                        context,
                        advised,
                    )?
                }
                Combine::Leaf | Combine::PassThrough | Combine::DrillAcross => {
                    return Err(branch_input.broken())
                }
            };
            return Ok(((finished(node)?, None), types));
        }
        let branch = StrategyInput {
            delivery: Delivery::Rows,
            ..operand.input.clone()
        };
        let node = combine(
            read_model,
            branch_input,
            decided,
            Some(alias),
            context,
            advised,
        )?;
        let schema = &node.common().output_schema;
        let combined = Combined::of(decided);
        let stages = FrameStages::of(
            read_model,
            &branch,
            Some(alias),
            context,
            schema,
            &combined,
            &decided.operands,
            &[],
        )?;
        volatile::copied(read_model, &branch, &stages, advised)?;
        for binding in &branch.names {
            let definition = branch.definition(read_model, binding)?;
            if planned(read_model, &branch, binding, &definition)? != SemanticRole::Measure {
                continue;
            }
            let (argument, _) =
                field(Some(alias), binding.name.as_str()).map_err(|e| branch.refused(e))?;
            let tree = (stages.read.row_level.iter())
                .find_map(|row| (row.name == argument).then_some(&row.tree))
                .ok_or_else(|| branch.broken_tree(binding))?;
            let argument_type =
                (schema.type_of(tree)).map_err(|error| branch.node_refused(error))?;
            types.push((
                binding.name.clone(),
                argument_type.data_type,
                operand.typed_by_no_branch(&binding.name),
            ));
        }
        Ok((
            (node, Some(((operand, decided), passed, branch, stages))),
            types,
        ))
    };
    for operand in branches.iter().copied() {
        match read_branch(operand, advised) {
            Ok((entry, types)) => {
                built.push(entry);
                for (name, data_type, by_no_branch) in types {
                    let votes = held.entry(name.clone()).or_default();
                    if by_no_branch {
                        unvoted.entry(name).or_default().push(data_type);
                    } else {
                        votes.push(data_type);
                    }
                }
            }
            Err(_) if lenient => {}
            Err(error) => return Err(error),
        }
    }
    let voted: BTreeSet<SemanticsName> = (held.iter())
        .filter(|(_, votes)| !votes.is_empty())
        .map(|(name, _)| name.clone())
        .collect();
    for (name, votes) in unvoted {
        if !voted.contains(&name) {
            held.entry(name).or_default().extend(votes);
        }
    }
    let refused = |error| input.node_refused(error);
    let inputs = (built.into_iter())
        .map(|(node, rows)| {
            let Some(((operand, decided), passed, branch, mut stages)) = rows else {
                return Ok(node);
            };
            let refused = |error| branch.refused(error);
            for measure in held.keys() {
                let (argument, _) = field(Some(alias), measure.as_str()).map_err(refused)?;
                let fill = voted.contains(measure) && operand.typed_by_no_branch(measure);
                if let Some((target, differs)) = least_restrictive(&held, measure) {
                    if fill || differs {
                        let row_level = mem::take(&mut stages.read.row_level);
                        stages.read.row_level = (row_level.into_iter())
                            .map(|mut row| {
                                if row.name == argument {
                                    row.tree = match fill {
                                        true => Expr::Literal(Literal::Null(target.clone())),
                                        false => Expr::Cast {
                                            input: Box::new(row.tree),
                                            target: target.clone(),
                                            on_failure: CastFailure::Error,
                                        },
                                    };
                                }
                                row
                            })
                            .collect();
                    }
                }
                let condition = condition_field(Some(alias), measure).map_err(refused)?;
                write(
                    &mut stages,
                    measure,
                    condition,
                    Expr::Literal(Literal::Boolean(true)),
                );
            }
            for name in &branch.null_filled {
                let data_type = match least_restrictive(&held, name) {
                    Some((target, _)) => target,
                    None => branch.null_fill_type(read_model, name)?,
                };
                let field = field(Some(alias), name.as_str()).map_err(refused)?;
                write(
                    &mut stages,
                    name,
                    field,
                    Expr::Literal(Literal::Null(data_type)),
                );
            }
            let reread = || {
                let mut scratch = Vec::new();
                combine(
                    read_model,
                    &operand.input,
                    decided,
                    Some(alias),
                    context,
                    &mut scratch,
                )
            };
            let built = stages.build(read_model, &branch, Some(alias), node, &reread)?;
            let rows = traced(built, passed);
            stage_over_rows(
                read_model,
                own,
                written,
                context,
                rows,
                on_measures.clone(),
                advised,
            )
        })
        .filter(|built| !(lenient && built.is_err()))
        .collect::<Result<Vec<_>, PlanError>>()?;
    if inputs.is_empty() {
        return Err(input.broken());
    }
    let order = (inputs.first().map(every_field).transpose())
        .map_err(|error| input.refused(error))?
        .unwrap_or_default();
    let inputs = (inputs.into_iter())
        .map(|branch| in_order(branch, order.clone()).map_err(refused))
        .collect::<Result<Vec<_>, PlanError>>()?;
    let unique = is_unique(read_model, &input.alias);
    let union = match <[PlanNode; 1]>::try_from(inputs) {
        Ok([branch]) if unique => step_distinct(branch, order).map_err(refused)?,
        Ok([branch]) => branch,
        Err(inputs) => {
            let op = if unique {
                MultisetOp::UnionDistinct
            } else {
                MultisetOp::UnionAll
            };
            step_union(inputs, op).map_err(refused)?
        }
    };
    Ok(union)
}

fn stage_over_rows(
    read_model: &dyn ReadModel,
    own: &StrategyInput,
    written: Option<&str>,
    context: &RequestContext,
    rows: PlanNode,
    on_measures: Option<PhysicalExpr>,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let delivery = own.delivery;
    if delivery == Delivery::Rows {
        return Ok(rows);
    }
    let alias = own.alias.as_str();
    let refused = |error| own.refused(error);
    let schema = &rows.common().output_schema;
    let rows_below = &Combined::Own(Delivery::Rows);
    let mut stages = FrameStages::of(
        read_model,
        own,
        written,
        context,
        schema,
        rows_below,
        &[],
        &[],
    )?;
    volatile::copied(read_model, own, &stages, advised)?;
    for name in &own.null_filled {
        let (_, read) = field(Some(alias), name.as_str()).map_err(refused)?;
        let (name_field, _) = field(written, name.as_str()).map_err(refused)?;
        (stages.read.row_level).push(OutputField::new(name.clone(), name_field, read));
    }
    let read =
        (stages.serving_every_read(written, delivery)).map_err(|error| own.refused(error))?;
    match delivery {
        Delivery::FinalAggregates => read.finish_branch(read_model, own, rows, on_measures),
        _ => (read.aggregate(rows, delivery)).map_err(|error| own.node_refused(error)),
    }
}

fn null_output(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    alias: &str,
    name: &SemanticsName,
    data_type: Option<DataType>,
) -> Result<OutputField, PlanError> {
    let (field_name, _) =
        field(Some(alias), name.as_str()).map_err(|error| input.refused(error))?;
    let data_type = match data_type {
        Some(data_type) => data_type,
        None => input.null_fill_type(read_model, name)?,
    };
    let null = Literal::Null(data_type);
    Ok(OutputField::new(
        name.clone(),
        field_name,
        Expr::Literal(null),
    ))
}

fn no_branch(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    alias: &str,
) -> Result<PlanNode, PlanError> {
    let arm = UnionsetArm::of(read_model, &input.alias).ok_or_else(|| input.broken())?;
    let refused = |error| input.refused(error);
    let mut fields = Vec::new();
    for binding in arm.names(read_model, input)? {
        let definition = input.definition(read_model, &binding)?;
        let measure = planned(read_model, input, &binding, &definition)? == SemanticRole::Measure;
        let data_type = measure
            .then(|| row_type(read_model, &input.path, &input.alias, binding.bit))
            .flatten();
        fields.push(null_output(
            read_model,
            input,
            alias,
            &binding.name,
            data_type,
        )?);
        if measure {
            let (condition, _) = condition_field(Some(alias), &binding.name).map_err(refused)?;
            fields.push(OutputField::new(
                binding.name.clone(),
                condition,
                Expr::Literal(Literal::Boolean(true)),
            ));
        }
    }
    for name in &input.null_filled {
        let null = null_output(read_model, input, alias, name, None)?;
        if !holds(&fields, &null.name) {
            fields.push(null);
        }
    }
    let nothing = step_empty(
        Schema {
            columns: Vec::new(),
        },
        Vec::new(),
    )
    .map_err(|error| input.node_refused(error))?;
    step_project(nothing, by_semantics(fields)).map_err(|error| input.node_refused(error))
}

fn row_type(
    read_model: &dyn ReadModel,
    path: &FramePath,
    source: &SourceAlias,
    bit: u32,
) -> Option<DataType> {
    match read_model.source_kind(source)? {
        RoleDefinition::Dataset { .. } => {
            let signature = read_model.signature(source, bit)?;
            let tree = read_model.expr_tree(source, &signature.expression_id, &[])?;
            let (argument, _) = row_parts(&tree)?;
            let physical = read_model.physical_source(source)?;
            let columns = (physical.columns.iter())
                .filter_map(|column| typed(&physical, path, column.name.clone()).ok())
                .collect();
            let typed = Schema { columns }.type_of(argument).ok()?;
            Some(typed.data_type)
        }
        RoleDefinition::Joinset { .. }
        | RoleDefinition::Unionset { .. }
        | RoleDefinition::Grainset { .. } => {
            let types: Vec<_> = (read_model.operands(source).iter())
                .filter(|operand| read_model.mask(operand).contains(bit))
                .filter_map(|operand| row_type(read_model, path, operand, bit))
                .collect();
            if types.is_empty() {
                return None;
            }
            promoted(&types)
        }
        _ => None,
    }
}

fn aggregated(
    read_model: &dyn ReadModel,
    union: &StrategyInput,
    (input, decided): (&StrategyInput, &Decided),
    written: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    staged(
        read_model,
        (input, decided),
        written,
        context,
        advised,
        |stages| {
            for name in &input.null_filled {
                let named = field(written, name.as_str());
                let (name_field, leaf) = named.map_err(|error| input.refused(error))?;
                let null = Expr::Literal(Literal::Null(input.null_fill_type(read_model, name)?));
                let read = &mut stages.read;
                (read.row_level).push(OutputField::new(name.clone(), name_field.clone(), null));
                if union.outputs(name) {
                    let grouped = [
                        &mut read.bucketing,
                        &mut read.groupings,
                        &mut read.derived,
                        &mut stages.output,
                    ];
                    let filled = OutputField::new(name.clone(), name_field, leaf);
                    for fields in grouped {
                        fields.push(filled.clone());
                    }
                }
            }
            Ok(())
        },
    )
}

fn formula_nodes(
    read_model: &dyn ReadModel,
    own: &StrategyInput,
    branch: &StrategyInput,
    written: Option<&str>,
    node: PlanNode,
) -> Result<PlanNode, PlanError> {
    let refused = |error| own.refused(error);
    let (nodes, places) = FormulaNode::of_frame(read_model, own)?;
    let (mut fields, mut dropped) = (Vec::new(), Vec::new());
    let (grouping, schema) = (Grouping::of(read_model, own)?, &node.common().output_schema);
    let leaves = own.leaves(read_model)?;
    for binding in own.names.iter().chain(&leaves) {
        let Some(places) = places.get(&binding.name) else {
            continue;
        };
        let calls = read_calls(
            read_model,
            own,
            &grouping,
            schema,
            written,
            binding,
            (None, &[]),
        )?;
        if calls.len() != places.len() {
            return Err(own.broken_tree(binding));
        }
        for ((entry, call), place) in calls.into_iter().zip(places) {
            let node = (place.and_then(|place| nodes.get(place)))
                .ok_or_else(|| own.broken_tree(binding))?;
            if (fields.iter()).any(|(held, _)| held == &entry) {
                continue;
            }
            let read = if branch.null_filled.contains(&node.measure) {
                null_node(read_model, own, schema, &call, &node.measure)?
            } else {
                let (name, read) = node.field(branch, written).map_err(refused)?;
                if !node.in_measure_field(branch) && !own.formula_nodes.contains(node) {
                    dropped.push(name);
                }
                read
            };
            fields.push((entry, read));
            if !own.outputs(&node.measure) {
                dropped.push(field(written, node.measure.as_str()).map_err(refused)?.0);
            }
        }
    }
    if branch.delivery == Delivery::PartialAggregates {
        for binding in own.names.iter().chain(&leaves) {
            let definition = own.definition(read_model, binding)?;
            if planned(read_model, own, binding, &definition)? != SemanticRole::Measure {
                continue;
            }
            let (plain, read) = field(written, binding.name.as_str()).map_err(refused)?;
            let calls = read_calls(
                read_model,
                own,
                &grouping,
                schema,
                written,
                binding,
                (None, &[]),
            )?;
            for (entry, _) in calls {
                if entry == plain || (fields.iter()).any(|(held, _)| held == &entry) {
                    continue;
                }
                fields.push((entry, read.clone()));
                dropped.push(plain.clone());
            }
        }
    }
    if fields.is_empty() {
        return Ok(node);
    }
    let passed = (every_field(&node).map_err(refused)?.into_iter())
        .filter(|(name, _)| !dropped.contains(name));
    step_project(node, passed.chain(fields).collect()).map_err(|error| own.node_refused(error))
}

pub(super) fn read_by_readers(
    read_model: &dyn ReadModel,
    own: &StrategyInput,
    written: Option<&str>,
    node: PlanNode,
    (readers, served): (&[BoundName], &[BoundName]),
) -> Result<PlanNode, PlanError> {
    if readers.is_empty() {
        return Ok(node);
    }
    let refused = |error| own.refused(error);
    let (nodes, places) = FormulaNode::of_readers(read_model, own, readers, served)?;
    let (grouping, schema) = (Grouping::of(read_model, own)?, &node.common().output_schema);
    let mut fields = Vec::new();
    for reader in readers {
        let Some(places) = places.get(&reader.name) else {
            continue;
        };
        let read = (None, served);
        let calls = read_calls(read_model, own, &grouping, schema, written, reader, read)?;
        if calls.len() != places.len() {
            return Err(own.broken_tree(reader));
        }
        for ((entry, _), place) in calls.into_iter().zip(places) {
            let node = (place.and_then(|place| nodes.get(place)))
                .ok_or_else(|| own.broken_tree(reader))?;
            if (fields.iter()).any(|(held, _)| held == &entry) {
                continue;
            }
            let (_, read) = node.field(own, written).map_err(refused)?;
            fields.push((entry, read));
        }
    }
    if fields.is_empty() {
        return Ok(node);
    }
    let passed = every_field(&node).map_err(refused)?;
    let fields = passed.into_iter().chain(fields).collect();
    step_project(node, fields).map_err(|error| own.node_refused(error))
}

fn read_calls(
    read_model: &dyn ReadModel,
    own: &StrategyInput,
    grouping: &Grouping,
    schema: &Schema,
    written: Option<&str>,
    binding: &BoundName,
    (through, served): (Option<&Linked>, &[BoundName]),
) -> Result<Vec<(Name, Aggregate)>, PlanError> {
    let mut calls = read_calls_under(
        read_model,
        (own, own),
        grouping,
        schema,
        written,
        binding,
        (through, served, None),
    )?;
    if through.is_some() {
        return Ok(calls);
    }
    let definition = own.definition(read_model, binding)?;
    if planned(read_model, own, binding, &definition)? != SemanticRole::Metric {
        return Ok(calls);
    }
    for node in (own.nodes_of(&binding.name)).filter(|n| n.finishes_apart(own)) {
        let apart = own.under(node);
        calls.extend(read_calls_under(
            read_model,
            (own, &apart),
            grouping,
            schema,
            written,
            binding,
            (None, served, Some(node.stem())),
        )?);
    }
    Ok(calls)
}

fn read_calls_under(
    read_model: &dyn ReadModel,
    (own, finishing): (&StrategyInput, &StrategyInput),
    grouping: &Grouping,
    schema: &Schema,
    written: Option<&str>,
    binding: &BoundName,
    (through, served, top): (Option<&Linked>, &[BoundName], Option<String>),
) -> Result<Vec<(Name, Aggregate)>, PlanError> {
    let naming = Naming {
        qualifier: written,
        numbered: 0,
    };
    let finished = finished(
        read_model,
        finishing,
        &[],
        grouping,
        schema,
        binding,
        naming,
    )?;
    let (tree, plain, mut links) = match finished {
        Finished::Step(tree) if grouping.finishes(read_model, finishing, binding)? => {
            let plain = matches!(tree, Expr::Function(Call::Aggregate(..)));
            (tree, plain, Vec::new())
        }
        Finished::Joined(tree, links) => (tree, false, links),
        Finished::TwoSteps(two_steps) => {
            let plain = matches!(two_steps.formula, Expr::Function(Call::Aggregate(..)));
            let finals = own.delivery == Delivery::FinalAggregates;
            (two_steps.formula, plain && !finals, two_steps.linked)
        }
        Finished::Step(_) => return Err(own.broken_tree(binding)),
    };
    let stem = through.map_or(Ok(top), |link| link.stem(read_model, own))?;
    if let Some(reader) = through {
        links.iter_mut().for_each(|link| link.through(reader));
    }
    let (mut calls, mut n) = (Vec::new(), 0);
    for node in tree.pre_order() {
        match node {
            Expr::Function(Call::Aggregate(call, _)) => {
                n += 1;
                let entry = stemmed_entry(written, stem.as_deref(), &binding.name, n, plain);
                calls.push((entry.map_err(|e| own.refused(e))?.0, call.clone()));
            }
            Expr::Column(column) => {
                let link = (links.iter())
                    .find(|link| link.field == *column)
                    .filter(|link| !served.contains(&link.binding));
                if let Some(link) = link {
                    let linked = &link.binding;
                    let read = read_calls(
                        read_model,
                        own,
                        grouping,
                        schema,
                        written,
                        linked,
                        (Some(link), served),
                    );
                    calls.extend(read?);
                }
            }
            _ => {}
        }
    }
    Ok(calls)
}

fn null_node(
    read_model: &dyn ReadModel,
    own: &StrategyInput,
    schema: &Schema,
    call: &Aggregate,
    measure: &SemanticsName,
) -> Result<PhysicalExpr, PlanError> {
    let null = Expr::Literal(Literal::Null(own.null_fill_type(read_model, measure)?));
    let args = (call.args().iter())
        .map(|arg| match arg {
            Expr::Literal(_) => arg.clone(),
            _ => null.clone(),
        })
        .collect();
    let name = call.function().declarations.canonical.name;
    let over_null = catalog_call(name, args).map_err(|error| own.refused(error))?;
    let typed = (schema.type_of(&over_null)).map_err(|error| own.node_refused(error))?;
    Ok(Expr::Literal(Literal::Null(typed.data_type)))
}

fn promoted(types: &[DataType]) -> Option<DataType> {
    Some(promote(&CanonicalTypeSystem, types)).filter(DataType::is_valid)
}

fn least_restrictive(
    held: &BTreeMap<SemanticsName, Vec<DataType>>,
    measure: &SemanticsName,
) -> Option<(DataType, bool)> {
    let types = held.get(measure)?;
    let differs =
        (types.split_first()).is_some_and(|(first, rest)| rest.iter().any(|other| other != first));
    let target = promoted(types)?;
    Some((target, differs))
}

fn write(
    stages: &mut FrameStages,
    semantics: &SemanticsName,
    (name, leaf): (Name, PhysicalExpr),
    tree: PhysicalExpr,
) {
    let row_level = &mut stages.read.row_level;
    if !holds(row_level, &name) {
        row_level.push(OutputField::new(semantics.clone(), name.clone(), tree));
        (stages.output).push(OutputField::new(semantics.clone(), name, leaf));
    }
}
