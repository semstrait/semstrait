use semstrait_common::SemanticsName;
use semstrait_common::{Tree, ValidateError};
use semstrait_ir::{Delivery, PhysicalExpr, SemanticRole};
use semstrait_manifest::{RoleDefinition, SourceAlias};
use semstrait_sql::expr::{Call, Expr, ExprOperationExt, Literal, NamedReference, Over};

use crate::advisory::advise;
use crate::formula::planned;
use crate::frame::{Decided, Frame, FrameKind};
use crate::grain::{is_dimension, SourceGrain};
use crate::read_model::{is_grainset, ReadModel, Relationship};
use crate::semi_additive::RankMarker;
use crate::steps::{window, Aggregate};
use crate::strategy::{qualifies, StrategyInput};
use crate::tree::aggregate_calls;
use crate::{Advisory, BoundName, Combine, FramePath, PlanError};

const EVERY_ROW: [&str; 2] = ["min", "max"];

type Chain = Vec<Step>;

#[derive(Clone)]
struct Step {
    join: FramePath,
    joinset: SourceAlias,
    end: SourceAlias,
    repeats: bool,
    serves: bool,
    branches: usize,
    aggregates: bool,
    identity: Option<(Vec<NamedReference>, bool)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowNumberMarker {
    join: FramePath,
    joinset: SourceAlias,
    route: Vec<(SourceAlias, bool)>,
    pub(crate) identity: Vec<NamedReference>,
    inferred: bool,
    unmarked: Vec<i64>,
}

enum ChainRead {
    Markers(Vec<RowNumberMarker>),
    Refused(FramePath),
}

pub(crate) type RowNumbers = (BoundName, Vec<Vec<RowNumberMarker>>);

pub(crate) struct FanOutWalk<'a> {
    pub(crate) read_model: &'a dyn ReadModel,
    pub(crate) grain: &'a [SemanticsName],
    pub(crate) input: &'a StrategyInput,
    pub(crate) operands: &'a [Frame],
}

impl FanOutWalk<'_> {
    pub(crate) fn of_frame(&self) -> Result<Vec<RowNumbers>, PlanError> {
        let input = self.input;
        (input.names.iter().chain(&input.leaves(self.read_model)?))
            .map(|binding| Ok((binding.clone(), self.of_binding(binding)?)))
            .collect()
    }

    fn of_binding(&self, binding: &BoundName) -> Result<Vec<Vec<RowNumberMarker>>, PlanError> {
        let (read_model, input) = (self.read_model, self.input);
        let relationship = match read_model.relationship(&input.alias) {
            Some(relationship) if input.delivery != Delivery::Rows => relationship,
            _ => return Ok(Vec::new()),
        };
        let tree = input.own_tree(read_model, binding, &[])?;
        (aggregate_calls(&tree))
            .map(|call| {
                if !reads_marker(call) {
                    return Ok(Vec::new());
                }
                let (results, _) = results(call.args());
                let read = (results.iter())
                    .map(|(position, result)| {
                        let read = self.of_result(&relationship, &binding.name, result)?;
                        Ok((*position, read))
                    })
                    .collect::<Result<Vec<_>, PlanError>>()?;
                let mut markers: Vec<RowNumberMarker> = Vec::new();
                for marker in read.iter().filter_map(|(_, marker)| marker.as_ref()) {
                    if !markers.contains(marker) {
                        markers.push(marker.clone());
                    }
                }
                for marker in &mut markers {
                    marker.unmarked = (read.iter())
                        .filter(|(_, read)| read.as_ref() != Some(&*marker))
                        .map(|(position, _)| *position)
                        .collect();
                }
                Ok(markers)
            })
            .collect()
    }

    fn of_result(
        &self,
        relationship: &Relationship,
        semantics: &SemanticsName,
        result: &[&NamedReference],
    ) -> Result<Option<RowNumberMarker>, PlanError> {
        let mut reads = Vec::new();
        for leaf in result {
            for chain in chains(self.read_model, self.input, self.operands, leaf)? {
                let read = self.of_chain(relationship, &chain)?;
                reads.push((branch(&chain), read));
            }
        }
        let as_they_are = |rows: &Option<Option<SourceAlias>>| {
            (reads.iter()).any(|(entered, read)| {
                (entered.is_none() || entered == rows)
                    && matches!(read, ChainRead::Markers(read) if read.is_empty())
            })
        };
        let (mut markers, mut refused) = (Vec::new(), None);
        for (_, read) in reads.iter().filter(|(entered, _)| !as_they_are(entered)) {
            match read {
                ChainRead::Markers(read) => {
                    for marker in read {
                        if !markers.contains(marker) {
                            markers.push(marker.clone());
                        }
                    }
                }
                ChainRead::Refused(join) => {
                    refused.get_or_insert_with(|| join.clone());
                }
            }
        }
        if let Some(path) = refused {
            let semantics = semantics.clone();
            return Err(PlanError::FanOutMarkerNotPlaceable { semantics, path });
        }
        let mut markers = markers.into_iter();
        match (markers.next(), markers.next()) {
            (_, Some(second)) => Err(PlanError::FanOutMarkerNotPlaceable {
                semantics: semantics.clone(),
                path: second.join,
            }),
            (read, None) => Ok(read),
        }
    }

    fn of_chain(&self, relationship: &Relationship, chain: &Chain) -> Result<ChainRead, PlanError> {
        let Self {
            read_model,
            grain,
            input,
            operands,
        } = *self;
        let Some(entered) = chain.first().map(|step| &step.end) else {
            return Ok(ChainRead::Markers(Vec::new()));
        };
        let (fanned, others): (Vec<&Frame>, Vec<&Frame>) =
            (operands.iter()).partition(|operand| &operand.input.alias == entered);
        let rows = (fanned.iter()).any(|operand| operand.input.delivery == Delivery::Rows);
        let condition = relationship.condition_names();
        let source = SourceGrain::of_decided(read_model, &input.alias, operands);
        let grouped = |name: &str| {
            condition.contains(&name)
                || (input.names.iter())
                    .find(|own| own.name.as_str() == name)
                    .is_some_and(|own| holds(grain, input, &source, own))
        };
        let removed = !rows
            && others.iter().all(|other| {
                let dimensions = (other.input.names.iter())
                    .filter(|name| is_dimension(read_model, name.bit))
                    .map(|name| name.name.as_str());
                other.input.delivery != Delivery::Rows
                    && (dimensions.chain(other.input.passed_up.iter().map(NamedReference::name)))
                        .all(grouped)
            });
        let counted = (chain.iter().enumerate().rev())
            .filter(|(at, step)| step.repeats && if *at == 0 { !removed } else { rows });
        let mut markers: Vec<RowNumberMarker> = Vec::new();
        for (at, step) in counted {
            let through = chain.iter().take(at).any(|above| above.branches > 0);
            let into = !step.serves && chain.iter().skip(at).any(|below| below.branches > 1);
            if through || into {
                return Ok(ChainRead::Refused(step.join.clone()));
            }
            let route = if step.serves { at + 1 } else { chain.len() };
            if markers.iter().any(|marker| marker.route.len() == route) {
                continue;
            }
            let steps = chain.get(..route).ok_or_else(|| input.broken())?;
            let (identity, inferred) = (steps.last())
                .and_then(|step| step.identity.clone())
                .ok_or_else(|| input.broken())?;
            let route: Vec<_> = (steps.iter())
                .map(|step| (step.end.clone(), step.repeats))
                .collect();
            markers.push(RowNumberMarker {
                join: step.join.clone(),
                joinset: step.joinset.clone(),
                route,
                identity,
                inferred,
                unmarked: Vec::new(),
            });
        }
        Ok(ChainRead::Markers(markers))
    }
}

impl RowNumberMarker {
    pub(crate) fn position(arguments: &[PhysicalExpr]) -> Option<PhysicalExpr> {
        results(arguments).1
    }

    pub(crate) fn or_unmarked(
        &self,
        once: PhysicalExpr,
        position: Option<&PhysicalExpr>,
    ) -> PhysicalExpr {
        let Some(position) = position else {
            return once;
        };
        (self.unmarked.iter()).fold(once, |read, unmarked| {
            read | position
                .clone()
                .eq(Expr::Literal(Literal::BigInt(*unmarked)))
        })
    }

    pub(crate) fn window(partition_by: Vec<PhysicalExpr>) -> Result<PhysicalExpr, ValidateError> {
        let over = Over {
            partition_by,
            order_by: Vec::new(),
            window_frame: None,
        };
        window("row_number", over)
    }

    fn determines(&self, chain: &Chain) -> bool {
        let shared = (self.route.iter().zip(chain))
            .take_while(|((end, _), step)| end == &step.end)
            .count();
        !(self.route.iter().skip(shared)).any(|(_, repeats)| *repeats)
    }
}

pub(crate) fn grain(read_model: &dyn ReadModel, names: &[BoundName]) -> Vec<SemanticsName> {
    (names.iter())
        .filter(|name| is_dimension(read_model, name.bit))
        .map(|name| name.name.clone())
        .collect()
}

pub(crate) fn placed(
    read_model: &dyn ReadModel,
    grain: &[SemanticsName],
    root: &Frame,
) -> Result<Vec<Advisory>, PlanError> {
    let mut advised = Vec::new();
    place(read_model, root, grain.to_vec(), &mut advised)?;
    Ok(advised)
}

fn place(
    read_model: &dyn ReadModel,
    frame: &Frame,
    mut held: Vec<SemanticsName>,
    advised: &mut Vec<Advisory>,
) -> Result<(), PlanError> {
    let decided = frame.decided();
    if let Some(decided) =
        decided.filter(|d| matches!(d.decision.combine, Combine::Join | Combine::Cross))
    {
        let input = &frame.input;
        if input.delivery == Delivery::PartialAggregates {
            rule_1(read_model, input, decided, &held)?;
        }
        for (_, marker) in marked(&decided.row_numbers).filter(|(_, marker)| marker.inferred) {
            let (unit, _) = marker.route.last().ok_or_else(|| input.broken())?;
            let advisory = Advisory::FanOut {
                operand: input.name(read_model, unit)?,
                path: marker.join.clone(),
            };
            advise(advised, advisory);
        }
        held.retain(|name| {
            !(decided.operands.iter())
                .all(|operand| (operand.input.names.iter()).any(|own| &own.name == name))
        });
    }
    if let Some(decided) = decided.filter(|d| d.decision.combine == Combine::Stack) {
        if !frame.gives_rows() {
            rule_2(read_model, frame, decided)?;
        }
    }
    (frame.frames().into_iter())
        .try_for_each(|operand| place(read_model, operand, held.clone(), advised))
}

fn rule_2(read_model: &dyn ReadModel, frame: &Frame, decided: &Decided) -> Result<(), PlanError> {
    let input = &frame.input;
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        let tree = input.own_tree(read_model, binding, &[])?;
        let calls = aggregate_calls(&tree).filter(|call| reads_marker(call));
        for (_, result) in calls.flat_map(|call| results(call.args()).0) {
            for branch in (decided.operands.iter()).filter(|branch| branch.gives_rows()) {
                let mut joins = Vec::new();
                for leaf in &result {
                    let entered = entered(read_model, frame, leaf)?;
                    if !(entered.iter()).any(|read| read.input.alias == branch.input.alias) {
                        continue;
                    }
                    for chain in inside(read_model, branch, &requalified(branch, leaf)?)? {
                        let repeated = chain.into_iter().find(|step| step.repeats);
                        joins.push(repeated.map(|step| step.join));
                    }
                }
                let repeated: Option<Vec<_>> = joins.into_iter().collect();
                if let Some(path) = repeated.and_then(|joins| joins.into_iter().next()) {
                    let semantics = binding.name.clone();
                    return Err(PlanError::FanOutMarkerNotPlaceable { semantics, path });
                }
            }
        }
    }
    Ok(())
}

fn rule_1(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    decided: &Decided,
    held: &[SemanticsName],
) -> Result<(), PlanError> {
    let operands = decided.operands.as_slice();
    let marked: Vec<_> = marked(&decided.row_numbers).collect();
    if marked.is_empty() {
        return Ok(());
    }
    let alias = input.alias.as_str();
    let own = (input.names.iter())
        .filter(|name| is_dimension(read_model, name.bit))
        .map(|name| NamedReference::qualified(alias, name.name.as_str()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| input.broken())?;
    let groupings = (own.iter().chain(&input.passed_up))
        .map(|leaf| {
            let grouping = input.bound(read_model, leaf.name())?;
            Ok((grouping, chains(read_model, input, operands, leaf)?))
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    let source = SourceGrain::of_decided(read_model, &input.alias, operands);
    for (semantics, marker) in marked {
        let relationship =
            (read_model.relationship(&marker.joinset)).ok_or_else(|| input.broken())?;
        let condition = relationship.condition_names();
        for (grouping, chains) in &groupings {
            let determined = condition.contains(&grouping.name.as_str())
                || chains.iter().all(|chain| marker.determines(chain))
                || holds(held, input, &source, grouping);
            if !determined {
                return Err(PlanError::FanOutMarkerNotPlaceable {
                    semantics: semantics.clone(),
                    path: marker.join.clone(),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn marker_reads(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    row_numbers: &[RowNumbers],
) -> Result<Vec<NamedReference>, PlanError> {
    let mut read: Vec<_> = (marked(row_numbers))
        .flat_map(|(_, marker)| marker.identity.iter().cloned())
        .collect();
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        let ranks = RankMarker::of_binding(read_model, input, operands, binding, &[])?;
        for rank in ranks.iter().flatten() {
            read.extend(rank.reads().filter_map(|name| rank.leaf(&name.name)));
        }
    }
    let mut reads = Vec::new();
    for column in read {
        if !reads.contains(&column) {
            reads.push(column);
        }
    }
    Ok(reads)
}

pub(crate) fn datasets(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    binding: &BoundName,
) -> Result<Vec<Vec<SourceAlias>>, PlanError> {
    if read_model.relationship(&input.alias).is_none() {
        return Ok(Vec::new());
    }
    let tree = input.own_tree(read_model, binding, &[])?;
    (aggregate_calls(&tree))
        .map(|call| {
            let mut datasets = Vec::new();
            for value in call.args().iter().flat_map(values) {
                for chain in chains(read_model, input, operands, value)? {
                    let step = chain.last().ok_or_else(|| input.broken())?;
                    if !datasets.contains(&step.end) {
                        datasets.push(step.end.clone());
                    }
                }
            }
            Ok(datasets)
        })
        .collect()
}

fn marked(row_numbers: &[RowNumbers]) -> impl Iterator<Item = (&SemanticsName, &RowNumberMarker)> {
    (row_numbers.iter()).flat_map(|(binding, calls)| {
        (calls.iter().flatten()).map(move |marker| (&binding.name, marker))
    })
}

fn holds(
    held: &[SemanticsName],
    input: &StrategyInput,
    source: &SourceGrain,
    name: &BoundName,
) -> bool {
    let asked =
        (held.contains(&name.name)).then(|| input.requested_granularity.get(&name.name).copied());
    source.kept(asked, name.bit)
}

pub(crate) fn reads_rows_as_they_are(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    argument: &PhysicalExpr,
) -> Result<bool, PlanError> {
    if read_model.relationship(&input.alias).is_none() {
        return Ok(true);
    }
    for value in values(argument) {
        let chains = chains(read_model, input, operands, value)?;
        if (chains.iter().flatten()).any(|step| step.repeats || step.aggregates) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn gives_aggregates(
    read_model: &dyn ReadModel,
    frame: &Frame,
    leaf: &NamedReference,
) -> Result<bool, PlanError> {
    if is_grainset(read_model, &frame.input.alias) {
        return Ok(true);
    }
    for branch in entered(read_model, frame.unit(), leaf)? {
        if gives_aggregates(read_model, branch, leaf)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn chains(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    leaf: &NamedReference,
) -> Result<Vec<Chain>, PlanError> {
    if qualifies(&input.alias, leaf) {
        let binding = input.bound(read_model, leaf.name())?;
        let tree = input.own_tree(read_model, &binding, &[])?;
        let mut walked = Vec::new();
        for value in values(&tree) {
            walked.extend(chains(read_model, input, operands, value)?);
        }
        return Ok(walked);
    }
    let holds = |operand: &&Frame| {
        let alias = &operand.input.alias;
        qualifies(alias, leaf)
            || (read_model.operand_closure(alias).iter())
                .any(|constituent| qualifies(constituent, leaf))
    };
    let operand = (operands.iter())
        .find(holds)
        .ok_or_else(|| input.broken())?;
    let demand = &operand.input;
    let serves = qualifies(&demand.alias, leaf) && {
        let binding = demand.bound(read_model, leaf.name())?;
        let definition = demand.definition(read_model, &binding)?;
        planned(read_model, demand, &binding, &definition)? == SemanticRole::Measure
    };
    let step = Step {
        join: input.path.clone(),
        joinset: input.alias.clone(),
        end: demand.alias.clone(),
        repeats: input.relationship(read_model)?.repeats(&demand.alias),
        serves,
        branches: entered(read_model, operand.unit(), leaf)?.len(),
        aggregates: gives_aggregates(read_model, operand, leaf)?,
        identity: identity(read_model, operand),
    };
    Ok((inside(read_model, operand, leaf)?.into_iter())
        .map(|mut chain| {
            chain.insert(0, step.clone());
            chain
        })
        .collect())
}

fn inside(
    read_model: &dyn ReadModel,
    frame: &Frame,
    leaf: &NamedReference,
) -> Result<Vec<Chain>, PlanError> {
    let input = &frame.input;
    let decided = match &frame.kind {
        FrameKind::Compute(compute) => {
            let mut walked = Vec::new();
            for group in &compute.groups {
                walked.extend(inside(read_model, &group.frame, leaf)?);
            }
            return Ok(walked);
        }
        FrameKind::Decided(decided) => decided,
    };
    match (decided.decision.combine, decided.operands.as_slice()) {
        (Combine::Join | Combine::Cross, operands) => chains(read_model, input, operands, leaf),
        (Combine::PassThrough, [read]) if qualifies(&input.alias, leaf) => {
            inside(read_model, read, &requalified(read, leaf)?)
        }
        (Combine::PassThrough, [read]) => inside(read_model, read, leaf),
        (Combine::PassThrough, _) => Err(input.broken()),
        (Combine::Stack, _) => {
            let mut walked = Vec::new();
            for branch in entered(read_model, frame, leaf)? {
                walked.extend(inside(read_model, branch, &requalified(branch, leaf)?)?);
            }
            Ok(walked)
        }
        (Combine::Leaf | Combine::DrillAcross, _) => Ok(vec![Chain::new()]),
    }
}

fn entered<'f>(
    read_model: &dyn ReadModel,
    frame: &'f Frame,
    leaf: &NamedReference,
) -> Result<Vec<&'f Frame>, PlanError> {
    let Some(decided) = frame
        .decided()
        .filter(|d| d.decision.combine == Combine::Stack)
    else {
        return Ok(Vec::new());
    };
    let bit = frame.input.bound(read_model, leaf.name())?.bit;
    Ok((decided.operands.iter())
        .filter(|branch| read_model.mask(&branch.input.alias).contains(bit))
        .collect())
}

fn branch(chain: &Chain) -> Option<Option<SourceAlias>> {
    let at = chain.iter().position(|step| step.branches > 0)?;
    Some(chain.get(at + 1).map(|next| next.joinset.clone()))
}

fn requalified(frame: &Frame, leaf: &NamedReference) -> Result<NamedReference, PlanError> {
    let input = &frame.input;
    NamedReference::qualified(input.alias.as_str(), leaf.name()).ok_or_else(|| input.broken())
}

fn reads_marker(call: &Aggregate) -> bool {
    let function = call.function().declarations.canonical.name;
    !call.distinct() && !EVERY_ROW.contains(&function)
}

fn results(arguments: &[PhysicalExpr]) -> (Vec<Numbered<'_>>, Option<PhysicalExpr>) {
    let [case @ Expr::Case { .. }] = arguments else {
        let whole = arguments.iter().flat_map(values).collect();
        return (vec![(1, whole)], None);
    };
    let mut results = Vec::new();
    let position = numbered(Some(case), &mut results);
    (results, Some(position))
}

type Numbered<'e> = (i64, Vec<&'e NamedReference>);

fn numbered<'e>(result: Option<&'e PhysicalExpr>, results: &mut Vec<Numbered<'e>>) -> PhysicalExpr {
    if let Some(Expr::Case { whens, else_ }) = result {
        return Expr::Case {
            whens: (whens.iter())
                .map(|(when, then)| (when.clone(), numbered(Some(then), results)))
                .collect(),
            else_: Some(Box::new(numbered(else_.as_deref(), results))),
        };
    }
    let position = results.last().map_or(1, |(last, _)| last + 1);
    results.push((position, result.map(values).unwrap_or_default()));
    Expr::Literal(Literal::BigInt(position))
}

pub(crate) fn values(tree: &PhysicalExpr) -> Vec<&NamedReference> {
    match tree {
        Expr::Column(column) => vec![column],
        Expr::Case { whens, else_ } => (whens.iter().map(|(_, then)| then))
            .chain(else_.as_deref())
            .flat_map(values)
            .collect(),
        Expr::Function(Call::Aggregate(call, _)) => call.args().iter().flat_map(values).collect(),
        node => node.children().into_iter().flat_map(values).collect(),
    }
}

fn identity(read_model: &dyn ReadModel, frame: &Frame) -> Option<(Vec<NamedReference>, bool)> {
    let unit = &frame.input.alias;
    let operands = frame
        .decided()
        .map_or(&[][..], |decided| decided.operands.as_slice());
    let keyset = match (read_model.source_kind(unit), operands) {
        (Some(RoleDefinition::Grainset { keysets, .. }), [child]) => {
            (keysets.first()).and_then(|first| read_model.keyset_named(&child.input.alias, first))
        }
        _ => read_model.keysets(unit).into_iter().next(),
    };
    let inferred = keyset.is_none();
    let columns = keyset.unwrap_or_else(|| read_model.mask(unit));
    let identity = (columns.set_positions().into_iter())
        .filter(|bit| is_dimension(read_model, *bit))
        .map(|bit| NamedReference::qualified(unit.as_str(), read_model.name_of(bit)?.as_str()))
        .collect::<Option<_>>()?;
    Some((identity, inferred))
}
