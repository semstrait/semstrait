use std::collections::{BTreeMap, BTreeSet};
use std::ops::BitAnd;
use std::{iter, mem};

use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_ir::{Delivery, PhysicalExpr, Schema, SemanticRole};
use semstrait_manifest::{KeptField, KeptLeaf, SemanticBitmask, SemanticDefinition, SourceAlias};
use semstrait_sql::datatype::TypeFamily;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal, NamedReference};
use semstrait_sql::functions::Additivity;

use crate::advisory::advise;
use crate::error::Lacking;
use crate::fan_out::{marker_reads, FanOutWalk, RowNumbers};
use crate::formula::{
    finished, planned, planned_as, pushes_down, reads_joined_rows, ComputedFields, Finished,
    Grouping, Naming,
};
use crate::grain::{additivity, SourceGrain};
use crate::grainset_metrics::{formula_groups, ConditionedMetric, FormulaGroup, GroupKind, Groups};
use crate::grainset_strategy::routed;
use crate::predicate::{above_joins, at_join, beyond, split_at_joins, AtJoin, JoinEnd};
use crate::read_model::{is_dataset, is_grainset, is_unique, literal_on, ReadModel, Relationship};
use crate::resolver::name_order;
use crate::routing::{fixed_granularity, Route};
use crate::steps::reference;
use crate::strategy::{qualifies, FormulaNode, Strategies, StrategyInput, UnionsetArm};
use crate::tree::{aggregate_calls, column_leaves};
use crate::{Advisory, BoundName, Combine, Decision, Operand, PlanError, Predicate};

pub(crate) struct Frame {
    pub(crate) input: StrategyInput,
    pub(crate) kind: FrameKind,
}

pub(crate) enum FrameKind {
    Decided(Box<Decided>),
    Compute(Compute),
}

pub(crate) struct Decided {
    pub(crate) decision: Decision,
    pub(crate) operands: Vec<Frame>,
    pub(crate) excluded: Vec<Frame>,
    pub(crate) empty_read: Option<Delivery>,
    pub(crate) unservable: Vec<Lacking>,
    pub(crate) above_join: Option<PhysicalExpr>,
    pub(crate) row_numbers: Vec<RowNumbers>,
    pub(crate) read_again: Option<ReadAgain>,
}

pub(crate) struct ReadAgain {
    pub(crate) own: StrategyInput,
    pub(crate) step1: Vec<StepRead>,
    pub(crate) readers: Vec<BoundName>,
    pub(crate) served: Vec<BoundName>,
}

pub(crate) struct StepRead {
    pub(crate) metrics: Vec<SemanticsName>,
    pub(crate) frame: Option<Frame>,
    pub(crate) readers: Vec<BoundName>,
    pub(crate) served: Vec<BoundName>,
}

pub(crate) type StepReadOf<'a> = (&'a StepRead, &'a StrategyInput, &'a Decided, bool);

pub(crate) struct Compute {
    pub(crate) groups: Vec<GroupFrame>,
    pub(crate) after_join: Vec<ConditionedMetric>,
}

pub(crate) struct GroupFrame {
    pub(crate) group: FormulaGroup,
    pub(crate) frame: Frame,
}

impl Frame {
    pub(crate) fn decided(&self) -> Option<&Decided> {
        match &self.kind {
            FrameKind::Decided(decided) => Some(decided.as_ref()),
            FrameKind::Compute(_) => None,
        }
    }

    pub(crate) fn combine(&self) -> Option<Combine> {
        self.decided().map(|decided| decided.decision.combine)
    }

    pub(crate) fn frames(&self) -> Vec<&Frame> {
        match &self.kind {
            FrameKind::Decided(decided) => decided.operands.iter().collect(),
            FrameKind::Compute(compute) => {
                compute.groups.iter().map(|group| &group.frame).collect()
            }
        }
    }

    pub(crate) fn unique_without_keyset(&self, read_model: &dyn ReadModel) -> Vec<Advisory> {
        let source = &self.input.alias;
        let own =
            (is_unique(read_model, source) && read_model.keysets(source).is_empty()).then(|| {
                Advisory::UniqueWithoutKeyset {
                    path: self.input.path.clone(),
                    semantic_source: self.input.semantic_source.clone(),
                }
            });
        let below = (self.frames().into_iter())
            .flat_map(|operand| operand.unique_without_keyset(read_model));
        own.into_iter().chain(below).collect()
    }

    pub(crate) fn tied_children(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<Vec<Advisory>, PlanError> {
        let mut advised = Vec::new();
        self.advise_ties(read_model, &mut advised)?;
        Ok(advised)
    }

    fn advise_ties(
        &self,
        read_model: &dyn ReadModel,
        advised: &mut Vec<Advisory>,
    ) -> Result<(), PlanError> {
        let input = &self.input;
        let decided = self
            .decided()
            .filter(|_| is_grainset(read_model, &input.alias));
        if let Some(decided) = decided {
            let children = read_model.operands(&input.alias);
            let routed = routed(read_model, &children, input)
                .ok()
                .and_then(|(_, _, routed)| routed);
            let followed = routed.filter(|routed| decided.follows(read_model, &routed.route));
            let name = |alias: &SourceAlias| read_model.name(alias).ok_or_else(|| input.broken());
            for (chosen, tied) in followed.into_iter().flat_map(|routed| routed.ties) {
                let tied = tied.iter().map(name).collect::<Result<Vec<_>, _>>()?;
                if let Some(advisory) = name_order(input.path.clone(), &name(&chosen)?, tied) {
                    advise(advised, advisory);
                }
            }
        }
        let step1 = (self.decided())
            .and_then(|decided| decided.read_again.as_ref())
            .into_iter()
            .flat_map(|again| &again.step1)
            .filter_map(|read| read.frame.as_ref());
        (self.frames().into_iter().chain(step1))
            .try_for_each(|operand| operand.advise_ties(read_model, advised))
    }

    pub(crate) fn unit(&self) -> &Self {
        match self.decided() {
            Some(decided) if decided.decision.combine == Combine::PassThrough => {
                decided.operands.first().map_or(self, Self::unit)
            }
            _ => self,
        }
    }

    pub(crate) fn gives_rows(&self) -> bool {
        let unit = self.unit();
        unit.combine() == Some(Combine::Leaf) || unit.input.delivery == Delivery::Rows
    }

    pub(crate) fn typed_by_no_branch(&self, name: &SemanticsName) -> bool {
        let read: Vec<&Frame> = match self.decided() {
            Some(decided) if decided.empty_read.is_some() && decided.excluded.is_empty() => {
                return true
            }
            Some(decided) if decided.empty_read.is_some() => decided.excluded.iter().collect(),
            _ => self.frames(),
        };
        let mut giving = (read.into_iter())
            .filter(|operand| operand.input.names.iter().any(|bound| &bound.name == name))
            .peekable();
        giving.peek().is_some() && giving.all(|operand| operand.typed_by_no_branch(name))
    }
}

impl ReadAgain {
    pub(crate) fn step1<'a>(&'a self, own: &'a Decided) -> Option<Vec<StepReadOf<'a>>> {
        (self.step1.iter())
            .map(|read| match &read.frame {
                Some(frame) => Some((read, &frame.input, frame.decided()?, true)),
                None => Some((read, &self.own, own, false)),
            })
            .collect()
    }
}

impl Decided {
    fn follows(&self, read_model: &dyn ReadModel, route: &Route) -> bool {
        match route {
            Route::Child(child) => self.decision.stand_in == read_model.name(child),
            Route::DrillAcross { chosen, .. } => (self.decision.operands.iter())
                .map(|operand| &operand.source)
                .eq(chosen.iter().map(|(child, _)| child)),
        }
    }
}

pub(crate) fn decide(
    strategies: &Strategies,
    read_model: &dyn ReadModel,
    grain: &[SemanticsName],
    mut input: StrategyInput,
) -> Result<Frame, PlanError> {
    if let Some(groups) = groups_to_compute(read_model, &mut input)? {
        return compute_frame(strategies, read_model, grain, input, groups);
    }
    let decision = strategies.dispatch(read_model, &input)?;
    let mut above = Vec::new();
    let mut excluded = Vec::new();
    let mut empty_read = None;
    let mut unservable = Vec::new();
    let mut row_numbers = Vec::new();
    let operands = match (&decision.stand_in, decision.combine) {
        (None, Combine::Leaf) => Vec::new(),
        (None, Combine::Join | Combine::Cross) => {
            let relationship = input.relationship(read_model)?;
            let ends = relationship.ends().map(|(operand, preserved)| {
                let served = (decision.operands.iter())
                    .find(|read| &read.source == operand)
                    .map_or(&[][..], |read| read.names.as_slice());
                JoinEnd {
                    operand,
                    served,
                    preserved,
                }
            });
            let mut moved: BTreeMap<&SourceAlias, Vec<Predicate>> = BTreeMap::new();
            let predicates = input.request_predicates(read_model)?;
            input.predicates.clear();
            for predicate in predicates {
                if input.reads_null_filled(&predicate) {
                    input.predicates.push(predicate);
                    continue;
                }
                for placed in at_join(&input, predicate, &ends)? {
                    match placed {
                        AtJoin::Into(operand, copy) => moved.entry(operand).or_default().push(copy),
                        AtJoin::Above(copy) => above.push(copy),
                        AtJoin::Kept(predicate) => input.predicates.push(predicate),
                    }
                }
            }
            let kept = input.columns(read_model)?;
            let pushed = pushed_down(read_model, &input)?;
            let mut formulas = pre_aggregated(read_model, &input, &relationship, &ends, &pushed)?;
            let decided = |conditions: &[NamedReference],
                           formulas: &PreAggregated,
                           parts: &BTreeSet<SemanticsName>| {
                let demand = |end: &JoinEnd| {
                    let predicates = moved.get(end.operand).cloned().unwrap_or_default();
                    let moved_only = |name: &SemanticsName| {
                        (predicates.iter().flat_map(|predicate| &predicate.leaves))
                            .any(|leaf| &leaf.name == name)
                            && !(kept.iter().flat_map(|(_, columns)| columns))
                                .chain(above.iter().flat_map(column_leaves))
                                .any(|column| column.name() == name.as_str())
                    };
                    let into = formulas.get(end.operand);
                    let pre_aggregated_only = |name: &SemanticsName| {
                        let reads = |column: &NamedReference| {
                            qualifies(end.operand, column) && column.name() == name.as_str()
                        };
                        into.is_some_and(|into| {
                            (kept.iter())
                                .filter(|(_, columns)| columns.iter().any(reads))
                                .all(|(binding, _)| into.contains_key(&binding.name))
                                && !above.iter().flat_map(column_leaves).any(reads)
                        })
                    };
                    let constituents = read_model.operand_closure(end.operand);
                    let condition = (conditions.iter())
                        .filter(|leaf| qualifies(end.operand, leaf))
                        .map(|leaf| leaf.name());
                    let passed_up = (conditions.iter())
                        .filter(|leaf| constituents.iter().any(|alias| qualifies(alias, leaf)))
                        .cloned()
                        .collect();
                    let served = (end.served.iter())
                        .filter(|name| !moved_only(name) && !pre_aggregated_only(name));
                    let bound = (served.map(SemanticsName::as_str).chain(condition))
                        .map(|name| input.bound(read_model, name))
                        .collect::<Result<Vec<_>, _>>()?;
                    let names = beyond(&[], &bound).into_iter().cloned().collect();
                    let mut demand = input.operand(
                        read_model,
                        end.operand,
                        names,
                        input.delivery,
                        predicates,
                        passed_up,
                    )?;
                    demand.pre_aggregated = into.cloned().unwrap_or_default();
                    Ok(demand)
                };
                let demands = ends.iter().map(demand).collect::<Result<Vec<_>, _>>()?;
                let units = (demands.iter())
                    .map(|demand| answering(strategies, read_model, demand))
                    .collect::<Result<Vec<_>, _>>()?;
                let deliveries =
                    end_deliveries(read_model, &input, &demands, &units, &pushed, parts)?;
                (demands.into_iter().zip(deliveries))
                    .map(|(demand, delivery)| {
                        let demand = StrategyInput { delivery, ..demand };
                        decide(strategies, read_model, grain, demand)
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            let mut conditions = Vec::new();
            for leaf in (relationship.trees().flat_map(column_leaves)).chain(&input.passed_up) {
                if !conditions.contains(leaf) {
                    conditions.push(leaf.clone());
                }
            }
            let mut parts = BTreeSet::new();
            let mut operands = decided(&conditions, &formulas, &parts)?;
            parts = walked_parts(read_model, &input, &operands, &pushed)?;
            if !parts.is_empty() {
                for into in formulas.values_mut() {
                    into.retain(|metric, _| !parts.contains(metric));
                }
                formulas.retain(|_, into| !into.is_empty());
                operands = decided(&conditions, &formulas, &parts)?;
            }
            loop {
                let walk = FanOutWalk {
                    read_model,
                    grain,
                    input: &input,
                    operands: &operands,
                };
                let marks = walk.of_frame()?;
                let outside: Vec<_> = (marker_reads(read_model, &input, &operands, &marks)?)
                    .into_iter()
                    .filter(|column| !conditions.contains(column))
                    .collect();
                if outside.is_empty() {
                    row_numbers = marks;
                    break operands;
                }
                conditions.extend(outside);
                operands = decided(&conditions, &formulas, &parts)?;
            }
        }
        (None, Combine::Stack) => {
            let mut pruned = Vec::new();
            if let Some(arm) = UnionsetArm::of(read_model, &input.alias) {
                unservable = arm.unservable(read_model, &input)?;
                unservable.retain(|(branch, _)| {
                    (decision.operands.iter())
                        .all(|operand| read_model.name(&operand.source).as_ref() != Some(branch))
                });
                if !unservable.is_empty() {
                    let branches = arm.branches(read_model, &input)?;
                    pruned = (branches.unservable.iter())
                        .map(|(branch, _)| arm.operand(read_model, branch, &branches.names))
                        .collect();
                }
            }
            let placed = place(&mut input);
            let read = (&decision.operands[..], &pruned[..]);
            let (delivery, finer) = branch_delivery(read_model, &input, read)?;
            let again = read_again(strategies, read_model, grain, &input, &placed, &finer)?;
            if let Some(decided) = again {
                return Ok(Frame {
                    input,
                    kind: FrameKind::Decided(decided),
                });
            }
            let aggregating = match delivery {
                Delivery::Rows => None,
                _ => Some(Aggregating::of(read_model, &input, &placed)?),
            };
            let branch = |operand: &Operand| {
                let filled = input.filling(operand);
                let rows = input.branch_input(read_model, &filled, &placed, delivery)?;
                let composed =
                    (aggregating.as_ref()).filter(|_| !is_dataset(read_model, &operand.source));
                let Some(aggregating) = composed else {
                    return decide(strategies, read_model, grain, rows);
                };
                let aggregates = aggregating.operand(read_model, &input, &filled, delivery)?;
                let frame = decide(strategies, read_model, grain, aggregates)?;
                if frame.gives_rows() {
                    return decide(strategies, read_model, grain, rows);
                }
                Ok(frame)
            };
            if decision.operands.is_empty() {
                empty_read = Some(delivery);
                excluded = (strategies.dispatch(read_model, &input))
                    .map(|folded| {
                        folded
                            .operands
                            .iter()
                            .filter_map(|operand| branch(operand).ok())
                            .collect()
                    })
                    .unwrap_or_default();
            }
            (decision.operands.iter())
                .map(branch)
                .collect::<Result<_, _>>()?
        }
        (stand_in, Combine::PassThrough) => {
            let read = match (stand_in, decision.operands.as_slice()) {
                (Some(unit), []) => read_model.alias(unit.as_str()),
                (None, [operand]) => Some(operand.source.clone()),
                _ => None,
            };
            let read = read.ok_or_else(|| input.broken())?;
            let mut demand = passed_through(read_model, &input, &read)?;
            if is_grainset(read_model, &input.alias) {
                let grainset = SourceGrain::of_demand(read_model, &input);
                fix(&mut demand, &grainset, grainset.granularity());
            }
            input.predicates.clear();
            vec![decide(strategies, read_model, grain, demand)?]
        }
        (Some(_), _) => return Err(input.broken()),
        (None, Combine::DrillAcross) => {
            let (placed, joined) = split_at_joins(read_model, &mut input)?;
            let reader = |name: &SemanticsName| {
                (decision.operands.iter())
                    .find(|operand| operand.names.contains(name))
                    .map(|operand| operand.source.as_str())
            };
            above.extend(above_joins(&input, &joined, reader)?);
            let output: Vec<_> = (input.names.iter())
                .chain(joined.iter().flat_map(|predicate| &predicate.leaves))
                .map(|binding| &binding.name)
                .collect();
            let grainset = SourceGrain::of_demand(read_model, &input);
            let fixed = (decision.operands.first())
                .and_then(|first| fixed_granularity(read_model, &grainset, &first.source));
            let child = |operand: &Operand| {
                let predicates = placed.clone();
                let names = (operand.names.iter())
                    .filter(|name| output.contains(name))
                    .map(|name| input.bound(read_model, name.as_str()))
                    .collect::<Result<_, _>>()?;
                let mut demand = input.operand(
                    read_model,
                    &operand.source,
                    names,
                    Delivery::FinalAggregates,
                    predicates,
                    Vec::new(),
                )?;
                demand.conditions = (input.conditions.iter())
                    .filter(|(measure, _)| operand.names.contains(measure))
                    .map(|(measure, condition)| (measure.clone(), condition.clone()))
                    .collect();
                demand.formula_nodes = (input.formula_nodes.iter())
                    .filter(|node| operand.names.contains(&node.measure))
                    .cloned()
                    .collect();
                demand.part_key.clone_from(&input.part_key);
                fix(&mut demand, &grainset, fixed);
                decide(strategies, read_model, grain, demand)
            };
            (decision.operands.iter())
                .map(child)
                .collect::<Result<_, _>>()?
        }
    };
    Ok(Frame {
        input,
        kind: FrameKind::Decided(Box::new(Decided {
            decision,
            operands,
            excluded,
            empty_read,
            unservable,
            above_join: above.into_iter().reduce(BitAnd::bitand),
            row_numbers,
            read_again: None,
        })),
    })
}

fn read_again(
    strategies: &Strategies,
    read_model: &dyn ReadModel,
    grain: &[SemanticsName],
    input: &StrategyInput,
    placed: &[Predicate],
    finer: &[BoundName],
) -> Result<Option<Box<Decided>>, PlanError> {
    if finer.is_empty() {
        return Ok(None);
    }
    let grains = step_grains(read_model, input)?;
    if grains.is_empty() {
        return Ok(None);
    }
    let predicates: Vec<_> = placed.iter().chain(&input.predicates).cloned().collect();
    let bits = |names: &[BoundName]| SemanticBitmask::from_positions(names.iter().map(|n| n.bit));
    let below: Vec<_> = (predicates.iter())
        .filter(|predicate| predicate.in_branches())
        .cloned()
        .collect();
    let mut reads: Vec<(&BTreeSet<SemanticsName>, Vec<BoundName>, ComputedFields)> = Vec::new();
    for (dimensions, metrics) in &grains {
        let mut stepped = Vec::new();
        for metric in metrics {
            let key = part_key_of(read_model, input, metric)?;
            if key.is_empty() {
                stepped.push(metric.clone());
                continue;
            }
            let same = |(held, _, keyed): &&mut (_, _, ComputedFields)| {
                *held == dimensions
                    && keyed.len() == key.len()
                    && (keyed.iter()).all(|(_, tree)| key.iter().any(|(_, own)| own == tree))
            };
            match reads.iter_mut().find(same) {
                Some((_, held, _)) => held.push(metric.clone()),
                None => reads.push((dimensions, vec![metric.clone()], key)),
            }
        }
        if !stepped.is_empty() {
            reads.push((dimensions, stepped, Vec::new()));
        }
    }
    let mut step1 = Vec::new();
    for (at, (dimensions, metrics, key)) in reads.iter().enumerate() {
        let others: Vec<_> = (reads.iter().enumerate())
            .filter(|(other, _)| *other != at)
            .flat_map(|(_, (_, others, _))| others)
            .collect();
        let own: Vec<_> = metrics.iter().collect();
        let held = |of: &[&BoundName], linked: &SemanticDefinition| {
            of.iter().any(|metric| metric.bit == linked.bit_position)
        };
        let (mut apart, mut served) = (Vec::new(), Vec::new());
        for name in &input.names {
            if others.contains(&name) {
                apart.push(name.clone());
                continue;
            }
            if metrics.contains(name) {
                continue;
            }
            let linked = input.linked(read_model, name)?;
            if !linked.iter().any(|linked| held(&others[..], linked)) {
                continue;
            }
            apart.push(name.clone());
            let links = linked.into_iter().filter(|linked| held(&own[..], linked));
            served.extend(links.map(BoundName::of));
        }
        let mut names: Vec<_> = (input.names.iter())
            .filter(|name| !apart.contains(name))
            .cloned()
            .collect();
        for metric in served {
            if !names.contains(&metric) {
                names.push(metric);
            }
        }
        let (mut through, mut kept) = (Vec::new(), Vec::new());
        let mut at = 0;
        while let Some(name) = names.get(at).cloned() {
            let links: Vec<_> = (input.linked(read_model, &name)?.into_iter())
                .filter(|linked| {
                    linked.role_definition.aggregation().is_some()
                        && planned_as(read_model, &input.alias, linked)
                            == Some(SemanticRole::Metric)
                })
                .map(BoundName::of)
                .collect();
            if !finer.contains(&name) || links.is_empty() {
                at += 1;
                continue;
            }
            through.push(names.remove(at));
            for link in links {
                if !kept.contains(&link) {
                    kept.push(link.clone());
                }
                if !names.contains(&link) {
                    names.push(link);
                }
            }
        }
        let (nodes, places) = FormulaNode::of_readers(read_model, input, &through, &kept)?;
        let read_nodes =
            (places.values().flatten().flatten()).filter_map(|place| nodes.get(*place));
        for node in read_nodes {
            let measure = input.bound(read_model, node.measure.as_str())?;
            if !names.contains(&measure) {
                names.push(measure);
            }
        }
        let mut requested_granularity = input.requested_granularity.clone();
        for name in (dimensions.iter()).filter(|_| key.is_empty()) {
            requested_granularity.remove(name);
            if !names.iter().any(|held| &held.name == name) {
                names.push(input.bound(read_model, name.as_str())?);
            }
        }
        let demanded = (input.demanded.difference(&bits(&apart)))
            .difference(&bits(&through))
            .union(&bits(&names));
        let read = StrategyInput {
            demanded,
            names,
            predicates: below.clone(),
            requested_granularity,
            part_key: key.clone(),
            formula_nodes: nodes,
            ..input.clone()
        };
        let Some(frame) = final_stack(decide(strategies, read_model, grain, read)) else {
            return Ok(None);
        };
        step1.push(StepRead {
            metrics: metrics.iter().map(|metric| metric.name.clone()).collect(),
            frame: Some(frame),
            readers: through,
            served: kept,
        });
    }
    let mut readers = Vec::new();
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        if finer.contains(binding) || readers.contains(binding) {
            continue;
        }
        let linked = input.linked(read_model, binding)?;
        if (linked.iter())
            .any(|linked| finer.iter().any(|metric| metric.bit == linked.bit_position))
        {
            readers.push(binding.clone());
        }
    }
    let apart = |name: &BoundName| finer.contains(name) || readers.contains(name);
    let (moved, predicates): (Vec<_>, Vec<_>) = (predicates.into_iter())
        .partition(|predicate| !predicate.in_branches() && predicate.leaves.iter().any(apart));
    let mut names: Vec<_> = (input.names.iter())
        .filter(|name| !apart(name))
        .cloned()
        .collect();
    for leaf in moved.iter().flat_map(|predicate| &predicate.leaves) {
        if !apart(leaf) && !names.contains(leaf) {
            names.push(leaf.clone());
        }
    }
    let (nodes, _) = FormulaNode::of_readers(read_model, input, &readers, finer)?;
    for node in &nodes {
        let measure = input.bound(read_model, node.measure.as_str())?;
        if !names.contains(&measure) {
            names.push(measure);
        }
    }
    let added: Vec<_> = (names.iter())
        .filter(|name| !input.names.contains(name))
        .cloned()
        .collect();
    let demanded = (input.demanded.difference(&bits(finer)))
        .difference(&bits(&readers))
        .union(&bits(&added));
    let mut own = StrategyInput {
        demanded,
        names,
        predicates,
        formula_nodes: nodes,
        ..input.clone()
    };
    if !aggregates(read_model, &own)? {
        let first = step1.first_mut().and_then(|read| read.frame.take());
        let Frame { input: read, kind } = first.ok_or_else(|| input.broken())?;
        let again = ReadAgain {
            own: read,
            step1,
            readers: Vec::new(),
            served: Vec::new(),
        };
        return Ok(read_as(kind, again));
    }
    let grouped: BTreeSet<_> = (grains.iter())
        .flat_map(|(dimensions, _)| dimensions)
        .collect();
    let held = own.names.len();
    for read in step1.iter().filter_map(|read| read.frame.as_ref()) {
        for leaf in read.input.read_leaves(read_model)? {
            let leaf = input.bound(read_model, &leaf)?;
            if !grouped.contains(&leaf.name) && !own.names.contains(&leaf) {
                own.names.push(leaf);
            }
        }
    }
    own.demanded = own.demanded.union(&bits(&own.names[held..]));
    let Some(Frame { input: own, kind }) = final_stack(decide(strategies, read_model, grain, own))
    else {
        return Ok(None);
    };
    let again = ReadAgain {
        own,
        step1,
        readers,
        served: finer.to_vec(),
    };
    Ok(read_as(kind, again))
}

fn aggregates(read_model: &dyn ReadModel, input: &StrategyInput) -> Result<bool, PlanError> {
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        let definition = input.definition(read_model, binding)?;
        let role = planned(read_model, input, binding, &definition)?;
        if matches!(role, SemanticRole::Measure | SemanticRole::Metric) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn read_as(kind: FrameKind, again: ReadAgain) -> Option<Box<Decided>> {
    let FrameKind::Decided(mut decided) = kind else {
        return None;
    };
    decided.read_again = Some(again);
    Some(decided)
}

fn final_stack(frame: Result<Frame, PlanError>) -> Option<Frame> {
    let frame = frame.ok()?;
    let decided = frame.decided()?;
    let finals = (decided.operands.iter())
        .all(|operand| operand.input.delivery == Delivery::FinalAggregates);
    let stacked = decided.decision.combine == Combine::Stack
        && decided.empty_read.is_none()
        && decided.read_again.is_none()
        && !decided.operands.is_empty();
    (stacked && finals).then_some(frame)
}

fn groups_to_compute(
    read_model: &dyn ReadModel,
    input: &mut StrategyInput,
) -> Result<Option<Groups>, PlanError> {
    if input.delivery != Delivery::FinalAggregates {
        return Ok(None);
    }
    formula_groups(read_model, input)
}

fn compute_frame(
    strategies: &Strategies,
    read_model: &dyn ReadModel,
    grain: &[SemanticsName],
    mut input: StrategyInput,
    groups: Groups,
) -> Result<Frame, PlanError> {
    for parts in (groups.groups.iter()).filter_map(FormulaGroup::parts) {
        let numeric = (read_model.definition(parts.metric.bit))
            .and_then(|definition| definition.data_type)
            .is_some_and(|data_type| data_type.family() == Some(TypeFamily::Number));
        if !numeric {
            return Err(PlanError::UngroupedMetricFormula {
                path: input.path.clone(),
                metric: input.bound_by_bit(read_model, parts.metric.bit)?.name,
                name: input.bound_by_bit(read_model, parts.ungrouped)?.name,
            });
        }
    }
    let (before, after) = split_at_joins(read_model, &mut input)?;
    input.predicates = after;
    let mut frames = Vec::new();
    for (index, group) in groups.groups.into_iter().enumerate() {
        let held = (group.expansion.under(read_model)).ok_or_else(|| input.broken())?;
        let names = (held.names.iter())
            .map(|bit| input.bound_by_bit(read_model, *bit))
            .collect::<Result<Vec<_>, _>>()?;
        let leaves = before.iter().flat_map(|predicate| &predicate.leaves);
        let demanded =
            SemanticBitmask::from_positions(names.iter().chain(leaves).map(|name| name.bit));
        let mut path = input.path.clone();
        path.0.push(DataKindName::new(format!(
            "{}#group{index}",
            input.semantic_source
        )));
        let mut demand = StrategyInput {
            path,
            names,
            demanded,
            predicates: before.clone(),
            conditions: held.conditions,
            formula_nodes: held.nodes,
            null_filled: Vec::new(),
            stand_ins: Vec::new(),
            formula_grain: input.formula_grain || matches!(group.kind, GroupKind::FormulaGrain(_)),
            ..input.clone()
        };
        for name in (demand.names.iter()).filter(|name| group.fine().contains(&name.bit)) {
            demand.requested_granularity.remove(&name.name);
        }
        if let Some(parts) = group.parts() {
            demand.part_key.extend(parts.key.columns.iter().cloned());
        }
        let frame = decide(strategies, read_model, grain, demand)?;
        frames.push(GroupFrame { group, frame });
    }
    Ok(Frame {
        input,
        kind: FrameKind::Compute(Compute {
            groups: frames,
            after_join: groups.after_join,
        }),
    })
}

pub(crate) fn fix(demand: &mut StrategyInput, grainset: &SourceGrain, fixed: Option<Granularity>) {
    let Some(fixed) = fixed else {
        return;
    };
    for name in &demand.names {
        let asked = demand.requested_granularity.get(&name.name).copied();
        if let Some(granularity) = grainset.fixed(name.bit, asked, fixed) {
            demand
                .requested_granularity
                .insert(name.name.clone(), granularity);
        }
    }
}

fn passed_through(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    read: &SourceAlias,
) -> Result<StrategyInput, PlanError> {
    let predicates = input.request_predicates(read_model)?;
    let mut demand = input
        .operand(
            read_model,
            read,
            input.names.clone(),
            input.delivery,
            predicates,
            input.passed_up.clone(),
        )?
        .null_filling(read_model, input.null_filled.clone())?;
    demand.conditions = input.conditions.clone();
    demand.formula_nodes = input.formula_nodes.clone();
    demand.part_key.clone_from(&input.part_key);
    Ok(demand)
}

fn answering(
    strategies: &Strategies,
    read_model: &dyn ReadModel,
    demand: &StrategyInput,
) -> Result<Option<StrategyInput>, PlanError> {
    let rows = StrategyInput {
        delivery: Delivery::Rows,
        ..demand.clone()
    };
    let Some(unit) = strategies.dispatch(read_model, &rows)?.stand_in else {
        return Ok(None);
    };
    let read = read_model
        .alias(unit.as_str())
        .ok_or_else(|| demand.broken())?;
    passed_through(read_model, demand, &read).map(Some)
}

pub(crate) fn place(input: &mut StrategyInput) -> Vec<Predicate> {
    let (placed, kept) =
        (mem::take(&mut input.predicates).into_iter()).partition::<Vec<_>, _>(|predicate| {
            predicate.in_branches() && !input.reads_null_filled(predicate)
        });
    input.predicates = kept;
    placed
}

pub(crate) struct Aggregating {
    nodes: Vec<FormulaNode>,
    stepped: BTreeSet<SemanticsName>,
    keyed: BTreeSet<SemanticsName>,
    above: BTreeSet<SemanticsName>,
    predicates: Vec<Predicate>,
}

impl Aggregating {
    pub(crate) fn of(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        placed: &[Predicate],
    ) -> Result<Self, PlanError> {
        let own = (input.predicates.iter()).filter(|predicate| predicate.in_branches());
        let predicates = (placed.iter()).chain(own).cloned().collect();
        let (nodes, _) = FormulaNode::of_frame(read_model, input)?;
        let stepped = step_dimensions(read_model, input)?;
        let keyed = (input.key_leaves(read_model)?.into_iter())
            .map(|leaf| leaf.name)
            .collect();
        let above = (input.predicates.iter())
            .filter(|predicate| !predicate.in_branches())
            .flat_map(|predicate| &predicate.leaves)
            .map(|leaf| leaf.name.clone())
            .collect();
        Ok(Self {
            nodes,
            stepped,
            keyed,
            above,
            predicates,
        })
    }

    pub(crate) fn operand(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        operand: &Operand,
        delivery: Delivery,
    ) -> Result<StrategyInput, PlanError> {
        let nodes: Vec<_> = (self.nodes.iter())
            .filter(|node| operand.names.contains(&node.measure))
            .cloned()
            .collect();
        let conditions = input.lowered_onto(read_model, &operand.names)?;
        let outputs = (operand.names.iter())
            .filter(|name| {
                input.outputs(name)
                    || nodes.iter().any(|node| &node.measure == *name)
                    || self.stepped.contains(*name)
                    || (delivery == Delivery::PartialAggregates && self.keyed.contains(*name))
                    || self.above.contains(*name)
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut aggregates = input.branch_input(
            read_model,
            &Operand {
                names: outputs,
                ..operand.clone()
            },
            &self.predicates,
            delivery,
        )?;
        aggregates.conditions = conditions;
        if delivery == Delivery::FinalAggregates && !self.keyed.is_empty() {
            aggregates.part_key.clone_from(&input.part_key);
        }
        for node in (nodes.iter()).filter(|n| n.index == 1 && !input.outputs(&n.measure)) {
            aggregates = aggregates.under(node);
        }
        aggregates.formula_nodes = nodes;
        Ok(aggregates)
    }
}

fn branch_delivery(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    (branches, pruned): (&[Operand], &[Operand]),
) -> Result<(Delivery, Vec<BoundName>), PlanError> {
    if is_unique(read_model, &input.alias) || !input.pre_aggregated.is_empty() {
        return Ok((Delivery::Rows, Vec::new()));
    }
    let grouping = Grouping::of(read_model, input)?;
    let mut finer = Vec::new();
    for metric in unfinished(read_model, input, &grouping)? {
        if pushes_down(read_model, input, &metric)? {
            return Ok((Delivery::Rows, Vec::new()));
        }
        if !grouping.steps_at(read_model, input, &metric)? {
            finer.push(metric);
        }
    }
    let finals = input.delivery == Delivery::FinalAggregates;
    if finals && finer.is_empty() && disjoint(read_model, input, branches)? {
        return Ok((Delivery::FinalAggregates, finer));
    }
    let again =
        finals && !finer.is_empty() && reads_again(read_model, input, (branches, pruned), &finer)?;
    if !again {
        finer.clear();
    }
    Ok((delivery(read_model, input)?, finer))
}

fn reads_again(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    (branches, pruned): (&[Operand], &[Operand]),
    finer: &[BoundName],
) -> Result<bool, PlanError> {
    for metric in finer {
        let definition = input.definition(read_model, metric)?;
        if definition.role_definition.aggregation().is_none()
            && part_key_of(read_model, input, metric)?.is_empty()
        {
            return Ok(false);
        }
    }
    let mut branches = branches.to_vec();
    if !pruned.is_empty() && step_grains(read_model, input)?.len() > 1 {
        branches.extend_from_slice(pruned);
    }
    let drills = |branch: &Operand| {
        is_grainset(read_model, &branch.source) && !one_child_serves(read_model, branch)
    };
    Ok(branches.iter().any(drills) && disjoint(read_model, input, &branches)?)
}

fn one_child_serves(read_model: &dyn ReadModel, branch: &Operand) -> bool {
    let names = (branch.names.iter()).filter_map(|name| read_model.bit_of(name.as_str()));
    let names = SemanticBitmask::from_positions(names);
    let children = read_model.operands(&branch.source);
    (children.iter()).any(|child| names.is_subset_of(&read_model.mask(child)))
}

fn unfinished(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    grouping: &Grouping,
) -> Result<Vec<BoundName>, PlanError> {
    let mut unfinished = Vec::new();
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        let linked = (input.linked(read_model, binding)?.into_iter()).map(BoundName::of);
        for metric in iter::once(binding.clone()).chain(linked) {
            if !grouping.finishes(read_model, input, &metric)?
                && !unfinished.contains(&metric)
                && !keyed(read_model, input, &metric)?
            {
                unfinished.push(metric);
            }
        }
    }
    Ok(unfinished)
}

fn keyed(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    metric: &BoundName,
) -> Result<bool, PlanError> {
    if input.part_key.is_empty() {
        return Ok(false);
    }
    let key = part_key_of(read_model, input, metric)?;
    let held = |tree: &PhysicalExpr| (input.part_key.iter()).any(|(_, held)| held == tree);
    Ok(!key.is_empty() && key.iter().all(|(_, tree)| held(tree)))
}

fn part_key_of(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    metric: &BoundName,
) -> Result<ComputedFields, PlanError> {
    let definition = input.definition(read_model, metric)?;
    let numeric = (definition.data_type.as_ref())
        .is_some_and(|data_type| data_type.family() == Some(TypeFamily::Number));
    if definition.role_definition.aggregation().is_some() || !numeric {
        return Ok(Vec::new());
    }
    let grouping = Grouping::of(read_model, input)?;
    let schema = Schema {
        columns: Vec::new(),
    };
    let naming = Naming::default();
    match finished(read_model, input, &[], &grouping, &schema, metric, naming)? {
        Finished::TwoSteps(two_steps) => Ok(two_steps.key),
        Finished::Step(_) | Finished::Joined(..) => Ok(Vec::new()),
    }
}

fn pushed_down(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
) -> Result<Vec<BoundName>, PlanError> {
    let grouping = Grouping::of(read_model, input)?;
    let mut pushed = Vec::new();
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        if pushes_down(read_model, input, binding)?
            && !grouping.finishes(read_model, input, binding)?
        {
            pushed.push(binding.clone());
        }
    }
    Ok(pushed)
}

fn step_dimensions(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
) -> Result<BTreeSet<SemanticsName>, PlanError> {
    let grains = step_grains(read_model, input)?.into_iter();
    Ok(grains.flat_map(|(dimensions, _)| dimensions).collect())
}

type StepGrain = (BTreeSet<SemanticsName>, Vec<BoundName>);

fn step_grains(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
) -> Result<Vec<StepGrain>, PlanError> {
    let grouping = Grouping::of(read_model, input)?;
    let mut grains: Vec<StepGrain> = Vec::new();
    for metric in unfinished(read_model, input, &grouping)? {
        if pushes_down(read_model, input, &metric)? {
            continue;
        }
        let linked = input.linked(read_model, &metric)?;
        let dimensions: BTreeSet<_> = (grouping.ungrouped(read_model, input, &metric, &linked)?)
            .into_iter()
            .map(|(dimension, _, _)| dimension.name.clone())
            .collect();
        if dimensions.is_empty() {
            continue;
        }
        match grains.iter_mut().find(|(held, _)| held == &dimensions) {
            Some((_, metrics)) => metrics.push(metric),
            None => grains.push((dimensions, vec![metric])),
        }
    }
    Ok(grains)
}

fn delivery(read_model: &dyn ReadModel, demand: &StrategyInput) -> Result<Delivery, PlanError> {
    if demand.delivery == Delivery::Rows {
        return Ok(Delivery::Rows);
    }
    let combines = |entry: &SemanticDefinition| {
        matches!(
            additivity(&entry.role_definition),
            Some(Additivity::Additive | Additivity::SemiAdditive)
        )
    };
    let leaves = demand.leaves(read_model)?;
    for binding in demand.names.iter().chain(&leaves) {
        let definitions = demand.aggregate_definitions(read_model, binding, &[])?;
        if !definitions.iter().all(combines) {
            return Ok(Delivery::Rows);
        }
    }
    Ok(Delivery::PartialAggregates)
}

fn disjoint(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    branches: &[Operand],
) -> Result<bool, PlanError> {
    if branches.is_empty() {
        return Ok(false);
    }
    if let [branch] = branches {
        return Ok(is_grainset(read_model, &branch.source) && !one_child_serves(read_model, branch));
    }
    let differ = |left: &Literal, right: &Literal| {
        let unequal = PhysicalExpr::Literal(left.clone()).neq(PhysicalExpr::Literal(right.clone()));
        matches!(unequal.fold(), Expr::Literal(Literal::Boolean(true)))
    };
    let separates = |&bit: &u32| {
        let literals = (branches.iter())
            .map(|branch| literal_on(read_model, &branch.source, bit))
            .collect::<Option<Vec<_>>>();
        literals.is_some_and(|literals| {
            (literals.iter().enumerate())
                .all(|(n, left)| (literals.iter().skip(n + 1)).all(|right| differ(left, right)))
        })
    };
    Ok((Grouping::of(read_model, input)?.grouped().iter()).any(separates))
}

type PreAggregated = BTreeMap<SourceAlias, BTreeMap<SemanticsName, PhysicalExpr>>;

fn pre_aggregated(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    relationship: &Relationship,
    ends: &[JoinEnd],
    pushed: &[BoundName],
) -> Result<PreAggregated, PlanError> {
    let mut formulas: BTreeMap<_, BTreeMap<_, _>> = BTreeMap::new();
    if input.delivery == Delivery::Rows {
        return Ok(formulas);
    }
    for binding in pushed {
        if (input.signature(read_model, binding)?.filter_expr_id).is_some()
            || input.conditions.contains_key(&binding.name)
        {
            continue;
        }
        let linked = input.linked(read_model, binding)?;
        let measures: Vec<_> = (linked.iter())
            .filter(|linked| {
                planned_as(read_model, &input.alias, linked) == Some(SemanticRole::Measure)
            })
            .collect();
        let tree = input.own_tree(read_model, binding, &[])?;
        let alone = |end: &&JoinEnd| {
            let mask = read_model.mask(end.operand);
            column_leaves(&tree).all(|leaf| qualifies(end.operand, leaf))
                && (measures.iter()).all(|measure| mask.contains(measure.bit_position))
        };
        let Some(end) = (ends.iter().find(alone))
            .filter(|end| end.preserved && !relationship.repeats(end.operand))
        else {
            continue;
        };
        let kept = (measures.iter())
            .map(|measure| {
                let root = (read_model.signature(&input.alias, measure.bit_position))
                    .ok_or_else(|| input.broken_tree(binding))?
                    .expression_id;
                let field = reference(measure.name.as_str()).map_err(|e| input.refused(e))?;
                Ok(KeptField {
                    leaf: KeptLeaf::Link(root),
                    field,
                })
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        let root = input.signature(read_model, binding)?.expression_id;
        let formula = input.tree(read_model, binding, &root, &kept)?;
        let into = formulas.entry(end.operand.clone()).or_default();
        into.insert(binding.name.clone(), formula);
    }
    Ok(formulas)
}

fn walked_parts(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operands: &[Frame],
    pushed: &[BoundName],
) -> Result<BTreeSet<SemanticsName>, PlanError> {
    let mut parts = BTreeSet::new();
    for binding in pushed {
        let tree = input.own_tree(read_model, binding, &[])?;
        if !reads_joined_rows(read_model, input, operands, &tree)? {
            parts.insert(binding.name.clone());
        }
    }
    Ok(parts)
}

fn end_deliveries(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    demands: &[StrategyInput],
    units: &[Option<StrategyInput>],
    pushed: &[BoundName],
    parts: &BTreeSet<SemanticsName>,
) -> Result<Vec<Delivery>, PlanError> {
    let units: Vec<_> = (demands.iter().zip(units))
        .map(|(demand, unit)| unit.as_ref().unwrap_or(demand))
        .collect();
    let mut deliveries = (units.iter())
        .map(|unit| delivery(read_model, unit))
        .collect::<Result<Vec<_>, _>>()?;
    let rows = |deliveries: &mut [Delivery], tree: &PhysicalExpr| {
        for (delivery, demand) in deliveries.iter_mut().zip(demands) {
            if column_leaves(tree).any(|leaf| qualifies(&demand.alias, leaf)) {
                *delivery = Delivery::Rows;
            }
        }
    };
    for formula in input.pre_aggregated.values() {
        rows(&mut deliveries, &input.lowered(read_model, formula)?);
    }
    for binding in input.names.iter().chain(&input.leaves(read_model)?) {
        let tree = input.own_tree(read_model, binding, &[])?;
        let at_an_end =
            (demands.iter()).any(|demand| demand.pre_aggregated.contains_key(&binding.name));
        if pushed.contains(binding) && !at_an_end && !parts.contains(&binding.name) {
            rows(&mut deliveries, &tree);
        }
        for call in aggregate_calls(&tree) {
            let leaves: Vec<_> = call.args().iter().flat_map(column_leaves).collect();
            let (mut joined, mut conditioned) = (leaves.is_empty(), false);
            let mut reads = vec![false; demands.len()];
            for ((read, demand), unit) in reads.iter_mut().zip(demands).zip(&units) {
                for leaf in (leaves.iter()).filter(|leaf| qualifies(&demand.alias, leaf)) {
                    let binding = unit.bound(read_model, leaf.name())?;
                    let definition = unit.definition(read_model, &binding)?;
                    let role = planned(read_model, unit, &binding, &definition)?;
                    joined |= role != SemanticRole::Measure;
                    conditioned |= (unit.signature(read_model, &binding)?.filter_expr_id).is_some();
                    *read = true;
                }
            }
            let rows = (deliveries.iter().zip(&reads))
                .any(|(delivery, &read)| read && *delivery == Delivery::Rows);
            let beside = reads.iter().filter(|&&read| read).count() > 1 && (conditioned || rows);
            for (delivery, read) in deliveries.iter_mut().zip(reads) {
                if joined || (beside && read) {
                    *delivery = Delivery::Rows;
                }
            }
        }
    }
    Ok(deliveries)
}
