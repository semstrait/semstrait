use std::collections::BTreeMap;

use semstrait_common::{DataKindName, Granularity, SemanticsName, ValidateError};
use semstrait_ir::{Delivery, PhysicalExpr, SemanticRole};
use semstrait_manifest::{KeptField, KeptLeaf, SemanticBitmask, SourceAlias};
use semstrait_sql::expr::{CastFailure, Expr, NamedReference};
use semstrait_sql::functions::AggregateFunction;

use crate::formula::{below_root_cast, planned_as};
use crate::frame::fix;
use crate::grain::{is_dimension, RequestGrain, SourceGrain};
use crate::parts::{part_key, summed, PartKey};
use crate::read_model::{is_grainset, ReadModel};
use crate::routing::{fixed_granularity, route, Route};
use crate::steps::{reference, reserved};
use crate::strategy::{FormulaNode, StrategyInput};
use crate::tree::{column_leaves, replace_columns};
use crate::{FramePath, PlanError};

pub(crate) fn owns_metric(read_model: &dyn ReadModel, alias: &SourceAlias, bit: u32) -> bool {
    is_grainset(read_model, alias)
        && (read_model.definition(bit)).is_some_and(|d| d.role() == SemanticRole::Metric)
        && read_model.signature(alias, bit).is_some()
}

pub(crate) fn groups_by(read_model: &dyn ReadModel, alias: &SourceAlias, bit: u32) -> bool {
    if is_dimension(read_model, bit) {
        return true;
    }
    let Some(definition) = read_model.definition(bit) else {
        return false;
    };
    if definition.role() != SemanticRole::Metric || owns_metric(read_model, alias, bit) {
        return false;
    }
    let children = read_model.operands(alias);
    let serving = (children.iter()).find(|child| read_model.mask(child).contains(bit));
    serving.and_then(|child| planned_as(read_model, child, &definition))
        == Some(SemanticRole::Dimension)
}

pub(crate) fn aggregation(
    read_model: &dyn ReadModel,
    bit: u32,
) -> Option<(&'static AggregateFunction, bool)> {
    read_model.definition(bit)?.role_definition.aggregation()
}

pub(crate) struct Signature {
    pub(crate) tree: PhysicalExpr,
    pub(crate) filter: Option<PhysicalExpr>,
    pub(crate) links: Vec<(NamedReference, u32)>,
}

impl Signature {
    pub(crate) fn link(&self, column: &NamedReference) -> Option<u32> {
        (self.links.iter())
            .find(|(marker, _)| marker == column)
            .map(|(_, bit)| *bit)
    }
}

pub(crate) fn signature(
    read_model: &dyn ReadModel,
    alias: &SourceAlias,
    bit: u32,
) -> Option<Signature> {
    let stored = read_model.signature(alias, bit)?;
    let (mut kept, mut links) = (Vec::new(), Vec::new());
    for linked in read_model.linked(alias, &stored.expression_id)? {
        let own = read_model.signature(alias, linked.bit_position)?;
        let marker = reference(&reserved("link", &linked.name)).ok()?;
        kept.push(KeptField {
            leaf: KeptLeaf::Link(own.expression_id),
            field: marker.clone(),
        });
        links.push((marker, linked.bit_position));
    }
    let tree = read_model.expr_tree(alias, &stored.expression_id, &kept)?;
    let filter = stored
        .filter_expr_id
        .and_then(|filter| read_model.expr_tree(alias, &filter, &[]));
    Some(Signature {
        tree,
        filter,
        links,
    })
}

fn scoped(outer: Option<&PhysicalExpr>, filter: Option<&PhysicalExpr>) -> Option<PhysicalExpr> {
    match (outer, filter) {
        (Some(outer), Some(filter)) => Some(outer.clone() & filter.clone()),
        (outer, filter) => outer.or(filter).cloned(),
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ReachKind {
    Final,
    Formula,
    Leaf,
}

#[derive(Clone, PartialEq)]
pub(crate) struct ConditionedRead {
    pub(crate) bit: u32,
    pub(crate) condition: Option<PhysicalExpr>,
}

pub(crate) struct Reach {
    pub(crate) column: NamedReference,
    pub(crate) kind: ReachKind,
    pub(crate) read: ConditionedRead,
}

impl Signature {
    pub(crate) fn reaches(
        &self,
        read_model: &dyn ReadModel,
        alias: &SourceAlias,
        outer: Option<&PhysicalExpr>,
    ) -> Option<Vec<Reach>> {
        let scope = scoped(outer, self.filter.as_ref());
        let mut reaches = Vec::new();
        for column in column_leaves(&self.tree) {
            let (kind, bit, condition) = match self.link(column) {
                Some(linked) if aggregation(read_model, linked).is_some() => {
                    (ReachKind::Final, linked, scope.clone())
                }
                Some(linked) => (ReachKind::Formula, linked, scope.clone()),
                None => {
                    let bit = read_model.bit_of(column.name())?;
                    let condition = scope.clone().filter(|_| !groups_by(read_model, alias, bit));
                    (ReachKind::Leaf, bit, condition)
                }
            };
            reaches.push(Reach {
                column: column.clone(),
                kind,
                read: ConditionedRead { bit, condition },
            });
        }
        Some(reaches)
    }
}

fn leaf_reads(
    read_model: &dyn ReadModel,
    alias: &SourceAlias,
    bit: u32,
    outer: Option<&PhysicalExpr>,
) -> Option<LeafReads> {
    let mut collector = Collector {
        read_model,
        alias,
        visited: Vec::new(),
        reads: LeafReads {
            values: Vec::new(),
            leaves: Vec::new(),
        },
    };
    collector.collect(ConditionedRead {
        bit,
        condition: outer.cloned(),
    })?;
    Some(collector.reads)
}

struct LeafReads {
    values: Vec<ConditionedRead>,
    leaves: Vec<ConditionedRead>,
}

struct Collector<'a> {
    read_model: &'a dyn ReadModel,
    alias: &'a SourceAlias,
    visited: Vec<ConditionedRead>,
    reads: LeafReads,
}

impl Collector<'_> {
    fn collect(&mut self, read: ConditionedRead) -> Option<()> {
        if !push_new(&mut self.visited, read.clone()) {
            return Some(());
        }
        let signature = signature(self.read_model, self.alias, read.bit)?;
        for reach in signature.reaches(self.read_model, self.alias, read.condition.as_ref())? {
            let kept = match reach.kind {
                ReachKind::Formula => {
                    self.collect(reach.read)?;
                    continue;
                }
                ReachKind::Final => &mut self.reads.values,
                ReachKind::Leaf => &mut self.reads.leaves,
            };
            push_new(kept, reach.read);
        }
        Some(())
    }
}

pub(crate) struct Expansion {
    reads: Vec<ConditionedRead>,
}

impl Expansion {
    fn over(keys: &[u32]) -> Self {
        let plain = (keys.iter()).map(|bit| ConditionedRead {
            bit: *bit,
            condition: None,
        });
        Self {
            reads: plain.collect(),
        }
    }

    fn take(&mut self, read: ConditionedRead) {
        push_new(&mut self.reads, read);
    }

    pub(crate) fn demanded(&self) -> Vec<(u32, Option<PhysicalExpr>)> {
        let (plain, conditioned): (Vec<_>, Vec<_>) =
            (self.reads.iter()).partition(|read| read.condition.is_none());
        (plain.into_iter().chain(conditioned))
            .map(|read| (read.bit, read.condition.clone()))
            .collect()
    }

    fn only_conditioned(&self, bit: u32) -> bool {
        let reads = (self.reads.iter()).filter(|read| read.bit == bit);
        let (plain, conditioned): (Vec<_>, Vec<_>) =
            reads.partition(|read| read.condition.is_none());
        conditioned.len() == 1 && plain.is_empty()
    }

    pub(crate) fn under(&self, read_model: &dyn ReadModel) -> Option<Conditioned> {
        let plain = (self.reads.iter()).filter(|read| read.condition.is_none());
        let mut conditioned = Conditioned {
            conditions: BTreeMap::new(),
            nodes: Vec::new(),
            names: plain.map(|read| read.bit).collect(),
        };
        for read in &self.reads {
            let Some(condition) = &read.condition else {
                continue;
            };
            let measure = read_model.name_of(read.bit)?;
            push_new(&mut conditioned.names, read.bit);
            if self.only_conditioned(read.bit) {
                conditioned.conditions.insert(measure, condition.clone());
                continue;
            }
            FormulaNode::place(&mut conditioned.nodes, measure, Some(condition.clone()));
        }
        Some(conditioned)
    }
}

pub(crate) struct Conditioned {
    pub(crate) conditions: BTreeMap<SemanticsName, PhysicalExpr>,
    pub(crate) nodes: Vec<FormulaNode>,
    pub(crate) names: Vec<u32>,
}

pub(crate) enum Unexpanded {
    Unreadable(u32),
    Refused(ValidateError),
}

impl Unexpanded {
    pub(crate) fn error(
        self,
        read_model: &dyn ReadModel,
        path: &FramePath,
        source: &DataKindName,
    ) -> PlanError {
        let (path, semantic_source) = (path.clone(), source.clone());
        match self {
            Self::Refused(error) => PlanError::ConstructionRefused { path, error },
            Self::Unreadable(bit) => match read_model.name_of(bit) {
                Some(semantics) => PlanError::BrokenTree {
                    path,
                    semantic_source,
                    semantics,
                },
                None => PlanError::BrokenSource {
                    path,
                    semantic_source,
                },
            },
        }
    }
}

pub(crate) struct GroupMetric {
    pub(crate) bit: u32,
    pub(crate) aggregation: (&'static AggregateFunction, bool),
}

pub(crate) struct ConditionedMetric {
    pub(crate) metric: GroupMetric,
    pub(crate) condition: Option<PhysicalExpr>,
}

pub(crate) struct Valued {
    pub(crate) marker: NamedReference,
    pub(crate) read: ConditionedRead,
}

pub(crate) struct Parts {
    pub(crate) metric: GroupMetric,
    pub(crate) condition: Option<PhysicalExpr>,
    pub(crate) key: PartKey,
    pub(crate) values: Vec<Valued>,
    pub(crate) reads: Vec<u32>,
    pub(crate) ungrouped: u32,
    pub(crate) on_failure: CastFailure,
}

pub(crate) struct FormulaGrain {
    pub(crate) metrics: Vec<GroupMetric>,
    pub(crate) condition: Option<PhysicalExpr>,
    pub(crate) fine: Vec<u32>,
    extra: Vec<u32>,
}

pub(crate) enum GroupKind {
    Base,
    FormulaGrain(FormulaGrain),
    Parts(Parts),
}

pub(crate) struct FormulaGroup {
    pub(crate) expansion: Expansion,
    pub(crate) kind: GroupKind,
}

impl FormulaGroup {
    pub(crate) fn fine(&self) -> &[u32] {
        match &self.kind {
            GroupKind::FormulaGrain(grain) => &grain.fine,
            GroupKind::Base | GroupKind::Parts(_) => &[],
        }
    }

    pub(crate) fn extra(&self) -> &[u32] {
        match &self.kind {
            GroupKind::FormulaGrain(grain) => &grain.extra,
            GroupKind::Base | GroupKind::Parts(_) => &[],
        }
    }

    pub(crate) fn parts(&self) -> Option<&Parts> {
        match &self.kind {
            GroupKind::Parts(parts) => Some(parts),
            GroupKind::Base | GroupKind::FormulaGrain(_) => None,
        }
    }

    pub(crate) fn demanded(&self) -> Vec<(u32, Option<PhysicalExpr>)> {
        let mut demanded = self.expansion.demanded();
        let reads = (self.parts().into_iter()).flat_map(|parts| &parts.reads);
        for bit in reads {
            if !demanded.iter().any(|(held, _)| held == bit) {
                demanded.push((*bit, None));
            }
        }
        demanded
    }
}

pub(crate) struct Groups {
    pub(crate) groups: Vec<FormulaGroup>,
    pub(crate) after_join: Vec<ConditionedMetric>,
}

pub(crate) fn condition_reads(
    read_model: &dyn ReadModel,
    demanded: &[(u32, Option<PhysicalExpr>)],
) -> Vec<(u32, SemanticBitmask)> {
    (demanded.iter())
        .filter_map(|(bit, condition)| {
            let read = column_leaves(condition.as_ref()?)
                .filter_map(|leaf| read_model.bit_of(leaf.name()));
            Some((*bit, SemanticBitmask::from_positions(read)))
        })
        .collect()
}

fn demanded(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
) -> Result<Vec<(u32, Option<PhysicalExpr>)>, PlanError> {
    let names =
        (input.names.iter()).map(|name| (name.bit, input.conditions.get(&name.name).cloned()));
    let after = (input.predicates.iter()).filter(|predicate| !predicate.in_branches());
    let leaves = after.flat_map(|predicate| &predicate.leaves);
    let mut demanded: Vec<_> = names.chain(leaves.map(|leaf| (leaf.bit, None))).collect();
    for node in &input.formula_nodes {
        let bit = input.bound(read_model, node.measure.as_str())?.bit;
        demanded.push((bit, node.condition.clone()));
    }
    Ok(demanded)
}

pub(crate) fn formula_groups(
    read_model: &dyn ReadModel,
    input: &mut StrategyInput,
) -> Result<Option<Groups>, PlanError> {
    let demanded = demanded(read_model, input)?;
    if !(demanded.iter()).any(|(bit, _)| owns_metric(read_model, &input.alias, *bit)) {
        return Ok(None);
    }
    let grainset = SourceGrain::of_demand(read_model, input);
    let request = request_grain(read_model, input);
    let at = (&input.path, &input.semantic_source);
    let fixed = time_axis(read_model, &input.alias, &grainset, &request, &demanded, at)?;
    fix(input, &grainset, fixed);
    expand(read_model, &input.alias, &demanded, &granular(input))
        .map_err(|unexpanded| unexpanded.error(read_model, &input.path, &input.semantic_source))
}

fn request_grain(read_model: &dyn ReadModel, input: &StrategyInput) -> RequestGrain {
    let selected = (input.names.iter()).map(|name| {
        (
            name.bit,
            input.requested_granularity.get(&name.name).copied(),
        )
    });
    RequestGrain::new(read_model, selected)
}

fn granular(input: &StrategyInput) -> Vec<u32> {
    (input.names.iter())
        .filter(|name| input.requested_granularity.contains_key(&name.name))
        .map(|name| name.bit)
        .collect()
}

pub(crate) fn time_axis(
    read_model: &dyn ReadModel,
    alias: &SourceAlias,
    grainset: &SourceGrain,
    request: &RequestGrain,
    demanded: &[(u32, Option<PhysicalExpr>)],
    at: (&FramePath, &DataKindName),
) -> Result<Option<Granularity>, PlanError> {
    if !(demanded.iter()).any(|(bit, _)| owns_metric(read_model, alias, *bit)) {
        return Ok(None);
    }
    let Some(reads) = child_reads(read_model, alias, demanded) else {
        return Ok(None);
    };
    if let Some(route) = routed(read_model, alias, grainset, request, &reads, at)? {
        return Ok(routed_granularity(read_model, grainset, route));
    }
    if grainset.granularity().is_some() {
        return Ok(grainset.granularity());
    }
    let Ok(Some(groups)) = expand(read_model, alias, demanded, &request.granular()) else {
        return Ok(None);
    };
    let mut picked = Vec::new();
    for group in &groups.groups {
        let demanded = group.demanded();
        if let Some(route) = routed(read_model, alias, grainset, request, &demanded, at)? {
            picked.extend(routed_granularity(read_model, grainset, route));
        }
    }
    Ok((picked.into_iter()).max_by_key(|granularity| granularity.coarseness()))
}

fn routed(
    read_model: &dyn ReadModel,
    alias: &SourceAlias,
    grainset: &SourceGrain,
    request: &RequestGrain,
    reads: &[(u32, Option<PhysicalExpr>)],
    (path, source): (&FramePath, &DataKindName),
) -> Result<Option<Route>, PlanError> {
    let names = SemanticBitmask::from_positions(reads.iter().map(|(bit, _)| *bit));
    let children = read_model.operands(alias);
    let conditions = condition_reads(read_model, reads);
    let routed = route(
        read_model,
        grainset,
        &children,
        &names,
        &conditions,
        request,
        Delivery::FinalAggregates,
    )
    .map_err(|unrouted| unrouted.error(read_model, path, source))?;
    Ok(routed.map(|routed| routed.route))
}

fn routed_granularity(
    read_model: &dyn ReadModel,
    grainset: &SourceGrain,
    route: Route,
) -> Option<Granularity> {
    match route {
        Route::Child(child) => fixed_granularity(read_model, grainset, &child),
        Route::DrillAcross { granularity, .. } => granularity,
    }
}

fn child_reads(
    read_model: &dyn ReadModel,
    alias: &SourceAlias,
    demanded: &[(u32, Option<PhysicalExpr>)],
) -> Option<Vec<(u32, Option<PhysicalExpr>)>> {
    let mut pending: Vec<ConditionedRead> = (demanded.iter())
        .map(|(bit, condition)| ConditionedRead {
            bit: *bit,
            condition: condition.clone(),
        })
        .collect();
    let (mut seen, mut reads) = (Vec::new(), Vec::new());
    while let Some(read) = pending.pop() {
        if !push_new(&mut seen, read.clone()) {
            continue;
        }
        if !owns_metric(read_model, alias, read.bit) {
            reads.push((read.bit, read.condition));
            continue;
        }
        let condition = read.condition.as_ref();
        let LeafReads { values, leaves } = leaf_reads(read_model, alias, read.bit, condition)?;
        pending.extend(values.into_iter().chain(leaves));
    }
    Some(reads)
}

struct Expander<'a> {
    read_model: &'a dyn ReadModel,
    alias: &'a SourceAlias,
    keys: Vec<u32>,
    granular: &'a [u32],
    base: Expansion,
    formulas: Vec<FormulaGroup>,
    after_join: Vec<ConditionedMetric>,
    placed: Vec<ConditionedRead>,
}

impl Expander<'_> {
    fn place(&mut self, read: ConditionedRead) -> Result<(), Unexpanded> {
        if !push_new(&mut self.placed, read.clone()) {
            return Ok(());
        }
        let read_model = self.read_model;
        if !owns_metric(read_model, self.alias, read.bit) {
            self.base.take(read);
            return Ok(());
        }
        let ConditionedRead { bit, condition } = read;
        let LeafReads { values, leaves } =
            leaf_reads(read_model, self.alias, bit, condition.as_ref())
                .ok_or(Unexpanded::Unreadable(bit))?;
        let outside_keys = (leaves.iter())
            .any(|leaf| groups_by(read_model, self.alias, leaf.bit) && self.ungrouped(leaf.bit));
        let in_parts = (leaves.iter())
            .find(|leaf| groups_by(read_model, self.alias, leaf.bit) && self.ungrouped(leaf.bit))
            .map(|leaf| leaf.bit);
        match (aggregation(read_model, bit), in_parts) {
            (Some(own), _) if values.is_empty() || outside_keys => {
                self.at_formula_grain(bit, own, condition, leaves.into_iter().chain(values));
            }
            (None, Some(ungrouped)) => self.in_parts(bit, condition, ungrouped)?,
            (own, _) => {
                for value in values {
                    self.place(value)?;
                }
                for leaf in leaves {
                    self.base.take(leaf);
                }
                if let Some(aggregation) = own {
                    self.after_join.push(ConditionedMetric {
                        metric: GroupMetric { bit, aggregation },
                        condition,
                    });
                }
            }
        }
        Ok(())
    }

    fn ungrouped(&self, bit: u32) -> bool {
        !self.keys.contains(&bit) || self.granular.contains(&bit)
    }

    fn in_parts(
        &mut self,
        bit: u32,
        condition: Option<PhysicalExpr>,
        ungrouped: u32,
    ) -> Result<(), Unexpanded> {
        let mut parts = Inlining::default();
        let tree = self
            .inlined(bit, condition.as_ref(), &mut parts)
            .ok_or(Unexpanded::Unreadable(bit))?;
        let (body, on_failure) = below_root_cast(&tree);
        let metric = self
            .read_model
            .name_of(bit)
            .ok_or(Unexpanded::Unreadable(bit))?;
        let is_value =
            |column: &NamedReference| (parts.values.iter()).any(|value| &value.marker == column);
        let key = part_key(body, &is_value, &metric, None).map_err(Unexpanded::Refused)?;
        let mut expansion = Expansion::over(&self.keys);
        for value in &parts.values {
            expansion.take(value.read.clone());
        }
        let aggregation = summed().map_err(Unexpanded::Refused)?;
        self.formulas.push(FormulaGroup {
            expansion,
            kind: GroupKind::Parts(Parts {
                metric: GroupMetric { bit, aggregation },
                condition,
                key,
                values: parts.values,
                reads: parts.reads,
                ungrouped,
                on_failure,
            }),
        });
        Ok(())
    }

    fn inlined(
        &self,
        bit: u32,
        outer: Option<&PhysicalExpr>,
        parts: &mut Inlining,
    ) -> Option<PhysicalExpr> {
        let read_model = self.read_model;
        let signature = signature(read_model, self.alias, bit)?;
        let mut by: Vec<(NamedReference, PhysicalExpr)> = Vec::new();
        for reach in signature.reaches(read_model, self.alias, outer)? {
            if by.iter().any(|(held, _)| held == &reach.column) {
                continue;
            }
            let read = match reach.kind {
                ReachKind::Formula => {
                    self.inlined(reach.read.bit, reach.read.condition.as_ref(), parts)?
                }
                ReachKind::Leaf if groups_by(read_model, self.alias, reach.read.bit) => {
                    push_new(&mut parts.reads, reach.read.bit);
                    Expr::Column(reference(read_model.name_of(reach.read.bit)?.as_str()).ok()?)
                }
                ReachKind::Final | ReachKind::Leaf => {
                    Expr::Column(parts.value(read_model, reach.read.bit, reach.read.condition)?)
                }
            };
            by.push((reach.column, read));
        }
        replace_columns(signature.tree, &by).ok()
    }

    fn at_formula_grain(
        &mut self,
        bit: u32,
        aggregation: (&'static AggregateFunction, bool),
        condition: Option<PhysicalExpr>,
        reads: impl IntoIterator<Item = ConditionedRead>,
    ) {
        let reads: Vec<ConditionedRead> = reads.into_iter().collect();
        let dimensions: Vec<u32> = (reads.iter())
            .map(|read| read.bit)
            .filter(|bit| groups_by(self.read_model, self.alias, *bit))
            .collect();
        let mut extra: Vec<u32> = (dimensions.iter().copied())
            .filter(|bit| !self.keys.contains(bit))
            .collect();
        let mut fine: Vec<u32> = (dimensions.iter().copied())
            .filter(|bit| self.granular.contains(bit))
            .collect();
        for bits in [&mut extra, &mut fine] {
            bits.sort_unstable();
            bits.dedup();
        }
        let metric = GroupMetric { bit, aggregation };
        let held = (self.formulas.iter_mut()).find_map(|group| match &mut group.kind {
            GroupKind::FormulaGrain(grain)
                if grain.extra == extra && grain.fine == fine && grain.condition == condition =>
            {
                Some((&mut grain.metrics, &mut group.expansion))
            }
            GroupKind::Base | GroupKind::FormulaGrain(_) | GroupKind::Parts(_) => None,
        });
        match held {
            Some((metrics, expansion)) => {
                metrics.push(metric);
                for read in reads {
                    expansion.take(read);
                }
            }
            None => {
                let mut expansion = Expansion::over(&self.keys);
                for read in reads {
                    expansion.take(read);
                }
                self.formulas.push(FormulaGroup {
                    expansion,
                    kind: GroupKind::FormulaGrain(FormulaGrain {
                        metrics: vec![metric],
                        condition,
                        fine,
                        extra,
                    }),
                });
            }
        }
    }
}

pub(crate) fn expand(
    read_model: &dyn ReadModel,
    alias: &SourceAlias,
    demanded: &[(u32, Option<PhysicalExpr>)],
    granular: &[u32],
) -> Result<Option<Groups>, Unexpanded> {
    if !(demanded.iter()).any(|(bit, _)| owns_metric(read_model, alias, *bit)) {
        return Ok(None);
    }
    let mut keys: Vec<u32> = Vec::new();
    for (bit, _) in demanded {
        if groups_by(read_model, alias, *bit) {
            push_new(&mut keys, *bit);
        }
    }
    let mut expander = Expander {
        read_model,
        alias,
        keys,
        granular,
        base: Expansion::over(&[]),
        formulas: Vec::new(),
        after_join: Vec::new(),
        placed: Vec::new(),
    };
    for (bit, condition) in demanded {
        expander.place(ConditionedRead {
            bit: *bit,
            condition: condition.clone(),
        })?;
    }
    let base = expander.base;
    let reads_values = (base.reads.iter())
        .any(|read| read.condition.is_some() || !groups_by(read_model, alias, read.bit));
    let mut groups = Vec::new();
    if reads_values {
        groups.push(FormulaGroup {
            expansion: base,
            kind: GroupKind::Base,
        });
    }
    groups.extend(expander.formulas);
    Ok(Some(Groups {
        groups,
        after_join: expander.after_join,
    }))
}

#[derive(Default)]
struct Inlining {
    values: Vec<Valued>,
    reads: Vec<u32>,
}

impl Inlining {
    fn value(
        &mut self,
        read_model: &dyn ReadModel,
        bit: u32,
        condition: Option<PhysicalExpr>,
    ) -> Option<NamedReference> {
        let read = ConditionedRead { bit, condition };
        if let Some(held) = (self.values.iter()).find(|value| value.read == read) {
            return Some(held.marker.clone());
        }
        let name = read_model.name_of(bit)?;
        let numbered = format_args!("value{}", self.values.len() + 1);
        let marker = reference(&reserved(numbered, &name)).ok()?;
        self.values.push(Valued {
            marker: marker.clone(),
            read,
        });
        Some(marker)
    }
}

fn push_new<T: PartialEq>(held: &mut Vec<T>, item: T) -> bool {
    let new = !held.contains(&item);
    if new {
        held.push(item);
    }
    new
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use semstrait_common::io::InMemoryFileSystem;
    use semstrait_manifest::{compile, ManifestProvenance};
    use semstrait_model::parse;

    use super::*;
    use crate::read_model::ManifestReadModel;

    fn doubling_chain(depth: usize) -> String {
        let mut metrics = String::from(
            "    - {name: m0, data_type: double, expr: {divide: [{cast: {input: cost, target: double}}, clicks]}}\n",
        );
        for level in 1..=depth {
            let below = level - 1;
            metrics += &format!(
                "    - {{name: m{level}, data_type: double, expr: {{add: [m{below}, m{below}]}}}}\n"
            );
        }
        let refs: Vec<String> = (0..=depth)
            .map(|level| format!("{{ref: m{level}}}"))
            .collect();
        format!(
            r#"
semantic_model:
  name: chain
  dimensions:
    - {{name: campaign, data_type: string, type: categorical}}
  measures:
    - {{name: cost, data_type: bigint, agg: sum}}
    - {{name: clicks, data_type: bigint, agg: sum}}
  metrics:
{metrics}  grainsets:
    - name: g
      dimensions: [{{ref: campaign, is_key: true}}]
      measures: [{{ref: cost}}, {{ref: clicks}}]
      metrics: [{refs}]
      datasets:
        - name: g_cost
          extras:
            storage: {{format: parquet, paths: ['s3://lake/g_cost/']}}
            semantic_mapping: {{campaign: campaign, cost: cost}}
        - name: g_clicks
          extras:
            storage: {{format: parquet, paths: ['s3://lake/g_clicks/']}}
            semantic_mapping: {{campaign: campaign, clicks: clicks}}
"#,
            refs = refs.join(", ")
        )
    }

    #[tokio::test]
    async fn a_chain_that_reads_each_metric_twice_expands_each_metric_once() {
        let (model, _) = parse(&doubling_chain(20))
            .expect("the chain parses")
            .build()
            .expect("the chain builds");
        let provenance = ManifestProvenance::new(UNIX_EPOCH, UNIX_EPOCH + Duration::from_secs(1));
        let (manifest, _) = compile(model, None, &InMemoryFileSystem::new(), provenance)
            .await
            .expect("the chain compiles");
        let read_model = ManifestReadModel::new(&manifest);
        let alias = read_model.alias("g").expect("the Grainset's alias");
        let bit = read_model.bit_of("m20").expect("the chain's last Metric");

        let reads = leaf_reads(&read_model, &alias, bit, None).expect("the chain reads");

        let names: Vec<_> = (reads.values.iter().chain(&reads.leaves))
            .filter_map(|read| read_model.name_of(read.bit))
            .collect();
        assert_eq!(
            names,
            [SemanticsName::new("cost"), SemanticsName::new("clicks")]
        );
    }
}
