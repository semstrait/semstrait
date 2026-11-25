mod computed;
mod drill_across;
mod stack;
mod volatile;

use std::collections::BTreeMap;
use std::{iter, mem};

use semstrait_common::{DataKindName, SemanticsName};
use semstrait_common::{Granularity, Tree, ValidateError};
use semstrait_ir::{
    field_name, Cardinality, Delivery, IrErrorKind, JoinType, JoinsetJoinType, Name, Parameter,
    PhysicalExpr, PlanNode, Schema, SemAnnotation, SemanticPlan, SemanticRole, SourceRole,
};
use semstrait_manifest::{join_filter_onto_aggregates, RoleDefinition, SourceAlias};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    map_nodes, AggregateAllowed, Call, CastFailure, Expr, ExprOperationExt, Literal,
    NamedReference, NullOrdering, SortDir, SortKey,
};
use semstrait_sql::functions::Additivity;

use crate::error::call_refused;
use crate::fan_out::{RowNumberMarker, RowNumbers};
use crate::formula::{
    below_root_cast, cast_where_differs, finished, gives_field, operand_conditions, planned,
    push_down, ComputedFields, Finished, Grouping, Linked, Naming, TwoSteps,
};
use crate::frame::{Decided, Frame, FrameKind, StepReadOf};
use crate::grain::{call_additivity, GranularityFit};
use crate::predicate::over_fields;
use crate::read_model::{is_unionset, ReadModel};
use crate::semi_additive::RankMarker;
use crate::steps::{
    aggregated_over, catalog_call, condition_field, field, formula_field, leaf_field, reserved,
    stemmed_entry, step_agg, step_cross, step_distinct, step_empty, step_fetch, step_filter,
    step_join, step_project, step_scan, step_sort, unbucketed, Aggregate,
};
use crate::strategy::{qualifies, FormulaNode, StrategyInput};
use crate::tree::{aggregate_calls, column_leaves, replace_columns};
use crate::{
    Advisory, BoundName, Combine, DecisionValue, Direction, OrderKey, Placement, PlanError,
    Predicate, Request, RequestContext, UnbuildableState,
};

pub(crate) fn plan_root(
    read_model: &dyn ReadModel,
    root: Frame,
    compositions: usize,
    granularity_fit: GranularityFit,
    request: &Request,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<SemanticPlan, PlanError> {
    let input = &root.input;
    let refused = |error| input.node_refused(error);
    let mut node = build(read_model, &root, None, context, advised)?;
    let output_names = (node.common().output_schema.columns.iter())
        .map(|field| Name::new(field.name.as_str()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| input.refused(error))?;
    let (order_by, offset, limit) = (&request.order_by, request.offset, request.limit);
    if !order_by.is_empty() {
        let keys = (order_by.iter().map(sort_key))
            .collect::<Result<_, _>>()
            .map_err(|error| input.refused(error))?;
        node = step_sort(node, keys).map_err(refused)?;
    }
    if offset.is_some() || limit.is_some() {
        let bound = |rows: u64| i64::try_from(rows).unwrap_or(i64::MAX);
        node = step_fetch(node, offset.map_or(0, bound), limit.map(bound)).map_err(refused)?;
    }
    let rank = [SemAnnotation::Rank {
        scope_root: input.semantic_source.clone(),
        compositions: u32::try_from(compositions).unwrap_or(u32::MAX),
        granularity_fit: granularity_fit.group,
        granularity: granularity_fit.granularity,
    }];
    Ok(SemanticPlan::new(traced(node, rank), output_names))
}

fn build(
    read_model: &dyn ReadModel,
    frame: &Frame,
    qualifier: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let input = &frame.input;
    let decided = match &frame.kind {
        FrameKind::Compute(compute) => {
            return computed::built(read_model, input, compute, qualifier, context, advised)
        }
        FrameKind::Decided(decided) => decided,
    };
    match decided.decision.combine {
        Combine::PassThrough => {
            let (unit, passed) = answering(read_model, frame)?;
            Ok(traced(
                build(read_model, unit, qualifier, context, advised)?,
                passed,
            ))
        }
        Combine::DrillAcross => {
            drill_across::joined(read_model, input, decided, qualifier, context, advised)
        }
        Combine::Leaf | Combine::Join | Combine::Cross | Combine::Stack => staged(
            read_model,
            (input, decided),
            qualifier,
            context,
            advised,
            |_| Ok(()),
        ),
    }
}

fn staged(
    read_model: &dyn ReadModel,
    (input, decided): (&StrategyInput, &Decided),
    qualifier: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
    fill: impl FnOnce(&mut FrameStages) -> Result<(), PlanError>,
) -> Result<PlanNode, PlanError> {
    let own = (decided.read_again.as_ref()).map_or(input, |again| &again.own);
    let node = combine(read_model, own, decided, qualifier, context, advised)?;
    let reread = || {
        combine(
            read_model,
            own,
            decided,
            qualifier,
            context,
            &mut Vec::new(),
        )
    };
    let combined = Combined::of(decided);
    let mut stages = FrameStages::of(
        read_model,
        input,
        qualifier,
        context,
        &node.common().output_schema,
        &combined,
        &decided.operands,
        &decided.row_numbers,
    )?;
    fill(&mut stages)?;
    if let Some(again) = &decided.read_again {
        let step1 = again.step1(decided).ok_or_else(|| input.broken())?;
        stages.read_again(read_model, input, &step1, qualifier, context, advised)?;
    }
    volatile::in_two_reads(read_model, input, decided, &stages, advised)?;
    volatile::copied(read_model, input, &stages, advised)?;
    stages.over(read_model, input, &combined, qualifier, node, &reread)
}

fn combined(decided: &Decided) -> Delivery {
    match decided.operands.iter().chain(&decided.excluded).next() {
        Some(operand) => operand.input.delivery,
        None => decided.empty_read.unwrap_or(Delivery::Rows),
    }
}

enum Combined {
    Own(Delivery),
    Operands(Vec<(SourceAlias, Delivery)>),
}

impl Combined {
    fn of(decided: &Decided) -> Self {
        match decided.decision.combine {
            Combine::Join | Combine::Cross => Self::Operands(
                (decided.operands.iter())
                    .map(|operand| (operand.input.alias.clone(), operand.input.delivery))
                    .collect(),
            ),
            Combine::Leaf | Combine::Stack | Combine::PassThrough | Combine::DrillAcross => {
                Self::Own(combined(decided))
            }
        }
    }

    fn partial(&self, call: &Aggregate) -> bool {
        let Self::Operands(operands) = self else {
            return false;
        };
        (call.args().iter().flat_map(column_leaves)).any(|leaf| {
            (operands.iter()).any(|(alias, delivery)| {
                *delivery == Delivery::PartialAggregates && qualifies(alias, leaf)
            })
        })
    }
}

fn answering<'f>(
    read_model: &dyn ReadModel,
    frame: &'f Frame,
) -> Result<(&'f Frame, Vec<SemAnnotation>), PlanError> {
    let Some(decided) = frame
        .decided()
        .filter(|d| d.decision.combine == Combine::PassThrough)
    else {
        return Ok((frame, Vec::new()));
    };
    let input = &frame.input;
    let [kept] = decided.operands.as_slice() else {
        return Err(input.broken());
    };
    let (answered_by, pruned) = match &decided.decision.stand_in {
        Some(stand_in) => (Some(stand_in.clone()), Vec::new()),
        None => {
            let pruned = (input.relationship(read_model)?.ends().into_iter())
                .filter(|(end, _)| **end != kept.input.alias)
                .map(|(end, _)| pruned(read_model, input, end))
                .collect::<Result<_, _>>()?;
            (None, pruned)
        }
    };
    let subject = trace(read_model, input, None, None, None, answered_by, pruned)?;
    let (unit, mut passed) = answering(read_model, kept)?;
    passed.push(subject);
    Ok((unit, passed))
}

fn combine(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    decided: &Decided,
    qualifier: Option<&str>,
    context: &RequestContext,
    advised: &mut Vec<Advisory>,
) -> Result<PlanNode, PlanError> {
    let decision = &decided.decision;
    let refused = |error| input.node_refused(error);
    match decision.combine {
        Combine::Leaf => {
            let read = (decision.physical_read.as_ref())
                .ok_or_else(|| input.not_allowed(DecisionValue::NoPhysicalRead))?;
            let label = Name::new(input.physical_source(read_model)?.name)
                .map_err(|error| input.refused(error))?;
            let trace = vec![trace(
                read_model,
                input,
                Some(label),
                None,
                None,
                None,
                Vec::new(),
            )?];
            let schema = read.schema.clone();
            match &read.reader {
                Some(reader) => step_scan(reader.clone(), schema, trace),
                None => step_empty(schema, trace),
            }
            .map_err(refused)
        }
        Combine::Join | Combine::Cross => {
            let relationship = input.relationship(read_model)?;
            let mut operand = |alias: &SourceAlias| {
                let operand = (decided.operands.iter())
                    .find(|operand| &operand.input.alias == alias)
                    .ok_or_else(|| input.broken())?;
                build(read_model, operand, Some(alias.as_str()), context, advised)
            };
            let (left, right) = (operand(&relationship.from)?, operand(&relationship.to)?);
            let cardinality = vec![trace(
                read_model,
                input,
                None,
                Some(relationship.cardinality),
                None,
                None,
                Vec::new(),
            )?];
            let combined = match (relationship.join_type, relationship.on) {
                (JoinsetJoinType::Join(join_type), Some(on)) => {
                    let join = step_join(left, right, join_type, on, relationship.filter);
                    join.map_err(refused)?.with_trace(cardinality)
                }
                (JoinsetJoinType::Cross, None) => {
                    let cross = step_cross(left, right).with_trace(cardinality);
                    match relationship.filter {
                        Some(filter) => step_filter(cross, filter).map_err(refused)?,
                        None => cross,
                    }
                }
                _ => return Err(input.broken()),
            };
            match &decided.above_join {
                Some(copies) => step_filter(combined, copies.clone()).map_err(refused),
                None => Ok(combined),
            }
        }
        Combine::Stack => stack::union(read_model, input, decided, qualifier, context, advised),
        Combine::PassThrough | Combine::DrillAcross => Err(input.broken()),
    }
}

type Marked = (SemanticsName, Name, Marker);

enum Marker {
    Rank(RankMarker),
    RowNumber(RowNumberMarker, Vec<PhysicalExpr>),
}

#[derive(Default)]
struct CallMarkers {
    rank: Option<RankMarker>,
    row_numbers: Vec<RowNumberMarker>,
}

#[derive(Clone)]
struct OutputField {
    semantics: SemanticsName,
    name: Name,
    tree: PhysicalExpr,
}

impl OutputField {
    fn new(semantics: SemanticsName, name: Name, tree: PhysicalExpr) -> Self {
        Self {
            semantics,
            name,
            tree,
        }
    }
}

impl From<OutputField> for (Name, PhysicalExpr) {
    fn from(field: OutputField) -> Self {
        (field.name, field.tree)
    }
}

#[derive(Default, Clone)]
struct Stages {
    row_level: Vec<OutputField>,
    before_aggregation: Option<PhysicalExpr>,
    bucketing: Vec<OutputField>,
    marking: Vec<OutputField>,
    groupings: Vec<OutputField>,
    aggregates: Vec<(Name, Aggregate)>,
    derived: Vec<OutputField>,
}

struct FrameStages {
    read: Stages,
    subtrees: Vec<Subtree>,
    joined: Vec<(Name, PhysicalExpr)>,
    after_aggregation: Option<PhysicalExpr>,
    output: Vec<OutputField>,
}

struct Subtree {
    step1: Stages,
    base: Stages,
    grain: ComputedFields,
    key: ComputedFields,
    filters: Vec<PhysicalExpr>,
    stem: Option<String>,
    marked: Vec<Marked>,
    values: Vec<Value>,
    after: Option<PhysicalExpr>,
    linked: Vec<Subtree>,
    over_finals: bool,
    read: Option<PlanNode>,
    join: JoinType,
}

type Finishing = (Finished, Vec<CallMarkers>);

type Finish<'a> =
    dyn FnMut(&StrategyInput, &BoundName, Option<&str>) -> Result<Finishing, PlanError> + 'a;

struct Placing<'a> {
    read_model: &'a dyn ReadModel,
    input: &'a StrategyInput,
    qualifier: Option<&'a str>,
    finish: &'a mut Finish<'a>,
    one_form: bool,
    in_branches: bool,
}

impl Placing<'_> {
    fn place(
        &mut self,
        subtrees: &mut Vec<Subtree>,
        read: &Stages,
        metric: &SemanticsName,
        name: Name,
        two_steps: TwoSteps,
        markers: Vec<CallMarkers>,
    ) -> Result<(), PlanError> {
        if let Some(subtree) = (subtrees.iter_mut()).find(|subtree| subtree.shares(&two_steps)) {
            return subtree.add(self, metric, name, two_steps, markers);
        }
        let opened = Subtree::open(read, metric, &two_steps);
        let mut opened = opened.map_err(|error| self.input.refused(error))?;
        opened.add(self, metric, name, two_steps, markers)?;
        subtrees.push(opened);
        Ok(())
    }

    fn place_link(
        &mut self,
        subtrees: &mut Vec<Subtree>,
        read: &Stages,
        link: Linked,
    ) -> Result<(), PlanError> {
        let input = self.input;
        let stem = if self.one_form {
            link.stem(self.read_model, input)?
        } else {
            None
        };
        let finishing = (self.finish)(input, &link.binding, stem.as_deref())?;
        let (Finished::TwoSteps(mut two_steps), markers) = finishing else {
            return Err(input.broken_tree(&link.binding));
        };
        if !self.in_branches {
            for filter in &link.filters {
                two_steps.formula = join_filter_onto_aggregates(&two_steps.formula, filter)
                    .map_err(|error| input.refused(call_refused(error)))?;
            }
        }
        for nested in &mut two_steps.linked {
            nested.through(&link);
        }
        two_steps.filters.extend(link.filters);
        two_steps.stem = stem;
        let (name, _) = leaf_field(link.field).map_err(|error| input.refused(error))?;
        self.place(subtrees, read, &link.binding.name, name, two_steps, markers)
    }
}

struct Value {
    metric: SemanticsName,
    name: Name,
    aliases: Vec<Name>,
    own: bool,
    call: Aggregate,
    value_type: DataType,
    on_failure: CastFailure,
}

impl Subtree {
    fn open(
        read: &Stages,
        metric: &SemanticsName,
        two_steps: &TwoSteps,
    ) -> Result<Self, ValidateError> {
        let mut step1 = read.clone();
        for (field, tree) in two_steps.grain.iter().chain(&two_steps.key) {
            let (name, leaf) = leaf_field(field.clone())?;
            if holds(&step1.groupings, &name) {
                continue;
            }
            let tree = replace_columns(tree.clone(), &two_steps.dimensions)?;
            (step1.row_level).push(OutputField::new(metric.clone(), name.clone(), tree));
            (step1.bucketing).push(OutputField::new(metric.clone(), name.clone(), leaf.clone()));
            (step1.groupings).push(OutputField::new(metric.clone(), name, leaf));
        }
        let base = Stages {
            derived: step1.groupings.clone(),
            ..step1.clone()
        };
        Ok(Self {
            step1,
            base,
            grain: two_steps.grain.clone(),
            key: two_steps.key.clone(),
            filters: two_steps.filters.clone(),
            stem: two_steps.stem.clone(),
            marked: Vec::new(),
            values: Vec::new(),
            after: None,
            linked: Vec::new(),
            over_finals: false,
            read: None,
            join: JoinType::Inner,
        })
    }

    fn shares(&self, two_steps: &TwoSteps) -> bool {
        let held: Vec<_> = (self.key.iter()).map(|(_, tree)| tree).collect();
        let key: Vec<_> = (two_steps.key.iter()).map(|(_, tree)| tree).collect();
        same_set(&self.grain, &two_steps.grain)
            && same_set(&held, &key)
            && same_set(&self.filters, &two_steps.filters)
            && self.stem == two_steps.stem
    }

    fn add(
        &mut self,
        placing: &mut Placing<'_>,
        metric: &SemanticsName,
        name: Name,
        two_steps: TwoSteps,
        markers: Vec<CallMarkers>,
    ) -> Result<(), PlanError> {
        let (input, qualifier) = (placing.input, placing.qualifier);
        let refused = |error| input.refused(error);
        let own = field(qualifier, metric.as_str()).map_err(refused)?.0 == name;
        if let Some(held) = (self.values.iter_mut()).find(|held| &held.metric == metric) {
            held.own |= own;
            held.aliases.push(name);
            return Ok(());
        }
        let held = (two_steps.key.iter()).filter_map(|(field, tree)| {
            let (held, _) = (self.key.iter()).find(|(_, held)| held == tree)?;
            Some((field.clone(), Expr::Column(held.clone())))
        });
        let tree = replace_columns(two_steps.formula, &held.collect::<Vec<_>>());
        let formula = formula_field(qualifier, metric).map_err(refused)?;
        let (entry, value) = formula.clone();
        let step1 = &mut self.step1;
        let (above, marked) = (step1)
            .aggregated(
                qualifier,
                metric,
                self.stem.as_deref(),
                formula,
                tree.map_err(refused)?,
                markers,
            )
            .map_err(refused)?;
        (step1.derived).push(OutputField::new(metric.clone(), entry, above));
        self.marked.extend(marked);
        let call = aggregated_over(two_steps.aggregation, value).map_err(refused)?;
        self.values.push(Value {
            metric: metric.clone(),
            name,
            aliases: Vec::new(),
            own,
            call,
            value_type: two_steps.value_type,
            on_failure: two_steps.on_failure,
        });
        for link in two_steps.linked {
            placing.place_link(&mut self.linked, &self.base, link)?;
        }
        Ok(())
    }

    fn mark(
        &mut self,
        input: &StrategyInput,
        qualifier: Option<&str>,
        over_finals: bool,
    ) -> Result<(), ValidateError> {
        self.over_finals = over_finals;
        let marked = mem::take(&mut self.marked);
        self.step1.mark(input, qualifier, marked)?;
        (self.linked.iter_mut()).try_for_each(|linked| linked.mark(input, qualifier, over_finals))
    }

    fn finishes(&self, predicate: &Predicate) -> bool {
        (predicate.leaves.iter())
            .all(|leaf| (self.values.iter()).any(|value| value.own && value.metric == leaf.name))
    }

    fn build(
        mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        qualifier: Option<&str>,
        node: PlanNode,
        reread: &dyn Fn() -> Result<PlanNode, PlanError>,
        grouping: Vec<(Name, PhysicalExpr)>,
    ) -> Result<PlanNode, PlanError> {
        let refused = |error| input.node_refused(error);
        let derived = mem::take(&mut self.step1.derived);
        let step1 = self.step1.groupings.clone();
        let node = if self.over_finals {
            node
        } else {
            (self.step1.aggregate(node, input.delivery)).map_err(refused)?
        };
        let linked = self.linked;
        let node = join_subtrees(read_model, input, qualifier, node, &step1, linked, reread)?;
        let node = derivation(node, derived).map_err(refused)?;
        let entries = (self.values.iter()).map(|value| (value.name.clone(), value.call.clone()));
        let node = step_agg(node, grouping, entries.collect()).map_err(refused)?;
        let node = typed_values(input, node, |name| {
            let value = (self.values.iter()).find(|value| &value.name == name)?;
            Some(Declared {
                value_type: value.value_type.clone(),
                on_failure: value.on_failure,
                aliases: value.aliases.clone(),
            })
        })?;
        match self.after {
            Some(after) => step_filter(node, after).map_err(refused),
            None => Ok(node),
        }
    }
}

impl FrameStages {
    pub(super) fn serving_every_read(
        self,
        qualifier: Option<&str>,
        delivery: Delivery,
    ) -> Result<Stages, ValidateError> {
        let mut read = self.read;
        let mut pending = self.subtrees;
        while let Some(subtree) = pending.pop() {
            pending.extend(subtree.linked);
            let stepped = (subtree.grain.iter().chain(&subtree.key))
                .map(|(field, _)| Ok(leaf_field(field.clone())?.0))
                .collect::<Result<Vec<_>, ValidateError>>()?;
            let step1 = subtree.step1;
            for row in &step1.row_level {
                if !stepped.contains(&row.name) || holds(&read.groupings, &row.name) {
                    held_or_pushed(&mut read.row_level, row);
                    continue;
                }
                for column in column_leaves(&row.tree) {
                    let (name, leaf) = leaf_field(column.clone())?;
                    let passed = OutputField::new(row.semantics.clone(), name, leaf);
                    for stage in [
                        &mut read.row_level,
                        &mut read.bucketing,
                        &mut read.groupings,
                    ] {
                        held_or_pushed(stage, &passed);
                    }
                }
            }
            for bucket in &step1.bucketing {
                if !stepped.contains(&bucket.name) {
                    held_or_pushed(&mut read.bucketing, bucket);
                }
            }
            for marker in &step1.marking {
                held_or_pushed(&mut read.marking, marker);
            }
            let entries = step1.entries(qualifier, subtree.stem.as_deref(), delivery)?;
            for (metric, written, call) in entries {
                if (read.aggregates.iter()).any(|(held, _)| held == &written) {
                    continue;
                }
                if delivery == Delivery::FinalAggregates {
                    let (_, leaf) = field(None, written.as_str())?;
                    (read.derived).push(OutputField::new(metric, written.clone(), leaf));
                }
                read.aggregates.push((written, call));
            }
        }
        Ok(read)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the frame's parts that the decide pass gives, each read apart"
    )]
    fn of(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        qualifier: Option<&str>,
        context: &RequestContext,
        below: &Schema,
        combined: &Combined,
        operands: &[Frame],
        row_numbers: &[RowNumbers],
    ) -> Result<Self, PlanError> {
        let refused = |error| input.refused(error);
        let (mut stages, mut output) = (Stages::default(), Vec::new());
        let mut subtrees: Vec<Subtree> = Vec::new();
        let grouping = Grouping::of(read_model, input)?;
        let (mut numbered, mut held) = (0, Vec::new());
        let mut finish = |input: &StrategyInput, binding: &BoundName, stem: Option<&str>| {
            let naming = Naming {
                qualifier,
                numbered,
            };
            let mut finished = finished(
                read_model, input, operands, &grouping, below, binding, naming,
            )?;
            numbered += finished.linked().len();
            let kept = (finished.linked().iter())
                .map(|link| Ok(input.signature(read_model, &link.binding)?.expression_id))
                .collect::<Result<Vec<_>, PlanError>>()?;
            let ranks = RankMarker::of_binding(read_model, input, operands, binding, &kept)?;
            let mut row_numbers = (row_numbers.iter())
                .find(|(bound, _)| bound == binding)
                .map_or_else(Vec::new, |(_, calls)| calls.clone())
                .into_iter();
            let mut markers: Vec<_> = (ranks.into_iter())
                .map(|rank| CallMarkers {
                    rank,
                    row_numbers: row_numbers.next().unwrap_or_default(),
                })
                .collect();
            let over_finals = matches!(combined, Combined::Own(Delivery::FinalAggregates))
                && matches!(finished, Finished::TwoSteps(_));
            let tree = finished.computed();
            *tree = if over_finals {
                let mut calls = 0;
                let values = tree.clone().transform(&mut |node| {
                    let Expr::Function(Call::Aggregate(..)) = &node else {
                        return Ok(node);
                    };
                    calls += 1;
                    Ok(stemmed_entry(qualifier, stem, &binding.name, calls, false)?.1)
                });
                values.map_err(refused)?
            } else if let Combined::Own(Delivery::PartialAggregates) = combined {
                let plain = matches!(tree, Expr::Function(Call::Aggregate(..)));
                (aggregate_calls(tree).try_for_each(combined_again))
                    .map_err(|state| input.unbuildable(state))?;
                let mut calls = 0;
                let partials = tree.clone().transform(&mut |node| {
                    let Expr::Function(Call::Aggregate(call, _)) = &node else {
                        return Ok(node);
                    };
                    calls += 1;
                    let alias = Some(input.alias.as_str());
                    let (_, partial) = stemmed_entry(alias, stem, &binding.name, calls, plain)?;
                    reaggregated(call, partial, input.delivery)
                });
                partials.map_err(refused)?
            } else {
                let mut rejoined = |node: &PhysicalExpr| {
                    let Expr::Function(Call::Aggregate(call, _)) = node else {
                        return None;
                    };
                    if !combined.partial(call) {
                        return None;
                    }
                    let Some((argument, filter)) = row_parts(node) else {
                        return Some(Err(input.broken_tree(binding)));
                    };
                    if let Err(state) = combined_again(call) {
                        return Some(Err(input.unbuildable(state)));
                    }
                    let partial = reaggregated(call, argument.clone(), input.delivery);
                    let partial = partial.and_then(|partial| match filter {
                        Some(filter) => {
                            join_filter_onto_aggregates(&partial, filter).map_err(call_refused)
                        }
                        None => Ok(partial),
                    });
                    Some(partial.map_err(refused))
                };
                let rebuilt =
                    map_nodes(tree, &mut rejoined, &|error| refused(call_refused(error)))?;
                let own_filter = (input.signature(read_model, binding)?.filter_expr_id).is_some();
                conditioned(input, below, &binding.name, &rebuilt, own_filter)?
            };
            if over_finals {
                markers.clear();
            }
            let calls = aggregate_calls(tree).count();
            let marked =
                (markers.iter()).any(|call| call.rank.is_some() || !call.row_numbers.is_empty());
            if marked && calls != markers.len() {
                return Err(input.broken_tree(binding));
            }
            if input.delivery == Delivery::PartialAggregates {
                held.extend(markers.iter_mut().filter_map(|call| call.rank.take()));
            }
            Ok::<_, PlanError>((finished, markers))
        };
        let (mut sums, mut standing) = (Vec::new(), Vec::new());
        for (metric, formula) in &input.pre_aggregated {
            let lowered = input.lowered(read_model, formula)?;
            let sum = match input.delivery {
                Delivery::Rows => None,
                _ => push_down(read_model, input, operands, &lowered, below)?,
            };
            let Some(sum) = sum else {
                for leaf in column_leaves(formula) {
                    let binding = input.bound(read_model, leaf.name())?;
                    if !input.names.contains(&binding) && !standing.contains(&binding) {
                        standing.push(binding);
                    }
                }
                continue;
            };
            let own_filter = aggregate_calls(&sum).any(|call| call.filter().is_some());
            sums.push((
                metric.clone(),
                conditioned(input, below, metric, &sum, own_filter)?,
            ));
        }
        let leaves = match combined {
            Combined::Own(Delivery::PartialAggregates) => StrategyInput {
                conditions: BTreeMap::new(),
                formula_nodes: Vec::new(),
                ..input.clone()
            }
            .leaves(read_model)?,
            _ => input.leaves(read_model)?,
        };
        let selected = (input.names.iter().chain(&standing)).map(|binding| (binding, true));
        let demand = selected.chain(leaves.iter().map(|binding| (binding, false)));
        let mut finishes = Vec::new();
        for (binding, selected) in demand {
            let definition = input.definition(read_model, binding)?;
            let (name, leaf) = field(qualifier, binding.name.as_str()).map_err(refused)?;
            let semantics = &binding.name;
            let named = |tree| OutputField::new(semantics.clone(), name.clone(), tree);
            let role = planned(read_model, input, binding, &definition)?;
            match (role, input.delivery) {
                (SemanticRole::Dimension | SemanticRole::Filter, _) => {
                    let tree = input.own_tree(read_model, binding, &[])?;
                    stages.row_level.push(named(tree));
                    if selected {
                        let data_type = definition.data_type.as_ref();
                        let bucket = bucket(input, context, semantics, data_type, leaf.clone())
                            .map_err(refused)?;
                        stages.bucketing.push(named(bucket));
                        stages.groupings.push(named(leaf.clone()));
                        stages.derived.push(named(leaf.clone()));
                    }
                }
                (SemanticRole::Measure, Delivery::Rows) if selected => {
                    let (Finished::Step(tree), _) = finish(input, binding, None)? else {
                        return Err(input.broken_tree(binding));
                    };
                    let (argument, filter) =
                        row_parts(&tree).ok_or_else(|| input.broken_tree(binding))?;
                    stages.row_level.push(named(argument.clone()));
                    if let Some(filter) = filter {
                        let (reserved, read) =
                            condition_field(qualifier, semantics).map_err(refused)?;
                        let condition =
                            OutputField::new(semantics.clone(), reserved.clone(), filter.clone());
                        stages.row_level.push(condition);
                        output.push(OutputField::new(semantics.clone(), reserved, read));
                    }
                }
                (
                    SemanticRole::Measure | SemanticRole::Metric,
                    Delivery::PartialAggregates | Delivery::FinalAggregates,
                ) => {
                    let value = (name.clone(), leaf.clone());
                    let finishing = finish(input, binding, None)?;
                    finishes.push((semantics.clone(), value, finishing, None));
                    for node in (input.nodes_of(semantics)).filter(|n| n.finishes_apart(input)) {
                        let field = node.field(input, qualifier).map_err(refused)?;
                        let finishing = finish(&input.under(node), binding, Some(&node.stem()))?;
                        finishes.push((semantics.clone(), field, finishing, Some(node)));
                    }
                }
                _ => return Err(input.broken_tree(binding)),
            }
            if selected {
                output.push(OutputField::new(semantics.clone(), name, leaf));
            }
        }
        for (metric, sum) in sums {
            let (name, leaf) = field(qualifier, metric.as_str()).map_err(refused)?;
            output.push(OutputField::new(metric.clone(), name.clone(), leaf.clone()));
            finishes.push((
                metric,
                (name, leaf),
                (Finished::Step(sum), Vec::new()),
                None,
            ));
        }
        for passed_up in &input.passed_up {
            let semantics = SemanticsName::new(passed_up.name());
            let (name, leaf) = leaf_field(passed_up.clone()).map_err(refused)?;
            let tree = match input.passed_up_binding(read_model, passed_up)? {
                Some(binding) => input.own_tree(read_model, &binding, &[])?,
                None => leaf.clone(),
            };
            let named = |tree| OutputField::new(semantics.clone(), name.clone(), tree);
            stages.row_level.push(named(tree));
            if input.delivery != Delivery::Rows {
                stages.bucketing.push(named(leaf.clone()));
                stages.groupings.push(named(leaf.clone()));
            }
            output.push(OutputField::new(semantics, name, leaf));
        }
        for (key, tree) in &input.part_key {
            let semantics = SemanticsName::new(key.name());
            let (name, leaf) = field(qualifier, key.name()).map_err(refused)?;
            let named = |tree| OutputField::new(semantics.clone(), name.clone(), tree);
            stages
                .row_level
                .push(named(input.lowered(read_model, tree)?));
            if input.delivery != Delivery::Rows {
                stages.bucketing.push(named(leaf.clone()));
                stages.groupings.push(named(leaf.clone()));
            }
            stages.derived.push(named(leaf.clone()));
            output.push(named(leaf));
        }
        let placed = |placement| {
            (input.predicates.iter()).filter(move |predicate| predicate.placement == placement)
        };
        if !matches!(combined, Combined::Own(Delivery::PartialAggregates)) {
            stages.before_aggregation =
                conjunction(placed(Placement::BeforeAggregation), qualifier).map_err(refused)?;
        }
        let (second_read, mut marked, mut joined) = (stages.clone(), Vec::new(), Vec::new());
        let mut placing = Placing {
            read_model,
            input,
            qualifier,
            finish: &mut finish,
            one_form: matches!(
                combined,
                Combined::Own(Delivery::PartialAggregates | Delivery::FinalAggregates)
            ) || is_unionset(read_model, &input.alias),
            in_branches: matches!(combined, Combined::Own(Delivery::PartialAggregates)),
        };
        for (semantics, (name, leaf), (finished, markers), node) in finishes {
            let (tree, links) = match finished {
                Finished::Step(tree) => (tree, None),
                Finished::Joined(tree, links) => (tree, Some(links)),
                Finished::TwoSteps(two_steps) => {
                    let (subtrees, read) = (&mut subtrees, &second_read);
                    placing.place(subtrees, read, &semantics, name, two_steps, markers)?;
                    continue;
                }
            };
            let entries = stages.aggregates.len();
            let stem = node.map(FormulaNode::stem);
            let value = (name.clone(), leaf.clone());
            let aggregated =
                stages.aggregated(qualifier, &semantics, stem.as_deref(), value, tree, markers);
            let (above, aggregated) = aggregated.map_err(refused)?;
            marked.extend(aggregated);
            let Some(links) = links else {
                let below = below_root_cast(&above).0.clone();
                let fields = match node {
                    Some(_) => vec![(name, leaf)],
                    None => {
                        (stages.derived).push(OutputField::new(semantics.clone(), name, above));
                        (input.nodes_of(&semantics))
                            .filter(|node| !node.finishes_apart(input))
                            .filter(|node| !node.in_measure_field(input))
                            .map(|node| node.field(input, qualifier))
                            .collect::<Result<_, _>>()
                            .map_err(refused)?
                    }
                };
                for (own, read) in fields {
                    let derived = OutputField::new(semantics.clone(), own.clone(), below.clone());
                    stages.derived.push(derived);
                    output.push(OutputField::new(semantics.clone(), own, read));
                }
                continue;
            };
            for (entry, _) in stages.aggregates.iter().skip(entries) {
                let (entry, read) = field(None, entry.as_str()).map_err(refused)?;
                (stages.derived).push(OutputField::new(semantics.clone(), entry, read));
            }
            if node.is_some() {
                let given = OutputField::new(semantics.clone(), name.clone(), leaf.clone());
                output.push(given);
            }
            joined.push((name, above));
            for link in links {
                placing.place_link(&mut subtrees, &second_read, link)?;
            }
        }
        stages.mark(input, qualifier, marked).map_err(refused)?;
        stages.group(qualifier, &held).map_err(refused)?;
        let after = placed(Placement::AfterAggregation);
        let over_finals = matches!(combined, Combined::Own(Delivery::FinalAggregates));
        for subtree in &mut subtrees {
            subtree
                .mark(input, qualifier, over_finals)
                .map_err(refused)?;
            let alone = after
                .clone()
                .filter(|predicate| subtree.finishes(predicate));
            subtree.after = conjunction(alone, qualifier).map_err(refused)?;
        }
        let above = after.filter(|predicate| !subtrees.iter().any(|s| s.finishes(predicate)));
        let after_aggregation = conjunction(above, qualifier).map_err(refused)?;
        Ok(Self {
            read: stages,
            subtrees,
            joined,
            after_aggregation,
            output,
        })
    }

    fn read_again(
        &mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        reads: &[StepReadOf<'_>],
        qualifier: Option<&str>,
        context: &RequestContext,
        advised: &mut Vec<Advisory>,
    ) -> Result<(), PlanError> {
        let refused = |error| input.refused(error);
        let mut grains = Vec::new();
        for &(step_read, step1_input, step1, again) in reads {
            let (mut grain, mut bucketed) = (Vec::new(), Vec::new());
            for added in &step1_input.names {
                let per_day = input.requested_granularity.contains_key(&added.name)
                    && !step1_input.requested_granularity.contains_key(&added.name);
                if input.names.contains(added) && !per_day {
                    continue;
                }
                let unbucketed = unbucketed(qualifier, &added.name).map_err(refused)?;
                let (name, _) = leaf_field(unbucketed).map_err(refused)?;
                let (own, read) = field(qualifier, added.name.as_str()).map_err(refused)?;
                if per_day {
                    let data_type = (read_model.definition(added.bit)).and_then(|d| d.data_type);
                    let value = bucket(
                        input,
                        context,
                        &added.name,
                        data_type.as_ref(),
                        read.clone(),
                    );
                    bucketed.push((own, value.map_err(refused)?));
                }
                grain.push((name, read));
            }
            grains.push((step_read, step1_input, step1, again, grain, bucketed));
        }
        let projected = |at: usize,
                         stepped: &ComputedFields,
                         advised: &mut Vec<Advisory>|
         -> Result<PlanNode, PlanError> {
            let (step_read, step1_input, step1, _, grain, bucketed) = &grains[at];
            let node = combine(read_model, step1_input, step1, qualifier, context, advised)?;
            let readers = (&step_read.readers[..], &step_read.served[..]);
            let node = stack::read_by_readers(read_model, step1_input, qualifier, node, readers)?;
            let mut fields = every_field(&node).map_err(refused)?;
            for (own, tree) in &mut fields {
                if let Some((_, value)) = (bucketed.iter()).find(|(name, _)| *name == *own) {
                    tree.clone_from(value);
                }
            }
            for (held, _) in stepped {
                let (name, _) = leaf_field(held.clone()).map_err(refused)?;
                let read = (grain.iter()).find(|(added, _)| added == &name);
                fields.push(read.cloned().ok_or_else(|| input.broken())?);
            }
            step_project(node, fields).map_err(|error| input.node_refused(error))
        };
        let mut read_by = Vec::new();
        for subtree in &mut self.subtrees {
            if subtree.grain.is_empty() && subtree.key.is_empty() {
                read_by.push(None);
                continue;
            }
            let serves = |metrics: &[SemanticsName]| {
                (subtree.values.iter()).any(|value| metrics.contains(&value.metric))
            };
            let at = (grains.iter()).position(|(step_read, ..)| serves(&step_read.metrics));
            let at = at.ok_or_else(|| input.broken())?;
            subtree.read = Some(projected(at, &subtree.grain, advised)?);
            let mut pending: Vec<&mut Subtree> = subtree.linked.iter_mut().collect();
            while let Some(Subtree {
                grain,
                linked,
                read,
                ..
            }) = pending.pop()
            {
                *read = Some(projected(at, grain, &mut Vec::new())?);
                pending.extend(linked.iter_mut());
            }
            read_by.push(grains[at].3.then_some(at));
        }
        let based = self.read.aggregates.is_empty();
        let left = read_by.first().copied().flatten().filter(|_| based);
        let mut widened = false;
        let joined = (self.subtrees.iter_mut()).zip(read_by);
        for (subtree, read) in joined.skip(usize::from(based)) {
            subtree.join = if read == left && !widened {
                JoinType::Inner
            } else if read == left || !based {
                JoinType::Left
            } else {
                widened = true;
                JoinType::Full
            };
        }
        Ok(())
    }

    fn build(
        mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        qualifier: Option<&str>,
        node: PlanNode,
        reread: &dyn Fn() -> Result<PlanNode, PlanError>,
    ) -> Result<PlanNode, PlanError> {
        let refused = |error| input.node_refused(error);
        if input.delivery == Delivery::PartialAggregates {
            return (self.read.partials().aggregate(node, input.delivery)).map_err(refused);
        }
        let own = |read: Stages, node| read.derive(node, input.delivery).map_err(refused);
        let node = self.with_subtrees(read_model, input, qualifier, node, reread, own)?;
        self.after(input, qualifier, node)
    }

    fn with_subtrees(
        &mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        qualifier: Option<&str>,
        node: PlanNode,
        reread: &dyn Fn() -> Result<PlanNode, PlanError>,
        own: impl FnOnce(Stages, PlanNode) -> Result<PlanNode, PlanError>,
    ) -> Result<PlanNode, PlanError> {
        let demand = self.read.groupings.clone();
        let mut subtrees = mem::take(&mut self.subtrees).into_iter().peekable();
        let node = match subtrees.next_if(|_| self.read.aggregates.is_empty()) {
            Some(mut subtree) => {
                let kept = (demand.iter())
                    .map(|field| (field.name.clone(), field.tree.clone()))
                    .collect();
                let node = subtree.read.take().unwrap_or(node);
                subtree.build(read_model, input, qualifier, node, reread, kept)?
            }
            None => own(mem::take(&mut self.read), node)?,
        };
        join_subtrees(
            read_model, input, qualifier, node, &demand, subtrees, reread,
        )
    }

    fn over(
        mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        combined: &Combined,
        qualifier: Option<&str>,
        node: PlanNode,
        reread: &dyn Fn() -> Result<PlanNode, PlanError>,
    ) -> Result<PlanNode, PlanError> {
        if !matches!(combined, Combined::Own(Delivery::FinalAggregates)) {
            return self.build(read_model, input, qualifier, node, reread);
        }
        self.after_aggregation = split_after_aggregation(read_model, input, qualifier)?.1;
        self.finish(read_model, input, qualifier, node, reread)
    }

    fn finish(
        mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        qualifier: Option<&str>,
        node: PlanNode,
        reread: &dyn Fn() -> Result<PlanNode, PlanError>,
    ) -> Result<PlanNode, PlanError> {
        let mut derived = mem::take(&mut self.read.derived);
        for written in &mut derived {
            if is_key(input, &written.semantics)
                || !is_metric(read_model, input, &written.semantics)?
            {
                let (_, leaf) =
                    field(None, written.name.as_str()).map_err(|error| input.refused(error))?;
                written.tree = leaf;
            }
        }
        let own = |_, node| derivation(node, derived).map_err(|error| input.node_refused(error));
        let node = self.with_subtrees(read_model, input, qualifier, node, reread, own)?;
        self.after(input, qualifier, node)
    }

    fn after(
        self,
        input: &StrategyInput,
        qualifier: Option<&str>,
        mut node: PlanNode,
    ) -> Result<PlanNode, PlanError> {
        let refused = |error| input.node_refused(error);
        if !self.joined.is_empty() {
            let passed = every_field(&node).map_err(|error| input.refused(error))?;
            let fields = passed.into_iter().chain(self.joined).collect();
            node = step_project(node, fields).map_err(refused)?;
        }
        let after_aggregation = self.after_aggregation;
        if let Some(filter) = after_aggregation.filter(|_| input.delivery != Delivery::Rows) {
            node = step_filter(node, filter).map_err(refused)?;
        }
        in_order(node, ordered(qualifier, self.output)).map_err(refused)
    }
}

impl Stages {
    fn derive(mut self, node: PlanNode, delivery: Delivery) -> Result<PlanNode, IrErrorKind> {
        let derived = mem::take(&mut self.derived);
        derivation(self.aggregate(node, delivery)?, derived)
    }

    fn partials(mut self) -> Self {
        for derived in &self.derived {
            let mut read = column_leaves(&derived.tree);
            let (Some(entry), None) = (read.next(), read.next()) else {
                continue;
            };
            let entry = field_name(entry);
            let partial =
                (self.aggregates.iter_mut()).find(|(written, _)| written.as_str() == entry);
            if let Some((written, _)) = partial {
                *written = derived.name.clone();
            }
        }
        self
    }

    fn entries(
        self,
        qualifier: Option<&str>,
        stem: Option<&str>,
        delivery: Delivery,
    ) -> Result<Vec<(SemanticsName, Name, Aggregate)>, ValidateError> {
        let mut aggregates = self.aggregates;
        let mut entries = Vec::new();
        for derived in &self.derived {
            for read in column_leaves(&derived.tree) {
                let read = field_name(read);
                let Some(at) =
                    (aggregates.iter()).position(|(written, _)| written.as_str() == read)
                else {
                    continue;
                };
                let (mut written, call) = aggregates.remove(at);
                let metric = &derived.semantics;
                if written == derived.name {
                    let plain = delivery != Delivery::FinalAggregates;
                    written = stemmed_entry(qualifier, stem, metric, 1, plain)?.0;
                }
                entries.push((metric.clone(), written, call));
            }
        }
        Ok(entries)
    }

    fn aggregate(self, node: PlanNode, delivery: Delivery) -> Result<PlanNode, IrErrorKind> {
        let mut node = step_project(node, by_semantics(self.row_level))?;
        if let Some(filter) = self.before_aggregation {
            node = step_filter(node, filter)?;
        }
        if delivery == Delivery::Rows {
            return Ok(node);
        }
        let bucketing = by_semantics(self.bucketing);
        if computes(&bucketing) {
            node = step_project(node, bucketing)?;
        }
        if !self.marking.is_empty() {
            node = step_project(node, by_semantics(self.marking))?;
        }
        let groupings = self.groupings.into_iter().map(Into::into);
        if self.aggregates.is_empty() {
            step_distinct(node, groupings.collect())
        } else {
            step_agg(node, groupings.collect(), self.aggregates)
        }
    }

    fn finish_branch(
        mut self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        node: PlanNode,
        on_measures: Option<PhysicalExpr>,
    ) -> Result<PlanNode, PlanError> {
        let refused = |error| input.node_refused(error);
        let (mut finished, mut formulas) = (Vec::new(), Vec::new());
        for derived in mem::take(&mut self.derived) {
            if !is_key(input, &derived.semantics)
                && is_metric(read_model, input, &derived.semantics)?
            {
                formulas.push(derived);
            } else {
                finished.push(derived);
            }
        }
        for formula in formulas {
            for read in column_leaves(&formula.tree) {
                let (name, leaf) =
                    leaf_field(read.clone()).map_err(|error| input.refused(error))?;
                if !holds(&finished, &name) {
                    finished.push(OutputField::new(formula.semantics.clone(), name, leaf));
                }
            }
        }
        let node = (self.aggregate(node, input.delivery)).map_err(refused)?;
        let node = derivation(node, finished).map_err(refused)?;
        match on_measures {
            Some(filter) => step_filter(node, filter).map_err(refused),
            None => Ok(node),
        }
    }

    fn aggregated(
        &mut self,
        qualifier: Option<&str>,
        semantics: &SemanticsName,
        stem: Option<&str>,
        (name, value): (Name, PhysicalExpr),
        tree: PhysicalExpr,
        markers: Vec<CallMarkers>,
    ) -> Result<(PhysicalExpr, Vec<Marked>), ValidateError> {
        let plain = matches!(tree, Expr::Function(Call::Aggregate(..)));
        let stem_or_empty = stem.unwrap_or_default();
        let (mut operands, mut entries, mut rows) = (0, 0, 0);
        let (mut markers, mut marked) = (markers.into_iter(), Vec::new());
        let above = tree.transform(&mut |node| {
            let Expr::Function(Call::Aggregate(call, _)) = &node else {
                return Ok(node);
            };
            let condition = call.filter().map(|_| call.args().len());
            let CallMarkers { rank, row_numbers } = markers.next().unwrap_or_default();
            let selected = (!row_numbers.is_empty())
                .then(|| RowNumberMarker::position(call.args()))
                .flatten();
            let mut fields = Vec::new();
            let read = (node.children().into_iter().cloned().enumerate())
                .map(|(position, operand)| match operand {
                    Expr::Literal(_) => Ok(operand),
                    operand => {
                        operands += 1;
                        let purpose = if condition == Some(position) {
                            "cond"
                        } else {
                            "arg"
                        };
                        let purpose = format_args!("{stem_or_empty}{purpose}{operands}");
                        let name = reserved(purpose, semantics);
                        let leaf = self.reserve(qualifier, semantics, name, operand)?;
                        fields.push(leaf.clone());
                        Ok(leaf)
                    }
                })
                .collect::<Result<_, ValidateError>>()?;
            let mut rebuilt = node.with_new_children(read)?;
            entries += 1;
            let position = match selected {
                Some(tree) => {
                    let purpose = format_args!("{stem_or_empty}pos{entries}");
                    let name = reserved(purpose, semantics);
                    let leaf = self.reserve(qualifier, semantics, name, tree)?;
                    fields.push(leaf.clone());
                    Some(leaf)
                }
                None => None,
            };
            let mut reads = |marker: Marker, purpose: &str, numbered: usize| {
                let purpose = format_args!("{stem_or_empty}{purpose}{numbered}");
                let (reserved, leaf) = field(qualifier, reserved(purpose, semantics))?;
                let once = leaf.eq(Expr::Literal(Literal::BigInt(1)));
                let read = match &marker {
                    Marker::Rank(_) => once,
                    Marker::RowNumber(row_number, _) => {
                        row_number.or_unmarked(once, position.as_ref())
                    }
                };
                rebuilt = join_filter_onto_aggregates(&rebuilt, &read).map_err(call_refused)?;
                marked.push((semantics.clone(), reserved, marker));
                Ok::<_, ValidateError>(())
            };
            if let Some(rank) = rank {
                reads(Marker::Rank(rank), "rank", entries)?;
            }
            for row_number in row_numbers {
                rows += 1;
                reads(Marker::RowNumber(row_number, fields.clone()), "row", rows)?;
            }
            let Expr::Function(Call::Aggregate(call, _)) = rebuilt else {
                return Err(ValidateError::CallRefused {
                    reason: format!("an aggregate call of `{semantics}` rebuilt as another node"),
                });
            };
            let (written, leaf) = if plain {
                (name.clone(), value.clone())
            } else {
                stemmed_entry(qualifier, stem, semantics, entries, false)?
            };
            self.aggregates.push((written, call));
            Ok(leaf)
        })?;
        Ok((above, marked))
    }

    fn mark(
        &mut self,
        input: &StrategyInput,
        qualifier: Option<&str>,
        marked: Vec<Marked>,
    ) -> Result<(), ValidateError> {
        let in_dataset = |stages: &mut Self, column: NamedReference| {
            let name = SemanticsName::new(column.parts().join("_"));
            let bucketed =
                (input.requested_granularity).contains_key(&SemanticsName::new(column.name()));
            let reserved = bucketed.then(|| unbucketed(qualifier, &name)).transpose()?;
            stages.read_through(column, reserved)
        };
        let mut windows = Vec::new();
        for (semantics, name, marker) in marked {
            let mut partition: Vec<_> = (self.groupings.iter())
                .map(|grouping| grouping.tree.clone())
                .collect();
            let window = match marker {
                Marker::Rank(marker) => {
                    let by = &marker.by.name;
                    let (by_field, by_leaf) = match marker.leaf(by) {
                        Some(column) => in_dataset(self, column)?,
                        None if input.requested_granularity.contains_key(by) => {
                            leaf_field(unbucketed(qualifier, by)?)?
                        }
                        None => field(qualifier, by.as_str())?,
                    };
                    let mut reads = vec![(by, by_field)];
                    for axis in &marker.axes {
                        let (axis_field, axis_leaf) = match marker.leaf(&axis.name) {
                            Some(column) => in_dataset(self, column)?,
                            None => field(qualifier, axis.name.as_str())?,
                        };
                        if !partition.contains(&axis_leaf) {
                            partition.push(axis_leaf);
                        }
                        reads.push((&axis.name, axis_field));
                    }
                    self.pass(qualifier, reads)?;
                    marker.window(partition, by_leaf)?
                }
                Marker::RowNumber(marker, operands) => {
                    for column in marker.identity {
                        partition.push(in_dataset(self, column)?.1);
                    }
                    partition.extend(operands);
                    RowNumberMarker::window(partition)?
                }
            };
            windows.push(OutputField::new(semantics, name, window));
        }
        if !windows.is_empty() {
            let passed = (self.bucketing.iter()).map(|bucket| {
                let (_, leaf) = field(None, bucket.name.as_str())?;
                let semantics = bucket.semantics.clone();
                Ok(OutputField::new(semantics, bucket.name.clone(), leaf))
            });
            self.marking = passed
                .chain(windows.into_iter().map(Ok))
                .collect::<Result<_, ValidateError>>()?;
        }
        Ok(())
    }

    fn group(&mut self, qualifier: Option<&str>, held: &[RankMarker]) -> Result<(), ValidateError> {
        for marker in held {
            for read in marker.reads() {
                let (name, leaf) = match marker.leaf(&read.name) {
                    Some(column) => self.read_through(column, None)?,
                    None => field(qualifier, read.name.as_str())?,
                };
                if !holds(&self.groupings, &name) {
                    self.pass(qualifier, [(&read.name, name.clone())])?;
                    (self.groupings).push(OutputField::new(read.name.clone(), name, leaf));
                }
            }
        }
        Ok(())
    }

    fn read_through(
        &mut self,
        column: NamedReference,
        reserved: Option<NamedReference>,
    ) -> Result<(Name, PhysicalExpr), ValidateError> {
        let semantics = SemanticsName::new(column.name());
        let (written, leaf) = leaf_field(reserved.unwrap_or_else(|| column.clone()))?;
        let read = [Expr::Column(column), leaf.clone()];
        for (stage, tree) in [&mut self.row_level, &mut self.bucketing]
            .into_iter()
            .zip(read)
        {
            if !holds(stage, &written) {
                stage.push(OutputField::new(semantics.clone(), written.clone(), tree));
            }
        }
        Ok((written, leaf))
    }

    fn reserve(
        &mut self,
        qualifier: Option<&str>,
        semantics: &SemanticsName,
        name: String,
        tree: PhysicalExpr,
    ) -> Result<PhysicalExpr, ValidateError> {
        let (reserved, leaf) = field(qualifier, name)?;
        (self.row_level).push(OutputField::new(semantics.clone(), reserved.clone(), tree));
        (self.bucketing).push(OutputField::new(semantics.clone(), reserved, leaf.clone()));
        Ok(leaf)
    }

    fn pass<'n>(
        &mut self,
        qualifier: Option<&str>,
        reads: impl IntoIterator<Item = (&'n SemanticsName, Name)>,
    ) -> Result<(), ValidateError> {
        for (read, reserved) in reads {
            if !holds(&self.bucketing, &reserved) {
                let (_, value) = field(qualifier, read.as_str())?;
                (self.bucketing).push(OutputField::new(read.clone(), reserved, value));
            }
        }
        Ok(())
    }
}

fn bucket(
    input: &StrategyInput,
    context: &RequestContext,
    name: &SemanticsName,
    data_type: Option<&DataType>,
    value: PhysicalExpr,
) -> Result<PhysicalExpr, ValidateError> {
    let (Delivery::FinalAggregates, Some(&granularity)) =
        (input.delivery, input.requested_granularity.get(name))
    else {
        return Ok(value);
    };
    let text = |text: &str| Expr::Literal(Literal::String(text.into()));
    let timezone = matches!(data_type, Some(DataType::Timestamp { .. }))
        .then(|| text(context.timezone.name()));
    let week_start = (granularity == Granularity::Week).then(|| text(context.week_start.name()));
    let args = [text(&granularity.to_string()), value].into_iter();
    catalog_call(
        "date_trunc",
        args.chain(timezone).chain(week_start).collect(),
    )
}

fn same_set<T: PartialEq>(left: &[T], right: &[T]) -> bool {
    left.iter().all(|item| right.contains(item)) && right.iter().all(|item| left.contains(item))
}

fn combined_again(call: &Aggregate) -> Result<(), UnbuildableState> {
    let function = call.function();
    match call_additivity(function, call.distinct()) {
        Additivity::Additive => Ok(()),
        _ => Err(UnbuildableState::PartialAggregateNotCombined {
            function: function.declarations.canonical.name,
        }),
    }
}

const COUNTS: [&str; 2] = ["count", "regr_count"];

fn reaggregated(
    call: &Aggregate,
    partial: PhysicalExpr,
    delivery: Delivery,
) -> Result<PhysicalExpr, ValidateError> {
    let name = call.function().declarations.canonical.name;
    if !COUNTS.contains(&name) {
        return catalog_call(name, vec![partial]);
    }
    let sum = catalog_call("sum", vec![partial])?;
    match delivery {
        Delivery::FinalAggregates => {
            catalog_call("coalesce", vec![sum, Expr::Literal(Literal::BigInt(0))])
        }
        _ => Ok(sum),
    }
}

fn held_or_pushed(stage: &mut Vec<OutputField>, field: &OutputField) {
    if !holds(stage, &field.name) {
        stage.push(field.clone());
    }
}

fn holds(stage: &[OutputField], name: &Name) -> bool {
    (stage.iter()).any(|field| &field.name == name)
}

struct Declared {
    value_type: DataType,
    on_failure: CastFailure,
    aliases: Vec<Name>,
}

fn typed_values(
    input: &StrategyInput,
    node: PlanNode,
    declared: impl Fn(&Name) -> Option<Declared>,
) -> Result<PlanNode, PlanError> {
    let mut casts = Vec::new();
    for column in &node.common().output_schema.columns {
        let (name, leaf) =
            field(None, column.name.as_str()).map_err(|error| input.refused(error))?;
        let Some(value) = declared(&name) else {
            casts.push((name, leaf));
            continue;
        };
        let (to, on_failure) = (&value.value_type, value.on_failure);
        let cast = cast_where_differs(leaf, &column.data_type, to, on_failure);
        let aliases = (value.aliases.iter()).map(|alias| (alias.clone(), cast.clone()));
        casts.extend(iter::once((name, cast.clone())).chain(aliases));
    }
    if computes(&casts) {
        return step_project(node, casts).map_err(|error| input.node_refused(error));
    }
    Ok(node)
}

fn computes(fields: &[(Name, PhysicalExpr)]) -> bool {
    (fields.iter()).any(
        |(name, tree)| !matches!(tree, Expr::Column(column) if field_name(column) == name.as_str()),
    )
}

fn derivation(node: PlanNode, derived: Vec<OutputField>) -> Result<PlanNode, IrErrorKind> {
    let derived = by_semantics(derived);
    if computes(&derived) {
        return step_project(node, derived);
    }
    Ok(node)
}

fn join_subtrees(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    qualifier: Option<&str>,
    node: PlanNode,
    grouping: &[OutputField],
    subtrees: impl IntoIterator<Item = Subtree>,
    reread: &dyn Fn() -> Result<PlanNode, PlanError>,
) -> Result<PlanNode, PlanError> {
    let mut numbered = (1_usize..).zip(subtrees);
    numbered.try_fold(node, |node, (k, mut subtree)| {
        let joined = (grouping.iter())
            .map(|grouping| {
                let (semantics, leaf) = (&grouping.semantics, &grouping.tree);
                let Expr::Column(read) = leaf else {
                    return Err(ValidateError::NodeRefused {
                        reason: format!(
                            "the grouping of `{semantics}` that a join reads is no field"
                        ),
                    });
                };
                let name = reserved(format_args!("join{k}"), read.name());
                let (name, reserved) = field(qualifier, name)?;
                let on = leaf.clone().is_not_distinct_from(reserved.clone());
                Ok(((name, leaf.clone()), (on, (read.clone(), reserved))))
            })
            .collect::<Result<Vec<_>, ValidateError>>()
            .map_err(|error| input.refused(error))?;
        let (kept, joins): (Vec<_>, Vec<_>) = joined.into_iter().unzip();
        let (on, keys): (Vec<_>, Vec<_>) = joins.into_iter().unzip();
        let read = match subtree.read.take() {
            Some(read) => read,
            None => reread()?,
        };
        let join = subtree.join;
        let joining = subtree.build(read_model, input, qualifier, read, reread, kept)?;
        if join == JoinType::Full && !on.is_empty() {
            let node = drill_across::joined_full(read_model, input, on, node, joining)?;
            return coalesced(input, node, keys);
        }
        match on.into_iter().reduce(|left, right| left & right) {
            Some(on) => {
                step_join(node, joining, join, on, None).map_err(|error| input.node_refused(error))
            }
            None => Ok(step_cross(node, joining)),
        }
    })
}

fn coalesced(
    input: &StrategyInput,
    node: PlanNode,
    keys: Vec<(NamedReference, PhysicalExpr)>,
) -> Result<PlanNode, PlanError> {
    let refused = |error| input.refused(error);
    let mut fields = every_field(&node).map_err(refused)?;
    for (read, reserved) in keys {
        let (own, leaf) = leaf_field(read).map_err(refused)?;
        let either = catalog_call("coalesce", vec![leaf, reserved]).map_err(refused)?;
        for (_, tree) in (fields.iter_mut()).filter(|(name, _)| *name == own) {
            tree.clone_from(&either);
        }
    }
    step_project(node, fields).map_err(|error| input.node_refused(error))
}

fn row_parts(tree: &PhysicalExpr) -> Option<(&PhysicalExpr, Option<&PhysicalExpr>)> {
    let mut calls = aggregate_calls(tree);
    let (Some(call), None) = (calls.next(), calls.next()) else {
        return None;
    };
    let args = call.args();
    let mut arguments = (args.iter()).filter(|arg| !matches!(arg, Expr::Literal(_)));
    match (arguments.next(), arguments.next(), args) {
        (Some(argument), None, _) | (None, None, [argument]) => Some((argument, call.filter())),
        _ => None,
    }
}

fn conditioned(
    input: &StrategyInput,
    below: &Schema,
    measure: &SemanticsName,
    tree: &PhysicalExpr,
    own_filter: bool,
) -> Result<PhysicalExpr, PlanError> {
    let refused = |error| input.refused(error);
    let mut read = |node: &PhysicalExpr| {
        let Expr::Function(Call::Aggregate(call, _)) = node else {
            return None;
        };
        Some(match operand_conditions(call, &gives_field(below)) {
            Ok(Some(conditions)) => (conditions.iter())
                .try_fold(node.clone(), |node, condition| {
                    join_filter_onto_aggregates(&node, condition)
                })
                .map_err(|error| refused(call_refused(error))),
            Ok(None) if !own_filter => Err(PlanError::AmbiguousOperandCondition {
                path: input.path.clone(),
                semantics: measure.clone(),
                semantic_source: input.semantic_source.clone(),
            }),
            Ok(None) => Ok(node.clone()),
            Err(error) => Err(refused(error)),
        })
    };
    map_nodes(tree, &mut read, &|error| refused(call_refused(error)))
}

fn sort_key(key: &OrderKey) -> Result<SortKey<Parameter, AggregateAllowed>, ValidateError> {
    let (_, expr) = field(None, key.name.as_str())?;
    let dir = match key.direction {
        Direction::Asc => SortDir::Asc {
            nulls: NullOrdering::Last,
        },
        Direction::Desc => SortDir::Desc {
            nulls: NullOrdering::First,
        },
    };
    Ok(SortKey { expr, dir })
}

fn conjunction<'p>(
    predicates: impl IntoIterator<Item = &'p Predicate>,
    qualifier: Option<&str>,
) -> Result<Option<PhysicalExpr>, ValidateError> {
    let trees = (predicates.into_iter())
        .map(|predicate| over_fields(&predicate.tree, |_| qualifier))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(trees.into_iter().reduce(|left, right| left & right))
}

fn split_after_aggregation(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    qualifier: Option<&str>,
) -> Result<(Option<PhysicalExpr>, Option<PhysicalExpr>), PlanError> {
    let (mut on_measures, mut on_metrics) = (Vec::new(), Vec::new());
    let after = |predicate: &&Predicate| predicate.placement == Placement::AfterAggregation;
    for predicate in input.predicates.iter().filter(after) {
        let mut on_metric = false;
        for leaf in &predicate.leaves {
            on_metric |= is_metric(read_model, input, &leaf.name)?;
        }
        if on_metric {
            on_metrics.push(predicate);
        } else {
            on_measures.push(predicate);
        }
    }
    let refused = |error| input.refused(error);
    Ok((
        conjunction(on_measures, qualifier).map_err(refused)?,
        conjunction(on_metrics, qualifier).map_err(refused)?,
    ))
}

fn is_key(input: &StrategyInput, name: &SemanticsName) -> bool {
    (input.part_key.iter()).any(|(key, _)| key.name() == name.as_str())
}

fn is_metric(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    name: &SemanticsName,
) -> Result<bool, PlanError> {
    let binding = input.bound(read_model, name.as_str())?;
    let definition = input.definition(read_model, &binding)?;
    Ok(planned(read_model, input, &binding, &definition)? == SemanticRole::Metric)
}

fn ordered(qualifier: Option<&str>, fields: Vec<OutputField>) -> Vec<(Name, PhysicalExpr)> {
    match qualifier {
        Some(_) => by_semantics(fields),
        None => fields.into_iter().map(Into::into).collect(),
    }
}

fn in_order(node: PlanNode, fields: Vec<(Name, PhysicalExpr)>) -> Result<PlanNode, IrErrorKind> {
    let names = (node.common().output_schema.columns.iter()).map(|column| column.name.as_str());
    if names.eq(fields.iter().map(|(name, _)| name.as_str())) {
        return Ok(node);
    }
    step_project(node, fields)
}

fn every_field(node: &PlanNode) -> Result<Vec<(Name, PhysicalExpr)>, ValidateError> {
    (node.common().output_schema.columns.iter())
        .map(|column| field(None, column.name.as_str()))
        .collect()
}

fn by_semantics(mut fields: Vec<OutputField>) -> Vec<(Name, PhysicalExpr)> {
    fields.sort_by(|left, right| {
        (left.semantics.cmp(&right.semantics)).then_with(|| left.name.cmp(&right.name))
    });
    fields.into_iter().map(Into::into).collect()
}

#[allow(
    clippy::too_many_arguments,
    reason = "one semantic source's trace line carries every optional key a decision can set"
)]
fn trace(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    physical_source: Option<Name>,
    cardinality: Option<Cardinality>,
    delivery: Option<Delivery>,
    answered_by: Option<DataKindName>,
    pruned: Vec<DataKindName>,
) -> Result<SemAnnotation, PlanError> {
    let source_role = match read_model.source_kind(&input.alias) {
        Some(RoleDefinition::Dataset { .. }) => SourceRole::Dataset,
        Some(RoleDefinition::Joinset { .. }) => SourceRole::Joinset,
        Some(RoleDefinition::Unionset { .. }) => SourceRole::Unionset,
        Some(RoleDefinition::Grainset { .. }) => SourceRole::Grainset,
        _ => return Err(input.broken()),
    };
    Ok(SemAnnotation::SemanticSource {
        semantic_source: input.semantic_source.clone(),
        source_role,
        originators: read_model.originators(&input.alias),
        physical_source,
        cardinality,
        delivery,
        answered_by,
        pruned,
    })
}

fn pruned(
    read_model: &dyn ReadModel,
    input: &StrategyInput,
    operand: &SourceAlias,
) -> Result<DataKindName, PlanError> {
    input.name(read_model, operand)
}

fn traced(node: PlanNode, subjects: impl IntoIterator<Item = SemAnnotation>) -> PlanNode {
    let held = node.common().semantic_meta.clone();
    node.with_trace(held.into_iter().chain(subjects))
}

#[cfg(test)]
mod tests {
    use semstrait_sql::expr::{AggregateCall, FunctionRef};
    use semstrait_sql::functions::registry;

    use super::*;

    fn declared(name: &str, args: Vec<PhysicalExpr>) -> Aggregate {
        let Some(FunctionRef::Aggregate(function, AggregateAllowed)) = registry().get(name) else {
            panic!("the catalog declares the aggregate `{name}`");
        };
        AggregateCall::builder(function, args)
            .build()
            .expect("an aggregate call")
    }

    #[test]
    fn a_distinct_count_does_not_combine_again() {
        let x = || Expr::Column(NamedReference::unqualified("x").expect("a name"));
        let Some(FunctionRef::Aggregate(count, AggregateAllowed)) = registry().get("count") else {
            panic!("the catalog declares the aggregate `count`");
        };
        let distinct = (AggregateCall::builder(count, vec![x()])
            .distinct(true)
            .build())
        .expect("a DISTINCT count");

        assert_eq!(
            combined_again(&distinct),
            Err(UnbuildableState::PartialAggregateNotCombined { function: "count" })
        );
        assert_eq!(combined_again(&declared("sum", vec![x()])), Ok(()));
        assert_eq!(combined_again(&declared("count", vec![x()])), Ok(()));
    }

    #[test]
    fn a_regr_count_re_aggregates_as_a_count_does() -> Result<(), ValidateError> {
        let column = |name: &str| Expr::Column(NamedReference::unqualified(name).expect("a name"));
        let regr_count = declared("regr_count", vec![column("y"), column("x")]);
        let count = declared("count", vec![column("y")]);
        let partial = column("partial");

        for delivery in [Delivery::PartialAggregates, Delivery::FinalAggregates] {
            assert_eq!(
                reaggregated(&regr_count, partial.clone(), delivery),
                reaggregated(&count, partial.clone(), delivery),
                "{delivery:?}"
            );
        }
        let sum = catalog_call("sum", vec![partial.clone()])?;
        let zero = Expr::Literal(Literal::BigInt(0));
        assert_eq!(
            reaggregated(&regr_count, partial, Delivery::FinalAggregates),
            catalog_call("coalesce", vec![sum, zero])
        );
        Ok(())
    }
}
