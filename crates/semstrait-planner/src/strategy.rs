use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::BitAnd;
use std::{fmt, iter};

use semstrait_common::{DataKindName, EntityId, SemanticsName};
use semstrait_common::{Granularity, Tree, ValidateError};
use semstrait_ir::{Delivery, IrErrorKind, MultisetOp, Name, PhysicalExpr, SemanticRole};
use semstrait_manifest::{
    join_filter_onto_aggregates, KeptField, KeptLeaf, PhysicalExprId, RoleDefinition, Selector,
    SemanticBitmask, SemanticDefinition, SemanticSignature, SourceAlias,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, Literal, NamedReference, SemanticLeaf};

use crate::binding::BoundRequest;
use crate::dataset_strategy::DatasetStrategy;
use crate::decision::Allowed;
use crate::error::{call_refused, Lacking};
use crate::fan_out::values;
use crate::formula::{planned, planned_as};
use crate::frame::{place, Aggregating};
use crate::grainset_strategy::GrainsetStrategy;
use crate::joinset_strategy::JoinsetStrategy;
use crate::predicate::{beyond, leaves_beyond, read_back, Read};
use crate::read_model::{is_grainset, literal_on, ReadModel, Relationship};
use crate::resolver::{nearest_children, Resolution, Structural};
use crate::routing::{maps_whole, with_conditions};
use crate::semi_additive::RankMarker;
use crate::steps::{reference, stemmed_entry};
use crate::tree::{aggregate_calls, column_leaves, replace_columns};
use crate::unionset_strategy::UnionsetStrategy;
use crate::{
    BoundName, Combine, Decision, DecisionValue, FramePath, Operand, PhysicalSource, PlanError,
    Predicate, UnbuildableState,
};

pub trait Strategy<Arm>: fmt::Debug + Send + Sync {
    fn decide(
        &self,
        read_model: &dyn ReadModel,
        arm: &Arm,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError>;
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyInput {
    pub path: FramePath,
    pub semantic_source: DataKindName,
    pub alias: SourceAlias,
    pub names: Vec<BoundName>,
    pub null_filled: Vec<SemanticsName>,
    pub demanded: SemanticBitmask,
    pub requested_granularity: BTreeMap<SemanticsName, Granularity>,
    pub stand_ins: Vec<DataKindName>,
    pub delivery: Delivery,
    pub predicates: Vec<Predicate>,
    pub conditions: BTreeMap<SemanticsName, PhysicalExpr>,
    pub pre_aggregated: BTreeMap<SemanticsName, PhysicalExpr>,
    pub formula_nodes: Vec<FormulaNode>,
    pub passed_up: Vec<NamedReference>,
    pub part_key: Vec<(NamedReference, PhysicalExpr)>,
    pub(crate) formula_grain: bool,
}

impl StrategyInput {
    pub(crate) fn root(resolution: Resolution, bound: BoundRequest) -> Self {
        Self {
            path: FramePath(vec![resolution.scope_root.clone()]),
            semantic_source: resolution.scope_root,
            alias: resolution.alias,
            names: bound.selected,
            null_filled: Vec::new(),
            demanded: resolution.demanded,
            requested_granularity: resolution.requested_granularity,
            stand_ins: (resolution.alternatives.into_iter())
                .map(|(_, unit)| unit)
                .collect(),
            delivery: Delivery::FinalAggregates,
            predicates: bound.predicates,
            conditions: BTreeMap::new(),
            pre_aggregated: BTreeMap::new(),
            formula_nodes: Vec::new(),
            passed_up: Vec::new(),
            part_key: Vec::new(),
            formula_grain: false,
        }
    }

    pub(crate) fn operand(
        &self,
        read_model: &dyn ReadModel,
        operand: &SourceAlias,
        names: Vec<BoundName>,
        delivery: Delivery,
        predicates: Vec<Predicate>,
        passed_up: Vec<NamedReference>,
    ) -> Result<Self, PlanError> {
        let source = self.name(read_model, operand)?;
        let mut path = self.path.clone();
        path.0.push(source.clone());
        let leaves = predicates.iter().flat_map(|predicate| &predicate.leaves);
        let demanded =
            SemanticBitmask::from_positions(names.iter().chain(leaves).map(|name| name.bit));
        Ok(Self {
            path,
            semantic_source: source,
            alias: operand.clone(),
            names,
            null_filled: Vec::new(),
            demanded,
            requested_granularity: self.requested_granularity.clone(),
            stand_ins: Vec::new(),
            delivery,
            predicates,
            conditions: BTreeMap::new(),
            pre_aggregated: BTreeMap::new(),
            formula_nodes: Vec::new(),
            passed_up,
            part_key: Vec::new(),
            formula_grain: false,
        })
    }

    pub(crate) fn null_filling(
        self,
        read_model: &dyn ReadModel,
        null_filled: Vec<SemanticsName>,
    ) -> Result<Self, PlanError> {
        let bits = (null_filled.iter())
            .map(|name| self.bound(read_model, name.as_str()).map(|bound| bound.bit))
            .collect::<Result<Vec<_>, _>>()?;
        let demanded = (self.demanded).difference(&SemanticBitmask::from_positions(bits));
        Ok(Self {
            null_filled,
            demanded,
            ..self
        })
    }

    pub(crate) fn reads_null_filled(&self, predicate: &Predicate) -> bool {
        (predicate.leaves.iter()).any(|leaf| self.null_filled.contains(&leaf.name))
    }

    pub(crate) fn with_typed_nulls(
        &self,
        read_model: &dyn ReadModel,
        mut predicate: Predicate,
    ) -> Result<Predicate, PlanError> {
        let mut nulls = BTreeMap::new();
        for leaf in (predicate.leaves.iter()).filter(|leaf| self.null_filled.contains(&leaf.name)) {
            nulls.insert(&leaf.name, self.null_fill_type(read_model, &leaf.name)?);
        }
        let tree = predicate.tree.transform(&mut |node| {
            let Expr::Leaf(leaf) = &node else {
                return Ok(node);
            };
            let Some(Read::Semantics { name, .. }) = Read::of(leaf) else {
                return Ok(node);
            };
            let null = nulls.get(&name).cloned().map(Literal::Null);
            Ok(null.map_or(node, Expr::Literal))
        });
        predicate.tree = tree.map_err(|error| self.refused(error))?;
        predicate
            .leaves
            .retain(|leaf| !self.null_filled.contains(&leaf.name));
        Ok(predicate)
    }

    pub(crate) fn leaves(&self, read_model: &dyn ReadModel) -> Result<Vec<BoundName>, PlanError> {
        self.leaves_of(read_model, &self.predicates)
    }

    fn leaves_of(
        &self,
        read_model: &dyn ReadModel,
        predicates: &[Predicate],
    ) -> Result<Vec<BoundName>, PlanError> {
        let mut predicate = leaves_beyond(&self.names, predicates);
        predicate.retain(|leaf| !self.null_filled.contains(&leaf.name));
        let mut conditioned = Vec::new();
        let nodes = (self.formula_nodes.iter()).filter_map(|node| node.condition.as_ref());
        let keys = self.part_key.iter().map(|(_, key)| key);
        for condition in self.conditions.values().chain(nodes).chain(keys) {
            conditioned.extend(self.condition_leaves(read_model, condition)?);
        }
        let joins = read_model.relationship(&self.alias).is_some();
        let mut markers = Vec::new();
        for binding in self.names.iter().chain(predicate.iter().copied()) {
            if joins && !values(&self.own_tree(read_model, binding, &[])?).is_empty() {
                continue;
            }
            markers.extend(RankMarker::of_binding(read_model, self, &[], binding, &[])?);
        }
        let marked = (markers.iter().flatten()).flat_map(RankMarker::reads);
        let read = (predicate.iter().copied())
            .chain(&conditioned)
            .chain(marked);
        Ok(beyond(&self.names, read).into_iter().cloned().collect())
    }

    pub(crate) fn condition_reads(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<Vec<(u32, SemanticBitmask)>, PlanError> {
        let nodes = (self.formula_nodes.iter())
            .filter_map(|node| Some((&node.measure, node.condition.as_ref()?)));
        let mut reads = (self.conditions.iter().chain(nodes))
            .map(|(measure, condition)| {
                let read = self.condition_leaves(read_model, condition)?;
                let read = SemanticBitmask::from_positions(read.iter().map(|name| name.bit));
                Ok((self.bound(read_model, measure.as_str())?.bit, read))
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        let mut keyed = Vec::new();
        for (_, key) in &self.part_key {
            keyed.extend(self.condition_leaves(read_model, key)?);
        }
        if !keyed.is_empty() {
            let keyed = SemanticBitmask::from_positions(keyed.iter().map(|name| name.bit));
            reads.extend((self.names.iter()).map(|name| (name.bit, keyed.clone())));
        }
        Ok(reads)
    }

    pub(crate) fn key_leaves(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<Vec<BoundName>, PlanError> {
        let mut read = Vec::new();
        for (_, key) in &self.part_key {
            read.extend(self.condition_leaves(read_model, key)?);
        }
        Ok(beyond(&self.names, &read).into_iter().cloned().collect())
    }

    pub(crate) fn condition_leaves(
        &self,
        read_model: &dyn ReadModel,
        condition: &PhysicalExpr,
    ) -> Result<Vec<BoundName>, PlanError> {
        (column_leaves(condition))
            .filter(|leaf| !(self.null_filled.iter()).any(|name| name.as_str() == leaf.name()))
            .map(|leaf| self.bound(read_model, leaf.name()))
            .collect()
    }

    fn summed_leaves(&self, read_model: &dyn ReadModel) -> Result<Vec<BoundName>, PlanError> {
        let mut leaves = Vec::new();
        for formula in self.pre_aggregated.values() {
            leaves.extend(self.condition_leaves(read_model, formula)?);
        }
        Ok(leaves)
    }

    pub(crate) fn lowered_onto(
        &self,
        read_model: &dyn ReadModel,
        names: &[SemanticsName],
    ) -> Result<BTreeMap<SemanticsName, PhysicalExpr>, PlanError> {
        let mut lowered = BTreeMap::new();
        for name in names {
            let own = self.own_filter(read_model, &self.bound(read_model, name.as_str())?)?;
            let held = self.conditions.get(name).cloned();
            if let Some(condition) = own.into_iter().chain(held).reduce(BitAnd::bitand) {
                lowered.insert(name.clone(), condition);
            }
        }
        Ok(lowered)
    }

    pub(crate) fn nodes_of<'n>(
        &'n self,
        measure: &'n SemanticsName,
    ) -> impl Iterator<Item = &'n FormulaNode> {
        (self.formula_nodes.iter()).filter(move |node| &node.measure == measure)
    }

    pub(crate) fn under(&self, node: &FormulaNode) -> Self {
        let mut input = self.clone();
        match &node.condition {
            Some(condition) => input
                .conditions
                .insert(node.measure.clone(), condition.clone()),
            None => input.conditions.remove(&node.measure),
        };
        input
    }

    pub(crate) fn with_bits(
        &self,
        read_model: &dyn ReadModel,
        mut names: Vec<BoundName>,
        bits: impl IntoIterator<Item = u32>,
    ) -> Result<Vec<BoundName>, PlanError> {
        for bit in bits {
            if !names.iter().any(|name| name.bit == bit) {
                let name = read_model.name_of(bit).ok_or_else(|| self.broken())?;
                names.push(BoundName { name, bit });
            }
        }
        Ok(names)
    }

    pub(crate) fn filling(&self, operand: &Operand) -> Operand {
        Operand {
            null_filled: [&operand.null_filled[..], &self.null_filled].concat(),
            ..operand.clone()
        }
    }

    pub(crate) fn branch_input(
        &self,
        read_model: &dyn ReadModel,
        operand: &Operand,
        predicates: &[Predicate],
        delivery: Delivery,
    ) -> Result<Self, PlanError> {
        let names = (operand.names.iter())
            .map(|name| self.bound(read_model, name.as_str()))
            .collect::<Result<_, _>>()?;
        self.operand(
            read_model,
            &operand.source,
            names,
            delivery,
            predicates.to_vec(),
            Vec::new(),
        )?
        .null_filling(read_model, operand.null_filled.clone())
    }

    pub(crate) fn outputs(&self, name: &SemanticsName) -> bool {
        (self.names.iter()).any(|binding| &binding.name == name)
            || (self.passed_up.iter()).any(|leaf| leaf.name() == name.as_str())
    }

    pub(crate) fn read_leaves(&self, read_model: &dyn ReadModel) -> Result<Vec<String>, PlanError> {
        let mut read: Vec<_> = (self.columns(read_model)?.into_iter())
            .flat_map(|(_, columns)| columns)
            .collect();
        for leaf in &self.passed_up {
            if let Some(binding) = self.passed_up_binding(read_model, leaf)? {
                read.extend(self.columns_of(read_model, &binding)?);
            }
        }
        let mut leaves: Vec<String> = Vec::new();
        for column in read {
            if !leaves.iter().any(|held| held == column.name()) {
                leaves.push(column.name().to_owned());
            }
        }
        Ok(leaves)
    }

    pub(crate) fn columns(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<Vec<(BoundName, Vec<NamedReference>)>, PlanError> {
        let mut read = [self.names.clone(), self.leaves(read_model)?].concat();
        let summed = self.summed_leaves(read_model)?;
        read.extend(beyond(&read, &summed).into_iter().cloned());
        (read.into_iter())
            .map(|binding| {
                let columns = self.columns_of(read_model, &binding)?;
                Ok((binding, columns))
            })
            .collect()
    }

    pub(crate) fn columns_of(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
    ) -> Result<Vec<NamedReference>, PlanError> {
        let signature = self.signature(read_model, binding)?;
        let mut columns = Vec::new();
        for root in iter::once(&signature.expression_id).chain(&signature.filter_expr_id) {
            let tree = self.tree(read_model, binding, root, &[])?;
            columns.extend(column_leaves(&tree).cloned());
        }
        Ok(columns)
    }

    pub(crate) fn passed_up_binding(
        &self,
        read_model: &dyn ReadModel,
        leaf: &NamedReference,
    ) -> Result<Option<BoundName>, PlanError> {
        if !qualifies(&self.alias, leaf) {
            return Ok(None);
        }
        self.bound(read_model, leaf.name()).map(Some)
    }

    pub(crate) fn reads_alone(
        &self,
        read_model: &dyn ReadModel,
        relationship: &Relationship,
        operand: &SourceAlias,
    ) -> Result<bool, PlanError> {
        if !relationship.may_read_alone(operand) {
            return Ok(false);
        }
        let leaves = self.leaves_of(read_model, &self.request_predicates(read_model)?)?;
        let summed = self.summed_leaves(read_model)?;
        let mask = read_model.mask(operand);
        let exposed = (self.names.iter().chain(&leaves).chain(&summed))
            .all(|binding| mask.contains(binding.bit));
        let closure = read_model.operand_closure(operand);
        let passed_up = (self.passed_up.iter())
            .all(|leaf| (iter::once(operand).chain(&closure)).any(|alias| qualifies(alias, leaf)));
        Ok(exposed && passed_up)
    }

    pub(crate) fn may_stand_in(&self, read_model: &dyn ReadModel, stand_in: &DataKindName) -> bool {
        let Some(alias) = read_model.alias(stand_in.as_str()) else {
            return false;
        };
        let mut kept = read_model.operand_closure(&alias);
        kept.insert(alias);
        self.stand_ins.contains(stand_in) && prunable(read_model, &self.alias, &kept)
    }

    pub(crate) fn request_predicates(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<Vec<Predicate>, PlanError> {
        if is_grainset(read_model, &self.alias) {
            return Ok(self.predicates.clone());
        }
        (self.predicates.iter())
            .map(|predicate| match predicate.leaves.as_slice() {
                [filter] if self.definition(read_model, filter)?.role() == SemanticRole::Filter => {
                    let root = self.signature(read_model, filter)?.expression_id;
                    let stored = self.tree(read_model, filter, &root, &[])?;
                    let bit_of = |name: &str| read_model.bit_of(name);
                    let broken = || self.broken_tree(filter);
                    read_back(&stored, predicate.placement, bit_of, broken)
                }
                _ => Ok(predicate.clone()),
            })
            .collect()
    }

    pub(crate) fn signature(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
    ) -> Result<SemanticSignature, PlanError> {
        (read_model.signature(&self.alias, binding.bit)).ok_or_else(|| self.no_signature(binding))
    }

    pub(crate) fn definition(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
    ) -> Result<SemanticDefinition, PlanError> {
        (read_model.definition(binding.bit)).ok_or_else(|| self.no_signature(binding))
    }

    pub(crate) fn no_signature(&self, binding: &BoundName) -> PlanError {
        no_signature(&self.path, &self.semantic_source, binding)
    }

    pub(crate) fn physical_source(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<PhysicalSource, PlanError> {
        (read_model.physical_source(&self.alias)).ok_or_else(|| self.broken())
    }

    pub(crate) fn relationship(
        &self,
        read_model: &dyn ReadModel,
    ) -> Result<Relationship, PlanError> {
        (read_model.relationship(&self.alias)).ok_or_else(|| self.broken())
    }

    pub(crate) fn name(
        &self,
        read_model: &dyn ReadModel,
        source: &SourceAlias,
    ) -> Result<DataKindName, PlanError> {
        (read_model.name(source)).ok_or_else(|| self.broken())
    }

    pub(crate) fn bound(
        &self,
        read_model: &dyn ReadModel,
        name: &str,
    ) -> Result<BoundName, PlanError> {
        let bit = (read_model.bit_of(name)).ok_or_else(|| self.broken())?;
        Ok(BoundName {
            name: SemanticsName::new(name),
            bit,
        })
    }

    pub(crate) fn bound_by_bit(
        &self,
        read_model: &dyn ReadModel,
        bit: u32,
    ) -> Result<BoundName, PlanError> {
        let name = (read_model.name_of(bit)).ok_or_else(|| self.broken())?;
        Ok(BoundName { name, bit })
    }

    pub(crate) fn bound_by_id(
        &self,
        read_model: &dyn ReadModel,
        id: &EntityId,
    ) -> Result<BoundName, PlanError> {
        let definition = (read_model.definition_of(id)).ok_or_else(|| self.broken())?;
        Ok(BoundName {
            name: definition.name,
            bit: definition.bit_position,
        })
    }

    pub(crate) fn broken(&self) -> PlanError {
        PlanError::BrokenSource {
            path: self.path.clone(),
            semantic_source: self.semantic_source.clone(),
        }
    }

    pub(crate) fn unbuildable(&self, state: UnbuildableState) -> PlanError {
        PlanError::Unbuildable {
            path: self.path.clone(),
            semantic_source: self.semantic_source.clone(),
            state,
        }
    }

    pub(crate) fn not_allowed(&self, value: DecisionValue) -> PlanError {
        PlanError::DecisionNotAllowed {
            path: self.path.clone(),
            semantic_source: self.semantic_source.clone(),
            value,
        }
    }

    pub(crate) fn tree(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
        root: &PhysicalExprId,
        kept: &[KeptField],
    ) -> Result<PhysicalExpr, PlanError> {
        (read_model.expr_tree(&self.alias, root, kept)).ok_or_else(|| self.broken_tree(binding))
    }

    pub(crate) fn own_tree(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
        kept: &[KeptField],
    ) -> Result<PhysicalExpr, PlanError> {
        let signature = self.signature(read_model, binding)?;
        let tree = self.tree(read_model, binding, &signature.expression_id, kept)?;
        let filters = self.filters_on(read_model, binding)?;
        (filters.iter()).try_fold(tree, |tree, filter| {
            join_filter_onto_aggregates(&tree, filter)
                .map_err(|error| self.refused(call_refused(error)))
        })
    }

    pub(crate) fn filters_on(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
    ) -> Result<Vec<PhysicalExpr>, PlanError> {
        let mut filters = Vec::from_iter(self.own_filter(read_model, binding)?);
        if let Some(condition) = self.conditions.get(&binding.name) {
            filters.push(self.lowered(read_model, condition)?);
        }
        Ok(filters)
    }

    pub(crate) fn lowered(
        &self,
        read_model: &dyn ReadModel,
        tree: &PhysicalExpr,
    ) -> Result<PhysicalExpr, PlanError> {
        let mut lowered = Vec::new();
        for leaf in column_leaves(tree) {
            let name = self.bound(read_model, leaf.name())?;
            let own = if self.null_filled.contains(&name.name) {
                Expr::Literal(Literal::Null(self.null_fill_type(read_model, &name.name)?))
            } else {
                self.own_tree(read_model, &name, &[])?
            };
            lowered.push((leaf.clone(), own));
        }
        replace_columns(tree.clone(), &lowered).map_err(|error| self.refused(error))
    }

    pub(crate) fn null_fill_type(
        &self,
        read_model: &dyn ReadModel,
        name: &SemanticsName,
    ) -> Result<DataType, PlanError> {
        let binding = self.bound(read_model, name.as_str())?;
        null_type(read_model, binding.bit).ok_or_else(|| self.no_signature(&binding))
    }

    pub(crate) fn own_filter(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
    ) -> Result<Option<PhysicalExpr>, PlanError> {
        let signature = self.signature(read_model, binding)?;
        (signature.filter_expr_id.as_ref())
            .map(|filter| self.tree(read_model, binding, filter, &[]))
            .transpose()
    }

    pub(crate) fn linked(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
    ) -> Result<Vec<SemanticDefinition>, PlanError> {
        let root = self.signature(read_model, binding)?.expression_id;
        (read_model.linked(&self.alias, &root)).ok_or_else(|| self.broken_tree(binding))
    }

    pub(crate) fn aggregate_definitions(
        &self,
        read_model: &dyn ReadModel,
        binding: &BoundName,
        kept: &[PhysicalExprId],
    ) -> Result<Vec<SemanticDefinition>, PlanError> {
        let (path, source) = (&self.path, &self.semantic_source);
        aggregate_definitions(read_model, path, source, &self.alias, binding, kept)
    }

    pub(crate) fn broken_tree(&self, binding: &BoundName) -> PlanError {
        broken_tree(&self.path, &self.semantic_source, binding)
    }

    pub(crate) fn refused(&self, error: ValidateError) -> PlanError {
        PlanError::ConstructionRefused {
            path: self.path.clone(),
            error,
        }
    }

    pub(crate) fn node_refused(&self, error: IrErrorKind) -> PlanError {
        PlanError::NodeRefused {
            path: self.path.clone(),
            error,
        }
    }
}

pub(crate) fn aggregate_definitions(
    read_model: &dyn ReadModel,
    path: &FramePath,
    source: &DataKindName,
    alias: &SourceAlias,
    binding: &BoundName,
    kept: &[PhysicalExprId],
) -> Result<Vec<SemanticDefinition>, PlanError> {
    let root = (read_model.signature(alias, binding.bit))
        .ok_or_else(|| no_signature(path, source, binding))?
        .expression_id;
    (read_model.aggregate_definitions(alias, &root, kept))
        .ok_or_else(|| broken_tree(path, source, binding))
}

fn null_type(read_model: &dyn ReadModel, bit: u32) -> Option<DataType> {
    read_model.definition(bit)?.data_type
}

fn no_signature(path: &FramePath, source: &DataKindName, binding: &BoundName) -> PlanError {
    PlanError::NoSignature {
        path: path.clone(),
        semantic_source: source.clone(),
        semantics: binding.name.clone(),
    }
}

fn broken_tree(path: &FramePath, source: &DataKindName, binding: &BoundName) -> PlanError {
    PlanError::BrokenTree {
        path: path.clone(),
        semantic_source: source.clone(),
        semantics: binding.name.clone(),
    }
}

fn prunable(read_model: &dyn ReadModel, unit: &SourceAlias, kept: &BTreeSet<SourceAlias>) -> bool {
    if kept.contains(unit) {
        return true;
    }
    let Some(relationship) = read_model.relationship(unit) else {
        return false;
    };
    let reaches = |end: &&SourceAlias| {
        kept.contains(*end)
            || (read_model.operand_closure(end).iter())
                .any(|constituent| kept.contains(constituent))
    };
    let read: Vec<_> = ([&relationship.from, &relationship.to].into_iter())
        .filter(reaches)
        .collect();
    let leaves_out = match read.as_slice() {
        [alone] => relationship.may_read_alone(alone),
        [_, _] => (kept.iter()).any(|own| {
            (read_model.relationship(own)).is_some_and(|own| own.joins_as(&relationship))
        }),
        _ => false,
    };
    leaves_out && (read.iter()).all(|end| prunable(read_model, end, kept))
}

pub(crate) fn qualifies(alias: &SourceAlias, leaf: &NamedReference) -> bool {
    (leaf.parts().first()).is_some_and(|first| first == alias.as_str())
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FormulaNode {
    pub measure: SemanticsName,
    pub condition: Option<PhysicalExpr>,
    pub index: usize,
}

pub(crate) type FormulaNodes = (
    Vec<FormulaNode>,
    BTreeMap<SemanticsName, Vec<Option<usize>>>,
);

impl FormulaNode {
    pub(crate) fn place(
        nodes: &mut Vec<FormulaNode>,
        measure: SemanticsName,
        condition: Option<PhysicalExpr>,
    ) -> usize {
        let held =
            (nodes.iter()).position(|node| node.measure == measure && node.condition == condition);
        held.unwrap_or_else(|| {
            let index = 1
                + (nodes.iter())
                    .filter(|node| node.measure == measure)
                    .count();
            nodes.push(FormulaNode {
                measure,
                condition,
                index,
            });
            nodes.len() - 1
        })
    }

    pub(crate) fn of_frame(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
    ) -> Result<FormulaNodes, PlanError> {
        let leaves = input.leaves(read_model)?;
        Self::of_reads(read_model, input, input.names.iter().chain(&leaves), &[])
    }

    pub(crate) fn of_readers(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        readers: &[BoundName],
        served: &[BoundName],
    ) -> Result<FormulaNodes, PlanError> {
        Self::of_reads(read_model, input, readers, served)
    }

    fn of_reads<'b>(
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        bindings: impl IntoIterator<Item = &'b BoundName>,
        served: &[BoundName],
    ) -> Result<FormulaNodes, PlanError> {
        let (mut kept, mut roots) = (Vec::new(), Vec::new());
        for link in served {
            let root = input.signature(read_model, link)?.expression_id;
            let field = reference(link.name.as_str()).map_err(|error| input.refused(error))?;
            kept.push(KeptField {
                leaf: KeptLeaf::Link(root),
                field,
            });
            roots.push(root);
        }
        let (mut nodes, mut read) = (input.formula_nodes.clone(), BTreeMap::new());
        for binding in bindings {
            let definition = input.definition(read_model, binding)?;
            if planned(read_model, input, binding, &definition)? != SemanticRole::Metric {
                continue;
            }
            let tree = input.own_tree(read_model, binding, &kept)?;
            let entries = input.aggregate_definitions(read_model, binding, &roots)?;
            if aggregate_calls(&tree).count() != entries.len() {
                return Err(input.broken_tree(binding));
            }
            let mut places = Vec::new();
            for (call, entry) in aggregate_calls(&tree).zip(entries) {
                if planned_as(read_model, &input.alias, &entry) != Some(SemanticRole::Measure) {
                    places.push(None);
                    continue;
                }
                let condition = call.filter().cloned();
                places.push(Some(Self::place(&mut nodes, entry.name, condition)));
            }
            for node in (input.nodes_of(&binding.name)).filter(|n| n.finishes_apart(input)) {
                let apart = input.under(node);
                let tree = apart.own_tree(read_model, binding, &kept)?;
                let entries = apart.aggregate_definitions(read_model, binding, &roots)?;
                if aggregate_calls(&tree).count() != entries.len() {
                    return Err(input.broken_tree(binding));
                }
                for (call, entry) in aggregate_calls(&tree).zip(entries) {
                    if planned_as(read_model, &input.alias, &entry) != Some(SemanticRole::Measure) {
                        places.push(None);
                        continue;
                    }
                    let condition = call.filter().cloned();
                    places.push(Some(Self::place(&mut nodes, entry.name, condition)));
                }
            }
            read.insert(binding.name.clone(), places);
        }
        Ok((nodes, read))
    }

    pub(crate) fn finishes_apart(&self, demand: &StrategyInput) -> bool {
        demand.conditions.get(&self.measure) != self.condition.as_ref()
    }

    pub(crate) fn in_measure_field(&self, demand: &StrategyInput) -> bool {
        demand.delivery == Delivery::PartialAggregates && !self.finishes_apart(demand)
    }

    pub(crate) fn stem(&self) -> String {
        format!("node{}", self.index)
    }

    pub(crate) fn field(
        &self,
        demand: &StrategyInput,
        qualifier: Option<&str>,
    ) -> Result<(Name, PhysicalExpr), ValidateError> {
        let stem = if self.in_measure_field(demand) {
            None
        } else {
            Some(self.stem())
        };
        stemmed_entry(qualifier, stem.as_deref(), &self.measure, 1, true)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetArm {
    pub physical_source: PhysicalSource,
    pub selector: Selector,
}

impl Relationship {
    fn refused(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        decision: &Decision,
    ) -> Result<Option<DecisionValue>, PlanError> {
        if let Some(foreign) = decision.foreign(read_model, &[self.from.clone(), self.to.clone()]) {
            return Ok(Some(foreign));
        }
        let ends = self.ends().map(|(end, _)| end);
        let combine = decision.combine;
        let not_read = |end: &SourceAlias| input.name(read_model, end).map(DecisionValue::NotRead);
        let refused = match (&decision.stand_in, combine, decision.operands.as_slice()) {
            (None, Combine::Join | Combine::Cross, _) => {
                let join_type = (combine != Combine::from(self.join_type))
                    .then_some(DecisionValue::Combine(combine));
                (decision.unread(ends).map(not_read).transpose()?).or(join_type)
            }
            (None, Combine::PassThrough, [kept]) => {
                let kept = &kept.source;
                let alone = input.reads_alone(read_model, self, kept)?;
                (ends.into_iter().find(|end| *end != kept))
                    .filter(|_| !alone)
                    .map(not_read)
                    .transpose()?
            }
            (Some(stand_in), Combine::PassThrough, []) => {
                let answers = input.may_stand_in(read_model, stand_in);
                (!answers).then(|| DecisionValue::StandIn(stand_in.clone()))
            }
            (Some(_), _, _) | (None, Combine::PassThrough, _) => {
                Some(DecisionValue::Combine(combine))
            }
            (None, _, _) => None,
        };
        Ok(refused)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct UnionsetArm {
    pub branches: Vec<SourceAlias>,
    pub mode: MultisetOp,
    pub null_fill: bool,
}

pub(crate) struct Branches<'a> {
    pub(crate) names: Vec<BoundName>,
    pub(crate) read: Vec<&'a SourceAlias>,
    pub(crate) unservable: Vec<(&'a SourceAlias, Vec<SemanticsName>)>,
}

impl UnionsetArm {
    pub(crate) fn of(read_model: &dyn ReadModel, alias: &SourceAlias) -> Option<Self> {
        match read_model.source_kind(alias)? {
            RoleDefinition::Unionset {
                mode, null_fill, ..
            } => Some(Self {
                branches: read_model.operands(alias),
                mode,
                null_fill,
            }),
            _ => None,
        }
    }

    pub(crate) fn names(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
    ) -> Result<Vec<BoundName>, PlanError> {
        let names = (input.read_leaves(read_model)?.iter())
            .map(|leaf| input.bound(read_model, leaf))
            .collect::<Result<Vec<_>, _>>()?;
        if matches!(self.mode, MultisetOp::UnionDistinct) {
            let keyset =
                (read_model.keysets(&input.alias).first()).map(|mask| mask.set_positions());
            return input.with_bits(read_model, names, keyset.unwrap_or_default());
        }
        Ok(names)
    }

    pub(crate) fn branches<'a>(
        &'a self,
        read_model: &'a dyn ReadModel,
        input: &'a StrategyInput,
    ) -> Result<Branches<'a>, PlanError> {
        let names = self.names(read_model, input)?;
        let (mut read, mut unservable) = (Vec::new(), Vec::new());
        let mut placing = input.clone();
        let placed = place(&mut placing);
        let aggregating = Aggregating::of(read_model, &placing, &placed)?;
        for branch in self.required(read_model, &input.predicates) {
            let operand = self.operand(read_model, branch, &names);
            let filled = input.filling(&operand);
            let asked = (is_grainset(read_model, branch))
                .then(|| {
                    aggregating.operand(read_model, &placing, &filled, Delivery::PartialAggregates)
                })
                .transpose()?;
            let lacks = match asked {
                Some(asked) => self.lacks(read_model, branch, &asked)?,
                None => Vec::new(),
            };
            if lacks.is_empty() {
                read.push(branch);
            } else {
                unservable.push((branch, lacks));
            }
        }
        if read.is_empty() {
            read = unservable.drain(..).map(|(branch, _)| branch).collect();
        }
        Ok(Branches {
            names,
            read,
            unservable,
        })
    }

    fn lacks(
        &self,
        read_model: &dyn ReadModel,
        branch: &SourceAlias,
        asked: &StrategyInput,
    ) -> Result<Vec<SemanticsName>, PlanError> {
        let children = read_model.operands(branch);
        let conditions = asked.condition_reads(read_model)?;
        if maps_whole(read_model, &children, &asked.demanded, &conditions) {
            return Ok(Vec::new());
        }
        let bits = with_conditions(&asked.demanded, &conditions);
        let reads = asked.with_bits(read_model, asked.names.clone(), bits.set_positions())?;
        let nearest = nearest_children(read_model, &Structural, &reads, branch);
        Ok((reads.iter())
            .map(|name| &name.name)
            .filter(|name| nearest.iter().any(|(_, lacks)| lacks.contains(name)))
            .cloned()
            .collect())
    }

    pub(crate) fn unservable(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
    ) -> Result<Vec<Lacking>, PlanError> {
        let branches = self.branches(read_model, input)?;
        (branches.unservable.into_iter())
            .map(|(branch, lacks)| Ok((input.name(read_model, branch)?, lacks)))
            .collect()
    }

    pub(crate) fn operand(
        &self,
        read_model: &dyn ReadModel,
        branch: &SourceAlias,
        names: &[BoundName],
    ) -> Operand {
        let mut operand = Operand {
            source: branch.clone(),
            names: Vec::new(),
            null_filled: Vec::new(),
        };
        for name in names {
            let served = if self.null_fills(read_model, branch, name.bit) {
                &mut operand.null_filled
            } else {
                &mut operand.names
            };
            served.push(name.name.clone());
        }
        operand
    }

    pub(crate) fn null_fills(
        &self,
        read_model: &dyn ReadModel,
        branch: &SourceAlias,
        bit: u32,
    ) -> bool {
        self.null_fill && !read_model.mask(branch).contains(bit)
    }

    fn refused(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        decision: &Decision,
    ) -> Result<Option<DecisionValue>, PlanError> {
        if let Some(foreign) = decision.foreign(read_model, &self.branches) {
            return Ok(Some(foreign));
        }
        let branches = self.branches(read_model, input)?;
        (decision.unread(branches.read))
            .map(|branch| input.name(read_model, branch).map(DecisionValue::NotRead))
            .transpose()
    }

    pub(crate) fn required<'a>(
        &'a self,
        read_model: &'a dyn ReadModel,
        predicates: &'a [Predicate],
    ) -> impl Iterator<Item = &'a SourceAlias> + 'a {
        let excludes = move |branch: &SourceAlias| {
            let literal_of = |leaf: &SemanticLeaf| match Read::of(leaf)? {
                Read::Semantics {
                    name,
                    accessor: false,
                    ..
                } => {
                    let bit = read_model.bit_of(name.as_str())?;
                    if self.null_fills(read_model, branch, bit) {
                        return null_type(read_model, bit)
                            .map(|null| Cow::Owned(Literal::Null(null)));
                    }
                    literal_on(read_model, branch, bit).map(Cow::Owned)
                }
                _ => None,
            };
            let folds_false = |predicate: &Predicate| {
                let folded = predicate.tree.clone().fold_with(&literal_of);
                let value = match &folded {
                    Expr::Leaf(leaf) => literal_of(leaf),
                    Expr::Literal(literal) => Some(Cow::Borrowed(literal)),
                    _ => None,
                };
                matches!(
                    value.as_deref(),
                    Some(Literal::Boolean(false) | Literal::Null(_))
                )
            };
            (predicates.iter()).any(|predicate| predicate.in_branches() && folds_false(predicate))
        };
        (self.branches.iter()).filter(move |branch| !excludes(branch))
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GrainsetArm {
    pub children: Vec<SourceAlias>,
}

impl GrainsetArm {
    fn refused(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
        decision: &Decision,
    ) -> Result<Option<DecisionValue>, PlanError> {
        let child = |stand_in: &&DataKindName| {
            (read_model.alias(stand_in.as_str()))
                .is_some_and(|alias| self.children.contains(&alias))
        };
        if let Some(stand_in) = (decision.stand_in.as_ref()).filter(|stand_in| !child(stand_in)) {
            return Ok(Some(DecisionValue::StandIn(stand_in.clone())));
        }
        if let Some(foreign) = decision.foreign(read_model, &self.children) {
            return Ok(Some(foreign));
        }
        let combine = decision.combine;
        match (&decision.stand_in, combine, decision.operands.len()) {
            (Some(_), Combine::PassThrough, 0) => Ok(None),
            (None, Combine::DrillAcross, 2..) => {
                let served = |name: &BoundName| {
                    (decision.operands.iter()).any(|operand| operand.names.contains(&name.name))
                };
                let not_read = (self.children.iter()).find(|&child| {
                    let mask = read_model.mask(child);
                    decision.unread([child]).is_some()
                        && (input.names.iter()).any(|name| mask.contains(name.bit) && !served(name))
                });
                (not_read.map(|child| input.name(read_model, child).map(DecisionValue::NotRead)))
                    .transpose()
            }
            (_, Combine::PassThrough | Combine::DrillAcross, _) => {
                Ok(Some(DecisionValue::Combine(combine)))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Strategies {
    pub(crate) dataset: Box<dyn Strategy<DatasetArm>>,
    pub(crate) joinset: Box<dyn Strategy<Relationship>>,
    pub(crate) unionset: Box<dyn Strategy<UnionsetArm>>,
    pub(crate) grainset: Box<dyn Strategy<GrainsetArm>>,
}

impl Default for Strategies {
    fn default() -> Self {
        Self {
            dataset: Box::new(DatasetStrategy),
            joinset: Box::new(JoinsetStrategy),
            unionset: Box::new(UnionsetStrategy),
            grainset: Box::new(GrainsetStrategy),
        }
    }
}

impl Strategies {
    pub(crate) fn dispatch(
        &self,
        read_model: &dyn ReadModel,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        let source = read_model.source_kind(&input.alias);
        let (decision, allowed) = match source.ok_or_else(|| input.broken())? {
            RoleDefinition::Dataset { selector, .. } => {
                let arm = DatasetArm {
                    physical_source: input.physical_source(read_model)?,
                    selector,
                };
                (
                    self.dataset.decide(read_model, &arm, input)?,
                    &Allowed::DATASET,
                )
            }
            RoleDefinition::Joinset { .. } => {
                let relationship = input.relationship(read_model)?;
                let decision = self.joinset.decide(read_model, &relationship, input)?;
                if let Some(value) = relationship.refused(read_model, input, &decision)? {
                    return Err(input.not_allowed(value));
                }
                (decision, &Allowed::JOINSET)
            }
            RoleDefinition::Unionset { .. } => {
                let arm =
                    UnionsetArm::of(read_model, &input.alias).ok_or_else(|| input.broken())?;
                let decision = self.unionset.decide(read_model, &arm, input)?;
                if decision.stand_in.is_none() {
                    if let Some(value) = arm.refused(read_model, input, &decision)? {
                        return Err(input.not_allowed(value));
                    }
                }
                (decision, &Allowed::UNIONSET)
            }
            RoleDefinition::Grainset { .. } => {
                let arm = GrainsetArm {
                    children: read_model.operands(&input.alias),
                };
                let decision = self.grainset.decide(read_model, &arm, input)?;
                if let Some(value) = arm.refused(read_model, input, &decision)? {
                    return Err(input.not_allowed(value));
                }
                (decision, &Allowed::GRAINSET)
            }
            role_definition => {
                return Err(PlanError::StrategyNotFoundForSemanticSource {
                    path: input.path.clone(),
                    semantic_source: input.semantic_source.clone(),
                    role_definition,
                })
            }
        };
        match allowed.refused(read_model, &decision) {
            Some(value) => Err(input.not_allowed(value)),
            None => Ok(decision),
        }
    }
}
