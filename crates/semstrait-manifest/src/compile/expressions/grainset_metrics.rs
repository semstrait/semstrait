use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::{EntityId, Granularity, SemanticsName, Tree};
use semstrait_ir::{SemanticRole, TemporalType};
use semstrait_model::{SemanticInterface, UnresolvedSemanticExpr};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, NamedReference};

use super::bodies::{self, SourceBodies, SourceFilters};
use super::projection::{
    self, alias_column, any_operand_serves, ComposedFilters, ComposedLowering, ComposedProjection,
};
use crate::compile::assemble::{seal_signatures, ManifestAssembly, SourcePlan, Vocabulary};
use crate::compile::definitions;
use crate::compile::expressions;
use crate::compile::keysets::KeysetPlan;
use crate::compile::types::{Provenance, Reference, ResolvedExprEntry, SemanticsIndex};
use crate::error::CompileError;
use crate::identity::{mint_expr_id, mint_signature_id};
use crate::manifest::expression::{ExprRoot, Link, PhysicalExprId};
use crate::manifest::expression_layer::ExprLayer;
use crate::manifest::final_values::{final_value_layer, ChildPools};
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::semantic_source::RoleDefinition;
use crate::manifest::serving::Serving;
use crate::manifest::source_alias::SourceAliases;

#[derive(Clone, Copy)]
pub(super) struct GrainsetSite<'a, 'd> {
    pub name: &'a str,
    pub public: bool,
    pub role_definition: &'d RoleDefinition,
    pub index: &'a SemanticsIndex<'d>,
    pub definitions: &'d [SemanticDefinition],
    pub assembly: &'a ManifestAssembly,
    pub keysets: &'a KeysetPlan,
}

struct Candidate<'d> {
    name: SemanticsName,
    definition: &'d SemanticDefinition,
    body: Vec<SemanticsName>,
    filter: Vec<SemanticsName>,
}

impl Candidate<'_> {
    fn reads(&self) -> impl Iterator<Item = &SemanticsName> {
        self.body.iter().chain(&self.filter)
    }
}

#[derive(Clone, Default)]
struct Scope<'d> {
    filter: Vec<&'d SemanticDefinition>,
    formula_grain: Vec<&'d SemanticDefinition>,
    part_key: Vec<&'d SemanticDefinition>,
}

impl<'d> Scope<'d> {
    fn key(&self) -> [BTreeSet<&'d EntityId>; 3] {
        [
            ids(&self.filter),
            ids(&self.formula_grain),
            ids(&self.part_key),
        ]
    }
}

fn ids<'d>(read: &[&'d SemanticDefinition]) -> BTreeSet<&'d EntityId> {
    read.iter().map(|definition| &definition.id).collect()
}

type Visited<'d> = BTreeSet<(&'d EntityId, [BTreeSet<&'d EntityId>; 3])>;

enum BodyRead<'d> {
    Linked(&'d SemanticDefinition),
    Plain(&'d SemanticDefinition),
}

type BodyReads<'b, 'd> = &'b dyn Fn(&SemanticDefinition) -> Vec<BodyRead<'d>>;

fn grains(role_definition: &RoleDefinition) -> &[EntityId] {
    match role_definition {
        RoleDefinition::Grainset { grains, .. } => grains,
        _ => &[],
    }
}

struct ConditionedRead<'d> {
    definition: &'d SemanticDefinition,
    scope: Scope<'d>,
}

impl<'d> ConditionedRead<'d> {
    fn names(&self) -> BTreeSet<&'d EntityId> {
        let scope = &self.scope;
        let scoped = (scope.filter.iter())
            .chain(&scope.formula_grain)
            .chain(&scope.part_key);
        let ids = scoped.map(|definition| &definition.id);
        ids.chain([&self.definition.id]).collect()
    }
}

impl<'a, 'd> GrainsetSite<'a, 'd> {
    fn serving(&self) -> Serving<'a> {
        Serving::new(&self.assembly.semantic_sources, &self.assembly.signatures)
    }

    fn children(&self) -> &'d [EntityId] {
        grains(self.role_definition)
    }

    fn child_serves(&self, definition: &SemanticDefinition) -> bool {
        any_operand_serves(self.role_definition, &definition.id, self.serving())
    }

    fn some_child_maps_whole(&self, names: &BTreeSet<&EntityId>) -> bool {
        (self.children().iter()).any(|child| self.maps_whole(child, names))
    }

    fn maps_whole(&self, source: &EntityId, names: &BTreeSet<&EntityId>) -> bool {
        let computed_reads = |grainset: &EntityId, definition: &EntityId| {
            let mut leaves = Vec::new();
            self.stored_leaves(
                (grainset, definition),
                (&Scope::default(), false),
                &mut leaves,
                &mut BTreeSet::new(),
            );
            (leaves.iter())
                .map(|leaf| leaf.names().into_iter().cloned().collect())
                .collect()
        };
        self.serving()
            .maps_whole_reading(source, names, &computed_reads)
    }

    fn columns_read(&self, root: &PhysicalExprId) -> Vec<&'d SemanticDefinition> {
        (self.assembly.expressions.columns(root).into_iter())
            .filter_map(|(_, column)| {
                (self.definitions.iter())
                    .find(|definition| definition.name.as_str() == column.name())
            })
            .collect()
    }

    fn stored_reads(&self, grainset: &EntityId, metric: &SemanticDefinition) -> Vec<BodyRead<'d>> {
        let signatures = &self.assembly.signatures;
        let Some(signature) = signatures.get(&mint_signature_id(grainset, &metric.id)) else {
            return Vec::new();
        };
        let plain = (self.columns_read(&signature.expression_id).into_iter()).map(BodyRead::Plain);
        let links: BTreeSet<PhysicalExprId> = (self.assembly.expressions)
            .links_of(&signature.expression_id)
            .collect();
        let owned = (self.assembly.semantic_sources.get(grainset).into_iter())
            .flat_map(|source| &source.semantic_signature_ids)
            .filter_map(|id| signatures.get(id));
        let linked = owned
            .filter(|held| links.contains(&held.expression_id))
            .filter_map(|held| {
                (self.definitions.iter()).find(|definition| definition.id == held.definition_id)
            })
            .map(BodyRead::Linked);
        plain.chain(linked).collect()
    }

    fn stored_leaves(
        &self,
        (grainset, metric): (&EntityId, &EntityId),
        (outer, keys_in_scope): (&Scope<'d>, bool),
        leaves: &mut Vec<ConditionedRead<'d>>,
        visited: &mut Visited<'d>,
    ) {
        let (Some(source), Some(signature), Some(definition)) = (
            self.assembly.semantic_sources.get(grainset),
            (self.assembly.signatures).get(&mint_signature_id(grainset, metric)),
            (self.definitions.iter()).find(|definition| definition.id == *metric),
        ) else {
            return;
        };
        if !visited.insert((&definition.id, outer.key())) {
            return;
        }
        let body = self.stored_reads(grainset, definition);
        let role_definition = &source.role_definition;
        let mut scope = outer.clone();
        if let Some(filter) = &signature.filter_expr_id {
            let dimensions = (self.columns_read(filter).into_iter())
                .filter(|read| self.is_dimension_in(role_definition, read));
            scope.filter.extend(dimensions);
        }
        let aggregates = definition.role_definition.aggregation().is_some();
        if aggregates {
            let dimensions = body.iter().filter_map(|read| match read {
                BodyRead::Plain(plain) if self.is_dimension_in(role_definition, plain) => {
                    Some(*plain)
                }
                _ => None,
            });
            scope.formula_grain.extend(dimensions);
        } else if !keys_in_scope {
            let reads = |metric: &SemanticDefinition| self.stored_reads(grainset, metric);
            self.key_reads(
                definition,
                &reads,
                role_definition,
                &mut scope.part_key,
                &mut BTreeSet::new(),
            );
        }
        for read in body {
            match read {
                BodyRead::Plain(leaf)
                    if !self.is_dimension_in(role_definition, leaf)
                        && !scope.part_key.iter().any(|key| key.id == leaf.id) =>
                {
                    leaves.push(ConditionedRead {
                        definition: leaf,
                        scope: scope.clone(),
                    });
                }
                BodyRead::Plain(_) => {}
                BodyRead::Linked(linked) => self.stored_leaves(
                    (grainset, &linked.id),
                    (&scope, !aggregates),
                    leaves,
                    visited,
                ),
            }
        }
    }

    fn dimensions<'n>(
        &self,
        names: impl IntoIterator<Item = &'n SemanticsName>,
    ) -> Vec<&'d SemanticDefinition> {
        (names.into_iter())
            .filter_map(|name| self.index.get(name))
            .filter(|definition| self.is_dimension(definition))
            .collect()
    }

    fn is_dimension(&self, definition: &SemanticDefinition) -> bool {
        self.is_dimension_in(self.role_definition, definition)
    }

    fn is_dimension_in(&self, grainset: &RoleDefinition, definition: &SemanticDefinition) -> bool {
        definition.role() == SemanticRole::Dimension || self.is_row_level_in(grainset, definition)
    }

    fn candidates(
        &self,
        bodies: &SourceBodies<'_>,
        filters: &SourceFilters<'_>,
    ) -> BTreeMap<SemanticsName, Candidate<'d>> {
        let filter_of: BTreeMap<&SemanticsName, &&UnresolvedSemanticExpr> =
            filters.into_iter().collect();
        let mut candidates = BTreeMap::new();
        for (name, body) in bodies {
            let Some(definition) = (self.index.get(name))
                .filter(|definition| definition.role() == SemanticRole::Metric)
                .filter(|definition| !self.child_serves(definition))
            else {
                continue;
            };
            let candidate = Candidate {
                name: name.clone(),
                definition,
                body: referenced_in_order(body),
                filter: filter_of
                    .get(name)
                    .map(|f| referenced_in_order(f))
                    .unwrap_or_default(),
            };
            candidates.insert(name.clone(), candidate);
        }
        let aggregating: BTreeSet<SemanticsName> = (candidates.values())
            .filter(|candidate| {
                (candidate.body.iter())
                    .any(|read| self.aggregate_valued(read, &candidates, &mut BTreeSet::new()))
            })
            .map(|candidate| candidate.name.clone())
            .collect();
        candidates.retain(|name, _| aggregating.contains(name));
        candidates
    }

    fn aggregate_valued(
        &self,
        name: &SemanticsName,
        candidates: &BTreeMap<SemanticsName, Candidate<'d>>,
        visiting: &mut BTreeSet<SemanticsName>,
    ) -> bool {
        let Some(definition) = self.index.get(name) else {
            return false;
        };
        match definition.role() {
            SemanticRole::Measure => true,
            SemanticRole::Metric if definition.role_definition.aggregation().is_some() => true,
            SemanticRole::Metric => match candidates.get(name) {
                Some(candidate) if visiting.insert(name.clone()) => (candidate.body.iter())
                    .any(|read| self.aggregate_valued(read, candidates, visiting)),
                Some(_) => false,
                None => {
                    self.child_serves(definition)
                        && self.pools().final_layer(self.children(), definition)
                            != ExprLayer::Scalar
                }
            },
            _ => false,
        }
    }

    fn pools(&self) -> ChildPools<'a> {
        ChildPools {
            serving: self.serving(),
            expressions: &self.assembly.expressions,
        }
    }

    fn verdict(
        &self,
        candidate: &Candidate<'d>,
        candidates: &BTreeMap<SemanticsName, Candidate<'d>>,
        computed: &BTreeSet<SemanticsName>,
        bodies: &SourceBodies<'_>,
    ) -> Result<(), CompileError> {
        let unmapped = candidate.reads().find(|read| {
            !computed.contains(*read)
                && !(self.index.get(read)).is_some_and(|definition| self.child_serves(definition))
        });
        if let Some(name) = unmapped {
            return Err(CompileError::GrainsetMetricReadsUnmapped {
                grainset: self.name.to_owned(),
                metric: candidate.name.as_str().to_owned(),
                name: name.as_str().to_owned(),
            });
        }
        let mut leaves = Vec::new();
        self.collect_leaves(
            candidate,
            candidates,
            computed,
            &Scope::default(),
            &mut leaves,
            &mut BTreeSet::new(),
        );
        self.check_filters_mapped(candidate, &leaves)?;
        self.check_part_key(candidate, candidates, computed, &leaves, bodies)?;
        self.check_shared_keyset(candidate, &leaves, bodies)
    }

    fn candidate_reads(
        &self,
        candidates: &BTreeMap<SemanticsName, Candidate<'d>>,
        computed: &BTreeSet<SemanticsName>,
        metric: &SemanticDefinition,
    ) -> Vec<BodyRead<'d>> {
        let body = (candidates.get(&metric.name)).map_or(&[][..], |candidate| &candidate.body);
        (body.iter())
            .filter_map(|name| {
                let definition = self.index.get(name)?;
                Some(
                    match candidates.get(name).filter(|_| computed.contains(name)) {
                        Some(linked) => BodyRead::Linked(linked.definition),
                        None => BodyRead::Plain(definition),
                    },
                )
            })
            .collect()
    }

    fn key_reads(
        &self,
        metric: &'d SemanticDefinition,
        body: BodyReads<'_, 'd>,
        grainset: &RoleDefinition,
        reads: &mut Vec<&'d SemanticDefinition>,
        visited: &mut BTreeSet<&'d EntityId>,
    ) {
        if !visited.insert(&metric.id) {
            return;
        }
        for read in body(metric) {
            match read {
                BodyRead::Linked(linked) if linked.role_definition.aggregation().is_none() => {
                    self.key_reads(linked, body, grainset, reads, visited);
                }
                BodyRead::Linked(_) => {}
                BodyRead::Plain(definition)
                    if self.is_dimension_in(grainset, definition)
                        && !reads.iter().any(|read| read.id == definition.id) =>
                {
                    reads.push(definition);
                }
                BodyRead::Plain(_) => {}
            }
        }
    }

    fn is_row_level_in(&self, grainset: &RoleDefinition, definition: &SemanticDefinition) -> bool {
        definition.role() == SemanticRole::Metric
            && definition.role_definition.aggregation().is_none()
            && any_operand_serves(grainset, &definition.id, self.serving())
            && self.pools().final_layer(grains(grainset), definition) == ExprLayer::Scalar
    }

    fn axes_read(
        &self,
        read: &'d SemanticDefinition,
        bodies: &SourceBodies<'_>,
    ) -> Vec<&'d SemanticDefinition> {
        let mut axes = Vec::new();
        let mut seen = BTreeSet::new();
        let mut pending = vec![read];
        while let Some(read) = pending.pop() {
            if !seen.insert(&read.id) {
                continue;
            }
            if read.role() == SemanticRole::Dimension {
                axes.push(read);
                continue;
            }
            let names =
                (bodies.get(&read.name)).map_or_else(Vec::new, |body| referenced_in_order(body));
            let reads = (names.iter()).filter_map(|name| self.index.get(name));
            pending.extend(reads.rev());
        }
        axes
    }

    fn check_part_key(
        &self,
        candidate: &Candidate<'d>,
        candidates: &BTreeMap<SemanticsName, Candidate<'d>>,
        computed: &BTreeSet<SemanticsName>,
        leaves: &[ConditionedRead<'d>],
        bodies: &SourceBodies<'_>,
    ) -> Result<(), CompileError> {
        if candidate.definition.role_definition.aggregation().is_some() {
            return Ok(());
        }
        let mut reads = Vec::new();
        let body = |metric: &SemanticDefinition| self.candidate_reads(candidates, computed, metric);
        self.key_reads(
            candidate.definition,
            &body,
            self.role_definition,
            &mut reads,
            &mut BTreeSet::new(),
        );
        let unformed = |leaf: &ConditionedRead<'d>, read: &SemanticDefinition, child: &EntityId| {
            CompileError::GrainsetMetricPartKeyUnformed {
                grainset: self.name.to_owned(),
                metric: candidate.name.as_str().to_owned(),
                measure: leaf.definition.name.as_str().to_owned(),
                dimension: read.name.as_str().to_owned(),
                child: self.child_name(child),
            }
        };
        for leaf in leaves {
            let mut names = leaf.names();
            for read in &reads {
                let passed = names.clone();
                names.insert(&read.id);
                if self.some_child_maps_whole(&names) {
                    continue;
                }
                let gives = |child: &&EntityId| self.serving().serves(child, &leaf.definition.id);
                let failing = (self.children().iter().filter(gives))
                    .find(|child| self.maps_whole(child, &passed))
                    .or_else(|| self.children().iter().find(gives));
                if let Some(child) = failing {
                    return Err(unformed(leaf, read, child));
                }
            }
        }
        let forming: Vec<(&ConditionedRead<'d>, Vec<&'d EntityId>)> = (leaves.iter())
            .map(|leaf| {
                let mut names = leaf.names();
                names.extend(reads.iter().map(|read| &read.id));
                let children = (self.children().iter()).filter(|child| {
                    self.serving().serves(child, &leaf.definition.id)
                        && self.maps_whole(child, &names)
                });
                (leaf, children.collect())
            })
            .collect();
        let axes = (reads.iter()).flat_map(|read| self.axes_read(read, bodies));
        for axis in axes {
            let grains: Vec<Granularity> = (forming.iter())
                .flat_map(|(_, children)| children)
                .filter_map(|child| self.axis_grain(child, &axis.id))
                .collect();
            for (leaf, children) in &forming {
                for child in children {
                    let Some(grain) = self.axis_grain(child, &axis.id) else {
                        continue;
                    };
                    if grains.iter().any(|finer| !grain.serves(*finer)) {
                        return Err(unformed(leaf, axis, child));
                    }
                }
            }
        }
        Ok(())
    }

    fn axis_grain(&self, child: &EntityId, dimension: &EntityId) -> Option<Granularity> {
        let shape = self
            .assembly
            .semantic_sources
            .get(child)?
            .temporal
            .as_ref()?;
        let on_axis = match &shape.temporal_type {
            TemporalType::Scd {
                valid_from,
                valid_to,
                ..
            } => valid_from == dimension || valid_to == dimension,
            other => other.time_axis() == Some(dimension),
        };
        on_axis.then_some(shape.granularity).flatten()
    }

    fn child_name(&self, child: &EntityId) -> String {
        (self.assembly.semantic_sources.get(child)).map_or_else(
            || child.as_str().to_owned(),
            |source| source.name.as_str().to_owned(),
        )
    }

    fn collect_leaves(
        &self,
        candidate: &Candidate<'d>,
        candidates: &BTreeMap<SemanticsName, Candidate<'d>>,
        computed: &BTreeSet<SemanticsName>,
        outer: &Scope<'d>,
        leaves: &mut Vec<ConditionedRead<'d>>,
        visited: &mut Visited<'d>,
    ) {
        if !visited.insert((&candidate.definition.id, outer.key())) {
            return;
        }
        let mut scope = outer.clone();
        scope.filter.extend(self.dimensions(&candidate.filter));
        if candidate.definition.role_definition.aggregation().is_some() {
            let body =
                |metric: &SemanticDefinition| self.candidate_reads(candidates, computed, metric);
            self.key_reads(
                candidate.definition,
                &body,
                self.role_definition,
                &mut scope.formula_grain,
                &mut BTreeSet::new(),
            );
        }
        for name in &candidate.body {
            let Some(definition) = self.index.get(name) else {
                continue;
            };
            match candidates.get(name).filter(|_| computed.contains(name)) {
                Some(linked) => {
                    self.collect_leaves(linked, candidates, computed, &scope, leaves, visited);
                }
                None if self.is_dimension(definition) => {}
                None => leaves.push(ConditionedRead {
                    definition,
                    scope: scope.clone(),
                }),
            }
        }
    }

    fn check_filters_mapped(
        &self,
        candidate: &Candidate<'d>,
        leaves: &[ConditionedRead<'d>],
    ) -> Result<(), CompileError> {
        for leaf in leaves {
            let mut names = BTreeSet::from([&leaf.definition.id]);
            for dimension in &leaf.scope.filter {
                names.insert(&dimension.id);
                if !self.some_child_maps_whole(&names) {
                    return Err(CompileError::GrainsetMetricFilterUnmapped {
                        grainset: self.name.to_owned(),
                        metric: candidate.name.as_str().to_owned(),
                        measure: leaf.definition.name.as_str().to_owned(),
                        dimension: dimension.name.as_str().to_owned(),
                    });
                }
            }
        }
        Ok(())
    }

    fn check_shared_keyset(
        &self,
        candidate: &Candidate<'d>,
        leaves: &[ConditionedRead<'d>],
        bodies: &SourceBodies<'_>,
    ) -> Result<(), CompileError> {
        let all: BTreeSet<&EntityId> = leaves.iter().flat_map(ConditionedRead::names).collect();
        let shared = self.some_child_maps_whole(&all)
            || self.keyset_holders().iter().any(|holders| {
                leaves.iter().all(|leaf| {
                    let names = leaf.names();
                    holders.iter().any(|holder| self.maps_whole(holder, &names))
                })
            });
        if shared && self.formula_grain_at_one_granularity(leaves, bodies) {
            return Ok(());
        }
        Err(CompileError::GrainsetMetricWithoutKeyset {
            grainset: self.name.to_owned(),
            metric: candidate.name.as_str().to_owned(),
        })
    }

    fn formula_grain_at_one_granularity(
        &self,
        leaves: &[ConditionedRead<'d>],
        bodies: &SourceBodies<'_>,
    ) -> bool {
        let mut by_axis: BTreeMap<&'d EntityId, Vec<Granularity>> = BTreeMap::new();
        for leaf in leaves {
            let names = leaf.names();
            let giving: Vec<&'d EntityId> = (self.children().iter())
                .filter(|child| {
                    self.serving().serves(child, &leaf.definition.id)
                        && self.maps_whole(child, &names)
                })
                .collect();
            let axes =
                (leaf.scope.formula_grain.iter()).flat_map(|read| self.axes_read(read, bodies));
            for axis in axes {
                let read = (giving.iter()).filter_map(|child| self.axis_grain(child, &axis.id));
                by_axis.entry(&axis.id).or_default().extend(read);
            }
        }
        by_axis.values().all(|grains| {
            (grains.iter()).all(|grain| grains.iter().all(|finer| grain.serves(*finer)))
        })
    }

    fn keyset_holders(&self) -> Vec<Vec<&'d EntityId>> {
        let keysets = (self.children().iter()).flat_map(|child| self.keysets.guaranteed(child));
        keysets
            .map(|keyset| {
                let members: BTreeSet<&EntityId> = keyset.members.iter().collect();
                (self.children().iter())
                    .filter(|grain| self.maps_whole(grain, &members))
                    .collect()
            })
            .collect()
    }
}

fn referenced_in_order(expr: &UnresolvedSemanticExpr) -> Vec<SemanticsName> {
    let mut read = Vec::new();
    definitions::collect_referenced_semantics(expr, &mut read);
    let mut seen = BTreeSet::new();
    read.retain(|name| seen.insert(name.clone()));
    read
}

fn computed<'m>(
    site: &GrainsetSite<'_, '_>,
    mut bodies: SourceBodies<'m>,
    filters: &SourceFilters<'_>,
) -> Result<SourceBodies<'m>, CompileError> {
    let candidates = site.candidates(&bodies, filters);
    let mut undecided: Vec<&Candidate<'_>> = candidates.values().collect();
    let mut computed = BTreeSet::new();
    let mut refusals = Vec::new();
    loop {
        let pending: BTreeSet<&SemanticsName> = undecided.iter().map(|c| &c.name).collect();
        let ready = undecided
            .iter()
            .position(|candidate| candidate.reads().all(|read| !pending.contains(read)));
        let Some(position) = ready else { break };
        let candidate = undecided.remove(position);
        match site.verdict(candidate, &candidates, &computed, &bodies) {
            Ok(()) => {
                computed.insert(candidate.name.clone());
            }
            Err(refusal) => refusals.push(refusal),
        }
    }
    if let Some(refusal) = refusals.into_iter().next().filter(|_| site.public) {
        return Err(refusal);
    }
    bodies.retain(|name| computed.contains(name));
    Ok(bodies)
}

pub(super) fn final_values<'d>(
    site: &GrainsetSite<'_, 'd>,
    interface: &SemanticInterface,
    own_bodies: &SourceBodies<'_>,
    alias: &str,
) -> Result<Vec<ComposedProjection<'d>>, CompileError> {
    let serving = site.serving();
    let mut projections = Vec::new();
    for name in definitions::exposed_semantics(interface) {
        let Some(definition) = site.index.get(&name) else {
            continue;
        };
        if own_bodies.contains_key(&name) || definition.role() == SemanticRole::Filter {
            continue;
        }
        let operands: Vec<&'d EntityId> = (site.children().iter())
            .filter(|child| serving.serves(child, &definition.id))
            .collect();
        if operands.is_empty() {
            continue;
        }
        let column = alias_column(alias, name.as_str(), site.name)?;
        let read_type = definition.data_type.clone().unwrap_or(DataType::Unknown);
        let entry = ResolvedExprEntry {
            expr: column.clone(),
            inferred_type: read_type.clone(),
            family: read_type.family(),
            argument_type: None,
            referenced_columns: Vec::new(),
            provenance: Provenance::new(),
        };
        projections.push(ComposedProjection {
            name,
            definition,
            column,
            operands,
            entry,
            read_type,
        });
    }
    Ok(projections)
}

#[derive(Default)]
pub(super) struct FinalValueLayers {
    alias: String,
    layers: BTreeMap<SemanticsName, ExprLayer>,
}

impl FinalValueLayers {
    pub(super) fn of(
        site: &GrainsetSite<'_, '_>,
        alias: &str,
        projections: &[ComposedProjection<'_>],
    ) -> Self {
        let pools = site.pools();
        let layers = (projections.iter())
            .map(|projection| {
                let layer = pools.final_layer(site.children(), projection.definition);
                (projection.name.clone(), layer)
            })
            .collect();
        Self {
            alias: alias.to_owned(),
            layers,
        }
    }

    pub(super) fn of_column(&self, column: &NamedReference) -> ExprLayer {
        final_value_layer(&self.alias, column, |name| {
            self.layers.get(&SemanticsName::from(name)).copied()
        })
    }
}

pub(super) fn read_final_values<'e>(
    unit: &EntityId,
    projections: &[ComposedProjection<'_>],
    entries: impl IntoIterator<Item = (&'e SemanticsName, &'e mut ResolvedExprEntry)>,
) -> Result<(), CompileError> {
    let columns: BTreeMap<PhysicalExprId, &Expr<Link>> = (projections.iter())
        .map(|projection| {
            let signature = mint_signature_id(unit, &projection.definition.id);
            let link = mint_expr_id(&signature, ExprRoot::SignatureExpression);
            (link, &projection.column)
        })
        .collect();
    for (name, entry) in entries {
        entry.expr = entry
            .expr
            .clone()
            .transform(&mut |node| match &node {
                Expr::Leaf(Link(id)) => Ok(columns.get(id).map_or(node.clone(), |c| (*c).clone())),
                _ => Ok(node),
            })
            .map_err(|error| CompileError::TypeInferenceFailure {
                node: name.as_str().to_owned(),
                reason: error.to_string(),
            })?;
    }
    Ok(())
}

pub(in crate::compile) fn seal_grainset(
    unit: &SourcePlan<'_>,
    interface: &SemanticInterface,
    vocabulary: Vocabulary<'_>,
    role_definition: &RoleDefinition,
    aliases: &SourceAliases,
    keysets: &KeysetPlan,
    assembly: &mut ManifestAssembly,
) -> Result<Vec<EntityId>, CompileError> {
    let definitions = vocabulary.definitions;
    let index = definitions::interface_index(interface, definitions);
    let serving = Serving::new(&assembly.semantic_sources, &assembly.signatures);
    let bodies = projection::own_bodies(
        interface,
        vocabulary.pool_bodies,
        &index,
        role_definition,
        serving,
    );
    let site = GrainsetSite {
        name: unit.name.as_str(),
        public: unit.public,
        role_definition,
        index: &index,
        definitions,
        assembly,
        keysets,
    };
    let filters = bodies::filter_view(interface, vocabulary.pool_filters);
    let own_bodies = computed(&site, bodies, &filters)?;
    if unit.public {
        projection::check_exposed_served(
            unit.name.as_str(),
            interface,
            definitions,
            role_definition,
            &own_bodies,
            serving,
        )?;
    }
    if own_bodies.into_iter().next().is_none() {
        return Ok(Vec::new());
    }
    let alias = aliases.semantic_alias(&unit.id)?;
    let projections = final_values(&site, interface, &own_bodies, alias.as_str())?;
    let final_layers = FinalValueLayers::of(&site, alias.as_str(), &projections);
    let copies = vec![projection::copy_table(&projections, |projection| {
        Ok(projection.column.clone())
    })?];
    let (
        mut semantics,
        ComposedFilters {
            mut named_filters,
            mut filters,
            dropped,
        },
    ) = projection::lower_signature_entries(ComposedLowering {
        unit,
        interface,
        vocabulary,
        index: &index,
        reference: Reference::Link,
        place: None,
        own_bodies: &own_bodies,
        projections: &projections,
        copies: &copies,
        filtered_projections: BTreeSet::new(),
    })?;
    let entries = (semantics.iter_mut().chain(named_filters.iter_mut()))
        .map(|(name, entry)| (&*name, entry))
        .chain(filters.iter_mut());
    read_final_values(&unit.id, &projections, entries)?;
    let dropped = bodies::with_linkers(&unit.id, &index, &semantics, dropped);
    let trees = projection::bodied_trees(&index, semantics, named_filters, &dropped);
    seal_signatures(
        &unit.id,
        trees,
        filters,
        |expr, owner, root, pool| {
            expressions::emit_reading_final_values(expr, owner, root, pool, &|column| {
                final_layers.of_column(column)
            })
        },
        assembly,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, UNIX_EPOCH};

    use semstrait_common::io::InMemoryFileSystem;
    use semstrait_model::parse;

    use crate::{compile, ManifestProvenance};

    fn diamond_chain(depth: usize) -> String {
        let mut metrics = String::from(
            "    - {name: left0, data_type: double, expr: {divide: [budget, clicks]}}\n    \
             - {name: right0, data_type: double, expr: {divide: [clicks, budget]}}\n",
        );
        for level in 1..=depth {
            let below = level - 1;
            for (name, op) in [("left", "add"), ("right", "divide")] {
                metrics += &format!(
                    "    - {{name: {name}{level}, data_type: double, \
                     expr: {{{op}: [left{below}, right{below}]}}}}\n"
                );
            }
        }
        let refs: Vec<String> = (0..=depth)
            .flat_map(|level| ["left", "right"].map(|name| format!("{{ref: {name}{level}}}")))
            .collect();
        format!(
            r#"
semantic_model:
  name: ads
  dimensions:
    - {{name: campaign, data_type: string, type: categorical}}
  measures:
    - {{name: budget, data_type: double, agg: sum}}
    - {{name: clicks, data_type: double, agg: sum}}
  metrics:
{metrics}  grainsets:
    - name: spend
      dimensions: [{{ref: campaign, is_key: true}}]
      measures: [{{ref: budget}}, {{ref: clicks}}]
      metrics: [{refs}]
      datasets:
        - name: spend_budget
          extras:
            storage: {{format: parquet, paths: ['s3://lake/spend_budget/']}}
            semantic_mapping: {{campaign: campaign, budget: budget}}
        - name: spend_clicks
          extras:
            storage: {{format: parquet, paths: ['s3://lake/spend_clicks/']}}
            semantic_mapping: {{campaign: campaign, clicks: clicks}}
"#,
            refs = refs.join(", ")
        )
    }

    #[test]
    fn a_chain_that_reaches_each_metric_twice_compiles_without_walking_each_path() {
        let depth = 64;
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let (model, _) = parse(&diamond_chain(depth))
                .expect("the chain parses")
                .build()
                .expect("the chain builds");
            let provenance =
                ManifestProvenance::new(UNIX_EPOCH, UNIX_EPOCH + Duration::from_secs(1));
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a runtime for the compile");
            let (manifest, _) = runtime
                .block_on(compile(model, None, &InMemoryFileSystem::new(), provenance))
                .expect("the chain compiles");
            let stored = (manifest.semantic_sources.values())
                .find(|source| source.name.as_str() == "spend")
                .map(|spend| spend.semantic_signature_ids.len());
            sender.send(stored).expect("the test waits for the compile");
        });

        let stored = receiver
            .recv_timeout(Duration::from_secs(60))
            .expect("the chain of 2^64 paths compiles within 60 s");
        assert_eq!(
            stored,
            Some(2 * (depth + 1)),
            "the Grainset computes each Metric of the chain"
        );
    }
}
