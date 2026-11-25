use std::collections::BTreeMap;

use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_ir::{Delivery, PhysicalExpr};
use semstrait_manifest::{RoleDefinition, SemanticBitmask, SourceAlias};

use crate::error::Lacking;
use crate::grain::{lossy_reaggregation, GranularityFit, RequestGrain, Served, SourceGrain};
use crate::grainset_metrics::{condition_reads, expand, time_axis};
use crate::read_model::{is_dataset, is_grainset, ReadModel};
use crate::routing::{not_servable, route, Route};
use crate::{Advisory, BoundName, FramePath, LookedIn, PlanError, Selection};

pub(crate) struct Resolution {
    pub(crate) scope_root: DataKindName,
    pub(crate) alias: SourceAlias,
    pub(crate) demanded: SemanticBitmask,
    pub(crate) requested_granularity: BTreeMap<SemanticsName, Granularity>,
    pub(crate) alternatives: Vec<(Rank, DataKindName)>,
    pub(crate) compositions: usize,
    pub(crate) granularity_fit: GranularityFit,
}

struct Candidate {
    name: DataKindName,
    alias: SourceAlias,
    rank: Rank,
    aggregates: Aggregates,
}

type Aggregate = (FramePath, SourceAlias, Served, SemanticBitmask);

type Aggregates = Vec<Aggregate>;

pub(crate) type Rank = (usize, GranularityFit, u32);

pub(crate) trait Complexity {
    fn compositions(&self, read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> usize;
    fn exposed_semantics(&self, mask: &SemanticBitmask) -> u32;
}

pub(crate) struct Structural;

impl Complexity for Structural {
    fn compositions(&self, read_model: &(impl ReadModel + ?Sized), source: &SourceAlias) -> usize {
        let closure = read_model.operand_closure(source);
        (std::iter::once(source).chain(&closure))
            .filter(|unit| !is_dataset(read_model, unit))
            .count()
    }

    fn exposed_semantics(&self, mask: &SemanticBitmask) -> u32 {
        mask.count_ones()
    }
}

pub(crate) fn resolve(
    read_model: &impl ReadModel,
    complexity: &impl Complexity,
    from: Option<&DataKindName>,
    select: &[Selection],
    bindings: &[BoundName],
) -> Result<(Resolution, Vec<Advisory>), PlanError> {
    let public = read_model.public_sources();
    let demanded = SemanticBitmask::from_positions(bindings.iter().map(|binding| binding.bit));
    let request = RequestGrain::of(read_model, select, bindings);
    let (root, ambiguity) = match from {
        Some(from) => (
            named_root(
                read_model, complexity, &public, from, &demanded, bindings, &request,
            )?,
            None,
        ),
        None => field_first(
            read_model, complexity, &public, &demanded, bindings, &request,
        )?,
    };
    let mut advisories: Vec<_> = ambiguity.into_iter().collect();
    for (path, unit, served, names) in &root.aggregates {
        let bound: Vec<_> = (names.set_positions().into_iter())
            .filter_map(|bit| read_model.definition(bit))
            .map(BoundName::of)
            .collect();
        for lossy in lossy_reaggregation(read_model, path, unit, *served, &bound)? {
            if !advisories.contains(&lossy) {
                advisories.push(lossy);
            }
        }
    }
    let joinset = |unit: &SourceAlias| {
        matches!(
            read_model.source_kind(unit),
            Some(RoleDefinition::Joinset { .. })
        )
    };
    let scope = if joinset(&root.alias) {
        read_model.originated_by(&root.alias)
    } else {
        BTreeMap::new()
    };
    let sub_compositions = (scope.iter()).filter(|(_, (unit, _))| joinset(unit));
    let alternatives = ranked(
        read_model,
        complexity,
        sub_compositions,
        &demanded,
        bindings,
        &request,
    )
    .candidates
    .into_iter()
    .take_while(|(rank, ..)| *rank < root.rank)
    .map(|(rank, unit, ..)| (rank, unit.clone()))
    .collect();
    let resolution = Resolution {
        compositions: root.rank.0,
        granularity_fit: root.rank.1,
        scope_root: root.name,
        alias: root.alias,
        demanded,
        requested_granularity: (select.iter())
            .filter_map(|selection| Some((selection.name().clone(), selection.granularity()?)))
            .collect(),
        alternatives,
    };
    Ok((resolution, advisories))
}

fn named_root(
    read_model: &impl ReadModel,
    complexity: &impl Complexity,
    public: &BTreeMap<DataKindName, (SourceAlias, SemanticBitmask)>,
    from: &DataKindName,
    demanded: &SemanticBitmask,
    bindings: &[BoundName],
    request: &RequestGrain,
) -> Result<Candidate, PlanError> {
    let (alias, mask) = public.get(from).ok_or_else(|| PlanError::UnknownName {
        name: from.to_string(),
        looked_in: LookedIn::PublicSources,
    })?;
    if let Some(name) = missing(bindings, mask).next() {
        return Err(PlanError::UnknownName {
            name: name.to_string(),
            looked_in: LookedIn::Source(from.clone()),
        });
    }
    let (aggregates, fit) = candidate_test(
        read_model,
        complexity,
        (from, alias, mask),
        demanded,
        bindings,
        request,
    )?;
    Ok(Candidate {
        name: from.clone(),
        alias: alias.clone(),
        rank: rank(read_model, complexity, alias, mask, fit),
        aggregates,
    })
}

fn field_first(
    read_model: &impl ReadModel,
    complexity: &impl Complexity,
    public: &BTreeMap<DataKindName, (SourceAlias, SemanticBitmask)>,
    demanded: &SemanticBitmask,
    bindings: &[BoundName],
    request: &RequestGrain,
) -> Result<(Candidate, Option<Advisory>), PlanError> {
    let Ranked { candidates, broken } =
        ranked(read_model, complexity, public, demanded, bindings, request);
    let mut candidates = candidates.into_iter();
    let Some((least, chosen, alias, aggregates)) = candidates.next() else {
        if let Some(broken) = broken {
            return Err(broken);
        }
        let units = public
            .iter()
            .map(|(name, (alias, mask))| (name, alias, mask));
        return Err(PlanError::NoCandidate {
            nearest: nearest(read_model, complexity, bindings, units),
        });
    };
    let tied: Vec<DataKindName> = candidates
        .take_while(|(rank, ..)| *rank == least)
        .map(|(_, name, ..)| name.clone())
        .collect();
    let chosen = Candidate {
        name: chosen.clone(),
        alias: alias.clone(),
        rank: least,
        aggregates,
    };
    let ambiguity = name_order(FramePath(vec![chosen.name.clone()]), &chosen.name, tied);
    Ok((chosen, ambiguity))
}

pub(crate) fn tied_stand_in(
    alternatives: &[(Rank, DataKindName)],
    stand_in: &DataKindName,
    path: &FramePath,
    may_stand_in: impl Fn(&DataKindName) -> bool,
) -> Option<Advisory> {
    let (rank, _) = (alternatives.iter()).find(|(_, unit)| unit == stand_in)?;
    let tied = (alternatives.iter())
        .filter(|(other, unit)| other == rank && unit != stand_in && may_stand_in(unit))
        .map(|(_, unit)| unit.clone())
        .collect();
    name_order(path.clone(), stand_in, tied)
}

pub(crate) fn name_order(
    path: FramePath,
    chosen: &DataKindName,
    tied: Vec<DataKindName>,
) -> Option<Advisory> {
    (!tied.is_empty()).then(|| Advisory::AmbiguousCandidates {
        path,
        chosen: chosen.clone(),
        tied,
    })
}

struct Ranked<'u> {
    candidates: Vec<(Rank, &'u DataKindName, &'u SourceAlias, Aggregates)>,
    broken: Option<PlanError>,
}

pub(crate) fn nearest<'u>(
    read_model: &(impl ReadModel + ?Sized),
    complexity: &impl Complexity,
    bindings: &[BoundName],
    units: impl IntoIterator<Item = (&'u DataKindName, &'u SourceAlias, &'u SemanticBitmask)>,
) -> Vec<Lacking> {
    let mut nearest: Vec<_> = (units.into_iter())
        .map(|(name, alias, mask)| {
            let lacks: Vec<SemanticsName> = missing(bindings, mask).cloned().collect();
            let structural = (
                complexity.compositions(read_model, alias),
                complexity.exposed_semantics(mask),
            );
            (lacks.len(), structural, name.clone(), lacks)
        })
        .collect();
    nearest.sort();
    (nearest.into_iter())
        .map(|(_, _, name, lacks)| (name, lacks))
        .collect()
}

pub(crate) fn nearest_children(
    read_model: &(impl ReadModel + ?Sized),
    complexity: &impl Complexity,
    bindings: &[BoundName],
    grainset: &SourceAlias,
) -> Vec<Lacking> {
    let children: Vec<_> = (read_model.operands(grainset).into_iter())
        .filter_map(|child| Some((read_model.name(&child)?, read_model.mask(&child), child)))
        .collect();
    let units = (children.iter()).map(|(name, mask, alias)| (name, alias, mask));
    nearest(read_model, complexity, bindings, units)
}

fn ranked<'u>(
    read_model: &impl ReadModel,
    complexity: &impl Complexity,
    units: impl IntoIterator<Item = (&'u DataKindName, &'u (SourceAlias, SemanticBitmask))>,
    demanded: &SemanticBitmask,
    bindings: &[BoundName],
    request: &RequestGrain,
) -> Ranked<'u> {
    let mut candidates = Vec::new();
    let mut broken = None;
    for (name, (alias, mask)) in units {
        match candidate_test(
            read_model,
            complexity,
            (name, alias, mask),
            demanded,
            bindings,
            request,
        ) {
            Ok((aggregates, fit)) => {
                let rank = rank(read_model, complexity, alias, mask, fit);
                candidates.push((rank, name, alias, aggregates));
            }
            Err(PlanError::GrainNotServable { .. }) => {}
            Err(error) => {
                broken.get_or_insert(error);
            }
        }
    }
    candidates.sort_by_key(|&(rank, name, ..)| (rank, name));
    Ranked { candidates, broken }
}

fn candidate_test(
    read_model: &impl ReadModel,
    complexity: &impl Complexity,
    (source, alias, mask): (&DataKindName, &SourceAlias, &SemanticBitmask),
    demanded: &SemanticBitmask,
    bindings: &[BoundName],
    request: &RequestGrain,
) -> Result<(Aggregates, GranularityFit), PlanError> {
    let grain = SourceGrain::of(read_model, alias, mask);
    let path = FramePath(vec![source.clone()]);
    if !is_grainset(read_model, alias) {
        let Some(served) = grain.candidate(demanded, request) else {
            return Err(PlanError::GrainNotServable {
                path,
                source: source.clone(),
                requested: grain.unserved(request),
                source_granularity: grain.granularity(),
                nearest: Vec::new(),
            });
        };
        let aggregate = (path, alias.clone(), served, demanded.clone());
        return Ok((vec![aggregate], grain.granularity_fit(request)));
    }
    let test = GrainsetTest {
        read_model,
        alias,
        source,
        path: &path,
        grain: &grain,
        children: read_model.operands(alias),
    };
    let demand: Vec<_> = (demanded.set_positions().into_iter())
        .map(|bit| (bit, None))
        .collect();
    let at = (&path, source);
    let fixed = time_axis(read_model, alias, &grain, request, &demand, at)?;
    let fixed_request = request.fixed(&grain, fixed);
    test.grouped(&demand, &fixed_request, request)?
        .ok_or_else(|| {
            let nearest = nearest_children(read_model, complexity, bindings, alias);
            not_servable(&grain, request, path.clone(), source.clone(), nearest)
        })
}

struct GrainsetTest<'t, R: ReadModel> {
    read_model: &'t R,
    alias: &'t SourceAlias,
    source: &'t DataKindName,
    path: &'t FramePath,
    grain: &'t SourceGrain,
    children: Vec<SourceAlias>,
}

type Answer = Option<(Aggregates, GranularityFit)>;

impl<R: ReadModel> GrainsetTest<'_, R> {
    fn grouped(
        &self,
        demand: &[(u32, Option<PhysicalExpr>)],
        request: &RequestGrain,
        asked: &RequestGrain,
    ) -> Result<Answer, PlanError> {
        let read_model = self.read_model;
        let groups = expand(read_model, self.alias, demand, &request.granular())
            .map_err(|unexpanded| unexpanded.error(read_model, self.path, self.source))?;
        let Some(groups) = groups else {
            let demanded = SemanticBitmask::from_positions(demand.iter().map(|(bit, _)| *bit));
            let conditions = condition_reads(read_model, demand);
            return self.answer(&demanded, &conditions, request, asked);
        };
        let mut answered: Answer = None;
        for group in &groups.groups {
            let read = request.without_granularity(group.fine());
            let read = read.with_dimensions(read_model, group.extra());
            let Some((aggregates, fit)) = self.grouped(&group.demanded(), &read, asked)? else {
                return Ok(None);
            };
            let unserved = (aggregates.iter()).any(|(_, child, _, _)| {
                SourceGrain::of_source(read_model, child)
                    .unserved(request)
                    .is_some()
            });
            if unserved {
                return Ok(None);
            }
            match &mut answered {
                Some((held, _)) => held.extend(aggregates),
                None => answered = Some((aggregates, fit)),
            }
        }
        Ok(answered)
    }

    fn answer(
        &self,
        demanded: &SemanticBitmask,
        conditions: &[(u32, SemanticBitmask)],
        request: &RequestGrain,
        asked: &RequestGrain,
    ) -> Result<Answer, PlanError> {
        let (read_model, grain) = (self.read_model, self.grain);
        let routed = route(
            read_model,
            grain,
            &self.children,
            demanded,
            conditions,
            request,
            Delivery::FinalAggregates,
        )
        .map_err(|unrouted| unrouted.error(read_model, self.path, self.source))?;
        match routed.map(|routed| routed.route) {
            Some(Route::Child(answering)) => Ok(self
                .child(&answering, demanded, request, asked)?
                .map(|(aggregate, fit)| (vec![aggregate], fit))),
            Some(Route::DrillAcross {
                granularity,
                chosen,
            }) => {
                let Some((first, names)) = chosen.first() else {
                    return Ok(None);
                };
                let Some((_, fit)) = self.child(first, names, request, asked)? else {
                    return Ok(None);
                };
                let request = request.fixed(grain, granularity);
                let mut aggregates = Vec::with_capacity(chosen.len());
                for (unit, names) in &chosen {
                    let Some((aggregate, _)) = self.child(unit, names, &request, asked)? else {
                        return Ok(None);
                    };
                    aggregates.push(aggregate);
                }
                Ok(Some((aggregates, fit)))
            }
            None => Ok(None),
        }
    }

    fn child(
        &self,
        child: &SourceAlias,
        names: &SemanticBitmask,
        request: &RequestGrain,
        asked: &RequestGrain,
    ) -> Result<Option<(Aggregate, GranularityFit)>, PlanError> {
        let unit = SourceGrain::of_source(self.read_model, child);
        let Some(served) = unit.candidate(names, request) else {
            return Ok(None);
        };
        let name = self.read_model.name(child).ok_or_else(|| self.broken())?;
        let mut path = self.path.clone();
        path.0.push(name);
        let aggregate = (path, child.clone(), served, names.clone());
        Ok(Some((aggregate, unit.granularity_fit(asked))))
    }

    fn broken(&self) -> PlanError {
        PlanError::BrokenSource {
            path: self.path.clone(),
            semantic_source: self.source.clone(),
        }
    }
}

fn rank(
    read_model: &impl ReadModel,
    complexity: &impl Complexity,
    source: &SourceAlias,
    mask: &SemanticBitmask,
    fit: GranularityFit,
) -> Rank {
    let compositions = complexity.compositions(read_model, source);
    (compositions, fit, complexity.exposed_semantics(mask))
}

fn missing<'b>(
    bindings: &'b [BoundName],
    mask: &'b SemanticBitmask,
) -> impl Iterator<Item = &'b SemanticsName> {
    (bindings.iter())
        .filter(|binding| !mask.contains(binding.bit))
        .map(|binding| &binding.name)
}
