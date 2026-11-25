use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_ir::{Delivery, SemanticRole};
use semstrait_manifest::{SemanticBitmask, SourceAlias};

use crate::error::Lacking;
use crate::formula::planned_as;
use crate::grain::{coarseness_of, RequestGrain, SourceGrain};
use crate::read_model::ReadModel;
use crate::{FramePath, PlanError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Route {
    Child(SourceAlias),
    DrillAcross {
        granularity: Option<Granularity>,
        chosen: Vec<(SourceAlias, SemanticBitmask)>,
    },
}

pub(crate) type Tie = (SourceAlias, Vec<SourceAlias>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Routed {
    pub(crate) route: Route,
    pub(crate) ties: Vec<Tie>,
}

struct Picked<'c> {
    child: &'c SourceAlias,
    tied: Vec<&'c SourceAlias>,
}

impl Picked<'_> {
    fn tie(&self) -> Option<Tie> {
        (!self.tied.is_empty()).then(|| {
            let tied = self.tied.iter().map(|child| (*child).clone());
            (self.child.clone(), tied.collect())
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Unrouted {
    NoDefinition,
    NoRole {
        child: SourceAlias,
        semantics: SemanticsName,
    },
}

impl Unrouted {
    pub(crate) fn error(
        self,
        read_model: &dyn ReadModel,
        path: &FramePath,
        source: &DataKindName,
    ) -> PlanError {
        let broken = || PlanError::BrokenSource {
            path: path.clone(),
            semantic_source: source.clone(),
        };
        match self {
            Self::NoDefinition => broken(),
            Self::NoRole { child, semantics } => match read_model.name(&child) {
                Some(child) => PlanError::BrokenTree {
                    path: path.clone(),
                    semantic_source: child,
                    semantics,
                },
                None => broken(),
            },
        }
    }
}

pub(crate) fn route(
    read_model: &dyn ReadModel,
    grainset: &SourceGrain,
    children: &[SourceAlias],
    demanded: &SemanticBitmask,
    conditions: &[(u32, SemanticBitmask)],
    request: &RequestGrain,
    delivery: Delivery,
) -> Result<Option<Routed>, Unrouted> {
    if grainset.unserved(request).is_some() {
        return Ok(None);
    }
    let request = request.fixed(grainset, grainset.granularity());
    let per_name = by_name(
        read_model, grainset, children, demanded, conditions, &request,
    );
    if delivery == Delivery::FinalAggregates && matches!(per_name, Ok(Some(_))) {
        return per_name;
    }
    let whole = with_conditions(demanded, conditions);
    match pick(read_model, grainset, children, &whole, &request) {
        Some(picked) => Ok(Some(Routed {
            route: Route::Child(picked.child.clone()),
            ties: picked.tie().into_iter().collect(),
        })),
        None => per_name,
    }
}

fn by_name(
    read_model: &dyn ReadModel,
    grainset: &SourceGrain,
    children: &[SourceAlias],
    demanded: &SemanticBitmask,
    conditions: &[(u32, SemanticBitmask)],
    request: &RequestGrain,
) -> Result<Option<Routed>, Unrouted> {
    let Some(Demand { split, read }) = partition_demand(read_model, children, demanded)? else {
        return Ok(None);
    };
    let alone = |bit, request: &RequestGrain| {
        let names = read.union(&under(conditions, bit));
        pick(read_model, grainset, children, &names, request)
    };
    let coarsest = (split.iter())
        .map(|bit| alone(*bit, request))
        .collect::<Option<Vec<_>>>()
        .and_then(|picked| {
            (picked.into_iter())
                .map(|picked| picked.child)
                .max_by_key(|child| coarseness_of(read_model.granularity(child)))
        });
    let Some(coarsest) = coarsest else {
        return Ok(None);
    };
    let granularity = fixed_granularity(read_model, grainset, coarsest);
    let request = request.fixed(grainset, granularity);
    let mut chosen: Vec<(SourceAlias, SemanticBitmask)> = Vec::new();
    let mut ties: Vec<Tie> = Vec::new();
    for bit in split {
        let Some(picked) = alone(bit, &request) else {
            return Ok(None);
        };
        let child = picked.child;
        if let Some(tie) = picked.tie().filter(|tie| !ties.contains(tie)) {
            ties.push(tie);
        }
        let name = SemanticBitmask::from_positions([bit]);
        match chosen.iter_mut().find(|(held, _)| held == child) {
            Some((_, names)) => *names = names.union(&name),
            None => chosen.push((child.clone(), read.union(&name))),
        }
    }
    if let [(child, _)] = chosen.as_slice() {
        let route = Route::Child(child.clone());
        return Ok(Some(Routed { route, ties }));
    }
    let joined: Vec<_> = (chosen.iter()).map(|(child, _)| child.clone()).collect();
    let shared = keyset_holders(read_model, &joined).any(|holders| holders.len() == joined.len());
    chosen.sort_by_cached_key(|(child, _)| {
        let fit = SourceGrain::of_source(read_model, child).child_fit(&request, true);
        (fit, read_model.name(child))
    });
    let route = Route::DrillAcross {
        granularity,
        chosen,
    };
    Ok(shared.then_some(Routed { route, ties }))
}

fn under(conditions: &[(u32, SemanticBitmask)], bit: u32) -> SemanticBitmask {
    (conditions.iter())
        .filter(|(measure, _)| *measure == bit)
        .fold(
            SemanticBitmask::from_positions([bit]),
            |names, (_, read)| names.union(read),
        )
}

pub(crate) fn with_conditions(
    demanded: &SemanticBitmask,
    conditions: &[(u32, SemanticBitmask)],
) -> SemanticBitmask {
    (conditions.iter()).fold(demanded.clone(), |names, (_, read)| names.union(read))
}

struct Demand {
    split: Vec<u32>,
    read: SemanticBitmask,
}

fn partition_demand(
    read_model: &dyn ReadModel,
    children: &[SourceAlias],
    demanded: &SemanticBitmask,
) -> Result<Option<Demand>, Unrouted> {
    let mut split = Vec::new();
    let mut read = Vec::new();
    for bit in demanded.set_positions() {
        let definition = read_model.definition(bit).ok_or(Unrouted::NoDefinition)?;
        let Some(child) = (children.iter()).find(|child| read_model.mask(child).contains(bit))
        else {
            return Ok(None);
        };
        let role = planned_as(read_model, child, &definition).ok_or_else(|| Unrouted::NoRole {
            child: child.clone(),
            semantics: definition.name.clone(),
        })?;
        if matches!(role, SemanticRole::Measure | SemanticRole::Metric) {
            split.push(bit);
        } else {
            read.push(bit);
        }
    }
    Ok(Some(Demand {
        split,
        read: SemanticBitmask::from_positions(read),
    }))
}

fn keyset_holders<'c>(
    read_model: &dyn ReadModel,
    children: &'c [SourceAlias],
) -> impl Iterator<Item = Vec<&'c SourceAlias>> {
    let keysets: Vec<_> = (children.iter())
        .flat_map(|child| read_model.keysets(child))
        .collect();
    let masks: Vec<_> = (children.iter())
        .map(|child| (child, read_model.mask(child)))
        .collect();
    keysets.into_iter().map(move |keyset| {
        (masks.iter())
            .filter(|(_, mask)| keyset.is_subset_of(mask))
            .map(|(child, _)| *child)
            .collect()
    })
}

pub(crate) fn maps_whole(
    read_model: &dyn ReadModel,
    children: &[SourceAlias],
    demanded: &SemanticBitmask,
    conditions: &[(u32, SemanticBitmask)],
) -> bool {
    let whole = with_conditions(demanded, conditions);
    if (children.iter()).any(|child| whole.is_subset_of(&read_model.mask(child))) {
        return true;
    }
    let demand = match partition_demand(read_model, children, demanded) {
        Ok(Some(demand)) => demand,
        Ok(None) => return false,
        Err(_) => return true,
    };
    !demand.split.is_empty()
        && keyset_holders(read_model, children).any(|holders| {
            demand.split.iter().all(|bit| {
                let names = demand.read.union(&under(conditions, *bit));
                (holders.iter()).any(|child| names.is_subset_of(&read_model.mask(child)))
            })
        })
}

pub(crate) fn fixed_granularity(
    read_model: &dyn ReadModel,
    grainset: &SourceGrain,
    child: &SourceAlias,
) -> Option<Granularity> {
    grainset
        .granularity()
        .or_else(|| read_model.granularity(child))
}

fn with_filter_reads(
    read_model: &dyn ReadModel,
    child: &SourceAlias,
    names: &SemanticBitmask,
) -> SemanticBitmask {
    let is_filter = |bit: &u32| {
        (read_model.definition(*bit))
            .is_some_and(|definition| definition.role() == SemanticRole::Filter)
    };
    let bodies = (names.set_positions().into_iter())
        .filter(is_filter)
        .filter_map(|bit| read_model.signature(child, bit))
        .filter_map(|signature| read_model.linked(child, &signature.expression_id))
        .flatten()
        .map(|definition| definition.bit_position);
    names.union(&SemanticBitmask::from_positions(bodies))
}

fn pick<'c>(
    read_model: &dyn ReadModel,
    grainset: &SourceGrain,
    children: &'c [SourceAlias],
    names: &SemanticBitmask,
    request: &RequestGrain,
) -> Option<Picked<'c>> {
    let fit_first = request.selects_without_granularity(grainset);
    let serving: Vec<_> = (children.iter())
        .filter_map(|child| {
            let grain = SourceGrain::of_source(read_model, child);
            let read = with_filter_reads(read_model, child, names);
            let serves = (grain.has_axis() || !grainset.reads_axis(&read))
                && grain.candidate(names, request).is_some();
            serves.then(|| (grain.child_fit(request, fit_first), child))
        })
        .collect();
    let &(best, child) = serving.iter().min_by_key(|(fit, _)| *fit)?;
    let tied = (serving.iter())
        .filter(|(fit, other)| *fit == best && *other != child)
        .map(|(_, other)| *other)
        .collect();
    Some(Picked { child, tied })
}

pub(crate) fn not_servable(
    grainset: &SourceGrain,
    request: &RequestGrain,
    path: FramePath,
    source: DataKindName,
    nearest: Vec<Lacking>,
) -> PlanError {
    PlanError::GrainNotServable {
        path,
        source,
        requested: grainset.asked(request),
        source_granularity: grainset.granularity(),
        nearest,
    }
}
