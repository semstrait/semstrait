use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::Granularity;
use semstrait_common::{DataKindName, SemanticsName};

use crate::error::CompileError;
use crate::manifest::bitmap::SemanticBitmap;
use crate::manifest::semantic_source::{RoleDefinition, SemanticSource};
use semstrait_ir::{ScdType, TemporalShape, TemporalType};

use super::assemble::{constituents, SourcePlan};
use super::types::SemanticsIndex;

pub type ShapeParticipant<'a> = (&'a EntityId, Option<&'a TemporalShape>);

fn scd_type(authored: semstrait_model::ScdType) -> ScdType {
    match authored {
        semstrait_model::ScdType::Type1 => ScdType::Type1,
        semstrait_model::ScdType::Type2 => ScdType::Type2,
        other => unreachable!("semstrait-model added a ScdType variant: {other:?}"),
    }
}

pub(crate) fn derive_composed_shapes(
    sources: &mut BTreeMap<EntityId, SemanticSource>,
) -> Result<(), CompileError> {
    for id in bottom_up_order(sources) {
        let Some(entry) = sources.get(&id) else {
            continue;
        };
        let derived = composed_shape(entry, sources)?;
        if let Some(entry) = sources.get_mut(&id) {
            entry.temporal = derived;
        }
    }
    Ok(())
}

pub(crate) fn derive_sealed_shape(sources: &mut BTreeMap<EntityId, SemanticSource>, id: &EntityId) {
    let Some(Ok(derived)) = (sources.get(id)).map(|entry| composed_shape(entry, sources)) else {
        return;
    };
    if let Some(entry) = sources.get_mut(id) {
        entry.temporal = derived;
    }
}

pub fn composed_shape(
    entry: &SemanticSource,
    sources: &BTreeMap<EntityId, SemanticSource>,
) -> Result<Option<TemporalShape>, CompileError> {
    let stated = entry.temporal.as_ref().map(|shape| &shape.temporal_type);
    let kind = match &entry.role_definition {
        RoleDefinition::Unionset { branches, .. } => {
            unanimous_type(&entry.id, &published_by(branches, sources), stated)?
        }
        RoleDefinition::Joinset { .. } => {
            stated_or_highest_rank_type(&entry.id, &reach(entry, sources, |_| true), stated)?
        }
        RoleDefinition::Grainset { grains, .. } => {
            let constituents = published_by(grains, sources);
            unanimous_type(&entry.id, &constituents, stated)?
        }
        _ => return Ok(entry.temporal.clone()),
    };
    let carried: Vec<(&EntityId, Granularity)> =
        reach(entry, sources, |operand| granularity_of(operand).is_none())
            .into_iter()
            .filter_map(|(id, shape)| Some((id, shape?.granularity?)))
            .collect();
    let granularity = match granularity_of(entry) {
        None => unanimous_granularity(carried.iter().map(|(_, granularity)| *granularity)),
        Some(stated) => {
            let unserved: Vec<(EntityId, Granularity)> = (carried.into_iter())
                .filter(|(_, granularity)| !granularity.serves(stated))
                .map(|(id, granularity)| (id.clone(), granularity))
                .collect();
            if !unserved.is_empty() {
                return Err(CompileError::StatedGranularityNotServed {
                    composite: entry.id.clone(),
                    stated,
                    constituents: unserved,
                });
            }
            Some(stated)
        }
    };
    let unrouted = kind.as_ref().and(granularity).is_none();
    if unrouted && carries_axis(entry, kind.is_some(), sources) {
        if let Some(composite) = owed_grain(entry, kind.is_some(), sources) {
            let composite = composite.clone();
            let constituent = entry.id.clone();
            return Err(if kind.is_none() {
                CompileError::ComposedConstituentMissingTemporalShape {
                    composite,
                    constituent,
                }
            } else {
                CompileError::ComposedConstituentMissingGranularity {
                    composite,
                    constituent,
                }
            });
        }
    }
    Ok(kind.map(|kind| TemporalShape::new(kind, granularity)))
}

fn carries_axis(
    entry: &SemanticSource,
    shaped: bool,
    sources: &BTreeMap<EntityId, SemanticSource>,
) -> bool {
    shaped
        || entry.temporal.is_some()
        || reach(entry, sources, |_| true)
            .iter()
            .any(|(_, published)| published.is_some())
}

fn granularity_of(entry: &SemanticSource) -> Option<Granularity> {
    entry.temporal.as_ref()?.granularity
}

fn owed_grain<'s>(
    entry: &SemanticSource,
    shaped: bool,
    sources: &'s BTreeMap<EntityId, SemanticSource>,
) -> Option<&'s EntityId> {
    if matches!(
        entry.role_definition,
        RoleDefinition::Unionset {
            null_fill: false,
            ..
        }
    ) {
        return None;
    }
    let owed = |originator: &&SemanticSource| {
        (shaped || matches!(originator.role_definition, RoleDefinition::Grainset { .. }))
            && constituents(originator, sources).contains(&entry.id)
    };
    (entry.originators.iter())
        .filter_map(|id| sources.get(id))
        .find(owed)
        .map(|originator| &originator.id)
}

fn published_by<'s>(
    operands: &'s [EntityId],
    sources: &'s BTreeMap<EntityId, SemanticSource>,
) -> Vec<ShapeParticipant<'s>> {
    operands
        .iter()
        .filter_map(|id| {
            sources
                .get(id)
                .map(|operand| (id, operand.temporal.as_ref()))
        })
        .collect()
}

fn reach<'s>(
    entry: &'s SemanticSource,
    sources: &'s BTreeMap<EntityId, SemanticSource>,
    descend: impl Fn(&SemanticSource) -> bool,
) -> Vec<ShapeParticipant<'s>> {
    let mut reached: Vec<ShapeParticipant<'s>> = Vec::new();
    let mut seen: BTreeSet<&'s EntityId> = BTreeSet::new();
    let mut frontier: Vec<&'s EntityId> = entry.role_definition.operands();
    while let Some(id) = frontier.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(operand) = sources.get(id) else {
            continue;
        };
        reached.push((id, operand.temporal.as_ref()));
        if descend(operand) {
            frontier.extend(operand.role_definition.operands());
        }
    }
    reached.sort_by(|left, right| left.0.cmp(right.0));
    reached
}

pub fn unanimous_type(
    unionset: &EntityId,
    constituents: &[ShapeParticipant<'_>],
    stated: Option<&TemporalType>,
) -> Result<Option<TemporalType>, CompileError> {
    let mut agreed: Option<&TemporalType> = None;
    for claim in constituents
        .iter()
        .filter_map(|(_, published)| published.map(|shape| &shape.temporal_type))
        .chain(stated)
    {
        match agreed {
            None => agreed = Some(claim),
            Some(held) if held == claim => {}
            Some(_) => return Err(disagreement(unionset, constituents, stated)),
        }
    }
    Ok(agreed.cloned())
}

#[must_use]
pub fn highest_rank_type(participants: &[ShapeParticipant<'_>]) -> Option<TemporalType> {
    let mut top: Option<&TemporalType> = None;
    let mut tied = false;
    for candidate in participants
        .iter()
        .filter_map(|(_, published)| published.map(|shape| &shape.temporal_type))
    {
        match top {
            Some(held) if held.rank() > candidate.rank() => {}
            Some(held) if held.rank() == candidate.rank() => tied |= held != candidate,
            _ => {
                top = Some(candidate);
                tied = false;
            }
        }
    }
    if tied {
        return None;
    }
    top.cloned()
}

pub(crate) fn stated_or_highest_rank_type(
    joinset: &EntityId,
    participants: &[ShapeParticipant<'_>],
    stated: Option<&TemporalType>,
) -> Result<Option<TemporalType>, CompileError> {
    let Some(stated) = stated else {
        return Ok(highest_rank_type(participants));
    };
    let published = || {
        participants
            .iter()
            .filter_map(|(_, published)| published.map(|shape| &shape.temporal_type))
    };
    let Some(top) = published().map(TemporalType::rank).max() else {
        return Ok(Some(stated.clone()));
    };
    if published().any(|shape| shape.rank() == top && shape == stated) {
        return Ok(Some(stated.clone()));
    }
    Err(disagreement(joinset, participants, Some(stated)))
}

pub(crate) fn unanimous_granularity(
    granularities: impl IntoIterator<Item = Granularity>,
) -> Option<Granularity> {
    let mut agreed: Option<Granularity> = None;
    for granularity in granularities {
        match agreed {
            None => agreed = Some(granularity),
            Some(held) if held == granularity => {}
            Some(_) => return None,
        }
    }
    agreed
}

pub fn distinct_grains(
    sources: &BTreeMap<EntityId, SemanticSource>,
    definitions: &SemanticBitmap,
) -> Vec<CompileError> {
    sources
        .values()
        .filter_map(|grainset| match &grainset.role_definition {
            RoleDefinition::Grainset { grains, .. } => Some((grainset, grains)),
            _ => None,
        })
        .flat_map(|(grainset, grains)| {
            let children: Vec<&SemanticSource> =
                grains.iter().filter_map(|id| sources.get(id)).collect();
            repeated_groups(&children)
                .into_iter()
                .map(|group| CompileError::GrainsetGrainRepeated {
                    grainset: grainset.name.clone(),
                    children: group
                        .iter()
                        .map(|repeat| authored_name(&repeat.name, &grainset.name))
                        .collect(),
                    time_axis: axis_name(group[0], definitions).zip(granularity_of(group[0])),
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn repeated_groups<'a>(children: &[&'a SemanticSource]) -> Vec<Vec<&'a SemanticSource>> {
    let mut groups: Vec<Vec<&SemanticSource>> = Vec::new();
    for child in children.iter().copied().filter(|child| routable(child)) {
        let same_grain = |other: &&SemanticSource| {
            granularity_of(other) == granularity_of(child)
                && other.semantic_bitmask == child.semantic_bitmask
        };
        match groups.iter_mut().find(|group| same_grain(&group[0])) {
            Some(group) => group.push(child),
            None => groups.push(vec![child]),
        }
    }
    groups.retain(|group| group.len() > 1);
    groups
}

fn authored_name(child: &DataKindName, grainset: &DataKindName) -> DataKindName {
    child
        .as_str()
        .strip_prefix(grainset.as_str())
        .and_then(|rest| rest.strip_prefix('.'))
        .map_or_else(|| child.clone(), DataKindName::from)
}

fn routable(entry: &SemanticSource) -> bool {
    entry.temporal.is_none() || granularity_of(entry).is_some()
}

fn axis_name(entry: &SemanticSource, definitions: &SemanticBitmap) -> Option<SemanticsName> {
    let axis = match &entry.temporal.as_ref()?.temporal_type {
        TemporalType::Scd { valid_from, .. } => valid_from,
        other => other.time_axis()?,
    };
    Some(
        definitions
            .entries
            .get(axis)
            .map_or_else(|| SemanticsName::from(axis.as_str()), |d| d.name.clone()),
    )
}

fn disagreement(
    composite: &EntityId,
    constituents: &[ShapeParticipant<'_>],
    stated: Option<&TemporalType>,
) -> CompileError {
    CompileError::ComposedTemporalDisagreement {
        composite: composite.clone(),
        shapes: constituents
            .iter()
            .filter_map(|(id, published)| {
                published.map(|shape| ((*id).clone(), shape.temporal_type.clone()))
            })
            .chain(stated.map(|claim| (composite.clone(), claim.clone())))
            .collect(),
    }
}

#[must_use]
pub fn bottom_up_order(sources: &BTreeMap<EntityId, SemanticSource>) -> Vec<EntityId> {
    let mut order: Vec<EntityId> = Vec::with_capacity(sources.len());
    let mut emitted: BTreeSet<&EntityId> = BTreeSet::new();
    for id in sources.keys() {
        let mut path: Vec<&EntityId> = Vec::new();
        visit(id, sources, &mut emitted, &mut path, &mut order);
    }
    order
}

fn visit<'s>(
    id: &'s EntityId,
    sources: &'s BTreeMap<EntityId, SemanticSource>,
    emitted: &mut BTreeSet<&'s EntityId>,
    path: &mut Vec<&'s EntityId>,
    order: &mut Vec<EntityId>,
) {
    if emitted.contains(id) || path.contains(&id) {
        return;
    }
    let Some(entry) = sources.get(id) else {
        return;
    };
    path.push(id);
    for operand in entry.role_definition.operands() {
        visit(operand, sources, emitted, path, order);
    }
    path.pop();
    emitted.insert(id);
    order.push(id.clone());
}

pub(super) fn resolve_authored_temporal(
    authored: &semstrait_model::TemporalShape,
    semantics: &SemanticsIndex<'_>,
    unit: &SourcePlan<'_>,
) -> Result<TemporalShape, CompileError> {
    use semstrait_model::TemporalShapeKind;

    let identify = |name: &SemanticsName| -> Result<EntityId, CompileError> {
        semantics
            .get(name)
            .map(|definition| definition.id.clone())
            .ok_or_else(|| CompileError::UnknownReference {
                name: name.as_str().to_owned(),
                scope: crate::error::ReferenceScope::DataKind(unit.id.clone()),
            })
    };
    let temporal_type = match &authored.kind {
        TemporalShapeKind::Timeseries(body) => {
            TemporalType::timeseries(identify(&body.occurred_at)?)
        }
        TemporalShapeKind::Events(body) => TemporalType::events(identify(&body.event_time)?),
        TemporalShapeKind::Snapshot(body) => {
            TemporalType::snapshot(identify(&body.snapshotted_at)?)
        }
        TemporalShapeKind::Scd(body) => TemporalType::scd(
            scd_type(body.scd_type),
            identify(&body.valid_from)?,
            identify(&body.valid_to)?,
        ),
        other => {
            return Err(CompileError::TypeInferenceFailure {
                node: format!("dataset {} temporal", unit.name.as_str()),
                reason: format!(
                    "authored temporal shape {other:?} has no persisted counterpart, so carrying \
                     it would record a grain against a shape the artifact cannot state"
                ),
            })
        }
    };
    Ok(TemporalShape::new(temporal_type, authored.granularity))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::manifest::bitmap::SemanticBitmask;
    use crate::manifest::expression::{ExprRoot, ManifestExpressions};
    use crate::manifest::semantic_source::Selector;
    use crate::manifest::source_alias::SourceAlias;
    use semstrait_common::DataKindName;
    use semstrait_ir::MultisetOp;
    use semstrait_sql::expr::{Expr, NamedReference};

    use semstrait_ir::{CrossFilter, Integrity};

    fn shape(kind: TemporalType, granularity: Option<Granularity>) -> TemporalShape {
        TemporalShape::new(kind, granularity)
    }

    fn one_of_each() -> [TemporalShape; 4] {
        [
            shape(
                TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
                Some(Granularity::Day),
            ),
            shape(
                TemporalType::timeseries("sem.occurred"),
                Some(Granularity::Day),
            ),
            shape(
                TemporalType::snapshot("sem.captured"),
                Some(Granularity::Day),
            ),
            shape(TemporalType::events("sem.event"), Some(Granularity::Day)),
        ]
    }

    fn ids(count: usize) -> Vec<EntityId> {
        (0..count).map(|i| format!("e{i}")).collect()
    }

    fn participants<'a>(
        ids: &'a [EntityId],
        shapes: &'a [Option<&'a TemporalShape>],
    ) -> Vec<ShapeParticipant<'a>> {
        ids.iter().zip(shapes.iter().copied()).collect()
    }

    fn child(name: &str, temporal: Option<TemporalShape>, mask: &[u32]) -> SemanticSource {
        let mut child = entry(
            name,
            temporal,
            RoleDefinition::dataset(format!("src.{name}"), Selector::Table),
        );
        child.semantic_bitmask = SemanticBitmask::from_positions(mask.iter().copied());
        child
    }

    fn grainset_of(children: Vec<SemanticSource>) -> BTreeMap<EntityId, SemanticSource> {
        let ids = children.iter().map(|child| child.id.clone()).collect();
        let parent = entry("orders", None, RoleDefinition::grainset(ids, Vec::new()));
        collection(children.into_iter().chain([parent]).collect())
    }

    fn repeated(errors: &[CompileError]) -> Vec<(&DataKindName, Vec<&str>)> {
        errors
            .iter()
            .map(|error| match error {
                CompileError::GrainsetGrainRepeated {
                    grainset, children, ..
                } => (
                    grainset,
                    children.iter().map(DataKindName::as_str).collect(),
                ),
                other => panic!("expected the repeated-grain variant, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn distinctness_names_the_valid_from_of_an_scd_pair() {
        let definitions = SemanticBitmap::freeze([]).expect("empty registry");
        let scd = shape(
            TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
            Some(Granularity::Day),
        );
        let errors = distinct_grains(
            &grainset_of(vec![
                child("a", Some(scd.clone()), &[0]),
                child("b", Some(scd), &[0]),
            ]),
            &definitions,
        );
        assert!(
            matches!(
                errors.as_slice(),
                [CompileError::GrainsetGrainRepeated { time_axis: Some((axis, Granularity::Day)), .. }]
                    if *axis == SemanticsName::from("sem.from")
            ),
            "an scd pair names its valid_from: {errors:?}"
        );
        let events = shape(TemporalType::events("sem.event"), Some(Granularity::Day));
        let errors = distinct_grains(
            &grainset_of(vec![
                child("a", Some(events.clone()), &[0]),
                child("b", Some(events), &[0]),
            ]),
            &definitions,
        );
        assert!(
            matches!(
                errors.as_slice(),
                [CompileError::GrainsetGrainRepeated {
                    time_axis: Some(_),
                    ..
                }]
            ),
            "an events pair names its event time: {errors:?}"
        );
    }

    #[test]
    fn distinctness_abstains_on_a_shape_with_no_granularity_and_reports_every_repeat() {
        let definitions = SemanticBitmap::freeze([]).expect("empty registry");
        let day = shape(TemporalType::events("sem.event"), Some(Granularity::Day));
        let axis_only = shape(TemporalType::events("sem.event"), None);

        assert!(
            distinct_grains(
                &grainset_of(vec![
                    child("a", None, &[0]),
                    child("b", None, &[0, 1]),
                    child("c", Some(day.clone()), &[0]),
                ]),
                &definitions,
            )
            .is_empty(),
            "two constituents with no time axis over two masks are two children"
        );
        let errors = distinct_grains(
            &grainset_of(vec![
                child("a", None, &[0]),
                child("b", None, &[0]),
                child("c", Some(day.clone()), &[0]),
            ]),
            &definitions,
        );
        assert!(
            matches!(
                errors.as_slice(),
                [CompileError::GrainsetGrainRepeated {
                    time_axis: None,
                    ..
                }]
            ),
            "two constituents with no time axis over one mask repeat: {errors:?}"
        );
        assert!(
            distinct_grains(
                &grainset_of(vec![
                    child("a", Some(axis_only.clone()), &[0]),
                    child("b", Some(axis_only), &[0]),
                    child("c", Some(day.clone()), &[0]),
                ]),
                &definitions,
            )
            .is_empty(),
            "a shape with no granularity carries no grain to repeat"
        );
        assert!(
            distinct_grains(
                &grainset_of(vec![
                    child("a", Some(day.clone()), &[0]),
                    child("b", Some(day.clone()), &[0, 1]),
                ]),
                &definitions,
            )
            .is_empty(),
            "one grain over two masks is two children"
        );

        let errors = distinct_grains(
            &grainset_of(vec![
                child("a", Some(day.clone()), &[0]),
                child("b", Some(day.clone()), &[0]),
                child("c", Some(day), &[0]),
            ]),
            &definitions,
        );
        let reported = repeated(&errors);
        let [(_, children)] = reported.as_slice() else {
            panic!("one grain and one mask carried three times is one repeat: {errors:?}");
        };
        assert_eq!(
            children.len(),
            3,
            "every child that repeats is named, so a three-way collision is legible rather \
             than reported as one pair"
        );
    }

    #[test]
    fn distinctness_reports_each_repeating_group_and_names_children_as_authored() {
        let definitions = SemanticBitmap::freeze([]).expect("empty registry");
        let day = shape(TemporalType::events("sem.event"), Some(Granularity::Day));
        let month = shape(TemporalType::events("sem.event"), Some(Granularity::Month));

        let errors = distinct_grains(
            &grainset_of(vec![
                child("orders.a", Some(day.clone()), &[0]),
                child("orders.b", Some(day), &[0]),
                child("orders.c", Some(month.clone()), &[0]),
                child("orders.d", Some(month), &[0]),
            ]),
            &definitions,
        );

        let reported: Vec<Vec<&str>> = repeated(&errors)
            .into_iter()
            .map(|(_, children)| children)
            .collect();
        assert_eq!(reported, vec![vec!["a", "b"], vec!["c", "d"]]);
    }

    fn entry(name: &str, temporal: Option<TemporalShape>, role: RoleDefinition) -> SemanticSource {
        SemanticSource::new(
            name.to_owned(),
            DataKindName::from(name),
            SourceAlias::from(name),
            true,
            SemanticBitmask::empty(),
            temporal,
            role,
            Vec::new(),
        )
    }

    fn collection(entries: Vec<SemanticSource>) -> BTreeMap<EntityId, SemanticSource> {
        entries
            .into_iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect()
    }

    #[test]
    fn the_rank_is_the_total_order_scd_timeseries_snapshot_events() {
        let [scd, timeseries, snapshot, events] = one_of_each();
        let ranks: Vec<u8> = [&scd, &timeseries, &snapshot, &events]
            .iter()
            .map(|shape| shape.temporal_type.rank())
            .collect();
        assert_eq!(ranks, vec![0, 1, 2, 3], "the roster is ordinal and gapless");
        for pair in ranks.windows(2) {
            assert!(pair[0] < pair[1], "the order is strict, got {ranks:?}");
        }
    }

    #[test]
    fn the_maximum_is_invariant_to_the_order_of_its_operands() {
        let shapes = one_of_each();
        let ids = ids(4);
        let expected = shapes[3].temporal_type.clone();
        for order in permutations(&[0, 1, 2, 3]) {
            let supplied: Vec<Option<&TemporalShape>> =
                order.iter().map(|i| Some(&shapes[*i])).collect();
            assert_eq!(
                highest_rank_type(&participants(&ids, &supplied)),
                Some(expected.clone()),
                "the maximum moved when the operands were reordered as {order:?}"
            );
        }
    }

    #[test]
    fn the_derivation_is_invariant_to_bracketing_and_to_direction() {
        let claims = [
            shape(TemporalType::events("sem.order"), Some(Granularity::Day)),
            shape(TemporalType::events("sem.ship"), Some(Granularity::Day)),
            shape(
                TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
                Some(Granularity::Month),
            ),
        ];
        let mut published: Vec<Option<TemporalShape>> = Vec::new();
        for order in permutations(&[0, 1, 2]) {
            for left_deep in [true, false] {
                published.push(chained(&claims, &order, left_deep));
            }
        }
        assert_eq!(published.len(), 12, "3! orders in both bracketings");
        assert!(
            published.windows(2).all(|pair| pair[0] == pair[1]),
            "the derived shape depends on bracketing or on direction: {published:?}"
        );
        assert_eq!(
            published[0], None,
            "two `events` tie at the top over different axes, so the chain publishes none"
        );
    }

    #[test]
    fn the_derivation_is_invariant_over_a_chain_that_publishes() {
        let claims = [
            shape(TemporalType::events("sem.order"), Some(Granularity::Day)),
            shape(
                TemporalType::snapshot("sem.captured"),
                Some(Granularity::Day),
            ),
            shape(
                TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
                Some(Granularity::Day),
            ),
        ];
        let expected = Some(shape(
            TemporalType::events("sem.order"),
            Some(Granularity::Day),
        ));
        for order in permutations(&[0, 1, 2]) {
            for left_deep in [true, false] {
                assert_eq!(
                    chained(&claims, &order, left_deep),
                    expected,
                    "order {order:?}, left_deep {left_deep}"
                );
            }
        }
    }

    fn chained(
        claims: &[TemporalShape; 3],
        order: &[usize],
        left_deep: bool,
    ) -> Option<TemporalShape> {
        let names = ["x", "y", "z"];
        let mut entries: Vec<SemanticSource> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                entry(
                    name,
                    Some(claims[order[i]].clone()),
                    RoleDefinition::dataset(format!("src.{name}"), Selector::Table),
                )
            })
            .collect();
        let (inner, outer) = if left_deep {
            (("x", "y"), ("inner", "z"))
        } else {
            (("y", "z"), ("x", "inner"))
        };
        entries.push(entry("inner", None, joined(inner.0, inner.1)));
        entries.push(entry("outer", None, joined(outer.0, outer.1)));
        let mut sources = collection(entries);
        derive_composed_shapes(&mut sources).expect("a chain of joins derives");
        sources["outer"].temporal.clone()
    }

    fn joined(from: &str, to: &str) -> RoleDefinition {
        let mut pool = ManifestExpressions::default();
        let condition = pool
            .emit(
                &Expr::Column(
                    NamedReference::qualified("x", "region").expect("an alias and a name"),
                ),
                from,
                ExprRoot::JoinsetOn,
                Vec::<EntityId>::new(),
            )
            .expect("emit");
        RoleDefinition::Joinset {
            from: from.to_owned(),
            to: to.to_owned(),
            on_expr_id: Some(condition),
            filter_expr_id: None,
            join_type: semstrait_ir::JoinsetJoinType::Join(semstrait_ir::JoinType::Inner),
            cardinality: semstrait_ir::Cardinality::ManyToOne,
            integrity: Integrity::Assumed,
            cross_filter: CrossFilter::None,
            description: None,
            ai_context: None,
        }
    }

    fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut all = Vec::new();
        for (i, head) in items.iter().enumerate() {
            let mut rest = items.to_vec();
            rest.remove(i);
            for mut tail in permutations(&rest) {
                tail.insert(0, *head);
                all.push(tail);
            }
        }
        all
    }

    #[test]
    fn an_operand_with_no_shape_abstains_from_the_maximum() {
        let shapes = one_of_each();
        let ids = ids(3);
        assert_eq!(
            highest_rank_type(&participants(&ids, &[None, Some(&shapes[0]), None])),
            Some(shapes[0].temporal_type.clone()),
            "an unclassified operand is not a shape to outrank or to disagree with"
        );
        assert_eq!(
            highest_rank_type(&participants(&ids, &[None, None, None])),
            None,
            "every operand abstaining publishes no axis"
        );
    }

    #[test]
    fn equal_rank_with_different_bodies_yields_absence() {
        let one = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let other = shape(TemporalType::events("sem.ship"), Some(Granularity::Day));
        let lower = shape(
            TemporalType::scd(ScdType::Type1, "f", "t"),
            Some(Granularity::Day),
        );
        let ids = ids(3);
        assert_eq!(
            highest_rank_type(&participants(
                &ids,
                &[Some(&one), Some(&other), Some(&lower)]
            )),
            None,
            "the tie is not resolved by falling back to the rank below it"
        );
        assert_eq!(
            highest_rank_type(&participants(&ids, &[Some(&one), Some(&one), Some(&lower)])),
            Some(one.temporal_type.clone()),
            "equal rank with the *same* body is agreement, not a tie"
        );
    }

    #[test]
    fn a_tie_below_the_top_rank_does_not_erase_the_top() {
        let top = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let one = shape(TemporalType::snapshot("sem.a"), Some(Granularity::Day));
        let other = shape(TemporalType::snapshot("sem.b"), Some(Granularity::Day));
        let ids = ids(3);
        assert_eq!(
            highest_rank_type(&participants(&ids, &[Some(&one), Some(&other), Some(&top)])),
            Some(top.temporal_type.clone()),
            "only a tie *at the top* leaves the surface unclassified"
        );
    }

    #[test]
    fn unanimity_over_absence_alone_publishes_nothing() {
        let ids = ids(2);
        let held = "stack".to_owned();
        assert_eq!(
            unanimous_type(&held, &participants(&ids, &[None, None]), None).expect("agreeing"),
            None,
            "agreeing includes both being absent"
        );
    }

    #[test]
    fn a_stated_claim_joins_the_set_rather_than_overriding_it() {
        let stated = TemporalType::events("sem.order");
        let ids = ids(2);
        let held = "stack".to_owned();
        assert_eq!(
            unanimous_type(&held, &participants(&ids, &[None, None]), Some(&stated))
                .expect("absent constituents agree vacuously"),
            Some(stated.clone()),
            "the stated claim stands where nothing contradicts it"
        );
        let other = shape(TemporalType::events("sem.ship"), Some(Granularity::Day));
        let error = unanimous_type(
            &held,
            &participants(&ids, &[Some(&other), None]),
            Some(&stated),
        )
        .expect_err("a stated claim does not override a constituent");
        let CompileError::ComposedTemporalDisagreement { shapes, .. } = error else {
            panic!("expected the disagreement variant");
        };
        assert_eq!(
            shapes,
            vec![
                (ids[0].clone(), other.temporal_type.clone()),
                (held.clone(), stated),
            ],
            "the payload lists the constituents first and the stack's own claim last"
        );
    }

    #[test]
    fn unanimity_admits_a_participant_that_publishes_nothing() {
        let events = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let ids = ids(2);
        let held = "stack".to_owned();
        assert_eq!(
            unanimous_type(&held, &participants(&ids, &[Some(&events), None]), None)
                .expect("agreeing"),
            Some(events.temporal_type.clone()),
            "an absent participant agrees vacuously rather than forcing absence"
        );
    }

    #[test]
    fn unanimity_compares_the_whole_shape_and_not_the_variant_tag() {
        let one = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let other = shape(TemporalType::events("sem.ship"), Some(Granularity::Day));
        let ids = ids(2);
        let held = "stack".to_owned();
        let error = unanimous_type(
            &held,
            &participants(&ids, &[Some(&one), Some(&other)]),
            None,
        )
        .expect_err("two axes must be reported");
        let CompileError::ComposedTemporalDisagreement { composite, shapes } = error else {
            panic!("expected the disagreement variant");
        };
        assert_eq!(composite, held);
        assert_eq!(
            shapes,
            vec![
                (ids[0].clone(), one.temporal_type.clone()),
                (ids[1].clone(), other.temporal_type.clone()),
            ],
            "the payload names who disagreed and what each publishes, in order"
        );
    }

    #[test]
    fn unanimity_reads_the_scd_type() {
        let one = shape(TemporalType::scd(ScdType::Type1, "f", "t"), None);
        let other = shape(TemporalType::scd(ScdType::Type2, "f", "t"), None);
        let ids = ids(2);
        assert!(unanimous_type(
            &"stack".to_owned(),
            &participants(&ids, &[Some(&one), Some(&other)]),
            None
        )
        .is_err());
    }

    #[test]
    fn the_grain_is_no_part_of_unanimity_and_folds_on_its_own() {
        let day = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let month = shape(TemporalType::events("sem.order"), Some(Granularity::Month));
        let none = shape(TemporalType::events("sem.order"), None);
        let ids = ids(2);
        let held = "stack".to_owned();
        assert!(
            unanimous_type(
                &held,
                &participants(&ids, &[Some(&day), Some(&month)]),
                None
            )
            .is_ok(),
            "one axis, two grains: not a disagreement"
        );
        let granularities = |supplied: &[Option<&TemporalShape>]| {
            (participants(&ids, supplied).into_iter())
                .filter_map(|(_, shape)| shape?.granularity)
                .collect::<Vec<Granularity>>()
        };
        assert_eq!(
            unanimous_granularity(granularities(&[Some(&day), Some(&month)])),
            None,
            "differing grains guarantee no single grain"
        );
        assert_eq!(
            unanimous_granularity(granularities(&[Some(&day), Some(&none)])),
            Some(Granularity::Day),
            "a participant with no grain abstains rather than erasing one"
        );
        assert_eq!(
            unanimous_granularity(granularities(&[None, None])),
            None,
            "nothing to agree on guarantees no grain"
        );
    }

    #[test]
    fn a_unanimous_grain_rides_a_unanimous_kind_and_a_tie_publishes_neither() {
        let day = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let other_axis = shape(TemporalType::events("sem.ship"), Some(Granularity::Day));

        let agreed = collection(vec![
            entry(
                "l",
                Some(day.clone()),
                RoleDefinition::dataset("src.l", Selector::Table),
            ),
            entry(
                "r",
                Some(day.clone()),
                RoleDefinition::dataset("src.r", Selector::Table),
            ),
            entry("j", None, joined("l", "r")),
        ]);
        assert_eq!(
            composed_shape(&agreed["j"], &agreed).expect("a join derives"),
            Some(day.clone()),
            "one kind at one grain over both operands is agreement, so the join keeps \
             the grain the operands guarantee"
        );

        let tied = collection(vec![
            entry(
                "l",
                Some(day),
                RoleDefinition::dataset("src.l", Selector::Table),
            ),
            entry(
                "r",
                Some(other_axis),
                RoleDefinition::dataset("src.r", Selector::Table),
            ),
            entry("j", None, joined("l", "r")),
        ]);
        assert_eq!(
            composed_shape(&tied["j"], &tied).expect("a join derives"),
            None,
            "a tie at the top rank over different axes leaves the surface unclassified, \
             and no shape can carry a grain alone"
        );
    }

    #[test]
    fn the_walk_puts_every_operand_before_the_entry_composing_it() {
        let leaf_a = entry("a", None, RoleDefinition::dataset("src.a", Selector::Table));
        let leaf_b = entry("b", None, RoleDefinition::dataset("src.b", Selector::Table));
        let stack = entry(
            "stack",
            None,
            RoleDefinition::unionset(
                MultisetOp::UnionAll,
                vec!["a".to_owned(), "b".to_owned()],
                true,
            ),
        );
        let order = bottom_up_order(&collection(vec![stack, leaf_a, leaf_b]));
        let position = |name: &str| {
            order
                .iter()
                .position(|id| id.as_str() == name)
                .unwrap_or_else(|| panic!("`{name}` must be visited"))
        };
        assert!(position("a") < position("stack"));
        assert!(position("b") < position("stack"));
        assert_eq!(
            order.len(),
            3,
            "each entry once, however many parents name it"
        );
    }

    #[test]
    fn a_cyclic_collection_terminates_the_walk() {
        let one = entry(
            "one",
            None,
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["two".to_owned()], true),
        );
        let two = entry(
            "two",
            None,
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["one".to_owned()], true),
        );
        let mut sources = collection(vec![one, two]);
        assert_eq!(bottom_up_order(&sources).len(), 2);
        assert!(
            derive_composed_shapes(&mut sources).is_ok(),
            "a cycle is not this stage's refusal"
        );
    }

    #[test]
    fn a_derived_shape_feeds_the_entry_above_it() {
        let events = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let scd = shape(
            TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
            Some(Granularity::Day),
        );
        let mut sources = stack_joined_to_events();
        derive_composed_shapes(&mut sources).expect("the fixture must derive");
        assert_eq!(
            sources["stack"].temporal,
            Some(scd),
            "the stack publishes what its two constituents agree on"
        );
        assert_eq!(
            sources["joined"].temporal,
            Some(events),
            "and the unit above it outranks that derived value"
        );
    }

    #[test]
    fn a_shape_derived_at_each_seal_is_the_shape_the_final_derivation_stores() {
        let mut derived = stack_joined_to_events();
        derive_composed_shapes(&mut derived).expect("the fixture must derive");
        let authored = stack_joined_to_events();
        let mut sealed = BTreeMap::new();
        for id in bottom_up_order(&authored) {
            sealed.insert(id.clone(), authored[&id].clone());
            derive_sealed_shape(&mut sealed, &id);
        }
        assert_eq!(
            sealed, derived,
            "each unit seals after its operands, so the shape it derives at its seal is the \
             shape the final derivation stores"
        );
        derive_composed_shapes(&mut sealed).expect("the sealed shapes derive");
        assert_eq!(
            sealed, derived,
            "and the final derivation over the sealed shapes stores them unchanged"
        );
    }

    #[test]
    fn a_unit_whose_derivation_refuses_at_its_seal_keeps_its_authored_shape() {
        let mut sources = collection(vec![
            entry(
                "a",
                Some(shape(
                    TemporalType::events("sem.order"),
                    Some(Granularity::Day),
                )),
                RoleDefinition::dataset("src.a", Selector::Table),
            ),
            entry(
                "b",
                Some(shape(
                    TemporalType::snapshot("sem.captured"),
                    Some(Granularity::Day),
                )),
                RoleDefinition::dataset("src.b", Selector::Table),
            ),
            entry(
                "stack",
                None,
                RoleDefinition::unionset(
                    MultisetOp::UnionAll,
                    vec!["a".to_owned(), "b".to_owned()],
                    true,
                ),
            ),
        ]);
        derive_sealed_shape(&mut sources, &"stack".to_owned());
        assert_eq!(
            sources["stack"].temporal, None,
            "the branches disagree, so the stack keeps the shape it states"
        );
        assert!(
            derive_composed_shapes(&mut sources).is_err(),
            "and the final derivation refuses the stack"
        );
    }

    fn stack_joined_to_events() -> BTreeMap<EntityId, SemanticSource> {
        let events = shape(TemporalType::events("sem.order"), Some(Granularity::Day));
        let scd = shape(
            TemporalType::scd(ScdType::Type2, "sem.from", "sem.to"),
            Some(Granularity::Day),
        );
        collection(vec![
            entry(
                "a",
                Some(scd.clone()),
                RoleDefinition::dataset("src.a", Selector::Table),
            ),
            entry(
                "b",
                Some(scd),
                RoleDefinition::dataset("src.b", Selector::Table),
            ),
            entry(
                "stack",
                None,
                RoleDefinition::unionset(
                    MultisetOp::UnionAll,
                    vec!["a".to_owned(), "b".to_owned()],
                    true,
                ),
            ),
            entry(
                "c",
                Some(events),
                RoleDefinition::dataset("src.c", Selector::Table),
            ),
            entry("joined", None, joined("stack", "c")),
        ])
    }
}
