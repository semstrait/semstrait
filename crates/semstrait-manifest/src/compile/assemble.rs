use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::{DataKindName, Granularity, SemanticsName};
use semstrait_ir::MultisetOp;
use semstrait_model::{
    ComplexDataKind, DataKind, GrainsetBody, Integrity, JoinsetBody, LeafExtras, NestedDataKindRef,
    Relationship, SemanticInterface, SemanticModel, TemporalShape, UnionMode, UnionsetBody,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::Expr;

use crate::error::{CompileError, ReferenceSite};
use crate::identity::{mint_reduction_leaf_id, mint_semantic_source_id, mint_signature_id};
use crate::manifest::bitmap::{SemanticBitmap, SemanticBitmask};
use crate::manifest::expression::{ExprRoot, Link, ManifestExpressions, PhysicalExprId};
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::{RoleDefinition, SemanticSource, SourceRole};
use crate::manifest::serving::Serving;
use crate::manifest::source_alias::{SourceAlias, SourceAliases};

use super::expressions::columns::ColumnTypes;
use super::expressions::MappedLeaf;
use super::qualifiers;
use super::sources::{self, LeafSource};
use super::temporal::resolve_authored_temporal;
use super::types::{LeafPlace, Reference, ResolvedExprEntry, SemanticsIndex};
use super::{definitions, expressions, keysets, relationships};

pub const ORIGINATORS_RULE: &str =
    "semantic_sources.originators must name at most one originator until \
     reuse-as-shared-source lands";

pub const ORIGINATOR_AGREEMENT_RULE: &str =
    "an originator must list the unit it produced among its role_definition operands, unless that \
     unit is a composition over units the same originator produced";

#[derive(Debug)]
pub(crate) struct SourcePlan<'m> {
    pub(crate) id: EntityId,
    pub(crate) name: DataKindName,
    pub(crate) public: bool,
    pub(crate) originators: Vec<EntityId>,
    pub(crate) form: PlannedForm<'m>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LeafForm<'m> {
    pub(crate) extras: &'m LeafExtras,
    pub(crate) interface: &'m SemanticInterface,
    pub(crate) place: LeafPlace,
}

#[derive(Debug)]
pub(crate) enum PlannedForm<'m> {
    Leaf(LeafForm<'m>),
    Joinset {
        constituents: Vec<(DataKindName, EntityId)>,
        operands: (EntityId, EntityId),
        relationships: Vec<&'m Relationship>,
        interface: &'m SemanticInterface,
        stated_shape: Option<&'m TemporalShape>,
        grainset_child: bool,
    },
    Unionset {
        mode: UnionMode,
        branches: Vec<EntityId>,
        interface: &'m SemanticInterface,
        stated_shape: Option<&'m TemporalShape>,
        null_fill: bool,
        grainset_child: bool,
    },
    Grainset {
        grains: Vec<EntityId>,
        interface: &'m SemanticInterface,
        stated_shape: Option<&'m TemporalShape>,
    },
    CrossGrain {
        finer: (DataKindName, EntityId),
        coarser: (DataKindName, EntityId),
        interface: &'m SemanticInterface,
    },
}

impl<'m> PlannedForm<'m> {
    pub(crate) const fn source_role(&self) -> SourceRole {
        match self {
            Self::Leaf { .. } => SourceRole::Dataset,
            Self::Joinset { .. } | Self::CrossGrain { .. } => SourceRole::Joinset,
            Self::Unionset { .. } => SourceRole::Unionset,
            Self::Grainset { .. } => SourceRole::Grainset,
        }
    }

    pub(crate) const fn interface(&self) -> &'m SemanticInterface {
        match self {
            Self::Leaf(LeafForm { interface, .. })
            | Self::Joinset { interface, .. }
            | Self::Unionset { interface, .. }
            | Self::Grainset { interface, .. }
            | Self::CrossGrain { interface, .. } => interface,
        }
    }
}

pub(crate) fn plan_semantic_sources(
    model: &SemanticModel,
) -> Result<Vec<SourcePlan<'_>>, CompileError> {
    let mut plan: Vec<SourcePlan<'_>> = Vec::new();

    for dataset in model.datasets.values() {
        let base = &dataset.body.base;
        let interface = &dataset.semantic_interface;
        let form = PlannedForm::Leaf(LeafForm {
            extras: &base.extras,
            interface,
            place: LeafPlace::Root,
        });
        plan.push(Placement::root(&base.name, base.id.as_deref(), interface).plan(form));
    }
    for set in model.unionsets.values() {
        let base = &set.body.base;
        let at = Placement::root(&base.name, base.id.as_deref(), &set.semantic_interface);
        plan_unionset(&set.body, set.children_ref(), at, model, &mut plan)?;
    }
    for set in model.grainsets.values() {
        let base = &set.body.base;
        let at = Placement::root(&base.name, base.id.as_deref(), &set.semantic_interface);
        plan_grainset(&set.body, set.children_ref(), at, model, &mut plan)?;
    }
    for set in model.joinsets.values() {
        let base = &set.body.base;
        let at = Placement::root(&base.name, base.id.as_deref(), &set.semantic_interface);
        plan_joinset(&set.body, set.children_ref(), at, model, &mut plan)?;
    }
    Ok(plan)
}

#[derive(Clone)]
struct Placement<'m> {
    name: DataKindName,
    id: EntityId,
    public: bool,
    originators: Vec<EntityId>,
    interface: &'m SemanticInterface,
    place: LeafPlace,
}

impl<'m> Placement<'m> {
    fn root(name: &str, id: Option<&str>, interface: &'m SemanticInterface) -> Self {
        let name = DataKindName::from(name);
        Self {
            id: unit_id(id, name.as_str()),
            name,
            public: true,
            originators: Vec::new(),
            interface,
            place: LeafPlace::Root,
        }
    }

    fn nested(&self, name: &str, id: Option<&str>, place: LeafPlace) -> Self {
        let name = child_name(self.name.as_str(), name);
        Self {
            id: unit_id(id, name.as_str()),
            name,
            public: false,
            originators: vec![self.id.clone()],
            interface: self.interface,
            place,
        }
    }

    fn plan(self, form: PlannedForm<'m>) -> SourcePlan<'m> {
        SourcePlan {
            id: self.id,
            name: self.name,
            public: self.public,
            originators: self.originators,
            form,
        }
    }
}

#[derive(Debug)]
struct PlannedChild {
    name: DataKindName,
    id: EntityId,
    granularities: Vec<Granularity>,
}

impl PlannedChild {
    fn new(name: &str, id: EntityId, granularities: Vec<Granularity>) -> Self {
        Self {
            name: DataKindName::from(name),
            id,
            granularities,
        }
    }

    fn granularity(&self) -> Option<Granularity> {
        super::temporal::unanimous_granularity(self.granularities.iter().copied())
    }
}

fn reach_granularities(
    declared: Option<&TemporalShape>,
    children: &[PlannedChild],
) -> Vec<Granularity> {
    match declared.and_then(|shape| shape.granularity) {
        Some(granularity) => vec![granularity],
        None => (children.iter())
            .flat_map(|child| child.granularities.iter().copied())
            .collect(),
    }
}

fn plan_children<'m>(
    at: &Placement<'m>,
    place: LeafPlace,
    children: impl Iterator<Item = NestedDataKindRef<'m>>,
    model: &'m SemanticModel,
    plan: &mut Vec<SourcePlan<'m>>,
) -> Result<Vec<PlannedChild>, CompileError> {
    let mut planned = Vec::new();
    for child in children {
        let name = child.name();
        let nested = |id: Option<&str>| {
            let nested = at.nested(name, id, place);
            (nested.id.clone(), nested)
        };
        let (id, granularities) = match child {
            NestedDataKindRef::Dataset(dataset) => {
                let base = &dataset.body.base;
                let (id, nested) = nested(base.id.as_deref());
                plan.push(nested.plan(PlannedForm::Leaf(LeafForm {
                    extras: &base.extras,
                    interface: at.interface,
                    place,
                })));
                let granularities =
                    (base.extras.temporal.iter()).filter_map(|shape| shape.granularity);
                (id, granularities.collect())
            }
            NestedDataKindRef::Unionset(set) => {
                let (id, nested) = nested(set.body.base.id.as_deref());
                let granularities =
                    plan_unionset(&set.body, set.children_ref(), nested, model, plan)?;
                (id, granularities)
            }
            NestedDataKindRef::Grainset(set) => {
                let (id, nested) = nested(set.body.base.id.as_deref());
                let granularities =
                    plan_grainset(&set.body, set.children_ref(), nested, model, plan)?;
                (id, granularities)
            }
            NestedDataKindRef::Joinset(set) => {
                let (id, nested) = nested(set.body.base.id.as_deref());
                let granularities =
                    plan_joinset(&set.body, set.children_ref(), nested, model, plan)?;
                (id, granularities)
            }
            _ => {
                let site = ReferenceSite::SemanticSource(at.id.clone());
                let rule = "a nested data kind is a Dataset, Unionset, Grainset or Joinset";
                return Err(CompileError::StructuralViolation { site, rule });
            }
        };
        planned.push(PlannedChild::new(name, id, granularities));
    }
    Ok(planned)
}

fn plan_unionset<'m>(
    body: &'m UnionsetBody,
    children: impl Iterator<Item = NestedDataKindRef<'m>>,
    at: Placement<'m>,
    model: &'m SemanticModel,
    plan: &mut Vec<SourcePlan<'m>>,
) -> Result<Vec<Granularity>, CompileError> {
    let branches = plan_children(&at, LeafPlace::UnionsetBranch, children, model, plan)?;
    let stated_shape = body.base.extras.temporal.as_ref();
    let granularities = reach_granularities(stated_shape, &branches);
    let form = PlannedForm::Unionset {
        mode: body.mode,
        branches: branches.into_iter().map(|branch| branch.id).collect(),
        interface: at.interface,
        stated_shape,
        null_fill: true,
        grainset_child: at.place == LeafPlace::GrainsetChild,
    };
    plan.push(at.plan(form));
    Ok(granularities)
}

fn plan_grainset<'m>(
    body: &'m GrainsetBody,
    children: impl Iterator<Item = NestedDataKindRef<'m>>,
    at: Placement<'m>,
    model: &'m SemanticModel,
    plan: &mut Vec<SourcePlan<'m>>,
) -> Result<Vec<Granularity>, CompileError> {
    let grains = plan_children(&at, LeafPlace::GrainsetChild, children, model, plan)?;
    let (id, name, interface) = (at.id.clone(), at.name.clone(), at.interface);
    let stated_shape = body.base.extras.temporal.as_ref();
    let reach = reach_granularities(stated_shape, &grains);
    plan.push(at.plan(PlannedForm::Grainset {
        grains: grains.iter().map(|grain| grain.id.clone()).collect(),
        interface,
        stated_shape,
    }));

    if interface.keysets().is_empty() {
        return Ok(reach);
    }

    for (finer, coarser) in rollup_pairs(&grains) {
        let unit = child_name(
            name.as_str(),
            &format!("{}_{}", finer.name.as_str(), coarser.name.as_str()),
        );
        plan.push(SourcePlan {
            id: mint_semantic_source_id(unit.as_str()),
            name: unit,
            public: false,
            originators: vec![id.clone()],
            form: PlannedForm::CrossGrain {
                finer: (finer.name.clone(), finer.id.clone()),
                coarser: (coarser.name.clone(), coarser.id.clone()),
                interface,
            },
        });
    }
    Ok(reach)
}

fn rollup_pairs(grains: &[PlannedChild]) -> Vec<(&PlannedChild, &PlannedChild)> {
    let mut pairs = Vec::new();
    for finer in grains {
        for coarser in grains {
            let (Some(from), Some(to)) = (finer.granularity(), coarser.granularity()) else {
                continue;
            };
            if from.coarseness() < to.coarseness() {
                pairs.push((finer, coarser));
            }
        }
    }
    pairs
}

fn plan_joinset<'m>(
    body: &'m JoinsetBody,
    children: impl Iterator<Item = NestedDataKindRef<'m>>,
    at: Placement<'m>,
    model: &'m SemanticModel,
    plan: &mut Vec<SourcePlan<'m>>,
) -> Result<Vec<Granularity>, CompileError> {
    let children = plan_children(&at, LeafPlace::JoinsetConstituent, children, model, plan)?;
    let stated_shape = body.base.extras.temporal.as_ref();
    let granularities = reach_granularities(stated_shape, &children);
    let constituents: Vec<(DataKindName, EntityId)> = (children.into_iter())
        .map(|child| (child.name, child.id))
        .collect();

    let in_scope = super::relationships::relationships_in_scope(
        model.relationships.values(),
        &body.relationships,
    );
    let compositions = enumerate_sub_compositions(at.name.as_str(), &constituents, &in_scope)?;

    for composed in compositions {
        let form = PlannedForm::Joinset {
            constituents: constituents.clone(),
            operands: composed.operands,
            relationships: composed.relationships,
            interface: at.interface,
            stated_shape: stated_shape.filter(|_| composed.applies_every_relationship),
            grainset_child: composed.applies_every_relationship
                && at.place == LeafPlace::GrainsetChild,
        };
        plan.push(if composed.applies_every_relationship {
            at.clone().plan(form)
        } else {
            SourcePlan {
                id: mint_semantic_source_id(composed.name.as_str()),
                name: composed.name,
                public: false,
                originators: vec![at.id.clone()],
                form,
            }
        });
    }
    Ok(granularities)
}

pub const MAX_IMPLICIT_ENUMERATION_COUNT: usize = 2000;

type Members = BTreeSet<usize>;

type Applied = BTreeSet<usize>;

#[derive(Debug, Clone, Copy)]
struct Edge<'m> {
    relationship: &'m Relationship,
    from: usize,
    to: usize,
    bypassable: bool,
}

impl Edge<'_> {
    fn opposite(&self, member: usize) -> Option<usize> {
        match member {
            m if m == self.from => Some(self.to),
            m if m == self.to => Some(self.from),
            _ => None,
        }
    }

    fn spans(&self, members: &Members) -> bool {
        members.contains(&self.from) && members.contains(&self.to)
    }
}

struct Composed<'m> {
    name: DataKindName,
    operands: (EntityId, EntityId),
    relationships: Vec<&'m Relationship>,
    applies_every_relationship: bool,
}

struct ChainStep {
    joined: usize,
    closing: Vec<usize>,
}

fn enumerate_sub_compositions<'m>(
    joinset: &str,
    constituents: &[(DataKindName, EntityId)],
    in_scope: &[&'m Relationship],
) -> Result<Vec<Composed<'m>>, CompileError> {
    let edges = joinset_edges(constituents, in_scope);
    let units = enumerate_units(constituents.len(), &edges).map_err(|units| {
        CompileError::EnumerationCeilingExceeded {
            joinset: joinset.to_owned(),
            units,
            ceiling: MAX_IMPLICIT_ENUMERATION_COUNT,
        }
    })?;

    let mut realised: BTreeMap<(Members, Applied), Composed<'m>> = BTreeMap::new();
    for (members, applied) in &units {
        realise_chain(
            joinset,
            constituents,
            &edges,
            members,
            applied,
            &mut realised,
        );
    }

    let whole = (
        (0..constituents.len()).collect::<Members>(),
        (0..edges.len()).collect::<Applied>(),
    );
    if !realised.contains_key(&whole) {
        return Err(no_spanning_composition(joinset, constituents));
    }

    let mut out: Vec<Composed<'m>> = Vec::with_capacity(realised.len());
    for (key, mut composed) in realised {
        composed.applies_every_relationship = key == whole;
        out.push(composed);
    }

    let mut claimed: BTreeMap<&str, &Composed<'m>> = BTreeMap::new();
    for composed in out.iter().filter(|one| !one.applies_every_relationship) {
        if let Some(held) = claimed.insert(composed.name.as_str(), composed) {
            return Err(name_collision(joinset, composed, held));
        }
    }
    for (bare, _) in constituents {
        let held = child_name(joinset, bare.as_str());
        if let Some(composed) = claimed.get(held.as_str()) {
            return Err(constituent_name_collision(joinset, composed, &held));
        }
    }
    Ok(out)
}

fn joinset_edges<'m>(
    constituents: &[(DataKindName, EntityId)],
    in_scope: &[&'m Relationship],
) -> Vec<Edge<'m>> {
    let index = |name: &DataKindName| {
        constituents
            .iter()
            .position(|(bare, _)| bare.as_str() == name.as_str())
    };
    in_scope
        .iter()
        .filter_map(|relationship| {
            let from = index(&relationship.from)?;
            let to = index(&relationship.to)?;
            (from != to).then_some(Edge {
                relationship,
                from,
                to,
                bypassable: matches!(
                    relationship.integrity,
                    Integrity::Enforced | Integrity::Assumed
                ),
            })
        })
        .collect()
}

fn enumerate_units(
    constituents: usize,
    edges: &[Edge<'_>],
) -> Result<Vec<(Members, Applied)>, usize> {
    let blocks = mandatory_blocks(constituents, edges);
    let mut block_members: BTreeMap<usize, Members> = BTreeMap::new();
    for (member, block) in blocks.iter().enumerate() {
        block_members.entry(*block).or_default().insert(member);
    }
    let mut adjacency: BTreeMap<usize, BTreeSet<usize>> = block_members
        .keys()
        .map(|b| (*b, BTreeSet::new()))
        .collect();
    for edge in edges {
        let (left, right) = (blocks[edge.from], blocks[edge.to]);
        if left != right {
            adjacency.entry(left).or_default().insert(right);
            adjacency.entry(right).or_default().insert(left);
        }
    }

    let mut units: Vec<(Members, Applied)> = Vec::new();
    let mut seen: BTreeSet<BTreeSet<usize>> = BTreeSet::new();
    let mut frontier: Vec<BTreeSet<usize>> = Vec::new();
    for block in block_members.keys() {
        let seed: BTreeSet<usize> = BTreeSet::from([*block]);
        seen.insert(seed.clone());
        frontier.push(seed);
    }
    let mut next: usize = 0;
    while next < frontier.len() {
        let chosen = frontier[next].clone();
        next += 1;
        let members: Members = chosen
            .iter()
            .flat_map(|block| block_members[block].iter().copied())
            .collect();
        if members.len() >= 2 {
            enumerate_relationship_sets(&members, chosen.len(), edges, &mut units)?;
        }
        for block in &chosen {
            for neighbour in &adjacency[block] {
                if chosen.contains(neighbour) {
                    continue;
                }
                let mut grown = chosen.clone();
                grown.insert(*neighbour);
                if seen.insert(grown.clone()) {
                    frontier.push(grown);
                }
            }
        }
    }
    Ok(units)
}

fn mandatory_blocks(constituents: usize, edges: &[Edge<'_>]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..constituents).collect();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            let grandparent = parent[parent[node]];
            parent[node] = grandparent;
            node = grandparent;
        }
        node
    }
    for edge in edges.iter().filter(|edge| !edge.bypassable) {
        let (left, right) = (root(&mut parent, edge.from), root(&mut parent, edge.to));
        if left != right {
            parent[left.max(right)] = left.min(right);
        }
    }
    (0..constituents)
        .map(|node| root(&mut parent, node))
        .collect()
}

fn enumerate_relationship_sets(
    members: &Members,
    block_count: usize,
    edges: &[Edge<'_>],
    units: &mut Vec<(Members, Applied)>,
) -> Result<(), usize> {
    let mandatory: Vec<usize> = (0..edges.len())
        .filter(|index| edges[*index].spans(members) && !edges[*index].bypassable)
        .collect();
    let free: Vec<usize> = (0..edges.len())
        .filter(|index| edges[*index].spans(members) && edges[*index].bypassable)
        .collect();

    let slack =
        u32::try_from(free.len().saturating_sub(block_count.saturating_sub(1))).unwrap_or(u32::MAX);
    let lower_bound = 1_usize.checked_shl(slack).unwrap_or(usize::MAX);
    let proven = units.len().saturating_add(lower_bound);
    if proven > MAX_IMPLICIT_ENUMERATION_COUNT {
        return Err(proven);
    }

    let mut chosen: Vec<usize> = Vec::new();
    walk_relationship_sets(0, members, &mandatory, &free, &mut chosen, edges, units)
}

fn walk_relationship_sets(
    index: usize,
    members: &Members,
    mandatory: &[usize],
    free: &[usize],
    chosen: &mut Vec<usize>,
    edges: &[Edge<'_>],
    units: &mut Vec<(Members, Applied)>,
) -> Result<(), usize> {
    let optimistic = mandatory
        .iter()
        .chain(chosen.iter())
        .chain(free[index..].iter())
        .copied();
    if !connects(members, optimistic, edges) {
        return Ok(());
    }
    if index == free.len() {
        let applied: Applied = mandatory.iter().chain(chosen.iter()).copied().collect();
        units.push((members.clone(), applied));
        if units.len() > MAX_IMPLICIT_ENUMERATION_COUNT {
            return Err(units.len());
        }
        return Ok(());
    }
    walk_relationship_sets(index + 1, members, mandatory, free, chosen, edges, units)?;
    chosen.push(free[index]);
    let deeper = walk_relationship_sets(index + 1, members, mandatory, free, chosen, edges, units);
    chosen.pop();
    deeper
}

fn connects(
    members: &Members,
    applied: impl IntoIterator<Item = usize>,
    edges: &[Edge<'_>],
) -> bool {
    let mut component: BTreeMap<usize, usize> = members
        .iter()
        .enumerate()
        .map(|(slot, member)| (*member, slot))
        .collect();
    for index in applied {
        let edge = &edges[index];
        let (Some(&left), Some(&right)) = (component.get(&edge.from), component.get(&edge.to))
        else {
            continue;
        };
        if left != right {
            for slot in component.values_mut() {
                if *slot == right {
                    *slot = left;
                }
            }
        }
    }
    let mut slots = component.values();
    let first = slots.next();
    slots.all(|slot| Some(slot) == first)
}

fn realise_chain<'m>(
    joinset: &str,
    constituents: &[(DataKindName, EntityId)],
    edges: &[Edge<'m>],
    members: &Members,
    applied: &Applied,
    realised: &mut BTreeMap<(Members, Applied), Composed<'m>>,
) {
    let Some(start) = members.iter().copied().min_by(|left, right| {
        constituents[*left]
            .0
            .as_str()
            .cmp(constituents[*right].0.as_str())
    }) else {
        return;
    };

    let mut qualifier = DataKindName::from(joinset);
    let mut left_operand = constituents[start].1.clone();
    let mut prefix: Members = BTreeSet::from([start]);
    let mut prefix_applied: Applied = BTreeSet::new();

    for step in canonical_chain(constituents, edges, members, applied, start) {
        let causing = edges[step.closing[0]];
        let name = child_name(qualifier.as_str(), causing.relationship.name.as_str());
        prefix.insert(step.joined);
        prefix_applied.extend(step.closing.iter().copied());
        let (joined_name, joined) = &constituents[step.joined];
        let relationships: Vec<&Relationship> = (step.closing.iter())
            .map(|index| edges[*index].relationship)
            .collect();
        let operands = if relationships::planning_relationship(&relationships)
            .is_some_and(|planning| planning.from == *joined_name)
        {
            (joined.clone(), left_operand.clone())
        } else {
            (left_operand.clone(), joined.clone())
        };
        realised
            .entry((prefix.clone(), prefix_applied.clone()))
            .or_insert_with(|| Composed {
                name: name.clone(),
                operands,
                relationships,
                applies_every_relationship: false,
            });
        left_operand = mint_semantic_source_id(name.as_str());
        qualifier = name;
    }
}

fn canonical_chain(
    constituents: &[(DataKindName, EntityId)],
    edges: &[Edge<'_>],
    members: &Members,
    applied: &Applied,
    start: usize,
) -> Vec<ChainStep> {
    let mut prefix: Members = BTreeSet::from([start]);
    let mut steps: Vec<ChainStep> = Vec::with_capacity(members.len().saturating_sub(1));
    while prefix.len() < members.len() {
        let reachable = |member: usize| {
            applied.iter().any(|index| {
                edges[*index]
                    .opposite(member)
                    .is_some_and(|other| prefix.contains(&other))
            })
        };
        let Some(joined) = members
            .iter()
            .copied()
            .filter(|member| !prefix.contains(member) && reachable(*member))
            .min_by(|left, right| {
                constituents[*left]
                    .0
                    .as_str()
                    .cmp(constituents[*right].0.as_str())
            })
        else {
            break;
        };
        let mut closing: Vec<usize> = applied
            .iter()
            .copied()
            .filter(|index| {
                edges[*index]
                    .opposite(joined)
                    .is_some_and(|other| prefix.contains(&other))
            })
            .collect();
        let causing = closing
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                edges[**left]
                    .relationship
                    .name
                    .as_str()
                    .cmp(edges[**right].relationship.name.as_str())
            })
            .map(|(slot, _)| slot)
            .unwrap_or_default();
        let first = closing.remove(causing);
        closing.insert(0, first);
        prefix.insert(joined);
        steps.push(ChainStep { joined, closing });
    }
    steps
}

fn no_spanning_composition(
    joinset: &str,
    constituents: &[(DataKindName, EntityId)],
) -> CompileError {
    let present: Vec<&str> = constituents.iter().map(|(bare, _)| bare.as_str()).collect();
    CompileError::RelationshipResolution {
        relationship: joinset.to_owned(),
        reason: format!(
            "no set of the relationships in its scope joins all of its constituents {present:?}, \
             so there is no composition for its own name to resolve to; a Joinset is composed by \
             its relationships, and one they do not span composes nothing"
        ),
    }
}

fn name_collision(joinset: &str, later: &Composed<'_>, held: &Composed<'_>) -> CompileError {
    CompileError::EnumeratedUnitNameCollision {
        joinset: joinset.to_owned(),
        derived_name: later.name.as_str().to_owned(),
        first: held.name.as_str().to_owned(),
        second: later
            .relationships
            .iter()
            .map(|relationship| relationship.name.clone())
            .collect(),
    }
}

fn constituent_name_collision(
    joinset: &str,
    later: &Composed<'_>,
    held: &DataKindName,
) -> CompileError {
    CompileError::EnumeratedUnitNameCollision {
        joinset: joinset.to_owned(),
        derived_name: later.name.as_str().to_owned(),
        first: held.as_str().to_owned(),
        second: later
            .relationships
            .iter()
            .map(|relationship| relationship.name.clone())
            .collect(),
    }
}

fn child_name(parent: &str, name: &str) -> DataKindName {
    DataKindName::from(format!("{parent}.{name}"))
}

pub(super) fn unit_id(authored: Option<&str>, qualified_name: &str) -> EntityId {
    authored.map_or_else(
        || mint_semantic_source_id(qualified_name),
        ToOwned::to_owned,
    )
}

pub(crate) fn leaf_declarations<'m>(
    plan: &[SourcePlan<'m>],
) -> Vec<(DataKindName, &'m LeafExtras)> {
    plan.iter()
        .filter_map(|unit| match unit.form {
            PlannedForm::Leaf(LeafForm { extras, .. }) => Some((unit.name.clone(), extras)),
            PlannedForm::Unionset { .. }
            | PlannedForm::Joinset { .. }
            | PlannedForm::Grainset { .. }
            | PlannedForm::CrossGrain { .. } => None,
        })
        .collect()
}

pub(crate) type LeafReads<'d> = BTreeMap<EntityId, &'d LeafSource>;

pub(crate) fn reduce_multi_source_leaves<'m, 'd>(
    plan: Vec<SourcePlan<'m>>,
    declared: &'d BTreeMap<DataKindName, Vec<LeafSource>>,
) -> (Vec<SourcePlan<'m>>, LeafReads<'d>) {
    let mut reduced: Vec<SourcePlan<'m>> = Vec::with_capacity(plan.len());
    let mut reads: LeafReads<'d> = BTreeMap::new();

    for mut unit in plan {
        let PlannedForm::Leaf(leaf) = unit.form else {
            reduced.push(unit);
            continue;
        };
        let sources: &'d [LeafSource] = declared.get(&unit.name).map_or(&[], Vec::as_slice);

        if sources.len() < 2 {
            if let Some(only) = sources.first() {
                reads.insert(unit.id.clone(), only);
            }
            reduced.push(unit);
            continue;
        }

        let leaves: BTreeMap<(DataKindName, EntityId), &'d LeafSource> = (sources.iter())
            .map(|source| {
                let name = DataKindName::from(format!("{}_{}", unit.name, source.name));
                ((name, mint_reduction_leaf_id(&unit.id, &source.id)), source)
            })
            .collect();
        let mut branches: Vec<EntityId> = Vec::with_capacity(leaves.len());
        for ((name, id), source) in leaves {
            branches.push(id.clone());
            reads.insert(id.clone(), source);
            reduced.push(SourcePlan {
                id,
                name,
                public: false,
                originators: vec![unit.id.clone()],
                form: PlannedForm::Leaf(leaf),
            });
        }
        unit.form = PlannedForm::Unionset {
            mode: UnionMode::All,
            branches,
            interface: leaf.interface,
            stated_shape: None,
            null_fill: false,
            grainset_child: false,
        };
        reduced.push(unit);
    }

    (reduced, reads)
}

pub(crate) fn complete_semantic_sources(
    sources: &mut BTreeMap<EntityId, SemanticSource>,
    registry: &SemanticBitmap,
    signatures: &BTreeMap<EntityId, crate::manifest::SemanticSignature>,
) -> Result<(), CompileError> {
    if let Some(cycle) = detect_operand_cycle(sources) {
        return Err(CompileError::CyclicSemanticSource { cycle });
    }

    let masks = served_masks(sources, registry, signatures);
    for (id, mask) in masks {
        if let Some(entry) = sources.get_mut(&id) {
            entry.semantic_bitmask = mask;
        }
    }

    for entry in sources.values() {
        if !originators_within_reservation(entry) {
            return Err(CompileError::StructuralViolation {
                site: ReferenceSite::SemanticSource(entry.id.clone()),
                rule: ORIGINATORS_RULE,
            });
        }
    }
    for entry in sources.values() {
        if !originators_agree_with_operands(entry, sources) {
            return Err(CompileError::StructuralViolation {
                site: ReferenceSite::SemanticSource(entry.id.clone()),
                rule: ORIGINATOR_AGREEMENT_RULE,
            });
        }
    }
    Ok(())
}

#[must_use]
pub fn detect_operand_cycle(sources: &BTreeMap<EntityId, SemanticSource>) -> Option<Vec<EntityId>> {
    let mut settled: BTreeSet<&EntityId> = BTreeSet::new();
    for id in sources.keys() {
        let mut path: Vec<&EntityId> = Vec::new();
        if let Some(cycle) = walk_operands(id, sources, &mut settled, &mut path) {
            return Some(cycle);
        }
    }
    None
}

fn walk_operands<'s>(
    id: &'s EntityId,
    sources: &'s BTreeMap<EntityId, SemanticSource>,
    settled: &mut BTreeSet<&'s EntityId>,
    path: &mut Vec<&'s EntityId>,
) -> Option<Vec<EntityId>> {
    if settled.contains(id) {
        return None;
    }
    if let Some(start) = path.iter().position(|seen| *seen == id) {
        let mut cycle: Vec<EntityId> = path[start..].iter().map(|e| (*e).clone()).collect();
        cycle.push(id.clone());
        return Some(cycle);
    }
    let entry = sources.get(id)?;
    path.push(id);
    for operand in entry.role_definition.operands() {
        if let Some(cycle) = walk_operands(operand, sources, settled, path) {
            return Some(cycle);
        }
    }
    path.pop();
    settled.insert(id);
    None
}

#[must_use]
pub fn originators_within_reservation(entry: &SemanticSource) -> bool {
    entry.originators.len() <= 1
}

#[must_use]
pub fn originators_agree_with_operands(
    entry: &SemanticSource,
    sources: &BTreeMap<EntityId, SemanticSource>,
) -> bool {
    if entry.originators.is_empty() {
        return true;
    }
    let composed_by_its_originator = entry.originators.iter().any(|originator| {
        sources
            .get(originator)
            .is_some_and(|parent| parent.role_definition.operands().contains(&&entry.id))
    });
    let composed_in_its_scope = sources.values().any(|sibling| {
        sibling.originators == entry.originators
            && sibling.role_definition.operands().contains(&&entry.id)
    });
    let operands = entry.role_definition.operands();
    let composes_its_scope = !operands.is_empty()
        && operands.iter().all(|operand| {
            sources
                .get(*operand)
                .is_some_and(|held| held.originators == entry.originators)
        });
    let unresolvable = entry
        .originators
        .iter()
        .all(|originator| !sources.contains_key(originator));
    composed_by_its_originator || composed_in_its_scope || composes_its_scope || unresolvable
}

pub(crate) fn constituents<'s>(
    entry: &'s SemanticSource,
    sources: &'s BTreeMap<EntityId, SemanticSource>,
) -> BTreeSet<&'s EntityId> {
    let joinset =
        |held: &SemanticSource| matches!(held.role_definition, RoleDefinition::Joinset { .. });
    let enumerated = |held: &SemanticSource| {
        joinset(held) && (held.originators.iter()).any(|id| sources.get(id).is_some_and(joinset))
    };
    let mut seen = BTreeSet::from([&entry.id]);
    let mut found = BTreeSet::new();
    let mut pending = entry.role_definition.operands();
    while let Some(operand) = pending.pop() {
        if !seen.insert(operand) {
            continue;
        }
        match sources.get(operand) {
            Some(unit) if enumerated(unit) => pending.extend(unit.role_definition.operands()),
            _ => {
                found.insert(operand);
            }
        }
    }
    found
}

fn served_masks(
    sources: &BTreeMap<EntityId, SemanticSource>,
    registry: &SemanticBitmap,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
) -> BTreeMap<EntityId, SemanticBitmask> {
    let serving = Serving::new(sources, signatures);
    (sources.keys())
        .map(|id| (id.clone(), registry.mask_of(serving.served(id))))
        .collect()
}

pub(super) fn leaf_source<'s>(
    leaf: &SourcePlan<'_>,
    reads: &LeafReads<'s>,
    sources: &'s BTreeMap<EntityId, sources::ResolvedSource>,
) -> Result<(&'s sources::ResolvedSource, &'s LeafSource), CompileError> {
    reads
        .get(&leaf.id)
        .copied()
        .and_then(|read| Some((sources.get(&read.id)?, read)))
        .ok_or_else(|| dataset_without_source(leaf.name.as_str()))
}

pub(crate) fn reduction_columns(
    plan: &[SourcePlan<'_>],
    leaf_reads: &LeafReads<'_>,
    sources: &BTreeMap<EntityId, sources::ResolvedSource>,
) -> Result<BTreeMap<EntityId, BTreeSet<String>>, CompileError> {
    let by_id: BTreeMap<&EntityId, &SourcePlan<'_>> =
        plan.iter().map(|unit| (&unit.id, unit)).collect();
    let mut unioned = BTreeMap::new();
    for unit in plan {
        let PlannedForm::Unionset {
            branches,
            null_fill: false,
            ..
        } = &unit.form
        else {
            continue;
        };
        let mut columns = BTreeSet::new();
        for branch in branches.iter().filter_map(|id| by_id.get(id)) {
            let (source, _read) = leaf_source(branch, leaf_reads, sources)?;
            let native = source.native.iter().flat_map(|native| &native.columns);
            columns.extend(native.map(|column| column.name.clone()));
        }
        unioned.insert(unit.id.clone(), columns);
    }
    Ok(unioned)
}

pub(super) fn dataset_without_source(dataset: &str) -> CompileError {
    CompileError::DatasetWithoutSource {
        dataset_id: dataset.to_owned(),
    }
}
pub(super) struct ManifestAssembly {
    pub(super) signatures: BTreeMap<EntityId, crate::manifest::SemanticSignature>,
    pub(super) semantic_sources: BTreeMap<EntityId, SemanticSource>,
    pub(super) expressions: crate::manifest::expression::ManifestExpressions,
    pub(super) argument_types: BTreeMap<EntityId, DataType>,
}
impl ManifestAssembly {
    pub(super) fn new() -> Self {
        Self {
            signatures: BTreeMap::new(),
            semantic_sources: BTreeMap::new(),
            expressions: crate::manifest::expression::ManifestExpressions::default(),
            argument_types: BTreeMap::new(),
        }
    }

    pub(super) fn push_source(
        &mut self,
        unit: &SourcePlan<'_>,
        sealed: SealedUnit,
        aliases: &SourceAliases,
    ) -> Result<(), CompileError> {
        let entry = SemanticSource::new(
            unit.id.clone(),
            unit.name.clone(),
            aliases.semantic_alias(&unit.id)?.clone(),
            unit.public,
            crate::manifest::bitmap::SemanticBitmask::empty(),
            sealed.temporal,
            sealed.role_definition,
            unit.originators.clone(),
        )
        .with_semantic_signature_ids(sealed.signature_ids);
        self.semantic_sources.insert(unit.id.clone(), entry);
        Ok(())
    }

    pub(super) fn check_column_qualifiers(
        &self,
        aliases: &SourceAliases,
    ) -> Result<(), CompileError> {
        let fault = (self.semantic_sources.values())
            .flat_map(|entry| {
                qualifiers::misplaced_columns(
                    entry,
                    &self.semantic_sources,
                    &self.signatures,
                    aliases,
                    &self.expressions,
                )
            })
            .next();
        fault.map_or(Ok(()), Err)
    }
}

pub(super) struct SealedUnit {
    pub(super) role_definition: RoleDefinition,
    pub(super) temporal: Option<semstrait_ir::TemporalShape>,
    pub(super) signature_ids: Vec<EntityId>,
}

#[derive(Clone, Copy)]
pub(super) struct Vocabulary<'a> {
    pub(super) definitions: &'a [SemanticDefinition],
    pub(super) pool_bodies:
        &'a BTreeMap<semstrait_common::SemanticsName, semstrait_model::UnresolvedSemanticExpr>,
    pub(super) pool_filters:
        &'a BTreeMap<semstrait_common::SemanticsName, semstrait_model::UnresolvedSemanticExpr>,
    pub(super) columns: &'a BTreeMap<EntityId, ColumnTypes>,
    pub(super) reduction_columns: &'a BTreeMap<EntityId, BTreeSet<String>>,
}

pub(crate) fn seal_order<'p, 'm>(plan: &'p [SourcePlan<'m>]) -> Vec<&'p SourcePlan<'m>> {
    let index: BTreeMap<&EntityId, usize> = plan
        .iter()
        .enumerate()
        .rev()
        .map(|(at, unit)| (&unit.id, at))
        .collect();
    let mut state: Vec<Visit> = vec![Visit::Unseen; plan.len()];
    let mut ordered: Vec<&'p SourcePlan<'m>> = Vec::with_capacity(plan.len());
    for at in 0..plan.len() {
        push_after_operands(at, plan, &index, &mut state, &mut ordered);
    }
    ordered
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Visit {
    Unseen,
    InProgress,
    Emitted,
}

fn push_after_operands<'p, 'm>(
    at: usize,
    plan: &'p [SourcePlan<'m>],
    index: &BTreeMap<&'p EntityId, usize>,
    state: &mut Vec<Visit>,
    ordered: &mut Vec<&'p SourcePlan<'m>>,
) {
    if state[at] != Visit::Unseen {
        return;
    }
    state[at] = Visit::InProgress;
    for operand in planned_operands(&plan[at].form) {
        if let Some(next) = index.get(operand) {
            push_after_operands(*next, plan, index, state, ordered);
        }
    }
    state[at] = Visit::Emitted;
    ordered.push(&plan[at]);
}

fn planned_operands<'p>(form: &'p PlannedForm<'_>) -> Vec<&'p EntityId> {
    match form {
        PlannedForm::Leaf(_) => Vec::new(),
        PlannedForm::Joinset { operands, .. } => vec![&operands.0, &operands.1],
        PlannedForm::Unionset { branches, .. } => branches.iter().collect(),
        PlannedForm::Grainset { grains, .. } => grains.iter().collect(),
        PlannedForm::CrossGrain { finer, coarser, .. } => vec![&finer.1, &coarser.1],
    }
}

pub(super) fn seal_unit(
    unit: &SourcePlan<'_>,
    vocabulary: Vocabulary<'_>,
    keysets: &keysets::KeysetPlan,
    leaf_reads: &LeafReads<'_>,
    sources: &BTreeMap<EntityId, sources::ResolvedSource>,
    aliases: &SourceAliases,
    assembly: &mut ManifestAssembly,
) -> Result<SealedUnit, CompileError> {
    let definitions = vocabulary.definitions;
    let resolve_stated = |stated: Option<&TemporalShape>| {
        stated
            .map(|authored| {
                let index = definitions::semantics_index_from_defs(definitions);
                resolve_authored_temporal(authored, &index, unit)
            })
            .transpose()
    };
    match &unit.form {
        PlannedForm::Leaf(leaf) => {
            let (source, read) = leaf_source(unit, leaf_reads, sources)?;
            let alias = aliases.physical_alias(&source.id)?;
            let (temporal, signature_ids) =
                seal_leaf(unit, leaf, vocabulary, (source, read), alias, assembly)?;
            Ok(SealedUnit {
                role_definition: RoleDefinition::dataset(source.id.clone(), read.selector.clone()),
                temporal,
                signature_ids,
            })
        }
        PlannedForm::Unionset {
            mode,
            branches,
            interface,
            stated_shape,
            null_fill,
            ..
        } => {
            let stated = resolve_stated(*stated_shape)?;
            let mode = match mode {
                UnionMode::All => MultisetOp::UnionAll,
                UnionMode::Unique => MultisetOp::UnionDistinct,
                other => unreachable!("semstrait-model added a UnionMode variant: {other:?}"),
            };
            let role_definition = RoleDefinition::unionset(mode, branches.clone(), *null_fill);
            let signature_ids = expressions::projection::seal_composed_projection(
                unit,
                interface,
                vocabulary,
                &role_definition,
                aliases,
                assembly,
            )?;
            Ok(SealedUnit {
                role_definition,
                temporal: stated,
                signature_ids,
            })
        }
        PlannedForm::Grainset {
            grains,
            interface,
            stated_shape,
        } => {
            let stated = resolve_stated(*stated_shape)?;
            let role_definition = RoleDefinition::grainset(grains.clone(), Vec::new());
            let signature_ids = expressions::grainset_metrics::seal_grainset(
                unit,
                interface,
                vocabulary,
                &role_definition,
                aliases,
                keysets,
                assembly,
            )?;
            Ok(SealedUnit {
                role_definition,
                temporal: stated,
                signature_ids,
            })
        }
        PlannedForm::CrossGrain {
            finer,
            coarser,
            interface,
        } => {
            let (finer, coarser) = (
                constituent_of(finer, aliases)?,
                constituent_of(coarser, aliases)?,
            );
            let role_definition = relationships::resolve_cross_grain(
                &unit.name,
                &unit.id,
                finer,
                coarser,
                relationships::JoinsetInputs {
                    constituents: &[finer, coarser],
                    interface,
                    definitions,
                    keysets,
                    serving: Serving::new(&assembly.semantic_sources, &assembly.signatures),
                },
                &mut assembly.expressions,
            )?;
            let signature_ids = expressions::projection::seal_composed_projection(
                unit,
                interface,
                vocabulary,
                &role_definition,
                aliases,
                assembly,
            )?;
            Ok(SealedUnit {
                role_definition,
                temporal: None,
                signature_ids,
            })
        }
        PlannedForm::Joinset {
            constituents,
            operands,
            relationships: applied,
            interface,
            stated_shape,
            ..
        } => {
            let stated = resolve_stated(*stated_shape)?;
            let constituents: Vec<relationships::Constituent<'_>> = constituents
                .iter()
                .map(|planned| constituent_of(planned, aliases))
                .collect::<Result<_, CompileError>>()?;
            let role_definition = relationships::resolve_joinset(
                applied,
                (&operands.0, &operands.1),
                &unit.id,
                relationships::JoinsetInputs {
                    constituents: &constituents,
                    interface,
                    definitions,
                    keysets,
                    serving: Serving::new(&assembly.semantic_sources, &assembly.signatures),
                },
                &mut assembly.expressions,
            )?;
            let signature_ids = expressions::projection::seal_composed_projection(
                unit,
                interface,
                vocabulary,
                &role_definition,
                aliases,
                assembly,
            )?;
            Ok(SealedUnit {
                role_definition,
                temporal: stated,
                signature_ids,
            })
        }
    }
}

pub(super) fn constituent_of<'a>(
    planned: &'a (DataKindName, EntityId),
    aliases: &'a SourceAliases,
) -> Result<relationships::Constituent<'a>, CompileError> {
    let (name, id) = planned;
    Ok(relationships::Constituent {
        name,
        id,
        alias: aliases.semantic_alias(id)?.as_str(),
    })
}
pub(crate) fn condition_handles(
    entry: &SemanticSource,
) -> impl Iterator<Item = crate::manifest::expression::PhysicalExprId> {
    let (on, filter) = match &entry.role_definition {
        RoleDefinition::Joinset {
            on_expr_id,
            filter_expr_id,
            ..
        } => (*on_expr_id, *filter_expr_id),
        RoleDefinition::Dataset { .. }
        | RoleDefinition::Unionset { .. }
        | RoleDefinition::Grainset { .. } => (None, None),
    };
    on.into_iter().chain(filter)
}
pub(super) fn seal_leaf(
    unit: &SourcePlan<'_>,
    leaf: &LeafForm<'_>,
    vocabulary: Vocabulary<'_>,
    (source, read): (&sources::ResolvedSource, &LeafSource),
    alias: &SourceAlias,
    assembly: &mut ManifestAssembly,
) -> Result<(Option<semstrait_ir::TemporalShape>, Vec<EntityId>), CompileError> {
    let data_kind_id: EntityId = unit.id.clone();
    let &LeafForm {
        extras,
        interface,
        place,
    } = leaf;
    let explicit = matches!(
        extras.semantic_mapping,
        semstrait_model::SemanticMapping::Explicit(_)
    );

    let Vocabulary {
        definitions,
        pool_bodies,
        pool_filters,
        columns: column_types,
        reduction_columns,
    } = vocabulary;
    let index = definitions::semantics_index_from_defs(definitions);

    let temporal = extras
        .temporal
        .as_ref()
        .map(|authored| resolve_authored_temporal(authored, &index, unit))
        .transpose()?;

    let exposed = definitions::exposed_semantics(interface);
    let bodies = expressions::bodies::body_view(interface, pool_bodies);
    let body_free = body_free(
        unit,
        leaf,
        &exposed,
        &index,
        &bodies,
        source,
        reduction_columns,
    )?;
    let dimensions: BTreeMap<&str, (&SemanticDefinition, Option<&str>)> = (exposed.iter())
        .filter_map(|name| index.get(name))
        .filter(|definition| {
            matches!(
                definition.role_definition,
                crate::manifest::SemanticRoleDefinition::Dimension { .. }
            )
        })
        .map(|definition| {
            let leaf = (body_free.iter()).find(|(held, _)| held.name == definition.name);
            let column = leaf.and_then(|(_, leaf)| leaf.as_ref()?.mapped_column());
            (definition.name.as_str(), (definition, column))
        })
        .collect();
    super::catalog::partitions::check_claims(
        extras
            .storage
            .as_ref()
            .map_or(&[], |storage| &storage.partition_def),
        read,
        &source.storage.target,
        &dimensions,
    )?;
    let columns =
        (column_types.get(&source.id)).ok_or_else(|| dataset_without_source(unit.name.as_str()))?;

    let mut entries: BTreeMap<SemanticsName, ResolvedExprEntry> = BTreeMap::new();
    for (definition, leaf) in body_free {
        let name = &definition.name;
        let Some(entry) = expressions::realise(definition, source, leaf, columns)? else {
            if matches!(
                definition.role_definition,
                crate::manifest::SemanticRoleDefinition::Metric { .. }
            ) {
                return Err(CompileError::MetricWithoutExpression {
                    semantics: name.to_string(),
                    semantic_source: unit.name.to_string(),
                });
            }
            continue;
        };

        let entry = expressions::lower::reconcile_boundary(
            entry,
            definition.data_type.as_ref(),
            name.as_str(),
        )?;
        entries.insert(name.clone(), entry);
    }

    let scoped_index = definitions::interface_index(interface, definitions);
    let scope = expressions::bodies::SourceScope {
        source_name: unit.name.as_str(),
        source_id: &data_kind_id,
        exposed: &exposed,
        index: &scoped_index,
        definitions,
        manual_mapping: explicit,
        composed_filter_unit: None,
        reference: Reference::Link,
        columns,
        place: Some(place),
    };
    let filters = expressions::bodies::filter_view(interface, pool_filters);
    let expressions::bodies::SourceLowered {
        bodies: lowered,
        filters: lowered_filters,
        dropped,
        ..
    } = expressions::bodies::lower_source(&expressions::bodies::SourceLowering {
        scope,
        bodies: &bodies,
        body_free: &entries,
        filters: &filters,
    })?;
    entries.retain(|name, _| !dropped.contains(name));

    let realisations = entries
        .into_iter()
        .chain(lowered)
        .filter_map(|(name, entry)| {
            Some(SignatureTree {
                definition: index.get(&name)?,
                expr: entry.expr,
                argument_type: entry.argument_type,
            })
        });
    let signature_ids = seal_signatures(
        &data_kind_id,
        realisations,
        lowered_filters,
        |tree, owner, root, pool| {
            expressions::emit_source_attached(&tree, owner, root, &source.id, alias, pool)
        },
        assembly,
    )?;

    Ok((temporal, signature_ids))
}

fn body_free<'d>(
    unit: &SourcePlan<'_>,
    leaf: &LeafForm<'_>,
    exposed: &BTreeSet<SemanticsName>,
    index: &SemanticsIndex<'d>,
    bodies: &expressions::bodies::SourceBodies<'_>,
    source: &sources::ResolvedSource,
    reduction_columns: &BTreeMap<EntityId, BTreeSet<String>>,
) -> Result<Vec<(&'d SemanticDefinition, Option<MappedLeaf>)>, CompileError> {
    let explicit = match &leaf.extras.semantic_mapping {
        semstrait_model::SemanticMapping::Explicit(entries) => Some(entries),
        _ => None,
    };
    if explicit.is_none() && leaf.place != LeafPlace::Root && source.native.is_none() {
        return Err(CompileError::AutoMappingWithoutSchema {
            dataset: unit.name.to_string(),
        });
    }
    for mapped_name in explicit.into_iter().flat_map(|entries| entries.keys()) {
        if !exposed.contains(mapped_name) {
            return Err(mapping_names_unexposed_semantics(
                unit.name.as_str(),
                mapped_name.as_str(),
            ));
        }
        if bodies.contains_key(mapped_name) {
            return Err(mapping_names_bodied_semantics(
                unit.name.as_str(),
                mapped_name.as_str(),
            ));
        }
    }
    let sibling_columns =
        (unit.originators.first()).and_then(|originator| reduction_columns.get(originator));
    let mut body_free = Vec::new();
    for name in exposed {
        let Some(definition) = index.get(name) else {
            continue;
        };
        if bodies.contains_key(name) {
            continue;
        }
        let mapped = explicit.and_then(|entries| entries.get(name));
        if expressions::needs_mapping_entry(definition) {
            match (explicit, mapped, leaf.place) {
                (Some(_), None, LeafPlace::Root) => {
                    return Err(unmapped_exposed_semantics(
                        unit.name.as_str(),
                        name.as_str(),
                    ))
                }
                (Some(_), None, _) => continue,
                (Some(_), Some(_), _) | (None, _, LeafPlace::Root) => {}
                (None, _, _) => {
                    let observed = match sibling_columns {
                        Some(columns) => columns.contains(name.as_str()),
                        None => source.native.as_ref().is_some_and(|native| {
                            native
                                .columns
                                .iter()
                                .any(|column| column.name == name.as_str())
                        }),
                    };
                    if !observed {
                        continue;
                    }
                }
            }
        }
        body_free.push((definition, expressions::mapped_leaf(definition, mapped)?));
    }
    Ok(body_free)
}

pub(super) fn column_types(
    plan: &[SourcePlan<'_>],
    definitions: &[SemanticDefinition],
    pool_bodies: &BTreeMap<SemanticsName, semstrait_model::UnresolvedSemanticExpr>,
    dialect: Option<semstrait_common::Dialect>,
    leaf_reads: &LeafReads<'_>,
    sources: &BTreeMap<EntityId, sources::ResolvedSource>,
    reduction_columns: &BTreeMap<EntityId, BTreeSet<String>>,
) -> Result<BTreeMap<EntityId, ColumnTypes>, CompileError> {
    let index = definitions::semantics_index_from_defs(definitions);
    let mut mapped: BTreeMap<&EntityId, Vec<(String, &SemanticDefinition)>> = BTreeMap::new();
    for unit in plan {
        let PlannedForm::Leaf(leaf) = &unit.form else {
            continue;
        };
        let (source, _) = leaf_source(unit, leaf_reads, sources)?;
        let exposed = definitions::exposed_semantics(leaf.interface);
        let bodies = expressions::bodies::body_view(leaf.interface, pool_bodies);
        let leaves = body_free(
            unit,
            leaf,
            &exposed,
            &index,
            &bodies,
            source,
            reduction_columns,
        )?;
        mapped
            .entry(&source.id)
            .or_default()
            .extend(leaves.into_iter().filter_map(|(definition, leaf)| {
                Some((leaf?.mapped_column()?.to_owned(), definition))
            }));
    }
    Ok((sources.iter())
        .map(|(id, source)| {
            let mapped = (mapped.get(id).into_iter().flatten())
                .map(|(column, definition)| (column.as_str(), *definition));
            let types = ColumnTypes::new(source.native.as_ref(), mapped, dialect);
            (id.clone(), types)
        })
        .collect())
}

pub(super) struct SignatureTree<'d> {
    pub(super) definition: &'d SemanticDefinition,
    pub(super) expr: Expr<Link>,
    pub(super) argument_type: Option<DataType>,
}

pub(super) fn seal_signatures<'d>(
    source_id: &EntityId,
    signatures: impl IntoIterator<Item = SignatureTree<'d>>,
    mut filters: BTreeMap<SemanticsName, ResolvedExprEntry>,
    mut emit: impl FnMut(
        Expr<Link>,
        &str,
        ExprRoot,
        &mut ManifestExpressions,
    ) -> Result<PhysicalExprId, CompileError>,
    assembly: &mut ManifestAssembly,
) -> Result<Vec<EntityId>, CompileError> {
    let mut stored = Vec::new();
    for SignatureTree {
        definition,
        expr,
        argument_type,
    } in signatures
    {
        let id = mint_signature_id(source_id, &definition.id);
        let expression_id = emit(
            expr,
            &id,
            ExprRoot::SignatureExpression,
            &mut assembly.expressions,
        )?;
        stored.push((definition, id, expression_id, argument_type));
    }
    let mut signature_ids = Vec::with_capacity(stored.len());
    for (definition, id, expression_id, argument_type) in stored {
        let mut signature =
            SemanticSignature::new(id.clone(), definition.id.clone(), expression_id);
        if let Some(filter) = filters.remove(&definition.name) {
            let filter_id = emit(
                filter.expr,
                &id,
                ExprRoot::SignatureFilter,
                &mut assembly.expressions,
            )?;
            signature = signature.with_filter(filter_id);
        }
        if let Some(argument_type) = argument_type {
            assembly.argument_types.insert(id.clone(), argument_type);
        }
        assembly.signatures.insert(id.clone(), signature);
        signature_ids.push(id);
    }
    Ok(signature_ids)
}

fn mapping_names_bodied_semantics(dataset_name: &str, semantic: &str) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: format!("dataset {dataset_name} semantic_mapping"),
        reason: format!(
            "the explicit `semantic_mapping` block maps `{semantic}`, which authors an \
             `expr:` body and so realises from that body rather than over a mapped \
             leaf. Whether a mapping entry may override, compose with or be ignored \
             beside a body is a rule no ADR states, so this is refused rather than \
             guessed"
        ),
    }
}

fn unmapped_exposed_semantics(dataset_name: &str, semantic: &str) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: format!("dataset {dataset_name} semantic_mapping"),
        reason: format!(
            "the explicit `semantic_mapping` block names no entry for the exposed \
             semantics `{semantic}`, and nothing above a root Dataset null-fills it"
        ),
    }
}

fn mapping_names_unexposed_semantics(dataset_name: &str, semantic: &str) -> CompileError {
    CompileError::TypeInferenceFailure {
        node: format!("dataset {dataset_name} semantic_mapping"),
        reason: format!(
            "the explicit `semantic_mapping` block maps `{semantic}`, which this leaf's \
             interface does not expose. Whether a mapping entry may widen an interface is \
             a rule no ADR states, so this is refused rather than read as an implicit \
             exposure"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::manifest::semantic_source::{RoleDefinition, Selector};
    use crate::manifest::source_alias::SourceAlias;

    fn entry(id: &str, name: &str, role_definition: RoleDefinition) -> SemanticSource {
        SemanticSource::new(
            id,
            DataKindName::from(name),
            SourceAlias::from(""),
            true,
            SemanticBitmask::empty(),
            None,
            role_definition,
            Vec::new(),
        )
    }

    fn roster(
        entries: impl IntoIterator<Item = SemanticSource>,
    ) -> BTreeMap<EntityId, SemanticSource> {
        entries
            .into_iter()
            .map(|entry| (entry.id.clone(), entry))
            .collect()
    }

    fn planned_child(name: &str, granularity: Option<Granularity>) -> PlannedChild {
        PlannedChild {
            name: DataKindName::from(name),
            id: EntityId::from(format!("dk.{name}")),
            granularities: granularity.into_iter().collect(),
        }
    }

    #[test]
    fn every_ordered_grain_pair_runs_finer_to_coarser_exactly_once() {
        let scrambled = [
            planned_child("yearly", Some(Granularity::Year)),
            planned_child("hourly", Some(Granularity::Hour)),
            planned_child("monthly", Some(Granularity::Month)),
            planned_child("daily", Some(Granularity::Day)),
        ];
        let pairs: Vec<(&str, &str)> = rollup_pairs(&scrambled)
            .into_iter()
            .map(|(finer, coarser)| (finer.name.as_str(), coarser.name.as_str()))
            .collect();

        let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
        for (finer, coarser) in &pairs {
            assert!(seen.insert((finer, coarser)), "`{finer}_{coarser}` repeats");
            assert!(
                !seen.contains(&(*coarser, *finer)),
                "`{coarser}_{finer}` is the reverse direction and must not be emitted"
            );
        }
        assert_eq!(pairs.len(), 6, "four grains give six pairs, got {pairs:?}");

        let mut sorted = pairs;
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![
                ("daily", "monthly"),
                ("daily", "yearly"),
                ("hourly", "daily"),
                ("hourly", "monthly"),
                ("hourly", "yearly"),
                ("monthly", "yearly"),
            ],
            "every pair, and each in the rollup direction only"
        );
    }

    #[test]
    fn a_grain_with_no_authored_grain_pairs_with_nothing() {
        let grains = [
            planned_child("daily", Some(Granularity::Day)),
            planned_child("unstamped", None),
            planned_child("monthly", Some(Granularity::Month)),
        ];
        let pairs: Vec<(&str, &str)> = rollup_pairs(&grains)
            .into_iter()
            .map(|(finer, coarser)| (finer.name.as_str(), coarser.name.as_str()))
            .collect();
        assert_eq!(
            pairs,
            vec![("daily", "monthly")],
            "the ungrained grain is neither finer nor coarser than anything"
        );
    }

    #[test]
    fn a_loop_over_operand_edges_is_found() {
        let sources = roster([
            entry(
                "dk.a",
                "a",
                RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.b".to_owned()], true),
            ),
            entry(
                "dk.b",
                "b",
                RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.a".to_owned()], true),
            ),
        ]);
        let cycle = detect_operand_cycle(&sources).expect("the loop must be found");
        assert!(cycle.contains(&"dk.a".to_owned()), "got {cycle:?}");
        assert!(cycle.contains(&"dk.b".to_owned()), "got {cycle:?}");
    }

    #[test]
    fn a_self_operand_is_a_cycle() {
        let sources = roster([entry(
            "dk.a",
            "a",
            RoleDefinition::grainset(vec!["dk.a".to_owned()], Vec::new()),
        )]);
        assert_eq!(
            detect_operand_cycle(&sources),
            Some(vec!["dk.a".to_owned(), "dk.a".to_owned()])
        );
    }

    #[test]
    fn one_operand_under_two_parents_is_not_a_cycle() {
        let sources = roster([
            entry(
                "dk.leaf",
                "leaf",
                RoleDefinition::dataset("src.orders", Selector::Table),
            ),
            entry(
                "dk.one",
                "one",
                RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.leaf".to_owned()], true),
            ),
            entry(
                "dk.two",
                "two",
                RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.leaf".to_owned()], true),
            ),
        ]);
        assert_eq!(detect_operand_cycle(&sources), None);
    }

    #[test]
    fn a_dataset_arm_never_contributes_an_operand_edge() {
        let sources = roster([entry(
            "dk.a",
            "a",
            RoleDefinition::dataset("dk.a", Selector::Table),
        )]);
        assert_eq!(detect_operand_cycle(&sources), None);
    }

    #[test]
    fn a_dangling_operand_is_not_reported_as_a_cycle() {
        let sources = roster([entry(
            "dk.a",
            "a",
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.missing".to_owned()], true),
        )]);
        assert_eq!(detect_operand_cycle(&sources), None);
    }

    #[test]
    fn the_two_originator_rules_read_the_operand_edges() {
        let mut child = entry(
            "dk.child",
            "u.child",
            RoleDefinition::dataset("src.orders", Selector::Table),
        );
        child.originators = vec!["dk.u".to_owned()];
        let parent = entry(
            "dk.u",
            "u",
            RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.child".to_owned()], true),
        );
        let sources = roster([child.clone(), parent]);
        assert!(originators_within_reservation(&child));
        assert!(originators_agree_with_operands(&child, &sources));

        let mut shared = child.clone();
        shared.originators = vec!["dk.u".to_owned(), "dk.v".to_owned()];
        assert!(!originators_within_reservation(&shared));

        let orphaned = roster([
            child.clone(),
            entry(
                "dk.u",
                "u",
                RoleDefinition::unionset(MultisetOp::UnionAll, vec!["dk.other".to_owned()], true),
            ),
        ]);
        assert!(!originators_agree_with_operands(&child, &orphaned));
    }

    fn empty_interface() -> &'static SemanticInterface {
        static INTERFACE: std::sync::OnceLock<SemanticInterface> = std::sync::OnceLock::new();
        INTERFACE.get_or_init(SemanticInterface::default)
    }

    fn planned(id: &str, name: &str) -> SourcePlan<'static> {
        SourcePlan {
            id: id.to_owned(),
            name: DataKindName::from(name),
            public: true,
            originators: Vec::new(),
            form: PlannedForm::Unionset {
                mode: UnionMode::All,
                branches: Vec::new(),
                interface: empty_interface(),
                stated_shape: None,
                null_fill: true,
                grainset_child: false,
            },
        }
    }

    fn planned_over(id: &str, name: &str, branches: &[&str]) -> SourcePlan<'static> {
        let mut unit = planned(id, name);
        if let PlannedForm::Unionset { branches: held, .. } = &mut unit.form {
            *held = branches.iter().map(|branch| (*branch).to_owned()).collect();
        }
        unit
    }

    fn order_of<'p>(plan: &'p [SourcePlan<'static>]) -> Vec<&'p str> {
        seal_order(plan)
            .into_iter()
            .map(|unit| unit.id.as_str())
            .collect()
    }

    #[test]
    fn seal_order_walks_every_operand_before_its_composer() {
        let plan = vec![
            planned_over("dk.top", "top", &["dk.mid"]),
            planned("dk.leaf", "leaf"),
            planned_over("dk.mid", "mid", &["dk.leaf"]),
        ];
        assert_eq!(order_of(&plan), vec!["dk.leaf", "dk.mid", "dk.top"]);
    }

    #[test]
    fn seal_order_keeps_a_unit_whose_operand_the_plan_does_not_hold() {
        let plan = vec![planned_over("dk.top", "top", &["dk.absent"])];
        assert_eq!(order_of(&plan), vec!["dk.top"]);
    }

    #[test]
    fn seal_order_emits_every_unit_once_over_a_cyclic_plan() {
        let plan = vec![
            planned_over("dk.a", "a", &["dk.b"]),
            planned_over("dk.b", "b", &["dk.a"]),
        ];
        assert_eq!(order_of(&plan), vec!["dk.b", "dk.a"]);
    }

    fn triple() -> Vec<(DataKindName, EntityId)> {
        ["b", "c", "a"]
            .into_iter()
            .map(|name| (DataKindName::from(name), format!("dk.{name}")))
            .collect()
    }

    fn declared(name: &str, from: &str, to: &str, integrity: Integrity) -> Relationship {
        Relationship::builder()
            .name(name)
            .from(from)
            .to(to)
            .on(["k"])
            .many_to_one()
            .integrity(integrity)
            .build()
            .expect("builds")
    }

    fn units_of(
        constituents: &[(DataKindName, EntityId)],
        relationships: &[Relationship],
    ) -> BTreeSet<(Vec<String>, Vec<String>)> {
        let in_scope: Vec<&Relationship> = relationships.iter().collect();
        let edges = joinset_edges(constituents, &in_scope);
        enumerate_units(constituents.len(), &edges)
            .expect("under the ceiling")
            .into_iter()
            .map(|(members, applied)| {
                (
                    members
                        .iter()
                        .map(|m| constituents[*m].0.as_str().to_owned())
                        .collect::<BTreeSet<String>>()
                        .into_iter()
                        .collect(),
                    applied
                        .iter()
                        .map(|e| edges[*e].relationship.name.clone())
                        .collect::<BTreeSet<String>>()
                        .into_iter()
                        .collect(),
                )
            })
            .collect()
    }

    fn owned(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn integrity_decides_which_member_sets_are_candidates() {
        let constituents = triple();
        for bypassable in [Integrity::Enforced, Integrity::Assumed] {
            let relationships = vec![
                declared("r1", "a", "b", bypassable),
                declared("r2", "b", "c", Integrity::Assumed),
            ];
            assert_eq!(
                units_of(&constituents, &relationships),
                BTreeSet::from([
                    (owned(&["a", "b"]), owned(&["r1"])),
                    (owned(&["b", "c"]), owned(&["r2"])),
                    (owned(&["a", "b", "c"]), owned(&["r1", "r2"])),
                ]),
                "`{bypassable:?}` detaches, so every connected subset is a candidate"
            );
        }
        let relationships = vec![
            declared("r1", "a", "b", Integrity::None),
            declared("r2", "b", "c", Integrity::Assumed),
        ];
        assert_eq!(
            units_of(&constituents, &relationships),
            BTreeSet::from([
                (owned(&["a", "b"]), owned(&["r1"])),
                (owned(&["a", "b", "c"]), owned(&["r1", "r2"])),
            ]),
            "`{{b, c}}` would detach `b` from a mandatory relationship"
        );
    }

    #[test]
    fn mandatory_relationships_glue_transitively() {
        let constituents = triple();
        let relationships = vec![
            declared("r1", "a", "b", Integrity::None),
            declared("r2", "b", "c", Integrity::None),
        ];
        assert_eq!(
            units_of(&constituents, &relationships),
            BTreeSet::from([(owned(&["a", "b", "c"]), owned(&["r1", "r2"]))])
        );
    }

    #[test]
    fn one_member_set_carries_every_connecting_relationship_set() {
        let constituents = triple();
        let relationships = vec![
            declared("r1", "a", "b", Integrity::Assumed),
            declared("r2", "b", "c", Integrity::Assumed),
            declared("r3", "a", "c", Integrity::Assumed),
        ];
        let units = units_of(&constituents, &relationships);
        let whole = owned(&["a", "b", "c"]);
        let over_whole: BTreeSet<&Vec<String>> = units
            .iter()
            .filter(|(members, _)| *members == whole)
            .map(|(_, applied)| applied)
            .collect();
        assert_eq!(
            over_whole.len(),
            4,
            "three spanning trees plus the sub-graph applying all three: {over_whole:?}"
        );
        assert!(
            over_whole.contains(&owned(&["r1", "r2", "r3"])),
            "the unit applying every relationship is one of them: {over_whole:?}"
        );
        assert_eq!(units.len(), 7);
    }

    #[test]
    fn a_relationship_reaching_outside_the_joinset_is_no_edge() {
        let constituents = triple();
        let relationships = vec![
            declared("r1", "a", "b", Integrity::Assumed),
            declared("elsewhere", "a", "orders", Integrity::None),
        ];
        let in_scope: Vec<&Relationship> = relationships.iter().collect();
        assert_eq!(
            joinset_edges(&constituents, &in_scope).len(),
            1,
            "only `r1` spans two constituents"
        );
        assert_eq!(
            units_of(&constituents, &relationships),
            BTreeSet::from([(owned(&["a", "b"]), owned(&["r1"]))])
        );
    }

    #[test]
    fn the_ceiling_refuses_rather_than_truncates() {
        let names: Vec<String> = (0..6).map(|i| format!("c{i}")).collect();
        let constituents: Vec<(DataKindName, EntityId)> = names
            .iter()
            .map(|name| (DataKindName::from(name.as_str()), format!("dk.{name}")))
            .collect();
        let mut relationships = Vec::new();
        for (i, left) in names.iter().enumerate() {
            for right in &names[i + 1..] {
                relationships.push(declared(
                    &format!("r_{left}_{right}"),
                    left,
                    right,
                    Integrity::Assumed,
                ));
            }
        }
        let in_scope: Vec<&Relationship> = relationships.iter().collect();
        let edges = joinset_edges(&constituents, &in_scope);
        let reached = enumerate_units(constituents.len(), &edges)
            .expect_err("six pairwise-related constituents breach the ceiling");
        assert!(
            reached > MAX_IMPLICIT_ENUMERATION_COUNT,
            "the count reached is over the ceiling: {reached}"
        );

        let star: Vec<(DataKindName, EntityId)> = ["hub", "a", "b", "c", "d", "e"]
            .into_iter()
            .map(|name| (DataKindName::from(name), format!("dk.{name}")))
            .collect();
        let spokes: Vec<Relationship> = ["a", "b", "c", "d", "e"]
            .into_iter()
            .map(|spoke| declared(&format!("r_{spoke}"), spoke, "hub", Integrity::Assumed))
            .collect();
        assert_eq!(
            units_of(&star, &spokes).len(),
            31,
            "one unit per non-empty subset of the five spokes"
        );
    }
}
