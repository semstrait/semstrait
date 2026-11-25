use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::DataKindName;
use semstrait_common::EntityId;
use semstrait_ir::{JoinType, SemanticRole};
use semstrait_model::{Relationship, SemanticInterface};

use crate::error::CompileError;
use crate::identity::mint_signature_id;
use crate::manifest::keyset::{Keyset, KeysetName};
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::{RoleDefinition, SemanticSource};

use super::assemble::{PlannedForm, SourcePlan};
use super::definitions;
use super::relationships::{planning_relationship, resolve_basis, resolve_tuple, Basis};
use super::types::SemanticsIndex;

pub(crate) type Members = BTreeSet<EntityId>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GuaranteedKeyset {
    pub(crate) id: EntityId,
    pub(crate) name: KeysetName,
    pub(crate) members: Members,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Boundary {
    Unconstrained,
    Elements(Vec<Members>),
}

impl Boundary {
    pub(crate) fn admits(&self, group: &Members) -> Result<(), EntityId> {
        let Self::Elements(elements) = self else {
            return Ok(());
        };
        if group.is_empty() {
            return Ok(());
        }
        let mut best: Option<(usize, &EntityId)> = None;
        for element in elements {
            let mut outside = group.difference(element);
            let Some(first) = outside.next() else {
                return Ok(());
            };
            let count = 1 + outside.count();
            if best.is_none_or(|(fewest, _)| count < fewest) {
                best = Some((count, first));
            }
        }
        Err(best.map_or_else(
            || group.iter().next().cloned().unwrap_or_default(),
            |(_, dimension)| dimension.clone(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fold {
    Union,
    Intersection,
}

#[must_use]
pub(crate) fn union_boundary(sides: &[Vec<Members>]) -> Vec<Members> {
    product(sides, Fold::Union)
}

#[must_use]
pub(crate) fn intersection_boundary(grains: &[Vec<Members>]) -> Vec<Members> {
    let mut elements = product(grains, Fold::Intersection);
    elements.retain(|element| !element.is_empty());
    elements
}

fn product(sides: &[Vec<Members>], fold: Fold) -> Vec<Members> {
    let Some((first, rest)) = sides.split_first() else {
        return Vec::new();
    };
    if sides.iter().any(Vec::is_empty) {
        return Vec::new();
    }
    let mut elements: BTreeSet<Members> = first.iter().cloned().collect();
    for side in rest {
        let mut folded: BTreeSet<Members> = BTreeSet::new();
        for partial in &elements {
            for group in side {
                let mut element = partial.clone();
                match fold {
                    Fold::Union => element.extend(group.iter().cloned()),
                    Fold::Intersection => element.retain(|member| group.contains(member)),
                }
                folded.insert(element);
            }
        }
        elements = folded;
    }
    elements.into_iter().collect()
}

fn side_groups(sides: &[&[GuaranteedKeyset]]) -> Vec<Vec<Members>> {
    sides
        .iter()
        .map(|side| side.iter().map(|keyset| keyset.members.clone()).collect())
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Side<'a> {
    pub(crate) name: &'a str,
    pub(crate) keysets: &'a [GuaranteedKeyset],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DerivableSides {
    Both,
    From,
    To,
    Neither,
}

impl DerivableSides {
    #[must_use]
    pub(crate) fn of(join_type: JoinType) -> Self {
        #[allow(unreachable_patterns)]
        match join_type {
            JoinType::Inner => Self::Both,
            JoinType::Left => Self::From,
            JoinType::Right => Self::To,
            JoinType::Full => Self::Neither,
            _ => Self::Neither,
        }
    }
}

#[must_use]
pub(crate) fn derive_joinset_keysets(
    from: Side<'_>,
    to: Side<'_>,
    basis: Option<&Members>,
    sides: DerivableSides,
) -> Vec<(KeysetName, Members)> {
    let Some(basis) = basis else {
        return Vec::new();
    };
    let mut candidates: Vec<(KeysetName, Members)> = Vec::new();

    for left in from.keysets {
        for right in to.keysets {
            candidates.push((
                KeysetName::from(format!(
                    "{}_{}",
                    qualified(from.name, &left.name),
                    qualified(to.name, &right.name)
                )),
                left.members.union(&right.members).cloned().collect(),
            ));
        }
    }

    let determined = |side: Side<'_>| {
        side.keysets
            .iter()
            .any(|keyset| !keyset.members.is_empty() && keyset.members.is_subset(basis))
    };
    if determined(from) {
        candidates.extend(standalone(to));
    }
    if determined(to) {
        candidates.extend(standalone(from));
    }

    let allowed = derivable_members(from, to, sides);
    candidates.retain(|(_, group)| !group.is_empty() && group.is_subset(&allowed));

    minimal(candidates)
}

fn standalone(side: Side<'_>) -> Vec<(KeysetName, Members)> {
    side.keysets
        .iter()
        .map(|keyset| {
            (
                KeysetName::from(qualified(side.name, &keyset.name)),
                keyset.members.clone(),
            )
        })
        .collect()
}

fn qualified(side: &str, name: &KeysetName) -> String {
    format!("{side}.{name}")
}

fn derivable_members(from: Side<'_>, to: Side<'_>, sides: DerivableSides) -> Members {
    let pool = |side: Side<'_>| -> Members {
        side.keysets
            .iter()
            .flat_map(|keyset| keyset.members.iter().cloned())
            .collect()
    };
    match sides {
        DerivableSides::Both => pool(from).union(&pool(to)).cloned().collect(),
        DerivableSides::From => pool(from),
        DerivableSides::To => pool(to),
        DerivableSides::Neither => Members::new(),
    }
}

fn minimal(candidates: Vec<(KeysetName, Members)>) -> Vec<(KeysetName, Members)> {
    let mut by_group: BTreeMap<Members, KeysetName> = BTreeMap::new();
    for (name, group) in candidates {
        by_group
            .entry(group)
            .and_modify(|held| {
                if name < *held {
                    *held = name.clone();
                }
            })
            .or_insert(name);
    }
    let groups: Vec<Members> = by_group.keys().cloned().collect();
    let mut out: Vec<(KeysetName, Members)> = by_group
        .into_iter()
        .filter(|(group, _)| {
            !groups
                .iter()
                .any(|other| other.len() < group.len() && other.is_subset(group))
        })
        .map(|(group, name)| (name, group))
        .collect();
    out.sort();
    out
}

#[derive(Debug, Default)]
pub(crate) struct KeysetPlan {
    pub(crate) roster: BTreeMap<EntityId, Keyset>,
    membership: BTreeMap<EntityId, BTreeMap<EntityId, Vec<EntityId>>>,
    guaranteed: BTreeMap<EntityId, Vec<GuaranteedKeyset>>,
}

impl KeysetPlan {
    #[must_use]
    pub(crate) fn guaranteed(&self, source: &EntityId) -> &[GuaranteedKeyset] {
        self.guaranteed.get(source).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn membership(
        &self,
    ) -> impl Iterator<Item = (&EntityId, &BTreeMap<EntityId, Vec<EntityId>>)> {
        self.membership.iter()
    }

    fn record(&mut self, source: &EntityId, groups: Vec<(KeysetName, Members)>) {
        let mut recorded = Vec::with_capacity(groups.len());
        for (name, members) in groups {
            let keyset = Keyset::new(source.as_str(), name.clone());
            let id = keyset.id.clone();
            self.roster.insert(id.clone(), keyset);
            let attribution = self.membership.entry(source.clone()).or_default();
            for member in &members {
                attribution
                    .entry(member.clone())
                    .or_default()
                    .push(id.clone());
            }
            recorded.push(GuaranteedKeyset { id, name, members });
        }
        if let Some(attribution) = self.membership.get_mut(source) {
            for keysets in attribution.values_mut() {
                keysets.sort();
                keysets.dedup();
            }
        }
        self.guaranteed.insert(source.clone(), recorded);
    }
}

pub(crate) fn attribute_signatures(
    plan: &KeysetPlan,
    signatures: &mut BTreeMap<EntityId, SemanticSignature>,
) {
    for (source, attribution) in plan.membership() {
        for (semantic, keysets) in attribution {
            if let Some(signature) = signatures.get_mut(&mint_signature_id(source, semantic)) {
                signature.keysets = keysets.clone();
            }
        }
    }
}

fn authored_groups(
    source: &EntityId,
    interface: &SemanticInterface,
    semantics: &SemanticsIndex<'_>,
) -> Result<Vec<(KeysetName, Members)>, CompileError> {
    let mut groups = Vec::new();
    for (name, member_names) in interface.keysets() {
        let keyset_name = KeysetName::from(name);
        let mut members = Members::new();
        for member in &member_names {
            let pooled =
                semantics
                    .get(member)
                    .ok_or_else(|| CompileError::KeysetMemberNotPooled {
                        data_kind: source.clone(),
                        keyset: keyset_name.clone(),
                        site: member.as_str().to_owned(),
                    })?;
            if pooled.role() != SemanticRole::Dimension {
                return Err(CompileError::KeysetMemberNotADimension {
                    data_kind: source.clone(),
                    keyset: keyset_name.clone(),
                    site: member.as_str().to_owned(),
                    role: pooled.role(),
                });
            }
            members.insert(pooled.id.clone());
        }
        groups.push((keyset_name, members));
    }
    Ok(groups)
}

pub(crate) struct Stage<'p, 'd> {
    names: BTreeMap<&'p EntityId, &'p str>,
    definitions: &'d [SemanticDefinition],
    registry: SemanticsIndex<'d>,
}

impl<'p, 'd> Stage<'p, 'd> {
    pub(crate) fn new(plan: &'p [SourcePlan<'_>], definitions: &'d [SemanticDefinition]) -> Self {
        Self {
            names: (plan.iter())
                .map(|unit| (&unit.id, unit.name.as_str()))
                .collect(),
            definitions,
            registry: definitions::semantics_index_from_defs(definitions),
        }
    }

    pub(crate) fn record(
        &self,
        unit: &SourcePlan<'_>,
        served: &BTreeSet<&EntityId>,
        resolved: &mut KeysetPlan,
    ) -> Result<(), CompileError> {
        let mut authored = authored_groups(&unit.id, unit.form.interface(), &self.registry)?;
        authored.retain(|(_, members)| members.iter().all(|member| served.contains(member)));
        if matches!(unit.form, PlannedForm::Leaf(_)) {
            resolved.record(&unit.id, authored);
            return Ok(());
        }
        self.record_composite(&unit.id, &unit.form, authored, resolved)
    }

    fn record_composite(
        &self,
        id: &EntityId,
        form: &PlannedForm<'_>,
        authored: Vec<(KeysetName, Members)>,
        resolved: &mut KeysetPlan,
    ) -> Result<(), CompileError> {
        let (boundary, derived) = match form {
            PlannedForm::Unionset { branches, .. } => {
                let sides: Vec<&[GuaranteedKeyset]> = branches
                    .iter()
                    .map(|branch| resolved.guaranteed(branch))
                    .collect();
                (
                    Boundary::Elements(union_boundary(&side_groups(&sides))),
                    Vec::new(),
                )
            }
            PlannedForm::Joinset {
                operands,
                relationships,
                interface,
                constituents,
                ..
            } => {
                self.joinset_keysets(operands, relationships, interface, constituents, resolved)?
            }
            PlannedForm::Grainset { grains, .. } => {
                let sides: Vec<&[GuaranteedKeyset]> = grains
                    .iter()
                    .map(|grain| resolved.guaranteed(grain))
                    .collect();
                (
                    Boundary::Elements(intersection_boundary(&side_groups(&sides))),
                    Vec::new(),
                )
            }
            PlannedForm::CrossGrain { finer, coarser, .. } => {
                let sides: Vec<&[GuaranteedKeyset]> = vec![
                    resolved.guaranteed(&finer.1),
                    resolved.guaranteed(&coarser.1),
                ];
                (
                    Boundary::Elements(union_boundary(&side_groups(&sides))),
                    Vec::new(),
                )
            }
            PlannedForm::Leaf(_) => (Boundary::Unconstrained, Vec::new()),
        };

        for (name, group) in &authored {
            if let Err(dimension) = boundary.admits(group) {
                return Err(CompileError::KeysetOutsideBoundary {
                    composite: id.clone(),
                    keyset: name.clone(),
                    group: group.iter().cloned().collect(),
                    dimension,
                });
            }
        }

        let claimed: BTreeSet<&KeysetName> = authored.iter().map(|(name, _)| name).collect();
        if let Some((name, _)) = derived.iter().find(|(name, _)| claimed.contains(name)) {
            return Err(CompileError::DerivedKeysetNameCollision {
                composite: id.clone(),
                keyset: name.clone(),
            });
        }

        let mut groups = authored;
        groups.extend(derived);
        resolved.record(id, groups);
        Ok(())
    }

    fn joinset_keysets(
        &self,
        operands: &(EntityId, EntityId),
        relationships: &[&Relationship],
        interface: &SemanticInterface,
        constituents: &[(DataKindName, EntityId)],
        resolved: &KeysetPlan,
    ) -> Result<(Boundary, Vec<(KeysetName, Members)>), CompileError> {
        let Some(planning) = planning_relationship(relationships) else {
            return Ok((Boundary::Unconstrained, Vec::new()));
        };
        let semantics = definitions::interface_index(interface, self.definitions);
        let side_of = |endpoint: &DataKindName| -> &[GuaranteedKeyset] {
            constituents
                .iter()
                .find(|(name, _)| name == endpoint)
                .map_or(&[], |(_, id)| resolved.guaranteed(id))
        };
        let basis = resolve_basis(
            planning,
            &semantics,
            &semantics,
            side_of(&planning.from),
            side_of(&planning.to),
        )?;
        let Basis::Keys(keys) = basis else {
            return Ok((Boundary::Unconstrained, Vec::new()));
        };

        let from = Side {
            name: self.name_of(&operands.0),
            keysets: resolved.guaranteed(&operands.0),
        };
        let to = Side {
            name: self.name_of(&operands.1),
            keysets: resolved.guaranteed(&operands.1),
        };
        let boundary =
            Boundary::Elements(union_boundary(&side_groups(&[from.keysets, to.keysets])));
        let members: Members = keys.iter().map(|key| key.id.clone()).collect();
        let join_type = resolve_tuple(planning)?.join_type;
        let derived =
            derive_joinset_keysets(from, to, Some(&members), DerivableSides::of(join_type));
        Ok((boundary, derived))
    }

    fn name_of<'a>(&'a self, operand: &'a EntityId) -> &'a str {
        self.names.get(operand).copied().unwrap_or(operand.as_str())
    }
}

pub(crate) fn attribute_grainsets(
    plan: &KeysetPlan,
    sources: &mut BTreeMap<EntityId, SemanticSource>,
) {
    for (id, source) in sources {
        if let RoleDefinition::Grainset { keysets, .. } = &mut source.role_definition {
            *keysets = plan
                .guaranteed(id)
                .iter()
                .map(|guaranteed| guaranteed.id.clone())
                .collect();
            keysets.sort();
            keysets.dedup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(members: &[&str]) -> Members {
        members.iter().map(|m| (*m).to_owned()).collect()
    }

    fn groups(each: &[&[&str]]) -> Vec<Members> {
        each.iter().map(|members| group(members)).collect()
    }

    fn keyset(name: &str, members: &[&str]) -> GuaranteedKeyset {
        GuaranteedKeyset {
            id: format!("ks.{name}"),
            name: KeysetName::from(name),
            members: group(members),
        }
    }

    fn side<'a>(name: &'a str, keysets: &'a [GuaranteedKeyset]) -> Side<'a> {
        Side { name, keysets }
    }

    fn derived(out: &[(KeysetName, Members)]) -> Vec<(String, Vec<String>)> {
        out.iter()
            .map(|(name, members)| {
                (
                    name.as_str().to_owned(),
                    members.iter().cloned().collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    #[test]
    fn the_union_boundary_is_the_product_of_unions_over_every_branch() {
        let boundary = union_boundary(&[groups(&[&["a"], &["b"]]), groups(&[&["a"], &["c"]])]);
        assert_eq!(
            boundary,
            groups(&[&["a"], &["a", "b"], &["a", "c"], &["b", "c"]]),
            "shared dimensions collapse under union, so `a`+`a` is one element `a`"
        );
    }

    #[test]
    fn identical_sides_do_not_enumerate_their_combinations() {
        let branch = groups(&[&["a"], &["b"]]);
        let sides = vec![branch; 30];
        assert_eq!(
            union_boundary(&sides),
            groups(&[&["a"], &["a", "b"], &["b"]]),
            "one keyset per side over thirty identical sides folds to the three distinct \
             unions of `a` and `b`"
        );
    }

    #[test]
    fn a_branch_with_no_keyset_leaves_no_boundary_element() {
        assert!(union_boundary(&[groups(&[&["a"]]), Vec::new()]).is_empty());
        assert!(union_boundary(&[]).is_empty());
    }

    #[test]
    fn the_intersection_boundary_is_adr_0048s_worked_example() {
        let boundary = intersection_boundary(&[
            groups(&[&["campaign_id", "country", "region"]]),
            groups(&[&["campaign_id", "age"]]),
            groups(&[&["campaign_id", "country"]]),
        ]);
        assert_eq!(boundary, groups(&[&["campaign_id"]]));
    }

    #[test]
    fn an_empty_intersection_element_is_dropped() {
        let boundary = intersection_boundary(&[groups(&[&["a"]]), groups(&[&["b"], &["a", "b"]])]);
        assert_eq!(
            boundary,
            groups(&[&["a"]]),
            "the `a`∩`b` choice drops; the `a`∩`ab` choice survives as `a`"
        );
    }

    #[test]
    fn a_group_outside_the_intersection_boundary_is_rejected_by_admits() {
        let boundary = Boundary::Elements(intersection_boundary(&[
            groups(&[&["campaign_id", "country", "region"]]),
            groups(&[&["campaign_id", "age"]]),
            groups(&[&["campaign_id", "country"]]),
        ]));
        let group = group(&["campaign_id", "country"]);
        let dimension = boundary
            .admits(&group)
            .expect_err("`country` is outside the `[campaign_id]` boundary");
        assert_eq!(dimension, "country", "admits names the offending dimension");
    }

    #[test]
    fn compliance_is_subset_of_some_element() {
        let boundary = Boundary::Elements(groups(&[&["a", "b"], &["c"]]));
        assert!(boundary.admits(&group(&["a", "b"])).is_ok(), "equality");
        assert!(boundary.admits(&group(&["a"])).is_ok(), "narrower");
        assert!(boundary.admits(&group(&["c"])).is_ok(), "another element");
        assert_eq!(
            boundary.admits(&group(&["a", "b", "z"])),
            Err("z".to_owned()),
            "a superset of an element is rejected, naming the member outside it"
        );
    }

    #[test]
    fn a_rejection_names_the_dimension_of_the_closest_element() {
        let boundary = Boundary::Elements(groups(&[&["a"], &["b", "c", "d"]]));
        assert_eq!(
            boundary.admits(&group(&["b", "c", "z"])),
            Err("z".to_owned()),
            "the `bcd` element leaves only `z` outside; the `a` element would leave three"
        );
    }

    #[test]
    fn an_unconstrained_boundary_admits_every_group() {
        assert!(Boundary::Unconstrained
            .admits(&group(&["anything", "at", "all"]))
            .is_ok());
        assert_eq!(
            Boundary::Elements(Vec::new()).admits(&group(&["a"])),
            Err("a".to_owned()),
            "no element is a ceiling of nothing, not an absent ceiling"
        );
    }

    #[test]
    fn an_empty_group_always_complies() {
        assert!(Boundary::Elements(Vec::new())
            .admits(&Members::new())
            .is_ok());
    }

    #[test]
    fn the_pair_rule_unions_every_operand_keyset_combination() {
        let left = [keyset("k", &["a"])];
        let right = [keyset("k", &["b"])];
        let out = derive_joinset_keysets(
            side("orders", &left),
            side("customers", &right),
            Some(&group(&["z"])),
            DerivableSides::Both,
        );
        assert_eq!(
            derived(&out),
            vec![(
                "orders.k_customers.k".to_owned(),
                vec!["a".to_owned(), "b".to_owned()]
            )],
            "the pair rule's name is the underscore-join of both qualified names in \
             operand order"
        );
    }

    #[test]
    fn the_determination_rule_finds_the_fact_side_key_and_minimality_prunes_the_pair() {
        let orders = [keyset("natural", &["order_id"])];
        let customers = [keyset("natural", &["customer_id"])];
        let out = derive_joinset_keysets(
            side("orders", &orders),
            side("customers", &customers),
            Some(&group(&["customer_id"])),
            DerivableSides::Both,
        );
        assert_eq!(
            derived(&out),
            vec![("orders.natural".to_owned(), vec!["order_id".to_owned()])],
            "one composed row is one order, and `[order_id]` identifies it"
        );
    }

    #[test]
    fn minimality_prunes_a_strict_superset_and_keeps_incomparable_groups() {
        let left = [keyset("one", &["a"]), keyset("two", &["b"])];
        let right = [keyset("k", &["c"])];
        let out = derive_joinset_keysets(
            side("l", &left),
            side("r", &right),
            Some(&group(&["c"])),
            DerivableSides::Both,
        );
        assert_eq!(
            derived(&out),
            vec![
                ("l.one".to_owned(), vec!["a".to_owned()]),
                ("l.two".to_owned(), vec!["b".to_owned()]),
            ]
        );
    }

    #[test]
    fn one_group_reached_twice_is_one_entity_under_the_name_least_candidate() {
        let left = [keyset("zz", &["a"])];
        let right = [keyset("aa", &["a"])];
        let out = derive_joinset_keysets(
            side("l", &left),
            side("r", &right),
            Some(&group(&["a"])),
            DerivableSides::Both,
        );
        assert_eq!(
            derived(&out),
            vec![("l.zz".to_owned(), vec!["a".to_owned()])],
            "`l.zz` < `l.zz_r.aa` and < `r.aa`, all three claiming `[a]`"
        );
    }

    #[test]
    fn a_preserved_side_restricts_derivation_to_groups_wholly_inside_it() {
        let left = [keyset("k", &["a"])];
        let right = [keyset("k", &["b"])];
        let derive = |sides| {
            derived(&derive_joinset_keysets(
                side("l", &left),
                side("r", &right),
                Some(&group(&["b"])),
                sides,
            ))
        };
        assert_eq!(
            derive(DerivableSides::Both),
            vec![("l.k".to_owned(), vec!["a".to_owned()])],
            "unrestricted: `l.k` stands alone and minimality prunes the pair's [a, b]"
        );
        assert_eq!(
            derive(DerivableSides::From),
            vec![("l.k".to_owned(), vec!["a".to_owned()])],
            "`[a]` lies wholly inside the non-nullable `from` side"
        );
        assert!(
            derive(DerivableSides::To).is_empty(),
            "with `from` nullable, the only surviving group is drawn from the wrong side"
        );
    }

    #[test]
    fn a_full_outer_join_derives_nothing() {
        let left = [keyset("k", &["a"])];
        let right = [keyset("k", &["b"])];
        assert!(derive_joinset_keysets(
            side("l", &left),
            side("r", &right),
            Some(&group(&["a"])),
            DerivableSides::Neither,
        )
        .is_empty());
        assert_eq!(DerivableSides::of(JoinType::Full), DerivableSides::Neither);
        assert_eq!(DerivableSides::of(JoinType::Inner), DerivableSides::Both);
        assert_eq!(DerivableSides::of(JoinType::Left), DerivableSides::From);
        assert_eq!(DerivableSides::of(JoinType::Right), DerivableSides::To);
    }

    #[test]
    fn a_cross_basis_derives_nothing() {
        let left = [keyset("k", &["a"])];
        let right = [keyset("k", &["b"])];
        assert!(derive_joinset_keysets(
            side("l", &left),
            side("r", &right),
            None,
            DerivableSides::Both,
        )
        .is_empty());
    }

    #[test]
    fn an_operand_with_no_keyset_derives_nothing() {
        let left = [keyset("k", &["a"])];
        assert!(derive_joinset_keysets(
            side("l", &left),
            side("r", &[]),
            Some(&group(&["a"])),
            DerivableSides::Both,
        )
        .is_empty());
    }

    #[test]
    fn three_way_composition_recurses_and_the_qualification_chains() {
        let inner = derive_joinset_keysets(
            side("j.a", &[keyset("k", &["x"])]),
            side("j.b", &[keyset("k", &["y"])]),
            Some(&group(&["y"])),
            DerivableSides::Both,
        );
        assert_eq!(
            derived(&inner),
            vec![("j.a.k".to_owned(), vec!["x".to_owned()])]
        );

        let carried: Vec<GuaranteedKeyset> = inner
            .into_iter()
            .map(|(name, members)| GuaranteedKeyset {
                id: format!("ks.{name}"),
                name,
                members,
            })
            .collect();
        let outer = derive_joinset_keysets(
            side("j.r1", &carried),
            side("j.c", &[keyset("k", &["z"])]),
            Some(&group(&["unrelated"])),
            DerivableSides::Both,
        );
        assert_eq!(
            derived(&outer),
            vec![(
                "j.r1.j.a.k_j.c.k".to_owned(),
                vec!["x".to_owned(), "z".to_owned()]
            )],
            "the inner name survives inside the outer one, so the outer id is unique \
             for the reason the inner one was"
        );
    }

    #[test]
    fn the_pair_rules_groups_are_the_boundary_elements() {
        let left = [keyset("one", &["a"]), keyset("two", &["b"])];
        let right = [keyset("k", &["c"])];
        let boundary = union_boundary(&side_groups(&[&left, &right]));
        assert_eq!(boundary, groups(&[&["a", "c"], &["b", "c"]]));
        let pairs = derive_joinset_keysets(
            side("l", &left),
            side("r", &right),
            Some(&group(&["nothing"])),
            DerivableSides::Both,
        );
        let mut pair_groups: Vec<Members> = pairs.into_iter().map(|(_, members)| members).collect();
        pair_groups.sort();
        assert_eq!(pair_groups, boundary);
    }
}
