use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::{DataKindName, SemanticsName};
use semstrait_ir::{
    Cardinality as IrCardinality, CrossFilter as IrCrossFilter, Integrity as IrIntegrity, JoinType,
    JoinsetJoinType, SemanticRole,
};
use semstrait_model::{
    Cardinality as ModelCardinality, CrossFilter, Integrity, JoinKeySpec, Optional, Relationship,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{map_nodes, Call, Expr, ExprOperationExt, SemanticLeaf};

use super::expressions::columns::ColumnTypes;
use super::expressions::lower::{
    refused_rebuild, refused_window, Lowered, LoweredExpr, LoweredLeaf,
};

use crate::error::{CompileError, ReferenceSite, RelationshipSide};
use crate::manifest::expression::{ExprRoot, Link, ManifestExpressions};
use crate::manifest::keyset::KeysetName;
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::{RoleDefinition, SemanticSource};
use crate::manifest::serving::Serving;

use super::assemble::condition_handles;
use super::definitions::interface_index;
use super::expressions::emit_composition_attached;
use super::keysets::GuaranteedKeyset;
use super::qualifiers::qualified_column;
use super::types::SemanticsIndex;

fn integrity(authored: semstrait_model::Integrity) -> IrIntegrity {
    match authored {
        semstrait_model::Integrity::Enforced => IrIntegrity::Enforced,
        semstrait_model::Integrity::Assumed => IrIntegrity::Assumed,
        semstrait_model::Integrity::None => IrIntegrity::None,
        other => unreachable!("semstrait-model added an Integrity variant: {other:?}"),
    }
}

fn cross_filter_ir(authored: semstrait_model::CrossFilter) -> IrCrossFilter {
    match authored {
        semstrait_model::CrossFilter::None => IrCrossFilter::None,
        semstrait_model::CrossFilter::Left => IrCrossFilter::Left,
        semstrait_model::CrossFilter::Right => IrCrossFilter::Right,
        semstrait_model::CrossFilter::Both => IrCrossFilter::Both,
        other => unreachable!("semstrait-model added a CrossFilter variant: {other:?}"),
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Constituent<'a> {
    pub(super) name: &'a DataKindName,
    pub(super) id: &'a EntityId,
    pub(super) alias: &'a str,
}

#[derive(Clone, Copy)]
pub(super) struct JoinsetInputs<'a> {
    pub(super) constituents: &'a [Constituent<'a>],
    pub(super) interface: &'a semstrait_model::SemanticInterface,
    pub(super) definitions: &'a [SemanticDefinition],
    pub(super) keysets: &'a super::keysets::KeysetPlan,
    pub(super) serving: Serving<'a>,
}

pub(super) fn relationships_in_scope<'m>(
    root: impl IntoIterator<Item = &'m Relationship>,
    local: &'m [Relationship],
) -> Vec<&'m Relationship> {
    let shadowed: BTreeSet<&str> = local.iter().map(|r| r.name.as_str()).collect();
    let mut in_scope: Vec<&'m Relationship> = local.iter().collect();
    let mut inherited: BTreeMap<&'m str, &'m Relationship> = BTreeMap::new();
    for relationship in root {
        if !shadowed.contains(relationship.name.as_str()) {
            inherited.insert(relationship.name.as_str(), relationship);
        }
    }
    in_scope.extend(inherited.into_values());
    in_scope
}

pub(super) fn resolve_joinset(
    applied: &[&Relationship],
    operands: (&EntityId, &EntityId),
    owner: &EntityId,
    resolved: JoinsetInputs<'_>,
    pool: &mut ManifestExpressions,
) -> Result<RoleDefinition, CompileError> {
    let JoinsetInputs {
        constituents,
        interface,
        definitions,
        keysets,
        serving,
    } = resolved;
    let (Some(causing), Some(planning)) = (applied.first(), planning_relationship(applied)) else {
        return Err(CompileError::StructuralViolation {
            site: crate::error::ReferenceSite::SemanticSource(operands.0.clone()),
            rule: CAUSING_RELATIONSHIP_RULE,
        });
    };
    for relationship in applied {
        resolve_tuple(relationship)?;
    }
    let tuple = resolve_tuple(planning)?;

    let semantics = interface_index(interface, definitions);

    let mut conjuncts = Vec::with_capacity(applied.len());
    let mut relationship_filters = Vec::new();
    for relationship in applied {
        let from = resolve_endpoint(
            relationship,
            &relationship.from,
            RelationshipSide::From,
            constituents,
        )?;
        let to = resolve_endpoint(
            relationship,
            &relationship.to,
            RelationshipSide::To,
            constituents,
        )?;
        let subject = Subject::Relationship(relationship);
        let terms = ConstituentTerms { subject, serving };
        let basis = resolve_basis(
            relationship,
            &semantics,
            &semantics,
            keysets.guaranteed(from.id),
            keysets.guaranteed(to.id),
        )?;
        if let (JoinKeySpec::Declared(_), Basis::Keys(keys)) = (&relationship.on, &basis) {
            check_declared_basis_mapped(relationship, keys, [from, to], serving)?;
        }
        match basis {
            Basis::Cross => {}
            Basis::Keys(keys) => {
                conjuncts.push(equi_condition(subject, &keys, from, to, &terms)?);
            }
        }
        if let Some(filter) = relationship.filter.as_ref() {
            relationship_filters.push(lower_relationship_filter(
                relationship,
                filter,
                from,
                to,
                &semantics,
                &terms,
            )?);
        }
    }

    let on_expr_id = conjoin(conjuncts)
        .map(|condition| emit_composition_attached(condition, owner, ExprRoot::JoinsetOn, pool))
        .transpose()?;
    let filter_expr_id = conjoin(relationship_filters)
        .map(|filter| emit_composition_attached(filter, owner, ExprRoot::JoinsetFilter, pool))
        .transpose()?;

    let join_type = match on_expr_id {
        None => JoinsetJoinType::Cross,
        Some(_) => JoinsetJoinType::Join(tuple.join_type),
    };
    let straight = is_straight(applied, operands, constituents);
    Ok(RoleDefinition::Joinset {
        from: operands.0.clone(),
        to: operands.1.clone(),
        on_expr_id,
        filter_expr_id,
        join_type,
        cardinality: tuple.cardinality,
        integrity: integrity(tuple.integrity),
        cross_filter: cross_filter_ir(tuple.cross_filter),
        description: straight.then(|| causing.description.clone()).flatten(),
        ai_context: straight
            .then(|| causing.ai_context.clone())
            .flatten()
            .map(Into::into),
    })
}

fn is_straight(
    applied: &[&Relationship],
    operands: (&EntityId, &EntityId),
    constituents: &[Constituent<'_>],
) -> bool {
    let is_constituent =
        |id: &EntityId| constituents.iter().any(|constituent| constituent.id == id);
    applied.len() == 1 && is_constituent(operands.0) && is_constituent(operands.1)
}

pub const CAUSING_RELATIONSHIP_RULE: &str =
    "every enumerated unit carries the relationship that caused it, and it is the first of the \
     relationships its condition is built from";

pub const SEALED_OPERAND_RULE: &str = "every operand is sealed before its composer";

pub(super) fn planning_relationship<'r>(closing: &[&'r Relationship]) -> Option<&'r Relationship> {
    (closing.iter())
        .filter(|relationship| !matches!(relationship.on, JoinKeySpec::Cross))
        .min_by_key(|relationship| relationship.name.as_str())
        .or(closing.first())
        .copied()
}

pub(super) fn resolve_endpoint<'a>(
    relationship: &Relationship,
    endpoint: &DataKindName,
    side: RelationshipSide,
    constituents: &'a [Constituent<'a>],
) -> Result<Constituent<'a>, CompileError> {
    constituents
        .iter()
        .find(|constituent| constituent.name == endpoint)
        .copied()
        .ok_or_else(|| {
            let present: Vec<&str> = constituents
                .iter()
                .map(|constituent| constituent.name.as_str())
                .collect();
            resolution_failure(
                relationship,
                format!(
                    "its `{side}` endpoint `{endpoint}` names no constituent of the Joinset that \
                     holds it; the constituents are {present:?}"
                ),
            )
        })
}

pub(super) struct ResolvedTuple {
    pub(super) cardinality: IrCardinality,
    pub(super) integrity: Integrity,
    pub(super) cross_filter: CrossFilter,
    pub(super) join_type: JoinType,
}

pub(super) fn resolve_tuple(relationship: &Relationship) -> Result<ResolvedTuple, CompileError> {
    guard_authored_cardinality(relationship)?;
    let (cardinality, optional, cross_filter) = resolve_defaults(
        relationship.cardinality,
        relationship.integrity,
        relationship.optional,
        relationship.cross_filter,
    )
    .map_err(|reason| resolution_failure(relationship, reason))?;
    let cardinality = cardinality_to_ir(cardinality)
        .map_err(|reason| resolution_failure(relationship, reason))?;
    let join_type =
        derive_join_type(optional).map_err(|reason| resolution_failure(relationship, reason))?;
    guard_cardinality_pair(
        &relationship.name,
        cardinality,
        cross_filter_ir(cross_filter),
    )?;
    Ok(ResolvedTuple {
        cardinality,
        integrity: relationship.integrity,
        cross_filter,
        join_type,
    })
}

fn guard_authored_cardinality(relationship: &Relationship) -> Result<(), CompileError> {
    if relationship.cardinality.is_none() && !matches!(relationship.on, JoinKeySpec::Cross) {
        return Err(CompileError::RelationshipMissingCardinality {
            relationship: relationship.name.clone(),
        });
    }
    Ok(())
}

pub fn guard_cardinality_pair(
    subject: &str,
    cardinality: IrCardinality,
    cross_filter: IrCrossFilter,
) -> Result<(), CompileError> {
    use semstrait_ir::CrossFilter;
    if matches!(cardinality, IrCardinality::ManyToMany)
        && matches!(cross_filter, CrossFilter::Left | CrossFilter::Right)
    {
        return Err(CompileError::RelationshipManyToManyCrossFilterDirectional {
            relationship: subject.to_owned(),
            cardinality,
            cross_filter,
        });
    }
    Ok(())
}

fn resolve_defaults(
    cardinality: Option<ModelCardinality>,
    integrity: Integrity,
    authored_optional: Option<Optional>,
    authored_cross_filter: Option<CrossFilter>,
) -> Result<(ModelCardinality, Optional, CrossFilter), String> {
    match cardinality {
        None => Ok((
            ModelCardinality::ManyToMany,
            authored_optional.unwrap_or(Optional::None),
            authored_cross_filter.unwrap_or(CrossFilter::None),
        )),
        Some(ModelCardinality::ManyToOne) => {
            let optional = authored_optional.unwrap_or(match integrity {
                Integrity::None => Optional::Left,
                _ => Optional::None,
            });
            let cross_filter = authored_cross_filter.unwrap_or(CrossFilter::Left);
            Ok((ModelCardinality::ManyToOne, optional, cross_filter))
        }
        Some(ModelCardinality::OneToMany) => {
            let optional = authored_optional.unwrap_or(match integrity {
                Integrity::Enforced => Optional::None,
                _ => Optional::Left,
            });
            let cross_filter = authored_cross_filter.unwrap_or(CrossFilter::Right);
            Ok((ModelCardinality::OneToMany, optional, cross_filter))
        }
        Some(symmetric @ (ModelCardinality::OneToOne | ModelCardinality::ManyToMany)) => {
            let optional = authored_optional.ok_or_else(|| required_symmetric("optional"))?;
            let cross_filter =
                authored_cross_filter.ok_or_else(|| required_symmetric("cross_filter"))?;
            Ok((symmetric, optional, cross_filter))
        }
        #[allow(unreachable_patterns)]
        _ => Err("its authored cardinality has no row in the defaults matrix".to_owned()),
    }
}

fn required_symmetric(field: &str) -> String {
    format!(
        "`{field}` is required under a OneToOne / ManyToMany cardinality and no default is \
         defensible, so nothing here fabricates one"
    )
}

fn cardinality_to_ir(cardinality: ModelCardinality) -> Result<IrCardinality, String> {
    #[allow(unreachable_patterns)]
    Ok(match cardinality {
        ModelCardinality::OneToOne => IrCardinality::OneToOne,
        ModelCardinality::OneToMany => IrCardinality::OneToMany,
        ModelCardinality::ManyToOne => IrCardinality::ManyToOne,
        ModelCardinality::ManyToMany => IrCardinality::ManyToMany,
        _ => return Err("its cardinality has no IR planning counterpart".to_owned()),
    })
}

fn derive_join_type(optional: Optional) -> Result<JoinType, String> {
    #[allow(unreachable_patterns)]
    Ok(match optional {
        Optional::None => JoinType::Inner,
        Optional::Left => JoinType::Left,
        Optional::Right => JoinType::Right,
        Optional::Both => JoinType::Full,
        _ => {
            return Err(
                "its resolved `optional` has no JoinType counterpart, so the \
                        optional→join_type bijection no longer holds and `optional` would have \
                        to be persisted again"
                    .to_owned(),
            )
        }
    })
}

pub(super) enum Basis<'d> {
    Cross,
    Keys(Vec<&'d SemanticDefinition>),
}

pub(super) fn resolve_basis<'d>(
    relationship: &Relationship,
    from_semantics: &SemanticsIndex<'d>,
    to_semantics: &SemanticsIndex<'d>,
    from_keysets: &[GuaranteedKeyset],
    to_keysets: &[GuaranteedKeyset],
) -> Result<Basis<'d>, CompileError> {
    match &relationship.on {
        JoinKeySpec::Cross => Ok(Basis::Cross),
        JoinKeySpec::Declared(names) => Ok(Basis::Keys(resolve_declared_basis(
            relationship,
            names,
            from_semantics,
            to_semantics,
        )?)),
        JoinKeySpec::Auto => Ok(Basis::Keys(derive_basis(
            Subject::Relationship(relationship),
            from_semantics,
            to_semantics,
            from_keysets,
            to_keysets,
        )?)),
        _ => Err(resolution_failure(
            relationship,
            "its authored join basis is a form this build has no resolution for".to_owned(),
        )),
    }
}

fn resolve_declared_basis<'d>(
    relationship: &Relationship,
    names: &[SemanticsName],
    from_semantics: &SemanticsIndex<'d>,
    to_semantics: &SemanticsIndex<'d>,
) -> Result<Vec<&'d SemanticDefinition>, CompileError> {
    let mut seen: BTreeSet<&SemanticsName> = BTreeSet::new();
    let mut keys = Vec::with_capacity(names.len());
    for name in names {
        if !seen.insert(name) {
            return Err(CompileError::RelationshipJoinKeyDuplicated {
                relationship: relationship.name.clone(),
                key: name.as_str().to_owned(),
            });
        }
        let from = key_on_side(relationship, name, from_semantics, RelationshipSide::From)?;
        let to = key_on_side(relationship, name, to_semantics, RelationshipSide::To)?;
        if from.id != to.id {
            return Err(CompileError::RelationshipJoinKeyIdMismatch {
                relationship: relationship.name.clone(),
                key: name.as_str().to_owned(),
                from_id: from.id.clone(),
                to_id: to.id.clone(),
            });
        }
        keys.push(from);
    }
    Ok(keys)
}

fn check_declared_basis_mapped(
    relationship: &Relationship,
    keys: &[&SemanticDefinition],
    sides: [Constituent<'_>; 2],
    serving: Serving<'_>,
) -> Result<(), CompileError> {
    let mut basis = BTreeSet::new();
    for key in keys {
        basis.insert(&key.id);
        if let Some(side) = sides
            .iter()
            .find(|side| !serving.maps_whole(side.id, &basis))
        {
            let semantic_source =
                serving
                    .name(side.id)
                    .ok_or_else(|| CompileError::StructuralViolation {
                        site: ReferenceSite::SemanticSource(side.id.clone()),
                        rule: SEALED_OPERAND_RULE,
                    })?;
            return Err(CompileError::JoinKeyNotMapped {
                relationship: relationship.name.clone(),
                key: key.name.as_str().to_owned(),
                semantic_source: semantic_source.as_str().to_owned(),
            });
        }
    }
    Ok(())
}

fn key_on_side<'a>(
    relationship: &Relationship,
    key: &SemanticsName,
    semantics: &SemanticsIndex<'a>,
    side: RelationshipSide,
) -> Result<&'a SemanticDefinition, CompileError> {
    let definition =
        semantics
            .get(key)
            .ok_or_else(|| CompileError::RelationshipJoinKeyUnknownOnSide {
                relationship: relationship.name.clone(),
                key: key.as_str().to_owned(),
                side,
            })?;
    if definition.role() != SemanticRole::Dimension {
        return Err(CompileError::RelationshipJoinKeyNotADimension {
            relationship: relationship.name.clone(),
            key: key.as_str().to_owned(),
            role: definition.role(),
        });
    }
    Ok(definition)
}

fn derive_basis<'d>(
    subject: Subject<'_>,
    from_semantics: &SemanticsIndex<'d>,
    to_semantics: &SemanticsIndex<'d>,
    from_keysets: &[GuaranteedKeyset],
    to_keysets: &[GuaranteedKeyset],
) -> Result<Vec<&'d SemanticDefinition>, CompileError> {
    let mut qualifying: Vec<(&KeysetName, Vec<&'d SemanticDefinition>)> = from_keysets
        .iter()
        .chain(to_keysets)
        .filter(|keyset| !keyset.members.is_empty())
        .filter_map(|keyset| {
            let keys = served_definitions(&keyset.members, from_semantics)?;
            served_definitions(&keyset.members, to_semantics)?;
            Some((&keyset.name, keys))
        })
        .collect();
    qualifying.sort_by(|(left, left_keys), (right, right_keys)| {
        (left.cmp(right)).then_with(|| {
            (left_keys.iter().map(|key| &key.name)).cmp(right_keys.iter().map(|key| &key.name))
        })
    });
    let Some((_, keys)) = qualifying.into_iter().next() else {
        return Err(subject.failure(
            "it found no guaranteed keyset both sides serve, so it has no basis to derive; a \
             cartesian product must be written as `on: cross`, which is a different claim"
                .to_owned(),
        ));
    };
    Ok(keys)
}

pub(super) fn resolve_cross_grain(
    unit: &DataKindName,
    owner: &EntityId,
    finer: Constituent<'_>,
    coarser: Constituent<'_>,
    resolved: JoinsetInputs<'_>,
    pool: &mut ManifestExpressions,
) -> Result<RoleDefinition, CompileError> {
    let JoinsetInputs {
        interface,
        definitions,
        keysets,
        serving,
        ..
    } = resolved;
    let subject = Subject::CrossGrain(unit);
    let semantics = interface_index(interface, definitions);
    let keys = derive_basis(
        subject,
        &semantics,
        &semantics,
        keysets.guaranteed(finer.id),
        keysets.guaranteed(coarser.id),
    )?;
    let terms = ConstituentTerms { subject, serving };
    let condition = equi_condition(subject, &keys, finer, coarser, &terms)?;
    let on_expr_id = emit_composition_attached(condition, owner, ExprRoot::JoinsetOn, pool)?;

    let cardinality = IrCardinality::ManyToMany;
    let cross_filter = CrossFilter::None;
    guard_cardinality_pair(unit.as_str(), cardinality, cross_filter_ir(cross_filter))?;
    Ok(RoleDefinition::Joinset {
        from: finer.id.clone(),
        to: coarser.id.clone(),
        on_expr_id: Some(on_expr_id),
        filter_expr_id: None,
        join_type: JoinsetJoinType::Join(JoinType::Left),
        cardinality,
        integrity: integrity(Integrity::None),
        cross_filter: cross_filter_ir(cross_filter),
        description: None,
        ai_context: None,
    })
}

fn served_definitions<'d>(
    members: &BTreeSet<EntityId>,
    semantics: &SemanticsIndex<'d>,
) -> Option<Vec<&'d SemanticDefinition>> {
    let mut definitions: Vec<&'d SemanticDefinition> = Vec::with_capacity(members.len());
    for member in members {
        let definition = (semantics.names())
            .filter_map(|name| semantics.get(name))
            .find(|definition| &definition.id == member)?;
        definitions.push(definition);
    }
    definitions.sort_by(|left, right| left.name.cmp(&right.name));
    Some(definitions)
}

struct ConstituentTerms<'a> {
    subject: Subject<'a>,
    serving: Serving<'a>,
}

impl ConstituentTerms<'_> {
    fn term(
        &self,
        definition: &SemanticDefinition,
        constituent: Constituent<'_>,
    ) -> Result<Option<Expr<Link>>, CompileError> {
        (self.serving)
            .serves(constituent.id, &definition.id)
            .then(|| {
                qualified_column(
                    constituent.alias,
                    definition.name.as_str(),
                    self.subject.name(),
                )
                .map(Expr::Column)
            })
            .transpose()
    }
}

fn equi_condition(
    subject: Subject<'_>,
    keys: &[&SemanticDefinition],
    from: Constituent<'_>,
    to: Constituent<'_>,
    terms: &ConstituentTerms<'_>,
) -> Result<Expr<Link>, CompileError> {
    let mut conjuncts = Vec::with_capacity(keys.len());
    for definition in keys {
        let side = |constituent: Constituent<'_>| {
            terms
                .term(definition, constituent)?
                .ok_or_else(|| unserved_key(subject, &definition.name, constituent))
        };
        conjuncts.push(side(from)?.eq(side(to)?));
    }
    conjoin(conjuncts).ok_or_else(|| {
        subject.failure(
            "its basis resolved to no equi-terms, so there is no condition to persist".to_owned(),
        )
    })
}

fn conjoin(conjuncts: Vec<Expr<Link>>) -> Option<Expr<Link>> {
    let mut terms = conjuncts.into_iter();
    let first = terms.next()?;
    Some(terms.fold(first, |left, right| left & right))
}

fn unserved_key(
    subject: Subject<'_>,
    key: &SemanticsName,
    constituent: Constituent<'_>,
) -> CompileError {
    subject.failure(format!(
        "its basis names `{key}`, which `{}` does not serve, so there is no expression for \
             that side of the equality",
        constituent.name
    ))
}

fn lower_relationship_filter(
    relationship: &Relationship,
    filter: &UnresolvedSemanticExpr,
    from: Constituent<'_>,
    to: Constituent<'_>,
    semantics: &SemanticsIndex<'_>,
    terms: &ConstituentTerms<'_>,
) -> Result<Expr<Link>, CompileError> {
    let tree: LoweredExpr = map_nodes(
        filter,
        &mut |node| match node {
            Expr::Leaf(leaf) => Some(relationship_filter_leaf(
                relationship,
                leaf,
                from,
                to,
                semantics,
                terms,
            )),
            Expr::Column(column) => Some(Err(resolution_failure(
                relationship,
                format!(
                    "its `filter:` names the physical column `{}` directly; a relationship \
                     filter spans two constituents, so a column with no side has no schema to \
                     resolve against — name the semantics instead",
                    column.name()
                ),
            ))),
            Expr::Function(Call::Window(_)) => Some(Err(refused_window(&relationship.name))),
            _ => None,
        },
        &|error| {
            refused_rebuild(&relationship.name, error, |error| {
                resolution_failure(
                    relationship,
                    format!(
                        "its `filter:` holds an expression form this build cannot lower \
                         ({error}), and dropping the subtree would persist a relationship \
                         filter weaker than the one authored"
                    ),
                )
            })
        },
    )?;
    Ok(Lowered::new(tree, [])
        .typed_condition(&ColumnTypes::default(), &relationship.name)?
        .expr)
}

fn relationship_filter_leaf(
    relationship: &Relationship,
    leaf: &SemanticLeaf,
    from: Constituent<'_>,
    to: Constituent<'_>,
    semantics: &SemanticsIndex<'_>,
    terms: &ConstituentTerms<'_>,
) -> Result<LoweredExpr, CompileError> {
    let named = match leaf {
        SemanticLeaf::Untyped(literal) => {
            return Ok(Expr::Leaf(LoweredLeaf::Untyped(literal.clone())))
        }
        SemanticLeaf::Field(name)
        | SemanticLeaf::Dimension(name, _)
        | SemanticLeaf::Measure(name, _)
        | SemanticLeaf::Metric(name, _) => name,
        _ => {
            return Err(resolution_failure(
                relationship,
                "its `filter:` holds a leaf form this build has no resolution for".to_owned(),
            ))
        }
    };
    let name = SemanticsName::from(named.name());
    let definition = semantics.get(&name).ok_or_else(|| {
        resolution_failure(
            relationship,
            format!("its `filter:` names `{name}`, which neither side exposes"),
        )
    })?;
    if matches!(
        definition.role(),
        SemanticRole::Measure | SemanticRole::Metric
    ) {
        return Err(CompileError::MeasureOrMetricInRelationshipFilter {
            relationship: relationship.name.clone(),
            semantics: name.as_str().to_owned(),
            role: definition.role(),
        });
    }
    for constituent in [from, to] {
        if let Some(tree) = terms.term(definition, constituent)? {
            return Ok(Expr::Leaf(LoweredLeaf::Splice {
                tree,
                data_type: definition.data_type.clone().unwrap_or(DataType::Unknown),
            }));
        }
    }
    Err(resolution_failure(
        relationship,
        format!("its `filter:` names `{name}`, which neither side serves"),
    ))
}

fn resolution_failure(relationship: &Relationship, reason: String) -> CompileError {
    CompileError::RelationshipResolution {
        relationship: relationship.name.clone(),
        reason,
    }
}

#[derive(Clone, Copy)]
enum Subject<'a> {
    Relationship(&'a Relationship),
    CrossGrain(&'a DataKindName),
}

impl<'a> Subject<'a> {
    fn name(self) -> &'a str {
        match self {
            Self::Relationship(relationship) => &relationship.name,
            Self::CrossGrain(unit) => unit.as_str(),
        }
    }

    fn failure(self, reason: String) -> CompileError {
        match self {
            Self::Relationship(relationship) => resolution_failure(relationship, reason),
            Self::CrossGrain(unit) => CompileError::CrossGrainResolution {
                unit: unit.as_str().to_owned(),
                reason,
            },
        }
    }
}

pub(crate) fn unserved_condition_leaves<'d>(
    semantic_sources: &BTreeMap<EntityId, SemanticSource>,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
    definitions: impl IntoIterator<Item = &'d SemanticDefinition>,
    pool: &ManifestExpressions,
) -> Vec<CompileError> {
    let names: BTreeMap<&EntityId, String> = (definitions.into_iter())
        .map(|definition| (&definition.id, folded(definition.name.as_str())))
        .collect();
    let by_alias: BTreeMap<&str, &SemanticSource> = (semantic_sources.values())
        .map(|entry| (entry.alias.as_str(), entry))
        .collect();
    let serving = Serving::new(semantic_sources, signatures);
    let serves = |constituent: &SemanticSource, name: &str| {
        let name = folded(name);
        (serving.served(&constituent.id).into_iter()).any(|id| names.get(id) == Some(&name))
    };
    (semantic_sources.values())
        .flat_map(condition_handles)
        .flat_map(|root| pool.columns(&root))
        .filter_map(|(expression, column)| {
            let [alias, name] = column.parts() else {
                return None;
            };
            let constituent = by_alias.get(alias.as_str())?;
            (!serves(constituent, name)).then(|| CompileError::UnservedConditionLeaf {
                expression,
                column: column.clone(),
                semantic_source: constituent.id.clone(),
            })
        })
        .collect()
}

fn folded(value: &str) -> String {
    semstrait_ir::Name::folded(value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::manifest::bitmap::SemanticBitmask;
    use crate::manifest::expression::PhysicalExprId;
    use crate::manifest::semantic_source::Selector;
    use crate::manifest::source_alias::SourceAlias;
    use semstrait_ir::PhysicalExpr;
    use semstrait_sql::expr::NamedReference;

    #[test]
    fn a_many_to_many_pair_refuses_only_a_directional_cross_filter() {
        use semstrait_ir::CrossFilter;

        for direction in [CrossFilter::Left, CrossFilter::Right] {
            let error = guard_cardinality_pair("r", IrCardinality::ManyToMany, direction)
                .expect_err("a directional filter is refused");
            match error {
                CompileError::RelationshipManyToManyCrossFilterDirectional {
                    cardinality,
                    cross_filter,
                    ..
                } => {
                    assert_eq!(cardinality, IrCardinality::ManyToMany);
                    assert_eq!(cross_filter, direction);
                }
                other => panic!("expected the dedicated variant, got {other:?}"),
            }
        }
        for legal in [CrossFilter::None, CrossFilter::Both] {
            assert!(guard_cardinality_pair("r", IrCardinality::ManyToMany, legal).is_ok());
        }
        assert!(
            guard_cardinality_pair("r", IrCardinality::ManyToOne, CrossFilter::Left).is_ok(),
            "the rule is about ManyToMany, not about direction"
        );
    }

    #[test]
    fn the_defaults_matrix_fills_only_an_absent_value() {
        assert_eq!(
            resolve_defaults(
                Some(ModelCardinality::ManyToOne),
                Integrity::None,
                None,
                None
            )
            .expect("a row exists"),
            (
                ModelCardinality::ManyToOne,
                Optional::Left,
                CrossFilter::Left
            )
        );
        assert_eq!(
            resolve_defaults(
                Some(ModelCardinality::OneToMany),
                Integrity::Enforced,
                None,
                None
            )
            .expect("a row exists"),
            (
                ModelCardinality::OneToMany,
                Optional::None,
                CrossFilter::Right
            )
        );
        assert_eq!(
            resolve_defaults(
                Some(ModelCardinality::ManyToOne),
                Integrity::Assumed,
                Some(Optional::Both),
                Some(CrossFilter::Both)
            )
            .expect("a row exists"),
            (
                ModelCardinality::ManyToOne,
                Optional::Both,
                CrossFilter::Both
            )
        );
        assert_eq!(
            resolve_defaults(None, Integrity::Assumed, None, None).expect("the absent row exists"),
            (
                ModelCardinality::ManyToMany,
                Optional::None,
                CrossFilter::None
            )
        );
        assert!(resolve_defaults(
            Some(ModelCardinality::OneToOne),
            Integrity::Assumed,
            None,
            Some(CrossFilter::Both)
        )
        .is_err());
    }

    fn leaf_entry(id: &str, alias: &str) -> SemanticSource {
        SemanticSource::new(
            id,
            DataKindName::from(id),
            SourceAlias::from(alias),
            false,
            SemanticBitmask::empty(),
            None,
            RoleDefinition::dataset("src", Selector::Prefix),
            Vec::new(),
        )
        .with_semantic_signature_ids(vec![format!("{id}.customer_id")])
    }

    fn join_entry(id: &str, on_expr_id: PhysicalExprId) -> SemanticSource {
        SemanticSource::new(
            id,
            DataKindName::from(id),
            SourceAlias::from("j"),
            true,
            SemanticBitmask::empty(),
            None,
            RoleDefinition::Joinset {
                from: "dk.a".to_owned(),
                to: "dk.b".to_owned(),
                on_expr_id: Some(on_expr_id),
                filter_expr_id: None,
                join_type: JoinsetJoinType::Join(JoinType::Inner),
                cardinality: IrCardinality::ManyToOne,
                integrity: integrity(Integrity::Assumed),
                cross_filter: cross_filter_ir(CrossFilter::Left),
                description: None,
                ai_context: None,
            },
            Vec::new(),
        )
    }

    fn qualified(alias: &str, column: &str) -> PhysicalExpr {
        Expr::Column(NamedReference::qualified(alias, column).expect("an alias and a name"))
    }

    fn unserved(on: &PhysicalExpr) -> (PhysicalExprId, Vec<CompileError>) {
        let mut pool = ManifestExpressions::default();
        let on_expr_id = pool
            .emit(on, "dk.j", ExprRoot::JoinsetOn, [])
            .expect("emit");
        let customer_id = definition("customer_id", SemanticRole::Dimension);
        let signatures: BTreeMap<EntityId, SemanticSignature> = ["dk.a", "dk.b"]
            .into_iter()
            .map(|id| {
                let signature = SemanticSignature::new(
                    format!("{id}.customer_id"),
                    customer_id.id.clone(),
                    on_expr_id,
                );
                (signature.id.clone(), signature)
            })
            .collect();
        let semantic_sources: BTreeMap<EntityId, SemanticSource> = [
            leaf_entry("dk.a", "a_orders"),
            leaf_entry("dk.b", "b_cust"),
            join_entry("dk.j", on_expr_id),
        ]
        .into_iter()
        .map(|entry| (entry.id.clone(), entry))
        .collect();
        let faults =
            unserved_condition_leaves(&semantic_sources, &signatures, [&customer_id], &pool);
        (on_expr_id, faults)
    }

    #[test]
    fn a_condition_leaf_naming_a_served_semantics_passes_case_insensitively() {
        let (_, faults) =
            unserved(&qualified("a_orders", "customer_id").eq(qualified("b_cust", "CUSTOMER_ID")));
        assert!(faults.is_empty(), "both leaves are served: {faults:?}");
    }

    #[test]
    fn a_condition_leaf_naming_an_unserved_semantics_is_refused() {
        let (on_expr_id, faults) =
            unserved(&qualified("a_orders", "nowhere").eq(qualified("elsewhere", "nowhere")));
        let [CompileError::UnservedConditionLeaf {
            expression,
            column,
            semantic_source,
        }] = faults.as_slice()
        else {
            panic!("expected one UnservedConditionLeaf, got {faults:?}");
        };
        assert_eq!(*expression, on_expr_id);
        assert_eq!(column.parts(), ["a_orders", "nowhere"]);
        assert_eq!(semantic_source, "dk.a");
    }

    fn definition(name: &str, role: SemanticRole) -> SemanticDefinition {
        SemanticDefinition::new(
            format!("sem.{name}"),
            SemanticsName::from(name),
            None,
            crate::manifest::semantic_definition::SemanticRoleDefinition::plain(role),
        )
    }

    fn declared(keys: &[&str]) -> Relationship {
        Relationship::builder()
            .name("r")
            .from("a")
            .to("b")
            .on(keys.iter().copied())
            .many_to_one()
            .build()
            .expect("builds")
    }

    #[test]
    fn all_four_join_basis_rules_fire_on_a_declared_basis() {
        let dimension = definition("customer_id", SemanticRole::Dimension);
        let measure = definition("revenue", SemanticRole::Measure);
        let both = [dimension.clone(), measure.clone()];
        let index = super::super::definitions::semantics_index_from_defs(&both);
        let empty = SemanticsIndex::new();

        for (from, to, side) in [
            (&empty, &index, RelationshipSide::From),
            (&index, &empty, RelationshipSide::To),
        ] {
            match resolve_declared_basis(
                &declared(&["customer_id"]),
                &["customer_id".into()],
                from,
                to,
            )
            .expect_err("a key the side does not expose is refused")
            {
                CompileError::RelationshipJoinKeyUnknownOnSide { key, side: got, .. } => {
                    assert_eq!(key, "customer_id");
                    assert_eq!(got, side);
                }
                other => panic!("expected RelationshipJoinKeyUnknownOnSide, got {other:?}"),
            }
        }

        match resolve_declared_basis(&declared(&["revenue"]), &["revenue".into()], &index, &index)
            .expect_err("a Measure is not a join key")
        {
            CompileError::RelationshipJoinKeyNotADimension { key, role, .. } => {
                assert_eq!(key, "revenue");
                assert_eq!(role, SemanticRole::Measure);
            }
            other => panic!("expected RelationshipJoinKeyNotADimension, got {other:?}"),
        }

        let renamed = [SemanticDefinition::new(
            "sem.other",
            SemanticsName::from("customer_id"),
            None,
            crate::manifest::semantic_definition::SemanticRoleDefinition::plain(
                SemanticRole::Dimension,
            ),
        )];
        let other_side = super::super::definitions::semantics_index_from_defs(&renamed);
        match resolve_declared_basis(
            &declared(&["customer_id"]),
            &["customer_id".into()],
            &index,
            &other_side,
        )
        .expect_err("one name resolving to two Semantics is refused")
        {
            CompileError::RelationshipJoinKeyIdMismatch {
                key,
                from_id,
                to_id,
                ..
            } => {
                assert_eq!(key, "customer_id");
                assert_eq!(from_id, "sem.customer_id");
                assert_eq!(to_id, "sem.other");
            }
            other => panic!("expected RelationshipJoinKeyIdMismatch, got {other:?}"),
        }

        match resolve_declared_basis(
            &declared(&["customer_id"]),
            &["customer_id".into(), "customer_id".into()],
            &index,
            &index,
        )
        .expect_err("a repeated key is refused")
        {
            CompileError::RelationshipJoinKeyDuplicated { key, .. } => {
                assert_eq!(key, "customer_id");
            }
            other => panic!("expected RelationshipJoinKeyDuplicated, got {other:?}"),
        }

        let keys = resolve_declared_basis(
            &declared(&["customer_id"]),
            &["customer_id".into()],
            &index,
            &index,
        )
        .expect("a Dimension both sides expose resolves");
        assert_eq!(
            keys.iter().map(|key| &key.name).collect::<Vec<_>>(),
            [&SemanticsName::from("customer_id")]
        );
    }

    #[test]
    fn a_local_declaration_shadows_the_root_one_it_names() {
        let declared = |name: &str, from: &str| {
            Relationship::builder()
                .name(name)
                .from(from)
                .to("customers")
                .on(["customer_id"])
                .many_to_one()
                .build()
                .expect("builds")
        };
        let root: BTreeMap<String, Relationship> = [
            ("r1".to_owned(), declared("r1", "root_orders")),
            ("r2".to_owned(), declared("r2", "root_returns")),
        ]
        .into_iter()
        .collect();
        let local = vec![declared("r1", "local_orders")];
        let in_scope = relationships_in_scope(root.values(), &local);
        let seen: Vec<(&str, &str)> = in_scope
            .iter()
            .map(|r| (r.name.as_str(), r.from.as_str()))
            .collect();
        assert_eq!(
            seen,
            vec![("r1", "local_orders"), ("r2", "root_returns")],
            "the shadow wins for its own name and nothing else moves"
        );
    }
}
