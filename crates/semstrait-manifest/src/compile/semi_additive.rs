use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::SemanticsName;
use semstrait_model::SemanticModel;
use semstrait_sql::expr::Expr;

use super::assemble::{self, SourcePlan};
use super::definitions::{self, semantics_index_from_defs};
use crate::error::CompileError;
use crate::identity::mint_signature_id;
use crate::manifest::expression::{LinkStep, ManifestExpressions, PhysicalExprId};
use crate::manifest::semantic_definition::{DimensionType, SemanticDefinition};
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::{RoleDefinition, SemanticSource};
use crate::manifest::serving::Serving;

pub(super) fn check_semi_additive_sources(
    model: &SemanticModel,
    plan: &[SourcePlan<'_>],
    definitions: &[SemanticDefinition],
    sources: &BTreeMap<EntityId, SemanticSource>,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
    pool: &ManifestExpressions,
) -> Result<(), CompileError> {
    let authored = definitions::authored_semi_additivity(model);
    let plan_by_id: BTreeMap<&EntityId, &SourcePlan<'_>> =
        plan.iter().map(|unit| (&unit.id, unit)).collect();
    let roster = semantics_index_from_defs(definitions);
    let serving = Serving::new(sources, signatures);

    for unit in assemble::seal_order(plan) {
        let Some(source) = (sources.get(&unit.id))
            .filter(|source| !matches!(source.role_definition, RoleDefinition::Grainset { .. }))
        else {
            continue;
        };
        for definition in definitions {
            let Some(semi) = authored.get(&definition.name) else {
                continue;
            };
            if !serving.serves(&source.id, &definition.id) {
                continue;
            }

            let axis_defs = definitions::semi_axis_definitions(
                |name| roster.get(name),
                &definition.name,
                &semi.axes,
            )?;

            for read in datasets_read(
                source,
                &definition.id,
                sources,
                signatures,
                pool,
                Depth::Top,
            ) {
                let no_default = || CompileError::SemiAdditiveByNoDefault {
                    semantics: definition.name.to_string(),
                    semantic_source: source.name.to_string(),
                };
                let by_def: &SemanticDefinition = match &semi.by {
                    Some(by) => {
                        roster
                            .get(by)
                            .ok_or_else(|| CompileError::SemiAdditiveByInvalid {
                                semantics: definition.name.to_string(),
                                by: by.to_string(),
                                semantic_source: source.name.to_string(),
                            })?
                    }
                    None => {
                        let time_axes: BTreeSet<&EntityId> = read
                            .iter()
                            .filter_map(|id| sources.get(id))
                            .filter_map(source_time_axis)
                            .collect();
                        let mut axes = time_axes.into_iter();
                        let axis_id = match (axes.next(), axes.next()) {
                            (Some(axis_id), None) => axis_id,
                            _ => return Err(no_default()),
                        };
                        definitions
                            .iter()
                            .find(|d| &d.id == axis_id)
                            .ok_or_else(no_default)?
                    }
                };

                if !matches!(
                    by_def.role_definition.category(),
                    Some(DimensionType::Temporal(_))
                ) || semi.axes.contains(&by_def.name)
                {
                    return Err(CompileError::SemiAdditiveByInvalid {
                        semantics: definition.name.to_string(),
                        by: by_def.name.to_string(),
                        semantic_source: source.name.to_string(),
                    });
                }

                for name_def in std::iter::once(by_def).chain(axis_defs.iter().copied()) {
                    if !read.is_empty() && !dimension_is_read(&name_def.name, &read, &plan_by_id) {
                        return Err(CompileError::SemiAdditiveDimensionNotRead {
                            semantics: definition.name.to_string(),
                            dimension: name_def.name.to_string(),
                            semantic_source: source.name.to_string(),
                        });
                    }
                    if !dimension_is_unambiguous(&name_def.id, &read, sources, serving) {
                        return Err(CompileError::SemiAdditiveDimensionAmbiguous {
                            semantics: definition.name.to_string(),
                            dimension: name_def.name.to_string(),
                            semantic_source: source.name.to_string(),
                        });
                    }
                }

                if !serving.serves(&source.id, &by_def.id) {
                    return Err(CompileError::SemiAdditiveByNotServed {
                        semantics: definition.name.to_string(),
                        by: by_def.name.to_string(),
                        semantic_source: source.name.to_string(),
                    });
                }
            }

            for axis_def in &axis_defs {
                if !serving.serves(&source.id, &axis_def.id) {
                    return Err(CompileError::SemiAdditiveAxisNotServed {
                        semantics: definition.name.to_string(),
                        axis: axis_def.name.to_string(),
                        semantic_source: source.name.to_string(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn dimension_is_read(
    name: &SemanticsName,
    read: &BTreeSet<EntityId>,
    plan_by_id: &BTreeMap<&EntityId, &SourcePlan<'_>>,
) -> bool {
    read.iter().any(|id| {
        plan_by_id.get(id).copied().is_some_and(|unit| {
            definitions::exposed_semantics(unit.form.interface()).contains(name)
        })
    })
}

fn dimension_is_unambiguous(
    name_id: &EntityId,
    read: &BTreeSet<EntityId>,
    sources: &BTreeMap<EntityId, SemanticSource>,
    serving: Serving<'_>,
) -> bool {
    let mapping: Vec<&EntityId> = read
        .iter()
        .filter(|id| serving.serves(id, name_id))
        .collect();
    if mapping.len() < 2 {
        return true;
    }
    mapping
        .iter()
        .filter(|id| sources.get(**id).and_then(source_time_axis) == Some(name_id))
        .count()
        == 1
}

fn source_time_axis(source: &SemanticSource) -> Option<&EntityId> {
    source.temporal.as_ref()?.temporal_type.time_axis()
}

enum Depth {
    Top,
    Below,
}

fn datasets_read(
    source: &SemanticSource,
    definition_id: &EntityId,
    sources: &BTreeMap<EntityId, SemanticSource>,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
    pool: &ManifestExpressions,
    depth: Depth,
) -> Vec<BTreeSet<EntityId>> {
    let top = matches!(depth, Depth::Top);
    let reads_a_column = |source: &SemanticSource| {
        signature_expression(source, definition_id, signatures)
            .is_some_and(|root| !column_aliases(root, pool).is_empty())
    };
    let own = |counts: bool| vec![counts.then(|| source.id.clone()).into_iter().collect()];
    let serving = Serving::new(sources, signatures);
    let serves = |source: &SemanticSource| serving.serves(&source.id, definition_id);
    let operands =
        || (source.role_definition.operands().into_iter()).filter_map(|id| sources.get(id));
    let below = |operand: &SemanticSource| {
        datasets_read(
            operand,
            definition_id,
            sources,
            signatures,
            pool,
            Depth::Below,
        )
    };
    match &source.role_definition {
        RoleDefinition::Dataset { .. } => own(top || reads_a_column(source)),
        RoleDefinition::Unionset {
            null_fill: false, ..
        } => own(top || operands().any(reads_a_column)),
        RoleDefinition::Joinset { .. } => {
            let Some(root) = signature_expression(source, definition_id, signatures) else {
                return own(false);
            };
            let aliases = column_aliases(root, pool);
            read_together(
                operands()
                    .filter(|operand| aliases.contains(operand.alias.as_str()))
                    .map(below),
            )
        }
        RoleDefinition::Unionset { .. } if !serves(source) => own(false),
        RoleDefinition::Unionset { .. } => {
            read_together(operands().filter(|branch| serves(branch)).map(below))
        }
        RoleDefinition::Grainset { .. } => {
            let children: Vec<_> = (operands().filter(|child| serves(child)))
                .flat_map(below)
                .collect();
            if children.is_empty() {
                own(false)
            } else {
                children
            }
        }
    }
}

fn read_together(
    operands: impl Iterator<Item = Vec<BTreeSet<EntityId>>>,
) -> Vec<BTreeSet<EntityId>> {
    operands.fold(vec![BTreeSet::new()], |sets, operand| {
        (sets.iter())
            .flat_map(|set| (operand.iter()).map(move |read| set.union(read).cloned().collect()))
            .collect()
    })
}

fn signature_expression(
    source: &SemanticSource,
    definition_id: &EntityId,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
) -> Option<PhysicalExprId> {
    signatures
        .get(&mint_signature_id(&source.id, definition_id))
        .map(|signature| signature.expression_id)
}

fn column_aliases(root: PhysicalExprId, pool: &ManifestExpressions) -> BTreeSet<String> {
    pool.walk_links([root], |entry| entry.expr.links())
        .filter_map(|step| match step {
            LinkStep::Left(_, entry) => Some(entry),
            LinkStep::Closes | LinkStep::Absent => None,
        })
        .flat_map(|entry| entry.expr.nodes())
        .filter_map(|node| match node {
            Expr::Column(column) => column.parts().first().cloned(),
            _ => None,
        })
        .collect()
}
