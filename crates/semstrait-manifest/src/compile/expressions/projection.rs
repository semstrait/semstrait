use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::SemanticsName;
use semstrait_common::{EntityId, Tree};
use semstrait_ir::{JoinType, JoinsetJoinType, PhysicalExpr, SemanticRole};
use semstrait_model::{SemanticInterface, UnresolvedSemanticExpr};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Call, Expr};

use super::bodies::{SourceBodies, SourceLowered, SourceScope};
use super::columns::ColumnTypes;
use super::lower::{reconcile_boundary, Lowered, LoweredLeaf};
use crate::compile::assemble::{
    seal_signatures, ManifestAssembly, PlannedForm, SignatureTree, SourcePlan, Vocabulary,
};
use crate::compile::qualifiers::qualified_column;
use crate::compile::types::{LeafPlace, Provenance, Reference, ResolvedExprEntry, SemanticsIndex};
use crate::compile::{definitions, expressions};
use crate::error::{CompileError, ReferenceSite};
use crate::identity::mint_signature_id;
use crate::manifest::expression::Link;
use crate::manifest::expression_layer::ExprLayer;
use crate::manifest::semantic_definition::{SemanticDefinition, SemanticRoleDefinition};
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::RoleDefinition;
use crate::manifest::serving::Serving;
use crate::manifest::source_alias::SourceAliases;

pub(in crate::compile) fn seal_composed_projection(
    unit: &SourcePlan<'_>,
    interface: &SemanticInterface,
    vocabulary: Vocabulary<'_>,
    role_definition: &RoleDefinition,
    aliases: &SourceAliases,
    assembly: &mut ManifestAssembly,
) -> Result<Vec<EntityId>, CompileError> {
    let definitions = vocabulary.definitions;
    let index = definitions::interface_index(interface, definitions);
    let serving = Serving::new(&assembly.semantic_sources, &assembly.signatures);
    let own_bodies = own_bodies(
        interface,
        vocabulary.pool_bodies,
        &index,
        role_definition,
        serving,
    );
    let explicit = match role_definition {
        RoleDefinition::Unionset { null_fill, .. } => *null_fill && unit.public,
        RoleDefinition::Joinset { .. } => unit.public,
        RoleDefinition::Dataset { .. } | RoleDefinition::Grainset { .. } => false,
    };
    if explicit {
        check_exposed_served(
            unit.name.as_str(),
            interface,
            definitions,
            role_definition,
            &own_bodies,
            serving,
        )?;
    }
    let (projections, reference, copies) = match role_definition {
        RoleDefinition::Joinset {
            from,
            to,
            join_type,
            ..
        } => {
            let project = |name: &str, definition: &SemanticDefinition| {
                join_projection(
                    unit,
                    name,
                    definition,
                    (from, to),
                    *join_type,
                    aliases,
                    serving,
                )
            };
            let projections = exposed_projections(
                interface,
                &index,
                role_definition,
                &own_bodies,
                serving,
                assembly,
                project,
            )?;
            let copies = [from, to]
                .into_iter()
                .map(|operand| {
                    let alias = aliases.semantic_alias(operand)?;
                    copy_table(&projections, |projection| {
                        if projection.operands.contains(&operand) {
                            let name = projection.name.as_str();
                            alias_column(alias.as_str(), name, unit.name.as_str())
                        } else {
                            Ok(projection.column.clone())
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (projections, Reference::Inline, copies)
        }
        RoleDefinition::Unionset { branches, .. } => {
            let project = |name: &str, _: &SemanticDefinition| {
                Ok(ProjectedColumn {
                    column: alias_column(
                        aliases.semantic_alias(&unit.id)?.as_str(),
                        name,
                        unit.name.as_str(),
                    )?,
                    operands: branches.iter().collect(),
                })
            };
            let projections = exposed_projections(
                interface,
                &index,
                role_definition,
                &own_bodies,
                serving,
                assembly,
                project,
            )?;
            let copies = vec![copy_table(&projections, |projection| {
                Ok(projection.column.clone())
            })?];
            (projections, Reference::Link, copies)
        }
        RoleDefinition::Dataset { .. } | RoleDefinition::Grainset { .. } => return Ok(Vec::new()),
    };

    let grainset_child = matches!(
        unit.form,
        PlannedForm::Unionset {
            grainset_child: true,
            ..
        } | PlannedForm::Joinset {
            grainset_child: true,
            ..
        }
    );
    let filtered_projections = (projections.iter())
        .filter(|projection| projection.definition.role() == SemanticRole::Measure)
        .map(|projection| &projection.name)
        .collect();
    let (
        semantics,
        ComposedFilters {
            named_filters,
            filters,
            dropped,
        },
    ) = lower_signature_entries(ComposedLowering {
        unit,
        interface,
        vocabulary,
        index: &index,
        reference,
        place: grainset_child.then_some(LeafPlace::GrainsetChild),
        own_bodies: &own_bodies,
        projections: &projections,
        copies: &copies,
        filtered_projections,
    })?;
    let dropped = expressions::bodies::with_linkers(&unit.id, &index, &semantics, dropped);
    let projected = (projections.into_iter())
        .filter(|projection| !dropped.contains(&projection.name))
        .map(
            |ComposedProjection {
                 definition,
                 entry,
                 read_type,
                 ..
             }| SignatureTree {
                definition,
                expr: entry.expr,
                argument_type: (definition.role() == SemanticRole::Measure).then_some(read_type),
            },
        );
    let bodied = bodied_trees(&index, semantics, named_filters, &dropped);
    let signature_ids = seal_signatures(
        &unit.id,
        projected.chain(bodied),
        filters,
        expressions::emit_composition_attached,
        assembly,
    )?;
    if let RoleDefinition::Joinset { from, to, .. } = role_definition {
        check_measures_mapped_whole(
            unit.name.as_str(),
            [from, to],
            &signature_ids,
            definitions,
            aliases,
            assembly,
        )?;
    }
    Ok(signature_ids)
}

pub(super) fn bodied_trees<'d>(
    index: &SemanticsIndex<'d>,
    semantics: Vec<(SemanticsName, ResolvedExprEntry)>,
    named_filters: Vec<(SemanticsName, ResolvedExprEntry)>,
    dropped: &BTreeSet<SemanticsName>,
) -> Vec<SignatureTree<'d>> {
    (semantics.into_iter().chain(named_filters))
        .filter(|(name, _)| !dropped.contains(name))
        .filter_map(|(name, entry)| {
            Some(SignatureTree {
                definition: index.get(&name)?,
                expr: entry.expr,
                argument_type: entry.argument_type,
            })
        })
        .collect()
}

pub(crate) const STORED_TREE_RULE: &str =
    "a signature's stored tree reassembles over its source's signatures";

pub(crate) const COMPOSED_LEAF_RULE: &str =
    "a composed unit's column leaf is `<operand alias>.<semantics name>` of an operand";

fn check_measures_mapped_whole(
    unit: &str,
    operands: [&EntityId; 2],
    signature_ids: &[EntityId],
    definitions: &[SemanticDefinition],
    aliases: &SourceAliases,
    assembly: &ManifestAssembly,
) -> Result<(), CompileError> {
    let serving = Serving::new(&assembly.semantic_sources, &assembly.signatures);
    let by_alias = (operands.into_iter())
        .map(|operand| Ok((aliases.semantic_alias(operand)?.as_str(), operand)))
        .collect::<Result<BTreeMap<_, _>, CompileError>>()?;
    let by_name: BTreeMap<&str, &SemanticDefinition> = (definitions.iter())
        .map(|definition| (definition.name.as_str(), definition))
        .collect();
    let signatures: Vec<&SemanticSignature> = (signature_ids.iter())
        .filter_map(|id| assembly.signatures.get(id))
        .collect();
    let expressions = &assembly.expressions;
    let aggregates_rows = |signature: &SemanticSignature, role: &SemanticRoleDefinition| match role
    {
        SemanticRoleDefinition::Measure { .. } => true,
        SemanticRoleDefinition::Metric {
            aggregation: Some(_),
            ..
        } => (expressions.links_of(&signature.expression_id)).all(|link| {
            (expressions.get(&link)).is_some_and(|entry| entry.layer == ExprLayer::Scalar)
        }),
        _ => false,
    };
    for signature in &signatures {
        let Some(measure) = (definitions.iter())
            .find(|definition| definition.id == signature.definition_id)
            .filter(|definition| aggregates_rows(signature, &definition.role_definition))
        else {
            continue;
        };
        let violation = |rule| CompileError::StructuralViolation {
            site: ReferenceSite::Signature(signature.id.clone()),
            rule,
        };
        let roots = [Some(signature.expression_id), signature.filter_expr_id];
        let trees: Vec<PhysicalExpr> = (roots.iter().flatten())
            .map(|root| {
                (assembly.expressions.reassemble(root, &signatures, &[]))
                    .ok_or_else(|| violation(STORED_TREE_RULE))
            })
            .collect::<Result<_, _>>()?;
        let leaves = (trees.iter().flat_map(Tree::pre_order)).filter_map(|node| match node {
            Expr::Column(column) => Some(column),
            _ => None,
        });
        let mut read: BTreeMap<&EntityId, BTreeSet<&EntityId>> = BTreeMap::new();
        for column in leaves {
            let leaf = match column.parts() {
                [alias, name] => (by_alias.get(alias.as_str())).zip(by_name.get(name.as_str())),
                _ => None,
            };
            let Some((operand, definition)) = leaf else {
                return Err(violation(COMPOSED_LEAF_RULE));
            };
            read.entry(*operand).or_default().insert(&definition.id);
        }
        if read
            .iter()
            .any(|(operand, names)| !serving.maps_whole(operand, names))
        {
            return Err(unserved_composed_semantics(unit, measure.name.as_str()));
        }
    }
    Ok(())
}

pub(in crate::compile) fn check_exposed_served(
    unit_name: &str,
    interface: &SemanticInterface,
    definitions: &[SemanticDefinition],
    role_definition: &RoleDefinition,
    own_bodies: &SourceBodies<'_>,
    serving: Serving<'_>,
) -> Result<(), CompileError> {
    let index = definitions::semantics_index_from_defs(definitions);
    for name in definitions::exposed_semantics(interface) {
        let Some(definition) = index.get(&name) else {
            continue;
        };
        if !own_bodies.contains_key(&name)
            && !any_operand_serves(role_definition, &definition.id, serving)
        {
            return Err(unserved_composed_semantics(unit_name, name.as_str()));
        }
    }
    Ok(())
}

pub(super) fn own_bodies<'m>(
    interface: &'m SemanticInterface,
    pool: &'m BTreeMap<SemanticsName, UnresolvedSemanticExpr>,
    index: &SemanticsIndex<'_>,
    role_definition: &RoleDefinition,
    serving: Serving<'_>,
) -> SourceBodies<'m> {
    let lowers = |definition: &SemanticDefinition| match (role_definition, definition.role()) {
        (
            RoleDefinition::Joinset { .. }
            | RoleDefinition::Unionset {
                null_fill: true, ..
            },
            SemanticRole::Filter | SemanticRole::Metric,
        ) => true,
        (RoleDefinition::Joinset { .. }, SemanticRole::Measure) => {
            !any_operand_serves(role_definition, &definition.id, serving)
        }
        (RoleDefinition::Grainset { .. }, SemanticRole::Metric) => true,
        _ => false,
    };
    let mut bodies = expressions::bodies::body_view(interface, pool);
    bodies.retain(|name| index.get(name).is_some_and(lowers));
    bodies
}

fn of_role<'m>(
    bodies: &SourceBodies<'m>,
    index: &SemanticsIndex<'_>,
    role: SemanticRole,
) -> SourceBodies<'m> {
    let mut out = bodies.clone();
    out.retain(|name| {
        index
            .get(name)
            .is_some_and(|definition| definition.role() == role)
    });
    out
}

fn exposed_projections<'a>(
    interface: &SemanticInterface,
    index: &SemanticsIndex<'a>,
    role_definition: &RoleDefinition,
    own_bodies: &SourceBodies<'_>,
    serving: Serving<'_>,
    assembly: &ManifestAssembly,
    project: impl Fn(&str, &SemanticDefinition) -> Result<ProjectedColumn<'a>, CompileError>,
) -> Result<Vec<ComposedProjection<'a>>, CompileError> {
    let mut projections = Vec::new();

    for name in definitions::exposed_semantics(interface) {
        let Some(definition) = index.get(&name) else {
            continue;
        };
        if own_bodies.contains_key(&name)
            || !any_operand_serves(role_definition, &definition.id, serving)
        {
            continue;
        }
        let aggregation = match &definition.role_definition {
            SemanticRoleDefinition::Measure {
                aggregation,
                distinct,
                ..
            } => Some((*aggregation, *distinct)),
            SemanticRoleDefinition::Dimension { .. }
            | SemanticRoleDefinition::Metric { .. }
            | SemanticRoleDefinition::Filter => None,
        };

        let ProjectedColumn { column, operands } = project(name.as_str(), definition)?;

        let read_type = match aggregation {
            Some(_) => {
                let mut given = operands.iter().filter_map(|operand| {
                    assembly
                        .argument_types
                        .get(&mint_signature_id(operand, &definition.id))
                });
                match given.next() {
                    Some(first) if given.all(|other| other == first) => first.clone(),
                    _ => DataType::Unknown,
                }
            }
            None => definition.data_type.clone().unwrap_or(DataType::Unknown),
        };
        let read = Lowered::new(
            Expr::Leaf(LoweredLeaf::Splice {
                tree: column.clone(),
                data_type: read_type.clone(),
            }),
            [],
        );
        let lowered = match aggregation {
            Some(aggregation) => read.aggregated(aggregation, name.as_str())?,
            None => read,
        };
        let declared = definition.data_type.as_ref();
        let typed = lowered.typed(&ColumnTypes::default(), declared, name.as_str())?;
        let typed = match declared {
            Some(_) => reconcile_boundary(typed, declared, name.as_str())?,
            None => typed,
        };
        projections.push(ComposedProjection {
            name,
            definition,
            column,
            operands,
            entry: typed,
            read_type,
        });
    }
    Ok(projections)
}

pub(super) struct ComposedProjection<'a> {
    pub(super) name: SemanticsName,
    pub(super) definition: &'a SemanticDefinition,
    pub(super) column: Expr<Link>,
    pub(super) operands: Vec<&'a EntityId>,
    pub(super) entry: ResolvedExprEntry,
    pub(super) read_type: DataType,
}

#[derive(Debug)]
struct ProjectedColumn<'a> {
    column: Expr<Link>,
    operands: Vec<&'a EntityId>,
}

pub(super) fn copy_table(
    projections: &[ComposedProjection<'_>],
    column: impl Fn(&ComposedProjection<'_>) -> Result<Expr<Link>, CompileError>,
) -> Result<BTreeMap<SemanticsName, ResolvedExprEntry>, CompileError> {
    projections
        .iter()
        .map(|projection| {
            let entry = ResolvedExprEntry {
                expr: column(projection)?,
                inferred_type: projection.read_type.clone(),
                family: projection.read_type.family(),
                argument_type: None,
                referenced_columns: Vec::new(),
                provenance: Provenance::new(),
            };
            Ok((projection.name.clone(), entry))
        })
        .collect()
}

pub(super) struct ComposedLowering<'a> {
    pub(super) unit: &'a SourcePlan<'a>,
    pub(super) interface: &'a SemanticInterface,
    pub(super) vocabulary: Vocabulary<'a>,
    pub(super) index: &'a SemanticsIndex<'a>,
    pub(super) reference: Reference,
    pub(super) place: Option<LeafPlace>,
    pub(super) own_bodies: &'a SourceBodies<'a>,
    pub(super) projections: &'a [ComposedProjection<'a>],
    pub(super) copies: &'a [BTreeMap<SemanticsName, ResolvedExprEntry>],
    pub(super) filtered_projections: BTreeSet<&'a SemanticsName>,
}

pub(super) fn lower_signature_entries(
    lowering: ComposedLowering<'_>,
) -> Result<(Vec<(SemanticsName, ResolvedExprEntry)>, ComposedFilters), CompileError> {
    let ComposedLowering {
        unit,
        interface,
        vocabulary,
        index,
        reference,
        place,
        own_bodies,
        projections,
        copies,
        filtered_projections,
    } = lowering;
    let exposed = definitions::exposed_semantics(interface);
    let scope = SourceScope {
        source_name: unit.name.as_str(),
        source_id: &unit.id,
        exposed: &exposed,
        index,
        definitions: vocabulary.definitions,
        manual_mapping: false,
        composed_filter_unit: Some(unit.name.as_str()),
        reference,
        columns: &ColumnTypes::default(),
        place,
    };
    let SourceLowered {
        bodies: semantics,
        row_level_metrics,
        ..
    } = lower_semantics(scope, own_bodies, projections)?;
    let mut filtered = filtered_projections;
    filtered.extend(semantics.iter().map(|(name, _)| name));
    let mut filters = composed_filters(
        scope,
        interface,
        vocabulary.pool_filters,
        own_bodies,
        &filtered,
        copies,
    )?;
    let semantics =
        expressions::bodies::hold_own_filters(semantics, &row_level_metrics, &mut filters.filters);
    Ok((semantics, filters))
}

fn lower_semantics(
    scope: SourceScope<'_>,
    own_bodies: &SourceBodies<'_>,
    projections: &[ComposedProjection<'_>],
) -> Result<SourceLowered, CompileError> {
    let mut body_free: BTreeMap<_, _> = (projections.iter())
        .map(|projection| (projection.name.clone(), projection.entry.clone()))
        .collect();
    let mut semantics = SourceLowered::default();
    for (role, reference) in [
        (SemanticRole::Measure, scope.reference),
        (SemanticRole::Metric, Reference::Link),
    ] {
        let source = expressions::bodies::lower_source(&expressions::bodies::SourceLowering {
            scope: SourceScope { reference, ..scope },
            bodies: &of_role(own_bodies, scope.index, role),
            body_free: &body_free,
            filters: &expressions::bodies::SourceFilters::default(),
        })?;
        body_free.extend(source.bodies.iter().cloned());
        semantics.bodies.extend(source.bodies);
        semantics.row_level_metrics.extend(source.row_level_metrics);
    }
    Ok(semantics)
}

pub(super) struct ComposedFilters {
    pub(super) named_filters: Vec<(SemanticsName, ResolvedExprEntry)>,
    pub(super) filters: BTreeMap<SemanticsName, ResolvedExprEntry>,
    pub(super) dropped: BTreeSet<SemanticsName>,
}

fn joined(
    copies: impl IntoIterator<Item = impl IntoIterator<Item = (SemanticsName, ResolvedExprEntry)>>,
) -> Vec<(SemanticsName, ResolvedExprEntry)> {
    let mut copies = copies.into_iter();
    let first = copies.next().into_iter().flatten();
    let mut others: Vec<BTreeMap<_, _>> = copies.map(BTreeMap::from_iter).collect();
    first
        .map(|(name, mut tree)| {
            for copy in others.iter_mut().filter_map(|other| other.remove(&name)) {
                if copy.expr != tree.expr {
                    tree.expr = tree.expr & copy.expr;
                }
            }
            (name, tree)
        })
        .collect()
}

fn composed_filters(
    scope: SourceScope<'_>,
    interface: &SemanticInterface,
    pool_filters: &BTreeMap<SemanticsName, UnresolvedSemanticExpr>,
    own_bodies: &SourceBodies<'_>,
    filtered: &BTreeSet<&SemanticsName>,
    copies: &[BTreeMap<SemanticsName, ResolvedExprEntry>],
) -> Result<ComposedFilters, CompileError> {
    let named = of_role(own_bodies, scope.index, SemanticRole::Filter);
    let named_copies = (copies.iter())
        .map(|copy| {
            expressions::bodies::lower_source(&expressions::bodies::SourceLowering {
                scope,
                bodies: &named,
                body_free: copy,
                filters: &expressions::bodies::SourceFilters::default(),
            })
            .map(|source| source.bodies)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut filters = expressions::bodies::filter_view(interface, pool_filters);
    filters.retain(|name| filtered.contains(name));
    let site_named = expressions::bodies::site_authored_filter_names(interface);
    let tables: Vec<_> = (copies.iter())
        .map(|copy| expressions::bodies::seed_table(copy, scope.source_id))
        .collect();
    let mut signature_filters = BTreeMap::new();
    let mut dropped = BTreeSet::new();
    for (name, expr) in &filters {
        let singleton = expressions::bodies::SourceFilters::singleton(name.clone(), expr);
        let per_copy = (tables.iter())
            .map(|table| {
                expressions::bodies::lower_source_filters(&expressions::bodies::SourceFiltering {
                    scope,
                    filters: &singleton,
                    resolved: table,
                })
            })
            .collect::<Result<Vec<_>, _>>();
        match per_copy {
            Ok(lowered_copies) => {
                dropped.extend(
                    lowered_copies
                        .iter()
                        .flat_map(|lowered| lowered.dropped.clone()),
                );
                signature_filters.extend(joined(
                    lowered_copies.into_iter().map(|lowered| lowered.filters),
                ));
            }
            Err(CompileError::ColumnInComposedFilter { .. })
                if !site_named.contains(name) && !own_bodies.contains_key(name) => {}
            Err(other) => return Err(other),
        }
    }

    Ok(ComposedFilters {
        named_filters: joined(named_copies),
        filters: signature_filters,
        dropped,
    })
}

fn join_projection<'a>(
    unit: &SourcePlan<'_>,
    name: &str,
    definition: &SemanticDefinition,
    operands: (&'a EntityId, &'a EntityId),
    join_type: JoinsetJoinType,
    aliases: &SourceAliases,
    serving: Serving<'_>,
) -> Result<ProjectedColumn<'a>, CompileError> {
    let serves = |operand: &EntityId| serving.serves(operand, &definition.id);
    let side = |operand: &'a EntityId| -> Result<ProjectedColumn<'a>, CompileError> {
        Ok(ProjectedColumn {
            column: alias_column(
                aliases.semantic_alias(operand)?.as_str(),
                name,
                unit.name.as_str(),
            )?,
            operands: vec![operand],
        })
    };
    let coalesce =
        |first: &'a EntityId, second: &'a EntityId| -> Result<ProjectedColumn<'a>, CompileError> {
            let node = format!("{name} coalesce");
            let function = expressions::lower::catalog_function("coalesce", &node)?;
            Ok(ProjectedColumn {
                column: Call::new(function, vec![side(first)?.column, side(second)?.column])
                    .map(Expr::Function)
                    .map_err(|error| expressions::lower::refused_call(&node, error))?,
                operands: vec![first, second],
            })
        };
    match (serves(operands.0), serves(operands.1)) {
        (true, true) => match join_type {
            JoinsetJoinType::Join(JoinType::Right) => coalesce(operands.1, operands.0),
            JoinsetJoinType::Cross | JoinsetJoinType::Join(_) => coalesce(operands.0, operands.1),
        },
        (true, false) => side(operands.0),
        (false, true) => side(operands.1),
        (false, false) => Err(unserved_composed_semantics(unit.name.as_str(), name)),
    }
}

pub(super) fn any_operand_serves(
    role_definition: &RoleDefinition,
    definition_id: &EntityId,
    serving: Serving<'_>,
) -> bool {
    (role_definition.operands().into_iter()).any(|operand| serving.serves(operand, definition_id))
}

pub(super) fn alias_column(
    alias: &str,
    name: &str,
    unit: &str,
) -> Result<Expr<Link>, CompileError> {
    qualified_column(alias, name, unit).map(Expr::Column)
}

fn unserved_composed_semantics(unit: &str, semantic: &str) -> CompileError {
    CompileError::UnservedComposedSemantics {
        semantics: semantic.to_owned(),
        semantic_source: unit.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use semstrait_common::DataKindName;
    use semstrait_model::{Agg, UnionMode};

    use crate::manifest::bitmap::SemanticBitmask;
    use crate::manifest::expression::PhysicalExprId;
    use crate::manifest::semantic_definition::DimensionType;
    use crate::manifest::semantic_source::{Selector, SemanticSource};
    use crate::manifest::source_alias::SourceAlias;

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

    fn projected_definition(name: &str) -> SemanticDefinition {
        SemanticDefinition::new(
            format!("sd.{name}"),
            SemanticsName::from(name),
            None,
            SemanticRoleDefinition::Dimension {
                category: DimensionType::Categorical,
            },
        )
    }

    fn served_by(
        serving: &[&EntityId],
        definition: &SemanticDefinition,
    ) -> (
        BTreeMap<EntityId, SemanticSource>,
        BTreeMap<EntityId, crate::manifest::SemanticSignature>,
    ) {
        let id = |operand: &EntityId| mint_signature_id(operand, &definition.id);
        let sources = (serving.iter())
            .map(|operand| {
                let entry = SemanticSource::new(
                    operand.as_str(),
                    DataKindName::from(operand.as_str()),
                    SourceAlias::from("ssd0"),
                    true,
                    SemanticBitmask::empty(),
                    None,
                    RoleDefinition::dataset("src", Selector::Prefix),
                    Vec::new(),
                )
                .with_semantic_signature_ids(vec![id(operand)]);
                ((*operand).clone(), entry)
            })
            .collect();
        let signatures = (serving.iter())
            .map(|operand| {
                let signature = crate::manifest::SemanticSignature::new(
                    id(operand),
                    definition.id.clone(),
                    PhysicalExprId::new(uuid::Uuid::nil()),
                );
                (id(operand), signature)
            })
            .collect();
        (sources, signatures)
    }

    fn join_operands() -> (EntityId, EntityId, SourceAliases) {
        let from = EntityId::from("dk.orders");
        let to = EntityId::from("dk.customers");
        let aliases = SourceAliases {
            semantic: BTreeMap::from([
                (from.clone(), SourceAlias::from("orders")),
                (to.clone(), SourceAlias::from("customers")),
            ]),
            physical: BTreeMap::new(),
        };
        (from, to, aliases)
    }

    const JOIN_TYPES: [JoinsetJoinType; 5] = [
        JoinsetJoinType::Join(JoinType::Inner),
        JoinsetJoinType::Join(JoinType::Left),
        JoinsetJoinType::Join(JoinType::Right),
        JoinsetJoinType::Join(JoinType::Full),
        JoinsetJoinType::Cross,
    ];

    #[test]
    fn a_name_one_side_serves_is_read_from_that_side_for_every_join_kind() {
        let unit = planned("dk.sales", "sales");
        let definition = projected_definition("region");
        let (from, to, aliases) = join_operands();
        let only_from = served_by(&[&from], &definition);
        let only_to = served_by(&[&to], &definition);

        for join_type in JOIN_TYPES {
            assert_eq!(
                join_projection(
                    &unit,
                    "region",
                    &definition,
                    (&from, &to),
                    join_type,
                    &aliases,
                    Serving::new(&only_from.0, &only_from.1),
                )
                .expect("the `from` side serves it")
                .column,
                alias_column("orders", "region", "sales").expect("an alias and a name"),
                "{join_type:?} reads the only side that serves the name"
            );
            assert_eq!(
                join_projection(
                    &unit,
                    "region",
                    &definition,
                    (&from, &to),
                    join_type,
                    &aliases,
                    Serving::new(&only_to.0, &only_to.1),
                )
                .expect("the `to` side serves it")
                .column,
                alias_column("customers", "region", "sales").expect("an alias and a name"),
                "{join_type:?} reads the only side that serves the name"
            );
        }
    }

    #[test]
    fn a_name_neither_side_serves_is_refused() {
        let unit = planned("dk.sales", "sales");
        let definition = projected_definition("region");
        let (from, to, aliases) = join_operands();
        let empty = served_by(&[], &definition);

        for join_type in JOIN_TYPES {
            let error = join_projection(
                &unit,
                "region",
                &definition,
                (&from, &to),
                join_type,
                &aliases,
                Serving::new(&empty.0, &empty.1),
            )
            .expect_err("neither side serves it");
            let CompileError::UnservedComposedSemantics {
                semantics,
                semantic_source,
            } = error
            else {
                panic!("expected UnservedComposedSemantics, got {error:?}");
            };
            assert_eq!(semantic_source, "sales", "{join_type:?} names the unit");
            assert_eq!(semantics, "region", "{join_type:?} names the semantics");
        }
    }

    #[test]
    fn a_projection_reports_the_operand_whose_column_it_reads() {
        let unit = planned("dk.sales", "sales");
        let definition = projected_definition("region");
        let (from, to, aliases) = join_operands();
        let both = served_by(&[&from, &to], &definition);
        let (orders, customers) = ((&from, "orders"), (&to, "customers"));

        for (join_type, expected) in [
            (JoinsetJoinType::Join(JoinType::Inner), [orders, customers]),
            (JoinsetJoinType::Join(JoinType::Left), [orders, customers]),
            (JoinsetJoinType::Join(JoinType::Right), [customers, orders]),
            (JoinsetJoinType::Join(JoinType::Full), [orders, customers]),
            (JoinsetJoinType::Cross, [orders, customers]),
        ] {
            let read = join_projection(
                &unit,
                "region",
                &definition,
                (&from, &to),
                join_type,
                &aliases,
                Serving::new(&both.0, &both.1),
            )
            .expect("both sides serve it");
            let Expr::Function(call) = &read.column else {
                panic!("{join_type:?} reads a `coalesce`, got {:?}", read.column);
            };
            let columns = expected.map(|(_, alias)| {
                alias_column(alias, "region", "sales").expect("an alias and a name")
            });
            assert_eq!(
                (call.name(), call.args()),
                ("coalesce", columns.as_slice()),
                "{join_type:?} coalesces in the table's order"
            );
            assert_eq!(
                read.operands,
                expected.map(|(operand, _)| operand),
                "{join_type:?} reads the operands its column names"
            );
        }
    }

    fn summed(name: &str) -> SemanticDefinition {
        let sum = Agg::of_name("sum").expect("a catalog aggregate");
        SemanticDefinition::new(
            format!("sd.{name}"),
            SemanticsName::from(name),
            None,
            SemanticRoleDefinition::Measure {
                aggregation: sum.function(),
                distinct: sum.distinct(),
                additivity: None,
            },
        )
    }

    fn mapped_whole(
        parts: Option<&[&str]>,
        definition: &SemanticDefinition,
    ) -> (EntityId, Result<(), CompileError>) {
        let (from, to, aliases) = join_operands();
        let (sources, signatures) = served_by(&[&from], definition);
        let mut assembly = ManifestAssembly::new();
        assembly.semantic_sources = sources;
        assembly.signatures = signatures;
        let owner = EntityId::from("dk.sales");
        let id = mint_signature_id(&owner, &definition.id);
        let root = match parts {
            Some(parts) => {
                let column = semstrait_sql::expr::NamedReference::new(parts.iter().copied())
                    .expect("a column");
                (assembly.expressions)
                    .emit(
                        &Expr::Column(column),
                        owner.as_str(),
                        crate::manifest::expression::ExprRoot::SignatureExpression,
                        [],
                    )
                    .expect("a stored tree")
            }
            None => PhysicalExprId::new(uuid::Uuid::nil()),
        };
        let signature =
            crate::manifest::SemanticSignature::new(id.clone(), definition.id.clone(), root);
        assembly.signatures.insert(id.clone(), signature);
        let checked = check_measures_mapped_whole(
            "sales",
            [&from, &to],
            std::slice::from_ref(&id),
            std::slice::from_ref(definition),
            &aliases,
            &assembly,
        );
        (id, checked)
    }

    #[test]
    fn an_unreadable_stored_tree_is_a_structural_violation() {
        let (id, checked) = mapped_whole(None, &summed("amount"));
        assert_eq!(
            checked,
            Err(CompileError::StructuralViolation {
                site: ReferenceSite::Signature(id),
                rule: STORED_TREE_RULE,
            })
        );
    }

    #[test]
    fn a_composed_leaf_that_names_no_operands_semantics_is_a_structural_violation() {
        let definition = summed("amount");
        for parts in [&["amount"][..], &["zz", "amount"], &["orders", "nope"]] {
            let (id, checked) = mapped_whole(Some(parts), &definition);
            assert_eq!(
                checked,
                Err(CompileError::StructuralViolation {
                    site: ReferenceSite::Signature(id),
                    rule: COMPOSED_LEAF_RULE,
                }),
                "{parts:?}"
            );
        }
        let (_, checked) = mapped_whole(Some(&["orders", "amount"]), &definition);
        assert_eq!(checked, Ok(()));
    }
}
