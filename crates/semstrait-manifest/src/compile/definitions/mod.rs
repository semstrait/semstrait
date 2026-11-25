use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::Tree;
use semstrait_common::{DataKindName, SemanticsName};
use semstrait_model::{
    AiContext, DataKindFilter, Dimension, Measure, Metric, SemanticInterface as ModelInterface,
    SemanticModel, SemanticSite, UnresolvedDataType, UnresolvedSemanticExpr as SemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, NamedReference, SemanticLeaf};
use semstrait_sql::functions::AggregateFunction;
use serde::Serialize;

use crate::error::{CompileError, ReferenceScope, ShapeAxis};
use crate::identity::mint_semantic_id;
use crate::manifest::semantic_definition::{
    BucketSpec, BucketedDimensionBody, DimensionType, MetadataDimensionBody, MetadataSource,
    PathTokenRef, SemanticDefinition, SemanticRoleDefinition, TemporalDimensionBody,
};
use semstrait_ir::{AdditivityType, SemanticRole, SemiAdditivity, SemiAdditivityStrategy};

use super::types::SemanticsIndex;

pub mod graph;

pub fn build_semantic_definitions(
    model: &SemanticModel,
) -> Result<Vec<SemanticDefinition>, CompileError> {
    let mut roster: BTreeMap<SemanticsName, SemanticDefinition> = BTreeMap::new();

    for dimension in model.dimensions.values() {
        insert_definition(
            &mut roster,
            dimension_definition(dimension, dimension.entity_id())?,
        )?;
    }
    for measure in model.measures.values() {
        insert_definition(
            &mut roster,
            measure_definition(measure, measure.entity_id())?,
        )?;
    }
    for metric in model.metrics.values() {
        insert_definition(&mut roster, metric_definition(metric, metric.entity_id())?)?;
    }

    for (data_kind, interface) in public_interfaces(model) {
        for site in interface.dimensions.values() {
            match site {
                SemanticSite::Inline(decl) => {
                    insert_definition(&mut roster, dimension_definition(&decl.body, None)?)?;
                }
                SemanticSite::Reference(reference) => {
                    require_declared(&roster, reference.name(), &data_kind)?;
                }
                _ => {}
            }
        }
        for site in interface.measures.values() {
            match site {
                SemanticSite::Inline(measure) => {
                    insert_definition(&mut roster, measure_definition(measure, None)?)?;
                }
                SemanticSite::Reference(reference) => {
                    require_declared(&roster, reference.name(), &data_kind)?;
                }
                _ => {}
            }
        }
        for site in interface.metrics.values() {
            match site {
                SemanticSite::Inline(metric) => {
                    insert_definition(&mut roster, metric_definition(metric, None)?)?;
                }
                SemanticSite::Reference(reference) => {
                    require_declared(&roster, reference.name(), &data_kind)?;
                }
                _ => {}
            }
        }
        for filter in interface.filters.values() {
            insert_definition(&mut roster, filter_definition(filter))?;
        }
    }

    check_no_case_insensitive_duplicate(&roster)?;
    check_no_reserved_prefix(&roster)?;
    resolve_semi_additive_axes(model, &mut roster)?;

    Ok(roster.into_values().collect())
}

fn check_no_case_insensitive_duplicate(
    roster: &BTreeMap<SemanticsName, SemanticDefinition>,
) -> Result<(), CompileError> {
    let mut seen: BTreeMap<String, &SemanticsName> = BTreeMap::new();
    for name in roster.keys() {
        let folded = semstrait_ir::Name::folded(name.as_str()).collect();
        if let Some(first) = seen.insert(folded, name) {
            return Err(CompileError::DuplicateDefinitionName {
                first: first.as_str().to_owned(),
                second: name.as_str().to_owned(),
            });
        }
    }
    Ok(())
}

fn check_no_reserved_prefix(
    roster: &BTreeMap<SemanticsName, SemanticDefinition>,
) -> Result<(), CompileError> {
    let reserved = roster
        .keys()
        .find_map(|name| Some((name, semstrait_ir::Name::reserved_prefix(name.as_str())?)));
    if let Some((name, prefix)) = reserved {
        return Err(CompileError::ReservedDefinitionName {
            name: name.as_str().to_owned(),
            prefix,
        });
    }
    Ok(())
}

fn resolve_semi_additive_axes(
    model: &SemanticModel,
    roster: &mut BTreeMap<SemanticsName, SemanticDefinition>,
) -> Result<(), CompileError> {
    for (name, semi) in authored_semi_additivity(model) {
        let axes: Vec<EntityId> = semi_axis_definitions(|n| roster.get(n), &name, &semi.axes)?
            .into_iter()
            .map(|axis_def| axis_def.id.clone())
            .collect();
        let by: Option<EntityId> = match &semi.by {
            Some(by_name) => Some(
                roster
                    .get(by_name)
                    .ok_or_else(|| CompileError::SemiAdditiveByUnknown {
                        semantics: name.to_string(),
                        by: by_name.to_string(),
                    })?
                    .id
                    .clone(),
            ),
            None => None,
        };
        if let Some(definition) = roster.get_mut(&name) {
            if let SemanticRoleDefinition::Measure {
                additivity: Some(AdditivityType::Semi(stored)),
                ..
            } = &mut definition.role_definition
            {
                stored.axes = axes;
                stored.by = by;
            }
        }
    }
    Ok(())
}

pub(super) fn authored_semi_additivity(
    model: &SemanticModel,
) -> BTreeMap<SemanticsName, semstrait_model::SemiAdditivity> {
    let mut out: BTreeMap<SemanticsName, semstrait_model::SemiAdditivity> = BTreeMap::new();
    for measure in model.measures.values() {
        if let Some(semstrait_model::AdditivityType::Semi(semi)) = &measure.additivity {
            out.insert(measure.name().clone(), semi.clone());
        }
    }
    for (_, interface) in public_interfaces(model) {
        for site in interface.measures.values() {
            if let SemanticSite::Inline(measure) = site {
                if let Some(semstrait_model::AdditivityType::Semi(semi)) = &measure.additivity {
                    out.insert(measure.name().clone(), semi.clone());
                }
            }
        }
    }
    out
}

pub(super) fn semi_axis_definitions<'d>(
    lookup: impl Fn(&SemanticsName) -> Option<&'d SemanticDefinition>,
    semantics: &SemanticsName,
    axes: &[SemanticsName],
) -> Result<Vec<&'d SemanticDefinition>, CompileError> {
    axes.iter()
        .map(|axis| {
            let axis_def = lookup(axis).ok_or_else(|| CompileError::SemiAdditiveAxisUnknown {
                semantics: semantics.to_string(),
                axis: axis.to_string(),
            })?;
            match axis_def.role_definition.role() {
                SemanticRole::Dimension => Ok(axis_def),
                role => Err(CompileError::SemiAdditiveAxisNotADimension {
                    semantics: semantics.to_string(),
                    axis: axis.to_string(),
                    role,
                }),
            }
        })
        .collect()
}

fn public_interfaces(model: &SemanticModel) -> Vec<(DataKindName, &ModelInterface)> {
    let mut out: Vec<(DataKindName, &ModelInterface)> = Vec::new();
    for dataset in model.datasets.values() {
        out.push((
            DataKindName::from(dataset.body.base.name.as_str()),
            &dataset.semantic_interface,
        ));
    }
    for unionset in model.unionsets.values() {
        out.push((
            DataKindName::from(unionset.body.base.name.as_str()),
            &unionset.semantic_interface,
        ));
    }
    for grainset in model.grainsets.values() {
        out.push((
            DataKindName::from(grainset.body.base.name.as_str()),
            &grainset.semantic_interface,
        ));
    }
    for joinset in model.joinsets.values() {
        out.push((
            DataKindName::from(joinset.body.base.name.as_str()),
            &joinset.semantic_interface,
        ));
    }
    out
}

fn insert_definition(
    roster: &mut BTreeMap<SemanticsName, SemanticDefinition>,
    definition: SemanticDefinition,
) -> Result<(), CompileError> {
    match roster.get(&definition.name) {
        Some(existing) if existing.role() != definition.role() => {
            Err(CompileError::SemanticKindMismatch {
                authored_kind: definition.role(),
                registered_kind: existing.role(),
                name: definition.name.as_str().to_owned(),
            })
        }
        Some(existing) => check_shape_agrees(existing, &definition),
        None => {
            roster.insert(definition.name.clone(), definition);
            Ok(())
        }
    }
}

fn describe_axis_value<T: Serialize>(value: Option<T>) -> String {
    match value {
        Some(v) => match serde_json::to_value(&v) {
            Ok(serde_json::Value::String(s)) => s,
            Ok(other) => other.to_string(),
            Err(_) => "<unrenderable>".to_owned(),
        },
        None => "none".to_owned(),
    }
}

fn axis_differs<T: PartialEq + Serialize>(
    name: &str,
    axis: ShapeAxis,
    authored: Option<T>,
    existing: Option<T>,
) -> Result<(), CompileError> {
    if authored == existing {
        return Ok(());
    }
    Err(CompileError::SemanticShapeMismatch {
        name: name.to_owned(),
        axis,
        authored: describe_axis_value(authored),
        registered: describe_axis_value(existing),
    })
}

fn describe_aggregation(value: Option<(&'static AggregateFunction, bool)>) -> String {
    match value {
        Some((function, true)) => format!("{}_distinct", function.declarations.canonical.name),
        Some((function, false)) => function.declarations.canonical.name.to_owned(),
        None => "none".to_owned(),
    }
}

fn aggregation_differs(
    name: &str,
    authored: Option<(&'static AggregateFunction, bool)>,
    existing: Option<(&'static AggregateFunction, bool)>,
) -> Result<(), CompileError> {
    if authored == existing {
        return Ok(());
    }
    Err(CompileError::SemanticShapeMismatch {
        name: name.to_owned(),
        axis: ShapeAxis::Aggregation,
        authored: describe_aggregation(authored),
        registered: describe_aggregation(existing),
    })
}

fn check_shape_agrees(
    existing: &SemanticDefinition,
    authored: &SemanticDefinition,
) -> Result<(), CompileError> {
    let name = authored.name.as_str();
    axis_differs(
        name,
        ShapeAxis::DataType,
        authored.data_type.clone(),
        existing.data_type.clone(),
    )?;
    axis_differs(
        name,
        ShapeAxis::Category,
        authored.role_definition.category(),
        existing.role_definition.category(),
    )?;
    aggregation_differs(
        name,
        authored.role_definition.aggregation(),
        existing.role_definition.aggregation(),
    )?;
    axis_differs(
        name,
        ShapeAxis::Additivity,
        authored.role_definition.additivity(),
        existing.role_definition.additivity(),
    )?;
    Ok(())
}

fn require_declared(
    roster: &BTreeMap<SemanticsName, SemanticDefinition>,
    name: &SemanticsName,
    data_kind: &DataKindName,
) -> Result<(), CompileError> {
    if roster.contains_key(name) {
        return Ok(());
    }
    Err(CompileError::UnknownReference {
        name: name.as_str().to_owned(),
        scope: ReferenceScope::DataKind(data_kind.as_str().to_owned()),
    })
}

fn declared_type(
    name: &SemanticsName,
    data_type: &UnresolvedDataType,
) -> Result<DataType, CompileError> {
    data_type
        .resolve()
        .ok_or_else(|| CompileError::TypeInferenceFailure {
            node: name.as_str().to_owned(),
            reason: format!(
                "its semantics declares data_type `{}`, which is not a valid data type",
                data_type.spelling()
            ),
        })
}

impl From<semstrait_model::AiContext> for crate::manifest::semantic_definition::AiContext {
    fn from(authored: semstrait_model::AiContext) -> Self {
        Self {
            synonyms: authored.synonyms,
            description: authored.description,
            examples: authored.examples,
            unit: authored.unit,
        }
    }
}

impl TryFrom<&semstrait_model::DimensionType> for DimensionType {
    type Error = &'static str;

    fn try_from(authored: &semstrait_model::DimensionType) -> Result<Self, Self::Error> {
        Ok(match authored {
            semstrait_model::DimensionType::Categorical => Self::Categorical,
            semstrait_model::DimensionType::Binary => Self::Binary,
            semstrait_model::DimensionType::Geo => Self::Geo,
            semstrait_model::DimensionType::Temporal(body) => {
                Self::Temporal(TemporalDimensionBody {
                    granularities: body.granularities.clone(),
                })
            }
            semstrait_model::DimensionType::Bucketed(body) => {
                Self::Bucketed(BucketedDimensionBody {
                    buckets: body.buckets.iter().map(BucketSpec::from).collect(),
                })
            }
            semstrait_model::DimensionType::Metadata(body) => {
                Self::Metadata(MetadataDimensionBody {
                    source: MetadataSource::try_from(&body.source)?,
                })
            }
            other => unreachable!("semstrait-model added a DimensionType variant: {other:?}"),
        })
    }
}

impl From<&semstrait_model::BucketSpec> for BucketSpec {
    fn from(authored: &semstrait_model::BucketSpec) -> Self {
        Self {
            name: authored.name.clone(),
            lower: authored.lower.clone(),
            upper: authored.upper.clone(),
        }
    }
}

impl TryFrom<&semstrait_model::MetadataSource> for MetadataSource {
    type Error = &'static str;

    fn try_from(authored: &semstrait_model::MetadataSource) -> Result<Self, Self::Error> {
        match authored {
            semstrait_model::MetadataSource::Path(reference) => Ok(Self::Path(PathTokenRef {
                token: reference.token,
            })),
            _ => Err("unknown"),
        }
    }
}

fn additivity_type(authored: semstrait_model::AdditivityType) -> AdditivityType {
    match authored {
        semstrait_model::AdditivityType::Full => AdditivityType::Full,
        semstrait_model::AdditivityType::Non => AdditivityType::Non,
        semstrait_model::AdditivityType::Semi(semi) => AdditivityType::Semi(SemiAdditivity::new(
            semi.axes
                .iter()
                .map(|axis| axis.as_str().to_owned())
                .collect(),
            semi.by.as_ref().map(|by| by.as_str().to_owned()),
            semi_additivity_strategy(semi.strategy),
        )),
        other => unreachable!("semstrait-model added an AdditivityType variant: {other:?}"),
    }
}

fn semi_additivity_strategy(
    authored: semstrait_model::SemiAdditivityStrategy,
) -> SemiAdditivityStrategy {
    match authored {
        semstrait_model::SemiAdditivityStrategy::Latest => SemiAdditivityStrategy::Latest,
        semstrait_model::SemiAdditivityStrategy::Earliest => SemiAdditivityStrategy::Earliest,
        other => {
            unreachable!("semstrait-model added a SemiAdditivityStrategy variant: {other:?}")
        }
    }
}

fn dimension_definition(
    dimension: &Dimension,
    authored_id: Option<&str>,
) -> Result<SemanticDefinition, CompileError> {
    let category = DimensionType::try_from(&dimension.dim_type).map_err(|form| {
        CompileError::MetadataExtractionUnsupported {
            semantic: dimension.name().as_str().to_owned(),
            form,
        }
    })?;
    Ok(definition_from(
        authored_id,
        dimension.name(),
        SemanticRoleDefinition::Dimension { category },
        Some(declared_type(dimension.name(), dimension.data_type())?),
        dimension.body.description.as_deref(),
        dimension.body.ai_context.as_ref(),
    ))
}

fn measure_definition(
    measure: &Measure,
    authored_id: Option<&str>,
) -> Result<SemanticDefinition, CompileError> {
    Ok(definition_from(
        authored_id,
        measure.name(),
        SemanticRoleDefinition::Measure {
            aggregation: measure.agg.function(),
            distinct: measure.agg.distinct(),
            additivity: measure.additivity.clone().map(additivity_type),
        },
        Some(declared_type(measure.name(), measure.data_type())?),
        measure.body.description.as_deref(),
        measure.body.ai_context.as_ref(),
    ))
}

fn metric_definition(
    metric: &Metric,
    authored_id: Option<&str>,
) -> Result<SemanticDefinition, CompileError> {
    Ok(definition_from(
        authored_id,
        metric.name(),
        SemanticRoleDefinition::Metric {
            aggregation: metric.agg.map(|agg| agg.function()),
            distinct: metric.agg.is_some_and(|agg| agg.distinct()),
        },
        Some(declared_type(metric.name(), metric.data_type())?),
        metric.body.description.as_deref(),
        metric.body.ai_context.as_ref(),
    ))
}

fn filter_definition(filter: &DataKindFilter) -> SemanticDefinition {
    definition_from(
        None,
        &filter.name,
        SemanticRoleDefinition::Filter,
        None,
        filter.description.as_deref(),
        filter.ai_context.as_ref(),
    )
}

fn definition_from(
    authored_id: Option<&str>,
    name: &SemanticsName,
    role_definition: SemanticRoleDefinition,
    data_type: Option<DataType>,
    description: Option<&str>,
    ai_context: Option<&AiContext>,
) -> SemanticDefinition {
    let id = authored_id.map_or_else(
        || mint_semantic_id(role_definition.role(), name.as_str()),
        ToOwned::to_owned,
    );
    SemanticDefinition::new(id, name.clone(), data_type, role_definition).with_documentation(
        description.map(ToOwned::to_owned),
        ai_context.cloned().map(Into::into),
    )
}

#[must_use]
pub fn semantics_index_from_defs(defs: &[SemanticDefinition]) -> SemanticsIndex<'_> {
    let by_name: BTreeMap<SemanticsName, &SemanticDefinition> =
        defs.iter().map(|def| (def.name.clone(), def)).collect();
    SemanticsIndex::from_map(by_name)
}

#[must_use]
pub(super) fn exposed_semantics(interface: &ModelInterface) -> BTreeSet<SemanticsName> {
    let mut out: BTreeSet<SemanticsName> = BTreeSet::new();
    out.extend(interface.dimensions.values().map(|s| s.name().clone()));
    out.extend(interface.measures.values().map(|s| s.name().clone()));
    out.extend(interface.metrics.values().map(|s| s.name().clone()));
    out.extend(interface.filters.values().map(|f| f.name.clone()));
    out
}

#[must_use]
pub(super) fn interface_index<'a>(
    interface: &ModelInterface,
    defs: &'a [SemanticDefinition],
) -> SemanticsIndex<'a> {
    let exposed = exposed_semantics(interface);
    let by_name: BTreeMap<SemanticsName, &'a SemanticDefinition> = defs
        .iter()
        .filter(|def| exposed.contains(&def.name))
        .map(|def| (def.name.clone(), def))
        .collect();
    SemanticsIndex::from_map(by_name)
}

#[must_use]
pub(super) fn pool_bodies(model: &SemanticModel) -> BTreeMap<SemanticsName, SemanticExpr> {
    let mut out: BTreeMap<SemanticsName, SemanticExpr> = BTreeMap::new();
    for dimension in model.dimensions.values() {
        if let Some(expr) = dimension.expr() {
            out.insert(dimension.name().clone(), expr.clone());
        }
    }
    for measure in model.measures.values() {
        if let Some(expr) = measure.expr() {
            out.insert(measure.name().clone(), expr.clone());
        }
    }
    for metric in model.metrics.values() {
        if let Some(expr) = metric.expr() {
            out.insert(metric.name().clone(), expr.clone());
        }
    }
    out
}

#[must_use]
pub(super) fn pool_filters(model: &SemanticModel) -> BTreeMap<SemanticsName, SemanticExpr> {
    let mut out: BTreeMap<SemanticsName, SemanticExpr> = BTreeMap::new();
    for measure in model.measures.values() {
        if let Some(filter) = measure.filter.as_ref() {
            out.insert(measure.name().clone(), filter.clone());
        }
    }
    for metric in model.metrics.values() {
        if let Some(filter) = metric.filter.as_ref() {
            out.insert(metric.name().clone(), filter.clone());
        }
    }
    out
}

pub(super) fn build_reference_graph(
    bodies: &BTreeMap<SemanticsName, &SemanticExpr>,
    exposed: &BTreeSet<SemanticsName>,
    registered: &BTreeSet<SemanticsName>,
    source: &str,
    source_id: &EntityId,
) -> Result<BTreeMap<SemanticsName, BTreeSet<SemanticsName>>, CompileError> {
    let mut graph: BTreeMap<SemanticsName, BTreeSet<SemanticsName>> = exposed
        .iter()
        .map(|name| (name.clone(), BTreeSet::new()))
        .collect();

    for (owner, body) in bodies {
        let mut referenced: BTreeSet<SemanticsName> = BTreeSet::new();
        collect_referenced_semantics(body, &mut referenced);
        for target in &referenced {
            if !exposed.contains(target) {
                return Err(if registered.contains(target) {
                    CompileError::UnexposedReference {
                        semantics: owner.as_str().to_owned(),
                        referenced: target.as_str().to_owned(),
                        semantic_source: source.to_owned(),
                    }
                } else {
                    CompileError::UnknownReference {
                        name: target.as_str().to_owned(),
                        scope: ReferenceScope::DataKind(source_id.clone()),
                    }
                });
            }
        }
        if let Some(edges) = graph.get_mut(owner) {
            edges.extend(referenced);
        }
    }

    Ok(graph)
}

pub(in crate::compile) fn collect_referenced_semantics(
    expr: &SemanticExpr,
    out: &mut impl Extend<SemanticsName>,
) {
    if let Expr::Leaf(leaf) = expr {
        if let Some(referenced) = semantic_leaf_target(leaf) {
            out.extend([SemanticsName::from(referenced.name())]);
        }
    }
    for child in expr.children() {
        collect_referenced_semantics(child, out);
    }
}

fn semantic_leaf_target(leaf: &SemanticLeaf) -> Option<&NamedReference> {
    match leaf {
        SemanticLeaf::Field(name)
        | SemanticLeaf::Dimension(name, _)
        | SemanticLeaf::Measure(name, _)
        | SemanticLeaf::Metric(name, _) => Some(name),
        SemanticLeaf::Untyped(_) => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::identity::mint_semantic_id;
    use semstrait_common::identity::{mint_named_entity_id, EntityNamespace};
    use semstrait_model::{Dimension, Measure, Metric, Relationship};
    use semstrait_sql::expr::Literal;

    fn names(raw: &[&str]) -> BTreeSet<SemanticsName> {
        raw.iter().map(|n| name(n)).collect()
    }

    fn dimension(name: &str, ty: DataType) -> Dimension {
        Dimension::builder(name).data_type(ty).categorical().build()
    }

    fn measure(name: &str, ty: DataType) -> Measure {
        Measure::builder(name)
            .data_type(ty)
            .agg(semstrait_model::Agg::of_name("sum").expect("a catalog aggregate"))
            .build()
    }

    fn metric(name: &str, ty: DataType) -> Metric {
        Metric::builder(name).data_type(ty).build()
    }

    fn model_with(
        dimensions: Vec<Dimension>,
        measures: Vec<Measure>,
        metrics: Vec<Metric>,
        relationships: Vec<Relationship>,
    ) -> SemanticModel {
        let mut m = SemanticModel::default();
        for d in dimensions {
            m.dimensions
                .insert(d.entity_id().expect("builder mints id").to_owned(), d);
        }
        for me in measures {
            m.measures
                .insert(me.entity_id().expect("builder mints id").to_owned(), me);
        }
        for mt in metrics {
            m.metrics
                .insert(mt.entity_id().expect("builder mints id").to_owned(), mt);
        }
        for mut r in relationships {
            let id = r
                .id
                .clone()
                .unwrap_or_else(|| mint_named_entity_id(EntityNamespace::Relationship, &r.name));
            r.id = Some(id.clone());
            m.relationships.insert(id, r);
        }
        m
    }

    fn field(name: &str) -> SemanticExpr {
        Expr::Leaf(SemanticLeaf::Field(
            NamedReference::unqualified(name).expect("a name"),
        ))
    }

    fn measure_leaf(name: &str) -> SemanticExpr {
        Expr::Leaf(SemanticLeaf::Measure(
            NamedReference::unqualified(name).expect("a name"),
            None,
        ))
    }

    fn col(name: &str) -> SemanticExpr {
        Expr::Column(NamedReference::unqualified(name).expect("a name"))
    }

    fn int(value: i32) -> SemanticExpr {
        Expr::Literal(Literal::Integer(value))
    }

    fn add(left: SemanticExpr, right: SemanticExpr) -> SemanticExpr {
        left + right
    }

    fn name(s: &str) -> SemanticsName {
        SemanticsName::from(s)
    }

    #[test]
    fn definitions_are_minted_per_pool_with_roles_and_types() {
        let model = model_with(
            vec![dimension("region", DataType::String)],
            vec![measure("revenue", DataType::Double)],
            vec![metric("margin", DataType::Double)],
            vec![],
        );
        let defs = build_semantic_definitions(&model).expect("mint");
        assert_eq!(defs.len(), 3);

        let by_name: BTreeMap<&str, &SemanticDefinition> =
            defs.iter().map(|d| (d.name.as_str(), d)).collect();

        let region = by_name["region"];
        assert_eq!(region.role(), SemanticRole::Dimension);
        assert_eq!(region.data_type, Some(DataType::String));
        assert_eq!(
            region.id,
            mint_semantic_id(SemanticRole::Dimension, "region")
        );
        assert_ne!(region.id, "region");

        assert_eq!(by_name["revenue"].role(), SemanticRole::Measure);
        assert_eq!(by_name["revenue"].data_type, Some(DataType::Double));
        assert_eq!(by_name["margin"].role(), SemanticRole::Metric);
    }

    #[test]
    fn no_definition_leaves_this_stage_with_a_position() {
        let model = model_with(
            vec![dimension("region", DataType::String)],
            vec![measure("revenue", DataType::Double)],
            vec![metric("margin", DataType::Double)],
            vec![],
        );
        let defs = build_semantic_definitions(&model).expect("build");
        for def in &defs {
            assert_eq!(
                def.bit_position, 0,
                "`{}` must carry no position of this stage's making",
                def.name
            );
        }
    }

    #[test]
    fn two_carriers_claiming_one_name_are_refused() {
        let model = model_with(
            vec![dimension("amount", DataType::String)],
            vec![measure("amount", DataType::Double)],
            vec![],
            vec![],
        );
        match build_semantic_definitions(&model) {
            Err(CompileError::SemanticKindMismatch {
                authored_kind,
                registered_kind,
                name,
            }) => {
                assert_eq!(name, "amount", "names the contested name");
                assert_ne!(
                    authored_kind, registered_kind,
                    "both roles are named so the author can see the pair"
                );
            }
            other => panic!("expected SemanticKindMismatch, got {other:?}"),
        }
    }

    #[test]
    fn two_occurrences_disagreeing_on_data_type_are_refused() {
        let mut roster: BTreeMap<SemanticsName, SemanticDefinition> = BTreeMap::new();
        insert_definition(
            &mut roster,
            dimension_definition(&dimension("region", DataType::String), None)
                .expect("a valid type"),
        )
        .expect("first occurrence seeds the roster");
        let outcome = insert_definition(
            &mut roster,
            dimension_definition(&dimension("region", DataType::Integer), None)
                .expect("a valid type"),
        );
        match outcome {
            Err(CompileError::SemanticShapeMismatch { name, axis, .. }) => {
                assert_eq!(name, "region");
                assert_eq!(axis, crate::error::ShapeAxis::DataType);
            }
            other => panic!("expected SemanticShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn two_occurrences_agreeing_on_every_axis_are_one_entry() {
        let mut roster: BTreeMap<SemanticsName, SemanticDefinition> = BTreeMap::new();
        insert_definition(
            &mut roster,
            dimension_definition(&dimension("region", DataType::String), None)
                .expect("a valid type"),
        )
        .expect("first occurrence seeds the roster");
        insert_definition(
            &mut roster,
            dimension_definition(&dimension("region", DataType::String), None)
                .expect("a valid type"),
        )
        .expect("an agreeing occurrence of the same name is not a conflict");
        assert_eq!(roster.len(), 1, "one name holds one entry");
    }

    #[test]
    fn definitions_are_byte_stable_across_runs() {
        let model = model_with(
            vec![
                dimension("b_dim", DataType::String),
                dimension("a_dim", DataType::String),
            ],
            vec![measure("c_meas", DataType::Double)],
            vec![],
            vec![],
        );
        let one = build_semantic_definitions(&model).expect("build");
        let two = build_semantic_definitions(&model).expect("build");
        assert_eq!(one, two, "the roster is deterministic");
    }

    #[test]
    fn semantics_index_borrows_the_roster() {
        let model = model_with(
            vec![dimension("region", DataType::String)],
            vec![measure("revenue", DataType::Double)],
            vec![],
            vec![],
        );
        let defs = build_semantic_definitions(&model).expect("build");
        let index = semantics_index_from_defs(&defs);
        let revenue = index.get(&name("revenue")).expect("registered");
        assert_eq!(revenue.role(), SemanticRole::Measure);
        assert!(index.get(&name("absent")).is_none());
    }

    #[test]
    fn reference_graph_every_exposed_semantics_is_a_node() {
        let exposed = names(&["margin", "region", "revenue"]);
        let source_id: EntityId = "orders-id".to_owned();
        let graph =
            build_reference_graph(&BTreeMap::new(), &exposed, &exposed, "orders", &source_id)
                .expect("no body, no reference");
        assert_eq!(graph.len(), 3);
        assert!(graph.values().all(BTreeSet::is_empty));
    }

    #[test]
    fn reference_graph_edges_are_the_bodys_semantic_leaves() {
        let exposed = names(&["cost", "margin", "revenue"]);
        let margin = add(measure_leaf("revenue"), field("cost"));
        let revenue = add(col("amount"), int(1));
        let bodies = BTreeMap::from([(name("margin"), &margin), (name("revenue"), &revenue)]);

        let source_id: EntityId = "orders-id".to_owned();
        let graph = build_reference_graph(&bodies, &exposed, &exposed, "orders", &source_id)
            .expect("resolves");
        assert_eq!(graph[&name("margin")], names(&["cost", "revenue"]));
        assert!(
            graph[&name("revenue")].is_empty(),
            "a column leaf is not a semantic reference"
        );
    }

    #[test]
    fn reference_graph_self_loop_edge_is_kept() {
        let exposed = names(&["loop_m"]);
        let body = measure_leaf("loop_m");
        let bodies = BTreeMap::from([(name("loop_m"), &body)]);
        let source_id: EntityId = "orders-id".to_owned();
        let graph = build_reference_graph(&bodies, &exposed, &exposed, "orders", &source_id)
            .expect("resolves");
        assert!(graph[&name("loop_m")].contains(&name("loop_m")));
    }

    #[test]
    fn reference_graph_refuses_a_name_the_source_does_not_expose() {
        let exposed = names(&["margin"]);
        let registered = names(&["cost", "margin", "revenue"]);
        let body = add(measure_leaf("revenue"), measure_leaf("cost"));
        let bodies = BTreeMap::from([(name("margin"), &body)]);
        let source_id: EntityId = "orders-id".to_owned();

        let error = build_reference_graph(&bodies, &exposed, &registered, "orders", &source_id)
            .expect_err("`cost` is registered but not exposed here");
        match error {
            CompileError::UnexposedReference {
                semantics,
                referenced,
                semantic_source,
            } => {
                assert_eq!(semantics, "margin");
                assert_eq!(referenced, "cost");
                assert_eq!(semantic_source, "orders");
            }
            other => panic!("expected UnexposedReference, got {other:?}"),
        }

        let unknown =
            build_reference_graph(&bodies, &exposed, &names(&["margin"]), "orders", &source_id)
                .expect_err("`cost` is no definition at all");
        match unknown {
            CompileError::UnknownReference { scope, .. } => {
                assert_eq!(
                    scope,
                    ReferenceScope::DataKind(source_id.clone()),
                    "the scope carries the source's EntityId, not its name"
                );
            }
            other => panic!("expected UnknownReference, got {other:?}"),
        }
    }

    #[test]
    fn reference_graph_feeds_topological_order_deterministically() {
        use super::super::types::RecursionState;
        use super::graph::topological_order;

        let exposed = names(&["margin", "revenue"]);
        let body = measure_leaf("revenue");
        let bodies = BTreeMap::from([(name("margin"), &body)]);
        let source_id: EntityId = "orders-id".to_owned();
        let graph = build_reference_graph(&bodies, &exposed, &exposed, "orders", &source_id)
            .expect("resolves");
        assert_eq!(
            graph,
            build_reference_graph(&bodies, &exposed, &exposed, "orders", &source_id)
                .expect("resolves"),
            "the reference graph is byte-stable"
        );

        let mut rec = RecursionState::new();
        let order = topological_order(&graph, &mut rec).expect("acyclic");
        let pos = |n: &str| order.iter().position(|x| x.as_str() == n).expect("present");
        assert!(pos("revenue") < pos("margin"));
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn pooled_bodies_are_the_pool_declarations_exprs() {
        let model = model_with(
            vec![dimension("region", DataType::String)],
            vec![measure("revenue", DataType::Double)],
            vec![Metric::builder("margin")
                .data_type(DataType::Double)
                .expr(measure_leaf("revenue"))
                .build()],
            vec![],
        );
        let bodies = pool_bodies(&model);
        assert_eq!(bodies.keys().collect::<Vec<_>>(), vec![&name("margin")]);
    }
}
