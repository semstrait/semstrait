use std::collections::BTreeSet;

use semstrait_common::{Granularity, SemanticsName};
use semstrait_ir::SemanticRole;
use semstrait_manifest::{DimensionType, SemanticDefinition, SemanticRoleDefinition};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, NamedReference, RowExpr, SemanticLeaf};
use semstrait_sql::DocumentDialect;

use crate::formula::planned_as;
use crate::predicate::{beyond, leaves_beyond, reads, typed, Read};
use crate::read_model::ReadModel;
use crate::{LookedIn, Placement, PlanError, Predicate, Request, Selection};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundName {
    pub name: SemanticsName,
    pub bit: u32,
}

impl BoundName {
    pub(crate) fn of(definition: SemanticDefinition) -> Self {
        Self {
            name: definition.name,
            bit: definition.bit_position,
        }
    }
}

pub(crate) struct BoundRequest {
    pub(crate) selected: Vec<BoundName>,
    pub(crate) predicates: Vec<Predicate>,
}

impl BoundRequest {
    pub(crate) fn demanded(&self) -> Vec<BoundName> {
        (self.selected.iter())
            .chain(leaves_beyond(&self.selected, &self.predicates))
            .cloned()
            .collect()
    }
}

type Bound = (BoundName, SemanticDefinition);

pub(crate) fn bind(
    read_model: &impl ReadModel,
    request: &Request,
    dialect: DocumentDialect,
) -> Result<BoundRequest, PlanError> {
    let (selected, definitions): (Vec<_>, Vec<_>) = (request.select.iter())
        .map(|selection| lookup(read_model, selection.name()))
        .collect::<Result<_, _>>()?;
    let predicate_reads = (request.predicate.iter().flat_map(reads))
        .map(|read| bound_read(read_model, read))
        .collect::<Result<Vec<_>, _>>()?;
    let filters = (request.filters.iter())
        .map(|filter| activated(read_model, filter))
        .collect::<Result<Vec<_>, _>>()?;
    let each = |check: fn(&Selection, &SemanticDefinition) -> Result<(), PlanError>| {
        (request.select.iter().zip(&definitions))
            .try_for_each(|(selection, definition)| check(selection, definition))
    };
    each(role_matches)?;
    select_shape(&selected)?;
    reads_no_filter(&predicate_reads)?;
    each(temporal_if_granular)?;
    each(selectable)?;
    ordered_by_selected(request)?;
    (predicate_reads.iter()).try_for_each(|(read, bound)| leaf_admitted(read, bound.as_ref()))?;
    each(grain_fits_type)?;
    let predicate = (request.predicate.as_ref())
        .map(|tree| placed_and_typed(read_model, tree, &predicate_reads, dialect))
        .transpose()?;
    let predicates = predicate.into_iter().chain(filters).collect();
    Ok(BoundRequest {
        selected,
        predicates,
    })
}

fn lookup(read_model: &impl ReadModel, name: &SemanticsName) -> Result<Bound, PlanError> {
    (read_model.bit_of(name.as_str()))
        .and_then(|bit| Some((bit, read_model.definition(bit)?)))
        .map(|(bit, definition)| {
            let name = name.clone();
            (BoundName { name, bit }, definition)
        })
        .ok_or_else(|| PlanError::UnknownName {
            name: name.to_string(),
            looked_in: LookedIn::SemanticDefinitions,
        })
}

fn bound_read<'p>(
    read_model: &impl ReadModel,
    read: Read<'p>,
) -> Result<(Read<'p>, Option<Bound>), PlanError> {
    let bound = match &read {
        Read::Semantics { name, .. } => Some(lookup(read_model, name)?),
        Read::Column(_) => None,
    };
    Ok((read, bound))
}

fn activated(read_model: &impl ReadModel, filter: &SemanticsName) -> Result<Predicate, PlanError> {
    let (bound, definition) = lookup(read_model, filter)?;
    let role = definition.role();
    let field = (NamedReference::unqualified(filter.as_str()))
        .filter(|_| role == SemanticRole::Filter)
        .ok_or_else(|| PlanError::UnknownName {
            name: filter.to_string(),
            looked_in: LookedIn::NamedFilters,
        })?;
    Ok(Predicate {
        tree: Expr::Leaf(SemanticLeaf::Field(field)),
        placement: Placement::of([(&bound.name, role)])?,
        leaves: vec![bound],
    })
}

fn placed_and_typed(
    read_model: &impl ReadModel,
    tree: &RowExpr<SemanticLeaf>,
    reads: &[(Read<'_>, Option<Bound>)],
    dialect: DocumentDialect,
) -> Result<Predicate, PlanError> {
    let sources = read_model.public_sources();
    let dimension = |definition: &SemanticDefinition| {
        (sources.values())
            .filter(|(_, mask)| mask.contains(definition.bit_position))
            .all(|(source, _)| {
                planned_as(read_model, source, definition) == Some(SemanticRole::Dimension)
            })
    };
    let role = |definition: &SemanticDefinition| match definition.role() {
        SemanticRole::Metric if dimension(definition) => SemanticRole::Dimension,
        role => role,
    };
    let semantics = || reads.iter().filter_map(|(_, bound)| bound.as_ref());
    let placement =
        Placement::of(semantics().map(|(bound, definition)| (&bound.name, role(definition))))?;
    let leaves = beyond(&[], semantics().map(|(bound, _)| bound));
    Ok(Predicate {
        tree: typed(tree.clone(), read_model, dialect)?,
        leaves: leaves.into_iter().cloned().collect(),
        placement,
    })
}

fn role_matches(selection: &Selection, definition: &SemanticDefinition) -> Result<(), PlanError> {
    let role = definition.role();
    match selection.role() {
        Some(selected) if selected != role => Err(PlanError::SelectionRoleMismatch {
            name: selection.name().clone(),
            selected,
            role,
        }),
        _ => Ok(()),
    }
}

fn select_shape(selected: &[BoundName]) -> Result<(), PlanError> {
    if selected.is_empty() {
        return Err(PlanError::EmptySelect);
    }
    let mut bits = BTreeSet::new();
    match (selected.iter()).find(|bound| !bits.insert(bound.bit)) {
        Some(bound) => Err(PlanError::RepeatedSelection {
            name: bound.name.clone(),
        }),
        None => Ok(()),
    }
}

fn reads_no_filter(reads: &[(Read<'_>, Option<Bound>)]) -> Result<(), PlanError> {
    let filter = (reads.iter().filter_map(|(_, bound)| bound.as_ref()))
        .find(|(_, definition)| definition.role() == SemanticRole::Filter);
    match filter {
        Some((bound, _)) => Err(PlanError::FilterInRequestPredicate {
            name: bound.name.clone(),
        }),
        None => Ok(()),
    }
}

fn temporal_if_granular(
    selection: &Selection,
    definition: &SemanticDefinition,
) -> Result<(), PlanError> {
    let temporal = matches!(
        definition.role_definition,
        SemanticRoleDefinition::Dimension {
            category: DimensionType::Temporal(_)
        }
    );
    match selection.granularity() {
        Some(requested) if !temporal => Err(PlanError::GranularityOnNonTemporalDimension {
            name: selection.name().clone(),
            requested,
        }),
        _ => Ok(()),
    }
}

fn selectable(selection: &Selection, definition: &SemanticDefinition) -> Result<(), PlanError> {
    match definition.role() {
        role @ SemanticRole::Filter => Err(PlanError::NotSelectable {
            name: selection.name().clone(),
            role,
        }),
        _ => Ok(()),
    }
}

fn ordered_by_selected(request: &Request) -> Result<(), PlanError> {
    let unselected = (request.order_by.iter())
        .find(|key| !(request.select.iter()).any(|selection| selection.name() == &key.name));
    match unselected {
        Some(key) => Err(PlanError::OrderByNotSelected {
            name: key.name.clone(),
        }),
        None => Ok(()),
    }
}

fn leaf_admitted(read: &Read<'_>, bound: Option<&Bound>) -> Result<(), PlanError> {
    match (read, bound) {
        (Read::Column(column), _) => Err(PlanError::ColumnInRequestPredicate {
            column: (*column).clone(),
        }),
        (
            Read::Semantics {
                name,
                accessor: true,
                ..
            },
            _,
        ) => Err(PlanError::AccessorInRequestPredicate { name: name.clone() }),
        (
            Read::Semantics {
                name,
                stated: Some(stated),
                ..
            },
            Some((_, definition)),
        ) if *stated != definition.role() => Err(PlanError::PredicateRoleMismatch {
            name: name.clone(),
            stated: *stated,
            role: definition.role(),
        }),
        _ => Ok(()),
    }
}

fn grain_fits_type(
    selection: &Selection,
    definition: &SemanticDefinition,
) -> Result<(), PlanError> {
    match (selection.granularity(), &definition.data_type) {
        (Some(requested @ (Granularity::Minute | Granularity::Hour)), Some(DataType::Date)) => {
            Err(PlanError::GranularityFinerThanType {
                name: selection.name().clone(),
                requested,
                data_type: DataType::Date,
            })
        }
        _ => Ok(()),
    }
}
